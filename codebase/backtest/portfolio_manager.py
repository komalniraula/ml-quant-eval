import numpy as np
import pandas as pd
from scipy import sparse
from .trade import Trade

class PortfolioManager:
    def __init__(self, df_main, initial_capital, max_holding_days=5):
        self.df_main = df_main
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.max_holding_days = max_holding_days
        self.active_trades = []
        self.trade_history = []
        self.daily_pnl = {}
        self.equity_curve = {pd.Timestamp.min: initial_capital}  # Initialize with starting capital
        
        # Create lookups for efficient access
        self._create_lookups()
        
    def _create_lookups(self):
        """Create efficient lookups for prices and volumes"""
        # Create lookups directly from df_main
        lookup_data = self.df_main.set_index(['date', 'permno'])
        self.price_lookup = lookup_data['adj_prc'].to_dict()
        self.vol_lookup = lookup_data['adv20'].to_dict()
        self.volatility_lookup = lookup_data['garch_vol'].to_dict()
        
        # Single date-indexed dataframe for other lookups
        date_indexed = self.df_main.drop_duplicates('date').set_index('date')
        self.ffr_lookup = date_indexed['fed_funds_rate'].to_dict()
        self.market_return_lookup = date_indexed['vwretd'].to_dict()
    
    def _calculate_max_shares(self, permno, current_date, price, allocated_money=None):      
        """Calculate maximum number of shares based on liquidity and capital"""
        # Default allocated money if not provided
        if allocated_money is None:
            allocated_money = self.available_capital / 10  # Default to 10% of available capital
        
        # Calculate shares based on allocated money
        capital_shares = 0
        if price > 0:
            capital_shares = int(allocated_money / price)
        
        # Get 20-day average volume with proper error handling
        try:
            adv20 = self.vol_lookup.get((current_date, permno), 0)
            
            # Handle NaN, None, or zero values
            if adv20 is None or np.isnan(adv20) or adv20 <= 0:
                # If ADV20 is invalid, return capital-based shares
                return max(1, capital_shares)
            
            # Limit to 10% of average volume
            liquidity_shares = int(adv20 * 0.1)
            
            # Take minimum of capital-based shares and liquidity-based shares
            max_shares = min(capital_shares, liquidity_shares)
            
            # Ensure at least 1 share if we have capital
            return max(1, max_shares) if capital_shares > 0 else 0
            
        except (TypeError, ValueError) as e:
            # Fall back to capital-based shares
            return max(1, capital_shares)
            
    def reset_capital(self, amount):
        """Reset available capital (called at the start of each quarter)"""
        self.available_capital = amount
        
    def process_trading_day(self, current_date, signals, current_data):
        """Process a single trading day"""
        trade_updates = []
        
        # First update financing costs for all active trades
        fed_funds_rate = self.ffr_lookup.get(current_date, 0.02)  # Default to 2% if missing
        for trade in self.active_trades:
            trade.update_daily_financing(current_date, fed_funds_rate)
            
            # Optional: Update market value for active trades (for internal tracking only)
            price_black = self.price_lookup.get((current_date, trade.permno_black))
            price_white = self.price_lookup.get((current_date, trade.permno_white))
            
            if price_black is not None and price_white is not None:
                trade.update_market_value(current_date, price_black, price_white)
        
        # Then check for exits (z-score reversal or max holding period)
        closed_trades = self._process_exits(current_date, current_data)
        
        # Update available capital from closed trades
        for trade in closed_trades:
            # Return the invested capital plus profit/loss
            self.available_capital += (trade.investment_black + trade.investment_white + trade.net_pnl)
            # Add to trade history (only for closed trades)
            self.trade_history.append(trade)
            # Add to trade updates (for logging) - only adding CLOSED trades
            trade_updates.append(trade.to_dict())
        
        # Calculate daily PnL from closed trades only
        day_pnl = sum([trade.net_pnl for trade in closed_trades])
        
        # Then process new entries if we have signals and available capital
        new_trades = self._process_entries(current_date, signals, current_data)
        
        # Update equity curve for accounting purposes
        prev_equity = max(self.equity_curve.values())
        self.equity_curve[current_date] = prev_equity + day_pnl
        
        # Save daily PnL
        self.daily_pnl[current_date] = day_pnl
        
        # Return only the updates for CLOSED trades
        return trade_updates
    
    def _check_exit_conditions(self, trade, current_date, current_z_diff):
        """Check if a trade should be exited based on the specified conditions"""
        # Condition 1: Z-score mean reversion toward zero
        if trade.side == 'short_black_long_white' and current_z_diff <= 0:
            return True, 'mean_reversion'
        elif trade.side == 'long_black_short_white' and current_z_diff >= 0:
            return True, 'mean_reversion'
            
        # Condition 2: Max holding period reached
        if trade.days_held >= self.max_holding_days:
            return True, 'max_holding'
            
        return False, None
    
    def _process_exits(self, current_date, current_data):
        """Check active trades for exit conditions"""
        closed_trades = []
        remaining_trades = []
        
        for trade in self.active_trades:
            # Get stock permnos
            permno_black = trade.permno_black
            permno_white = trade.permno_white
            
            # Get z-scores efficiently
            z_col = f"z_{trade.zscore_method}_{trade.horizon}d_lb{trade.lookback}"
            
            # Check if data exists for both stocks
            black_data = current_data[current_data['permno'] == permno_black]
            white_data = current_data[current_data['permno'] == permno_white]
            
            if black_data.empty or white_data.empty or z_col not in black_data.columns or z_col not in white_data.columns:
                remaining_trades.append(trade)
                continue
                
            # Get current z-scores
            z_black = black_data[z_col].values[0]
            z_white = white_data[z_col].values[0]
            
            # Calculate current z-diff
            current_z_diff = z_black - z_white
            
            # Check exit conditions
            should_exit, exit_reason = self._check_exit_conditions(trade, current_date, current_z_diff)
            
            if should_exit:
                # Get exit prices
                exit_price_black = self.price_lookup.get((current_date, permno_black))
                exit_price_white = self.price_lookup.get((current_date, permno_white))
                
                if exit_price_black is None or exit_price_white is None:
                    # Can't exit if prices are missing, keep the trade
                    remaining_trades.append(trade)
                    continue
                
                # Close the trade
                trade.close_trade(current_date, exit_price_black, exit_price_white, 
                                 exit_reason, current_z_diff)
                
                closed_trades.append(trade)
            else:
                # Keep track of active trades
                remaining_trades.append(trade)
        
        # Update active trades list
        self.active_trades = remaining_trades
        return closed_trades
    
    def _process_entries(self, current_date, signals, current_data):
        """Process new trade entries with liquidity constraints"""
        if not signals or self.available_capital <= 0:
            return []
            
        # Calculate volatility for each pair in signals for position sizing
        pairs_volatility = {}
        total_inv_vol = 0
        
        # Track rejection reasons
        missing_volatility = 0
        
        for sig in signals:
            permno_black = sig['permno_black']
            permno_white = sig['permno_white']
            
            # Get GARCH volatilities from lookup table
            vol_black = self.volatility_lookup.get((current_date, permno_black))
            vol_white = self.volatility_lookup.get((current_date, permno_white))
            
            if vol_black is None or vol_white is None or vol_black == 0 or vol_white == 0:
                missing_volatility += 1
                continue
                
            # Use combined volatility for the pair
            pair_vol = (vol_black + vol_white) / 2
            pair_key = (permno_black, permno_white)
            pairs_volatility[pair_key] = pair_vol
            
            # Calculate inverse volatility
            inv_vol = 1 / pair_vol
            total_inv_vol += inv_vol
        
        if total_inv_vol == 0:
            return []
            
        # Allocate capital by inverse volatility
        capital_allocations = {}
        for pair_key, vol in pairs_volatility.items():
            inv_vol = 1 / vol
            allocation = (inv_vol / total_inv_vol) * self.available_capital
            capital_allocations[pair_key] = allocation
        
        # Execute trades
        executed_trades = []
        capital_used = 0
        
        # Track rejection reasons
        missing_price = 0
        zero_shares = 0
        insufficient_capital = 0
        
        for sig in signals:
            permno_black = sig['permno_black']
            permno_white = sig['permno_white']
            pair_key = (permno_black, permno_white)
            
            if pair_key not in capital_allocations:
                continue
                
            # Get stock prices
            px_b = self.price_lookup.get((current_date, permno_black))
            px_w = self.price_lookup.get((current_date, permno_white))
            
            if px_b is None or px_w is None or px_b <= 0 or px_w <= 0:
                missing_price += 1
                continue
                
            # Allocate capital to the pair
            pair_capital = capital_allocations[pair_key]
            
            # Split capital equally between black and white stocks
            inv_b = inv_w = pair_capital / 2
            
            # Calculate maximum shares based on liquidity constraint (10% of ADV20)
            max_shares_b = self._calculate_max_shares(permno_black, current_date, px_b, inv_b)
            max_shares_w = self._calculate_max_shares(permno_white, current_date, px_w, inv_w)
            
            # Calculate shares based on capital allocation
            capital_shares_b = int(inv_b / px_b)
            capital_shares_w = int(inv_w / px_w)
            
            # Apply liquidity constraint
            sh_b = min(capital_shares_b, max_shares_b) if max_shares_b > 0 else capital_shares_b
            sh_w = min(capital_shares_w, max_shares_w) if max_shares_w > 0 else capital_shares_w
            
            # Skip if not enough shares can be purchased
            if sh_b == 0 or sh_w == 0:
                zero_shares += 1
                continue
                
            # Recalculate actual investment based on constrained shares
            inv_b = sh_b * px_b
            inv_w = sh_w * px_w
            
            # Calculate transaction costs
            entry_tc = 0.01 * (sh_b + sh_w)
            
            # Check if we have enough capital
            total_cost = inv_b + inv_w + entry_tc
            if total_cost > (self.available_capital - capital_used):
                insufficient_capital += 1
                continue
                
            # Record the capital used
            capital_used += total_cost
            
            # Create new trade
            new_trade = Trade(
                entry_date=current_date,
                permno_black=permno_black,
                permno_white=permno_white,
                side=sig['signal'],
                z_diff_entry=sig['z_diff'],
                investment_black=inv_b,
                investment_white=inv_w,
                shares_black=sh_b,
                shares_white=sh_w,
                entry_price_black=px_b,
                entry_price_white=px_w,
                entry_transaction_cost=entry_tc,
                zscore_method=sig.get('zscore_method', 'ou'),
                horizon=sig.get('horizon', 5),
                lookback=sig.get('lookback', 20)
            )
            
            # Add to active trades list
            self.active_trades.append(new_trade)
            executed_trades.append(new_trade)
        
        # Update available capital
        self.available_capital -= capital_used
        
        return executed_trades

    def mark_to_market_open_positions(self, final_date):
        """Close all open positions at the end of the backtest period using latest prices"""
        closed_trades = []
        
        # Skip if no active trades
        if not self.active_trades:
            return []
        
        for trade in self.active_trades:
            # Get exit prices for the final date
            exit_price_black = self.price_lookup.get((final_date, trade.permno_black))
            exit_price_white = self.price_lookup.get((final_date, trade.permno_white))
            
            # Skip if prices are missing
            if exit_price_black is None or exit_price_white is None:
                # Try to find the last available prices
                dates = sorted(self.price_lookup.keys(), key=lambda x: x[0])
                for date, permno in reversed(dates):
                    if date < final_date and permno == trade.permno_black:
                        exit_price_black = self.price_lookup.get((date, permno))
                        break
                
                for date, permno in reversed(dates):
                    if date < final_date and permno == trade.permno_white:
                        exit_price_white = self.price_lookup.get((date, permno))
                        break
                
                # If still no prices, skip this trade
                if exit_price_black is None or exit_price_white is None:
                    continue
            
            # Close the trade with "end_of_period" as reason
            trade.close_trade(final_date, exit_price_black, exit_price_white, 
                             'end_of_period', 0)  # Use 0 as z_diff_exit
            
            # Add to closed trades and trade history
            closed_trades.append(trade)
            self.trade_history.append(trade)
        
        # Update active trades list (should be empty now)
        self.active_trades = []
        
        # Update equity curve with the PnL from these trades
        if closed_trades:
            day_pnl = sum([trade.net_pnl for trade in closed_trades])
            if final_date not in self.equity_curve:
                prev_equity = max(self.equity_curve.values())
                self.equity_curve[final_date] = prev_equity + day_pnl
            else:
                self.equity_curve[final_date] += day_pnl
        
        # Return the closed trades
        return closed_trades
        
    def get_trade_history(self):
        """Return the trade history for analysis"""
        return self.trade_history