import uuid
import numpy as np

class Trade:
    """
    Class to represent a pair trade with all relevant information.
    
    Tracks entry and exit information, calculates PnL, and handles financing costs.
    """
    # Valid trading sides
    VALID_SIDES = ['short_black_long_white', 'long_black_short_white']
    
    # Financing cost parameters
    DEFAULT_SHORT_SPREAD = 0.01  # 100 bps over Fed Funds
    DEFAULT_LONG_SPREAD = 0.015  # 150 bps over Fed Funds
    DAYS_PER_YEAR = 365
    
    def __init__(self, entry_date, permno_black, permno_white, side, z_diff_entry,
                 investment_black, investment_white, shares_black, shares_white,
                 entry_price_black, entry_price_white, entry_transaction_cost,
                 zscore_method, horizon, lookback, 
                 short_spread=DEFAULT_SHORT_SPREAD, 
                 long_spread=DEFAULT_LONG_SPREAD):
        """Initialize a new trade."""
        # Validate inputs
        self._validate_inputs(entry_date, side, investment_black, investment_white, 
                              shares_black, shares_white, entry_price_black, 
                              entry_price_white, entry_transaction_cost)
        
        # Generate unique trade ID
        self.trade_id = str(uuid.uuid4())
        
        # Entry information
        self.entry_date = entry_date
        self.permno_black = permno_black
        self.permno_white = permno_white
        self.side = side
        self.z_diff_entry = z_diff_entry
        self.investment_black = investment_black
        self.investment_white = investment_white
        self.shares_black = shares_black
        self.shares_white = shares_white
        self.entry_price_black = entry_price_black
        self.entry_price_white = entry_price_white
        self.entry_transaction_cost = entry_transaction_cost
        
        # Parameters used for this trade
        self.zscore_method = zscore_method
        self.horizon = horizon
        self.lookback = lookback
        
        # Financing parameters
        self.short_spread = short_spread
        self.long_spread = long_spread
        
        # Status tracking
        self.status = 'open'
        self.exit_date = None
        self.days_held = 0
        
        # Exit information
        self.exit_price_black = None
        self.exit_price_white = None
        self.exit_transaction_cost = None
        self.financing_cost = 0
        self.z_diff_exit = None
        self.exit_reason = None
        self.gross_pnl = None
        self.net_pnl = None
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = self.investment_black + self.investment_white
        self.current_value = self.peak_value
        
        # Track daily values for analysis
        self.daily_values = {entry_date: self.peak_value}
        self.daily_pnl = {}
    
    def _validate_inputs(self, entry_date, side, investment_black, investment_white, 
                     shares_black, shares_white, price_black, price_white, 
                     transaction_cost):
        """Validate inputs to ensure they make sense."""
        # Check side is valid
        if side not in self.VALID_SIDES:
            raise ValueError(f"side must be one of {self.VALID_SIDES}, got {side}")
        
        # Check investments are positive
        if investment_black <= 0 or investment_white <= 0:
            raise ValueError("Investment amounts must be positive")
        
        # Check shares are positive integers
        if not isinstance(shares_black, int) or shares_black <= 0:
            raise ValueError(f"shares_black must be a positive integer, got {shares_black}")
        if not isinstance(shares_white, int) or shares_white <= 0:
            raise ValueError(f"shares_white must be a positive integer, got {shares_white}")
        
        # Check prices are positive
        if price_black <= 0 or price_white <= 0:
            raise ValueError("Prices must be positive")
        
        # Check transaction cost is non-negative
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
    
    def update_daily_financing(self, current_date, fed_funds_rate):
        """Update daily financing costs for open positions."""
        # Calculate daily financing cost based on position side
        if self.side == 'short_black_long_white':
            # Short black (credit at short rate), Long white (debit at long rate)
            black_daily_cost = -self.investment_black * (fed_funds_rate + self.short_spread) / self.DAYS_PER_YEAR
            white_daily_cost = self.investment_white * (fed_funds_rate + self.long_spread) / self.DAYS_PER_YEAR
        else:
            # Long black (debit at long rate), Short white (credit at short rate)
            black_daily_cost = self.investment_black * (fed_funds_rate + self.long_spread) / self.DAYS_PER_YEAR
            white_daily_cost = -self.investment_white * (fed_funds_rate + self.short_spread) / self.DAYS_PER_YEAR
        
        daily_financing = black_daily_cost + white_daily_cost
        self.financing_cost += daily_financing
        
        # Increment days held
        self.days_held += 1
        
        return daily_financing
    
    def update_market_value(self, current_date, current_price_black, current_price_white):
        """Update the market value of the position and track drawdowns."""
        # Calculate current value of both positions
        if self.side == 'short_black_long_white':
            # Short black, long white
            black_value = self.investment_black - (self.shares_black * (current_price_black - self.entry_price_black))
            white_value = self.investment_white + (self.shares_white * (current_price_white - self.entry_price_white))
        else:
            # Long black, short white
            black_value = self.investment_black + (self.shares_black * (current_price_black - self.entry_price_black))
            white_value = self.investment_white - (self.shares_white * (current_price_white - self.entry_price_white))
        
        # Calculate current total value
        current_value = black_value + white_value
        
        # Calculate unrealized PnL
        unrealized_pnl = current_value - (self.investment_black + self.investment_white)
        
        # Update peak value if current value is higher
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Update max drawdown if current drawdown is larger
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Store daily values
        self.daily_values[current_date] = current_value
        self.daily_pnl[current_date] = unrealized_pnl
        
        # Update current value
        self.current_value = current_value
        
        return current_value
    
    def close_trade(self, exit_date, exit_price_black, exit_price_white, exit_reason, z_diff_exit):
        """Close the trade and calculate PnL."""
        self.exit_date = exit_date
        self.exit_price_black = exit_price_black
        self.exit_price_white = exit_price_white
        self.exit_reason = exit_reason
        self.z_diff_exit = z_diff_exit
        
        # Calculate transaction costs for exit (0.01 per share)
        self.exit_transaction_cost = 0.01 * (self.shares_black + self.shares_white)
        
        # Calculate PnL for each leg
        if self.side == 'short_black_long_white':
            # Short black: profit when price falls
            black_pnl = -self.shares_black * (exit_price_black - self.entry_price_black)
            # Long white: profit when price rises
            white_pnl = self.shares_white * (exit_price_white - self.entry_price_white)
        else:  # long_black_short_white
            # Long black: profit when price rises
            black_pnl = self.shares_black * (exit_price_black - self.entry_price_black)
            # Short white: profit when price falls
            white_pnl = -self.shares_white * (exit_price_white - self.entry_price_white)
        
        # Calculate gross and net PnL
        self.gross_pnl = black_pnl + white_pnl
        
        # Total costs include entry and exit transaction costs plus financing
        total_costs = self.entry_transaction_cost + self.exit_transaction_cost + self.financing_cost
        
        # Calculate net PnL
        self.net_pnl = self.gross_pnl - total_costs
        
        # Update status
        self.status = 'closed'
        
        # Final update to daily values
        self.daily_values[exit_date] = self.investment_black + self.investment_white + self.net_pnl
        
        # Calculate ROI
        self.roi = self.net_pnl / (self.investment_black + self.investment_white)
        
        return self.net_pnl
    
    def to_dict(self):
        """Convert trade object to dictionary for logging and analysis."""
        return {
            'trade_id': self.trade_id,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'permno_black': self.permno_black,
            'permno_white': self.permno_white,
            'side': self.side,
            'z_diff_entry': self.z_diff_entry,
            'z_diff_exit': self.z_diff_exit,
            'investment_black': self.investment_black,
            'investment_white': self.investment_white,
            'shares_black': self.shares_black,
            'shares_white': self.shares_white,
            'entry_price_black': self.entry_price_black,
            'entry_price_white': self.entry_price_white,
            'exit_price_black': self.exit_price_black,
            'exit_price_white': self.exit_price_white,
            'entry_transaction_cost': self.entry_transaction_cost,
            'exit_transaction_cost': self.exit_transaction_cost,
            'financing_cost': self.financing_cost,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'roi': getattr(self, 'roi', None),
            'exit_reason': self.exit_reason,
            'status': self.status,
            'days_held': self.days_held,
            'max_drawdown': self.max_drawdown,
            'zscore_method': self.zscore_method,
            'horizon': self.horizon,
            'lookback': self.lookback
        }