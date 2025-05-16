import numpy as np
import pandas as pd

def calculate_trade_based_metrics(trade_df, market_returns, ffr_lookup=None, initial_capital=1_000_000_000):
    """
    Calculate performance metrics based solely on trade log data and returns daily returns data
    for graphing in reports.
    
    Parameters:
    -----------
    trade_df : pandas DataFrame
        DataFrame containing trade information with columns: 
        entry_date, exit_date, net_pnl, etc.
    market_returns : dict or Series
        Market returns indexed by date (vwretd values)
    ffr_lookup : dict or None
        Federal Funds Rate lookup by date. If None, will use 0.02 as default.
    initial_capital : float
        Initial capital for calculating returns
        
    Returns:
    --------
    dict : Dictionary of performance metrics and daily returns data
    """
    # Ensure we have trades to analyze
    if len(trade_df) == 0:
        empty_returns = pd.DataFrame(columns=['date', 'return'])
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'alpha': 0,
            'beta': 0,
            'max_drawdown': 0,
            'hit_rate': 0,
            'num_trades': 0,
            'avg_trade_pnl': 0,
            'avg_holding_period': 0,
            'num_trading_days': 0,
            'daily_returns': empty_returns
        }
    
    # Convert dates to datetime if they aren't already
    if not pd.api.types.is_datetime64_dtype(trade_df['exit_date']):
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
    if not pd.api.types.is_datetime64_dtype(trade_df['entry_date']):
        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
    
    # Sort trades by exit date
    trade_df = trade_df.sort_values('exit_date')
    
    # Calculate basic trade metrics
    num_trades = len(trade_df)
    hit_rate = (trade_df['net_pnl'] > 0).mean()
    avg_trade_pnl = trade_df['net_pnl'].mean()
    avg_holding = trade_df['days_held'].mean() if 'days_held' in trade_df.columns else 0
    
    # Get unique trading days (both entry and exit dates)
    trading_days = sorted(set(trade_df['entry_date'].tolist() + 
                             trade_df['exit_date'].dropna().tolist()))
    num_trading_days = len(trading_days)
    
    # Calculate equity curve for performance metrics
    equity_curve = {}
    current_equity = initial_capital
    
    # Group trades by exit date and calculate daily PnL
    for date, group in trade_df.groupby('exit_date'):
        day_pnl = group['net_pnl'].sum()
        current_equity += day_pnl
        equity_curve[date] = current_equity
    
    # Convert to Series for easier manipulation
    equity_series = pd.Series(equity_curve)
    equity_series = equity_series.sort_index()
    
    # Handle case with insufficient data points
    if len(equity_series) <= 1:
        empty_returns = pd.DataFrame(columns=['date', 'return'])
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'alpha': 0,
            'beta': 0,
            'max_drawdown': 0,
            'hit_rate': hit_rate,
            'num_trades': num_trades,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_holding_period': avg_holding,
            'num_trading_days': num_trading_days,
            'daily_returns': empty_returns
        }
    
    # Calculate daily returns
    daily_returns = equity_series.pct_change().fillna(0)
    
    # Create a DataFrame of dates and returns for graphing
    returns_df = pd.DataFrame({
        'date': daily_returns.index,
        'return': daily_returns.values
    })
    
    # Get corresponding market returns
    if isinstance(market_returns, dict):
        market_returns_series = pd.Series({date: market_returns.get(date, 0) 
                                        for date in daily_returns.index})
    else:
        # Assume it's already a Series
        market_returns_series = market_returns.loc[daily_returns.index]
    
    # Calculate average risk-free rate from Fed Funds Rate data if available
    if ffr_lookup is not None:
        # Extract Fed Funds Rates for trading days
        trading_day_rates = [ffr_lookup.get(date, 0) for date in trading_days if date in ffr_lookup]
        
        # Calculate average Fed Funds Rate during trading period
        if trading_day_rates:
            avg_ffr = sum(trading_day_rates) / len(trading_day_rates)
            # Convert annual rate to daily rate based on trading days
            daily_rfr = avg_ffr / num_trading_days
        else:
            # Default to 2% if no rates found
            daily_rfr = 0.02 / num_trading_days
    else:
        # Default to 2% if no FFR lookup provided
        daily_rfr = 0.02 / num_trading_days
    
    # Calculate metrics based on daily returns
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    # Calculate Sharpe ratio (based on daily returns)
    sharpe_ratio = 0
    if std_daily_return > 0:
        sharpe_ratio = mean_daily_return / std_daily_return * np.sqrt(len(daily_returns))
    
    # Calculate Sortino ratio (based on daily returns)
    downside_daily_returns = daily_returns[daily_returns < 0]
    downside_std_daily = downside_daily_returns.std() if len(downside_daily_returns) > 0 else 0
    sortino_ratio = 0
    if downside_std_daily > 0:
        sortino_ratio = mean_daily_return / downside_std_daily * np.sqrt(len(daily_returns))
    
    # Calculate CAPM metrics (Beta, Alpha)
    beta = 0
    alpha = 0
    if len(daily_returns) > 1 and len(market_returns_series) > 1:
        # Calculate beta
        cov = np.cov(daily_returns, market_returns_series)[0, 1]
        var = np.var(market_returns_series)
        if var > 0:
            beta = cov / var
            
            # Calculate alpha (based on actual trading days)
            expected_return = daily_rfr + beta * (market_returns_series.mean() - daily_rfr)
            alpha = (daily_returns.mean() - expected_return) * len(daily_returns)
    
    # Calculate drawdowns
    peak = equity_series.expanding().max()
    drawdowns = (equity_series - peak) / peak
    max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
    
    # Return comprehensive metrics and daily returns data
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'num_trades': num_trades,
        'avg_trade_pnl': avg_trade_pnl,
        'avg_holding_period': avg_holding,
        'num_trading_days': num_trading_days,
        'daily_returns': returns_df,
        'avg_fed_funds_rate': avg_ffr if ffr_lookup is not None else 0.02  # Include average FFR in the results
    }