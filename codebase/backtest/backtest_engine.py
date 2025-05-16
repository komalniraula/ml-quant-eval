import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .signal_generator import SignalGenerator
from .portfolio_manager import PortfolioManager
from .performance import calculate_trade_based_metrics

def _process_quarter_parallel(quarter, df_main, filtered_pairs, signal_generator, initial_capital, max_holding_days):
    """Process a single quarter in parallel"""
    # Filter data for this quarter
    quarter_data = df_main[df_main['quarter'] == quarter]
    if quarter_data.empty:
        print(f"Warning: No data found for quarter {quarter}")
        return {'quarter': quarter, 'trade_log': [], 'performance': {}}
    
    print(f"Processing calendar quarter {quarter}")
    
    # Extract quarter start and end dates for better reporting
    quarter_dates = sorted(quarter_data['date'].unique())
    quarter_start = quarter_dates[0]
    quarter_end = quarter_dates[-1]
    print(f"Quarter period: {quarter_start.strftime('%Y-%m-%d')} to {quarter_end.strftime('%Y-%m-%d')}")
    
    # Create portfolio manager for this quarter
    portfolio_manager = PortfolioManager(
        quarter_data, 
        initial_capital,
        max_holding_days=max_holding_days
    )
    
    # Reset capital
    portfolio_manager.reset_capital(initial_capital)
    
    # Group data by date for faster access
    date_grouped_data = {date: group for date, group in quarter_data.groupby('date')}
    
    # Process each trading day
    trade_log = []
    signals_count = 0
    trades_count = 0
    
    # Process each trading day in the quarter
    for current_date in quarter_dates:
        # Get signals for the current date
        try:
            signals = signal_generator.generate_signals(current_date)
            signals_count += len(signals)
        except Exception as e:
            print(f"Error generating signals for date {current_date}: {str(e)}")
            signals = []
        
        # Process the trading day
        try:
            day_results = portfolio_manager.process_trading_day(
                current_date, 
                signals,
                date_grouped_data[current_date]
            )
            
            if day_results:
                trade_log.extend(day_results)
                trades_count += len(day_results)
                
        except Exception as e:
            print(f"Error processing trading day {current_date}: {str(e)}")
    
    print(f"Quarter {quarter} summary: {signals_count} signals generated, {trades_count} trades executed")
    
    # Return the trade log for this quarter
    return {
        'quarter': quarter,
        'trade_log': trade_log,
        'performance': {}  # We'll calculate this later from the trade log
    }
    
class BacktestEngine:
    def __init__(self, df_main, df_pairs, hyperparams):
        self.hyperparams = hyperparams
        self.initial_capital = hyperparams['INITIAL_CAPITAL']
        
        # Select specific columns directly instead of filtering
        zscore_method = hyperparams['ZSCORE_METHOD']
        lookback_period = hyperparams['LOOKBACK_PERIOD']
        horizon = hyperparams['HORIZON']
        
        z_col = f'z_{zscore_method}_{horizon}d_lb{lookback_period}'
        base_cols = ['date', 'permno', 'trading_start', 'group_id', 'adj_prc', 'fed_funds_rate', 'adv20', 'vwretd', 'garch_vol']
        future_return_col = f'future_cumret_{horizon}d'
        
        # Select only required columns
        needed_cols = base_cols + [z_col]
        if future_return_col in df_main.columns:
            needed_cols.append(future_return_col)
            
        needed_cols = [col for col in needed_cols if col in df_main.columns]
        
        # Clean data
        self.df_main = df_main[needed_cols].copy()
        self.df_main = self.df_main.replace([np.inf, -np.inf], np.nan)
        self.df_main = self.df_main.dropna()
        
        # Keep a copy of the pairs data
        self.df_pairs = df_pairs
        
        # Pre-process data for faster lookups
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess data for efficient backtest execution"""
        # Clean data by replacing infinites with NaN and dropping NaN values
        self.df_main = self.df_main.replace([np.inf, -np.inf], np.nan)
        self.df_main = self.df_main.dropna()
        
        # Make sure date is datetime
        if not pd.api.types.is_datetime64_dtype(self.df_main['date']):
            self.df_main['date'] = pd.to_datetime(self.df_main['date'])
        
        # Create a quarter column based on date
        self.df_main['quarter'] = self.df_main['date'].dt.to_period('Q').astype(str)
        
        # Get unique quarters for processing
        self.quarters = sorted(self.df_main['quarter'].unique())
        
        print(f"Identified {len(self.quarters)} calendar quarters for processing")
        
        # Filter pairs based on correlation and cointegration thresholds
        corr_threshold = self.hyperparams['CORRELATION_THRESHOLD']
        coint_threshold = self.hyperparams['COINTEGRATION_THRESHOLD']
        
        # Apply filters if thresholds are provided AND columns exist
        filter_condition = True  # Default to include all pairs
        
        # Check correlation columns
        if corr_threshold is not None:
            if 'corr' in self.df_pairs.columns:
                filter_condition = filter_condition & (self.df_pairs['corr'] >= corr_threshold)
            elif 'correlation' in self.df_pairs.columns:
                filter_condition = filter_condition & (self.df_pairs['correlation'] >= corr_threshold)
            else:
                print("Warning: No correlation column found in pairs data, skipping correlation filter")
        
        # Check cointegration columns
        if coint_threshold is not None:
            if 'coint_pval' in self.df_pairs.columns:
                filter_condition = filter_condition & (self.df_pairs['coint_pval'] <= coint_threshold)
            elif 'pval' in self.df_pairs.columns:
                filter_condition = filter_condition & (self.df_pairs['pval'] <= coint_threshold)
            elif 'p_value' in self.df_pairs.columns:
                filter_condition = filter_condition & (self.df_pairs['p_value'] <= coint_threshold)
            else:
                print("Warning: No cointegration p-value column found in pairs data, skipping cointegration filter")
        
        # Apply the filter
        self.filtered_pairs = self.df_pairs[filter_condition].copy()
        
        # Log preprocessing results
        print(f"Preprocessing complete: {len(self.filtered_pairs)} pairs after filtering")
    
    def run_backtest(self):
        """Run the full backtest using the specified hyperparameters"""
        print("Running pre-backtest diagnostics...")
        self.run_diagnostics()
        
        # Initialize signal generator with the selected parameters
        zscore_method = self.hyperparams['ZSCORE_METHOD']
        zscore_threshold = self.hyperparams['ZSCORE_THRESHOLD']
        lookback_period = self.hyperparams['LOOKBACK_PERIOD']
        horizon = self.hyperparams['HORIZON']
        max_holding_days = self.hyperparams['MAX_HOLDING_DAYS']
        
        # Create optimized dataset with only needed columns
        z_col = f'z_{zscore_method}_{horizon}d_lb{lookback_period}'
        needed_cols = ['date', 'permno', 'quarter', 'group_id', 'adj_prc', 'fed_funds_rate', 
                      'adv20', 'vwretd', 'garch_vol', z_col]
        
        # Check if all needed columns exist
        needed_cols = [col for col in needed_cols if col in self.df_main.columns]
        
        # Only keep needed columns in memory
        self.optimized_df = self.df_main[needed_cols].copy()
        
        # Pre-sort data for faster operations
        self.optimized_df.sort_values(['date', 'permno'], inplace=True)
        
        # Create signal generator with optimized dataset
        signal_generator = SignalGenerator(
            self.optimized_df, 
            self.filtered_pairs,
            zscore_method=zscore_method,
            zscore_threshold=zscore_threshold,
            horizon=horizon,
            lookback_period=lookback_period
        )
        
        # Use fixed n_jobs=4 for parallel processing
        n_jobs = 4
        
        # Precompute signals with progress bar
        signal_generator.precompute_signals_parallel(horizon=horizon, n_jobs=n_jobs)
        
        # Check if we have any quarters to process
        if len(self.quarters) == 0:
            print("No quarters found to process! Check data filtering.")
            # Return empty results
            return {
                'trade_log': pd.DataFrame(),
                'performance': {},
                'hyperparams': self.hyperparams,
                'quarterly_results': {}
            }
        
        # Process quarters in batches to reduce memory pressure
        batch_size = 4  # Adjust based on your system's memory
        all_closed_trades = []
        quarterly_results = {}
        
        for i in range(0, len(self.quarters), batch_size):
            batch_quarters = self.quarters[i:i+batch_size]
            
            # Process quarters in parallel
            n_jobs = 4
            batch_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_quarter_parallel)(
                    quarter,
                    self.optimized_df,
                    self.filtered_pairs,
                    signal_generator,
                    self.initial_capital,
                    max_holding_days
                )
                for quarter in tqdm(batch_quarters, desc=f"Processing Quarters Batch {i//batch_size+1}")
            )
            
            # Collect results
            for result in batch_results:
                all_closed_trades.extend(result['trade_log'])
                quarterly_results[result['quarter']] = result['performance']
            
            # Force garbage collection after each batch
            gc.collect()
        
        print(f"\nCollected {len(all_closed_trades)} closed trades across all quarters")
        
        # Calculate performance metrics from trade log
        performance_metrics = {}
        
        if len(all_closed_trades) > 0:
            # Convert trade_log to DataFrame for analysis
            trade_df = pd.DataFrame(all_closed_trades) if all_closed_trades else pd.DataFrame()
            
            # Calculate metrics directly from trades
            if len(trade_df) > 0:
                # Create lookups for market returns and Fed Funds Rate from the original data
                date_indexed = self.df_main.drop_duplicates('date').set_index('date')
                market_return_lookup = date_indexed['vwretd'].to_dict()
                ffr_lookup = date_indexed['fed_funds_rate'].to_dict()
                
                # Calculate comprehensive metrics directly from trade log
                performance_metrics = calculate_trade_based_metrics(
                    trade_df=trade_df,
                    market_returns=market_return_lookup,
                    ffr_lookup=ffr_lookup,
                    initial_capital=self.initial_capital
                )
                
                # Save daily returns data to file for graphing
                if 'daily_returns' in performance_metrics:
                    daily_returns_df = performance_metrics['daily_returns']
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    daily_returns_file = f'daily_returns_{timestamp}.csv'
                    daily_returns_df.to_csv(daily_returns_file, index=False)
                    print(f"Daily returns data saved to {daily_returns_file}")
                
                # Print metrics
                print(f"\nCalculated metrics from trade data:")
                print(f"Number of trades: {performance_metrics['num_trades']}")
                print(f"Number of unique trading days: {performance_metrics['num_trading_days']}")
                print(f"Average Fed Funds Rate: {performance_metrics['avg_fed_funds_rate']*100:.2f}%")
                print(f"Hit rate: {performance_metrics['hit_rate']*100:.2f}%")
                print(f"Average trade PnL: ${performance_metrics['avg_trade_pnl']:,.2f}")
                print(f"Average holding period: {performance_metrics['avg_holding_period']:.2f} days")
                print(f"Sharpe ratio (based on trading days): {performance_metrics['sharpe_ratio']:.4f}")
                print(f"Sortino ratio (based on trading days): {performance_metrics['sortino_ratio']:.4f}")
                print(f"CAPM Alpha: {performance_metrics['alpha']:.6f}")
                print(f"CAPM Beta: {performance_metrics['beta']:.4f}")
                print(f"Max drawdown: {performance_metrics['max_drawdown']*100:.2f}%")
        else:
            print("No closed trades found, using empty metrics")
            
        # Return combined results
        results = {
            'trade_log': pd.DataFrame(all_closed_trades) if all_closed_trades else pd.DataFrame(),
            'performance': performance_metrics,
            'hyperparams': self.hyperparams,
            'quarterly_results': quarterly_results
        }
        
        return results
        
    def run_diagnostics(self):
        """Run diagnostic checks to identify potential issues"""
        print("\n=== DIAGNOSTIC REPORT ===\n")
        
        # 1. Check for pairs after filtering
        if hasattr(self, "filtered_pairs"):
            print(f"Filtered pairs: {len(self.filtered_pairs)} of {len(self.df_pairs)} original pairs")
            if len(self.filtered_pairs) == 0:
                print("CRITICAL ERROR: No pairs remain after correlation/cointegration filtering!")
                
                # Check correlation threshold
                corr_threshold = self.hyperparams.get('CORRELATION_THRESHOLD')
                if corr_threshold is not None:
                    for corr_col in ['corr', 'correlation']:
                        if corr_col in self.df_pairs.columns:
                            corr_values = self.df_pairs[corr_col].dropna()
                            print(f"{corr_col} stats: min={corr_values.min():.4f}, max={corr_values.max():.4f}, mean={corr_values.mean():.4f}")
                            above_threshold = (corr_values >= corr_threshold).sum()
                            print(f"Values >= {corr_threshold}: {above_threshold} ({above_threshold/len(corr_values)*100:.2f}%)")
                            break
                
                # Check cointegration threshold
                coint_threshold = self.hyperparams.get('COINTEGRATION_THRESHOLD')
                if coint_threshold is not None:
                    for coint_col in ['coint_pval', 'pval', 'p_value']:
                        if coint_col in self.df_pairs.columns:
                            coint_values = self.df_pairs[coint_col].dropna()
                            print(f"{coint_col} stats: min={coint_values.min():.4f}, max={coint_values.max():.4f}, mean={coint_values.mean():.4f}")
                            below_threshold = (coint_values <= coint_threshold).sum()
                            print(f"Values <= {coint_threshold}: {below_threshold} ({below_threshold/len(coint_values)*100:.2f}%)")
                            break
        
        # 2. Check for z-score columns
        zscore_method = self.hyperparams['ZSCORE_METHOD']
        lookback_period = self.hyperparams['LOOKBACK_PERIOD']
        horizon = self.hyperparams['HORIZON']
        
        z_col = f'z_{zscore_method}_{horizon}d_lb{lookback_period}'
        print(f"\nChecking for z-score column: {z_col}")
        
        if z_col in self.df_main.columns:
            z_values = self.df_main[z_col].dropna()
            z_threshold = self.hyperparams['ZSCORE_THRESHOLD']
            
            print(f"Z-score column stats:")
            print(f"  - Non-null values: {len(z_values)} out of {len(self.df_main)} ({len(z_values)/len(self.df_main)*100:.2f}%)")
            print(f"  - Range: {z_values.min():.4f} to {z_values.max():.4f}")
            print(f"  - Values exceeding threshold {z_threshold}: {(abs(z_values) >= z_threshold).sum()} ({(abs(z_values) >= z_threshold).sum()/len(z_values)*100:.2f}%)")
        else:
            print(f"CRITICAL ERROR: Z-score column '{z_col}' not found in data!")
            z_cols = [col for col in self.df_main.columns if col.startswith('z_')]
            if z_cols:
                print(f"Available z-score columns: {z_cols}")
            else:
                print("No z-score columns found in data!")
        
        # 3. Check for essential columns
        required_cols = ['date', 'permno', 'group_id', 'adj_prc', 'fed_funds_rate', 'adv20', 'vwretd', 'garch_vol']
        missing_cols = [col for col in required_cols if col not in self.df_main.columns]
        
        if missing_cols:
            print(f"\nMISSING REQUIRED COLUMNS: {missing_cols}")
        else:
            print("\nAll required base columns are present")
        
        # 4. Check for NaN values in essential columns
        print("\nNaN check for essential columns:")
        for col in required_cols:
            if col in self.df_main.columns:
                null_count = self.df_main[col].isna().sum()
                null_pct = null_count / len(self.df_main) * 100
                print(f"  - {col}: {null_count} NaN values ({null_pct:.2f}%)")
        
        print("\n=== END OF DIAGNOSTIC REPORT ===\n")