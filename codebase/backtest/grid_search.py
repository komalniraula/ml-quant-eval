import os
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from itertools import product

from .backtest_engine import BacktestEngine

def run_hyperparameter_grid_search(df_main, df_pairs, param_grid, output_file='backtest_results.csv'):
    """Run backtest with different hyperparameter combinations"""
    results = []
    
    # Generate parameter combinations more efficiently
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = []
    
    # Get all parameter combinations except INITIAL_CAPITAL
    non_capital_keys = [k for k in keys if k != 'INITIAL_CAPITAL']
    non_capital_values = [param_grid[k] for k in non_capital_keys]
    
    # Generate combinations with product
    for combination in product(*non_capital_values):
        params = dict(zip(non_capital_keys, combination))
        params['INITIAL_CAPITAL'] = param_grid['INITIAL_CAPITAL']
        param_combinations.append(params)
    
    print(f"Running {len(param_combinations)} parameter combinations")
    
    # Use a checkpointing mechanism
    checkpoint_file = f"checkpoint_{os.path.basename(output_file)}.pkl"
    completed_runs = set()
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results = checkpoint_data.get('results', [])
                completed_runs = set(checkpoint_data.get('completed', []))
                print(f"Loaded {len(results)} previous results from checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}. Starting fresh.")
        results = []
        completed_runs = set()
    
    # Run backtest for each combination
    for i, params in enumerate(param_combinations):
        # Skip already completed runs
        params_str = str(params)
        if params_str in completed_runs:
            print(f"Skipping combination {i+1}/{len(param_combinations)}: already completed")
            continue
            
        print(f"Running combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Create a different random seed for each run for reproducibility
            seed = hash(params_str) % 10000
            np.random.seed(seed)
            
            backtest = BacktestEngine(df_main, df_pairs, params)
            result = backtest.run_backtest()
            
            # Extract performance metrics
            performance = result['performance']

            # Save trade log to file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not result['trade_log'].empty:
                trade_log_file = f"trade_log_{params['ZSCORE_METHOD']}_{params['ZSCORE_THRESHOLD']}_{params['LOOKBACK_PERIOD']}_{params['HORIZON']}_{params['MAX_HOLDING_DAYS']}_{timestamp}.csv"
                result['trade_log'].to_csv(trade_log_file, index=False)
                print(f"Saved {len(result['trade_log'])} trades to {trade_log_file}")
            else:
                print("No trades to save!")
            
            # Combine parameters and performance metrics for output
            result_row = {
                'CORRELATION_THRESHOLD': params['CORRELATION_THRESHOLD'],
                'COINTEGRATION_THRESHOLD': params['COINTEGRATION_THRESHOLD'],
                'ZSCORE_METHOD': params['ZSCORE_METHOD'],
                'ZSCORE_THRESHOLD': params['ZSCORE_THRESHOLD'],
                'LOOKBACK_PERIOD': params['LOOKBACK_PERIOD'],
                'HORIZON': params['HORIZON'],
                'MAX_HOLDING_DAYS': params['MAX_HOLDING_DAYS'],
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'sortino_ratio': performance.get('sortino_ratio', 0),
                'alpha': performance.get('alpha', 0),
                'beta': performance.get('beta', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'hit_rate': performance.get('hit_rate', 0),
                'num_trades': performance.get('num_trades', 0),
                'avg_trade_pnl': performance.get('avg_trade_pnl', 0),
                'avg_holding_period': performance.get('avg_holding_period', 0),
                'num_trading_days': performance.get('num_trading_days', 0)
            }
            
            results.append(result_row)
            completed_runs.add(params_str)
            
            # Save checkpoint after each successful run
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'results': results, 'completed': list(completed_runs)}, f)
                
                # Save to CSV as well
                pd.DataFrame(results).to_csv(output_file, index=False)
            except Exception as save_err:
                print(f"Error saving checkpoint: {str(save_err)}")
            
        except Exception as e:
            print(f"Error running combination {i+1}: {params}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()
            
            # Add a row with error information
            error_row = {
                'CORRELATION_THRESHOLD': params['CORRELATION_THRESHOLD'],
                'COINTEGRATION_THRESHOLD': params['COINTEGRATION_THRESHOLD'],
                'ZSCORE_METHOD': params['ZSCORE_METHOD'],
                'ZSCORE_THRESHOLD': params['ZSCORE_THRESHOLD'],
                'LOOKBACK_PERIOD': params['LOOKBACK_PERIOD'],
                'HORIZON': params['HORIZON'],
                'MAX_HOLDING_DAYS': params['MAX_HOLDING_DAYS'],
                'error': str(e),
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'alpha': 0,
                'beta': 0,
                'max_drawdown': 0,
                'hit_rate': 0,
                'num_trades': 0,
                'avg_trade_pnl': 0,
                'avg_holding_period': 0,
                'num_trading_days': 0
            }
            results.append(error_row)
            
            # Save checkpoint and CSV after error
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'results': results, 'completed': list(completed_runs)}, f)
                pd.DataFrame(results).to_csv(output_file, index=False)
            except Exception as save_err:
                print(f"Error saving checkpoint after error: {str(save_err)}")
    
    # Final save and return
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        return results_df
    except Exception as final_err:
        print(f"Error saving final results: {str(final_err)}")
        return pd.DataFrame(results)