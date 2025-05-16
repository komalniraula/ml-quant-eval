import pandas as pd
import time
import traceback

from .grid_search import run_hyperparameter_grid_search  # Uncomment this import

def run_backtest(df_main_path='final_backtest_data.csv', 
               df_pairs_path='corr_coin.csv',
               period='train'):
    """Main function to run the backtest"""
    print("Loading data...")
    
    try:
        # Load the datasets with the correct filenames
        try:
            df_merged_filtered = pd.read_csv(df_main_path)
            print(f"Successfully loaded {df_main_path}")
        except Exception as e:
            print(f"Error loading {df_main_path}: {str(e)}")
            raise
            
        try:
            df_pairs = pd.read_csv(df_pairs_path)
            print(f"Successfully loaded {df_pairs_path}")
            print(f"Columns in df_pairs: {list(df_pairs.columns)}")
        except Exception as e:
            print(f"Error loading {df_pairs_path}: {str(e)}")
            raise

        # Rename column names if needed
        if 'permno_1' in df_pairs.columns and 'permno_2' in df_pairs.columns:
            df_pairs.rename(columns={'permno_1': 'permno_black', 'permno_2': 'permno_white'}, inplace=True)
        
        # Convert date columns to datetime
        df_merged_filtered['date'] = pd.to_datetime(df_merged_filtered['date'])
        
        # Filter data based on period
        if period.lower() == 'train':
            start_date = '2015-01-01'
            end_date = '2021-12-31'
            period_name = "in-sample"
        elif period.lower() == 'test':
            start_date = '2022-01-01'
            end_date = '2024-12-31'
            period_name = "out-of-sample"
        else:
            raise ValueError(f"Invalid period: {period}. Use 'train' or 'test'.")
        
        # Filter main dataframe by date
        date_mask = (df_merged_filtered['date'] >= start_date) & (df_merged_filtered['date'] <= end_date)
        df_merged_filtered = df_merged_filtered[date_mask].copy()

        # Define quarters based on calendar date
        df_merged_filtered['quarter'] = df_merged_filtered['date'].dt.to_period('Q').astype(str)
        
        # Filter pairs by date range if formation_date exists
        if 'formation_date' in df_pairs.columns:
            df_pairs['formation_date'] = pd.to_datetime(df_pairs['formation_date'])
            date_mask = (df_pairs['formation_date'] >= start_date) & (df_pairs['formation_date'] <= end_date)
            df_pairs = df_pairs[date_mask].copy()
            print(f"Filtered pairs: {len(df_pairs)} within date range")
        
        # Print data overview
        quarters = df_merged_filtered['quarter'].unique()
        print(f"\n=== Data overview for {period_name} period ({start_date} to {end_date}) ===")
        print(f"Date range: {df_merged_filtered['date'].min()} to {df_merged_filtered['date'].max()}")
        print(f"Number of trading days: {df_merged_filtered['date'].nunique()}")
        print(f"Number of stocks: {df_merged_filtered['permno'].nunique()}")
        print(f"Number of calendar quarters: {len(quarters)}")
        print(f"Number of pairs: {len(df_pairs)}")
        
        # Define hyperparameter grid
        param_grid = {
            'COINTEGRATION_THRESHOLD': [0.05],
            'CORRELATION_THRESHOLD': [0.9], #0.5, 0.7, ],
            'ZSCORE_METHOD': ['classical'], #'ou'],
            'ZSCORE_THRESHOLD': [1],
            'LOOKBACK_PERIOD': [10],
            'HORIZON': [10],
            'MAX_HOLDING_DAYS': [10],
            'INITIAL_CAPITAL': 1_000_000_000
        }
        
        # Output file path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'backtest_results_{period}_{timestamp}.csv'
        
        # Calculate number of combinations
        num_combinations = 1
        for key, values in param_grid.items():
            if isinstance(values, list):
                num_combinations *= len(values)
        
        print(f"Starting grid search with {num_combinations} combinations...")
        
        # Run grid search
        results = run_hyperparameter_grid_search(df_merged_filtered, df_pairs, param_grid, output_file)
        
        # Print summary of best results
        if not results.empty:
            print("\nTop 5 parameter combinations by Sharpe ratio:")
            top_sharpe = results.sort_values('sharpe_ratio', ascending=False).head(5)
            print(top_sharpe[['ZSCORE_METHOD', 'ZSCORE_THRESHOLD', 'LOOKBACK_PERIOD', 'HORIZON', 'MAX_HOLDING_DAYS', 'sharpe_ratio', 'sortino_ratio', 'alpha']])
            
            # Save the best performing parameters for future use
            try:
                best_params_idx = results['sharpe_ratio'].idxmax()
                best_params = results.loc[best_params_idx].to_dict()
                with open(f'best_params_{period}_{timestamp}.txt', 'w') as f:
                    for k, v in best_params.items():
                        f.write(f"{k}: {v}\n")
                
                print(f"\nResults saved to {output_file}")
                print(f"Best parameters saved to best_params_{period}_{timestamp}.txt")
            except Exception as e:
                print(f"Error saving best parameters: {str(e)}")
        else:
            print("No valid results were generated. Check the error logs.")
        
        return results
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        return None