import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class SignalGenerator:
    def __init__(self, df_main, df_pairs, zscore_method='ou', zscore_threshold=1.5, horizon=5, lookback_period=20):
        self.df_main = df_main
        self.df_pairs = df_pairs
        self.zscore_method = zscore_method
        self.zscore_threshold = zscore_threshold
        self.lookback_period = lookback_period
        self.precomputed_signals = None
        self.horizon = horizon
        
        # Add diagnostic print
        print(f"SignalGenerator initialized with {len(df_pairs)} pairs")
        print(f"Z-score method: {zscore_method}, threshold: {zscore_threshold}, lookback: {lookback_period}")
        
        # Check sample z-score values
        z_col = f'z_{zscore_method}_{self.horizon}d_lb{lookback_period}'
        if z_col in df_main.columns:
            z_values = df_main[z_col].dropna()
            print(f"Z-score column '{z_col}' stats:")
            print(f"  - Non-null values: {len(z_values)} out of {len(df_main)} ({len(z_values)/len(df_main)*100:.2f}%)")
            print(f"  - Range: {z_values.min():.4f} to {z_values.max():.4f}")
            print(f"  - Values exceeding threshold {zscore_threshold}: {(abs(z_values) >= zscore_threshold).sum()} ({(abs(z_values) >= zscore_threshold).sum()/len(z_values)*100:.2f}%)")
        else:
            print(f"WARNING: Z-score column '{z_col}' not found in data!")

    def precompute_signals_parallel(self, horizon=5, n_jobs=4):
        """Precompute signals for all dates and pairs in parallel"""
        z_col = f'z_{self.zscore_method}_{horizon}d_lb{self.lookback_period}'
        print(f"Precomputing signals for z-score column: {z_col}")
        
        # Verify z-score column exists
        if z_col not in self.df_main.columns:
            print(f"ERROR: Z-score column '{z_col}' not found in data columns!")
            print(f"Available columns: {self.df_main.columns}")
            return
        
        # Group by group_id for efficient processing
        group_ids = self.df_pairs['group_id'].unique()
        print(f"Processing {len(group_ids)} unique group_ids")
        
        chunk_size = max(1, len(group_ids) // n_jobs)
        chunked_groups = [group_ids[i:i + chunk_size] for i in range(0, len(group_ids), chunk_size)]
        
        # Precompute group dictionaries
        print("Building group dictionaries...")
        group_df_main_dict = {}
        for group_id in group_ids:
            group_data = self.df_main[self.df_main['group_id'] == group_id]
            if 'permno' in group_data.columns and z_col in group_data.columns and 'date' in group_data.columns:
                filtered_data = group_data[['permno', z_col, 'date']].dropna()
                group_df_main_dict[group_id] = filtered_data
                if len(filtered_data) < 10 and len(filtered_data) > 0:
                    print(f"  Group {group_id}: Only {len(filtered_data)} records with valid z-scores")
            else:
                print(f"  WARNING: Missing required columns for group {group_id}")
        
        print(f"Created dictionaries for {len(group_df_main_dict)} groups")
        
        # Process chunks in parallel
        all_results = []
        signal_counts = []
        
        for chunk_idx, chunk in enumerate(chunked_groups):
            # Process each group in parallel within the chunk
            parallel_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._process_group_signal)(
                    group_id,
                    group_df_main_dict,
                    self.df_pairs[self.df_pairs['group_id'] == group_id],
                    z_col,
                    self.zscore_threshold,
                    horizon
                )
                for group_id in chunk
            )
            
            # Count signals in this chunk
            chunk_signals = 0
            # Append non-empty results
            for df in parallel_results:
                if not df.empty:
                    all_results.append(df)
                    chunk_signals += len(df)
            
            signal_counts.append(chunk_signals)
            print(f"Chunk {chunk_idx+1}: Generated {chunk_signals} signals")
        
        # Concatenate results
        if all_results:
            self.precomputed_signals = pd.concat(all_results).reset_index(drop=True)
            print(f"Total signals generated: {len(self.precomputed_signals)}")
            print(f"Signals span {self.precomputed_signals['date'].nunique()} unique trading days")
            
            # Distribution of signals
            if len(self.precomputed_signals) > 0:
                signal_counts = self.precomputed_signals['signal'].value_counts()
                print(f"Signal distribution: {dict(signal_counts)}")
        else:
            print("WARNING: No signals were generated!")
            self.precomputed_signals = pd.DataFrame()

    def _process_group_signal(self, group_id, group_df_main_dict, df_pairs_group, z_col, zscore_threshold, horizon):
        """Process signals for a specific group (used for parallel processing)"""
        if group_id not in group_df_main_dict:
            return pd.DataFrame()
        
        group_df_main = group_df_main_dict[group_id]
        
        if group_df_main.empty or df_pairs_group.empty:
            return pd.DataFrame()
        
        # Create efficient lookups
        permnos = group_df_main['permno'].values
        z_scores = group_df_main[z_col].values
        dates = group_df_main['date'].unique()
        
        # Create lookup dictionaries
        z_map = dict(zip(permnos, z_scores))
        
        # Process in chunks for memory efficiency
        chunk_size = 1000
        results = []
        
        for i in range(0, len(df_pairs_group), chunk_size):
            df_chunk = df_pairs_group.iloc[i:i+chunk_size].copy()
            
            # Map z-scores efficiently
            df_chunk['z_black'] = df_chunk['permno_black'].map(z_map)
            df_chunk['z_white'] = df_chunk['permno_white'].map(z_map)
            
            # Drop rows with NaN z-scores
            df_chunk = df_chunk.dropna(subset=['z_black', 'z_white'])
            
            if df_chunk.empty:
                continue
                
            # Calculate z_diff
            df_chunk['z_diff'] = df_chunk['z_black'] - df_chunk['z_white']
            
            # Process each date
            for date in dates:
                df_date = df_chunk.copy()
                df_date['date'] = date
                
                # Generate signals using vectorized operations
                z_diff_values = df_date['z_diff'].values
                mask_short = z_diff_values >= zscore_threshold
                mask_long = z_diff_values <= -zscore_threshold
                
                if not (np.any(mask_short) or np.any(mask_long)):
                    continue
                    
                signals = np.full(len(z_diff_values), '', dtype=object)
                signals[mask_short] = 'short_black_long_white'
                signals[mask_long] = 'long_black_short_white'
                
                df_date['signal'] = signals
                
                # Filter valid signals only
                df_date = df_date[signals != '']
                
                if len(df_date) > 0:
                    # Add method info for later reference
                    df_date['zscore_method'] = self.zscore_method
                    df_date['horizon'] = horizon
                    df_date['lookback'] = self.lookback_period
                    results.append(df_date)
            
            # Clear memory
            del df_chunk
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.concat(results)
        return result_df

    def generate_signals(self, date):
        """Get signals for a specific date"""
        if self.precomputed_signals is None or self.precomputed_signals.empty:
            return []

        signals_today = self.precomputed_signals[self.precomputed_signals['date'] == date]
        
        if signals_today.empty:
            return []
        
        result = signals_today[['date', 'permno_black', 'permno_white', 'signal', 'z_diff', 
                               'zscore_method', 'horizon', 'lookback']].to_dict('records')
        
        return result