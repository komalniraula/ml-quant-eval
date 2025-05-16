if __name__ == "__main__":
    # Specify period as 'train' or 'test'
    period = 'test'
    
    print(f"Running backtest for period: {period}")
    
    # Run the backtest
    results = run_backtest(
        df_main_path='final_backtest_data.csv',
        df_pairs_path='corr_coin.csv',
        period=period
    )
    
    if results is not None:
        print("Backtest completed successfully!")
    else:
        print("Backtest failed. Check error logs.")