# Designing Alpha: Machine Learning & Stochastic Models for Systematic Trading

This project implements a systematic mean-reversion trading strategy using machine learning and stochastic models. It explores the generation of alpha by constructing peer portfolios with K-Nearest Neighbors (KNN) clustering, estimating volatility with GARCH(1,1), and forecasting returns using the Ornstein-Uhlenbeck (OU) process.  

---

## 📖 Project Overview

This research investigates whether portfolios formed based on fundamental similarity and filtered through rigorous statistical techniques can consistently deliver statistically significant alpha.

Key Components:
- **Peer Portfolio Formation:** K-Nearest Neighbors clustering based on firms' quarterly financial reports.
- **Volatility Estimation:** GARCH(1,1) modeling.
- **Forecasting:** OU Process for mean-reversion behavior.
- **Signal Generation:** Classical and OU-based z-scores.
- **Backtesting:** In-sample (2015–2021) and Out-of-sample (2022–2024).
- **Performance Metrics:** Sharpe Ratio, Sortino Ratio, CAPM Alpha, Hit Rate.

---

## 📊 Results Summary

| Period        | Sharpe | Sortino | CAPM Alpha | Hit Rate (%) |
|----------------|--------|---------|------------|--------------|
| In-Sample      | 0.53   | 0.94    | -0.040     | 61.1%        |
| Out-of-Sample  | 0.36   | 0.39    | 0.002      | 54.82%       |

---

## 📚 Methodology

1. **Data Sources**  
   - CRSP: Daily stock prices and returns.  
   - Compustat: Quarterly financial fundamentals.  
   - FRED: Federal Funds Rate for financing cost.

2. **Capital Allocation**  
   - $1 billion invested fresh every quarter.
   - Inverse volatility weighting for position sizing.

3. **Trading Constraints**  
   - Transaction Cost: $0.01 per share.
   - Financing Costs:  
     - Long = Fed Funds + 1.5%  
     - Short = Fed Funds + 1.0%  
   - Max Holding Period: 20 Trading Days.
   - **Liquidity Constraint:** Can only trade up to **10% of the 20-day average trading volume**.

---

## 🧩 Implementation Details

- **Language:** Python 3.13  
- **Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `arch`, `joblib`, `matplotlib`, `seaborn`  
- **Parallelization:** Joblib with `n_jobs=4`

---

## 📂 Project Structure

```
codebase/
├── backtest/
│   ├── backtest_engine.py — Main backtest execution logic.
│   ├── grid_search.py — Hyperparameter tuning and model selection.
│   ├── portfolio_manager.py — Capital allocation and portfolio rebalancing logic.
│   ├── performance.py — Calculates Sharpe, Sortino, Alpha, Hit Rate, Drawdown, etc.
│   ├── signal_generator.py — Generates trading signals based on z-scores.
│   ├── trade.py — Trade execution logic and transaction cost adjustments.
│   ├── run.py — Main script to run complete backtest pipeline.
│   ├── main.py — Entry point for modular testing and debugging.
│   └── backtest.ipynb — Jupyter Notebook for exploratory backtesting.
├── database/
│   └── corr_coin.csv — Final list of identified pairs (with correlation and cointegration metrics).
├── test_reports/
│   ├── in_sample_results.csv — Performance metrics and trade logs for in-sample period (2015–2021).
│   └── out_sample_results.csv — Performance metrics and trade logs for out-of-sample period (2022–2024).
├── .gitignore — Specifies files and folders to exclude from version control.
├── requirements.txt - libraries
└── README.md — Project overview, methodology, and instructions.
```

## 📓 Jupyter Notebooks

**Important**: The following notebooks must be executed in this sequence before running the backtest to collect and prepare the data properly:

1. **value&growth_quaterly(compustat).ipynb** — Identifying Growth and Value portfolios using Compustat data.
2. **market_data(CRSP).ipynb** — Processing and analysis of stock price data from CRSP.
3. **fed_rates_data(FRB WRDS).ipynb** — Data collection and preprocessing for Federal Funds Rates.
4. **knn_clustering.ipynb** — K-Nearest Neighbors clustering to form peer portfolios based on fundamentals.
5. **GARCH, OU & data analysis.ipynb** — GARCH volatility modeling and OU parameter estimation.

After running these notebooks in sequence, you can proceed with the backtesting process.

## 🚀 Getting Started

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/komalniraula/ml-quant-eval.git

# Navigate to the project directory
cd ml-quant-eval

# Install required packages
pip install -r requirements.txt

# Run notebooks in sequence to prepare data
jupyter notebook value\&growth_quaterly\(compustat\).ipynb
jupyter notebook market_data\(CRSP\).ipynb
jupyter notebook fed_rates_data\(FRB\ WRDS\).ipynb
jupyter notebook knn_clustering.ipynb
jupyter notebook "GARCH, OU & data analysis.ipynb"

Note: Ensure that all datasets are placed in the appropriate folders within the codebase directory structure after running these notebooks. The backtest framework expects the prepared data to be in the same folder.

# Navigate to the backtest directory
cd codebase/backtest

# Run the backtest with default parameters
python main.py