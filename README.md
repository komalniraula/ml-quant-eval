# Systematic Alpha: Mean-Reversion Trading Strategy

This project implements a systematic mean-reversion trading strategy using machine learning and stochastic models. It explores the generation of alpha by constructing peer portfolios with K-Means clustering, estimating volatility with GARCH(1,1), and forecasting returns using the Ornstein-Uhlenbeck (OU) process.  

> 📅 **Final Report Date:** May 15, 2025  
> 🎓 **Institution:** New York University Stern School of Business  
> 👤 **Author:** Komal Niraula

---

## 📖 Project Overview

This research investigates whether portfolios formed based on fundamental similarity and filtered through rigorous statistical techniques can consistently deliver statistically significant alpha.

Key Components:
- **Peer Portfolio Formation:** K-Means clustering based on firm fundamentals.
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

---

## 🧩 Implementation Details

- **Language:** Python 3.13  
- **Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `arch`, `joblib`, `matplotlib`, `seaborn`
- **Parallelization:** Joblib with `n_jobs=4`

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/systematic-alpha-mean-reversion.git
cd systematic-alpha-mean-reversion

# Run backtest (example script)
python run_backtest.py
