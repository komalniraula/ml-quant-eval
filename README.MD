# ML-Quant-Eval

This repository focuses on **evaluating quantitative trading strategies** using machine learning techniques, specifically targeting **mean-reversion** and **trend-following** approaches. The primary goal is to analyze whether fundamental-based peer portfolios, formed using a k-Nearest Neighbors (k-NN) clustering approach, can generate statistically significant alpha.

---

## 📚 Project Overview

- **Portfolio Formation**  
  Portfolios are dynamically constructed each quarter using k-NN clustering based on company fundamentals (e.g., ROE, size, leverage). Companies are grouped into peer portfolios of fundamentally similar firms.

- **Trading Strategies**  
  - **Mean-Reversion**: Positions are taken based on z-scores of peer-relative returns. Go long on undervalued stocks and short overvalued ones relative to their peer group.  
  - **Trend-Following**: Positions are initiated based on trend momentum within peer groups.

- **Volatility & Forecasting Models**  
  - **GARCH(1,1)** is used to estimate time-varying volatility.  
  - **Ornstein-Uhlenbeck (OU)** processes are employed for mean-reversion forecasting.

- **Backtesting Framework**  
  A custom backtesting engine is implemented to simulate daily trading decisions, position management, and portfolio rebalancing.  
  - Transaction costs: $0.01 per share  
  - Financing costs:  
    - Long = Fed Funds Rate + 1.5%  
    - Short = Fed Funds Rate + 1.0%  
  - Risk-managed capital allocation based on inverse volatility and Sharpe ratio estimates.

- **Data Sources**  
  - Price Data: **CRSP**  
  - Fundamental Data: **Compustat**  
  - Risk-Free Rates: **Federal reserve board data from WRDS**

---

## 📅 Current Status

- Initial strategy evaluation scripts and notebooks are uploaded.  
- A well-structured and modularized project with organized scripts and classes will be uploaded soon.

---

## 📈 Upcoming Features

- Interactive performance reports and visualizations.  
- Modular pipeline for easy strategy extension.  
- Enhanced risk management and transaction cost modeling.
