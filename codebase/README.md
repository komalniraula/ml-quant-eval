# Codebase

This folder contains all the Python code for the systematic trading strategy implementation.

## Structure

The main implementation is in the `backtest/` folder, which contains:

```
backtest/
├── backtest_engine.py — Main simulation engine for backtesting
├── grid_search.py — Hyperparameter optimization tools
├── portfolio_manager.py — Portfolio construction and management
├── performance.py — Performance metrics calculation
├── signal_generator.py — Z-score based signal generation
├── trade.py — Trade execution and cost modeling
├── run.py — Runner script for backtesting
├── main.py — Main entry point with parameter configuration
└── backtest.ipynb — Interactive notebook for exploring results
```

## Installation

Before running the code, install the required dependencies using the `requirements.txt` file in the main project folder:

```bash
pip install -r ../requirements.txt
```

This will install all necessary libraries like pandas, numpy, scikit-learn, statsmodels, and other dependencies.

## Running the Code

### Quick Start

To run the backtest with default parameters:

```bash
cd codebase/backtest
python main.py
```

### Configuration

Parameters are defined in `main.py` as a dictionary:

```python
param_grid = {
    'COINTEGRATION_THRESHOLD': [0.05],
    'CORRELATION_THRESHOLD': [0.9],
    'ZSCORE_METHOD': ['classical'],
    'ZSCORE_THRESHOLD': [1],
    'LOOKBACK_PERIOD': [10],
    'HORIZON': [10],
    'MAX_HOLDING_DAYS': [10],
    'INITIAL_CAPITAL': 1_000_000_000
}
```

You can modify the parameters by changing the values in these lists. The backtest will automatically run for all combinations of parameters.

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

### Interactive Analysis

For interactive exploration, use the Jupyter notebook:

```bash
cd codebase/backtest
jupyter notebook backtest.ipynb
```