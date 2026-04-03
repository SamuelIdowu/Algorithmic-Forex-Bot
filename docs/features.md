# 🧠 Algo Trader Features

This document outlines the features of the **Algo Trader** application, a Python-based algorithmic trading framework.

## 🏗️ Core Framework

- **Modular Architecture**: Designed for extensibility with separate modules for data, strategies, backtesting, and live trading.
- **Dual Mode**: Supports both **Backtesting** (historical data) and **Live/Paper Trading** (real-time execution).
- **Environment Configuration**: Secure management of API keys and settings via `.env` files.

## 💹 Trading Capabilities

### Backtesting
- **Engine**: Powered by `backtrader` for robust strategy simulation.
- **Data Feeds**: Seamless integration with historical data from `yfinance`.
- **Performance Metrics**: Automatic calculation of portfolio value, returns, and trade logs.
- **Visualization**: Built-in plotting of strategy performance.

### Live & Paper Trading
- **Execution**: Real-time trade execution using `ccxt` (Crypto) and `Alpaca` (Stocks).
- **Paper Trading**: Risk-free strategy testing with Alpaca's paper trading API.
- **Order Types**: Support for market orders (buy/sell).

## 📈 Strategies

The framework includes a modular strategy system allowing users to define custom logic. Built-in strategies include:

1.  **Moving Average Crossover**:
    -   **Logic**: Buys when Short SMA > Long SMA, Sells when Short SMA < Long SMA.
    -   **Customization**: Configurable short and long window periods.

2.  **RSI (Relative Strength Index)**:
    -   **Logic**: Mean reversion strategy. Buys when oversold (RSI < 30), Sells when overbought (RSI > 70).
    -   **Customization**: Configurable RSI period and buy/sell thresholds.

## 🔌 Data & Integrations

- **Stock Market**:
    -   `yfinance`: Historical data for backtesting.
    -   `Alpha Vantage`: Market data API support.
    -   `Alpaca`: Brokerage integration for trading.
- **Cryptocurrency**:
    -   `ccxt`: Unified API for connecting to multiple crypto exchanges (e.g., Binance).
- **Analysis Tools**:
    -   `pandas` & `numpy`: Data manipulation and numerical analysis.
    -   `TA-Lib` & `pandas_ta`: Comprehensive library of technical indicators.

## 📊 Visualization & Reporting

- **Charts**: `matplotlib` integration for rendering trade history and equity curves.
- **Logging**: Detailed console logging of trade signals, order execution, and portfolio status.
