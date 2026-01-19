# 🤖 Algo Trader Bot - User Guide

This guide details the capabilities of your AI-powered trading bot and how to use them.

## 📋 Prerequisites

Ensure you are in the `algo_trader` directory and your environment is active:

```bash
cd "algo_trader"

conda activate algo_trader
```

---

## 🧠 1. Machine Learning (Predictive Mode)

The bot can train a Random Forest model on historical data to predict future price movements.

### Train a Model
Train a model for a specific symbol (e.g., Apple, Gold).

**For Stocks (Default AAPL):**
```bash
python train_model.py --symbol AAPL --start 2020-01-01 --end 2023-01-01
```

**For Gold (XAU/USD using GC=F):**
```bash
python train_model.py --symbol GC=F --start 2020-01-01 --end 2025-12-09 --model_path models/gold_model.pkl --scaler_path models/gold_scaler.pkl
```

### Make a Prediction (Next Candle)
Predict whether the price will go UP or DOWN for the next period.

**For Stocks:**
```bash
python predict.py --symbol AAPL
```

**For Gold:**
```bash
python predict.py --symbol GC=F --model_path models/gold_model.pkl --scaler_path models/gold_scaler.pkl
```

python predict.py --symbol GC=F --model_path models/gold_model.pkl --scaler_path models/gold_scaler.pkl
```

---

## ➕ 2. Adding Other Pairs

You can use any symbol available on **Yahoo Finance**.

1.  **Find the Symbol**: Go to [Yahoo Finance](https://finance.yahoo.com/) and search for the pair you want (e.g., `BTC-USD` for Bitcoin, `EURUSD=X` for Euro/USD, `TSLA` for Tesla).
2.  **Train a Model**: Run the training command with the new symbol and a unique model path.

    **Example for Bitcoin (BTC-USD):**
    ```bash
    python train_model.py --symbol BTC-USD --start 2020-01-01 --end 2025-12-02 --model_path models/btc_model.pkl --scaler_path models/btc_scaler.pkl
    ```

3.  **Predict**: Use the same symbol and model path to make predictions.

    **Example for Bitcoin:**
    ```bash
    python predict.py --symbol BTC-USD --model_path models/btc_model.pkl --scaler_path models/btc_scaler.pkl
    ```

---

## 📉 2. Backtesting

Test strategies on historical data to see how they would have performed.

### Strategies
You can select a strategy using the `STRATEGY` environment variable:

*   `moving_average`: Simple Moving Average Crossover.
*   `rsi`: Relative Strength Index strategy.
*   `ml_predictive`: Uses your trained ML model to buy/sell (Requires training first!).
*   `ml_predictive_risk_managed`: ML strategy with Stop Loss and Take Profit.

### Run a Backtest

**Standard ML Backtest:**
```bash
STRATEGY=ml_predictive MODE=backtest python main.py
```

**Risk-Managed ML Backtest:**
```bash
STRATEGY=ml_predictive_risk_managed MODE=backtest python main.py
```

**Moving Average Backtest:**
```bash
STRATEGY=moving_average MODE=backtest python main.py
```

---

## 💵 3. Paper Trading (Live Demo)

Trade with fake money using real-time market data via Alpaca.

### Setup
1.  Get your API Keys from [Alpaca Paper Trading](https://alpaca.markets/).
2.  Add them to your `.env` file:
    ```
    ALPACA_API_KEY=your_key
    ALPACA_API_SECRET=your_secret
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
    ```

### Run Paper Trader
### Run Paper Trader
```bash
# Default (AAPL)
MODE=paper python main.py

# Specific Symbol
TRADING_SYMBOL=SPY MODE=paper python main.py
```
*Note: The current `main.py` runs the ML strategy for paper trading. Ensure you have trained a model for the symbol you are trading!*

---

## 🛠️ Configuration

The bot is configured via the `.env` file and environment variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODE` | `backtest`, `paper`, or `live` | `backtest` |
| `STRATEGY` | Strategy to run in backtest | `moving_average` |
| `TRADING_SYMBOL` | Symbol to trade in paper/live mode | `AAPL` |
| `ALPACA_API_KEY` | Alpaca Key ID | - |
| `ALPACA_API_SECRET` | Alpaca Secret Key | - |
