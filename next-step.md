# 🤖 Algo Trader: AI Development Guide

## 🎯 Project Objective
We are upgrading a **reactive** algorithmic trading framework (based on simple indicators like SMA/RSI) into a **predictive** machine learning-based trading bot. The goal is to use historical data to train models that predict future price movements (Buy/Sell/Hold) rather than just reacting to past crossovers.

## 🛠️ Tech Stack & Constraints
- **Language:** Python 3.9+
- **Core Engine:** `backtrader` (Backtesting), `ccxt` (Crypto Live), `Alpaca` (Stock Live).
- **ML Libraries:** `scikit-learn` (initial models), `pandas`, `numpy`.
- **Data Handling:** `yfinance` (fetching), `sqlite3` (storage/caching).
- **Environment:** Variables managed via `.env` (API keys must NEVER be hardcoded).

---

## 🗺️ Implementation Roadmap (Step-by-Step)

### Phase 1: Data Pipeline Upgrade (The Foundation)
*Goal: Move from on-the-fly downloading to a persistent database for training ML models.*

1.  **Create Database Manager (`data/db_manager.py`):**
    -   Implement a class to handle SQLite connections.
    -   Create a schema table `market_data` with columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`.
2.  **Update Data Loader (`utils/data_loader.py`):**
    -   Modify the `fetch_data` function.
    -   **Logic:** Check DB first. If data exists, load from DB. If not, download via `yfinance`, save to DB, then return data.
    -   Add a function `prepare_training_data(symbol)` that fetches raw data and prepares it for the ML model.

### Phase 2: Feature Engineering (The Input)
*Goal: Convert raw prices into meaningful signals for the AI.*

1.  **Create Feature Engineer (`utils/features.py`):**
    -   Create a function `add_technical_features(df)`.
    -   **Add:**
        -   **RSI:** 14-period Relative Strength Index.
        -   **SMA_Ratio:** Close price divided by SMA_20 (normalization).
        -   **Log Returns:** `np.log(df.close / df.close.shift(1))`.
        -   **Volatility:** Rolling standard deviation or ATR.
        -   **Lagged Values:** Shifted returns (t-1, t-2) to give the model context of recent history.
    -   **Clean:** Drop `NaN` values resulting from rolling windows.

### Phase 3: The Machine Learning Strategy (The Brain)
*Goal: Implement a strategy that uses a trained model to make decisions.*

1.  **Create Training Script (`train_model.py`):**
    -   Load data using `db_manager`.
    -   Apply `add_technical_features`.
    -   **Target Variable:** Create a 'Target' column where 1 = Price UP next candle, 0 = Price DOWN.
    -   Split data: Train (80%) / Test (20%).
    -   Train a `RandomForestClassifier` from `scikit-learn`.
    -   Save the trained model using `joblib` or `pickle` to a `models/` directory.
2.  **Develop ML Strategy (`strategies/ml_predictive.py`):**
    -   Inherit from `backtrader.Strategy`.
    -   **Init:** Load the saved `.pkl` model.
    -   **Next:**
        -   Construct a single-row dataframe of the *current* candle's features.
        -   Run `model.predict()`.
        -   **Signal:** If Prediction == 1 and no position -> **BUY**. If Prediction == 0 and has position -> **SELL**.

### Phase 4: Risk Management (The Shield)
*Goal: Protect capital when predictions are wrong.*

1.  **Enhance Strategy (`strategies/ml_predictive.py`):**
    -   Add `StopLoss` and `TakeProfit` parameters.
    -   **Logic:** Upon entry, calculate:
        -   `stop_price = buy_price - (2 * ATR)`
        -   `target_price = buy_price + (3 * ATR)`
    -   Send Bracket Orders (Entry + Stop Loss + Take Profit) if supported by the broker/engine.

---

## 📝 Coding Standards for AI
1.  **Documentation:** All functions must have a docstring explaining inputs and outputs.
2.  **Type Hinting:** Use Python type hints (e.g., `def calculate_rsi(prices: pd.Series) -> pd.Series:`).
3.  **Error Handling:** Wrap external API calls (yfinance, exchanges) in `try/except` blocks.
4.  **Logging:** Do not use `print()`. Use the standard `logging` library to record buy/sell signals and errors.

## 🧪 Testing Instructions
- **Backtest First:** Always run `main.py` in `MODE=backtest` before attempting live connections.
- **Verification:** When generating code for Phase 2, ensure features are calculated correctly by plotting them alongside price data using `matplotlib`.