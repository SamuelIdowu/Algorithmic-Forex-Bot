# 🧠 Algorithmic Trading Project Context

## 🏗️ Overview
This project is a **Python-based algorithmic trading framework** designed for both **backtesting** and **live (real-money or paper)** trading.  
It uses free financial APIs, modular code, and supports both stock and crypto trading.

---

## ⚙️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|----------|
| **Language** | Python | Core logic, data, and execution |
| **Data APIs** | `yfinance`, `alpha_vantage`, `finnhub`, `ccxt` | Market data (stocks, forex, crypto) |
| **Analysis** | `pandas`, `numpy`, `matplotlib`, `seaborn` | Data cleaning, analysis, plotting |
| **Indicators** | `TA-Lib`, `pandas_ta` | Technical indicators |
| **Backtesting** | `backtrader`, `vectorbt` | Strategy simulation |
| **Live Trading** | `Alpaca` (stocks) / `ccxt` (crypto) | Real or paper trade execution |
| **Visualization/UI** | `matplotlib`, `plotly`, `dash`, `streamlit` | Charts and dashboards |
| **Environment** | `.env`, `python-dotenv` | Store API keys and config safely |
 
---

## 📁 Project Structure

```
algo_trader/
│
├── data/
│   ├── raw/               # raw downloaded data
│   └── processed/         # cleaned data
│
├── strategies/
│   └── moving_average.py  # example strategy file
│
├── backtest/
│   └── backtest_engine.py # backtest runner
│
├── live_trading/
│   └── trader.py          # live/paper trading logic
│
├── utils/
│   ├── data_loader.py     # fetch market data
│   ├── indicators.py      # technical indicator functions
│   └── config.py          # load .env and global settings
│
├── main.py                # entry point
├── requirements.txt       # dependencies
├── .env                   # environment variables (ignored by Git)
└── README.md
```

---

## 📦 Requirements

### `requirements.txt`
```txt
pandas
numpy
matplotlib
yfinance
backtrader
python-dotenv
ccxt
pandas_ta
```

---

## ⚙️ Configuration

### `.env`
```env
MODE=backtest        # or 'live'
API_KEY=your_api_key
API_SECRET=your_api_secret
```

### `utils/config.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
MODE = os.getenv("MODE", "backtest")
```

---

## 💡 Strategy Example

### `strategies/moving_average.py`
```python
import backtrader as bt

class MovingAverageStrategy(bt.Strategy):
    params = dict(short_window=20, long_window=50)

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(period=self.p.short_window)
        self.sma_long = bt.indicators.SimpleMovingAverage(period=self.p.long_window)

    def next(self):
        if not self.position:
            if self.sma_short[0] > self.sma_long[0]:
                self.buy()
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.sell()
```

---

## 🧪 Backtesting Module

### `backtest/backtest_engine.py`
```python
import backtrader as bt
import yfinance as yf
from strategies.moving_average import MovingAverageStrategy

def run_backtest(symbol="AAPL", start="2022-01-01", end="2023-01-01"):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=yf.download(symbol, start, end))
    cerebro.adddata(data)
    cerebro.addstrategy(MovingAverageStrategy)
    cerebro.broker.setcash(10000)
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()
```

---

## 💸 Live Trading Module

### `live_trading/trader.py`
```python
import ccxt
from utils.config import API_KEY, API_SECRET

def place_trade(symbol="BTC/USDT", side="buy", amount=0.001):
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })
    order = exchange.create_market_order(symbol, side, amount)
    print("Order executed:", order)
```

---

## 🚀 Main Entry File

### `main.py`
```python
from utils.config import MODE
from backtest.backtest_engine import run_backtest
from live_trading.trader import place_trade

if MODE == "backtest":
    run_backtest("AAPL")
else:
    place_trade("BTC/USDT", "buy", 0.001)
```

---

## 🧩 How to Run

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Configure
Edit `.env` with your keys and mode (backtest/live).

### 3. Run the Project
```bash
python main.py
```

---

## 📊 Viewing Results
During backtesting:
- You’ll see console logs of portfolio values.  
- `backtrader` automatically plots performance charts.  

During live mode:
- Orders are printed in the console.  
- You can extend with `plotly` or `dash` to visualize trade history or equity curves.

---

## 🧠 Next Steps
- Add more strategies in `strategies/`.  
- Expand `indicators.py` with RSI, MACD, or Bollinger Bands.  
- Implement a `logger` to record every trade.  
- Connect to a cloud server (AWS, Render, or PythonAnywhere) for continuous operation.  
