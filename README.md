# 🧠 Algorithmic Trading Bot

A Python-based algorithmic trading framework designed for both **backtesting** and **live/paper trading**. This project uses free financial APIs, modular code, and supports both stock and crypto trading.

## 🚀 Features

- **Multi-API support**: Alpha Vantage, Alpaca, Yahoo Finance
- **Backtesting**: Backtrader integration for strategy testing
- **Live Trading**: Support for paper and live trading with Alpaca
- **Forex Support**: Data fetching for major currency pairs via Alpha Vantage
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Modular Design**: Clean, maintainable code structure

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|----------|
| **Language** | Python | Core logic, data, and execution |
| **Data APIs** | `yfinance`, `alpha_vantage`, `finnhub`, `ccxt` | Market data (stocks, forex, crypto) |
| **Analysis** | `pandas`, `numpy`, `matplotlib`, `seaborn` | Data cleaning, analysis, plotting |
| **Indicators** | `TA-Lib`, `pandas_ta` | Technical indicators |
| **Backtesting** | `backtrader` | Strategy simulation |
| **Live Trading** | `Alpaca` (stocks) / `ccxt` (crypto) | Real or paper trade execution |
| **Visualization** | `matplotlib`, `plotly` | Charts and dashboards |
| **Environment** | `.env`, `python-dotenv` | Store API keys and config safely |

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

## 📦 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd algo_trader
```

### 2. Setup Environment (Conda)
```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate algo_trader
```

### 3. Configure API Keys
1. Get an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Get API keys from [Alpaca](https://alpaca.markets/) (for live/paper trading)
3. Edit `.env` with your keys and desired mode:

```env
# Trading Mode
MODE=backtest        # or 'live' or 'paper'

# Alpaca API Keys (get from https://alpaca.markets/)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use for paper trading

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

## 🧪 Usage

### 1. Backtesting
Set `MODE=backtest` in your `.env` file and run:
```bash
python main.py
```

This will run the default moving average strategy on AAPL stock data.

### 2. Paper Trading
Set `MODE=paper` in your `.env` file and run:
```bash
python main.py
```

This will run the trading bot in paper trading mode using Alpaca's paper trading API.

### 3. Live Trading
Set `MODE=live` in your `.env` file and run:
```bash
python main.py
```

⚠️ **Warning**: This will use real money. Only use with money you can afford to lose.

## 💡 Strategies

The framework comes with two built-in strategies:

1. **Moving Average Crossover**: Buys when short MA crosses above long MA, sells when short MA crosses below long MA
2. **RSI Strategy**: Buys when RSI is below 30 (oversold), sells when RSI is above 70 (overbought)

You can create additional strategies by extending the `bt.Strategy` class in the `strategies/` directory.

## 📊 Adding Custom Indicators

The `utils/indicators.py` file contains many built-in technical indicators. You can add your own by following the same pattern:

```python
def calculate_your_indicator(data: pd.Series, parameter: int = 14) -> pd.Series:
    # Your indicator calculation here
    return result
```

## 🧩 Running Different Strategies

To run a different strategy in backtesting, modify the `main.py` file:

```python
run_backtest(
    symbol="AAPL", 
    strategy_class=YourCustomStrategy, 
    start="2022-01-01", 
    end="2023-01-01",
    initial_cash=10000
)
```

## 🔒 Security

- Store all API keys in the `.env` file, which is git-ignored
- Never commit your `.env` file to version control
- Use paper trading mode to test strategies before going live
- Regularly monitor and audit your trading activity

## 🤖 Best Practices

- Start with paper trading to test your strategies
- Use stop-losses to limit potential losses
- Backtest extensively with historical data before live trading
- Monitor your bot regularly when running live
- Keep your API keys secure

## 📈 Next Steps

- Add more sophisticated strategies
- Implement risk management features
- Create a dashboard for real-time monitoring
- Add support for cryptocurrency trading with CCXT
- Implement advanced technical indicators

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This is for educational purposes only. Trading financial instruments is risky, and past performance is not indicative of future results. The authors are not responsible for any losses incurred from using this software.