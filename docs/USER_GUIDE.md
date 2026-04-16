# 🤖 Algo Trader — User Guide

> **Version 2.0** · AI Hedge Fund + Legacy Bot

---

## 📋 Prerequisites

```bash
cd "algo_trader"
uv sync  # install all dependencies
```

---

## 🖥️ Interactive CLI (recommended)

The fastest way to use everything — no flag memorisation needed.

```bash
python cli.py
```

```
  █████╗ ██╗   ██╗
 ██╔══██╗██║   ██║   AI Hedge Fund v2.0
 ...

? What would you like to do?
  🤖  AI Hedge Fund   — autonomous agent loop
  🔄  Retrain Model   — force model retrain
  ─── Legacy Tools ─────────────────────────
  📈  Train Model
  🔮  Predict
  📉  Backtest
  💵  Paper Trading
  ─── Dashboard ────────────────────────────
  🚪  Exit
```

---

## 🤖 AI Hedge Fund — Agent Loop

The core of v2.0. A fully autonomous pipeline that runs on a schedule:

```
MarketDataAnalyst → QuantAnalyst → SentimentAnalyst → FundamentalsAnalyst
                                                         ↓
                                              RiskManager (weighted vote)
                                                         ↓
                                             PortfolioManager (orders + DB)
                                                         ↓
                                          RetrainerAgent (auto-retrain check)
```

### Via CLI
Run `python cli.py` → select **🤖 AI Hedge Fund**.

### Via Command Line

| Command | Description |
|:---|:---|
| `python run_agent.py --mode backtest --once` | One-shot backtest cycle (smoke test) |
| `python run_agent.py --mode paper` | Continuous paper trading (hourly loop) |
| `python run_agent.py --mode live` | Live trading ⚠️ |
| `python run_agent.py --symbols BTC-USD EURUSD=X --interval 30` | Custom symbols, 30-min cycle |
| `python run_agent.py --retrain BTC-USD` | Force retrain for BTC-USD |
| `python run_agent.py --disable-agent SentimentAnalyst` | Skip an agent |
| `python run_agent.py --once` | Single cycle then exit |

### Agent Pipeline

| Priority | Agent | Role | Data Source |
|:---:|:---|:---|:---|
| 1 | `MarketDataAnalyst` | Fetch OHLCV + features | Yahoo Finance → SQLite |
| 2 | `QuantAnalyst` | ML signal + confidence | Symbol `.pkl` model |
| 3 | `SentimentAnalyst` | NLP sentiment | AV News API + VADER |
| 4 | `FundamentalsAnalyst` | Fundamental signal | CoinGecko / yfinance |
| 6 | `RiskManager` | Weighted vote + ATR sizing | Analyst outputs |
| 7 | `PortfolioManager` | Order dispatch + DB logging | RiskManager output |
| 9 | `RetrainerAgent` | Auto-retrain on degraded win rate | train_model.py |

### Self-Improvement Layers

| Layer | Mechanism | Where |
|:---|:---|:---|
| **L1** | Auto-retrains model when win rate < threshold or model too old | `RetrainerAgent` |
| **L2** | Adapts analyst weights based on per-analyst historical accuracy | `RiskManager` + `agent_performance` table |
| **L3** | Reinforcement Learning *(planned — stable-baselines3)* | — |

### Configuration (`.env`)

```env
AGENT_SYMBOLS=BTC-USD,EURUSD=X,GC=F
AGENT_INTERVAL_MINUTES=60
AGENT_MODE=backtest          # backtest | paper | live
AGENT_INITIAL_CAPITAL=10000
SENTIMENT_ENGINE=vader        # vader | groq
REQUIRED_VOTE_SCORE=0.55     # min score to BUY/SELL
MAX_DRAWDOWN_PCT=15           # halt if portfolio drops > 15%
RISK_PER_TRADE_PCT=1
ATR_SL_MULT=2.0
ATR_TP_MULT=3.0
RETRAIN_WIN_RATE_THRESHOLD=0.45
RETRAIN_MAX_AGE_DAYS=30
```

---

## 🤖📱 Telegram Bot Integration

Monitor market analysis and control the bot from any device via Telegram.

### Features
- **Real-time Alerts**: Push notifications for every BUY/SELL signal including entry prices and CIO rationales.
- **Interactive Commands**: Use the built-in menu or commands to manage the bot.
- **AI Chat**: Directly consult with the LLM fund manager about strategies or predictions.

### Setup
1. Message [@BotFather](https://t.me/botfather) on Telegram to create a bot and get a **Token**.
2. Find your **Chat ID** (using a bot like `@userinfobot`).
3. Update `.env`:
   ```env
   TELEGRAM_ENABLED=True
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```
4. Run: `python cli.py` → **🤖📱 Telegram Bot** (or `python telegram_bot.py`).

### Commands
- `/start` - Launch the interactive main menu.
- `/status` - System status and prediction metrics.
- `/predictions` - Prediction accuracy & analytics.
- `/signals` - Recent prediction signals with TP/SL levels.
- `/analyze <symbol>` - Fetch the latest CIO reasoning for a symbol.
- `/list_symbols` - View all natively supported symbols and categories.
- `/train [symbol]` - Interactive menu to train models. If symbol provided (e.g. `/train AAPL`), goes straight to interval selection.
- `/predict [symbol]` - Fetch a one-shot prediction. Supports arguments (e.g. `/predict BTC-USD`).
- `/tp [symbol]` - Run a **Train → Predict** chain in one click.
- `/retrain <symbol>` - Manually trigger the RetrainerAgent.
- `/backtest [symbol]` - Run legacy backtesting. Supports arguments.
- `/paper [symbol]` - Launch legacy paper trading. Supports arguments.
- `/config` - **Interactive Config Editor**: Modify trade and risk settings (SL/TP mults, risk %) directly from your phone.
- `/help` - List all commands and features.

> [!TIP]
> **Custom Symbols**: If a symbol isn't in the native picker, use the **➕ Custom Ticker** button or provide it as a command argument. Most Yahoo Finance tickers are supported!

---

## 💬 AI Consultation (CLI)

Consult with the AI Fund Manager directly in your terminal. The manager has full context of your database (trade history, analyst performance).

```bash
python chat_cli.py
```

- Ask: *"Why did the bot signal SELL on BTC last week?"*
- Ask: *"What is our current biggest risk?"*
- Ask: *"Which analyst is performing best?"*

---

---

## 📈 Train a Model (Legacy)

```bash
# Default model path (auto-inferred from symbol)
python train_model.py --symbol BTC-USD --start 2020-01-01 --end 2025-12-31

# Custom paths
python train_model.py --symbol GC=F \
  --start 2020-01-01 --end 2025-12-31 \
  --model_path models/gold_model.pkl \
  --scaler_path models/gold_scaler.pkl
```

> **Tip**: use `python cli.py` → **📈 Train Model** to avoid memorising paths.

---

## 🔮 Predict (Next Candle)

```bash
python predict.py --symbol BTC-USD
python predict.py --symbol GC=F --model_path models/gold_model.pkl --scaler_path models/gold_scaler.pkl
```

---

## 📉 Backtest (Legacy Backtrader)

```bash
STRATEGY=ml_predictive MODE=backtest python main.py
STRATEGY=ml_predictive_risk_managed MODE=backtest python main.py
STRATEGY=moving_average MODE=backtest python main.py
STRATEGY=rsi MODE=backtest python main.py
```

| `STRATEGY` variable | Description |
|:---|:---|
| `moving_average` | SMA crossover |
| `rsi` | RSI oscillator |
| `ml_predictive` | ML model signals |
| `ml_predictive_risk_managed` | ML + ATR stop-loss/take-profit |

---

## 💵 Paper Trading (Legacy — single symbol)

```bash
MODE=paper TRADING_SYMBOL=SPY python main.py
```

> For **multi-symbol** paper trading use `run_agent.py --mode paper` instead.

---

## 🛠️ Full Configuration Reference

| Variable | Description | Default |
|:---|:---|:---|
| `MODE` | `backtest` / `paper` / `live` | `backtest` |
| `STRATEGY` | Legacy strategy name | `moving_average` |
| `TRADING_SYMBOL` | Legacy single symbol | `AAPL` |
| `ALPACA_API_KEY` | Alpaca key ID | — |
| `ALPACA_API_SECRET` | Alpaca secret | — |
| `ALPHA_VANTAGE_API_KEY` | AV key (news + price data) | — |
| `AGENT_SYMBOLS` | Agent loop symbol list | `BTC-USD,EURUSD=X,GC=F` |
| `AGENT_INTERVAL_MINUTES` | Minutes between agent cycles | `60` |
| `AGENT_MODE` | Agent loop default mode | `backtest` |
| `AGENT_INITIAL_CAPITAL` | Starting capital ($) | `10000` |
| `SENTIMENT_ENGINE` | `vader` (offline) or `groq` | `vader` |
| `NEWS_API_KEY` | NewsAPI key (optional) | — |
| `GROQ_API_KEY` | Groq key (only if `SENTIMENT_ENGINE=groq`) | — |
| `REQUIRED_VOTE_SCORE` | Weighted score threshold to trade | `0.55` |
| `MAX_DRAWDOWN_PCT` | Max portfolio drawdown before halt | `15` |
| `RISK_PER_TRADE_PCT` | % of capital risked per trade | `1` |
| `ATR_SL_MULT` | Stop-loss = entry − (mult × ATR) | `2.0` |
| `ATR_TP_MULT` | Take-profit = entry + (mult × ATR) | `3.0` |
| `RETRAIN_WIN_RATE_THRESHOLD` | Trigger retrain below this win rate | `0.45` |
| `RETRAIN_MAX_AGE_DAYS` | Trigger retrain if model older than N days | `30` |
| `TELEGRAM_ENABLED` | Enable push alerts and bot server | `False` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot API token | — |
| `TELEGRAM_CHAT_ID` | Telegram user/chat ID for alerts | — |

---

## 🗂️ Project Structure (v2.0)

```
algo_trader/
├── agents/                    # Plugin agents
│   ├── base_agent.py
│   ├── chat_agent.py          # Context-aware chat logic
│   ├── quant_analyst.py
│   ├── ...
├── dashboard/                 # Streamlit UI
├── data/
│   └── db_manager.py          # SQLite engine
├── docs/                      # Documentation
├── models/                    # Trained models (.pkl)
├── utils/
│   ├── config.py
│   └── telegram_utils.py      # Notification helpers
├── cli.py                     # Interactive CLI
├── chat_cli.py                # Console AI Chat
├── telegram_bot.py            # Telegram Server
├── run_agent.py               # Main agent loop
└── .env                       # Secrets
```

---

## 🔌 Adding a New Agent

1. Create `agents/my_new_agent.py`
2. Subclass `BaseAgent` and set `name`, `role`, `priority`
3. Implement `run(context)` → mutate `context` → return it
4. The registry auto-discovers it on next run — no registration needed

```python
from agents.base_agent import BaseAgent

class MyNewAgent(BaseAgent):
    name     = "MyNewAgent"
    role     = "analyst"
    priority = 5           # runs between Sentiment and Risk

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        context["my_signal"] = {s: "BULLISH" for s in symbols}
        return context
```
