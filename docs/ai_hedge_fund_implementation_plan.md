# 🤖 AI Hedge Fund — Master Implementation Plan

> **Version 2.0** — Updated with full architectural decisions.
> Upgrades the existing ML trading bot into a fully autonomous, self-improving,
> multi-specialist AI Hedge Fund. Pure Python. No agent framework required.

---

## 🏛️ Core Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Framework** | ❌ None (pure Python) | Full control, no lock-in, deterministic execution critical for live trading |
| **LLM** | ✅ Optional (Phase 4 only) | ML models handle decisions; LLM only adds value for NLP sentiment |
| **Sentiment (default)** | VADER (offline, free) | Covers 80% of value with zero cost or latency |
| **Sentiment (upgrade)** | Groq API (llama-3, free) | Context-aware, smarter — easy swap-in later |
| **Learning** | 3 layers (stats-based) | scikit-learn + numpy + SQLite — no RL framework needed for layers 1–2 |
| **Architecture** | Plugin-based agents | New agents added by dropping a file, zero core changes |

---

## 🗺️ Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │         run_agent.py             │
                    │    Autonomous Orchestrator Loop  │
                    │  Perceive → Reason → Act → Reflect│
                    └────────────┬────────────────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │         agents/registry.py          │
               │   Plugin Registry (auto-discovers   │
               │   all agents in agents/ folder)     │
               └──┬──────────┬──────────┬────────────┘
                  │          │          │
         ┌────────▼──┐ ┌─────▼────┐ ┌──▼───────────┐
         │  Market   │ │  Quant   │ │  Sentiment   │
         │  Data     │ │ Analyst  │ │   Analyst    │
         │ Analyst   │ │ (ML/RF)  │ │(VADER / LLM) │
         └────────┬──┘ └─────┬────┘ └──┬───────────┘
                  │          │         │
               ┌──▼──────────▼─────────▼──┐
               │       Fundamentals        │
               │         Analyst           │
               └──────────────┬────────────┘
                              │ Signals
                    ┌─────────▼──────────┐
                    │    Risk Manager    │
                    │  Weighted Voting   │◄── agent_performance DB
                    │  Position Sizing   │    (adapts weights over time)
                    └─────────┬──────────┘
                              │ Approved Signal
                    ┌─────────▼──────────┐
                    │ Portfolio Manager  │
                    │ Order Dispatch     │──► Buy / Sell / Hold
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Memory Layer     │
                    │   (SQLite DB)      │──► Self-Improvement Loop
                    └────────────────────┘
```

---

## 📦 Scalable Plugin Architecture

The key to making the agent extensible is a **plugin registry**. Adding a new analyst never requires touching the core orchestrator.

### `agents/registry.py` [NEW]
```python
import importlib, inspect, pkgutil
from agents.base_agent import BaseAgent

def discover_agents() -> list[BaseAgent]:
    """Auto-discovers all BaseAgent subclasses in the agents/ package."""
    agents = []
    for _, module_name, _ in pkgutil.iter_modules(["agents"]):
        module = importlib.import_module(f"agents.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                agents.append(obj())
    return agents
```

**To add a new agent in the future:**
1. Create `agents/my_new_analyst.py`
2. Inherit from `BaseAgent`, implement `run(context) -> dict`
3. Done — the registry discovers and runs it automatically ✅

### `agents/base_agent.py` [NEW]
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    name: str            # Display name
    role: str            # "analyst" | "manager" | "monitor"
    priority: int = 0    # Execution order (lower = earlier)
    enabled: bool = True # Toggle agents via config without deleting them

    @abstractmethod
    def run(self, context: dict) -> dict:
        """Accepts shared context, returns updated context with own signal."""

    def on_trade_result(self, symbol: str, pnl: float) -> None:
        """Optional hook: called after trade closes for self-improvement."""
        pass
```

---

## 🧠 Three Layers of Self-Improvement (No Framework Needed)

### Layer 1 — Auto-Retraining (`agents/retrainer.py`)
*When the model starts losing, retrain it.*

```
Every N trades per symbol:
  win_rate = wins / last_20_trades
  if win_rate < 0.45 OR model_age > 30 days:
    → fetch fresh data from SQLite
    → call train_model.py programmatically
    → joblib.load() new .pkl into QuantAnalyst (hot-swap, no restart)
    → log retraining event to DB
```
**Libraries:** `scikit-learn`, `joblib`, `sqlite3` — all already in project.

---

### Layer 2 — Dynamic Signal Weight Adaptation (`agents/risk_manager.py`)
*Trust analysts that are more accurate, downweight ones that are wrong.*

```python
# Stored in agent_performance table
quant_accuracy      = quant_correct / quant_total       # e.g. 0.72
sentiment_accuracy  = sentiment_correct / sent_total    # e.g. 0.48
fund_accuracy       = fund_correct / fund_total         # e.g. 0.61

# Normalise to weights that sum to 1
weights = np.array([quant_accuracy, sentiment_accuracy, fund_accuracy])
weights = weights / weights.sum()   # e.g. [0.40, 0.27, 0.33]

# Weighted vote instead of simple majority
score = np.dot(signal_values, weights)
action = "BUY" if score > 0.55 else "SELL" if score < 0.45 else "HOLD"
```
The Risk Manager **automatically adjusts trust** based on each analyst's track record — computed from the `agent_trades` DB table after every closed trade.

**Libraries:** `numpy`, `sqlite3` — already in project.

---

### Layer 3 — Reinforcement Learning (Future Upgrade)
*The model learns buy/sell/hold rewards through trial and simulation.*

```python
# future: agents/rl_quant_analyst.py
from stable_baselines3 import PPO
# State: market features | Action: BUY/SELL/HOLD | Reward: realised PnL
model = PPO("MlpPolicy", trading_env)
model.learn(total_timesteps=500_000)
```
**When to add:** After Layer 1 & 2 are running and you have 3+ months of live trade history.
**New install needed:** `stable-baselines3`, `gymnasium`

---

## 📋 Phase-by-Phase Implementation

---

### Phase 1 — Scalable Scaffold (30 min)

**Files to create:**
```
algo_trader/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base with name/role/priority/enabled
│   └── registry.py            # Auto-discovers all agents in package
├── dashboard/
│   └── app.py                 # Placeholder
└── run_agent.py               # Main loop (stub)
```

**Checklist:** ✅ COMPLETE
- [x] Create `agents/` package with `__init__.py`
- [x] Create `agents/base_agent.py` (abstract class with `run`, `on_trade_result`, `enabled`)
- [x] Create `agents/registry.py` (auto-discovery via `pkgutil`)
- [x] Create `run_agent.py` stub
- [x] Create `dashboard/` folder + placeholder `app.py`
- [x] Verify: `python -c "from agents.registry import discover_agents; print('OK')"` → `Registry: 7 agents loaded`

---

### Phase 2 — Market Data Analyst (1–2 hrs)

**File:** `agents/market_data_analyst.py`

- Wraps `utils/data_loader.py` + `utils/features.py`
- Accepts `symbols` list + `timeframe` from config
- Returns OHLCV + feature DataFrame per symbol in context
- Per-symbol `try/except` — one bad symbol never blocks others

**Signal output:**
```json
{ "BTC-USD": { "ohlcv": "...", "latest_close": 64200.0, "features": "..." } }
```

**Checklist:** ✅ COMPLETE
- [x] Create `agents/market_data_analyst.py`
- [x] Accept symbols + timeframe, re-use `fetch_data()` (no duplication)
- [x] Return structured context dict per symbol
- [x] Per-symbol error isolation (`try/except`)
- [x] Smoke test: live cycle fetched `BTC-USD: close=65881.80, atr=2572.90, rows=342`

---

### Phase 3 — Quant Analyst (1–2 hrs)

**File:** `agents/quant_analyst.py`

- Maps each symbol → its trained `.pkl` model + scaler
- Fallback: uses `ml_strategy_model.pkl` if no symbol-specific model exists
- Returns `quant_signal` (BUY/SELL) and `quant_confidence` (0–1)
- Supports hot-reload via `reload_model(symbol)` for Layer 1 retraining

**Signal output:**
```json
{ "BTC-USD": { "quant_signal": "BUY", "quant_confidence": 0.82 } }
```

**Checklist:** ✅ COMPLETE
- [x] Create `agents/quant_analyst.py`
- [x] Symbol → model file mapping dict (7 symbols mapped, models/ contains 18 `.pkl` files)
- [x] Fallback to `ml_strategy_model.pkl` if no match
- [x] Implement `reload_model(symbol)` for hot-swap during retraining
- [x] Return `quant_signal` + `quant_confidence`
- [x] Live test: `BTC-USD: SELL (conf=0.533)` in end-to-end cycle

---

### Phase 4 — Sentiment Analyst (2–3 hrs)

**File:** `agents/sentiment_analyst.py`

Two-tier implementation — start with VADER, upgrade to LLM later:

| Tier | Library | Cost | Accuracy |
|---|---|---|---|
| **Default** | `vaderSentiment` | Free, offline | Good |
| **Upgrade** | `groq` (llama-3) | Free API | Excellent |

The switch is controlled by a single `.env` variable: `SENTIMENT_ENGINE=vader` or `SENTIMENT_ENGINE=groq`

**Signal output:**
```json
{ "BTC-USD": { "sentiment_signal": "BULLISH", "sentiment_score": 0.63, "engine": "vader" } }
```

**Checklist:** ✅ COMPLETE
- [x] Install: `pip install vaderSentiment requests` (installed, in `requirements.txt`)
- [x] Create `agents/sentiment_analyst.py`
- [x] Implement VADER scorer (default path)
- [x] Implement Groq LLM scorer (optional upgrade path, toggled by `SENTIMENT_ENGINE` env var)
- [x] Fetch headlines: Alpha Vantage News API + NewsAPI fallback
- [x] Graceful degradation: return `NEUTRAL` if API unavailable
- [x] `SENTIMENT_ENGINE=vader` in `.env`
- [x] Live test: `BTC-USD: BULLISH (score=0.199, engine=vader)`

---

### Phase 5 — Fundamentals Analyst (2–3 hrs)

**File:** `agents/fundamentals_analyst.py`

- **Crypto:** CoinGecko `/simple/price` endpoint (free, no API key)
- **Stocks:** `yfinance.Ticker(symbol).info` (P/E, earnings growth)
- **Forex:** Interest rate data (optional) or skip/return NEUTRAL

**Signal output:**
```json
{ "BTC-USD": { "fundamentals_signal": "BULLISH", "reason": "Dominance rising" } }
```

**Checklist:** ✅ COMPLETE
- [x] Create `agents/fundamentals_analyst.py`
- [x] Detect asset class (crypto/stock/forex/futures) from symbol format
- [x] Crypto path: CoinGecko `/coins/{id}` (no key, 7d price change + market cap rank)
- [x] Futures path: CoinGecko `/simple/price` (e.g. GC=F → gold)
- [x] Stock path: `yfinance.Ticker.info` (P/E, earnings growth, short ratio)
- [x] Return `NEUTRAL` on any error (never crash the loop)
- [x] Live test: `BTC-USD [crypto]: BEARISH — 7d change -6.1%`

---

### Phase 6 — Risk Manager with Adaptive Weighting (2–3 hrs)

**File:** `agents/risk_manager.py`

Implements **Layer 2 self-improvement** — weights adapt based on analyst accuracy from DB.

```python
# Reads agent_performance table from SQLite
weights = memory.get_analyst_weights()   # numpy array, sums to 1
score   = np.dot(signal_values, weights)
action  = "BUY" if score > 0.55 else "SELL" if score < 0.45 else "HOLD"
```

Also computes: `stop_loss = entry - (2 × ATR)`, `take_profit = entry + (3 × ATR)`.

**Checklist:** ✅ COMPLETE
- [x] Create `agents/risk_manager.py`
- [x] Implement weighted voting (reads accuracy from `agent_performance` DB table via `db.get_analyst_weights()`)
- [x] Initial weights: equal (1/3 each) until 5+ observations per analyst
- [x] ATR-based position sizing: `size = (capital × RISK_PER_TRADE_PCT) / (atr × ATR_SL_MULT)`
- [x] ATR-based stop-loss and take-profit prices
- [x] Max drawdown guard: blocks all entries if portfolio drawdown > `MAX_DRAWDOWN_PCT`
- [x] Live test: `BTC-USD: HOLD (score=0.333, size=0.02)` — 3-way split signal correctly held

---

### Phase 7 — Portfolio Manager (2–3 hrs)

**File:** `agents/portfolio_manager.py`

**New DB tables in `data/db_manager.py`:**
```sql
CREATE TABLE agent_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT, action TEXT, price REAL, size REAL,
    stop_loss REAL, take_profit REAL,
    quant_signal TEXT, quant_confidence REAL,
    sentiment_signal TEXT, fundamentals_signal TEXT,
    vote_score REAL, weights_used TEXT,  -- JSON snapshot of weights
    pnl REAL                             -- Filled on trade close
);

CREATE TABLE agent_state (
    symbol TEXT PRIMARY KEY,
    position_size REAL, entry_price REAL,
    stop_loss REAL, take_profit REAL,
    cash_deployed REAL, last_updated DATETIME
);

CREATE TABLE agent_performance (
    symbol TEXT, analyst TEXT,         -- 'quant' | 'sentiment' | 'fundamentals'
    correct INTEGER, total INTEGER,    -- Running accuracy counters
    weight REAL, last_updated DATETIME,
    PRIMARY KEY (symbol, analyst)
);
```

**Checklist:** ✅ COMPLETE
- [x] Create `agents/portfolio_manager.py`
- [x] Add **4** new tables to `data/db_manager.py` (`agent_trades`, `agent_state`, `agent_performance`, `retraining_log`)
- [x] Allocation cap: refuse BUY if > 80% capital deployed
- [x] Backtest mode: log trades, skip real order
- [x] Paper/live mode: dispatches to `live_trading/trader.py → place_order()`
- [x] Log every cycle (including HOLDs) to `agent_trades` — full audit trail
- [x] On trade close: updates `pnl`, updates `agent_performance` accuracy counters for all 3 analysts
- [x] Live test: HOLD logged to `agent_trades` table in backtest cycle

---

### Phase 8 — Autonomous Orchestrator Loop (2 hrs)

**File:** `run_agent.py`

```python
from agents.registry import discover_agents

def run_cycle(symbols, mode):
    context = {"symbols": symbols, "mode": mode}
    for agent in sorted(discover_agents(), key=lambda a: a.priority):
        if agent.enabled:
            context = agent.run(context)

schedule.every(interval).minutes.do(run_cycle, symbols, mode)
```

**CLI flags:**
```bash
python run_agent.py --mode paper                   # Safe paper trading
python run_agent.py --mode backtest                # Historical sim
python run_agent.py --mode live                    # Real money⚠️
python run_agent.py --symbols BTC-USD EURUSD       # Override symbols
python run_agent.py --interval 60                  # Minutes between cycles
python run_agent.py --retrain BTC-USD              # Manual retrain trigger
python run_agent.py --disable-agent sentiment      # Toggle agent off
```

**Checklist:** ✅ COMPLETE
- [x] Install: `pip install schedule` (installed)
- [x] Implement `run_cycle()` using registry (auto-discovers all 7 agents)
- [x] `argparse` CLI: `--mode`, `--symbols`, `--interval`, `--retrain`, `--disable-agent`, `--once`
- [x] Startup banner: mode, symbols, agents loaded, interval, timestamp
- [x] Graceful shutdown on `KeyboardInterrupt`
- [x] End-to-end test: `python run_agent.py --mode backtest --symbols BTC-USD --once` exit code 0
- [x] `agent_trades` table populated after cycle (HOLD row logged)

---

### Phase 9 — Self-Retraining Module — Layer 1 (2 hrs)

**File:** `agents/retrainer.py`

Implements **Layer 1 self-improvement**.

**Trigger conditions:**
- Win rate < 45% over last 20 trades for a symbol
- Model age > 30 days since last retrain
- Manual: `--retrain <SYMBOL>` CLI flag

**Flow:**
1. Query `agent_trades` for last 20 closed trades per symbol
2. Compute win rate: `wins / 20`
3. If trigger: call `train_model.py` for that symbol
4. Hot-swap `.pkl` into `QuantAnalyst` without restart
5. Log to `retraining_log` table

**Checklist:** ✅ COMPLETE
- [x] Create `agents/retrainer.py` — priority=9, runs as the Reflect step
- [x] Query `agent_trades` for per-symbol win rate (last 20 closed trades)
- [x] Trigger logic: win rate < `RETRAIN_WIN_RATE_THRESHOLD` OR model age > `RETRAIN_MAX_AGE_DAYS` OR `--retrain` flag
- [x] Call `train_model.train_model()` programmatically with correct symbol paths
- [x] Hot-reload via `QuantAnalyst.reload_model(symbol)` — thread-safe cache eviction
- [x] `retraining_log` table in `data/db_manager.py` with `log_retraining()` + `get_last_retrain_date()`
- [x] `--retrain BTC-USD` flag wired in `run_agent.py`

---

### Phase 10 — Streamlit Dashboard (2–3 hrs)

**File:** `dashboard/app.py`

| Panel | Data Source |
|---|---|
| 📊 Portfolio | `agent_state` table |
| 📈 Equity Curve | `agent_trades.pnl` cumulated |
| 🧾 Trade Log | Last 50 rows of `agent_trades` |
| 🤖 Agent Status | Last run time, next run, mode, agents loaded |
| 🌡️ Signal Heatmap | Latest signals per symbol per analyst |
| ⚖️ Analyst Weights | Current weights from `agent_performance` |
| 🔁 Retraining Log | `retraining_log` table |

```bash
streamlit run dashboard/app.py   # Separate terminal
```

**Checklist:** ✅ COMPLETE
- [x] Install: `pip install streamlit plotly` (both in `requirements.txt`)
- [x] Create `dashboard/app.py` (15 KB) reading from `data/market_data.db`
- [x] All 7 panels implemented: Portfolio · Equity Curve · Trade Log · Agent Status · Signal Heatmap · Analyst Weights · Retraining Log
- [x] Auto-refresh every 60 seconds via `<meta http-equiv="refresh">`
- [x] Launch: `streamlit run dashboard/app.py` → `http://localhost:8501`

---

## 📦 Dependencies

Add to `requirements.txt`:
```
# Agent loop
schedule

# Sentiment (default — no API key)
vaderSentiment

# Sentiment upgrade (optional)
groq

# Dashboard
streamlit
plotly

# Reinforcement Learning (Layer 3 — future)
# stable-baselines3
# gymnasium
```

---

## ⚙️ Environment Variables (`.env` additions)

```env
# Agent config
AGENT_SYMBOLS=BTC-USD,EURUSD=X,GC=F
AGENT_INTERVAL_MINUTES=60
AGENT_MODE=paper              # paper | backtest | live

# Sentiment
SENTIMENT_ENGINE=vader         # vader | groq
NEWS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here     # Only needed if SENTIMENT_ENGINE=groq

# Risk thresholds
REQUIRED_VOTE_SCORE=0.55       # Weighted score threshold for BUY/SELL
MAX_DRAWDOWN_PCT=15            # Stop trading if portfolio drops > 15%
RISK_PER_TRADE_PCT=1           # Risk 1% of capital per trade

# Retraining
RETRAIN_WIN_RATE_THRESHOLD=0.45
RETRAIN_MAX_AGE_DAYS=30
```

---

## 🚀 Recommended Build Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 8   ← Working autonomous agent (no sentiment)
                                   ↓
                           Phase 6 + 7   ← Weighted voting + memory
                                   ↓
                           Phase 4 + 5   ← Plug in sentiment + fundamentals
                                   ↓
                           Phase 9       ← Auto-retraining (Layer 1)
                                         (Layer 2 activates automatically from Phase 6)
                                   ↓
                           Phase 10      ← Dashboard last
```

---

## ☑️ Master Progress Tracker

- [x] **Phase 1** — Scalable Scaffold (plugin registry + base agent)
- [x] **Phase 2** — Market Data Analyst
- [x] **Phase 3** — Quant Analyst (with hot-reload)
- [x] **Phase 4** — Sentiment Analyst (VADER default, Groq upgrade)
- [x] **Phase 5** — Fundamentals Analyst
- [x] **Phase 6** — Risk Manager (adaptive weighted voting)
- [x] **Phase 7** — Portfolio Manager + DB schema
- [x] **Phase 8** — Autonomous Orchestrator Loop + CLI
- [x] **Phase 9** — Self-Retraining (Layer 1)
- [x] **Phase 10** — Streamlit Dashboard
- [ ] **Future** — RL Quant Analyst (Layer 3, stable-baselines3)

---

## 🔮 Future Agent Slots (Drop-in Ready)

Thanks to the plugin registry, these can be added later with **zero changes to core code**:

| Agent | What it does |
|---|---|
| `agents/macro_analyst.py` | Fed rates, CPI, GDP calendar events |
| `agents/onchain_analyst.py` | Bitcoin on-chain (exchange flows, whale activity) |
| `agents/options_analyst.py` | Put/Call ratio, implied volatility signals |
| `agents/rl_quant_analyst.py` | Reinforcement Learning model (Layer 3) |
| `agents/news_event_analyst.py` | Earnings, halvings, regulatory events |
| `agents/correlation_analyst.py` | Cross-asset correlation signals (DXY, VIX, Gold) |
