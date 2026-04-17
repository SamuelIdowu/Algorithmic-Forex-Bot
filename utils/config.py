import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Master Symbol Lists ──────────────────────────────────────────────────────
# Grouped choices shown in the autonomous agent multi-select picker
AGENT_PAIR_GROUPS = [
    ("── Forex (Major)", [
        ("EUR/USD  (EURUSD=X)",  "EURUSD=X"),
        ("GBP/USD  (GBPUSD=X)",  "GBPUSD=X"),
        ("USD/JPY  (USDJPY=X)",  "USDJPY=X"),
        ("AUD/USD  (AUDUSD=X)",  "AUDUSD=X"),
        ("USD/CAD  (USDCAD=X)",  "USDCAD=X"),
        ("NZD/USD  (NZDUSD=X)",  "NZDUSD=X"),
        ("USD/CHF  (USDCHF=X)",  "USDCHF=X"),
    ]),
    ("── Forex (Cross)", [
        ("EUR/GBP  (EURGBP=X)",  "EURGBP=X"),
        ("EUR/JPY  (EURJPY=X)",  "EURJPY=X"),
        ("GBP/JPY  (GBPJPY=X)",  "GBPJPY=X"),
        ("AUD/JPY  (AUDJPY=X)",  "AUDJPY=X"),
        ("EUR/AUD  (EURAUD=X)",  "EURAUD=X"),
        ("GBP/CAD  (GBPCAD=X)",  "GBPCAD=X"),
    ]),
    ("── Crypto (Large Cap)", [
        ("BTC/USD  (BTC-USD)",   "BTC-USD"),
        ("ETH/USD  (ETH-USD)",   "ETH-USD"),
        ("BNB/USD  (BNB-USD)",   "BNB-USD"),
        ("SOL/USD  (SOL-USD)",   "SOL-USD"),
        ("XRP/USD  (XRP-USD)",   "XRP-USD"),
    ]),
    ("── Crypto (Mid Cap)", [
        ("ADA/USD  (ADA-USD)",   "ADA-USD"),
        ("DOGE/USD (DOGE-USD)",  "DOGE-USD"),
        ("AVAX/USD (AVAX-USD)",  "AVAX-USD"),
        ("LINK/USD (LINK-USD)",  "LINK-USD"),
        ("DOT/USD  (DOT-USD)",   "DOT-USD"),
        ("MATIC/USD(MATIC-USD)", "MATIC-USD"),
        ("LTC/USD  (LTC-USD)",   "LTC-USD"),
    ]),
    ("── Commodities", [
        ("Gold     (GC=F)",      "GC=F"),
        ("Silver   (SI=F)",      "SI=F"),
        ("Crude Oil(CL=F)",      "CL=F"),
        ("Copper   (HG=F)",      "HG=F"),
        ("Nat Gas  (NG=F)",      "NG=F"),
        ("Wheat    (ZW=F)",      "ZW=F"),
        ("Corn     (ZC=F)",      "ZC=F"),
    ]),
    ("── Indices / ETFs", [
        ("S&P 500  (SPY)",       "SPY"),
        ("NASDAQ   (QQQ)",       "QQQ"),
        ("Dow Jones(DIA)",       "DIA"),
        ("Russell  (IWM)",       "IWM"),
        ("VIX      (^VIX)",      "^VIX"),
    ]),
    ("── US Tech / Growth Stocks", [
        ("Apple    (AAPL)",      "AAPL"),
        ("Microsoft(MSFT)",      "MSFT"),
        ("Alphabet (GOOGL)",     "GOOGL"),
        ("Amazon   (AMZN)",      "AMZN"),
        ("NVIDIA   (NVDA)",      "NVDA"),
        ("Tesla    (TSLA)",      "TSLA"),
        ("Meta     (META)",      "META"),
        ("Netflix  (NFLX)",      "NFLX"),
        ("AMD      (AMD)",       "AMD"),
        ("Intel    (INTC)",      "INTC"),
    ]),
]

# Flat list of all tickers available in AGENT_PAIR_GROUPS
ALL_SUPPORTED_SYMBOLS = []
for _, pairs in AGENT_PAIR_GROUPS:
    for _, ticker in pairs:
        ALL_SUPPORTED_SYMBOLS.append(ticker)


# ─── Legacy Backtest Settings (deprecated — kept for backward compat) ─────────
# These are no longer used by the insights engine but may exist in old .env files
# MODE, STRATEGY, TRADING_SYMBOL, ALPACA_*

# API keys still used by sentiment/data agents
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ─── AI Hedge Fund Agent Config ───────────────────────────────────────────────

# Comma-separated list of symbols the agent loop analyzes
# If set to 'ALL', use ALL_SUPPORTED_SYMBOLS from the master list
AGENT_SYMBOL_DEFAULTS = "BTC-USD,EURUSD=X,GC=F"
_raw_symbols = os.getenv("AGENT_SYMBOLS", AGENT_SYMBOL_DEFAULTS)
if _raw_symbols.upper() == "ALL":
    AGENT_SYMBOLS = ALL_SUPPORTED_SYMBOLS
else:
    AGENT_SYMBOLS = [s.strip() for s in _raw_symbols.split(",") if s.strip()]

# ──────────────────────────────────────────────────────────────────────────────

AGENT_INTERVAL_MINUTES = int(os.getenv("AGENT_INTERVAL_MINUTES", "60"))

# ─── Sentiment ────────────────────────────────────────────────────────────────
SENTIMENT_ENGINE = os.getenv("SENTIMENT_ENGINE", "vader")  # vader | groq
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_70B = os.getenv("GROQ_MODEL_70B", "llama-3.3-70b-versatile")
GROQ_MODEL_8B = os.getenv("GROQ_MODEL_8B", "llama-3.1-8b-instant")

# ─── Risk Thresholds ──────────────────────────────────────────────────────────
# Weighted vote score must exceed this to trigger BUY (and drop below 1-threshold for SELL)
REQUIRED_VOTE_SCORE = float(os.getenv("REQUIRED_VOTE_SCORE", "0.55"))
# Maximum drawdown threshold before halting signals (risk management)
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "15"))
# Fraction of capital risked per trade (for ATR-based position sizing)
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1")) / 100.0
# ATR multipliers for stop-loss and take-profit
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "2.0"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "3.0"))

# ─── Retraining ───────────────────────────────────────────────────────────────
RETRAIN_WIN_RATE_THRESHOLD = float(os.getenv("RETRAIN_WIN_RATE_THRESHOLD", "0.45"))
RETRAIN_MAX_AGE_DAYS = int(os.getenv("RETRAIN_MAX_AGE_DAYS", "30"))
RETRAIN_BOOTSTRAP = os.getenv("RETRAIN_BOOTSTRAP", "True").lower() == "true"

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def validate_config():
    """Validate that required configuration values are present."""
    if TELEGRAM_ENABLED:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_ENABLED is True but token or chat_id is missing.")
    return True

def write_config(updates: dict[str, str]) -> None:
    """Write/update key=value pairs in the .env file and update module globals."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    import re
    written = set()
    new_lines = []
    for line in lines:
        matched = False
        for key, val in updates.items():
            if re.match(rf"^{key}\s*=", line.strip()):
                comment_match = re.search(r"(\s*#.*)$", line)
                comment = comment_match.group(1) if comment_match else ""
                new_lines.append(f"{key}={val}{comment}\n")
                written.add(key)
                matched = True
                
                # Update global variable in this module
                if key in globals():
                    # Attempt to cast to existing type
                    orig_val = globals()[key]
                    try:
                        if isinstance(orig_val, bool):
                            globals()[key] = str(val).lower() == "true"
                        elif isinstance(orig_val, int):
                            globals()[key] = int(float(val))
                        elif isinstance(orig_val, float):
                            globals()[key] = float(val)
                        elif isinstance(orig_val, list):
                            globals()[key] = [s.strip() for s in str(val).split(",") if s.strip()]
                        else:
                            globals()[key] = str(val)
                    except:
                        globals()[key] = str(val)
                break
        if not matched:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}={val}\n")
            # If it's a new key, just add as string for now
            globals()[key] = str(val)

    with open(env_path, "w") as f:
        f.writelines(new_lines)
    return True

def format_price(price: float) -> str:
    """
    Robust price formatter that adjusts precision based on magnitude.
    Ensures enough significant figures for both high-priced (BTC) and 
    low-priced (Forex, SHIB) assets.
    """
    if price is None:
        return "—"
    
    if price == 0:
        return "0.00"

    # Handle various price magnitudes with appropriate precision
    if price < 0.0001:
        return f"{price:.8f}"
    elif price < 0.01:
        return f"{price:.6f}"
    elif price < 1.0:
        return f"{price:.5f}"
    elif price < 100:
        # Most Forex (except JPY) and low stocks
        return f"{price:.5f}"
    elif price < 10000:
        # Stocks, JPY pairs, Gold
        return f"{price:.3f}"
    else:
        # High priced assets (BTC, Indices)
        return f"{price:,.2f}"