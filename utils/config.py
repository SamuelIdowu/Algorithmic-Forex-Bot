import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Trading mode
MODE = os.getenv("MODE", "backtest")
STRATEGY = os.getenv("STRATEGY", "moving_average")
TRADING_SYMBOL = os.getenv("TRADING_SYMBOL", "AAPL")

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Finnhub API key
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Default settings
INITIAL_CAPITAL = 10000
COMMISSION_PER_TRADE = 0.0  # Set to 0 for no commission

# Strategy Settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
RISK_REWARD_SL_MULT = float(os.getenv("RR_SL_MULT", "2.0"))
RISK_REWARD_TP_MULT = float(os.getenv("RR_TP_MULT", "3.0"))
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
LOOKBACK_PERIOD = int(os.getenv("LOOKBACK_PERIOD", "100"))

# Forex settings
FOREX_TRADING_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]

# ─── AI Hedge Fund Agent Config ───────────────────────────────────────────────

# Comma-separated list of symbols the agent loop trades
_raw_symbols = os.getenv("AGENT_SYMBOLS", "BTC-USD,EURUSD=X,GC=F")
AGENT_SYMBOLS = [s.strip() for s in _raw_symbols.split(",") if s.strip()]

AGENT_INTERVAL_MINUTES = int(os.getenv("AGENT_INTERVAL_MINUTES", "60"))
AGENT_MODE = os.getenv("AGENT_MODE", "backtest")   # backtest | paper | live

# Initial capital used by the portfolio manager in backtest/paper mode
AGENT_INITIAL_CAPITAL = float(os.getenv("AGENT_INITIAL_CAPITAL", "10000"))

# ─── Sentiment ────────────────────────────────────────────────────────────────
SENTIMENT_ENGINE = os.getenv("SENTIMENT_ENGINE", "vader")  # vader | groq
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_70B = os.getenv("GROQ_MODEL_70B", "llama-3.3-70b-versatile")
GROQ_MODEL_8B = os.getenv("GROQ_MODEL_8B", "llama-3.1-8b-instant")

# ─── Risk Thresholds ──────────────────────────────────────────────────────────
# Weighted vote score must exceed this to trigger BUY (and drop below 1-threshold for SELL)
REQUIRED_VOTE_SCORE = float(os.getenv("REQUIRED_VOTE_SCORE", "0.55"))
# Stop all new entries if total portfolio drawdown exceeds this %
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
    """Validate that required configuration values are present based on the mode"""
    if MODE in ["live", "paper"]:
        required_keys = ["ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"]
        for key in required_keys:
            if not os.getenv(key):
                raise ValueError(f"Missing required environment variable: {key}")
    
    if MODE == "backtest":
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("Missing required environment variable: ALPHA_VANTAGE_API_KEY")
    
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