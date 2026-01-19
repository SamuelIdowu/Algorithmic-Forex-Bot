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

# Forex settings
FOREX_TRADING_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]

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
    
    return True