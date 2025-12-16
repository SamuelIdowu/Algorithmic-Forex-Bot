import yfinance as yf
import pandas as pd

def test_ticker(symbol):
    print(f"Testing {symbol}...")
    try:
        # Try to fetch 1 day of data
        data = yf.download(symbol, period="1d", interval="1h")
        if data.empty:
            print(f"FAIL: No data found for {symbol}")
        else:
            print(f"SUCCESS: Fetched {len(data)} rows for {symbol}")
            print(data.head())
    except Exception as e:
        print(f"ERROR: {e}")

print("Checking yfinance version:", yf.__version__)
test_ticker("euro usd")
test_ticker("EURUSD=X")
