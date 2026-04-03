import yfinance as yf
import pandas as pd
from datetime import datetime

symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "EURUSD=X"]

print(f"Fetching data for {symbols}...")
data = yf.download(
    tickers=symbols,
    period="2d",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    progress=False
)

movers = []
for sym in symbols:
    try:
        if len(symbols) > 1:
            ticker_data = data[sym]
        else:
            ticker_data = data
        
        if ticker_data.empty or len(ticker_data) < 2:
            print(f"No data for {sym}")
            continue
        
        # yf.download returns TitleCase columns by default
        current_price = ticker_data['Close'].iloc[-1]
        prev_price = ticker_data['Close'].iloc[-2]
        
        if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
            print(f"NaN for {sym}")
            continue
            
        change_pct = ((current_price / prev_price) - 1) * 100
        movers.append({"symbol": sym, "change": float(change_pct)})
    except Exception as e:
        print(f"Error for {sym}: {e}")

print("\nTop Movers Results:")
for mover in movers:
    print(f"{mover['symbol']}: {mover['change']:.2f}%")
