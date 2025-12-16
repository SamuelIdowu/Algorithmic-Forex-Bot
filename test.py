import sys
import os

# Add the project root to the python path
# We are in algo_trader directory, so we need to go up one level to import algo_trader package
# Actually, if we are in algo_trader, we might need to adjust imports.
# But let's try to append the parent directory to sys.path
sys.path.append(os.path.dirname(os.getcwd()))

from algo_trader.utils.data_loader import get_forex_data
from algo_trader.utils.config import FOREX_TRADING_PAIRS

def test_forex_fetching():
    print("Testing Forex Data Fetching...")
    
    # Test with the first pair from config
    pair = FOREX_TRADING_PAIRS[0]
    from_currency, to_currency = pair.split('/')
    
    print(f"Fetching data for {pair}...")
    data = get_forex_data(from_currency, to_currency, outputsize='compact')
    
    if not data.empty:
        print("Success! Data received:")
        print(data.head())
        print(f"Shape: {data.shape}")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    test_forex_fetching()
