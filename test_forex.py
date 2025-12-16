import sys
import os

# Add the project root to the python path
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import get_forex_data
from utils.config import FOREX_TRADING_PAIRS

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
        
        # Visualize results
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['close'], label='Close Price')
            plt.title(f'{pair} Forex Data')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            output_file = 'forex_chart.png'
            plt.savefig(output_file)
            print(f"Chart saved to {output_file}")
        except ImportError:
            print("Matplotlib not installed, skipping visualization.")
        except Exception as e:
            print(f"Error creating chart: {e}")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    test_forex_fetching()
