import sys
import os
import asyncio
import pandas as pd

# Add the parent directory to sys.path to import algo_trader modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web.server import get_symbol_details

async def test_details_api(symbol="BTC-USD"):
    print(f"\n--- Testing details API for {symbol} ---")
    try:
        result = await get_symbol_details(symbol)
        
        print(f"Symbol: {result['symbol']}")
        print(f"Current Price: {result['current_price']}")
        print(f"History points: {len(result['history'])}")
        print(f"Signal History points: {len(result.get('signal_history', []))}")
        print(f"Market Bias: {result.get('bias', 0.0):.2f}")
        
        if result['consensus']:
            c = result['consensus']
            print(f"Latest Consensus: {c['action']} ({c['confidence']:.2f})")
            print(f"Breakdown: Q:{c['quant']} S:{c['sentiment']} F:{c['fundamentals']}")
        else:
            print("No consensus data found.")
        
        if result['trades']:
            t = result['trades'][0]
            print(f"Latest Trade: {t['action']} at {t['price']}")
            print(f"Trade Setup: {t.get('setup')}")
        else:
            print("No recent trades found.")
            
        print(f"Indicators: {result['indicators']}")
        
        # Validation checks
        assert "signal_history" in result, "Missing signal_history"
        assert "bias" in result, "Missing bias"
        if result['trades']:
            assert "setup" in result['trades'][0], "Missing setup triggers in trade"
            
        print("\n✅ API Validation Successful")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_details_api("BTC-USD"))
    asyncio.run(test_details_api("AAPL"))
