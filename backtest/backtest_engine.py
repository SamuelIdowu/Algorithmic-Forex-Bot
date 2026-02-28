import backtrader as bt
import pandas as pd
import yfinance as yf
from strategies.moving_average import MovingAverageStrategy, RSIStrategy
import matplotlib.pyplot as plt


def run_backtest(symbol="AAPL", strategy_class=MovingAverageStrategy, start=None, end=None, initial_cash=10000, asset_type="stock", interval="1d", **kwargs):
    """
    Run a backtest with the specified strategy and data
    """
    # Handle dynamic dates
    if end is None:
        end = pd.Timestamp.now()
    else:
        end = pd.to_datetime(end)
        
    if start is None:
        start = end - pd.Timedelta(days=365)
    else:
        start = pd.to_datetime(start)

    # Calculate a buffer period to ensure we have enough data for indicators (e.g. 50-day SMA)
    # We'll fetch 100 days prior to the start date
    buffer_days = 100
    fetch_start_date = start - pd.Timedelta(days=buffer_days)
    print(f"\n{'='*60}")
    print(f"BACKTEST: {symbol} | Interval: {interval}")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"{'='*60}")
    print(f"Fetching data from {fetch_start_date.date()} (100 days buffer) to ensure indicators can be calculated...")

    # Map symbols if needed
    if symbol == "XAU/USD":
        symbol = "GC=F"

    # Get the data
    try:
        # If interval is intraday, we must use yfinance as our local forex loader is daily only
        if interval != "1d" or asset_type == "stock":
            data = yf.download(symbol, start=fetch_start_date, end=end, interval=interval, auto_adjust=True)
        elif asset_type == "forex":
            from utils.data_loader import get_forex_data
            # Split symbol if needed, e.g., "EUR/USD" -> "EUR", "USD"
            if '/' in symbol:
                from_curr, to_curr = symbol.split('/')
            else:
                # Assume standard 6 char format like EURUSD
                from_curr, to_curr = symbol[:3], symbol[3:]
            
            data = get_forex_data(from_curr, to_curr, outputsize='full')
            # Filter by date
            if not data.empty:
                data = data[(data.index >= fetch_start_date) & (data.index <= end)]
                # Add dummy volume for backtrader if missing
                if 'volume' not in data.columns:
                    data['volume'] = 0
        else:
            print(f"Unsupported asset type: {asset_type}")
            return

        if data.empty:
            print(f"No data found for symbol {symbol} in the specified date range.")
            return
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return

    # Ensure the data is in the right format for backtrader
    # yfinance should return a pandas DataFrame, but let's make sure
    if not isinstance(data, pd.DataFrame):
        print(f"Data provider returned unexpected type: {type(data)}")
        return

    # Ensure the data is in the right format for backtrader
    # Convert column names to lowercase to match backtrader expectations
    # Check if column names contain tuples (this can happen with multi-index columns)
    if isinstance(data.columns, pd.MultiIndex):
        # If it's a MultiIndex, flatten it
        data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    else:
        # If it's a regular Index, just convert to lowercase
        data.columns = [col.lower() for col in data.columns]

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(strategy_class, **kwargs)

    # Create a data feed from the downloaded data
    # Note: We feed the buffered data so indicators can calculate.
    # The strategy will run over the buffer period too.
    data_feed = bt.feeds.PandasData(
        dataname=data,
        fromdate=fetch_start_date,
        todate=pd.to_datetime(end)
    )
    
    # Add the Data Feed to Cerebro
    cerebro.adddata(data_feed)

    # Set our desired cash start
    cerebro.broker.setcash(initial_cash)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print(f"\nStarting Portfolio Value: ${cerebro.broker.getvalue():.2f}")

    # Run over everything
    try:
        strat = cerebro.run()
        
        # Print out the final result
        final_value = cerebro.broker.getvalue()
        profit_loss = final_value - initial_cash
        profit_loss_pct = (profit_loss / initial_cash) * 100
        print(f"\nFinal Portfolio Value: ${final_value:.2f}")
        print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
        print(f"{'='*60}\n")
    
        # Plot the result if in an environment that supports it
        try:
            # Create a custom title for the plot
            plot_title = f"{symbol} - {interval} | {start.date()} to {end.date()}"
            cerebro.plot(style='candlestick', iplot=False)
        except Exception as plot_error:
            print(f"Plotting not available (this is normal in headless environments): {plot_error}")
    except Exception as e:
        print(f"Error during backtesting: {e}")


def compare_strategies(symbol="AAPL", start="2022-01-01", end="2023-01-01"):
    """
    Compare different strategies using yfinance data
    """
    # Get the data using yfinance
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            print(f"No data found for symbol {symbol} in the specified date range.")
            return
    except Exception as e:
        print(f"Error fetching data for {symbol} from yfinance: {e}")
        return

    # Create multiple cerebro entities for different strategies
    strategies = {
        "Moving Average": MovingAverageStrategy,
        "RSI Strategy": RSIStrategy
    }
    
    results = {}
    
    for strat_name, strat_class in strategies.items():
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strat_class)
        
        # Create a data feed
        data_feed = bt.feeds.PandasData(
            dataname=data,
            fromdate=pd.to_datetime(start),
            todate=pd.to_datetime(end)
        )
        
        # Add the Data Feed to Cerebro
        cerebro.adddata(data_feed)

        # Set our desired cash start
        cerebro.broker.setcash(10000)

        # Add a FixedSize sizer according to the stake
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        print(f'Running {strat_name} strategy...')
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
        
        # Run over everything
        strat = cerebro.run()
        
        final_value = cerebro.broker.getvalue()
        results[strat_name] = final_value
        
        print(f'{strat_name} Final Portfolio Value: {final_value:.2f}')
        print("---")

    return results


def compare_strategies(symbol="AAPL", start="2022-01-01", end="2023-01-01"):
    """
    Compare different strategies
    """
    # Initialize the data loader
    loader = AlphaVantageDataLoader(ALPHA_VANTAGE_API_KEY)
    
    # Get the data
    data = loader.get_daily_data(symbol)
    
    # Filter data based on the date range
    data = data[(data.index >= start) & (data.index <= end)]
    
    if data.empty:
        print(f"No data found for symbol {symbol} in the specified date range.")
        return

    # Create multiple cerebro entities for different strategies
    strategies = {
        "Moving Average": MovingAverageStrategy,
        "RSI Strategy": RSIStrategy
    }
    
    results = {}
    
    for strat_name, strat_class in strategies.items():
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strat_class)
        
        # Create a data feed
        data_feed = bt.feeds.PandasData(
            dataname=data,
            fromdate=pd.to_datetime(start),
            todate=pd.to_datetime(end)
        )
        
        # Add the Data Feed to Cerebro
        cerebro.adddata(data_feed)

        # Set our desired cash start
        cerebro.broker.setcash(10000)

        # Add a FixedSize sizer according to the stake
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        print(f'Running {strat_name} strategy...')
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
        
        # Run over everything
        strat = cerebro.run()
        
        final_value = cerebro.broker.getvalue()
        results[strat_name] = final_value
        
        print(f'{strat_name} Final Portfolio Value: {final_value:.2f}')
        print("---")

    return results


if __name__ == "__main__":
    # Example usage
    # Run single strategy
    run_backtest(symbol="AAPL", strategy_class=MovingAverageStrategy, start="2022-01-01", end="2023-01-01")
    
    # Or compare multiple strategies
    # compare_strategies(symbol="AAPL", start="2022-01-01", end="2023-01-01")