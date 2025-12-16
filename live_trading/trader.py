import alpaca_trade_api as tradeapi
from utils.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, MODE
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlpacaTrader:
    """
    A class to handle live trading operations using the Alpaca API
    """
    def __init__(self):
        # Validate that we're in live/paper mode
        if MODE == "backtest":
            raise Exception("Cannot trade in backtest mode. Set MODE=live or MODE=paper in .env")

        # Initialize the Alpaca API
        try:
            self.api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_API_SECRET,
                base_url=ALPACA_BASE_URL
            )
            logger.info("Alpaca API connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_account_info(self):
        """
        Get current account information
        """
        try:
            account = self.api.get_account()
            return account
        except Exception as e:
            logger.error(f"Error retrieving account info: {e}")
            return None

    def get_positions(self):
        """
        Get current positions
        """
        try:
            positions = self.api.list_positions()
            return positions
        except Exception as e:
            logger.error(f"Error retrieving positions: {e}")
            return []

    def place_order(self, symbol, side, qty, order_type='market', time_in_force='gtc'):
        """
        Place an order on Alpaca
        
        Args:
            symbol (str): The stock symbol
            side (str): 'buy' or 'sell'
            qty (int): Quantity to buy or sell
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            time_in_force (str): 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            logger.info(f"Order submitted: {order.side} {order.qty} of {order.symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def get_latest_price(self, symbol):
        """
        Get the latest price for a symbol
        """
        try:
            barset = self.api.get_barset(symbol, 'minute', limit=1)
            bar = barset[symbol][0]
            return bar.c  # Close price
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def cancel_order(self, order_id):
        """
        Cancel an order by its ID
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id):
        """
        Get the status of an order by its ID
        """
        try:
            order = self.api.get_order(order_id)
            return order
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    def get_historical_data(self, symbol, timeframe='1D', limit=1000):
        """
        Get historical data for a symbol
        """
        try:
            barset = self.api.get_barset(symbol, timeframe, limit=limit)
            return barset[symbol].df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def run_trading_strategy(self, symbol, strategy_func, qty=1, max_orders_per_day=5):
        """
        Run a trading strategy
        """
        orders_today = 0
        last_day = None

        while True:
            try:
                # Check if it's market hours
                clock = self.api.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed. Waiting...")
                    # Wait until market opens
                    time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                    time.sleep(time_to_open)
                    continue

                # Check if we've started a new trading day
                current_day = clock.timestamp.date()
                if current_day != last_day:
                    last_day = current_day
                    orders_today = 0

                # Get current data
                current_price = self.get_latest_price(symbol)
                if current_price is None:
                    logger.warning(f"Could not get current price for {symbol}, skipping...")
                    time.sleep(60)  # Wait 1 minute before trying again
                    continue

                # Execute strategy function
                action = strategy_func(symbol, current_price)

                # Place order if strategy signals one
                if action and orders_today < max_orders_per_day:
                    side = action.get('side')
                    if side in ['buy', 'sell']:
                        order = self.place_order(symbol, side, qty)
                        if order:
                            orders_today += 1
                            logger.info(f"Placed {side} order for {qty} shares of {symbol}")

                # Wait before next check (e.g., check every 5 minutes)
                time.sleep(300)  # 5 minutes

            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before continuing


def simple_moving_average_strategy(symbol, current_price):
    """
    A simple example strategy that compares current price to a moving average
    (This is a very basic example - real strategies should be much more sophisticated)
    """
    try:
        # Get historical data for moving average calculation
        trader = AlpacaTrader()
        hist_data = trader.get_historical_data(symbol, '1Min', 20)  # 20 min MA
        
        if len(hist_data) < 20:
            return None  # Not enough data yet
            
        # Calculate simple moving average
        sma = hist_data['close'].mean()
        
        # Simple strategy: buy if price is below MA by 1%, sell if above by 1%
        if current_price < sma * 0.99:  # 1% below MA
            return {'side': 'buy', 'reason': f'Price {current_price} below SMA {sma:.2f}'}
        elif current_price > sma * 1.01:  # 1% above MA
            return {'side': 'sell', 'reason': f'Price {current_price} above SMA {sma:.2f}'}
        else:
            return None  # No action
    except Exception as e:
        logger.error(f"Error in moving average strategy: {e}")
        return None


def run_paper_trade_example():
    """
    Example of running a paper trade
    """
    if MODE != 'paper' and MODE != 'live':
        logger.error("This function requires MODE to be 'paper' or 'live'")
        return

    trader = AlpacaTrader()
    
    # Display account info
    account = trader.get_account_info()
    if account:
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Buying Power: {account.buying_power}")
        logger.info(f"Portfolio Value: {account.portfolio_value}")
    
    # Example: Place a paper trade
    # trader.place_order('AAPL', 'buy', 1)
    
    # Example: Get positions
    positions = trader.get_positions()
    for position in positions:
        logger.info(f"Position: {position.symbol}, Qty: {position.qty}, Value: {position.market_value}")


if __name__ == "__main__":
    # Example usage
    if MODE in ['paper', 'live']:
        run_paper_trade_example()
    else:
        logger.info(f"Current mode is {MODE}, skipping live trading example")