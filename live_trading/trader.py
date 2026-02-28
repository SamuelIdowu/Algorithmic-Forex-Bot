import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from utils.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, MODE, TRADING_SYMBOL
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
        Get the latest price for a symbol, supporting Stocks, Crypto, and some Forex via fallbacks
        """
        # Try Stock Bars (US Equity)
        try:
            bars = self.api.get_bars(symbol, TimeFrame.Minute, limit=1)
            if bars:
                return bars[-1].c
        except Exception as e:
            logger.debug(f"get_bars failed for {symbol}: {e}")

        # Try Latest Trade (Works for Stocks and sometimes Crypto)
        try:
            if hasattr(self.api, 'get_latest_trade'):
                trade = self.api.get_latest_trade(symbol)
                # handle both object with .price and dict
                price = getattr(trade, 'price', None) or (trade.get('p') if hasattr(trade, 'get') else None)
                if price:
                    logger.info(f"Got price via get_latest_trade for {symbol}: {price}")
                    return float(price)
        except Exception as e:
            logger.debug(f"get_latest_trade failed for {symbol}: {e}")

        # Try Crypto Bars
        try:
            if hasattr(self.api, 'get_crypto_bars'):
                # Note: get_crypto_bars might return different structure depending on valid exchanges
                bars = self.api.get_crypto_bars(symbol, TimeFrame.Minute, limit=1)
                if bars:
                     # Crypto bars often have .close not .c
                     last_bar = bars[-1]
                     price = getattr(last_bar, 'close', None) or getattr(last_bar, 'c', None)
                     if price:
                         logger.info(f"Got price via get_crypto_bars for {symbol}: {price}")
                         return float(price)
        except Exception as e:
             logger.debug(f"get_crypto_bars failed for {symbol}: {e}")
             
        # Try Crypto Quotes (Last resort)
        try:
            if hasattr(self.api, 'get_latest_crypto_quote'):
                quote = self.api.get_latest_crypto_quote(symbol, exchange='CBSE') # try coinbase
                if quote:
                    price = getattr(quote, 'ask_price', None) or getattr(quote, 'ap', None)
                    if price:
                        logger.info(f"Got price via get_latest_crypto_quote for {symbol}: {price}")
                        return float(price)
        except Exception:
            pass

        logger.error(f"Error getting latest price for {symbol}: All methods failed")
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
            # Map timeframe string to TimeFrame object
            tf = TimeFrame.Day
            if timeframe == '1Min': tf = TimeFrame.Minute
            elif timeframe == '15Min': tf = TimeFrame.Minute # Approximation or need specific TimeFrame
            elif timeframe == '1H': tf = TimeFrame.Hour
            
            # Use v2 API get_bars
            # We need to specify start time to get enough data
            # For daily data, we want past 100 days to be safe
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
            
            bars = self.api.get_bars(symbol, tf, start=start_date, limit=limit).df
            
            # Alpaca v2 returns dataframe with columns: open, high, low, close, volume, trade_count, vwap
            # We need to ensure it matches what our features module expects (lowercase is fine)
            return bars
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
                # Check if it's market hours (Only for US Equities)
                should_check_market_hours = True
                try:
                    asset = self.api.get_asset(symbol)
                    # Check if asset class is 'us_equity'
                    asset_class = getattr(asset, 'class_', None) or (asset._raw.get('class') if hasattr(asset, '_raw') else None)
                    
                    if asset_class and asset_class != 'us_equity':
                        should_check_market_hours = False
                        logger.info(f"Asset class is {asset_class}, skipping market hours check.")
                except Exception as e:
                    # If we can't get the asset, fall back to symbol heuristics
                    if '/' in symbol: 
                         should_check_market_hours = False
                         logger.info(f"Symbol {symbol} appears to be Crypto/Forex, skipping market hours check.")
                    else:
                        logger.warning(f"Could not check asset class for {symbol}: {e}")

                clock = self.api.get_clock()
                if should_check_market_hours:
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
                action = strategy_func(self, symbol, current_price)

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
            return {'side': 'buy', 'reason': f'Price {current_price:.5f} below SMA {sma:.5f}'}
        elif current_price > sma * 1.01:  # 1% above MA
            return {'side': 'sell', 'reason': f'Price {current_price:.5f} above SMA {sma:.5f}'}
        else:
            return None  # No action
    except Exception as e:
        logger.error(f"Error in moving average strategy: {e}")
        return None


def ml_trading_strategy(trader, symbol, current_price):
    """
    ML-based trading strategy using the trained Random Forest model
    """
    import joblib
    import numpy as np
    from utils.features import add_technical_features
    
    # Static cache for model and scaler to avoid reloading every time
    if not hasattr(ml_trading_strategy, "model"):
        try:
            ml_trading_strategy.model = joblib.load("models/ml_strategy_model.pkl")
            ml_trading_strategy.scaler = joblib.load("models/ml_strategy_scaler.pkl")
            logger.info("ML model and scaler loaded for live trading")
            if hasattr(ml_trading_strategy.scaler, 'feature_names_in_'):
                logger.info(f"Expected features: {ml_trading_strategy.scaler.feature_names_in_.tolist()}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None

    try:
        # 1. Get historical data (need enough for features, e.g., 50 bars)
        # We use '1Day' bars if the model was trained on daily data, or '15Min'/'1Min' if intraday
        # Assuming the model is trained on Daily data for now based on train_model.py defaults
        # But for live testing we might want shorter timeframe? 
        # Let's stick to what the user likely trained on. If they trained on Daily, we need Daily.
        # However, for "paper trading demo" usually people want to see action. 
        # Let's assume Daily for correctness with current codebase.
        hist_data = trader.get_historical_data(symbol, '1Day', limit=60)
        
        if len(hist_data) < 50:
            logger.warning(f"Not enough historical data for {symbol}: {len(hist_data)}")
            return None

        # Filter to only OHLCV columns to match training data (Yahoo Finance style)
        # Alpaca returns extra columns like 'trade_count', 'vwap' which cause feature mismatch
        hist_data = hist_data[['open', 'high', 'low', 'close', 'volume']]

        # 2. Calculate features
        # The features function expects a DataFrame with open, high, low, close, volume
        # Alpaca returns columns: open, high, low, close, volume (lowercase? need to check)
        # Alpaca SDK usually returns lowercase columns in .df property
        
        df_with_features = add_technical_features(hist_data)
        
        # 3. Prepare latest feature vector
        # Take the last row
        current_features = df_with_features.iloc[-1:].dropna(axis=1, how='all')
        
        # Filter columns to match model input
        # feature_cols = [col for col in current_features.columns 
        #                if col not in ['target', 'future_close'] and np.issubdtype(current_features[col].dtype, np.number)]
        
        # current_features_filtered = current_features[feature_cols]
        
        # Explicitly reorder columns to match scaler's expectation
        if hasattr(ml_trading_strategy.scaler, 'feature_names_in_'):
             expected_features = ml_trading_strategy.scaler.feature_names_in_
             # Check if we have all features
             missing_features = [f for f in expected_features if f not in current_features.columns]
             if missing_features:
                 logger.error(f"Missing features: {missing_features}")
                 return None
             
             current_features_filtered = current_features[expected_features]
        else:
             # Fallback for older sklearn versions or scalers without feature_names_in_
             feature_cols = [col for col in current_features.columns 
                        if col not in ['target', 'future_close'] and np.issubdtype(current_features[col].dtype, np.number)]
             current_features_filtered = current_features[feature_cols]

        # Check feature count
        
        # Check feature count
        if len(current_features_filtered.columns) != ml_trading_strategy.scaler.n_features_in_:
            logger.warning(f"Feature mismatch: expected {ml_trading_strategy.scaler.n_features_in_}, got {len(current_features_filtered.columns)}")
            return None
            
        # 4. Predict
        features_scaled = ml_trading_strategy.scaler.transform(current_features_filtered)
        prediction = ml_trading_strategy.model.predict(features_scaled)[0]
        prediction_proba = ml_trading_strategy.model.predict_proba(features_scaled)[0]
        confidence = max(prediction_proba)
        
        logger.info(f"Prediction for {symbol}: {prediction} (Confidence: {confidence:.2f})")
        
        # 5. Generate Signal
        if prediction == 1 and confidence > 0.6:
            return {'side': 'buy', 'reason': f'ML Buy Signal (Conf: {confidence:.2f})'}
        elif prediction == 0 and confidence > 0.6:
            return {'side': 'sell', 'reason': f'ML Sell Signal (Conf: {confidence:.2f})'}
            
        return None
        
    except Exception as e:
        logger.error(f"Error in ML strategy: {e}")
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
    
    logger.info(f"Starting ML Trading Strategy for {TRADING_SYMBOL}...")
    # Run the trading loop with the ML strategy
    trader.run_trading_strategy(
        symbol=TRADING_SYMBOL,
        strategy_func=ml_trading_strategy,
        qty=1,
        max_orders_per_day=5
    )


if __name__ == "__main__":
    # Example usage
    if MODE in ['paper', 'live']:
        run_paper_trade_example()
    else:
        logger.info(f"Current mode is {MODE}, skipping live trading example")