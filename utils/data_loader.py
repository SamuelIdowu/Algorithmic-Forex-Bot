import pandas as pd
import yfinance as yf
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.foreignexchange import ForeignExchange
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    TimeSeries = None
    ForeignExchange = None

from utils.config import ALPHA_VANTAGE_API_KEY
import ccxt
import os
from data.db_manager import DatabaseManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProvider:
    """
    A class to provide market data from multiple sources
    """

    def __init__(self):
        if ALPHA_VANTAGE_AVAILABLE:
            self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            self.forex = ForeignExchange(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        else:
            self.alpha_vantage = None
            self.forex = None
            logger.warning("Alpha Vantage library not available. Some features may be disabled.")
            
        self.db_manager = DatabaseManager()
    
    def get_alpha_vantage_data(self, symbol, start_date=None, end_date=None, function='TIME_SERIES_DAILY', outputsize='full'):
        """
        Get data from Alpha Vantage with database caching

        Args:
            symbol (str): The stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            function (str): The API function to use (e.g., TIME_SERIES_DAILY, TIME_SERIES_INTRADAY)
            outputsize (str): 'compact' for 100 records or 'full' for maximum
        """
        # Check if data exists in database first
        if self.db_manager.symbol_exists(symbol, start_date, end_date):
            logger.info(f"Loading {symbol} data from database")
            data = self.db_manager.load_data(symbol, start_date, end_date)
            return data
        else:
            logger.info(f"Data for {symbol} not in database, fetching from Alpha Vantage")

        try:
            if function == 'TIME_SERIES_DAILY':
                data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize=outputsize)
            elif function == 'TIME_SERIES_INTRADAY':
                data, meta_data = self.alpha_vantage.get_intraday(symbol=symbol, interval='5min', outputsize=outputsize)
            else:
                raise ValueError(f"Unsupported function: {function}")

            # Rename columns to standard format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            data.rename(columns=column_mapping, inplace=True)

            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Sort by date to ensure chronological order
            data.sort_index(inplace=True)

            # Save to database
            try:
                self.db_manager.save_data(data, symbol)
                logger.info(f"Saved {symbol} data to database")
            except Exception as db_error:
                logger.error(f"Failed to save {symbol} data to database: {db_error}")

            # Filter data based on start and end dates if specified
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]

            return data

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_forex_data(self, from_currency, to_currency, start_date=None, end_date=None, outputsize='full'):
        """
        Get Forex data from Alpha Vantage with database caching
        """
        symbol = f"{from_currency}/{to_currency}"

        # Check if data exists in database first
        if self.db_manager.symbol_exists(symbol, start_date, end_date):
            logger.info(f"Loading {symbol} forex data from database")
            data = self.db_manager.load_data(symbol, start_date, end_date)
            return data
        else:
            logger.info(f"Data for {symbol} not in database, fetching from Alpha Vantage")

        try:
            data, meta_data = self.forex.get_currency_exchange_daily(
                from_symbol=from_currency,
                to_symbol=to_currency,
                outputsize=outputsize
            )

            # Rename columns to standard format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            }
            data.rename(columns=column_mapping, inplace=True)

            # Add a volume column with default value since forex data doesn't have volume
            data['volume'] = 0

            # Convert index to datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Sort by date
            data.sort_index(inplace=True)

            # Save to database
            try:
                self.db_manager.save_data(data, symbol)
                logger.info(f"Saved {symbol} forex data to database")
            except Exception as db_error:
                logger.error(f"Failed to save {symbol} forex data to database: {db_error}")

            # Filter data based on start and end dates if specified
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]

            return data

        except Exception as e:
            logger.error(f"Error fetching forex data for {from_currency}/{to_currency}: {e}")
            return pd.DataFrame()

    def get_yfinance_data(self, symbol, start_date, end_date):
        """
        Get data from Yahoo Finance with database caching
        """
        # Check if data exists in database first
        # Check if data exists in database first
        # We also need to check if the data is up-to-date if an end_date is specified
        data_exists = self.db_manager.symbol_exists(symbol, start_date)
        latest_timestamp = self.db_manager.get_latest_timestamp(symbol)
        earliest_timestamp = self.db_manager.get_earliest_timestamp(symbol)
        
        need_fetch = True
        
        if data_exists and latest_timestamp and earliest_timestamp:
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                start_date_dt = pd.to_datetime(start_date)
                # If we have data covering the requested range
                if latest_timestamp >= end_date_dt and earliest_timestamp <= start_date_dt:
                    need_fetch = False
                else:
                    logger.info(f"Data in DB ({earliest_timestamp} to {latest_timestamp}) does not cover requested range ({start_date} to {end_date}). Fetching new data.")
            else:
                # If no end_date specified, we assume we want up to now? 
                # Or just use what we have? 
                # Existing logic was: need_fetch = False. Let's keep it but maybe check start_date?
                if start_date:
                    start_date_dt = pd.to_datetime(start_date)
                    if earliest_timestamp <= start_date_dt:
                        need_fetch = False
                    else:
                        logger.info(f"Data in DB starts at {earliest_timestamp}, requested start {start_date}. Fetching new data.")
                else:
                    need_fetch = False
        
        if not need_fetch:
            logger.info(f"Loading {symbol} data from database")
            data = self.db_manager.load_data(symbol, start_date, end_date)
            return data
        else:
            logger.info(f"Data for {symbol} not in database or outdated, fetching from Yahoo Finance")

        import time
        max_retries = 3
        retry_delay = 2
        
        # Sanitize symbol for Yahoo Finance (e.g. EUR/USD -> EURUSD=X)
        yf_symbol = symbol
        if '/' in symbol:
            yf_symbol = symbol.replace('/', '') + '=X'
            logger.info(f"Converted symbol {symbol} to {yf_symbol} for Yahoo Finance")
        
        for attempt in range(max_retries):
            try:
                # Add auto_adjust=True to silence FutureWarning and get adjusted data
                data = yf.download(yf_symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

                if data.empty:
                    logger.warning(f"No data found for {symbol} from Yahoo Finance")
                    return pd.DataFrame()

                # Ensure the data is in the right format for our database
                if isinstance(data.columns, pd.MultiIndex):
                    # If it's a MultiIndex, flatten it
                    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
                else:
                    # If it's a regular Index, just convert to lowercase
                    data.columns = [col.lower() for col in data.columns]

                # Save to database
                try:
                    self.db_manager.save_data(data, symbol)
                    logger.info(f"Saved {symbol} data to database")
                except Exception as db_error:
                    logger.error(f"Failed to save {symbol} data to database: {db_error}")

                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for {symbol}")
                    return pd.DataFrame()
    
    def get_crypto_data(self, symbol, exchange='binance', timeframe='1d', limit=100):
        """
        Get cryptocurrency data using CCXT
        """
        try:
            ex = getattr(ccxt, exchange)()
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching crypto data for {symbol} from {exchange}: {e}")
            return pd.DataFrame()
    
    def save_data(self, data, symbol, data_dir='data/raw'):
        """
        Save data to file
        """
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename based on symbol and date
        filename = f"{data_dir}/{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Save the data
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        
        return filename
    
    def load_data(self, filepath):
        """
        Load data from a saved file
        """
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return pd.DataFrame()
    def prepare_training_data(self, symbol, start_date=None, end_date=None):
        """
        Prepare data for training the ML model.
        Fetches data from DB or downloads it if missing.
        """
        # For training, we generally want as much data as possible, or a specific range.
        # We'll use yfinance as the default source for training data as it has good historical coverage.
        logger.info(f"Preparing training data for {symbol}")
        return self.get_yfinance_data(symbol, start_date, end_date)




def get_stock_data(symbol, start_date, end_date, provider='alpha_vantage'):
    """
    Get stock data from specified provider
    """
    provider_instance = DataProvider()

    if provider == 'alpha_vantage':
        data = provider_instance.get_alpha_vantage_data(symbol, start_date=start_date, end_date=end_date)
    elif provider == 'yfinance':
        data = provider_instance.get_yfinance_data(symbol, start_date, end_date)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return data


def get_crypto_data(symbol, exchange='binance', timeframe='1d', limit=100):
    """
    Get cryptocurrency data
    """
    provider_instance = DataProvider()
    return provider_instance.get_crypto_data(symbol, exchange, timeframe, limit)


def get_forex_data(from_currency, to_currency, start_date=None, end_date=None, outputsize='full'):
    """
    Get Forex data
    """
    provider_instance = DataProvider()
    return provider_instance.get_forex_data(from_currency, to_currency, start_date, end_date, outputsize)


def prepare_training_data(symbol, start_date=None, end_date=None):
    """
    Prepare training data for a symbol
    """
    provider_instance = DataProvider()
    return provider_instance.prepare_training_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # Example usage
    dp = DataProvider()
    
    # Get stock data from Alpha Vantage
    data = dp.get_alpha_vantage_data('AAPL')
    print("Alpha Vantage data shape:", data.shape)
    print(data.head())
    
    # Save the data
    if not data.empty:
        dp.save_data(data, 'AAPL')