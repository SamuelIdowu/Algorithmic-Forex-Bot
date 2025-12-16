import sqlite3
import pandas as pd
from typing import Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    A class to manage SQLite database operations for market data.
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_connection()
        self._create_table()
    
    def _create_connection(self):
        """Create a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_table(self):
        """Create the market_data table if it doesn't exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (timestamp, symbol)
                )
            ''')
            self.conn.commit()
            logger.info("Market data table created or already exists")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, symbol: str):
        """
        Save market data to the database.
        
        Args:
            df (pd.DataFrame): DataFrame containing market data with columns: 
                               ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            symbol (str): Symbol of the asset
        """
        try:
            df = df.copy()
            
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # Handle timestamp
            if 'timestamp' not in df.columns:
                # If timestamp is not a column, try to get it from index
                if isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
                elif df.index.name == 'timestamp':
                    df['timestamp'] = df.index.astype(str)
            else:
                # If it is a column, ensure it is string
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert data into the database
            # We use 'replace' if we want to update, but 'append' is safer for history.
            # However, 'append' will fail on duplicate PK. 
            # Since we have (timestamp, symbol) PK, we should probably handle duplicates.
            # But for now, let's stick to 'append' and rely on the user/caller to not insert duplicates,
            # or catch the error. The error log showed IntegrityError, which is expected for duplicates.
            # But the specific error was NOT NULL timestamp, which means timestamp was missing.
            
            # To avoid "database is locked" or other issues, we might want to use a context manager for connection,
            # but self.conn is persistent.
            
            df.to_sql('market_data', self.conn, if_exists='append', index=False)
            logger.info(f"Saved {len(df)} rows for {symbol} to database")
        except sqlite3.IntegrityError:
            # If we hit a constraint error (likely duplicate), we might want to ignore or log warning.
            # But if it's NOT NULL timestamp, that's a bug we just fixed.
            # If it's UNIQUE constraint, it means data already exists.
            logger.warning(f"Data for {symbol} likely already exists (IntegrityError).")
        except Exception as e:
            logger.error(f"Error saving data for {symbol} to database: {e}")
            raise
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data from the database for a given symbol and optional date range.
        
        Args:
            symbol (str): Symbol of the asset
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
        
        Returns:
            pd.DataFrame: DataFrame containing market data
        """
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            
            if not df.empty:
                # Convert timestamp to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Loaded {len(df)} rows for {symbol} from database")
            else:
                logger.info(f"No data found for {symbol} in the specified date range")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol} from database: {e}")
            raise
    
    def symbol_exists(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """
        Check if data exists for a symbol in the database within an optional date range.
        
        Args:
            symbol (str): Symbol of the asset
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
        
        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            query = "SELECT 1 FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            exists = result is not None
            if exists:
                logger.info(f"Data for {symbol} exists in database")
            else:
                logger.info(f"Data for {symbol} does not exist in database")
            
            return exists
        except Exception as e:
            logger.error(f"Error checking if data exists for {symbol}: {e}")
            return False
    
    def get_available_symbols(self) -> list:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            list: List of available symbols
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_data")
            symbols = [row[0] for row in cursor.fetchall()]
            return symbols
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def get_latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Get the latest timestamp for a symbol in the database.
        
        Args:
            symbol (str): Symbol of the asset
            
        Returns:
            pd.Timestamp: The latest timestamp, or None if no data exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM market_data WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting latest timestamp for {symbol}: {e}")
            return None
    
    def close_connection(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """Ensure the database connection is closed when the object is destroyed."""
        self.close_connection()


if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    
    # Example: Check if AAPL data exists
    exists = db_manager.symbol_exists("AAPL")
    print(f"AAPL data exists: {exists}")
    
    # Example: Load AAPL data
    df = db_manager.load_data("AAPL")
    print(f"Loaded {len(df)} rows for AAPL")
    if not df.empty:
        print(df.head())