import sqlite3
import json
import pandas as pd
from typing import Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all SQLite database operations.

    Tables:
      • market_data       — OHLCV price history (existing)
      • agent_trades      — Every trade cycle logged here (Phase 7)
      • agent_state       — Current open positions per symbol (Phase 7)
      • agent_performance — Per-analyst accuracy counters (Phase 7 / Layer 2)
      • retraining_log    — Records each retraining event (Phase 9)
    """

    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self._create_connection()
        self._create_tables()

    def _create_connection(self):
        """Create a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """Create all required tables if they do not exist."""
        try:
            cursor = self.conn.cursor()

            # ── Existing: market price history ─────────────────────────────
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

            # ── New: full trade audit trail ────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_trades (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol              TEXT NOT NULL,
                    action              TEXT NOT NULL,
                    price               REAL,
                    size                REAL,
                    stop_loss           REAL,
                    take_profit         REAL,
                    quant_signal        TEXT,
                    quant_confidence    REAL,
                    sentiment_signal    TEXT,
                    fundamentals_signal TEXT,
                    vote_score          REAL,
                    weights_used        TEXT,
                    pnl                 REAL,
                    closed_at           DATETIME,
                    mode                TEXT,
                    cio_memo            TEXT
                )
            ''')

            # ── New: current open positions ────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_state (
                    symbol          TEXT PRIMARY KEY,
                    position_size   REAL DEFAULT 0,
                    entry_price     REAL DEFAULT 0,
                    stop_loss       REAL DEFAULT 0,
                    take_profit     REAL DEFAULT 0,
                    cash_deployed   REAL DEFAULT 0,
                    trade_id        INTEGER,
                    last_updated    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ── New: per-analyst accuracy for adaptive weighting ───────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_performance (
                    symbol       TEXT NOT NULL,
                    analyst      TEXT NOT NULL,
                    correct      INTEGER DEFAULT 0,
                    total        INTEGER DEFAULT 0,
                    weight       REAL DEFAULT 0.333,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, analyst)
                )
            ''')

            # ── New: retraining event log ──────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS retraining_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol       TEXT NOT NULL,
                    trigger      TEXT,
                    win_rate     REAL,
                    model_age_days INTEGER,
                    success      INTEGER DEFAULT 0,
                    notes        TEXT
                )
            ''')

            self.conn.commit()
            logger.info("All database tables created / verified.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════
    # Existing market_data helpers (unchanged)
    # ═══════════════════════════════════════════════════════════════════════

    def save_data(self, df: pd.DataFrame, symbol: str):
        """Save market data to the database."""
        try:
            df = df.copy()
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            if 'timestamp' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df['timestamp'] = df.index.strftime('%Y-%m-%d %H:%M:%S')
                elif df.index.name == 'timestamp':
                    df['timestamp'] = df.index.astype(str)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            df.to_sql('market_data', self.conn, if_exists='append', index=False)
            logger.info(f"Saved {len(df)} rows for {symbol} to database")
        except sqlite3.IntegrityError:
            logger.warning(f"Data for {symbol} likely already exists (IntegrityError).")
        except Exception as e:
            logger.error(f"Error saving data for {symbol} to database: {e}")
            raise

    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
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
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_data")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def get_latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
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

    def get_earliest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MIN(timestamp) FROM market_data WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting earliest timestamp for {symbol}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # New: agent_trades helpers
    # ═══════════════════════════════════════════════════════════════════════

    def log_trade(self, symbol: str, action: str, price: float, size: float,
                  stop_loss: float, take_profit: float,
                  quant_signal: str, quant_confidence: float,
                  sentiment_signal: str, fundamentals_signal: str,
                  vote_score: float, weights_used: dict,
                  mode: str = "backtest", cio_memo: str = "") -> int:
        """
        Insert a new trade cycle row. Returns the inserted row id.
        PnL and closed_at are filled in later by close_trade().
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO agent_trades
                (symbol, action, price, size, stop_loss, take_profit,
                 quant_signal, quant_confidence, sentiment_signal,
                 fundamentals_signal, vote_score, weights_used, mode, cio_memo)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                symbol, action, price, size, stop_loss, take_profit,
                quant_signal, round(quant_confidence, 4),
                sentiment_signal, fundamentals_signal,
                round(vote_score, 4), json.dumps(weights_used), mode, cio_memo
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error logging trade for {symbol}: {e}")
            return -1

    def close_trade(self, trade_id: int, pnl: float):
        """Update a trade row with PnL when position closes."""
        try:
            self.conn.execute(
                "UPDATE agent_trades SET pnl=?, closed_at=CURRENT_TIMESTAMP WHERE id=?",
                (round(pnl, 4), trade_id)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error closing trade id={trade_id}: {e}")

    def get_recent_trades(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Return last N closed trades for a symbol (used by Retrainer)."""
        try:
            query = '''
                SELECT * FROM agent_trades
                WHERE symbol=? AND pnl IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            '''
            df = pd.read_sql_query(query, self.conn, params=[symbol, limit])
            return df
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_trades(self, limit: int = 50) -> pd.DataFrame:
        """Return the most recent trades across all symbols."""
        try:
            query = "SELECT * FROM agent_trades ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, self.conn, params=[limit])
        except Exception as e:
            logger.error(f"Error fetching all trades: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # New: agent_state helpers
    # ═══════════════════════════════════════════════════════════════════════

    def get_open_position(self, symbol: str) -> Optional[dict]:
        """Return the current open position for a symbol, or None."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM agent_state WHERE symbol=?", (symbol,))
            row = cursor.fetchone()
            if row and row["position_size"] and row["position_size"] != 0:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error fetching open position for {symbol}: {e}")
            return None

    def set_position(self, symbol: str, position_size: float, entry_price: float,
                     stop_loss: float, take_profit: float, trade_id: int):
        """Upsert an open position."""
        try:
            cash_deployed = position_size * entry_price
            self.conn.execute('''
                INSERT INTO agent_state (symbol, position_size, entry_price, stop_loss,
                                         take_profit, cash_deployed, trade_id, last_updated)
                VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    position_size=excluded.position_size,
                    entry_price=excluded.entry_price,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    cash_deployed=excluded.cash_deployed,
                    trade_id=excluded.trade_id,
                    last_updated=CURRENT_TIMESTAMP
            ''', (symbol, position_size, entry_price, stop_loss, take_profit, cash_deployed, trade_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error setting position for {symbol}: {e}")

    def clear_position(self, symbol: str):
        """Remove position record when trade closes."""
        try:
            self.conn.execute(
                "UPDATE agent_state SET position_size=0, entry_price=0, "
                "stop_loss=0, take_profit=0, cash_deployed=0, trade_id=NULL, "
                "last_updated=CURRENT_TIMESTAMP WHERE symbol=?", (symbol,)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error clearing position for {symbol}: {e}")

    def get_total_deployed(self) -> float:
        """Return total cash deployed across all open positions."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT SUM(cash_deployed) FROM agent_state WHERE position_size != 0")
            result = cursor.fetchone()
            return float(result[0] or 0.0)
        except Exception as e:
            logger.error(f"Error getting total deployed: {e}")
            return 0.0

    def get_portfolio_value(self) -> Optional[float]:
        """Approximate portfolio value = initial capital (mark-to-market not tracked here)."""
        try:
            from utils.config import AGENT_INITIAL_CAPITAL
            cursor = self.conn.cursor()
            cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM agent_trades WHERE pnl IS NOT NULL")
            total_pnl = float(cursor.fetchone()[0] or 0.0)
            return AGENT_INITIAL_CAPITAL + total_pnl
        except Exception as e:
            logger.error(f"Error computing portfolio value: {e}")
            return None

    def get_all_positions(self) -> pd.DataFrame:
        """Return all open positions."""
        try:
            return pd.read_sql_query(
                "SELECT * FROM agent_state WHERE position_size != 0", self.conn
            )
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # New: agent_performance helpers (Layer 2 adaptive weights)
    # ═══════════════════════════════════════════════════════════════════════

    def get_analyst_weights(self, symbol: str) -> Optional[dict]:
        """
        Return normalised accuracy-based weights for each analyst.
        Returns None if insufficient history (callers use equal weights).
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT analyst, correct, total, weight FROM agent_performance WHERE symbol=?",
                (symbol,)
            )
            rows = cursor.fetchall()
            if not rows or len(rows) < 3:
                return None

            accuracies = {}
            for row in rows:
                if row["total"] >= 5:   # need at least 5 observations
                    accuracies[row["analyst"]] = row["correct"] / row["total"]

            if len(accuracies) < 3:
                return None

            arr = [accuracies.get("quant", 1/3),
                   accuracies.get("sentiment", 1/3),
                   accuracies.get("fundamentals", 1/3)]
            total = sum(arr)
            if total == 0:
                return None
            norm = [v / total for v in arr]
            return {"quant": norm[0], "sentiment": norm[1], "fundamentals": norm[2]}
        except Exception as e:
            logger.error(f"Error fetching analyst weights for {symbol}: {e}")
            return None

    def update_analyst_performance(self, symbol: str, analyst: str,
                                   correct: bool):
        """Increment accuracy counters for an analyst after a trade closes."""
        try:
            self.conn.execute('''
                INSERT INTO agent_performance (symbol, analyst, correct, total, last_updated)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol, analyst) DO UPDATE SET
                    correct = correct + ?,
                    total   = total + 1,
                    last_updated = CURRENT_TIMESTAMP
            ''', (symbol, analyst, int(correct), int(correct)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating analyst performance: {e}")

    def get_performance_table(self) -> pd.DataFrame:
        """Return full agent_performance table for the dashboard."""
        try:
            return pd.read_sql_query(
                "SELECT *, CAST(correct AS REAL)/NULLIF(total,0) AS accuracy "
                "FROM agent_performance ORDER BY symbol, analyst",
                self.conn
            )
        except Exception as e:
            logger.error(f"Error fetching performance table: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # New: retraining_log helpers
    # ═══════════════════════════════════════════════════════════════════════

    def log_retraining(self, symbol: str, trigger: str, win_rate: float,
                       model_age_days: int, success: bool, notes: str = ""):
        """Record a retraining event."""
        try:
            self.conn.execute('''
                INSERT INTO retraining_log (symbol, trigger, win_rate, model_age_days, success, notes)
                VALUES (?,?,?,?,?,?)
            ''', (symbol, trigger, round(win_rate, 4), model_age_days, int(success), notes))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging retraining event: {e}")

    def get_retraining_log(self, limit: int = 50) -> pd.DataFrame:
        """Return the most recent retraining events."""
        try:
            return pd.read_sql_query(
                "SELECT * FROM retraining_log ORDER BY timestamp DESC LIMIT ?",
                self.conn, params=[limit]
            )
        except Exception as e:
            logger.error(f"Error fetching retraining log: {e}")
            return pd.DataFrame()

    def get_last_retrain_date(self, symbol: str) -> Optional[datetime]:
        """Return the datetime of the last successful retraining for a symbol."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT MAX(timestamp) FROM retraining_log WHERE symbol=? AND success=1",
                (symbol,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            return None
        except Exception as e:
            logger.error(f"Error fetching last retrain date for {symbol}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Connection management
    # ═══════════════════════════════════════════════════════════════════════

    def close_connection(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        self.close_connection()


if __name__ == "__main__":
    db_manager = DatabaseManager()
    exists = db_manager.symbol_exists("AAPL")
    print(f"AAPL data exists: {exists}")
    df = db_manager.load_data("AAPL")
    print(f"Loaded {len(df)} rows for AAPL")
    if not df.empty:
        print(df.head())


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

    def get_earliest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Get the earliest timestamp for a symbol in the database.
        
        Args:
            symbol (str): Symbol of the asset
            
        Returns:
            pd.Timestamp: The earliest timestamp, or None if no data exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MIN(timestamp) FROM market_data WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting earliest timestamp for {symbol}: {e}")
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