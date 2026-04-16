import sqlite3
import json
import pandas as pd
from typing import Optional
import logging
from datetime import datetime, timedelta

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

            # ── New: predictions audit trail (insights engine) ─────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol              TEXT NOT NULL,
                    action              TEXT NOT NULL,
                    entry_price         REAL,
                    stop_loss           REAL,
                    take_profit         REAL,
                    vote_score          REAL,
                    weights_used        TEXT,
                    quant_signal        TEXT,
                    quant_confidence    REAL,
                    sentiment_signal    TEXT,
                    fundamentals_signal TEXT,
                    cio_memo            TEXT,
                    current_price       REAL
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

            # ── New: real-time news storage ────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol      TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    source      TEXT,
                    sentiment   TEXT,
                    url         TEXT
                )
            ''')

            # ── New: persistent monitors ──────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_monitors (
                    chat_id     INTEGER NOT NULL,
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL,
                    last_run    DATETIME,
                    PRIMARY KEY (chat_id, symbol)
                )
            ''')

            # ── New: prediction tracker (virtual trade lifecycle) ─────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_tracker (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id           INTEGER REFERENCES predictions(id),
                    source                  TEXT NOT NULL,
                    symbol                  TEXT NOT NULL,
                    direction               TEXT NOT NULL,
                    timeframe               TEXT NOT NULL,
                    entry_price             REAL NOT NULL,
                    take_profit             REAL,
                    stop_loss               REAL,
                    status                  TEXT DEFAULT 'ACTIVE',
                    current_price           REAL,
                    pnl_percent             REAL,
                    rsi                     REAL,
                    atr                     REAL,
                    bb_pos                  REAL,
                    ml_confidence           REAL,
                    quant_confidence        REAL,
                    vote_score              REAL,
                    tp_hit_at               DATETIME,
                    sl_hit_at               DATETIME,
                    direction_check_at      DATETIME,
                    direction_result        TEXT,
                    final_price_at_outcome  REAL,
                    max_profit_reached      REAL,
                    max_loss_reached        REAL,
                    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_checked_at         DATETIME,
                    expires_at              DATETIME,
                    reasoning               TEXT,
                    agent_insights_json     TEXT
                )
            ''')

            # ── New: price check log (audit trail for active predictions) ─
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_check_log (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracker_id              INTEGER REFERENCES prediction_tracker(id),
                    checked_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price_at_check          REAL,
                    pnl_at_check            REAL,
                    status_at_check         TEXT
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
                ORDER BY id DESC LIMIT ?
            '''
            return pd.read_sql_query(query, self.conn, params=(symbol, limit))
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return pd.DataFrame()

    def get_latest_consensus(self, symbol: str) -> pd.DataFrame:
        """Return the single most recent analyst cycle for a symbol (even if HOLD)."""
        try:
            query = "SELECT * FROM agent_trades WHERE symbol=? ORDER BY id DESC LIMIT 1"
            return pd.read_sql_query(query, self.conn, params=(symbol,))
        except Exception as e:
            logger.error(f"Error getting latest consensus for {symbol}: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════
    # New: predictions helpers (insights engine)
    # ═══════════════════════════════════════════════════════════════════════

    def log_prediction(self, symbol: str, action: str, entry_price: float,
                       stop_loss: float, take_profit: float, vote_score: float,
                       weights_used: dict, quant_signal: str, quant_confidence: float,
                       sentiment_signal: str, fundamentals_signal: str,
                       cio_memo: str, current_price: float) -> int:
        """
        Insert a prediction record. Returns the inserted row id.
        Used by PredictionLogger to log every analysis cycle without executing trades.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO predictions
                (symbol, action, entry_price, stop_loss, take_profit,
                 vote_score, weights_used, quant_signal, quant_confidence,
                 sentiment_signal, fundamentals_signal, cio_memo, current_price)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                symbol, action, entry_price, stop_loss, take_profit,
                round(vote_score, 4), json.dumps(weights_used),
                quant_signal, round(quant_confidence, 4),
                sentiment_signal, fundamentals_signal, cio_memo,
                round(current_price, 6) if current_price else None,
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error logging prediction for {symbol}: {e}")
            return -1

    def get_recent_predictions(self, symbol: str = None, limit: int = 20) -> pd.DataFrame:
        """Return recent predictions, optionally filtered by symbol."""
        try:
            if symbol:
                query = "SELECT * FROM predictions WHERE symbol=? ORDER BY id DESC LIMIT ?"
                return pd.read_sql_query(query, self.conn, params=(symbol, limit))
            else:
                query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ?"
                return pd.read_sql_query(query, self.conn, params=[limit])
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {e}")
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
    # Baseline & performance helpers
    # ═══════════════════════════════════════════════════════════════════════

    def get_prediction_baseline(self) -> Optional[float]:
        """Return a virtual baseline = notional capital + cumulative PnL (insights engine metric)."""
        try:
            NOTIONAL_CAPITAL = 10000.0
            cursor = self.conn.cursor()
            cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM agent_trades WHERE pnl IS NOT NULL")
            total_pnl = float(cursor.fetchone()[0] or 0.0)
            return NOTIONAL_CAPITAL + total_pnl
        except Exception as e:
            logger.error(f"Error computing prediction baseline: {e}")
            return None

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
    # New: news helpers
    # ═══════════════════════════════════════════════════════════════════════

    def save_news(self, symbol: str, headlines: list[dict]):
        """
        Save a batch of headlines to the database.
        Expected headline format: {'title': str, 'source': str, 'sentiment': str, 'url': str}
        """
        try:
            cursor = self.conn.cursor()
            for h in headlines:
                cursor.execute('''
                    INSERT INTO news (symbol, title, source, sentiment, url)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    h.get("title", ""),
                    h.get("source", "Unknown"),
                    h.get("sentiment", "NEUTRAL"),
                    h.get("url", "")
                ))
            self.conn.commit()
            logger.info(f"Saved {len(headlines)} headlines for {symbol}")
        except Exception as e:
            logger.error(f"Error saving news for {symbol}: {e}")

    def get_recent_news(self, limit: int = 15) -> list[dict]:
        """Return the most recent news headlines across all symbols."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT symbol, title, source, sentiment, url, timestamp
                FROM news
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "symbol": r["symbol"],
                    "title": r["title"],
                    "source": r["source"],
                    "sentiment": r["sentiment"].lower() if isinstance(r["sentiment"], str) else "neutral",
                    "time": r["timestamp"] # UI will parse this
                })
            
            # Correction: sentiment mapping
            for res in results:
                 s = res.get("sentiment", "neutral").lower()
                 if "bullish" in s:
                     res["sentiment"] = "bullish"
                 elif "bearish" in s:
                     res["sentiment"] = "bearish"
                 else:
                     res["sentiment"] = "neutral"

            return results
        except Exception as e:
            logger.error(f"Error fetching recent news: {e}")
            return []

    def get_news(self, symbol: str, limit: int = 5) -> list[dict]:
        """Return the most recent news headlines for a specific symbol."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT title, source, sentiment, url, timestamp
                FROM news
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def save_monitor(self, chat_id: int, symbol: str, interval: str, last_run: Optional[datetime] = None):
        """Save or update a monitor."""
        try:
            self.conn.execute('''
                INSERT INTO active_monitors (chat_id, symbol, interval, last_run)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, symbol) DO UPDATE SET
                    interval=excluded.interval,
                    last_run=excluded.last_run
            ''', (chat_id, symbol, interval, last_run.isoformat() if last_run else None))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving monitor: {e}")

    def delete_monitor(self, chat_id: int, symbol: str):
        """Delete a monitor."""
        try:
            if symbol == "ALL":
                self.conn.execute("DELETE FROM active_monitors WHERE chat_id=?", (chat_id,))
            else:
                self.conn.execute("DELETE FROM active_monitors WHERE chat_id=? AND symbol=?", (chat_id, symbol))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error deleting monitor: {e}")

    def get_active_monitors(self) -> list[dict]:
        """Load all active monitors from DB."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM active_monitors")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "chat_id": r["chat_id"],
                    "symbol": r["symbol"],
                    "interval": r["interval"],
                    "last_run": datetime.fromisoformat(r["last_run"]) if r["last_run"] else None
                })
            return results
        except Exception as e:
            logger.error(f"Error fetching monitors: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════
    # New: prediction tracker helpers (virtual trade lifecycle)
    # ═══════════════════════════════════════════════════════════════════════

    def create_prediction_tracker(self, tracker_data: dict) -> int:
        """
        Insert a new prediction tracker entry.

        tracker_data should contain:
          - prediction_id (optional FK link)
          - source: "agent_loop", "standalone", or "backfill"
          - symbol
          - direction: "UP"/"DOWN" or "BUY"/"SELL"/"HOLD"
          - timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
          - entry_price
          - take_profit (optional)
          - stop_loss (optional)
          - status (default "ACTIVE")
          - current_price (optional)
          - rsi, atr, bb_pos (optional technicals)
          - ml_confidence, quant_confidence, vote_score (optional)
          - reasoning (optional)
          - agent_insights_json (optional JSON string)

        Returns the inserted row id, or -1 on error.
        Automatically calculates expires_at from timeframe.
        """
        try:
            timeframe = tracker_data.get("timeframe", "1h")
            timeframe_hours = {
                "1m": 1, "5m": 2, "15m": 6, "30m": 12,
                "1h": 24, "4h": 96, "1d": 576, "1w": 1008
            }
            hours = timeframe_hours.get(timeframe, 24)
            now = datetime.now()
            expires_at = now + timedelta(hours=hours)

            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_tracker
                (prediction_id, source, symbol, direction, timeframe,
                 entry_price, take_profit, stop_loss, status,
                 current_price, pnl_percent,
                 rsi, atr, bb_pos,
                 ml_confidence, quant_confidence, vote_score,
                 reasoning, agent_insights_json,
                 created_at, last_checked_at, expires_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                tracker_data.get("prediction_id"),
                tracker_data.get("source", "standalone"),
                tracker_data["symbol"],
                tracker_data["direction"],
                timeframe,
                tracker_data["entry_price"],
                tracker_data.get("take_profit"),
                tracker_data.get("stop_loss"),
                tracker_data.get("status", "ACTIVE"),
                tracker_data.get("current_price"),
                tracker_data.get("pnl_percent"),
                tracker_data.get("rsi"),
                tracker_data.get("atr"),
                tracker_data.get("bb_pos"),
                tracker_data.get("ml_confidence"),
                tracker_data.get("quant_confidence"),
                tracker_data.get("vote_score"),
                tracker_data.get("reasoning"),
                tracker_data.get("agent_insights_json"),
                now.isoformat(),
                now.isoformat(),
                expires_at.isoformat()
            ))
            self.conn.commit()
            logger.info(f"Created prediction tracker for {tracker_data['symbol']} "
                        f"(direction={tracker_data['direction']}, expires={expires_at})")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating prediction tracker: {e}")
            return -1

    def get_active_trackers(self, symbol: str = None) -> pd.DataFrame:
        """
        Return all trackers with status='ACTIVE'.
        Optionally filtered by symbol.
        """
        try:
            if symbol:
                query = "SELECT * FROM prediction_tracker WHERE status='ACTIVE' AND symbol=? ORDER BY created_at DESC"
                return pd.read_sql_query(query, self.conn, params=(symbol,))
            else:
                query = "SELECT * FROM prediction_tracker WHERE status='ACTIVE' ORDER BY created_at DESC"
                return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Error fetching active trackers: {e}")
            return pd.DataFrame()

    def get_tracker_by_id(self, tracker_id: int) -> dict:
        """Return a single tracker row as a dict."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM prediction_tracker WHERE id=?", (tracker_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}
        except Exception as e:
            logger.error(f"Error fetching tracker id={tracker_id}: {e}")
            return {}

    def get_trackers_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Return all trackers for a symbol, ordered by created_at DESC."""
        try:
            query = "SELECT * FROM prediction_tracker WHERE symbol=? ORDER BY created_at DESC"
            return pd.read_sql_query(query, self.conn, params=(symbol,))
        except Exception as e:
            logger.error(f"Error fetching trackers for {symbol}: {e}")
            return pd.DataFrame()

    def update_tracker_status(self, tracker_id: int, status: str,
                              current_price: float = None,
                              pnl_percent: float = None) -> None:
        """Update tracker status, current_price, pnl_percent, and last_checked_at."""
        try:
            self.conn.execute('''
                UPDATE prediction_tracker
                SET status=?, current_price=?, pnl_percent=?, last_checked_at=?
                WHERE id=?
            ''', (
                status,
                current_price,
                pnl_percent,
                datetime.now().isoformat(),
                tracker_id
            ))
            self.conn.commit()
            logger.info(f"Updated tracker id={tracker_id} status={status}")
        except Exception as e:
            logger.error(f"Error updating tracker status for id={tracker_id}: {e}")

    def update_tracker_outcome(self, tracker_id: int, outcome_type: str,
                               price: float) -> None:
        """
        Set outcome fields based on the outcome type.

        outcome_type can be:
          - 'tp_hit': sets status=WON_TP, tp_hit_at=now, final_price_at_outcome=price
          - 'sl_hit': sets status=LOST_SL, sl_hit_at=now, final_price_at_outcome=price
          - 'direction_win': sets status=WON_DIRECTION, direction_check_at=now,
                             direction_result='WIN', final_price_at_outcome=price
          - 'direction_loss': sets status=EXPIRED, direction_check_at=now,
                              direction_result='LOSS', final_price_at_outcome=price
          - 'expired': sets status=EXPIRED, direction_check_at=now
        """
        try:
            now = datetime.now().isoformat()

            outcome_map = {
                "tp_hit": (
                    "WON_TP", price, None, None, now, None, None
                ),
                "sl_hit": (
                    "LOST_SL", None, price, now, None, None, None
                ),
                "direction_win": (
                    "WON_DIRECTION", None, None, None, now, "WIN", price
                ),
                "direction_loss": (
                    "EXPIRED", None, None, None, now, "LOSS", price
                ),
                "expired": (
                    "EXPIRED", None, None, None, now, None, None
                ),
            }

            if outcome_type not in outcome_map:
                logger.error(f"Unknown outcome type: {outcome_type}")
                return

            (status, tp_hit_at, sl_hit_at, tp_sl_time,
             direction_check_at, direction_result, final_price) = outcome_map[outcome_type]

            # Build dynamic update based on outcome type
            if outcome_type == "tp_hit":
                self.conn.execute('''
                    UPDATE prediction_tracker
                    SET status=?, tp_hit_at=?, final_price_at_outcome=?, last_checked_at=?
                    WHERE id=?
                ''', (status, tp_hit_at, final_price, now, tracker_id))
            elif outcome_type == "sl_hit":
                self.conn.execute('''
                    UPDATE prediction_tracker
                    SET status=?, sl_hit_at=?, final_price_at_outcome=?, last_checked_at=?
                    WHERE id=?
                ''', (status, sl_hit_at, final_price, now, tracker_id))
            elif outcome_type in ("direction_win", "direction_loss", "expired"):
                self.conn.execute('''
                    UPDATE prediction_tracker
                    SET status=?, direction_check_at=?, direction_result=?,
                        final_price_at_outcome=?, last_checked_at=?
                    WHERE id=?
                ''', (status, direction_check_at, direction_result, final_price, now, tracker_id))

            self.conn.commit()
            logger.info(f"Set outcome for tracker id={tracker_id}: {outcome_type}")
        except Exception as e:
            logger.error(f"Error updating tracker outcome for id={tracker_id}: {e}")

    def update_tracker_extremes(self, tracker_id: int, current_price: float) -> None:
        """
        Update max_profit_reached and max_loss_reached based on current price.
        Compares current price against entry_price to determine if it's a new high or low.
        """
        try:
            tracker = self.get_tracker_by_id(tracker_id)
            if not tracker:
                logger.warning(f"Tracker id={tracker_id} not found for extremes update")
                return

            entry_price = tracker.get("entry_price")
            if not entry_price:
                return

            direction = tracker.get("direction", "")
            max_profit = tracker.get("max_profit_reached")
            max_loss = tracker.get("max_loss_reached")

            # For UP/BUY: profit = price goes up, loss = price goes down
            # For DOWN/SELL: profit = price goes down, loss = price goes up
            is_bullish = direction in ("UP", "BUY")

            if is_bullish:
                # Profit when price rises above entry
                if current_price > entry_price:
                    if max_profit is None or current_price > max_profit:
                        max_profit = current_price
                # Loss when price falls below entry
                if current_price < entry_price:
                    if max_loss is None or current_price < max_loss:
                        max_loss = current_price
            else:
                # Profit when price falls below entry (short)
                if current_price < entry_price:
                    if max_profit is None or current_price < max_profit:
                        max_profit = current_price
                # Loss when price rises above entry (short)
                if current_price > entry_price:
                    if max_loss is None or current_price > max_loss:
                        max_loss = current_price

            self.conn.execute('''
                UPDATE prediction_tracker
                SET max_profit_reached=?, max_loss_reached=?,
                    current_price=?, last_checked_at=?
                WHERE id=?
            ''', (max_profit, max_loss, current_price, datetime.now().isoformat(), tracker_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating tracker extremes for id={tracker_id}: {e}")

    def log_price_check(self, tracker_id: int, price: float,
                        pnl: float, status: str) -> None:
        """Insert a row into price_check_log."""
        try:
            self.conn.execute('''
                INSERT INTO price_check_log
                (tracker_id, checked_at, price_at_check, pnl_at_check, status_at_check)
                VALUES (?,?,?,?,?)
            ''', (tracker_id, datetime.now().isoformat(), price, pnl, status))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging price check for tracker id={tracker_id}: {e}")

    def get_tracker_stats(self, symbol: str = None,
                          time_range_days: int = None) -> dict:
        """
        Calculate win rate statistics for prediction trackers.

        Optionally filtered by symbol and time_range_days.
        Returns dict with all stats.
        """
        try:
            where_clauses = []
            params = []

            if symbol:
                where_clauses.append("symbol=?")
                params.append(symbol)

            if time_range_days:
                where_clauses.append("created_at >= datetime('now', ?)")
                params.append(f"-{time_range_days} days")

            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            cursor = self.conn.cursor()

            # Total predictions
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker{where_sql}", params
            )
            total = cursor.fetchone()["cnt"]

            # Active count
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker "
                f"WHERE status='ACTIVE'{where_sql}", params
            )
            active = cursor.fetchone()["cnt"]

            # Completed count (WON_* or LOST_*)
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker "
                f"WHERE status IN ('WON_TP','WON_DIRECTION','LOST_SL'){where_sql}", params
            )
            completed = cursor.fetchone()["cnt"]

            # TP hits
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker "
                f"WHERE status='WON_TP'{where_sql}", params
            )
            tp_hits = cursor.fetchone()["cnt"]

            # Directional wins
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker "
                f"WHERE status='WON_DIRECTION'{where_sql}", params
            )
            directional_wins = cursor.fetchone()["cnt"]

            # Completed with direction result
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM prediction_tracker "
                f"WHERE direction_result IS NOT NULL{where_sql}", params
            )
            completed_with_direction = cursor.fetchone()["cnt"]

            # Avg PnL at outcome
            cursor.execute(
                f"SELECT AVG(pnl_percent) as avg_pnl FROM prediction_tracker "
                f"WHERE status IN ('WON_TP','WON_DIRECTION','LOST_SL'){where_sql}", params
            )
            avg_pnl_row = cursor.fetchone()
            avg_pnl = avg_pnl_row["avg_pnl"] if avg_pnl_row else None

            # Avg time to outcome (hours)
            cursor.execute(
                f"SELECT AVG(julianday(COALESCE(tp_hit_at, sl_hit_at, direction_check_at)) "
                f"- julianday(created_at)) * 24 as avg_hours FROM prediction_tracker "
                f"WHERE status IN ('WON_TP','WON_DIRECTION','LOST_SL'){where_sql}", params
            )
            avg_time_row = cursor.fetchone()
            avg_time = avg_time_row["avg_hours"] if avg_time_row else None

            tp_win_rate = (tp_hits / completed * 100) if completed > 0 else 0
            directional_accuracy = (directional_wins / completed_with_direction * 100) \
                if completed_with_direction > 0 else 0

            return {
                "total_predictions": total,
                "active_count": active,
                "completed_count": completed,
                "tp_hits": tp_hits,
                "tp_win_rate": round(tp_win_rate, 2),
                "directional_wins": directional_wins,
                "directional_accuracy": round(directional_accuracy, 2),
                "avg_pnl_at_outcome": round(avg_pnl, 4) if avg_pnl else None,
                "avg_time_to_outcome_hours": round(avg_time, 2) if avg_time else None,
            }
        except Exception as e:
            logger.error(f"Error computing tracker stats: {e}")
            return {
                "total_predictions": 0,
                "active_count": 0,
                "completed_count": 0,
                "tp_hits": 0,
                "tp_win_rate": 0,
                "directional_wins": 0,
                "directional_accuracy": 0,
                "avg_pnl_at_outcome": None,
                "avg_time_to_outcome_hours": None,
            }

    def backfill_trackers_from_predictions(self) -> int:
        """
        One-time migration: read existing predictions table and create tracker entries
        for predictions that don't have a corresponding tracker.

        Missing fields (rsi, atr, etc) are left as NULL.
        source='backfill', timeframe='1h' (default for historical data).

        Returns count of trackers created.
        """
        try:
            cursor = self.conn.cursor()

            # Find predictions without a corresponding tracker
            cursor.execute('''
                SELECT p.* FROM predictions p
                LEFT JOIN prediction_tracker pt ON pt.prediction_id = p.id
                WHERE pt.id IS NULL
            ''')
            rows = cursor.fetchall()

            if not rows:
                logger.info("No predictions to backfill — all already have trackers.")
                return 0

            count = 0
            for row in rows:
                tracker_data = {
                    "prediction_id": row["id"],
                    "source": "backfill",
                    "symbol": row["symbol"],
                    "direction": row["action"],
                    "timeframe": "1h",
                    "entry_price": row["entry_price"] or 0.0,
                    "take_profit": row.get("take_profit"),
                    "stop_loss": row.get("stop_loss"),
                    "status": "ACTIVE",
                    "current_price": row.get("current_price"),
                    "quant_confidence": row.get("quant_confidence"),
                    "vote_score": row.get("vote_score"),
                    "reasoning": row.get("cio_memo"),
                }

                tracker_id = self.create_prediction_tracker(tracker_data)
                if tracker_id > 0:
                    count += 1

            logger.info(f"Backfilled {count} trackers from predictions table.")
            return count
        except Exception as e:
            logger.error(f"Error backfilling trackers: {e}")
            return 0

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