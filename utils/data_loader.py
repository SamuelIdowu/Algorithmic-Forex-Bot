"""
Data Loader — simplified to yfinance + SQLite caching only.

The agent pipeline exclusively uses get_yfinance_data(). All legacy
Alpha Vantage and CCXT methods have been removed.
"""
import logging
import os
import time

import pandas as pd
import yfinance as yf

from data.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Provides market data via Yahoo Finance with SQLite caching.

    On every request the cache is checked first; data is only downloaded
    from Yahoo Finance when the DB does not cover the requested date range.
    """

    def __init__(self):
        self.db_manager = DatabaseManager()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_yfinance_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance with SQLite caching.

        Args:
            symbol:     Yahoo Finance ticker (e.g. "BTC-USD", "EURUSD=X").
            start_date: ISO date string "YYYY-MM-DD".
            end_date:   ISO date string "YYYY-MM-DD" (default: today).
            interval:   Candle size — "1d", "1h", "15m", "5m".

        Returns:
            pd.DataFrame with lowercase OHLCV columns, DatetimeIndex.
            Empty DataFrame on failure.
        """
        # Use interval-aware key so daily and intraday bars don't mix
        db_symbol = symbol if interval == "1d" else f"{symbol}_{interval}"

        if self._cache_covers(db_symbol, start_date, end_date):
            logger.info(f"[DataProvider] Loading {db_symbol} from cache")
            return self.db_manager.load_data(db_symbol, start_date, end_date)

        logger.info(
            f"[DataProvider] Fetching {db_symbol} from Yahoo Finance "
            f"({start_date} → {end_date or 'today'}, interval={interval})"
        )

        # Yahoo Finance uses "EURUSD=X" format; convert "EUR/USD" style
        yf_symbol = symbol.replace("/", "") + "=X" if "/" in symbol else symbol

        df = self._download_with_retry(yf_symbol, start_date, end_date, interval)
        if df.empty:
            return df

        # Normalise column names to lowercase
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0].lower() if isinstance(col, tuple) else col.lower()
                for col in df.columns
            ]
        else:
            df.columns = [c.lower() for c in df.columns]

        try:
            self.db_manager.save_data(df, db_symbol)
            logger.info(f"[DataProvider] Cached {db_symbol} ({len(df)} rows)")
        except Exception as exc:
            logger.warning(f"[DataProvider] Cache write failed for {db_symbol}: {exc}")

        return df

    def prepare_training_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Convenience wrapper used by train_model.py."""
        return self.get_yfinance_data(symbol, start_date, end_date)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _cache_covers(
        self, db_symbol: str, start_date: str, end_date: str
    ) -> bool:
        """Return True if the DB already has data covering the requested range."""
        if not self.db_manager.symbol_exists(db_symbol, start_date):
            return False
        latest = self.db_manager.get_latest_timestamp(db_symbol)
        earliest = self.db_manager.get_earliest_timestamp(db_symbol)
        if not (latest and earliest):
            return False
        if end_date:
            end_dt = pd.to_datetime(end_date)
            start_dt = pd.to_datetime(start_date) if start_date else None
            if latest >= end_dt and (start_dt is None or earliest <= start_dt):
                return True
            logger.info(
                f"[DataProvider] Cache for {db_symbol} "
                f"({earliest.date()} → {latest.date()}) doesn't cover "
                f"({start_date} → {end_date}). Refreshing."
            )
            return False
        if start_date:
            start_dt = pd.to_datetime(start_date)
            return earliest <= start_dt
        return True

    @staticmethod
    def _download_with_retry(
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> pd.DataFrame:
        """Download from yfinance with simple retry logic."""
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    logger.warning(f"[DataProvider] yfinance returned empty for {symbol}")
                return df
            except Exception as exc:
                logger.warning(
                    f"[DataProvider] Attempt {attempt + 1}/{max_retries} "
                    f"failed for {symbol}: {exc}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.error(f"[DataProvider] All {max_retries} attempts failed for {symbol}")
        return pd.DataFrame()


# ── Module-level convenience functions (used by legacy train/predict scripts) ──

def prepare_training_data(
    symbol: str, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    return DataProvider().prepare_training_data(symbol, start_date, end_date)


def get_yfinance_data(
    symbol: str,
    start_date: str,
    end_date: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    return DataProvider().get_yfinance_data(symbol, start_date, end_date, interval)