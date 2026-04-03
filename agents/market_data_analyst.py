"""
Market Data Analyst — Phase 2

Fetches OHLCV data and computes technical features for each symbol.
Runs first in the pipeline (priority=1) so all downstream agents have
fresh data in context["market_data"].

Per-symbol try/except ensures one bad symbol never blocks others.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

from agents.base_agent import BaseAgent
from utils.data_loader import DataProvider
from utils.features import add_technical_features

logger = logging.getLogger(__name__)


class MarketDataAnalyst(BaseAgent):
    """
    Fetches OHLCV market data and engineers technical features for all
    configured symbols. Populates context["market_data"].
    """

    name = "MarketDataAnalyst"
    role = "analyst"
    priority = 1  # Runs first — every other agent depends on this data

    def __init__(self):
        self._provider = DataProvider()

    def run(self, context: dict) -> dict:
        """
        Args:
            context: Must contain "symbols" (list[str]) and optionally
                     "lookback_days" (int, default 365).

        Returns:
            context with context["market_data"][symbol] = {
                "ohlcv": pd.DataFrame,
                "latest_close": float,
                "atr": float,
                "features": pd.DataFrame (last row),
            }
        """
        symbols = context.get("symbols", [])
        lookback_days = context.get("lookback_days", 365)
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        market_data: dict = {}

        for symbol in symbols:
            try:
                logger.info(f"[MarketDataAnalyst] Fetching {symbol} ({start_date} → {end_date})")
                df = self._provider.get_yfinance_data(symbol, start_date, end_date)

                if df is None or df.empty:
                    logger.warning(f"[MarketDataAnalyst] No data returned for {symbol}, skipping")
                    continue

                # Ensure lowercase column names
                df.columns = [c.lower() for c in df.columns]

                # Compute technical features
                df_feat = add_technical_features(df)
                if df_feat.empty:
                    logger.warning(f"[MarketDataAnalyst] Feature engineering produced empty df for {symbol}")
                    continue

                latest = df_feat.iloc[-1]
                atr = float(latest.get("atr", 0.0))

                market_data[symbol] = {
                    "ohlcv": df,
                    "latest_close": float(latest["close"]),
                    "atr": atr,
                    "features": df_feat,   # full feature df; quant analyst uses last row
                }
                logger.info(
                    f"[MarketDataAnalyst] {symbol}: close={latest['close']:.4f}, "
                    f"atr={atr:.4f}, rows={len(df_feat)}"
                )

            except Exception as exc:
                logger.error(f"[MarketDataAnalyst] Error processing {symbol}: {exc}", exc_info=True)
                # Continue with remaining symbols — never crash the loop

        context["market_data"] = market_data
        logger.info(f"[MarketDataAnalyst] Done. Data available for: {list(market_data.keys())}")
        return context
