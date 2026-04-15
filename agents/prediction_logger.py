"""
Prediction Logger — Audit Trail for Insights Engine

Replaces the old PortfolioManager. No longer executes trades or manages positions.
Instead, it logs every analysis cycle to the database as a prediction record,
creating a full audit trail of signals, confidence scores, and analyst consensus.

Output: context["prediction_log"][symbol] = {
    "action": str,
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "vote_score": float,
    "logged_id": int,
}
"""
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from data.db_manager import DatabaseManager
from utils.telegram_utils import send_telegram_message_sync

logger = logging.getLogger(__name__)


class PredictionLogger(BaseAgent):
    """
    Logs prediction signals from the RiskManager to the database.
    Provides an audit trail without executing any trades.
    """

    name = "PredictionLogger"
    role = "manager"
    priority = 7   # After RiskManager

    def __init__(self):
        self._db = DatabaseManager()

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        risk    = context.get("risk", {})
        quant   = context.get("quant", {})
        sentiment = context.get("sentiment", {})
        fundamentals = context.get("fundamentals", {})
        cio     = context.get("cio", {})
        market  = context.get("market_data", {})

        prediction_log = {}

        for symbol in symbols:
            try:
                r = risk.get(symbol, {})
                action = r.get("action", "HOLD")

                # Log prediction to DB
                logged_id = self._db.log_prediction(
                    symbol=symbol,
                    action=action,
                    entry_price=r.get("entry_price", 0),
                    stop_loss=r.get("stop_loss", 0),
                    take_profit=r.get("take_profit", 0),
                    vote_score=r.get("vote_score", 0),
                    weights_used=r.get("weights_used", {}),
                    quant_signal=quant.get(symbol, {}).get("quant_signal", ""),
                    quant_confidence=quant.get(symbol, {}).get("quant_confidence", 0),
                    sentiment_signal=sentiment.get(symbol, {}).get("sentiment_signal", ""),
                    fundamentals_signal=fundamentals.get(symbol, {}).get("fundamentals_signal", ""),
                    cio_memo=cio.get(symbol, {}).get("memo", ""),
                    current_price=market.get(symbol, {}).get("latest_close", 0),
                )

                prediction_log[symbol] = {
                    "action":      action,
                    "entry_price": r.get("entry_price", 0),
                    "stop_loss":   r.get("stop_loss", 0),
                    "take_profit": r.get("take_profit", 0),
                    "vote_score":  r.get("vote_score", 0),
                    "logged_id":   logged_id,
                }

                # ── Telegram alert for actionable signals ──────────────────
                if action in ("BUY", "SELL"):
                    cio_memo = cio.get(symbol, {}).get("memo", "")
                    emoji = "🚀" if action == "BUY" else "📉"
                    msg = (
                        f"{emoji} <b>{action} {symbol}</b>\n"
                        f"Entry: {r.get('entry_price', 0):.2f}\n"
                        f"SL: {r.get('stop_loss', 0):.2f} | TP: {r.get('take_profit', 0):.2f}\n"
                        f"Score: {r.get('vote_score', 0):.2f}\n"
                        f"Confidence: {quant.get(symbol, {}).get('quant_confidence', 0):.1%}\n"
                        f"Memo: {cio_memo}"
                    )
                    send_telegram_message_sync(msg)

                logger.info(
                    f"[PredictionLogger] {symbol}: {action} "
                    f"(score={r.get('vote_score', 0):.3f})"
                )

            except Exception as exc:
                logger.error(f"[PredictionLogger] Error for {symbol}: {exc}", exc_info=True)
                prediction_log[symbol] = {
                    "action": "ERROR",
                    "entry_price": 0, "stop_loss": 0, "take_profit": 0,
                    "vote_score": 0, "logged_id": None,
                }

        context["prediction_log"] = prediction_log
        return context
