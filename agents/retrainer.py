"""
Retrainer Agent — Phase 9 (Layer 1 Self-Improvement)

Checks per-symbol win rates and model ages against configured thresholds.
When a retrain is triggered it:
  1. Calls train_model.train_model() programmatically for that symbol
  2. Hot-swaps the fresh .pkl into QuantAnalyst without restarting the loop
  3. Logs the event to the retraining_log table

Trigger conditions (any one suffices):
  • Win rate < RETRAIN_WIN_RATE_THRESHOLD over last 20 closed trades
  • Model age > RETRAIN_MAX_AGE_DAYS since last successful retrain
  • Manual flag: context["force_retrain"] == symbol

Runs last in each cycle (priority=9) — the "Reflect" step.
"""
import logging
import os
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from utils.config import (
    RETRAIN_WIN_RATE_THRESHOLD,
    RETRAIN_MAX_AGE_DAYS,
    RETRAIN_BOOTSTRAP,
    AGENT_INTERVAL_MINUTES,
)
from utils.ml_utils import get_model_paths, get_model_interval, MODELS_DIR

logger = logging.getLogger(__name__)


# Fallback paths (for when symbol-specific paths aren't found)
_FALLBACK_PATHS = {
    "model_path":  "models/ml_strategy_model.pkl",
    "scaler_path": "models/ml_strategy_scaler.pkl",
}


class RetrainerAgent(BaseAgent):
    """
    Self-retraining module. Runs at the end of every cycle (priority=9).
    """

    name = "RetrainerAgent"
    role = "monitor"
    priority = 9   # Runs last — "Reflect" step

    def __init__(self):
        self._db = None
        self._quant_agent = None   # set lazily to avoid circular imports

    def _get_db(self):
        if self._db is None:
            from data.db_manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def run(self, context: dict) -> dict:
        symbols = context.get("symbols", [])
        force   = context.get("force_retrain")   # set by --retrain CLI flag

        for symbol in symbols:
            try:
                self._check_and_retrain(symbol, forced=(force == symbol), context=context)
            except Exception as exc:
                logger.error(f"[RetrainerAgent] Error for {symbol}: {exc}", exc_info=True)

        return context

    # ──────────────────────────────────────────────────────────────────────────

    def _check_and_retrain(self, symbol: str, forced: bool, context: dict):
        db = self._get_db()

        # ── Win rate check ────────────────────────────────────────────────
        recent = db.get_recent_trades(symbol, limit=20)
        win_rate = 0.5   # default assumption

        if len(recent) >= 5:
            wins = (recent["pnl"] > 0).sum()
            win_rate = float(wins) / len(recent)
            logger.debug(f"[RetrainerAgent] {symbol}: win_rate={win_rate:.2f} over {len(recent)} trades")

        win_rate_trigger = win_rate < RETRAIN_WIN_RATE_THRESHOLD and len(recent) >= 10

        # ── Model age check ───────────────────────────────────────────────
        last_retrain = db.get_last_retrain_date(symbol)
        model_path, _ = self._get_paths(symbol)
        file_exists = os.path.exists(model_path)

        model_age_days = 9999
        age_trigger = False

        if not file_exists and RETRAIN_BOOTSTRAP:
            age_trigger = True
            logger.info(f"[RetrainerAgent] {symbol}: model file missing, triggering bootstrap retrain.")
        elif last_retrain:
            model_age_days = (datetime.utcnow() - last_retrain).days
            age_trigger = model_age_days > RETRAIN_MAX_AGE_DAYS
        elif file_exists:
            # Use file mtime as fallback if no DB record
            mtime = datetime.utcfromtimestamp(os.path.getmtime(model_path))
            model_age_days = (datetime.utcnow() - mtime).days
            age_trigger = model_age_days > RETRAIN_MAX_AGE_DAYS

        if not (forced or win_rate_trigger or age_trigger):
            logger.debug(
                f"[RetrainerAgent] {symbol}: no retrain needed "
                f"(win_rate={win_rate:.2f}, age={model_age_days}d)"
            )
            return

        trigger_reason = (
            "forced" if forced
            else ("win_rate" if win_rate_trigger else "model_age")
        )
        logger.info(
            f"[RetrainerAgent] 🔄 Retraining {symbol} "
            f"(trigger={trigger_reason}, win_rate={win_rate:.2f}, age={model_age_days}d)"
        )

        success = self._retrain(symbol, context)

        db.log_retraining(
            symbol=symbol,
            trigger=trigger_reason,
            win_rate=win_rate,
            model_age_days=model_age_days,
            success=success,
            notes=f"Forced={forced}",
        )

    def _retrain(self, symbol: str, context: dict) -> bool:
        """Call train_model.train_model() and hot-reload into QuantAnalyst."""
        from utils.config import AGENT_INTERVAL_MINUTES
        interval = get_model_interval(AGENT_INTERVAL_MINUTES)
        model_path, scaler_path = get_model_paths(symbol, interval)

        try:
            from train_model import train_model

            # Use 5 years of data for training (daily) or 60 days (intraday)
            days = 5 * 365 if interval == "1d" else 59
            start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date   = datetime.utcnow().strftime("%Y-%m-%d")

            logger.info(f"[RetrainerAgent] Training {symbol} on {interval} candles ({start_date} → {end_date}) ...")
            result = train_model(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                model_path=model_path,
                scaler_path=scaler_path,
                tune=False,
                interval=interval,
            )

            if result is None:
                logger.error(f"[RetrainerAgent] train_model returned None for {symbol}")
                return False

            logger.info(f"[RetrainerAgent] Training complete for {symbol}. Accuracy: {result[2]:.4f}")

        except Exception as exc:
            logger.error(f"[RetrainerAgent] Training failed for {symbol}: {exc}", exc_info=True)
            return False

        # ── Hot-swap into QuantAnalyst ────────────────────────────────────
        try:
            # Find QuantAnalyst instance in context agents list
            quant_ref = None
            for agent in context.get("_agents", []):
                if agent.name == "QuantAnalyst":
                    quant_ref = agent
                    break

            if quant_ref is None:
                # Fallback: create instance and call reload
                from agents.quant_analyst import QuantAnalyst
                quant_ref = QuantAnalyst()

            success = quant_ref.reload_model(symbol)
            if success:
                logger.info(f"[RetrainerAgent] ✅ Hot-reload successful for {symbol}")
            else:
                logger.warning(f"[RetrainerAgent] Hot-reload returned False for {symbol}")
            return success

        except Exception as exc:
            logger.error(f"[RetrainerAgent] Hot-reload failed for {symbol}: {exc}", exc_info=True)
            return False
    def _get_paths(self, symbol: str) -> tuple[str, str]:
        """Resolve model and scaler paths using standard naming convention."""
        from utils.config import AGENT_INTERVAL_MINUTES
        interval = get_model_interval(AGENT_INTERVAL_MINUTES)
        return get_model_paths(symbol, interval)
