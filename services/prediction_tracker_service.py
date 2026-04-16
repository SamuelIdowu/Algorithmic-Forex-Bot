"""
PredictionTrackerService — manages the lifecycle of trading predictions.

Responsibilities:
  1. Create tracker entries from prediction results
  2. Check current prices and evaluate outcomes (TP/SL hits)
  3. Batch-update all active trackers
  4. Evaluate directional accuracy on expired trackers
  5. Format human-readable status/summary messages
"""

import json
import logging
from datetime import datetime
from typing import Optional

import yfinance as yf

from data.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ── Symbol formatting for yfinance ──────────────────────────────────────────

def format_symbol(symbol: str) -> str:
    """Convert forex symbols like EUR/USD → EURUSD=X for yfinance."""
    if "/" in symbol:
        return symbol.replace("/", "") + "=X"
    return symbol


def get_current_price(symbol: str) -> Optional[float]:
    """Fetch the latest close price via yfinance."""
    try:
        yf_symbol = format_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        # Fallback: try 1-day with 1h interval
        hist = ticker.history(period="5d", interval="1h")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"Failed to fetch price for {symbol}: {e}")
    return None


# ── Helper: percentage distance calculations ────────────────────────────────

def _tp_progress(entry: float, current: float, tp: float, direction: str) -> float:
    """How far price has traveled toward take-profit (0-100+%)."""
    if direction in ("UP", "BUY"):
        denom = tp - entry
    else:
        denom = entry - tp
    if denom == 0:
        return 100.0
    if direction in ("UP", "BUY"):
        return (current - entry) / denom * 100
    else:
        return (entry - current) / denom * 100


def _sl_distance(entry: float, current: float, sl: float, direction: str) -> float:
    """How far price is from stop-loss (0 = at SL, 100 = at entry)."""
    if direction in ("UP", "BUY"):
        denom = entry - sl
    else:
        denom = sl - entry
    if denom == 0:
        return 100.0
    if direction in ("UP", "BUY"):
        return (current - sl) / denom * 100
    else:
        return (sl - current) / denom * 100


def _pnl_percent(entry: float, current: float, direction: str) -> float:
    """Calculate PnL percentage based on direction."""
    if entry == 0:
        return 0.0
    if direction in ("UP", "BUY"):
        return (current - entry) / entry * 100
    else:
        return (entry - current) / entry * 100


# ── Service class ───────────────────────────────────────────────────────────

class PredictionTrackerService:
    """Business-logic layer for prediction tracker lifecycle management."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        if db_manager is None:
            self.db = DatabaseManager()
        else:
            self.db = db_manager

    # ─────────────────────────────────────────────────────────────────────
    # 1. Create tracker from prediction result
    # ─────────────────────────────────────────────────────────────────────

    def create_tracker_from_prediction(
        self,
        prediction_result: dict,
        source: str = "standalone",
        prediction_id: int = None,
    ) -> int:
        """
        Take the result dict from predict_next_movement() and create a tracker.

        Returns the new tracker_id, or -1 on failure.
        """
        try:
            symbol = prediction_result["symbol"]
            direction = prediction_result["prediction"]  # "UP" or "DOWN"
            timeframe = prediction_result.get("interval", "1h")
            entry = prediction_result["entry"]
            tp = prediction_result.get("tp")
            sl = prediction_result.get("sl")
            confidence = prediction_result.get("confidence", 0.0)
            rsi = prediction_result.get("rsi")
            atr = prediction_result.get("atr")
            bb_pos = prediction_result.get("bb_pos")
            reasoning = prediction_result.get("reasoning", "")

            # Extract agent insights if available
            agent_insights = prediction_result.get("agent_insights")
            agent_insights_json = None
            quant_confidence = None
            vote_score = None

            if agent_insights:
                agent_insights_json = json.dumps(agent_insights)
                quant_confidence = agent_insights.get("quant", {}).get("confidence")
                vote_score = agent_insights.get("risk_manager", {}).get("vote_score")

            # Determine action from risk manager if present
            action = prediction_result.get("action", direction)

            tracker_data = {
                "prediction_id": prediction_id,
                "source": source,
                "symbol": symbol,
                "direction": direction,
                "timeframe": timeframe,
                "entry_price": entry,
                "take_profit": tp,
                "stop_loss": sl,
                "status": "ACTIVE",
                "current_price": entry,
                "pnl_percent": 0.0,
                "rsi": rsi,
                "atr": atr,
                "bb_pos": bb_pos,
                "ml_confidence": confidence,
                "quant_confidence": quant_confidence,
                "vote_score": vote_score,
                "reasoning": reasoning,
                "agent_insights_json": agent_insights_json,
            }

            tracker_id = self.db.create_prediction_tracker(tracker_data)
            if tracker_id > 0:
                logger.info(
                    f"Created tracker #{tracker_id} for {symbol} "
                    f"direction={direction} entry={entry}"
                )
            return tracker_id

        except KeyError as e:
            logger.error(f"Missing required field in prediction_result: {e}")
            return -1
        except Exception as e:
            logger.error(f"Error creating tracker from prediction: {e}")
            return -1

    # ─────────────────────────────────────────────────────────────────────
    # 2. Check single prediction
    # ─────────────────────────────────────────────────────────────────────

    def check_prediction(self, tracker_id: int) -> dict:
        """
        Fetch current price for a tracker and evaluate outcomes.

        Returns dict with:
          {tracker_id, status, current_price, pnl_percent, outcome}
        """
        tracker = self.db.get_tracker_by_id(tracker_id)
        if not tracker:
            logger.warning(f"Tracker #{tracker_id} not found")
            return {
                "tracker_id": tracker_id,
                "status": "NOT_FOUND",
                "current_price": None,
                "pnl_percent": None,
                "outcome": None,
            }

        current_status = tracker["status"]
        if current_status != "ACTIVE":
            return {
                "tracker_id": tracker_id,
                "status": current_status,
                "current_price": tracker.get("current_price"),
                "pnl_percent": tracker.get("pnl_percent"),
                "outcome": "ALREADY_RESOLVED",
            }

        symbol = tracker["symbol"]
        entry = tracker["entry_price"]
        tp = tracker.get("take_profit")
        sl = tracker.get("stop_loss")
        direction = tracker["direction"]

        # Fetch current price
        current_price = get_current_price(symbol)
        if current_price is None:
            logger.warning(f"Could not fetch price for {symbol}")
            return {
                "tracker_id": tracker_id,
                "status": current_status,
                "current_price": None,
                "pnl_percent": None,
                "outcome": "PRICE_FETCH_FAILED",
            }

        # Calculate PnL
        pnl = _pnl_percent(entry, current_price, direction)

        # Update max profit / max loss extremes
        self.db.update_tracker_extremes(tracker_id, current_price)

        # Determine outcome
        outcome = None
        new_status = current_status

        if direction in ("UP", "BUY"):
            tp_hit = current_price >= tp if tp else False
            sl_hit = current_price <= sl if sl else False
        else:
            # DOWN / SELL — TP is below, SL is above
            tp_hit = current_price <= tp if tp else False
            sl_hit = current_price >= sl if sl else False

        if tp_hit:
            outcome = "TP_HIT"
            new_status = "WON_TP"
            self.db.update_tracker_outcome(tracker_id, "tp_hit", current_price)
            logger.info(
                f"Tracker #{tracker_id} TP HIT — {symbol} @ {current_price:.5f}"
            )
        elif sl_hit:
            outcome = "SL_HIT"
            new_status = "LOST_SL"
            self.db.update_tracker_outcome(tracker_id, "sl_hit", current_price)
            logger.info(
                f"Tracker #{tracker_id} SL HIT — {symbol} @ {current_price:.5f}"
            )
        else:
            # Still active — update price and PnL
            self.db.update_tracker_status(tracker_id, current_status, current_price, pnl)

        # Log price check
        self.db.log_price_check(tracker_id, current_price, pnl, new_status)

        return {
            "tracker_id": tracker_id,
            "status": new_status,
            "current_price": current_price,
            "pnl_percent": round(pnl, 4),
            "outcome": outcome,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 3. Batch update all active trackers
    # ─────────────────────────────────────────────────────────────────────

    def update_all_active_trackers(self) -> dict:
        """
        Check every active tracker and collect results.

        Returns summary:
          {total_checked, outcomes_found: {tp_hits, sl_hits, unchanged}}
        """
        trackers_df = self.db.get_active_trackers()
        if trackers_df.empty:
            return {
                "total_checked": 0,
                "outcomes_found": {"tp_hits": 0, "sl_hits": 0, "unchanged": 0},
            }

        tp_hits = 0
        sl_hits = 0
        unchanged = 0

        for _, row in trackers_df.iterrows():
            tracker_id = row["id"]
            result = self.check_prediction(tracker_id)
            outcome = result.get("outcome")

            if outcome == "TP_HIT":
                tp_hits += 1
            elif outcome == "SL_HIT":
                sl_hits += 1
            else:
                unchanged += 1

        total = tp_hits + sl_hits + unchanged
        logger.info(
            f"Batch check complete: {total} checked, "
            f"{tp_hits} TP hits, {sl_hits} SL hits, {unchanged} unchanged"
        )

        return {
            "total_checked": total,
            "outcomes_found": {
                "tp_hits": tp_hits,
                "sl_hits": sl_hits,
                "unchanged": unchanged,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # 4. Evaluate expired trackers
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_expired_trackers(self) -> dict:
        """
        Find ACTIVE trackers where expires_at < now, fetch final price,
        and evaluate directional accuracy.

        Returns: {evaluated_count, directional_wins, directional_losses}
        """
        trackers_df = self.db.get_active_trackers()
        if trackers_df.empty:
            return {
                "evaluated_count": 0,
                "directional_wins": 0,
                "directional_losses": 0,
            }

        now = datetime.now()
        evaluated = 0
        wins = 0
        losses = 0

        for _, row in trackers_df.iterrows():
            tracker_id = row["id"]
            expires_at_str = row.get("expires_at")
            if not expires_at_str:
                continue

            try:
                expires_at = datetime.fromisoformat(expires_at_str)
            except (ValueError, TypeError):
                continue

            if expires_at >= now:
                continue  # Not yet expired

            # Fetch final price
            symbol = row["symbol"]
            final_price = get_current_price(symbol)
            if final_price is None:
                logger.warning(
                    f"Cannot evaluate expired tracker #{tracker_id} — "
                    f"price fetch failed for {symbol}"
                )
                continue

            entry = row["entry_price"]
            direction = row["direction"]

            # Directional accuracy
            if direction in ("UP", "BUY"):
                is_win = final_price > entry
            else:
                is_win = final_price < entry

            if is_win:
                self.db.update_tracker_outcome(tracker_id, "direction_win", final_price)
                wins += 1
                logger.info(
                    f"Tracker #{tracker_id} DIRECTION WIN — {symbol} "
                    f"entry={entry} final={final_price:.5f}"
                )
            else:
                self.db.update_tracker_outcome(tracker_id, "direction_loss", final_price)
                losses += 1
                logger.info(
                    f"Tracker #{tracker_id} DIRECTION LOSS — {symbol} "
                    f"entry={entry} final={final_price:.5f}"
                )

            # Log price check
            pnl = _pnl_percent(entry, final_price, direction)
            self.db.log_price_check(
                tracker_id, final_price, pnl,
                "WON_DIRECTION" if is_win else "EXPIRED"
            )
            evaluated += 1

        logger.info(
            f"Expired evaluation: {evaluated} evaluated, "
            f"{wins} wins, {losses} losses"
        )

        return {
            "evaluated_count": evaluated,
            "directional_wins": wins,
            "directional_losses": losses,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 5. Human-readable status text
    # ─────────────────────────────────────────────────────────────────────

    def get_prediction_status_text(self, tracker_id: int) -> str:
        """
        Format a human-readable status message for a single tracker.
        """
        tracker = self.db.get_tracker_by_id(tracker_id)
        if not tracker:
            return f"❌ Tracker #{tracker_id} not found."

        symbol = tracker["symbol"]
        status = tracker["status"]
        direction = tracker["direction"]
        timeframe = tracker["timeframe"]
        entry = tracker["entry_price"]
        tp = tracker.get("take_profit")
        sl = tracker.get("stop_loss")
        current = tracker.get("current_price")
        max_profit = tracker.get("max_profit_reached")
        max_loss = tracker.get("max_loss_reached")
        expires_at_str = tracker.get("expires_at")

        # Build direction emoji
        dir_emoji = "📈" if direction in ("UP", "BUY") else "📉"

        lines = [
            f"{dir_emoji} {symbol} Prediction #{tracker_id}",
            f"Status: {status}",
            f"Direction: {direction} | Timeframe: {timeframe}",
        ]

        # Current price & PnL
        if current is not None:
            pnl = _pnl_percent(entry, current, direction)
            pnl_sign = "+" if pnl >= 0 else ""
            lines.append(
                f"Entry: ${entry:.4f} | Current: ${current:.4f} ({pnl_sign}{pnl:.2f}%)"
            )
        else:
            lines.append(f"Entry: ${entry:.4f} | Current: N/A")

        # TP / SL progress
        if tp is not None and current is not None:
            prog = _tp_progress(entry, current, tp, direction)
            lines.append(f"TP: ${tp:.4f} ({prog:.0f}% reached)")
        elif tp is not None:
            lines.append(f"TP: ${tp:.4f}")

        if sl is not None and current is not None:
            dist = _sl_distance(entry, current, sl, direction)
            lines.append(f"SL: ${sl:.4f} ({dist:.0f}% away)")
        elif sl is not None:
            lines.append(f"SL: ${sl:.4f}")

        # Extremes
        if max_profit is not None:
            lines.append(f"Max Profit: ${max_profit:.4f}")
        if max_loss is not None:
            lines.append(f"Max Loss: ${max_loss:.4f}")

        # Expiry
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                delta = expires_at - datetime.now()
                if delta.total_seconds() > 0:
                    hours = int(delta.total_seconds() // 3600)
                    minutes = int((delta.total_seconds() % 3600) // 60)
                    if hours > 0:
                        lines.append(f"Expires in: {hours} hours {minutes} min")
                    else:
                        lines.append(f"Expires in: {minutes} minutes")
                else:
                    lines.append("⏰ Expired")
            except (ValueError, TypeError):
                pass

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # 6. Summary of all active predictions
    # ─────────────────────────────────────────────────────────────────────

    def get_active_predictions_summary(self, symbol: str = None) -> str:
        """
        Format a summary of all active predictions, optionally filtered by symbol.
        """
        trackers_df = self.db.get_active_trackers(symbol)
        if trackers_df.empty:
            if symbol:
                return f"📈 No active predictions for {symbol}."
            return "📈 No active predictions."

        # Group by symbol
        grouped = {}
        for _, row in trackers_df.iterrows():
            sym = row["symbol"]
            if sym not in grouped:
                grouped[sym] = []
            grouped[sym].append(row)

        total = len(trackers_df)
        lines = [f"📈 Active Predictions ({total} total)", ""]

        for sym, rows in grouped.items():
            dir_emoji = "📈"
            lines.append(f"{sym}: {len(rows)} active")

            for row in rows:
                tid = row["id"]
                direction = row["direction"]
                entry = row["entry_price"]
                current = row.get("current_price")
                tp = row.get("take_profit")

                if direction in ("DOWN", "SELL"):
                    dir_emoji = "📉"
                else:
                    dir_emoji = "📈"

                if current is not None:
                    pnl = _pnl_percent(entry, current, direction)
                    pnl_sign = "+" if pnl >= 0 else ""
                    current_str = f"→ {current:.4f} ({pnl_sign}{pnl:.2f}%)"
                else:
                    current_str = "→ N/A"

                tp_str = f" [TP: {tp:.4f}]" if tp else ""

                lines.append(
                    f"  {dir_emoji} #{tid} {direction} @ {entry:.4f} {current_str}{tp_str}"
                )

            lines.append("")

        return "\n".join(lines)
