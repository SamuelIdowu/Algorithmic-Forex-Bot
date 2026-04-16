"""
Analytics Service — Comprehensive Win Rate & Performance Analytics

Provides high-level reporting for prediction tracker performance across
multiple dimensions: symbols, timeframes, confidence ranges, and time trends.
"""
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from data.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Aggregates and analyzes prediction tracker data for reporting.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager or DatabaseManager()

    def get_comprehensive_report(self, symbol: str = None, time_range_days: int = 30) -> dict:
        """
        Returns a comprehensive analytics report with all metrics.
        """
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=time_range_days)
            start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

            # Base WHERE clause
            base_where = "WHERE created_at >= ? AND created_at <= ?"
            base_params = [start_str, end_str]

            if symbol:
                base_where += " AND symbol = ?"
                base_params.append(symbol)

            # ── Overview Stats ─────────────────────────────────────
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed
                FROM prediction_tracker {base_where}
            """, base_params)

            overview_row = cursor.fetchone()
            total = overview_row["total"]
            active = overview_row["active"]
            completed = overview_row["completed"]
            completion_rate = completed / total if total > 0 else 0

            # ── Win Rates ─────────────────────────────────────────
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins,
                    SUM(CASE WHEN status = 'LOST_SL' THEN 1 ELSE 0 END) as lost_sl,
                    SUM(CASE WHEN status IN ('EXPIRED', 'LOSS') AND direction_result = 'LOSS' THEN 1 ELSE 0 END) as dir_losses
                FROM prediction_tracker {base_where}
            """, base_params)

            wr_row = cursor.fetchone()
            tp_hits = wr_row["tp_hits"]
            dir_wins = wr_row["dir_wins"]
            lost_sl = wr_row["lost_sl"]

            tp_hit_rate = tp_hits / completed if completed > 0 else 0
            directional_accuracy = dir_wins / completed if completed > 0 else 0
            combined_score = (tp_hit_rate + directional_accuracy) / 2 if completed > 0 else 0

            # ── Performance Stats ─────────────────────────────────
            cursor.execute(f"""
                SELECT 
                    AVG(pnl_percent) as avg_pnl,
                    MAX(pnl_percent) as best_trade,
                    MIN(pnl_percent) as worst_trade
                FROM prediction_tracker 
                {base_where} AND pnl_percent IS NOT NULL AND status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL')
            """, base_params)

            perf_row = cursor.fetchone()
            avg_pnl = perf_row["avg_pnl"] or 0
            best_trade = perf_row["best_trade"] or 0
            worst_trade = perf_row["worst_trade"] or 0

            # Calculate avg win / avg loss
            cursor.execute(f"""
                SELECT 
                    AVG(CASE WHEN pnl_percent > 0 THEN pnl_percent END) as avg_win,
                    AVG(CASE WHEN pnl_percent < 0 THEN pnl_percent END) as avg_loss
                FROM prediction_tracker 
                {base_where} AND pnl_percent IS NOT NULL AND status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL')
            """, base_params)

            win_loss_row = cursor.fetchone()
            avg_win = win_loss_row["avg_win"] or 0
            avg_loss = win_loss_row["avg_loss"] or 0

            # Profit factor
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            # Avg time to outcome
            cursor.execute(f"""
                SELECT AVG(
                    CASE 
                        WHEN tp_hit_at IS NOT NULL THEN (julianday(tp_hit_at) - julianday(created_at)) * 24
                        WHEN sl_hit_at IS NOT NULL THEN (julianday(sl_hit_at) - julianday(created_at)) * 24
                        WHEN direction_check_at IS NOT NULL THEN (julianday(direction_check_at) - julianday(created_at)) * 24
                        ELSE NULL
                    END
                ) as avg_hours
                FROM prediction_tracker 
                {base_where} AND status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL')
            """, base_params)

            time_row = cursor.fetchone()
            avg_time_hours = time_row["avg_hours"] or 0

            # ── By Symbol ──────────────────────────────────────────
            cursor.execute(f"""
                SELECT symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins,
                    AVG(pnl_percent) as avg_pnl
                FROM prediction_tracker {base_where}
                GROUP BY symbol
                ORDER BY avg_pnl DESC
            """, base_params)

            by_symbol = {}
            for row in cursor.fetchall():
                sym = row["symbol"]
                sym_completed = row["completed"]
                by_symbol[sym] = {
                    "total": row["total"],
                    "completed": sym_completed,
                    "active": row["active"],
                    "tp_hit_rate": row["tp_hits"] / sym_completed if sym_completed > 0 else 0,
                    "directional_accuracy": row["dir_wins"] / sym_completed if sym_completed > 0 else 0,
                    "avg_pnl": round(row["avg_pnl"], 2) if row["avg_pnl"] else 0
                }

            # ── By Timeframe ──────────────────────────────────────
            cursor.execute(f"""
                SELECT timeframe,
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins
                FROM prediction_tracker {base_where}
                GROUP BY timeframe
                ORDER BY total DESC
            """, base_params)

            by_timeframe = {}
            for row in cursor.fetchall():
                tf = row["timeframe"]
                tf_completed = row["completed"]
                by_timeframe[tf] = {
                    "total": row["total"],
                    "tp_hit_rate": row["tp_hits"] / tf_completed if tf_completed > 0 else 0,
                    "directional_accuracy": row["dir_wins"] / tf_completed if tf_completed > 0 else 0
                }

            # ── By Confidence Range ───────────────────────────────
            by_confidence = {
                "high (>80%)": self._get_confidence_bucket(base_where, base_params, 0.8, 1.0),
                "medium (60-80%)": self._get_confidence_bucket(base_where, base_params, 0.6, 0.8),
                "low (<60%)": self._get_confidence_bucket(base_where, base_params, 0, 0.6)
            }

            conn.close()

            # ── Top Performers ────────────────────────────────────
            top_performers = sorted(
                [{"symbol": s, "win_rate": v["tp_hit_rate"], "count": v["completed"]} 
                 for s, v in by_symbol.items() if v["completed"] >= 5],
                key=lambda x: x["win_rate"],
                reverse=True
            )[:3]

            return {
                "period": {
                    "start_date": start_str,
                    "end_date": end_str,
                    "days": time_range_days
                },
                "overview": {
                    "total_predictions": total,
                    "active": active,
                    "completed": completed,
                    "completion_rate": round(completion_rate, 3)
                },
                "win_rates": {
                    "tp_hit_rate": round(tp_hit_rate, 3),
                    "tp_hits": tp_hits,
                    "directional_accuracy": round(directional_accuracy, 3),
                    "directional_wins": dir_wins,
                    "combined_score": round(combined_score, 3)
                },
                "performance": {
                    "avg_pnl_percent": round(avg_pnl, 2),
                    "best_trade_percent": round(best_trade, 2),
                    "worst_trade_percent": round(worst_trade, 2),
                    "profit_factor": round(profit_factor, 2),
                    "avg_win_percent": round(avg_win, 2),
                    "avg_loss_percent": round(avg_loss, 2),
                    "avg_time_to_outcome_hours": round(avg_time_hours, 1)
                },
                "by_symbol": by_symbol,
                "by_timeframe": by_timeframe,
                "by_confidence_range": by_confidence,
                "top_performers": top_performers
            }

        except Exception as e:
            logger.error(f"[Analytics] Comprehensive report failed: {e}", exc_info=True)
            return {}

    def _get_confidence_bucket(self, base_where: str, base_params: list, 
                                min_conf: float, max_conf: float) -> dict:
        """Get stats for a confidence range."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check for ml_confidence first, fallback to quant_confidence
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins,
                    AVG(pnl_percent) as avg_pnl
                FROM prediction_tracker 
                {base_where}
                AND (
                    (ml_confidence IS NOT NULL AND ml_confidence >= ? AND ml_confidence < ?)
                    OR (quant_confidence IS NOT NULL AND quant_confidence >= ? AND quant_confidence < ?)
                )
            """, base_params + [min_conf, max_conf, min_conf, max_conf])

            row = cursor.fetchone()
            conn.close()

            if not row or row["total"] == 0:
                return {"total": 0, "tp_hit_rate": 0, "directional_accuracy": 0, "avg_pnl": 0}

            completed = row["completed"]
            return {
                "total": row["total"],
                "tp_hit_rate": row["tp_hits"] / completed if completed > 0 else 0,
                "directional_accuracy": row["dir_wins"] / completed if completed > 0 else 0,
                "avg_pnl": round(row["avg_pnl"], 2) if row["avg_pnl"] else 0
            }
        except Exception as e:
            logger.error(f"[Analytics] Confidence bucket failed: {e}")
            return {"total": 0, "tp_hit_rate": 0, "directional_accuracy": 0, "avg_pnl": 0}

    def get_symbol_breakdown(self, time_range_days: int = 30) -> list[dict]:
        """
        Returns per-symbol stats, sorted by win rate descending.
        """
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            start_date = (datetime.utcnow() - timedelta(days=time_range_days)).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                SELECT symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins,
                    AVG(pnl_percent) as avg_pnl,
                    MAX(pnl_percent) as best_trade,
                    MIN(pnl_percent) as worst_trade,
                    AVG(
                        CASE 
                            WHEN tp_hit_at IS NOT NULL THEN (julianday(tp_hit_at) - julianday(created_at)) * 24
                            WHEN sl_hit_at IS NOT NULL THEN (julianday(sl_hit_at) - julianday(created_at)) * 24
                            WHEN direction_check_at IS NOT NULL THEN (julianday(direction_check_at) - julianday(created_at)) * 24
                            ELSE NULL
                        END
                    ) as avg_time_hours
                FROM prediction_tracker 
                WHERE created_at >= ?
                GROUP BY symbol
                ORDER BY 
                    CASE WHEN SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) > 0
                    THEN CAST(SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) AS REAL) / 
                         SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END)
                    ELSE 0 END DESC
            """, (start_date,))

            results = []
            for row in cursor.fetchall():
                completed = row["completed"]
                results.append({
                    "symbol": row["symbol"],
                    "total_predictions": row["total"],
                    "completed": completed,
                    "active": row["active"],
                    "tp_hit_rate": round(row["tp_hits"] / completed, 3) if completed > 0 else 0,
                    "directional_accuracy": round(row["dir_wins"] / completed, 3) if completed > 0 else 0,
                    "avg_pnl": round(row["avg_pnl"], 2) if row["avg_pnl"] else 0,
                    "best_trade": round(row["best_trade"], 2) if row["best_trade"] else 0,
                    "worst_trade": round(row["worst_trade"], 2) if row["worst_trade"] else 0,
                    "avg_time_to_outcome_hours": round(row["avg_time_hours"], 1) if row["avg_time_hours"] else 0
                })

            conn.close()
            return results

        except Exception as e:
            logger.error(f"[Analytics] Symbol breakdown failed: {e}")
            return []

    def get_timeframe_performance(self, time_range_days: int = 30) -> dict:
        """Returns performance breakdown by timeframe."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            start_date = (datetime.utcnow() - timedelta(days=time_range_days)).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                SELECT timeframe,
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                    SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins,
                    AVG(pnl_percent) as avg_pnl,
                    AVG(
                        CASE 
                            WHEN tp_hit_at IS NOT NULL THEN (julianday(tp_hit_at) - julianday(created_at)) * 24
                            WHEN sl_hit_at IS NOT NULL THEN (julianday(sl_hit_at) - julianday(created_at)) * 24
                            WHEN direction_check_at IS NOT NULL THEN (julianday(direction_check_at) - julianday(created_at)) * 24
                            ELSE NULL
                        END
                    ) as avg_time_hours
                FROM prediction_tracker 
                WHERE created_at >= ?
                GROUP BY timeframe
                ORDER BY total DESC
            """, (start_date,))

            results = {}
            for row in cursor.fetchall():
                tf = row["timeframe"]
                completed = row["completed"]
                results[tf] = {
                    "total": row["total"],
                    "tp_hit_rate": round(row["tp_hits"] / completed, 3) if completed > 0 else 0,
                    "directional_accuracy": round(row["dir_wins"] / completed, 3) if completed > 0 else 0,
                    "avg_pnl": round(row["avg_pnl"], 2) if row["avg_pnl"] else 0,
                    "avg_time_hours": round(row["avg_time_hours"], 1) if row["avg_time_hours"] else 0
                }

            conn.close()
            return results

        except Exception as e:
            logger.error(f"[Analytics] Timeframe performance failed: {e}")
            return {}

    def get_confidence_analysis(self, time_range_days: int = 30) -> dict:
        """Analyzes how confidence correlates with outcomes."""
        try:
            buckets = {
                "high_confidence": {
                    "range": "80-100%",
                    "min": 0.8, "max": 1.0
                },
                "medium_confidence": {
                    "range": "60-80%",
                    "min": 0.6, "max": 0.8
                },
                "low_confidence": {
                    "range": "0-60%",
                    "min": 0.0, "max": 0.6
                }
            }

            start_date = (datetime.utcnow() - timedelta(days=time_range_days)).strftime("%Y-%m-%d %H:%M:%S")

            for key, bucket in buckets.items():
                stats = self._get_confidence_bucket(
                    "WHERE created_at >= ?",
                    [start_date],
                    bucket["min"],
                    bucket["max"]
                )
                buckets[key].update(stats)

            # Calculate correlation (simplified: check if higher confidence = higher win rate)
            high_wr = buckets["high_confidence"].get("tp_hit_rate", 0)
            med_wr = buckets["medium_confidence"].get("tp_hit_rate", 0)
            low_wr = buckets["low_confidence"].get("tp_hit_rate", 0)

            # Perfect positive correlation = 1.0, inverse = 0.0
            if high_wr >= med_wr >= low_wr:
                correlation = 1.0
            elif high_wr <= med_wr <= low_wr:
                correlation = 0.0
            else:
                correlation = 0.5  # Partial/no correlation

            buckets["correlation"] = correlation
            return buckets

        except Exception as e:
            logger.error(f"[Analytics] Confidence analysis failed: {e}")
            return {}

    def get_recent_outcomes(self, limit: int = 10) -> list[dict]:
        """Returns most recent completed predictions."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    id as tracker_id, symbol, direction, status,
                    entry_price, final_price_at_outcome as exit_price,
                    pnl_percent, timeframe, created_at,
                    CASE 
                        WHEN tp_hit_at IS NOT NULL THEN tp_hit_at
                        WHEN sl_hit_at IS NOT NULL THEN sl_hit_at
                        WHEN direction_check_at IS NOT NULL THEN direction_check_at
                    END as outcome_at,
                    CASE 
                        WHEN tp_hit_at IS NOT NULL THEN (julianday(tp_hit_at) - julianday(created_at)) * 24
                        WHEN sl_hit_at IS NOT NULL THEN (julianday(sl_hit_at) - julianday(created_at)) * 24
                        WHEN direction_check_at IS NOT NULL THEN (julianday(direction_check_at) - julianday(created_at)) * 24
                    END as duration_hours
                FROM prediction_tracker 
                WHERE status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL')
                ORDER BY 
                    COALESCE(tp_hit_at, sl_hit_at, direction_check_at, created_at) DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "tracker_id": row["tracker_id"],
                    "symbol": row["symbol"],
                    "direction": row["direction"],
                    "status": row["status"],
                    "entry_price": row["entry_price"],
                    "exit_price": row["exit_price"],
                    "pnl_percent": round(row["pnl_percent"], 2) if row["pnl_percent"] else 0,
                    "timeframe": row["timeframe"],
                    "created_at": row["created_at"],
                    "outcome_at": row["outcome_at"],
                    "duration_hours": round(row["duration_hours"], 1) if row["duration_hours"] else 0
                })

            conn.close()
            return results

        except Exception as e:
            logger.error(f"[Analytics] Recent outcomes failed: {e}")
            return []

    def get_trend_analysis(self) -> dict:
        """Analyzes win rate trends over time."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            now = datetime.utcnow()

            periods = {
                "last_7d": now - timedelta(days=7),
                "last_14d": now - timedelta(days=14),
                "last_30d": now - timedelta(days=30)
            }

            trends = {}
            for label, start_date in periods.items():
                start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")

                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status NOT IN ('ACTIVE', 'PENDING', 'NEUTRAL') THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'WON_TP' THEN 1 ELSE 0 END) as tp_hits,
                        SUM(CASE WHEN status = 'WON_DIRECTION' THEN 1 ELSE 0 END) as dir_wins
                    FROM prediction_tracker 
                    WHERE created_at >= ?
                """, (start_str,))

                row = cursor.fetchone()
                completed = row["completed"]
                wins = row["tp_hits"] + row["dir_wins"]
                win_rate = wins / completed if completed > 0 else 0

                trends[label] = {
                    "win_rate": round(win_rate, 3),
                    "completed": completed
                }

            conn.close()

            # Determine trend direction
            wr_7d = trends["last_7d"]["win_rate"]
            wr_14d = trends["last_14d"]["win_rate"]
            wr_30d = trends["last_30d"]["win_rate"]

            if wr_7d > wr_14d > wr_30d:
                trend = "improving"
                strength = round(wr_7d - wr_30d, 3)
            elif wr_7d < wr_14d < wr_30d:
                trend = "declining"
                strength = round(wr_30d - wr_7d, 3)
            else:
                trend = "stable"
                strength = round(abs(wr_14d - wr_30d), 3)

            trends["trend"] = trend
            trends["trend_strength"] = strength

            return trends

        except Exception as e:
            logger.error(f"[Analytics] Trend analysis failed: {e}")
            return {}
