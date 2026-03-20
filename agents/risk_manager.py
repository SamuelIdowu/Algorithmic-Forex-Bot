"""
Risk Manager — Phase 6 (Layer 2 Self-Improvement)

Aggregates signals from all analysts using adaptive weighted voting.
Weights are stored in the agent_performance DB table and automatically
adjust based on each analyst's historical accuracy. Equal weights (1/3)
are used initially until sufficient trade history exists.

Also computes:
  • ATR-based position size
  • Stop-loss and take-profit prices
  • Max-drawdown guard (blocks entry if portfolio drawdown > threshold)

Output in context["risk"][symbol]:
    {
        "action":        "BUY" | "SELL" | "HOLD",
        "vote_score":    float (0-1),
        "position_size": float (notional),
        "stop_loss":     float,
        "take_profit":   float,
        "weights_used":  {"quant": float, "sentiment": float, "fundamentals": float},
    }
"""
import json
import logging
import numpy as np

from agents.base_agent import BaseAgent
from utils.config import (
    REQUIRED_VOTE_SCORE,
    MAX_DRAWDOWN_PCT,
    RISK_PER_TRADE_PCT,
    ATR_SL_MULT,
    ATR_TP_MULT,
    AGENT_INITIAL_CAPITAL,
)

logger = logging.getLogger(__name__)

_SIGNAL_VALUE = {
    # Quant signals
    "BUY":      1.0,
    "SELL":     0.0,
    "HOLD":     0.5,
    # Sentiment/fundamentals signals
    "BULLISH":  1.0,
    "BEARISH":  0.0,
    "NEUTRAL":  0.5,
}


class RiskManager(BaseAgent):
    """
    Adaptive weighted voting risk manager.
    Reads analyst accuracy weights from DB; falls back to equal weights.
    """

    name = "RiskManager"
    role = "manager"
    priority = 6   # After all analysts

    def __init__(self):
        super().__init__()
        self._db = None   # Lazily initialised to avoid import cycles

    def _get_db(self):
        if self._db is None:
            from data.db_manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def run(self, context: dict) -> dict:
        symbols    = context.get("symbols", [])
        mode       = context.get("mode", "backtest")
        risk_out   = {}

        # Portfolio-level drawdown guard
        portfolio_value = self._get_portfolio_value(mode)
        drawdown_pct    = self._calc_drawdown(portfolio_value)
        drawdown_blocked = drawdown_pct > MAX_DRAWDOWN_PCT

        if drawdown_blocked:
            logger.warning(
                f"[RiskManager] Portfolio drawdown {drawdown_pct:.1f}% "
                f"exceeds limit {MAX_DRAWDOWN_PCT}%. All entries BLOCKED."
            )

        for symbol in symbols:
            try:
                result = self._evaluate(symbol, context, drawdown_blocked)
                risk_out[symbol] = result
                
                # Log CIO reasoning specifically if available
                cio_memo = context.get("cio", {}).get(symbol, {}).get("memo", "")
                if cio_memo:
                    logger.info(f"[RiskManager] CIO Reasoning: {cio_memo}")

                logger.info(
                    f"[RiskManager] {symbol}: {result['action']} "
                    f"(score={result['vote_score']:.3f}, "
                    f"size={result['position_size']:.2f})"
                )
            except Exception as exc:
                logger.error(f"[RiskManager] Error for {symbol}: {exc}", exc_info=True)
                risk_out[symbol] = self._hold(symbol)

        context["risk"] = risk_out
        context["portfolio_value"] = portfolio_value
        context["drawdown_pct"]    = drawdown_pct
        return context

    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(self, symbol: str, context: dict, drawdown_blocked: bool) -> dict:
        """Compute risk decision for a single symbol."""
        quant       = context.get("quant", {}).get(symbol, {})
        sentiment   = context.get("sentiment", {}).get(symbol, {})
        fundamentals = context.get("fundamentals", {}).get(symbol, {})
        cio         = context.get("cio", {}).get(symbol, {})
        market_data  = context.get("market_data", {}).get(symbol, {})

        # ── Signals ──────────────────────────────────────────────────────────
        q_val  = _SIGNAL_VALUE.get(quant.get("quant_signal", "HOLD"), 0.5)
        s_val  = _SIGNAL_VALUE.get(sentiment.get("sentiment_signal", "NEUTRAL"), 0.5)
        f_val  = _SIGNAL_VALUE.get(fundamentals.get("fundamentals_signal", "NEUTRAL"), 0.5)
        
        # Incorporate CIO intelligence (Llama-3 reasoning)
        cio_val = _SIGNAL_VALUE.get(cio.get("cio_signal", "NEUTRAL"), 0.5)

        # ── Adaptive weights from DB ──────────────────────────────────────
        weights = self._get_weights(symbol)
        w_q, w_s, w_f = weights["quant"], weights["sentiment"], weights["fundamentals"]
        
        # Give CIO a 25% base weight, scale others down
        w_cio = 0.25
        w_q *= (1 - w_cio)
        w_s *= (1 - w_cio)
        w_f *= (1 - w_cio)

        # ── Weighted vote ─────────────────────────────────────────────────
        score = float(np.dot([q_val, s_val, f_val, cio_val], [w_q, w_s, w_f, w_cio]))

        if drawdown_blocked:
            action = "HOLD"
        elif score > REQUIRED_VOTE_SCORE:
            action = "BUY"
        elif score < (1.0 - REQUIRED_VOTE_SCORE):
            action = "SELL"
        else:
            action = "HOLD"

        # ── ATR-based position sizing ─────────────────────────────────────
        atr            = float(market_data.get("atr", 1.0)) or 1.0
        latest_close   = float(market_data.get("latest_close", 1.0)) or 1.0
        portfolio_value = context.get("portfolio_value", AGENT_INITIAL_CAPITAL)
        risk_amount    = portfolio_value * RISK_PER_TRADE_PCT
        position_size  = risk_amount / (atr * ATR_SL_MULT)

        stop_loss    = latest_close - (ATR_SL_MULT * atr)
        take_profit  = latest_close + (ATR_TP_MULT * atr)

        return {
            "action":        action,
            "vote_score":    round(score, 4),
            "position_size": round(position_size, 6),
            "stop_loss":     round(stop_loss, 6),
            "take_profit":   round(take_profit, 6),
            "entry_price":   round(latest_close, 6),
            "weights_used":  weights,
            "cio_memo":      cio.get("memo", ""),
        }

    def _get_weights(self, symbol: str) -> dict:
        """Retrieve normalised analyst weights from DB (equal fallback)."""
        equal = {"quant": 1/3, "sentiment": 1/3, "fundamentals": 1/3}
        try:
            db = self._get_db()
            weights = db.get_analyst_weights(symbol)
            if weights:
                return weights
        except Exception as exc:
            logger.debug(f"[RiskManager] Weight lookup failed for {symbol}: {exc}")
        return equal

    def _get_portfolio_value(self, mode: str) -> float:
        """Get current portfolio value from DB or use initial capital."""
        try:
            db = self._get_db()
            val = db.get_portfolio_value()
            return val if val else AGENT_INITIAL_CAPITAL
        except Exception:
            return AGENT_INITIAL_CAPITAL

    def _calc_drawdown(self, current_value: float) -> float:
        """Estimate drawdown % from initial capital."""
        if current_value >= AGENT_INITIAL_CAPITAL:
            return 0.0
        return (AGENT_INITIAL_CAPITAL - current_value) / AGENT_INITIAL_CAPITAL * 100.0

    @staticmethod
    def _hold(symbol: str) -> dict:
        return {
            "action":        "HOLD",
            "vote_score":    0.5,
            "position_size": 0.0,
            "stop_loss":     0.0,
            "take_profit":   0.0,
            "entry_price":   0.0,
            "weights_used":  {"quant": 1/3, "sentiment": 1/3, "fundamentals": 1/3},
        }
