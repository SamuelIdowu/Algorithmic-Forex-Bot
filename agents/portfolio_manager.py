"""
Portfolio Manager — Phase 7

Dispatches trade orders based on the RiskManager's approved signals.
Maintains the full audit trail and position state in SQLite.

Modes:
  • backtest — logs trades to DB, does NOT call broker
  • paper    — calls trader.py (Alpaca paper account)
  • live     — calls trader.py (real money ⚠️)

Self-improvement hook:
  After a position closes, this module:
    1. Computes realised PnL
    2. Calls db.close_trade() to record PnL
    3. Calls db.update_analyst_performance() for all three analysts
       (correct if signal agreed with profitable direction)

Output: context["portfolio"][symbol] = {
    "executed_action": str,
    "trade_id": int | None,
    "reason": str,
}
"""
import logging

from agents.base_agent import BaseAgent
from data.db_manager import DatabaseManager
from utils.config import AGENT_INITIAL_CAPITAL
from utils.telegram_utils import send_telegram_message_sync

logger = logging.getLogger(__name__)

_MAX_CAPITAL_DEPLOYED = 0.80   # refuse BUY if > 80% of capital is in market


class PortfolioManager(BaseAgent):
    """
    Executes orders approved by RiskManager and manages position lifecycle.
    """

    name = "PortfolioManager"
    role = "manager"
    priority = 7   # After RiskManager

    def __init__(self):
        self._db = DatabaseManager()

    def run(self, context: dict) -> dict:
        symbols    = context.get("symbols", [])
        mode       = context.get("mode", "backtest")
        portfolio  = {}

        portfolio_value = context.get("portfolio_value", AGENT_INITIAL_CAPITAL)
        total_deployed  = self._db.get_total_deployed()
        deployed_pct    = total_deployed / portfolio_value if portfolio_value > 0 else 0

        for symbol in symbols:
            try:
                result = self._process_symbol(
                    symbol, context, mode, portfolio_value, deployed_pct
                )
                portfolio[symbol] = result
                logger.info(
                    f"[PortfolioManager] {symbol}: {result['executed_action']} "
                    f"— {result['reason']}"
                )
            except Exception as exc:
                logger.error(f"[PortfolioManager] Error for {symbol}: {exc}", exc_info=True)
                portfolio[symbol] = {"executed_action": "ERROR", "trade_id": None, "reason": str(exc)}

        context["portfolio"] = portfolio
        return context

    # ──────────────────────────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str, context: dict, mode: str,
                        portfolio_value: float, deployed_pct: float) -> dict:
        risk   = context.get("risk", {}).get(symbol, {})
        action = risk.get("action", "HOLD")

        quant        = context.get("quant", {}).get(symbol, {})
        sentiment    = context.get("sentiment", {}).get(symbol, {})
        fundamentals = context.get("fundamentals", {}).get(symbol, {})

        open_pos = self._db.get_open_position(symbol)

        # ── Handle open position SL/TP check ─────────────────────────────
        if open_pos:
            current_price = (
                context.get("market_data", {})
                       .get(symbol, {})
                       .get("latest_close", 0)
            )
            sl = open_pos.get("stop_loss", 0)
            tp = open_pos.get("take_profit", 0)
            entry = open_pos.get("entry_price", 0)
            size  = open_pos.get("position_size", 0)
            trade_id = open_pos.get("trade_id")

            if current_price and sl and current_price <= sl:
                return self._close_position(
                    symbol, current_price, entry, size, trade_id,
                    "STOP_LOSS", context, mode
                )
            if current_price and tp and current_price >= tp:
                return self._close_position(
                    symbol, current_price, entry, size, trade_id,
                    "TAKE_PROFIT", context, mode
                )

            # If risk says SELL explicitly, close position
            if action == "SELL":
                return self._close_position(
                    symbol, current_price, entry, size, trade_id,
                    "SIGNAL_SELL", context, mode
                )

            return {"executed_action": "HOLD_OPEN", "trade_id": trade_id,
                    "reason": "Position open, no exit trigger yet"}

        # ── No open position — consider entering ──────────────────────────
        if action != "BUY":
            return self._log_hold(symbol, risk, quant, sentiment, fundamentals, mode)

        # Allocation cap
        if deployed_pct >= _MAX_CAPITAL_DEPLOYED:
            return {"executed_action": "BLOCKED",
                    "trade_id": None,
                    "reason": f"Capital cap reached ({deployed_pct*100:.1f}% deployed)"}

        # Execute BUY
        return self._open_position(symbol, risk, quant, sentiment, fundamentals, mode, context)

    def _open_position(self, symbol: str, risk: dict, quant: dict,
                       sentiment: dict, fundamentals: dict, mode: str, context: dict) -> dict:
        price      = risk.get("entry_price", 0)
        size       = risk.get("position_size", 0)
        stop_loss  = risk.get("stop_loss", 0)
        take_profit = risk.get("take_profit", 0)
        vote_score = risk.get("vote_score", 0)
        weights    = risk.get("weights_used", {})
        cio_memo   = risk.get("cio_memo", "")

        trade_id = self._db.log_trade(
            symbol=symbol, action="BUY", price=price, size=size,
            stop_loss=stop_loss, take_profit=take_profit,
            quant_signal=quant.get("quant_signal", ""),
            quant_confidence=quant.get("quant_confidence", 0),
            sentiment_signal=sentiment.get("sentiment_signal", ""),
            fundamentals_signal=fundamentals.get("fundamentals_signal", ""),
            vote_score=vote_score,
            weights_used=weights,
            mode=mode,
            cio_memo=cio_memo,
        )

        if trade_id > 0:
            self._db.set_position(
                symbol, size, price, stop_loss, take_profit, trade_id
            )

        # Paper / live: call trader
        if mode in ("paper", "live"):
            self._dispatch_order(symbol, "buy", size, mode)

        # ── Telegram Alert ──────────────────────────────────────────────────
        cio_memo = context.get("cio", {}).get(symbol, {}).get("memo", "")
        msg = (
            f"🚀 <b>BUY {symbol}</b>\n"
            f"Price: {price:.2f}\n"
            f"Size: {size:.4f}\n"
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}\n"
            f"Score: {vote_score:.2f}\n"
            f"Memo: {cio_memo}"
        )
        send_telegram_message_sync(msg)

        return {
            "executed_action": "BUY",
            "trade_id": trade_id,
            "reason": f"Entry @ {price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}",
        }

    def _close_position(self, symbol: str, current_price: float, entry_price: float,
                        size: float, trade_id: int, trigger: str,
                        context: dict, mode: str) -> dict:
        pnl = (current_price - entry_price) * size

        # Close trade in DB
        if trade_id:
            self._db.close_trade(trade_id, pnl)

        self._db.clear_position(symbol)

        # Update analyst performance (correct = trade was profitable)
        trade_won = pnl > 0
        quant        = context.get("quant", {}).get(symbol, {})
        sentiment    = context.get("sentiment", {}).get(symbol, {})
        fundamentals = context.get("fundamentals", {}).get(symbol, {})

        q_correct = trade_won == (quant.get("quant_signal") == "BUY")
        s_correct = trade_won == (sentiment.get("sentiment_signal") == "BULLISH")
        f_correct = trade_won == (fundamentals.get("fundamentals_signal") == "BULLISH")

        self._db.update_analyst_performance(symbol, "quant",        q_correct)
        self._db.update_analyst_performance(symbol, "sentiment",    s_correct)
        self._db.update_analyst_performance(symbol, "fundamentals", f_correct)

        # Notify agents of trade result
        for agent in context.get("_agents", []):
            try:
                agent.on_trade_result(symbol, pnl)
            except Exception:
                pass

        if mode in ("paper", "live"):
            self._dispatch_order(symbol, "sell", size, mode)

        # ── Telegram Alert ──────────────────────────────────────────────────
        emoji = "💰" if trade_won else "📉"
        msg = (
            f"{emoji} <b>CLOSE {symbol}</b>\n"
            f"PnL: {pnl:+.4f}\n"
            f"Trigger: {trigger}\n"
            f"Accuracy: Q={'✅' if q_correct else '❌'} | S={'✅' if s_correct else '❌'}"
        )
        send_telegram_message_sync(msg)

        return {
            "executed_action": f"CLOSE ({trigger})",
            "trade_id": trade_id,
            "reason": f"PnL={pnl:.4f}, trigger={trigger}",
        }

    def liquidate_all_positions(self, mode: str, context: dict) -> list:
        """Close all open positions in the broker and database."""
        df_positions = self._db.get_all_positions()
        if df_positions.empty:
            return []

        results = []
        for _, row in df_positions.iterrows():
            symbol = row['symbol']
            size   = row['position_size']
            entry  = row['entry_price']
            trade_id = row['trade_id']

            # Get current price for PnL calculation
            current_price = (
                context.get("market_data", {})
                       .get(symbol, {})
                       .get("latest_close", 0)
            )
            # If current price not in context, try to fetch it? 
            # For now, if 0, PnL will be wrong but position will be closed.
            
            res = self._close_position(
                symbol, current_price, entry, size, trade_id,
                "MANUAL_LIQUIDATE", context, mode
            )
            results.append(res)
            
        return results

    def _log_hold(self, symbol: str, risk: dict, quant: dict,
                  sentiment: dict, fundamentals: dict, mode: str) -> dict:
        """Log HOLD cycles for full audit trail."""
        self._db.log_trade(
            symbol=symbol, action="HOLD", price=0, size=0,
            stop_loss=0, take_profit=0,
            quant_signal=quant.get("quant_signal", ""),
            quant_confidence=quant.get("quant_confidence", 0),
            sentiment_signal=sentiment.get("sentiment_signal", ""),
            fundamentals_signal=fundamentals.get("fundamentals_signal", ""),
            vote_score=risk.get("vote_score", 0.5),
            weights_used=risk.get("weights_used", {}),
            mode=mode,
            cio_memo=risk.get("cio_memo", ""),
        )
        return {
            "executed_action": "HOLD",
            "trade_id": None,
            "reason": f"No entry signal (score={risk.get('vote_score', 0):.3f})",
        }

    @staticmethod
    def _dispatch_order(symbol: str, side: str, size: float, mode: str):
        """Call the live trading module to place the actual order."""
        try:
            from live_trading.trader import AlpacaTrader
            trader = AlpacaTrader()
            trader.place_order(symbol, side, size)
            logger.info(f"[PortfolioManager] {mode.upper()} order dispatched: {side} {size} {symbol}")
        except Exception as exc:
            logger.error(f"[PortfolioManager] Order dispatch failed: {exc}")
