"""
utils/signal_printer.py
=======================
Pretty-prints a per-symbol TRADE SIGNAL SUMMARY box to STDOUT at the end
of each agent-loop cycle.

Usage (in run_agent.py):
    from utils.signal_printer import print_trade_signals
    print_trade_signals(context)

Data it reads from context (all already present after a full cycle):
    context["quant"][symbol]        — quant_signal, quant_confidence
    context["sentiment"][symbol]    — sentiment_signal
    context["fundamentals"][symbol] — fundamentals_signal
    context["risk"][symbol]         — action, vote_score, entry_price,
                                      stop_loss, take_profit, position_size,
                                      weights_used
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import utils.config as config

# ANSI colour codes (degrade gracefully if terminal doesn't support them)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

_WIDTH = 64   # inner width of the box (between ║ characters)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pad(text: str, width: int = _WIDTH) -> str:
    """Left-justify *visible* text in a padded field (ANSI codes are invisible)."""
    # Strip ANSI for length measurement
    import re
    visible = re.sub(r"\033\[[0-9;]*m", "", text)
    pad = width - len(visible)
    return text + (" " * max(pad, 0))


def _row(content: str) -> str:
    return f"║  {_pad(content, _WIDTH - 2)}║"


def _divider(char: str = "═") -> str:
    return f"╠{'═' * (_WIDTH + 2)}╣" if char == "═" else f"╠{'─' * (_WIDTH + 2)}╣"


def _top() -> str:
    return f"╔{'═' * (_WIDTH + 2)}╗"


def _bot() -> str:
    return f"╚{'═' * (_WIDTH + 2)}╝"


def _action_emoji(action: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪", "HOLD_OPEN": "🔵"}.get(action, "❓")


def _signal_color(signal: str) -> str:
    if signal in ("BUY", "BULLISH"):
        return _GREEN + signal + _RESET
    if signal in ("SELL", "BEARISH"):
        return _RED + signal + _RESET
    return _YELLOW + signal + _RESET


def _confidence_bar(conf: float, width: int = 10) -> str:
    """Return a simple ASCII bar, e.g. ████░░░░░░  62.3%"""
    filled = round(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{conf * 100:.1f}%"
    return f"{bar} {pct}"


def _pct_change(current: float, target: float) -> str:
    if not current:
        return "N/A"
    pct = (target - current) / current * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _rr(entry: float, sl: float, tp: float) -> str:
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    if not risk:
        return "N/A"
    ratio = reward / risk
    return f"1 : {ratio:.2f}"


def _fmt_price(price: float) -> str:
    return config.format_price(price)


def _notional(entry: float, size: float) -> str:
    """Return a human-readable notional value string."""
    if not entry or not size:
        return "—"
    val = entry * size
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"${val:,.2f}"
    return f"${val:.4f}"


# ─── Core printer ─────────────────────────────────────────────────────────────

def _build_signal_box(symbol: str, context: dict) -> list[str]:
    """Return a list of box lines for one symbol."""
    quant        = context.get("quant",        {}).get(symbol, {})
    sentiment    = context.get("sentiment",    {}).get(symbol, {})
    fundamentals = context.get("fundamentals", {}).get(symbol, {})
    risk         = context.get("risk",         {}).get(symbol, {})

    # ── Raw values ────────────────────────────────────────────────────────
    action        = risk.get("action", "HOLD")
    vote_score    = risk.get("vote_score", 0.5)
    entry         = risk.get("entry_price", 0.0)
    sl            = risk.get("stop_loss", 0.0)
    tp            = risk.get("take_profit", 0.0)
    size          = risk.get("position_size", 0.0)
    q_signal      = quant.get("quant_signal", "—")
    q_conf        = quant.get("quant_confidence", 0.0)
    s_signal      = sentiment.get("sentiment_signal",    "—")
    f_signal      = fundamentals.get("fundamentals_signal", "—")
    weights       = risk.get("weights_used", {})
    ts            = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Derived ───────────────────────────────────────────────────────────
    emoji         = _action_emoji(action)
    action_col    = _GREEN + action + _RESET if action == "BUY" else (_RED + action + _RESET if action == "SELL" else action)
    sl_pct        = _pct_change(entry, sl)
    tp_pct        = _pct_change(entry, tp)
    rr_str        = _rr(entry, sl, tp)
    conf_bar      = _confidence_bar(vote_score)
    low_conf_warn = vote_score < 0.60 and action != "HOLD"

    # Header title
    title = f"  TRADE SIGNAL SUMMARY  ·  {ts}"

    lines = [_top()]
    lines.append(_row(_BOLD + _CYAN + title + _RESET))
    lines.append(_divider())

    # Symbol + Action
    lines.append(_row(f"  Symbol      │ {_BOLD}{symbol}{_RESET}"))
    lines.append(_row(f"  Action      │ {emoji} {_BOLD}{action_col}{_RESET}   score {vote_score:.3f}"))
    lines.append(_row(f"  Confidence  │ {conf_bar}"))

    if entry:
        lines.append(_row(f"  Entry       │ {_fmt_price(entry)}"))
        lines.append(_row(
            f"  Stop Loss   │ {_fmt_price(sl)}  "
            f"({_RED}{sl_pct}{_RESET})"
        ))
        lines.append(_row(
            f"  Take Profit │ {_fmt_price(tp)}  "
            f"({_GREEN}{tp_pct}{_RESET})  R:R {rr_str}"
        ))
        lines.append(_row(f"  Size        │ {size:.6f} units"))

    lines.append(_divider("─"))

    # Analyst breakdown
    w_q = weights.get("quant",        1/3)
    w_s = weights.get("sentiment",    1/3)
    w_f = weights.get("fundamentals", 1/3)

    lines.append(_row(f"  🤖 Quant        {_signal_color(q_signal):<30} conf {q_conf:.1%}  wt {w_q:.0%}"))
    lines.append(_row(f"  📰 Sentiment    {_signal_color(s_signal):<30}              wt {w_s:.0%}"))
    lines.append(_row(f"  📊 Fundamentals {_signal_color(f_signal):<30}              wt {w_f:.0%}"))

    # Low-confidence warning
    if low_conf_warn:
        lines.append(_divider("─"))
        lines.append(_row(f"  ⚠️  {_YELLOW}LOW CONFIDENCE — verify before acting{_RESET}"))

    # ── Manual Trade Instructions ─────────────────────────────────────────
    lines.append(_divider("─"))
    lines.append(_row(f"  {_BOLD}📋 HOW TO PLACE THIS TRADE (Manual){_RESET}"))
    lines.append(_row(""))

    is_trade = action in ("BUY", "SELL")

    if is_trade and entry:
        direction  = "BUY / LONG" if action == "BUY" else "SELL / SHORT"
        dir_color  = _GREEN if action == "BUY" else _RED
        notional   = _notional(entry, size)

        lines.append(_row(f"  Direction   │ {dir_color}{_BOLD}{direction}{_RESET}"))
        lines.append(_row(f"  Instrument  │ {symbol}"))
        lines.append(_row(f"  Entry       │ Market order near {_BOLD}{_fmt_price(entry)}{_RESET}"))
        lines.append(_row(
            f"  Stop Loss   │ {_BOLD}{_fmt_price(sl)}{_RESET}  "
            f"({_RED}{sl_pct}{_RESET})  ← set this immediately"
        ))
        lines.append(_row(
            f"  Take Profit │ {_BOLD}{_fmt_price(tp)}{_RESET}  "
            f"({_GREEN}{tp_pct}{_RESET})  ← optional limit order"
        ))
        lines.append(_row(
            f"  Size        │ {_BOLD}{size:.6f} units{_RESET}  (≈ {notional} notional)"
        ))
        lines.append(_row(f"  Risk:Reward │ {rr_str}"))
        if low_conf_warn:
            lines.append(_row(""))
            lines.append(_row(f"  {_YELLOW}⚠  Low confidence — consider reduced size or skip{_RESET}"))
    else:
        lines.append(_row(f"  {_YELLOW}No trade — stand aside{_RESET}"))
        lines.append(_row(f"  Reason: {_DIM}signal not strong enough{_RESET}"))

    lines.append(_bot())
    return lines


def print_trade_signals(context: dict[str, Any]) -> None:
    """
    Print a signal summary box for every symbol in context["symbols"].

    Call this at the end of run_cycle() in run_agent.py.
    """
    symbols: list[str] = context.get("symbols", [])
    if not symbols:
        return

    header = (
        "\n"
        + "═" * (_WIDTH + 4) + "\n"
        + f"  📡  CYCLE SIGNAL REPORT  —  {len(symbols)} symbol(s)\n"
        + "═" * (_WIDTH + 4)
    )
    print(header)

    for symbol in symbols:
        box = _build_signal_box(symbol, context)
        print("\n".join(box))
        print()
