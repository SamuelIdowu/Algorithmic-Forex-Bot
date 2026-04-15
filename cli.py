"""
ENSOTRADE Insights Engine — Interactive CLI

Main menu choices:
  1. 📈🔮 Train → Predict     — train then signal (full chain)
  2. 📈  Train Model          — train / retrain an ML model
  3. 🔮  Predict (Next Candle) — single-symbol ML prediction
  4. 🔍  Analyze Symbol       — multi-agent deep analysis
  5. 📋  Trackers             — manage monitoring schedules
  6. ⚙️   Configure           — risk/reward params
  7. 📊  Dashboard            — open insights portal
  8. 💬  AI Chat              — institutional consultation
  9. 🤖📱 Telegram Bot         — run the interactive bot
 10. 🚪  Exit
"""
import os
import re
import sys
import subprocess

import questionary
from questionary import Style

# ─── Colour theme ──────────────────────────────────────────────────────────────
_STYLE = Style([
    ("qmark",        "fg:#00c896 bold"),
    ("question",     "bold"),
    ("answer",       "fg:#4a9eff bold"),
    ("pointer",      "fg:#00c896 bold"),
    ("highlighted",  "fg:#00c896 bold"),
    ("selected",     "fg:#4a9eff"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888"),
    ("text",         ""),
    ("disabled",     "fg:#858585 italic"),
])

# ─── Symbol helpers ───────────────────────────────────────────────────────────

# Flat list used by the single-symbol picker (Train / Predict / Analyze)
COMMON_SYMBOLS = [
    # ── Forex ──────────────────────────────────────────────────────────────
    "EURUSD=X",  "GBPUSD=X",  "USDJPY=X",  "AUDUSD=X",  "USDCAD=X",
    "NZDUSD=X",  "USDCHF=X",  "EURGBP=X",  "EURJPY=X",  "GBPJPY=X",
    # ── Crypto ─────────────────────────────────────────────────────────────
    "BTC-USD",   "ETH-USD",   "BNB-USD",   "SOL-USD",   "XRP-USD",
    "ADA-USD",   "DOGE-USD",  "AVAX-USD",  "LINK-USD",  "DOT-USD",
    # ── Commodities ────────────────────────────────────────────────────────
    "GC=F",      "SI=F",      "CL=F",      "HG=F",      "NG=F",
    "ZW=F",      "ZC=F",
    # ── Indices ────────────────────────────────────────────────────────────
    "SPY",       "QQQ",       "DIA",       "IWM",       "^VIX",
    # ── US Stocks ──────────────────────────────────────────────────────────
    "AAPL",  "MSFT",  "GOOGL", "AMZN",  "NVDA",  "TSLA",  "META",
    "NFLX",  "AMD",   "INTC",
    # ── Other ──────────────────────────────────────────────────────────────
    "Other (enter manually)",
]

from utils.config import AGENT_PAIR_GROUPS, AGENT_SYMBOL_DEFAULTS


def _pick_symbol(prompt: str = "Select symbol:") -> str:
    sel = questionary.select(prompt, choices=COMMON_SYMBOLS, style=_STYLE).ask()
    if sel == "Other (enter manually)":
        return questionary.text("Custom symbol (Yahoo Finance format):", style=_STYLE).ask().strip()
    return sel


def _pick_symbols_multi(prompt: str = "Select pairs to analyze:") -> list[str]:
    """Multi-select checkbox picker with grouped pairs for the autonomous agent."""
    # Build flat questionary choices list, inserting separators for groups
    choices: list = []
    for group_label, pairs in AGENT_PAIR_GROUPS:
        choices.append(questionary.Separator(group_label))
        for display, ticker in pairs:
            choices.append(questionary.Choice(display, value=ticker))

    choices.append(questionary.Separator("── Custom ─────────────────────────────────"))
    choices.append(questionary.Choice("➕  Add custom symbol manually", value="__custom__"))

    # Pre-tick the defaults
    defaults = [t for t in AGENT_SYMBOL_DEFAULTS.split(",")]
    for c in choices:
        if isinstance(c, questionary.Choice) and c.value in defaults:
            c.checked = True

    selected: list[str] = questionary.checkbox(
        prompt,
        choices=choices,
        style=_STYLE,
        instruction="(Space to select, Enter to confirm)",
    ).ask() or []

    result = [s for s in selected if s != "__custom__"]

    if "__custom__" in selected:
        raw = questionary.text(
            "Custom symbols (comma-separated, Yahoo Finance format):",
            style=_STYLE,
        ).ask()
        result += [s.strip() for s in raw.split(",") if s.strip()]

    if not result:
        # Fallback if nothing selected
        print("  ⚠️  No pairs selected — using defaults.")
        result = AGENT_SYMBOL_DEFAULTS.split(",")

    return result


def _run(command: list[str], env: dict = None):
    """Stream a subprocess and print its output live."""
    print(f"\n🚀  Running: {' '.join(command)}\n{'─'*60}")
    try:
        subprocess.run(command, check=True, env=env or os.environ.copy())
        print(f"{'─'*60}\n✅  Done!\n")
    except subprocess.CalledProcessError as e:
        print(f"{'─'*60}\n❌  Failed (exit code {e.returncode})\n")
    except KeyboardInterrupt:
        print("\n⚠️   Stopped by user.\n")


# ─── .env helpers ─────────────────────────────────────────────────────────────

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env(key: str, default: str = "") -> str:
    """Read a single key from the .env file (fallback to env var then default)."""
    try:
        with open(_ENV_PATH) as f:
            for line in f:
                m = re.match(rf"^{key}\s*=\s*(.*)", line.strip())
                if m:
                    # Strip inline comments and surrounding whitespace/quotes
                    val = m.group(1).split("#")[0].strip().strip('"\'')
                    return val
    except FileNotFoundError:
        pass
    return os.getenv(key, default)


def _write_env(updates: dict[str, str]) -> None:
    """Write/update key=value pairs in the .env file (in-place)."""
    try:
        with open(_ENV_PATH) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    written = set()
    new_lines = []
    for line in lines:
        matched = False
        for key, val in updates.items():
            if re.match(rf"^{key}\s*=", line.strip()):
                # Preserve inline comment if present
                comment_match = re.search(r"(\s*#.*)$", line)
                comment = comment_match.group(1) if comment_match else ""
                new_lines.append(f"{key}={val}{comment}\n")
                written.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    # Append any keys that didn't exist yet
    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}={val}\n")

    with open(_ENV_PATH, "w") as f:
        f.writelines(new_lines)


# ══════════════════════════════════════════════════════════════════════════════
# Menu handlers
# ══════════════════════════════════════════════════════════════════════════════

def menu_retrain():
    """🔄 Retrain a model via the RetrainerAgent"""
    print("\n🔄  Manual Retrain\n")
    symbol = _pick_symbol("Symbol to retrain:")
    _run([sys.executable, "run_agent.py", "--retrain", symbol])


def menu_train():
    """📈 Train ML Model (train_model.py)"""
    print("\n📈  Train ML Model\n")
    symbol = _pick_symbol("Symbol to train:")

    interval = questionary.select(
        "Candle interval:",
        choices=[
            questionary.Choice("1d  — Daily (full history, most robust)",      value="1d"),
            questionary.Choice("1h  — Hourly (2 years, good for day trading)", value="1h"),
            questionary.Choice("30m — 30-min (60 days max from Yahoo Finance)", value="30m"),
            questionary.Choice("15m — 15-min (60 days max from Yahoo Finance)", value="15m"),
            questionary.Choice("5m  — 5-min  (60 days max, noisy)",            value="5m"),
        ],
        style=_STYLE,
    ).ask()

    # Sensible default date range per interval
    _default_start = {
        "1d": "2020-01-01", "1h": "2023-01-01",
        "30m": "2025-01-01", "15m": "2025-01-01", "5m": "2025-01-01",
    }
    start_date = questionary.text(
        "Start date (YYYY-MM-DD):", default=_default_start.get(interval, "2020-01-01"), style=_STYLE
    ).ask()
    end_date = questionary.text("End date (YYYY-MM-DD):", default="2026-02-28", style=_STYLE).ask()

    # Auto-name includes interval suffix for intraday (e.g. btc-usd_15m_model.pkl)
    safe_name   = symbol.lower().replace("/", "_").replace("=", "_")
    tf_suffix   = f"_{interval}" if interval != "1d" else ""
    model_path  = f"models/{safe_name}{tf_suffix}_model.pkl"
    scaler_path = f"models/{safe_name}{tf_suffix}_scaler.pkl"

    if not questionary.confirm(f"Auto paths? ({model_path})", default=True, style=_STYLE).ask():
        model_path  = questionary.text("Model path:",  default=model_path,  style=_STYLE).ask()
        scaler_path = questionary.text("Scaler path:", default=scaler_path, style=_STYLE).ask()

    hyperopt = questionary.confirm("Run hyperparameter tuning? (slower)", default=False, style=_STYLE).ask()

    cmd = [
        sys.executable, "train_model.py",
        "--symbol",      symbol,
        "--start",       start_date,
        "--end",         end_date,
        "--interval",    interval,
        "--model_path",  model_path,
        "--scaler_path", scaler_path,
    ]
    if hyperopt:
        cmd.append("--tune")

    _run(cmd)

    # ── Offer to predict immediately with the freshly trained model ───────
    if questionary.confirm(
        "\n🔮  Run a prediction now with this model?",
        default=True, style=_STYLE
    ).ask():
        _run_predict(
            symbol=symbol,
            timeframe=interval,
            model_path=model_path,
            scaler_path=scaler_path,
        )


def _run_predict(
    symbol: str,
    timeframe: str | None = None,
    model_path: str | None = None,
    scaler_path: str | None = None,
) -> None:
    """
    Core predict runner — can be called standalone or as part of a chain.
    If timeframe/paths are passed in (from a train chain), those values are
    used as pre-filled defaults; the user can still override them.
    """
    if symbol is None:
        symbol = _pick_symbol("Symbol:")

    # Timeframe — pre-filled or interactive
    if timeframe is None:
        timeframe = questionary.select(
            "Candle interval (must match the model you trained):",
            choices=["1d", "1h", "30m", "15m", "5m"],
            style=_STYLE,
        ).ask()

    # Build canonical paths matching train_model.py naming
    safe_name  = symbol.lower().replace("/", "_").replace("=", "_")
    tf_suffix  = f"_{timeframe}" if timeframe != "1d" else ""
    _model_def  = model_path  or f"models/{safe_name}{tf_suffix}_model.pkl"
    _scaler_def = scaler_path or f"models/{safe_name}{tf_suffix}_scaler.pkl"

    if not questionary.confirm(f"Auto model path? ({_model_def})", default=True, style=_STYLE).ask():
        _model_def  = questionary.text("Model path:",  default=_model_def,  style=_STYLE).ask()
        _scaler_def = questionary.text("Scaler path:", default=_scaler_def, style=_STYLE).ask()

    lookback = "100"
    sl_mult  = _read_env("ATR_SL_MULT", "2.0")
    tp_mult  = _read_env("ATR_TP_MULT", "3.0")
    if questionary.confirm(
        f"Customise lookback / ATR multipliers? (SL={sl_mult}×, TP={tp_mult}×)",
        default=False, style=_STYLE
    ).ask():
        lookback = questionary.text("Lookback (candles):",   default=lookback, style=_STYLE).ask()
        sl_mult  = questionary.text("Stop-loss ATR mult:",   default=sl_mult,  style=_STYLE).ask()
        tp_mult  = questionary.text("Take-profit ATR mult:", default=tp_mult,  style=_STYLE).ask()

    _run([
        sys.executable, "predict.py",
        "--symbol",      symbol,
        "--model_path",  _model_def,
        "--scaler_path", _scaler_def,
        "--interval",    timeframe,
        "--lookback",    lookback,
        "--sl_mult",     sl_mult,
        "--tp_mult",     tp_mult,
    ])


def menu_predict():
    """🔮 Predict (Next Candle)"""
    print("\n🔮  Next-Candle Prediction\n")
    _run_predict(symbol=_pick_symbol("Symbol:"))


def menu_train_predict():
    """📈🔮 Train then Predict — full chain"""
    print("\n📈🔮  Train → Predict Chain\n")
    print("  Step 1 of 2: Train the model")
    print("  Step 2 of 2: Immediately predict using that model\n")
    menu_train()


def menu_analyze():
    """🔍 Multi-Agent Deep Analysis"""
    print("\n🔍  Multi-Agent Analysis\n")
    symbol = _pick_symbol("Symbol to analyze:")
    timeframe = questionary.select(
        "Candle interval:",
        choices=["1d", "1h", "30m", "15m", "5m"],
        style=_STYLE,
    ).ask()

    # Run the insights engine once for this symbol
    cmd = [
        sys.executable, "run_agent.py",
        "--symbols", symbol,
        "--interval", "1",
        "--once",
    ]
    _run(cmd)


def menu_dashboard():
    """📊 Launch Insights Portal"""
    print("\n📊  Launching ENSOTRADE Insights Portal…\n")
    print("  → Open http://localhost:8000 in your browser")
    print("  → Press Ctrl+C here to stop the dashboard\n")
    _run([sys.executable, "web/server.py"])


def menu_configure():
    """⚙️ Configure Trade Settings (edits .env live)"""
    print("\n⚙️  Configure Analysis Parameters\n")
    print("  Current values loaded from .env — press Enter to keep unchanged.\n")

    # ── Read current values ────────────────────────────────────────────────
    cur_sl      = _read_env("ATR_SL_MULT",           "2.0")
    cur_tp      = _read_env("ATR_TP_MULT",           "3.0")
    cur_risk    = _read_env("RISK_PER_TRADE_PCT",    "1")
    cur_vote    = _read_env("REQUIRED_VOTE_SCORE",   "0.55")
    cur_symbols = _read_env("AGENT_SYMBOLS",         "BTC-USD,EURUSD=X,GC=F")

    # ── Derive current R:R for display ────────────────────────────────────
    try:
        cur_rr = f"1 : {float(cur_tp) / float(cur_sl):.2f}"
    except ZeroDivisionError:
        cur_rr = "N/A"

    print(f"  Current R:R  →  SL={cur_sl}×ATR  |  TP={cur_tp}×ATR  |  Ratio {cur_rr}")
    print(f"  Risk/signal  →  {cur_risk}% notional")
    print(f"  Vote thresh  →  {cur_vote}\n")

    # ── ATR Stop-Loss Multiplier ───────────────────────────────────────────
    sl_mult = questionary.text(
        f"Stop-Loss ATR multiplier (current: {cur_sl}):",
        default=cur_sl, style=_STYLE,
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a number (e.g. 2.0)",
    ).ask()

    # ── ATR Take-Profit Multiplier ────────────────────────────────────────
    tp_mult = questionary.text(
        f"Take-Profit ATR multiplier (current: {cur_tp}):",
        default=cur_tp, style=_STYLE,
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a number (e.g. 3.0)",
    ).ask()

    # ── Risk per Signal ────────────────────────────────────────────────────
    risk_pct = questionary.text(
        f"Notional risk per signal % (current: {cur_risk}):",
        default=cur_risk, style=_STYLE,
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a number (e.g. 1 or 0.5)",
    ).ask()

    # ── Vote score threshold ───────────────────────────────────────────────
    vote_score = questionary.text(
        f"Min vote score to trigger signal 0–1 (current: {cur_vote}):",
        default=cur_vote, style=_STYLE,
        validate=lambda v: (
            v.replace(".", "", 1).isdigit() and 0 < float(v) < 1
        ) or "Enter a decimal between 0 and 1 (e.g. 0.55)",
    ).ask()

    # ── Symbols ────────────────────────────────────────────────────────────
    symbols = questionary.text(
        f"Symbols, comma-separated (current: {cur_symbols}):",
        default=cur_symbols, style=_STYLE,
    ).ask()

    # ── Preview ────────────────────────────────────────────────────────────
    try:
        rr_preview = f"1 : {float(tp_mult) / float(sl_mult):.2f}"
    except (ZeroDivisionError, ValueError):
        rr_preview = "N/A"

    print(f"""
  ┌─────────────────────────────────────────┐
  │  📋  New Analysis Configuration         │
  ├─────────────────────────────────────────┤
  │  Stop-Loss   : {sl_mult:>6} × ATR              │
  │  Take-Profit : {tp_mult:>6} × ATR              │
  │  R:R Ratio   : {rr_preview:<24} │
  │  Risk/signal : {risk_pct:>5}% notional          │
  │  Vote thresh : {vote_score:<24} │
  │  Symbols     : {symbols[:24]:<24} │
  └─────────────────────────────────────────┘
""")

    if questionary.confirm("Save these settings to .env?", default=True, style=_STYLE).ask():
        _write_env({
            "ATR_SL_MULT":         sl_mult,
            "ATR_TP_MULT":         tp_mult,
            "RISK_PER_TRADE_PCT":  risk_pct,
            "REQUIRED_VOTE_SCORE": vote_score,
            "AGENT_SYMBOLS":       symbols,
        })
        print("\n✅  Settings saved to .env — will be applied on the next agent run.\n")
    else:
        print("\n⏭️   Cancelled — no changes made.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

_BANNER = r"""
  ███████╗███╗   ██╗██████╗
  ██╔════╝████╗  ██║██╔══██╗
  ███████╗██╔██╗ ██║██║  ██║
  ╚════██║██║╚██╗██║██║  ██║
  ███████║██║ ╚████║██████╔╝
  ╚══════╝╚═╝  ╚═══╝╚═════╝  INSIGHTS ENGINE v3.0
"""

_MENU_CHOICES = [
    questionary.Choice("📈🔮 Train → Predict  — train then signal",      value="train_predict"),
    questionary.Choice("📈  Train Model       — train an ML model",     value="train"),
    questionary.Choice("🔮  Predict           — one-shot next-candle",  value="predict"),
    questionary.Separator("─── Analysis ─────────────────────────────────"),
    questionary.Choice("🔄  Retrain Model     — force model retrain",   value="retrain"),
    questionary.Choice("🔍  Analyze Symbol    — multi-agent deep analysis", value="analyze"),
    questionary.Separator("─── Settings ──────────────────────────────────"),
    questionary.Choice("⚙️   Configure        — risk/reward params",    value="configure"),
    questionary.Separator("─── Interfaces ────────────────────────────────"),
    questionary.Choice("📊  Dashboard         — open insights portal",  value="dashboard"),
    questionary.Choice("💬  AI Chat           — institutional consultation", value="chat"),
    questionary.Choice("🤖📱 Telegram Bot      — run the interactive bot",    value="telegram"),
    questionary.Separator(),
    questionary.Choice("🚪  Exit",                                       value="exit"),
]

_ACTION_MAP = {
    "train_predict": menu_train_predict,
    "train":         menu_train,
    "predict":       menu_predict,
    "retrain":       menu_retrain,
    "analyze":       menu_analyze,
    "configure":     menu_configure,
    "dashboard":     menu_dashboard,
    "chat":          lambda: _run([sys.executable, "chat_cli.py"]),
    "telegram":      lambda: _run([sys.executable, "telegram_bot.py"]),
}


def main():
    print(_BANNER)

    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=_MENU_CHOICES,
            style=_STYLE,
        ).ask()

        if choice is None or choice == "exit":
            print("\nGoodbye! 👋\n")
            break

        handler = _ACTION_MAP.get(choice)
        if handler:
            handler()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋\n")
