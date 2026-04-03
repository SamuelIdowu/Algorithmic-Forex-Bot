"""
Autonomous Orchestrator Loop — Phase 8

The main entry point for the AI Hedge Fund agent loop.

Usage:
    python run_agent.py --mode paper                   # Safe paper trading
    python run_agent.py --mode backtest                # Historical simulation
    python run_agent.py --mode live                    # Real money ⚠️
    python run_agent.py --symbols BTC-USD EURUSD=X     # Override symbols
    python run_agent.py --interval 60                  # Minutes between cycles
    python run_agent.py --retrain BTC-USD              # Trigger manual retrain
    python run_agent.py --disable-agent sentiment      # Skip an agent by name

Architecture:
    Perceive  → MarketDataAnalyst   (data + features)
    Reason    → QuantAnalyst        (ML signal)
              → SentimentAnalyst    (NLP signal)
              → FundamentalsAnalyst (fundamental signal)
    Act       → RiskManager         (weighted vote + sizing)
              → PortfolioManager    (order dispatch + DB logging)
    Reflect   → RetrainerAgent      (auto-retrain if win rate drops)
"""
import argparse
import logging
import sys
import time
from datetime import datetime

import schedule

from agents.registry import discover_agents
from utils.config import AGENT_SYMBOLS, AGENT_INTERVAL_MINUTES, AGENT_MODE
from utils.signal_printer import print_trade_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_agent")

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🤖  AI HEDGE FUND — AUTONOMOUS AGENT LOOP         ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:      {mode:<51}║
║  Symbols:   {symbols:<51}║
║  Interval:  {interval} min{pad_interval:<47}║
║  Agents:    {agents:<51}║
║  Started:   {started:<51}║
╚══════════════════════════════════════════════════════════════╝
"""


def run_cycle(symbols: list[str], mode: str, disabled: list[str]) -> dict:
    """
    Execute one full Perceive→Reason→Act→Reflect cycle.

    All agents are discovered fresh from the registry each cycle so
    that new agent files added to disk are picked up without restart.
    """
    logger.info(f"─── Cycle START ({mode.upper()}) ─── {datetime.utcnow().isoformat()}")
    agents = discover_agents(disabled=disabled)

    # Inject agent list so PortfolioManager can call on_trade_result hooks
    context: dict = {
        "symbols":  symbols,
        "mode":     mode,
        "_agents":  agents,
    }

    for agent in agents:
        try:
            context = agent.run(context)
        except Exception as exc:
            logger.error(f"Agent {agent.name} raised an unhandled exception: {exc}", exc_info=True)
            # Continue — bad agent never stops the loop

    logger.info(f"─── Cycle END ─── actions: "
                + str({s: context.get("portfolio", {}).get(s, {}).get("executed_action", "?")
                        for s in symbols}))

    # ── Print actionable trade signal summary ────────────────────────────
    print_trade_signals(context)

    return context


def trigger_retrain(symbol: str):
    """Force a retrain for a specific symbol via the RetrainerAgent directly."""
    logger.info(f"Manual retrain triggered for {symbol}")
    try:
        from agents.retrainer import RetrainerAgent
        context: dict = {"symbols": [symbol], "mode": "retrain", "force_retrain": symbol}
        RetrainerAgent().run(context)
    except Exception as exc:
        logger.error(f"Manual retrain failed: {exc}", exc_info=True)


def print_banner(mode: str, symbols: list[str], interval: int, agents: list):
    agent_names = ", ".join(a.name for a in agents)
    started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(_BANNER.format(
        mode=mode,
        symbols=", ".join(symbols),
        interval=interval,
        pad_interval="",
        agents=agent_names[:51],
        started=started,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Autonomous Agent Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "paper", "live"], default=AGENT_MODE,
        help="Execution mode (default: from .env AGENT_MODE)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=AGENT_SYMBOLS,
        help="Space-separated symbol list (default: from .env AGENT_SYMBOLS)"
    )
    parser.add_argument(
        "--interval", type=int, default=AGENT_INTERVAL_MINUTES,
        help="Minutes between each cycle (default: from .env AGENT_INTERVAL_MINUTES)"
    )
    parser.add_argument(
        "--retrain", metavar="SYMBOL",
        help="Manually trigger retraining for a symbol and exit"
    )
    parser.add_argument(
        "--disable-agent", nargs="+", dest="disable_agent", default=[],
        metavar="NAME",
        help="Agent names to skip (e.g. --disable-agent sentiment fundamentals)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit (useful for testing)"
    )

    args = parser.parse_args()

    # ── Manual retrain mode ───────────────────────────────────────────────
    if args.retrain:
        trigger_retrain(args.retrain)
        return

    # ── Discovery check ───────────────────────────────────────────────────
    agents = discover_agents(disabled=args.disable_agent)
    if not agents:
        logger.error("No agents discovered. Check the agents/ package. Exiting.")
        sys.exit(1)

    print_banner(args.mode, args.symbols, args.interval, agents)

    # ── Single cycle mode (backtest / smoke test) ────────────────────────
    if args.once or args.mode == "backtest":
        run_cycle(args.symbols, args.mode, args.disable_agent)
        logger.info("Single-cycle run complete.")
        return

    # ── Continuous scheduled loop ─────────────────────────────────────────
    logger.info(f"Scheduling cycle every {args.interval} minute(s). Press Ctrl+C to stop.")

    # Run once immediately, then on schedule
    run_cycle(args.symbols, args.mode, args.disable_agent)

    schedule.every(args.interval).minutes.do(
        run_cycle, args.symbols, args.mode, args.disable_agent
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(15)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Goodbye.")


if __name__ == "__main__":
    main()
