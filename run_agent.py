"""
ENSOTRADE Insights Engine — Autonomous Analysis Loop

The main entry point for the AI-powered market analysis system.

Usage:
    python run_agent.py                              # Continuous insights loop
    python run_agent.py --symbols BTC-USD EURUSD=X   # Override symbols
    python run_agent.py --interval 60                # Minutes between cycles
    python run_agent.py --once                       # Run single cycle then exit
    python run_agent.py --retrain BTC-USD            # Trigger manual retrain
    python run_agent.py --disable-agent sentiment    # Skip an agent by name

Architecture:
    Perceive  → MarketDataAnalyst    (data + features)
    Reason    → QuantAnalyst         (ML signal)
              → SentimentAnalyst     (NLP signal)
              → FundamentalsAnalyst  (fundamental signal)
    Synthesize→ ChiefInvestmentOfficer (executive summary)
    Score     → RiskManager          (weighted vote + signal scoring)
    Log       → PredictionLogger     (DB audit trail)
    Reflect   → RetrainerAgent       (auto-retrain if accuracy drops)
"""
import argparse
import logging
import sys
import time
from datetime import datetime

import schedule

from agents.registry import discover_agents
from utils.config import AGENT_SYMBOLS, AGENT_INTERVAL_MINUTES
from utils.signal_printer import print_trade_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_agent")

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  ENSOTRADE INSIGHTS ENGINE — AUTONOMOUS LOOP    ║
╠══════════════════════════════════════════════════════════════╣
║  Symbols:   {symbols:<51}║
║  Interval:  {interval} min{pad_interval:<47}║
║  Agents:    {agents:<51}║
║  Started:   {started:<51}║
╚══════════════════════════════════════════════════════════════╝
"""


def run_cycle(symbols: list[str], disabled: list[str]) -> dict:
    """
    Execute one full Perceive→Reason→Synthesize→Log cycle.

    All agents are discovered fresh from the registry each cycle so
    that new agent files added to disk are picked up without restart.
    """
    logger.info(f"─── Cycle START — {datetime.utcnow().isoformat()}")
    agents = discover_agents(disabled=disabled)

    context: dict = {
        "symbols":  symbols,
        "_agents":  agents,
    }

    for agent in agents:
        try:
            context = agent.run(context)
        except Exception as exc:
            logger.error(f"Agent {agent.name} raised an unhandled exception: {exc}", exc_info=True)
            # Continue — bad agent never stops the loop

    logger.info(f"─── Cycle END ─── signals: "
                + str({s: context.get("risk", {}).get(s, {}).get("action", "?")
                        for s in symbols}))

    # ── Print actionable insight summary ────────────────────────────
    print_trade_signals(context)

    return context


def trigger_retrain(symbol: str):
    """Force a retrain for a specific symbol via the RetrainerAgent directly."""
    logger.info(f"Manual retrain triggered for {symbol}")
    try:
        from agents.retrainer import RetrainerAgent
        context: dict = {"symbols": [symbol], "force_retrain": symbol}
        RetrainerAgent().run(context)
    except Exception as exc:
        logger.error(f"Manual retrain failed: {exc}", exc_info=True)


def print_banner(symbols: list[str], interval: int, agents: list):
    agent_names = ", ".join(a.name for a in agents)
    started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(_BANNER.format(
        symbols=", ".join(symbols),
        interval=interval,
        pad_interval="",
        agents=agent_names[:51],
        started=started,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="ENSOTRADE Insights Engine — Autonomous Analysis Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    print_banner(args.symbols, args.interval, agents)

    # ── Single cycle mode (smoke test) ──────────────────────────────────
    if args.once:
        run_cycle(args.symbols, args.disable_agent)
        logger.info("Single-cycle run complete.")
        return

    # ── Continuous scheduled loop ─────────────────────────────────────────
    logger.info(f"Scheduling analysis cycle every {args.interval} minute(s). Press Ctrl+C to stop.")

    # Run once immediately, then on schedule
    run_cycle(args.symbols, args.disable_agent)

    schedule.every(args.interval).minutes.do(
        run_cycle, args.symbols, args.disable_agent
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(15)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Goodbye.")


if __name__ == "__main__":
    main()
