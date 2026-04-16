#!/usr/bin/env python3
"""
Backfill Prediction Trackers
============================

One-time migration script to create tracker entries for existing predictions
that don't have corresponding trackers yet.

This script is idempotent — safe to run multiple times. It will only create
trackers for predictions that are missing them (via LEFT JOIN check in the
DatabaseManager).

Usage:
    python scripts/backfill_trackers.py
    python scripts/backfill_trackers.py --symbol EUR/USD
    python scripts/backfill_trackers.py --dry-run

Backfilled entries receive:
    - source='backfill'
    - timeframe='1h' (default for historical data)
    - Technical fields (RSI, ATR, BB pos) left as NULL
"""

import argparse
import logging
import sys
import os

# Ensure project root is on the Python path so `data.db_manager` resolves.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.db_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(char='=', length=60):
    """Print a visual separator line."""
    print(char * length)


def main():
    parser = argparse.ArgumentParser(
        description='Backfill prediction trackers — create tracker entries for '
                    'existing predictions that lack them.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python scripts/backfill_trackers.py          # backfill all symbols\n'
            '  python scripts/backfill_trackers.py --dry-run  # preview only\n'
            '  python scripts/backfill_trackers.py --symbol EUR/USD  # specific symbol\n'
        )
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Only backfill predictions for this specific symbol'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be backfilled without actually creating trackers'
    )
    args = parser.parse_args()

    db = DatabaseManager()

    # ── Step 1: Show current state ──────────────────────────────────────
    print()
    print_separator()
    print("📊  PREDICTION TRACKER BACKFILL")
    print_separator()

    # Count existing predictions (fetch a large set to cover all history)
    limit = 10000
    if args.symbol:
        preds = db.get_recent_predictions(symbol=args.symbol, limit=limit)
        print(f"\nFiltering by symbol: {args.symbol}")
    else:
        preds = db.get_recent_predictions(limit=limit)

    total_predictions = len(preds)
    print(f"\nTotal predictions in database: {total_predictions}")

    if total_predictions == 0:
        print("\nNo predictions to backfill. Exiting.")
        return

    # Show date range
    if not preds.empty and 'timestamp' in preds.columns:
        min_ts = preds['timestamp'].min()
        max_ts = preds['timestamp'].max()
        print(f"Date range: {min_ts} to {max_ts}")

    # Show symbol breakdown
    if not preds.empty and 'symbol' in preds.columns:
        symbol_counts = preds.groupby('symbol').size().sort_values(ascending=False)
        print("\nPredictions by symbol:")
        for sym, count in symbol_counts.items():
            print(f"  {sym}: {count}")

    if args.dry_run:
        print(f"\n🔍  DRY RUN: Would attempt to backfill up to {total_predictions} predictions")
        print("   (Actual count may be lower — predictions that already have trackers are skipped)")
        print("\nRemove --dry-run to actually create trackers.")
        return

    # ── Step 2: Run backfill ────────────────────────────────────────────
    print()
    print_separator('-')
    print("⏳  Running backfill...")
    print_separator('-')

    try:
        trackers_created = db.backfill_trackers_from_predictions()

        print(f"\n✅  Backfill complete!")
        print(f"   Trackers created: {trackers_created}")

        if trackers_created == 0:
            print("   (No predictions needed backfilling — all already have trackers)")

        # ── Step 3: Show summary ────────────────────────────────────────
        print()
        print_separator('-')
        print("📈  TRACKER SUMMARY")
        print_separator('-')

        import sqlite3
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*)                                          AS total,
                SUM(CASE WHEN status = 'ACTIVE'      THEN 1 ELSE 0 END) AS active,
                SUM(CASE WHEN status LIKE 'WON%'     THEN 1 ELSE 0 END) AS won,
                SUM(CASE WHEN status = 'LOST_SL'     THEN 1 ELSE 0 END) AS lost_sl,
                SUM(CASE WHEN status = 'EXPIRED'     THEN 1 ELSE 0 END) AS expired,
                SUM(CASE WHEN source = 'backfill'    THEN 1 ELSE 0 END) AS backfilled,
                SUM(CASE WHEN source = 'agent_loop'  THEN 1 ELSE 0 END) AS from_agent_loop,
                SUM(CASE WHEN source = 'standalone'  THEN 1 ELSE 0 END) AS from_standalone
            FROM prediction_tracker
        """)

        row = cursor.fetchone()
        if row:
            total_trackers = row['total'] or 0
            print(f"\nTotal trackers: {total_trackers}")
            print(f"  Active:         {row['active'] or 0}")
            print(f"  Won (TP/Dir):   {row['won'] or 0}")
            print(f"  Lost (SL hit):  {row['lost_sl'] or 0}")
            print(f"  Expired:        {row['expired'] or 0}")
            print(f"\nBy source:")
            print(f"  Backfilled:     {row['backfilled'] or 0}")
            print(f"  Agent loop:     {row['from_agent_loop'] or 0}")
            print(f"  Standalone:     {row['from_standalone'] or 0}")

        conn.close()

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        print(f"\n❌  Backfill failed: {e}")
        print("Check logs for details.")
        sys.exit(1)

    print()
    print_separator()


if __name__ == "__main__":
    main()
