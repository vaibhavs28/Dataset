"""
auto_nifty500_updater.py
-------------------------
High-Speed Automated 15-Minute Intraday (1-Min & 75-Min) Database Updater for Nifty 500.

Features:
1. Updates Nifty 500 symbols every 15 minutes (:00, :15, :30, :45 marks).
2. Strictly active only during IST Market Hours (09:00 AM to 04:00 PM IST, Monday - Friday).
3. Maximum throughput: 20 concurrent threads with Upstox API v2 Keep-Alive session and thread-safe rate limiter.
4. Sub-second batch ingestion directly into DuckDB `candles_1m`, `intraday_candles` (75m), and `daily_candles`.
5. Zero UI lag: Operates as an independent background daemon process with atomic status tracking.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from typing import List, Dict, Any, Optional
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timezone
    IST = timezone(timedelta(hours=5, minutes=30))

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import database
import parquet_loader
import sync_75m_intraday

# Status tracking file
STATUS_FILE = PROJECT_ROOT / "data" / "nifty500_sync_status.json"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("auto_nifty500_updater")


def is_market_hours_ist(now: Optional[datetime] = None) -> bool:
    """
    Checks whether current time is strictly within the IST market trading window:
    Monday through Friday between 09:00:00 AM IST and 04:00:00 PM IST (09:00 - 16:00).
    """
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    # Mon=0, Fri=4, Sat=5, Sun=6
    if now.weekday() >= 5:
        return False

    t = now.time()
    return (dtime(9, 0, 0) <= t <= dtime(16, 0, 0))


def get_nifty_500_symbols() -> List[str]:
    """
    Resolves top 500 liquid equities for Nifty 500, prioritizing Nifty 50 and active swing stocks.
    """
    all_db_syms = database.get_all_symbols()
    parquet_syms = parquet_loader.get_parquet_symbols()
    all_equities = sorted(list(set(all_db_syms + parquet_syms)))
    all_equities = [s for s in all_equities if not s.startswith("0")]

    seen = set()
    nifty500 = []

    # 1. Nifty 50 constituents first
    for s in config.NIFTY_50_SYMBOLS:
        if s in all_equities and s not in seen:
            seen.add(s)
            nifty500.append(s)

    # 2. Curated swing stocks (high liquidity & momentum)
    for s in config.SWING_STOCK_SYMBOLS:
        if s in all_equities and s not in seen:
            seen.add(s)
            nifty500.append(s)
        if len(nifty500) >= 500:
            break

    # 3. Fill up to 500 from all equities if needed
    if len(nifty500) < 500:
        for s in all_equities:
            if s not in seen:
                seen.add(s)
                nifty500.append(s)
            if len(nifty500) >= 500:
                break

    return nifty500


def update_status_file(data: Dict[str, Any]):
    """Atomically writes current updater status to data/nifty500_sync_status.json."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = STATUS_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(STATUS_FILE)
    except Exception as e:
        logger.warning(f"Could not update status file: {e}")


def get_sync_status() -> Dict[str, Any]:
    """Reads latest updater status from data/nifty500_sync_status.json."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "UNKNOWN",
        "last_run_timestamp": "Never",
        "next_run_timestamp": "Unknown",
        "total_symbols": 0,
        "synced_symbols": 0,
        "elapsed_seconds": 0
    }


def run_nifty500_sync_cycle(
    max_workers: int = 20,
    ignore_market_hours: bool = False
) -> Dict[str, Any]:
    """
    Executes a single high-speed sync cycle for Nifty 500:
    - Ingests 1-minute intraday bars from Upstox
    - Resamples to 75m bars
    - Commits directly into DuckDB candles_1m, intraday_candles, and daily_candles
    - Updates per-symbol Parquet files
    """
    now_ist = datetime.now(IST)

    if not ignore_market_hours and not is_market_hours_ist(now_ist):
        msg = f"Skipping sync: Outside IST market hours (09:00 to 16:00 Mon-Fri). Current IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}"
        logger.info(msg)
        update_status_file({
            "status": "OUTSIDE_MARKET_HOURS",
            "message": msg,
            "current_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
            "market_window": "09:00 - 16:00 IST (Mon-Fri)"
        })
        return {"status": "SKIPPED", "reason": "OUTSIDE_MARKET_HOURS"}

    symbols = get_nifty_500_symbols()
    total = len(symbols)
    logger.info(f"⚡ Starting 15-Minute Nifty 500 Database Update ({total} symbols, {max_workers} workers)...")

    start_time = datetime.now(IST)
    update_status_file({
        "status": "SYNCING",
        "started_at": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_symbols": total,
        "market_window": "09:00 - 16:00 IST (Mon-Fri)"
    })

    # Execute high-throughput sync
    results = sync_75m_intraday.sync_all_symbols_75m(
        symbols=symbols,
        universe="Nifty 500",
        max_workers=max_workers,
        save_parquet=True,
        save_db=True,
        batch_commit_size=50
    )

    end_time = datetime.now(IST)
    elapsed = (end_time - start_time).total_seconds()

    # Calculate next scheduled 15-minute slot
    next_minute = (now_ist.minute // 15 + 1) * 15
    if next_minute >= 60:
        next_run = (now_ist + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_run = now_ist.replace(minute=next_minute, second=0, microsecond=0)

    status_data = {
        "status": "COMPLETED",
        "last_run_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "next_run_timestamp": next_run.strftime("%Y-%m-%d %H:%M:%S IST"),
        "universe": "Nifty 500",
        "total_symbols": total,
        "synced_symbols": results.get("synced_count", 0),
        "failed_symbols": results.get("failed_count", 0),
        "bars_1m_ingested": results.get("total_1m_bars", 0),
        "bars_75m_ingested": results.get("total_75m_bars", 0),
        "elapsed_seconds": round(elapsed, 2),
        "speed_bars_per_second": round(results.get("total_1m_bars", 0) / max(elapsed, 0.01), 1),
        "market_window": "09:00 - 16:00 IST (Mon-Fri)"
    }

    update_status_file(status_data)
    logger.info(f"✨ Nifty 500 sync completed in {elapsed:.2f}s ({results.get('total_1m_bars', 0)} 1m bars ingested). Next sync: {next_run.strftime('%H:%M:%S IST')}")
    return status_data


def run_15m_daemon(max_workers: int = 20):
    """
    Continuous background daemon running every 15 minutes during IST market hours.
    Syncs precisely at :00, :15, :30, :45 minute marks.
    """
    logger.info("🛰️ Starting Nifty 500 15-Minute Upstox Sync Daemon...")
    logger.info("Operating Window: 09:00 AM - 04:00 PM IST (Monday - Friday)")
    logger.info(f"Target Universe: Nifty 500 ({max_workers} concurrent threads)")

    last_slot = None

    while True:
        try:
            now_ist = datetime.now(IST)

            # Check market hours
            if is_market_hours_ist(now_ist):
                slot_id = (now_ist.strftime("%Y-%m-%d"), now_ist.hour, now_ist.minute // 15)

                if (now_ist.minute % 15 == 0) and (last_slot != slot_id):
                    logger.info(f"🔔 15-Minute Interval Trigger: {now_ist.strftime('%H:%M:%S IST')}")
                    last_slot = slot_id
                    run_nifty500_sync_cycle(max_workers=max_workers)

            else:
                update_status_file({
                    "status": "OUTSIDE_MARKET_HOURS",
                    "current_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "market_window": "09:00 - 16:00 IST (Mon-Fri)"
                })

            time.sleep(10)

        except KeyboardInterrupt:
            logger.info("Nifty 500 15m daemon terminated by user.")
            break
        except Exception as e:
            logger.error(f"Nifty 500 daemon loop error: {e}", exc_info=True)
            time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty 500 15-Minute Intraday Database Updater")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in background every 15 minutes")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle now and exit")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent worker threads (default: 20)")
    parser.add_argument("--ignore-market-hours", action="store_true", help="Force sync execution outside market hours")

    args = parser.parse_args()

    if args.daemon:
        run_15m_daemon(max_workers=args.workers)
    else:
        res = run_nifty500_sync_cycle(max_workers=args.workers, ignore_market_hours=args.ignore_market_hours)
        print("\n--- Nifty 500 15-Min Sync Summary ---")
        for k, v in res.items():
            print(f"{k}: {v}")
