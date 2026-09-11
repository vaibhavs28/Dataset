"""
24/7 Automated Market Data Engine & Sync Daemon
Runs continuously 24 hours a day, 7 days a week:
- Market Hours (Mon-Fri 09:15 - 15:30 IST): Periodic live sync of batch quotes and candles.
- Market Close (15:35 IST): Full End-of-Day candle reconciliation & sector analysis.
- Off-Market & Weekends: Intelligent low-power idle sleep with hourly heartbeat status.
- Self-healing: Catches all network/API exceptions, never crashes, auto-reconnects.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
import duckdb_store
import batch_downloader

# Configure persistent logging
LOG_FILE = config.DATA_DIR / "daemon_engine.log"
HEARTBEAT_FILE = config.DATA_DIR / "daemon_heartbeat.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("24x7_daemon")

IST = ZoneInfo("Asia/Kolkata")


def get_now_ist() -> datetime:
    """Returns current timestamp localized to Indian Standard Time (IST)."""
    return datetime.now(IST)


def is_market_open(now_dt: datetime) -> bool:
    """
    Indian equity markets trade Monday-Friday strictly between 09:15 and 15:30 IST.
    Returns True if current IST time is within active market hours.
    """
    # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday
    if now_dt.weekday() >= 5:
        return False
    current_minutes = now_dt.hour * 60 + now_dt.minute
    return (9 * 60 + 15) <= current_minutes <= (15 * 60 + 30)


def write_heartbeat(status: str, details: dict):
    """Writes persistent status to JSON for UI / monitoring tools."""
    try:
        data = {
            "timestamp": get_now_ist().isoformat(),
            "status": status,
            "pid": os.getpid(),
            "details": details
        }
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write heartbeat: {e}")


def run_eod_sync():
    """Performs full end-of-day market reconciliation."""
    logger.info("🌅 Running End-of-Day (EOD) Market Reconciliation...")
    try:
        res = batch_downloader.sync_live_market_batch()
        logger.info(f"✓ EOD Sync Complete: Updated {res.get('synced', 0)}/{res.get('total', 0)} equities in {res.get('duration', 0.0):.1f}s")
        write_heartbeat("EOD_COMPLETED", res)
    except Exception as e:
        logger.error(f"EOD Sync Exception: {e}", exc_info=True)
        write_heartbeat("EOD_ERROR", {"error": str(e)})


def run_intraday_sync():
    """Performs regular intraday interval sync during market hours."""
    logger.info("⚡ Market is Open. Syncing live batch market data...")
    try:
        res = batch_downloader.sync_live_market_batch()
        logger.info(f"✓ Live Intraday Sync Complete: {res.get('synced', 0)} equities updated in {res.get('duration', 0.0):.1f}s")
        write_heartbeat("INTRADAY_SYNCED", res)
    except Exception as e:
        logger.error(f"Intraday Sync Exception: {e}", exc_info=True)
        write_heartbeat("INTRADAY_ERROR", {"error": str(e)})


def main_daemon_loop(sync_interval_mins: int = 15):
    """
    Main 24*7 infinite control loop with intelligent adaptive sleep.
    """
    logger.info("=" * 65)
    logger.info("🚀 24*7 AUTOMATED MARKET DATA DAEMON STARTED")
    logger.info(f"   PID: {os.getpid()}")
    logger.info(f"   Timezone: Asia/Kolkata (IST)")
    logger.info(f"   Intraday Sync Interval: {sync_interval_mins} minutes")
    logger.info(f"   Log File: {LOG_FILE}")
    logger.info("=" * 65)

    last_eod_date = None

    while True:
        try:
            now_ist = get_now_ist()
            now_time_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
            today_date = now_ist.date()

            if is_market_open(now_ist):
                logger.info(f"[{now_time_str}] Market is LIVE (09:15 - 15:30 IST). Starting sync cycle...")
                run_intraday_sync()
                # Sleep for configured interval during market hours
                sleep_secs = sync_interval_mins * 60
                logger.info(f"Next intraday sync in {sync_interval_mins} minutes ({sleep_secs}s)...")
                time.sleep(sleep_secs)

            else:
                # Market is CLOSED
                # Check if we need to run EOD sync (weekday after 15:30, not run yet today)
                if now_ist.weekday() < 5 and now_ist.hour >= 15 and now_ist.minute >= 35 and last_eod_date != today_date:
                    run_eod_sync()
                    last_eod_date = today_date

                # Calculate time to next market open (09:15 IST)
                target_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
                if now_ist >= target_open:
                    target_open += timedelta(days=1)

                # Skip weekends
                while target_open.weekday() >= 5:
                    target_open += timedelta(days=1)

                wait_secs = (target_open - now_ist).total_seconds()
                wait_hours = wait_secs / 3600.0

                logger.info(f"[{now_time_str}] ⚪ Market Closed. Next open: {target_open.strftime('%A, %d %b %Y %H:%M IST')} (~{wait_hours:.1f} hours away).")
                write_heartbeat("MARKET_CLOSED", {"next_market_open": target_open.isoformat(), "wait_hours": round(wait_hours, 1)})

                # Sleep in 15-minute chunks so we can respond to shutdowns or time changes gracefully
                sleep_chunk = min(900, max(30, int(wait_secs)))
                time.sleep(sleep_chunk)

        except KeyboardInterrupt:
            logger.info("🛑 24*7 Daemon received shutdown signal. Exiting gracefully.")
            write_heartbeat("STOPPED", {"reason": "SIGINT"})
            break
        except Exception as e:
            logger.error(f"Unexpected error in daemon loop: {e}", exc_info=True)
            write_heartbeat("EXCEPTION", {"error": str(e)})
            time.sleep(30)


if __name__ == "__main__":
    interval = 15
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        interval = int(sys.argv[1])
    main_daemon_loop(sync_interval_mins=interval)
