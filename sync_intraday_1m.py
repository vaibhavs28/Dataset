"""
sync_intraday_1m.py
-------------------
The Chartink Model: Single High-Speed 1-Minute Live Ingestion Daemon.
Fetches 1-minute intraday candles from Upstox API v2 and appends directly into
`hot_intraday.db` (local DuckDB / RAM table).

- Operates strictly 09:15 AM - 03:30 PM IST (Mon-Fri).
- Zero multi-timeframe disk resampling during live market hours.
- Zero Parquet rewriting.
- Scanner, alerts, and 75m broadcaster dynamically query `hot_intraday.db`.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Any
import concurrent.futures

import requests
import pandas as pd
import numpy as np

import config
import hot_intraday
import auto_nifty500_updater

IST = ZoneInfo("Asia/Kolkata")
STATUS_FILE = config.DATA_DIR / "nifty500_sync_status.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sync_intraday_1m")


def is_market_hours_ist(now: Optional[datetime] = None) -> bool:
    """Returns True strictly during Indian Stock Market regular trading hours (09:15 - 15:30 IST, Mon-Fri)."""
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    cur_time = now.time()
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    return market_open <= cur_time <= market_close


def fetch_symbol_1m_candles(
    instrument_key: str,
    access_token: str,
    days_back: int = 1
) -> Optional[pd.DataFrame]:
    """
    Fetches raw 1-minute intraday candles from Upstox API v2.
    """
    url = f"{config.UPSTOX_BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            candles = data.get("data", {}).get("candles", [])
            if candles:
                # Format: [timestamp, open, high, low, close, volume, open_interest]
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
        elif resp.status_code == 401:
            logger.warning("Upstox Access Token expired or unauthorized.")
        return None
    except Exception as e:
        logger.debug(f"Error fetching 1m candles for {instrument_key}: {e}")
        return None


def run_1m_sync_cycle(
    symbols: Optional[List[str]] = None,
    max_workers: int = 25,
    ignore_market_hours: bool = False
) -> Dict[str, Any]:
    """
    Executes a single fast 1-minute ingestion cycle across the requested universe:
    1. Fetches raw 1-min candles from Upstox in parallel.
    2. Writes directly into `hot_intraday.db`.
    Takes ~5 to 15 seconds across 500 stocks.
    """
    now_ist = datetime.now(IST)
    if not ignore_market_hours and not is_market_hours_ist(now_ist):
        msg = f"⏸️ Outside market hours ({now_ist.strftime('%H:%M:%S IST')}). 1m feed operates 09:15 - 15:30 IST Mon-Fri."
        logger.info(msg)
        return {"status": "SKIPPED", "reason": "OUTSIDE_MARKET_HOURS"}

    if not symbols:
        symbols = auto_nifty500_updater.get_nifty_500_symbols()

    total = len(symbols)
    start_t = datetime.now()
    token = config.UPSTOX_ACCESS_TOKEN

    logger.info(f"⚡ Starting 1-Minute Live Ingestion across {total} stocks ({max_workers} workers)...")

    # Map symbols to Upstox instrument keys
    import database
    all_inst = database.get_all_instruments(exchange="NSE_EQ")
    sym_to_key = {i["trading_symbol"].replace("-EQ", ""): i["instrument_key"] for i in all_inst if i.get("instrument_key")}

    all_rows = []
    synced_count = 0
    failed_count = 0

    def _worker(sym: str) -> Optional[pd.DataFrame]:
        clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
        ikey = sym_to_key.get(clean) or f"NSE_EQ|{clean}"
        if not token:
            return None
        df = fetch_symbol_1m_candles(ikey, token)
        if df is not None and not df.empty:
            df["symbol"] = clean
            return df
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, s): s for s in symbols}
        for fut in concurrent.futures.as_completed(futures):
            df_res = fut.result()
            if df_res is not None and not df_res.empty:
                all_rows.append(df_res)
                synced_count += 1
            else:
                failed_count += 1

    total_bars = 0
    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        total_bars = hot_intraday.append_1m_batch(combined_df)

    elapsed = (datetime.now() - start_t).total_seconds()

    status_data = {
        "status": "COMPLETED",
        "last_run_timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "feed_mode": "1M_HOT_INTRADAY",
        "total_symbols": total,
        "synced_symbols": synced_count,
        "failed_symbols": failed_count,
        "bars_1m_appended": total_bars,
        "elapsed_seconds": round(elapsed, 2),
        "speed_bars_per_sec": round(total_bars / max(elapsed, 0.01), 1)
    }

    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass

    logger.info(f"✨ 1-Min feed cycle completed in {elapsed:.2f}s ({total_bars} bars appended to hot_intraday.db).")
    return status_data


def run_daemon(max_workers: int = 25, poll_seconds: int = 60):
    """
    Continuous 1-minute live feed daemon during market hours.
    Sleeps until the start of the next minute.
    """
    logger.info("🛰️ Starting Chartink-Style 1-Minute Live Ingestion Daemon...")
    logger.info("Writing directly to hot_intraday.db with zero disk-file or multi-TF overhead.")

    symbols = auto_nifty500_updater.get_nifty_500_symbols()

    while True:
        try:
            now = datetime.now(IST)
            if is_market_hours_ist(now):
                run_1m_sync_cycle(symbols=symbols, max_workers=max_workers)
            else:
                time.sleep(15)
                continue

            # Sleep until next minute boundary
            now_after = datetime.now(IST)
            sec_to_next_min = 60 - now_after.second
            time.sleep(max(5, sec_to_next_min))
        except KeyboardInterrupt:
            logger.info("Stopped 1m live feed daemon.")
            break
        except Exception as e:
            logger.error(f"Error in 1m daemon loop: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Minute Live Feed Ingestion Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run continuously every 1 minute")
    parser.add_argument("--workers", type=int, default=25, help="Worker threads")
    parser.add_argument("--force", action="store_true", help="Ignore market hours for one-shot test")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(max_workers=args.workers)
    else:
        res = run_1m_sync_cycle(max_workers=args.workers, ignore_market_hours=args.force)
        print(f"Cycle Result: {res}")
