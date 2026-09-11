"""
mass_equity_sync.py - Mass sync of all pure equities & indices from 2022-01-01 to current date.
Batches 50 stocks with configurable pause and persistent checkpointing.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path

import config
import database
import downloader
import instruments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mass_sync")

CHECKPOINT_FILE = config.DATA_DIR / "mass_equity_sync_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {"completed_symbols": {}, "failed_symbols": {}, "total_bars": 0, "last_updated": None}


def save_checkpoint(cp: dict):
    cp["last_updated"] = datetime.now().isoformat()
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(cp, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def run_mass_sync(batch_size: int = 50, pause_seconds: int = 8, from_date: str = "2022-01-01"):
    database.init_db()
    
    # Get all active equity and index symbols
    all_syms = database.get_all_symbols()
    # Prioritize popular large caps and indices first
    priority = ["NIFTY", "BANKNIFTY", "SENSEX", "BANKEX", "FINNIFTY", "MIDCPNIFTY", "BSE500"] + config.NIFTY_50_SYMBOLS
    ordered_syms = []
    seen = set()
    for s in priority:
        if s in all_syms and s not in seen:
            ordered_syms.append(s)
            seen.add(s)
    for s in all_syms:
        if s not in seen and not s.startswith("0"):
            ordered_syms.append(s)
            seen.add(s)

    cp = load_checkpoint()
    completed = cp.get("completed_symbols", {})
    failed = cp.get("failed_symbols", {})
    total_bars = cp.get("total_bars", 0)

    remaining = [s for s in ordered_syms if s not in completed]
    total_symbols = len(ordered_syms)
    logger.info(f"Total Universe: {total_symbols} stocks & indices. Already completed: {len(completed)}. Remaining: {len(remaining)}.")

    today_str = datetime.today().strftime("%Y-%m-%d")

    # Group into batches of batch_size
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
    
    for b_idx, batch in enumerate(batches):
        batch_num = b_idx + 1
        logger.info(f"--- Starting Batch {batch_num}/{len(batches)} ({len(batch)} symbols) ---")
        batch_synced = 0

        for s_idx, sym in enumerate(batch):
            global_idx = len(completed) + 1
            try:
                # Sync candles from 2022-01-01 to today
                cnt = downloader.sync_symbol_history(sym, from_date=from_date, to_date=today_str)
                completed[sym] = {
                    "bars": cnt,
                    "date_synced": today_str,
                    "timestamp": datetime.now().isoformat()
                }
                total_bars += cnt
                batch_synced += 1
                logger.info(f"[{global_idx}/{total_symbols}] {sym}: Synced {cnt} bars (Batch {batch_num})")
                time.sleep(0.12)  # Safe rate limiter
            except Exception as e:
                logger.error(f"Error syncing {sym}: {e}")
                failed[sym] = str(e)

        # Save checkpoint after every batch
        cp["completed_symbols"] = completed
        cp["failed_symbols"] = failed
        cp["total_bars"] = total_bars
        save_checkpoint(cp)

        logger.info(f"Batch {batch_num} complete! Total progress: {len(completed)}/{total_symbols} symbols ({total_bars:,} total bars).")

        if b_idx < len(batches) - 1:
            logger.info(f"Pausing {pause_seconds}s before next batch of {batch_size} stocks...")
            time.sleep(pause_seconds)

    logger.info(f"🎉 Mass Equity Sync Completed! Successfully synced {len(completed)} stocks ({total_bars:,} total candles).")


if __name__ == "__main__":
    run_mass_sync(batch_size=50, pause_seconds=8, from_date="2022-01-01")
