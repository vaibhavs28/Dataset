"""
download_1min.py - Authentic 1-Minute Historical Data Fetcher (Strictly Upstox API).
- 100% Upstox API only (no third-party GitHub raw or Yahoo Finance).
- Appends into active train shard (train-00008.parquet etc.) up to 1.60 GB before rolling over.
- Batches of 50 stocks with pause between batches and checkpoint resume.
- If historical data not available, downloads from 2022-01-01 to current date.
- Automatically resamples 1-min candles to session-aligned 75-min candles.
"""

import sys
import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd

import config
import database
import instruments
import upstox_parquet_updater
from parquet_loader import resample_1min_to_75min, import_symbol_to_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_1min")


def download_upstox_1min_symbol(
    symbol: str,
    from_date: str = "2022-01-01",
    to_date: Optional[str] = None
) -> dict:
    """
    Downloads 1-minute historical data for a symbol (equity or index) strictly from Upstox API.
    Appends into active train-*.parquet shard up to 1.60 GB.
    Resamples 1-min data to 75-min session candles and stores in SQLite.
    """
    sym = symbol.upper().strip()
    logger.info(f"Downloading 1-minute data for {sym} strictly from Upstox API...")
    res = upstox_parquet_updater.update_symbol_parquet_till_date(sym, save_to_shard=True)
    logger.info(f"Sync result for {sym}: status={res.get('status')}, bars_added={res.get('bars_added', 0)}")
    return res


def download_upstox_1min_batch(
    symbols: List[str],
    batch_size: int = 50,
    pause_seconds: int = 10,
    resume: bool = True
) -> dict:
    """
    Batch-downloads 1-minute historical data strictly from Upstox for a list of symbols.
    Appends into single train-*.parquet shard until 1.60 GB, rolling over to next shard.
    Pauses after every 50 stocks, saving checkpoint in sync_state.json.
    """
    logger.info(f"Starting Upstox 1-min batch download for {len(symbols)} symbols...")
    return upstox_parquet_updater.sync_database_in_batches(
        symbols=symbols,
        batch_size=batch_size,
        pause_seconds=pause_seconds,
        resume=resume
    )


if __name__ == "__main__":
    print("==================================================================")
    print("🚀 Upstox 1-Minute Historical Data Fetcher (100% Upstox API Only)")
    print("==================================================================")
    print("1. Download Index 1-min data via Upstox (NIFTY 50, BANKNIFTY)")
    print("2. Download Stock 1-min data via Upstox (e.g. RELIANCE, MARUTI)")
    print("3. Sync All Stocks in Batches of 50 with 10s pause (Single file till 1.60 GB)")

    choice = sys.argv[1] if len(sys.argv) > 1 else "1"

    if choice == "1":
        idx_sym = sys.argv[2] if len(sys.argv) > 2 else "NIFTY"
        download_upstox_1min_symbol(idx_sym)
    elif choice == "2":
        stock = sys.argv[2] if len(sys.argv) > 2 else "RELIANCE"
        download_upstox_1min_symbol(stock)
    elif choice == "3":
        database.init_db()
        all_syms = database.get_all_symbols()
        download_upstox_1min_batch(all_syms, batch_size=50, pause_seconds=10, resume=True)
    else:
        stock = sys.argv[1]
        download_upstox_1min_symbol(stock)
