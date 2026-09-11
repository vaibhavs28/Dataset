"""
generate_all_equity_data.py - High-Performance Authentic Upstox Market Data Engine.
- Scope: All active/liquid Equities (NSE Equities + BSE Group A/B). Illiquid BSE stocks load on demand.
- Speed: Multi-threaded chunk fetching (5 workers) + single batch write per 50 stocks (no 1GB file thrashing).
- Date Range: Daily candles from 2020-01-01 to today; 1-minute candles from 2022-01-01 to today (Upstox API max).
- Shards: Appends into single train-*.parquet shard until 1.60 GB, then rolls over to next shard.
- Resampling: Resamples 1-min bars into genuine 75-min session candles and updates SQLite.
- Checkpoint: Fault-tolerant batch checkpointing in data/equity_data_checkpoint.json.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
import database
import instruments
import downloader
import upstox_parquet_updater
from parquet_loader import resample_1min_to_75min

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("equity_engine")

CHECKPOINT_FILE = config.DATA_DIR / "equity_data_checkpoint.json"
PARQUET_DIR = config.DATA_DIR / "1min"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {
        "completed_symbols": {},
        "failed_symbols": {},
        "total_1min_bars": 0,
        "total_daily_bars": 0,
        "last_updated": None
    }


def save_checkpoint(cp: dict):
    cp["last_updated"] = datetime.now().isoformat()
    try:
        tmp_file = CHECKPOINT_FILE.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(cp, f, indent=2)
        tmp_file.replace(CHECKPOINT_FILE)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def get_target_equity_symbols() -> List[str]:
    """
    Returns active & liquid equity symbols:
    - All pure NSE Equities (3,374)
    - BSE Group A & B exclusive equities (124)
    Illiquid BSE stocks (Groups X, XT, T, Z, etc.) are excluded and loaded on demand.
    """
    database.init_db()
    with database.get_connection() as conn:
        cursor = conn.cursor()
        # 1. Pure NSE equities
        cursor.execute("""
            SELECT DISTINCT trading_symbol 
            FROM instruments 
            WHERE exchange = 'NSE_EQ' AND trading_symbol NOT LIKE '0%'
            ORDER BY trading_symbol ASC;
        """)
        nse_equities = [row[0] for row in cursor.fetchall()]

        # 2. BSE Group A & B equities not already in NSE
        cursor.execute("""
            SELECT DISTINCT trading_symbol 
            FROM instruments 
            WHERE exchange = 'BSE_EQ' AND instrument_type IN ('A', 'B') 
              AND trading_symbol NOT IN (SELECT trading_symbol FROM instruments WHERE exchange = 'NSE_EQ')
              AND trading_symbol NOT LIKE '0%'
            ORDER BY trading_symbol ASC;
        """)
        bse_liquid = [row[0] for row in cursor.fetchall()]

    # Prioritize popular large-caps first
    priority = [s for s in config.NIFTY_50_SYMBOLS if s in nse_equities]
    seen = set(priority)
    ordered = list(priority)

    for s in nse_equities:
        if s not in seen:
            ordered.append(s)
            seen.add(s)

    for s in bse_liquid:
        if s not in seen:
            ordered.append(s)
            seen.add(s)

    return ordered


def fetch_symbol_1min_parallel(sym: str, inst_key: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches 1-minute historical chunks in parallel using ThreadPoolExecutor with HTTP Keep-Alive."""
    chunks = upstox_parquet_updater.generate_date_chunks(start_dt, end_dt, chunk_days=28)
    if not chunks:
        return pd.DataFrame()

    def _fetch_one_chunk(dates):
        f_d, t_d = dates
        return upstox_parquet_updater.fetch_upstox_1min_chunk(inst_key, f_d, t_d)

    with ThreadPoolExecutor(max_workers=8) as executor:
        chunk_results = list(executor.map(_fetch_one_chunk, chunks))

    all_raw = []
    for c_list in chunk_results:
        if c_list:
            all_raw.extend(c_list)

    if not all_raw:
        return pd.DataFrame()

    df = upstox_parquet_updater.parse_upstox_candles_to_dataframe(sym, all_raw)
    return df


def run_accelerated_sync(batch_size: int = 50, pause_seconds: int = 10):
    database.init_db()
    symbols = get_target_equity_symbols()
    total_symbols = len(symbols)
    logger.info(f"🚀 Accelerated Engine: Target universe has {total_symbols} active liquid equities. Illiquid BSE stocks will load on demand.")

    cp = load_checkpoint()
    completed = cp.get("completed_symbols", {})
    failed = cp.get("failed_symbols", {})
    total_1min = cp.get("total_1min_bars", 0)
    total_daily = cp.get("total_daily_bars", 0)

    remaining = [s for s in symbols if s not in completed]
    logger.info(f"Already completed: {len(completed)}. Remaining to process: {len(remaining)} symbols.")

    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
    total_batches = len(batches)

    today_str = datetime.today().strftime("%Y-%m-%d")
    start_1min_dt = datetime(2022, 1, 1)
    end_1min_dt = datetime.now() + timedelta(days=1)

    for b_idx, batch in enumerate(batches):
        batch_num = b_idx + 1
        logger.info(f"\n=======================================================")
        logger.info(f"=== Starting Batch {batch_num}/{total_batches} ({len(batch)} equities) ===")
        logger.info(f"=======================================================")

        active_shard = upstox_parquet_updater.get_active_train_shard_path(min_shard_idx=0)
        shard_size_mb = active_shard.stat().st_size / (1024 * 1024) if active_shard.exists() else 0.0
        logger.info(f"Active Parquet Shard: {active_shard.name} ({shard_size_mb:.2f} MB / 1,600 MB)")

        batch_frames = []
        batch_75m_records = []
        batch_daily_records = []
        batch_updated_symbols = []

        batch_start_t = time.time()

        for s_idx, sym in enumerate(batch):
            global_idx = len(completed) + 1
            inst_key = instruments.resolve_instrument_key(sym) or f"NSE_EQ|{sym}"
            sym_t0 = time.time()

            try:
                # 1. Fetch clean daily candles from 2020-01-01 to today (single API call)
                daily_candles = downloader.fetch_upstox_daily_candles(inst_key, "2020-01-01", today_str)
                daily_cnt = len(daily_candles)
                
                if daily_candles:
                    for c in daily_candles:
                        c["trading_symbol"] = sym
                    batch_daily_records.extend(daily_candles)

                    # Prune 1-min chunk range to only when the stock actually started trading
                    earliest_date = min(c["date"] for c in daily_candles)
                    try:
                        earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
                        stock_start_1min = max(earliest_dt, start_1min_dt)
                    except Exception:
                        stock_start_1min = start_1min_dt

                    # 2. Fetch 1-min data in parallel (8 workers, stock_start_1min to today)
                    df_1min = fetch_symbol_1min_parallel(sym, inst_key, stock_start_1min, end_1min_dt)
                else:
                    df_1min = pd.DataFrame()

                bars_1min = len(df_1min) if not df_1min.empty else 0

                if not df_1min.empty:
                    batch_frames.append(df_1min)
                    batch_updated_symbols.append(sym)

                    # 3. Resample 1-min into authentic 75-min session candles
                    df_75m = resample_1min_to_75min(df_1min)
                    if not df_75m.empty:
                        df_reset = df_75m.reset_index()
                        ts_list = df_reset["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                        o_list = df_reset["open"].astype(float).tolist()
                        h_list = df_reset["high"].astype(float).tolist()
                        l_list = df_reset["low"].astype(float).tolist()
                        c_list = df_reset["close"].astype(float).tolist()
                        v_list = df_reset["volume"].astype(int).tolist()
                        for i in range(len(df_reset)):
                            batch_75m_records.append({
                                "instrument_key": inst_key,
                                "trading_symbol": sym,
                                "timeframe": "75m",
                                "timestamp": ts_list[i],
                                "open": o_list[i],
                                "high": h_list[i],
                                "low": l_list[i],
                                "close": c_list[i],
                                "volume": v_list[i]
                            })

                completed[sym] = {
                    "daily_bars": daily_cnt,
                    "1min_bars": bars_1min,
                    "active_shard": active_shard.name,
                    "synced_at": datetime.now().isoformat()
                }
                failed.pop(sym, None)
                total_daily += daily_cnt
                total_1min += bars_1min

                sym_elapsed = time.time() - sym_t0
                logger.info(f"[{global_idx}/{total_symbols}] {sym}: +{daily_cnt} daily bars, +{bars_1min:,} 1-min bars ({sym_elapsed:.1f}s)")
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")
                failed[sym] = str(e)

        # --- FLUSH ENTIRE BATCH TO DISK & SQLITE IN ONE SHOT ---
        flush_t0 = time.time()
        logger.info(f"Flushing Batch {batch_num} to disk & database...")

        # 1. Upsert daily candles in bulk
        if batch_daily_records:
            database.upsert_candles(batch_daily_records)

        # 2. Upsert 75m candles in bulk
        if batch_75m_records:
            # Wipe stale 75m candles for these symbols first
            with database.get_connection() as conn:
                cur = conn.cursor()
                cur.executemany(
                    "DELETE FROM intraday_candles WHERE trading_symbol = ? AND timeframe = '75m'",
                    [(s,) for s in batch_updated_symbols]
                )
                conn.commit()
            database.upsert_intraday_candles(batch_75m_records)

        # 3. Append all 50 stocks into active train-*.parquet shard ONCE
        if batch_frames:
            combined_df = pd.concat(batch_frames, ignore_index=True)
            active_shard = upstox_parquet_updater.append_dataframe_to_train_shard(combined_df)

        flush_elapsed = time.time() - flush_t0
        batch_total_time = time.time() - batch_start_t

        # Save checkpoint after each batch
        cp["completed_symbols"] = completed
        cp["failed_symbols"] = failed
        cp["total_1min_bars"] = total_1min
        cp["total_daily_bars"] = total_daily
        save_checkpoint(cp)

        new_size_mb = active_shard.stat().st_size / (1024 * 1024) if active_shard.exists() else 0.0
        logger.info(
            f"✅ Batch {batch_num}/{total_batches} finished in {batch_total_time:.1f}s! "
            f"Progress: {len(completed)}/{total_symbols} stocks. Active Shard: {active_shard.name} ({new_size_mb:.2f} MB / 1,600 MB)"
        )

        if b_idx < total_batches - 1:
            logger.info(f"Pausing {pause_seconds}s before next batch...")
            time.sleep(pause_seconds)

    logger.info(f"🎉 Complete! Synced {len(completed)} equities ({total_1min:,} 1-min bars, {total_daily:,} daily bars).")


if __name__ == "__main__":
    run_accelerated_sync(batch_size=50, pause_seconds=10)
