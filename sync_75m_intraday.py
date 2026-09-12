"""
sync_75m_intraday.py
--------------------
Automated 75-Minute Intraday Data Synchronization Engine for Upstox API.

Features:
1. Ingests live 1-minute OHLCV candles from Upstox API v2 for all target equity symbols.
2. Strictly aligns and resamples to genuine 75-minute candles for the 5 daily NSE sessions:
     - Candle 1: 09:15 - 10:30 IST
     - Candle 2: 10:30 - 11:45 IST
     - Candle 3: 11:45 - 13:00 IST
     - Candle 4: 13:00 - 14:15 IST
     - Candle 5: 14:15 - 15:30 IST
3. Persists candle data across:
     - DuckDB native `intraday_candles` (75m) & `daily_candles` (forming bar)
     - SQLite `intraday_candles` (75m) & `candles` (forming bar)
     - Dedicated Parquet storage (`data/by_symbol/{symbol}.parquet`)
4. Compliant with Upstox API v2 rate limits (20 req/s, 480 req/min) via thread-safe rate limiter.
5. Multi-threaded batch execution with high throughput.
6. Operates as an automated background daemon at candle closes, or callable on-demand via CLI / UI.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
import database
import instruments
import parquet_loader
import upstox_parquet_updater
from upstox_parquet_updater import UpstoxRateLimiter, _global_rate_limiter

try:
    import duckdb_store
except ImportError:
    duckdb_store = None

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("sync_75m_intraday")

# Scheduled 75-Minute Candle Close Times in IST (hours, minutes, label)
CANDLE_CLOSE_SCHEDULE = [
    (10, 30, "Candle 1 (09:15 - 10:30 IST)"),
    (11, 45, "Candle 2 (10:30 - 11:45 IST)"),
    (13, 0,  "Candle 3 (11:45 - 13:00 IST)"),
    (14, 15, "Candle 4 (13:00 - 14:15 IST)"),
    (15, 30, "Candle 5 (14:15 - 15:30 IST)")
]


def resolve_symbol_keys_batch(symbols: List[str]) -> Dict[str, str]:
    """
    Fast resolution of instrument keys for a list of symbols.
    Checks instruments table in DuckDB / SQLite in a single batch query.
    """
    key_map = {}
    clean_syms = [s.upper().strip() for s in symbols]

    # Preload from known index map
    for s in clean_syms:
        if s in instruments.INDEX_KEY_MAP:
            key_map[s] = instruments.INDEX_KEY_MAP[s]

    remaining = [s for s in clean_syms if s not in key_map]
    if not remaining:
        return key_map

    # Query database for all remaining symbols at once
    try:
        with database.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(remaining))
            cursor.execute(f"""
                SELECT trading_symbol, instrument_key 
                FROM instruments 
                WHERE trading_symbol IN ({placeholders})
                ORDER BY 
                    CASE WHEN instrument_key LIKE '%|INE%' OR instrument_key LIKE '%|INF%' OR instrument_key LIKE '%|IN9%' THEN 0 ELSE 1 END,
                    CASE WHEN exchange LIKE 'NSE%' THEN 1 ELSE 2 END;
            """, remaining)
            for row in cursor.fetchall():
                sym, ikey = row[0].upper(), row[1]
                if sym not in key_map:
                    key_map[sym] = ikey
    except Exception as e:
        logger.warning(f"Batch key resolution fallback: {e}")

    # For any symbols still unmapped, use fallback resolver
    for s in remaining:
        if s not in key_map:
            resolved = instruments.resolve_instrument_key(s)
            if resolved:
                key_map[s] = resolved

    return key_map


def fetch_and_resample_symbol(
    symbol: str,
    instrument_key: Optional[str] = None,
    rate_limiter: Optional[UpstoxRateLimiter] = None,
    save_parquet: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Worker task:
    1. Fetches today's live 1-min bars from Upstox API for a single symbol.
    2. Resamples to 75m candles.
    3. If save_parquet is True, updates the symbol's parquet partition.
    4. Prepares daily forming bar and 75m records for database upsert.

    Returns (daily_records, records_75, result_summary).
    """
    sym = symbol.upper().strip()
    limiter = rate_limiter or _global_rate_limiter

    inst_key = instrument_key or instruments.resolve_instrument_key(sym)
    if not inst_key:
        return [], [], {"symbol": sym, "success": False, "error": "Instrument key not found"}

    try:
        raw_bars = upstox_parquet_updater.fetch_upstox_1min_intraday(inst_key, rate_limiter=limiter)
        if not raw_bars:
            return [], [], {"symbol": sym, "success": False, "error": "No intraday candles returned"}

        # Parse to DataFrame
        df_today = upstox_parquet_updater.parse_upstox_candles_to_dataframe(sym, raw_bars)

        # Save to single-symbol Parquet
        if save_parquet and not df_today.empty:
            try:
                upstox_parquet_updater.append_bars_to_symbol_file(sym, df_today)
            except Exception as pe:
                logger.debug(f"Parquet append warning for {sym}: {pe}")

        # Form today's live daily candle
        first_bar = raw_bars[-1]
        latest_bar = raw_bars[0]
        today_date = first_bar[0].split("T")[0]

        daily_records = [{
            "instrument_key": inst_key,
            "trading_symbol": sym,
            "date": today_date,
            "open": float(first_bar[1]),
            "high": float(max(b[2] for b in raw_bars)),
            "low": float(min(b[3] for b in raw_bars)),
            "close": float(latest_bar[4]),
            "volume": int(sum(b[5] for b in raw_bars if b[5])),
            "open_interest": int(latest_bar[6]) if len(latest_bar) > 6 and latest_bar[6] else 0
        }]

        # Resample to 75-minute candles
        records_75 = []
        try:
            df_75_today = parquet_loader.resample_1min_to_75min(df_today)
            if not df_75_today.empty:
                for ts, row in df_75_today.iterrows():
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    records_75.append({
                        "instrument_key": inst_key,
                        "trading_symbol": sym,
                        "timeframe": "75m",
                        "timestamp": ts_str,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume", 0))
                    })
        except Exception as e75:
            logger.warning(f"Could not resample 75m for {sym}: {e75}")

        meta = {
            "symbol": sym,
            "success": True,
            "bars_1m": len(raw_bars),
            "bars_75m": len(records_75),
            "latest_close": float(latest_bar[4]),
            "today_date": today_date,
            "error": None
        }
        return daily_records, records_75, meta

    except Exception as e:
        return [], [], {"symbol": sym, "success": False, "error": str(e)}


def sync_symbol_75m_from_upstox(
    symbol: str,
    instrument_key: Optional[str] = None,
    rate_limiter: Optional[UpstoxRateLimiter] = None,
    save_parquet: bool = True,
    save_db: bool = True
) -> Dict[str, Any]:
    """
    Synchronizes 75m intraday data for a single symbol immediately and writes to DB.
    """
    daily_records, records_75, meta = fetch_and_resample_symbol(
        symbol=symbol,
        instrument_key=instrument_key,
        rate_limiter=rate_limiter,
        save_parquet=save_parquet
    )

    if meta["success"] and save_db:
        if daily_records:
            database.upsert_candles(daily_records)
        if records_75:
            database.upsert_intraday_candles(records_75)

    return meta


def sync_all_symbols_75m(
    symbols: Optional[List[str]] = None,
    universe: str = "Nifty 50",
    max_workers: int = 10,
    save_parquet: bool = True,
    save_db: bool = True,
    batch_commit_size: int = 50,
    progress_callback = None
) -> Dict[str, Any]:
    """
    High-throughput multi-threaded 75-minute intraday synchronization for target stocks.

    Parameters:
    - symbols: Optional custom list of symbols. If None, resolves from universe.
    - universe: 'Nifty 50', 'Nifty 100', 'Top 200 Liquid Equities', 'Nifty 500', or 'All Database Equities'.
    - max_workers: Number of concurrent worker threads (default 10).
    - save_parquet: Whether to append to per-symbol parquet files.
    - save_db: Whether to commit to SQLite & DuckDB.
    - batch_commit_size: Number of symbols to accumulate before committing to database.
    - progress_callback: Optional callable(completed_count, total_count, current_symbol).

    Returns summary dictionary with detailed execution metrics.
    """
    start_time = datetime.now()

    # Determine symbols list
    if symbols is None or len(symbols) == 0:
        import auto_75m_broadcaster
        symbols = auto_75m_broadcaster.get_target_equities(universe)

    total_symbols = len(symbols)
    logger.info(f"🚀 Initiating 75-Min Intraday Upstox Sync for {total_symbols} stocks (Universe: {universe}, Workers: {max_workers})...")

    # Step 1: Pre-resolve all instrument keys
    key_map = resolve_symbol_keys_batch(symbols)

    success_count = 0
    failed_count = 0
    total_1m_bars = 0
    total_75m_bars = 0
    errors = []

    daily_buffer = []
    intraday_buffer = []

    rate_limiter = UpstoxRateLimiter(max_per_sec=20.0, max_per_min=480)

    def commit_buffers():
        nonlocal daily_buffer, intraday_buffer
        if not save_db:
            daily_buffer.clear()
            intraday_buffer.clear()
            return

        if daily_buffer:
            try:
                database.upsert_candles(daily_buffer)
            except Exception as e:
                logger.error(f"Error committing batch daily candles: {e}")
            daily_buffer.clear()

        if intraday_buffer:
            try:
                database.upsert_intraday_candles(intraday_buffer)
            except Exception as e:
                logger.error(f"Error committing batch intraday candles: {e}")
            intraday_buffer.clear()

    # Step 2: Concurrently fetch and resample
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(
                fetch_and_resample_symbol,
                sym,
                key_map.get(sym),
                rate_limiter,
                save_parquet
            ): sym for sym in symbols
        }

        completed = 0
        for future in as_completed(future_to_sym):
            completed += 1
            sym = future_to_sym[future]
            try:
                daily_recs, recs_75, meta = future.result()
                if meta["success"]:
                    success_count += 1
                    total_1m_bars += meta["bars_1m"]
                    total_75m_bars += meta["bars_75m"]
                    daily_buffer.extend(daily_recs)
                    intraday_buffer.extend(recs_75)
                else:
                    failed_count += 1
                    errors.append(f"{sym}: {meta.get('error', 'unknown')}")

                # Flush database buffers periodically
                if len(daily_buffer) >= batch_commit_size:
                    commit_buffers()

            except Exception as exc:
                failed_count += 1
                errors.append(f"{sym}: {exc}")

            if progress_callback:
                try:
                    progress_callback(completed, total_symbols, sym)
                except Exception:
                    pass

            if completed % 25 == 0 or completed == total_symbols:
                logger.info(f"⏳ Progress: [{completed}/{total_symbols}] symbols processed ({success_count} synced, {failed_count} skipped/error)")

    # Final commit of remaining records in buffer
    commit_buffers()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ 75-Min Intraday Upstox Sync finished in {elapsed:.2f}s.")
    logger.info(f"📊 Summary: {success_count}/{total_symbols} synced successfully, {total_75m_bars} 75m bars and {total_1m_bars} 1m bars ingested.")

    return {
        "universe": universe,
        "total_symbols": total_symbols,
        "synced_count": success_count,
        "failed_count": failed_count,
        "total_1m_bars": total_1m_bars,
        "total_75m_bars": total_75m_bars,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "errors": errors[:50]
    }


def run_75m_sync_daemon(
    universe: str = "All Database Equities",
    max_workers: int = 10,
    poll_interval: int = 10
):
    """
    Dedicated background daemon that monitors market hours and syncs all symbols
    immediately upon reaching every 75-minute candle close.
    """
    logger.info("🛰️ Starting Standalone 75-Minute Intraday Upstox Sync Daemon...")
    logger.info(f"Scheduled close times: 10:30, 11:45, 13:00, 14:15, 15:30 IST (Mon-Fri)")
    logger.info(f"Universe: {universe} | Concurrent Workers: {max_workers}")

    last_triggered_slot = None

    while True:
        try:
            now = datetime.now()
            is_weekday = (now.weekday() < 5)

            if is_weekday:
                cur_date_str = now.strftime("%Y-%m-%d")
                for h, m, label in CANDLE_CLOSE_SCHEDULE:
                    if (now.hour == h) and (now.minute == m):
                        slot_key = (cur_date_str, h, m)
                        if last_triggered_slot != slot_key:
                            logger.info(f"🔔 CANDLE CLOSE TRIGGER: Syncing intraday 75m data for {label}...")
                            last_triggered_slot = slot_key
                            sync_all_symbols_75m(
                                universe=universe,
                                max_workers=max_workers
                            )
                            break

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Sync daemon stopped by user.")
            break
        except Exception as err:
            logger.error(f"Sync daemon error: {err}", exc_info=True)
            time.sleep(poll_interval)


# =========================================================================
# CLI COMMAND LINE INTERFACE
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="75-Minute Upstox Intraday Candle Database Sync")
    parser.add_argument("--universe", type=str, default="Nifty 50", help="Universe: 'Nifty 50', 'Nifty 100', 'Top 200 Liquid Equities', 'Nifty 500', 'All Database Equities'")
    parser.add_argument("--symbol", type=str, help="Sync a single symbol immediately (e.g. --symbol RELIANCE)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent worker threads (default 10)")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in background syncing at every 75m close")
    parser.add_argument("--sync-now", action="store_true", help="Sync all symbols in universe once immediately and exit")

    args = parser.parse_args()

    if args.symbol:
        sym = args.symbol.upper().strip()
        print(f"Fetching and syncing 75m intraday data from Upstox for {sym}...")
        res = sync_symbol_75m_from_upstox(sym)
        print("Result:", res)
    elif args.daemon:
        run_75m_sync_daemon(universe=args.universe, max_workers=args.workers)
    else:
        # Default action: sync universe now
        res = sync_all_symbols_75m(universe=args.universe, max_workers=args.workers)
        print(f"\n--- Sync Results Summary ---")
        print(f"Universe: {res['universe']}")
        print(f"Total: {res['total_symbols']}, Synced: {res['synced_count']}, Failed/Skipped: {res['failed_count']}")
        print(f"75m Bars Ingested: {res['total_75m_bars']}, 1m Bars: {res['total_1m_bars']}")
        print(f"Elapsed Time: {res['elapsed_seconds']}s")
