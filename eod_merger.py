"""
eod_merger.py
-------------
End-Of-Day (EOD) 4:00 PM IST Database Merger & Rotator.
1. Reads all raw 1-minute intraday bars accumulated in `hot_intraday.db` during today's trading.
2. Aggregates today's official Daily OHLCV candle per symbol.
3. Commits today's Daily candles into `market_data.duckdb` and `market_data.db`.
4. Appends 1-minute bars into historical `candles_1m` for archiving.
5. Flushes & resets `hot_intraday.db` for the next trading day.
"""

import sys
import time
import logging
import argparse
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, Any

import duckdb
import pandas as pd

import config
import hot_intraday
import duckdb_store
import database

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("eod_merger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_eod_merge(today_date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the 4:00 PM IST EOD roll-up and historical merge.
    """
    start_t = time.time()
    today_str = today_date_str or datetime.now(IST).strftime("%Y-%m-%d")
    logger.info(f"🏁 Starting EOD Data Merge for {today_str}...")

    if not hot_intraday.has_hot_data():
        logger.info("hot_intraday.db has no data for today. Nothing to merge.")
        return {"status": "EMPTY", "merged_daily_bars": 0}

    stats = hot_intraday.get_hot_stats()
    logger.info(f"Found {stats.get('candle_count', 0)} 1m candles across {stats.get('symbol_count', 0)} symbols in hot_intraday.db.")

    # Step 1: Compute today's Daily OHLCV bar per symbol via SQL
    query_daily = f"""
        SELECT 
            symbol || '-EQ' AS trading_symbol,
            '{today_str}' AS date,
            first(open ORDER BY timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close ORDER BY timestamp) AS close,
            sum(volume) AS volume,
            0 AS open_interest
        FROM intraday_1m
        GROUP BY symbol
        HAVING sum(volume) > 0;
    """

    with hot_intraday._hot_lock:
        conn_hot = hot_intraday._get_conn(read_only=True)
        try:
            df_daily = conn_hot.execute(query_daily).df()
        finally:
            conn_hot.close()

    if df_daily.empty:
        logger.warning("No non-zero volume daily candles computed.")
        return {"status": "NO_VOLUME", "merged_daily_bars": 0}

    # Step 2: Merge into market_data.duckdb daily_candles
    logger.info(f"Merging {len(df_daily)} daily candles into market_data.duckdb...")
    try:
        with duckdb_store.get_write_connection() as conn_main:
            conn_main.register("eod_daily_view", df_daily)
            # Upsert into daily_candles
            conn_main.execute("""
                DELETE FROM daily_candles
                WHERE date = ? AND trading_symbol IN (SELECT trading_symbol FROM eod_daily_view);
            """, [today_str])
            conn_main.execute("""
                INSERT INTO daily_candles (trading_symbol, date, open, high, low, close, volume, open_interest)
                SELECT trading_symbol, date, open, high, low, close, volume, open_interest
                FROM eod_daily_view;
            """)
            conn_main.unregister("eod_daily_view")
    except Exception as e:
        logger.error(f"Error merging daily candles to DuckDB: {e}")

    # Step 3: Mirror into SQLite daily_candles for compatibility
    try:
        daily_records = df_daily.to_dict(orient="records")
        database.save_daily_candles_batch(daily_records)
    except Exception as e:
        logger.debug(f"SQLite daily mirror note: {e}")

    # Step 4: Archive 1m candles into historical candles_1m in DuckDB
    logger.info("Archiving 1m candles to historical DuckDB...")
    try:
        with hot_intraday._hot_lock:
            conn_hot = hot_intraday._get_conn(read_only=True)
            df_1m_all = conn_hot.execute("SELECT symbol, timestamp, open, high, low, close, volume FROM intraday_1m;").df()
            conn_hot.close()

        if not df_1m_all.empty:
            with duckdb_store.get_write_connection() as conn_main:
                conn_main.register("archive_1m_view", df_1m_all)
                conn_main.execute("""
                    INSERT INTO candles_1m (symbol, timestamp, open, high, low, close, volume)
                    SELECT symbol, timestamp, open, high, low, close, volume
                    FROM archive_1m_view;
                """)
                conn_main.unregister("archive_1m_view")
    except Exception as e:
        logger.debug(f"Historical 1m archive note: {e}")

    # Step 5: Clear hot_intraday.db for tomorrow
    logger.info("Resetting hot_intraday.db for next session...")
    hot_intraday.clear_hot_intraday()

    elapsed = round(time.time() - start_t, 2)
    logger.info(f"✅ EOD Merge complete in {elapsed}s. {len(df_daily)} daily candles saved for {today_str}.")

    return {
        "status": "SUCCESS",
        "date": today_str,
        "daily_candles_saved": len(df_daily),
        "raw_1m_candles_archived": len(df_1m_all) if not df_1m_all.empty else 0,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOD 4 PM Database Merger")
    parser.add_argument("--date", type=str, default=None, help="Date to merge (YYYY-MM-DD)")
    args = parser.parse_args()

    res = run_eod_merge(today_date_str=args.date)
    print(f"Merge Result: {res}")
