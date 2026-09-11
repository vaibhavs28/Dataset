"""
resample_all_fake_stocks.py - Detects and resamples all stocks that have fake/simulated staircase 75m candles.
Strictly replaces them with authentic 75m candles resampled from 1-minute data.
"""

import time
import logging
import sqlite3
import pandas as pd
from typing import List

import config
import database
import parquet_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("resample_fake")


def find_symbols_with_fake_75m() -> List[str]:
    """
    Identifies all symbols in intraday_candles that contain fake linear staircase candles
    (e.g., where open==high and low==close on multiple bars, or identical volume across all 5 session slots).
    """
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT trading_symbol
            FROM intraday_candles
            WHERE timeframe = '75m' AND ((open = high AND low = close) OR (open = low AND high = close))
            GROUP BY trading_symbol
            HAVING count(*) >= 4;
        """)
        symbols_by_ohlc = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT DISTINCT trading_symbol FROM (
                SELECT trading_symbol, substr(timestamp, 1, 10) as dt
                FROM intraday_candles
                WHERE timeframe = '75m'
                GROUP BY trading_symbol, substr(timestamp, 1, 10)
                HAVING count(*) = 5 AND count(distinct volume) = 1 AND min(volume) > 0
            )
        """)
        symbols_by_vol = [row[0] for row in cursor.fetchall()]

    all_fake = sorted(list(set(symbols_by_ohlc + symbols_by_vol)))
    return all_fake


def purge_all_fake_staircase_candles():
    """Directly deletes all fake staircase bars where all 5 session bars had identical volume."""
    logger.info("Purging all fake staircase daily candle sets from intraday_candles...")
    with database.get_connection() as conn:
        cursor = conn.cursor()
        # Delete by day where 5 bars had identical synthetic volume
        cursor.execute("""
            DELETE FROM intraday_candles
            WHERE (trading_symbol, substr(timestamp, 1, 10)) IN (
                SELECT trading_symbol, substr(timestamp, 1, 10)
                FROM intraday_candles
                WHERE timeframe = '75m'
                GROUP BY trading_symbol, substr(timestamp, 1, 10)
                HAVING count(*) = 5 AND count(distinct volume) = 1 AND min(volume) > 0
            );
        """)
        deleted_vol = cursor.rowcount
        conn.commit()
    logger.info(f"Purged {deleted_vol:,} fake staircase bars with synthetic identical volume.")


def resample_all():
    database.init_db()
    
    # 1. Identify all affected symbols
    fake_symbols = find_symbols_with_fake_75m()
    parquet_symbols = set(parquet_loader.get_parquet_symbols())
    logger.info(f"Identified {len(fake_symbols)} symbols with fake staircase candles.")

    # 2. First purge synthetic identical volume bars
    purge_all_fake_staircase_candles()

    # 3. For symbols present in the authentic 1-minute Parquet dataset, resample fresh!
    resampled_count = 0
    skipped_count = 0
    
    for idx, sym in enumerate(fake_symbols):
        if sym in parquet_symbols:
            logger.info(f"[{idx + 1}/{len(fake_symbols)}] Resampling 75m candles for {sym} from authentic 1-minute Parquet...")
            ok = parquet_loader.import_symbol_to_database(sym)
            if ok:
                resampled_count += 1
        else:
            # If not in parquet, ensure any remaining fake records are wiped clean
            with database.get_connection() as conn:
                conn.cursor().execute("DELETE FROM intraday_candles WHERE trading_symbol = ? AND timeframe = '75m'", (sym,))
                conn.commit()
            skipped_count += 1

    logger.info(f"✅ Completed! Successfully resampled {resampled_count} symbols from authentic 1-min data. Cleaned {skipped_count} symbols.")


if __name__ == "__main__":
    resample_all()
