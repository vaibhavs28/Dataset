#!/usr/bin/env python3
"""
Purge all non-NSE instruments, BSE daily/intraday candles, and BSE parquet files,
leaving strictly pure NSE Equities and NSE Indices.
"""

import logging
from pathlib import Path
import database
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_nse")

def clean_to_nse_only():
    by_symbol_dir = config.DATA_DIR / "by_symbol"
    
    with database.get_connection() as conn:
        logger.info("Getting list of valid NSE symbols (NSE_EQ and NSE_INDEX)...")
        nse_syms = set(r[0] for r in conn.execute(
            "SELECT DISTINCT trading_symbol FROM instruments WHERE exchange IN ('NSE_EQ', 'NSE_INDEX');"
        ).fetchall())
        logger.info(f"Retained {len(nse_syms)} pure NSE symbols.")

        # 1. Purge instruments
        logger.info("Purging BSE / non-NSE instruments...")
        c1 = conn.execute("DELETE FROM instruments WHERE exchange NOT IN ('NSE_EQ', 'NSE_INDEX');").rowcount
        conn.commit()
        logger.info(f"Removed {c1} non-NSE instruments.")

        # 2. Purge daily candles
        logger.info("Purging BSE / non-NSE daily candles...")
        c2 = conn.execute("""
            DELETE FROM daily_candles 
            WHERE trading_symbol NOT IN (
                SELECT trading_symbol FROM instruments WHERE exchange IN ('NSE_EQ', 'NSE_INDEX')
            );
        """).rowcount
        conn.commit()
        logger.info(f"Removed {c2:,} non-NSE daily candle rows.")

        # 3. Purge intraday candles
        logger.info("Purging BSE / non-NSE intraday candles...")
        c3 = conn.execute("""
            DELETE FROM intraday_candles 
            WHERE trading_symbol NOT IN (
                SELECT trading_symbol FROM instruments WHERE exchange IN ('NSE_EQ', 'NSE_INDEX')
            );
        """).rowcount
        conn.commit()
        logger.info(f"Removed {c3:,} non-NSE intraday candle rows.")

    # 4. Remove BSE Parquet files
    if by_symbol_dir.exists():
        logger.info("Checking data/by_symbol for BSE-only parquet files...")
        deleted_files = 0
        freed_bytes = 0
        for p_file in list(by_symbol_dir.glob("*.parquet")):
            if p_file.stem not in nse_syms:
                freed_bytes += p_file.stat().st_size
                p_file.unlink()
                deleted_files += 1
        logger.info(f"Deleted {deleted_files} non-NSE parquet files ({freed_bytes / (1024*1024):.2f} MB freed).")

    # 5. Vacuum SQLite database to reclaim disk space
    logger.info("Vacuuming SQLite database to reclaim disk space...")
    with database.get_connection() as conn:
        conn.execute("VACUUM;")
    logger.info("✅ VACUUM complete!")

    # Final stats
    stats = database.get_db_stats()
    logger.info("=" * 50)
    logger.info(f"Final DB Stats: {stats}")
    remaining_files = len(list(by_symbol_dir.glob('*.parquet')))
    logger.info(f"Final Parquet Files in data/by_symbol: {remaining_files} files (strictly NSE Equities & Indices)")
    logger.info("=" * 50)


if __name__ == "__main__":
    clean_to_nse_only()
