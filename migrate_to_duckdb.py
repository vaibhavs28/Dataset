"""
Migration script to migrate instruments and daily_candles from SQLite (market_data.db)
into ultra-fast DuckDB (market_data.duckdb).
"""
import time
import sqlite3
import logging
from pathlib import Path
import pandas as pd
import duckdb

import config
import duckdb_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_to_duckdb")


def migrate_sqlite_to_duckdb(chunk_size: int = 500_000):
    sqlite_path = config.DB_PATH
    if not sqlite_path.exists():
        logger.warning(f"SQLite database {sqlite_path} does not exist. Nothing to migrate.")
        return

    duck_conn = duckdb_store.get_connection()

    logger.info(f"Connecting to SQLite: {sqlite_path}")
    s_conn = sqlite3.connect(str(sqlite_path))

    # 1. Migrate instruments
    logger.info("Migrating instruments...")
    t0 = time.time()
    try:
        df_inst = pd.read_sql("SELECT * FROM instruments", s_conn)
        if not df_inst.empty:
            duck_conn.register("_temp_inst", df_inst)
            duck_conn.execute("""
                INSERT OR REPLACE INTO instruments
                SELECT * FROM _temp_inst;
            """)
            duck_conn.unregister("_temp_inst")
            logger.info(f"Successfully migrated {len(df_inst):,} instruments in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"Error migrating instruments: {e}")

    # 2. Migrate daily_candles in chunks
    logger.info("Migrating daily_candles...")
    t0 = time.time()
    total_candles = 0
    try:
        cur = s_conn.cursor()
        cur.execute("SELECT count(*) FROM daily_candles;")
        total_in_sqlite = cur.fetchone()[0]
        logger.info(f"Total daily candles to migrate: {total_in_sqlite:,}")

        for i, chunk in enumerate(pd.read_sql("SELECT * FROM daily_candles", s_conn, chunksize=chunk_size)):
            t_chunk = time.time()
            duck_conn.register("_temp_chunk", chunk)
            duck_conn.execute("""
                INSERT OR REPLACE INTO daily_candles
                SELECT 
                    instrument_key,
                    trading_symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    open_interest
                FROM _temp_chunk;
            """)
            duck_conn.unregister("_temp_chunk")
            total_candles += len(chunk)
            logger.info(f"Chunk {i+1}: Migrated {len(chunk):,} candles (Total: {total_candles:,}/{total_in_sqlite:,}) in {time.time()-t_chunk:.2f}s")

        logger.info(f"Finished daily_candles migration! Total {total_candles:,} rows in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"Error migrating daily_candles: {e}")

    # 3. Create indices for sub-millisecond lookups
    logger.info("Creating DuckDB indexes on daily_candles...")
    t_idx = time.time()
    try:
        duck_conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_sym_date ON daily_candles (trading_symbol, date);")
        duck_conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_date ON daily_candles (date);")
        logger.info(f"Indexes created in {time.time()-t_idx:.2f}s")
    except Exception as e:
        logger.warning(f"Index creation note: {e}")

    # Verification
    inst_count = duck_conn.execute("SELECT count(*) FROM instruments;").fetchone()[0]
    daily_count = duck_conn.execute("SELECT count(*) FROM daily_candles;").fetchone()[0]
    logger.info(f"DuckDB Verification: {inst_count:,} instruments, {daily_count:,} daily candles.")
    s_conn.close()


if __name__ == "__main__":
    migrate_sqlite_to_duckdb()
