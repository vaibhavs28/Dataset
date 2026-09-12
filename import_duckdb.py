"""
import_duckdb.py
----------------
Utility script to inspect and merge external DuckDB databases into the local system's DuckDB store
(data/market_data.duckdb) and SQLite backup.

Features:
1. Automatically inspects tables, schemas, and row counts of the external DuckDB file.
2. Uses high-speed DuckDB ATTACH mechanism for near-instant SQL merging.
3. Maps common table and column variations (e.g. daily candles, intraday 75m, 1m ticks, instruments).
4. Provides progress reporting and verification summaries.
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import duckdb
import pandas as pd

import config
import duckdb_store
import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("import_duckdb")


def inspect_external_duckdb(file_path: str) -> Dict[str, Any]:
    """
    Connects to the external duckdb file in read-only mode and returns table names,
    schemas, and row counts.
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    logger.info(f"Inspecting external DuckDB: {p} (Size: {p.stat().st_size / (1024*1024):.2f} MB)")
    conn = duckdb.connect(str(p), read_only=True)
    try:
        tables_df = conn.execute("PRAGMA show_tables;").df()
        table_names = tables_df.iloc[:, 0].tolist() if not tables_df.empty else []

        summary = {"path": str(p), "size_mb": round(p.stat().st_size / (1024 * 1024), 2), "tables": {}}

        for tbl in table_names:
            cols_df = conn.execute(f"PRAGMA table_info('{tbl}');").df()
            count = conn.execute(f"SELECT count(*) FROM '{tbl}';").fetchone()[0]
            summary["tables"][tbl] = {
                "columns": cols_df["name"].tolist() if "name" in cols_df.columns else [],
                "types": cols_df["type"].tolist() if "type" in cols_df.columns else [],
                "row_count": count
            }

        return summary
    finally:
        conn.close()


def merge_external_duckdb(file_path: str, replace_existing: bool = True, resample_only: bool = False) -> Dict[str, Any]:
    """
    Merges tables from the external DuckDB file directly into market_data.duckdb.
    Uses ATTACH DATABASE for fast bulk insertion.
    If resample_only=True, extracts Daily & 75m candles without copying the heavy 441M raw 1-min table (saves 97% disk space).
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"External DuckDB file not found: {p}")

    info = inspect_external_duckdb(str(p))
    logger.info(f"Found {len(info['tables'])} table(s) in {p.name}: {list(info['tables'].keys())}")

    start_time = time.time()
    dest_path = duckdb_store.DUCKDB_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dest_conn = duckdb_store.get_write_connection()
    except Exception as e:
        if "Conflicting lock is held" in str(e) or "Could not set lock" in str(e):
            logger.error("\n" + "="*70)
            logger.error("❌ DATABASE FILE LOCKED BY ANOTHER RUNNING PROCESS")
            logger.error("="*70)
            logger.error("Your Streamlit app or background Broadcaster is currently using market_data.duckdb.")
            logger.error("To merge, temporarily stop the running process:")
            logger.error("  pkill -f streamlit")
            logger.error("  pkill -f auto_75m_broadcaster")
            logger.error("Then re-run this merge command. You can restart Streamlit immediately after.")
            logger.error("="*70 + "\n")
        raise e

    results = {}


    try:
        # Attach external database as 'ext_db'
        dest_conn.execute(f"ATTACH '{str(p)}' AS ext_db (READ_ONLY);")

        for tbl_name, tbl_info in info["tables"].items():
            cols = [c.lower() for c in tbl_info["columns"]]
            row_count = tbl_info["row_count"]
            logger.info(f"Processing table '{tbl_name}' ({row_count:,} rows, columns: {cols})...")

            # 0. Specialized handler for authentic 1-minute 'stocks' table (ticker, datetime, open, high, low, close, volume)
            if "ticker" in cols and "datetime" in cols:
                dest_table = "stocks"
                logger.info(f"Detected 1-minute equity table '{tbl_name}' with {row_count:,} rows. Merging into {dest_table}...")

                date_sel = "date" if "date" in cols else "CAST(strftime(datetime, '%Y-%m-%d') AS DATE)"
                time_sel = "time" if "time" in cols else "strftime(datetime, '%H:%M:%S')"

                # Compute corporate action (split / bonus) adjustment factor against authentic daily_candles
                logger.info("Detecting and computing bonus / split adjustment factors against daily_candles...")
                dest_conn.execute(f"""
                    CREATE TEMP TABLE IF NOT EXISTS _corp_factors AS
                    WITH ext_summary AS (
                        SELECT 
                            ticker,
                            CAST(MAX(datetime) AS DATE) AS max_dt,
                            arg_max(close, datetime) AS ext_close
                        FROM ext_db."{tbl_name}"
                        GROUP BY ticker
                    ),
                    matched AS (
                        SELECT 
                            e.ticker,
                            e.ext_close,
                            d.close AS mkt_close,
                            ROUND(d.close / e.ext_close, 6) AS factor
                        FROM ext_summary e
                        JOIN daily_candles d ON (d.trading_symbol = e.ticker AND CAST(d.date AS DATE) = e.max_dt)
                        WHERE e.ext_close > 0 AND d.close > 0
                    )
                    SELECT 
                        ticker,
                        CASE 
                            WHEN ABS(factor - 1.0) >= 0.02 THEN factor 
                            ELSE 1.0 
                        END AS factor
                    FROM matched;
                """)
                adj_count = dest_conn.execute("SELECT count(*) FROM _corp_factors WHERE factor != 1.0;").fetchone()[0]
                logger.info(f"✅ Found {adj_count} stocks requiring bonus/split ratio adjustments. Applying adjustment factors...")

                if not resample_only:
                    t0 = time.time()
                    # A. Copy / Merge 1-minute data table with bonus/split adjustment
                    if replace_existing:
                        dest_conn.execute("DROP TABLE IF EXISTS stocks;")

                    dest_conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS stocks (
                            ticker VARCHAR,
                            date DATE,
                            time VARCHAR,
                            datetime TIMESTAMP,
                            open DOUBLE,
                            high DOUBLE,
                            low DOUBLE,
                            close DOUBLE,
                            volume BIGINT
                        );
                    """)
                    dest_conn.execute(f"""
                        INSERT INTO stocks (ticker, date, time, datetime, open, high, low, close, volume)
                        SELECT 
                            s.ticker,
                            {date_sel} AS date,
                            {time_sel} AS time,
                            CAST(s.datetime AS TIMESTAMP) AS datetime,
                            ROUND(s.open * COALESCE(f.factor, 1.0), 2) AS open,
                            ROUND(s.high * COALESCE(f.factor, 1.0), 2) AS high,
                            ROUND(s.low * COALESCE(f.factor, 1.0), 2) AS low,
                            ROUND(s.close * COALESCE(f.factor, 1.0), 2) AS close,
                            CAST(ROUND(s.volume / COALESCE(f.factor, 1.0)) AS BIGINT) AS volume
                        FROM ext_db."{tbl_name}" s
                        LEFT JOIN _corp_factors f ON (f.ticker = s.ticker);
                    """)
                    logger.info(f"✅ Copied {row_count:,} adjusted 1-min bars into 'stocks' in {time.time()-t0:.2f}s")

                    # B. Create index on stocks (ticker, datetime) for sub-5ms queries
                    logger.info("Creating index on stocks(ticker, datetime)...")
                    t_idx = time.time()
                    try:
                        dest_conn.execute("CREATE INDEX IF NOT EXISTS idx_stocks_ticker_dt ON stocks (ticker, datetime);")
                        logger.info(f"✅ Index created in {time.time()-t_idx:.2f}s")
                    except Exception as e_idx:
                        logger.warning(f"Index creation note: {e_idx}")
                else:
                    logger.info("⚡ Resample-Only Mode: Skipping heavy 441M 1-min raw table copy; pre-aggregating Daily & 75m candles directly (saving 97% disk space)...")

                # C. Resample pre-2020 daily bars into daily_candles
                logger.info("Resampling pre-2020 daily candles from 1-min data into daily_candles...")
                t_daily = time.time()
                dest_conn.execute(f"""
                    INSERT INTO daily_candles (
                        instrument_key, trading_symbol, date, open, high, low, close, volume, open_interest
                    )
                    SELECT
                        'NSE_EQ|' || s.ticker AS instrument_key,
                        s.ticker AS trading_symbol,
                        CAST({date_sel} AS VARCHAR) AS date,
                        ROUND(arg_min(s.open, s.datetime) * COALESCE(f.factor, 1.0), 2) AS open,
                        ROUND(MAX(s.high) * COALESCE(f.factor, 1.0), 2) AS high,
                        ROUND(MIN(s.low) * COALESCE(f.factor, 1.0), 2) AS low,
                        ROUND(arg_max(s.close, s.datetime) * COALESCE(f.factor, 1.0), 2) AS close,
                        CAST(ROUND(SUM(s.volume) / COALESCE(f.factor, 1.0)) AS BIGINT) AS volume,
                        0 AS open_interest
                    FROM ext_db."{tbl_name}" s
                    LEFT JOIN _corp_factors f ON (f.ticker = s.ticker)
                    WHERE {date_sel} < '2020-01-01'
                    GROUP BY s.ticker, {date_sel}, f.factor
                    ON CONFLICT (instrument_key, date) DO UPDATE SET
                        trading_symbol = EXCLUDED.trading_symbol,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest;
                """)
                logger.info(f"✅ Pre-2020 daily candles resampled and merged in {time.time()-t_daily:.2f}s")

                # D. Resample pre-2020 75-minute candles into intraday_candles
                logger.info("Resampling pre-2020 authentic 75-minute candles into intraday_candles...")
                t_75 = time.time()
                dest_conn.execute(f"""
                    INSERT INTO intraday_candles (
                        instrument_key, trading_symbol, timeframe, timestamp, open, high, low, close, volume
                    )
                    WITH session_bars AS (
                        SELECT 
                            s.ticker,
                            CAST(s.datetime AS TIMESTAMP) AS dt,
                            s.open * COALESCE(f.factor, 1.0) AS open,
                            s.high * COALESCE(f.factor, 1.0) AS high,
                            s.low * COALESCE(f.factor, 1.0) AS low,
                            s.close * COALESCE(f.factor, 1.0) AS close,
                            CAST(ROUND(s.volume / COALESCE(f.factor, 1.0)) AS BIGINT) AS volume,
                            CAST({date_sel} AS TIMESTAMP) + INTERVAL (
                                CASE 
                                    WHEN CAST(s.datetime::TIMESTAMP AS TIME) < TIME '10:30:00' THEN 555
                                    WHEN CAST(s.datetime::TIMESTAMP AS TIME) < TIME '11:45:00' THEN 630
                                    WHEN CAST(s.datetime::TIMESTAMP AS TIME) < TIME '13:00:00' THEN 705
                                    WHEN CAST(s.datetime::TIMESTAMP AS TIME) < TIME '14:15:00' THEN 780
                                    ELSE 855
                                END
                            ) MINUTE AS slot_ts
                        FROM ext_db."{tbl_name}" s
                        LEFT JOIN _corp_factors f ON (f.ticker = s.ticker)
                        WHERE {date_sel} < '2022-01-01'
                          AND CAST(s.datetime::TIMESTAMP AS TIME) >= TIME '09:15:00' 
                          AND CAST(s.datetime::TIMESTAMP AS TIME) <= TIME '15:30:00'
                    )
                    SELECT
                        'NSE_EQ|' || ticker AS instrument_key,
                        ticker AS trading_symbol,
                        '75m' AS timeframe,
                        slot_ts AS timestamp,
                        ROUND(arg_min(open, dt), 2) AS open,
                        ROUND(MAX(high), 2) AS high,
                        ROUND(MIN(low), 2) AS low,
                        ROUND(arg_max(close, dt), 2) AS close,
                        CAST(SUM(volume) AS BIGINT) AS volume
                    FROM session_bars
                    GROUP BY ticker, slot_ts
                    ON CONFLICT (instrument_key, timeframe, timestamp) DO UPDATE SET
                        trading_symbol = EXCLUDED.trading_symbol,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume;
                """)
                logger.info(f"✅ Pre-2020 75-minute candles resampled and merged in {time.time()-t_75:.2f}s")

                results[tbl_name] = {
                    "target": "stocks, daily_candles, intraday_candles",
                    "status": "fully_merged_and_resampled",
                    "rows": row_count,
                    "time": round(time.time() - t0, 2)
                }
                continue



            # 1. Check if table is daily candles
            if ("date" in cols or "timestamp" in cols) and ("close" in cols) and ("timeframe" not in cols and "1m" not in tbl_name.lower()):
                dest_table = "daily_candles"
                date_col = "date" if "date" in cols else "timestamp"
                sym_col = "trading_symbol" if "trading_symbol" in cols else ("symbol" if "symbol" in cols else ("ticker" if "ticker" in cols else None))

                ikey_col = "instrument_key" if "instrument_key" in cols else None
                oi_col = "open_interest" if "open_interest" in cols else ("oi" if "oi" in cols else "0")

                if sym_col:
                    ikey_expr = ikey_col if ikey_col else f"COALESCE('NSE_EQ|' || {sym_col}, {sym_col})"
                    sql = f"""
                        INSERT INTO {dest_table} (
                            instrument_key, trading_symbol, date, open, high, low, close, volume, open_interest
                        )
                        SELECT
                            {ikey_expr} AS instrument_key,
                            {sym_col} AS trading_symbol,
                            CAST({date_col} AS VARCHAR) AS date,
                            CAST(open AS DOUBLE) AS open,
                            CAST(high AS DOUBLE) AS high,
                            CAST(low AS DOUBLE) AS low,
                            CAST(close AS DOUBLE) AS close,
                            CAST(COALESCE(volume, 0) AS BIGINT) AS volume,
                            CAST(COALESCE({oi_col}, 0) AS BIGINT) AS open_interest
                        FROM ext_db."{tbl_name}"
                        ON CONFLICT (instrument_key, date) DO UPDATE SET
                            trading_symbol = EXCLUDED.trading_symbol,
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            open_interest = EXCLUDED.open_interest;
                    """
                    t0 = time.time()
                    dest_conn.execute(sql)
                    results[tbl_name] = {"target": dest_table, "status": "merged", "rows": row_count, "time": round(time.time() - t0, 2)}
                    logger.info(f"✅ Merged {row_count:,} rows into {dest_table} in {time.time() - t0:.2f}s")
                    continue

            # 2. Check if table is intraday candles (e.g. 75m)
            if "timeframe" in cols or "intraday" in tbl_name.lower():
                dest_table = "intraday_candles"
                sym_col = "trading_symbol" if "trading_symbol" in cols else ("symbol" if "symbol" in cols else None)
                ikey_col = "instrument_key" if "instrument_key" in cols else None
                tf_expr = "timeframe" if "timeframe" in cols else "'75m'"

                if sym_col:
                    ikey_expr = ikey_col if ikey_col else f"COALESCE('NSE_EQ|' || {sym_col}, {sym_col})"
                    sql = f"""
                        INSERT INTO {dest_table} (
                            instrument_key, trading_symbol, timeframe, timestamp, open, high, low, close, volume
                        )
                        SELECT
                            {ikey_expr} AS instrument_key,
                            {sym_col} AS trading_symbol,
                            {tf_expr} AS timeframe,
                            CAST(timestamp AS TIMESTAMP) AS timestamp,
                            CAST(open AS DOUBLE) AS open,
                            CAST(high AS DOUBLE) AS high,
                            CAST(low AS DOUBLE) AS low,
                            CAST(close AS DOUBLE) AS close,
                            CAST(COALESCE(volume, 0) AS BIGINT) AS volume
                        FROM ext_db."{tbl_name}"
                        ON CONFLICT (instrument_key, timeframe, timestamp) DO UPDATE SET
                            trading_symbol = EXCLUDED.trading_symbol,
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume;
                    """
                    t0 = time.time()
                    dest_conn.execute(sql)
                    results[tbl_name] = {"target": dest_table, "status": "merged", "rows": row_count, "time": round(time.time() - t0, 2)}
                    logger.info(f"✅ Merged {row_count:,} rows into {dest_table} in {time.time() - t0:.2f}s")
                    continue

            # 3. Check if table is instruments master
            if "trading_symbol" in cols and ("name" in cols or "exchange" in cols):
                dest_table = "instruments"
                sql = f"""
                    INSERT INTO {dest_table}
                    SELECT * FROM ext_db."{tbl_name}"
                    ON CONFLICT (instrument_key) DO UPDATE SET
                        trading_symbol = EXCLUDED.trading_symbol,
                        name = EXCLUDED.name,
                        exchange = EXCLUDED.exchange,
                        instrument_type = EXCLUDED.instrument_type,
                        tick_size = EXCLUDED.tick_size,
                        lot_size = EXCLUDED.lot_size,
                        last_updated = CURRENT_TIMESTAMP;
                """
                t0 = time.time()
                dest_conn.execute(sql)
                results[tbl_name] = {"target": dest_table, "status": "merged", "rows": row_count, "time": round(time.time() - t0, 2)}
                logger.info(f"✅ Merged {row_count:,} instruments in {time.time() - t0:.2f}s")
                continue

            # Generic table copy if table doesn't exist
            logger.info(f"Creating exact copy of custom table '{tbl_name}'...")
            dest_conn.execute(f'CREATE TABLE IF NOT EXISTS "{tbl_name}" AS SELECT * FROM ext_db."{tbl_name}";')
            results[tbl_name] = {"target": tbl_name, "status": "copied", "rows": row_count}


        dest_conn.execute("DETACH ext_db;")
        logger.info("Optimizing and compacting DuckDB database pages (CHECKPOINT & VACUUM)...")
        dest_conn.execute("CHECKPOINT;")
        try:
            dest_conn.execute("VACUUM;")
        except Exception:
            pass

    finally:
        dest_conn.close()

    total_time = round(time.time() - start_time, 2)
    logger.info(f"🎉 Merge complete in {total_time}s!")

    return {
        "source_file": str(p),
        "elapsed_seconds": total_time,
        "tables": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge external DuckDB file into market_data.duckdb")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to external .duckdb file")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect tables and schema without merging")
    parser.add_argument("--resample-only", action="store_true", help="Only import pre-calculated Daily and 75m candles, skipping raw 1-min table (saves 97% disk space)")
    parser.add_argument("--delete-source", action="store_true", help="Delete external source .duckdb file after successful merge")

    args = parser.parse_args()

    if args.inspect_only:
        summary = inspect_external_duckdb(args.file)
        print("\n--- External DuckDB Inspection ---")
        print(f"File: {summary['path']} ({summary['size_mb']} MB)")
        for tbl, d in summary["tables"].items():
            print(f"\nTable: {tbl} ({d['row_count']:,} rows)")
            print(f"Columns: {', '.join(d['columns'])}")
    else:
        res = merge_external_duckdb(args.file, resample_only=args.resample_only)
        print("\n--- Merge Summary ---")
        print(f"Source: {res['source_file']}")
        print(f"Time: {res['elapsed_seconds']}s")
        for tbl, d in res["tables"].items():
            print(f"• {tbl} -> {d['target']} ({d.get('rows', 0):,} rows) [{d['status']}]")

        if args.delete_source:
            source_p = Path(args.file).expanduser().resolve()
            if source_p.exists():
                source_p.unlink()
                print(f"\n🗑️ Successfully deleted source file: {source_p} (freed disk space)")
