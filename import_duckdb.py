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


def merge_external_duckdb(file_path: str, replace_existing: bool = True) -> Dict[str, Any]:
    """
    Merges tables from the external DuckDB file directly into market_data.duckdb.
    Uses ATTACH DATABASE for fast bulk insertion.
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"External DuckDB file not found: {p}")

    info = inspect_external_duckdb(str(p))
    logger.info(f"Found {len(info['tables'])} table(s) in {p.name}: {list(info['tables'].keys())}")

    start_time = time.time()
    dest_path = duckdb_store.DUCKDB_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    dest_conn = duckdb_store.get_write_connection()
    results = {}

    try:
        # Attach external database as 'ext_db'
        dest_conn.execute(f"ATTACH '{str(p)}' AS ext_db (READ_ONLY);")

        for tbl_name, tbl_info in info["tables"].items():
            cols = [c.lower() for c in tbl_info["columns"]]
            row_count = tbl_info["row_count"]
            logger.info(f"Processing table '{tbl_name}' ({row_count:,} rows, columns: {cols})...")

            # 1. Check if table is daily candles
            if ("date" in cols or "timestamp" in cols) and ("close" in cols) and ("timeframe" not in cols and "1m" not in tbl_name.lower()):
                dest_table = "daily_candles"
                date_col = "date" if "date" in cols else "timestamp"
                sym_col = "trading_symbol" if "trading_symbol" in cols else ("symbol" if "symbol" in cols else None)
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

    args = parser.parse_args()

    if args.inspect_only:
        summary = inspect_external_duckdb(args.file)
        print("\n--- External DuckDB Inspection ---")
        print(f"File: {summary['path']} ({summary['size_mb']} MB)")
        for tbl, d in summary["tables"].items():
            print(f"\nTable: {tbl} ({d['row_count']:,} rows)")
            print(f"Columns: {', '.join(d['columns'])}")
    else:
        res = merge_external_duckdb(args.file)
        print("\n--- Merge Summary ---")
        print(f"Source: {res['source_file']}")
        print(f"Time: {res['elapsed_seconds']}s")
        for tbl, d in res["tables"].items():
            print(f"• {tbl} -> {d['target']} ({d.get('rows', 0):,} rows) [{d['status']}]")
