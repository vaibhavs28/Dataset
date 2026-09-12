#!/usr/bin/env python3
"""
check_db_status.py
------------------
Inspects tables, row counts, and RELIANCE candle availability in market_data.duckdb.
"""
import sys
from pathlib import Path
import duckdb
import config

def check_status():
    p = config.DATA_DIR / "market_data.duckdb"
    print("\n" + "="*60)
    print("📊 MARKET DATA DATABASE DIAGNOSTIC")
    print("="*60)
    print(f"📁 Database Path: {p}")
    if not p.exists():
        print("❌ market_data.duckdb does not exist!")
        return

    print(f"📦 Database Size: {p.stat().st_size / (1024*1024):.2f} MB")
    conn = duckdb.connect(str(p), read_only=True)
    try:
        tables = [r[0] for r in conn.execute("PRAGMA show_tables;").fetchall()]
        print(f"\n📑 Tables Found ({len(tables)}):")
        for tbl in tables:
            cnt = conn.execute(f"SELECT count(*) FROM {tbl};").fetchone()[0]
            print(f"  • {tbl:20s}: {cnt:>12,} rows")

        print("\n🔍 Checking RELIANCE Availability:")
        for tbl in ["daily_candles", "intraday_candles", "stocks"]:
            if tbl in tables:
                col = "ticker" if tbl == "stocks" else "trading_symbol"
                cnt = conn.execute(f"SELECT count(*) FROM {tbl} WHERE {col} = 'RELIANCE';").fetchone()[0]
                if cnt > 0:
                    dt_col = "datetime" if tbl == "stocks" else ("timestamp" if tbl == "intraday_candles" else "date")
                    min_dt, max_dt = conn.execute(f"SELECT min({dt_col}), max({dt_col}) FROM {tbl} WHERE {col} = 'RELIANCE';").fetchone()
                    print(f"  ✅ {tbl:20s}: {cnt:>10,} bars ({min_dt} to {max_dt})")
                else:
                    print(f"  ⚠️ {tbl:20s}: 0 bars")

        by_sym = config.DATA_DIR / "by_symbol"
        if by_sym.exists():
            pqs = list(by_sym.glob("*.parquet"))
            print(f"\n📦 Per-Symbol Parquets: {len(pqs)} files in {by_sym}")
            rel_pq = by_sym / "RELIANCE.parquet"
            if rel_pq.exists():
                print(f"  ✅ RELIANCE.parquet: {rel_pq.stat().st_size / (1024*1024):.2f} MB")
        else:
            print(f"\n📦 Per-Symbol Parquets: 0 files ({by_sym} not created yet)")

    finally:
        conn.close()
    print("="*60 + "\n")

if __name__ == "__main__":
    check_status()
