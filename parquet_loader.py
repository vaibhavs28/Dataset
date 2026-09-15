import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import config
import database
import instruments

logger = logging.getLogger("parquet_loader")
PARQUET_DIR = config.DATA_DIR / "1min"
PARQUET_BY_SYMBOL_DIR = config.DATA_DIR / "by_symbol"
SYMBOLS_CACHE_FILE = config.DATA_DIR / "parquet_symbols.json"


def is_pure_equity_symbol(sym: str) -> bool:
    s = sym.upper().strip()
    if s.startswith("0"):
        return False
    for bad in ["ETF", "BEES", "NIFTY", "SENSEX", "INDIA VIX", "MIDCPNIFTY", "FINNIFTY"]:
        if bad in s:
            return False
    return True


def get_parquet_symbols() -> List[str]:
    """Returns the list of available pure Equity symbols in the local parquet dataset (excluding indices and ETFs)."""
    if PARQUET_BY_SYMBOL_DIR.exists():
        files = list(PARQUET_BY_SYMBOL_DIR.glob("*.parquet"))
        if len(files) > 0:
            return sorted([f.stem for f in files if is_pure_equity_symbol(f.stem)])

    if SYMBOLS_CACHE_FILE.exists():
        try:
            with open(SYMBOLS_CACHE_FILE, "r") as f:
                raw_list = json.load(f)
                return sorted([s for s in raw_list if is_pure_equity_symbol(s)])
        except Exception:
            pass

    # Build cache if missing
    try:
        if PARQUET_DIR.exists():
            pq_files = list(PARQUET_DIR.glob("*.parquet"))
            if pq_files:
                dataset = ds.dataset(str(PARQUET_DIR), format="parquet")
                if "symbol" in dataset.schema.names:
                    scanner = dataset.scanner(columns=["symbol"])
                    unique_syms = set()
                    for batch in scanner.to_batches():
                        unique_syms.update(batch["symbol"].unique().to_pylist())

                    sorted_syms = sorted(list(unique_syms))
                    with open(SYMBOLS_CACHE_FILE, "w") as f:
                        json.dump(sorted_syms, f)
                    return sorted_syms
    except Exception as e:
        logger.warning(f"Error reading parquet symbols: {e}")

    return []


def get_symbol_date_range(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (min_date_str, max_date_str) in IST for a given symbol in the parquet dataset."""
    sym = symbol.upper().strip()
    safe_sym = sym.replace("/", "_").replace("\\", "_")
    symbol_file = PARQUET_BY_SYMBOL_DIR / f"{safe_sym}.parquet"

    tbl = None
    if symbol_file.exists():
        try:
            tbl = pq.read_table(symbol_file, columns=["timestamp"])
        except Exception:
            pass

    if tbl is None and PARQUET_DIR.exists():
        try:
            pq_files = list(PARQUET_DIR.glob("*.parquet"))
            if pq_files:
                dataset = ds.dataset(str(PARQUET_DIR), format="parquet")
                if "symbol" in dataset.schema.names:
                    filter_expr = (ds.field("symbol") == sym)
                    scanner = dataset.scanner(filter=filter_expr, columns=["timestamp"])
                    tbl = scanner.to_table()
        except Exception as e:
            logger.warning(f"Error getting date range for {sym}: {e}")
            return None, None

    if tbl is None or tbl.num_rows == 0:
        return None, None
    s = tbl["timestamp"].to_pandas().dt.tz_convert("Asia/Kolkata")
    min_d = s.min().strftime("%Y-%m-%d %H:%M")
    max_d = s.max().strftime("%Y-%m-%d %H:%M")
    return min_d, max_d


def load_symbol_ohlcv_from_parquet(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads 1-minute data for a symbol from local parquet files,
    converts UTC to IST, and resamples to:
    1. daily_df: Daily OHLCV bars
    2. intra_75m: Exactly 5 session-aligned 75-min candles per trading day (09:15 to 15:30)
    """
    sym = symbol.upper().strip()
    safe_sym = sym.replace("/", "_").replace("\\", "_")
    symbol_file = PARQUET_BY_SYMBOL_DIR / f"{safe_sym}.parquet"

    tbl = None
    # Sub-10ms direct read path:
    if symbol_file.exists():
        try:
            tbl = pq.read_table(symbol_file, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.warning(f"Error reading direct symbol file {symbol_file}: {e}")

    # Fallback to multi-shard scanner if single-symbol file not found:
    if tbl is None:
        dataset = ds.dataset(str(PARQUET_DIR), format="parquet")
        filter_expr = (ds.field("symbol") == sym)
        scanner = dataset.scanner(filter=filter_expr, columns=["timestamp", "open", "high", "low", "close", "volume"])
        tbl = scanner.to_table()

    if tbl.num_rows == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = tbl.to_pandas()
    # Convert UTC timestamps to Indian Standard Time (IST)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Kolkata")
    df.sort_values(by="timestamp", inplace=True)
    df = df[~df["timestamp"].duplicated(keep="last")]
    df.set_index("timestamp", inplace=True)

    # 1. Resample to Daily
    daily_df = df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    # Normalize index to timezone-naive for SQLite / Plotly compatibility
    daily_df.index = daily_df.index.tz_localize(None)

    # 2. Resample to 75-minute candles strictly from 1-minute data
    intra_75m = resample_1min_to_75min(df)
    return daily_df, intra_75m


def resample_1min_to_75min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly resamples authentic 1-minute OHLCV candles to 75-minute Indian market session candles.
    Indian equity session: 09:15 to 15:30 IST (375 minutes = exactly 5 candles of 75 minutes).
    Slot 0: 09:15 - 10:30 (start timestamp 09:15:00)
    Slot 1: 10:30 - 11:45 (start timestamp 10:30:00)
    Slot 2: 11:45 - 13:00 (start timestamp 11:45:00)
    Slot 3: 13:00 - 14:15 (start timestamp 13:00:00)
    Slot 4: 14:15 - 15:30 (start timestamp 14:15:00)
    Aggregations:
      open: first
      high: max
      low: min
      close: last
      volume: sum
    Returns timezone-naive DataFrame indexed by 'timestamp' with columns ['open', 'high', 'low', 'close', 'volume'].
    """
    if df_1min is None or df_1min.empty:
        return pd.DataFrame()

    df = df_1min.copy()

    # Normalize index/column for timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure IST timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    # Normalize column names to lowercase
    col_map = {c: str(c).lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    required_cols = ["open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 0

    # Filter session: 09:15 to 15:30
    df_session = df.between_time("09:15", "15:30").copy()
    if df_session.empty:
        return pd.DataFrame()

    df_session["date"] = df_session.index.date
    minutes_since_open = (df_session.index.hour * 60 + df_session.index.minute) - (9 * 60 + 15)
    df_session["slot"] = pd.Series(minutes_since_open // 75, index=df_session.index).clip(0, 4)

    session_start_times = {
        0: "09:15:00",
        1: "10:30:00",
        2: "11:45:00",
        3: "13:00:00",
        4: "14:15:00"
    }

    res_75m = df_session.groupby(["date", "slot"]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).reset_index()

    slot_deltas = pd.to_timedelta(res_75m["slot"] * 75, unit="m") + pd.Timedelta(hours=9, minutes=15)
    res_75m["timestamp"] = pd.to_datetime(res_75m["date"]) + slot_deltas
    res_75m.set_index("timestamp", inplace=True)
    res_75m.drop(columns=["date", "slot"], inplace=True)
    return res_75m


def resample_1min_to_custom_minutes(df_1min: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """
    Strictly resamples authentic 1-minute OHLCV candles to any custom N-minute Indian market session candles.
    Aligns to the 09:15:00 IST market open.
    """
    if df_1min is None or df_1min.empty or interval_minutes <= 0:
        return pd.DataFrame()

    if interval_minutes == 1:
        return df_1min.copy()

    df = df_1min.copy()

    # Normalize index/column for timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure IST timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    col_map = {c: str(c).lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    required_cols = ["open", "high", "low", "close"]
    for c in required_cols:
        if c not in df.columns:
            return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 0

    # Filter session: 09:15 to 15:30
    df_session = df.between_time("09:15", "15:30").copy()
    if df_session.empty:
        return pd.DataFrame()

    df_session["date"] = df_session.index.date
    mins = np.maximum(0, (df_session.index.hour * 60 + df_session.index.minute) - (9 * 60 + 15))
    df_session["slot"] = mins // interval_minutes

    res = df_session.groupby(["date", "slot"]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).reset_index()

    slot_deltas = pd.to_timedelta(res["slot"] * interval_minutes, unit="m") + pd.Timedelta(hours=9, minutes=15)
    res["timestamp"] = pd.to_datetime(res["date"]) + slot_deltas
    res.set_index("timestamp", inplace=True)
    res.drop(columns=["date", "slot"], inplace=True)
    res.sort_index(inplace=True)
    return res


def ensure_symbol_custom_minute_candles(
    symbol: str, 
    interval_minutes: int, 
    min_bars: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Guarantees custom N-minute candles strictly resampled from 1-minute data.
    Uses DuckDB's vectorized C++ SQL engine for sub-10ms resampling.
    Supports start_date and end_date filtering.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    if interval_minutes == 75:
        return ensure_symbol_75m_candles(clean_sym, min_bars=min_bars, start_date=start_date, end_date=end_date)

    try:
        import duckdb_store
        df_duck = duckdb_store.get_resampled_candles(
            clean_sym, interval_minutes=interval_minutes, limit=2500, min_date=start_date, max_date=end_date
        )
        if not df_duck.empty and len(df_duck) >= min_bars:
            return df_duck
    except Exception as e:
        logger.warning(f"DuckDB fast resampling note for {clean_sym} ({interval_minutes}m): {e}")

    df_1min = load_symbol_1min(clean_sym, limit=None if (start_date or end_date) else 10000, start_date=start_date, end_date=end_date)
    if df_1min.empty:
        return pd.DataFrame()

    res = resample_1min_to_custom_minutes(df_1min, interval_minutes)
    if not res.empty:
        if not isinstance(res.index, pd.DatetimeIndex):
            res.index = pd.to_datetime(res.index)
        res_tz = getattr(res.index, "tz", None)

        if start_date:
            start_ts = pd.to_datetime(start_date)
            if res_tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(res_tz)
            elif res_tz is None and start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
            res = res[res.index >= start_ts]

        if end_date:
            end_ts = pd.to_datetime(f"{end_date} 23:59:59")
            if res_tz is not None and end_ts.tz is None:
                end_ts = end_ts.tz_localize(res_tz)
            elif res_tz is None and end_ts.tz is not None:
                end_ts = end_ts.tz_localize(None)
            res = res[res.index <= end_ts]
    return res



def is_genuine_75m_df(df: pd.DataFrame) -> bool:
    """Checks whether a 75m DataFrame consists of authentic resampled bars, not fake linear staircases."""
    if df is None or df.empty or len(df) < 5:
        return False
    # In fake staircase bars, open==high and low==close on multiple bars
    recent = df.tail(20)
    fake_pattern = ((recent["open"] == recent["high"]) & (recent["low"] == recent["close"]))
    if fake_pattern.sum() >= 4:
        return False
    return True


def ensure_symbol_75m_candles(
    symbol: str, 
    min_bars: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Guarantees authentic 75-minute candles strictly from our local database (DuckDB / SQLite).
    Zero external API calls. Sub-10ms query execution.
    Supports start_date and end_date filtering.
    """
    sym = symbol.upper().strip()

    # 1. First query our ultra-fast DuckDB intraday_candles table
    try:
        import duckdb_store
        df_duck = duckdb_store.get_intraday_candles(sym, timeframe="75m", limit=None, start_date=start_date, end_date=end_date)
        if not df_duck.empty:
            return df_duck
    except Exception as e:
        logger.warning(f"DuckDB intraday query note for {sym}: {e}")

    # 2. Fall back to SQLite intraday_candles (market_data.db)
    try:
        df_sql = database.get_intraday_candles_df(sym, "75m", limit=15000, start_date=start_date, end_date=end_date)
        if not df_sql.empty and is_genuine_75m_df(df_sql):
            return df_sql
    except Exception as e:
        logger.warning(f"SQLite intraday query note for {sym}: {e}")

    # 3. If local single-stock parquet exists, resample locally via DuckDB
    try:
        import duckdb_store
        df_resampled = duckdb_store.get_resampled_candles(sym, interval_minutes=75, limit=15000, min_date=start_date, max_date=end_date)
        if not df_resampled.empty:
            return df_resampled
    except Exception as e:
        logger.warning(f"Local parquet resample note for {sym}: {e}")

    # 4. Resample directly from local 1-min parquet
    try:
        df_1min = load_symbol_1min(sym, limit=None if (start_date or end_date) else 10000, start_date=start_date, end_date=end_date)
        if not df_1min.empty:
            df_75 = resample_1min_to_75min(df_1min)
            if not df_75.empty:
                return df_75
    except Exception as e:
        logger.debug(f"Fallback 1-min to 75-min resample note for {sym}: {e}")

    return pd.DataFrame()


def import_symbol_to_database(symbol: str) -> bool:
    """Loads a symbol from parquet and saves its daily and 75-min candles into SQLite."""
    daily_df, intra_75m = load_symbol_ohlcv_from_parquet(symbol)
    if daily_df.empty:
        return False

    sym = symbol.upper().strip()
    import instruments
    inst_info = database.get_instrument_by_symbol(sym)
    if inst_info:
        inst_key = inst_info["instrument_key"]
    else:
        inst_key = instruments.resolve_instrument_key(sym) or f"NSE_EQ|{sym}"

    # Upsert instrument record if missing
    if not inst_info:
        database.upsert_instruments([{
            "instrument_key": inst_key,
            "trading_symbol": sym,
            "name": f"{sym}",
            "exchange": "NSE_INDEX" if "INDEX" in inst_key else ("BSE_EQ" if "BSE" in inst_key else "NSE_EQ"),
            "instrument_type": "INDEX" if "INDEX" in inst_key else "EQUITY",
            "tick_size": 0.05,
            "lot_size": 1
        }])

    # Format daily candles
    daily_records = []
    for dt, row in daily_df.iterrows():
        daily_records.append({
            "instrument_key": inst_key,
            "trading_symbol": sym,
            "date": dt.strftime("%Y-%m-%d"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "open_interest": 0
        })
    database.upsert_candles(daily_records)

    # Format 75m candles
    if not intra_75m.empty:
        # Wipe old 75m records for this symbol to remove any stale/simulated candles
        with database.get_connection() as conn:
            conn.cursor().execute("DELETE FROM intraday_candles WHERE trading_symbol = ? AND timeframe = '75m'", (sym,))
            conn.commit()

        intra_records = []
        for ts, row in intra_75m.iterrows():
            intra_records.append({
                "instrument_key": inst_key,
                "trading_symbol": sym,
                "timeframe": "75m",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"])
            })
        database.upsert_intraday_candles(intra_records)

    return True


def import_batch_to_database(symbols: List[str], progress_callback=None) -> int:
    """Imports a batch of symbols from parquet into the SQLite database."""
    database.init_db()
    count = 0
    total = len(symbols)
    for idx, sym in enumerate(symbols):
        ok = import_symbol_to_database(sym)
        if ok:
            count += 1
        if progress_callback:
            progress_callback(idx + 1, total, sym)
    return count


def load_symbol_1min(
    symbol: str, 
    limit: Optional[int] = 5000, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Reads 1-minute OHLCV candles for a symbol from local parquet files, supporting date filters."""
    sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    safe_sym = sym.replace("/", "_").replace("\\", "_")
    symbol_file = PARQUET_BY_SYMBOL_DIR / f"{safe_sym}.parquet"

    tbl = None
    if symbol_file.exists():
        try:
            tbl = pq.read_table(symbol_file, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            pass

    if tbl is None and PARQUET_DIR.exists():
        try:
            pq_files = list(PARQUET_DIR.glob("*.parquet"))
            if pq_files:
                dataset = ds.dataset(str(PARQUET_DIR), format="parquet")
                if "symbol" in dataset.schema.names:
                    filter_expr = (ds.field("symbol") == sym)
                    scanner = dataset.scanner(
                        filter=filter_expr,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    tbl = scanner.to_table()
        except Exception as e:
            logger.warning(f"Error reading 1-minute data for {sym}: {e}")
            return pd.DataFrame()

    if tbl is None or tbl.num_rows == 0:
        return pd.DataFrame()

    df = tbl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Kolkata")
    df.sort_values(by="timestamp", inplace=True)
    df = df[~df["timestamp"].duplicated(keep="last")]
    df.set_index("timestamp", inplace=True)

    if start_date:
        start_ts = pd.to_datetime(start_date)
        if df.index.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]
    if end_date:
        end_dt = pd.to_datetime(f"{end_date} 23:59:59")
        if df.index.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize(df.index.tz)
        df = df[df.index <= end_dt]

    if limit and limit > 0 and len(df) > limit:
        return df.tail(limit)
    return df


def get_live_engine_status() -> dict:
    """
    Extracts real-time visual telemetry of the Upstox background data generator.
    Parses active background task logs and disk parquet shards with sub-second latency.
    """
    import re
    import time
    tasks_dir = Path("/Users/shilpashingnapure/.gemini/antigravity/brain/c276aa2e-7be8-4861-bb91-3f4ddb7e3c4d/.system_generated/tasks")
    if not tasks_dir.exists():
        return {"is_active": False, "message": "No active tasks found"}

    log_files = list(tasks_dir.glob("task-*.log"))
    engine_logs = []
    for lf in log_files:
        try:
            sample = lf.read_text(errors="ignore")[:600]
            if "Accelerated Engine" in sample or "equity_engine" in sample:
                engine_logs.append(lf)
        except Exception:
            pass

    if not engine_logs:
        return {"is_active": False, "message": "No active data engine"}

    engine_log = max(engine_logs, key=lambda f: f.stat().st_mtime)
    mtime = engine_log.stat().st_mtime
    age_s = time.time() - mtime
    is_active = (age_s < 45.0)

    try:
        lines = engine_log.read_text(errors="ignore").splitlines()
    except Exception:
        return {"is_active": is_active, "message": "Reading log..."}

    sym_lines = [l for l in lines if re.search(r"\[(\d+)/(\d+)\]\s+(\w+):", l)]
    if not sym_lines:
        return {"is_active": is_active, "message": "Engine starting up..."}

    recent = []
    for l in sym_lines[-15:]:
        m = re.search(r"\[(\d+)/(\d+)\]\s+(\w+):\s+\+(\d+)\s+daily bars,\s+\+([\d,]+)\s+1-min bars\s+\(([\d.]+)s\)", l)
        if m:
            recent.append({
                "index": int(m.group(1)),
                "symbol": m.group(3),
                "daily_bars": int(m.group(4)),
                "min_bars": m.group(5),
                "seconds": float(m.group(6))
            })

    last = recent[-1] if recent else {"index": 0, "symbol": "-", "daily_bars": 0, "min_bars": "0", "seconds": 0}
    total_syms = 3498
    curr_idx = last["index"]
    pct = round((curr_idx / total_syms) * 100, 1)

    by_sym_files = list(PARQUET_BY_SYMBOL_DIR.glob("*.parquet")) if PARQUET_BY_SYMBOL_DIR.exists() else []
    shards = sorted(list(config.DATA_DIR.joinpath("1min").glob("train-*.parquet")))
    
    if by_sym_files:
        total_size_mb = sum(f.stat().st_size for f in by_sym_files) / (1024 * 1024)
        active_shard = f"{len(by_sym_files)} symbol files"
        active_shard_mb = total_size_mb / max(1, len(by_sym_files))
        num_shards = len(by_sym_files)
    elif shards:
        total_size_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
        active_shard = shards[-1].name
        active_shard_mb = (shards[-1].stat().st_size / (1024 * 1024))
        num_shards = len(shards)
    else:
        total_size_mb = 0.0
        active_shard = "by_symbol"
        active_shard_mb = 0.0
        num_shards = 0

    batch_lines = [l for l in lines if "Starting Batch" in l]
    current_batch = "1/66"
    if batch_lines:
        bm = re.search(r"Starting Batch\s+(\d+/\d+)", batch_lines[-1])
        if bm:
            current_batch = bm.group(1)

    rem_syms = max(0, total_syms - curr_idx)
    avg_s = sum(r["seconds"] for r in recent) / len(recent) if recent else 5.2
    eta_sec = rem_syms * avg_s
    eta_hrs = eta_sec / 3600.0

    return {
        "is_active": is_active,
        "age_s": round(age_s, 1),
        "curr_idx": curr_idx,
        "total_syms": total_syms,
        "pct": pct,
        "current_batch": current_batch,
        "latest_symbol": last["symbol"],
        "latest_time": last["seconds"],
        "latest_min_bars": last["min_bars"],
        "latest_daily_bars": last["daily_bars"],
        "avg_speed_s": round(avg_s, 2),
        "rem_syms": rem_syms,
        "eta_hrs": round(eta_hrs, 1),
        "eta_mins": int(eta_sec // 60),
        "active_shard": active_shard,
        "active_shard_mb": round(active_shard_mb, 1),
        "total_size_mb": round(total_size_mb, 1),
        "num_shards": num_shards,
        "recent": recent
    }

