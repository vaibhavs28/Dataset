import os
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import duckdb

import config

logger = logging.getLogger("duckdb_store")
DUCKDB_PATH = config.DATA_DIR / "market_data.duckdb"

_lock = threading.Lock()
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Returns a shared, thread-safe, read-only DuckDB connection.
    Always opens in read_only=True so multiple processes (Streamlit, UI, Scanners)
    can query concurrently without OS file lock contention.
    """
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
                if not DUCKDB_PATH.exists() or DUCKDB_PATH.stat().st_size < 1024 * 1024:
                    logger.warning(f"DuckDB database not found at {DUCKDB_PATH}. Attempting automatic download from GitHub...")
                    try:
                        import download_dataset
                        download_dataset.download()
                    except Exception as dl_err:
                        logger.error(f"Could not auto-download database: {dl_err}. You can manually run: python3 download_dataset.py")
                try:
                    _conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=True)
                    _conn.execute("PRAGMA threads=4;")
                    _conn.execute("PRAGMA memory_limit='4GB';")
                except Exception as e:
                    logger.warning(f"Error opening DuckDB in read-only mode: {e}")
                    _conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=True)
    return _conn


def get_write_connection() -> duckdb.DuckDBPyConnection:
    """
    Returns a short-lived write connection. Caller must close it immediately after writing.
    """
    conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=False)
    conn.execute("PRAGMA threads=4;")
    try:
        _init_schema(conn)
    except Exception:
        pass
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection):
    """Initializes the database schema if tables do not exist."""
    # 1. Instruments master table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            instrument_key VARCHAR PRIMARY KEY,
            trading_symbol VARCHAR NOT NULL,
            name VARCHAR,
            exchange VARCHAR NOT NULL,
            instrument_type VARCHAR,
            tick_size DOUBLE,
            lot_size INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Daily OHLCV candles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_candles (
            instrument_key VARCHAR NOT NULL,
            trading_symbol VARCHAR NOT NULL,
            date VARCHAR NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            open_interest BIGINT DEFAULT 0,
            PRIMARY KEY (instrument_key, date)
        );
    """)

    # 3. 1-minute historical candles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_1m (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            oi BIGINT DEFAULT 0,
            PRIMARY KEY (symbol, timestamp)
        );
    """)

    # 4. Intraday (e.g. 75m) candles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intraday_candles (
            instrument_key VARCHAR NOT NULL,
            trading_symbol VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            PRIMARY KEY (instrument_key, timeframe, timestamp)
        );
    """)


def get_candles_df(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame of daily candles for a given symbol, indexed by DatetimeIndex (date).
    Columns: open, high, low, close, volume, open_interest.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    conn = get_connection()

    query = """
        SELECT date, open, high, low, close, volume, open_interest
        FROM daily_candles
        WHERE (trading_symbol = ? OR trading_symbol = ? || '-EQ')
    """
    params = [clean_sym, clean_sym]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC;"

    with _lock:
        df = conn.execute(query, params).df()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def get_intraday_candles(symbol: str, timeframe: str = "75m", limit: int = 2500) -> pd.DataFrame:
    """
    Returns authentic pre-calculated intraday candles directly from DuckDB's intraday_candles table.
    Sub-10ms query execution.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    conn = get_connection()
    with _lock:
        try:
            df = conn.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM intraday_candles
                WHERE (trading_symbol = ? OR trading_symbol = ? || '-EQ')
                  AND timeframe = ?
                ORDER BY timestamp ASC;
            """, [clean_sym, clean_sym, timeframe]).df()
        except Exception as e:
            logger.warning(f"Error querying intraday_candles for {clean_sym}: {e}")
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df.set_index("timestamp", inplace=True)
    return df.tail(limit)


def get_latest_candle_date(symbol: str) -> Optional[str]:
    """Returns the most recent candle date (YYYY-MM-DD) for a symbol."""
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    conn = get_connection()
    with _lock:
        res = conn.execute("""
            SELECT max(date)
            FROM daily_candles
            WHERE trading_symbol = ? OR trading_symbol = ? || '-EQ'
        """, [clean_sym, clean_sym]).fetchone()
    return res[0] if res and res[0] else None


def get_all_symbols(include_indices: bool = True, exchange: Optional[str] = None) -> List[str]:
    """Returns all unique Equity and Index trading symbols strictly from NSE or specified exchange."""
    conn = get_connection()

    # Check if instruments table is populated; if fresh/empty, automatically sync from Upstox
    with _lock:
        try:
            inst_cnt = conn.execute("SELECT count(*) FROM instruments;").fetchone()[0]
        except Exception:
            inst_cnt = 0

    if inst_cnt < 500:
        try:
            import instruments
            instruments.sync_all_instruments()
        except Exception as e:
            logger.warning(f"Auto-syncing instruments in get_all_symbols: {e}")

    with _lock:
        try:
            if exchange:
                query = "SELECT DISTINCT trading_symbol FROM instruments WHERE exchange = ? AND trading_symbol NOT LIKE '0%';"
                rows = conn.execute(query, [exchange]).fetchall()
            elif include_indices:
                query = """
                    SELECT DISTINCT trading_symbol 
                    FROM instruments 
                    WHERE exchange IN ('NSE_EQ', 'NSE_INDEX')
                      AND trading_symbol NOT LIKE '0%'
                    ORDER BY trading_symbol ASC;
                """
                rows = conn.execute(query).fetchall()
            else:
                query = """
                    SELECT DISTINCT trading_symbol 
                    FROM instruments 
                    WHERE exchange = 'NSE_EQ'
                      AND instrument_type IN ('EQUITY', 'EQ', 'BE', 'SM', 'BZ')
                      AND trading_symbol NOT LIKE '0%'
                    ORDER BY trading_symbol ASC;
                """
                rows = conn.execute(query).fetchall()

            if rows:
                return sorted(list(set(r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0])))
        except Exception as e:
            logger.warning(f"Error querying instruments for symbols: {e}")

        # Fallback to daily_candles if instruments table is empty
        try:
            rows = conn.execute("SELECT DISTINCT trading_symbol FROM daily_candles WHERE trading_symbol NOT LIKE '0%';").fetchall()
            if rows:
                return sorted(list(set(r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0])))
        except Exception:
            pass

    return []


def get_alignment_scanner_symbols() -> List[str]:
    """
    Returns only pure NSE Equity symbols for the Multi-Timeframe Alignment Scanner.
    Strictly excludes:
    - All Indices (NSE_INDEX, BSE_INDEX, instrument_type == 'INDEX', NIFTY 50, BANKNIFTY, SENSEX, etc.)
    - All Index ETFs & Trackers (*ETF*, *BEES*, *NIFTY*, *SENSEX*, etc.)
    - All BSE Equity stocks (exchange == 'BSE_EQ')
    - Any debt/bond instruments starting with '0'
    """
    conn = get_connection()
    with _lock:
        try:
            inst_cnt = conn.execute("SELECT count(*) FROM instruments;").fetchone()[0]
        except Exception:
            inst_cnt = 0

    if inst_cnt < 500:
        try:
            import instruments
            instruments.sync_all_instruments()
        except Exception as e:
            logger.warning(f"Auto-syncing instruments in get_alignment_scanner_symbols: {e}")

    with _lock:
        try:
            rows = conn.execute("""
                SELECT DISTINCT trading_symbol 
                FROM instruments 
                WHERE exchange = 'NSE_EQ' 
                  AND instrument_type IN ('EQ', 'BE', 'SM', 'BZ', 'EQUITY')
                  AND trading_symbol NOT LIKE '0%'
                  AND trading_symbol NOT LIKE '%NIFTY%'
                  AND trading_symbol NOT LIKE '%SENSEX%'
                  AND trading_symbol NOT LIKE '%BEES%'
                  AND trading_symbol NOT LIKE '%ETF%'
                  AND trading_symbol NOT LIKE 'INDIA VIX%'
                ORDER BY trading_symbol ASC;
            """).fetchall()
            if rows:
                return sorted(list(set(r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0])))
        except Exception as e:
            logger.warning(f"Error querying alignment scanner symbols from DuckDB: {e}")

        # Fallback to daily_candles
        try:
            rows = conn.execute("""
                SELECT DISTINCT trading_symbol 
                FROM daily_candles 
                WHERE trading_symbol NOT LIKE '0%'
                  AND trading_symbol NOT LIKE '%NIFTY%'
                  AND trading_symbol NOT LIKE '%SENSEX%'
                  AND trading_symbol NOT LIKE '%BEES%'
                  AND trading_symbol NOT LIKE '%ETF%'
                  AND trading_symbol NOT LIKE 'INDIA VIX%'
                ORDER BY trading_symbol ASC;
            """).fetchall()
            if rows:
                return sorted(list(set(r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0])))
        except Exception:
            pass

    return []


def get_db_stats() -> Dict[str, Any]:
    """Returns overview statistics of stored database records from DuckDB."""
    conn = get_connection()
    with _lock:
        try:
            inst_count = conn.execute("SELECT COUNT(*) FROM instruments;").fetchone()[0]
        except Exception:
            inst_count = 0

        try:
            candles_res = conn.execute("SELECT COUNT(DISTINCT trading_symbol), COUNT(*) FROM daily_candles;").fetchone()
            symbols_with_data = candles_res[0] or 0
            total_candles = candles_res[1] or 0
        except Exception:
            symbols_with_data = 0
            total_candles = 0

        try:
            date_range = conn.execute("SELECT MIN(date), MAX(date) FROM daily_candles;").fetchone()
            earliest = date_range[0] if date_range else None
            latest = date_range[1] if date_range else None
        except Exception:
            earliest = None
            latest = None

        return {
            "total_instruments": inst_count,
            "symbols_with_candles": symbols_with_data,
            "total_candles": total_candles,
            "earliest_date": earliest,
            "latest_date": latest
        }


def save_daily_candles(candles: List[dict]):
    """Upserts daily candles into DuckDB."""
    if not candles:
        return

    df = pd.DataFrame(candles)
    # Ensure required columns exist
    for col, default in [
        ("instrument_key", ""), ("trading_symbol", ""), ("date", ""),
        ("open", 0.0), ("high", 0.0), ("low", 0.0), ("close", 0.0),
        ("volume", 0), ("open_interest", 0)
    ]:
        if col not in df.columns:
            df[col] = default

    with _lock:
        conn = get_write_connection()
        try:
            conn.register("_temp_candles_in", df)
            conn.execute("""
                INSERT INTO daily_candles
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
                FROM _temp_candles_in
                ON CONFLICT (instrument_key, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest;
            """)
            conn.unregister("_temp_candles_in")
        finally:
            conn.close()


def get_resampled_candles(
    symbol: str,
    interval_minutes: int = 75,
    limit: int = 2500,
    min_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs high-speed in-database C++ SQL time-bucketing resampling from 1m candles.
    Reads either from DuckDB candles_1m table or directly from symbol parquet file.
    Takes ~2-10ms per stock instead of 1000ms+ in Pandas.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    safe_sym = clean_sym.replace("/", "_").replace("\\", "_")
    conn = get_connection()

    symbol_parquet = config.DATA_DIR / "by_symbol" / f"{safe_sym}.parquet"

    # Decide data source: local symbol parquet file or candles_1m table
    if symbol_parquet.exists():
        from_source = f"read_parquet('{str(symbol_parquet)}')"
        where_clause = "WHERE 1=1"
        params = []
    else:
        from_source = "candles_1m"
        where_clause = "WHERE symbol = ?"
        params = [clean_sym]

    query = f"""
        SELECT 
            time_bucket(INTERVAL '{interval_minutes} minutes', timestamp) AS bucket_time,
            first(open) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close) AS close,
            sum(volume) AS volume
        FROM {from_source}
        {where_clause}
    """

    if min_date:
        query += " AND timestamp >= ?::TIMESTAMP"
        params.append(min_date)

    query += f"""
        GROUP BY 1
        ORDER BY bucket_time DESC
        LIMIT {limit};
    """

    try:
        with _lock:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error resampling candles for {symbol} ({interval_minutes}m): {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.sort_values("bucket_time", ascending=True, inplace=True)
    df["bucket_time"] = pd.to_datetime(df["bucket_time"])
    
    if df["bucket_time"].dt.tz is None:
        df["bucket_time"] = df["bucket_time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    else:
        df["bucket_time"] = df["bucket_time"].dt.tz_convert("Asia/Kolkata")

    df.set_index("bucket_time", inplace=True)
    return df


def get_1m_candle_count(symbol: Optional[str] = None) -> int:
    """Returns row count of 1-minute candles for a symbol or across the database."""
    conn = get_connection()
    with _lock:
        if symbol:
            clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
            res = conn.execute("SELECT count(*) FROM candles_1m WHERE symbol = ?", [clean_sym]).fetchone()
        else:
            res = conn.execute("SELECT count(*) FROM candles_1m").fetchone()
    return res[0] if res else 0


def get_instrument_by_symbol(symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches instrument record by trading symbol from DuckDB."""
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    conn = get_connection()

    with _lock:
        if exchange:
            row = conn.execute("""
                SELECT instrument_key, trading_symbol, name, exchange, instrument_type, tick_size, lot_size
                FROM instruments
                WHERE (trading_symbol = ? OR trading_symbol = ? || '-EQ') AND exchange = ?
                ORDER BY CASE WHEN instrument_key LIKE '%|INE%' OR instrument_key LIKE '%|INF%' OR instrument_key LIKE '%|IN9%' THEN 0 ELSE 1 END
                LIMIT 1;
            """, [clean_sym, clean_sym, exchange]).fetchone()
        else:
            row = conn.execute("""
                SELECT instrument_key, trading_symbol, name, exchange, instrument_type, tick_size, lot_size
                FROM instruments
                WHERE (trading_symbol = ? OR trading_symbol = ? || '-EQ')
                ORDER BY 
                    CASE WHEN exchange = 'NSE_INDEX' THEN 0 WHEN exchange = 'NSE_EQ' THEN 1 ELSE 2 END,
                    CASE WHEN instrument_key LIKE '%|INE%' OR instrument_key LIKE '%|INF%' OR instrument_key LIKE '%|IN9%' THEN 0 ELSE 1 END
                LIMIT 1;
            """, [clean_sym, clean_sym]).fetchone()

    if not row:
        return None

    return {
        "instrument_key": row[0],
        "trading_symbol": row[1],
        "name": row[2],
        "exchange": row[3],
        "instrument_type": row[4],
        "tick_size": row[5],
        "lot_size": row[6]
    }


def upsert_instruments(instruments: List[Dict[str, Any]]):
    """Inserts or updates instruments in DuckDB."""
    if not instruments:
        return

    df = pd.DataFrame(instruments)
    for col, default in [
        ("instrument_key", ""), ("trading_symbol", ""), ("name", ""),
        ("exchange", "NSE_EQ"), ("instrument_type", "EQUITY"),
        ("tick_size", 0.05), ("lot_size", 1)
    ]:
        if col not in df.columns:
            df[col] = default

    with _lock:
        conn = get_write_connection()
        try:
            conn.register("_temp_inst_in", df)
            conn.execute("""
                INSERT INTO instruments (
                    instrument_key, trading_symbol, name, exchange, instrument_type, tick_size, lot_size, last_updated
                )
                SELECT
                    instrument_key, trading_symbol, name, exchange, instrument_type, tick_size, lot_size, CURRENT_TIMESTAMP
                FROM _temp_inst_in
                ON CONFLICT (instrument_key) DO UPDATE SET
                    trading_symbol = EXCLUDED.trading_symbol,
                    name = EXCLUDED.name,
                    exchange = EXCLUDED.exchange,
                    instrument_type = EXCLUDED.instrument_type,
                    tick_size = EXCLUDED.tick_size,
                    lot_size = EXCLUDED.lot_size,
                    last_updated = CURRENT_TIMESTAMP;
            """)
            conn.unregister("_temp_inst_in")
        finally:
            conn.close()

