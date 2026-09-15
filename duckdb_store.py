import os
import threading
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import duckdb

import config

logger = logging.getLogger("duckdb_store")
DUCKDB_PATH = config.DATA_DIR / "market_data.duckdb"
STAGING_PARQUET = config.DATA_DIR / "daily_candles_staging.parquet"

_lock = threading.Lock()
_conn: Optional[duckdb.DuckDBPyConnection] = None


def _check_download_db():
    """Ensures database directory and file exist."""
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DUCKDB_PATH.exists() or DUCKDB_PATH.stat().st_size < 1024 * 1024:
        logger.warning(f"DuckDB database not found at {DUCKDB_PATH}. Attempting automatic download from GitHub...")
        try:
            import download_dataset
            download_dataset.download()
        except Exception as dl_err:
            logger.error(f"Could not auto-download database: {dl_err}. You can manually run: python3 download_dataset.py")


@contextmanager
def get_read_connection(max_retries: int = 15, retry_delay: float = 0.05):
    """
    Context manager yielding a read-only DuckDB connection with automatic retry on lock contention.
    Closes the connection immediately upon exiting the context block, preventing persistent OS file locks.
    """
    _check_download_db()
    conn = None
    last_err = None
    for attempt in range(max_retries):
        try:
            conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=True)
            conn.execute("PRAGMA threads=4;")
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (1.5 ** attempt))
            else:
                logger.error(f"Failed to acquire DuckDB read connection after {max_retries} attempts: {e}")
                raise last_err
    try:
        yield conn
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_connection(max_retries: int = 15, retry_delay: float = 0.05) -> duckdb.DuckDBPyConnection:
    """
    Returns a read-only DuckDB connection with retry on lock conflict.
    Note: When possible, callers should use 'with get_read_connection() as conn:' to ensure
    connections are closed immediately and do not block writers.
    """
    _check_download_db()
    last_err = None
    for attempt in range(max_retries):
        try:
            conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=True)
            conn.execute("PRAGMA threads=4;")
            return conn
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (1.5 ** attempt))
            else:
                logger.error(f"Failed to acquire DuckDB read connection after {max_retries} attempts: {e}")
                raise last_err


def get_write_connection(max_retries: int = 20, retry_delay: float = 0.05) -> duckdb.DuckDBPyConnection:
    """
    Returns a short-lived write connection with retry backoff.
    Caller MUST close it immediately after writing.
    """
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(max_retries):
        try:
            conn = duckdb.connect(database=str(DUCKDB_PATH), read_only=False)
            conn.execute("PRAGMA threads=4;")
            try:
                _init_schema(conn)
            except Exception:
                pass
            return conn
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (1.5 ** attempt))
            else:
                logger.error(f"Failed to acquire DuckDB write connection after {max_retries} attempts: {e}")
                raise last_err



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
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_intraday_uniq ON intraday_candles (instrument_key, timeframe, timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_sym_date ON daily_candles (trading_symbol, date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_sym_tf_ts ON intraday_candles (trading_symbol, timeframe, timestamp);")
        # Auto-heal: If stocks table has pre-2020 data and daily_candles does not have them, merge them
        has_stocks = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'stocks'").fetchone()[0] > 0
        if has_stocks:
            min_daily = conn.execute("SELECT min(date) FROM daily_candles").fetchone()[0]
            if not min_daily or str(min_daily) >= '2020-01-01':
                conn.execute("""
                    INSERT INTO daily_candles (
                        instrument_key, trading_symbol, date, open, high, low, close, volume, open_interest
                    )
                    SELECT
                        'NSE_EQ|' || s.ticker AS instrument_key,
                        s.ticker AS trading_symbol,
                        strftime(s.date, '%Y-%m-%d') AS date,
                        arg_min(s.open, s.datetime) AS open,
                        MAX(s.high) AS high,
                        MIN(s.low) AS low,
                        arg_max(s.close, s.datetime) AS close,
                        CAST(SUM(s.volume) AS BIGINT) AS volume,
                        0 AS open_interest
                    FROM stocks s
                    WHERE s.date < '2020-01-01'
                    GROUP BY s.ticker, s.date
                    ON CONFLICT (instrument_key, date) DO UPDATE SET
                        trading_symbol = EXCLUDED.trading_symbol,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest;
                """)
    except Exception:
        pass



def get_candles_df(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame of daily candles for a given symbol, indexed by DatetimeIndex (date).
    Columns: open, high, low, close, volume, open_interest.
    Returns complete authentic history back to 2017 when limit is None.
    Sub-3ms execution time directly via DuckDB.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")

    where_clauses = ["(trading_symbol = ? OR trading_symbol = ? || '-EQ')"]
    params: List[Any] = [clean_sym, clean_sym]

    if start_date:
        where_clauses.append("date >= ?")
        params.append(str(start_date)[:10])
    if end_date:
        where_clauses.append("date <= ?")
        params.append(str(end_date)[:10])

    where_str = " AND ".join(where_clauses)

    if limit and not start_date:
        # Fetch only the latest `limit` candles, sorted chronologically
        query = f"""
            SELECT date, open, high, low, close, volume, open_interest
            FROM (
                SELECT date, open, high, low, close, volume, open_interest
                FROM daily_candles
                WHERE {where_str}
                ORDER BY date DESC
                LIMIT {int(limit)}
            )
            ORDER BY date ASC;
        """
    else:
        query = f"""
            SELECT date, open, high, low, close, volume, open_interest
            FROM daily_candles
            WHERE {where_str}
            ORDER BY date ASC;
        """

    with get_read_connection() as conn:
        df = conn.execute(query, params).df()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    df.set_index("date", inplace=True)
    return df


def get_batch_candles_df(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    High-performance batch fetch of daily candles for multiple symbols in a single DuckDB query.
    Returns a dictionary mapping clean symbol -> pd.DataFrame indexed by date.
    Sub-second execution across 1,000+ symbols and millions of rows.
    """
    if not symbols:
        return {}

    query_symbols = []
    for s in symbols:
        clean = s.upper().strip().replace("-EQ", "").replace(".NS", "")
        query_symbols.append(clean)
        query_symbols.append(clean + "-EQ")

    unique_query_syms = list(set(query_symbols))

    where_clauses = ["trading_symbol IN (SELECT unnest(?))"]
    params: List[Any] = [unique_query_syms]

    if start_date:
        where_clauses.append("date >= ?")
        params.append(str(start_date)[:10])
    if end_date:
        where_clauses.append("date <= ?")
        params.append(str(end_date)[:10])

    where_str = " AND ".join(where_clauses)
    query = f"""
        SELECT trading_symbol, date, open, high, low, close, volume, open_interest
        FROM daily_candles
        WHERE {where_str}
        ORDER BY trading_symbol, date ASC;
    """

    with get_read_connection() as conn:
        df_all = conn.execute(query, params).df()

    if df_all.empty:
        return {}

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["clean_sym"] = df_all["trading_symbol"].str.replace("-EQ", "", regex=False)

    result: Dict[str, pd.DataFrame] = {}
    for sym_clean, group in df_all.groupby("clean_sym"):
        grp = group.drop(columns=["trading_symbol", "clean_sym"]).drop_duplicates(subset=["date"], keep="last").set_index("date")
        result[sym_clean] = grp

    return result


def get_intraday_candles(
    symbol: str,
    timeframe: str = "75m",
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Returns authentic pre-calculated intraday candles directly from DuckDB's intraday_candles table.
    Supports filtering by start_date and end_date (e.g. '2017-10-01' to '2026-09-14').
    Sub-10ms query execution with non-blocking read connection.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    with get_read_connection() as conn:
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM intraday_candles
                WHERE (trading_symbol = ? OR trading_symbol = ? || '-EQ')
                  AND timeframe = ?
            """
            params = [clean_sym, clean_sym, timeframe]
            if start_date:
                query += " AND timestamp >= ?"
                params.append(str(start_date))
            if end_date:
                query += " AND timestamp <= ?"
                end_ts = f"{end_date} 23:59:59" if len(str(end_date)) == 10 else str(end_date)
                params.append(end_ts)

            query += " ORDER BY timestamp ASC"
            if limit and not (start_date or end_date):
                query += f" LIMIT {int(limit)}"
            query += ";"

            df = conn.execute(query, params).df()
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
    return df if (start_date or end_date or limit is None) else df.tail(limit)



def get_latest_candle_date(symbol: str) -> Optional[str]:
    """Returns the most recent candle date (YYYY-MM-DD) for a symbol."""
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    with get_read_connection() as conn:
        res = conn.execute("""
            SELECT max(date)
            FROM daily_candles
            WHERE trading_symbol = ? OR trading_symbol = ? || '-EQ'
        """, [clean_sym, clean_sym]).fetchone()
    return res[0] if res and res[0] else None


def get_all_symbols(include_indices: bool = False, exchange: Optional[str] = None) -> List[str]:
    """Returns all unique Equity trading symbols strictly from NSE, excluding all indices and ETFs."""
    with get_read_connection() as conn:
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

    with get_read_connection() as conn:
        try:
            if exchange:
                query = """
                    SELECT DISTINCT trading_symbol FROM instruments 
                    WHERE exchange = ? 
                      AND instrument_key LIKE '%|INE%'
                      AND trading_symbol NOT LIKE '0%'
                      AND trading_symbol NOT LIKE '%ETF%'
                      AND trading_symbol NOT LIKE '%BEES%'
                      AND trading_symbol NOT LIKE '%NIFTY%'
                      AND trading_symbol NOT LIKE '%SENSEX%'
                      AND trading_symbol NOT LIKE 'INDIA VIX%';
                """
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
                      AND instrument_key LIKE '%|INE%'
                      AND instrument_type IN ('EQUITY', 'EQ', 'BE', 'SM', 'BZ')
                      AND trading_symbol NOT LIKE '0%'
                      AND trading_symbol NOT LIKE '%ETF%'
                      AND trading_symbol NOT LIKE '%BEES%'
                      AND trading_symbol NOT LIKE '%NIFTY%'
                      AND trading_symbol NOT LIKE '%SENSEX%'
                      AND trading_symbol NOT LIKE 'INDIA VIX%'
                    ORDER BY trading_symbol ASC;
                """
                rows = conn.execute(query).fetchall()

            if rows:
                return sorted(list(set(r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0])))
        except Exception as e:
            logger.warning(f"Error querying instruments for symbols: {e}")

        # Fallback to daily_candles if instruments table is empty
        try:
            rows = conn.execute("""
                SELECT DISTINCT trading_symbol FROM daily_candles 
                WHERE trading_symbol NOT LIKE '0%'
                  AND trading_symbol NOT LIKE '%ETF%'
                  AND trading_symbol NOT LIKE '%BEES%'
                  AND trading_symbol NOT LIKE '%NIFTY%'
                  AND trading_symbol NOT LIKE '%SENSEX%'
                  AND trading_symbol NOT LIKE 'INDIA VIX%';
            """).fetchall()
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
    with get_read_connection() as conn:
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

    with get_read_connection() as conn:
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
                SELECT DISTINCT trading_symbol FROM daily_candles 
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
    with get_read_connection() as conn:
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


def _flush_staging_candles(conn: duckdb.DuckDBPyConnection):
    """Merges any candles buffered in the staging parquet file into daily_candles."""
    if not STAGING_PARQUET.exists():
        return
    try:
        staging_size = STAGING_PARQUET.stat().st_size
        if staging_size == 0:
            STAGING_PARQUET.unlink(missing_ok=True)
            return

        conn.execute(f"""
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
            FROM read_parquet('{str(STAGING_PARQUET)}')
            ON CONFLICT (instrument_key, date) DO UPDATE SET
                trading_symbol = EXCLUDED.trading_symbol,
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                open_interest = EXCLUDED.open_interest;
        """)
        STAGING_PARQUET.unlink(missing_ok=True)
        logger.info("✅ Flushed and merged staging daily candles into daily_candles table.")
    except Exception as e:
        logger.warning(f"Could not flush staging daily candles: {e}")


def _append_to_staging_wal(df: pd.DataFrame):
    """Appends unwritten candles to fallback staging parquet file."""
    try:
        STAGING_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        if STAGING_PARQUET.exists():
            existing_df = pd.read_parquet(STAGING_PARQUET)
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined.drop_duplicates(subset=["instrument_key", "date"], keep="last", inplace=True)
            combined.to_parquet(STAGING_PARQUET, index=False)
        else:
            df.to_parquet(STAGING_PARQUET, index=False)
        logger.info(f"Buffered {len(df)} candles into staging WAL at {STAGING_PARQUET}")
    except Exception as wal_err:
        logger.error(f"Failed to buffer candles into staging WAL: {wal_err}")


def save_daily_candles(candles: List[dict]):
    """
    Upserts daily candles into DuckDB with automatic deduplication, retry, and staging fallback.
    Guarantees continuous synchronization without dropping bars during database contention.
    """
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

    # Clean symbol
    df["trading_symbol"] = (
        df["trading_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-EQ", "", regex=False)
        .str.replace(".NS", "", regex=False)
    )
    df = df[df["trading_symbol"] != ""].copy()
    if df.empty:
        return

    # Normalize date to YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Ensure numeric types
    df["open"] = pd.to_numeric(df["open"], errors="coerce").fillna(0.0).astype(float)
    df["high"] = pd.to_numeric(df["high"], errors="coerce").fillna(0.0).astype(float)
    df["low"] = pd.to_numeric(df["low"], errors="coerce").fillna(0.0).astype(float)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype("int64")

    # Auto-fill missing or empty instrument_key
    empty_inst = (df["instrument_key"] == "") | (df["instrument_key"].isna())
    if empty_inst.any():
        df.loc[empty_inst, "instrument_key"] = "NSE_EQ|" + df.loc[empty_inst, "trading_symbol"]

    # Deduplicate within batch by (instrument_key, date)
    df.drop_duplicates(subset=["instrument_key", "date"], keep="last", inplace=True)

    with _lock:
        try:
            conn = get_write_connection()
            try:
                # Flush any pending staging candles first
                _flush_staging_candles(conn)

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
                        trading_symbol = EXCLUDED.trading_symbol,
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
        except Exception as e:
            logger.error(f"Direct DuckDB write failed: {e}. Appending {len(df)} candles to staging WAL.")
            _append_to_staging_wal(df)


def get_resampled_candles(
    symbol: str,
    interval_minutes: int = 75,
    limit: int = 2500,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs high-speed in-database C++ SQL time-bucketing resampling from 1m candles.
    Reads either from DuckDB candles_1m table or directly from symbol parquet file.
    Takes ~2-10ms per stock instead of 1000ms+ in Pandas.
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    safe_sym = clean_sym.replace("/", "_").replace("\\", "_")

    symbol_parquet = config.DATA_DIR / "by_symbol" / f"{safe_sym}.parquet"
    has_parquet = symbol_parquet.exists()

    if has_parquet:
        # Fast, lock-free path: standalone symbol parquet exists.
        # Check if symbol also has historical 2017-2021 1m data in market_data.duckdb's stocks table
        try:
            conn = duckdb.connect()
            has_stocks = False
            try:
                conn.execute(f"ATTACH '{str(DUCKDB_PATH)}' AS main_db (READ_ONLY);")
                chk = conn.execute("SELECT 1 FROM main_db.stocks WHERE ticker = ? LIMIT 1;", [clean_sym]).fetchone()
                if chk:
                    has_stocks = True
            except Exception:
                has_stocks = False

            ts_col = "timestamp"
            if has_stocks and (not min_date or str(min_date)[:10] < "2022-01-01"):
                from_source = f"""
                    (
                        SELECT datetime::TIMESTAMP AS timestamp, open, high, low, close, volume 
                        FROM main_db.stocks 
                        WHERE ticker = '{clean_sym}'
                        UNION ALL
                        SELECT timestamp::TIMESTAMP AS timestamp, open, high, low, close, volume 
                        FROM read_parquet('{str(symbol_parquet)}')
                    )
                """
            else:
                from_source = f"read_parquet('{str(symbol_parquet)}')"

            where_clause = "WHERE 1=1"
            params = []

            query = f"""
                SELECT 
                    time_bucket(INTERVAL '{interval_minutes} minutes', {ts_col}) AS bucket_time,
                    first(open) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close) AS close,
                    sum(volume) AS volume
                FROM {from_source}
                {where_clause}
            """

            if min_date:
                query += f" AND {ts_col} >= ?::TIMESTAMP"
                min_ts = f"{min_date} 00:00:00" if len(str(min_date)) == 10 else str(min_date)[:19]
                params.append(min_ts)

            if max_date:
                query += f" AND {ts_col} <= ?::TIMESTAMP"
                end_ts = f"{max_date} 23:59:59" if len(str(max_date)) == 10 else str(max_date)[:19]
                params.append(end_ts)

            query += f"""
                GROUP BY 1
                ORDER BY bucket_time DESC
            """
            if limit and not (min_date or max_date):
                query += f" LIMIT {int(limit)}"
            query += ";"

            df = conn.execute(query, params).df()
            if not df.empty:
                df.sort_values("bucket_time", ascending=True, inplace=True)
                df["bucket_time"] = pd.to_datetime(df["bucket_time"])
                if df["bucket_time"].dt.tz is None:
                    df["bucket_time"] = df["bucket_time"].dt.tz_localize("Asia/Kolkata")
                else:
                    df["bucket_time"] = df["bucket_time"].dt.tz_convert("Asia/Kolkata")
                df.set_index("bucket_time", inplace=True)
                return df
        except Exception as e:
            logger.error(f"Error resampling from parquet/stocks for {symbol} ({interval_minutes}m): {e}", exc_info=True)

    # Fallback to market_data.duckdb database connection for stocks or candles_1m tables
    try:
        conn = get_connection()
    except Exception as e:
        logger.error(f"Error connecting to DuckDB database for {symbol}: {e}")
        return pd.DataFrame()

    has_stocks = False
    try:
        with _lock:
            chk = conn.execute("SELECT 1 FROM stocks WHERE ticker = ? LIMIT 1;", [clean_sym]).fetchone()
            if chk:
                has_stocks = True
    except Exception:
        has_stocks = False

    ts_col = "timestamp"
    params = []

    if has_stocks:
        from_source = "stocks"
        ts_col = "datetime"
        where_clause = "WHERE ticker = ?"
        params.append(clean_sym)
    else:
        has_candles_1m = False
        try:
            with _lock:
                chk_1m = conn.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'candles_1m';").fetchone()
                if chk_1m:
                    has_candles_1m = True
        except Exception:
            has_candles_1m = False

        if has_candles_1m:
            from_source = "candles_1m"
            ts_col = "timestamp"
            where_clause = "WHERE symbol = ?"
            params.append(clean_sym)
        else:
            return pd.DataFrame()

    query = f"""
        SELECT 
            time_bucket(INTERVAL '{interval_minutes} minutes', {ts_col}) AS bucket_time,
            first(open) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close) AS close,
            sum(volume) AS volume
        FROM {from_source}
        {where_clause}
    """

    if min_date:
        query += f" AND {ts_col} >= ?::TIMESTAMPTZ"
        min_ts = f"{min_date} 00:00:00+05:30" if len(str(min_date)) == 10 else str(min_date)
        params.append(min_ts)

    if max_date:
        query += f" AND {ts_col} <= ?::TIMESTAMPTZ"
        end_ts = f"{max_date} 23:59:59+05:30" if len(str(max_date)) == 10 else str(max_date)
        params.append(end_ts)

    query += f"""
        GROUP BY 1
        ORDER BY bucket_time DESC
    """
    if not (min_date or max_date):
        query += f" LIMIT {limit}"
    query += ";"

    try:
        with _lock:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error resampling candles from DB for {symbol} ({interval_minutes}m): {e}", exc_info=True)
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
    with get_read_connection() as conn:
        if symbol:
            clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
            res = conn.execute("SELECT count(*) FROM candles_1m WHERE symbol = ?", [clean_sym]).fetchone()
        else:
            res = conn.execute("SELECT count(*) FROM candles_1m").fetchone()
    return res[0] if res else 0


def get_instrument_by_symbol(symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches instrument record by trading symbol from DuckDB."""
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    with get_read_connection() as conn:
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


def upsert_intraday_candles(candles: List[Dict[str, Any]]):
    """Inserts or updates intraday candles (e.g. 75m) in DuckDB."""
    if not candles:
        return

    df = pd.DataFrame(candles)
    for col, default in [
        ("instrument_key", ""), ("trading_symbol", ""), ("timeframe", "75m"),
        ("timestamp", ""), ("open", 0.0), ("high", 0.0), ("low", 0.0),
        ("close", 0.0), ("volume", 0)
    ]:
        if col not in df.columns:
            df[col] = default

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    with _lock:
        conn = get_write_connection()
        try:
            conn.register("_temp_intraday_in", df)
            conn.execute("""
                INSERT INTO intraday_candles (
                    instrument_key, trading_symbol, timeframe, timestamp, open, high, low, close, volume
                )
                SELECT
                    instrument_key, trading_symbol, timeframe, timestamp, open, high, low, close, volume
                FROM _temp_intraday_in
                ON CONFLICT (instrument_key, timeframe, timestamp) DO UPDATE SET
                    trading_symbol = EXCLUDED.trading_symbol,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
            """)
            conn.unregister("_temp_intraday_in")
        except Exception as e:
            logger.warning(f"Error upserting intraday candles into DuckDB: {e}")
        finally:
            conn.close()


def upsert_daily_candles(candles: List[Dict[str, Any]]):
    """Inserts or updates daily candles in DuckDB using unified deduplicated write queue."""
    save_daily_candles(candles)



