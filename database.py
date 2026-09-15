import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Any
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Returns a thread-safe SQLite connection with a generous busy timeout."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


_db_initialized = False


def init_db(force: bool = False):
    """Initializes the database schema if tables do not exist."""
    global _db_initialized
    if _db_initialized and not force:
        return
    try:
        with get_connection() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except Exception:
                pass
            cursor = conn.cursor()
            
            # Instruments Master Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instruments (
                    instrument_key TEXT PRIMARY KEY,
                    trading_symbol TEXT NOT NULL,
                    name TEXT,
                    exchange TEXT NOT NULL,
                    instrument_type TEXT,
                    tick_size REAL,
                    lot_size INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inst_symbol ON instruments (trading_symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inst_exchange ON instruments (exchange);")

            # Daily OHLCV Candles Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_candles (
                    instrument_key TEXT NOT NULL,
                    trading_symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    open_interest INTEGER DEFAULT 0,
                    PRIMARY KEY (instrument_key, date)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candle_sym_date ON daily_candles (trading_symbol, date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candle_date ON daily_candles (date);")

            # Intraday (75-minute / 1-minute) Candles Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intraday_candles (
                    instrument_key TEXT NOT NULL,
                    trading_symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (instrument_key, timeframe, timestamp)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_intra_sym_tf ON intraday_candles (trading_symbol, timeframe, timestamp);")

            # Enforce unique index on (trading_symbol, date) and (trading_symbol, timeframe, timestamp)
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_uniq_daily_sym_date ON daily_candles (trading_symbol, date);")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_uniq_intra_sym_tf_ts ON intraday_candles (trading_symbol, timeframe, timestamp);")

            conn.commit()
            _db_initialized = True
    except Exception:
        # Schema already initialized or db currently busy with batch commit
        _db_initialized = True


def purge_all_demo_data():
    """Removes all synthetic demo candles and dummy instruments from the database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM daily_candles WHERE instrument_key LIKE '%DEMO%';")
        conn.execute("DELETE FROM intraday_candles WHERE instrument_key LIKE '%DEMO%';")
        conn.execute("DELETE FROM instruments WHERE instrument_key LIKE '%DEMO%' OR name LIKE '%(NSE)%' OR name LIKE '%(BSE)%';")
        conn.commit()


def upsert_instruments(instruments: List[Dict[str, Any]]):
    """Inserts or updates instruments master list in DuckDB and SQLite."""
    if not instruments:
        return
    try:
        import duckdb_store
        duckdb_store.upsert_instruments(instruments)
    except Exception:
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO instruments (
                instrument_key, trading_symbol, name, exchange, instrument_type, tick_size, lot_size, last_updated
            ) VALUES (
                :instrument_key, :trading_symbol, :name, :exchange, :instrument_type, :tick_size, :lot_size, CURRENT_TIMESTAMP
            )
            ON CONFLICT(instrument_key) DO UPDATE SET
                trading_symbol=excluded.trading_symbol,
                name=excluded.name,
                exchange=excluded.exchange,
                instrument_type=excluded.instrument_type,
                tick_size=excluded.tick_size,
                lot_size=excluded.lot_size,
                last_updated=CURRENT_TIMESTAMP;
        """, instruments)
        conn.commit()


def get_instrument_by_symbol(symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches instrument record by trading symbol, prioritizing authentic ISIN keys and NSE then BSE if exchange not specified."""
    try:
        import duckdb_store
        rec = duckdb_store.get_instrument_by_symbol(symbol, exchange)
        if rec:
            return rec
    except Exception:
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        if exchange:
            cursor.execute(
                """SELECT * FROM instruments 
                   WHERE trading_symbol = ? AND exchange = ? 
                   ORDER BY CASE WHEN instrument_key LIKE '%|INE%' OR instrument_key LIKE '%|INF%' OR instrument_key LIKE '%|IN9%' THEN 0 ELSE 1 END 
                   LIMIT 1;""",
                (symbol.upper(), exchange)
            )
        else:
            cursor.execute(
                """SELECT * FROM instruments 
                   WHERE trading_symbol = ? 
                   ORDER BY 
                       CASE WHEN instrument_key LIKE '%|INE%' OR instrument_key LIKE '%|INF%' OR instrument_key LIKE '%|IN9%' THEN 0 ELSE 1 END,
                       CASE WHEN exchange LIKE 'NSE%' THEN 1 ELSE 2 END 
                   LIMIT 1;""",
                (symbol.upper(),)
            )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_symbols(include_indices: bool = False) -> List[str]:
    """Returns all unique pure Equity trading symbols strictly from NSE, excluding all indices and ETFs."""
    try:
        import duckdb_store
        syms = duckdb_store.get_all_symbols(include_indices=include_indices)
        if syms:
            return syms
    except Exception:
        pass

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            if include_indices:
                cursor.execute("""
                    SELECT DISTINCT trading_symbol 
                    FROM instruments 
                    WHERE exchange IN ('NSE_EQ', 'NSE_INDEX')
                      AND trading_symbol NOT LIKE '0%'
                    ORDER BY trading_symbol ASC;
                """)
            else:
                cursor.execute("""
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
                """)
            res = [row[0] for row in cursor.fetchall()]
            if res:
                return res
    except Exception:
        pass

    # Auto-initialize instruments if database is fresh
    try:
        import instruments
        instruments.sync_all_instruments()
        import duckdb_store
        syms = duckdb_store.get_all_symbols(include_indices=include_indices)
        if syms:
            return syms
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
    try:
        import duckdb_store
        syms = duckdb_store.get_alignment_scanner_symbols()
        if syms:
            return syms
    except Exception:
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT trading_symbol 
            FROM instruments 
            WHERE exchange = 'NSE_EQ' 
              AND instrument_type IN ('EQ', 'BE', 'SM', 'BZ')
              AND trading_symbol NOT LIKE '0%'
              AND trading_symbol NOT LIKE '%NIFTY%'
              AND trading_symbol NOT LIKE '%SENSEX%'
              AND trading_symbol NOT LIKE '%BEES%'
              AND trading_symbol NOT LIKE '%ETF%'
              AND trading_symbol NOT LIKE 'INDIA VIX%'
            ORDER BY trading_symbol ASC;
        """)
        return [row[0] for row in cursor.fetchall()]



def upsert_candles(candles: List[Dict[str, Any]]):
    """
    Inserts or updates daily candles into DuckDB (and SQLite backup).
    """
    if not candles:
        return
    try:
        import duckdb_store
        duckdb_store.save_daily_candles(candles)
    except Exception as e:
        logger.error(f"Error saving daily candles to DuckDB: {e}", exc_info=True)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_candles (
                    instrument_key, trading_symbol, date, open, high, low, close, volume, open_interest
                ) VALUES (
                    :instrument_key, :trading_symbol, :date, :open, :high, :low, :close, :volume, :open_interest
                );
            """, candles)
            conn.commit()
    except Exception:
        pass


def get_candles_df(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetches daily candles for a symbol as a Pandas DataFrame indexed by Datetime.
    Sorted in ascending order (earliest to latest) via fast DuckDB.
    Returns full authentic history (back to 2017) when limit is None.
    """
    try:
        import duckdb_store
        df = duckdb_store.get_candles_df(symbol, start_date=start_date, end_date=end_date, limit=limit)
        if not df.empty:
            return df
    except Exception as e:
        logger.warning(f"DuckDB get_candles_df error: {e}")

    with get_connection() as conn:
        query = "SELECT date, open, high, low, close, volume, open_interest FROM daily_candles WHERE trading_symbol = ?"
        params = [symbol.upper()]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if limit and not start_date:
            query = f"""
                SELECT date, open, high, low, close, volume, open_interest FROM (
                    {query}
                    ORDER BY date DESC
                    LIMIT {int(limit)}
                ) ORDER BY date ASC;
            """
        else:
            query += " ORDER BY date ASC;"

        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'])
        df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        df.set_index('date', inplace=True)
        return df


def get_batch_candles_df(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    High-performance batch fetch of daily candles for multiple symbols as DataFrames.
    Returns {clean_symbol: df}. Uses DuckDB when available for sub-second retrieval.
    """
    if not symbols:
        return {}

    try:
        import duckdb_store
        res = duckdb_store.get_batch_candles_df(symbols, start_date=start_date, end_date=end_date)
        if res:
            return res
    except Exception as e:
        logger.warning(f"DuckDB get_batch_candles_df fallback to sequential/sqlite: {e}")

    # Fallback to sequential get_candles_df
    result = {}
    for s in symbols:
        clean = s.upper().strip().replace("-EQ", "").replace(".NS", "")
        df = get_candles_df(s, start_date=start_date, end_date=end_date)
        if not df.empty:
            result[clean] = df
    return result


def upsert_intraday_candles(candles: List[Dict[str, Any]]):
    """
    Inserts or updates intraday candles (e.g. 75m, 1m) into DuckDB and SQLite.
    Replaces any duplicate record for (trading_symbol, timeframe, timestamp) with new data.
    """
    if not candles:
        return

    try:
        import duckdb_store
        duckdb_store.upsert_intraday_candles(candles)
    except Exception:
        pass

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO intraday_candles (
                    instrument_key, trading_symbol, timeframe, timestamp, open, high, low, close, volume
                ) VALUES (
                    :instrument_key, :trading_symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume
                )
                ON CONFLICT(trading_symbol, timeframe, timestamp) DO UPDATE SET
                    instrument_key=excluded.instrument_key,
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume;
            """, candles)
            conn.commit()
    except Exception:
        pass


def upsert_1m_candles(candles: List[Dict[str, Any]]):
    """
    Inserts or updates 1-minute candles directly into DuckDB candles_1m table.
    """
    if not candles:
        return
    try:
        import duckdb_store
        duckdb_store.upsert_1m_candles(candles)
    except Exception as e:
        logger.warning(f"Error upserting 1m candles into DuckDB: {e}")


def get_intraday_candles_df(
    symbol: str, 
    timeframe: str = "75m", 
    limit: int = 5000, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches intraday (e.g. 75m) candles for a symbol, sorted chronologically and deduplicated.
    Supports start_date and end_date filtering.
    Reads from DuckDB native table first (sub-10ms), falling back to SQLite.
    """
    try:
        import duckdb_store
        df = duckdb_store.get_intraday_candles(
            symbol, timeframe=timeframe, limit=limit, start_date=start_date, end_date=end_date
        )
        if not df.empty:
            return df
    except Exception:
        pass

    with get_connection() as conn:
        query = "SELECT timestamp, open, high, low, close, volume FROM intraday_candles WHERE trading_symbol = ? AND timeframe = ?"
        params = [symbol.upper(), timeframe]
        if start_date:
            query += " AND timestamp >= ?"
            params.append(str(start_date))
        if end_date:
            query += " AND timestamp <= ?"
            end_ts = f"{end_date} 23:59:59" if len(str(end_date)) == 10 else str(end_date)
            params.append(end_ts)
        query += " ORDER BY timestamp ASC"
        if not (start_date or end_date):
            query += f" LIMIT {limit}"
        query += ";"

        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.set_index('timestamp', inplace=True)
        return df


def get_latest_candle_date(symbol: str) -> Optional[str]:
    """Returns the latest candle date stored for a symbol, or None if no data."""
    try:
        import duckdb_store
        dt = duckdb_store.get_latest_candle_date(symbol)
        if dt:
            return dt
    except Exception:
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_candles WHERE trading_symbol = ?;", (symbol.upper(),))
        res = cursor.fetchone()
        return res[0] if res and res[0] else None


def get_db_stats() -> Dict[str, Any]:
    """Returns overview statistics of stored database records."""
    try:
        import duckdb_store
        stats = duckdb_store.get_db_stats()
        if stats.get("total_instruments", 0) > 0 or stats.get("total_candles", 0) > 0:
            return stats
    except Exception:
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM instruments;")
        inst_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT trading_symbol), COUNT(*) FROM daily_candles;")
        candles_res = cursor.fetchone()
        symbols_with_data = candles_res[0] or 0
        total_candles = candles_res[1] or 0

        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_candles;")
        date_range = cursor.fetchone()

        return {
            "total_instruments": inst_count,
            "symbols_with_candles": symbols_with_data,
            "total_candles": total_candles,
            "earliest_date": date_range[0],
            "latest_date": date_range[1]
        }
