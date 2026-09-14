"""
crypto_store.py
---------------
Dedicated DuckDB storage and manager for Cryptocurrency data (BTCUSD and ETHUSD).
Maintains a separate database file (data/crypto_market.duckdb) isolated from equity data.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Dedicated DuckDB database file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(CRYPTO_DATA_DIR, exist_ok=True)
CRYPTO_DB_PATH = os.path.join(CRYPTO_DATA_DIR, "crypto_market.duckdb")

_lock = threading.Lock()
_connection_pool: Optional[duckdb.DuckDBPyConnection] = None

SUPPORTED_CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD"]
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]


def get_crypto_connection() -> duckdb.DuckDBPyConnection:
    """Returns a thread-safe connection to the separate crypto DuckDB database."""
    global _connection_pool
    with _lock:
        if _connection_pool is None:
            _connection_pool = duckdb.connect(CRYPTO_DB_PATH, read_only=False)
            _init_crypto_schema(_connection_pool)
        return _connection_pool


def _init_crypto_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes tables for crypto candles and live tickers."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS crypto_candles (
        symbol VARCHAR NOT NULL,
        timeframe VARCHAR NOT NULL,
        timestamp BIGINT NOT NULL,
        date_str VARCHAR NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume DOUBLE NOT NULL,
        PRIMARY KEY (symbol, timeframe, timestamp)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS crypto_tickers (
        symbol VARCHAR PRIMARY KEY,
        mark_price DOUBLE,
        last_price DOUBLE,
        high_24h DOUBLE,
        low_24h DOUBLE,
        volume_24h DOUBLE,
        change_24h DOUBLE,
        timestamp BIGINT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)


def upsert_crypto_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """
    Upserts candle data for a given crypto symbol and timeframe into crypto_candles table.
    Expected DataFrame columns: ['open', 'high', 'low', 'close', 'volume'] with DatetimeIndex
    or timestamp / time column.
    """
    if df is None or df.empty:
        return 0

    symbol = symbol.upper().strip()
    timeframe = timeframe.lower().strip()

    df_prep = df.copy()
    if isinstance(df_prep.index, pd.DatetimeIndex):
        ts_series = pd.Series(df_prep.index.astype("int64") // 10**9)
        date_str_series = pd.Series(df_prep.index.strftime("%Y-%m-%d %H:%M:%S"))
    elif "timestamp" in df_prep.columns:
        ts_series = df_prep["timestamp"].astype("int64")
        date_str_series = pd.to_datetime(ts_series, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    elif "time" in df_prep.columns:
        ts_series = df_prep["time"].astype("int64")
        date_str_series = pd.to_datetime(ts_series, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        logger.error(f"Cannot find timestamp/index in crypto DataFrame for {symbol} {timeframe}")
        return 0

    rows_to_insert = []
    for i in range(len(df_prep)):
        ts = int(ts_series.iloc[i])
        d_str = str(date_str_series.iloc[i])
        o = float(df_prep["open"].iloc[i])
        h = float(df_prep["high"].iloc[i])
        l = float(df_prep["low"].iloc[i])
        c = float(df_prep["close"].iloc[i])
        v = float(df_prep["volume"].iloc[i]) if "volume" in df_prep.columns else 0.0
        rows_to_insert.append((symbol, timeframe, ts, d_str, o, h, l, c, v))

    if not rows_to_insert:
        return 0

    temp_df = pd.DataFrame(rows_to_insert, columns=[
        "symbol", "timeframe", "timestamp", "date_str", "open", "high", "low", "close", "volume"
    ])

    conn = get_crypto_connection()
    with _lock:
        conn.register("temp_crypto_incoming", temp_df)
        conn.execute("""
        INSERT OR REPLACE INTO crypto_candles 
        SELECT symbol, timeframe, timestamp, date_str, open, high, low, close, volume 
        FROM temp_crypto_incoming;
        """)
        conn.unregister("temp_crypto_incoming")

    return len(rows_to_insert)


def get_crypto_candles(
    symbol: str,
    timeframe: str = "1d",
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Retrieves crypto candles sorted in chronological ascending order.
    Returns DataFrame with DatetimeIndex and ['open', 'high', 'low', 'close', 'volume'].
    """
    symbol = symbol.upper().strip()
    timeframe = timeframe.lower().strip()
    conn = get_crypto_connection()

    query = "SELECT timestamp, date_str, open, high, low, close, volume FROM crypto_candles WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]

    if start_date:
        query += " AND date_str >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date_str <= ?"
        params.append(str(end_date))

    if limit:
        # Fetch latest limit rows in ascending order
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        query = f"SELECT * FROM ({query}) sub ORDER BY timestamp ASC"
    else:
        query += " ORDER BY timestamp ASC"

    with _lock:
        df = conn.execute(query, params).df()

    if df.empty:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("dt", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def upsert_crypto_ticker(ticker_data: Dict[str, Any]) -> None:
    """Saves or updates the live ticker snapshot for a crypto symbol."""
    if not ticker_data or "symbol" not in ticker_data:
        return

    symbol = str(ticker_data["symbol"]).upper().strip()
    mark_price = float(ticker_data.get("mark_price", 0.0) or 0.0)
    last_price = float(ticker_data.get("close", 0.0) or ticker_data.get("last_price", mark_price) or 0.0)
    high_24h = float(ticker_data.get("high", 0.0) or ticker_data.get("high_24h", last_price) or 0.0)
    low_24h = float(ticker_data.get("low", 0.0) or ticker_data.get("low_24h", last_price) or 0.0)
    volume_24h = float(ticker_data.get("volume", 0.0) or ticker_data.get("volume_24h", 0.0) or 0.0)
    change_24h = float(ticker_data.get("change_24h", 0.0) or 0.0)
    ts = int(ticker_data.get("timestamp", time.time()) or time.time())

    conn = get_crypto_connection()
    with _lock:
        conn.execute("""
        INSERT OR REPLACE INTO crypto_tickers (
            symbol, mark_price, last_price, high_24h, low_24h, volume_24h, change_24h, timestamp, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, [symbol, mark_price, last_price, high_24h, low_24h, volume_24h, change_24h, ts])


def get_crypto_ticker(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches the latest live ticker info for a given crypto symbol."""
    symbol = symbol.upper().strip()
    conn = get_crypto_connection()
    with _lock:
        res = conn.execute("""
        SELECT symbol, mark_price, last_price, high_24h, low_24h, volume_24h, change_24h, timestamp, updated_at 
        FROM crypto_tickers WHERE symbol = ?
        """, [symbol]).fetchone()

    if not res:
        return None

    return {
        "symbol": res[0],
        "mark_price": res[1],
        "last_price": res[2],
        "high_24h": res[3],
        "low_24h": res[4],
        "volume_24h": res[5],
        "change_24h": res[6],
        "timestamp": res[7],
        "updated_at": res[8]
    }


def get_all_crypto_tickers() -> Dict[str, Dict[str, Any]]:
    """Returns all current crypto tickers keyed by symbol."""
    conn = get_crypto_connection()
    with _lock:
        rows = conn.execute("""
        SELECT symbol, mark_price, last_price, high_24h, low_24h, volume_24h, change_24h, timestamp, updated_at 
        FROM crypto_tickers
        """).fetchall()

    tickers = {}
    for r in rows:
        tickers[r[0]] = {
            "symbol": r[0],
            "mark_price": r[1],
            "last_price": r[2],
            "high_24h": r[3],
            "low_24h": r[4],
            "volume_24h": r[5],
            "change_24h": r[6],
            "timestamp": r[7],
            "updated_at": r[8]
        }
    return tickers


def seed_crypto_initial_data_if_empty() -> None:
    """
    Seeds comprehensive initial historical candles and tickers for BTCUSD and ETHUSD
    if database is empty, so that charts, indicators, and screener work instantly out-of-the-box.
    """
    conn = get_crypto_connection()
    with _lock:
        count = conn.execute("SELECT COUNT(*) FROM crypto_candles").fetchone()[0]

    if count > 0:
        return

    logger.info("Seeding initial crypto market historical candles for BTCUSD and ETHUSD...")

    np.random.seed(42)
    # Generate Daily candles (365 days)
    end_dt = datetime.utcnow()
    dates_daily = [end_dt - timedelta(days=365 - i) for i in range(365)]

    # BTC daily series around 60,000 to 95,000
    btc_price = 62000.0
    btc_daily_rows = []
    for dt in dates_daily:
        change = np.random.normal(75.0, 1200.0)
        btc_price = max(25000.0, btc_price + change)
        high = btc_price + abs(np.random.normal(800.0, 400.0))
        low = btc_price - abs(np.random.normal(800.0, 400.0))
        open_p = btc_price + np.random.normal(0.0, 300.0)
        vol = float(np.random.randint(15000, 85000))
        ts = int(dt.timestamp())
        d_str = dt.strftime("%Y-%m-%d 00:00:00")
        btc_daily_rows.append(("BTCUSD", "1d", ts, d_str, open_p, high, low, btc_price, vol))

    # ETH daily series around 2,400 to 3,800
    eth_price = 2600.0
    eth_daily_rows = []
    for dt in dates_daily:
        change = np.random.normal(3.0, 60.0)
        eth_price = max(1200.0, eth_price + change)
        high = eth_price + abs(np.random.normal(40.0, 20.0))
        low = eth_price - abs(np.random.normal(40.0, 20.0))
        open_p = eth_price + np.random.normal(0.0, 15.0)
        vol = float(np.random.randint(80000, 350000))
        ts = int(dt.timestamp())
        d_str = dt.strftime("%Y-%m-%d 00:00:00")
        eth_daily_rows.append(("ETHUSD", "1d", ts, d_str, open_p, high, low, eth_price, vol))

    # Intraday 1h candles (last 240 hours = 10 days)
    hours = [end_dt - timedelta(hours=240 - i) for i in range(240)]
    btc_1h_rows = []
    eth_1h_rows = []
    b_curr = btc_price
    e_curr = eth_price

    for dt in hours:
        b_curr += np.random.normal(15.0, 250.0)
        b_h = b_curr + abs(np.random.normal(150.0, 80.0))
        b_l = b_curr - abs(np.random.normal(150.0, 80.0))
        b_o = b_curr + np.random.normal(0.0, 60.0)
        b_v = float(np.random.randint(800, 4500))
        ts = int(dt.timestamp())
        btc_1h_rows.append(("BTCUSD", "1h", ts, dt.strftime("%Y-%m-%d %H:%M:%S"), b_o, b_h, b_l, b_curr, b_v))

        e_curr += np.random.normal(0.8, 18.0)
        e_h = e_curr + abs(np.random.normal(12.0, 5.0))
        e_l = e_curr - abs(np.random.normal(12.0, 5.0))
        e_o = e_curr + np.random.normal(0.0, 4.0)
        e_v = float(np.random.randint(3000, 18000))
        eth_1h_rows.append(("ETHUSD", "1h", ts, dt.strftime("%Y-%m-%d %H:%M:%S"), e_o, e_h, e_l, e_curr, e_v))

    all_seed = btc_daily_rows + eth_daily_rows + btc_1h_rows + eth_1h_rows
    seed_df = pd.DataFrame(all_seed, columns=[
        "symbol", "timeframe", "timestamp", "date_str", "open", "high", "low", "close", "volume"
    ])

    with _lock:
        conn.register("seed_df_temp", seed_df)
        conn.execute("""
        INSERT OR REPLACE INTO crypto_candles 
        SELECT symbol, timeframe, timestamp, date_str, open, high, low, close, volume 
        FROM seed_df_temp;
        """)
        conn.unregister("seed_df_temp")

    # Initial tickers
    upsert_crypto_ticker({
        "symbol": "BTCUSD",
        "mark_price": round(btc_price, 2),
        "last_price": round(btc_price, 2),
        "high_24h": round(btc_price * 1.025, 2),
        "low_24h": round(btc_price * 0.978, 2),
        "volume_24h": 54200.0,
        "change_24h": 2.15,
        "timestamp": int(end_dt.timestamp())
    })

    upsert_crypto_ticker({
        "symbol": "ETHUSD",
        "mark_price": round(eth_price, 2),
        "last_price": round(eth_price, 2),
        "high_24h": round(eth_price * 1.032, 2),
        "low_24h": round(eth_price * 0.981, 2),
        "volume_24h": 195000.0,
        "change_24h": 3.42,
        "timestamp": int(end_dt.timestamp())
    })

    logger.info("✅ Crypto initial seed complete (BTCUSD and ETHUSD populated).")


# Auto-seed on load if database is empty
try:
    seed_crypto_initial_data_if_empty()
except Exception as e:
    logger.warning(f"Failed to auto-seed crypto data: {e}")
