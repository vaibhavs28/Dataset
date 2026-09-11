import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

import config
import database
import instruments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upstox_parquet_updater")

PARQUET_DIR = config.DATA_DIR / "1min"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_BY_SYMBOL_DIR = config.DATA_DIR / "by_symbol"
PARQUET_BY_SYMBOL_DIR.mkdir(parents=True, exist_ok=True)

# Exact PyArrow schema matching train-00000.parquet to train-00007.parquet
PARQUET_SCHEMA = pa.schema([
    ("symbol", pa.large_string()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float32()),
    ("high", pa.float32()),
    ("low", pa.float32()),
    ("close", pa.float32()),
    ("volume", pa.int64()),
    ("oi", pa.int64())
])

# Maximum size per train-*.parquet shard before rolling over to the next shard (1.60 GB)
MAX_SHARD_BYTES = int(1.60 * 1024 * 1024 * 1024)


import threading
import collections

# Global persistent session with Keep-Alive connection pooling for massive speedup
_http_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=1)
_http_session.mount("https://", _adapter)
_http_session.mount("http://", _adapter)


class UpstoxRateLimiter:
    """
    Client-side rate limiter enforcing Upstox API v2 compliance:
    - High performance rate: 20 requests / second (within the 50/sec limit)
    - Per-minute rate: 480 requests / minute (within the 500/min limit)
    - Thread-safe non-blocking coordination across concurrent worker threads:
      worker threads schedule their time slot under a lightweight lock and sleep outside the lock.
    """
    def __init__(self, max_per_sec: float = 20.0, max_per_min: int = 480):
        self.min_interval = 1.0 / max(0.1, max_per_sec)
        self.max_per_min = max_per_min
        self.call_timestamps = collections.deque()
        self.next_allowed_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        sleep_sec = 0.0
        with self._lock:
            now = time.time()
            while self.call_timestamps and now - self.call_timestamps[0] >= 60.0:
                self.call_timestamps.popleft()

            target_time = max(now, self.next_allowed_time)
            if len(self.call_timestamps) >= self.max_per_min:
                oldest = self.call_timestamps[0]
                target_time = max(target_time, oldest + 60.05)

            sleep_sec = target_time - now
            scheduled_time = target_time
            self.next_allowed_time = scheduled_time + self.min_interval
            self.call_timestamps.append(scheduled_time)

        if sleep_sec > 0:
            time.sleep(sleep_sec)


# Global rate limiter instance (thread-safe, non-blocking, high throughput)
_global_rate_limiter = UpstoxRateLimiter(max_per_sec=20.0, max_per_min=480)


def get_symbol_parquet_latest_timestamp(symbol: str) -> Optional[datetime]:
    """
    Finds the latest timestamp for a symbol in the local parquet dataset (UTC datetime).
    """
    sym = symbol.upper().strip()
    try:
        dataset = ds.dataset(str(PARQUET_DIR), format="parquet")
        filter_expr = (ds.field("symbol") == sym)
        scanner = dataset.scanner(filter=filter_expr, columns=["timestamp"])
        tbl = scanner.to_table()
        if tbl.num_rows == 0:
            return None
        ts_series = tbl["timestamp"].to_pandas()
        max_ts = ts_series.max()
        if pd.isna(max_ts):
            return None
        return max_ts.to_pydatetime()
    except Exception as e:
        logger.error(f"Error checking latest timestamp for {sym} in parquet: {e}")
        return None


def fetch_upstox_1min_chunk(
    instrument_key: str,
    from_date: str,
    to_date: str,
    token: Optional[str] = None,
    rate_limiter: Optional[UpstoxRateLimiter] = None
) -> List[list]:
    """
    Calls Upstox Historical Candle API for 1-minute interval data.
    URL: /v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}
    Upstox allows a max retrieval range of 1 month per 1-minute request.
    Handles rate limiting, HTTP 429 retries with exponential backoff.
    """
    token = token or config.UPSTOX_ACCESS_TOKEN
    limiter = rate_limiter or _global_rate_limiter

    import urllib.parse
    quoted_key = urllib.parse.quote(instrument_key)
    url = f"{config.UPSTOX_BASE_URL}/historical-candle/{quoted_key}/1minute/{to_date}/{from_date}"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    max_retries = 4
    for attempt in range(max_retries):
        limiter.wait()
        try:
            response = _http_session.get(url, headers=headers, timeout=20)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = 2.0 * (2 ** attempt)
                try:
                    h_val = response.headers.get("Retry-After")
                    if h_val:
                        retry_after = max(float(h_val), retry_after)
                except Exception:
                    pass
                logger.warning(f"Upstox 429 Too Many Requests for {instrument_key}. Retrying in {retry_after:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_after)
                continue

            if response.status_code in (401, 403):
                resp_text = response.text
                # If CloudFront/AWS WAF policy restriction (e.g. "not allowed by policy"):
                if "not allowed by policy" in resp_text or "WAF" in resp_text:
                    logger.warning(f"Upstox policy restriction on {instrument_key} ({resp_text.strip()}). Backing off and retrying...")
                    time.sleep(2.5 * (attempt + 1))
                    if "Authorization" in headers:
                        del headers["Authorization"]
                    continue

                # Upstox Historical Candle API v2 is public! If an invalid or expired token was passed,
                # immediately retry without Authorization header.
                if "Authorization" in headers:
                    logger.info(f"Retrying {instrument_key} historical candles without Bearer token...")
                    del headers["Authorization"]
                    time.sleep(1.0)
                    continue

                # Check if genuinely token expired
                if any(kw in resp_text.lower() for kw in ("invalid_token", "token_expired", "token expired", "invalid token")):
                    err_msg = f"Upstox Authentication Error ({response.status_code}): {resp_text}"
                    logger.error(err_msg)
                    raise PermissionError(err_msg)
                else:
                    logger.warning(f"Upstox returned {response.status_code} for {instrument_key}: {resp_text.strip()}")
                    return []

            if response.status_code != 200:
                logger.warning(f"Upstox API returned status {response.status_code} for {instrument_key}: {response.text}")
                return []

            data = response.json()
            if data.get("status") != "success" or not data.get("data", {}).get("candles"):
                return []

            return data["data"]["candles"]

        except requests.RequestException as req_err:
            logger.warning(f"Request error fetching {instrument_key} ({from_date} to {to_date}): {req_err}")
            if attempt == max_retries - 1:
                return []
            time.sleep(1.5 * (attempt + 1))

    return []


def fetch_upstox_1min_intraday(
    instrument_key: str,
    token: Optional[str] = None,
    rate_limiter: Optional[UpstoxRateLimiter] = None
) -> List[list]:
    """
    Fetches today's live/intraday 1-minute candles from Upstox.
    URL: /v2/historical-candle/intraday/{instrument_key}/1minute
    """
    token = token or config.UPSTOX_ACCESS_TOKEN
    limiter = rate_limiter or _global_rate_limiter

    import urllib.parse
    quoted_key = urllib.parse.quote(instrument_key)
    url = f"{config.UPSTOX_BASE_URL}/historical-candle/intraday/{quoted_key}/1minute"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    limiter.wait()
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and data.get("data", {}).get("candles"):
                return data["data"]["candles"]
        elif response.status_code in (401, 403) and "Authorization" in headers:
            del headers["Authorization"]
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data", {}).get("candles"):
                    return data["data"]["candles"]
    except Exception as e:
        logger.warning(f"Failed to fetch intraday 1min candles for {instrument_key}: {e}")

    return []


def generate_date_chunks(start_dt: datetime, end_dt: datetime, chunk_days: int = 28) -> List[Tuple[str, str]]:
    """
    Splits date range [start_dt, end_dt] into chunks of at most chunk_days (<= 30 days per Upstox 1-min API).
    Returns list of (from_date_str, to_date_str) in 'YYYY-MM-DD' format.
    """
    chunks = []
    curr = start_dt
    while curr < end_dt:
        chunk_end = min(curr + timedelta(days=chunk_days), end_dt)
        from_str = curr.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")
        chunks.append((from_str, to_str))
        curr = chunk_end + timedelta(days=1)
    return chunks


def parse_upstox_candles_to_dataframe(symbol: str, raw_candles: List[list]) -> pd.DataFrame:
    """
    Parses Upstox candle lists [timestamp_str, open, high, low, close, volume, oi] into
    a DataFrame adhering to the target Parquet schema with UTC timestamp using vectorized pandas.
    """
    if not raw_candles:
        return pd.DataFrame()

    sym = symbol.upper().strip()
    df = pd.DataFrame(raw_candles)
    if df.shape[1] < 6:
        return pd.DataFrame()
    if df.shape[1] == 6:
        df[6] = 0
    df = df.iloc[:, :7]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    df["symbol"] = sym

    # Vectorized UTC timestamp parsing
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["close"] = df["close"].astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype("int64")

    # Reorder columns to target schema
    cols_order = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "oi"]
    df = df[cols_order]

    df.sort_values(by="timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    return df


def get_active_train_shard_path(min_shard_idx: int = 0) -> Path:
    """
    Returns the path to the current active train-NNNNN.parquet shard file starting from train-00000.parquet.
    If the active shard file size >= MAX_SHARD_BYTES (1.60 GB), rolls over to the next shard (e.g. train-00001.parquet).
    """
    existing_shards = sorted(list(PARQUET_DIR.glob("train-*.parquet")))
    update_shard_nums = []
    for p in existing_shards:
        try:
            num_part = p.stem.split("train-")[-1]
            num = int(num_part)
            if num >= min_shard_idx:
                update_shard_nums.append(num)
        except (ValueError, IndexError):
            continue

    if not update_shard_nums:
        active_num = min_shard_idx
    else:
        active_num = max(update_shard_nums)

    active_path = PARQUET_DIR / f"train-{active_num:05d}.parquet"
    if active_path.exists() and active_path.stat().st_size >= MAX_SHARD_BYTES:
        active_num += 1
        active_path = PARQUET_DIR / f"train-{active_num:05d}.parquet"

    return active_path


def append_dataframe_to_train_shard(df: pd.DataFrame) -> Optional[Path]:
    """
    Appends DataFrame to the active train-NNNNN.parquet shard file in data/1min/ conforming strictly to PARQUET_SCHEMA.
    Rolls over to train-00009.parquet, train-00010.parquet etc. when the file reaches 1.60 GB.
    Performs atomic file writing to prevent corruption.
    """
    if df is None or df.empty:
        return None

    # Cast columns strictly to PARQUET_SCHEMA types
    clean_df = df.copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str)
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
    clean_df["open"] = clean_df["open"].astype("float32")
    clean_df["high"] = clean_df["high"].astype("float32")
    clean_df["low"] = clean_df["low"].astype("float32")
    clean_df["close"] = clean_df["close"].astype("float32")
    clean_df["volume"] = clean_df["volume"].astype("int64")
    clean_df["oi"] = clean_df["oi"].astype("int64")

    clean_df = clean_df[["symbol", "timestamp", "open", "high", "low", "close", "volume", "oi"]]
    new_table = pa.Table.from_pandas(clean_df, schema=PARQUET_SCHEMA, preserve_index=False)

    active_path = get_active_train_shard_path()
    tmp_path = active_path.with_suffix(".parquet.tmp")

    if active_path.exists():
        try:
            existing_table = pq.read_table(str(active_path))
            combined_table = pa.concat_tables([existing_table, new_table], promote_options="default")
        except Exception as e:
            logger.error(f"Error reading existing shard {active_path.name}: {e}. Writing new table.")
            combined_table = new_table
    else:
        combined_table = new_table

    pq.write_table(combined_table, str(tmp_path), compression="snappy")
    tmp_path.replace(active_path)

    new_size_mb = active_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(df):,} bars to {active_path.name} (Total rows: {combined_table.num_rows:,}, Size: {new_size_mb:.2f} MB / 1,600 MB)")

    # Also update dedicated single-symbol partition for sub-10ms queries
    try:
        for sym_name, sym_df in df.groupby("symbol"):
            append_bars_to_symbol_file(str(sym_name), sym_df)
    except Exception as e:
        logger.warning(f"Error syncing to by_symbol partitions: {e}")

    return active_path


def append_bars_to_symbol_file(symbol: str, df: pd.DataFrame) -> Optional[Path]:
    """Appends bars directly to the dedicated single-symbol parquet file for instant retrieval."""
    if df is None or df.empty:
        return None
    safe_sym = symbol.replace("/", "_").replace("\\", "_").strip()
    target_file = PARQUET_BY_SYMBOL_DIR / f"{safe_sym}.parquet"

    clean_df = df.copy()
    clean_df["symbol"] = clean_df["symbol"].astype(str)
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
    clean_df["open"] = clean_df["open"].astype("float32")
    clean_df["high"] = clean_df["high"].astype("float32")
    clean_df["low"] = clean_df["low"].astype("float32")
    clean_df["close"] = clean_df["close"].astype("float32")
    clean_df["volume"] = clean_df["volume"].astype("int64")
    clean_df["oi"] = clean_df["oi"].astype("int64")
    clean_df = clean_df[["symbol", "timestamp", "open", "high", "low", "close", "volume", "oi"]]

    new_table = pa.Table.from_pandas(clean_df, schema=PARQUET_SCHEMA, preserve_index=False)

    if target_file.exists():
        try:
            existing = pq.read_table(target_file)
            combined = pa.concat_tables([existing, new_table])
            sorted_idx = pc.sort_indices(combined["timestamp"])
            combined = combined.take(sorted_idx)
            pq.write_table(combined, target_file, compression="snappy")
            return target_file
        except Exception as e:
            logger.warning(f"Error merging with existing symbol file {target_file.name}: {e}")

    pq.write_table(new_table, target_file, compression="snappy")
    return target_file


def update_symbol_parquet_till_date(
    symbol: str,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    save_to_shard: bool = True
) -> Dict[str, any]:
    """
    Fetches missing 1-minute historical candles from Upstox for a symbol from its
    latest timestamp in data/1min/ up to today.
    If save_to_shard=True: appends immediately to the active train shard and updates SQLite cache.
    If save_to_shard=False: returns the new DataFrame in res['df'] for batch accumulation.
    """
    sym = symbol.upper().strip()
    inst_key = instruments.resolve_instrument_key(sym)
    if not inst_key:
        inst_key = f"NSE_EQ|{sym}"

    # 1. Determine starting timestamp
    latest_ts_utc = get_symbol_parquet_latest_timestamp(sym)
    now_utc = datetime.now(tz=pd.Timestamp.utcnow().tzinfo)

    if latest_ts_utc is not None:
        # Start from next minute
        start_dt = latest_ts_utc + timedelta(minutes=1)
    else:
        # If stock not in parquet yet, fetch from 2020-01-01 to current date per user instruction
        from datetime import timezone
        start_dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    if start_dt >= now_utc:
        logger.info(f"{sym} is already up to date ({latest_ts_utc}).")
        return {"status": "up_to_date", "bars_added": 0, "latest_date": str(latest_ts_utc)}

    # Convert to naive UTC datetimes for chunking (include +1 day so latest session is requested)
    start_dt_naive = start_dt.replace(tzinfo=None)
    now_dt_naive = (now_utc + timedelta(days=1)).replace(tzinfo=None)

    chunks = generate_date_chunks(start_dt_naive, now_dt_naive, chunk_days=28)
    if not chunks:
        return {"status": "up_to_date", "bars_added": 0, "latest_date": str(latest_ts_utc)}

    logger.info(f"Syncing {sym} from {chunks[0][0]} to {chunks[-1][1]} across {len(chunks)} chunks...")

    all_raw_candles = []
    total_chunks = len(chunks)

    try:
        for idx, (f_date, t_date) in enumerate(chunks):
            if progress_callback:
                progress_callback(f"Fetching {sym} [{idx + 1}/{total_chunks}]: {f_date} to {t_date}", idx + 1, total_chunks)
            
            candles = fetch_upstox_1min_chunk(inst_key, f_date, t_date, token=token)
            if candles:
                all_raw_candles.extend(candles)

        # Also attempt today's intraday chunk
        intraday_candles = fetch_upstox_1min_intraday(inst_key, token=token)
        if intraday_candles:
            all_raw_candles.extend(intraday_candles)
    except PermissionError as auth_err:
        return {"status": "auth_error", "symbol": sym, "error": str(auth_err), "bars_added": 0, "latest_date": str(latest_ts_utc)}

    if not all_raw_candles:
        logger.info(f"No new candles returned from Upstox for {sym}.")
        return {"status": "no_data", "bars_added": 0, "latest_date": str(latest_ts_utc)}

    # Parse and build DataFrame
    df_new = parse_upstox_candles_to_dataframe(sym, all_raw_candles)
    if df_new.empty:
        return {"status": "no_data", "bars_added": 0, "latest_date": str(latest_ts_utc)}

    # Filter strictly newer than latest_ts_utc
    if latest_ts_utc is not None:
        df_new = df_new[df_new["timestamp"] > latest_ts_utc]

    if df_new.empty:
        return {"status": "up_to_date", "bars_added": 0, "latest_date": str(latest_ts_utc)}

    new_latest_date = df_new["timestamp"].max()

    if save_to_shard:
        out_file = append_dataframe_to_train_shard(df_new)
        try:
            from parquet_loader import SYMBOLS_CACHE_FILE, resample_1min_to_75min, import_symbol_to_database
            if SYMBOLS_CACHE_FILE.exists():
                try:
                    SYMBOLS_CACHE_FILE.unlink()
                except Exception:
                    pass
            # Immediately resample to 75m and upsert to database
            intra_75m = resample_1min_to_75min(df_new)
            if not intra_75m.empty:
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

            import_symbol_to_database(sym)
        except Exception as db_err:
            logger.warning(f"Could not refresh SQLite cache for {sym}: {db_err}")

        return {
            "status": "success",
            "symbol": sym,
            "bars_added": len(df_new),
            "file_created": str(out_file.name) if out_file else None,
            "latest_date": str(new_latest_date)
        }
    else:
        return {
            "status": "success",
            "symbol": sym,
            "bars_added": len(df_new),
            "df": df_new,
            "latest_date": str(new_latest_date)
        }


def sync_symbols_parquet_batch(
    symbols: List[str],
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, dict], None]] = None
) -> dict:
    """
    Syncs 1-minute historical data till date for a list of symbols with rate limiting,
    accumulating new candles and writing them into the active train shard in one operation.
    """
    total = len(symbols)
    synced_symbols = 0
    total_bars_added = 0
    results = []
    batch_frames = []
    updated_syms = []

    for idx, sym in enumerate(symbols):
        try:
            res = update_symbol_parquet_till_date(
                sym,
                token=token,
                progress_callback=lambda msg, c, t: progress_callback(idx + 1, total, f"{sym}: {msg}", {}) if progress_callback else None,
                save_to_shard=False
            )
            bars = res.get("bars_added", 0)
            total_bars_added += bars
            synced_symbols += 1
            if res.get("status") == "success" and "df" in res and not res["df"].empty:
                batch_frames.append(res["df"])
                updated_syms.append(sym)
            results.append(res)
            if progress_callback:
                progress_callback(idx + 1, total, sym, res)
        except Exception as e:
            logger.error(f"Error syncing {sym}: {e}")
            results.append({"status": "error", "symbol": sym, "error": str(e)})

    # Flush all candles into active train shard
    shard_file = None
    if batch_frames:
        comb_df = pd.concat(batch_frames, ignore_index=True)
        shard_file = append_dataframe_to_train_shard(comb_df)
        try:
            from parquet_loader import import_batch_to_database
            import_batch_to_database(updated_syms)
        except Exception as e:
            logger.warning(f"Error refreshing database cache: {e}")

    return {
        "total_symbols": total,
        "synced_symbols": synced_symbols,
        "total_bars_added": total_bars_added,
        "shard_file": shard_file.name if shard_file else None,
        "results": results
    }



SYNC_STATE_FILE = config.DATA_DIR / "sync_state.json"


def load_sync_state() -> dict:
    """Loads previous sync state from disk if it exists."""
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load sync state from {SYNC_STATE_FILE}: {e}")
    return {
        "completed_symbols": {},
        "failed_symbols": {},
        "total_bars_added": 0,
        "last_batch_index": 0,
        "last_updated": None
    }


def save_sync_state(state: dict):
    """Atomically saves sync state to disk."""
    state["last_updated"] = datetime.now().isoformat()
    tmp_file = SYNC_STATE_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(SYNC_STATE_FILE)
    except Exception as e:
        logger.error(f"Failed to save sync state to {SYNC_STATE_FILE}: {e}")


def reset_sync_state():
    """Resets the sync checkpoint file."""
    if SYNC_STATE_FILE.exists():
        try:
            SYNC_STATE_FILE.unlink()
            logger.info("Sync state reset successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sync state file: {e}")
            return False
    return True


def sync_database_in_batches(
    symbols: Optional[List[str]] = None,
    batch_size: int = 50,
    pause_seconds: int = 10,
    token: Optional[str] = None,
    resume: bool = True,
    progress_callback: Optional[Callable[[dict], None]] = None
) -> dict:
    """
    Syncs 1-minute historical data for the entire database in batches of `batch_size`
    stocks (default: 50), with a `pause_seconds` pause (default: 10s) between batches.
    Maintains checkpoint in data/sync_state.json for fault tolerance and resume capability.
    """
    import parquet_loader
    if symbols is None or len(symbols) == 0:
        symbols = parquet_loader.get_parquet_symbols()

    if not symbols:
        return {"status": "error", "message": "No symbols found in Parquet dataset."}

    state = load_sync_state() if resume else {
        "completed_symbols": {},
        "failed_symbols": {},
        "total_bars_added": 0,
        "last_batch_index": 0,
        "last_updated": None
    }

    total_symbols = len(symbols)
    # Split into batches of batch_size
    batches = [symbols[i:i + batch_size] for i in range(0, total_symbols, batch_size)]
    total_batches = len(batches)

    completed_symbols = state.get("completed_symbols", {})
    failed_symbols = state.get("failed_symbols", {})
    total_bars = state.get("total_bars_added", 0)

    logger.info(
        f"Starting batch database sync: {total_symbols} symbols in {total_batches} batches of {batch_size} "
        f"(Pause: {pause_seconds}s). Resume: {resume} ({len(completed_symbols)} already completed)."
    )

    stopped_due_to_auth = False
    auth_error_detail = None

    for b_idx, batch in enumerate(batches):
        batch_num = b_idx + 1
        logger.info(f"=== Starting Batch {batch_num}/{total_batches} ({len(batch)} symbols) ===")

        if progress_callback:
            progress_callback({
                "type": "batch_start",
                "batch": batch_num,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "symbols": batch
            })

        batch_frames = []
        batch_updated_symbols = []

        def _flush_batch_to_shard():
            nonlocal batch_frames, batch_updated_symbols
            if batch_frames:
                try:
                    combined_batch_df = pd.concat(batch_frames, ignore_index=True)
                    shard_file = append_dataframe_to_train_shard(combined_batch_df)
                    from parquet_loader import import_batch_to_database
                    import_batch_to_database(batch_updated_symbols)
                    logger.info(f"Batch {batch_num} flush: Saved {len(combined_batch_df):,} bars across {len(batch_updated_symbols)} symbols to {shard_file.name}.")
                except Exception as ex:
                    logger.error(f"Error flushing batch {batch_num} to shard: {ex}")
                finally:
                    batch_frames = []
                    batch_updated_symbols = []

        for s_idx, sym in enumerate(batch):
            global_idx = b_idx * batch_size + s_idx + 1

            if resume and sym in completed_symbols:
                sym_info = completed_symbols[sym]
                if sym_info.get("status") in ("success", "skipped_up_to_date"):
                    if progress_callback:
                        progress_callback({
                            "type": "symbol_skipped",
                            "symbol": sym,
                            "global_index": global_idx,
                            "total_symbols": total_symbols,
                            "batch": batch_num,
                            "total_batches": total_batches,
                            "batch_index": s_idx + 1,
                            "batch_size": len(batch),
                            "info": sym_info
                        })
                    continue

            # Execute update for symbol (accumulating in batch)
            def _sub_cb(msg, cur, tot):
                if progress_callback:
                    progress_callback({
                        "type": "symbol_step",
                        "symbol": sym,
                        "message": msg,
                        "global_index": global_idx,
                        "total_symbols": total_symbols,
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "batch_index": s_idx + 1,
                        "batch_size": len(batch)
                    })

            res = update_symbol_parquet_till_date(sym, token=token, progress_callback=_sub_cb, save_to_shard=False)

            if res.get("status") == "auth_error":
                stopped_due_to_auth = True
                auth_error_detail = res.get("error")
                logger.error(f"Stopping batch sync due to authentication error on {sym}: {auth_error_detail}")
                _flush_batch_to_shard()
                save_sync_state(state)
                if progress_callback:
                    progress_callback({
                        "type": "auth_error",
                        "symbol": sym,
                        "global_index": global_idx,
                        "total_symbols": total_symbols,
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "error": auth_error_detail
                    })
                return {
                    "status": "auth_error",
                    "error": auth_error_detail,
                    "symbol": sym,
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "completed_count": len(completed_symbols),
                    "total_symbols": total_symbols,
                    "total_bars_added": total_bars
                }

            if res.get("status") in ("success", "skipped_up_to_date"):
                if res.get("status") == "success" and "df" in res and not res["df"].empty:
                    batch_frames.append(res["df"])
                    batch_updated_symbols.append(sym)

                completed_symbols[sym] = {
                    "status": res.get("status"),
                    "bars_added": res.get("bars_added", 0),
                    "latest_date": res.get("latest_date"),
                    "updated_at": datetime.now().isoformat()
                }
                total_bars += res.get("bars_added", 0)
                failed_symbols.pop(sym, None)
            else:
                failed_symbols[sym] = {
                    "status": "error",
                    "error": res.get("error", "Unknown error"),
                    "updated_at": datetime.now().isoformat()
                }

            state["completed_symbols"] = completed_symbols
            state["failed_symbols"] = failed_symbols
            state["total_bars_added"] = total_bars
            state["last_batch_index"] = batch_num
            save_sync_state(state)

            if progress_callback:
                progress_callback({
                    "type": "symbol_completed",
                    "symbol": sym,
                    "global_index": global_idx,
                    "total_symbols": total_symbols,
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "batch_index": s_idx + 1,
                    "batch_size": len(batch),
                    "result": res,
                    "total_bars_added": total_bars,
                    "completed_count": len(completed_symbols)
                })

        # End of batch: flush accumulated batch DataFrames into the active train shard
        _flush_batch_to_shard()

        # Check active shard size
        active_shard = get_active_train_shard_path()
        shard_size_mb = active_shard.stat().st_size / (1024 * 1024) if active_shard.exists() else 0.0

        # Pause if more batches remain
        if batch_num < total_batches and not stopped_due_to_auth:
            logger.info(f"Batch {batch_num}/{total_batches} finished. Active shard: {active_shard.name} ({shard_size_mb:.1f} MB). Pausing {pause_seconds}s before next batch...")
            if progress_callback:
                progress_callback({
                    "type": "batch_pause_start",
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "pause_seconds": pause_seconds,
                    "active_shard": active_shard.name,
                    "shard_size_mb": shard_size_mb
                })
            for remaining in range(pause_seconds, 0, -1):
                if progress_callback:
                    progress_callback({
                        "type": "batch_pause_tick",
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "remaining_seconds": remaining,
                        "pause_seconds": pause_seconds,
                        "active_shard": active_shard.name,
                        "shard_size_mb": shard_size_mb
                    })
                time.sleep(1)

            if progress_callback:
                progress_callback({
                    "type": "batch_pause_end",
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "active_shard": active_shard.name,
                    "shard_size_mb": shard_size_mb
                })

    return {
        "status": "completed",
        "total_symbols": total_symbols,
        "total_batches": total_batches,
        "completed_count": len(completed_symbols),
        "failed_count": len(failed_symbols),
        "total_bars_added": total_bars
    }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update 1-minute Parquet historical dataset from Upstox API V2 till date.")
    parser.add_argument("--symbol", type=str, help="Specific symbol to sync (e.g. RELIANCE)")
    parser.add_argument("--nifty50", action="store_true", help="Sync all 50 Nifty 50 constituents")
    parser.add_argument("--all", action="store_true", help="Sync entire database (all 2,500+ symbols) in 50-stock batches with 10s pause")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for database sync (default: 50)")
    parser.add_argument("--pause", type=int, default=10, help="Pause duration between batches in seconds (default: 10)")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh sync instead of resuming from previous state")
    parser.add_argument("--reset-state", action="store_true", help="Reset sync checkpoint state")
    parser.add_argument("--token", type=str, default=None, help="Optional Upstox access token override")
    args = parser.parse_args()

    if args.reset_state:
        reset_sync_state()
        print("Sync state checkpoint reset.")

    if args.symbol:
        sym = args.symbol.upper()
        print(f"Updating 1-minute Parquet data for {sym}...")
        res = update_symbol_parquet_till_date(sym, token=args.token)
        print("Result:", res)
    elif args.nifty50:
        print(f"Updating 1-minute Parquet data for Nifty 50 ({len(config.NIFTY_50_SYMBOLS)} stocks)...")
        res = sync_symbols_parquet_batch(config.NIFTY_50_SYMBOLS, token=args.token)
        print(f"Sync complete: {res['synced_symbols']}/{res['total_symbols']} symbols, {res['total_bars_added']:,} bars added.")
    elif args.all:
        print(f"Starting Whole Database Sync (Batch size: {args.batch_size}, Pause: {args.pause}s, Resume: {not args.fresh})...")
        def _cli_cb(ev):
            t = ev.get("type")
            if t == "batch_start":
                print(f"\n[Batch {ev['batch']}/{ev['total_batches']}] Starting batch of {ev['batch_size']} symbols...")
            elif t == "symbol_completed":
                r = ev.get("result", {})
                print(f"  [{ev['global_index']}/{ev['total_symbols']}] {ev['symbol']}: {r.get('status')} (+{r.get('bars_added', 0)} bars) | Overall: {ev['completed_count']} completed")
            elif t == "symbol_skipped":
                print(f"  [{ev['global_index']}/{ev['total_symbols']}] {ev['symbol']}: Already synced (Skipped)")
            elif t == "batch_pause_start":
                print(f"[Pause] Pausing {ev['pause_seconds']}s before next batch...")
            elif t == "auth_error":
                print(f"\n[AUTH ERROR] {ev.get('error')}. Progress saved at symbol {ev.get('symbol')}.")
        res = sync_database_in_batches(
            batch_size=args.batch_size,
            pause_seconds=args.pause,
            token=args.token,
            resume=not args.fresh,
            progress_callback=_cli_cb
        )
        print("\nWhole Database Sync Summary:", res)
    else:
        print("Usage:")
        print("  python3 upstox_parquet_updater.py --symbol RELIANCE")
        print("  python3 upstox_parquet_updater.py --nifty50")
        print("  python3 upstox_parquet_updater.py --all --batch-size 50 --pause 10")


