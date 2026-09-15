import time
import logging
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Callable

import config
import database
import instruments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("downloader")


def fetch_upstox_daily_candles(
    instrument_key: str,
    from_date: str,
    to_date: str,
    token: Optional[str] = None
) -> List[dict]:
    """
    Calls Upstox Historical Candle API for daily candles.
    URL: /v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}
    Dates in format YYYY-MM-DD.
    """
    token = token or config.UPSTOX_ACCESS_TOKEN
    headers = {
        "Accept": "application/json"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    import urllib.parse
    quoted_key = urllib.parse.quote(instrument_key)
    url = f"{config.UPSTOX_BASE_URL}/historical-candle/{quoted_key}/day/{to_date}/{from_date}"
    
    try:
        from upstox_parquet_updater import _http_session
        response = _http_session.get(url, headers=headers, timeout=20)
    except Exception:
        response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        logger.warning(f"Upstox API returned {response.status_code} for {instrument_key}: {response.text}")
        return []

    data = response.json()
    if data.get("status") != "success" or not data.get("data", {}).get("candles"):
        return []

    raw_candles = data["data"]["candles"]
    # Upstox candle format: [timestamp, open, high, low, close, volume, open_interest]
    candles = []
    for c in raw_candles:
        # Extract YYYY-MM-DD from timestamp string
        dt_str = c[0].split("T")[0]
        candles.append({
            "instrument_key": instrument_key,
            "date": dt_str,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": int(c[5]) if c[5] is not None else 0,
            "open_interest": int(c[6]) if len(c) > 6 and c[6] is not None else 0
        })
    return candles


def sync_symbol_history(symbol: str, from_date: str = "2022-01-01", to_date: Optional[str] = None) -> int:
    """
    Downloads authentic historical daily candles for a given symbol from 1 Jan 2022 (or custom from_date)
    to the current date directly from Upstox, saving cleanly to SQLite without duplicates or spikes.
    """
    database.init_db()
    sym = symbol.upper().strip()
    inst_key = instruments.resolve_instrument_key(sym)
    if not inst_key:
        logger.warning(f"Could not resolve instrument key for symbol: {sym}")
        return 0

    today_str = to_date or datetime.today().strftime("%Y-%m-%d")
    current_today = datetime.today().strftime("%Y-%m-%d")

    # Fetch daily candles from Upstox starting from 2022-01-01
    candles = fetch_upstox_daily_candles(inst_key, from_date, today_str)
    if not candles and from_date != "2022-01-01":
        candles = fetch_upstox_daily_candles(inst_key, "2022-01-01", today_str)

    added_bars = 0
    if candles:
        # Add trading symbol to each record
        for c in candles:
            c["trading_symbol"] = sym
        database.upsert_candles(candles)
        added_bars = len(candles)

    # If syncing up to today, verify if today's candle was returned by Upstox daily EOD endpoint.
    # Note: Upstox /historical-candle/.../day only includes today's candle late at night after market settlement.
    # If today's candle is missing, automatically synthesize today's live daily candle from Upstox 1-min intraday feed.
    if today_str >= current_today:
        has_today = any(c.get("date") == current_today for c in candles) if candles else False
        if not has_today:
            db_latest = database.get_latest_candle_date(sym)
            if not db_latest or db_latest < current_today:
                if sync_live_market_candles(sym):
                    added_bars += 1

    if added_bars == 0 and not candles:
        logger.warning(f"No daily or intraday candles returned from Upstox for {sym} ({inst_key})")
        return 0

    # Clean daily candles saved; 75m candles are strictly resampled from 1-min data on demand
    logger.info(f"Successfully synced {added_bars} clean daily candles for {sym} from {from_date} to {today_str}")
    return added_bars


def sync_watchlist(symbols: List[str], progress_callback: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """
    Syncs candles for all symbols in the watchlist from 2022-01-01 to today.
    """
    database.init_db()
    total = len(symbols)
    synced = 0
    errors = []

    for idx, sym in enumerate(symbols):
        try:
            count = sync_symbol_history(sym, from_date="2022-01-01")
            synced += 1
            if progress_callback:
                progress_callback(idx + 1, total, sym)
            time.sleep(0.1)  # Respect rate limits
        except Exception as e:
            logger.error(f"Failed to sync {sym}: {e}")
            errors.append((sym, str(e)))

    return {"total": total, "synced": synced, "errors": errors}


def seed_demo_data(symbols: Optional[List[str]] = None, years: int = 5):
    """
    Downloads authentic real market data from 1 Jan 2022 to today from Upstox.
    Purges any previous synthetic demo data to guarantee zero chart spikes.
    """
    database.purge_all_demo_data()
    symbols = symbols or config.NIFTY_50_SYMBOLS[:20]
    logger.info(f"Downloading authentic Upstox historical data from 2022-01-01 for {len(symbols)} symbols...")

    total_synced = 0
    for sym in symbols:
        try:
            cnt = sync_symbol_history(sym, from_date="2022-01-01")
            total_synced += cnt
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error downloading {sym}: {e}")

    logger.info(f"Successfully downloaded authentic data for {len(symbols)} stocks. Total bars: {total_synced}.")
    return total_synced


def update_daily_market_data_one_click(
    symbols: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 6
) -> dict:
    """
    One-click daily market update.
    Scans stocks, identifies symbols missing today's candle,
    and syncs them up to today via Upstox daily EOD or live intraday synthesis.
    Threaded, fast, and robust!
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    database.init_db()
    today_str = datetime.today().strftime("%Y-%m-%d")

    if not symbols:
        # Get all symbols that have candle data in the database
        with database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT trading_symbol FROM daily_candles ORDER BY trading_symbol ASC;")
            symbols = [r[0] for r in cursor.fetchall()]

    if not symbols:
        symbols = config.NIFTY_50_SYMBOLS

    # Prioritize watchlist / Nifty 50 stocks first
    nifty_set = set(config.NIFTY_50_SYMBOLS)
    symbols = sorted(symbols, key=lambda s: (0 if s in nifty_set else 1, s))

    # Identify which symbols are already up-to-date vs which need updating
    to_update = []
    skipped_count = 0
    for sym in symbols:
        latest_date = database.get_latest_candle_date(sym)
        if latest_date and latest_date >= today_str:
            skipped_count += 1
        else:
            to_update.append((sym, latest_date))

    total_symbols = len(symbols)
    needed_count = len(to_update)
    updated_count = 0
    total_new_bars = 0
    errors = []

    if needed_count == 0:
        if progress_callback:
            progress_callback(total_symbols, total_symbols, "All up to date")
        return {
            "total_symbols": total_symbols,
            "updated_symbols": 0,
            "already_up_to_date": skipped_count,
            "new_bars_added": 0,
            "today_date": today_str,
            "errors": []
        }

    # Worker function for thread pool
    def _worker(item):
        sym, latest_date = item
        from_date = latest_date or "2022-01-01"
        cnt = sync_symbol_history(sym, from_date=from_date, to_date=today_str)
        return sym, cnt

    # Run with bounded thread pool (safe with UpstoxRateLimiter)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, item): item[0] for item in to_update}
        for future in as_completed(futures):
            sym = futures[future]
            completed += 1
            try:
                sym_name, cnt = future.result()
                if cnt > 0:
                    updated_count += 1
                    total_new_bars += cnt
                else:
                    latest = database.get_latest_candle_date(sym_name)
                    if latest and latest >= today_str:
                        updated_count += 1
            except Exception as e:
                logger.error(f"Error updating {sym}: {e}")
                errors.append((sym, str(e)))

            if progress_callback:
                progress_callback(completed, needed_count, sym)

    return {
        "total_symbols": total_symbols,
        "updated_symbols": updated_count,
        "already_up_to_date": skipped_count,
        "new_bars_added": total_new_bars,
        "today_date": today_str,
        "errors": errors
    }


def sync_live_market_candles(symbol: str) -> bool:
    """
    Fetches today's live intraday 1-minute bars from Upstox API,
    appends them to data/by_symbol/{symbol}.parquet,
    and upserts today's live forming daily candle into SQLite.
    Returns True if live candles were updated.
    """
    sym = symbol.upper().strip()
    inst_key = instruments.resolve_instrument_key(sym)
    if not inst_key:
        return False

    try:
        import upstox_parquet_updater
        raw_bars = upstox_parquet_updater.fetch_upstox_1min_intraday(inst_key)
        if not raw_bars:
            return False

        # 1. Update single-symbol Parquet file with today's live 1-min ticks
        df_today = upstox_parquet_updater.parse_upstox_candles_to_dataframe(sym, raw_bars)
        upstox_parquet_updater.append_bars_to_symbol_file(sym, df_today)

        # 2. Form today's live daily candle and upsert into SQLite
        first_bar = raw_bars[-1]
        latest_bar = raw_bars[0]
        today_date = first_bar[0].split("T")[0]

        daily_record = [{
            "instrument_key": inst_key,
            "trading_symbol": sym,
            "date": today_date,
            "open": float(first_bar[1]),
            "high": float(max(b[2] for b in raw_bars)),
            "low": float(min(b[3] for b in raw_bars)),
            "close": float(latest_bar[4]),
            "volume": int(sum(b[5] for b in raw_bars if b[5])),
            "open_interest": int(latest_bar[6]) if len(latest_bar) > 6 and latest_bar[6] else 0
        }]
        database.upsert_candles(daily_record)

        # 3. Form today's 75-minute candles and upsert into SQLite
        try:
            import parquet_loader
            df_75_today = parquet_loader.resample_1min_to_75min(df_today)
            if not df_75_today.empty:
                records_75 = []
                for ts, row in df_75_today.iterrows():
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    records_75.append({
                        "instrument_key": inst_key,
                        "trading_symbol": sym,
                        "timeframe": "75m",
                        "timestamp": ts_str,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume", 0))
                    })
                database.upsert_intraday_candles(records_75)
        except Exception as e75:
            logger.warning(f"Could not resample/upsert live 75m candles for {sym}: {e75}")

        # 4. Upsert today's 1-minute bars directly into DuckDB candles_1m
        try:
            records_1m = []
            for b in raw_bars:
                records_1m.append({
                    "symbol": sym,
                    "timestamp": b[0],
                    "open": float(b[1]),
                    "high": float(b[2]),
                    "low": float(b[3]),
                    "close": float(b[4]),
                    "volume": int(b[5]) if b[5] else 0,
                    "oi": int(b[6]) if len(b) > 6 and b[6] else 0
                })
            database.upsert_1m_candles(records_1m)
        except Exception as e1m:
            logger.warning(f"Could not upsert live 1m candles for {sym}: {e1m}")

        return True
    except Exception as e:
        logger.warning(f"Could not sync live intraday candles for {sym}: {e}")
        return False
