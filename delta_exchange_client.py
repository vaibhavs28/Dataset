"""
delta_exchange_client.py
------------------------
REST client for Delta Exchange (India & Global) to fetch live and historical
cryptocurrency data exclusively for BTCUSD and ETHUSD.
Caches and updates candles in the dedicated crypto DuckDB store (crypto_store.py).
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import requests
import pandas as pd

import crypto_store

logger = logging.getLogger(__name__)

# Primary and fallback base URLs
DELTA_BASE_URLS = [
    "https://api.india.delta.exchange",
    "https://api.delta.exchange",
]


# Resolution mapping
TIMEFRAME_TO_DELTA_RES = {
    "1m": "1m",
    "1-Minute": "1m",
    "3m": "3m",
    "3-Minute": "3m",
    "5m": "5m",
    "5-Minute": "5m",
    "15m": "15m",
    "15-Minute": "15m",
    "30m": "30m",
    "30-Minute": "30m",
    "1h": "1h",
    "60-Minute (1H)": "1h",
    "1 Hour": "1h",
    "2h": "2h",
    "2-Hour": "2h",
    "4h": "4h",
    "4-Hour": "4h",
    "6h": "6h",
    "1d": "1d",
    "Daily": "1d",
    "Daily (1D)": "1d",
    "1w": "1w",
    "Weekly": "1w",
    "Weekly (1W)": "1w"
}


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}


def _make_delta_request(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 6) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Tries endpoints across configured Delta Exchange base URLs."""
    last_err = None
    for base_url in DELTA_BASE_URLS:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success", False) or "result" in data:
                    return data, None
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)
            continue

    return None, last_err


def fetch_delta_candles(
    symbol: str = "BTCUSD",
    resolution: str = "1d",
    start: Optional[int] = None,
    end: Optional[int] = None,
    limit: int = 500
) -> Tuple[bool, pd.DataFrame, Optional[str]]:
    """
    Fetches historical OHLCV candles from Delta Exchange /v2/history/candles.
    Returns (success, df, error_message).
    Upserts valid candles into the dedicated crypto DuckDB store.
    """
    symbol = symbol.upper().strip()
    res_code = TIMEFRAME_TO_DELTA_RES.get(resolution, resolution)

    sec_per_bar = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
        "1d": 86400, "1w": 604800
    }.get(res_code, 900)

    now_ts = int(time.time())
    if end is None:
        end_ts = now_ts
    else:
        end_ts = int(end)

    if start is None:
        bar_count = min(max(limit, 300), 2000)
        start_ts = end_ts - (sec_per_bar * bar_count)
    else:
        start_ts = int(start)

    params: Dict[str, Any] = {
        "symbol": symbol,
        "resolution": res_code,
        "start": start_ts,
        "end": end_ts
    }

    data, err = _make_delta_request("/v2/history/candles", params=params)

    if data and "result" in data:
        raw_candles = data.get("result", [])
        if raw_candles:
            # Delta returns candles in reverse chronological order: reverse to ascending
            candles = sorted(raw_candles, key=lambda c: c.get("time", 0))
            rows = []
            for c in candles:
                t_val = c.get("time")
                if not t_val:
                    continue
                rows.append({
                    "dt": pd.to_datetime(t_val, unit="s"),
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "volume": float(c.get("volume", 0.0)),
                })

            if rows:
                df = pd.DataFrame(rows).set_index("dt")
                # Store into dedicated crypto DuckDB
                crypto_store.upsert_crypto_candles(df, symbol, res_code)
                return True, df, None

    # Fallback to local crypto store if network request failed / offline
    df_local = crypto_store.get_crypto_candles(symbol, timeframe=res_code, limit=limit)
    if not df_local.empty:
        return True, df_local, f"Loaded from local crypto DuckDB ({err or 'Offline Mode'})"

    return False, pd.DataFrame(), err or "No data returned from Delta Exchange"


def fetch_delta_ticker(symbol: str = "BTCUSD") -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Fetches live 24h ticker data for BTCUSD or ETHUSD from Delta Exchange.
    Returns (success, ticker_dict, error_message).
    """
    symbol = symbol.upper().strip()

    # 1. Try single product ticker: /v2/tickers/{symbol}
    data, err = _make_delta_request(f"/v2/tickers/{symbol}")
    if data and "result" in data:
        res = data.get("result", {})
        if isinstance(res, dict) and res:
            mark = float(res.get("mark_price", 0.0) or 0.0)
            close = float(res.get("close", 0.0) or mark)
            ticker_info = {
                "symbol": symbol,
                "mark_price": mark,
                "last_price": close,
                "high_24h": float(res.get("high", close * 1.02) or close),
                "low_24h": float(res.get("low", close * 0.98) or close),
                "volume_24h": float(res.get("volume", 0.0) or 0.0),
                "change_24h": float(res.get("change_24h", 0.0) or 0.0),
                "timestamp": int(res.get("timestamp", time.time()) or time.time())
            }
            crypto_store.upsert_crypto_ticker(ticker_info)
            return True, ticker_info, None

    # 2. Try global tickers list: /v2/tickers
    data_all, err_all = _make_delta_request("/v2/tickers")
    if data_all and "result" in data_all:
        for t in data_all.get("result", []):
            if t.get("symbol") == symbol:
                mark = float(t.get("mark_price", 0.0) or 0.0)
                close = float(t.get("close", 0.0) or mark)
                ticker_info = {
                    "symbol": symbol,
                    "mark_price": mark,
                    "last_price": close,
                    "high_24h": float(t.get("high", close * 1.02) or close),
                    "low_24h": float(t.get("low", close * 0.98) or close),
                    "volume_24h": float(t.get("volume", 0.0) or 0.0),
                    "change_24h": float(t.get("change_24h", 0.0) or 0.0),
                    "timestamp": int(t.get("timestamp", time.time()) or time.time())
                }
                crypto_store.upsert_crypto_ticker(ticker_info)
                return True, ticker_info, None

    # Fallback to local stored ticker
    local_ticker = crypto_store.get_crypto_ticker(symbol)
    if local_ticker:
        return True, local_ticker, f"Cached ticker ({err or err_all or 'Local Store'})"

    return False, {}, err or "Failed to fetch Delta ticker"


def sync_all_delta_crypto_history(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    progress_cb=None
) -> Dict[str, Any]:
    """
    Syncs complete historical candles for BTCUSD and ETHUSD across key resolutions.
    """
    if symbols is None:
        symbols = crypto_store.SUPPORTED_CRYPTO_SYMBOLS
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "15m", "5m", "1m"]

    total_tasks = len(symbols) * len(timeframes)
    completed = 0
    results: Dict[str, Any] = {"success_count": 0, "fail_count": 0, "details": []}

    for sym in symbols:
        # Also fetch live ticker
        fetch_delta_ticker(sym)
        for tf in timeframes:
            success, df, err = fetch_delta_candles(symbol=sym, resolution=tf)
            if success and not df.empty:
                results["success_count"] += 1
                results["details"].append(f"✅ {sym} [{tf}]: {len(df)} candles synced")
            else:
                results["fail_count"] += 1
                results["details"].append(f"⚠️ {sym} [{tf}]: {err}")

            completed += 1
            if progress_cb:
                progress_cb(completed, total_tasks, f"Syncing {sym} ({tf})...")

    return results
