"""
screener_engine.py
------------------
High-performance Stock Screener & Filter Engine modeled after Chartink.
Supports multi-timeframe condition evaluation (Daily, Weekly, Monthly, 75m),
complex operators (>, <, crossed_above, crossed_below), custom indicators,
pre-built Chartink screener presets, and Chartink query string parsing.
"""

import os
import threading
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re
import numpy as np
import pandas as pd

import scanner
import database
import parquet_loader
import duckdb_store
import config

logger = logging.getLogger("screener_engine")


@dataclass
class ScreenerClause:
    """Represents a single filter condition row in the screener."""
    timeframe: str = "Daily"       # "Daily", "Weekly", "Monthly", "75-Min"
    offset: int = 0                # 0 = latest candle, 1 = 1 candle ago, etc.
    lhs: str = "Close"             # "Close", "Volume", "RSI_14", "EMA_20", etc.
    operator: str = ">"            # ">", "<", ">=", "<=", "==", "crossed_above", "crossed_below"
    rhs_type: str = "Indicator"    # "Indicator" or "Number"
    rhs_indicator: str = "EMA_20"  # if rhs_type == "Indicator"
    rhs_value: float = 0.0         # if rhs_type == "Number"
    rhs_offset: int = 0
    multiplier: float = 1.0        # e.g., Volume > 2.0 * SMA_Vol_20
    rhs_timeframe: Optional[str] = None # e.g., "Daily" when comparing 15-Min Close < Daily EMA_20


@dataclass
class ScreenerConfig:
    """Overall Screener setup containing multiple clauses."""
    name: str = "Custom Screener"
    logic: str = "ALL"             # "ALL" (AND logic) or "ANY" (OR logic)
    universe: str = "Nifty 500"    # "Nifty 50", "Nifty 100", "Nifty 500", "All Equities"
    clauses: List[ScreenerClause] = field(default_factory=list)


INDICATOR_OPTIONS = [
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "RSI_14",
    "RSI_9",
    "RSI_21",
    "RSI_EMA3",
    "RSI_WMA21",
    "EMA_5",
    "EMA_8",
    "EMA_9",
    "EMA_10",
    "EMA_13",
    "EMA_20",
    "EMA_21",
    "EMA_26",
    "EMA_34",
    "EMA_50",
    "EMA_100",
    "EMA_200",
    "SMA_10",
    "SMA_20",
    "SMA_30",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "SuperTrend",
    "MACD_Line",
    "MACD_Signal",
    "MACD_Hist",
    "ATR_14",
    "Vol_SMA_20",
    "52_Week_High",
    "52_Week_Low",
    "Prev_Day_High",
    "Prev_Day_Low",
    "Prev_Day_Close",
]


OPERATOR_OPTIONS = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "crossed_above",
    "crossed_below",
    "abs_pct_lte"   # abs(lhs-rhs)/lhs*100 <= rhs_value  (MA-squeeze condition)
]


def compute_screener_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all standard indicators used for screening on a given OHLCV DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"] if "volume" in out.columns else pd.Series(1, index=out.index)

    # Price aliases
    out["Close"] = close
    out["Open"] = out["open"]
    out["High"] = high
    out["Low"] = low
    out["Volume"] = volume

    # RSIs
    out["RSI_14"] = scanner.calculate_rsi(close, span=14)
    out["RSI_9"] = scanner.calculate_rsi(close, span=9)
    out["RSI_21"] = scanner.calculate_rsi(close, span=21)
    out["RSI_EMA3"] = scanner.calculate_ema(out["RSI_9"], span=3)
    out["RSI_WMA21"] = scanner.calculate_wma(out["RSI_9"], period=21)

    # EMAs
    out["EMA_5"] = scanner.calculate_ema(close, span=5)
    out["EMA_8"] = scanner.calculate_ema(close, span=8)
    out["EMA_9"] = scanner.calculate_ema(close, span=9)
    out["EMA_10"] = scanner.calculate_ema(close, span=10)
    out["EMA_13"] = scanner.calculate_ema(close, span=13)
    out["EMA_20"] = scanner.calculate_ema(close, span=20)
    out["EMA_21"] = scanner.calculate_ema(close, span=21)
    out["EMA_26"] = scanner.calculate_ema(close, span=26)
    out["EMA_34"] = scanner.calculate_ema(close, span=34)
    out["EMA_50"] = scanner.calculate_ema(close, span=50)
    out["EMA_100"] = scanner.calculate_ema(close, span=100)
    out["EMA_200"] = scanner.calculate_ema(close, span=200)

    # SMAs
    out["SMA_10"] = close.rolling(window=10, min_periods=1).mean()
    out["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    out["SMA_30"] = close.rolling(window=30, min_periods=1).mean()
    out["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    out["SMA_100"] = close.rolling(window=100, min_periods=1).mean()
    out["SMA_200"] = close.rolling(window=200, min_periods=1).mean()

    # SuperTrend (10, 3.0)
    st_df = scanner.calculate_supertrend(out, period=10, multiplier=3.0)
    if "Supertrend" in st_df.columns:
        out["SuperTrend"] = st_df["Supertrend"]
    elif "SuperTrend" in st_df.columns:
        out["SuperTrend"] = st_df["SuperTrend"]
    else:
        out["SuperTrend"] = out["close"]

    if "Trend_Direction" in st_df.columns:
        out["ST_Direction"] = st_df["Trend_Direction"]
    elif "ST_Direction" in st_df.columns:
        out["ST_Direction"] = st_df["ST_Direction"]
    else:
        out["ST_Direction"] = 1

    # MACD (12, 26, 9)
    macd_df = scanner.calculate_macd(close, fast=12, slow=26, signal=9)
    out["MACD_Line"] = macd_df["MACD_Line"]
    out["MACD_Signal"] = macd_df["MACD_Signal"]
    out["MACD_Hist"] = macd_df["MACD_Hist"]

    # ATR (14)
    out["ATR_14"] = scanner.calculate_atr(out, period=14)

    # Volume SMA (20)
    out["Vol_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()

    # 52-Week High & Low (approx 250 bars)
    out["52_Week_High"] = high.rolling(window=min(len(high), 250), min_periods=1).max()
    out["52_Week_Low"] = low.rolling(window=min(len(low), 250), min_periods=1).min()

    # Previous Candle High / Low / Close
    out["Prev_Day_High"] = high.shift(1)
    out["Prev_Day_Low"] = low.shift(1)
    out["Prev_Day_Close"] = close.shift(1)

    return out


def compute_needed_indicators(df: pd.DataFrame, needed_cols: set) -> pd.DataFrame:
    """
    Computes ONLY the indicators requested for a specific screener configuration.
    Eliminates calculating 30+ unneeded indicators per stock, speeding up full scans by 10x-20x.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"] if "volume" in out.columns else pd.Series(1, index=out.index)

    # Base price aliases
    out["Close"] = close
    out["Open"] = out["open"]
    out["High"] = high
    out["Low"] = low
    out["Volume"] = volume

    for col in needed_cols:
        ensure_indicator(out, col)

    return out


def ensure_indicator(df_ind: pd.DataFrame, col_name: str) -> bool:
    """
    Ensures an indicator column exists in df_ind, dynamically computing it on the fly
    for arbitrary periods (e.g. EMA_21, EMA_100, SMA_10, SMA_100, RSI_7, ATR_21).
    """
    if df_ind is None or df_ind.empty or not col_name:
        return False
    if col_name in df_ind.columns:
        return True
    for c in df_ind.columns:
        if c.lower() == col_name.lower():
            return True

    # Check for price aliases
    c_lower = col_name.lower()
    if c_lower in ("close", "open", "high", "low", "volume") and c_lower in df_ind.columns:
        df_ind[col_name] = df_ind[c_lower]
        return True

    if "close" not in df_ind.columns:
        return False

    close = df_ind["close"]

    # Dynamic EMA: EMA_(\d+)
    m_ema = re.match(r"^EMA_(\d+)$", col_name, re.IGNORECASE)
    if m_ema:
        p = int(m_ema.group(1))
        ema_series = scanner.calculate_ema(close, span=p)
        if ema_series.isna().all() and len(close) > 3:
            ema_series = close.ewm(span=p, min_periods=1, adjust=False).mean()
        df_ind[col_name] = ema_series
        return True


    # Dynamic SMA: SMA_(\d+)
    m_sma = re.match(r"^SMA_(\d+)$", col_name, re.IGNORECASE)
    if m_sma:
        p = int(m_sma.group(1))
        df_ind[col_name] = close.rolling(window=p, min_periods=1).mean()
        return True

    # Dynamic RSI: RSI_(\d+)
    m_rsi = re.match(r"^RSI_(\d+)$", col_name, re.IGNORECASE)
    if m_rsi:
        p = int(m_rsi.group(1))
        df_ind[col_name] = scanner.calculate_rsi(close, span=p)
        return True

    # Dynamic ATR: ATR_(\d+)
    m_atr = re.match(r"^ATR_(\d+)$", col_name, re.IGNORECASE)
    if m_atr:
        p = int(m_atr.group(1))
        df_ind[col_name] = scanner.calculate_atr(df_ind, period=p)
        return True

    # Dynamic Volume SMA: Vol_SMA_(\d+)
    m_vsma = re.match(r"^Vol_SMA_(\d+)$", col_name, re.IGNORECASE)
    if m_vsma:
        p = int(m_vsma.group(1))
        vol = df_ind["volume"] if "volume" in df_ind.columns else pd.Series(1, index=df_ind.index)
        df_ind[col_name] = vol.rolling(window=p, min_periods=1).mean()
        return True

    return False


def evaluate_clause(
    df_ind: pd.DataFrame,
    clause: ScreenerClause,
    df_rhs_ind: Optional[pd.DataFrame] = None
) -> Tuple[bool, float, float]:
    """
    Evaluates a single ScreenerClause on the indicator dataframe.
    df_rhs_ind: optional separate DataFrame used to resolve the RHS indicator when
                clause.rhs_timeframe is set (cross-timeframe comparisons, e.g. 15min Close < Daily EMA_20).
    Returns: (passed: bool, lhs_value: float, rhs_value: float)
    """
    if df_ind.empty or len(df_ind) < 3:
        return False, 0.0, 0.0

    lhs_col = clause.lhs
    ensure_indicator(df_ind, lhs_col)
    if lhs_col not in df_ind.columns:
        # Case-insensitive fallback
        matched = [c for c in df_ind.columns if c.lower() == lhs_col.lower()]
        if matched:
            lhs_col = matched[0]
        else:
            return False, 0.0, 0.0

    s_lhs = df_ind[lhs_col]
    idx_curr = -(1 + clause.offset)
    idx_prev = -(2 + clause.offset)

    if abs(idx_curr) > len(s_lhs) or abs(idx_prev) > len(s_lhs):
        return False, 0.0, 0.0

    val_lhs_curr = float(s_lhs.iloc[idx_curr])
    val_lhs_prev = float(s_lhs.iloc[idx_prev])

    if np.isnan(val_lhs_curr):
        return False, 0.0, 0.0

    # Determine RHS values
    if clause.rhs_type == "Number":
        val_rhs_curr = float(clause.rhs_value)
        val_rhs_prev = float(clause.rhs_value)
    else:
        rhs_col = clause.rhs_indicator
        # For cross-timeframe clauses, resolve from the provided rhs df (e.g. Daily EMA when lhs is 15-Min)
        target_rhs_df = df_rhs_ind if (df_rhs_ind is not None and clause.rhs_timeframe) else df_ind
        ensure_indicator(target_rhs_df, rhs_col)
        if rhs_col not in target_rhs_df.columns:
            matched_r = [c for c in target_rhs_df.columns if c.lower() == rhs_col.lower()]
            if matched_r:
                rhs_col = matched_r[0]
            else:
                return False, val_lhs_curr, 0.0
        s_rhs = target_rhs_df[rhs_col] * clause.multiplier
        r_curr = -(1 + clause.rhs_offset)
        r_prev = -(2 + clause.rhs_offset)
        if abs(r_curr) > len(s_rhs) or abs(r_prev) > len(s_rhs):
            return False, val_lhs_curr, 0.0
        val_rhs_curr = float(s_rhs.iloc[r_curr])
        val_rhs_prev = float(s_rhs.iloc[r_prev])

    if np.isnan(val_rhs_curr):
        return False, val_lhs_curr, 0.0

    op = clause.operator
    passed = False
    if op == ">":
        passed = (val_lhs_curr > val_rhs_curr)
    elif op == "<":
        passed = (val_lhs_curr < val_rhs_curr)
    elif op == ">=":
        passed = (val_lhs_curr >= val_rhs_curr)
    elif op == "<=":
        passed = (val_lhs_curr <= val_rhs_curr)
    elif op == "==":
        passed = abs(val_lhs_curr - val_rhs_curr) < 1e-4
    elif op == "crossed_above":
        passed = (val_lhs_curr > val_rhs_curr) and (val_lhs_prev <= val_rhs_prev)
    elif op == "crossed_below":
        passed = (val_lhs_curr < val_rhs_curr) and (val_lhs_prev >= val_rhs_prev)
    elif op == "abs_pct_lte":
        # abs(lhs - rhs) / lhs * 100 <= rhs_value (percentage)
        # Used for MA squeeze: abs(EMA_9 - EMA_13) / EMA_9 * 100 <= 0.01
        if val_lhs_curr != 0:
            diff_pct = abs(val_lhs_curr - val_rhs_curr) / abs(val_lhs_curr) * 100.0
            passed = diff_pct <= clause.rhs_value
        else:
            passed = (abs(val_lhs_curr - val_rhs_curr) <= clause.rhs_value)

    return passed, val_lhs_curr, val_rhs_curr


def evaluate_stock(
    symbol: str,
    timeframe_dfs: Dict[str, pd.DataFrame],
    cfg: ScreenerConfig,
    return_clause_results: bool = False
) -> Any:
    """
    Evaluates all clauses for a single symbol across required timeframes.
    Returns dictionary with match info or None if criteria not met.
    If return_clause_results is True, returns (res, clause_results).
    """
    if not cfg.clauses:
        return (None, []) if return_clause_results else None

    clause_results = []
    indicator_summary = {}

    daily_df = timeframe_dfs.get("Daily")
    if daily_df is None or daily_df.empty:
        return (None, []) if return_clause_results else None

    ltp = float(daily_df["close"].iloc[-1])
    prev_close = float(daily_df["close"].iloc[-2]) if len(daily_df) >= 2 else ltp
    change_pct = ((ltp - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    vol = int(daily_df["volume"].iloc[-1]) if "volume" in daily_df.columns else 0

    for i, clause in enumerate(cfg.clauses):
        tf = clause.timeframe
        df_tf = timeframe_dfs.get(tf)
        if df_tf is None or df_tf.empty:
            clause_results.append(False)
            continue

        # Cross-timeframe RHS lookup (e.g. 15min Close < Daily EMA_20)
        df_rhs_tf = timeframe_dfs.get(clause.rhs_timeframe) if clause.rhs_timeframe else None

        passed, lhs_val, rhs_val = evaluate_clause(df_tf, clause, df_rhs_ind=df_rhs_tf)
        clause_results.append(passed)

        # Store for display
        indicator_summary[f"C{i+1}_Result"] = "✅" if passed else "❌"
        indicator_summary[f"C{i+1}_LHS"] = round(lhs_val, 2)
        indicator_summary[f"C{i+1}_RHS"] = round(rhs_val, 2)


    if cfg.logic == "ALL":
        overall_match = all(clause_results)
    else:  # ANY
        overall_match = any(clause_results)

    if not overall_match:
        return (None, clause_results) if return_clause_results else None

    res = {
        "Symbol": symbol,
        "LTP": round(ltp, 2),
        "Change_%": round(change_pct, 2),
        "Volume": vol,
        "Clauses_Passed": sum(clause_results),
        "Total_Clauses": len(cfg.clauses),
    }
    res.update(indicator_summary)
    return (res, clause_results) if return_clause_results else res


def run_screen(
    symbols: List[str],
    cfg: ScreenerConfig,
    as_of_date: Optional[Any] = None,
    as_of_time: Optional[str] = None,
    data_provider_fn=None,
    progress_callback=None
) -> pd.DataFrame:
    """
    Executes a screener across a list of symbols.
    Supports historical date & time backtesting via `as_of_date` and `as_of_time`.
    data_provider_fn(symbol, timeframe) returns raw OHLCV DataFrame.
    """
    # ── Ultra-Fast DuckDB SQL Path (Sub-20ms instant scan) ───────────────────
    # If scanning latest market data (no historical backtest date) and snapshot exists,
    # run direct vectorized C++ SQL query against `screener_flat_snapshot`.
    if as_of_date is None and data_provider_fn is None and len(symbols) > 4:
        try:
            import screener_snapshot
            if screener_snapshot.has_screener_snapshot():
                if progress_callback:
                    try:
                        progress_callback(len(symbols), len(symbols), "⚡ Instant DuckDB Vector Query")
                    except Exception:
                        pass
                df_sql = screener_snapshot.query_screener_snapshot(cfg, universe_symbols=symbols)
                if df_sql is not None and not df_sql.empty:
                    df_sql.attrs["clause_pass_counts"] = [len(df_sql)] * len(cfg.clauses)
                    df_sql.attrs["evaluated_count"] = len(symbols)
                    df_sql.attrs["engine"] = "DUCKDB_SQL_VECTOR"
                    return df_sql
                elif df_sql is not None and df_sql.empty:
                    df_sql.attrs["clause_pass_counts"] = [0] * len(cfg.clauses)
                    df_sql.attrs["evaluated_count"] = len(symbols)
                    df_sql.attrs["engine"] = "DUCKDB_SQL_VECTOR"
                    return df_sql
        except Exception as e:
            logger.debug(f"DuckDB SQL fast path fallback to memory engine: {e}")

    matches = []
    needed_tfs = set(c.timeframe for c in cfg.clauses)
    # Also include any rhs_timeframe (cross-TF comparisons like 15min Close < Daily EMA)
    for c in cfg.clauses:
        if c.rhs_timeframe:
            needed_tfs.add(c.rhs_timeframe)
    if not needed_tfs:
        needed_tfs = {"Daily"}

    needed_indicators_by_tf: Dict[str, set] = {}
    for c in cfg.clauses:
        tf = c.timeframe or "Daily"
        if tf not in needed_indicators_by_tf:
            needed_indicators_by_tf[tf] = set()
        if c.lhs:
            needed_indicators_by_tf[tf].add(c.lhs)
        if c.rhs_type == "Indicator" and c.rhs_indicator:
            # For cross-TF clauses put rhs_indicator into rhs_timeframe's dict
            rhs_tf = c.rhs_timeframe if c.rhs_timeframe else tf
            if rhs_tf not in needed_indicators_by_tf:
                needed_indicators_by_tf[rhs_tf] = set()
            needed_indicators_by_tf[rhs_tf].add(c.rhs_indicator)

    close_to_start = {
        "10:30": "09:15:00",
        "11:45": "10:30:00",
        "13:00": "11:45:00",
        "14:15": "13:00:00",
        "15:30": "14:15:00",
    }
    time_str = as_of_time[:5] if as_of_time else "15:30"
    time_cutoff = close_to_start.get(time_str, "14:15:00")

    clause_pass_counts = [0] * len(cfg.clauses)
    evaluated_count = 0

    # ── Pre-split clauses by tier (daily-resampable vs intraday) ────────────
    # Daily-tier: Monthly/Weekly/Daily can be computed from batch-loaded daily data
    # Intraday-tier: 75-Min, 15-Min need separate per-stock DuckDB/parquet loads
    DAILY_TIER_TFS = {"Daily", "Weekly", "Monthly"}
    INTRADAY_TIER_TFS = {"75-Min", "15-Min"}

    daily_tier_clauses  = [c for c in cfg.clauses if c.timeframe in DAILY_TIER_TFS]
    intraday_75m_clauses = [c for c in cfg.clauses if c.timeframe == "75-Min"]
    intraday_15m_clauses = [c for c in cfg.clauses if c.timeframe == "15-Min"
                            or (c.rhs_timeframe and c.rhs_timeframe not in DAILY_TIER_TFS)]
    has_intraday = bool(intraday_75m_clauses or intraday_15m_clauses)
    has_75m = "75-Min" in needed_tfs
    has_15m = "15-Min" in needed_tfs

    # Build daily-tier-only screener config for pre-filter
    daily_tier_cfg = ScreenerConfig(
        name=cfg.name, logic=cfg.logic,
        clauses=daily_tier_clauses
    ) if daily_tier_clauses else None

    batch_daily: Dict[str, pd.DataFrame] = {}
    if not data_provider_fn and len(symbols) > 1:
        try:
            batch_daily = database.get_batch_candles_df(symbols)
        except Exception as e:
            logger.warning(f"Batch daily load error in run_screen: {e}")

    # ── Stage A: evaluate daily-tier across all symbols (fast, all in RAM) ─
    daily_tier_pass: Dict[str, Dict[str, pd.DataFrame]] = {}  # sym -> tf_dfs that passed

    def _build_daily_tf_dfs(sym: str, daily_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build the Daily/Weekly/Monthly indicator frames from daily raw data."""
        tf_dfs: Dict[str, pd.DataFrame] = {}
        tf_dfs["Daily"] = compute_needed_indicators(daily_raw, needed_indicators_by_tf.get("Daily", set()))
        if "Weekly" in needed_tfs:
            w_raw = scanner.resample_ohlcv(daily_raw, "weekly")
            tf_dfs["Weekly"] = compute_needed_indicators(w_raw, needed_indicators_by_tf.get("Weekly", set()))
        if "Monthly" in needed_tfs:
            m_raw = scanner.resample_ohlcv(daily_raw, "monthly")
            tf_dfs["Monthly"] = compute_needed_indicators(m_raw, needed_indicators_by_tf.get("Monthly", set()))
        return tf_dfs

    def _get_daily_raw(sym: str) -> Optional[pd.DataFrame]:
        clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
        if data_provider_fn:
            return data_provider_fn(sym, "Daily")
        raw = batch_daily.get(clean)
        if (raw is None or raw.empty) and not batch_daily:
            raw = database.get_candles_df(sym)
        return raw

    clause_pass_counts = [0] * len(cfg.clauses)
    evaluated_count = 0

    # If ALL logic and has intraday tier, pre-filter by daily tier first
    use_funnel = (cfg.logic == "ALL" and has_intraday and daily_tier_clauses and not data_provider_fn)

    daily_raw_cache: Dict[str, pd.DataFrame] = {}  # store raw daily for later intraday stage

    if use_funnel and len(symbols) > 4:
        pre_filter_workers = min(32, max(4, (os.cpu_count() or 4) * 4))
        # Phase 1 progress: report each pre-filter completion
        pre_completed = 0

        def _check_daily_tier(sym: str):
            daily_raw = _get_daily_raw(sym)
            if daily_raw is None or daily_raw.empty or len(daily_raw) < 5:
                return sym, None, None

            if not isinstance(daily_raw.index, pd.DatetimeIndex):
                daily_raw = daily_raw.copy()
                daily_raw.index = pd.to_datetime(daily_raw.index)

            if as_of_date is not None:
                target_dt = pd.to_datetime(as_of_date)
                eod_naive = target_dt.replace(hour=23, minute=59, second=59)
                d_tz = getattr(daily_raw.index, "tz", None)
                if d_tz is not None:
                    daily_raw = daily_raw[daily_raw.index <= eod_naive.tz_localize(d_tz)]
                else:
                    daily_raw = daily_raw[daily_raw.index <= eod_naive]
                if daily_raw.empty or len(daily_raw) < 5:
                    return sym, None, None

            tf_dfs = _build_daily_tf_dfs(sym, daily_raw)
            if daily_tier_cfg is not None:
                res, _cr = evaluate_stock(sym, tf_dfs, daily_tier_cfg, return_clause_results=True)
                if res is None:
                    return sym, None, None
            return sym, daily_raw, tf_dfs

        # ── Phase 1: pre-filter with live progress ───────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=pre_filter_workers) as ex:
            future_map = {ex.submit(_check_daily_tier, sym): sym for sym in symbols}
            for fut in concurrent.futures.as_completed(future_map):
                pre_completed += 1
                if progress_callback:
                    try:
                        progress_callback(pre_completed, total_symbols, f"[Pre-filter] {future_map[fut]}")
                    except Exception:
                        pass
                sym, daily_raw, tf_dfs = fut.result()
                if daily_raw is not None:
                    daily_tier_pass[sym] = tf_dfs
                    daily_raw_cache[sym] = daily_raw

        passed_syms = list(daily_tier_pass.keys())
        logger.info(
            f"⚡ Funnel pre-filter: {len(passed_syms)}/{len(symbols)} stocks passed daily-tier "
            f"(skipped {len(symbols)-len(passed_syms)} intraday loads)"
        )

        # ── Stage B: batch-load intraday for the small set that passed ──────
        batch_75m: Dict[str, pd.DataFrame] = {}
        batch_15m: Dict[str, pd.DataFrame] = {}

        if passed_syms:
            if has_75m:
                try:
                    batch_75m = duckdb_store.get_batch_resampled_candles(
                        passed_syms, interval_minutes=75, limit_per_symbol=500)
                except Exception as e:
                    logger.debug(f"Batch 75m resample error: {e}")
            if has_15m:
                try:
                    batch_15m = duckdb_store.get_batch_resampled_candles(
                        passed_syms, interval_minutes=15, limit_per_symbol=500)
                except Exception as e:
                    logger.debug(f"Batch 15m resample error: {e}")

        def _eval_funneled_sym(sym: str) -> Tuple[Optional[Dict], Optional[List[bool]]]:
            tf_dfs = daily_tier_pass.get(sym)
            if tf_dfs is None:
                return None, None

            daily_raw = daily_raw_cache.get(sym)

            # Add 75-Min
            if has_75m:
                clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
                intra_75 = batch_75m.get(clean)
                if intra_75 is None or intra_75.empty:
                    intra_75 = parquet_loader.ensure_symbol_75m_candles(sym, min_bars=20)
                if intra_75 is not None and not intra_75.empty:
                    if not isinstance(intra_75.index, pd.DatetimeIndex):
                        intra_75.index = pd.to_datetime(intra_75.index)
                    if as_of_date is not None:
                        target_dt = pd.to_datetime(f"{str(as_of_date)[:10]} {time_cutoff}")
                        tz = getattr(intra_75.index, "tz", None)
                        intra_75 = intra_75[intra_75.index <= (target_dt.tz_localize(tz) if tz else target_dt)]
                    if intra_75 is not None and not intra_75.empty:
                        tf_dfs["75-Min"] = compute_needed_indicators(
                            intra_75, needed_indicators_by_tf.get("75-Min", set()))

            # Add 15-Min
            if has_15m:
                clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
                intra_15 = batch_15m.get(clean)
                if intra_15 is None or intra_15.empty:
                    intra_15 = parquet_loader.ensure_symbol_custom_minute_candles(
                        sym, interval_minutes=15, min_bars=10)
                if intra_15 is not None and not intra_15.empty:
                    if not isinstance(intra_15.index, pd.DatetimeIndex):
                        intra_15.index = pd.to_datetime(intra_15.index)
                    if as_of_date is not None:
                        target_dt = pd.to_datetime(f"{str(as_of_date)[:10]} {time_cutoff}")
                        tz = getattr(intra_15.index, "tz", None)
                        intra_15 = intra_15[intra_15.index <= (target_dt.tz_localize(tz) if tz else target_dt)]
                    if intra_15 is not None and not intra_15.empty:
                        tf_dfs["15-Min"] = compute_needed_indicators(
                            intra_15, needed_indicators_by_tf.get("15-Min", set()))

            eval_res, clause_results = evaluate_stock(sym, tf_dfs, cfg, return_clause_results=True)
            if eval_res is not None and daily_raw is not None:
                as_of_str = str(daily_raw.index[-1])[:10]
                eval_res["Scan_Date"] = as_of_str
                eval_res["Scan_Time"] = time_str
            return eval_res, clause_results

        # ── Phase 2: intraday evaluation only on passed_syms ────────────────
        # Continue progress from where phase 1 left off (total_symbols is still len(symbols))
        # We already reported pre_completed = total_symbols during phase 1
        # Now re-report as "intraday evaluation" sub-phase on the small set
        intra_workers = min(32, max(4, (os.cpu_count() or 4) * 4))
        intra_completed = 0
        intra_total = len(passed_syms)

        with concurrent.futures.ThreadPoolExecutor(max_workers=intra_workers) as executor:
            future_to_sym = {executor.submit(_eval_funneled_sym, sym): sym for sym in passed_syms}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                intra_completed += 1
                if progress_callback:
                    try:
                        # Show intraday progress as sub-step: "499+k/500 (evaluating k/N 75m+15m)"
                        progress_callback(
                            total_symbols,  # keep bar at 100% (pre-filter done)
                            total_symbols,
                            f"[Intraday {intra_completed}/{intra_total}] {sym}"
                        )
                    except Exception:
                        pass
                try:
                    eval_res, clause_results = future.result()
                    if clause_results:
                        evaluated_count += 1
                        for i, p in enumerate(clause_results):
                            if p and i < len(clause_pass_counts):
                                clause_pass_counts[i] += 1
                    if eval_res is not None:
                        matches.append(eval_res)
                except Exception as e:
                    logger.error(f"Error screening {sym}: {e}")

        if matches:
            df_out = pd.DataFrame(matches)
            df_out.attrs["clause_pass_counts"] = clause_pass_counts
            df_out.attrs["evaluated_count"] = evaluated_count
            return df_out
        return pd.DataFrame()


    # ── Non-funneled path (no intraday, custom data_provider_fn, or small list) ──
    def _eval_screen_sym(sym: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[bool]]]:
        clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
        daily_raw = _get_daily_raw(sym)

        if daily_raw is None or daily_raw.empty or len(daily_raw) < 5:
            return None, None

        if not isinstance(daily_raw.index, pd.DatetimeIndex):
            daily_raw = daily_raw.copy()
            daily_raw.index = pd.to_datetime(daily_raw.index)

        full_daily = daily_raw
        if as_of_date is not None:
            target_dt = pd.to_datetime(as_of_date)
            eod_naive = target_dt.replace(hour=23, minute=59, second=59)
            d_tz = getattr(daily_raw.index, "tz", None)
            if d_tz is not None:
                daily_raw = daily_raw[daily_raw.index <= eod_naive.tz_localize(d_tz)]
            else:
                daily_raw = daily_raw[daily_raw.index <= eod_naive]
            if daily_raw.empty or len(daily_raw) < 5:
                return None, None

        tf_dfs = _build_daily_tf_dfs(sym, daily_raw)

        # 75-Min intraday
        if "75-Min" in needed_tfs:
            intra_raw = data_provider_fn(sym, "75-Min") if data_provider_fn else parquet_loader.ensure_symbol_75m_candles(sym, min_bars=20)
            if intra_raw is not None and not intra_raw.empty:
                if not isinstance(intra_raw.index, pd.DatetimeIndex):
                    intra_raw = intra_raw.copy()
                    intra_raw.index = pd.to_datetime(intra_raw.index)
                if as_of_date is not None:
                    target_dt = pd.to_datetime(f"{str(as_of_date)[:10]} {time_cutoff}")
                    tz = getattr(intra_raw.index, "tz", None)
                    intra_raw = intra_raw[intra_raw.index <= (target_dt.tz_localize(tz) if tz else target_dt)]
                if intra_raw is not None and not intra_raw.empty:
                    tf_dfs["75-Min"] = compute_needed_indicators(intra_raw, needed_indicators_by_tf.get("75-Min", set()))

        # 15-Min intraday
        if "15-Min" in needed_tfs:
            intra_15 = data_provider_fn(sym, "15-Min") if data_provider_fn else parquet_loader.ensure_symbol_custom_minute_candles(sym, interval_minutes=15, min_bars=10)
            if intra_15 is not None and not intra_15.empty:
                if not isinstance(intra_15.index, pd.DatetimeIndex):
                    intra_15 = intra_15.copy()
                    intra_15.index = pd.to_datetime(intra_15.index)
                if as_of_date is not None:
                    target_dt = pd.to_datetime(f"{str(as_of_date)[:10]} {time_cutoff}")
                    tz = getattr(intra_15.index, "tz", None)
                    intra_15 = intra_15[intra_15.index <= (target_dt.tz_localize(tz) if tz else target_dt)]
                if intra_15 is not None and not intra_15.empty:
                    tf_dfs["15-Min"] = compute_needed_indicators(intra_15, needed_indicators_by_tf.get("15-Min", set()))

        eval_res, clause_results = evaluate_stock(sym, tf_dfs, cfg, return_clause_results=True)
        if eval_res is not None:
            as_of_str = str(daily_raw.index[-1])[:10]
            eval_res["Scan_Date"] = as_of_str
            eval_res["Scan_Time"] = time_str
            if len(full_daily) > len(daily_raw):
                future_close = float(full_daily["close"].iloc[-1])
                curr_p = float(eval_res.get("LTP", 0.0))
                if curr_p > 0:
                    eval_res["Return_Since_Scan_%"] = round(((future_close - curr_p) / curr_p) * 100.0, 2)
                    eval_res["Latest_Close"] = round(future_close, 2)

        return eval_res, clause_results


    completed_count = 0
    total_symbols = len(symbols)
    max_workers = min(32, max(4, (os.cpu_count() or 4) * 4)) if total_symbols > 4 else 1
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {executor.submit(_eval_screen_sym, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_symbols, sym)
                    except Exception:
                        pass
                try:
                    eval_res, clause_results = future.result()
                    if clause_results:
                        evaluated_count += 1
                        for i, p in enumerate(clause_results):
                            if p:
                                clause_pass_counts[i] += 1
                    if eval_res is not None:
                        matches.append(eval_res)
                except Exception as e:
                    logger.error(f"Error screening {sym}: {e}")
    else:
        for sym in symbols:
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, total_symbols, sym)
                except Exception:
                    pass
            eval_res, clause_results = _eval_screen_sym(sym)
            if clause_results:
                evaluated_count += 1
                for i, p in enumerate(clause_results):
                    if p:
                        clause_pass_counts[i] += 1
            if eval_res is not None:
                matches.append(eval_res)

    df_res = pd.DataFrame(matches) if matches else pd.DataFrame()
    if not df_res.empty and "Change_%" in df_res.columns:
        df_res = df_res.sort_values(by="Change_%", ascending=False)

    df_res.attrs["diag"] = {
        "total_universe": len(symbols),
        "evaluated_stocks": evaluated_count,
        "clause_stats": [
            {
                "index": i + 1,
                "desc": f"{c.timeframe} {c.lhs} {c.operator} " + (f"{c.multiplier} * {c.rhs_indicator}" if c.rhs_type == "Indicator" else f"{c.rhs_value}"),
                "passed_count": clause_pass_counts[i] if i < len(clause_pass_counts) else 0,
                "passed_pct": round(clause_pass_counts[i] / max(evaluated_count, 1) * 100, 1) if i < len(clause_pass_counts) else 0.0
            }
            for i, c in enumerate(cfg.clauses)
        ],
        "logic": cfg.logic
    }
    return df_res


def evaluate_stock_waterfall(
    symbol: str,
    daily_df: pd.DataFrame,
    intra_75_df: Optional[pd.DataFrame] = None,
    as_of_date: Optional[Any] = None,
    as_of_time: Optional[str] = None,
    intra_75_provider: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates exact Chartink 'positional-scan-364' rules via a strict Waterfall Model:
    Monthly Pass ➔ Weekly Pass ➔ Daily Pass ➔ 75-Min Pass.
    If any stage fails, the stock cannot progress to subsequent stages.
    If Monthly fails, the stock is completely excluded (returns None).
    Supports historical date & time slicing if as_of_date/as_of_time are provided.
    Supports lazy evaluation of 75m intraday data to eliminate redundant disk queries.
    """
    if daily_df is None or daily_df.empty:
        return None

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df = daily_df.copy()
        daily_df.index = pd.to_datetime(daily_df.index)

    full_daily = daily_df
    latest_market_close = float(full_daily["close"].iloc[-1])

    # Determine candle close and start time mappings
    candle_close_map = {
        (9, 15): "10:30",
        (10, 30): "11:45",
        (11, 45): "13:00",
        (13, 0): "14:15",
        (14, 15): "15:30",
    }
    close_to_start = {
        "10:30": "09:15:00",
        "11:45": "10:30:00",
        "13:00": "11:45:00",
        "14:15": "13:00:00",
        "15:30": "14:15:00",
    }

    scan_time_str = "15:30"
    if as_of_time:
        scan_time_str = as_of_time[:5]
    elif intra_75_df is not None and not intra_75_df.empty:
        last_intra_dt = intra_75_df.index[-1]
        if hasattr(last_intra_dt, "hour"):
            scan_time_str = candle_close_map.get((last_intra_dt.hour, last_intra_dt.minute), f"{last_intra_dt.hour:02d}:{last_intra_dt.minute:02d}")

    def _slice_intra_75(idf: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if idf is None or idf.empty:
            return idf
        if not isinstance(idf.index, pd.DatetimeIndex):
            idf = idf.copy()
            idf.index = pd.to_datetime(idf.index)
        if as_of_date is not None:
            time_cutoff = close_to_start.get(scan_time_str, "14:15:00")
            dt_str = str(as_of_date)[:10]
            cutoff_dt = pd.to_datetime(f"{dt_str} {time_cutoff}")
            if idf.index.tz is not None:
                idf = idf[idf.index <= cutoff_dt.tz_localize(idf.index.tz)]
            else:
                idf = idf[idf.index <= cutoff_dt]
        return idf

    if as_of_date is not None:
        target_dt = pd.to_datetime(as_of_date)
        eod_naive = target_dt.replace(hour=23, minute=59, second=59)
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.to_datetime(daily_df.index)
        d_tz = getattr(daily_df.index, "tz", None)
        if d_tz is not None:
            daily_df = daily_df[daily_df.index <= eod_naive.tz_localize(d_tz)]
        else:
            daily_df = daily_df[daily_df.index <= eod_naive]

    if daily_df is None or daily_df.empty or len(daily_df) < 15:
        return None

    close = daily_df["close"].values
    ltp = float(close[-1])

    if ltp < 100.0 or ltp > 10000.0:
        return None

    prev_close = float(close[-2]) if len(close) >= 2 else ltp
    change_pct = ((ltp - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    vol = int(daily_df["volume"].iloc[-1]) if "volume" in daily_df.columns else 0
    as_of_str = str(daily_df.index[-1])[:10]
    has_forward_data = (len(full_daily) > len(daily_df))
    fwd_return_pct = round(((latest_market_close - ltp) / ltp) * 100.0, 2) if has_forward_data else 0.0

    # ==========================================
    # 1. MONTHLY STAGE (Macro Trend & Hilega Milega)
    # ==========================================
    m_raw = scanner.resample_ohlcv(daily_df, "monthly")
    if m_raw.empty or len(m_raw) < 3:
        return None

    m_close = m_raw["close"]
    m_ema5 = scanner.calculate_ema(m_close, span=5).values
    m_ema20 = scanner.calculate_ema(m_close, span=20).values
    m_ema50 = scanner.calculate_ema(m_close, span=50).values
    m_rsi = scanner.calculate_rsi(m_close, span=9).values
    m_rsi_ema3 = scanner.calculate_ema(pd.Series(m_rsi), span=3).values
    m_rsi_wma21 = scanner.calculate_wma(pd.Series(m_rsi), period=21).values

    m_c = float(m_close.iloc[-1])
    m_e5 = float(m_ema5[-1])
    m_e20 = float(m_ema20[-1])
    m_e50 = float(m_ema50[-1])
    m_r = float(m_rsi[-1])
    m_re3 = float(m_rsi_ema3[-1])
    m_rw21 = float(m_rsi_wma21[-1])

    # Rules: Close > 5 EMA, Close > 20 EMA, 5 EMA > 20 EMA, RSI(9) > 50, RSI < 88, RSI > EMA3, RSI > WMA21
    m_pass = (
        (m_c > m_e5) and
        (m_c > m_e20) and
        (m_e5 > m_e20) and
        (50.0 < m_r < 88.0) and
        (m_r >= m_re3) and
        (m_r >= m_rw21)
    )

    # In strict waterfall: If Monthly fails, stock is hidden!
    if not m_pass:
        return None

    stage = 1
    w_pass = False
    d_pass = False
    q4_pass = False

    # ==========================================
    # 2. WEEKLY STAGE (Intermediate Trend)
    # ==========================================
    w_raw = scanner.resample_ohlcv(daily_df, "weekly")
    if not w_raw.empty and len(w_raw) >= 5:
        w_close = w_raw["close"]
        w_ema20 = scanner.calculate_ema(w_close, span=20).values
        w_ema50 = scanner.calculate_ema(w_close, span=50).values
        w_rsi = scanner.calculate_rsi(w_close, span=9).values
        w_rsi_ema3 = scanner.calculate_ema(pd.Series(w_rsi), span=3).values
        w_rsi_wma21 = scanner.calculate_wma(pd.Series(w_rsi), period=21).values

        w_c = float(w_close.iloc[-1])
        w_e20 = float(w_ema20[-1])
        w_e50 = float(w_ema50[-1])
        w_r = float(w_rsi[-1])
        w_re3 = float(w_rsi_ema3[-1])
        w_rw21 = float(w_rsi_wma21[-1])

        # Rules: Close > 20 EMA, 20 EMA > 50 EMA, RSI > 50, RSI > EMA3, RSI > WMA21
        w_pass = (
            (w_c > w_e20) and
            (w_e20 > w_e50) and
            (w_r > 50.0) and
            (w_r >= w_re3) and
            (w_r >= w_rw21)
        )
        if w_pass:
            stage = 2
    else:
        w_c, w_e20, w_e50, w_r = np.nan, np.nan, np.nan, np.nan

    # ==========================================
    # 3. DAILY STAGE (Setup & 20 EMA Bounce)
    # ==========================================
    d_c, d_o, d_e20, d_e50, d_r = np.nan, np.nan, np.nan, np.nan, np.nan
    if w_pass:
        d_close = daily_df["close"]
        d_open = daily_df["open"]
        d_ema20 = scanner.calculate_ema(d_close, span=20).values
        d_ema50 = scanner.calculate_ema(d_close, span=50).values
        d_rsi = scanner.calculate_rsi(d_close, span=9).values
        d_rsi_ema3 = scanner.calculate_ema(pd.Series(d_rsi), span=3).values
        d_rsi_wma21 = scanner.calculate_wma(pd.Series(d_rsi), period=21).values

        d_c = float(d_close.iloc[-1])
        d_o = float(d_open.iloc[-1])
        d_e20 = float(d_ema20[-1])
        d_e50 = float(d_ema50[-1])
        d_r = float(d_rsi[-1])
        d_re3 = float(d_rsi_ema3[-1])
        d_rw21 = float(d_rsi_wma21[-1])

        # Rules: Open <= 20 EMA and Close >= 20 EMA (bounce / crossover), 20 EMA > 50 EMA, RSI > 50, RSI > EMA3, RSI > WMA21
        # Allow 1% proximity tolerance for real-market bounce
        d_bounce = (d_o <= d_e20 * 1.015) and (d_c >= d_e20 * 0.985)
        d_trend = (d_e20 > d_e50)
        d_hilega = (d_r > 50.0) and (d_r >= d_re3) and (d_r >= d_rw21)

        d_pass = (d_bounce and d_trend and d_hilega)
        if d_pass:
            stage = 3

    # ==========================================
    # 4. 75-MIN STAGE (Q4 Intraday Trigger)
    # ==========================================
    q4_c, q4_e20, q4_r = np.nan, np.nan, np.nan
    if d_pass:
        if intra_75_df is None:
            if intra_75_provider is not None:
                try:
                    intra_75_df = intra_75_provider(symbol)
                except TypeError:
                    intra_75_df = intra_75_provider(symbol, "75-Min")
            else:
                intra_75_df = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=20)
            if intra_75_df is not None and not intra_75_df.empty:
                intra_75_df = _slice_intra_75(intra_75_df)

        if intra_75_df is not None and not intra_75_df.empty and len(intra_75_df) >= 10:
            q_close = intra_75_df["close"]
            if as_of_time and as_of_time[:5] != "15:30":
                ltp = float(q_close.iloc[-1])
            q_ema5 = scanner.calculate_ema(q_close, span=5).values
            q_ema20 = scanner.calculate_ema(q_close, span=20).values
            q_rsi = scanner.calculate_rsi(q_close, span=9).values
            q_rsi_ema3 = scanner.calculate_ema(pd.Series(q_rsi), span=3).values
            q_rsi_wma21 = scanner.calculate_wma(pd.Series(q_rsi), period=21).values

            q4_c = float(q_close.iloc[-1])
            q4_e5 = float(q_ema5[-1])
            q4_e20 = float(q_ema20[-1])
            q4_r = float(q_rsi[-1])
            q4_re3 = float(q_rsi_ema3[-1])
            q4_rw21 = float(q_rsi_wma21[-1])

            # Rules: 75m Close > 20 EMA, 5 EMA > 20 EMA, RSI(9) > 50, RSI > EMA3, RSI > WMA21
            q4_pass = (
                (q4_c > q4_e20) and
                (q4_e5 >= q4_e20) and
                (q4_r > 50.0) and
                (q4_r >= q4_re3) and
                (q4_r >= q4_rw21)
            )
            if q4_pass:
                stage = 4

    stage_labels = {
        4: "🏆 Stage 4 (M+W+D+75m Aligned)",
        3: "🚀 Stage 3 (M+W+D Aligned)",
        2: "🟢 Stage 2 (M+W Aligned)",
        1: "🟡 Stage 1 (Monthly Macro Pass)"
    }

    return {
        "Symbol": symbol,
        "Scan Date": as_of_str,
        "Scan Time": scan_time_str,
        "LTP": round(ltp, 2),
        "1D Return (%)": round(change_pct, 2),
        "Return Since Scan (%)": fwd_return_pct if has_forward_data else None,
        "Latest Price": round(latest_market_close, 2) if has_forward_data else round(ltp, 2),
        "Volume": vol,
        "Stage": stage,
        "Waterfall Stage": stage_labels[stage],
        "Monthly": "✅ PASS",
        "Weekly": "✅ PASS" if w_pass else "❌ FAIL",
        "Daily": "✅ PASS" if d_pass else "❌ FAIL",
        "75-Min": "✅ PASS" if q4_pass else "❌ FAIL",
        "M_RSI": round(m_r, 1),
        "M_EMA5": round(m_e5, 2),
        "M_EMA20": round(m_e20, 2),
        "W_RSI": round(w_r, 1) if not np.isnan(w_r) else np.nan,
        "W_EMA20": round(w_e20, 2) if not np.isnan(w_e20) else np.nan,
        "D_RSI": round(d_r, 1) if not np.isnan(d_r) else np.nan,
        "D_EMA20": round(d_e20, 2) if not np.isnan(d_e20) else np.nan,
        "75m_RSI": round(q4_r, 1) if not np.isnan(q4_r) else np.nan,
        "75m_EMA20": round(q4_e20, 2) if not np.isnan(q4_e20) else np.nan,
    }


def evaluate_intraday_scan_19122704(
    symbol: str,
    daily_df: pd.DataFrame,
    intra_75_df: Optional[pd.DataFrame] = None,
    intra_15_df: Optional[pd.DataFrame] = None,
    as_of_date: Optional[Any] = None,
    as_of_time: Optional[str] = None,
    intra_75_provider = None,
    intra_15_provider = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a stock against Chartink 'intraday-scan-19122704' (75m+15m Intraday Multi-TF Breakdown).
    Five-stage sequential funnel:
      - Stage 1: Monthly Breakdown (Close < EMA5 < EMA20 < EMA50, RSI9 < 50, RSI9 < RSI_EMA3, RSI9 < RSI_WMA21)
      - Stage 2: Weekly Breakdown (Close < EMA20 < EMA50 < EMA200, RSI9 < 50, RSI9 < RSI_EMA3, RSI9 < RSI_WMA21)
      - Stage 3: Daily Breakdown (Close < EMA20 < EMA50 < EMA200, RSI9 < 50, RSI9 < RSI_EMA3, RSI9 < RSI_WMA21)
      - Stage 4: 75m Breakdown Trigger (75m EMA20 <= EMA50 <= EMA200, 75m Close < EMA20)
      - Stage 5: 15m Precision Entry (15m Close < Daily EMA20, prev bar bearish, gap-down open, current bar bearish,
                  EMA20 < EMA50 < EMA200, MA squeeze on 9/13/20/26)
    """
    if daily_df is None or daily_df.empty or len(daily_df) < 15:
        return None

    if as_of_date is not None:
        target_dt = pd.to_datetime(as_of_date)
        eod_naive = target_dt.replace(hour=23, minute=59, second=59)
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.to_datetime(daily_df.index)
        d_tz = getattr(daily_df.index, "tz", None)
        if d_tz is not None:
            daily_df = daily_df[daily_df.index <= eod_naive.tz_localize(d_tz)]
        else:
            daily_df = daily_df[daily_df.index <= eod_naive]

    if daily_df is None or daily_df.empty or len(daily_df) < 15:
        return None

    close = daily_df["close"].values
    ltp = float(close[-1])
    vol = int(daily_df["volume"].iloc[-1]) if "volume" in daily_df.columns else 0

    # Monthly Resample
    m_df = scanner.resample_ohlcv(daily_df, "monthly")
    if m_df is None or m_df.empty or len(m_df) < 4:
        return None

    m_c_s = m_df["close"]
    m_ema5 = scanner.calculate_ema(m_c_s, span=5).values
    m_ema20 = scanner.calculate_ema(m_c_s, span=20).values
    m_ema50 = scanner.calculate_ema(m_c_s, span=50).values
    m_rsi = scanner.calculate_rsi(m_c_s, span=9).values
    m_rsi_ema3 = scanner.calculate_ema(pd.Series(m_rsi), span=3).values
    m_rsi_wma21 = scanner.calculate_wma(pd.Series(m_rsi), period=21).values

    m_c = float(m_c_s.iloc[-1])
    m_e5 = float(m_ema5[-1])
    m_e20 = float(m_ema20[-1])
    m_e50 = float(m_ema50[-1]) if len(m_ema50) > 0 and not np.isnan(m_ema50[-1]) else m_e20
    m_r = float(m_rsi[-1])
    m_re3 = float(m_rsi_ema3[-1])
    m_rw21 = float(m_rsi_wma21[-1])

    m_pass = (
        (m_c < m_e5) and
        (m_e5 < m_e20) and
        (m_e20 < m_e50) and
        (m_r < 50.0) and
        (m_r < m_re3) and
        (m_r < m_rw21)
    )
    if not m_pass:
        return None

    stage = 1

    # Weekly Resample
    w_df = scanner.resample_ohlcv(daily_df, "weekly")
    w_pass = False
    w_c = w_e20 = w_e50 = w_e200 = w_r = w_re3 = w_rw21 = np.nan

    if w_df is not None and not w_df.empty and len(w_df) >= 10:
        w_c_s = w_df["close"]
        w_ema20_s = scanner.calculate_ema(w_c_s, span=20).values
        w_ema50_s = scanner.calculate_ema(w_c_s, span=50).values
        w_ema200_s = scanner.calculate_ema(w_c_s, span=200).values
        w_rsi_s = scanner.calculate_rsi(w_c_s, span=9).values
        w_rsi_ema3_s = scanner.calculate_ema(pd.Series(w_rsi_s), span=3).values
        w_rsi_wma21_s = scanner.calculate_wma(pd.Series(w_rsi_s), period=21).values

        w_c = float(w_c_s.iloc[-1])
        w_e20 = float(w_ema20_s[-1])
        w_e50 = float(w_ema50_s[-1]) if len(w_ema50_s) > 0 and not np.isnan(w_ema50_s[-1]) else w_e20
        w_e200 = float(w_ema200_s[-1]) if len(w_ema200_s) > 0 and not np.isnan(w_ema200_s[-1]) else w_e50
        w_r = float(w_rsi_s[-1])
        w_re3 = float(w_rsi_ema3_s[-1])
        w_rw21 = float(w_rsi_wma21_s[-1])

        w_pass = (
            (w_c < w_e20) and
            (w_e20 < w_e50) and
            (w_e50 < w_e200) and
            (w_r < 50.0) and
            (w_r < w_re3) and
            (w_r < w_rw21)
        )
        if w_pass:
            stage = 2

    # Daily Analysis
    d_pass = False
    d_c = d_e20 = d_e50 = d_e200 = d_r = d_re3 = d_rw21 = np.nan

    if w_pass and len(daily_df) >= 15:
        d_c_s = daily_df["close"]
        d_ema20_s = scanner.calculate_ema(d_c_s, span=20).values
        d_ema50_s = scanner.calculate_ema(d_c_s, span=50).values
        d_ema200_s = scanner.calculate_ema(d_c_s, span=200).values
        d_rsi_s = scanner.calculate_rsi(d_c_s, span=9).values
        d_rsi_ema3_s = scanner.calculate_ema(pd.Series(d_rsi_s), span=3).values
        d_rsi_wma21_s = scanner.calculate_wma(pd.Series(d_rsi_s), period=21).values

        d_c = float(d_c_s.iloc[-1])
        d_e20 = float(d_ema20_s[-1])
        d_e50 = float(d_ema50_s[-1]) if len(d_ema50_s) > 0 and not np.isnan(d_ema50_s[-1]) else d_e20
        d_e200 = float(d_ema200_s[-1]) if len(d_ema200_s) > 0 and not np.isnan(d_ema200_s[-1]) else d_e50
        d_r = float(d_rsi_s[-1])
        d_re3 = float(d_rsi_ema3_s[-1])
        d_rw21 = float(d_rsi_wma21_s[-1])

        d_pass = (
            (d_c < d_e20) and
            (d_e20 < d_e50) and
            (d_e50 < d_e200) and
            (d_r < 50.0) and
            (d_r < d_re3) and
            (d_r < d_rw21)
        )
        if d_pass:
            stage = 3

    # 75-Min Analysis (Dynamic hot_intraday / Lazy Loading)
    q4_pass = False
    q4_c = q4_e20 = q4_e50 = q4_e200 = np.nan

    if d_pass:
        if intra_75_df is None:
            # Priority 0: Live hot_intraday.db stream (resampled in < 2ms)
            try:
                import hot_intraday
                if hot_intraday.has_hot_data():
                    intra_75_df = hot_intraday.get_resampled_candles(symbol, interval_minutes=75, limit=50)
            except Exception as e:
                logger.debug(f"hot_intraday 75m note for {symbol}: {e}")

            if intra_75_df is None or intra_75_df.empty:
                if intra_75_provider is not None:
                    try:
                        intra_75_df = intra_75_provider(symbol, "75-Min")
                    except TypeError:
                        intra_75_df = intra_75_provider(symbol)
                else:
                    intra_75_df = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=20)

        if intra_75_df is not None and not intra_75_df.empty and len(intra_75_df) >= 10:
            q_close = intra_75_df["close"]
            q_ema20 = scanner.calculate_ema(q_close, span=20).values
            q_ema50 = scanner.calculate_ema(q_close, span=50).values
            q_ema200 = scanner.calculate_ema(q_close, span=200).values

            q4_c = float(q_close.iloc[-1])
            q4_e20 = float(q_ema20[-1])
            q4_e50 = float(q_ema50[-1]) if len(q_ema50) > 0 and not np.isnan(q_ema50[-1]) else q4_e20
            q4_e200 = float(q_ema200[-1]) if len(q_ema200) > 0 and not np.isnan(q_ema200[-1]) else q4_e50

            q4_pass = (
                (q4_e20 <= q4_e50) and
                (q4_e50 <= q4_e200) and
                (q4_c < q4_e20)
            )
            if q4_pass:
                stage = 4

    # ─── Stage 5: 15-Min Precision Entry (Dynamic hot_intraday / Lazy Loading) ───
    q5_pass = False
    q5_c = q5_e9 = q5_e13 = q5_e20 = q5_e50 = q5_sma26 = np.nan

    if q4_pass:
        if intra_15_df is None:
            # Priority 0: Live hot_intraday.db stream (resampled in < 2ms)
            try:
                import hot_intraday
                if hot_intraday.has_hot_data():
                    intra_15_df = hot_intraday.get_resampled_candles(symbol, interval_minutes=15, limit=50)
            except Exception as e:
                logger.debug(f"hot_intraday 15m note for {symbol}: {e}")

            if intra_15_df is None or intra_15_df.empty:
                if intra_15_provider is not None:
                    try:
                        intra_15_df = intra_15_provider(symbol, "15-Min")
                    except TypeError:
                        intra_15_df = intra_15_provider(symbol)
                else:
                    intra_15_df = parquet_loader.ensure_symbol_custom_minute_candles(symbol, interval_minutes=15, min_bars=10)

        if intra_15_df is not None and not intra_15_df.empty and len(intra_15_df) >= 5:
            q15_close = intra_15_df["close"]
            q15_open  = intra_15_df["open"]
            q15_e9    = scanner.calculate_ema(q15_close, span=9).values
            q15_e13   = scanner.calculate_ema(q15_close, span=13).values
            q15_e20   = scanner.calculate_ema(q15_close, span=20).values
            q15_e50   = scanner.calculate_ema(q15_close, span=50).values
            q15_e200  = scanner.calculate_ema(q15_close, span=200).values
            q15_sma26 = q15_close.rolling(window=26, min_periods=1).mean().values

            q5_c    = float(q15_close.iloc[-1])
            q5_e9   = float(q15_e9[-1])
            q5_e13  = float(q15_e13[-1])
            q5_e20  = float(q15_e20[-1])
            q5_e50  = float(q15_e50[-1]) if not np.isnan(q15_e50[-1]) else q5_e20
            q5_sma26 = float(q15_sma26[-1])

            # Daily EMA_20 for cross-TF clause 24
            d_e20_for_15 = d_e20 if not np.isnan(d_e20) else q5_e20

            # Clause 24: 15m Close < Daily EMA_20 (cross-TF)
            c24 = (q5_c < d_e20_for_15)
            # Clause 25: previous 15m bar was bearish
            c25 = len(q15_close) >= 2 and (float(q15_close.iloc[-2]) < float(q15_open.iloc[-2]))
            # Clause 26: current 15m bar opened below prior 15m bar close (gap-down)
            c26 = len(q15_close) >= 2 and (float(q15_open.iloc[-1]) < float(q15_close.iloc[-2]))
            # Clause 27: current 15m bar is bearish
            c27 = (q5_c < float(q15_open.iloc[-1]))
            # Clause 28: 15m EMA_20 < EMA_50
            c28 = (q5_e20 < q5_e50)
            # Clause 29: 15m EMA_50 < EMA_200
            c29 = (q5_e50 < float(q15_e200[-1])) if not np.isnan(q15_e200[-1]) else False
            # Clause 30-32: MA squeeze (% diff <= 0.01%)
            def _pct_diff(a, b): return abs(a - b) / abs(a) * 100.0 if a != 0 else 0.0
            c30 = _pct_diff(q5_e9, q5_e13) <= 0.01
            c31 = _pct_diff(q5_e13, q5_e20) <= 0.01
            c32 = _pct_diff(q5_e20, q5_sma26) <= 0.01

            q5_pass = c24 and c25 and c26 and c27 and c28 and c29 and c30 and c31 and c32
            if q5_pass:
                stage = 5

    stage_labels = {
        5: "🚨 Stage 5 (FULL SIGNAL M+W+D+75m+15m)",
        4: "🏆 Stage 4 (Full Breakdown M+W+D+75m)",
        3: "🚀 Stage 3 (Daily Breakdown)",
        2: "🟢 Stage 2 (Weekly Breakdown)",
        1: "🟡 Stage 1 (Monthly Breakdown)"
    }

    as_of_str = str(daily_df.index[-1])[:10]
    return {
        "Symbol": symbol,
        "Scan Date": as_of_str,
        "LTP": round(ltp, 2),
        "Volume": vol,
        "Stage": stage,
        "Breakdown Stage": stage_labels[stage],
        "Monthly": "✅ PASS",
        "Weekly": "✅ PASS" if w_pass else "❌ FAIL",
        "Daily": "✅ PASS" if d_pass else "❌ FAIL",
        "75-Min": "✅ PASS" if q4_pass else "❌ FAIL",
        "15-Min": "✅ PASS" if q5_pass else "❌ FAIL",
        "M_RSI": round(m_r, 1),
        "M_EMA5": round(m_e5, 2),
        "M_EMA20": round(m_e20, 2),
        "W_RSI": round(w_r, 1) if not np.isnan(w_r) else np.nan,
        "W_EMA20": round(w_e20, 2) if not np.isnan(w_e20) else np.nan,
        "D_RSI": round(d_r, 1) if not np.isnan(d_r) else np.nan,
        "D_EMA20": round(d_e20, 2) if not np.isnan(d_e20) else np.nan,
        "75m_EMA20": round(q4_e20, 2) if not np.isnan(q4_e20) else np.nan,
        "75m_EMA50": round(q4_e50, 2) if not np.isnan(q4_e50) else np.nan,
        "15m_Close": round(q5_c, 2) if not np.isnan(q5_c) else np.nan,
        "15m_EMA9": round(q5_e9, 2) if not np.isnan(q5_e9) else np.nan,
        "15m_EMA20": round(q5_e20, 2) if not np.isnan(q5_e20) else np.nan,
        "15m_EMA50": round(q5_e50, 2) if not np.isnan(q5_e50) else np.nan,
    }



def run_waterfall_scan(
    symbols: List[str],
    as_of_date: Optional[Any] = None,
    as_of_time: Optional[str] = None,
    data_provider_fn=None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes the complete Chartink 'positional-scan-364' waterfall screening across symbols.
    Supports historical date and time backtesting via `as_of_date` and `as_of_time`.
    Returns filtered dataframes for each stage.
    """
    results = []
    total_scanned = len(symbols)
    if total_scanned == 0:
        empty_df = pd.DataFrame()
        return {
            "all_waterfall": empty_df,
            "stage_4_full": empty_df,
            "stage_3_daily": empty_df,
            "stage_2_weekly": empty_df,
            "stage_1_monthly": empty_df,
            "counts": {"total": 0, "as_of_date": str(as_of_date) if as_of_date else "Latest Live", "as_of_time": str(as_of_time) if as_of_time else "15:30", "m_pass": 0, "w_pass": 0, "d_pass": 0, "q4_pass": 0}
        }

    # 1. High-performance batch retrieval of daily candles in a single DuckDB query
    batch_daily = {}
    if not data_provider_fn and total_scanned > 1:
        try:
            batch_daily = database.get_batch_candles_df(symbols)
        except Exception as e:
            logger.warning(f"Batch daily load error, fallback to individual queries: {e}")

    def _eval_sym(sym: str) -> Optional[Dict[str, Any]]:
        clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
        daily_df = None
        if data_provider_fn:
            daily_df = data_provider_fn(sym, "Daily")
        else:
            daily_df = batch_daily.get(clean)
            if (daily_df is None or daily_df.empty) and not batch_daily:
                daily_df = database.get_candles_df(sym)

        if daily_df is None or daily_df.empty or len(daily_df) < 15:
            return None

        # 2. Lazy evaluation: 75m intraday is only fetched if Stage 3 passes or intraday slice requested
        return evaluate_stock_waterfall(
            sym,
            daily_df,
            intra_75_df=None,
            as_of_date=as_of_date,
            as_of_time=as_of_time,
            intra_75_provider=data_provider_fn
        )

    completed_count = 0
    max_workers = min(32, max(4, (os.cpu_count() or 4) * 4)) if total_scanned > 4 else 1
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {executor.submit(_eval_sym, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_scanned, sym)
                    except Exception:
                        pass
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    logger.error(f"Error evaluating {sym}: {e}")
    else:
        for sym in symbols:
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, total_scanned, sym)
                except Exception:
                    pass
            res = _eval_sym(sym)
            if res is not None:
                results.append(res)

    if not results:
        empty_df = pd.DataFrame()
        return {
            "all_waterfall": empty_df,
            "stage_4_full": empty_df,
            "stage_3_daily": empty_df,
            "stage_2_weekly": empty_df,
            "stage_1_monthly": empty_df,
            "counts": {"total": total_scanned, "as_of_date": str(as_of_date) if as_of_date else "Latest Live", "as_of_time": str(as_of_time) if as_of_time else "15:30", "m_pass": 0, "w_pass": 0, "d_pass": 0, "q4_pass": 0}
        }

    sort_cols = ["Stage", "1D Return (%)"]
    if "Return Since Scan (%)" in results[0] and results[0]["Return Since Scan (%)"] is not None:
        sort_cols = ["Stage", "Return Since Scan (%)"]
    df_all = pd.DataFrame(results).sort_values(by=sort_cols, ascending=[False, False])
    df_s4 = df_all[df_all["Stage"] == 4].copy()
    df_s3 = df_all[df_all["Stage"] >= 3].copy()
    df_s2 = df_all[df_all["Stage"] >= 2].copy()
    df_s1 = df_all[df_all["Stage"] >= 1].copy()

    counts = {
        "total": total_scanned,
        "as_of_date": str(as_of_date) if as_of_date else "Latest Live",
        "as_of_time": str(as_of_time) if as_of_time else "15:30",
        "m_pass": len(df_s1),
        "w_pass": len(df_s2),
        "d_pass": len(df_s3),
        "q4_pass": len(df_s4)
    }

    return {
        "all_waterfall": df_all,
        "stage_4_full": df_s4,
        "stage_3_daily": df_s3,
        "stage_2_weekly": df_s2,
        "stage_1_monthly": df_s1,
        "counts": counts
    }


# ==========================================
# PRE-BUILT POPULAR CHARTINK SCREENER PRESETS
# ==========================================
def get_screener_preset(preset_name: str) -> ScreenerConfig:
    """Returns pre-configured ScreenerConfig for iconic Chartink screeners."""
    p_lower = preset_name.lower()

    if "19122704" in p_lower or "intraday scan" in p_lower or "intraday trade" in p_lower:
        return ScreenerConfig(
            name="⚡ Chartink Intraday Scan (19122704 - 75m+15m Multi-TF Breakdown)",
            logic="ALL",
            clauses=[
                # ── Stage 1: Monthly Macro Breakdown ──────────────────────────────
                ScreenerClause(timeframe="Monthly", lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_5"),
                ScreenerClause(timeframe="Monthly", lhs="EMA_5",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Monthly", lhs="EMA_20", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="Monthly", lhs="RSI_9",  operator="<",  rhs_type="Number",    rhs_value=50.0),
                ScreenerClause(timeframe="Monthly", lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_EMA3"),
                ScreenerClause(timeframe="Monthly", lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_WMA21"),
                # ── Stage 2: Weekly Breakdown ──────────────────────────────────────
                ScreenerClause(timeframe="Weekly",  lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Weekly",  lhs="EMA_20", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="Weekly",  lhs="EMA_50", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_200"),
                ScreenerClause(timeframe="Weekly",  lhs="RSI_9",  operator="<",  rhs_type="Number",    rhs_value=50.0),
                ScreenerClause(timeframe="Weekly",  lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_EMA3"),
                ScreenerClause(timeframe="Weekly",  lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_WMA21"),
                # ── Stage 3: Daily Breakdown ───────────────────────────────────────
                ScreenerClause(timeframe="Daily",   lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily",   lhs="EMA_20", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="Daily",   lhs="EMA_50", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_200"),
                ScreenerClause(timeframe="Daily",   lhs="RSI_9",  operator="<",  rhs_type="Number",    rhs_value=50.0),
                ScreenerClause(timeframe="Daily",   lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_EMA3"),
                ScreenerClause(timeframe="Daily",   lhs="RSI_9",  operator="<",  rhs_type="Indicator", rhs_indicator="RSI_WMA21"),
                # ── Stage 4: 75-Min Execution Trigger ────────────────────────────
                ScreenerClause(timeframe="75-Min",  lhs="EMA_20", operator="<=", rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="75-Min",  lhs="EMA_50", operator="<=", rhs_type="Indicator", rhs_indicator="EMA_200"),
                ScreenerClause(timeframe="75-Min",  lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_20"),
                # ── Stage 5: 15-Min Intraday Precision Entry ─────────────────────
                # Clause 24: 15min Close < Daily EMA_20  (cross-timeframe)
                ScreenerClause(timeframe="15-Min",  lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="EMA_20",
                               rhs_timeframe="Daily"),
                # Clause 25: previous 15m bar was bearish (offset=1 means bar[-2])
                ScreenerClause(timeframe="15-Min",  lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="Open",
                               offset=1, rhs_offset=1),
                # Clause 26: current 15m bar opened below prior 15m bar Close (gap-down open)
                ScreenerClause(timeframe="15-Min",  lhs="Open",   operator="<",  rhs_type="Indicator", rhs_indicator="Close",
                               offset=0, rhs_offset=1),
                # Clause 27: current 15m bar is bearish (Close < Open)
                ScreenerClause(timeframe="15-Min",  lhs="Close",  operator="<",  rhs_type="Indicator", rhs_indicator="Open"),
                # Clause 28: 15m EMA_20 < EMA_50
                ScreenerClause(timeframe="15-Min",  lhs="EMA_20", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_50"),
                # Clause 29: 15m EMA_50 < EMA_200
                ScreenerClause(timeframe="15-Min",  lhs="EMA_50", operator="<",  rhs_type="Indicator", rhs_indicator="EMA_200"),
                # Clause 30: MA squeeze  abs(EMA_9 - EMA_13)/EMA_9*100 <= 0.01%
                ScreenerClause(timeframe="15-Min",  lhs="EMA_9",  operator="abs_pct_lte", rhs_type="Number",
                               rhs_value=0.01, rhs_indicator="EMA_13"),
                # Clause 31: abs(EMA_13 - EMA_20)/EMA_13*100 <= 0.01%
                ScreenerClause(timeframe="15-Min",  lhs="EMA_13", operator="abs_pct_lte", rhs_type="Number",
                               rhs_value=0.01, rhs_indicator="EMA_20"),
                # Clause 32: abs(EMA_20 - SMA_26)/EMA_20*100 <= 0.01%
                ScreenerClause(timeframe="15-Min",  lhs="EMA_20", operator="abs_pct_lte", rhs_type="Number",
                               rhs_value=0.01, rhs_indicator="SMA_26"),
            ]
        )

    elif "rsi momentum" in p_lower or "rsi > 60" in p_lower:
        return ScreenerConfig(
            name="🚀 RSI Momentum Surge",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=60.0),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="EMA_20", operator=">", rhs_type="Indicator", rhs_indicator="EMA_50"),
            ]
        )
    elif "52" in p_lower or "breakout" in p_lower or "year high" in p_lower:
        return ScreenerConfig(
            name="🏔️ 52-Week / Multi-Year High Breakout",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">=", rhs_type="Indicator", rhs_indicator="52_Week_High"),
                ScreenerClause(timeframe="Daily", lhs="Volume", operator=">", rhs_type="Indicator", rhs_indicator="Vol_SMA_20", multiplier=1.5),
            ]
        )
    elif "supertrend" in p_lower:
        return ScreenerConfig(
            name="🏹 SuperTrend Fresh Bullish Reversal",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Close", operator="crossed_above", rhs_type="Indicator", rhs_indicator="SuperTrend"),
                ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=50.0),
            ]
        )
    elif "triple ema" in p_lower or "ribbon" in p_lower:
        return ScreenerConfig(
            name="🌊 Triple EMA Bullish Stack (9 > 20 > 50)",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="EMA_9", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="EMA_20", operator=">", rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_9"),
            ]
        )
    elif "volume shocker" in p_lower or "volume" in p_lower:
        return ScreenerConfig(
            name="💥 Volume Shocker & Price Breakout",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Volume", operator=">", rhs_type="Indicator", rhs_indicator="Vol_SMA_20", multiplier=2.5),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="Open"),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="Prev_Day_High"),
            ]
        )
    elif "hilega" in p_lower:
        return ScreenerConfig(
            name="⚡ Hilega Milega Bullish Alignment (NK Sir)",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="RSI_9", operator=">=", rhs_type="Number", rhs_value=50.0),
                ScreenerClause(timeframe="Daily", lhs="RSI_EMA3", operator=">=", rhs_type="Indicator", rhs_indicator="RSI_WMA21"),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
            ]
        )
    elif "pullback" in p_lower or "dip" in p_lower:
        return ScreenerConfig(
            name="🎯 Pullback to 20 EMA (Dip Buying Setup)",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Low", operator="<=", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="EMA_20", operator=">", rhs_type="Indicator", rhs_indicator="EMA_50"),
                ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=50.0),
            ]
        )
    elif "oversold" in p_lower:
        return ScreenerConfig(
            name="🔻 Oversold Bounce Setup (RSI < 30)",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="RSI_14", operator="<", rhs_type="Number", rhs_value=30.0),
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="Open"),
            ]
        )
    else:
        # Default custom
        return ScreenerConfig(
            name="Custom Screener",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=50.0),
            ]
        )


def parse_chartink_query(query: str) -> List[ScreenerClause]:
    """
    Parses a plain-text Chartink query, screenshot OCR text, or condition string into a list of ScreenerClauses.
    Supports:
      - 'Daily Close > Daily 20 EMA'
      - '[ Latest ] Close Greater than [ Latest ] 20 EMA'
      - '[ Latest ] Volume Greater than [ Latest ] 20 SMA Volume * 1.5'
      - 'Open <= 20 EMA and Close >= 20 EMA'
      - 'Daily RSI(14) > 60'
      - 'Weekly Close crossed above Weekly 200 EMA'
    """
    clauses = []
    lines = [l.strip() for l in query.splitlines() if l.strip()]

    # Normalize split if line contains ' and ' between two complete conditions
    expanded_lines = []
    for line in lines:
        # Check if line contains ' and ' between statements (e.g. Open <= 20 EMA and Close >= 20 EMA)
        sub_parts = re.split(r"\s+\band\b\s+", line, flags=re.IGNORECASE)
        # Only split if both parts look like expressions (have an operator or comparison)
        if len(sub_parts) > 1 and all(re.search(r"(>|<|=|above|below|greater|less)", p, re.I) for p in sub_parts):
            expanded_lines.extend(sub_parts)
        else:
            expanded_lines.append(line)

    for line in expanded_lines:
        # Strip brackets around tokens e.g. [ Latest ] -> Latest
        cleaned_line = re.sub(r"\[\s*([^\]]+?)\s*\]", r" \1 ", line)
        cleaned_line = re.sub(r"\(\s*(latest|daily|weekly|monthly|intraday|\-?\d+\s*day[s]?\s*ago)\s*\)", r" \1 ", cleaned_line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r"\s+", " ", cleaned_line).strip().rstrip(".;,")

        tf = "Daily"
        if re.search(r"\bweekly\b", cleaned_line, re.IGNORECASE):
            tf = "Weekly"
        elif re.search(r"\bmonthly\b", cleaned_line, re.IGNORECASE):
            tf = "Monthly"
        elif re.search(r"\b75\b|\b75m\b|\bintraday\b|\bhourly\b|\bhour\b", cleaned_line, re.IGNORECASE):
            tf = "75-Min"

        # Check for multiplier at end (e.g. '* 1.5' or 'number 2' or '* 2')
        multiplier = 1.0
        mult_match = re.search(r"(\*|multiplied\s+by|x)\s*([\d\.]+)\s*$", cleaned_line, re.IGNORECASE)
        if mult_match:
            try:
                multiplier = float(mult_match.group(2))
                cleaned_line = cleaned_line[:mult_match.start()].strip()
            except ValueError:
                pass

        # Split line by operator (longest match first)
        op_pattern = (
            r"(crossed\s+above|crosses\s+above|crossed\s+over|"
            r"crossed\s+below|crosses\s+below|crossed\s+under|"
            r"greater\s+than\s+or\s+equal\s+to|greater\s+than\s+equal|>=|=>|"
            r"less\s+than\s+or\s+equal\s+to|less\s+than\s+equal|<=|=<|"
            r"greater\s+than|higher\s+than|above|>|"
            r"less\s+than|lower\s+than|below|<|"
            r"equal\s+to|equals|==|=)"
        )
        m_op = re.search(op_pattern, cleaned_line, re.IGNORECASE)
        if m_op:
            op_raw = m_op.group(1).lower()
            if "crossed" in op_raw or "crosses" in op_raw:
                op = "crossed_above" if "above" in op_raw or "over" in op_raw else "crossed_below"
            elif "greater" in op_raw and ("equal" in op_raw or "=" in op_raw):
                op = ">="
            elif "less" in op_raw and ("equal" in op_raw or "=" in op_raw):
                op = "<="
            elif ">=" in op_raw or "=>" in op_raw:
                op = ">="
            elif "<=" in op_raw or "=<" in op_raw:
                op = "<="
            elif "greater" in op_raw or "above" in op_raw or "higher" in op_raw or ">" in op_raw:
                op = ">"
            elif "less" in op_raw or "below" in op_raw or "lower" in op_raw or "<" in op_raw:
                op = "<"
            elif "equal" in op_raw or "==" in op_raw or "=" in op_raw:
                op = "=="
            else:
                op = ">"

            lhs_part = cleaned_line[:m_op.start()].strip()
            rhs_part = cleaned_line[m_op.end():].strip()
        else:
            op = ">"
            lhs_part = cleaned_line
            rhs_part = ""

        def _detect_ind(text: str, default: str = "Close") -> str:
            t = text.lower()
            # 52-Week High / Low
            if "52" in t and ("high" in t or "hi" in t):
                return "52_Week_High"
            if "52" in t and ("low" in t or "lo" in t):
                return "52_Week_Low"

            # Previous Day High / Low / Close
            if "prev" in t or "1 day ago" in t or "-1 day" in t:
                if "high" in t:
                    return "Prev_Day_High"
                if "low" in t:
                    return "Prev_Day_Low"
                if "close" in t:
                    return "Prev_Day_Close"

            # SuperTrend
            if "supertrend" in t or "super_trend" in t:
                return "SuperTrend"

            # MACD
            if "macd" in t:
                if "signal" in t:
                    return "MACD_Signal"
                if "hist" in t:
                    return "MACD_Hist"
                return "MACD_Line"

            # Volume and Vol SMA
            if "volume" in t:
                if "sma" in t or "ma" in t:
                    return "Vol_SMA_20"
                return "Volume"

            # RSI and variations
            if "rsi" in t:
                if "ema" in t or ("3" in t and "rsi" in t):
                    return "RSI_EMA3"
                if "wma" in t or ("21" in t and "rsi" in t):
                    return "RSI_WMA21"
                m_rsi = re.search(r"rsi\s*\([^)]*?(\d+)[^)]*?\)|rsi\s*(\d+)", t)
                if m_rsi:
                    p = int(m_rsi.group(1) or m_rsi.group(2))
                    return f"RSI_{p}"
                return "RSI_14"

            # EMAs
            if "ema" in t:
                m_ema = re.search(r"ema\s*\([^)]*?(\d+)[^)]*?\)|(\d+)\s*ema|ema\s*(\d+)", t)
                if m_ema:
                    p = int(m_ema.group(1) or m_ema.group(2) or m_ema.group(3))
                    return f"EMA_{p}"
                return "EMA_20"

            # SMAs
            if "sma" in t:
                m_sma = re.search(r"sma\s*\([^)]*?(\d+)[^)]*?\)|(\d+)\s*sma|sma\s*(\d+)", t)
                if m_sma:
                    p = int(m_sma.group(1) or m_sma.group(2) or m_sma.group(3))
                    return f"SMA_{p}"
                return "SMA_20"


            # Standard price items
            if "close" in t:
                return "Close"
            elif "open" in t:
                return "Open"
            elif "high" in t:
                return "High"
            elif "low" in t:
                return "Low"

            return default

        lhs = _detect_ind(lhs_part, default="Close")

        # Detect RHS: number or indicator
        rhs_clean = rhs_part.strip().rstrip(".;,")
        num_match = re.search(r"^(?:number\s*\(?\s*|₹\s*|\$\s*)?([\d\.]+)\s*\)?$", rhs_clean, re.IGNORECASE)
        has_ind_word = any(k in rhs_clean.lower() for k in ["ema", "sma", "rsi", "close", "open", "high", "low", "volume", "supertrend", "macd"])
        if num_match and not has_ind_word:
            clauses.append(ScreenerClause(
                timeframe=tf,
                lhs=lhs,
                operator=op,
                rhs_type="Number",
                rhs_value=float(num_match.group(1)),
                multiplier=multiplier
            ))
        else:
            rhs_ind = _detect_ind(rhs_clean, default="EMA_20")
            clauses.append(ScreenerClause(
                timeframe=tf,
                lhs=lhs,
                operator=op,
                rhs_type="Indicator",
                rhs_indicator=rhs_ind,
                multiplier=multiplier
            ))

    return clauses
