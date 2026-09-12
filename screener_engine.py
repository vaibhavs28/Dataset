"""
screener_engine.py
------------------
High-performance Stock Screener & Filter Engine modeled after Chartink.
Supports multi-timeframe condition evaluation (Daily, Weekly, Monthly, 75m),
complex operators (>, <, crossed_above, crossed_below), custom indicators,
pre-built Chartink screener presets, and Chartink query string parsing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re
import numpy as np
import pandas as pd

import scanner
import database
import parquet_loader
import config


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
    "RSI_EMA3",
    "RSI_WMA21",
    "EMA_9",
    "EMA_13",
    "EMA_20",
    "EMA_26",
    "EMA_50",
    "EMA_200",
    "SMA_20",
    "SMA_50",
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
    "crossed_below"
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
    out["RSI_EMA3"] = scanner.calculate_ema(out["RSI_9"], span=3)
    out["RSI_WMA21"] = scanner.calculate_wma(out["RSI_9"], period=21)

    # EMAs
    out["EMA_9"] = scanner.calculate_ema(close, span=9)
    out["EMA_13"] = scanner.calculate_ema(close, span=13)
    out["EMA_20"] = scanner.calculate_ema(close, span=20)
    out["EMA_26"] = scanner.calculate_ema(close, span=26)
    out["EMA_50"] = scanner.calculate_ema(close, span=50)
    out["EMA_200"] = scanner.calculate_ema(close, span=200)

    # SMAs
    out["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    out["SMA_50"] = close.rolling(window=50, min_periods=1).mean()

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


def evaluate_clause(df_ind: pd.DataFrame, clause: ScreenerClause) -> Tuple[bool, float, float]:
    """
    Evaluates a single ScreenerClause on the indicator dataframe.
    Returns: (passed: bool, lhs_value: float, rhs_value: float)
    """
    if df_ind.empty or len(df_ind) < 3:
        return False, 0.0, 0.0

    lhs_col = clause.lhs
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
        if rhs_col not in df_ind.columns:
            matched_r = [c for c in df_ind.columns if c.lower() == rhs_col.lower()]
            if matched_r:
                rhs_col = matched_r[0]
            else:
                return False, val_lhs_curr, 0.0
        s_rhs = df_ind[rhs_col] * clause.multiplier
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

    return passed, val_lhs_curr, val_rhs_curr


def evaluate_stock(
    symbol: str,
    timeframe_dfs: Dict[str, pd.DataFrame],
    cfg: ScreenerConfig
) -> Optional[Dict[str, Any]]:
    """
    Evaluates all clauses for a single symbol across required timeframes.
    Returns dictionary with match info or None if criteria not met.
    """
    if not cfg.clauses:
        return None

    clause_results = []
    indicator_summary = {}

    daily_df = timeframe_dfs.get("Daily")
    if daily_df is None or daily_df.empty:
        return None

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

        passed, lhs_val, rhs_val = evaluate_clause(df_tf, clause)
        clause_results.append(passed)

        # Store for display
        label = f"Rule #{i+1}: {clause.timeframe} {clause.lhs} {clause.operator} {clause.rhs_indicator if clause.rhs_type=='Indicator' else clause.rhs_value}"
        indicator_summary[f"C{i+1}_Result"] = "✅" if passed else "❌"
        indicator_summary[f"C{i+1}_LHS"] = round(lhs_val, 2)
        indicator_summary[f"C{i+1}_RHS"] = round(rhs_val, 2)

    if cfg.logic == "ALL":
        overall_match = all(clause_results)
    else:  # ANY
        overall_match = any(clause_results)

    if not overall_match:
        return None

    res = {
        "Symbol": symbol,
        "LTP": round(ltp, 2),
        "Change_%": round(change_pct, 2),
        "Volume": vol,
        "Clauses_Passed": sum(clause_results),
        "Total_Clauses": len(cfg.clauses),
    }
    res.update(indicator_summary)
    return res


def run_screen(
    symbols: List[str],
    cfg: ScreenerConfig,
    as_of_date: Optional[Any] = None,
    data_provider_fn=None
) -> pd.DataFrame:
    """
    Executes a screener across a list of symbols.
    Supports historical date backtesting via `as_of_date`.
    data_provider_fn(symbol, timeframe) returns raw OHLCV DataFrame.
    """
    matches = []
    needed_tfs = set(c.timeframe for c in cfg.clauses)
    if not needed_tfs:
        needed_tfs = {"Daily"}

    for sym in symbols:
        tf_dfs = {}
        daily_raw = None

        # Fetch daily first
        if data_provider_fn:
            daily_raw = data_provider_fn(sym, "Daily")
        else:
            daily_raw = database.get_candles_df(sym)

        if daily_raw is None or daily_raw.empty or len(daily_raw) < 5:
            continue

        if not isinstance(daily_raw.index, pd.DatetimeIndex):
            daily_raw = daily_raw.copy()
            daily_raw.index = pd.to_datetime(daily_raw.index)

        full_daily = daily_raw
        if as_of_date is not None:
            target_dt = pd.to_datetime(as_of_date)
            eod_naive = target_dt.replace(hour=23, minute=59, second=59)
            if daily_raw.index.tz is not None:
                daily_raw = daily_raw[daily_raw.index <= eod_naive.tz_localize(daily_raw.index.tz)]
            else:
                daily_raw = daily_raw[daily_raw.index <= eod_naive]
            if daily_raw.empty or len(daily_raw) < 5:
                continue

        # Prepare Daily indicators
        tf_dfs["Daily"] = compute_screener_indicators(daily_raw)

        # Weekly resample
        if "Weekly" in needed_tfs:
            w_raw = scanner.resample_ohlcv(daily_raw, "weekly")
            tf_dfs["Weekly"] = compute_screener_indicators(w_raw)

        # Monthly resample
        if "Monthly" in needed_tfs:
            m_raw = scanner.resample_ohlcv(daily_raw, "monthly")
            tf_dfs["Monthly"] = compute_screener_indicators(m_raw)

        # 75-Min intraday
        if "75-Min" in needed_tfs:
            intra_raw = None
            if data_provider_fn:
                intra_raw = data_provider_fn(sym, "75-Min")
            else:
                intra_raw = parquet_loader.ensure_symbol_75m_candles(sym, min_bars=50)
            if intra_raw is not None and not intra_raw.empty:
                if not isinstance(intra_raw.index, pd.DatetimeIndex):
                    intra_raw = intra_raw.copy()
                    intra_raw.index = pd.to_datetime(intra_raw.index)
                if as_of_date is not None:
                    target_dt = pd.to_datetime(as_of_date)
                    eod_naive = target_dt.replace(hour=23, minute=59, second=59)
                    if intra_raw.index.tz is not None:
                        intra_raw = intra_raw[intra_raw.index <= eod_naive.tz_localize(intra_raw.index.tz)]
                    else:
                        intra_raw = intra_raw[intra_raw.index <= eod_naive]
                if intra_raw is not None and not intra_raw.empty:
                    tf_dfs["75-Min"] = compute_screener_indicators(intra_raw)

        eval_res = evaluate_stock(sym, tf_dfs, cfg)
        if eval_res is not None:
            as_of_str = str(daily_raw.index[-1])[:10]
            eval_res["Scan_Date"] = as_of_str
            if len(full_daily) > len(daily_raw):
                future_close = float(full_daily["close"].iloc[-1])
                curr_p = float(eval_res.get("LTP", 0.0))
                if curr_p > 0:
                    eval_res["Return_Since_Scan_%"] = round(((future_close - curr_p) / curr_p) * 100.0, 2)
                    eval_res["Latest_Close"] = round(future_close, 2)
            matches.append(eval_res)

    if not matches:
        return pd.DataFrame()

    df_res = pd.DataFrame(matches)
    # Sort by % change descending
    if "Change_%" in df_res.columns:
        df_res = df_res.sort_values(by="Change_%", ascending=False)
    return df_res


def evaluate_stock_waterfall(
    symbol: str,
    daily_df: pd.DataFrame,
    intra_75_df: Optional[pd.DataFrame] = None,
    as_of_date: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates exact Chartink 'positional-scan-364' rules via a strict Waterfall Model:
    Monthly Pass ➔ Weekly Pass ➔ Daily Pass ➔ 75-Min Pass.
    If any stage fails, the stock cannot progress to subsequent stages.
    If Monthly fails, the stock is completely excluded (returns None).
    Supports historical date slicing if as_of_date is provided.
    """
    if daily_df is None or daily_df.empty:
        return None

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df = daily_df.copy()
        daily_df.index = pd.to_datetime(daily_df.index)

    full_daily = daily_df
    latest_market_close = float(full_daily["close"].iloc[-1])

    if as_of_date is not None:
        target_dt = pd.to_datetime(as_of_date)
        eod_naive = target_dt.replace(hour=23, minute=59, second=59)
        if daily_df.index.tz is not None:
            daily_df = daily_df[daily_df.index <= eod_naive.tz_localize(daily_df.index.tz)]
        else:
            daily_df = daily_df[daily_df.index <= eod_naive]

        if intra_75_df is not None and not intra_75_df.empty:
            if not isinstance(intra_75_df.index, pd.DatetimeIndex):
                intra_75_df = intra_75_df.copy()
                intra_75_df.index = pd.to_datetime(intra_75_df.index)
            if intra_75_df.index.tz is not None:
                intra_75_df = intra_75_df[intra_75_df.index <= eod_naive.tz_localize(intra_75_df.index.tz)]
            else:
                intra_75_df = intra_75_df[intra_75_df.index <= eod_naive]

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
    if d_pass and intra_75_df is not None and not intra_75_df.empty and len(intra_75_df) >= 10:
        q_close = intra_75_df["close"]
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


def run_waterfall_scan(
    symbols: List[str],
    as_of_date: Optional[Any] = None,
    data_provider_fn=None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes the complete Chartink 'positional-scan-364' waterfall screening across symbols.
    Supports historical date backtesting via `as_of_date`.
    Returns filtered dataframes for each stage.
    """
    results = []
    total_scanned = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(i + 1, total_scanned, sym)

        daily_df = None
        if data_provider_fn:
            daily_df = data_provider_fn(sym, "Daily")
        else:
            daily_df = database.get_candles_df(sym)

        if daily_df is None or daily_df.empty or len(daily_df) < 15:
            continue

        intra_75_df = None
        if data_provider_fn:
            intra_75_df = data_provider_fn(sym, "75-Min")
        else:
            intra_75_df = parquet_loader.ensure_symbol_75m_candles(sym, min_bars=20)

        eval_res = evaluate_stock_waterfall(sym, daily_df, intra_75_df, as_of_date=as_of_date)
        if eval_res is not None:
            results.append(eval_res)

    if not results:
        empty_df = pd.DataFrame()
        return {
            "all_waterfall": empty_df,
            "stage_4_full": empty_df,
            "stage_3_daily": empty_df,
            "stage_2_weekly": empty_df,
            "stage_1_monthly": empty_df,
            "counts": {"total": total_scanned, "as_of_date": str(as_of_date) if as_of_date else "Latest Live", "m_pass": 0, "w_pass": 0, "d_pass": 0, "q4_pass": 0}
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

    if "rsi momentum" in p_lower or "rsi > 60" in p_lower:
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
    Parses a plain-text Chartink query or condition line into a list of ScreenerClauses.
    Example inputs:
      - 'Daily Close > Daily 20 EMA'
      - 'Daily RSI(14) > 60'
      - 'Weekly Close crossed above Weekly 200 EMA'
    """
    clauses = []
    lines = [l.strip() for l in query.splitlines() if l.strip()]

    for line in lines:
        tf = "Daily"
        if re.search(r"\bweekly\b", line, re.IGNORECASE):
            tf = "Weekly"
        elif re.search(r"\bmonthly\b", line, re.IGNORECASE):
            tf = "Monthly"
        elif re.search(r"\b75\b", line, re.IGNORECASE) or re.search(r"\bintraday\b", line, re.IGNORECASE):
            tf = "75-Min"

        # Split line by operator
        op_pattern = r"(crossed\s+above|crosses\s+above|crossed\s+below|crosses\s+below|>=|<=|>|<|==)"
        m_op = re.search(op_pattern, line, re.IGNORECASE)
        if m_op:
            op_raw = m_op.group(1).lower()
            if "above" in op_raw:
                op = "crossed_above"
            elif "below" in op_raw:
                op = "crossed_below"
            else:
                op = op_raw
            lhs_part = line[:m_op.start()].strip()
            rhs_part = line[m_op.end():].strip()
        else:
            op = ">"
            lhs_part = line
            rhs_part = ""

        def _detect_ind(text: str, default: str = "Close") -> str:
            t = text.lower()
            if "close" in t:
                return "Close"
            elif "open" in t:
                return "Open"
            elif "high" in t and "52" not in t:
                return "High"
            elif "low" in t:
                return "Low"
            elif "volume" in t:
                return "Volume"
            elif "52" in t and "high" in t:
                return "52_Week_High"
            elif "52" in t and "low" in t:
                return "52_Week_Low"
            elif "supertrend" in t:
                return "SuperTrend"
            elif "rsi" in t:
                if "9" in t:
                    return "RSI_9"
                return "RSI_14"
            elif "ema" in t:
                m_ema = re.search(r"(\d+)\s*ema|ema\s*(\d+)", t)
                if m_ema:
                    p = m_ema.group(1) or m_ema.group(2)
                    return f"EMA_{p}"
                return "EMA_20"
            elif "sma" in t:
                m_sma = re.search(r"(\d+)\s*sma|sma\s*(\d+)", t)
                if m_sma:
                    p = m_sma.group(1) or m_sma.group(2)
                    return f"SMA_{p}"
                return "SMA_20"
            return default

        lhs = _detect_ind(lhs_part, default="Close")

        # Detect RHS: number or indicator
        num_match = re.match(r"^[\d\.]+$", rhs_part.strip())
        if num_match:
            clauses.append(ScreenerClause(timeframe=tf, lhs=lhs, operator=op, rhs_type="Number", rhs_value=float(rhs_part.strip())))
        else:
            rhs_ind = _detect_ind(rhs_part, default="EMA_20")
            clauses.append(ScreenerClause(timeframe=tf, lhs=lhs, operator=op, rhs_type="Indicator", rhs_indicator=rhs_ind))

    return clauses
