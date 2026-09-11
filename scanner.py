import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

import database
import config

logger = logging.getLogger("scanner")


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "monthly") -> pd.DataFrame:
    """
    Resamples daily OHLCV dataframe into 'monthly', 'weekly', or 'daily'.
    Expects DatetimeIndex.
    """
    if df.empty:
        return df

    tf = timeframe.lower()
    if tf in ("daily", "1d", "d"):
        return df.copy()

    rule = "ME" if tf in ("monthly", "1m", "m") else "W"

    try:
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
    except Exception:
        # Fallback for older pandas versions
        legacy_rule = "M" if rule == "ME" else "W"
        resampled = df.resample(legacy_rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    return resampled


def calculate_rsi(series: pd.Series, span: int = 9) -> pd.Series:
    """
    Computes Relative Strength Index (RSI) using exponential moving average with specified span (default: 9).
    """
    if series.empty or len(series) < 2:
        return pd.Series(index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(span=span, adjust=False).mean()
    avg_loss = loss.ewm(span=span, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_ema(series: pd.Series, span: int = 3) -> pd.Series:
    """
    Computes Exponential Moving Average (EMA) of a series with specified span.
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_wma(series: pd.Series, period: int = 21) -> pd.Series:
    """
    Computes Weighted Moving Average (WMA) with linear weights [1, 2, ..., period].
    Handles series with fewer observations gracefully using expanding linear weights.
    """
    if series.empty:
        return series

    res = pd.Series(index=series.index, dtype=float)
    n = len(series)
    for i in range(n):
        start_idx = max(0, i - period + 1)
        sub = series.iloc[start_idx : i + 1].values
        w = np.arange(1, len(sub) + 1)
        res.iloc[i] = np.dot(sub, w) / w.sum()
    return res


def calculate_indicator(df: pd.DataFrame, indicator_type: str = "EMA", period: int = 5) -> pd.DataFrame:
    """
    Computes technical indicator (EMA, SMA, RSI) on close price.
    """
    df = df.copy()
    col_name = f"{indicator_type.upper()}_{period}"

    if indicator_type.upper() == "EMA":
        df[col_name] = df["close"].ewm(span=period, adjust=False).mean()
    elif indicator_type.upper() == "SMA":
        df[col_name] = df["close"].rolling(window=period).mean()
    elif indicator_type.upper() == "RSI":
        df[col_name] = calculate_rsi(df["close"], span=period)
    else:
        raise ValueError(f"Unsupported indicator: {indicator_type}")

    return df


def calculate_weekly_cpr(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Weekly CPR (TC, P, BC) and R1, S1 pivot levels based on previous week OHLC.
    Week ends on Sunday ('W-SUN').
    """
    if daily_df.empty:
        return pd.DataFrame()

    weekly = daily_df.resample("W-SUN").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    if len(weekly) < 2:
        return pd.DataFrame()

    prev_h = weekly["high"].shift(1)
    prev_l = weekly["low"].shift(1)
    prev_c = weekly["close"].shift(1)

    weekly["P"] = (prev_h + prev_l + prev_c) / 3.0
    weekly["BC"] = (prev_h + prev_l) / 2.0
    weekly["TC"] = (2 * weekly["P"]) - weekly["BC"]
    weekly["R1"] = (2 * weekly["P"]) - prev_l
    weekly["S1"] = (2 * weekly["P"]) - prev_h
    weekly["CPR_Width"] = (weekly["TC"] - weekly["BC"]).abs()
    weekly["CPR_Width_Pct"] = (weekly["CPR_Width"] / weekly["P"]) * 100

    return weekly


def attach_weekly_cpr_to_intraday(intra_df: pd.DataFrame, daily_df: pd.DataFrame, current_week_only: bool = True) -> pd.DataFrame:
    """
    Merges Weekly CPR / Pivot levels (Pivot, R1, S1) onto 75-minute candles.
    When current_week_only=True, only attaches levels to the current active week,
    avoiding continuous lines spanning across previous weeks.
    """
    if intra_df.empty or daily_df.empty:
        return intra_df

    df = intra_df.copy()
    weekly = calculate_weekly_cpr(daily_df)
    if weekly.empty:
        return df

    weekly["week_end"] = weekly.index.date
    pivot_cols = ["P", "R1", "S1", "TC", "BC"]
    pivot_map = weekly.set_index("week_end")[pivot_cols]

    intra_dates = pd.DataFrame({"timestamp": df.index})
    intra_dates["week_end"] = intra_dates["timestamp"].dt.to_period("W-SUN").dt.end_time.dt.date

    merged = intra_dates.merge(pivot_map, on="week_end", how="left")
    merged.set_index(df.index, inplace=True)

    current_week_end = intra_dates["week_end"].max()

    for col in pivot_cols:
        series = merged[col]
        if current_week_only:
            # Mask out non-current weeks to NaN so lines don't continuously connect across weeks
            series = series.where(merged["week_end"] == current_week_end, np.nan)
        df[f"Weekly_{col}"] = series

    return df


def scan_symbol(
    symbol: str,
    timeframe: str = "monthly",
    indicator_type: str = "EMA",
    period: int = 5,
    condition: str = "Close > Indicator"
) -> Optional[Dict[str, Any]]:
    """
    Evaluates scanning criteria for a single symbol.
    Returns result dictionary if condition matches, else None.
    """
    daily_df = database.get_candles_df(symbol)
    if daily_df.empty or len(daily_df) < (period + 2):
        return None

    tf_df = resample_ohlcv(daily_df, timeframe=timeframe)
    if len(tf_df) < (period + 1):
        return None

    ind_col = f"{indicator_type.upper()}_{period}"
    calc_df = calculate_indicator(tf_df, indicator_type=indicator_type, period=period)

    # Need at least current and previous candle
    latest = calc_df.iloc[-1]
    prev = calc_df.iloc[-2]

    curr_close = float(latest["close"])
    curr_ind = float(latest[ind_col])
    prev_close = float(prev["close"])
    prev_ind = float(prev[ind_col])

    if pd.isna(curr_ind) or pd.isna(prev_ind):
        return None

    pct_diff = round(((curr_close - curr_ind) / curr_ind) * 100, 2)

    matched = False
    status = ""

    if condition == "Close > Indicator":
        if curr_close > curr_ind:
            matched = True
            status = f"Above {ind_col}"
    elif condition == "Close < Indicator":
        if curr_close < curr_ind:
            matched = True
            status = f"Below {ind_col}"
    elif condition == "Bullish Crossover":
        # Crossed above: previous close <= indicator and current close > indicator
        if prev_close <= prev_ind and curr_close > curr_ind:
            matched = True
            status = f"Crossed Above {ind_col}"
    elif condition == "Bearish Crossover":
        # Crossed below: previous close >= indicator and current close < indicator
        if prev_close >= prev_ind and curr_close < curr_ind:
            matched = True
            status = f"Crossed Below {ind_col}"
    elif condition == "RSI > 50":
        rsi = calculate_rsi(tf_df["close"], span=period)
        if float(rsi.iloc[-1]) > 50.0:
            matched = True
            status = f"RSI({period}) > 50"
    elif condition == "EMA 3 > WMA 21 on RSI":
        rsi = calculate_rsi(tf_df["close"], span=period)
        ema3 = calculate_ema(rsi, span=3)
        wma21 = calculate_wma(rsi, period=21)
        if float(ema3.iloc[-1]) > float(wma21.iloc[-1]):
            matched = True
            status = "EMA 3 > WMA 21 on RSI"
    elif condition == "RSI > 50 & EMA 3 > WMA 21":
        rsi = calculate_rsi(tf_df["close"], span=period)
        ema3 = calculate_ema(rsi, span=3)
        wma21 = calculate_wma(rsi, period=21)
        if float(rsi.iloc[-1]) >= 50.0 and float(ema3.iloc[-1]) >= float(wma21.iloc[-1]):
            matched = True
            status = "RSI >= 50 & EMA 3 >= WMA 21"

    if not matched:
        return None

    inst_meta = database.get_instrument_by_symbol(symbol) or {}
    candle_date = latest.name.strftime("%Y-%m-%d") if hasattr(latest.name, "strftime") else str(latest.name)

    return {
        "Symbol": symbol.upper(),
        "Name": inst_meta.get("name", symbol),
        "Close": round(curr_close, 2),
        ind_col: round(curr_ind, 2),
        "Diff %": pct_diff,
        "Timeframe": timeframe.capitalize(),
        "Status": status,
        "Candle Date": candle_date,
        "Volume": int(latest["volume"])
    }


def run_scan(
    symbols: Optional[List[str]] = None,
    timeframe: str = "monthly",
    indicator_type: str = "EMA",
    period: int = 5,
    condition: str = "Close > Indicator"
) -> pd.DataFrame:
    """
    Scans multiple symbols for a specified single indicator condition.
    """
    database.init_db()
    if not symbols:
        symbols = database.get_all_symbols()

    results = []
    for sym in symbols:
        res = scan_symbol(sym, timeframe=timeframe, indicator_type=indicator_type, period=period, condition=condition)
        if res:
            results.append(res)

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    df_res.sort_values(by="Diff %", ascending=False, inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    return df_res


def evaluate_multiframe_alignment(
    symbol: str,
    stage_filter: Optional[str] = None,
    names_map: Optional[Dict[str, str]] = None,
    duckdb_cursor: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates the exact user strategy across timeframes with zero blocking network calls:
    1. Monthly: Close > 5 EMA and 5 EMA > 20 EMA
    2. Weekly:  Close > 20 EMA and 20 EMA > 50 EMA > 200 EMA
    3. Daily:   Close > 20 EMA and 20 EMA > 50 EMA > 200 EMA
    4. 75-Min:  Candles available with 5 EMA and 20 EMA
    """
    clean_sym = symbol.upper().strip().replace("-EQ", "").replace(".NS", "")
    
    # 1. Fetch daily candles via dedicated DuckDB cursor or database fallback
    if duckdb_cursor is not None:
        try:
            daily_df = duckdb_cursor.execute(
                "SELECT date, open, high, low, close, volume FROM daily_candles WHERE trading_symbol = ? ORDER BY date ASC;",
                [clean_sym]
            ).df()
            if not daily_df.empty:
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                daily_df.set_index("date", inplace=True)
        except Exception:
            daily_df = database.get_candles_df(clean_sym)
    else:
        daily_df = database.get_candles_df(clean_sym)

    if daily_df.empty or len(daily_df) < 50:
        return None

    # --- 1. Monthly Timeframe ---
    m_df = resample_ohlcv(daily_df, "monthly")
    if len(m_df) < 20:
        return None
    m_df["EMA_5"] = m_df["close"].ewm(span=5, adjust=False).mean()
    m_df["EMA_20"] = m_df["close"].ewm(span=20, adjust=False).mean()
    m_last = m_df.iloc[-1]
    m_close = float(m_last["close"])
    m_ema5 = float(m_last["EMA_5"])
    m_ema20 = float(m_last["EMA_20"])

    m_rule1 = m_close > m_ema5
    m_rule2 = m_ema5 > m_ema20
    monthly_pass = m_rule1 and m_rule2

    # Calculate RSI (span=9) on Monthly
    m_rsi_series = calculate_rsi(m_df["close"], span=9)
    m_rsi_ema3_series = calculate_ema(m_rsi_series, span=3)
    m_rsi_wma21_series = calculate_wma(m_rsi_series, period=21)

    m_rsi = round(float(m_rsi_series.iloc[-1]), 2)
    m_rsi_ema3 = round(float(m_rsi_ema3_series.iloc[-1]), 2)
    m_rsi_wma21 = round(float(m_rsi_wma21_series.iloc[-1]), 2)

    m_rsi_rule_50 = m_rsi >= 50.0
    m_rsi_rule_cross = m_rsi_ema3 >= m_rsi_wma21
    m_rsi_pass = m_rsi_rule_50 and m_rsi_rule_cross

    # Early-exit filters on Monthly
    if stage_filter == "Monthly RSI Scan Only (RSI>=50 & EMA3>=WMA21)" and not m_rsi_pass:
        return None
    if stage_filter == "Stage 3 + Monthly RSI (RSI>=50 & EMA3>=WMA21)" and (not monthly_pass or not m_rsi_pass):
        return None
    if stage_filter in ("Stage 3 (Full Alignment Only)", "Stage 2+ (M+W Aligned)", "Stage 1+ (Monthly Pass)") and not monthly_pass:
        return None

    # --- 2. Weekly Timeframe ---
    w_df = resample_ohlcv(daily_df, "weekly")
    w_df["EMA_20"] = w_df["close"].ewm(span=20, adjust=False).mean()
    w_df["EMA_50"] = w_df["close"].ewm(span=50, adjust=False).mean()
    w_df["EMA_200"] = w_df["close"].ewm(span=200, adjust=False).mean()
    w_last = w_df.iloc[-1]
    w_close = float(w_last["close"])
    w_ema20 = float(w_last["EMA_20"])
    w_ema50 = float(w_last["EMA_50"])
    w_ema200 = float(w_last["EMA_200"])

    w_rule1 = w_close > w_ema20
    w_rule2 = w_ema20 > w_ema50
    w_rule3 = w_ema50 > w_ema200
    weekly_pass = w_rule1 and w_rule2 and w_rule3

    # Early-exit filters on Weekly
    if stage_filter in ("Stage 3 (Full Alignment Only)", "Stage 3 + Monthly RSI (RSI>=50 & EMA3>=WMA21)", "Stage 2+ (M+W Aligned)") and not weekly_pass:
        return None

    # --- 3. Daily Timeframe ---
    d_df = daily_df.copy()
    d_df["EMA_20"] = d_df["close"].ewm(span=20, adjust=False).mean()
    d_df["EMA_50"] = d_df["close"].ewm(span=50, adjust=False).mean()
    d_df["EMA_200"] = d_df["close"].ewm(span=200, adjust=False).mean()
    d_last = d_df.iloc[-1]
    d_close = float(d_last["close"])
    d_ema20 = float(d_last["EMA_20"])
    d_ema50 = float(d_last["EMA_50"])
    d_ema200 = float(d_last["EMA_200"])

    d_rule1 = d_close > d_ema20
    d_rule2 = d_ema20 > d_ema50
    d_rule3 = d_ema50 > d_ema200
    daily_pass = d_rule1 and d_rule2 and d_rule3

    # Early-exit filters on Daily
    if stage_filter in ("Stage 3 (Full Alignment Only)", "Stage 3 + Monthly RSI (RSI>=50 & EMA3>=WMA21)") and not daily_pass:
        return None

    # Calculate overall stage
    if monthly_pass and weekly_pass and daily_pass:
        stage = "Stage 3: FULL ALIGNMENT 🚀"
        score = 3
    elif monthly_pass and weekly_pass:
        stage = "Stage 2: M+W ALIGNED 🟢"
        score = 2
    elif monthly_pass:
        stage = "Stage 1: MONTHLY ONLY 🟡"
        score = 1
    else:
        stage = "NO ALIGNMENT ⚪"
        score = 0

    # --- 4. 75-Minute Timeframe ---
    # Only load 75m candles if the stock has some alignment or if explicitly requested
    intra_pass = False
    i_close = d_close
    i_ema9 = d_close
    i_ema26 = d_close
    i_rsi = 0.0

    if score > 0 or stage_filter is None or stage_filter == "All":
        intra_df = pd.DataFrame()
        if duckdb_cursor is not None:
            try:
                intra_df = duckdb_cursor.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM intraday_candles WHERE trading_symbol = ? AND timeframe = '75m' ORDER BY timestamp DESC LIMIT 60;",
                    [clean_sym]
                ).df()
                if not intra_df.empty:
                    intra_df = intra_df.iloc[::-1].reset_index(drop=True)
            except Exception:
                intra_df = pd.DataFrame()
        
        if intra_df.empty:
            try:
                import parquet_loader
                intra_df = parquet_loader.ensure_symbol_75m_candles(clean_sym, min_bars=26)
            except Exception:
                intra_df = pd.DataFrame()

        if not intra_df.empty and len(intra_df) >= 26:
            intra_df["EMA_9"] = intra_df["close"].ewm(span=9, adjust=False).mean()
            intra_df["EMA_26"] = intra_df["close"].ewm(span=26, adjust=False).mean()
            i_last = intra_df.iloc[-1]
            i_close = float(i_last["close"])
            i_ema9 = float(i_last["EMA_9"])
            i_ema26 = float(i_last["EMA_26"])
            intra_pass = (i_close > i_ema26) and (i_ema9 > i_ema26)
            try:
                i_rsi = round(float(calculate_rsi(intra_df["close"], span=9).iloc[-1]), 2)
            except Exception:
                i_rsi = 0.0

    w_rsi = round(float(calculate_rsi(w_df["close"], span=9).iloc[-1]), 2) if (not w_df.empty and "close" in w_df.columns) else 0.0
    d_rsi = round(float(calculate_rsi(d_df["close"], span=9).iloc[-1]), 2) if (not d_df.empty and "close" in d_df.columns) else 0.0

    # Instrument metadata name lookup (in-memory map is O(1))
    if names_map and clean_sym in names_map:
        stock_name = names_map[clean_sym]
    else:
        inst_meta = database.get_instrument_by_symbol(clean_sym) or {}
        stock_name = inst_meta.get("name", clean_sym)

    return {
        "Symbol": clean_sym,
        "Name": stock_name,
        "Price": round(d_close, 2),
        "Stage": stage,
        "Score": score,
        # Monthly RSI Scan details
        "Monthly_RSI_Match": "✅ PASS" if m_rsi_pass else "❌ FAIL",
        "M_RSI": m_rsi,
        "M_RSI_EMA3": m_rsi_ema3,
        "M_RSI_WMA21": m_rsi_wma21,
        "M_RSI>=50": "✅" if m_rsi_rule_50 else "❌",
        "M_EMA3>=WMA21": "✅" if m_rsi_rule_cross else "❌",
        # RSI values
        "Daily_RSI": d_rsi,
        "75m_RSI": i_rsi,
        "Weekly_RSI": w_rsi,
        "Monthly_RSI": m_rsi,
        # Monthly details
        "Monthly Match": "✅ PASS" if monthly_pass else "❌ FAIL",
        "M_Close>5EMA": "✅" if m_rule1 else "❌",
        "M_5>20EMA": "✅" if m_rule2 else "❌",
        "M_Close": round(m_close, 2),
        "M_EMA5": round(m_ema5, 2),
        "M_EMA20": round(m_ema20, 2),
        # Weekly details
        "Weekly Match": "✅ PASS" if weekly_pass else "❌ FAIL",
        "W_Close>20EMA": "✅" if w_rule1 else "❌",
        "W_20>50EMA": "✅" if w_rule2 else "❌",
        "W_50>200EMA": "✅" if w_rule3 else "❌",
        "W_Close": round(w_close, 2),
        "W_EMA20": round(w_ema20, 2),
        "W_EMA50": round(w_ema50, 2),
        "W_EMA200": round(w_ema200, 2),
        # Daily details
        "Daily Match": "✅ PASS" if daily_pass else "❌ FAIL",
        "D_Close>20EMA": "✅" if d_rule1 else "❌",
        "D_20>50EMA": "✅" if d_rule2 else "❌",
        "D_50>200EMA": "✅" if d_rule3 else "❌",
        "D_Close": round(d_close, 2),
        "D_EMA20": round(d_ema20, 2),
        "D_EMA50": round(d_ema50, 2),
        "D_EMA200": round(d_ema200, 2),
        # 75m details
        "75m Match": "✅ PASS" if intra_pass else "❌ FAIL",
        "75m_Close": round(i_close, 2),
        "75m_EMA9": round(i_ema9, 2),
        "75m_EMA26": round(i_ema26, 2),
        "Volume": int(d_last["volume"])
    }


def run_multiframe_alignment_scan(symbols: Optional[List[str]] = None, stage_filter: str = "All") -> pd.DataFrame:
    """
    Runs the full multi-timeframe alignment scan across all symbols with high-performance
    DuckDB per-thread cursors, early-exit filtering, and zero blocking network calls.
    Strictly excludes:
    - All Indices (NSE_INDEX, BSE_INDEX, NIFTY, BANKNIFTY, SENSEX, etc.)
    - All BSE equity stocks
    Only scans pure NSE Equity stocks.
    """
    database.init_db()
    valid_nse_equities = set(database.get_alignment_scanner_symbols())

    import duckdb_store
    conn = duckdb_store.get_connection()

    # Preload instrument names into memory for instant O(1) lookups
    names_map = {}
    try:
        with duckdb_store._lock:
            rows = conn.execute("SELECT trading_symbol, name FROM instruments WHERE trading_symbol IS NOT NULL;").fetchall()
            names_map = {r[0].replace("-EQ", "").replace(".NS", ""): r[1] for r in rows if r[0]}
    except Exception as e:
        logger.warning(f"Note loading instruments names map: {e}")

    if not symbols:
        # Pre-filter: only scan symbols that actually have >= 50 daily candles in DuckDB
        db_symbols = []
        try:
            with duckdb_store._lock:
                rows = conn.execute("""
                    SELECT trading_symbol 
                    FROM daily_candles 
                    WHERE trading_symbol NOT LIKE '0%'
                    GROUP BY trading_symbol 
                    HAVING COUNT(*) >= 50;
                """).fetchall()
                db_symbols = [r[0].replace("-EQ", "").replace(".NS", "") for r in rows if r[0]]
        except Exception as e:
            logger.warning(f"Error querying DuckDB symbols for scanner: {e}")

        if not db_symbols:
            with database.get_connection() as sql_conn:
                cursor = sql_conn.cursor()
                cursor.execute("SELECT DISTINCT trading_symbol FROM daily_candles WHERE trading_symbol NOT LIKE '0%';")
                db_symbols = [row[0] for row in cursor.fetchall()]

        symbols = [s for s in db_symbols if s in valid_nse_equities]
    else:
        symbols = [s for s in symbols if s in valid_nse_equities]

    if not symbols:
        return pd.DataFrame()

    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(sym):
        try:
            # Each worker gets its own DuckDB cursor for lock-free parallel execution
            c = conn.cursor()
            return evaluate_multiframe_alignment(
                sym,
                stage_filter=stage_filter,
                names_map=names_map,
                duckdb_cursor=c
            )
        except Exception:
            return None

    # Multi-threaded parallel scanning across CPU threads
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_eval, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                if stage_filter == "Stage 3 (Full Alignment Only)" and res["Score"] < 3:
                    continue
                elif stage_filter == "Stage 2+ (M+W Aligned)" and res["Score"] < 2:
                    continue
                elif stage_filter == "Stage 1+ (Monthly Pass)" and res["Score"] < 1:
                    continue
                elif stage_filter == "Stage 3 + Monthly RSI (RSI>=50 & EMA3>=WMA21)" and (res["Score"] < 3 or res["Monthly_RSI_Match"] != "✅ PASS"):
                    continue
                elif stage_filter == "Monthly RSI Scan Only (RSI>=50 & EMA3>=WMA21)" and res["Monthly_RSI_Match"] != "✅ PASS":
                    continue
                results.append(res)

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    df_res.sort_values(by=["Score", "Price"], ascending=[False, False], inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    return df_res


def calculate_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Computes Simple Moving Average (SMA) of a series."""
    return series.rolling(window=period, min_periods=1).mean()


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Computes Bollinger Bands (Middle, Upper, Lower) for a price series.
    """
    if series.empty:
        return pd.DataFrame(columns=["BB_Middle", "BB_Upper", "BB_Lower"], index=series.index)
    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std(ddof=0).fillna(0)
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return pd.DataFrame({
        "BB_Middle": middle,
        "BB_Upper": upper,
        "BB_Lower": lower
    }, index=series.index)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Computes MACD Line, Signal Line, and Histogram.
    """
    if series.empty:
        return pd.DataFrame(columns=["MACD_Line", "MACD_Signal", "MACD_Hist"], index=series.index)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return pd.DataFrame({
        "MACD_Line": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": macd_hist
    }, index=series.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Average True Range (ATR) using Wilder's smoothing.
    """
    if df.empty or len(df) < 2:
        return pd.Series(index=df.index, dtype=float)
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Computes standard TradingView Supertrend indicator.
    Returns DataFrame with 'Supertrend' price line and 'Trend_Direction' (+1 for bullish green, -1 for bearish red).
    """
    if df.empty or len(df) < period:
        return pd.DataFrame(columns=["Supertrend", "Trend_Direction"], index=df.index)

    hl2 = (df["high"] + df["low"]) / 2.0
    atr = calculate_atr(df, period=period)

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n, dtype=int)

    closes = df["close"].values
    b_upper = basic_upper.values
    b_lower = basic_lower.values

    # Initialize first bar
    final_upper[0] = b_upper[0]
    final_lower[0] = b_lower[0]
    supertrend[0] = final_upper[0]
    direction[0] = 1

    for i in range(1, n):
        # Calculate Final Upper Band
        if b_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
            final_upper[i] = b_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Calculate Final Lower Band
        if b_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
            final_lower[i] = b_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Determine Supertrend value and trend direction
        prev_st = supertrend[i - 1]
        prev_dir = direction[i - 1]

        if prev_dir == 1:
            if closes[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if closes[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    return pd.DataFrame({
        "Supertrend": supertrend,
        "Trend_Direction": direction
    }, index=df.index)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Computes Volume Weighted Average Price (VWAP) for intraday data.
    Resets daily if DatetimeIndex is present, or computes cumulative VWAP.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype=float)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].fillna(0)
    vp = typical_price * volume

    if isinstance(df.index, pd.DatetimeIndex):
        # Group by trading date for intraday reset
        dates = df.index.date
        cum_vp = vp.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
        vwap = cum_vp / cum_vol.replace(0, np.nan)
        return vwap.ffill().bfill()
    else:
        cum_vp = vp.cumsum()
        cum_vol = volume.cumsum()
        return (cum_vp / cum_vol.replace(0, np.nan)).ffill().bfill()


def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """
    Computes Stochastic Oscillator (%K, %D).
    """
    if df.empty or len(df) < period:
        return pd.DataFrame(columns=["Stoch_K", "Stoch_D"], index=df.index)

    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)

    fast_k = 100.0 * ((df["close"] - low_min) / denom)
    stoch_k = fast_k.rolling(window=smooth_k, min_periods=1).mean().fillna(50.0)
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=1).mean().fillna(50.0)

    return pd.DataFrame({
        "Stoch_K": stoch_k,
        "Stoch_D": stoch_d
    }, index=df.index)


def calculate_cpr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Central Pivot Range (Pivot, Bottom Central, Top Central) and S1/S2/R1/R2 levels.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    prev_h = df["high"].shift(1)
    prev_l = df["low"].shift(1)
    prev_c = df["close"].shift(1)

    pivot = (prev_h + prev_l + prev_c) / 3.0
    bc = (prev_h + prev_l) / 2.0
    tc = (pivot - bc) + pivot
    r1 = (2.0 * pivot) - prev_l
    s1 = (2.0 * pivot) - prev_h
    r2 = pivot + (prev_h - prev_l)
    s2 = pivot - (prev_h - prev_l)

    return pd.DataFrame({
        "CPR_Pivot": pivot,
        "CPR_BC": bc,
        "CPR_TC": tc,
        "CPR_R1": r1,
        "CPR_S1": s1,
        "CPR_R2": r2,
        "CPR_S2": s2
    }, index=df.index)

