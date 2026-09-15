"""
strategy_engine.py
------------------
Advanced Strategy Builder, Simulator, and Backtesting Engine for Upstox Scanner.
Supports single-stock and portfolio/basket backtesting with realistic execution,
stop-loss, profit targets, trailing stops, brokerage/slippage, and performance analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import scanner


@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: Any
    entry_price: float
    exit_time: Any = None
    exit_price: float = 0.0
    quantity: int = 1
    pnl_rupees: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""  # "Target", "StopLoss", "TrailingStop", "SignalExit", "EndOfData"
    duration_bars: int = 0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    planned_rr: float = 0.0
    realized_rr: float = 0.0
    risk_reward: str = ""


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    timeframe: str
    start_time: Any
    end_time: Any
    initial_capital: float
    final_equity: float
    total_net_pnl: float
    total_net_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_rupees: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    indicators_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    avg_risk_reward: float = 0.0
    planned_risk_reward: float = 0.0


@dataclass
class StrategyConfig:
    strategy_type: str = "Hilega_Milega"  # "Hilega_Milega", "Triple_EMA", "SuperTrend", "CPR_Breakout", "MACD_Cross", "Chartink_75_Waterfall", "Custom"
    name: str = "Hilega Milega Momentum"
    # Indicator Parameters
    rsi_span: int = 9
    ema_fast: int = 9
    ema_mid: int = 20
    ema_slow: int = 50
    st_period: int = 10
    st_multiplier: float = 3.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # CPR / Waterfall Exits
    use_cpr_exits: bool = False
    cpr_target_level: str = "R1"  # "R1", "R2", etc.
    cpr_stop_level: str = "S_05"  # "S_05" (midpoint between Pivot and S1)
    # Custom Rules (if strategy_type == "Custom")
    custom_ind1: str = "RSI"  # e.g., "RSI", "Close", "EMA_Fast", "MACD"
    custom_op: str = "crosses_above"  # "crosses_above", "crosses_below", "greater_than", "less_than"
    custom_ind2: str = "WMA_21"  # e.g., "EMA_Fast", "EMA_Mid", "Value", "WMA_21"
    custom_val: float = 50.0
    custom_filter_ind: Optional[str] = "Close"
    custom_filter_op: Optional[str] = "greater_than"
    custom_filter_target: Optional[str] = "EMA_Mid"
    custom_filter_val: float = 0.0
    # Risk Management & Exit Rules
    target_pct: float = 4.0  # 0 to disable
    stop_loss_pct: float = 2.0  # 0 to disable
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 1.5
    exit_on_signal_reversal: bool = True
    max_holding_bars: int = 0  # 0 for unlimited
    # Capital & Sizing
    position_sizing: str = "percent"  # "percent", "fixed_amount", "fixed_shares"
    size_value: float = 100.0  # 100% of capital or ₹ amount or shares
    slippage_brokerage_pct: float = 0.05  # 0.05% per trade (buy + sell = 0.10%)


def prepare_indicators(df: pd.DataFrame, cfg: StrategyConfig, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Computes all standard indicators needed by strategies.
    Ensures clean DatetimeIndex and calculated indicator columns.
    When cfg.strategy_type == 'Chartink_75_Waterfall' or CPR exits are enabled,
    attaches Weekly CPR levels (P, BC, TC, R1, S1, S_05) and multi-timeframe stages.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    close = out["close"]

    # 1. Hilega Milega indicators
    out["RSI"] = scanner.calculate_rsi(close, span=cfg.rsi_span)
    out["RSI_EMA3"] = scanner.calculate_ema(out["RSI"], span=3)
    out["RSI_WMA21"] = scanner.calculate_wma(out["RSI"], period=21)

    # 2. EMAs
    out["EMA_Fast"] = scanner.calculate_ema(close, span=cfg.ema_fast)
    out["EMA_Mid"] = scanner.calculate_ema(close, span=cfg.ema_mid)
    out["EMA_Slow"] = scanner.calculate_ema(close, span=cfg.ema_slow)
    out["EMA_200"] = scanner.calculate_ema(close, span=200)

    # 3. SuperTrend
    st_df = scanner.calculate_supertrend(out, period=cfg.st_period, multiplier=cfg.st_multiplier)
    if not st_df.empty and "ST_Direction" in st_df.columns:
        out["ST_Direction"] = st_df["ST_Direction"]
        out["SuperTrend"] = st_df["SuperTrend"]
    else:
        out["ST_Direction"] = 0
        out["SuperTrend"] = close

    # 4. MACD
    macd_df = scanner.calculate_macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    if not macd_df.empty and "MACD" in macd_df.columns:
        out["MACD"] = macd_df["MACD"]
        out["MACD_Signal"] = macd_df["MACD_Signal"]
        out["MACD_Hist"] = macd_df["MACD_Hist"]
    else:
        out["MACD"] = 0
        out["MACD_Signal"] = 0
        out["MACD_Hist"] = 0

    # 5. ATR for dynamic volatility sizing
    out["ATR"] = scanner.calculate_atr(out, period=14)

    # 6. Volume SMA
    if "volume" in out.columns:
        out["Vol_SMA20"] = out["volume"].rolling(window=20, min_periods=1).mean()
    else:
        out["Vol_SMA20"] = 1.0

    # 7. Weekly CPR and Multi-Timeframe Waterfall Integration
    try:
        if len(out) >= 2 and (out.index[1] - out.index[0]).total_seconds() >= 80000:
            daily_res = out.copy()
        else:
            daily_res = out.resample("D").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()

        if symbol:
            try:
                import database
                db_daily = database.get_candles_df(symbol)
                if db_daily is not None and len(db_daily) > len(daily_res):
                    daily_res = db_daily
            except Exception:
                pass

        if len(daily_res) >= 2:
            weekly = daily_res.resample("W-SUN").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(weekly) >= 2:
                prev_h = weekly["high"].shift(1)
                prev_l = weekly["low"].shift(1)
                prev_c = weekly["close"].shift(1)
                weekly["P"] = (prev_h + prev_l + prev_c) / 3.0
                weekly["BC"] = (prev_h + prev_l) / 2.0
                weekly["TC"] = (2 * weekly["P"]) - weekly["BC"]
                weekly["R1"] = (2 * weekly["P"]) - prev_l
                weekly["S1"] = (2 * weekly["P"]) - prev_h
                # Weekly CPR 0.5 Support = Midpoint between Pivot and S1
                weekly["S_05"] = (weekly["P"] + weekly["S1"]) / 2.0

                weekly["week_end"] = weekly.index.date
                pivot_cols = ["P", "BC", "TC", "R1", "S1", "S_05"]
                pivot_map = weekly.set_index("week_end")[pivot_cols]

                bar_dates = pd.DataFrame({"timestamp": out.index})
                bar_dates["week_end"] = bar_dates["timestamp"].dt.to_period("W-SUN").dt.end_time.dt.date
                merged = bar_dates.merge(pivot_map, on="week_end", how="left")
                merged.set_index(out.index, inplace=True)
                for col in pivot_cols:
                    out[f"Weekly_{col}"] = merged[col].ffill()

        # Multi-Timeframe Waterfall Stage alignment
        if cfg.strategy_type == "Chartink_75_Waterfall":
            m_pass_series = pd.Series(True, index=out.index)
            try:
                m_raw = daily_res.resample("ME").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last"
                }).dropna()
                if len(m_raw) >= 3:
                    m_c = m_raw["close"]
                    m_e5 = scanner.calculate_ema(m_c, span=5)
                    m_e20 = scanner.calculate_ema(m_c, span=20)
                    m_r = scanner.calculate_rsi(m_c, span=9)
                    m_re3 = scanner.calculate_ema(m_r, span=3)
                    m_rw21 = scanner.calculate_wma(m_r, period=21)
                    m_raw["m_pass"] = (
                        (m_c > m_e5) & (m_c > m_e20) & (m_e5 > m_e20) &
                        (m_r > 50.0) & (m_r < 88.0) &
                        (m_r >= m_re3) & (m_r >= m_rw21)
                    )
                    m_map = m_raw[["m_pass"]].copy()
                    m_map["month_key"] = m_map.index.to_period("M")
                    b_df = pd.DataFrame({"timestamp": out.index, "month_key": out.index.to_period("M")})
                    m_merged = b_df.merge(m_map[["month_key", "m_pass"]], on="month_key", how="left")
                    m_pass_series = m_merged["m_pass"].fillna(False).values
            except Exception:
                pass

            w_pass_series = pd.Series(True, index=out.index)
            try:
                if len(weekly) >= 5:
                    w_c = weekly["close"]
                    w_e20 = scanner.calculate_ema(w_c, span=20)
                    w_e50 = scanner.calculate_ema(w_c, span=50)
                    w_r = scanner.calculate_rsi(w_c, span=9)
                    w_re3 = scanner.calculate_ema(w_r, span=3)
                    w_rw21 = scanner.calculate_wma(w_r, period=21)
                    weekly["w_pass"] = (
                        (w_c > w_e20) & (w_e20 > w_e50) &
                        (w_r > 50.0) & (w_r >= w_re3) & (w_r >= w_rw21)
                    )
                    w_map = weekly[["week_end", "w_pass"]].copy()
                    b_df = pd.DataFrame({"timestamp": out.index, "week_end": out.index.dt.to_period("W-SUN").dt.end_time.dt.date})
                    w_merged = b_df.merge(w_map, on="week_end", how="left")
                    w_pass_series = w_merged["w_pass"].fillna(False).values
            except Exception:
                pass

            d_pass_series = pd.Series(True, index=out.index)
            try:
                if len(daily_res) >= 15:
                    d_c = daily_res["close"]
                    d_o = daily_res["open"]
                    d_e20 = scanner.calculate_ema(d_c, span=20)
                    d_e50 = scanner.calculate_ema(d_c, span=50)
                    d_r = scanner.calculate_rsi(d_c, span=9)
                    d_re3 = scanner.calculate_ema(d_r, span=3)
                    d_rw21 = scanner.calculate_wma(d_r, period=21)
                    d_bounce = (d_o <= d_e20 * 1.015) & (d_c >= d_e20 * 0.985)
                    d_trend = (d_e20 > d_e50)
                    d_hilega = (d_r > 50.0) & (d_r >= d_re3) & (d_r >= d_rw21)
                    daily_res["d_pass"] = d_bounce & d_trend & d_hilega
                    d_map = pd.DataFrame({"day_date": daily_res.index.date, "d_pass": daily_res["d_pass"].values})
                    b_df = pd.DataFrame({"timestamp": out.index, "day_date": out.index.date})
                    d_merged = b_df.merge(d_map, on="day_date", how="left")
                    d_pass_series = d_merged["d_pass"].fillna(False).values
            except Exception:
                pass

            # 75-Min Stage (or Bar Trigger)
            q_c = out["close"]
            q_e5 = scanner.calculate_ema(q_c, span=5)
            q_e20 = scanner.calculate_ema(q_c, span=20)
            q_r = scanner.calculate_rsi(q_c, span=9)
            q_re3 = scanner.calculate_ema(q_r, span=3)
            q_rw21 = scanner.calculate_wma(q_r, period=21)
            q4_pass = (
                (q_c > q_e20) &
                (q_e5 >= q_e20) &
                (q_r > 50.0) &
                (q_r >= q_re3) &
                (q_r >= q_rw21)
            )

            if "Waterfall_Stage4" in df.columns:
                out["Waterfall_Stage4"] = df["Waterfall_Stage4"]
            else:
                out["Waterfall_Stage4"] = (
                    pd.Series(m_pass_series, index=out.index).fillna(False) &
                    pd.Series(w_pass_series, index=out.index).fillna(False) &
                    pd.Series(d_pass_series, index=out.index).fillna(False) &
                    q4_pass
                )

        # Chartink Intraday Scan #19122704 Stage alignment (Breakdown)
        if cfg.strategy_type == "Chartink_Intraday_Scan_19122704":
            m_pass_series = pd.Series(True, index=out.index)
            try:
                m_raw = daily_res.resample("ME").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last"
                }).dropna()
                if len(m_raw) >= 3:
                    m_c = m_raw["close"]
                    m_e5 = scanner.calculate_ema(m_c, span=5)
                    m_e20 = scanner.calculate_ema(m_c, span=20)
                    m_e50 = scanner.calculate_ema(m_c, span=50)
                    m_r = scanner.calculate_rsi(m_c, span=9)
                    m_re3 = scanner.calculate_ema(m_r, span=3)
                    m_rw21 = scanner.calculate_wma(m_r, period=21)
                    m_raw["m_pass"] = (
                        (m_c < m_e5) & (m_e5 < m_e20) & (m_e20 < m_e50) &
                        (m_r < 50.0) & (m_r < m_re3) & (m_r < m_rw21)
                    )
                    m_map = m_raw[["m_pass"]].copy()
                    m_map["month_key"] = m_map.index.to_period("M")
                    b_df = pd.DataFrame({"timestamp": out.index, "month_key": out.index.to_period("M")})
                    m_merged = b_df.merge(m_map[["month_key", "m_pass"]], on="month_key", how="left")
                    m_pass_series = m_merged["m_pass"].fillna(False).values
            except Exception:
                pass

            w_pass_series = pd.Series(True, index=out.index)
            try:
                if len(weekly) >= 5:
                    w_c = weekly["close"]
                    w_e20 = scanner.calculate_ema(w_c, span=20)
                    w_e50 = scanner.calculate_ema(w_c, span=50)
                    w_e200 = scanner.calculate_ema(w_c, span=200)
                    w_r = scanner.calculate_rsi(w_c, span=9)
                    w_re3 = scanner.calculate_ema(w_r, span=3)
                    w_rw21 = scanner.calculate_wma(w_r, period=21)
                    weekly["w_pass"] = (
                        (w_c < w_e20) & (w_e20 < w_e50) & (w_e50 < w_e200) &
                        (w_r < 50.0) & (w_r < w_re3) & (w_r < w_rw21)
                    )
                    w_map = weekly[["week_end", "w_pass"]].copy()
                    b_df = pd.DataFrame({"timestamp": out.index, "week_end": out.index.dt.to_period("W-SUN").dt.end_time.dt.date})
                    w_merged = b_df.merge(w_map, on="week_end", how="left")
                    w_pass_series = w_merged["w_pass"].fillna(False).values
            except Exception:
                pass

            d_pass_series = pd.Series(True, index=out.index)
            try:
                if len(daily_res) >= 15:
                    d_c = daily_res["close"]
                    d_e20 = scanner.calculate_ema(d_c, span=20)
                    d_e50 = scanner.calculate_ema(d_c, span=50)
                    d_e200 = scanner.calculate_ema(d_c, span=200)
                    d_r = scanner.calculate_rsi(d_c, span=9)
                    d_re3 = scanner.calculate_ema(d_r, span=3)
                    d_rw21 = scanner.calculate_wma(d_r, period=21)
                    daily_res["d_pass"] = (
                        (d_c < d_e20) & (d_e20 < d_e50) & (d_e50 < d_e200) &
                        (d_r < 50.0) & (d_r < d_re3) & (d_r < d_rw21)
                    )
                    d_map = pd.DataFrame({"day_date": daily_res.index.date, "d_pass": daily_res["d_pass"].values})
                    b_df = pd.DataFrame({"timestamp": out.index, "day_date": out.index.date})
                    d_merged = b_df.merge(d_map, on="day_date", how="left")
                    d_pass_series = d_merged["d_pass"].fillna(False).values
            except Exception:
                pass

            # 75-Min Stage
            q_c = out["close"]
            q_e20 = scanner.calculate_ema(q_c, span=20)
            q_e50 = scanner.calculate_ema(q_c, span=50)
            q_e200 = scanner.calculate_ema(q_c, span=200)
            q4_pass = (q_e20 <= q_e50) & (q_e50 <= q_e200) & (q_c < q_e20)

            out["Breakdown_Stage4"] = (
                pd.Series(m_pass_series, index=out.index).fillna(False) &
                pd.Series(w_pass_series, index=out.index).fillna(False) &
                pd.Series(d_pass_series, index=out.index).fillna(False) &
                q4_pass
            )

        # Preserve any pre-existing Weekly CPR columns if already provided in df
        for col in ["P", "BC", "TC", "R1", "S1", "S_05"]:
            if f"Weekly_{col}" in df.columns:
                out[f"Weekly_{col}"] = df[f"Weekly_{col}"]
    except Exception as e:
        pass

    return out


def generate_strategy_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Generates Boolean entry and exit signal series based on StrategyConfig.
    1 = Long Entry, -1 = Exit Signal, 0 = Hold
    """
    df_sig = df.copy()
    n = len(df_sig)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    if n < 5:
        df_sig["signal_entry"] = False
        df_sig["signal_exit"] = False
        return df_sig

    close = df_sig["close"].values
    rsi = df_sig["RSI"].values
    rsi_ema3 = df_sig["RSI_EMA3"].values
    rsi_wma21 = df_sig["RSI_WMA21"].values
    ema_fast = df_sig["EMA_Fast"].values
    ema_mid = df_sig["EMA_Mid"].values
    ema_slow = df_sig["EMA_Slow"].values
    st_dir = df_sig["ST_Direction"].values
    macd = df_sig["MACD"].values
    macd_sig = df_sig["MACD_Signal"].values

    stype = cfg.strategy_type

    if stype == "Hilega_Milega":
        # Entry: RSI > EMA(3) and RSI > WMA(21) and RSI > 50 (or crossover)
        for i in range(1, n):
            crossover = (rsi[i] > rsi_ema3[i]) and (rsi[i-1] <= rsi_ema3[i-1])
            is_bullish = (rsi[i] > rsi_wma21[i]) and (rsi[i] > 50.0)
            if crossover and is_bullish:
                entries[i] = True
            # Signal Exit: RSI crosses below EMA(3) or RSI < 48
            if cfg.exit_on_signal_reversal:
                if (rsi[i] < rsi_ema3[i] and rsi[i-1] >= rsi_ema3[i-1]) or (rsi[i] < 45.0):
                    exits[i] = True

    elif stype == "Triple_EMA":
        # Entry: EMA 9 > EMA 20 > EMA 50 AND Close > EMA 9
        for i in range(1, n):
            fast_cross = (ema_fast[i] > ema_mid[i]) and (ema_fast[i-1] <= ema_mid[i-1])
            aligned = (ema_mid[i] > ema_slow[i]) and (close[i] > ema_fast[i])
            if (fast_cross and aligned) or (ema_fast[i] > ema_mid[i] > ema_slow[i] and close[i] > ema_fast[i] and close[i-1] <= ema_fast[i-1]):
                entries[i] = True
            if cfg.exit_on_signal_reversal:
                if close[i] < ema_mid[i] or ema_fast[i] < ema_mid[i]:
                    exits[i] = True

    elif stype == "SuperTrend":
        # Entry: SuperTrend switches to Bullish (1)
        for i in range(1, n):
            if st_dir[i] == 1 and st_dir[i-1] == -1:
                entries[i] = True
            if cfg.exit_on_signal_reversal:
                if st_dir[i] == -1:
                    exits[i] = True

    elif stype == "MACD_Cross":
        # Entry: MACD crosses above Signal and MACD > 0
        for i in range(1, n):
            macd_cross = (macd[i] > macd_sig[i]) and (macd[i-1] <= macd_sig[i-1])
            if macd_cross and (macd[i] > 0 or rsi[i] > 50):
                entries[i] = True
            if cfg.exit_on_signal_reversal:
                if macd[i] < macd_sig[i] and macd[i-1] >= macd_sig[i-1]:
                    exits[i] = True

    elif stype == "CPR_Breakout":
        # Weekly CPR breakout if attached
        if "Weekly_TC" in df_sig.columns and "Weekly_BC" in df_sig.columns:
            tc = df_sig["Weekly_TC"].values
            bc = df_sig["Weekly_BC"].values
            for i in range(1, n):
                if close[i] > tc[i] and close[i-1] <= tc[i-1]:
                    entries[i] = True
                if cfg.exit_on_signal_reversal:
                    if close[i] < bc[i]:
                        exits[i] = True
        else:
            # Fallback to Bollinger Breakout
            sma20 = df_sig["close"].rolling(20).mean().values
            std20 = df_sig["close"].rolling(20).std().values
            upper = sma20 + 2.0 * std20
            for i in range(1, n):
                if close[i] > upper[i] and close[i-1] <= upper[i-1]:
                    entries[i] = True
                if cfg.exit_on_signal_reversal:
                    if close[i] < sma20[i]:
                        exits[i] = True

    elif stype == "Custom":
        def _get_series(col_name):
            col_map = {
                "RSI": rsi,
                "RSI_EMA3": rsi_ema3,
                "WMA_21": rsi_wma21,
                "Close": close,
                "EMA_Fast": ema_fast,
                "EMA_Mid": ema_mid,
                "EMA_Slow": ema_slow,
                "MACD": macd,
                "MACD_Signal": macd_sig,
                "SuperTrend": df_sig["SuperTrend"].values
            }
            return col_map.get(col_name, close)

        s1 = _get_series(cfg.custom_ind1)
        if cfg.custom_ind2 == "Value":
            s2 = np.full(n, cfg.custom_val)
        else:
            s2 = _get_series(cfg.custom_ind2)

        op = cfg.custom_op
        for i in range(1, n):
            cond = False
            if op == "crosses_above":
                cond = (s1[i] > s2[i]) and (s1[i-1] <= s2[i-1])
            elif op == "crosses_below":
                cond = (s1[i] < s2[i]) and (s1[i-1] >= s2[i-1])
            elif op == "greater_than":
                cond = (s1[i] > s2[i])
            elif op == "less_than":
                cond = (s1[i] < s2[i])

            # Optional filter
            if cond and cfg.custom_filter_ind and cfg.custom_filter_target:
                f_s1 = _get_series(cfg.custom_filter_ind)
                f_s2 = _get_series(cfg.custom_filter_target) if cfg.custom_filter_target != "Value" else np.full(n, cfg.custom_filter_val)
                f_op = cfg.custom_filter_op or "greater_than"
                if f_op == "greater_than" and not (f_s1[i] > f_s2[i]):
                    cond = False
                elif f_op == "less_than" and not (f_s1[i] < f_s2[i]):
                    cond = False

            if cond:
                entries[i] = True

            # Exit rule for custom
            if cfg.exit_on_signal_reversal:
                if op in ("crosses_above", "greater_than") and s1[i] < s2[i]:
                    exits[i] = True
                elif op in ("crosses_below", "less_than") and s1[i] > s2[i]:
                    exits[i] = True

    elif stype == "Chartink_75_Waterfall":
        # Authentic Chartink Positional Scan #364 Stage 4 Buy-Only trigger
        stage4 = df_sig["Waterfall_Stage4"].values if "Waterfall_Stage4" in df_sig.columns else np.zeros(n, dtype=bool)
        q_ema5 = scanner.calculate_ema(df_sig["close"], span=5).values
        q_ema20 = scanner.calculate_ema(df_sig["close"], span=20).values
        for i in range(1, n):
            fresh_stage4 = bool(stage4[i] and not stage4[i-1])
            ema_cross = bool(stage4[i] and (q_ema5[i] > q_ema20[i]) and (q_ema5[i-1] <= q_ema20[i-1]))
            # Only Buy (Long Only)
            if fresh_stage4 or ema_cross:
                entries[i] = True
            if cfg.exit_on_signal_reversal:
                if (q_ema5[i] < q_ema20[i] and q_ema5[i-1] >= q_ema20[i-1]) or (rsi[i] < 45.0):
                    exits[i] = True

    elif stype == "Chartink_Intraday_Scan_19122704":
        stage4 = df_sig["Breakdown_Stage4"].values if "Breakdown_Stage4" in df_sig.columns else np.zeros(n, dtype=bool)
        q_ema20 = scanner.calculate_ema(df_sig["close"], span=20).values
        for i in range(1, n):
            fresh_stage4 = bool(stage4[i] and not stage4[i-1])
            if fresh_stage4 or (stage4[i] and df_sig["close"].iloc[i] < q_ema20[i] and df_sig["close"].iloc[i-1] >= q_ema20[i-1]):
                entries[i] = True
            if cfg.exit_on_signal_reversal:
                if df_sig["close"].iloc[i] > q_ema20[i] or rsi[i] > 55.0:
                    exits[i] = True

    df_sig["signal_entry"] = entries
    df_sig["signal_exit"] = exits
    df_sig["signal"] = np.where(entries, 1, np.where(exits, -1, 0))
    return df_sig


def run_backtest(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    symbol: str = "STOCK",
    timeframe: str = "Daily",
    initial_capital: float = 100000.0,
    slippage_pct: float = 0.05
) -> BacktestResult:
    """
    Executes a bar-by-bar simulation avoiding lookahead bias.
    Tracks equity curve, trade P&L, stop loss, target, and trailing stop triggers.
    """
    if df is None or df.empty or len(df) < 5:
        return BacktestResult(
            symbol=symbol,
            strategy_name=cfg.name,
            timeframe=timeframe,
            start_time=None,
            end_time=None,
            initial_capital=initial_capital,
            final_equity=initial_capital,
            total_net_pnl=0.0,
            total_net_pnl_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_rupees=0.0,
            avg_trade_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            expectancy=0.0,
            equity_curve=pd.DataFrame(columns=["time", "equity", "cash", "in_position", "peak", "high_watermark", "drawdown_rupees", "drawdown_pct"])
        )

    # 1. Prepare indicators & signals
    df_prep = prepare_indicators(df, cfg, symbol=symbol)
    df_sig = generate_strategy_signals(df_prep, cfg)

    # 2. Simulation state
    cash = float(initial_capital)
    current_equity = float(initial_capital)
    open_trade: Optional[Trade] = None
    trades: List[Trade] = []
    trade_id_counter = 1

    dates = df_sig.index
    opens = df_sig["open"].values
    highs = df_sig["high"].values
    lows = df_sig["low"].values
    closes = df_sig["close"].values
    entries = df_sig["signal_entry"].values
    exits = df_sig["signal_exit"].values

    equity_points = []
    highest_price_in_trade = 0.0

    fee_mult = 1.0 - (slippage_pct / 100.0)
    fee_add = 1.0 + (slippage_pct / 100.0)

    for i in range(len(df_sig)):
        bar_date = dates[i]
        o_price = opens[i]
        h_price = highs[i]
        l_price = lows[i]
        c_price = closes[i]

        # Check existing open position exits
        if open_trade is not None:
            open_trade.duration_bars += 1
            highest_price_in_trade = max(highest_price_in_trade, h_price)

            entry_p = open_trade.entry_price
            qty = open_trade.quantity
            exit_price = None
            exit_reason = None

            # Calculate MFE & MAE
            unrealized_high = (h_price - entry_p) / entry_p * 100.0
            unrealized_low = (l_price - entry_p) / entry_p * 100.0
            open_trade.max_favorable_excursion = max(open_trade.max_favorable_excursion, unrealized_high)
            open_trade.max_adverse_excursion = min(open_trade.max_adverse_excursion, unrealized_low)

            # Weekly CPR Dynamic Levels
            use_cpr = cfg.use_cpr_exits or (cfg.strategy_type == "Chartink_75_Waterfall")
            weekly_r1 = df_sig["Weekly_R1"].iloc[i] if ("Weekly_R1" in df_sig.columns and not pd.isna(df_sig["Weekly_R1"].iloc[i])) else None
            weekly_s_05 = df_sig["Weekly_S_05"].iloc[i] if ("Weekly_S_05" in df_sig.columns and not pd.isna(df_sig["Weekly_S_05"].iloc[i])) else None

            # 1. Stop Loss Trigger (Weekly CPR 0.5 Support or fixed %)
            if use_cpr and weekly_s_05 is not None and weekly_s_05 > 0 and weekly_s_05 < entry_p:
                stop_threshold = float(weekly_s_05)
                if l_price <= stop_threshold:
                    exit_price = min(o_price, stop_threshold) * fee_mult
                    exit_reason = "Stop Loss (Weekly CPR 0.5 Support)"
            elif cfg.stop_loss_pct > 0:
                stop_threshold = entry_p * (1.0 - cfg.stop_loss_pct / 100.0)
                if l_price <= stop_threshold:
                    # Slips to open if open gapped below stop
                    exit_price = min(o_price, stop_threshold) * fee_mult
                    exit_reason = "Stop Loss Hit"

            # 2. Trailing Stop Loss Trigger
            if exit_price is None and cfg.use_trailing_stop and cfg.trailing_stop_pct > 0:
                trail_threshold = highest_price_in_trade * (1.0 - cfg.trailing_stop_pct / 100.0)
                if l_price <= trail_threshold:
                    exit_price = min(o_price, trail_threshold) * fee_mult
                    exit_reason = "Trailing Stop"

            # 3. Take Profit Target Trigger (Weekly CPR R1 or fixed %)
            if exit_price is None:
                if use_cpr and weekly_r1 is not None and weekly_r1 > entry_p:
                    target_threshold = float(weekly_r1)
                    if h_price >= target_threshold:
                        exit_price = max(o_price, target_threshold) * fee_mult
                        exit_reason = "Target (Weekly CPR R1)"
                elif cfg.target_pct > 0:
                    target_threshold = entry_p * (1.0 + cfg.target_pct / 100.0)
                    if h_price >= target_threshold:
                        exit_price = max(o_price, target_threshold) * fee_mult
                        exit_reason = "Target Achieved"

            # 4. Max Holding Bars
            if exit_price is None and cfg.max_holding_bars > 0 and open_trade.duration_bars >= cfg.max_holding_bars:
                exit_price = c_price * fee_mult
                exit_reason = "Time Exit"

            # 5. Signal Exit / Reversal
            if exit_price is None and exits[i]:
                exit_price = c_price * fee_mult
                exit_reason = "Signal Exit"

            # Execute exit if triggered
            if exit_price is not None:
                open_trade.exit_time = bar_date
                open_trade.exit_price = round(exit_price, 2)
                open_trade.exit_reason = exit_reason
                pnl = (open_trade.exit_price - open_trade.entry_price) * qty
                open_trade.pnl_rupees = round(pnl, 2)
                open_trade.pnl_percent = round((open_trade.exit_price / open_trade.entry_price - 1.0) * 100.0, 2)
                
                # Risk-to-Reward calculation
                eff_stop_pct = ((open_trade.entry_price - weekly_s_05) / open_trade.entry_price * 100.0) if (use_cpr and weekly_s_05 is not None and weekly_s_05 > 0 and weekly_s_05 < open_trade.entry_price) else cfg.stop_loss_pct
                eff_target_pct = ((weekly_r1 - open_trade.entry_price) / open_trade.entry_price * 100.0) if (use_cpr and weekly_r1 is not None and weekly_r1 > open_trade.entry_price) else cfg.target_pct

                if eff_stop_pct > 0:
                    r_mult = open_trade.pnl_percent / eff_stop_pct
                    open_trade.realized_rr = round(r_mult, 2)
                    if r_mult >= 0:
                        open_trade.risk_reward = f"1 : {r_mult:.2f} (+{r_mult:.2f}R)"
                    else:
                        open_trade.risk_reward = f"-1 : {abs(r_mult):.2f} ({r_mult:.2f}R)"
                if eff_stop_pct > 0 and eff_target_pct > 0:
                    open_trade.planned_rr = round(eff_target_pct / eff_stop_pct, 2)

                trades.append(open_trade)
                cash += open_trade.exit_price * qty
                open_trade = None

        # Check for new Long Entry if not currently in a position
        if open_trade is None and entries[i]:
            entry_price = c_price * fee_add
            # Position sizing
            if cfg.position_sizing == "fixed_amount":
                alloc_capital = min(cash, cfg.size_value)
            elif cfg.position_sizing == "fixed_shares":
                alloc_capital = min(cash, cfg.size_value * entry_price)
            else:  # "percent"
                alloc_capital = cash * (min(100.0, max(5.0, cfg.size_value)) / 100.0)

            qty = int(alloc_capital // entry_price)
            if qty > 0:
                cost = qty * entry_price
                cash -= cost

                # Initial planned R:R calculation at entry
                trade_w_r1 = df_sig["Weekly_R1"].iloc[i] if ("Weekly_R1" in df_sig.columns and not pd.isna(df_sig["Weekly_R1"].iloc[i])) else None
                trade_w_s05 = df_sig["Weekly_S_05"].iloc[i] if ("Weekly_S_05" in df_sig.columns and not pd.isna(df_sig["Weekly_S_05"].iloc[i])) else None
                use_cpr_entry = cfg.use_cpr_exits or (cfg.strategy_type == "Chartink_75_Waterfall")
                init_stop_pct = ((entry_price - trade_w_s05) / entry_price * 100.0) if (use_cpr_entry and trade_w_s05 is not None and trade_w_s05 > 0 and trade_w_s05 < entry_price) else cfg.stop_loss_pct
                init_tgt_pct = ((trade_w_r1 - entry_price) / entry_price * 100.0) if (use_cpr_entry and trade_w_r1 is not None and trade_w_r1 > entry_price) else cfg.target_pct
                init_planned_rr = round(init_tgt_pct / init_stop_pct, 2) if (init_stop_pct > 0 and init_tgt_pct > 0) else 0.0

                open_trade = Trade(
                    trade_id=trade_id_counter,
                    symbol=symbol,
                    direction="LONG",
                    entry_time=bar_date,
                    entry_price=round(entry_price, 2),
                    quantity=qty,
                    planned_rr=init_planned_rr
                )
                trade_id_counter += 1
                highest_price_in_trade = h_price

        # Record daily equity point
        if open_trade is not None:
            unrealized = (c_price * open_trade.quantity)
            current_equity = cash + unrealized
        else:
            current_equity = cash

        equity_points.append({
            "time": bar_date,
            "equity": round(current_equity, 2),
            "cash": round(cash, 2),
            "in_position": open_trade is not None
        })

    # Close open position at end of backtest data
    if open_trade is not None:
        last_price = closes[-1] * fee_mult
        open_trade.exit_time = dates[-1]
        open_trade.exit_price = round(last_price, 2)
        open_trade.exit_reason = "End of Data"
        pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.quantity
        open_trade.pnl_rupees = round(pnl, 2)
        open_trade.pnl_percent = round((open_trade.exit_price / open_trade.entry_price - 1.0) * 100.0, 2)

        # Risk-to-Reward calculation
        if cfg.stop_loss_pct > 0:
            r_mult = open_trade.pnl_percent / cfg.stop_loss_pct
            open_trade.realized_rr = round(r_mult, 2)
            if r_mult >= 0:
                open_trade.risk_reward = f"1 : {r_mult:.2f} (+{r_mult:.2f}R)"
            else:
                open_trade.risk_reward = f"-1 : {abs(r_mult):.2f} ({r_mult:.2f}R)"
        if cfg.stop_loss_pct > 0 and cfg.target_pct > 0:
            open_trade.planned_rr = round(cfg.target_pct / cfg.stop_loss_pct, 2)

        trades.append(open_trade)
        cash += open_trade.exit_price * open_trade.quantity
        current_equity = cash
        if equity_points:
            equity_points[-1]["equity"] = round(current_equity, 2)

    # Build Equity Curve DataFrame & Drawdown
    eq_df = pd.DataFrame(equity_points)
    if not eq_df.empty:
        if "time" in eq_df.columns:
            eq_df.index = pd.to_datetime(eq_df["time"])
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["high_watermark"] = eq_df["peak"]
        eq_df["drawdown_rupees"] = eq_df["peak"] - eq_df["equity"]
        eq_df["drawdown_pct"] = (eq_df["drawdown_rupees"] / eq_df["peak"]) * 100.0
        max_dd_pct = float(eq_df["drawdown_pct"].max())
        max_dd_rupees = float(eq_df["drawdown_rupees"].max())
    else:
        eq_df = pd.DataFrame(columns=["time", "equity", "cash", "in_position", "peak", "high_watermark", "drawdown_rupees", "drawdown_pct"])
        max_dd_pct = 0.0
        max_dd_rupees = 0.0

    # Summary Performance Metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl_rupees > 0])
    losing_trades = len([t for t in trades if t.pnl_rupees <= 0])
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    total_net_pnl = current_equity - initial_capital
    total_net_pnl_pct = (total_net_pnl / initial_capital) * 100.0

    gross_gains = sum([t.pnl_rupees for t in trades if t.pnl_rupees > 0])
    gross_losses = abs(sum([t.pnl_rupees for t in trades if t.pnl_rupees < 0]))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

    avg_win = (gross_gains / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (gross_losses / losing_trades) if losing_trades > 0 else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    avg_trade_pnl = (total_net_pnl / total_trades) if total_trades > 0 else 0.0
    expectancy = ((win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss)) if total_trades > 0 else 0.0

    planned_rr = round(cfg.target_pct / cfg.stop_loss_pct, 2) if (cfg.stop_loss_pct > 0 and cfg.target_pct > 0) else 0.0
    r_multiples = [t.realized_rr for t in trades if hasattr(t, "realized_rr") and t.realized_rr != 0.0]
    avg_rr = round(float(np.mean(r_multiples)), 2) if r_multiples else 0.0

    return BacktestResult(
        symbol=symbol,
        strategy_name=cfg.name,
        timeframe=timeframe,
        start_time=dates[0] if len(dates) > 0 else None,
        end_time=dates[-1] if len(dates) > 0 else None,
        initial_capital=round(initial_capital, 2),
        final_equity=round(current_equity, 2),
        total_net_pnl=round(total_net_pnl, 2),
        total_net_pnl_pct=round(total_net_pnl_pct, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_rupees=round(max_dd_rupees, 2),
        avg_trade_pnl=round(avg_trade_pnl, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        win_loss_ratio=round(win_loss_ratio, 2),
        expectancy=round(expectancy, 2),
        trades=trades,
        equity_curve=eq_df,
        indicators_df=df_sig,
        avg_risk_reward=avg_rr,
        planned_risk_reward=planned_rr
    )


def run_basket_backtest(
    symbol_dfs: Dict[str, pd.DataFrame],
    cfg: StrategyConfig,
    timeframe: str = "Daily",
    capital_per_stock: float = 50000.0,
    slippage_pct: float = 0.05
) -> Dict[str, Any]:
    """
    Runs strategy backtesting concurrently across a dictionary of stock DataFrames.
    Aggregates multi-symbol portfolio results and provides stock ranking.
    """
    results: List[BacktestResult] = []

    for sym, df in symbol_dfs.items():
        if df is not None and len(df) >= 20:
            res = run_backtest(
                df=df,
                cfg=cfg,
                symbol=sym,
                timeframe=timeframe,
                initial_capital=capital_per_stock,
                slippage_pct=slippage_pct
            )
            results.append(res)

    if not results:
        return {
            "total_symbols": 0,
            "total_capital": 0.0,
            "final_equity": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "leaderboard": pd.DataFrame(),
            "results": []
        }

    total_cap = capital_per_stock * len(results)
    final_eq = sum([r.final_equity for r in results])
    net_pnl = final_eq - total_cap
    net_pnl_pct = (net_pnl / total_cap * 100.0) if total_cap > 0 else 0.0

    all_trades_count = sum([r.total_trades for r in results])
    all_wins = sum([r.winning_trades for r in results])
    overall_win_rate = (all_wins / all_trades_count * 100.0) if all_trades_count > 0 else 0.0

    # Build Leaderboard
    records = []
    for r in results:
        records.append({
            "Symbol": r.symbol,
            "Net P&L (₹)": r.total_net_pnl,
            "Return (%)": r.total_net_pnl_pct,
            "Win Rate (%)": r.win_rate,
            "Trades": r.total_trades,
            "Wins": r.winning_trades,
            "Losses": r.losing_trades,
            "Profit Factor": r.profit_factor,
            "Max DD (%)": r.max_drawdown_pct,
            "Avg Risk:Reward": f"{r.avg_risk_reward:+.2f}R" if r.avg_risk_reward != 0 else (f"1 : {r.win_loss_ratio:.2f}" if r.win_loss_ratio > 0 else "-")
        })

    leaderboard = pd.DataFrame(records)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(by="Net P&L (₹)", ascending=False).reset_index(drop=True)

    return {
        "total_symbols": len(results),
        "total_capital": round(total_cap, 2),
        "final_equity": round(final_eq, 2),
        "total_pnl": round(net_pnl, 2),
        "total_pnl_pct": round(net_pnl_pct, 2),
        "total_trades": all_trades_count,
        "win_rate": round(overall_win_rate, 1),
        "leaderboard": leaderboard,
        "results": results
    }


def get_preset_strategy(preset_name: str) -> StrategyConfig:
    """
    Returns pre-configured StrategyConfig for popular, battle-tested setups.
    """
    p_lower = preset_name.lower().replace(" ", "_").replace("-", "_")
    if "19122704" in p_lower or "intraday_scan" in p_lower:
        return StrategyConfig(
            strategy_type="Chartink_Intraday_Scan_19122704",
            name="⚡ Chartink Intraday 75m Scan (19122704 Breakdown)",
            use_cpr_exits=True,
            cpr_target_level="S1",
            cpr_stop_level="R_05",
            target_pct=3.0,
            stop_loss_pct=1.5,
            use_trailing_stop=True,
            trailing_stop_pct=1.5,
            exit_on_signal_reversal=True
        )
    elif "waterfall" in p_lower or "chartink" in p_lower:
        return StrategyConfig(
            strategy_type="Chartink_75_Waterfall",
            name="🏆 Chartink 75m Waterfall (Weekly CPR R1 / 0.5 SL)",
            use_cpr_exits=True,
            cpr_target_level="R1",
            cpr_stop_level="S_05",
            target_pct=5.0,
            stop_loss_pct=2.5,
            use_trailing_stop=False,
            exit_on_signal_reversal=True
        )
    elif "hilega" in p_lower:
        return StrategyConfig(
            strategy_type="Hilega_Milega",
            name="Hilega Milega Momentum (NK Sir)",
            rsi_span=9,
            target_pct=5.0,
            stop_loss_pct=2.5,
            use_trailing_stop=True,
            trailing_stop_pct=2.0,
            exit_on_signal_reversal=True
        )
    elif "triple_ema" in p_lower or "ribbon" in p_lower:
        return StrategyConfig(
            strategy_type="Triple_EMA",
            name="Triple EMA Ribbon Trend Rider",
            ema_fast=9,
            ema_mid=20,
            ema_slow=50,
            target_pct=6.0,
            stop_loss_pct=2.5,
            use_trailing_stop=True,
            trailing_stop_pct=2.0,
            exit_on_signal_reversal=True
        )
    elif "supertrend" in p_lower:
        return StrategyConfig(
            strategy_type="SuperTrend",
            name="SuperTrend Volatility Rider (10, 3.0)",
            st_period=10,
            st_multiplier=3.0,
            target_pct=8.0,
            stop_loss_pct=3.0,
            use_trailing_stop=False,
            exit_on_signal_reversal=True
        )
    elif "macd" in p_lower:
        return StrategyConfig(
            strategy_type="MACD_Cross",
            name="MACD Bullish Crossover",
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            target_pct=4.5,
            stop_loss_pct=2.0,
            use_trailing_stop=True,
            trailing_stop_pct=1.5,
            exit_on_signal_reversal=True
        )
    elif "cpr" in p_lower:
        return StrategyConfig(
            strategy_type="CPR_Breakout",
            name="CPR / Range Breakout Strategy",
            target_pct=5.0,
            stop_loss_pct=2.0,
            use_trailing_stop=True,
            trailing_stop_pct=1.5,
            exit_on_signal_reversal=True
        )
    else:  # Default Custom
        return StrategyConfig(
            strategy_type="Custom",
            name="Custom User Strategy",
            custom_ind1="RSI",
            custom_op="crosses_above",
            custom_ind2="WMA_21",
            custom_val=50.0,
            target_pct=4.0,
            stop_loss_pct=2.0
        )
