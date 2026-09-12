"""
strategy_ui.py
--------------
Rich interactive UI for the Strategy Lab & Testing page in Upstox Scanner.
Supports Single-Stock Backtesting, Multi-Stock Basket Simulations,
and Strategy Comparison with interactive Plotly charts, KPI metrics,
candlestick trade overlays, and CSV export.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import config
import database
import scanner
import parquet_loader
import tradingview_charts
from strategy_engine import (
    StrategyConfig,
    BacktestResult,
    prepare_indicators,
    generate_strategy_signals,
    run_backtest,
    run_basket_backtest,
    get_preset_strategy,
)


def calculate_advanced_metrics(result: BacktestResult, initial_capital: float, df_ind: pd.DataFrame = None) -> dict:
    """Calculates hedge-fund grade risk and performance metrics (CAGR, Sharpe, Sortino, Calmar, Streaks)."""
    eq_df = result.equity_curve
    if eq_df is None or eq_df.empty or len(eq_df) < 2:
        return {
            "cagr_pct": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "max_dd_bars": 0, "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "payoff_ratio": 0.0, "bench_return_pct": 0.0, "alpha_pct": 0.0
        }

    start_dt = pd.to_datetime(eq_df.index[0])
    end_dt = pd.to_datetime(eq_df.index[-1])
    total_days = max(1, (end_dt - start_dt).days)
    years = total_days / 365.25

    # 1. CAGR
    final_eq = float(result.final_equity)
    init_cap = float(initial_capital)
    if years > 0 and init_cap > 0 and final_eq > 0:
        cagr_pct = ((final_eq / init_cap) ** (1.0 / max(years, 0.05)) - 1.0) * 100.0
    else:
        cagr_pct = result.total_net_pnl_pct

    # 2. Sharpe & Sortino (annualized with 6% risk-free rate)
    pct_changes = eq_df["equity"].pct_change().dropna()
    rf_daily = 0.06 / 252.0
    excess_returns = pct_changes - rf_daily

    std_ret = pct_changes.std()
    if std_ret > 0 and len(pct_changes) > 5:
        sharpe = float((excess_returns.mean() / std_ret) * np.sqrt(252))
    else:
        sharpe = 0.0

    downside_returns = pct_changes[pct_changes < 0]
    downside_std = downside_returns.std()
    if downside_std > 0 and len(downside_returns) > 3:
        sortino = float((excess_returns.mean() / downside_std) * np.sqrt(252))
    else:
        sortino = 0.0

    # 3. Calmar Ratio
    max_dd = abs(float(result.max_drawdown_pct))
    calmar = (cagr_pct / max_dd) if max_dd > 0 else 0.0

    # 4. Max Drawdown Duration (consecutive bars in drawdown)
    in_dd = eq_df["drawdown_pct"] > 0
    max_dd_bars = 0
    curr_dd_bars = 0
    for val in in_dd:
        if val:
            curr_dd_bars += 1
            if curr_dd_bars > max_dd_bars:
                max_dd_bars = curr_dd_bars
        else:
            curr_dd_bars = 0

    # 5. Consecutive Wins and Losses
    max_wins, max_losses = 0, 0
    curr_w, curr_l = 0, 0
    for t in result.trades:
        if t.pnl_percent >= 0:
            curr_w += 1
            curr_l = 0
            if curr_w > max_wins:
                max_wins = curr_w
        else:
            curr_l += 1
            curr_w = 0
            if curr_l > max_losses:
                max_losses = curr_l

    # 6. Payoff Ratio
    payoff = (abs(result.avg_win) / abs(result.avg_loss)) if (result.avg_loss and abs(result.avg_loss) > 0) else 0.0

    # 7. Benchmark Buy & Hold Return
    bench_return_pct = 0.0
    if df_ind is not None and not df_ind.empty and "close" in df_ind.columns:
        c_first = df_ind["close"].dropna().iloc[0]
        c_last = df_ind["close"].dropna().iloc[-1]
        if c_first > 0:
            bench_return_pct = ((c_last - c_first) / c_first) * 100.0

    alpha_pct = result.total_net_pnl_pct - bench_return_pct

    return {
        "cagr_pct": round(cagr_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "max_dd_bars": max_dd_bars,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "payoff_ratio": round(payoff, 2),
        "bench_return_pct": round(bench_return_pct, 2),
        "alpha_pct": round(alpha_pct, 2)
    }


def calculate_monthly_returns_matrix(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Computes month-by-month and year-by-year return % matrix for heatmap presentation."""
    if equity_df is None or equity_df.empty or len(equity_df) < 5:
        return pd.DataFrame()

    df = equity_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Resample to end of month equity
    m_df = df["equity"].resample("ME").last().dropna()
    if m_df.empty:
        return pd.DataFrame()

    first_eq = df["equity"].iloc[0]
    returns = m_df.pct_change() * 100.0
    if not m_df.empty and first_eq > 0:
        returns.iloc[0] = ((m_df.iloc[0] - first_eq) / first_eq) * 100.0

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    matrix = {}

    for dt, ret in returns.items():
        yr = dt.year
        m_name = month_names[dt.month - 1]
        if yr not in matrix:
            matrix[yr] = {m: np.nan for m in month_names}
            matrix[yr]["Year"] = yr
        matrix[yr][m_name] = round(float(ret), 2)

    if not matrix:
        return pd.DataFrame()

    res_df = pd.DataFrame(list(matrix.values()))

    annual_totals = []
    for idx, row in res_df.iterrows():
        yr = row["Year"]
        yr_pts = df[df.index.year == yr]["equity"]
        if not yr_pts.empty and yr_pts.iloc[0] > 0:
            yr_ret = ((yr_pts.iloc[-1] - yr_pts.iloc[0]) / yr_pts.iloc[0]) * 100.0
        else:
            yr_ret = 0.0
        annual_totals.append(round(yr_ret, 2))

    res_df["Year Return"] = annual_totals
    res_df.sort_values("Year", ascending=False, inplace=True)
    cols = ["Year"] + month_names + ["Year Return"]
    return res_df[cols]


def calculate_exit_reason_breakdown(trades: list, stop_loss_pct: float = 0.0) -> pd.DataFrame:
    """Aggregates trades by exit reason for risk diagnostics including Risk:Reward."""
    if not trades:
        return pd.DataFrame()
    reasons = {}
    for t in trades:
        r = t.exit_reason or "Signal Reversal"
        if r not in reasons:
            reasons[r] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "total_pct": 0.0, "r_list": []}
        reasons[r]["count"] += 1
        if t.pnl_percent >= 0:
            reasons[r]["wins"] += 1
        else:
            reasons[r]["losses"] += 1
        reasons[r]["total_pnl"] += t.pnl_rupees
        reasons[r]["total_pct"] += t.pnl_percent
        if hasattr(t, "realized_rr") and t.realized_rr != 0.0:
            reasons[r]["r_list"].append(t.realized_rr)
        elif stop_loss_pct > 0:
            reasons[r]["r_list"].append(t.pnl_percent / stop_loss_pct)

    records = []
    for r, d in reasons.items():
        cnt = d["count"]
        avg_r = (sum(d["r_list"]) / len(d["r_list"])) if d["r_list"] else 0.0
        records.append({
            "Exit Reason": r,
            "Trades": cnt,
            "Win Rate (%)": round(d["wins"] / cnt * 100.0, 1),
            "Total P&L (₹)": round(d["total_pnl"], 2),
            "Avg Return (%)": round(d["total_pct"] / cnt, 2),
            "Avg Risk:Reward": f"{avg_r:+.2f}R" if avg_r != 0 else "-",
            "Wins": d["wins"],
            "Losses": d["losses"]
        })
    res_df = pd.DataFrame(records)
    res_df.sort_values("Trades", ascending=False, inplace=True)
    return res_df


def _get_theme_colors(theme: str) -> dict:
    is_light = (str(theme).lower() == "light")
    return {
        "is_light": is_light,
        "template": "plotly_white" if is_light else "plotly_dark",
        "paper_bg": "#FFFFFF" if is_light else "#131722",
        "plot_bg": "#FFFFFF" if is_light else "#131722",
        "grid_clr": "#E2E8F0" if is_light else "#2A2E39",
        "card_bg": "#F8FAFC" if is_light else "#1E222D",
        "card_border": "#CBD5E1" if is_light else "#2A2E39",
        "text_primary": "#0F172A" if is_light else "#F8FAFC",
        "text_secondary": "#64748B" if is_light else "#94A3B8",
        "green": "#10B981",
        "red": "#EF4444",
        "blue": "#3B82F6",
        "orange": "#F59E0B",
        "purple": "#8B5CF6",
    }


def render_metric_card(title: str, value: str, subtext: str = "", delta_color: str = "normal", tc: dict = None):
    bg = tc["card_bg"] if tc else "#1E222D"
    border = tc["card_border"] if tc else "#2A2E39"
    text_color = tc["text_primary"] if tc else "#FFFFFF"
    sub_color = tc["text_secondary"] if tc else "#94A3B8"
    
    val_color = text_color
    if delta_color == "green":
        val_color = "#10B981"
    elif delta_color == "red":
        val_color = "#EF4444"
    elif delta_color == "blue":
        val_color = "#3B82F6"

    html = f"""
    <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: {sub_color}; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 24px; font-weight: 800; color: {val_color}; margin: 4px 0 2px 0;">{value}</div>
        <div style="font-size: 11.5px; color: {sub_color};">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_equity_and_drawdown(result: BacktestResult, tc: dict, benchmark_df: pd.DataFrame = None) -> go.Figure:
    eq_df = result.equity_curve
    if eq_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trades executed", showarrow=False)
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
        subplot_titles=(f"📈 Portfolio Equity Curve (Initial: ₹{result.initial_capital:,.0f} ➔ Final: ₹{result.final_equity:,.0f})", "🔻 Underwater Drawdown (%)")
    )

    x_vals = eq_df.index
    # Equity curve line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=eq_df["equity"],
            mode="lines",
            name="Strategy Equity (₹)",
            line=dict(color=tc["blue"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)"
        ),
        row=1, col=1
    )

    # High watermark line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=eq_df["high_watermark"],
            mode="lines",
            name="Peak Equity",
            line=dict(color="rgba(148, 163, 184, 0.6)", width=1.5, dash="dash")
        ),
        row=1, col=1
    )

    # Benchmark line (Buy & Hold)
    if benchmark_df is not None and not benchmark_df.empty and "close" in benchmark_df.columns:
        b_clean = benchmark_df.dropna(subset=["close"])
        if not b_clean.empty:
            c0 = b_clean["close"].iloc[0]
            if c0 > 0:
                bench_vals = result.initial_capital * (b_clean["close"] / c0)
                fig.add_trace(
                    go.Scatter(
                        x=b_clean.index,
                        y=bench_vals,
                        mode="lines",
                        name="Buy & Hold Benchmark",
                        line=dict(color="#A855F7", width=1.5, dash="dot")
                    ),
                    row=1, col=1
                )

    # Drawdown % area
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=-eq_df["drawdown_pct"],
            mode="lines",
            name="Drawdown (%)",
            line=dict(color=tc["red"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template=tc["template"],
        paper_bgcolor=tc["paper_bg"],
        plot_bgcolor=tc["plot_bg"],
        height=480,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=tc["grid_clr"])
    fig.update_yaxes(gridcolor=tc["grid_clr"])
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig


def plot_candlesticks_with_trade_markers(
    df_ind: pd.DataFrame,
    result: BacktestResult,
    cfg: StrategyConfig,
    tc: dict
) -> go.Figure:
    """Renders Candlesticks with Buy (▲) and Sell (▼) overlays and relevant indicators."""
    if df_ind.empty:
        fig = go.Figure()
        return fig

    show_rsi_subplot = ("RSI" in df_ind.columns and cfg.strategy_type in ("Hilega_Milega", "Custom"))
    
    if show_rsi_subplot:
        rows = 2
        row_heights = [0.72, 0.28]
        subtitles = (f"Candlesticks with Executed Trades ({result.symbol} - {result.timeframe})", "Hilega Milega / RSI")
    else:
        rows = 1
        row_heights = [1.0]
        subtitles = (f"Candlesticks with Executed Trades ({result.symbol} - {result.timeframe})",)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=subtitles
    )

    x_vals = df_ind.index

    # Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df_ind["open"],
            high=df_ind["high"],
            low=df_ind["low"],
            close=df_ind["close"],
            name="Price",
            increasing_line_color=tc["green"],
            decreasing_line_color=tc["red"],
            increasing_fillcolor=tc["green"],
            decreasing_fillcolor=tc["red"],
        ),
        row=1, col=1
    )

    # Indicator Overlays on Price
    if "EMA_Fast" in df_ind.columns and cfg.strategy_type in ("Triple_EMA", "Custom"):
        fig.add_trace(go.Scatter(x=x_vals, y=df_ind["EMA_Fast"], name=f"EMA {cfg.ema_fast}", line=dict(color="#38BDF8", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=df_ind["EMA_Mid"], name=f"EMA {cfg.ema_mid}", line=dict(color="#F59E0B", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=df_ind["EMA_Slow"], name=f"EMA {cfg.ema_slow}", line=dict(color="#EC4899", width=1.4)), row=1, col=1)

    if "SuperTrend" in df_ind.columns and cfg.strategy_type in ("SuperTrend", "Custom"):
        fig.add_trace(go.Scatter(x=x_vals, y=df_ind["SuperTrend"], name="SuperTrend", line=dict(color="#10B981", width=1.5, dash="dot")), row=1, col=1)

    # Trade Buy and Sell Markers
    buys_x, buys_y, buys_txt = [], [], []
    sells_x, sells_y, sells_txt = [], [], []

    for t in result.trades:
        buys_x.append(t.entry_time)
        buys_y.append(t.entry_price * 0.99)
        buys_txt.append(f"BUY #{t.trade_id}<br>₹{t.entry_price:,.2f}<br>Qty: {t.quantity}")

        if t.exit_time:
            sells_x.append(t.exit_time)
            sells_y.append(t.exit_price * 1.01)
            pnl_sign = "+" if t.pnl_percent >= 0 else ""
            sells_txt.append(f"EXIT #{t.trade_id} ({t.exit_reason})<br>₹{t.exit_price:,.2f}<br>P&L: {pnl_sign}{t.pnl_percent:.2f}% (₹{t.pnl_rupees:,.2f})")

    if buys_x:
        fig.add_trace(
            go.Scatter(
                x=buys_x,
                y=buys_y,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=13, color=tc["green"], line=dict(width=1, color="#FFFFFF")),
                text=["▲" for _ in buys_x],
                textposition="bottom center",
                name="Buy Entry",
                hovertext=buys_txt,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    if sells_x:
        fig.add_trace(
            go.Scatter(
                x=sells_x,
                y=sells_y,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=13, color=tc["red"], line=dict(width=1, color="#FFFFFF")),
                text=["▼" for _ in sells_x],
                textposition="top center",
                name="Sell Exit",
                hovertext=sells_txt,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    # Subplot: RSI / Hilega Milega
    if show_rsi_subplot:
        fig.add_trace(go.Scatter(x=x_vals, y=df_ind["RSI"], name="RSI", line=dict(color="#10B981", width=1.5)), row=2, col=1)
        if "RSI_EMA3" in df_ind.columns:
            fig.add_trace(go.Scatter(x=x_vals, y=df_ind["RSI_EMA3"], name="RSI EMA 3", line=dict(color="#EF4444", width=1.2)), row=2, col=1)
        if "RSI_WMA21" in df_ind.columns:
            fig.add_trace(go.Scatter(x=x_vals, y=df_ind["RSI_WMA21"], name="RSI WMA 21", line=dict(color="#3B82F6", width=1.2)), row=2, col=1)
        # Reference lines 50, 70, 30
        fig.add_hline(y=50, line=dict(color="rgba(148, 163, 184, 0.6)", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(16, 185, 129, 0.4)", width=1, dash="dot"), row=2, col=1)
        fig.update_yaxes(range=[15, 85], row=2, col=1)

    fig.update_layout(
        template=tc["template"],
        paper_bgcolor=tc["paper_bg"],
        plot_bgcolor=tc["plot_bg"],
        height=580,
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor=tc["grid_clr"])
    fig.update_yaxes(gridcolor=tc["grid_clr"])
    return fig


def load_candles_for_simulation(symbol: str, timeframe: str, custom_minutes: int = 15) -> pd.DataFrame:
    """Loads and resamples authentic candles for simulation across any timeframe (Daily, Weekly, Monthly, or custom minutes)."""
    import re
    tf_str = str(timeframe).strip()
    if tf_str == "Weekly":
        daily = database.get_candles_df(symbol)
        return scanner.resample_ohlcv(daily, "weekly") if (daily is not None and not daily.empty) else pd.DataFrame()
    elif tf_str == "Monthly":
        daily = database.get_candles_df(symbol)
        return scanner.resample_ohlcv(daily, "monthly") if (daily is not None and not daily.empty) else pd.DataFrame()
    elif tf_str == "Daily":
        daily = database.get_candles_df(symbol)
        return daily if (daily is not None and not daily.empty) else pd.DataFrame()
    elif tf_str in ("75-Min", "75m"):
        df_75 = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=50)
        if df_75 is not None and not df_75.empty:
            return df_75

    # Intraday minutes extraction (e.g. 1-Min, 3-Min, 5-Min, 10-Min, 15-Min, 30-Min, 45-Min, 60-Min, 125-Min, Custom)
    minutes = custom_minutes
    m = re.search(r"(\d+)", tf_str)
    if m:
        minutes = int(m.group(1))
    elif "custom" in tf_str.lower():
        minutes = custom_minutes

    try:
        df_intra = parquet_loader.ensure_symbol_custom_minute_candles(symbol, interval_minutes=minutes, min_bars=20)
        if df_intra is not None and not df_intra.empty:
            return df_intra
    except Exception as e:
        logger.warning(f"Note loading {minutes}-min candles for {symbol}: {e}")

    # Fallback to daily
    daily = database.get_candles_df(symbol)
    return daily if (daily is not None and not daily.empty) else pd.DataFrame()


def render_strategy_lab_page(theme: str = "dark"):
    """Main entry point rendering the Strategy Lab & Testing Page."""
    tc = _get_theme_colors(theme)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <h2 style="margin: 0; color: {tc['text_primary']};">🧪 Strategy Lab & Testing</h2>
            <p style="margin: 0; color: {tc['text_secondary']}; font-size: 14px;">
                Visual Strategy Builder, Bar-by-Bar Simulator, and Multi-Stock Portfolio Backtester.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Available Stocks Universe
    db_symbols = database.get_all_symbols()
    parquet_symbols = parquet_loader.get_parquet_symbols()
    popular_indices = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LICI", "LT", "TATAMOTORS"]
    all_symbols = sorted(list(set(popular_indices + db_symbols + parquet_symbols)))
    all_symbols = [s for s in all_symbols if not s.startswith("0")]

    if not all_symbols:
        st.warning("No candle data available in database or parquet. Please sync data first.")
        return

    # 2. Strategy Builder Configuration Header
    st.markdown("---")
    with st.expander("🛠️ **Strategy Builder & Risk Settings**", expanded=True):
        col_st1, col_st2, col_st3 = st.columns([1.8, 1.4, 1.4])

        with col_st1:
            strategy_options = [
                "🏆 Chartink 75m Waterfall (Weekly CPR R1 / 0.5 SL)",
                "⚡ Hilega Milega Momentum (NK Sir)",
                "🌊 Triple EMA Ribbon Trend (9/20/50)",
                "🏹 SuperTrend Volatility Rider (10, 3.0)",
                "📊 MACD Bullish Crossover",
                "🎯 CPR / Range Breakout",
                "🛠️ Custom Rule Builder"
            ]
            selected_strat_name = st.selectbox(
                "Select Strategy Preset / Engine:",
                options=strategy_options,
                index=0,
                key="slab_strat_choice"
            )

        with col_st2:
            initial_cap = st.number_input("Initial Capital (₹):", min_value=10000.0, max_value=10000000.0, value=100000.0, step=10000.0)

        with col_st3:
            slippage_pct = st.number_input("Slippage & Brokerage (%) / trade:", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help="0.05% applied to buy + sell = 0.10% roundtrip friction")

        # Config instantiation from preset
        cfg = get_preset_strategy(selected_strat_name)
        cfg.slippage_brokerage_pct = slippage_pct

        # Waterfall Strategy Info Banner
        if "Waterfall" in selected_strat_name:
            st.markdown(f"""
            <div style="margin-top: 6px; margin-bottom: 12px; padding: 10px 16px; background: rgba(59, 130, 246, 0.08); border-left: 4px solid #38BDF8; border-radius: 6px;">
                <div style="font-weight: 700; color: #38BDF8; font-size: 14px; margin-bottom: 4px;">
                    🏆 Chartink 75-Min Waterfall Positional Strategy (Scan #364)
                </div>
                <div style="font-size: 12.5px; color: {tc['text_secondary']}; line-height: 1.5;">
                    • <b>Direction</b>: Long Only (Buy Only) on Stage 4 alignment (Monthly + Weekly + Daily + 75m Trigger).<br>
                    • <b>🎯 Take Profit (TP)</b>: <b>Weekly CPR R1</b> resistance level (from prior week's range).<br>
                    • <b>🛑 Stop Loss (SL)</b>: <b>Weekly CPR 0.5 Support</b> midpoint between Weekly Pivot and S1: <code>(P + S1) / 2</code>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Custom Rule Builder Panel
        if "Custom" in selected_strat_name:
            st.markdown("##### ⚙️ Custom Rule Definition")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cfg.custom_ind1 = st.selectbox("Trigger Indicator:", ["RSI", "Close", "EMA_Fast", "MACD"], index=0)
            with c2:
                cfg.custom_op = st.selectbox("Operator:", ["crosses_above", "crosses_below", "greater_than", "less_than"], index=0)
            with c3:
                cfg.custom_ind2 = st.selectbox("Target Benchmark:", ["WMA_21", "EMA_Mid", "EMA_Slow", "Value (Fixed Number)"], index=0)
            with c4:
                cfg.custom_val = float(st.number_input("Fixed Value (if selected):", value=50.0, step=1.0))

        # Risk Management Controls Row
        st.markdown("##### 🛡️ Risk Management & Exit Rules")
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            cfg.target_pct = float(st.number_input("Profit Target (%):", min_value=0.0, max_value=100.0, value=float(cfg.target_pct), step=0.5, help="0 to disable fixed target (Weekly CPR R1 takes precedence for Waterfall)"))
        with r_col2:
            cfg.stop_loss_pct = float(st.number_input("Stop Loss (%):", min_value=0.0, max_value=50.0, value=float(cfg.stop_loss_pct), step=0.5, help="0 to disable fixed stop loss (Weekly CPR 0.5 takes precedence for Waterfall)"))
        with r_col3:
            use_trail = st.checkbox("Trailing Stop Loss", value=cfg.use_trailing_stop)
            cfg.use_trailing_stop = use_trail
            trail_pct = float(st.number_input("Trailing Stop (%):", min_value=0.5, max_value=20.0, value=float(cfg.trailing_stop_pct), step=0.5)) if use_trail else 2.0
            cfg.trailing_stop_pct = trail_pct
        with r_col4:
            cfg.exit_on_signal_reversal = st.checkbox("Exit on Signal Reversal", value=cfg.exit_on_signal_reversal, help="Close trade immediately when strategy gives exit/reversal signal")

        if "Waterfall" in selected_strat_name:
            st.markdown(f"""
            <div style="margin-top: 4px; margin-bottom: 8px; padding: 6px 14px; background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; display: inline-flex; align-items: center; gap: 10px;">
                <span style="font-size: 13px; font-weight: 700; color: #10B981;">🎯 Dynamic Weekly CPR Exits: TP @ Weekly R1 &bull; SL @ Weekly CPR 0.5 Support</span>
                <span style="font-size: 11.5px; opacity: 0.85; color: {tc['text_secondary']};">(Planned Risk:Reward dynamically calculated per trade)</span>
            </div>
            """, unsafe_allow_html=True)
        elif cfg.stop_loss_pct > 0 and cfg.target_pct > 0:
            planned_rr_ratio = cfg.target_pct / cfg.stop_loss_pct
            st.markdown(f"""
            <div style="margin-top: 4px; margin-bottom: 8px; padding: 6px 14px; background: rgba(59, 130, 246, 0.10); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 6px; display: inline-flex; align-items: center; gap: 10px;">
                <span style="font-size: 13px; font-weight: 700; color: #38BDF8;">🎯 Planned Risk : Reward = 1 : {planned_rr_ratio:.2f}</span>
                <span style="font-size: 11.5px; opacity: 0.85; color: {tc['text_secondary']};">(Risk ₹{cfg.stop_loss_pct:.1f}% to make ₹{cfg.target_pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

    # 3. Main Tabs
    tab_single, tab_basket, tab_compare = st.tabs([
        "🎯 Single-Stock Simulation",
        "🧺 Multi-Stock Basket Backtest",
        "⚖️ Strategy Comparison"
    ])

    # ==========================================
    # TAB 1: SINGLE-STOCK SIMULATION
    # ==========================================
    with tab_single:
        tf_options = [
            "75-Min", "Daily", "125-Min", "60-Min (1h)", "45-Min", "30-Min", "15-Min", "5-Min", "3-Min", "1-Min",
            "Weekly", "Monthly", "Custom (Minutes)"
        ]
        default_tf_idx = 0 if "Waterfall" in selected_strat_name else 1
        s_col1, s_col2, s_col3, s_col4 = st.columns([1.8, 1.4, 1.4, 1.4])
        with s_col1:
            default_sym_idx = all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0
            sel_sym = st.selectbox("Select Stock:", options=all_symbols, index=default_sym_idx, key="slab_single_sym")
        with s_col2:
            sel_tf = st.selectbox("Timeframe:", options=tf_options, index=default_tf_idx, key="slab_single_tf")
            if sel_tf == "Custom (Minutes)":
                custom_min_val = st.number_input("Custom Minutes:", min_value=1, max_value=375, value=10, step=1, key="slab_single_custom_min")
                effective_tf = f"{int(custom_min_val)}-Min"
            else:
                effective_tf = sel_tf
        with s_col3:
            lookback_choice = st.selectbox("Historical Lookback:", ["All Available Data", "Past 2 Years", "Past 1 Year", "Past 6 Months"], index=0)
        with s_col4:
            st.write("")
            st.write("")
            run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True, key="slab_run_single_btn")

        # Auto-run on first load or button press
        sim_key = f"slab_result_{sel_sym}_{effective_tf}_{selected_strat_name}"
        if run_btn or sim_key not in st.session_state:
            with st.spinner(f"Simulating {selected_strat_name} on {sel_sym} ({effective_tf})..."):
                df_raw = load_candles_for_simulation(sel_sym, effective_tf)
                if df_raw.empty or len(df_raw) < 15:
                    st.warning(f"Insufficient candle data for {sel_sym} on {effective_tf}. Try syncing data or another timeframe.")
                else:
                    # Filter lookback if requested
                    if lookback_choice == "Past 6 Months":
                        cutoff = datetime.today() - pd.Timedelta(days=180)
                        df_raw = df_raw[df_raw.index >= cutoff]
                    elif lookback_choice == "Past 1 Year":
                        cutoff = datetime.today() - pd.Timedelta(days=365)
                        df_raw = df_raw[df_raw.index >= cutoff]
                    elif lookback_choice == "Past 2 Years":
                        cutoff = datetime.today() - pd.Timedelta(days=730)
                        df_raw = df_raw[df_raw.index >= cutoff]

                    # Prepare indicators & backtest
                    df_ind = prepare_indicators(df_raw, cfg)
                    result = run_backtest(
                        df_ind,
                        cfg,
                        symbol=sel_sym,
                        timeframe=effective_tf,
                        initial_capital=initial_cap,
                        slippage_pct=slippage_pct
                    )
                    st.session_state[sim_key] = (result, df_ind)

        # Render Results Dashboard
        if sim_key in st.session_state:
            res, df_ind = st.session_state[sim_key]
            adv = calculate_advanced_metrics(res, initial_cap, df_ind)

            # 1. Primary Metric Cards Row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            pnl_delta = "green" if res.total_net_pnl >= 0 else "red"
            with m1:
                render_metric_card("Net Return", f"{res.total_net_pnl_pct:+.2f}%", f"₹{res.total_net_pnl:+,.2f}", pnl_delta, tc)
            with m2:
                render_metric_card("Win Rate", f"{res.win_rate:.1f}%", f"{res.winning_trades} Won / {res.losing_trades} Lost", "blue", tc)
            with m3:
                render_metric_card("Profit Factor", f"{res.profit_factor:.2f}" if res.profit_factor > 0 else "N/A", "Gross Wins / Losses", "normal", tc)
            with m4:
                render_metric_card("Max Drawdown", f"-{res.max_drawdown_pct:.2f}%", f"₹-{res.max_drawdown_rupees:,.0f}", "red", tc)
            with m5:
                render_metric_card("Total Trades", f"{res.total_trades}", f"Avg P&L: ₹{res.avg_trade_pnl:+,.0f}", "normal", tc)
            with m6:
                render_metric_card("Trade Expectancy", f"₹{res.expectancy:+,.2f}", "Avg Profit / Trade", pnl_delta, tc)

            # 2. Portfolio Equity Curve & Underwater Drawdown Section
            st.markdown("---")
            eq_head_col1, eq_head_col2 = st.columns([3.0, 1.5])
            with eq_head_col1:
                st.markdown("#### 📈 Portfolio Equity Curve & Underwater Drawdown")
                st.caption(f"Initial Capital: **₹{res.initial_capital:,.0f}** ➔ Peak High-Watermark: **₹{res.equity_curve['high_watermark'].max():,.0f}** ➔ Final Equity: **₹{res.final_equity:,.0f}**")
            with eq_head_col2:
                eq_view_mode = st.radio(
                    "Equity View Engine:",
                    ["⚡ Interactive TradingView", "📊 Plotly Advanced"],
                    horizontal=True,
                    key=f"eq_view_engine_{sel_sym}"
                )

            if eq_view_mode == "⚡ Interactive TradingView":
                components.html(
                    tradingview_charts.generate_equity_drawdown_chart_html(
                        res.equity_curve,
                        initial_capital=initial_cap,
                        final_equity=res.final_equity,
                        symbol=res.symbol,
                        strategy_name=cfg.name,
                        benchmark_df=df_ind,
                        height=460,
                        theme=theme,
                        chart_id=f"tv_eq_{sel_sym}_{res.timeframe.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"
                    ),
                    height=480
                )
            else:
                st.plotly_chart(plot_equity_and_drawdown(res, tc, benchmark_df=df_ind), use_container_width=True)

            # 3. Institutional Risk & Performance Analytics Dashboard
            with st.expander("🛡️ **Advanced Risk & Health KPI Dashboard**", expanded=True):
                rk1, rk2, rk3, rk4 = st.columns(4)
                with rk1:
                    cagr_clr = "green" if adv["cagr_pct"] >= 0 else "red"
                    render_metric_card("CAGR (Annualized)", f"{adv['cagr_pct']:+.2f}%", f"Buy & Hold: {adv['bench_return_pct']:+.2f}% (Alpha: {adv['alpha_pct']:+.2f}%)", cagr_clr, tc)
                with rk2:
                    s_clr = "green" if adv["sharpe_ratio"] >= 1.0 else ("blue" if adv["sharpe_ratio"] >= 0.5 else "normal")
                    render_metric_card("Sharpe Ratio", f"{adv['sharpe_ratio']:.2f}", f"Sortino: {adv['sortino_ratio']:.2f} (Risk-Free: 6%)", s_clr, tc)
                with rk3:
                    render_metric_card("Calmar Ratio", f"{adv['calmar_ratio']:.2f}", f"Max DD Duration: {adv['max_dd_bars']} bars", "normal", tc)
                with rk4:
                    payoff_sub = f"Avg R:R: {res.avg_risk_reward:+.2f}R | Planned: 1:{res.planned_risk_reward:.2f}" if res.planned_risk_reward > 0 else f"Streaks: {adv['max_consecutive_wins']}W / {adv['max_consecutive_losses']}L"
                    render_metric_card("Risk:Reward & Payoff", f"1 : {adv['payoff_ratio']:.2f}", payoff_sub, "blue", tc)

            # 4. Monthly & Annual Returns Matrix (Heatmap Table)
            m_matrix = calculate_monthly_returns_matrix(res.equity_curve)
            if not m_matrix.empty:
                with st.expander("📅 **Monthly & Annual Returns Matrix (Heatmap)**", expanded=False):
                    def _color_matrix(val):
                        if pd.isna(val) or not isinstance(val, (int, float)):
                            return ""
                        if val > 5.0:
                            return "background-color: rgba(16, 185, 129, 0.35); color: #10B981; font-weight: 700;"
                        elif val > 0.0:
                            return "background-color: rgba(16, 185, 129, 0.15); color: #10B981; font-weight: 600;"
                        elif val < -5.0:
                            return "background-color: rgba(239, 68, 68, 0.35); color: #EF4444; font-weight: 700;"
                        elif val < 0.0:
                            return "background-color: rgba(239, 68, 68, 0.15); color: #EF4444; font-weight: 600;"
                        return ""

                    format_dict = {m: "{:+.2f}%" for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year Return"]}
                    format_dict["Year"] = "{:d}"

                    styled_matrix = m_matrix.style.format(format_dict, na_rep="-").map(_color_matrix, subset=[c for c in m_matrix.columns if c != "Year"])
                    st.dataframe(styled_matrix, use_container_width=True)

            # 5. Candlestick Chart with Trade Execution Markers
            st.markdown("---")
            tv_head_col1, tv_head_col2 = st.columns([3.0, 1.5])
            with tv_head_col1:
                st.markdown("#### 🕯️ Trade Executions Overlaid on Candles")
                st.caption(f"Interactive TradingView chart with official Buy (🟢 arrowUp) and Exit (🔴/🟢 arrowDown) markers, indicators, hover cards, and fullscreen mode.")
            with tv_head_col2:
                chart_view_mode = st.radio(
                    "Candle Engine:",
                    ["⚡ TradingView (Markers)", "📊 Plotly Classic"],
                    horizontal=True,
                    key=f"chart_view_engine_{sel_sym}"
                )

            if chart_view_mode == "⚡ TradingView (Markers)":
                is_intra = (res.timeframe not in ("Daily", "Weekly", "Monthly"))
                safe_tf_id = res.timeframe.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                components.html(
                    tradingview_charts.generate_strategy_backtest_chart_html(
                        df_ind,
                        trades=res.trades,
                        symbol=res.symbol,
                        strategy_name=cfg.name,
                        timeframe=res.timeframe,
                        height=640,
                        theme=theme,
                        is_intraday=is_intra,
                        chart_id=f"tv_strat_{sel_sym}_{safe_tf_id}",
                        show_volume=False
                    ),
                    height=660
                )
            else:
                st.plotly_chart(plot_candlesticks_with_trade_markers(df_ind, res, cfg, tc), use_container_width=True)

            # 6. Exit Reason & Win/Loss Diagnostics
            if res.trades:
                ex_df = calculate_exit_reason_breakdown(res.trades, stop_loss_pct=cfg.stop_loss_pct)
                if not ex_df.empty:
                    with st.expander("📊 **Exit Reason Diagnostics & Trade Distribution**", expanded=False):
                        st.dataframe(
                            ex_df.style.format({
                                "Win Rate (%)": "{:.1f}%",
                                "Total P&L (₹)": "₹{:,.2f}",
                                "Avg Return (%)": "{:+.2f}%"
                            }).map(lambda v: "color: #10B981; font-weight: 600;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: 600;" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Total P&L (₹)", "Avg Return (%)"]),
                            use_container_width=True,
                            hide_index=True
                        )

            # 7. Complete Trade Log Table
            st.markdown("#### 📜 Complete Trade Log")
            if res.trades:
                trade_rows = []
                planned_rr_str = f"1 : {cfg.target_pct / cfg.stop_loss_pct:.2f}" if (cfg.stop_loss_pct > 0 and cfg.target_pct > 0) else "-"
                for t in res.trades:
                    if hasattr(t, "risk_reward") and t.risk_reward:
                        rr_display = t.risk_reward
                    elif cfg.stop_loss_pct > 0:
                        r_mult = t.pnl_percent / cfg.stop_loss_pct
                        rr_display = f"1 : {r_mult:.2f} (+{r_mult:.2f}R)" if r_mult >= 0 else f"-1 : {abs(r_mult):.2f} ({r_mult:.2f}R)"
                    else:
                        rr_display = "-"

                    trade_rows.append({
                        "Trade #": t.trade_id,
                        "Type": t.direction,
                        "Entry Date": pd.to_datetime(t.entry_time).strftime("%d %b %Y %H:%M") if hasattr(t.entry_time, "strftime") else str(t.entry_time)[:16],
                        "Entry (₹)": round(t.entry_price, 2),
                        "Exit Date": pd.to_datetime(t.exit_time).strftime("%d %b %Y %H:%M") if (t.exit_time and hasattr(t.exit_time, "strftime")) else str(t.exit_time)[:16],
                        "Exit (₹)": round(t.exit_price, 2) if t.exit_price else np.nan,
                        "Qty": t.quantity,
                        "Net P&L (₹)": round(t.pnl_rupees, 2),
                        "Return (%)": round(t.pnl_percent, 2),
                        "Risk:Reward (R)": rr_display,
                        "Planned R:R": planned_rr_str,
                        "Exit Reason": t.exit_reason,
                        "Bars Held": t.duration_bars,
                        "MFE (%)": round(t.max_favorable_excursion, 2),
                        "MAE (%)": round(t.max_adverse_excursion, 2),
                    })
                trades_df = pd.DataFrame(trade_rows)
                
                # Format dataframe styling
                def _color_rr_cell(val):
                    if isinstance(val, str):
                        if val.startswith("1 :") or "(+" in val or "+ " in val:
                            return "color: #10B981; font-weight: 600;"
                        elif val.startswith("-1 :") or "(-" in val or "- " in val:
                            return "color: #EF4444; font-weight: 600;"
                    return ""

                st.dataframe(
                    trades_df.style.format({
                        "Entry (₹)": "₹{:,.2f}",
                        "Exit (₹)": "₹{:,.2f}",
                        "Net P&L (₹)": "₹{:,.2f}",
                        "Return (%)": "{:+.2f}%",
                        "MFE (%)": "{:+.2f}%",
                        "MAE (%)": "{:+.2f}%",
                    }).map(lambda v: "color: #10B981; font-weight: 600;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: 600;" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Net P&L (₹)", "Return (%)"]
                    ).map(_color_rr_cell, subset=["Risk:Reward (R)"]),
                    use_container_width=True,
                    height=360
                )

                # Download CSV
                csv_bytes = trades_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Trade Log ({sel_sym} - {cfg.name}).csv",
                    data=csv_bytes,
                    file_name=f"trade_log_{sel_sym}_{cfg.strategy_type}.csv",
                    mime="text/csv",
                    key="slab_download_trades_csv"
                )
            else:
                st.info("No trades were generated under these parameter constraints. Try widening targets/stop-losses or testing another timeframe.")

    # ==========================================
    # TAB 2: MULTI-STOCK BASKET BACKTEST
    # ==========================================
    with tab_basket:
        st.markdown("#### 🧺 Portfolio Basket Simulation")
        st.caption("Test how this strategy performs across an entire basket of equities with equal capital allocation.")

        b_col1, b_col2, b_col3, b_col4 = st.columns([1.6, 1.2, 1.2, 1.2])
        with b_col1:
            basket_universe_choice = st.selectbox(
                "Select Stock Basket:",
                ["Nifty 50 Equities", "Top 20 Liquid Leaders", "Custom Selection"],
                index=0,
                key="slab_basket_choice"
            )
        with b_col2:
            basket_tf = st.selectbox("Timeframe:", ["Daily", "75-Min", "15-Min", "5-Min", "Weekly", "Custom (Minutes)"], index=0, key="slab_basket_tf")
            if basket_tf == "Custom (Minutes)":
                basket_custom_min = st.number_input("Minutes:", min_value=1, max_value=375, value=15, step=1, key="slab_basket_custom_min")
                effective_basket_tf = f"{int(basket_custom_min)}-Min"
            else:
                effective_basket_tf = basket_tf
        with b_col3:
            cap_per_stock = st.number_input("Capital per Stock (₹):", min_value=10000.0, value=50000.0, step=10000.0)
        with b_col4:
            st.write("")
            st.write("")
            run_basket_btn = st.button("🚀 Run Basket Backtest", type="primary", use_container_width=True, key="slab_run_basket_btn")

        if basket_universe_choice == "Custom Selection":
            custom_basket_syms = st.multiselect("Select Equities:", options=all_symbols, default=popular_indices[:8])
        elif basket_universe_choice == "Top 20 Liquid Leaders":
            custom_basket_syms = popular_indices + ["BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH", "POWERGRID"]
        else:
            custom_basket_syms = config.NIFTY_50_SYMBOLS

        if run_basket_btn:
            p_bar = st.progress(0)
            p_msg = st.empty()

            basket_data = {}
            total_n = len(custom_basket_syms)

            for i, sym in enumerate(custom_basket_syms):
                p_bar.progress(min((i + 1) / max(total_n, 1), 1.0))
                p_msg.caption(f"Loading & backtesting {sym} ({i+1}/{total_n})...")
                df_b = load_candles_for_simulation(sym, effective_basket_tf)
                if not df_b.empty and len(df_b) >= 20:
                    basket_data[sym] = df_b

            p_msg.caption("Aggregating portfolio performance & generating leaderboard...")
            basket_res = run_basket_backtest(
                basket_data,
                cfg,
                timeframe=effective_basket_tf,
                capital_per_stock=cap_per_stock,
                slippage_pct=slippage_pct
            )
            p_bar.empty()
            p_msg.empty()
            st.session_state["slab_basket_result"] = basket_res

        if "slab_basket_result" in st.session_state:
            b_res = st.session_state["slab_basket_result"]

            # Aggregated Basket Metrics
            bk1, bk2, bk3, bk4, bk5 = st.columns(5)
            pnl_c = "green" if b_res["total_pnl"] >= 0 else "red"
            with bk1:
                render_metric_card("Portfolio Capital", f"₹{b_res['total_capital']:,.0f}", f"{b_res['total_symbols']} Active Equities", "normal", tc)
            with bk2:
                render_metric_card("Portfolio Net P&L", f"{b_res['total_pnl_pct']:+.2f}%", f"₹{b_res['total_pnl']:+,.2f}", pnl_c, tc)
            with bk3:
                render_metric_card("Overall Win Rate", f"{b_res['win_rate']:.1f}%", f"{b_res['total_trades']} Total Closed Trades", "blue", tc)
            with bk4:
                render_metric_card("Final Portfolio Value", f"₹{b_res['final_equity']:,.0f}", f"Initial: ₹{b_res['total_capital']:,.0f}", pnl_c, tc)
            with bk5:
                render_metric_card("Strategy Edge", f"{cfg.name[:18]}", f"Timeframe: {effective_basket_tf}", "normal", tc)

            # Leaderboard Table
            st.markdown("##### 🏆 Basket Equities Leaderboard")
            lead_df = b_res["leaderboard"]
            if not lead_df.empty:
                st.dataframe(
                    lead_df.style.format({
                        "Net P&L (₹)": "₹{:,.2f}",
                        "Return (%)": "{:+.2f}%",
                        "Win Rate (%)": "{:.1f}%",
                        "Profit Factor": "{:.2f}",
                        "Max DD (%)": "{:.2f}%"
                    }).map(lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Net P&L (₹)", "Return (%)"]),
                    use_container_width=True,
                    height=450
                )

                # Download Leaderboard CSV
                l_csv = lead_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Basket Leaderboard CSV",
                    data=l_csv,
                    file_name=f"basket_leaderboard_{cfg.strategy_type}.csv",
                    mime="text/csv",
                    key="slab_download_basket_csv"
                )

    # ==========================================
    # TAB 3: STRATEGY COMPARISON
    # ==========================================
    with tab_compare:
        st.markdown("#### ⚖️ Head-to-Head Strategy Comparison")
        st.caption("Compare two trading setups directly on the same stock and timeframe to identify the superior edge.")

        cmp_col1, cmp_col2, cmp_col3, cmp_col4, cmp_col5 = st.columns([1.4, 1.4, 1.1, 1.1, 1.0])
        with cmp_col1:
            strat_a_name = st.selectbox("Strategy A:", options=strategy_options, index=0, key="slab_cmp_strat_a")
        with cmp_col2:
            strat_b_name = st.selectbox("Strategy B:", options=strategy_options, index=1, key="slab_cmp_strat_b")
        with cmp_col3:
            cmp_sym = st.selectbox("Benchmark Stock:", options=all_symbols, index=default_sym_idx, key="slab_cmp_sym")
        with cmp_col4:
            cmp_tf = st.selectbox("Timeframe:", ["Daily", "75-Min", "15-Min", "5-Min", "Weekly", "Custom (Minutes)"], index=0, key="slab_cmp_tf")
            if cmp_tf == "Custom (Minutes)":
                cmp_custom_min = st.number_input("Minutes:", min_value=1, max_value=375, value=15, step=1, key="slab_cmp_custom_min")
                effective_cmp_tf = f"{int(cmp_custom_min)}-Min"
            else:
                effective_cmp_tf = cmp_tf
        with cmp_col5:
            st.write("")
            st.write("")
            cmp_btn = st.button("⚖️ Compare", type="primary", use_container_width=True, key="slab_run_cmp_btn")

        if cmp_btn:
            with st.spinner(f"Comparing {strat_a_name} vs {strat_b_name} on {cmp_sym} ({effective_cmp_tf})..."):
                df_cmp_raw = load_candles_for_simulation(cmp_sym, effective_cmp_tf)
                if not df_cmp_raw.empty and len(df_cmp_raw) >= 30:
                    cfg_a = get_preset_strategy(strat_a_name)
                    cfg_b = get_preset_strategy(strat_b_name)

                    df_a = prepare_indicators(df_cmp_raw, cfg_a)
                    res_a = run_backtest(df_a, cfg_a, symbol=cmp_sym, timeframe=effective_cmp_tf, initial_capital=initial_cap, slippage_pct=slippage_pct)

                    df_b = prepare_indicators(df_cmp_raw, cfg_b)
                    res_b = run_backtest(df_b, cfg_b, symbol=cmp_sym, timeframe=effective_cmp_tf, initial_capital=initial_cap, slippage_pct=slippage_pct)

                    st.session_state["slab_cmp_res"] = (res_a, res_b, strat_a_name, strat_b_name)

        if "slab_cmp_res" in st.session_state:
            res_a, res_b, name_a, name_b = st.session_state["slab_cmp_res"]

            # Head-to-head comparison metrics table
            cmp_metrics_data = [
                {"Metric": "Strategy Name", "Strategy A": name_a, "Strategy B": name_b},
                {"Metric": "Net Return (%)", "Strategy A": f"{res_a.total_net_pnl_pct:+.2f}%", "Strategy B": f"{res_b.total_net_pnl_pct:+.2f}%"},
                {"Metric": "Net P&L (₹)", "Strategy A": f"₹{res_a.total_net_pnl:+,.2f}", "Strategy B": f"₹{res_b.total_net_pnl:+,.2f}"},
                {"Metric": "Planned Risk:Reward", "Strategy A": f"1 : {res_a.planned_risk_reward:.2f}" if res_a.planned_risk_reward > 0 else "Discretionary", "Strategy B": f"1 : {res_b.planned_risk_reward:.2f}" if res_b.planned_risk_reward > 0 else "Discretionary"},
                {"Metric": "Avg Realized R:R (Payoff)", "Strategy A": f"1 : {abs(res_a.avg_win / res_a.avg_loss):.2f} ({res_a.avg_risk_reward:+.2f}R)" if (res_a.avg_loss and abs(res_a.avg_loss) > 0) else "-", "Strategy B": f"1 : {abs(res_b.avg_win / res_b.avg_loss):.2f} ({res_b.avg_risk_reward:+.2f}R)" if (res_b.avg_loss and abs(res_b.avg_loss) > 0) else "-"},
                {"Metric": "Win Rate (%)", "Strategy A": f"{res_a.win_rate:.1f}% ({res_a.winning_trades}/{res_a.total_trades})", "Strategy B": f"{res_b.win_rate:.1f}% ({res_b.winning_trades}/{res_b.total_trades})"},
                {"Metric": "Profit Factor", "Strategy A": f"{res_a.profit_factor:.2f}", "Strategy B": f"{res_b.profit_factor:.2f}"},
                {"Metric": "Max Drawdown (%)", "Strategy A": f"-{res_a.max_drawdown_pct:.2f}%", "Strategy B": f"-{res_b.max_drawdown_pct:.2f}%"},
                {"Metric": "Total Trades", "Strategy A": f"{res_a.total_trades}", "Strategy B": f"{res_b.total_trades}"},
                {"Metric": "Trade Expectancy", "Strategy A": f"₹{res_a.expectancy:+,.2f}", "Strategy B": f"₹{res_b.expectancy:+,.2f}"},
            ]
            st.dataframe(pd.DataFrame(cmp_metrics_data), use_container_width=True, hide_index=True)

            # Dual Equity Curve Chart
            fig_cmp = go.Figure()
            if not res_a.equity_curve.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=res_a.equity_curve.index,
                    y=res_a.equity_curve["equity"],
                    mode="lines",
                    name=f"Strategy A: {name_a[:25]}",
                    line=dict(color=tc["blue"], width=2.5)
                ))
            if not res_b.equity_curve.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=res_b.equity_curve.index,
                    y=res_b.equity_curve["equity"],
                    mode="lines",
                    name=f"Strategy B: {name_b[:25]}",
                    line=dict(color=tc["green"], width=2.5)
                ))

            fig_cmp.update_layout(
                title=f"Dual Equity Curve Comparison on {res_a.symbol}",
                template=tc["template"],
                paper_bgcolor=tc["paper_bg"],
                plot_bgcolor=tc["plot_bg"],
                height=450,
                margin=dict(l=40, r=40, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_cmp.update_xaxes(gridcolor=tc["grid_clr"])
            fig_cmp.update_yaxes(gridcolor=tc["grid_clr"])
            st.plotly_chart(fig_cmp, use_container_width=True)
