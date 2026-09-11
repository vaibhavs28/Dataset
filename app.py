import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

import config
import database
import scanner
import downloader
import instruments
import auth
import parquet_loader
import tradingview_charts
import upstox_parquet_updater
import sector_data
import importlib
import re

importlib.reload(tradingview_charts)
importlib.reload(scanner)
importlib.reload(database)
importlib.reload(downloader)
importlib.reload(instruments)
importlib.reload(upstox_parquet_updater)
importlib.reload(parquet_loader)
importlib.reload(sector_data)


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Multi-Timeframe EMA Alignment Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .rule-card {
        background-color: #151924;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 13px;
    }
    .badge-stage3 {
        background-color: #10B981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-stage2 {
        background-color: #3B82F6;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-stage1 {
        background-color: #F59E0B;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    title: str,
    ema_dict: dict,
    show_volume: bool = True,
    height: int = 440,
    is_intraday: bool = False,
    pivot_dict: dict = None,
    show_rsi: bool = True,
    rsi_span: int = 9,
    highlight_rsi_50: bool = True,
    theme: str = "light"
):
    """
    Renders a Plotly candlestick chart with EMA overlays, optional Weekly Pivots, and RSI indicator.
    Supports both Dark and Light themes.
    """
    is_light = (str(theme).lower() == "light")
    chart_template = "plotly_white" if is_light else "plotly_dark"
    paper_bg = "#FFFFFF" if is_light else "#131722"
    plot_bg = "#FFFFFF" if is_light else "#131722"
    grid_clr = "#E0E3EB" if is_light else "#2A2E39"

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(height=height, template=chart_template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig

    # Format x-axis labels to remove weekend / overnight gap voids
    if hasattr(df.index, "strftime"):
        if is_intraday:
            x_labels = [dt.strftime("%d %b %H:%M") for dt in df.index]
        else:
            x_labels = [dt.strftime("%d %b '%y") for dt in df.index]
    else:
        x_labels = [str(x) for x in df.index]

    has_rsi = show_rsi and ("RSI" in df.columns)

    if show_volume and has_rsi:
        rows = 3
        row_heights = [0.60, 0.18, 0.22]
        subplot_titles = (f"{symbol.upper()} - {title}", "Volume", "Hilega Milega(by NK sir) [anuragM]")
        vol_row = 2
        rsi_row = 3
    elif show_volume and not has_rsi:
        rows = 2
        row_heights = [0.75, 0.25]
        subplot_titles = (f"{symbol.upper()} - {title}", "Volume")
        vol_row = 2
        rsi_row = None
    elif not show_volume and has_rsi:
        rows = 2
        row_heights = [0.75, 0.25]
        subplot_titles = (f"{symbol.upper()} - {title}", "Hilega Milega(by NK sir) [anuragM]")
        vol_row = None
        rsi_row = 2
    else:
        rows = 1
        row_heights = [1.0]
        subplot_titles = (f"{symbol.upper()} - {title}",)
        vol_row = None
        rsi_row = None

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_labels,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#089981",
            decreasing_line_color="#F23645"
        ),
        row=1, col=1
    )

    # Line Chart (Close price)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df["close"],
            name="Line (Close)",
            mode="lines",
            line=dict(color="#2962FF", width=2)
        ),
        row=1, col=1
    )

    # Overlay EMAs & Anchored VWAPs
    color_palette = {
        "EMA_5": "#4CAF50",    # Green
        "EMA_9": "#4CAF50",    # Green (Fast trigger)
        "EMA_13": "#38BDF8",   # Sky Blue
        "EMA_20": "#2962FF",   # Blue
        "Daily_EMA_20": "#2962FF",  # Blue (Higher-timeframe Daily 20 EMA)
        "EMA_26": "#9C27B0",   # Purple
        "EMA_50": "#FF5252",   # Red
        "EMA_200": "#131722" if is_light else "#FFFFFF",  # Black in light theme, White in dark theme
        "AVWAP_MAR2020": "#38BDF8",  # Sky Blue Solid
        "AVWAP_JUN2022": "#38BDF8"   # Sky Blue Dotted
    }

    for col_name, cfg_or_color in ema_dict.items():
        if col_name in df.columns:
            if isinstance(cfg_or_color, dict):
                c_clr = cfg_or_color.get("color", color_palette.get(col_name, "#38BDF8"))
                c_name = cfg_or_color.get("name", col_name)
                c_dash = cfg_or_color.get("dash", "dot" if col_name == "AVWAP_JUN2022" else "solid")
                c_width = cfg_or_color.get("width", 1.8)
            else:
                c_clr = cfg_or_color if cfg_or_color else color_palette.get(col_name, "#FFFFFF")
                c_name = "AVWAP Mar 2020" if col_name == "AVWAP_MAR2020" else ("AVWAP Jun 2022" if col_name == "AVWAP_JUN2022" else col_name)
                c_dash = "dot" if col_name == "AVWAP_JUN2022" else "solid"
                c_width = 1.8

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df[col_name],
                    mode="lines",
                    name=c_name,
                    line=dict(color=c_clr, width=c_width, dash=c_dash)
                ),
                row=1, col=1
            )

    # Overlay Pivots / CPR Levels (Weekly CPR, R1, S1)
    if pivot_dict:
        for p_col, p_cfg in pivot_dict.items():
            if p_col in df.columns and not df[p_col].dropna().empty:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=df[p_col],
                        mode="lines",
                        name=p_cfg.get("name", p_col),
                        line=dict(
                            color=p_cfg.get("color", "#FFFFFF"),
                            width=p_cfg.get("width", 1.8),
                            dash=p_cfg.get("dash", "dash")
                        ),
                        connectgaps=False
                    ),
                    row=1, col=1
                )

    # Volume
    if vol_row and "volume" in df.columns:
        vol_colors = ["#089981" if c >= o else "#F23645" for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=df["volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.6
            ),
            row=vol_row, col=1
        )

    # Hilega Milega Subplot Panel
    if rsi_row and "RSI" in df.columns:
        has_e3 = "RSI_EMA3" in df.columns and not df["RSI_EMA3"].dropna().empty
        has_w21 = "RSI_WMA21" in df.columns and not df["RSI_WMA21"].dropna().empty

        # Cloud above 50 (Pink) and below 50 (Soft Blue) filled till RSI line
        if "RSI" in df.columns and not df["RSI"].dropna().empty:
            rsi_vals = df["RSI"].to_numpy(dtype=float)
            above_50 = np.where(np.isnan(rsi_vals), np.nan, np.maximum(rsi_vals, 50.0))
            below_50 = np.where(np.isnan(rsi_vals), np.nan, np.minimum(rsi_vals, 50.0))
            fifty_line = [50.0] * len(df)

            # 1. Pink cloud above 50
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=fifty_line,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=rsi_row, col=1
            )
            pink_col = "rgba(255, 237, 237, 0.75)" if is_light else "rgba(248, 113, 113, 0.25)"
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=above_50,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=pink_col,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=rsi_row, col=1
            )

            # 2. Soft blue cloud below 50
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=fifty_line,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=rsi_row, col=1
            )
            blue_col = "rgba(233, 239, 255, 0.75)" if is_light else "rgba(96, 165, 250, 0.25)"
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=below_50,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=blue_col,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=rsi_row, col=1
            )

        # 2. Primary RSI(9) Line: Black in Light, White in Dark
        rsi_clr = "#131722" if is_light else "#F8FAFC"
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df["RSI"],
                mode="lines",
                name=f"RSI ({rsi_span}) [Black]",
                line=dict(color=rsi_clr, width=1.8),
                hovertemplate="<b>Hilega Milega RSI (9)</b>: %{y:.2f}<extra></extra>"
            ),
            row=rsi_row, col=1
        )

        # 3. Fast EMA 3 on RSI: Exact TradingView Green (#4CAF50)
        if has_e3:
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["RSI_EMA3"],
                    mode="lines",
                    name="EMA 3 (Green)",
                    line=dict(color="#4CAF50", width=2.0),
                    hovertemplate="<b>EMA 3 (Green)</b>: %{y:.2f}<extra></extra>"
                ),
                row=rsi_row, col=1
            )

        # 4. Slow WMA 21 on RSI: Exact TradingView Red (#FF5252)
        if has_w21:
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["RSI_WMA21"],
                    mode="lines",
                    name="WMA 21 (Red)",
                    line=dict(color="#FF5252", width=2.0),
                    hovertemplate="<b>WMA 21 (Red)</b>: %{y:.2f}<extra></extra>"
                ),
                row=rsi_row, col=1
            )

        # 5. Exact TradingView Blue 50 Line (#7695F9) with Badge
        fig.add_hline(
            y=50,
            line=dict(color="#7695F9", width=2, dash="solid"),
            annotation_text="50",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#7695F9", family="monospace"),
            annotation_bgcolor="rgba(255, 255, 255, 0.95)" if is_light else "rgba(30, 58, 138, 0.95)",
            annotation_bordercolor="#7695F9",
            annotation_borderwidth=1.5,
            row=rsi_row, col=1
        )

        # 70 and 30 dashed reference lines
        fig.add_hline(
            y=70, line=dict(color="#94A3B8", width=1, dash="dash"),
            row=rsi_row, col=1
        )
        fig.add_hline(
            y=30, line=dict(color="#94A3B8", width=1, dash="dash"),
            row=rsi_row, col=1
        )

        fig.update_yaxes(
            title_text="Hilega Milega",
            range=[0, 100],
            tickvals=[30, 50, 70],
            gridcolor=grid_clr,
            row=rsi_row, col=1
        )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template=chart_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#131722" if is_light else "#D1D4DC")
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor=grid_clr)
    if vol_row:
        fig.update_yaxes(title_text="Vol", row=vol_row, col=1, gridcolor=grid_clr)

    # Category x-axis removes all empty overnight & weekend spaces!
    fig.update_xaxes(type="category", gridcolor=grid_clr, nticks=8)

    return fig


@st.cache_data(ttl=600, show_spinner=False)
def get_master_alignment_scan() -> pd.DataFrame:
    """Caches master scan results across the universe for instantaneous sub-millisecond filtering."""
    return scanner.run_multiframe_alignment_scan()


def get_cached_alignment_scan(stage_filter: str) -> pd.DataFrame:
    """Instantaneous in-memory filtering from the cached master scan."""
    df = get_master_alignment_scan()
    if df.empty or stage_filter == "All":
        return df
    if stage_filter == "Stage 3 (Full Alignment Only)":
        return df[df["Score"] == 3].reset_index(drop=True)
    elif stage_filter == "Stage 2+ (M+W Aligned)":
        return df[df["Score"] >= 2].reset_index(drop=True)
    elif stage_filter == "Stage 1+ (Monthly Pass)":
        return df[df["Score"] >= 1].reset_index(drop=True)
    elif stage_filter in ("Stage 3 + Hilega Milega Bullish (RSI>=50 & EMA3>=WMA21)", "Stage 3 + Monthly RSI (RSI>=50 & EMA3>=WMA21)"):
        return df[(df["Score"] == 3) & (df["Monthly_RSI_Match"] == "✅ PASS")].reset_index(drop=True)
    elif stage_filter in ("Hilega Milega Bullish Only (RSI>=50 & EMA3>=WMA21)", "Monthly RSI Scan Only (RSI>=50 & EMA3>=WMA21)"):
        return df[df["Monthly_RSI_Match"] == "✅ PASS"].reset_index(drop=True)
    return df


def main():
    try:
        database.init_db()
    except Exception:
        pass

    try:
        db_stats = database.get_db_stats()
    except Exception:
        db_stats = {"total_instruments": 3498, "symbols_with_candles": 3450, "total_candles": 4200000, "earliest_date": "2020-01-01", "latest_date": "2026-09-10"}

    try:
        from zoneinfo import ZoneInfo
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

    # Indian equity markets trade Monday-Friday (0-4) strictly between 09:15 and 15:30 IST
    is_live_hours = (now_ist.weekday() < 5 and (9 * 60 + 15 <= now_ist.hour * 60 + now_ist.minute <= 15 * 60 + 30))

    # App Title & Header
    col_title, col_stat1, col_stat2, col_stat3, col_stat4 = st.columns([2.6, 1, 1, 1, 1])
    with col_title:
        st.title("🎯 Multi-Timeframe EMA Alignment Scanner")
        st.caption("Monthly (5>20) ➔ Weekly (20>50>200) ➔ Daily (20>50>200) ➔ 75-Min Execution")

    with col_stat1:
        st.metric("Symbols in DB", db_stats["symbols_with_candles"])
    with col_stat2:
        st.metric("Total Daily Candles", f"{db_stats['total_candles']:,}")
    with col_stat3:
        st.metric("Latest Date", db_stats["latest_date"] or "No Data")
    with col_stat4:
        if is_live_hours:
            st.metric("Market Status", "🟢 LIVE (IST)", help="Indian equity markets open (09:15 - 15:30). Live 1-min Upstox ticks update on demand.")
        else:
            st.metric("Market Status", "⚪ Closed", help="Indian equity markets closed.")

    st.divider()

    # Live Visual Upstox Sync Engine Progress Tracker
    engine_st = parquet_loader.get_live_engine_status()
    if engine_st.get("is_active"):
        with st.container():
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%); 
                        border: 1.5px solid #0284C7; border-radius: 12px; padding: 14px 20px; margin-bottom: 14px;
                        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: #10B981; box-shadow: 0 0 10px #10B981;"></span>
                        <b style="color: #38BDF8; font-size: 15px; letter-spacing: 0.5px;">LIVE UPSTOX DATA SYNC ACTIVE</b>
                    </div>
                    <div style="font-size: 12.5px; color: #94A3B8;">
                        Batch <b style="color: #38BDF8;">{engine_st['current_batch']}</b> &nbsp;|&nbsp; 
                        Active Shard: <code style="color: #F8FAFC; background: #334155; padding: 2px 6px; border-radius: 4px;">{engine_st['active_shard']}</code> ({engine_st['active_shard_mb']} MB)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            prog_msg = f"⚡ Synced {engine_st['curr_idx']:,} / {engine_st['total_syms']:,} Equities ({engine_st['pct']}%) | Latest: {engine_st['latest_symbol']} (+{engine_st['latest_min_bars']} 1-min bars in {engine_st['latest_time']}s)"
            st.progress(min(1.0, max(0.0, engine_st["pct"] / 100.0)), text=prog_msg)

            pk1, pk2, pk3, pk4, pk5 = st.columns([1.5, 1.2, 1.5, 1.5, 1.0])
            pk1.metric("Equities Completed", f"{engine_st['curr_idx']} / {engine_st['total_syms']} ({engine_st['pct']}%)")
            pk2.metric("Processing Speed", f"{engine_st['avg_speed_s']}s / stock")
            pk3.metric("Parquet Dataset Size", f"{engine_st['total_size_mb']:,.1f} MB ({engine_st['num_shards']} shards)")
            pk4.metric("Est. Time Remaining", f"~{engine_st['eta_hrs']} Hours")
            with pk5:
                st.write("")
                if st.button("🔄 Refresh", key="sync_live_ref_btn", use_container_width=True, type="secondary"):
                    st.rerun()

            with st.expander("📊 Live Stream: Recently Downloaded Stocks & Real-Time Metrics"):
                if engine_st.get("recent"):
                    rec_df = pd.DataFrame(engine_st["recent"])[["index", "symbol", "daily_bars", "min_bars", "seconds"]]
                    rec_df.columns = ["Stock #", "Symbol", "Daily Bars", "1-Min Bars", "Duration (s)"]
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
            st.divider()

    # Sidebar: Theme & Page Navigation
    st.sidebar.markdown("### 🎨 Display Theme")
    theme_choice = st.sidebar.radio(
        "Display Theme",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.get("app_theme", "light") == "light" else 1,
        horizontal=True,
        key="app_theme_radio",
        label_visibility="collapsed"
    )
    theme = "light" if "Light" in theme_choice else "dark"
    st.session_state["app_theme"] = theme

    st.sidebar.markdown("### 🧭 Pages")
    selected_page = st.sidebar.radio(
        "Select Page",
        [
            "📊 Quad-Chart View",
            "🚀 Alignment Scanner",
            "📈 Trading Terminal",
            "🗺️ Market Heatmap"
        ],
        index=0,
        key="app_active_nav_page",
        label_visibility="collapsed"
    )


    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Daily Market Update")
    today_display = datetime.today().strftime("%d %b %Y")
    st.sidebar.caption(f"📅 Today: **{today_display}**")
    update_scope = st.sidebar.radio(
        "Update Scope",
        options=["Nifty 50 & Top Stocks (⚡ ~5s)", "All Database Stocks (3,300+)"],
        index=0,
        key="one_click_scope_choice",
        label_visibility="collapsed"
    )
    if st.sidebar.button("⚡ One-Click Update to Today", use_container_width=True, type="primary", key="one_click_daily_update_btn"):
        p_bar = st.sidebar.progress(0)
        p_txt = st.sidebar.empty()
        def _daily_cb(cur, tot, sym):
            p_bar.progress(min(cur / max(tot, 1), 1.0))
            p_txt.caption(f"Updating {sym} ({cur}/{tot})...")
        target_syms = config.NIFTY_50_SYMBOLS if "5s" in update_scope else None
        with st.spinner("Syncing latest market data across stocks via fast DuckDB batch API..."):
            try:
                import batch_downloader
                res = batch_downloader.sync_live_market_batch(symbols=target_syms, progress_callback=_daily_cb)
                st.sidebar.success(f"✅ Fast Batch Sync: Updated {res['synced']:,}/{res['total']:,} stocks in {res['duration']:.1f}s directly into DuckDB!")
            except Exception as e:
                res = downloader.update_daily_market_data_one_click(symbols=target_syms, progress_callback=_daily_cb)
                st.sidebar.success(f"✅ Updated {res['updated_symbols']} stocks ({res['new_bars_added']} new candles)! {res['already_up_to_date']} were already up to date.")
        p_bar.empty()
        p_txt.empty()
        st.rerun()

    if engine_st.get("is_active"):
        st.sidebar.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 8px; padding: 10px; margin-top: 10px;">
            <div style="display: flex; align-items: center; gap: 6px; font-weight: 600; color: #10B981; font-size: 13px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
                Sync Active ({engine_st['pct']}%)
            </div>
            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">
                {engine_st['curr_idx']}/{engine_st['total_syms']} stocks &bull; {engine_st['latest_symbol']}<br/>
                Shard: {engine_st['active_shard']} ({engine_st['active_shard_mb']} MB)<br/>
                ETA: ~{engine_st['eta_hrs']}h remaining
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    if theme == "light":
        st.markdown("""
        <style>
            /* Global Background & Base Text */
            .stApp, .stApp [data-testid="stAppViewContainer"] {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }
            /* Sidebar Styling in Light Mode */
            [data-testid="stSidebar"], section[data-testid="stSidebar"] {
                background-color: #FFFFFF !important;
                border-right: 1px solid #E2E8F0 !important;
            }
            [data-testid="stSidebar"] p, 
            [data-testid="stSidebar"] span, 
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div {
                color: #1E293B !important;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4 {
                color: #0F172A !important;
                font-weight: 700 !important;
            }
            /* Header Metric Cards in Light Mode */
            div[data-testid="stMetric"] {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            }
            div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
                color: #475569 !important;
                font-size: 13px !important;
                font-weight: 600 !important;
            }
            div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
                color: #0F172A !important;
                font-size: 22px !important;
                font-weight: 700 !important;
            }
            /* Quick Symbol Chips (Buttons) in Light Mode */
            div[data-testid="stHorizontalBlock"] button, .stButton > button {
                background-color: #F1F5F9 !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
            }
            div[data-testid="stHorizontalBlock"] button:hover, .stButton > button:hover {
                background-color: #E2E8F0 !important;
                border-color: #94A3B8 !important;
                color: #0284C7 !important;
            }
            /* Primary Action Buttons */
            .stButton > button[kind="primary"] {
                background-color: #0284C7 !important;
                color: #FFFFFF !important;
                border: none !important;
            }
            /* Form Labels & Dropdown Headers in Light Mode */
            label, .stSelectbox label, .stRadio label, .stCheckbox label, .stNumberInput label {
                color: #1E293B !important;
                font-weight: 600 !important;
            }
            /* Selectbox Input Controls */
            div[data-baseweb="select"] > div {
                background-color: #FFFFFF !important;
                border-color: #CBD5E1 !important;
                color: #0F172A !important;
            }
            div[data-baseweb="select"] * {
                color: #0F172A !important;
            }
            /* Markdown headings & body */
            h1, h2, h3, h4, h5, h6 {
                color: #0F172A !important;
            }
            p, span {
                color: #1E293B;
            }
            .rule-card {
                background-color: #EFF6FF !important;
                border-left: 4px solid #2563EB !important;
                color: #1E3A8A !important;
            }
            .stTabs [data-baseweb="tab"] {
                color: #475569 !important;
            }
            .stTabs [aria-selected="true"] {
                color: #0284C7 !important;
                font-weight: 700 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp, .stApp [data-testid="stAppViewContainer"] {
                background-color: #0F172A !important;
                color: #F8FAFC !important;
            }
            [data-testid="stSidebar"], section[data-testid="stSidebar"] {
                background-color: #1E293B !important;
                border-right: 1px solid #334155 !important;
            }
            [data-testid="stSidebar"] p, 
            [data-testid="stSidebar"] span, 
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div {
                color: #F1F5F9 !important;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4 {
                color: #F8FAFC !important;
                font-weight: 700 !important;
            }
            div[data-testid="stMetric"] {
                background-color: #1E293B !important;
                border: 1px solid #334155 !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
            }
            div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
                color: #94A3B8 !important;
                font-size: 13px !important;
            }
            div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
                color: #F8FAFC !important;
                font-size: 22px !important;
                font-weight: 700 !important;
            }
            div[data-testid="stHorizontalBlock"] button, .stButton > button {
                background-color: #1E293B !important;
                color: #F1F5F9 !important;
                border: 1px solid #334155 !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
            }
            div[data-testid="stHorizontalBlock"] button:hover, .stButton > button:hover {
                background-color: #334155 !important;
                border-color: #0284C7 !important;
                color: #38BDF8 !important;
            }
            .stButton > button[kind="primary"] {
                background-color: #0284C7 !important;
                color: #FFFFFF !important;
                border: none !important;
            }
            label, .stSelectbox label, .stRadio label, .stCheckbox label, .stNumberInput label {
                color: #E2E8F0 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar.expander("📦 Load from Local Parquet Files (data/1min)", expanded=False):

        p_syms = parquet_loader.get_parquet_symbols()
        st.write(f"📁 **8 Parquet Files Detected** (~10.4 GB)")
        st.caption(f"Total Symbols Available: **{len(p_syms):,} stocks & indices**")
        
        load_mode = st.selectbox("Batch Mode", ["Nifty 50 Constituents", "Specific Symbol"])
        if load_mode == "Specific Symbol":
            target_p = st.selectbox("Select Stock to Import", options=p_syms, index=p_syms.index("RELIANCE") if "RELIANCE" in p_syms else 0)
        
        if st.button("🚀 Load into Scanner Database", use_container_width=True):
            if load_mode == "Specific Symbol":
                to_load = [target_p]
            else:
                to_load = [s for s in config.NIFTY_50_SYMBOLS if s in p_syms]
            
            p_bar = st.progress(0)
            p_txt = st.empty()
            def cb(c, t, s):
                p_bar.progress(c / t)
                p_txt.text(f"Resampling & importing {s} ({c}/{t})...")
            
            cnt = parquet_loader.import_batch_to_database(to_load, progress_callback=cb)
            st.success(f"Successfully loaded {cnt} stocks from Parquet into SQLite!")
            st.rerun()

    with st.sidebar.expander("🔑 Upstox API Credentials", expanded=False):
        api_key = st.text_input("API Key", value=config.UPSTOX_API_KEY, type="password")
        api_secret = st.text_input("API Secret", value=config.UPSTOX_API_SECRET, type="password")
        redirect_uri = st.text_input("Redirect URI", value=config.UPSTOX_REDIRECT_URI)

        if st.button("Save Credentials", use_container_width=True):
            auth.save_credentials(api_key, api_secret, redirect_uri)
            st.success("Credentials saved!")

        if config.UPSTOX_API_KEY:
            login_url = auth.get_login_url()
            st.markdown(f"[👉 Login with Upstox]({login_url})", unsafe_allow_html=True)
            auth_code = st.text_input("Paste Redirect Code")
            if st.button("Generate Token", use_container_width=True):
                try:
                    tok = auth.exchange_code_for_token(auth_code)
                    st.success("Token saved!")
                except Exception as e:
                    st.error(str(e))

        direct_tok = st.text_input("Or Paste Access Token", value=config.UPSTOX_ACCESS_TOKEN, type="password")
        if st.button("Save Token", use_container_width=True):
            auth.save_access_token(direct_tok)
            st.success("Token saved!")

    with st.sidebar.expander("📥 Download Upstox Market Data", expanded=False):
        sync_target = st.selectbox("Watchlist", ["Nifty 50 (50 stocks)", "All DB Instruments", "Custom Symbol"])
        custom_sym = st.text_input("Custom Symbol (if chosen)").upper()

        if st.button("⚡ Start Upstox Download", use_container_width=True):
            syms_to_sync = [custom_sym] if (sync_target == "Custom Symbol" and custom_sym) else config.NIFTY_50_SYMBOLS
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(curr, total, symbol):
                progress_bar.progress(curr / total)
                status_text.text(f"Fetching {symbol} ({curr}/{total})...")

            result = downloader.sync_watchlist(syms_to_sync, progress_callback=update_progress)
            st.success(f"Downloaded {result['synced']} symbols successfully!")
            st.rerun()

    with st.sidebar.expander("🌐 Full Database Mass Sync (2022 - Today)", expanded=False):
        import mass_equity_sync
        cp = mass_equity_sync.load_checkpoint()
        comp_cnt = len(cp.get("completed_symbols", {}))
        tot_bars = cp.get("total_bars", 0)
        st.write(f"📁 **{comp_cnt:,} Equities Synced** ({tot_bars:,} bars in DB)")
        st.caption("Syncs all equities from 1 Jan 2022 to today in 50-stock batches with persistent checkpoints.")
        if st.button("🚀 Run Mass Equity Sync (50/Batch)", use_container_width=True):
            with st.spinner("Syncing equities from 1 Jan 2022 to today..."):
                mass_equity_sync.run_mass_sync(batch_size=50, pause_seconds=2)
            st.success("Mass sync completed!")
            st.rerun()

    with st.sidebar.expander("📥 Download Nifty 50 Data (2022 - Today)", expanded=False):
        st.write("Quickly downloads authentic daily candles from 1 Jan 2022 to today for Nifty 50 large caps.")
        if st.button("⚡ Download Nifty 50 (2022 - Today)", use_container_width=True):
            with st.spinner("Downloading authentic market data from 1 Jan 2022 to today..."):
                cnt = downloader.seed_demo_data()
            st.success(f"Downloaded {cnt:,} authentic candles from 1 Jan 2022 to today!")
            st.rerun()

    # Multi-Page Router: Only the active page executes, ensuring sub-second response times!
    if selected_page == "📊 Quad-Chart View":
        db_symbols = database.get_all_symbols()
        parquet_symbols = parquet_loader.get_parquet_symbols()
        popular_indices = ["RELIANCE", "NIFTY", "BANKNIFTY", "TCS", "HDFCBANK", "INFY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
        all_options = sorted(list(set(popular_indices + db_symbols + parquet_symbols)))
        # Strictly keep only pure Equities and Indices (filter out all debt/bonds starting with '0')
        all_options = [s for s in all_options if not s.startswith("0")]

        if not all_options:
            st.warning("No stocks available in local database or parquet files.")
        else:
            # Check query parameters for initial load or iframe dropdown selection
            qp_stock = st.query_params.get("stock")
            if qp_stock and qp_stock.upper() in all_options:
                st.session_state["quad_sym_select"] = qp_stock.upper()
                del st.query_params["stock"]

            if "quad_sym_select" not in st.session_state:
                st.session_state["quad_sym_select"] = "RELIANCE" if "RELIANCE" in all_options else (all_options[0] if all_options else "RELIANCE")

            tf_opts = [
                "75m (75-Min)",
                "1m (1-Min)",
                "3m (3-Min)",
                "5m (5-Min)",
                "15m (15-Min)",
                "30m (30-Min)",
                "60m (1-Hour)",
                "125m (125-Min)",
                "⚙️ Custom Minutes"
            ]

            qp_tf = st.query_params.get("quad_tf")
            if qp_tf:
                qp_tf_clean = str(qp_tf).strip().lower()
                for opt in tf_opts:
                    if opt.lower().startswith(qp_tf_clean):
                        st.session_state["quad_tf_dropdown"] = opt
                        break
                del st.query_params["quad_tf"]

            if "quad_tf_dropdown" not in st.session_state:
                st.session_state["quad_tf_dropdown"] = "75m (75-Min)"

            # Bridge input to receive seamless in-place stock/timeframe updates from chart toolbar without page reload
            st.markdown(
                """<style>
                div[data-testid="stTextInput"]:has(input[aria-label="quad_bridge"]) {
                    display: none !important;
                    height: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
                </style>""",
                unsafe_allow_html=True
            )
            bridge_val = st.text_input("quad_bridge", key="quad_bridge_input", label_visibility="collapsed")
            if bridge_val:
                try:
                    b_data = json.loads(bridge_val)
                    b_ts = b_data.get("_ts", 0)
                    last_bridge_ts = st.session_state.get("quad_last_bridge_ts", 0)
                    if b_ts > last_bridge_ts:
                        st.session_state["quad_last_bridge_ts"] = b_ts
                        if "stock" in b_data and b_data["stock"] and b_data["stock"].upper() in all_options:
                            st.session_state["quad_sym_select"] = b_data["stock"].upper()
                        if "tf" in b_data and b_data["tf"]:
                            b_tf_clean = str(b_data["tf"]).strip().lower()
                            matched = False
                            for opt in tf_opts:
                                if opt.lower().startswith(b_tf_clean):
                                    st.session_state["quad_tf_dropdown"] = opt
                                    matched = True
                                    break
                            if not matched:
                                m_num = re.search(r"(\d+)", b_tf_clean)
                                if m_num:
                                    mins_val = int(m_num.group(1))
                                    cust_label = f"{mins_val}m ({mins_val}-Min)"
                                    if cust_label not in tf_opts:
                                        tf_opts.insert(-1, cust_label)
                                    st.session_state["quad_tf_dropdown"] = cust_label
                                    st.session_state["quad_custom_mins"] = mins_val
                except Exception:
                    pass

            # Quick symbol chips
            def _set_quad_sym(s):
                st.session_state["quad_sym_select"] = s

            q_chip_cols = st.columns(8)
            q_chip_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "BAJFINANCE", "SBIN", "TITAN", "TATAMOTORS"]
            for i, qsym in enumerate(q_chip_symbols):
                if qsym in all_options:
                    q_chip_cols[i].button(qsym, key=f"qchip_{qsym}", use_container_width=True, on_click=_set_quad_sym, args=(qsym,))

            col_stock_sel, col_tf_sel, col_sync_btn, col_engine, col_rsi_s, col_rsi_chk, col_vol_chk = st.columns([1.6, 1.1, 0.7, 1.0, 0.6, 0.6, 0.6])
            with col_stock_sel:
                sel_stock = st.selectbox(
                    "Select Stock (Search 2,500+ NSE stocks):",
                    options=all_options,
                    key="quad_sym_select"
                )
            with col_tf_sel:
                selected_tf_opt = st.selectbox(
                    "Q4 Timeframe:",
                    options=tf_opts,
                    key="quad_tf_dropdown"
                )
                if "Custom" in selected_tf_opt:
                    target_mins = int(st.number_input("Minutes:", min_value=1, max_value=400, value=45, step=5, key="quad_custom_mins"))
                else:
                    m = re.search(r"(\d+)m", selected_tf_opt)
                    target_mins = int(m.group(1)) if m else 75
                intra_tf_label = f"{target_mins}m"
            with col_sync_btn:
                st.write("")
                st.write("")
                force_sync = st.button("🔄 Sync Live", key="btn_quad_sync_live", use_container_width=True, help="Force sync live market ticks for this stock")

            with col_engine:
                chart_engine = st.radio("Chart Engine", ["TradingView", "Plotly"], horizontal=True, index=0)
            with col_rsi_s:
                rsi_span = int(st.number_input("RSI Span", min_value=2, max_value=100, value=9, step=1))
            with col_rsi_chk:
                show_rsi_panel = st.checkbox("Show Hilega Milega (NK Sir)", value=True)
            with col_vol_chk:
                show_volume = st.checkbox("Show Volume", value=True)

            today_str = datetime.today().strftime("%Y-%m-%d")
            latest_d = database.get_latest_candle_date(sel_stock)

            # Check if stock is missing today's candle or missing historical data
            if force_sync or not latest_d:
                with st.spinner(f"⚡ Syncing complete market history for {sel_stock} from Upstox (2022 to today)..."):
                    downloader.sync_symbol_history(sel_stock, from_date="2022-01-01", to_date=today_str)
            elif latest_d < today_str:
                with st.spinner(f"⚡ Syncing latest market data for {sel_stock}..."):
                    try:
                        import batch_downloader
                        res = batch_downloader.sync_live_market_batch(symbols=[sel_stock])
                        if not res.get("synced"):
                            downloader.sync_symbol_history(sel_stock, from_date=latest_d, to_date=today_str)
                    except Exception:
                        downloader.sync_symbol_history(sel_stock, from_date=latest_d, to_date=today_str)

            # Fetch multi-timeframe candle data for selected stock
            daily_candles = database.get_candles_df(sel_stock)

            # Extra fallback: if daily_candles is still empty, attempt direct historical fetch
            if daily_candles.empty:
                with st.spinner(f"📥 Fetching authentic candles for {sel_stock} from Upstox..."):
                    downloader.sync_symbol_history(sel_stock, from_date="2022-01-01", to_date=today_str)
                    daily_candles = database.get_candles_df(sel_stock)

            if daily_candles.empty:
                st.warning(f"No candle data available for {sel_stock} on Upstox.")
            else:
                if len(daily_candles) < 50:
                    st.info(f"ℹ️ **Newly Listed / Short-History Stock:** **{sel_stock}** has **{len(daily_candles)} trading sessions** on NSE (debuted on {daily_candles.index.min().strftime('%d %b %Y')}). Displaying all available session candles. (Long-term 200-period EMAs require 200 sessions to calculate).")

                # 1. Monthly DataFrame
                m_df = scanner.resample_ohlcv(daily_candles, "monthly")
                m_df["EMA_5"] = m_df["close"].ewm(span=5, adjust=False).mean()
                m_df["EMA_20"] = m_df["close"].ewm(span=20, adjust=False).mean()
                m_df["AVWAP_MAR2020"] = scanner.calculate_anchored_vwap(m_df, "2020-03-01")
                m_df["AVWAP_JUN2022"] = scanner.calculate_anchored_vwap(m_df, "2022-06-01")
                m_df["RSI"] = scanner.calculate_rsi(m_df["close"], span=rsi_span)
                m_df["RSI_EMA3"] = scanner.calculate_ema(m_df["RSI"], span=3)
                m_df["RSI_WMA21"] = scanner.calculate_wma(m_df["RSI"], period=21)

                # 2. Weekly DataFrame
                w_df = scanner.resample_ohlcv(daily_candles, "weekly")
                w_df["EMA_20"] = w_df["close"].ewm(span=20, adjust=False).mean()
                w_df["EMA_50"] = w_df["close"].ewm(span=50, adjust=False).mean()
                w_df["EMA_200"] = w_df["close"].ewm(span=200, adjust=False).mean()
                w_df["RSI"] = scanner.calculate_rsi(w_df["close"], span=rsi_span)
                w_df["RSI_EMA3"] = scanner.calculate_ema(w_df["RSI"], span=3)
                w_df["RSI_WMA21"] = scanner.calculate_wma(w_df["RSI"], period=21)

                # 3. Daily DataFrame
                d_df = daily_candles.copy()
                d_df["EMA_20"] = d_df["close"].ewm(span=20, adjust=False).mean()
                d_df["EMA_50"] = d_df["close"].ewm(span=50, adjust=False).mean()
                d_df["EMA_200"] = d_df["close"].ewm(span=200, adjust=False).mean()
                d_df["RSI"] = scanner.calculate_rsi(d_df["close"], span=rsi_span)
                d_df["RSI_EMA3"] = scanner.calculate_ema(d_df["RSI"], span=3)
                d_df["RSI_WMA21"] = scanner.calculate_wma(d_df["RSI"], period=21)

                # 4. Intraday DataFrame (strictly resampled from 1-minute data)
                if target_mins == 75:
                    intra_df = parquet_loader.ensure_symbol_75m_candles(sel_stock, min_bars=100)
                else:
                    intra_df = parquet_loader.ensure_symbol_custom_minute_candles(sel_stock, interval_minutes=target_mins, min_bars=100)
                
                if intra_df is not None and not intra_df.empty:
                    # Requested EMAs: 9, 13, 20, 26, 50, 200
                    intra_df["EMA_9"] = intra_df["close"].ewm(span=9, adjust=False).mean()
                    intra_df["EMA_13"] = intra_df["close"].ewm(span=13, adjust=False).mean()
                    intra_df["EMA_20"] = intra_df["close"].ewm(span=20, adjust=False).mean()
                    intra_df["EMA_26"] = intra_df["close"].ewm(span=26, adjust=False).mean()
                    intra_df["EMA_50"] = intra_df["close"].ewm(span=50, adjust=False).mean()
                    intra_df["EMA_200"] = intra_df["close"].ewm(span=200, adjust=False).mean()

                    # Attach Daily EMA 20 mapped onto 75-min intraday candles
                    if d_df is not None and not d_df.empty:
                        d_ema20 = d_df["EMA_20"] if "EMA_20" in d_df.columns else d_df["close"].ewm(span=20, adjust=False).mean()
                        d_dt_idx = pd.to_datetime(d_df.index)
                        d_dates = d_dt_idx.tz_localize(None).date if hasattr(d_dt_idx, 'tz_localize') and d_dt_idx.tz is not None else d_dt_idx.date
                        d_map = pd.Series(d_ema20.values, index=d_dates)
                        d_map = d_map[~d_map.index.duplicated(keep="last")]

                        i_dt_idx = pd.to_datetime(intra_df.index)
                        i_dates = i_dt_idx.tz_localize(None).date if hasattr(i_dt_idx, 'tz_localize') and i_dt_idx.tz is not None else i_dt_idx.date
                        intra_df["Daily_EMA_20"] = pd.Series([d_map.get(d, np.nan) for d in i_dates], index=intra_df.index).ffill().bfill()

                    intra_df["RSI"] = scanner.calculate_rsi(intra_df["close"], span=rsi_span)
                    intra_df["RSI_EMA3"] = scanner.calculate_ema(intra_df["RSI"], span=3)
                    intra_df["RSI_WMA21"] = scanner.calculate_wma(intra_df["RSI"], period=21)

                    # Attach Weekly CPR levels (TC, P, BC) + R1 and S1
                    intra_df = scanner.attach_weekly_cpr_to_intraday(intra_df, daily_candles)
                else:
                    intra_df = pd.DataFrame()

                # Alignment status summary banner
                res_meta = scanner.evaluate_multiframe_alignment(sel_stock)
                if res_meta:
                    st.info(f"**{sel_stock} Status:** {res_meta['Stage']} | Monthly: {res_meta['Monthly Match']} | Weekly: {res_meta['Weekly Match']} | Daily: {res_meta['Daily Match']} | 75-Min: {res_meta['75m Match']}")

                pivot_dict_75 = {
                    "Weekly_R1": {"color": "#EF4444", "dash": "dash", "width": 1.8, "name": "Weekly R1"},
                    "Weekly_P":  {"color": "#38BDF8", "dash": "dash", "width": 2.0, "name": "Weekly Pivot (P)"},
                    "Weekly_S1": {"color": "#10B981", "dash": "dash", "width": 1.8, "name": "Weekly S1"}
                }

                if chart_engine == "TradingView":
                    c_lead, c_fs = st.columns([3, 1])
                    with c_lead:
                        st.caption("💡 **Interactive 4-Quadrant Matrix:** Click **'⛶ FULLSCREEN'** (or press **'F'**) to expand all 4 charts across your monitor. Use **mousewheel** on any chart to zoom smoothly.")
                    with c_fs:
                        theater_mode = st.toggle("⛶ Ultra Height (1,100px)", value=False, key="quad_theater_mode")

                    chart_render_height = 1100 if theater_mode else 880
                    quad_html = tradingview_charts.generate_quad_chart_html(
                        symbol=sel_stock,
                        monthly_df=m_df,
                        weekly_df=w_df,
                        daily_df=d_df.tail(1500) if len(d_df) > 1500 else d_df,
                        intra_df=intra_df.tail(2500) if (intra_df is not None and len(intra_df) > 2500) else intra_df,
                        rsi_span=rsi_span,
                        show_rsi=show_rsi_panel,
                        show_volume=show_volume,
                        show_candles=True,
                        show_line=True,
                        pivot_dict_75=pivot_dict_75,
                        height=chart_render_height,
                        theme=theme,
                        all_symbols=all_options,
                        intra_tf_label=intra_tf_label
                    )
                    components.html(quad_html, height=chart_render_height + 15)
                else:
                    # 2x2 Grid of Plotly Charts Fallback
                    chart_h = 440 if show_rsi_panel else 380
                    grid_col1, grid_col2 = st.columns(2)

                    with grid_col1:
                        st.subheader("1️⃣ Monthly Chart (EMAs + AVWAP Mar'20 / Jun'22)")
                        m_rsi = m_df["RSI"].iloc[-1]
                        m_ema3 = m_df["RSI_EMA3"].iloc[-1]
                        m_wma21 = m_df["RSI_WMA21"].iloc[-1]
                        rsi_50_badge = "🟢 RSI ≥ 50" if m_rsi >= 50 else "🔴 RSI < 50"
                        cross_badge = "🟢 EMA 3 ≥ WMA 21" if m_ema3 >= m_wma21 else "🔴 EMA 3 < WMA 21"
                        st.caption(f"Rule: Close > 5 EMA (Green) & 5>20 EMA (Blue) | AVWAP Mar'20 (Sky Blue) | AVWAP Jun'22 (Sky Blue Dotted) | **RSI ({rsi_span}):** {m_rsi:.1f} ({rsi_50_badge})")
                        fig_m = create_candlestick_chart(
                            m_df.tail(48), sel_stock, "Monthly", {
                                "EMA_5": "#4CAF50",
                                "EMA_20": "#2962FF",
                                "AVWAP_MAR2020": {"color": "#38BDF8", "name": "AVWAP Mar 2020", "dash": "solid"},
                                "AVWAP_JUN2022": {"color": "#38BDF8", "name": "AVWAP Jun 2022", "dash": "dot"}
                            },
                            height=chart_h, is_intraday=False, show_volume=show_volume, show_rsi=show_rsi_panel, rsi_span=rsi_span,
                            highlight_rsi_50=True, theme=theme
                        )
                        st.plotly_chart(fig_m, use_container_width=True)

                    with grid_col2:
                        st.subheader("2️⃣ Weekly Chart (20 + 50 + 200 EMA)")
                        w_rsi = w_df["RSI"].iloc[-1]
                        st.caption(f"Rule: Close > 20 EMA (Blue) > 50 EMA (Red) > 200 EMA (Black) | **RSI ({rsi_span}):** {w_rsi:.1f}")
                        fig_w = create_candlestick_chart(
                            w_df.tail(100), sel_stock, "Weekly", {"EMA_20": "#2962FF", "EMA_50": "#FF5252", "EMA_200": "#131722" if theme == "light" else "#FFFFFF"},
                            height=chart_h, is_intraday=False, show_volume=show_volume, show_rsi=show_rsi_panel, rsi_span=rsi_span,
                            theme=theme
                        )
                        st.plotly_chart(fig_w, use_container_width=True)

                    grid_col3, grid_col4 = st.columns(2)

                    with grid_col3:
                        st.subheader("3️⃣ Daily Chart (20 + 50 + 200 EMA)")
                        d_rsi = d_df["RSI"].iloc[-1]
                        st.caption(f"Rule: Close > 20 EMA (Blue) > 50 EMA (Red) > 200 EMA (Black) | **RSI ({rsi_span}):** {d_rsi:.1f}")
                        fig_d = create_candlestick_chart(
                            d_df.tail(120), sel_stock, "Daily", {"EMA_20": "#2962FF", "EMA_50": "#FF5252", "EMA_200": "#131722" if theme == "light" else "#FFFFFF"},
                            height=chart_h, is_intraday=False, show_volume=show_volume, show_rsi=show_rsi_panel, rsi_span=rsi_span,
                            theme=theme
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                    with grid_col4:
                        st.subheader("4️⃣ 75-Min Chart (EMAs + Daily 20 EMA + Weekly Pivot)")
                        i_last = intra_df.iloc[-1]
                        i_rsi = intra_df["RSI"].iloc[-1]
                        d_ema20_val = i_last.get("Daily_EMA_20", 0)
                        d_ema20_str = f" | Daily 20 EMA: ₹{d_ema20_val:.2f}" if d_ema20_val else ""
                        ema20_val = i_last.get("EMA_20", 0)
                        ema20_str = f" | 20 EMA (Blue): ₹{ema20_val:.2f}" if ema20_val else ""
                        st.caption(f"**Close:** ₹{i_last['close']:.2f} | **RSI ({rsi_span}):** {i_rsi:.1f} | 9 EMA (Green): ₹{i_last['EMA_9']:.2f} | 13 EMA (Sky Blue): ₹{i_last['EMA_13']:.2f}{ema20_str}{d_ema20_str} | 26 EMA (Purple): ₹{i_last['EMA_26']:.2f} | 50 EMA (Red): ₹{i_last['EMA_50']:.2f} | 200 EMA (Black): ₹{i_last['EMA_200']:.2f}")
                        fig_75 = create_candlestick_chart(
                            intra_df.tail(80), sel_stock, "75-Minute (EMAs + Daily 20 EMA + Weekly Pivot)", {
                                "EMA_9": "#4CAF50",
                                "EMA_13": "#38BDF8",
                                "EMA_20": "#2962FF",
                                "Daily_EMA_20": "#2962FF",
                                "EMA_26": "#9C27B0",
                                "EMA_50": "#FF5252",
                                "EMA_200": "#131722" if theme == "light" else "#FFFFFF"
                            }, height=chart_h, is_intraday=True, pivot_dict=pivot_dict_75,
                            show_volume=show_volume, show_rsi=show_rsi_panel, rsi_span=rsi_span,
                            theme=theme
                        )
                        st.plotly_chart(fig_75, use_container_width=True)


    elif selected_page == "🚀 Alignment Scanner":
        st.subheader("🚀 Multi-Timeframe EMA Alignment Scanner")
        st.markdown("""
        <div class="rule-card">
            <b>Strategy Rules Enforced:</b><br/>
            • <b>Universe:</b> Pure NSE Equities only (all Indices &amp; BSE stocks strictly excluded)<br/>
            • <b>Monthly (Macro):</b> Price Close &gt; 5 EMA &nbsp;AND&nbsp; 5 EMA &gt; 20 EMA<br/>
            • <b>Hilega Milega (NK Sir):</b> RSI(9) &ge; 50 (Blue Line) &nbsp;AND&nbsp; EMA 3 (Green) &ge; WMA 21 (Red)<br/>
            • <b>Weekly (Trend):</b> Price Close &gt; 20 EMA &nbsp;AND&nbsp; 20 EMA &gt; 50 EMA &gt; 200 EMA<br/>
            • <b>Daily (Setup):</b> Price Close &gt; 20 EMA &nbsp;AND&nbsp; 20 EMA &gt; 50 EMA &gt; 200 EMA<br/>
            • <b>75-Min (Trigger):</b> Price Close &gt; 20 EMA &nbsp;AND&nbsp; 5 EMA &gt; 20 EMA
        </div>
        """, unsafe_allow_html=True)

        col_sf1, col_sf2 = st.columns([3, 1])
        with col_sf1:
            stage_filter = st.selectbox(
                "Filter Alignment Stage",
                options=[
                    "All",
                    "Stage 3 (Full Alignment Only)",
                    "Stage 2+ (M+W Aligned)",
                    "Stage 1+ (Monthly Pass)",
                    "Stage 3 + Hilega Milega Bullish (RSI>=50 & EMA3>=WMA21)",
                    "Hilega Milega Bullish Only (RSI>=50 & EMA3>=WMA21)"
                ],
                index=0
            )
        with col_sf2:
            st.write("")
            st.write("")
            if st.button("🔄 Re-Run Alignment Scan", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        with st.spinner("Scanning alignment across stocks in local database..."):
            df_align = get_cached_alignment_scan(stage_filter)

        if df_align.empty:
            st.info("No stocks match the selected filter, or the local database is empty.")
            st.caption("Tip: Use 'Seed 5-Year Demo Data' or 'Start Upstox Download' in the sidebar.")
        else:
            # Metric Summary Cards
            s3_count = len(df_align[df_align["Score"] == 3])
            s2_count = len(df_align[df_align["Score"] == 2])
            rsi_pass_count = len(df_align[df_align["Monthly_RSI_Match"] == "✅ PASS"]) if "Monthly_RSI_Match" in df_align.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stage 3: Full Alignment 🚀", s3_count)
            m2.metric("Stage 2: M + W Aligned 🟢", s2_count)
            m3.metric("Monthly RSI Match (50 & EMA3>WMA21) ⚡", rsi_pass_count)
            m4.metric("Total Scanned", len(df_align))

            display_cols = [
                "Symbol", "Price", "Stage",
                "Monthly Match", "Monthly_RSI_Match",
                "M_RSI", "M_RSI_EMA3", "M_RSI_WMA21",
                "M_RSI>=50", "M_EMA3>=WMA21",
                "Weekly Match", "Daily Match", "75m Match",
                "Daily_RSI", "75m_RSI",
                "M_Close>5EMA", "M_5>20EMA",
                "W_Close>20EMA", "W_20>50EMA", "W_50>200EMA",
                "D_Close>20EMA", "D_20>50EMA", "D_50>200EMA"
            ]
            
            sub_df = df_align[[c for c in display_cols if c in df_align.columns]]

            # Enforce exactly 2 decimal places on all float columns in Alignment Scanner
            float_cols = sub_df.select_dtypes(include=["float", "float64"]).columns.tolist()
            format_dict = {col: "{:.2f}" for col in float_cols}

            st.dataframe(
                sub_df.style.format(format_dict).map(
                    lambda v: "background-color: #064E3B; color: #34D399; font-weight: bold;" if v == "✅ PASS" or v == "✅"
                    else ("background-color: #7F1D1D; color: #F87171;" if v == "❌ FAIL" or v == "❌" else "")
                ),
                use_container_width=True,
                height=480
            )

            # Export to CSV with all floats rounded to 2 decimal places
            csv_data = df_align.round(2).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Alignment Scan to CSV",
                data=csv_data,
                file_name=f"alignment_scan_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    elif selected_page == "📈 Trading Terminal":
        db_symbols = database.get_all_symbols()
        parquet_symbols = parquet_loader.get_parquet_symbols()
        popular_indices = ["RELIANCE", "NIFTY", "BANKNIFTY", "TCS", "HDFCBANK", "INFY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
        all_options = sorted(list(set(popular_indices + db_symbols + parquet_symbols)))
        all_options = [s for s in all_options if not s.startswith("0")]

        st.subheader("📈 TradingView Advanced Trading Terminal")
        st.caption("Professional TradingView charting suite with full-screen mode, zoom controls, 1-min to Monthly timeframes, EMAs, SMAs, Bollinger Bands, Supertrend, VWAP, MACD, CPR, and RSI.")

        # Top row: Controls
        t_col1, t_col2, t_col3 = st.columns([2.2, 1.4, 1.4])
        with t_col1:
            term_stock = st.selectbox(
                "Select Stock (Search 2,500+ NSE symbols):",
                options=all_options if all_options else ["BAJFINANCE", "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
                index=all_options.index(st.session_state.get("term_sym_select", "RELIANCE")) if (all_options and st.session_state.get("term_sym_select", "RELIANCE") in all_options) else 0,
                key="term_sym_select"
            )

        with t_col2:
            tf_options = [
                "Daily", "75-Minute", "1-Minute", "3-Minute", "5-Minute", 
                "15-Minute", "30-Minute", "60-Minute (1H)", "125-Minute", 
                "Weekly", "Monthly", "⚙️ Custom Timeframe"
            ]
            term_tf = st.selectbox("Timeframe", tf_options, index=0, key="term_tf_sel")

        with t_col3:
            clean_sym = term_stock.upper().replace("-EQ", "").replace(".NS", "")
            st.write("")
            st.write("")
            st.link_button(f"🚀 Open {clean_sym} in TradingView Web", f"https://in.tradingview.com/chart/?symbol=NSE:{clean_sym}", use_container_width=True)

        # Indicator Toggles Row (Organized Expander)
        with st.expander("🛠️ Indicator Settings & Overlay Toggles", expanded=True):
            ind_c1, ind_c2, ind_c3, ind_c4, ind_c5 = st.columns(5)
            with ind_c1:
                st.markdown("**Moving Averages**")
                term_ema = st.checkbox("EMAs (9, 20, 50, 200)", value=True, key="term_ema_chk")
                term_sma = st.checkbox("SMAs (20, 50, 200)", value=False, key="term_sma_chk")
            with ind_c2:
                st.markdown("**Volatility & Trend**")
                term_bb = st.checkbox("Bollinger Bands (20, 2)", value=False, key="term_bb_chk")
                term_st = st.checkbox("Supertrend (10, 3)", value=False, key="term_st_chk")
            with ind_c3:
                st.markdown("**Intraday & Levels**")
                term_vwap = st.checkbox("VWAP (Intraday)", value=True, key="term_vwap_chk")
                term_piv = st.checkbox("CPR / Weekly Pivots", value=True, key="term_piv_chk")
            with ind_c4:
                st.markdown("**Oscillators**")
                term_rsi = st.checkbox("RSI Panel (14)", value=True, key="term_rsi_chk")
                term_macd = st.checkbox("MACD Panel (12, 26, 9)", value=True, key="term_macd_chk")
            with ind_c5:
                st.markdown("**Volume & Momentum**")
                term_vol = st.checkbox("Volume + 20 MA", value=True, key="term_vol_chk")
                term_stoch = st.checkbox("Stochastic (14, 3, 3)", value=False, key="term_stoch_chk")

        custom_val = 10
        custom_unit = "Minutes"
        if term_tf == "⚙️ Custom Timeframe":
            cust_c1, cust_c2, cust_c3 = st.columns([1.5, 1.5, 3.0])
            with cust_c1:
                custom_val = st.number_input("Interval Value", min_value=1, max_value=500, value=10, step=1, key="term_custom_val")
            with cust_c2:
                custom_unit = st.selectbox("Unit", ["Minutes", "Hours", "Days", "Weeks"], index=0, key="term_custom_unit")
            with cust_c3:
                display_tf = f"{custom_val}-{custom_unit[:-1]}"
                st.info(f"Active Custom Timeframe: **{display_tf}** (Authentic multi-timeframe resampling)")
        else:
            display_tf = term_tf

        # Quick chips for top symbols
        def _set_term_sym(s):
            st.session_state["term_sym_select"] = s

        chip_cols = st.columns(8)
        chip_symbols = ["BAJFINANCE", "TCS", "RELIANCE", "INFY", "SBIN", "TITAN", "NTPC", "KOTAKBANK"]
        for i, csym in enumerate(chip_symbols):
            chip_cols[i].button(csym, key=f"chip_{csym}", use_container_width=True, on_click=_set_term_sym, args=(csym,))

        # Pull today's live Upstox ticks if market is active/recent
        downloader.sync_live_market_candles(term_stock)

        # Load data for term_stock
        term_candles = database.get_candles_df(term_stock)
        if term_candles.empty:
            if term_stock in parquet_symbols:
                with st.spinner(f"Loading {term_stock} from Parquet dataset..."):
                    parquet_loader.import_symbol_to_database(term_stock)
                    term_candles = database.get_candles_df(term_stock)

            if term_candles.empty:
                with st.spinner(f"📥 Downloading historical candles for {term_stock} from Upstox..."):
                    downloader.sync_symbol_history(term_stock, from_date="2022-01-01")
                    term_candles = database.get_candles_df(term_stock)

        if term_candles.empty:
            st.warning(f"No candle data available for {term_stock}.")
        else:
            # Quote metrics banner
            last_close = term_candles["close"].iloc[-1]
            prev_close = term_candles["close"].iloc[-2] if len(term_candles) > 1 else last_close
            chg = last_close - prev_close
            chg_pct = (chg / prev_close) * 100 if prev_close > 0 else 0
            hi_52 = term_candles["high"].tail(250).max()
            lo_52 = term_candles["low"].tail(250).min()

            m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
            m_c1.metric(f"NSE:{clean_sym}", f"₹{last_close:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
            m_c2.metric("52-Week High", f"₹{hi_52:,.2f}")
            m_c3.metric("52-Week Low", f"₹{lo_52:,.2f}")
            m_c4.metric("Volume", f"{int(term_candles['volume'].iloc[-1]):,} shares")
            m_c5.metric("Active Timeframe", f"{display_tf}")

            # Resample based on chosen timeframe
            is_intra = False

            if term_tf == "Monthly":
                term_df = scanner.resample_ohlcv(term_candles, "monthly")
                term_df["AVWAP_MAR2020"] = scanner.calculate_anchored_vwap(term_df, "2020-03-01")
                term_df["AVWAP_JUN2022"] = scanner.calculate_anchored_vwap(term_df, "2022-06-01")
                is_intra = False

            elif term_tf == "Weekly":
                term_df = scanner.resample_ohlcv(term_candles, "weekly")
                is_intra = False

            elif term_tf == "Daily":
                term_df = term_candles.tail(1500).copy() if len(term_candles) > 1500 else term_candles.copy()
                is_intra = False

            elif term_tf == "⚙️ Custom Timeframe" and custom_unit == "Days":
                term_df = term_candles.resample(f"{custom_val}D").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna()
                is_intra = False

            elif term_tf == "⚙️ Custom Timeframe" and custom_unit == "Weeks":
                term_df = term_candles.resample(f"{custom_val}W-FRI").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna()
                is_intra = False

            else:
                # All Intraday Minute Timeframes (1m, 3m, 5m, 15m, 30m, 60m, 75m, 125m, or custom minutes/hours)
                minute_map = {
                    "1-Minute": 1,
                    "3-Minute": 3,
                    "5-Minute": 5,
                    "15-Minute": 15,
                    "30-Minute": 30,
                    "60-Minute (1H)": 60,
                    "75-Minute": 75,
                    "125-Minute": 125
                }
                if term_tf in minute_map:
                    target_mins = minute_map[term_tf]
                elif term_tf == "⚙️ Custom Timeframe" and custom_unit == "Hours":
                    target_mins = custom_val * 60
                else:  # Custom Minutes
                    target_mins = custom_val

                with st.spinner(f"Resampling authentic {target_mins}-minute candles for {term_stock}..."):
                    term_df = parquet_loader.ensure_symbol_custom_minute_candles(term_stock, interval_minutes=target_mins)

                if term_df.empty:
                    st.info(f"1-minute Parquet history not yet cached for {term_stock}. Displaying daily candles.")
                    term_df = term_candles.tail(200)
                    is_intra = False
                else:
                    is_intra = True

            clean_chart_id = re.sub(r'[^a-zA-Z0-9_]', '_', f"adv_term_{term_stock}_{display_tf}")
            term_chart_html = tradingview_charts.generate_advanced_terminal_html(
                term_df,
                term_stock,
                f"{display_tf} Trading Terminal",
                indicators={
                    "ema": term_ema,
                    "sma": term_sma,
                    "bb": term_bb,
                    "supertrend": term_st,
                    "vwap": term_vwap and is_intra,
                    "cpr": term_piv,
                    "volume": term_vol,
                    "rsi": term_rsi,
                    "macd": term_macd,
                    "stoch": term_stoch
                },
                height=740,
                is_intraday=is_intra,
                chart_id=clean_chart_id,
                theme=theme
            )
            components.html(term_chart_html, height=760)

    elif selected_page == "🗺️ Market Heatmap":
        st.subheader("🗺️ Market Heatmap (Definedge Sectoral Classification)")
        st.caption("Visual top-down sector breadth and performance analysis across NSE equities with Daily (1D), Weekly (1W), Monthly (1M), and Yearly (1Y) timeframes.")

        # Heatmap Filter Controls Row
        h_col1, h_col2, h_col3, h_col4 = st.columns([1.8, 1.8, 1.6, 1.6])
        with h_col1:
            tf_filter = st.selectbox(
                "📅 Performance Timeframe:",
                ["Daily (1D)", "Weekly (1W)", "Monthly (1M)", "Yearly (1Y)"],
                index=0,
                key="hm_tf_filter"
            )
            tf_ret_col = {
                "Daily (1D)": "Return_1D",
                "Weekly (1W)": "Return_1W",
                "Monthly (1M)": "Return_1M",
                "Yearly (1Y)": "Return_1Y"
            }[tf_filter]

        with h_col2:
            all_sectors_options = ["All Sectors"] + sector_data.DEFINEDGE_SECTORS_LIST + list(sector_data.NIFTY_SECTORAL_INDICES.keys())
            sector_filter = st.selectbox(
                "🏢 Definedge Sector / Index:",
                all_sectors_options,
                index=0,
                key="hm_sector_filter"
            )

        with h_col3:
            univ_filter = st.selectbox(
                "🎯 Stock Universe:",
                ["Top 100 (CoinMarketCap View)", "Top 250", "Top 500 (Most Liquid)", "Nifty 50", "All NSE Equities (3,000+)"],
                index=0,
                key="hm_univ_filter"
            )

        with h_col4:
            size_mode = st.selectbox(
                "⚖️ Tile Sizing:",
                ["Definedge Equal-Weight", "Turnover / Volume-Weight"],
                index=1,
                key="hm_size_mode"
            )

        # Load Heatmap Data (Cached for snappy responsiveness)
        @st.cache_data(ttl=60)
        def _get_cached_heatmap_df():
            return sector_data.calculate_market_heatmap_data()

        with st.spinner("Calculating multi-timeframe sectoral returns from SQLite daily candles..."):
            df_hm = _get_cached_heatmap_df()

        if df_hm.empty:
            st.warning("No market data available to render the Heatmap. Please ensure daily candles are loaded.")
        else:
            df_filtered = df_hm.copy()

            # Filter by Sector or Nifty Index if selected
            if sector_filter != "All Sectors":
                if sector_filter in sector_data.NIFTY_SECTORAL_INDICES:
                    allowed_syms = set(sector_data.NIFTY_SECTORAL_INDICES[sector_filter])
                    df_filtered = df_filtered[df_filtered["Symbol"].isin(allowed_syms)]
                else:
                    df_filtered = df_filtered[df_filtered["Sector"] == sector_filter]

            # Filter by Universe (Top N by Turnover)
            if univ_filter == "Top 100 (CoinMarketCap View)":
                df_filtered = df_filtered.sort_values(by="Turnover", ascending=False).head(100)
            elif univ_filter == "Top 250":
                df_filtered = df_filtered.sort_values(by="Turnover", ascending=False).head(250)
            elif univ_filter == "Top 500 (Most Liquid)":
                df_filtered = df_filtered.sort_values(by="Turnover", ascending=False).head(500)
            elif univ_filter == "Nifty 50":
                n50 = set(config.NIFTY_50_SYMBOLS)
                df_filtered = df_filtered[df_filtered["Symbol"].isin(n50)]

            # Sizing column
            if size_mode == "Definedge Equal-Weight":
                df_filtered["Tile_Size"] = 1.0
            else:
                df_filtered["Tile_Size"] = df_filtered["Turnover"].clip(lower=1000)

            # Market Breadth Summary Cards
            tot_stocks = len(df_filtered)
            advances = int((df_filtered[tf_ret_col] > 0).sum())
            declines = int((df_filtered[tf_ret_col] < 0).sum())
            unchanged = tot_stocks - advances - declines
            ad_ratio = round(advances / max(1, declines), 2)

            # Top Performing & Lagging Sectors
            sector_perf = df_hm.groupby("Sector")[tf_ret_col].mean().sort_values(ascending=False)
            top_sector = f"{sector_perf.index[0]} ({sector_perf.iloc[0]:+.2f}%)" if not sector_perf.empty else "N/A"
            bottom_sector = f"{sector_perf.index[-1]} ({sector_perf.iloc[-1]:+.2f}%)" if not sector_perf.empty else "N/A"

            b_c1, b_c2, b_c3, b_c4, b_c5, b_c6 = st.columns(6)
            b_c1.metric("Stocks Displayed", f"{tot_stocks:,}")
            b_c2.metric("🟢 Advances", f"{advances}", f"{round(advances/tot_stocks*100, 1)}%" if tot_stocks else "0%")
            b_c3.metric("🔴 Declines", f"{declines}", f"{round(declines/tot_stocks*100, 1)}%" if tot_stocks else "0%")
            b_c4.metric("⚖️ A/D Ratio", f"{ad_ratio}x", "Bullish" if ad_ratio > 1.2 else ("Bearish" if ad_ratio < 0.8 else "Neutral"))
            b_c5.metric("🏆 Top Sector", top_sector)
            b_c6.metric("⚠️ Lagging Sector", bottom_sector)

            # Definedge Sector Leaderboard Bar Chart (Top-Down Flow)
            with st.expander("📊 Definedge Sector Relative Strength Leaderboard", expanded=False):
                sector_perf_df = sector_perf.reset_index()
                sector_perf_df.columns = ["Sector", "Mean_Return"]
                sector_perf_df["Color"] = sector_perf_df["Mean_Return"].apply(lambda v: "#089981" if v >= 0 else "#F23645")

                fig_lead = go.Figure()
                fig_lead.add_trace(go.Bar(
                    x=sector_perf_df["Mean_Return"],
                    y=sector_perf_df["Sector"],
                    orientation="h",
                    marker=dict(color=sector_perf_df["Color"]),
                    text=sector_perf_df["Mean_Return"].apply(lambda v: f"{v:+.2f}%"),
                    textposition="auto"
                ))
                fig_lead.update_layout(
                    title=f"Sector Average Returns ({tf_filter})",
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=max(350, len(sector_perf_df) * 22),
                    margin=dict(l=180, r=30, t=40, b=30),
                    xaxis_title="% Average Return",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_lead, use_container_width=True)

            # Render CoinMarketCap / Finviz-Style Squarified Market Heatmap
            cmc_heatmap_html = sector_data.generate_cmc_heatmap_html(
                df_filtered,
                active_tf=tf_filter,
                theme=theme,
                height=740
            )
            components.html(cmc_heatmap_html, height=760)

            # Top 10 Gainers & Losers within Selected Scope
            st.markdown(f"### 🏆 Top Movers ({tf_filter})")
            m_col1, m_col2 = st.columns(2)

            with m_col1:
                st.markdown("##### 🟢 Top 10 Gainers")
                top_gainers = df_filtered.sort_values(by=tf_ret_col, ascending=False).head(10)
                g_display = top_gainers[["Symbol", "Company_Name", "Sector", "LTP", tf_ret_col, "Volume"]].copy()
                g_display.columns = ["Symbol", "Company", "Sector", "LTP (₹)", f"{tf_filter} %", "Volume"]
                st.dataframe(
                    g_display.style.format({
                        "LTP (₹)": "₹{:,.2f}",
                        f"{tf_filter} %": "{:+.2f}%",
                        "Volume": "{:,}"
                    }).map(lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 else "", subset=[f"{tf_filter} %"]),
                    use_container_width=True,
                    height=380
                )

            with m_col2:
                st.markdown("##### 🔴 Top 10 Losers")
                top_losers = df_filtered.sort_values(by=tf_ret_col, ascending=True).head(10)
                l_display = top_losers[["Symbol", "Company_Name", "Sector", "LTP", tf_ret_col, "Volume"]].copy()
                l_display.columns = ["Symbol", "Company", "Sector", "LTP (₹)", f"{tf_filter} %", "Volume"]
                st.dataframe(
                    l_display.style.format({
                        "LTP (₹)": "₹{:,.2f}",
                        f"{tf_filter} %": "{:+.2f}%",
                        "Volume": "{:,}"
                    }).map(lambda v: "color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else "", subset=[f"{tf_filter} %"]),
                    use_container_width=True,
                    height=380
                )


if __name__ == "__main__":
    main()
