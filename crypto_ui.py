"""
crypto_ui.py
------------
Streamlit UI for the dedicated Crypto Trading Terminal.
Provides the exact same desktop-grade TradingView Advanced Trading Terminal
experience as the equity trading terminal, powered by Delta Exchange live & historical data
strictly for BTCUSD and ETHUSD, stored in data/crypto_market.duckdb.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

import crypto_store
import delta_exchange_client
import scanner
import tradingview_charts
from tradingview_charts import generate_advanced_terminal_html, render_tradingview_cloud_widget


def _get_theme_styles(theme: str) -> dict:
    is_light = (str(theme).lower() == "light")
    return {
        "is_light": is_light,
        "card_bg": "#FFFFFF" if is_light else "#1E222D",
        "card_border": "#CBD5E1" if is_light else "#2A2E39",
        "text_primary": "#0F172A" if is_light else "#F8FAFC",
        "text_secondary": "#64748B" if is_light else "#94A3B8",
        "green": "#10B981",
        "red": "#EF4444",
        "accent": "#3B82F6"
    }


def render_metric_card(label: str, value: str, sub: str = "", color: str = "normal", styles: dict = None):
    c_map = {
        "green": "#10B981",
        "red": "#EF4444",
        "blue": "#38BDF8",
        "orange": "#F59E0B",
        "purple": "#A855F7",
        "normal": styles["text_primary"] if styles else "#F8FAFC"
    }
    val_color = c_map.get(color, c_map["normal"])
    st.markdown(f"""
    <div style="background: {styles['card_bg']}; border: 1px solid {styles['card_border']}; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
        <div style="color: {styles['text_secondary']}; font-size: 12px; font-weight: 500; text-transform: uppercase;">{label}</div>
        <div style="color: {val_color}; font-size: 22px; font-weight: 700; margin: 4px 0 2px 0;">{value}</div>
        <div style="color: {styles['text_secondary']}; font-size: 11px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_crypto_page(theme: str = "dark"):
    """
    Main render function for the Crypto Trading Terminal page.
    Replicates the complete advanced terminal functionality of the equity terminal.
    """
    styles = _get_theme_styles(theme)

    st.subheader("🪙 TradingView Advanced Crypto Trading Terminal")
    st.caption("Professional TradingView charting suite for BTCUSD & ETHUSD with full-screen mode, drawing tools, zoom controls, 1-min to Weekly timeframes, EMAs, SMAs, Bollinger Bands, Supertrend, VWAP, MACD, CPR, and RSI (Delta Exchange API).")

    # Quick Asset Selection Chips
    def _set_crypto_sym(s):
        st.session_state["crypto_symbol_select"] = s

    chip_col1, chip_col2 = st.columns([1, 1])
    with chip_col1:
        st.button("⚡ Bitcoin (BTCUSD Perpetual)", key="chip_btcusd", use_container_width=True, on_click=_set_crypto_sym, args=("BTCUSD",))
    with chip_col2:
        st.button("⚡ Ethereum (ETHUSD Perpetual)", key="chip_ethusd", use_container_width=True, on_click=_set_crypto_sym, args=("ETHUSD",))

    # Top row: Controls
    t_col1, t_col2, t_col3 = st.columns([2.0, 1.6, 1.4])
    with t_col1:
        crypto_symbol = st.selectbox(
            "Select Crypto Asset (Delta Exchange):",
            options=["BTCUSD", "ETHUSD"],
            index=0 if st.session_state.get("crypto_symbol_select", "BTCUSD") == "BTCUSD" else 1,
            key="crypto_symbol_select"
        )

    with t_col2:
        tf_options = [
            "1-Minute", "3-Minute", "5-Minute", "15-Minute", "30-Minute", 
            "60-Minute (1H)", "4-Hour", "Daily", "Weekly"
        ]
        term_tf = st.selectbox("Timeframe", tf_options, index=3, key="crypto_term_tf_sel") # default 15-Minute

    with t_col3:
        st.write("")
        st.write("")
        btn_c1, btn_c2 = st.columns([1.1, 1.1])
        with btn_c1:
            term_force_sync = st.button("🔄 Sync Live", key="btn_crypto_sync_live", use_container_width=True, help="Force sync live market ticks from Delta Exchange")
        with btn_c2:
            tv_link_sym = "BINANCE:BTCUSDT" if crypto_symbol == "BTCUSD" else "BINANCE:ETHUSDT"
            st.link_button("🚀 TV Web", f"https://in.tradingview.com/chart/?symbol={tv_link_sym}", use_container_width=True)

    # Date Range Controls Row
    today_date = date.today()
    dr_c1, dr_c2, dr_c3 = st.columns([1.6, 1.2, 1.2])

    with dr_c1:
        date_range_preset = st.selectbox(
            "📅 Date Range Selection",
            [
                "All Available History",
                "Last 1 Year",
                "Last 6 Months",
                "Last 30 Days",
                "Last 7 Days",
                "Custom Date Range"
            ],
            index=0,
            key="crypto_date_preset"
        )

    if date_range_preset == "Last 1 Year":
        def_start = today_date - timedelta(days=365)
        def_end = today_date
    elif date_range_preset == "Last 6 Months":
        def_start = today_date - timedelta(days=180)
        def_end = today_date
    elif date_range_preset == "Last 30 Days":
        def_start = today_date - timedelta(days=30)
        def_end = today_date
    elif date_range_preset == "Last 7 Days":
        def_start = today_date - timedelta(days=7)
        def_end = today_date
    elif date_range_preset == "Custom Date Range":
        def_start = today_date - timedelta(days=60)
        def_end = today_date
    else:
        def_start = None
        def_end = None

    if "crypto_prev_preset" not in st.session_state:
        st.session_state["crypto_prev_preset"] = date_range_preset

    if date_range_preset != st.session_state["crypto_prev_preset"]:
        st.session_state["crypto_prev_preset"] = date_range_preset
        if def_start:
            st.session_state["crypto_from_date"] = def_start
        if def_end:
            st.session_state["crypto_to_date"] = def_end

    with dr_c2:
        from_date_input = st.date_input(
            "From Date",
            value=st.session_state.get("crypto_from_date", def_start if def_start else date(2023, 1, 1)),
            disabled=date_range_preset.startswith("All Available History"),
            key="crypto_from_date"
        )
    with dr_c3:
        to_date_input = st.date_input(
            "To Date",
            value=st.session_state.get("crypto_to_date", def_end if def_end else today_date),
            disabled=date_range_preset.startswith("All Available History"),
            key="crypto_to_date"
        )

    filter_start = str(from_date_input) if not date_range_preset.startswith("All Available History") else None
    filter_end = str(to_date_input) if not date_range_preset.startswith("All Available History") else None

    # Indicator Toggles Row (Identical to Equity Trading Terminal)
    with st.expander("🛠️ Indicator Settings & Overlay Toggles", expanded=True):
        ind_c1, ind_c2, ind_c3, ind_c4, ind_c5 = st.columns(5)
        with ind_c1:
            st.markdown("**Moving Averages**")
            term_ema = st.checkbox("EMAs (9, 20, 50, 200)", value=True, key="crypto_ema_chk")
            term_sma = st.checkbox("SMAs (20, 50, 200)", value=False, key="crypto_sma_chk")
        with ind_c2:
            st.markdown("**Volatility & Trend**")
            term_bb = st.checkbox("Bollinger Bands (20, 2)", value=False, key="crypto_bb_chk")
            term_st = st.checkbox("Supertrend (10, 3)", value=False, key="crypto_st_chk")
        with ind_c3:
            st.markdown("**Intraday & Levels**")
            term_vwap = st.checkbox("VWAP (Intraday)", value=True, key="crypto_vwap_chk")
            term_piv = st.checkbox("CPR / Weekly Pivots", value=True, key="crypto_piv_chk")
        with ind_c4:
            st.markdown("**Oscillators**")
            term_rsi = st.checkbox("RSI Panel (14)", value=True, key="crypto_rsi_chk")
            term_macd = st.checkbox("MACD Panel (12, 26, 9)", value=True, key="crypto_macd_chk")
        with ind_c5:
            st.markdown("**Volume & Momentum**")
            term_vol = st.checkbox("Volume + 20 MA", value=True, key="crypto_vol_chk")
            term_stoch = st.checkbox("Stochastic (14, 3, 3)", value=False, key="crypto_stoch_chk")

    # Map timeframe to Delta resolution
    res_code = delta_exchange_client.TIMEFRAME_TO_DELTA_RES.get(term_tf, "15m")
    is_intra = (res_code not in ["1d", "1w"])

    # Fetch live ticker for quote metrics banner
    success, ticker_data, t_note = delta_exchange_client.fetch_delta_ticker(crypto_symbol)
    if not ticker_data:
        ticker_data = crypto_store.get_crypto_ticker(crypto_symbol) or {}

    last_p = float(ticker_data.get("last_price", 0.0) or 0.0)
    chg_24 = float(ticker_data.get("change_24h", 0.0) or 0.0)
    high_24 = float(ticker_data.get("high_24h", last_p * 1.02) or 0.0)
    low_24 = float(ticker_data.get("low_24h", last_p * 0.98) or 0.0)
    vol_24 = float(ticker_data.get("volume_24h", 0.0) or 0.0)

    # Quote Metrics Banner
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    with m_c1:
        render_metric_card(f"DELTA:{crypto_symbol}", f"${last_p:,.2f}", f"24h: {chg_24:+.2f}%", "green" if chg_24 >= 0 else "red", styles)
    with m_c2:
        render_metric_card("24h High", f"${high_24:,.2f}", "24-Hour Peak", "blue", styles)
    with m_c3:
        render_metric_card("24h Low", f"${low_24:,.2f}", "24-Hour Floor", "orange", styles)
    with m_c4:
        render_metric_card("24h Volume", f"{vol_24:,.0f}", "Contracts Traded", "normal", styles)
    with m_c5:
        render_metric_card("Active Timeframe", f"{term_tf}", "🟢 24/7 LIVE (Delta)", "green", styles)

    # Sync Live Trigger
    if term_force_sync:
        with st.spinner(f"⚡ Syncing live market candles for {crypto_symbol} ({term_tf}) from Delta Exchange..."):
            delta_exchange_client.sync_all_delta_crypto_history(
                symbols=[crypto_symbol],
                timeframes=[res_code, "1d", "4h", "1h"]
            )
            st.success(f"Synced {crypto_symbol} from Delta Exchange!")

    # Fetch Candles for Terminal
    success_c, term_candles, c_note = delta_exchange_client.fetch_delta_candles(
        symbol=crypto_symbol,
        resolution=res_code,
        limit=1000
    )

    if term_candles.empty:
        # Fallback to local store
        term_candles = crypto_store.get_crypto_candles(crypto_symbol, timeframe=res_code, limit=1000)

    if term_candles.empty:
        st.warning(f"No candle data available for {crypto_symbol} ({term_tf}). Click '🔄 Sync Live' above to fetch data from Delta Exchange.")
        if c_note:
            st.caption(f"ℹ️ Delta Connection: {c_note}")
    else:
        term_df = term_candles.copy()
        if not isinstance(term_df.index, pd.DatetimeIndex):
            term_df.index = pd.to_datetime(term_df.index)

        # Apply Date Range Filtering
        term_tz = getattr(term_df.index, "tz", None)
        if filter_start:
            start_dt = pd.to_datetime(filter_start)
            if term_tz is not None and start_dt.tz is None:
                start_dt = start_dt.tz_localize(term_tz)
            elif term_tz is None and start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            term_df = term_df[term_df.index >= start_dt]
        if filter_end:
            end_dt = pd.to_datetime(f"{filter_end} 23:59:59")
            if term_tz is not None and end_dt.tz is None:
                end_dt = end_dt.tz_localize(term_tz)
            elif term_tz is None and end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)
            term_df = term_df[term_df.index <= end_dt]

        # Cap maximum visual candles to keep rendering snappy
        MAX_TERMINAL_BARS = 15000
        if len(term_df) > MAX_TERMINAL_BARS:
            st.caption(f"⚡ Showing latest **{MAX_TERMINAL_BARS:,}** {term_tf} candles from selected date range ({len(term_df):,} total bars available) for instant responsiveness.")
            term_df = term_df.tail(MAX_TERMINAL_BARS)

        # Terminal Engine Mode Selector
        engine_col1, engine_col2 = st.columns([3.0, 1.0])
        with engine_col2:
            chart_engine = st.radio(
                "Terminal Engine:",
                ["📈 Advanced Terminal", "🌐 Official TradingView"],
                horizontal=True,
                key="crypto_terminal_engine_sel"
            )

        if "Official TradingView" in chart_engine:
            tv_sym = f"BINANCE:{crypto_symbol}T" if crypto_symbol in ["BTCUSD", "ETHUSD"] else f"BINANCE:{crypto_symbol}"
            render_tradingview_cloud_widget(tv_sym, height=740)
        else:
            # Generate and Render Desktop-Grade Advanced Terminal HTML
            clean_chart_id = re.sub(r'[^a-zA-Z0-9_]', '_', f"adv_crypto_{crypto_symbol}_{res_code}")
            term_chart_html = generate_advanced_terminal_html(
                df=term_df,
                symbol=f"DELTA:{crypto_symbol}",
                timeframe_name=f"{term_tf} Crypto Terminal",
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
            st.components.v1.html(term_chart_html, height=760)

    # Historical Data Table & Inspection Expander
    with st.expander(f"📋 **Inspect Historical Candles & Export CSV ({crypto_symbol})**", expanded=False):
        df_table = crypto_store.get_crypto_candles(crypto_symbol, timeframe=res_code, limit=100)
        if not df_table.empty:
            df_display = df_table.copy()
            df_display["Change %"] = df_display["close"].pct_change() * 100.0
            df_display = df_display.sort_index(ascending=False)

            fmt_dict = {
                "open": "${:,.2f}",
                "high": "${:,.2f}",
                "low": "${:,.2f}",
                "close": "${:,.2f}",
                "volume": "{:,.0f}",
                "Change %": "{:+.2f}%"
            }
            st.dataframe(df_display.style.format(fmt_dict), use_container_width=True, height=320)

            csv_crypto = df_display.to_csv().encode("utf-8")
            st.download_button(
                label=f"📥 Download {crypto_symbol} ({term_tf}) CSV",
                data=csv_crypto,
                file_name=f"{crypto_symbol}_{res_code}_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
