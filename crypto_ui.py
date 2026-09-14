"""
crypto_ui.py
------------
Streamlit UI for the dedicated Crypto Trading Terminal.
Supports live & historical data from Delta Exchange strictly for BTCUSD and ETHUSD.
Uses the isolated data/crypto_market.duckdb database.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import crypto_store
import delta_exchange_client
import scanner
from tradingview_charts import generate_lightweight_chart_html, render_tradingview_cloud_widget


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


def compute_crypto_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators for crypto OHLCV dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    close = out["close"]

    out["EMA_9"] = scanner.calculate_ema(close, span=9)
    out["EMA_20"] = scanner.calculate_ema(close, span=20)
    out["EMA_50"] = scanner.calculate_ema(close, span=50)
    out["EMA_200"] = scanner.calculate_ema(close, span=200)
    out["RSI_14"] = scanner.calculate_rsi(close, span=14)

    # SuperTrend
    st_df = scanner.calculate_supertrend(out, period=10, multiplier=3.0)
    if "Supertrend" in st_df.columns:
        out["SuperTrend"] = st_df["Supertrend"]
    elif "SuperTrend" in st_df.columns:
        out["SuperTrend"] = st_df["SuperTrend"]

    return out


def render_crypto_page(theme: str = "dark"):
    """Main render function for the Crypto Terminal page."""
    styles = _get_theme_styles(theme)

    # Page Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div>
            <h2 style="margin: 0; color: {styles['text_primary']};">🪙 Delta Exchange Crypto Terminal</h2>
            <p style="margin: 0; color: {styles['text_secondary']}; font-size: 14px;">
                Live 24/7/365 market data &amp; historical analysis strictly for <b>BTCUSD</b> and <b>ETHUSD</b> (Separate Database: <code>crypto_market.duckdb</code>).
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.8, 1.8, 1.4, 1.2])

    with ctrl_col1:
        crypto_symbol = st.selectbox(
            "Select Crypto Asset:",
            options=["BTCUSD", "ETHUSD"],
            index=0 if st.session_state.get("crypto_active_symbol", "BTCUSD") == "BTCUSD" else 1,
            key="crypto_symbol_select_box"
        )
        st.session_state["crypto_active_symbol"] = crypto_symbol

    with ctrl_col2:
        timeframe_options = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
        active_tf = st.selectbox(
            "Candle Timeframe:",
            options=timeframe_options,
            index=3, # default 1h
            key="crypto_tf_select_box"
        )

    with ctrl_col3:
        chart_mode = st.radio(
            "Chart Engine:",
            ["📊 Local Data Chart", "🌐 Live TradingView"],
            horizontal=True,
            key="crypto_chart_mode"
        )

    with ctrl_col4:
        st.write("")
        st.write("")
        if st.button("⚡ Sync Delta API", type="primary", use_container_width=True, key="crypto_sync_delta_btn"):
            p_bar = st.progress(0)
            status_txt = st.empty()
            def _cb(cur, tot, msg):
                p_bar.progress(cur / tot)
                status_txt.caption(msg)
            with st.spinner("Connecting to Delta Exchange..."):
                sync_res = delta_exchange_client.sync_all_delta_crypto_history(
                    symbols=[crypto_symbol],
                    timeframes=[active_tf, "1d", "4h", "1h"],
                    progress_cb=_cb
                )
            p_bar.empty()
            status_txt.empty()
            st.success(f"Synced {crypto_symbol} from Delta Exchange!")
            st.rerun()

    # Fetch live ticker data
    success, ticker_data, t_note = delta_exchange_client.fetch_delta_ticker(crypto_symbol)
    if not ticker_data:
        ticker_data = crypto_store.get_crypto_ticker(crypto_symbol) or {}

    last_p = float(ticker_data.get("last_price", 0.0) or 0.0)
    mark_p = float(ticker_data.get("mark_price", last_p) or last_p)
    chg_24 = float(ticker_data.get("change_24h", 0.0) or 0.0)
    high_24 = float(ticker_data.get("high_24h", last_p * 1.02) or 0.0)
    low_24 = float(ticker_data.get("low_24h", last_p * 0.98) or 0.0)
    vol_24 = float(ticker_data.get("volume_24h", 0.0) or 0.0)

    # Ticker Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(
            f"{crypto_symbol} Price",
            f"${last_p:,.2f}",
            f"24h: {chg_24:+.2f}%",
            "green" if chg_24 >= 0 else "red",
            styles
        )
    with m2:
        render_metric_card("24h High", f"${high_24:,.2f}", "24-Hour Peak", "blue", styles)
    with m3:
        render_metric_card("24h Low", f"${low_24:,.2f}", "24-Hour Bottom", "orange", styles)
    with m4:
        render_metric_card("24h Volume", f"{vol_24:,.0f}", "Contracts Traded", "normal", styles)
    with m5:
        render_metric_card("Market Status", "🟢 24/7 LIVE", "Delta Exchange", "green", styles)

    # Render Chart
    if "Live TradingView" in chart_mode:
        tv_sym = f"BINANCE:{crypto_symbol}T" if crypto_symbol in ["BTCUSD", "ETHUSD"] else f"BINANCE:{crypto_symbol}"
        st.markdown(f"#### 🌐 Official TradingView Cloud Terminal — `{crypto_symbol}`")
        render_tradingview_cloud_widget(tv_sym, height=620)
    else:
        # Load local DuckDB data
        success_c, df_raw, c_note = delta_exchange_client.fetch_delta_candles(
            symbol=crypto_symbol,
            resolution=active_tf,
            limit=400
        )

        if df_raw.empty:
            st.warning(f"No candle data available for {crypto_symbol} ({active_tf}). Click '⚡ Sync Delta API' above.")
        else:
            df_ind = compute_crypto_indicators(df_raw)

            # Technical Status Pills
            last_row = df_ind.iloc[-1]
            rsi_val = float(last_row.get("RSI_14", 50.0))
            ema20_val = float(last_row.get("EMA_20", last_p))
            ema50_val = float(last_row.get("EMA_50", last_p))
            ema200_val = float(last_row.get("EMA_200", last_p))
            is_above_20ema = (last_p >= ema20_val)
            is_golden = (ema50_val >= ema200_val)

            t_cols = st.columns(4)
            with t_cols[0]:
                st.info(f"📊 **RSI (14):** `{rsi_val:.1f}` ({'Overbought ⚡' if rsi_val > 70 else ('Oversold 🔻' if rsi_val < 30 else 'Neutral ⚖️')})")
            with t_cols[1]:
                st.info(f"📈 **20 EMA:** `${ema20_val:,.2f}` ({'Bullish ✅' if is_above_20ema else 'Bearish ❌'})")
            with t_cols[2]:
                st.info(f"🌊 **Trend (50 vs 200 EMA):** `{'Golden Cross 🚀' if is_golden else 'Death Cross ⚠️'}`")
            with t_cols[3]:
                st.info(f"💾 **Data Source:** `Delta Exchange` ({len(df_raw)} candles)")

            # Render Lightweight Charts HTML
            is_intra = (active_tf in ["1m", "5m", "15m", "1h", "4h"])
            html_content = generate_lightweight_chart_html(
                df_ind,
                symbol=crypto_symbol,
                timeframe=active_tf,
                is_intraday=is_intra,
                theme=theme,
                chart_height=600
            )
            st.components.v1.html(html_content, height=640, scrolling=False)

    # Historical Data Table & Inspection Expander
    with st.expander(f"📋 **Inspect Historical Candles & Export CSV ({crypto_symbol})**", expanded=False):
        df_table = crypto_store.get_crypto_candles(crypto_symbol, timeframe=active_tf, limit=100)
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
                label=f"📥 Download {crypto_symbol} ({active_tf}) CSV",
                data=csv_crypto,
                file_name=f"{crypto_symbol}_{active_tf}_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
