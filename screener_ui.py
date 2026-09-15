"""
screener_ui.py
--------------
Streamlit UI for the Chartink-Style Stock Screener in Upstox Scanner.
Featuring:
1. Chartink Positional Swing Scanner (positional-scan-364) with strict Waterfall Model
   (Monthly Pass ➔ Weekly Pass ➔ Daily Pass ➔ 75-Min Pass). Strictly hides non-passing stocks.
2. Custom Condition Builder & Chartink Query Importer.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import config
import database
import parquet_loader
from screener_engine import (
    ScreenerClause,
    ScreenerConfig,
    INDICATOR_OPTIONS,
    OPERATOR_OPTIONS,
    run_screen,
    run_waterfall_scan,
    get_screener_preset,
    parse_chartink_query,
)
from chartink_ocr import parse_chartink_screenshot
import screener_snapshot


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
        "blue": "#3B82F6",
        "orange": "#F59E0B",
        "purple": "#8B5CF6",
    }


def render_metric_card(title: str, value: str, subtext: str = "", delta_color: str = "normal", styles: dict = None):
    bg = styles["card_bg"] if styles else "#1E222D"
    border = styles["card_border"] if styles else "#2A2E39"
    text_color = styles["text_primary"] if styles else "#FFFFFF"
    sub_color = styles["text_secondary"] if styles else "#94A3B8"

    val_color = text_color
    if delta_color == "green":
        val_color = "#10B981"
    elif delta_color == "red":
        val_color = "#EF4444"
    elif delta_color == "blue":
        val_color = "#3B82F6"
    elif delta_color == "purple":
        val_color = "#8B5CF6"

    html = f"""
    <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: {sub_color}; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 22px; font-weight: 800; color: {val_color}; margin: 3px 0 2px 0;">{value}</div>
        <div style="font-size: 11px; color: {sub_color};">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_target_equities(universe_choice: str) -> list:
    """Helper to fetch valid equity symbols for selected universe."""
    all_db_syms = database.get_all_symbols()
    parquet_syms = parquet_loader.get_parquet_symbols()
    all_equities = sorted(list(set(all_db_syms + parquet_syms)))
    all_equities = [s for s in all_equities if not s.startswith("0")]

    if universe_choice in ("Swing Stock", "Swing Stocks", "Swing Stocks (446)"):
        return [s for s in config.SWING_STOCK_SYMBOLS if s in all_equities]
    elif universe_choice == "Nifty 50":
        return [s for s in config.NIFTY_50_SYMBOLS if s in all_equities]
    elif universe_choice == "Nifty 100":
        return all_equities[:100]
    elif universe_choice == "Top 200 Liquid Equities":
        return all_equities[:200]
    elif universe_choice == "Nifty 500":
        return all_equities[:500]
    else:
        return all_equities


def render_screener_page(theme: str = "dark"):
    """Main render function for the Chartink-Style Stock Screener."""
    styles = _get_theme_styles(theme)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <h2 style="margin: 0; color: {styles['text_primary']};">🔍 Chartink Stock Screener & Waterfall Scanner</h2>
            <p style="margin: 0; color: {styles['text_secondary']}; font-size: 14px;">
                Filter NSE stocks using multi-timeframe Waterfall Alignment (Monthly ➔ Weekly ➔ Daily ➔ 75-Min) and custom criteria.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_waterfall, tab_custom, tab_crypto = st.tabs([
        "🌊 Chartink Positional Waterfall Scan (positional-scan-364)",
        "🛠️ Custom Condition Screener (Dynamic Rule Builder)",
        "🪙 Crypto Screener (Delta Exchange BTC & ETH)"
    ])

    # =========================================================================
    # TAB 1: CHARTINK POSITIONAL WATERFALL SCANNER (positional-scan-364)
    # =========================================================================
    with tab_waterfall:
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <b style="color: #38BDF8; font-size: 14px;">🌊 Positional Scan (Chartink #364) — Waterfall Filtering Rules</b>
                <a href="https://chartink.com/screener/positional-scan-364" target="_blank" style="font-size: 12px; color: #38BDF8; text-decoration: none;">🔗 Open on Chartink.com</a>
            </div>
            <div style="font-size: 12.5px; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                • <b>Universe & Price:</b> Cash Equities (₹100 to ₹10,000)<br/>
                • <b>Step 1 (Monthly Macro):</b> Close &gt; 5 EMA &gt; 20 EMA &nbsp;|&nbsp; 50 &lt; RSI(9) &lt; 88 &nbsp;|&nbsp; RSI(9) &ge; EMA(3) &ge; WMA(21)<br/>
                • <b>Step 2 (Weekly Trend):</b> Close &gt; 20 EMA &gt; 50 EMA &nbsp;|&nbsp; RSI(9) &gt; 50 &nbsp;|&nbsp; RSI(9) &ge; EMA(3) &ge; WMA(21)<br/>
                • <b>Step 3 (Daily Setup):</b> Open &le; 20 EMA and Close &ge; 20 EMA (Bounce / Cross) &nbsp;|&nbsp; 20 EMA &gt; 50 EMA &nbsp;|&nbsp; RSI(9) &gt; 50 &ge; EMA(3)<br/>
                • <b>Step 4 (75m Trigger):</b> Close &gt; 20 EMA &nbsp;|&nbsp; 5 EMA &ge; 20 EMA &nbsp;|&nbsp; RSI(9) &gt; 50 &ge; EMA(3)<br/>
                <i>Stocks that fail any tier are strictly excluded from subsequent stages.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        w_col1, w_col2, w_col3 = st.columns([1.5, 1.7, 1.8])
        with w_col1:
            wf_universe = st.selectbox(
                "Stock Universe to Scan:",
                ["Swing Stock", "Nifty 50", "Nifty 100", "Top 200 Liquid Equities", "Nifty 500", "All Database Equities"],
                index=0,
                key="wf_universe_select"
            )

        with w_col2:
            wf_date_mode = st.radio(
                "Scan Date Mode:",
                ["⚡ Latest Live Data", "📅 Historical Date (As-Of)"],
                index=1,
                horizontal=True,
                key="wf_date_mode_select"
            )

        with w_col3:
            wf_stage_filter = st.selectbox(
                "Waterfall Table Display Filter:",
                [
                    "🏆 Stage 4: Full Alignment Only (M + W + D + 75m) [Strict]",
                    "🚀 Stage 3+: Daily Bounce Pass (M + W + D)",
                    "🟢 Stage 2+: Weekly Trend Pass (M + W)",
                    "🟡 Stage 1+: Monthly Macro Pass (M Only)",
                    "🌊 All Waterfall Tiers (All Passing Stocks)"
                ],
                index=0,
                key="wf_stage_filter_select",
                help="Only stocks passing the selected tier are shown in the table. All failing stocks are hidden."
            )

        as_of_param = None
        as_of_time_param = None
        if "Historical" in wf_date_mode:
            if "pending_wf_as_of_date" in st.session_state:
                st.session_state["wf_as_of_date_input"] = st.session_state.pop("pending_wf_as_of_date")
            hist_c1, hist_c2, hist_c3 = st.columns([1.5, 1.4, 2.1])
            with hist_c1:
                # Default to latest database trading date or today
                latest_dt = None
                try:
                    import duckdb_store
                    stats = duckdb_store.get_db_stats()
                    if stats and stats.get("latest_date"):
                        latest_dt = pd.to_datetime(stats["latest_date"]).date()
                except Exception:
                    pass
                default_dt = latest_dt or datetime.today().date()
                selected_as_of = st.date_input(
                    "📅 Select Historical Scan Date:",
                    value=default_dt,
                    min_value=datetime(2020, 1, 1).date(),
                    max_value=datetime.today().date(),
                    key="wf_as_of_date_input"
                )
                as_of_param = selected_as_of.strftime("%Y-%m-%d")

            with hist_c2:
                selected_time_opt = st.selectbox(
                    "⏰ Select Candle / Time:",
                    [
                        "15:30 (Market Close / EOD)",
                        "14:15 (75m Candle 4 Close)",
                        "13:00 (75m Candle 3 Close)",
                        "11:45 (75m Candle 2 Close)",
                        "10:30 (75m Candle 1 Close)"
                    ],
                    index=0,
                    key="wf_as_of_time_input"
                )
                as_of_time_param = selected_time_opt[:5]

            with hist_c3:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 7px 12px; border-radius: 4px; margin-top: 14px; font-size: 11.5px; color: {styles['text_secondary']};">
                    <b>🎯 Backtest Active: {as_of_param} {as_of_time_param}</b><br/>
                    Candles sliced up to <b>{as_of_param} {as_of_time_param}</b>. Zero future data is leaked.
                    Forward returns to present day calculated automatically.
                </div>
                """, unsafe_allow_html=True)

            # Quick presets
            st.markdown("<div style='font-size: 11px; color: #94A3B8; margin-bottom: 4px;'>⚡ Quick Date Presets:</div>", unsafe_allow_html=True)
            q_cols = st.columns(5)
            with q_cols[0]:
                latest_label = f"📅 {default_dt.strftime('%d %b %Y')}" if default_dt else "📅 Latest"
                if st.button(latest_label, key="q_btn_latest_db", use_container_width=True):
                    st.session_state["pending_wf_as_of_date"] = default_dt
                    st.rerun()
            with q_cols[1]:
                if st.button("📅 01 Sep 2026", key="q_btn_1sep", use_container_width=True):
                    st.session_state["pending_wf_as_of_date"] = datetime(2026, 9, 1).date()
                    st.rerun()
            with q_cols[2]:
                if st.button("📅 14 Aug 2026", key="q_btn_14aug", use_container_width=True):
                    st.session_state["pending_wf_as_of_date"] = datetime(2026, 8, 14).date()
                    st.rerun()
            with q_cols[3]:
                if st.button("📅 01 Aug 2026", key="q_btn_1aug", use_container_width=True):
                    st.session_state["pending_wf_as_of_date"] = datetime(2026, 8, 1).date()
                    st.rerun()
            with q_cols[4]:
                if st.button("📅 01 Jul 2026", key="q_btn_1jul", use_container_width=True):
                    st.session_state["pending_wf_as_of_date"] = datetime(2026, 7, 1).date()
                    st.rerun()

        w_btn_col1, w_btn_col2 = st.columns([3.5, 1.5])
        with w_btn_col2:
            run_wf_btn = st.button("🚀 Run Waterfall Scan", type="primary", use_container_width=True, key="run_wf_scan_btn")

        # Trigger scan strictly on button press
        scan_id = f"{as_of_param}_{as_of_time_param}" if as_of_param else "latest"
        wf_cache_key = f"wf_res_{wf_universe}_{scan_id}"
        if run_wf_btn:
            target_syms = get_target_equities(wf_universe)
            p_bar = st.progress(0.0)
            p_txt = st.empty()
            wf_start_t = time.time()
            last_wf_update = [0.0]

            def _wf_progress(curr, total, sym):
                now = time.time()
                if curr == 1 or curr == total or (now - last_wf_update[0] >= 0.35):
                    last_wf_update[0] = now
                    pct = min(curr / max(total, 1), 1.0)
                    p_bar.progress(pct)
                    elapsed = max(now - wf_start_t, 0.001)
                    speed = curr / elapsed
                    rem_secs = (total - curr) / speed if speed > 0 else 0
                    eta_str = f"{int(rem_secs)}s" if rem_secs < 60 else f"{int(rem_secs // 60)}m {int(rem_secs % 60)}s"
                    p_txt.markdown(
                        f"⚡ **Scanning `{sym}`** — **{curr}/{total}** ({pct*100:.1f}%) | "
                        f"⏱️ **ETA:** ~{eta_str} remaining ({speed:.1f} stocks/sec)"
                    )

            wf_results = run_waterfall_scan(target_syms, as_of_date=as_of_param, as_of_time=as_of_time_param, progress_callback=_wf_progress)
            p_bar.empty()
            p_txt.empty()
            st.session_state[wf_cache_key] = wf_results

        # Render Waterfall Dashboard
        if wf_cache_key in st.session_state:
            wf_data = st.session_state[wf_cache_key]
            counts = wf_data.get("counts", {})
            as_of_label = f"{counts.get('as_of_date', 'Latest')} {counts.get('as_of_time', '')}".strip()

            # Funnel Progression Metric Cards
            fk1, fk2, fk3, fk4, fk5 = st.columns(5)
            with fk1:
                render_metric_card("Total Scanned", f"{counts.get('total', 0):,}", f"{wf_universe} ({as_of_label})", "normal", styles)
            with fk2:
                render_metric_card("Step 1: Monthly Pass", f"{counts.get('m_pass', 0):,}", "Macro Trend Bullish", "purple", styles)
            with fk3:
                render_metric_card("Step 2: Weekly Pass", f"{counts.get('w_pass', 0):,}", "Intermediate Trend Pass", "blue", styles)
            with fk4:
                render_metric_card("Step 3: Daily Pass", f"{counts.get('d_pass', 0):,}", "20 EMA Bounce Setup", "orange", styles)
            with fk5:
                render_metric_card("Step 4: 75m Trigger 🏆", f"{counts.get('q4_pass', 0):,}", "Full 4-TF Alignment", "green", styles)

            # Filter data according to user selection
            if "Stage 4" in wf_stage_filter:
                active_df = wf_data["stage_4_full"]
                tier_title = "🏆 Stage 4: Full Alignment (M + W + D + 75m) [Strictly 4/4 Passing]"
            elif "Stage 3" in wf_stage_filter:
                active_df = wf_data["stage_3_daily"]
                tier_title = "🚀 Stage 3: Monthly + Weekly + Daily Passing Stocks"
            elif "Stage 2" in wf_stage_filter:
                active_df = wf_data["stage_2_weekly"]
                tier_title = "🟢 Stage 2: Monthly + Weekly Passing Stocks"
            elif "Stage 1" in wf_stage_filter:
                active_df = wf_data["stage_1_monthly"]
                tier_title = "🟡 Stage 1: Monthly Macro Passing Stocks"
            else:
                active_df = wf_data["all_waterfall"]
                tier_title = "🌊 Complete Waterfall Alignment Table"

            st.markdown("---")
            st.markdown(f"#### {tier_title} ({len(active_df)} stocks) — [Scan As-Of: {as_of_label}]")

            if active_df.empty:
                st.warning(f"⚠️ No stocks qualified under '{wf_stage_filter}' as of {as_of_label}. All non-passing stocks have been hidden.")
                st.caption("Tip: Select 'Stage 3+' or 'Stage 2+' to see stocks in earlier stages of alignment on that date.")
            else:
                display_cols = ["Symbol", "Scan Date", "Scan Time", "LTP", "1D Return (%)"]
                if "Return Since Scan (%)" in active_df.columns and active_df["Return Since Scan (%)"].notna().any():
                    display_cols.extend(["Return Since Scan (%)", "Latest Price"])
                display_cols.extend([
                    "Volume", "Waterfall Stage", "Monthly", "Weekly", "Daily", "75-Min", "15-Min",
                    "M_RSI", "W_RSI", "D_RSI", "75m_RSI"
                ])
                show_df = active_df[[c for c in display_cols if c in active_df.columns]].copy()

                fmt_dict = {
                    "LTP": "₹{:,.2f}",
                    "Latest Price": "₹{:,.2f}",
                    "1D Return (%)": "{:+.2f}%",
                    "Return Since Scan (%)": "{:+.2f}%",
                    "Volume": "{:,}",
                    "M_RSI": "{:.1f}",
                    "W_RSI": "{:.1f}",
                    "D_RSI": "{:.1f}",
                    "75m_RSI": "{:.1f}"
                }
                return_subsets = [c for c in ["1D Return (%)", "Return Since Scan (%)"] if c in show_df.columns]

                # Style dataframe: green/red on return, green badge on pass
                st_styled = show_df.style.format(fmt_dict)
                if return_subsets:
                    st_styled = st_styled.map(
                        lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=return_subsets
                    )
                st_styled = st_styled.map(
                    lambda v: "background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold;" if v == "✅ PASS" else ("color: #94A3B8;" if v == "❌ FAIL" else ""),
                    subset=[c for c in ["Monthly", "Weekly", "Daily", "75-Min", "15-Min"] if c in show_df.columns]
                )


                st.dataframe(
                    st_styled,
                    use_container_width=True,
                    height=min(500, max(260, len(show_df) * 38))
                )

                # Quick Chart Launcher & CSV Download
                wf_act1, wf_act2, wf_act3 = st.columns([1.5, 1.5, 2.0])
                with wf_act1:
                    wf_sel_sym = st.selectbox("Select Stock to View:", options=show_df["Symbol"].tolist(), key="wf_launch_sym")
                    if st.button("📊 Open in Quad-Chart View", key="wf_open_quad", use_container_width=True):
                        st.session_state["quad_sym_select"] = wf_sel_sym
                        st.session_state["app_active_nav_page"] = "📊 Quad-Chart View"
                        st.rerun()
                with wf_act2:
                    st.write("")
                    st.write("")
                    if st.button("📈 Open in Trading Terminal", key="wf_open_term", use_container_width=True):
                        st.session_state["term_symbol_select"] = wf_sel_sym
                        st.session_state["app_active_nav_page"] = "📈 Trading Terminal"
                        st.rerun()
                with wf_act3:
                    st.write("")
                    st.write("")
                    csv_wf = show_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Waterfall Scan CSV",
                        data=csv_wf,
                        file_name=f"waterfall_scan_364_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="wf_csv_dl_btn"
                    )
        else:
            st.markdown("---")
            st.info(f"👆 Ready to scan **{wf_universe}**! Select your Date & Universe parameters above, then click **'🚀 Run Waterfall Scan'** to start.")

    # =========================================================================
    # TAB 2: CUSTOM CONDITION SCREENER & BUILDER
    # =========================================================================
    with tab_custom:
        st.markdown("#### 🛠️ Custom Condition Screener (Dynamic Rule Builder)")

        # Safely apply pending widget state updates BEFORE widgets are instantiated
        if "pending_preset_select" in st.session_state:
            st.session_state["scr_preset_select"] = st.session_state.pop("pending_preset_select")
        if "pending_logic_select" in st.session_state:
            st.session_state["scr_logic_select"] = st.session_state.pop("pending_logic_select")

        p_col1, p_col2, p_col3, p_col4 = st.columns([1.8, 1.2, 1.2, 1.0])

        with p_col1:
            preset_names = [
                "⚡ Intraday Scan (Chartink 19122704 - 75m Multi-TF Breakdown)",
                "🚀 RSI Momentum Surge (RSI > 60 & Close > 20 EMA)",
                "🏔️ 52-Week / Multi-Year High Breakout",
                "🏹 SuperTrend Fresh Bullish Reversal",
                "🌊 Triple EMA Bullish Stack (9 > 20 > 50)",
                "💥 Volume Shocker & Price Breakout",
                "⚡ Hilega Milega Bullish Alignment (NK Sir)",
                "🎯 Pullback to 20 EMA (Dip Buying Setup)",
                "🔻 Oversold Bounce Setup (RSI < 30)",
                "🛠️ Custom Screener Builder"
            ]
            selected_preset = st.selectbox("Select Screener Preset:", options=preset_names, index=0, key="scr_preset_select")

        with p_col2:
            universe_choice = st.selectbox(
                "Stock Universe:",
                ["Swing Stock", "Nifty 50", "Nifty 100", "Top 200 Liquid Equities", "Nifty 500", "All Database Equities"],
                index=0,
                key="scr_universe_select"
            )

        with p_col3:
            cust_date_mode = st.radio(
                "Scan Date:",
                ["⚡ Latest Live", "📅 Historical Date"],
                index=1,
                horizontal=True,
                key="scr_cust_date_mode"
            )

        with p_col4:
            filter_logic = st.radio(
                "Match Logic:",
                ["ALL (AND)", "ANY (OR)"],
                index=0,
                horizontal=True,
                key="scr_logic_select"
            )
            logic_val = "ALL" if "ALL" in filter_logic else "ANY"

        cust_as_of_param = None
        cust_as_of_time_param = None
        if "Historical" in cust_date_mode:
            c_d1, c_d2, c_d3 = st.columns([1.5, 1.4, 2.1])
            with c_d1:
                c_latest_dt = None
                try:
                    import duckdb_store
                    stats = duckdb_store.get_db_stats()
                    if stats and stats.get("latest_date"):
                        c_latest_dt = pd.to_datetime(stats["latest_date"]).date()
                except Exception:
                    pass
                c_sel_date = st.date_input(
                    "📅 Select Scan Date:",
                    value=c_latest_dt or datetime.today().date(),
                    min_value=datetime(2020, 1, 1).date(),
                    max_value=datetime.today().date(),
                    key="scr_cust_as_of_date_input"
                )
                cust_as_of_param = c_sel_date.strftime("%Y-%m-%d")
            with c_d2:
                c_sel_time = st.selectbox(
                    "⏰ Select Time:",
                    [
                        "15:30 (Market Close / EOD)",
                        "14:15 (75m Candle 4 Close)",
                        "13:00 (75m Candle 3 Close)",
                        "11:45 (75m Candle 2 Close)",
                        "10:30 (75m Candle 1 Close)"
                    ],
                    index=0,
                    key="scr_cust_as_of_time_input"
                )
                cust_as_of_time_param = c_sel_time[:5]
            with c_d3:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 7px 12px; border-radius: 4px; margin-top: 14px; font-size: 11.5px; color: {styles['text_secondary']};">
                    <b>🎯 Backtest Active: {cust_as_of_param} {cust_as_of_time_param}</b><br/>Slicing candles up to <b>{cust_as_of_param} {cust_as_of_time_param}</b>.
                </div>
                """, unsafe_allow_html=True)

        # Synchronize preset into session state clauses
        if "last_loaded_preset" not in st.session_state or st.session_state["last_loaded_preset"] != selected_preset:
            st.session_state["last_loaded_preset"] = selected_preset
            preset_cfg = get_screener_preset(selected_preset)
            st.session_state["screener_clauses"] = [
                {
                    "timeframe": c.timeframe,
                    "lhs": c.lhs,
                    "operator": c.operator,
                    "rhs_type": c.rhs_type,
                    "rhs_indicator": c.rhs_indicator,
                    "rhs_value": c.rhs_value,
                    "multiplier": c.multiplier
                }
                for c in preset_cfg.clauses
            ]
            st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1

        # 1. Chartink Scanner Screenshot-to-Scanner Uploader (OCR Vision Engine)
        with st.expander("📸 **Upload Chartink Scanner Screenshot (Auto-Create Scanner)**", expanded=True):
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3B82F6; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; color: {styles['text_secondary']};">
                📸 <b>Screenshot-to-Scanner:</b> Upload any screenshot of a Chartink scanner (PNG, JPG, WEBP).
                The AI OCR engine will automatically extract every timeframe, indicator, and operator, populate the rules, and execute the scan!
            </div>
            """, unsafe_allow_html=True)

            scr_col1, scr_col2 = st.columns([1.5, 2.5])
            with scr_col1:
                uploaded_scr = st.file_uploader(
                    "Upload Scanner Screenshot (PNG, JPG, WEBP):",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="chartink_scanner_screenshot_uploader",
                    help="Upload a screenshot of the Chartink filter rules (e.g. from Chartink.com)"
                )
                if uploaded_scr is not None:
                    st.image(uploaded_scr, caption="Uploaded Chartink Screenshot", use_container_width=True)

            with scr_col2:
                if uploaded_scr is not None:
                    scr_cache_key = f"scr_ocr_{uploaded_scr.name}_{uploaded_scr.size}"
                    if scr_cache_key not in st.session_state:
                        with st.spinner("🤖 Scanning and extracting Chartink conditions via OCR..."):
                            st.session_state[scr_cache_key] = parse_chartink_screenshot(uploaded_scr)

                    ocr_data = st.session_state[scr_cache_key]

                    if ocr_data["success"] and (ocr_data["clauses"] or ocr_data["cleaned_lines"]):
                        st.success(f"🎉 Successfully extracted **{len(ocr_data['clauses'])} condition(s)** with **{ocr_data['logic']}** match logic!")
                        detected_text = "\n".join(ocr_data["cleaned_lines"]) if ocr_data["cleaned_lines"] else ocr_data.get("raw_text", "")
                        st.markdown("**Detected Scanner Rules (Inspect or Tweak below):**")
                        user_ocr_rules = st.text_area(
                            "Detected Rules from Screenshot:",
                            value=detected_text,
                            height=120,
                            key=f"ocr_editor_{uploaded_scr.name}_{uploaded_scr.size}",
                            help="Review or fine-tune detected rules. Any edits will be used when you click 'Create & Run Scanner'."
                        )

                        btn_ocr_1, btn_ocr_2 = st.columns(2)
                        with btn_ocr_1:
                            if st.button("⚡ Create & Run Scanner Now", type="primary", use_container_width=True, key="btn_create_and_run_ocr"):
                                parsed_user_clauses = parse_chartink_query(user_ocr_rules) if user_ocr_rules.strip() else ocr_data["clauses"]
                                target_clauses = parsed_user_clauses if parsed_user_clauses else ocr_data["clauses"]
                                if target_clauses:
                                    st.session_state["screener_clauses"] = [
                                        {
                                            "timeframe": c.timeframe,
                                            "lhs": c.lhs,
                                            "operator": c.operator,
                                            "rhs_type": c.rhs_type,
                                            "rhs_indicator": c.rhs_indicator,
                                            "rhs_value": c.rhs_value,
                                            "multiplier": c.multiplier
                                        }
                                        for c in target_clauses
                                    ]
                                    st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
                                    st.session_state["last_loaded_preset"] = "🛠️ Custom Screener Builder"
                                    st.session_state["pending_preset_select"] = "🛠️ Custom Screener Builder"
                                    if ocr_data["logic"] == "ANY":
                                        st.session_state["pending_logic_select"] = "ANY (OR)"
                                    else:
                                        st.session_state["pending_logic_select"] = "ALL (AND)"
                                    st.session_state["auto_trigger_custom_scan"] = True
                                    st.rerun()

                        with btn_ocr_2:
                            if st.button("📝 Load into Rule Builder (Edit)", type="secondary", use_container_width=True, key="btn_load_ocr_builder"):
                                parsed_user_clauses = parse_chartink_query(user_ocr_rules) if user_ocr_rules.strip() else ocr_data["clauses"]
                                target_clauses = parsed_user_clauses if parsed_user_clauses else ocr_data["clauses"]
                                if target_clauses:
                                    st.session_state["screener_clauses"] = [
                                        {
                                            "timeframe": c.timeframe,
                                            "lhs": c.lhs,
                                            "operator": c.operator,
                                            "rhs_type": c.rhs_type,
                                            "rhs_indicator": c.rhs_indicator,
                                            "rhs_value": c.rhs_value,
                                            "multiplier": c.multiplier
                                        }
                                        for c in target_clauses
                                    ]
                                    st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
                                    st.session_state["last_loaded_preset"] = "🛠️ Custom Screener Builder"
                                    st.session_state["pending_preset_select"] = "🛠️ Custom Screener Builder"
                                    if ocr_data["logic"] == "ANY":
                                        st.session_state["pending_logic_select"] = "ANY (OR)"
                                    else:
                                        st.session_state["pending_logic_select"] = "ALL (AND)"
                                    st.rerun()

                    elif ocr_data.get("error"):
                        st.error(f"⚠️ {ocr_data['error']}")
                        st.info("💡 **Tip:** If Tesseract is not installed on your server, run `sudo apt-get install -y tesseract-ocr` or paste your rules below.")
                        manual_rules = st.text_area(
                            "Type / Paste Chartink Rules directly:",
                            placeholder="Daily Close > Daily 20 EMA\nDaily RSI(14) > 60",
                            height=90,
                            key="manual_rules_fallback"
                        )
                        if st.button("⚡ Convert & Run Scanner", type="primary", use_container_width=True, key="btn_run_fallback"):
                            parsed = parse_chartink_query(manual_rules)
                            if parsed:
                                st.session_state["screener_clauses"] = [
                                    {
                                        "timeframe": c.timeframe,
                                        "lhs": c.lhs,
                                        "operator": c.operator,
                                        "rhs_type": c.rhs_type,
                                        "rhs_indicator": c.rhs_indicator,
                                        "rhs_value": c.rhs_value,
                                        "multiplier": c.multiplier
                                    }
                                    for c in parsed
                                ]
                                st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
                                st.session_state["last_loaded_preset"] = "🛠️ Custom Screener Builder"
                                st.session_state["pending_preset_select"] = "🛠️ Custom Screener Builder"
                                st.session_state["auto_trigger_custom_scan"] = True
                                st.rerun()
                    else:
                        st.warning("Could not identify specific conditions from the screenshot. Please check the image clarity or paste the query text below.")
                else:
                    st.info("""
                    **Quick Steps:**
                    1. Capture a screenshot of any Chartink scanner filters (e.g. `Cmd + Shift + 4` on Mac or Snipping Tool).
                    2. Upload or drag & drop the image into the box on the left.
                    3. Click **'Create & Run Scanner Now'** to scan across 3,000+ NSE equities immediately!
                    """)

        # 2. Chartink Link & Text Query Importer Expander
        with st.expander("📋 **Chartink Text Query / Link Importer (Paste & Convert)**", expanded=False):
            st.caption("Paste any Chartink condition string or URL below to automatically load its rules into the filter builder.")
            c_query_col, c_btn_col = st.columns([3.5, 1.0])
            with c_query_col:
                pasted_query = st.text_area(
                    "Paste Chartink Rule / Query:",
                    placeholder="Example:\nDaily Close > Daily 20 EMA\nDaily RSI(14) > 60\nWeekly Close > Weekly 200 EMA",
                    height=70,
                    key="scr_chartink_query_input",
                    label_visibility="collapsed"
                )
            with c_btn_col:
                st.write("")
                st.write("")
                if st.button("⚡ Parse & Load", type="secondary", use_container_width=True, key="scr_parse_chartink_btn"):
                    if pasted_query.strip():
                        parsed_clauses = parse_chartink_query(pasted_query)
                        if parsed_clauses:
                            st.session_state["screener_clauses"] = [
                                {
                                    "timeframe": c.timeframe,
                                    "lhs": c.lhs,
                                    "operator": c.operator,
                                    "rhs_type": c.rhs_type,
                                    "rhs_indicator": c.rhs_indicator,
                                    "rhs_value": c.rhs_value,
                                    "multiplier": c.multiplier
                                }
                                for c in parsed_clauses
                            ]
                            st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
                            st.session_state["last_loaded_preset"] = "🛠️ Custom Screener Builder"
                            st.session_state["pending_preset_select"] = "🛠️ Custom Screener Builder"
                            st.success(f"✅ Successfully converted {len(parsed_clauses)} Chartink rules!")
                            st.rerun()


        # Condition Builder rows
        clauses = st.session_state.get("screener_clauses", [])
        c_ver = st.session_state.get("clauses_version", 0)
        to_delete_idx = None

        for i, c in enumerate(clauses):
            row_cols = st.columns([1.2, 1.6, 1.4, 1.3, 1.6, 0.6])

            # Dynamically include any custom indicator in options
            lhs_opts = list(INDICATOR_OPTIONS)
            if c.get("lhs") and c["lhs"] not in lhs_opts:
                lhs_opts.append(c["lhs"])

            rhs_opts = list(INDICATOR_OPTIONS)
            if c.get("rhs_indicator") and c["rhs_indicator"] not in rhs_opts:
                rhs_opts.append(c["rhs_indicator"])

            with row_cols[0]:
                c["timeframe"] = st.selectbox(
                    f"TF #{i+1}",
                    ["Daily", "Weekly", "Monthly", "75-Min", "15-Min"],
                    index=["Daily", "Weekly", "Monthly", "75-Min", "15-Min"].index(c["timeframe"]) if c["timeframe"] in ["Daily", "Weekly", "Monthly", "75-Min", "15-Min"] else 0,
                    key=f"c_tf_{c_ver}_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[1]:
                c["lhs"] = st.selectbox(
                    f"LHS #{i+1}",
                    lhs_opts,
                    index=lhs_opts.index(c["lhs"]) if c["lhs"] in lhs_opts else 0,
                    key=f"c_lhs_{c_ver}_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[2]:
                c["operator"] = st.selectbox(
                    f"Op #{i+1}",
                    OPERATOR_OPTIONS,
                    index=OPERATOR_OPTIONS.index(c["operator"]) if c["operator"] in OPERATOR_OPTIONS else 0,
                    key=f"c_op_{c_ver}_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[3]:
                c["rhs_type"] = st.selectbox(
                    f"Type #{i+1}",
                    ["Indicator", "Number"],
                    index=0 if c["rhs_type"] == "Indicator" else 1,
                    key=f"c_rtype_{c_ver}_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[4]:
                if c["rhs_type"] == "Indicator":
                    c["rhs_indicator"] = st.selectbox(
                        f"RHS Ind #{i+1}",
                        rhs_opts,
                        index=rhs_opts.index(c["rhs_indicator"]) if c["rhs_indicator"] in rhs_opts else 11,
                        key=f"c_rind_{c_ver}_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    c["rhs_value"] = float(st.number_input(
                        f"Value #{i+1}",
                        value=float(c.get("rhs_value", 0.0)),
                        step=1.0,
                        key=f"c_rval_{c_ver}_{i}",
                        label_visibility="collapsed"
                    ))
            with row_cols[5]:
                if st.button("🗑️", key=f"del_c_{c_ver}_{i}", help="Delete this condition"):
                    to_delete_idx = i

        if to_delete_idx is not None:
            clauses.pop(to_delete_idx)
            st.session_state["screener_clauses"] = clauses
            st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
            st.rerun()

        # ── DuckDB Vectorized SQL Snapshot Engine Status & Rebuild ──────────
        meta = screener_snapshot.get_snapshot_metadata()
        has_snap = screener_snapshot.has_screener_snapshot()

        snap_c1, snap_c2 = st.columns([3.2, 1.2])
        with snap_c1:
            if has_snap:
                st.markdown(
                    f"<div style='font-size: 0.85rem; color: #10B981; padding: 4px 0;'>"
                    f"⚡ <b>DuckDB Vectorized SQL Engine Active (Sub-20ms instant scan)</b> | "
                    f"Snapshot: <b>{meta.get('total_symbols', 0)} stocks</b> | "
                    f"Updated: <code>{meta.get('last_updated', 'Active')}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("ℹ️ Snapshot table not yet built. First live scan will build it automatically.")
        with snap_c2:
            if st.button("🔄 Rebuild Snapshot", key="scr_rebuild_snap_btn", use_container_width=True, help="Recomputes all 94 technical indicators across all stocks into DuckDB"):
                with st.spinner("Materializing wide indicators table in DuckDB..."):
                    res = screener_snapshot.build_screener_snapshot()
                    st.success(f"✅ Snapshot ready! {res.get('total_symbols', 0)} stocks in {res.get('elapsed_seconds', 0)}s.")
                    st.rerun()

        add_col, scan_col = st.columns([1.5, 2.5])
        with add_col:
            if st.button("➕ Add Condition", key="scr_add_cond_btn", use_container_width=True):
                clauses.append({
                    "timeframe": "Daily",
                    "lhs": "Close",
                    "operator": ">",
                    "rhs_type": "Indicator",
                    "rhs_indicator": "EMA_20",
                    "rhs_value": 0.0,
                    "multiplier": 1.0
                })
                st.session_state["screener_clauses"] = clauses
                st.session_state["clauses_version"] = st.session_state.get("clauses_version", 0) + 1
                st.rerun()

        with scan_col:
            run_custom_scan = st.button("🔍 Run Custom Screener Scan", type="primary", use_container_width=True, key="scr_run_custom_scan_btn")

        auto_run = st.session_state.pop("auto_trigger_custom_scan", False)
        if run_custom_scan or auto_run:
            target_symbols = get_target_equities(universe_choice)
            screener_clauses = [
                ScreenerClause(
                    timeframe=c["timeframe"],
                    lhs=c["lhs"],
                    operator=c["operator"],
                    rhs_type=c["rhs_type"],
                    rhs_indicator=c.get("rhs_indicator", "EMA_20"),
                    rhs_value=float(c.get("rhs_value", 0.0)),
                    multiplier=float(c.get("multiplier", 1.0))
                )
                for c in clauses
            ]
            cfg = ScreenerConfig(name=selected_preset, logic=logic_val, universe=universe_choice, clauses=screener_clauses)
            p_bar = st.progress(0.0)
            p_txt = st.empty()
            scr_start_t = time.time()
            last_scr_update = [0.0]

            def _scr_progress(curr, total, sym):
                now = time.time()
                if curr == 1 or curr == total or (now - last_scr_update[0] >= 0.35):
                    last_scr_update[0] = now
                    pct = min(curr / max(total, 1), 1.0)
                    p_bar.progress(pct)
                    elapsed = max(now - scr_start_t, 0.001)
                    speed = curr / elapsed
                    rem_secs = (total - curr) / speed if speed > 0 else 0
                    eta_str = f"{int(rem_secs)}s" if rem_secs < 60 else f"{int(rem_secs // 60)}m {int(rem_secs % 60)}s"
                    p_txt.markdown(
                        f"🔍 **Evaluating `{sym}`** — **{curr}/{total}** ({pct*100:.1f}%) | "
                        f"⏱️ **ETA:** ~{eta_str} remaining ({speed:.1f} stocks/sec)"
                    )

            res_df = run_screen(target_symbols, cfg, as_of_date=cust_as_of_param, as_of_time=cust_as_of_time_param, progress_callback=_scr_progress)
            p_bar.empty()
            p_txt.empty()
            st.session_state["last_screener_results"] = res_df
            st.session_state["last_screener_total_scanned"] = len(target_symbols)

        if "last_screener_results" in st.session_state:
            res_df = st.session_state["last_screener_results"]
            total_scanned = st.session_state.get("last_screener_total_scanned", 0)

            st.markdown("---")
            st.markdown("### 📊 Custom Screener Results")

            if res_df.empty:
                st.warning("⚠️ No stocks matched all the specified conditions in the selected universe.")
                diag = getattr(res_df, "attrs", {}).get("diag", {})
                if diag and diag.get("clause_stats"):
                    st.markdown(f"#### 🔍 Filter Diagnostic Breakdown ({diag.get('evaluated_stocks', 0)} Stocks Evaluated — Match Logic: `{diag.get('logic', 'ALL')}`)")
                    diag_data = []
                    for cs in diag["clause_stats"]:
                        diag_data.append({
                            "Rule #": f"Rule #{cs['index']}",
                            "Condition": cs["desc"],
                            "Stocks Passed": f"{cs['passed_count']} / {diag['evaluated_stocks']}",
                            "Pass Rate": f"{cs['passed_pct']}%"
                        })
                    st.table(pd.DataFrame(diag_data))
                    zero_clauses = [cs for cs in diag["clause_stats"] if cs["passed_count"] == 0]
                    if zero_clauses:
                        rule_names = ", ".join(f"Rule #{z['index']}" for z in zero_clauses)
                        st.info(f"💡 **Bottleneck:** {rule_names} returned 0 matching stocks. Consider adjusting its threshold or operator.")
            else:
                diag = getattr(res_df, "attrs", {}).get("diag", {})
                if diag and diag.get("clause_stats"):
                    with st.expander("🔍 View Per-Rule Filtering Statistics", expanded=False):
                        diag_data = []
                        for cs in diag["clause_stats"]:
                            diag_data.append({
                                "Rule #": f"Rule #{cs['index']}",
                                "Condition": cs["desc"],
                                "Stocks Passed": f"{cs['passed_count']} / {diag['evaluated_stocks']}",
                                "Pass Rate": f"{cs['passed_pct']}%"
                            })
                        st.table(pd.DataFrame(diag_data))

                m1, m2, m3, m4 = st.columns(4)

                match_count = len(res_df)
                gainers_count = len(res_df[res_df["Change_%"] > 0]) if "Change_%" in res_df.columns else 0
                avg_return = res_df["Change_%"].mean() if "Change_%" in res_df.columns else 0.0
                top_stock = res_df.iloc[0]["Symbol"] if not res_df.empty else "N/A"

                with m1:
                    render_metric_card("Matching Stocks", f"{match_count:,}", f"{(match_count/max(total_scanned,1)*100):.1f}% of universe", "green", styles)
                with m2:
                    render_metric_card("Gainers vs Losers", f"{gainers_count} 🟢 / {match_count - gainers_count} 🔴", "Today's direction", "blue", styles)
                with m3:
                    render_metric_card("Average Return", f"{avg_return:+.2f}%", "Across matched stocks", "green" if avg_return >= 0 else "red", styles)
                with m4:
                    render_metric_card("Top Gainer", f"{top_stock}", f"{res_df.iloc[0]['Change_%']:+.2f}%" if (not res_df.empty and 'Change_%' in res_df.columns) else "", "normal", styles)

                display_df = res_df.rename(columns={
                    "Change_%": "1D Return (%)",
                    "LTP": "LTP (₹)",
                    "Clauses_Passed": "Rules Passed",
                    "Scan_Date": "Scan Date",
                    "Scan_Time": "Scan Time",
                    "Return_Since_Scan_%": "Return Since Scan (%)",
                    "Latest_Close": "Latest Price"
                })
                fmt_custom = {
                    "LTP (₹)": "₹{:,.2f}",
                    "1D Return (%)": "{:+.2f}%",
                    "Return Since Scan (%)": "{:+.2f}%",
                    "Latest Price": "₹{:,.2f}",
                    "Volume": "{:,}"
                }
                color_subsets = [c for c in ["1D Return (%)", "Return Since Scan (%)"] if c in display_df.columns]
                st_custom_styled = display_df.style.format(fmt_custom)
                if color_subsets:
                    st_custom_styled = st_custom_styled.map(
                        lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=color_subsets
                    )
                st.dataframe(
                    st_custom_styled,
                    use_container_width=True,
                    height=min(500, max(240, len(display_df) * 36))
                )

    # =========================================================================
    # TAB 3: CRYPTO SCREENER (DELTA EXCHANGE BTC & ETH)
    # =========================================================================
    with tab_crypto:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.08); border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <b style="color: #F59E0B; font-size: 14px;">🪙 Delta Exchange Crypto Screener — BTCUSD &amp; ETHUSD</b>
                <span style="font-size: 12px; color: #10B981; font-weight: bold;">🟢 24/7 LIVE MARKET</span>
            </div>
            <div style="font-size: 12.5px; color: {styles['text_secondary']}; margin-top: 6px; line-height: 1.5;">
                • <b>Separate Crypto Database:</b> Operating independently on <code>data/crypto_market.duckdb</code>.<br/>
                • <b>Universe:</b> Delta Exchange Perpetuals (<b>BTCUSD</b> and <b>ETHUSD</b>).<br/>
                • <b>Multi-Timeframe Engine:</b> Screens 20/50/200 EMAs, RSI(14), and SuperTrend across Daily, 4h, 1h, and 15m.
            </div>
        </div>
        """, unsafe_allow_html=True)

        cr_col1, cr_col2, cr_col3 = st.columns([2.2, 1.2, 1.0])
        with cr_col1:
            crypto_preset = st.selectbox(
                "Select Crypto Screener Strategy:",
                [
                    "🚀 Strong Bullish Trend (Daily Close > 20 EMA & RSI > 50)",
                    "🌊 Multi-Timeframe Alignment (Daily & 4h & 1h Above 20 EMA)",
                    "⚡ Golden Cross (50 EMA > 200 EMA on Daily)",
                    "🎯 20 EMA Pullback Setup (Daily Close near 20 EMA & RSI > 45)",
                    "🏹 SuperTrend Bullish (Daily Supertrend Positive)",
                    "🔻 Oversold Dip Hunter (RSI < 40 on Daily or 4h)",
                    "📊 Full Multi-Timeframe Status Table (All Metrics)"
                ],
                index=0,
                key="crypto_screener_preset_select"
            )

        with cr_col2:
            st.write("")
            st.write("")
            refresh_crypto_screen = st.button("🔄 Refresh Crypto Scan", type="primary", use_container_width=True, key="btn_refresh_crypto_screen")

        with cr_col3:
            st.write("")
            st.write("")
            if st.button("⚡ Sync Delta Data", use_container_width=True, key="btn_sync_delta_from_screener"):
                import delta_exchange_client
                with st.spinner("Syncing latest BTC & ETH candles from Delta Exchange..."):
                    delta_exchange_client.sync_all_delta_crypto_history()
                st.success("Delta Exchange candles updated!")
                st.rerun()

        # Perform Crypto Screening Evaluation
        import crypto_store
        import scanner

        crypto_symbols = ["BTCUSD", "ETHUSD"]
        results_crypto = []

        for c_sym in crypto_symbols:
            t_info = crypto_store.get_crypto_ticker(c_sym) or {}
            df_d = crypto_store.get_crypto_candles(c_sym, "1d", limit=100)
            df_4h = crypto_store.get_crypto_candles(c_sym, "4h", limit=100)
            df_1h = crypto_store.get_crypto_candles(c_sym, "1h", limit=100)

            last_p = float(t_info.get("last_price", 0.0) or (df_d.iloc[-1]["close"] if not df_d.empty else 0.0))
            chg_24 = float(t_info.get("change_24h", 0.0) or 0.0)

            # Daily indicators
            d_rsi = 50.0
            d_20ema = last_p
            d_50ema = last_p
            d_200ema = last_p
            d_st_bull = True
            if not df_d.empty:
                c_d = df_d["close"]
                d_rsi = float(scanner.calculate_rsi(c_d, 14).iloc[-1])
                d_20ema = float(scanner.calculate_ema(c_d, 20).iloc[-1])
                d_50ema = float(scanner.calculate_ema(c_d, 50).iloc[-1])
                d_200ema = float(scanner.calculate_ema(c_d, 200).iloc[-1])
                st_d = scanner.calculate_supertrend(df_d, 10, 3.0)
                if "Trend_Direction" in st_d.columns:
                    d_st_bull = bool(st_d["Trend_Direction"].iloc[-1] == 1)

            # 4h indicators
            h4_rsi = 50.0
            h4_20ema = last_p
            if not df_4h.empty:
                c_4h = df_4h["close"]
                h4_rsi = float(scanner.calculate_rsi(c_4h, 14).iloc[-1])
                h4_20ema = float(scanner.calculate_ema(c_4h, 20).iloc[-1])

            # 1h indicators
            h1_rsi = 50.0
            h1_20ema = last_p
            if not df_1h.empty:
                c_1h = df_1h["close"]
                h1_rsi = float(scanner.calculate_rsi(c_1h, 14).iloc[-1])
                h1_20ema = float(scanner.calculate_ema(c_1h, 20).iloc[-1])

            # Evaluate strategy
            passes = False
            if "Strong Bullish" in crypto_preset:
                passes = (last_p > d_20ema and d_rsi > 50)
            elif "Multi-Timeframe" in crypto_preset:
                passes = (last_p > d_20ema and last_p > h4_20ema and last_p > h1_20ema)
            elif "Golden Cross" in crypto_preset:
                passes = (d_50ema >= d_200ema)
            elif "Pullback" in crypto_preset:
                passes = (abs(last_p - d_20ema) / max(d_20ema, 1) < 0.02 and d_rsi >= 45)
            elif "SuperTrend" in crypto_preset:
                passes = d_st_bull
            elif "Oversold" in crypto_preset:
                passes = (d_rsi < 40 or h4_rsi < 40)
            else:
                passes = True

            results_crypto.append({
                "Symbol": c_sym,
                "Asset": "Bitcoin" if "BTC" in c_sym else "Ethereum",
                "LTP ($)": last_p,
                "24h Change (%)": chg_24,
                "Screening Result": "✅ QUALIFIED" if passes else "❌ FAILED",
                "Daily RSI": round(d_rsi, 1),
                "Daily > 20 EMA": "✅ YES" if last_p >= d_20ema else "❌ NO",
                "4h RSI": round(h4_rsi, 1),
                "4h > 20 EMA": "✅ YES" if last_p >= h4_20ema else "❌ NO",
                "1h RSI": round(h1_rsi, 1),
                "1h > 20 EMA": "✅ YES" if last_p >= h1_20ema else "❌ NO",
                "SuperTrend": "🟢 Bullish" if d_st_bull else "🔴 Bearish",
                "Golden Cross (50>200)": "🚀 YES" if d_50ema >= d_200ema else "⚠️ NO"
            })

        df_cr_results = pd.DataFrame(results_crypto)

        # Metric Cards
        qualified_count = len(df_cr_results[df_cr_results["Screening Result"] == "✅ QUALIFIED"])
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1:
            render_metric_card("Qualified Coins", f"{qualified_count} / {len(df_cr_results)}", crypto_preset[:25] + "...", "green" if qualified_count > 0 else "red", styles)
        with cm2:
            btc_row = df_cr_results[df_cr_results["Symbol"] == "BTCUSD"].iloc[0] if not df_cr_results.empty else {}
            render_metric_card("BTCUSD Price", f"${btc_row.get('LTP ($)', 0):,.2f}", f"24h: {btc_row.get('24h Change (%)', 0):+.2f}%", "normal", styles)
        with cm3:
            eth_row = df_cr_results[df_cr_results["Symbol"] == "ETHUSD"].iloc[0] if not df_cr_results.empty else {}
            render_metric_card("ETHUSD Price", f"${eth_row.get('LTP ($)', 0):,.2f}", f"24h: {eth_row.get('24h Change (%)', 0):+.2f}%", "normal", styles)
        with cm4:
            render_metric_card("Market Status", "🟢 24/7/365 OPEN", "Delta Exchange", "green", styles)

        st.markdown("---")
        st.markdown(f"#### 📊 Crypto Screener Results — `{crypto_preset}`")

        fmt_cr = {
            "LTP ($)": "${:,.2f}",
            "24h Change (%)": "{:+.2f}%",
            "Daily RSI": "{:.1f}",
            "4h RSI": "{:.1f}",
            "1h RSI": "{:.1f}"
        }

        st_cr_styled = df_cr_results.style.format(fmt_cr).map(
            lambda v: "background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold;" if v == "✅ QUALIFIED"
            else ("background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold;" if v == "❌ FAILED" else ""),
            subset=["Screening Result"]
        ).map(
            lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0
            else ("color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["24h Change (%)"]
        )

        st.dataframe(st_cr_styled, use_container_width=True, height=180)

        # Quick Launch into Crypto Terminal
        cq1, cq2, cq3 = st.columns([1.5, 1.5, 2.0])
        with cq1:
            sel_cr_launch = st.selectbox("Select Coin to Trade / View:", ["BTCUSD", "ETHUSD"], key="crypto_screen_launch_sym")
        with cq2:
            st.write("")
            st.write("")
            if st.button("🚀 Open in 🪙 Crypto Terminal", type="primary", use_container_width=True, key="btn_open_crypto_term_from_screen"):
                st.session_state["crypto_active_symbol"] = sel_cr_launch
                st.session_state["app_active_nav_page"] = "🪙 Crypto Terminal"
                st.rerun()
        with cq3:
            st.write("")
            st.write("")
            csv_cr = df_cr_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Crypto Scan CSV",
                data=csv_cr,
                file_name=f"crypto_screener_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_dl_crypto_screen_csv"
            )

