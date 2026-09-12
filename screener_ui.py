"""
screener_ui.py
--------------
Streamlit UI for the Chartink-Style Stock Screener in Upstox Scanner.
Featuring:
1. Chartink Positional Swing Scanner (positional-scan-364) with strict Waterfall Model
   (Monthly Pass ➔ Weekly Pass ➔ Daily Pass ➔ 75-Min Pass). Strictly hides non-passing stocks.
2. Custom Condition Builder & Chartink Query Importer.
"""

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

    if universe_choice == "Nifty 50":
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

    tab_waterfall, tab_custom = st.tabs([
        "🌊 Chartink Positional Waterfall Scan (positional-scan-364)",
        "🛠️ Custom Condition Screener (Dynamic Rule Builder)"
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
                ["Nifty 50", "Nifty 100", "Top 200 Liquid Equities", "Nifty 500", "All Database Equities"],
                index=0,
                key="wf_universe_select"
            )

        with w_col2:
            wf_date_mode = st.radio(
                "Scan Date Mode:",
                ["⚡ Latest Live Data", "📅 Historical Date (As-Of)"],
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
        if "Historical" in wf_date_mode:
            hist_c1, hist_c2 = st.columns([1.6, 2.4])
            with hist_c1:
                # Default to 5 Sept 2026 or previous trading day
                default_dt = datetime(2026, 9, 5).date()
                selected_as_of = st.date_input(
                    "📅 Select Historical Scan Date:",
                    value=default_dt,
                    min_value=datetime(2020, 1, 1).date(),
                    max_value=datetime.today().date(),
                    key="wf_as_of_date_input"
                )
                as_of_param = selected_as_of.strftime("%Y-%m-%d")

            with hist_c2:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 7px 12px; border-radius: 4px; margin-top: 14px; font-size: 11.5px; color: {styles['text_secondary']};">
                    <b>🎯 Backtest Mode Active (As-Of: {as_of_param}):</b><br/>
                    All candles are sliced up to <b>{as_of_param} EOD</b>. Zero future data is used in indicators. 
                    Forward returns to present day will be calculated automatically.
                </div>
                """, unsafe_allow_html=True)

            # Quick presets
            st.markdown("<div style='font-size: 11px; color: #94A3B8; margin-bottom: 4px;'>⚡ Quick Date Presets:</div>", unsafe_allow_html=True)
            q_cols = st.columns(5)
            with q_cols[0]:
                if st.button("📅 05 Sep 2026", key="q_btn_5sep", use_container_width=True):
                    st.session_state["wf_as_of_date_input"] = datetime(2026, 9, 5).date()
                    st.rerun()
            with q_cols[1]:
                if st.button("📅 01 Sep 2026", key="q_btn_1sep", use_container_width=True):
                    st.session_state["wf_as_of_date_input"] = datetime(2026, 9, 1).date()
                    st.rerun()
            with q_cols[2]:
                if st.button("📅 14 Aug 2026", key="q_btn_14aug", use_container_width=True):
                    st.session_state["wf_as_of_date_input"] = datetime(2026, 8, 14).date()
                    st.rerun()
            with q_cols[3]:
                if st.button("📅 01 Aug 2026", key="q_btn_1aug", use_container_width=True):
                    st.session_state["wf_as_of_date_input"] = datetime(2026, 8, 1).date()
                    st.rerun()
            with q_cols[4]:
                if st.button("📅 01 Jul 2026", key="q_btn_1jul", use_container_width=True):
                    st.session_state["wf_as_of_date_input"] = datetime(2026, 7, 1).date()
                    st.rerun()

        w_btn_col1, w_btn_col2 = st.columns([3.5, 1.5])
        with w_btn_col2:
            run_wf_btn = st.button("🚀 Run Waterfall Scan", type="primary", use_container_width=True, key="run_wf_scan_btn")

        # Auto-run or button press
        scan_id = as_of_param if as_of_param else "latest"
        wf_cache_key = f"wf_res_{wf_universe}_{scan_id}"
        if run_wf_btn or wf_cache_key not in st.session_state:
            target_syms = get_target_equities(wf_universe)
            p_bar = st.progress(0)
            p_txt = st.empty()

            def _wf_progress(curr, total, sym):
                if curr % 5 == 0 or curr == total:
                    p_bar.progress(min(curr / max(total, 1), 1.0))
                    p_txt.caption(f"Screening {sym} ({curr}/{total})...")

            wf_results = run_waterfall_scan(target_syms, as_of_date=as_of_param, progress_callback=_wf_progress)
            p_bar.empty()
            p_txt.empty()
            st.session_state[wf_cache_key] = wf_results

        # Render Waterfall Dashboard
        if wf_cache_key in st.session_state:
            wf_data = st.session_state[wf_cache_key]
            counts = wf_data.get("counts", {})
            as_of_label = counts.get("as_of_date", "Latest Live")

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
                display_cols = ["Symbol", "Scan Date", "LTP", "1D Return (%)"]
                if "Return Since Scan (%)" in active_df.columns and active_df["Return Since Scan (%)"].notna().any():
                    display_cols.extend(["Return Since Scan (%)", "Latest Price"])
                display_cols.extend([
                    "Volume", "Waterfall Stage", "Monthly", "Weekly", "Daily", "75-Min",
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
                    subset=[c for c in ["Monthly", "Weekly", "Daily", "75-Min"] if c in show_df.columns]
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

    # =========================================================================
    # TAB 2: CUSTOM CONDITION SCREENER & BUILDER
    # =========================================================================
    with tab_custom:
        st.markdown("#### 🛠️ Custom Condition Screener (Dynamic Rule Builder)")
        p_col1, p_col2, p_col3, p_col4 = st.columns([1.8, 1.2, 1.2, 1.0])

        with p_col1:
            preset_names = [
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
                ["Nifty 50", "Nifty 100", "Top 200 Liquid Equities", "Nifty 500", "All Database Equities"],
                index=0,
                key="scr_universe_select"
            )

        with p_col3:
            cust_date_mode = st.radio(
                "Scan Date:",
                ["⚡ Latest Live", "📅 Historical Date"],
                index=0,
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
        if "Historical" in cust_date_mode:
            c_d1, c_d2 = st.columns([1.5, 3.5])
            with c_d1:
                c_sel_date = st.date_input(
                    "📅 Select Scan Date:",
                    value=datetime(2026, 9, 5).date(),
                    min_value=datetime(2020, 1, 1).date(),
                    max_value=datetime.today().date(),
                    key="scr_cust_as_of_date_input"
                )
                cust_as_of_param = c_sel_date.strftime("%Y-%m-%d")
            with c_d2:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 7px 12px; border-radius: 4px; margin-top: 14px; font-size: 11.5px; color: {styles['text_secondary']};">
                    <b>🎯 Backtest Mode Active (As-Of: {cust_as_of_param}):</b> Slicing candles up to <b>{cust_as_of_param} EOD</b>. Returns since scan will be included.
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

        # Chartink Link & Query Importer Expander
        with st.expander("📋 **Chartink Query / Link Importer (Paste & Convert)**", expanded=False):
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
                            st.success(f"✅ Successfully converted {len(parsed_clauses)} Chartink rules!")
                            st.rerun()

        # Condition Builder rows
        clauses = st.session_state.get("screener_clauses", [])
        to_delete_idx = None

        for i, c in enumerate(clauses):
            row_cols = st.columns([1.2, 1.6, 1.4, 1.3, 1.6, 0.6])

            with row_cols[0]:
                c["timeframe"] = st.selectbox(
                    f"TF #{i+1}",
                    ["Daily", "Weekly", "Monthly", "75-Min"],
                    index=["Daily", "Weekly", "Monthly", "75-Min"].index(c["timeframe"]) if c["timeframe"] in ["Daily", "Weekly", "Monthly", "75-Min"] else 0,
                    key=f"c_tf_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[1]:
                c["lhs"] = st.selectbox(
                    f"LHS #{i+1}",
                    INDICATOR_OPTIONS,
                    index=INDICATOR_OPTIONS.index(c["lhs"]) if c["lhs"] in INDICATOR_OPTIONS else 0,
                    key=f"c_lhs_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[2]:
                c["operator"] = st.selectbox(
                    f"Op #{i+1}",
                    OPERATOR_OPTIONS,
                    index=OPERATOR_OPTIONS.index(c["operator"]) if c["operator"] in OPERATOR_OPTIONS else 0,
                    key=f"c_op_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[3]:
                c["rhs_type"] = st.selectbox(
                    f"Type #{i+1}",
                    ["Indicator", "Number"],
                    index=0 if c["rhs_type"] == "Indicator" else 1,
                    key=f"c_rtype_{i}",
                    label_visibility="collapsed"
                )
            with row_cols[4]:
                if c["rhs_type"] == "Indicator":
                    c["rhs_indicator"] = st.selectbox(
                        f"RHS Ind #{i+1}",
                        INDICATOR_OPTIONS,
                        index=INDICATOR_OPTIONS.index(c["rhs_indicator"]) if c["rhs_indicator"] in INDICATOR_OPTIONS else 11,
                        key=f"c_rind_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    c["rhs_value"] = float(st.number_input(
                        f"Value #{i+1}",
                        value=float(c.get("rhs_value", 0.0)),
                        step=1.0,
                        key=f"c_rval_{i}",
                        label_visibility="collapsed"
                    ))
            with row_cols[5]:
                if st.button("🗑️", key=f"del_c_{i}", help="Delete this condition"):
                    to_delete_idx = i

        if to_delete_idx is not None:
            clauses.pop(to_delete_idx)
            st.session_state["screener_clauses"] = clauses
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
                st.rerun()

        with scan_col:
            run_custom_scan = st.button("🔍 Run Custom Screener Scan", type="primary", use_container_width=True, key="scr_run_custom_scan_btn")

        if run_custom_scan:
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
            p_bar = st.progress(0)
            res_df = run_screen(target_symbols, cfg, as_of_date=cust_as_of_param)
            p_bar.empty()
            st.session_state["last_screener_results"] = res_df
            st.session_state["last_screener_total_scanned"] = len(target_symbols)

        if "last_screener_results" in st.session_state:
            res_df = st.session_state["last_screener_results"]
            total_scanned = st.session_state.get("last_screener_total_scanned", 0)

            st.markdown("---")
            st.markdown("### 📊 Custom Screener Results")

            if res_df.empty:
                st.warning("⚠️ No stocks matched all the specified conditions in the selected universe.")
            else:
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

                display_df = res_df.rename(columns={"Change_%": "1D Return (%)", "LTP": "LTP (₹)", "Clauses_Passed": "Rules Passed"})
                fmt_custom = {
                    "LTP (₹)": "₹{:,.2f}",
                    "1D Return (%)": "{:+.2f}%",
                    "Return_Since_Scan_%": "{:+.2f}%",
                    "Latest_Close": "₹{:,.2f}",
                    "Volume": "{:,}"
                }
                color_subsets = [c for c in ["1D Return (%)", "Return_Since_Scan_%"] if c in display_df.columns]
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
