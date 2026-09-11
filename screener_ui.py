"""
screener_ui.py
--------------
Streamlit UI for the Chartink-Style Stock Screener in Upstox Scanner.
Enables traders to visually build, edit, and run complex multi-timeframe
technical filter criteria across NSE equities with instant results.
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

    html = f"""
    <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: {sub_color}; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 22px; font-weight: 800; color: {val_color}; margin: 3px 0 2px 0;">{value}</div>
        <div style="font-size: 11px; color: {sub_color};">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_screener_page(theme: str = "dark"):
    """Main render function for the Chartink-Style Stock Screener."""
    styles = _get_theme_styles(theme)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <h2 style="margin: 0; color: {styles['text_primary']};">🔍 Chartink-Style Stock Screener</h2>
            <p style="margin: 0; color: {styles['text_secondary']}; font-size: 14px;">
                Filter NSE stocks using multi-timeframe technical conditions, moving average crossovers, RSI, and custom criteria.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Preset & Universe Control Panel
    st.markdown("---")
    p_col1, p_col2, p_col3 = st.columns([2.0, 1.3, 1.2])

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
        filter_logic = st.radio(
            "Filter Match Logic:",
            ["Pass ALL (AND)", "Pass ANY (OR)"],
            index=0,
            horizontal=True,
            key="scr_logic_select"
        )
        logic_val = "ALL" if "ALL" in filter_logic else "ANY"

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

    # 2. Chartink Link & Query Importer Expander
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
                    else:
                        st.warning("Could not automatically recognize conditions. Try standard syntax like 'Daily Close > Daily 20 EMA'.")

    # 3. Interactive Condition Builder UI
    st.markdown("##### ⚙️ **Active Screener Filters**")
    clauses = st.session_state.get("screener_clauses", [])

    if not clauses:
        st.info("No active filters. Click '+ Add Condition' or choose a preset above.")

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

    # Add condition row button
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
        run_scan_btn = st.button("🔍 Run Stock Screener Scan", type="primary", use_container_width=True, key="scr_run_scan_btn")

    # 4. Execution & Results
    if run_scan_btn or "last_screener_results" in st.session_state:
        if run_scan_btn:
            # Build target symbol list based on universe
            all_db_syms = database.get_all_symbols()
            parquet_syms = parquet_loader.get_parquet_symbols()
            all_equities = sorted(list(set(all_db_syms + parquet_syms)))
            all_equities = [s for s in all_equities if not s.startswith("0")]

            if universe_choice == "Nifty 50":
                target_symbols = [s for s in config.NIFTY_50_SYMBOLS if s in all_equities]
            elif universe_choice == "Nifty 100":
                target_symbols = all_equities[:100]
            elif universe_choice == "Top 200 Liquid Equities":
                target_symbols = all_equities[:200]
            elif universe_choice == "Nifty 500":
                target_symbols = all_equities[:500]
            else:
                target_symbols = all_equities

            # Convert session dicts to ScreenerClause objects
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
            p_text = st.empty()

            total_n = len(target_symbols)
            matches_list = []

            for idx, sym in enumerate(target_symbols):
                if idx % 10 == 0 or idx == total_n - 1:
                    p_bar.progress(min((idx + 1) / max(total_n, 1), 1.0))
                    p_text.caption(f"Screening {sym} ({idx+1}/{total_n})...")

            res_df = run_screen(target_symbols, cfg)
            p_bar.empty()
            p_text.empty()
            st.session_state["last_screener_results"] = res_df
            st.session_state["last_screener_total_scanned"] = len(target_symbols)

        # Display results
        res_df = st.session_state.get("last_screener_results", pd.DataFrame())
        total_scanned = st.session_state.get("last_screener_total_scanned", 0)

        st.markdown("---")
        st.markdown("### 📊 Screener Results")

        if res_df.empty:
            st.warning("⚠️ No stocks matched all the specified conditions in the selected universe.")
            st.caption("Tip: Try relaxing conditions, switching match logic to 'Pass ANY (OR)', or widening the stock universe.")
        else:
            # Summary Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            match_count = len(res_df)
            gainers_count = len(res_df[res_df["Change_%"] > 0])
            avg_return = res_df["Change_%"].mean()
            top_stock = res_df.iloc[0]["Symbol"] if not res_df.empty else "N/A"

            with m1:
                render_metric_card("Matching Stocks", f"{match_count:,}", f"{(match_count/max(total_scanned,1)*100):.1f}% of universe", "green", styles)
            with m2:
                render_metric_card("Gainers vs Losers", f"{gainers_count} 🟢 / {match_count - gainers_count} 🔴", "Today's direction", "blue", styles)
            with m3:
                render_metric_card("Average Return", f"{avg_return:+.2f}%", "Across matched stocks", "green" if avg_return >= 0 else "red", styles)
            with m4:
                render_metric_card("Top Gainer", f"{top_stock}", f"{res_df.iloc[0]['Change_%']:+.2f}%" if not res_df.empty else "", "normal", styles)

            # Results Table
            st.markdown(f"##### 🎯 Matched Stocks ({match_count} found)")

            # Format and color code the dataframe
            display_df = res_df.copy()
            # Rename columns nicely
            rename_map = {
                "Change_%": "1D Return (%)",
                "LTP": "LTP (₹)",
                "Clauses_Passed": "Rules Passed"
            }
            display_df = display_df.rename(columns=rename_map)

            # Style with green/red for returns
            st.dataframe(
                display_df.style.format({
                    "LTP (₹)": "₹{:,.2f}",
                    "1D Return (%)": "{:+.2f}%",
                    "Volume": "{:,}"
                }).map(lambda v: "color: #10B981; font-weight: bold;" if isinstance(v, (int, float)) and v > 0 else ("color: #EF4444; font-weight: bold;" if isinstance(v, (int, float)) and v < 0 else ""), subset=["1D Return (%)"]),
                use_container_width=True,
                height=min(500, max(240, len(display_df) * 36))
            )

            # 1-Click Launchers into Quad Chart & Terminal
            st.markdown("##### 🚀 Quick Chart Launcher")
            launch_sym = st.selectbox("Select Matched Stock to Inspect:", options=display_df["Symbol"].tolist(), key="scr_launch_sym")
            c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.5, 2.0])

            with c_btn1:
                if st.button("📊 Open in Quad-Chart View", key="btn_open_quad", use_container_width=True):
                    st.session_state["quad_sym_select"] = launch_sym
                    st.session_state["app_active_nav_page"] = "📊 Quad-Chart View"
                    st.rerun()

            with c_btn2:
                if st.button("📈 Open in Trading Terminal", key="btn_open_term", use_container_width=True):
                    st.session_state["term_symbol_select"] = launch_sym
                    st.session_state["app_active_nav_page"] = "📈 Trading Terminal"
                    st.rerun()

            with c_btn3:
                # CSV Export
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Filtered Stocks CSV",
                    data=csv_bytes,
                    file_name=f"screener_{datetime.today().strftime('%Y%m%d')}_{selected_preset[:15]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="scr_csv_dl_btn"
                )
