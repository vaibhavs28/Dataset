"""
alert_ui.py
-----------
Streamlit UI for the Alert & Monitoring Station in Upstox Scanner.
Supports creating, monitoring, and testing:
1. Static Price Cross Alerts
2. Trendline & S/R Breakout Alerts
3. Dynamic Indicator Cross Alerts
4. Chartink Waterfall #364 Triggers
Multi-channel delivery to Telegram, WhatsApp, Email, Webhooks, and In-App Audio.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import config
import database
import parquet_loader
import alert_engine
import quadrant_image_generator
import auto_75m_broadcaster
import sync_75m_intraday


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


def get_all_equities_list() -> list:
    db_symbols = database.get_all_symbols()
    parquet_symbols = parquet_loader.get_parquet_symbols()
    popular = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "TATAMOTORS"]
    all_syms = sorted(list(set(popular + db_symbols + parquet_symbols)))
    return [s for s in all_syms if not s.startswith("0")]


def render_alert_page(theme: str = "dark"):
    """Main render function for the Alert & Monitoring Station."""
    styles = _get_theme_styles(theme)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div>
            <h2 style="margin: 0; color: {styles['text_primary']};">🔔 Multi-Channel Alert & Monitoring Station</h2>
            <p style="margin: 0; color: {styles['text_secondary']}; font-size: 14px;">
                Real-time alerts on Price Crosses, Trendlines, Dynamic Indicators, and Chartink Waterfall Setups via Telegram, WhatsApp, Email, and Webhook.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    all_symbols = get_all_equities_list()
    if not all_symbols:
        all_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]

    tab_monitor, tab_broadcaster, tab_create, tab_settings, tab_audit = st.tabs([
        "📋 Active Alerts & Monitor",
        "📸 75-Min Quadrant Broadcaster",
        "➕ Create New Alert",
        "⚙️ Notification Channels",
        "📜 Trigger History & Audit Log"
    ])

    # =========================================================================
    # TAB 1: ACTIVE ALERTS & MONITOR
    # =========================================================================
    with tab_monitor:
        alerts = alert_engine.get_all_alerts()
        active_count = len([a for a in alerts if a["is_active"] == 1])
        triggered_count = sum([a.get("trigger_count", 0) for a in alerts])
        cfg = alert_engine.get_channel_config()
        active_channels_count = sum([1 for ch in ["telegram", "whatsapp", "email", "webhook"] if cfg.get(ch, {}).get("enabled")])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Active Alerts", f"{active_count}", f"of {len(alerts)} total created", "green", styles)
        with m2:
            render_metric_card("Total Triggers Fired", f"{triggered_count}", "Historical triggers", "blue", styles)
        with m3:
            render_metric_card("Channels Enabled", f"{active_channels_count} / 4", "Telegram, WA, Email, Webhook", "purple", styles)
        with m4:
            st.write("")
            if st.button("⚡ Check All Alerts Now", type="primary", use_container_width=True, key="btn_check_all_alerts"):
                with st.spinner("Checking active alerts against fresh market data..."):
                    fired = alert_engine.check_all_active_alerts()
                    if fired:
                        st.success(f"🚨 {len(fired)} alert(s) triggered and notifications dispatched!")
                        # Play browser chime
                        st.markdown("""
                        <audio autoplay>
                            <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
                        </audio>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("✅ All active alerts evaluated. No conditions met currently.")
                    st.rerun()

        st.markdown("---")
        st.markdown("##### 🎯 Active Alert Rules")

        if not alerts:
            st.info("No alerts created yet. Click '➕ Create New Alert' tab to set your first alert!")
        else:
            for a in alerts:
                aid = a["alert_id"]
                sym = a["symbol"]
                atype = a["alert_type"]
                cond = a["condition"]
                is_active = (a["is_active"] == 1)
                t_count = a.get("trigger_count", 0)
                channels = a.get("channels", [])
                mode = a.get("trigger_mode", "ONCE")

                # Format human readable condition text
                if atype == "STATIC_PRICE":
                    cond_txt = f"LTP {cond.get('operator')} ₹{cond.get('target_price', 0):,.2f}"
                elif atype == "TRENDLINE_SR":
                    if cond.get("sr_type") == "Horizontal":
                        cond_txt = f"{cond.get('kind', 'Resistance')} Cross @ ₹{cond.get('level', 0):,.2f}"
                    else:
                        cond_txt = f"Sloping Trendline Breakout (p1: ₹{cond.get('price1')}, p2: ₹{cond.get('price2')})"
                elif atype == "INDICATOR":
                    rule = cond.get("rule", "EMA_Cross")
                    tf = cond.get("timeframe", "Daily")
                    if rule == "EMA_Cross":
                        cond_txt = f"{tf} EMA({cond.get('fast_ema')}) > EMA({cond.get('slow_ema')}) Cross"
                    elif rule == "RSI_Level":
                        cond_txt = f"{tf} RSI({cond.get('span', 14)}) {cond.get('operator')} {cond.get('threshold')}"
                    elif rule == "Hilega_Milega":
                        cond_txt = f"{tf} Hilega Milega Bullish Cross (EMA3 > WMA21 & RSI>=50)"
                    else:
                        cond_txt = f"{tf} SuperTrend Bullish Flip"
                else:
                    cond_txt = "Chartink Positional Scan #364 (Stage 4 Full Alignment)"

                # Channel badges
                ch_emojis = {
                    "telegram": "📱 TG",
                    "whatsapp": "💬 WA",
                    "email": "📧 Email",
                    "webhook": "🌐 Webhook",
                    "in_app": "🔔 In-App"
                }
                ch_str = " | ".join([ch_emojis.get(c, c) for c in channels])

                r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1.5, 3.2, 1.8, 1.2, 1.3])
                with r_col1:
                    status_badge = "🟢 ACTIVE" if is_active else "⏸️ PAUSED"
                    st.markdown(f"**{sym}** &nbsp; <span style='font-size:12px; color:{'#10B981' if is_active else '#94A3B8'};'>{status_badge}</span>", unsafe_allow_html=True)
                with r_col2:
                    st.write(f"**{atype.replace('_', ' ')}**: {cond_txt}")
                    if a.get("note"):
                        st.caption(f"📝 {a.get('note')}")
                with r_col3:
                    st.caption(f"{ch_str} • Mode: `{mode}` (Fired: {t_count})")
                with r_col4:
                    if st.button("🔔 Test Fire", key=f"test_fire_{aid}", use_container_width=True, help="Simulate immediate alert dispatch to test channels"):
                        deliv = alert_engine.dispatch_alert(
                            symbol=sym,
                            alert_type=atype,
                            trigger_price=cond.get("target_price", 100.0) or 100.0,
                            headline=f"Manual Test Trigger: {cond_txt}",
                            details="This is a test notification verifying your alert channels.",
                            selected_channels=channels
                        )
                        st.success(f"Dispatched test alert to {list(deliv.keys())}!")
                with r_col5:
                    b_col_a, b_col_b = st.columns(2)
                    with b_col_a:
                        toggle_txt = "⏸️" if is_active else "▶️"
                        if st.button(toggle_txt, key=f"toggle_act_{aid}", help="Pause / Resume alert"):
                            alert_engine.update_alert_status(aid, not is_active)
                            st.rerun()
                    with b_col_b:
                        if st.button("🗑️", key=f"del_act_{aid}", help="Delete alert"):
                            alert_engine.delete_alert(aid)
                            st.rerun()

    # =========================================================================
    # TAB 2: 75-MIN QUADRANT BROADCASTER
    # =========================================================================
    with tab_broadcaster:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h4 style="margin: 0; color: {styles['text_primary']};">📸 75-Min Waterfall & 4-Quadrant Screenshot Broadcaster</h4>
                <p style="margin: 0; color: {styles['text_secondary']}; font-size: 13px;">
                    Automated scan at every 75-min candle close (10:30, 11:45, 13:00, 14:15, 15:30 IST).
                    Captures 1600x1200 4-quadrant candlestick charts (Monthly + Weekly + Daily + 75m) and broadcasts to Telegram, Email, and WhatsApp.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sched = auto_75m_broadcaster.get_75m_schedule_status()
        m_status = "🟢 MARKET OPEN" if sched["is_market_hours"] else "⏸️ MARKET CLOSED"
        m_color = "green" if sched["is_market_hours"] else "orange"

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            render_metric_card("Market Session", m_status, "NSE Trading Hours: 09:15 - 15:30", m_color, styles)
        with col_s2:
            render_metric_card("Next 75m Candle Close", sched["next_candle_label"], f"Scheduled: {sched['next_candle_time'][-8:]} IST", "blue", styles)
        with col_s3:
            render_metric_card("Countdown Timer", sched["time_remaining_str"], f"Candle #{sched['candle_idx']} of 5", "purple", styles)
        with col_s4:
            cfg = alert_engine.get_channel_config()
            active_chs = [ch.capitalize() for ch in ["telegram", "whatsapp", "email", "webhook"] if cfg.get(ch, {}).get("enabled")]
            ch_summary = ", ".join(active_chs) if active_chs else "In-App Only"
            render_metric_card("Active Broadcast Channels", f"{len(active_chs)} Connected", ch_summary, "green" if active_chs else "normal", styles)

        st.markdown("---")

        # Controls
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 1.8, 1.8])
        with ctrl_col1:
            bc_universe = st.selectbox(
                "Scanning Universe:",
                ["All Database Equities", "Nifty 50", "Nifty 100", "Top 200 Liquid Equities", "Nifty 500"],
                index=0,
                key="bc_universe_select"
            )
        with ctrl_col2:
            bc_stage = st.selectbox(
                "Filter Tier:",
                [
                    "Stage 4: Full Alignment Only (M + W + D + 75m) [Strict]",
                    "Stage 3: M + W + D Aligned",
                    "Stage 2: M + W Aligned",
                    "All Waterfall Stages"
                ],
                index=0,
                key="bc_stage_select"
            )
            stage_num = 4 if "Stage 4" in bc_stage else (3 if "Stage 3" in bc_stage else (2 if "Stage 2" in bc_stage else 0))
        with ctrl_col3:
            st.write("")
            st.write("")
            btn_sync_now = st.button("🔄 Sync Upstox 75m Now", use_container_width=True, key="btn_sync_75m_upstox_now")
        with ctrl_col4:
            st.write("")
            st.write("")
            trigger_now = st.button("⚡ Run Scan & Broadcast", type="primary", use_container_width=True, key="btn_run_75m_now")

        auto_sync_upstox = st.checkbox(
            "🔄 Auto-sync live Upstox 75m intraday data into database before scanning",
            value=True,
            key="cb_auto_sync_upstox"
        )

        if btn_sync_now:
            sync_prog = st.progress(0.0, text=f"Initiating Upstox sync for {bc_universe}...")
            def _sync_cb(curr, tot, sym):
                frac = min(1.0, curr / max(1, tot))
                sync_prog.progress(frac, text=f"Syncing Upstox 75m: {sym} ({curr}/{tot})...")

            with st.spinner(f"Ingesting live 75-min intraday candles from Upstox API for {bc_universe}..."):
                s_res = sync_75m_intraday.sync_all_symbols_75m(
                    universe=bc_universe,
                    progress_callback=_sync_cb
                )
                sync_prog.progress(1.0, text="Sync complete!")
                st.success(
                    f"✅ Ingestion Complete in {s_res['elapsed_seconds']}s! "
                    f"Updated {s_res['synced_count']}/{s_res['total_symbols']} stocks. "
                    f"Added {s_res['total_75m_bars']} new 75m candles into DuckDB, SQLite & Parquet."
                )

        if trigger_now:
            with st.spinner(f"Running 75-Min Waterfall Scan across {bc_universe} (Auto-sync: {auto_sync_upstox})..."):
                bc_res = auto_75m_broadcaster.run_75m_waterfall_broadcast(
                    universe=bc_universe,
                    stage_filter=stage_num,
                    sync_first=auto_sync_upstox
                )
                q_count = bc_res.get("qualifying_count", 0)
                el_sec = bc_res.get("elapsed_seconds", 0.0)
                if q_count > 0:
                    st.success(f"🎉 Scan Complete in {el_sec:.1f}s! Found {q_count} qualifying stock(s). 4-Quadrant screenshots generated and dispatched!")
                else:
                    st.info(f"✅ Scan Complete in {el_sec:.1f}s. Scanned {bc_res.get('scanned_count')} stocks in {bc_universe}. Currently 0 stocks match Stage {stage_num} criteria.")


        # Single Stock Instant Test Section
        with st.expander("🖼️ **On-Demand Single Stock Quadrant Preview & Test**", expanded=False):
            t_col1, t_col2, t_col3 = st.columns([2, 2, 2])
            with t_col1:
                test_sym = st.selectbox("Select Stock to Inspect:", options=all_symbols, index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0, key="quad_test_sym")
            with t_col2:
                st.write("")
                st.write("")
                btn_gen_test = st.button("📸 Generate 4-Quadrant Chart", use_container_width=True, key="btn_gen_test_quad")
            with t_col3:
                st.write("")
                st.write("")
                btn_test_dispatch = st.button("🚀 Test Broadcast to Telegram & Email", use_container_width=True, key="btn_test_quad_dispatch")

            if btn_gen_test or btn_test_dispatch:
                with st.spinner(f"Rendering 1600x1200 Quad-Chart for {test_sym}..."):
                    gen_path = quadrant_image_generator.generate_stock_quadrant(test_sym)
                    if gen_path and os.path.exists(gen_path):
                        st.success(f"✅ Generated high-resolution Quad-Chart ({os.path.getsize(gen_path)//1024} KB)")
                        st.image(gen_path, caption=f"{test_sym} - Institutional 4-Quadrant Analysis (Monthly, Weekly, Daily, 75m)", use_container_width=True)

                        with open(gen_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download {test_sym} 1600x1200 Image",
                                data=f.read(),
                                file_name=f"{test_sym}_quadrant.png",
                                mime="image/png",
                                key=f"dl_test_{test_sym}"
                            )

                        if btn_test_dispatch:
                            daily_df = database.get_candles_df(test_sym)
                            ltp = float(daily_df["close"].iloc[-1]) if (daily_df is not None and not daily_df.empty) else 100.0
                            deliv = alert_engine.dispatch_alert(
                                symbol=test_sym,
                                alert_type="WATERFALL_75M",
                                trigger_price=ltp,
                                headline=f"Manual Test Broadcast: {test_sym}",
                                details="4-Quadrant candlestick screenshot verification.",
                                selected_channels=["telegram", "email", "in_app"],
                                screenshot_path=gen_path
                            )
                            st.success(f"Dispatched test alert with screenshot! Delivery status: {list(deliv.keys())}")
                    else:
                        st.error(f"Could not generate quadrant screenshot for {test_sym}. Check data availability.")

        st.markdown("---")

        # Gallery of Recent Quadrant Screenshots
        st.markdown("##### 🖼️ Recent 75-Min Quadrant Screenshots Gallery")
        recent_screens = auto_75m_broadcaster.get_recent_quadrant_screenshots(limit=12)

        if not recent_screens:
            st.info("No quadrant screenshots generated yet. Click '⚡ Run 75-Min Scan & Broadcast Now' or use the Single Stock Preview above!")
        else:
            # Display in a responsive 2-column grid
            for i in range(0, len(recent_screens), 2):
                g_col1, g_col2 = st.columns(2)
                
                # Item 1
                item1 = recent_screens[i]
                with g_col1:
                    with st.container():
                        st.markdown(f"**{item1['symbol']}** • `{item1['modified_at']}` • `{item1['size_kb']} KB`")
                        if os.path.exists(item1["path"]):
                            st.image(item1["path"], use_container_width=True)
                            with open(item1["path"], "rb") as f:
                                st.download_button(
                                    label=f"📥 Download {item1['symbol']}",
                                    data=f.read(),
                                    file_name=item1["filename"],
                                    mime="image/png",
                                    key=f"dl_sc_{i}"
                                )

                # Item 2
                if i + 1 < len(recent_screens):
                    item2 = recent_screens[i + 1]
                    with g_col2:
                        with st.container():
                            st.markdown(f"**{item2['symbol']}** • `{item2['modified_at']}` • `{item2['size_kb']} KB`")
                            if os.path.exists(item2["path"]):
                                st.image(item2["path"], use_container_width=True)
                                with open(item2["path"], "rb") as f:
                                    st.download_button(
                                        label=f"📥 Download {item2['symbol']}",
                                        data=f.read(),
                                        file_name=item2["filename"],
                                        mime="image/png",
                                        key=f"dl_sc_{i+1}"
                                    )

        # Broadcast History Section
        st.markdown("---")
        with st.expander("📜 **75-Min Broadcast Execution History**", expanded=False):
            b_history = auto_75m_broadcaster.get_broadcast_history(limit=20)
            if not b_history:
                st.caption("No automated broadcasts recorded yet.")
            else:
                h_rows = []
                for bh in b_history:
                    stocks_str = ", ".join([s["symbol"] for s in bh.get("stocks", [])]) if bh.get("stocks") else "None"
                    h_rows.append({
                        "Time": bh.get("timestamp"),
                        "Candle Slot": bh.get("candle_slot"),
                        "Universe": bh.get("universe"),
                        "Scanned": bh.get("scanned_count"),
                        "Qualifying (Stage 4)": bh.get("qualifying_count"),
                        "Stocks Dispatched": stocks_str,
                        "Elapsed Time": f"{bh.get('elapsed_seconds', 0)}s"
                    })
                st.dataframe(pd.DataFrame(h_rows), use_container_width=True)

    # =========================================================================
    # TAB 3: CREATE NEW ALERT
    # =========================================================================
    with tab_create:
        st.markdown("#### ➕ Create New Alert")
        c_col1, c_col2 = st.columns([1.5, 2.5])

        with c_col1:
            sel_sym = st.selectbox("1. Select Stock:", options=all_symbols, index=all_symbols.index("RELIANCE") if "RELIANCE" in all_symbols else 0, key="new_alert_sym")
            
            # Fetch current LTP
            daily_df = database.get_candles_df(sel_sym)
            curr_ltp = float(daily_df["close"].iloc[-1]) if (daily_df is not None and not daily_df.empty) else 0.0
            st.caption(f"💰 Current LTP for **{sel_sym}**: **₹{curr_ltp:,.2f}**")

            alert_category = st.selectbox(
                "2. Alert Type:",
                [
                    "🎯 Static Price Cross (Target ₹)",
                    "📐 Trendline / Support & Resistance",
                    "⚡ Dynamic Indicator (EMA, RSI, SuperTrend, Hilega Milega)",
                    "🌊 Positional Scan #364 (Stage 4 Full Alignment)"
                ],
                key="new_alert_type_choice"
            )

        with c_col2:
            cond_dict = {}

            if "Static Price" in alert_category:
                atype_val = "STATIC_PRICE"
                p_sub1, p_sub2 = st.columns(2)
                with p_sub1:
                    op = st.selectbox("Condition Operator:", [">= (Crosses Above)", "<= (Crosses Below)", "touches (Within 0.5%)"], key="new_p_op")
                    op_code = ">=" if ">=" in op else ("<=" if "<=" in op else "touches")
                    cond_dict["operator"] = op_code
                with p_sub2:
                    tgt = st.number_input("Target Price (₹):", min_value=1.0, max_value=100000.0, value=float(round(curr_ltp * 1.02, 2)), step=1.0, key="new_p_tgt")
                    cond_dict["target_price"] = tgt

            elif "Trendline" in alert_category:
                atype_val = "TRENDLINE_SR"
                sr_mode = st.radio("Trendline Mode:", ["Horizontal Support / Resistance", "Sloping Trendline"], horizontal=True, key="new_sr_mode")
                if "Horizontal" in sr_mode:
                    cond_dict["sr_type"] = "Horizontal"
                    sr_c1, sr_c2 = st.columns(2)
                    with sr_c1:
                        kind = st.selectbox("Level Kind:", ["Resistance (Breakout Above)", "Support (Breakdown Below)"], key="new_sr_kind")
                        cond_dict["kind"] = "Resistance" if "Resistance" in kind else "Support"
                    with sr_c2:
                        lvl = st.number_input("Level Price (₹):", value=float(curr_ltp), step=1.0, key="new_sr_lvl")
                        cond_dict["level"] = lvl
                else:
                    cond_dict["sr_type"] = "Sloping_Trendline"
                    tl1, tl2, tl3 = st.columns(3)
                    with tl1:
                        cond_dict["price1"] = st.number_input("Point 1 Price (₹):", value=float(round(curr_ltp * 0.95, 2)), key="new_tl_p1")
                    with tl2:
                        cond_dict["price2"] = st.number_input("Point 2 Price (₹):", value=float(curr_ltp), key="new_tl_p2")
                    with tl3:
                        cond_dict["bars_span"] = st.number_input("Bars between points:", min_value=2, max_value=200, value=20, key="new_tl_bars")

            elif "Dynamic Indicator" in alert_category:
                atype_val = "INDICATOR"
                i_rule = st.selectbox("Select Indicator Setup:", [
                    "EMA Crossover (e.g. 9 EMA > 20 EMA)",
                    "RSI Level Threshold (e.g. RSI > 60)",
                    "Hilega Milega Bullish Cross (EMA3 > WMA21 & RSI>=50)",
                    "SuperTrend Bullish Flip (Reverses to Bullish)"
                ], key="new_ind_rule")
                
                cond_dict["timeframe"] = st.selectbox("Indicator Timeframe:", ["Daily", "Weekly", "75-Min"], key="new_ind_tf")

                if "EMA Crossover" in i_rule:
                    cond_dict["rule"] = "EMA_Cross"
                    em1, em2 = st.columns(2)
                    with em1:
                        cond_dict["fast_ema"] = int(st.number_input("Fast EMA Span:", min_value=2, max_value=100, value=9, key="new_ind_f_ema"))
                    with em2:
                        cond_dict["slow_ema"] = int(st.number_input("Slow EMA Span:", min_value=5, max_value=500, value=20, key="new_ind_s_ema"))
                elif "RSI Level" in i_rule:
                    cond_dict["rule"] = "RSI_Level"
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        cond_dict["span"] = int(st.number_input("RSI Span:", min_value=2, max_value=100, value=14, key="new_ind_rsi_span"))
                    with r2:
                        cond_dict["operator"] = st.selectbox("Operator:", ["> (Crosses Above)", "< (Crosses Below)"], key="new_ind_rsi_op").split(" ")[0]
                    with r3:
                        cond_dict["threshold"] = float(st.number_input("Threshold Value:", min_value=0.0, max_value=100.0, value=60.0, key="new_ind_rsi_thresh"))
                elif "Hilega" in i_rule:
                    cond_dict["rule"] = "Hilega_Milega"
                    st.caption("Triggers when RSI(9) EMA(3) crosses above WMA(21) and RSI is above 50 (NK Sir rule).")
                else:
                    cond_dict["rule"] = "SuperTrend"
                    st.caption("Triggers immediately when 10-period, 3.0 multiplier SuperTrend flips from Bearish to Bullish.")

            else:
                atype_val = "WATERFALL_STAGE4"
                st.info("🌊 Triggers automatically when this stock satisfies all 4 tiers of the Chartink Positional Scan (Monthly + Weekly + Daily + 75-Min)!")

        st.markdown("---")
        st.markdown("##### 📡 Delivery Channels & Alert Frequency")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.markdown("Select Channels to Notify:")
            c_tg = st.checkbox("📱 Telegram Bot", value=True, key="new_ch_tg")
            c_wa = st.checkbox("💬 WhatsApp (CallMeBot / Webhook)", value=False, key="new_ch_wa")
            c_em = st.checkbox("📧 Email Alert (SMTP)", value=False, key="new_ch_em")
            c_wh = st.checkbox("🌐 Webhook (Discord / Slack / Automation)", value=False, key="new_ch_wh")
            c_aud = st.checkbox("🔔 In-App Audio & Visual Chime", value=True, key="new_ch_aud")

            selected_channels = []
            if c_tg: selected_channels.append("telegram")
            if c_wa: selected_channels.append("whatsapp")
            if c_em: selected_channels.append("email")
            if c_wh: selected_channels.append("webhook")
            if c_aud: selected_channels.append("in_app")

        with d_col2:
            trig_mode = st.radio("Trigger Mode:", ["Trigger Once & Deactivate", "Recurring (Trigger Every Confirmation Bar)"], index=0, key="new_trig_mode")
            mode_val = "ONCE" if "Once" in trig_mode else "RECURRING"
            note_val = st.text_input("Custom Alert Note (Optional):", placeholder="e.g. Swing breakout buy entry", key="new_alert_note")

        st.write("")
        if st.button("💾 Save & Activate Alert", type="primary", use_container_width=True, key="btn_save_new_alert"):
            if not selected_channels:
                st.warning("Please select at least one delivery channel.")
            else:
                aid = alert_engine.create_alert(
                    symbol=sel_sym,
                    alert_type=atype_val,
                    condition=cond_dict,
                    channels=selected_channels,
                    trigger_mode=mode_val,
                    note=note_val
                )
                st.success(f"✅ Alert #{aid} for {sel_sym} successfully created and activated!")
                st.rerun()

    # =========================================================================
    # TAB 3: NOTIFICATION CHANNELS CONFIGURATION
    # =========================================================================
    with tab_settings:
        st.markdown("#### ⚙️ Notification Channel Settings")
        st.caption("Configure your API tokens and credentials once. All settings are stored securely in local SQLite.")

        cfg = alert_engine.get_channel_config()

        # 1. TELEGRAM SETTINGS
        with st.expander("📱 **Telegram Bot Configuration**", expanded=True):
            st.markdown("""
            **How to get Telegram Bot in 20 seconds:**
            1. Open Telegram, search for `@BotFather`, and send `/newbot` to get your **Bot Token**.
            2. Start your bot, search for `@userinfobot`, and send `/start` to get your **Chat ID**.
            """)
            tg_en = st.checkbox("Enable Telegram Notifications", value=cfg["telegram"].get("enabled", False), key="cfg_tg_en")
            tc1, tc2 = st.columns(2)
            with tc1:
                tg_token = st.text_input("Telegram Bot Token:", value=cfg["telegram"].get("bot_token", ""), type="password", key="cfg_tg_token")
            with tc2:
                tg_chat = st.text_input("Telegram Chat ID:", value=cfg["telegram"].get("chat_id", ""), key="cfg_tg_chat")

            if st.button("📲 Test Telegram Notification", key="btn_test_tg"):
                with st.spinner("Sending test Telegram message..."):
                    ok, msg = alert_engine.send_telegram(tg_token, tg_chat, "✅ <b>Upstox Alert System</b>: Telegram notifications connected successfully!")
                    if ok:
                        st.success("✅ Telegram test message sent successfully!")
                    else:
                        st.error(f"❌ Failed to send Telegram message: {msg}")

        # 2. WHATSAPP SETTINGS
        with st.expander("💬 **WhatsApp Configuration (CallMeBot / Webhook)**", expanded=False):
            st.markdown("""
            **Free 1-Minute Setup with CallMeBot:**
            1. Add CallMeBot to your phone contacts: `+34 644 76 66 43`.
            2. Send message: `I allow callmebot to send me messages` to that number on WhatsApp.
            3. You will receive your **API Key** instantly.
            """)
            wa_en = st.checkbox("Enable WhatsApp Notifications", value=cfg["whatsapp"].get("enabled", False), key="cfg_wa_en")
            wa_mode = st.radio("WhatsApp Delivery Method:", ["CallMeBot API (Free)", "Custom Webhook Endpoint"], index=0 if cfg["whatsapp"].get("mode", "callmebot") == "callmebot" else 1, key="cfg_wa_mode")

            if "CallMeBot" in wa_mode:
                wc1, wc2 = st.columns(2)
                with wc1:
                    wa_phone = st.text_input("Phone Number with Country Code (e.g. +919876543210):", value=cfg["whatsapp"].get("phone", ""), key="cfg_wa_phone")
                with wc2:
                    wa_key = st.text_input("CallMeBot API Key:", value=cfg["whatsapp"].get("api_key", ""), type="password", key="cfg_wa_key")
                wa_wh_url = ""
            else:
                wa_wh_url = st.text_input("Custom WhatsApp Webhook URL:", value=cfg["whatsapp"].get("webhook_url", ""), key="cfg_wa_wh_url")
                wa_phone = ""
                wa_key = ""

            if st.button("💬 Test WhatsApp Notification", key="btn_test_wa"):
                with st.spinner("Sending test WhatsApp message..."):
                    ok, msg = alert_engine.send_whatsapp(
                        phone=wa_phone,
                        api_key=wa_key,
                        message="*Upstox Alert System*: WhatsApp notifications connected successfully!",
                        mode="callmebot" if "CallMeBot" in wa_mode else "webhook",
                        webhook_url=wa_wh_url
                    )
                    if ok:
                        st.success("✅ WhatsApp test message sent successfully!")
                    else:
                        st.error(f"❌ Failed to send WhatsApp message: {msg}")

        # 3. EMAIL (SMTP) SETTINGS
        with st.expander("📧 **Email Configuration (SMTP)**", expanded=False):
            st.caption("Supports Gmail, Outlook, Yahoo, and private SMTP servers.")
            em_en = st.checkbox("Enable Email Notifications", value=cfg["email"].get("enabled", False), key="cfg_em_en")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                em_host = st.text_input("SMTP Host:", value=cfg["email"].get("smtp_host", "smtp.gmail.com"), key="cfg_em_host")
            with ec2:
                em_port = int(st.number_input("SMTP Port:", min_value=25, max_value=65535, value=int(cfg["email"].get("smtp_port", 587)), key="cfg_em_port"))
            with ec3:
                em_to = st.text_input("Recipient Email:", value=cfg["email"].get("to_address", ""), key="cfg_em_to")

            ec4, ec5 = st.columns(2)
            with ec4:
                em_user = st.text_input("SMTP Username / Email:", value=cfg["email"].get("username", ""), key="cfg_em_user")
            with ec5:
                em_pass = st.text_input("SMTP App Password:", value=cfg["email"].get("password", ""), type="password", help="For Gmail, generate an App Password from Google Account > Security", key="cfg_em_pass")

            if st.button("📧 Test Email Notification", key="btn_test_em"):
                with st.spinner("Sending test email..."):
                    ok, msg = alert_engine.send_email(
                        smtp_host=em_host,
                        smtp_port=em_port,
                        username=em_user,
                        password=em_pass,
                        to_address=em_to,
                        subject="Upstox Alert Test: Email Connected",
                        message="Upstox Pro Multi-Timeframe Scanner: Email alerts have been configured successfully."
                    )
                    if ok:
                        st.success(f"✅ Test email delivered to {em_to}!")
                    else:
                        st.error(f"❌ Failed to send email: {msg}")

        # 4. WEBHOOK / DISCORD SETTINGS
        with st.expander("🌐 **Webhook Endpoint (Discord / Slack / Automation)**", expanded=False):
            wh_en = st.checkbox("Enable Webhook Notifications", value=cfg["webhook"].get("enabled", False), key="cfg_wh_en")
            wh_url = st.text_input("Webhook Target URL:", value=cfg["webhook"].get("url", ""), placeholder="https://discord.com/api/webhooks/...", key="cfg_wh_url")

            if st.button("🌐 Test Webhook Notification", key="btn_test_wh"):
                with st.spinner("Dispatching test payload to webhook..."):
                    payload = {"event": "test_connection", "message": "Upstox Alert Webhook connected successfully!", "timestamp": str(datetime.now())}
                    ok, msg = alert_engine.send_webhook(wh_url, payload)
                    if ok:
                        st.success("✅ Webhook payload delivered successfully!")
                    else:
                        st.error(f"❌ Webhook dispatch failed: {msg}")

        st.write("")
        if st.button("💾 Save All Channel Settings", type="primary", use_container_width=True, key="btn_save_all_cfg"):
            new_cfg = {
                "telegram": {
                    "enabled": tg_en,
                    "bot_token": tg_token,
                    "chat_id": tg_chat
                },
                "whatsapp": {
                    "enabled": wa_en,
                    "phone": wa_phone if "CallMeBot" in wa_mode else "",
                    "api_key": wa_key if "CallMeBot" in wa_mode else "",
                    "mode": "callmebot" if "CallMeBot" in wa_mode else "webhook",
                    "webhook_url": wa_wh_url if "Webhook" in wa_mode else ""
                },
                "email": {
                    "enabled": em_en,
                    "smtp_host": em_host,
                    "smtp_port": em_port,
                    "username": em_user,
                    "password": em_pass,
                    "to_address": em_to
                },
                "webhook": {
                    "enabled": wh_en,
                    "url": wh_url
                },
                "audio": {"enabled": True}
            }
            alert_engine.save_channel_config(new_cfg)
            st.success("✅ All channel credentials and configurations saved successfully!")
            st.rerun()

    # =========================================================================
    # TAB 4: TRIGGER HISTORY & AUDIT LOG
    # =========================================================================
    with tab_audit:
        st.markdown("#### 📜 Alert Trigger History & Audit Log")
        logs = alert_engine.get_alert_logs(limit=50)

        if not logs:
            st.info("No alert triggers recorded yet.")
        else:
            log_rows = []
            for l in logs:
                deliv_st = l.get("delivery_status", {})
                deliv_summary = ", ".join([f"{k}: {'✅' if v.get('success') else '❌'}" for k, v in deliv_st.items()])
                log_rows.append({
                    "Time": str(l.get("triggered_at", ""))[:19],
                    "Symbol": l.get("symbol"),
                    "Alert Type": l.get("alert_type", "").replace("_", " "),
                    "Price (₹)": f"₹{float(l.get('trigger_price', 0)):,.2f}",
                    "Message": l.get("message"),
                    "Channel Delivery": deliv_summary
                })
            df_logs = pd.DataFrame(log_rows)
            st.dataframe(df_logs, use_container_width=True, height=400)

            # Export log CSV
            csv_logs = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Alert Audit Logs CSV",
                data=csv_logs,
                file_name=f"alert_audit_logs_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="btn_dl_alert_logs"
            )
