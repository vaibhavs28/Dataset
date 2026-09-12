"""
TradingView Chart Generator for Streamlit
Provides:
1. TradingView Lightweight Charts (v4.x) for local multi-timeframe Parquet data (75m, Daily, Weekly, Monthly)
   with Candlesticks, Volume, EMAs (5, 9, 13, 20, 26, 50, 200), Weekly Pivots (R1, P, S1),
   and RSI panel (RSI curve, EMA 3, WMA 21, solid white 50 line with linewidth 2, light white shading above 50).
   Includes:
   - Fullscreen button (Native browser API + CSS fallback)
   - Clean right-side Y-axis with NO labels (lastValueVisible=false, axisLabelVisible=false)
   - Guaranteed visible hover tooltip for every candle showing Date, OHLC, % Change, Volume, and EMAs
2. Official TradingView Advanced Real-Time Cloud Widget for live full-terminal charting.
"""

import json
import re
import pandas as pd
import numpy as np
import scanner


def prepare_chart_data(df: pd.DataFrame, is_intraday: bool = False) -> tuple:
    """
    Transforms a DataFrame into clean JSON-serializable structures for Lightweight Charts.
    Ensures timestamps are strictly ascending integers or ISO date strings.
    """
    if df.empty:
        return df, [], []

    df_clean = df.copy()
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index)

    df_clean = df_clean.sort_index()
    df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

    candles = []
    volumes = []
    
    for dt, row in df_clean.iterrows():
        if is_intraday:
            if dt.tzinfo is None:
                dt_ist = dt.tz_localize("Asia/Kolkata")
            else:
                dt_ist = dt.tz_convert("Asia/Kolkata")
            t_val = int(dt_ist.timestamp())
        else:
            t_val = dt.strftime("%Y-%m-%d")
        
        o = float(row.get("open", 0))
        h = float(row.get("high", 0))
        l = float(row.get("low", 0))
        c = float(row.get("close", 0))
        v = float(row.get("volume", 0)) if "volume" in row and not pd.isna(row["volume"]) else 0
        
        if o > 0 and h > 0 and l > 0 and c > 0:
            candles.append({
                "time": t_val,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2)
            })
            volumes.append({
                "time": t_val,
                "value": round(v, 2),
                "color": "rgba(8, 153, 129, 0.4)" if c >= o else "rgba(242, 54, 69, 0.4)"
            })

    return df_clean, candles, volumes


def generate_lightweight_chart_html(
    df: pd.DataFrame,
    symbol: str,
    timeframe_name: str,
    ema_dict: dict,
    pivot_dict: dict = None,
    show_rsi: bool = True,
    show_volume: bool = True,
    rsi_span: int = 9,
    height: int = 450,
    is_intraday: bool = False,
    chart_id: str = "tv_chart",
    theme: str = "light"
) -> str:
    """
    Renders an HTML snippet with TradingView Lightweight Charts.
    Supports both Dark and Light themes with real-time in-canvas toggle.
    """
    is_light = (str(theme).lower() == "light")
    bg_init = "#ffffff" if is_light else "#131722"
    txt_init = "#787B86"

    if df.empty:
        return f"""<div style="height:{height}px; background:{bg_init}; color:{txt_init}; display:flex; align-items:center; justify-content:center; font-family:sans-serif; border-radius:8px; border:1px solid {'#e0e3eb' if is_light else '#2A2E39'};">No data available for {symbol}</div>"""

    df_clean, candles, volumes = prepare_chart_data(df, is_intraday=is_intraday)
    
    if not candles:
        return f"""<div style="height:{height}px; background:{bg_init}; color:{txt_init}; display:flex; align-items:center; justify-content:center; font-family:sans-serif; border-radius:8px; border:1px solid {'#e0e3eb' if is_light else '#2A2E39'};">Insufficient price data for {symbol}</div>"""

    # Sanitize chart_id so it is 100% valid as a JS identifier (removes hyphens, ampersands, spaces)
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', str(chart_id))

    # Prepare EMAs series data
    emas_data = {}
    for col_name in ema_dict:
        if col_name in df_clean.columns:
            series_pts = []
            for dt, row in df_clean.iterrows():
                val = row[col_name]
                if not pd.isna(val) and val > 0:
                    t_val = int(dt.timestamp()) if is_intraday else dt.strftime("%Y-%m-%d")
                    series_pts.append({"time": t_val, "value": round(float(val), 2)})
            if series_pts:
                c_val = ema_dict[col_name]
                if isinstance(c_val, dict):
                    c = c_val.get("color", "#38BDF8")
                    l_style = c_val.get("lineStyle", 0)
                    l_width = c_val.get("lineWidth", 1.8)
                    l_title = c_val.get("name", col_name)
                else:
                    c = str(c_val)
                    l_style = 1 if col_name == "AVWAP_JUN2022" else 0
                    l_width = 1.8 if "AVWAP" in col_name else (2.2 if col_name == "Daily_EMA_20" else 1.5)
                    l_title = "AVWAP Mar 2020" if col_name == "AVWAP_MAR2020" else ("AVWAP Jun 2022" if col_name == "AVWAP_JUN2022" else col_name)

                # If light theme and color is white, use dark slate
                if is_light and (c.upper() in ["#FFFFFF", "WHITE"]):
                    c = "#1E293B"
                emas_data[col_name] = {
                    "color": c,
                    "lineStyle": l_style,
                    "lineWidth": l_width,
                    "title": l_title,
                    "data": series_pts
                }

    # Prepare Pivot levels data
    pivots_data = {}
    if pivot_dict:
        for p_col, p_cfg in pivot_dict.items():
            if p_col in df_clean.columns:
                series_pts = []
                for dt, row in df_clean.iterrows():
                    val = row[p_col]
                    if not pd.isna(val) and val > 0:
                        t_val = int(dt.timestamp()) if is_intraday else dt.strftime("%Y-%m-%d")
                        series_pts.append({"time": t_val, "value": round(float(val), 2)})
                if series_pts:
                    pivots_data[p_col] = {
                        "color": p_cfg.get("color", "#FFFFFF"),
                        "name": p_cfg.get("name", p_col),
                        "data": series_pts
                    }

    # Prepare RSI data
    has_rsi = show_rsi and ("RSI" in df_clean.columns) and not df_clean["RSI"].dropna().empty
    rsi_pts = []
    rsi_ema3_pts = []
    rsi_wma21_pts = []

    if has_rsi:
        for dt, row in df_clean.iterrows():
            t_val = int(dt.timestamp()) if is_intraday else dt.strftime("%Y-%m-%d")
            r_val = row.get("RSI")
            if not pd.isna(r_val):
                rsi_pts.append({"time": t_val, "value": round(float(r_val), 2)})
            
            if "RSI_EMA3" in df_clean.columns:
                e3 = row.get("RSI_EMA3")
                if not pd.isna(e3):
                    rsi_ema3_pts.append({"time": t_val, "value": round(float(e3), 2)})
            
            if "RSI_WMA21" in df_clean.columns:
                w21 = row.get("RSI_WMA21")
                if not pd.isna(w21):
                    rsi_wma21_pts.append({"time": t_val, "value": round(float(w21), 2)})

    # Calculate dimensions
    header_h = 36
    avail_h = height - header_h
    if has_rsi:
        main_h = max(220, int(avail_h * 0.68))
        rsi_h = max(100, int(avail_h * 0.32))
    else:
        main_h = avail_h
        rsi_h = 0

    candles_json = json.dumps(candles)
    volumes_json = json.dumps(volumes)
    emas_json = json.dumps(emas_data)
    pivots_json = json.dumps(pivots_data)
    rsi_json = json.dumps(rsi_pts)
    rsi_ema3_json = json.dumps(rsi_ema3_pts)
    rsi_wma21_json = json.dumps(rsi_wma21_pts)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{symbol} {timeframe_name} - TradingView</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            background-color: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }}
        body.light-theme {{
            background-color: #ffffff;
            color: #131722;
        }}
        .chart-wrapper {{
            width: 100%;
            height: {height}px;
            display: flex;
            flex-direction: column;
            position: relative;
            background-color: #131722;
            border-radius: 8px;
            border: 1px solid #2A2E39;
            overflow: hidden;
        }}
        .chart-wrapper.light-theme {{
            background-color: #ffffff;
            border-color: #e0e3eb;
        }}
        .chart-wrapper.css-fullscreen {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 999999 !important;
            border-radius: 0 !important;
            border: none !important;
        }}
        .header-bar {{
            height: 36px;
            padding: 0 10px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e222d;
            border-bottom: 1px solid #2A2E39;
            user-select: none;
            flex-shrink: 0;
            z-index: 10;
        }}
        .light-theme .header-bar {{
            background: #f0f3fa;
            border-bottom: 1px solid #e0e3eb;
        }}
        .header-left {{
            display: flex;
            align-items: center;
            gap: 12px;
            overflow: hidden;
        }}
        .title-badge {{
            color: #00E5FF;
            font-weight: 700;
            font-size: 13px;
        }}
        .light-theme .title-badge {{
            color: #0052FF;
        }}
        .tf-badge {{
            color: #94A3B8;
            font-size: 11px;
            margin-left: 4px;
        }}
        .light-theme .tf-badge {{
            color: #64748b;
        }}
        .ohlc-legend {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            font-family: 'SF Mono', Consolas, Monaco, monospace;
        }}
        .ohlc-item {{
            color: #787B86;
        }}
        .light-theme .ohlc-item {{
            color: #64748b;
        }}
        .ohlc-val {{
            color: #D1D4DC;
            font-weight: 600;
        }}
        .light-theme .ohlc-val {{
            color: #131722;
        }}
        .header-right {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .btn-ctrl {{
            background: #2A2E39;
            color: #D1D4DC;
            border: 1px solid #363C4E;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 500;
        }}
        .btn-ctrl:hover {{
            background: #363C4E;
            color: #FFFFFF;
            border-color: #4A5268;
        }}
        .btn-ctrl.active {{
            background: #2962FF;
            color: #FFFFFF;
            border-color: #2962FF;
        }}
        .light-theme .btn-ctrl {{
            background: #ffffff;
            color: #131722;
            border: 1px solid #d1d5db;
        }}
        .light-theme .btn-ctrl:hover {{
            background: #e2e8f0;
            border-color: #94a3b8;
        }}
        .light-theme .btn-ctrl.active {{
            background: #2962FF;
            color: #ffffff;
            border-color: #2962FF;
        }}
        .badge-tv {{
            background: #2962FF;
            color: #FFFFFF;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .chart-content-body {{
            display: flex;
            flex-direction: row;
            width: 100%;
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        .tv-draw-bar {{
            width: 38px;
            background: #1e222d;
            border-right: 1px solid #2A2E39;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 0;
            gap: 4px;
            flex-shrink: 0;
            user-select: none;
            z-index: 25;
        }}
        .light-theme .tv-draw-bar {{
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}
        .tv-tool-btn {{
            width: 28px;
            height: 28px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
            color: #94A3B8;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            padding: 0;
        }}
        .tv-tool-btn:hover {{
            background: #2A2E39;
            color: #FFFFFF;
        }}
        .light-theme .tv-tool-btn {{
            color: #64748b;
        }}
        .light-theme .tv-tool-btn:hover {{
            background: #e2e8f0;
            color: #0f172a;
        }}
        .tv-tool-btn.active {{
            background: #2962FF !important;
            color: #FFFFFF !important;
            border-color: #2962FF !important;
            box-shadow: 0 0 8px rgba(41, 98, 255, 0.4);
        }}
        .tv-tool-sep {{
            width: 20px;
            height: 1px;
            background: #2A2E39;
            margin: 4px 0;
        }}
        .light-theme .tv-tool-sep {{
            background: #e2e8f0;
        }}
        .tv-color-wrapper {{
            position: relative;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid #363C4E;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            margin: 2px 0;
        }}
        .tv-color-picker {{
            position: absolute;
            top: -10px;
            left: -10px;
            width: 44px;
            height: 44px;
            border: none;
            cursor: pointer;
            opacity: 0;
        }}
        .tv-color-dot {{
            width: 100%;
            height: 100%;
            background-color: #00E5FF;
            border-radius: 50%;
        }}
        .tv-chart-pane-wrapper {{
            display: flex;
            flex-direction: column;
            flex: 1;
            height: 100%;
            position: relative;
            overflow: hidden;
        }}
        .tv-main-pane {{
            width: 100%;
            height: {main_h}px;
            position: relative;
            flex: 1;
        }}
        .tv-draw-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 15;
            pointer-events: none;
        }}
        #main_container_{safe_id} {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        #rsi_container_{safe_id} {{
            width: 100%;
            height: {rsi_h}px;
            position: relative;
            border-top: 1px solid #2A2E39;
            flex-shrink: 0;
        }}
        .light-theme #rsi_container_{safe_id} {{
            border-top: 1px solid #e0e3eb;
        }}
        /* Canvas crosshair cursor enforcement */
        .tv-chart-pane-wrapper,
        .tv-main-pane,
        #main_container_{safe_id},
        #main_container_{safe_id} canvas,
        #rsi_container_{safe_id},
        #rsi_container_{safe_id} canvas,
        .tv-draw-canvas {{
            cursor: crosshair !important;
        }}
        .single-crosshair-v {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            border-left: 1px dashed rgba(255, 255, 255, 0.45);
            z-index: 35;
            pointer-events: none;
            display: none;
        }}
        .light-theme .single-crosshair-v {{
            border-left: 1px dashed rgba(30, 41, 59, 0.45);
        }}
        /* Remove any background tint on RSI / Hilega Milega pane */
        .rsi-bull-tint {{
            display: none !important;
        }}
        /* Hilega Milega Header & Live Value Badges */
        .hm-header {{
            position: absolute;
            top: 5px;
            left: 10px;
            z-index: 10;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 11px;
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .hm-title {{
            color: #94A3B8;
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        .light-theme .hm-title {{
            color: #1E293B;
        }}
        .hm-val {{
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 11px;
            font-weight: 700;
        }}
        .hm-rsi {{
            color: #F8FAFC;
        }}
        .light-theme .hm-rsi {{
            color: #131722;
        }}
        .hm-ema3 {{
            color: #4CAF50;
        }}
        .hm-wma21 {{
            color: #FF5252;
        }}
        /* Floating Interactive Tooltip attached directly to chart-wrapper */
        .tv-floating-tooltip {{
            display: none;
            position: absolute;
            width: 205px;
            padding: 9px 12px;
            background: rgba(19, 23, 34, 0.96);
            border: 1px solid #2962FF;
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.75);
            pointer-events: none;
            z-index: 99999 !important;
            font-size: 11px;
            color: #D1D4DC;
            backdrop-filter: blur(6px);
            line-height: 1.5;
        }}
        .light-theme .tv-floating-tooltip {{
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid #2962FF;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            color: #131722;
        }}
        .tt-date {{
            font-weight: 700;
            color: #00E5FF;
            margin-bottom: 5px;
            padding-bottom: 3px;
            border-bottom: 1px solid #2A2E39;
            font-size: 11px;
        }}
        .light-theme .tt-date {{
            color: #0052FF;
            border-bottom: 1px solid #e0e3eb;
        }}
        .tt-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 2px 0;
        }}
        .tt-lbl {{
            color: #94A3B8;
        }}
        .light-theme .tt-lbl {{
            color: #64748b;
        }}
        .tt-val {{
            font-weight: 600;
            font-family: 'SF Mono', Consolas, Monaco, monospace;
        }}
    </style>
</head>
<body class="{'light-theme' if is_light else ''}">
    <div id="wrapper_{safe_id}" class="chart-wrapper {'light-theme' if is_light else ''}">
        <div class="header-bar">
            <div class="header-left">
                <div>
                    <span class="title-badge">{symbol.upper()}</span>
                    <span class="tf-badge">{timeframe_name}</span>
                </div>
                <div id="legend_{safe_id}" class="ohlc-legend">
                    <span class="ohlc-item">O: <span id="leg_o_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-item">H: <span id="leg_h_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-item">L: <span id="leg_l_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-item">C: <span id="leg_c_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-item">Chg: <span id="leg_chg_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-item">Vol: <span id="leg_v_{safe_id}" class="ohlc-val">--</span></span>
                </div>
            </div>
            <div class="header-right">
                <button id="theme_btn_{safe_id}" class="btn-ctrl" type="button" title="Toggle Dark / Light Theme">{'☀️ Light' if is_light else '🌙 Dark'}</button>
                <button id="vol_btn_{safe_id}" class="btn-ctrl {'active' if show_volume else ''}" type="button" title="Hide / Show Volume">📊 Volume</button>
                <span class="badge-tv">TRADINGVIEW</span>
                <button id="fs_btn_{safe_id}" class="btn-ctrl" type="button">⛶ Fullscreen</button>
            </div>
        </div>
        <div class="chart-content-body">
            <div id="draw_toolbar_{safe_id}" class="tv-draw-bar">
                <button class="tv-tool-btn active" data-tool="pointer" title="Cursor / Pointer (Crosshair)">✛</button>
                <button class="tv-tool-btn" data-tool="trendline" title="Trend Line">╱</button>
                <button class="tv-tool-btn" data-tool="horizontal" title="Horizontal Line (Support/Resistance)">―</button>
                <button class="tv-tool-btn" data-tool="ray" title="Horizontal Ray">⟶</button>
                <button class="tv-tool-btn" data-tool="rectangle" title="Rectangle / Supply & Demand Zone">▭</button>
                <button class="tv-tool-btn" data-tool="fib" title="Fibonacci Retracement">≡</button>
                <button class="tv-tool-btn" data-tool="brush" title="Brush / Freehand">✎</button>
                <button class="tv-tool-btn" data-tool="text" title="Text Annotation">T</button>
                <div class="tv-tool-sep"></div>
                <div class="tv-color-wrapper" title="Pick Drawing Color">
                    <div id="color_dot_{safe_id}" class="tv-color-dot" style="background-color: #00E5FF;"></div>
                    <input type="color" id="draw_color_{safe_id}" value="#00E5FF" class="tv-color-picker" />
                </div>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" id="draw_undo_{safe_id}" title="Undo Last Drawing (Ctrl+Z)">↩</button>
                <button class="tv-tool-btn" id="draw_clear_{safe_id}" title="Clear All Drawings">🗑</button>
            </div>
            <div class="tv-chart-pane-wrapper">
                <div class="single-crosshair-v" id="ch_v_{safe_id}"></div>
                <div id="main_pane_{safe_id}" class="tv-main-pane">
                    <div id="main_container_{safe_id}"></div>
                    <canvas id="draw_canvas_{safe_id}" class="tv-draw-canvas"></canvas>
                </div>
                {"<div id='rsi_container_" + safe_id + "'><div class='hm-header'><span class='hm-title'>Hilega Milega(by NK sir) [anuragM]</span><span class='hm-val hm-rsi' id='hm_rsi_val_" + safe_id + "'></span><span class='hm-val hm-ema3' id='hm_ema3_val_" + safe_id + "'></span><span class='hm-val hm-wma21' id='hm_wma21_val_" + safe_id + "'></span></div><div class='rsi-bull-tint'></div></div>" if has_rsi else ""}
            </div>
        </div>
    </div>

    <script>
        (function() {{
            const wrapper = document.getElementById("wrapper_{safe_id}");
            const mainContainer = document.getElementById("main_container_{safe_id}");
            const candlesData = {candles_json};
            const volumesData = {volumes_json};
            const emasData = {emas_json};
            const pivotsData = {pivots_json};
            const hasRsi = {str(has_rsi).lower()};
            let currentTheme = '{ "light" if is_light else "dark" }';
            const isLightInit = (currentTheme === 'light');

            // Dynamically create floating tooltip and append to wrapper for guaranteed top z-index stacking
            const tooltip = document.createElement("div");
            tooltip.className = "tv-floating-tooltip";
            tooltip.id = "tooltip_{safe_id}";
            wrapper.appendChild(tooltip);

            // Chart Configuration: No labels on right side y-axis
            const chartOptions = {{
                layout: {{
                    background: {{ color: isLightInit ? '#ffffff' : '#131722' }},
                    textColor: isLightInit ? '#131722' : '#94A3B8',
                    fontSize: 11
                }},
                grid: {{
                    vertLines: {{ color: isLightInit ? '#f0f3fa' : '#1f2430' }},
                    horzLines: {{ color: isLightInit ? '#f0f3fa' : '#1f2430' }}
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                    autoScale: true,
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.25
                    }}
                }},
                localization: {{
                    locale: 'en-IN',
                    dateFormat: 'yyyy-MM-dd',
                    timeFormatter: (time) => {{
                        if (typeof time === 'number') {{
                            const d = new Date(time * 1000);
                            return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }}) + ' ' +
                                   d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + ' IST';
                        }}
                        return String(time);
                    }},
                }},
                timeScale: {{
                    borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                    timeVisible: {str(is_intraday).lower()},
                    secondsVisible: false,
                    rightOffset: 8,
                    tickMarkFormatter: (time, tickMarkType, locale) => {{
                        if (typeof time === 'number') {{
                            const d = new Date(time * 1000);
                            if (tickMarkType === 0) {{
                                return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', year: 'numeric' }});
                            }} else if (tickMarkType === 1) {{
                                return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', month: 'short' }});
                            }} else if (tickMarkType === 2) {{
                                return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short' }});
                            }} else {{
                                return d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }});
                            }}
                        }}
                        return null;
                    }},
                }},
                handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
            }};

            const chart = LightweightCharts.createChart(mainContainer, chartOptions);

            // Candlestick Series: NO label on right y-axis (lastValueVisible=false)
            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#F23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#F23645',
                lastValueVisible: false,
                priceLineVisible: false
            }});
            candleSeries.setData(candlesData);

            // Volume Series: NO label on right y-axis
            let isVolVisible = {'true' if show_volume else 'false'};
            const volumeSeries = chart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: 'volume',
                lastValueVisible: false,
                priceLineVisible: false,
                visible: isVolVisible
            }});
            chart.priceScale('volume').applyOptions({{
                scaleMargins: {{
                    top: 0.82,
                    bottom: 0
                }}
            }});
            volumeSeries.setData(volumesData);

            const volBtn = document.getElementById("vol_btn_{safe_id}");
            if (volBtn) {{
                volBtn.addEventListener("click", () => {{
                    isVolVisible = !isVolVisible;
                    volumeSeries.applyOptions({{ visible: isVolVisible }});
                    volBtn.classList.toggle("active", isVolVisible);
                }});
            }}

            // Overlay EMAs & Indicators: NO labels on right y-axis
            const emaSeriesMap = {{}};
            for (const [colName, colCfg] of Object.entries(emasData)) {{
                let lStyle = LightweightCharts.LineStyle.Solid;
                if (colCfg.lineStyle === 1 || colCfg.lineStyle === 'dotted') {{
                    lStyle = LightweightCharts.LineStyle.Dotted;
                }} else if (colCfg.lineStyle === 2 || colCfg.lineStyle === 'dashed') {{
                    lStyle = LightweightCharts.LineStyle.Dashed;
                }}
                const emaLine = chart.addLineSeries({{
                    color: colCfg.color,
                    lineWidth: colCfg.lineWidth || 1.8,
                    lineStyle: lStyle,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    title: ''
                }});
                emaLine.setData(colCfg.data);
                emaSeriesMap[colName] = {{ series: emaLine, color: colCfg.color, title: colCfg.title || colName }};
            }}

            // Overlay Weekly Pivots (R1, P, S1): NO labels on right y-axis
            for (const [pCol, pCfg] of Object.entries(pivotsData)) {{
                const pivotLine = chart.addLineSeries({{
                    color: pCfg.color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    title: ''
                }});
                pivotLine.setData(pCfg.data);
            }}

            if (candlesData && candlesData.length) {{
                const tot = candlesData.length;
                const vBars = Math.min(tot, { "80" if is_intraday else "120" });
                chart.timeScale().setVisibleLogicalRange({{
                    from: Math.max(0, tot - vBars),
                    to: tot + 3,
                }});
            }} else {{
                chart.timeScale().fitContent();
            }}

            // History loaded notification pill & range change listener
            const histBadge = document.createElement("div");
            histBadge.className = "tv-hist-badge";
            histBadge.id = "hist_badge_{safe_id}";
            wrapper.appendChild(histBadge);

            let histTimer = null;
            chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                if (!range) return;
                if (range.from <= 10 && candlesData && candlesData.length > 30) {{
                    const fc = candlesData[0];
                    let dLabel = '';
                    if (typeof fc.time === 'number') {{
                        const fd = new Date(fc.time * 1000);
                        dLabel = fd.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric' }}) + ' ' +
                                 fd.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + ' IST';
                    }} else {{
                        dLabel = String(fc.time);
                    }}
                    histBadge.textContent = `📅 History Loaded: ${{dLabel}} (${{candlesData.length}} bars)`;
                    histBadge.style.opacity = '1';
                    clearTimeout(histTimer);
                    histTimer = setTimeout(() => {{ histBadge.style.opacity = '0'; }}, 2500);
                }}
            }});

            // Setup RSI Subchart if enabled
            let rsiChart = null;
            let hmCloudSeries = null;
            let rsiSeries = null;
            if (hasRsi) {{
                const rsiContainer = document.getElementById("rsi_container_{safe_id}");
                const rsiData = {rsi_json};
                const rsiEma3Data = {rsi_ema3_json};
                const rsiWma21Data = {rsi_wma21_json};

                rsiChart = LightweightCharts.createChart(rsiContainer, {{
                    layout: {{
                        background: {{ color: isLightInit ? '#ffffff' : '#131722' }},
                        textColor: isLightInit ? '#131722' : '#94A3B8',
                        fontSize: 10
                    }},
                    grid: {{
                        vertLines: {{ color: isLightInit ? '#f0f3fa' : '#1f2430' }},
                        horzLines: {{ color: isLightInit ? '#f0f3fa' : '#1f2430' }}
                    }},
                    crosshair: {{
                        mode: LightweightCharts.CrosshairMode.Normal,
                        horzLine: {{
                            labelVisible: false
                        }}
                    }},
                    rightPriceScale: {{
                        borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                        autoScale: false,
                        scaleMargins: {{ top: 0.05, bottom: 0.05 }}
                    }},
                    timeScale: {{
                        borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                        timeVisible: {str(is_intraday).lower()},
                        secondsVisible: false,
                        visible: false,
                        rightOffset: 8
                    }},
                    handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                    handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
                }});

                // Baseline Cloud for Hilega Milega (Pink cloud above 50, Soft blue cloud below 50)
                try {{
                    hmCloudSeries = rsiChart.addBaselineSeries({{
                        baseValue: {{ type: 'price', price: 50 }},
                        topFillColor1: isLightInit ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                        topFillColor2: isLightInit ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                        bottomFillColor1: isLightInit ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                        bottomFillColor2: isLightInit ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                        topLineColor: isLightInit ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                        bottomLineColor: isLightInit ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)',
                        lineWidth: 1,
                        lastValueVisible: false,
                        priceLineVisible: false
                    }});
                    if (rsiData && rsiData.length > 0) {{
                        hmCloudSeries.setData(rsiData);
                    }}
                }} catch(e) {{}}

                // 1. Primary RSI(9) Line: Black in Light Mode, White in Dark Mode
                const rsiColor = isLightInit ? '#131722' : '#F8FAFC';
                rsiSeries = rsiChart.addLineSeries({{
                    color: rsiColor,
                    lineWidth: 1.8,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'RSI(9)'
                }});
                rsiSeries.setData(rsiData);

                // 2. Fast EMA 3 on RSI: Exact TradingView Green (#4CAF50)
                let ema3Series = null;
                if (rsiEma3Data && rsiEma3Data.length > 0) {{
                    ema3Series = rsiChart.addLineSeries({{
                        color: '#4CAF50',
                        lineWidth: 1.8,
                        priceLineVisible: false,
                        lastValueVisible: true,
                        title: 'EMA(3)'
                    }});
                    ema3Series.setData(rsiEma3Data);
                }}

                // 3. Slow WMA 21 on RSI: Exact TradingView Red (#FF5252)
                let wma21Series = null;
                if (rsiWma21Data && rsiWma21Data.length > 0) {{
                    wma21Series = rsiChart.addLineSeries({{
                        color: '#FF5252',
                        lineWidth: 1.8,
                        priceLineVisible: false,
                        lastValueVisible: true,
                        title: 'WMA(21)'
                    }});
                    wma21Series.setData(rsiWma21Data);
                }}

                // 4. Exact TradingView Blue 50 Line (#7695F9) with Price Badge
                rsiSeries.createPriceLine({{
                    price: 50,
                    color: '#7695F9',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    axisLabelVisible: true,
                    title: '50'
                }});

                // 5. Reference Dashed Lines for Overbought 70 and Oversold 30
                rsiSeries.createPriceLine({{
                    price: 70,
                    color: '#94A3B8',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: false,
                    title: ''
                }});
                rsiSeries.createPriceLine({{
                    price: 30,
                    color: '#94A3B8',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: false,
                    title: ''
                }});

                // Real-time Header Value Display on Hover
                function updateHmHeader(param) {{
                    const elR = document.getElementById("hm_rsi_val_{safe_id}");
                    const elE = document.getElementById("hm_ema3_val_{safe_id}");
                    const elW = document.getElementById("hm_wma21_val_{safe_id}");
                    if (!elR) return;
                    if (!param || !param.time) {{
                        const lastR = rsiData.length ? rsiData[rsiData.length-1].value : null;
                        const lastE = rsiEma3Data.length ? rsiEma3Data[rsiEma3Data.length-1].value : null;
                        const lastW = rsiWma21Data.length ? rsiWma21Data[rsiWma21Data.length-1].value : null;
                        elR.textContent = lastR !== null ? ("RSI: " + Number(lastR).toFixed(2)) : '';
                        elE.textContent = lastE !== null ? ("EMA(3): " + Number(lastE).toFixed(2)) : '';
                        elW.textContent = lastW !== null ? ("WMA(21): " + Number(lastW).toFixed(2)) : '';
                        return;
                    }}
                    const rVal = param.seriesData.get(rsiSeries);
                    const eVal = ema3Series ? param.seriesData.get(ema3Series) : null;
                    const wVal = wma21Series ? param.seriesData.get(wma21Series) : null;
                    if (rVal && rVal.value !== undefined) {{
                        elR.textContent = "RSI: " + Number(rVal.value).toFixed(2);
                        elR.style.color = (rVal.value >= 50) ? '#089981' : '#F23645';
                    }}
                    if (eVal && eVal.value !== undefined) elE.textContent = "EMA(3): " + Number(eVal.value).toFixed(2);
                    if (wVal && wVal.value !== undefined) elW.textContent = "WMA(21): " + Number(wVal.value).toFixed(2);
                }}
                chart.subscribeCrosshairMove(updateHmHeader);
                rsiChart.subscribeCrosshairMove(updateHmHeader);
                updateHmHeader(null);

                // Sync time scale between Main Price Chart and RSI Chart
                let isSyncing = false;
                chart.timeScale().subscribeVisibleTimeRangeChange(range => {{
                    if (isSyncing || !range) return;
                    isSyncing = true;
                    try {{ rsiChart.timeScale().setVisibleRange(range); }} catch(e) {{}}
                    isSyncing = false;
                }});

                rsiChart.timeScale().subscribeVisibleTimeRangeChange(range => {{
                    if (isSyncing || !range) return;
                    isSyncing = true;
                    try {{ chart.timeScale().setVisibleRange(range); }} catch(e) {{}}
                    isSyncing = false;
                }});

                rsiChart.timeScale().fitContent();
            }}

            // Tooltip handler: update on every candle hover
            const legO = document.getElementById("leg_o_{safe_id}");
            const legH = document.getElementById("leg_h_{safe_id}");
            const legL = document.getElementById("leg_l_{safe_id}");
            const legC = document.getElementById("leg_c_{safe_id}");
            const legChg = document.getElementById("leg_chg_{safe_id}");
            const legV = document.getElementById("leg_v_{safe_id}");
            const chVEl = document.getElementById("ch_v_{safe_id}");

            function handleCrosshair(param) {{
                if (!param || !param.point) {{
                    if (chVEl) chVEl.style.display = "none";
                    tooltip.style.display = "none";
                    return;
                }}

                // 1. Extend vertical crosshair line across the entire canvas height (Price Chart + Hilega Milega)
                if (chVEl) {{
                    chVEl.style.left = param.point.x + "px";
                    chVEl.style.display = "block";
                }}

                if (!param.time) {{
                    tooltip.style.display = "none";
                    return;
                }}

                let candle = param.seriesData ? param.seriesData.get(candleSeries) : null;
                if (!candle && candlesData && candlesData.length) {{
                    const tMatch = (typeof param.time === 'number') ? param.time : (param.time.year ? `${{param.time.year}}-${{String(param.time.month).padStart(2,'0')}}-${{String(param.time.day).padStart(2,'0')}}` : String(param.time));
                    const cPt = candlesData.find(p => p.time === tMatch || p.time === param.time);
                    if (cPt) candle = cPt;
                }}

                if (!candle || candle.open === undefined) {{
                    tooltip.style.display = "none";
                    return;
                }}

                tooltip.style.display = "block";

                // Format timestamp
                let timeStr = "";
                if (typeof param.time === 'number') {{
                    const d = new Date(param.time * 1000);
                    timeStr = d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }}) + " " +
                              d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + " IST";
                }} else if (typeof param.time === 'object') {{
                    timeStr = `${{param.time.year}}-${{String(param.time.month).padStart(2,'0')}}-${{String(param.time.day).padStart(2,'0')}}`;
                }} else {{
                    timeStr = String(param.time);
                }}

                const o = Number(candle.open);
                const h = Number(candle.high);
                const l = Number(candle.low);
                const c = Number(candle.close);
                const diff = c - o;
                const pct = o > 0 ? (diff / o) * 100 : 0;
                const isUp = c >= o;
                const chgColor = isUp ? "#089981" : "#F23645";
                const chgSign = diff >= 0 ? "+" : "";

                // Update Header Bar Legend
                if (legO) legO.innerText = "₹" + o.toFixed(2);
                if (legH) legH.innerText = "₹" + h.toFixed(2);
                if (legL) legL.innerText = "₹" + l.toFixed(2);
                if (legC) {{
                    legC.innerText = "₹" + c.toFixed(2);
                    legC.style.color = chgColor;
                }}
                if (legChg) {{
                    legChg.innerText = `${{chgSign}}${{pct.toFixed(2)}}%`;
                    legChg.style.color = chgColor;
                }}

                const volData = param.seriesData ? param.seriesData.get(volumeSeries) : null;
                if (legV && volData && volData.value !== undefined) {{
                    legV.innerText = Number(volData.value).toLocaleString();
                }}

                // Build Floating Tooltip
                let ttHtml = `
                    <div class="tt-date">${{timeStr}}</div>
                    <div class="tt-row"><span class="tt-lbl">Open:</span><span class="tt-val">₹${{o.toFixed(2)}}</span></div>
                    <div class="tt-row"><span class="tt-lbl">High:</span><span class="tt-val">₹${{h.toFixed(2)}}</span></div>
                    <div class="tt-row"><span class="tt-lbl">Low:</span><span class="tt-val">₹${{l.toFixed(2)}}</span></div>
                    <div class="tt-row"><span class="tt-lbl">Close:</span><span class="tt-val" style="color:${{chgColor}}">₹${{c.toFixed(2)}} (${{chgSign}}${{pct.toFixed(2)}}%)</span></div>
                `;

                if (volData && volData.value !== undefined) {{
                    ttHtml += `<div class="tt-row"><span class="tt-lbl">Volume:</span><span class="tt-val">${{Number(volData.value).toLocaleString()}}</span></div>`;
                }}

                // Overlay active EMAs & Indicators
                for (const [colName, seriesInfo] of Object.entries(emaSeriesMap)) {{
                    const emaVal = param.seriesData ? param.seriesData.get(seriesInfo.series) : null;
                    if (emaVal && emaVal.value !== undefined) {{
                        const dispName = seriesInfo.title || colName;
                        ttHtml += `<div class="tt-row"><span class="tt-lbl" style="color:${{seriesInfo.color}}">${{dispName}}:</span><span class="tt-val">₹${{Number(emaVal.value).toFixed(2)}}</span></div>`;
                    }}
                }}

                tooltip.innerHTML = ttHtml;

                // Position tooltip relative to wrapper (accounting for 38px toolbar)
                const headerOffset = 36;
                const drawBarOffset = 38;
                const ttW = 205;
                const ttH = tooltip.offsetHeight || 150;
                let left = param.point.x + drawBarOffset + 18;
                let top = param.point.y + headerOffset + 18;

                if (left + ttW > wrapper.clientWidth - 10) {{
                    left = param.point.x + drawBarOffset - ttW - 18;
                }}
                if (top + ttH > wrapper.clientHeight - 10) {{
                    top = param.point.y + headerOffset - ttH - 18;
                }}

                tooltip.style.left = Math.max(8, left) + "px";
                tooltip.style.top = Math.max(headerOffset + 5, top) + "px";
            }}

            chart.subscribeCrosshairMove(handleCrosshair);
            if (rsiChart) {{
                rsiChart.subscribeCrosshairMove(handleCrosshair);
            }}
            if (wrapper) {{
                wrapper.addEventListener('mouseleave', () => {{
                    if (chVEl) chVEl.style.display = 'none';
                    if (tooltip) tooltip.style.display = 'none';
                }});
            }}

            // --- TRADING TERMINAL DRAWING ENGINE ---
            const mainPane = document.getElementById("main_pane_{safe_id}");
            const drawCanvas = document.getElementById("draw_canvas_{safe_id}");
            const drawCtx = drawCanvas ? drawCanvas.getContext("2d") : null;
            const drawToolbar = document.getElementById("draw_toolbar_{safe_id}");
            const colorInput = document.getElementById("draw_color_{safe_id}");
            const colorDot = document.getElementById("color_dot_{safe_id}");
            const undoBtn = document.getElementById("draw_undo_{safe_id}");
            const clearBtn = document.getElementById("draw_clear_{safe_id}");

            let activeTool = 'pointer';
            let activeColor = '#00E5FF';
            let drawings = [];
            let isDrawing = false;
            let drawStartPoint = null;
            let tempCurrentPoint = null;
            let brushPoints = [];

            function screenToChart(px, py) {{
                let logical = 0;
                let price = 0;
                try {{
                    const l = chart.timeScale().coordinateToLogical(px);
                    if (l !== null && !isNaN(l)) logical = l;
                }} catch(e) {{}}
                try {{
                    const p = candleSeries.coordinateToPrice(py);
                    if (p !== null && !isNaN(p)) price = p;
                }} catch(e) {{}}
                return {{ logical: logical, price: price, origX: px, origY: py }};
            }}

            function chartToScreen(pt) {{
                let x = pt.origX;
                let y = pt.origY;
                try {{
                    const cx = chart.timeScale().logicalToCoordinate(pt.logical);
                    if (cx !== null && !isNaN(cx)) x = cx;
                }} catch(e) {{}}
                try {{
                    const cy = candleSeries.priceToCoordinate(pt.price);
                    if (cy !== null && !isNaN(cy)) y = cy;
                }} catch(e) {{}}
                return {{ x: x, y: y }};
            }}

            function resizeDrawLayer() {{
                if (!drawCanvas || !mainPane) return;
                const dpr = window.devicePixelRatio || 1;
                const w = mainPane.clientWidth;
                const h = mainPane.clientHeight;
                drawCanvas.width = w * dpr;
                drawCanvas.height = h * dpr;
                drawCanvas.style.width = w + "px";
                drawCanvas.style.height = h + "px";
                if (drawCtx) {{
                    drawCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
                }}
                renderDrawings();
            }}

            function hexToRgba(hex, alpha) {{
                let c = hex.replace('#', '');
                if (c.length === 3) {{
                    c = c.split('').map(x => x + x).join('');
                }}
                const num = parseInt(c, 16);
                const r = (num >> 16) & 255;
                const g = (num >> 8) & 255;
                const b = num & 255;
                return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
            }}

            function drawRoundedRect(ctx, x, y, w, h, r) {{
                ctx.beginPath();
                ctx.moveTo(x + r, y);
                ctx.arcTo(x + w, y, x + w, y + h, r);
                ctx.arcTo(x + w, y + h, x, y + h, r);
                ctx.arcTo(x, y + h, x, y, r);
                ctx.arcTo(x, y, x + w, y, r);
                ctx.closePath();
            }}

            function drawPriceBadge(ctx, price, y, color) {{
                const w = mainPane ? mainPane.clientWidth : drawCanvas.width;
                if (y < 0 || y > (mainPane ? mainPane.clientHeight : drawCanvas.height)) return;
                const priceStr = "₹" + Number(price).toFixed(2);
                ctx.save();
                ctx.font = "bold 10px 'SF Mono', Consolas, Monaco, monospace";
                const textW = ctx.measureText(priceStr).width;
                const badgeW = textW + 12;
                const badgeH = 18;
                const badgeX = w - badgeW - 55;
                const badgeY = y - badgeH / 2;

                ctx.fillStyle = color;
                drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
                ctx.fill();

                ctx.fillStyle = "#ffffff";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(priceStr, badgeX + badgeW / 2, badgeY + badgeH / 2);
                ctx.restore();
            }}

            function drawSingleItem(ctx, item, w, h) {{
                ctx.save();
                const color = item.color || '#00E5FF';
                ctx.strokeStyle = color;
                ctx.fillStyle = color;
                ctx.lineWidth = item.width || 2;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                if (item.type === 'trendline') {{
                    const s1 = chartToScreen(item.p1);
                    const s2 = chartToScreen(item.p2);
                    ctx.beginPath();
                    ctx.moveTo(s1.x, s1.y);
                    ctx.lineTo(s2.x, s2.y);
                    ctx.stroke();

                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(s1.x, s1.y, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(s2.x, s2.y, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                }} else if (item.type === 'horizontal') {{
                    let y = item.origY;
                    try {{
                        const cy = candleSeries.priceToCoordinate(item.price);
                        if (cy !== null && !isNaN(cy)) y = cy;
                    }} catch(e) {{}}
                    ctx.setLineDash([5, 4]);
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(w, y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    drawPriceBadge(ctx, item.price, y, color);
                }} else if (item.type === 'ray') {{
                    const s1 = chartToScreen(item.p1);
                    ctx.setLineDash([5, 3]);
                    ctx.beginPath();
                    ctx.moveTo(s1.x, s1.y);
                    ctx.lineTo(w, s1.y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(s1.x, s1.y, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                    drawPriceBadge(ctx, item.p1.price, s1.y, color);
                }} else if (item.type === 'rectangle') {{
                    const s1 = chartToScreen(item.p1);
                    const s2 = chartToScreen(item.p2);
                    const rx = Math.min(s1.x, s2.x);
                    const ry = Math.min(s1.y, s2.y);
                    const rw = Math.abs(s2.x - s1.x);
                    const rh = Math.abs(s2.y - s1.y);

                    ctx.fillStyle = hexToRgba(color, 0.18);
                    ctx.fillRect(rx, ry, rw, rh);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(rx, ry, rw, rh);

                    ctx.font = "9px sans-serif";
                    ctx.fillStyle = color;
                    ctx.fillText("ZONE", rx + 4, ry + 12);
                }} else if (item.type === 'fib') {{
                    const s1 = chartToScreen(item.p1);
                    const s2 = chartToScreen(item.p2);
                    const pDiff = item.p2.price - item.p1.price;
                    const minX = Math.min(s1.x, s2.x);
                    const maxX = Math.max(w - 60, Math.max(s1.x, s2.x) + 40);
                    const fibLevels = [
                        {{ r: 0.0, label: "0.0% (₹" }},
                        {{ r: 0.236, label: "23.6% (₹" }},
                        {{ r: 0.382, label: "38.2% (₹" }},
                        {{ r: 0.500, label: "50.0% (₹" }},
                        {{ r: 0.618, label: "61.8% (₹" }},
                        {{ r: 0.786, label: "78.6% (₹" }},
                        {{ r: 1.0, label: "100.0% (₹" }}
                    ];

                    const p382 = item.p1.price + 0.382 * pDiff;
                    const p618 = item.p1.price + 0.618 * pDiff;
                    let y382 = s1.y, y618 = s2.y;
                    try {{
                        const c1 = candleSeries.priceToCoordinate(p382);
                        const c2 = candleSeries.priceToCoordinate(p618);
                        if (c1 !== null) y382 = c1;
                        if (c2 !== null) y618 = c2;
                    }} catch(e) {{}}
                    const gyMin = Math.min(y382, y618);
                    const gyH = Math.abs(y618 - y382);
                    ctx.fillStyle = "rgba(255, 215, 0, 0.12)";
                    ctx.fillRect(minX, gyMin, maxX - minX, gyH);

                    for (const fib of fibLevels) {{
                        const lvlPrice = item.p1.price + fib.r * pDiff;
                        let ly = s1.y;
                        try {{
                            const c = candleSeries.priceToCoordinate(lvlPrice);
                            if (c !== null && !isNaN(c)) ly = c;
                        }} catch(e) {{}}

                        ctx.strokeStyle = fib.r === 0.618 || fib.r === 0.5 ? '#FACC15' : color;
                        ctx.lineWidth = fib.r === 0.618 || fib.r === 0.5 ? 1.5 : 1;
                        ctx.beginPath();
                        ctx.moveTo(minX, ly);
                        ctx.lineTo(maxX, ly);
                        ctx.stroke();

                        ctx.font = "bold 9px 'SF Mono', Consolas, Monaco, monospace";
                        ctx.fillStyle = fib.r === 0.618 || fib.r === 0.5 ? '#FACC15' : color;
                        ctx.fillText(fib.label + lvlPrice.toFixed(2) + ")", minX + 5, ly - 3);
                    }}
                }} else if (item.type === 'brush') {{
                    if (!item.points || item.points.length < 2) return;
                    ctx.beginPath();
                    const s0 = chartToScreen(item.points[0]);
                    ctx.moveTo(s0.x, s0.y);
                    for (let i = 1; i < item.points.length; i++) {{
                        const sp = chartToScreen(item.points[i]);
                        ctx.lineTo(sp.x, sp.y);
                    }}
                    ctx.stroke();
                }} else if (item.type === 'text') {{
                    const s1 = chartToScreen(item.p1);
                    ctx.font = "bold 11px sans-serif";
                    const tw = ctx.measureText(item.text).width;
                    const th = 20;
                    const bx = s1.x + 5;
                    const by = s1.y - th / 2;

                    ctx.fillStyle = currentTheme === 'light' ? "rgba(255,255,255,0.92)" : "rgba(30,34,45,0.92)";
                    drawRoundedRect(ctx, bx, by, tw + 14, th, 4);
                    ctx.fill();

                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    ctx.fillStyle = color;
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText(item.text, bx + 7, s1.y);
                }}
                ctx.restore();
            }}

            function renderDrawings() {{
                if (!drawCtx || !drawCanvas || !mainPane) return;
                const w = mainPane.clientWidth;
                const h = mainPane.clientHeight;
                drawCtx.clearRect(0, 0, w, h);

                for (const d of drawings) {{
                    drawSingleItem(drawCtx, d, w, h);
                }}

                if (isDrawing && drawStartPoint && tempCurrentPoint) {{
                    const previewItem = {{
                        type: activeTool,
                        p1: drawStartPoint,
                        p2: tempCurrentPoint,
                        color: activeColor,
                        width: 2,
                        isPreview: true
                    }};
                    drawSingleItem(drawCtx, previewItem, w, h);
                }} else if (isDrawing && activeTool === 'brush' && brushPoints.length > 1) {{
                    const previewItem = {{
                        type: 'brush',
                        points: brushPoints,
                        color: activeColor,
                        width: 2,
                        isPreview: true
                    }};
                    drawSingleItem(drawCtx, previewItem, w, h);
                }}
            }}

            function setTool(toolName) {{
                activeTool = toolName;
                const toolBtns = drawToolbar ? drawToolbar.querySelectorAll(".tv-tool-btn[data-tool]") : [];
                toolBtns.forEach(btn => {{
                    btn.classList.toggle("active", btn.getAttribute("data-tool") === toolName);
                }});

                if (activeTool === 'pointer') {{
                    drawCanvas.style.pointerEvents = 'none';
                    drawCanvas.style.cursor = 'default';
                }} else {{
                    drawCanvas.style.pointerEvents = 'auto';
                    drawCanvas.style.cursor = 'crosshair';
                }}
            }}

            if (drawToolbar) {{
                drawToolbar.addEventListener("click", (e) => {{
                    const btn = e.target.closest(".tv-tool-btn[data-tool]");
                    if (btn) {{
                        const t = btn.getAttribute("data-tool");
                        setTool(t);
                    }}
                }});
            }}

            if (colorInput) {{
                colorInput.addEventListener("input", (e) => {{
                    activeColor = e.target.value;
                    if (colorDot) colorDot.style.backgroundColor = activeColor;
                }});
            }}

            if (undoBtn) {{
                undoBtn.addEventListener("click", () => {{
                    if (drawings.length > 0) {{
                        drawings.pop();
                        renderDrawings();
                    }}
                }});
            }}

            if (clearBtn) {{
                clearBtn.addEventListener("click", () => {{
                    if (drawings.length > 0 && confirm("Clear all drawings?")) {{
                        drawings = [];
                        renderDrawings();
                    }}
                }});
            }}

            if (drawCanvas) {{
                drawCanvas.addEventListener("mousedown", (e) => {{
                    if (activeTool === 'pointer') return;
                    const rect = drawCanvas.getBoundingClientRect();
                    const px = e.clientX - rect.left;
                    const py = e.clientY - rect.top;

                    if (activeTool === 'horizontal') {{
                        const pt = screenToChart(px, py);
                        drawings.push({{ type: 'horizontal', price: pt.price, origY: py, color: activeColor, width: 2 }});
                        renderDrawings();
                        return;
                    }}

                    if (activeTool === 'ray') {{
                        const pt = screenToChart(px, py);
                        drawings.push({{ type: 'ray', p1: pt, color: activeColor, width: 2 }});
                        renderDrawings();
                        return;
                    }}

                    if (activeTool === 'text') {{
                        const pt = screenToChart(px, py);
                        const textVal = prompt("Enter chart annotation:", "Key Level");
                        if (textVal && textVal.trim()) {{
                            drawings.push({{ type: 'text', p1: pt, text: textVal.trim(), color: activeColor }});
                            renderDrawings();
                        }}
                        return;
                    }}

                    if (activeTool === 'brush') {{
                        isDrawing = true;
                        brushPoints = [screenToChart(px, py)];
                        renderDrawings();
                        return;
                    }}

                    if (['trendline', 'rectangle', 'fib'].includes(activeTool)) {{
                        isDrawing = true;
                        drawStartPoint = screenToChart(px, py);
                        tempCurrentPoint = drawStartPoint;
                        renderDrawings();
                    }}
                }});

                drawCanvas.addEventListener("mousemove", (e) => {{
                    if (!isDrawing) return;
                    const rect = drawCanvas.getBoundingClientRect();
                    const px = e.clientX - rect.left;
                    const py = e.clientY - rect.top;

                    if (activeTool === 'brush') {{
                        brushPoints.push(screenToChart(px, py));
                        renderDrawings();
                    }} else if (['trendline', 'rectangle', 'fib'].includes(activeTool)) {{
                        tempCurrentPoint = screenToChart(px, py);
                        renderDrawings();
                    }}
                }});

                window.addEventListener("mouseup", (e) => {{
                    if (!isDrawing) return;
                    const rect = drawCanvas.getBoundingClientRect();
                    const px = e.clientX - rect.left;
                    const py = e.clientY - rect.top;

                    if (activeTool === 'brush') {{
                        if (brushPoints.length > 1) {{
                            drawings.push({{ type: 'brush', points: brushPoints, color: activeColor, width: 2 }});
                        }}
                        brushPoints = [];
                    }} else if (['trendline', 'rectangle', 'fib'].includes(activeTool)) {{
                        if (drawStartPoint) {{
                            const endPt = screenToChart(px, py);
                            const dist = Math.hypot(px - drawStartPoint.origX, py - drawStartPoint.origY);
                            if (dist > 5) {{
                                drawings.push({{
                                    type: activeTool,
                                    p1: drawStartPoint,
                                    p2: endPt,
                                    color: activeColor,
                                    width: 2
                                }});
                            }}
                        }}
                    }}

                    isDrawing = false;
                    drawStartPoint = null;
                    tempCurrentPoint = null;
                    renderDrawings();
                }});

                window.addEventListener("keydown", (e) => {{
                    if (e.key === "Escape") {{
                        if (isDrawing) {{
                            isDrawing = false;
                            drawStartPoint = null;
                            tempCurrentPoint = null;
                            brushPoints = [];
                            renderDrawings();
                        }} else {{
                            setTool('pointer');
                        }}
                    }}
                    if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {{
                        if (drawings.length > 0) {{
                            drawings.pop();
                            renderDrawings();
                        }}
                    }}
                }});
            }}

            // Continuous price scale & time scale sync engine
            // Keeps drawings mathematically locked to candles and price coordinates when dragging Y-axis or zooming
            let lastSingleScaleState = null;
            function syncSingleChartDrawings() {{
                if (!chart || !candleSeries) return;
                const h = mainContainer ? (mainContainer.clientHeight || 400) : 400;
                const pTop = candleSeries.coordinateToPrice(0);
                const pBot = candleSeries.coordinateToPrice(h);

                let rangeStr = '';
                try {{
                    const r = chart.timeScale().getVisibleLogicalRange();
                    if (r && r.from !== null && r.to !== null) {{
                        rangeStr = r.from.toFixed(2) + ':' + r.to.toFixed(2);
                    }}
                }} catch(e) {{}}

                if (!lastSingleScaleState || lastSingleScaleState.pTop !== pTop || lastSingleScaleState.pBot !== pBot || lastSingleScaleState.rangeStr !== rangeStr) {{
                    lastSingleScaleState = {{ pTop, pBot, rangeStr }};
                    renderDrawings();
                }}
                requestAnimationFrame(syncSingleChartDrawings);
            }}
            requestAnimationFrame(syncSingleChartDrawings);

            chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                renderDrawings();
            }});

            if (mainContainer) {{
                mainContainer.addEventListener('pointermove', (e) => {{
                    if (e.buttons > 0) renderDrawings();
                }});
                mainContainer.addEventListener('wheel', () => {{
                    renderDrawings();
                }}, {{ passive: true }});
            }}

            // Fullscreen functionality
            const fsBtn = document.getElementById("fs_btn_{safe_id}");
            function resizeChart() {{
                const isFs = document.fullscreenElement === wrapper || wrapper.classList.contains("css-fullscreen");
                const drawBarW = 38;
                const newW = wrapper.clientWidth - drawBarW;
                const totalH = isFs ? window.innerHeight : {height};
                const headH = 36;
                const availH = totalH - headH;
                const newMainH = hasRsi ? Math.max(180, Math.floor(availH * 0.70)) : availH;
                const newRsiH = hasRsi ? Math.max(80, Math.floor(availH * 0.30)) : 0;

                if (mainPane) mainPane.style.height = newMainH + "px";
                mainContainer.style.height = newMainH + "px";
                chart.applyOptions({{ width: newW, height: newMainH }});
                chart.timeScale().fitContent();

                if (hasRsi && rsiChart) {{
                    const rsiContainer = document.getElementById("rsi_container_{safe_id}");
                    if (rsiContainer) rsiContainer.style.height = newRsiH + "px";
                    rsiChart.applyOptions({{ width: newW, height: newRsiH }});
                    rsiChart.timeScale().fitContent();
                }}
                resizeDrawLayer();
            }}

            function toggleFullScreen() {{
                const isFs = document.fullscreenElement === wrapper || wrapper.classList.contains("css-fullscreen");
                if (!isFs) {{
                    if (wrapper.requestFullscreen) {{
                        wrapper.requestFullscreen().then(() => {{
                            fsBtn.innerHTML = "✖ Exit";
                            setTimeout(resizeChart, 60);
                        }}).catch(() => {{
                            wrapper.classList.add("css-fullscreen");
                            fsBtn.innerHTML = "✖ Exit";
                            setTimeout(resizeChart, 60);
                        }});
                    }} else {{
                        wrapper.classList.add("css-fullscreen");
                        fsBtn.innerHTML = "✖ Exit";
                        setTimeout(resizeChart, 60);
                    }}
                }} else {{
                    if (document.fullscreenElement) {{
                        document.exitFullscreen().catch(() => {{}});
                    }}
                    wrapper.classList.remove("css-fullscreen");
                    fsBtn.innerHTML = "⛶ Fullscreen";
                    setTimeout(resizeChart, 60);
                }}
            }}

            if (fsBtn) {{
                fsBtn.addEventListener("click", toggleFullScreen);
            }}

            document.addEventListener("fullscreenchange", () => {{
                if (document.fullscreenElement === wrapper) {{
                    fsBtn.innerHTML = "✖ Exit";
                }} else {{
                    fsBtn.innerHTML = "⛶ Fullscreen";
                    wrapper.classList.remove("css-fullscreen");
                }}
                setTimeout(resizeChart, 60);
            }});

            // Dynamic In-Canvas Theme Switching
            const themeBtn = document.getElementById("theme_btn_{safe_id}");
            function applyTheme(newTheme) {{
                currentTheme = newTheme;
                const isLight = (newTheme === 'light');
                document.body.classList.toggle("light-theme", isLight);
                wrapper.classList.toggle("light-theme", isLight);
                if (themeBtn) {{
                    themeBtn.innerHTML = isLight ? "☀️ Light" : "🌙 Dark";
                }}

                const bgCol = isLight ? '#ffffff' : '#131722';
                const txtCol = isLight ? '#131722' : '#94A3B8';
                const gridCol = isLight ? '#f0f3fa' : '#1f2430';
                const borderCol = isLight ? '#e0e3eb' : '#2A2E39';

                chart.applyOptions({{
                    layout: {{ background: {{ color: bgCol }}, textColor: txtCol }},
                    grid: {{ vertLines: {{ color: gridCol }}, horzLines: {{ color: gridCol }} }},
                    rightPriceScale: {{ borderColor: borderCol }},
                    timeScale: {{ borderColor: borderCol }}
                }});

                if (rsiChart) {{
                    rsiChart.applyOptions({{
                        layout: {{ background: {{ color: bgCol }}, textColor: txtCol }},
                        grid: {{ vertLines: {{ color: gridCol }}, horzLines: {{ color: gridCol }} }},
                        rightPriceScale: {{ borderColor: borderCol }},
                        timeScale: {{ borderColor: borderCol }}
                    }});
                }}
                if (hmCloudSeries) {{
                    hmCloudSeries.applyOptions({{
                        topFillColor1: isLight ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                        topFillColor2: isLight ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                        bottomFillColor1: isLight ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                        bottomFillColor2: isLight ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                        topLineColor: isLight ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                        bottomLineColor: isLight ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)'
                    }});
                }}
                if (rsiSeries) {{
                    rsiSeries.applyOptions({{
                        color: isLight ? '#131722' : '#F8FAFC'
                    }});
                }}
                renderDrawings();
            }}

            if (themeBtn) {{
                themeBtn.addEventListener("click", () => {{
                    applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
                }});
            }}

            window.addEventListener('resize', () => {{
                setTimeout(resizeChart, 50);
            }});

            setTimeout(resizeChart, 20);
        }})();
    </script>
</body>
</html>"""
    return html_code


def render_tradingview_cloud_widget(
    symbol: str,
    interval: str = "D",
    height: int = 650,
    theme: str = "light"
) -> str:
    """
    Renders the official TradingView Advanced Real-Time Chart widget.
    Supports Indian NSE equities (e.g. NSE:BAJFINANCE, NSE:RELIANCE).
    """
    clean_sym = symbol.upper().replace("-EQ", "").replace(".NS", "")
    tv_symbol = f"NSE:{clean_sym}"
    is_light = (str(theme).lower() == "light")
    bg_col = "#ffffff" if is_light else "#131722"
    tb_bg = "#f0f3fa" if is_light else "#131722"
    border_col = "#e0e3eb" if is_light else "#2A2E39"
    widget_theme = "light" if is_light else "dark"

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: {bg_col};
            overflow: hidden;
        }}
        .tradingview-widget-container {{
            width: 100%;
            height: {height}px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {border_col};
        }}
    </style>
</head>
<body>
    <div class="tradingview-widget-container">
        <div id="tradingview_widget_{clean_sym}" style="width: 100%; height: 100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "{interval}",
            "timezone": "Asia/Kolkata",
            "theme": "{widget_theme}",
            "style": "1",
            "locale": "in",
            "toolbar_bg": "{tb_bg}",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "hide_side_toolbar": false,
            "withdateranges": true,
            "studies": [
                "RSI@tv-basicstudies",
                "MASimple@tv-basicstudies"
            ],
            "container_id": "tradingview_widget_{clean_sym}"
        }});
        </script>
    </div>
</body>
</html>"""
    return html_code


def generate_advanced_terminal_html(
    df: pd.DataFrame,
    symbol: str,
    timeframe_name: str,
    indicators: dict = None,
    chart_type: str = "candlestick",
    height: int = 720,
    is_intraday: bool = False,
    chart_id: str = "tv_adv_terminal",
    theme: str = "light"
) -> str:
    """
    Renders an Advanced Desktop-Grade TradingView Terminal using Lightweight Charts v4.
    Supports:
    - Interactive Zoom In (+), Zoom Out (-), Reset View (⟲), Fullscreen (⛶)
    - Bottom Timeframe Range Bar: [1D] [5D] [1M] [3M] [6M] [1Y] [ALL]
    - Chart Types: Candlesticks, Hollow Candles, Line, Area
    - Indicators:
      * Overlays: EMAs (9, 20, 50, 200), SMAs (20, 50, 200), Bollinger Bands (20, 2),
                  Supertrend (10, 3), VWAP (intraday), CPR / Weekly Pivots, Volume + Volume MA(20)
      * Sub-Panels: RSI (14 with 70/30/50 bands + EMA3/WMA21), MACD (12, 26, 9 with colored histogram),
                    Stochastic (14, 3, 3 with 80/20 bands)
    - Synchronized time scales across all panes
    - Full right-axis price badges and crosshair readings
    - Left Drawing Toolbar (Trendlines, Horizontal lines, Rays, Boxes, Fib, Brush, Text)
    """
    import scanner

    indicators = indicators or {}
    show_ema = indicators.get("ema", True)
    show_sma = indicators.get("sma", False)
    show_bb = indicators.get("bb", False)
    show_supertrend = indicators.get("supertrend", False)
    show_vwap = indicators.get("vwap", False) and is_intraday
    show_cpr = indicators.get("cpr", False)
    show_volume = indicators.get("volume", True)
    show_rsi = indicators.get("rsi", True)
    show_macd = indicators.get("macd", False)
    show_stoch = indicators.get("stoch", False)

    is_light = (str(theme).lower() == "light")
    bg_init = "#ffffff" if is_light else "#131722"
    txt_init = "#787B86"

    if df.empty:
        return f"""<div style="height:{height}px; background:{bg_init}; color:{txt_init}; display:flex; align-items:center; justify-content:center; font-family:sans-serif; border-radius:8px; border:1px solid {'#e0e3eb' if is_light else '#2A2E39'};">No chart data available for {symbol}</div>"""

    df_calc = df.copy()
    if not isinstance(df_calc.index, pd.DatetimeIndex):
        df_calc.index = pd.to_datetime(df_calc.index)
    df_calc.sort_index(inplace=True)
    df_calc = df_calc[~df_calc.index.duplicated(keep="last")]

    # Calculate indicators if not already in df_calc
    if show_ema:
        for span, col in [(5, "EMA_5"), (9, "EMA_9"), (20, "EMA_20"), (50, "EMA_50"), (200, "EMA_200")]:
            if col not in df_calc.columns:
                df_calc[col] = df_calc["close"].ewm(span=span, adjust=False).mean()

    if show_sma:
        for period, col in [(20, "SMA_20"), (50, "SMA_50"), (200, "SMA_200")]:
            if col not in df_calc.columns:
                df_calc[col] = df_calc["close"].rolling(window=period, min_periods=1).mean()

    if show_bb:
        if "BB_Upper" not in df_calc.columns:
            bb_res = scanner.calculate_bollinger_bands(df_calc["close"], period=20, num_std=2.0)
            df_calc["BB_Upper"] = bb_res["BB_Upper"]
            df_calc["BB_Middle"] = bb_res["BB_Middle"]
            df_calc["BB_Lower"] = bb_res["BB_Lower"]

    if show_supertrend:
        if "Supertrend" not in df_calc.columns:
            st_res = scanner.calculate_supertrend(df_calc, period=10, multiplier=3.0)
            df_calc["Supertrend"] = st_res["Supertrend"]
            df_calc["Trend_Direction"] = st_res["Trend_Direction"]

    if show_vwap:
        if "VWAP" not in df_calc.columns:
            df_calc["VWAP"] = scanner.calculate_vwap(df_calc)

    if show_cpr:
        if "CPR_Pivot" not in df_calc.columns:
            cpr_res = scanner.calculate_cpr(df_calc)
            for c in cpr_res.columns:
                df_calc[c] = cpr_res[c]

    if show_rsi:
        if "RSI" not in df_calc.columns:
            df_calc["RSI"] = scanner.calculate_rsi(df_calc["close"], span=14)
            df_calc["RSI_EMA3"] = scanner.calculate_ema(df_calc["RSI"], span=3)
            df_calc["RSI_WMA21"] = scanner.calculate_wma(df_calc["RSI"], period=21)

    if show_macd:
        if "MACD_Line" not in df_calc.columns:
            macd_res = scanner.calculate_macd(df_calc["close"], fast=12, slow=26, signal=9)
            df_calc["MACD_Line"] = macd_res["MACD_Line"]
            df_calc["MACD_Signal"] = macd_res["MACD_Signal"]
            df_calc["MACD_Hist"] = macd_res["MACD_Hist"]

    if show_stoch:
        if "Stoch_K" not in df_calc.columns:
            stoch_res = scanner.calculate_stochastic(df_calc, period=14, smooth_k=3, smooth_d=3)
            df_calc["Stoch_K"] = stoch_res["Stoch_K"]
            df_calc["Stoch_D"] = stoch_res["Stoch_D"]

    if show_volume and "volume" in df_calc.columns:
        if "Volume_MA20" not in df_calc.columns:
            df_calc["Volume_MA20"] = df_calc["volume"].rolling(window=20, min_periods=1).mean()

    # Extract clean JSON points
    candles = []
    volumes = []
    vol_mas = []
    emas_data = {"EMA_5": [], "EMA_9": [], "EMA_20": [], "EMA_50": [], "EMA_200": []}
    smas_data = {"SMA_20": [], "SMA_50": [], "SMA_200": []}
    bb_upper, bb_middle, bb_lower = [], [], []
    supertrend_pts = []
    vwap_pts = []
    cpr_data = {"Pivot": [], "BC": [], "TC": [], "R1": [], "S1": []}
    rsi_pts, rsi_ema3_pts, rsi_wma21_pts = [], [], []
    macd_pts, signal_pts, hist_pts = [], [], []
    stoch_k_pts, stoch_d_pts = [], []

    for dt, row in df_calc.iterrows():
        if is_intraday:
            dt_ist = dt.tz_localize("Asia/Kolkata") if dt.tzinfo is None else dt.tz_convert("Asia/Kolkata")
            t_val = int(dt_ist.timestamp())
        else:
            t_val = dt.strftime("%Y-%m-%d")

        o = float(row.get("open", 0))
        h = float(row.get("high", 0))
        l = float(row.get("low", 0))
        c = float(row.get("close", 0))
        v = float(row.get("volume", 0)) if "volume" in row and not pd.isna(row["volume"]) else 0

        if o > 0 and h > 0 and l > 0 and c > 0:
            candles.append({"time": t_val, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2)})
            v_color = "rgba(8, 153, 129, 0.45)" if c >= o else "rgba(242, 54, 69, 0.45)"
            volumes.append({"time": t_val, "value": round(v, 2), "color": v_color})

            if show_volume and "Volume_MA20" in row and not pd.isna(row["Volume_MA20"]):
                vol_mas.append({"time": t_val, "value": round(float(row["Volume_MA20"]), 2)})

            if show_ema:
                for k in emas_data:
                    if k in row and not pd.isna(row[k]) and row[k] > 0:
                        emas_data[k].append({"time": t_val, "value": round(float(row[k]), 2)})

            if show_sma:
                for k in smas_data:
                    if k in row and not pd.isna(row[k]) and row[k] > 0:
                        smas_data[k].append({"time": t_val, "value": round(float(row[k]), 2)})

            if show_bb and "BB_Upper" in row and not pd.isna(row["BB_Upper"]):
                bb_upper.append({"time": t_val, "value": round(float(row["BB_Upper"]), 2)})
                bb_middle.append({"time": t_val, "value": round(float(row["BB_Middle"]), 2)})
                bb_lower.append({"time": t_val, "value": round(float(row["BB_Lower"]), 2)})

            if show_supertrend and "Supertrend" in row and not pd.isna(row["Supertrend"]):
                st_val = round(float(row["Supertrend"]), 2)
                st_dir = int(row.get("Trend_Direction", 1))
                supertrend_pts.append({
                    "time": t_val,
                    "value": st_val,
                    "color": "#10B981" if st_dir == 1 else "#EF4444"
                })

            if show_vwap and "VWAP" in row and not pd.isna(row["VWAP"]):
                vwap_pts.append({"time": t_val, "value": round(float(row["VWAP"]), 2)})

            if show_cpr and "CPR_Pivot" in row and not pd.isna(row["CPR_Pivot"]):
                cpr_data["Pivot"].append({"time": t_val, "value": round(float(row["CPR_Pivot"]), 2)})
                cpr_data["BC"].append({"time": t_val, "value": round(float(row.get("CPR_BC", 0)), 2)})
                cpr_data["TC"].append({"time": t_val, "value": round(float(row.get("CPR_TC", 0)), 2)})
                cpr_data["R1"].append({"time": t_val, "value": round(float(row.get("CPR_R1", 0)), 2)})
                cpr_data["S1"].append({"time": t_val, "value": round(float(row.get("CPR_S1", 0)), 2)})

            if show_rsi and "RSI" in row and not pd.isna(row["RSI"]):
                rsi_pts.append({"time": t_val, "value": round(float(row["RSI"]), 2)})
                if "RSI_EMA3" in row and not pd.isna(row["RSI_EMA3"]):
                    rsi_ema3_pts.append({"time": t_val, "value": round(float(row["RSI_EMA3"]), 2)})
                if "RSI_WMA21" in row and not pd.isna(row["RSI_WMA21"]):
                    rsi_wma21_pts.append({"time": t_val, "value": round(float(row["RSI_WMA21"]), 2)})

            if show_macd and "MACD_Line" in row and not pd.isna(row["MACD_Line"]):
                m_line = round(float(row["MACD_Line"]), 2)
                s_line = round(float(row.get("MACD_Signal", 0)), 2)
                h_val = round(float(row.get("MACD_Hist", 0)), 2)
                macd_pts.append({"time": t_val, "value": m_line})
                signal_pts.append({"time": t_val, "value": s_line})
                h_color = "rgba(8, 153, 129, 0.75)" if h_val >= 0 else "rgba(242, 54, 69, 0.75)"
                hist_pts.append({"time": t_val, "value": h_val, "color": h_color})

            if show_stoch and "Stoch_K" in row and not pd.isna(row["Stoch_K"]):
                stoch_k_pts.append({"time": t_val, "value": round(float(row["Stoch_K"]), 2)})
                stoch_d_pts.append({"time": t_val, "value": round(float(row.get("Stoch_D", 50)), 2)})

    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', str(chart_id))

    # Calculate pane heights
    sub_count = sum([1 for p in [show_rsi and rsi_pts, show_macd and macd_pts, show_stoch and stoch_k_pts] if p])
    header_h = 42
    range_bar_h = 36
    fixed_overhead = header_h + range_bar_h
    avail_h = max(400, height - fixed_overhead)
    sub_pane_h = 130 if sub_count > 0 else 0
    main_h = max(260, avail_h - (sub_count * sub_pane_h))

    candles_json = json.dumps(candles)
    volumes_json = json.dumps(volumes)
    vol_ma_json = json.dumps(vol_mas)
    emas_json = json.dumps(emas_data)
    smas_json = json.dumps(smas_data)
    bb_json = json.dumps({"upper": bb_upper, "middle": bb_middle, "lower": bb_lower})
    supertrend_json = json.dumps(supertrend_pts)
    vwap_json = json.dumps(vwap_pts)
    cpr_json = json.dumps(cpr_data)
    rsi_json = json.dumps(rsi_pts)
    rsi_ema3_json = json.dumps(rsi_ema3_pts)
    rsi_wma21_json = json.dumps(rsi_wma21_pts)
    macd_json = json.dumps({"macd": macd_pts, "signal": signal_pts, "hist": hist_pts})
    stoch_json = json.dumps({"k": stoch_k_pts, "d": stoch_d_pts})

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{symbol} {timeframe_name} - Advanced Terminal</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            background-color: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, sans-serif;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }}
        body.light-theme {{
            background-color: #ffffff;
            color: #131722;
        }}
        .chart-terminal-wrapper {{
            width: 100%;
            height: {height}px;
            display: flex;
            flex-direction: column;
            position: relative;
            background-color: #131722;
            border-radius: 8px;
            border: 1px solid #2A2E39;
            overflow: hidden;
            user-select: none;
        }}
        .chart-terminal-wrapper.light-theme {{
            background-color: #ffffff;
            border-color: #e0e3eb;
        }}
        .chart-terminal-wrapper.css-fullscreen {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 999999 !important;
            border-radius: 0 !important;
            border: none !important;
        }}
        /* Top Navigation Header Bar */
        .tv-top-bar {{
            height: {header_h}px;
            padding: 0 12px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e222d;
            border-bottom: 1px solid #2A2E39;
            flex-shrink: 0;
            z-index: 20;
        }}
        .light-theme .tv-top-bar {{
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }}
        .tv-top-left {{
            display: flex;
            align-items: center;
            gap: 12px;
            overflow: hidden;
        }}
        .sym-badge {{
            font-size: 14px;
            font-weight: 800;
            color: #38BDF8;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .light-theme .sym-badge {{
            color: #0284c7;
        }}
        .live-dot {{
            width: 7px;
            height: 7px;
            background: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10B981;
        }}
        .tf-pill {{
            background: #2A2E39;
            color: #CBD5E1;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .light-theme .tf-pill {{
            background: #e2e8f0;
            color: #334155;
        }}
        .ohlc-display {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            font-family: 'SF Mono', Consolas, Monaco, monospace;
        }}
        .ohlc-label {{ color: #787B86; }}
        .ohlc-val {{ color: #E2E8F0; font-weight: 600; }}
        .light-theme .ohlc-val {{ color: #0F172A; }}
        
        .tv-top-right {{
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }}
        .ctrl-btn {{
            background: #2A2E39;
            color: #CBD5E1;
            border: 1px solid #363C4E;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 11px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .ctrl-btn:hover {{
            background: #363C4E;
            color: #FFFFFF;
            border-color: #4A5268;
        }}
        .ctrl-btn.active {{
            background: #2962FF;
            color: #FFFFFF;
            border-color: #2962FF;
        }}
        .light-theme .ctrl-btn {{
            background: #ffffff;
            color: #334155;
            border: 1px solid #cbd5e1;
        }}
        .light-theme .ctrl-btn:hover {{
            background: #f1f5f9;
            color: #0f172a;
        }}
        .light-theme .ctrl-btn.active {{
            background: #2962FF;
            color: #ffffff;
            border-color: #2962FF;
        }}

        /* Body Area with Left Drawing Toolbar and Panes */
        .chart-body-container {{
            display: flex;
            flex-direction: row;
            width: 100%;
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        .tv-side-bar {{
            width: 40px;
            background: #1e222d;
            border-right: 1px solid #2A2E39;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 0;
            gap: 4px;
            flex-shrink: 0;
            z-index: 25;
        }}
        .light-theme .tv-side-bar {{
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}
        .side-tool-btn {{
            width: 30px;
            height: 30px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
            color: #94A3B8;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }}
        .side-tool-btn:hover {{
            background: #2A2E39;
            color: #FFFFFF;
        }}
        .light-theme .side-tool-btn:hover {{
            background: #e2e8f0;
            color: #0f172a;
        }}
        .side-tool-btn.active {{
            background: #2962FF !important;
            color: #FFFFFF !important;
            border-color: #2962FF !important;
        }}
        .side-sep {{
            width: 22px;
            height: 1px;
            background: #2A2E39;
            margin: 4px 0;
        }}
        .light-theme .side-sep {{
            background: #e2e8f0;
        }}
        .side-color-wrap {{
            position: relative;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            overflow: hidden;
            border: 2px solid #363C4E;
            cursor: pointer;
        }}
        .side-color-input {{
            position: absolute;
            top: -10px;
            left: -10px;
            width: 40px;
            height: 40px;
            cursor: pointer;
            opacity: 0;
        }}
        .side-color-preview {{
            width: 100%;
            height: 100%;
            background-color: #00E5FF;
        }}

        /* Center Panes Column */
        .panes-column {{
            display: flex;
            flex-direction: column;
            flex: 1;
            height: 100%;
            overflow: hidden;
            position: relative;
            cursor: crosshair !important;
        }}
        .pane-main, .pane-sub, .pane-main canvas, .pane-sub canvas, .draw-canvas {{
            cursor: crosshair !important;
        }}
        .adv-crosshair-v {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            border-left: 1px dashed rgba(255, 255, 255, 0.45);
            z-index: 35;
            pointer-events: none;
            display: none;
        }}
        .light-theme .adv-crosshair-v {{
            border-left: 1px dashed rgba(30, 41, 59, 0.45);
        }}
        .pane-main {{
            width: 100%;
            height: {main_h}px;
            position: relative;
            flex: 1;
        }}
        .pane-sub {{
            width: 100%;
            height: {sub_pane_h}px;
            position: relative;
            border-top: 1px solid #2A2E39;
            flex-shrink: 0;
        }}
        .light-theme .pane-sub {{
            border-top: 1px solid #e2e8f0;
        }}
        .pane-badge-title {{
            position: absolute;
            top: 6px;
            left: 10px;
            font-size: 10px;
            font-weight: 700;
            color: #94A3B8;
            background: rgba(19, 23, 34, 0.7);
            padding: 2px 6px;
            border-radius: 3px;
            z-index: 10;
            pointer-events: none;
        }}
        .light-theme .pane-badge-title {{
            background: rgba(255, 255, 255, 0.85);
            color: #475569;
        }}

        /* Bottom Timeframe Range Bar (TradingView style) */
        .tv-bottom-bar {{
            height: {range_bar_h}px;
            padding: 0 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e222d;
            border-top: 1px solid #2A2E39;
            flex-shrink: 0;
            font-size: 11px;
            z-index: 20;
        }}
        .light-theme .tv-bottom-bar {{
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
        }}
        .range-btn-group {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .range-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: #94A3B8;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .range-btn:hover {{
            background: #2A2E39;
            color: #FFFFFF;
        }}
        .range-btn.active {{
            background: #2962FF;
            color: #FFFFFF;
        }}
        .light-theme .range-btn:hover {{
            background: #e2e8f0;
            color: #0f172a;
        }}
        .light-theme .range-btn.active {{
            background: #2962FF;
            color: #ffffff;
        }}
        .tz-note {{
            color: #64748B;
            font-size: 11px;
        }}

        /* Floating Info Tooltip */
        .tv-tip {{
            display: none;
            position: absolute;
            width: 210px;
            padding: 8px 12px;
            background: rgba(19, 23, 34, 0.95);
            border: 1px solid #2962FF;
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7);
            pointer-events: none;
            z-index: 99999;
            font-size: 11px;
            color: #D1D4DC;
            backdrop-filter: blur(4px);
        }}
        .light-theme .tv-tip {{
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
            color: #131722;
        }}
        .tip-header {{
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 4px;
            padding-bottom: 2px;
            border-bottom: 1px solid #2A2E39;
        }}
        .tip-row {{
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
        }}
        .tip-key {{ color: #94A3B8; }}
        .tip-val {{ font-weight: 600; font-family: monospace; }}

        /* Drawing Canvas Overlay */
        .draw-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 15;
            pointer-events: none;
        }}
    </style>
</head>
<body class="{'light-theme' if is_light else ''}">
    <div id="term_wrapper_{safe_id}" class="chart-terminal-wrapper {'light-theme' if is_light else ''}">
        <!-- Top Navigation Header Bar -->
        <div class="tv-top-bar">
            <div class="tv-top-left">
                <div class="sym-badge">
                    <span class="live-dot"></span>
                    <span>NSE:{symbol.upper()}</span>
                </div>
                <div class="tf-pill">{timeframe_name}</div>
                <div id="legend_{safe_id}" class="ohlc-display">
                    <span class="ohlc-label">O: <span id="leg_o_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-label">H: <span id="leg_h_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-label">L: <span id="leg_l_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-label">C: <span id="leg_c_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-label">Chg: <span id="leg_chg_{safe_id}" class="ohlc-val">--</span></span>
                    <span class="ohlc-label">Vol: <span id="leg_v_{safe_id}" class="ohlc-val">--</span></span>
                </div>
            </div>
            <div class="tv-top-right">
                <!-- Zoom Controls -->
                <button id="btn_zoom_in_{safe_id}" class="ctrl-btn" type="button" title="Zoom In">🔍 +</button>
                <button id="btn_zoom_out_{safe_id}" class="ctrl-btn" type="button" title="Zoom Out">🔍 -</button>
                <button id="btn_reset_view_{safe_id}" class="ctrl-btn" type="button" title="Reset View / Fit Content">⟲ Reset</button>
                
                <!-- Chart Type Selector -->
                <button id="type_candle_{safe_id}" class="ctrl-btn active" type="button" title="Candlesticks">🕯️</button>
                <button id="type_line_{safe_id}" class="ctrl-btn" type="button" title="Line Chart">📈</button>
                <button id="type_area_{safe_id}" class="ctrl-btn" type="button" title="Area Chart">🏔️</button>
                
                <!-- Theme Toggle & Fullscreen -->
                <button id="theme_btn_{safe_id}" class="ctrl-btn" type="button" title="Toggle Theme">{'☀️ Light' if is_light else '🌙 Dark'}</button>
                <button id="fs_btn_{safe_id}" class="ctrl-btn" type="button" title="Fullscreen">⛶ Fullscreen</button>
            </div>
        </div>

        <!-- Body with Left Toolbar and Stacked Panes -->
        <div class="chart-body-container">
            <!-- Left Drawing Tools -->
            <div class="tv-side-bar">
                <button class="side-tool-btn active" data-tool="pointer" title="Crosshair Pointer">✛</button>
                <button class="side-tool-btn" data-tool="trendline" title="Trendline">╱</button>
                <button class="side-tool-btn" data-tool="horizontal" title="Horizontal Support/Resistance">―</button>
                <button class="side-tool-btn" data-tool="ray" title="Horizontal Ray">⟶</button>
                <button class="side-tool-btn" data-tool="rect" title="Rectangle / Zone">▭</button>
                <button class="side-tool-btn" data-tool="fib" title="Fibonacci Retracement">≡</button>
                <button class="side-tool-btn" data-tool="brush" title="Brush / Freehand">✎</button>
                <button class="side-tool-btn" data-tool="text" title="Text Note">T</button>
                <div class="side-sep"></div>
                <div class="side-color-wrap" title="Drawing Color">
                    <div id="color_dot_{safe_id}" class="side-color-preview"></div>
                    <input type="color" id="color_input_{safe_id}" value="#00E5FF" class="side-color-input" />
                </div>
                <div class="side-sep"></div>
                <button id="tool_undo_{safe_id}" class="side-tool-btn" title="Undo Last Drawing">↶</button>
                <button id="tool_clear_{safe_id}" class="side-tool-btn" title="Clear All Drawings">🗑</button>
            </div>

            <!-- Panes Column -->
            <div class="panes-column" id="panes_col_{safe_id}">
                <div class="adv-crosshair-v" id="ch_v_{safe_id}"></div>
                <!-- Main Price Pane -->
                <div class="pane-main" id="main_pane_{safe_id}">
                    <canvas id="draw_canvas_{safe_id}" class="draw-canvas"></canvas>
                    <div id="main_chart_{safe_id}" style="width:100%; height:100%;"></div>
                </div>

                <!-- Sub-Panels (Stacked) -->
                {'<div class="pane-sub" id="rsi_pane_' + safe_id + '"><span class="pane-badge-title">Hilega Milega(by NK sir) [anuragM]</span><div id="rsi_chart_' + safe_id + '" style="width:100%; height:100%;"></div></div>' if (show_rsi and rsi_pts) else ''}
                {'<div class="pane-sub" id="macd_pane_' + safe_id + '"><span class="pane-badge-title">MACD (12, 26, 9)</span><div id="macd_chart_' + safe_id + '" style="width:100%; height:100%;"></div></div>' if (show_macd and macd_pts) else ''}
                {'<div class="pane-sub" id="stoch_pane_' + safe_id + '"><span class="pane-badge-title">Stochastic (14, 3, 3)</span><div id="stoch_chart_' + safe_id + '" style="width:100%; height:100%;"></div></div>' if (show_stoch and stoch_k_pts) else ''}
            </div>
        </div>

        <!-- Bottom Timeframe Range Selector Bar -->
        <div class="tv-bottom-bar">
            <div class="range-btn-group">
                <button class="range-btn" data-range="1D" type="button">1D</button>
                <button class="range-btn" data-range="5D" type="button">5D</button>
                <button class="range-btn" data-range="1M" type="button">1M</button>
                <button class="range-btn" data-range="3M" type="button">3M</button>
                <button class="range-btn" data-range="6M" type="button">6M</button>
                <button class="range-btn" data-range="1Y" type="button">1Y</button>
                <button class="range-btn active" data-range="ALL" type="button">ALL</button>
            </div>
            <div class="tz-note">UTC+5:30 (IST) • Real-Time Upstox Feed</div>
        </div>

        <!-- Tooltip -->
        <div id="tooltip_{safe_id}" class="tv-tip"></div>
    </div>

    <script>
        (function() {{
            const wrapper = document.getElementById("term_wrapper_{safe_id}");
            const mainContainer = document.getElementById("main_chart_{safe_id}");
            const tooltip = document.getElementById("tooltip_{safe_id}");
            let isLight = { 'true' if is_light else 'false' };
            const isIntraday = { str(is_intraday).lower() };

            const candles = {candles_json};
            const volumes = {volumes_json};
            const volMas = {vol_ma_json};
            const emas = {emas_json};
            const smas = {smas_json};
            const bb = {bb_json};
            const supertrend = {supertrend_json};
            const vwap = {vwap_json};
            const cpr = {cpr_json};
            const rsi = {rsi_json};
            const rsiEma3 = {rsi_ema3_json};
            const rsiWma21 = {rsi_wma21_json};
            const macd = {macd_json};
            const stoch = {stoch_json};

            // Main Chart Setup
            const mainOptions = {{
                layout: {{
                    background: {{ color: isLight ? '#ffffff' : '#131722' }},
                    textColor: isLight ? '#131722' : '#94A3B8',
                    fontSize: 11
                }},
                grid: {{
                    vertLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }},
                    horzLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }}
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: isLight ? '#e0e3eb' : '#2A2E39',
                    autoScale: true,
                    scaleMargins: {{ top: 0.08, bottom: 0.22 }},
                    visible: true
                }},
                timeScale: {{
                    borderColor: isLight ? '#e0e3eb' : '#2A2E39',
                    timeVisible: isIntraday,
                    secondsVisible: false,
                    rightOffset: 10,
                    barSpacing: 8,
                    minBarSpacing: 0.5
                }},
                handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true, axisReset: true }}
            }};

            const mainChart = LightweightCharts.createChart(mainContainer, mainOptions);

            // Add Price Series (Default Candlestick with active price badge)
            let candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#F23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#F23645',
                lastValueVisible: true,
                priceLineVisible: true
            }});
            candleSeries.setData(candles);

            let lineSeries = null;
            let areaSeries = null;

            // Volume Series
            let volumeSeries = null;
            if (volumes && volumes.length > 0) {{
                volumeSeries = mainChart.addHistogramSeries({{
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: 'volume',
                    lastValueVisible: false,
                    priceLineVisible: false
                }});
                mainChart.priceScale('volume').applyOptions({{
                    scaleMargins: {{ top: 0.82, bottom: 0 }}
                }});
                volumeSeries.setData(volumes);
            }}

            // Volume MA
            if (volMas && volMas.length > 0) {{
                const volMaSeries = mainChart.addLineSeries({{
                    priceScaleId: 'volume',
                    color: isLight ? '#64748b' : '#94A3B8',
                    lineWidth: 1.5,
                    lastValueVisible: false,
                    priceLineVisible: false
                }});
                volMaSeries.setData(volMas);
            }}

            // EMAs Overlay
            const emaColors = {{ "EMA_5": "#4CAF50", "EMA_9": "#4CAF50", "EMA_13": "#38BDF8", "EMA_20": "#2962FF", "Daily_EMA_20": "#2962FF", "EMA_26": "#9C27B0", "EMA_50": "#FF5252", "EMA_200": isLight ? "#131722" : "#FFFFFF" }};
            for (const [col, pts] of Object.entries(emas)) {{
                if (pts && pts.length > 0) {{
                    const s = mainChart.addLineSeries({{
                        color: emaColors[col] || "#FFFFFF",
                        lineWidth: 1.8,
                        lastValueVisible: false,
                        priceLineVisible: false
                    }});
                    s.setData(pts);
                }}
            }}

            // SMAs Overlay
            const smaColors = {{ "SMA_20": "#FFD700", "SMA_50": "#F43F5E", "SMA_200": "#8B5CF6" }};
            for (const [col, pts] of Object.entries(smas)) {{
                if (pts && pts.length > 0) {{
                    const s = mainChart.addLineSeries({{
                        color: smaColors[col] || "#FFFFFF",
                        lineWidth: 1.8,
                        lineStyle: LightweightCharts.LineStyle.Dotted,
                        lastValueVisible: false,
                        priceLineVisible: false
                    }});
                    s.setData(pts);
                }}
            }}

            // Bollinger Bands Overlay
            if (bb && bb.upper && bb.upper.length > 0) {{
                const bbUp = mainChart.addLineSeries({{ color: '#3B82F6', lineWidth: 1.2, lastValueVisible: false, priceLineVisible: false }});
                const bbMid = mainChart.addLineSeries({{ color: '#F59E0B', lineWidth: 1.2, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false }});
                const bbLow = mainChart.addLineSeries({{ color: '#3B82F6', lineWidth: 1.2, lastValueVisible: false, priceLineVisible: false }});
                bbUp.setData(bb.upper);
                bbMid.setData(bb.middle);
                bbLow.setData(bb.lower);
            }}

            // Supertrend Overlay
            if (supertrend && supertrend.length > 0) {{
                const stSeries = mainChart.addLineSeries({{
                    lineWidth: 2.2,
                    lastValueVisible: true,
                    priceLineVisible: false
                }});
                stSeries.setData(supertrend);
            }}

            // VWAP Overlay
            if (vwap && vwap.length > 0) {{
                const vwapSeries = mainChart.addLineSeries({{
                    color: '#EC4899',
                    lineWidth: 2.0,
                    lastValueVisible: true,
                    priceLineVisible: false
                }});
                vwapSeries.setData(vwap);
            }}

            // CPR Overlay
            if (cpr && cpr.Pivot && cpr.Pivot.length > 0) {{
                const pSeries = mainChart.addLineSeries({{ color: '#38BDF8', lineWidth: 1.8, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false }});
                pSeries.setData(cpr.Pivot);
                if (cpr.TC && cpr.TC.length > 0) {{
                    const tcSeries = mainChart.addLineSeries({{ color: '#818CF8', lineWidth: 1.4, lineStyle: LightweightCharts.LineStyle.Dotted, lastValueVisible: false, priceLineVisible: false }});
                    tcSeries.setData(cpr.TC);
                }}
                if (cpr.BC && cpr.BC.length > 0) {{
                    const bcSeries = mainChart.addLineSeries({{ color: '#818CF8', lineWidth: 1.4, lineStyle: LightweightCharts.LineStyle.Dotted, lastValueVisible: false, priceLineVisible: false }});
                    bcSeries.setData(cpr.BC);
                }}
                if (cpr.R1 && cpr.R1.length > 0) {{
                    const r1Series = mainChart.addLineSeries({{ color: '#EF4444', lineWidth: 1.4, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false }});
                    r1Series.setData(cpr.R1);
                }}
                if (cpr.S1 && cpr.S1.length > 0) {{
                    const s1Series = mainChart.addLineSeries({{ color: '#10B981', lineWidth: 1.4, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false }});
                    s1Series.setData(cpr.S1);
                }}
            }}

            // Sub-Charts Array for Synchronized Scrolling & Zooming
            const allCharts = [mainChart];

            // Setup RSI Sub-Chart
            let rsiChart = null;
            let hmCloud = null;
            let rsiSeries = null;
            const rsiElem = document.getElementById("rsi_chart_{safe_id}");
            if (rsiElem && rsi && rsi.length > 0) {{
                rsiChart = LightweightCharts.createChart(rsiElem, {{
                    layout: {{ background: {{ color: isLight ? '#ffffff' : '#131722' }}, textColor: isLight ? '#131722' : '#94A3B8', fontSize: 10 }},
                    grid: {{ vertLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }}, horzLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }} }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: isLight ? '#e0e3eb' : '#2A2E39', autoScale: false, scaleMargins: {{ top: 0.05, bottom: 0.05 }} }},
                    timeScale: {{ visible: false, rightOffset: 10, barSpacing: 8 }},
                    handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                    handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }}
                }});
                // Baseline Cloud for Hilega Milega (Pink cloud above 50, Soft blue cloud below 50)
                try {{
                    hmCloud = rsiChart.addBaselineSeries({{
                        baseValue: {{ type: 'price', price: 50 }},
                        topFillColor1: isLight ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                        topFillColor2: isLight ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                        bottomFillColor1: isLight ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                        bottomFillColor2: isLight ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                        topLineColor: isLight ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                        bottomLineColor: isLight ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)',
                        lineWidth: 1,
                        lastValueVisible: false,
                        priceLineVisible: false
                    }});
                    if (rsi && rsi.length > 0) hmCloud.setData(rsi);
                }} catch(e) {{}}

                const rsiColor = isLight ? '#131722' : '#F8FAFC';
                rsiSeries = rsiChart.addLineSeries({{ color: rsiColor, lineWidth: 1.8, lastValueVisible: true, priceLineVisible: false, title: 'RSI(9)' }});
                rsiSeries.setData(rsi);
                if (rsiEma3 && rsiEma3.length > 0) {{
                    const e3 = rsiChart.addLineSeries({{ color: '#4CAF50', lineWidth: 1.8, lastValueVisible: true, priceLineVisible: false, title: 'EMA(3)' }});
                    e3.setData(rsiEma3);
                }}
                if (rsiWma21 && rsiWma21.length > 0) {{
                    const w21 = rsiChart.addLineSeries({{ color: '#FF5252', lineWidth: 1.8, lastValueVisible: true, priceLineVisible: false, title: 'WMA(21)' }});
                    w21.setData(rsiWma21);
                }}
                rsiSeries.createPriceLine({{ price: 50, color: '#7695F9', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: '50' }});
                rsiSeries.createPriceLine({{ price: 70, color: '#94A3B8', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false, title: '' }});
                rsiSeries.createPriceLine({{ price: 30, color: '#94A3B8', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false, title: '' }});
                allCharts.push(rsiChart);
            }}

            // Setup MACD Sub-Chart
            let macdChart = null;
            const macdElem = document.getElementById("macd_chart_{safe_id}");
            if (macdElem && macd && macd.macd && macd.macd.length > 0) {{
                macdChart = LightweightCharts.createChart(macdElem, {{
                    layout: {{ background: {{ color: isLight ? '#ffffff' : '#131722' }}, textColor: isLight ? '#131722' : '#94A3B8', fontSize: 10 }},
                    grid: {{ vertLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }}, horzLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }} }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: isLight ? '#e0e3eb' : '#2A2E39', autoScale: true }},
                    timeScale: {{ visible: false, rightOffset: 10, barSpacing: 8 }},
                    handleScroll: {{ mouseWheel: true, pressedMouseMove: true }},
                    handleScale: {{ axisPressedMouseMove: true, mouseWheel: true }}
                }});
                const histSeries = macdChart.addHistogramSeries({{ lastValueVisible: false, priceLineVisible: false }});
                histSeries.setData(macd.hist);
                const mLine = macdChart.addLineSeries({{ color: '#2962FF', lineWidth: 2, lastValueVisible: true, priceLineVisible: false }});
                mLine.setData(macd.macd);
                const sLine = macdChart.addLineSeries({{ color: '#FF6D00', lineWidth: 1.5, lastValueVisible: false, priceLineVisible: false }});
                sLine.setData(macd.signal);
                mLine.createPriceLine({{ price: 0, color: '#64748B', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false }});
                allCharts.push(macdChart);
            }}

            // Setup Stochastic Sub-Chart
            let stochChart = null;
            const stochElem = document.getElementById("stoch_chart_{safe_id}");
            if (stochElem && stoch && stoch.k && stoch.k.length > 0) {{
                stochChart = LightweightCharts.createChart(stochElem, {{
                    layout: {{ background: {{ color: isLight ? '#ffffff' : '#131722' }}, textColor: isLight ? '#131722' : '#94A3B8', fontSize: 10 }},
                    grid: {{ vertLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }}, horzLines: {{ color: isLight ? '#f0f3fa' : '#1f2430' }} }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: isLight ? '#e0e3eb' : '#2A2E39', autoScale: false, scaleMargins: {{ top: 0.05, bottom: 0.05 }} }},
                    timeScale: {{ visible: false, rightOffset: 10, barSpacing: 8 }},
                    handleScroll: {{ mouseWheel: true, pressedMouseMove: true }},
                    handleScale: {{ axisPressedMouseMove: true, mouseWheel: true }}
                }});
                const kSeries = stochChart.addLineSeries({{ color: '#38BDF8', lineWidth: 1.8, lastValueVisible: true, priceLineVisible: false }});
                kSeries.setData(stoch.k);
                const dSeries = stochChart.addLineSeries({{ color: '#F59E0B', lineWidth: 1.5, lastValueVisible: false, priceLineVisible: false }});
                dSeries.setData(stoch.d);
                kSeries.createPriceLine({{ price: 80, color: '#EF4444', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '80' }});
                kSeries.createPriceLine({{ price: 20, color: '#10B981', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '20' }});
                allCharts.push(stochChart);
            }}

            // Bi-Directional Synchronized Scrolling & Zooming
            let isSyncing = false;
            allCharts.forEach((c, idx) => {{
                c.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (isSyncing || !range) return;
                    isSyncing = true;
                    allCharts.forEach((other, otherIdx) => {{
                        if (idx !== otherIdx) {{
                            try {{ other.timeScale().setVisibleLogicalRange(range); }} catch(e) {{}}
                        }}
                    }});
                    isSyncing = false;
                }});
            }});

            // Initial view: show last 100 bars
            if (candles && candles.length > 0) {{
                const tot = candles.length;
                const initBars = Math.min(tot, isIntraday ? 90 : 120);
                mainChart.timeScale().setVisibleLogicalRange({{
                    from: Math.max(0, tot - initBars),
                    to: tot + 3
                }});
            }} else {{
                mainChart.timeScale().fitContent();
            }}

            // Zoom In (+) Button
            const btnZoomIn = document.getElementById("btn_zoom_in_{safe_id}");
            if (btnZoomIn) {{
                btnZoomIn.addEventListener("click", () => {{
                    const range = mainChart.timeScale().getVisibleLogicalRange();
                    if (!range) return;
                    const delta = (range.to - range.from) * 0.25;
                    mainChart.timeScale().setVisibleLogicalRange({{
                        from: range.from + delta,
                        to: range.to - delta
                    }});
                }});
            }}

            // Zoom Out (-) Button
            const btnZoomOut = document.getElementById("btn_zoom_out_{safe_id}");
            if (btnZoomOut) {{
                btnZoomOut.addEventListener("click", () => {{
                    const range = mainChart.timeScale().getVisibleLogicalRange();
                    if (!range) return;
                    const delta = (range.to - range.from) * 0.25;
                    mainChart.timeScale().setVisibleLogicalRange({{
                        from: range.from - delta,
                        to: range.to + delta
                    }});
                }});
            }}

            // Reset View (⟲) Button
            const btnReset = document.getElementById("btn_reset_view_{safe_id}");
            if (btnReset) {{
                btnReset.addEventListener("click", () => {{
                    mainChart.timeScale().fitContent();
                }});
            }}

            // Range Bar Buttons: [1D] [5D] [1M] [3M] [6M] [1Y] [ALL]
            const rangeBtns = wrapper.querySelectorAll(".range-btn");
            rangeBtns.forEach(btn => {{
                btn.addEventListener("click", () => {{
                    rangeBtns.forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    const rangeType = btn.getAttribute("data-range");
                    const tot = candles.length;
                    if (rangeType === "ALL" || !tot) {{
                        mainChart.timeScale().fitContent();
                        return;
                    }}
                    const mapBars = {{
                        "1D": isIntraday ? 75 : 1,
                        "5D": isIntraday ? 375 : 5,
                        "1M": isIntraday ? 1600 : 22,
                        "3M": isIntraday ? 4800 : 66,
                        "6M": isIntraday ? 9600 : 132,
                        "1Y": isIntraday ? 19000 : 250
                    }};
                    const bCount = mapBars[rangeType] || 100;
                    mainChart.timeScale().setVisibleLogicalRange({{
                        from: Math.max(0, tot - bCount),
                        to: tot + 3
                    }});
                }});
            }});

            // Chart Type Toggles: Candlestick, Line, Area
            const btnCandle = document.getElementById("type_candle_{safe_id}");
            const btnLine = document.getElementById("type_line_{safe_id}");
            const btnArea = document.getElementById("type_area_{safe_id}");

            function setChartType(type) {{
                [btnCandle, btnLine, btnArea].forEach(b => b && b.classList.remove("active"));
                if (type === "candle") {{
                    if (btnCandle) btnCandle.classList.add("active");
                    candleSeries.applyOptions({{ visible: true }});
                    if (lineSeries) lineSeries.applyOptions({{ visible: false }});
                    if (areaSeries) areaSeries.applyOptions({{ visible: false }});
                }} else if (type === "line") {{
                    if (btnLine) btnLine.classList.add("active");
                    candleSeries.applyOptions({{ visible: false }});
                    if (areaSeries) areaSeries.applyOptions({{ visible: false }});
                    if (!lineSeries) {{
                        lineSeries = mainChart.addLineSeries({{ color: '#38BDF8', lineWidth: 2, lastValueVisible: true }});
                        lineSeries.setData(candles.map(c => ({{ time: c.time, value: c.close }})));
                    }} else {{
                        lineSeries.applyOptions({{ visible: true }});
                    }}
                }} else if (type === "area") {{
                    if (btnArea) btnArea.classList.add("active");
                    candleSeries.applyOptions({{ visible: false }});
                    if (lineSeries) lineSeries.applyOptions({{ visible: false }});
                    if (!areaSeries) {{
                        areaSeries = mainChart.addAreaSeries({{
                            topColor: 'rgba(56, 189, 248, 0.4)',
                            bottomColor: 'rgba(56, 189, 248, 0.0)',
                            lineColor: '#38BDF8',
                            lineWidth: 2,
                            lastValueVisible: true
                        }});
                        areaSeries.setData(candles.map(c => ({{ time: c.time, value: c.close }})));
                    }} else {{
                        areaSeries.applyOptions({{ visible: true }});
                    }}
                }}
            }}

            if (btnCandle) btnCandle.addEventListener("click", () => setChartType("candle"));
            if (btnLine) btnLine.addEventListener("click", () => setChartType("line"));
            if (btnArea) btnArea.addEventListener("click", () => setChartType("area"));

            // Theme Toggle
            const themeBtn = document.getElementById("theme_btn_{safe_id}");
            if (themeBtn) {{
                themeBtn.addEventListener("click", () => {{
                    isLight = !isLight;
                    wrapper.classList.toggle("light-theme", isLight);
                    document.body.classList.toggle("light-theme", isLight);
                    themeBtn.textContent = isLight ? "☀️ Light" : "🌙 Dark";

                    const bg = isLight ? '#ffffff' : '#131722';
                    const txt = isLight ? '#131722' : '#94A3B8';
                    const gridColor = isLight ? '#f0f3fa' : '#1f2430';
                    const borderC = isLight ? '#e0e3eb' : '#2A2E39';

                    allCharts.forEach(c => {{
                        c.applyOptions({{
                            layout: {{ background: {{ color: bg }}, textColor: txt }},
                            grid: {{ vertLines: {{ color: gridColor }}, horzLines: {{ color: gridColor }} }},
                            rightPriceScale: {{ borderColor: borderC }},
                            timeScale: {{ borderColor: borderC }}
                        }});
                    }});
                    if (hmCloud) {{
                        hmCloud.applyOptions({{
                            topFillColor1: isLight ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                            topFillColor2: isLight ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                            bottomFillColor1: isLight ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                            bottomFillColor2: isLight ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                            topLineColor: isLight ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                            bottomLineColor: isLight ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)'
                        }});
                    }}
                    if (rsiSeries) {{
                        rsiSeries.applyOptions({{
                            color: isLight ? '#131722' : '#F8FAFC'
                        }});
                    }}
                }});
            }}

            // Fullscreen Toggle
            const fsBtn = document.getElementById("fs_btn_{safe_id}");
            if (fsBtn) {{
                fsBtn.addEventListener("click", () => {{
                    if (!document.fullscreenElement) {{
                        if (wrapper.requestFullscreen) wrapper.requestFullscreen().catch(fallbackFs);
                        else fallbackFs();
                    }} else {{
                        if (document.exitFullscreen) document.exitFullscreen();
                    }}
                }});
                function fallbackFs() {{
                    wrapper.classList.toggle("css-fullscreen");
                    handleResize();
                }}
                document.addEventListener("fullscreenchange", () => {{
                    const isFs = !!document.fullscreenElement;
                    fsBtn.classList.toggle("active", isFs);
                    handleResize();
                }});
            }}

            // Crosshair & Tooltip handler
            const legO = document.getElementById("leg_o_{safe_id}");
            const legH = document.getElementById("leg_h_{safe_id}");
            const legL = document.getElementById("leg_l_{safe_id}");
            const legC = document.getElementById("leg_c_{safe_id}");
            const legChg = document.getElementById("leg_chg_{safe_id}");
            const legV = document.getElementById("leg_v_{safe_id}");

            mainChart.subscribeCrosshairMove(param => {{
                if (!param || !param.time || !param.point) {{
                    tooltip.style.display = "none";
                    return;
                }}
                const c = param.seriesData.get(candleSeries) || (lineSeries && param.seriesData.get(lineSeries)) || (areaSeries && param.seriesData.get(areaSeries));
                if (!c) {{
                    tooltip.style.display = "none";
                    return;
                }}

                const o = Number(c.open !== undefined ? c.open : c.value);
                const h = Number(c.high !== undefined ? c.high : c.value);
                const l = Number(c.low !== undefined ? c.low : c.value);
                const closeVal = Number(c.close !== undefined ? c.close : c.value);
                const diff = closeVal - o;
                const pct = o > 0 ? (diff / o) * 100 : 0;
                const isUp = closeVal >= o;
                const cColor = isUp ? "#089981" : "#F23645";

                if (legO) legO.textContent = "₹" + o.toFixed(2);
                if (legH) legH.textContent = "₹" + h.toFixed(2);
                if (legL) legL.textContent = "₹" + l.toFixed(2);
                if (legC) {{
                    legC.textContent = "₹" + closeVal.toFixed(2);
                    legC.style.color = cColor;
                }}
                if (legChg) {{
                    legChg.textContent = `${{diff >= 0 ? '+' : ''}}${{pct.toFixed(2)}}%`;
                    legChg.style.color = cColor;
                }}

                if (volumeSeries) {{
                    const vData = param.seriesData.get(volumeSeries);
                    if (vData && legV) legV.textContent = Number(vData.value).toLocaleString('en-IN');
                }}

                // Update Floating Tooltip
                let tStr = "";
                if (typeof param.time === "number") {{
                    const d = new Date(param.time * 1000);
                    tStr = d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }}) + ' ' +
                           d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }});
                }} else {{
                    tStr = String(param.time);
                }}

                tooltip.innerHTML = `
                    <div class="tip-header">${{tStr}}</div>
                    <div class="tip-row"><span class="tip-key">Open</span><span class="tip-val">₹${{o.toFixed(2)}}</span></div>
                    <div class="tip-row"><span class="tip-key">High</span><span class="tip-val">₹${{h.toFixed(2)}}</span></div>
                    <div class="tip-row"><span class="tip-key">Low</span><span class="tip-val">₹${{l.toFixed(2)}}</span></div>
                    <div class="tip-row"><span class="tip-key">Close</span><span class="tip-val" style="color:${{cColor}}">₹${{closeVal.toFixed(2)}}</span></div>
                    <div class="tip-row"><span class="tip-key">Change</span><span class="tip-val" style="color:${{cColor}}">${{diff >= 0 ? '+' : ''}}${{pct.toFixed(2)}}%</span></div>
                `;
                tooltip.style.display = "block";
                const wRect = wrapper.getBoundingClientRect();
                const tipX = Math.min(param.point.x + 55, wRect.width - 230);
                const tipY = Math.min(param.point.y + 40, wRect.height - 150);
                tooltip.style.left = tipX + "px";
                tooltip.style.top = tipY + "px";
            }});

            const chVEl = document.getElementById("ch_v_{safe_id}");
            function updateAdvCrosshair(param) {{
                if (!param || !param.point) {{
                    if (chVEl) chVEl.style.display = "none";
                    return;
                }}
                if (chVEl) {{
                    chVEl.style.left = param.point.x + "px";
                    chVEl.style.display = "block";
                }}
            }}
            allCharts.forEach(c => {{
                try {{ c.subscribeCrosshairMove(updateAdvCrosshair); }} catch(e) {{}}
            }});
            const panesCol = document.getElementById("panes_col_{safe_id}");
            if (panesCol) {{
                panesCol.addEventListener("mouseleave", () => {{
                    if (chVEl) chVEl.style.display = "none";
                    if (tooltip) tooltip.style.display = "none";
                }});
            }}

            // Stop mousewheel from bubbling to parent iframe
            mainContainer.addEventListener("wheel", e => {{
                e.stopPropagation();
            }}, {{ passive: false }});

            function handleResize() {{
                const mainBox = mainContainer.getBoundingClientRect();
                mainChart.resize(mainBox.width, mainBox.height);
                if (rsiChart && rsiElem) {{
                    const rBox = rsiElem.getBoundingClientRect();
                    rsiChart.resize(rBox.width, rBox.height);
                }}
                if (macdChart && macdElem) {{
                    const mBox = macdElem.getBoundingClientRect();
                    macdChart.resize(mBox.width, mBox.height);
                }}
                if (stochChart && stochElem) {{
                    const sBox = stochElem.getBoundingClientRect();
                    stochChart.resize(sBox.width, sBox.height);
                }}
            }}

            window.addEventListener("resize", handleResize);
            setTimeout(handleResize, 50);
            setTimeout(handleResize, 300);
        }})();
    </script>
</body>
</html>"""
    return html_code


def generate_quad_chart_html(
    symbol: str,
    monthly_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    intra_df: pd.DataFrame,
    rsi_span: int = 9,
    show_rsi: bool = True,
    show_volume: bool = True,
    show_candles: bool = True,
    show_line: bool = True,
    pivot_dict_75: dict = None,
    height: int = 880,
    theme: str = "light",
    all_symbols: list = None,
    intra_tf_label: str = "75m"
) -> str:
    """
    Renders a unified 4-quadrant multi-timeframe dashboard powered by TradingView Lightweight Charts.
    Features:
    - 2x2 Grid displaying Monthly, Weekly, Daily, and 75-Minute charts simultaneously.
    - Master Fullscreen button that puts ALL 4 QUADRANTS into full screen simultaneously (50% x 50% each).
    - Quick Zoom buttons (1M, 1W, 1D, 75m) to focus on an individual quadrant and restore back to 2x2 grid.
    - RSI Panel with Solid 50 Reference Line (width 2), EMA 3, WMA 21, and light white bull zone above 50.
    - Clean right price scale with NO indicator labels.
    - Floating OHLC & Indicator tooltips on candle hover.
    - Interactive Theme Toggle (Dark / Light) with dynamic real-time updates.
    """
    is_light = (str(theme).lower() == "light")

    def _extract_payload(df, ema_dict, pivot_dict=None, is_intraday=False):
        if df is None or df.empty:
            return {"candles": [], "volumes": [], "emas": {}, "pivots": {}, "rsi": [], "rsi_ema3": [], "rsi_wma21": [], "hasRsi": False}

        df_clean = df.copy()
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        df_clean = df_clean.sort_index()
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

        def _get_t(dt):
            if is_intraday:
                if dt.tzinfo is None:
                    return int(dt.tz_localize("Asia/Kolkata").timestamp())
                return int(dt.tz_convert("Asia/Kolkata").timestamp())
            return dt.strftime("%Y-%m-%d")

        candles, volumes = [], []
        for dt, row in df_clean.iterrows():
            t_val = _get_t(dt)
            o = float(row.get("open", 0))
            h = float(row.get("high", 0))
            l = float(row.get("low", 0))
            c = float(row.get("close", 0))
            v = float(row.get("volume", 0)) if "volume" in row and not pd.isna(row["volume"]) else 0
            if o > 0 and h > 0 and l > 0 and c > 0:
                candles.append({"time": t_val, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2)})
                volumes.append({"time": t_val, "value": round(v, 2), "color": "rgba(8, 153, 129, 0.4)" if c >= o else "rgba(242, 54, 69, 0.4)"})

        emas_data = {}
        for col_name, color_or_cfg in ema_dict.items():
            if col_name in df_clean.columns:
                series_pts = []
                for dt, row in df_clean.iterrows():
                    val = row[col_name]
                    if not pd.isna(val) and val > 0:
                        t_val = _get_t(dt)
                        series_pts.append({"time": t_val, "value": round(float(val), 2)})
                if series_pts:
                    if isinstance(color_or_cfg, dict):
                        c_clr = color_or_cfg.get("color", "#38BDF8")
                        c_style = color_or_cfg.get("lineStyle", 0)
                        c_width = color_or_cfg.get("lineWidth", 1.8)
                        c_name = color_or_cfg.get("name", col_name)
                    else:
                        c_clr = str(color_or_cfg)
                        c_style = 1 if col_name == "AVWAP_JUN2022" else 0
                        c_width = 1.8 if "AVWAP" in col_name else (2.2 if col_name == "Daily_EMA_20" else 1.5)
                        c_name = "AVWAP Mar 2020" if col_name == "AVWAP_MAR2020" else ("AVWAP Jun 2022" if col_name == "AVWAP_JUN2022" else col_name)

                    final_color = c_clr
                    if is_light and (str(c_clr).upper() in ["#FFFFFF", "WHITE"]):
                        final_color = "#1E293B"
                    emas_data[col_name] = {
                        "color": final_color,
                        "lineStyle": c_style,
                        "lineWidth": c_width,
                        "name": c_name,
                        "data": series_pts
                    }

        pivots_data = {}
        if pivot_dict:
            for p_col, p_cfg in pivot_dict.items():
                if p_col in df_clean.columns:
                    series_pts = []
                    for dt, row in df_clean.iterrows():
                        val = row[p_col]
                        if not pd.isna(val) and val > 0:
                            t_val = _get_t(dt)
                            series_pts.append({"time": t_val, "value": round(float(val), 2)})
                    if series_pts:
                        p_color = p_cfg.get("color", "#FFFFFF")
                        if is_light and (str(p_color).upper() in ["#FFFFFF", "WHITE"]):
                            p_color = "#1E293B"
                        pivots_data[p_col] = {"color": p_color, "name": p_cfg.get("name", p_col), "data": series_pts}

        # RSI points & Hilega Milega indicator extraction
        has_rsi_panel = show_rsi and ("RSI" in df_clean.columns)
        rsi_pts, rsi_ema3_pts, rsi_wma21_pts = [], [], []
        if has_rsi_panel:
            for dt, row in df_clean.iterrows():
                t_val = _get_t(dt)
                r_val = row.get("RSI")
                if not pd.isna(r_val):
                    rsi_pts.append({"time": t_val, "value": round(float(r_val), 2)})
                if "RSI_EMA3" in df_clean.columns and not pd.isna(row.get("RSI_EMA3")):
                    rsi_ema3_pts.append({"time": t_val, "value": round(float(row["RSI_EMA3"]), 2)})
                if "RSI_WMA21" in df_clean.columns and not pd.isna(row.get("RSI_WMA21")):
                    rsi_wma21_pts.append({"time": t_val, "value": round(float(row["RSI_WMA21"]), 2)})

        return {
            "candles": candles,
            "volumes": volumes,
            "emas": emas_data,
            "pivots": pivots_data,
            "rsi": rsi_pts,
            "rsi_ema3": rsi_ema3_pts,
            "rsi_wma21": rsi_wma21_pts,
            "hasRsi": has_rsi_panel,
            "insufficientCandles": (len(df_clean) < 9 or len(rsi_pts) == 0),
            "candleCount": len(df_clean)
        }

    # Ensure EMA 20 and Daily EMA 20 are available on intra_df
    if intra_df is not None and not intra_df.empty:
        c_col = "Close" if "Close" in intra_df.columns else ("close" if "close" in intra_df.columns else None)
        if c_col and "EMA_20" not in intra_df.columns:
            intra_df["EMA_20"] = intra_df[c_col].ewm(span=20, adjust=False).mean()
        if daily_df is not None and not daily_df.empty and "Daily_EMA_20" not in intra_df.columns:
            try:
                dc_col = "Close" if "Close" in daily_df.columns else ("close" if "close" in daily_df.columns else None)
                d_ema20 = daily_df["EMA_20"] if "EMA_20" in daily_df.columns else (daily_df[dc_col].ewm(span=20, adjust=False).mean() if dc_col else None)
                if d_ema20 is not None:
                    d_dt_idx = pd.to_datetime(daily_df.index)
                    d_dates = d_dt_idx.tz_localize(None).date if hasattr(d_dt_idx, 'tz_localize') and d_dt_idx.tz is not None else d_dt_idx.date
                    d_map = pd.Series(d_ema20.values, index=d_dates)
                    d_map = d_map[~d_map.index.duplicated(keep="last")]

                    i_dt_idx = pd.to_datetime(intra_df.index)
                    i_dates = i_dt_idx.tz_localize(None).date if hasattr(i_dt_idx, 'tz_localize') and i_dt_idx.tz is not None else i_dt_idx.date
                    intra_df["Daily_EMA_20"] = pd.Series([d_map.get(d, np.nan) for d in i_dates], index=intra_df.index).ffill().bfill()
            except Exception:
                pass
                pass

    # Ensure AVWAP March 2020 and June 2022 are available on monthly_df
    if monthly_df is not None and not monthly_df.empty:
        if "AVWAP_MAR2020" not in monthly_df.columns:
            monthly_df["AVWAP_MAR2020"] = scanner.calculate_anchored_vwap(monthly_df, "2020-03-01")
        if "AVWAP_JUN2022" not in monthly_df.columns:
            monthly_df["AVWAP_JUN2022"] = scanner.calculate_anchored_vwap(monthly_df, "2022-06-01")

    # Extract payloads for all 4 quadrants
    m_payload = _extract_payload(
        monthly_df,
        {
            "EMA_5": "#4CAF50",
            "EMA_20": "#2962FF",
            "AVWAP_MAR2020": {"color": "#38BDF8", "lineStyle": 0, "lineWidth": 1.8, "name": "AVWAP Mar 2020"},
            "AVWAP_JUN2022": {"color": "#38BDF8", "lineStyle": 1, "lineWidth": 1.8, "name": "AVWAP Jun 2022"}
        },
        is_intraday=False
    )
    w_payload = _extract_payload(weekly_df, {"EMA_20": "#2962FF", "EMA_50": "#FF5252", "EMA_200": "#131722" if is_light else "#FFFFFF"}, is_intraday=False)
    d_payload = _extract_payload(daily_df, {"EMA_20": "#2962FF", "EMA_50": "#FF5252", "EMA_200": "#131722" if is_light else "#FFFFFF"}, is_intraday=False)
    cur_tf_raw = str(intra_tf_label).strip()
    cur_tf_low = cur_tf_raw.lower()
    is_q4_intraday = True
    if any(x in cur_tf_low for x in ["day", "week", "month", "quarter", "daily", "weekly", "monthly"]) or cur_tf_low.endswith(("d", "w", "mo")):
        is_q4_intraday = False

    p75 = pivot_dict_75 or {
        "Weekly_R1": {"color": "#EF4444", "name": "Weekly R1"},
        "Weekly_P":  {"color": "#38BDF8", "name": "Weekly Pivot"},
        "Weekly_S1": {"color": "#10B981", "name": "Weekly S1"}
    }
    intra_payload = _extract_payload(
        intra_df,
        {
            "EMA_9": "#4CAF50",
            "EMA_13": "#38BDF8",
            "EMA_20": "#2962FF",
            "Daily_EMA_20": "#2962FF",
            "EMA_26": "#9C27B0",
            "EMA_50": "#FF5252",
            "EMA_200": "#131722" if is_light else "#FFFFFF"
        },
        pivot_dict=p75 if is_q4_intraday else None,
        is_intraday=is_q4_intraday
    )

    quad_data_json = json.dumps({
        "symbol": symbol,
        "m": m_payload,
        "w": w_payload,
        "d": d_payload,
        "intra": intra_payload,
        "intra_is_intraday": is_q4_intraday,
        "showRsi": show_rsi,
        "showVolume": show_volume,
        "showCandles": show_candles,
        "showLine": show_line,
        "rsiSpan": rsi_span,
        "theme": "light" if is_light else "dark",
        "intraTfLabel": intra_tf_label
    })

    clean_sym = symbol.upper().replace("-EQ", "").replace(".NS", "")

    # 1. Prepare Stock dropdown options
    all_syms = list(all_symbols) if all_symbols else [clean_sym]
    if clean_sym not in all_syms:
        all_syms = [clean_sym] + all_syms
    stock_opt_list = []
    for s in all_syms:
        s_clean = str(s).strip().upper().replace("-EQ", "").replace(".NS", "")
        sel_attr = " selected" if s_clean == clean_sym else ""
        stock_opt_list.append(f'<option value="{s_clean}"{sel_attr}>{s_clean}</option>')
    stock_options_html = "".join(stock_opt_list)

    # 2. Prepare Timeframe dropdown options for Quadrant 4 (Minutes, Days, Weeks, Months)
    intraday_tfs = [
        ("1m", "1m (1-Min)"),
        ("3m", "3m (3-Min)"),
        ("5m", "5m (5-Min)"),
        ("10m", "10m (10-Min)"),
        ("15m", "15m (15-Min)"),
        ("30m", "30m (30-Min)"),
        ("60m", "60m (1-Hour)"),
        ("75m", "75m (75-Min)"),
        ("125m", "125m (125-Min)"),
    ]
    daily_tfs = [
        ("1D", "1D (Daily)"),
        ("2D", "2D (2-Day)"),
        ("3D", "3D (3-Day)"),
        ("5D", "5D (5-Day)"),
    ]
    weekly_tfs = [
        ("1W", "1W (Weekly)"),
        ("2W", "2W (2-Week)"),
    ]
    monthly_tfs = [
        ("1MO", "1M (Monthly)"),
        ("3MO", "3M (Quarterly)"),
    ]

    cur_tf = str(intra_tf_label).strip()
    cur_tf_lower = cur_tf.lower()

    tf_opt_list = []
    matched = False

    # Group 1: Minutes
    tf_opt_list.append('<optgroup label="⏱️ Intraday Minutes">')
    for tf_val, tf_title in intraday_tfs:
        sel_attr = ""
        if tf_val.lower() == cur_tf_lower or (cur_tf_lower == tf_val.lower().replace("m", "")):
            sel_attr = " selected"
            matched = True
        tf_opt_list.append(f'<option value="{tf_val}"{sel_attr}>{tf_title}</option>')
    tf_opt_list.append('<option value="custom_m">⚙️ Custom Minutes...</option>')
    tf_opt_list.append('</optgroup>')

    # Group 2: Days
    tf_opt_list.append('<optgroup label="📅 Daily & Multi-Day">')
    for tf_val, tf_title in daily_tfs:
        sel_attr = ""
        if tf_val.lower() == cur_tf_lower or (cur_tf_lower in ("1d", "daily", "d") and tf_val == "1D"):
            sel_attr = " selected"
            matched = True
        tf_opt_list.append(f'<option value="{tf_val}"{sel_attr}>{tf_title}</option>')
    tf_opt_list.append('<option value="custom_d">⚙️ Custom Days...</option>')
    tf_opt_list.append('</optgroup>')

    # Group 3: Weeks
    tf_opt_list.append('<optgroup label="📆 Weekly & Multi-Week">')
    for tf_val, tf_title in weekly_tfs:
        sel_attr = ""
        if tf_val.lower() == cur_tf_lower or (cur_tf_lower in ("1w", "weekly", "w") and tf_val == "1W"):
            sel_attr = " selected"
            matched = True
        tf_opt_list.append(f'<option value="{tf_val}"{sel_attr}>{tf_title}</option>')
    tf_opt_list.append('<option value="custom_w">⚙️ Custom Weeks...</option>')
    tf_opt_list.append('</optgroup>')

    # Group 4: Months
    tf_opt_list.append('<optgroup label="🌙 Monthly & Multi-Month">')
    for tf_val, tf_title in monthly_tfs:
        sel_attr = ""
        if tf_val.lower() == cur_tf_lower or (cur_tf_lower in ("1m", "1mo", "monthly", "mo") and tf_val == "1MO"):
            sel_attr = " selected"
            matched = True
        tf_opt_list.append(f'<option value="{tf_val}"{sel_attr}>{tf_title}</option>')
    tf_opt_list.append('<option value="custom_mo">⚙️ Custom Months...</option>')
    tf_opt_list.append('</optgroup>')

    if not matched and cur_tf:
        tf_opt_list.append(f'<option value="{cur_tf}" selected>Custom ({cur_tf})</option>')
    tf_options_html = "".join(tf_opt_list)

    q4_sub_label = "Trigger: 9>13>20>26 EMA | Daily 20 EMA | Pivots" if is_q4_intraday else f"Analysis ({intra_tf_label}): 20>50>200 EMA"

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{clean_sym} - 4-Quadrant Multi-Timeframe TradingView</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            background-color: #0b0e14;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }}
        body.light-theme {{
            background-color: #f8fafc !important;
            color: #131722 !important;
        }}
        .quad-root {{
            width: 100%;
            height: {height}px;
            display: flex;
            flex-direction: column;
            background-color: #0b0e14;
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid #1f2937;
        }}
        .quad-root:fullscreen,
        .quad-root:-webkit-full-screen,
        .quad-root.css-fullscreen {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 9999999 !important;
            border-radius: 0 !important;
            border: none !important;
            background-color: #0b0e14 !important;
        }}
        .quad-master-header {{
            height: 40px;
            background: #131722;
            border-bottom: 1px solid #2A2E39;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            user-select: none;
            flex-shrink: 0;
            z-index: 10;
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: none;
        }}
        .quad-master-header::-webkit-scrollbar {{
            display: none;
        }}
        .qm-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
        }}
        .qm-brand {{
            background: #2962FF;
            color: #ffffff;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }}
        .qm-stock {{
            font-size: 14px;
            font-weight: 800;
            color: #00E5FF;
            letter-spacing: 0.5px;
        }}
        .qm-badge {{
            font-size: 11px;
            color: #94A3B8;
            font-weight: 600;
        }}
        .qm-select {{
            background: #1e222d;
            color: #00E5FF;
            border: 1px solid #2A2E39;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            outline: none;
            transition: all 0.2s;
            max-width: 150px;
            font-family: inherit;
        }}
        .qm-select:hover, .qm-select:focus {{
            border-color: #2962FF;
            background: #242938;
            box-shadow: 0 0 8px rgba(41, 98, 255, 0.4);
        }}
        .qm-select option {{
            background: #131722;
            color: #ffffff;
            font-weight: 600;
            padding: 3px;
        }}
        .qm-select-tf {{
            color: #76FF03;
            border-color: rgba(118, 255, 3, 0.35);
            background: rgba(118, 255, 3, 0.08);
            max-width: 115px;
        }}
        .qm-select-tf:hover, .qm-select-tf:focus {{
            border-color: #76FF03;
            box-shadow: 0 0 8px rgba(118, 255, 3, 0.4);
        }}
        .light-theme .qm-select {{
            background: #f1f5f9;
            color: #0052FF;
            border-color: #cbd5e1;
        }}
        .light-theme .qm-select option {{
            background: #ffffff;
            color: #0f172a;
        }}
        .light-theme .qm-select-tf {{
            color: #16a34a;
            border-color: #86efac;
            background: #f0fdf4;
        }}
        .qm-btn-fs-prominent {{
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #ffffff !important;
            border: 1px solid #3B82F6;
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            letter-spacing: 0.5px;
            box-shadow: 0 0 10px rgba(37, 99, 235, 0.45);
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .qm-btn-fs-prominent:hover {{
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
            box-shadow: 0 0 16px rgba(37, 99, 235, 0.75);
            transform: scale(1.03);
        }}
        .qm-btn-fs-prominent.is-fs {{
            background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
            border-color: #EF4444;
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.6);
        }}
        .qm-right {{
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }}
        .qm-btn-group {{
            display: flex;
            background: #1e222d;
            border: 1px solid #2A2E39;
            border-radius: 4px;
            padding: 2px;
            gap: 2px;
        }}
        .qm-btn {{
            background: transparent;
            color: #94A3B8;
            border: none;
            border-radius: 3px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .qm-btn:hover {{
            background: #2A2E39;
            color: #ffffff;
        }}
        .qm-btn.active {{
            background: #2962FF;
            color: #ffffff;
        }}
        .qm-btn-fs {{
            background: #2A2E39;
            color: #ffffff;
            border: 1px solid #363C4E;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .qm-btn-fs:hover {{
            background: #363C4E;
            border-color: #00E5FF;
            color: #00E5FF;
        }}
        .qm-btn-fs.is-fs {{
            background: #B91C1C;
            border-color: #EF4444;
            color: #ffffff;
        }}
        .qm-btn-vol {{
            background: #2A2E39;
            color: #D1D4DC;
            border: 1px solid #363C4E;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s;
        }}
        .qm-btn-vol:hover {{
            background: #363C4E;
            color: #ffffff;
        }}
        .qm-btn-vol.active {{
            background: #1e3a8a;
            border-color: #3b82f6;
            color: #93c5fd;
        }}
        .quad-content-row {{
            display: flex;
            flex-direction: row;
            flex: 1;
            width: 100%;
            min-height: 0;
            overflow: hidden;
            position: relative;
        }}
        .tv-draw-bar {{
            width: 40px;
            background: #181c27;
            border-right: 1px solid #2A2E39;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 6px 0;
            gap: 2px;
            flex-shrink: 0;
            user-select: none;
            z-index: 25;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        .tv-draw-bar::-webkit-scrollbar {{
            width: 3px;
        }}
        .tv-draw-bar::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
        }}
        .light-theme .tv-draw-bar {{
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}
        .light-theme .tv-draw-bar::-webkit-scrollbar-thumb {{
            background: rgba(0, 0, 0, 0.15);
        }}
        .tv-tool-btn {{
            width: 28px;
            height: 28px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
            color: #94A3B8;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            padding: 0;
            flex-shrink: 0;
        }}
        .tv-tool-btn:hover {{
            background: #2A2E39;
            color: #FFFFFF;
        }}
        .light-theme .tv-tool-btn {{
            color: #64748b;
        }}
        .light-theme .tv-tool-btn:hover {{
            background: #e2e8f0;
            color: #0f172a;
        }}
        .tv-tool-btn.active {{
            background: #2962FF !important;
            color: #FFFFFF !important;
            border-color: #2962FF !important;
            box-shadow: 0 0 8px rgba(41, 98, 255, 0.4);
        }}
        .tv-tool-btn.magnet-active {{
            background: #00E5FF !important;
            color: #0F172A !important;
            border-color: #00E5FF !important;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.5);
        }}
        .tv-tool-sep {{
            width: 20px;
            height: 1px;
            background: #2A2E39;
            margin: 2px 0;
            flex-shrink: 0;
        }}
        .light-theme .tv-tool-sep {{
            background: #e2e8f0;
        }}
        .tv-color-wrapper {{
            position: relative;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid #363C4E;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            margin: 2px 0;
            flex-shrink: 0;
        }}
        .tv-color-picker {{
            position: absolute;
            top: -10px;
            left: -10px;
            width: 44px;
            height: 44px;
            border: none;
            cursor: pointer;
            opacity: 0;
        }}
        .tv-color-dot {{
            width: 100%;
            height: 100%;
            background-color: #00E5FF;
            border-radius: 50%;
        }}
        .tv-width-select {{
            width: 32px;
            height: 20px;
            font-size: 10px;
            font-weight: 700;
            background: #1e222d;
            color: #94A3B8;
            border: 1px solid #2A2E39;
            border-radius: 4px;
            padding: 0 1px;
            text-align: center;
            cursor: pointer;
            outline: none;
            margin: 2px 0;
            flex-shrink: 0;
        }}
        .light-theme .tv-width-select {{
            background: #ffffff;
            color: #1e293b;
            border-color: #cbd5e1;
        }}
        .qc-main-wrap {{
            flex: 68;
            position: relative;
            width: 100%;
            min-height: 0;
        }}
        .qc-draw-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 15;
            pointer-events: none;
        }}
        .hist-badge {{
            position: absolute;
            bottom: 25px;
            left: 10px;
            background: rgba(15, 23, 42, 0.9);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 10px;
            font-weight: 600;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 35;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}
        .light-theme .hist-badge {{
            background: rgba(255, 255, 255, 0.95);
            color: #0284c7;
            border: 1px solid #bae6fd;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        .quad-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 6px;
            flex: 1;
            min-height: 0;
            padding: 6px;
            box-sizing: border-box;
            background: #080a0f;
        }}
        /* Focused layout overrides */
        .quad-grid.focus-m {{
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
        }}
        .quad-grid.focus-m > .quad-card:not(#card_m) {{ display: none !important; }}

        .quad-grid.focus-w {{
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
        }}
        .quad-grid.focus-w > .quad-card:not(#card_w) {{ display: none !important; }}

        .quad-grid.focus-d {{
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
        }}
        .quad-grid.focus-d > .quad-card:not(#card_d) {{ display: none !important; }}

        .quad-grid.focus-75 {{
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
        }}
        .quad-grid.focus-75 > .quad-card:not(#card_75) {{ display: none !important; }}

        .quad-card {{
            background: #131722;
            border: 1px solid #2A2E39;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
            position: relative;
        }}
        .qc-header {{
            height: 30px;
            background: #181c27;
            border-bottom: 1px solid #2A2E39;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 8px;
            font-size: 11px;
            user-select: none;
            flex-shrink: 0;
            z-index: 5;
        }}
        .qc-header-left {{
            display: flex;
            align-items: center;
            gap: 8px;
            overflow: hidden;
            white-space: nowrap;
        }}
        .qc-tf {{
            font-weight: 700;
            font-size: 11px;
            padding: 1px 5px;
            border-radius: 3px;
        }}
        .qc-tf-m {{ background: rgba(255, 215, 0, 0.15); color: #FFD700; }}
        .qc-tf-w {{ background: rgba(0, 229, 255, 0.15); color: #00E5FF; }}
        .qc-tf-d {{ background: rgba(255, 145, 0, 0.15); color: #FF9100; }}
        .qc-tf-75 {{ background: rgba(118, 255, 3, 0.15); color: #76FF03; }}

        .qc-sub {{
            color: #94A3B8;
            font-size: 10px;
            font-family: monospace;
        }}
        .qc-legend {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
            font-family: 'SF Mono', Consolas, monospace;
            color: #787B86;
        }}
        .qc-legend b {{
            color: #D1D4DC;
            font-weight: 600;
        }}
        .qc-header-right {{
            display: flex;
            align-items: center;
            gap: 4px;
            flex-shrink: 0;
        }}
        .qc-btn-zoom {{
            background: #2A2E39;
            color: #D1D4DC;
            border: 1px solid #363C4E;
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 11px;
            cursor: pointer;
            line-height: 1;
            transition: all 0.15s;
        }}
        .qc-btn-zoom:hover {{
            background: #363C4E;
            color: #00E5FF;
        }}
        .qc-btn-zoom.active {{
            background: #2962FF;
            color: #ffffff;
            border-color: #2962FF;
        }}
        .qc-body,
        .qc-main-wrap,
        .qc-main-canvas,
        .qc-main-canvas canvas,
        .qc-rsi-canvas,
        .qc-rsi-canvas canvas,
        .qc-draw-canvas {{
            cursor: crosshair !important;
        }}
        .qc-body {{
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            min-height: 0;
            overflow: hidden;
            cursor: crosshair !important;
        }}
        .qc-crosshair-v {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            border-left: 1px dashed rgba(255, 255, 255, 0.40);
            pointer-events: none;
            z-index: 35;
            display: none;
        }}
        .light-theme .qc-crosshair-v,
        body.light-theme .qc-crosshair-v {{
            border-left: 1px dashed rgba(0, 0, 0, 0.40);
        }}
        .qc-main-canvas {{
            flex: 68;
            position: relative;
            width: 100%;
            min-height: 0;
        }}
        .qc-rsi-canvas {{
            flex: 32;
            position: relative;
            width: 100%;
            min-height: 0;
            border-top: 1px solid #2A2E39;
        }}
        .light-theme .qc-rsi-canvas {{
            border-top: 1px solid #e0e3eb;
        }}
        .qc-rsi-canvas .hm-header {{
            position: absolute;
            top: 4px;
            left: 8px;
            z-index: 10;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 10px;
            display: flex;
            align-items: center;
        }}
        .hm-vals {{
            font-size: 10px;
            font-family: 'SF Mono', Consolas, Monaco, monospace;
            color: #94A3B8;
            margin-left: 8px;
            font-weight: 600;
        }}
        .light-theme .hm-vals {{
            color: #475569;
        }}
        .qc-rsi-canvas .hm-title {{
            color: #94A3B8;
            font-weight: 700;
        }}
        .light-theme .qc-rsi-canvas .hm-title {{
            color: #1E293B;
        }}
        .rsi-bull-tint {{
            display: none !important;
        }}
        /* Tooltip */
        .tv-floating-tooltip {{
            display: none;
            position: absolute;
            width: 190px;
            padding: 7px 10px;
            background: rgba(19, 23, 34, 0.96);
            border: 1px solid #2962FF;
            border-radius: 6px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.8);
            pointer-events: none;
            z-index: 99999 !important;
            font-size: 10px;
            color: #D1D4DC;
            backdrop-filter: blur(5px);
            line-height: 1.4;
        }}
        .tt-date {{
            font-weight: 700;
            color: #00E5FF;
            margin-bottom: 4px;
            padding-bottom: 2px;
            border-bottom: 1px solid #2A2E39;
        }}
        .tt-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 1px 0;
        }}
        .tt-lbl {{ color: #94A3B8; }}
        .tt-val {{ font-weight: 600; color: #FFFFFF; }}
        .tt-up {{ color: #089981 !important; }}
        .tt-dn {{ color: #F23645 !important; }}

        /* Light Theme Overrides */
        .quad-root.light-theme {{
            background-color: #f8f9fa !important;
            border-color: #e0e3eb !important;
            color: #131722 !important;
        }}
        .quad-root.light-theme:fullscreen,
        .quad-root.light-theme:-webkit-full-screen,
        .quad-root.light-theme.css-fullscreen {{
            background-color: #f8f9fa !important;
        }}
        .light-theme .quad-master-header {{
            background: #ffffff !important;
            border-bottom: 1px solid #e0e3eb !important;
        }}
        .light-theme .qm-brand {{
            background: #2563EB !important;
            color: #ffffff !important;
        }}
        .light-theme .qm-stock {{
            color: #0052FF !important;
        }}
        .light-theme .qm-badge {{
            color: #64748b !important;
        }}
        .light-theme #sync_status_badge {{
            color: #0284c7 !important;
        }}
        .light-theme .qm-select {{
            background: #ffffff !important;
            color: #0052FF !important;
            border-color: #cbd5e1 !important;
        }}
        .light-theme .qm-select option {{
            background: #ffffff !important;
            color: #0f172a !important;
        }}
        .light-theme .qm-select-tf {{
            color: #16a34a !important;
            border-color: #86efac !important;
            background: #f0fdf4 !important;
        }}
        .light-theme .qm-btn-group {{
            background: #f0f3fa !important;
            border-color: #e0e3eb !important;
        }}
        .light-theme .qm-btn {{
            color: #475569 !important;
        }}
        .light-theme .qm-btn:hover {{
            background: #e2e8f0 !important;
            color: #1e293b !important;
        }}
        .light-theme .qm-btn.active {{
            background: #2962FF !important;
            color: #ffffff !important;
        }}
        .light-theme .qm-btn-fs,
        .light-theme .qm-btn-vol {{
            background: #ffffff !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
        }}
        .light-theme .qm-btn-fs:hover,
        .light-theme .qm-btn-vol:hover {{
            background: #f1f5f9 !important;
            color: #0052FF !important;
            border-color: #2962FF !important;
        }}
        .light-theme .qm-btn-vol.active {{
            background: #2962FF !important;
            border-color: #2962FF !important;
            color: #ffffff !important;
        }}
        .light-theme .tv-draw-bar {{
            background: #ffffff !important;
            border-right: 1px solid #e0e3eb !important;
        }}
        .light-theme .tv-tool-btn {{
            color: #64748b !important;
        }}
        .light-theme .tv-tool-btn:hover {{
            background: #f1f5f9 !important;
            color: #0f172a !important;
        }}
        .light-theme .tv-tool-btn.active {{
            background: #2962FF !important;
            color: #ffffff !important;
            border-color: #2962FF !important;
        }}
        .light-theme .tv-tool-sep {{
            background: #e0e3eb !important;
        }}
        .light-theme .tv-color-wrapper {{
            border-color: #cbd5e1 !important;
        }}
        .light-theme .quad-grid {{
            background: #f0f3fa !important;
        }}
        .light-theme .quad-card {{
            background: #ffffff !important;
            border-color: #e0e3eb !important;
        }}
        .light-theme .qc-header {{
            background: #ffffff !important;
            border-bottom: 1px solid #e0e3eb !important;
        }}
        .light-theme .qc-tf-m {{ background: #fef9c3 !important; color: #854d0e !important; }}
        .light-theme .qc-tf-w {{ background: #e0f2fe !important; color: #0369a1 !important; }}
        .light-theme .qc-tf-d {{ background: #ffedd5 !important; color: #c2410c !important; }}
        .light-theme .qc-tf-75 {{ background: #dcfce7 !important; color: #15803d !important; }}
        .light-theme .qc-sub {{
            color: #64748b !important;
        }}
        .light-theme .qc-legend {{
            color: #64748b !important;
        }}
        .light-theme .qc-legend b {{
            color: #1e293b !important;
        }}
        .light-theme .qc-btn-zoom {{
            background: #ffffff !important;
            color: #475569 !important;
            border-color: #cbd5e1 !important;
        }}
        .light-theme .qc-btn-zoom:hover {{
            background: #f1f5f9 !important;
            color: #0052FF !important;
        }}
        .light-theme .qc-btn-zoom.active {{
            background: #2962FF !important;
            color: #ffffff !important;
            border-color: #2962FF !important;
        }}
        .light-theme .qc-rsi-canvas {{
            border-top: 1px solid #e0e3eb !important;
        }}
        .light-theme .rsi-bull-tint {{
            background: rgba(41, 98, 255, 0.06) !important;
            border-bottom: 1px dashed rgba(41, 98, 255, 0.25) !important;
        }}
        .light-theme .tv-floating-tooltip {{
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid #2962FF !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
            color: #1e293b !important;
        }}
        .light-theme .tt-date {{
            color: #0052FF !important;
            border-bottom: 1px solid #e0e3eb !important;
        }}
        .light-theme .tt-lbl {{
            color: #64748b !important;
        }}
        .light-theme .tt-val {{
            color: #0F172A !important;
        }}
    </style>
</head>
<body class="{'light-theme' if is_light else ''}">
    <div id="quad_root" class="quad-root {'light-theme' if is_light else ''}">
        <!-- Master Navigation Header -->
        <div class="quad-master-header">
            <div class="qm-left">
                <span class="qm-brand">TRADINGVIEW</span>
                <div style="display:inline-flex; align-items:center; gap:4px;">
                    <span style="font-size:10px; color:#64748B; font-weight:700;">STOCK:</span>
                    <select id="qm_stock_select" class="qm-select" onchange="onQuadStockChange(this.value)" title="Select Stock">
                        {stock_options_html}
                    </select>
                </div>
                <div style="display:inline-flex; align-items:center; gap:4px;">
                    <span style="font-size:10px; color:#64748B; font-weight:700;">TF:</span>
                    <select id="qm_tf_select" class="qm-select qm-select-tf" onchange="onQuadTfChange(this.value)" title="Select Intraday Timeframe for Quadrant 4">
                        {tf_options_html}
                    </select>
                </div>
                <button id="btn_quad_fs_main" class="qm-btn-fs-prominent" onclick="toggleQuadFullscreen()" title="Toggle Fullscreen Mode (Press F or Esc)">⛶ FULLSCREEN</button>
                <span class="qm-badge" id="sync_status_badge" style="color: #00E5FF; font-weight: 700; margin-left: 4px;">🔄 Sync: 4 Active</span>
            </div>
            <div class="qm-right">
                <!-- Master Zoom Controls -->
                <div class="qm-btn-group" title="Zoom & Reset View">
                    <button id="btn_q_zoom_in" class="qm-btn" onclick="masterZoom(-0.25)" title="Zoom In (+)">🔍+</button>
                    <button id="btn_q_zoom_out" class="qm-btn" onclick="masterZoom(0.25)" title="Zoom Out (-)">🔍-</button>
                    <button id="btn_q_reset" class="qm-btn" onclick="masterReset()" title="Reset View (⟲)">⟲ Reset</button>
                </div>
                <!-- Time-Range Jump Buttons -->
                <div class="qm-btn-group" title="Quick Time Range">
                    <span style="font-size:10px; color:#64748B; font-weight:700; align-self:center; padding:0 3px;">RANGE:</span>
                    <button class="qm-btn qm-range-btn" data-range="1D" onclick="masterTimeRange('1D')" title="Jump to 1 Day range">1D</button>
                    <button class="qm-btn qm-range-btn" data-range="5D" onclick="masterTimeRange('5D')" title="Jump to 5 Days range">5D</button>
                    <button class="qm-btn qm-range-btn" data-range="1M" onclick="masterTimeRange('1M')" title="Jump to 1 Month range">1M</button>
                    <button class="qm-btn qm-range-btn" data-range="1Y" onclick="masterTimeRange('1Y')" title="Jump to 1 Year range">1Y</button>
                    <button class="qm-btn qm-range-btn active" data-range="ALL" onclick="masterTimeRange('ALL')" title="Fit All Data">ALL</button>
                </div>
                <div class="qm-btn-group">
                    <button id="btn_q_all" class="qm-btn active" onclick="setQuadMode('all')" title="Display all 4 quadrants in 2x2 grid">⊞ 4 Quadrants</button>
                    <button id="btn_q_m" class="qm-btn" onclick="setQuadMode('m')" title="Focus Monthly chart">1M</button>
                    <button id="btn_q_w" class="qm-btn" onclick="setQuadMode('w')" title="Focus Weekly chart">1W</button>
                    <button id="btn_q_d" class="qm-btn" onclick="setQuadMode('d')" title="Focus Daily chart">1D</button>
                    <button id="btn_q_75" class="qm-btn" onclick="setQuadMode('75')" title="Focus {intra_tf_label} chart">{intra_tf_label}</button>
                </div>
                <div class="qm-btn-group" title="Chart Display Style (All 4 Quadrants)">
                    <button id="btn_q_candles" class="qm-btn {'active' if show_candles else ''}" onclick="toggleMasterCandles()" title="Toggle Candlesticks on all 4 quadrants">🕯️</button>
                    <button id="btn_q_line" class="qm-btn {'active' if show_line else ''}" onclick="toggleMasterLine()" title="Toggle Line Chart on all 4 quadrants">📈</button>
                </div>
                <button id="btn_q_tooltip" class="qm-btn-vol active" onclick="toggleQuadTooltip()" title="Toggle Floating Tooltip (ON/OFF)">💬 Tooltip</button>
                <button id="btn_q_theme" class="qm-btn-vol" onclick="toggleQuadTheme()" title="Toggle Dark / Light Theme">{'☀️ Light' if is_light else '🌙 Dark'}</button>
                <button id="btn_q_vol" class="qm-btn-vol {'active' if show_volume else ''}" onclick="toggleQuadVolume()" title="Hide / Show Volume on all 4 quadrants">📊 Vol</button>
            </div>
        </div>

        <!-- Body row with Left Drawing Toolbar + 2x2 Quadrant Grid -->
        <div class="quad-content-row">
            <div id="draw_toolbar_quad" class="tv-draw-bar">
                <button class="tv-tool-btn active" id="btn_draw_sync_quad" title="Sync drawings across all 4 quadrants (ON)">🔄</button>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn active" data-tool="pointer" title="Cursor / Selection & Edit (✛)">✛</button>
                <button class="tv-tool-btn" id="btn_draw_magnet_quad" title="Magnet Mode (Snap to Candle OHLC) [M]">🧲</button>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" data-tool="trendline" title="Trend Line (╱)">╱</button>
                <button class="tv-tool-btn" data-tool="ray" title="Horizontal Ray (⟶)">⟶</button>
                <button class="tv-tool-btn" data-tool="extended" title="Extended Line (↔)">↔</button>
                <button class="tv-tool-btn" data-tool="arrow" title="Arrow (↗)">↗</button>
                <button class="tv-tool-btn" data-tool="channel" title="Parallel Channel (∥)">∥</button>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" data-tool="horizontal" title="Horizontal Line (―)">―</button>
                <button class="tv-tool-btn" data-tool="vertical" title="Vertical Line (│)">│</button>
                <button class="tv-tool-btn" data-tool="cross" title="Cross Line (┼)">┼</button>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" data-tool="rectangle" title="Rectangle / Supply & Demand Zone (▭)">▭</button>
                <button class="tv-tool-btn" data-tool="circle" title="Circle / Ellipse (◯)">◯</button>
                <button class="tv-tool-btn" data-tool="fib" title="Fibonacci Retracement (≡)">≡</button>
                <button class="tv-tool-btn" data-tool="measure" title="Price & Date Measure Ruler (📏)">📏</button>
                <button class="tv-tool-btn" data-tool="risk_reward" title="Long/Short Risk-Reward Box (🎯)">🎯</button>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" data-tool="brush" title="Smooth Brush / Freehand (✎)">✎</button>
                <button class="tv-tool-btn" data-tool="highlighter" title="Highlighter Marker (🖍️)">🖍️</button>
                <button class="tv-tool-btn" data-tool="text" title="Text Annotation (T)">T</button>
                <button class="tv-tool-btn" data-tool="callout" title="Callout Bubble (💬)">💬</button>
                <div class="tv-tool-sep"></div>
                <div class="tv-color-wrapper" title="Pick Drawing Color">
                    <div id="color_dot_quad" class="tv-color-dot" style="background-color: #00E5FF;"></div>
                    <input type="color" id="draw_color_quad" value="#00E5FF" class="tv-color-picker" />
                </div>
                <select id="draw_width_quad" class="tv-width-select" title="Line Stroke Width">
                    <option value="1">1px</option>
                    <option value="2" selected>2px</option>
                    <option value="3">3px</option>
                    <option value="4">4px</option>
                </select>
                <div class="tv-tool-sep"></div>
                <button class="tv-tool-btn" id="draw_undo_quad" title="Undo Last Drawing (Ctrl+Z)">↩</button>
                <button class="tv-tool-btn" id="draw_delete_selected_quad" title="Delete Selected Drawing (Del/Backspace)">✕</button>
                <button class="tv-tool-btn" id="draw_clear_quad" title="Clear All Drawings">🗑</button>
            </div>
            <div id="quad_grid" class="quad-grid">
                <!-- 1. MONTHLY -->
                <div class="quad-card" id="card_m">
                    <div class="qc-header">
                        <div class="qc-header-left">
                            <span class="qc-tf qc-tf-m">1M</span>
                            <span class="qc-sub">Macro: 5>20 EMA | AVWAP (Mar'20, Jun'22) | RSI 50</span>
                            <div class="qc-legend" id="legend_m"></div>
                        </div>
                        <div class="qc-header-right">
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('m', -0.25)" title="Zoom In (+)">+</button>
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('m', 0.25)" title="Zoom Out (-)">-</button>
                            <button class="qc-btn-zoom" onclick="resetQuadrant('m')" title="Reset View (⟲)">⟲</button>
                            <button class="qc-btn-zoom {'active' if show_candles else ''}" id="btn_candle_m" onclick="toggleCardCandle('m')" title="Toggle Candlesticks">🕯️</button>
                            <button class="qc-btn-zoom {'active' if show_line else ''}" id="btn_line_m" onclick="toggleCardLine('m')" title="Toggle Line Chart">📈</button>
                            <button class="qc-btn-zoom {'active' if show_volume else ''}" id="btn_vol_m" onclick="toggleCardVolume('m')" title="Hide / Show Volume">Vol</button>
                            <button class="qc-btn-zoom" onclick="toggleZoomCard('m')" title="Zoom / Restore Monthly">⛶</button>
                        </div>
                    </div>
                    <div class="qc-body" id="body_m">
                        <div class="qc-crosshair-v" id="ch_v_m"></div>
                        <div class="qc-main-wrap" id="wrap_main_m">
                            <div class="qc-main-canvas" id="canvas_main_m"></div>
                            <canvas id="draw_canvas_m" class="qc-draw-canvas"></canvas>
                        </div>
                        <div class="qc-rsi-canvas" id="canvas_rsi_m" style="{'display: block;' if show_rsi else 'display: none;'}">
                            <div class="hm-header"><span class="hm-title">Hilega Milega(by NK sir) [anuragM]</span><span class="hm-vals" id="hm_vals_m"></span></div>
                            <div class="rsi-bull-tint"></div>
                        </div>
                        <div class="tv-floating-tooltip" id="tt_m"></div>
                        <div class="hist-badge" id="hist_badge_m"></div>
                    </div>
                </div>

                <!-- 2. WEEKLY -->
                <div class="quad-card" id="card_w">
                    <div class="qc-header">
                        <div class="qc-header-left">
                            <span class="qc-tf qc-tf-w">1W</span>
                            <span class="qc-sub">Trend: 20>50>200 EMA</span>
                            <div class="qc-legend" id="legend_w"></div>
                        </div>
                        <div class="qc-header-right">
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('w', -0.25)" title="Zoom In (+)">+</button>
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('w', 0.25)" title="Zoom Out (-)">-</button>
                            <button class="qc-btn-zoom" onclick="resetQuadrant('w')" title="Reset View (⟲)">⟲</button>
                            <button class="qc-btn-zoom {'active' if show_candles else ''}" id="btn_candle_w" onclick="toggleCardCandle('w')" title="Toggle Candlesticks">🕯️</button>
                            <button class="qc-btn-zoom {'active' if show_line else ''}" id="btn_line_w" onclick="toggleCardLine('w')" title="Toggle Line Chart">📈</button>
                            <button class="qc-btn-zoom {'active' if show_volume else ''}" id="btn_vol_w" onclick="toggleCardVolume('w')" title="Hide / Show Volume">Vol</button>
                            <button class="qc-btn-zoom" onclick="toggleZoomCard('w')" title="Zoom / Restore Weekly">⛶</button>
                        </div>
                    </div>
                    <div class="qc-body" id="body_w">
                        <div class="qc-crosshair-v" id="ch_v_w"></div>
                        <div class="qc-main-wrap" id="wrap_main_w">
                            <div class="qc-main-canvas" id="canvas_main_w"></div>
                            <canvas id="draw_canvas_w" class="qc-draw-canvas"></canvas>
                        </div>
                        <div class="qc-rsi-canvas" id="canvas_rsi_w" style="{'display: block;' if show_rsi else 'display: none;'}">
                            <div class="hm-header"><span class="hm-title">Hilega Milega(by NK sir) [anuragM]</span><span class="hm-vals" id="hm_vals_w"></span></div>
                            <div class="rsi-bull-tint"></div>
                        </div>
                        <div class="tv-floating-tooltip" id="tt_w"></div>
                        <div class="hist-badge" id="hist_badge_w"></div>
                    </div>
                </div>

                <!-- 3. DAILY -->
                <div class="quad-card" id="card_d">
                    <div class="qc-header">
                        <div class="qc-header-left">
                            <span class="qc-tf qc-tf-d">1D</span>
                            <span class="qc-sub">Setup: 20>50>200 EMA</span>
                            <div class="qc-legend" id="legend_d"></div>
                        </div>
                        <div class="qc-header-right">
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('d', -0.25)" title="Zoom In (+)">+</button>
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('d', 0.25)" title="Zoom Out (-)">-</button>
                            <button class="qc-btn-zoom" onclick="resetQuadrant('d')" title="Reset View (⟲)">⟲</button>
                            <button class="qc-btn-zoom {'active' if show_candles else ''}" id="btn_candle_d" onclick="toggleCardCandle('d')" title="Toggle Candlesticks">🕯️</button>
                            <button class="qc-btn-zoom {'active' if show_line else ''}" id="btn_line_d" onclick="toggleCardLine('d')" title="Toggle Line Chart">📈</button>
                            <button class="qc-btn-zoom {'active' if show_volume else ''}" id="btn_vol_d" onclick="toggleCardVolume('d')" title="Hide / Show Volume">Vol</button>
                            <button class="qc-btn-zoom" onclick="toggleZoomCard('d')" title="Zoom / Restore Daily">⛶</button>
                        </div>
                    </div>
                    <div class="qc-body" id="body_d">
                        <div class="qc-crosshair-v" id="ch_v_d"></div>
                        <div class="qc-main-wrap" id="wrap_main_d">
                            <div class="qc-main-canvas" id="canvas_main_d"></div>
                            <canvas id="draw_canvas_d" class="qc-draw-canvas"></canvas>
                        </div>
                        <div class="qc-rsi-canvas" id="canvas_rsi_d" style="{'display: block;' if show_rsi else 'display: none;'}">
                            <div class="hm-header"><span class="hm-title">Hilega Milega(by NK sir) [anuragM]</span><span class="hm-vals" id="hm_vals_d"></span></div>
                            <div class="rsi-bull-tint"></div>
                        </div>
                        <div class="tv-floating-tooltip" id="tt_d"></div>
                        <div class="hist-badge" id="hist_badge_d"></div>
                    </div>
                </div>

                <!-- 4. 75-MINUTE -->
                <div class="quad-card" id="card_75">
                    <div class="qc-header">
                        <div class="qc-header-left">
                            <span class="qc-tf qc-tf-75">{intra_tf_label}</span>
                            <span class="qc-sub">{q4_sub_label}</span>
                            <div class="qc-legend" id="legend_75"></div>
                        </div>
                        <div class="qc-header-right">
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('75', -0.25)" title="Zoom In (+)">+</button>
                            <button class="qc-btn-zoom" onclick="zoomQuadrant('75', 0.25)" title="Zoom Out (-)">-</button>
                            <button class="qc-btn-zoom" onclick="resetQuadrant('75')" title="Reset View (⟲)">⟲</button>
                            <button class="qc-btn-zoom {'active' if show_candles else ''}" id="btn_candle_75" onclick="toggleCardCandle('75')" title="Toggle Candlesticks">🕯️</button>
                            <button class="qc-btn-zoom {'active' if show_line else ''}" id="btn_line_75" onclick="toggleCardLine('75')" title="Toggle Line Chart">📈</button>
                            <button class="qc-btn-zoom {'active' if show_volume else ''}" id="btn_vol_75" onclick="toggleCardVolume('75')" title="Hide / Show Volume">Vol</button>
                            <button class="qc-btn-zoom" onclick="toggleZoomCard('75')" title="Zoom / Restore {intra_tf_label}">⛶</button>
                        </div>
                    </div>
                    <div class="qc-body" id="body_75">
                        <div class="qc-crosshair-v" id="ch_v_75"></div>
                        <div class="qc-main-wrap" id="wrap_main_75">
                            <div class="qc-main-canvas" id="canvas_main_75"></div>
                            <canvas id="draw_canvas_75" class="qc-draw-canvas"></canvas>
                        </div>
                        <div class="qc-rsi-canvas" id="canvas_rsi_75" style="{'display: block;' if show_rsi else 'display: none;'}">
                            <div class="hm-header"><span class="hm-title">Hilega Milega(by NK sir) [anuragM]</span><span class="hm-vals" id="hm_vals_75"></span></div>
                            <div class="rsi-bull-tint"></div>
                        </div>
                        <div class="tv-floating-tooltip" id="tt_75"></div>
                        <div class="hist-badge" id="hist_badge_75"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const QUAD_DATA = {quad_data_json};
        const quadsRegistry = {{}};
        var globalTooltipEnabled = true;

        function initQuadrant(id, payload, isIntraday) {{
            const mainContainer = document.getElementById("canvas_main_" + id);
            const rsiContainer = document.getElementById("canvas_rsi_" + id);
            const legendEl = document.getElementById("legend_" + id);
            const tooltipEl = document.getElementById("tt_" + id);
            const bodyEl = document.getElementById("body_" + id);
            const hasRsi = !!(payload && payload.hasRsi);

            if (!mainContainer) return null;

            const w = mainContainer.clientWidth || 300;
            const h = mainContainer.clientHeight || 200;

            const isLightInit = (QUAD_DATA.theme === 'light');

            const chartOptions = {{
                width: w,
                height: h,
                layout: {{
                    background: {{ type: 'solid', color: isLightInit ? '#ffffff' : '#131722' }},
                    textColor: isLightInit ? '#131722' : '#94A3B8',
                    fontSize: 10,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, sans-serif",
                }},
                grid: {{
                    vertLines: {{ color: isLightInit ? '#f0f3fa' : '#1E222D', style: 1 }},
                    horzLines: {{ color: isLightInit ? '#f0f3fa' : '#1E222D', style: 1 }},
                }},
                crosshair: {{
                    mode: 0,
                    vertLine: {{
                        color: isLightInit ? 'rgba(0, 0, 0, 0.22)' : 'rgba(255, 255, 255, 0.28)',
                        width: 1,
                        style: 3,
                        labelVisible: true,
                        labelBackgroundColor: isLightInit ? '#2563EB' : '#1E293B',
                    }},
                    horzLine: {{
                        color: isLightInit ? 'rgba(0, 0, 0, 0.22)' : 'rgba(255, 255, 255, 0.28)',
                        width: 1,
                        style: 3,
                        labelVisible: true,
                        labelBackgroundColor: isLightInit ? '#2563EB' : '#1E293B',
                    }},
                }},
                rightPriceScale: {{
                    visible: true,
                    borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                    borderVisible: true,
                    scaleMargins: {{ top: 0.1, bottom: 0.18 }},
                    alignLabels: true,
                }},
                localization: {{
                    locale: 'en-IN',
                    dateFormat: 'yyyy-MM-dd',
                    timeFormatter: (time) => {{
                        if (typeof time === 'number') {{
                            const d = new Date(time * 1000);
                            return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }}) + ' ' +
                                   d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + ' IST';
                        }}
                        return String(time);
                    }},
                }},
                timeScale: {{
                    borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                    timeVisible: isIntraday,
                    secondsVisible: false,
                    borderVisible: false,
                    tickMarkFormatter: (time, tickMarkType, locale) => {{
                        if (typeof time === 'number') {{
                            const d = new Date(time * 1000);
                            if (tickMarkType === 0) {{
                                return d.getFullYear();
                            }} else if (tickMarkType === 1) {{
                                return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', month: 'short' }});
                            }} else if (tickMarkType === 2) {{
                                return d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: 'numeric' }});
                            }} else {{
                                return d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }});
                            }}
                        }}
                        return null;
                    }},
                }},
                handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
            }};

            const mainChart = LightweightCharts.createChart(mainContainer, chartOptions);

            // Candles with active right price badge and dashed current price line
            const isCandleInit = QUAD_DATA.showCandles !== false;
            const candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#089981',
                downColor: '#F23645',
                borderVisible: false,
                wickUpColor: '#089981',
                wickDownColor: '#F23645',
                visible: isCandleInit,
                lastValueVisible: true,        // Active Price Badge on Right Y-axis
                priceLineVisible: true,        // Active Price Line
                priceLineWidth: 1,
                priceLineStyle: LightweightCharts.LineStyle.Dashed,
            }});
            if (payload.candles && payload.candles.length) {{
                candleSeries.setData(payload.candles);
            }}

            // Line Series (Close price line chart)
            const isLineInit = QUAD_DATA.showLine !== false;
            const lineSeries = mainChart.addLineSeries({{
                color: '#2962FF',
                lineWidth: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                axisLabelVisible: false,
                title: '',
                visible: isLineInit,
            }});
            if (payload.candles && payload.candles.length) {{
                const lineData = payload.candles.map(c => ({{ time: c.time, value: c.close }}));
                lineSeries.setData(lineData);
            }}

            // Volume
            const isVolInit = QUAD_DATA.showVolume !== false;
            const volumeSeries = mainChart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: 'volume',
                lastValueVisible: false,
                priceLineVisible: false,
                visible: isVolInit,
            }});
            mainChart.priceScale('volume').applyOptions({{
                scaleMargins: {{
                    top: 0.82,
                    bottom: 0,
                }},
            }});
            if (payload.volumes && payload.volumes.length) {{
                volumeSeries.setData(payload.volumes);
            }}

            // EMAs
            // EMAs & Indicator Overlays (AVWAP, etc.)
            const emaSeriesMap = {{}};
            if (payload.emas) {{
                Object.keys(payload.emas).forEach(emaName => {{
                    const cfg = payload.emas[emaName];
                    let lStyle = LightweightCharts.LineStyle.Solid;
                    if (cfg.lineStyle === 1 || cfg.lineStyle === 'dotted') {{
                        lStyle = LightweightCharts.LineStyle.Dotted;
                    }} else if (cfg.lineStyle === 2 || cfg.lineStyle === 'dashed') {{
                        lStyle = LightweightCharts.LineStyle.Dashed;
                    }}
                    const s = mainChart.addLineSeries({{
                        color: cfg.color,
                        lineWidth: cfg.lineWidth || ((emaName === 'Daily_EMA_20') ? 2.2 : 1.5),
                        lineStyle: lStyle,
                        lastValueVisible: false,
                        priceLineVisible: false,
                        axisLabelVisible: false,
                        title: '',
                    }});
                    s.setData(cfg.data);
                    emaSeriesMap[emaName] = {{ series: s, color: cfg.color, title: cfg.name || emaName }};
                }});
            }}

            // Pivots
            if (payload.pivots) {{
                Object.keys(payload.pivots).forEach(pName => {{
                    const pCfg = payload.pivots[pName];
                    const ps = mainChart.addLineSeries({{
                        color: pCfg.color,
                        lineWidth: 1.8,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        lastValueVisible: false,
                        priceLineVisible: false,
                        axisLabelVisible: false,
                        title: '',
                    }});
                    ps.setData(pCfg.data);
                }});
            }}

            // RSI Chart
            let rsiChart = null;
            let hmCloud = null;
            let rsiSeries = null;
            let e3Series = null;
            let w21Series = null;
            if (payload.hasRsi && rsiContainer) {{
                const rw = rsiContainer.clientWidth || 300;
                const rh = rsiContainer.clientHeight || 100;
                const rsiOptions = {{
                    width: rw,
                    height: rh,
                    layout: {{
                        background: {{ type: 'solid', color: isLightInit ? '#ffffff' : '#131722' }},
                        textColor: isLightInit ? '#131722' : '#94A3B8',
                        fontSize: 9,
                    }},
                    grid: {{
                        vertLines: {{ color: isLightInit ? '#f0f3fa' : '#1E222D', style: 1 }},
                        horzLines: {{ color: isLightInit ? '#f0f3fa' : '#1E222D', style: 1 }},
                    }},
                    crosshair: {{
                        mode: 0,
                        vertLine: {{ color: isLightInit ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.2)', width: 1, style: 3, labelVisible: false }},
                        horzLine: {{ color: isLightInit ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.2)', width: 1, style: 3, labelVisible: false }},
                    }},
                    rightPriceScale: {{
                        visible: true,
                        borderColor: isLightInit ? '#e0e3eb' : '#2A2E39',
                        borderVisible: false,
                        scaleMargins: {{ top: 0.08, bottom: 0.08 }},
                    }},
                    timeScale: {{
                        visible: false,
                        borderVisible: false,
                    }},
                    handleScroll: {{ mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
                    handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
                }};

                rsiChart = LightweightCharts.createChart(rsiContainer, rsiOptions);

                // Baseline Cloud for Hilega Milega (Pink cloud above 50, Soft blue cloud below 50)
                try {{
                    hmCloud = rsiChart.addBaselineSeries({{
                        baseValue: {{ type: 'price', price: 50 }},
                        topFillColor1: isLightInit ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                        topFillColor2: isLightInit ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                        bottomFillColor1: isLightInit ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                        bottomFillColor2: isLightInit ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                        topLineColor: isLightInit ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                        bottomLineColor: isLightInit ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)',
                        lineWidth: 1,
                        lastValueVisible: false,
                        priceLineVisible: false
                    }});
                    if (payload.rsi && payload.rsi.length) hmCloud.setData(payload.rsi);
                }} catch(e) {{}}

                const rsiColor = isLightInit ? '#131722' : '#F8FAFC';
                rsiSeries = rsiChart.addLineSeries({{
                    color: rsiColor,
                    lineWidth: 1.8,
                    lastValueVisible: true,
                    priceLineVisible: false,
                    title: 'RSI(9)',
                }});
                if (payload.rsi && payload.rsi.length) {{
                    rsiSeries.setData(payload.rsi);
                }}

                if (payload.rsi_ema3 && payload.rsi_ema3.length) {{
                    e3Series = rsiChart.addLineSeries({{
                        color: '#4CAF50',
                        lineWidth: 1.8,
                        lastValueVisible: true,
                        priceLineVisible: false,
                        title: 'EMA(3)',
                    }});
                    e3Series.setData(payload.rsi_ema3);
                }}

                if (payload.rsi_wma21 && payload.rsi_wma21.length) {{
                    w21Series = rsiChart.addLineSeries({{
                        color: '#FF5252',
                        lineWidth: 1.8,
                        lastValueVisible: true,
                        priceLineVisible: false,
                        title: 'WMA(21)',
                    }});
                    w21Series.setData(payload.rsi_wma21);
                }}

                // Solid 50 Blue Line (#7695F9) with Price Badge
                // Solid 50 Blue Line (#7695F9) with Price Badge
                rsiSeries.createPriceLine({{ price: 50, color: '#7695F9', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: '50' }});
                rsiSeries.createPriceLine({{ price: 70, color: '#94A3B8', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false, title: '' }});
                rsiSeries.createPriceLine({{ price: 30, color: '#94A3B8', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false, title: '' }});

                // Status message in header if awaiting 9 candles
                const hmValsHeader = document.getElementById('hm_vals_' + id);
                if ((!payload.rsi || payload.rsi.length === 0) && hmValsHeader) {{
                    const cCount = payload.candleCount || 0;
                    hmValsHeader.innerHTML = `<span style="color:#f59e0b;font-size:10px;font-weight:600;">(Awaiting 9 completed candles: ${{cCount}}/9)</span>`;
                }}

                // Fluid, non-blocking visible ranges sync between main & RSI charts
                let isSyncing = false;
                let syncRaf = null;
                mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (isSyncing || !range || !rsiChart) return;
                    if (syncRaf) cancelAnimationFrame(syncRaf);
                    syncRaf = requestAnimationFrame(() => {{
                        if (!rsiChart) return;
                        isSyncing = true;
                        try {{
                            const cur = rsiChart.timeScale().getVisibleLogicalRange();
                            if (!cur || Math.abs(cur.from - range.from) > 0.05 || Math.abs(cur.to - range.to) > 0.05) {{
                                rsiChart.timeScale().setVisibleLogicalRange(range);
                            }}
                        }} catch(e) {{}}
                        isSyncing = false;
                    }});
                }});
                rsiChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (isSyncing || !range || !mainChart) return;
                    if (syncRaf) cancelAnimationFrame(syncRaf);
                    syncRaf = requestAnimationFrame(() => {{
                        if (!mainChart) return;
                        isSyncing = true;
                        try {{
                            const cur = mainChart.timeScale().getVisibleLogicalRange();
                            if (!cur || Math.abs(cur.from - range.from) > 0.05 || Math.abs(cur.to - range.to) > 0.05) {{
                                mainChart.timeScale().setVisibleLogicalRange(range);
                            }}
                        }} catch(e) {{}}
                        isSyncing = false;
                    }});
                }});
            }}

            // Pre-index candle & indicator data for instant O(1) crosshair performance
            const cMap = new Map();
            (payload.candles || []).forEach(p => {{ cMap.set(p.time, p); }});
            const rMap = new Map();
            (payload.rsi || []).forEach(p => {{ rMap.set(p.time, p); }});
            const eMap = new Map();
            (payload.rsi_ema3 || []).forEach(p => {{ eMap.set(p.time, p); }});
            const wMap = new Map();
            (payload.rsi_wma21 || []).forEach(p => {{ wMap.set(p.time, p); }});
            const emaDataMaps = {{}};
            if (payload.emas) {{
                Object.keys(payload.emas).forEach(eName => {{
                    const dMap = new Map();
                    (payload.emas[eName].data || []).forEach(pt => {{
                        dMap.set(pt.time, pt.value);
                    }});
                    emaDataMaps[eName] = dMap;
                }});
            }}

            // Initial view range
            if (payload.candles && payload.candles.length) {{
                const totalBars = payload.candles.length;
                const viewBars = Math.min(totalBars, isIntraday ? 75 : 90);
                mainChart.timeScale().setVisibleLogicalRange({{
                    from: Math.max(0, totalBars - viewBars),
                    to: totalBars + 2,
                }});
                const lastC = payload.candles[payload.candles.length - 1];
                const prevC = payload.candles.length > 1 ? payload.candles[payload.candles.length - 2] : lastC;
                const chg = lastC.close - prevC.close;
                const chgP = (chg / prevC.close) * 100;
                const clr = chg >= 0 ? '#089981' : '#F23645';
                if (legendEl) {{
                    legendEl.innerHTML = `C: <b>₹${{lastC.close.toFixed(2)}}</b> <span style="color:${{clr}}">${{chg >= 0 ? '+' : ''}}${{chgP.toFixed(2)}}%</span>`;
                }}
            }}

            // History loaded notification pill & range change listener
            let histTimer = null;
            mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                if (!range) return;
                if (range.from <= 10 && payload.candles && payload.candles.length > 30) {{
                    const fc = payload.candles[0];
                    let dLabel = '';
                    if (typeof fc.time === 'number') {{
                        const fd = new Date(fc.time * 1000);
                        dLabel = fd.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric' }}) + ' ' +
                                 fd.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + ' IST';
                    }} else {{
                        dLabel = String(fc.time);
                    }}
                    const histBadge = document.getElementById("hist_badge_" + id);
                    if (histBadge) {{
                        histBadge.textContent = `📅 History Loaded: ${{dLabel}} (${{payload.candles.length}} bars)`;
                        histBadge.style.opacity = '1';
                        clearTimeout(histTimer);
                        histTimer = setTimeout(() => {{ histBadge.style.opacity = '0'; }}, 2500);
                    }}
                }}
            }});

            // Crosshair Tooltip & Extended Vertical Crosshair into Hilega Milega Pane
            const chVEl = document.getElementById("ch_v_" + id);
            const hmValsEl = document.getElementById("hm_vals_" + id);

            function handleUnifiedCrosshair(param, source) {{
                if (!param || !param.time || !param.point || param.point.x < 0) {{
                    if (tooltipEl) tooltipEl.style.display = 'none';
                    if (chVEl) chVEl.style.display = 'none';
                    return;
                }}

                // 1. Extend vertical crosshair line across the entire canvas height (Price Chart + Hilega Milega)
                if (chVEl) {{
                    chVEl.style.left = param.point.x + 'px';
                    chVEl.style.display = 'block';
                }}

                const tMatch = (typeof param.time === 'number') ? param.time : (param.time.year ? `${{param.time.year}}-${{String(param.time.month).padStart(2,'0')}}-${{String(param.time.day).padStart(2,'0')}}` : String(param.time));

                // 2. Update Hilega Milega indicator readings (RSI, EMA 3, WMA 21) in real time
                let rNum = null, eNum = null, wNum = null, rColor = '#089981';
                if (hasRsi) {{
                    let rVal = null, eVal = null, wVal = null;
                    if (param.seriesData && param.seriesData.get) {{
                        if (rsiSeries) rVal = param.seriesData.get(rsiSeries);
                        if (e3Series) eVal = param.seriesData.get(e3Series);
                        if (w21Series) wVal = param.seriesData.get(w21Series);
                    }}

                    // Fallback to O(1) Map lookup
                    if (!rVal || rVal.value === undefined) {{
                        const rPt = rMap.get(tMatch) ?? rMap.get(param.time);
                        if (rPt) rVal = {{ value: rPt.value }};
                    }}
                    if (!eVal || eVal.value === undefined) {{
                        const ePt = eMap.get(tMatch) ?? eMap.get(param.time);
                        if (ePt) eVal = {{ value: ePt.value }};
                    }}
                    if (!wVal || wVal.value === undefined) {{
                        const wPt = wMap.get(tMatch) ?? wMap.get(param.time);
                        if (wPt) wVal = {{ value: wPt.value }};
                    }}

                    if (rVal && rVal.value !== undefined) {{
                        rNum = Number(rVal.value).toFixed(1);
                        eNum = (eVal && eVal.value !== undefined) ? Number(eVal.value).toFixed(1) : '--';
                        wNum = (wVal && wVal.value !== undefined) ? Number(wVal.value).toFixed(1) : '--';
                        rColor = rVal.value >= 50 ? '#089981' : '#F23645';
                        if (hmValsEl) {{
                            hmValsEl.innerHTML = `<span style="color:${{rColor}};font-weight:700;">RSI: ${{rNum}}</span> | <span style="color:#4CAF50;">EMA(3): ${{eNum}}</span> | <span style="color:#FF5252;">WMA(21): ${{wNum}}</span>`;
                        }}
                    }} else if (hmValsEl) {{
                        hmValsEl.innerHTML = `<span style="color:#64748B;font-size:10px;">RSI: -- (Awaiting 9 completed candles)</span>`;
                    }}
                }}

                // 3. Update OHLC & Indicator Tooltip + Card Legend
                let bar = null;
                if (param.seriesData && param.seriesData.get) {{
                    bar = param.seriesData.get(candleSeries);
                }}
                let o = 0, h = 0, l = 0, c = 0;
                if (bar && typeof bar.open === 'number' && typeof bar.close === 'number') {{
                    o = bar.open; h = bar.high; l = bar.low; c = bar.close;
                }} else {{
                    let lVal = null;
                    if (param.seriesData && param.seriesData.get && lineSeries) {{
                        lVal = param.seriesData.get(lineSeries);
                    }}
                    if (lVal && lVal.value !== undefined) {{
                        o = h = l = c = lVal.value;
                    }} else {{
                        const cPt = cMap.get(tMatch) || cMap.get(param.time);
                        if (cPt) {{
                            o = cPt.open; h = cPt.high; l = cPt.low; c = cPt.close;
                        }}
                    }}
                }}

                if (o > 0 || c > 0) {{
                    const diff = c - o;
                    const pct = o !== 0 ? (diff / o) * 100 : 0;
                    const sign = diff >= 0 ? '+' : '';
                    const diffCls = diff >= 0 ? 'tt-up' : 'tt-dn';

                    let tStr = '';
                    if (typeof param.time === 'number') {{
                        const d = new Date(param.time * 1000);
                        tStr = d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', month: 'short', day: 'numeric', year: 'numeric' }}) + ' ' +
                               d.toLocaleTimeString('en-IN', {{ timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }}) + ' IST';
                    }} else if (param.time) {{
                        tStr = param.time.year ? `${{param.time.year}}-${{String(param.time.month).padStart(2,'0')}}-${{String(param.time.day).padStart(2,'0')}}` : String(param.time);
                    }}

                    if (legendEl) {{
                        legendEl.innerHTML = `O: <b>${{o.toFixed(1)}}</b> H: <b>${{h.toFixed(1)}}</b> L: <b>${{l.toFixed(1)}}</b> C: <b>${{c.toFixed(1)}}</b> <span style="color:${{diff >= 0 ? '#089981' : '#F23645'}}">${{sign}}${{pct.toFixed(2)}}%</span>`;
                    }}

                    if (tooltipEl && bodyEl) {{
                        if (!globalTooltipEnabled) {{
                            tooltipEl.style.display = 'none';
                        }} else {{
                            let emasHtml = '';
                            Object.keys(emaSeriesMap).forEach(eName => {{
                                const sObj = emaSeriesMap[eName];
                                let eValNum = null;
                                if (param.seriesData && param.seriesData.get) {{
                                    const eVal = param.seriesData.get(sObj.series);
                                    if (eVal && eVal.value !== undefined) eValNum = eVal.value;
                                }}
                                if (eValNum === null && emaDataMaps[eName]) {{
                                    const fallbackVal = emaDataMaps[eName].get(tMatch) ?? emaDataMaps[eName].get(param.time);
                                    if (fallbackVal !== undefined && fallbackVal !== null) eValNum = fallbackVal;
                                }}
                                if (eValNum !== null) {{
                                    const dispName = (eName === 'Daily_EMA_20') ? 'Daily 20 EMA' : (sObj.title || eName);
                                    emasHtml += `<div class="tt-row"><span class="tt-lbl" style="color:${{sObj.color}}">${{dispName}}</span><span class="tt-val">₹${{Number(eValNum).toFixed(2)}}</span></div>`;
                                }}
                            }});

                            let hmHtml = '';
                            if (hasRsi && rNum !== null) {{
                                hmHtml = `
                                    <div style="border-top: 1px solid rgba(255,255,255,0.12); margin: 3px 0; padding-top: 2px;"></div>
                                    <div class="tt-row"><span class="tt-lbl">RSI (9)</span><span class="tt-val" style="color:${{rColor}}; font-weight:700;">${{rNum}}</span></div>
                                    <div class="tt-row"><span class="tt-lbl" style="color:#4CAF50;">EMA (3)</span><span class="tt-val">${{eNum}}</span></div>
                                    <div class="tt-row"><span class="tt-lbl" style="color:#FF5252;">WMA (21)</span><span class="tt-val">${{wNum}}</span></div>
                                `;
                            }}

                            tooltipEl.innerHTML = `
                                <div class="tt-date">${{tStr}}</div>
                                <div class="tt-row"><span class="tt-lbl">Open</span><span class="tt-val">₹${{o.toFixed(2)}}</span></div>
                                <div class="tt-row"><span class="tt-lbl">High</span><span class="tt-val">₹${{h.toFixed(2)}}</span></div>
                                <div class="tt-row"><span class="tt-lbl">Low</span><span class="tt-val">₹${{l.toFixed(2)}}</span></div>
                                <div class="tt-row"><span class="tt-lbl">Close</span><span class="tt-val">₹${{c.toFixed(2)}}</span></div>
                                <div class="tt-row"><span class="tt-lbl">Change</span><span class="tt-val ${{diffCls}}">${{sign}}${{diff.toFixed(2)}} (${{sign}}${{pct.toFixed(2)}}%)</span></div>
                                ${{emasHtml}}
                                ${{hmHtml}}
                            `;
                            tooltipEl.style.display = 'block';

                            const boxW = 200;
                            const boxH = hasRsi ? 210 : 160;
                            let left = param.point.x + 15;
                            let mouseY = param.point.y;
                            if (source === 'rsi' && rsiContainer) {{
                                mouseY += rsiContainer.offsetTop;
                            }}
                            let top = mouseY + 10;
                            const maxW = bodyEl.clientWidth;
                            const maxH = bodyEl.clientHeight;

                            if (left + boxW > maxW) left = param.point.x - boxW - 15;
                            if (left < 5) left = 5;
                            if (top + boxH > maxH) top = mouseY - boxH - 10;
                            if (top < 5) top = 5;

                            tooltipEl.style.left = left + 'px';
                            tooltipEl.style.top = top + 'px';
                        }}
                    }}
                }}
            }}

            mainChart.subscribeCrosshairMove(p => handleUnifiedCrosshair(p, 'main'));
            if (rsiChart) {{
                rsiChart.subscribeCrosshairMove(p => handleUnifiedCrosshair(p, 'rsi'));
            }}
            if (bodyEl) {{
                bodyEl.addEventListener('mouseleave', () => {{
                    if (chVEl) chVEl.style.display = 'none';
                    if (tooltipEl) tooltipEl.style.display = 'none';
                }});
            }}

            function candleTimeToSeconds(t) {{
                if (typeof t === 'number') {{
                    return id === '75' ? (t + 2250) : t;
                }}
                if (typeof t === 'string') {{
                    const parts = t.split('-');
                    if (parts.length === 3) {{
                        const y = parseInt(parts[0], 10);
                        const m = parseInt(parts[1], 10);
                        const d = parseInt(parts[2], 10);
                        if (id === 'm') {{
                            return Math.floor(Date.UTC(y, m - 1, 15, 6, 52, 30) / 1000);
                        }}
                        if (id === 'w') {{
                            const dt = new Date(Date.UTC(y, m - 1, d, 6, 52, 30));
                            const uDay = dt.getUTCDay();
                            let shift = (uDay === 0) ? -4 : (uDay === 5 ? -2 : (uDay === 1 ? 2 : (3 - uDay)));
                            dt.setUTCDate(dt.getUTCDate() + shift);
                            return Math.floor(dt.getTime() / 1000);
                        }}
                        return Math.floor(Date.UTC(y, m - 1, d, 6, 52, 30) / 1000);
                    }}
                }}
                if (typeof t === 'object' && t && t.year) {{
                    const y = t.year;
                    const m = t.month;
                    const d = t.day;
                    if (id === 'm') {{
                        return Math.floor(Date.UTC(y, m - 1, 15, 6, 52, 30) / 1000);
                    }}
                    if (id === 'w') {{
                        const dt = new Date(Date.UTC(y, m - 1, d, 6, 52, 30));
                        const uDay = dt.getUTCDay();
                        let shift = (uDay === 0) ? -4 : (uDay === 5 ? -2 : (uDay === 1 ? 2 : (3 - uDay)));
                        dt.setUTCDate(dt.getUTCDate() + shift);
                        return Math.floor(dt.getTime() / 1000);
                    }}
                    return Math.floor(Date.UTC(y, m - 1, d, 6, 52, 30) / 1000);
                }}
                return 0;
            }}

            const candleTimestamps = (payload.candles || []).map(c => candleTimeToSeconds(c.time));

            return {{ mainChart, candleSeries, lineSeries, candleTimestamps, rsiChart, hmCloud, rsiSeries, emaSeriesMap, volumeSeries, isVolVisible: isVolInit, isCandleVisible: isCandleInit, isLineVisible: isLineInit, mainContainer, rsiContainer, bodyEl, hasRsi: payload.hasRsi, payload }};
        }}

        // Initialize all 4 quadrants
        quadsRegistry['m'] = initQuadrant('m', QUAD_DATA.m, false);
        quadsRegistry['w'] = initQuadrant('w', QUAD_DATA.w, false);
        quadsRegistry['d'] = initQuadrant('d', QUAD_DATA.d, false);
        quadsRegistry['75'] = initQuadrant('75', QUAD_DATA.intra, !!QUAD_DATA.intra_is_intraday);

        // --- QUADRANT ZOOM & TIME-RANGE JUMP ENGINE ---
        let currentQuadMode = 'all';

        function zoomQuadrant(id, factor) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart) return;
            const ts = q.mainChart.timeScale();
            const range = ts.getVisibleLogicalRange();
            if (!range) return;
            const span = range.to - range.from;
            const delta = span * factor;
            ts.setVisibleLogicalRange({{
                from: range.from - delta,
                to: range.to + delta
            }});
        }}

        function resetQuadrant(id) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart) return;
            q.mainChart.timeScale().fitContent();
        }}

        function masterZoom(factor) {{
            const activeIds = (currentQuadMode !== 'all') ? [currentQuadMode] : ['m', 'w', 'd', '75'];
            activeIds.forEach(id => {{
                zoomQuadrant(id, factor);
            }});
        }}

        function masterReset() {{
            const activeIds = (currentQuadMode !== 'all') ? [currentQuadMode] : ['m', 'w', 'd', '75'];
            activeIds.forEach(id => {{
                resetQuadrant(id);
            }});
        }}

        function applyTimeRangeToQuadrant(id, rangeType) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart || !q.payload || !q.payload.candles) return;
            const candles = q.payload.candles;
            const tot = candles.length;
            if (rangeType === 'ALL' || !tot) {{
                q.mainChart.timeScale().fitContent();
                return;
            }}
            const barMap = {{
                '75': {{ '1D': 5, '5D': 25, '1M': 110, '1Y': 1250 }},
                'd':  {{ '1D': 5, '5D': 5,  '1M': 22,  '1Y': 252 }},
                'w':  {{ '1D': 4, '5D': 4,  '1M': 5,   '1Y': 52 }},
                'm':  {{ '1D': 6, '5D': 6,  '1M': 6,   '1Y': 12 }}
            }};
            const count = (barMap[id] && barMap[id][rangeType]) ? barMap[id][rangeType] : 50;
            q.mainChart.timeScale().setVisibleLogicalRange({{
                from: Math.max(0, tot - count),
                to: tot + 2
            }});
        }}

        function masterTimeRange(rangeType) {{
            document.querySelectorAll('.qm-range-btn').forEach(b => {{
                b.classList.toggle('active', b.getAttribute('data-range') === rangeType);
            }});
            const activeIds = (currentQuadMode !== 'all') ? [currentQuadMode] : ['m', 'w', 'd', '75'];
            activeIds.forEach(id => {{
                applyTimeRangeToQuadrant(id, rangeType);
            }});
        }}

        // --- 4-QUADRANT SYNCHRONIZED DRAWING TOOLS ENGINE ---
        const quadToolbar = document.getElementById("draw_toolbar_quad");
        const syncBtnQuad = document.getElementById("btn_draw_sync_quad");
        const magnetBtnQuad = document.getElementById("btn_draw_magnet_quad");
        const colorInputQuad = document.getElementById("draw_color_quad");
        const colorDotQuad = document.getElementById("color_dot_quad");
        const widthInputQuad = document.getElementById("draw_width_quad");
        const undoBtnQuad = document.getElementById("draw_undo_quad");
        const deleteSelBtnQuad = document.getElementById("draw_delete_selected_quad");
        const clearBtnQuad = document.getElementById("draw_clear_quad");

        let activeQuadTool = 'pointer';
        let activeQuadColor = '#00E5FF';
        let activeQuadWidth = 2;
        let isSyncEnabled = true;
        let isMagnetEnabled = false;
        let lastSnappedPoint = null;
        const globalQuadDrawings = [];
        let selectedDrawing = null;
        let isQuadDrawing = false;
        let activeDrawingQuad = null;
        let quadStartPoint = null;
        let quadCurrentPoint = null;
        let quadBrushPoints = [];
        let isDraggingHandle = null;
        let isDraggingDrawing = false;
        let dragStartScreen = null;
        let dragInitialPoints = null;

        function logicalToTimestamp(id, logical) {{
            const q = quadsRegistry[id];
            if (!q || !q.candleTimestamps || q.candleTimestamps.length === 0) return Math.floor(Date.now() / 1000);
            const tsList = q.candleTimestamps;
            const len = tsList.length;
            if (len === 1) return tsList[0];

            if (logical <= 0) {{
                const step = Math.max(1, tsList[1] - tsList[0]);
                return tsList[0] + logical * step;
            }}
            if (logical >= len - 1) {{
                const step = Math.max(1, tsList[len - 1] - tsList[len - 2]);
                return tsList[len - 1] + (logical - (len - 1)) * step;
            }}

            const base = Math.floor(logical);
            const frac = logical - base;
            return tsList[base] + frac * (tsList[base + 1] - tsList[base]);
        }}

        function timestampToLogical(id, targetTs) {{
            const q = quadsRegistry[id];
            if (!q || !q.candleTimestamps || q.candleTimestamps.length === 0) return 0;
            const tsList = q.candleTimestamps;
            const len = tsList.length;
            if (len === 1) return 0;

            if (targetTs <= tsList[0]) {{
                const step = Math.max(1, tsList[1] - tsList[0]);
                return (targetTs - tsList[0]) / step;
            }}
            if (targetTs >= tsList[len - 1]) {{
                const step = Math.max(1, tsList[len - 1] - tsList[len - 2]);
                return (len - 1) + (targetTs - tsList[len - 1]) / step;
            }}

            let low = 0, high = len - 1;
            while (low <= high) {{
                const mid = (low + high) >> 1;
                const midVal = tsList[mid];
                if (midVal === targetTs) return mid;
                if (midVal < targetTs) low = mid + 1;
                else high = mid - 1;
            }}

            const idxLow = Math.max(0, Math.min(len - 2, high));
            const idxHigh = idxLow + 1;
            const t0 = tsList[idxLow];
            const t1 = tsList[idxHigh];
            const span = Math.max(1, t1 - t0);
            const frac = (targetTs - t0) / span;
            return idxLow + frac;
        }}

        function getActivePriceSeries(q) {{
            if (!q) return null;
            if (q.candleSeries && q.isCandleVisible !== false) return q.candleSeries;
            if (q.lineSeries && q.isLineVisible !== false) return q.lineSeries;
            return q.candleSeries || q.lineSeries || null;
        }}

        function quadScreenToChart(id, px, py) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart) return {{ time: Math.floor(Date.now() / 1000), price: 0, origX: px, origY: py, sourceQuad: id }};
            const s = getActivePriceSeries(q);
            let logical = 0;
            let price = 0;
            try {{
                const l = q.mainChart.timeScale().coordinateToLogical(px);
                if (l !== null && !isNaN(l)) logical = l;
            }} catch(e) {{}}
            try {{
                if (s) {{
                    const p = s.coordinateToPrice(py);
                    if (p !== null && !isNaN(p)) price = p;
                }}
            }} catch(e) {{}}

            const time = logicalToTimestamp(id, logical);
            return {{ time: time, price: price, logical: logical, origX: px, origY: py, sourceQuad: id }};
        }}

        function quadLogicalToCoordinate(q, logical) {{
            if (!q || !q.mainChart || logical === null || isNaN(logical)) return null;
            const ts = q.mainChart.timeScale();
            const i0 = Math.floor(logical);
            const i1 = i0 + 1;
            try {{
                const c0 = ts.logicalToCoordinate(i0);
                const c1 = ts.logicalToCoordinate(i1);
                if (c0 !== null && c1 !== null && !isNaN(c0) && !isNaN(c1) && c0 !== c1) {{
                    return c0 + (logical - i0) * (c1 - c0);
                }}
            }} catch(e) {{}}

            try {{
                const range = ts.getVisibleLogicalRange();
                if (range && range.from !== null && range.to !== null) {{
                    const rf = Math.round(range.from);
                    const rt = Math.round(range.to);
                    if (rt !== rf) {{
                        const xf = ts.logicalToCoordinate(rf);
                        const xt = ts.logicalToCoordinate(rt);
                        if (xf !== null && xt !== null && !isNaN(xf) && !isNaN(xt) && xt !== xf) {{
                            const step = (xt - xf) / (rt - rf);
                            return xf + (logical - rf) * step;
                        }}
                    }}
                }}
            }} catch(e) {{}}

            return null;
        }}

        function quadChartToScreen(id, pt) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart || !pt) return {{ x: pt ? (pt.origX || 0) : 0, y: pt ? (pt.origY || 0) : 0 }};
            const s = getActivePriceSeries(q);

            let logical = 0;
            if (pt.time !== undefined && pt.time !== null && pt.time !== 0) {{
                logical = timestampToLogical(id, pt.time);
            }} else if (pt.logical !== undefined && pt.sourceQuad === id) {{
                logical = pt.logical;
            }}

            let x = 0;
            let xResolved = false;
            const cx = quadLogicalToCoordinate(q, logical);
            if (cx !== null && !isNaN(cx)) {{
                x = cx;
                xResolved = true;
            }}

            if (!xResolved) {{
                try {{
                    const range = q.mainChart.timeScale().getVisibleLogicalRange();
                    const w = q.mainContainer ? (q.mainContainer.clientWidth || 300) : 300;
                    if (range && range.from !== null && range.to !== null && range.to !== range.from) {{
                        x = (logical - range.from) / (range.to - range.from) * w;
                        xResolved = true;
                    }}
                }} catch(e) {{}}
            }}
            if (!xResolved && pt.origX !== undefined && pt.sourceQuad === id) {{
                x = pt.origX;
            }}

            let y = 0;
            let yResolved = false;
            try {{
                if (s && pt.price !== undefined && pt.price !== null) {{
                    const cy = s.priceToCoordinate(pt.price);
                    if (cy !== null && !isNaN(cy)) {{
                        y = cy;
                        yResolved = true;
                    }} else {{
                        const h = q.mainContainer ? (q.mainContainer.clientHeight || 200) : 200;
                        const p0 = s.coordinateToPrice(0);
                        const ph = s.coordinateToPrice(h);
                        if (p0 !== null && ph !== null && p0 !== ph) {{
                            y = (p0 - pt.price) / (p0 - ph) * h;
                            yResolved = true;
                        }}
                    }}
                }}
            }} catch(e) {{}}
            if (!yResolved && pt.origY !== undefined && pt.sourceQuad === id) {{
                y = pt.origY;
            }}

            return {{ x, y }};
        }}

        function getMagnetOrChartPoint(id, px, py) {{
            const normalPt = quadScreenToChart(id, px, py);
            if (!isMagnetEnabled) return normalPt;
            const q = quadsRegistry[id];
            if (!q || !q.payload || !q.payload.candles || q.payload.candles.length === 0) return normalPt;
            const s = getActivePriceSeries(q);
            if (!s) return normalPt;

            let logical = q.mainChart.timeScale().coordinateToLogical(px);
            if (logical === null || isNaN(logical)) return normalPt;
            let idx = Math.round(logical);
            if (idx < 0) idx = 0;
            if (idx >= q.payload.candles.length) idx = q.payload.candles.length - 1;

            const c = q.payload.candles[idx];
            if (!c) return normalPt;

            const cx = q.mainChart.timeScale().logicalToCoordinate(idx);
            if (cx === null || isNaN(cx)) return normalPt;

            const prices = [c.open, c.high, c.low, c.close];
            let bestPrice = null;
            let bestDist = Infinity;
            let bestY = null;

            for (const p of prices) {{
                const pyCoord = s.priceToCoordinate(p);
                if (pyCoord !== null && !isNaN(pyCoord)) {{
                    const d = Math.hypot(px - cx, py - pyCoord);
                    if (d < bestDist) {{
                        bestDist = d;
                        bestPrice = p;
                        bestY = pyCoord;
                    }}
                }}
            }}

            if (bestDist <= 35 && bestPrice !== null) {{
                lastSnappedPoint = {{ id, x: cx, y: bestY }};
                return {{
                    time: logicalToTimestamp(id, idx),
                    price: bestPrice,
                    logical: idx,
                    origX: cx,
                    origY: bestY,
                    sourceQuad: id,
                    isSnapped: true
                }};
            }}
            lastSnappedPoint = null;
            return normalPt;
        }}

        function distToSegment(px, py, x1, y1, x2, y2) {{
            const l2 = Math.hypot(x2 - x1, y2 - y1);
            if (l2 === 0) return Math.hypot(px - x1, py - y1);
            let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (l2 * l2);
            t = Math.max(0, Math.min(1, t));
            return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
        }}

        function findDrawingAtPoint(id, px, py) {{
            const q = quadsRegistry[id];
            if (!q || !q.mainChart) return null;
            const s = getActivePriceSeries(q);
            const w = q.mainContainer.clientWidth;

            for (let i = globalQuadDrawings.length - 1; i >= 0; i--) {{
                const item = globalQuadDrawings[i];
                if (!item.isSync && item.sourceQuad !== id) continue;

                if (item.type === 'horizontal') {{
                    const pt = quadChartToScreen(id, {{ price: item.price, time: 0 }});
                    if (Math.abs(py - pt.y) <= 8) return item;
                }} else if (item.type === 'vertical') {{
                    const pt = quadChartToScreen(id, {{ time: item.time, price: 0 }});
                    if (Math.abs(px - pt.x) <= 8) return item;
                }} else if (item.type === 'cross') {{
                    if (item.p1) {{
                        const s1 = quadChartToScreen(id, item.p1);
                        if (Math.abs(px - s1.x) <= 8 || Math.abs(py - s1.y) <= 8) return item;
                    }}
                }} else if (['trendline', 'arrow', 'ray', 'extended'].includes(item.type)) {{
                    const s1 = quadChartToScreen(id, item.p1);
                    const s2 = quadChartToScreen(id, item.p2);
                    if (distToSegment(px, py, s1.x, s1.y, s2.x, s2.y) <= 8) return item;
                    if (item.type === 'ray') {{
                        const dx = s2.x - s1.x, dy = s2.y - s1.y;
                        const endY = dx !== 0 ? s1.y + (dy / dx) * (w - s1.x) : s2.y;
                        if (distToSegment(px, py, s1.x, s1.y, w, endY) <= 8) return item;
                    }} else if (item.type === 'extended') {{
                        const dx = s2.x - s1.x, dy = s2.y - s1.y;
                        const slope = dx !== 0 ? (dy / dx) : 0;
                        const y0 = s1.y - slope * s1.x;
                        const y1 = s1.y + slope * (w - s1.x);
                        if (distToSegment(px, py, 0, y0, w, y1) <= 8) return item;
                    }}
                }} else if (['rectangle', 'measure', 'risk_reward'].includes(item.type)) {{
                    const s1 = quadChartToScreen(id, item.p1);
                    const s2 = quadChartToScreen(id, item.p2);
                    const rx = Math.min(s1.x, s2.x), ry = Math.min(s1.y, s2.y);
                    const rw = Math.abs(s2.x - s1.x), rh = Math.abs(s2.y - s1.y);
                    if (px >= rx - 5 && px <= rx + rw + 5 && py >= ry - 5 && py <= ry + rh + 5) return item;
                }} else if (item.type === 'circle') {{
                    const s1 = quadChartToScreen(id, item.p1);
                    const s2 = quadChartToScreen(id, item.p2);
                    const cx = (s1.x + s2.x) / 2, cy = (s1.y + s2.y) / 2;
                    const rx = Math.max(3, Math.abs(s2.x - s1.x) / 2);
                    const ry = Math.max(3, Math.abs(s2.y - s1.y) / 2);
                    const d = Math.pow((px - cx) / rx, 2) + Math.pow((py - cy) / ry, 2);
                    if (Math.abs(d - 1) <= 0.4 || d <= 1) return item;
                }} else if (item.type === 'channel') {{
                    const s1 = quadChartToScreen(id, item.p1);
                    const s2 = quadChartToScreen(id, item.p2);
                    const offY = item.channelOffsetY || 30;
                    if (distToSegment(px, py, s1.x, s1.y, s2.x, s2.y) <= 8 || distToSegment(px, py, s1.x, s1.y + offY, s2.x, s2.y + offY) <= 8) return item;
                }} else if (['brush', 'highlighter'].includes(item.type)) {{
                    if (item.points) {{
                        for (let j = 0; j < item.points.length - 1; j++) {{
                            const sp1 = quadChartToScreen(id, item.points[j]);
                            const sp2 = quadChartToScreen(id, item.points[j + 1]);
                            if (distToSegment(px, py, sp1.x, sp1.y, sp2.x, sp2.y) <= (item.type === 'highlighter' ? 12 : 8)) return item;
                        }}
                    }}
                }} else if (['text', 'callout'].includes(item.type)) {{
                    const s1 = quadChartToScreen(id, item.p1);
                    if (Math.hypot(px - s1.x, py - s1.y) <= 30) return item;
                }}
            }}
            return null;
        }}

        function findHandleAtPoint(id, item, px, py) {{
            if (!item) return null;
            if (item.p1) {{
                const s1 = quadChartToScreen(id, item.p1);
                if (Math.hypot(px - s1.x, py - s1.y) <= 9) return 'p1';
            }}
            if (item.p2) {{
                const s2 = quadChartToScreen(id, item.p2);
                if (Math.hypot(px - s2.x, py - s2.y) <= 9) return 'p2';
            }}
            if (item.p3) {{
                const s3 = quadChartToScreen(id, item.p3);
                if (Math.hypot(px - s3.x, py - s3.y) <= 9) return 'p3';
            }}
            return null;
        }}

        function resizeQuadDrawLayer(id) {{
            const q = quadsRegistry[id];
            const canvas = document.getElementById('draw_canvas_' + id);
            if (!q || !canvas || !q.mainContainer) return;
            const dpr = window.devicePixelRatio || 1;
            const w = q.mainContainer.clientWidth;
            const h = q.mainContainer.clientHeight;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            canvas.style.width = w + "px";
            canvas.style.height = h + "px";
            const ctx = canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            renderQuadDrawings(id);
        }}

        function hexToRgbaQuad(hex, alpha) {{
            let c = (hex || '#00E5FF').replace('#', '');
            if (c.length === 3) c = c.split('').map(x => x + x).join('');
            const num = parseInt(c, 16);
            return `rgba(${{(num >> 16) & 255}}, ${{(num >> 8) & 255}}, ${{num & 255}}, ${{alpha}})`;
        }}

        function drawQuadRoundedRect(ctx, x, y, w, h, r) {{
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.arcTo(x + w, y, x + w, y + h, r);
            ctx.arcTo(x + w, y + h, x, y + h, r);
            ctx.arcTo(x, y + h, x, y, r);
            ctx.arcTo(x, y, x + w, y, r);
            ctx.closePath();
        }}

        function drawQuadPriceBadge(ctx, price, y, color, w) {{
            if (y < 0 || price === undefined || price === null) return;
            const priceStr = "₹" + Number(price).toFixed(2);
            ctx.save();
            ctx.font = "bold 9px 'SF Mono', Consolas, Monaco, monospace";
            const textW = ctx.measureText(priceStr).width;
            const badgeW = textW + 10;
            const badgeH = 16;
            const badgeX = w - badgeW - 55;
            const badgeY = y - badgeH / 2;

            ctx.fillStyle = color;
            drawQuadRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
            ctx.fill();

            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(priceStr, badgeX + badgeW / 2, badgeY + badgeH / 2);
            ctx.restore();
        }}

        function drawQuadTimeBadge(ctx, timeVal, x, color, h) {{
            let tStr = '';
            if (typeof timeVal === 'number') {{
                const d = new Date(timeVal * 1000);
                tStr = d.toLocaleDateString('en-IN', {{ timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric' }});
            }} else {{
                tStr = String(timeVal);
            }}
            ctx.save();
            ctx.font = "bold 8.5px 'SF Mono', Consolas, monospace";
            const textW = ctx.measureText(tStr).width;
            const badgeW = textW + 10;
            const badgeH = 15;
            const badgeX = Math.max(5, x - badgeW / 2);
            const badgeY = h - badgeH - 4;
            ctx.fillStyle = color;
            drawQuadRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
            ctx.fill();
            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(tStr, badgeX + badgeW / 2, badgeY + badgeH / 2);
            ctx.restore();
        }}

        function drawSelectionHandles(ctx, id, item) {{
            ctx.save();
            ctx.fillStyle = "#ffffff";
            ctx.strokeStyle = "#2962FF";
            ctx.lineWidth = 2;
            const pts = [];
            if (item.p1) pts.push(quadChartToScreen(id, item.p1));
            if (item.p2) pts.push(quadChartToScreen(id, item.p2));
            if (item.p3) pts.push(quadChartToScreen(id, item.p3));
            for (const p of pts) {{
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }}
            ctx.restore();
        }}

        function drawQuadSingleItem(ctx, id, item, w, h) {{
            ctx.save();
            const color = item.color || '#00E5FF';
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = item.width || 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            if (item.type === 'trendline') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(s2.x, s2.y);
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(s1.x, s1.y, 3, 0, Math.PI * 2);
                ctx.arc(s2.x, s2.y, 3, 0, Math.PI * 2);
                ctx.fill();
            }} else if (item.type === 'ray') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const dx = s2.x - s1.x;
                const dy = s2.y - s1.y;
                let endY = s2.y;
                if (dx !== 0) {{
                    endY = s1.y + (dy / dx) * (w - s1.x);
                }}
                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(w, endY);
                ctx.stroke();
                drawQuadPriceBadge(ctx, item.p2 ? item.p2.price : item.p1.price, endY, color, w);
            }} else if (item.type === 'extended') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const dx = s2.x - s1.x;
                const dy = s2.y - s1.y;
                let y0 = s1.y, y1 = s2.y;
                if (dx !== 0) {{
                    const slope = dy / dx;
                    y0 = s1.y - slope * s1.x;
                    y1 = s1.y + slope * (w - s1.x);
                }}
                ctx.beginPath();
                ctx.moveTo(0, y0);
                ctx.lineTo(w, y1);
                ctx.stroke();
            }} else if (item.type === 'arrow') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(s2.x, s2.y);
                ctx.stroke();

                const angle = Math.atan2(s2.y - s1.y, s2.x - s1.x);
                const aLen = 11;
                ctx.beginPath();
                ctx.moveTo(s2.x, s2.y);
                ctx.lineTo(s2.x - aLen * Math.cos(angle - Math.PI / 6), s2.y - aLen * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(s2.x - aLen * Math.cos(angle + Math.PI / 6), s2.y - aLen * Math.sin(angle + Math.PI / 6));
                ctx.closePath();
                ctx.fill();
            }} else if (item.type === 'channel') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                let offY = item.channelOffsetY || 30;
                if (item.channelPriceDelta !== undefined && item.channelPriceDelta !== null) {{
                    const sOff = quadChartToScreen(id, {{ price: (item.p1 ? item.p1.price : 0) + item.channelPriceDelta, time: item.p1 ? item.p1.time : 0 }});
                    offY = sOff.y - s1.y;
                }}
                ctx.fillStyle = hexToRgbaQuad(color, 0.12);
                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(s2.x, s2.y);
                ctx.lineTo(s2.x, s2.y + offY);
                ctx.lineTo(s1.x, s1.y + offY);
                ctx.closePath();
                ctx.fill();

                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(s2.x, s2.y);
                ctx.moveTo(s1.x, s1.y + offY);
                ctx.lineTo(s2.x, s2.y + offY);
                ctx.stroke();

                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y + offY / 2);
                ctx.lineTo(s2.x, s2.y + offY / 2);
                ctx.stroke();
                ctx.setLineDash([]);
            }} else if (item.type === 'horizontal') {{
                const pt = quadChartToScreen(id, {{ price: item.price, time: 0 }});
                const y = pt.y;
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
                ctx.setLineDash([]);
                drawQuadPriceBadge(ctx, item.price, y, color, w);
            }} else if (item.type === 'vertical') {{
                const pt = quadChartToScreen(id, {{ time: item.time, price: 0 }});
                const x = pt.x;
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
                ctx.setLineDash([]);
                drawQuadTimeBadge(ctx, item.time, x, color, h);
            }} else if (item.type === 'cross') {{
                const s1 = quadChartToScreen(id, item.p1);
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(0, s1.y);
                ctx.lineTo(w, s1.y);
                ctx.moveTo(s1.x, 0);
                ctx.lineTo(s1.x, h);
                ctx.stroke();
                ctx.setLineDash([]);
                drawQuadPriceBadge(ctx, item.p1 ? item.p1.price : 0, s1.y, color, w);
                drawQuadTimeBadge(ctx, item.p1 ? item.p1.time : 0, s1.x, color, h);
            }} else if (item.type === 'rectangle') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const rx = Math.min(s1.x, s2.x), ry = Math.min(s1.y, s2.y);
                const rw = Math.abs(s2.x - s1.x), rh = Math.abs(s2.y - s1.y);

                ctx.fillStyle = hexToRgbaQuad(color, 0.16);
                ctx.fillRect(rx, ry, rw, rh);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(rx, ry, rw, rh);

                ctx.font = "bold 8px sans-serif";
                ctx.fillStyle = color;
                ctx.fillText("ZONE", rx + 4, ry + 10);
            }} else if (item.type === 'circle') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const cx = (s1.x + s2.x) / 2, cy = (s1.y + s2.y) / 2;
                const rx = Math.max(3, Math.abs(s2.x - s1.x) / 2);
                const ry = Math.max(3, Math.abs(s2.y - s1.y) / 2);

                ctx.fillStyle = hexToRgbaQuad(color, 0.15);
                ctx.beginPath();
                ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }} else if (item.type === 'fib') {{
                const s = getActivePriceSeries(quadsRegistry[id]);
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const pDiff = item.p2.price - item.p1.price;
                const minX = Math.max(0, Math.min(s1.x, s2.x));
                const maxX = Math.max(w - 55, Math.max(s1.x, s2.x) + 30);
                const fibLevels = [
                    {{ r: 0.0, label: "0% (" }},
                    {{ r: 0.236, label: "23.6% (" }},
                    {{ r: 0.382, label: "38.2% (" }},
                    {{ r: 0.500, label: "50% (" }},
                    {{ r: 0.618, label: "61.8% (" }},
                    {{ r: 0.786, label: "78.6% (" }},
                    {{ r: 1.0, label: "100% (" }}
                ];

                const p382 = item.p1.price + 0.382 * pDiff;
                const p618 = item.p1.price + 0.618 * pDiff;
                let y382 = s1.y, y618 = s2.y;
                if (s) {{
                    try {{
                        const c1 = s.priceToCoordinate(p382);
                        const c2 = s.priceToCoordinate(p618);
                        if (c1 !== null && !isNaN(c1)) y382 = c1;
                        if (c2 !== null && !isNaN(c2)) y618 = c2;
                    }} catch(e) {{}}
                }}
                const gyMin = Math.min(y382, y618);
                const gyH = Math.abs(y618 - y382);
                ctx.fillStyle = "rgba(255, 215, 0, 0.12)";
                ctx.fillRect(minX, gyMin, maxX - minX, gyH);

                for (const fib of fibLevels) {{
                    const lvlPrice = item.p1.price + fib.r * pDiff;
                    let ly = s1.y;
                    if (s) {{
                        try {{
                            const c = s.priceToCoordinate(lvlPrice);
                            if (c !== null && !isNaN(c)) ly = c;
                        }} catch(e) {{}}
                    }}

                    ctx.strokeStyle = fib.r === 0.618 || fib.r === 0.5 ? '#FACC15' : color;
                    ctx.lineWidth = fib.r === 0.618 || fib.r === 0.5 ? 1.5 : 1;
                    ctx.beginPath();
                    ctx.moveTo(minX, ly);
                    ctx.lineTo(maxX, ly);
                    ctx.stroke();

                    ctx.font = "bold 8px 'SF Mono', Consolas, Monaco, monospace";
                    ctx.fillStyle = fib.r === 0.618 || fib.r === 0.5 ? '#FACC15' : color;
                    ctx.fillText(fib.label + lvlPrice.toFixed(1) + ")", minX + 4, ly - 2);
                }}
            }} else if (item.type === 'measure') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const minX = Math.min(s1.x, s2.x), minY = Math.min(s1.y, s2.y);
                const rw = Math.abs(s2.x - s1.x), rh = Math.abs(s2.y - s1.y);
                const isGain = (item.p2.price >= item.p1.price);
                const boxColor = isGain ? '#089981' : '#F23645';
                ctx.fillStyle = hexToRgbaQuad(boxColor, 0.14);
                ctx.fillRect(minX, minY, rw, rh);
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.strokeRect(minX, minY, rw, rh);
                ctx.setLineDash([]);

                ctx.beginPath();
                ctx.moveTo(s1.x, s1.y);
                ctx.lineTo(s2.x, s2.y);
                ctx.stroke();

                const pDiff = item.p2.price - item.p1.price;
                const pct = item.p1.price ? (pDiff / item.p1.price) * 100 : 0;
                const log1 = timestampToLogical(id, item.p1.time);
                const log2 = timestampToLogical(id, item.p2.time);
                const bars = Math.abs(Math.round(log2 - log1));
                const sign = pDiff >= 0 ? '+' : '';
                const lblText = `${{sign}}₹${{pDiff.toFixed(2)}} (${{sign}}${{pct.toFixed(2)}}%) | ${{bars}} bars`;

                ctx.font = "bold 9px 'SF Mono', Consolas, monospace";
                const tw = ctx.measureText(lblText).width;
                const bx = minX + rw / 2 - (tw + 12) / 2;
                const by = minY + rh / 2 - 9;
                ctx.fillStyle = boxColor;
                drawQuadRoundedRect(ctx, bx, by, tw + 12, 18, 4);
                ctx.fill();
                ctx.fillStyle = "#ffffff";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(lblText, bx + (tw + 12) / 2, by + 9);
            }} else if (item.type === 'risk_reward') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const isLong = (item.p2.price >= item.p1.price);
                const riskDiff = Math.abs(item.p2.price - item.p1.price) / 2;
                const stopPrice = isLong ? item.p1.price - riskDiff : item.p1.price + riskDiff;
                const stopPt = quadChartToScreen(id, {{ price: stopPrice, time: item.p1.time }});

                const rx = Math.min(s1.x, s2.x);
                const rw = Math.max(90, Math.abs(s2.x - s1.x));

                const tTop = Math.min(s1.y, s2.y);
                const tH = Math.abs(s2.y - s1.y);
                ctx.fillStyle = "rgba(8, 153, 129, 0.22)";
                ctx.fillRect(rx, tTop, rw, tH);
                ctx.strokeStyle = "#089981";
                ctx.lineWidth = 1;
                ctx.strokeRect(rx, tTop, rw, tH);

                const sTop = Math.min(s1.y, stopPt.y);
                const sH = Math.abs(stopPt.y - s1.y);
                ctx.fillStyle = "rgba(242, 54, 69, 0.22)";
                ctx.fillRect(rx, sTop, rw, sH);
                ctx.strokeStyle = "#F23645";
                ctx.strokeRect(rx, sTop, rw, sH);

                ctx.strokeStyle = "#FFFFFF";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(rx, s1.y);
                ctx.lineTo(rx + rw, s1.y);
                ctx.stroke();

                const rrText = `R:R 2.00 | Target: ₹${{item.p2.price.toFixed(1)}} | Stop: ₹${{stopPrice.toFixed(1)}}`;
                ctx.font = "bold 8.5px 'SF Mono', Consolas, monospace";
                const tw = ctx.measureText(rrText).width;
                const bx = rx + rw / 2 - (tw + 10) / 2;
                const by = s1.y - 8;
                ctx.fillStyle = "rgba(20, 24, 35, 0.9)";
                drawQuadRoundedRect(ctx, bx, by, tw + 10, 16, 3);
                ctx.fill();
                ctx.strokeStyle = "#2962FF";
                ctx.stroke();
                ctx.fillStyle = "#FFFFFF";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(rrText, bx + (tw + 10) / 2, by + 8);
            }} else if (['brush', 'highlighter'].includes(item.type)) {{
                if (!item.points || item.points.length < 2) return;
                const isHighlighter = (item.type === 'highlighter');
                ctx.lineWidth = isHighlighter ? 14 : (item.width || 2);
                ctx.strokeStyle = isHighlighter ? hexToRgbaQuad(color, 0.28) : color;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                const screenPts = item.points.map(pt => quadChartToScreen(id, pt));
                ctx.beginPath();
                ctx.moveTo(screenPts[0].x, screenPts[0].y);

                if (screenPts.length === 2) {{
                    ctx.lineTo(screenPts[1].x, screenPts[1].y);
                }} else {{
                    for (let i = 1; i < screenPts.length - 1; i++) {{
                        const xc = (screenPts[i].x + screenPts[i + 1].x) / 2;
                        const yc = (screenPts[i].y + screenPts[i + 1].y) / 2;
                        ctx.quadraticCurveTo(screenPts[i].x, screenPts[i].y, xc, yc);
                    }}
                    ctx.lineTo(screenPts[screenPts.length - 1].x, screenPts[screenPts.length - 1].y);
                }}
                ctx.stroke();
            }} else if (item.type === 'text') {{
                const s1 = quadChartToScreen(id, item.p1);
                ctx.font = "bold 10px sans-serif";
                const tw = ctx.measureText(item.text || "Note").width;
                const th = 18;
                const bx = s1.x + 4;
                const by = s1.y - th / 2;

                ctx.fillStyle = (QUAD_DATA.theme === 'light') ? "rgba(255,255,255,0.92)" : "rgba(30,34,45,0.92)";
                drawQuadRoundedRect(ctx, bx, by, tw + 12, th, 4);
                ctx.fill();

                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = color;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(item.text || "Note", bx + 6, s1.y);
            }} else if (item.type === 'callout') {{
                const s1 = quadChartToScreen(id, item.p1);
                const s2 = quadChartToScreen(id, item.p2);
                const text = item.text || "Note";
                ctx.font = "bold 9.5px sans-serif";
                const tw = ctx.measureText(text).width;
                const bw = tw + 16, bh = 22;
                const bx = s2.x - bw / 2, by = s2.y - bh / 2;

                ctx.beginPath();
                ctx.moveTo(s2.x, s2.y);
                ctx.lineTo(s1.x, s1.y);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = (QUAD_DATA.theme === 'light') ? "#FFFFFF" : "#1e222d";
                drawQuadRoundedRect(ctx, bx, by, bw, bh, 5);
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.stroke();
                ctx.fillStyle = (QUAD_DATA.theme === 'light') ? "#0F172A" : "#FFFFFF";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(text, bx + bw / 2, by + bh / 2);
            }}

            if (selectedDrawing && selectedDrawing.id === item.id) {{
                drawSelectionHandles(ctx, id, item);
            }}

            ctx.restore();
        }}

        function renderQuadDrawings(id) {{
            const q = quadsRegistry[id];
            const canvas = document.getElementById('draw_canvas_' + id);
            if (!q || !canvas || !q.mainContainer) return;
            const ctx = canvas.getContext('2d');
            const w = q.mainContainer.clientWidth;
            const h = q.mainContainer.clientHeight;
            ctx.clearRect(0, 0, w, h);

            for (const d of globalQuadDrawings) {{
                if (d.isSync || d.sourceQuad === id) {{
                    drawQuadSingleItem(ctx, id, d, w, h);
                }}
            }}

            if (activeDrawingQuad === id && isQuadDrawing && quadStartPoint && quadCurrentPoint) {{
                const previewItem = {{
                    type: activeQuadTool,
                    p1: quadStartPoint,
                    p2: quadCurrentPoint,
                    color: activeQuadColor,
                    width: activeQuadWidth,
                    isPreview: true
                }};
                drawQuadSingleItem(ctx, id, previewItem, w, h);
            }} else if (activeDrawingQuad === id && isQuadDrawing && ['brush', 'highlighter'].includes(activeQuadTool) && quadBrushPoints.length > 1) {{
                const previewItem = {{
                    type: activeQuadTool,
                    points: quadBrushPoints,
                    color: activeQuadColor,
                    width: activeQuadWidth,
                    isPreview: true
                }};
                drawQuadSingleItem(ctx, id, previewItem, w, h);
            }}

            if (isMagnetEnabled && lastSnappedPoint && lastSnappedPoint.id === id) {{
                ctx.save();
                ctx.strokeStyle = "#00E5FF";
                ctx.fillStyle = "rgba(0, 229, 255, 0.4)";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(lastSnappedPoint.x, lastSnappedPoint.y, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                ctx.restore();
            }}
        }}

        function renderAllQuadDrawings() {{
            ['m', 'w', 'd', '75'].forEach(k => renderQuadDrawings(k));
        }}

        function setQuadTool(toolName) {{
            activeQuadTool = toolName;
            if (activeQuadTool !== 'pointer') {{
                selectedDrawing = null;
            }}
            const toolBtns = quadToolbar ? quadToolbar.querySelectorAll(".tv-tool-btn[data-tool]") : [];
            toolBtns.forEach(btn => {{
                btn.classList.toggle("active", btn.getAttribute("data-tool") === toolName);
            }});

            ['m', 'w', 'd', '75'].forEach(k => {{
                const canvas = document.getElementById('draw_canvas_' + k);
                if (canvas) {{
                    if (activeQuadTool === 'pointer') {{
                        canvas.style.pointerEvents = 'auto';
                        canvas.style.cursor = 'default';
                    }} else {{
                        canvas.style.pointerEvents = 'auto';
                        canvas.style.cursor = 'crosshair';
                    }}
                }}
            }});
            renderAllQuadDrawings();
        }}

        if (quadToolbar) {{
            quadToolbar.addEventListener("click", (e) => {{
                const btn = e.target.closest(".tv-tool-btn[data-tool]");
                if (btn) {{
                    setQuadTool(btn.getAttribute("data-tool"));
                }}
            }});
        }}

        if (magnetBtnQuad) {{
            magnetBtnQuad.addEventListener("click", () => {{
                isMagnetEnabled = !isMagnetEnabled;
                magnetBtnQuad.classList.toggle("magnet-active", isMagnetEnabled);
                magnetBtnQuad.title = isMagnetEnabled ? "Magnet Mode: ON (Snap to Candle OHLC) [M]" : "Magnet Mode: OFF (Snap to Candle OHLC) [M]";
                renderAllQuadDrawings();
            }});
        }}

        if (syncBtnQuad) {{
            syncBtnQuad.addEventListener("click", () => {{
                isSyncEnabled = !isSyncEnabled;
                syncBtnQuad.classList.toggle("active", isSyncEnabled);
                syncBtnQuad.title = isSyncEnabled ? "Sync drawings across all 4 quadrants (ON)" : "Sync drawings across all 4 quadrants (OFF)";
                const statusBadge = document.getElementById("sync_status_badge");
                if (statusBadge) {{
                    statusBadge.innerHTML = isSyncEnabled ? "🔄 Sync: 4 Charts Active" : "🔒 Sync: Disabled";
                    statusBadge.style.color = isSyncEnabled ? "#00E5FF" : "#94A3B8";
                }}
            }});
        }}

        if (colorInputQuad) {{
            colorInputQuad.addEventListener("input", (e) => {{
                activeQuadColor = e.target.value;
                if (colorDotQuad) colorDotQuad.style.backgroundColor = activeQuadColor;
                if (selectedDrawing) {{
                    selectedDrawing.color = activeQuadColor;
                    renderAllQuadDrawings();
                }}
            }});
        }}

        if (widthInputQuad) {{
            widthInputQuad.addEventListener("change", (e) => {{
                activeQuadWidth = parseInt(e.target.value, 10) || 2;
                if (selectedDrawing) {{
                    selectedDrawing.width = activeQuadWidth;
                    renderAllQuadDrawings();
                }}
            }});
        }}

        if (undoBtnQuad) {{
            undoBtnQuad.addEventListener("click", () => {{
                if (globalQuadDrawings.length > 0) {{
                    globalQuadDrawings.pop();
                    selectedDrawing = null;
                    renderAllQuadDrawings();
                }}
            }});
        }}

        if (deleteSelBtnQuad) {{
            deleteSelBtnQuad.addEventListener("click", () => {{
                if (selectedDrawing) {{
                    const idx = globalQuadDrawings.findIndex(d => d.id === selectedDrawing.id);
                    if (idx !== -1) {{
                        globalQuadDrawings.splice(idx, 1);
                        selectedDrawing = null;
                        renderAllQuadDrawings();
                    }}
                }}
            }});
        }}

        if (clearBtnQuad) {{
            clearBtnQuad.addEventListener("click", () => {{
                if (globalQuadDrawings.length > 0 && confirm("Clear all drawings across all 4 quadrants?")) {{
                    globalQuadDrawings.length = 0;
                    selectedDrawing = null;
                    renderAllQuadDrawings();
                }}
            }});
        }}

        ['m', 'w', 'd', '75'].forEach(id => {{
            const canvas = document.getElementById('draw_canvas_' + id);
            if (!canvas) return;

            canvas.addEventListener("mousedown", (e) => {{
                const rect = canvas.getBoundingClientRect();
                const px = e.clientX - rect.left;
                const py = e.clientY - rect.top;

                if (activeQuadTool === 'pointer') {{
                    if (selectedDrawing) {{
                        const handle = findHandleAtPoint(id, selectedDrawing, px, py);
                        if (handle) {{
                            isDraggingHandle = handle;
                            activeDrawingQuad = id;
                            dragStartScreen = {{ px, py }};
                            return;
                        }}
                    }}

                    const clicked = findDrawingAtPoint(id, px, py);
                    if (clicked) {{
                        selectedDrawing = clicked;
                        isDraggingDrawing = true;
                        activeDrawingQuad = id;
                        dragStartScreen = {{ px, py }};
                        dragInitialPoints = {{
                            s1: clicked.p1 ? quadChartToScreen(id, clicked.p1) : null,
                            s2: clicked.p2 ? quadChartToScreen(id, clicked.p2) : null,
                            s3: clicked.p3 ? quadChartToScreen(id, clicked.p3) : null,
                            horizY: clicked.type === 'horizontal' ? quadChartToScreen(id, {{ price: clicked.price, time: 0 }}).y : py,
                            vertX: clicked.type === 'vertical' ? quadChartToScreen(id, {{ time: clicked.time, price: 0 }}).x : px,
                            points: clicked.points ? clicked.points.map(p => quadChartToScreen(id, p)) : null
                        }};
                    }} else {{
                        selectedDrawing = null;
                    }}
                    renderAllQuadDrawings();
                    return;
                }}

                activeDrawingQuad = id;
                const pt = getMagnetOrChartPoint(id, px, py);

                if (activeQuadTool === 'horizontal') {{
                    globalQuadDrawings.push({{
                        id: Date.now(),
                        type: 'horizontal',
                        price: pt.price,
                        origY: py,
                        color: activeQuadColor,
                        width: activeQuadWidth,
                        sourceQuad: id,
                        isSync: isSyncEnabled
                    }});
                    renderAllQuadDrawings();
                    return;
                }}

                if (activeQuadTool === 'vertical') {{
                    globalQuadDrawings.push({{
                        id: Date.now(),
                        type: 'vertical',
                        time: pt.time,
                        origX: px,
                        color: activeQuadColor,
                        width: activeQuadWidth,
                        sourceQuad: id,
                        isSync: isSyncEnabled
                    }});
                    renderAllQuadDrawings();
                    return;
                }}

                if (activeQuadTool === 'cross') {{
                    globalQuadDrawings.push({{
                        id: Date.now(),
                        type: 'cross',
                        p1: pt,
                        color: activeQuadColor,
                        width: activeQuadWidth,
                        sourceQuad: id,
                        isSync: isSyncEnabled
                    }});
                    renderAllQuadDrawings();
                    return;
                }}

                if (activeQuadTool === 'text') {{
                    const textVal = prompt("Enter chart annotation:", "Key Level");
                    if (textVal && textVal.trim()) {{
                        globalQuadDrawings.push({{
                            id: Date.now(),
                            type: 'text',
                            p1: pt,
                            text: textVal.trim(),
                            color: activeQuadColor,
                            width: activeQuadWidth,
                            sourceQuad: id,
                            isSync: isSyncEnabled
                        }});
                        renderAllQuadDrawings();
                    }}
                    return;
                }}

                if (['brush', 'highlighter'].includes(activeQuadTool)) {{
                    isQuadDrawing = true;
                    quadBrushPoints = [pt];
                    renderQuadDrawings(id);
                    return;
                }}

                isQuadDrawing = true;
                quadStartPoint = pt;
                quadCurrentPoint = pt;
                renderQuadDrawings(id);
            }});

            canvas.addEventListener("mousemove", (e) => {{
                const rect = canvas.getBoundingClientRect();
                const px = e.clientX - rect.left;
                const py = e.clientY - rect.top;

                if (activeQuadTool === 'pointer') {{
                    if (isDraggingHandle && selectedDrawing) {{
                        const newPt = getMagnetOrChartPoint(id, px, py);
                        selectedDrawing[isDraggingHandle] = newPt;
                        renderAllQuadDrawings();
                        return;
                    }}
                    if (isDraggingDrawing && selectedDrawing && dragStartScreen && dragInitialPoints) {{
                        const dx = px - dragStartScreen.px;
                        const dy = py - dragStartScreen.py;
                        if (selectedDrawing.p1 && dragInitialPoints.s1) {{
                            selectedDrawing.p1 = quadScreenToChart(id, dragInitialPoints.s1.x + dx, dragInitialPoints.s1.y + dy);
                        }}
                        if (selectedDrawing.p2 && dragInitialPoints.s2) {{
                            selectedDrawing.p2 = quadScreenToChart(id, dragInitialPoints.s2.x + dx, dragInitialPoints.s2.y + dy);
                        }}
                        if (selectedDrawing.p3 && dragInitialPoints.s3) {{
                            selectedDrawing.p3 = quadScreenToChart(id, dragInitialPoints.s3.x + dx, dragInitialPoints.s3.y + dy);
                        }}
                        if (selectedDrawing.type === 'horizontal') {{
                            const qPt = quadScreenToChart(id, px, dragInitialPoints.horizY + dy);
                            selectedDrawing.price = qPt.price;
                        }}
                        if (selectedDrawing.type === 'vertical') {{
                            const qPt = quadScreenToChart(id, dragInitialPoints.vertX + dx, py);
                            selectedDrawing.time = qPt.time;
                        }}
                        if (selectedDrawing.points && dragInitialPoints.points) {{
                            selectedDrawing.points = dragInitialPoints.points.map(sp => quadScreenToChart(id, sp.x + dx, sp.y + dy));
                        }}
                        renderAllQuadDrawings();
                        return;
                    }}

                    if (selectedDrawing && findHandleAtPoint(id, selectedDrawing, px, py)) {{
                        canvas.style.cursor = 'crosshair';
                    }} else if (findDrawingAtPoint(id, px, py)) {{
                        canvas.style.cursor = 'move';
                    }} else {{
                        canvas.style.cursor = 'default';
                    }}
                    return;
                }}

                if (!isQuadDrawing || activeDrawingQuad !== id) {{
                    if (isMagnetEnabled) {{
                        getMagnetOrChartPoint(id, px, py);
                        renderQuadDrawings(id);
                    }}
                    return;
                }}

                const pt = getMagnetOrChartPoint(id, px, py);
                if (['brush', 'highlighter'].includes(activeQuadTool)) {{
                    quadBrushPoints.push(pt);
                    renderQuadDrawings(id);
                }} else {{
                    quadCurrentPoint = pt;
                    renderQuadDrawings(id);
                }}
            }});
        }});

        window.addEventListener("mouseup", (e) => {{
            if (isDraggingHandle || isDraggingDrawing) {{
                isDraggingHandle = null;
                isDraggingDrawing = false;
                dragStartScreen = null;
                dragInitialPoints = null;
                renderAllQuadDrawings();
            }}

            if (!isQuadDrawing || !activeDrawingQuad) return;
            const id = activeDrawingQuad;
            const canvas = document.getElementById('draw_canvas_' + id);
            if (canvas) {{
                const rect = canvas.getBoundingClientRect();
                const px = e.clientX - rect.left;
                const py = e.clientY - rect.top;
                const endPt = getMagnetOrChartPoint(id, px, py);

                if (['brush', 'highlighter'].includes(activeQuadTool)) {{
                    if (quadBrushPoints.length > 1) {{
                        globalQuadDrawings.push({{
                            id: Date.now(),
                            type: activeQuadTool,
                            points: quadBrushPoints,
                            color: activeQuadColor,
                            width: activeQuadWidth,
                            sourceQuad: id,
                            isSync: isSyncEnabled
                        }});
                    }}
                    quadBrushPoints = [];
                }} else if (activeQuadTool === 'callout') {{
                    if (quadStartPoint) {{
                        const textVal = prompt("Enter callout note:", "Key Pivot") || "Note";
                        globalQuadDrawings.push({{
                            id: Date.now(),
                            type: 'callout',
                            p1: quadStartPoint,
                            p2: endPt,
                            text: textVal.trim(),
                            color: activeQuadColor,
                            width: activeQuadWidth,
                            sourceQuad: id,
                            isSync: isSyncEnabled
                        }});
                    }}
                }} else if (['trendline', 'ray', 'extended', 'arrow', 'channel', 'rectangle', 'circle', 'fib', 'measure', 'risk_reward'].includes(activeQuadTool)) {{
                    if (quadStartPoint) {{
                        const dist = Math.hypot(px - quadStartPoint.origX, py - quadStartPoint.origY);
                        if (dist > 5) {{
                            globalQuadDrawings.push({{
                                id: Date.now(),
                                type: activeQuadTool,
                                p1: quadStartPoint,
                                p2: endPt,
                                channelOffsetY: (activeQuadTool === 'channel') ? 35 : 0,
                                channelPriceDelta: (activeQuadTool === 'channel' && quadStartPoint.price) ? (quadStartPoint.price * 0.015) : 0,
                                color: activeQuadColor,
                                width: activeQuadWidth,
                                sourceQuad: id,
                                isSync: isSyncEnabled
                            }});
                        }}
                    }}
                }}
            }}

            isQuadDrawing = false;
            activeDrawingQuad = null;
            quadStartPoint = null;
            quadCurrentPoint = null;
            lastSnappedPoint = null;
            renderAllQuadDrawings();
        }});

        window.addEventListener("keydown", (e) => {{
            if (e.key === "Escape") {{
                if (isQuadDrawing) {{
                    isQuadDrawing = false;
                    activeDrawingQuad = null;
                    quadStartPoint = null;
                    quadCurrentPoint = null;
                    quadBrushPoints = [];
                    renderAllQuadDrawings();
                }} else {{
                    selectedDrawing = null;
                    setQuadTool('pointer');
                }}
            }}
            if (e.key === 'm' || e.key === 'M') {{
                if (magnetBtnQuad) {{
                    isMagnetEnabled = !isMagnetEnabled;
                    magnetBtnQuad.classList.toggle("magnet-active", isMagnetEnabled);
                    renderAllQuadDrawings();
                }}
            }}
            if (e.key === 'Delete' || e.key === 'Backspace') {{
                if (selectedDrawing) {{
                    const idx = globalQuadDrawings.findIndex(d => d.id === selectedDrawing.id);
                    if (idx !== -1) {{
                        globalQuadDrawings.splice(idx, 1);
                        selectedDrawing = null;
                        renderAllQuadDrawings();
                    }}
                }}
            }}
            if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {{
                if (globalQuadDrawings.length > 0) {{
                    globalQuadDrawings.pop();
                    selectedDrawing = null;
                    renderAllQuadDrawings();
                }}
            }}
        }});

        // Continuous price scale & time scale sync engine
        // Keeps drawings mathematically locked to candles and price coordinates when dragging Y-axis or zooming
        const lastQuadScaleState = {{}};

        function syncQuadDrawingsWithChart() {{
            if (globalQuadDrawings.length === 0 && !isQuadDrawing) {{
                requestAnimationFrame(syncQuadDrawingsWithChart);
                return;
            }}
            Object.keys(quadsRegistry).forEach(k => {{
                const q = quadsRegistry[k];
                if (!q || !q.mainChart) return;
                const s = getActivePriceSeries(q);
                if (!s) return;

                const h = q.mainContainer ? (q.mainContainer.clientHeight || 200) : 200;
                const pTop = s.coordinateToPrice(0);
                const pBot = s.coordinateToPrice(h);

                let rangeStr = '';
                try {{
                    const r = q.mainChart.timeScale().getVisibleLogicalRange();
                    if (r && r.from !== null && r.to !== null) {{
                        rangeStr = r.from.toFixed(2) + ':' + r.to.toFixed(2);
                    }}
                }} catch(e) {{}}

                const prev = lastQuadScaleState[k];
                if (!prev || prev.pTop !== pTop || prev.pBot !== pBot || prev.rangeStr !== rangeStr) {{
                    lastQuadScaleState[k] = {{ pTop, pBot, rangeStr }};
                    renderQuadDrawings(k);
                }}
            }});
            requestAnimationFrame(syncQuadDrawingsWithChart);
        }}
        requestAnimationFrame(syncQuadDrawingsWithChart);

        // Also subscribe explicitly to time scale changes and DOM interaction events
        Object.keys(quadsRegistry).forEach(k => {{
            const q = quadsRegistry[k];
            if (q && q.mainChart) {{
                q.mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                    renderQuadDrawings(k);
                }});
            }}
            const wrap = document.getElementById('wrap_main_' + k);
            if (wrap) {{
                wrap.addEventListener('pointermove', (e) => {{
                    if (e.buttons > 0) renderQuadDrawings(k);
                }});
                wrap.addEventListener('wheel', () => {{
                    renderQuadDrawings(k);
                }}, {{ passive: true }});
            }}
        }});

        function resizeAll() {{
            const root = document.getElementById('quad_root');
            if (!root) return;
            const isFs = document.fullscreenElement === root || root.classList.contains('css-fullscreen');
            const grid = document.getElementById('quad_grid');
            if (!grid) return;
            const isSingleFocus = grid.className.includes('focus-');

            const drawBarW = 38;
            const rootW = (root.clientWidth || window.innerWidth || 1000) - drawBarW;
            const rootH = isFs ? (window.innerHeight || 800) : {height};
            const gridW = rootW - 14;
            const gridH = rootH - 38 - 14;

            const cardW = isSingleFocus ? gridW : Math.max(240, Math.floor((gridW - 8) / 2));
            const cardH = isSingleFocus ? gridH : Math.max(200, Math.floor((gridH - 8) / 2));
            const availBodyH = Math.max(160, cardH - 32);

            Object.keys(quadsRegistry).forEach(key => {{
                const q = quadsRegistry[key];
                if (!q) return;
                const card = document.getElementById('card_' + key);
                if (!card || card.offsetParent === null) return;

                const hasRsi = q.hasRsi;
                const mainH = hasRsi ? Math.max(120, Math.floor(availBodyH * 0.68)) : availBodyH;
                const rsiH = hasRsi ? Math.max(50, availBodyH - mainH) : 0;

                const mainWrap = document.getElementById('wrap_main_' + key);
                if (mainWrap) {{
                    mainWrap.style.height = mainH + 'px';
                }}
                if (q.mainContainer && q.mainChart) {{
                    q.mainContainer.style.width = cardW + 'px';
                    q.mainContainer.style.height = mainH + 'px';
                    if (q.mainChart.applyOptions) {{
                        q.mainChart.applyOptions({{ width: cardW, height: mainH }});
                    }} else if (q.mainChart.resize) {{
                        q.mainChart.resize(cardW, mainH);
                    }}
                }}
                if (hasRsi && q.rsiContainer && q.rsiChart) {{
                    q.rsiContainer.style.width = cardW + 'px';
                    q.rsiContainer.style.height = rsiH + 'px';
                    if (q.rsiChart.applyOptions) {{
                        q.rsiChart.applyOptions({{ width: cardW, height: rsiH }});
                    }} else if (q.rsiChart.resize) {{
                        q.rsiChart.resize(cardW, rsiH);
                    }}
                }}
                resizeQuadDrawLayer(key);
            }});
        }}

        function setQuadMode(mode) {{
            currentQuadMode = mode;
            const grid = document.getElementById('quad_grid');
            grid.className = 'quad-grid';

            ['all', 'm', 'w', 'd', '75'].forEach(k => {{
                const btn = document.getElementById('btn_q_' + k);
                if (btn) btn.classList.toggle('active', k === mode);
            }});

            if (mode !== 'all') {{
                grid.classList.add('focus-' + mode);
            }}

            setTimeout(resizeAll, 30);
            setTimeout(resizeAll, 120);
        }}

        function toggleZoomCard(tf) {{
            const grid = document.getElementById('quad_grid');
            if (grid.classList.contains('focus-' + tf)) {{
                setQuadMode('all');
            }} else {{
                setQuadMode(tf);
            }}
        }}

        let globalVolVisible = QUAD_DATA.showVolume !== false;

        function toggleQuadVolume() {{
            globalVolVisible = !globalVolVisible;
            const vBtn = document.getElementById('btn_q_vol');
            if (vBtn) {{
                vBtn.classList.toggle('active', globalVolVisible);
            }}

            Object.keys(quadsRegistry).forEach(key => {{
                const q = quadsRegistry[key];
                if (q && q.volumeSeries) {{
                    q.isVolVisible = globalVolVisible;
                    q.volumeSeries.applyOptions({{ visible: globalVolVisible }});
                    const cardVolBtn = document.getElementById('btn_vol_' + key);
                    if (cardVolBtn) cardVolBtn.classList.toggle('active', globalVolVisible);
                }}
            }});
        }}

        function toggleCardVolume(key) {{
            const q = quadsRegistry[key];
            if (!q || !q.volumeSeries) return;
            q.isVolVisible = !q.isVolVisible;
            q.volumeSeries.applyOptions({{ visible: q.isVolVisible }});
            const cardVolBtn = document.getElementById('btn_vol_' + key);
            if (cardVolBtn) cardVolBtn.classList.toggle('active', q.isVolVisible);
        }}

        let globalCandlesVisible = QUAD_DATA.showCandles !== false;
        let globalLineVisible = QUAD_DATA.showLine !== false;

        function updateMasterStyleBtns() {{
            const cBtn = document.getElementById('btn_q_candles');
            const lBtn = document.getElementById('btn_q_line');
            if (cBtn) cBtn.classList.toggle('active', globalCandlesVisible);
            if (lBtn) lBtn.classList.toggle('active', globalLineVisible);
        }}

        function toggleMasterCandles() {{
            if (globalCandlesVisible && !globalLineVisible) {{
                globalLineVisible = true;
            }}
            globalCandlesVisible = !globalCandlesVisible;
            updateMasterStyleBtns();

            Object.keys(quadsRegistry).forEach(key => {{
                const q = quadsRegistry[key];
                if (q && q.candleSeries) {{
                    q.isCandleVisible = globalCandlesVisible;
                    q.candleSeries.applyOptions({{ visible: globalCandlesVisible }});
                    const cCardBtn = document.getElementById('btn_candle_' + key);
                    if (cCardBtn) cCardBtn.classList.toggle('active', globalCandlesVisible);
                }}
                if (q && q.lineSeries) {{
                    q.isLineVisible = globalLineVisible;
                    q.lineSeries.applyOptions({{ visible: globalLineVisible }});
                    const lCardBtn = document.getElementById('btn_line_' + key);
                    if (lCardBtn) lCardBtn.classList.toggle('active', globalLineVisible);
                }}
            }});
        }}

        function toggleMasterLine() {{
            if (globalLineVisible && !globalCandlesVisible) {{
                globalCandlesVisible = true;
            }}
            globalLineVisible = !globalLineVisible;
            updateMasterStyleBtns();

            Object.keys(quadsRegistry).forEach(key => {{
                const q = quadsRegistry[key];
                if (q && q.lineSeries) {{
                    q.isLineVisible = globalLineVisible;
                    q.lineSeries.applyOptions({{ visible: globalLineVisible }});
                    const lCardBtn = document.getElementById('btn_line_' + key);
                    if (lCardBtn) lCardBtn.classList.toggle('active', globalLineVisible);
                }}
                if (q && q.candleSeries) {{
                    q.isCandleVisible = globalCandlesVisible;
                    q.candleSeries.applyOptions({{ visible: globalCandlesVisible }});
                    const cCardBtn = document.getElementById('btn_candle_' + key);
                    if (cCardBtn) cCardBtn.classList.toggle('active', globalCandlesVisible);
                }}
            }});
        }}

        function toggleCardCandle(key) {{
            const q = quadsRegistry[key];
            if (!q || !q.candleSeries) return;
            if (q.isCandleVisible && !q.isLineVisible) {{
                q.isLineVisible = true;
                if (q.lineSeries) q.lineSeries.applyOptions({{ visible: true }});
                const lBtn = document.getElementById('btn_line_' + key);
                if (lBtn) lBtn.classList.add('active');
            }}
            q.isCandleVisible = !q.isCandleVisible;
            q.candleSeries.applyOptions({{ visible: q.isCandleVisible }});
            const cBtn = document.getElementById('btn_candle_' + key);
            if (cBtn) cBtn.classList.toggle('active', q.isCandleVisible);
        }}

        function toggleCardLine(key) {{
            const q = quadsRegistry[key];
            if (!q || !q.lineSeries) return;
            if (q.isLineVisible && !q.isCandleVisible) {{
                q.isCandleVisible = true;
                if (q.candleSeries) q.candleSeries.applyOptions({{ visible: true }});
                const cBtn = document.getElementById('btn_candle_' + key);
                if (cBtn) cBtn.classList.add('active');
            }}
            q.isLineVisible = !q.isLineVisible;
            q.lineSeries.applyOptions({{ visible: q.isLineVisible }});
            const lBtn = document.getElementById('btn_line_' + key);
            if (lBtn) lBtn.classList.toggle('active', q.isLineVisible);
        }}

        function toggleQuadTooltip() {{
            globalTooltipEnabled = !globalTooltipEnabled;
            const btn = document.getElementById('btn_q_tooltip');
            if (btn) {{
                btn.classList.toggle('active', globalTooltipEnabled);
                btn.innerHTML = globalTooltipEnabled ? "💬 Tooltip" : "💬 Tooltip (Off)";
            }}
            if (!globalTooltipEnabled) {{
                document.querySelectorAll('.tv-floating-tooltip').forEach(el => {{
                    el.style.display = 'none';
                }});
            }}
        }}

        function toggleQuadTheme() {{
            const isCurrentlyLight = (QUAD_DATA.theme === 'light');
            const nextTheme = isCurrentlyLight ? 'dark' : 'light';
            QUAD_DATA.theme = nextTheme;
            const isLight = (nextTheme === 'light');

            const root = document.getElementById('quad_root');
            if (root) {{
                root.classList.toggle('light-theme', isLight);
            }}
            if (document.body) {{
                document.body.classList.toggle('light-theme', isLight);
            }}
            const themeBtn = document.getElementById('btn_q_theme');
            if (themeBtn) {{
                themeBtn.innerHTML = isLight ? "☀️ Light" : "🌙 Dark";
            }}

            const bgCol = isLight ? '#ffffff' : '#131722';
            const txtCol = isLight ? '#131722' : '#94A3B8';
            const gridCol = isLight ? '#f0f3fa' : '#1E222D';
            const borderCol = isLight ? '#e0e3eb' : '#2A2E39';

            const opts = {{
                layout: {{ background: {{ type: 'solid', color: bgCol }}, textColor: txtCol }},
                grid: {{ vertLines: {{ color: gridCol }}, horzLines: {{ color: gridCol }} }},
                rightPriceScale: {{ borderColor: borderCol }},
                timeScale: {{ borderColor: borderCol }},
                crosshair: {{
                    vertLine: {{
                        color: isLight ? 'rgba(0, 0, 0, 0.22)' : 'rgba(255, 255, 255, 0.28)',
                        labelBackgroundColor: isLight ? '#2563EB' : '#1E293B',
                    }},
                    horzLine: {{
                        color: isLight ? 'rgba(0, 0, 0, 0.22)' : 'rgba(255, 255, 255, 0.28)',
                        labelBackgroundColor: isLight ? '#2563EB' : '#1E293B',
                    }}
                }}
            }};

            Object.keys(quadsRegistry).forEach(key => {{
                const q = quadsRegistry[key];
                if (!q) return;
                if (q.mainChart) q.mainChart.applyOptions(opts);
                if (q.rsiChart) q.rsiChart.applyOptions(opts);
                if (q.hmCloud) {{
                    q.hmCloud.applyOptions({{
                        topFillColor1: isLight ? 'rgba(244, 143, 177, 0.45)' : 'rgba(239, 83, 80, 0.35)',
                        topFillColor2: isLight ? 'rgba(244, 143, 177, 0.20)' : 'rgba(239, 83, 80, 0.15)',
                        bottomFillColor1: isLight ? 'rgba(144, 202, 249, 0.20)' : 'rgba(33, 150, 243, 0.15)',
                        bottomFillColor2: isLight ? 'rgba(144, 202, 249, 0.45)' : 'rgba(33, 150, 243, 0.35)',
                        topLineColor: isLight ? 'rgba(233, 30, 99, 0.35)' : 'rgba(244, 67, 54, 0.35)',
                        bottomLineColor: isLight ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.35)'
                    }});
                }}
                if (q.rsiSeries) {{
                    q.rsiSeries.applyOptions({{
                        color: isLight ? '#131722' : '#F8FAFC'
                    }});
                }}
                if (q.emaSeriesMap && q.emaSeriesMap['EMA_200']) {{
                    q.emaSeriesMap['EMA_200'].series.applyOptions({{
                        color: isLight ? '#131722' : '#FFFFFF'
                    }});
                }}
            }});
            renderAllQuadDrawings();
        }}

        function triggerQuadUpdate(stock, tf) {{
            try {{
                const pDoc = window.parent.document;
                if (!pDoc) return false;

                // 1. Remember fullscreen status so chart automatically restores fullscreen after data reload
                const root = document.getElementById('quad_root');
                const isFs = (document.fullscreenElement === root) || (root && root.classList.contains('css-fullscreen'));
                try {{
                    if (window.parent && window.parent.sessionStorage) {{
                        if (isFs) {{
                            window.parent.sessionStorage.setItem('quad_is_fullscreen', '1');
                        }} else {{
                            window.parent.sessionStorage.removeItem('quad_is_fullscreen');
                        }}
                    }}
                }} catch(e) {{}}

                // 2. Show loading badge on chart header
                const badge = document.getElementById('sync_status_badge');
                if (badge) {{
                    badge.innerHTML = '⏳ Loading ' + (stock || tf || '') + '...';
                    badge.style.color = '#FACC15';
                }}

                // 3. Helper to find React Fiber on an element or its children
                function getFiber(el) {{
                    if (!el) return null;
                    const k = Object.keys(el).find(key => key.startsWith('__reactFiber$') || key.startsWith('__reactInternalInstance$'));
                    if (k && el[k]) return el[k];
                    const input = el.querySelector('input');
                    if (input) {{
                        const ik = Object.keys(input).find(key => key.startsWith('__reactFiber$') || key.startsWith('__reactInternalInstance$'));
                        if (ik && input[ik]) return input[ik];
                    }}
                    return null;
                }}

                // 4. Update Streamlit Selectbox via React Fiber
                function setSelectboxViaFiber(boxEl, targetValue) {{
                    if (!boxEl || !targetValue) return false;
                    let curr = getFiber(boxEl);
                    let depth = 0;
                    while (curr && depth < 30) {{
                        depth++;
                        if (curr.memoizedProps && typeof curr.memoizedProps.onChange === 'function') {{
                            if (Array.isArray(curr.memoizedProps.options)) {{
                                const opts = curr.memoizedProps.options;
                                const tClean = String(targetValue).trim().toUpperCase();
                                let matchedOpt = opts.find(o => String(o).trim().toUpperCase() === tClean);
                                if (!matchedOpt) {{
                                    matchedOpt = opts.find(o => String(o).trim().toUpperCase().startsWith(tClean));
                                }}
                                if (!matchedOpt && tClean.includes('M')) {{
                                    matchedOpt = opts.find(o => String(o).toLowerCase().includes('custom'));
                                }}
                                if (matchedOpt !== undefined) {{
                                    try {{
                                        curr.memoizedProps.onChange(matchedOpt);
                                        return true;
                                    }} catch(err) {{
                                        console.warn("onChange call failed:", err);
                                    }}
                                }}
                            }}
                        }}
                        curr = curr.return;
                    }}
                    return false;
                }}

                const selectBoxes = Array.from(pDoc.querySelectorAll('[data-testid="stSelectbox"]'));
                let stockUpdated = false;
                let tfUpdated = false;

                if (stock) {{
                    let stockBox = selectBoxes.find(b => {{
                        const txt = (b.innerText || '').toLowerCase();
                        return txt.includes('select stock') || txt.includes('search 2,500');
                    }}) || selectBoxes[0];
                    if (stockBox) {{
                        stockUpdated = setSelectboxViaFiber(stockBox, stock);
                    }}
                }}

                if (tf) {{
                    let tfBox = selectBoxes.find(b => {{
                        const txt = (b.innerText || '').toLowerCase();
                        return txt.includes('timeframe') || txt.includes('q4');
                    }}) || (selectBoxes.length > 1 ? selectBoxes[1] : null);
                    if (tfBox) {{
                        tfUpdated = setSelectboxViaFiber(tfBox, tf);
                    }}
                }}

                // 5. Secondary fallback: bridge input with React valueTracker reset
                const bridgeInput = pDoc.querySelector('input[aria-label="quad_bridge"]') || pDoc.querySelector('[data-testid="stTextInput"] input');
                if (bridgeInput) {{
                    try {{
                        const payload = JSON.stringify({{ stock: stock, tf: tf, _ts: Date.now() }});
                        if (bridgeInput._valueTracker) {{
                            bridgeInput._valueTracker.setValue('');
                        }}
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
                        nativeInputValueSetter.call(bridgeInput, payload);
                        bridgeInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        bridgeInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        bridgeInput.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                        bridgeInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                    }} catch(e) {{
                        console.warn("Bridge input fallback failed:", e);
                    }}
                }}

                return stockUpdated || tfUpdated;
            }} catch(e) {{
                console.error("Bridge update error:", e);
            }}
            return false;
        }}

        function onQuadStockChange(sym) {{
            if (!sym) return;
            triggerQuadUpdate(sym, null);
        }}

        function onQuadTfChange(tf) {{
            if (!tf) return;
            let targetTf = tf;
            if (tf === 'custom' || tf === 'custom_m') {{
                const customVal = prompt('Enter custom timeframe in minutes (e.g. 10, 45, 90, 120):', '45');
                if (!customVal || isNaN(parseInt(customVal))) {{
                    const sel = document.getElementById('qm_tf_select');
                    if (sel) sel.value = '{cur_tf}';
                    return;
                }}
                targetTf = parseInt(customVal) + 'm';
            }} else if (tf === 'custom_d') {{
                const customVal = prompt('Enter custom timeframe in days (e.g. 2, 3, 5):', '2');
                if (!customVal || isNaN(parseInt(customVal))) {{
                    const sel = document.getElementById('qm_tf_select');
                    if (sel) sel.value = '{cur_tf}';
                    return;
                }}
                targetTf = parseInt(customVal) + 'D';
            }} else if (tf === 'custom_w') {{
                const customVal = prompt('Enter custom timeframe in weeks (e.g. 2, 3):', '2');
                if (!customVal || isNaN(parseInt(customVal))) {{
                    const sel = document.getElementById('qm_tf_select');
                    if (sel) sel.value = '{cur_tf}';
                    return;
                }}
                targetTf = parseInt(customVal) + 'W';
            }} else if (tf === 'custom_mo') {{
                const customVal = prompt('Enter custom timeframe in months (e.g. 2, 3, 6):', '3');
                if (!customVal || isNaN(parseInt(customVal))) {{
                    const sel = document.getElementById('qm_tf_select');
                    if (sel) sel.value = '{cur_tf}';
                    return;
                }}
                targetTf = parseInt(customVal) + 'M';
            }}
            triggerQuadUpdate(null, targetTf);
        }}

        function toggleQuadFullscreen() {{
            const root = document.getElementById('quad_root');
            const fsBtns = [document.getElementById('btn_quad_fs_main')].filter(Boolean);
            const isFs = document.fullscreenElement === root || root.classList.contains('css-fullscreen');

            function updateFsBtns(inFs) {{
                fsBtns.forEach(btn => {{
                    if (!btn) return;
                    if (inFs) {{
                        btn.innerHTML = "✖ Exit Fullscreen";
                        btn.classList.add("is-fs");
                    }} else {{
                        btn.innerHTML = "⛶ FULLSCREEN";
                        btn.classList.remove("is-fs");
                    }}
                }});
            }}

            function setParentIframeFs(enable) {{
                try {{
                    if (window.parent && window.parent.document) {{
                        const iframes = window.parent.document.querySelectorAll('iframe');
                        for (let ifr of iframes) {{
                            if (ifr.contentWindow === window) {{
                                if (enable) {{
                                    ifr.setAttribute('data-orig-style', ifr.getAttribute('style') || '');
                                    ifr.style.position = 'fixed';
                                    ifr.style.top = '0';
                                    ifr.style.left = '0';
                                    ifr.style.width = '100vw';
                                    ifr.style.height = '100vh';
                                    ifr.style.zIndex = '99999999';
                                    ifr.style.border = 'none';
                                    ifr.style.background = '#0b0e14';
                                }} else {{
                                    const orig = ifr.getAttribute('data-orig-style') || '';
                                    ifr.setAttribute('style', orig);
                                }}
                                break;
                            }}
                        }}
                    }}
                }} catch (e) {{}}
            }}

            if (!isFs) {{
                // Always restore 4-quadrant layout when entering fullscreen
                setQuadMode('all');
                const reqFs = root.requestFullscreen || root.webkitRequestFullscreen || root.mozRequestFullScreen || root.msRequestFullscreen;
                if (reqFs) {{
                    try {{
                        const p = reqFs.call(root);
                        if (p && p.then) {{
                            p.then(() => {{
                                updateFsBtns(true);
                            }}).catch(() => {{
                                root.classList.add('css-fullscreen');
                                setParentIframeFs(true);
                                updateFsBtns(true);
                            }});
                        }} else {{
                            updateFsBtns(true);
                        }}
                    }} catch (e) {{
                        root.classList.add('css-fullscreen');
                        setParentIframeFs(true);
                        updateFsBtns(true);
                    }}
                }} else {{
                    root.classList.add('css-fullscreen');
                    setParentIframeFs(true);
                    updateFsBtns(true);
                }}
                try {{
                    if (window.parent && window.parent.sessionStorage) {{
                        window.parent.sessionStorage.setItem('quad_is_fullscreen', '1');
                    }}
                }} catch(e) {{}}
            }} else {{
                try {{
                    if (window.parent && window.parent.sessionStorage) {{
                        window.parent.sessionStorage.removeItem('quad_is_fullscreen');
                    }}
                }} catch(e) {{}}
                if (document.fullscreenElement) {{
                    const exitFs = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
                    if (exitFs) exitFs.call(document).catch(() => {{}});
                }}
                root.classList.remove('css-fullscreen');
                setParentIframeFs(false);
                updateFsBtns(false);
            }}
            setTimeout(resizeAll, 40);
            setTimeout(resizeAll, 150);
            setTimeout(resizeAll, 300);
        }}

        document.addEventListener('fullscreenchange', () => {{
            const root = document.getElementById('quad_root');
            const fsBtns = [document.getElementById('btn_quad_fs_main')].filter(Boolean);
            const inFs = !!document.fullscreenElement;
            fsBtns.forEach(btn => {{
                if (!btn) return;
                if (inFs) {{
                    btn.innerHTML = "✖ Exit Fullscreen";
                    btn.classList.add("is-fs");
                }} else {{
                    btn.innerHTML = "⛶ FULLSCREEN";
                    btn.classList.remove("is-fs");
                }}
            }});
            if (!inFs) {{
                root.classList.remove('css-fullscreen');
                setParentIframeFs(false);
                try {{
                    if (window.parent && window.parent.sessionStorage) {{
                        window.parent.sessionStorage.removeItem('quad_is_fullscreen');
                    }}
                }} catch(e) {{}}
            }}
            setTimeout(resizeAll, 50);
            setTimeout(resizeAll, 200);
        }});

        // Keyboard Shortcut: 'F' toggles fullscreen, 'Esc' exits
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'f' || e.key === 'F') {{
                if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {{
                    toggleQuadFullscreen();
                }}
            }} else if (e.key === 'Escape') {{
                const root = document.getElementById('quad_root');
                if (root && root.classList.contains('css-fullscreen')) {{
                    toggleQuadFullscreen();
                }}
            }}
        }});

        window.addEventListener('resize', () => {{
            setTimeout(resizeAll, 40);
        }});

        // ResizeObserver ensures instant responsive adaptation
        if (window.ResizeObserver) {{
            const ro = new ResizeObserver(() => {{
                requestAnimationFrame(resizeAll);
            }});
            ro.observe(document.getElementById('quad_grid'));
        }}

        // Auto-restore fullscreen if user was in fullscreen before stock/tf reload
        function checkAutoRestoreFullscreen() {{
            try {{
                if (window.parent && window.parent.sessionStorage) {{
                    if (window.parent.sessionStorage.getItem('quad_is_fullscreen') === '1') {{
                        const root = document.getElementById('quad_root');
                        if (root) {{
                            root.classList.add('css-fullscreen');
                            const fsBtns = [document.getElementById('btn_quad_fs_main')].filter(Boolean);
                            fsBtns.forEach(btn => {{
                                if (btn) {{
                                    btn.innerHTML = "✖ Exit Fullscreen";
                                    btn.classList.add("is-fs");
                                }}
                            }});
                            if (window.parent && window.parent.document) {{
                                const iframes = window.parent.document.querySelectorAll('iframe');
                                for (let ifr of iframes) {{
                                    if (ifr.contentWindow === window) {{
                                        ifr.setAttribute('data-orig-style', ifr.getAttribute('style') || '');
                                        ifr.style.position = 'fixed';
                                        ifr.style.top = '0';
                                        ifr.style.left = '0';
                                        ifr.style.width = '100vw';
                                        ifr.style.height = '100vh';
                                        ifr.style.zIndex = '99999999';
                                        ifr.style.border = 'none';
                                        ifr.style.background = '#0b0e14';
                                        break;
                                    }}
                                }}
                            }}
                            setTimeout(resizeAll, 40);
                            setTimeout(resizeAll, 120);
                        }}
                    }}
                }}
            }} catch(e) {{}}
        }}

        checkAutoRestoreFullscreen();

        // Run resize immediately on DOM load and with progressive delays
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
            resizeAll();
        }} else {{
            document.addEventListener('DOMContentLoaded', resizeAll);
        }}
        setTimeout(resizeAll, 50);
        setTimeout(resizeAll, 200);
        setTimeout(resizeAll, 500);
    </script>
</body>
</html>"""
    return html_code

