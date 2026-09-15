"""
quadrant_image_generator.py
---------------------------
Institutional-Grade Headless Quad-Chart Screenshot Generator for Upstox Scanner.
Generates a 1600x1200 4-quadrant PNG composite image showing:
  - Top-Left: Monthly Candlesticks + 5/20 EMAs + RSI 9 Hilega Milega Subplot
  - Top-Right: Weekly Candlesticks + 20/50 EMAs + RSI 9 Hilega Milega Subplot
  - Bottom-Left: Daily Candlesticks + 20/50 EMAs + RSI 9 Hilega Milega Subplot
  - Bottom-Right: 75-Min Candlesticks + 5/20 EMAs + RSI 9 Trigger Subplot
Built entirely with Pillow (PIL) for sub-50ms rendering with zero browser dependencies.
Supports both Light and Dark themes (defaults to crisp Light Theme) and authentic IST timestamps.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config
import scanner
import database
import parquet_loader
import logging

logger = logging.getLogger("quadrant_generator")

IST = ZoneInfo("Asia/Kolkata")

SCREENSHOTS_DIR = config.DATA_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Theme color palettes
THEME_PALETTES: Dict[str, Dict[str, Any]] = {
    "light": {
        "bg": "#F8FAFC",
        "header_bg": "#FFFFFF",
        "header_border": "#CBD5E1",
        "symbol_color": "#0284C7",
        "ltp_color": "#0F172A",
        "badge_bg": "#DCFCE7",
        "badge_border": "#16A34A",
        "badge_text": "#15803D",
        "panel_bg": "#FFFFFF",
        "panel_border": "#CBD5E1",
        "panel_header_bg": "#F1F5F9",
        "panel_title_color": "#0F172A",
        "price_grid": "#F1F5F9",
        "axis_text": "#64748B",
        "date_tick_text": "#64748B",
        "candle_green": "#16A34A",
        "candle_red": "#DC2626",
        "rsi_bg": "#F8FAFC",
        "rsi_border": "#CBD5E1",
        "rsi_50": "#CBD5E1",
        "rsi_70": "#FCA5A5",
        "rsi_30": "#86EFAC",
        "rsi_text_70": "#DC2626",
        "rsi_text_50": "#64748B",
        "rsi_text_30": "#16A34A",
        "rsi_legend": "#334155",
        "rsi_line": "#059669",
        "re3_line": "#DC2626",
        "rw21_line": "#2563EB",
        "sub_text": "#64748B",
        "brand_color": "#0284C7",
        "ema_colors": {
            "monthly": ("#0284C7", "#D97706"),
            "weekly": ("#D97706", "#9333EA"),
            "daily": ("#D97706", "#9333EA"),
            "q4": ("#0284C7", "#D97706"),
        }
    },
    "dark": {
        "bg": "#0F121C",
        "header_bg": "#1E222D",
        "header_border": "#2A2E39",
        "symbol_color": "#38BDF8",
        "ltp_color": "#FFFFFF",
        "badge_bg": "#064E3B",
        "badge_border": "#10B981",
        "badge_text": "#34D399",
        "panel_bg": "#131722",
        "panel_border": "#2A2E39",
        "panel_header_bg": "#1E222D",
        "panel_title_color": "#38BDF8",
        "price_grid": "#1E222D",
        "axis_text": "#64748B",
        "date_tick_text": "#64748B",
        "candle_green": "#10B981",
        "candle_red": "#EF4444",
        "rsi_bg": "#0F121C",
        "rsi_border": "#1E222D",
        "rsi_50": "#334155",
        "rsi_70": "#7F1D1D",
        "rsi_30": "#064E3B",
        "rsi_text_70": "#EF4444",
        "rsi_text_50": "#64748B",
        "rsi_text_30": "#10B981",
        "rsi_legend": "#94A3B8",
        "rsi_line": "#10B981",
        "re3_line": "#EF4444",
        "rw21_line": "#3B82F6",
        "sub_text": "#94A3B8",
        "brand_color": "#38BDF8",
        "ema_colors": {
            "monthly": ("#38BDF8", "#F59E0B"),
            "weekly": ("#F59E0B", "#EC4899"),
            "daily": ("#F59E0B", "#EC4899"),
            "q4": ("#38BDF8", "#F59E0B"),
        }
    }
}


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempts to load a clean system TrueType font, falling back to default."""
    font_paths = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _format_timestamp(ts_val: Any, is_intra: bool = False) -> str:
    """Formats a candle index/timestamp to IST string."""
    try:
        ts = pd.to_datetime(ts_val)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(IST)
        else:
            ts = ts.tz_localize(IST)
        if is_intra:
            return ts.strftime("%d %b %H:%M IST")
        else:
            return ts.strftime("%d %b %Y")
    except Exception:
        return str(ts_val)[:10]


def _draw_single_panel(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    title: str,
    bbox: Tuple[int, int, int, int],
    ema_pairs: Tuple[str, str] = ("EMA_5", "EMA_20"),
    ema_colors: Tuple[str, str] = ("#0284C7", "#D97706"),
    max_bars: int = 40,
    pal: Optional[Dict[str, Any]] = None
):
    """
    Renders one timeframe panel: Candlesticks on top (72%), RSI on bottom (28%).
    bbox: (x1, y1, x2, y2)
    """
    if pal is None:
        pal = THEME_PALETTES["light"]

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Panel border & background
    draw.rectangle([x1, y1, x2, y2], fill=pal["panel_bg"], outline=pal["panel_border"], width=1)

    # Title header
    font_title = _get_font(13)
    font_sm = _get_font(10)
    draw.rectangle([x1, y1, x2, y1 + 26], fill=pal["panel_header_bg"])
    draw.text((x1 + 10, y1 + 6), title, fill=pal["panel_title_color"], font=font_title)

    if df is None or df.empty or len(df) < 5:
        draw.text((x1 + width // 3, y1 + height // 2), "Insufficient Candle Data", fill=pal["axis_text"], font=font_title)
        return

    sub_df = df.tail(max_bars).copy()
    n_bars = len(sub_df)
    is_intraday = bool(re.search(r"\b(\d+[\s-]*(min|m)|intraday)\b", title, re.IGNORECASE))

    # Calculate indicators if missing
    close = sub_df["close"]
    high = sub_df["high"]
    low = sub_df["low"]
    open_p = sub_df["open"]

    # Latest candle time label in panel header
    last_bar_ts = sub_df.index[-1]
    latest_ts_str = _format_timestamp(last_bar_ts, is_intra=is_intraday)
    t_len = int(draw.textlength(title, font=font_title))
    ts_x = x1 + 15 + t_len
    if ts_x + 130 < (x2 - 180):
        draw.text((ts_x, y1 + 7), f"[{latest_ts_str}]", fill=pal["sub_text"], font=font_sm)

    # RSI Hilega Milega
    if "RSI_9" in sub_df.columns:
        rsi = sub_df["RSI_9"]
    elif "RSI" in sub_df.columns:
        rsi = sub_df["RSI"]
    else:
        rsi = scanner.calculate_rsi(close, span=9)

    if "RSI_EMA3" in sub_df.columns:
        rsi_ema3 = sub_df["RSI_EMA3"]
    else:
        rsi_ema3 = scanner.calculate_ema(rsi, span=3)

    if "RSI_WMA21" in sub_df.columns:
        rsi_wma21 = sub_df["RSI_WMA21"]
    else:
        rsi_wma21 = scanner.calculate_wma(rsi, period=21)

    # EMAs
    e1_span = int(ema_pairs[0].split("_")[1]) if "_" in ema_pairs[0] else 5
    e2_span = int(ema_pairs[1].split("_")[1]) if "_" in ema_pairs[1] else 20
    ema1 = sub_df[ema_pairs[0]] if ema_pairs[0] in sub_df.columns else scanner.calculate_ema(close, span=e1_span)
    ema2 = sub_df[ema_pairs[1]] if ema_pairs[1] in sub_df.columns else scanner.calculate_ema(close, span=e2_span)

    # Price chart area (from y1+30 to y1 + height * 0.70)
    p_top = y1 + 30
    p_bot = y1 + int(height * 0.70)
    p_h = p_bot - p_top

    # RSI chart area (from y1 + height * 0.74 to y2 - 10)
    r_top = y1 + int(height * 0.74)
    r_bot = y2 - 10
    r_h = r_bot - r_top

    # Min/max price for scaling
    p_min = float(min(low.min(), ema1.dropna().min() if not ema1.dropna().empty else low.min(), ema2.dropna().min() if not ema2.dropna().empty else low.min())) * 0.995
    p_max = float(max(high.max(), ema1.dropna().max() if not ema1.dropna().empty else high.max(), ema2.dropna().max() if not ema2.dropna().empty else high.max())) * 1.005
    p_range = max(p_max - p_min, 1e-4)

    def price_to_y(p: float) -> int:
        return int(p_bot - ((p - p_min) / p_range) * p_h)

    def rsi_to_y(r: float) -> int:
        r_clamped = max(10.0, min(90.0, r))
        return int(r_bot - ((r_clamped - 10.0) / 80.0) * r_h)

    # Draw grid lines for price
    for p_step in np.linspace(p_min, p_max, 4):
        gy = price_to_y(p_step)
        draw.line([(x1 + 10, gy), (x2 - 55, gy)], fill=pal["price_grid"], width=1)
        draw.text((x2 - 50, gy - 6), f"{p_step:,.0f}", fill=pal["axis_text"], font=font_sm)

    # Bar width & spacing
    chart_w = width - 65
    slot_w = chart_w / max_bars
    candle_w = max(2, int(slot_w * 0.65))

    ema1_pts = []
    ema2_pts = []
    rsi_pts = []
    re3_pts = []
    rw21_pts = []

    for i in range(n_bars):
        bx = int(x1 + 10 + i * slot_w + slot_w / 2)
        o_val = float(open_p.iloc[i])
        c_val = float(close.iloc[i])
        h_val = float(high.iloc[i])
        l_val = float(low.iloc[i])

        y_o = price_to_y(o_val)
        y_c = price_to_y(c_val)
        y_h = price_to_y(h_val)
        y_l = price_to_y(l_val)

        is_green = (c_val >= o_val)
        clr = pal["candle_green"] if is_green else pal["candle_red"]

        # Wick
        draw.line([(bx, y_h), (bx, y_l)], fill=clr, width=1)
        # Body
        top_b = min(y_o, y_c)
        bot_b = max(y_o, y_c)
        if bot_b - top_b < 1:
            bot_b = top_b + 1
        half_w = candle_w // 2
        draw.rectangle([(bx - half_w, top_b), (bx + half_w, bot_b)], fill=clr, outline=clr)

        # Indicator coordinates
        if not np.isnan(ema1.iloc[i]):
            ema1_pts.append((bx, price_to_y(float(ema1.iloc[i]))))
        if not np.isnan(ema2.iloc[i]):
            ema2_pts.append((bx, price_to_y(float(ema2.iloc[i]))))

        if not np.isnan(rsi.iloc[i]):
            rsi_pts.append((bx, rsi_to_y(float(rsi.iloc[i]))))
        if not np.isnan(rsi_ema3.iloc[i]):
            re3_pts.append((bx, rsi_to_y(float(rsi_ema3.iloc[i]))))
        if not np.isnan(rsi_wma21.iloc[i]):
            rw21_pts.append((bx, rsi_to_y(float(rsi_wma21.iloc[i]))))

    # Draw EMAs
    if len(ema1_pts) > 1:
        draw.line(ema1_pts, fill=ema_colors[0], width=2)
    if len(ema2_pts) > 1:
        draw.line(ema2_pts, fill=ema_colors[1], width=2)

    # EMA Legend in top right of panel
    draw.text((x2 - 170, y1 + 7), f"{ema_pairs[0]}: {ema1.iloc[-1]:.1f}", fill=ema_colors[0], font=font_sm)
    draw.text((x2 - 85, y1 + 7), f"{ema_pairs[1]}: {ema2.iloc[-1]:.1f}", fill=ema_colors[1], font=font_sm)

    # Date / Time ticks along X-axis (between p_bot and r_top)
    tick_indices = [0, n_bars // 3, (2 * n_bars) // 3, n_bars - 1]
    for tidx in tick_indices:
        if 0 <= tidx < n_bars:
            t_bx = int(x1 + 10 + tidx * slot_w + slot_w / 2)
            t_val = sub_df.index[tidx]
            try:
                t_dt = pd.to_datetime(t_val)
                if t_dt.tzinfo is not None:
                    t_dt = t_dt.tz_convert(IST)
                else:
                    t_dt = t_dt.tz_localize(IST)
                lbl = t_dt.strftime("%d %b %H:%M") if is_intraday else t_dt.strftime("%d %b '%y")
            except Exception:
                lbl = str(t_val)[:10]

            draw.line([(t_bx, p_bot - 3), (t_bx, p_bot + 1)], fill=pal["axis_text"], width=1)
            txt_x = max(x1 + 4, min(t_bx - 20, x2 - 70))
            draw.text((txt_x, p_bot + 3), lbl, fill=pal["date_tick_text"], font=font_sm)

    # RSI Subplot Section
    draw.rectangle([x1, r_top - 6, x2, r_bot + 4], fill=pal["rsi_bg"], outline=pal["rsi_border"], width=1)
    # 50, 70, 30 reference lines
    y_50 = rsi_to_y(50.0)
    y_70 = rsi_to_y(70.0)
    y_30 = rsi_to_y(30.0)
    draw.line([(x1 + 10, y_50), (x2 - 40, y_50)], fill=pal["rsi_50"], width=1)
    draw.line([(x1 + 10, y_70), (x2 - 40, y_70)], fill=pal["rsi_70"], width=1)
    draw.line([(x1 + 10, y_30), (x2 - 40, y_30)], fill=pal["rsi_30"], width=1)
    draw.text((x2 - 35, y_50 - 5), "50", fill=pal["rsi_text_50"], font=font_sm)
    draw.text((x2 - 35, y_70 - 5), "70", fill=pal["rsi_text_70"], font=font_sm)
    draw.text((x2 - 35, y_30 - 5), "30", fill=pal["rsi_text_30"], font=font_sm)

    if len(rsi_pts) == 0:
        draw.text((x1 + 20, r_top + r_h // 2 - 6), f"⚠️ Awaiting 9 completed candles for RSI(9) ({len(sub_df)}/9 bars completed)", fill=pal["axis_text"], font=font_sm)
    else:
        if len(rsi_pts) > 1:
            draw.line(rsi_pts, fill=pal["rsi_line"], width=2)  # RSI(9)
        if len(re3_pts) > 1:
            draw.line(re3_pts, fill=pal["re3_line"], width=1)  # EMA 3
        if len(rw21_pts) > 1:
            draw.line(rw21_pts, fill=pal["rw21_line"], width=1)  # WMA 21

        # RSI value text
        curr_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 50.0
        curr_e3 = rsi_ema3.dropna().iloc[-1] if not rsi_ema3.dropna().empty else 50.0
        curr_w21 = rsi_wma21.dropna().iloc[-1] if not rsi_wma21.dropna().empty else 50.0
        draw.text((x1 + 10, r_top - 4), f"RSI(9): {curr_rsi:.1f} | EMA(3): {curr_e3:.1f} | WMA(21): {curr_w21:.1f}", fill=pal["rsi_legend"], font=font_sm)


def generate_quadrant_image(
    symbol: str,
    monthly_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    intra_75_df: pd.DataFrame,
    ltp: Optional[float] = None,
    change_pct: Optional[float] = None,
    stage_label: str = "🏆 STAGE 4 FULL ALIGNMENT QUALIFIED",
    out_path: Optional[str] = None,
    q4_title: Optional[str] = None,
    theme: str = "light"
) -> str:
    """
    Renders a 1600x1200 composite Quad-Chart screenshot image for a stock.
    Supports 'light' (default) and 'dark' themes with accurate IST timestamps.
    Returns: Absolute file path to the generated PNG image.
    """
    theme_key = "dark" if (theme and str(theme).lower().strip() == "dark") else "light"
    pal = THEME_PALETTES[theme_key]

    img_w, img_h = 1600, 1200
    img = Image.new("RGB", (img_w, img_h), color=pal["bg"])
    draw = ImageDraw.Draw(img)

    # 1. Header Banner (Height: 80px)
    draw.rectangle([(0, 0), (img_w, 75)], fill=pal["header_bg"], outline=pal["header_border"], width=1)

    font_sym = _get_font(26)
    font_bold = _get_font(14)
    font_sub = _get_font(12)

    # Symbol & Branding
    draw.text((25, 12), symbol.upper(), fill=pal["symbol_color"], font=font_sym)

    if ltp is None and daily_df is not None and not daily_df.empty:
        ltp = float(daily_df["close"].iloc[-1])
        if len(daily_df) >= 2:
            prev = float(daily_df["close"].iloc[-2])
            change_pct = ((ltp - prev) / prev) * 100.0
        else:
            change_pct = 0.0

    ltp_str = f"₹{ltp:,.2f}" if ltp else "₹---"
    chg_str = f"{change_pct:+.2f}%" if change_pct is not None else "+0.00%"
    chg_clr = pal["candle_green"] if (change_pct and change_pct >= 0) else pal["candle_red"]

    draw.text((230, 18), ltp_str, fill=pal["ltp_color"], font=font_bold)
    draw.text((320, 18), f"({chg_str})", fill=chg_clr, font=font_bold)

    # Stage Badge in Center
    draw.rectangle([(620, 14), (1150, 48)], fill=pal["badge_bg"], outline=pal["badge_border"], width=2)
    draw.text((640, 20), stage_label, fill=pal["badge_text"], font=font_bold)

    # Timestamp in Indian Standard Time (IST) on right
    now_ist = datetime.now(IST)
    now_str = now_ist.strftime("%d %b %Y %I:%M:%S %p IST")
    draw.text((img_w - 380, 16), f"Generated: {now_str}", fill=pal["sub_text"], font=font_sub)
    draw.text((img_w - 380, 36), "Upstox Pro 75m Auto-Broadcaster", fill=pal["brand_color"], font=font_sub)

    # 2. Quadrants Layout (Margin: 15px, Gap: 15px)
    pad = 15
    grid_top = 85
    col_w = (img_w - (pad * 3)) // 2  # ~770
    row_h = (img_h - grid_top - (pad * 2)) // 2  # ~540

    ema_cols = pal["ema_colors"]

    # Q1: Monthly (Top-Left) - Data strictly from 2020 to current date
    m_bars = len(monthly_df) if (monthly_df is not None and not monthly_df.empty) else 84
    bbox_q1 = (pad, grid_top, pad + col_w, grid_top + row_h)
    _draw_single_panel(
        draw, monthly_df, "1. MONTHLY MACRO TREND (2020 - Present | 5 & 20 EMA)",
        bbox_q1, ema_pairs=("EMA_5", "EMA_20"), ema_colors=ema_cols["monthly"], max_bars=max(m_bars, 36), pal=pal
    )

    # Q2: Weekly (Top-Right)
    bbox_q2 = (pad * 2 + col_w, grid_top, img_w - pad, grid_top + row_h)
    _draw_single_panel(
        draw, weekly_df, "2. WEEKLY INTERMEDIATE (20 & 50 EMA)",
        bbox_q2, ema_pairs=("EMA_20", "EMA_50"), ema_colors=ema_cols["weekly"], max_bars=40, pal=pal
    )

    # Q3: Daily (Bottom-Left)
    bbox_q3 = (pad, grid_top + row_h + pad, pad + col_w, img_h - pad)
    _draw_single_panel(
        draw, daily_df, "3. DAILY SETUP & 20 EMA BOUNCE",
        bbox_q3, ema_pairs=("EMA_20", "EMA_50"), ema_colors=ema_cols["daily"], max_bars=45, pal=pal
    )

    # Q4: Bottom-Right
    q4_name = q4_title or "4. 75-MIN INTRADAY TRIGGER (5 & 20 EMA)"
    bbox_q4 = (pad * 2 + col_w, grid_top + row_h + pad, img_w - pad, img_h - pad)
    _draw_single_panel(
        draw, intra_75_df, q4_name,
        bbox_q4, ema_pairs=("EMA_5", "EMA_20"), ema_colors=ema_cols["q4"], max_bars=40, pal=pal
    )

    # Save image
    if not out_path:
        ts = now_ist.strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.upper()}_quadrant_{theme_key}_{ts}.png"
        out_path = str(SCREENSHOTS_DIR / filename)

    img.save(out_path, format="PNG", optimize=True)
    return out_path


def generate_stock_quadrant(
    symbol: str,
    stage_label: str = "🏆 STAGE 4 FULL ALIGNMENT QUALIFIED",
    out_path: Optional[str] = None,
    q4_timeframe: str = "75m",
    theme: str = "light"
) -> Optional[str]:
    """
    Convenience function: Automatically loads Daily, Monthly, Weekly, and Q4 data
    (75m, custom minutes, Daily, Weekly, Monthly) for `symbol`, generates the composite
    1600x1200 image, and returns the path. Defaults to crisp Light Theme and IST time.
    """
    try:
        daily_df = database.get_candles_df(symbol)
        if daily_df is None or daily_df.empty or len(daily_df) < 15:
            return None

        monthly_df = scanner.resample_ohlcv(daily_df, "monthly")
        weekly_df = scanner.resample_ohlcv(daily_df, "weekly")

        # In quad chart, ONLY Monthly needs data from 2020 to current date
        if monthly_df is not None and not monthly_df.empty:
            monthly_df["EMA_5"] = scanner.calculate_ema(monthly_df["close"], span=5)
            monthly_df["EMA_20"] = scanner.calculate_ema(monthly_df["close"], span=20)
            monthly_df["RSI_9"] = scanner.calculate_rsi(monthly_df["close"], span=9)
            monthly_df["RSI_EMA3"] = scanner.calculate_ema(monthly_df["RSI_9"], span=3)
            monthly_df["RSI_WMA21"] = scanner.calculate_wma(monthly_df["RSI_9"], period=21)

            m_tz = getattr(monthly_df.index, "tz", None)
            start_2020 = pd.to_datetime("2020-01-01")
            m_cutoff = start_2020.tz_localize(m_tz) if m_tz is not None else (start_2020.tz_localize(None) if getattr(start_2020, 'tz', None) is not None else start_2020)
            m_sliced = monthly_df[monthly_df.index >= m_cutoff]
            if not m_sliced.empty:
                monthly_df = m_sliced

        tf_clean = str(q4_timeframe).strip().lower()
        q4_df = None
        q4_title = f"4. {q4_timeframe.upper()} TRIGGER (Close > 20 EMA & 5 EMA >= 20 EMA)"

        if tf_clean in ("1d", "daily", "d"):
            q4_df = daily_df.copy()
            q4_title = "4. DAILY TIMEFRAME (Setup & 20 EMA)"
        elif tf_clean in ("1w", "weekly", "w"):
            q4_df = weekly_df.copy()
            q4_title = "4. WEEKLY TIMEFRAME (Trend & 20/50 EMA)"
        elif tf_clean in ("monthly", "mo", "month", "1mo"):
            q4_df = monthly_df.copy()
            q4_title = "4. MONTHLY TIMEFRAME (Macro Trend & 5/20 EMA)"
        elif re.match(r"^(\d+)d(ays?)?$", tf_clean):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-DAY TIMEFRAME"
        elif re.match(r"^(\d+)w(eeks?)?$", tf_clean):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-WEEK TIMEFRAME"
        elif re.match(r"^(\d+)\s*(mo|months?)$", tf_clean):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-MONTH TIMEFRAME"
        else:
            # Intraday minutes (75m, 15m, 5m, 75min, etc.)
            m_num = re.search(r"(\d+)", tf_clean)
            mins = int(m_num.group(1)) if m_num else 75
            if mins == 75:
                q4_df = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=20)
                q4_title = "4. 75-MIN INTRADAY TRIGGER (5 & 20 EMA)"
            else:
                q4_df = parquet_loader.ensure_symbol_custom_minute_candles(symbol, interval_minutes=mins, min_bars=20)
                q4_title = f"4. {mins}-MIN INTRADAY TRIGGER"

        # Ensure we have valid data for Q4
        if q4_df is None or q4_df.empty or len(q4_df) < 5:
            # Fallback 1: Try custom minute resampling if not already attempted
            try:
                m_num = re.search(r"(\d+)", tf_clean)
                mins = int(m_num.group(1)) if m_num else 75
                q4_df = parquet_loader.ensure_symbol_custom_minute_candles(symbol, interval_minutes=mins, min_bars=5)
            except Exception:
                pass

        # Fallback 2: If intraday is still unavailable (< 5 bars), gracefully display recent Daily candles
        if q4_df is None or q4_df.empty or len(q4_df) < 5:
            logger.info(f"Q4 intraday {q4_timeframe} data limited for {symbol}. Gracefully displaying recent Daily bars.")
            q4_df = daily_df.tail(60).copy()
            q4_title = f"4. DAILY TIMEFRAME ({q4_timeframe.upper()} Ingesting / Syncing)"

        ltp = float(daily_df["close"].iloc[-1])
        change_pct = 0.0
        if len(daily_df) >= 2:
            prev = float(daily_df["close"].iloc[-2])
            change_pct = ((ltp - prev) / prev) * 100.0

        return generate_quadrant_image(
            symbol=symbol,
            monthly_df=monthly_df,
            weekly_df=weekly_df,
            daily_df=daily_df,
            intra_75_df=q4_df,
            ltp=ltp,
            change_pct=change_pct,
            stage_label=stage_label,
            out_path=out_path,
            q4_title=q4_title,
            theme=theme
        )
    except Exception as err:
        logger.error(f"Failed to generate quadrant image for {symbol}: {err}", exc_info=True)
        return None
