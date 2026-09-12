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
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config
import scanner
import database
import parquet_loader

SCREENSHOTS_DIR = config.DATA_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _draw_single_panel(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    title: str,
    bbox: Tuple[int, int, int, int],
    ema_pairs: Tuple[str, str] = ("EMA_5", "EMA_20"),
    ema_colors: Tuple[str, str] = ("#38BDF8", "#F59E0B"),
    max_bars: int = 40
):
    """
    Renders one timeframe panel: Candlesticks on top (72%), RSI on bottom (28%).
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Panel border & background
    draw.rectangle([x1, y1, x2, y2], fill="#131722", outline="#2A2E39", width=1)

    # Title header
    font_title = _get_font(13)
    font_sm = _get_font(10)
    draw.rectangle([x1, y1, x2, y1 + 26], fill="#1E222D")
    draw.text((x1 + 10, y1 + 6), title, fill="#38BDF8", font=font_title)

    if df is None or df.empty or len(df) < 5:
        draw.text((x1 + width // 3, y1 + height // 2), "Insufficient Candle Data", fill="#64748B", font=font_title)
        return

    sub_df = df.tail(max_bars).copy()
    n_bars = len(sub_df)

    # Calculate indicators if missing
    close = sub_df["close"]
    high = sub_df["high"]
    low = sub_df["low"]
    open_p = sub_df["open"]

    # RSI Hilega Milega
    rsi = scanner.calculate_rsi(close, span=9)
    rsi_ema3 = scanner.calculate_ema(rsi, span=3)
    rsi_wma21 = scanner.calculate_wma(rsi, period=21)

    # EMAs
    e1_span = int(ema_pairs[0].split("_")[1]) if "_" in ema_pairs[0] else 5
    e2_span = int(ema_pairs[1].split("_")[1]) if "_" in ema_pairs[1] else 20
    ema1 = scanner.calculate_ema(close, span=e1_span)
    ema2 = scanner.calculate_ema(close, span=e2_span)

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
        draw.line([(x1 + 10, gy), (x2 - 55, gy)], fill="#1E222D", width=1)
        draw.text((x2 - 50, gy - 6), f"{p_step:,.0f}", fill="#64748B", font=font_sm)

    # Bar width & spacing
    chart_w = width - 65
    slot_w = chart_w / max_bars
    candle_w = max(2, int(slot_w * 0.65))

    candle_coords = []
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
        clr = "#10B981" if is_green else "#EF4444"

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

    # RSI Subplot Section
    draw.rectangle([x1, r_top - 6, x2, r_bot + 4], fill="#0F121C", outline="#1E222D", width=1)
    # 50, 70, 30 reference lines
    y_50 = rsi_to_y(50.0)
    y_70 = rsi_to_y(70.0)
    y_30 = rsi_to_y(30.0)
    draw.line([(x1 + 10, y_50), (x2 - 40, y_50)], fill="#334155", width=1)
    draw.line([(x1 + 10, y_70), (x2 - 40, y_70)], fill="#7F1D1D", width=1)
    draw.line([(x1 + 10, y_30), (x2 - 40, y_30)], fill="#064E3B", width=1)
    draw.text((x2 - 35, y_50 - 5), "50", fill="#64748B", font=font_sm)
    draw.text((x2 - 35, y_70 - 5), "70", fill="#EF4444", font=font_sm)

    if len(rsi_pts) == 0:
        draw.text((x1 + 20, r_top + r_h // 2 - 6), f"⚠️ Awaiting 9 completed candles for RSI(9) ({len(sub_df)}/9 bars completed)", fill="#64748B", font=font_sm)
    else:
        if len(rsi_pts) > 1:
            draw.line(rsi_pts, fill="#10B981", width=2)  # RSI(9)
        if len(re3_pts) > 1:
            draw.line(re3_pts, fill="#EF4444", width=1)  # EMA 3
        if len(rw21_pts) > 1:
            draw.line(rw21_pts, fill="#3B82F6", width=1)  # WMA 21

        # RSI value text
        curr_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 50.0
        curr_e3 = rsi_ema3.dropna().iloc[-1] if not rsi_ema3.dropna().empty else 50.0
        curr_w21 = rsi_wma21.dropna().iloc[-1] if not rsi_wma21.dropna().empty else 50.0
        draw.text((x1 + 10, r_top - 4), f"RSI(9): {curr_rsi:.1f} | EMA(3): {curr_e3:.1f} | WMA(21): {curr_w21:.1f}", fill="#94A3B8", font=font_sm)


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
    q4_title: Optional[str] = None
) -> str:
    """
    Renders a 1600x1200 composite Quad-Chart screenshot image for a stock.
    Returns: Absolute file path to the generated PNG image.
    """
    img_w, img_h = 1600, 1200
    img = Image.new("RGB", (img_w, img_h), color="#0F121C")
    draw = ImageDraw.Draw(img)

    # 1. Header Banner (Height: 80px)
    draw.rectangle([(0, 0), (img_w, 75)], fill="#1E222D", outline="#2A2E39", width=1)

    font_sym = _get_font(26)
    font_bold = _get_font(14)
    font_sub = _get_font(12)

    # Symbol & Branding
    draw.text((25, 12), symbol.upper(), fill="#38BDF8", font=font_sym)

    if ltp is None and daily_df is not None and not daily_df.empty:
        ltp = float(daily_df["close"].iloc[-1])
        if len(daily_df) >= 2:
            prev = float(daily_df["close"].iloc[-2])
            change_pct = ((ltp - prev) / prev) * 100.0
        else:
            change_pct = 0.0

    ltp_str = f"₹{ltp:,.2f}" if ltp else "₹---"
    chg_str = f"{change_pct:+.2f}%" if change_pct is not None else "+0.00%"
    chg_clr = "#10B981" if (change_pct and change_pct >= 0) else "#EF4444"

    draw.text((230, 18), ltp_str, fill="#FFFFFF", font=font_bold)
    draw.text((320, 18), f"({chg_str})", fill=chg_clr, font=font_bold)

    # Stage Badge in Center
    draw.rectangle([(620, 14), (1150, 48)], fill="#064E3B", outline="#10B981", width=2)
    draw.text((640, 20), stage_label, fill="#34D399", font=font_bold)

    # Timestamp & Engine label on right
    now_str = datetime.now().strftime("%d %b %Y %H:%M:%S")
    draw.text((img_w - 320, 16), f"Generated: {now_str}", fill="#94A3B8", font=font_sub)
    draw.text((img_w - 320, 36), "Upstox Pro 75m Auto-Broadcaster", fill="#38BDF8", font=font_sub)

    # 2. Quadrants Layout (Margin: 15px, Gap: 15px)
    pad = 15
    grid_top = 85
    col_w = (img_w - (pad * 3)) // 2  # ~770
    row_h = (img_h - grid_top - (pad * 2)) // 2  # ~540

    # Q1: Monthly (Top-Left)
    bbox_q1 = (pad, grid_top, pad + col_w, grid_top + row_h)
    _draw_single_panel(
        draw, monthly_df, f"1. MONTHLY MACRO TREND (Close > 5 EMA > 20 EMA)",
        bbox_q1, ema_pairs=("EMA_5", "EMA_20"), ema_colors=("#38BDF8", "#F59E0B"), max_bars=36
    )

    # Q2: Weekly (Top-Right)
    bbox_q2 = (pad * 2 + col_w, grid_top, img_w - pad, grid_top + row_h)
    _draw_single_panel(
        draw, weekly_df, f"2. WEEKLY INTERMEDIATE TREND (Close > 20 EMA > 50 EMA)",
        bbox_q2, ema_pairs=("EMA_20", "EMA_50"), ema_colors=("#F59E0B", "#EC4899"), max_bars=40
    )

    # Q3: Daily (Bottom-Left)
    bbox_q3 = (pad, grid_top + row_h + pad, pad + col_w, img_h - pad)
    _draw_single_panel(
        draw, daily_df, f"3. DAILY SETUP & 20 EMA BOUNCE (Open <= 20 EMA & Close >= 20 EMA)",
        bbox_q3, ema_pairs=("EMA_20", "EMA_50"), ema_colors=("#F59E0B", "#EC4899"), max_bars=45
    )

    # Q4: 75-Min (Bottom-Right)
    # Q4: Bottom-Right
    q4_name = q4_title or "4. 75-MIN INTRADAY TRIGGER (Close > 20 EMA & 5 EMA >= 20 EMA)"
    bbox_q4 = (pad * 2 + col_w, grid_top + row_h + pad, img_w - pad, img_h - pad)
    _draw_single_panel(
        draw, intra_75_df, q4_name,
        bbox_q4, ema_pairs=("EMA_5", "EMA_20"), ema_colors=("#38BDF8", "#F59E0B"), max_bars=40
    )

    # Save image
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.upper()}_quadrant_{ts}.png"
        out_path = str(SCREENSHOTS_DIR / filename)

    img.save(out_path, format="PNG", optimize=True)
    return out_path


def generate_stock_quadrant(
    symbol: str,
    stage_label: str = "🏆 STAGE 4 FULL ALIGNMENT QUALIFIED",
    out_path: Optional[str] = None,
    q4_timeframe: str = "75m"
) -> Optional[str]:
    """
    Convenience function: Automatically loads Daily, Monthly, Weekly, and Q4 data
    (75m, custom minutes, Daily, Weekly, Monthly) for `symbol`, generates the composite
    1600x1200 image, and returns the path.
    """
    try:
        import re
        daily_df = database.get_candles_df(symbol)
        if daily_df is None or daily_df.empty or len(daily_df) < 15:
            return None

        monthly_df = scanner.resample_ohlcv(daily_df, "monthly")
        weekly_df = scanner.resample_ohlcv(daily_df, "weekly")

        tf_clean = str(q4_timeframe).strip().lower()
        q4_df = None
        q4_title = f"4. {q4_timeframe.upper()} TRIGGER (Close > 20 EMA & 5 EMA >= 20 EMA)"

        if tf_clean in ("1d", "daily", "d"):
            q4_df = daily_df.copy()
            q4_title = "4. DAILY TIMEFRAME (Setup & 20 EMA)"
        elif tf_clean in ("1w", "weekly", "w"):
            q4_df = weekly_df.copy()
            q4_title = "4. WEEKLY TIMEFRAME (Trend & 20/50 EMA)"
        elif tf_clean in ("1m", "monthly", "mo", "month"):
            q4_df = monthly_df.copy()
            q4_title = "4. MONTHLY TIMEFRAME (Macro Trend & 5/20 EMA)"
        elif re.match(r"^(\d+)d$", tf_clean):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-DAY TIMEFRAME"
        elif re.match(r"^(\d+)w$", tf_clean):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-WEEK TIMEFRAME"
        elif re.match(r"^(\d+)m(o|onth)?$", tf_clean) and not tf_clean.endswith("min"):
            q4_df = scanner.resample_ohlcv(daily_df, tf_clean)
            q4_title = f"4. {q4_timeframe.upper()} MULTI-MONTH TIMEFRAME"
        else:
            # Intraday minutes
            m_num = re.search(r"(\d+)", tf_clean)
            mins = int(m_num.group(1)) if m_num else 75
            if mins == 75:
                q4_df = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=20)
                q4_title = "4. 75-MIN INTRADAY TRIGGER (Close > 20 EMA & 5 EMA >= 20 EMA)"
            else:
                q4_df = parquet_loader.ensure_symbol_custom_minute_candles(symbol, interval_minutes=mins, min_bars=20)
                q4_title = f"4. {mins}-MIN INTRADAY TRIGGER"

        if q4_df is None or q4_df.empty:
            q4_df = parquet_loader.ensure_symbol_75m_candles(symbol, min_bars=20)

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
            q4_title=q4_title
        )
    except Exception as err:
        return None
