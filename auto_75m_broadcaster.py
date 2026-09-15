"""
auto_75m_broadcaster.py
-----------------------
Automated 75-Minute Intraday Waterfall Scan & Quadrant Screenshot Broadcaster.

Operates on the 5 intraday 75-minute candle closes in the Indian Stock Market (NSE):
  - Candle 1: 09:15 - 10:30 (Closes at 10:30:00 IST)
  - Candle 2: 10:30 - 11:45 (Closes at 11:45:00 IST)
  - Candle 3: 11:45 - 13:00 (Closes at 13:00:00 IST)
  - Candle 4: 13:00 - 14:15 (Closes at 14:15:00 IST)
  - Candle 5: 14:15 - 15:30 (Closes at 15:30:00 IST)

When triggered:
1. Runs Chartink Positional Scan #364 Waterfall across equities.
2. Identifies qualifying stocks meeting Stage 4 (Full Alignment: Monthly + Weekly + Daily + 75m).
3. Generates high-resolution 1600x1200 4-quadrant candlestick screenshot images.
4. Broadcasts alerts with screenshots to Telegram, Email, WhatsApp, and Webhooks.
5. Records results in audit logs and broadcast history for UI gallery viewing.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

import config
import database
import parquet_loader
import screener_engine
import quadrant_image_generator
import alert_engine

IST = ZoneInfo("Asia/Kolkata")


def get_now_ist() -> datetime:
    """Returns current datetime localized to Indian Standard Time (IST) as naive datetime for math."""
    return datetime.now(IST).replace(tzinfo=None)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("auto_75m_broadcaster")

# Standard NSE 75-minute candle close times (IST)
CANDLE_CLOSE_SCHEDULE = [
    (10, 30, "Candle 1 (09:15 - 10:30)"),
    (11, 45, "Candle 2 (10:30 - 11:45)"),
    (13, 0,  "Candle 3 (11:45 - 13:00)"),
    (14, 15, "Candle 4 (13:00 - 14:15)"),
    (15, 30, "Candle 5 (14:15 - 15:30)")
]

HISTORY_FILE = config.DATA_DIR / "broadcast_history.json"


def get_75m_schedule_status(now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Computes current status relative to NSE 75-minute candle closes in Indian Standard Time (IST).
    Returns countdown, next candle close time, active candle label, and market open status.
    """
    if now is None:
        now = get_now_ist()
    elif now.tzinfo is not None:
        now = now.astimezone(IST).replace(tzinfo=None)

    is_weekday = (now.weekday() < 5)  # Mon=0, Fri=4
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    is_market_hours = is_weekday and (market_open_time <= now <= market_close_time)

    # Find next upcoming candle close today
    next_close = None
    active_slot_label = ""
    candle_idx = 0

    if is_weekday and now < market_close_time:
        for i, (h, m, label) in enumerate(CANDLE_CLOSE_SCHEDULE):
            c_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now < c_time:
                next_close = c_time
                active_slot_label = label
                candle_idx = i + 1
                break

    # If all candles passed today or weekend, next close is next trading day 10:30
    if next_close is None:
        days_ahead = 1
        if now.weekday() == 4:  # Friday -> Monday
            days_ahead = 3
        elif now.weekday() == 5:  # Saturday -> Monday
            days_ahead = 2
        elif now.weekday() == 6:  # Sunday -> Monday
            days_ahead = 1
        elif now >= market_close_time:
            days_ahead = 1

        next_date = now + timedelta(days=days_ahead)
        while next_date.weekday() >= 5:  # Skip any weekend
            next_date += timedelta(days=1)
        next_close = next_date.replace(hour=10, minute=30, second=0, microsecond=0)
        active_slot_label = "Next Trading Session Candle 1 (10:30 IST)"
        candle_idx = 1

    secs_remaining = max(0, int((next_close - now).total_seconds()))
    hours = secs_remaining // 3600
    minutes = (secs_remaining % 3600) // 60
    seconds = secs_remaining % 60

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    else:
        time_str = f"{minutes}m {seconds}s"

    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "is_weekday": is_weekday,
        "is_market_hours": is_market_hours,
        "next_candle_time": next_close.strftime("%Y-%m-%d %H:%M:%S"),
        "next_candle_label": active_slot_label,
        "candle_idx": candle_idx,
        "seconds_remaining": secs_remaining,
        "time_remaining_str": time_str
    }


def get_target_equities(universe_choice: str = "All Database Equities") -> List[str]:
    """Returns candidate symbols based on universe selection."""
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


def record_broadcast_history(entry: Dict[str, Any]):
    """Appends broadcast run metadata to persistent history file."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.insert(0, entry)
        # Retain last 100 broadcasts
        history = history[:100]

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to record broadcast history: {e}")


def get_broadcast_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves previous broadcast session records."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            return history[:limit]
    except Exception as e:
        logger.error(f"Failed to read broadcast history: {e}")
        return []


def get_recent_quadrant_screenshots(limit: int = 30) -> List[Dict[str, Any]]:
    """Returns list of recent quadrant screenshot files ordered by modification time."""
    screen_dir = config.DATA_DIR / "screenshots"
    if not screen_dir.exists():
        return []

    items = []
    for p in screen_dir.glob("*.png"):
        try:
            stat = p.stat()
            # Extract symbol from filename (e.g. RELIANCE_quadrant_20260912_004524.png)
            parts = p.stem.split("_quadrant_")
            sym = parts[0] if parts else p.stem
            items.append({
                "symbol": sym,
                "path": str(p.resolve()),
                "filename": p.name,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
                "timestamp_epoch": stat.st_mtime
            })
        except Exception:
            continue

    items.sort(key=lambda x: x["timestamp_epoch"], reverse=True)
    return items[:limit]


def run_75m_waterfall_broadcast(
    universe: str = "All Database Equities",
    stage_filter: int = 4,
    channels: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    candle_label: Optional[str] = None,
    sync_first: bool = True,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes a complete 75-min Waterfall Scan, generates 4-quadrant screenshots
    for every qualifying stock, and dispatches multi-channel alerts.
    When sync_first is True, updates database with the latest 75m intraday candles from Upstox first.
    """
    start_time = get_now_ist()
    if candle_label is None:
        sched = get_75m_schedule_status(start_time)
        candle_label = sched.get("next_candle_label", "Intraday 75m Scan")

    if symbols is None or len(symbols) == 0:
        symbols = get_target_equities(universe)

    # 0. Sync fresh intraday 75m candle data from Upstox if requested
    if sync_first:
        try:
            import sync_75m_intraday
            logger.info(f"🔄 Ingesting live 75-minute candle data from Upstox for {len(symbols)} stocks (Universe: {universe})...")
            sync_res = sync_75m_intraday.sync_all_symbols_75m(
                symbols=symbols,
                universe=universe,
                progress_callback=progress_callback
            )
            logger.info(
                f"🔄 Upstox Ingestion Complete: {sync_res.get('synced_count', 0)}/{len(symbols)} stocks updated "
                f"({sync_res.get('total_75m_bars', 0)} 75m bars added) in {sync_res.get('elapsed_seconds', 0)}s."
            )
        except Exception as se:
            logger.warning(f"Live Upstox 75m sync skipped or encountered error: {se}")

    logger.info(f"⚡ Starting 75-Min Waterfall Scan across {len(symbols)} stocks (Universe: {universe}, Stage Filter: {stage_filter})...")


    # 1. Run Waterfall Scan
    scan_res = screener_engine.run_waterfall_scan(symbols, progress_callback=progress_callback)

    if stage_filter == 4:
        qualifying_df = scan_res.get("stage_4_full", pd.DataFrame())
    elif stage_filter == 3:
        qualifying_df = scan_res.get("stage_3_daily", pd.DataFrame())
    elif stage_filter == 2:
        qualifying_df = scan_res.get("stage_2_weekly", pd.DataFrame())
    elif stage_filter == 1:
        qualifying_df = scan_res.get("stage_1_monthly", pd.DataFrame())
    else:
        qualifying_df = scan_res.get("all_waterfall", pd.DataFrame())

    qualifying_count = len(qualifying_df)
    logger.info(f"Scan complete. Found {qualifying_count} qualifying stocks matching Stage {stage_filter}.")

    # Default channels if not specified
    if channels is None:
        cfg = alert_engine.get_channel_config()
        channels = [ch for ch in ["telegram", "whatsapp", "email", "webhook"] if cfg.get(ch, {}).get("enabled")]
        if not channels:
            channels = ["telegram", "email", "in_app"]

    dispatched_stocks = []

    if qualifying_count > 0:
        for _, row in qualifying_df.iterrows():
            sym = str(row["Symbol"])
            ltp = float(row.get("LTP", 0.0))
            ret1d = float(row.get("1D Return (%)", 0.0))
            stg = int(row.get("Stage", 4))
            stg_lbl = row.get("Waterfall Stage", f"Stage {stg}")

            logger.info(f"📸 Generating 1600x1200 4-Quadrant Screenshot (Light Theme) for {sym}...")
            img_path = quadrant_image_generator.generate_stock_quadrant(
                symbol=sym,
                stage_label=f"{stg_lbl} QUALIFIED",
                theme="light"
            )

            # Construct structured alert message
            headline = f"🏆 75-Min Waterfall Alert: {sym} qualified {stg_lbl}!"
            details = (
                f"LTP: ₹{ltp:,.2f} ({ret1d:+.2f}%)\n"
                f"• Monthly: {row.get('Monthly', 'N/A')} (RSI: {row.get('M_RSI')}, EMA5: ₹{row.get('M_EMA5')})\n"
                f"• Weekly: {row.get('Weekly', 'N/A')} (EMA20: ₹{row.get('W_EMA20')})\n"
                f"• Daily: {row.get('Daily', 'N/A')} (EMA20: ₹{row.get('D_EMA20')})\n"
                f"• 75-Min: {row.get('75-Min', 'N/A')} (RSI9: {row.get('75m_RSI')}, EMA20: ₹{row.get('75m_EMA20')})"
            )

            # Dispatch with screenshot
            logger.info(f"🚀 Dispatching alert with photo attachment across channels: {channels}...")
            deliv = alert_engine.dispatch_alert(
                symbol=sym,
                alert_type="WATERFALL_75M",
                trigger_price=ltp,
                headline=headline,
                details=details,
                selected_channels=channels,
                screenshot_path=img_path
            )

            # Log to SQLite alert audit
            alert_engine.log_alert_trigger(
                alert_id=0,
                symbol=sym,
                alert_type="WATERFALL_75M",
                trigger_price=ltp,
                message=f"{headline} | {details}",
                delivery_results=deliv
            )

            dispatched_stocks.append({
                "symbol": sym,
                "ltp": ltp,
                "change_pct": ret1d,
                "stage": stg,
                "stage_label": stg_lbl,
                "screenshot_path": img_path,
                "delivery": {k: {"success": v[0], "message": v[1]} for k, v in deliv.items()}
            })

    elapsed_secs = max(0.0, (get_now_ist() - start_time).total_seconds())

    broadcast_record = {
        "broadcast_id": f"bc_{int(start_time.timestamp())}",
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "candle_slot": candle_label,
        "universe": universe,
        "stage_filter": stage_filter,
        "scanned_count": len(symbols),
        "qualifying_count": qualifying_count,
        "elapsed_seconds": round(elapsed_secs, 2),
        "stocks": dispatched_stocks,
        "channels": channels
    }

    record_broadcast_history(broadcast_record)
    logger.info(f"✅ Broadcast cycle completed in {elapsed_secs:.1f}s. {len(dispatched_stocks)} stock alerts sent.")

    return broadcast_record


def run_daemon(
    universe: str = "All Database Equities",
    stage_filter: int = 4,
    channels: Optional[List[str]] = None,
    sync_first: bool = True,
    poll_interval: int = 10
):
    """
    Continuous background daemon that monitors the system clock and executes
    the Waterfall Scan & Screenshot Broadcast immediately at every 75-min candle close.
    Automatically updates database with fresh 75-minute candle data from Upstox.
    """
    logger.info("🛰️ Starting 75-Minute Intraday Waterfall Broadcaster Daemon...")
    logger.info(f"Schedule: 10:30, 11:45, 13:00, 14:15, 15:30 IST (Mon-Fri)")
    logger.info(f"Universe: {universe} | Stage Filter: Stage {stage_filter} | Auto-Sync: {sync_first}")

    last_triggered_slot = None  # (YYYY-MM-DD, hour, minute)

    while True:
        try:
            now = get_now_ist()
            is_weekday = (now.weekday() < 5)

            if is_weekday:
                cur_date_str = now.strftime("%Y-%m-%d")
                for h, m, label in CANDLE_CLOSE_SCHEDULE:
                    # Check if current time matches scheduled close (within window)
                    if (now.hour == h) and (now.minute == m):
                        slot_key = (cur_date_str, h, m)
                        if last_triggered_slot != slot_key:
                            logger.info(f"🔔 TRIGGER EVENT: Reached 75-Min Candle Close: {label}")
                            last_triggered_slot = slot_key
                            run_75m_waterfall_broadcast(
                                universe=universe,
                                stage_filter=stage_filter,
                                channels=channels,
                                candle_label=label,
                                sync_first=sync_first
                            )
                            break

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Broadcaster daemon stopped by user.")
            break
        except Exception as err:
            logger.error(f"Daemon loop exception: {err}", exc_info=True)
            time.sleep(poll_interval)


# =========================================================================
# CLI COMMAND LINE INTERFACE
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="75-Min Intraday Waterfall Scan & Screenshot Broadcaster")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon waiting for 75m candle closes")
    parser.add_argument("--once", action="store_true", help="Execute scan and broadcast immediately once and exit")
    parser.add_argument("--universe", type=str, default="All Database Equities", help="Universe: 'All Database Equities', 'Nifty 50', 'Nifty 100', 'Nifty 500'")
    parser.add_argument("--stage", type=int, default=4, help="Stage filter (default 4 for Full Alignment Only)")
    parser.add_argument("--no-sync", action="store_true", help="Skip live Upstox intraday data sync before scanning")
    parser.add_argument("--test-symbol", type=str, help="Generate screenshot and dispatch test alert for a single symbol immediately")

    args = parser.parse_args()

    if args.test_symbol:
        sym = args.test_symbol.upper()
        print(f"Testing quadrant screenshot & alert broadcast for: {sym}")
        img = quadrant_image_generator.generate_stock_quadrant(sym)
        print(f"Generated screenshot: {img}")
        if img and os.path.exists(img):
            print(f"File size: {os.path.getsize(img)} bytes")
            res = alert_engine.dispatch_alert(
                symbol=sym,
                alert_type="WATERFALL_75M",
                trigger_price=100.0,
                headline=f"Test 75-Min Quadrant Broadcast for {sym}",
                details="Testing 1600x1200 composite image delivery.",
                selected_channels=["telegram", "email", "in_app"],
                screenshot_path=img
            )
            print("Dispatch delivery result:", res)
    elif args.daemon:
        run_daemon(universe=args.universe, stage_filter=args.stage, sync_first=(not args.no_sync))
    elif args.once:
        run_75m_waterfall_broadcast(universe=args.universe, stage_filter=args.stage, sync_first=(not args.no_sync))
    else:
        parser.print_help()

