"""
auto_15m_broadcaster.py
-----------------------
Automated 15-Minute Chartink Intraday Scan (#19122704) & Screenshot Broadcaster.
Dynamically resamples live 1-minute bars directly from `hot_intraday.db` (< 30ms).

Operates on every 15-minute candle close during NSE market hours (09:15 - 15:30 IST):
  09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00, 11:15, 11:30, 11:45,
  12:00, 12:15, 12:30, 12:45, 13:00, 13:15, 13:30, 13:45, 14:00, 14:15,
  14:30, 14:45, 15:00, 15:15, 15:30 IST.

When triggered:
1. Dynamically resamples 15m and 75m candles from `hot_intraday.db` in RAM.
2. Runs the 5-stage sequential breakdown scan (#19122704):
     Stage 1: Monthly Pass
     Stage 2: Weekly Pass
     Stage 3: Daily Pass
     Stage 4: 75-Min Breakdown Trigger (⚡)
     Stage 5: 15-Min Precision Entry Trigger (🚨 Full Signal)
3. Generates 1600x1200 4-quadrant chart screenshots.
4. Broadcasts alerts to Telegram, WhatsApp, Email, and Webhooks.
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
import concurrent.futures

import pandas as pd

import config
import database
import hot_intraday
import screener_engine
import quadrant_image_generator
import alert_engine

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("auto_15m_broadcaster")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

HISTORY_FILE = config.DATA_DIR / "broadcast_history_15m.json"


def get_now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def is_market_hours_ist(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = get_now_ist()
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_next_15m_slot(now: Optional[datetime] = None) -> Tuple[datetime, str]:
    """Returns the next upcoming 15-minute close today, or tomorrow 09:30."""
    if now is None:
        now = get_now_ist()

    # Next 15m boundary
    minute = (now.minute // 15 + 1) * 15
    if minute >= 60:
        nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        nxt = now.replace(minute=minute, second=0, microsecond=0)

    # If before market open today (09:30)
    mkt_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if nxt < mkt_start:
        nxt = mkt_start

    # If after market close (15:30), roll to next trading day 09:30
    mkt_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if nxt > mkt_end or now.weekday() >= 5:
        days_ahead = 1
        candidate = now + timedelta(days=days_ahead)
        while candidate.weekday() >= 5:
            days_ahead += 1
            candidate = now + timedelta(days=days_ahead)
        nxt = candidate.replace(hour=9, minute=30, second=0, microsecond=0)

    label = f"{nxt.strftime('%H:%M')} IST Close"
    return nxt, label


def execute_15m_broadcast_cycle(
    universe: str = "Nifty 500",
    stage_filter: int = 4,
    channels: Optional[List[str]] = None,
    force: bool = False,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes a single 15-minute broadcast cycle for Chartink Scan 19122704.
    stage_filter: 4 = Stage 4 (M+W+D+75m) + Stage 5 (15m Precision Entry), 5 = Only Stage 5.
    """
    start_time = get_now_ist()
    now_dt = datetime.now(IST)

    if not force and not is_market_hours_ist(start_time):
        logger.info(f"Outside market hours ({start_time.strftime('%H:%M:%S IST')}). Skipping 15m broadcast.")
        return {"status": "SKIPPED", "reason": "OUTSIDE_MARKET_HOURS"}

    logger.info(f"⚡ Starting 15-Minute Chartink #19122704 Scan (Stage Filter: {stage_filter}, Universe: {universe})...")

    # 1. Resolve symbols
    import auto_nifty500_updater
    u_low = universe.lower()
    if "500" in u_low:
        symbols = auto_nifty500_updater.get_nifty_500_symbols()
    elif "50" in u_low:
        symbols = config.NIFTY_50_SYMBOLS
    else:
        symbols = database.get_all_symbols()

    if progress_callback:
        try:
            progress_callback(0, len(symbols), f"Resolving {len(symbols)} symbols...")
        except Exception:
            pass

    # 2. Check live hot_intraday.db
    if hot_intraday.has_hot_data():
        stats = hot_intraday.get_hot_stats()
        logger.info(f"⚡ Live feed active: hot_intraday.db ({stats.get('candle_count', 0)} bars) — dynamic 15m+75m resample active.")
    else:
        logger.info("ℹ️ hot_intraday.db idle. Falling back to DuckDB store.")

    # 3. Batch load daily candles
    batch_daily = database.get_batch_candles_df(symbols)

    # 4. Augment with live market ticks if streaming
    if hot_intraday.has_hot_data():
        try:
            live_bars = hot_intraday.get_batch_today_daily_bars(symbols)
            for sym_k, bar in live_bars.items():
                if sym_k in batch_daily and batch_daily[sym_k] is not None and not batch_daily[sym_k].empty:
                    orig_df = batch_daily[sym_k]
                    last_date_str = str(orig_df.index[-1])[:10]
                    today_date_str = str(bar["timestamp"])[:10]
                    if last_date_str != today_date_str:
                        live_row = pd.DataFrame([{
                            "open": bar["open"],
                            "high": bar["high"],
                            "low": bar["low"],
                            "close": bar["close"],
                            "volume": bar["volume"]
                        }], index=[pd.to_datetime(today_date_str)])
                        batch_daily[sym_k] = pd.concat([orig_df, live_row])
                    else:
                        batch_daily[sym_k].iloc[-1, batch_daily[sym_k].columns.get_loc("close")] = bar["close"]
                        batch_daily[sym_k].iloc[-1, batch_daily[sym_k].columns.get_loc("high")] = max(batch_daily[sym_k].iloc[-1]["high"], bar["high"])
                        batch_daily[sym_k].iloc[-1, batch_daily[sym_k].columns.get_loc("low")] = min(batch_daily[sym_k].iloc[-1]["low"], bar["low"])
                        batch_daily[sym_k].iloc[-1, batch_daily[sym_k].columns.get_loc("volume")] = bar["volume"]
        except Exception as e:
            logger.warning(f"Error augmenting batch_daily with live bars: {e}")

    if progress_callback:
        try:
            progress_callback(0, len(symbols), f"Ready. Starting evaluation across {len(symbols)} stocks...")
        except Exception:
            pass

    qualifying_stocks = []
    logger.info(f"Evaluating 5-stage funnel across {len(symbols)} stocks...")

    def _evaluate_sym(sym: str) -> Optional[Dict[str, Any]]:
        clean = sym.upper().strip().replace("-EQ", "").replace(".NS", "")
        d_df = batch_daily.get(clean)
        if d_df is None or d_df.empty:
            return None
        res = screener_engine.evaluate_intraday_scan_19122704(clean, d_df)
        if res is not None:
            stg = res.get("Stage", 0)
            if stg >= stage_filter:
                return res
        return None

    completed_count = 0
    total_symbols = len(symbols)
    eval_workers = min(8, max(2, os.cpu_count() or 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
        futures = {executor.submit(_evaluate_sym, s): s for s in symbols}
        for fut in concurrent.futures.as_completed(futures):
            sym = futures[fut]
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, total_symbols, sym)
                except Exception:
                    pass
            res = fut.result()
            if res is not None:
                qualifying_stocks.append(res)

    # Sort so highest stage (Stage 5) comes first, then by largest price move
    qualifying_stocks.sort(
        key=lambda x: (x.get("Stage", 0), abs(float(x.get("1D Return (%)", 0.0)))),
        reverse=True
    )

    logger.info(f"Scan complete. Found {len(qualifying_stocks)} stocks meeting Stage {stage_filter}+.")

    # 4. Dispatch alerts with screenshots (capped to top 5 to prevent OOM memory saturation)
    MAX_SCREENSHOTS = 5
    import gc

    if channels is None:
        cfg = alert_engine.get_channel_config()
        channels = [ch for ch in ["telegram", "whatsapp", "email", "webhook"] if cfg.get(ch, {}).get("enabled")]
        if not channels:
            channels = ["telegram", "email", "in_app"]

    dispatched = []
    ch_url = "https://chartink.com/screener/intraday-scan-19122704"

    for idx, row in enumerate(qualifying_stocks):
        sym = str(row["Symbol"])
        ltp = float(row.get("LTP", 0.0))
        stg = int(row.get("Stage", 5))
        stg_lbl = row.get("Waterfall Stage", f"Stage {stg}")
        ret1d = float(row.get("1D Return (%)", 0.0))

        headline = f"🚨 Chartink 15m Signal: {sym} qualified {stg_lbl} @ ₹{ltp:,.2f}"
        details = (
            f"<b>LTP:</b> ₹{ltp:,.2f} ({ret1d:+.2f}%)\n"
            f"<b>Strategy:</b> <a href='{ch_url}'>Chartink Intraday Scan #19122704</a>\n"
            f"<b>Stage:</b> {stg_lbl}\n"
            f"• <b>Monthly:</b> {row.get('Monthly', 'PASS')}\n"
            f"• <b>Weekly:</b> {row.get('Weekly', 'PASS')}\n"
            f"• <b>Daily:</b> {row.get('Daily', 'PASS')}\n"
            f"• <b>75-Min:</b> {row.get('75-Min', 'PASS')}\n"
            f"• <b>15-Min:</b> {row.get('15-Min', 'PASS')}"
        )

        img_path = None
        if idx < MAX_SCREENSHOTS:
            logger.info(f"📸 Generating 4-Quadrant Screenshot ({idx+1}/{min(MAX_SCREENSHOTS, len(qualifying_stocks))}) for {sym} ({stg_lbl})...")
            try:
                img_path = quadrant_image_generator.generate_stock_quadrant(
                    symbol=sym,
                    stage_label=f"{stg_lbl} QUALIFIED",
                    theme="light"
                )
            except Exception as e:
                logger.error(f"Error generating quad screenshot for {sym}: {e}")
            gc.collect()

        deliv = alert_engine.dispatch_alert(
            symbol=sym,
            alert_type="INTRADAY_SCAN_19122704",
            trigger_price=ltp,
            headline=headline,
            details=details,
            selected_channels=channels,
            screenshot_path=img_path,
            ignore_market_hours=force
        )

        dispatched.append({
            "symbol": sym,
            "ltp": ltp,
            "stage": stg,
            "stage_label": stg_lbl,
            "delivery": deliv
        })

    elapsed = round((get_now_ist() - start_time).total_seconds(), 2)
    logger.info(f"✅ 15-Min broadcast cycle finished in {elapsed}s. {len(dispatched)} alerts sent.")

    return {
        "status": "COMPLETED",
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "qualifying_count": len(qualifying_stocks),
        "alerts_dispatched": len(dispatched),
        "elapsed_seconds": elapsed,
        "qualifying_stocks": qualifying_stocks,
        "scanned_count": len(symbols)
    }


def run_daemon(universe: str = "Nifty 500", stage_filter: int = 5):
    """
    Continuous daemon running every 15 minutes during IST market hours.
    Fires on every 15-minute slot (:00, :15, :30, :45).
    """
    logger.info("🛰️ Starting Chartink Intraday #19122704 15-Minute Broadcaster Daemon...")
    logger.info(f"Target Universe: {universe} | Minimum Stage Filter: Stage {stage_filter}")
    logger.info("Operating Window: 09:15 AM - 03:30 PM IST (Mon-Fri)")

    while True:
        try:
            now = get_now_ist()
            nxt, label = get_next_15m_slot(now)
            sleep_secs = max(5, int((nxt - now).total_seconds()))

            logger.info(f"⏳ Next 15-Min Broadcast at {nxt.strftime('%H:%M:%S IST')} (~{sleep_secs // 60}m {sleep_secs % 60}s). Sleeping...")
            time.sleep(sleep_secs)

            # Fire cycle
            execute_15m_broadcast_cycle(universe=universe, stage_filter=stage_filter)

        except KeyboardInterrupt:
            logger.info("Stopped 15m broadcaster daemon.")
            break
        except Exception as e:
            logger.error(f"Error in 15m broadcaster loop: {e}", exc_info=True)
            time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="15-Minute Chartink Intraday Broadcaster")
    parser.add_argument("--daemon", action="store_true", help="Run continuous 15-min daemon")
    parser.add_argument("--universe", type=str, default="Nifty 500", help="Target universe")
    parser.add_argument("--stage", type=int, default=5, help="Stage filter (5=Full Signal, 4=Stage 4+)")
    parser.add_argument("--force", action="store_true", help="Force run outside market hours")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(universe=args.universe, stage_filter=args.stage)
    else:
        res = execute_15m_broadcast_cycle(universe=args.universe, stage_filter=args.stage, force=args.force)
        print(f"Cycle Result: {res}")
