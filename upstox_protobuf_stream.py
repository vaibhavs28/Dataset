"""
upstox_protobuf_stream.py
-------------------------
Upstox Protobuf Market Data WebSocket Feed Streamer.
Connects to Upstox WebSocket v3 API, decodes Google Protocol Buffer (Protobuf) ticks,
and dispatches to the local Python TickAccumulator.

Pipeline:
  [ Upstox Protobuf Stream ] ──► (Real-Time Websocket Ticks)
                                            │
                                            ▼
                             [ Local Python Accumulator ]
                              Groups ticks into 1-Min bins
                              Calculates Close, RSI, SMA20
                                            │
                                            ▼
                             [ DuckDB Hot RAM Storage ]
                              Overwrites `latest_snapshot` (5ms)
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from typing import List, Dict, Optional, Any
from zoneinfo import ZoneInfo
from datetime import datetime

import upstox_client
from upstox_client.feeder import MarketDataStreamerV3

import config
import database
import auto_nifty500_updater
from tick_accumulator import global_accumulator

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("upstox_protobuf_stream")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

STATUS_FILE = config.DATA_DIR / "websocket_stream_status.json"


def get_default_instrument_keys(universe: str = "Nifty 500") -> Tuple[List[str], Dict[str, str]]:
    """
    Returns a list of Upstox instrument keys (e.g. 'NSE_EQ|INE002A01018')
    and a lookup dict mapping instrument_key -> clean symbol.
    """
    if universe.lower() == "nifty 50":
        target_syms = config.NIFTY_50_SYMBOLS
    else:
        target_syms = auto_nifty500_updater.get_nifty_500_symbols()

    clean_target = {s.upper().strip().replace("-EQ", "").replace(".NS", "") for s in target_syms}

    all_inst = database.get_all_instruments(exchange="NSE_EQ")
    key_to_sym: Dict[str, str] = {}
    keys: List[str] = []

    for inst in all_inst:
        sym = inst.get("trading_symbol", "").replace("-EQ", "")
        ikey = inst.get("instrument_key", "")
        if sym in clean_target and ikey:
            key_to_sym[ikey] = sym
            keys.append(ikey)

    # Fallback if instruments table isn't populated
    if not keys:
        for s in clean_target:
            ikey = f"NSE_EQ|{s}"
            key_to_sym[ikey] = s
            keys.append(ikey)

    return keys, key_to_sym


class UpstoxProtobufStreamer:
    """
    Manages the Upstox MarketDataStreamerV3 Protobuf WebSocket connection.
    Feeds decoded live ticks into the global TickAccumulator.
    """

    def __init__(self, universe: str = "Nifty 500", mode: str = "full"):
        self.universe = universe
        self.mode = mode
        self.streamer: Optional[MarketDataStreamerV3] = None
        self.keys: List[str] = []
        self.key_to_sym: Dict[str, str] = {}
        self.is_connected = False
        self.tick_count = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def update_status(self, status: str, message: str = "") -> None:
        payload = {
            "status": status,
            "universe": self.universe,
            "subscribed_keys_count": len(self.keys),
            "total_ticks_received": self.tick_count,
            "is_connected": self.is_connected,
            "message": message,
            "last_updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        }
        try:
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def on_open(self) -> None:
        self.is_connected = True
        logger.info("🟢 Upstox Protobuf WebSocket connected successfully!")
        self.update_status("CONNECTED", "WebSocket connection open and streaming.")

    def on_close(self, *args) -> None:
        self.is_connected = False
        logger.warning(f"🔴 Upstox Protobuf WebSocket disconnected. Event details: {args}")
        self.update_status("DISCONNECTED", "WebSocket connection closed.")

    def on_error(self, error: Any) -> None:
        logger.error(f"⚠️ Upstox Protobuf WebSocket Error: {error}")
        self.update_status("ERROR", str(error))

    def on_message(self, message: Dict[str, Any]) -> None:
        """
        Dispatches incoming decoded protobuf ticks into the TickAccumulator.
        """
        feeds = message.get("feeds", {})
        if not feeds:
            return

        now_t = time.time()
        with self._lock:
            self.tick_count += len(feeds)

        for ikey, feed_data in feeds.items():
            sym = self.key_to_sym.get(ikey)
            if not sym:
                # Extract symbol from ikey format 'NSE_EQ|INE...' or 'NSE_EQ|RELIANCE'
                sym = ikey.split("|")[-1]

            ltp = 0.0
            vtt = 0

            # 1. FullFeed structure (Mode = 'full')
            ff = feed_data.get("fullFeed", {})
            if ff:
                market_ff = ff.get("marketFF", {})
                ltpc = market_ff.get("ltpc", {})
                ltp = float(ltpc.get("ltp", 0.0))
                vtt = int(market_ff.get("vtt", 0))

            # 2. LTPC structure (Mode = 'ltpc')
            if ltp == 0.0:
                ltpc = feed_data.get("ltpc", {})
                ltp = float(ltpc.get("ltp", 0.0))

            if ltp > 0:
                global_accumulator.process_tick(
                    symbol=sym,
                    ltp=ltp,
                    vtt=vtt if vtt > 0 else None,
                    tick_time=now_t
                )

    def start(self) -> None:
        """Initializes and runs the Upstox MarketDataStreamerV3."""
        token = config.UPSTOX_ACCESS_TOKEN
        if not token:
            logger.error("UPSTOX_ACCESS_TOKEN is missing. Please authenticate Upstox first.")
            self.update_status("UNAUTHORIZED", "Missing UPSTOX_ACCESS_TOKEN")
            return

        api_client = upstox_client.ApiClient()
        api_client.configuration.access_token = token

        self.keys, self.key_to_sym = get_default_instrument_keys(self.universe)
        logger.info(f"📡 Subscribing to {len(self.keys)} instruments for universe '{self.universe}' (Mode: {self.mode})...")

        # Upstox streamer v3 subscribes up to subscription limit
        self.streamer = MarketDataStreamerV3(
            api_client=api_client,
            instrumentKeys=self.keys,
            mode=self.mode
        )

        # Wire event handlers
        self.streamer.on(MarketDataStreamerV3.Event["OPEN"], self.on_open)
        self.streamer.on(MarketDataStreamerV3.Event["CLOSE"], self.on_close)
        self.streamer.on(MarketDataStreamerV3.Event["ERROR"], self.on_error)
        self.streamer.on(MarketDataStreamerV3.Event["MESSAGE"], self.on_message)

        # Connect with auto-reconnect
        logger.info("Connecting to Upstox Market Data Protobuf WebSocket...")
        self.streamer.auto_reconnect(enable=True, interval=5, retryCount=50)
        self.streamer.connect()


def run_stream_daemon(universe: str = "Nifty 500"):
    """Runs the streamer in the foreground with periodic status heartbeats."""
    streamer = UpstoxProtobufStreamer(universe=universe, mode="full")
    streamer.start()

    logger.info("Websocket client running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            stats = streamer.tick_count
            accum_ticks = global_accumulator.total_ticks_processed
            logger.info(f"📊 WebSocket Heartbeat: {stats} ticks received ({accum_ticks} accumulator processed).")
            streamer.update_status("STREAMING")
    except KeyboardInterrupt:
        logger.info("Shutting down WebSocket streamer...")
        if streamer.streamer:
            streamer.streamer.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upstox Protobuf WebSocket Streamer")
    parser.add_argument("--universe", type=str, default="Nifty 500", help="Universe: Nifty 500 or Nifty 50")
    args = parser.parse_args()

    run_stream_daemon(universe=args.universe)
