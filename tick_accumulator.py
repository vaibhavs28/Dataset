"""
tick_accumulator.py
-------------------
Local High-Speed Python Tick Accumulator for Real-Time Protobuf Feeds.
Part of the Chartink-Style Real-Time Architecture:
  [ Upstox Protobuf Stream ] ──► (Real-Time Websocket Ticks)
                                            │
                                            ▼
                             [ Local Python Accumulator ]
                              Groups ticks into 1-Min bins
                              Calculates Close, RSI, SMA20, EMA20
                                            │
                                            ▼
                             [ DuckDB Hot RAM Storage ]
                              Overwrites `latest_snapshot` (5ms)
"""

import time
import logging
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import duckdb

import config
import hot_intraday

logger = logging.getLogger("tick_accumulator")
IST = ZoneInfo("Asia/Kolkata")


class SymbolState:
    """Maintains in-memory rolling state and current 1-minute bin for a symbol."""
    __slots__ = (
        "symbol", "current_minute_ts", "open", "high", "low", "close",
        "volume_in_minute", "last_vtt", "prev_close",
        "close_history", "current_rsi", "current_sma20", "current_ema20",
        "last_tick_time"
    )

    def __init__(self, symbol: str, prev_close: float = 0.0):
        self.symbol = symbol
        self.current_minute_ts: Optional[int] = None  # epoch second floored to 60s
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0
        self.volume_in_minute: int = 0
        self.last_vtt: int = 0  # last seen volume traded today
        self.prev_close: float = prev_close

        # Rolling history for indicator calculation
        self.close_history: deque = deque(maxlen=60)  # last 60 minutes
        self.current_rsi: float = 50.0
        self.current_sma20: float = 0.0
        self.current_ema20: float = 0.0
        self.last_tick_time: float = time.time()


class TickAccumulator:
    """
    Thread-safe in-memory tick aggregator.
    1. Bins ticks into 1-minute OHLCV candles.
    2. Computes rolling RSI-14, SMA-20, EMA-20.
    3. Flushes completed 1-minute bars to hot_intraday.db.
    4. Periodically updates `latest_snapshot` in DuckDB RAM (in ~5ms).
    """

    def __init__(self, flush_interval_secs: float = 3.0):
        self.states: Dict[str, SymbolState] = {}
        self.lock = threading.RLock()
        self.completed_bars: List[Dict[str, Any]] = []
        self.flush_interval_secs = flush_interval_secs
        self.last_snapshot_flush = time.time()
        self.total_ticks_processed = 0

    def get_or_create_state(self, symbol: str, prev_close: float = 0.0) -> SymbolState:
        with self.lock:
            state = self.states.get(symbol)
            if state is None:
                state = SymbolState(symbol, prev_close=prev_close)
                self.states[symbol] = state
            return state

    def process_tick(
        self,
        symbol: str,
        ltp: float,
        volume_delta: int = 0,
        vtt: Optional[int] = None,
        tick_time: Optional[float] = None
    ) -> None:
        """
        Ingests a single live tick and updates the active 1-minute bin.
        Thread-safe, sub-microsecond processing time.
        """
        if ltp <= 0:
            return

        now_t = tick_time if tick_time is not None else time.time()
        minute_epoch = int(now_t // 60) * 60

        with self.lock:
            self.total_ticks_processed += 1
            state = self.get_or_create_state(symbol)
            state.last_tick_time = now_t

            # Calculate incremental volume from cumulative VTT if provided
            vol = volume_delta
            if vtt is not None and vtt > 0:
                if state.last_vtt > 0 and vtt >= state.last_vtt:
                    vol = vtt - state.last_vtt
                state.last_vtt = vtt

            # Check if minute has turned
            if state.current_minute_ts is None:
                # First tick for this symbol
                state.current_minute_ts = minute_epoch
                state.open = ltp
                state.high = ltp
                state.low = ltp
                state.close = ltp
                state.volume_in_minute = vol
            elif state.current_minute_ts == minute_epoch:
                # Same minute: update high, low, close, volume
                if ltp > state.high:
                    state.high = ltp
                if ltp < state.low:
                    state.low = ltp
                state.close = ltp
                state.volume_in_minute += vol
            else:
                # Minute has completed! Emit completed bar
                bar_dt = datetime.fromtimestamp(state.current_minute_ts, tz=IST)
                self.completed_bars.append({
                    "symbol": symbol,
                    "timestamp": bar_dt,
                    "open": state.open,
                    "high": state.high,
                    "low": state.low,
                    "close": state.close,
                    "volume": state.volume_in_minute
                })

                # Update rolling indicator history
                state.close_history.append(state.close)
                self._update_indicators(state)

                # Start new minute bin
                state.current_minute_ts = minute_epoch
                state.open = ltp
                state.high = ltp
                state.low = ltp
                state.close = ltp
                state.volume_in_minute = vol

            # Check periodic snapshot flush
            if now_t - self.last_snapshot_flush >= self.flush_interval_secs:
                self.flush_to_duckdb()

    def _update_indicators(self, state: SymbolState) -> None:
        """Fast incremental update for SMA20, EMA20, RSI14 using rolling close history."""
        hist = state.close_history
        n = len(hist)
        if n == 0:
            return

        # SMA20
        span_sma = min(20, n)
        state.current_sma20 = sum(list(hist)[-span_sma:]) / span_sma

        # EMA20
        if state.current_ema20 == 0.0:
            state.current_ema20 = state.close
        else:
            k = 2.0 / (20.0 + 1.0)
            state.current_ema20 = (state.close * k) + (state.current_ema20 * (1.0 - k))

        # RSI14 (simplified Wilder calculation on rolling differences)
        if n >= 15:
            arr = np.array(list(hist)[-15:])
            diffs = np.diff(arr)
            gains = np.maximum(diffs, 0.0)
            losses = np.maximum(-diffs, 0.0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                state.current_rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                state.current_rsi = 100.0 - (100.0 / (1.0 + rs))

    def flush_to_duckdb(self) -> None:
        """
        1. Appends completed 1m candles into `intraday_1m` in hot_intraday.db.
        2. Overwrites `latest_snapshot` table in hot_intraday.db (< 5ms).
        """
        self.last_snapshot_flush = time.time()

        # Step 1: Flush completed bars to intraday_1m
        bars_to_flush = []
        with self.lock:
            if self.completed_bars:
                bars_to_flush = list(self.completed_bars)
                self.completed_bars.clear()

        if bars_to_flush:
            df_bars = pd.DataFrame(bars_to_flush)
            hot_intraday.append_1m_batch(df_bars)

        # Step 2: Overwrite latest_snapshot in DuckDB RAM
        snapshot_rows = []
        now_dt = datetime.now(IST)
        with self.lock:
            for sym, st in self.states.items():
                if st.close <= 0:
                    continue
                chg_pct = 0.0
                if st.prev_close > 0:
                    chg_pct = round(((st.close - st.prev_close) / st.prev_close) * 100.0, 2)

                snapshot_rows.append({
                    "symbol": sym,
                    "ltp": st.close,
                    "open": st.open,
                    "high": st.high,
                    "low": st.low,
                    "close": st.close,
                    "volume": st.volume_in_minute,
                    "rsi_14": round(st.current_rsi, 2),
                    "sma_20": round(st.current_sma20, 2),
                    "ema_20": round(st.current_ema20, 2),
                    "change_pct": chg_pct,
                    "last_updated": now_dt
                })

        if snapshot_rows:
            df_snap = pd.DataFrame(snapshot_rows)
            _overwrite_latest_snapshot(df_snap)


def _overwrite_latest_snapshot(df: pd.DataFrame) -> None:
    """Overwrites `latest_snapshot` table in hot_intraday.db in < 5ms."""
    with hot_intraday._hot_lock:
        conn = hot_intraday._get_conn(read_only=False)
        try:
            conn.register("snap_view", df)
            conn.execute("""
                CREATE OR REPLACE TABLE latest_snapshot AS
                SELECT 
                    symbol, ltp, open, high, low, close, volume,
                    rsi_14, sma_20, ema_20, change_pct, last_updated
                FROM snap_view;
            """)
            conn.unregister("snap_view")
        except Exception as e:
            logger.debug(f"Error overwriting latest_snapshot: {e}")
        finally:
            conn.close()


def query_latest_snapshot(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Returns instant live snapshot DataFrame from DuckDB memory in < 5ms.
    Used by Scanners, Alerts, and Streamlit for real-time monitoring.
    """
    with hot_intraday._hot_lock:
        conn = hot_intraday._get_conn(read_only=True)
        try:
            # Check if table exists
            has_tbl = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name='latest_snapshot' LIMIT 1;"
            ).fetchone()
            if not has_tbl:
                return pd.DataFrame()

            if symbols:
                clean_syms = list({s.upper().strip().replace("-EQ", "").replace(".NS", "") for s in symbols})
                placeholders = ", ".join(["?"] * len(clean_syms))
                df = conn.execute(
                    f"SELECT * FROM latest_snapshot WHERE symbol IN ({placeholders}) ORDER BY volume DESC;",
                    clean_syms
                ).df()
            else:
                df = conn.execute("SELECT * FROM latest_snapshot ORDER BY volume DESC;").df()
            return df
        except Exception as e:
            logger.debug(f"Error reading latest_snapshot: {e}")
            return pd.DataFrame()
        finally:
            conn.close()


# Global singleton accumulator
global_accumulator = TickAccumulator(flush_interval_secs=2.0)
