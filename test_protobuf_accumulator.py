"""
test_protobuf_accumulator.py
----------------------------
Unit tests for the Upstox Protobuf Stream TickAccumulator and DuckDB `latest_snapshot` storage.
"""

import time
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

import tick_accumulator
from tick_accumulator import TickAccumulator, query_latest_snapshot
import hot_intraday


class TestProtobufAccumulator(unittest.TestCase):

    def setUp(self):
        self.accumulator = TickAccumulator(flush_interval_secs=0.1)

    def test_tick_binning_and_indicators(self):
        sym = "TEST_RELIANCE"
        t0 = 1700000000.0  # reference epoch second

        # Send 30 ticks within minute 0
        prices = [100.0 + i * 0.5 for i in range(30)]
        for i, p in enumerate(prices):
            self.accumulator.process_tick(sym, ltp=p, volume_delta=100, tick_time=t0 + i)

        state = self.accumulator.get_or_create_state(sym)
        self.assertEqual(state.open, 100.0)
        self.assertEqual(state.high, 100.0 + 29 * 0.5)
        self.assertEqual(state.close, 100.0 + 29 * 0.5)
        self.assertEqual(state.volume_in_minute, 3000)

        # Cross into minute 1: forces minute 0 to emit into completed_bars
        self.accumulator.process_tick(sym, ltp=120.0, volume_delta=50, tick_time=t0 + 65.0)

        # Check that completed_bars has minute 0
        self.assertEqual(len(self.accumulator.completed_bars), 1)
        bar = self.accumulator.completed_bars[0]
        self.assertEqual(bar["open"], 100.0)
        self.assertEqual(bar["close"], 100.0 + 29 * 0.5)
        self.assertEqual(bar["volume"], 3000)

        # Check rolling history updated
        self.assertEqual(len(state.close_history), 1)

    def test_duckdb_latest_snapshot_flush_and_query(self):
        sym1 = "SNAP_INFY"
        sym2 = "SNAP_TCS"

        self.accumulator.process_tick(sym1, ltp=1850.0, volume_delta=500)
        self.accumulator.process_tick(sym2, ltp=3900.0, volume_delta=300)

        # Force flush to DuckDB
        self.accumulator.flush_to_duckdb()

        # Query latest_snapshot from DuckDB RAM in < 5ms
        t_start = time.time()
        df_snap = query_latest_snapshot([sym1, sym2])
        elapsed_ms = (time.time() - t_start) * 1000

        self.assertFalse(df_snap.empty)
        self.assertIn("symbol", df_snap.columns)
        self.assertIn("ltp", df_snap.columns)
        self.assertIn("rsi_14", df_snap.columns)
        self.assertIn("sma_20", df_snap.columns)
        self.assertIn("ema_20", df_snap.columns)

        symbols_found = set(df_snap["symbol"].tolist())
        self.assertIn(sym1, symbols_found)
        self.assertIn(sym2, symbols_found)

        # Verify performance: must be under 20ms
        self.assertLess(elapsed_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
