"""
test_hot_intraday.py
--------------------
Unit tests for hot_intraday.db in-memory/local storage and dynamic resampling.
"""

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

import hot_intraday

IST = ZoneInfo("Asia/Kolkata")


class TestHotIntraday(unittest.TestCase):

    def setUp(self):
        # Create synthetic 1-minute bars for today
        now = datetime.now(IST).replace(hour=9, minute=15, second=0, microsecond=0)
        times = [now + timedelta(minutes=i) for i in range(75)]
        closes = 1000.0 + np.cumsum(np.random.randn(75) * 0.5)
        self.sample_1m = pd.DataFrame({
            "symbol": ["TATAMOTORS"] * 75,
            "timestamp": times,
            "open": closes - 0.2,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.random.randint(500, 5000, size=75)
        })

    def test_append_and_resample(self):
        # 1. Append batch
        count = hot_intraday.append_1m_batch(self.sample_1m)
        self.assertEqual(count, 75)

        # 2. Check stats
        stats = hot_intraday.get_hot_stats()
        self.assertGreaterEqual(stats["candle_count"], 75)
        self.assertGreaterEqual(stats["symbol_count"], 1)

        # 3. Resample to 15m
        df_15 = hot_intraday.get_resampled_candles("TATAMOTORS", interval_minutes=15)
        self.assertFalse(df_15.empty)
        # 75 mins / 15 = 5 candles
        self.assertGreaterEqual(len(df_15), 4)
        self.assertIn("close", df_15.columns)
        self.assertIn("open", df_15.columns)

        # 4. Resample to 75m
        df_75 = hot_intraday.get_resampled_candles("TATAMOTORS", interval_minutes=75)
        self.assertFalse(df_75.empty)
        self.assertGreaterEqual(len(df_75), 1)

        # 5. Batch resample
        batch_res = hot_intraday.get_batch_resampled_candles(["TATAMOTORS"], interval_minutes=15)
        self.assertIn("TATAMOTORS", batch_res)
        self.assertFalse(batch_res["TATAMOTORS"].empty)

        # 6. Clear hot intraday
        cleared = hot_intraday.clear_hot_intraday()
        self.assertTrue(cleared)


if __name__ == "__main__":
    unittest.main()
