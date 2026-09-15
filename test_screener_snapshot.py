"""
test_screener_snapshot.py
-------------------------
Unit tests for the DuckDB Precomputed Wide Table & SQL Query Engine.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from screener_engine import ScreenerConfig, ScreenerClause
import screener_snapshot


class TestScreenerSnapshot(unittest.TestCase):

    def setUp(self):
        # Generate synthetic daily data for testing snapshot calculation
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(150)]
        close = 1000.0 + np.sin(np.linspace(0, 10, 150)) * 50.0
        open_p = close - 2.0
        high = close + 5.0
        low = close - 5.0
        vol = np.random.randint(100000, 500000, size=150)
        self.synthetic_daily = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol
        }, index=pd.DatetimeIndex(dates))

    def test_compute_stock_snapshot(self):
        row = screener_snapshot.compute_stock_snapshot("TESTSYM", self.synthetic_daily)
        self.assertIsNotNone(row)
        self.assertEqual(row["symbol"], "TESTSYM")
        self.assertIn("d_ema20", row)
        self.assertIn("d_rsi9", row)
        self.assertIn("w_rsi9", row)
        self.assertIn("m_rsi9", row)
        self.assertGreater(row["ltp"], 0)

    def test_compile_clauses_to_sql_simple(self):
        cfg = ScreenerConfig(
            name="Test Scan",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_20"),
                ScreenerClause(timeframe="Daily", lhs="RSI_9", operator="<", rhs_type="Number", rhs_value=50.0)
            ]
        )
        sql, params = screener_snapshot.compile_clauses_to_sql(cfg)
        self.assertIn("d_close > d_ema20", sql)
        self.assertIn("d_rsi9 < 50.0", sql)
        self.assertIn("screener_flat_snapshot", sql)

    def test_compile_clauses_to_sql_cross_tf_and_squeeze(self):
        cfg = ScreenerConfig(
            name="Test 19122704 Stage 5",
            logic="ALL",
            clauses=[
                # Cross-TF
                ScreenerClause(timeframe="15-Min", lhs="Close", operator="<", rhs_type="Indicator", rhs_indicator="EMA_20", rhs_timeframe="Daily"),
                # MA squeeze
                ScreenerClause(timeframe="15-Min", lhs="EMA_9", operator="abs_pct_lte", rhs_type="Number", rhs_value=0.01, rhs_indicator="EMA_13")
            ]
        )
        sql, params = screener_snapshot.compile_clauses_to_sql(cfg)
        self.assertIn("intra15_close < d_ema20", sql)
        self.assertIn("intra15_ema9", sql)
        self.assertIn("intra15_ema13", sql)
        self.assertIn("0.01", sql)

    def test_compile_with_universe_symbols(self):
        cfg = ScreenerConfig(
            name="Universe Test",
            logic="ALL",
            clauses=[ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Number", rhs_value=100.0)]
        )
        sql, params = screener_snapshot.compile_clauses_to_sql(cfg, universe_symbols=["RELIANCE", "TCS"])
        self.assertIn("symbol IN (?, ?)", sql)
        self.assertEqual(len(params), 2)


if __name__ == "__main__":
    unittest.main()
