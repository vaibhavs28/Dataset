"""
test_sync_75m_intraday.py
-------------------------
Unit tests for the 75-minute Upstox intraday sync engine and database persistence.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

import database
import duckdb_store
import sync_75m_intraday
import parquet_loader


class TestSync75mIntraday(unittest.TestCase):

    def setUp(self):
        self.test_symbol = "TEST_SYNC_STOCK"
        self.test_key = "NSE_EQ|INE999A01099"

    def test_resample_1min_to_75min_logic(self):
        """Verify that 1-minute bars are resampled accurately to 75m bars for Indian market hours."""
        # Create synthetic 1-min bars from 09:15 to 11:45 (2 x 75m intervals)
        times = pd.date_range("2026-09-11 09:15:00", "2026-09-11 11:45:00", freq="1min", tz="Asia/Kolkata")
        df_1m = pd.DataFrame({
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 10
        }, index=times)

        df_75m = parquet_loader.resample_1min_to_75min(df_1m)
        self.assertFalse(df_75m.empty)
        # Should contain slot 0 (09:15) and slot 1 (10:30)
        self.assertGreaterEqual(len(df_75m), 2)
        self.assertEqual(df_75m.iloc[0]["open"], 100.0)
        self.assertEqual(df_75m.iloc[0]["high"], 105.0)

    @patch("upstox_parquet_updater.fetch_upstox_1min_intraday")
    @patch("instruments.resolve_instrument_key")
    def test_sync_symbol_75m_from_upstox(self, mock_resolve, mock_fetch):
        """Verify that live 1-min Upstox candle response is parsed, resampled, and upserted into DB."""
        mock_resolve.return_value = self.test_key

        # Mock Upstox API response format: [timestamp, open, high, low, close, volume, oi]
        # Times in reverse chronological order as Upstox returns them (or ascending)
        mock_candles = [
            ["2026-09-11T10:30:00+05:30", 102.0, 106.0, 101.5, 105.0, 1500, 0],
            ["2026-09-11T10:00:00+05:30", 101.0, 103.0, 100.5, 102.0, 1200, 0],
            ["2026-09-11T09:15:00+05:30", 100.0, 102.0, 99.0, 101.0, 2000, 0],
        ]
        mock_fetch.return_value = mock_candles

        res = sync_75m_intraday.sync_symbol_75m_from_upstox(
            symbol=self.test_symbol,
            save_parquet=False,
            save_db=True
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["symbol"], self.test_symbol)
        self.assertGreater(res["bars_1m"], 0)

        # Verify records stored in DuckDB or SQLite
        df_read = database.get_intraday_candles_df(self.test_symbol, timeframe="75m")
        self.assertFalse(df_read.empty)
        self.assertIn("open", df_read.columns)
        self.assertIn("close", df_read.columns)

    def test_duckdb_upsert_intraday_candles(self):
        """Verify duckdb_store.upsert_intraday_candles commits and reads correctly."""
        records = [{
            "instrument_key": self.test_key,
            "trading_symbol": self.test_symbol,
            "timeframe": "75m",
            "timestamp": "2026-09-11 09:15:00",
            "open": 500.0,
            "high": 510.0,
            "low": 495.0,
            "close": 505.0,
            "volume": 25000
        }]
        duckdb_store.upsert_intraday_candles(records)

        df = duckdb_store.get_intraday_candles(self.test_symbol, timeframe="75m")
        self.assertFalse(df.empty)
        last_row = df.iloc[-1]
        self.assertEqual(last_row["close"], 505.0)


if __name__ == "__main__":
    unittest.main()
