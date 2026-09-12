import unittest
import pandas as pd
from datetime import datetime
import duckdb_store
import database
import parquet_loader

class TestDateRangeFilter(unittest.TestCase):
    def test_duckdb_resampled_candles_date_range(self):
        # AARTIIND has parquet data covering 2022-2026
        df = duckdb_store.get_resampled_candles(
            "AARTIIND",
            interval_minutes=75,
            min_date="2022-01-01",
            max_date="2023-12-31"
        )
        if not df.empty:
            self.assertGreater(len(df), 100)
            first_dt = df.index[0]
            last_dt = df.index[-1]
            self.assertGreaterEqual(first_dt.strftime("%Y-%m-%d"), "2022-01-01")
            self.assertLessEqual(last_dt.strftime("%Y-%m-%d"), "2023-12-31")

    def test_parquet_loader_custom_minute_date_range(self):
        df = parquet_loader.ensure_symbol_custom_minute_candles(
            "AARTIIND",
            interval_minutes=75,
            start_date="2022-01-01",
            end_date="2023-12-31"
        )
        if not df.empty:
            self.assertGreater(len(df), 100)
            first_dt = df.index[0]
            last_dt = df.index[-1]
            self.assertGreaterEqual(first_dt.strftime("%Y-%m-%d"), "2022-01-01")
            self.assertLessEqual(last_dt.strftime("%Y-%m-%d"), "2023-12-31")

    def test_database_intraday_candles_df_signature(self):
        # Verify function accepts start_date and end_date without errors
        df = database.get_intraday_candles_df(
            "RELIANCE",
            timeframe="75m",
            limit=10,
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
