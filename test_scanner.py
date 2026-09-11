import unittest
import pandas as pd
import numpy as np
import database
import scanner
import downloader


class TestScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()
        # Seed test data for 5 stocks
        downloader.seed_demo_data(symbols=["TEST_A", "TEST_B"], years=2)

    def test_database_stats(self):
        stats = database.get_db_stats()
        self.assertGreater(stats["total_candles"], 0)
        self.assertGreaterEqual(stats["symbols_with_candles"], 2)

    def test_resampling_monthly(self):
        df = database.get_candles_df("TEST_A")
        self.assertFalse(df.empty)
        
        monthly_df = scanner.resample_ohlcv(df, timeframe="monthly")
        self.assertFalse(monthly_df.empty)
        self.assertGreaterEqual(len(monthly_df), 12)  # At least 12 months for 2 years
        self.assertIn("close", monthly_df.columns)
        self.assertIn("open", monthly_df.columns)

    def test_ema_calculation(self):
        df = database.get_candles_df("TEST_A")
        monthly_df = scanner.resample_ohlcv(df, timeframe="monthly")
        ind_df = scanner.calculate_indicator(monthly_df, indicator_type="EMA", period=5)
        
        self.assertIn("EMA_5", ind_df.columns)
        # Check that EMA is calculated and not all NaN
        valid_ema = ind_df["EMA_5"].dropna()
        self.assertGreater(len(valid_ema), 0)

    def test_scanner_run(self):
        results = scanner.run_scan(symbols=["TEST_A", "TEST_B"], timeframe="monthly", period=5, condition="Close > Indicator")
        # Results should be a DataFrame (might be empty or have matching rows depending on generated values)
        self.assertIsInstance(results, pd.DataFrame)
        if not results.empty:
            self.assertIn("Symbol", results.columns)
            self.assertIn("Close", results.columns)
            self.assertIn("EMA_5", results.columns)
            self.assertIn("Diff %", results.columns)


    def test_rsi_calculation(self):
        df = database.get_candles_df("TEST_A")
        rsi_series = scanner.calculate_rsi(df["close"], span=14)
        self.assertFalse(rsi_series.empty)
        # RSI must be bounded between 0 and 100
        self.assertTrue((rsi_series >= 0).all())
        self.assertTrue((rsi_series <= 100).all())

        ind_df = scanner.calculate_indicator(df, indicator_type="RSI", period=14)
        self.assertIn("RSI_14", ind_df.columns)
        self.assertTrue((ind_df["RSI_14"] >= 0).all())
        self.assertTrue((ind_df["RSI_14"] <= 100).all())

    def test_monthly_rsi_scan(self):
        df = database.get_candles_df("TEST_A")
        m_df = scanner.resample_ohlcv(df, timeframe="monthly")
        rsi = scanner.calculate_rsi(m_df["close"], span=14)
        ema3 = scanner.calculate_ema(rsi, span=3)
        wma21 = scanner.calculate_wma(rsi, period=21)

        self.assertEqual(len(rsi), len(ema3))
        self.assertEqual(len(rsi), len(wma21))
        self.assertFalse(ema3.isna().all())
        self.assertFalse(wma21.isna().all())

        # Test scan symbol condition
        res = scanner.scan_symbol("TEST_A", timeframe="monthly", condition="RSI > 50")
        if res:
            self.assertIn("Symbol", res)


if __name__ == "__main__":
    unittest.main()
