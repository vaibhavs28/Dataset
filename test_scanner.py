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

    def test_database_stats(self):
        stats = database.get_db_stats()
        self.assertGreater(stats["total_candles"], 0)
        self.assertGreaterEqual(stats["symbols_with_candles"], 2)

    def test_resampling_monthly(self):
        df = database.get_candles_df("RELIANCE")
        self.assertFalse(df.empty)
        
        monthly_df = scanner.resample_ohlcv(df, timeframe="monthly")
        self.assertFalse(monthly_df.empty)
        self.assertGreaterEqual(len(monthly_df), 12)  # At least 12 months for 2 years
        self.assertIn("close", monthly_df.columns)
        self.assertIn("open", monthly_df.columns)

    def test_ema_calculation(self):
        df = database.get_candles_df("RELIANCE")
        monthly_df = scanner.resample_ohlcv(df, timeframe="monthly")
        ind_df = scanner.calculate_indicator(monthly_df, indicator_type="EMA", period=5)
        
        self.assertIn("EMA_5", ind_df.columns)
        # Check that EMA is calculated and not all NaN
        valid_ema = ind_df["EMA_5"].dropna()
        self.assertGreater(len(valid_ema), 0)

    def test_scanner_run(self):
        results = scanner.run_scan(symbols=["RELIANCE", "TCS"], timeframe="monthly", period=5, condition="Close > Indicator")
        # Results should be a DataFrame (might be empty or have matching rows depending on generated values)
        self.assertIsInstance(results, pd.DataFrame)
        if not results.empty:
            self.assertIn("Symbol", results.columns)
            self.assertIn("Close", results.columns)
            self.assertIn("EMA_5", results.columns)
            self.assertIn("Diff %", results.columns)


    def test_rsi_calculation(self):
        df = database.get_candles_df("RELIANCE")
        rsi_series = scanner.calculate_rsi(df["close"], span=14)
        self.assertFalse(rsi_series.empty)
        # Standard indicator: first 14 candles must be NaN (warmup period)
        self.assertTrue(rsi_series.iloc[:14].isna().all())
        # Subsequent candles must be bounded between 0 and 100
        valid_rsi = rsi_series.dropna()
        self.assertFalse(valid_rsi.empty)
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

        ind_df = scanner.calculate_indicator(df, indicator_type="RSI", period=14)
        self.assertIn("RSI_14", ind_df.columns)
        valid_ind = ind_df["RSI_14"].dropna()
        self.assertFalse(valid_ind.empty)
        self.assertTrue((valid_ind >= 0).all())
        self.assertTrue((valid_ind <= 100).all())

    def test_monthly_rsi_scan(self):
        df = database.get_candles_df("RELIANCE")
        m_df = scanner.resample_ohlcv(df, timeframe="monthly")
        rsi = scanner.calculate_rsi(m_df["close"], span=14)
        ema3 = scanner.calculate_ema(rsi, span=3)
        wma21 = scanner.calculate_wma(rsi, period=21)

        self.assertEqual(len(rsi), len(ema3))
        self.assertEqual(len(rsi), len(wma21))
        self.assertFalse(ema3.isna().all())
        self.assertFalse(wma21.isna().all())

        # Test scan symbol condition
        res = scanner.scan_symbol("RELIANCE", timeframe="monthly", condition="RSI > 50")
        if res:
            self.assertIn("Symbol", res)

    def test_anchored_vwap(self):
        dates = pd.date_range("2019-01-01", "2024-01-01", freq="ME")
        df = pd.DataFrame({
            "high": [110.0] * len(dates),
            "low": [90.0] * len(dates),
            "close": [100.0] * len(dates),
            "volume": [1000] * len(dates)
        }, index=dates)

        avwap_mar20 = scanner.calculate_anchored_vwap(df, "2020-03-01")
        avwap_jun22 = scanner.calculate_anchored_vwap(df, "2022-06-01")

        self.assertTrue(pd.isna(avwap_mar20.loc["2020-02-29"]))
        self.assertAlmostEqual(avwap_mar20.loc["2020-03-31"], 100.0)
        self.assertTrue(pd.isna(avwap_jun22.loc["2022-05-31"]))
        self.assertAlmostEqual(avwap_jun22.loc["2022-06-30"], 100.0)


if __name__ == "__main__":
    unittest.main()

