"""
test_screener.py
----------------
Unit and integration tests for screener_engine.py.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from screener_engine import (
    ScreenerClause,
    ScreenerConfig,
    compute_screener_indicators,
    evaluate_clause,
    evaluate_stock,
    run_screen,
    get_screener_preset,
    parse_chartink_query,
)


def create_test_ohlcv(n_bars: int = 60, base_price: float = 100.0) -> pd.DataFrame:
    """Creates synthetic OHLCV dataframe."""
    np.random.seed(42)
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    close = base_price + np.cumsum(np.random.normal(0.5, 1.2, n_bars))
    high = close + 1.5
    low = close - 1.5
    open_p = close - 0.2
    volume = np.random.randint(50000, 200000, n_bars)
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=pd.DatetimeIndex(dates))
    return df


class TestScreenerEngine(unittest.TestCase):

    def setUp(self):
        self.df = create_test_ohlcv(60, 100.0)
        self.df_ind = compute_screener_indicators(self.df)

    def test_indicators_computed(self):
        self.assertIn("RSI_14", self.df_ind.columns)
        self.assertIn("RSI_9", self.df_ind.columns)
        self.assertIn("EMA_20", self.df_ind.columns)
        self.assertIn("EMA_50", self.df_ind.columns)
        self.assertIn("SuperTrend", self.df_ind.columns)
        self.assertIn("52_Week_High", self.df_ind.columns)
        self.assertIn("Vol_SMA_20", self.df_ind.columns)

    def test_evaluate_clause_number(self):
        # RSI > 0 is always true
        c = ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=0.0)
        passed, lhs, rhs = evaluate_clause(self.df_ind, c)
        self.assertTrue(passed)
        self.assertGreater(lhs, 0.0)

        # RSI > 100 is always false
        c2 = ScreenerClause(timeframe="Daily", lhs="RSI_14", operator=">", rhs_type="Number", rhs_value=100.0)
        passed2, lhs2, rhs2 = evaluate_clause(self.df_ind, c2)
        self.assertFalse(passed2)

    def test_evaluate_clause_indicator_comparison(self):
        # High >= Low is always true
        c = ScreenerClause(timeframe="Daily", lhs="High", operator=">=", rhs_type="Indicator", rhs_indicator="Low")
        passed, lhs, rhs = evaluate_clause(self.df_ind, c)
        self.assertTrue(passed)

    def test_all_presets_valid(self):
        preset_names = [
            "🚀 RSI Momentum Surge",
            "🏔️ 52-Week / Multi-Year High Breakout",
            "🏹 SuperTrend Fresh Bullish Reversal",
            "🌊 Triple EMA Bullish Stack (9 > 20 > 50)",
            "💥 Volume Shocker & Price Breakout",
            "⚡ Hilega Milega Bullish Alignment (NK Sir)",
            "🎯 Pullback to 20 EMA (Dip Buying Setup)",
            "🔻 Oversold Bounce Setup (RSI < 30)"
        ]
        for name in preset_names:
            cfg = get_screener_preset(name)
            self.assertIsInstance(cfg, ScreenerConfig)
            self.assertGreater(len(cfg.clauses), 0)

    def test_chartink_query_parser(self):
        query = "Daily Close > Daily 20 EMA\nDaily RSI(14) > 60"
        clauses = parse_chartink_query(query)
        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0].lhs, "Close")
        self.assertEqual(clauses[0].operator, ">")
        self.assertEqual(clauses[0].rhs_indicator, "EMA_20")
        self.assertEqual(clauses[1].lhs, "RSI_14")
        self.assertEqual(clauses[1].operator, ">")
        self.assertEqual(clauses[1].rhs_value, 60.0)

    def test_run_screen_mock_data(self):
        stocks_data = {
            "STOCK_UP": create_test_ohlcv(50, 100.0),
            "STOCK_DOWN": create_test_ohlcv(50, 50.0),
        }
        cfg = ScreenerConfig(
            name="Test Screener",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="High", operator=">", rhs_type="Indicator", rhs_indicator="Low")
            ]
        )
        res = run_screen(list(stocks_data.keys()), cfg, data_provider_fn=lambda s, tf: stocks_data.get(s))
        self.assertFalse(res.empty)
        self.assertEqual(len(res), 2)
        self.assertIn("Symbol", res.columns)
        self.assertIn("LTP", res.columns)

    def test_run_waterfall_scan(self):
        from screener_engine import run_waterfall_scan
        stocks_data = {
            "STOCK_A": create_test_ohlcv(120, 200.0),
            "STOCK_B": create_test_ohlcv(120, 500.0),
        }
        res_dict = run_waterfall_scan(
            list(stocks_data.keys()),
            data_provider_fn=lambda s, tf: stocks_data.get(s)
        )
        self.assertIn("all_waterfall", res_dict)
        self.assertIn("stage_4_full", res_dict)
        self.assertIn("counts", res_dict)
        self.assertEqual(res_dict["counts"]["total"], 2)

    def test_historical_as_of_date_waterfall(self):
        from screener_engine import evaluate_stock_waterfall, run_waterfall_scan
        df = create_test_ohlcv(100, 200.0)
        # Select date halfway through
        mid_date = df.index[45].strftime("%Y-%m-%d")
        
        # Sliced directly or via as_of_date
        res_hist = evaluate_stock_waterfall("TEST_SYM", df, as_of_date=mid_date, as_of_time="13:00")
        if res_hist is not None:
            self.assertEqual(res_hist["Scan Date"], mid_date)
            self.assertEqual(res_hist["Scan Time"], "13:00")
            self.assertIsNotNone(res_hist.get("Return Since Scan (%)"))
            self.assertEqual(res_hist["LTP"], round(float(df.iloc[45]["close"]), 2))

    def test_historical_as_of_date_screen(self):
        df = create_test_ohlcv(100, 150.0)
        mid_date = df.index[50].strftime("%Y-%m-%d")
        cfg = ScreenerConfig(
            name="Test Screener",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="High", operator=">", rhs_type="Indicator", rhs_indicator="Low")
            ]
        )
        res = run_screen(["TEST_SYM"], cfg, as_of_date=mid_date, as_of_time="11:45", data_provider_fn=lambda s, tf: df)
        self.assertFalse(res.empty)
        self.assertEqual(res.iloc[0]["Scan_Date"], mid_date)
        self.assertEqual(res.iloc[0]["Scan_Time"], "11:45")
        self.assertIn("Return_Since_Scan_%", res.columns)

    def test_dynamic_indicator_calculation(self):
        from screener_engine import ensure_indicator
        df_copy = self.df.copy()
        # Test dynamic calculation of arbitrary periods
        self.assertTrue(ensure_indicator(df_copy, "EMA_100"))
        self.assertIn("EMA_100", df_copy.columns)
        self.assertTrue(ensure_indicator(df_copy, "EMA_21"))
        self.assertIn("EMA_21", df_copy.columns)
        self.assertTrue(ensure_indicator(df_copy, "SMA_10"))
        self.assertIn("SMA_10", df_copy.columns)
        self.assertTrue(ensure_indicator(df_copy, "RSI_21"))
        self.assertIn("RSI_21", df_copy.columns)

        # Test evaluate_clause with dynamic indicator
        c = ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Indicator", rhs_indicator="EMA_100")
        passed, lhs, rhs = evaluate_clause(df_copy, c)
        self.assertIsInstance(passed, (bool, np.bool_))
        self.assertGreater(rhs, 0.0)

    def test_chartink_query_parser_exact_periods(self):
        query = (
            "[ Latest ] Close Greater than [ Latest ] EMA(close, 100)\n"
            "[ Latest ] Close Greater than Number 100\n"
            "[ Latest ] Close Less than Number 5000\n"
            "[ Latest ] SMA(close, 200) > [ Latest ] EMA(close, 50)"
        )
        clauses = parse_chartink_query(query)
        self.assertEqual(len(clauses), 4)
        self.assertEqual(clauses[0].lhs, "Close")
        self.assertEqual(clauses[0].operator, ">")
        self.assertEqual(clauses[0].rhs_indicator, "EMA_100")

        self.assertEqual(clauses[1].lhs, "Close")
        self.assertEqual(clauses[1].operator, ">")
        self.assertEqual(clauses[1].rhs_type, "Number")
        self.assertEqual(clauses[1].rhs_value, 100.0)

        self.assertEqual(clauses[2].lhs, "Close")
        self.assertEqual(clauses[2].operator, "<")
        self.assertEqual(clauses[2].rhs_type, "Number")
        self.assertEqual(clauses[2].rhs_value, 5000.0)

        self.assertEqual(clauses[3].lhs, "SMA_200")
        self.assertEqual(clauses[3].operator, ">")
        self.assertEqual(clauses[3].rhs_indicator, "EMA_50")

    def test_run_screen_diagnostics(self):
        stocks_data = {
            "STOCK_A": create_test_ohlcv(50, 100.0),
            "STOCK_B": create_test_ohlcv(50, 50.0),
        }
        # Impossible filter: Close > 1,000,000
        cfg = ScreenerConfig(
            name="Impossible Screener",
            logic="ALL",
            clauses=[
                ScreenerClause(timeframe="Daily", lhs="Close", operator=">", rhs_type="Number", rhs_value=1000000.0)
            ]
        )
        res = run_screen(list(stocks_data.keys()), cfg, data_provider_fn=lambda s, tf: stocks_data.get(s))
        self.assertTrue(res.empty)
        self.assertIn("diag", res.attrs)
        diag = res.attrs["diag"]
        self.assertEqual(diag["evaluated_stocks"], 2)
        self.assertEqual(diag["clause_stats"][0]["passed_count"], 0)

    def test_intraday_scan_19122704_preset(self):
        cfg = get_screener_preset("intraday-scan-19122704")
        self.assertIn("19122704", cfg.name)
        self.assertGreaterEqual(len(cfg.clauses), 20)
        # Check presence of Monthly, Weekly, Daily, 75-Min timeframes
        tfs = {c.timeframe for c in cfg.clauses}
        self.assertEqual(tfs, {"Monthly", "Weekly", "Daily", "75-Min"})

    def test_evaluate_intraday_scan_19122704(self):
        from screener_engine import evaluate_intraday_scan_19122704
        # Downtrending synthetic data
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(120)]
        close = 1000.0 - np.arange(120) * 3.0
        df_down = pd.DataFrame({
            "open": close + 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": [100000] * 120
        }, index=pd.DatetimeIndex(dates))

        res = evaluate_intraday_scan_19122704("TEST_BEAR", df_down)
        if res is not None:
            self.assertIn(res["Stage"], [1, 2, 3, 4])
            self.assertEqual(res["Symbol"], "TEST_BEAR")


if __name__ == "__main__":
    unittest.main()


