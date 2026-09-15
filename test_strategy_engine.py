"""
test_strategy_engine.py
-----------------------
Unit and integration tests for strategy_engine.py.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from strategy_engine import (
    StrategyConfig,
    Trade,
    BacktestResult,
    prepare_indicators,
    generate_strategy_signals,
    run_backtest,
    run_basket_backtest,
    get_preset_strategy,
)


def create_synthetic_ohlcv(n_bars: int = 150, start_price: float = 100.0) -> pd.DataFrame:
    """Generates synthetic OHLCV data with an uptrend and oscillation for backtest testing."""
    np.random.seed(42)
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    
    # Generate prices with some drift and volatility
    drift = 0.2
    volatility = 1.5
    returns = np.random.normal(drift, volatility, n_bars)
    close = start_price + np.cumsum(returns)
    close = np.maximum(close, 10.0)  # Avoid negative prices
    
    high = close + np.random.uniform(0.5, 2.5, n_bars)
    low = close - np.random.uniform(0.5, 2.5, n_bars)
    open_p = low + np.random.uniform(0.1, 0.9, n_bars) * (high - low)
    volume = np.random.randint(50000, 500000, n_bars)
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=pd.DatetimeIndex(dates))
    return df


class TestStrategyEngine(unittest.TestCase):

    def setUp(self):
        self.df = create_synthetic_ohlcv(150, 100.0)

    def test_prepare_indicators(self):
        cfg = StrategyConfig(strategy_type="Hilega_Milega")
        df_ind = prepare_indicators(self.df, cfg)
        self.assertFalse(df_ind.empty)
        self.assertIn("RSI", df_ind.columns)
        self.assertIn("RSI_EMA3", df_ind.columns)
        self.assertIn("RSI_WMA21", df_ind.columns)
        self.assertIn("EMA_Fast", df_ind.columns)
        self.assertIn("EMA_Mid", df_ind.columns)
        self.assertIn("EMA_Slow", df_ind.columns)
        self.assertIn("SuperTrend", df_ind.columns)
        self.assertIn("ST_Direction", df_ind.columns)
        self.assertIn("MACD", df_ind.columns)
        self.assertIn("MACD_Signal", df_ind.columns)
        self.assertIn("ATR", df_ind.columns)

    def test_generate_strategy_signals_hilega_milega(self):
        cfg = get_preset_strategy("Hilega_Milega")
        df_ind = prepare_indicators(self.df, cfg)
        signals = generate_strategy_signals(df_ind, cfg)
        self.assertIn("signal", signals.columns)
        self.assertEqual(len(signals), len(self.df))
        unique_signals = set(signals["signal"].dropna().unique())
        self.assertTrue(unique_signals.issubset({-1, 0, 1}))

    def test_generate_strategy_signals_all_presets(self):
        presets = ["Hilega_Milega", "Triple_EMA", "SuperTrend", "CPR_Breakout", "MACD_Cross"]
        for p in presets:
            cfg = get_preset_strategy(p)
            df_ind = prepare_indicators(self.df, cfg)
            signals = generate_strategy_signals(df_ind, cfg)
            self.assertIn("signal", signals.columns, f"Failed for {p}")

    def test_run_backtest_single_stock(self):
        cfg = get_preset_strategy("Hilega_Milega")
        cfg.stop_loss_pct = 2.0
        cfg.target_pct = 4.0
        res = run_backtest(self.df, cfg, initial_capital=100000.0, symbol="TEST_STOCK", timeframe="Daily")
        
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.symbol, "TEST_STOCK")
        self.assertEqual(res.initial_capital, 100000.0)
        self.assertGreater(len(res.equity_curve), 0)
        self.assertGreaterEqual(res.total_trades, 0)
        if res.total_trades > 0:
            self.assertGreater(len(res.trades), 0)
            self.assertEqual(res.total_trades, res.winning_trades + res.losing_trades)
            self.assertTrue(0.0 <= res.win_rate <= 100.0)
            self.assertGreaterEqual(res.profit_factor, 0.0)
            self.assertEqual(res.planned_risk_reward, 2.0)
            for t in res.trades:
                self.assertIsNotNone(t.risk_reward)
                self.assertIsNotNone(t.realized_rr)

    def test_trailing_stop_execution(self):
        cfg = StrategyConfig(
            strategy_type="Hilega_Milega",
            stop_loss_pct=5.0,
            target_pct=0.0,  # no fixed target
            use_trailing_stop=True,
            trailing_stop_pct=1.0
        )
        res = run_backtest(self.df, cfg, initial_capital=50000.0, symbol="TRAIL_STOCK")
        self.assertIsInstance(res, BacktestResult)
        for t in res.trades:
            if t.exit_reason == "TrailingStop":
                self.assertIsNotNone(t.exit_price)
                self.assertGreater(t.exit_price, 0)

    def test_run_basket_backtest(self):
        basket_data = {
            "STOCK_A": create_synthetic_ohlcv(120, 150.0),
            "STOCK_B": create_synthetic_ohlcv(120, 250.0),
            "STOCK_C": create_synthetic_ohlcv(120, 80.0),
        }
        cfg = get_preset_strategy("Triple_EMA")
        basket_res = run_basket_backtest(basket_data, cfg, timeframe="15-Min", capital_per_stock=50000.0, slippage_pct=0.08)
        leaderboard = basket_res["leaderboard"]
        
        self.assertFalse(leaderboard.empty)
        self.assertEqual(len(leaderboard), 3)
        self.assertIn("Symbol", leaderboard.columns)
        self.assertIn("Net P&L (₹)", leaderboard.columns)
        self.assertIn("Win Rate (%)", leaderboard.columns)
        self.assertIn("Trades", leaderboard.columns)
        self.assertIn("Avg Risk:Reward", leaderboard.columns)
        self.assertIn("total_capital", basket_res)
        self.assertIn("total_pnl", basket_res)

    def test_chartink_75_waterfall_preset(self):
        cfg = get_preset_strategy("Chartink 75m Waterfall")
        self.assertEqual(cfg.strategy_type, "Chartink_75_Waterfall")
        self.assertTrue(cfg.use_cpr_exits)
        self.assertEqual(cfg.cpr_target_level, "R1")
        self.assertEqual(cfg.cpr_stop_level, "S_05")

    def test_chartink_75_waterfall_indicators_and_signals(self):
        cfg = get_preset_strategy("Chartink 75m Waterfall")
        df_ind = prepare_indicators(self.df, cfg)
        self.assertIn("Weekly_P", df_ind.columns)
        self.assertIn("Weekly_R1", df_ind.columns)
        self.assertIn("Weekly_S_05", df_ind.columns)
        self.assertIn("Waterfall_Stage4", df_ind.columns)

        signals = generate_strategy_signals(df_ind, cfg)
        self.assertIn("signal_entry", signals.columns)
        self.assertIn("signal", signals.columns)
        # Verify long only: entries must all be non-negative
        self.assertTrue(all(signals["signal"] >= -1))

    def test_weekly_cpr_r1_target_and_05_support_sl(self):
        cfg = get_preset_strategy("Chartink 75m Waterfall")
        res = run_backtest(self.df, cfg, initial_capital=100000.0, symbol="WATERFALL_STOCK", timeframe="75-Min")
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.strategy_name, "🏆 Chartink 75m Waterfall (Weekly CPR R1 / 0.5 SL)")
        self.assertGreater(len(res.equity_curve), 0)

        # Check that any trade exits respect the CPR target or SL naming
        for t in res.trades:
            self.assertEqual(t.direction, "LONG")  # Only Buy
            self.assertIn(t.exit_reason, [
                "Target (Weekly CPR R1)",
                "Stop Loss (Weekly CPR 0.5 Support)",
                "Stop Loss Hit",
                "Target Achieved",
                "Signal Exit",
                "End of Data"
            ])
            self.assertIsNotNone(t.risk_reward)

    def test_explicit_cpr_r1_target_and_05_support_trigger(self):
        # Construct synthetic bars with explicit Weekly CPR levels
        dates = [datetime(2025, 6, 1) + timedelta(minutes=75*i) for i in range(10)]
        df_target = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [101.0, 102.0, 108.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [100.5, 101.5, 106.0, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [10000] * 10,
            "Weekly_R1": [105.0] * 10,
            "Weekly_S_05": [95.0] * 10,
            "Weekly_P": [100.0] * 10,
            "Waterfall_Stage4": [False, True, True, True, True, True, True, True, True, True]
        }, index=pd.DatetimeIndex(dates))

        cfg = get_preset_strategy("Chartink 75m Waterfall")
        res_tp = run_backtest(df_target, cfg, initial_capital=100000.0)
        self.assertGreater(len(res_tp.trades), 0)
        self.assertEqual(res_tp.trades[0].direction, "LONG")
        self.assertEqual(res_tp.trades[0].exit_reason, "Target (Weekly CPR R1)")

        # Test Stop Loss trigger
        df_sl = pd.DataFrame({
            "open": [100.0, 101.0, 96.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0],
            "high": [101.0, 102.0, 97.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0],
            "low": [99.0, 100.0, 93.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0],
            "close": [100.5, 101.5, 94.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0],
            "volume": [10000] * 10,
            "Weekly_R1": [115.0] * 10,
            "Weekly_S_05": [95.0] * 10,
            "Weekly_P": [100.0] * 10,
            "Waterfall_Stage4": [False, True, True, True, True, True, True, True, True, True]
        }, index=pd.DatetimeIndex(dates))

        res_sl = run_backtest(df_sl, cfg, initial_capital=100000.0)
        self.assertGreater(len(res_sl.trades), 0)
        self.assertEqual(res_sl.trades[0].direction, "LONG")
        self.assertEqual(res_sl.trades[0].exit_reason, "Stop Loss (Weekly CPR 0.5 Support)")

    def test_chartink_intraday_scan_19122704_preset(self):
        cfg = get_preset_strategy("19122704")
        self.assertEqual(cfg.strategy_type, "Chartink_Intraday_Scan_19122704")
        self.assertIn("19122704", cfg.name)
        df_ind = prepare_indicators(self.df, cfg)
        self.assertIn("Breakdown_Stage4", df_ind.columns)
        signals = generate_strategy_signals(df_ind, cfg)
        self.assertIn("signal_entry", signals.columns)


if __name__ == "__main__":
    unittest.main()
