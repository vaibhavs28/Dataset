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


if __name__ == "__main__":
    unittest.main()
