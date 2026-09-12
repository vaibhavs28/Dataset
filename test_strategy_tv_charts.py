"""
test_strategy_tv_charts.py
--------------------------
Unit tests for TradingView trade execution charts, equity curve & drawdown HTML generation,
and advanced quantitative analytics in Strategy Lab.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import tradingview_charts
from strategy_engine import StrategyConfig, BacktestResult, Trade
import strategy_ui


class TestStrategyTradingViewCharts(unittest.TestCase):

    def setUp(self):
        # Create synthetic OHLCV data for 30 daily bars
        dates = pd.date_range("2023-01-01", periods=40, freq="B")
        base_price = 1000.0
        np.random.seed(42)
        noise = np.cumsum(np.random.randn(40) * 10)
        close = base_price + noise
        open_p = close + np.random.randn(40) * 5
        high = np.maximum(open_p, close) + np.abs(np.random.randn(40) * 8)
        low = np.minimum(open_p, close) - np.abs(np.random.randn(40) * 8)
        volume = np.random.randint(50000, 500000, size=40)

        self.df = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "EMA_Fast": close * 0.99,
            "EMA_Mid": close * 0.98,
            "EMA_Slow": close * 0.97,
            "RSI": np.linspace(35, 75, 40)
        }, index=dates)

        # Create sample trades
        self.trades = [
            Trade(
                trade_id=1,
                symbol="RELIANCE",
                direction="LONG",
                entry_time=dates[2],
                entry_price=float(open_p[2]),
                exit_time=dates[8],
                exit_price=float(close[8]),
                quantity=50,
                pnl_rupees=float((close[8] - open_p[2]) * 50),
                pnl_percent=float((close[8] - open_p[2]) / open_p[2] * 100),
                exit_reason="Target Achieved",
                duration_bars=6,
                max_favorable_excursion=4.2,
                max_adverse_excursion=-0.8
            ),
            Trade(
                trade_id=2,
                symbol="RELIANCE",
                direction="LONG",
                entry_time=dates[15],
                entry_price=float(open_p[15]),
                exit_time=dates[20],
                exit_price=float(close[20]),
                quantity=50,
                pnl_rupees=float((close[20] - open_p[15]) * 50),
                pnl_percent=float((close[20] - open_p[15]) / open_p[15] * 100),
                exit_reason="Stop Loss Hit",
                duration_bars=5,
                max_favorable_excursion=1.1,
                max_adverse_excursion=-2.5
            )
        ]

        # Sample equity curve
        eq_vals = [100000.0]
        for i in range(1, 40):
            ret = np.random.randn() * 0.01
            eq_vals.append(eq_vals[-1] * (1.0 + ret))
        
        eq_arr = np.array(eq_vals)
        hwm_arr = np.maximum.accumulate(eq_arr)
        dd_arr = (hwm_arr - eq_arr) / hwm_arr * 100.0

        self.equity_df = pd.DataFrame({
            "equity": eq_arr,
            "high_watermark": hwm_arr,
            "drawdown_pct": dd_arr
        }, index=dates)

        self.result = BacktestResult(
            symbol="RELIANCE",
            strategy_name="Hilega Milega",
            timeframe="Daily",
            start_time=dates[0],
            end_time=dates[-1],
            initial_capital=100000.0,
            final_equity=eq_arr[-1],
            total_net_pnl=eq_arr[-1] - 100000.0,
            total_net_pnl_pct=(eq_arr[-1] - 100000.0) / 1000.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=50.0,
            profit_factor=1.45,
            max_drawdown_pct=float(np.max(dd_arr)),
            max_drawdown_rupees=2500.0,
            avg_trade_pnl=500.0,
            avg_win=1500.0,
            avg_loss=-1000.0,
            win_loss_ratio=1.5,
            expectancy=250.0,
            trades=self.trades,
            equity_curve=self.equity_df
        )

    def test_generate_strategy_backtest_chart_html(self):
        html = tradingview_charts.generate_strategy_backtest_chart_html(
            df=self.df,
            trades=self.trades,
            symbol="RELIANCE",
            strategy_name="⚡ Hilega Milega Momentum",
            timeframe="Daily",
            height=620,
            theme="dark"
        )
        self.assertIsInstance(html, str)
        self.assertIn("lightweight-charts", html)
        self.assertIn("BUY #1", html)
        self.assertIn("EXIT #1", html)
        self.assertIn("arrowUp", html)
        self.assertIn("arrowDown", html)
        self.assertIn("RELIANCE", html)
        # Volume should be hidden by default
        self.assertIn("const showVolumeInit = false;", html)
        self.assertIn("toggleVolume()", html)

        # When show_volume=True is explicitly requested
        html_vol = tradingview_charts.generate_strategy_backtest_chart_html(
            df=self.df,
            trades=self.trades,
            symbol="RELIANCE",
            strategy_name="⚡ Hilega Milega Momentum",
            timeframe="Daily",
            show_volume=True
        )
        self.assertIn("const showVolumeInit = true;", html_vol)

    def test_generate_strategy_chart_empty_data(self):
        html = tradingview_charts.generate_strategy_backtest_chart_html(
            df=pd.DataFrame(),
            trades=[],
            symbol="TEST",
            strategy_name="Empty",
            timeframe="Daily"
        )
        self.assertIn("No candle data available", html)

    def test_generate_equity_drawdown_chart_html(self):
        html = tradingview_charts.generate_equity_drawdown_chart_html(
            equity_df=self.equity_df,
            initial_capital=100000.0,
            final_equity=self.result.final_equity,
            symbol="RELIANCE",
            strategy_name="Hilega Milega",
            benchmark_df=self.df,
            height=450,
            theme="dark"
        )
        self.assertIsInstance(html, str)
        self.assertIn("Portfolio Equity Curve", html)
        self.assertIn("Drawdown (%)", html)
        self.assertIn("Buy & Hold Benchmark", html)
        self.assertIn("Peak Equity", html)

    def test_calculate_advanced_metrics(self):
        metrics = strategy_ui.calculate_advanced_metrics(self.result, 100000.0, self.df)
        self.assertIn("cagr_pct", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("calmar_ratio", metrics)
        self.assertIn("max_consecutive_wins", metrics)
        self.assertIn("max_consecutive_losses", metrics)
        self.assertIn("payoff_ratio", metrics)
        self.assertEqual(metrics["payoff_ratio"], 1.5)

    def test_calculate_monthly_returns_matrix(self):
        matrix = strategy_ui.calculate_monthly_returns_matrix(self.equity_df)
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertIn("Year", matrix.columns)
        self.assertIn("Year Return", matrix.columns)
        self.assertIn("Jan", matrix.columns)

    def test_calculate_exit_reason_breakdown(self):
        breakdown = strategy_ui.calculate_exit_reason_breakdown(self.trades, stop_loss_pct=2.0)
        self.assertIsInstance(breakdown, pd.DataFrame)
        self.assertEqual(len(breakdown), 2)
        self.assertIn("Target Achieved", breakdown["Exit Reason"].values)
        self.assertIn("Stop Loss Hit", breakdown["Exit Reason"].values)
        self.assertIn("Avg Risk:Reward", breakdown.columns)

    def test_custom_timeframe_and_risk_reward(self):
        self.trades[0].risk_reward = "1 : 2.00 (+2.00R)"
        html = tradingview_charts.generate_strategy_backtest_chart_html(
            df=self.df,
            trades=self.trades,
            symbol="RELIANCE",
            strategy_name="Custom Strategy",
            timeframe="15-Min",
            height=620,
            theme="dark"
        )
        self.assertIn("15-Min", html)
        self.assertIn("1 : 2.00 (+2.00R)", html)
        self.assertIn("R:R:", html)

    def test_load_candles_for_simulation_custom_tf(self):
        df_daily = strategy_ui.load_candles_for_simulation("RELIANCE", "Daily")
        self.assertFalse(df_daily.empty)
        df_15m = strategy_ui.load_candles_for_simulation("RELIANCE", "15-Min")
        self.assertFalse(df_15m.empty)
        df_cust = strategy_ui.load_candles_for_simulation("RELIANCE", "Custom (Minutes)", custom_minutes=10)
        self.assertFalse(df_cust.empty)


if __name__ == "__main__":
    unittest.main()
