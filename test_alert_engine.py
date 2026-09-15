"""
test_alert_engine.py
--------------------
Unit tests for alert_engine.py.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import alert_engine


def create_mock_candles(n_bars: int = 30, start_price: float = 100.0, trend: float = 0.5) -> pd.DataFrame:
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    close = start_price + np.cumsum(np.full(n_bars, trend))
    high = close + 1.0
    low = close - 1.0
    open_p = close - trend
    volume = np.full(n_bars, 100000)
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=pd.DatetimeIndex(dates))


class TestAlertEngine(unittest.TestCase):

    def setUp(self):
        alert_engine.init_alerts_db()

    def test_alert_crud_and_status(self):
        alert_id = alert_engine.create_alert(
            symbol="TEST_RELIANCE",
            alert_type="STATIC_PRICE",
            condition={"target_price": 2500.0, "operator": ">="},
            channels=["telegram", "in_app"],
            trigger_mode="ONCE",
            note="Test Alert"
        )
        self.assertIsInstance(alert_id, int)
        self.assertGreater(alert_id, 0)

        # Retrieve
        all_alerts = alert_engine.get_all_alerts()
        found = [a for a in all_alerts if a["alert_id"] == alert_id]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["symbol"], "TEST_RELIANCE")
        self.assertEqual(found[0]["condition"]["target_price"], 2500.0)

        # Toggle status
        alert_engine.update_alert_status(alert_id, False)
        active_alerts = alert_engine.get_active_alerts()
        self.assertNotIn(alert_id, [a["alert_id"] for a in active_alerts])

        # Delete
        alert_engine.delete_alert(alert_id)
        all_after_del = alert_engine.get_all_alerts()
        self.assertNotIn(alert_id, [a["alert_id"] for a in all_after_del])

    def test_channel_config_persistence(self):
        cfg = alert_engine.get_channel_config()
        self.assertIn("telegram", cfg)
        self.assertIn("whatsapp", cfg)
        self.assertIn("email", cfg)

        cfg["telegram"]["bot_token"] = "mock_test_token"
        cfg["telegram"]["chat_id"] = "mock_test_chat"
        alert_engine.save_channel_config(cfg)

        loaded_cfg = alert_engine.get_channel_config()
        self.assertEqual(loaded_cfg["telegram"]["bot_token"], "mock_test_token")
        self.assertEqual(loaded_cfg["telegram"]["chat_id"], "mock_test_chat")

    def test_evaluate_static_price_alert(self):
        # Starts at 100 and ends around 115
        df = create_mock_candles(30, start_price=100.0, trend=0.5)
        last_price = float(df["close"].iloc[-1])
        target = last_price - 0.2  # Price crossed above target

        alert = {
            "alert_id": 999,
            "symbol": "TEST",
            "alert_type": "STATIC_PRICE",
            "condition": {"target_price": target, "operator": ">="}
        }
        res = alert_engine.evaluate_single_alert(alert, df)
        self.assertIsNotNone(res)
        trig_price, headline, details = res
        self.assertIn("crossed above", headline)

    def test_evaluate_horizontal_sr_alert(self):
        df = create_mock_candles(30, start_price=200.0, trend=1.0)
        last_price = float(df["close"].iloc[-1])
        level = last_price - 0.5

        alert = {
            "alert_id": 998,
            "symbol": "TEST_SR",
            "alert_type": "TRENDLINE_SR",
            "condition": {"sr_type": "Horizontal", "kind": "Resistance", "level": level}
        }
        res = alert_engine.evaluate_single_alert(alert, df)
        self.assertIsNotNone(res)
        trig_price, headline, details = res
        self.assertIn("Resistance", headline)

    def test_evaluate_indicator_rsi_alert(self):
        df = create_mock_candles(40, start_price=100.0, trend=1.5)
        alert = {
            "alert_id": 997,
            "symbol": "TEST_RSI",
            "alert_type": "INDICATOR",
            "condition": {"rule": "RSI_Level", "timeframe": "Daily", "span": 14, "threshold": 50.0, "operator": ">"}
        }
        res = alert_engine.evaluate_single_alert(alert, df)
        # Should evaluate without exception
        if res is not None:
            trig_p, head, det = res
            self.assertIn("RSI", head)

    def test_is_market_hours_ist(self):
        """Test Indian Market Alert Timing: 09:00 AM - 04:00 PM IST (Mon-Fri)."""
        # Monday 10:30 IST (Open)
        t_mon_open = datetime(2026, 9, 14, 10, 30, 0, tzinfo=alert_engine.IST)
        self.assertTrue(alert_engine.is_market_hours_ist(t_mon_open))

        # Monday 08:30 IST (Before 9 AM -> False)
        t_mon_early = datetime(2026, 9, 14, 8, 30, 0, tzinfo=alert_engine.IST)
        self.assertFalse(alert_engine.is_market_hours_ist(t_mon_early))

        # Monday 16:30 IST (After 4 PM -> False)
        t_mon_late = datetime(2026, 9, 14, 16, 30, 0, tzinfo=alert_engine.IST)
        self.assertFalse(alert_engine.is_market_hours_ist(t_mon_late))

        # Saturday 11:00 IST (Weekend -> False)
        t_sat = datetime(2026, 9, 12, 11, 0, 0, tzinfo=alert_engine.IST)
        self.assertFalse(alert_engine.is_market_hours_ist(t_sat))

    def test_dispatch_alert_market_hours_enforcement(self):
        """Test that automated alerts outside 9 AM - 4 PM IST are suppressed unless ignore_market_hours=True."""
        # Simulated late evening
        late_dt = datetime(2026, 9, 14, 21, 0, 0, tzinfo=alert_engine.IST)
        if not alert_engine.is_market_hours_ist(late_dt):
            # Without ignore_market_hours outside market hours, it should be suppressed
            # If current real time is outside market hours:
            if not alert_engine.is_market_hours_ist():
                res = alert_engine.dispatch_alert(
                    symbol="TEST_HOURS",
                    alert_type="STATIC_PRICE",
                    trigger_price=100.0,
                    headline="Test",
                    details="Test",
                    selected_channels=["in_app"],
                    ignore_market_hours=False
                )
                self.assertIn("market_hours", res)
                self.assertFalse(res["market_hours"][0])

            # With ignore_market_hours=True, it succeeds
            res_forced = alert_engine.dispatch_alert(
                symbol="TEST_HOURS",
                alert_type="STATIC_PRICE",
                trigger_price=100.0,
                headline="Test Forced",
                details="Test Forced",
                selected_channels=["in_app"],
                ignore_market_hours=True
            )
            self.assertTrue(res_forced["in_app"][0])


if __name__ == "__main__":
    unittest.main()
