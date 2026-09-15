"""
test_auto_nifty500_updater.py
------------------------------
Unit tests for auto_nifty500_updater module.
"""

import unittest
from datetime import datetime, time as dtime
import pytz
import auto_nifty500_updater

IST = pytz.timezone("Asia/Kolkata")

class TestAutoNifty500Updater(unittest.TestCase):

    def test_market_hours_ist(self):
        # Test Monday 10:15 IST (Market Open)
        dt_market_open = IST.localize(datetime(2026, 9, 14, 10, 15, 0)) # Monday
        self.assertTrue(auto_nifty500_updater.is_market_hours_ist(dt_market_open))

        # Test Monday 08:45 IST (Before Market Open)
        dt_pre_market = IST.localize(datetime(2026, 9, 14, 8, 45, 0))
        self.assertFalse(auto_nifty500_updater.is_market_hours_ist(dt_pre_market))

        # Test Monday 16:05 IST (After Market Window)
        dt_post_market = IST.localize(datetime(2026, 9, 14, 16, 5, 0))
        self.assertFalse(auto_nifty500_updater.is_market_hours_ist(dt_post_market))

        # Test Saturday 11:00 IST (Weekend)
        dt_saturday = IST.localize(datetime(2026, 9, 19, 11, 0, 0))
        self.assertFalse(auto_nifty500_updater.is_market_hours_ist(dt_saturday))

        # Test Sunday 14:00 IST (Weekend)
        dt_sunday = IST.localize(datetime(2026, 9, 20, 14, 0, 0))
        self.assertFalse(auto_nifty500_updater.is_market_hours_ist(dt_sunday))

    def test_get_nifty_500_symbols(self):
        symbols = auto_nifty500_updater.get_nifty_500_symbols()
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)
        self.assertLessEqual(len(symbols), 500)
        # Should include core bluechips
        self.assertIn("RELIANCE", symbols)
        self.assertIn("TCS", symbols)
        self.assertIn("HDFCBANK", symbols)

    def test_status_file_rw(self):
        test_payload = {
            "status": "TEST_RUN",
            "total_symbols": 500,
            "synced_symbols": 500,
            "elapsed_seconds": 24.5
        }
        auto_nifty500_updater.update_status_file(test_payload)
        read_back = auto_nifty500_updater.get_sync_status()
        self.assertEqual(read_back["status"], "TEST_RUN")
        self.assertEqual(read_back["total_symbols"], 500)

if __name__ == "__main__":
    unittest.main()
