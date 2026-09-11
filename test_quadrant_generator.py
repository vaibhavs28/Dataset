"""
test_quadrant_generator.py
---------------------------
Automated test suite for the 75-Minute Waterfall Alert & Quadrant Screenshot Engine.
Verifies:
1. Schedule calculation & countdown timers for 75-min candle closes.
2. Headless 1600x1200 4-quadrant PNG rendering.
3. Waterfall scan & alert dispatch pipeline with image attachment.
4. Broadcast history logging & screenshot gallery retrieval.
"""

import os
import unittest
from datetime import datetime
from PIL import Image

import config
import auto_75m_broadcaster
import quadrant_image_generator
import alert_engine


class TestQuadrantGenerator(unittest.TestCase):

    def test_01_schedule_calculation(self):
        """Test 75-min schedule status calculation."""
        sched = auto_75m_broadcaster.get_75m_schedule_status()
        self.assertIn("next_candle_label", sched)
        self.assertIn("seconds_remaining", sched)
        self.assertIn("time_remaining_str", sched)
        self.assertGreaterEqual(sched["seconds_remaining"], 0)
        self.assertIn(sched["candle_idx"], [1, 2, 3, 4, 5])

        # Test specific time simulation (e.g. 10:15 on a Monday)
        sim_time = datetime(2026, 9, 14, 10, 15, 0)
        sim_sched = auto_75m_broadcaster.get_75m_schedule_status(sim_time)
        self.assertTrue(sim_sched["is_market_hours"])
        self.assertEqual(sim_sched["candle_idx"], 1)
        self.assertEqual(sim_sched["seconds_remaining"], 15 * 60)

    def test_02_quadrant_image_generation(self):
        """Test generation of 1600x1200 4-quadrant screenshot for RELIANCE."""
        img_path = quadrant_image_generator.generate_stock_quadrant(
            symbol="RELIANCE",
            stage_label="🏆 STAGE 4 FULL ALIGNMENT QUALIFIED"
        )
        self.assertIsNotNone(img_path)
        self.assertTrue(os.path.exists(img_path))

        # Check image dimensions
        with Image.open(img_path) as im:
            self.assertEqual(im.size, (1600, 1200))
            self.assertEqual(im.format, "PNG")

        # Check non-empty file size (> 10KB)
        file_size = os.path.getsize(img_path)
        self.assertGreater(file_size, 10_000)

    def test_03_recent_quadrant_screenshots(self):
        """Test retrieval of generated screenshots for UI gallery."""
        screens = auto_75m_broadcaster.get_recent_quadrant_screenshots(limit=5)
        self.assertIsInstance(screens, list)
        self.assertGreaterEqual(len(screens), 1)
        self.assertEqual(screens[0]["symbol"], "RELIANCE")
        self.assertTrue(os.path.exists(screens[0]["path"]))

    def test_04_broadcast_pipeline_sample(self):
        """Test waterfall broadcast execution on sample symbols."""
        bc_res = auto_75m_broadcaster.run_75m_waterfall_broadcast(
            universe="Nifty 50",
            stage_filter=4,
            symbols=["RELIANCE", "TCS"],
            channels=["in_app"]
        )
        self.assertIn("broadcast_id", bc_res)
        self.assertIn("scanned_count", bc_res)
        self.assertEqual(bc_res["scanned_count"], 2)

        history = auto_75m_broadcaster.get_broadcast_history(limit=5)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["broadcast_id"], bc_res["broadcast_id"])


if __name__ == "__main__":
    unittest.main()
