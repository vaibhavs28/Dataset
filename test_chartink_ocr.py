"""
test_chartink_ocr.py
--------------------
Unit tests for Chartink screenshot OCR and intelligent rule parsing.
"""

import io
import unittest
from PIL import Image, ImageDraw

from chartink_ocr import (
    clean_chartink_ocr_text,
    find_tesseract_binary,
    parse_chartink_screenshot,
    extract_text_from_image,
)
from screener_engine import parse_chartink_query


class TestChartinkOCR(unittest.TestCase):

    def test_find_tesseract(self):
        bin_path = find_tesseract_binary()
        # Should find tesseract on Mac or Linux if installed
        self.assertIsNotNone(bin_path)

    def test_clean_chartink_ocr_text_logic(self):
        text_all = """
        Stock passes all of the below filters in cash segment:
        [ Latest ] Close Greater than [ Latest ] 20 EMA
        Run Scan
        """
        logic, lines = clean_chartink_ocr_text(text_all)
        self.assertEqual(logic, "ALL")
        self.assertEqual(len(lines), 1)
        self.assertIn("Close Greater than", lines[0])

        text_any = """
        Stock passes any of the below filters in cash segment:
        [ Latest ] RSI 14 Greater than 70
        [ Weekly ] Close Greater than [ Weekly ] 200 EMA
        Submit
        """
        logic, lines = clean_chartink_ocr_text(text_any)
        self.assertEqual(logic, "ANY")
        self.assertEqual(len(lines), 2)

    def test_parse_chartink_query_complex(self):
        query = """
        [ Latest ] Close Greater than [ Latest ] 20 EMA
        [ Latest ] Volume Greater than [ Latest ] 20 SMA Volume * 1.5
        [ Weekly ] Close Crossed above [ Weekly ] 200 EMA
        [ Monthly ] 5 EMA Greater than [ Monthly ] 20 EMA
        Latest RSI(14) Greater than 60
        Open <= 20 EMA and Close >= 20 EMA
        """
        clauses = parse_chartink_query(query)
        self.assertGreaterEqual(len(clauses), 6)

        # Check multiplier
        vol_clause = [c for c in clauses if c.lhs == "Volume"]
        self.assertTrue(len(vol_clause) > 0)
        self.assertEqual(vol_clause[0].multiplier, 1.5)
        self.assertEqual(vol_clause[0].rhs_indicator, "Vol_SMA_20")

        # Check crossed above
        cross_clause = [c for c in clauses if c.operator == "crossed_above"]
        self.assertTrue(len(cross_clause) > 0)
        self.assertEqual(cross_clause[0].timeframe, "Weekly")

        # Check Monthly 5 EMA > 20 EMA
        m_clause = [c for c in clauses if c.timeframe == "Monthly"]
        self.assertTrue(len(m_clause) > 0)
        self.assertEqual(m_clause[0].lhs, "EMA_5")

    def test_end_to_end_synthetic_screenshot(self):
        # Create image with text
        img = Image.new("RGB", (800, 240), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = [
            "Stock passes all of the below filters in cash segment:",
            "[ Latest ] Close Greater than [ Latest ] 20 EMA",
            "[ Latest ] RSI(14) Greater than 60",
            "[ Weekly ] Close Greater than [ Weekly ] 200 EMA",
        ]
        y = 20
        for l in lines:
            draw.text((30, y), l, fill=(0, 0, 0))
            y += 40

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        result = parse_chartink_screenshot(buf)
        self.assertTrue(result["success"])
        self.assertEqual(result["logic"], "ALL")
        self.assertGreaterEqual(len(result["clauses"]), 2)


if __name__ == "__main__":
    unittest.main()
