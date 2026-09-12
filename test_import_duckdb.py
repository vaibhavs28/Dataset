"""
test_import_duckdb.py
---------------------
Unit tests for external DuckDB inspection and merge utility.
"""

import unittest
import tempfile
from pathlib import Path
import duckdb

import duckdb_store
import import_duckdb


class TestImportDuckDB(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ext_db_path = Path(self.temp_dir.name) / "test_external.duckdb"

        # Create a sample external duckdb file
        conn = duckdb.connect(str(self.ext_db_path))
        conn.execute("""
            CREATE TABLE test_daily (
                symbol VARCHAR,
                date VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            );
        """)
        conn.execute("""
            INSERT INTO test_daily VALUES
            ('DEMO_STOCK', '2026-09-11', 100.0, 105.0, 98.0, 102.5, 50000),
            ('DEMO_STOCK', '2026-09-12', 103.0, 107.0, 102.0, 106.0, 60000);
        """)
        conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspect_external_duckdb(self):
        info = import_duckdb.inspect_external_duckdb(str(self.ext_db_path))
        self.assertIn("test_daily", info["tables"])
        self.assertEqual(info["tables"]["test_daily"]["row_count"], 2)
        self.assertIn("close", info["tables"]["test_daily"]["columns"])

    def test_merge_external_duckdb(self):
        res = import_duckdb.merge_external_duckdb(str(self.ext_db_path))
        self.assertIn("test_daily", res["tables"])
        self.assertEqual(res["tables"]["test_daily"]["status"], "merged")

        # Verify data accessible in duckdb_store
        df = duckdb_store.get_candles_df("DEMO_STOCK")
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["close"].iloc[-1], 106.0)


if __name__ == "__main__":
    unittest.main()
