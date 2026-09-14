"""
test_crypto_system.py
---------------------
Unit tests for the dedicated Crypto system (crypto_store.py, delta_exchange_client.py).
Verifies isolated database, candle storage, live ticker management, and screener logic.
"""

import os
import unittest
import pandas as pd
from datetime import datetime

import crypto_store
import delta_exchange_client
import scanner


class TestCryptoSystem(unittest.TestCase):

    def setUp(self):
        crypto_store.seed_crypto_initial_data_if_empty()

    def test_database_is_separate(self):
        # Database must be in data/crypto_market.duckdb and exist
        self.assertTrue(os.path.exists(crypto_store.CRYPTO_DB_PATH))
        self.assertIn("crypto_market.duckdb", crypto_store.CRYPTO_DB_PATH)

    def test_supported_symbols_only(self):
        # Only BTCUSD and ETHUSD supported
        self.assertEqual(sorted(crypto_store.SUPPORTED_CRYPTO_SYMBOLS), ["BTCUSD", "ETHUSD"])

    def test_get_crypto_candles(self):
        for sym in ["BTCUSD", "ETHUSD"]:
            df_d = crypto_store.get_crypto_candles(sym, "1d", limit=30)
            self.assertFalse(df_d.empty, f"Expected non-empty daily candles for {sym}")
            self.assertTrue(isinstance(df_d.index, pd.DatetimeIndex))
            for col in ["open", "high", "low", "close", "volume"]:
                self.assertIn(col, df_d.columns)
            # High >= Low
            self.assertTrue((df_d["high"] >= df_d["low"]).all())

    def test_crypto_tickers(self):
        for sym in ["BTCUSD", "ETHUSD"]:
            t = crypto_store.get_crypto_ticker(sym)
            self.assertIsNotNone(t, f"Ticker should exist for {sym}")
            self.assertEqual(t["symbol"], sym)
            self.assertGreater(t["last_price"], 0.0)
            self.assertGreater(t["high_24h"], 0.0)
            self.assertGreater(t["low_24h"], 0.0)

    def test_upsert_candles(self):
        now_ts = int(datetime.utcnow().timestamp())
        df_test = pd.DataFrame([{
            "dt": pd.to_datetime(now_ts, unit="s"),
            "open": 95000.0,
            "high": 96000.0,
            "low": 94500.0,
            "close": 95800.0,
            "volume": 1200.0
        }]).set_index("dt")

        inserted = crypto_store.upsert_crypto_candles(df_test, "BTCUSD", "1h")
        self.assertEqual(inserted, 1)

        df_fetched = crypto_store.get_crypto_candles("BTCUSD", "1h", limit=5)
        self.assertFalse(df_fetched.empty)

    def test_crypto_indicators_computation(self):
        df_d = crypto_store.get_crypto_candles("BTCUSD", "1d", limit=100)
        c = df_d["close"]
        rsi = scanner.calculate_rsi(c, 14)
        ema20 = scanner.calculate_ema(c, 20)
        ema50 = scanner.calculate_ema(c, 50)
        ema200 = scanner.calculate_ema(c, 200)

        self.assertFalse(rsi.empty)
        self.assertFalse(ema20.empty)
        self.assertFalse(ema50.empty)
        self.assertFalse(ema200.empty)
        self.assertTrue(0 <= rsi.iloc[-1] <= 100)

    def test_delta_exchange_client_resilience(self):
        # Even if network is restricted in sandbox, client gracefully returns local candles
        success, df, note = delta_exchange_client.fetch_delta_candles("BTCUSD", "1d", limit=10)
        self.assertTrue(success)
        self.assertFalse(df.empty)

        success_t, ticker, note_t = delta_exchange_client.fetch_delta_ticker("BTCUSD")
        self.assertTrue(success_t)
        self.assertIn("last_price", ticker)


if __name__ == "__main__":
    unittest.main()
