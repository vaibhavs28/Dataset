#!/usr/bin/env python3
"""
Purges all Index, ETF, and Debt/Bond records from market_data.duckdb,
retaining strictly 100% pure NSE corporate equities.
"""
import logging
import time
import duckdb_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("purge_non_equities")

def purge_all():
    logger.info("🧹 Starting complete purge of non-equities (Indices, ETFs, Debt, Bonds)...")
    t0 = time.time()
    conn = duckdb_store.get_write_connection()
    try:
        # 1. Purge instruments
        logger.info("Purging non-equity instruments...")
        conn.execute("""
            CREATE TABLE instruments_pure AS 
            SELECT * FROM instruments 
            WHERE exchange = 'NSE_EQ' 
              AND instrument_key LIKE '%|INE%'
              AND instrument_type IN ('EQ', 'BE', 'SM', 'BZ', 'EQUITY')
              AND trading_symbol NOT LIKE '0%'
              AND trading_symbol NOT LIKE '%ETF%' 
              AND trading_symbol NOT LIKE '%BEES%' 
              AND trading_symbol NOT LIKE '%NIFTY%' 
              AND trading_symbol NOT LIKE '%SENSEX%' 
              AND trading_symbol NOT LIKE 'INDIA VIX%';
            DROP TABLE instruments;
            ALTER TABLE instruments_pure RENAME TO instruments;
        """)

        # 2. Purge daily_candles
        logger.info("Purging non-equity daily_candles...")
        conn.execute("""
            CREATE TABLE daily_candles_pure AS 
            SELECT * FROM daily_candles 
            WHERE trading_symbol NOT LIKE '%ETF%' 
              AND trading_symbol NOT LIKE '%BEES%' 
              AND trading_symbol NOT LIKE '%NIFTY%' 
              AND trading_symbol NOT LIKE '%SENSEX%' 
              AND trading_symbol NOT LIKE 'INDIA VIX%' 
              AND trading_symbol NOT LIKE '0%'
              AND instrument_key NOT LIKE 'NSE_INDEX%'
              AND instrument_key NOT LIKE '%|INF%';
            DROP TABLE daily_candles;
            ALTER TABLE daily_candles_pure RENAME TO daily_candles;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_uniq ON daily_candles (instrument_key, date);
            CREATE INDEX IF NOT EXISTS idx_daily_sym_date ON daily_candles (trading_symbol, date);
        """)

        # 3. Purge intraday_candles
        logger.info("Purging non-equity intraday_candles...")
        conn.execute("""
            CREATE TABLE intraday_candles_pure AS 
            SELECT * FROM intraday_candles 
            WHERE trading_symbol NOT LIKE '%ETF%' 
              AND trading_symbol NOT LIKE '%BEES%' 
              AND trading_symbol NOT LIKE '%NIFTY%' 
              AND trading_symbol NOT LIKE '%SENSEX%' 
              AND trading_symbol NOT LIKE 'INDIA VIX%' 
              AND trading_symbol NOT LIKE '0%'
              AND instrument_key NOT LIKE 'NSE_INDEX%'
              AND instrument_key NOT LIKE '%|INF%';
            DROP TABLE intraday_candles;
            ALTER TABLE intraday_candles_pure RENAME TO intraday_candles;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_intraday_uniq ON intraday_candles (instrument_key, timeframe, timestamp);
            CREATE INDEX IF NOT EXISTS idx_intraday_sym_tf_ts ON intraday_candles (trading_symbol, timeframe, timestamp);
        """)

        # 4. Purge stocks (if exists)
        has_stocks = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'stocks';").fetchone()[0]
        if has_stocks > 0:
            logger.info("Purging non-equity records from stocks table...")
            conn.execute("""
                CREATE TABLE stocks_pure AS 
                SELECT * FROM stocks 
                WHERE ticker NOT LIKE '%ETF%' 
                  AND ticker NOT LIKE '%BEES%' 
                  AND ticker NOT LIKE '%NIFTY%' 
                  AND ticker NOT LIKE '%SENSEX%' 
                  AND ticker NOT LIKE 'INDIA VIX%' 
                  AND ticker NOT LIKE '0%';
                DROP TABLE stocks;
                ALTER TABLE stocks_pure RENAME TO stocks;
                CREATE INDEX IF NOT EXISTS idx_stocks_ticker_dt ON stocks (ticker, datetime);
            """)

        logger.info("Optimizing and compacting DuckDB database pages (CHECKPOINT & VACUUM)...")
        conn.execute("CHECKPOINT;")
        try:
            conn.execute("VACUUM;")
        except Exception:
            pass

        rem_inst = conn.execute("SELECT count(*) FROM instruments;").fetchone()[0]
        rem_daily = conn.execute("SELECT count(*) FROM daily_candles;").fetchone()[0]
        rem_intra = conn.execute("SELECT count(*) FROM intraday_candles;").fetchone()[0]
        logger.info(f"🎉 Purge complete in {time.time()-t0:.2f}s!")
        logger.info(f"📊 Remaining 100% Pure Equities: {rem_inst:,} instruments | {rem_daily:,} daily bars | {rem_intra:,} 75m bars")
    finally:
        conn.close()

if __name__ == "__main__":
    purge_all()
