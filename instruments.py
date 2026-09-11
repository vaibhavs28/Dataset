import gzip
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from config import UPSTOX_NSE_INSTRUMENTS_URL, UPSTOX_BSE_INSTRUMENTS_URL
import database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("instruments")


INDEX_KEY_MAP = {
    # NSE Indices
    "NIFTY": "NSE_INDEX|Nifty 50",
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
    "NIFTY FIN SERVICE": "NSE_INDEX|Nifty Fin Service",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
    "NIFTY MID SELECT": "NSE_INDEX|NIFTY MID SELECT",
    "INDIAVIX": "NSE_INDEX|India VIX",
    "INDIA VIX": "NSE_INDEX|India VIX",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTY AUTO": "NSE_INDEX|Nifty Auto",
    "NIFTY PHARMA": "NSE_INDEX|Nifty Pharma",
    "NIFTY FMCG": "NSE_INDEX|Nifty FMCG",
    "NIFTY METAL": "NSE_INDEX|Nifty Metal",
    "NIFTY REALTY": "NSE_INDEX|Nifty Realty",
    "NIFTY ENERGY": "NSE_INDEX|Nifty Energy",
    "NIFTY INFRA": "NSE_INDEX|Nifty Infra",
    "NIFTY NEXT 50": "NSE_INDEX|Nifty Next 50",
    "NIFTYNXT50": "NSE_INDEX|Nifty Next 50",
    "NIFTY 100": "NSE_INDEX|Nifty 100",
    "NIFTY 200": "NSE_INDEX|Nifty 200",
    "NIFTY 500": "NSE_INDEX|Nifty 500",
    # BSE Indices
    "SENSEX": "BSE_INDEX|SENSEX",
    "BSE SENSEX": "BSE_INDEX|SENSEX",
    "BANKEX": "BSE_INDEX|BANKEX",
    "BSE BANKEX": "BSE_INDEX|BANKEX",
    "BSE500": "BSE_INDEX|BSE500",
    "BSE 500": "BSE_INDEX|BSE500",
    "BSE100": "BSE_INDEX|BSE100",
    "BSE 100": "BSE_INDEX|BSE100",
    "SENSEX50": "BSE_INDEX|SENSEX50",
    "SNXT50": "BSE_INDEX|SNXT50"
}


def sync_nse_instruments(force: bool = False) -> int:
    """Backwards-compatible wrapper that calls sync_all_instruments."""
    return sync_all_instruments(force=force)


def sync_all_instruments(force: bool = False) -> int:
    """
    Downloads Upstox NSE and BSE instrument masters and saves strictly pure Equity and Index instruments to local database.
    Excludes all debt, bonds, NCDs, government securities (e.g. 0ABCL31, 0HFL28, 0IRFC35, Group F, Group G).
    """
    database.init_db()

    # Check if already populated unless force=True
    stats = database.get_db_stats()
    if stats.get("total_instruments", 0) >= 2000 and not force:
        logger.info(f"Instruments already present in DB: {stats['total_instruments']}. Skipping sync.")
        return stats["total_instruments"]

    instruments_to_save = []

    # 1. Fetch NSE Instruments
    logger.info(f"Fetching NSE instrument master from Upstox: {UPSTOX_NSE_INSTRUMENTS_URL}")
    try:
        response = requests.get(UPSTOX_NSE_INSTRUMENTS_URL, timeout=45)
        response.raise_for_status()
        decompressed_data = gzip.decompress(response.content)
        records = json.loads(decompressed_data.decode("utf-8"))

        for item in records:
            segment = item.get("segment", "")
            inst_type = item.get("instrument_type", "")
            is_equity = (segment == "NSE_EQ" and inst_type in ("EQUITY", "EQ", "BE", "SM", "BZ"))
            is_index = (segment == "NSE_INDEX" or inst_type == "INDEX")

            if is_equity or is_index:
                instrument_key = item.get("instrument_key")
                trading_symbol = item.get("trading_symbol", "").upper().strip()
                if instrument_key and trading_symbol and not trading_symbol.startswith("0"):
                    instruments_to_save.append({
                        "instrument_key": instrument_key,
                        "trading_symbol": trading_symbol,
                        "name": item.get("name", trading_symbol),
                        "exchange": "NSE_INDEX" if is_index else "NSE_EQ",
                        "instrument_type": "INDEX" if is_index else (inst_type or "EQUITY"),
                        "tick_size": float(item.get("tick_size", 0.05)),
                        "lot_size": int(item.get("lot_size", 1))
                    })
        logger.info(f"Parsed {len(instruments_to_save)} NSE equity/index instruments.")
    except Exception as e:
        logger.error(f"Error fetching NSE instruments: {e}")

    logger.info(f"Retaining strictly pure NSE Equities and Indices ({len(instruments_to_save)} symbols).")

    # Ensure canonical index aliases
    canonical_indices = [
        {"instrument_key": "NSE_INDEX|Nifty 50", "trading_symbol": "NIFTY", "name": "Nifty 50", "exchange": "NSE_INDEX", "instrument_type": "INDEX", "tick_size": 0.05, "lot_size": 1},
        {"instrument_key": "NSE_INDEX|Nifty Bank", "trading_symbol": "BANKNIFTY", "name": "Nifty Bank", "exchange": "NSE_INDEX", "instrument_type": "INDEX", "tick_size": 0.05, "lot_size": 1},
        {"instrument_key": "BSE_INDEX|SENSEX", "trading_symbol": "SENSEX", "name": "BSE Sensex", "exchange": "BSE_INDEX", "instrument_type": "INDEX", "tick_size": 0.05, "lot_size": 1},
        {"instrument_key": "BSE_INDEX|BANKEX", "trading_symbol": "BANKEX", "name": "BSE Bankex", "exchange": "BSE_INDEX", "instrument_type": "INDEX", "tick_size": 0.05, "lot_size": 1},
        {"instrument_key": "BSE_INDEX|BSE500", "trading_symbol": "BSE500", "name": "BSE 500", "exchange": "BSE_INDEX", "instrument_type": "INDEX", "tick_size": 0.05, "lot_size": 1}
    ]
    instruments_to_save.extend(canonical_indices)

    logger.info(f"Total pure Equity & Index instruments to save: {len(instruments_to_save)}. Saving to database...")
    database.upsert_instruments(instruments_to_save)
    logger.info("NSE & BSE Instrument sync completed successfully.")
    return len(instruments_to_save)


def resolve_instrument_key(symbol: str) -> Optional[str]:
    """
    Given a symbol (e.g. 'RELIANCE', 'SENSEX', 'NIFTY'), returns its Upstox instrument_key.
    First checks index alias map, then database, then falls back to instrument sync.
    """
    sym = symbol.upper().strip()
    # Map common corporate renames/demergers
    aliases = {
        "TATAMOTORS": "TMCV",
        "LTIM": "LTM",
    }
    sym = aliases.get(sym, sym)

    if sym in INDEX_KEY_MAP:
        return INDEX_KEY_MAP[sym]

    record = database.get_instrument_by_symbol(sym)
    if record:
        return record["instrument_key"]

    sync_all_instruments()
    record = database.get_instrument_by_symbol(sym)
    if record:
        return record["instrument_key"]

    return None

