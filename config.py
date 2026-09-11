import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Database configuration
DB_PATH = DATA_DIR / "market_data.db"

# Upstox API Credentials
UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "https://127.0.0.1:5000/")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# Upstox API Endpoints
UPSTOX_AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
UPSTOX_BASE_URL = "https://api.upstox.com/v2"
UPSTOX_NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_BSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"

# Default Curated Watchlists (Nifty 50 constituents for fast starter scans)
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "SBIN",
    "HINDUNILVR", "ITC", "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS", "NTPC", "AXISBANK",
    "ADANIPORTS", "POWERGRID", "M&M", "COALINDIA", "BAJAJFINSV", "ULTRACEMCO",
    "NESTLEIND", "WIPRO", "JSWSTEEL", "GRASIM", "TECHM", "TATASTEEL", "HINDALCO",
    "SBILIFE", "DRREDDY", "BAJAJ-AUTO", "CIPLA", "EICHERMOT", "BPCL", "HEROMOTOCO",
    "LTIM", "APOLLOHOSP", "DIVISLAB", "TATACONSUM", "SHRIRAMFIN", "BRITANNIA",
    "BEL", "TRENT", "ASIANPAINT"
]
