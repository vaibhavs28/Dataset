# 📈 Upstox Technical Stock Scanner & Local Database Dashboard

A stock market scanner and interactive dashboard built with **Python**, **Upstox API v2**, **SQLite**, and **Streamlit**.

It stores historical market data locally on your PC so you can scan hundreds of stocks across multiple timeframes (Monthly, Weekly, Daily) with zero API rate-limit delays.

---

## 🌟 Key Features

1. **Multi-Timeframe Scanning**:
   - **Monthly Close > 5 EMA** (Default requested scan)
   - Weekly & Daily timeframes with configurable periods (e.g. 5 EMA, 9 EMA, 20 EMA, 50 SMA).
   - Rules: `Close > Indicator`, `Close < Indicator`, `Bullish Crossover`, `Bearish Crossover`.
2. **Local SQLite Database**:
   - Stores daily OHLCV bars locally in `data/market_data.db`.
   - Dynamic resampling into Monthly / Weekly candles on-the-fly.
   - Run scans instantly offline without burning Upstox API quotas.
3. **Interactive Streamlit Web Dashboard**:
   - Live sortable results table with color gradients based on % distance from EMA.
   - Interactive TradingView-style Candlestick chart with EMA line and volume subplots.
   - One-click CSV export of scan results.
4. **Upstox API Integration & Data Sync**:
   - OAuth 2.0 login link generator + token storage.
   - Direct token paste support.
   - Incremental candle sync (only fetches new days).
   - Built-in realistic mock data seeder so you can test immediately even without API keys!

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /Users/shilpashingnapure/.gemini/antigravity/scratch/upstox_scanner
python3 -m pip install -r requirements.txt
```

### 2. Launch Dashboard
```bash
streamlit run app.py
```
This opens the dashboard at `http://localhost:8501` in your browser.

---

## 🔑 Upstox API Setup (Optional for Live Data)

To download live NSE data from Upstox:
1. Log in to the [Upstox Developer Portal](https://developer.upstox.com/) and create an app.
2. Note your **API Key**, **API Secret**, and set Redirect URI (e.g. `https://127.0.0.1:5000/`).
3. In the dashboard sidebar under **"Upstox API Credentials"**:
   - Enter your `API Key` and `API Secret`.
   - Click **"Click here to Login with Upstox"** -> Authenticate -> Copy the `code=` parameter from the URL bar into the dashboard and click **Generate Token**.
   - Or, generate your token in Upstox Developer console and paste it into **"Upstox Access Token"**.
4. In the sidebar under **"Download / Sync Market Data"**, select **Nifty 50** and click **Start Upstox Download**.

---

## 🧪 Testing with Mock Data (Zero Setup)
If you want to immediately see how the scanner and charts look:
1. Open the dashboard.
2. Expand **"Demo / Mock Data"** in the sidebar.
3. Click **"Seed Demo Data (20 Stocks)"**.
4. The scanner immediately populates with 3 years of daily candles for 20 large-cap stocks.

---

## 📁 Project Architecture

- `app.py`: Streamlit frontend with controls, tables, and Plotly charts.
- `scanner.py`: Resampling engine (`resample('ME')`) and indicator calculation (`Close > 5 EMA`).
- `database.py`: SQLite schema and queries for instruments and daily candles.
- `downloader.py`: Upstox historical candles API sync + mock data generator.
- `instruments.py`: NSE instrument master fetcher and symbol resolver.
- `auth.py`: Upstox OAuth login URL generator and token manager.
- `config.py`: Configuration variables and default Nifty 50 watchlist.
