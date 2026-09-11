#!/bin/bash
# ==============================================================================
# start_247.sh - 24*7 Automated Runner for Upstox Market Engine & Web Dashboard
# Works on macOS and Linux (including Home Server 192.168.0.100)
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

mkdir -p "$DIR/data"
mkdir -p "$DIR/logs"

WEB_LOG="$DIR/logs/streamlit_web.log"
DAEMON_LOG="$DIR/logs/daemon_engine.log"
PID_FILE="$DIR/logs/processes.pid"

echo "================================================================="
echo "  🚀 STARTING 24*7 UPSTOX MARKET PLATFORM & DAEMON"
echo "================================================================="

# 1. Check if database exists, auto-download if needed
if [ ! -f "$DIR/data/market_data.duckdb" ]; then
    echo "⚡ DuckDB database not found locally. Downloading full dataset from GitHub..."
    python3 download_dataset.py
fi

# 2. Stop any existing running instances
echo "Checking for running instances..."
pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "python3 daemon_runner.py" 2>/dev/null || true
sleep 1

# 3. Start 24*7 Market Data Daemon in Background
echo "Starting 24*7 Background Market Data Daemon..."
nohup python3 daemon_runner.py 15 >> "$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
echo "✓ 24*7 Sync Daemon started (PID: $DAEMON_PID) -> logging to $DAEMON_LOG"

# 4. Start Streamlit Web Dashboard in Background
echo "Starting Streamlit Web Dashboard on port 8501..."
nohup streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    >> "$WEB_LOG" 2>&1 &
WEB_PID=$!
echo "✓ Streamlit Dashboard started (PID: $WEB_PID) -> logging to $WEB_LOG"

# Save PIDs
echo "$DAEMON_PID" > "$PID_FILE"
echo "$WEB_PID" >> "$PID_FILE"

sleep 2

echo "================================================================="
echo "  ✅ 24*7 SERVICES ARE RUNNING SUCCESSFULLY!"
echo "================================================================="
echo "  • Web Dashboard URL: http://localhost:8501 (or http://<server-ip>:8501)"
echo "  • Market Sync Daemon: Active 24*7 (auto-syncs on market hours)"
echo "  • Log Files:"
echo "      Dashboard Log: $WEB_LOG"
echo "      Daemon Log:    $DAEMON_LOG"
echo "  • To stop all services: ./stop_247.sh"
echo "================================================================="
