#!/bin/bash
# stop_247.sh - Stops the 24*7 Upstox Market Platform and Daemon

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$DIR/logs/processes.pid"

echo "Stopping 24*7 processes..."

if [ -f "$PID_FILE" ]; then
    while read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "python3 daemon_runner.py" 2>/dev/null || true

echo "✓ All 24*7 services have been stopped."
