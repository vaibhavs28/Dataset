#!/bin/bash
# install_systemd.sh - Installs and activates 24*7 systemd services on Linux home server

set -e

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./systemd/install_systemd.sh)"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$DIR/.." && pwd )"

# Auto-detect venv or fallback
if [ -f "$REPO_DIR/venv/bin/streamlit" ]; then
    VENV_DIR="$REPO_DIR/venv"
elif [ -f "/home/trader/Dataset/venv/bin/streamlit" ]; then
    VENV_DIR="/home/trader/Dataset/venv"
else
    VENV_DIR=""
fi

echo "=========================================================="
echo "Installing 24*7 systemd services..."
if [ -n "$VENV_DIR" ]; then
    echo "✓ Detected Python Virtual Environment at: $VENV_DIR"
    PYTHON_EXEC="$VENV_DIR/bin/python3"
    STREAMLIT_EXEC="$VENV_DIR/bin/streamlit"
else
    PYTHON_EXEC="/usr/bin/python3"
    STREAMLIT_EXEC="/usr/local/bin/streamlit"
fi

cat <<EOF > /etc/systemd/system/upstox-web.service
[Unit]
Description=Upstox Streamlit Market Dashboard (24*7)
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=$REPO_DIR
ExecStart=$STREAMLIT_EXEC run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

cat <<EOF > /etc/systemd/system/upstox-daemon.service
[Unit]
Description=Upstox 24*7 Market Data Sync Daemon
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=$REPO_DIR
ExecStart=$PYTHON_EXEC daemon_runner.py 15
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable upstox-web.service
systemctl enable upstox-daemon.service

systemctl restart upstox-web.service
systemctl restart upstox-daemon.service

echo "✓ 24*7 services successfully installed and running on system boot!"
echo "Check status:"
echo "  sudo systemctl status upstox-web"
echo "  sudo systemctl status upstox-daemon"
