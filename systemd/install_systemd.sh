#!/bin/bash
# install_systemd.sh - Installs and activates 24*7 systemd services on Linux home server

set -e

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./systemd/install_systemd.sh)"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Installing systemd services for 24*7 operation..."
cp "$DIR/upstox-web.service" /etc/systemd/system/
cp "$DIR/upstox-daemon.service" /etc/systemd/system/

systemctl daemon-reload
systemctl enable upstox-web.service
systemctl enable upstox-daemon.service

systemctl restart upstox-web.service
systemctl restart upstox-daemon.service

echo "✓ 24*7 services successfully installed and running on system boot!"
echo "Check status:"
echo "  sudo systemctl status upstox-web"
echo "  sudo systemctl status upstox-daemon"
