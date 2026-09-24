#!/usr/bin/env bash
set -euo pipefail

echo "[QIRE] Updating Ubuntu packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git tmux rsync unzip zip htop build-essential

echo "[QIRE] System ready. Next: bash server_setup/install_qire_engine.sh"
