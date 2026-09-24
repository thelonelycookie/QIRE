#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[QIRE] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[QIRE] Installing server requirements..."
pip install -r requirements_server.txt

echo "[QIRE] Running server health check..."
python qire_ai_console.py --server-health --deep

echo "[QIRE] Installation finished. Use tmux scripts in server_setup/."
