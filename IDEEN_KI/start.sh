#!/usr/bin/env bash
# Startet die lokale QIRE-Ideen-KI (Dauerbetrieb). Stoppen: ./stopp.sh
cd "$(dirname "$0")" || exit 1
rm -f STOP
mkdir -p daten
echo "Ideen-KI startet ... (Stopp: ./stopp.sh)"
python3 -m schmiede.betrieb "$@" >> daten/betrieb.log 2>&1
echo "Ideen-KI beendet."
