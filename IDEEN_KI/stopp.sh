#!/usr/bin/env bash
# Sanfter Stopp (Datei STOP, < 5 min). Notfall: ./stopp.sh --hart (kill -9 aller Prozesse)
cd "$(dirname "$0")" || exit 1
echo stop > STOP
if [ "$1" = "--hart" ] || [ "$1" = "/hart" ]; then
  [ -f daten/betrieb.pid ] && kill -9 "$(cat daten/betrieb.pid)" 2>/dev/null
  pkill -9 -x llama-server 2>/dev/null
  echo "Alles hart beendet."
else
  echo "STOP-Datei angelegt. Beendigung nach dem laufenden Schritt (< 5 min)."
fi
