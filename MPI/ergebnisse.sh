#!/bin/bash
# Auf Raven: Stand aller Laeufe, Ergebniszeilen (zum Kopieren in den Chat) und ein Paket der Rohdaten.
#   bash MPI/ergebnisse.sh
cd "$(dirname "$0")/.."
echo "== Warteschlange"; squeue -u "$USER"
echo "== Letzte Jobs"; sacct -u "$USER" -S now-3days --format=JobID,JobName%15,State,Elapsed,ExitCode -X 2>/dev/null | tail -15
for id in I-046-v2 T-ternaer-butterfly I-044-v4 I-047 I-043-1B I-048 I-049 I-045-v5; do
    f=$(ls -t INSTITUT/ergebnisse/"$id"__2*.json 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        echo "== $id: FERTIG ($f)"
        grep -h "| $id |" $(ls -t logs/*.out 2>/dev/null) 2>/dev/null | head -1 | cut -c1-3000
    else
        z=$(ls INSTITUT/ergebnisse/_zwischen_"$id".json 2>/dev/null)
        echo "== $id: nicht fertig${z:+ (Zwischenstand vorhanden)}"
        l=$(grep -l "Start $id " $(ls -t logs/*.out 2>/dev/null | head -30) 2>/dev/null | head -1)
        [ -n "$l" ] && { tail -3 "$l"; e="${l%.out}.err"; [ -s "$e" ] && { echo "-- Fehlerlog:"; tail -5 "$e"; }; }
    fi
done
tar czf ~/qire_ergebnisse.tgz INSTITUT/ergebnisse/*__2*.json 2>/dev/null && echo "== Paket: ~/qire_ergebnisse.tgz (am PC holen, siehe ANLEITUNG)"
