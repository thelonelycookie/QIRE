#!/bin/bash
# Auf dem Raven-Login-Knoten: Code aktualisieren, Daten vorladen, alle freigegebenen Laeufe einreichen,
# die weder fertig sind noch schon in der Warteschlange stehen.
#   bash MPI/alles_starten.sh            (bash MPI/alles_starten.sh --trocken: nur anzeigen)
set -uo pipefail
cd "$(dirname "$0")/.."
TROCKEN=""; [ "${1:-}" = "--trocken" ] && TROCKEN="--trocken"
git stash -q 2>/dev/null; git pull -q || { echo "git pull gescheitert -- abgebrochen." >&2; exit 1; }
module load python-waterboa/2025.06 2>/dev/null; . ~/venvs/qire/bin/activate
mkdir -p logs
REG=logs/eingereicht.txt; touch "$REG"
fertig() { ls INSTITUT/ergebnisse/"$1"__2*.json >/dev/null 2>&1; }
laeuft() {  # irgendein fuer diese ID eingereichter Job noch in der Warteschlange?
    for j in $(awk -v id="$1" '$1==id {print $2}' "$REG"); do squeue -h -j "$j" 2>/dev/null | grep -q . && return 0; done
    # auch von Hand eingereichte Jobs: laufende Jobs, deren Log "Start <ID> " enthaelt
    for j in $(squeue -u "$USER" -h -t R -o %i 2>/dev/null); do grep -qs "Start $1 " logs/*-"$j".out && return 0; done
    return 1; }
merke() {  # Ausgabe von starte.sh mitschreiben, Job-IDs registrieren
    tee /dev/stderr | grep -o "Job [0-9]*" | awk -v id="$1" '{print id, $2}' >> "$REG"; }
cpu() {  # ID KERNE SPEICHER ZEIT KETTE
    if fertig "$1"; then echo "= $1: fertig"; return; fi
    if laeuft "$1"; then echo "= $1: laeuft schon"; return; fi
    echo "+ $1 einreichen"
    QIRE_KERNE=$2 QIRE_SPEICHER=$3 QIRE_ZEIT=$4 MPI/slurm/starte.sh cpu "$1" --kette "$5" $TROCKEN 2>&1 | merke "$1"
}
# Daten, die Rechenknoten ohne Internet brauchen
( cd INSTITUT && python -c "from abteilungen.struktur_gross import _cifar_laden as c; c()" >/dev/null 2>&1 ) && echo "CIFAR ok" || echo "WARNUNG: CIFAR nicht geladen"
cpu I-046-v2            72 100G 02:00:00 1
cpu T-ternaer-butterfly 72 100G 03:00:00 1
cpu I-044-v4            72 200G 24:00:00 2
cpu I-047               72 100G 24:00:00 2
# GPU: nur wenn Daten da sind und der Rauchtest schon gelaufen ist
if fertig I-043-1B; then echo "= I-043-1B: fertig"
elif laeuft I-043-1B; then echo "= I-043-1B: laeuft schon"
elif [ ! -d /ptmp/$USER/i043 ]; then echo "! I-043-1B: erst Daten laden: bash MPI/daten_i043.sh (danach erneut starten)"
elif ! ls INSTITUT/ergebnisse/I-043-1B-schnell__*.json >/dev/null 2>&1; then echo "+ I-043-1B Rauchtest"; QIRE_ZEIT=01:00:00 MPI/slurm/starte.sh gpulauf I-043-1B --schnell $TROCKEN 2>&1 | merke I-043-1B
else echo "+ I-043-1B Volllauf"; MPI/slurm/starte.sh gpulauf I-043-1B --kette 3 $TROCKEN 2>&1 | merke I-043-1B
fi
squeue -u "$USER"
