#!/bin/bash
# QIRE -- CPU-Lauf `direktor.py lauf <ID>` auf dem MPI-Cluster, mit Wiederaufnahme.
#
# Aufruf (empfohlen ueber MPI/slurm/starte.sh, das Partition/Account aus MPI/lokal.env einsetzt):
#     MPI/slurm/starte.sh cpu I-050 [--schnell] [--kette 3]
# oder direkt, im Projektordner:  mkdir -p logs && sbatch MPI/slurm/cpu_lauf.sbatch I-050
#
# ALLE #SBATCH-Werte sind ANNAHMEN (Kerne, RAM, Zeit, Partition). Zugang/Cluster-Details sind noch unbekannt
# (HANDOVER 06.10.: "Details folgen"). Vor dem ersten Lauf an die Cluster-Doku anpassen.
#
# Wiederaufnahme: Die Module des Instituts sichern nach JEDEM Seed (labor/training.lauf_je_seed ->
# ergebnisse/_zwischen_<ID>*). --requeue + SIGUSR1 vor Zeitende: der Job beendet python sauber und stellt sich
# selbst neu in die Warteschlange; der Neustart ueberspringt fertige Seeds. Nach hoechstens $MAX_NEUSTARTS Neustarten Schluss.
#
#SBATCH --job-name=qire-lauf
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
##SBATCH --partition=<Name>     (steht in MPI/lokal.env -> starte.sh setzt es auf der Kommandozeile)
##SBATCH --account=<Name>

set -uo pipefail
LAUF_ID="${1:?Aufruf: sbatch cpu_lauf.sbatch <ID> [--schnell]}"
shift
MAX_NEUSTARTS="${MAX_NEUSTARTS:-20}"
SIGNAL_VORLAUF="${SIGNAL_VORLAUF:-900}"          # muss zu --signal=B:USR1@... passen
WURZEL="${QIRE_WURZEL:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -f "$WURZEL/INSTITUT/direktor.py" ]; then
    echo "FEHLER: $WURZEL/INSTITUT/direktor.py nicht gefunden. sbatch im Projektordner starten oder QIRE_WURZEL setzen." >&2
    exit 2
fi
if [ -f "$WURZEL/MPI/lokal.env" ]; then set -a; . "$WURZEL/MPI/lokal.env"; set +a; fi

# --- Software (ANNAHME: module + venv oder conda) ---
if [ -n "${QIRE_MODULE:-}" ] && command -v module >/dev/null 2>&1; then module load $QIRE_MODULE; fi
case "${QIRE_ENV_ART:-venv}" in
    conda) eval "$(conda shell.bash hook)"; conda activate "${QIRE_ENV:?QIRE_ENV fehlt}" ;;
    *)     . "${QIRE_ENV:?QIRE_ENV fehlt}/bin/activate" ;;
esac

# --- Umgebung wie CLAUDE.md "Technik": ein Thread je Prozess, Prozesse = Kerne ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export INSTITUT_PROZESSE="${SLURM_CPUS_PER_TASK:-4}"
export INSTITUT_ORT="mpi-${SLURMD_NODENAME:-knoten}"
export QIRE_DATEN="${QIRE_DATEN:-$WURZEL/INSTITUT/daten}"      # Rechenknoten evtl. ohne Internet -> Daten vorab hinlegen
cd "$WURZEL/INSTITUT" || exit 2

echo "[$(date +%F_%T)] Start $LAUF_ID auf ${SLURMD_NODENAME:-?}, Job ${SLURM_JOB_ID:-?}, Neustart Nr. ${SLURM_RESTART_COUNT:-0}, Prozesse $INSTITUT_PROZESSE"
if [ "${SLURM_RESTART_COUNT:-0}" -gt "$MAX_NEUSTARTS" ]; then
    echo "Zu viele Neustarts ($MAX_NEUSTARTS) -- Abbruch. Log pruefen." >&2
    exit 3
fi

# Schon fertig (z. B. spaeteres Glied einer --kette)? Dann nichts tun.
if python - "$LAUF_ID" <<'PY'
import json, sys
try:
    d = json.load(open(f"laeufe/{sys.argv[1]}.status.json", encoding="utf-8"))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get("zustand") == "fertig" and not sys.argv[1].endswith("schnell") else 1)
PY
then
    echo "$LAUF_ID ist laut laeufe/$LAUF_ID.status.json fertig -- nichts zu tun."
    exit 0
fi

unterbrochen=0
python direktor.py lauf "$LAUF_ID" "$@" &
PID=$!
# Zeitende naht: python Gnadenfrist geben (laufender Seed darf enden), dann sauber beenden und neu einreihen.
trap 'unterbrochen=1; echo "[$(date +%F_%T)] SIGUSR1: Zeitlimit naht."; ( sleep $((SIGNAL_VORLAUF > 180 ? SIGNAL_VORLAUF - 120 : 30)); kill -TERM "$PID" 2>/dev/null ) &' USR1
wait "$PID"; RC=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done      # wait kehrt beim Signal vorzeitig zurueck

echo "[$(date +%F_%T)] python beendet mit Code $RC (unterbrochen=$unterbrochen)"
if [ "$unterbrochen" = 1 ] && [ "$RC" != 0 ]; then
    echo "Stelle Job neu in die Warteschlange (Wiederaufnahme je Seed)."
    scontrol requeue "${SLURM_JOB_ID:?}" && exit 0
    echo "scontrol requeue nicht erlaubt -> mit starte.sh --kette erneut einreihen." >&2
fi
exit "$RC"
