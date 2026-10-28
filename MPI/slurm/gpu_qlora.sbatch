#!/bin/bash
# QIRE -- GPU-Job fuer das spaetere QLoRA-Training der Ideen-KI (Plan S3 Bewerter / S4 Erfinder).
#
# Aufruf:  MPI/slurm/starte.sh gpu s3        (oder s4)
# Steuerung ueber Umgebungsvariablen (starte.sh uebernimmt sie, oder per `sbatch --export=ALL,...`):
#     QLORA_STUFE     s3 | s4                                  (nur Beschriftung/Ausgabeordner)
#     BASIS_MODELL    HF-Name oder lokaler Pfad des Basismodells (ANNAHME: Rechenknoten ohne Internet ->
#                     Modell vorab nach $HF_HOME legen, siehe ANLEITUNG.md)
#     TRAIN_DATEN     JSONL {"prompt","antwort"} (Format: MPI/training/qlora_train.py --hilfe)
#     AUSGABE         Ordner fuer Checkpoints + Adapter (Standard: $WURZEL/MPI/adapter/<stufe>)
#     QLORA_ARGS      weitere Argumente fuer qlora_train.py (z. B. "--epochen 3 --lr 2e-4")
#
# Wiederaufnahme: qlora_train.py nimmt den neuesten checkpoint-* in $AUSGABE wieder auf. SIGUSR1 vor Zeitende
# -> der Trainer speichert einen Checkpoint und beendet sich; der Job stellt sich neu in die Warteschlange.
# Bewertungsgrundlage fuer "besser als Basis" sind zurueckgehaltene Ideen (--val-anteil), nie die Trainingsdaten.
#
# ALLE #SBATCH-Werte sind ANNAHMEN. Passend zur 8-GB-Zielhardware des Plans reicht fuer 3-4B-Modelle eine
# Karte mit >= 16 GB; fuer groessere Modelle (Erfinder ~20-30B, MoE) mehr VRAM erfragen.
#
#SBATCH --job-name=qire-qlora
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
##SBATCH --partition=<GPU-Partition>
##SBATCH --account=<Name>

set -uo pipefail
WURZEL="${QIRE_WURZEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_NEUSTARTS="${MAX_NEUSTARTS:-10}"
if [ -f "$WURZEL/MPI/lokal.env" ]; then set -a; . "$WURZEL/MPI/lokal.env"; set +a; fi
STUFE="${QLORA_STUFE:-s3}"
: "${BASIS_MODELL:?BASIS_MODELL fehlt (HF-Name oder lokaler Pfad)}"
: "${TRAIN_DATEN:?TRAIN_DATEN fehlt (JSONL)}"
AUSGABE="${AUSGABE:-$WURZEL/MPI/adapter/$STUFE}"

if [ -n "${QIRE_MODULE:-}" ] && command -v module >/dev/null 2>&1; then module load $QIRE_MODULE; fi
case "${QIRE_ENV_ART:-venv}" in
    conda) eval "$(conda shell.bash hook)"; conda activate "${QIRE_ENV:?QIRE_ENV fehlt}" ;;
    *)     . "${QIRE_ENV:?QIRE_ENV fehlt}/bin/activate" ;;
esac

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
if [ "${QIRE_OFFLINE:-1}" = 1 ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export HF_HOME="${HF_HOME:-$WURZEL/modelle/hf}"
mkdir -p "$AUSGABE"
cd "$WURZEL" || exit 2

echo "[$(date +%F_%T)] QLoRA $STUFE: Basis=$BASIS_MODELL Daten=$TRAIN_DATEN Ausgabe=$AUSGABE Job=${SLURM_JOB_ID:-?} Neustart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "WARNUNG: nvidia-smi nicht verfuegbar"
if [ "${SLURM_RESTART_COUNT:-0}" -gt "$MAX_NEUSTARTS" ]; then echo "Zu viele Neustarts" >&2; exit 3; fi

unterbrochen=0
# shellcheck disable=SC2086
python MPI/training/qlora_train.py --basis "$BASIS_MODELL" --daten "$TRAIN_DATEN" --ausgabe "$AUSGABE" --stufe "$STUFE" ${QLORA_ARGS:-} &
PID=$!
trap 'unterbrochen=1; echo "[$(date +%F_%T)] SIGUSR1: Checkpoint anfordern."; kill -TERM "$PID" 2>/dev/null' USR1
wait "$PID"; RC=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done

echo "[$(date +%F_%T)] qlora_train beendet mit Code $RC (unterbrochen=$unterbrochen)"
# Das Skript beendet sich nach SIGTERM mit Code 75 (EX_TEMPFAIL) = "Checkpoint gesichert, bitte fortsetzen".
if [ "$RC" = 75 ] || { [ "$unterbrochen" = 1 ] && [ "$RC" != 0 ]; }; then
    echo "Stelle Job neu in die Warteschlange."
    scontrol requeue "${SLURM_JOB_ID:?}" && exit 0
fi
exit "$RC"
