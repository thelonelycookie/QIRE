#!/usr/bin/env bash
# T-002 -- EINE Zelle je Array-Element. Wird von t002_steuerung.sh abgeschickt,
# nicht von Hand.
#
# Jede Zeile der Plandatei ($PLAN) ist eine vollstaendige Argumentliste fuer
#   python -m quantum_sim.sprachmodell <Zeile>
# z. B. "lernratensuche --arm dense --lr 0.0006 --seed 2199 --schritte 13000"
# oder  "zelle --arm dense --seed 2101 --schritte 13000".
# Element i rechnet Zeile i+1. Die Plandatei wird nach dem Abschicken nicht
# mehr veraendert (neue Plaene bekommen neue Dateinamen) -- die Zuordnung
# Index -> Zelle bleibt also stabil, auch wenn Jobs neu gestartet werden.
#
# ANNAHMEN (am MPI pruefen und hier oder per SBATCH_EXTRA anpassen):
#   * GPU-Anforderung ueber --gres=gpu:1 (manche Cluster: --gpus=1)
#   * Partition/Account: nicht gesetzt -> SBATCH_EXTRA="--partition=... --account=..."
#   * Zeitlimit 3 h: Schaetzung ~74 min je Zelle auf einer RTX 4090 (nur
#     butterfly_ffn gemessen), doppelte Reserve. Auf langsameren Karten erhoehen.
#   * Rechenknoten ohne Internet: alles Noetige (Code, Korpus, Python-Umgebung)
#     liegt auf einem gemeinsamen Dateisystem.
#
#SBATCH --job-name=t002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -uo pipefail

: "${PLAN:?PLAN (Plandatei) fehlt -- bitte ueber t002_steuerung.sh abschicken}"
: "${QIRE_CODE:?QIRE_CODE (Ordner A mit quantum_sim/) fehlt}"
: "${QIRE_KORPUS:?QIRE_KORPUS (Ordner mit train.npy, val.npy, bpe_4096.json) fehlt}"

if [ -n "${QIRE_UMGEBUNG:-}" ]; then
  # z. B. eine Datei mit "module load cuda" und "source /pfad/venv/bin/activate"
  # shellcheck disable=SC1090
  source "$QIRE_UMGEBUNG"
fi
PY="${QIRE_PYTHON:-python}"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

INDEX="${SLURM_ARRAY_TASK_ID:?nur als Job-Array sinnvoll}"
ZEILE="$(sed -n "$((INDEX + 1))p" "$PLAN")"
if [ -z "$ZEILE" ]; then
  echo "ABBRUCH: keine Zeile $((INDEX + 1)) in $PLAN"
  exit 1
fi

cd "$QIRE_CODE" || exit 1
echo "=== $(date -u '+%F %T UTC')  Knoten $(hostname)  Job ${SLURM_ARRAY_JOB_ID:-?}_${INDEX}"
echo "=== Zelle: $ZEILE"

"$PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ABBRUCH: keine GPU auf diesem Knoten -- Zelle NICHT gerechnet")
print(f"  torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY
if [ $? -ne 0 ]; then
  exit 1
fi

# Wortaufteilung von $ZEILE ist hier gewollt (Argumentliste).
# shellcheck disable=SC2086
"$PY" -u -m quantum_sim.sprachmodell $ZEILE
STATUS=$?
echo "=== $(date -u '+%F %T UTC')  Ende mit Status $STATUS"
exit $STATUS
