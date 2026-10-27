#!/bin/bash
# QIRE -- GPU-Lauf `direktor.py lauf <ID>` auf MPCDF Raven (ganzer Knoten, 4 x A100-40GB), mit Wiederaufnahme.
#
# Aufruf (empfohlen ueber MPI/slurm/starte.sh):
#     MPI/slurm/starte.sh gpulauf I-043-1B [--schnell] [--kette 3]
# oder direkt, im Projektordner:  mkdir -p logs && sbatch MPI/slurm/gpu_lauf.sbatch I-043-1B
#
# Das Modul (abteilungen/struktur_adapter_1b.py) startet je sichtbarer GPU EINEN Arbeiter (spawn) und verteilt die Aufgaben
# (Tuning, Messung, Negativkontrolle) darauf: 1 Aufgabe je GPU, 4 je Knoten. Wiederaufnahme wie cpu_lauf.sbatch: jede fertige Aufgabe steht atomar in
# INSTITUT/ergebnisse/_zwischen_<ID>.json, innerhalb einer Aufgabe gibt es alle 500 Schritte einen Checkpoint (_ckpt_<ID>/, nur Adapter + Optimierer).
# --requeue + SIGUSR1 vor Zeitende: python wird beendet, der Job reiht sich neu ein; der Neustart setzt fort (hoechstens $MAX_NEUSTARTS).
#
# Werte aus dem GPU-Test (27.10.): Partition gpu, Account msp_gpu, --gres=gpu:a100:N (bis 4 je Knoten), 24-h-Limit, 18 Kerne und ca. 120 GB je GPU,
# Rechenknoten ohne Internet (Daten VORAB auf dem Login-Knoten: bash MPI/daten_i043.sh). starte.sh passt gres/Kerne/RAM an QIRE_GPUS an.
#
#SBATCH --job-name=qire-gpulauf
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --account=msp_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --gres=gpu:a100:4
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -uo pipefail
LAUF_ID="${1:?Aufruf: sbatch gpu_lauf.sbatch <ID> [--schnell]}"
shift
MAX_NEUSTARTS="${MAX_NEUSTARTS:-10}"
SIGNAL_VORLAUF="${SIGNAL_VORLAUF:-600}"          # muss zu --signal=B:USR1@... passen
WURZEL="${QIRE_WURZEL:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -f "$WURZEL/INSTITUT/direktor.py" ]; then
    echo "FEHLER: $WURZEL/INSTITUT/direktor.py nicht gefunden. sbatch im Projektordner starten oder QIRE_WURZEL setzen." >&2
    exit 2
fi
if [ -f "$WURZEL/MPI/lokal.env" ]; then set -a; . "$WURZEL/MPI/lokal.env"; set +a; fi

# --- Software wie gpu_test.sbatch: GPU-venv (torch cu126, transformers, peft ...) ---
module load python-waterboa/2025.06
. "${QIRE_ENV_GPU:-$HOME/venvs/qire-gpu}/bin/activate"
# libgomp von torch beim Start vorladen: sonst auf MPCDF Raven "cannot allocate memory in static TLS block"
GOMP="$(ls "$VIRTUAL_ENV"/lib/python3*/site-packages/torch/lib/libgomp*.so* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
# Modell und Daten liegen vorab auf /ptmp (Rechenknoten ohne Internet)
export HF_HOME="${HF_HOME:-/ptmp/$USER/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export I043_DATEN="${I043_DATEN:-/ptmp/$USER/i043}"
export TOKENIZERS_PARALLELISM=false
# Hauptprozess (Vortest, Auswertung) mit 1 Thread; die Arbeiter setzen ihre Threads selbst (I043_THREADS, Standard 4)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I043_THREADS="${I043_THREADS:-4}"
export INSTITUT_ORT="mpi-gpu-${SLURMD_NODENAME:-knoten}"
export PYTHONUNBUFFERED=1
cd "$WURZEL/INSTITUT" || exit 2

echo "[$(date +%F_%T)] Start $LAUF_ID auf ${SLURMD_NODENAME:-?}, Job ${SLURM_JOB_ID:-?}, Neustart Nr. ${SLURM_RESTART_COUNT:-0}, GPUs ${SLURM_GPUS_ON_NODE:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || echo "WARNUNG: nvidia-smi nicht verfuegbar" >&2
if [ ! -f "$I043_DATEN/meta.json" ]; then
    echo "FEHLER: $I043_DATEN/meta.json fehlt. Erst auf dem Login-Knoten:  bash MPI/daten_i043.sh" >&2
    exit 4
fi
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
# Zeitende naht: den Arbeitern Gnadenfrist geben, dann sauber beenden (Checkpoints in den Aufgaben bleiben) und neu einreihen.
trap 'unterbrochen=1; echo "[$(date +%F_%T)] SIGUSR1: Zeitlimit naht."; ( sleep $((SIGNAL_VORLAUF > 180 ? SIGNAL_VORLAUF - 120 : 30)); kill -TERM "$PID" 2>/dev/null ) &' USR1
wait "$PID"; RC=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done      # wait kehrt beim Signal vorzeitig zurueck

echo "[$(date +%F_%T)] python beendet mit Code $RC (unterbrochen=$unterbrochen)"
if [ "$unterbrochen" = 1 ] && [ "$RC" != 0 ]; then
    echo "Stelle Job neu in die Warteschlange (Wiederaufnahme je Aufgabe, Checkpoint alle 500 Schritte)."
    scontrol requeue "${SLURM_JOB_ID:?}" && exit 0
    echo "scontrol requeue nicht erlaubt -> mit starte.sh --kette erneut einreihen." >&2
fi
exit "$RC"
