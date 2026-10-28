#!/bin/bash
# QIRE -- erster Test auf dem MPCDF (Raven/Viper): `direktor.py vortest`, nur Tuning-Seeds, kurz.
#   MPI/slurm/starte.sh test         (oder im Projektordner: mkdir -p logs && sbatch MPI/slurm/erster_test.sbatch)
#SBATCH --job-name=qire-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -uo pipefail
WURZEL="${QIRE_WURZEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -f "$WURZEL/MPI/lokal.env" ]; then set -a; . "$WURZEL/MPI/lokal.env"; set +a; fi
if [ -n "${QIRE_MODULE:-}" ] && command -v module >/dev/null 2>&1; then module load $QIRE_MODULE; fi
case "${QIRE_ENV_ART:-venv}" in
    conda) eval "$(conda shell.bash hook)"; conda activate "${QIRE_ENV:?QIRE_ENV fehlt}" ;;
    *)     . "${QIRE_ENV:?QIRE_ENV fehlt}/bin/activate" ;;
esac
# libgomp von torch beim Start vorladen: sonst auf MPCDF Raven "cannot allocate memory in static TLS block"
GOMP="$(ls "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"/lib/python3*/site-packages/torch/lib/libgomp*.so* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export INSTITUT_PROZESSE="${SLURM_CPUS_PER_TASK:-4}" INSTITUT_ORT="mpi-${SLURMD_NODENAME:-knoten}"
export QIRE_DATEN="${QIRE_DATEN:-$WURZEL/INSTITUT/daten}"
echo "[$(date +%F_%T)] Knoten ${SLURMD_NODENAME:-?}, Job ${SLURM_JOB_ID:-?}, $(python --version 2>&1)"
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'Threads', torch.get_num_threads())"
cd "$WURZEL/INSTITUT" || exit 2
time python direktor.py vortest; RC=$?
echo "[$(date +%F_%T)] Fertig, Code $RC"
exit $RC
