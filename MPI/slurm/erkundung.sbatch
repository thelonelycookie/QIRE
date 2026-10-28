#!/bin/bash
# Erkundungspaket (Rauchtests, Tuning-Seeds 950-954, keine Befunde): alle INSTITUT/erkundung/K*.py, E*.py.
#   mkdir -p logs && sbatch MPI/slurm/erkundung.sbatch [--nur K01 K03]
# Wiederaufnahme: fertige roh/*.json werden uebersprungen -> bei Zeitlimit einfach erneut einreichen.
#SBATCH --job-name=qire-erkundung
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
set -uo pipefail
WURZEL="${QIRE_WURZEL:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WURZEL/MPI/lokal.env" ] && { set -a; . "$WURZEL/MPI/lokal.env"; set +a; }
module load python-waterboa/2025.06 2>/dev/null; . "${QIRE_ENV:-$HOME/venvs/qire}/bin/activate"
GOMP="$(ls "$VIRTUAL_ENV"/lib/python3*/site-packages/torch/lib/libgomp*.so* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 INSTITUT_PROZESSE="${SLURM_CPUS_PER_TASK:-72}"
cd "$WURZEL/INSTITUT" && python3 erkundung/lauf_alle.py "$@"
