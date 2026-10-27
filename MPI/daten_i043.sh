#!/bin/bash
# QIRE I-043-1B: deutsche Wikipedia -> Token-Straeme (uint16) unter /ptmp/$USER/i043.  NUR AUF DEM LOGIN-KNOTEN (braucht Internet).
#   bash MPI/daten_i043.sh                 # Standard: ca. 31 Mio Train-, 1,05 Mio Val-, 2,1 Mio Test-Tokens (ein paar Minuten, Download ca. 0,8-1,5 GB)
#   bash MPI/daten_i043.sh --quelle gsd    # Fallback UD_German-GSD (klein; Profil "gsd" noetig: I043_PROFIL=gsd)
# Weitere Argumente gehen an MPI/daten_i043.py (--ziel, --train-tokens, --shard, ...).  Lizenz/Begruendung: Kopf von daten_i043.py.
set -euo pipefail
WURZEL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$WURZEL/MPI/lokal.env" ] && { set -a; . "$WURZEL/MPI/lokal.env"; set +a; }
if command -v module >/dev/null 2>&1; then module load python-waterboa/2025.06 2>/dev/null || true; fi
. ~/venvs/qire-gpu/bin/activate
export HF_HOME="${HF_HOME:-/ptmp/$USER/hf}"
export I043_DATEN="${I043_DATEN:-/ptmp/$USER/i043}"
mkdir -p "$I043_DATEN"
# Tokenizer kommt aus dem schon geladenen Modell-Cache; Datensatz-Download braucht Internet (Login-Knoten).
python "$WURZEL/MPI/daten_i043.py" --ziel "$I043_DATEN" "$@"
ls -l "$I043_DATEN"
echo "fertig. In MPI/lokal.env (falls abweichend):  I043_DATEN=$I043_DATEN   HF_HOME=$HF_HOME"
