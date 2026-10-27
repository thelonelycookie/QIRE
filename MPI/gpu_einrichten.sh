#!/bin/bash
# Einmalig auf dem Raven-Login-Knoten (hat Internet): GPU-Umgebung + LLaeMmlein-1B vorab laden.
#   bash MPI/gpu_einrichten.sh
# Eigene venv (~/venvs/qire-gpu), weil ~/venvs/qire das CPU-Rad von torch hat.
set -euo pipefail
module load python-waterboa/2025.06
[ -d ~/venvs/qire-gpu ] || python3 -m venv ~/venvs/qire-gpu
. ~/venvs/qire-gpu/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q "transformers>=4.40" "peft>=0.10" "accelerate>=0.29" "bitsandbytes>=0.43" datasets sentencepiece protobuf numpy scipy
export HF_HOME="${HF_HOME:-/ptmp/$USER/hf}"
mkdir -p "$HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["LSX-UniWue/LLaMmlein_1B", "LSX-UniWue/LLaMmlein_120M"]:
    print("lade", m, "->", snapshot_download(m))
PY
echo "fertig. HF_HOME=$HF_HOME  (in MPI/lokal.env eintragen: HF_HOME=$HF_HOME)"
