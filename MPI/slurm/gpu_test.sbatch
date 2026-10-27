#!/bin/bash
# QIRE -- erster GPU-Test auf Raven (gpudev, 15 min): CUDA, LLaeMmlein-1B offline laden, Vorwaerts-/Rueckwaertsschritt mit LoRA,
# Speicher und Tokens/s.   Aufruf im Projektordner:  mkdir -p logs && sbatch MPI/slurm/gpu_test.sbatch
#SBATCH --job-name=qire-gputest
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpudev
#SBATCH --account=msp_gpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
set -uo pipefail
WURZEL="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "$WURZEL/MPI/lokal.env" ]; then set -a; . "$WURZEL/MPI/lokal.env"; set +a; fi
module load python-waterboa/2025.06
. ~/venvs/qire-gpu/bin/activate
GOMP="$(ls "$VIRTUAL_ENV"/lib/python3*/site-packages/torch/lib/libgomp*.so* 2>/dev/null | head -1)"
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP"
export HF_HOME="${HF_HOME:-/ptmp/$USER/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
print("torch", torch.__version__, "cuda", torch.version.cuda, "verfuegbar", torch.cuda.is_available())
m = "LSX-UniWue/LLaMmlein_1B"
tok = AutoTokenizer.from_pretrained(m)
tok.pad_token = tok.pad_token or tok.eos_token
mod = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16).cuda()
print("Parameter (Mio.):", round(sum(p.numel() for p in mod.parameters()) / 1e6, 1))
mod = get_peft_model(mod, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
mod.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=1e-4)
x = tok(["Die Quantenphysik beschreibt die Natur auf kleinsten Skalen."] * 8, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to("cuda")
torch.cuda.synchronize(); t = time.time()
for i in range(10):
    l = mod(**x, labels=x["input_ids"]).loss; l.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize(); dt = time.time() - t
print(f"Verlust {l.item():.3f}; {10*8*512/dt:.0f} Tokens/s Training; Speicher max {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
mod.eval()
with torch.no_grad():
    out = mod.generate(**tok("Ein Quantencomputer ist", return_tensors="pt").to("cuda"), max_new_tokens=30)
print("Probe:", tok.decode(out[0], skip_special_tokens=True))
print("GPU-TEST BESTANDEN")
PY
