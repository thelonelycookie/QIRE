#!/usr/bin/env bash
# Startpaket fuer einen RunPod-GPU-Pod -- T-002, das Sprachmodell.
#
# WAS DIESES SKRIPT TUT
#   1. fehlende Pakete nachinstallieren (torch bringt das RunPod-Image mit)
#   2. den Korpus HERUNTERLADEN UND AUFBEREITEN -- direkt auf dem Pod, nicht
#      hochladen. Der Shard kommt von HuggingFace und ist im Rechenzentrum in
#      Sekunden da; 795 MB Token ueber die eigene Leitung zu schieben waere
#      unnoetig.
#   3. Durchsatz messen, damit die Schrittzahl zum Budget passt
#   4. eine kurze Probe fahren, bevor Stunden verbrannt werden
#   5. den eigentlichen Lauf starten
#
# WAS ES BEWUSST NICHT TUT
#   Zeitmessungen fuer die Effizienzaussage. Die bleiben auf dem Rechner, auf
#   dem die Vergleichsbasis liegt (T-001, 5. August). Eine Zahl vom Laptop und
#   eine vom Pod in einer Aussage zu mischen ist genau der Fehler, der die
#   819x-Aussage gekostet hat. Der Pod liefert QUALITAET, nicht Effizienz.
#
# Aufruf auf dem Pod:
#   bash server_setup/runpod_start.sh            # Probe + Lauf mit 20.000 Schritten
#   bash server_setup/runpod_start.sh 40000      # andere Schrittzahl
#   SCHRITTE=5000 NUR_PROBE=1 bash server_setup/runpod_start.sh

set -euo pipefail

SCHRITTE="${1:-${SCHRITTE:-20000}}"
NUR_PROBE="${NUR_PROBE:-0}"

echo "=============================================================="
echo "T-002 auf RunPod -- $SCHRITTE Schritte je Arm"
echo "=============================================================="

echo "--- 1/5  Pakete ---"
# Vollstaendige Liste, aus den Importen von sprachmodell.py und
# korpus_deutsch.py verfolgt. Die erste Fassung hatte scikit-learn und psutil
# vergessen -- beide werden ueber qire_ai.benchmark -> qire_ai.metrics gezogen
# und haben den Lauf am Pod nach dem Korpus abgebrochen.
python -m pip install --quiet --upgrade pip
python -m pip install --quiet \
  numpy scipy pandas requests psutil scikit-learn pyarrow tokenizers
python - <<'PY'
import importlib
fehlt = [m for m in ("numpy", "scipy", "pandas", "requests", "psutil",
                     "sklearn", "pyarrow", "tokenizers", "torch")
         if not importlib.util.find_spec(m)]
if fehlt:
    raise SystemExit(f"  FEHLT WEITERHIN: {', '.join(fehlt)} -- abgebrochen")
print("  alle benoetigten Pakete vorhanden")
PY
python - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  CUDA verfuegbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Speicher: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
else:
    print("  WARNUNG: keine GPU gefunden -- der Lauf waere hier nicht schneller")
    print("  als auf dem Laptop. Pod-Konfiguration pruefen.")
PY

echo "--- 2/5  Korpus ---"
python -m quantum_sim.korpus_deutsch aufbereiten

echo "--- 3/5  Durchsatz je Arm ---"
python -m quantum_sim.sprachmodell durchsatz

echo "--- 4/5  Probe (60 Schritte, 2 Seeds) ---"
python -m quantum_sim.sprachmodell probe

if [ "$NUR_PROBE" = "1" ]; then
  echo "NUR_PROBE gesetzt -- hier ist Schluss."
  exit 0
fi

echo "--- 5/5  Lauf ---"
python -m quantum_sim.sprachmodell trainieren "$SCHRITTE"

echo "=============================================================="
echo "FERTIG. Rohdaten liegen in results/experiments/T-002__*.json"
echo
echo "Zurueckholen auf den eigenen Rechner, damit q_summary und q_audit"
echo "sie lesen koennen:"
echo "  scp -P <port> root@<host>:/workspace/qire_ai_console_v1_0/results/experiments/T-002__*.json ."
echo "=============================================================="
