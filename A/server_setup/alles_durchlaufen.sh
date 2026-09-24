#!/usr/bin/env bash
# T-002 vollstaendig und unbeaufsichtigt: Lernratensuche, Hauptlauf, Auswertung.
#
# ZWECK
# Nach dem Start ist kein Eingriff mehr noetig. Der eigene Rechner darf aus
# sein -- der Pod rechnet weiter, weil alles unter `nohup`/`setsid` haengt und
# nicht an der SSH-Sitzung.
#
# ALLES LANDET AUF DEM DAUERHAFTEN VOLUME (/workspace/results), nicht im
# Container-Dateisystem. Ein Stoppen des Pods verliert damit keine Ergebnisse.
#
# Aufruf:
#   cd /workspace && setsid nohup bash server_setup/alles_durchlaufen.sh > results/alles.log 2>&1 &

set -uo pipefail
cd /workspace

SCHRITTE="${SCHRITTE:-13000}"     # 13.000 x 64 x 128 = 106 Mio. Token je Arm,
                                  # also rund 20 Token je Parameter
export PYTHONIOENCODING=utf-8
mkdir -p results

melde() { echo; echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC')  $* ==="; echo; }

melde "START -- $SCHRITTE Schritte je Arm"
python - <<'PY'
import torch
print(f"  torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
PY

# --- 1. Lernratensuche -----------------------------------------------------
# Verlaengert das Gitter selbst nach oben, wenn ein Arm auf dem Rand landet.
if [ -f results/korpus/lr_map.json ]; then
  melde "Lernratensuche uebersprungen -- lr_map.json existiert schon"
else
  melde "1/3  Lernratensuche"
  python -u -m quantum_sim.sprachmodell lernratensuche 2>&1 | tee results/lr_suche.log
  if [ ! -f results/korpus/lr_map.json ]; then
    melde "ABBRUCH: Lernratensuche hat keine lr_map.json erzeugt"
    exit 1
  fi
fi

# --- 2. Hauptlauf ----------------------------------------------------------
melde "2/3  Hauptlauf"
python -u -m quantum_sim.sprachmodell trainieren "$SCHRITTE" 2>&1 | tee results/T-002_lauf.log
STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
  melde "Hauptlauf mit Status $STATUS beendet -- Zwischenstaende liegen in results/experiments"
  exit "$STATUS"
fi

# --- 3. Textprobe je Arm ---------------------------------------------------
# Damit am Ende nicht nur Zahlen dastehen, sondern sichtbar ist, was die
# Modelle schreiben.
melde "3/3  Textproben"
for arm in dense butterfly_ffn butterfly_qkv narrow; do
  echo "--- $arm ---"
  python -u -m quantum_sim.sprachmodell erzeuge "$arm" "Die Quantenmechanik ist" \
    2>&1 | tail -3 || echo "  (Textprobe fehlgeschlagen)"
  echo
done | tee results/T-002_textproben.log

melde "FERTIG"
echo "Rohdaten:    results/experiments/T-002__*.json"
echo "Modelle:     results/modelle/*.pt"
echo "Textproben:  results/T-002_textproben.log"
echo
echo "Der Pod rechnet ab hier nichts mehr und kostet weiter Geld."
echo "In der RunPod-Oberflaeche stoppen, sobald die Ergebnisse geholt sind."
