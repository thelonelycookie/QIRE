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

export PYTHONIOENCODING=utf-8
mkdir -p results

melde() { echo; echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC')  $* ==="; echo; }

# Vorregistriert, EINMAL in quantum_sim/sprachmodell.py (SCHRITTE = 13.000:
# 13.000 x 64 x 128 = 106 Mio. Token, rund 20 Token je Parameter von dense).
# Hier nur gelesen, nicht mehr per Umgebungsvariable ueberschreibbar.
SCHRITTE="$(python -m quantum_sim.sprachmodell schritte)" || exit 1
melde "START -- $SCHRITTE Schritte je Arm"
python - <<'PY'
import torch
print(f"  torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
PY

# --- 0. Selbsttest ---------------------------------------------------------
python -u -m quantum_sim.sprachmodell selbsttest || { melde "ABBRUCH: Selbsttest"; exit 1; }

# --- 1. Lernratensuche -----------------------------------------------------
# Volles Budget (= Messlauf), verlaengert das Gitter nach oben UND unten, wenn
# ein Arm auf dem Rand landet. Wiederaufnehmbar: rechnet nur fehlende Zellen,
# deshalb kein "lr_map.json existiert -> ueberspringen" mehr.
melde "1/3  Lernratensuche"
python -u -m quantum_sim.sprachmodell lernratensuche 2>&1 | tee -a results/lr_suche.log
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -f results/t002_zellen/T-002/lr_map.json ]; then
  melde "ABBRUCH: Lernratensuche nicht vollstaendig -- erneut starten setzt fort"
  exit 1
fi

# --- 2. Hauptlauf ----------------------------------------------------------
melde "2/3  Hauptlauf"
python -u -m quantum_sim.sprachmodell trainieren 2>&1 | tee -a results/T-002_lauf.log
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
