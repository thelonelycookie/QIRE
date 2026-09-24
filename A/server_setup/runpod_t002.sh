#!/usr/bin/env bash
# T-002 auf einem RunPod-GPU-Pod, unbeaufsichtigt und abbruchfest.
#
# WAS SEIT DEM 12. AUGUST ANDERS IST
# Damals ging ein fertiger 22-Stunden-Lauf verloren, weil der Pod geloescht
# wurde und sein Datentraeger mit ihm. Drei Aenderungen:
#
#   1. /workspace liegt auf einem NETZWERK-VOLUME. Das ist ein eigenstaendiges
#      Objekt und ueberlebt das Loeschen des Pods.
#   2. `trainieren` nimmt einen abgebrochenen Lauf WIEDER AUF. Bereits
#      gerechnete (Arm, Seed)-Paare werden uebersprungen.
#   3. Nach jedem Seed liegt der Zwischenstand auf Platte -- ein Abbruch bei
#      Seed 7 kostet hoechstens den laufenden Seed.
#
# Aufruf auf dem Pod:
#   cd /workspace && setsid nohup bash server_setup/runpod_t002.sh > results/lauf.log 2>&1 &

set -uo pipefail
cd /workspace
SCHRITTE="${1:-${SCHRITTE:-13000}}"
export PYTHONIOENCODING=utf-8
mkdir -p results

melde() { echo; echo "=== $(date -u '+%F %T UTC')  $* ==="; echo; }

melde "START -- $SCHRITTE Schritte je Arm"
python - <<'PY'
import importlib, torch
fehlt = [m for m in ("numpy","scipy","pandas","requests","psutil",
                     "sklearn","pyarrow","tokenizers") if not importlib.util.find_spec(m)]
if fehlt:
    raise SystemExit(f"  FEHLT: {', '.join(fehlt)}")
print(f"  torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("  ABBRUCH: keine GPU -- Pod falsch konfiguriert")
PY
[ $? -ne 0 ] && exit 1

melde "1/4  Korpus"
python -u -m quantum_sim.korpus_deutsch aufbereiten

melde "2/4  Lernratensuche"
if [ -f results/korpus/lr_map.json ]; then
  echo "  lr_map.json vorhanden -- uebersprungen"
else
  python -u -m quantum_sim.sprachmodell lernratensuche 2>&1 | tee results/lr_suche.log
fi

melde "3/4  Hauptlauf (nimmt Abgebrochenes wieder auf)"
python -u -m quantum_sim.sprachmodell trainieren "$SCHRITTE" 2>&1 | tee -a results/T-002_lauf.log
STATUS=${PIPESTATUS[0]}
if [ "$STATUS" -ne 0 ]; then
  melde "Hauptlauf beendet mit Status $STATUS -- Zwischenstand in results/experiments"
  echo "Erneut starten setzt dort fort, wo er stehengeblieben ist."
  exit "$STATUS"
fi

melde "4/4  Textproben"
for arm in dense butterfly_ffn butterfly_qkv narrow; do
  echo "--- $arm ---"
  python -u -m quantum_sim.sprachmodell erzeuge "$arm" "Die Quantenmechanik ist" 2>&1 | tail -3 \
    || echo "  (fehlgeschlagen)"
  echo
done | tee results/T-002_textproben.log

melde "FERTIG"
echo "Ergebnisse holen:"
echo "  scp -P <port> -i ~/.ssh/runpod_qire root@<host>:/workspace/results/experiments/T-002__*.json ."
echo "Danach Pod stoppen -- er kostet weiter."
