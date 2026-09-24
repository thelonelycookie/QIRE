#!/usr/bin/env bash
# T-002 auf einem SLURM-Cluster steuern -- laeuft auf dem LOGIN-Knoten.
#
#   bash server_setup/cluster/t002_steuerung.sh pruefen          # Umgebung + Selbsttest (CPU, Sekunden)
#   bash server_setup/cluster/t002_steuerung.sh rechenbedarf     # Schaetzung GPU-Stunden
#   bash server_setup/cluster/t002_steuerung.sh lr               # offene LR-Zellen als Job-Array abschicken
#   bash server_setup/cluster/t002_steuerung.sh messung          # offene Messzellen als Job-Array abschicken
#   bash server_setup/cluster/t002_steuerung.sh status           # Stand der Zellen
#   bash server_setup/cluster/t002_steuerung.sh zusammenfuehren  # einsammeln + analyse() + Sammeldatei
#   bash server_setup/cluster/t002_steuerung.sh auto             # alles nacheinander, kettet sich selbst
#
# Ablauf: LR-Startgitter (16 Zellen) -> bis zu 3 Verlaengerungsrunden (je <= 4
# Zellen) -> lr_map.json -> Messlauf (32 Zellen) -> zusammenfuehren. Jede
# Phase haengt von der vorigen ab; `plan` im Python-Modul rechnet jedes Mal
# aus den vorhandenen Zellendateien nach, was noch fehlt. Ein erneuter Aufruf
# rechnet also nie etwas doppelt, und abgebrochene Zellen werden einfach
# wieder eingeplant.
#
# Umgebungsvariablen (siehe README.md):
#   QIRE_KORPUS      Pflicht. Ordner mit train.npy, val.npy, bpe_4096.json
#   QIRE_CODE        Ordner A (Standard: zwei Ebenen ueber diesem Skript)
#   QIRE_T002_ZELLEN Ergebnisordner (Standard: $QIRE_CODE/results/t002_zellen)
#   QIRE_PYTHON      Python mit torch+CUDA (Standard: python)
#   QIRE_UMGEBUNG    Datei, die vor jedem Python-Aufruf gesourct wird
#                    (module load ..., source venv/bin/activate)
#   SBATCH_EXTRA     zusaetzliche sbatch-Optionen fuer GPU-Jobs (--partition, --account)
#   WEITER_EXTRA     sbatch-Optionen fuer den kleinen CPU-Kettenjob in `auto`
#   MAX_PARALLEL     hoechstens so viele Zellen gleichzeitig (Standard 16)
#   MIT_ZUSATZARM=1  butterfly_ffn_zentriert mitrechnen (NUR nach Maltes
#                    Entscheidung und Tagebucheintrag)
#   ERZWINGEN=1      Schutz gegen Doppel-Abschicken uebergehen

set -uo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELBST="$HIER/$(basename "${BASH_SOURCE[0]}")"
export QIRE_CODE="${QIRE_CODE:-$(cd "$HIER/../.." && pwd)}"
export QIRE_T002_ZELLEN="${QIRE_T002_ZELLEN:-$QIRE_CODE/results/t002_zellen}"
export PYTHONIOENCODING=utf-8
PY="${QIRE_PYTHON:-python}"
if [ -n "${QIRE_UMGEBUNG:-}" ]; then
  # shellcheck disable=SC1090
  source "$QIRE_UMGEBUNG"
fi
PLAENE="$QIRE_T002_ZELLEN/plaene"
LOGS="$QIRE_T002_ZELLEN/logs"
mkdir -p "$PLAENE" "$LOGS"
ZUSATZ=""
if [ "${MIT_ZUSATZARM:-0}" = "1" ]; then
  ZUSATZ="--mit-zusatzarm"
fi

melde() { echo; echo "=== $(date -u '+%F %T UTC')  $* ==="; }

modul() {
  ( cd "$QIRE_CODE" || exit 1
    # shellcheck disable=SC2086
    "$PY" -m quantum_sim.sprachmodell "$@" )
}

korpus_da() {
  : "${QIRE_KORPUS:?QIRE_KORPUS muss auf den Ordner mit train.npy, val.npy, bpe_4096.json zeigen}"
  export QIRE_KORPUS
  local fehlt=0
  for f in train.npy val.npy bpe_4096.json; do
    if [ ! -f "$QIRE_KORPUS/$f" ]; then
      echo "  FEHLT: $QIRE_KORPUS/$f"
      fehlt=1
    fi
  done
  return $fehlt
}

laufende_jobs() {
  squeue -h -u "$USER" -n t002 -o %i 2>/dev/null
}

JID=""
# schicke <lr|messung>
#   0 = Array abgeschickt (JID gesetzt), 1 = nichts offen,
#   2 = Fehler, 3 = gleicher Plan wie beim letzten Mal (Zellen scheitern)
schicke() {
  local phase="$1" plan n vorher
  plan="$PLAENE/${phase}_$(date -u '+%Y%m%d_%H%M%S').txt"
  # shellcheck disable=SC2086
  if ! modul plan "$phase" $ZUSATZ > "$plan"; then
    echo "  'plan $phase' fehlgeschlagen (siehe oben)."
    return 2
  fi
  n=$(grep -c . "$plan")
  if [ "$n" -eq 0 ]; then
    echo "  $phase: keine offenen Zellen."
    mv "$plan" "$plan.leer"
    return 1
  fi
  vorher=$(ls -1 "$PLAENE/${phase}_"*.txt 2>/dev/null | sort | grep -v -F "$plan" | tail -n 1)
  if [ -n "$vorher" ] && cmp -s "$plan" "$vorher" && [ "${ERZWINGEN:-0}" != "1" ]; then
    echo "  ACHTUNG: derselbe Plan wie $vorher -- diese Zellen sind nach dem"
    echo "  letzten Array immer noch offen, sie scheitern also. Logs in $LOGS pruefen."
    echo "  Mit ERZWINGEN=1 trotzdem abschicken."
    mv "$plan" "$plan.wiederholt"
    return 3
  fi
  echo "  $phase: $n offene Zellen -> $plan"
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable \
          --array=0-$((n - 1))%"${MAX_PARALLEL:-16}" \
          --output="$LOGS/%x_%A_%a.out" \
          --export=ALL,PLAN="$plan" \
          ${SBATCH_EXTRA:-} \
          "$HIER/t002_zelle.sbatch")
  if [ -z "$JID" ]; then
    echo "  sbatch hat keine Job-ID geliefert."
    return 2
  fi
  echo "  abgeschickt: Job-Array $JID"
  return 0
}

nicht_doppelt() {
  local l
  l=$(laufende_jobs)
  if [ -n "$l" ] && [ "${ERZWINGEN:-0}" != "1" ]; then
    echo "  Es laufen noch t002-Jobs ($(echo "$l" | tr '\n' ' ')). Erst abwarten,"
    echo "  sonst werden dieselben Zellen doppelt gerechnet. (ERZWINGEN=1 uebergeht das.)"
    return 1
  fi
  return 0
}

kette() {
  # Kleiner CPU-Job, der nach dem Array (egal wie es endet) `auto` erneut
  # aufruft. ANNAHME: sbatch ist auch von Rechenknoten aus erreichbar. Wenn
  # nicht: `auto` nach Ende jedes Arrays von Hand auf dem Login-Knoten starten.
  local wid
  # shellcheck disable=SC2086
  wid=$(sbatch --parsable --dependency=afterany:"$1" --job-name=t002_weiter \
          --time=00:30:00 --cpus-per-task=1 --mem=4G \
          --output="$LOGS/weiter_%j.out" --export=ALL \
          ${WEITER_EXTRA:-} --wrap="bash '$SELBST' auto")
  echo "  Folgejob $wid startet nach Ende von $1 und schickt die naechste Phase ab."
}

befehl="${1:-status}"
case "$befehl" in
  pruefen)
    melde "Pruefen: Korpus, Python, Selbsttest"
    korpus_da || { echo "  ABBRUCH: Korpus unvollstaendig."; exit 1; }
    echo "  Korpus vollstaendig: $QIRE_KORPUS"
    "$PY" - <<'PY'
import importlib.util, sys
fehlt = [m for m in ("torch", "numpy", "scipy", "sklearn", "psutil", "tokenizers")
         if not importlib.util.find_spec(m)]
if fehlt:
    sys.exit(f"  FEHLT: {', '.join(fehlt)}")
import torch
print(f"  torch {torch.__version__}, CUDA gebaut fuer {torch.version.cuda}, "
      f"hier sichtbar: {torch.cuda.is_available()} (auf dem Login-Knoten oft False)")
PY
    [ $? -eq 0 ] || exit 1
    echo "  Vorregistrierte Schrittzahl: $(modul schritte)"
    modul selbsttest || { echo "  ABBRUCH: Selbsttest fehlgeschlagen."; exit 1; }
    ;;
  rechenbedarf)
    # shellcheck disable=SC2086
    modul rechenbedarf $ZUSATZ
    ;;
  status)
    # shellcheck disable=SC2086
    modul status $ZUSATZ
    laufende_jobs | sed 's/^/  laufend: /'
    ;;
  lr|messung)
    korpus_da || exit 1
    nicht_doppelt || exit 1
    melde "Phase $befehl"
    schicke "$befehl"
    rc=$?
    if [ "$befehl" = "lr" ] && [ $rc -eq 1 ]; then
      echo "  Lernratensuche vollstaendig, lr_map.json geschrieben. Weiter mit: messung"
    fi
    [ $rc -le 1 ] || exit $rc
    ;;
  auto)
    korpus_da || exit 1
    nicht_doppelt || exit 0
    melde "auto: naechste Phase bestimmen"
    schicke lr
    rc=$?
    if [ $rc -eq 0 ]; then kette "$JID"; exit 0; fi
    [ $rc -eq 1 ] || exit $rc
    schicke messung
    rc=$?
    if [ $rc -eq 0 ]; then kette "$JID"; exit 0; fi
    [ $rc -eq 1 ] || exit $rc
    melde "Alle Zellen vorhanden -- zusammenfuehren"
    modul zusammenfuehren
    ;;
  zusammenfuehren)
    melde "Zusammenfuehren"
    modul zusammenfuehren
    ;;
  *)
    echo "unbekannt: $befehl. Erlaubt: pruefen | rechenbedarf | status | lr | messung | auto | zusammenfuehren"
    exit 1
    ;;
esac
