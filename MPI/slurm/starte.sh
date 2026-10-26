#!/bin/bash
# Reicht QIRE-Jobs ein und setzt Partition/Account/GPU-Angabe aus MPI/lokal.env (ohne die .sbatch-Dateien zu aendern).
#   MPI/slurm/starte.sh cpu <ID> [--schnell] [--kette N] [--trocken]
#   MPI/slurm/starte.sh test                            (erster Test: vortest, 30 min)
#   MPI/slurm/starte.sh gpu <s3|s4> [--trocken]        (BASIS_MODELL, TRAIN_DATEN ... als Umgebungsvariablen)
# --kette N : N Glieder hintereinander (Slurm --dependency=afterany); noetig, wenn `scontrol requeue` verboten ist.
#             Glieder, die ihren Lauf fertig vorfinden, beenden sich sofort.
# QIRE_KERNE / QIRE_SPEICHER / QIRE_ZEIT (Umgebung oder lokal.env): ueberschreiben --cpus-per-task / --mem / --time.
# --trocken : nur den sbatch-Befehl anzeigen.
set -euo pipefail
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WURZEL="$(cd "$HIER/../.." && pwd)"
[ -f "$WURZEL/MPI/lokal.env" ] && { set -a; . "$WURZEL/MPI/lokal.env"; set +a; }
ART="${1:?Aufruf: starte.sh cpu <ID> | gpu <s3|s4>}"; shift
TROCKEN=0; KETTE=1; REST=()
while [ $# -gt 0 ]; do
    case "$1" in
        --trocken) TROCKEN=1 ;;
        --kette)   KETTE="${2:?--kette N}"; shift ;;
        *)         REST+=("$1") ;;
    esac
    shift
done
[ "$TROCKEN" = 1 ] || mkdir -p "$WURZEL/logs"
cd "$WURZEL"
SB=(sbatch --parsable --export=ALL,QIRE_WURZEL="$WURZEL")
case "$ART" in
    cpu)
        [ "${#REST[@]}" -ge 1 ] || { echo "Lauf-ID fehlt" >&2; exit 2; }
        [ -n "${SLURM_PARTITION_CPU:-}" ] && SB+=(--partition="$SLURM_PARTITION_CPU")
        [ -n "${QIRE_KERNE:-}" ] && SB+=(--cpus-per-task="$QIRE_KERNE")
        [ -n "${QIRE_SPEICHER:-}" ] && SB+=(--mem="$QIRE_SPEICHER")
        [ -n "${QIRE_ZEIT:-}" ] && SB+=(--time="$QIRE_ZEIT")
        SKRIPT="MPI/slurm/cpu_lauf.sbatch"
        # Lauf-ID muss in der Agenda stehen (sonst ist der Code veraltet, z. B. git pull an lokalen Aenderungen gescheitert)
        grep -q "\"${REST[0]}\"" INSTITUT/agenda.json || { echo "FEHLER: ${REST[0]} fehlt in INSTITUT/agenda.json. Erst 'git stash && git pull'." >&2; exit 2; } ;;
    test)
        [ -n "${SLURM_PARTITION_CPU:-}" ] && SB+=(--partition="$SLURM_PARTITION_CPU")
        SKRIPT="MPI/slurm/erster_test.sbatch" ;;
    gpu)
        [ "${#REST[@]}" -ge 1 ] || { echo "Stufe s3|s4 fehlt" >&2; exit 2; }
        export QLORA_STUFE="${REST[0]}"; REST=()
        [ -n "${SLURM_PARTITION_GPU:-}" ] && SB+=(--partition="$SLURM_PARTITION_GPU")
        [ -n "${SLURM_GPU_ANGABE:-}" ] && SB+=(--gres="$SLURM_GPU_ANGABE")
        SKRIPT="MPI/slurm/gpu_qlora.sbatch" ;;
    *) echo "Unbekannte Art: $ART" >&2; exit 2 ;;
esac
[ -n "${SLURM_ACCOUNT:-}" ] && SB+=(--account="$SLURM_ACCOUNT")
# Code aktuell? (nur Warnung; Rechenknoten brauchen das nicht)
if [ "$TROCKEN" = 0 ] && git fetch -q origin main 2>/dev/null; then
    HINTER=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
    [ "$HINTER" -gt 0 ] && echo "WARNUNG: lokaler Code liegt $HINTER Commit(s) hinter origin/main. 'git stash && git pull' ausfuehren?" >&2
fi
VORHER=""
for i in $(seq 1 "$KETTE"); do
    BEF=("${SB[@]}")
    [ -n "$VORHER" ] && BEF+=(--dependency="afterany:$VORHER")
    BEF+=("$SKRIPT" ${REST[@]+"${REST[@]}"})
    if [ "$TROCKEN" = 1 ]; then echo "${BEF[*]}"; VORHER="JOB$i"; continue; fi
    VORHER="$("${BEF[@]}")"
    echo "Glied $i/$KETTE eingereiht: Job $VORHER"
done
