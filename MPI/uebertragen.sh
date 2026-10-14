#!/bin/bash
# Daten/Code zum MPI-Cluster und Ergebnisse zurueck (rsync ueber SSH). KEINE Zugangsdaten in diesem Skript:
# Host, Benutzer und Schluessel stehen nur in deiner ~/.ssh/config unter dem Alias aus MPI/lokal.env (MPI_SSH_ALIAS).
#   MPI/uebertragen.sh hin      [--trocken]   Code + Daten -> Cluster   (ohne Ergebnisse, Git-Ordner, Geheimnisse)
#   MPI/uebertragen.sh zurueck  [--trocken]   Ergebnisse/Statusdateien -> lokal (nur NEUE/geaenderte Dateien, nichts loeschen)
#   MPI/uebertragen.sh modell <ordner>        grosse Modelle/Datensaetze einzeln hochladen
# --trocken zeigt nur, was passieren wuerde (rsync -n). Es wird NIE mit --delete synchronisiert.
set -euo pipefail
WURZEL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$WURZEL/MPI/lokal.env" ] && { set -a; . "$WURZEL/MPI/lokal.env"; set +a; }
: "${MPI_SSH_ALIAS:?MPI_SSH_ALIAS fehlt (MPI/lokal.env)}"; : "${MPI_ZIEL_ORDNER:?MPI_ZIEL_ORDNER fehlt (MPI/lokal.env)}"
RICHTUNG="${1:?hin | zurueck | modell <ordner>}"; shift
N=(); [ "${1:-}" = "--trocken" ] && { N=(-n); shift; }
R=(rsync -avh --partial --info=progress2 "${N[@]}")
case "$RICHTUNG" in
    hin)
        "${R[@]}" --exclude='.git/' --exclude='__pycache__/' --exclude='logs/' --exclude='MPI/lokal.env' \
            --exclude='*.key' --exclude='*.pem' --exclude='id_rsa*' --exclude='id_ed25519*' --exclude='*.token' --exclude='*token*.txt' --exclude='*secret*' \
            --exclude='INSTITUT/ergebnisse/' --exclude='MPI/adapter/' --exclude='MPI/checkpoints/' \
            "$WURZEL/INSTITUT" "$WURZEL/A" "$WURZEL/MPI" "$WURZEL/CLAUDE.md" "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/"
        # Ergebnisse (inkl. _zwischen_*) getrennt: nur HINAUF, was dem Lauf als Wiederaufnahme-Stand dient
        "${R[@]}" --update "$WURZEL/INSTITUT/ergebnisse/" "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/INSTITUT/ergebnisse/" ;;
    zurueck)
        "${R[@]}" --update "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/INSTITUT/ergebnisse/" "$WURZEL/INSTITUT/ergebnisse/"
        "${R[@]}" --update "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/INSTITUT/laeufe/" "$WURZEL/INSTITUT/laeufe/"
        "${R[@]}" --update "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/MPI/adapter/" "$WURZEL/MPI/adapter/" || true
        "${R[@]}" --update "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/logs/" "$WURZEL/logs/" || true
        echo "Hinweis: agenda.json/LABORBUCH.md werden NICHT zurueckgespielt (Merge von Hand / durch Claude)." ;;
    modell)
        QUELLE="${1:?Ordner}"; "${R[@]}" "$QUELLE" "$MPI_SSH_ALIAS:$MPI_ZIEL_ORDNER/modelle/" ;;
    *) echo "Unbekannt: $RICHTUNG" >&2; exit 2 ;;
esac
