#!/bin/bash
# Reicht die drei Laeufe der Welle 27.10.2026 auf MPCDF Raven ein (je ein ganzer CPU-Knoten, 72 Kerne, 24 h).
#   bash MPI/slurm/welle_2026-10-27.sh            # einreichen
#   bash MPI/slurm/welle_2026-10-27.sh --trocken  # nur anzeigen
# Vorher: git pull; INSTITUT/daten/ mit mnist.npz, fashion_mnist.npz, de_gsd_{train,dev,test}.conllu befuellen
# (Rechenknoten haben kein Internet). Danach Ergebnisse per MPI/uebertragen.sh zurueckholen.
# Schaetzung (Container-Schrittzeiten, Aufschlag 1,3): I-048 ~2 h, I-049 ~8 h, I-045-v5 ~19 h (daher Kette 2).
set -euo pipefail
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WURZEL="$(cd "$HIER/../.." && pwd)"
for f in mnist.npz fashion_mnist.npz de_gsd_train.conllu de_gsd_dev.conllu de_gsd_test.conllu; do
    [ -f "${QIRE_DATEN:-$WURZEL/INSTITUT/daten}/$f" ] || { echo "FEHLER: Daten fehlen: $f" >&2; [ "${1:-}" = --trocken ] || exit 2; }
done
export QIRE_KERNE=72 QIRE_SPEICHER=${QIRE_SPEICHER:-200G} QIRE_ZEIT=24:00:00
"$HIER/starte.sh" cpu I-048    --kette 1 "$@"
"$HIER/starte.sh" cpu I-049    --kette 1 "$@"
"$HIER/starte.sh" cpu I-045-v5 --kette 2 "$@"
