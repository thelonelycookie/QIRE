"""Startet das Audit (qire_ai.audit) aus dem Buendel B heraus.

WARUM ES DAS GIBT
Das Audit lief bisher nur im Archiv: `code/` ist ein flacher Spiegel, dessen
Module sich relativ importieren (`from .result_store import ...`) und deshalb
als Einzelskript nicht starten. Dieses Skript laedt `code/` als Paket
`qire_ai` -- am Spiegel selbst wird nichts geaendert -- und ruft das Audit mit
den Rohdaten aus `daten/` und dem Tagebuchauszug `TAGEBUCH_B.md` auf.

Run (aus diesem Ordner, mit dem Interpreter des Projekts):
    python pruefen_audit.py
Weitere Optionen werden durchgereicht, z. B. --daten <weiterer Ordner>.
"""

import sys
import types
from pathlib import Path

HIER = Path(__file__).resolve().parent

paket = types.ModuleType("qire_ai")
paket.__path__ = [str(HIER / "code")]
sys.modules["qire_ai"] = paket

from qire_ai import audit  # noqa: E402

if __name__ == "__main__":
    argumente = ["--daten", str(HIER / "daten"),
                 "--tagebuch", str(HIER / "TAGEBUCH_B.md")] + sys.argv[1:]
    raise SystemExit(0 if audit.main(argumente) == 0 else 1)
