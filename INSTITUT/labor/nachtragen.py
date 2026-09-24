"""Lehre zu einem automatischen FEHLERBUCH-Eintrag nachtragen -- gezielt ueber
die Punkt-ID, nicht ueber „den ersten Platzhalter“ (zweimal schiefgegangen, 24.09.).

    python -m labor.nachtragen I-008 "Lehre: ..."
"""
import sys
from pathlib import Path

PFAD = Path(__file__).resolve().parent.parent / "FEHLERBUCH.md"
PLATZHALTER = "- *Was lernen wir daraus? — vom Menschen/Claude nachzutragen.*"


def nachtragen(pid: str, text: str) -> bool:
    zeilen = PFAD.read_text(encoding="utf-8").split("\n")
    kopf = [i for i, z in enumerate(zeilen) if z.startswith("### ") and f"— {pid} (automatisch)" in z]
    for start in reversed(kopf):                     # neuester Eintrag dieses Punkts zuerst
        for i in range(start + 1, len(zeilen)):
            if zeilen[i].startswith("### "):
                break
            if zeilen[i] == PLATZHALTER:
                zeilen[i] = "- **Lehre:** " + text
                PFAD.write_text("\n".join(zeilen), encoding="utf-8")
                return True
    return False


if __name__ == "__main__":
    print("eingetragen" if nachtragen(sys.argv[1], " ".join(sys.argv[2:])) else "kein offener Platzhalter fuer " + sys.argv[1])
