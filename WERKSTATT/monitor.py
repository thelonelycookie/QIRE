"""Überwacht einen langen Lauf ohne LLM und schreibt den Stand nach
WERKSTATT/ergebnisse/<auftrag>/status.json. Am Ende (oder bei Absturz)
committet und pusht es den Status, damit der Planer in der Cloud ihn sieht.

Aufruf:  A\\py.cmd ..\\WERKSTATT\\monitor.py --pid 1234 --log lauf.log --auftrag W-001
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

WERKSTATT = Path(__file__).resolve().parent
PROJEKT = WERKSTATT.parent


def letzte_zeilen(log: Path, n: int = 15) -> list[str]:
    if not log or not log.exists():
        return []
    with log.open(encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


def git_status_pushen(datei: Path, auftrag: str, text: str) -> None:
    rel = datei.relative_to(PROJEKT)
    subprocess.run(["git", "add", str(rel)], cwd=PROJEKT)
    subprocess.run(["git", "commit", "-m", f"{auftrag}: Monitor — {text}"], cwd=PROJEKT)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=PROJEKT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--log", type=Path)
    p.add_argument("--auftrag", required=True)
    p.add_argument("--intervall", type=int, default=60, help="Sekunden")
    p.add_argument("--ram-grenze", type=float, default=90.0, help="Prozent RAM für Warnung")
    a = p.parse_args()

    ziel = WERKSTATT / "ergebnisse" / a.auftrag
    ziel.mkdir(parents=True, exist_ok=True)
    datei = ziel / "status.json"
    start = time.time()
    warnung = False

    while psutil.pid_exists(a.pid):
        ram = psutil.virtual_memory().percent
        stand = {
            "auftrag": a.auftrag,
            "pid": a.pid,
            "zustand": "laeuft",
            "zeit": datetime.now().isoformat(timespec="seconds"),
            "laufzeit_min": round((time.time() - start) / 60, 1),
            "ram_prozent": ram,
            "log_ende": letzte_zeilen(a.log),
        }
        datei.write_text(json.dumps(stand, ensure_ascii=False, indent=2), encoding="utf-8")
        if ram > a.ram_grenze and not warnung:
            git_status_pushen(datei, a.auftrag, f"WARNUNG RAM {ram:.0f} %")
            warnung = True
        time.sleep(a.intervall)

    ende = letzte_zeilen(a.log, 40)
    fehler = any("Traceback" in z or "MemoryError" in z for z in ende)
    stand = {
        "auftrag": a.auftrag,
        "pid": a.pid,
        "zustand": "abgestuerzt" if fehler else "beendet",
        "zeit": datetime.now().isoformat(timespec="seconds"),
        "laufzeit_min": round((time.time() - start) / 60, 1),
        "log_ende": ende,
    }
    datei.write_text(json.dumps(stand, ensure_ascii=False, indent=2), encoding="utf-8")
    git_status_pushen(datei, a.auftrag, stand["zustand"])


if __name__ == "__main__":
    main()
