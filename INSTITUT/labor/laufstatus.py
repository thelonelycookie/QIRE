"""Lauf-Waechter (Plan R1.4): Herzschlag + Fortschritt in `laeufe/<ID>.status.json`.

Warum: Wartebefehle duerfen NUR diese kleine Datei lesen -- kein `pgrep`, kein
Log-Lesen (das hat beim T-006-Lauf den Speicher gesprengt).

    from labor import laufstatus
    laufstatus.start("I-050", gesamt=10)          # beim Start (macht direktor.lauf automatisch)
    laufstatus.herzschlag("I-050", seed=3, gesamt=10, text="Arm B")   # nach jedem Seed
    laufstatus.fertig("I-050", ok=True)           # am Ende (auch bei Absturz: ok=False)
    laufstatus.lese("I-050")                      # dict oder None, mit Feld "zustand"
    laufstatus.warte_bis_fertig("I-050", timeout=3600)

Zustand: laeuft | fertig | fehler | tot (Herzschlag aelter als TOT_NACH Sekunden).
Schreiben ist atomar (tmp + os.replace). Reine Standardbibliothek.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
LAEUFE = WURZEL / "laeufe"
TOT_NACH = 600          # Sekunden ohne Herzschlag -> "tot" (10 min)


def pfad(pid: str) -> Path:
    return LAEUFE / f"{pid}.status.json"


def _schreibe(pid: str, d: dict):
    LAEUFE.mkdir(parents=True, exist_ok=True)
    p = pfad(pid)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _roh(pid: str):
    try:
        return json.loads(pfad(pid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rechner() -> str:
    return os.environ.get("INSTITUT_ORT") or platform.node() or "unbekannt"


def start(pid: str, gesamt: int | None = None, text: str = "") -> None:
    jetzt = time.time()
    _schreibe(pid, {"id": pid, "zustand": "laeuft", "pid": os.getpid(), "rechner": _rechner(),
                    "plattform": platform.system(), "start": jetzt, "herzschlag": jetzt,
                    "seed": 0, "gesamt": gesamt, "text": text, "ok": None})


def herzschlag(pid: str, seed: int | None = None, gesamt: int | None = None, text: str | None = None) -> None:
    """Fortschritt melden (Seed `seed` von `gesamt` fertig/in Arbeit). Legt die Datei bei Bedarf an."""
    d = _roh(pid)
    if d is None or d.get("zustand") != "laeuft":
        start(pid, gesamt, text or "")
        d = _roh(pid) or {}
    d["herzschlag"] = time.time()
    d["pid"], d["rechner"] = os.getpid(), _rechner()
    if seed is not None:
        d["seed"] = seed
    if gesamt is not None:
        d["gesamt"] = gesamt
    if text is not None:
        d["text"] = text
    _schreibe(pid, d)


def fertig(pid: str, ok: bool = True, text: str = "") -> None:
    d = _roh(pid) or {"id": pid, "start": time.time(), "pid": os.getpid(), "rechner": _rechner()}
    jetzt = time.time()
    d.update({"zustand": "fertig" if ok else "fehler", "ok": ok, "herzschlag": jetzt, "ende": jetzt})
    if text:
        d["text"] = text
    if ok and d.get("gesamt"):
        d["seed"] = d["gesamt"]
    _schreibe(pid, d)


def lese(pid: str):
    """Statusdatei als dict (mit abgeleitetem `zustand`, `alter_s`) oder None. Liest nur diese Datei."""
    d = _roh(pid)
    if d is None:
        return None
    d["alter_s"] = time.time() - d.get("herzschlag", 0)
    if d.get("zustand") == "laeuft" and d["alter_s"] > TOT_NACH:
        d["zustand"] = "tot"
    return d


def kurz(d: dict) -> str:
    """Eine Zeile fuer status/warte."""
    fort = f"Seed {d.get('seed', 0)}/{d['gesamt']}" if d.get("gesamt") else f"Seed {d.get('seed', 0)}"
    return (f"{d['id']}: {d['zustand']} ({fort}, {d.get('rechner', '?')}, PID {d.get('pid', '?')}, "
            f"Herzschlag vor {int(d['alter_s'])} s)" + (f" -- {d['text']}" if d.get("text") else ""))


def warte_bis_fertig(pid: str, timeout: float = 3600, intervall: float = 30, ausgabe=print) -> str:
    """Blockiert bis fertig/fehler/tot oder timeout. Rueckgabe: Zustand oder 'timeout' / 'keine_datei'."""
    ende = time.time() + timeout
    while True:
        d = lese(pid)
        if d is None:
            return "keine_datei"
        if d["zustand"] != "laeuft":
            return d["zustand"]
        if time.time() >= ende:
            return "timeout"
        if ausgabe:
            ausgabe(kurz(d))
        time.sleep(min(intervall, max(0.0, ende - time.time())))
