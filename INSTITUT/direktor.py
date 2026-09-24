"""Der Direktor des Instituts -- koordiniert die Abteilungen, ohne KI-Tokens.

    python direktor.py status          # STAND.md neu schreiben und zeigen
    python direktor.py vortest         # Vortest aller Abteilungen (Pflicht)
    python direktor.py naechstes       # naechsten offenen Punkt der Agenda rechnen
    python direktor.py lauf C-bewusst  # einen bestimmten Punkt rechnen
    python direktor.py lauf C-bewusst --schnell   # Rauchtest (Minuten)
    python direktor.py alle            # alle offenen Punkte nacheinander (ueber Nacht)
    python direktor.py ideen neu 8     # Ideenschmiede: 8 Rohlinge ziehen
    python direktor.py ideen leiten I-003   # Karte durchs Tor an ihre Abteilung

Was er tut: Er liest `agenda.json`, startet die Abteilung, sichert die
Rohdaten, liest das vorregistrierte Urteil und schaltet danach die
Folgefragen frei, die fuer genau diesen Ausgang VOR dem Lauf festgelegt
wurden („vertiefen statt aufgeben“). Jeder H0-/HU-Ausgang landet mit
Datum im FEHLERBUCH -- ein Fehlschlag ist ein Ergebnis.

Token-Sparen: Claude (oder wer auch immer weiterforscht) liest danach nur
`STAND.md` (eine Seite) statt Logs und Rohdaten.

Umgebungsvariable INSTITUT_PROZESSE=5 setzt die Zahl paralleler Prozesse.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent
sys.path.insert(0, str(WURZEL))
AGENDA = WURZEL / "agenda.json"

from labor.gemeinsam import ERGEBNISSE, speichere  # noqa: E402


def lade():
    return json.loads(AGENDA.read_text(encoding="utf-8"))


def sichere(agenda):
    AGENDA.write_text(json.dumps(agenda, indent=1, ensure_ascii=False), encoding="utf-8")


def neuestes_ergebnis(pid):
    d = sorted(ERGEBNISSE.glob(f"{pid}__*.json"))
    return json.loads(d[-1].read_text()) if d else None


def naechster(agenda, ausser=()):
    """Offener Punkt mit Modul, hoechste Prioritaet zuerst (1 = Q3 ... 3 = Q1).
    Q1-Punkte (nur Analogie) werden nicht gestrichen, nur spaeter gerechnet."""
    offen = [p for p in agenda["punkte"] if p["status"] == "offen" and p.get("modul") and p["id"] not in ausser]
    return min(offen, key=lambda p: p.get("prioritaet", 1), default=None)


def status():
    agenda = lade()
    z = ["# STAND des Instituts (automatisch, `python direktor.py status`)", "",
         f"Erzeugt {time.strftime('%d.%m.%Y %H:%M')}. Eine Seite -- mehr muss niemand lesen, "
         "um weiterzumachen.", "",
         "| Punkt | Abteilung | Prio | Status | Frage | Letztes Urteil |", "|---|---|---|---|---|---|"]
    for p in agenda["punkte"]:
        e = neuestes_ergebnis(p["id"])
        u = "; ".join(f"{k}: {v}" for k, v in (e or {}).get("urteil", {}).items()) or p.get("notiz", "–")
        z.append(f"| {p['id']} | {p['abteilung']} | {p.get('prioritaet', 1)} | {p['status']} | {p['frage']} | {u} |")
    try:
        from abteilungen.ideenschmiede import lade as lade_ideen
        karten = lade_ideen()["karten"]
        z += ["", "**Ideenschmiede:** " + " -> ".join(
            f"{s} {sum(k['status'] == s for k in karten)}"
            for s in ("roh", "geprueft", "neu", "weitergeleitet", "verworfen"))]
    except Exception:
        pass
    z += ["", "Naechster offener Punkt: " + ((naechster(agenda) or {}).get("id") or "keiner")]
    (WURZEL / "STAND.md").write_text("\n".join(z) + "\n", encoding="utf-8")
    print("\n".join(z))


def fehlerbuch(pid, urteil):
    schlecht = {k: v for k, v in urteil.items() if v.startswith(("H0", "HU"))}
    if not schlecht:
        return
    with open(WURZEL / "FEHLERBUCH.md", "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%d.%m.%Y %H:%M')} — {pid} (automatisch)\n")
        for k, v in schlecht.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("- *Was lernen wir daraus? — vom Menschen/Claude nachzutragen.*\n")


def lauf(pid, schnell=False):
    agenda = lade()
    p = next(q for q in agenda["punkte"] if q["id"] == pid)
    mod = importlib.import_module(p["modul"])
    if hasattr(mod, "vortest") and not mod.vortest():
        print("Vortest DURCHGEFALLEN -- kein Lauf.")
        return
    fn = getattr(mod, p.get("funktion", "lauf"))
    e = fn(schnell=schnell, **p.get("argumente", {}))
    pfad = speichere(pid + ("-schnell" if schnell else ""), e)
    print("gesichert:", pfad)
    print("Urteil:", e.get("urteil"))
    if schnell:
        return
    # Agenda NEU laden: waehrend des (stundenlangen) Laufs kann sie jemand
    # geaendert haben -- nur diesen Punkt und seine Folgen anfassen.
    agenda = lade()
    p = next(q for q in agenda["punkte"] if q["id"] == pid)
    p["status"] = "erledigt"
    fehlerbuch(pid, e.get("urteil", {}))
    # Vorregistrierte Folgefragen freischalten: {"C-1:H0": ["C-bewusst"], ...}
    for bedingung, folgen in p.get("folgen", {}).items():
        frage, marke = bedingung.split(":")
        if e.get("urteil", {}).get(frage, "").startswith(marke):
            for f in folgen:
                q = next(x for x in agenda["punkte"] if x["id"] == f)
                if q["status"] == "wartet":
                    q["status"] = "offen"
                    print(f"  -> Folgefrage freigeschaltet: {f} (weil {bedingung})")
    sichere(agenda)
    status()


if __name__ == "__main__":
    arg = sys.argv[1:] or ["status"]
    if arg[0] == "status":
        status()
    elif arg[0] == "vortest":
        for p in lade()["punkte"]:
            if p.get("modul"):
                m = importlib.import_module(p["modul"])
                if hasattr(m, "vortest"):
                    print(p["id"], "Vortest:", "bestanden" if m.vortest() else "DURCHGEFALLEN")
    elif arg[0] == "naechstes":
        p = naechster(lade())
        lauf(p["id"], "--schnell" in arg) if p else print("Agenda leer.")
    elif arg[0] == "ideen":
        from abteilungen import ideenschmiede
        if len(arg) > 1 and arg[1] == "neu":
            ideenschmiede.neu(int(arg[2]) if len(arg) > 2 else 8)
        elif len(arg) > 2 and arg[1] == "leiten":
            ideenschmiede.leiten(arg[2])
        ideenschmiede.liste()
    elif arg[0] == "alle":
        erledigt = set()
        while True:
            p = naechster(lade(), erledigt)
            if not p:
                print("Agenda leer -- alles gerechnet, was freigeschaltet ist.")
                break
            erledigt.add(p["id"])
            try:
                lauf(p["id"])
            except Exception as fehler:          # ein Absturz haelt die Nacht nicht an
                print("ABSTURZ in", p["id"], ":", repr(fehler))
    elif arg[0] == "lauf":
        lauf(arg[1], "--schnell" in arg)
