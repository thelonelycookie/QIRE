"""Der Direktor des Instituts -- koordiniert die Abteilungen, ohne KI-Tokens.

    python direktor.py status          # STAND.md neu schreiben und zeigen
    python direktor.py vortest         # Vortest aller Abteilungen (Pflicht)
    python direktor.py naechstes       # naechsten offenen Punkt der Agenda rechnen
    python direktor.py lauf C-bewusst  # einen bestimmten Punkt rechnen
    python direktor.py lauf C-bewusst --schnell   # Rauchtest (Minuten)
    python direktor.py alle            # alle offenen Punkte nacheinander (ueber Nacht)
    python direktor.py ideen neu 8     # Ideenschmiede: 8 Rohlinge ziehen
    python direktor.py ideen leiten I-003   # Karte durchs Tor an ihre Abteilung
    python direktor.py offen           # nur Punkte, die auf Malte warten
    python direktor.py neu I-050 [Abteilung]   # Modulgeruest aus vorlagen/ + Agenda-Eintrag (idee)
    python direktor.py warte I-050 [--timeout=3600]   # liest NUR laeufe/<ID>.status.json

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
import re
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent
sys.path.insert(0, str(WURZEL))
AGENDA = WURZEL / "agenda.json"

from labor import laufstatus  # noqa: E402
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
        u = "; ".join(f"{k}: {_text(v)}" for k, v in (e or {}).get("urteil", {}).items()) or p.get("notiz", "–")
        z.append(f"| {p['id']} | {p['abteilung']} | {p.get('prioritaet', 1)} | {p['status']} | {p['frage']} | {u} |")
    try:
        from abteilungen.ideenschmiede import lade as lade_ideen
        karten = lade_ideen()["karten"]
        z += ["", "**Ideenschmiede:** " + " -> ".join(
            f"{s} {sum(k['status'] == s for k in karten)}"
            for s in ("roh", "geprueft", "neu", "weitergeleitet", "verworfen"))]
    except Exception:
        pass
    z += _uebersicht(agenda)
    z += ["", "Naechster offener Punkt: " + ((naechster(agenda) or {}).get("id") or "keiner")]
    (WURZEL / "STAND.md").write_text("\n".join(z) + "\n", encoding="utf-8")
    print("\n".join(z))


def _uebersicht(agenda):
    """R1.1: laeuft wo / wartet auf Malte / fertig. Liest agenda + laeufe/*.status.json (sonst nichts)."""
    z = ["", "## Uebersicht", ""]
    laeuft = []
    for p in agenda["punkte"]:
        d = laufstatus.lese(p["id"])
        if d and d["zustand"] in ("laeuft", "tot"):
            laeuft.append("- " + laufstatus.kurz(d) + (" **(Herzschlag veraltet?)**" if d["zustand"] == "tot" else ""))
        elif p.get("wartet_auf") == "lauf" and p["status"] == "offen":
            laeuft.append(f"- {p['id']}: geplant, noch nicht gestartet (Ort: {_ort(p)})")
    z += ["**Laeuft / geplant:**"] + (laeuft or ["- nichts"])
    malte = [p for p in agenda["punkte"] if p.get("wartet_auf") == "malte"]
    z += ["", f"**Wartet auf Malte ({len(malte)}):**"] + ([f"- {p['id']} ({p['status']}): {_kurz(p)}" for p in malte] or ["- nichts"])
    fertig = [p["id"] for p in agenda["punkte"] if p["status"] == "erledigt"]
    z += ["", f"**Fertig ({len(fertig)}):** " + ", ".join(fertig)]
    return z


def _ort(p):
    return {"pc": "Maltes PC", "container": "Container"}.get(p.get("ort", ""), "offen")


def _kurz(p, n=110):
    t = (p.get("notiz") or p.get("frage") or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[:n - 1] + "…"


def offen():
    """Nur Maltes Entscheidungen (wartet_auf == malte), kompakt."""
    ps = [p for p in lade()["punkte"] if p.get("wartet_auf") == "malte"]
    print(f"Wartet auf Malte: {len(ps)}")
    for p in ps:
        print(f"- {p['id']} [{p['abteilung']}, {p['status']}] {_kurz(p)}")


def neu(pid, abteilung="Neu"):
    """R1.3: Modulgeruest aus vorlagen/modul_vorlage.py + Agenda-Eintrag (status idee)."""
    if not re.fullmatch(r"[A-Za-z]+-[0-9A-Za-z]+(-[0-9A-Za-z]+)*", pid):
        sys.exit(f"Ungueltige ID: {pid!r} (Beispiel: I-050 oder I-050-v2)")
    agenda = lade()
    name = pid.lower().replace("-", "_")
    ziel = WURZEL / "abteilungen" / f"{name}.py"
    if any(p["id"] == pid for p in agenda["punkte"]) or ziel.exists():
        sys.exit(f"{pid} existiert schon (Agenda oder {ziel.name}) -- nichts veraendert.")
    vorlage = (WURZEL / "vorlagen" / "modul_vorlage.py").read_text(encoding="utf-8")
    ziel.write_text(vorlage.replace("__ID__", pid).replace("__ABTEILUNG__", abteilung), encoding="utf-8")
    agenda["punkte"].append({
        "id": pid, "abteilung": abteilung, "status": "idee", "modul": f"abteilungen.{name}",
        "frage": "TODO (Heilmeier im Modulkopf ausfuellen)", "prioritaet": 1, "folgen": {},
        "notiz": f"Geruest {time.strftime('%d.%m.%Y')} aus vorlagen/modul_vorlage.py; erst Vorregistrierung + Vortest, dann status offen.",
        "wartet_auf": "malte", "ort": ""})
    sichere(agenda)
    print(f"Angelegt: abteilungen/{name}.py + Agenda-Eintrag {pid} (status idee).")


def warte(pid, timeout=3600):
    z = laufstatus.warte_bis_fertig(pid, timeout)
    d = laufstatus.lese(pid)
    print(f"{pid}: {z}" + (f" -- {laufstatus.kurz(d)}" if d else ""))
    return z == "fertig"


def _text(v):
    """Urteil als Text -- manche Module liefern {"urteil": "...", ...} (I-020, 24.09. abgestuerzt)."""
    return v.get("urteil", str(v)) if isinstance(v, dict) else str(v)


def fehlerbuch(pid, urteil):
    schlecht = {k: _text(v) for k, v in urteil.items() if _text(v).startswith(("H0", "HU"))}
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
    laufstatus.start(pid, text="schnell" if schnell else "")      # R1.4: Herzschlag-Datei
    try:
        e = fn(schnell=schnell, **p.get("argumente", {}))
    except BaseException as fehler:
        laufstatus.fertig(pid, ok=False, text=repr(fehler)[:200])
        raise
    laufstatus.fertig(pid, ok=True)
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
        if _text(e.get("urteil", {}).get(frage, "")).startswith(marke):
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
    elif arg[0] == "offen":
        offen()
    elif arg[0] == "neu":
        if len(arg) < 2:
            sys.exit("Aufruf: direktor.py neu I-0xx [abteilung]")
        neu(arg[1], arg[2] if len(arg) > 2 else "Neu")
    elif arg[0] == "warte":
        if len(arg) < 2:
            sys.exit("Aufruf: direktor.py warte <ID> [--timeout=Sekunden]")
        t = next((float(a.split("=")[1]) for a in arg if a.startswith("--timeout=")), 3600)
        sys.exit(0 if warte(arg[1], t) else 1)
    elif arg[0] == "lauf":
        lauf(arg[1], "--schnell" in arg)
