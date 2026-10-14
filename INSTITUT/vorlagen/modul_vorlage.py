"""Abteilung __ABTEILUNG__ -- Karte __ID__ (Modulgeruest aus vorlagen/modul_vorlage.py).

HEILMEIER (VOR dem ersten Lauf ausfuellen, dann committen, dann rechnen):
  Was wollen wir?   TODO
  Wie heute?        TODO
  Was ist neu?      TODO (Literatursuche, Vorgabe 7)
  Woran merken wir? TODO (Metrik, Schwelle, Abbruchbedingung)

VORREGISTRIERT: siehe HYPOTHESEN. Urteile sind Strings, die mit H1/H0/HU
beginnen (der Direktor schreibt H0/HU ins FEHLERBUCH). Lernrate je Arm ueber
labor.gemeinsam.gittersuche_stabil/waehle_rate (Stabilitaetsabstand), NICHT
ueber eine einfache 2-Seed-Mittelwertsuche. Wiederaufnahme: jeder Seed wird
einzeln in laeufe/__ID__.zwischen.json gesichert (atomar); ein abgebrochener
Lauf rechnet nur die fehlenden Seeds nach. Herzschlag nach jedem Seed
(labor.laufstatus) -- `python direktor.py warte __ID__` liest nur diese Datei.

PROTOKOLL: Mess-Seeds MESS_SEEDS, Tuning-Seeds TUNE_SEEDS (Validierung, nie Test).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labor import laufstatus  # noqa: E402
from labor.gemeinsam import (ERGEBNISSE, WURZEL, gittersuche_stabil, parallel,  # noqa: E402
                             wilcoxon_exakt, zusammenfassung)

KARTE = "__ID__"
MESS_SEEDS = list(range(100, 110))      # TODO: frische Seeds, in LABORBUCH eintragen
TUNE_SEEDS = [190, 191]                 # TODO: Validierung aus dem Trainingsteil
SCHWELLE = 0.01                         # TODO: Mindesteffekt (vorab begruenden, nicht an Aufloesung vorbei)
ARME = ("pruef", "kontrolle")           # TODO: Pruefarm gegen Kontrollarm(e)

HYPOTHESEN = {
    "H1": "TODO: Pruefarm schlaegt Kontrolle um >= SCHWELLE (gepaart ueber Seeds, Wilcoxon p < 0,05).",
}


# ---------------------------------------------------------------- Urteil (rein, testbar)
def urteil_h1(d, schwelle=SCHWELLE) -> str:
    """d = Differenzen je Seed (Pruefarm - Kontrolle). Jeder Zweig ist im Vortest erzwungen."""
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return "HU (zu wenige Seeds)"
    m = float(d.mean())
    p = wilcoxon_exakt(d)
    if m >= schwelle and p < 0.05:
        return f"H1 (mittel {m:+.4f}, p {p:.3f})"
    ki_oben = m + 1.645 * float(d.std(ddof=1)) / np.sqrt(len(d))
    if ki_oben < schwelle:
        return f"H0 (KI-oben {ki_oben:+.4f} < {schwelle})"
    return f"HU (mittel {m:+.4f}, KI-oben {ki_oben:+.4f} ueberdeckt {schwelle})"


# ---------------------------------------------------------------- Arbeiter (TODO: echte Aufgabe)
def _arbeiter(a) -> dict:
    """a = (arm, seed, lr). Rueckgabe {'acc': ...} (hoeher = besser; auch fuer gittersuche_stabil).
    TODO: ersetzen. Platzhalter: synthetisches Rauschmodell, damit das Geruest laeuft."""
    arm, seed, lr = a
    rng = np.random.default_rng(seed * 7 + (1 if arm == ARME[0] else 0))
    return {"acc": float(0.8 - abs(np.log2(lr / 0.08)) * 0.02 + rng.normal(0, 0.005)
                         + (0.02 if arm == ARME[0] else 0.0))}


def waehle_raten(prozesse=None, gitter=(0.02, 0.04, 0.08, 0.16, 0.32)):
    """Eigene Lernrate je Arm mit Stabilitaetsabstand (gittersuche_stabil -> waehle_rate)."""
    info, _ = gittersuche_stabil(list(ARME), lambda arm, lr: [(arm, s, lr) for s in TUNE_SEEDS],
                                 lambda aufg: parallel(_arbeiter, aufg, prozesse), gitter=gitter)
    return {arm: info[arm]["lr"] for arm in ARME}, info


# ---------------------------------------------------------------- Zwischenstand je Seed
def _lade_zw(pfad: Path, kennung: dict) -> dict:
    if not pfad.exists():
        return {}
    inhalt = json.loads(pfad.read_text(encoding="utf-8"))
    if inhalt.get("kennung") != kennung:
        raise ValueError(f"Zwischendatei {pfad} stammt aus anderen Einstellungen -- nicht mischen; umbenennen")
    return inhalt["seeds"]


def _schreibe_zw(pfad: Path, kennung: dict, seeds: dict):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"kennung": kennung, "seeds": seeds}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, pfad)


# ---------------------------------------------------------------- Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, **_):
    t0 = time.time()
    seeds = MESS_SEEDS[:2] if schnell else MESS_SEEDS
    gitter = (0.04, 0.08, 0.16) if schnell else (0.02, 0.04, 0.08, 0.16, 0.32)
    raten, info = waehle_raten(prozesse, gitter)
    kennung = {"karte": KARTE, "schnell": schnell, "raten": raten, "seeds": seeds}
    pfad = Path(zwischen) if zwischen else WURZEL / "laeufe" / f"{KARTE}{'-schnell' if schnell else ''}.zwischen.json"
    zw = _lade_zw(pfad, kennung)
    laufstatus.herzschlag(KARTE, seed=len(zw), gesamt=len(seeds), text="Tuning fertig")
    for s in seeds:
        if str(s) in zw:
            continue                                   # Wiederaufnahme: Seed schon gerechnet
        zeilen = parallel(_arbeiter, [(arm, s, raten[arm]) for arm in ARME], prozesse)
        zw[str(s)] = {arm: z["acc"] for arm, z in zip(ARME, zeilen)}
        _schreibe_zw(pfad, kennung, zw)
        laufstatus.herzschlag(KARTE, seed=len(zw), gesamt=len(seeds), text=f"Seed {s}")
    d = [zw[str(s)][ARME[0]] - zw[str(s)][ARME[1]] for s in seeds]
    return {"abteilung": "__ABTEILUNG__", "modul": Path(__file__).stem, "karte": KARTE, "version": 1,
            "hypothesen": HYPOTHESEN, "raten": raten, "tuning": info, "zeilen": zw,
            "auswertung": {"differenz": zusammenfassung(d), "p_wilcoxon": wilcoxon_exakt(d)},
            "urteil": {"H1": urteil_h1(d)}, "zwischendatei": str(pfad), "sekunden": time.time() - t0}


# ---------------------------------------------------------------- Vortest (Pflicht)
def vortest() -> bool:
    """Jeder Urteilszweig wird auf synthetischen Daten ERZWUNGEN; Wiederaufnahme wird geprueft."""
    import tempfile
    ok = True
    rng = np.random.default_rng(0)
    faelle = {"H1": 0.05 + rng.normal(0, 0.002, 10),      # klar ueber Schwelle
              "H0": rng.normal(0, 0.001, 10),             # klar darunter
              "HU": rng.normal(SCHWELLE, 0.02, 4),        # KI ueberdeckt Schwelle
              "HU (zu wenige": np.array([0.1])}
    for erwartet, d in faelle.items():
        u = urteil_h1(d)
        gut = isinstance(u, str) and u.startswith(erwartet)
        print(f"  Zweig {erwartet}: {'ok' if gut else 'FALSCH -> ' + str(u)}")
        ok &= gut
    with tempfile.TemporaryDirectory() as tmp:           # Zwischenstand-Roundtrip
        p, k = Path(tmp) / "z.json", {"karte": KARTE}
        _schreibe_zw(p, k, {"1": {"a": 0.5}})
        gut = _lade_zw(p, k) == {"1": {"a": 0.5}}
        try:
            _lade_zw(p, {"karte": "anders"})
            gut = False
        except ValueError:
            pass
        print(f"  Zwischenstand/Wiederaufnahme: {'ok' if gut else 'FALSCH'}")
        ok &= gut
    r, _ = waehle_raten(prozesse=1, gitter=(0.04, 0.08, 0.16))   # Lernrate ueber gittersuche_stabil
    gut = all(v > 0 for v in r.values())
    print(f"  gittersuche_stabil: {'ok' if gut else 'FALSCH'} {r}")
    return bool(ok and gut)


if __name__ == "__main__":
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
