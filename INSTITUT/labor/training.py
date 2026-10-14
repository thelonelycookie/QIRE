"""Generische Trainingsschleife des Instituts (Plan R2.1) -- fuer NEUE Module.

Bestehende Module behalten ihre eigenen Schleifen (Ergebnisse festgeschrieben). Diese Datei buendelt, was
dort immer wieder neu gebaut wurde, mit den Protokollregeln (UEBERGABE.md §3, FEHLERBUCH.md) eingebaut:

  * Vollbatch-Adam mit festem Seed (wie labor.gemeinsam.trainiere)          -> trainiere_adam
  * Lernrate je Arm auf eigenen Tuning-Seeds MIT Stabilitaetsabstand,
    Gitter in beide Richtungen verlaengert (kanonisch: gemeinsam.gittersuche_stabil)
                                                                              -> raten_je_arm
  * Wiederaufnahme je Seed ueber eine Zwischenstand-Datei (CLAUDE.md: Pflicht;
    Rohdaten sofort sichern, Protokollregel 5)                               -> lauf_je_seed
  * Ausreisser-/Instabilitaetserkennung VOR der Urteilsbildung (I-035-Lehre:
    nicht erst deskriptiv danach)                                            -> pruefe_stabilitaet
  * BLAS-/OpenMP-Threads = 1 in Parallelprozessen (Fehlerbuch: Ueberbuchung) -> blas_threads_eins

Typischer Ablauf eines neuen Moduls:
    blas_threads_eins()
    info, roh = raten_je_arm(arme, trainiere_eins, tuning_seeds=(900, 901, 902, 903))
    ergebnis = lauf_je_seed(messseeds, lambda s: {arm: trainiere_eins(arm, s, info[arm]["lr"]) for arm in arme},
                            zwischen_pfad=ERGEBNISSE / "_zwischen_MEINE-ID.json")
    stab = pruefe_stabilitaet({arm: [ergebnis[s][arm] for s in sorted(ergebnis)] for arm in arme})
    urteil = statistik.urteil_h1_h0_hu(..., stabil=stab["ok"])   # erst NACH der Pruefung urteilen
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from labor import gemeinsam

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ----------------------------------------------------------------- BLAS-Threads
def blas_threads_eins(warnen: bool = True) -> bool:
    """Setzt BLAS/OpenMP-Threads auf 1 (Fehlerbuch: jede NumPy-SVD in jedem Parallelprozess startet sonst
    so viele Threads wie Kerne). Wirkt auf KINDPROZESSE zuverlaessig (Umgebung wird beim Start geerbt) und
    auf torch in diesem Prozess; fuer NumPy/BLAS im eigenen Prozess muss die Variable VOR 'import numpy'
    gesetzt sein: starte Laeufe daher mit `OMP_NUM_THREADS=1 python3 ...` (CLAUDE.md). Rueckgabe: True,
    wenn alle Variablen schon vor dem Aufruf auf 1 standen."""
    schon = all(os.environ.get(v) == "1" for v in _BLAS_VARS)
    if warnen and not schon and "numpy" in sys.modules:
        warnings.warn("BLAS-Threads standen nicht auf 1 (OMP_NUM_THREADS=1 vor dem Start setzen); "
                      "Kindprozesse und torch werden jetzt auf 1 gesetzt, NumPy-BLAS im Hauptprozess evtl. nicht.",
                      RuntimeWarning, stacklevel=2)
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    torch.set_num_threads(1)
    return schon


# ----------------------------------------------------------------- Training
def trainiere_adam(modell, x, y, schritte: int = 1500, lr: float = 0.02, params=None, seed: int = 0,
                   verlust: Callable = torch.nn.functional.cross_entropy, fehler_bei_nan: bool = False) -> dict:
    """Vollbatch-Adam (klein genug, dass Vollbatch schneller als Minibatch ist). Seed wird VOR dem Training
    gesetzt (Initialisierung des Aufrufers muss vorher oder mit demselben Seed geschehen). Rueckgabe
    {'verlust', 'endlich', 'verlauf'} -- 'verlauf' sind ~20 Verlustwerte zur Diagnose; 'endlich' ist False,
    sobald ein Verlust NaN/inf wurde (dann wird abgebrochen, das Modell gilt als instabil)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None else modell.parameters(), lr=lr)
    takt = max(1, schritte // 20)
    verlauf, v = [], float("nan")
    for i in range(schritte):
        opt.zero_grad()
        l = verlust(modell(x), y)
        l.backward()
        opt.step()
        v = float(l.item())
        if not math.isfinite(v):
            if fehler_bei_nan:
                raise FloatingPointError(f"Verlust nicht endlich bei Schritt {i}")
            return {"verlust": v, "endlich": False, "verlauf": verlauf}
        if i % takt == 0:
            verlauf.append(v)
    return {"verlust": v, "endlich": True, "verlauf": verlauf}


# ----------------------------------------------------------------- Ratenwahl je Arm
def _arbeiter(aufgabe):
    """Pickelbarer Arbeiter fuer gemeinsam.parallel: aufgabe = (arm, seed, lr, fn); fn(arm, seed, lr) -> acc."""
    arm, seed, lr, fn = aufgabe
    torch.set_num_threads(1)
    return {"acc": float(fn(arm, seed, lr))}


def raten_je_arm(arme: Iterable[str], trainiere_eins: Callable, tuning_seeds=(900, 901, 902, 903),
                 gitter=(0.02, 0.04, 0.08, 0.16, 0.32), prozesse: Optional[int] = None, **kw):
    """Eigene Lernrate JE ARM auf eigenen Tuning-Seeds mit Stabilitaetsabstand -- ruft die kanonische
    gemeinsam.gittersuche_stabil. trainiere_eins(arm, seed, lr) -> Genauigkeit (Validierung/Tuning-Daten,
    NIE die Test-/Mess-Seeds; Rauchtests nur auf Tuning-Seeds). trainiere_eins muss auf Modulebene definiert
    sein, wenn prozesse > 1. Die Tuning-Seeds duerfen sich nicht mit den Mess-Seeds ueberschneiden. Mindestens 3
    Tuning-Seeds: die einfache 2-Seed-Ratensuche kippt zuverlaessig (FEHLERBUCH, dreimal in einer Nacht).
    Rueckgabe wie gittersuche_stabil: (info[arm], roh[(arm, lr)][seed])."""
    arme = list(arme)
    if len(tuning_seeds) < 3:
        raise ValueError("mindestens 3 Tuning-Seeds (2-Seed-Ratensuche kippt, Fehlerbuch)")

    def aufgaben_fuer(arm, lr):
        return [(arm, s, lr, trainiere_eins) for s in tuning_seeds]

    def auswerten(aufgaben):
        return gemeinsam.parallel(_arbeiter, aufgaben, prozesse=prozesse)

    return gemeinsam.gittersuche_stabil(arme, aufgaben_fuer, auswerten, gitter=gitter, **kw)


# ----------------------------------------------------------------- Wiederaufnahme je Seed
def _atomar_schreiben(pfad: Path, inhalt: dict):
    tmp = pfad.with_suffix(pfad.suffix + ".tmp")
    tmp.write_text(json.dumps(inhalt, indent=1, ensure_ascii=False))
    os.replace(tmp, pfad)


def lauf_je_seed(seeds: Iterable[int], fn: Callable[[int], dict], zwischen_pfad, melde: Callable = print) -> dict:
    """Fuehrt fn(seed) -> dict (JSON-faehig) je Seed aus, sichert nach JEDEM Seed atomar in `zwischen_pfad`
    und ueberspringt beim Neustart bereits fertige Seeds (Wiederaufnahme je Seed; Protokollregel 5, CLAUDE.md).
    Rueckgabe {seed(int): ergebnis}. Zwischenstand-Datei danach committen (lange Laeufe)."""
    pfad = Path(zwischen_pfad)
    fertig = {}
    if pfad.exists():
        fertig = {int(k): v for k, v in json.loads(pfad.read_text()).items()}
    seeds = list(seeds)
    for s in seeds:
        if s in fertig:
            melde(f"Seed {s}: aus Zwischenstand uebernommen")
            continue
        fertig[s] = fn(s)
        pfad.parent.mkdir(parents=True, exist_ok=True)
        _atomar_schreiben(pfad, {str(k): v for k, v in sorted(fertig.items())})
        melde(f"Seed {s}: fertig und gesichert")
    gewollt = set(seeds)
    return {s: fertig[s] for s in sorted(fertig) if s in gewollt}


# ----------------------------------------------------------------- Stabilitaet VOR dem Urteil
def pruefe_stabilitaet(je_arm: dict, kollaps: float = 0.05, mad_faktor: float = 6.0, zufall: Optional[float] = None,
                       min_streuung: float = 1e-4) -> dict:
    """Ausreisser-/Instabilitaetspruefung je Arm, VOR der Urteilsbildung aufrufen (I-035: nicht erst danach).
    je_arm {arm: [Wert je Seed]} (gleiche Seed-Reihenfolge; Genauigkeit, hoeher = besser).
    Gemeldet werden Seed-Indizes mit
      * nicht endlichem Wert,
      * Kollaps: Wert < Median des Arms - `kollaps` (absolut, Genauigkeitsskala), oder <= `zufall`, falls gegeben,
      * robustem Ausreisser: |Wert - Median| > mad_faktor * max(1,4826*MAD, min_streuung).
    Ein Arm ist instabil, wenn mindestens ein Kollaps vorliegt; er gilt als 'mit Ausreisser', wenn robuste
    Ausreisser (auch nach oben) existieren. Rueckgabe {'ok': bool (kein instabiler Arm), 'arme': {...}, 'text'}.
    'ok' False -> Urteil HU (statistik.urteil_h1_h0_hu(stabil=False)), nie ein stilles H0."""
    aus, ok = {}, True
    for arm, w in je_arm.items():
        a = np.asarray(w, dtype=float)
        nicht_endlich = [int(i) for i in np.flatnonzero(~np.isfinite(a))]
        g = np.where(np.isfinite(a), a, np.nan)
        med = float(np.nanmedian(g)) if np.isfinite(g).any() else float("nan")
        mad = float(np.nanmedian(np.abs(g - med))) if np.isfinite(g).any() else float("nan")
        skala = max(1.4826 * mad, min_streuung) if np.isfinite(mad) else min_streuung
        koll = [int(i) for i in range(len(a)) if np.isfinite(a[i]) and
                (a[i] < med - kollaps or (zufall is not None and a[i] <= zufall))]
        ausr = [int(i) for i in range(len(a)) if np.isfinite(a[i]) and abs(a[i] - med) > mad_faktor * skala]
        instabil = bool(koll or nicht_endlich)
        ok &= not instabil
        aus[arm] = {"median": med, "mad": mad, "nicht_endlich": nicht_endlich, "kollaps": koll,
                    "ausreisser": ausr, "instabil": instabil}
    teile = [f"{arm}: " + ", ".join(f"{k} {v}" for k, v in (("Kollaps", e['kollaps']), ("nicht endlich", e['nicht_endlich']),
             ("Ausreisser", e['ausreisser'])) if v) for arm, e in aus.items() if e["kollaps"] or e["nicht_endlich"] or e["ausreisser"]]
    return {"ok": ok, "arme": aus, "text": "; ".join(teile) if teile else "stabil"}
