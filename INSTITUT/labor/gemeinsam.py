"""Gemeinsames Labor aller Abteilungen: Daten, Statistik, Ablage.

Bewusst klein gehalten. Laeuft auf CPU in Minuten (sklearn-Ziffern, 8x8 = 64
Pixel = 2^6 Amplituden = 6 Qubits). Kein Download noetig.
"""
from __future__ import annotations

import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

WURZEL = Path(__file__).resolve().parent.parent
ERGEBNISSE = WURZEL / "ergebnisse"


def ziffern(seed: int, anteil_test: float = 0.3):
    """sklearn-Ziffern (1.797 Bilder, 10 Klassen), fester Split je Seed."""
    from sklearn.datasets import load_digits
    d = load_digits()
    x = torch.tensor(d.data, dtype=torch.float32) / 16.0
    y = torch.tensor(d.target, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    n_test = int(len(y) * anteil_test)
    te, tr = perm[:n_test], perm[n_test:]
    return x[tr], y[tr], x[te], y[te]


def genauigkeit(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def trainiere(modell, x, y, schritte=1500, lr=0.02, params=None, seed=0):
    """Vollbatch-Adam. Klein genug, dass Vollbatch schneller ist als Minibatch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None else modell.parameters(), lr=lr)
    for _ in range(schritte):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(modell(x), y)
        loss.backward()
        opt.step()
    return loss.item()


# ---------------------------------------------------------------- Statistik
def wilcoxon_exakt(diff) -> float:
    """Exakter zweiseitiger Wilcoxon-Vorzeichen-Rang-Test (kleine n, keine
    Asymptotik -- Lehre aus Q-011: scipys asymptotischer Wert war falsch)."""
    d = np.asarray([v for v in diff if v != 0.0])
    n = len(d)
    if n == 0:
        return 1.0
    raenge = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = raenge[d > 0].sum()
    alle = [sum(r for r, s in zip(raenge, v) if s) for v in product([0, 1], repeat=n)]
    alle = np.array(alle)
    mitte = raenge.sum() / 2
    return float(np.mean(np.abs(alle - mitte) >= abs(w - mitte) - 1e-9))


def zusammenfassung(werte) -> dict:
    a = np.asarray(werte, dtype=float)
    return {"mittel": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


# ---------------------------------------------------------------- Ablage
def speichere(hyp_id: str, inhalt: dict) -> Path:
    """Rohdaten sofort sichern (Protokollregel 5)."""
    ERGEBNISSE.mkdir(exist_ok=True)
    stempel = time.strftime("%Y%m%d_%H%M%S")
    pfad = ERGEBNISSE / f"{hyp_id}__{stempel}.json"
    i = 2
    while pfad.exists():
        pfad = ERGEBNISSE / f"{hyp_id}__{stempel}_{i:02d}.json"
        i += 1
    pfad.write_text(json.dumps(inhalt, indent=1, ensure_ascii=False))
    return pfad


def neuestes(hyp_id: str):
    dateien = sorted(ERGEBNISSE.glob(f"{hyp_id}__*.json"))
    return json.loads(dateien[-1].read_text()) if dateien else None


def _init_arbeiter():
    torch.set_num_threads(1)


def parallel(fn, aufgaben, prozesse=None):
    """Unabhaengige Trainings auf mehrere Kerne verteilen (je 1 Thread).
    Auf Maltes Ryzen 5 3600: prozesse=5 laesst einen Kern frei."""
    import multiprocessing as mp
    import os
    prozesse = prozesse or int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))
    if prozesse <= 1:
        _init_arbeiter()
        return [fn(a) for a in aufgaben]
    # Kindprozesse erben die Umgebung beim Start: BLAS/OpenMP auf 1 Thread, sonst
    # startet jede NumPy-SVD so viele Threads wie Kerne (Ueberbuchung, 24.09.).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
        return pool.map(fn, aufgaben)


# ---------------------------------------------------------------- Ratenwahl mit Stabilitaetsabstand
# Direktor 25.09., nach der DRITTEN Instabilitaet einer Nacht (C-referenz, I-003-
# Struktur, I-035): eine Rate, die nur nach dem Mittel ueber 2 Tuning-Seeds gewaehlt
# wird, kippt oft direkt an der Stabilitaetskante (der naechste Gitterpunkt kollabiert
# schon). Diese Logik stand zuerst in abteilungen/skala_v2.py (dort erstmals gebaut,
# als Antwort auf genau dieses Problem in C-skala) -- hierher gehoben, damit sie
# KANONISCH wird: neue Module sollen sie standardmaessig verwenden, nicht die
# "einfache" 2-Seed-argmax-Suche neu erfinden (die hat jetzt dreimal denselben Fehler
# reproduziert). Wer aus Zeitgruenden doch die einfache Suche nimmt, sollte das im
# Docstring begruenden UND zusaetzlich eine Ausreisser-Pruefung vor der Urteilsbildung
# einbauen (nicht nur deskriptiv nach dem Urteil, wie es I-035 zuerst tat).
def waehle_rate(werte: dict, kollaps: float = 0.05, notfall: bool = False) -> dict:
    """werte {lr: [acc je Tuning-Seed]} (gleiche Seed-Reihenfolge je lr). Kriterium:
    schlechtester Tuning-Seed (min), nicht das Mittel -- eine Rate ist nur waehlbar,
    wenn ihr oberer Gitternachbar auf KEINEM Tuning-Seed kollabiert (Genauigkeit
    < beste Rate desselben Seeds - `kollaps`), sonst eine Stufe tiefer, wiederholt.
    Rueckgabe {'lr', 'erweitern' (None | 'unten' | 'oben'), 'argmax' (min-Kriterium),
    'argmax_mittel', 'abgestuft' (Stufen unter argmax), 'kollabiert' {lr: [Seed-Index]},
    'am_rand'}. Mit notfall=True (Gitter erschoepft, keine weitere Runde) wird nie
    erweitert: ein Maximum am oberen Rand gilt als 'oberer Nachbar ungeprueft' ->
    eine Stufe tiefer."""
    g = sorted(werte)
    n_s = {len(v) for v in werte.values()}
    if len(n_s) != 1:
        raise ValueError("ungleich viele Tuning-Seeds je Rate")
    n_s = n_s.pop()
    krit = {lr: float(min(werte[lr])) for lr in g}
    mittel = {lr: float(np.mean(werte[lr])) for lr in g}
    beste = [max(werte[lr][s] for lr in g) for s in range(n_s)]
    koll = {lr: [s for s in range(n_s) if werte[lr][s] < beste[s] - kollaps] for lr in g}
    i = max(range(len(g)), key=lambda j: (krit[g[j]], -j))       # Gleichstand -> kleinere Rate
    aus = {"argmax": g[i], "argmax_mittel": max(g, key=lambda lr: (mittel[lr], -lr)),
           "kriterium": {str(lr): krit[lr] for lr in g}, "mittel": {str(lr): mittel[lr] for lr in g},
           "kollabiert": {str(lr): koll[lr] for lr in g}, "erweitern": None, "lr": None,
           "abgestuft": 0, "am_rand": False}
    if not notfall:
        if i == 0:
            aus["erweitern"] = "unten"
            return aus
        if i == len(g) - 1:
            aus["erweitern"] = "oben"
            return aus
    j = i
    if j == len(g) - 1:          # nur im Notfall: oberer Nachbar nie gemessen
        j -= 1
        aus["am_rand"] = True
    while j >= 0 and koll[g[j + 1]]:
        j -= 1
    if j < 0:
        if not notfall:
            aus["erweitern"] = "unten"
            return aus
        j, aus["am_rand"] = 0, True
    if notfall and i == 0:
        aus["am_rand"] = True
    aus["lr"], aus["abgestuft"] = g[j], i - j
    return aus


def gittersuche_stabil(schluessel, aufgaben_fuer, auswerten, gitter=(0.02, 0.04, 0.08, 0.16, 0.32),
                       faktor: int = 2, kollaps: float = 0.05, runden: int = 6):
    """Eigene Lernrate je Schluessel MIT Stabilitaetsabstand (waehle_rate), Gitter in
    beide Richtungen automatisch verlaengert (Faktor `faktor`). `aufgaben_fuer(schl, lr)`
    -> Liste von Aufgaben, Seed an Stelle 1 jeder Aufgabe; `auswerten(aufgaben)` ->
    Liste von dicts mit Schluessel 'acc' (z. B. diese `parallel`-Funktion mit einem
    passenden Arbeiter). Alle offenen Schluessel einer Runde laufen gemeinsam parallel.
    Rueckgabe: (info, roh) -- info[schl] = {'lr', 'argmax', 'argmax_mittel', 'abgestuft',
    'am_rand', 'runden', 'gitter': {...}}, roh[(schl, lr)][seed] = Ergebnis."""
    gitter = {s: sorted(gitter) for s in schluessel}
    roh, info, offen = {}, {}, set(schluessel)

    def werte(s):
        return {lr: [roh[(s, lr)][sd]["acc"] for sd in sorted(roh[(s, lr)])] for lr in gitter[s]}

    def eintrag(s, w, runde):
        return {"lr": w["lr"], "argmax": w["argmax"], "argmax_mittel": w["argmax_mittel"],
                "abgestuft": w["abgestuft"], "am_rand": w["am_rand"], "runden": runde,
                "gitter": {str(lr): {"min": w["kriterium"][str(lr)], "mittel": w["mittel"][str(lr)],
                                     "je_seed": [roh[(s, lr)][sd]["acc"] for sd in sorted(roh[(s, lr)])],
                                     "kollabiert_seeds": [sorted(roh[(s, lr)])[i] for i in w["kollabiert"][str(lr)]]}
                           for lr in gitter[s]}}

    for runde in range(1, runden + 1):
        paare = [(s, lr) for s in sorted(offen) for lr in gitter[s] if (s, lr) not in roh]
        aufg, zu = [], []
        for s, lr in paare:
            for a in aufgaben_fuer(s, lr):
                aufg.append(a)
                zu.append((s, lr))
        for a, (s, lr), e in zip(aufg, zu, auswerten(aufg)):
            roh.setdefault((s, lr), {})[a[1]] = e
        for s in sorted(offen):
            w = waehle_rate(werte(s), kollaps=kollaps)
            if w["erweitern"] == "unten":
                gitter[s] = [gitter[s][0] / faktor] + gitter[s]
            elif w["erweitern"] == "oben":
                gitter[s] = gitter[s] + [gitter[s][-1] * faktor]
            else:
                info[s] = eintrag(s, w, runde)
                offen.discard(s)
        if not offen:
            break
    for s in sorted(offen):      # Notfall: Gitter erschoepft, nur mit gemessenen Punkten
        gitter[s] = [lr for lr in gitter[s] if (s, lr) in roh]
        info[s] = eintrag(s, waehle_rate(werte(s), kollaps=kollaps, notfall=True), runden)
    return info, roh


# ---------------------------------------------------------------- grosse Bilddaten
_QUELLEN = {
    "mnist": "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "fashion_mnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
_DATEIEN = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _roh(name):
    """1) Maltes Cache aus Projekt A (results/datasets/<name>.npz), 2) eigener
    Cache, 3) Download der IDX-Dateien (einmalig, ~30 MB)."""
    import gzip
    import os
    import urllib.request
    kandidaten = [Path(os.environ.get("QIRE_DATEN", "_")) / f"{name}.npz",
                  WURZEL.parent / "A" / "results" / "datasets" / f"{name}.npz",
                  WURZEL / "daten" / f"{name}.npz"]
    for k in kandidaten:
        if k.exists():
            b = np.load(k)
            return b["x"].astype(np.float32), b["y"].astype(np.int64)
    ordner = WURZEL / "daten"
    ordner.mkdir(exist_ok=True)
    teile = []
    for f in _DATEIEN:
        ziel = ordner / f"{name}_{f}"
        if not ziel.exists():
            urllib.request.urlretrieve(_QUELLEN[name] + f, ziel)
        with gzip.open(ziel) as g:
            daten = g.read()
        teile.append(np.frombuffer(daten, np.uint8, offset=16 if "images" in f else 8))
    x = np.concatenate([teile[0], teile[2]]).reshape(-1, 784).astype(np.float32) / 255.0
    y = np.concatenate([teile[1], teile[3]]).astype(np.int64)
    np.savez_compressed(ordner / f"{name}.npz", x=x, y=y)
    return x, y


def bilder(name, seed, n_train=10_000, n_test=5_000, seite=16):
    """MNIST/Fashion auf seite x seite verkleinert (16x16 = 256 = 2^8 -> 8 Qubits).
    Zufaellige Teilmenge je Seed; Test stets aus den offiziellen Testbildern."""
    x, y = _roh(name)
    x = torch.tensor(x).reshape(-1, 1, 28, 28)
    x = torch.nn.functional.interpolate(x, size=(seite, seite), mode="area").reshape(len(y), -1)
    y = torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    tr = torch.randperm(60_000, generator=g)[:n_train]
    te = 60_000 + torch.randperm(10_000, generator=g)[:n_test]
    return x[tr], y[tr], x[te], y[te]
