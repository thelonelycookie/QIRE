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
    with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
        return pool.map(fn, aufgaben)


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
