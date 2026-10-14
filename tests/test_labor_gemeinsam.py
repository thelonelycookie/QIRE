"""Tests fuer INSTITUT/labor/gemeinsam.py (klein, CPU, ohne Netz/Datensatz-Download)."""
import math

import numpy as np
import pytest
import torch

from labor import gemeinsam as G


def test_ziffern_split_ist_reproduzierbar_und_disjunkt():
    x1, y1, xt1, yt1 = G.ziffern(3)
    x2, y2, xt2, yt2 = G.ziffern(3)
    assert x1.shape[1] == 64 and len(y1) + len(yt1) == 1797
    assert torch.equal(x1, x2) and torch.equal(yt1, yt2)
    assert not torch.equal(G.ziffern(4)[0], x1)           # anderer Seed -> anderer Split
    assert 0.0 <= float(x1.min()) and float(x1.max()) <= 1.0


@pytest.mark.parametrize("n", [1, 3, 5, 8])
def test_wilcoxon_exakt_alle_positiv_ist_2_hoch_minus_n_plus_1(n):
    assert G.wilcoxon_exakt([0.1 * (i + 1) for i in range(n)]) == pytest.approx(2.0 / 2 ** n)


def test_wilcoxon_exakt_symmetrisch_und_nullen():
    assert G.wilcoxon_exakt([0.0, 0.0]) == 1.0
    assert 0.0 < G.wilcoxon_exakt([1, -2, 3, -4]) <= 1.0
    assert G.wilcoxon_exakt([1, 2, 3, 4, 5, 6]) == G.wilcoxon_exakt([-1, -2, -3, -4, -5, -6])


def test_wilcoxon_exakt_gegen_scipy_exakt():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = [0.5, -0.2, 0.9, 1.3, -0.1, 0.7, 0.35]
    erwartet = scipy_stats.wilcoxon(d, method="exact").pvalue
    assert G.wilcoxon_exakt(d) == pytest.approx(erwartet, abs=1e-9)


def test_zusammenfassung():
    z = G.zusammenfassung([1.0, 2.0, 3.0])
    assert z["mittel"] == pytest.approx(2.0) and z["std"] == pytest.approx(1.0) and z["n"] == 3
    assert G.zusammenfassung([5.0])["std"] == 0.0


def test_genauigkeit():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    assert G.genauigkeit(logits, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)


def test_trainiere_senkt_verlust_und_ist_reproduzierbar():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(120, 4, generator=g)
    y = (x[:, 0] > 0).long()

    def neu():
        torch.manual_seed(1)
        return torch.nn.Linear(4, 2)

    m = neu()
    v0 = torch.nn.functional.cross_entropy(m(x), y).item()
    v1 = G.trainiere(m, x, y, schritte=60, lr=0.1, seed=1)
    assert v1 < v0
    m2 = neu()
    assert G.trainiere(m2, x, y, schritte=60, lr=0.1, seed=1) == pytest.approx(v1)


def test_speichere_schreibt_json_ohne_zu_ueberschreiben(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "ERGEBNISSE", tmp_path)
    p1 = G.speichere("T-test", {"a": 1})
    p2 = G.speichere("T-test", {"a": 2})
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p1.name.startswith("T-test__")


# ---- Ratenwahl mit Stabilitaetsabstand (Lehre aus C-referenz, I-003, I-035)
def test_waehle_rate_stuft_ab_wenn_oberer_nachbar_kollabiert():
    werte = {0.02: [0.90, 0.91], 0.04: [0.93, 0.94], 0.08: [0.95, 0.95], 0.16: [0.50, 0.94], 0.32: [0.4, 0.4]}
    w = G.waehle_rate(werte)
    # Bestes Kriterium (schlechtester Seed) liegt bei 0.08; der obere Nachbar 0.16 kollabiert auf Seed 0 -> eine Stufe tiefer
    assert w["lr"] == 0.04 and w["abgestuft"] == 1


def test_waehle_rate_verlangt_gleich_viele_seeds():
    with pytest.raises(ValueError):
        G.waehle_rate({0.02: [0.9, 0.9], 0.04: [0.9]})


def test_waehle_rate_erweitert_am_rand():
    w = G.waehle_rate({0.02: [0.9, 0.9], 0.04: [0.8, 0.8], 0.08: [0.7, 0.7]})
    assert w["erweitern"] == "unten" and w["lr"] is None
    assert not math.isnan(w["kriterium"]["0.02"])


def test_parallel_einprozess_gleich_listenverarbeitung():
    assert G.parallel(abs, [-1, 2, -3], prozesse=1) == [1, 2, 3]
