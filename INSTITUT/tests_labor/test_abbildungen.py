"""Tests fuer labor/abbildungen.py (Ausgabe nur in tmp_path)."""
import numpy as np

from labor import abbildungen as A


def test_komma():
    assert A.komma(93.34) == "93,3" and A.komma(-2.25, 2) == "−2,25"


def test_befund_balken_und_kurven_export(tmp_path):
    fig = A.befund_balken(["a", "b"], [1.0, -0.5], [(0.5, 1.5), (-1.0, 0.2)], urteile=["H1", "HU (x)"], referenz=0.0)
    out = A.speichere(fig, "t1", tmp_path)
    assert out["pdf"].stat().st_size > 1000 and out["png"].stat().st_size > 1000
    assert out["png"].stat().st_size < 200_000                       # "klein"
    w = np.random.default_rng(0).normal(0.9, 0.01, (6, 4))
    out = A.speichere(A.kurven_ueber_seeds([1, 2, 3, 4], w), "t2", tmp_path)
    assert out["pdf"].exists()


def test_beispiele_aus_ergebnis_jsons(tmp_path):
    erg = A.beispiele(tmp_path)
    assert len(erg) == 3 and all(e["pdf"].exists() and e["png"].exists() for e in erg)
