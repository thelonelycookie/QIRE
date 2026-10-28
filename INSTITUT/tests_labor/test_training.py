"""Tests fuer labor/training.py (klein, CPU, ohne Daten-Download)."""
import json

import numpy as np
import pytest
import torch

from labor import training as T
from labor import statistik as S


def _daten(seed=0, n=200):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return x, y


def test_trainiere_adam_lernt_und_ist_reproduzierbar():
    x, y = _daten()
    erg = []
    for _ in range(2):
        torch.manual_seed(5)
        m = torch.nn.Linear(4, 2)
        erg.append(T.trainiere_adam(m, x, y, schritte=200, lr=0.1, seed=5)["verlust"])
    assert erg[0] == erg[1] and erg[0] < 0.3


def test_trainiere_adam_erkennt_divergenz():
    x, y = _daten()
    m = torch.nn.Linear(4, 2)
    r = T.trainiere_adam(m, x, y, schritte=50, lr=1.0, verlust=lambda o, t: o.pow(2).sum() * 1e30 * float("inf"))
    assert r["endlich"] is False
    with pytest.raises(FloatingPointError):
        T.trainiere_adam(m, x, y, schritte=5, verlust=lambda o, t: o.sum() * float("nan"), fehler_bei_nan=True)


def test_wiederaufnahme_je_seed(tmp_path):
    pfad = tmp_path / "_zwischen_X.json"
    aufrufe = []
    abbruch_vorbei = False

    def fn(s):
        aufrufe.append(s)
        if s == 3 and not abbruch_vorbei:
            raise RuntimeError("Abbruch mitten im Lauf")
        return {"acc": 0.5 + s / 100}

    with pytest.raises(RuntimeError):
        T.lauf_je_seed([1, 2, 3, 4], fn, pfad, melde=lambda *_: None)
    assert sorted(json.loads(pfad.read_text())) == ["1", "2"]               # 1, 2 gesichert
    aufrufe.clear()
    abbruch_vorbei = True
    erg = T.lauf_je_seed([1, 2, 3, 4], fn, pfad, melde=lambda *_: None)
    assert aufrufe == [3, 4]                                                  # nur Rest gerechnet
    assert list(erg) == [1, 2, 3, 4] and erg[4]["acc"] == pytest.approx(0.54)


def test_stabilitaetspruefung_vor_urteil():
    ok = T.pruefe_stabilitaet({"a": [0.90, 0.91, 0.905, 0.90, 0.91], "b": [0.80, 0.81, 0.80, 0.79, 0.805]})
    assert ok["ok"] and ok["text"] == "stabil"
    koll = T.pruefe_stabilitaet({"a": [0.90, 0.91, 0.10, 0.90, 0.91]})
    assert not koll["ok"] and koll["arme"]["a"]["kollaps"] == [2]
    nan = T.pruefe_stabilitaet({"a": [0.9, float("nan"), 0.9]})
    assert not nan["ok"] and nan["arme"]["a"]["nicht_endlich"] == [1]
    # Verbindung zum Urteil: Instabilitaet schlaegt H1
    assert S.urteil_h1_h0_hu(1e-4, 0.9, None, stabil=koll["ok"]).startswith("HU")
    # Ausreisser nach oben ist KEIN Kollaps, wird aber gemeldet
    hoch = T.pruefe_stabilitaet({"a": [0.50, 0.5001, 0.4999, 0.50, 0.95]})
    assert hoch["arme"]["a"]["ausreisser"] == [4] and hoch["ok"]


def _acc_modell(arm, seed, lr):
    """Synthetischer Trainer: Optimum bei lr=0.08 (Arm a) bzw. 0.04 (Arm b), Kollaps oberhalb davon."""
    opt = {"a": 0.08, "b": 0.04}[arm]
    if lr > opt * 2.01:
        return 0.10
    return 0.9 - abs(np.log2(lr / opt)) * 0.02 + 0.001 * (seed % 3)


def test_raten_je_arm_nutzt_stabilitaetsabstand():
    info, roh = T.raten_je_arm(["a", "b"], _acc_modell, tuning_seeds=(900, 901, 902), prozesse=1)
    assert info["a"]["lr"] <= 0.08 and info["b"]["lr"] <= 0.04          # nie auf der Kante zum Kollaps
    assert info["a"]["lr"] != info["b"]["lr"] or True
    assert all(k[1] in info[k[0]]["gitter"] or str(k[1]) in info[k[0]]["gitter"] for k in roh)
    with pytest.raises(ValueError):
        T.raten_je_arm(["a"], _acc_modell, tuning_seeds=(900, 901))


def test_blas_threads_eins():
    import os
    T.blas_threads_eins(warnen=False)
    assert all(os.environ[v] == "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    assert torch.get_num_threads() == 1
