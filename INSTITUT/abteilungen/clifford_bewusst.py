"""Abteilung Clifford -- Vertiefung: Clifford-bewusstes Training (C-bewusst).

Befund aus C-clifford v1: Die trainierten Winkel liegen im Mittel 0,390 rad
vom Clifford-Gitter entfernt -- genau so weit wie gleichverteilte Winkel
(pi/8). Das Training „weiss“ nichts vom Gitter. Idee: waehrend des Trainings
einen Strafterm lambda(t) * mean(sin^2(2*theta)) hochfahren (null genau auf
k*pi/2), damit die Schaltung von selbst multiplikationsfrei wird.
Verwandt mit quantisierungsbewusstem Training (QAT); neu ist nur das Ziel:
das Clifford-Gitter der Quanteninformatik statt eines Integer-Gitters.

Vorregistriert (vor jedem Lauf dieses Moduls):
  B-1: bewusst_k0 > normal_k0 (gerundet, frischer Kopf), exakter Wilcoxon p<0,05.
  B-2: bewusst_k0 ist nicht unterlegen gegen voll (einseitiges 95-%-KI > -2 pp).
  B-3: beschreibend: Abstand zum Gitter nach Training, Kurve ueber k.
Lernraten: voll wie in clifford.py gesucht; lambda auf Tuning-Seeds gesucht
(Gitter in beide Richtungen); Kopf-Rate nach Rundung je Variante eigen.
Abweichung von Regel 1a, offen benannt: das bewusste Training benutzt die
Rate von voll (gemeinsame Suche lr x lambda waere 9x teurer).
Mess-Seeds 20-29 (frisch), Tuning-Seeds 904/905.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen.clifford import (QUBITS_JE_DATENSATZ, _baue, _gerundet, _lr_waehlen, daten,
                                  runde_clifford)
from labor.gemeinsam import genauigkeit, parallel, trainiere, wilcoxon_exakt, zusammenfassung

TOLERANZ = 0.02
STUFEN = [0, 2, 4]
HYPOTHESEN = {
    "B-1": "Clifford-bewusstes Training + Rundung schlaegt normales Training + Rundung (k=0).",
    "B-2": "Clifford-bewusstes Training + Rundung ist dem vollen Modell nicht unterlegen (-2 pp).",
    "B-3": "beschreibend: Gitterabstand und Genauigkeit ueber k.",
}


def trainiere_bewusst(m, x, y, schritte, lr, lam, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for t in range(schritte):
        opt.zero_grad()
        strafe = torch.sin(2 * m.theta).pow(2).mean()
        loss = torch.nn.functional.cross_entropy(m(x), y) + lam * (t / schritte) * strafe
        loss.backward()
        opt.step()


def _punkt(a):
    """Tuning: (variante, lam, lr_kopf, s, schritte, ds, lr_voll) -> Validierungsgenauigkeit nach Rundung k=0."""
    variante, lam, lr_kopf, s, schritte, ds, lr_voll = a
    xtr, ytr, _, _ = daten(ds, s)
    nv = len(ytr) // 4
    m, _ = _baue("voll", s, QUBITS_JE_DATENSATZ[ds])
    if variante == "bewusst":
        trainiere_bewusst(m, xtr[nv:], ytr[nv:], schritte, lr_voll, lam, s)
    else:
        trainiere(m, xtr[nv:], ytr[nv:], schritte, lr_voll, seed=s)
    r = _gerundet(m, 0, s)
    trainiere(r, xtr[nv:], ytr[nv:], schritte, lr_kopf, seed=s)
    return genauigkeit(r(xtr[:nv]), ytr[:nv])


def _suche(feste, variable, gitter, tuning_seeds):
    """Eindimensionale Gittersuche mit Verlaengerung in beide Richtungen."""
    gitter, wert = sorted(gitter), {}
    for _ in range(4):
        neu = [g for g in gitter if g not in wert]
        erg = parallel(_punkt, [feste(g, s) for g in neu for s in tuning_seeds])
        for i, g in enumerate(neu):
            wert[g] = float(np.mean(erg[i * len(tuning_seeds):(i + 1) * len(tuning_seeds)]))
        best = max(gitter, key=wert.get)
        if best == gitter[0]:
            gitter = [gitter[0] / 10 if variable == "lam" else gitter[0] / 4] + gitter
        elif best == gitter[-1]:
            gitter = gitter + [gitter[-1] * 10 if variable == "lam" else gitter[-1] * 4]
        else:
            return best, wert, False
    return max(wert, key=wert.get), wert, True


def _seed(a):
    s, schritte, lr, ds = a
    xtr, ytr, xte, yte = daten(ds, s)
    n = QUBITS_JE_DATENSATZ[ds]
    z = {"seed": s}
    normal, _ = _baue("voll", s, n)
    trainiere(normal, xtr, ytr, schritte, lr["voll"], seed=s)
    z["voll"] = genauigkeit(normal(xte), yte)
    bew, _ = _baue("voll", s, n)
    trainiere_bewusst(bew, xtr, ytr, schritte, lr["voll"], lr["lam"], s)
    z["bewusst_voll"] = genauigkeit(bew(xte), yte)
    for name, m, lrk in (("normal", normal, lr["kopf_normal"]), ("bewusst", bew, lr["kopf_bewusst"])):
        th = m.theta.detach()
        z[f"{name}_abstand"] = float((th - runde_clifford(th)).abs().mean())
        for k in STUFEN:
            r = _gerundet(m, k, s)
            trainiere(r, xtr, ytr, schritte, lrk, seed=s)
            z[f"{name}_k{k}"] = genauigkeit(r(xte), yte)
    return z


def lauf(seeds=range(20, 30), schritte=1500, tuning_seeds=(904, 905), datensatz="ziffern", schnell=False):
    if schnell:
        seeds, schritte, tuning_seeds = range(2), 150, (904,)
    t0 = time.time()
    lr = {"voll": _lr_waehlen(["voll"], tuning_seeds, schritte, datensatz)["voll"]["lr"]}
    info = {}
    lr["kopf_normal"], info["kopf_normal"], r1 = _suche(
        lambda g, s: ("normal", 0.0, g, s, schritte, datensatz, lr["voll"]), "lr", [0.02, 0.08, 0.32], tuning_seeds)
    lr["lam"], info["lam"], r2 = _suche(
        lambda g, s: ("bewusst", g, lr["kopf_normal"], s, schritte, datensatz, lr["voll"]), "lam", [0.03, 0.3, 3.0], tuning_seeds)
    lr["kopf_bewusst"], info["kopf_bewusst"], r3 = _suche(
        lambda g, s: ("bewusst", lr["lam"], g, s, schritte, datensatz, lr["voll"]), "lr", [0.02, 0.08, 0.32], tuning_seeds)
    info["am_rand"] = [r1, r2, r3]
    zeilen = parallel(_seed, [(s, schritte, lr, datensatz) for s in seeds])
    aus = auswertung(zeilen)
    return {"abteilung": "clifford_bewusst", "datensatz": datensatz, "hypothesen": HYPOTHESEN, "lr": lr,
            "suche": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in info.items()},
            "zeilen": zeilen, "auswertung": aus, "urteil": urteil(aus), "sekunden": time.time() - t0}


def auswertung(zeilen):
    from scipy.stats import t as t_vert
    a = lambda k: np.array([z[k] for z in zeilen])
    aus = {k: zusammenfassung(a(k)) for k in zeilen[0] if k != "seed"}
    d1 = a("bewusst_k0") - a("normal_k0")
    aus["B1_diff"], aus["B1_p"] = float(d1.mean()), wilcoxon_exakt(d1)
    d2 = a("bewusst_k0") - a("voll")
    se, q = d2.std(ddof=1) / math.sqrt(len(d2)), t_vert.ppf(0.95, len(d2) - 1)
    aus["B2_diff"], aus["B2_ki_unten"], aus["B2_ki_oben"] = float(d2.mean()), float(d2.mean() - q * se), float(d2.mean() + q * se)
    return aus


def urteil(a):
    u = {"B-1": ("H1: bewusstes Training rundet besser" if a["B1_diff"] > 0 and a["B1_p"] < 0.05
                 else "H0: kein nachweisbarer Vorteil des bewussten Trainings")}
    if a["B2_ki_unten"] > -TOLERANZ:
        u["B-2"] = "H1: multiplikationsfreie Schaltung nicht unterlegen (-2 pp)"
    elif a["B2_ki_oben"] < -TOLERANZ:
        u["B-2"] = "H0: nachweislich mehr als 2 pp schlechter"
    else:
        u["B-2"] = "HU: KI ueberdeckt die Grenze"
    return u


def vortest(ziehungen=5):
    rng = np.random.default_rng(3)
    ok = True
    grenz_fehl = 0
    # Grenzfall HU braucht Streuung, sonst entscheidet der Zufall der Ziehung (Generatorfehler, 24.09.)
    for voll, nk0, bk0, erw, sd in [(0.96, 0.93, 0.955, ("H1", "H1"), .004), (0.96, 0.93, 0.93, ("H0", "H0"), .004),
                                    (0.96, 0.90, 0.94, ("H1", "HU"), .012)]:
        for _ in range(ziehungen):
            zeilen = []
            for s in range(10):
                z = {"seed": s, "voll": voll + rng.normal(0, sd), "bewusst_voll": voll + rng.normal(0, .004)}
                for v, mu in (("normal", nk0), ("bewusst", bk0)):
                    z[f"{v}_abstand"] = 0.3
                    for k in STUFEN:
                        z[f"{v}_k{k}"] = mu + rng.normal(0, sd)
                zeilen.append(z)
            u = urteil(auswertung(zeilen))
            if not (u["B-1"].startswith(erw[0]) and u["B-2"].startswith(erw[1])):
                if (voll, nk0, bk0) == (0.96, 0.93, 0.93) and u["B-2"].startswith("H0"):
                    continue          # B-1 im Nullfall: Fehlalarm mit Rate alpha erlaubt
                if erw[1] == "HU" and u["B-1"].startswith("H1"):
                    grenz_fehl += 1   # genau auf der Grenze kippt das KI mit Rate ~alpha -- einmal erlaubt
                    continue
                ok = False
                print("  VORTEST FEHLER", voll, nk0, bk0, u)
    if grenz_fehl > 1:
        ok = False
        print("  VORTEST FEHLER: Grenzfall kippt zu oft", grenz_fehl)
    return ok
