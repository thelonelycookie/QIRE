"""Abteilung Auslese -- das Schuss-Budget der Born-Auslese (Karte I-008, Q3).

Auf echter Quantenhardware gibt es die Born-Verteilung p = |U x|^2 nicht exakt,
sondern nur S Messungen („Schuesse“): p_dach = Multinomial(S, p) / S. Die Zahl
der Schuesse ist die eigentliche Laufzeit eines Quantenrechners. Frage: Wie
viele Schuesse braucht der Klassifikator -- und welche Auslese ist am
schusssparsamsten?

Drei Auslesen, alle aus DERSELBEN Messung in der Rechenbasis (keine Extra-Laeufe):
  born     : alle 2^n Wahrscheinlichkeiten
  paritaet : <Z_S> fuer |S| <= 2 (1 + n + n(n-1)/2 Werte) -- Walsh-Hadamard von p
  z        : <Z_q> (n Werte)
Die Koepfe werden auf exakten Merkmalen trainiert (wie im Simulator), dann mit
S Schuessen ausgewertet. Mess-Seeds 40-47 (Ziffern), 50-52 (MNIST, beschreibend).

Vorregistriert:
  A-1: Bei S = 1.000 liegt die born-Auslese mehr als 1 pp unter exakt (sonst ist
       die Schusszahl kein Engpass -> Linie ruht).  Einseitiges 95-%-KI.
  A-2: Bei S = 100 schlaegt paritaet die born-Auslese (exakter Wilcoxon p<0,05):
       wenige, gemittelte Merkmale sind robuster gegen Schussrauschen.
  A-3: beschreibend: Genauigkeit ueber S in {10, 100, 1.000, 10.000, exakt}.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from abteilungen.clifford import QUBITS_JE_DATENSATZ, QireZelle, daten
from labor.gemeinsam import genauigkeit, parallel, trainiere, wilcoxon_exakt, zusammenfassung

SCHUESSE = [10, 100, 1_000, 10_000]
LR, SCHRITTE = 0.08, 1500          # Rate von 'voll' aus C-clifford-v2/C-mnist (dort je Arm gesucht, nie am Rand)
HYPOTHESEN = {
    "A-1": "born-Auslese verliert bei S=1.000 mehr als 1 pp gegenueber exakt (KI).",
    "A-2": "paritaet schlaegt born bei S=100 (Wilcoxon).",
    "A-3": "beschreibend: Kurven ueber S.",
}


def paritaets_matrix(n):
    idx = np.arange(2 ** n)
    bit = lambda q: (idx >> (n - 1 - q)) & 1
    spalten = [np.ones(2 ** n)]
    spalten += [1 - 2 * bit(q) for q in range(n)]
    spalten += [(1 - 2 * bit(a)) * (1 - 2 * bit(b)) for a in range(n) for b in range(a + 1, n)]
    return torch.tensor(np.stack(spalten, 1), dtype=torch.float32)


def _seed(a):
    s, ds = a
    n = QUBITS_JE_DATENSATZ[ds]
    xtr, ytr, xte, yte = daten(ds, s)
    m = QireZelle(n=n)
    torch.manual_seed(s)
    trainiere(m, xtr, ytr, SCHRITTE, LR, seed=s)       # Schaltung + born-Kopf
    with torch.no_grad():
        ptr, pte = m.merkmale(xtr), m.merkmale(xte)
    Pm, Zm = paritaets_matrix(n), paritaets_matrix(n)[:, 1:n + 1]
    koepfe = {"born": (m.kopf, lambda p: p)}
    for name, M in (("paritaet", Pm), ("z", Zm)):
        k = nn.Linear(M.shape[1], 10)
        torch.manual_seed(s)
        trainiere(k, ptr @ M, ytr, SCHRITTE, LR, seed=s)
        koepfe[name] = (k, lambda p, M=M: p @ M)
    z = {"seed": s, "datensatz": ds}
    g = torch.Generator().manual_seed(7000 + s)
    for name, (kopf, merk) in koepfe.items():
        with torch.no_grad():
            z[f"{name}_exakt"] = genauigkeit(kopf(merk(pte)), yte)
            for S in SCHUESSE:
                zaehl = torch.stack([torch.bincount(torch.multinomial(p.clamp_min(0), S, replacement=True, generator=g),
                                                    minlength=2 ** n) for p in pte]).float() / S
                z[f"{name}_S{S}"] = genauigkeit(kopf(merk(zaehl)), yte)
    return z


def lauf(schnell=False):
    t0 = time.time()
    auf = [(s, "ziffern") for s in (range(40, 42) if schnell else range(40, 48))]
    if not schnell:
        auf += [(s, "mnist") for s in range(50, 53)]
    zeilen = parallel(_seed, auf)
    zif = [z for z in zeilen if z["datensatz"] == "ziffern"]
    aus = auswertung(zif)
    aus["mnist_beschreibend"] = {k: zusammenfassung([z[k] for z in zeilen if z["datensatz"] == "mnist"])["mittel"]
                                 for k in zeilen[0] if k not in ("seed", "datensatz")} if not schnell else None
    return {"abteilung": "auslese", "hypothesen": HYPOTHESEN, "zeilen": zeilen, "auswertung": aus,
            "urteil": urteil(aus), "sekunden": time.time() - t0}


def auswertung(zeilen):
    from scipy.stats import t as t_vert
    a = lambda k: np.array([z[k] for z in zeilen])
    aus = {k: zusammenfassung(a(k)) for k in zeilen[0] if k not in ("seed", "datensatz")}
    d1 = a("born_S1000") - a("born_exakt")
    se = d1.std(ddof=1) / math.sqrt(len(d1)) if len(d1) > 1 else 1.0
    q = t_vert.ppf(0.95, max(1, len(d1) - 1))
    aus["A1_diff"], aus["A1_ki_oben"], aus["A1_ki_unten"] = float(d1.mean()), float(d1.mean() + q * se), float(d1.mean() - q * se)
    d2 = a("paritaet_S100") - a("born_S100")
    aus["A2_diff"], aus["A2_p"] = float(d2.mean()), wilcoxon_exakt(d2)
    return aus


def urteil(a):
    u = {}
    if a["A1_ki_oben"] < -0.01:
        u["A-1"] = "H1: Schuesse sind ein Engpass (born verliert bei S=1.000 nachweislich > 1 pp)"
    elif a["A1_ki_unten"] > -0.01:
        u["A-1"] = "H0: bei S=1.000 nachweislich < 1 pp Verlust -- Schuesse kein Engpass"
    else:
        u["A-1"] = "HU: KI ueberdeckt die 1-pp-Grenze"
    u["A-2"] = ("H1: paritaet robuster als born bei S=100" if a["A2_diff"] > 0 and a["A2_p"] < 0.05
                else ("H0: born signifikant besser" if a["A2_diff"] < 0 and a["A2_p"] < 0.05
                      else "HU: kein Unterschied nachweisbar"))
    return u


def vortest(ziehungen=5):
    rng = np.random.default_rng(5)
    ok, grenz = True, 0
    for born_ex, born1000, par100, born100, erw in [
            (0.96, 0.90, 0.90, 0.80, ("H1", "H1")), (0.96, 0.958, 0.80, 0.90, ("H0", "H0")),
            (0.96, 0.95, 0.85, 0.85, ("HU", "HU"))]:
        for _ in range(ziehungen):
            zeilen = [{"seed": s, "datensatz": "ziffern", "born_exakt": born_ex + rng.normal(0, .003),
                       "born_S1000": born1000 + rng.normal(0, .003), "paritaet_S100": par100 + rng.normal(0, .003),
                       "born_S100": born100 + rng.normal(0, .003)} for s in range(8)]
            u = urteil(auswertung(zeilen))
            if not (u["A-1"].startswith(erw[0]) and u["A-2"].startswith(erw[1])):
                if erw == ("HU", "HU") and u["A-1"].startswith("HU"):
                    grenz += 1        # Nullfall A-2: Fehlalarm mit Rate alpha erlaubt
                    continue
                ok = False
                print("  VORTEST FEHLER", erw, u)
    # Baustein: Paritaetsmerkmale einer Basiszustands-Verteilung sind +-1
    P = paritaets_matrix(3)
    e = torch.zeros(8); e[5] = 1
    if not torch.all((e @ P).abs() == 1):
        ok = False
        print("  VORTEST FEHLER: Paritaeten falsch")
    return ok and grenz <= 1


if __name__ == "__main__":
    import sys
    from labor.gemeinsam import speichere
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
    e = lauf(schnell="--schnell" in sys.argv)
    print(speichere("I-008" + ("-schnell" if "--schnell" in sys.argv else ""), e))
    print(e["urteil"])
