"""Abteilung Verschraenkung -- der Verschraenkungs-Kompass.

IDEE (aus der DMRG-Quantenchemie): Wie gut ein Matrixprodukt (MPS / Tensor-
Train) einen Zustand trifft, haengt stark von der REIHENFOLGE der Orbitale ab.
Man ordnet sie so, dass stark verschraenkte (hohe gegenseitige Information)
nebeneinander liegen -- per Fiedler-Vektor der MI-Matrix (Legeza/Solyom 2003,
Rissler/Noack/White 2006, Barcza u. a. 2011).

UEBERTRAGUNG: Eine trainierte Gewichtsmatrix 256x256 ist ein „Zustand“ auf
16 Bits (8 Ausgangs-, 8 Eingangsbits). Wir ordnen die 16 Bits nach ihrer
gegenseitigen Information und komprimieren erst dann als Tensor-Train.

FRAGE V-1: Ist die MI-Ordnung bei gleicher Parameterzahl besser als eine
zufaellige Ordnung?  V-2: Ist sie besser als die beste „menschliche“
Ordnung (natuerlich o1..o8 i1..i8 / verzahnt o1 i1 o2 i2 ...)?
Kontrolle (Rotes Team): gewoehnliche Niedrigrang-SVD gleicher Parameterzahl.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from labor.gemeinsam import bilder, genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt

BITS = 16
BUDGETS = [512, 1024, 2048, 4096]     # Parameter der komprimierten 256x256-Matrix (dicht: 65.536)
N_ZUFALL = 8
HYPOTHESEN = {
    "V-1": "MI-Ordnung: relativer Fehler <= 0,9 x mittlerer Fehler zufaelliger Ordnungen "
           "beim mittleren Budget (2.048), Wilcoxon ueber Seeds p<0,05.",
    "V-2": "MI-Ordnung schlaegt die beste der drei menschlichen Ordnungen (natuerlich, "
           "verzahnt, Morton/Z-Kurve der Pixel) beim mittleren Budget, Wilcoxon p<0,05.",
    "V-3": "Beschreibend: Genauigkeit des Netzes mit komprimierter Schicht je Ordnung, "
           "gegen Niedrigrang-SVD gleicher Parameterzahl.",
}


# ------------------------------------------------------------ Tensor-Train
def tt_komprimiere(vektor, ordnung, budget):
    """TT-SVD einer 2^16-Tabelle in gegebener Bitreihenfolge; groesste gemeinsame
    Bonddimension chi, deren Parameterzahl <= budget. Gibt (Rekonstruktion, param)."""
    T = vektor.reshape([2] * BITS).transpose(ordnung)
    lo, hi = 1, 256                     # Parameterzahl waechst monoton mit chi -> Bisektion
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _tt_svd(T, mid)[1] <= budget:
            lo = mid
        else:
            hi = mid - 1
    rek, p = _tt_svd(T, lo)
    inv = np.argsort(ordnung)
    return rek.reshape([2] * BITS).transpose(inv).reshape(-1), p


def _tt_svd(T, chi):
    kerne, r, rest = [], 1, T.reshape(1, -1)
    for k in range(BITS - 1):
        M = rest.reshape(r * 2, -1)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        r_neu = min(chi, len(S))
        kerne.append(U[:, :r_neu].reshape(r, 2, r_neu))
        rest = S[:r_neu, None] * Vt[:r_neu]
        r = r_neu
    kerne.append(rest.reshape(r, 2, 1))
    param = sum(k.size for k in kerne)
    v = kerne[0].reshape(2, -1)
    for k in kerne[1:]:
        v = (v.reshape(-1, k.shape[0]) @ k.reshape(k.shape[0], -1))
    return v.reshape(-1), param


def mi_matrix(vektor):
    """Gegenseitige Information zwischen allen Bitpaaren des normierten Zustands."""
    T = (vektor / np.linalg.norm(vektor)).reshape([2] * BITS)

    def entropie(achsen):
        M = np.moveaxis(T, achsen, list(range(len(achsen)))).reshape(2 ** len(achsen), -1)
        ew = np.clip(np.linalg.eigvalsh(M @ M.T), 1e-15, None)
        return float(-(ew * np.log(ew)).sum())
    s1 = [entropie([i]) for i in range(BITS)]
    I = np.zeros((BITS, BITS))
    for i in range(BITS):
        for j in range(i + 1, BITS):
            I[i, j] = I[j, i] = s1[i] + s1[j] - entropie([i, j])
    return I


def fiedler_ordnung(I):
    L = np.diag(I.sum(1)) - I
    _, v = np.linalg.eigh(L)
    return list(np.argsort(v[:, 1]))


def svd_niedrigrang(W, budget):
    r = max(1, budget // (W.shape[0] + W.shape[1]))
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r], r * (W.shape[0] + W.shape[1])


# ------------------------------------------------------------ Lauf
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 10)

    def forward(self, x):
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


def _seed(a):
    s, datensatz, schritte = a
    xtr, ytr, xte, yte = bilder(datensatz, s)
    torch.manual_seed(s)
    m = MLP()
    W0 = m.a.weight.detach().clone()
    trainiere(m, xtr, ytr, schritte, 3e-3, seed=s)
    # Erste Schicht (Eingang = Pixel, Bits haben Bedeutung: 4 Zeilen-, 4 Spaltenbits).
    # Komprimiert wird nur das GELERNTE dW = W - W0; der Zufallsstart ist
    # unkomprimierbar und bleibt (wie bei LoRA) als fester Anteil stehen.
    W = (m.a.weight.detach() - W0).numpy().astype(np.float64)   # Bits: o1..o8 i1..i8
    v = W.reshape(-1)
    I = mi_matrix(v)
    rng = np.random.default_rng(1000 + s)
    ordnungen = {"natuerlich": list(range(16)),
                 "verzahnt": [x for k in range(8) for x in (k, 8 + k)],
                 "morton": list(range(8)) + [8, 12, 9, 13, 10, 14, 11, 15],
                 "mi_fiedler": fiedler_ordnung(I)}
    ordnungen.update({f"zufall_{k}": list(rng.permutation(16)) for k in range(N_ZUFALL)})
    z = {"seed": s, "acc_dicht": genauigkeit(m(xte), yte), "mi_ordnung": [int(x) for x in ordnungen["mi_fiedler"]],
         "mi_matrix": I.round(5).tolist()}

    def acc_mit(Wneu):
        alt = m.a.weight.data.clone()
        m.a.weight.data = W0 + torch.tensor(Wneu, dtype=torch.float32)
        a = genauigkeit(m(xte), yte)
        m.a.weight.data = alt
        return a

    for B in BUDGETS:
        for name, o in ordnungen.items():
            rek, p = tt_komprimiere(v, o, B)
            z[f"{name}_B{B}_fehler"] = float(np.linalg.norm(rek - v) / np.linalg.norm(v))
            z[f"{name}_B{B}_param"] = int(p)
            if not name.startswith("zufall") or name == "zufall_0":
                z[f"{name}_B{B}_acc"] = acc_mit(rek.reshape(256, 256))
        rek, p = svd_niedrigrang(W, B)
        z[f"svd_B{B}_fehler"] = float(np.linalg.norm(rek - W) / np.linalg.norm(W))
        z[f"svd_B{B}_param"], z[f"svd_B{B}_acc"] = int(p), acc_mit(rek)
        z[f"zufall_B{B}_fehler"] = float(np.mean([z[f"zufall_{k}_B{B}_fehler"] for k in range(N_ZUFALL)]))
    return z


def lauf(datensatz="mnist", seeds=range(8), schritte=600, schnell=False):
    if schnell:
        seeds, schritte = range(2), 60
    t0 = time.time()
    zeilen = parallel(_seed, [(s, datensatz, schritte) for s in seeds])
    aus = auswertung(zeilen)
    return {"abteilung": "kompass", "datensatz": datensatz, "hypothesen": HYPOTHESEN, "zeilen": zeilen,
            "auswertung": aus, "urteil": urteil(aus), "sekunden": time.time() - t0}


def auswertung(zeilen):
    a = lambda k: np.array([z[k] for z in zeilen])
    B = 2048
    aus = {"n": len(zeilen)}
    for b in BUDGETS:
        for name in ("mi_fiedler", "zufall", "natuerlich", "verzahnt", "morton", "svd"):
            aus[f"{name}_B{b}_fehler"] = float(a(f"{name}_B{b}_fehler").mean())
            if f"{name}_B{b}_acc" in zeilen[0]:
                aus[f"{name}_B{b}_acc"] = float(a(f"{name}_B{b}_acc").mean())
    aus["acc_dicht"] = float(a("acc_dicht").mean())
    mi, zu = a(f"mi_fiedler_B{B}_fehler"), a(f"zufall_B{B}_fehler")
    mensch = np.minimum.reduce([a(f"{n}_B{B}_fehler") for n in ("natuerlich", "verzahnt", "morton")])
    aus["V1_verhaeltnis"] = float((mi / zu).mean())
    aus["V1_p"] = wilcoxon_exakt(mi - zu)
    aus["V2_diff"] = float((mi - mensch).mean())
    aus["V2_p"] = wilcoxon_exakt(mi - mensch)
    return aus


def urteil(a):
    u = {}
    u["V-1"] = ("H1: MI-Ordnung >=10 % besser als Zufall" if a["V1_verhaeltnis"] <= 0.9 and a["V1_p"] < 0.05
                else ("HU: besser, aber < 10 %" if a["V1_verhaeltnis"] < 1 and a["V1_p"] < 0.05
                      else "H0: MI-Ordnung nicht nachweisbar besser als Zufall"))
    u["V-2"] = ("H1: MI schlaegt beste menschliche Ordnung" if a["V2_diff"] < 0 and a["V2_p"] < 0.05
                else ("H0: menschliche Ordnung signifikant besser" if a["V2_diff"] > 0 and a["V2_p"] < 0.05
                      else "HU: kein Unterschied nachweisbar"))
    return u


def vortest(ziehungen=5):
    rng = np.random.default_rng(1)
    ok = True
    fehl_null = 0
    for mi_f, zu_f, me_f, erw in [(0.5, 0.8, 0.6, ("H1", "H1")), (0.78, 0.8, 0.6, ("HU", "H0")),
                                  (0.8, 0.8, 0.8, ("H0", "HU"))]:
        for _ in range(ziehungen):
            zeilen = []
            for s in range(8):
                z = {"seed": s, "acc_dicht": 0.97}
                for b in BUDGETS:
                    for n, mu in [("mi_fiedler", mi_f), ("zufall", zu_f), ("natuerlich", me_f),
                                  ("verzahnt", me_f + 0.05), ("morton", me_f + 0.02), ("svd", 0.7)]:
                        z[f"{n}_B{b}_fehler"] = mu + rng.normal(0, 0.003)
                zeilen.append(z)
            u = urteil(auswertung(zeilen))
            if not (u["V-1"].startswith(erw[0]) and u["V-2"].startswith(erw[1])):
                if mi_f == zu_f == me_f:      # Nullfall: Fehlalarm mit Rate alpha erlaubt
                    fehl_null += 1
                    continue
                ok = False
                print("  VORTEST FEHLER", mi_f, zu_f, me_f, u)
    if fehl_null > 1:
        ok = False
        print("  VORTEST FEHLER: Nullfall schlaegt zu oft an", fehl_null)
    # Baustein-Pruefung: TT mit grossem Budget muss exakt sein; MI eines Produktzustands = 0
    v = np.random.default_rng(0).normal(size=2 ** 16)
    rek, _ = tt_komprimiere(v, list(range(16)), 10 ** 7)
    if np.linalg.norm(rek - v) / np.linalg.norm(v) > 1e-8:
        ok = False
        print("  VORTEST FEHLER: TT-SVD nicht exakt")
    prod = np.ones(1)
    for _ in range(16):
        prod = np.kron(prod, np.random.default_rng(len(prod)).normal(size=2))
    if mi_matrix(prod).max() > 1e-6:
        ok = False
        print("  VORTEST FEHLER: MI eines Produktzustands != 0")
    return ok


if __name__ == "__main__":
    import sys
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
