import os, sys, time, math
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.path.insert(0, "/home/user/QIRE/INSTITUT")
import numpy as np, torch
torch.set_num_threads(2)
from abteilungen.clifford import QireZelle, LAGEN, GITTER, _cnot_ring_perm
from abteilungen.clifford_suche import (Bewerter, suche, kanonisch, unitaer_clifford, unitaer_winkel, normiere, born,
                                        ridge_loo, kopf_logistisch)
from abteilungen.maske_block import matrix_vm, rref_basis, span
from abteilungen.maske_eigenschaft import maske_aus_form
from labor.gemeinsam import bilder
N = 8

def daten(name, seed, n_tr, n_te):
    xtr, ytr, xte, yte = bilder(name, seed, n_train=n_tr, n_test=n_te, seite=16)
    return normiere(xtr), ytr, normiere(xte), yte

def clifford_suche(xtr, ytr, seed, steps=600, politur=100):
    bew = Bewerter(xtr, ytr, N)
    rng = np.random.default_rng(980_000 + seed)
    zuf = [bew(kanonisch(rng.integers(0, 4, (LAGEN, N))))[0] for _ in range(30)]
    skala = float(np.std(zuf)) + 1e-9
    start = rng.integers(0, 4, (LAGEN, N))
    best, s, kurve = suche(bew, start, rng, skala, schritte=steps, politur_max=politur)
    return best, s, float(np.median(zuf)), bew.bewertungen

def magie_winkel(codes, xtr, ytr, k, seed, sweeps=2):
    """k freie Winkel (zufaellige Positionen) auf einem Clifford-Geruest, Koordinatensuche auf Gitter pi/8 (LOO-Ridge)."""
    rng = np.random.default_rng(990_000 + seed)
    th = np.asarray(codes, float) * GITTER
    pos = rng.choice(LAGEN * N, size=k, replace=False)
    def sc(t): return ridge_loo(born(xtr, unitaer_winkel(t, N)), ytr)[0]
    cur = sc(th)
    for _ in range(sweeps):
        for p in pos:
            l, q = divmod(int(p), N)
            bestv, bestt = cur, th[l, q]
            for g in range(16):
                t2 = th.copy(); t2[l, q] = g * math.pi / 8
                v = sc(t2)
                if v > bestv: bestv, bestt = v, t2[l, q]
            th[l, q] = bestt; cur = bestv
    return th, pos, cur

def bent_U(k_bits, rang):
    """Blockweiser Walsh ueber Unterraum V (die k niedrigsten Pixelindexbits bzw. alle), Maske = bent/alternierende Form vom Rang `rang`."""
    qb = list(range(N - k_bits, N))  # Pixelindexbits (k_bits niedrigste)
    # V = span der k_bits niedrigsten Pixelbits
    basis = rref_basis([1 << b for b in range(k_bits)])
    V = span(basis)
    k = len(basis)
    B = np.zeros((k, k), dtype=np.int64)
    for i in range(0, rang, 2):
        B[i, i + 1] = B[i + 1, i] = 1
    m = maske_aus_form(B, V, 2 ** N, basis)
    return matrix_vm(V, m, basis)
