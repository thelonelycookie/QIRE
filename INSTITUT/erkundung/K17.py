"""K17 -- Zufallsmerkmale + geschlossener Kopf, Suche statt Gradient (Erkundung Welle 2, KEIN Befund).

MECHANISMUS: Feste Fastfood-Merkmale z² (H D2 H D1 x, ±1-Diagonalen aus dem Seed, 0 gelernte Bytes in der Merkmalsschicht), danach
  ein Ridge-Kopf in EINEM Durchgang (Normalgleichung, kein Backprop). Optional: die ±1-Vorzeichen werden per Bit-Flip-Suche
  (Hügelklettern auf dem Validierungsfehler des Ridge-Kopfes) statt per Gradient gewählt.
TEST (MNIST 16x16, Zusatz Fashion 16x16, Hochrechnung MNIST 32x32): Merkmalszahl D = 1N, 4N, 16N (N = Eingangsbreite 256; auf 32x32 nur 1N, 4N,
  16N wären 16384 Merkmale, zu groß für die Normalgleichung im Budget).
  Teil A: Fastfood z² + Ridge bei D = 1N/4N/16N.
  Teil B: Suche (Bit-Flip, gleiche Zahl Auswertungen) gegen beste von 400 Zufallsdraws (Auswahl jeweils auf Validierung), D = 2N.
KONTROLLEN (hart): (1) Zufalls-ReLU-Merkmale (dicht Gauß) gleicher Zahl + Ridge; (2) dichte Gauß-Merkmale mit z² (trennt Struktur von Aktivierung);
  (3) Fastfood mit ReLU statt z²; (4) trainierter 1-Schicht-MLP (Adam, 20 Epochen, Lernrate auf Validierung) mit gleichen idealen Inferenz-FLOPs wie Fastfood+Kopf.
AUSWEIS: Parameter (gelernt), FLOPs (ideal gezählt: 2 WHT je Block = 2·D0·log2 D0, 3·D0 für Vorzeichen+Quadrat, Kopf 20·D), Bytes (fp16-Kopf + ggf. 1 Bit je Vorzeichen).
Nur Tuning-Seeds 950-959. Wirkung klassisch (I-038); „quanten-inspiriert“.
"""
from __future__ import annotations
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hilfe_k17_20 as H
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abteilungen.struktur_gross import Fastfood
from labor.gemeinsam import bilder

KID = "K17"


def ff_flops(D0, m):
    return int(m * (2 * D0 * math.log2(D0) + 3 * D0) + 20 * m * D0)


def dicht_flops(D0, D):
    return int(2 * D0 * D + D + 20 * D)


class Merkmale:
    """Fest, ohne gelernte Parameter. art: ff_z2 | ff_relu | dicht_relu | dicht_z2."""

    def __init__(self, art, D0, m, gen):
        self.art, self.D0, self.m = art, D0, m
        if art.startswith("ff"):
            self.ff = Fastfood(D0, m * D0, False, gen)
        else:
            self.W = torch.randn(m * D0, D0, generator=gen) / math.sqrt(D0)

    def __call__(self, x):
        with torch.no_grad():
            z = self.ff(x) if self.art.startswith("ff") else x @ self.W.T
            return z * z if self.art.endswith("z2") else F.relu(z)

    def setze_vorzeichen(self, d1, d2):
        self.ff.d1.copy_(d1)
        self.ff.d2.copy_(d2)


def standardisiere(phi_tr, *anderes):
    mu, sd = phi_tr.mean(0, keepdim=True), phi_tr.std(0, keepdim=True).clamp_min(1e-6)
    return [(p - mu) / sd for p in (phi_tr,) + anderes]


def ridge_loese(P, Y, lams):
    """Normalgleichung (float64); Y one-hot, zentriert. Gibt {lam: W} zurück (lam relativ zu spur(G)/D)."""
    G = (P.T @ P).double()
    B = (P.T @ Y).double()
    s = float(torch.trace(G)) / G.shape[0]
    out = {}
    eye = torch.eye(G.shape[0], dtype=torch.float64)
    for l in lams:
        L = torch.linalg.cholesky(G + l * s * eye)
        out[l] = torch.cholesky_solve(B, L).float()
    return out


def onehot(y):
    return F.one_hot(y, 10).float()


LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def ridge_komplett(phi_tr, ytr, phi_va, yva, phi_te, yte):
    Ptr, Pva, Pte = standardisiere(phi_tr, phi_va, phi_te)
    Y = onehot(ytr)
    ym = Y.mean(0, keepdim=True)
    Ws = ridge_loese(Ptr, Y - ym, LAMS)
    va = {l: float(((Pva @ W + ym).argmax(1) == yva).float().mean()) for l, W in Ws.items()}
    lb = max(va, key=va.get)
    W = Ws[lb]
    return {"lam": lb, "acc_val": va[lb], "acc_test": float(((Pte @ W + ym).argmax(1) == yte).float().mean()),
            "acc_train": float(((Ptr @ W + ym).argmax(1) == ytr).float().mean())}


def mlp_kontrolle(xtr, ytr, xva, yva, xte, yte, h, seed, epochen):
    best = None
    for lr in (1e-3, 3e-3, 1e-2):
        torch.manual_seed(seed)
        netz = nn.Sequential(nn.Linear(xtr.shape[1], h), nn.ReLU(), nn.Linear(h, 10))
        opt = torch.optim.Adam(netz.parameters(), lr=lr)
        n, bs = len(xtr), 128
        spe = math.ceil(n / bs)
        tot = epochen * spe
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / tot)))
        g = torch.Generator().manual_seed(seed + 3)
        for e in range(epochen):
            p = torch.randperm(n, generator=g)
            for k in range(spe):
                b = p[k * bs:(k + 1) * bs]
                l = F.cross_entropy(netz(xtr[b]), ytr[b])
                opt.zero_grad(); l.backward(); opt.step(); sched.step()
        with torch.no_grad():
            va = float((netz(xva).argmax(1) == yva).float().mean())
            te = float((netz(xte).argmax(1) == yte).float().mean())
        if best is None or va > best["acc_val"]:
            best = {"lr": lr, "acc_val": va, "acc_test": te}
    best["parameter"] = int(sum(p.numel() for p in netz.parameters()))
    best["h"] = h
    return best


def teil_a(name, seite, mults, seed, schnell):
    n_tr = 2000 if schnell else 10000
    X, Y, Xte, Yte = bilder(name, seed, n_train=n_tr + 2000, n_test=1000 if schnell else 5000, seite=seite)
    xtr, ytr, xva, yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    mu = xtr.mean(0, keepdim=True)
    xtr, xva, xte = xtr - mu, xva - mu, Xte - mu
    D0 = seite * seite
    out = {}
    for m in mults:
        D = m * D0
        zeile = {"D": D, "m": m}
        gen = lambda: torch.Generator().manual_seed(seed * 1009 + m)
        for art in ("ff_z2", "dicht_relu", "dicht_z2", "ff_relu"):
            t = time.time()
            mk = Merkmale(art, D0, m, gen())
            r = ridge_komplett(mk(xtr), ytr, mk(xva), yva, mk(xte), Yte)
            r["sekunden"] = time.time() - t
            r["flops_inferenz"] = ff_flops(D0, m) if art.startswith("ff") else dicht_flops(D0, D)
            r["parameter_gelernt"] = 10 * D + 10                      # nur Kopf
            r["bytes_fp16"] = 2 * (10 * D + 10) + (0 if art.startswith("ff") else 2 * D * D0)   # dicht: Merkmalsmatrix müsste gespeichert werden
            r["bytes_seed_variante"] = 2 * (10 * D + 10)
            zeile[art] = r
        h = max(2, int(ff_flops(D0, m) / (2 * D0 + 21)))
        t = time.time()
        zeile["mlp_flops_gleich"] = mlp_kontrolle(xtr, ytr, xva, yva, xte, Yte, h, seed, 3 if schnell else 20)
        zeile["mlp_flops_gleich"]["flops_inferenz"] = int(2 * D0 * h + h + 20 * h)
        zeile["mlp_flops_gleich"]["bytes_fp16"] = 2 * (D0 * h + h + 10 * h + 10)
        zeile["mlp_flops_gleich"]["sekunden"] = time.time() - t
        out[f"{m}N"] = zeile
    return out


def teil_b(name, seed, schnell, m=2, n_ev=400, n_ev_lang=1600):
    """Bit-Flip-Suche gegen beste von n_ev Zufallsdraws; gleiche Zahl Auswertungen; Auswahl auf Validierung."""
    seite = 16
    n_tr = 1000 if schnell else 4000
    n_ev = 20 if schnell else n_ev
    n_ev_lang = 40 if schnell else n_ev_lang
    X, Y, Xte, Yte = bilder(name, seed, n_train=n_tr + 2000, n_test=1000 if schnell else 5000, seite=seite)
    xtr, ytr, xva, yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]
    mu = xtr.mean(0, keepdim=True)
    xtr, xva, xte = xtr - mu, xva - mu, Xte - mu
    D0 = 256
    Ytr = onehot(ytr)
    ym = Ytr.mean(0, keepdim=True)
    gen = torch.Generator().manual_seed(seed * 77 + 5)
    mk = Merkmale("ff_z2", D0, m, gen)

    def auswerten(d1, d2, lam=1e-2):
        mk.setze_vorzeichen(d1, d2)
        Ptr, Pva = standardisiere(mk(xtr), mk(xva))
        W = ridge_loese(Ptr, Ytr - ym, (lam,))[lam]
        P = Pva @ W + ym
        return float(((P - onehot(yva)) ** 2).mean()), float((P.argmax(1) == yva).float().mean())

    def zufall():
        return ((torch.randint(0, 2, (m, D0), generator=gen) * 2 - 1).float(), (torch.randint(0, 2, (m, D0), generator=gen) * 2 - 1).float())

    def testacc(d1, d2):
        mk.setze_vorzeichen(d1, d2)
        r = ridge_komplett(mk(xtr), ytr, mk(xva), yva, mk(xte), Yte)
        return r["acc_test"], r["acc_val"]

    res = {"m": m, "n_train_suche": n_tr, "lam_suche": 1e-2}
    for tag, n in (("kurz", n_ev), ("lang", n_ev_lang)):
        t = time.time()
        draws = [zufall() for _ in range(n)]
        sc = [auswerten(*d) for d in draws]
        mse = np.array([s[0] for s in sc])
        bi = int(mse.argmin())
        # Suche: Start = erster Draw, n-1 Flips (je 1 Bit), annehmen bei besserem Val-MSE
        d1, d2 = [t_.clone() for t_ in draws[0]]
        cur = sc[0][0]
        verlauf = [cur]
        rg = torch.Generator().manual_seed(seed * 13 + n)
        for _ in range(n - 1):
            which = int(torch.randint(0, 2, (1,), generator=rg))
            j = int(torch.randint(0, m * D0, (1,), generator=rg))
            dd = d1 if which == 0 else d2
            dd.view(-1)[j] *= -1
            v, _ = auswerten(d1, d2)
            if v < cur:
                cur = v
            else:
                dd.view(-1)[j] *= -1
            verlauf.append(cur)
        rb = testacc(*draws[bi])
        rs = testacc(d1, d2)
        r0 = testacc(*draws[0])
        # Zufallsstreuung der Testgenauigkeit: 5 beliebige Draws
        zs = [testacc(*draws[i])[0] for i in range(min(5, n))]
        res[tag] = {"auswertungen": n, "beste_zufall": {"val_mse": float(mse[bi]), "acc_test": rb[0], "acc_val": rb[1]},
                    "suche": {"val_mse": cur, "acc_test": rs[0], "acc_val": rs[1]},
                    "start_draw": {"val_mse": float(sc[0][0]), "acc_test": r0[0]},
                    "zufall_mittel_val_mse": float(mse.mean()), "zufall_sd_val_mse": float(mse.std()),
                    "zufall_acc_test_5draws_mittel": float(np.mean(zs)), "zufall_acc_test_5draws_sd": float(np.std(zs)),
                    "suche_verlauf_val_mse_jede_10": verlauf[::10], "sekunden": time.time() - t,
                    "bytes_vorzeichen_bits": int(2 * m * D0)}
    return res


def main(seed: int, schnell: bool = False) -> dict:
    H.pruefe_seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    r = {"kid": KID, "seed": seed, "schnell": schnell}
    r["A_mnist16"] = teil_a("mnist", 16, (1,) if schnell else (1, 4, 16), seed, schnell)
    r["A_fashion16"] = teil_a("fashion_mnist", 16, (1,) if schnell else (1, 4, 16), seed, schnell)
    r["A_mnist32_hochrechnung"] = teil_a("mnist", 32, (1,) if schnell else (1, 4), seed, schnell)
    r["B_suche_mnist16"] = teil_b("mnist", seed, schnell)
    r["hinweis"] = ("Erkundung, kein Befund. 16N auf 32x32 (16384 Merkmale) entfällt (Gram 16384² zu groß). "
                    "bytes_fp16 'dicht': enthält die Merkmalsmatrix; Fastfood-Merkmale kosten 0 Bytes (Seed), Vorzeichensuche 1 Bit je Vorzeichen.")
    r["sekunden_gesamt"] = time.time() - t0
    H.speichere(KID, seed, r, schnell)
    return r


if __name__ == "__main__":
    H.cli(KID, main)
