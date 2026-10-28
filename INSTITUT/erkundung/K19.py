"""K19 -- Lokale Blöcke auf der Pixelseite (Z-Kurve, getrennte Filter) (Erkundung Welle 2, KEIN Befund).

MECHANISMUS: Erste Schicht als Blockdiagonale über Pixelgruppen, die in der Z-Kurve (Morton-Ordnung) benachbart sind (Blockgröße 16 = 4x4-Kachel,
  Blockgröße 64 = 8x8-Quadrant), bzw. Filter als Produkt a⊗b (Rang-1-Filter aus Zeilen- und Spaltenvektor).
TEST: MNIST und Fashion 16x16, n_train = 200 / 1000 / 6000, Test = 5000 offizielle Testbilder, Validierung 1000 (aus dem Trainingsrest, zum Wählen der Lernrate).
  Netz: Block-/Spezialschicht -> ReLU -> dichter Kopf (10). Adam, volle Epochen (>= 1000 Schritte), Lernrate je Arm auf Validierung aus {1e-3, 3e-3, 1e-2}.
ARME (Blockteil, H = 16·b verdeckte Einheiten):
  z16    Z-Kurve, 16 Blöcke à 16 Pixel (4x4-Kacheln), je b Einheiten
  z64    Z-Kurve, 4 Quadranten à 64 Pixel (8x8), je b Einheiten (gleiche Blockgewichte wie z16, aber nur 4b Einheiten)
  streif Rasterordnung, 16 Blöcke à 16 Pixel (Zeilenstreifen 1x16)
  perm   zufällige feste Permutation, 16 Blöcke à 16 Pixel  [KONTROLLE: gleiche Struktur ohne Nachbarschaft]
  dicht_flops  dichte Schicht 256->h mit gleichen Gesamt-MACs wie z16 (inkl. Kopf)   [KONTROLLE]
  dicht_param  dichte Schicht mit gleicher Gesamt-Parameterzahl (inkl. Bias und Kopf)  [KONTROLLE]
  dicht_breit  dicht 256->H (gleiche Breite H, viele Parameter) [obere Schranke]
  dicht_kronmacs  dicht mit den (naiv gezählten) MACs von kron  [KONTROLLE für kron]
  kron   Filter a⊗b (je Einheit 16+16 Zahlen), H Einheiten  vs  lr  U·V (Rang r mit gleicher Parameterzahl)  [KONTROLLE für kron]
b = 8 (H = 128). Ausweis: Parameter gesamt, MACs je Bild, Genauigkeit Test/Val. Nur Tuning-Seeds. Quanten-inspiriert, klassisch.
"""
from __future__ import annotations
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hilfe_k17_20 as H
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from labor.gemeinsam import bilder

KID = "K19"
B_UNITS = 8


def morton_perm(seite=16):
    """perm[k] = Rasterindex des k-ten Pixels in Z-Ordnung."""
    def bits(r, c):
        z = 0
        for i in range(int(math.log2(seite))):
            z |= ((r >> i) & 1) << (2 * i + 1) | ((c >> i) & 1) << (2 * i)
        return z
    z = [(bits(r, c), r * seite + c) for r in range(seite) for c in range(seite)]
    return torch.tensor([i for _, i in sorted(z)])


class Block(nn.Module):
    """Blockdiagonale: G Gruppen à bs Pixel -> je u Einheiten; Pixelordnung perm."""
    def __init__(self, perm, bs, u, gen):
        super().__init__()
        self.register_buffer("perm", perm)
        self.G, self.bs, self.u = 256 // bs, bs, u
        self.W = nn.Parameter(torch.randn(self.G, bs, u, generator=gen) / math.sqrt(bs))
        self.b = nn.Parameter(torch.zeros(self.G * u))

    def forward(self, x):
        x = x[:, self.perm].reshape(len(x), self.G, self.bs)
        return torch.einsum("bgi,giu->bgu", x, self.W).reshape(len(x), -1) + self.b

    def macs(self):
        return self.G * self.bs * self.u


class Kron(nn.Module):
    """Einheit k: a_k^T X b_k (Rang-1-Filter a⊗b auf dem 16x16-Bild)."""
    def __init__(self, Hn, gen, s=16):
        super().__init__()
        self.s = s
        self.A = nn.Parameter(torch.randn(Hn, s, generator=gen) / math.sqrt(s) ** 1.0)
        self.Bm = nn.Parameter(torch.randn(Hn, s, generator=gen) / math.sqrt(s) ** 1.0)
        self.b = nn.Parameter(torch.zeros(Hn))

    def forward(self, x):
        X = x.reshape(len(x), self.s, self.s)
        return torch.einsum("bij,ki,kj->bk", X, self.A, self.Bm) + self.b

    def macs(self):
        return self.A.shape[0] * (self.s * self.s + self.s)


class LowRank(nn.Module):
    def __init__(self, Hn, r, gen):
        super().__init__()
        self.V = nn.Parameter(torch.randn(256, r, generator=gen) / 16.0)
        self.U = nn.Parameter(torch.randn(r, Hn, generator=gen) / math.sqrt(r))
        self.b = nn.Parameter(torch.zeros(Hn))
        self.r, self.Hn = r, Hn

    def forward(self, x):
        return (x @ self.V) @ self.U + self.b

    def macs(self):
        return self.r * (256 + self.Hn)


class Dicht(nn.Module):
    def __init__(self, h, gen):
        super().__init__()
        self.l = nn.Linear(256, h)
        with torch.no_grad():
            self.l.weight.copy_(torch.randn(h, 256, generator=gen) / 16.0)
        self.h = h

    def forward(self, x):
        return self.l(x)

    def macs(self):
        return 256 * self.h


class Netz(nn.Module):
    def __init__(self, erste, h):
        super().__init__()
        self.erste, self.kopf = erste, nn.Linear(h, 10)
        self.h = h

    def forward(self, x):
        return self.kopf(F.relu(self.erste(x)))

    def macs(self):
        return self.erste.macs() + self.h * 10

    def params(self):
        return int(sum(p.numel() for p in self.parameters()))


def baue(arm, gen):
    b = B_UNITS
    Hn = 16 * b
    zp = morton_perm()
    ident = torch.arange(256)
    rp = torch.randperm(256, generator=gen)
    if arm == "z16": return Netz(Block(zp, 16, b, gen), Hn)
    if arm == "z64": return Netz(Block(zp, 64, b, gen), 4 * b)       # gleiche Blockgewichte wie z16 (4*64*b = 16*16*b), aber nur 4b Einheiten
    if arm == "streif": return Netz(Block(ident, 16, b, gen), Hn)
    if arm == "perm": return Netz(Block(rp, 16, b, gen), Hn)
    if arm == "dicht_breit": return Netz(Dicht(Hn, gen), Hn)
    ref = Netz(Block(zp, 16, b, gen), Hn)
    if arm == "dicht_flops":
        h = max(1, round(ref.macs() / (256 + 10)))
        return Netz(Dicht(h, gen), h)
    if arm == "dicht_param":
        h = max(1, round((ref.params() - 10) / (256 + 1 + 10 + 0.0)))   # 256h+h Bias + 10h+10
        return Netz(Dicht(h, gen), h)
    if arm == "kron": return Netz(Kron(Hn, gen), Hn)
    if arm == "dicht_kronmacs":
        h = max(1, round(Kron(Hn, gen).macs() / 266))
        return Netz(Dicht(h, gen), h)
    if arm == "lr":
        kp = Hn * (32 + 1)
        r = max(1, round((kp - Hn) / (256 + Hn)))
        return Netz(LowRank(Hn, r, gen), Hn)
    raise ValueError(arm)


ARME = ("z16", "z64", "streif", "perm", "dicht_flops", "dicht_param", "dicht_breit", "kron", "dicht_kronmacs", "lr")


def trainiere(netz, xtr, ytr, xva, yva, xte, yte, lr, seed, schnell):
    n = len(xtr)
    bs = min(128, n)
    spe = math.ceil(n / bs)
    ep = max(40, math.ceil(1000 / spe))
    if schnell: ep = 6
    tot = ep * spe
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / tot)))
    g = torch.Generator().manual_seed(seed + 3)
    for e in range(ep):
        p = torch.randperm(n, generator=g)
        for k in range(spe):
            b = p[k * bs:(k + 1) * bs]
            l = F.cross_entropy(netz(xtr[b]), ytr[b])
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc = lambda x, y: float((netz(x).argmax(1) == y).float().mean())
        return acc(xva, yva), acc(xte, yte), acc(xtr, ytr)


def main(seed: int, schnell: bool = False) -> dict:
    H.pruefe_seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    r = {"kid": KID, "seed": seed, "schnell": schnell, "b": B_UNITS, "ergebnis": {}}
    ns = (200,) if schnell else (200, 1000, 6000)
    for name in ("mnist",) if schnell else ("mnist", "fashion_mnist"):
        X, Y, Xte, Yte = bilder(name, seed, n_train=7000, n_test=1000 if schnell else 5000, seite=16)
        xva, yva = X[6000:7000], Y[6000:7000]
        for n in ns:
            xtr, ytr = X[:n], Y[:n]
            zeile = {}
            for arm in ARME:
                best = None
                for lr in (1e-3, 3e-3, 1e-2):
                    torch.manual_seed(seed)
                    netz = baue(arm, torch.Generator().manual_seed(seed * 17 + 1))
                    va, te, tr_ = trainiere(netz, xtr, ytr, xva, yva, Xte, Yte, lr, seed, schnell)
                    if best is None or va > best["acc_val"]:
                        best = {"lr": lr, "acc_val": va, "acc_test": te, "acc_train": tr_, "parameter": netz.params(), "macs": netz.macs(), "einheiten": netz.h}
                zeile[arm] = best
            r["ergebnis"][f"{name}_n{n}"] = zeile
    r["hinweis"] = ("Erkundung, kein Befund. Kontrolle = perm (gleiche Blockstruktur, zufällige Pixel) und dichte Netze gleicher MACs/Parameter. "
                    "dicht_flops und dicht_param fallen auf fast dasselbe h zusammen, weil MACs ≈ Gewichte; Unterschied nur Bias/Kopf. kron gegen lr (gleiche Parameter).")
    r["sekunden_gesamt"] = time.time() - t0
    H.speichere(KID, seed, r, schnell)
    return r


if __name__ == "__main__":
    H.cli(KID, main)
