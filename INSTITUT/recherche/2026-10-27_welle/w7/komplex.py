"""w7: komplexwertige Phasen-Schichten mit Interferenz (Erkundung, Tuning-Seeds 960-964).
Schicht: z_j = sum_k W_jk x_k, Ausgang |z|^2 (Born). Arme:
  real   : reell dicht, ReLU/GELU
  kdicht : komplex dicht (Re,Im frei, 2 Param/Eintrag), |z|^2
  phase  : nur Phase theta, Betrag fest 1/sqrt(n), 1 Param/Eintrag, |z|^2
FLOPs (Eingang reell): reell 2*n*H; komplex/Phase 2 reelle Matmuls = 4*n*H (Zaehlung: 1 MAC = 2 FLOP).
"""
import math, sys, os, json, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)


class Born(nn.Module):
    """Eingang reell (B,n) -> |W x|^2 (B,H). art: kdicht | phase."""
    def __init__(self, n, H, art, gen=None):
        super().__init__()
        self.art, self.n, self.H = art, n, H
        if art == "kdicht":
            s = 1 / math.sqrt(n)
            self.re = nn.Parameter(torch.randn(H, n, generator=gen) * s)
            self.im = nn.Parameter(torch.randn(H, n, generator=gen) * s)
        else:
            self.th = nn.Parameter(torch.rand(H, n, generator=gen) * 2 * math.pi)
        self.skala = nn.Parameter(torch.tensor(1.0)) if art == "phase" else None
        self.b = nn.Parameter(torch.zeros(H))

    def forward(self, x):
        if self.art == "kdicht":
            a, c = x @ self.re.T, x @ self.im.T
        else:
            s = self.skala / math.sqrt(self.n)
            a, c = x @ torch.cos(self.th).T * s, x @ torch.sin(self.th).T * s
        return a * a + c * c + self.b


def param_born(n, H, art):
    return {"kdicht": 2 * n * H, "phase": n * H + 1}[art] + H


def flops_born(n, H, art):
    return 4 * n * H            # zwei reelle Matmuls (Re, Im), Eingang reell; +Quadrat vernachlaessigt


class MLP(nn.Module):
    def __init__(self, n, H, art, k=10, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed * 7 + 1)
        if art == "real":
            self.l1 = nn.Linear(n, H)
            self.akt = F.relu
        else:
            self.l1 = Born(n, H, art, g)
            self.akt = nn.Identity()
        self.norm = nn.LayerNorm(H, elementwise_affine=False)   # gleich fuer alle Arme (0 Parameter)
        self.l2 = nn.Linear(H, k)

    def forward(self, x):
        return self.l2(self.norm(self.akt(self.l1(x))))


def mlp_kosten(n, H, art, k=10):
    if art == "real":
        return n * H + H + H * k + k, 2 * n * H + 2 * H * k
    return param_born(n, H, art) + H * k + k, flops_born(n, H, art) + 2 * H * k


def finde_H(n, art, ziel, nach="param", k=10):
    idx = 0 if nach == "param" else 1
    return min(range(2, 4096), key=lambda H: abs(mlp_kosten(n, H, art, k)[idx] - ziel))


def trainiere_mlp(art, H, lr, seed, tr, va, epochen=30, bs=256):
    x, y = tr
    torch.manual_seed(seed)
    m = MLP(x.shape[1], H, art, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(y); sch = epochen * (n // bs); t = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / sch)))
    g = torch.Generator().manual_seed(seed + 3)
    for _ in range(epochen):
        p = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            b = p[i:i + bs]
            l = F.cross_entropy(m(x[b]), y[b])
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc = lambda d: (m(d[0]).argmax(1) == d[1]).float().mean().item()
        return acc(va), m
