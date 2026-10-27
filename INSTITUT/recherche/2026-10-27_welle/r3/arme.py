"""r3: Kontrolle zu w7. Ist der Vorsprung der Nur-Phase-Netze nur der Quadrat-Effekt? (Tuning-Seeds 980-984)
Arme (Schicht 1, Eingang n reell, danach LayerNorm ohne Param., Linear->10):
 relu    : Linear(n,H), ReLU
 sq      : Linear(n,H), z^2
 sqpaar  : Linear(n,2H) -> z1^2+z2^2  (Born-Struktur ohne Phase = komplex dicht, Bias vor Quadrat)
 kdicht  : Re/Im frei, |z|^2 + Bias (w7-Variante)
 phase   : nur Phase, Betrag fest 1/sqrt(n), |z|^2 (w7)
 zufall  : Phasen zufaellig und fest, Betraege r_jk gelernt, |z|^2
Suffix _F: Breite so, dass FLOPs = Phase bei gleichem Budget (nur relu, sq).
FLOPs/Beispiel (1 MAC=2): reell 2nH; Paar/komplex 4nH; Kopf 2*H*10."""
import sys, os, math
sys.path.insert(0, "/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w7")
import torch, torch.nn as nn, torch.nn.functional as F
from komplex import Born
torch.set_num_threads(1)

class Zufall(nn.Module):
    def __init__(s, n, H, gen):
        super().__init__()
        s.n = n
        th = torch.rand(H, n, generator=gen) * 2 * math.pi
        s.register_buffer("c", torch.cos(th)); s.register_buffer("s", torch.sin(th))
        s.r = nn.Parameter(torch.full((H, n), 1 / math.sqrt(n)))
        s.b = nn.Parameter(torch.zeros(H))
    def forward(s, x):
        a = x @ (s.r * s.c).T; c = x @ (s.r * s.s).T
        return a * a + c * c + s.b

class Paar(nn.Module):
    def __init__(s, n, H):
        super().__init__(); s.l = nn.Linear(n, 2 * H); s.H = H
    def forward(s, x):
        z = s.l(x); return z[:, :s.H] ** 2 + z[:, s.H:] ** 2

class Quad(nn.Module):
    def __init__(s, n, H):
        super().__init__(); s.l = nn.Linear(n, H)
    def forward(s, x): return s.l(x) ** 2

def basis(arm): return arm.split("_")[0]

def kosten(n, H, arm, k=10):
    a = basis(arm)
    p1 = {"relu": n*H+H, "sq": n*H+H, "sqpaar": 2*n*H+2*H, "kdicht": 2*n*H+H,
          "phase": n*H+1+H, "zufall": n*H+H}[a]
    f1 = (2 if a in ("relu", "sq") else 4) * n * H
    return p1 + H*k + k, f1 + 2*H*k

def finde_H(n, arm, ziel, idx):
    return min(range(2, 4096), key=lambda H: abs(kosten(n, H, arm)[idx] - ziel))

class Netz(nn.Module):
    def __init__(s, n, H, arm, seed, k=10):
        super().__init__()
        g = torch.Generator().manual_seed(seed * 7 + 1); a = basis(arm)
        s.l1 = {"relu": lambda: nn.Linear(n, H), "sq": lambda: Quad(n, H), "sqpaar": lambda: Paar(n, H),
                "kdicht": lambda: Born(n, H, "kdicht", g), "phase": lambda: Born(n, H, "phase", g),
                "zufall": lambda: Zufall(n, H, g)}[a]()
        s.relu = a == "relu"
        s.norm = nn.LayerNorm(H, elementwise_affine=False); s.l2 = nn.Linear(H, k)
    def forward(s, x):
        h = s.l1(x)
        if s.relu: h = F.relu(h)
        return s.l2(s.norm(h))

def trainiere(arm, H, lr, seed, tr, te, epochen=20, bs=256):
    x, y = tr; torch.manual_seed(seed)
    m = Netz(x.shape[1], H, arm, seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(y); sch = epochen * (n // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / sch)))
    g = torch.Generator().manual_seed(seed + 3)
    for _ in range(epochen):
        p = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            b = p[i:i + bs]
            l = F.cross_entropy(m(x[b]), y[b])
            if not torch.isfinite(l): return 0.1
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return (m(te[0]).argmax(1) == te[1]).float().mean().item()
