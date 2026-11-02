"""r1: Hadamard+Niedrigbit bei GLEICHEN BYTES (Erkundung, Tuning-Seeds 970-972). MNIST-MLP.
Aufruf: OMP_NUM_THREADS=1 python3 r1_gleiche_bytes.py mnist
Arme: gross (h=512) 3/2 Bit roh+Clip, Hadamard+Clip, Zufalls-Orthogonale+Clip (Kontrolle);
Gegner bei gleichen Bytes: kleines dichtes fp16, Niedrigrang fp16 (trainiert, Breite 512)."""
import sys, json, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parents[1]))                 # .../INSTITUT/recherche -> w4 import
sys.path.insert(0, str(HIER.parents[2]))                 # .../INSTITUT
sys.path.insert(0, str(HIER.parent / "w4"))
torch.set_num_threads(1)
import w4_hadamard_quant as W4
SEEDS = (970, 971, 972)
H_GROSS = 512

class MLP(nn.Module):
    def __init__(self, h):
        super().__init__(); self.l1, self.l2, self.l3 = nn.Linear(784, h), nn.Linear(h, h), nn.Linear(h, 10)
    def forward(self, x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

class LR(nn.Module):                       # Niedrigrang: W = U V, ersetzt jede Linearschicht (Rang r)
    def __init__(self, i, o, r):
        super().__init__(); self.v = nn.Linear(i, r, bias=False); self.u = nn.Linear(r, o)
    def forward(self, x): return self.u(self.v(x))
class MLPLR(nn.Module):
    def __init__(self, h, r):
        super().__init__(); self.l1, self.l2, self.l3 = LR(784, h, r), LR(h, h, r), LR(h, 10, min(r, 10))
    def forward(self, x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

def bytes_fp16(m):                         # alle Gewichte+Bias in fp16 (wie w4: Bias 2 B)
    return sum(p.numel() for p in m.parameters()) * 2
def bytes_q(h, bits, rot):                 # wie QLinear.bytes
    tot = 0
    for i, o in ((784, h), (h, h), (h, 10)):
        ip = 1024 if (rot and i == 784) else i
        tot += o * ip * bits / 8 + o * 2 + o * 2 + (ip / 8 if rot else 0)
    return tot

def finde(bauer, ziel):                    # groesster Parameter p mit Bytes <= ziel
    p, best = 1, 1
    while bytes_fp16(bauer(p)) <= ziel: best = p; p += 1
    return best

def orth_matrix(n, seed):                  # Kontrolle: zufaellige dichte Orthogonale (Haar), gleiche Bytes angesetzt
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g)); return q * torch.sign(torch.diagonal(r))[None, :]

def quant_orth(net, bits, seed):
    qn = W4.quantisiere(net, bits, rot=True, clip=True, pad_to={784: 1024})
    k = 0
    for m in qn.modules():
        if isinstance(m, W4.QLinear):          # Q ersetzen und Gewichte neu quantisieren
            k += 1; lin = [c for c in net.modules() if isinstance(c, nn.Linear)][k - 1]
            Wt = lin.weight.data.clone()
            if m.pad: Wt = F.pad(Wt, (0, m.pad))
            m.Q = orth_matrix(Wt.shape[1], seed + k); m.W = W4.quant_rows(Wt @ m.Q.T, bits, True)
    return qn

def train(net, xtr, ytr, ep, seed, lr=2e-3):
    opt = torch.optim.Adam(net.parameters(), lr); g = torch.Generator().manual_seed(seed)
    n = ep * math.ceil(len(xtr) / 128); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n); 
    for _ in range(ep):
        p = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(p), 128):
            b = p[i:i + 128]; l = F.cross_entropy(net(xtr[b]), ytr[b]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    return net

def mnist(ep=10):
    from labor.gemeinsam import _roh
    x, y = _roh("mnist"); x = torch.tensor(x); y = torch.tensor(y)
    xtr, ytr, xva, yva, xte, yte = x[:55000], y[:55000], x[55000:60000], y[55000:60000], x[60000:], y[60000:]
    acc = lambda n, X, Y: float((n(X).argmax(1) == Y).float().mean())
    res = {}
    for seed in SEEDS:
        t0 = time.time(); r = {}
        torch.manual_seed(seed); gross = train(MLP(H_GROSS), xtr, ytr, ep, seed)
        r["gross_fp"] = {"val": acc(gross, xva, yva), "test": acc(gross, xte, yte), "bytes16": bytes_fp16(gross)}
        for bits in (3, 2):
            arme = {}
            for nm, rot, kind in (("roh_clip", False, "n"), ("had_clip", True, "n"), ("orth_clip", True, "o")):
                qn = quant_orth(gross, bits, 5000 + seed) if kind == "o" else W4.quantisiere(gross, bits, rot, True, None, pad_to={784: 1024})
                arme[nm] = {"val": acc(qn, xva, yva), "test": acc(qn, xte, yte), "bytes": bytes_q(H_GROSS, bits, rot)}
            ziel = bytes_q(H_GROSS, bits, True)         # Budget = rotiertes Modell (hoechste Bytes der Quant-Arme)
            h = finde(lambda p: MLP(p), ziel)
            torch.manual_seed(seed); kl = train(MLP(h), xtr, ytr, ep, seed)
            arme["klein_dicht_fp16"] = {"h": h, "val": acc(kl, xva, yva), "test": acc(kl, xte, yte), "bytes": bytes_fp16(kl)}
            rk = finde(lambda p: MLPLR(H_GROSS, p), ziel)
            torch.manual_seed(seed); lr_ = train(MLPLR(H_GROSS, rk), xtr, ytr, ep, seed)
            arme["niedrigrang_fp16"] = {"rang": rk, "val": acc(lr_, xva, yva), "test": acc(lr_, xte, yte), "bytes": bytes_fp16(lr_)}
            r[f"b{bits}"] = arme; r[f"b{bits}"]["ziel_bytes"] = ziel
        r["zeit"] = time.time() - t0; res[seed] = r
        print(seed, {k: (round(v["test"], 4) if isinstance(v, dict) and "test" in v else None) for k, v in r.items()}, round(r["zeit"]), flush=True)
        for b in ("b3", "b2"): print(b, {k: (round(v["test"], 4), v.get("h", v.get("rang"))) for k, v in r[b].items() if isinstance(v, dict)}, flush=True)
        json.dump(res, open(HIER / "r1_mnist.json", "w"), indent=1)

if __name__ == "__main__":
    mnist()
