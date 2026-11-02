"""w6: adaptive Messung / frühes Aussteigen. Erkundung, Tuning-Seeds 950-952, 1 Thread."""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labor.gemeinsam import bilder

W, L, EP, BS = 128, 4, 25, 128
TAUS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]

class Exit(nn.Module):
    def __init__(s, d_in, widths, heads=True):
        super().__init__()
        ins = [d_in] + widths[:-1]
        s.ls = nn.ModuleList(nn.Linear(i, o) for i, o in zip(ins, widths))
        s.hs = nn.ModuleList(nn.Linear(o, 10) for o in widths)
    def forward(s, x):
        out = []
        for l, h in zip(s.ls, s.hs):
            x = F.relu(l(x)); out.append(h(x))
        return out
    def flops_bis(s, k, kopf_alle=True):
        # FLOPs (2*MAC) bis Lage k; Auslesen aller Köpfe 0..k (adaptiv) oder nur k (statisch)
        f = sum(2 * l.in_features * l.out_features for l in s.ls[:k + 1])
        hk = range(k + 1) if kopf_alle else [k]
        return f + sum(2 * s.hs[j].in_features * 10 for j in hk)

def train(m, xt, yt, seed, alle=True):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP * (len(yt) // BS))
    for _ in range(EP):
        p = torch.randperm(len(yt))
        for i in range(0, len(yt) - BS + 1, BS):
            b = p[i:i + BS]; o = m(xt[b])
            loss = sum(F.cross_entropy(z, yt[b]) for z in o) if alle else F.cross_entropy(o[-1], yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

@torch.no_grad()
def kurve(m, x, y, taus, tau_val=None):
    P = [F.softmax(z, 1) for z in m(x)]
    mx = torch.stack([p.max(1).values for p in P]); pr = torch.stack([p.argmax(1) for p in P])
    res = []
    for t in taus:
        ex = torch.full((len(y),), len(P) - 1)
        for k in reversed(range(len(P) - 1)):
            ex = torch.where(mx[k] >= t, torch.tensor(k), ex)
        acc = (pr[ex, torch.arange(len(y))] == y).float().mean().item()
        fl = np.mean([m.flops_bis(int(k)) for k in ex])
        res.append((t, acc, float(fl), float(np.mean(ex.numpy()) + 1)))
    return res

def lauf(name, seed):
    xt, yt, xte, yte = bilder(name, seed, 10_000, 5_000)
    xv, yv, xt, yt = xt[:2000], yt[:2000], xt[2000:], yt[2000:]
    out = {"adaptiv": None, "statisch": {}, "ref_lage": {}}
    m = Exit(256, [W] * L); train(m, xt, yt, seed)
    out["adaptiv"] = {"val": kurve(m, xv, yv, TAUS), "test": kurve(m, xte, yte, TAUS),
                      "params": sum(p.numel() for p in m.parameters())}
    # Referenz: derselbe Mehrkopf-Stamm, feste Lage k (Ausgang k, kein Aussteigen)
    P = [F.softmax(z, 1) for z in m(xte)]
    for k in range(L):
        out["ref_lage"][k] = ((P[k].argmax(1) == yte).float().mean().item(), m.flops_bis(k, False))
    # statisch gleich teuer: 1 und 2 Schichten, Breiten-Gitter, nur Endkopf
    for d, ws in [(1, [16, 32, 64, 128, 256, 512]), (2, [16, 32, 64, 128, 256]), (3, [32, 64, 128])]:
        for w in ws:
            s = Exit(256, [w] * d); train(s, xt, yt, seed + 7, alle=False)
            with torch.no_grad(): a = (s(xte)[-1].argmax(1) == yte).float().mean().item()
            out["statisch"][f"d{d}_w{w}"] = (a, s.flops_bis(d - 1, False), sum(p.numel() for p in s.parameters()) )
    return out

if __name__ == "__main__":
    name = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]]
    res = {}
    for s in seeds:
        t = time.time(); res[s] = lauf(name, s); print(name, s, f"{time.time()-t:.0f}s", flush=True)
    json.dump(res, open(Path(__file__).parent / f"w6_roh_{name}.json", "w"))
