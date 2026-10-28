"""K11 Fliegen-Hash-Kopf (Erkundung, kein Befund).
Mechanismus: feste Hadamard-Expansion auf m=8192 Merkmale (m/n Blöcke H·D_b·x mit festen Zufallsvorzeichen D_b), nur die k=64 größten Werte bleiben,
nur diese gehen in die gelernte lineare Auslese (m x 10, Einträge nur für die k aktiven). Feste Eingabemerkmale: (a) Pixel 16x16 (n=256),
(b) eingefrorene Merkmale eines kleinen CNN (n=64, 3 Epochen auf den Trainingsbildern der Zelle). MNIST und Fashion, n_train = 1000 und 8000 (+2000 Validierung für die Lernrate), Test 5000.
Arme: fly_top (Top-k der Werte) | fly_zufallsspalten (k feste zufällige Spalten, nicht datenabhängig) | fly_zufallsprobe (k zufällige Spalten je Bild, fest) |
      dicht_flopgleich (n -> h -> 10, ReLU, h so, dass FLOPs = FLOPs Fly) | dicht_paramgleich (h so, dass gelernte Parameter ~ Fly-Auslese m*10) |
      fly_top_k16 / fly_top_k256 (Nebenarme: andere k).
FLOPs je Bild (Kopf ab den festen Merkmalen): Fly = m*(log2 n + 1) (schnelle Walsh-Transformation + Vorzeichen) + m (Auswahl) + 2*k*10; dicht = 2*(n+10)*h.
Lernrate je Arm auf der Validierung aus (3e-4,1e-3,3e-3,1e-2,3e-2); 30 Epochen Adam, Batch 128.
Aufruf: python3 erkundung/K11.py --seed 950 [--schnell]   ->   erkundung/roh/K11_s<seed>.json"""
import math
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bytelm import cli, speichere, speichere_teil, lade_teil, INSTITUT  # noqa: E402
from labor.gemeinsam import bilder  # noqa: E402

KID = "K11"
M, K = 8192, 64
LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def walsh(n):
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def expansion(x, seed):
    """x (B,n) -> (B,M) feste Merkmale: M/n Blöcke H (D_b x) / sqrt(n)."""
    n = x.shape[1]
    g = torch.Generator().manual_seed(seed + 4242)
    D = torch.randint(0, 2, (M // n, n), generator=g).float() * 2 - 1
    H = walsh(n) / math.sqrt(n)
    return ((x[:, None, :] * D) @ H).reshape(x.shape[0], M)


def fly_idx(X, art, k, seed, chunk=500):
    """Rückgabe (idx int64 (N,k), val (N,k)) für die Auswahlart."""
    g = torch.Generator().manual_seed(seed + 777)
    fest = torch.randperm(M, generator=g)[:k]
    I, V = [], []
    for i in range(0, len(X), chunk):
        z = expansion(X[i:i + chunk], seed)
        if art == "top":
            v, ix = z.topk(k, dim=1)
        elif art == "festspalten":
            ix = fest.expand(len(z), k)
            v = z.gather(1, ix)
        else:                                       # zufallsprobe: k zufällige Spalten je Bild (fest, nicht vom Wert abhängig)
            ix = torch.stack([torch.randperm(M, generator=g)[:k] for _ in range(len(z))])
            v = z.gather(1, ix)
        I.append(ix)
        V.append(v)
    return torch.cat(I), torch.cat(V)


def cnn_merkmale(xtr, ytr, xs, seed, epochen=3):
    """Kleines CNN (conv16-pool-conv32-pool-lin64) auf den Trainingsbildern trainieren, 64-d-Merkmale aller Mengen zurück (eingefroren)."""
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128, 64), nn.ReLU())
    net = nn.Sequential(f, nn.Linear(64, 10))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(epochen):
        p = torch.randperm(len(xtr))
        for i in range(0, len(p), 64):
            b = p[i:i + 64]
            opt.zero_grad()
            F.cross_entropy(net(xtr[b].reshape(-1, 1, 16, 16)), ytr[b]).backward()
            opt.step()
    with torch.no_grad():
        return [f(x.reshape(-1, 1, 16, 16)) for x in xs]


def trainiere_kopf(modell, daten_tr, daten_va, daten_te, lr, ep, seed):
    """daten_* = (eingabe, y); eingabe = Tensor (dicht) oder (idx,val) (Fly). modell(eingabe_batch) -> Logits."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(modell.parameters(), lr)
    (a, y) = daten_tr
    n = len(y)
    sel = lambda e, b: (e[0][b], e[1][b]) if isinstance(e, tuple) else e[b]
    T = ep * math.ceil(n / 128)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T)
    for _ in range(ep):
        p = torch.randperm(n)
        for i in range(0, n, 128):
            b = p[i:i + 128]
            l = F.cross_entropy(modell(sel(a, b)), y[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        acc = lambda d: float((modell(d[0]).argmax(1) == d[1]).float().mean())
        return acc(daten_va), acc(daten_te)


class FlyAuslese(nn.Module):
    def __init__(self):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.zeros(M, 10)), nn.Parameter(torch.zeros(10))

    def forward(self, e):
        idx, val = e
        return F.embedding_bag(idx, self.w, per_sample_weights=val, mode="sum") + self.b


def dicht(n, h):
    return nn.Sequential(nn.Linear(n, h), nn.ReLU(), nn.Linear(h, 10))


def zelle(name, n_train, seed, ep, merkmal):
    xtr, ytr, xte, yte = bilder(name, seed, n_train=n_train + 2000, n_test=5000)
    xva, yva, xtr, ytr = xtr[n_train:], ytr[n_train:], xtr[:n_train], ytr[:n_train]
    mu = xtr.mean(0)
    xs = [x - mu for x in (xtr, xva, xte)]
    if merkmal == "cnn":
        xs = cnn_merkmale(xtr, ytr, [xtr, xva, xte], seed)
        mu2, s = xs[0].mean(0), xs[0].std() + 1e-6
        xs = [(x - mu2) / s for x in xs]
    n = xs[0].shape[1]
    fl_fly = lambda k: M * (math.log2(n) + 1) + M + 2 * k * 10
    h_fl = lambda k: max(2, round(fl_fly(k) / (2 * (n + 10)) - 0.5))
    h_par = max(2, round((M * 10) / (n + 11 + 1)))
    ys = (ytr, yva, yte)
    arme = {}
    for art, k, nm in (("top", K, "fly_top"), ("festspalten", K, "fly_zufallsspalten"), ("zufallsprobe", K, "fly_zufallsprobe"),
                       ("top", 16, "fly_top_k16"), ("top", 256, "fly_top_k256")):
        sets = [fly_idx(x, art, k, seed) for x in xs]
        arme[nm] = (lambda: FlyAuslese(), [(s, y) for s, y in zip(sets, ys)], {"flops": fl_fly(k), "param": M * 10 + 10, "aktive_param": k * 10 + 10})
    h = h_fl(K)
    arme["dicht_flopgleich"] = (lambda h=h: dicht(n, h), [(x, y) for x, y in zip(xs, ys)], {"flops": 2 * (n + 10) * h, "param": (n + 1) * h + (h + 1) * 10, "h": h})
    arme["dicht_paramgleich"] = (lambda h=h_par: dicht(n, h_par), [(x, y) for x, y in zip(xs, ys)], {"flops": 2 * (n + 10) * h_par, "param": (n + 1) * h_par + (h_par + 1) * 10, "h": h_par})
    out = {}
    for nm, (mk, ds, info) in arme.items():
        res = {}
        for lr in LRS:
            va, te = trainiere_kopf(mk(), ds[0], ds[1], ds[2], lr, ep, seed)
            res[str(lr)] = (va, te)
        best = max(res, key=lambda r: res[r][0])
        out[nm] = {"lr": float(best), "acc_val": res[best][0], "acc_test": res[best][1], "alle_lr_val": {r: v[0] for r, v in res.items()}, **info}
    return out


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    teil = lade_teil(KID, seed, schnell)
    plan = [(ds, nt, mm) for ds in ("mnist", "fashion_mnist") for mm in ("pixel", "cnn") for nt in ((1000,) if schnell else (1000, 8000))]
    if schnell:
        plan = plan[:2]
    ep = 4 if schnell else 30
    for ds, nt, mm in plan:
        key = f"{ds}/{mm}/n{nt}"
        if key in teil:
            continue
        print(f"[K11 s{seed}] {key} ...", flush=True)
        teil[key] = zelle(ds, nt, seed, ep, mm)
        for a, r in teil[key].items():
            print(f"   {a:20s} val={r['acc_val']:.4f} test={r['acc_test']:.4f} lr={r['lr']} flops={r['flops']:.0f}", flush=True)
        speichere_teil(KID, seed, schnell, teil)
    out = {"kid": KID, "seed": seed, "schnell": schnell, "beschreibung": __doc__, "zellen": teil, "sekunden_gesamt_dieser_aufruf": time.time() - t0}
    speichere(KID, seed, schnell, out)
    return out


if __name__ == "__main__":
    cli(KID, main)
