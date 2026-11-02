"""w4: Hadamard-Rotation vor Niedrigbit-Quantisierung (Erkundung, Tuning-Seeds 930-934).
Aufruf: OMP_NUM_THREADS=1 python3 w4_hadamard_quant.py sprache|mnist
Rotation: y = W x = (W Q^T)(Q x), Q = H*D/sqrt(n) (D = zufaellige Vorzeichen), Gewichte werden nach der Rotation je Zeile quantisiert."""
import sys, math, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
WURZEL = Path(__file__).resolve().parents[3]          # .../INSTITUT
sys.path.insert(0, str(WURZEL))
torch.set_num_threads(1)
SEEDS = (930, 931, 932)
BITS = (8, 4, 3, 2)
OUT = Path(__file__).parent

def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

def rot_matrix(n, seed):
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    return hadamard(n) * s[None, :]                    # H D ; orthogonal

def quant_rows(W, bits, clip=False):
    """symmetrisch, pro Zeile, Rundung zum Naechsten; clip: Skala per MSE-Gittersuche."""
    qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bs = None, None
    for f in ((1.0,) if not clip else tuple(np.linspace(0.4, 1.0, 13))):
        s = amax * f / qmax
        Wq = (W / s).round().clamp(-qmax, qmax) * s
        e = ((W - Wq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs = e, Wq
        else:
            m = e < best
            best = torch.where(m, e, best); bs = torch.where(m, Wq, bs)
    return bs

def fq_act(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s

class QLinear(nn.Module):
    """Linear mit nachtraeglich quantisierten Gewichten (simuliert), optional Rotation und Aktivierungsquantisierung."""
    def __init__(self, lin, bits, rot, clip=False, abits=None, seed=0, pad_to=None):
        super().__init__()
        W = lin.weight.data.clone()
        n = W.shape[1]; self.n = n; self.pad = (pad_to or n) - n
        if self.pad: W = F.pad(W, (0, self.pad))
        self.Q = rot_matrix(W.shape[1], seed) if rot else None
        Wr = W @ self.Q.T if rot else W
        self.W = quant_rows(Wr, bits, clip) if bits else Wr
        self.b = lin.bias.data.clone() if lin.bias is not None else None
        self.abits = abits
        self.bits = bits or 16
        self.shape = tuple(W.shape)
    def forward(self, x):
        if self.pad: x = F.pad(x, (0, self.pad))
        if self.Q is not None: x = x @ self.Q.T
        if self.abits: x = fq_act(x, self.abits)
        return F.linear(x, self.W, self.b)
    def bytes(self):
        o, i = self.shape
        return o * i * self.bits / 8 + o * 2 + (i / 8 if self.Q is not None else 0) + (o * 2 if self.b is not None else 0)
    def flops(self):    # MAC integer + Rotation (n log2 n Additionen)
        o, i = self.shape
        return o * i, (i * int(math.log2(i)) if self.Q is not None else 0)

def quantisiere(netz, bits, rot, clip=False, abits=None, pad_to=None):
    netz = copy.deepcopy(netz)
    k = [0]
    def ersetze(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, QLinear(c, bits, rot, clip, abits, seed=1000 + k[0], pad_to=(pad_to or {}).get(c.in_features)))
            else:
                ersetze(c)
    ersetze(netz); return netz

def erlaubt(bits, clip, ab):
    if ab and clip: return False
    if ab and bits < 4: return False
    if clip and bits == 8: return False
    return True

def stat(netz):
    q = [m for m in netz.modules() if isinstance(m, QLinear)]
    return sum(m.bytes() for m in q), sum(m.flops()[0] for m in q), sum(m.flops()[1] for m in q)

# ------------------------------------------------------------ Sprache
def sprache():
    from abteilungen import struktur_sprache as S
    tr, va, te = S.daten(False)
    te100 = te[:100_000]
    D = 128; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2500; BS = 8; LR = 3e-3
    res = {}
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        netz = S.Sprachmodell("D", seed, d=D)
        opt = torch.optim.Adam(netz.parameters(), lr=LR)
        warm = STEPS // 20
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (STEPS - warm))))
        g = torch.Generator().manual_seed(seed + 5); offs = torch.arange(S.KONTEXT)
        netz.train()
        for _ in range(STEPS):
            i = torch.randint(0, len(tr) - S.KONTEXT - 1, (BS,), generator=g)
            l = F.cross_entropy(netz(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0); opt.step(); sch.step()
        r = {"train_s": time.time() - t0, "fp_test": S.bpc_auswerten(netz, te100)}
        nlin = sum(p.numel() for n_, p in netz.bl.named_parameters() if n_.endswith("weight") and p.dim() == 2)
        r["params_total"] = S.param_gesamt(netz); r["params_linear"] = nlin
        r["fp16_bytes_linear"] = nlin * 2
        # Gewichts-Ausreisser: Kurtosis der Zeilen vor/nach Rotation (Mittel ueber Matrizen)
        ku = {"roh": [], "rot": []}
        for m in netz.modules():
            if isinstance(m, nn.Linear):
                W = m.weight.data; Wr = W @ rot_matrix(W.shape[1], 5).T
                for nm, X in (("roh", W), ("rot", Wr)):
                    z = (X - X.mean(1, keepdim=True)) / X.std(1, keepdim=True); ku[nm].append(float((z ** 4).mean(1).mean()))
        r["kurt_roh"], r["kurt_rot"] = float(np.mean(ku["roh"])), float(np.mean(ku["rot"]))
        for bits in BITS:
            for rot in (False, True):
                for clip in (False, True):
                    for ab in (None, 8):
                        if not erlaubt(bits, clip, ab): continue
                        qn = quantisiere(netz, bits, rot, clip, ab)
                        key = f"b{bits}_{'rot' if rot else 'roh'}_{'clip' if clip else 'rtn'}_{'A8' if ab else 'Afp'}"
                        r[key] = {"test": S.bpc_auswerten(qn, te100)}
                        r[key]["bytes"], r[key]["mac"], r[key]["add"] = stat(qn)
        res[seed] = r
        print(seed, "fp", round(r["fp_test"], 3), "t", round(r["train_s"]), flush=True)
        json.dump(res, open(OUT / "w4_sprache.json", "w"), indent=1)

# ------------------------------------------------------------ MNIST
class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(784, h), nn.Linear(h, h), nn.Linear(h, 10)
    def forward(self, x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

def mnist():
    from labor.gemeinsam import _roh
    x, y = _roh("mnist"); x = torch.tensor(x); y = torch.tensor(y)
    xtr, ytr, xte, yte = x[:55000], y[:55000], x[60000:], y[60000:]
    xva, yva = x[55000:60000], y[55000:60000]
    acc = lambda n, X, Y: float((n(X).argmax(1) == Y).float().mean())
    res = {}
    for seed in SEEDS:
        torch.manual_seed(seed); net = MLP(); opt = torch.optim.Adam(net.parameters(), 2e-3)
        g = torch.Generator().manual_seed(seed)
        for ep in range(8):
            p = torch.randperm(len(xtr), generator=g)
            for i in range(0, len(p), 128):
                b = p[i:i + 128]; l = F.cross_entropy(net(xtr[b]), ytr[b]); opt.zero_grad(); l.backward(); opt.step()
        r = {"fp_val": acc(net, xva, yva), "fp_test": acc(net, xte, yte), "params": sum(p.numel() for p in net.parameters())}
        for bits in BITS:
            for rot in (False, True):
                for clip in (False, True):
                    for ab in (None, 8):
                        if not erlaubt(bits, clip, ab): continue
                        qn = quantisiere(net, bits, rot, clip, ab, pad_to={784: 1024})
                        key = f"b{bits}_{'rot' if rot else 'roh'}_{'clip' if clip else 'rtn'}_{'A8' if ab else 'Afp'}"
                        r[key] = {"val": acc(qn, xva, yva), "test": acc(qn, xte, yte)}
                        r[key]["bytes"], r[key]["mac"], r[key]["add"] = stat(qn)
        res[seed] = r; print(seed, r["fp_test"], flush=True)
        json.dump(res, open(OUT / "w4_mnist.json", "w"), indent=1)

if __name__ == "__main__":
    {"sprache": sprache, "mnist": mnist}[sys.argv[1]]()
