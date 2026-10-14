"""v3: Kontrollen K1-K3 als Erkundung. Gleicher Kopf (kopf_logistisch, l2=3e-3) fuer alle U-Arme."""
import os, sys, json, math, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT",2)))
sys.path.insert(0, "/home/user/QIRE/INSTITUT")
from abteilungen.clifford import daten, QireZelle
from abteilungen.clifford_suche import kopf_logistisch, normiere, unitaer_clifford

OUT = "/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v3/"
N = 8; D = 256; L2 = 3e-3
STEPS = int(os.environ.get("STEPS", 500))

def walsh(n=N):
    H = torch.ones(1, 1)
    for _ in range(n):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(2 ** n)

def haar(g):
    A = torch.randn(D, D, generator=g)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))

def feat(xn, U, nl="sq"):
    z = xn @ U.T
    return z ** 2 if nl == "sq" else (z.abs() if nl == "abs" else z)

def ev(ftr, ytr, fte, yte):
    return kopf_logistisch(ftr, ytr, fte, yte, l2=L2)[0]

# ---------- lernbare Zellen: liefern U (D x D) oder wirken direkt auf x
class Cayley(torch.nn.Module):
    def __init__(s, init=0.0, g=None):
        super().__init__()
        s.A = torch.nn.Parameter(init * torch.randn(D, D, generator=g))
        s.I = torch.eye(D)
    def U(s):
        S = s.A - s.A.T
        return torch.linalg.solve(s.I + S, s.I - S)       # Cayley: orthogonal

class Butterfly(torch.nn.Module):
    """8 Stufen, je 128 Givens-Winkel (Paare i, i^2^k): 1024 Winkel."""
    def __init__(s, init, g):
        super().__init__()
        s.th = torch.nn.Parameter(init * (torch.rand(8, 128, generator=g) * 2 - 1))
        s.lo, s.hi = [], []
        for k in range(8):
            idx = torch.arange(D); lo = idx[(idx >> k) & 1 == 0]
            s.lo.append(lo); s.hi.append(lo | (1 << k))
        s.perm = [torch.cat([s.lo[k], s.hi[k]]).argsort() for k in range(8)]
    def apply(s, x):
        if not hasattr(s, "P"):
            idx = torch.arange(D)
            s.P = [torch.eye(D)[:, idx ^ (1 << k)] for k in range(8)]   # (x P)_j = x_{j^bit}
            s.sgn = [torch.where((idx >> k) & 1 == 0, -1.0, 1.0) for k in range(8)]
        for k in range(8):
            c = torch.cos(s.th[k]); sn = torch.sin(s.th[k])
            cc = torch.zeros(D).index_put((s.lo[k],), c).index_put((s.hi[k],), c) if False else torch.cat([c, c])[torch.cat([s.lo[k], s.hi[k]]).argsort()]
            ss = torch.cat([sn, sn])[torch.cat([s.lo[k], s.hi[k]]).argsort()] * s.sgn[k]
            x = x * cc[None, :] + (x @ s.P[k]) * ss[None, :]
        return x

class Kopfmodell(torch.nn.Module):
    def __init__(s, zelle, kind):
        super().__init__(); s.z, s.kind = zelle, kind
        s.kopf = torch.nn.Linear(D, 10)
    def merk(s, xn):
        if s.kind == "cayley": return (xn @ s.z.U().T) ** 2
        if s.kind == "butterfly": return (xn @ s.z.apply(torch.eye(D))) ** 2
        if s.kind == "qire": return (xn @ s.z.unitaer().T) ** 2
    def forward(s, xn):
        f = s.merk(xn)
        return s.kopf((f - 1.0 / D) * 16)       # grobe feste Skalierung (Merkmale ~1/256)

def trainiere_u(kind, xn, y, lr, init, seed, steps=STEPS, xv=None, yv=None):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    if kind == "cayley": z = Cayley(init, g)
    elif kind == "butterfly": z = Butterfly(init, g)
    elif kind == "qire": z = QireZelle(n=N, lagen=4, cnot=True)
    elif kind == "qire_nocnot": z = QireZelle(n=N, lagen=4, cnot=False)
    kk = "qire" if kind.startswith("qire") else kind
    m = Kopfmodell(z, kk)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(xn), y).backward()
        opt.step()
    with torch.no_grad():
        return m, (m.merk(xn))

def u_val(m, kk, x):
    with torch.no_grad(): return m.merk(x)

def mlp(xtr, ytr, xte, yte, lr, seed, h=9, epochs=60, wd=1e-4):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-3)
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(D, h), torch.nn.ReLU(), torch.nn.Linear(h, 10))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        p = torch.randperm(len(ytr))
        for i in range(0, len(p), 128):
            j = p[i:i + 128]; opt.zero_grad()
            torch.nn.functional.cross_entropy(net(a[j]), ytr[j]).backward(); opt.step()
    with torch.no_grad(): return float((net(b).argmax(1) == yte).float().mean())

def nparam_mlp(h): return D * h + h + h * 10 + 10

# ---------- Tuning (Seed 959, Validierungsanteil 1000 aus den 6000 Trainingsbildern)
GRID = {"cayley": [(0.005, 0.0), (0.02, 0.0)],
        "butterfly": [(0.03, 0.3), (0.1, math.pi)],
        "qire": [(0.02, 0), (0.08, 0)], "qire_nocnot": [(0.02, 0), (0.08, 0)]}

def tune(ds):
    xtr, ytr, _, _ = daten(ds, 959)
    xn = normiere(xtr); a, b, ya, yb = xn[:5000], xn[5000:], ytr[:5000], ytr[5000:]
    best = {}
    for kind, gr in GRID.items():
        res = []
        for lr, init in gr:
            m, fa = trainiere_u(kind, a, ya, lr, init, 959)
            fb = u_val(m, kind, b)
            res.append((ev(fa, ya, fb, yb), lr, init))
        best[kind] = max(res)[1:]; print(ds, "tune", kind, sorted(res, reverse=True), flush=True)
    res = []
    for lr in (0.001, 0.003, 0.01, 0.03):
        res.append((mlp(xtr[:5000], ytr[:5000], xtr[5000:], ytr[5000:], lr, 959), lr))
    best["mlp"] = max(res)[1]; print(ds, "tune mlp", res, flush=True)
    # l2 des Pixel-Softmax (fairness)
    res = []
    for l2 in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        res.append((kopf_logistisch(xtr[:5000], ytr[:5000], xtr[5000:], ytr[5000:], l2=l2)[0], l2))
    best["pix_l2"] = max(res)[1]; print(ds, "tune pixl2", res, flush=True)
    return best

def messe(ds, seed, hp):
    xtr, ytr, xte, yte = daten(ds, seed)
    ntr, nte = normiere(xtr), normiere(xte)
    g = torch.Generator().manual_seed(seed + 7000)
    r = {"seed": seed, "ds": ds}
    t0 = time.time()
    # Referenzen ohne Quadrieren
    r["pix_softmax"] = kopf_logistisch(xtr, ytr, xte, yte, l2=L2)[0]
    r["pix_softmax_l2opt"] = kopf_logistisch(xtr, ytr, xte, yte, l2=hp["pix_l2"])[0]
    r["pix_norm_lin"] = ev(ntr, ytr, nte, yte)
    r["mlp9"] = mlp(xtr, ytr, xte, yte, hp["mlp"], seed, h=9)
    r["mlp10"] = mlp(xtr, ytr, xte, yte, hp["mlp"], seed, h=10)
    # feste U
    I = torch.eye(D); H = walsh()
    r["ident_sq"] = ev(feat(ntr, I), ytr, feat(nte, I), yte)
    r["ident_abs"] = ev(feat(ntr, I, "abs"), ytr, feat(nte, I, "abs"), yte)
    Ur = haar(g)
    for nl in ("sq", "abs", "lin"):
        r["haar_" + nl] = ev(feat(ntr, Ur, nl), ytr, feat(nte, Ur, nl), yte)
    # Haar: mehrere Ziehungen (Streuung innerhalb Seed klein, 3 Ziehungen)
    h3 = [r["haar_sq"]]
    for _ in range(2):
        U2 = haar(g); h3.append(ev(feat(ntr, U2), ytr, feat(nte, U2), yte))
    r["haar_sq_3"] = h3
    # Fastfood: H D ; H D2 H D1 ; (und H D P H D mit Permutation: 2 Stufen mit Perm)
    sg = lambda: (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    U1 = H * sg()[None, :]
    r["ff1"] = ev(feat(ntr, U1), ytr, feat(nte, U1), yte)
    U2 = H @ torch.diag(sg()) @ H @ torch.diag(sg())
    r["ff2"] = ev(feat(ntr, U2), ytr, feat(nte, U2), yte)
    U3 = H @ torch.diag(sg()) @ H @ torch.diag(sg()) @ H @ torch.diag(sg())
    r["ff3"] = ev(feat(ntr, U3), ytr, feat(nte, U3), yte)
    r["walsh_nosign"] = ev(feat(ntr, H), ytr, feat(nte, H), yte)
    # zufaellige Clifford (Codes mod 4 -> 1 Bit kanonisch; 20 Ziehungen, Median)
    cl = []
    for _ in range(20):
        codes = torch.randint(0, 4, (4, N), generator=g).numpy()
        Uc = unitaer_clifford(codes, N)
        cl.append(ev(feat(ntr, Uc), ytr, feat(nte, Uc), yte))
    r["clifford_zufall_liste"] = cl; r["clifford_zufall_median"] = float(np.median(cl))
    # zufaellige (ungelernte) QIRE-Winkel mit/ohne CNOT
    for cn in (True, False):
        zz = QireZelle(n=N, lagen=4, cnot=cn)
        with torch.no_grad(): Uq = zz.unitaer()
        key = "qire_zufallswinkel" + ("" if cn else "_nocnot")
        r[key] = ev(feat(ntr, Uq), ytr, feat(nte, Uq), yte)
    # gelernt
    for kind in ("cayley", "butterfly", "qire", "qire_nocnot"):
        lr, init = hp[kind]
        m, ftr = trainiere_u(kind, ntr, ytr, lr, init, seed)
        fte = u_val(m, kind, nte)
        r[kind] = ev(ftr, ytr, fte, yte)
        with torch.no_grad():
            r[kind + "_eigener_kopf"] = float((m(nte).argmax(1) == yte).float().mean())
        # Winkel-Gitter-Abstand fuer QIRE (ist die gelernte Zelle nahe Clifford?)
        if kind.startswith("qire"):
            th = (m.z.theta.detach() % (math.pi / 2)); th = torch.minimum(th, math.pi / 2 - th)
            r[kind + "_clifford_abstand_mean"] = float(th.mean())
    r["sekunden"] = time.time() - t0
    return r

if __name__ == "__main__":
    ds = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
    hpf = OUT + f"hp_{ds}.json"
    if os.path.exists(hpf): hp = json.load(open(hpf))
    else:
        hp = tune(ds); json.dump(hp, open(hpf, "w"))
    for s in seeds:
        r = messe(ds, s, hp)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if "liste" not in k}), flush=True)
        with open(OUT + f"roh_{ds}.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
