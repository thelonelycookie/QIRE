"""w5: Nachtraegliche Kompression nach Verschraenkungsentropie (Erkundung, Tuning-Seeds 940-944).
MNIST 16x16, MLP 256-512-256-128-10; komprimiert werden die drei ersten Matrizen.
Methoden: gleichmaessiger Rang, Entropie (Perplexitaet), Wasserfuellen (Schwelle auf normiertem
Schmidt-Spektrum), globale Singulaerwert-Schwelle (Kontrolle), TT-SVD (4 Kerne), Magnitudenpruning
(Index-Bits mitgezaehlt), plus Kontrolle 'scratch' (Niedrigrang von Anfang an trainiert).
Aufruf: OMP_NUM_THREADS=1 python3 w5_kompression.py [--schnell]"""
import sys, json, time, math
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parents[2]))
from labor import gemeinsam as G

DIMS = [256, 512, 256, 128, 10]
KOMP = [0, 1, 2]                      # komprimierte Schichten
FAKT = {256: (4, 4, 4, 4), 512: (4, 4, 4, 8), 128: (4, 4, 4, 2)}
SEEDS = [940, 941, 942, 943, 944]
FRAKT = [0.05, 0.1, 0.2]              # Anteil der dichten Parameter der komprimierten Matrizen
EP, EPFT = 25, 4
SCHNELL = "--schnell" in sys.argv
if SCHNELL: SEEDS, EP, EPFT = SEEDS[:1], 3, 1

def mlp(): return torch.nn.ModuleList([torch.nn.Linear(DIMS[i], DIMS[i+1]) for i in range(4)])
def vor(layers, x, Ws):
    for i, l in enumerate(layers):
        W = Ws[i] if Ws[i] is not None else l.weight
        x = x @ W.T + l.bias
        if i < 3: x = torch.relu(x)
    return x
def trainiere(layers, Wfn, params, xtr, ytr, ep, lr, seed, mask=None):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(params, lr=lr)
    n = len(ytr)
    for e in range(ep):
        p = torch.randperm(n, generator=g)
        for k in range(0, n, 128):
            b = p[k:k+128]
            loss = torch.nn.functional.cross_entropy(vor(layers, xtr[b], Wfn()), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
def acc(layers, Ws, x, y):
    with torch.no_grad(): return G.genauigkeit(vor(layers, x, Ws), y)

# ---------- Schmidt-Spektrum
def spektrum(W):
    s = torch.linalg.svdvals(W); p = s**2/(s**2).sum()
    S = -(p[p > 0]*torch.log(p[p > 0])).sum().item()
    return s, p, S
def rang_grenze(m, n): return min(m, n)
def kosten(r, m, n): return r*(m+n)

# ---------- Rangverteilungen (Budget in Parametern)
def alloc_gleich(Wl, B):
    c = B/sum(m+n for m, n in (W.shape for W in Wl)); return [max(1, int(c)) for W in Wl]
def bisekt(f, lo, hi, B, it=60):                # f(c) -> Ränge; Kosten monoton in c
    for _ in range(it):
        mid = (lo+hi)/2
        if sum(kosten(r, *W.shape) for r, W in zip(f(mid), Wl_glob)) > B: hi = mid
        else: lo = mid
    return f(lo)
def mk_alloc(Wl, B, art):
    global Wl_glob; Wl_glob = Wl
    sp = [spektrum(W) for W in Wl]
    if art == "gleich":
        return alloc_gleich(Wl, B)
    if art == "entropie":                         # r ~ Perplexitaet exp(S)
        pe = [math.exp(S) for _, _, S in sp]
        return bisekt(lambda c: [int(max(1, min(min(W.shape), round(c*q)))) for q, W in zip(pe, Wl)], 0, 1e4, B)
    if art == "wasser":                           # Schwelle auf normiertem Spektrum p_k
        return bisekt(lambda lam: [int(max(1, (p > lam).sum())) for _, p, _ in sp], 1e-9, 1, B, 80)[:] if False else _wasser(sp, Wl, B)
    if art == "global_sv":                        # absolute Schwelle auf s_k^2
        return _wasser(sp, Wl, B, absolut=True)
def _wasser(sp, Wl, B, absolut=False):
    lo, hi = 0.0, 1.0 if not absolut else max(float((s**2).max()) for s, _, _ in sp)
    best = [1]*len(Wl)
    for _ in range(80):
        lam = (lo+hi)/2
        r = [int(max(1, ((s**2 if absolut else p) > lam).sum())) for s, p, _ in sp]
        if sum(kosten(a, *W.shape) for a, W in zip(r, Wl)) > B: lo = lam
        else: hi = lam; best = r
    return best

def lowrank_init(W, r):
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r]*s[:r].sqrt()).clone(), (s[:r].sqrt()[:, None]*Vh[:r]).clone()

# ---------- TT-Matrix
def tt_svd(W, fm, fn, R):
    m, n = W.shape; d = len(fm)
    T = W.reshape(*fm, *fn)
    perm = [x for k in range(d) for x in (k, d+k)]
    T = T.permute(*perm).contiguous()            # (m1,n1,m2,n2,...)
    cores, r0 = [], 1
    C = T.reshape(1, -1)
    for k in range(d-1):
        C = C.reshape(r0*fm[k]*fn[k], -1)
        U, s, Vh = torch.linalg.svd(C, full_matrices=False)
        r = min(R, len(s))
        cores.append(U[:, :r].reshape(r0, fm[k], fn[k], r).clone())
        C = s[:r, None]*Vh[:r]; r0 = r
    cores.append(C.reshape(r0, fm[-1], fn[-1], 1).clone())
    return cores
def tt_voll(cores, fm, fn):
    d = len(cores); X = cores[0]                 # (1,m1,n1,r)
    for c in cores[1:]:
        X = torch.einsum("amnb,bpqc->ampnqc", X, c).reshape(1, X.shape[1]*c.shape[1], X.shape[2]*c.shape[2], c.shape[3])
    # Ordnung der Indizes: (m1 m2.., n1 n2..) mit m1 am groebsten -> passt zu reshape(*fm,*fn)
    return X[0, :, :, 0]
def tt_voll_korrekt(cores, fm, fn):
    X = cores[0][0]                              # (m1,n1,r)
    for c in cores[1:]:
        X = torch.einsum("mnb,bpqc->mpnqc", X, c)
        s = X.shape; X = X.reshape(s[0]*s[1], s[2]*s[3], s[4])
    X = X[:, :, 0]                               # (m1m2.., n1n2..) mit m1 groebstem Index
    return X
def tt_par(cores): return sum(c.numel() for c in cores)
def tt_flops(cores):
    # Anwenden auf Vektor Kern fuer Kern (obere Schranke, Standardkontraktion)
    d = len(cores); tot = 0
    fm = [c.shape[1] for c in cores]; fn = [c.shape[2] for c in cores]
    for k, c in enumerate(cores):
        rest = math.prod(fn[k+1:]) * math.prod(fm[:k])
        tot += c.numel()*rest
    return tot
def tt_waehle(W, B):
    fm = FAKT[W.shape[0]]; fn = FAKT[W.shape[1]]
    best = None
    for R in range(1, 200):
        cs = tt_svd(W, fm, fn, R)
        if tt_par(cs) > B: break
        best = cs
    return best, fm, fn
def tt_budget_teilen(Wl, B):
    # TT: gleiche Fraktion je Matrix (gleichmaessig nach Parametern, wie 'gleich')
    tot = sum(W.numel() for W in Wl)
    return [B*W.numel()/tot for W in Wl]

# ---------- Pruning
def prune(Wl, B_bytes, lokal):
    def ib(W): return math.ceil(math.log2(W.numel()))
    if lokal:
        tot = sum(W.numel() for W in Wl); out = []
        for W in Wl:
            b = B_bytes*W.numel()/tot; k = int(b/(4+ib(W)/8)); out.append(k)
        ks = out
        masks = []
        for W, k in zip(Wl, ks):
            t = W.abs().flatten().topk(max(1, k)).values[-1]; masks.append((W.abs() >= t).float())
        return masks
    # global: Schwelle auf |w|, Kosten pro Eintrag je Matrix verschieden
    allw = torch.cat([W.abs().flatten() for W in Wl])
    lo, hi = 0.0, float(allw.max())
    for _ in range(60):
        t = (lo+hi)/2
        c = sum(float((W.abs() >= t).sum())*(4+ib(W)/8) for W in Wl)
        if c > B_bytes: lo = t
        else: hi = t
    return [(W.abs() >= hi).float() for W in Wl]

# ---------- Auswertung der komprimierten Form (mit/ohne Nachtraining)
def lauf(seed, name_ds="mnist"):
    xtr, ytr, xte, yte = G.bilder(name_ds, seed, 10_000, 5_000, 16)
    xva, yva, xtr, ytr = xtr[-2000:], ytr[-2000:], xtr[:-2000], ytr[:-2000]
    torch.manual_seed(seed); L = mlp()
    trainiere(L, lambda: [None]*4, list(L.parameters()), xtr, ytr, EP, 1e-3, seed)
    dicht = acc(L, [None]*4, xte, yte); dicht_va = acc(L, [None]*4, xva, yva)
    Wl = [L[i].weight.detach().clone() for i in KOMP]
    Ftot = sum(W.numel() for W in Wl)
    rest = [p for i in (0, 1, 2, 3) for p in (L[i].bias,)] + [L[3].weight]
    res = {"seed": seed, "dicht": dicht, "dicht_va": dicht_va, "entropien": [], "ptot": Ftot, "ergebnis": []}
    for W in Wl:
        s, p, S = spektrum(W); res["entropien"].append({"form": list(W.shape), "S": S, "ppl": math.exp(S), "max_S": math.log(min(W.shape)), "r90": int((torch.cumsum(p, 0) < 0.9).sum())+1})
    base_flops = Ftot
    for f in FRAKT:
        B = int(f*Ftot)
        def fin(art, Ws_fn, params, flops, par, byt, ft_fn=None):
            """ohne und mit Nachtraining"""
            Wfull = [None]*4
            with torch.no_grad(): W0 = Ws_fn()
            a0 = acc(L2, [*W0, None], xte, yte); v0 = acc(L2, [*W0, None], xva, yva)
            for q in L2.parameters(): q.requires_grad_(False)
            for q in L2[3].parameters(): q.requires_grad_(True)
            for i in range(3): L2[i].bias.requires_grad_(True)
            trainiere(L2, lambda: [*Ws_fn(), None], params + [q for q in L2.parameters() if q.requires_grad], xtr, ytr, EPFT, 3e-4, seed+7)
            with torch.no_grad(): W1 = Ws_fn()
            a1 = acc(L2, [*W1, None], xte, yte); v1 = acc(L2, [*W1, None], xva, yva)
            res["ergebnis"].append({"f": f, "art": art, "ohne": a0, "mit": a1, "ohne_va": v0, "mit_va": v1, "par": par, "bytes": byt, "flops": flops})
        import copy
        # Niedrigrang-Varianten
        for art in ["gleich", "entropie", "wasser", "global_sv"]:
            L2 = copy.deepcopy(L)
            r = mk_alloc(Wl, B, art)
            fa = [lowrank_init(W, rr) for W, rr in zip(Wl, r)]
            A = [torch.nn.Parameter(a) for a, b in fa]; Bp = [torch.nn.Parameter(b) for a, b in fa]
            fin("lr_"+art, lambda: [A[i] @ Bp[i] for i in range(3)], A+Bp,
                sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl)), sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl)),
                4*sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl)))
            res["ergebnis"][-1]["raenge"] = r
        # TT
        L2 = copy.deepcopy(L)
        Bs = tt_budget_teilen(Wl, B); tts = []; fms = []
        for W, b in zip(Wl, Bs):
            cs, fm, fn = tt_waehle(W, b); tts.append([torch.nn.Parameter(c) for c in cs]); fms.append((fm, fn))
        par = sum(tt_par(c) for c in tts)
        fin("tt", lambda: [tt_voll_korrekt(tts[i], *fms[i]) for i in range(3)], [c for t in tts for c in t],
            sum(tt_flops(t) for t in tts), par, 4*par)
        res["ergebnis"][-1]["tt_raenge"] = [[c.shape[3] for c in t][:-1] for t in tts]
        # Pruning (Byte-Budget = 4*B)
        for lokal in (True, False):
            L2 = copy.deepcopy(L)
            ms = prune(Wl, 4*B, lokal)
            Wp = [W*m for W, m in zip(Wl, ms)]
            P = [torch.nn.Parameter(w.clone()) for w in Wp]
            nnz = int(sum(m.sum() for m in ms))
            byt = sum(float(m.sum())*(4+math.ceil(math.log2(W.numel()))/8) for m, W in zip(ms, Wl))
            fin("prune_lokal" if lokal else "prune_global", lambda: [P[i]*ms[i] for i in range(3)], P, nnz, nnz, byt)
        # Scratch-Kontrolle (Niedrigrang von Anfang an, gleiche Raenge wie 'gleich')
        r = mk_alloc(Wl, B, "gleich")
        torch.manual_seed(seed+11)
        L3 = mlp()
        A = [torch.nn.Parameter(torch.randn(W.shape[0], rr)*(2/ (W.shape[1]+rr))**0.5*0.7) for W, rr in zip(Wl, r)]
        Bp = [torch.nn.Parameter(torch.randn(rr, W.shape[1])*(2/(W.shape[1]+rr))**0.5*0.7) for W, rr in zip(Wl, r)]
        trainiere(L3, lambda: [A[i] @ Bp[i] for i in range(3)]+[None], A+Bp+[L3[3].weight]+[q.bias for q in L3],
                  xtr, ytr, EP, 1e-3, seed+13)
        with torch.no_grad(): Ws = [A[i] @ Bp[i] for i in range(3)]
        res["ergebnis"].append({"f": f, "art": "scratch_gleich", "ohne": acc(L3, [*Ws, None], xte, yte), "mit": float("nan"),
                                "ohne_va": acc(L3, [*Ws, None], xva, yva), "mit_va": float("nan"),
                                "par": sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl)), "flops": sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl)), "bytes": 4*sum(rr*(W.shape[0]+W.shape[1]) for rr, W in zip(r, Wl))})
    return res

if __name__ == "__main__":
    t0 = time.time(); alle = []
    for s in SEEDS:
        alle.append(lauf(s)); print("Seed", s, "fertig", round(time.time()-t0), "s", "dicht", alle[-1]["dicht"], flush=True)
        (HIER/("roh_schnell.json" if SCHNELL else "roh.json")).write_text(json.dumps(alle))
