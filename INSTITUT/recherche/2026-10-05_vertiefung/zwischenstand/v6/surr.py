import os, sys, time, math, json, itertools
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
sys.path.insert(0, "/home/user/QIRE/INSTITUT")
L = 4

def ring_perm(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    for q in range(n):
        bits[:, (q + 1) % n] ^= bits[:, q]
    neu = (bits << (n - 1 - np.arange(n))).sum(1)
    perm = np.empty_like(neu); perm[neu] = idx
    return perm

def sim(X, th, n, perm):
    """X: (m,N) normierte Amplituden; th (L,n) Winkel; -> Born (m,N)."""
    m = X.shape[0]; psi = X.copy()
    for l in range(L):
        for q in range(n):
            c, s = math.cos(th[l, q] / 2), math.sin(th[l, q] / 2)
            v = psi.reshape(m, 2 ** q, 2, 2 ** (n - q - 1))
            a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
            n0 = c * a0 - s * a1; n1 = s * a0 + c * a1
            psi = np.stack([n0, n1], axis=2).reshape(m, -1)
        psi = psi[:, perm]
    return psi ** 2

def make_images(n, m, rng):
    d = np.load("/home/user/QIRE/INSTITUT/daten/mnist.npz")
    x = d["x"][:m].astype(np.float32).reshape(m, 28, 28)
    import torch
    side = 2 ** (n // 2) if n % 2 == 0 else None
    t = torch.tensor(x)[:, None]
    mode = "area" if side <= 28 else "bilinear"
    t = torch.nn.functional.interpolate(t, size=(side, side), mode=mode) if mode == "area" else torch.nn.functional.interpolate(t, size=(side, side), mode="bilinear", align_corners=False)
    X = t.reshape(m, -1).numpy().astype(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

B3 = np.array([[1, math.cos(2 * math.pi * g / 3), math.sin(2 * math.pi * g / 3)] for g in range(3)])
B3inv = np.linalg.inv(B3)
GRID = 2 * math.pi * np.arange(3) / 3

def coeffs(X, codes, pos, n, perm):
    """Koeffizienten von p(theta) in der Basis (1,cos,sin)^k : Tensor (3,)*k + (m,N)."""
    k = len(pos)
    base = np.asarray(codes, float) * (math.pi / 2)
    P = np.zeros((3,) * k + X.shape + (), dtype=np.float64) if k else None
    if k == 0:
        return sim(X, base, n, perm)[None]
    for gi in itertools.product(range(3), repeat=k):
        th = base.copy()
        for (l, q), g in zip(pos, gi): th[l, q] = GRID[g]
        P[gi] = sim(X, th, n, perm)
    for ax in range(k):
        P = np.moveaxis(np.tensordot(B3inv, P, axes=([1], [ax])), 0, ax)
    return P.reshape(3 ** k, *X.shape)

def analyse(C, k):
    """C: (3^k, m, N) Koeffizienten. Energien (Parseval) je Term, ohne konstanten Term."""
    K = C.shape[0]
    wts = np.ones(K)
    if k:
        w1 = np.array([1.0, .5, .5]); wts = w1
        for _ in range(k - 1): wts = np.outer(wts, w1).ravel()
    E = wts * (C.reshape(K, -1) ** 2).sum(1)
    tot = E[1:].sum() if K > 1 else 0.0
    res = {"K": K, "total_var": float(tot)}
    if K > 1 and tot > 0:
        Es = np.sort(E[1:])[::-1]; cs = np.cumsum(Es) / tot
        res["support"] = int((E[1:] > 1e-12 * tot).sum())
        for q in (0.9, 0.99, 0.999):
            res[f"n{q}"] = int(np.searchsorted(cs, q - 1e-12) + 1)
        res["eff_rang"] = float(tot ** 2 / (Es ** 2).sum())
    return res, E

def trunc_err(X, codes, pos, n, perm, C, E, k, m_terms, rng, ntest=40):
    """rel. RMS-Fehler der abgeschnittenen Entwicklung (konstanter Term immer dabei) auf Zufallswinkeln."""
    base = np.asarray(codes, float) * (math.pi / 2)
    order = np.argsort(-E[1:]) + 1
    keep = np.concatenate([[0], order[:m_terms]]); mask = np.zeros(C.shape[0], bool); mask[keep] = True
    err2 = tot2 = 0.0
    for _ in range(ntest):
        th = base.copy(); ang = rng.uniform(0, 2 * math.pi, k)
        for (l, q), a in zip(pos, ang): th[l, q] = a
        p = sim(X, th, n, perm)
        # Basisfunktionen
        f = np.ones(1)
        for a in ang: f = np.outer(f, [1, math.cos(a), math.sin(a)]).ravel()
        ph = np.tensordot(f * mask, C, axes=([0], [0]))
        err2 += ((p - ph) ** 2).sum(); tot2 += ((p - np.tensordot(f[:1], C[:1], axes=([0], [0]))) ** 2).sum()
    return math.sqrt(err2 / tot2)

def check_branches(X, codes, pos, n, perm, rng):
    """2^k Clifford-Zweige: U(theta)=sum_s w_s(theta) C_s mit Ry(t)=cos(t/2) I + sin(t/2) Ry(pi)."""
    k = len(pos); base = np.asarray(codes, float) * (math.pi / 2)
    ang = rng.uniform(0, 2 * math.pi, k); th = base.copy()
    for (l, q), a in zip(pos, ang): th[l, q] = a
    ref = np.sqrt(sim(X, th, n, perm)) # |amp|
    # Amplituden signed: simuliere ohne Quadrat
    def amp(th_):
        return np.sqrt(sim(X, th_, n, perm))  # nur Betrag
    return None

def amp_sim(X, th, n, perm):
    m = X.shape[0]; psi = X.copy()
    for l in range(L):
        for q in range(n):
            c, s = math.cos(th[l, q] / 2), math.sin(th[l, q] / 2)
            v = psi.reshape(m, 2 ** q, 2, 2 ** (n - q - 1))
            a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
            psi = np.stack([c * a0 - s * a1, s * a0 + c * a1], axis=2).reshape(m, -1)
        psi = psi[:, perm]
    return psi

def branch_check(X, codes, pos, n, perm, rng):
    k = len(pos); base = np.asarray(codes, float) * (math.pi / 2)
    ang = rng.uniform(0, 2 * math.pi, k); th = base.copy()
    for (l, q), a in zip(pos, ang): th[l, q] = a
    ref = amp_sim(X, th, n, perm)
    tot = np.zeros_like(ref)
    for s in itertools.product([0, 1], repeat=k):
        t2 = base.copy(); w = 1.0
        for (l, q), a, si in zip(pos, ang, s):
            t2[l, q] = 0.0 if si == 0 else math.pi
            w *= math.cos(a / 2) if si == 0 else math.sin(a / 2)
        tot += w * amp_sim(X, t2, n, perm)
    return float(np.abs(ref - tot).max())

if __name__ == "__main__":
    n = int(sys.argv[1]); ks = [int(a) for a in sys.argv[2].split(",")]; seeds = [int(a) for a in sys.argv[3].split(",")]
    m = int(sys.argv[4]); places = sys.argv[5].split(",")
    perm = ring_perm(n); out = []
    X = make_images(n, m, None)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        codes = rng.integers(0, 4, (L, n))
        for k in ks:
            for place in places:
                if place == "spread": pos = [divmod(int(p), n) for p in rng.choice(L * n, size=k, replace=False)]
                elif place == "layer0": pos = [(0, q) for q in range(k)] if k <= n else None
                elif place == "qubit0": pos = [(l, 0) for l in range(k)] if k <= L else None
                if pos is None: continue
                if place == "qubit0" and k > L: continue
                t0 = time.time()
                C = coeffs(X, codes, pos, n, perm)
                res, E = analyse(C, k)
                res.update(n=n, k=k, seed=seed, place=place)
                if k >= 1:
                    for mt in sorted({max(1, int(res.get("n0.99", 1))), max(1, int(res.get("n0.9", 1)))}):
                        res[f"err_m{mt}"] = trunc_err(X, codes, pos, n, perm, C, E, k, mt, rng, ntest=10)
                    res["branch_err"] = branch_check(X[:4], codes, pos, n, perm, rng) if k <= 8 else None
                res["sek"] = time.time() - t0
                out.append(res); print(json.dumps(res), flush=True)
    json.dump(out, open(f"surr_n{n}_{sys.argv[2].replace(',','_')}_{sys.argv[3].replace(',','_')}.json", "w"))
