import json, sys
from common import *
name = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
NTR, NTE = 1000, 2000
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
out = {}

def norm_tr(K):
    return K * (len(K) / np.trace(K))

def kern_classical(x):
    G = (x @ x.T).double().numpy()
    d2 = np.maximum(2 - 2 * G, 0)
    n = len(G); med = np.median(d2[np.triu_indices(n, 1)])
    return {"linear": G, "quadratisch": G ** 2, "rbf_med": np.exp(-d2 / med), "rbf_eng": np.exp(-4 * d2 / med)}

EIG = {}
def g_all(Phi, Kcs, lams):
    N_ = len(Phi); Phi = Phi * math.sqrt(N_ / float((Phi ** 2).sum()))
    res = {}
    KQ = Phi @ Phi.T
    cl = dict(Kcs); cl["SELBST"] = KQ
    for nm, Kc in cl.items():
        if nm in EIG: w, V = EIG[nm]
        else:
            Kc = norm_tr(Kc); w, V = np.linalg.eigh(Kc); w = np.maximum(w, 0)
            if nm != "SELBST": EIG[nm] = (w, V)
        Z = V.T @ Phi
        res[nm] = {}
        for lam in lams:
            M = Z.T @ (Z / (w + lam)[:, None])
            res[nm][str(lam)] = float(math.sqrt(np.linalg.eigvalsh(M)[-1]))
    return res

def rkhs_test(Phi, Kcs, rng, ntr=800, nt=20):
    N_ = len(Phi); Phi = Phi * math.sqrt(N_ / float((Phi ** 2).sum()))
    Phi = Phi - Phi.mean(0)
    Beta = rng.standard_normal((Phi.shape[1], nt)); Y = Phi @ Beta; Y /= Y.std(0, keepdims=True)
    res = {}
    for nm, Kc in Kcs.items():
        Kc = norm_tr(Kc); Kte = Kc[ntr:, :ntr]
        if ("r", nm) in EIG: w, V = EIG[("r", nm)]
        else:
            w, V = np.linalg.eigh(Kc[:ntr, :ntr]); w = np.maximum(w, 0); EIG[("r", nm)] = (w, V)
        best = 9e9
        for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            A = V @ ((V.T @ Y[:ntr]) / (w + lam)[:, None])
            best = min(best, float(((Kte @ A - Y[ntr:]) ** 2).mean()))
        res[nm] = best
    return res

def kr_acc(K_tr, K_te, ytr, yte, lams):
    n = len(K_tr); Y = np.eye(10)[ytr.numpy()]; va = np.arange(n) % 5 == 0; tr = ~va
    best, bl = -1, None
    for lam in lams:
        A = np.linalg.solve(K_tr[np.ix_(tr, tr)] + lam * np.eye(tr.sum()), Y[tr])
        a = (K_tr[np.ix_(va, tr)] @ A).argmax(1) == ytr.numpy()[va]
        if a.mean() > best: best, bl = a.mean(), lam
    A = np.linalg.solve(K_tr + bl * np.eye(n), Y)
    return float(((K_te @ A).argmax(1) == yte.numpy()).mean()), bl

for seed in seeds:
    t0 = time.time(); EIG.clear()
    xtr, ytr, xte, yte = daten(name, seed, NTR, NTE)
    cod, s, med, nb = clifford_suche(xtr, ytr, seed, steps=500, politur=60)
    codes = kanonisch(cod); print('suche', time.time()-t0, flush=True)
    rng = np.random.default_rng(970_000 + seed)
    rnd = rng.integers(0, 4, (LAGEN, N))
    th4, pos4, sc4 = magie_winkel(cod, xtr, ytr, 4, seed)
    print('magie', time.time()-t0, flush=True)
    thg = rng.uniform(0, 2 * math.pi, (LAGEN, N))
    Us = {"identitaet": torch.eye(256),
          "walsh_voll": bent_U(8, 0),
          "bent8": bent_U(8, 8),
          "bent4block": bent_U(4, 4),
          "zufall_clifford": unitaer_clifford(rnd % 2, N),
          "suche_clifford": unitaer_clifford(codes, N),
          "suche_k4": unitaer_winkel(th4, N),
          "generisch_k32": unitaer_winkel(thg, N)}
    Kcs_tr = kern_classical(xtr)
    Kc_full = kern_classical(torch.cat([xtr, xte]))
    r = {"seed": seed, "such_bew": nb, "loo_suche": s[0], "loo_zufallsmedian": med, "loo_k4": sc4, "pos4": [int(p) for p in pos4]}
    for nm, U in Us.items():
        Phi = born(xtr, U).double().numpy()
        g = g_all(Phi, Kcs_tr, LAMS)
        rk = rkhs_test(Phi, Kcs_tr, np.random.default_rng(seed))
        acc_kopf = kopf_logistisch(born(xtr, U), ytr, born(xte, U), yte, l2=1e-2)[0]
        Pt = born(xte, U).double().numpy()
        KQ = Phi @ Phi.T; KQte = Pt @ Phi.T; sc = len(Phi) / np.trace(KQ)
        acc_kq, _ = kr_acc(KQ * sc, KQte * sc, ytr, yte, [1e-3, 1e-2, 1e-1, 1, 10])
        r[nm] = {"g": g, "rkhs_nmse": rk, "acc_kopf": acc_kopf, "acc_krr_KQ": acc_kq}
    acc_cl = {}
    for k_, Kf in Kc_full.items():
        Kn = Kf * (NTR / np.trace(Kf[:NTR, :NTR]))
        acc_cl[k_] = kr_acc(Kn[:NTR, :NTR], Kn[NTR:, :NTR], ytr, yte, [1e-3, 1e-2, 1e-1, 1, 10])[0]
    r["acc_klassisch_krr"] = acc_cl
    out[str(seed)] = r
    json.dump(out, open(f"geo_{name}_{sys.argv[2].replace(',','_')}.json", "w"))
    print(name, seed, f"{time.time()-t0:.0f}s", round(r["loo_suche"],3), {k: round(v['g']['quadratisch']['0.001'], 2) for k, v in r.items() if isinstance(v, dict) and 'g' in v}, flush=True)
