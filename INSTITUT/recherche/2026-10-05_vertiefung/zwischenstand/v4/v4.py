"""v4 Holevo/Schuesse: Information pro Schuss, Chernoff, Genauigkeit gegen Schusszahl (Erkundung)."""
import os, sys, json, time, itertools
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, "/home/user/QIRE/INSTITUT")
import numpy as np, torch
torch.set_num_threads(2)
from abteilungen.clifford import daten
from abteilungen.clifford_suche import unitaer_clifford, normiere
from abteilungen.maske_block import matrix_vm, rref_basis, unterraum_aus_qubits
from abteilungen.maske_eigenschaft import maske_aus_form

OUT = "/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v4/"
SEEDS = list(range(960, 970))
SGRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
HEAD_S = [1, 3, 10, 30, 100, 300, 1000]
HEAD_CELLS = ["A_id", "B4x4_r0", "B4x4_r4", "B16_r8", "C_rand0", "D_suche"]
L2 = 0.003
LOG2 = np.log(2)

# ---------------------------------------------------------------- Zellen
QB = {2: (3, 7), 4: (2, 3, 6, 7), 6: (1, 2, 3, 5, 6, 7), 8: tuple(range(8))}

def bent_form(k, r):
    B = np.zeros((k, k), np.int64)
    h = k // 2
    for i in range(r // 2):
        B[i, i + h] = B[i + h, i] = 1
    return B

def zelle_W(k, r):
    V = unterraum_aus_qubits(QB[k])
    m = maske_aus_form(bent_form(k, r), V)
    return matrix_vm(V, m)

def alle_zellen(seed, such_codes):
    z = {"A_id": torch.eye(256)}
    for k, rs in [(2, (0, 2)), (4, (0, 2, 4)), (6, (0, 6)), (8, (0, 2, 4, 6, 8))]:
        for r in rs:
            nm = {2: "B2x2", 4: "B4x4", 6: "B8x8", 8: "B16"}[k] + f"_r{r}"
            z[nm] = zelle_W(k, r)
    g = np.random.default_rng(960_000 + seed)
    for j in range(5):
        z[f"C_rand{j}"] = unitaer_clifford(g.integers(0, 4, (4, 8)), 8)
    z["D_suche"] = unitaer_clifford(np.array(such_codes), 8)
    return z

# ---------------------------------------------------------------- Information
def H(p, ax=-1):
    p = np.asarray(p, np.float64)
    return -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(ax)

def info_C(q, pi):
    """I(C;X) in Bit aus Klassenverteilungen q (K,B), Prior pi."""
    qb = pi @ q
    return float(H(qb) - pi @ H(q))

def marg(q, qubits, n=8):
    K = q.shape[0]
    t = q.reshape((K,) + (2,) * n)
    drop = tuple(1 + i for i in range(n) if i not in qubits)
    return t.sum(axis=drop).reshape(K, -1) if drop else q

def chernoff_paare(q, e):
    """klassische Chernoff-Information (Bit) fuer alle Klassenpaare, q geglaettet."""
    qs = (q + e) / (1 + 256 * e)
    s = np.linspace(0.01, 0.99, 99)
    lq = np.log(qs)
    K = len(q)
    C = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            v = np.log(np.exp(s[:, None] * lq[a] + (1 - s[:, None]) * lq[b]).sum(1))
            C[a, b] = C[b, a] = -v.min() / LOG2
    return C

def holevo_quanten(xtr, ytr, pi):
    X = normiere(xtr).double().numpy()
    rho = [None] * 10
    for c in range(10):
        Xc = X[ytr.numpy() == c]
        rho[c] = Xc.T @ Xc / len(Xc)
    rb = sum(pi[c] * rho[c] for c in range(10))
    Sv = lambda r: float(H(np.clip(np.linalg.eigvalsh(r), 0, None)))
    chiC = Sv(rb) - sum(pi[c] * Sv(rho[c]) for c in range(10))
    chi_img = Sv(X.T @ X / len(X))     # reine Zustaende: S(rho_bar)
    # Quanten-Chernoff
    ev, U = zip(*[np.linalg.eigh(r) for r in rho])
    ss = np.linspace(0.05, 0.95, 19)
    pot = []
    for c in range(10):
        e = np.clip(ev[c], 1e-12, None)
        pot.append([(U[c] * e ** s) @ U[c].T for s in ss])
    xi = np.zeros((10, 10))
    for a in range(10):
        for b in range(a + 1, 10):
            v = [np.sum(pot[a][i] * pot[b][len(ss) - 1 - i].T) for i in range(len(ss))]
            xi[a, b] = xi[b, a] = -np.log2(min(v))
    return chiC, chi_img, xi

# ---------------------------------------------------------------- Schuesse
def nested_counts(P, slist, gen):
    """P (n,256) Zeilen normiert; nested: Zaehlungen nach den ersten S Schuessen."""
    n = P.shape[0]
    idx = torch.multinomial(P, max(slist), replacement=True, generator=gen)
    cnt = torch.zeros(n, P.shape[1])
    out, prev = {}, 0
    for S in slist:
        cnt.scatter_add_(1, idx[:, prev:S], torch.ones(n, S - prev))
        out[S] = cnt.clone()
        prev = S
    return out

def ml_acc(cnt, logq, logpi, y):
    pred = (cnt @ logq.T + logpi).argmax(1)
    return float((pred == y).float().mean()), pred

def mi_pred(y, pred):
    M = np.zeros((10, 10))
    np.add.at(M, (y.numpy(), pred.numpy()), 1)
    M /= M.sum()
    return float(H(M.sum(0)) + H(M.sum(1)) - H(M.ravel()))

def kopf_fit(ftr, ytr, max_iter=200):
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    A = torch.cat([(ftr - mu) / sd, torch.ones(len(ftr), 1)], 1)
    W = torch.zeros(A.shape[1], 10, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20)
    def f():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(A @ W, ytr) + L2 * (W[:-1] ** 2).sum()
        l.backward(); return l
    opt.step(f)
    return W.detach(), mu, sd

def kopf_pred(h, fte):
    W, mu, sd = h
    B = torch.cat([(fte - mu) / sd, torch.ones(len(fte), 1)], 1)
    return (B @ W).argmax(1)

# ---------------------------------------------------------------- Seed
def seed_lauf(seed, such_codes, mit_kopf=True):
    t0 = time.time()
    xtr, ytr, xte, yte = daten("mnist", seed)
    xtr_n, xte_n = normiere(xtr), normiere(xte)
    pi = np.bincount(ytr.numpy(), minlength=10) / len(ytr)
    logpi = torch.tensor(np.log(pi), dtype=torch.float32)
    chiC, chi_img, xiQ = holevo_quanten(xtr, ytr, pi)
    res = {"seed": seed, "chi_C": chiC, "chi_img": chi_img, "xiQ_mean": float(xiQ[np.triu_indices(10, 1)].mean()),
           "xiQ_min": float(xiQ[np.triu_indices(10, 1)].min()), "H_C": float(H(pi)), "zellen": {}}
    Z = alle_zellen(seed, such_codes)
    gen = torch.Generator().manual_seed(seed)
    nfit = 4800
    for nm, U in Z.items():
        Ptr = ((xtr_n @ U.T) ** 2).double(); Pte = ((xte_n @ U.T) ** 2).double()
        Ptr = (Ptr / Ptr.sum(1, keepdim=True)).float(); Pte = (Pte / Pte.sum(1, keepdim=True)).float()
        Pn = Ptr.numpy().astype(np.float64); Pt = Pte.numpy().astype(np.float64)
        q_tr = np.stack([Pn[ytr.numpy() == c].mean(0) for c in range(10)])
        q_te = np.stack([Pt[yte.numpy() == c].mean(0) for c in range(10)])
        r = {}
        r["I_C_train"] = info_C(q_tr, pi); r["I_C_test"] = info_C(q_te, pi)
        r["I_img_test"] = float(H(Pt.mean(0)) - H(Pt).mean())
        r["I_img_train"] = float(H(Pn.mean(0)) - H(Pn).mean())
        # m Qubits: alle Teilmengen
        mq = {}
        for m in range(1, 9):
            subs = list(itertools.combinations(range(8), m))
            it = [info_C(marg(q_tr, s), pi) for s in subs]
            best = subs[int(np.argmax(it))]
            mq[m] = {"best_test": info_C(marg(q_te, best), pi), "best_train": float(max(it)),
                     "mean_all_train": float(np.mean(it)), "min_train": float(min(it)), "best": list(best)}
        r["mq"] = mq
        # ML-Glaettung auf Validierung (Fit 4800 / Val 1200)
        qf = np.stack([Pn[:nfit][ytr.numpy()[:nfit] == c].mean(0) for c in range(10)])
        Pval = Ptr[nfit:]; yval = ytr[nfit:]
        cv = nested_counts(Pval, [10, 100], gen)
        best_e, best_a = None, -1
        for e in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            lq = torch.tensor(np.log((qf + e) / (1 + 256 * e)), dtype=torch.float32)
            a = np.mean([ml_acc(cv[S], lq, logpi, yval)[0] for S in (10, 100)])
            if a > best_a: best_a, best_e = a, e
        r["ml_eps"] = best_e
        q_final = (q_tr + best_e) / (1 + 256 * best_e)
        logq = torch.tensor(np.log(q_final), dtype=torch.float32)
        C = chernoff_paare(q_tr, best_e)
        iu = np.triu_indices(10, 1)
        r["chern_mean"] = float(C[iu].mean()); r["chern_min"] = float(C[iu].min()); r["chern_max"] = float(C[iu].max())
        # Union-Schranke-Vorhersage (Klassenmittel-Welt) in nats
        Cn = C * LOG2
        pred_acc = {S: float(np.mean([1 - min(1, sum(np.exp(-S * Cn[c, b]) for b in range(10) if b != c)) for c in range(10)])) for S in (1, 3, 10, 30, 100, 300, 1000)}
        r["union_acc"] = pred_acc
        # Schuss-Simulation Test
        cte = nested_counts(Pte, SGRID, gen)
        ml, mlI = {}, {}
        for S in SGRID:
            a, p = ml_acc(cte[S], logq, logpi, yte); ml[S] = a; mlI[S] = mi_pred(yte, p)
        r["ml_acc"] = ml; r["ml_I"] = mlI
        # exakter ML-Entscheider (S -> unendlich): Log-Likelihood mit Verteilung P_i
        # Kopf
        if mit_kopf and nm in HEAD_CELLS or (mit_kopf and nm.startswith("C_rand") and False):
            hx = kopf_fit(Ptr, ytr)
            pe = kopf_pred(hx, Pte); r["head_exact_inf"] = float((pe == yte).float().mean())
            hx_acc, hm_acc, hm_I, hx_I = {}, {}, {}, {}
            ctr = None
            for S in HEAD_S:
                f_te = cte[S] / S
                px = kopf_pred(hx, f_te); hx_acc[S] = float((px == yte).float().mean()); hx_I[S] = mi_pred(yte, px)
                f_tr = nested_counts(Ptr, [S], gen)[S] / S
                hm = kopf_fit(f_tr, ytr, max_iter=100)
                pm = kopf_pred(hm, f_te); hm_acc[S] = float((pm == yte).float().mean()); hm_I[S] = mi_pred(yte, pm)
            r["head_exact_S"] = hx_acc; r["head_matched_S"] = hm_acc
            r["head_exact_I"] = hx_I; r["head_matched_I"] = hm_I
        # ML mit exakter Verteilung (Grenzwert S->inf): log q_c . P_i
        pinf = (Pte @ logq.T + logpi).argmax(1)
        r["ml_inf"] = float((pinf == yte).float().mean()); r["ml_inf_I"] = mi_pred(yte, pinf)
        res["zellen"][nm] = r
        print(f"  seed {seed} {nm}: I_C={r['I_C_test']:.3f} I_img={r['I_img_test']:.3f} ml10={ml[10]:.3f} ml100={ml[100]:.3f} ml_inf={r['ml_inf']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    return res

if __name__ == "__main__":
    z = json.load(open("/home/user/QIRE/INSTITUT/ergebnisse/_zwischen_I-036-v3.json"))["zeilen"]
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS
    for s in seeds:
        codes = z[f"mess|mnist|{s - 200}"]["suche_codes"]
        res = seed_lauf(s, codes)
        json.dump(res, open(OUT + f"seed_{s}.json", "w"))
        print("seed fertig", s, flush=True)
