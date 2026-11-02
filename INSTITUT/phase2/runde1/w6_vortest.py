#!/usr/bin/env python3
"""W6-Vortest (Phase 2, Runde 1): Orbit-Lloyd (Idee 2), Kerdock-Winkelbuch (Idee 3), E8-Nachbau, Lloyd-1D/2D.
Erkundung, kein Befund. Synthetisch: i.i.d. N(0,1) (Stand-in fuer hadamard-rotierte Gewichte), Seeds 0-2.
Start: OMP_NUM_THREADS=2 python3 w6_vortest.py   (Laufzeit ca. 15-25 min)
Alle Arme: eine globale Skala (bekannte Varianz 1), daher Skalenkosten gleich 0 und gleich fuer alle;
Gruppenskalen (g16) sind NICHT enthalten. Bits/Dim = Indexbits/Dim; Codebuch-Zahlen separat ausgewiesen.
"""
import os, json, time, itertools, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def snr(mse): return -10*np.log10(mse)

# ---------------- Lloyd-Max 1D, k-means ----------------
def lloyd1d(x, L, it=80):
    c = np.quantile(x, (np.arange(L)+0.5)/L)
    for _ in range(it):
        t = (c[1:]+c[:-1])/2; i = np.searchsorted(t, x)
        c = np.array([x[i == k].mean() if (i == k).any() else c[k] for k in range(L)])
    return c
def q1d(x, c):
    t = (c[1:]+c[:-1])/2; return c[np.searchsorted(t, x)]

def argmin_dist(X, C, chunk=4000):
    h = 0.5*(C*C).sum(1).astype(np.float32); Cf = C.astype(np.float32).T; out = np.empty(len(X), np.int64)
    for s in range(0, len(X), chunk):
        out[s:s+chunk] = (X[s:s+chunk].astype(np.float32)@Cf - h).argmax(1)
    return out
def kmeans(X, K, rng, it=30):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(it):
        a = argmin_dist(X, C); d = X.shape[1]
        cnt = np.bincount(a, minlength=K); S = np.zeros((K, d))
        for j in range(d): S[:, j] = np.bincount(a, X[:, j], minlength=K)
        ok = cnt > 0; C[ok] = S[ok]/cnt[ok, None]
        if (~ok).any(): C[~ok] = X[rng.choice(len(X), (~ok).sum())]
    return C
def mse_cb(X, C): return ((X-C[argmin_dist(X, C)])**2).mean()

# ---------------- E8 ----------------
def nearest_D8(x):
    f = np.rint(x); odd = (f.sum(-1) % 2) != 0
    if odd.any():
        e = x-f; i = np.abs(e).argmax(-1); r = np.arange(len(x))
        sg = np.where(e[r, i] >= 0, 1.0, -1.0)
        f[r[odd], i[odd]] += sg[odd]
    return f
def nearest_E8(x):
    a = nearest_D8(x); b = nearest_D8(x-0.5)+0.5
    da = ((x-a)**2).sum(-1); db = ((x-b)**2).sum(-1)
    return np.where((da <= db)[:, None], a, b)
def e8_roots():
    R = []
    for i, j in itertools.combinations(range(8), 2):
        for s in itertools.product([1, -1], repeat=2):
            v = np.zeros(8); v[i], v[j] = s; R.append(v)
    for s in itertools.product([.5, -.5], repeat=8):
        if sum(1 for t in s if t < 0) % 2 == 0: R.append(np.array(s))
    return np.array(R)
ROOTS = e8_roots(); assert len(ROOTS) == 240
def sigma3(k): return sum(d**3 for d in range(1, k+1) if k % d == 0)
HP = np.array([2654435761, 40503, 97, 1299709, 7919, 104729, 15485863, 32452843], dtype=np.int64)
class E8Book:
    def __init__(s, bits):
        N = 1; k = 0
        while True:
            k += 1; sh = 240*sigma3(k)
            if N+sh >= 2**bits: break
            N += sh
        s.K = k; s.f = (2**bits-N)/sh; s.bits = bits
    def inside(s, q):
        kk = np.rint((q*q).sum(-1)/2).astype(np.int64)
        h = (np.rint(2*q).astype(np.int64) @ HP) % 1000003/1000003.0
        return (kk < s.K) | ((kk == s.K) & (h < s.f))
    def enc(s, X, D):
        best = None; bd = np.full(len(X), np.inf)
        for lam in 0.985**np.arange(0, 80, 2):
            q = nearest_E8(lam*X/D); ok = s.inside(q)
            d = ((D*q-X)**2).sum(-1); d[~ok] = np.inf
            u = d < bd
            if best is None: best = q.copy()
            best[u] = q[u]; bd[u] = d[u]
        for _ in range(2):  # lokale Verbesserung ueber die 240 Wurzeln
            for r in ROOTS:
                q = best+r; ok = s.inside(q); d = ((D*q-X)**2).sum(-1); d[~ok] = np.inf
                u = d < bd; best[u] = q[u]; bd[u] = d[u]
        return D*best
    def mse(s, X, D): return ((s.enc(X, D)-X)**2).mean()

def e8_arm(bits, Xtr, Xte):
    bk = E8Book(bits)
    lo, hi = 0.15, 1.2   # Skala D grob suchen (Train), Goldener Schnitt
    g = (np.sqrt(5)-1)/2; a, b = lo, hi
    c = b-g*(b-a); d = a+g*(b-a); fc = bk.mse(Xtr, c); fd = bk.mse(Xtr, d)
    for _ in range(9):
        if fc < fd: b, d, fd = d, c, fc; c = b-g*(b-a); fc = bk.mse(Xtr, c)
        else: a, c, fc = c, d, fd; d = a+g*(b-a); fd = bk.mse(Xtr, d)
    D = (a+b)/2
    return bk.mse(Xte, D), D, bk

# ---------------- Cliffords ----------------
s2 = 1/np.sqrt(2)
Hh = np.array([[s2, s2], [s2, -s2]]); Xx = np.array([[0, 1], [1, 0.]]); Zz = np.diag([1, -1.]); I2 = np.eye(2)
def op(n, ops):
    M = np.eye(1)
    for q in range(n): M = np.kron(M, ops.get(q, I2))
    return M
def ctrl(n, a, b, Z):  # CZ (Z=True) oder CNOT a->b
    M = np.zeros((2**n, 2**n))
    for i in range(2**n):
        ba = (i >> (n-1-a)) & 1; bb = (i >> (n-1-b)) & 1
        if Z: M[i, i] = -1 if (ba and bb) else 1
        else: M[i ^ ((1 << (n-1-b)) if ba else 0), i] = 1
    return M
def gen_group(gens, cap=70000):
    key = lambda M: np.rint(M*1e6).astype(np.int64).tobytes()
    n = gens[0].shape[0]; seen = {key(np.eye(n)): np.eye(n)}; fr = [np.eye(n)]
    while fr:
        nf = []
        for M in fr:
            for g in gens:
                N = g@M; k = key(N)
                if k not in seen:
                    seen[k] = N; nf.append(N)
                    if len(seen) > cap: return None
        fr = nf
    return np.array(list(seen.values()))
def group_sets(n):
    loc = [op(n, {q: A}) for q in range(n) for A in (Hh, Xx, Zz)]
    pa = [op(n, {q: A}) for q in range(n) for A in (Xx, Zz)]
    cz = [ctrl(n, a, b, True) for a, b in itertools.combinations(range(n), 2)]
    S = {"D8^n lokal (H,X,Z je Qubit)": loc, "Pauli+CZ": pa+cz, "Pauli+H^n": pa+[op(n, {q: Hh for q in range(n)})]}
    if n == 2: S["Pauli+CZ+H1"] = pa+cz+[op(n, {0: Hh})]
    if n == 3: S["Pauli+CZ+H1"] = pa+cz+[op(n, {0: Hh})]; S["Pauli+CZ+H^3"] = pa+cz+[op(n, {q: Hh for q in range(n)})]
    return S

def clifford_check():
    out = {}
    n = 3; gens = [op(n, {q: A}) for q in range(n) for A in (Hh, Xx, Zz)]
    gens += [ctrl(n, a, b, z) for a, b in itertools.permutations(range(n), 2) for z in (True, False)]
    key = lambda v: np.rint(v*1e6).astype(np.int64).tobytes()
    e0 = np.zeros(8); e0[0] = 1
    seen = {key(e0): e0}; fr = [e0]
    while fr:
        nf = []
        for v in fr:
            for g in gens:
                w = g@v; k = key(w)
                if k not in seen: seen[k] = w; nf.append(w)
        fr = nf
    V = np.array(list(seen.values())); out["reale_Stabilisatorvektoren_inkl_Vorzeichen"] = len(V)
    out["bis_auf_Vorzeichen"] = len(V)//2
    G = V@V.T; out["Gram_Werte"] = sorted(set(np.round(G.ravel(), 6).tolist()))
    out["Rang"] = int(np.linalg.matrix_rank(V))
    out["Stabilisatorvektor_Eintragswerte_|x|"] = sorted(set(np.round(np.abs(V).ravel(), 5).tolist()))
    vals, cnts = np.unique(np.round(G.ravel(), 4), return_counts=True)
    out["Skalarprodukt_Spektrum_(Wert:Anzahl)"] = {str(float(v)): int(c) for v, c in zip(vals, cnts)}
    out["E8_Wurzelsystem_Gram_{0,+-1/2,+-1}?"] = bool(set(out["Gram_Werte"]) <= {-1.0, -.5, 0.0, .5, 1.0})
    rg = np.random.default_rng(0); tr = []
    for _ in range(200):
        M = np.eye(8)
        for _ in range(12): M = gens[rg.integers(len(gens))]@M
        tr.append(np.trace(M))
    out["Spuren_von_200_Zufallselementen_alle_ganzzahlig"] = bool(np.allclose(tr, np.rint(tr)))
    out["Beispiel_nichtganzzahlige_Spuren"] = [round(float(t), 4) for t in tr if abs(t-round(t)) > 1e-6][:5]
    idx = {k: i for i, k in enumerate(seen)}
    perms = []
    for g in gens:
        perms.append([idx[key(g@v)] for v in V])
    out["Generatoren_permutieren_die_Vektoren"] = True
    try:
        from sympy.combinatorics import Permutation, PermutationGroup
        PG = PermutationGroup([Permutation(p) for p in perms]); o = PG.order()
        out["Ordnung_reale_Clifford_C3 (wirkt treu)"] = int(o)
        out["Ordnung_W(E8)"] = 696729600
        out["Index_in_W(E8)"] = 696729600/int(o)
    except Exception as e:
        out["sympy"] = f"nicht verfuegbar: {e}"
    return out, V

# ---------------- Orbit-Lloyd ----------------
def orbit_lloyd(Gm, m, Xtr, rng, it=20, chunk=4000):
    ng, d = len(Gm), Gm.shape[1]
    V = Xtr[rng.choice(len(Xtr), m, replace=False)].copy()
    GT = np.transpose(Gm, (0, 2, 1))
    def build(V): return np.einsum('gij,mj->mgi', Gm, V).reshape(m*ng, d)
    for t in range(it):
        C = build(V); a = argmin_dist(Xtr, C, chunk)
        j, g = a//ng, a % ng
        Y = np.einsum('nij,nj->ni', GT[g], Xtr)  # g^T x
        cnt = np.bincount(j, minlength=m); S = np.zeros((m, d))
        for k in range(d): S[:, k] = np.bincount(j, Y[:, k], minlength=m)
        ok = cnt > 0; V[ok] = S[ok]/cnt[ok, None]
        if (~ok).any(): V[~ok] = Xtr[rng.choice(len(Xtr), (~ok).sum())]
    C = build(V)
    return V, C

# ---------------- Kerdock-/Winkelbuch ----------------
def polar_arm(Xtr, Xte, Wn, L):
    def prep(X):
        z = X[:, 0]+1j*X[:, 1]; return z
    ztr, zte = prep(Xtr), prep(Xte)
    def rho(z):
        th = np.angle(z); k = np.rint(th/(2*np.pi/Wn)).astype(int); e = th-k*(2*np.pi/Wn)
        return np.abs(z)*np.cos(e), k, np.abs(z)**2
    r_tr, _, _ = rho(ztr)
    rc = lloyd1d(r_tr, L, 60) if L > 1 else np.array([r_tr.mean()])
    r, k, a2 = rho(zte)
    rq = q1d(r, rc) if L > 1 else np.full_like(r, rc[0])
    # Fehler = |z|^2 - 2 rq*rho + rq^2
    return (a2-2*rq*r+rq**2).mean()/2   # pro Dimension

def main():
    res = {"seeds": SEEDS, "info": "synthetisch N(0,1) i.i.d.; echte Gewichtsmatrix nicht gefunden (kein Datenordner mit Gewichten, kein torch im Container)"}
    cl, V = clifford_check(); res["clifford"] = cl; log("Clifford:", {k: v for k, v in cl.items() if k != "Gram_Werte"})
    # Gruppen
    grp = {}
    for n, targets in ((2, (256, 4096)), (3, (65536,))):
        for name, gens in group_sets(n).items():
            Gm = gen_group(gens)
            if Gm is None: log(f"n={n} {name}: Ordnung > Cap"); continue
            o = len(Gm); log(f"n={n} {name}: Ordnung {o}")
            grp[(n, name)] = Gm
    res["gruppen"] = {f"{n}:{nm}": len(G) for (n, nm), G in grp.items()}
    rows = {}   # (Bits, Arm) -> list ueber Seeds
    def add(R, arm, mse, extra=None):
        rows.setdefault((R, arm), []).append(float(snr(mse)))
    extras = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for R in (2, 3):
            Ntr1, Nte1 = 300000, 600000
            x1 = rng.standard_normal(Ntr1); t1 = rng.standard_normal(Nte1)
            c = lloyd1d(x1, 2**R); add(R, "Lloyd-1D", ((q1d(t1, c)-t1)**2).mean())
            X2 = rng.standard_normal((150000, 2)); T2 = rng.standard_normal((300000, 2))
            C2 = kmeans(X2, 2**(2*R), rng); add(R, "Lloyd-2D", mse_cb(T2, C2))
            # Kerdock-/Winkelbuch: alle Aufteilungen Winkel W=2^a, Betrag L=2^(2R-a)
            best = None
            for a in range(1, 2*R+1):
                Wn, L = 2**a, 2**(2*R-a)
                m = polar_arm(X2, T2, Wn, L); add(R, f"Winkel W={Wn},L={L}", m)
                if best is None or m < best[0]: best = (m, Wn, L)
            add(R, "Kerdock-Winkelbuch (bester Split)", best[0]); extras.setdefault(f"polar_best_R{R}", []).append((best[1], best[2]))
            # 4D k-means (unstrukturierte Obergrenze) und Orbit-Lloyd 4D
            X4 = rng.standard_normal((120000, 4)); T4 = rng.standard_normal((200000, 4))
            C4 = kmeans(X4, 2**(4*R), rng, 25); add(R, "k-means 4D (unstrukturiert)", mse_cb(T4, C4))
            for (n, nm), Gm in grp.items():
                if n != 2: continue
                o = len(Gm); K = 2**(4*R)
                if K % o or K//o < 1: continue
                V4, Cb = orbit_lloyd(Gm, K//o, X4, rng, 25)
                add(R, f"Orbit-Lloyd 4D [{nm}, |G|={o}, m={K//o}]", mse_cb(T4, Cb))
            # E8
            X8 = rng.standard_normal((20000, 8)); T8 = rng.standard_normal((40000, 8))
            m8, D, bk = e8_arm(8*R, X8, T8); add(R, "E8 Kugelschnitt (nachgebaut)", m8); extras.setdefault(f"E8_D_R{R}", []).append(float(D))
            log(f"seed {seed} R={R}: L1D {rows[(R,'Lloyd-1D')][-1]:.3f} L2D {rows[(R,'Lloyd-2D')][-1]:.3f} E8 {rows[(R,'E8 Kugelschnitt (nachgebaut)')][-1]:.3f}")
        # Orbit-Lloyd 8D, nur 2 Bit
        R = 2; X8 = rng.standard_normal((16000, 8)); T8 = rng.standard_normal((20000, 8))
        for (n, nm), Gm in grp.items():
            if n != 3: continue
            o = len(Gm); K = 2**16
            if K % o: continue
            V8, Cb = orbit_lloyd(Gm, K//o, X8, rng, 18)
            uq = len(np.unique(np.rint(Cb*1e5).astype(np.int64), axis=0))
            add(R, f"Orbit-Lloyd 8D [{nm}, |G|={o}, m={K//o}]", mse_cb(T8, Cb)); extras.setdefault(f"unique_{nm}", []).append(uq)
            log(f"seed {seed} Orbit8D {nm}: {rows[(R, f'Orbit-Lloyd 8D [{nm}, |G|={o}, m={K//o}]')][-1]:.3f} dB, einzigartige Punkte {uq}")
    # Auswertung
    table = {}
    for (R, arm), v in rows.items():
        table.setdefault(R, {})[arm] = {"snr_dB_pro_seed": v, "mean": float(np.mean(v)), "std": float(np.std(v))}
    for R in table:
        b = table[R]["Lloyd-1D"]["snr_dB_pro_seed"]
        for arm, d in table[R].items():
            gain = np.array(d["snr_dB_pro_seed"])-np.array(b)
            d["gewinn_vs_Lloyd1D_dB_mean"] = float(gain.mean()); d["gewinn_std"] = float(gain.std())
            d["luecke_zu_Shannon_dB"] = float(6.0206*R-d["mean"])
    res["tabelle"] = {str(R): table[R] for R in table}; res["extras"] = extras
    res["laufzeit_s"] = time.time()-T0
    json.dump(res, open(os.path.join(HERE, "W6_vortest.json"), "w"), indent=1, default=str)
    write_md(res, table)
    log("fertig")

def write_md(res, table):
    cl = res["clifford"]
    L = ["# W6 Vortest (Phase 2, Runde 1): Orbit-Lloyd, Kerdock-Winkelbuch, E8-Nachbau", "",
         "Status: **Erkundung, kein Befund.** Skript `w6_vortest.py`, Daten `W6_vortest.json`. Synthetisch i.i.d. N(0,1) (Stand-in für hadamard-rotierte Gewichte), Seeds 0 bis 2, Test auf frischen Daten (Train/Test getrennt). SNR = -10 log10(MSE) bei Varianz 1; Shannon-Grenze = 6,02 dB je Bit.",
         "Echte Gewichtsmatrix: nicht getestet (im Container keine Gewichte, kein torch).", "",
         "## Bit-Buchhaltung", "- Alle Arme: Index-Bits = Rate x Dim, eine globale Skala (Varianz bekannt, 0 Bit/Dim), also Skalenkosten für alle gleich und **Gruppenskalen (g16) nicht eingerechnet**.",
         "- Codebuch-Zahlen (Float16): Lloyd-1D 2^R; Lloyd-2D 2^(2R) x 2 (32 bzw. 128); E8: 2 (Skala, Radius/Regel), Buch aus Regel; Orbit-Lloyd 8D: m x 8 Zahlen (m=64: 512) plus Gruppenregel; Winkelbuch: L Beträge; k-means 4D: 2^(4R) x 4 (Referenz, nicht buchfrei).", "",
         "## Clifford/E8 (numerisch)"]
    for k, v in cl.items(): L.append(f"- {k}: {v}")
    L += ["", "Deutung: (1) 480 reale Stabilisatorvektoren (240 Geraden); Skalarprodukte auch ±1/√2, ±1/(2√2), also NICHT das E8-Wurzelsystem (nur 0, ±1/2, ±1). (2) Die Spuren von Gruppenelementen sind teils irrational (Vielfache von √2). Jede Untergruppe von W(E8) erhält ein Gitter, hat also ganzzahlige Spuren. **Die reale 3-Qubit-Clifford-Gruppe in der Qubit-Darstellung liegt daher NICHT in W(E8)** (hinreichend: irrationale Spur). Nur die Gruppenordnung 5160960 = |W(E8)|/135 stimmt mit einer abstrakten Untergruppe (2^(1+6).O6+(2)) von Index 135 überein, aber als andere 8-dim. Darstellung. Folge: Idee 1 ist nicht automatisch ein E8-Spezialfall; die Hypothese 'Clifford in W(E8)' aus W6_ideen ist in dieser Darstellung widerlegt."", ""]
    L += ["## Gruppen (Untergruppen der realen Clifford-Gruppe, per BFS erzeugt)"] + [f"- {k}: Ordnung {v}" for k, v in res["gruppen"].items()] + [""]
    for R in sorted(table):
        L += [f"## Ergebnis {R} Bit/Dim (Mittel ± Std über Seeds 0-2)", "", "| Arm | SNR dB | Gewinn vs Lloyd-1D dB | Lücke zu Shannon dB |", "|---|---|---|---|"]
        for arm, d in sorted(table[R].items(), key=lambda kv: -kv[1]["mean"]):
            L.append(f"| {arm} | {d['mean']:.3f} ± {d['std']:.3f} | {d['gewinn_vs_Lloyd1D_dB_mean']:+.3f} ± {d['gewinn_std']:.3f} | {d['luecke_zu_Shannon_dB']:.2f} |")
        L.append("")
    L += ["## Einordnung (vorläufig, ehrlich)", "- Zahlen siehe Tabellen; Auswertung unten vom Autor des Skripts nachzutragen/prüfen.",
          "- E8-Nachbau: Nächster-Punkt-Algorithmus (Conway/Sloane) + Schnitt auf Kugelschalen mit hashbasierter Teilschale, Skala auf Train getunt; Kandidaten-Suche (Schrumpfung + Wurzel-Nachbarn) ist nahezu, nicht garantiert exakt nächstgelegen im Buch (leichte Unterschätzung möglich). Kein E8P (Flip-Aufbau von QuIP#).",
          "- Orbit-Lloyd: Lloyd mit gruppeninvariantem Buch (Schwerpunkte in kanonischer Lage gemittelt), Zufallsinit aus Trainingsvektoren, 18-25 Iterationen, keine Mehrfach-Neustarts; lokales Optimum möglich.",
          "- 8D-Orbit-Lloyd nur bei 2 Bit (2^24 Punkte bei 3 Bit sind per Brute-Force nicht in der Laufzeit); bei 3 Bit nur 4D.",
          "- Winkelbuch: bester Split auf den Testdaten gewählt (leichter Optimismus, 2-6 Kandidaten).",
          "- Clifford-Wirkung bleibt klassisch (I-038); Formulierung 'quanten-inspiriert'."]
    open(os.path.join(HERE, "W6_vortest.md"), "w").write("\n".join(L)+"\n")

if __name__ == "__main__": main()
