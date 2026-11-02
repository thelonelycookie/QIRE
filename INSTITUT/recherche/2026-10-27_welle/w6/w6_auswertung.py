import json, numpy as np
def env(pts):  # obere Hülle (flops aufsteigend, acc steigend)
    pts = sorted(pts); h = []; best = -1
    for f, a in pts:
        if a > best: h.append((f, a)); best = a
    return h
def interp(h, f):
    fs, as_ = zip(*h)
    return float(np.interp(f, fs, as_)) if f >= fs[0] else float("nan")
for name in ["mnist", "fashion_mnist"]:
    R = json.load(open(f"w6_roh_{name}.json")); print("##", name)
    taus = [r[0] for r in next(iter(R.values()))["adaptiv"]["test"]]
    print("tau | acc_ad | FLOPs_ad | mittl.Lagen | acc_stat(gleiche FLOPs) | diff(pp)")
    for i, t in enumerate(taus):
        A = np.array([[r["adaptiv"]["test"][i][1:] for r in [R[s]]][0] for s in R])
        d = []; st = []
        for s in R:
            r = R[s]; acc, fl, _ = r["adaptiv"]["test"][i][1:] + (0,) if False else (r["adaptiv"]["test"][i][1], r["adaptiv"]["test"][i][2], 0)
            pts = [(v[1], v[0]) for v in r["statisch"].values()] + [(v[1], v[0]) for v in r["ref_lage"].values()]
            sa = interp(env(pts), fl); st.append(sa); d.append(acc - sa)
        d = np.array(d) * 100
        print(f"{t:5.3f} | {A[:,0].mean()*100:.2f}±{A[:,0].std(ddof=1)*100:.2f} | {A[:,1].mean():.0f} | {A[:,2].mean():.2f} | {np.nanmean(st)*100:.2f} | {np.nanmean(d):+.2f}±{np.nanstd(d,ddof=1):.2f}")
    print("Statisch (Mittel acc%, FLOPs, Params):")
    for k in next(iter(R.values()))["statisch"]:
        v = np.array([R[s]["statisch"][k] for s in R]); print(f"  {k}: {v[:,0].mean()*100:.2f}±{v[:,0].std(ddof=1)*100:.2f}, {v[0,1]:.0f}, {v[0,2]:.0f}")
    print("Referenz feste Lage (Mehrkopf-Stamm):")
    for k in range(4):
        v = np.array([R[s]["ref_lage"][str(k)] for s in R]); print(f"  Lage{k+1}: {v[:,0].mean()*100:.2f}±{v[:,0].std(ddof=1)*100:.2f}, {v[0,1]:.0f}")
    # Val-gewählte Schwelle: kleinstes tau mit val-acc >= val-acc(tau=1.01) - 0.005
    sel = []
    for s in R:
        v = R[s]["adaptiv"]["val"]; full = v[-1][1]
        j = next(i for i, r in enumerate(v) if r[1] >= full - 0.005); sel.append(j)
    print("val-gewählte tau-Indizes:", [taus[j] for j in sel])
    for s, j in zip(R, sel):
        t = R[s]["adaptiv"]["test"]; print(f"  seed {s}: tau={taus[j]} acc {t[j][1]*100:.2f} FLOPs {t[j][2]:.0f} (voll {t[-1][1]*100:.2f}, {t[-1][2]:.0f})")
