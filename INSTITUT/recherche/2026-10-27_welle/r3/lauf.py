import sys, os, json, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/user/QIRE/INSTITUT"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labor.gemeinsam import bilder
from arme import *
SEEDS = [980, 981, 982, 983, 984]; LRS = [0.01, 0.03, 0.1, 0.3]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ergebnis.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time(); n = 256
for ds in ["mnist", "fashion_mnist"]:
    for B in [5000, 18000]:
        # Referenz: Phase-FLOPs bei Budget B
        Hp = finde_H(n, "phase", B, 0); Fp = kosten(n, Hp, "phase")[1]
        arme = {a: finde_H(n, a, B, 0) for a in ["relu", "sq", "sqpaar", "phase", "zufall"]}
        arme["relu_F"] = finde_H(n, "relu", Fp, 1); arme["sq_F"] = finde_H(n, "sq", Fp, 1)
        for arm, h in arme.items():
            P, Fl = kosten(n, h, arm)
            k0 = f"{ds}|{B}|{arm}|lr"
            if k0 not in res:
                x, y, _, _ = bilder(ds, 980, 10000, 2000); k = 7500; best = None
                for lr in LRS:
                    a = trainiere(arm, h, lr, 980, (x[:k], y[:k]), (x[k:], y[k:]))
                    if best is None or a > best[0]: best = (a, lr)
                res[k0] = best[1]; json.dump(res, open(OUT, "w"))
            lr = res[k0]
            for s in SEEDS:
                key = f"{ds}|{B}|{arm}|{s}"
                if key in res: continue
                xt, yt, xe, ye = bilder(ds, s, 10000, 5000)
                res[key] = {"acc": trainiere(arm, h, lr, s, (xt, yt), (xe, ye)), "H": h, "P": P, "FLOPs": Fl, "lr": lr}
                json.dump(res, open(OUT, "w"))
            print(ds, B, arm, "H", h, "P", P, "F", Fl, "lr", lr, f"{time.time()-t0:.0f}s", flush=True)
