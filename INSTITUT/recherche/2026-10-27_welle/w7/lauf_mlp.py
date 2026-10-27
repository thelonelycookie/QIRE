import sys, os, json, time
sys.path.insert(0, "/home/user/QIRE/INSTITUT"); sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from labor.gemeinsam import bilder
from komplex import *
SEEDS = [960, 961, 962, 963, 964]
BUDGETS = [5000, 18000]
OUT = os.path.join(os.path.dirname(__file__), "mlp_ergebnis.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()
for ds in ["mnist", "fashion"]:
    for B in BUDGETS:
        # Arme: real (Param=B), kdicht (Param=B), phase (Param=B), phase_flop (FLOPs wie real)
        n = 256
        Hr = finde_H(n, "real", B)
        H = {"real": Hr,
             "kdicht": finde_H(n, "kdicht", B),
             "phase": finde_H(n, "phase", B),
             "phase_flop": finde_H(n, "phase", mlp_kosten(n, Hr, "real")[1], "flops")}
        for arm, h in H.items():
            art = "phase" if arm == "phase_flop" else arm
            P, Fl = mlp_kosten(n, h, art)
            # Lernrate auf Validierung (Seed 960, Val = letztes Viertel des Trainings), dann alle Seeds
            key0 = f"{ds}|{B}|{arm}|lr"
            if key0 not in res:
                x, y, _, _ = bilder(ds, 960, 10000, 2000)
                k = 7500
                best = None
                for lr in [0.003, 0.01, 0.03, 0.1]:
                    a, _ = trainiere_mlp(art, h, lr, 960, (x[:k], y[:k]), (x[k:], y[k:]))
                    if best is None or a > best[0]: best = (a, lr)
                res[key0] = best[1]; json.dump(res, open(OUT, "w"))
            lr = res[key0]
            for s in SEEDS:
                key = f"{ds}|{B}|{arm}|{s}"
                if key in res: continue
                xt, yt, xe, ye = bilder(ds, s, 10000, 5000)
                a, _ = trainiere_mlp(art, h, lr, s, (xt, yt), (xe, ye))
                res[key] = {"acc": a, "H": h, "P": P, "FLOPs": Fl, "lr": lr}
                json.dump(res, open(OUT, "w"))
            print(ds, B, arm, "H", h, "P", P, "F", Fl, "lr", lr, f"{time.time()-t0:.0f}s", flush=True)
