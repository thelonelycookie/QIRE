import json, numpy as np, os
d = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(f"{d}/mlp_ergebnis.json"))
print("| Daten | Budget | Arm | H | Param | FLOPs | Test-Acc % (Mittel ± sd, n=5) | lr |\n|---|---|---|---|---|---|---|---|")
for ds in ["mnist", "fashion_mnist"]:
    for B in [5000, 18000]:
        for arm in ["real", "kdicht", "phase", "phase_flop"]:
            v = [r[f"{ds}|{B}|{arm}|{s}"] for s in range(960, 965)]
            a = np.array([x["acc"] for x in v]) * 100
            print(f"| {ds} | {B} | {arm} | {v[0]['H']} | {v[0]['P']} | {v[0]['FLOPs']} | {a.mean():.2f} ± {a.std(ddof=1):.2f} | {v[0]['lr']} |")
if os.path.exists(f"{d}/lm_ergebnis.json"):
    l = json.load(open(f"{d}/lm_ergebnis.json"))
    for arm in ["real", "kdicht", "phase", "phase_flop"]:
        v = [l[k] for k in l if k.startswith(arm + "|9")]
        if v: print("LM", arm, f"{np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v)>1 else 0:.4f}", len(v), l[arm + "|lr"])
