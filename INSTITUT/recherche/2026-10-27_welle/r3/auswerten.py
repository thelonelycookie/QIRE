import json, os, numpy as np
r = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ergebnis.json")))
print("| Daten | Budget | Arm | H | Param | FLOPs | Acc % (n=5) | lr |\n|---|---|---|---|---|---|---|---|")
for ds in ["mnist", "fashion_mnist"]:
    for B in [5000, 18000]:
        arms = [k.split("|")[2] for k in r if k.startswith(f"{ds}|{B}|") and k.endswith("|lr")]
        for a in arms:
            v = [r[f"{ds}|{B}|{a}|{s}"] for s in range(980, 985) if f"{ds}|{B}|{a}|{s}" in r]
            if not v: continue
            x = np.array([e["acc"] for e in v]) * 100
            print(f"| {ds} | {B//1000}k | {a} | {v[0]['H']} | {v[0]['P']} | {v[0]['FLOPs']} | {x.mean():.2f} ± {x.std(ddof=1) if len(x)>1 else 0:.2f} | {v[0]['lr']} |")
