import json,sys,numpy as np,collections
z=[json.loads(l) for l in open(sys.argv[1])]
g=collections.OrderedDict()
for r in z: g.setdefault(r["arm"],[]).append(r)
print("| Arm | n | bpc Val (Mittel±sd) | Param FFN | FLOPs FFN/Token | Param ges | s/Lauf |\n|---|---|---|---|---|---|---|")
for a,rs in g.items():
    b=np.array([r["bpc"] for r in rs]); r0=rs[0]
    print(f"| {a} | {len(b)} | {b.mean():.3f} ± {b.std(ddof=1) if len(b)>1 else 0:.3f} | {r0['param_ffn']} | {r0['flops_ffn_token']} | {r0['param_ges']} | {np.mean([r['sek'] for r in rs]):.0f} |")
print("Kurve (Val 8k, Mittel):")
for a,rs in g.items(): print(a,np.mean([r["kurve_val8k"] for r in rs],0).round(3).tolist())
