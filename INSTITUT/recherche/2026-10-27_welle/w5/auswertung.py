import json, numpy as np, sys
from pathlib import Path
R = json.loads((Path(__file__).parent/"roh.json").read_text())
n = len(R); print("Seeds", n)
print("dicht Test", f"{100*np.mean([r['dicht'] for r in R]):.2f} ± {100*np.std([r['dicht'] for r in R],ddof=1):.2f}")
for i in range(3):
    e=[r['entropien'][i] for r in R]
    print("Matrix",i,e[0]['form'],"S/Smax", f"{np.mean([x['S']/x['max_S'] for x in e]):.3f}","ppl",f"{np.mean([x['ppl'] for x in e]):.1f}","r90",np.mean([x['r90'] for x in e]))
arts=[]; 
for x in R[0]['ergebnis']:
    if x['art'] not in arts: arts.append(x['art'])
for f in sorted({x['f'] for x in R[0]['ergebnis']}):
    print(f"\n== Budget f={f} (Param komprimiert {int(f*R[0]['ptot'])})")
    print(f"{'Art':16s} {'ohne':>13s} {'mit NT':>13s} {'Param':>8s} {'Byte':>8s} {'FLOPs':>8s}")
    for a in arts:
        xs=[[y for y in r['ergebnis'] if y['f']==f and y['art']==a][0] for r in R]
        o=np.array([x['ohne'] for x in xs])*100; m=np.array([x['mit'] for x in xs])*100
        sd=lambda v: np.std(v,ddof=1) if len(v)>1 else 0
        print(f"{a:16s} {o.mean():6.2f}±{sd(o):4.2f} {m.mean():6.2f}±{sd(m):4.2f} {int(np.mean([x['par'] for x in xs])):8d} {int(np.mean([x['bytes'] for x in xs])):8d} {int(np.mean([x['flops'] for x in xs])):8d}")
    # gepaarte Diffs mit NT: entropie/wasser vs gleich
    for a,b in [("lr_entropie","lr_gleich"),("lr_wasser","lr_gleich"),("lr_global_sv","lr_gleich"),("lr_global_sv","tt"),("lr_gleich","prune_global")]:
        for k in ("ohne","mit"):
            d=np.array([[y for y in r['ergebnis'] if y['f']==f and y['art']==a][0][k]-[y for y in r['ergebnis'] if y['f']==f and y['art']==b][0][k] for r in R])*100
            print(f"  {a}-{b} [{k}]: {d.mean():+.2f} ± {np.std(d,ddof=1):.2f}pp")
print("\nRaenge (Seed 0):")
for f in sorted({x['f'] for x in R[0]['ergebnis']}):
    for x in R[0]['ergebnis']:
        if x['f']==f and ('raenge' in x or 'tt_raenge' in x): print(f, x['art'], x.get('raenge', x.get('tt_raenge')))
