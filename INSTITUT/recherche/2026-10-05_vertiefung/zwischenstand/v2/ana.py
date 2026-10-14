import json,glob,numpy as np
from scipy.stats import wilcoxon
Z=[json.loads(l) for f in glob.glob("roh_*.jsonl") for l in open(f)]
ref={"mnist":(0.9359,0.9017),"fashion_mnist":(0.8233,0.7993)}
def ms(x): x=np.array(x)*100; return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
for ds in ("mnist","fashion_mnist"):
    zs=[z for z in Z if z["ds"]==ds]; print("\n###",ds,"Seeds",len(zs), "Identität",ms([z["identitaet"] for z in zs]))
    for V in ("block2x2","block4x4","block8x8","global8"):
        k=zs[0][V]["k"]; print(f"-- {V} (k={k})")
        keine=np.array([z[V]["keine"] for z in zs]); print("   keine Maske           ",ms(keine))
        rm={}
        for r in range(2,k+1,2):
            m=np.array([np.mean(z[V]["rang"][str(r)]) for z in zs]); rm[r]=m
            inner=np.mean([np.std(z[V]["rang"][str(r)],ddof=1) if len(z[V]["rang"][str(r)])>1 else 0 for z in zs])*100
            allv=np.array([z[V]["rang"][str(r)] for z in zs])
            # Varianzzerlegung: Streuung innerhalb Rang (Form) vs. zwischen Seeds
            print(f"   Rang {r}: Mittel {ms(m)} | Streuung zw. Formen (gleicher Seed) {inner:.2f} pp | Spanne {100*allv.min():.1f}-{100*allv.max():.1f}")
        # Zufallsrang gleichverteilt (Rang 0 = keine Maske)
        zr=np.mean([keine]+[rm[r] for r in rm],axis=0); print("   Zufall, Rang gleichverteilt 0..k",ms(zr))
        bk=np.array([z[V]["bent_kanon"] for z in zs]); ba=np.array([z[V]["bent_adj"] for z in zs])
        bz=rm[k]
        print("   Bent kanon (i<->i+k/2)",ms(bk),"| Bent adj (i<->i+1)",ms(ba),"| Zufalls-Bent",ms(bz))
        d=lambda a,b:(100*(a-b).mean(),100*(a-b).std(ddof=1)/np.sqrt(len(a)),int((a>b).sum()))
        for nm,a,b in (("bent_kanon - keine",bk,keine),("bent_kanon - Zufalls-Bent",bk,bz),("bent_kanon - Zufall(Rang gl.)",bk,zr),("Zufalls-Bent - keine",bz,keine),("bent_adj - keine",ba,keine),("bent_kanon - bent_adj",bk,ba)):
            m,se,n=d(a,b); print(f"     {nm:30s} {m:+.2f} pp ± {se:.2f} (se), {n}/{len(a)} positiv")
        # Rang-Invariante: Varianz zwischen Rängen vs innerhalb
        allf=[z[V]["rang"][str(r)] for z in zs for r in range(2,k+1,2)]
    print()
