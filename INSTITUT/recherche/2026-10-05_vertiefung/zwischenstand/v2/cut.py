import os,sys,json
sys.path.insert(0,"/home/user/QIRE/INSTITUT")
import numpy as np
from scipy.stats import spearmanr
from abteilungen.maske_block import rref_basis
from abteilungen.maske_eigenschaft import rang_gf2
d=json.load(open("/home/user/QIRE/INSTITUT/ergebnisse/I-036-v3__20261005_200505.json"))
rng=np.random.default_rng(1)
def partial(x,y,c):
    x,y,c=map(np.asarray,(x,y,c))
    from scipy.stats import rankdata
    X,Y,C=rankdata(x),rankdata(y),rankdata(c)
    def res(a): 
        A=np.c_[C,np.ones_like(C)]; return a-A@np.linalg.lstsq(A,a,rcond=None)[0]
    return np.corrcoef(res(X),res(Y))[0,1]
for ds,zs in d["zeilen"].items():
    rows=[]
    for z in zs:
        if not z.get("analysierbar"): continue
        V=np.array(z["V_s"]); B=np.array(z["form_s"]); k=z["k"]
        basis=rref_basis(V)
        # Zeilen/Spalten-Schnitt
        Vr=[int(v) for v in V if v&15==0 and v]; Vc=[int(v) for v in V if v>>4==0 and v]
        dr=int(np.log2(len(Vr)+1)); dc=int(np.log2(len(Vc)+1))
        cr=None
        if dr+dc==k:
            def koord(v):
                # Koordinaten von v in rref-Basis: Ruecksubstitution
                t=[];x=v
                for b in basis:
                    hb=1<<(b.bit_length()-1)
                    if x&hb: x^=b;t.append(1)
                    else: t.append(0)
                assert x==0; return np.array(t)
            Kr=np.array([koord(v) for v in rref_basis(Vr)]); Kc=np.array([koord(v) for v in rref_basis(Vc)])
            cr=rang_gf2((Kr@B@Kc.T)%2)
        cuts=[]
        for _ in range(100):
            p=rng.permutation(k); a,b=p[:k//2],p[k//2:]
            cuts.append(rang_gf2(B[np.ix_(a,b)]))
        gain=z["suche"]-z["ohne_maske"]
        rows.append(dict(k=k,rang=z["rang"],cut_rc=cr,cut_rand=np.mean(cuts),gain=gain,zg=z["zufall_gleicher_rang"]-z["ohne_maske"],formgain=z["suche"]-z["zufall_gleicher_rang"]))
    n=len(rows); print(ds,n,"k-Verteilung",{k:sum(r['k']==k for r in rows) for k in range(9)})
    g=[r["gain"] for r in rows]
    for key in ("rang","cut_rand","k"):
        x=[r[key] for r in rows]
        print(f"  {key:9s} vs gain: rho={spearmanr(x,g)[0]:+.2f}  partial(k)={partial(x,g,[r['k'] for r in rows]):+.2f}  vs formgain(suche-zufall_gl.Rang): rho={spearmanr(x,[r['formgain'] for r in rows])[0]:+.2f}")
    rc=[r for r in rows if r["cut_rc"] is not None]
    print("  Zeile|Spalte zerlegbar:",len(rc),"/",n)
    if len(rc)>=5:
        print("   cut_rc vs gain rho=%+.2f  (rang vs gain in Teilmenge %+.2f); cut_rc-Werte"%(spearmanr([r['cut_rc'] for r in rc],[r['gain'] for r in rc])[0],spearmanr([r['rang'] for r in rc],[r['gain'] for r in rc])[0]),sorted(set(r['cut_rc'] for r in rc)))
    # Bent-Anteil
    bent=[r for r in rows if r["rang"]==r["k"]]; nb=[r for r in rows if r["rang"]<r["k"]]
    print("  bent (rang==k):",len(bent),"Gewinn %.2f pp"%(100*np.mean([r['gain'] for r in bent]) if bent else float('nan')),"| nicht bent:",len(nb),"Gewinn %.2f pp"%(100*np.mean([r['gain'] for r in nb]) if nb else float('nan')))
    print("  rang-k Defizit (k-rang) vs gain rho=%+.2f"%spearmanr([r['k']-r['rang'] for r in rows],g)[0])
    from collections import Counter
    print("  (k,rang):",sorted(Counter((r['k'],r['rang']) for r in rows).items()))
