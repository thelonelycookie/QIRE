import json, glob, numpy as np, sys
D="/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v4/"
R=[json.load(open(f)) for f in sorted(glob.glob(D+"seed_96*.json"))]
print("Seeds:",[r['seed'] for r in R])
def ms(x,f="%.3f"): x=np.array(x,float); return (f+" ± "+f)%(x.mean(),x.std(ddof=1) if len(x)>1 else 0)
cells=list(R[0]['zellen'])
def getc(nm,key):
    if nm=="C_rand":
        return [np.mean([r['zellen'][f"C_rand{j}"][key] for j in range(5)]) for r in R]
    return [r['zellen'][nm][key] for r in R]
def getd(nm,key,S):
    if nm=="C_rand":
        return [np.mean([r['zellen'][f"C_rand{j}"][key][str(S)] for j in range(5)]) for r in R]
    return [r['zellen'][nm][key][str(S)] for r in R]
names=[c for c in cells if not c.startswith("C_rand")]+["C_rand"]
print("chi_C (Holevo Klassenensemble, Bit)",ms([r['chi_C'] for r in R]),"chi_img",ms([r['chi_img'] for r in R]),"xiQ mean",ms([r['xiQ_mean'] for r in R]),"xiQ min",ms([r['xiQ_min'] for r in R]),"H_C",ms([r['H_C'] for r in R]))
print("\n== Information (Bit je Schuss), Test-Mittel; I_C aus Klassenmitteln (Test), I_img pro Bild; Chernoff mean/min")
for nm in names:
    print(f"{nm:10s} I_C {ms(getc(nm,'I_C_test'))} (train {np.mean(getc(nm,'I_C_train')):.3f}) | I_img {ms(getc(nm,'I_img_test'))} | chern_mean {ms(getc(nm,'chern_mean'))} min {ms(getc(nm,'chern_min'))} | eps {set(r['zellen'][nm]['ml_eps'] if nm!='C_rand' else 0 for r in R)}")
print("\n== ML-Entscheider Genauigkeit gegen S")
Ss=[1,2,3,5,10,20,30,50,100,200,300,500,1000]
print("cell      "+" ".join(f"S={s:<5d}" for s in Ss)+" inf")
for nm in names:
    row=[np.mean(getd(nm,'ml_acc',s)) for s in Ss]
    print(f"{nm:10s}"+" ".join(f"{a:.3f} " for a in row)+f" {np.mean(getc(nm,'ml_inf')):.3f}   sd@100 {np.std(getd(nm,'ml_acc',100),ddof=1):.3f}")
print("\n== Kopf (nur Kopf-Zellen): matched / exakt-Kopf auf Haeufigkeiten; head_inf")
HS=[1,3,10,30,100,300,1000]
for nm in ["A_id","B4x4_r0","B4x4_r4","B16_r8","C_rand0","D_suche"]:
    m=[ms(getd(nm,'head_matched_S',s),"%.3f") for s in HS]
    e=[np.mean(getd(nm,'head_exact_S',s)) for s in HS]
    print(f"{nm:9s} matched:"," ".join(m)," | exact-Kopf:"," ".join(f"{a:.3f}" for a in e)," | inf",ms(getc(nm,'head_exact_inf')))
# S needed
def s_need(acc_by_S, target):
    Sl=np.array(sorted(acc_by_S)); a=np.array([acc_by_S[s] for s in Sl])
    for i in range(len(Sl)):
        if a[i]>=target:
            if i==0: return float(Sl[0])
            f=(target-a[i-1])/(a[i]-a[i-1]) if a[i]>a[i-1] else 1
            return float(np.exp(np.log(Sl[i-1])+f*(np.log(Sl[i])-np.log(Sl[i-1]))))
    return np.nan
def need_stats(nm,key,target,rel=None):
    out=[]
    for r in R:
        cs=[r['zellen'][nm]] if nm!="C_rand" else [r['zellen'][f"C_rand{j}"] for j in range(5)]
        v=[]
        for c in cs:
            d={int(k):x for k,x in c[key].items()}
            if rel: 
                tt = rel*(c['ml_inf'] if key=='ml_acc' else c['head_exact_inf']) 
            else: tt=target
            v.append(s_need(d,tt))
        out.append(np.mean(v) if not np.isnan(v).all() else np.nan)
    out=np.array(out); ok=~np.isnan(out)
    return ok.sum(), (np.exp(np.mean(np.log(out[ok]))) if ok.any() else np.nan)
print("\n== S fuer Ziel (geometrisches Mittel ueber erreichende Seeds, n/10)")
for nm in names:
    print(f"{nm:9s} ML: 70% {need_stats(nm,'ml_acc',0.70)} 80% {need_stats(nm,'ml_acc',0.80)} 90% {need_stats(nm,'ml_acc',0.90)} | 99% von ML-inf {need_stats(nm,'ml_acc',None,0.99)}")
for nm in ["A_id","B4x4_r0","B4x4_r4","B16_r8","C_rand0","D_suche"]:
    print(f"{nm:9s} Kopf(matched): 80% {need_stats(nm,'head_matched_S',0.80)} 90% {need_stats(nm,'head_matched_S',0.90)} 99%von inf {need_stats(nm,'head_matched_S',None,0.99)} 95%von inf {need_stats(nm,'head_matched_S',None,0.95)}")
print("\n== Rang-Dosis: gepaart (Mittel ± sd ueber Seeds)")
def pair(a,b,key): 
    d=np.array(getc(a,key))-np.array(getc(b,key)); return f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} (n+={int((d>0).sum())}/10)"
for a,b in [("B2x2_r2","B2x2_r0"),("B4x4_r2","B4x4_r0"),("B4x4_r4","B4x4_r2"),("B4x4_r4","B4x4_r0"),("B8x8_r6","B8x8_r0"),("B16_r2","B16_r0"),("B16_r4","B16_r0"),("B16_r6","B16_r0"),("B16_r8","B16_r0"),("B16_r8","B16_r2"),("D_suche","C_rand"),("A_id","B4x4_r4"),("B4x4_r4","D_suche")]:
    print(f"{a} - {b}: dI_C {pair(a,b,'I_C_test')} | dI_img {pair(a,b,'I_img_test')} | dChern {pair(a,b,'chern_mean')} | dML_inf {pair(a,b,'ml_inf')}")
print("Rang-Reihen I_C: B4x4",[round(np.mean(getc(f'B4x4_r{r}','I_C_test')),3) for r in (0,2,4)],"B16",[round(np.mean(getc(f'B16_r{r}','I_C_test')),3) for r in (0,2,4,6,8)],"B8x8",[round(np.mean(getc(f'B8x8_r{r}','I_C_test')),3) for r in (0,6)],"B2",[round(np.mean(getc(f'B2x2_r{r}','I_C_test')),3) for r in (0,2)])
print("Pearson I_C vs ML_inf ueber alle (Zelle,Seed)-Paare:")
xs=[];ys=[];zs=[]
for r in R:
    for nm,c in r['zellen'].items(): xs.append(c['I_C_test']); ys.append(c['ml_inf']); zs.append(c['chern_mean'])
print(np.corrcoef(xs,ys)[0,1], np.corrcoef(zs,ys)[0,1], np.corrcoef(xs,zs)[0,1])
print("\n== m-Qubit: I(C;X_m) [Bit] bestes (Train gewaehlt, Test gemessen) / Mittel aller Teilmengen")
for nm in ["A_id","B4x4_r4","B16_r8","C_rand","D_suche"]:
    row=[]
    for m in range(1,9):
        if nm=="C_rand":
            b=[np.mean([r['zellen'][f"C_rand{j}"]['mq'][str(m)]['best_test'] for j in range(5)]) for r in R]; a=[np.mean([r['zellen'][f"C_rand{j}"]['mq'][str(m)]['mean_all_train'] for j in range(5)]) for r in R]
        else:
            b=[r['zellen'][nm]['mq'][str(m)]['best_test'] for r in R]; a=[r['zellen'][nm]['mq'][str(m)]['mean_all_train'] for r in R]
        row.append(f"{np.mean(b):.3f}/{np.mean(a):.3f}")
    print(f"{nm:8s} "," ".join(row))
print("Bestes Teilmengen-Muster A_id seed",R[0]['seed'],[R[0]['zellen']['A_id']['mq'][str(m)]['best'] for m in range(1,8)])
# Fano / eta
def h2(p): return 0 if p<=0 or p>=1 else -p*np.log2(p)-(1-p)*np.log2(1-p)
def IF(A,HC=3.32): Pe=1-A; return HC-h2(Pe)-Pe*np.log2(9)
print("\n== Fano/Wirkungsgrad. I_F(A): noetige Information in Bit; S_min=I_F/I_img1;  eta_S = S_min/S_noetig")
for nm in names:
    iimg=np.mean(getc(nm,'I_img_test')); ic=np.mean(getc(nm,'I_C_test'))
    print(f"{nm:9s} I_img1={iimg:.3f} I_C1={ic:.3f}",end=" ")
    for A in (0.8,0.9):
        n,sn=need_stats(nm,'head_matched_S',A) if nm in ["A_id","B4x4_r0","B4x4_r4","B16_r8","C_rand0","D_suche"] else (0,np.nan)
        nm2,sm=need_stats(nm,'ml_acc',A)
        smin=IF(A)/iimg
        print(f"| A={A}: I_F={IF(A):.2f} S_min={smin:.2f} S_ML={sm:.0f}(n={nm2}) eta_ML={smin/sm if sm==sm else float('nan'):.3f} S_head={sn:.0f}(n={n}) eta_head={smin/sn if sn==sn else float('nan'):.3f}",end=" ")
    print()
print("\n== realisierte Information I(C;Chat_S) [Bit]  und je Schuss; ML und Kopf")
for nm in ["A_id","B4x4_r4","B16_r8","D_suche"]:
    print(nm,"ML:",[ (S, round(np.mean(getd(nm,'ml_I',S)),3)) for S in (1,3,10,30,100,1000)],"inf",round(np.mean(getc(nm,'ml_inf_I')),3))
    if nm in ["A_id","B4x4_r4","B16_r8","D_suche"]: print("   Kopf:",[ (S, round(np.mean(getd(nm,'head_matched_I',S)),3)) for S in HS])
