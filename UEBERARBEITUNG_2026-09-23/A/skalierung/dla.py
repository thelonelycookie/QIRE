import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import sys, time, math, itertools
sys.path.insert(0,"/home/user/QIRE/UEBERARBEITUNG_2026-09-23/A")
import numpy as np, torch
torch.set_num_threads(1)
T0=time.process_time()

# ---- DLA via Pauli strings: (x,z) bitmasks ----
def cnot_conj(p,c,t):
    x,z=p
    if (x>>c)&1: x^=(1<<t)
    if (z>>t)&1: z^=(1<<c)
    return (x,z)
def ring(p,n):
    for q in range(n): p=cnot_conj(p,q,(q+1)%n)
    return p
def anti(a,b):
    return (bin(a[0]&b[1]).count("1")+bin(a[1]&b[0]).count("1"))&1
def mul(a,b): return (a[0]^b[0],a[1]^b[1])
def closure(gens):
    S=set(gens); 
    changed=True
    while changed:
        changed=False
        L=list(S)
        for i in range(len(L)):
            for j in range(i+1,len(L)):
                if anti(L[i],L[j]):
                    m=mul(L[i],L[j])
                    if m not in S: S.add(m); changed=True
    return S
def gens(n,ls):
    g=set()
    for l in ls:
        for q in range(n):
            for p in ((1<<q,1<<q),(0,1<<q)):   # Y=(x,z both), Z=(0,z)
                for _ in range(l): p=ring(p,n)
                g.add(p)
                p=((1<<q,1<<q)) if p is None else p
    return g
def gens2(n,ls):
    g=set()
    for l in ls:
        for q in range(n):
            for base in ((1<<q,1<<q),(0,1<<q)):
                p=base
                for _ in range(l): p=ring(p,n)
                g.add(p)
    return g
def order(n):
    p0=[(1<<q,0) for q in range(n)]+[(0,1<<q) for q in range(n)]
    p=list(p0);k=0
    while True:
        p=[ring(x,n) for x in p];k+=1
        if p==p0: return k
print("== DLA (Pauli-Abschluss) ==")
res={}
for n in range(2,7):
    o=order(n); row=[]
    for L in range(1,min(o,8)+1):
        d=len(closure(gens2(n,range(1,L+1))))
        row.append(d)
    res[n]=(o,row)
    d_all=len(closure(gens2(n,range(1,o+1))))
    print(f"n={n} Ordnung des Rings={o} dim(DLA) L=1..{len(row)}: {row} | alle Potenzen: {d_all} | su(2^n): {4**n-1} | 3n={3*n}",flush=True)

# ---- Gradientenvarianz ----
from quantum_sim.butterfly_layer import ButterflyQuantumLayer
def gradvar(n,L,R=200,seed=0,which=(0,0,0)):
    g=torch.Generator().manual_seed(seed*1000+n*10+L)
    lay=ButterflyQuantumLayer(n,L).eval()
    out=[]
    with torch.no_grad():
        for _ in range(R):
            th=torch.rand(L,n,2,generator=g)*2*math.pi
            x=torch.randn(1,2**n,generator=g)
            def f(sh):
                t=th.clone(); t[which]+=sh
                lay.theta.copy_(t); return float(lay(x)[0,0])
            out.append((f(math.pi/2)-f(-math.pi/2))/2)  # Parameter-Shift, exakt fuer Ry/Rz
    a=np.array(out); return a.var(ddof=1), a.mean()
print("== Gradientenvarianz Var d<Z0>/dtheta ==")
tab={}
for L in (1,3,5):
    for which,name in (((0,0,0),"erste Lage Ry q0"),((L-1,0,0),"letzte Lage Ry q0")):
        ns=[];vs=[]
        for n in range(2,11):
            v,m=gradvar(n,L,which=which); ns.append(n);vs.append(v)
            tab[(L,name,n)]=v
        ns=np.array(ns);vs=np.array(vs)
        sl,ic=np.polyfit(ns,np.log2(vs),1)
        sl2,_=np.polyfit(ns[ns>=5],np.log2(vs[ns>=5]),1)
        print(f"L={L} {name}: "+" ".join(f"{n}:{v:.3e}" for n,v in zip(ns,vs))+f" | Steigung a(n=2..10)={-sl:.3f}, a(n>=5)={-sl2:.3f}",flush=True)
print("CPU s:",time.process_time()-T0)
