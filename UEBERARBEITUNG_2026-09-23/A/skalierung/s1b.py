"""S1b: Gradientenvarianz Var d<Z_0>/dtheta fuer CNOT-Varianten x Eingaben.
V0 Ring nach jeder Lage | V1 Ring nur zwischen den Lagen | V2 Kette | V3 Kette zwischen den Lagen.
Parameter-Shift (Ry exakt). 1 Thread. Gemessen werden zwei Parameter: Ry q0 der ersten Lage ('erst')
und Ry q0 der letzten Lage ('letzt'; bei L=1 identisch)."""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import sys, math, time, json
sys.path.insert(0,"/home/user/QIRE/UEBERARBEITUNG_2026-09-23/A")
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(1)
from quantum_sim.butterfly_layer import ButterflyQuantumLayer, apply_1q, cnot_permutation
T0=time.process_time()
MN=np.load("/home/user/QIRE/UEBERARBEITUNG_2026-09-23/A/results/datasets/mnist.npz")["x"].astype(np.float32)
VAR={"V0":("ring",False),"V1":("ring",True),"V2":("kette",False),"V3":("kette",True)}
def perms(n,typ):
    pairs=[(q,(q+1)%n) for q in range(n)] if typ=="ring" else [(q,q+1) for q in range(n-1)]
    return [cnot_permutation(n,c,t) for c,t in pairs]
def z0(lay,x,P,skip_last):
    n=lay.n; s=x.to(torch.complex64); s=s/s.norm(dim=1,keepdim=True)
    for l in range(lay.L):
        for q in range(n): s=apply_1q(s,lay._gate(l,q),q,n)
        if not(skip_last and l==lay.L-1):
            for p in P: s=s[:,p]
    pr=(s.real**2+s.imag**2).reshape(-1,2,2**(n-1))
    return float((pr[:,0].sum()-pr[:,1].sum()))
def img(n,idx):
    h=2**((n+1)//2); w=2**(n//2)
    t=torch.from_numpy(MN[idx]).reshape(1,1,28,28)
    t=F.interpolate(t,size=(h,w),mode="area" if h*w<=784 and h<=28 and w<=28 else "bilinear",
                    **({} if h<=28 and w<=28 else {"align_corners":False}))
    return t.reshape(1,-1)
def grads(var,eingabe,n,L,R=200):
    typ,skip=VAR[var]; P=perms(n,typ)
    g=torch.Generator().manual_seed(1000*n+L)   # gleiche theta/x fuer alle Varianten (gepaart)
    rng=np.random.RandomState(1000*n+L); ids=rng.randint(0,len(MN),R)
    lay=ButterflyQuantumLayer(n,L).eval(); out={"erst":[],"letzt":[]}
    with torch.no_grad():
        for r in range(R):
            th=torch.rand(L,n,2,generator=g)*2*math.pi
            xr=torch.randn(1,2**n,generator=g)
            x=xr if eingabe=="zufall" else img(n,ids[r])
            for nm,w in (("erst",(0,0,0)),("letzt",(L-1,0,0))):
                def f(s):
                    t=th.clone(); t[w]+=s; lay.theta.copy_(t); return z0(lay,x,P,skip)
                out[nm].append((f(math.pi/2)-f(-math.pi/2))/2)
    return {k:np.array(v) for k,v in out.items()}
def fit(ns,vs): return -np.polyfit(ns,np.log2(vs),1)[0]
if __name__=="__main__":
    NS=list(range(2,11)); res={}; brng=np.random.RandomState(0); B=1000
    lines=[]
    for ein in ("zufall","mnist"):
      for var in VAR:
        for L in (1,3,5):
          G={int(n):grads(var,ein,int(n),L) for n in NS}
          for nm in ("erst","letzt"):
            if L==1 and nm=="letzt": continue
            vs=np.array([G[n][nm].var(ddof=1) for n in NS]); vs=np.maximum(vs,1e-30)
            a=fit(NS,vs); aa=[]
            for _ in range(B):
                v=[]
                for n in NS:
                    s=G[n][nm]; v.append(max(s[brng.randint(0,len(s),len(s))].var(ddof=1),1e-30))
                aa.append(fit(NS,np.array(v)))
            lo,hi=np.percentile(aa,[2.5,97.5])
            nz=np.mean([np.mean(np.abs(G[n][nm])<1e-6) for n in NS]); nz10=np.mean(np.abs(G[10][nm])<1e-6)
            key=f"{ein}|{var}|L{L}|{nm}"
            res[key]=dict(a=a,lo=lo,hi=hi,v8=vs[6],v10=vs[8],nullanteil_n10=nz10,nullanteil_alle_n=nz,vs=vs.tolist())
            print(f"{key}: a={a:.2f} [{lo:.2f},{hi:.2f}] Var8={vs[6]:.2e} Var10={vs[8]:.2e} null(n=10)={nz10:.2f} null(n2..10)={nz:.2f} | CPU {time.process_time()-T0:.0f}s",flush=True)
    json.dump(res,open(os.path.join(os.path.dirname(__file__),"s1b_ergebnis.json"),"w"),indent=1)
