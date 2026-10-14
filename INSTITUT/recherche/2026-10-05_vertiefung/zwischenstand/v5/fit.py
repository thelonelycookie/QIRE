import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(970)
B0=2.5616; Ls_=np.array([1,2,5,10.]); Y=np.array([2.5174,2.4964,2.4442,2.4115]); G=B0-Y
LSTAR=4080; SE=0.001   # SE des Seed-Mittels: sd ~0,0004-0,0029 / sqrt(8)
LORA=2.2875
def power(p,L): return p[0]*L**p[1]
def sqrtlaw(p,L): return p[0]*np.sqrt(L)
def loglin(p,L): return p[0]*np.log1p(L/p[1])     # gain = d*ln(1+L/L0)
def lomax(E):
    def f(p,L):
        Ls,al=np.exp(p[0]),np.exp(p[1]); return (B0-E)*(1-(1+L/Ls)**(-al))
    return f
def fit(f,p0,y=G,lb=-np.inf,ub=np.inf):
    r=least_squares(lambda p:f(p,Ls_)-y,p0,bounds=(lb,ub)); return r.x,float((r.fun**2).sum())
out={}
def row(name,f,p0,lb=-np.inf,ub=np.inf,cap=False):
    p,rss=fit(f,p0,lb=lb,ub=ub)
    pred=lambda p,L:B0-f(p,np.asarray(L,float))
    bs=[]
    for _ in range(400):
        yb=G+rng.normal(0,SE,4)
        try:
            pb,_=fit(f,p,yb,lb,ub); bs.append(pred(pb,[20,40,100,LSTAR]))
        except Exception: pass
    bs=np.array(bs); lo,hi=np.percentile(bs,[5,95],axis=0)
    pr=pred(p,[20,40,100,LSTAR])
    print(f"{name:34s} RSS {rss:.1e} | L=20 {pr[0]:.3f} [{lo[0]:.3f},{hi[0]:.3f}] | L=40 {pr[1]:.3f} [{lo[1]:.3f},{hi[1]:.3f}] | L=100 {pr[2]:.3f} [{lo[2]:.3f},{hi[2]:.3f}] | L*=4080 {pr[3]:.3f}  p={np.round(p,3)}")
    return p
print("Basis",B0,"| LoRA",LORA,"| Messpunkte",Y)
p=row("Potenzgesetz Gewinn a*L^b",power,[0.045,0.5])
print("   -> L bei dem Potenzgesetz LoRA erreicht:", (G_:=B0-LORA)/p[0], "^(1/b) =", (G_/p[0])**(1/p[1]))
row("Wurzelgesetz a*sqrt(L) (b=0,5)",sqrtlaw,[0.045])
row("log: d*ln(1+L/L0)",loglin,[0.05,1.0],lb=[0,1e-3])
# Potenzgesetz mit Cap: Verlust darf bei L* nicht unter Boden E fallen -> b <= b_max(E)
for E in [0.0,1.5,2.0]:
    bmax=np.log((B0-E)/ p[0])/np.log(LSTAR)
    print(f"Potenz mit Boden E={E}: b <= {bmax:.3f}  (frei gefitteter b={p[1]:.3f})")
    pe,rss=fit(power,[0.045,min(0.5,bmax)],lb=[0,0],ub=[1,bmax]); print(f"    Fit mit b<=bmax: a={pe[0]:.4f} b={pe[1]:.3f} RSS {rss:.1e}; L=20 {B0-power(pe,20):.3f} L=40 {B0-power(pe,40):.3f}")
# Lomax mit Boden E und Plateau bei L* (98 % des Gewinns schon bei L*)
for E in [1.5,2.0,2.2,2.3,2.35]:
    f=lomax(E)
    # Randbedingung ueber Penalty: (1+L*/Ls)^-al <= 0.02
    best=None
    for s0 in [0,3,6]:
        for a0 in [-1.5,-0.5,0.5]:
            try:
                r=least_squares(lambda p:np.concatenate([f(p,Ls_)-G, [max(0,((1+LSTAR/np.exp(p[0]))**(-np.exp(p[1]))-0.02))*50]]),[s0,a0])
                if best is None or r.cost<best.cost: best=r
            except Exception: pass
    p=best.x; rss=float(((f(p,Ls_)-G)**2).sum()); rest=(1+LSTAR/np.exp(p[0]))**(-np.exp(p[1]))
    pr=B0-f(p,np.array([20,40,100,LSTAR]))
    print(f"Lomax E={E}: Ls={np.exp(p[0]):.3g} alpha={np.exp(p[1]):.3g} Rest(L*)={rest:.3f} RSS {rss:.1e} | L=20 {pr[0]:.3f} L=40 {pr[1]:.3f} L=100 {pr[2]:.3f} L*=4080 {pr[3]:.3f}")
