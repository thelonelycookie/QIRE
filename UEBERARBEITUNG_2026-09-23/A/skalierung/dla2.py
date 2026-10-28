import os
os.environ["OMP_NUM_THREADS"]="1"
import sys, math, time
sys.path.insert(0,"/home/user/QIRE/UEBERARBEITUNG_2026-09-23/A")
import numpy as np, torch
torch.set_num_threads(1)
from quantum_sim.butterfly_layer import ButterflyQuantumLayer
T0=time.process_time()
def gv(n,L,which,inp,R=200):
    g=torch.Generator().manual_seed(7+n*10+L); lay=ButterflyQuantumLayer(n,L).eval(); out=[]
    with torch.no_grad():
        for _ in range(R):
            th=torch.rand(L,n,2,generator=g)*2*math.pi
            x=torch.randn(1,2**n,generator=g) if inp=="zufall" else torch.eye(1,2**n)
            def f(s):
                t=th.clone(); t[which]+=s; lay.theta.copy_(t); return float(lay(x)[0,0])
            out.append((f(math.pi/2)-f(-math.pi/2))/2)
    return np.var(out,ddof=1)
for inp in ("zufall","|0..0>"):
  for L in (1,3,5):
    for nm,w in (("erste Lage Ry q1",(0,1,0)),("letzte Lage Ry q1",(L-1,1,0))):
        if L==1 and nm.startswith("letzte"): continue
        ns=list(range(2,11)); vs=np.array([gv(n,L,w,inp) for n in ns]); ns=np.array(ns)
        s,_=np.polyfit(ns,np.log2(vs),1); s5,_=np.polyfit(ns[3:],np.log2(vs[3:]),1)
        print(f"{inp} L={L} {nm}: "+" ".join(f"{n}:{v:.3e}" for n,v in zip(ns,vs))+f" | a(2..10)={-s:.3f} a(5..10)={-s5:.3f}",flush=True)
print("CPU s",time.process_time()-T0)
