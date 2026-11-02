import numpy as np, torch
torch.set_num_threads(1)
rng=np.random.default_rng(0)
x=rng.standard_normal(2_000_000)
# Lloyd-Max Gauss je Bitbreite + Entropie
def lloyd(x,k,it=60):
    c=np.quantile(x,(np.arange(k)+.5)/k)
    for _ in range(it):
        b=(c[1:]+c[:-1])/2; i=np.searchsorted(b,x)
        c=np.array([x[i==j].mean() for j in range(k)])
    b=(c[1:]+c[:-1])/2; i=np.searchsorted(b,x)
    mse=((x-c[i])**2).mean(); p=np.bincount(i,minlength=k)/len(x); p=p[p>0]
    return mse,-(p*np.log2(p)).sum()
print("Bit | RD-Grenze D=2^-2R | Lloyd-Max MSE | Entropie H | Luecke dB")
for B in [1,2,3,4]:
    m,H=lloyd(x,2**B); rd=2**(-2*B)
    print(B,round(rd,4),round(m,4),round(H,3),round(10*np.log10(m/rd),2))
# Entropie-codiert einheitlich: gleicher MSE bei R-D => Verlust 0.254 bit bei hoher Rate
# Rate bei gegebenem Lloyd-MSE:  R_rd=0.5*log2(1/mse)
for B in [1,2,3,4]:
    m,H=lloyd(x,2**B); print(B,"R_rd(mse)=",round(0.5*np.log2(1/m),3),"Bit statt",B)
# Wirkung auf eine Schicht: relativer Fehler der Ausgabe y=Wx ist = sqrt(mse)
# Hadamard-Effekt: schwere Ausreisser (Student-t) -> nach Rotation ~Gauss
from scipy.linalg import hadamard
d=256; H=hadamard(d)/np.sqrt(d)
W=rng.standard_t(3,size=(d,d)); W[:, :3]*=10  # Ausreisser-Spalten
def q2(W,B=2):
    s=W.std(axis=1,keepdims=True); z=W/s
    m,_=lloyd(rng.standard_normal(200000),2**B,30)
    k=2**B; c=np.quantile(rng.standard_normal(500000),(np.arange(k)+.5)/k)
    # Lloyd-Zentren neu
    g=rng.standard_normal(500000)
    for _ in range(30):
        b=(c[1:]+c[:-1])/2; i=np.searchsorted(b,g); c=np.array([g[i==j].mean() for j in range(k)])
    b=(c[1:]+c[:-1])/2; i=np.searchsorted(b,z); return c[i]*s
for name,A in [("ohne Rotation",W),("mit Hadamard",W@H)]:
    Q=q2(A); print(name,"rel. Fehler^2",round(((A-Q)**2).sum()/(A**2).sum(),4))
