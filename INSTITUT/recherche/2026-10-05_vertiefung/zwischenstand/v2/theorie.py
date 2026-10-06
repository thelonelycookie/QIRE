import numpy as np, scipy.linalg as sl, itertools
rng=np.random.default_rng(5)
def rang(B):
    B=B.copy()%2; r=0
    for c in range(B.shape[1]):
        p=[i for i in range(r,len(B)) if B[i,c]]
        if not p: continue
        B[[r,p[0]]]=B[[p[0],r]]
        for i in range(len(B)):
            if i!=r and B[i,c]: B[i]^=B[r]
        r+=1
    return r
bad=0;tot=0;stat={}
for n in (4,6,8):
  H=sl.hadamard(2**n); X=((np.arange(2**n)[:,None]>>np.arange(n))&1)
  for _ in range(300):
    U=np.triu(rng.integers(0,2,(n,n)))
    if rng.random()<.5: U=np.triu(U,1)*(rng.random()<.5)   # niedrigere Raenge
    lin=rng.integers(0,2,n)
    q=(np.einsum('xi,ij,xj->x',X,U,X)+X@lin)%2
    f=(-1.0)**q; W=H@f
    r=rang((U+U.T)%2); h=r//2
    vals=np.unique(np.abs(W).astype(int)); supp=(np.abs(W)>0).sum()
    ok=(set(vals)-{0}=={2**(n-h)}) and supp==2**(2*h)
    # Verschiebung der Born-Merkmale: Amplitude = 2^-h * Summe ueber 2^2h Walsh-Koeff.
    tot+=1; bad+=not ok; stat[(n,r)]=stat.get((n,r),0)+1
print("Satz geprueft an",tot,"zufaelligen quadratischen Funktionen (n=4,6,8, mit linearem Anteil), Fehler:",bad); print(sorted(stat.items()))
# Merkmalsidentitaet: feature_j = (H(s*p))_j^2 = 2^-2h (sum_{a in Traeger} eps * c_{j+a})^2 , c=Hp/sqrt?
n=6; Hn=sl.hadamard(2**n)/2**(n/2); X=((np.arange(2**n)[:,None]>>np.arange(n))&1)
for kk in (0,2,4,6):
    B=np.zeros((n,n),int)
    for i in range(kk//2): B[2*i,2*i+1]=1
    q=(X@np.triu(B)*X).sum(1)%2; s=(-1.0)**q
    p=rng.random(2**n); c=Hn@p; feat=(Hn@(s*p))**2
    sh=sl.hadamard(2**n)@s/2**n*2**n   # ungeschwaecht
    sup=np.flatnonzero(np.abs(sh)>0); h=kk//2
    j=7; amp=sum(sh[a]/2**(n-h)*0 for a in sup)  # Platzhalter
    # Amplitude_j = sum_a  shat(a)/2^n * c(j^a)  (Faltung), shat(a)=sh[a]
    amp=sum(sh[a]/2**n*c[j^a] for a in sup)
    print("Rang",kk,"Traeger",len(sup),"=",2**kk,"; Faltung stimmt:",np.isclose(amp**2,feat[j]),"; Gewicht je Term +-",sh[sup[0]]/2**n if len(sup) else None,"= 2^-h:",2.0**-h)
# Bent: Summe der Diagonal-Terme konstant
B=np.zeros((n,n),int)
for i in range(3): B[2*i,2*i+1]=1
q=(X@np.triu(B)*X).sum(1)%2; s=(-1.0)**q
p=rng.random(2**n); c=Hn@p; feat=(Hn@(s*p))**2
print("bent: Diagonalterm sum c^2/2^n =",(c**2).sum()/2**n," Mittel der Merkmale",feat.mean(),"(gleich: Parseval)")
