import os, sys, json, time
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"/home/user/QIRE/INSTITUT")
import numpy as np, torch
torch.set_num_threads(1)
from abteilungen.clifford import daten
from abteilungen.clifford_suche import kopf_logistisch, normiere, born
from abteilungen.maske_block import matrix_vm, unterraum_aus_qubits, rref_basis
from abteilungen.maske_eigenschaft import maske_aus_form, zufall_form, rang_gf2

VS={"block2x2":[3,7],"block4x4":[2,3,6,7],"block8x8":[1,2,3,5,6,7],"global8":list(range(8))}
NF=8; L2=0.003
def hyp(k,r):
    B=np.zeros((k,k),np.int64)
    for i in range(r//2): B[2*i,2*i+1]=B[2*i+1,2*i]=1
    return B
def kanon(k):   # x^T y Paarung: i <-> i+k/2
    B=np.zeros((k,k),np.int64)
    for i in range(k//2): B[i,i+k//2]=B[i+k//2,i]=1
    return B
def adj(k): return hyp(k,k)

def lauf(a):
    ds,s=a
    xtr,ytr,xte,yte=daten(ds,s)
    xa,xb=normiere(xtr),normiere(xte)
    def ev(W): return kopf_logistisch(born(xa,W),ytr,born(xb,W),yte,l2=L2)[0]
    t0=time.time()
    z={"ds":ds,"seed":s}
    z["identitaet"]=ev(torch.eye(256))
    rng=np.random.default_rng(920000+s)
    for name,q in VS.items():
        V=unterraum_aus_qubits(q); k=len(q)
        r={"k":k,"keine":ev(matrix_vm(V,np.ones(256)))}
        r["rang"]={}
        for rk in range(2,k+1,2):
            B0=hyp(k,rk); accs=[]
            n=1 if rk==k and k==2 else NF
            for _ in range(n):
                B=zufall_form(B0,rng); assert rang_gf2(B)==rk
                accs.append(ev(matrix_vm(V,maske_aus_form(B,V))))
            r["rang"][rk]=accs
        r["bent_kanon"]=ev(matrix_vm(V,maske_aus_form(kanon(k),V)))
        r["bent_adj"]=ev(matrix_vm(V,maske_aus_form(adj(k),V))) if k>2 else r["bent_kanon"]
        z[name]=r
    z["sek"]=time.time()-t0
    return z
if __name__=="__main__":
    ds,s=sys.argv[1],int(sys.argv[2])
    z=lauf((ds,s)); print(json.dumps(z)[:1500]); print(z["sek"])
