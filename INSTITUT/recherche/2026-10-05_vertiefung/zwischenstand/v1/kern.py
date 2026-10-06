import os, sys, time, json, math
os.environ["OMP_NUM_THREADS"]="2"
sys.path.insert(0,"/home/user/QIRE/INSTITUT")
import numpy as np, torch
torch.set_num_threads(1)
from abteilungen.clifford import daten
from abteilungen.maske_block import matrix_vm, unterraum_aus_qubits
import torch.nn.functional as F

# Pixelindex = 16*Zeile+Spalte; Bits 7..4 Zeile, 3..0 Spalte
# Bloecke: (Zeilenbits, Spaltenbits) innerhalb des Blocks, gepaart j<->j (x^T y)
BLOCK = {"2x2":([4],[0]), "4x4":([4,5],[0,1]), "8x8":([4,5,6],[0,1,2]), "16x16":([4,5,6,7],[0,1,2,3])}
def qubits_of(bits): return [7-b for b in bits]
def V_of(blk):
    r,c=BLOCK[blk]; return unterraum_aus_qubits(qubits_of(r+c))
def bits_of(idx, b): return (idx>>b)&1
def mask_q(idx, form, bitlist):
    """(-1)^q(y), q quadratische Form in den Bits bitlist; form = obere Dreiecksmatrix (0/1)"""
    k=len(bitlist); q=np.zeros(len(idx),int)
    for i in range(k):
        for j in range(i+1,k):
            if form[i,j]: q ^= bits_of(idx,bitlist[i]) & bits_of(idx,bitlist[j])
    return 1-2*q
def gf2_rank(M):
    M=M.copy()%2; r=0; n,m=M.shape
    for c in range(m):
        p=[i for i in range(r,n) if M[i,c]]
        if not p: continue
        M[[r,p[0]]]=M[[p[0],r]]
        for i in range(n):
            if i!=r and M[i,c]: M[i]^=M[r]
        r+=1
    return r
def bent_form(blk):
    r,c=BLOCK[blk]; bl=r+c; k=len(bl); f=np.zeros((k,k),int)
    for j in range(len(r)): f[j,len(r)+j]=1
    return f,bl
def random_form(blk, rng):
    r,c=BLOCK[blk]; bl=r+c; k=len(bl)
    while True:
        U=np.triu(rng.integers(0,2,(k,k)),1); A=(U+U.T)%2
        if gf2_rank(A)==k: return U,bl     # voller Rang k (= bent), wie bent_form
def mask_vec(kind, blk, rng):
    idx=np.arange(256)
    if kind=="keine": return np.ones(256)
    if kind=="zufallszeichen": return rng.choice([-1.0,1.0],256)
    f,bl = bent_form(blk) if kind=="bent" else random_form(blk,rng)
    return mask_q(idx,f,bl).astype(float)
def merkmale(x, V, m):
    W=matrix_vm(V,m); xn=x/x.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return (xn@W.T)**2

def fisher(F_,y):
    mu=F_.mean(0); num=torch.zeros(F_.shape[1]); den=torch.zeros(F_.shape[1])
    for c in range(10):
        s=F_[y==c]; num+=len(s)*(s.mean(0)-mu)**2; den+=len(s)*s.var(0,unbiased=False)
    return num/(den+1e-12)
def topm(F_,y,m): return torch.argsort(fisher(F_,y),descending=True)[:m]

def fit_softmax(A,y,l2,iters=150,tau_bias=None):
    """Softmax auf A (n x d). tau_bias gesetzt: logits = tau*A + b (nur Bias, d=10)."""
    d=A.shape[1]
    if tau_bias is not None:
        b=torch.zeros(10,requires_grad=True); P=[b]
        f=lambda B: tau_bias*B+b
    else:
        W=torch.zeros(d,10,requires_grad=True); b=torch.zeros(10,requires_grad=True); P=[W,b]
        f=lambda B: B@W+b
    opt=torch.optim.LBFGS(P,max_iter=iters,line_search_fn="strong_wolfe",tolerance_grad=1e-7,tolerance_change=1e-10,history_size=15)
    def cl():
        opt.zero_grad(); L=F.cross_entropy(f(A),y)+(l2*(P[0]**2).sum() if tau_bias is None else 0); L.backward(); return L
    opt.step(cl)
    return lambda B: f(B).detach()
def acc(pred_fn,B,y): return float((pred_fn(B).argmax(1)==y).float().mean())

def val_split(n,seed):
    g=torch.Generator().manual_seed(seed+777); p=torch.randperm(n,generator=g); return p[1200:],p[:1200]
L2G=[1e-3,1e-2,1e-1]; TAUG=[5.,30.]
def head_w(Atr,ytr,Ate,yte,seed):
    """Softmax-Kopf, l2 auf Validierung gewaehlt, dann Neufit auf allem."""
    a,v=val_split(len(ytr),seed); best=max(L2G,key=lambda l:acc(fit_softmax(Atr[a],ytr[a],l),Atr[v],ytr[v]))
    return acc(fit_softmax(Atr,ytr,best),Ate,yte),best
def head_bias(Gtr,ytr,Gte,yte,seed):
    a,v=val_split(len(ytr),seed); best=max(TAUG,key=lambda t:acc(fit_softmax(Gtr[a],ytr[a],0,tau_bias=t),Gtr[v],ytr[v]))
    return acc(fit_softmax(Gtr,ytr,0,tau_bias=best),Gte,yte),best
def gruppen_summen(P,grp):
    G=torch.zeros(P.shape[0],10)
    for c in range(10): 
        idx=(grp==c).nonzero().flatten()
        if len(idx): G[:,c]=P[:,idx].sum(1)
    return (10*G).sqrt()
def gruppe_daten(Ftr,ytr):
    """Zuordnung je Ausgang: argmax_c (mu_cj - mu_j)/sd_j  (Trainingsdaten)"""
    mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9
    Z=torch.stack([(Ftr[ytr==c].mean(0)-mu)/sd for c in range(10)],1)
    return Z.argmax(1)
