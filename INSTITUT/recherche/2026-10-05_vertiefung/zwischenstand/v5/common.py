import os
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
def cnot_perm(n,c,t):
    idx=np.arange(2**n); cm,tm=1<<(n-1-c),1<<(n-1-t)
    return np.where((idx&cm)!=0, idx^tm, idx)
def ring_perm(n):
    # komponiere: wie butterfly_layer.komponiere -- Reihenfolge wird unten gegen Adapter geprueft
    p=np.arange(2**n)
    for q in range(n):
        p=p[cnot_perm(n,q,(q+1)%n)]  # Variante A
    return p
def ry_gen(n,q):
    """Erzeuger Y_q als antisymmetrische Matrix: d/dθ der Rotation [[c,-s],[s,c]] bei 0."""
    N=2**n; G=np.zeros((N,N))
    for x in range(N):
        if (x>>(n-1-q))&1==0:
            y=x|(1<<(n-1-q))
            G[y,x]+=1; G[x,y]-=1   # a'=c a - s b ; b'= s a + c b  -> d: (a,b)->(-b,a): row x gets -b, row y gets +a
    return G
def order(p):
    n=len(p); q=p.copy(); k=1
    while not np.array_equal(q,np.arange(n)):
        q=q[p]; k+=1
    return k
