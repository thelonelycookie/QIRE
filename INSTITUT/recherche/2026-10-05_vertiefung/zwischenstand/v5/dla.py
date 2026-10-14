import sys, time, json
from kern import *
from torch.func import jacfwd

def order_perm(p):
    p = p.numpy(); n=len(p); q=p.copy(); k=1; idn=np.arange(n)
    while not np.array_equal(q,idn): q=q[p]; k+=1
    return k

def gens(n, L):
    """Tangenten d R^T/d theta_{l,q} bei theta=0, als antisymmetrische (N,N); Liste der Laenge L*n."""
    f = lambda t: Rt(t.reshape(L,n), n)
    J = jacfwd(f)(torch.zeros(L*n, dtype=torch.float64))  # (N,N,L*n)
    G = J.permute(2,0,1).numpy()
    assert np.abs(G+G.transpose(0,2,1)).max() < 1e-12
    return G

def closure(G, tol=1e-9, dimmax=None, log=None):
    g, N, _ = G.shape
    iu = np.triu_indices(N, 1)
    vec = lambda M: M[..., iu[0], iu[1]]
    full = N*(N-1)//2
    Q = np.zeros((full, full)) if full <= 9000 else None
    Qm = []  # Zeilen
    Q = np.zeros((min(full,dimmax or full)+8, full))
    d = 0
    def add(C):
        nonlocal d
        newidx = []
        for _ in range(2):
            if d: C = C - (C @ Q[:d].T) @ Q[:d]
        for row in C:
            r = row.copy()
            for _ in range(2):
                if d: r = r - (Q[:d] @ r) @ Q[:d]
            nr = np.linalg.norm(r)
            if nr > tol:
                Q[d] = r/nr; d += 1; newidx.append(d-1)
        return newidx
    def unvec(v):
        M = np.zeros((N,N)); M[iu] = v; return M - M.T
    # Start: Generatoren
    V = vec(G); V = V/np.linalg.norm(V,axis=1,keepdims=True)
    new = add(V.copy())
    frontier = [unvec(Q[i]) for i in new]
    lvl = 0
    while frontier and d < full:
        lvl += 1
        Cs = []
        F = np.array(frontier)           # (f,N,N)
        for gi in range(g):
            Cs.append(vec(np.einsum('ij,fjk->fik', G[gi], F) - np.einsum('fij,jk->fik', F, G[gi])))
        C = np.concatenate(Cs)
        nrm = np.linalg.norm(C, axis=1)
        C = C[nrm > tol]
        before = d
        # in Bloecken, damit Speicher klein bleibt
        newall = []
        for s in range(0, len(C), 4000):
            newall += add(C[s:s+4000])
        frontier = [unvec(Q[i]) for i in newall]
        if log: log(f"  Stufe {lvl}: dim {before} -> {d}")
    return d

if __name__ == "__main__":
    n = int(sys.argv[1]); Lmax = int(sys.argv[2])
    p,_ = ring(n,1)
    print(f"n={n}: Ordnung der CNOT-Ring-Permutation P: {order_perm(p)}, dim so(2^n)={2**(n-1)*(2**n-1)}", flush=True)
    res = {}
    for L in range(1, Lmax+1):
        t=time.time()
        G = gens(n, L)
        d = closure(G)
        res[L]=d
        print(f"n={n} L={L}: dim DLA_L = {d}  ({time.time()-t:.1f}s)", flush=True)
        if d == 2**(n-1)*(2**n-1): break
    json.dump(res, open(f"dla_n{n}.json","w"))
