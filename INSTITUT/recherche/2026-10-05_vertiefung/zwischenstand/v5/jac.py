import sys, time, json
from kern import *
from torch.func import jacfwd

def jac_rank(n, L, r=None, seed=0, chunk=64):
    """Rang der Jacobi-Matrix theta (L*n) -> R^T X (X = Einheit oder r Zufallsvektoren)."""
    N = 2**n
    g = torch.Generator().manual_seed(seed)
    th0 = (torch.rand(L, n, generator=g, dtype=torch.float64) * 2*np.pi).reshape(-1)
    perm, rueck = ring(n, L)
    X = torch.eye(N, dtype=torch.float64) if r is None else torch.randn(N, r, generator=g, dtype=torch.float64)
    h0 = X.T[None]                                  # (1, Z, N) Zeilen
    f = lambda t: kette(h0, t.reshape(1, L, n), n, perm, rueck).reshape(-1)
    from torch.func import jvp, vmap
    c = L*n; cols = []
    eye = torch.eye(c, dtype=torch.float64)
    for a in range(0, c, chunk):
        V = eye[a:a+chunk]
        cols.append(vmap(lambda v: jvp(f, (th0,), (v,))[1])(V))   # (b, Z*N)
    J = torch.cat(cols).T                           # (Z*N, L*n)
    s = torch.linalg.svdvals(J)
    return s

def rang(s, rel=1e-9):
    return int((s > s[0]*rel).sum())

if __name__ == "__main__":
    n = int(sys.argv[1]); Ls = [int(x) for x in sys.argv[2].split(',')]
    r = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != 'full' else None
    out = {}
    for L in Ls:
        t = time.time()
        s = jac_rank(n, L, r)
        k = rang(s)
        # Luecke im Spektrum: kleinster gezaehlter / groesster verworfener
        sl = float(s[k-1]/s[0]); sd = float(s[k]/s[0]) if k < len(s) else 0.0
        out[L] = dict(params=n*L, rank=k, smin_rel=sl, next_rel=sd)
        print(f"n={n} L={L}: Parameter {n*L}, Rang {k}, kleinster gez. sv/s0={sl:.2e}, naechster={sd:.2e}  ({time.time()-t:.1f}s)", flush=True)
    json.dump(out, open(f"jac_n{n}_{'full' if r is None else r}.json","w"))
