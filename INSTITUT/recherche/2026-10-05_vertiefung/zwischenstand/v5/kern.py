import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import sys, types
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0,'/home/user/QIRE/UEBERARBEITUNG_2026-09-23/A')
sys.modules.setdefault('psutil',types.ModuleType('psutil'))
from quantum_sim import chatbot_adapter as ca
from quantum_sim.butterfly_layer import cnot_permutation
from quantum_sim.butterfly_fast import komponiere

def ring(n, L):
    p = komponiere([cnot_permutation(n, q, (q+1) % n) for q in range(n)])
    return p, torch.argsort(komponiere([p]*L))

def kette(h, theta, n, perm, rueck):
    """wie ca._kette, aber n frei. h (K,Z,N), theta (K,L,n)."""
    k = h.shape[0]; N = 2**n
    c, s = torch.cos(theta).to(h.dtype), torch.sin(theta).to(h.dtype)
    for lage in range(theta.shape[1]):
        for q in range(n):
            v = h.reshape(k, -1, 2**q, 2, 2**(n-1-q))
            a, b = v[:, :, :, 0, :], v[:, :, :, 1, :]
            cq, sq = c[:, lage, q].view(k,1,1,1), s[:, lage, q].view(k,1,1,1)
            h = torch.stack([cq*a - sq*b, sq*a + cq*b], dim=3).reshape(k, -1, N)
        h = h[..., perm]
    return h[..., rueck]

def Rt(theta, n):
    """R^T (N,N) fuer theta (L,n) -- wie Drehbank mit K=1."""
    L = theta.shape[0]; N = 2**n
    perm, rueck = ring(n, L)
    eye = torch.eye(N, dtype=theta.dtype)[None]
    return kette(eye, theta[None], n, perm, rueck)[0]

if __name__ == "__main__":
    # Verifikation gegen den Adapter-Code (n=8)
    torch.manual_seed(0)
    th = torch.randn(3, 8, dtype=torch.float64)
    p, r = ca._ring(3, True)
    eye = torch.eye(256, dtype=torch.float64)[None]
    ref = ca._kette(eye, th[None], p, r)[0]
    mine = Rt(th, 8)
    print("max Abw. gegen chatbot_adapter._kette:", float((ref-mine).abs().max()))
    print("orthogonal:", float((mine@mine.T-torch.eye(256,dtype=torch.float64)).abs().max()))
