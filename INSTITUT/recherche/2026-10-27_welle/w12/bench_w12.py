"""w12: Mikrobenchmark Batch 1, 1 Thread: Sz-FFN (echte FWHT) gegen dichte FFN. Kein Training."""
import sys, time, statistics, math, torch
torch.set_num_threads(1)
sys.path.insert(0, '/home/user/QIRE/INSTITUT')
from abteilungen.struktur_sprache import FFNStruktur

def fwht(x):  # x: (m,D), D Zweierpotenz, ohne Normierung, log2 D Stufen
    m, D = x.shape; h = 1
    while h < D:
        y = x.view(m, D // (2 * h), 2, h)
        x = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2).view(m, D)
        h *= 2
    return x

class FF:  # eine Fastfood-Matrix, echte FWHT
    def __init__(s, din, N, g):
        s.D = 1 << (din - 1).bit_length(); s.m = max(1, N // s.D); s.N = N; s.din = din
        s.d1 = torch.randn(s.m, s.D); s.d2 = torch.randn(s.m, s.D); s.b = torch.zeros(N)
        s.idx = torch.randperm(s.D)[:N] if N < s.D else None
    def __call__(s, x):
        if s.din < s.D: x = torch.nn.functional.pad(x, (0, s.D - s.din))
        z = fwht(x * s.d1)            # (m,D)
        z = fwht(z * s.d2).reshape(1, -1)
        if s.idx is not None: z = z[:, s.idx]
        return z + s.b

def sz_real(d):
    a, b = FF(d, 4 * d, 0), FF(4 * d, d, 0)
    return lambda x: b(torch.square(a(x)))

def dicht(d, h):
    l1, l2 = torch.nn.Linear(d, h), torch.nn.Linear(h, d)
    return lambda x: l2(torch.square(l1(x)))

def zeit(f, x, n=2000, warm=200):
    for _ in range(warm): f(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); f(x); t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e6

@torch.no_grad()
def main():
    torch.manual_seed(0)
    # Korrektheit: echte FWHT gegen Matmul-Fastfood
    for d in (256, 512):
        m = FFNStruktur(d, 4 * d, "ff", torch.square, torch.Generator().manual_seed(0)).eval()
        x = torch.randn(1, d); print(f"d={d} Sz as implemented (Repo, Matmul-WHT): {zeit(m, x, 1000):.1f} us")
        z = FF(d, 4 * d, 0); y = torch.randn(1, d)
        H = torch.tensor([[1.]]);
        while H.shape[0] < z.D: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        v = torch.randn(1, z.D); print("  FWHT-Test max.Fehler:", float((fwht(v) - v @ H).abs().max()))
        print(f"d={d} Sz echte FWHT:            {zeit(sz_real(d), x):.1f} us")
        for h in (388 if d == 256 else 580, 4 * d, 2048):
            print(f"d={d} dicht h={h}:           {zeit(dicht(d, h), x):.1f} us  ({2*d*h*4/1e6:.2f} MB Gewichte)")
main()
