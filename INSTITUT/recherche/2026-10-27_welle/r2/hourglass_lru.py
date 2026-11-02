"""r2: Hourglass (Pooling x4, kausal verschoben) und LRU-Rekurrenz im Byte-LM (Erkundung, Tuning-Seeds 975-977).
Aufruf: python3 hourglass_lru.py zaehlen | zeit | lauf <arm> <seed> <schritte> <lr> [val] | """
import sys, math, time, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
WURZEL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WURZEL))
from abteilungen.struktur_sprache import daten

D, KOEPFE, CTX, BATCH, K = 128, 4, 256, 8, 4
HF = 4 * D


class Block(nn.Module):
    def __init__(self, lru=False):
        super().__init__()
        self.lru = lru
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        if lru:
            N = D
            self.N = N
            r = torch.empty(N).uniform_(0.9, 0.999)
            self.nu = nn.Parameter(torch.log(-torch.log(r)))
            self.th = nn.Parameter(torch.log(torch.empty(N).uniform_(0, 2 * math.pi / 10) + 1e-4))
            self.bre, self.bim = nn.Parameter(torch.randn(N, D) / math.sqrt(2 * D)), nn.Parameter(torch.randn(N, D) / math.sqrt(2 * D))
            self.cre, self.cim = nn.Parameter(torch.randn(D, N) / math.sqrt(N)), nn.Parameter(torch.randn(D, N) / math.sqrt(N))
        else:
            self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.f1, self.f2 = nn.Linear(D, HF), nn.Linear(HF, D)

    def mixer(self, x):
        B, T, C = x.shape
        if not self.lru:
            q, k, v = self.qkv(x).reshape(B, T, 3, KOEPFE, C // KOEPFE).permute(2, 0, 3, 1, 4)
            return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        lam_log = torch.complex(-torch.exp(self.nu), torch.exp(self.th))          # log lambda, |lambda|<1
        gam = torch.sqrt(1 - torch.exp(2 * lam_log.real))
        u = torch.complex(x @ self.bre.T, x @ self.bim.T) * gam                   # (B,T,N)
        tau = torch.arange(T, dtype=torch.float32)[:, None]
        Kern = torch.exp(tau * lam_log[None, :])                                  # (T,N) = lambda^tau
        h = torch.fft.ifft(torch.fft.fft(u, n=2 * T, dim=1) * torch.fft.fft(Kern, n=2 * T, dim=0)[None], dim=1)[:, :T]
        return h.real @ self.cre.T - h.imag @ self.cim.T

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))


class LM(nn.Module):
    """hg=False: flaches Modell mit n Blöcken. hg=True: 1 Vorblock (voll), Pooling x4, n_mid Blöcke auf T/4, Upsampling + Skip, 1 Nachblock.
    lru0: Block 0 (Vorblock) nutzt LRU statt Attention."""
    def __init__(self, n=4, hg=False, n_mid=4, lru0=False):
        super().__init__()
        self.hg = hg
        self.emb, self.pos = nn.Embedding(256, D), nn.Embedding(CTX, D)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        if hg:
            self.pre, self.post = Block(lru0), Block()
            self.mid = nn.ModuleList([Block() for _ in range(n_mid)])
            self.up = nn.Linear(D, D)
            self.lnm = nn.LayerNorm(D)
        else:
            self.bl = nn.ModuleList([Block(lru0 and i == 0) for i in range(n)])
        self.ln = nn.LayerNorm(D)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        if not self.hg:
            for b in self.bl: x = b(x)
        else:
            x = self.pre(x)
            B, T, C = x.shape
            # kausal: k-1 Nullen links, dann Mittel über Gruppen; Kurz-Token j deckt Positionen 4j-3..4j ab
            s = F.pad(x, (0, 0, K - 1, 0))[:, :T].reshape(B, T // K, K, C).mean(2)
            for b in self.mid: s = b(s)
            u = self.up(self.lnm(s)).repeat_interleave(K, dim=1)   # Position t nutzt Kurz-Token t//4 (deckt nur <= t ab)
            x = self.post(x + u)
        return self.ln(x) @ self.emb.weight.T


ARME = {"F4": dict(n=4), "F3": dict(n=3), "F6": dict(n=6), "HG": dict(hg=True), "LRU": dict(n=4, lru0=True), "HGLRU": dict(hg=True, lru0=True)}


def baue(arm): return LM(**ARME[arm])


def flops_token(arm):
    """Vorwärts-FLOPs pro Token (2 pro MAC). Attention: Scores+Werte, kausal gemittelt ~ 2*D*(T+1)/2 MAC. LRU: ideale Rekurrenz,
    Matmuls 4*D*N + Rekurrenz ~ N komplexe MAC (4 reelle) + Dämpfung. Einbettung/LayerNorm/Aktivierung nicht gezählt."""
    a = ARME[arm]; N = D
    def blk(T, lru, anteil):   # anteil = Tokens dieser Länge pro Eingabetoken
        ffn = 2 * D * HF
        if lru: mix = 4 * D * N + 4 * N
        else: mix = 4 * D * D + 2 * D * (T + 1) / 2
        return anteil * 2 * (ffn + mix)
    head = 2 * D * 256
    if not a.get("hg"):
        return sum(blk(CTX, a.get("lru0") and i == 0, 1) for i in range(a["n"])) + head
    T4 = CTX // K
    mid = sum(blk(T4, False, 1 / K) for _ in range(4))
    return blk(CTX, a.get("lru0", False), 1) + blk(CTX, False, 1) + mid + 2 * D * D / K + head   # + Upsampling-Linear auf Kurz-Token


def n_param(m): return sum(p.numel() for p in m.parameters())


def bpc(netz, text, bs=8):
    n = (len(text) - 1) // CTX
    x = text[:n * CTX].long().reshape(n, CTX); y = text[1:n * CTX + 1].long().reshape(n, CTX)
    netz.eval(); s = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            s += float(F.cross_entropy(netz(x[i:i + bs]).reshape(-1, 256), y[i:i + bs].reshape(-1), reduction="sum"))
    return s / (n * CTX) / math.log(2)


def us_token(netz, reps=5):
    netz.eval(); x = torch.randint(0, 256, (BATCH, CTX))
    with torch.no_grad():
        netz(x); t = []
        for _ in range(reps):
            t0 = time.perf_counter(); netz(x); t.append(time.perf_counter() - t0)
    return min(t) / (BATCH * CTX) * 1e6


def kausal_test():
    """Änderung an Byte t darf Ausgaben < t nicht ändern."""
    torch.manual_seed(0); ok = True
    for arm in ARME:
        m = baue(arm).eval(); x = torch.randint(0, 256, (1, CTX)); y = x.clone(); t = 100; y[0, t] = (y[0, t] + 1) % 256
        with torch.no_grad(): d = (m(x) - m(y)).abs().amax(-1)[0]
        print(arm, "max Diff vor t:", float(d[:t].max()), " ab t:", float(d[t:].max())); ok &= float(d[:t].max()) < 1e-5
    return ok


def lauf(arm, seed, schritte, lr, val=False):
    tr, va, te = daten(False)
    g = torch.Generator().manual_seed(seed + 5); torch.manual_seed(seed)
    netz = baue(arm); opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    offs = torch.arange(CTX); t0 = time.time(); netz.train()
    for _ in range(schritte):
        i = torch.randint(0, len(tr) - CTX - 1, (BATCH,), generator=g)
        l = F.cross_entropy(netz(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
        if not torch.isfinite(l): return dict(arm=arm, seed=seed, bpc=8.0, instabil=True)
        opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0); opt.step(); sch.step()
    tt = time.time() - t0
    return dict(arm=arm, seed=seed, schritte=schritte, lr=lr, val=val, bpc=bpc(netz, va[:32768] if val else te[:32768]), train_bpc_last=float(l) / math.log(2), s_train=tt,
                param=n_param(netz), mflops=flops_token(arm) / 1e6, us_token=us_token(netz) if seed == 975 and not val else None)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "zaehlen":
        for a in ARME: print(a, n_param(baue(a)), round(flops_token(a) / 1e6, 3))
        print("kausal:", kausal_test())
    elif m == "zeit":
        for a in ARME: print(a, round(us_token(baue(a)), 1), "us/Token")
        t = baue("F4").train(); x = torch.randint(0, 256, (BATCH, CTX)); t0 = time.time()
        for _ in range(3): F.cross_entropy(t(x).reshape(-1, 256), x.reshape(-1)).backward()
        print("F4 s/Schritt (fwd+bwd):", (time.time() - t0) / 3)
    elif m == "lauf":
        a, s, st, lr = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
        r = lauf(a, s, st, lr, len(sys.argv) > 6 and sys.argv[6] == "val")
        print(json.dumps(r)); open(Path(__file__).with_name("ergebnisse.jsonl"), "a").write(json.dumps(r) + "\n")
