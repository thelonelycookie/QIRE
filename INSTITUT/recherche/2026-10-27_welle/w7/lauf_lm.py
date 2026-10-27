"""w7: Byte-LM (UD_German-GSD), d=64, 2 Lagen; nur die FFN-Aufwaertsprojektion wechselt (real | kdicht | phase | phase_flop)."""
import sys, os, json, time, math, urllib.request
sys.path.insert(0, "/home/user/QIRE/INSTITUT"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from komplex import Born, param_born
torch.set_num_threads(1)
SEEDS = [960, 961, 962, 963, 964]
D, L, KOPF, CTX, BS = 64, 2, 4, 64, 16
SCHRITTE = int(os.environ.get("W7_SCHRITTE", 1200))
DAT = "/home/user/QIRE/INSTITUT/daten"
Q = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-{}.conllu"
OUT = os.path.join(os.path.dirname(__file__), "lm_ergebnis.json")

def text(t):
    p = f"{DAT}/de_gsd_{t}.conllu"
    if not os.path.exists(p):
        os.makedirs(DAT, exist_ok=True); urllib.request.urlretrieve(Q.format(t), p)
    s = [z[9:].rstrip("\n") for z in open(p, encoding="utf-8") if z.startswith("# text = ")]
    return torch.from_numpy(np.frombuffer(("\n".join(s) + "\n").encode(), np.uint8).copy())

def ffn_kosten(art, H):   # Param, FLOPs je Token und Lage (mit Bias)
    if art == "real": return 2 * D * H + H + D, 4 * D * H
    return param_born(D, H, art) + H * D + D, 4 * D * H + 2 * H * D

def finde(art, ziel, idx):
    return min(range(2, 2048), key=lambda H: abs(ffn_kosten(art, H)[idx] - ziel))

HR = 256
FORM = {"real": ("real", HR), "kdicht": ("kdicht", finde("kdicht", ffn_kosten("real", HR)[0], 0)),
        "phase": ("phase", finde("phase", ffn_kosten("real", HR)[0], 0)),
        "phase_flop": ("phase", finde("phase", ffn_kosten("real", HR)[1], 1))}

class FFN(nn.Module):
    def __init__(s, art, H, gen):
        super().__init__()
        s.up = nn.Linear(D, H) if art == "real" else Born(D, H, art, gen)
        s.art = art; s.norm = nn.LayerNorm(H, elementwise_affine=False); s.down = nn.Linear(H, D)
    def forward(s, x):
        h = s.up(x)
        return s.down(s.norm(F.gelu(h) if s.art == "real" else h))

class Blk(nn.Module):
    def __init__(s, art, H, gen):
        super().__init__()
        s.l1, s.l2 = nn.LayerNorm(D), nn.LayerNorm(D); s.qkv, s.pr = nn.Linear(D, 3 * D), nn.Linear(D, D); s.f = FFN(art, H, gen)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.l1(x)).reshape(B, T, 3, KOPF, C // KOPF).permute(2, 0, 3, 1, 4)
        x = x + s.pr(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        return x + s.f(s.l2(x))

class LM(nn.Module):
    def __init__(s, art, H, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed * 31 + 11)
        s.emb, s.pos = nn.Embedding(256, D), nn.Embedding(CTX, D)
        nn.init.normal_(s.emb.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)
        s.bl = nn.ModuleList([Blk(art, H, g) for _ in range(L)]); s.ln = nn.LayerNorm(D)
    def forward(s, i):
        x = s.emb(i) + s.pos(torch.arange(i.shape[1]))
        for b in s.bl: x = b(x)
        return s.ln(x) @ s.emb.weight.T

def bpc(m, t, n=None):
    n = (len(t) - 1) // CTX; x = t[:n * CTX].long().reshape(n, CTX); y = t[1:n * CTX + 1].long().reshape(n, CTX)
    m.eval(); s = 0.0
    with torch.no_grad():
        for i in range(0, n, 32): s += float(F.cross_entropy(m(x[i:i+32]).reshape(-1, 256), y[i:i+32].reshape(-1), reduction="sum"))
    return s / (n * CTX) / math.log(2)

def train(arm, lr, seed, tr, ev, schritte=SCHRITTE):
    art, H = FORM[arm]; torch.manual_seed(seed)
    m = LM(art, H, seed); opt = torch.optim.Adam(m.parameters(), lr=lr)
    w = schritte // 20
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / (schritte - w))))
    g = torch.Generator().manual_seed(seed + 5); o = torch.arange(CTX); m.train()
    for _ in range(schritte):
        i = torch.randint(0, len(tr) - CTX - 1, (BS,), generator=g)
        l = F.cross_entropy(m(tr[i[:, None] + o].long()).reshape(-1, 256), tr[i[:, None] + o + 1].long().reshape(-1))
        if not torch.isfinite(l): return 8.0
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return bpc(m, ev)

if __name__ == "__main__":
    tr, dev, te = text("train"), text("dev")[:16384], text("test")[:32768]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for arm in FORM:
        key0 = f"{arm}|lr"
        if key0 not in res:
            best = min((train(arm, lr, 960, tr, dev, 600), lr) for lr in [1e-3, 3e-3, 1e-2])
            res[key0] = best[1]; json.dump(res, open(OUT, "w"))
        for s in SEEDS:
            k = f"{arm}|{s}"
            if k in res: continue
            res[k] = train(arm, res[key0], s, tr, te); json.dump(res, open(OUT, "w"))
        art, H = FORM[arm]; print(arm, H, ffn_kosten(art, H), "lr", res[key0], f"{time.time()-t0:.0f}s", flush=True)
