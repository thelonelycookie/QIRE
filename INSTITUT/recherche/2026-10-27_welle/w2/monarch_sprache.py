"""w2: gelernte Monarch-Matrizen in der FFN eines kleinen Byte-Sprachmodells (Erkundung, Tuning-Seeds 910-914).
Alle FFN-Arme haben (nahezu) gleiche Matmul-FLOPs UND gleiche Gewichtszahl. Aufruf: python3 monarch_sprache.py <modus> ..."""
import sys, math, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
WURZEL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WURZEL))
from abteilungen.struktur_sprache import daten, bpc_auswerten

D, LAGEN, KOEPFE, CTX, BATCH, H = 128, 2, 4, 64, 16, 512


class Monarch(nn.Module):
    """din=nb*p -> dout=q*s. Faktor 1: nb Blöcke p->q, Permutation (Transposition), Faktor 2: q Blöcke nb->s. Gewichte nb*p*q + q*nb*s."""
    def __init__(self, nb, p, q, s):
        super().__init__()
        self.nb, self.p, self.q, self.s = nb, p, q, s
        self.w1 = nn.Parameter(torch.randn(nb, p, q) / math.sqrt(p))
        self.w2 = nn.Parameter(torch.randn(q, nb, s) / math.sqrt(nb))
        self.bias = nn.Parameter(torch.zeros(q * s))

    def forward(self, x):
        sh = x.shape
        x = x.reshape(-1, self.nb, self.p)
        y = torch.einsum("bnp,npq->bnq", x, self.w1)       # Lage 1: lokale Blöcke
        y = y.transpose(1, 2)                              # Permutation (Mischen wie SWAP/Verschränkung)
        y = torch.einsum("bqn,qns->bqs", y, self.w2)       # Lage 2: lokale Blöcke
        return y.reshape(*sh[:-1], self.q * self.s) + self.bias

    def gew(self): return self.nb * self.p * self.q + self.q * self.nb * self.s


class LR(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__(); self.a = nn.Linear(din, r, bias=False); self.b = nn.Linear(r, dout)
    def forward(self, x): return self.b(self.a(x))


def mat(art, din, dout, r=None):
    if art == "dicht": return nn.Linear(din, dout)
    if art == "lr": return LR(din, dout, r)
    if art == "mon":   # b = Wurzel-artig: 128->512: 16x8 -> 32x16 ; 512->128: 32x16 -> 16x8
        return Monarch(16, 8, 32, 16) if dout > din else Monarch(32, 16, 16, 8)


# Arm -> (Struktur, Breite/Rang, Aktivierung). Budget je Matrix 12288 Gewichte (Monarch), Dicht h=96: 12288, LR r=19: 12160
ARME = {"dicht512": ("dicht", 512, "gelu"), "dicht96": ("dicht", 96, "gelu"), "lr19": ("lr", 19, "gelu"),
        "mon": ("mon", 512, "gelu"), "mon_z2": ("mon", 512, "z2"), "dicht96_z2": ("dicht", 96, "z2")}


class FFN(nn.Module):
    def __init__(self, arm):
        super().__init__()
        art, k, akt = ARME[arm]
        h = k if art != "lr" else H
        self.a = mat(art, D, h, k if art == "lr" else None); self.b = mat(art, h, D, k if art == "lr" else None)
        self.akt = F.gelu if akt == "gelu" else torch.square
    def forward(self, x): return self.b(self.akt(self.a(x)))


class Block(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj, self.ffn = nn.Linear(D, 3 * D), nn.Linear(D, D), FFN(arm)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, KOEPFE, C // KOEPFE).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.ffn(self.ln2(x))


class LM(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.pos = nn.Embedding(256, D), nn.Embedding(CTX, D)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Block(arm) for _ in range(LAGEN)]); self.ln = nn.LayerNorm(D)
    def forward(self, i):
        x = self.emb(i) + self.pos(torch.arange(i.shape[1]))
        for b in self.bl: x = b(x)
        return self.ln(x) @ self.emb.weight.T


def ffn_zaehlung(netz):
    """(FFN-Gewichte ohne Bias, FFN-Parameter mit Bias, FFN-FLOPs/Token = 2*Gewichte der Matmuls)"""
    gew = par = 0
    for b in netz.bl:
        for m in b.ffn.modules():
            if isinstance(m, nn.Linear): gew += m.weight.numel()
            elif isinstance(m, Monarch): gew += m.gew()
        par += sum(p.numel() for p in b.ffn.parameters())
    return gew, par, 2 * gew


def trainiere(arm, seed, lr, schritte, kurve_alle=150, test=False):
    tr, va, te = daten()
    netz = LM(arm, seed)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 7)
    warm = max(1, schritte // 20)
    kurve = []
    t0 = time.process_time()
    for s in range(schritte):
        f = (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (schritte - warm)))
        for gr in opt.param_groups: gr["lr"] = lr * max(f, 0.02)
        netz.train()
        ix = torch.randint(0, len(tr) - CTX - 1, (BATCH,), generator=g)
        x = torch.stack([tr[i:i + CTX] for i in ix]).long(); y = torch.stack([tr[i + 1:i + CTX + 1] for i in ix]).long()
        loss = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss): return {"kollaps": True, "val": 9.0, "kurve": kurve}
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0); opt.step()
        if (s + 1) % kurve_alle == 0 or s + 1 == schritte:
            kurve.append((s + 1, bpc_auswerten(netz, va[:16384], ctx=CTX)))
    r = {"val": kurve[-1][1], "kurve": kurve, "cpu_s": time.process_time() - t0, "kollaps": False}
    if test: r["test"] = bpc_auswerten(netz, te, ctx=CTX)
    return r


if __name__ == "__main__":
    modus = sys.argv[1]
    aus = Path(__file__).parent
    if modus == "zaehlung":
        for a in ARME:
            n = LM(a, 0); g, p, f = ffn_zaehlung(n); print(a, "FFN-Gew", g, "FFN-Par", p, "FFN-FLOPs/Tok", f, "gesamt", sum(x.numel() for x in n.parameters()))
    elif modus == "tuning":      # Lernratenwahl auf Val, Seed 913, kurz
        schritte = int(sys.argv[2]); res = {}
        for a in ("dicht512", "dicht96", "lr19", "mon"):
            for lr in (1e-3, 3e-3, 6e-3):
                r = trainiere(a, 913, lr, schritte, kurve_alle=schritte); res[f"{a}|{lr}"] = r["val"]; print(a, lr, round(r["val"], 4), round(r["cpu_s"]), flush=True)
        json.dump(res, open(aus / "tuning.json", "w"))
        lrs = {a: min((1e-3, 3e-3, 6e-3), key=lambda l: res[f"{a}|{l}"]) for a in ("dicht512", "dicht96", "lr19", "mon")}
        lrs.update({"mon_z2": lrs["mon"], "dicht96_z2": lrs["dicht96"]}); json.dump(lrs, open(aus / "lr.json", "w")); print(lrs)
    elif modus == "mess":
        schritte = int(sys.argv[2]); lrs = json.load(open(aus / "lr.json")); seeds = [int(s) for s in sys.argv[3].split(",")]
        pf = aus / "mess.json"; res = json.load(open(pf)) if pf.exists() else {}
        for sd in seeds:
            for a in ARME:
                k = f"{a}|{sd}"
                if k in res: continue
                res[k] = trainiere(a, sd, lrs[a], schritte, test=True); print(k, round(res[k]["val"], 4), round(res[k]["test"], 4), round(res[k]["cpu_s"]), flush=True)
                json.dump(res, open(pf, "w"))
