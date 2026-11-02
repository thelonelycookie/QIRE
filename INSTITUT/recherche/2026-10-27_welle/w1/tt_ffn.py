"""w1: Tensor-Train-FFN im Byte-Sprachmodell (Erkundung, Tuning-Seeds 900-904).
Aufruf: python3 tt_ffn.py <arm> <seed> <schritte> <lr> [val]   -> JSON-Zeile
Arme: D (dicht 128->H->128), Dn<h> (dicht schmal), LR<r> (Niedrigrang), TT<chi>.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
D, LAGEN, KOEPFE, CTX, BATCH, H = 128, 2, 4, 64, 16, 512
IN_M, OUT_M = (4, 4, 8), (8, 8, 8)      # 128 = 4*4*8 ; 512 = 8*8*8
DATEN = Path(__file__).resolve().parents[3] / "daten"


def text(teil):
    z = [l[9:].rstrip("\n") for l in (DATEN / f"de_gsd_{teil}.conllu").read_text(encoding="utf-8").splitlines(True) if l.startswith("# text = ")]
    return torch.from_numpy(np.frombuffer(("\n".join(z) + "\n").encode(), np.uint8).copy())


class TT(nn.Module):
    """Tensor-Train-Matrix W[(i1 i2 i3),(o1 o2 o3)], Kerne G_k[r_{k-1}, i_k, o_k, r_k]. Bias dicht."""
    def __init__(self, ins, outs, chi):
        super().__init__()
        self.ins, self.outs = ins, outs
        d = len(ins)
        self.r = [1] + [chi] * (d - 1) + [1]
        din, inner = math.prod(ins), math.prod(self.r)
        sig = (1.0 / (din * inner)) ** (1 / (2 * d))
        self.cores = nn.ParameterList([nn.Parameter(torch.randn(self.r[k], ins[k], outs[k], self.r[k + 1]) * sig) for k in range(d)])
        self.bias = nn.Parameter(torch.zeros(math.prod(outs)))

    def full(self):
        # Rekonstruktion schrittweise: T[(i..),(o..),r]
        T = self.cores[0][0]                        # [i1,o1,r1]
        I, O = T.shape[0], T.shape[1]
        T = T.reshape(I, O, -1)
        for k in range(1, len(self.ins)):
            G = self.cores[k]                       # [r,ik,ok,r']
            T = torch.einsum("ior,rjps->ijops", T, G).reshape(T.shape[0] * G.shape[1], T.shape[1] * G.shape[2], G.shape[3])
        return T[..., 0]                            # [din, dout]

    def forward(self, x):
        return x @ self.full() + self.bias

    def flops(self):
        ins, outs, r = self.ins, self.outs, self.r
        f = 0
        for k in range(len(ins)):
            f += 2 * math.prod(outs[:k]) * math.prod(ins[k + 1:]) * r[k] * ins[k] * outs[k] * r[k + 1]
        return f + math.prod(outs)


class LR(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__()
        self.u, self.v = nn.Linear(din, r, bias=False), nn.Linear(r, dout)
        nn.init.normal_(self.u.weight, std=1 / math.sqrt(din)); nn.init.normal_(self.v.weight, std=1 / math.sqrt(r))
        self.din, self.dout, self.r = din, dout, r
    def forward(self, x): return self.v(self.u(x))
    def flops(self): return 2 * self.r * (self.din + self.dout) + self.dout


class Dense(nn.Linear):
    def flops(self): return 2 * self.in_features * self.out_features + self.out_features


def ffn(arm, hid=None):
    """Liefert (lin1, lin2). Arm: D | Dn<h> | LR<r> | TT<chi>."""
    if arm == "D": return Dense(D, H), Dense(H, D)
    if arm.startswith("Dn"): h = int(arm[2:]); return Dense(D, h), Dense(h, D)
    if arm.startswith("LR"): r = int(arm[2:]); return LR(D, H, r), LR(H, D, r)
    if arm.startswith("TT"): c = int(arm[2:]); return TT(IN_M, OUT_M, c), TT(OUT_M, IN_M, c)
    raise ValueError(arm)


class Block(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.f1, self.f2 = ffn(arm)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, KOEPFE, C // KOEPFE).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.emb, self.pos = nn.Embedding(256, D), nn.Embedding(CTX, D)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Block(arm) for _ in range(LAGEN)])
        self.ln = nn.LayerNorm(D)
    def forward(self, i):
        x = self.emb(i) + self.pos(torch.arange(i.shape[1]))
        for b in self.bl: x = b(x)
        return self.ln(x) @ self.emb.weight.T


def zaehle(arm):
    n = LM(arm)
    pf = sum(p.numel() for b in n.bl for m in (b.f1, b.f2) for p in m.parameters())
    ff = sum(m.flops() for b in n.bl for m in (b.f1, b.f2))
    return pf, ff, sum(p.numel() for p in n.parameters())


def bpc(n, t, bs=16):
    k = (len(t) - 1) // CTX
    x, y = t[:k * CTX].long().reshape(k, CTX), t[1:k * CTX + 1].long().reshape(k, CTX)
    n.eval(); s = 0.0
    with torch.no_grad():
        for i in range(0, k, bs):
            s += float(F.cross_entropy(n(x[i:i + bs]).reshape(-1, 256), y[i:i + bs].reshape(-1), reduction="sum"))
    return s / (k * CTX) / math.log(2)


def lauf(arm, seed, schritte, lr, val=True):
    tr, va, te = text("train"), text("dev")[:40000], text("test")[:40000]
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed + 5)
    n = LM(arm); opt = torch.optim.Adam(n.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    offs = torch.arange(CTX); t0 = time.time(); kurve = []
    for s in range(schritte):
        n.train()
        i = torch.randint(0, len(tr) - CTX - 1, (BATCH,), generator=g)
        l = F.cross_entropy(n(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
        if not torch.isfinite(l): return {"arm": arm, "seed": seed, "bpc": 8.0, "instabil": True}
        opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(n.parameters(), 1.0); opt.step(); sch.step()
        if (s + 1) % max(1, schritte // 4) == 0 and s + 1 < schritte: kurve.append(round(bpc(n, va[:8000]), 4))
    pf, ff, pg = zaehle(arm)
    return {"arm": arm, "seed": seed, "lr": lr, "schritte": schritte, "bpc": bpc(n, va if val else te), "kurve_val8k": kurve,
            "param_ffn": pf, "flops_ffn_token": ff, "param_ges": pg, "sek": round(time.time() - t0, 1), "instabil": False, "val": val}


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "zaehle":
        for arm in a[2:]: print(arm, zaehle(arm))
    else:
        print(json.dumps(lauf(a[1], int(a[2]), int(a[3]), float(a[4]), len(a) < 6 or a[5] == "val")), flush=True)
