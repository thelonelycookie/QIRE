"""Gemeinsame Bausteine der Erkundungsskripte (Byte-LM, FLOP-Zählung, Ablage). Erkundung, kein Befund.
Byte-LM: Decoder-Transformer, Pre-LayerNorm, 4 Lagen, 4 Köpfe, FFN 4d (GELU), gelernte Positionen bis 256, Ausgabe = Einbettung (gebunden).
Daten: UD_German-GSD wie I-045 (abteilungen/struktur_sprache.daten); Tuning-Seeds lesen auf der Validierung (dev), nie auf test."""
from __future__ import annotations

import json
import math
import os
import sys
import time

HIER = os.path.dirname(os.path.abspath(__file__))
WURZEL = os.path.dirname(HIER)
if WURZEL not in sys.path:
    sys.path.insert(0, WURZEL)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abteilungen.struktur_sprache import daten as _daten

MAXCTX = 256


def daten(klein=False):
    return _daten(klein)


# ------------------------------------------------------------------ Modell
class Blk(nn.Module):
    def __init__(self, d, h, ff, hid=None):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.a, self.b = nn.Linear(d, ff), nn.Linear(ff, d)
        self.hid = hid if hid is not None else nn.Identity()   # Haken hinter der Aktivierung (Dropout-Varianten)
        self.qs = 1.0                                           # Zusatzskala auf q (nach isometrischer Verbreiterung)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.b(self.hid(F.gelu(self.a(self.ln2(x)))))


class LM(nn.Module):
    def __init__(self, d=128, lagen=4, h=4, ff=None, seed=0, hid_fabrik=None):
        super().__init__()
        torch.manual_seed(seed)
        self.d, self.lagen, self.nh, self.ff = d, lagen, h, ff or 4 * d
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(MAXCTX, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Blk(d, h, self.ff, hid_fabrik() if hid_fabrik else None) for _ in range(lagen)])
        self.ln = nn.LayerNorm(d)
        self.ausg = 1.0

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.bl:
            x = b(x)
        return (self.ln(x) @ self.emb.weight.T) * self.ausg


def n_param(netz, ohne_emb=False) -> int:
    return int(sum(p.numel() for n, p in netz.named_parameters() if p.requires_grad and not (ohne_emb and n.startswith(("emb", "pos")))))


def flops_fwd_token(d, lagen=4, ctx=128, ff=None) -> int:
    """Vorwärts-FLOPs je Token (1 FLOP = 1 Multiplikation oder Addition): qkv 6d², Ausgabe 2d², Attention 2·ctx·d (im Mittel ctx/2 Schlüssel),
    FFN 4·d·ff, Kopf 2·256·d. Gleiche Zählung wie flops_je_token in struktur_sprache (I-045-v4/v5)."""
    ff = ff or 4 * d
    return int(lagen * (6 * d * d + 2 * d * d + 2 * ctx * d + 4 * d * ff) + 2 * 256 * d)


def flops_train_token(d, lagen=4, ctx=128, ff=None) -> int:
    return 3 * flops_fwd_token(d, lagen, ctx, ff)        # Vorwärts + Rückwärts ≈ 3x


# ------------------------------------------------------------------ Auswertung
def bpc(netz, text, ctx=128, bs=16, max_bytes=65536) -> float:
    text = text[:max_bytes]
    n = (len(text) - 1) // ctx
    x = text[:n * ctx].long().reshape(n, ctx)
    y = text[1:n * ctx + 1].long().reshape(n, ctx)
    netz.eval()
    s = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            s += float(F.cross_entropy(netz(x[i:i + bs]).reshape(-1, 256), y[i:i + bs].reshape(-1), reduction="sum"))
    netz.train()
    return s / (n * ctx) / math.log(2)


# ------------------------------------------------------------------ Training
def lr_faktor(frac, warm=0.03):
    """Aufwärmen, dann Kosinus-Abfall auf 0, in Anteil frac ∈ [0,1] des Budgets (Schritte oder FLOPs)."""
    if frac < warm:
        return (frac + 1e-3) / warm
    return 0.5 * (1 + math.cos(math.pi * (frac - warm) / (1 - warm)))


def stapel(tr, g, batch, ctx):
    i = torch.randint(0, len(tr) - ctx - 1, (batch,), generator=g)
    offs = torch.arange(ctx)
    return tr[i[:, None] + offs].long(), tr[i[:, None] + offs + 1].long()


def lauf_schritte(netz, opt, tr, g, ctxe, lr, fracs, tokens=1024, klemme=1.0, nach_schritt=None, anlauf=0):
    """Führt len(ctxe) Schritte aus; ctxe[i] = Kontext im Schritt i (Batch = tokens/ctx), fracs[i] = Lernratenanteil.
    Rückgabe: letzter Trainingsverlust (Mittel der letzten 50), ob instabil."""
    netz.train()
    letzte = []
    for i, (c, fr) in enumerate(zip(ctxe, fracs)):
        for gr in opt.param_groups:
            gr["lr"] = lr * lr_faktor(fr) * (min(1.0, (i + 1) / anlauf) if anlauf else 1.0)
        x, y = stapel(tr, g, max(1, tokens // c), c)
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l):
            return {"instabil": True, "train_verlust": float("nan")}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), klemme)
        opt.step()
        letzte.append(float(l.detach()))
        if len(letzte) > 50:
            letzte.pop(0)
        if nach_schritt:
            nach_schritt(i)
    return {"instabil": False, "train_verlust": float(np.mean(letzte)) / math.log(2) if letzte else float("nan")}


def speichere(kid, seed, inhalt, schnell=False):
    ordner = os.path.join(HIER, "roh")
    os.makedirs(ordner, exist_ok=True)
    pfad = os.path.join(ordner, f"{kid}_s{seed}.json" if not schnell else f"{kid}_s{seed}_schnell.json")
    tmp = pfad + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(inhalt, f, indent=1, ensure_ascii=False, default=float)
    os.replace(tmp, pfad)
    return pfad


def cli(kid, main, beschr=""):
    import argparse
    a = argparse.ArgumentParser(description=beschr)
    a.add_argument("--seed", type=int, default=950)
    a.add_argument("--schnell", action="store_true", help="Rauchtest (<= 2 min im Container)")
    o = a.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    r = main(o.seed, schnell=o.schnell) if "schnell" in main.__code__.co_varnames[:main.__code__.co_argcount] else main(o.seed)
    r["sekunden"] = time.time() - t0
    print(json.dumps(r, indent=1, default=float)[:3000])
    print("gespeichert:", speichere(kid, o.seed, r, o.schnell))
