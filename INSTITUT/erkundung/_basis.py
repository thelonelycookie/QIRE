"""Gemeinsame Bausteine der Erkundungsskripte (Welle 2, 28.10.2026). Erkundung, kein Befund.
Nutzt abteilungen/struktur_sprache.py (Byte-LM-Daten, bpc) und abteilungen/kompression_bytes.py (Bild-Daten, Hadamard, Quantisierung)."""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

HIER = Path(__file__).resolve().parent
INST = HIER.parent
if str(INST) not in sys.path:
    sys.path.insert(0, str(INST))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import abteilungen.struktur_sprache as ss
import abteilungen.kompression_bytes as kb

torch.set_num_threads(1)
KONTEXT, BATCH = ss.KONTEXT, ss.BATCH
TUNING_SEEDS = tuple(range(950, 960))
LM_LR = 2e-3


def seeds_pruefen(seed: int):
    if seed not in TUNING_SEEDS:
        raise SystemExit(f"Seed {seed} ist kein Tuning-Seed (950-959). Erkundungen laufen nie auf Mess-Seeds.")


def rahmen(kid: str, main, **extra):
    """Standard-Kommandozeile: --seed N, --schnell. Speichert roh/<KID>_s<seed>.json."""
    a = argparse.ArgumentParser()
    a.add_argument("--seed", type=int, default=950)
    a.add_argument("--schnell", action="store_true")
    o = a.parse_args()
    seeds_pruefen(o.seed)
    t = time.time()
    erg = main(o.seed, schnell=o.schnell) if o.schnell else main(o.seed)
    erg["kandidat"], erg["seed"], erg["schnell"] = kid, o.seed, bool(o.schnell)
    erg["minuten"] = (time.time() - t) / 60
    (HIER / "roh").mkdir(exist_ok=True)
    pfad = HIER / "roh" / (f"{kid}_s{o.seed}" + ("_schnell" if o.schnell else "") + ".json")
    pfad.write_text(json.dumps(erg, indent=1, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)))
    print(f"{kid} s{o.seed} fertig in {erg['minuten']:.1f} min -> {pfad}", flush=True)
    return erg


# ------------------------------------------------------------------ Byte-LM mit frei wählbarer FFN
class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))


class LMBlockX(nn.Module):
    def __init__(self, d, h, ffn):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att, self.ffn = Attn(d, h), ffn

    def forward(self, x, lage=0):
        x = x + self.att(self.ln1(x))
        return x + self.ffn(self.ln2(x), lage) if getattr(self.ffn, "braucht_lage", False) else x + self.ffn(self.ln2(x))


class LMX(nn.Module):
    """Decoder wie kompression_bytes.LM (d, 4 Lagen, Kontext 128, gebundene Ausgabe), FFN pro Lage durch `ffns[l]` (Modul) gegeben."""

    def __init__(self, ffns, d=128, kopf=None, ausgabe=None, ctx=KONTEXT):
        super().__init__()
        self.d = d
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([LMBlockX(d, kb.koepfe(d), f) for f in ffns])
        self.ln = nn.LayerNorm(d)
        self.ausgabe = ausgabe          # None = gebundene Einbettung; sonst Modul d -> 256 (oder eigene Logit-Funktion)

    def features(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for l, b in enumerate(self.bl):
            x = b(x, l)
        return self.ln(x)

    def forward(self, idx):
        z = self.features(idx)
        return z @ self.emb.weight.T if self.ausgabe is None else self.ausgabe(z)


class FFNDichtX(nn.Module):
    def __init__(self, d, f, akt=F.gelu):
        super().__init__()
        self.a, self.b, self.akt = nn.Linear(d, f), nn.Linear(f, d), akt

    def forward(self, x):
        return self.b(self.akt(self.a(x)))


def zaehle_param(netz, nur_trainierbar=True):
    return int(sum(p.numel() for p in netz.parameters() if (p.requires_grad or not nur_trainierbar)))


def lm_macs(d, lagen, ffn_macs_je_lage, ctx=KONTEXT, kopf_macs=None):
    """MACs je Token: qkv 3d², proj d², Attention d(ctx+1), FFN, Ausgabe 256·d (oder kopf_macs)."""
    return int(lagen * (4 * d * d + d * (ctx + 1)) + lagen * ffn_macs_je_lage + (256 * d if kopf_macs is None else kopf_macs))


def lm_trainiere(netz, seed, T, lr=LM_LR, schnell=False, schritt_hook=None, ctx=KONTEXT, tr_anteil=1.0, kontext_plan=None):
    """Adam, 5 % Aufwärmen, Kosinus, Clip 1,0, Batch 8x128 (wie ss.trainiere). Gibt Val-bpc zurück (nie Test). kontext_plan(s)->Kontextlänge optional."""
    tr, va, _ = ss.daten(schnell)
    if tr_anteil < 1.0:
        tr = tr[: int(len(tr) * tr_anteil)]
    g = torch.Generator().manual_seed(seed + 5)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=lr)
    warm = max(1, T // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm))))
    netz.train()
    t0 = time.time()
    for s in range(T):
        c = ctx if kontext_plan is None else kontext_plan(s)
        i = torch.randint(0, len(tr) - c - 1, (BATCH,), generator=g)
        offs = torch.arange(c)
        x, y = tr[i[:, None] + offs].long(), tr[i[:, None] + offs + 1].long()
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l):
            return {"bpc": 8.0, "instabil": True, "sekunden": time.time() - t0}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        if schritt_hook:
            schritt_hook(s)
        opt.step()
        sched.step()
    sek = time.time() - t0
    return {"bpc": lm_val(netz, va), "instabil": False, "sekunden": sek, "schlusslehre_loss": float(l.detach())}


def lm_val(netz, va=None, ctx=KONTEXT):
    if va is None:
        va = ss.daten(False)[1]
    return ss.bpc_auswerten(netz, va[: ss.VAL_BYTES], ctx=ctx)


def lm_lr_waehlen(bauer, seed, schnell, T_kurz=400, gitter=(1e-3, 4e-3), **kw):
    """Kleine Lernratenwahl auf demselben Tuning-Seed (Val-bpc nach T_kurz Schritten); gibt (lr, tabelle) zurück."""
    tab = {}
    for lr in gitter:
        torch.manual_seed(seed)
        tab[lr] = lm_trainiere(bauer(), seed, T_kurz, lr=lr, schnell=schnell, **kw)["bpc"]
    return min(tab, key=tab.get), tab


# ------------------------------------------------------------------ Bild-MLPs
def mlp_trainiere(netz, ds, seed, ep, lr=1e-3, klein=False, bs=128, daten=None, verbose=False):
    """Wie kb.trainiere_bild (Adam, 5 % Aufwärmen, Kosinus), aber auf beliebigen Daten (xtr,ytr)."""
    xtr, ytr = daten if daten is not None else kb.bild_daten(ds, klein)[:2]
    N = len(xtr)
    spe = math.ceil(N / bs)
    n = ep * spe
    opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=lr)
    warm = max(1, n // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm))))
    netz.train()
    for e in range(ep):
        p = torch.randperm(N, generator=torch.Generator().manual_seed(seed * 7919 + e))
        for k in range(spe):
            b = p[k * bs:(k + 1) * bs]
            l = F.cross_entropy(netz(xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
    return float(l)


def acc(netz, X, Y):
    netz.eval()
    with torch.no_grad():
        return float((netz(X).argmax(1) == Y).float().mean())


def fwht(x):
    """Schnelle Walsh-Hadamard-Transformation über die letzte Achse (Länge 2^k), normiert (orthogonal)."""
    n = x.shape[-1]
    h = 1
    y = x
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], -2).reshape(*x.shape[:-1], n)
        h *= 2
    return y / math.sqrt(n)
