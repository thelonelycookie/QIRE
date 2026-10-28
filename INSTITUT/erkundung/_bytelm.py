"""Gemeinsame Bausteine der Erkundungsskripte K09/K10/K12 (Byte-Sprachmodell, deutscher Text UD_German-GSD).
Erkundung, kein Befund. Daten und Aufteilung wie abteilungen/struktur_sprache.py (train / dev = Tuning / test).
Modell: Decoder-Transformer, Pre-LayerNorm, 4 Lagen, Köpfe = d/64, Kontext 128, gelernte Positionen, Batch 8 x 128.
Training: Adam, 5 % Aufwärmen, Kosinus, Gradientenbegrenzung 1,0 (wie I-045). Die Lernrate wird je Arm auf der Validierung gewählt
(Gitter, bei Randtreffer höchstens zweimal um den Faktor 2,4 verlängert), dann folgt der lange Lauf mit der gewählten Rate.
FLOPs je Token: Matmuls (1 Multiplikation = 1 Addition = 1 Flop), Attention mit mittlerem Kontext 64; Elementweises (Aktivierung, Normierung) nicht gezählt."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HIER = Path(__file__).resolve().parent
INSTITUT = HIER.parent
for _p in (str(INSTITUT), str(INSTITUT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from abteilungen.struktur_sprache import daten  # noqa: E402

KONTEXT, BATCH, LAGEN = 128, 8, 4
LR_GITTER = (1e-3, 2.5e-3, 6e-3)
LR_FAKTOR = 2.4
VAL_BYTES = 65536
TUNING_SEEDS = tuple(range(950, 960))


# ------------------------------------------------------------------ Modell
class FFNDicht(nn.Module):
    """d -> h -> d. akt: 'gelu' | 'z2' | 'relu2' | 'z2bn<g>' (Born-Normierung in Gruppen g) | 'z2ln' (LayerNorm ohne Skala vor dem Quadrat)."""

    def __init__(self, d, h, akt="gelu"):
        super().__init__()
        self.a, self.b = nn.Linear(d, h), nn.Linear(h, d)
        self.akt_name, self.h = akt, h
        self.gruppe = int(akt[4:]) if akt.startswith("z2bn") else 0
        if self.gruppe:
            assert h % self.gruppe == 0
        self.ln = nn.LayerNorm(h, elementwise_affine=False) if akt == "z2ln" else None
        self.flops = 4 * d * h

    def aktiv(self, z):
        n = self.akt_name
        if n == "gelu":
            return F.gelu(z)
        if n == "z2":
            return z * z
        if n == "relu2":
            return F.relu(z) ** 2
        if n == "z2ln":
            return self.ln(z) ** 2
        s = (z * z).reshape(*z.shape[:-1], self.h // self.gruppe, self.gruppe)       # Born-Normierung: z^2 / (Mittel z^2 + eps) je Gruppe
        return (s / (s.mean(-1, keepdim=True) + 1e-6)).reshape(z.shape)

    def forward(self, x):
        return self.b(self.aktiv(self.a(x)))


class Lage(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = ffn

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.ffn(self.ln2(x))


class KopfVoll(nn.Module):
    """Voller 256-Wege-Softmax mit gebundener Einbettung (wie I-045)."""

    def __init__(self, emb):
        super().__init__()
        self.emb = [emb]          # Liste: kein doppelter Parameter-Eintrag
        self.flops = 2 * 256 * emb.weight.shape[1]
        self.eff_param = 256 * emb.weight.shape[1]

    def nll(self, h, y):
        return F.cross_entropy(h @ self.emb[0].weight.T, y, reduction="none")


class LM(nn.Module):
    def __init__(self, d, ffn_fn, kopf_fn=None, lagen=LAGEN, ctx=KONTEXT):
        super().__init__()
        self.d = d
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Lage(d, max(1, d // 64), ffn_fn(d, i)) for i in range(lagen)])
        self.ln = nn.LayerNorm(d)
        self.kopf = KopfVoll(self.emb) if kopf_fn is None else kopf_fn(d, self.emb)

    def hidden(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.bl:
            x = b(x)
        return self.ln(x)

    def nll(self, x, y):
        h = self.hidden(x)
        return self.kopf.nll(h.reshape(-1, self.d), y.reshape(-1))

    def zahlen(self) -> dict:
        d = self.d
        attn = 8 * d * d + 4 * (KONTEXT // 2) * d
        ffn = sum(b.ffn.flops for b in self.bl)
        kopf = self.kopf.flops
        return {"param_gesamt": int(sum(p.numel() for p in self.parameters())),
                "param_ffn": int(sum(p.numel() for b in self.bl for p in b.ffn.parameters())),
                "param_kopf_eigen": int(sum(p.numel() for p in self.kopf.parameters())),
                "param_kopf_effektiv": int(self.kopf.eff_param),
                "flops_ffn": int(ffn), "flops_attn": int(len(self.bl) * attn), "flops_kopf": int(kopf),
                "flops_gesamt": int(ffn + len(self.bl) * attn + kopf)}


def bpc(netz, text, ctx=KONTEXT, bs=16) -> float:
    n = (len(text) - 1) // ctx
    x = text[:n * ctx].long().reshape(n, ctx)
    y = text[1:n * ctx + 1].long().reshape(n, ctx)
    netz.eval()
    s = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            s += float(netz.nll(x[i:i + bs], y[i:i + bs]).sum())
    return s / (n * ctx) / math.log(2)


def trainiere(bauer, lr, seed, T, klein=False, mess=("val",), extra_eval=None):
    """bauer(seed) -> LM. Gleicher Datenstrom für alle Arme (Generator seed+5), gleiche Initialisierung der geteilten Teile je Seed."""
    torch.set_num_threads(1)
    tr, va, te = daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    torch.manual_seed(seed)
    netz = bauer(seed)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, T // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm))))
    offs = torch.arange(KONTEXT)
    t0 = time.time()
    netz.train()
    for _ in range(T):
        i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        l = netz.nll(x, y).mean()
        if not torch.isfinite(l):
            return {"instabil": True, "bpc_val": float("inf"), "bpc_test": float("inf")}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step()
        sched.step()
    out = {"instabil": False, "sekunden_training": time.time() - t0, "sek_je_schritt": (time.time() - t0) / max(1, T)}
    torch.manual_seed(seed + 999)           # feste Zufallszahlen für Arme mit Stichprobe bei der Auswertung
    if "val" in mess:
        out["bpc_val"] = bpc(netz, va[:VAL_BYTES])
    if "test" in mess:
        out["bpc_test"] = bpc(netz, te)
    if extra_eval is not None:
        out.update(extra_eval(netz, va[:VAL_BYTES], te, mess))
    return out


def arm_lauf(bauer, seed, T_tune, T_fin, klein=False, lrs=LR_GITTER, extra_eval=None, log=None) -> dict:
    """Lernrate wählen (Validierung, T_tune Schritte), dann langer Lauf (T_fin Schritte) mit Val- und Test-bpc."""
    tun = {}

    def ev(lr):
        lr = float(f"{lr:.6g}")
        if lr not in tun:
            r = trainiere(bauer, lr, seed, T_tune, klein, ("val",))
            tun[lr] = r["bpc_val"]
        return lr
    for lr in lrs:
        ev(lr)
    for _ in range(2):
        best = min(tun, key=tun.get)
        if best == min(tun) and tun[best] < float("inf"):
            ev(best / LR_FAKTOR)
        elif best == max(tun):
            ev(best * LR_FAKTOR)
        else:
            break
    best = min(tun, key=tun.get)
    r = trainiere(bauer, best, seed, T_fin, klein, ("val", "test"), extra_eval)
    r["lr"] = best
    r["tuning_val_bpc"] = {str(k): v for k, v in tun.items()}
    r.update(bauer(seed).zahlen())
    r["T_tune"], r["T_fin"] = T_tune, T_fin
    if log:
        log(f"   lr={best} val={r['bpc_val']:.4f} test={r['bpc_test']:.4f} ({r['sekunden_training']:.0f}s)")
    return r


# ------------------------------------------------------------------ Ablage / CLI
def pfad(kid, seed, schnell, teil=False):
    (HIER / "roh").mkdir(exist_ok=True)
    return HIER / "roh" / f"{kid}_s{seed}{'_schnell' if schnell else ''}{'.partial' if teil else ''}.json"


def lade_teil(kid, seed, schnell) -> dict:
    p = pfad(kid, seed, schnell, True)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def speichere_teil(kid, seed, schnell, zellen):
    p = pfad(kid, seed, schnell, True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(zellen, indent=1))
    os.replace(tmp, p)


def speichere(kid, seed, schnell, inhalt):
    p = pfad(kid, seed, schnell)
    p.write_text(json.dumps(inhalt, indent=1))
    t = pfad(kid, seed, schnell, True)
    if t.exists():
        t.unlink()
    return p


def cli(kid, main):
    a = argparse.ArgumentParser()
    a.add_argument("--seed", type=int, default=950)
    a.add_argument("--schnell", action="store_true")
    o = a.parse_args()
    assert o.seed in TUNING_SEEDS, "Erkundung nur auf Tuning-Seeds 950-959"
    t = time.time()
    r = main(o.seed, o.schnell)
    print(f"{kid} s{o.seed}{' schnell' if o.schnell else ''}: fertig in {(time.time() - t) / 60:.1f} min")
    return r


def lauf_zellen(kid, seed, schnell, zellen_def, kopf_text=""):
    """zellen_def: Liste (zellname, bauer, T_tune, T_fin, extra_eval). Wiederaufnahme je Zelle über die .partial-Datei."""
    t0 = time.time()
    teil = lade_teil(kid, seed, schnell)
    log = lambda s: print(s, flush=True)
    for name, bauer, tt, tf, ee in zellen_def:
        if name in teil:
            continue
        log(f"[{kid} s{seed}] {name} ...")
        teil[name] = arm_lauf(bauer, seed, tt, tf, klein=schnell, extra_eval=ee, log=log)
        speichere_teil(kid, seed, schnell, teil)
    out = {"kid": kid, "seed": seed, "schnell": schnell, "beschreibung": kopf_text, "zellen": teil,
           "sekunden_gesamt_dieser_aufruf": time.time() - t0}
    speichere(kid, seed, schnell, out)
    return out
