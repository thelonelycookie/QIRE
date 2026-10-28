"""Gemeinsame Bausteine der Erkundungsskripte E01..E09 (Ergaenzungen I-048). Baut auf abteilungen/kompression_bytes.py (kb) auf; kb wird nur importiert.
Kein Kandidatenskript (passt nicht auf das Muster [KE][0-9][0-9]*.py von lauf_alle.py).

Inhalt: Pfade/CLI, Training des grossen MLP, Quantisierung mit Gruppenskalen / Blockrotation / gemischten Bits (exakte Bytes), kleines dichtes fp16-Modell,
        Byte-LM mit relu2-Option und Top-K-Ueberspringen. Alles bewusst klein und ohne Zustand ausser dem Prozess-Cache."""
from __future__ import annotations

import argparse
import copy
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

import abteilungen.kompression_bytes as kb
import abteilungen.struktur_sprache as ss

torch.set_num_threads(1)
TUNING_SEEDS = tuple(range(950, 960))
SCHNELL_SEED = 959


# ============================================================== CLI / Speichern
def cli(kid: str, main):
    a = argparse.ArgumentParser(description=f"Erkundung {kid} (Tuning-Seeds 950-959, nie Mess-Seeds)")
    a.add_argument("--seed", type=int, default=SCHNELL_SEED)
    a.add_argument("--schnell", action="store_true", help="Rauchtest (<= 2 min im Container), schreibt nach roh/schnell/")
    o = a.parse_args()
    if o.seed not in TUNING_SEEDS:
        raise SystemExit(f"Seed {o.seed} ist kein Tuning-Seed (950-959).")
    t = time.time()
    r = main(o.seed, schnell=o.schnell)
    r["_kid"], r["_seed"], r["_schnell"], r["_sekunden"] = kid, o.seed, bool(o.schnell), round(time.time() - t, 1)
    ordner = os.path.join(HIER, "roh", "schnell") if o.schnell else os.path.join(HIER, "roh")
    os.makedirs(ordner, exist_ok=True)
    pfad = os.path.join(ordner, f"{kid}_s{o.seed}.json")
    tmp = pfad + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=1, default=_js)
    os.replace(tmp, pfad)
    print(f"{kid} seed {o.seed}: {r['_sekunden']} s -> {pfad}")
    return r


def _js(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def lg(*a):
    print(*a, flush=True)


# ============================================================== Profile
def profil(schnell: bool) -> dict:
    """Voll: MLP 30 Epochen auf 55 000 Bildern (kb-Profil); LM >= 4000 Schritte, d=128. Schnell: kleine Teilmengen, Rauchtest."""
    if schnell:
        return {"klein": True, "ep": 3, "ep_nt": 2, "T": 150, "T_sub": 150, "ds": ("mnist",)}
    return {"klein": False, "ep": 30, "ep_nt": 5, "T": 4000, "T_sub": 4000, "ds": ("mnist", "fashion")}


# ============================================================== MLP
def trainiere_mlp(ds, seed, dims, ep, lr, klein):
    torch.manual_seed(seed)
    n = kb.baue_bild(list(dims))
    info = kb.trainiere_bild(n, ds, seed, lr, ep, klein=klein)
    return n, info


def acc(netz, ds, klein, teil="test"):
    return kb.bewerte_bild(netz, ds, klein, teil)


def runde_fp16(netz):
    n = copy.deepcopy(netz)
    for p in n.parameters():
        p.data = p.data.half().float()
    return n


def klein_bester(ds, ziel, seed, ep, klein, formen=("flach", "zwei", "trichter"), lrs=(1e-3,)):
    """Kleines dichtes fp16-Modell (haertester Klassik-Gegner): jede Form mit der groessten Breite <= ziel Bytes, je (Form, lr) trainiert, Wahl nach Val-Genauigkeit."""
    best = None
    for fm in formen:
        for lr in lrs:
            torch.manual_seed(seed)
            n, d = kb.form_klein(ds, fm, ziel)
            kb.trainiere_bild(n, ds, seed, lr, ep, klein=klein)
            n = runde_fp16(n)
            va = acc(n, ds, klein, "val")
            if best is None or va > best["val"]:
                best = {"form": fm, "h": d["h"], "dims": d["dims"], "lr": lr, "val": va, "test": acc(n, ds, klein, "test"),
                        "bytes": kb.bytes_fp16_linear(n), "macs": kb.macs_linear(n), "ep": ep, "train_flops": train_flops(kb.macs_linear(n), ds, klein, ep)}
    return best


def train_flops(macs, ds, klein, ep):
    """Trainings-FLOPs der Linearschichten: 6 x MACs x Beispiele x Epochen (Vorwaerts 2, Rueckwaerts 4 je MAC)."""
    n = len(kb.bild_daten(ds, klein)[0])
    return 6.0 * macs * n * ep


# ============================================================== Quantisierung (Gruppenskalen, Blockrotation, gemischte Bits)
def pad_n(i: int, rot) -> int:
    if rot is None:
        return i
    if rot == "voll":
        return 1 << (i - 1).bit_length()
    return int(math.ceil(i / rot) * rot)


def rot_block(n: int, rot, seed: int):
    """Vorzeichen-Hadamard H·D; rot='voll' (wie kb.rot_matrix) oder Blockdiagonale aus Bloecken der Groesse rot (D ueber ganze Laenge)."""
    if rot == "voll" or rot >= n:
        return kb.rot_matrix(n, seed)
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    Hk = kb.hadamard(rot)
    return torch.block_diag(*([Hk] * (n // rot))) * s[None, :]


def quant_g(W, bits: int, g=None):
    """Zeilen- (g=None) oder Gruppenskala (je g Gewichte einer Zeile eigene Skala, MSE-Gitter wie kb.quant_rows); bits=1: Vorzeichen x mittlerer Betrag."""
    o, i = W.shape
    if g is not None and g < i:
        ng = int(math.ceil(i / g))
        Wp = F.pad(W, (0, ng * g - i)).reshape(o * ng, g)
        return quant_g(Wp, bits, None).reshape(o, ng * g)[:, :i]
    if bits == 1:
        return torch.sign(W + 1e-12) * W.abs().mean(1, keepdim=True)
    return kb.quant_rows(W, bits)


def bytes_schicht(o, i, zeilen_bits, rot, gruppe):
    """Bytes einer quantisierten Schicht: Gewichte bits*ip/8 je Zeile, Skalen (Zeile oder je Gruppe) 2 B, Bias 2 B je Zeile, Vorzeichenvektor ip/8 bei Rotation."""
    ip = pad_n(i, rot)
    ns = int(math.ceil(ip / gruppe)) if gruppe else 1
    return float(sum(b * ip / 8 for b in zeilen_bits) + o * ns * 2 + o * 2 + (ip / 8 if rot is not None else 0))


def zeilen_bits_fuer_budget(o, i, rot, gruppe, budget, bmin=1):
    """Gemischte Bits (b_lo / b_lo+1 je Zeile), so dass bytes_schicht <= budget und die Luecke < eine Zeile ist. Rueckgabe Liste der Bits je Zeile."""
    ip = pad_n(i, rot)
    ns = int(math.ceil(ip / gruppe)) if gruppe else 1
    uber = o * ns * 2 + o * 2 + (ip / 8 if rot is not None else 0)
    beta = (budget - uber) * 8 / (o * ip)
    lo = max(bmin, int(math.floor(beta + 1e-9)))
    hi_n = 0 if beta <= lo else min(o, int(math.floor((beta - lo) * o + 1e-9)))
    return [lo + 1] * hi_n + [lo] * (o - hi_n)


def quant_schicht(W, zeilen_bits, rot, gruppe, seed):
    """Simulierte Schicht -> Gewicht in der Originalbasis (o x i). Quantisiert in der gedrehten Basis (Rotation + Auffuellen wie kb)."""
    o, i = W.shape
    if rot is None:
        Wr, Q = W, None
    else:
        ip = pad_n(i, rot)
        Q = rot_block(ip, rot, seed)
        Wr = F.pad(W, (0, ip - i)) @ Q.T
    zb = torch.tensor(zeilen_bits)
    Wq = torch.zeros_like(Wr)
    for b in sorted(set(zeilen_bits)):
        m = zb == b
        Wq[m] = quant_g(Wr[m], int(b), gruppe)
    return (Wq @ Q)[:, :i] if Q is not None else Wq


def quantisiere_netz(netz, bits, rot, gruppe, seed, voll_budget=True, budget_bits=None):
    """Alle nn.Linear quantisieren. Budget je Schicht = kb.bytes_q_linear(o,i,budget_bits,True,pad=True) (= I-048-Ziel, Rotation+Zeilenskala) bei voll_budget,
    dann gemischte Bits bis zu den Bytes; sonst feste Bits. Rueckgabe (Netz, Gesamt-Bytes, Ziel-Bytes)."""
    n = copy.deepcopy(netz)
    ges, ziel = 0.0, 0.0
    k = 0
    for m in n.modules():
        if isinstance(m, nn.Linear):
            k += 1
            o, i = m.weight.shape
            if voll_budget:
                bud = kb.bytes_q_linear(o, i, budget_bits or bits, True, pad=True)
                zb = zeilen_bits_fuer_budget(o, i, rot, gruppe, bud)
            else:
                bud, zb = 0.0, [bits] * o
            m.weight.data = quant_schicht(m.weight.data, zb, rot, gruppe, seed * 101 + k)
            ges += bytes_schicht(o, i, zb, rot, gruppe)
            ziel += bud
    return n, ges, ziel


def relfehler(netz, netz_q):
    """Relativer Gewichtsfehler ||W - Wq||^2 / ||W||^2 ueber alle Linearschichten (Zusatzmass)."""
    a = b = 0.0
    for m, mq in zip(netz.modules(), netz_q.modules()):
        if isinstance(m, nn.Linear):
            a += float(((m.weight.detach() - mq.weight.detach()) ** 2).sum())
            b += float((m.weight ** 2).sum())
    return a / b


def kurtosis_zeilen(W):
    """Mittlere Exzess-Kurtosis der Zeilen von W."""
    x = W - W.mean(1, keepdim=True)
    v = (x ** 2).mean(1)
    k = (x ** 4).mean(1) / v.clamp_min(1e-20) ** 2 - 3
    return float(k.mean())


# ============================================================== Byte-LM (relu2-Option, Top-K-Ueberspringen)
class Block2(kb.LMBlock):
    def __init__(self, d, h, f, akt="gelu"):
        super().__init__(d, h, f)
        self.akt, self.topk = akt, None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        z = self.a(self.ln2(x))
        z = F.gelu(z) if self.akt == "gelu" else F.relu(z) ** 2
        if self.topk:
            idx = z.topk(self.topk, dim=-1).indices
            z = z * torch.zeros_like(z).scatter_(-1, idx, 1.0)
        return x + self.b(z)


class LM2(kb.LM):
    def __init__(self, d=128, lagen=4, akt="gelu"):
        super().__init__(d, lagen)
        self.bl = nn.ModuleList([Block2(d, self.hk, self.f, akt) for _ in range(lagen)])


def setze_topk(netz, K):
    for b in netz.bl:
        b.topk = K


def baue_lm(seed, akt="gelu", d=128, lagen=4):
    torch.manual_seed(seed)
    return LM2(d, lagen, akt)


def lm_linear_namen(netz):
    return [n for n, m in netz.named_modules() if isinstance(m, nn.Linear)]


def quantisiere_lm_sel(netz, bits, wahl, seed):
    """wahl(name) -> 'roh' | 'had' | 'orth' | None (None = Schicht bleibt fp32). Ersetzt die gewaehlten nn.Linear durch kb.QLinear."""
    n = copy.deepcopy(netz)
    k = [0]
    ges = [0.0]

    def ers(m, pre):
        for name, c in list(m.named_children()):
            full = pre + name
            if isinstance(c, nn.Linear):
                k[0] += 1
                md = wahl(full)
                if md is not None:
                    q = kb.QLinear(c, bits, md, seed * 101 + (5000 if md == "orth" else 0) + k[0])
                    setattr(m, name, q)
                    ges[0] += q.bytes()
                else:
                    ges[0] += 2 * (c.weight.numel() + c.bias.numel())
            else:
                ers(c, full + ".")

    ers(n, "")
    return n, ges[0]


def lm_bpc(netz, klein, teil="test"):
    return kb.bewerte_lm(netz, klein, teil)


def lm_flops_je_token(netz):
    d = netz.d
    lin = sum(m.weight.numel() for m in netz.modules() if isinstance(m, nn.Linear))
    return kb.lm_macs_gesamt(d, netz.lagen, lin)


# ============================================================== Hilfen fuer eigene Quantisierer
def rotiere(W, rot, seed):
    """(W in gedrehter Basis, Q); rot=None: unveraendert."""
    if rot is None:
        return W, None
    ip = pad_n(W.shape[1], rot)
    Q = rot_block(ip, rot, seed)
    return F.pad(W, (0, ip - W.shape[1])) @ Q.T, Q


def zurueck(Wq, Q, i):
    return (Wq @ Q)[:, :i] if Q is not None else Wq


def abbilde_netz(netz, fn):
    """Kopie, in der jede nn.Linear-Gewichtsmatrix durch fn(W, laufende_Nr) ersetzt wird."""
    n = copy.deepcopy(netz)
    k = 0
    for m in n.modules():
        if isinstance(m, nn.Linear):
            k += 1
            m.weight.data = fn(m.weight.data, k)
    return n


def mlp_formen(h, n_ein=784, n_aus=10):
    return ((h, n_ein), (h, h), (n_aus, h))


def ziel_mlp(h, bits):
    """I-048-Ziel (Rotation + Zeilenskala, MNIST-Eingang aufgefuellt) fuer ein MLP 784-h-h-10."""
    return int(math.floor(sum(kb.bytes_q_linear(o, i, bits, True, pad=True) for o, i in mlp_formen(h))))
