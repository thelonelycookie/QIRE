"""K20 -- Beschränkte Schwellen-Aktivierung (Hill, n=2) (Erkundung Welle 2, KEIN Befund).

MECHANISMUS: a = z²/(z²+θ²) liegt in [0,1) (θ gelernt, ein Skalar je Lage) und lässt sich auf festem Gitter [0,1] ohne Ausreißerbehandlung in 3-4 Bit speichern.
TEST: Byte-LM d=128, 4 Lagen, FFN 128->512->128 (kompression_bytes.LM-Aufbau), T = 4000 Schritte (schnell: 40), Aktivierung der FFN-Mitte wird im Training und bei
  der Auswertung mit STE auf `bits` Bit quantisiert (Standard 4 Bit; Aufruf-Variable BITS, Umgebung K20_BITS=3 für 3 Bit). Lernrate je Arm: Kurzläufe 250 Schritte
  mit {1e-3, 3e-3} auf Validierung (250 Schritte), dann voller Lauf. Zusätzlich wird das fertige Modell nachträglich bei 2/3/4/6 Bit und ohne Quantisierung ausgewertet.
ARME: hill (beschränkt, festes Gitter [0,1])  |  gelu (unbeschränkt, Min-Max-Gitter je Token, 4 B Skala+Nullpunkt je Token und Lage)  |  z2 (z², wie gelu)  |
  relu6 (beschränkt auf [0,6], festes Gitter)  [KONTROLLE: beschränkt]  |  gelu_klemmt (GELU auf [-0,17..4] geklemmt, festes Gitter; nur mit K20_EXTRA=1, Zeitbudget)
Ausweis: Parameter (gleich bis auf 4 θ), FFN-FLOPs je Token (gleich), Aktivierungs-Bytes je Token (Lagen · (512·bits/8 + Skalenkosten)), bpc Test je Aktivierungs-Bytes.
Nur Tuning-Seeds. Quanten-inspiriert, klassisch (I-038).
"""
from __future__ import annotations
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hilfe_k17_20 as H
import torch
import torch.nn as nn
import torch.nn.functional as F
import abteilungen.struktur_sprache as ss
import abteilungen.kompression_bytes as kb

KID = "K20"
BITS = int(os.environ.get("K20_BITS", "4"))
D, FB, LAGEN = 128, 512, 4


def quant_fest(a, bits, lo, hi):
    L = 2 ** bits - 1
    q = torch.round(((a - lo) / (hi - lo)).clamp(0, 1) * L) / L * (hi - lo) + lo
    return a + (q - a).detach()


def quant_minmax(a, bits):
    lo, hi = a.amin(-1, keepdim=True), a.amax(-1, keepdim=True)
    L = 2 ** bits - 1
    sp = (hi - lo).clamp_min(1e-8)
    q = torch.round((a - lo) / sp * L) / L * sp + lo
    return a + (q - a).detach()


class Akt(nn.Module):
    def __init__(self, art):
        super().__init__()
        self.art, self.bits = art, None
        if art == "hill":
            self.rho = nn.Parameter(torch.zeros(()))        # θ = softplus(rho)/softplus(0) -> 1 am Start

    def forward(self, z):
        k = self.art
        if k == "hill":
            th = F.softplus(self.rho) / math.log(2.0)
            z2 = z * z
            a, rng = z2 / (z2 + th * th + 1e-6), (0.0, 1.0)
        elif k == "gelu":
            a, rng = F.gelu(z), None
        elif k == "z2":
            a, rng = z * z, None
        elif k == "relu6":
            a, rng = F.relu6(z), (0.0, 6.0)
        elif k == "gelu_klemmt":
            a, rng = F.gelu(z).clamp(-0.17, 4.0), (-0.17, 4.0)
        else:
            raise ValueError(k)
        if self.bits is None:
            return a
        return quant_minmax(a, self.bits) if rng is None else quant_fest(a, self.bits, *rng)


class Blk(nn.Module):
    def __init__(self, d, h, f, art):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.a, self.b = nn.Linear(d, f), nn.Linear(f, d)
        self.akt = Akt(art)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.b(self.akt(self.a(self.ln2(x))))


class AktLM(kb.LM):
    def __init__(self, art, d=D, lagen=LAGEN, f=FB):
        super().__init__(d, lagen, f)
        self.bl = nn.ModuleList([Blk(d, self.hk, f, art) for _ in range(lagen)])
        self.art = art

    def setze_bits(self, bits):
        for b in self.bl:
            b.akt.bits = bits


def akt_bytes_je_token(art, bits):
    skala = 0 if art in ("hill", "relu6", "gelu_klemmt") else 4
    return LAGEN * (FB * bits / 8 + skala)


def trainiere(art, seed, lr, T, bits, klein):
    torch.manual_seed(seed)
    m = AktLM(art)
    m.setze_bits(bits)
    st = kb.trainiere_lm(m, seed, lr, T, klein=klein)
    return m, st


def main(seed: int, schnell: bool = False) -> dict:
    H.pruefe_seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    T = 40 if schnell else 4000
    Tt = 20 if schnell else 250
    arme = ("hill", "gelu", "z2", "relu6") + (("gelu_klemmt",) if os.environ.get("K20_EXTRA") == "1" else ())
    r = {"kid": KID, "seed": seed, "schnell": schnell, "T": T, "bits_training": BITS, "arme": {}}
    for art in arme:
        t = time.time()
        va, best = {}, None
        for lr in (1e-3, 3e-3):
            m, st = trainiere(art, seed, lr, Tt, BITS, schnell)
            va[lr] = 8.0 if st.get("instabil") else kb.bewerte_lm(m, schnell, "val")
        lr = min(va, key=va.get)
        m, st = trainiere(art, seed, lr, T, BITS, schnell)
        z = {"lr": lr, "tuning_val_bpc": {str(k): v for k, v in va.items()}, "instabil": bool(st.get("instabil"))}
        if not z["instabil"]:
            m.eval()
            ev = {}
            for b in ((None, 4, 2) if schnell else (None, 6, 4, 3, 2)):
                m.setze_bits(b)
                ev["fp" if b is None else f"bits{b}"] = kb.bewerte_lm(m, schnell, "test")
            m.setze_bits(BITS)
            z["bpc_test_nach_bits"] = ev
            z["bpc_test_trainierte_bits"] = ev[f"bits{BITS}"] if f"bits{BITS}" in ev else kb.bewerte_lm(m, schnell, "test")
            z["theta"] = [float(F.softplus(b.akt.rho) / math.log(2.0)) for b in m.bl] if art == "hill" else None
        z["parameter"] = int(sum(p.numel() for p in m.parameters()))
        z["ffn_flops_je_token"] = int(LAGEN * (2 * D * FB * 2))
        z["aktivierungs_bytes_je_token"] = {f"bits{b}": akt_bytes_je_token(art, b) for b in (2, 3, 4, 6)}
        z["sekunden"] = time.time() - t
        r["arme"][art] = z
    if all(not r["arme"][a]["instabil"] for a in arme):
        base = r["arme"]["relu6"]["bpc_test_trainierte_bits"]
        r["differenz_zu_relu6"] = {a: r["arme"][a]["bpc_test_trainierte_bits"] - base for a in arme}
        base = r["arme"]["gelu"]["bpc_test_trainierte_bits"]
        r["differenz_zu_gelu"] = {a: r["arme"][a]["bpc_test_trainierte_bits"] - base for a in arme}
    r["hinweis"] = ("Erkundung, kein Befund. Ein Seed, eine Trainings-Bitbreite; 'bpc_test_nach_bits' bei fremden Bits ist Nachher-Quantisierung eines mit "
                    f"{BITS} Bit trainierten Modells (nicht gleichwertig zu eigenem Training). Lernrate nur 2 Werte.")
    r["sekunden_gesamt"] = time.time() - t0
    H.speichere(KID, seed, r, schnell)
    return r


if __name__ == "__main__":
    H.cli(KID, main)
