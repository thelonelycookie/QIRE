"""Abteilung Kompression -- I-048 „Kompressionsstudie bei GLEICHEN BYTES der Linearschichten“ (L1 der Synthese vom 27.10.2026).
(Grundlage: Erkundungen r1 (gleiche Bytes, MNIST), w4 (Hadamard vor Niedrigbit-Quantisierung), w5 (Rangverteilung nach Verschraenkungsentropie), r3 (Kontrolle zu w7); alle auf Tuning-Seeds.
 Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“ (Hadamard-Gatter als Basiswechsel, Schmidt-Spektrum als Rangbudget), nicht „Quantenvorteil“.
 Es ist eine reine SPEICHER-/BYTE-Aussage: Rotation + Niedrigbit spart Bytes, nicht FLOPs (Bytes UND MACs werden beide ausgewiesen).)

FRAGE: Schlaegt ein GROSSES Modell, dessen Linearschichten nachtraeglich per Basiswechsel (Hadamard) und 2/3 Bit gespeichert werden, ein KLEINES dichtes fp16-Modell mit denselben Bytes?
  Ist der Gewinn eine Hadamard-Eigenschaft oder genuegt jede zufaellige Orthogonale (Kontrolle)? Hilft die Rangverteilung nach Verschraenkungsentropie gegenueber gleichmaessigem Rang bei gleichen Parametern?

DATENSAETZE / MODELLE (volles Training; Bild-Daten fest gesplittet, Seeds wirken auf Initialisierung und Batchreihenfolge):
  MNIST, Fashion-MNIST: MLP 784-512-512-10 (ReLU), 55 000 Train / 5 000 Val / 10 000 offizieller Test, 30 Epochen Adam (5 % Aufwaermen, Kosinus), Batch 128, kein Clip. Eingang 0..1.
  Byte-LM (UD_German-GSD, Byte-Ebene, fester Split aus struktur_sprache.py): Decoder-Transformer d=128, 4 Lagen, 2 Koepfe (Kopfdim 64), FFN 4d, Kontext 128, Batch 8x128,
    T = 16 000 Schritte (>= 8000; ca. 10 Epochen ueber die Trainingsbytes, wie I-045-v4/I-047), Adam, Kosinus, 5 % Aufwaermen, Clip 1,0. Test-bpc ueber den ganzen Testsatz (99 487 Bytes, wie v4).
  „Linearschichten“ = alle nn.Linear der Blocklagen (LM: qkv, proj, FFN a/b; Bild: alle drei). Einbettung, Positionen, LayerNorm bleiben in allen Armen fp (nicht im Budget; Gesamt-Bytes werden zusaetzlich ausgewiesen).

BYTE-BUDGET (je Datensatz und Bitzahl b in {3, 2}; Hauptpunkt 2 Bit; 3 Bit gleichrangig beschrieben; LM zusaetzlich 4 Bit beschreibend):
  Ziel(b) = Bytes des GROSSEN Modells mit Hadamard-Rotation bei b Bit: Gewichte b/8 B, Skala je Zeile 2 B, Bias je Zeile 2 B, Vorzeichenvektor n/8 B je Schicht (MNIST-Eingang 784 -> 1024 aufgefuellt).
  Alle anderen Arme duerfen hoechstens Ziel(b) Bytes haben (fp16 = 2 B je Gewicht und Bias; Skalen/Bias wie oben). Die Orthogonal-Kontrolle wird mit denselben Bytes angesetzt (Q aus dem Seed
  erzeugbar), hat aber n^2 statt n·log2(n) Zusatzoperationen (ausgewiesen).
ARME (je Budget):
  roh        gross, nachtraeglich roh quantisiert + Clip (je Zeile MSE-Skala, 13 Werte 0,4..1,0 x max)
  had        gross, Hadamard H·D/sqrt(n) (D zufaellige Vorzeichen je Schicht und Seed) + Clip       <- Hauptarm
  orth       gross, Zufalls-Orthogonale (Haar, QR) + Clip                                            <- Kontrolle
  klein      kleines DICHTES fp16-Modell von Grund auf trainiert; Form aus 3 Formen je Datensatz auf der Validierung gewaehlt (Breite getunt), jeweils groesste Breite <= Ziel
             Bild: flach (1 verdeckte Schicht), zwei (h,h), trichter (h, h/2); LM: d=64 (FFN-Breite angepasst), d=48 (FFN angepasst), d=64 mit 2 Lagen (FFN angepasst)
  nr         Niedrigrang fp16 (W = U V) von Grund auf trainiert, gleicher Rang in allen Linearschichten (groesster Rang <= Ziel, Rest +1 nach Kosten); Bild: letzte Schicht dicht
  svd_gleich     Nachkompression des fertig trainierten gross-fp32: Truncated-SVD, gleicher Rang je Matrix (+Rest), fp16, dann Nachtraining
  svd_entropie   dasselbe, Rang je Matrix ~ exp(S) mit S = Verschraenkungsentropie der normierten Singulaerwerte (Wasserfuell-Verteilung mit Obergrenze, Rest nach Bruchteil), gleiche Parameter
  (beschreibend) fp = gross-fp32 (Bezug); ref64 = LM d=64 mit FFN 4d in fp16 (nicht budgetgleich fuer 3/2 Bit; gleichbyte-Gegner fuer 4 Bit); LM 4 Bit: roh/had/orth.
LERNRATE UND FORM JE ARM auf der VALIDIERUNG des Tuning-Seeds 2890 (gittersuche_stabil mit Stabilitaetsabstand): gross nach fp-Validierung (die quantisierten Arme erben sie; konservativ gegen H1),
  klein je Form und Bitbudget (danach beste Form nach Val), nr je Bitbudget, Nachtraining svd_* je Variante und Bitbudget (eigene Lernrate-Gitter). Gilt fuer alle Mess-Seeds.
  Nachtraining: Bild 5 Epochen, LM 2000 Schritte (Aufwaermen/Kosinus wie oben).
MESS-SEEDS 2800-2809 (10), Negativkontrolle 2810-2811, Tuning/Rauchtest 2890.

HYPOTHESEN (Test-Metrik: Bild Genauigkeit in pp, LM Test-bpc; „Vorsprung“ je Seed so gepaart, dass POSITIV = erstgenannter Arm besser; gepaarter t-Test, n = 10, alpha 0,05):
  H1 (Holm ueber die 3 Datensaetze, 2 Bit): had >= klein + Rand: Vorsprung(had, klein) > 0,2 pp (Bild) bzw. > 0,02 bpc (LM).  HADAMARD_BESSER | H0 (Vorsprung sicher <= Rand) | HU.
  H2 (Holm ueber 6 Zellen = 3 Datensaetze x {3, 2} Bit): had ≡ orth, TOST mit Rand ±0,2 pp (Bild) bzw. ±0,02 bpc (LM).  AEQUIVALENT | UNTERSCHIEDLICH (zweiseitig p < 0,05, nicht aequivalent) | HU.
  H3 (Holm ueber 6 Zellen = 3 Datensaetze x {3, 2} Bit): svd_entropie besser als svd_gleich (gleiche Parameter, Rand 0).  ENTROPIE_BESSER | H0 (gleichmaessig sicher besser) | HU.
  H1b (3 Bit, beschreibend, gleicher Test ohne Holm). Alles andere beschreibend (Bytes, MACs, Zusatzoperationen, Gesamt-Bytes, Abstand roh/had, nr gegen klein, 4 Bit gegen ref64 im LM, Rangverteilungen).
DIVERGENZ / KOLLAPS: trainierter Arm diverg (Verlust nicht endlich, Bild-Acc < 50 %, LM-bpc >= Unigramm-Entropie) oder Seed schlechter als Zellmedian um > 5 pp (Bild) / > 0,2 bpc (LM) -> beteiligte Hypothesen = HU;
  Seed bleibt in den Zahlen. Parametergleichheit svd_entropie/svd_gleich auf 1 % noetig (sonst H3 = HU).
NEGATIVKONTROLLE (Seeds 2810/2811, je Datensatz; gross-fp und had 2 Bit): Bild: permutierte Trainingslabels, 2 Epochen, mittlere Test-Acc <= 14 %; LM: permutierte Trainingsbytes, 1000 Schritte,
  mittlerer Test-bpc >= 0,9 x Unigramm. Durchgefallen -> alle Hypothesen dieses Datensatzes = HU.
WIEDERAUFNAHME je Aufgabe (ergebnisse/_zwischen_I-048.json), Checkpoint im Training (Bild alle 1000, LM alle 2000 Schritte), fertige Gross-Modelle in ergebnisse/_ckpt_I-048/modelle/.
"""
from __future__ import annotations

import copy
import heapq
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import abteilungen.struktur_sprache as ss
from abteilungen.clifford_suche import holm
from abteilungen.struktur_gross import _lade_zwischen, _rechne, _schreibe_zwischen
from labor.gemeinsam import ERGEBNISSE, _roh, gittersuche_stabil, zusammenfassung

KARTE = "I-048"
SEEDS = tuple(range(2800, 2810))
NK_SEEDS = (2810, 2811)
TUNING_SEED = 2890
DATENSAETZE = ("mnist", "fashion", "lm")
BILD = ("mnist", "fashion")
BITS = (3, 2)                      # Hauptbudgets
BITS_ALLE = (4, 3, 2)              # 4 Bit beschreibend
MODI = ("roh", "had", "orth")
FORMEN_BILD = ("flach", "zwei", "trichter")
FORMEN_LM = ("d64", "d48", "d64L2")
ALPHA = 0.05
RAND = {"mnist": 0.2, "fashion": 0.2, "lm": 0.02}          # pp bzw. bpc
NK_ACC_MAX = 0.14
NK_FAKTOR = 0.9
KOLLAPS_BILD_PP = 5.0
KOLLAPS_LM = 0.2
PARAM_TOLERANZ = 0.01
_BELEGT = (set(range(2000, 2070)) | set(range(2090, 2120)) | set(range(2150, 2160)) | set(range(900, 1001)) | set(range(2200, 2205)) | set(range(2400, 2420))
           | set(range(2600, 2630)) | set(range(2660, 2670)) | set(range(2670, 2675)) | set(range(2680, 2690)) | {2498, 2499, 2692, 2693, 2698, 2699})
LR_GITTER = (3e-4, 1e-3, 3e-3)
LR_FAKTOR = 3
NT_GITTER = (1e-4, 3e-4, 1e-3)
KOLLAPS_SCORE = 0.02               # Tuning: Score = Val-Acc bzw. 1 - bpc/8 (0,02 = 2 pp bzw. 0,16 bpc)

PROFILE = {"standard": {"bild_ep": 30, "bild_nt_ep": 5, "lm_T": 16000, "lm_nt_T": 2000, "nk_ep": 2, "nk_T": 1000, "ck_bild": 1000, "ck_lm": 2000,
                        "seeds": SEEDS, "nk_seeds": NK_SEEDS, "tuning_seed": TUNING_SEED}}

HYPOTHESEN = {
    "H1": "had (2 Bit) >= klein (fp16, gleiche Bytes) + Rand: Vorsprung > 0,2 pp (MNIST, Fashion) bzw. > 0,02 bpc (LM); gepaart einseitig t, n=10, Holm ueber 3 Datensaetze. HADAMARD_BESSER | H0 (Vorsprung sicher <= Rand) | HU.",
    "H2": "had ≡ orth bei 3 und 2 Bit: TOST, Rand ±0,2 pp bzw. ±0,02 bpc, Holm ueber 6 Zellen. AEQUIVALENT | UNTERSCHIEDLICH | HU.",
    "H3": "svd_entropie besser als svd_gleich (gleiche Parameter), einseitig t, Holm ueber 6 Zellen (3 Datensaetze x {3,2} Bit). ENTROPIE_BESSER | H0 (gleichmaessig sicher besser) | HU.",
    "H1b": "wie H1 bei 3 Bit, beschreibend (kein Holm).",
    "DIVERGENZ": "trainierter Arm diverg oder Kollaps (Bild > 5 pp, LM > 0,2 bpc ueber Zellmedian): beteiligte Hypothese HU, Seed bleibt in den Zahlen.",
    "NK": "Negativkontrolle je Datensatz (Seeds 2810/2811): Bild mittlere Test-Acc <= 14 % bei permutierten Labels, LM mittlerer bpc >= 0,9 x Unigramm bei permutierten Bytes. Sonst alle Hypothesen des Datensatzes HU.",
    "B": "beschreibend: alle Zellen (Mittel ± sd), Bytes, MACs, Zusatzoperationen, Gesamt-Bytes, Lernraten, Formen, Raenge, Abstand roh/had, nr gegen klein, 4 Bit gegen ref64.",
}

# Sekunden je Schritt (1 Thread, Container) bzw. je Epoche; wird mit zeiten_messen() gemessen (Stand siehe LABORBUCH). Bild: je Epoche.
S_JE = {("bild", "gross"): 3.5, ("bild", "klein"): 1.2, ("bild", "nr"): 1.2,
        ("lm", "gross"): 0.09, ("lm", "klein"): 0.05, ("lm", "nr"): 0.065, ("lm", "ref64"): 0.04}
# Gemessen 27.10.2026 im Container (4 Kerne, 1 Thread, Median ueber 15 bzw. 30 Schritte): LM gross 0,083 s/Schritt, klein (d=64) 0,031, ref64 0,035; Bild gross 3,45 s/Epoche, klein (h=110) 0,83.


# ================================================================== Daten
_BD = {}


def bild_daten(ds: str, klein=False):
    """(xtr, ytr, xva, yva, xte, yte); 55 000 / 5 000 / 10 000 (offizieller Test). klein = Rauchtest."""
    if (ds, klein) not in _BD:
        x, y = _roh("mnist" if ds == "mnist" else "fashion_mnist")
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        t = (x[:55000], y[:55000], x[55000:60000], y[55000:60000], x[60000:], y[60000:])
        _BD[(ds, klein)] = (t[0][:2000], t[1][:2000], t[2][:500], t[3][:500], t[4][:1000], t[5][:1000]) if klein else t
    return _BD[(ds, klein)]


# ================================================================== Modelle
class LRLin(nn.Module):
    """Niedrigrang-Linearschicht W = U V (V: i -> r ohne Bias, U: r -> o mit Bias)."""

    def __init__(self, i, o, r):
        super().__init__()
        self.v = nn.Linear(i, r, bias=False)
        self.u = nn.Linear(r, o)

    def forward(self, x):
        return self.u(self.v(x))


class Netz(nn.Module):
    """Bild-MLP aus beliebigen Linearschichten (ReLU dazwischen)."""

    def __init__(self, schichten):
        super().__init__()
        self.ls = nn.ModuleList(schichten)

    def forward(self, x):
        for k, l in enumerate(self.ls):
            x = l(x)
            if k < len(self.ls) - 1:
                x = F.relu(x)
        return x


def baue_bild(dims):
    """dims = verdeckte Breiten, z. B. [512, 512]."""
    ein = [784] + list(dims)
    aus = list(dims) + [10]
    return Netz([nn.Linear(i, o) for i, o in zip(ein, aus)])


def koepfe(d: int) -> int:
    return d // 64 if d % 64 == 0 and d >= 128 else (d // 32 if d % 32 == 0 else d // 16)


class LMBlock(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.a, self.b = nn.Linear(d, f), nn.Linear(f, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.b(F.gelu(self.a(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, d=128, lagen=4, f=None, h=None, ctx=ss.KONTEXT):
        super().__init__()
        f = f or 4 * d
        h = h or koepfe(d)
        self.d, self.lagen, self.f, self.hk = d, lagen, f, h
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([LMBlock(d, h, f) for _ in range(lagen)])
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.bl:
            x = b(x)
        return self.ln(x) @ self.emb.weight.T


def lm_f_fit(d: int, lagen: int, ziel: float) -> int:
    """Groesste FFN-Breite f (Vielfaches von 8) mit fp16-Bytes der Linearschichten <= ziel. Je Lage: Gewichte 4d² + 2df, Biases 5d + f."""
    f = int((ziel / (2 * lagen) - 4 * d * d - 5 * d) / (2 * d + 1))
    return max(8, f - f % 8)


# ================================================================== Zaehlungen (Bytes, MACs)
def linear_namen(netz, ausnahme_letzte=False):
    """Namen aller nn.Linear (Reihenfolge der Module); fuer Bild-MLP ohne die letzte Schicht, wenn ausnahme_letzte."""
    namen = [n for n, m in netz.named_modules() if isinstance(m, nn.Linear)]
    return namen[:-1] if ausnahme_letzte else namen


def bytes_fp16_linear(netz) -> int:
    """fp16: 2 B je Gewicht und Bias aller nn.Linear (LRLin zaehlt ueber seine beiden Linear)."""
    return int(sum(2 * (m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)) for m in netz.modules() if isinstance(m, nn.Linear)))


def bytes_q_linear(o: int, i: int, bits: int, rot: bool, pad: bool = False) -> float:
    ip = 1024 if (pad and i == 784) else i
    return o * ip * bits / 8 + o * 2 + o * 2 + (ip / 8 if rot else 0)


def bytes_q_gross(ds: str, bits: int, rot: bool = True) -> float:
    if ds in BILD:
        formen = ((512, 784), (512, 512), (10, 512))
        return float(sum(bytes_q_linear(o, i, bits, rot, pad=True) for o, i in formen))
    d, f = 128, 512
    return float(4 * sum(bytes_q_linear(o, i, bits, rot) for o, i in ((3 * d, d), (d, d), (f, d), (d, f))))


def ziel_bytes(ds: str, bits: int) -> int:
    return int(math.floor(bytes_q_gross(ds, bits, True)))


def macs_linear(netz) -> int:
    return int(sum(m.weight.numel() for m in netz.modules() if isinstance(m, nn.Linear)))


def lm_macs_gesamt(d, lagen, lin_macs, ctx=ss.KONTEXT) -> int:
    """MACs je Token: Linearschichten + Attention (QK und AV, im Mittel (ctx+1)/2 Schluessel) + Ausgabekopf 256·d."""
    return int(lin_macs + lagen * d * (ctx + 1) + 256 * d)


def bytes_rest_fp16(netz) -> int:
    """fp16-Bytes aller NICHT-Linear-Parameter (Einbettung, Positionen, LayerNorm); in allen Armen gleich behandelt."""
    lin = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0) for m in netz.modules() if isinstance(m, nn.Linear))
    return int(2 * (sum(p.numel() for p in netz.parameters()) - lin))


# ================================================================== Quantisierung (aus w4/r1 uebernommen)
def hadamard(n: int):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def rot_matrix(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    return hadamard(n) * s[None, :]


def orth_matrix(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q * torch.sign(torch.diagonal(r))[None, :]


def quant_rows(W, bits: int):
    """Symmetrisch je Zeile, Rundung zum Naechsten, Skala je Zeile per MSE-Gittersuche (13 Werte 0,4..1,0 x max) = „Clip“."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bs = None, None
    for f in np.linspace(0.4, 1.0, 13):
        s = amax * float(f) / qmax
        Wq = (W / s).round().clamp(-qmax, qmax) * s
        e = ((W - Wq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs = e, Wq
        else:
            m = e < best
            best = torch.where(m, e, best)
            bs = torch.where(m, Wq, bs)
    return bs


class QLinear(nn.Module):
    """Linearschicht mit nachtraeglich quantisierten Gewichten (simuliert). modus: roh | had | orth."""

    def __init__(self, lin, bits, modus, seed, pad_to=None):
        super().__init__()
        W = lin.weight.data.clone()
        self.pad = (pad_to - W.shape[1]) if (pad_to and modus != "roh") else 0          # pad_to: aufgefuellte Eingangsbreite (nur MNIST-Eingang 784 -> 1024)
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.modus, self.bits = modus, bits
        self.Q = rot_matrix(W.shape[1], seed) if modus == "had" else (orth_matrix(W.shape[1], seed) if modus == "orth" else None)
        self.W = quant_rows(W @ self.Q.T if self.Q is not None else W, bits)
        self.b = lin.bias.data.clone()
        self.shape = tuple(W.shape)

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (0, self.pad))
        if self.Q is not None:
            x = x @ self.Q.T
        return F.linear(x, self.W, self.b)

    def bytes(self):
        o, i = self.shape
        return o * i * self.bits / 8 + o * 2 + o * 2 + (i / 8 if self.Q is not None else 0)

    def zaehlung(self):
        """(MAC, Zusatz-Additionen der schnellen Hadamard-Transformation, Zusatz-MACs der dichten Orthogonalen)."""
        o, i = self.shape
        return o * i, (i * int(math.log2(i)) if self.modus == "had" else 0), (i * i if self.modus == "orth" else 0)


def quantisiere(netz, bits, modus, seed, pad_to=None):
    """Ersetzt alle nn.Linear durch QLinear; pad_to = {Eingangsbreite: aufgefuellte Breite}, z. B. {784: 1024}."""
    netz = copy.deepcopy(netz)
    k = [0]

    def ersetze(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, QLinear(c, bits, modus, seed * 101 + (5000 if modus == "orth" else 0) + k[0], pad_to=(pad_to or {}).get(c.in_features)))
            else:
                ersetze(c)

    ersetze(netz)
    return netz


def stat_q(netz):
    q = [m for m in netz.modules() if isinstance(m, QLinear)]
    z = [m.zaehlung() for m in q]
    return {"bytes": float(sum(m.bytes() for m in q)), "mac": int(sum(a for a, _, _ in z)), "add": int(sum(b for _, b, _ in z)), "mac_orth": int(sum(c for _, _, c in z))}


# ================================================================== Niedrigrang, Entropie-Rang
def lowrank_ersetzen(netz, raenge: dict, quelle=None):
    """Ersetzt nn.Linear (Namen in raenge) durch LRLin. quelle=None: Zufallsinit; sonst Truncated-SVD der Gewichte (U√s, √s Vᵀ)."""
    for name, r in raenge.items():
        eltern = netz
        teile = name.split(".")
        for t in teile[:-1]:
            eltern = getattr(eltern, t)
        lin = getattr(eltern, teile[-1])
        o, i = lin.weight.shape
        neu = LRLin(i, o, r)
        if quelle is not None:
            U, s, Vh = torch.linalg.svd(lin.weight.data.double(), full_matrices=False)
            sq = s[:r].sqrt()
            neu.u.weight.data = (U[:, :r] * sq[None, :]).float()
            neu.v.weight.data = (sq[:, None] * Vh[:r]).float()
            neu.u.bias.data = lin.bias.data.clone()
        setattr(eltern, teile[-1], neu)
    return netz


def eff_raenge(netz, namen) -> dict:
    """Effektiver Rang exp(S) je Matrix, S = −Σ p ln p mit p = s²/Σs² (Schmidt-Spektrum)."""
    aus = {}
    mods = dict(netz.named_modules())
    for n in namen:
        s = torch.linalg.svdvals(mods[n].weight.data.double())
        p = s ** 2 / (s ** 2).sum()
        aus[n] = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum()))
    return aus


def rang_budget(formen, P: int, art: str, eff=None):
    """Raenge je Matrix (o, i) mit Σ r·(o+i) <= P. gleich: gleiche Gewichte; entropie: Gewichte exp(S).
    Wasserfuellen mit Obergrenze r <= min(o, i); danach Abrunden und Rest nach groesstem Bruchteil (bei Gleichstand billigste Matrix) auffuellen."""
    n = len(formen)
    kosten = [o + i for o, i in formen]
    maxr = [min(o, i) for o, i in formen]
    w = [1.0] * n if art == "gleich" else [float(e) for e in eff]
    r = [0.0] * n
    frei, rest = set(range(n)), float(P)
    while frei:
        c = rest / sum(w[i] * kosten[i] for i in frei)
        gedeckelt = [i for i in frei if c * w[i] > maxr[i]]
        if not gedeckelt:
            for i in frei:
                r[i] = c * w[i]
            break
        for i in gedeckelt:
            r[i] = float(maxr[i])
            rest -= maxr[i] * kosten[i]
            frei.discard(i)
    ri = [max(1, min(maxr[i], int(math.floor(r[i] + 1e-9)))) for i in range(n)]
    benutzt = sum(ri[i] * kosten[i] for i in range(n))
    while True:
        kand = [i for i in range(n) if ri[i] < maxr[i] and benutzt + kosten[i] <= P]
        if not kand:
            break
        i = max(kand, key=lambda j: (r[j] - ri[j], -kosten[j]))
        ri[i] += 1
        benutzt += kosten[i]
    return ri


def param_lr_budget(ds: str, ziel: int, netz_gross) -> tuple:
    """(P_rang, namen, formen): Parameter, die fuer die Faktoren der Niedrigrang-Matrizen uebrig bleiben (fp16: 2 B/Param), Namen/Formen der ersetzten Matrizen."""
    bild = ds in BILD
    namen = linear_namen(netz_gross, ausnahme_letzte=bild)
    mods = dict(netz_gross.named_modules())
    formen = [tuple(mods[n].weight.shape) for n in namen]
    fest = 0
    if bild:
        l3 = netz_gross.ls[-1]
        fest += l3.weight.numel() + l3.bias.numel()
    fest += sum(o for o, _ in formen)                      # Biases der ersetzten Matrizen (an U)
    return int(ziel // 2 - fest), namen, formen


# ================================================================== Training (Schrittschleife mit Checkpoint)
class _Abbruch(Exception):
    """Nur fuer den Vortest: simuliert einen harten Abbruch (24-h-Limit) mitten in einer Aufgabe."""


def _ckpt_pfad(ordner, key) -> str:
    return str(Path(ordner) / (key.replace("|", "_").replace("/", "-") + ".pt"))


def _schrittlauf(netz, n, lr, schritt_fn, ckpt, ckpt_alle, meta, g, clip=None, stopp_bei=None):
    """Adam, 5 % Aufwaermen + Kosinus; Checkpoint alle `ckpt_alle` Schritte (Modell, Optimierer, Plan, Generator, Zufallszustand). Wiederaufnahme nur bei gleichen Metadaten."""
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, n // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm))))
    start, t_vor = 0, 0.0
    if ckpt and Path(ckpt).exists():
        try:
            z = torch.load(ckpt, weights_only=True)
        except Exception:
            z = None
        if z is not None and z.get("meta") == meta:
            netz.load_state_dict(z["modell"])
            opt.load_state_dict(z["opt"])
            sched.load_state_dict(z["sched"])
            g.set_state(z["g"])
            torch.set_rng_state(z["rng"])
            start, t_vor = int(z["schritt"]), float(z["sekunden"])
    t0 = time.time()
    netz.train()
    for s in range(start, n):
        l = schritt_fn(s)
        if not torch.isfinite(l):
            if ckpt and Path(ckpt).exists():
                Path(ckpt).unlink()
            return {"instabil": True, "fortgesetzt_ab": start, "sekunden": t_vor + time.time() - t0}
        opt.zero_grad(set_to_none=True)
        l.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(netz.parameters(), clip)
        opt.step()
        sched.step()
        if ckpt and (s + 1) % ckpt_alle == 0 and s + 1 < n:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt + ".tmp"
            torch.save({"meta": meta, "modell": netz.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "g": g.get_state(),
                        "rng": torch.get_rng_state(), "schritt": s + 1, "sekunden": t_vor + time.time() - t0}, tmp)
            os.replace(tmp, ckpt)
        if stopp_bei is not None and s + 1 == stopp_bei:
            raise _Abbruch(f"simulierter Abbruch bei Schritt {s + 1}")
    if ckpt and Path(ckpt).exists():
        Path(ckpt).unlink()
    return {"instabil": False, "fortgesetzt_ab": start, "sekunden": t_vor + time.time() - t0}


def trainiere_bild(netz, ds, seed, lr, ep, klein=False, perm_labels=False, ckpt=None, ckpt_alle=1000, tag="", stopp_bei=None):
    torch.set_num_threads(1)
    xtr, ytr, _, _, _, _ = bild_daten(ds, klein)
    if perm_labels:
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 77))]
    N, bs = len(xtr), 128
    spe = math.ceil(N / bs)
    cache = {"e": -1, "p": None}

    def schritt(s):
        e, k = divmod(s, spe)
        if cache["e"] != e:
            cache["p"] = torch.randperm(N, generator=torch.Generator().manual_seed(seed * 7919 + e))
            cache["e"] = e
        b = cache["p"][k * bs:(k + 1) * bs]
        return F.cross_entropy(netz(xtr[b]), ytr[b])

    meta = {"ds": ds, "seed": int(seed), "lr": float(lr), "ep": int(ep), "klein": bool(klein), "perm": bool(perm_labels), "tag": tag, "np": int(sum(p.numel() for p in netz.parameters()))}
    return _schrittlauf(netz, ep * spe, lr, schritt, ckpt, ckpt_alle, meta, torch.Generator().manual_seed(seed + 5), clip=None, stopp_bei=stopp_bei)


def trainiere_lm(netz, seed, lr, T, klein=False, perm_bytes=False, ckpt=None, ckpt_alle=2000, tag="", stopp_bei=None):
    torch.set_num_threads(1)
    tr, _, _ = ss.daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    if perm_bytes:
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 77))]
    offs = torch.arange(ss.KONTEXT)

    def schritt(s):
        i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        return F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))

    meta = {"seed": int(seed), "lr": float(lr), "T": int(T), "klein": bool(klein), "perm": bool(perm_bytes), "tag": tag, "np": int(sum(p.numel() for p in netz.parameters()))}
    return _schrittlauf(netz, T, lr, schritt, ckpt, ckpt_alle, meta, g, clip=1.0, stopp_bei=stopp_bei)


def bewerte_bild(netz, ds, klein, teil):
    xtr, ytr, xva, yva, xte, yte = bild_daten(ds, klein)
    X, Y = (xva, yva) if teil == "val" else (xte, yte)
    netz.eval()
    with torch.no_grad():
        return float((netz(X).argmax(1) == Y).float().mean())


def bewerte_lm(netz, klein, teil):
    _, va, te = ss.daten(klein)
    return ss.bpc_auswerten(netz, va[:ss.VAL_BYTES] if teil == "val" else te)


# ================================================================== Einheiten (eine Aufgabe = ein Arm / eine Kette fuer einen Seed)
def _modell_pfad(sp, lr):
    return Path(sp["ckordner"]) / "modelle" / f"gross_{sp['ds']}_s{sp['seed']}_lr{lr:.6g}_n{sp['n']}_k{int(sp['klein'])}_p{int(sp.get('perm', False))}.pt"


def baue_gross(ds, seed):
    torch.manual_seed(seed)
    return baue_bild([512, 512]) if ds in BILD else LM(128, 4)


def trainiere_netz(sp, netz, lr, n, ckpt_tag, nt=False):
    """Trainiert `netz` fuer die Spezifikation sp; n = Epochen (Bild) bzw. Schritte (LM)."""
    key = f"{sp['ds']}_{ckpt_tag}_s{sp['seed']}_lr{lr:.6g}_n{n}_k{int(sp['klein'])}_p{int(sp.get('perm', False))}"
    ck = _ckpt_pfad(Path(sp["ckordner"]) / "ckpt", key)
    if sp["ds"] in BILD:
        return trainiere_bild(netz, sp["ds"], sp["seed"], lr, n, klein=sp["klein"], perm_labels=sp.get("perm", False), ckpt=ck, ckpt_alle=sp["ck"], tag=ckpt_tag)
    return trainiere_lm(netz, sp["seed"], lr, n, klein=sp["klein"], perm_bytes=sp.get("perm", False), ckpt=ck, ckpt_alle=sp["ck"], tag=ckpt_tag)


def hole_gross(sp, lr):
    """Trainiertes grosses fp32-Modell: aus Datei, sonst trainieren und speichern (atomar). Rueckgabe (netz, info)."""
    pfad = _modell_pfad(sp, lr)
    netz = baue_gross(sp["ds"], sp["seed"])
    if pfad.exists():
        try:
            z = torch.load(pfad, weights_only=True)
            netz.load_state_dict(z["modell"])
            return netz, z["info"]
        except Exception:
            pass
    info = trainiere_netz(sp, netz, lr, sp["n"], "gross")
    if not info["instabil"]:
        pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = pfad.with_name(pfad.name + f".tmp{os.getpid()}")
        torch.save({"modell": netz.state_dict(), "info": info}, tmp)
        os.replace(tmp, pfad)
    return netz, info


def metrik(sp, netz, teil):
    return bewerte_bild(netz, sp["ds"], sp["klein"], teil) if sp["ds"] in BILD else bewerte_lm(netz, sp["klein"], teil)


def score(sp, w) -> float:
    return w if sp["ds"] in BILD else 1 - w / 8


def form_klein(ds, shape, ziel):
    """(Modell, Beschreibung) der kleinen dichten fp16-Form mit hoechstens `ziel` Bytes der Linearschichten (groesste Breite)."""
    if ds in BILD:
        def bytes_(dims):
            ein, aus = [784] + dims, dims + [10]
            return sum(2 * (i * o + o) for i, o in zip(ein, aus))
        mk = {"flach": lambda h: [h], "zwei": lambda h: [h, h], "trichter": lambda h: [h, max(1, h // 2)]}[shape]
        h = 1
        while bytes_(mk(h + 1)) <= ziel:
            h += 1
        return baue_bild(mk(h)), {"form": shape, "dims": mk(h), "h": h}
    d, L = {"d64": (64, 4), "d48": (48, 4), "d64L2": (64, 2)}[shape]
    f = lm_f_fit(d, L, ziel)
    return LM(d, L, f), {"form": shape, "d": d, "lagen": L, "f": f, "koepfe": koepfe(d)}


def _gleich_kurz(sp):
    return {"ds": sp["ds"], "seed": sp["seed"]}


def einheit(sp: dict) -> dict:
    """Eine Aufgabe. sp: ds, art (gross|klein|nr|ref64|svd), seed, lr, val, perm, klein, n (Epochen bzw. Schritte), nt (Nachtraining), ck (Checkpoint-Abstand), ckordner,
    bits, shape (klein), var (svd), gross_lr (svd). val=True: Tuning (Score auf der Validierung); sonst Messung (Test)."""
    torch.set_num_threads(1)
    ds, art, seed, val = sp["ds"], sp["art"], sp["seed"], sp["val"]
    bild = ds in BILD
    out = {"ds": ds, "art": art, "seed": seed, "lr": sp["lr"], "bits": sp.get("bits"), "var": sp.get("var"), "shape": sp.get("shape"), "metrik": "acc" if bild else "bpc"}
    teil = "val" if val else "test"
    if art == "gross":
        netz, info = hole_gross(sp, sp["lr"])
        out.update(instabil=info["instabil"], sekunden=info["sekunden"])
        if info["instabil"]:
            out.update(fp=None, score=0.0 if bild else 0.0)
            return out
        w = metrik(sp, netz, teil)
        out["score"] = score(sp, w)
        if val:
            out["fp_val"] = w
            return out
        out["fp"] = w
        out["fp_val"] = metrik(sp, netz, "val")
        out["bytes_rest"] = bytes_rest_fp16(netz)
        out["bytes_fp16_linear"] = bytes_fp16_linear(netz)
        out["mac_lin"] = macs_linear(netz)
        bits_liste, modi = ((2,), ("had",)) if sp.get("perm") else (BITS_ALLE, MODI)
        out["q"] = {}
        pad = {784: 1024} if bild else None
        for b in bits_liste:
            out["q"][f"b{b}"] = {"ziel": ziel_bytes(ds, b)}
            for m in modi:
                qn = quantisiere(netz, b, m, seed, pad_to=pad)
                out["q"][f"b{b}"][m] = {"test": metrik(sp, qn, "test"), "val": metrik(sp, qn, "val"), **stat_q(qn)}
        return out
    if art == "svd":
        netz, info = hole_gross(sp, sp["gross_lr"])
        out.update(instabil=info["instabil"], gross_lr=sp["gross_lr"])
        if info["instabil"]:
            out.update(score=0.0)
            return out
        b = sp["bits"]
        ziel = ziel_bytes(ds, b)
        P, namen, formen = param_lr_budget(ds, ziel, netz)
        eff = eff_raenge(netz, namen)
        ra = rang_budget(formen, P, sp["var"], [eff[n] for n in namen])
        raenge = dict(zip(namen, ra))
        torch.manual_seed(seed)
        lowrank_ersetzen(netz, raenge, quelle=True)
        out["raenge"] = ra
        out["eff_rang"] = [eff[n] for n in namen]
        out["params_lr"] = int(sum(r * (o + i) for r, (o, i) in zip(ra, formen)))
        out["vor_nt"] = metrik(sp, netz, teil)
        sp2 = dict(sp)
        info2 = trainiere_netz(sp2, netz, sp["lr"], sp["nt"], f"svd_{sp['var']}_b{b}")
        out.update(instabil=info2["instabil"], sekunden=info2["sekunden"])
        w = 8.0 if (info2["instabil"] and not bild) else (0.0 if info2["instabil"] else metrik(sp, netz, teil))
        out["score"] = score(sp, w)
        out["wert"] = w
        out["bytes_fp16_linear"] = bytes_fp16_linear(netz)
        out["ziel"] = ziel
        out["mac_lin"] = macs_linear(netz)
        out["bytes_rest"] = bytes_rest_fp16(netz)
        return out
    # klein | nr | ref64: von Grund auf
    b = sp.get("bits")
    ziel = ziel_bytes(ds, b) if b else None
    torch.manual_seed(seed)
    if art == "klein":
        netz, form = form_klein(ds, sp["shape"], ziel)
        out["form"] = form
    elif art == "ref64":
        netz, form = LM(64, 4, 256), {"form": "ref64", "d": 64, "lagen": 4, "f": 256, "koepfe": koepfe(64)}
        out["form"] = form
    else:      # nr: gleicher Rang, von Grund auf
        netz = baue_gross(ds, seed)
        P, namen, formen = param_lr_budget(ds, ziel, netz)
        ra = rang_budget(formen, P, "gleich")
        lowrank_ersetzen(netz, dict(zip(namen, ra)), quelle=None)
        out["raenge"] = ra
    tag = f"{art}_b{b}_{sp.get('shape', '')}"
    info = trainiere_netz(sp, netz, sp["lr"], sp["n"], tag)
    out.update(instabil=info["instabil"], sekunden=info["sekunden"])
    w = 8.0 if (info["instabil"] and not bild) else (0.0 if info["instabil"] else metrik(sp, netz, teil))
    out["score"] = score(sp, w)
    out["wert"] = w
    out["bytes_fp16_linear"] = bytes_fp16_linear(netz)
    out["ziel"] = ziel
    out["mac_lin"] = macs_linear(netz)
    out["bytes_rest"] = bytes_rest_fp16(netz)
    out["params"] = int(sum(p.numel() for p in netz.parameters()))
    if not bild:
        d = getattr(netz, "d", 128)
        out["mac_gesamt"] = lm_macs_gesamt(d, netz.lagen, out["mac_lin"])
    return out


# ================================================================== Aufgabenlisten
def _schl(sp) -> str:
    return f"einheit|{sp['teil']}|{sp['ds']}|{sp['art']}|{sp.get('var') or ''}|{sp.get('shape') or ''}|b{sp.get('bits') or 0}|{sp['seed']}"


def _schl_lr(a) -> str:
    return f"lr|{a[0]}|{a[2]!r}|{a[1]}"


def _arbeiter(sp):
    r = einheit(sp)
    r["teil"] = sp["teil"]
    return _schl(sp), r


_ZAEHLER = {"lr": 0}


def _lr_arbeiter(a):
    schl, seed, lr, cfg = a
    _ZAEHLER["lr"] += 1
    sp = {**cfg, "seed": seed, "lr": lr, "val": True, "teil": "tuning"}
    r = einheit(sp)
    return _schl_lr(a), {"acc": r["score"], "instabil": r.get("instabil", False), "wert": r.get("wert", r.get("fp_val"))}


def _spec(P, ds, art, seed, ckordner, klein, teil="mess", **kw):
    bild = ds in BILD
    n = P["bild_ep"] if bild else P["lm_T"]
    nt = P["bild_nt_ep"] if bild else P["lm_nt_T"]
    return {"ds": ds, "art": art, "seed": seed, "teil": teil, "klein": klein, "n": n, "nt": nt, "ck": P["ck_bild"] if bild else P["ck_lm"],
            "ckordner": str(ckordner), "val": False, "lr": 0.0, **kw}


def tuning_schluessel(ds):
    """Alle Lernraten-Schluessel der Phase A (unabhaengig): gross, klein je Form und Bit, nr je Bit; LM zusaetzlich ref64."""
    formen = FORMEN_BILD if ds in BILD else FORMEN_LM
    s = [(f"{ds}|gross", {"art": "gross"})]
    s += [(f"{ds}|klein|b{b}|{f}", {"art": "klein", "bits": b, "shape": f}) for b in BITS for f in formen]
    s += [(f"{ds}|nr|b{b}", {"art": "nr", "bits": b}) for b in BITS]
    if ds == "lm":
        s += [(f"{ds}|ref64", {"art": "ref64"})]
    return s


def tuning_phase(daten_liste, P, klein, pfad, kennung, prozesse, ckordner, phase, hp_a=None):
    """Lernrate je Schluessel (Tuning-Seed, Validierung). phase 'A': gross/klein/nr/ref64; phase 'B': Nachtraining svd_gleich/svd_entropie (setzt hp_a['<ds>|gross'] voraus)."""
    cfgs = {}
    for ds in daten_liste:
        if phase == "A":
            for k, c in tuning_schluessel(ds):
                cfgs[k] = {**_spec(P, ds, c["art"], P["tuning_seed"], ckordner, klein, teil="tuning"), **{x: y for x, y in c.items() if x != "art"}, "art": c["art"]}
        else:
            for b in BITS:
                for var in ("gleich", "entropie"):
                    cfgs[f"{ds}|svd_{var}|b{b}"] = {**_spec(P, ds, "svd", P["tuning_seed"], ckordner, klein, teil="tuning"), "bits": b, "var": var, "gross_lr": hp_a[f"{ds}|gross"]["lr"]}

    def aufg_fuer(s, lr):
        return [(s, P["tuning_seed"], lr, cfgs[s])]

    def auswerten(aufg):
        lang = sorted(aufg, key=lambda a: -_kosten(a[3], tuning=True))
        res = _rechne(lang, pfad, kennung, fn=_lr_arbeiter, schluessel=_schl_lr, prozesse=prozesse)
        return [res[_schl_lr(a)] for a in aufg]

    gitter = LR_GITTER if phase == "A" else NT_GITTER
    info, _ = gittersuche_stabil(list(cfgs), aufg_fuer, auswerten, gitter=gitter, faktor=LR_FAKTOR, kollaps=KOLLAPS_SCORE)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"], "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def waehle_formen(hp_a):
    """Beste Form je (ds, bits) nach Val-Score bei der gewaehlten Lernrate (Gleichstand: frueher genannte Form); Rueckgabe {'<ds>|klein|b<b>': {'shape', 'lr'}}."""
    aus = {}
    for ds in DATENSAETZE:
        formen = FORMEN_BILD if ds in BILD else FORMEN_LM
        for b in BITS:
            def sc(f):
                e = hp_a[f"{ds}|klein|b{b}|{f}"]
                return e["gitter"][str(e["lr"])]
            best = max(formen, key=lambda f: (sc(f), -formen.index(f)))
            aus[f"{ds}|klein|b{b}"] = {"shape": best, "lr": hp_a[f"{ds}|klein|b{b}|{best}"]["lr"]}
    return aus


def _kosten(sp, tuning=False) -> float:
    """Geschaetzte Sekunden (1 Thread, Container) einer Aufgabe."""
    k = "bild" if sp["ds"] in BILD else "lm"
    if sp["art"] == "svd":
        return S_JE[(k, "nr")] * sp["nt"] + (30 if k == "bild" else 60)          # Nachtraining + SVD + Auswertung
    art = sp["art"] if (k, sp["art"]) in S_JE else "gross"
    t = S_JE[(k, art)] * sp["n"]
    if art == "gross" and not tuning:
        t += 60 if k == "bild" else 240                                         # Quantisierungs-Auswertung (3 Bit-Stufen x 3 Modi)
    return t


def aufgaben_mess(hp, formen, P, klein, ckordner):
    """Mess- und NK-Aufgaben. Stufe 1: gross (mit allen Quant-Armen), klein, nr, ref64; Stufe 2: svd (benutzt die Gross-Modelle)."""
    stufe1, stufe2, nk = [], [], []
    for ds in DATENSAETZE:
        lr_g = hp[f"{ds}|gross"]["lr"]
        for s in P["seeds"]:
            stufe1.append(_spec(P, ds, "gross", s, ckordner, klein, lr=lr_g))
            for b in BITS:
                f = formen[f"{ds}|klein|b{b}"]
                stufe1.append(_spec(P, ds, "klein", s, ckordner, klein, lr=f["lr"], bits=b, shape=f["shape"]))
                stufe1.append(_spec(P, ds, "nr", s, ckordner, klein, lr=hp[f"{ds}|nr|b{b}"]["lr"], bits=b))
                for var in ("gleich", "entropie"):
                    stufe2.append(_spec(P, ds, "svd", s, ckordner, klein, lr=hp[f"{ds}|svd_{var}|b{b}"]["lr"], bits=b, var=var, gross_lr=lr_g))
            if ds == "lm":
                stufe1.append(_spec(P, ds, "ref64", s, ckordner, klein, lr=hp["lm|ref64"]["lr"]))
        for s in P["nk_seeds"]:
            sp = _spec(P, ds, "gross", s, ckordner, klein, teil="nk", lr=lr_g, perm=True)
            sp["n"] = P["nk_ep"] if ds in BILD else P["nk_T"]
            nk.append(sp)
    stufe1 += nk
    stufe1.sort(key=lambda sp: -_kosten(sp))
    stufe2.sort(key=lambda sp: -_kosten(sp))
    return stufe1, stufe2


def kosten_schaetzung(prozesse=72, aufschlag=1.3, tuning_raten=4.5, ueberhang=30.0, P=None):
    """Kernstunden und Wandzeit (LPT je Stufe, Stufen nacheinander) auf `prozesse` Kerne; aufschlag = Parallel-Aufschlag bei vollem Knoten (I-045 beobachtete 1,2-1,5)."""
    P = P or PROFILE["standard"]
    hp = {f"{ds}|{x}": {"lr": 1e-3} for ds in DATENSAETZE for x in ("gross", "nr|b3", "nr|b2", "svd_gleich|b3", "svd_gleich|b2", "svd_entropie|b3", "svd_entropie|b2", "ref64")}
    formen = {f"{ds}|klein|b{b}": {"shape": (FORMEN_BILD if ds in BILD else FORMEN_LM)[0], "lr": 1e-3} for ds in DATENSAETZE for b in BITS}
    s1, s2 = aufgaben_mess(hp, formen, P, False, "x")

    def lpt(liste):
        last = [0.0] * prozesse
        heapq.heapify(last)
        for t in sorted(liste, reverse=True):
            heapq.heappush(last, heapq.heappop(last) + t)
        return max(last)

    d1 = [_kosten(sp) * aufschlag + ueberhang for sp in s1]
    d2 = [_kosten(sp) * aufschlag + ueberhang for sp in s2]
    # Tuning: Phase A (parallel, tuning_raten Gitterpunkte je Schluessel), Phase B
    ta = [_kosten({**_spec(P, ds, c["art"], 1, "x", False), **c}, tuning=True) * aufschlag + ueberhang for ds in DATENSAETZE for _, c in tuning_schluessel(ds)]
    tb = [_kosten({**_spec(P, ds, "svd", 1, "x", False)}, tuning=True) * aufschlag + ueberhang for ds in DATENSAETZE for _ in range(2 * len(BITS))]
    kern_tuning = (sum(ta) + sum(tb)) * tuning_raten
    wand_tuning = 2 * max(ta) + 2 * max(tb)
    wand = lpt(d1) + lpt(d2)
    return {"aufgaben_mess": len(s1) + len(s2), "aufgaben_stufe1": len(s1), "aufgaben_stufe2": len(s2), "kernstunden_mess": (sum(d1) + sum(d2)) / 3600,
            "kernstunden_tuning": kern_tuning / 3600, "kernstunden_gesamt": (sum(d1) + sum(d2) + kern_tuning) / 3600, "laengste_aufgabe_h": max(d1 + d2) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand / 3600, "wand_gesamt_h": (wand_tuning + wand) / 3600, "prozesse": prozesse, "aufschlag": aufschlag}


# ================================================================== Auswertung
def _wert(ds, z):
    return 100.0 * z if ds in BILD else z


def _vs(ds, a, b):
    """Vorsprung von a gegenueber b in pp bzw. bpc: positiv = a besser."""
    return (a - b) if ds in BILD else (b - a)


def _t_p(d, seite):
    from abteilungen.struktur_gross import _t_p as tp
    return tp(d, seite)


def _stat_rand(d, rand):
    """Gepaart, positiv = Hypothese; einseitig gegen Rand `rand`: p_gr = P(mittel(d) > rand), p_kl = P(mittel(d) < rand)."""
    d = np.asarray(d, float)
    s = ss._stat(d, versatz=-rand)
    return s


def _tost(d, rand):
    d = np.asarray(d, float)
    s = ss._stat(d)
    if len(d) < 2:
        return s
    s["p_tost"] = max(_t_p(d + rand, "gr"), _t_p(d - rand, "kl"))
    s["rand"] = rand
    return s


def zellen_aus_zeilen(zeilen):
    """zeilen (Mess) -> {zelle: {seed: wert}}, {zelle: {seed: instabil}}, Meta je Zelle (Bytes, MACs ...). Zellname '<ds>|<arm>|b<bits>' (bits 0 = ohne)."""
    w, inst, meta = {}, {}, {}
    for z in zeilen:
        if z.get("teil") != "mess":
            continue
        ds, s = z["ds"], z["seed"]

        def setze(arm, b, wert, instabil, **m):
            k = f"{ds}|{arm}|b{b}"
            w.setdefault(k, {})[s] = wert
            inst.setdefault(k, {})[s] = bool(instabil)
            mm = meta.setdefault(k, {})
            for a, v in m.items():
                mm.setdefault(a, []).append(v)

        if z["art"] == "gross":
            if z.get("fp") is not None:
                setze("fp", 0, _wert(ds, z["fp"]), z["instabil"], mac_lin=z["mac_lin"], bytes_fp16=z["bytes_fp16_linear"], bytes_rest=z["bytes_rest"])
                for bk, qd in z["q"].items():
                    b = int(bk[1:])
                    for m in MODI:
                        if m not in qd:
                            continue
                        q = qd[m]
                        setze(m, b, _wert(ds, q["test"]), z["instabil"], bytes=q["bytes"], mac=q["mac"], add=q["add"], mac_orth=q["mac_orth"], ziel=qd["ziel"], val=_wert(ds, q["val"]))
            else:
                setze("fp", 0, float("nan"), True)
        elif z["art"] in ("klein", "nr", "svd", "ref64"):
            arm = f"svd_{z['var']}" if z["art"] == "svd" else z["art"]
            b = z.get("bits") or 0
            kw = {"bytes_fp16": z.get("bytes_fp16_linear"), "bytes_rest": z.get("bytes_rest"), "mac_lin": z.get("mac_lin"), "ziel": z.get("ziel")}
            if z["art"] == "svd":
                kw.update(params_lr=z["params_lr"], vor_nt=_wert(ds, z["vor_nt"]) if z.get("vor_nt") is not None else None)
                kw["raenge"] = z["raenge"]
                kw["eff_rang"] = z["eff_rang"]
            if z["art"] == "klein":
                kw["form"] = z["form"]["form"]
            if z["art"] == "nr":
                kw["rang"] = z["raenge"][0]
            if "mac_gesamt" in z:
                kw["mac_gesamt"] = z["mac_gesamt"]
            setze(arm, b, _wert(ds, z["wert"]), z["instabil"], **{a: v for a, v in kw.items() if v is not None})
    return w, inst, meta


def auswertung(zeilen: list, params: dict, unigramm: float, seeds=SEEDS) -> dict:
    w, inst, meta = zellen_aus_zeilen(zeilen)
    r = {"zellen": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {}, "nk": {}, "negativ": {}, "diff": {}, "p_holm": {}, "beschreibend": {}, "parameter": params,
         "unigramm_bpc": unigramm, "meta": {}}
    trainiert = ("fp", "klein", "nr", "svd_gleich", "svd_entropie", "ref64")
    for k, v in w.items():
        ds, arm, b = k.split("|")
        s = sorted(v)
        a = np.array([v[x] for x in s], float)
        r["zellen"][k] = zusammenfassung(a[np.isfinite(a)]) if np.isfinite(a).any() else {}
        if arm in trainiert:
            med = float(np.nanmedian(a))
            sl = KOLLAPS_BILD_PP if ds in BILD else KOLLAPS_LM
            r["kollaps"][k] = [x for x in s if not (_vs(ds, med, v[x]) <= sl)]            # schlechter als Median um > Schwelle
            div = [x for x in s if inst[k][x] or (v[x] < 50.0 if ds in BILD else not (v[x] < unigramm))]
            r["divergiert"][k] = div
            r["divergenzrate"][k] = len(div) / len(s)
        r["meta"][k] = {a_: (float(np.mean(x)) if isinstance(x[0], (int, float)) else x[0]) for a_, x in meta.get(k, {}).items() if x and a_ not in ("raenge", "eff_rang")}
        for a_ in ("raenge", "eff_rang"):
            if a_ in meta.get(k, {}):
                r["meta"][k][a_] = list(np.mean(np.array(meta[k][a_], float), axis=0))
    for ds in DATENSAETZE:
        nk = [z for z in zeilen if z.get("teil") == "nk" and z["ds"] == ds]
        if nk:
            fp = [_wert(ds, z["fp"]) for z in nk if z.get("fp") is not None]
            ha = [_wert(ds, z["q"]["b2"]["had"]["test"]) for z in nk if z.get("fp") is not None]
            r["negativ"][ds] = {"fp": float(np.mean(fp)), "had2": float(np.mean(ha)), "n": len(nk)}
            if ds in BILD:
                ok = max(r["negativ"][ds]["fp"], r["negativ"][ds]["had2"]) <= 100 * NK_ACC_MAX
            else:
                ok = min(r["negativ"][ds]["fp"], r["negativ"][ds]["had2"]) >= NK_FAKTOR * unigramm
            r["nk"][ds] = "BESTANDEN" if ok else "DURCHGEFALLEN"

    def paar(ds, a, b, ba, bb):
        x, y = w.get(f"{ds}|{a}|b{ba}"), w.get(f"{ds}|{b}|b{bb}")
        if not x or not y:
            return {}
        return {s: _vs(ds, x[s], y[s]) for s in sorted(set(x) & set(y))}

    for ds in DATENSAETZE:
        v = paar(ds, "had", "klein", 2, 2)
        if v:
            r["diff"][f"H1|{ds}"] = _stat_rand(list(v.values()), RAND[ds])
        v = paar(ds, "had", "klein", 3, 3)
        if v:
            r["diff"][f"H1b|{ds}"] = _stat_rand(list(v.values()), RAND[ds])
        for b in BITS:
            v = paar(ds, "had", "orth", b, b)
            if v:
                r["diff"][f"H2|{ds}|b{b}"] = _tost(list(v.values()), RAND[ds])
            v = paar(ds, "svd_entropie", "svd_gleich", b, b)
            if v:
                r["diff"][f"H3|{ds}|b{b}"] = _stat_rand(list(v.values()), 0.0)
    for fam, pk in (("H1", "p_gr"), ("H2", "p_tost"), ("H3", "p_gr")):
        prim = {k: r["diff"][k][pk] for k in r["diff"] if k.startswith(fam + "|") and pk in r["diff"][k]}
        if prim:
            r["p_holm"].update(holm(prim))
    besch = r["beschreibend"]
    for ds in DATENSAETZE:
        for b in BITS_ALLE:
            for a_, c_ in (("had", "roh"), ("orth", "roh"), ("had", "orth"), ("nr", "klein"), ("svd_entropie", "klein"), ("svd_gleich", "klein"), ("had", "svd_entropie"), ("had", "nr")):
                if b == 4 and (a_ in ("nr", "svd_entropie", "svd_gleich") or c_ in ("klein", "nr", "svd_entropie", "svd_gleich")):
                    continue
                v = paar(ds, a_, c_, b, b)
                if v and len(v) >= 2:
                    besch.setdefault(f"Vorsprung_{a_}_gegen_{c_}", {})[f"{ds}|b{b}"] = ss._stat(np.array(list(v.values())))
    v = paar("lm", "had", "ref64", 4, 0)
    if v and len(v) >= 2:
        besch["Vorsprung_had4_gegen_ref64_lm"] = ss._stat(np.array(list(v.values())))
    for ds in DATENSAETZE:
        for b in BITS:
            k1, k2 = f"{ds}|svd_entropie|b{b}", f"{ds}|svd_gleich|b{b}"
            if k1 in r["meta"] and k2 in r["meta"]:
                besch.setdefault("Parameter_svd_entropie_zu_gleich", {})[f"{ds}|b{b}"] = r["meta"][k1].get("params_lr", float("nan")) / r["meta"][k2].get("params_lr", float("nan"))
    return r


def _vorab(r, h, ds, zellen, pflicht_seeds=2):
    """Gemeinsame Vorbedingungen: Zellen vorhanden, NK, Divergenz, Kollaps. None = alles in Ordnung."""
    if h not in r["diff"] or "mittel" not in r["diff"][h] or h not in r["p_holm"] and not h.startswith("H1b"):
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if r["nk"].get(ds) != "BESTANDEN":
        return f"HU (Negativkontrolle {ds} nicht bestanden)"
    dv = {z: r["divergiert"][z] for z in zellen if r["divergiert"].get(z)}
    if dv:
        return f"HU (Divergenz in {dv})"
    k = {z: r["kollaps"][z] for z in zellen if r["kollaps"].get(z)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def _zahlen(d, einheit, holm_p=None):
    s = f"d {d['mittel']:+.3f} {einheit}, 90-%-KI [{d['ki'][0]:+.3f}; {d['ki'][1]:+.3f}], n {d['n']}"
    if holm_p is not None:
        s += f", p_Holm {holm_p:.3g}"
    return s.replace(".", ",")


def urteil(r: dict) -> dict:
    u = {}
    for ds in DATENSAETZE:
        n = r["negativ"].get(ds, {})
        if n:
            schwelle = "Schwelle 14 %" if ds in BILD else "Unigramm %.3f, Schwelle %sx" % (r["unigramm_bpc"], NK_FAKTOR)
            u[f"NK|{ds}"] = ("%s (fp %.2f, had2 %.2f; %s)" % (r["nk"].get(ds, "FEHLT"), n["fp"], n["had2"], schwelle)).replace(".", ",")
        else:
            u[f"NK|{ds}"] = "FEHLT"
    for ds in DATENSAETZE:
        e = "pp" if ds in BILD else "bpc"
        # H1
        h = f"H1|{ds}"
        v = _vorab(r, h, ds, [f"{ds}|fp|b0", f"{ds}|klein|b2", f"{ds}|had|b2"])
        if v:
            u[h] = v
        else:
            d = r["diff"][h]
            z = _zahlen(d, e, r["p_holm"][h])
            if r["p_holm"][h] < ALPHA:
                u[h] = f"HADAMARD_BESSER (had2 − klein > {RAND[ds]} {e} bei gleichen Bytes: {z})"
            elif d["p_kl"] < ALPHA / 2:
                u[h] = f"H0 (Vorsprung sicher <= {RAND[ds]} {e}: {z})"
            else:
                u[h] = f"HU ({z})"
        # H2, H3
        for b in BITS:
            h = f"H2|{ds}|b{b}"
            v = _vorab(r, h, ds, [f"{ds}|fp|b0", f"{ds}|had|b{b}", f"{ds}|orth|b{b}"])
            if v:
                u[h] = v
            else:
                d = r["diff"][h]
                z = _zahlen(d, e, r["p_holm"][h])
                if r["p_holm"][h] < ALPHA:
                    u[h] = f"AEQUIVALENT (had − orth innerhalb ±{RAND[ds]} {e}: {z})"
                elif d["p_zwei"] < ALPHA:
                    u[h] = f"UNTERSCHIEDLICH (zweiseitig p {d['p_zwei']:.3g}: {z})".replace(".", ",")
                else:
                    u[h] = f"HU ({z})"
            h = f"H3|{ds}|b{b}"
            pr = r["beschreibend"].get("Parameter_svd_entropie_zu_gleich", {}).get(f"{ds}|b{b}")
            v = _vorab(r, h, ds, [f"{ds}|fp|b0", f"{ds}|svd_entropie|b{b}", f"{ds}|svd_gleich|b{b}"])
            if v:
                u[h] = v
            elif pr is None or not (abs(pr - 1) <= PARAM_TOLERANZ):
                u[h] = f"HU (Parameter nicht gleich: Verhaeltnis {pr})"
            else:
                d = r["diff"][h]
                z = _zahlen(d, e, r["p_holm"][h])
                if r["p_holm"][h] < ALPHA:
                    u[h] = f"ENTROPIE_BESSER (svd_entropie − svd_gleich: {z})"
                elif d["p_kl"] < ALPHA / 2:
                    u[h] = f"H0 (gleichmaessig sicher besser: {z})"
                else:
                    u[h] = f"HU ({z})"
    teile = []
    for ds in DATENSAETZE:
        ze = ", ".join(f"{k.split('|', 1)[1]} {r['zellen'][k]['mittel']:.3f}".replace(".", ",") for k in sorted(r["zellen"]) if k.startswith(ds + "|") and r["zellen"][k])
        teile.append(f"{ds}: {ze}")
    u["B"] = "beschreibend: " + "; ".join(teile) + "; Rest in auswertung"
    return u


# ================================================================== Lauf
def zeiten_messen(n=15, warm=3):
    """Sekunden je Schritt (LM, 1 Thread, Batch 8x128) bzw. je Epoche (Bild, aus 30 Schritten hochgerechnet) fuer die Kostenkonstanten S_JE."""
    torch.set_num_threads(1)
    aus = {}
    tr, _, _ = ss.daten(True)
    offs = torch.arange(ss.KONTEXT)
    g = torch.Generator().manual_seed(1)
    modelle = {"lm_gross": LM(128, 4), "lm_klein_d64": LM(64, 4, 80), "lm_ref64": LM(64, 4, 256)}
    for nm, net in modelle.items():
        opt = torch.optim.Adam(net.parameters(), 1e-4)
        ts = []
        for k in range(warm + n):
            i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
            t0 = time.perf_counter()
            l = F.cross_entropy(net(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if k >= warm:
                ts.append(time.perf_counter() - t0)
        aus[nm] = float(np.median(ts))
    xtr = torch.rand(128 * 40, 784)
    ytr = torch.randint(0, 10, (128 * 40,))
    for nm, net in {"bild_gross": baue_bild([512, 512]), "bild_klein": baue_bild([110, 110])}.items():
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        ts = []
        for k in range(warm + 30):
            b = slice(k * 128 % (128 * 39), k * 128 % (128 * 39) + 128)
            t0 = time.perf_counter()
            l = F.cross_entropy(net(xtr[b]), ytr[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            if k >= warm:
                ts.append(time.perf_counter() - t0)
        aus[nm + "_epoche"] = float(np.median(ts)) * math.ceil(55000 / 128)
    return aus


def lauf(schnell=False, prozesse=None, zwischen=None, profil=None, **_):
    profil = profil or os.environ.get("I048_PROFIL", "standard")
    P = dict(PROFILE[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte; NICHT ueber `direktor.py lauf --schnell` (Vorregistrierung)
        P.update({"bild_ep": 1, "bild_nt_ep": 1, "lm_T": 24, "lm_nt_T": 8, "nk_ep": 1, "nk_T": 6, "ck_bild": 5, "ck_lm": 8, "seeds": (TUNING_SEED,), "nk_seeds": (TUNING_SEED,)})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT or set(P["nk_seeds"]) & _BELEGT or TUNING_SEED in P["seeds"] or TUNING_SEED in _BELEGT:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": ss._json_sicher(P), "lr_gitter": list(LR_GITTER), "nt_gitter": list(NT_GITTER),
               "tuning_seed": P["tuning_seed"], "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning_a" not in zw:
        ta = tuning_phase(DATENSAETZE, P, klein, pfad, kennung, prozesse, ckordner, "A")
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning_a"] = ta
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning A fertig: {({k: v['lr'] for k, v in ta.items()})}", flush=True)
    formen = waehle_formen(zw["tuning_a"])
    if "tuning_b" not in zw:
        hp_a = {**zw["tuning_a"], **{f"{ds}|gross": zw["tuning_a"][f"{ds}|gross"] for ds in DATENSAETZE}}
        tb = tuning_phase(DATENSAETZE, P, klein, pfad, kennung, prozesse, ckordner, "B", hp_a=hp_a)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning_b"] = tb
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning B fertig: {({k: v['lr'] for k, v in tb.items()})}", flush=True)
    hp = {**zw["tuning_a"], **zw["tuning_b"]}
    s1, s2 = aufgaben_mess(hp, formen, P, klein, ckordner)
    _rechne(s1, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    alle = _rechne(s2, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    tr, _, te = ss.daten(klein)
    params = {}
    for ds in DATENSAETZE:
        for b in BITS:
            params[f"{ds}|b{b}"] = {"ziel_bytes": ziel_bytes(ds, b), "form_klein": formen[f"{ds}|klein|b{b}"]}
        params[f"{ds}|bits4"] = {"ziel_bytes": ziel_bytes(ds, 4)}
    e = {"abteilung": "Kompression", "modul": "kompression_bytes", "karte": KARTE, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": P["tuning_seed"], "tuning_a": zw["tuning_a"], "tuning_b": zw["tuning_b"], "formen": formen,
         "einstellungen": {"bild_ep": P["bild_ep"], "bild_nt_ep": P["bild_nt_ep"], "lm_T": P["lm_T"], "lm_nt_T": P["lm_nt_T"], "nk_ep": P["nk_ep"], "nk_T": P["nk_T"],
                           "ckpt": [P["ck_bild"], P["ck_lm"]], "lr_gitter": list(LR_GITTER), "nt_gitter": list(NT_GITTER), "rand": RAND, "daten_bytes_lm": [len(x) for x in ss.daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, params, ss.unigramm_entropie(tr, te), seeds=P["seeds"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ================================================================== Vortest
def _synth(rng, vor_h1=1.0, d_orth=0.0, vor_h3=0.5, sd=0.15, n=10, koll=None, nk_ok=True, div=None, fehlt=None, param_abw=0.0):
    """Synthetische Zeilen fuer alle drei Datensaetze. Rechnung in pp-Aequivalenten: Bild Acc = pp/100, LM bpc = 3 − (pp − 97)·0,1 (also 0,2 pp ≙ 0,02 bpc).
    vor_h1 = Vorsprung had2 vor klein, d_orth = Vorsprung had vor orth, vor_h3 = Vorsprung svd_entropie vor svd_gleich (alle in pp-Aequivalenten)."""
    zl = []
    for ds in DATENSAETZE:
        bild = ds in BILD
        wert = (lambda pp: pp / 100.0) if bild else (lambda pp: 3.0 - (pp - 97.0) * 0.1)
        for i in range(n):
            seed = 2800 + i
            fp = 97.0 + rng.normal(0, 0.1)
            z = {"teil": "mess", "ds": ds, "seed": seed, "instabil": False, "lr": 1e-3}
            q = {}
            for b in BITS_ALLE:
                q[f"b{b}"] = {"ziel": 1000}
                h = fp - (2.0 if b == 2 else 0.5) + rng.normal(0, sd)
                for m, v in (("roh", h - 1.0 + rng.normal(0, sd)), ("had", h), ("orth", h - d_orth + rng.normal(0, sd))):
                    if fehlt == (ds, m):
                        continue
                    q[f"b{b}"][m] = {"test": wert(v), "val": wert(v), "bytes": 900.0, "mac": 10, "add": 5, "mac_orth": 0}
            zl.append({**z, "art": "gross", "fp": wert(fp), "fp_val": wert(fp), "q": q, "bytes_rest": 100, "bytes_fp16_linear": 2000, "mac_lin": 10,
                       "instabil": bool(div == (ds, "fp") and i == 1)})
            for b in BITS:
                hv = fp - (2.0 if b == 2 else 0.5)
                klw = hv - vor_h1 + rng.normal(0, sd)
                kz = klw - 10.0 if (koll == (ds, "klein") and i == 0) else klw
                zl.append({**z, "art": "klein", "bits": b, "wert": wert(kz), "form": {"form": "flach"}, "bytes_fp16_linear": 900, "ziel": 1000, "mac_lin": 5, "bytes_rest": 50})
                zl.append({**z, "art": "nr", "bits": b, "wert": wert(klw - 1.0 + rng.normal(0, sd)), "raenge": [12, 12], "bytes_fp16_linear": 900, "ziel": 1000, "mac_lin": 5, "bytes_rest": 50})
                g_ = klw + rng.normal(0, sd)
                for var, add in (("gleich", 0.0), ("entropie", vor_h3)):
                    pa = 5000 * (1 + (param_abw if var == "entropie" else 0.0))
                    zl.append({**z, "art": "svd", "bits": b, "var": var, "wert": wert(g_ + add + rng.normal(0, sd)), "raenge": [20, 10], "eff_rang": [30.0, 20.0], "params_lr": pa,
                               "vor_nt": wert(g_ - 5), "bytes_fp16_linear": 900, "ziel": 1000, "mac_lin": 5, "bytes_rest": 50})
            if ds == "lm":
                zl.append({**z, "art": "ref64", "bits": None, "wert": wert(fp - 0.3), "form": {"form": "ref64"}, "bytes_fp16_linear": 900, "ziel": None, "mac_lin": 5, "bytes_rest": 50})
        for i in range(2):
            nkw = (0.10 if nk_ok else 0.6) if bild else (7.5 if nk_ok else 3.0)
            zl.append({"teil": "nk", "ds": ds, "seed": 2810 + i, "art": "gross", "fp": nkw, "instabil": False,
                       "q": {"b2": {"ziel": 1, "had": {"test": nkw, "val": nkw, "bytes": 1.0, "mac": 1, "add": 1, "mac_orth": 0}}}})
    return zl


def _u(rng, **kw):
    z = _synth(rng, **kw)
    return urteil(auswertung(z, {}, 5.0))


def _fn_test(sp):
    return _schl(sp), {"v": sp["seed"] * 2}


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-048): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    # --- Hadamard / Orthogonale
    H = rot_matrix(64, 3)
    pruefe("Hadamard-Rotation orthogonal", torch.allclose(H @ H.T, torch.eye(64), atol=1e-5))
    O = orth_matrix(64, 3)
    pruefe("Zufalls-Orthogonale orthogonal", torch.allclose(O @ O.T, torch.eye(64), atol=1e-5))
    pruefe("Hadamard und Orthogonale verschieden", not torch.allclose(H, O))
    # --- Quantisierer
    W = torch.randn(16, 64)
    for b in (4, 3, 2):
        q = quant_rows(W, b)
        pruefe(f"{b} Bit: hoechstens {2 ** b - 1} Stufen je Zeile", all(len(torch.unique(q[i])) <= 2 ** b - 1 for i in range(16)))
    pruefe("Fehler sinkt mit Bitzahl", float(((quant_rows(W, 4) - W) ** 2).mean()) < float(((quant_rows(W, 3) - W) ** 2).mean()) < float(((quant_rows(W, 2) - W) ** 2).mean()))
    lin = nn.Linear(64, 16)
    x = torch.randn(32, 64)
    with torch.no_grad():
        y = lin(x)
        for modus in ("roh", "had", "orth"):
            rel8 = float(((QLinear(lin, 8, modus, 7)(x) - y) ** 2).mean() / (y ** 2).mean())
            rel2 = float(((QLinear(lin, 2, modus, 7)(x) - y) ** 2).mean() / (y ** 2).mean())
            pruefe(f"QLinear {modus}: 8 Bit nahe fp (relativer Fehler {rel8:.1e}), 2 Bit gröber aber beschränkt ({rel2:.2f})", rel8 < 1e-3 and rel8 < rel2 < 0.6)
        pruefe("QLinear: roh ohne Q, had/orth mit Q", QLinear(lin, 2, "roh", 7).Q is None and QLinear(lin, 2, "had", 7).Q is not None and QLinear(lin, 2, "orth", 7).Q is not None)
        pruefe("QLinear had: Padding 784 -> 1024 (nur had/orth), roh ohne", QLinear(nn.Linear(784, 4), 2, "had", 1, pad_to=1024).shape == (4, 1024) and QLinear(nn.Linear(784, 4), 2, "roh", 1, pad_to=1024).shape == (4, 784))
    # --- Bytes: Formel == Zaehlung des Moduls; Ziel-Budgets
    gb = baue_bild([512, 512])
    for b in (4, 3, 2):
        qn = quantisiere(gb, b, "had", 1, pad_to={784: 1024})
        pruefe(f"Bytes Bild {b} Bit: Formel == QLinear-Summe", abs(stat_q(qn)["bytes"] - bytes_q_gross("mnist", b, True)) < 1e-6)
        pruefe(f"Ziel Bild {b} Bit >= roh-Bytes (roh ohne Padding)", stat_q(quantisiere(gb, b, "roh", 1))["bytes"] <= ziel_bytes("mnist", b))
        pruefe(f"orth hat dieselben Bytes wie had", abs(stat_q(quantisiere(gb, b, "orth", 1, pad_to={784: 1024}))["bytes"] - stat_q(qn)["bytes"]) < 1e-6)
    gl = LM(128, 4)
    for b in (4, 3, 2):
        qn = quantisiere(gl, b, "had", 1)
        pruefe(f"Bytes LM {b} Bit: Formel == QLinear-Summe", abs(stat_q(qn)["bytes"] - bytes_q_gross("lm", b, True)) < 1e-6)
    pruefe("LM 2 Bit: 215 040 + 448 Byte (Erkundung w4)", abs(bytes_q_gross("lm", 2, True) - (215040 + 448)) < 1)
    pruefe("MNIST/LM-Zusatzoperationen Hadamard n·log2 n: LM", stat_q(quantisiere(gl, 2, "had", 1))["add"] == 4 * (128 * 7 + 128 * 7 + 128 * 7 + 512 * 9) and stat_q(quantisiere(gl, 2, "orth", 1))["mac_orth"] > 0)
    # --- kleine dichte Modelle
    for ds, formen in (("mnist", FORMEN_BILD), ("lm", FORMEN_LM)):
        for b in BITS:
            ziel = ziel_bytes(ds, b)
            for f_ in formen:
                net, form = form_klein(ds, f_, ziel)
                by = bytes_fp16_linear(net)
                pruefe(f"klein {ds} {b} Bit {f_}: Bytes {by} <= Ziel {ziel}", by <= ziel)
                nxt = None
                if ds == "lm":
                    d, L = form["d"], form["lagen"]
                    nxt = bytes_fp16_linear(LM(d, L, form["f"] + 8))
                    pruefe(f"klein lm {b} Bit {f_}: Formel f stimmt, groesste Breite", nxt > ziel and by > 0.9 * ziel and form["f"] >= form["d"] // 2)
                else:
                    dims = form["dims"]
                    h2 = form["h"] + 1
                    bigger = {"flach": [h2], "zwei": [h2, h2], "trichter": [h2, max(1, h2 // 2)]}[f_]
                    pruefe(f"klein bild {b} Bit {f_}: groesste Breite", bytes_fp16_linear(baue_bild(bigger)) > ziel and by > 0.95 * ziel)
    pruefe("LM Koepfe: d=128 -> 2, 64 -> 2, 48 -> 3", (koepfe(128), koepfe(64), koepfe(48)) == (2, 2, 3))
    # --- Rangverteilung
    formen = [(512, 784), (512, 512)]
    P = 50000
    rg = rang_budget(formen, P, "gleich")
    pruefe("Rang gleich: Σ r(o+i) <= P und knapp unter P", sum(r * (o + i) for r, (o, i) in zip(rg, formen)) <= P and sum(r * (o + i) for r, (o, i) in zip(rg, formen)) > P - 1296)
    re = rang_budget(formen, P, "entropie", [60.0, 30.0])
    pruefe("Rang entropie: Σ <= P, knapp unter P, mehr Rang fuer hohe Entropie", sum(r * (o + i) for r, (o, i) in zip(re, formen)) <= P and re[0] * 1296 > re[1] * 1024 * 1.1 and sum(r * (o + i) for r, (o, i) in zip(re, formen)) > P - 1296)
    pruefe("Rang gleich: Raenge fast gleich", max(rg) - min(rg) <= 1)
    pruefe("Rang-Obergrenze min(o, i)", max(rang_budget([(10, 100), (100, 100)], 10 ** 6, "entropie", [3.0, 90.0])) <= 100 and rang_budget([(10, 100), (100, 100)], 10 ** 6, "entropie", [3.0, 90.0])[0] == 10)
    for ds, b in (("mnist", 2), ("mnist", 3), ("lm", 2), ("lm", 3)):
        net = baue_gross(ds, 0)
        Pb, namen, fo = param_lr_budget(ds, ziel_bytes(ds, b), net)
        rr = rang_budget(fo, Pb, "gleich")
        n2 = baue_gross(ds, 0)
        lowrank_ersetzen(n2, dict(zip(namen, rr)))
        pruefe(f"Niedrigrang {ds} {b} Bit: fp16-Bytes <= Ziel ({bytes_fp16_linear(n2)} <= {ziel_bytes(ds, b)}) und > Ziel − 4 KB", bytes_fp16_linear(n2) <= ziel_bytes(ds, b) and bytes_fp16_linear(n2) > ziel_bytes(ds, b) - 4096)
        eff = eff_raenge(net, namen)
        re_ = rang_budget(fo, Pb, "entropie", [eff[n] for n in namen])
        n3 = baue_gross(ds, 0)
        lowrank_ersetzen(n3, dict(zip(namen, re_)))
        pg, pe = sum(r * (o + i) for r, (o, i) in zip(rr, fo)), sum(r * (o + i) for r, (o, i) in zip(re_, fo))
        pruefe(f"Parameter svd_entropie ≈ svd_gleich {ds} {b} Bit (Verhaeltnis {pe / pg:.4f})", abs(pe / pg - 1) <= PARAM_TOLERANZ and bytes_fp16_linear(n3) <= ziel_bytes(ds, b))
    # --- SVD: voller Rang reproduziert das Modell; effektive Raenge
    net = baue_bild([32, 32])
    x = torch.randn(5, 784)
    ref = net(x)
    n2 = copy.deepcopy(net)
    lowrank_ersetzen(n2, {"ls.0": 32, "ls.1": 32}, quelle=True)
    pruefe("SVD mit vollem Rang == Original", torch.allclose(n2(x), ref, atol=1e-4))
    n3 = copy.deepcopy(net)
    lowrank_ersetzen(n3, {"ls.0": 8, "ls.1": 8}, quelle=True)
    pruefe("SVD Rang 8: Form, Fehler > 0", n3(x).shape == ref.shape and float((n3(x) - ref).abs().max()) > 0)
    Wlow = torch.randn(40, 4) @ torch.randn(4, 30)
    ln = nn.Linear(30, 40)
    ln.weight.data = Wlow
    m_ = nn.Sequential(ln)
    pruefe("Effektiver Rang einer Rang-4-Matrix <= 4", eff_raenge(m_, ["0"])["0"] <= 4.0 + 1e-6)
    li = nn.Linear(10, 10)
    li.weight.data = torch.eye(10)
    pruefe("Effektiver Rang I10 = 10", abs(eff_raenge(nn.Sequential(li), ["0"])["0"] - 10) < 1e-6)
    # --- LM: Form, Kausalitaet, Trainierbarkeit
    m = LM(64, 2, 80)
    m.eval()
    xi = torch.randint(0, 256, (2, ss.KONTEXT))
    x2 = xi.clone()
    x2[:, 60] = (x2[:, 60] + 1) % 256
    pruefe("LM Form/Kausalitaet", m(xi).shape == (2, ss.KONTEXT, 256) and torch.allclose(m(xi)[:, :60], m(x2)[:, :60], atol=1e-5) and not torch.allclose(m(xi)[:, 60:], m(x2)[:, 60:]))
    unit = torch.arange(32).repeat(8)[None].repeat(4, 1)[:, :65]
    for nm, mm in (("LM", LM(64, 2, 80)), ("LM Niedrigrang", lowrank_ersetzen(LM(64, 2, 80), {n: 8 for n in linear_namen(LM(64, 2, 80))})), ("Bild", baue_bild([32, 32]))):
        opt = torch.optim.Adam(mm.parameters(), lr=1e-2 if nm == "Bild" else 2e-3)
        l0 = None
        for _ in range(40 if nm == "Bild" else 25):
            opt.zero_grad()
            if nm == "Bild":
                l = F.cross_entropy(mm(torch.rand(32, 784) * 0 + torch.eye(32, 784)), torch.arange(32) % 10)
            else:
                l = F.cross_entropy(mm(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {nm}", float(l.detach()) < 0.7 * l0)
    pruefe("Linearschichten LM: 16 (4 Lagen x 4), Bild 3", len(linear_namen(LM(128, 4))) == 16 and len(linear_namen(baue_bild([512, 512]))) == 3)
    pruefe("MACs Linear LM d=128: 786 432", macs_linear(LM(128, 4)) == 786432)
    return bool(ok)


def _vortest_lauf() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-048): {name}", flush=True)
            ok = False

    pruefe("Seeds frisch (10 Mess, 2 NK, Tuning 2890)", len(SEEDS) == 10 and len(NK_SEEDS) == 2 and not ((set(SEEDS) | set(NK_SEEDS)) & _BELEGT)
           and TUNING_SEED not in _BELEGT and TUNING_SEED not in SEEDS and not (set(SEEDS) & set(NK_SEEDS)) and SEEDS == tuple(range(2800, 2810)))
    P = PROFILE["standard"]
    hp = {f"{ds}|{x}": {"lr": 1e-3} for ds in DATENSAETZE for x in ("gross", "nr|b3", "nr|b2", "svd_gleich|b3", "svd_gleich|b2", "svd_entropie|b3", "svd_entropie|b2", "ref64")}
    formen = {f"{ds}|klein|b{b}": {"shape": (FORMEN_BILD if ds in BILD else FORMEN_LM)[0], "lr": 1e-3} for ds in DATENSAETZE for b in BITS}
    s1, s2 = aufgaben_mess(hp, formen, P, False, "x")
    schl = [_schl(sp) for sp in s1 + s2]
    # je Datensatz und Seed: gross 1 + (klein + nr) x 2 Bits = 4 + svd 4 -> 9; LM +ref64 -> 10; 10 Seeds; NK 2 x 3
    pruefe(f"Aufgabenzahl 3x10x9 + 10 (ref64) + 6 NK = 286, Schluessel eindeutig (ist {len(s1) + len(s2)})", len(s1) + len(s2) == 286 and len(set(schl)) == 286
           and sum(sp["teil"] == "nk" for sp in s1) == 6 and len(s2) == 120)
    pruefe("Stufe 1 enthaelt alle Gross-Modelle vor Stufe 2 (svd)", all(sp["art"] != "svd" for sp in s1) and all(sp["art"] == "svd" for sp in s2))
    pruefe("Lange zuerst", _kosten(s1[0]) >= _kosten(s1[-1]) and _kosten(s1[0]) == max(_kosten(sp) for sp in s1))
    k = kosten_schaetzung()
    pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (Aufschlag 1,3), laengste Aufgabe < 24 h", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    pruefe("Laufzeitplan auch mit Aufschlag 1,5 < 24 h", kosten_schaetzung(aufschlag=1.5)["wand_gesamt_h"] < 24)
    pruefe("Schlüssel der NK getrennt von Messung", all(sp["perm"] for sp in s1 if sp["teil"] == "nk") and not any(sp.get("perm") for sp in s1 if sp["teil"] == "mess"))
    with tempfile.TemporaryDirectory() as tmp:
        sp = _spec(PROFILE["standard"], "mnist", "gross", TUNING_SEED, tmp, True, teil="tuning", lr=1e-3)
        sp["n"], sp["ck"], sp["val"] = 2, 5, True
        # Divergenz-Pfad
        sp_d = {**sp, "lr": 1e9}
        r = einheit(sp_d)
        pruefe("Divergenz-Pfad (lr 1e9): instabil oder sehr schlecht", r["instabil"] or r["score"] < 0.3)
        # Checkpoint / Wiederaufnahme (Bild und LM)
        ref = trainiere_bild(baue_gross("mnist", 5), "mnist", 5, 1e-3, 1, klein=True)
        netz_a = baue_gross("mnist", 5)
        ck = str(Path(tmp) / "a.pt")
        try:
            trainiere_bild(netz_a, "mnist", 5, 1e-3, 1, klein=True, ckpt=ck, ckpt_alle=5, stopp_bei=7)
            pruefe("Bild: Abbruch ausgeloest", False)
        except _Abbruch:
            pass
        pruefe("Bild: Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        netz_b = baue_gross("mnist", 5)
        fort = trainiere_bild(netz_b, "mnist", 5, 1e-3, 1, klein=True, ckpt=ck, ckpt_alle=5)
        netz_r = baue_gross("mnist", 5)
        trainiere_bild(netz_r, "mnist", 5, 1e-3, 1, klein=True)
        pruefe("Bild: Fortsetzung ab Schritt 5", fort["fortgesetzt_ab"] == 5)
        pruefe("Bild: Fortsetzung == ungestoerter Lauf (Gewichte identisch)", all(torch.allclose(a, b, atol=1e-6) for a, b in zip(netz_b.state_dict().values(), netz_r.state_dict().values())))
        pruefe("Bild: Checkpoint nach Ende geloescht", not Path(ck).exists())
        ck2 = str(Path(tmp) / "b.pt")
        n1 = LM(64, 2, 80)
        try:
            trainiere_lm(n1, 3, 2e-3, 20, klein=True, ckpt=ck2, ckpt_alle=8, stopp_bei=12)
        except _Abbruch:
            pass
        n2 = LM(64, 2, 80)
        # gleiche Anfangsgewichte wie n1 sind fuer die Fortsetzung unerheblich (Checkpoint ueberschreibt); Referenz mit gleichem Seed:
        torch.manual_seed(9)
        nr1 = LM(64, 2, 80)
        torch.manual_seed(9)
        nr2 = LM(64, 2, 80)
        trainiere_lm(nr1, 3, 2e-3, 20, klein=True)
        torch.manual_seed(9)
        n1b = LM(64, 2, 80)
        try:
            trainiere_lm(n1b, 3, 2e-3, 20, klein=True, ckpt=ck2 + "x", ckpt_alle=8, stopp_bei=12)
        except _Abbruch:
            pass
        fl = trainiere_lm(nr2, 3, 2e-3, 20, klein=True, ckpt=ck2 + "x", ckpt_alle=8)
        pruefe("LM: Fortsetzung ab Schritt 8 und identisch zum ungestoerten Lauf", fl["fortgesetzt_ab"] == 8 and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(nr1.state_dict().values(), nr2.state_dict().values())))
        # Checkpoint mit anderer Lernrate wird nicht verwendet
        try:
            trainiere_lm(LM(64, 2, 80), 3, 2e-3, 20, klein=True, ckpt=ck2 + "y", ckpt_alle=8, stopp_bei=9)
        except _Abbruch:
            pass
        anders = trainiere_lm(LM(64, 2, 80), 3, 1e-3, 20, klein=True, ckpt=ck2 + "y", ckpt_alle=8)
        pruefe("Checkpoint mit anderer Lernrate wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        # Wiederaufnahme je Aufgabe (_rechne)
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [{"teil": "mess", "ds": "mnist", "art": "x", "seed": s_, "bits": 2} for s_ in range(9)]
        z1 = _rechne(a4[:4], pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        pruefe("Wiederaufnahme je Aufgabe", len(z1) == 4 and len(z2) == 9 and all(z2[k_] == z1[k_] for k_ in z1))
        # gross-Modell: erneuter Aufruf laedt die Datei statt zu trainieren
        sp2 = _spec(PROFILE["standard"], "mnist", "gross", TUNING_SEED, tmp, True, teil="tuning", lr=1e-3)
        sp2["n"], sp2["ck"] = 1, 5
        _, i1 = hole_gross(sp2, 1e-3)
        t0 = time.time()
        _, i2 = hole_gross(sp2, 1e-3)
        pruefe("Gross-Modell aus Datei (Cache)", _modell_pfad(sp2, 1e-3).exists() and time.time() - t0 < 2.0 and i1 == i2)
        # Prozesspool (spawn) kennt das Modul
        pp = Path(tmp) / "p.json"
        spp = {**_spec(PROFILE["standard"], "mnist", "klein", 1, tmp, True, bits=2, shape="flach", lr=1e-3), "n": 1, "ck": 5}
        try:
            zp = _rechne([spp], pp, {"k": 2}, fn=_arbeiter, prozesse=2, schluessel=_schl)
            pruefe("Prozesspool (spawn) rechnet eine Einheit", len(zp) == 1 and 0 <= list(zp.values())[0]["wert"] <= 1)
        except Exception as ex:
            pruefe(f"Prozesspool (spawn): {ex!r}", False)
        # Einheiten einzeln (klein, alle Arme auf kleinen Daten)
        for art, kw in (("klein", {"bits": 2, "shape": "zwei"}), ("nr", {"bits": 2}), ("svd", {"bits": 2, "var": "gleich", "gross_lr": 1e-3}), ("svd", {"bits": 3, "var": "entropie", "gross_lr": 1e-3}), ("gross", {})):
            s_ = {**_spec(PROFILE["standard"], "mnist", art, TUNING_SEED, tmp, True, lr=1e-3, **kw), "n": 1, "nt": 1, "ck": 5}
            r_ = einheit(s_)
            pruefe(f"Einheit {art} {kw} liefert Wert", math.isfinite(r_["score"]) and not r_["instabil"] and (art != "gross" or r_["q"]["b2"]["had"]["bytes"] > 0))
            if art == "svd":
                pruefe("svd: Parameter <= Budget, Raenge >= 1", r_["bytes_fp16_linear"] <= ziel_bytes("mnist", s_["bits"]) and min(r_["raenge"]) >= 1)
            if art == "klein":
                pruefe("klein: Bytes <= Ziel", r_["bytes_fp16_linear"] <= r_["ziel"])
    # --- Urteilszweige erzwungen
    rng = np.random.default_rng(13)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k_, wv in soll.items():
            gesehen.add((k_.split("|")[0], wv.split(" (")[0]))
            if not u[k_].startswith(wv):
                print(f"  Vortest FEHLER (I-048) {name}: {k_} soll {wv}, ist {u[k_]}", flush=True)
                ok = False

    alle = lambda fam, wert: {f"{fam}|{ds}" + (f"|b{b}" if fam != "H1" else ""): wert for ds in DATENSAETZE for b in (BITS if fam != "H1" else (None,))}
    u0 = _u(rng, vor_h1=1.0, d_orth=0.0, vor_h3=0.5, sd=0.02)
    erwarte("alles gut", u0, **{"NK|mnist": "BESTANDEN", "NK|lm": "BESTANDEN", **alle("H1", "HADAMARD_BESSER"), **alle("H2", "AEQUIVALENT"), **alle("H3", "ENTROPIE_BESSER")})
    u1 = _u(rng, vor_h1=-1.0, d_orth=1.0, vor_h3=-0.5, sd=0.02)
    erwarte("gegen die Hypothesen", u1, **{**alle("H1", "H0"), **alle("H2", "UNTERSCHIEDLICH"), **alle("H3", "H0")})
    u2 = _u(np.random.default_rng(1), vor_h1=0.0, d_orth=0.0, vor_h3=0.0, sd=0.6, n=3)
    erwarte("kein Unterschied, grosse Streuung", u2, **{**alle("H1", "HU"), **alle("H2", "HU"), **alle("H3", "HU")})
    erwarte("NK faellt", _u(rng, nk_ok=False), **{"NK|mnist": "DURCHGEFALLEN", "H1|mnist": "HU (Negativkontrolle", "H2|lm|b2": "HU (Negativkontrolle", "H3|fashion|b3": "HU (Negativkontrolle"})
    erwarte("Divergenz fp lm", _u(rng, div=("lm", "fp")), **{"H1|lm": "HU (Divergenz", "H1|mnist": "HADAMARD_BESSER"})
    erwarte("Kollaps klein mnist", _u(rng, koll=("mnist", "klein"), sd=0.05), **{"H1|mnist": "HU (Kollaps", "H1|fashion": "HADAMARD_BESSER"})
    erwarte("Zelle fehlt", _u(rng, fehlt=("fashion", "orth")), **{"H2|fashion|b2": "HU (Zellen fehlen", "H2|mnist|b2": "AEQUIVALENT"})
    erwarte("Parameter ungleich", _u(rng, param_abw=0.05), **{"H3|mnist|b2": "HU (Parameter nicht gleich", "H1|mnist": "HADAMARD_BESSER"})
    a_ = auswertung(_synth(rng, div=("lm", "fp")), {}, 5.0)
    pruefe("Divergenzrate 1/10 fuer lm|fp|b0, Seed bleibt in den Zahlen (n = 10)", abs(a_["divergenzrate"]["lm|fp|b0"] - 0.1) < 1e-9 and a_["diff"]["H1|lm"]["n"] == 10)
    pruefe("Holm-Familien: H1 (3), H2 (6), H3 (6)", sum(k.startswith("H1|") for k in a_["p_holm"]) == 3 and sum(k.startswith("H2|") for k in a_["p_holm"]) == 6
           and sum(k.startswith("H3|") for k in a_["p_holm"]) == 6 and "H1b|lm" in a_["diff"] and not any(k.startswith("H1b") for k in a_["p_holm"]))
    pruefe("Beschreibend: roh/had/orth, nr/klein, ref64 (LM 4 Bit)", "Vorsprung_had_gegen_roh" in a_["beschreibend"] and "Vorsprung_nr_gegen_klein" in a_["beschreibend"]
           and "Vorsprung_had4_gegen_ref64_lm" in a_["beschreibend"])
    # Richtung der Vorzeichen: Bild = acc grosser besser, LM = bpc kleiner besser
    pruefe("Vorzeichen: _vs", _vs("mnist", 98.0, 97.0) == 1.0 and abs(_vs("lm", 3.0, 3.1) - 0.1) < 1e-9)
    s_ = _stat_rand([0.5] * 5 + [0.51] * 5, 0.2)
    pruefe("Randtest: Mittel 0,5 > Rand 0,2 -> p_gr klein", s_["p_gr"] < 0.01)
    s_ = _tost([0.01, -0.02, 0.0, 0.01, -0.01, 0.02, 0.0, -0.01, 0.01, 0.0], 0.2)
    pruefe("TOST: Differenzen um 0 bei Rand 0,2 -> aequivalent", s_["p_tost"] < 0.01)
    erlaubt = ("HADAMARD_BESSER", "AEQUIVALENT", "UNTERSCHIEDLICH", "ENTROPIE_BESSER", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend", "FEHLT")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u(rng).values()))
    soll = {"H1": {"HADAMARD_BESSER", "H0", "HU"}, "H2": {"AEQUIVALENT", "UNTERSCHIEDLICH", "HU"}, "H3": {"ENTROPIE_BESSER", "H0", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {wv for k_, wv in gesehen if k_ == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    return bool(ok)


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok = _vortest_lauf() and ok
    print(f"  Vortest I-048: {'bestanden' if ok else 'DURCHGEFALLEN'}", flush=True)
    return bool(ok)
