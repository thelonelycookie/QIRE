# -*- coding: utf-8 -*-
"""Abteilung Struktur -- I-043-1B „Strukturadapter auf LLäMmlein-1B gegen LoRA“ (GPU, MPCDF Raven).
(Folge von I-044/I-045: Struktur hilft am ehesten HINTER gelernten Merkmalen. Im vortrainierten 1-Mrd.-Modell sitzen diese Merkmale schon.
 Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

FRAGE: Ist ein Strukturadapter (Fastfood/Hadamard-Mischung mit trainierten Diagonalen und Born-Quadrat z²) bei GLEICHER Zahl trainierbarer
  Parameter besser als LoRA beim Feinabstimmen von LLäMmlein-1B auf deutschen Text?
MODELL: LSX-UniWue/LLaMmlein_1B (Llama-Architektur, 22 Lagen, hidden 2048, 32 Köpfe, 4 KV-Köpfe, FFN 5632, Vokabular 32064, 1,1 Mrd. Parameter),
  alle Basisgewichte EINGEFROREN, bf16. Adapter: nur q_proj (2048 -> 2048) und v_proj (2048 -> 256) jeder Lage (44 Matrizen), parallel zur
  Basismatrix: y = W x + dW x. Adapterparameter fp32, Adapterrechnung fp32 (Ausgabe zurück nach bf16); alle Arme gleich behandelt.
DATEN: deutsche Wikipedia (HF-Datensatz wikimedia/wikipedia, Stand 20231101.de; Lizenz CC BY-SA 3.0 und GFDL), Skript MPI/daten_i043.py/.sh.
  Fester Split NACH ARTIKEL (SHA-1 der Artikel-ID): Training / Validierung (nur Tuning und Zwischenkurve) / Test (nur am Ende). Token-Ströme als
  uint16 (.npy) unter /ptmp/$USER/i043; Fenster zu 512 Tokens, Artikel durch </s> getrennt. Fallback: UD_German-GSD (CC BY-SA 4.0, wie I-045).
ARME (Parameterzahlen für die echten Maße siehe `param_*`; im Vortest gegen die gebauten Modelle gezählt):
  L16       LoRA r = 16, alpha 32 (Referenz, wie MPI/slurm/gpu_test.sbatch), 2.252.800
  L_budget  LoRA mit Rang r = 4 (alpha 8, gleiches alpha/r = 2): Rang so, dass die Parameterzahl der des Strukturadapters am nächsten kommt, 563.200
  FF        Fastfood-/Born-Adapter: dW x = c · D3 ⊙ H( D2 ⊙ fold( (H(D1 ⊙ x))² ) ), H = normierte Walsh-Hadamard (zwei kleine Matmuls),
            3 parallele Zweige mit eigenen Vorzeichen-Diagonalen, c = 2 fest, D3 = 0 am Start (Start = Basismodell), 574.464 (+2,0 % gegen L_budget)
  FF_gelu   wie FF mit GELU statt z² (Kontrolle: trägt das Born-Quadrat?), gleiche Parameter
  (basis    kein Training: Verlust des unveränderten Modells als Bezug.)
  QIRE-Adapter aus Projekt A (Drehungen, T-004..T-006) bewusst NICHT portiert: er hat auf dem 120M-Modell ein festes, winziges Budget (L10 = 1.920
  Parameter), sättigt (T-006, b = 0,08) und ist an 256er-Blöcke/8 Qubits gebunden; bei gleichem Budget wie LoRA ließe er sich nur durch Umbau
  (neue Schichtzahl, neue Blöcke) vergleichen, also nicht mehr „der Adapter aus Projekt A“. Der Fastfood-/Born-Adapter ist der Nachfolger
  (T-007), er wird hier portiert.
TRAINING: AdamW (Gewichtszerfall 0), Kosinus mit 100 Schritten linearem Aufwärmen, Gradientenbegrenzung 1,0, Batch 8 x 512 Tokens (4.096 Tokens/Schritt),
  PROFIL „standard“ = 6.000 Schritte = 24,58 Mio Tokens je Lauf. Lernrate je Arm auf EINEM Tuning-Seed bei 1.500 Schritten (¼ des Budgets,
  Validierung = erste 512 Fenster; gittersuche_stabil, Gitter 1e-4/3e-4/1e-3/3e-3, Faktor 3, beidseitig verlängerbar, Score = −Verlust, Kollaps 0,05 nats).
  Gleiche Datenreihenfolge und gleiche Initialisierungs-Zufallszahlen je Seed für alle Arme (gepaart).
  Wiederaufnahme: je Aufgabe in ergebnisse/_zwischen_I-043-1B.json; Checkpoint in der Aufgabe alle 500 Schritte (nur Adapter + Optimierer, wenige MB) in
  ergebnisse/_ckpt_I-043-1B/, exakt fortsetzbar (Datenreihenfolge ist Funktion von Seed und Schritt).
MESSUNG: Test-Verlust in nats/Token (nicht überlappende 512er-Fenster, alle Positionen) und bits/Zeichen (bpc, aus dem Zeichen/Token-Verhältnis des
  Test-Textes); Parameter; Tokens/s; Speicher; Adapter-FLOPs/Token (beschreibend).
HYPOTHESEN (gepaarter einseitiger t-Test, n = 5, Differenzen je Seed, Holm über H1, H2, α = 0,05):
  H1 FF < L_budget (Test-Verlust): STRUKTUR_BESSER | H0 (L_budget sicher besser) | HU.
  H2 FF < FF_gelu (Born-Quadrat trägt): BORN_TRAEGT | H0 (FF_gelu sicher besser) | HU.
  H3 beschreibend, ohne Urteil und ohne Holm: FF gegen L16 (4x mehr Parameter bei L16): Differenz, KI, Wilcoxon.
KONTROLLEN: Positivkontrolle PK: L16 verbessert den Test-Verlust gegenüber dem Basismodell im Mittel um >= 0,02 nats, sonst H1/H2 = HU (die Aufgabe
  trägt nicht). Negativkontrolle NK: L_budget und FF mit im Fenster PERMUTIERTEN Trainingstokens (Seeds = die ersten zwei Mess-Seeds, 1.500 Schritte,
  Test auf echtem Text): bestanden, wenn der mittlere Test-Verlust >= 0,9 x Basis-Verlust (ein Datenleck/Label-Fehler würde weit darunter liegen), sonst
  alles HU. Budgetgleichheit |FF/L_budget − 1| <= 5 % an den GEZÄHLTEN Lauf-Parametern, sonst H1 = HU. Divergenz (nicht endlicher Verlust) oder Kollaps
  (Seed > Median + 0,1 nats) in einem beteiligten Arm: dessen Hypothesen HU.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from abteilungen.clifford_suche import holm
from abteilungen.struktur_gross import _lade_zwischen, _schreibe_zwischen
from abteilungen.struktur_sprache import _stat
from labor.gemeinsam import ERGEBNISSE, WURZEL, gittersuche_stabil

KARTE = "I-043-1B"
MODELL_NAME = "LSX-UniWue/LLaMmlein_1B"
DIMS_1B = {"lagen": 22, "hidden": 2048, "kv": 256, "heads": 32, "kv_heads": 4, "inter": 5632, "vocab": 32064}
ZIELE = ("q_proj", "v_proj")
ARME = ("L16", "L_budget", "FF", "FF_gelu")
NK_ARME = ("L_budget", "FF")
LORA_R16 = 16
LORA_SKALA = 2.0                 # alpha / r (wie gpu_test: r = 16, alpha = 32)
FF_K = 3                         # parallele Fastfood-Zweige
FF_SKALA = 2.0                   # c, fest (wie T-007)
SEEDS = (2670, 2671, 2672, 2673, 2674)
TUNING_SEED = 2692
NK_SEEDS = SEEDS[:2]
# Seeds, die andere Karten belegen (Vortest prueft zusaetzlich gegen seeds.json): hier nur die Bloecke der letzten Wochen
_BELEGT = (set(range(2600, 2610)) | set(range(2620, 2630)) | set(range(2640, 2650)) | set(range(2660, 2670)) | {2694, 2695, 2696, 2697, 2698, 2699})
LR_GITTER = (1e-4, 3e-4, 1e-3, 3e-3)
LR_FAKTOR = 3
KOLLAPS_SCORE = 0.05             # Tuning: Score = -Verlust (nats)
KOLLAPS_VERLUST = 0.1            # Messung: Seed schlechter als Median + 0,1 nats -> HU
PK_MIN = 0.02                    # Positivkontrolle: L16 besser als Basis um >= 0,02 nats
NK_FAKTOR = 0.9
BUDGET_TOL = 0.05
ALPHA = 0.05
TUNING_VAL_FENSTER = 512

# ---- Laufzeitprotokoll (VOR dem Lauf festgelegt; Begruendung im LABORBUCH) ----
PROFILE = {
    "standard": {"schritte": 6000, "tuning_schritte": 1500, "nk_schritte": 1500, "bs": 8, "laenge": 512, "warm": 100, "ckpt_alle": 500,
                 "zwischen_evals": 3, "seeds": SEEDS, "nk_seeds": NK_SEEDS, "tuning_seed": TUNING_SEED},
    "fallback": {"schritte": 3000, "tuning_schritte": 750, "nk_schritte": 750, "bs": 8, "laenge": 512, "warm": 100, "ckpt_alle": 500,
                 "zwischen_evals": 3, "seeds": SEEDS, "nk_seeds": NK_SEEDS, "tuning_seed": TUNING_SEED},
    "gsd": {"schritte": 600, "tuning_schritte": 150, "nk_schritte": 150, "bs": 4, "laenge": 512, "warm": 30, "ckpt_alle": 100,
            "zwischen_evals": 3, "seeds": SEEDS, "nk_seeds": NK_SEEDS, "tuning_seed": TUNING_SEED},
    # nur Vortest (winziges Zufallsmodell, CPU); Seeds im Erkundungsblock 900-999, nie Messung
    "klein": {"schritte": 60, "tuning_schritte": 30, "nk_schritte": 20, "bs": 4, "laenge": 32, "warm": 5, "ckpt_alle": 25,
              "zwischen_evals": 2, "seeds": (901, 902, 903), "nk_seeds": (901, 902), "tuning_seed": 900},
}
RAUCH = {"schritte": 40, "tuning_schritte": 20, "nk_schritte": 10, "warm": 5, "ckpt_alle": 15, "zwischen_evals": 1, "eval_max": 128}

HYPOTHESEN = {
    "H1": "FF < L_budget bei gleicher Parameterzahl (Test-Verlust, gepaart einseitig t, Holm H1-H2). STRUKTUR_BESSER | H0 (L_budget sicher besser) | HU.",
    "H2": "FF < FF_gelu (Born-Quadrat traegt; gepaart einseitig t, Holm). BORN_TRAEGT | H0 (FF_gelu sicher besser) | HU.",
    "H3": "beschreibend: FF gegen L16 (L16 hat ca. 4x mehr Parameter), ohne Urteil.",
    "PK": f"Positivkontrolle: L16 besser als Basis um >= {PK_MIN} nats im Mittel, sonst H1/H2 = HU.",
    "NK": f"Negativkontrolle (L_budget, FF mit permutierten Trainingstokens): mittlerer Test-Verlust >= {NK_FAKTOR} x Basis-Verlust. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: Test-Verlust und bpc je Arm, Parameter, Tokens/s, Speicher, Adapter-FLOPs, Lernraten, Zwischenkurven.",
}
FOLGEFRAGEN = {
    "H1 STRUKTUR_BESSER": "Strukturadapter traegt am vortrainierten Modell: weitere Ziele (FFN, k/o), Skalierung (Tokens, Modellgroesse) und Rechenzeit-Vergleich.",
    "H1 H0": "Bei gleichem Budget ist LoRA besser: Struktur nur als Speicher-/Parameterargument (I-045-v4), nicht als Qualitaetsvorteil berichten.",
    "H2 BORN_TRAEGT": "Born-Quadrat traegt im Adapter: z² gegen andere Aktivierungen (squared ReLU) und gegen LoRA mit z² testen.",
}


# ------------------------------------------------------------------ Parameter- und FLOP-Zaehlung (aus den Abmessungen)
def matrizen(dims=DIMS_1B):
    """(d_in, d_aus) der Adapterziele je Lage: q_proj, v_proj."""
    return [(dims["hidden"], dims["hidden"]), (dims["hidden"], dims["kv"])]


def param_lora(r, dims=DIMS_1B) -> int:
    return dims["lagen"] * sum(r * (di + do) for di, do in matrizen(dims))


def param_ff(k=FF_K, dims=DIMS_1B) -> int:
    """je Zweig D1 (d_in) + D2 (d_aus) + D3 (d_aus)."""
    return dims["lagen"] * sum(k * (di + 2 * do) for di, do in matrizen(dims))


def rang_budget(ziel, dims=DIMS_1B) -> int:
    """Rang r, dessen LoRA-Parameterzahl der des Strukturadapters am naechsten kommt (Gleichstand: kleinerer Rang)."""
    return min(range(1, 257), key=lambda r: (abs(param_lora(r, dims) - ziel), r))


def zweig_spez(arm, dims=DIMS_1B) -> dict:
    """Bauvorschrift eines Arms: art lora|ff, Rang/Zweige, Aktivierung."""
    if arm == "L16":
        return {"art": "lora", "r": LORA_R16}
    if arm == "L_budget":
        return {"art": "lora", "r": rang_budget(param_ff(FF_K, dims), dims)}
    if arm in ("FF", "FF_gelu"):
        return {"art": "ff", "k": FF_K, "akt": "z2" if arm == "FF" else "gelu"}
    raise ValueError(arm)


def param_arm(arm, dims=DIMS_1B) -> int:
    s = zweig_spez(arm, dims)
    return param_lora(s["r"], dims) if s["art"] == "lora" else param_ff(s["k"], dims)


def _faktoren(n):
    """n = 2^p -> (a, b) mit a·b = n, a = 2^(p//2)."""
    p = int(math.log2(n))
    return 1 << (p // 2), 1 << (p - p // 2)


def flops_je_token(arm, dims=DIMS_1B) -> dict:
    """Adapter-FLOPs je Token (alle Lagen; 1 Multiplikation = 1 Addition = 1 Flop): Matmuls (exakt) und elementweise Operationen (grob).
    FF: je Zweig zwei Walsh-Transformationen als zwei kleine Matmuls, 2·d·(a+b) mit d = a·b."""
    s = zweig_spez(arm, dims)
    mm = el = 0
    for di, do in matrizen(dims):
        if s["art"] == "lora":
            mm += 2 * s["r"] * (di + do)
            el += do
        else:
            a1, b1 = _faktoren(di)
            a2, b2 = _faktoren(do)
            k = s["k"]
            mm += k * (2 * di * (a1 + b1) + 2 * do * (a2 + b2))
            el += k * (di + di + di + do + do + do) + do        # D1, Aktivierung, Falten, D2, D3, Zweigsumme, Skala
    return {"matmul": dims["lagen"] * mm, "elementweise": dims["lagen"] * el, "gesamt": dims["lagen"] * (mm + el)}


def flops_basis_je_token(dims=DIMS_1B) -> int:
    """Vorwaerts-FLOPs des Basismodells je Token (Projektionen, FFN, Ausgabe; ohne Aufmerksamkeitswerte)."""
    h, kv, i = dims["hidden"], dims["kv"], dims["inter"]
    return dims["lagen"] * 2 * (2 * h * h + 2 * h * kv + 3 * h * i) + 2 * h * dims["vocab"]


# ------------------------------------------------------------------ Adapter-Bausteine
def walsh(n: int) -> torch.Tensor:
    """Orthonormale Sylvester-Hadamard-Matrix 2^n x 2^n."""
    h = torch.ones(1, 1)
    for _ in range(n):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(2 ** n)


def _ist_zweierpotenz(n):
    return n > 0 and n & (n - 1) == 0


class LoraZweig(nn.Module):
    """dW x = s · B (A x), A: r x d_in (Start Kaiming-uniform wie peft), B: d_aus x r (Start 0 -> dW = 0). Parameter fp32, Rechnung fp32."""

    def __init__(self, d_in, d_aus, r, gen, skala=LORA_SKALA):
        super().__init__()
        self.skala = skala
        grenze = 1.0 / math.sqrt(d_in)
        self.A = nn.Parameter((torch.rand(r, d_in, generator=gen) * 2 - 1) * grenze)
        self.B = nn.Parameter(torch.zeros(d_aus, r))

    def forward(self, x):
        return self.skala * ((x.float() @ self.A.T) @ self.B.T)


class FFZweig(nn.Module):
    """Fastfood-/Born-Zweig:  dW x = c · Σ_k D3_k ⊙ H( D2_k ⊙ fold( act( H( D1_k ⊙ x ) ) ) ).
    H = orthonormale Walsh-Hadamard = Ha ⊗ Hb (zwei kleine Matmuls), auf d_in bzw. d_aus. fold: ist d_aus < d_in (v_proj), werden die d_in/d_aus
    Bloecke nach act summiert (÷√Bloecke). act = z² (Born: Betragsquadrat der gemischten Amplituden) oder GELU. D1, D2 Start Zufallsvorzeichen,
    D3 Start 0 -> dW = 0, Start = Basismodell. Parameter k·(d_in + 2·d_aus). Klassisch gerechnet (I-038)."""

    def __init__(self, d_in, d_aus, k, akt, gen, skala=FF_SKALA):
        super().__init__()
        if not (_ist_zweierpotenz(d_in) and _ist_zweierpotenz(d_aus) and d_in % d_aus == 0):
            raise ValueError(f"d_in {d_in}, d_aus {d_aus}: Zweierpotenzen, d_aus teilt d_in verlangt")
        if akt not in ("z2", "gelu"):
            raise ValueError(akt)
        self.d_in, self.d_aus, self.k, self.akt, self.skala = d_in, d_aus, k, akt, skala
        self.fold = d_in // d_aus
        a1, b1 = _faktoren(d_in)
        a2, b2 = _faktoren(d_aus)
        self.register_buffer("ha1", walsh(int(math.log2(a1))))
        self.register_buffer("hb1", walsh(int(math.log2(b1))))
        self.register_buffer("ha2", walsh(int(math.log2(a2))))
        self.register_buffer("hb2", walsh(int(math.log2(b2))))

        def zeichen(n):
            return (torch.randint(0, 2, (k, n), generator=gen) * 2 - 1).float()
        self.d1 = nn.Parameter(zeichen(d_in))
        self.d2 = nn.Parameter(zeichen(d_aus))
        self.d3 = nn.Parameter(torch.zeros(k, d_aus))

    @staticmethod
    def _wht(z, ha, hb):
        form = z.shape
        return (ha @ z.reshape(-1, ha.shape[0], hb.shape[0]) @ hb).reshape(form)

    def forward(self, x):
        x = x.float()
        kopf = x.shape[:-1]
        z = x.reshape(-1, 1, self.d_in) * self.d1                       # (N, k, d_in)
        z = self._wht(z, self.ha1, self.hb1)
        z = z * z if self.akt == "z2" else F.gelu(z)
        if self.fold > 1:
            z = z.reshape(z.shape[0], self.k, self.fold, self.d_aus).sum(-2) / math.sqrt(self.fold)
        z = self._wht(z * self.d2, self.ha2, self.hb2) * self.d3        # (N, k, d_aus)
        return (self.skala * z.sum(1)).reshape(*kopf, self.d_aus)


class AdapterLinear(nn.Module):
    """y = W x + dW x; W eingefroren. Beim Training wird der Adapterzweig neu gerechnet statt gespeichert (Speicher; die k Zweige belegen sonst
    mehrere GB Zwischenwerte je Lage)."""

    def __init__(self, basis, zweig, ckpt):
        super().__init__()
        self.basis, self.zweig, self.ckpt = basis, zweig, ckpt

    def forward(self, x):
        y = self.basis(x)
        if self.ckpt and self.training and torch.is_grad_enabled():
            z = torch.utils.checkpoint.checkpoint(self.zweig, x, use_reentrant=False)
        else:
            z = self.zweig(x)
        return y + z.to(y.dtype)


def adapter_einbauen(modell, arm, seed, ckpt_adapter=False):
    """Alle Basisgewichte einfrieren, Adapter auf q_proj und v_proj jeder Lage. Zufallszahlen aus eigenem Generator (Seed), feste Reihenfolge."""
    spez = zweig_spez(arm, _dims_von(modell))
    for p in modell.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    for schicht in modell.model.layers:
        for name in ZIELE:
            lin = getattr(schicht.self_attn, name)
            di, do = lin.in_features, lin.out_features
            z = (LoraZweig(di, do, spez["r"], gen) if spez["art"] == "lora" else FFZweig(di, do, spez["k"], spez["akt"], gen))
            setattr(schicht.self_attn, name, AdapterLinear(lin, z.to(lin.weight.device), ckpt_adapter))
    return modell


def _dims_von(modell) -> dict:
    c = modell.config
    kvk = getattr(c, "num_key_value_heads", c.num_attention_heads)
    return {"lagen": c.num_hidden_layers, "hidden": c.hidden_size, "kv": kvk * (c.hidden_size // c.num_attention_heads),
            "heads": c.num_attention_heads, "kv_heads": kvk, "inter": c.intermediate_size, "vocab": c.vocab_size}


def trainierbar(modell) -> int:
    return int(sum(p.numel() for p in modell.parameters() if p.requires_grad))


# ------------------------------------------------------------------ Daten
def daten_ordner() -> Path:
    if os.environ.get("I043_DATEN"):
        return Path(os.environ["I043_DATEN"])
    ptmp = Path("/ptmp") / os.environ.get("USER", "")
    return ptmp / "i043" if os.environ.get("USER") and ptmp.exists() else WURZEL / "daten" / "i043"


def daten_laden(ordner=None, laenge=512) -> dict:
    """Token-Strome (uint16, memmap) + meta.json -> Fenster zu `laenge` Tokens."""
    ordner = Path(ordner) if ordner else daten_ordner()
    if not (ordner / "meta.json").exists():
        raise FileNotFoundError(f"Daten fehlen: {ordner}/meta.json (erst MPI/daten_i043.sh auf dem Login-Knoten ausfuehren)")
    meta = json.loads((ordner / "meta.json").read_text(encoding="utf-8"))
    D = {"meta": meta, "ordner": str(ordner)}
    for teil in ("train", "val", "test"):
        a = np.load(ordner / f"{teil}.npy", mmap_mode="r")
        n = len(a) // laenge
        D[teil] = a[: n * laenge].reshape(n, laenge)
    return D


def daten_kennung(meta) -> dict:
    """Was in die Kennung der Zwischendatei gehoert: andere Daten -> nicht mischen."""
    return {k: meta.get(k) for k in ("quelle", "tokens", "sha256", "tokenizer")}


def daten_klein(laenge=32, vokabular=64) -> dict:
    """Vortest: zwei Markov-Quellen (A: Vortraining des winzigen Basismodells, B: Zieldomäne) als Token-Ströme."""
    def uebergang(seed):
        r = np.random.default_rng(seed)
        p = np.full((vokabular, vokabular), 0.02 / vokabular)
        for i in range(vokabular):
            p[i, r.choice(vokabular, 3, replace=False)] += np.array([0.5, 0.3, 0.2])
        return p / p.sum(1, keepdims=True)

    def strom(p, n, seed):
        r = np.random.default_rng(seed)
        cum = np.cumsum(p, 1)
        x = np.empty(n, dtype=np.uint16)
        x[0] = r.integers(vokabular)
        u = r.random(n)
        for i in range(1, n):
            x[i] = min(int(np.searchsorted(cum[x[i - 1]], u[i])), vokabular - 1)
        return x

    pb = uebergang(2)
    D = {"meta": {"quelle": "klein", "tokens": {"train": 40000, "val": 4096, "test": 4096}, "sha256": None, "tokenizer": None,
                  "zeichen": {"val": 4096, "test": 4096}}, "ordner": None}
    for teil, n, s in (("train", 40000, 11), ("val", 4096, 12), ("test", 4096, 13)):
        a = strom(pb, n, s)
        D[teil] = a[: (n // laenge) * laenge].reshape(-1, laenge)
    D["vortraining"] = strom(uebergang(1), 40000, 10)[: (40000 // laenge) * laenge].reshape(-1, laenge)
    return D


def reihenfolge(seed, n_fenster, schritte, bs) -> np.ndarray:
    """Fensterindizes aller Schritte (ohne Zuruecklegen, bei Bedarf mehrere Durchgaenge); Funktion von Seed -> Wiederaufnahme exakt."""
    rng = np.random.default_rng([int(seed), 43])
    brauche = schritte * bs
    teile, n = [], 0
    while n < brauche:
        teile.append(rng.permutation(n_fenster))
        n += n_fenster
    return np.concatenate(teile)[:brauche].reshape(schritte, bs)


def permutiere_fenster(x: torch.Tensor, seed, schritt) -> torch.Tensor:
    """Negativkontrolle: Tokens INNERHALB jedes Fensters zufaellig umordnen (Kontext traegt nichts, Unigramme bleiben)."""
    rng = np.random.default_rng([int(seed), int(schritt), 7])
    p = torch.from_numpy(np.argsort(rng.random(x.shape), axis=1)).to(x.device)
    return torch.gather(x, 1, p)


# ------------------------------------------------------------------ Modell-Fabriken
_KLEIN_BASIS = {}


def fabrik_klein(D=None):
    """Winziges Llama (2 Lagen), kurz auf Quelle A vortrainiert und zwischengespeichert (deterministisch): Ersatz fuer das 1B-Modell im Vortest."""
    from transformers import LlamaConfig, LlamaForCausalLM
    if "sd" not in _KLEIN_BASIS:
        D = D or daten_klein()
        torch.manual_seed(1234)
        cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=64, tie_word_embeddings=False)
        m = LlamaForCausalLM(cfg)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        w = D["vortraining"]
        for t in range(150):
            x = torch.from_numpy(w[(np.arange(8) + 8 * t) % len(w)].astype(np.int64))
            opt.zero_grad()
            m(input_ids=x, labels=x).loss.backward()
            opt.step()
        _KLEIN_BASIS["sd"] = {k: v.clone() for k, v in m.state_dict().items()}
        _KLEIN_BASIS["cfg"] = cfg
    m = LlamaForCausalLM(_KLEIN_BASIS["cfg"])
    m.load_state_dict(_KLEIN_BASIS["sd"])           # Basismodell: fuer ALLE Arme dasselbe, eingefroren (erbt nicht von einem Arm); Adapter beginnen frisch, eigene Rate je Arm
    return m.eval()


def fabrik_real(geraet="cuda"):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODELL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    m.config.use_cache = False
    return m.to(geraet).eval()


# ------------------------------------------------------------------ Training und Bewertung
class _Abbruch(Exception):
    """Nur fuer den Vortest: simuliert einen harten Abbruch (24-h-Limit) mitten in einer Aufgabe."""


@torch.no_grad()
def bewerte(modell, fenster, n=None, bs=16, geraet="cpu") -> float:
    """Mittlerer Verlust (nats/Token) ueber die ersten n Fenster (alle Vorhersagepositionen)."""
    war = modell.training
    modell.eval()
    n = len(fenster) if n is None else min(n, len(fenster))
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        x = torch.from_numpy(np.asarray(fenster[i:min(i + bs, n)]).astype(np.int64)).to(geraet)
        l = modell(input_ids=x, labels=x).loss.float()
        k = x.shape[0] * (x.shape[1] - 1)
        tot += float(l) * k
        cnt += k
    modell.train(war)
    return tot / cnt


def _json_sicher(x):
    if isinstance(x, dict):
        return {k: _json_sicher(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sicher(v) for v in x]
    return x


def ckpt_pfad(ordner, key) -> str:
    return str(Path(ordner) / (key.replace("|", "_").replace("/", "-") + ".pt"))


def _ckpt_speichern(pfad, meta, modell, opt, schritt, verlauf, zkurve):
    Path(pfad).parent.mkdir(parents=True, exist_ok=True)
    tr = {n: p.detach().cpu() for n, p in modell.named_parameters() if p.requires_grad}
    tmp = pfad + ".tmp"
    torch.save({"meta": meta, "param": tr, "opt": opt.state_dict(), "schritt": schritt, "verlauf": verlauf, "zkurve": zkurve}, tmp)
    os.replace(tmp, pfad)


def _ckpt_laden(pfad, meta):
    """Checkpoint nur verwenden, wenn die Metadaten (Arm, Rate, Seed, Schritte, Batch, ...) genau passen."""
    if not pfad or not os.path.exists(pfad):
        return None
    try:
        d = torch.load(pfad, map_location="cpu", weights_only=True)
    except Exception:
        return None
    return d if d.get("meta") == meta else None


def trainiere(arm, lr, seed, schritte, D, K, *, fabrik, geraet="cpu", ckpt=None, permutiert=False, val_fenster=None, mit_test=False,
              zwischen_evals=0, _stopp_bei=None, eltern=None):
    """Ein Lauf: Adapter-Arm `arm` auf Seed `seed`. Rueckgabe: Verlaeufe, val/test (nats/Token), Parameter, Tokens/s. `val_fenster`: Fensterzahl der
    Validierung am Ende (None = alle). Nicht endlicher Verlust -> {'endlich': False}. Checkpoint alle K['ckpt_alle'] Schritte, exakt fortsetzbar."""
    bs, L = K["bs"], K["laenge"]
    mikro = int(os.environ.get("I043_MIKRO", bs))
    if bs % mikro:
        raise ValueError("I043_MIKRO muss den Batch teilen")
    torch.manual_seed(seed)
    modell = fabrik()
    adapter_einbauen(modell, arm, seed, ckpt_adapter=(geraet == "cuda"))
    modell.train()
    params = [p for p in modell.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    meta = {"arm": arm, "lr": lr, "seed": seed, "schritte": schritte, "bs": bs, "laenge": L, "warm": K["warm"], "permutiert": permutiert,
            "daten": _json_sicher(daten_kennung(D["meta"])), "evals": zwischen_evals}
    start, verlauf, zkurve = 0, [], []
    d = _ckpt_laden(ckpt, meta)
    if d is not None:
        for n, p in modell.named_parameters():
            if p.requires_grad:
                p.data.copy_(d["param"][n].to(p.device))
        opt.load_state_dict(d["opt"])
        start, verlauf, zkurve = d["schritt"], list(d["verlauf"]), list(d["zkurve"])
    order = reihenfolge(seed, len(D["train"]), schritte, bs)
    eval_bei = {round(schritte * (i + 1) / (zwischen_evals + 1)) for i in range(zwischen_evals)}
    tv = TUNING_VAL_FENSTER
    t0, schritte_hier, summe, n_summe, nebenzeit = time.time(), 0, 0.0, 0, 0.0
    takt = max(1, schritte // 60)
    for t in range(start, schritte):
        if _stopp_bei is not None and t == _stopp_bei:
            raise _Abbruch(t)
        f = (t + 1) / K["warm"] if t < K["warm"] else 0.5 * (1 + math.cos(math.pi * (t - K["warm"]) / max(1, schritte - K["warm"])))
        for g in opt.param_groups:
            g["lr"] = lr * f
        x = torch.from_numpy(np.asarray(D["train"][np.sort(order[t])]).astype(np.int64))
        # Reihenfolge innerhalb des Batches ist egal (Mittel); sortierte Indizes machen den memmap-Zugriff schneller
        x = x.to(geraet)
        if permutiert:
            x = permutiere_fenster(x, seed, t)
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=geraet)
        for m in range(0, bs, mikro):
            xm = x[m:m + mikro]
            l = modell(input_ids=xm, labels=xm).loss * (xm.shape[0] / bs)
            l.backward()
            tot += l.detach().float()
        v = float(tot)
        if not math.isfinite(v):
            return {"endlich": False, "divergiert_bei": t, "val": float("nan"), "test": float("nan"), "trainierbar": trainierbar(modell),
                    "schritte": schritte, "verlauf_train": verlauf, "zwischenkurve": zkurve}
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        schritte_hier += 1
        summe, n_summe = summe + v, n_summe + 1
        if (t + 1) % takt == 0:
            verlauf.append([t + 1, summe / n_summe])
            summe, n_summe = 0.0, 0
        if (t + 1) in eval_bei and t + 1 < schritte:
            tn = time.time()
            zkurve.append([t + 1, bewerte(modell, D["val"], tv, geraet=geraet)])
            nebenzeit += time.time() - tn
        if ckpt and (t + 1) % K["ckpt_alle"] == 0 and t + 1 < schritte:
            tn = time.time()
            _ckpt_speichern(ckpt, meta, modell, opt, t + 1, verlauf, zkurve)
            nebenzeit += time.time() - tn
            if eltern is not None and os.getppid() != eltern:        # Hauptprozess weg (SIGTERM): nicht als Waise weiterrechnen
                raise SystemExit("Hauptprozess beendet")
    if geraet == "cuda":
        torch.cuda.synchronize()
    sek = time.time() - t0 - nebenzeit              # reine Trainingszeit (ohne Zwischenbewertungen und Checkpoints)
    emax = K.get("eval_max")
    val_fenster = val_fenster or emax
    r = {"endlich": True, "trainierbar": trainierbar(modell), "schritte": schritte, "tokens": schritte * bs * L,
         "val": bewerte(modell, D["val"], val_fenster, geraet=geraet), "verlauf_train": verlauf, "zwischenkurve": zkurve,
         "sekunden_training": sek, "tokens_s": schritte_hier * bs * L / sek if schritte_hier else float("nan"),
         "fortgesetzt_ab": start, "max_speicher_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if geraet == "cuda" else 0.0}
    if mit_test:
        r["test"] = bewerte(modell, D["test"], emax, geraet=geraet)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    return r


# ------------------------------------------------------------------ Aufgaben und Pool (je GPU ein Prozess)
_CTX = {}
_ELTERN = [None]


def _ctx(a):
    """Daten und Fabrik des Arbeiters (je Prozess einmal)."""
    if "D" not in _CTX:
        if a["klein"]:
            _CTX["D"] = daten_klein()
            _CTX["fabrik"] = lambda: fabrik_klein(_CTX["D"])
        else:
            _CTX["D"] = daten_laden(a["daten"], laenge=a["laenge"])
            geraet = a["geraet"]
            _CTX["fabrik"] = lambda: fabrik_real(geraet)
    return _CTX["D"], _CTX["fabrik"]


def _schl(a) -> str:
    return f"{a['teil']}|{a['arm']}|{a['seed']}|{a['lr']!r}|T{a['schritte']}" if a["teil"] != "basis" else "basis"


def _arbeiter(a):
    D, fabrik = _ctx(a)
    key = _schl(a)
    K = a["K"]
    if a["geraet"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if a["teil"] == "basis":
        t0 = time.time()
        m = fabrik()
        emax = K.get("eval_max")
        r = {"teil": "basis", "arm": "basis", "val": bewerte(m, D["val"], emax, geraet=a["geraet"]), "test": bewerte(m, D["test"], emax, geraet=a["geraet"]),
             "endlich": True, "sekunden": time.time() - t0, "trainierbar": 0}
        del m
    else:
        tun = a["teil"] == "tun"
        r = trainiere(a["arm"], a["lr"], a["seed"], a["schritte"], D, K, fabrik=fabrik, geraet=a["geraet"], ckpt=a["ckpt"],
                      permutiert=(a["teil"] == "nk"), val_fenster=(TUNING_VAL_FENSTER if tun else None), mit_test=(a["teil"] in ("mess", "nk")),
                      zwischen_evals=(0 if tun else K["zwischen_evals"]), eltern=_ELTERN[0])
        r.update({"teil": a["teil"], "arm": a["arm"], "seed": a["seed"], "lr": a["lr"]})
        if tun:
            r["acc"] = -r["val"] if r["endlich"] else -1e9           # Score fuer gittersuche_stabil (hoeher = besser)
    if a["geraet"] == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return key, r


def _init_arbeiter(q, gpu_modus, eltern):
    _ELTERN[0] = eltern
    if gpu_modus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())      # vor dem ersten CUDA-Aufruf: dieser Arbeiter sieht genau eine GPU
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(int(os.environ.get("I043_THREADS", 4 if gpu_modus else 1)))


def gpu_liste():
    """Sichtbare GPU-Nummern (Slurm setzt CUDA_VISIBLE_DEVICES)."""
    ev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if ev.strip():
        return [x.strip() for x in ev.split(",") if x.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def _rechne(aufgaben, pfad: Path, kennung: dict, prozesse=None, klein=False) -> dict:
    """Wie struktur_gross._rechne, aber GPU-bewusst: ein Arbeiter je GPU (spawn), SIGTERM beendet Arbeiter sauber; jede Aufgabe sofort atomar gesichert."""
    zeilen = _lade_zwischen(pfad, kennung)
    offen = [a for a in aufgaben if _schl(a) not in zeilen]
    if len(offen) < len(aufgaben):
        print(f"  Wiederaufnahme: {len(aufgaben) - len(offen)} von {len(aufgaben)} aus {pfad.name}", flush=True)
    gpu_modus = (not klein) and torch.cuda.is_available()
    ids = gpu_liste() if gpu_modus else []
    n = prozesse or int(os.environ.get("I043_ARBEITER", len(ids) if gpu_modus else os.environ.get("INSTITUT_PROZESSE", "1")))
    n = max(1, min(n, len(ids)) if gpu_modus else n)

    def ablegen(erg):
        k, z = erg
        zeilen[k] = z
        _schreibe_zwischen(pfad, kennung, zeilen)
        print(f"  fertig: {k} ({len(zeilen)} in Zwischendatei)", flush=True)

    if not offen:
        return zeilen
    if n <= 1 and not gpu_modus:
        for a in offen:
            ablegen(_arbeiter(a))
        return zeilen
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):       # Kindprozesse erben 1; GPU-Arbeiter setzen ihre Threads in _init_arbeiter
        os.environ.setdefault(var, "1")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in ids[:n]:
        q.put(i)
    pool = ctx.Pool(n, initializer=_init_arbeiter, initargs=(q, gpu_modus, os.getpid()))
    alt = signal.getsignal(signal.SIGTERM)

    def _term(signum, frame):
        print("  SIGTERM: beende Arbeiter (Checkpoints in den Aufgaben bleiben).", flush=True)
        pool.terminate()
        sys.exit(143)
    try:
        signal.signal(signal.SIGTERM, _term)
    except ValueError:        # nicht im Hauptthread
        pass
    try:
        for erg in pool.imap_unordered(_arbeiter, offen):
            ablegen(erg)
        pool.close()
    finally:
        pool.terminate()
        pool.join()
        try:
            signal.signal(signal.SIGTERM, alt)
        except ValueError:
            pass
    return zeilen


def _basis_aufgabe(**kw):
    return {"teil": "basis", "arm": "basis", "seed": 0, "lr": 0.0, "schritte": 0, **kw}


def _aufgabe(teil, arm, seed, lr, schritte, ctx):
    return {"teil": teil, "arm": arm, "seed": seed, "lr": lr, "schritte": schritte, "ckpt": ckpt_pfad(ctx["ckordner"], f"{teil}|{arm}|{seed}|{lr!r}|T{schritte}"), **{k: ctx[k] for k in ("K", "klein", "geraet", "daten", "laenge")}}


def aufgaben_mess(hp, P, ctx):
    """Mess-, NK- und Basisaufgabe; FF-Arme (langsamer) zuerst, dann nach Dauer."""
    auf = [_aufgabe("mess", a, s, hp[a]["lr"], P["schritte"], ctx) for a in ARME for s in P["seeds"]]
    auf += [_aufgabe("nk", a, s, hp[a]["lr"], P["nk_schritte"], ctx) for a in NK_ARME for s in P["nk_seeds"]]
    gewicht = {"FF": 1.7, "FF_gelu": 1.7, "L16": 1.0, "L_budget": 1.0}
    auf.sort(key=lambda a: -(a["schritte"] * gewicht[a["arm"]]))
    return [_basis_aufgabe(**{k: ctx[k] for k in ("K", "klein", "geraet", "daten", "laenge")}, ckpt=None)] + auf


def tuning(P, ctx, pfad, kennung, prozesse, klein):
    """Lernrate je Arm auf dem Tuning-Seed bei tuning_schritte; alle Raten aller Arme einer Runde laufen gemeinsam parallel; jede Aufgabe einzeln wiederaufnehmbar."""
    def aufgaben_fuer(arm, lr):                                  # gittersuche_stabil: Seed an Stelle 1
        return [(arm, P["tuning_seed"], lr)]

    def auswerten(aufg):
        ts = [_aufgabe("tun", arm, seed, lr, P["tuning_schritte"], ctx) for arm, seed, lr in aufg]
        res = _rechne(ts, pfad, kennung, prozesse=prozesse, klein=klein)
        return [res[_schl(a)] for a in ts]

    info, _ = gittersuche_stabil(list(ARME), aufgaben_fuer, auswerten, gitter=LR_GITTER, faktor=LR_FAKTOR, kollaps=KOLLAPS_SCORE)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


# ------------------------------------------------------------------ Kosten
def kosten_schaetzung(tokens_s_lora=13400.0, ff_faktor=0.6, gpus=4, P=None, tuning_raten=4.5, aufschlag=1.15, eval_s=90.0) -> dict:
    """GPU-Stunden und Wandzeit. tokens_s_lora: gemessen (gpu_test, 27.10.: 13.400 Tokens/s bei Batch 8 x 512). ff_faktor: ANNAHME (Durchsatz FF / LoRA,
    nicht gemessen; die Adapter-Elementweise-Rechnung in fp32 und das Neurechnen im Rueckwaertsschritt kosten Zeit; wird beim Rauchtest gemessen).
    LPT-Verteilung auf `gpus` Arbeiter, Tuning als Vorlauf in `runden` Wellen."""
    P = P or PROFILE["standard"]
    tok = P["bs"] * P["laenge"]
    rate = {"L16": tokens_s_lora, "L_budget": tokens_s_lora, "FF": tokens_s_lora * ff_faktor, "FF_gelu": tokens_s_lora * ff_faktor}
    sek = lambda arm, schritte: schritte * tok / rate[arm] * aufschlag + eval_s       # noqa: E731
    mess = [sek(a, P["schritte"]) for a in ARME for _ in P["seeds"]]
    nk = [sek(a, P["nk_schritte"]) for a in NK_ARME for _ in P["nk_seeds"]]
    tun = [sek(a, P["tuning_schritte"]) for a in ARME for _ in range(math.ceil(tuning_raten))]
    basis = [eval_s * 2]

    def lpt(ts):
        last = [0.0] * gpus
        for t in sorted(ts, reverse=True):
            i = min(range(gpus), key=last.__getitem__)
            last[i] += t
        return max(last)
    wand = lpt(tun) + lpt(mess + nk + basis)
    gesamt = sum(mess) + sum(nk) + sum(tun) + sum(basis)
    return {"gpu_stunden": gesamt / 3600, "wandzeit_h": wand / 3600, "gpus": gpus, "mess_gpu_h": sum(mess) / 3600, "nk_gpu_h": sum(nk) / 3600,
            "tuning_gpu_h": sum(tun) / 3600, "laengste_aufgabe_h": max(mess) / 3600, "ff_faktor": ff_faktor, "tokens_s_lora": tokens_s_lora,
            "tokens_je_lauf": P["schritte"] * tok}


# ------------------------------------------------------------------ Auswertung und Urteil
def auswertung(zeilen: list, basis: dict, params_lauf: dict, meta: dict) -> dict:
    """zeilen: Mess- und NK-Ergebniszeilen; basis: Zeile des unveraenderten Modells; params_lauf: gezaehlte trainierbare Parameter je Arm."""
    mess = [z for z in zeilen if z["teil"] == "mess"]
    seeds = sorted({z["seed"] for z in mess})
    wert = {a: {z["seed"]: z["test"] for z in mess if z["arm"] == a and z.get("endlich", True)} for a in ARME}
    diver = {a: sorted(z["seed"] for z in mess if z["arm"] == a and not z.get("endlich", True)) for a in ARME}
    # bpc aus Zeichen/Token des Test-Textes (beschreibend)
    tok_t = (meta.get("tokens") or {}).get("test")
    zch_t = (meta.get("zeichen") or {}).get("test")
    bpc_faktor = (tok_t / zch_t / math.log(2)) if tok_t and zch_t else None
    r = {"arme": {}, "kollaps": {}, "divergiert": diver, "negativ": {}, "diff": {}, "p_holm": {}, "parameter": params_lauf, "beschreibend": {},
         "basis": {"val": basis.get("val"), "test": basis.get("test")}, "bpc_faktor": bpc_faktor, "n_seeds": len(seeds)}

    def par(a, b):
        """Differenz b - a auf den Seeds, die in BEIDEN Armen endlich sind (positiv = a besser)."""
        gem = sorted(set(wert[a]) & set(wert[b]))
        return np.array([wert[b][s] - wert[a][s] for s in gem], dtype=float)

    for a in ARME:
        if wert[a]:
            w = np.array([wert[a][s] for s in sorted(wert[a])], dtype=float)
            r["arme"][a] = {"mittel": float(w.mean()), "sd": float(w.std(ddof=1)) if len(w) > 1 else float("nan"), "werte": [float(x) for x in w],
                            "seeds": sorted(wert[a]), "bpc": float(w.mean() * bpc_faktor) if bpc_faktor else None}
            med = float(np.median(w))
            r["kollaps"][a] = [s for s in sorted(wert[a]) if not (wert[a][s] <= med + KOLLAPS_VERLUST)]
    for h, (a, b) in (("H1", ("FF", "L_budget")), ("H2", ("FF", "FF_gelu"))):
        if wert[a] and wert[b]:
            r["diff"][h] = _stat(par(a, b))                                # positiv = FF besser
    for name, (a, b) in (("H3: L16 - FF (nats, positiv = FF besser)", ("FF", "L16")),
                         ("L_budget - L16 (nats, positiv = L16 besser)", ("L16", "L_budget")),
                         ("L_budget - FF_gelu (nats, positiv = FF_gelu besser)", ("FF_gelu", "L_budget"))):
        if wert[a] and wert[b]:
            r["beschreibend"][name] = _stat(par(a, b))
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1", "H2") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    nk = [z["test"] for z in zeilen if z["teil"] == "nk" and z.get("endlich", True)]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk), "erwartet_n": None} if nk else {}
    if basis.get("test") is not None and "L16" in r["arme"]:
        r["pk"] = {"verbesserung_L16": float(basis["test"] - r["arme"]["L16"]["mittel"]), "schwelle": PK_MIN}
    for a in ARME:
        if a in r["arme"] and basis.get("test") is not None:
            r["arme"][a]["verbesserung_gegen_basis"] = float(basis["test"] - r["arme"][a]["mittel"])
    r["tokens_s_mittel"] = {a: float(np.nanmean([z["tokens_s"] for z in mess if z["arm"] == a and "tokens_s" in z])) for a in ARME
                            if any(z["arm"] == a and "tokens_s" in z for z in mess)}
    r["speicher_gb_max"] = {a: float(max(z.get("max_speicher_gb", 0.0) for z in mess if z["arm"] == a)) for a in ARME if any(z["arm"] == a for z in mess)}
    return r


def _zahlen(r, h):
    d = r["diff"][h]
    return (f"d {d['mittel']:+.4f} nats, 90-%-KI [{d['ki'][0]:+.4f}; {d['ki'][1]:+.4f}], n {d['n']}, p_Holm {r['p_holm'][h]:.3g}").replace(".", ",")


def _nk(r) -> str:
    b = r["basis"].get("test")
    n = r["negativ"]
    return "BESTANDEN" if n and b is not None and n["mittel"] >= NK_FAKTOR * b else "DURCHGEFALLEN"


def _vorab(r, nk, h, arme):
    dv = {a: r["divergiert"][a] for a in arme if r["divergiert"].get(a)}
    if dv:
        return f"HU (Divergenz in {dv})"
    if h not in r["diff"] or "mittel" not in r["diff"][h] or h not in r["p_holm"]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    if "pk" not in r or r["pk"]["verbesserung_L16"] < PK_MIN:
        return f"HU (Positivkontrolle: L16 verbessert den Test-Verlust nicht um {PK_MIN} nats)"
    k = {a: r["kollaps"][a] for a in arme if r["kollaps"].get(a)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def urteil(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert L_budget/FF: {r['negativ'].get('mittel', float('nan')):.3f} nats; Basis {r['basis'].get('test') or float('nan'):.3f}; Schwelle {NK_FAKTOR}x)".replace(".", ",")}
    for h, arme, gut, schlecht in (("H1", ("FF", "L_budget"), "STRUKTUR_BESSER", "L_budget sicher besser als FF"),
                                   ("H2", ("FF", "FF_gelu"), "BORN_TRAEGT", "FF_gelu sicher besser als FF")):
        v = _vorab(r, nk, h, arme)
        if h == "H1" and v is None:
            p = r["parameter"]
            if "FF" not in p or "L_budget" not in p or abs(p["FF"] / p["L_budget"] - 1) > BUDGET_TOL:
                v = f"HU (Budgetgleichheit verletzt: {p.get('FF')} gegen {p.get('L_budget')} Parameter, Toleranz {BUDGET_TOL:.0%})"
        if v:
            u[h] = v
            continue
        z = _zahlen(r, h)
        if r["p_holm"][h] < ALPHA:
            u[h] = f"{gut} ({z})"
        elif r["diff"][h]["p_kl"] < ALPHA / 2:
            u[h] = f"H0 ({schlecht}; {z})"
        else:
            u[h] = f"HU ({z})"
    g = lambda a: f"{r['arme'][a]['mittel']:.4f}".replace(".", ",") if a in r["arme"] else "n/a"        # noqa: E731
    k = "H3 (beschreibend)"
    d3 = r["beschreibend"].get("H3: L16 - FF (nats, positiv = FF besser)")
    u[k] = (f"FF gegen L16: d {d3['mittel']:+.4f} nats (positiv = FF besser), 90-%-KI [{d3['ki'][0]:+.4f}; {d3['ki'][1]:+.4f}], n {d3['n']}".replace(".", ",") if d3 and "mittel" in d3 else "n/a")
    u["B"] = ("beschreibend: Test-Verlust (nats/Token) basis " + (f"{r['basis']['test']:.4f}".replace(".", ",") if r["basis"].get("test") is not None else "n/a") + ", "
              + ", ".join(f"{a} {g(a)}" for a in ARME) + "; Rest in auswertung")
    return u


# ------------------------------------------------------------------ Lauf
def _herzschlag_thread(stop):
    from labor import laufstatus

    def f():
        while not stop.wait(60):
            try:
                laufstatus.herzschlag(KARTE)
            except Exception:
                pass
    t = threading.Thread(target=f, daemon=True)
    t.start()
    return t


def lauf(schnell=False, prozesse=None, zwischen=None, profil=None, klein=False, **_):
    """Vollstaendiger Lauf (Tuning, Messung, Negativkontrolle, Basis, Auswertung, Urteil). `schnell`: Rauchtest NUR auf dem Tuning-Seed mit wenigen
    Schritten (echte Daten und echtes Modell: prueft die GPU-Pipeline und misst Tokens/s). `klein`: nur Vortest (winziges Zufallsmodell, CPU)."""
    profil = "klein" if klein else (profil or os.environ.get("I043_PROFIL", "standard"))
    P = dict(PROFILE[profil])
    if schnell:
        P.update(RAUCH)
        if not klein:
            P.update({"seeds": (TUNING_SEED,), "nk_seeds": (TUNING_SEED,)})
    elif not klein:
        if set(P["seeds"]) & _BELEGT or set(P["nk_seeds"]) & _BELEGT or P["tuning_seed"] in P["seeds"] or P["tuning_seed"] in _BELEGT:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    geraet = "cuda" if (not klein and torch.cuda.is_available()) else "cpu"
    if klein:
        D = daten_klein(P["laenge"])
        ordner = None
    else:
        D = daten_laden(None, P["laenge"])
        ordner = D["ordner"]
        q = D["meta"].get("quelle", "")
        if (profil == "gsd") != q.startswith("gsd"):
            raise ValueError(f"Profil {profil} passt nicht zur Datenquelle {q}")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE, "version": 1, "profil": ("rauch-" if schnell else "") + profil, "profil_werte": _json_sicher(P),
               "lr_gitter": list(LR_GITTER), "daten": _json_sicher(daten_kennung(D["meta"])), "modell": "klein" if klein else MODELL_NAME}
    ctx = {"K": {k: P.get(k) for k in ("bs", "laenge", "warm", "ckpt_alle", "zwischen_evals", "eval_max")}, "klein": klein, "geraet": geraet, "daten": ordner,
           "laenge": P["laenge"], "ckordner": str(ckordner)}
    stop = threading.Event()
    if not klein:
        _herzschlag_thread(stop)
    t0 = time.time()
    try:
        zw = _lade_zwischen(pfad, kennung)
        if "tuning" not in zw:
            hp0 = tuning(P, ctx, pfad, kennung, prozesse, klein)
            zw = _lade_zwischen(pfad, kennung)
            zw["tuning"] = hp0
            _schreibe_zwischen(pfad, kennung, zw)
            print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
        hp = zw["tuning"]
        alle = _rechne(aufgaben_mess(hp, P, ctx), pfad, kennung, prozesse=prozesse, klein=klein)
    finally:
        stop.set()
    zeilen = [z for k, z in alle.items() if k.split("|")[0] in ("mess", "nk")]
    basis = alle["basis"]
    params_lauf = {}
    for z in zeilen:
        if z["teil"] == "mess" and z.get("trainierbar") is not None:
            params_lauf[z["arm"]] = z["trainierbar"]
    dims = DIMS_1B
    e = {"abteilung": "Struktur", "modul": "struktur_adapter_1b", "karte": KARTE, "version": 1, "profil": profil, "schnell": schnell, "hypothesen": HYPOTHESEN,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": P["tuning_seed"], "tuning": hp,
         "einstellungen": {"schritte": P["schritte"], "tuning_schritte": P["tuning_schritte"], "nk_schritte": P["nk_schritte"], "batch": P["bs"],
                           "laenge": P["laenge"], "tokens_je_lauf": P["schritte"] * P["bs"] * P["laenge"], "lr_gitter": list(LR_GITTER), "ff_k": FF_K,
                           "lora_r16": LORA_R16, "rang_budget": zweig_spez("L_budget", dims)["r"], "ff_skala": FF_SKALA, "gpus": len(gpu_liste()) if geraet == "cuda" else 0,
                           "daten": daten_kennung(D["meta"]), "modell": MODELL_NAME},
         "parameter_formel": {a: param_arm(a, dims) for a in ARME},
         "flops_je_token": {a: flops_je_token(a, dims) for a in ARME} | {"basis_modell": flops_basis_je_token(dims)},
         "zwischendatei": str(pfad), "zeilen": zeilen, "basis_zeile": basis, "sekunden": time.time() - t0,
         "gpu_stunden": sum(z.get("sekunden_training", 0.0) for z in alle.values() if isinstance(z, dict)) / 3600}
    e["auswertung"] = auswertung(zeilen, basis, params_lauf, D["meta"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Vortest
def _synth(rng, basis=2.60, off=None, sd=0.004, n=5, koll=None, nk=2.55, diver=None, params=None, pk=0.10, nk_n=4):
    """Synthetische Zeilen fuer die Urteilszweige. off: Offsets der Arme gegen L_budget (negativ = besser)."""
    ref = {"L16": basis - pk, "L_budget": basis - 0.08, "FF": basis - 0.08, "FF_gelu": basis - 0.08}
    ref.update({a: ref["L_budget"] + v for a, v in (off or {}).items()})
    zeilen = []
    for s in range(1, n + 1):
        gem = rng.normal(0, sd)
        for a in ARME:
            z = {"teil": "mess", "arm": a, "seed": s, "test": ref[a] + gem + rng.normal(0, sd), "endlich": True, "tokens_s": 1000.0}
            if koll and a in koll and s == 1:
                z["test"] += 5 * KOLLAPS_VERLUST
            if diver and a in diver and s == 1:
                z["endlich"] = False
            zeilen.append(z)
    zeilen += [{"teil": "nk", "arm": a, "seed": s, "test": nk, "endlich": True} for a in NK_ARME for s in range(1, nk_n // 2 + 1)]
    pa = {"L16": 2252800, "L_budget": 563200, "FF": 574464, "FF_gelu": 574464}
    pa.update(params or {})
    meta = {"tokens": {"test": 2_000_000}, "zeichen": {"test": 8_000_000}}
    return zeilen, {"val": basis, "test": basis}, pa, meta


def _u(rng, **kw):
    zeilen, b, pa, meta = _synth(rng, **kw)
    return urteil(auswertung(zeilen, b, pa, meta))


def _vortest_zaehlung() -> bool:
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    # echte Abmessungen: L16 muss die 2,25 Mio. des GPU-Tests treffen
    if param_lora(16) != 2_252_800:
        fehler(f"L16 {param_lora(16)} != 2.252.800 (GPU-Test 27.10.)")
        ok = False
    ff, lb = param_arm("FF"), param_arm("L_budget")
    if abs(ff / lb - 1) > BUDGET_TOL or param_arm("FF_gelu") != ff:
        fehler(f"Budget FF {ff} gegen L_budget {lb}")
        ok = False
    if zweig_spez("L_budget")["r"] != 4 or ff != 574_464 or lb != 563_200:
        fehler(f"unerwartete Budgetwerte r={zweig_spez('L_budget')['r']} ff={ff} lb={lb}")
        ok = False
    # nach Rang geprueft: kein anderer Rang liegt naeher
    if any(abs(param_lora(r) - ff) < abs(lb - ff) for r in range(1, 40)):
        fehler("rang_budget nicht minimal")
        ok = False
    print(f"  Zaehlung: L16 {param_lora(16):,}, L_budget r={zweig_spez('L_budget')['r']} {lb:,}, FF {ff:,} ({ff / lb - 1:+.2%}), "
          f"FLOPs/Token Adapter FF {flops_je_token('FF')['gesamt']:,} L_budget {flops_je_token('L_budget')['gesamt']:,} Basis {flops_basis_je_token():,}")
    return ok


def _vortest_bausteine() -> bool:
    from scipy.linalg import hadamard
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    gen = torch.Generator().manual_seed(0)
    # Born-Zweig gegen explizite Hadamard-Referenz (q-artig 16->16 und v-artig 16->4, z² und GELU)
    for di, do in ((16, 16), (16, 4), (64, 8)):
        for akt in ("z2", "gelu"):
            z = FFZweig(di, do, 2, akt, gen, skala=1.7)
            z.d3.data = torch.randn_like(z.d3)
            x = torch.randn(5, di)
            H1, H2 = torch.tensor(hadamard(di), dtype=torch.float32) / math.sqrt(di), torch.tensor(hadamard(do), dtype=torch.float32) / math.sqrt(do)
            ref = torch.zeros(5, do)
            for k in range(2):
                u = (x * z.d1[k]) @ H1.T
                u = u * u if akt == "z2" else F.gelu(u)
                u = u.reshape(5, di // do, do).sum(1) / math.sqrt(di // do)
                ref += 1.7 * ((u * z.d2[k]) @ H2.T) * z.d3[k]
            if not torch.allclose(z(x), ref, atol=1e-5, rtol=1e-4):
                fehler(f"FFZweig {di}->{do} {akt} weicht von der Hadamard-Referenz ab ({(z(x) - ref).abs().max():.2e})")
                ok = False
            # Rechnet auch mit Batch x Folge
            if z(torch.randn(2, 3, di)).shape != (2, 3, do):
                fehler("Form")
                ok = False
    # Start = 0 und Gradientenfluss
    for spez in (LoraZweig(16, 8, 4, gen), FFZweig(16, 8, 2, "z2", gen)):
        x = torch.randn(6, 16)
        if spez(x).abs().max() != 0:
            fehler(f"{type(spez).__name__}: Start nicht 0")
            ok = False
    z = FFZweig(16, 8, 2, "z2", gen)
    (z(torch.randn(6, 16)) * torch.randn(6, 8)).sum().backward()
    if z.d3.grad.abs().sum() == 0 or z.d1.grad.abs().sum() != 0 or z.d2.grad.abs().sum() != 0:
        fehler("Start: Gradient darf nur in D3 fliessen")
        ok = False
    z.d3.data = torch.randn_like(z.d3)
    for p in z.parameters():
        p.grad = None
    (z(torch.randn(6, 16)) * torch.randn(6, 8)).sum().backward()
    if min(z.d1.grad.abs().sum(), z.d2.grad.abs().sum(), z.d3.grad.abs().sum()) == 0:
        fehler("nach D3 != 0 fliesst nicht in alle Diagonalen Gradient")
        ok = False
    lz = LoraZweig(16, 8, 4, gen)
    (lz(torch.randn(6, 16)) * 1.0).sum().backward()
    if lz.B.grad.abs().sum() == 0 or lz.A.grad.abs().sum() != 0:
        fehler("LoRA-Start: Gradient nur in B erwartet")
        ok = False
    # unzulaessige Abmessungen werden abgelehnt
    try:
        FFZweig(48, 16, 1, "z2", gen)
        fehler("Nicht-Zweierpotenz nicht abgelehnt")
        ok = False
    except ValueError:
        pass
    # Matmul-FLOPs der Formel gleich der FlopCounter-Zaehlung (unabhaengig)
    from torch.utils.flop_counter import FlopCounterMode
    z = FFZweig(64, 8, 3, "z2", gen)
    with FlopCounterMode(display=False) as fc:
        z(torch.randn(10, 64))
    a1, b1, a2, b2 = _faktoren(64) + _faktoren(8)
    erw = 10 * 3 * (2 * 64 * (a1 + b1) + 2 * 8 * (a2 + b2))
    if fc.get_total_flops() != erw:
        fehler(f"FLOP-Formel {erw} != gezaehlt {fc.get_total_flops()}")
        ok = False
    return ok


def _tiny_arm_modelle(D):
    """Hilfsfunktion: Basismodell und je ein Arm-Modell (CPU, fp32)."""
    out = {}
    for arm in ARME:
        out[arm] = adapter_einbauen(fabrik_klein(D), arm, 5)
    return out


def _vortest_modell() -> bool:
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    D = daten_klein()
    basis = fabrik_klein(D)
    dims = _dims_von(basis)
    x = torch.from_numpy(D["val"][:4].astype(np.int64))
    with torch.no_grad():
        lg0 = basis(input_ids=x).logits
    for arm in ARME:
        m = adapter_einbauen(fabrik_klein(D), arm, 5)
        sp = zweig_spez(arm, dims)
        soll = param_lora(sp["r"], dims) if sp["art"] == "lora" else param_ff(sp["k"], dims)
        if trainierbar(m) != soll:
            fehler(f"{arm}: gezaehlt {trainierbar(m)} != Formel {soll}")
            ok = False
        n_ad = sum(isinstance(mm, AdapterLinear) for mm in m.modules())
        if n_ad != 2 * dims["lagen"]:
            fehler(f"{arm}: {n_ad} Adapter statt {2 * dims['lagen']}")
            ok = False
        basis_grad = [n for n, p in m.named_parameters() if p.requires_grad and ".zweig." not in n]
        if basis_grad:
            fehler(f"{arm}: Basisgewichte nicht eingefroren: {basis_grad[:3]}")
            ok = False
        with torch.no_grad():
            if (m(input_ids=x).logits - lg0).abs().max() > 1e-6:
                fehler(f"{arm}: Start-Logits != Basis")
                ok = False
        # gleiche Seed -> gleiche Initialisierung, anderer Seed -> andere
        a1 = adapter_einbauen(fabrik_klein(D), arm, 5)
        a2 = adapter_einbauen(fabrik_klein(D), arm, 6)
        p1 = [p for p in a1.parameters() if p.requires_grad]
        p0 = [p for p in m.parameters() if p.requires_grad]
        p2 = [p for p in a2.parameters() if p.requires_grad]
        if not all(torch.equal(u, v) for u, v in zip(p0, p1)) or all(torch.equal(u, v) for u, v in zip(p0, p2)):
            fehler(f"{arm}: Seed-Reproduzierbarkeit der Initialisierung")
            ok = False
    # Checkpointing des Adapterzweigs aendert Verlust und Gradienten nicht
    m = adapter_einbauen(fabrik_klein(D), "FF", 5, ckpt_adapter=False)
    m2 = adapter_einbauen(fabrik_klein(D), "FF", 5, ckpt_adapter=True)
    for mm in (m, m2):
        for p in mm.parameters():
            if p.requires_grad:
                p.data.add_(0.1 * torch.randn(p.shape, generator=torch.Generator().manual_seed(int(p.numel()))))
        mm.train()
    g = []
    for mm in (m, m2):
        mm(input_ids=x, labels=x).loss.backward()
        g.append(torch.cat([p.grad.flatten() for p in mm.parameters() if p.requires_grad]))
    if not torch.allclose(g[0], g[1], atol=1e-6, rtol=1e-4):
        fehler("Neurechnen des Adapterzweigs aendert den Gradienten")
        ok = False
    return ok


def _vortest_training() -> bool:
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    import tempfile
    D = daten_klein()
    K = {k: PROFILE["klein"][k] for k in ("bs", "laenge", "warm", "ckpt_alle", "zwischen_evals")}
    fab = lambda: fabrik_klein(D)                               # noqa: E731
    base = bewerte(fab(), D["val"])
    print(f"  Basis-Val {base:.3f}")
    erg = {}
    for arm, lr in (("L16", 3e-3), ("L_budget", 3e-3), ("FF", 1e-2), ("FF_gelu", 1e-2)):
        r = trainiere(arm, lr, 901, 120, D, K, fabrik=fab, mit_test=True)
        erg[arm] = r
        if not r["endlich"] or not r["val"] < base - 0.05:
            fehler(f"{arm} lernt nicht: val {r.get('val')} gegen Basis {base}")
            ok = False
    print("  Val nach 120 Schritten: " + ", ".join(f"{a} {erg[a]['val']:.3f}" for a in erg))
    # Wiederaufnahme: Abbruch mitten in der Aufgabe, danach identisches Ergebnis; fremder Checkpoint wird ignoriert
    with tempfile.TemporaryDirectory() as td:
        pfad = str(Path(td) / "c.pt")
        ref = trainiere("FF", 1e-2, 902, 60, D, K, fabrik=fab, mit_test=True)
        try:
            trainiere("FF", 1e-2, 902, 60, D, K, fabrik=fab, ckpt=pfad, _stopp_bei=40)
            fehler("simulierter Abbruch blieb aus")
            ok = False
        except _Abbruch:
            pass
        if not os.path.exists(pfad):
            fehler("kein Checkpoint nach Abbruch")
            ok = False
        fort = trainiere("FF", 1e-2, 902, 60, D, K, fabrik=fab, ckpt=pfad, mit_test=True)
        if fort["fortgesetzt_ab"] != 25 or abs(fort["test"] - ref["test"]) > 1e-6 or abs(fort["val"] - ref["val"]) > 1e-6:
            fehler(f"Fortsetzung != ungestoert: ab {fort['fortgesetzt_ab']}, test {fort['test']} gegen {ref['test']}")
            ok = False
        if os.path.exists(pfad):
            fehler("Checkpoint nach Ende nicht geloescht")
            ok = False
        try:
            trainiere("FF", 1e-2, 902, 60, D, K, fabrik=fab, ckpt=pfad, _stopp_bei=40)
        except _Abbruch:
            pass
        fremd = trainiere("FF", 3e-3, 902, 60, D, K, fabrik=fab, ckpt=pfad)         # andere Rate -> Checkpoint passt nicht
        if fremd["fortgesetzt_ab"] != 0:
            fehler("fremder Checkpoint wurde verwendet")
            ok = False
    # Datenreihenfolge: Funktion von Seed und Schritt, gleiche Reihenfolge fuer alle Arme, verschiedene Seeds verschieden, ohne Zuruecklegen
    o1, o2, o3 = reihenfolge(7, 100, 10, 8), reihenfolge(7, 100, 10, 8), reihenfolge(8, 100, 10, 8)
    if not np.array_equal(o1, o2) or np.array_equal(o1, o3) or len(set(o1.flatten())) != 80:
        fehler("Datenreihenfolge")
        ok = False
    x = torch.arange(2 * 32).reshape(2, 32)
    px = permutiere_fenster(x, 3, 4)
    if torch.equal(px, x) or not torch.equal(px.sort(1).values, x.sort(1).values) or not torch.equal(px, permutiere_fenster(x, 3, 4)):
        fehler("Permutation der Negativkontrolle")
        ok = False
    # Negativkontrolle: Training auf permutierten Tokens verbessert den echten Verlust nicht wie echtes Training
    nkr = trainiere("FF", 1e-2, 901, 120, D, K, fabrik=fab, permutiert=True, mit_test=True)
    if not (erg["FF"]["test"] < nkr["test"] - 0.05):
        fehler(f"NK trennt nicht: echt {erg['FF']['test']:.3f} gegen permutiert {nkr['test']:.3f}")
        ok = False
    # nicht endlicher Verlust -> endlich False
    r = trainiere("FF", 1e38, 901, 30, D, K, fabrik=fab)
    if r["endlich"]:
        fehler("Divergenz nicht erkannt")
        ok = False
    return ok


def _vortest_urteile() -> bool:
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    rng = np.random.default_rng(0)

    def erwarte(name, u, h, marke):
        nonlocal ok
        if not str(u[h]).startswith(marke):
            fehler(f"{name}: {h} = {u[h]!r}, erwartet {marke}")
            ok = False

    u = _u(rng, off={"FF": -0.05})
    erwarte("FF besser", u, "H1", "STRUKTUR_BESSER")
    u = _u(rng, off={"FF": +0.05})
    erwarte("FF schlechter", u, "H1", "H0")
    u = _u(rng, off={"FF": 0.0}, sd=0.02)
    erwarte("gleich", u, "H1", "HU")
    u = _u(rng, off={"FF": -0.05, "FF_gelu": +0.02})
    erwarte("Born traegt", u, "H2", "BORN_TRAEGT")
    u = _u(rng, off={"FF": 0.0, "FF_gelu": -0.05})
    erwarte("Born schadet", u, "H2", "H0")
    u = _u(rng, off={"FF": 0.0, "FF_gelu": 0.0}, sd=0.02)
    erwarte("Born unentschieden", u, "H2", "HU")
    # Holm-Fall: H2 knapp (p zwischen 0,025 und 0,05 einzeln) wird nach Holm HU, wenn H1 klein
    u = _u(rng, off={"FF": -0.05, "FF_gelu": +0.0}, sd=0.004)
    if u["H1"][:15] != "STRUKTUR_BESSER":
        fehler("Holm-Fall: H1")
        ok = False
    # Kontrollen
    erwarte("NK durchgefallen", _u(rng, off={"FF": -0.05}, nk=2.0), "H1", "HU (Negativkontrolle")
    erwarte("NK durchgefallen", _u(rng, off={"FF": -0.05}, nk=2.0), "H2", "HU (Negativkontrolle")
    erwarte("PK durchgefallen", _u(rng, off={"FF": -0.05}, pk=0.0), "H1", "HU (Positivkontrolle")
    erwarte("Divergenz", _u(rng, off={"FF": -0.05}, diver=("FF",)), "H1", "HU (Divergenz")
    erwarte("Divergenz fremder Arm", _u(rng, off={"FF": -0.05}, diver=("L16",)), "H1", "STRUKTUR_BESSER")
    erwarte("Kollaps", _u(rng, off={"FF": -0.05}, koll=("L_budget",)), "H1", "HU (Kollaps")
    erwarte("Budget verletzt", _u(rng, off={"FF": -0.05}, params={"FF": 700000}), "H1", "HU (Budgetgleichheit")
    erwarte("Budget Rand ok", _u(rng, off={"FF": -0.05}, params={"FF": int(563200 * 1.049)}), "H1", "STRUKTUR_BESSER")
    zeilen, b, pa, meta = _synth(rng, n=1)
    erwarte("n=1", urteil(auswertung(zeilen, b, pa, meta)), "H1", "HU (Arme fehlen")
    zeilen, b, pa, meta = _synth(rng)
    zeilen = [z for z in zeilen if z["arm"] != "FF_gelu"]
    erwarte("Arm fehlt", urteil(auswertung(zeilen, b, pa, meta)), "H2", "HU (Arme fehlen")
    u = _u(rng, off={"FF": -0.05})
    if "H3 (beschreibend)" not in u or not u["H3 (beschreibend)"].startswith("FF gegen L16"):
        fehler("H3 beschreibend fehlt")
        ok = False
    # Mutationen: vertauschte Richtung wuerde FF-besser als H0 melden
    zeilen, b, pa, meta = _synth(rng, off={"FF": -0.05})
    r = auswertung(zeilen, b, pa, meta)
    if not r["diff"]["H1"]["mittel"] > 0:
        fehler("Vorzeichenkonvention (positiv = FF besser)")
        ok = False
    return ok


def _vortest_kosten_und_seeds() -> bool:
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    k = kosten_schaetzung()
    print(f"  Kosten (13.400 Tokens/s LoRA, FF 60 %, 4 GPUs, +15 %): {k['gpu_stunden']:.1f} GPU-h, Wandzeit {k['wandzeit_h']:.1f} h, laengste Aufgabe {k['laengste_aufgabe_h']:.2f} h")
    if not (5 < k["gpu_stunden"] < 60 and k["wandzeit_h"] < 20 and k["laengste_aufgabe_h"] < 4):
        fehler(f"Kostenschaetzung unplausibel: {k}")
        ok = False
    if kosten_schaetzung(gpus=1)["wandzeit_h"] < kosten_schaetzung(gpus=4)["wandzeit_h"]:
        fehler("Wandzeit sinkt nicht mit mehr GPUs")
        ok = False
    if PROFILE["standard"]["schritte"] * PROFILE["standard"]["bs"] * PROFILE["standard"]["laenge"] != 24_576_000:
        fehler("Tokenbudget")
        ok = False
    # Seeds: frisch gegen Register (nur diese ID) und gegen alle Ergebnisdateien, Blockregeln
    reg = json.loads((WURZEL / "seeds.json").read_text(encoding="utf-8"))
    mess, tun = set(SEEDS), {TUNING_SEED}
    for e in reg["eintraege"]:
        menge = {s for a, b in e["seeds"] for s in range(a, b + 1)}
        if e["id"] != KARTE and (menge & (mess | tun)):
            fehler(f"Seeds {sorted(menge & (mess | tun))} auch in {e['id']}")
            ok = False
    belegt_ids = [e for e in reg["eintraege"] if e["id"] == KARTE]
    if not any(e["zweck"] == "messung" and {s for a, b in e["seeds"] for s in range(a, b + 1)} >= mess for e in belegt_ids) or \
            not any(e["zweck"] == "tuning" and TUNING_SEED in {s for a, b in e["seeds"] for s in range(a, b + 1)} for e in belegt_ids):
        fehler("Seeds nicht im Register eingetragen")
        ok = False
    if any(s % 100 >= 90 for s in SEEDS) or TUNING_SEED % 100 < 90 or (set(SEEDS) & _BELEGT) or TUNING_SEED in _BELEGT:
        fehler("Seed-Blockregeln")
        ok = False
    return ok


def _vortest_lauf_klein() -> bool:
    """Ganze Pipeline auf CPU mit dem winzigen Modell: Tuning (Rundenlogik), Messung, NK, Basis, Auswertung, Urteil; danach Wiederaufnahme aus der Zwischendatei
    und ein Lauf ueber den Pool (2 Arbeiter, spawn)."""
    import tempfile
    ok = True
    fehler = lambda t: print("  FEHLER:", t)                    # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        pfad = Path(td) / "z.json"
        t0 = time.time()
        e = lauf(klein=True, schnell=False, prozesse=1, zwischen=pfad)
        a = e["auswertung"]
        print(f"  Lauf klein (1 Prozess): {time.time() - t0:.0f} s; Tuning {({k: v['lr'] for k, v in e['tuning'].items()})}; Urteile {list(e['urteil'])}")
        if set(e["tuning"]) != set(ARME) or any(v["lr"] not in [g for g in v["gitter"]] and not isinstance(v["lr"], float) for v in e["tuning"].values()):
            fehler("Tuning unvollstaendig")
            ok = False
        if len([z for z in e["zeilen"] if z["teil"] == "mess"]) != 4 * 3 or len([z for z in e["zeilen"] if z["teil"] == "nk"]) != 4:
            fehler("Zahl der Mess-/NK-Aufgaben")
            ok = False
        if a["basis"]["test"] is None or any(x not in a["arme"] for x in ARME):
            fehler("Auswertung unvollstaendig")
            ok = False
        if not all(z["trainierbar"] in (param_lora(16, _dims_klein()), param_lora(zweig_spez("L_budget", _dims_klein())["r"], _dims_klein()),
                                        param_ff(FF_K, _dims_klein())) for z in e["zeilen"] if z["teil"] == "mess"):
            fehler("gezaehlte Parameter im Lauf != Formel")
            ok = False
        if not all(k in e["urteil"] for k in ("H1", "H2", "NK", "H3 (beschreibend)")):
            fehler("Urteil unvollstaendig")
            ok = False
        json.dumps(e)                                          # speicherbar
        # Wiederaufnahme: alles steht in der Zwischendatei -> nichts wird neu gerechnet
        t1 = time.time()
        e2 = lauf(klein=True, schnell=False, prozesse=1, zwischen=pfad)
        if time.time() - t1 > 0.5 * (t1 - t0) or e2["auswertung"]["arme"] != a["arme"]:
            fehler("Wiederaufnahme rechnet neu oder ergibt anderes Ergebnis")
            ok = False
        # fremde Zwischendatei (anderes Profil) -> Abbruch statt Mischen
        try:
            lauf(klein=True, schnell=True, prozesse=1, zwischen=pfad)
            fehler("fremde Zwischendatei nicht abgelehnt")
            ok = False
        except ValueError:
            pass
        # Pool mit 2 Arbeitern (spawn): einzelne Aufgaben
        pfad2 = Path(td) / "z2.json"
        K = {k: PROFILE["klein"][k] for k in ("bs", "laenge", "warm", "ckpt_alle", "zwischen_evals")}
        ctx = {"K": K, "klein": True, "geraet": "cpu", "daten": None, "laenge": 32, "ckordner": str(Path(td) / "ck")}
        auf = [_basis_aufgabe(**{k: ctx[k] for k in ("K", "klein", "geraet", "daten", "laenge")}, ckpt=None),
               _aufgabe("mess", "FF", 901, 1e-2, 30, ctx), _aufgabe("mess", "L_budget", 901, 3e-3, 30, ctx)]
        kenn = {"karte": KARTE, "pooltest": 1}
        res = _rechne(auf, pfad2, kenn, prozesse=2, klein=True)
        if set(res) != {_schl(x) for x in auf} or not all(z.get("endlich") for z in res.values()):
            fehler("Pool-Ergebnis")
            ok = False
        ref = trainiere("FF", 1e-2, 901, 30, daten_klein(), K, fabrik=lambda: fabrik_klein(daten_klein()), mit_test=True)
        if abs(res[_schl(auf[1])]["test"] - ref["test"]) > 1e-5:
            fehler("Pool-Arbeiter ergibt anderes Ergebnis als der Hauptprozess")
            ok = False
    return ok


def _dims_klein():
    return {"lagen": 2, "hidden": 64, "kv": 32, "heads": 4, "kv_heads": 2, "inter": 128, "vocab": 64}


def vortest() -> bool:
    """Vortest (CPU, winziges Llama): Zaehlung und Budget, Bausteine gegen Hadamard-Referenz, Modell-Einbau, Training/Wiederaufnahme/Negativkontrolle,
    alle Urteilszweige, Kosten, Seeds, ganze Pipeline. Braucht `transformers`; fehlt es, wird (wie bei fehlenden Daten) uebersprungen."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("transformers fehlt (pip install transformers) -- Download gesperrt/nicht installiert, Vortest uebersprungen")
        return False
    torch.set_num_threads(1)
    ok = True
    for name, f in (("Zaehlung", _vortest_zaehlung), ("Bausteine", _vortest_bausteine), ("Modell", _vortest_modell), ("Training", _vortest_training),
                    ("Urteile", _vortest_urteile), ("Kosten/Seeds", _vortest_kosten_und_seeds), ("Pipeline", _vortest_lauf_klein)):
        t0 = time.time()
        g = f()
        print(f"  Vortest {name}: {'ok' if g else 'DURCHGEFALLEN'} ({time.time() - t0:.0f} s)")
        ok = ok and g
    return ok


if __name__ == "__main__":      # Pflicht: spawn-Kindprozesse importieren dieses Skript erneut
    print("VORTEST", "bestanden" if vortest() else "DURCHGEFALLEN")
