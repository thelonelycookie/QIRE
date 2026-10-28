"""Abteilung Struktur -- I-045 „Strukturschicht im kleinen Sprachmodell (Container)“.
(Folge von I-042/I-044: Struktur ersetzt dichte Schichten dort, wo die Eingabe schon gelernte Merkmale hat (CNN-Kopf, +1,8 pp, 85x weniger Parameter).
 Im Transformer liegt die FFN hinter der Attention. Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

FRAGE: Ersetzt eine Fastfood-Strukturschicht (trainierte Diagonalen, O(N)) die dichten FFN-Matrizen eines kleinen Decoder-Transformers?
DATEN: Deutscher Text auf Byte-Ebene (UTF-8, Vokabular 256). Quelle: Universal-Dependencies-Treebank UD_German-GSD (Sätze der Zeilen „# text =“,
  Lizenz CC BY-SA 4.0, ca. 22 MB Download, nur nötig, wenn INSTITUT/daten/de_gsd_*.conllu fehlt; im Repo selbst liegt kein deutscher Text).
  FESTER Split = der mitgelieferte der Treebank: train / dev (= Validierung, nur Tuning) / test (nur am Ende). Sätze durch '\\n' getrennt.
MODELL: Decoder-Transformer, Pre-LayerNorm, 4 Lagen, d=256, 4 Köpfe, Kontext 128, gelernte Positionen, Ausgabematrix = Einbettung (gebunden).
ARME (nur die FFN-Matrizen unterscheiden sich, Attention/Einbettung identisch, alle FFN-Matrizen mit Bias):
  D   dicht 256->1024->256, GELU                         (Referenz)
  S   Fastfood 256->1024 (4 Blöcke) und 1024->256 (1 Block, 256 von 1024 Ausgängen), je H D2 H D1 mit trainierten Diagonalen, GELU
  Sb  wie S, aber Breite 4096 (16 Blöcke / 4096->256): „breit“, weiterhin weniger FFN-Parameter als D
  L   LoRA-artig: je Matrix Rang r = ceil(Parameter(S-Matrizen) / (din+dout)) = 2 (5.120 statt 4.096 Gewichte, also eher GROSSZÜGIG zu L), GELU
  Sz  Nebenarm: S mit Nichtlinearität z² statt GELU (nur beschreibend, nicht in der Holm-Familie)
TRAINING: Adam (kein Weight Decay), Batch 8 x 128 Tokens (1.024 Tokens/Schritt; groessere Batches fallen bei Fastfood-Sb aus dem Cache, 4x langsamer), 5 % Aufwärmen, Kosinus-Abfall, Gradientenbegrenzung 1,0, Schrittzahl vorab fest (PROFILE),
  Lernrate je Arm auf Tuning-Seed 2499 auf der Validierung (gittersuche_stabil, Gitter in beide Richtungen verlängerbar). Seeds 2400-2404, Test nur am Ende.
MESSUNG: Test-Verlust in bits/Zeichen (bpc, alle Positionen, nicht überlappende 128er-Fenster); FFN-Parameter und Gesamtparameter; Zeit/Token der
  Inferenz (1 Thread, Batch 8 x 128, Median 7 Wdh., je Seed im Arbeiter gemessen = gepaart; zusätzlich saubere Einzelmessung im Einzelprozess am Ende).
HYPOTHESEN (Holm über H1-H3, gepaarter t-Test n=5, einseitig, Differenzen je Seed):
  H1 Sb <= D + 0,02 bpc (Nichtunterlegenheit) UND weniger FFN-Parameter als D: NICHT_UNTERLEGEN | H0 | HU.
  H2 S < L (kleinerer bpc bei gleichem Budget): S_BESSER | H0 (L sicher besser) | HU.
  H3 Sb schneller als D (Inferenzzeit je Token): SCHNELLER | H0 (sicher langsamer) | HU. (beschreibend: Verhältnis; gepaart: log-Verhältnis je Seed)
NEGATIVKONTROLLE: D und Sb auf Trainingstext mit permutierten Bytes (Zufallsreihenfolge; Kontext trägt nichts), Test auf echtem Test. Bestanden, wenn der
  mittlere Test-bpc >= 0,9 x Unigramm-Entropie des Test-Textes (gemessen mit Trainingshäufigkeiten) -- ein Leck (Zukunftsblick) würde sie unterschreiten.
"""
from __future__ import annotations

import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abteilungen.clifford_suche import holm
from abteilungen.struktur_gross import (Fastfood, _arr, _f, _lade_zwischen, _rechne, _schreibe_zwischen, _stoppuhr, _t_p, pad2)
from abteilungen.walsh_kontrolle import wilcoxon_schnell
from labor.gemeinsam import WURZEL, ERGEBNISSE, gittersuche_stabil, parallel, zusammenfassung

KARTE = "I-045"
SEEDS = tuple(range(2400, 2405))
TUNING_SEED = 2499
_BELEGT = (set(range(2000, 2070)) | set(range(2090, 2120)) | set(range(2150, 2160)) | set(range(900, 1001)) | set(range(2200, 2205))
           | set(range(2300, 2305)) | {2299, 2399})
ARME = ("D", "S", "Sb", "L", "Sz")
NK_ARME = ("D", "Sb")
NK_SEEDS = (2400, 2401)
D_MODELL, LAGEN, KOEPFE, KONTEXT, BATCH = 256, 4, 4, 128, 8
BREITE = {"D": 1024, "S": 1024, "Sb": 4096, "L": 1024, "Sz": 1024}
LR_GITTER = (5e-4, 1.5e-3, 4.5e-3)
LR_FAKTOR = 3
KOLLAPS_SCORE = 0.01            # Tuning: Score = 1 - bpc/8, also 0,08 bpc
KOLLAPS_BPC = 0.2               # Messung: Seed schlechter als Median + 0,2 bpc (in einem beteiligten Arm) -> HU
ALPHA = 0.05
NI_GRENZE = 0.02                # H1: 0,02 bpc
NK_FAKTOR = 0.9
VAL_BYTES = 65536
ZEIT_BATCH = 8
QUELLE = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-{}.conllu"
DATEIEN = WURZEL / "daten"

# ---- Laufzeitprotokoll (VOR dem Lauf festgelegt, Messung siehe LABORBUCH) ----
PROFILE = {
    "standard": {"schritte": 1000, "nk_schritte": 300},
    "fallback": {"schritte": 600, "nk_schritte": 200},
}

HYPOTHESEN = {
    "H1": "Sb <= D + 0,02 bpc (Nichtunterlegenheit, einseitig gepaart t, Holm H1-H3) UND weniger FFN-Parameter als D. NICHT_UNTERLEGEN | H0 (Sb sicher um mehr als 0,02 bpc schlechter) | HU.",
    "H2": "S < L bei gleichem Budget (kleiner bpc; einseitig gepaart t, Holm). S_BESSER | H0 (L sicher besser) | HU.",
    "H3": "Sb schneller als D (Inferenzzeit je Token, 1 Thread, gepaart log-Verhaeltnis je Seed, einseitig, Holm). SCHNELLER | H0 (sicher langsamer) | HU.",
    "NK": f"Negativkontrolle (D, Sb mit permutierten Trainingsbytes): mittlerer Test-bpc >= {NK_FAKTOR} x Unigramm-Entropie des Tests. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc je Arm, Parameter, Zeiten, Lernraten, Sz gegen S, Sb gegen D, S gegen D, Trainingszeit.",
}
FOLGEFRAGEN = {
    "H1 NICHT_UNTERLEGEN": "Struktur-FFN traegt im Sprachmodell: groesseres Modell/laengeres Training, dann Parameter- und Zeitvergleich bei gleicher Qualitaet.",
    "H1 H0": "FFN im Transformer braucht dichte Matrizen: Struktur nur in Randschichten (Einbettung/Kopf) oder als Zusatz testen.",
    "H3 SCHNELLER": "Zeitgewinn schon bei d=256: Skalierung auf d=1024 messen.",
    "H3 H0": "Fastfood ist bei kleinem d auf der CPU langsamer als BLAS: Gewinn nur in Parametern, nicht in Zeit; Zeit erst bei grossem d erwartbar.",
}


# ------------------------------------------------------------------ Daten
def _text_laden(teil: str) -> bytes:
    pfad = DATEIEN / f"de_gsd_{teil}.conllu"
    if not pfad.exists():
        DATEIEN.mkdir(exist_ok=True)
        urllib.request.urlretrieve(QUELLE.format(teil), pfad)
    sätze = [z[9:].rstrip("\n") for z in pfad.read_text(encoding="utf-8").splitlines(True) if z.startswith("# text = ")]
    return ("\n".join(sätze) + "\n").encode("utf-8")


_DATEN = {}


def daten(klein=False):
    """(train, val, test) als uint8-Tensoren (Bytes). klein = Rauchtest. Je Prozess gecacht."""
    if klein not in _DATEN:
        if "roh" not in _DATEN:
            _DATEN["roh"] = tuple(torch.from_numpy(np.frombuffer(_text_laden(t), np.uint8).copy()) for t in ("train", "dev", "test"))
        tr, va, te = _DATEN["roh"]
        _DATEN[klein] = (tr[:100_000], va[:8192], te[:8192]) if klein else (tr, va, te)
    return _DATEN[klein]


def unigramm_entropie(train, test) -> float:
    """Kreuzentropie (bits) des Tests unter den Trainings-Haeufigkeiten (Laplace-Glaettung)."""
    p = torch.bincount(train.long(), minlength=256).double() + 1
    p = p / p.sum()
    return float(-(torch.log2(p[test.long()])).mean())


# ------------------------------------------------------------------ Bausteine
class FFNDicht(nn.Module):
    def __init__(self, d, F_, akt=F.gelu):
        super().__init__()
        self.a, self.b = nn.Linear(d, F_), nn.Linear(F_, d)
        self.akt = akt

    def forward(self, x):
        return self.b(self.akt(self.a(x)))


class Matrix(nn.Module):
    """Eine FFN-Matrix: Fastfood (trainierte Diagonalen) oder niedriger Rang, jeweils mit Bias."""

    def __init__(self, din, dout, art, gen, r=2):
        super().__init__()
        if art == "ff":
            self.m = Fastfood(din, dout, True, gen)
        else:
            self.m = nn.Sequential(nn.Linear(din, r, bias=False), nn.Linear(r, dout, bias=False))
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):
        s = x.shape
        z = self.m(x.reshape(-1, s[-1]))
        return z.reshape(*s[:-1], -1) + self.bias


def rang_L(d, F_):
    """Rang je Matrix: ceil(Gewichte der S-Matrizen / (din+dout)) -- 2 bei 256/1024."""
    gew = 2 * max(1, F_ // pad2(d)) * pad2(d)
    return max(1, math.ceil(gew / (d + F_)))


class FFNStruktur(nn.Module):
    def __init__(self, d, F_, art, akt, gen, r=None):
        super().__init__()
        r = r or rang_L(d, 1024)
        self.a, self.b = Matrix(d, F_, art, gen, r), Matrix(F_, d, art, gen, r)
        self.akt = akt

    def forward(self, x):
        return self.b(self.akt(self.a(x)))


def _relu2(x):
    """squared ReLU (Primer): relu(x)^2."""
    return torch.square(F.relu(x))


# I-045-v2: Arm -> (Struktur, Aktivierung); Struktur "dicht" | "ff" (Fastfood) | "lr" (niedriger Rang). Alle Breite 1024.
ARME_V2 = {"D_gelu": ("dicht", F.gelu), "D_z2": ("dicht", torch.square), "Sz": ("ff", torch.square),
           "S_gelu": ("ff", F.gelu), "L_z2": ("lr", torch.square), "D_relu2": ("dicht", _relu2)}
BREITE.update({a: 1024 for a in ARME_V2})


# I-045-v4: budgetgleiche FFN. Struktur "dicht" | "lr" (niedriger Rang); Budget "param" (Parameterzahl wie Sz) | "flops" (FFN-FLOPs/Token wie Sz).
ARME_V4 = {"Dk_z2": ("dicht", torch.square, "param"), "Df_z2": ("dicht", torch.square, "flops"),
           "Df_gelu": ("dicht", F.gelu, "flops"), "Lk_z2": ("lr", torch.square, "flops")}


def ffn_param_lage(art, d, h, r=2) -> int:
    """FFN-Parameter je Lage (mit Bias): dicht 2·d·h + h + d; niedriger Rang 2·r·(d+h) + h + d."""
    return 2 * d * h + h + d if art == "dicht" else 2 * r * (d + h) + h + d


def ffn_flops_lage(art, d, h, r=2) -> int:
    """FFN-FLOPs je Token und Lage (Matmuls, 1 Multiplikation = 1 Addition = 1 Flop): dicht 4·d·h; niedriger Rang 4·r·(d+h)."""
    return 4 * d * h if art == "dicht" else 4 * r * (d + h)


def sz_param_lage(d) -> int:
    """Sz (Fastfood, Breite 4d) je Lage: 21·d (Diagonalen 16d, Bias 5d); in den Vortests gegen die echte Zaehlung geprueft."""
    return 21 * d


_SZ_FLOPS = {}


def sz_flops_lage(d) -> int:
    """FFN-FLOPs je Token und Lage von Sz (as implemented, aus der Modellstruktur gezaehlt)."""
    if d not in _SZ_FLOPS:
        _SZ_FLOPS[d] = _ffn_flops_modul(FFNStruktur(d, 4 * d, "ff", torch.square, torch.Generator().manual_seed(0)))
    return _SZ_FLOPS[d]


def _ffn_flops_modul(ffn) -> int:
    """FLOPs einer FFN je Token (Linear: 2·ein·aus; Fastfood: m Bloecke mit zwei Walsh-Transformationen als Matmuls plus Diagonalen)."""
    n = 0
    for m in ffn.modules():
        if isinstance(m, nn.Linear):
            n += 2 * m.in_features * m.out_features
        elif isinstance(m, Fastfood):
            n += m.m * (2 * (2 * m.D * (m.a + m.b)) + 2 * m.D)
    return int(n)


def breite_budget(art, d, r=2) -> int:
    """Ganzzahlige Breite h, deren FFN-Parameterzahl je Lage der von Sz am naechsten kommt (Budget "param")."""
    ziel = sz_param_lage(d)
    return min(range(1, 8 * d), key=lambda h: (abs(ffn_param_lage(art, d, h, r) - ziel), h))


def form_v4(arm, d):
    """(Struktur, Breite h, Rang r) eines v4-Arms. Dk: Breite aus dem Parameterbudget. Df: Breite aus dem FLOP-Budget. Lk: Breite 4d, Rang aus dem FLOP-Budget."""
    art, _, budget = ARME_V4[arm]
    if budget == "param":
        return art, breite_budget(art, d), None
    ziel = sz_flops_lage(d)
    if art == "dicht":
        return art, min(range(1, 8 * d), key=lambda h: (abs(ffn_flops_lage(art, d, h) - ziel), h)), None
    return art, 4 * d, min(range(1, d), key=lambda r: (abs(ffn_flops_lage(art, d, 4 * d, r) - ziel), r))


def baue_ffn(arm, d, gen):
    if arm in ARME_V4:                                 # I-045-v4: budgetgleiche kleine FFN (Breite aus der Parameterzaehlung)
        art, h, r = form_v4(arm, d)
        akt = ARME_V4[arm][1]
        return FFNDicht(d, h, akt) if art == "dicht" else FFNStruktur(d, h, "lr", akt, gen, r=r)
    F_ = BREITE[arm] if d == D_MODELL else 4 * d       # I-045-v3: bei anderer Modellbreite FFN-Breite 4·d (d = 256: unveraendert)
    if arm in ARME_V2 and arm != "Sz":
        art, akt = ARME_V2[arm]
        return FFNDicht(d, F_, akt) if art == "dicht" else FFNStruktur(d, F_, art, akt, gen)
    if arm == "D":
        return FFNDicht(d, F_)
    if arm == "L":
        return FFNStruktur(d, F_, "lr", F.gelu, gen)
    return FFNStruktur(d, F_, "ff", torch.square if arm == "Sz" else F.gelu, gen)


class Block(nn.Module):
    def __init__(self, arm, d, h, gen):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = baue_ffn(arm, d, gen)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.ffn(self.ln2(x))


class Sprachmodell(nn.Module):
    def __init__(self, arm, seed, d=D_MODELL, lagen=LAGEN, h=KOEPFE, ctx=KONTEXT):
        super().__init__()
        gen = torch.Generator().manual_seed(seed * 31 + 11)
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Block(arm, d, h, gen) for _ in range(lagen)])
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.bl:
            x = b(x)
        return self.ln(x) @ self.emb.weight.T


def param_ffn(netz) -> int:
    return int(sum(p.numel() for b in netz.bl for p in b.ffn.parameters() if p.requires_grad))


def param_gesamt(netz) -> int:
    return int(sum(p.numel() for p in netz.parameters() if p.requires_grad))


def bpc_auswerten(netz, text, ctx=KONTEXT, bs=8) -> float:
    """Mittlerer Verlust in bits/Zeichen ueber nicht ueberlappende Fenster (Vorhersage von Byte 1..ctx aus 0..ctx-1)."""
    n = (len(text) - 1) // ctx
    x = text[:n * ctx].long().reshape(n, ctx)
    y = text[1:n * ctx + 1].long().reshape(n, ctx)
    netz.eval()
    s = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            s += float(F.cross_entropy(netz(x[i:i + bs]).reshape(-1, 256), y[i:i + bs].reshape(-1), reduction="sum"))
    return s / (n * ctx) / math.log(2)


def zeit_je_token(netz, batch=ZEIT_BATCH) -> float:
    netz.eval()
    x = torch.randint(0, 256, (batch, KONTEXT))
    with torch.no_grad():
        return _stoppuhr(lambda: netz(x)) / (batch * KONTEXT)


# ------------------------------------------------------------------ Training
def trainiere(arm, lr, seed, schritte, klein=False, val=False, permutiert=False, zeit=False):
    torch.set_num_threads(1)
    tr, va, te = daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 77))]
    torch.manual_seed(seed)
    netz = Sprachmodell(arm, seed)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    offs = torch.arange(KONTEXT)
    t0 = time.time()
    netz.train()
    for _ in range(schritte):
        i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l):
            return {"bpc": 8.0, "instabil": True}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step()
        sched.step()
    t_train = time.time() - t0
    out = {"bpc": bpc_auswerten(netz, va[:VAL_BYTES] if val else te), "instabil": False, "sekunden_training": t_train}
    if zeit:
        out["zeit_je_token"] = zeit_je_token(netz)
    return out


def _lr_arbeiter(a):
    arm, s, lr, schritte, klein = a
    r = trainiere(arm, lr, s, schritte, klein=klein, val=True)
    return {"acc": 1 - r["bpc"] / 8, "bpc": r["bpc"], "instabil": r["instabil"]}      # 'acc' = Score (hoeher besser) fuer gittersuche_stabil


def tuning(arme, schritte, klein, tseed):
    info, _ = gittersuche_stabil(list(arme), lambda k, lr: [(k, tseed, lr, schritte, klein)],
                                 lambda aufg: parallel(_lr_arbeiter, aufg), gitter=LR_GITTER, faktor=LR_FAKTOR, kollaps=KOLLAPS_SCORE)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _arbeiter(a):
    teil, arm, s, lr, schritte, klein = a
    r = trainiere(arm, lr, s, schritte, klein=klein, permutiert=(teil == "nk"), zeit=(teil == "mess"))
    return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, **r}


def _schl(a):
    return f"einheit|{a[0]}|{a[1]}|{a[2]}"


def _zeit_arbeiter(a):
    torch.set_num_threads(1)
    b, arme = a if isinstance(a, tuple) else (a, ARME)
    return {arm: zeit_je_token(Sprachmodell(arm, 0), b) for arm in arme}


def zeit_im_ein_thread(batch=ZEIT_BATCH, arme=ARME):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter, ((batch, tuple(arme)),))


# ------------------------------------------------------------------ Auswertung (bpc: kleiner ist besser)
def _stat(d, versatz=0.0) -> dict:
    """d = (schlechterer Arm) - (besserer Arm) je Seed in bpc, positiv = Hypothese; Nichtunterlegenheit gegen -versatz."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return {"n": int(len(d))}
    from scipy.stats import t as tv
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(tv.ppf(0.95, len(d) - 1))
    return {"mittel": m, "sd": float(d.std(ddof=1)), "ki": [m - q * se, m + q * se], "positiv": int((d > 0).sum()), "n": int(len(d)),
            "p_gr": _t_p(d + versatz, "gr"), "p_kl": _t_p(d + versatz, "kl"), "p_zwei": _t_p(d, "zwei"),
            "wilcoxon_zweiseitig": wilcoxon_schnell(d) if len(set(np.round(d, 12))) > 1 else 1.0}


def auswertung(zeilen: list, params: dict, unigramm: float) -> dict:
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] == "mess"})
    bpc, zt, tt = {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            bpc.setdefault(z["arm"], {})[z["seed"]] = z["bpc"]
            if "zeit_je_token" in z:
                zt.setdefault(z["arm"], {})[z["seed"]] = z["zeit_je_token"]
            tt.setdefault(z["arm"], {})[z["seed"]] = z.get("sekunden_training", float("nan"))
    zl = [{"seed": s, **{a: bpc[a][s] for a in bpc if s in bpc[a]}} for s in seeds]
    zlt = [{"seed": s, **{a: zt[a][s] for a in zt if s in zt[a]}} for s in seeds]
    r = {"arme": {}, "kollaps": {}, "negativ": {}, "diff": {}, "p_holm": {}, "parameter": params, "beschreibend": {}, "unigramm_bpc": unigramm,
         "zeit_je_token_mittel": {}, "training_s_mittel": {a: float(np.nanmean(list(v.values()))) for a, v in tt.items()}}
    for a in ARME:
        if zl and a in zl[0]:
            r["arme"][a] = zusammenfassung(_arr(zl, a))
            w = _arr(zl, a)
            med = float(np.median(w))
            r["kollaps"][a] = [z["seed"] for z in zl if not (z[a] <= med + KOLLAPS_BPC)]
    for a, v in zt.items():
        r["zeit_je_token_mittel"][a] = float(np.mean(list(v.values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}
    if zl and "Sb" in zl[0] and "D" in zl[0]:
        r["diff"]["H1"] = _stat(_arr(zl, "D") - _arr(zl, "Sb"), versatz=NI_GRENZE)
        r["beschreibend"]["Sb - D (bpc, negativ = Sb besser)"] = _stat(_arr(zl, "D") - _arr(zl, "Sb"))
    if zl and "S" in zl[0] and "L" in zl[0]:
        r["diff"]["H2"] = _stat(_arr(zl, "L") - _arr(zl, "S"))
    if zlt and "Sb" in zlt[0] and "D" in zlt[0]:
        r["diff"]["H3"] = _stat(np.log(_arr(zlt, "D")) - np.log(_arr(zlt, "Sb")))
    for x, y in (("S", "D"), ("Sz", "S")):
        if zl and x in zl[0] and y in zl[0]:
            r["beschreibend"][f"{x} - {y} (bpc, negativ = {x} besser)"] = _stat(_arr(zl, y) - _arr(zl, x))
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1", "H2", "H3") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _zahlen(r, h, einheit="bpc"):
    d = r["diff"][h]
    return (f"d {d['mittel']:+.4f} {einheit}, 90-%-KI [{d['ki'][0]:+.4f}; {d['ki'][1]:+.4f}], n {d['n']}, "
            f"p_Holm {r['p_holm'][h]:.3g}").replace(".", ",")


def _nk(r) -> str:
    n = r["negativ"]
    return "BESTANDEN" if n and n["mittel"] >= NK_FAKTOR * r["unigramm_bpc"] else "DURCHGEFALLEN"


def _vorab(r, nk, h, arme, brauche_zeit=False):
    if h not in r["diff"] or "mittel" not in r["diff"][h] or h not in r["p_holm"]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    k = sorted({s for a in arme for s in r["kollaps"].get(a, [])})
    if k:
        return f"HU (Instabilitaet/Kollaps in Seeds {k})"
    return None


def _urteil_h1(r, nk) -> str:
    v = _vorab(r, nk, "H1", ("D", "Sb"))
    if v:
        return v
    z = _zahlen(r, "H1")
    p = r["parameter"]
    if "D" not in p or "Sb" not in p or not p["Sb"]["ffn"] < p["D"]["ffn"]:
        return f"HU (Sb hat nicht weniger FFN-Parameter als D; {z})"
    if r["p_holm"]["H1"] < ALPHA:
        return f"NICHT_UNTERLEGEN ({z}; FFN-Parameter Sb {p['Sb']['ffn']} gegen D {p['D']['ffn']})"
    if r["diff"]["H1"]["p_kl"] < ALPHA / 2:
        return f"H0 (Sb um mehr als {NI_GRENZE} bpc schlechter; {z})"
    return f"HU ({z})"


def _urteil_zweiseitig(r, nk, h, arme, gut, schlecht, einheit="bpc") -> str:
    v = _vorab(r, nk, h, arme)
    if v:
        return v
    z = _zahlen(r, h, einheit)
    if r["p_holm"][h] < ALPHA:
        return f"{gut} ({z})"
    if r["diff"][h]["p_kl"] < ALPHA / 2:
        return f"H0 ({schlecht}; {z})"
    return f"HU ({z})"


def urteil(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert D/Sb: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {NK_FAKTOR}x)"}
    u["H1"] = _urteil_h1(r, nk)
    u["H2"] = _urteil_zweiseitig(r, nk, "H2", ("S", "L"), "S_BESSER", "L sicher besser als S")
    u["H3"] = _urteil_zweiseitig(r, nk, "H3", ("D", "Sb"), "SCHNELLER", "Sb sicher langsamer als D", einheit="ln-Zeitverhaeltnis D/Sb")
    g = lambda a: f"{r['arme'][a]['mittel']:.4f}".replace(".", ",") if a in r["arme"] else "n/a"
    u["B"] = "beschreibend: Test-bpc " + ", ".join(f"{a} {g(a)}" for a in ARME) + "; Rest in auswertung"
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, off=None, sd=0.005, n=5, koll=None, nk=5.0, zeit=None):
    base = {"D": 2.40, "S": 2.55, "Sb": 2.41, "L": 2.60, "Sz": 2.56}
    base.update(off or {})
    zt = {"D": 1.0e-5, "S": 1.1e-5, "Sb": 0.6e-5, "L": 0.9e-5, "Sz": 1.1e-5}
    zt.update(zeit or {})
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for a, v in base.items():
            x = v + gem + rng.normal(0, sd)
            if koll == a and i == 0:
                x += 1.0
            zl.append({"teil": "mess", "arm": a, "seed": 2400 + i, "bpc": x, "sekunden_training": 10.0,
                       "zeit_je_token": zt[a] * (1 + rng.normal(0, 0.01))})
        if i < 2:
            for a in NK_ARME:
                zl.append({"teil": "nk", "arm": a, "seed": 2400 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _synth_param(off=None):
    p = {"D": {"ffn": 525_568, "gesamt": 3_000_000}, "S": {"ffn": 5_000, "gesamt": 1_500_000}, "Sb": {"ffn": 20_000, "gesamt": 1_520_000}}
    p.update(off or {})
    return p


def _u(rng, param_off=None, **kw):
    return urteil(auswertung(_synth(rng, **kw), _synth_param(param_off), 5.0))


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER: {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, KONTEXT))
    nz = {a: Sprachmodell(a, 0) for a in ARME}
    for a, m in nz.items():
        y = m(x)
        pruefe(f"Form {a}", y.shape == (2, KONTEXT, 256))
        # Kausalitaet: Aenderung eines spaeteren Bytes aendert fruehere Logits nicht
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 256
        m.eval()
        pruefe(f"Kausalitaet {a}", torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5) and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
    d, s, sb, l = (param_ffn(nz[a]) for a in ("D", "S", "Sb", "L"))
    pruefe("FFN-Parameter D = 4·(2·256·1024+1280)", d == 4 * (2 * 256 * 1024 + 1024 + 256))
    pruefe("FFN-Parameter S = 4·(4096 Diagonalen + 1280 Bias)", s == 4 * (2 * 4 * 256 + 2 * 1024 + 1024 + 256))
    pruefe("FFN-Parameter Sb = 4·(16384 + 4352 Bias)", sb == 4 * (2 * 16 * 256 + 2 * 4096 + 4096 + 256))
    pruefe("Sb hat weniger FFN-Parameter als D (mind. 10x)", sb * 10 < d)
    pruefe("L: Rang 2, Gewichte >= S und <= 1,3 S (ohne Bias)", rang_L(256, 1024) == 2 and
           1.0 <= (l - 4 * 1280) / (s - 4 * 1280) <= 1.3)
    pruefe("Sz und S gleich viele Parameter", param_ffn(nz["Sz"]) == s)
    pruefe("Sb Schicht 2: 4096->256 (ein Block, Auswahl 256)", nz["Sb"].bl[0].ffn.b.m.N == 256 and nz["Sb"].bl[0].ffn.b.m.m == 1 and nz["Sb"].bl[0].ffn.a.m.m == 16)
    # Trainierbarkeit: Wiederholungsmuster
    unit = torch.arange(32).repeat(8)[None].repeat(16, 1)[:, :KONTEXT + 1]
    for a in ARME:
        m = Sprachmodell(a, 0)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        l0 = None
        for k in range(30):
            opt.zero_grad()
            l = F.cross_entropy(m(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a}", float(l) < 0.7 * l0)
    # Daten: fester Split, nicht leer, Unigramm-Entropie plausibel
    tr, va, te = daten(klein=True)
    pruefe("Daten: Split nicht leer", len(tr) > 10_000 and len(va) > 1000 and len(te) > 1000)
    pruefe("Unigramm-Entropie plausibel (3-6,5 bpc)", 3.0 < unigramm_entropie(tr, te) < 6.5)
    pruefe("bpc einer unbeschulten Modells ~ 8 bit", 7.0 < bpc_auswerten(nz["D"], te[:2000]) < 10.0)
    pruefe("Seeds frisch", len(SEEDS) == 5 and not (set(SEEDS) & _BELEGT) and TUNING_SEED not in SEEDS and TUNING_SEED not in _BELEGT)
    pruefe("Profile: Fallback kürzer", all(PROFILE["fallback"][k] <= PROFILE["standard"][k] for k in PROFILE["standard"]))
    d_ = np.array([0.004, 0.006, 0.005, 0.004, 0.006])
    pruefe("NI: Sb 0,005 bpc besser/gleich ist nichtunterlegen", _stat(d_ * 0 - 0.005 + d_ * 0.1, NI_GRENZE)["p_gr"] < 0.05)
    pruefe("NI: Sb 0,05 bpc schlechter ist es nicht", _stat(d_ * 0.1 - 0.05, NI_GRENZE)["p_gr"] > 0.9)
    return ok


def _fn_test(a):
    return _schl(a), {"v": a[2] * 2}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as t:
        pfad = Path(t) / "z.json"
        kennung = {"k": 1}
        aufg = [("mess", a, s) for a in ("D", "Sb") for s in (1, 2)]
        z1 = _rechne(aufg[:2], pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        z2 = _rechne(aufg, pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        ok &= len(z1) == 2 and len(z2) == 4 and z2["einheit|mess|D|2"] == z1["einheit|mess|D|2"]
        try:
            _rechne(aufg, pfad, {"k": 2}, fn=_fn_test, prozesse=1, schluessel=_schl)
            ok = False
        except ValueError:
            pass
    if not ok:
        print("  Vortest FEHLER: Wiederaufnahme", flush=True)
    return ok


def _vortest_urteile() -> bool:
    rng = np.random.default_rng(5)
    ok = True
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    erwarte("Normalfall", _u(rng), **{"NK": "BESTANDEN", "H1": "NICHT_UNTERLEGEN", "H2": "S_BESSER", "H3": "SCHNELLER"})
    erwarte("Sb deutlich schlechter", _u(rng, off={"Sb": 2.60}), **{"H1": "H0"})
    erwarte("Sb knapp nicht erkennbar", _u(rng, off={"Sb": 2.425}, sd=0.01), **{"H1": "HU"})
    erwarte("Sb nicht weniger Parameter", _u(rng, param_off={"Sb": {"ffn": 600_000, "gesamt": 3_100_000}}), **{"H1": "HU (Sb hat nicht weniger"})
    erwarte("L sicher besser", _u(rng, off={"S": 2.65, "L": 2.55}), **{"H2": "H0"})
    erwarte("S gleich L", _u(rng, off={"S": 2.58, "L": 2.58}, sd=0.02), **{"H2": "HU"})
    erwarte("Sb langsamer", _u(rng, zeit={"Sb": 3e-5}), **{"H3": "H0"})
    erwarte("Zeit gleich", _u(rng, zeit={"Sb": 1.0e-5}), **{"H3": "HU"})
    erwarte("NK faellt (Leck)", _u(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle",
                                                  "H3": "HU (Negativkontrolle"})
    erwarte("Kollaps Sb", _u(rng, koll="Sb"), **{"H1": "HU (Instabilitaet", "H3": "HU (Instabilitaet", "H2": "S_BESSER"})
    erwarte("Kollaps L", _u(rng, koll="L"), **{"H2": "HU (Instabilitaet"})
    erwarte("Arm fehlt", urteil(auswertung([z for z in _synth(rng) if z["arm"] != "L"], _synth_param(), 5.0)), **{"H2": "HU (Arme fehlen"})
    erlaubt = ("NICHT_UNTERLEGEN", "S_BESSER", "SCHNELLER", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    if not all(isinstance(v, str) and v.startswith(erlaubt) for v in _u(rng).values()):
        print("  Vortest FEHLER: Urteil ausserhalb des Vokabulars", flush=True)
        ok = False
    soll = {"H1": {"NICHT_UNTERLEGEN", "H0", "HU"}, "H2": {"S_BESSER", "H0", "HU"}, "H3": {"SCHNELLER", "H0", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k, w in gesehen if k == h}
        if fehlt:
            print(f"  Vortest FEHLER: Zweige {fehlt} fuer {h} nicht erzwungen", flush=True)
            ok = False
    return ok


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok &= _vortest_wiederaufnahme()
    ok &= _vortest_urteile()
    ok &= _vortest_v2()
    ok &= _vortest_v3()
    ok &= _vortest_v4()
    return bool(ok)


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None):
    profil = profil or os.environ.get("I045_PROFIL", "standard")
    P = PROFILE[profil]
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte
        seeds, klein, nk_seeds = (TUNING_SEED,), True, (TUNING_SEED,)
        P = {"schritte": 20, "nk_schritte": 10}
    else:
        seeds, klein, nk_seeds = SEEDS, False, NK_SEEDS
        if set(seeds) & _BELEGT or TUNING_SEED in seeds:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": P, "lr_gitter": list(LR_GITTER),
               "tuning_seed": TUNING_SEED, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        zw["tuning"] = tuning(ARME, P["schritte"], klein, TUNING_SEED)
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = [("mess", a, s, hp[a]["lr"], P["schritte"], klein) for a in ARME for s in seeds]
    aufg += [("nk", a, s, hp[a]["lr"], P["nk_schritte"], klein) for a in NK_ARME for s in nk_seeds]
    kosten = {"Sb": 4, "D": 3, "S": 3, "Sz": 3, "L": 2}
    aufg.sort(key=lambda a: -kosten.get(a[1], 1))
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread()
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    netze = {a: Sprachmodell(a, 0) for a in ARME}
    params = {a: {"ffn": param_ffn(n), "gesamt": param_gesamt(n)} for a, n in netze.items()}
    tr, _, te = daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_sprache", "karte": KARTE, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN,
         "folgefragen": FOLGEFRAGEN, "seeds": list(seeds), "tuning_seed": TUNING_SEED, "tuning": hp,
         "einstellungen": {"schritte": P["schritte"], "nk_schritte": P["nk_schritte"], "batch": BATCH, "kontext": KONTEXT, "d": D_MODELL,
                           "lagen": LAGEN, "breite": BREITE, "lr_gitter": list(LR_GITTER), "daten_bytes": [len(x) for x in daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, params, unigramm_entropie(tr, te))
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
        e["zeitverhaeltnis_D_durch_arm"] = {a: zm["D"] / v for a, v in zm.items()}
    e["urteil"] = urteil(e["auswertung"])
    if zm is not None:
        e["urteil"]["B-Zeit"] = ("beschreibend: Zeit/Token Inferenz (Einzelprozess, 1 Thread), Verhaeltnis D/Arm: "
                                 + ", ".join(f"{a} {v:.2f}" for a, v in e["zeitverhaeltnis_D_durch_arm"].items()))
    return e


# ================================================================== I-045-v2: vorregistrierte Bestaetigung des Nebenbefunds Sz (Fastfood+z²) < D (GELU)
# Profil/Erweiterung von I-045 (gleiches Modell, Daten, Training, Tuning-Verfahren; nur Arme, Seeds, Hypothesen neu), keine Kopie.
# ARME: D_gelu (= D aus I-045) | D_z2 (dicht, z²) | Sz (Fastfood, z²) | S_gelu (= S aus I-045) | L_z2 (niedriger Rang 2, z²) | D_relu2 (dicht, squared ReLU), alle FFN-Breite 1024.
# HYPOTHESEN (Holm ueber H1-H3, gepaarter t-Test n=10, bpc, d = Referenz - Pruefarm je Seed):
#   H1 Sz < D_gelu (einseitig)                          : SZ_BESSER | H0 (D_gelu sicher besser) | HU
#   H2 Sz vs D_z2: TOST ±0,02 bpc ODER zweiseitig        : AEQUIVALENT (TOST) | UNTERSCHIED (zweiseitig, Richtung im Text) | HU
#      p_H2 = min(1, 2·min(p_TOST, p_zweiseitig)) (Bonferroni innerhalb H2, dann Holm); AEQUIVALENT hat Vorrang, wenn beide Holm-signifikant sind.
#   H3 D_z2 < D_gelu (einseitig, Aktivierungseffekt)     : AKTIVIERUNG | H0 (D_gelu sicher besser) | HU
# DIVERGENZ (vorab): Seed divergiert, wenn der Trainingsverlust nicht endlich wird (bpc := 8) ODER der End-Test-bpc >= Unigramm-Entropie des Tests.
#   Divergierte Seeds bleiben in allen Zahlen (kein Verwerfen), zaehlen aber als HU: jede Hypothese mit einem Arm, der in einem Seed divergiert
#   oder kollabiert (Median + 0,2 bpc), ist HU. Divergenzrate je Arm beschreibend. Tuning: Kollaps-Score wie I-045.
# NEGATIVKONTROLLE wie I-045, aber D_gelu und Sz (permutierte Bytes, Seeds 2410, 2411, 300 Schritte).
# BESCHREIBEND: Sz vs L_z2, Sz vs D_relu2, D_z2 vs D_relu2, S_gelu vs Sz (Struktur bei GELU), Parameter, Trainingszeit, Inferenzzeit, Divergenzrate.
KARTE_V2 = "I-045-v2"
SEEDS_V2 = tuple(range(2410, 2420))
TUNING_SEED_V2 = 2498
PROFILE_V2 = {"standard": {"schritte": 1000, "nk_schritte": 300, "seeds": SEEDS_V2},
              "acht": {"schritte": 1000, "nk_schritte": 300, "seeds": SEEDS_V2[:8]}}      # „acht“ nur bei Zeitnot (vorab)
ARME_M = tuple(ARME_V2)                    # Messarme
NK_ARME_V2 = ("D_gelu", "Sz")
NK_SEEDS_V2 = (2410, 2411)
TOST_GRENZE = 0.02
_BELEGT_V2 = _BELEGT | set(range(2400, 2405)) | {2499} | set(range(2210, 2225)) | set(range(2310, 2325))
HYPOTHESEN_V2 = {
    "H1": "Sz < D_gelu (bpc, einseitig gepaart t, Holm H1-H3). SZ_BESSER | H0 (D_gelu sicher besser) | HU.",
    "H2": f"Sz gegen D_z2: TOST +-{TOST_GRENZE} bpc ODER zweiseitig (p_H2 = min(1, 2 min(p_TOST, p_zwei)), Holm). AEQUIVALENT | UNTERSCHIED | HU.",
    "H3": "D_z2 < D_gelu (Aktivierungseffekt, einseitig, Holm). AKTIVIERUNG | H0 (D_gelu sicher besser) | HU.",
    "DIVERGENZ": "Seed divergiert: Verlust nicht endlich ODER End-bpc >= Unigramm-Entropie. Bleibt in den Zahlen, Hypothesen mit solchem Arm = HU.",
    "NK": f"Negativkontrolle (D_gelu, Sz, permutierte Trainingsbytes): mittlerer Test-bpc >= {NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc je Arm, Sz-L_z2, Sz-D_relu2, D_z2-D_relu2, S_gelu-Sz, Parameter, Zeiten, Divergenzrate, Lernraten.",
}


def auswertung_v2(zeilen: list, params: dict, unigramm: float) -> dict:
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] == "mess"})
    bpc, zt, tt, inst = {}, {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            bpc.setdefault(z["arm"], {})[z["seed"]] = z["bpc"]
            inst.setdefault(z["arm"], {})[z["seed"]] = bool(z.get("instabil"))
            if "zeit_je_token" in z:
                zt.setdefault(z["arm"], {})[z["seed"]] = z["zeit_je_token"]
            tt.setdefault(z["arm"], {})[z["seed"]] = z.get("sekunden_training", float("nan"))
    zl = [{"seed": s, **{a: bpc[a][s] for a in bpc if s in bpc[a]}} for s in seeds]
    zlt = [{"seed": s, **{a: zt[a][s] for a in zt if s in zt[a]}} for s in seeds]
    r = {"arme": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {}, "negativ": {}, "diff": {}, "p_holm": {}, "parameter": params,
         "beschreibend": {}, "unigramm_bpc": unigramm, "zeit_je_token_mittel": {}, "bpc_ohne_divergente": {},
         "training_s_mittel": {a: float(np.nanmean(list(v.values()))) for a, v in tt.items()}}
    for a in ARME_M:
        if zl and a in zl[0]:
            w = _arr(zl, a)
            r["arme"][a] = zusammenfassung(w)
            med = float(np.median(w))
            r["kollaps"][a] = [z["seed"] for z in zl if not (z[a] <= med + KOLLAPS_BPC)]
            r["divergiert"][a] = [z["seed"] for z in zl if inst[a].get(z["seed"]) or not (z[a] < unigramm)]
            r["divergenzrate"][a] = len(r["divergiert"][a]) / len(zl)
            ok = [z[a] for z in zl if z["seed"] not in r["divergiert"][a]]
            r["bpc_ohne_divergente"][a] = float(np.mean(ok)) if ok else None
    for a, v in zt.items():
        r["zeit_je_token_mittel"][a] = float(np.mean(list(v.values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def hat(*a):
        return bool(zl) and all(x in zl[0] for x in a)
    if hat("D_gelu", "Sz"):
        r["diff"]["H1"] = _stat(_arr(zl, "D_gelu") - _arr(zl, "Sz"))
    if hat("Sz", "D_z2"):
        d = _arr(zl, "D_z2") - _arr(zl, "Sz")
        st = _stat(d)
        if "p_gr" in st:
            st["p_tost"] = max(_t_p(d + TOST_GRENZE, "gr"), _t_p(d - TOST_GRENZE, "kl"))
            st["p_komposit"] = min(1.0, 2 * min(st["p_tost"], st["p_zwei"]))
        r["diff"]["H2"] = st
    if hat("D_gelu", "D_z2"):
        r["diff"]["H3"] = _stat(_arr(zl, "D_gelu") - _arr(zl, "D_z2"))
    for x, y in (("Sz", "L_z2"), ("Sz", "D_relu2"), ("D_z2", "D_relu2"), ("S_gelu", "Sz")):
        if hat(x, y):
            r["beschreibend"][f"{x} - {y} (bpc, negativ = {x} besser)"] = _stat(_arr(zl, y) - _arr(zl, x))
    prim = {}
    for h in ("H1", "H3"):
        if h in r["diff"] and "p_gr" in r["diff"][h]:
            prim[h] = r["diff"][h]["p_gr"]
    if "H2" in r["diff"] and "p_komposit" in r["diff"]["H2"]:
        prim["H2"] = r["diff"]["H2"]["p_komposit"]
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _vorab_v2(r, nk, h, arme):
    if h not in r["diff"] or "mittel" not in r["diff"][h] or h not in r["p_holm"]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    dv = {a: r["divergiert"].get(a, []) for a in arme if r["divergiert"].get(a)}
    if dv:
        return f"HU (Divergenz in {dv})"
    k = {a: r["kollaps"].get(a, []) for a in arme if r["kollaps"].get(a)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def urteil_v2(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert D_gelu/Sz: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {NK_FAKTOR}x)"}
    for h, arme, gut, schlecht in (("H1", ("D_gelu", "Sz"), "SZ_BESSER", "D_gelu sicher besser als Sz"),
                                   ("H3", ("D_gelu", "D_z2"), "AKTIVIERUNG", "D_gelu sicher besser als D_z2")):
        u[h] = _vorab_v2(r, nk, h, arme) or _urteil_zweiseitig(r, nk, h, arme, gut, schlecht)
    v = _vorab_v2(r, nk, "H2", ("Sz", "D_z2"))
    if v:
        u["H2"] = v
    else:
        d, ph = r["diff"]["H2"], r["p_holm"]["H2"]
        z = (f"D_z2-Sz {d['mittel']:+.4f} bpc, 90-%-KI [{d['ki'][0]:+.4f}; {d['ki'][1]:+.4f}], n {d['n']}, p_TOST {d['p_tost']:.3g}, "
             f"p_zweiseitig {d['p_zwei']:.3g}, p_H2_Holm {ph:.3g}").replace(".", ",")
        if ph < ALPHA and d["p_tost"] <= d["p_zwei"]:
            u["H2"] = f"AEQUIVALENT ({z})"
        elif ph < ALPHA:
            u["H2"] = f"UNTERSCHIED ({'Sz besser' if d['mittel'] > 0 else 'D_z2 besser'}; {z})"
        else:
            u["H2"] = f"HU ({z})"
    g = lambda a: f"{r['arme'][a]['mittel']:.4f}".replace(".", ",") if a in r["arme"] else "n/a"
    u["B"] = "beschreibend: Test-bpc " + ", ".join(f"{a} {g(a)}" for a in ARME_M) + "; Divergenzrate " + \
             ", ".join(f"{a} {r['divergenzrate'].get(a, float('nan')):.2f}" for a in ARME_M) + "; Rest in auswertung"
    return u


def lauf_v2(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I045V2_PROFIL", "standard")
    P = PROFILE_V2[profil]
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte
        seeds, klein, nk_seeds = (TUNING_SEED_V2,), True, (TUNING_SEED_V2,)
        P = {"schritte": 20, "nk_schritte": 10}
    else:
        seeds, klein, nk_seeds = P["seeds"], False, NK_SEEDS_V2
        if set(seeds) & _BELEGT_V2 or TUNING_SEED_V2 in seeds or TUNING_SEED_V2 in _BELEGT_V2:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V2}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE_V2, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": {k: list(v) if k == "seeds" else v for k, v in P.items()},
               "lr_gitter": list(LR_GITTER), "tuning_seed": TUNING_SEED_V2, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        zw["tuning"] = tuning(ARME_M, P["schritte"], klein, TUNING_SEED_V2)
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = [("mess", a, s, hp[a]["lr"], P["schritte"], klein) for a in ARME_M for s in seeds]
    aufg += [("nk", a, s, hp[a]["lr"], P["nk_schritte"], klein) for a in NK_ARME_V2 for s in nk_seeds]
    kosten = {"D_gelu": 3, "D_z2": 3, "D_relu2": 3, "Sz": 3, "S_gelu": 3, "L_z2": 2}
    aufg.sort(key=lambda a: -kosten.get(a[1], 1))
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread(arme=ARME_M)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    netze = {a: Sprachmodell(a, 0) for a in ARME_M}
    params = {a: {"ffn": param_ffn(n), "gesamt": param_gesamt(n)} for a, n in netze.items()}
    tr, _, te = daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_sprache", "karte": KARTE_V2, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_V2,
         "seeds": list(seeds), "tuning_seed": TUNING_SEED_V2, "tuning": hp,
         "einstellungen": {"schritte": P["schritte"], "nk_schritte": P["nk_schritte"], "batch": BATCH, "kontext": KONTEXT, "d": D_MODELL,
                           "lagen": LAGEN, "breite": {a: BREITE[a] for a in ARME_M}, "lr_gitter": list(LR_GITTER), "daten_bytes": [len(x) for x in daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_v2(zeilen, params, unigramm_entropie(tr, te))
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
        e["zeitverhaeltnis_D_gelu_durch_arm"] = {a: zm["D_gelu"] / v for a, v in zm.items()}
    e["urteil"] = urteil_v2(e["auswertung"])
    return e


def _synth_v2(rng, off=None, sd=0.005, n=10, koll=None, nk=5.0, div=None):
    base = {"D_gelu": 2.50, "D_z2": 2.40, "Sz": 2.40, "S_gelu": 2.62, "L_z2": 2.45, "D_relu2": 2.41}
    base.update(off or {})
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for a, v in base.items():
            x, inst = v + gem + rng.normal(0, sd), False
            if koll == a and i == 0:
                x += 1.0
            if div == a and i == 1:
                x, inst = 8.0, True
            zl.append({"teil": "mess", "arm": a, "seed": 2410 + i, "bpc": x, "instabil": inst, "sekunden_training": 10.0, "zeit_je_token": 1e-5})
        if i < 2:
            for a in NK_ARME_V2:
                zl.append({"teil": "nk", "arm": a, "seed": 2410 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u2(rng, **kw):
    p = {a: {"ffn": 1, "gesamt": 2} for a in ARME_M}
    return urteil_v2(auswertung_v2(_synth_v2(rng, **kw), p, 5.0))


def _vortest_v2() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-045-v2): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, KONTEXT))
    nz = {a: Sprachmodell(a, 0) for a in ARME_M}
    for a, m in nz.items():
        m.eval()
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 256
        pruefe(f"Form/Kausalitaet {a}", m(x).shape == (2, KONTEXT, 256) and torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5)
               and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
    pf = {a: param_ffn(n) for a, n in nz.items()}
    pruefe("Parameter: D_gelu = D_z2 = D_relu2 (dicht)", pf["D_gelu"] == pf["D_z2"] == pf["D_relu2"] == 4 * (2 * 256 * 1024 + 1024 + 256))
    pruefe("Parameter: Sz = S_gelu = Fastfood", pf["Sz"] == pf["S_gelu"] == param_ffn(Sprachmodell("S", 0)))
    pruefe("Parameter: L_z2 = L", pf["L_z2"] == param_ffn(Sprachmodell("L", 0)))
    pruefe("Aktivierungen: D_gelu GELU, D_z2 z², D_relu2 relu²", nz["D_gelu"].bl[0].ffn.akt is F.gelu and nz["D_z2"].bl[0].ffn.akt is torch.square
           and torch.allclose(_relu2(torch.tensor([-2.0, 3.0])), torch.tensor([0.0, 9.0])))
    pruefe("Sz identisch zu I-045 (Fastfood, z²)", nz["Sz"].bl[0].ffn.akt is torch.square and nz["Sz"].bl[0].ffn.a.m.m == 4)
    def _neu(a, sd):
        torch.manual_seed(sd)
        return list(Sprachmodell(a, sd).parameters())
    pruefe("Sz bei gleichem Seed reproduzierbar", all(torch.equal(p, q) for p, q in zip(_neu("Sz", 3), _neu("Sz", 3))))
    unit = torch.arange(32).repeat(8)[None].repeat(16, 1)[:, :KONTEXT + 1]
    for a in ARME_M:
        m = Sprachmodell(a, 0)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        l0 = None
        for k in range(30):
            opt.zero_grad()
            l = F.cross_entropy(m(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a}", float(l) < 0.7 * l0)
    pruefe("Seeds frisch (10, Tuning 2498 neu)", len(SEEDS_V2) == 10 and not (set(SEEDS_V2) & _BELEGT_V2) and TUNING_SEED_V2 not in _BELEGT_V2
           and set(PROFILE_V2["acht"]["seeds"]) <= set(SEEDS_V2))
    # Divergenz im Trainingsschritt: absurd hohe Lernrate muss als instabil/bpc 8 enden oder End-bpc >= Unigramm ergeben (Zweig der Erkennung)
    t = trainiere("Sz", 1e9, 0, 5, klein=True, val=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)

    rng = np.random.default_rng(7)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER (I-045-v2) {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    erwarte("Sz = D_z2 < D_gelu", _u2(rng), **{"NK": "BESTANDEN", "H1": "SZ_BESSER", "H2": "AEQUIVALENT", "H3": "AKTIVIERUNG"})
    erwarte("Sz deutlich besser als D_z2", _u2(rng, off={"Sz": 2.30}), **{"H2": "UNTERSCHIED (Sz besser"})
    erwarte("D_z2 deutlich besser als Sz", _u2(rng, off={"Sz": 2.50, "D_z2": 2.30}), **{"H2": "UNTERSCHIED (D_z2 besser"})
    erwarte("D_gelu besser als Sz", _u2(rng, off={"D_gelu": 2.30}), **{"H1": "H0", "H3": "H0"})
    erwarte("Kein Unterschied D_gelu", _u2(rng, off={"D_gelu": 2.40, "Sz": 2.40, "D_z2": 2.40}, sd=0.02), **{"H1": "HU", "H3": "HU"})
    erwarte("H2 unklar", _u2(rng, off={"Sz": 2.42}, sd=0.03), **{"H2": "HU"})
    erwarte("NK faellt", _u2(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle", "H3": "HU (Negativkontrolle"})
    erwarte("Divergenz Sz", _u2(rng, div="Sz"), **{"H1": "HU (Divergenz", "H2": "HU (Divergenz", "H3": "AKTIVIERUNG"})
    erwarte("Kollaps D_z2", _u2(rng, koll="D_z2"), **{"H3": "HU (Kollaps", "H1": "SZ_BESSER"})
    p = {a: {"ffn": 1, "gesamt": 2} for a in ARME_M}
    erwarte("Arm fehlt", urteil_v2(auswertung_v2([z for z in _synth_v2(rng) if z["arm"] != "D_z2"], p, 5.0)), **{"H2": "HU (Arme fehlen", "H3": "HU (Arme fehlen"})
    a_ = auswertung_v2(_synth_v2(rng, div="Sz"), p, 5.0)
    pruefe("Divergenzrate Sz = 0,1, Seed bleibt in den Zahlen (n=10)", abs(a_["divergenzrate"]["Sz"] - 0.1) < 1e-9 and a_["arme"]["Sz"]["n"] == 10
           and a_["diff"]["H1"]["n"] == 10)
    erlaubt = ("SZ_BESSER", "AEQUIVALENT", "UNTERSCHIED", "AKTIVIERUNG", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u2(rng).values()))
    soll = {"H1": {"SZ_BESSER", "H0", "HU"}, "H2": {"AEQUIVALENT", "UNTERSCHIED", "HU"}, "H3": {"AKTIVIERUNG", "H0", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k, w in gesehen if k == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    # Wiederaufnahme (generisch, gleiche Maschinerie)
    ok &= _vortest_wiederaufnahme()
    return bool(ok)


# ================================================================== I-045-v3: Skalierung Struktur × Quadrat (Trainingsschritte × Modellbreite)
# Profil/Erweiterung von I-045/-v2 (gleiche Daten UD_German-GSD Byte-Ebene, gleicher Split, gleiche Bausteine und Messung bpc), keine Kopie.
# Wirkung klassisch (I-038): „quanten-inspirierter Vorteil“, kein Quantenvorteil. Gerechnet wird auf dem MPCDF-Cluster Raven (bis 72 Prozesse, je 1 Thread,
# 24-h-Joblimit): Wiederaufnahme je Aufgabe (Arm, d, T, Seed) UND Checkpoint innerhalb einer Aufgabe alle 2000 Schritte.
# ARME (FFN-Breite 4·d, nur FFN verschieden): Sz (Fastfood + z²) | D_z2 (dicht + z²) | D_gelu (dicht, GELU) | D_relu2 (dicht, squared ReLU).
# FAKTOREN: Trainingsschritte T ∈ {1000, 4000, 16000} (je eigener Lauf, eigener Kosinus-Plan, 5 % Aufwaermen), Modellbreite d ∈ {256, 512} (Koepfe d/64;
#   sonst wie v2: 4 Lagen, Kontext 128, Batch 8×128, Adam, Gradientenbegrenzung 1,0).
# SEEDS: 8 frische Mess-Seeds 2600-2607, 2 eigene NK-Seeds 2608/2609, Tuning-Seed 2698.
# LERNRATE je (Arm, d) auf dem Tuning-Seed bei T = 4000 (Validierung = erste 64 kB dev, gittersuche_stabil wie v2); diese Rate gilt fuer ALLE T.
#   VORBEHALT: bei T = 1000 und T = 16000 ist die Rate nicht neu gesucht; der Kosinus-Plan skaliert mit, ob 4000-Schritt-Raten bei 16000 Schritten
#   optimal sind, ist offen (eher zu hoch als zu niedrig). Divergenzregel (unten) faengt Instabilitaet ab.
# HYPOTHESEN (gepaarter einseitiger t-Test, n = 8, Differenz D_z2 − Sz je Seed in bpc, Holm ueber H1a, H1b, alpha 0,05):
#   H1a Sz < D_z2 bei T = 16000, d = 256 : SZ_BESSER | H0 (D_z2 sicher besser) | HU
#   H1b dasselbe bei d = 512             : SZ_BESSER | H0 | HU
#   H2 (zweiseitig, NICHT in Holm, alpha 0,05): Trend des Vorsprungs V = D_z2 − Sz von T = 1000 nach T = 16000, je Seed gemittelt ueber beide d:
#      Δ = mean_d[V(16000, d) − V(1000, d)]; SCHRUMPFT (Δ < 0, p < alpha) | WAECHST (Δ > 0, p < alpha) | HU
# DIVERGENZ (vorab, wie v2, je Zelle Arm × T × d): Seed divergiert bei nicht endlichem Verlust (bpc := 8) ODER End-bpc >= Unigramm-Entropie;
#   Kollaps = Median + 0,2 bpc. Divergierte Seeds bleiben in allen Zahlen; jede Hypothese mit einer beteiligten Zelle mit Divergenz/Kollaps ist HU.
# NEGATIVKONTROLLE wie v2: D_z2 und Sz mit permutierten Trainingsbytes, d = 256, 1000 Schritte, Seeds 2608/2609; mittlerer Test-bpc >= 0,9 × Unigramm.
# BESCHREIBEND: alle Arme × T × d (bpc, Differenzen), FFN-/Gesamtparameter, Inferenzzeit je Token (Einzelprozess, 1 Thread), Trainingszeit, Divergenzrate, Lernraten.
# ERWARTUNG VORAB (kein Befund): Der Vorsprung schrumpft mit T (H2 eher SCHRUMPFT); Sz bleibt mindestens gleichwertig zu D_z2 (H1a/H1b eher SZ_BESSER oder HU, nicht H0).
KARTE_V3 = "I-045-v3"
SEEDS_V3 = tuple(range(2600, 2608))
NK_SEEDS_V3 = (2608, 2609)
TUNING_SEED_V3 = 2698
ARME_V3 = ("Sz", "D_z2", "D_gelu", "D_relu2")
NK_ARME_V3 = ("D_z2", "Sz")
T_WERTE = (1000, 4000, 16000)
D_WERTE = (256, 512)
CKPT_ALLE = 2000
PROFILE_V3 = {"standard": {"T": T_WERTE, "d": D_WERTE, "tuning_T": 4000, "nk_T": 1000, "nk_d": 256, "ckpt_alle": CKPT_ALLE,
                           "seeds": SEEDS_V3, "nk_seeds": NK_SEEDS_V3}}
_BELEGT_V3 = (_BELEGT_V2 | set(range(2410, 2420)) | {2498} | set(range(2330, 2345)) | set(range(2350, 2358)) | set(range(2460, 2470)) | {2479, 2399, 2299})
HYPOTHESEN_V3 = {
    "H1a": "Sz < D_z2 (bpc) bei T=16000, d=256 (einseitig gepaart t, n=8, Holm H1a-H1b). SZ_BESSER | H0 (D_z2 sicher besser) | HU.",
    "H1b": "Sz < D_z2 (bpc) bei T=16000, d=512 (einseitig gepaart t, n=8, Holm H1a-H1b). SZ_BESSER | H0 (D_z2 sicher besser) | HU.",
    "H2": "Trend des Vorsprungs V = D_z2 - Sz von T=1000 nach T=16000 (je Seed gemittelt ueber d=256 und 512; zweiseitig gepaart t, nicht in Holm). SCHRUMPFT | WAECHST | HU.",
    "DIVERGENZ": "Seed divergiert: Verlust nicht endlich ODER End-bpc >= Unigramm-Entropie; Kollaps: Median + 0,2 bpc (je Zelle Arm x T x d). Bleibt in den Zahlen, beteiligte Hypothese = HU.",
    "NK": f"Negativkontrolle (D_z2, Sz, d=256, 1000 Schritte, permutierte Trainingsbytes, Seeds 2608/2609): mittlerer Test-bpc >= {NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc aller Arme x T x d, Differenzen gegen Sz, FFN-/Gesamtparameter, Inferenzzeit/Token, Trainingszeit, Divergenzrate, Lernraten.",
}

# ---- Kostenmodell (Messung im Container: 1 Thread, Schritt 8×128 = Vorwaerts + Rueckwaerts + Adam; siehe schritt_zeiten) ----
# Gemessen 22.10.2026 im Container (4 Kerne, 1 Thread, Median ueber 15 bzw. 25 Schritte; Maximum zweier Messungen): Sekunden je Schritt.
S_JE_SCHRITT = {("Sz", 256): 0.32, ("D_z2", 256): 0.32, ("D_gelu", 256): 0.30, ("D_relu2", 256): 0.32,
                ("Sz", 512): 0.79, ("D_z2", 512): 1.03, ("D_gelu", 512): 0.99, ("D_relu2", 512): 0.97}


class _Abbruch(Exception):
    """Nur fuer den Vortest: simuliert einen harten Abbruch (24-h-Limit) mitten in einer Aufgabe."""


def _json_sicher(x):
    if isinstance(x, dict):
        return {k: _json_sicher(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sicher(v) for v in x]
    return x


def _ckpt_pfad(ordner, key) -> str:
    return str(Path(ordner) / (key.replace("|", "_").replace("/", "-") + ".pt"))


def trainiere_v3(arm, lr, seed, schritte, d=D_MODELL, klein=False, val=False, permutiert=False, ckpt=None, ckpt_alle=CKPT_ALLE, _stopp_bei=None):
    """Wie `trainiere`, aber mit Modellbreite d (Koepfe d/64) und Checkpoint in der Aufgabe (alle `ckpt_alle` Schritte):
    Modell, Optimierer, Lernraten-Plan, Schritt, Zufallszustaende. Wiederaufnahme nur, wenn die Metadaten (Arm, d, lr, Seed, Schritte, ...) passen."""
    torch.set_num_threads(1)
    tr, va, te = daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 77))]
    torch.manual_seed(seed)
    netz = Sprachmodell(arm, seed, d=d, h=d // 64)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    meta = {"arm": arm, "d": int(d), "lr": float(lr), "seed": int(seed), "schritte": int(schritte), "klein": bool(klein), "perm": bool(permutiert)}
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
    offs = torch.arange(KONTEXT)
    t0 = time.time()
    netz.train()
    for s in range(start, schritte):
        i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l):
            if ckpt and Path(ckpt).exists():
                Path(ckpt).unlink()
            return {"bpc": 8.0, "instabil": True, "fortgesetzt_ab": start}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step()
        sched.step()
        if ckpt and (s + 1) % ckpt_alle == 0 and s + 1 < schritte:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt + ".tmp"
            torch.save({"meta": meta, "modell": netz.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "g": g.get_state(),
                        "rng": torch.get_rng_state(), "schritt": s + 1, "sekunden": t_vor + time.time() - t0}, tmp)
            os.replace(tmp, ckpt)
        if _stopp_bei is not None and s + 1 == _stopp_bei:
            raise _Abbruch(f"simulierter Abbruch bei Schritt {s + 1}")
    t_train = t_vor + time.time() - t0
    out = {"bpc": bpc_auswerten(netz, va[:VAL_BYTES] if val else te), "instabil": False, "sekunden_training": t_train, "fortgesetzt_ab": start}
    if ckpt and Path(ckpt).exists():
        Path(ckpt).unlink()
    return out


def schritt_zeiten(arme=ARME_V3, ds=D_WERTE, n=20, warm=3):
    """Messung: Sekunden je Trainingsschritt (1 Thread, Batch 8×128, Vorwaerts + Rueckwaerts + Clip + Adam), Median ueber n Schritte."""
    torch.set_num_threads(1)
    tr, _, _ = daten(True)
    offs = torch.arange(KONTEXT)
    g = torch.Generator().manual_seed(1)
    aus = {}
    for d in ds:
        for a in arme:
            netz = Sprachmodell(a, 0, d=d, h=d // 64)
            opt = torch.optim.Adam(netz.parameters(), lr=1e-4)
            ts = []
            for k in range(warm + n):
                i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
                t0 = time.perf_counter()
                l = F.cross_entropy(netz(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
                opt.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
                opt.step()
                if k >= warm:
                    ts.append(time.perf_counter() - t0)
            aus[(a, d)] = float(np.median(ts))
    return aus


# ---- Aufgabenliste ----
def _schl_v3(a):
    """Aufgabe (teil, arm, d, T, seed, lr, klein, ckpt_alle, ckordner) -> Schluessel der Zwischendatei."""
    return f"einheit|{a[0]}|{a[1]}|d{a[2]}|T{a[3]}|{a[4]}"


def _schl_lr(a):
    """Tuning-Aufgabe (schl, seed, lr, d, T, klein, ckpt_alle, ckordner, arm)."""
    return f"lr|{a[0]}|{a[2]!r}|{a[1]}"


def _kosten_aufgabe(arm, d, T):
    return T * S_JE_SCHRITT.get((arm, d), 1.0) if S_JE_SCHRITT else T * d


def aufgaben_v3(hp, P, klein, ckordner):
    """Mess- und NK-Aufgaben, die langen (d = 512, T = 16000) zuerst."""
    aufg = [("mess", a, d, T, s, hp[f"{a}|{d}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
            for a in ARME_V3 for d in P["d"] for T in P["T"] for s in P["seeds"]]
    aufg += [("nk", a, P["nk_d"], P["nk_T"], s, hp[f"{a}|{P['nk_d']}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
             for a in NK_ARME_V3 for s in P["nk_seeds"]]
    aufg.sort(key=lambda a: -_kosten_aufgabe(a[1], a[2], a[3]))
    return aufg


def kosten_schaetzung(s_je_schritt=None, prozesse=72, aufschlag=1.3, tuning_raten=4.5, tuning_runden=2, ueberhang=30.0):
    """Kernstunden und Wandzeit (LPT-Planung der Aufgaben auf `prozesse` Kerne). aufschlag: Parallel-Aufschlag (Cache/Speicherbandbreite bei vollem Knoten,
    I-045 beobachtete 1,2-1,5). Tuning: im Mittel `tuning_raten` Raten je (Arm, d) in `tuning_runden` Runden (3 + Erweiterung); ueberhang: Auswertung/Laden je Aufgabe (s)."""
    z = s_je_schritt or S_JE_SCHRITT
    P = PROFILE_V3["standard"]
    dauer = lambda a, d, T: T * z[(a, d)] * aufschlag + ueberhang
    # Tuning: Runde 1 = 3 Raten je (Arm, d); weitere Runden nur Erweiterungen (tuning_raten − 3 im Mittel je Paar)
    t_aufg = [dauer(a, d, P["tuning_T"]) for a in ARME_V3 for d in P["d"]]
    kern_tuning = sum(t_aufg) * tuning_raten
    wand_tuning = tuning_runden * max(t_aufg)
    m_aufg = [dauer(a, d, T) for a in ARME_V3 for d in P["d"] for T in P["T"] for _ in P["seeds"]]
    m_aufg += [dauer(a, P["nk_d"], P["nk_T"]) for a in NK_ARME_V3 for _ in P["nk_seeds"]]
    last = [0.0] * prozesse
    import heapq
    heapq.heapify(last)
    for t in sorted(m_aufg, reverse=True):
        heapq.heappush(last, heapq.heappop(last) + t)
    wand_mess = max(last)
    return {"aufgaben_mess": len(m_aufg), "kernstunden_mess": sum(m_aufg) / 3600, "kernstunden_tuning": kern_tuning / 3600,
            "kernstunden_gesamt": (sum(m_aufg) + kern_tuning) / 3600, "laengste_aufgabe_h": max(m_aufg) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand_mess / 3600, "wand_gesamt_h": (wand_tuning + wand_mess) / 3600,
            "prozesse": prozesse, "aufschlag": aufschlag}


# ---- Arbeiter ----
def _arbeiter_v3(a):
    teil, arm, d, T, seed, lr, klein, ck_alle, ckordner = a
    key = _schl_v3(a)
    r = trainiere_v3(arm, lr, seed, T, d=d, klein=klein, permutiert=(teil == "nk"), ckpt=_ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"teil": teil, "arm": arm, "d": d, "T": T, "seed": seed, "lr": lr, **r}


_ZAEHLER = {"lr": 0}


def _lr_arbeiter_v3(a):
    schl, seed, lr, d, T, klein, ck_alle, ckordner, arm = a
    _ZAEHLER["lr"] += 1
    key = _schl_lr(a)
    r = trainiere_v3(arm, lr, seed, T, d=d, klein=klein, val=True, ckpt=_ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"acc": 1 - r["bpc"] / 8, "bpc": r["bpc"], "instabil": r["instabil"]}      # 'acc' = Score (hoeher besser) fuer gittersuche_stabil


def tuning_v3(P, klein, tseed, pfad, kennung, prozesse, ckordner, arme=None):
    """Lernrate je (Arm, d) bei T = tuning_T auf dem Tuning-Seed; alle Raten aller Paare einer Runde laufen gemeinsam parallel; jede Aufgabe einzeln wiederaufnehmbar."""
    schl = [f"{a}|{d}" for a in (arme or ARME_V3) for d in P["d"]]

    def aufgaben_fuer(s, lr):
        a, d = s.split("|")
        return [(s, tseed, lr, int(d), P["tuning_T"], klein, P["ckpt_alle"], str(ckordner), a)]

    def auswerten(aufg):
        lang = sorted(aufg, key=lambda a: -a[3])
        res = _rechne(lang, pfad, kennung, fn=_lr_arbeiter_v3, schluessel=_schl_lr, prozesse=prozesse)
        return [res[_schl_lr(a)] for a in aufg]

    info, _ = gittersuche_stabil(schl, aufgaben_fuer, auswerten, gitter=LR_GITTER, faktor=LR_FAKTOR, kollaps=KOLLAPS_SCORE)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _zeit_arbeiter_v3(paare):
    torch.set_num_threads(1)
    return {f"{a}|{d}": zeit_je_token(Sprachmodell(a, 0, d=d, h=d // 64)) for a, d in paare}


def zeit_im_ein_thread_v3(paare):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter_v3, (tuple(paare),))


# ---- Auswertung (bpc: kleiner ist besser; V = D_z2 − Sz, positiv = Sz besser) ----
def _zname(a, T, d):
    return f"{a}|T{T}|d{d}"


def auswertung_v3(zeilen: list, params: dict, unigramm: float, t_lo=T_WERTE[0], t_hi=T_WERTE[-1], ds=D_WERTE) -> dict:
    bpc, inst, tt = {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            k = (z["arm"], z["T"], z["d"])
            bpc.setdefault(k, {})[z["seed"]] = z["bpc"]
            inst.setdefault(k, {})[z["seed"]] = bool(z.get("instabil"))
            tt.setdefault(k, {})[z["seed"]] = z.get("sekunden_training", float("nan"))
    r = {"t_lo": t_lo, "t_hi": t_hi, "zellen": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {}, "bpc_ohne_divergente": {}, "negativ": {},
         "diff": {}, "p_holm": {}, "beschreibend": {}, "parameter": params, "unigramm_bpc": unigramm, "training_s_mittel": {}}
    for k, v in bpc.items():
        seeds = sorted(v)
        w = np.array([v[s] for s in seeds])
        n = _zname(*k)
        r["zellen"][n] = zusammenfassung(w)
        med = float(np.median(w))
        r["kollaps"][n] = [s for s in seeds if not (v[s] <= med + KOLLAPS_BPC)]
        r["divergiert"][n] = [s for s in seeds if inst[k][s] or not (v[s] < unigramm)]
        r["divergenzrate"][n] = len(r["divergiert"][n]) / len(seeds)
        ok = [v[s] for s in seeds if s not in r["divergiert"][n]]
        r["bpc_ohne_divergente"][n] = float(np.mean(ok)) if ok else None
        r["training_s_mittel"][n] = float(np.nanmean(list(tt[k].values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def vorsprung(T, d, ref="D_z2", pruef="Sz"):
        """Je Seed: bpc(ref) − bpc(pruef); positiv = pruef besser. {} wenn eine Zelle fehlt."""
        a, b = bpc.get((ref, T, d)), bpc.get((pruef, T, d))
        if not a or not b:
            return {}
        return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}

    for h, d in (("H1a", ds[0]), ("H1b", ds[-1])):
        v = vorsprung(t_hi, d)
        if v:
            r["diff"][h] = _stat(np.array(list(v.values())))
    r["beschreibend"]["V_T_d"] = {}
    for d in ds:
        for T in sorted({k[1] for k in bpc}):
            v = vorsprung(T, d)
            if v:
                r["beschreibend"]["V_T_d"][f"T{T}|d{d}"] = _stat(np.array(list(v.values())))
    # H2: Trend des Vorsprungs, je Seed ueber d gemittelt (gleiche Seeds in allen vier Zellen noetig)
    zeit = [(T, d) for d in ds for T in (t_lo, t_hi)]
    vs = {(T, d): vorsprung(T, d) for T, d in zeit}
    if all(vs[x] for x in zeit):
        gem = sorted(set.intersection(*(set(vs[x]) for x in zeit)))
        dl = np.array([np.mean([vs[(t_hi, d)][s] - vs[(t_lo, d)][s] for d in ds]) for s in gem])
        r["diff"]["H2"] = _stat(dl)
        for d in ds:
            dd = np.array([vs[(t_hi, d)][s] - vs[(t_lo, d)][s] for s in gem])
            r["beschreibend"][f"Trend_V_d{d}"] = _stat(dd)
    for ref in ("D_gelu", "D_relu2"):
        for d in ds:
            for T in sorted({k[1] for k in bpc}):
                v = vorsprung(T, d, ref=ref)
                if v:
                    r["beschreibend"].setdefault(f"{ref}-Sz", {})[f"T{T}|d{d}"] = _stat(np.array(list(v.values())))
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1a", "H1b") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _vorab_v3(r, nk, h, zellen):
    if h not in r["diff"] or "mittel" not in r["diff"][h] or (h in ("H1a", "H1b") and h not in r["p_holm"]):
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    dv = {z: r["divergiert"][z] for z in zellen if r["divergiert"].get(z)}
    if dv:
        return f"HU (Divergenz in {dv})"
    k = {z: r["kollaps"][z] for z in zellen if r["kollaps"].get(z)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def urteil_v3(r: dict, ds=D_WERTE) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert D_z2/Sz: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {NK_FAKTOR}x)"}
    t_lo, t_hi = r["t_lo"], r["t_hi"]
    for h, d in (("H1a", ds[0]), ("H1b", ds[-1])):
        v = _vorab_v3(r, nk, h, [_zname(a, t_hi, d) for a in ("D_z2", "Sz")])
        if v:
            u[h] = v
            continue
        z = _zahlen(r, h)
        if r["p_holm"][h] < ALPHA:
            u[h] = f"SZ_BESSER (D_z2-Sz bei T={t_hi}, d={d}: {z})"
        elif r["diff"][h]["p_kl"] < ALPHA / 2:
            u[h] = f"H0 (D_z2 sicher besser als Sz bei T={t_hi}, d={d}; {z})"
        else:
            u[h] = f"HU ({z})"
    v = _vorab_v3(r, nk, "H2", [_zname(a, T, d) for a in ("D_z2", "Sz") for T in (t_lo, t_hi) for d in ds])
    if v:
        u["H2"] = v
    else:
        s = r["diff"]["H2"]
        z = (f"Δ(V{t_hi} − V{t_lo}) {s['mittel']:+.4f} bpc, 90-%-KI [{s['ki'][0]:+.4f}; {s['ki'][1]:+.4f}], n {s['n']}, p_zweiseitig {s['p_zwei']:.3g}").replace(".", ",")
        if s["p_zwei"] < ALPHA:
            u["H2"] = f"{'SCHRUMPFT' if s['mittel'] < 0 else 'WAECHST'} ({z})"
        else:
            u["H2"] = f"HU ({z})"
    teile = []
    for d in ds:
        for T in sorted({int(k.split("|T")[1].split("|")[0]) for k in r["zellen"]}):
            ze = ", ".join(f"{a} {r['zellen'][_zname(a, T, d)]['mittel']:.3f}".replace(".", ",") for a in ARME_V3 if _zname(a, T, d) in r["zellen"])
            teile.append(f"d{d} T{T}: {ze}")
    u["B"] = "beschreibend: Test-bpc " + "; ".join(teile) + "; Rest in auswertung"
    return u


# ---- Lauf ----
def lauf_v3(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I045V3_PROFIL", "standard")
    P = dict(PROFILE_V3[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte (Checkpoint alle 15 Schritte)
        P.update({"T": (10, 20, 40), "tuning_T": 20, "nk_T": 10, "ckpt_alle": 15, "seeds": (TUNING_SEED_V3,), "nk_seeds": (TUNING_SEED_V3,)})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT_V3 or set(P["nk_seeds"]) & _BELEGT_V3 or TUNING_SEED_V3 in P["seeds"] or TUNING_SEED_V3 in _BELEGT_V3:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V3}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE_V3}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE_V3, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": _json_sicher(P),
               "lr_gitter": list(LR_GITTER), "tuning_seed": TUNING_SEED_V3, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        hp0 = tuning_v3(P, klein, TUNING_SEED_V3, pfad, kennung, prozesse, ckordner)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning"] = hp0
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = aufgaben_v3(hp, P, klein, ckordner)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_v3, schluessel=_schl_v3, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    paare = [(a, d) for a in ARME_V3 for d in P["d"]]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread_v3(paare)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {f"{a}|{d}": {"ffn": param_ffn(n), "gesamt": param_gesamt(n)}
              for a, d in paare for n in [Sprachmodell(a, 0, d=d, h=d // 64)]}
    tr, _, te = daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_sprache", "karte": KARTE_V3, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_V3,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": TUNING_SEED_V3, "tuning": hp,
         "einstellungen": {"T": list(P["T"]), "d": list(P["d"]), "tuning_T": P["tuning_T"], "nk_T": P["nk_T"], "nk_d": P["nk_d"], "ckpt_alle": P["ckpt_alle"],
                           "batch": BATCH, "kontext": KONTEXT, "lagen": LAGEN, "ffn_breite": "4*d", "lr_gitter": list(LR_GITTER),
                           "daten_bytes": [len(x) for x in daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_v3(zeilen, params, unigramm_entropie(tr, te), t_lo=min(P["T"]), t_hi=max(P["T"]), ds=tuple(P["d"]))
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
    e["urteil"] = urteil_v3(e["auswertung"], ds=tuple(P["d"]))
    return e


# ---- Vortest ----
def _synth_v3(rng, adv_lo=0.08, adv_hi=0.04, sd=0.004, n=8, koll=None, nk=5.0, div=None, fehlt=None):
    """Synthetische Zeilen: V = D_z2 − Sz = adv_lo bei T=1000, adv_hi bei T=16000 (T=4000 mittig), in beiden d."""
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for d in D_WERTE:
            for T in T_WERTE:
                adv = adv_lo + (adv_hi - adv_lo) * {1000: 0.0, 4000: 0.5, 16000: 1.0}[T]
                basis = {"Sz": 3.40 - 0.1 * (d == 512), "D_z2": 3.40 + adv - 0.1 * (d == 512), "D_gelu": 3.50, "D_relu2": 3.45}
                for a, v in basis.items():
                    if fehlt == (a, T, d):
                        continue
                    x, ist = v + gem + rng.normal(0, sd), False
                    if koll == (a, T, d) and i == 0:
                        x += 1.0
                    if div == (a, T, d) and i == 1:
                        x, ist = 8.0, True
                    zl.append({"teil": "mess", "arm": a, "d": d, "T": T, "seed": 2600 + i, "bpc": x, "instabil": ist, "sekunden_training": 10.0})
        if i < 2:
            for a in NK_ARME_V3:
                zl.append({"teil": "nk", "arm": a, "d": 256, "T": 1000, "seed": 2608 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u3(rng, **kw):
    p = {f"{a}|{d}": {"ffn": 1, "gesamt": 2} for a in ARME_V3 for d in D_WERTE}
    return urteil_v3(auswertung_v3(_synth_v3(rng, **kw), p, 5.0))


def _fn_test_v3(a):
    return _schl_v3(a), {"v": a[4] * 2}


def _vortest_v3() -> bool:
    import tempfile
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-045-v3): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    # Form, Kausalitaet, Koepfe, Parameter, Aktivierungen bei beiden Breiten
    for d in D_WERTE:
        x = torch.randint(0, 256, (2, KONTEXT))
        nz = {a: Sprachmodell(a, 0, d=d, h=d // 64) for a in ARME_V3}
        for a, m in nz.items():
            m.eval()
            x2 = x.clone()
            x2[:, 60] = (x2[:, 60] + 1) % 256
            pruefe(f"Form/Kausalitaet {a} d={d}", m(x).shape == (2, KONTEXT, 256) and torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5)
                   and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
            pruefe(f"Koepfe d/64 {a} d={d}", m.bl[0].h == d // 64)
        pf = {a: param_ffn(n) for a, n in nz.items()}
        dicht = 4 * (2 * d * 4 * d + 4 * d + d)
        pruefe(f"Parameter dicht = 4·(8d²+5d), d={d}", pf["D_gelu"] == pf["D_z2"] == pf["D_relu2"] == dicht)
        pruefe(f"Parameter Sz = 4·21d (Fastfood 8d + 8d Diagonalen + Bias 5d), d={d}", pf["Sz"] == 4 * 21 * d)
        pruefe(f"FFN-Breite 4·d, d={d}", nz["D_gelu"].bl[0].ffn.a.out_features == 4 * d and nz["Sz"].bl[0].ffn.a.m.N == 4 * d
               and nz["Sz"].bl[0].ffn.b.m.N == d)
        pruefe(f"Aktivierungen d={d}", nz["D_gelu"].bl[0].ffn.akt is F.gelu and nz["D_z2"].bl[0].ffn.akt is torch.square
               and nz["Sz"].bl[0].ffn.akt is torch.square and nz["D_relu2"].bl[0].ffn.akt is _relu2)
    pruefe("d = 256 unveraendert zu v2 (Sz, D_z2 gleiche Parameterzahl und Breite 1024)",
           param_ffn(Sprachmodell("Sz", 0)) == param_ffn(Sprachmodell("Sz", 0, d=256, h=4)) and BREITE["Sz"] == 1024)
    # Trainierbarkeit
    unit = torch.arange(32).repeat(8)[None].repeat(16, 1)[:, :KONTEXT + 1]
    for a in ARME_V3:
        m = Sprachmodell(a, 0, d=512, h=8)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        l0 = None
        for _ in range(25):
            opt.zero_grad()
            l = F.cross_entropy(m(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a} d=512", float(l.detach()) < 0.7 * l0)
    # Seeds, Profil, Aufgaben
    pruefe("Seeds frisch (8 Mess, 2 NK, Tuning 2698 neu)", len(SEEDS_V3) == 8 and len(NK_SEEDS_V3) == 2 and not ((set(SEEDS_V3) | set(NK_SEEDS_V3)) & _BELEGT_V3)
           and TUNING_SEED_V3 not in _BELEGT_V3 and TUNING_SEED_V3 not in SEEDS_V3 and not (set(SEEDS_V3) & set(NK_SEEDS_V3)))
    hp = {f"{a}|{d}": {"lr": 1e-3} for a in ARME_V3 for d in D_WERTE}
    P = PROFILE_V3["standard"]
    aufg = aufgaben_v3(hp, P, False, "x")
    schl = [_schl_v3(a) for a in aufg]
    pruefe("Aufgabenzahl 4·3·2·8 + 4 NK = 196, Schluessel eindeutig", len(aufg) == 196 and len(set(schl)) == 196 and sum(a[0] == "nk" for a in aufg) == 4)
    pruefe("Lange zuerst (d=512, T=16000), NK d=256 T=1000", aufg[0][2] == 512 and aufg[0][3] == 16000 and all(a[2] == 512 and a[3] == 16000 for a in aufg[:32])
           and all(a[2] == 256 and a[3] == 1000 for a in aufg if a[0] == "nk"))
    # Laufzeitplan (24-h-Joblimit, Messwerte aus dem Container)
    if S_JE_SCHRITT:
        k = kosten_schaetzung()
        pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (mit Aufschlag)", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    # Divergenz-Pfad
    t = trainiere_v3("Sz", 1e9, 0, 5, d=256, klein=True, val=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)
    # Checkpoint in der Aufgabe: Abbruch bei Schritt 15 (Checkpoint bei 10) -> Fortsetzung ergibt DASSELBE Ergebnis wie ungestoerter Lauf
    with tempfile.TemporaryDirectory() as tmp:
        ck = str(Path(tmp) / "a.pt")
        ref = trainiere_v3("Sz", 2e-3, 3, 30, d=256, klein=True, val=True)
        try:
            trainiere_v3("Sz", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=15)
            pruefe("Abbruch wurde ausgeloest", False)
        except _Abbruch:
            pass
        pruefe("Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        fort = trainiere_v3("Sz", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Fortsetzung ab Schritt 10", fort["fortgesetzt_ab"] == 10)
        pruefe("Fortsetzung = ungestoerter Lauf (bpc identisch)", abs(fort["bpc"] - ref["bpc"]) < 1e-9)
        pruefe("Checkpoint nach Ende geloescht", not Path(ck).exists())
        # fremder Checkpoint (andere Lernrate) wird ignoriert
        try:
            trainiere_v3("D_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=12)
        except _Abbruch:
            pass
        anders = trainiere_v3("D_z2", 1e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Checkpoint mit anderer Lernrate wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        # Wiederaufnahme je Aufgabe (Zwischendatei) und Tuning-Wiederaufnahme
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [("mess", a, d, 10, s, 1e-3, True, 5, tmp) for a in ("Sz", "D_z2") for d in (256, 512) for s in (1, 2)]
        z1 = _rechne(a4[:3], pfad, kennung, fn=_fn_test_v3, prozesse=1, schluessel=_schl_v3)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test_v3, prozesse=1, schluessel=_schl_v3)
        pruefe("Wiederaufnahme je Aufgabe (Arm, d, T, Seed)", len(z1) == 3 and len(z2) == 8 and all(z2[k] == z1[k] for k in z1))
        try:
            _rechne(a4, pfad, {"k": 2}, fn=_fn_test_v3, prozesse=1, schluessel=_schl_v3)
            pruefe("Fremde Zwischendatei wird abgelehnt", False)
        except ValueError:
            pass
        Pk = {**PROFILE_V3["standard"], "d": (256,), "tuning_T": 3, "ckpt_alle": 2}
        pt = Path(tmp) / "t.json"
        kt = {"karte": "t"}
        n0 = _ZAEHLER["lr"]
        t1 = tuning_v3(Pk, True, TUNING_SEED_V3, pt, kt, 1, Path(tmp) / "ck")
        n1 = _ZAEHLER["lr"]
        t2 = tuning_v3(Pk, True, TUNING_SEED_V3, pt, kt, 1, Path(tmp) / "ck")
        pruefe("Tuning: Raten je (Arm, d) gewaehlt", set(t1) == {f"{a}|256" for a in ARME_V3} and all(v["lr"] > 0 for v in t1.values()))
        pruefe("Tuning-Wiederaufnahme rechnet nichts neu und liefert dasselbe", n1 > n0 and _ZAEHLER["lr"] == n1 and t1 == t2)
    # Urteilszweige
    rng = np.random.default_rng(11)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER (I-045-v3) {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    erwarte("Vorsprung schrumpft", _u3(rng), **{"NK": "BESTANDEN", "H1a": "SZ_BESSER", "H1b": "SZ_BESSER", "H2": "SCHRUMPFT"})
    erwarte("Vorsprung waechst", _u3(rng, adv_lo=0.02, adv_hi=0.08), **{"H2": "WAECHST", "H1a": "SZ_BESSER"})
    erwarte("D_z2 besser", _u3(rng, adv_lo=-0.07, adv_hi=-0.07), **{"H1a": "H0", "H1b": "H0", "H2": "HU"})
    erwarte("Kein Unterschied", _u3(rng, adv_lo=0.0, adv_hi=0.0, sd=0.03), **{"H1a": "HU", "H1b": "HU", "H2": "HU"})
    erwarte("NK faellt", _u3(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1a": "HU (Negativkontrolle", "H1b": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle"})
    erwarte("Divergenz Sz d=512 T=16000", _u3(rng, div=("Sz", 16000, 512)), **{"H1a": "SZ_BESSER", "H1b": "HU (Divergenz", "H2": "HU (Divergenz"})
    erwarte("Kollaps D_z2 d=256 T=16000", _u3(rng, koll=("D_z2", 16000, 256)), **{"H1a": "HU (Kollaps", "H1b": "SZ_BESSER", "H2": "HU (Kollaps"})
    erwarte("Divergenz in T=4000 stoert die Hypothesen nicht", _u3(rng, div=("Sz", 4000, 256)), **{"H1a": "SZ_BESSER", "H2": "SCHRUMPFT"})
    erwarte("Zelle fehlt", _u3(rng, fehlt=("Sz", 16000, 512)), **{"H1a": "SZ_BESSER", "H1b": "HU (Zellen fehlen", "H2": "HU (Zellen fehlen"})
    p = {f"{a}|{d}": {"ffn": 1, "gesamt": 2} for a in ARME_V3 for d in D_WERTE}
    a_ = auswertung_v3(_synth_v3(rng, div=("Sz", 16000, 256)), p, 5.0)
    pruefe("Divergenzrate 1/8, Seed bleibt in den Zahlen (n = 8)", abs(a_["divergenzrate"]["Sz|T16000|d256"] - 0.125) < 1e-9
           and a_["zellen"]["Sz|T16000|d256"]["n"] == 8 and a_["diff"]["H1a"]["n"] == 8)
    pruefe("Holm nur ueber H1a, H1b (H2 nicht)", set(a_["p_holm"]) == {"H1a", "H1b"})
    erlaubt = ("SZ_BESSER", "SCHRUMPFT", "WAECHST", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u3(rng).values()))
    soll = {"H1a": {"SZ_BESSER", "H0", "HU"}, "H1b": {"SZ_BESSER", "H0", "HU"}, "H2": {"SCHRUMPFT", "WAECHST", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k, w in gesehen if k == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    return bool(ok)


# ================================================================== I-045-v4 „Budgetgleich: Struktur gegen kleine dichte FFN“
# VORREGISTRIERUNG: INSTITUT/LABORBUCH.md (Eintrag I-045-v4, geaendert 26.10. vor dem Start: Hauptgegner FLOP-gleich statt parametergleich).
# FRAGE: Ist Sz (Fastfood + z², Breite 4d) bei GLEICHEM FFN-Rechenaufwand (FLOPs/Token) besser als eine dichte bzw. niedrigrangige FFN?
# MODELL: d = 512 (8 Koepfe), 4 Lagen, Kontext 128, Batch 8x128, Adam, Kosinus, 5 % Aufwaermen, Clip 1,0. T = 16000 (Hauptpunkt), T = 4000 (beschreibend).
# ARME (nur die FFN verschieden): Sz Fastfood+z² (4d) | Df_z2 dicht+z², h=580 (FLOP-gleich) | Df_gelu dicht+GELU, h=580 | Lk_z2 Rang 116 + z², Breite 4d (FLOP-gleich)
#   | Dk_z2 dicht+z², h=10 (parametergleich, nur beschreibend) | D_z2 dicht+z² Breite 4d (Referenz, Obergrenze). h/Rang aus form_v4() (FLOP-Zaehlung), Toleranz +-3 % im Vortest.
# HYPOTHESEN (Holm ueber H1, H2; gepaarter einseitiger t, n = 8, T = 16000, Differenz Referenz − Sz je Seed in bpc, positiv = Sz besser):
#   H1 Sz < Df_z2 | H2 Sz < Lk_z2: STRUKTUR_BESSER | H0 (Referenz sicher besser) | HU.
# DIVERGENZ wie v3 (je Zelle Arm x T): nicht endlicher Verlust oder End-bpc >= Unigramm-Entropie; Kollaps = Median + 0,2 bpc; beteiligte Hypothese = HU.
# NK wie v3: Sz und Df_z2, d = 512, 1000 Schritte, permutierte Trainingsbytes, Seeds 2628/2629; mittlerer Test-bpc >= 0,9 x Unigramm.
# BESCHREIBEND: Abstand jedes Arms zu D_z2, Vorsprung gegen alle Arme (T = 4000 und 16000), Pareto bpc x FFN-Parameter x Zeit/Token, FLOPs/Token, Inferenzzeit/Token.
# ERWARTUNG VORAB (kein Befund): offen; v3 spricht eher gegen Sz (H1 eher H0 oder HU), Sz hat aber etwa 55-mal weniger Parameter als Df_z2.
KARTE_V4 = "I-045-v4"
SEEDS_V4 = tuple(range(2620, 2628))
NK_SEEDS_V4 = (2628, 2629)
TUNING_SEED_V4 = 2699
ARME_MESS_V4 = ("Sz", "Df_z2", "Df_gelu", "Lk_z2", "Dk_z2", "D_z2")
NK_ARME_V4 = ("Sz", "Df_z2")
T_WERTE_V4 = (4000, 16000)
D_V4 = 512
BUDGET_TOLERANZ = 0.03
PROFILE_V4 = {"standard": {"T": T_WERTE_V4, "d": (D_V4,), "tuning_T": 4000, "nk_T": 1000, "nk_d": D_V4, "ckpt_alle": CKPT_ALLE,
                           "seeds": SEEDS_V4, "nk_seeds": NK_SEEDS_V4}}
_BELEGT_V4 = _BELEGT_V3 | set(range(2600, 2610)) | {2698}
HYPOTHESEN_V4 = {
    "H1": "Sz < Df_z2 (bpc, gleiche FFN-FLOPs/Token) bei T=16000, d=512 (einseitig gepaart t, n=8, Holm H1-H2). STRUKTUR_BESSER | H0 (Df_z2 sicher besser) | HU.",
    "H2": "Sz < Lk_z2 (bpc, gleiche FFN-FLOPs/Token, Rang aus dem FLOP-Budget) bei T=16000, d=512 (einseitig gepaart t, n=8, Holm H1-H2). STRUKTUR_BESSER | H0 (Lk_z2 sicher besser) | HU.",
    "DIVERGENZ": HYPOTHESEN_V3["DIVERGENZ"],
    "NK": f"Negativkontrolle (Sz, Df_z2, d=512, 1000 Schritte, permutierte Trainingsbytes, Seeds 2628/2629): mittlerer Test-bpc >= {NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc aller Arme x T, Abstand zu D_z2, Vorsprung gegen alle Arme, Pareto bpc x FFN-Parameter x Zeit/Token, FLOPs/Token, Inferenzzeit/Token, Trainingszeit, Divergenzrate, Lernraten.",
}
# Sekunden je Trainingsschritt (1 Thread, Batch 8x128), Messung im Container (Datum im LABORBUCH); wird nach Messung gefuellt.
S_JE_SCHRITT_V4 = {"Sz": 0.79, "Df_z2": 0.49, "Df_gelu": 0.49, "Lk_z2": 0.63, "Dk_z2": 0.37, "D_z2": 1.03}   # Sz/D_z2 konservativ aus v3 (0,79/1,03; v4-Messung 0,67/0,89)


def flops_je_token(arm, d=D_V4, lagen=LAGEN, ctx=KONTEXT) -> dict:
    """Rechenaufwand der Inferenz je Token (Vorwaerts, 1 Flop = 1 Multiplikation oder Addition) AS IMPLEMENTED (Fastfood = zwei kleine Matmuls je Walsh-Transformation).
    ffn: alle Lagen; gesamt: zusaetzlich Attention (qkv, Ausgabe, im Mittel ctx/2 Schluessel) und Ausgabekopf 256·d."""
    netz = Sprachmodell(arm, 0, d=d, h=d // 64, lagen=1)
    ffn = _ffn_flops_modul(netz.bl[0].ffn)
    ffn *= lagen
    att = lagen * (2 * d * 3 * d + 2 * d * d + 2 * ctx * d)
    return {"ffn": int(ffn), "gesamt": int(ffn + att + 2 * 256 * d)}


def kosten_schaetzung_v4(s_je_schritt=None, prozesse=72, aufschlag=1.3, tuning_raten=4.5, tuning_runden=2, ueberhang=30.0):
    """Wie kosten_schaetzung (v3), fuer das v4-Profil."""
    import heapq
    z = s_je_schritt or S_JE_SCHRITT_V4
    P = PROFILE_V4["standard"]
    dauer = lambda a, T: T * z[a] * aufschlag + ueberhang
    t_aufg = [dauer(a, P["tuning_T"]) for a in ARME_MESS_V4]
    kern_tuning = sum(t_aufg) * tuning_raten
    wand_tuning = tuning_runden * max(t_aufg)
    m_aufg = [dauer(a, T) for a in ARME_MESS_V4 for T in P["T"] for _ in P["seeds"]]
    m_aufg += [dauer(a, P["nk_T"]) for a in NK_ARME_V4 for _ in P["nk_seeds"]]
    last = [0.0] * prozesse
    heapq.heapify(last)
    for t in sorted(m_aufg, reverse=True):
        heapq.heappush(last, heapq.heappop(last) + t)
    wand_mess = max(last)
    return {"aufgaben_mess": len(m_aufg), "kernstunden_mess": sum(m_aufg) / 3600, "kernstunden_tuning": kern_tuning / 3600,
            "kernstunden_gesamt": (sum(m_aufg) + kern_tuning) / 3600, "laengste_aufgabe_h": max(m_aufg) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand_mess / 3600, "wand_gesamt_h": (wand_tuning + wand_mess) / 3600,
            "prozesse": prozesse, "aufschlag": aufschlag}


def aufgaben_v4(hp, P, klein, ckordner):
    """Mess- und NK-Aufgaben, die langen (T = 16000, teuerste Arme) zuerst."""
    d = P["d"][0]
    aufg = [("mess", a, d, T, s, hp[f"{a}|{d}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
            for a in ARME_MESS_V4 for T in P["T"] for s in P["seeds"]]
    aufg += [("nk", a, P["nk_d"], P["nk_T"], s, hp[f"{a}|{P['nk_d']}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
             for a in NK_ARME_V4 for s in P["nk_seeds"]]
    kost = lambda a: a[3] * (S_JE_SCHRITT_V4.get(a[1], 1.0) if S_JE_SCHRITT_V4 else 1.0)
    aufg.sort(key=lambda a: -kost(a))
    return aufg


def pareto_nicht_dominiert(punkte: dict) -> list:
    """punkte: Arm -> (bpc, ffn_parameter, zeit_je_token); nicht dominiert = kein anderer Arm ist in allen drei <= und in mindestens einem <."""
    aus = []
    for a, p in punkte.items():
        if not any(b != a and all(x <= y for x, y in zip(q, p)) and any(x < y for x, y in zip(q, p)) for b, q in punkte.items()):
            aus.append(a)
    return sorted(aus)


def auswertung_v4(zeilen: list, params: dict, unigramm: float, t_lo=T_WERTE_V4[0], t_hi=T_WERTE_V4[-1], d=D_V4, flops=None, zeit=None) -> dict:
    bpc, inst, tt = {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            k = (z["arm"], z["T"], z["d"])
            bpc.setdefault(k, {})[z["seed"]] = z["bpc"]
            inst.setdefault(k, {})[z["seed"]] = bool(z.get("instabil"))
            tt.setdefault(k, {})[z["seed"]] = z.get("sekunden_training", float("nan"))
    r = {"t_lo": t_lo, "t_hi": t_hi, "d": d, "zellen": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {}, "bpc_ohne_divergente": {}, "negativ": {},
         "diff": {}, "p_holm": {}, "beschreibend": {}, "parameter": params, "unigramm_bpc": unigramm, "training_s_mittel": {}}
    for k, v in bpc.items():
        seeds = sorted(v)
        w = np.array([v[s] for s in seeds])
        n = _zname(*k)
        r["zellen"][n] = zusammenfassung(w)
        med = float(np.median(w))
        r["kollaps"][n] = [s for s in seeds if not (v[s] <= med + KOLLAPS_BPC)]
        r["divergiert"][n] = [s for s in seeds if inst[k][s] or not (v[s] < unigramm)]
        r["divergenzrate"][n] = len(r["divergiert"][n]) / len(seeds)
        ok = [v[s] for s in seeds if s not in r["divergiert"][n]]
        r["bpc_ohne_divergente"][n] = float(np.mean(ok)) if ok else None
        r["training_s_mittel"][n] = float(np.nanmean(list(tt[k].values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def vorsprung(T, ref, pruef="Sz"):
        """Je Seed: bpc(ref) − bpc(pruef); positiv = pruef besser. {} wenn eine Zelle fehlt."""
        a, b = bpc.get((ref, T, d)), bpc.get((pruef, T, d))
        if not a or not b:
            return {}
        return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}

    for h, ref in (("H1", "Df_z2"), ("H2", "Lk_z2")):
        v = vorsprung(t_hi, ref)
        if v:
            r["diff"][h] = _stat(np.array(list(v.values())))
    for ref in ("Df_z2", "Df_gelu", "Lk_z2", "Dk_z2", "D_z2"):
        for T in (t_lo, t_hi):
            v = vorsprung(T, ref)
            if v:
                r["beschreibend"].setdefault(f"Vorsprung_Sz_gegen_{ref}", {})[f"T{T}"] = _stat(np.array(list(v.values())))
    for a in ARME_MESS_V4:               # Abstand zur Obergrenze: bpc(Arm) − bpc(D_z2), positiv = Arm schlechter als volle dichte FFN
        if a == "D_z2":
            continue
        for T in (t_lo, t_hi):
            v = vorsprung(T, a, pruef="D_z2")
            if v:
                r["beschreibend"].setdefault(f"Abstand_{a}_zu_D_z2", {})[f"T{T}"] = _stat(np.array(list(v.values())))
    # Pareto (T = t_hi): bpc x FFN-Parameter x Zeit/Token (Zeit nur, wenn gemessen)
    pt = {}
    for a in ARME_MESS_V4:
        n = _zname(a, t_hi, d)
        pk = f"{a}|{d}"
        if n in r["zellen"] and pk in params and (zeit is None or pk in zeit):
            pt[a] = (r["zellen"][n]["mittel"], params[pk]["ffn"], (zeit or {}).get(pk, 0.0))
    r["beschreibend"]["pareto_T_hi"] = {"punkte": {a: list(p) for a, p in pt.items()}, "nicht_dominiert": pareto_nicht_dominiert(pt),
                                        "achsen": ["bpc", "ffn_parameter", "s_je_token" if zeit else "(Zeit nicht gemessen: 0)"]}
    if flops:
        r["beschreibend"]["flops_je_token"] = flops
    if zeit:
        r["beschreibend"]["zeit_s_je_token"] = zeit
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1", "H2") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _vorab_v4(r, nk, h, zellen):
    if h not in r["diff"] or "mittel" not in r["diff"][h] or h not in r["p_holm"]:
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    dv = {z: r["divergiert"][z] for z in zellen if r["divergiert"].get(z)}
    if dv:
        return f"HU (Divergenz in {dv})"
    k = {z: r["kollaps"][z] for z in zellen if r["kollaps"].get(z)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def urteil_v4(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert Sz/Df_z2: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {NK_FAKTOR}x)"}
    t_hi, d = r["t_hi"], r["d"]
    for h, ref in (("H1", "Df_z2"), ("H2", "Lk_z2")):
        v = _vorab_v4(r, nk, h, [_zname(a, t_hi, d) for a in (ref, "Sz")])
        if v:
            u[h] = v
            continue
        z = _zahlen(r, h)
        if r["p_holm"][h] < ALPHA:
            u[h] = f"STRUKTUR_BESSER ({ref}-Sz bei T={t_hi}, d={d}, gleiches Budget: {z})"
        elif r["diff"][h]["p_kl"] < ALPHA / 2:
            u[h] = f"H0 ({ref} sicher besser als Sz bei T={t_hi}, d={d}; {z})"
        else:
            u[h] = f"HU ({z})"
    teile = []
    for T in sorted({int(k.split("|T")[1].split("|")[0]) for k in r["zellen"]}):
        ze = ", ".join(f"{a} {r['zellen'][_zname(a, T, d)]['mittel']:.3f}".replace(".", ",") for a in ARME_MESS_V4 if _zname(a, T, d) in r["zellen"])
        teile.append(f"T{T}: {ze}")
    u["B"] = "beschreibend: Test-bpc " + "; ".join(teile) + "; Rest in auswertung"
    return u


def lauf_v4(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I045V4_PROFIL", "standard")
    P = dict(PROFILE_V4[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte (Checkpoint alle 15 Schritte)
        P.update({"T": (20, 40), "tuning_T": 20, "nk_T": 10, "ckpt_alle": 15, "seeds": (TUNING_SEED_V4,), "nk_seeds": (TUNING_SEED_V4,)})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT_V4 or set(P["nk_seeds"]) & _BELEGT_V4 or TUNING_SEED_V4 in P["seeds"] or TUNING_SEED_V4 in _BELEGT_V4:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V4}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE_V4}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE_V4, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": _json_sicher(P),
               "lr_gitter": list(LR_GITTER), "tuning_seed": TUNING_SEED_V4, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        hp0 = tuning_v3(P, klein, TUNING_SEED_V4, pfad, kennung, prozesse, ckordner, arme=ARME_MESS_V4)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning"] = hp0
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = aufgaben_v4(hp, P, klein, ckordner)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_v3, schluessel=_schl_v3, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    d = P["d"][0]
    paare = [(a, d) for a in ARME_MESS_V4]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread_v3(paare)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {f"{a}|{d}": {"ffn": param_ffn(n), "gesamt": param_gesamt(n), "form": (form_v4(a, d) if a in ARME_V4 else ("sz" if a == "Sz" else "dicht", 4 * d, None))}
              for a, d in paare for n in [Sprachmodell(a, 0, d=d, h=d // 64)]}
    fl = {f"{a}|{d}": flops_je_token(a, d) for a, d in paare}
    tr, _, te = daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_sprache", "karte": KARTE_V4, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_V4,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": TUNING_SEED_V4, "tuning": hp,
         "einstellungen": {"T": list(P["T"]), "d": d, "tuning_T": P["tuning_T"], "nk_T": P["nk_T"], "nk_d": P["nk_d"], "ckpt_alle": P["ckpt_alle"],
                           "batch": BATCH, "kontext": KONTEXT, "lagen": LAGEN, "form": {a: params[f"{a}|{d}"]["form"] for a in ARME_MESS_V4},
                           "lr_gitter": list(LR_GITTER), "daten_bytes": [len(x) for x in daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_v4(zeilen, params, unigramm_entropie(tr, te), t_lo=min(P["T"]), t_hi=max(P["T"]), d=d, flops=fl, zeit=zm)
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
    e["urteil"] = urteil_v4(e["auswertung"])
    return e


# ---- Vortest v4 ----
def _synth_v4(rng, adv1=0.10, adv2=0.10, sd=0.004, n=8, koll=None, nk=5.0, div=None, fehlt=None):
    """Synthetische Zeilen: Df_z2 = Sz + adv1, Lk_z2 = Sz + adv2 (positiv = Sz besser), bei T = 4000 und 16000."""
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for T in T_WERTE_V4:
            basis = {"Sz": 2.4, "Df_z2": 2.4 + adv1, "Df_gelu": 2.6, "Lk_z2": 2.4 + adv2, "Dk_z2": 2.9, "D_z2": 2.2}
            for a, v in basis.items():
                if fehlt == (a, T):
                    continue
                x, ist = v + gem + rng.normal(0, sd), False
                if koll == (a, T) and i == 0:
                    x += 1.0
                if div == (a, T) and i == 1:
                    x, ist = 8.0, True
                zl.append({"teil": "mess", "arm": a, "d": D_V4, "T": T, "seed": 2620 + i, "bpc": x, "instabil": ist, "sekunden_training": 10.0})
        if i < 2:
            for a in NK_ARME_V4:
                zl.append({"teil": "nk", "arm": a, "d": D_V4, "T": 1000, "seed": 2628 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u4(rng, **kw):
    p = {f"{a}|{D_V4}": {"ffn": 1, "gesamt": 2} for a in ARME_MESS_V4}
    return urteil_v4(auswertung_v4(_synth_v4(rng, **kw), p, 5.0))


def _vortest_v4() -> bool:
    import tempfile
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-045-v4): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    d = D_V4
    nz = {a: Sprachmodell(a, 0, d=d, h=d // 64) for a in ARME_MESS_V4}
    x = torch.randint(0, 256, (2, KONTEXT))
    for a, m in nz.items():
        m.eval()
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 256
        pruefe(f"Form/Kausalitaet {a}", m(x).shape == (2, KONTEXT, 256) and torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5)
               and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
        pruefe(f"Koepfe d/64 {a}", m.bl[0].h == d // 64)
    pf = {a: param_ffn(n) for a, n in nz.items()}
    pruefe("Sz-Parameter = 4·21·d (Zaehlung = Formel)", pf["Sz"] == LAGEN * sz_param_lage(d))
    pruefe("Formen exakt aus der Zaehlung: Dk h=10 (Parameter), Df h=580 (FLOPs), Lk Rang 116 bei Breite 2048 (FLOPs)",
           form_v4("Dk_z2", d) == ("dicht", 10, None) and form_v4("Df_z2", d) == ("dicht", 580, None) and form_v4("Lk_z2", d) == ("lr", 4 * d, 116))
    pruefe("Parameterformel = echte Zaehlung (Dk, Df, Lk)", pf["Dk_z2"] == LAGEN * ffn_param_lage("dicht", d, 10)
           and pf["Df_z2"] == LAGEN * ffn_param_lage("dicht", d, 580) and pf["Lk_z2"] == LAGEN * ffn_param_lage("lr", d, 4 * d, 116))
    pruefe("Df_gelu und Df_z2 gleiche Parameterzahl", pf["Df_gelu"] == pf["Df_z2"])
    pruefe("Dk parametergleich zu Sz (+-2 %)", abs(pf["Dk_z2"] / pf["Sz"] - 1) <= 0.02)
    fl0 = {a: flops_je_token(a, d) for a in ARME_MESS_V4}
    for a in ("Df_z2", "Df_gelu", "Lk_z2"):
        pruefe(f"FLOP-Gleichheit +-3 % {a} ({fl0[a]['ffn']} gegen Sz {fl0['Sz']['ffn']})", abs(fl0[a]["ffn"] / fl0["Sz"]["ffn"] - 1) <= BUDGET_TOLERANZ)
    pruefe("FLOPs Formel = Zaehlung (Df: 4·d·h, Lk: 4·r·(d+4d))", fl0["Df_z2"]["ffn"] == LAGEN * ffn_flops_lage("dicht", d, 580)
           and fl0["Lk_z2"]["ffn"] == LAGEN * ffn_flops_lage("lr", d, 4 * d, 116))
    pruefe("D_z2 voll = 4·(8d²+5d), Breite 4d", pf["D_z2"] == 4 * (2 * d * 4 * d + 4 * d + d) and nz["D_z2"].bl[0].ffn.a.out_features == 4 * d)
    pruefe("Breiten: Df dicht 580, Lk Rang 116 mit Breite 2048",
           nz["Df_z2"].bl[0].ffn.a.out_features == 580 and nz["Lk_z2"].bl[0].ffn.a.m[0].out_features == 116 and nz["Lk_z2"].bl[0].ffn.a.m[1].out_features == 4 * d)
    pruefe("Aktivierungen", nz["Df_z2"].bl[0].ffn.akt is torch.square and nz["Df_gelu"].bl[0].ffn.akt is F.gelu and nz["Dk_z2"].bl[0].ffn.akt is torch.square
           and nz["Lk_z2"].bl[0].ffn.akt is torch.square and nz["Sz"].bl[0].ffn.akt is torch.square and nz["D_z2"].bl[0].ffn.akt is torch.square)
    pruefe("v3-Arme unveraendert (Sz/D_z2 bei d=256: Breite 1024)", param_ffn(Sprachmodell("D_z2", 0)) == 4 * (2 * 256 * 1024 + 1024 + 256))
    fl = {a: flops_je_token(a, d) for a in ARME_MESS_V4}
    pruefe("FLOPs: D_z2 FFN = 4·2·2·d·4d", fl["D_z2"]["ffn"] == LAGEN * 4 * d * 4 * d and all(f["gesamt"] > f["ffn"] > 0 for f in fl.values()))
    unit = torch.arange(32).repeat(8)[None].repeat(16, 1)[:, :KONTEXT + 1]
    for a in ARME_MESS_V4:
        m = Sprachmodell(a, 0, d=d, h=8)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        l0 = None
        for _ in range(25):
            opt.zero_grad()
            l = F.cross_entropy(m(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a}", float(l.detach()) < 0.7 * l0)
    pruefe("Seeds frisch (8 Mess, 2 NK, Tuning 2699)", len(SEEDS_V4) == 8 and len(NK_SEEDS_V4) == 2 and not ((set(SEEDS_V4) | set(NK_SEEDS_V4)) & _BELEGT_V4)
           and TUNING_SEED_V4 not in _BELEGT_V4 and TUNING_SEED_V4 not in SEEDS_V4 and not (set(SEEDS_V4) & set(NK_SEEDS_V4)))
    hp = {f"{a}|{d}": {"lr": 1e-3} for a in ARME_MESS_V4}
    P = PROFILE_V4["standard"]
    aufg = aufgaben_v4(hp, P, False, "x")
    schl = [_schl_v3(a) for a in aufg]
    pruefe("Aufgabenzahl 6·2·8 + 4 NK = 100, Schluessel eindeutig", len(aufg) == 100 and len(set(schl)) == 100 and sum(a[0] == "nk" for a in aufg) == 4)
    pruefe("Lange zuerst (T=16000), NK d=512 T=1000", aufg[0][3] == 16000 and all(a[3] == 16000 for a in aufg[:48]) and all(a[2] == 512 and a[3] == 1000 for a in aufg if a[0] == "nk"))
    if S_JE_SCHRITT_V4:
        k = kosten_schaetzung_v4()
        pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (mit Aufschlag)", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    pruefe("Pareto: dominierter Arm wird erkannt", pareto_nicht_dominiert({"a": (1, 1, 1), "b": (2, 2, 2), "c": (0.5, 5, 1)}) == ["a", "c"])
    t = trainiere_v3("Df_z2", 1e9, 0, 5, d=256, klein=True, val=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        ck = str(Path(tmp) / "a.pt")
        ref = trainiere_v3("Lk_z2", 2e-3, 3, 30, d=256, klein=True, val=True)
        try:
            trainiere_v3("Lk_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=15)
            pruefe("Abbruch wurde ausgeloest", False)
        except _Abbruch:
            pass
        pruefe("Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        fort = trainiere_v3("Lk_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Fortsetzung ab Schritt 10", fort["fortgesetzt_ab"] == 10)
        pruefe("Fortsetzung = ungestoerter Lauf (bpc identisch)", abs(fort["bpc"] - ref["bpc"]) < 1e-9)
        pruefe("Checkpoint nach Ende geloescht", not Path(ck).exists())
        try:
            trainiere_v3("Df_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=12)
        except _Abbruch:
            pass
        anders = trainiere_v3("Df_z2", 1e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Checkpoint mit anderer Lernrate wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [("mess", a, 512, 10, s, 1e-3, True, 5, tmp) for a in ("Sz", "Df_z2", "Lk_z2") for s in (1, 2, 3)]
        z1 = _rechne(a4[:4], pfad, kennung, fn=_fn_test_v3, prozesse=1, schluessel=_schl_v3)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test_v3, prozesse=1, schluessel=_schl_v3)
        pruefe("Wiederaufnahme je Aufgabe (Arm, d, T, Seed)", len(z1) == 4 and len(z2) == 9 and all(z2[k] == z1[k] for k in z1))
        Pk = {**P, "d": (256,), "tuning_T": 3, "ckpt_alle": 2}
        pt = Path(tmp) / "t.json"
        kt = {"karte": "t4"}
        arme_k = ("Df_z2", "Lk_z2")
        n0 = _ZAEHLER["lr"]
        t1 = tuning_v3(Pk, True, TUNING_SEED_V4, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        n1 = _ZAEHLER["lr"]
        t2 = tuning_v3(Pk, True, TUNING_SEED_V4, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        pruefe("Tuning: Raten je Arm gewaehlt", set(t1) == {f"{a}|256" for a in arme_k} and all(v["lr"] > 0 for v in t1.values()))
        pruefe("Tuning-Wiederaufnahme rechnet nichts neu und liefert dasselbe", n1 > n0 and _ZAEHLER["lr"] == n1 and t1 == t2)
    rng = np.random.default_rng(13)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER (I-045-v4) {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    erwarte("Sz besser", _u4(rng), **{"NK": "BESTANDEN", "H1": "STRUKTUR_BESSER", "H2": "STRUKTUR_BESSER"})
    erwarte("Referenzen besser", _u4(rng, adv1=-0.08, adv2=-0.08), **{"H1": "H0", "H2": "H0"})
    erwarte("H1 gemischt", _u4(rng, adv1=0.10, adv2=-0.08), **{"H1": "STRUKTUR_BESSER", "H2": "H0"})
    erwarte("Kein Unterschied", _u4(rng, adv1=0.0, adv2=0.0, sd=0.03), **{"H1": "HU", "H2": "HU"})
    erwarte("NK faellt", _u4(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle"})
    erwarte("Divergenz Df_z2 T=16000", _u4(rng, div=("Df_z2", 16000)), **{"H1": "HU (Divergenz", "H2": "STRUKTUR_BESSER"})
    erwarte("Kollaps Lk_z2 T=16000", _u4(rng, koll=("Lk_z2", 16000)), **{"H1": "STRUKTUR_BESSER", "H2": "HU (Kollaps"})
    erwarte("Divergenz bei T=4000 stoert nicht", _u4(rng, div=("Sz", 4000)), **{"H1": "STRUKTUR_BESSER", "H2": "STRUKTUR_BESSER"})
    erwarte("Zelle fehlt", _u4(rng, fehlt=("Lk_z2", 16000)), **{"H1": "STRUKTUR_BESSER", "H2": "HU (Zellen fehlen"})
    p = {f"{a}|{D_V4}": {"ffn": 1, "gesamt": 2} for a in ARME_MESS_V4}
    a_ = auswertung_v4(_synth_v4(rng, div=("Sz", 16000)), p, 5.0)
    pruefe("Divergenzrate 1/8, Seed bleibt in den Zahlen (n = 8)", abs(a_["divergenzrate"]["Sz|T16000|d512"] - 0.125) < 1e-9
           and a_["zellen"]["Sz|T16000|d512"]["n"] == 8 and a_["diff"]["H1"]["n"] == 8)
    pruefe("Holm ueber H1, H2", set(a_["p_holm"]) == {"H1", "H2"})
    pruefe("Beschreibend: Abstand zu D_z2 und Pareto vorhanden", "Abstand_Df_z2_zu_D_z2" in a_["beschreibend"] and "pareto_T_hi" in a_["beschreibend"])
    erlaubt = ("STRUKTUR_BESSER", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u4(rng).values()))
    soll = {"H1": {"STRUKTUR_BESSER", "H0", "HU"}, "H2": {"STRUKTUR_BESSER", "H0", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k, w in gesehen if k == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    return bool(ok)


# ================================================================== I-045-v5 „Nachpruefung v3/v4: Datenmenge, Schritte, Lernrate je T, beide FLOP-Zaehlungen“
# VORREGISTRIERUNG: INSTITUT/LABORBUCH.md (Eintrag I-045-v5; Entwurf INSTITUT/vorreg_entwurf/I-045-v5.md). Grundlage: recherche/2026-10-27_welle w9 (Punkte 1, 2, 4), w12.
# ANLASS: In v3/v4 wurde T = 16000 Schritte auf nur 1,6 MB Text trainiert (10,15 Epochen); die Lernrate wurde nur bei T = 4000 gesucht und fuer alle T benutzt;
#   „gleiche FLOPs“ zaehlte Fastfood als dichte Matmuls (Ha·x·Hb). Ideal gezaehlt (FWHT) hat Sz 9-12x weniger Operationen; ein ideal gleich teurer dichter Gegner hat h≈42/46.
#   Der Nebenbefund „schmale dichte FFN schlaegt volle Breite bei T = 16000“ koennte Ueberanpassung oder eine zu hohe Lernrate sein.
# FAKTOREN: Datenmenge (voll = 1.614.280 Trainingsbytes | Haelfte = die ersten 50 % der Trainingsbytes) x Schritte T in {4000, 16000} x d in {256, 512};
#   Lernrate je (Arm, d, T) NEU auf dem Tuning-Seed (Validierung = erste 64 kB dev; bei VOLLER Datenmenge gesucht, gilt auch fuer die Haelfte).
# ARME (nur die FFN verschieden; Attention/Einbettung identisch; 4 Lagen, Kontext 128, Batch 8x128):
#   Sz    Fastfood + z², Breite 4d (geprueftes Struktur-FFN)
#   Df_z2 dicht + z², Breite aus gleichen MATMUL-FLOPs wie Sz (d = 256: h = 388, d = 512: h = 580)
#   Di_z2 dicht + z², Breite aus gleichen IDEALEN Operationen wie Sz (FWHT: 2D(1+log2 D) je Block; d = 256: h = 42, d = 512: h = 46)
#   Dk_z2 dicht + z², h = 10 (parametergleich zu Sz; beschreibend)
#   D_gelu dicht + GELU, volle Breite 4d (Referenz fuer H2)
# BEIDE ZAEHLUNGEN werden fuer alle Arme ausgewiesen: matmul (as implemented, `_ffn_flops_modul`) und ideal (`_ffn_flops_ideal_modul`).
# HYPOTHESEN (Test-bpc; gepaarter einseitiger t-Test, n = 8; Holm ueber H1a, H1b, H2a, H2b; alpha 0,05; T = 16000, VOLLE Datenmenge):
#   H1a/H1b  Sz < Di_z2 (bei ideal gleichen Operationen), d = 256 / 512      : SZ_BESSER | H0 (Di_z2 sicher besser) | HU
#   H2a/H2b  Vorsprung der schmalen dichten FFN W = D_gelu − Df_z2 > 0, d = 256 / 512 (Lernrate je T getunt)
#            : VORSPRUNG_BLEIBT (W > 0, Holm) | VORSPRUNG_VERSCHWUNDEN (TOST ±0,02 bpc gleichwertig ODER D_gelu sicher besser, p_kl < alpha/2) | HU
#   Nicht in Holm (explorativ-bestaetigend, zweiseitig, alpha 0,05 ohne Korrektur; T = 16000; je Seed ueber d = 256, 512 gemittelt):
#   H3  Ueberanpassungsluecke G = Test-bpc − Train-bpc (volle Daten): G(D_gelu) − G(Df_z2) : GROESSERE_LUECKE (D_gelu) | KLEINERE_LUECKE | HU
#   H4  Wechselwirkung mit der Datenmenge: W(Haelfte) − W(voll)                        : WAECHST_BEI_WENIGER_DATEN | SCHRUMPFT_BEI_WENIGER_DATEN | HU
# DIVERGENZ (vorab, wie v3; je Zelle Arm x T x d x Datenmenge): nicht endlicher Verlust (bpc := 8) ODER End-bpc >= Unigramm-Entropie; Kollaps = Median + 0,2 bpc.
#   Divergierte Seeds bleiben in allen Zahlen; jede Hypothese mit einer beteiligten Zelle mit Divergenz/Kollaps ist HU.
# NEGATIVKONTROLLE: Sz und Di_z2, d = 256, 1000 Schritte, volle Daten, permutierte Trainingsbytes, Seeds 2848/2849; mittlerer Test-bpc >= 0,9 x Unigramm-Entropie.
# BESCHREIBEND: alle Zellen (Test-/Val-/Train-bpc, Luecke), Vorsprung von Sz gegen jeden Arm, W je Zelle, Lernraten (Rand?), FFN-Parameter, beide FLOP-Zaehlungen, Zeit/Token, Epochen.
# ERWARTUNG VORAB (kein Befund): H1 offen (Di_z2 hat ~4x mehr Parameter, aber nur ~1/9 der Operationen; Sz koennte hier gleichziehen); H2: der Vorsprung schrumpft bei halber
#   Datenmenge nicht, sondern waechst (Ueberanpassung der breiten FFN, H4 WAECHST) und verschwindet mit neu getunter Lernrate teilweise.
KARTE_V5 = "I-045-v5"
SEEDS_V5 = tuple(range(2840, 2848))
NK_SEEDS_V5 = (2848, 2849)
TUNING_SEED_V5 = 2891
ARME_V5 = ("Sz", "Df_z2", "Di_z2", "Dk_z2", "D_gelu")
NK_ARME_V5 = ("Sz", "Di_z2")
T_WERTE_V5 = (4000, 16000)
D_WERTE_V5 = (256, 512)
ANTEILE_V5 = (1.0, 0.5)
LR_GITTER_V5 = (1e-4, 3e-4, 9e-4)      # um die v3/v4-Optima (1,7e-4 .. 5e-4) herum; Gitter in beide Richtungen verlaengerbar
LR_FAKTOR_V5 = 3
TUNING_RUNDEN_V5 = 2                    # hoechstens 2 Runden (Wandzeit); danach Notfallwahl unter den gemessenen Raten
TOST_V5 = 0.02
PROFILE_V5 = {"standard": {"T": T_WERTE_V5, "d": D_WERTE_V5, "anteile": ANTEILE_V5, "nk_T": 1000, "nk_d": 256, "nk_lr_T": 4000, "ckpt_alle": CKPT_ALLE,
                           "tuning_runden": TUNING_RUNDEN_V5, "seeds": SEEDS_V5, "nk_seeds": NK_SEEDS_V5}}
_BELEGT_V5 = _BELEGT_V4 | set(range(2620, 2630)) | {2699}
HYPOTHESEN_V5 = {
    "H1a": "Sz < Di_z2 (Test-bpc, ideal gleiche Operationen) bei T=16000, volle Daten, d=256 (einseitig gepaart t, n=8, Holm H1a-H2b). SZ_BESSER | H0 (Di_z2 sicher besser) | HU.",
    "H1b": "wie H1a bei d=512. SZ_BESSER | H0 | HU.",
    "H2a": "Vorsprung W = D_gelu − Df_z2 > 0 (Test-bpc, Lernrate je T getunt) bei T=16000, volle Daten, d=256 (einseitig gepaart t, Holm). VORSPRUNG_BLEIBT | VORSPRUNG_VERSCHWUNDEN (TOST ±0,02 bpc oder D_gelu sicher besser) | HU.",
    "H2b": "wie H2a bei d=512. VORSPRUNG_BLEIBT | VORSPRUNG_VERSCHWUNDEN | HU.",
    "H3": "Luecke G = Test-bpc − Train-bpc bei T=16000, volle Daten, je Seed ueber d gemittelt: G(D_gelu) − G(Df_z2), zweiseitig gepaart t (nicht in Holm). GROESSERE_LUECKE | KLEINERE_LUECKE | HU.",
    "H4": "W(Haelfte) − W(voll) bei T=16000, je Seed ueber d gemittelt, zweiseitig gepaart t (nicht in Holm). WAECHST_BEI_WENIGER_DATEN | SCHRUMPFT_BEI_WENIGER_DATEN | HU.",
    "DIVERGENZ": HYPOTHESEN_V3["DIVERGENZ"],
    "NK": f"Negativkontrolle (Sz, Di_z2, d=256, 1000 Schritte, volle Daten, permutierte Trainingsbytes, Seeds 2848/2849): mittlerer Test-bpc >= {NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: Test-/Val-/Train-bpc und Luecke aller Zellen, Vorsprung von Sz gegen jeden Arm, W je Zelle, Lernraten, FFN-Parameter, FLOPs (matmul und ideal), Zeit/Token, Epochen, Trainingszeit, Divergenzrate.",
}
# Sekunden je Trainingsschritt (1 Thread, Batch 8x128, Container, Median; siehe schritt_zeiten_v5 und LABORBUCH)
# Gemessen 27.10.2026 im Container (1 Thread, Median ueber 15 Schritte); konservativ aufgerundet bzw. Sz/D_gelu mit den Maxima aus v3 (0,32 / 0,79 / 1,03).
S_JE_SCHRITT_V5 = {("Sz", 256): 0.32, ("Df_z2", 256): 0.20, ("Di_z2", 256): 0.15, ("Dk_z2", 256): 0.15, ("D_gelu", 256): 0.32,
                   ("Sz", 512): 0.79, ("Df_z2", 512): 0.52, ("Di_z2", 512): 0.40, ("Dk_z2", 512): 0.40, ("D_gelu", 512): 1.03}


# ---- Zaehlungen ----
_SZ_IDEAL = {}


def _ffn_flops_ideal_modul(ffn) -> int:
    """FFN-FLOPs je Token und Lage IDEAL gezaehlt (1 Addition/Multiplikation = 1 Flop; Auswahl/Permutation kostenlos): Fastfood-Block 2D(1+log2 D) (zwei Diagonalen,
    zwei schnelle Walsh-Transformationen D·log2 D), dicht 2·ein·aus, niedriger Rang 2·(ein·r + r·aus); dazu Bias (aus) und Aktivierung (Breite)."""
    n = 0
    for mat in ((ffn.a, ffn.b) if isinstance(ffn, FFNStruktur) else ()):
        m = mat.m
        if isinstance(m, Fastfood):
            n += m.m * 2 * m.D * (1 + int(math.log2(m.D)))
        else:
            n += sum(2 * lin.in_features * lin.out_features for lin in m)
        n += mat.bias.numel()
    if isinstance(ffn, FFNStruktur):
        return int(n + ffn.a.bias.numel())
    h, d = ffn.a.out_features, ffn.a.in_features
    return int(4 * d * h + h + d + h)


def sz_flops_ideal_lage(d) -> int:
    """Ideale FFN-FLOPs je Token und Lage von Sz (d = 256: 43.264; d = 512: 94.720)."""
    if d not in _SZ_IDEAL:
        _SZ_IDEAL[d] = _ffn_flops_ideal_modul(FFNStruktur(d, 4 * d, "ff", torch.square, torch.Generator().manual_seed(0)))
    return _SZ_IDEAL[d]


def h_ideal(d) -> int:
    """Breite h der dichten FFN mit ideal gleichen Operationen wie Sz (4·d·h + 2h + d naechst sz_flops_ideal_lage): d = 256 -> 42, d = 512 -> 46."""
    ziel = sz_flops_ideal_lage(d)
    return min(range(1, 8 * d), key=lambda h: (abs(4 * d * h + 2 * h + d - ziel), h))


def baue_ffn_v5(arm, d, gen):
    return FFNDicht(d, h_ideal(d), torch.square) if arm == "Di_z2" else baue_ffn(arm, d, gen)


class Block5(Block):
    def __init__(self, arm, d, h, gen):
        nn.Module.__init__(self)
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = baue_ffn_v5(arm, d, gen)


class Sprachmodell5(Sprachmodell):
    """Wie `Sprachmodell` (gleiche Initialisierung fuer die alten Arme), zusaetzlich Arm Di_z2."""

    def __init__(self, arm, seed, d=D_MODELL, lagen=LAGEN, h=KOEPFE, ctx=KONTEXT):
        nn.Module.__init__(self)
        gen = torch.Generator().manual_seed(seed * 31 + 11)
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bl = nn.ModuleList([Block5(arm, d, h, gen) for _ in range(lagen)])
        self.ln = nn.LayerNorm(d)


def modell_v5(arm, seed, d):
    return Sprachmodell5(arm, seed, d=d, h=d // 64)


def flops_v5(arm, d, lagen=LAGEN) -> dict:
    """Beide Zaehlungen, FFN aller Lagen: matmul (as implemented) und ideal; dazu FFN-Parameter."""
    ffn = modell_v5(arm, 0, d).bl[0].ffn
    return {"matmul": int(_ffn_flops_modul(ffn) * lagen), "ideal": int(_ffn_flops_ideal_modul(ffn) * lagen),
            "ffn_parameter": int(sum(p.numel() for p in ffn.parameters()) * lagen)}


def daten_anteil(anteil, klein=False):
    """(train, val, test); train = die ersten `anteil` der Trainingsbytes (1.0 = alle)."""
    tr, va, te = daten(klein)
    return (tr if anteil >= 1.0 else tr[:int(len(tr) * anteil)]), va, te


# ---- Training mit Train-bpc, Datenanteil, Checkpoint ----
def trainiere_v5(arm, lr, seed, schritte, d=D_MODELL, anteil=1.0, klein=False, val=False, permutiert=False, ckpt=None, ckpt_alle=CKPT_ALLE, _stopp_bei=None):
    """Wie `trainiere_v3` (bei anteil = 1,0 und gleichem Arm bit-gleich), aber mit Datenanteil und zusaetzlich Train-/Val-bpc.
    Train-bpc = bpc auf den ersten 64 kB der tatsaechlich benutzten Trainingsbytes (bei voller und halber Datenmenge dieselben Bytes). val = nur Validierung (Tuning)."""
    torch.set_num_threads(1)
    tr, va, te = daten_anteil(anteil, klein)
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 77))]
    torch.manual_seed(seed)
    netz = modell_v5(arm, seed, d)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    meta = {"arm": arm, "d": int(d), "lr": float(lr), "seed": int(seed), "schritte": int(schritte), "klein": bool(klein), "perm": bool(permutiert),
            "anteil": float(anteil), "v": 5}
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
    offs = torch.arange(KONTEXT)
    t0 = time.time()
    netz.train()
    for s in range(start, schritte):
        i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l):
            if ckpt and Path(ckpt).exists():
                Path(ckpt).unlink()
            return {"bpc": 8.0, "bpc_train": 8.0, "bpc_val": 8.0, "instabil": True, "fortgesetzt_ab": start, "anteil": float(anteil)}
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step()
        sched.step()
        if ckpt and (s + 1) % ckpt_alle == 0 and s + 1 < schritte:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt + ".tmp"
            torch.save({"meta": meta, "modell": netz.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "g": g.get_state(),
                        "rng": torch.get_rng_state(), "schritt": s + 1, "sekunden": t_vor + time.time() - t0}, tmp)
            os.replace(tmp, ckpt)
        if _stopp_bei is not None and s + 1 == _stopp_bei:
            raise _Abbruch(f"simulierter Abbruch bei Schritt {s + 1}")
    t_train = t_vor + time.time() - t0
    bval = bpc_auswerten(netz, va[:VAL_BYTES])
    out = {"instabil": False, "sekunden_training": t_train, "fortgesetzt_ab": start, "anteil": float(anteil), "epochen": schritte * BATCH * KONTEXT / len(tr)}
    if val:
        out["bpc"] = bval
    else:
        out.update({"bpc": bpc_auswerten(netz, te), "bpc_val": bval, "bpc_train": bpc_auswerten(netz, tr[:VAL_BYTES])})
    if ckpt and Path(ckpt).exists():
        Path(ckpt).unlink()
    return out


def schritt_zeiten_v5(arme=ARME_V5, ds=D_WERTE_V5, n=20, warm=3):
    """Sekunden je Trainingsschritt (1 Thread, Batch 8x128, Vorwaerts + Rueckwaerts + Clip + Adam), Median ueber n Schritte."""
    torch.set_num_threads(1)
    tr, _, _ = daten(True)
    offs = torch.arange(KONTEXT)
    g = torch.Generator().manual_seed(1)
    aus = {}
    for d in ds:
        for a in arme:
            netz = modell_v5(a, 0, d)
            opt = torch.optim.Adam(netz.parameters(), lr=1e-4)
            ts = []
            for k in range(warm + n):
                i = torch.randint(0, len(tr) - KONTEXT - 1, (BATCH,), generator=g)
                t0 = time.perf_counter()
                l = F.cross_entropy(netz(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
                opt.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
                opt.step()
                if k >= warm:
                    ts.append(time.perf_counter() - t0)
            aus[(a, d)] = float(np.median(ts))
    return aus


# ---- Aufgaben ----
def _pct(anteil) -> int:
    return int(round(anteil * 100))


def _schl_v5(a):
    """Aufgabe (teil, arm, d, T, seed, lr, klein, ckpt_alle, ckordner, anteil) -> Schluessel der Zwischendatei."""
    return f"einheit|{a[0]}|{a[1]}|d{a[2]}|T{a[3]}|D{_pct(a[9])}|{a[4]}"


def _zname5(a, T, d, anteil) -> str:
    return f"{a}|T{T}|d{d}|D{_pct(anteil)}"


def _kosten_aufgabe_v5(arm, d, T):
    return T * S_JE_SCHRITT_V5.get((arm, d), 1.0) if S_JE_SCHRITT_V5 else T * d


def aufgaben_v5(hp, P, klein, ckordner):
    """Mess- und NK-Aufgaben; die teuersten (lange T, grosses d) zuerst. Lernrate je (Arm, d, T) aus dem Tuning (hp-Schluessel 'Arm|d|T')."""
    aufg = [("mess", a, d, T, s, hp[f"{a}|{d}|{T}"]["lr"], klein, P["ckpt_alle"], str(ckordner), an)
            for a in ARME_V5 for d in P["d"] for T in P["T"] for an in P["anteile"] for s in P["seeds"]]
    aufg += [("nk", a, P["nk_d"], P["nk_T"], s, hp[f"{a}|{P['nk_d']}|{P['nk_lr_T']}"]["lr"], klein, P["ckpt_alle"], str(ckordner), 1.0)
             for a in NK_ARME_V5 for s in P["nk_seeds"]]
    aufg.sort(key=lambda a: -_kosten_aufgabe_v5(a[1], a[2], a[3]))
    return aufg


def kosten_schaetzung_v5(s_je_schritt=None, prozesse=72, aufschlag=1.3, tuning_raten=3.5, ueberhang=30.0):
    """Kernstunden und Wandzeit (LPT auf `prozesse` Kernen). Tuning: im Mittel `tuning_raten` Raten je (Arm, d, T) in hoechstens TUNING_RUNDEN_V5 Runden
    (Wandzeit konservativ = Runden x laengste Tuningaufgabe)."""
    import heapq
    z = s_je_schritt or S_JE_SCHRITT_V5
    P = PROFILE_V5["standard"]
    dauer = lambda a, d, T: T * z[(a, d)] * aufschlag + ueberhang
    t_aufg = [dauer(a, d, T) for a in ARME_V5 for d in P["d"] for T in P["T"]]
    kern_tuning = sum(t_aufg) * tuning_raten
    wand_tuning = P["tuning_runden"] * max(t_aufg)
    m_aufg = [dauer(a, d, T) for a in ARME_V5 for d in P["d"] for T in P["T"] for _ in P["anteile"] for _ in P["seeds"]]
    m_aufg += [dauer(a, P["nk_d"], P["nk_T"]) for a in NK_ARME_V5 for _ in P["nk_seeds"]]
    last = [0.0] * prozesse
    heapq.heapify(last)
    for t in sorted(m_aufg, reverse=True):
        heapq.heappush(last, heapq.heappop(last) + t)
    wand_mess = max(last)
    return {"aufgaben_mess": len(m_aufg), "kernstunden_mess": sum(m_aufg) / 3600, "kernstunden_tuning": kern_tuning / 3600,
            "kernstunden_gesamt": (sum(m_aufg) + kern_tuning) / 3600, "laengste_aufgabe_h": max(m_aufg) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand_mess / 3600, "wand_gesamt_h": (wand_tuning + wand_mess) / 3600,
            "prozesse": prozesse, "aufschlag": aufschlag}


def _arbeiter_v5(a):
    teil, arm, d, T, seed, lr, klein, ck_alle, ckordner, anteil = a
    key = _schl_v5(a)
    r = trainiere_v5(arm, lr, seed, T, d=d, anteil=anteil, klein=klein, permutiert=(teil == "nk"), ckpt=_ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"teil": teil, "arm": arm, "d": d, "T": T, "seed": seed, "lr": lr, **r}


def _lr_arbeiter_v5(a):
    schl, seed, lr, d, T, klein, ck_alle, ckordner, arm = a
    _ZAEHLER["lr"] += 1
    key = _schl_lr(a)
    r = trainiere_v5(arm, lr, seed, T, d=d, anteil=1.0, klein=klein, val=True, ckpt=_ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"acc": 1 - r["bpc"] / 8, "bpc": r["bpc"], "instabil": r["instabil"]}


def tuning_v5(P, klein, tseed, pfad, kennung, prozesse, ckordner, arme=None):
    """Lernrate je (Arm, d, T) auf dem Tuning-Seed bei VOLLER Datenmenge und dem jeweiligen T (Validierung); alle Raten aller Schluessel einer Runde laufen
    gemeinsam parallel (lange zuerst), jede Aufgabe einzeln wiederaufnehmbar. Hoechstens P['tuning_runden'] Runden."""
    schl = [f"{a}|{d}|{T}" for a in (arme or ARME_V5) for d in P["d"] for T in P["T"]]

    def aufgaben_fuer(s, lr):
        a, d, T = s.split("|")
        return [(s, tseed, lr, int(d), int(T), klein, P["ckpt_alle"], str(ckordner), a)]

    def auswerten(aufg):
        lang = sorted(aufg, key=lambda a: -(a[4] * (S_JE_SCHRITT_V5.get((a[8], a[3]), 1.0) if S_JE_SCHRITT_V5 else a[3])))
        res = _rechne(lang, pfad, kennung, fn=_lr_arbeiter_v5, schluessel=_schl_lr, prozesse=prozesse)
        return [res[_schl_lr(a)] for a in aufg]

    info, _ = gittersuche_stabil(schl, aufgaben_fuer, auswerten, gitter=LR_GITTER_V5, faktor=LR_FAKTOR_V5, kollaps=KOLLAPS_SCORE, runden=P["tuning_runden"])
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"], "runden": v["runden"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _zeit_arbeiter_v5(paare):
    torch.set_num_threads(1)
    return {f"{a}|{d}": zeit_je_token(modell_v5(a, 0, d)) for a, d in paare}


def zeit_im_ein_thread_v5(paare):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter_v5, (tuple(paare),))


# ---- Auswertung (bpc: kleiner ist besser) ----
def _tost_p(v, grenze=TOST_V5) -> float:
    """TOST auf Gleichwertigkeit |Mittel| < grenze (gepaart): p = max(p(Mittel > −grenze), p(Mittel < grenze))."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return 1.0
    return float(max(_t_p(v + grenze, "gr"), _t_p(grenze - v, "gr")))


def auswertung_v5(zeilen: list, params: dict, unigramm: float, t_lo=T_WERTE_V5[0], t_hi=T_WERTE_V5[-1], ds=D_WERTE_V5, flops=None, zeit=None, tuning=None) -> dict:
    bpc, trn, val, inst, tt = {}, {}, {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            k = (z["arm"], z["T"], z["d"], z["anteil"])
            bpc.setdefault(k, {})[z["seed"]] = z["bpc"]
            trn.setdefault(k, {})[z["seed"]] = z.get("bpc_train", float("nan"))
            val.setdefault(k, {})[z["seed"]] = z.get("bpc_val", float("nan"))
            inst.setdefault(k, {})[z["seed"]] = bool(z.get("instabil"))
            tt.setdefault(k, {})[z["seed"]] = z.get("sekunden_training", float("nan"))
    r = {"t_lo": t_lo, "t_hi": t_hi, "ds": list(ds), "zellen": {}, "train": {}, "val": {}, "luecke": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {},
         "bpc_ohne_divergente": {}, "negativ": {}, "diff": {}, "tost_p": {}, "p_holm": {}, "beschreibend": {}, "parameter": params, "unigramm_bpc": unigramm,
         "training_s_mittel": {}}
    for k, v in bpc.items():
        seeds = sorted(v)
        w = np.array([v[s] for s in seeds])
        n = _zname5(*k)
        r["zellen"][n] = zusammenfassung(w)
        r["train"][n] = zusammenfassung([trn[k][s] for s in seeds])
        r["val"][n] = zusammenfassung([val[k][s] for s in seeds])
        r["luecke"][n] = zusammenfassung([v[s] - trn[k][s] for s in seeds])
        med = float(np.median(w))
        r["kollaps"][n] = [s for s in seeds if not (v[s] <= med + KOLLAPS_BPC)]
        r["divergiert"][n] = [s for s in seeds if inst[k][s] or not (v[s] < unigramm)]
        r["divergenzrate"][n] = len(r["divergiert"][n]) / len(seeds)
        ok = [v[s] for s in seeds if s not in r["divergiert"][n]]
        r["bpc_ohne_divergente"][n] = float(np.mean(ok)) if ok else None
        r["training_s_mittel"][n] = float(np.nanmean(list(tt[k].values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def vor(T, d, an, ref, pruef):
        """Je Seed: bpc(ref) − bpc(pruef); positiv = pruef besser. {} wenn eine Zelle fehlt."""
        a, b = bpc.get((ref, T, d, an)), bpc.get((pruef, T, d, an))
        if not a or not b:
            return {}
        return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}

    def luecke(T, d, an, arm):
        a, b = bpc.get((arm, T, d, an)), trn.get((arm, T, d, an))
        return {s: a[s] - b[s] for s in sorted(a)} if a and b else {}

    for h, d in (("H1a", ds[0]), ("H1b", ds[-1])):
        v = vor(t_hi, d, 1.0, "Di_z2", "Sz")
        if v:
            r["diff"][h] = _stat(np.array(list(v.values())))
    for h, d in (("H2a", ds[0]), ("H2b", ds[-1])):
        v = vor(t_hi, d, 1.0, "D_gelu", "Df_z2")
        if v:
            r["diff"][h] = _stat(np.array(list(v.values())))
            r["tost_p"][h] = _tost_p(list(v.values()))
    # H3: Luecken-Differenz, je Seed ueber d gemittelt
    gl = {d: (luecke(t_hi, d, 1.0, "D_gelu"), luecke(t_hi, d, 1.0, "Df_z2")) for d in ds}
    if all(a and b for a, b in gl.values()):
        gem = sorted(set.intersection(*(set(a) & set(b) for a, b in gl.values())))
        r["diff"]["H3"] = _stat(np.array([np.mean([gl[d][0][s] - gl[d][1][s] for d in ds]) for s in gem]))
    # H4: W(Haelfte) − W(voll), je Seed ueber d gemittelt
    wv = {(d, an): vor(t_hi, d, an, "D_gelu", "Df_z2") for d in ds for an in ANTEILE_V5}
    if all(wv.values()):
        gem = sorted(set.intersection(*(set(x) for x in wv.values())))
        r["diff"]["H4"] = _stat(np.array([np.mean([wv[(d, 0.5)][s] - wv[(d, 1.0)][s] for d in ds]) for s in gem]))
    # beschreibend
    B = r["beschreibend"]
    for ref in ARME_V5:
        if ref == "Sz":
            continue
        for d in ds:
            for T in sorted({k[1] for k in bpc}):
                for an in ANTEILE_V5:
                    v = vor(T, d, an, ref, "Sz")
                    if v:
                        B.setdefault(f"Vorsprung_Sz_gegen_{ref}", {})[f"T{T}|d{d}|D{_pct(an)}"] = _stat(np.array(list(v.values())))
    for d in ds:
        for T in sorted({k[1] for k in bpc}):
            for an in ANTEILE_V5:
                v = vor(T, d, an, "D_gelu", "Df_z2")
                if v:
                    B.setdefault("W_D_gelu_minus_Df_z2", {})[f"T{T}|d{d}|D{_pct(an)}"] = _stat(np.array(list(v.values())))
    if flops:
        B["flops_je_token"] = flops
    if zeit:
        B["zeit_s_je_token"] = zeit
    if tuning:
        B["lernraten"] = {k: {"lr": v["lr"], "am_rand": v["am_rand"]} for k, v in tuning.items()}
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1a", "H1b", "H2a", "H2b") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _vorab_v5(r, nk, h, zellen, holm_noetig=True):
    if h not in r["diff"] or "mittel" not in r["diff"][h] or (holm_noetig and h not in r["p_holm"]):
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    dv = {z: r["divergiert"][z] for z in zellen if r["divergiert"].get(z)}
    if dv:
        return f"HU (Divergenz in {dv})"
    k = {z: r["kollaps"][z] for z in zellen if r["kollaps"].get(z)}
    if k:
        return f"HU (Kollaps in {k})"
    return None


def _zahlen_zwei(r, h):
    s = r["diff"][h]
    return (f"{s['mittel']:+.4f} bpc, 90-%-KI [{s['ki'][0]:+.4f}; {s['ki'][1]:+.4f}], n {s['n']}, p_zweiseitig {s['p_zwei']:.3g}").replace(".", ",")


def urteil_v5(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutiert Sz/Di_z2: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {NK_FAKTOR}x)"}
    t_hi, ds = r["t_hi"], tuple(r["ds"])
    for h, d in (("H1a", ds[0]), ("H1b", ds[-1])):
        v = _vorab_v5(r, nk, h, [_zname5(a, t_hi, d, 1.0) for a in ("Di_z2", "Sz")])
        if v:
            u[h] = v
            continue
        z = _zahlen(r, h)
        if r["p_holm"][h] < ALPHA:
            u[h] = f"SZ_BESSER (Di_z2−Sz bei T={t_hi}, d={d}, ideal gleiche Operationen: {z})"
        elif r["diff"][h]["p_kl"] < ALPHA / 2:
            u[h] = f"H0 (Di_z2 sicher besser als Sz bei T={t_hi}, d={d}; {z})"
        else:
            u[h] = f"HU ({z})"
    for h, d in (("H2a", ds[0]), ("H2b", ds[-1])):
        v = _vorab_v5(r, nk, h, [_zname5(a, t_hi, d, 1.0) for a in ("D_gelu", "Df_z2")])
        if v:
            u[h] = v
            continue
        z = _zahlen(r, h)
        tp = r["tost_p"][h]
        if r["p_holm"][h] < ALPHA:
            u[h] = f"VORSPRUNG_BLEIBT (D_gelu−Df_z2 bei T={t_hi}, d={d}: {z}; TOST-p {tp:.3g})".replace(".", ",")
        elif tp < ALPHA:
            u[h] = f"VORSPRUNG_VERSCHWUNDEN (gleichwertig innerhalb ±{TOST_V5} bpc, TOST-p {tp:.3g}; {z})".replace(".", ",")
        elif r["diff"][h]["p_kl"] < ALPHA / 2:
            u[h] = f"VORSPRUNG_VERSCHWUNDEN (volle Breite D_gelu sicher besser; {z})"
        else:
            u[h] = f"HU ({z})"
    zl_h3 = [_zname5(a, t_hi, d, 1.0) for a in ("D_gelu", "Df_z2") for d in ds]
    v = _vorab_v5(r, nk, "H3", zl_h3, holm_noetig=False)
    if v:
        u["H3"] = v
    else:
        z = _zahlen_zwei(r, "H3")
        s = r["diff"]["H3"]
        u["H3"] = (f"GROESSERE_LUECKE (G(D_gelu) − G(Df_z2) {z})" if s["p_zwei"] < ALPHA and s["mittel"] > 0 else
                   f"KLEINERE_LUECKE (G(D_gelu) − G(Df_z2) {z})" if s["p_zwei"] < ALPHA else f"HU ({z})")
    zl_h4 = [_zname5(a, t_hi, d, an) for a in ("D_gelu", "Df_z2") for d in ds for an in ANTEILE_V5]
    v = _vorab_v5(r, nk, "H4", zl_h4, holm_noetig=False)
    if v:
        u["H4"] = v
    else:
        z = _zahlen_zwei(r, "H4")
        s = r["diff"]["H4"]
        u["H4"] = (f"WAECHST_BEI_WENIGER_DATEN (W(Haelfte) − W(voll) {z})" if s["p_zwei"] < ALPHA and s["mittel"] > 0 else
                   f"SCHRUMPFT_BEI_WENIGER_DATEN (W(Haelfte) − W(voll) {z})" if s["p_zwei"] < ALPHA else f"HU ({z})")
    teile = []
    for d in ds:
        for T in sorted({int(k.split("|T")[1].split("|")[0]) for k in r["zellen"]}):
            for an in ANTEILE_V5:
                ze = ", ".join(f"{a} {r['zellen'][_zname5(a, T, d, an)]['mittel']:.3f}".replace(".", ",") for a in ARME_V5 if _zname5(a, T, d, an) in r["zellen"])
                teile.append(f"d{d} T{T} D{_pct(an)}: {ze}")
    u["B"] = "beschreibend: Test-bpc " + "; ".join(teile) + "; Rest in auswertung"
    return u


# ---- Lauf ----
def lauf_v5(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I045V5_PROFIL", "standard")
    P = dict(PROFILE_V5[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, d = 256, wenige Schritte (Checkpoint alle 3 Schritte); d = 512 deckt der Vortest ab
        P.update({"T": (6, 12), "d": (256,), "nk_T": 6, "nk_lr_T": 6, "ckpt_alle": 3, "seeds": (TUNING_SEED_V5,), "nk_seeds": (TUNING_SEED_V5,)})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT_V5 or set(P["nk_seeds"]) & _BELEGT_V5 or TUNING_SEED_V5 in P["seeds"] or TUNING_SEED_V5 in _BELEGT_V5:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V5}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE_V5}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE_V5, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": _json_sicher(P),
               "lr_gitter": list(LR_GITTER_V5), "tuning_seed": TUNING_SEED_V5, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        hp0 = tuning_v5(P, klein, TUNING_SEED_V5, pfad, kennung, prozesse, ckordner)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning"] = hp0
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = aufgaben_v5(hp, P, klein, ckordner)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_v5, schluessel=_schl_v5, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    paare = [(a, d) for a in ARME_V5 for d in P["d"]]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread_v5(paare)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {f"{a}|{d}": {"ffn": param_ffn(n), "gesamt": param_gesamt(n)} for a, d in paare for n in [modell_v5(a, 0, d)]}
    fl = {f"{a}|{d}": flops_v5(a, d) for a, d in paare}
    tr, _, te = daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_sprache", "karte": KARTE_V5, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_V5,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": TUNING_SEED_V5, "tuning": hp,
         "einstellungen": {"T": list(P["T"]), "d": list(P["d"]), "anteile": list(P["anteile"]), "nk_T": P["nk_T"], "nk_d": P["nk_d"], "ckpt_alle": P["ckpt_alle"],
                           "tuning_runden": P["tuning_runden"], "batch": BATCH, "kontext": KONTEXT, "lagen": LAGEN, "lr_gitter": list(LR_GITTER_V5),
                           "h": {a: {d: (h_ideal(d) if a == "Di_z2" else form_v4(a, d)[1] if a in ARME_V4 else 4 * d) for d in P["d"]} for a in ARME_V5},
                           "daten_bytes": [len(x) for x in daten(klein)], "epochen_T_voll": {str(T): T * BATCH * KONTEXT / len(tr) for T in P["T"]}},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_v5(zeilen, params, unigramm_entropie(tr, te), t_lo=min(P["T"]), t_hi=max(P["T"]), ds=tuple(P["d"]), flops=fl, zeit=zm, tuning=hp)
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
    e["urteil"] = urteil_v5(e["auswertung"])
    return e


# ---- Vortest v5 ----
def _synth_v5(rng, adv1=0.10, adv2=0.08, adv2_half=None, gap_gelu=0.20, gap_df=0.10, sd=0.004, n=8, koll=None, nk=5.0, div=None, fehlt=None):
    """Synthetische Zeilen: bpc(Di_z2) = bpc(Sz) + adv1 (positiv = Sz besser); bpc(D_gelu) = bpc(Df_z2) + adv2 (voll) bzw. adv2_half (Haelfte);
    Train-bpc = Test-bpc − Luecke (D_gelu: gap_gelu, alle anderen: gap_df); in allen T, d und beiden Datenanteilen."""
    adv2_half = adv2 if adv2_half is None else adv2_half
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for d in D_WERTE_V5:
            for T in T_WERTE_V5:
                for an in ANTEILE_V5:
                    w = adv2 if an == 1.0 else adv2_half
                    basis = {"Sz": 3.4, "Di_z2": 3.4 + adv1, "Df_z2": 3.3, "Dk_z2": 3.6, "D_gelu": 3.3 + w}
                    for a, v in basis.items():
                        if fehlt == (a, T, d, an):
                            continue
                        x, ist = v + gem + rng.normal(0, sd), False
                        if koll == (a, T, d, an) and i == 0:
                            x += 1.0
                        if div == (a, T, d, an) and i == 1:
                            x, ist = 8.0, True
                        g = gap_gelu if a == "D_gelu" else gap_df
                        zl.append({"teil": "mess", "arm": a, "d": d, "T": T, "anteil": an, "seed": 2840 + i, "bpc": x, "bpc_train": x - g + rng.normal(0, sd), "bpc_val": x,
                                   "instabil": ist, "sekunden_training": 10.0})
        if i < 2:
            for a in NK_ARME_V5:
                zl.append({"teil": "nk", "arm": a, "d": 256, "T": 1000, "anteil": 1.0, "seed": 2848 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u5(rng, **kw):
    p = {f"{a}|{d}": {"ffn": 1, "gesamt": 2} for a in ARME_V5 for d in D_WERTE_V5}
    return urteil_v5(auswertung_v5(_synth_v5(rng, **kw), p, 5.0))


def _fn_test_v5(a):
    return _schl_v5(a), {"v": a[4] * 2}


def _vortest_v5() -> bool:
    import tempfile
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-045-v5): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    # --- Zaehlungen: beide FLOP-Zaehlungen, Breiten, Parameter
    pruefe("Sz ideal je Lage = 43.264 (d=256) / 94.720 (d=512) (w12)", sz_flops_ideal_lage(256) == 43264 and sz_flops_ideal_lage(512) == 94720)
    pruefe("Sz matmul je Lage = 397.312 / 1.187.840 (v4-Zaehlung unveraendert)", sz_flops_lage(256) == 397312 and sz_flops_lage(512) == 1187840)
    pruefe("Breiten: Di h = 42 / 46 (ideal), Df h = 388 / 580 (matmul), Dk h = 10", h_ideal(256) == 42 and h_ideal(512) == 46
           and form_v4("Df_z2", 256)[1] == 388 and form_v4("Df_z2", 512)[1] == 580 and form_v4("Dk_z2", 256)[1] == 10 and form_v4("Dk_z2", 512)[1] == 10)
    fl = {(a, d): flops_v5(a, d) for a in ARME_V5 for d in D_WERTE_V5}
    for d in D_WERTE_V5:
        sz = fl[("Sz", d)]
        pruefe(f"Df_z2: matmul-FLOPs gleich Sz (+-3 %), d={d}", abs(fl[("Df_z2", d)]["matmul"] / sz["matmul"] - 1) <= BUDGET_TOLERANZ)
        pruefe(f"Di_z2: ideale FLOPs gleich Sz (+-3 %), d={d}", abs(fl[("Di_z2", d)]["ideal"] / sz["ideal"] - 1) <= BUDGET_TOLERANZ)
        pruefe(f"Sz: ideal deutlich unter matmul (>= 8x), d={d}", sz["matmul"] / sz["ideal"] >= 8)
        pruefe(f"Dk_z2 parametergleich zu Sz (+-2 %), d={d}", abs(fl[("Dk_z2", d)]["ffn_parameter"] / sz["ffn_parameter"] - 1) <= 0.02)
        pruefe(f"Sz-Parameter = 4·21·d, d={d}", sz["ffn_parameter"] == LAGEN * sz_param_lage(d))
        pruefe(f"D_gelu: Breite 4d, Parameter 4·(8d²+5d), d={d}", fl[("D_gelu", d)]["ffn_parameter"] == 4 * (2 * d * 4 * d + 4 * d + d))
        pruefe(f"Dichte Arme: ideal ~ matmul (Bias/Aktivierung < 1 %), d={d}", all(0 < fl[(a, d)]["ideal"] / fl[(a, d)]["matmul"] - 1 < 0.02 for a in ("Df_z2", "D_gelu")))
        pruefe(f"Di_z2 hat mehr Parameter als Sz (Vorbehalt, ~4x), d={d}", 3.5 < fl[("Di_z2", d)]["ffn_parameter"] / sz["ffn_parameter"] < 5)
    # --- Modelle: Form, Kausalitaet, Koepfe, Aktivierungen, Gleichheit mit Sprachmodell fuer alte Arme
    for d in D_WERTE_V5:
        nz = {a: modell_v5(a, 0, d) for a in ARME_V5}
        x = torch.randint(0, 256, (2, KONTEXT))
        for a, m in nz.items():
            m.eval()
            x2 = x.clone()
            x2[:, 60] = (x2[:, 60] + 1) % 256
            pruefe(f"Form/Kausalitaet {a} d={d}", m(x).shape == (2, KONTEXT, 256) and torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5)
                   and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
            pruefe(f"Koepfe d/64 {a} d={d}", m.bl[0].h == d // 64)
        pruefe(f"Aktivierungen d={d}", all(nz[a].bl[0].ffn.akt is torch.square for a in ("Sz", "Df_z2", "Di_z2", "Dk_z2")) and nz["D_gelu"].bl[0].ffn.akt is F.gelu)
        pruefe(f"Di_z2 Breite {h_ideal(d)} d={d}", nz["Di_z2"].bl[0].ffn.a.out_features == h_ideal(d))
        pruefe(f"FFN-Breite Df {form_v4('Df_z2', d)[1]}, D_gelu 4d, d={d}", nz["Df_z2"].bl[0].ffn.a.out_features == form_v4("Df_z2", d)[1]
               and nz["D_gelu"].bl[0].ffn.a.out_features == 4 * d)
        gleich = True
        for a in ("Sz", "Df_z2", "Dk_z2", "D_gelu"):
            torch.manual_seed(5)
            m1 = Sprachmodell(a, 5, d=d, h=d // 64)
            torch.manual_seed(5)
            m2 = modell_v5(a, 5, d)
            s1, s2 = m1.state_dict(), m2.state_dict()
            gleich &= set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
        pruefe(f"Sprachmodell5 initialisiert die alten Arme bit-gleich zu Sprachmodell, d={d}", gleich)
    # --- Trainierbarkeit
    unit = torch.arange(32).repeat(8)[None].repeat(16, 1)[:, :KONTEXT + 1]
    for a in ARME_V5:
        m = modell_v5(a, 0, 512)
        m = Sprachmodell5(a, 0, d=512, h=8)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        l0 = None
        for _ in range(25):
            opt.zero_grad()
            l = F.cross_entropy(m(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a} d=512", float(l.detach()) < 0.7 * l0)
    # --- Seeds (auch gegen seeds.json: nur der eigene Eintrag darf sie benutzen)
    pruefe("Seeds frisch (8 Mess 2840-2847, 2 NK 2848/49, Tuning 2891)", SEEDS_V5 == tuple(range(2840, 2848)) and NK_SEEDS_V5 == (2848, 2849) and TUNING_SEED_V5 == 2891
           and not ((set(SEEDS_V5) | set(NK_SEEDS_V5)) & _BELEGT_V5) and TUNING_SEED_V5 not in _BELEGT_V5 | set(SEEDS_V5) | set(NK_SEEDS_V5))
    try:
        import json as _json
        sj = _json.loads((WURZEL / "seeds.json").read_text())
        meine = set(SEEDS_V5) | set(NK_SEEDS_V5) | {TUNING_SEED_V5}
        fremd = [e["id"] for e in sj["eintraege"] if e["id"] != KARTE_V5 and any(a <= s <= b for a, b in e["seeds"] for s in meine)]
        pruefe(f"Keine Seed-Ueberschneidung mit anderen Eintraegen in seeds.json ({fremd})", not fremd)
    except FileNotFoundError:
        pass
    # --- Aufgaben, Reihenfolge, Plan
    P = PROFILE_V5["standard"]
    hp = {f"{a}|{d}|{T}": {"lr": 1e-3} for a in ARME_V5 for d in P["d"] for T in P["T"]}
    aufg = aufgaben_v5(hp, P, False, "x")
    schl = [_schl_v5(a) for a in aufg]
    pruefe("Aufgabenzahl 5·2·2·2·8 + 4 NK = 324, Schluessel eindeutig", len(aufg) == 324 and len(set(schl)) == 324 and sum(a[0] == "nk" for a in aufg) == 4)
    pruefe("Lange zuerst (D_gelu, d=512, T=16000), NK d=256 T=1000 volle Daten", aufg[0][1] == "D_gelu" and aufg[0][2] == 512 and aufg[0][3] == 16000
           and all(a[2] == 256 and a[3] == 1000 and a[9] == 1.0 for a in aufg if a[0] == "nk"))
    pruefe("Beide Datenanteile je Zelle mit je 8 Seeds", all(sum(1 for a in aufg if a[:4] == ("mess", "Sz", 512, 16000) and a[9] == an) == 8 for an in ANTEILE_V5))
    pruefe("Lernrate je (Arm, d, T) aus hp", all(a[5] == hp[f"{a[1]}|{a[2]}|{a[3]}"]["lr"] for a in aufg if a[0] == "mess"))
    k = kosten_schaetzung_v5()
    pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (Aufschlag 1,3), laengste Aufgabe < 24 h", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    # --- Training: bit-gleich zu v3 bei voller Datenmenge; Train-/Val-bpc; Datenanteil
    a1 = trainiere_v3("Sz", 2e-3, 3, 12, d=256, klein=True)
    b1 = trainiere_v5("Sz", 2e-3, 3, 12, d=256, anteil=1.0, klein=True)
    pruefe("trainiere_v5 (anteil 1,0) = trainiere_v3 (Test-bpc identisch)", abs(a1["bpc"] - b1["bpc"]) < 1e-9)
    b2 = trainiere_v5("Di_z2", 2e-3, 3, 12, d=256, anteil=0.5, klein=True)
    pruefe("Train-/Val-/Test-bpc vorhanden und endlich", all(np.isfinite(b2[x]) for x in ("bpc", "bpc_train", "bpc_val")) and b2["epochen"] > b1["epochen"])
    tr1, tr5 = daten_anteil(1.0, True)[0], daten_anteil(0.5, True)[0]
    pruefe("Haelfte = erste 50 % der Trainingsbytes (Praefix), Val/Test unveraendert", len(tr5) == len(tr1) // 2 and torch.equal(tr5, tr1[:len(tr5)])
           and torch.equal(daten_anteil(0.5, True)[2], daten(True)[2]))
    pruefe("Epochen: T=16000 voll ≈ 10,15", abs(16000 * BATCH * KONTEXT / len(daten()[0]) - 10.15) < 0.05)
    t = trainiere_v5("Di_z2", 1e9, 0, 5, d=256, klein=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        ck = str(Path(tmp) / "a.pt")
        ref = trainiere_v5("Di_z2", 2e-3, 3, 30, d=256, anteil=0.5, klein=True)
        try:
            trainiere_v5("Di_z2", 2e-3, 3, 30, d=256, anteil=0.5, klein=True, ckpt=ck, ckpt_alle=10, _stopp_bei=15)
            pruefe("Abbruch wurde ausgeloest", False)
        except _Abbruch:
            pass
        pruefe("Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        fort = trainiere_v5("Di_z2", 2e-3, 3, 30, d=256, anteil=0.5, klein=True, ckpt=ck, ckpt_alle=10)
        pruefe("Fortsetzung ab Schritt 10", fort["fortgesetzt_ab"] == 10)
        pruefe("Fortsetzung = ungestoerter Lauf (Test-/Train-bpc identisch)", abs(fort["bpc"] - ref["bpc"]) < 1e-9 and abs(fort["bpc_train"] - ref["bpc_train"]) < 1e-9)
        pruefe("Checkpoint nach Ende geloescht", not Path(ck).exists())
        try:
            trainiere_v5("Sz", 2e-3, 3, 30, d=256, anteil=0.5, klein=True, ckpt=ck, ckpt_alle=10, _stopp_bei=12)
        except _Abbruch:
            pass
        anders = trainiere_v5("Sz", 2e-3, 3, 30, d=256, anteil=1.0, klein=True, ckpt=ck, ckpt_alle=10)
        pruefe("Checkpoint mit anderem Datenanteil wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [("mess", a, 512, 10, s, 1e-3, True, 5, tmp, an) for a in ("Sz", "Di_z2") for an in (1.0, 0.5) for s in (1, 2)]
        z1 = _rechne(a4[:3], pfad, kennung, fn=_fn_test_v5, prozesse=1, schluessel=_schl_v5)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test_v5, prozesse=1, schluessel=_schl_v5)
        pruefe("Wiederaufnahme je Aufgabe (Arm, d, T, Datenanteil, Seed)", len(z1) == 3 and len(z2) == 8 and all(z2[k] == z1[k] for k in z1))
        Pk = {**P, "d": (256,), "T": (3, 6), "tuning_runden": 2, "ckpt_alle": 2}
        pt = Path(tmp) / "t.json"
        kt = {"karte": "t5"}
        arme_k = ("Di_z2", "Sz")
        n0 = _ZAEHLER["lr"]
        t1 = tuning_v5(Pk, True, TUNING_SEED_V5, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        n1 = _ZAEHLER["lr"]
        t2 = tuning_v5(Pk, True, TUNING_SEED_V5, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        pruefe("Tuning: Rate je (Arm, d, T) gewaehlt", set(t1) == {f"{a}|256|{T}" for a in arme_k for T in (3, 6)} and all(v["lr"] > 0 and v["runden"] <= 2 for v in t1.values()))
        pruefe("Tuning-Wiederaufnahme rechnet nichts neu und liefert dasselbe", n1 > n0 and _ZAEHLER["lr"] == n1 and t1 == t2)
    # --- Urteilszweige
    rng = np.random.default_rng(17)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for kk, w in soll.items():
            gesehen.add((kk, w.split(" (")[0]))
            if not u[kk].startswith(w):
                print(f"  Vortest FEHLER (I-045-v5) {name}: {kk} soll {w}, ist {u[kk]}", flush=True)
                ok = False

    erwarte("Standard: Sz besser, Vorsprung bleibt, D_gelu mehr Luecke, Vorsprung waechst bei halben Daten", _u5(rng, adv2_half=0.15),
            **{"NK": "BESTANDEN", "H1a": "SZ_BESSER", "H1b": "SZ_BESSER", "H2a": "VORSPRUNG_BLEIBT", "H2b": "VORSPRUNG_BLEIBT", "H3": "GROESSERE_LUECKE", "H4": "WAECHST_BEI_WENIGER_DATEN"})
    erwarte("Di_z2 besser, Vorsprung weg (gleichwertig), Luecke kleiner, Vorsprung schrumpft", _u5(rng, adv1=-0.08, adv2=0.0, adv2_half=-0.05, gap_gelu=0.05, gap_df=0.15),
            **{"H1a": "H0", "H1b": "H0", "H2a": "VORSPRUNG_VERSCHWUNDEN", "H2b": "VORSPRUNG_VERSCHWUNDEN", "H3": "KLEINERE_LUECKE", "H4": "SCHRUMPFT_BEI_WENIGER_DATEN"})
    u_gl = _u5(rng, adv2=0.0)
    pruefe("H2 gleichwertig: Text nennt TOST", u_gl["H2a"].startswith("VORSPRUNG_VERSCHWUNDEN (gleichwertig"))
    u_um = _u5(rng, adv2=-0.10, adv2_half=-0.10)
    pruefe("H2 umgekehrt (D_gelu sicher besser): VORSPRUNG_VERSCHWUNDEN mit Begruendung", u_um["H2b"].startswith("VORSPRUNG_VERSCHWUNDEN (volle Breite"))
    erwarte("Kein Unterschied (grosses Rauschen)", _u5(np.random.default_rng(101), adv1=0.0, adv2=0.0, adv2_half=0.0, gap_gelu=0.1, gap_df=0.1, sd=0.03),
            **{"H1a": "HU", "H1b": "HU", "H2a": "HU", "H2b": "HU", "H3": "HU", "H4": "HU"})
    erwarte("NK faellt", _u5(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1a": "HU (Negativkontrolle", "H2b": "HU (Negativkontrolle", "H3": "HU (Negativkontrolle", "H4": "HU (Negativkontrolle"})
    erwarte("Divergenz Sz d=512 T=16000 voll", _u5(rng, div=("Sz", 16000, 512, 1.0)), **{"H1a": "SZ_BESSER", "H1b": "HU (Divergenz", "H2a": "VORSPRUNG_BLEIBT", "H2b": "VORSPRUNG_BLEIBT"})
    erwarte("Kollaps Df_z2 d=256 T=16000 voll", _u5(rng, koll=("Df_z2", 16000, 256, 1.0)), **{"H1a": "SZ_BESSER", "H2a": "HU (Kollaps", "H2b": "VORSPRUNG_BLEIBT", "H3": "HU (Kollaps", "H4": "HU (Kollaps"})
    erwarte("Divergenz D_gelu bei halber Datenmenge stoert nur H4", _u5(rng, div=("D_gelu", 16000, 512, 0.5)), **{"H2b": "VORSPRUNG_BLEIBT", "H3": "GROESSERE_LUECKE", "H4": "HU (Divergenz"})
    erwarte("Divergenz bei T=4000 stoert nichts", _u5(rng, div=("Sz", 4000, 256, 1.0), adv2_half=0.15), **{"H1a": "SZ_BESSER", "H4": "WAECHST_BEI_WENIGER_DATEN"})
    erwarte("Zelle fehlt (Di_z2 d=512)", _u5(rng, fehlt=("Di_z2", 16000, 512, 1.0)), **{"H1a": "SZ_BESSER", "H1b": "HU (Zellen fehlen"})
    p = {f"{a}|{d}": {"ffn": 1, "gesamt": 2} for a in ARME_V5 for d in D_WERTE_V5}
    a_ = auswertung_v5(_synth_v5(rng, div=("Sz", 16000, 256, 1.0)), p, 5.0)
    pruefe("Divergenzrate 1/8, Seed bleibt in den Zahlen (n = 8)", abs(a_["divergenzrate"]["Sz|T16000|d256|D100"] - 0.125) < 1e-9
           and a_["zellen"]["Sz|T16000|d256|D100"]["n"] == 8 and a_["diff"]["H1a"]["n"] == 8)
    pruefe("Holm nur ueber H1a, H1b, H2a, H2b (H3, H4 nicht)", set(a_["p_holm"]) == {"H1a", "H1b", "H2a", "H2b"})
    pruefe("Train-bpc und Luecke je Zelle ausgewiesen", "Sz|T16000|d256|D50" in a_["train"] and abs(a_["luecke"]["D_gelu|T16000|d512|D100"]["mittel"] - 0.20) < 0.01)
    pruefe("Beschreibend: Vorsprung Sz gegen alle Arme, W je Zelle", "Vorsprung_Sz_gegen_Di_z2" in a_["beschreibend"] and "W_D_gelu_minus_Df_z2" in a_["beschreibend"]
           and "T4000|d256|D50" in a_["beschreibend"]["W_D_gelu_minus_Df_z2"])
    erlaubt = ("SZ_BESSER", "VORSPRUNG_BLEIBT", "VORSPRUNG_VERSCHWUNDEN", "GROESSERE_LUECKE", "KLEINERE_LUECKE", "WAECHST_BEI_WENIGER_DATEN",
               "SCHRUMPFT_BEI_WENIGER_DATEN", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u5(rng).values()))
    soll = {"H1a": {"SZ_BESSER", "H0", "HU"}, "H1b": {"SZ_BESSER", "H0", "HU"}, "H2a": {"VORSPRUNG_BLEIBT", "VORSPRUNG_VERSCHWUNDEN", "HU"},
            "H2b": {"VORSPRUNG_BLEIBT", "VORSPRUNG_VERSCHWUNDEN", "HU"}, "H3": {"GROESSERE_LUECKE", "KLEINERE_LUECKE", "HU"},
            "H4": {"WAECHST_BEI_WENIGER_DATEN", "SCHRUMPFT_BEI_WENIGER_DATEN", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for kk, w in gesehen if kk == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    return bool(ok)


_vortest_bis_v4 = vortest


def vortest() -> bool:
    """Alle Vortests der Datei (bis v4 unveraendert) plus I-045-v5."""
    ok = _vortest_bis_v4()
    ok &= _vortest_v5()
    return bool(ok)
