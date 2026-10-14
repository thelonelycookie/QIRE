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
    def __init__(self, d, F_, art, akt, gen):
        super().__init__()
        r = rang_L(d, 1024)
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


def baue_ffn(arm, d, gen):
    F_ = BREITE[arm]
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
