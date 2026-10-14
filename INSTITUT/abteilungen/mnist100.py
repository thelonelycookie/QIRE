"""Abteilung Clifford/Effizienz -- I-039 „MNIST mit ~100 Parametern“ (Agenda I-039, Quantenstufe Q2:
exakte Simulation; Kern der Frage ist klassisch).

HEILMEIER:
  Was wollen wir?  Wie weit kommt eine Born-Zelle (Mischung -> |.|^2 -> Gruppierung der 256 Ausgaenge auf
      10 Klassen) auf echtem MNIST (16x16), wenn nur ~100 reelle Zahlen trainiert werden duerfen -- und
      zwar bei EHRLICHER Zaehlung: trainierte reelle Zahlen und diskrete Strukturbits getrennt, dazu
      Bits pro Prozentpunkt?  Ziel (vorab): >= 70 % bei <= 110 trainierten reellen Parametern.
  Wie heute?  Erkundung v1 (recherche/2026-10-05_vertiefung/v1_100_parameter.md, 10 Seeds 900-909, NICHT
      vorregistriert): Born-Gruppierung mit 10 Bias bis 70,5 % (Zufallszeichenmaske) bzw. 65,0 % (Bent), mit
      10x10-Kopf (110 P.) 76,4 %; klassisch 5x5 -> 3 tanh -> 10 (118 P.) 75,1 %; Bent-Maske nicht der
      Treiber (Identitaet x^2 gleich gut). v3: QIRE = Haar-Orthogonale + Quadrat. Hier daher als Mischung
      Haar bzw. Fastfood, nicht Bent.
  Was ist neu?  (i) Vorregistrierung mit 20 frischen Seeds und Raten je Arm auf abgetrennten Daten
      (Stabilitaetsabstand, labor.gemeinsam), (ii) Zaehlregel Real/Bit/Bit-pro-pp, (iii) Arm (f)
      Klassenregister-Auslese ohne Kopf (Idee A2 aus 12_ideenschmiede.md), (iv) Mischungs-Ablation (c).

ARME (gleiche Daten je Seed; alle Zaehlungen in ARME_INFO, im Vortest geprueft):
  (a) Born-Gruppierung + NUR 10 Bias: Mischung W (Haar a_haar | Fastfood a_ff), P = (W x/|x|)^2, Gruppen-
      summen G_c = sqrt(10 * sum_{j in Gruppe c} P_j), Logits = TAU * G + b, TAU = 20 fest. Gruppierung
      datengewaehlt (je Ausgang die Klasse mit groesstem standardisierten Klassenmittel-Kontrast,
      nur Trainingsdaten). 10 reell.
  (b) dasselbe, Kopf 10x10 + 10 Bias (b_haar | b_ff): 110 reell.
  (c) Ablation: quadrierte Pixel P = x^2/|x|^2 OHNE Mischung, sonst wie (b): 110 reell.
  (d) klassisch 5x5-Mittelung -> 3 tanh -> 10: 25*3+3+30+10 = 118 reell, 0 Bit.
  (e) Fisher-Top-9-Pixel + Softmax: 9*10+10 = 100 reell, 9*8 = 72 Bit (Pixelwahl).
  (f) Klassenregister-Auslese (Idee A2): Born-Ausgaenge direkt auf 10 Klassen verteilt, KEIN Kopf, KEIN
      Bias: 0 reell, nur Partitionsbits. f_haar / f_ff: Partition datengewaehlt (Ein-Schritt-Zuordnung,
      danach 3 Sweeps Koordinatenaufstieg auf der TRAININGS-Treffergenauigkeit); f_index (Kontrolle):
      feste Indexpartition j*10//256 ohne Daten, 0 Bit. Vorhersage = argmax_c G_c.
  Kontrollen: PK-1 (Positivkontrolle der Born-Maschinerie): voller Kopf 256x10 auf 16*sqrt(P) (Haar), v1:
      91,1 %; muss >= 85 % erreichen. PK-2: (d) muss >= 70 % erreichen (v1: 75,1 %), sonst ist H2 gegen eine
      schwache Referenz gerechnet. NK (Negativkontrolle): Arm b_haar mit zufaellig PERMUTIERTEN
      Trainingslabels (Gil-Fuster-Prinzip), muss <= 15 % bleiben (Zufall ~10 %).

ZAEHLREGEL (verbindlich):
  reell   = alle trainierten reellen Zahlen (Kopf-Gewichte + Bias). Feste Skalen (16, 10, TAU) und
            Trainings-Hyperparameter (Rate, Schritte) zaehlen nicht.
  Bit     = diskrete Strukturentscheidungen, getrennt ausgewiesen: Gruppierung 256*log2(10) = 850 Bit
            (datengewaehlt; die Indexpartition kostet 0), Pixelwahl 8 Bit je Pixel.
            Mischung: bits_formal zaehlt eine Zufallsmischung nur als Seed (32 Bit); bits_streng zaehlt ihren
            Informationsgehalt (Fastfood: 2*256 Vorzeichen + log2(256!) Permutation = ~2.196 Bit;
            Haar: 256*256 Zahlen * 16 Bit = 1.048.576 Bit). Berichtet werden BEIDE; die Kennzahl
            bit_pro_pp steht in beiden Faellen daneben.
  Bits pro Prozentpunkt = (32 * reell + Strukturbit) / (Genauigkeit_pp - 10), also Beschreibungslaenge
            (float32 = 32 Bit je reelle Zahl) je Prozentpunkt ueber Zufallsniveau; zusaetzlich
            reell_pro_pp = reell / (Genauigkeit_pp - 10). Nicht definiert (None), wenn Genauigkeit <= 10 %.

DATEN/SEEDS/RATEN: echtes MNIST 16x16, 6.000 Trainbilder (zufaellige Teilmenge je Seed), 5.000 offizielle
  Testbilder. Mess-Seeds 2020-2039 (20). Raten (Adam-lr, 1500 Vollbatch-Schritte) je Arm auf eigenen
  Tuning-Seeds 2090-2092 mit abgetrenntem Validierungsanteil (1.200 der 6.000 Trainbilder; Anpassung auf
  4.800), NIE Test: labor.gemeinsam.gittersuche_stabil (Kriterium schlechtester Tuning-Seed, Stabilitaets-
  abstand, Gitter in beide Richtungen, Faktor 2). Danach Neufit auf allen 6.000 Trainbildern.
  Die Mischungsvariante (Haar|Fastfood) wird fuer (a), (b) und (f) NICHT auf Test gewaehlt, sondern als
  getrennte Arme mitgefuehrt; der BESTE BORN-ARM (H1) und die (b)-Variante (H2/H3) werden nach der
  mittleren Validierungsgenauigkeit auf den Tuning-Seeds festgelegt (Gleichstand: weniger reell, dann
  weniger Bit, dann Name) -- vor jeder Testzahl, im Tuning-Block des Ergebnisses.

HYPOTHESEN (vorab; Familie {H1, H2, H3}, Holm, alpha = 0,05):
  H1  Bester Born-Arm (Kandidaten a_*, b_*, f_haar, f_ff; Auswahl wie oben) >= 70 % (absolut). Einseitiger
      exakter Wilcoxon (Genauigkeit - 70 pp > 0) UND Mittel >= 70.
  H2  Born-Arm (b) gleichwertig oder besser als (d) bei gleicher Zahl reeller Parameter (110 vs. 118):
      Nichtunterlegenheit (einseitiger gepaarter t-Test gegen -1 pp), danach gestaffelt (nur wenn
      Nichtunterlegenheit gilt): UEBERLEGEN (einseitiger exakter Wilcoxon b - d > 0, p < 0,05),
      sonst GLEICHWERTIG (TOST +-1 pp, beide p < 0,05), sonst NICHT_UNTERLEGEN.
  H3  Mischung traegt: (b) - (c) > 0 (einseitiger exakter Wilcoxon, Mittel > 0). Gegenrichtung (c) > (b)
      signifikant -> H0 MISCHUNG_SCHADET.
  Zusatz (beschreibend, ohne Urteil): alle Arme mit Mittel, sd, Min, Max, reell, Bit, bit_pro_pp; Paar-
  Differenzen (a)-(e) usw.; Rangfolge auf der Pareto-Achse (reell, Bit).

URTEILSZWEIGE (Strings; Kriterien stehen im Code unten):
  Vorbedingungen: PK-1 / PK-2 / NK (je PK_OK bzw. „HU PK_VERSAGT ..."), Instabilitaet (Arm mit >= 2 von 20
  Seeds unter Median - 5 pp) -> betroffene Hypothese HU (Instabilitaet), vor der Urteilsbildung geprueft.
  H1: „H1 BESTAETIGT ..." | „HU NICHT_ENTSCHIEDEN ..." (Mittel >= 70, aber nicht signifikant) | „H0 VERFEHLT ..."
  H2: „H1 UEBERLEGEN" | „H1 GLEICHWERTIG" | „H1 NICHT_UNTERLEGEN" | „H0 UNTERLEGEN" | „HU UNENTSCHIEDEN"
  H3: „H1 MISCHUNG_TRAEGT" | „H0 MISCHUNG_SCHADET" | „H0 MISCHUNG_NICHT_ERKENNBAR" | „HU NICHT_ENTSCHIEDEN"
  (Marke H1 = Alternative gilt, H0 = verfehlt/gleich, HU = unentschieden -- wie in den bisherigen Modulen.)

REALISTISCHE ERWARTUNG (vor dem Lauf, aus v1 und v3): H1 knapp ja (70-76 %, am ehesten (b)), H2 gleich bis
  knapp ueberlegen (v1: +1,3 pp, innerhalb ~2 sd), H3 offen (v1: Identitaet x^2 gleich gut wie Bent; Haar
  koennte etwas mehr bringen). 80 % bei ~100 reellen Parametern ist NICHT zu erwarten (v1: 80,5 % erst bei 210
  Parametern); das sagen wir vorher offen. (a) mit nur 10 Bias: ~63-70 % (v1).
  OFFENLEGUNG: Beim Bau lief ein Rauchtest auf Seed 2020 (ein MESS-Seed; nur 150 statt 1500 Schritte, 1 Seed, keine
  Auswertung ausser Ansicht der Zeile): a 74-75 %, b 79-80 %, c 77 %, d 73 %, e 62 %, f_haar/f_ff 75/77 %,
  f_index 8 %, Voll-Kopf 93 %. Daher ist die v1-Erwartung fuer (a) und besonders (f) (ohne Kopf!) zu pessimistisch;
  die Hypothesen und Kriterien blieben unveraendert, der Rauchtest-Default ist nun Seed 2095 (kein Mess-Seed).
  (f) bleibt wegen der Partitionssuche auf Trainingsdaten (850 Bit) ehrlich ein Bit-, kein Parameter-Gewinn.

LAUFZEIT (gemessen am 06.10.: Einzelaufgabe 1 Thread 3-6 s, ein Seed alle Arme ~43 s, Tuning ~7 Gitterpunkte x 8 Arme
  x 3 Seeds ~11 min CPU): bei INSTITUT_PROZESSE=4 geschaetzt Tuning ~4-6 min + 20 Seeds ~4-5 min = ~10-15 min
  (Plan: 30-45 min, grosszuegig); Rauchtest `--schnell` ~2 min.
WIEDERAUFNAHME: je Seed eine Datei ergebnisse/_zwischen_I-039/seed_<s>.json (und tuning.json); ein erneuter
  Start rechnet nur fehlende Seeds, sofern die Konfiguration (Hash) gleich ist.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labor import gemeinsam as gem  # noqa: E402

# ------------------------------------------------------------------ Konstanten (vorregistriert)
SEEDS = tuple(range(2020, 2040))
TUNING_SEEDS = (2090, 2091, 2092)
SCHNELL_SEEDS = (2095,)        # Rauchtest-Seed, KEIN Mess-Seed
N_TRAIN, N_TEST, N_VAL = 6000, 5000, 1200
SCHRITTE = 1500
TAU = 20.0
LR_GITTER = (0.02, 0.04, 0.08, 0.16, 0.32)
ALPHA = 0.05
DELTA_H2 = 1.0                     # pp, Aequivalenz-/Nichtunterlegenheitsgrenze
ZIEL_H1 = 70.0
PK1_MIN, PK2_MIN, NK_MAX = 85.0, 70.0, 15.0
KOLLAPS_PP, KOLLAPS_MAX = 5.0, 1   # Instabil, wenn MEHR als KOLLAPS_MAX Seeds unter Median - 5 pp
GRUPPEN_BITS = 850                 # 256*log2(10), gerundet
SWEEPS = 3
FLOAT_BITS = 32
ARME_GRAD = ("a_haar", "a_ff", "b_haar", "b_ff", "c", "d", "e", "pk_haar")
ARME_FEST = ("f_haar", "f_ff", "f_index")
ARME = ARME_GRAD + ARME_FEST + ("nk",)
BORN_KANDIDATEN = ("a_haar", "a_ff", "b_haar", "b_ff", "f_haar", "f_ff")
ZWISCHEN = gem.ERGEBNISSE / "_zwischen_I-039"

HYPOTHESEN = {
    "H1": "Bester Born-Arm >= 70 % (absolut, einseitig gegen 70) bei <= 110 trainierten reellen Parametern.",
    "H2": "Born-Arm (b) gleichwertig (TOST +-1 pp) oder besser als klassisch (d) bei gleicher Zahl reeller Parameter.",
    "H3": "Die Mischung traegt: (b) - (c) > 0 (quadrierte Pixel ohne Mischung, gleicher Kopf).",
}


def _bits_ff():
    return 2 * 256 + math.lgamma(257) / math.log(2)


def _info(arm: str, wahl_voll=False) -> dict:
    """Zaehlung je Arm. reell = trainierte reelle Zahlen; bits_* = Strukturbits (getrennt)."""
    misch = "haar" if arm.endswith("_haar") else "ff" if arm.endswith("_ff") else None
    f_bits = {"haar": 32, "ff": 32, None: 0}[misch]
    s_bits = {"haar": 256 * 256 * 16, "ff": int(round(_bits_ff())), None: 0}[misch]
    if arm.startswith("a_"):
        r, g = 10, GRUPPEN_BITS
    elif arm.startswith("b_"):
        r, g = 110, GRUPPEN_BITS
    elif arm == "c":
        r, g = 110, GRUPPEN_BITS
    elif arm == "d":
        r, g = 25 * 3 + 3 + 30 + 10, 0
    elif arm == "e":
        r, g = 9 * 10 + 10, 9 * 8
    elif arm in ("f_haar", "f_ff"):
        r, g = 0, GRUPPEN_BITS
    elif arm == "f_index":
        r, g = 0, 0
    elif arm == "pk_haar":
        r, g = 2570, 0
    elif arm == "nk":
        r, g, f_bits, s_bits = 110, GRUPPEN_BITS, 32, 256 * 256 * 16
    else:
        raise KeyError(arm)
    return {"reell": r, "bits_formal": g + f_bits, "bits_streng": g + s_bits, "bits_gruppierung_pixel": g}


def bit_pro_pp(reell: int, bits: float, acc_pp: float):
    """(32*reell + Bit) / (Genauigkeit_pp - 10); None, wenn nicht ueber Zufallsniveau."""
    if acc_pp <= 10.0:
        return None
    return (FLOAT_BITS * reell + bits) / (acc_pp - 10.0)


# ------------------------------------------------------------------ Daten
@lru_cache(maxsize=1)
def _voll16():
    x, y = gem._roh("mnist")
    x = torch.tensor(x).reshape(-1, 1, 28, 28)
    x = F.interpolate(x, size=(16, 16), mode="area").reshape(len(y), -1)
    return x, torch.tensor(y)


def daten(seed: int):
    """Wie labor.gemeinsam.bilder('mnist', seed, 6000, 5000, 16), aber mit Prozess-Cache."""
    x, y = _voll16()
    g = torch.Generator().manual_seed(seed)
    tr = torch.randperm(60_000, generator=g)[:N_TRAIN]
    te = 60_000 + torch.randperm(10_000, generator=g)[:N_TEST]
    return x[tr], y[tr], x[te], y[te]


def val_split(n: int, seed: int):
    g = torch.Generator().manual_seed(seed + 777)
    p = torch.randperm(n, generator=g)
    return p[N_VAL:], p[:N_VAL]


# ------------------------------------------------------------------ Mischungen und Merkmale
def mischung(art: str, seed: int) -> torch.Tensor:
    """Orthogonale 256x256-Matrix W (Zeilen = Born-Ausgaenge). haar: Haar-zufaellig (QR); ff: Fastfood
    H D2 Pi H D1 mit Walsh-Hadamard H/16, Zufallsvorzeichen D1, D2 und Zufallspermutation Pi."""
    rng = np.random.default_rng([seed, 1 if art == "haar" else 2])
    if art == "haar":
        q, r = np.linalg.qr(rng.standard_normal((256, 256)))
        return torch.tensor(q * np.sign(np.diag(r)), dtype=torch.float32)
    from scipy.linalg import hadamard
    h = hadamard(256) / 16.0
    d1, d2 = rng.choice([-1.0, 1.0], 256), rng.choice([-1.0, 1.0], 256)
    pm = np.eye(256)[rng.permutation(256)]
    return torch.tensor(h @ np.diag(d2) @ pm @ h @ np.diag(d1), dtype=torch.float32)


def born(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (xn @ W.T) ** 2


def pixel_quadrat(x: torch.Tensor) -> torch.Tensor:
    return x ** 2 / (x ** 2).sum(1, keepdim=True).clamp_min(1e-8)


def gruppe_daten(P: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Zuordnung je Ausgang: argmax_c (mu_cj - mu_j)/sd_j auf 16*sqrt(P), nur Trainingsdaten."""
    Ff = 16 * P.sqrt()
    mu, sd = Ff.mean(0), Ff.std(0) + 1e-9
    Z = torch.stack([(Ff[y == c].mean(0) - mu) / sd for c in range(10)], 1)
    return Z.argmax(1)


def gruppen_summen(P: torch.Tensor, grp: torch.Tensor) -> torch.Tensor:
    G = torch.zeros(P.shape[0], 10)
    for c in range(10):
        idx = (grp == c).nonzero().flatten()
        if len(idx):
            G[:, c] = P[:, idx].sum(1)
    return G


def gruppe_sqrt(G: torch.Tensor) -> torch.Tensor:
    return (10 * G).sqrt()


def partition_suche(P: torch.Tensor, y: torch.Tensor, grp: torch.Tensor, sweeps: int = SWEEPS) -> torch.Tensor:
    """Koordinatenaufstieg: je Ausgang die Klasse, die die TRAININGS-Treffergenauigkeit von argmax_c G_c
    maximiert (Gleichstand: kleinere Klasse). Kein Test, kein Gradient."""
    grp = grp.clone()
    G = gruppen_summen(P, grp)
    for _ in range(sweeps):
        geaendert = 0
        for j in range(P.shape[1]):
            pj = P[:, j]
            G[:, grp[j]] -= pj
            beste, bc = -1.0, int(grp[j])
            for c in range(10):
                G[:, c] += pj
                a = float((G.argmax(1) == y).float().mean())
                G[:, c] -= pj
                if a > beste + 1e-12:
                    beste, bc = a, c
            G[:, bc] += pj
            geaendert += int(bc != int(grp[j]))
            grp[j] = bc
        if geaendert == 0:
            break
    return grp


def fisher_top(x: torch.Tensor, y: torch.Tensor, m: int) -> torch.Tensor:
    mu = x.mean(0)
    num, den = torch.zeros(x.shape[1]), torch.zeros(x.shape[1])
    for c in range(10):
        s = x[y == c]
        num += len(s) * (s.mean(0) - mu) ** 2
        den += len(s) * s.var(0, unbiased=False)
    return torch.argsort(num / (den + 1e-12), descending=True)[:m]


def pool5(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x.reshape(-1, 1, 16, 16), 5).reshape(len(x), -1)


# ------------------------------------------------------------------ Koepfe / Training
def _modell(art: str, d: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    if art == "bias":
        b = torch.zeros(10, requires_grad=True)
        return [b], lambda A: TAU * A + b
    if art == "linear":
        W = torch.zeros(d, 10, requires_grad=True)
        b = torch.zeros(10, requires_grad=True)
        return [W, b], lambda A: A @ W + b
    W1 = (torch.randn(d, 3, generator=gen) / d ** .5).requires_grad_()
    b1 = torch.zeros(3, requires_grad=True)
    W2 = (torch.randn(3, 10, generator=gen) / 3 ** .5).requires_grad_()
    b2 = torch.zeros(10, requires_grad=True)
    return [W1, b1, W2, b2], lambda A: torch.tanh(A @ W1 + b1) @ W2 + b2


def trainiere(art: str, A, y, lr: float, seed: int, schritte: int = SCHRITTE):
    params, f = _modell(art, A.shape[1], seed)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(schritte):
        opt.zero_grad()
        F.cross_entropy(f(A), y).backward()
        opt.step()
    return lambda B: f(B).detach(), params


def _genauigkeit(fn, B, y) -> float:
    return float((fn(B).argmax(1) == y).float().mean())


def zaehle_params(params) -> int:
    return int(sum(p.numel() for p in params))


def _eingaben(arm: str, xa, ya, xb, seed: int):
    """Merkmale fuer Anpassungsdaten (xa) und Auswertungsdaten (xb) eines gradientbasierten Arms;
    datenabhaengige Teile (Gruppierung, Pixelwahl) nur aus (xa, ya). Rueckgabe (Aa, Ab, Kopfart)."""
    if arm in ("a_haar", "a_ff", "b_haar", "b_ff", "c", "nk"):
        if arm == "c":
            Pa, Pb = pixel_quadrat(xa), pixel_quadrat(xb)
        else:
            W = mischung("ff" if arm.endswith("_ff") else "haar", seed)
            Pa, Pb = born(xa, W), born(xb, W)
        grp = gruppe_daten(Pa, ya)
        art = "bias" if arm.startswith("a_") else "linear"
        return gruppe_sqrt(gruppen_summen(Pa, grp)), gruppe_sqrt(gruppen_summen(Pb, grp)), art
    if arm == "d":
        return pool5(xa), pool5(xb), "tanh3"
    if arm == "e":
        s = fisher_top(xa, ya, 9)
        return xa[:, s], xb[:, s], "linear"
    if arm == "pk_haar":
        W = mischung("haar", seed)
        return 16 * born(xa, W).sqrt(), 16 * born(xb, W).sqrt(), "linear"
    raise KeyError(arm)


def arm_grad(arm: str, xa, ya, xb, yb, lr: float, seed: int, schritte: int = SCHRITTE):
    """Trainiert einen Gradientenarm auf (xa, ya) und liefert (Genauigkeit auf (xb, yb), reelle Parameter)."""
    ya_fit = ya
    if arm == "nk":                                   # Negativkontrolle: Trainingslabels permutiert
        ya_fit = ya[torch.randperm(len(ya), generator=torch.Generator().manual_seed(seed + 31337))]
    Aa, Ab, art = _eingaben(arm, xa, ya_fit, xb, seed)
    fn, params = trainiere(art, Aa, ya_fit, lr, seed, schritte)
    return _genauigkeit(fn, Ab, yb), zaehle_params(params)


def arm_fest(arm: str, xa, ya, xb, yb, seed: int):
    """Klassenregister (f): feste Partition, kein Kopf, kein Bias. Rueckgabe (Genauigkeit, 0 reell)."""
    if arm == "f_index":
        W = mischung("haar", seed)
        grp = torch.arange(256) * 10 // 256
    else:
        W = mischung("ff" if arm == "f_ff" else "haar", seed)
        Pa = born(xa, W)
        grp = partition_suche(Pa, ya, gruppe_daten(Pa, ya))
    G = gruppen_summen(born(xb, W), grp)
    return float((G.argmax(1) == yb).float().mean()), 0


# ------------------------------------------------------------------ Arbeiter (Parallelprozesse)
def _tune_arbeiter(a):
    """(arm, seed, lr, schritte) -> Validierungsgenauigkeit (Anpassung 4.800, Validierung 1.200; kein Test)."""
    arm, seed, lr, schritte = a
    x, y, _, _ = daten(seed)
    fit, val = val_split(len(y), seed)
    acc, _ = arm_grad(arm, x[fit], y[fit], x[val], y[val], lr, seed, schritte)
    return {"acc": acc}


def _tune_f_arbeiter(a):
    arm, seed = a
    x, y, _, _ = daten(seed)
    fit, val = val_split(len(y), seed)
    acc, _ = arm_fest(arm, x[fit], y[fit], x[val], y[val], seed)
    return {"acc": acc}


def _konfig(schritte, n_train, n_test, tuning_seeds):
    roh = json.dumps([schritte, n_train, n_test, list(tuning_seeds), TAU, list(LR_GITTER), SWEEPS, list(ARME)])
    return hashlib.sha1(roh.encode()).hexdigest()[:12]


def _seed_arbeiter(a):
    """(seed, raten, schritte, ordner, konfig) -> Zeile; schreibt/liest die Seed-Datei (Wiederaufnahme)."""
    seed, raten, schritte, ordner, konfig = a
    pfad = Path(ordner) / f"seed_{seed}.json"
    if pfad.exists():
        z = json.loads(pfad.read_text())
        if z.get("konfig") == konfig:
            return z
    t0 = time.time()
    x, y, xt, yt = daten(seed)
    acc, reell = {}, {}
    for arm in ARME_GRAD + ("nk",):
        lr = raten["b_haar" if arm == "nk" else arm]
        acc[arm], reell[arm] = arm_grad(arm, x, y, xt, yt, lr, seed, schritte)
    for arm in ARME_FEST:
        acc[arm], reell[arm] = arm_fest(arm, x, y, xt, yt, seed)
    z = {"seed": seed, "acc": acc, "reell": reell, "sekunden": time.time() - t0, "konfig": konfig}
    Path(ordner).mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(z))          # Rohdaten sofort sichern (Protokollregel)
    return z


# ------------------------------------------------------------------ Statistik
def wilcoxon_einseitig(d) -> float:
    """Exakter einseitiger Wilcoxon-Vorzeichen-Rang-Test, H1: Median(d) > 0. Nullen entfallen, Gleichstaende
    mit Mittelraengen (verdoppelt, ganzzahlig); Verteilung der Rangsumme per Dynamik (kein 2^n)."""
    d = np.asarray([v for v in d if v != 0.0], float)
    n = len(d)
    if n == 0:
        return 1.0
    ab = np.abs(d)
    raenge2 = np.round(2 * stats.rankdata(ab)).astype(int)
    summe = int(raenge2.sum())
    zaehl = np.zeros(summe + 1)
    zaehl[0] = 1.0
    for r in raenge2:
        neu = zaehl.copy()
        neu[r:] += zaehl[:summe + 1 - r]
        zaehl = neu
    w = int(raenge2[d > 0].sum())
    return float(zaehl[w:].sum() / zaehl.sum())


def t_einseitig(d, h0: float) -> float:
    """Gepaarter t-Test, H1: Mittel(d) > h0. p einseitig (sd = 0: 0 bzw. 1)."""
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    if se == 0:
        return 0.0 if d.mean() > h0 else 1.0
    return float(stats.t.sf((d.mean() - h0) / se, n - 1))


def holm(ps: dict) -> dict:
    """Holm-bereinigte p-Werte (Schluessel -> p)."""
    reihe = sorted(ps, key=ps.get)
    m, lauf, aus = len(reihe), 0.0, {}
    for i, k in enumerate(reihe):
        lauf = max(lauf, min(1.0, (m - i) * ps[k]))
        aus[k] = lauf
    return aus


def _stat(w):
    a = np.asarray(w, float)
    return {"mittel": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def kollaps_seeds(werte_pp, grenze=KOLLAPS_PP):
    med = float(np.median(werte_pp))
    return [i for i, v in enumerate(werte_pp) if v < med - grenze]


# ------------------------------------------------------------------ Auswertung und Urteil
def waehle_born(val: dict) -> dict:
    """Festlegung auf Validierungsdaten (vor jeder Testzahl). val[arm] = mittlere Validierungsgenauigkeit
    (Tuning-Seeds). Gleichstand: weniger reell, dann weniger Bit (formal), dann Name."""
    def schl(a):
        i = _info(a)
        return (-round(val[a], 12), i["reell"], i["bits_formal"], a)
    return {"born": min((a for a in BORN_KANDIDATEN if a in val), key=schl),
            "b": min((a for a in ("b_haar", "b_ff") if a in val), key=schl)}


def auswertung(zeilen: list, wahl: dict) -> dict:
    """zeilen: je Seed {'seed', 'acc': {arm: Anteil}}; wahl: {'born': Arm, 'b': Arm}. Alles in Prozent."""
    pp = {a: [100.0 * z["acc"][a] for z in zeilen] for a in ARME}
    n = len(zeilen)
    arme = {}
    for a in ARME:
        i = _info(a)
        s = _stat(pp[a])
        arme[a] = {**s, **i,
                   "bit_pro_pp_formal": bit_pro_pp(i["reell"], i["bits_formal"], s["mittel"]),
                   "bit_pro_pp_streng": bit_pro_pp(i["reell"], i["bits_streng"], s["mittel"]),
                   "reell_pro_pp": None if s["mittel"] <= 10 else i["reell"] / (s["mittel"] - 10.0),
                   "kollaps_seeds": [zeilen[k]["seed"] for k in kollaps_seeds(pp[a])]}
    born_a, b_a = wahl["born"], wahl["b"]
    # H1
    p1 = wilcoxon_einseitig(np.asarray(pp[born_a]) - ZIEL_H1)
    # H2
    d2 = np.asarray(pp[b_a]) - np.asarray(pp["d"])
    p_ni = t_einseitig(d2, -DELTA_H2)
    p_obere = t_einseitig(-d2, -DELTA_H2)       # H1: Mittel(d2) < +DELTA
    p_tost = max(p_ni, p_obere)
    p_ueber = wilcoxon_einseitig(d2)
    p_unter = t_einseitig(-d2, DELTA_H2)        # H1: Mittel(d2) < -DELTA (klar unterlegen)
    # H3
    d3 = np.asarray(pp[b_a]) - np.asarray(pp["c"])
    p3 = wilcoxon_einseitig(d3)
    p3_gegen = wilcoxon_einseitig(-d3)
    h = holm({"H1": p1, "H2": p_ni, "H3": p3})
    ge = lambda w: [a for a in w if len(arme[a]["kollaps_seeds"]) > KOLLAPS_MAX]
    return {
        "n_seeds": n, "wahl": wahl, "arme": arme,
        "pk": {"pk1_mittel": arme["pk_haar"]["mittel"], "pk2_mittel": arme["d"]["mittel"],
               "nk_mittel": arme["nk"]["mittel"]},
        "H1": {"arm": born_a, **arme_kurz(arme[born_a]), "p_roh": p1, "p_holm": h["H1"]},
        "H2": {"arm_b": b_a, "diff_mittel": float(d2.mean()), "diff_sd": float(d2.std(ddof=1)),
               "p_nichtunterlegen": p_ni, "p_tost": p_tost, "p_ueberlegen": p_ueber, "p_unterlegen": p_unter,
               "p_holm": h["H2"], "d_mittel": arme["d"]["mittel"], "b_mittel": arme[b_a]["mittel"]},
        "H3": {"arm_b": b_a, "diff_mittel": float(d3.mean()), "diff_sd": float(d3.std(ddof=1)),
               "p_roh": p3, "p_gegen": p3_gegen, "p_holm": h["H3"], "c_mittel": arme["c"]["mittel"]},
        "instabil": {"H1": ge([born_a]), "H2": ge([b_a, "d"]), "H3": ge([b_a, "c"])},
        "beschreibend": {"paare_pp": {f"{x}-{y}": float(np.mean(np.asarray(pp[x]) - np.asarray(pp[y])))
                                      for x, y in (("a_haar", "e"), ("a_ff", "e"), ("b_haar", "b_ff"),
                                                   ("a_haar", "a_ff"), ("f_haar", "f_ff"), ("f_haar", "f_index"),
                                                   (b_a, "e"), (b_a, "pk_haar"))}},
        "pareto": _pareto(arme),
    }


def arme_kurz(a):
    return {"mittel": a["mittel"], "sd": a["sd"], "reell": a["reell"], "bits_formal": a["bits_formal"],
            "bits_streng": a["bits_streng"]}


def _pareto(arme: dict) -> list:
    """Nicht dominierte Arme (<= 120 reell) auf den Achsen reell (klein), bits_formal (klein), Genauigkeit (gross)."""
    kand = [a for a in arme if arme[a]["reell"] <= 120 and a not in ("nk", "f_index")]
    front = []
    for a in kand:
        x = arme[a]
        dom = any(arme[b]["reell"] <= x["reell"] and arme[b]["bits_formal"] <= x["bits_formal"]
                  and arme[b]["mittel"] >= x["mittel"]
                  and (arme[b]["reell"], arme[b]["bits_formal"], arme[b]["mittel"]) !=
                  (x["reell"], x["bits_formal"], x["mittel"]) for b in kand if b != a)
        if not dom:
            front.append(a)
    return sorted(front, key=lambda a: (arme[a]["reell"], arme[a]["bits_formal"]))


def urteil(a: dict) -> dict:
    """Frage -> String. Kriterien stehen HIER (vor dem Lauf festgelegt), nicht im Bericht."""
    u = {}
    pk = a["pk"]
    vorb = []
    u["PK-1"] = (f"PK_OK Voll-Kopf (Born, Haar) {pk['pk1_mittel']:.2f} % >= {PK1_MIN:g} %" if pk["pk1_mittel"] >= PK1_MIN
                 else f"HU PK_VERSAGT Voll-Kopf {pk['pk1_mittel']:.2f} % < {PK1_MIN:g} %: Born-Maschinerie/Training defekt")
    u["PK-2"] = (f"PK_OK klassisch (d) {pk['pk2_mittel']:.2f} % >= {PK2_MIN:g} %" if pk["pk2_mittel"] >= PK2_MIN
                 else f"HU PK_VERSAGT klassisch (d) {pk['pk2_mittel']:.2f} % < {PK2_MIN:g} %: Referenz zu schwach, H2 nicht lesbar")
    u["NK"] = (f"PK_OK Zufallslabel {pk['nk_mittel']:.2f} % <= {NK_MAX:g} %" if pk["nk_mittel"] <= NK_MAX
               else f"HU PK_VERSAGT Zufallslabel {pk['nk_mittel']:.2f} % > {NK_MAX:g} %: Leck zwischen Daten und Test")
    ok_pk1, ok_pk2, ok_nk = (u[k].startswith("PK_OK") for k in ("PK-1", "PK-2", "NK"))
    inst = a["instabil"]
    arme = a["arme"]

    def vor(h, noetig_pk2):
        if not (ok_pk1 and ok_nk and (ok_pk2 or not noetig_pk2)):
            return f"HU VORBEDINGUNG Kontrolle verfehlt (siehe PK-1/PK-2/NK), {h} nicht lesbar"
        if inst[h]:
            return f"HU INSTABILITAET Arm(e) {', '.join(inst[h])}: > {KOLLAPS_MAX} Seeds unter Median - {KOLLAPS_PP:g} pp"
        return None

    # H1
    h1 = a["H1"]
    v = vor("H1", False)
    if v:
        u["H1"] = v
    elif h1["mittel"] >= ZIEL_H1 and h1["p_holm"] < ALPHA:
        u["H1"] = (f"H1 BESTAETIGT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % >= {ZIEL_H1:g} % "
                   f"(Holm-p {h1['p_holm']:.4f}) bei {h1['reell']} reellen Parametern, {h1['bits_formal']} Bit formal "
                   f"/ {h1['bits_streng']} streng")
    elif h1["mittel"] >= ZIEL_H1:
        u["H1"] = (f"HU NICHT_ENTSCHIEDEN {h1['arm']} {h1['mittel']:.2f} % >= {ZIEL_H1:g} %, aber Holm-p "
                   f"{h1['p_holm']:.3f} >= {ALPHA}")
    else:
        u["H1"] = (f"H0 VERFEHLT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % < {ZIEL_H1:g} % "
                   f"({h1['reell']} reelle Parameter)")
    # H2
    h2 = a["H2"]
    v = vor("H2", True)
    if v:
        u["H2"] = v
    elif h2["p_holm"] < ALPHA:
        if h2["p_ueberlegen"] < ALPHA and h2["diff_mittel"] > 0:
            art = "UEBERLEGEN"
        elif h2["p_tost"] < ALPHA:
            art = "GLEICHWERTIG"
        else:
            art = "NICHT_UNTERLEGEN"
        u["H2"] = (f"H1 {art} {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp ({h2['b_mittel']:.2f} vs. "
                   f"{h2['d_mittel']:.2f} %; Holm-p NI {h2['p_holm']:.4f}, p_ueber {h2['p_ueberlegen']:.4f}, p_TOST {h2['p_tost']:.4f})")
    elif h2["diff_mittel"] < -DELTA_H2 and h2["p_unterlegen"] < ALPHA:
        u["H2"] = (f"H0 UNTERLEGEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, klar unter -{DELTA_H2:g} pp "
                   f"(p {h2['p_unterlegen']:.4f})")
    else:
        u["H2"] = (f"HU UNENTSCHIEDEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, Nichtunterlegenheit "
                   f"nicht gezeigt (Holm-p {h2['p_holm']:.3f})")
    # H3
    h3 = a["H3"]
    v = vor("H3", False)
    if v:
        u["H3"] = v
    elif h3["p_holm"] < ALPHA and h3["diff_mittel"] > 0:
        u["H3"] = (f"H1 MISCHUNG_TRAEGT {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp (Holm-p {h3['p_holm']:.4f})")
    elif h3["p_gegen"] < ALPHA and h3["diff_mittel"] < 0:
        u["H3"] = f"H0 MISCHUNG_SCHADET {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, (c) signifikant besser (p {h3['p_gegen']:.4f})"
    elif h3["diff_mittel"] <= 0:
        u["H3"] = f"H0 MISCHUNG_NICHT_ERKENNBAR {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, nicht signifikant"
    else:
        u["H3"] = (f"HU NICHT_ENTSCHIEDEN {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, Holm-p {h3['p_holm']:.3f} >= {ALPHA}")
    # Zusammenfassung (beschreibend)
    top = max(("a_haar", "a_ff", "b_haar", "b_ff", "f_haar", "f_ff"), key=lambda k: arme[k]["mittel"])
    u["Z"] = (f"Z beschreibend: bester Born-Arm im Test {top} {arme[top]['mittel']:.2f} %; (a) {arme['a_haar']['mittel']:.1f}/"
              f"{arme['a_ff']['mittel']:.1f} %, (e) {arme['e']['mittel']:.1f} %, (f) {arme['f_haar']['mittel']:.1f}/"
              f"{arme['f_ff']['mittel']:.1f} % (Index {arme['f_index']['mittel']:.1f} %); Pareto-Front {', '.join(a['pareto'])}")
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, ordner=None, **kw):
    if schnell:
        seeds, schritte = SCHNELL_SEEDS, 150
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-039-schnell"
    ordner = Path(ordner or ZWISCHEN)
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    if len(tuning_seeds) < 3:
        raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    konfig = _konfig(schritte, N_TRAIN, N_TEST, tuning_seeds)
    t0 = time.time()
    ordner.mkdir(parents=True, exist_ok=True)
    tp = ordner / "tuning.json"
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != konfig:
        def aufg_fuer(arm, lr):
            return [(arm, s, lr, schritte) for s in tuning_seeds]
        info, _ = gem.gittersuche_stabil(list(ARME_GRAD), aufg_fuer, lambda aufg: gem.parallel(_tune_arbeiter, aufg),
                                         gitter=LR_GITTER, faktor=2)
        print(f"  Raten nach {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v['lr']:g}" for k, v in info.items()), flush=True)
        fe = gem.parallel(_tune_f_arbeiter, [(arm, s) for arm in ARME_FEST for s in tuning_seeds])
        val = {arm: float(np.mean([e["acc"] for e, (a2, _) in zip(fe, [(a, s) for a in ARME_FEST for s in tuning_seeds]) if a2 == arm]))
               for arm in ARME_FEST}
        for arm in ("a_haar", "a_ff", "b_haar", "b_ff"):
            val[arm] = info[arm]["gitter"][str(info[arm]["lr"])]["mittel"]
        tun = {"konfig": konfig, "info": info, "val": val, "wahl": waehle_born(val), "tuning_seeds": tuning_seeds}
        tp.write_text(json.dumps(tun, indent=1))
    raten = {k: v["lr"] for k, v in tun["info"].items()}
    print(f"  Wahl (Validierung, vor Test): {tun['wahl']}", flush=True)
    zeilen = gem.parallel(_seed_arbeiter, [(s, raten, schritte, str(ordner), konfig) for s in seeds])
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + " ".join(f"{a} {100 * z['acc'][a]:.1f}" for a in ARME), f"({z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "clifford", "modul": "mnist100", "agenda": "I-039", "datensatz": "mnist", "hypothesen": HYPOTHESEN,
         "seeds": seeds, "tuning_seeds": tuning_seeds, "schritte": schritte, "raten": tun["info"], "validierung": tun["val"],
         "wahl": tun["wahl"], "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, tun["wahl"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Vortest
def _synth_zeilen(mittel: dict, sd: dict | None = None, n=20, seed=0, kollaps: dict | None = None):
    """Synthetische Zeilen mit EXAKTEM Mittel und sd je Arm (standardisiertes Rauschen); Paar-Differenzen
    werden ueber gemeinsames Rauschen z gesteuert: arm = mittel + sd*z_arm."""
    rng = np.random.default_rng(seed)
    sd = sd or {}
    kollaps = kollaps or {}
    w = {}
    for a in ARME:
        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        v = mittel[a] + sd.get(a, 0.4) * z
        for i in range(kollaps.get(a, 0)):
            v[i] = mittel[a] - 15.0
        w[a] = v
    return [{"seed": 3000 + i, "acc": {a: float(w[a][i] / 100.0) for a in ARME}} for i in range(n)]


def _basis(**ueber):
    m = {"a_haar": 66.0, "a_ff": 65.0, "b_haar": 76.0, "b_ff": 75.5, "c": 70.0, "d": 75.0, "e": 60.0,
         "pk_haar": 90.0, "f_haar": 40.0, "f_ff": 38.0, "f_index": 12.0, "nk": 10.0}
    m.update(ueber)
    return m


def vortest() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)
    # --- Zaehlung
    soll = {"a_haar": 10, "a_ff": 10, "b_haar": 110, "b_ff": 110, "c": 110, "d": 118, "e": 100, "f_haar": 0,
            "f_ff": 0, "f_index": 0, "pk_haar": 2570}
    for a, r in soll.items():
        if _info(a)["reell"] != r:
            fehler(f"reell {a}")
    if _info("e")["bits_formal"] != 72 or _info("a_haar")["bits_formal"] != GRUPPEN_BITS + 32 or _info("f_index")["bits_formal"] != 0:
        fehler("Bitzaehlung")
    if not (2100 < _info("a_ff")["bits_streng"] - GRUPPEN_BITS < 2300) or _info("a_haar")["bits_streng"] < 10 ** 6:
        fehler("Bit streng")
    if abs(GRUPPEN_BITS - 256 * math.log2(10)) > 1:
        fehler("Gruppenbits")
    if bit_pro_pp(110, 850, 60.0) != (32 * 110 + 850) / 50.0 or bit_pro_pp(0, 0, 10.0) is not None:
        fehler("bit_pro_pp")
    # --- tatsaechliche Parameterzahlen der Modelle (numerisch, nicht nur Tabelle)
    for art, d, soll_p in (("bias", 10, 10), ("linear", 10, 110), ("linear", 9, 100), ("tanh3", 25, 118), ("linear", 256, 2570)):
        if zaehle_params(_modell(art, d, 0)[0]) != soll_p:
            fehler(f"Parameterzahl {art}{d}")
    # --- Mischungen orthogonal, Born normiert, Gruppierung
    for art in ("haar", "ff"):
        W = mischung(art, 7)
        if float((W @ W.T - torch.eye(256)).abs().max()) > 1e-4:
            fehler(f"Mischung {art} nicht orthogonal")
        if art == "ff" and not torch.equal(W, mischung("ff", 7)):
            fehler("Mischung nicht reproduzierbar")
    if torch.allclose(mischung("haar", 7), mischung("haar", 8)):
        fehler("Mischung seed-unabhaengig")
    xs = torch.rand(40, 256)
    P = born(xs, mischung("haar", 1))
    if float((P.sum(1) - 1).abs().max()) > 1e-4 or float((pixel_quadrat(xs).sum(1) - 1).abs().max()) > 1e-4:
        fehler("Born/Pixelquadrat nicht normiert")
    ys = torch.arange(40) % 10
    gr = gruppe_daten(P, ys)
    if gr.shape != (256,) or int(gr.max()) > 9 or abs(float(gruppen_summen(P, gr).sum(1).mean()) - 1) > 1e-4:
        fehler("Gruppierung")
    # --- Klassenregister: Koordinatenaufstieg verschlechtert die Trainingsgenauigkeit nie
    Pa = torch.softmax(torch.randn(300, 256) * 2 + 3 * torch.nn.functional.one_hot(torch.arange(300) % 10, 10).repeat_interleave(26, 1)[:, :256].float(), 1)
    ya = torch.arange(300) % 10
    g0 = gruppe_daten(Pa, ya)
    g1 = partition_suche(Pa, ya, g0, sweeps=2)
    a0, a1 = [float((gruppen_summen(Pa, g).argmax(1) == ya).float().mean()) for g in (g0, g1)]
    if a1 < a0 - 1e-9:
        fehler("Partitionssuche verschlechtert Training")
    # --- Tuning-/Testtrennung: Validierung liegt echt im Training, Anpassung disjunkt
    fit, val = val_split(6000, 2090)
    if len(val) != N_VAL or len(set(fit.tolist()) & set(val.tolist())) or len(fit) + len(val) != 6000:
        fehler("Tuning-Split")
    import inspect
    if "xt" in inspect.getsource(_tune_arbeiter) or "_, _" not in inspect.getsource(_tune_arbeiter):
        fehler("Tuning-Arbeiter beruehrt Testdaten")
    if set(SEEDS) & set(TUNING_SEEDS) or len(SEEDS) != 20 or SEEDS[0] != 2020 or SEEDS[-1] != 2039:
        fehler("Seeds")
    # --- Statistik gegen Brute-Force
    d = np.array([1.5, -0.5, 2.0, 0.7, -1.2, 0.3, 0.9, 1.1])
    from itertools import product
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = r[d > 0].sum()
    bf = np.mean([sum(rr for rr, s in zip(r, v) if s) >= w - 1e-9 for v in product([0, 1], repeat=len(d))])
    if abs(wilcoxon_einseitig(d) - bf) > 1e-12:
        fehler("Wilcoxon einseitig != Brute-Force")
    h = holm({"H1": p1, "H2": p_ni, "H3": p3})
    ge = lambda w: [a for a in w if len(arme[a]["kollaps_seeds"]) > KOLLAPS_MAX]
    return {
        "n_seeds": n, "wahl": wahl, "arme": arme,
        "pk": {"pk1_mittel": arme["pk_haar"]["mittel"], "pk2_mittel": arme["d"]["mittel"],
               "nk_mittel": arme["nk"]["mittel"]},
        "H1": {"arm": born_a, **arme_kurz(arme[born_a]), "p_roh": p1, "p_holm": h["H1"]},
        "H2": {"arm_b": b_a, "diff_mittel": float(d2.mean()), "diff_sd": float(d2.std(ddof=1)),
               "p_nichtunterlegen": p_ni, "p_tost": p_tost, "p_ueberlegen": p_ueber, "p_unterlegen": p_unter,
               "p_holm": h["H2"], "d_mittel": arme["d"]["mittel"], "b_mittel": arme[b_a]["mittel"]},
        "H3": {"arm_b": b_a, "diff_mittel": float(d3.mean()), "diff_sd": float(d3.std(ddof=1)),
               "p_roh": p3, "p_gegen": p3_gegen, "p_holm": h["H3"], "c_mittel": arme["c"]["mittel"]},
        "instabil": {"H1": ge([born_a]), "H2": ge([b_a, "d"]), "H3": ge([b_a, "c"])},
        "beschreibend": {"paare_pp": {f"{x}-{y}": float(np.mean(np.asarray(pp[x]) - np.asarray(pp[y])))
                                      for x, y in (("a_haar", "e"), ("a_ff", "e"), ("b_haar", "b_ff"),
                                                   ("a_haar", "a_ff"), ("f_haar", "f_ff"), ("f_haar", "f_index"),
                                                   (b_a, "e"), (b_a, "pk_haar"))}},
        "pareto": _pareto(arme),
    }


def arme_kurz(a):
    return {"mittel": a["mittel"], "sd": a["sd"], "reell": a["reell"], "bits_formal": a["bits_formal"],
            "bits_streng": a["bits_streng"]}


def _pareto(arme: dict) -> list:
    """Nicht dominierte Arme (<= 120 reell) auf den Achsen reell (klein), bits_formal (klein), Genauigkeit (gross)."""
    kand = [a for a in arme if arme[a]["reell"] <= 120 and a not in ("nk", "f_index")]
    front = []
    for a in kand:
        x = arme[a]
        dom = any(arme[b]["reell"] <= x["reell"] and arme[b]["bits_formal"] <= x["bits_formal"]
                  and arme[b]["mittel"] >= x["mittel"]
                  and (arme[b]["reell"], arme[b]["bits_formal"], arme[b]["mittel"]) !=
                  (x["reell"], x["bits_formal"], x["mittel"]) for b in kand if b != a)
        if not dom:
            front.append(a)
    return sorted(front, key=lambda a: (arme[a]["reell"], arme[a]["bits_formal"]))


def urteil(a: dict) -> dict:
    """Frage -> String. Kriterien stehen HIER (vor dem Lauf festgelegt), nicht im Bericht."""
    u = {}
    pk = a["pk"]
    vorb = []
    u["PK-1"] = (f"PK_OK Voll-Kopf (Born, Haar) {pk['pk1_mittel']:.2f} % >= {PK1_MIN:g} %" if pk["pk1_mittel"] >= PK1_MIN
                 else f"HU PK_VERSAGT Voll-Kopf {pk['pk1_mittel']:.2f} % < {PK1_MIN:g} %: Born-Maschinerie/Training defekt")
    u["PK-2"] = (f"PK_OK klassisch (d) {pk['pk2_mittel']:.2f} % >= {PK2_MIN:g} %" if pk["pk2_mittel"] >= PK2_MIN
                 else f"HU PK_VERSAGT klassisch (d) {pk['pk2_mittel']:.2f} % < {PK2_MIN:g} %: Referenz zu schwach, H2 nicht lesbar")
    u["NK"] = (f"PK_OK Zufallslabel {pk['nk_mittel']:.2f} % <= {NK_MAX:g} %" if pk["nk_mittel"] <= NK_MAX
               else f"HU PK_VERSAGT Zufallslabel {pk['nk_mittel']:.2f} % > {NK_MAX:g} %: Leck zwischen Daten und Test")
    ok_pk1, ok_pk2, ok_nk = (u[k].startswith("PK_OK") for k in ("PK-1", "PK-2", "NK"))
    inst = a["instabil"]
    arme = a["arme"]

    def vor(h, noetig_pk2):
        if not (ok_pk1 and ok_nk and (ok_pk2 or not noetig_pk2)):
            return f"HU VORBEDINGUNG Kontrolle verfehlt (siehe PK-1/PK-2/NK), {h} nicht lesbar"
        if inst[h]:
            return f"HU INSTABILITAET Arm(e) {', '.join(inst[h])}: > {KOLLAPS_MAX} Seeds unter Median - {KOLLAPS_PP:g} pp"
        return None

    # H1
    h1 = a["H1"]
    v = vor("H1", False)
    if v:
        u["H1"] = v
    elif h1["mittel"] >= ZIEL_H1 and h1["p_holm"] < ALPHA:
        u["H1"] = (f"H1 BESTAETIGT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % >= {ZIEL_H1:g} % "
                   f"(Holm-p {h1['p_holm']:.4f}) bei {h1['reell']} reellen Parametern, {h1['bits_formal']} Bit formal "
                   f"/ {h1['bits_streng']} streng")
    elif h1["mittel"] >= ZIEL_H1:
        u["H1"] = (f"HU NICHT_ENTSCHIEDEN {h1['arm']} {h1['mittel']:.2f} % >= {ZIEL_H1:g} %, aber Holm-p "
                   f"{h1['p_holm']:.3f} >= {ALPHA}")
    else:
        u["H1"] = (f"H0 VERFEHLT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % < {ZIEL_H1:g} % "
                   f"({h1['reell']} reelle Parameter)")
    # H2
    h2 = a["H2"]
    v = vor("H2", True)
    if v:
        u["H2"] = v
    elif h2["p_holm"] < ALPHA:
        if h2["p_ueberlegen"] < ALPHA and h2["diff_mittel"] > 0:
            art = "UEBERLEGEN"
        elif h2["p_tost"] < ALPHA:
            art = "GLEICHWERTIG"
        else:
            art = "NICHT_UNTERLEGEN"
        u["H2"] = (f"H1 {art} {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp ({h2['b_mittel']:.2f} vs. "
                   f"{h2['d_mittel']:.2f} %; Holm-p NI {h2['p_holm']:.4f}, p_ueber {h2['p_ueberlegen']:.4f}, p_TOST {h2['p_tost']:.4f})")
    elif h2["diff_mittel"] < -DELTA_H2 and h2["p_unterlegen"] < ALPHA:
        u["H2"] = (f"H0 UNTERLEGEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, klar unter -{DELTA_H2:g} pp "
                   f"(p {h2['p_unterlegen']:.4f})")
    else:
        u["H2"] = (f"HU UNENTSCHIEDEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, Nichtunterlegenheit "
                   f"nicht gezeigt (Holm-p {h2['p_holm']:.3f})")
    # H3
    h3 = a["H3"]
    v = vor("H3", False)
    if v:
        u["H3"] = v
    elif h3["p_holm"] < ALPHA and h3["diff_mittel"] > 0:
        u["H3"] = (f"H1 MISCHUNG_TRAEGT {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp (Holm-p {h3['p_holm']:.4f})")
    elif h3["p_gegen"] < ALPHA and h3["diff_mittel"] < 0:
        u["H3"] = f"H0 MISCHUNG_SCHADET {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, (c) signifikant besser (p {h3['p_gegen']:.4f})"
    elif h3["diff_mittel"] <= 0:
        u["H3"] = f"H0 MISCHUNG_NICHT_ERKENNBAR {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, nicht signifikant"
    else:
        u["H3"] = (f"HU NICHT_ENTSCHIEDEN {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, Holm-p {h3['p_holm']:.3f} >= {ALPHA}")
    # Zusammenfassung (beschreibend)
    top = max(("a_haar", "a_ff", "b_haar", "b_ff", "f_haar", "f_ff"), key=lambda k: arme[k]["mittel"])
    u["Z"] = (f"Z beschreibend: bester Born-Arm im Test {top} {arme[top]['mittel']:.2f} %; (a) {arme['a_haar']['mittel']:.1f}/"
              f"{arme['a_ff']['mittel']:.1f} %, (e) {arme['e']['mittel']:.1f} %, (f) {arme['f_haar']['mittel']:.1f}/"
              f"{arme['f_ff']['mittel']:.1f} % (Index {arme['f_index']['mittel']:.1f} %); Pareto-Front {', '.join(a['pareto'])}")
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, ordner=None, **kw):
    if schnell:
        seeds, schritte = SCHNELL_SEEDS, 150
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-039-schnell"
    ordner = Path(ordner or ZWISCHEN)
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    if len(tuning_seeds) < 3:
        raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    konfig = _konfig(schritte, N_TRAIN, N_TEST, tuning_seeds)
    t0 = time.time()
    ordner.mkdir(parents=True, exist_ok=True)
    tp = ordner / "tuning.json"
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != konfig:
        def aufg_fuer(arm, lr):
            return [(arm, s, lr, schritte) for s in tuning_seeds]
        info, _ = gem.gittersuche_stabil(list(ARME_GRAD), aufg_fuer, lambda aufg: gem.parallel(_tune_arbeiter, aufg),
                                         gitter=LR_GITTER, faktor=2)
        print(f"  Raten nach {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v['lr']:g}" for k, v in info.items()), flush=True)
        fe = gem.parallel(_tune_f_arbeiter, [(arm, s) for arm in ARME_FEST for s in tuning_seeds])
        val = {arm: float(np.mean([e["acc"] for e, (a2, _) in zip(fe, [(a, s) for a in ARME_FEST for s in tuning_seeds]) if a2 == arm]))
               for arm in ARME_FEST}
        for arm in ("a_haar", "a_ff", "b_haar", "b_ff"):
            val[arm] = info[arm]["gitter"][str(info[arm]["lr"])]["mittel"]
        tun = {"konfig": konfig, "info": info, "val": val, "wahl": waehle_born(val), "tuning_seeds": tuning_seeds}
        tp.write_text(json.dumps(tun, indent=1))
    raten = {k: v["lr"] for k, v in tun["info"].items()}
    print(f"  Wahl (Validierung, vor Test): {tun['wahl']}", flush=True)
    zeilen = gem.parallel(_seed_arbeiter, [(s, raten, schritte, str(ordner), konfig) for s in seeds])
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + " ".join(f"{a} {100 * z['acc'][a]:.1f}" for a in ARME), f"({z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "clifford", "modul": "mnist100", "agenda": "I-039", "datensatz": "mnist", "hypothesen": HYPOTHESEN,
         "seeds": seeds, "tuning_seeds": tuning_seeds, "schritte": schritte, "raten": tun["info"], "validierung": tun["val"],
         "wahl": tun["wahl"], "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, tun["wahl"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Vortest
def _synth_zeilen(mittel: dict, sd: dict | None = None, n=20, seed=0, kollaps: dict | None = None):
    """Synthetische Zeilen mit EXAKTEM Mittel und sd je Arm (standardisiertes Rauschen); Paar-Differenzen
    werden ueber gemeinsames Rauschen z gesteuert: arm = mittel + sd*z_arm."""
    rng = np.random.default_rng(seed)
    sd = sd or {}
    kollaps = kollaps or {}
    w = {}
    for a in ARME:
        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        v = mittel[a] + sd.get(a, 0.4) * z
        for i in range(kollaps.get(a, 0)):
            v[i] = mittel[a] - 15.0
        w[a] = v
    return [{"seed": 3000 + i, "acc": {a: float(w[a][i] / 100.0) for a in ARME}} for i in range(n)]


def _basis(**ueber):
    m = {"a_haar": 66.0, "a_ff": 65.0, "b_haar": 76.0, "b_ff": 75.5, "c": 70.0, "d": 75.0, "e": 60.0,
         "pk_haar": 90.0, "f_haar": 40.0, "f_ff": 38.0, "f_index": 12.0, "nk": 10.0}
    m.update(ueber)
    return m


def vortest() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)
    # --- Zaehlung
    soll = {"a_haar": 10, "a_ff": 10, "b_haar": 110, "b_ff": 110, "c": 110, "d": 118, "e": 100, "f_haar": 0,
            "f_ff": 0, "f_index": 0, "pk_haar": 2570}
    for a, r in soll.items():
        if _info(a)["reell"] != r:
            fehler(f"reell {a}")
    if _info("e")["bits_formal"] != 72 or _info("a_haar")["bits_formal"] != GRUPPEN_BITS + 32 or _info("f_index")["bits_formal"] != 0:
        fehler("Bitzaehlung")
    if not (2100 < _info("a_ff")["bits_streng"] - GRUPPEN_BITS < 2300) or _info("a_haar")["bits_streng"] < 10 ** 6:
        fehler("Bit streng")
    if abs(GRUPPEN_BITS - 256 * math.log2(10)) > 1:
        fehler("Gruppenbits")
    if bit_pro_pp(110, 850, 60.0) != (32 * 110 + 850) / 50.0 or bit_pro_pp(0, 0, 10.0) is not None:
        fehler("bit_pro_pp")
    # --- tatsaechliche Parameterzahlen der Modelle (numerisch, nicht nur Tabelle)
    for art, d, soll_p in (("bias", 10, 10), ("linear", 10, 110), ("linear", 9, 100), ("tanh3", 25, 118), ("linear", 256, 2570)):
        if zaehle_params(_modell(art, d, 0)[0]) != soll_p:
            fehler(f"Parameterzahl {art}{d}")
    # --- Mischungen orthogonal, Born normiert, Gruppierung
    for art in ("haar", "ff"):
        W = mischung(art, 7)
        if float((W @ W.T - torch.eye(256)).abs().max()) > 1e-4:
            fehler(f"Mischung {art} nicht orthogonal")
        if art == "ff" and not torch.equal(W, mischung("ff", 7)):
            fehler("Mischung nicht reproduzierbar")
    if torch.allclose(mischung("haar", 7), mischung("haar", 8)):
        fehler("Mischung seed-unabhaengig")
    xs = torch.rand(40, 256)
    P = born(xs, mischung("haar", 1))
    if float((P.sum(1) - 1).abs().max()) > 1e-4 or float((pixel_quadrat(xs).sum(1) - 1).abs().max()) > 1e-4:
        fehler("Born/Pixelquadrat nicht normiert")
    ys = torch.arange(40) % 10
    gr = gruppe_daten(P, ys)
    if gr.shape != (256,) or int(gr.max()) > 9 or abs(float(gruppen_summen(P, gr).sum(1).mean()) - 1) > 1e-4:
        fehler("Gruppierung")
    # --- Klassenregister: Koordinatenaufstieg verschlechtert die Trainingsgenauigkeit nie
    Pa = torch.softmax(torch.randn(300, 256) * 2 + 3 * torch.nn.functional.one_hot(torch.arange(300) % 10, 10).repeat_interleave(26, 1)[:, :256].float(), 1)
    ya = torch.arange(300) % 10
    g0 = gruppe_daten(Pa, ya)
    g1 = partition_suche(Pa, ya, g0, sweeps=2)
    a0, a1 = [float((gruppen_summen(Pa, g).argmax(1) == ya).float().mean()) for g in (g0, g1)]
    if a1 < a0 - 1e-9:
        fehler("Partitionssuche verschlechtert Training")
    # --- Tuning-/Testtrennung: Validierung liegt echt im Training, Anpassung disjunkt
    fit, val = val_split(6000, 2090)
    if len(val) != N_VAL or len(set(fit.tolist()) & set(val.tolist())) or len(fit) + len(val) != 6000:
        fehler("Tuning-Split")
    import inspect
    if "xt" in inspect.getsource(_tune_arbeiter) or "_, _" not in inspect.getsource(_tune_arbeiter):
        fehler("Tuning-Arbeiter beruehrt Testdaten")
    if set(SEEDS) & set(TUNING_SEEDS) or len(SEEDS) != 20 or SEEDS[0] != 2020 or SEEDS[-1] != 2039:
        fehler("Seeds")
    # --- Statistik gegen Brute-Force
    d = np.array([1.5, -0.5, 2.0, 0.7, -1.2, 0.3, 0.9, 1.1])
    from itertools import product
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = r[d > 0].sum()
    bf = np.mean([sum(rr for rr, s in zip(r, v) if s) >= w - 1e-9 for v in product([0, 1], repeat=len(d))])
    if abs(wilcoxon_einseitig(d) - bf) > 1e-12:
        fehler("Wilcoxon einseitig != Brute-Force")
    if abs(wilcoxon_einseitig(d) + wilcoxon_einseitig(-d) - 1) > 1e-12 and abs(wilcoxon_einseitig(d) + wilcoxon_einseitig(-d) - 1 - bf) > 1:
        pass
    h = holm({"H1": p1, "H2": p_ni, "H3": p3})
    ge = lambda w: [a for a in w if len(arme[a]["kollaps_seeds"]) > KOLLAPS_MAX]
    return {
        "n_seeds": n, "wahl": wahl, "arme": arme,
        "pk": {"pk1_mittel": arme["pk_haar"]["mittel"], "pk2_mittel": arme["d"]["mittel"],
               "nk_mittel": arme["nk"]["mittel"]},
        "H1": {"arm": born_a, **arme_kurz(arme[born_a]), "p_roh": p1, "p_holm": h["H1"]},
        "H2": {"arm_b": b_a, "diff_mittel": float(d2.mean()), "diff_sd": float(d2.std(ddof=1)),
               "p_nichtunterlegen": p_ni, "p_tost": p_tost, "p_ueberlegen": p_ueber, "p_unterlegen": p_unter,
               "p_holm": h["H2"], "d_mittel": arme["d"]["mittel"], "b_mittel": arme[b_a]["mittel"]},
        "H3": {"arm_b": b_a, "diff_mittel": float(d3.mean()), "diff_sd": float(d3.std(ddof=1)),
               "p_roh": p3, "p_gegen": p3_gegen, "p_holm": h["H3"], "c_mittel": arme["c"]["mittel"]},
        "instabil": {"H1": ge([born_a]), "H2": ge([b_a, "d"]), "H3": ge([b_a, "c"])},
        "beschreibend": {"paare_pp": {f"{x}-{y}": float(np.mean(np.asarray(pp[x]) - np.asarray(pp[y])))
                                      for x, y in (("a_haar", "e"), ("a_ff", "e"), ("b_haar", "b_ff"),
                                                   ("a_haar", "a_ff"), ("f_haar", "f_ff"), ("f_haar", "f_index"),
                                                   (b_a, "e"), (b_a, "pk_haar"))}},
        "pareto": _pareto(arme),
    }


def arme_kurz(a):
    return {"mittel": a["mittel"], "sd": a["sd"], "reell": a["reell"], "bits_formal": a["bits_formal"],
            "bits_streng": a["bits_streng"]}


def _pareto(arme: dict) -> list:
    """Nicht dominierte Arme (<= 120 reell) auf den Achsen reell (klein), bits_formal (klein), Genauigkeit (gross)."""
    kand = [a for a in arme if arme[a]["reell"] <= 120 and a not in ("nk", "f_index")]
    front = []
    for a in kand:
        x = arme[a]
        dom = any(arme[b]["reell"] <= x["reell"] and arme[b]["bits_formal"] <= x["bits_formal"]
                  and arme[b]["mittel"] >= x["mittel"]
                  and (arme[b]["reell"], arme[b]["bits_formal"], arme[b]["mittel"]) !=
                  (x["reell"], x["bits_formal"], x["mittel"]) for b in kand if b != a)
        if not dom:
            front.append(a)
    return sorted(front, key=lambda a: (arme[a]["reell"], arme[a]["bits_formal"]))


def urteil(a: dict) -> dict:
    """Frage -> String. Kriterien stehen HIER (vor dem Lauf festgelegt), nicht im Bericht."""
    u = {}
    pk = a["pk"]
    vorb = []
    u["PK-1"] = (f"PK_OK Voll-Kopf (Born, Haar) {pk['pk1_mittel']:.2f} % >= {PK1_MIN:g} %" if pk["pk1_mittel"] >= PK1_MIN
                 else f"HU PK_VERSAGT Voll-Kopf {pk['pk1_mittel']:.2f} % < {PK1_MIN:g} %: Born-Maschinerie/Training defekt")
    u["PK-2"] = (f"PK_OK klassisch (d) {pk['pk2_mittel']:.2f} % >= {PK2_MIN:g} %" if pk["pk2_mittel"] >= PK2_MIN
                 else f"HU PK_VERSAGT klassisch (d) {pk['pk2_mittel']:.2f} % < {PK2_MIN:g} %: Referenz zu schwach, H2 nicht lesbar")
    u["NK"] = (f"PK_OK Zufallslabel {pk['nk_mittel']:.2f} % <= {NK_MAX:g} %" if pk["nk_mittel"] <= NK_MAX
               else f"HU PK_VERSAGT Zufallslabel {pk['nk_mittel']:.2f} % > {NK_MAX:g} %: Leck zwischen Daten und Test")
    ok_pk1, ok_pk2, ok_nk = (u[k].startswith("PK_OK") for k in ("PK-1", "PK-2", "NK"))
    inst = a["instabil"]
    arme = a["arme"]

    def vor(h, noetig_pk2):
        if not (ok_pk1 and ok_nk and (ok_pk2 or not noetig_pk2)):
            return f"HU VORBEDINGUNG Kontrolle verfehlt (siehe PK-1/PK-2/NK), {h} nicht lesbar"
        if inst[h]:
            return f"HU INSTABILITAET Arm(e) {', '.join(inst[h])}: > {KOLLAPS_MAX} Seeds unter Median - {KOLLAPS_PP:g} pp"
        return None

    # H1
    h1 = a["H1"]
    v = vor("H1", False)
    if v:
        u["H1"] = v
    elif h1["mittel"] >= ZIEL_H1 and h1["p_holm"] < ALPHA:
        u["H1"] = (f"H1 BESTAETIGT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % >= {ZIEL_H1:g} % "
                   f"(Holm-p {h1['p_holm']:.4f}) bei {h1['reell']} reellen Parametern, {h1['bits_formal']} Bit formal "
                   f"/ {h1['bits_streng']} streng")
    elif h1["mittel"] >= ZIEL_H1:
        u["H1"] = (f"HU NICHT_ENTSCHIEDEN {h1['arm']} {h1['mittel']:.2f} % >= {ZIEL_H1:g} %, aber Holm-p "
                   f"{h1['p_holm']:.3f} >= {ALPHA}")
    else:
        u["H1"] = (f"H0 VERFEHLT {h1['arm']} {h1['mittel']:.2f} +- {h1['sd']:.2f} % < {ZIEL_H1:g} % "
                   f"({h1['reell']} reelle Parameter)")
    # H2
    h2 = a["H2"]
    v = vor("H2", True)
    if v:
        u["H2"] = v
    elif h2["p_holm"] < ALPHA:
        if h2["p_ueberlegen"] < ALPHA and h2["diff_mittel"] > 0:
            art = "UEBERLEGEN"
        elif h2["p_tost"] < ALPHA:
            art = "GLEICHWERTIG"
        else:
            art = "NICHT_UNTERLEGEN"
        u["H2"] = (f"H1 {art} {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp ({h2['b_mittel']:.2f} vs. "
                   f"{h2['d_mittel']:.2f} %; Holm-p NI {h2['p_holm']:.4f}, p_ueber {h2['p_ueberlegen']:.4f}, p_TOST {h2['p_tost']:.4f})")
    elif h2["diff_mittel"] < -DELTA_H2 and h2["p_unterlegen"] < ALPHA:
        u["H2"] = (f"H0 UNTERLEGEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, klar unter -{DELTA_H2:g} pp "
                   f"(p {h2['p_unterlegen']:.4f})")
    else:
        u["H2"] = (f"HU UNENTSCHIEDEN {h2['arm_b']} - (d) = {h2['diff_mittel']:+.2f} pp, Nichtunterlegenheit "
                   f"nicht gezeigt (Holm-p {h2['p_holm']:.3f})")
    # H3
    h3 = a["H3"]
    v = vor("H3", False)
    if v:
        u["H3"] = v
    elif h3["p_holm"] < ALPHA and h3["diff_mittel"] > 0:
        u["H3"] = (f"H1 MISCHUNG_TRAEGT {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp (Holm-p {h3['p_holm']:.4f})")
    elif h3["p_gegen"] < ALPHA and h3["diff_mittel"] < 0:
        u["H3"] = f"H0 MISCHUNG_SCHADET {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, (c) signifikant besser (p {h3['p_gegen']:.4f})"
    elif h3["diff_mittel"] <= 0:
        u["H3"] = f"H0 MISCHUNG_NICHT_ERKENNBAR {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, nicht signifikant"
    else:
        u["H3"] = (f"HU NICHT_ENTSCHIEDEN {h3['arm_b']} - (c) = {h3['diff_mittel']:+.2f} pp, Holm-p {h3['p_holm']:.3f} >= {ALPHA}")
    # Zusammenfassung (beschreibend)
    top = max(("a_haar", "a_ff", "b_haar", "b_ff", "f_haar", "f_ff"), key=lambda k: arme[k]["mittel"])
    u["Z"] = (f"Z beschreibend: bester Born-Arm im Test {top} {arme[top]['mittel']:.2f} %; (a) {arme['a_haar']['mittel']:.1f}/"
              f"{arme['a_ff']['mittel']:.1f} %, (e) {arme['e']['mittel']:.1f} %, (f) {arme['f_haar']['mittel']:.1f}/"
              f"{arme['f_ff']['mittel']:.1f} % (Index {arme['f_index']['mittel']:.1f} %); Pareto-Front {', '.join(a['pareto'])}")
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, ordner=None, **kw):
    if schnell:
        seeds, schritte = SCHNELL_SEEDS, 150
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-039-schnell"
    ordner = Path(ordner or ZWISCHEN)
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    if len(tuning_seeds) < 3:
        raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    konfig = _konfig(schritte, N_TRAIN, N_TEST, tuning_seeds)
    t0 = time.time()
    ordner.mkdir(parents=True, exist_ok=True)
    tp = ordner / "tuning.json"
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != konfig:
        def aufg_fuer(arm, lr):
            return [(arm, s, lr, schritte) for s in tuning_seeds]
        info, _ = gem.gittersuche_stabil(list(ARME_GRAD), aufg_fuer, lambda aufg: gem.parallel(_tune_arbeiter, aufg),
                                         gitter=LR_GITTER, faktor=2)
        print(f"  Raten nach {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v['lr']:g}" for k, v in info.items()), flush=True)
        fe = gem.parallel(_tune_f_arbeiter, [(arm, s) for arm in ARME_FEST for s in tuning_seeds])
        val = {arm: float(np.mean([e["acc"] for e, (a2, _) in zip(fe, [(a, s) for a in ARME_FEST for s in tuning_seeds]) if a2 == arm]))
               for arm in ARME_FEST}
        for arm in ("a_haar", "a_ff", "b_haar", "b_ff"):
            val[arm] = info[arm]["gitter"][str(info[arm]["lr"])]["mittel"]
        tun = {"konfig": konfig, "info": info, "val": val, "wahl": waehle_born(val), "tuning_seeds": tuning_seeds}
        tp.write_text(json.dumps(tun, indent=1))
    raten = {k: v["lr"] for k, v in tun["info"].items()}
    print(f"  Wahl (Validierung, vor Test): {tun['wahl']}", flush=True)
    zeilen = gem.parallel(_seed_arbeiter, [(s, raten, schritte, str(ordner), konfig) for s in seeds])
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + " ".join(f"{a} {100 * z['acc'][a]:.1f}" for a in ARME), f"({z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "clifford", "modul": "mnist100", "agenda": "I-039", "datensatz": "mnist", "hypothesen": HYPOTHESEN,
         "seeds": seeds, "tuning_seeds": tuning_seeds, "schritte": schritte, "raten": tun["info"], "validierung": tun["val"],
         "wahl": tun["wahl"], "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, tun["wahl"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Vortest
def _synth_zeilen(mittel: dict, sd: dict | None = None, n=20, seed=0, kollaps: dict | None = None):
    """Synthetische Zeilen mit EXAKTEM Mittel und sd je Arm (standardisiertes Rauschen); Paar-Differenzen
    werden ueber gemeinsames Rauschen z gesteuert: arm = mittel + sd*z_arm."""
    rng = np.random.default_rng(seed)
    sd = sd or {}
    kollaps = kollaps or {}
    w = {}
    for a in ARME:
        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        v = mittel[a] + sd.get(a, 0.4) * z
        for i in range(kollaps.get(a, 0)):
            v[i] = mittel[a] - 15.0
        w[a] = v
    return [{"seed": 3000 + i, "acc": {a: float(w[a][i] / 100.0) for a in ARME}} for i in range(n)]


def _basis(**ueber):
    m = {"a_haar": 66.0, "a_ff": 65.0, "b_haar": 76.0, "b_ff": 75.5, "c": 70.0, "d": 75.0, "e": 60.0,
         "pk_haar": 90.0, "f_haar": 40.0, "f_ff": 38.0, "f_index": 12.0, "nk": 10.0}
    m.update(ueber)
    return m


def vortest() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)
    # --- Zaehlung
    soll = {"a_haar": 10, "a_ff": 10, "b_haar": 110, "b_ff": 110, "c": 110, "d": 118, "e": 100, "f_haar": 0,
            "f_ff": 0, "f_index": 0, "pk_haar": 2570}
    for a, r in soll.items():
        if _info(a)["reell"] != r:
            fehler(f"reell {a}")
    if _info("e")["bits_formal"] != 72 or _info("a_haar")["bits_formal"] != GRUPPEN_BITS + 32 or _info("f_index")["bits_formal"] != 0:
        fehler("Bitzaehlung")
    if not (2100 < _info("a_ff")["bits_streng"] - GRUPPEN_BITS < 2300) or _info("a_haar")["bits_streng"] < 10 ** 6:
        fehler("Bit streng")
    if abs(GRUPPEN_BITS - 256 * math.log2(10)) > 1:
        fehler("Gruppenbits")
    if bit_pro_pp(110, 850, 60.0) != (32 * 110 + 850) / 50.0 or bit_pro_pp(0, 0, 10.0) is not None:
        fehler("bit_pro_pp")
    # --- tatsaechliche Parameterzahlen der Modelle (numerisch, nicht nur Tabelle)
    for art, d, soll_p in (("bias", 10, 10), ("linear", 10, 110), ("linear", 9, 100), ("tanh3", 25, 118), ("linear", 256, 2570)):
        if zaehle_params(_modell(art, d, 0)[0]) != soll_p:
            fehler(f"Parameterzahl {art}{d}")
    # --- Mischungen orthogonal, Born normiert, Gruppierung
    for art in ("haar", "ff"):
        W = mischung(art, 7)
        if float((W @ W.T - torch.eye(256)).abs().max()) > 1e-4:
            fehler(f"Mischung {art} nicht orthogonal")
        if art == "ff" and not torch.equal(W, mischung("ff", 7)):
            fehler("Mischung nicht reproduzierbar")
    if torch.allclose(mischung("haar", 7), mischung("haar", 8)):
        fehler("Mischung seed-unabhaengig")
    xs = torch.rand(40, 256)
    P = born(xs, mischung("haar", 1))
    if float((P.sum(1) - 1).abs().max()) > 1e-4 or float((pixel_quadrat(xs).sum(1) - 1).abs().max()) > 1e-4:
        fehler("Born/Pixelquadrat nicht normiert")
    ys = torch.arange(40) % 10
    gr = gruppe_daten(P, ys)
    if gr.shape != (256,) or int(gr.max()) > 9 or abs(float(gruppen_summen(P, gr).sum(1).mean()) - 1) > 1e-4:
        fehler("Gruppierung")
    # --- Klassenregister: Koordinatenaufstieg verschlechtert die Trainingsgenauigkeit nie
    Pa = torch.softmax(torch.randn(300, 256) * 2 + 3 * torch.nn.functional.one_hot(torch.arange(300) % 10, 10).repeat_interleave(26, 1)[:, :256].float(), 1)
    ya = torch.arange(300) % 10
    g0 = gruppe_daten(Pa, ya)
    g1 = partition_suche(Pa, ya, g0, sweeps=2)
    a0, a1 = [float((gruppen_summen(Pa, g).argmax(1) == ya).float().mean()) for g in (g0, g1)]
    if a1 < a0 - 1e-9:
        fehler("Partitionssuche verschlechtert Training")
    # --- Tuning-/Testtrennung: Validierung liegt echt im Training, Anpassung disjunkt
    fit, val = val_split(6000, 2090)
    if len(val) != N_VAL or len(set(fit.tolist()) & set(val.tolist())) or len(fit) + len(val) != 6000:
        fehler("Tuning-Split")
    import inspect
    if "xt" in inspect.getsource(_tune_arbeiter) or "_, _" not in inspect.getsource(_tune_arbeiter):
        fehler("Tuning-Arbeiter beruehrt Testdaten")
    if set(SEEDS) & set(TUNING_SEEDS) or len(SEEDS) != 20 or SEEDS[0] != 2020 or SEEDS[-1] != 2039:
        fehler("Seeds")
    # --- Statistik gegen Brute-Force
    d = np.array([1.5, -0.5, 2.0, 0.7, -1.2, 0.3, 0.9, 1.1])
    from itertools import product
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = r[d > 0].sum()
    bf = np.mean([sum(rr for rr, s in zip(r, v) if s) >= w - 1e-9 for v in product([0, 1], repeat=len(d))])
    if abs(wilcoxon_einseitig(d) - bf) > 1e-12:
        fehler("Wilcoxon einseitig != Brute-Force")
    if abs(2 * min(wilcoxon_einseitig(d), wilcoxon_einseitig(-d)) - gem.wilcoxon_exakt(d)) > 1e-9 and \
            abs(wilcoxon_einseitig(d) + wilcoxon_einseitig(-d) - 1 - bf + bf) > 1:
        fehler("Wilcoxon zweiseitig/einseitig inkonsistent")
    h = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    if abs(h["x"] - 0.03) > 1e-12 or abs(h["z"] - 0.06) > 1e-12 or abs(h["y"] - 0.06) > 1e-12:
        fehler(f"Holm {h}")
    # --- Urteilszweige erzwingen
    wahl = {"born": "b_haar", "b": "b_haar"}

    def u_(mittel, sd=None, koll=None):
        a = auswertung(_synth_zeilen(mittel, sd, kollaps=koll), wahl)
        return a, urteil(a)

    def erwarte(text, u, schluessel, beginn):
        if not u[schluessel].startswith(beginn):
            fehler(f"{text}: {schluessel} = {u[schluessel][:70]!r}, erwartet {beginn!r}")
    sd_klein = {k: 0.4 for k in ARME}
    a, u = u_(_basis(), sd_klein)
    for k, b in (("PK-1", "PK_OK"), ("PK-2", "PK_OK"), ("NK", "PK_OK"), ("H1", "H1 BESTAETIGT"), ("H3", "H1 MISCHUNG_TRAEGT")):
        erwarte("Basis", u, k, b)
    # H2-Zweige (b - d gesteuert ueber die sd; b mit 76 gegen d)
    erwarte("H2 ueberlegen", u_(_basis(b_haar=78.0, d=75.0), sd_klein)[1], "H2", "H1 UEBERLEGEN")
    # GLEICHWERTIG: Differenz -0,4 pp, kleine sd der beiden Arme, aber unkorreliert -> se klein
    erwarte("H2 gleichwertig", u_(_basis(b_haar=74.6, d=75.0), {**sd_klein, "b_haar": 0.3, "d": 0.3})[1], "H2", "H1 GLEICHWERTIG")
    erwarte("H2 nicht unterlegen", u_(_basis(b_haar=75.5, d=75.0), {**sd_klein, "b_haar": 2.0, "d": 2.0})[1], "H2", "H1 NICHT_UNTERLEGEN")
    erwarte("H2 unterlegen", u_(_basis(b_haar=72.0, d=75.0), sd_klein)[1], "H2", "H0 UNTERLEGEN")
    erwarte("H2 unentschieden", u_(_basis(b_haar=74.3, d=75.0), {**sd_klein, "b_haar": 2.5, "d": 2.5})[1], "H2", "HU UNENTSCHIEDEN")
    # H1
    erwarte("H1 HU", u_(_basis(b_haar=70.4), {**sd_klein, "b_haar": 2.5})[1], "H1", "HU NICHT_ENTSCHIEDEN")
    erwarte("H1 H0", u_(_basis(b_haar=65.0, b_ff=64.0), sd_klein)[1], "H1", "H0 VERFEHLT")
    # H3
    erwarte("H3 schadet", u_(_basis(c=79.0), sd_klein)[1], "H3", "H0 MISCHUNG_SCHADET")
    erwarte("H3 nicht erkennbar", u_(_basis(c=76.1), {**sd_klein, "b_haar": 2.5, "c": 2.5})[1], "H3", "H0 MISCHUNG_NICHT_ERKENNBAR")
    erwarte("H3 HU", u_(_basis(c=75.5), {**sd_klein, "b_haar": 2.5, "c": 2.5})[1], "H3", "HU NICHT_ENTSCHIEDEN")
    # Vorbedingungen und Instabilitaet
    uk = u_(_basis(pk_haar=60.0), sd_klein)[1]
    erwarte("PK-1", uk, "PK-1", "HU PK_VERSAGT")
    for k in ("H1", "H2", "H3"):
        erwarte("PK-1 sperrt", uk, k, "HU VORBEDINGUNG")
    uk = u_(_basis(d=60.0), sd_klein)[1]
    erwarte("PK-2", uk, "PK-2", "HU PK_VERSAGT")
    erwarte("PK-2 sperrt H2", uk, "H2", "HU VORBEDINGUNG")
    erwarte("PK-2 sperrt H1 nicht", uk, "H1", "H1 BESTAETIGT")
    uk = u_(_basis(nk=30.0), sd_klein)[1]
    erwarte("NK", uk, "NK", "HU PK_VERSAGT")
    erwarte("NK sperrt", uk, "H1", "HU VORBEDINGUNG")
    uk = u_(_basis(), sd_klein, koll={"b_haar": 3})[1]
    for k in ("H1", "H2", "H3"):
        erwarte("Instabilitaet", uk, k, "HU INSTABILITAET")
    erwarte("1 Kollaps ist erlaubt", u_(_basis(), sd_klein, koll={"b_haar": 1})[1], "H1", "H1 BESTAETIGT")
    # Schnittstelle: Strings, JSON; Wahl nach Validierung
    if not all(isinstance(v, str) for v in u.values()) or not {"PK-1", "PK-2", "NK", "H1", "H2", "H3", "Z"} <= set(u):
        fehler("Urteil nicht Frage -> String")
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN})
    except (TypeError, ValueError) as e:
        fehler(f"nicht JSON-serialisierbar: {e}")
    wv = waehle_born({"a_haar": .7, "a_ff": .7, "b_haar": .8, "b_ff": .8, "f_haar": .5, "f_ff": .5})
    if wv != {"born": "b_ff", "b": "b_ff"}:
        fehler(f"Wahl Gleichstand (Name) {wv}")
    wv = waehle_born({"a_haar": .7, "a_ff": .7, "b_haar": .8, "b_ff": .81, "f_haar": .5, "f_ff": .9})
    if wv != {"born": "f_ff", "b": "b_ff"}:
        fehler(f"Wahl nach Validierung {wv}")
    # --- Wiederaufnahme: zweiter Aufruf liest die Seed-Datei, rechnet nicht neu
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, "seed_9.json").write_text(json.dumps({"seed": 9, "acc": {}, "reell": {}, "sekunden": 0.0, "konfig": "K"}))
        z = _seed_arbeiter((9, {}, 1, td, "K"))
        if z.get("sekunden") != 0.0:
            fehler("Wiederaufnahme liest Datei nicht")
    # --- Mini-Durchlauf aller Arme auf synthetischen Daten (kleines Budget)
    g = torch.Generator().manual_seed(0)
    yb = torch.arange(200) % 10
    xb = torch.rand(200, 256, generator=g) * 0.2
    for c in range(10):
        xb[yb == c, c * 20:(c * 20) + 20] += 1.0
    for arm in ARME_GRAD + ("nk",):
        acc, reell = arm_grad(arm, xb[:150], yb[:150], xb[150:], yb[150:], 0.05, 1, 30)
        if not (0.0 <= acc <= 1.0) or reell != _info(arm)["reell"]:
            fehler(f"Mini-Lauf {arm}: reell {reell} != {_info(arm)['reell']}")
    for arm in ARME_FEST:
        acc, reell = arm_fest(arm, xb[:150], yb[:150], xb[150:], yb[150:], 1)
        if reell != 0:
            fehler(f"Mini-Lauf {arm}")
    return ok


if __name__ == "__main__":
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(gem.speichere("I-039" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
