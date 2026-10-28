"""Abteilung Struktur -- I-047 „Tensornetz-FFN (MPS/Tensor-Train) im Sprachmodell bei gleichen FLOPs“.
(Folge von I-045-v4: Fastfood + z² verlor bei gleichem Rechenaufwand klar gegen schmale dichte und Niedrigrang-FFN; die Fastfood-Struktur spart nur Speicher, keine FLOPs.
 Hier: ein Tensornetz, das Parameter UND FLOPs spart. Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

IDEE: Die beiden FFN-Matrizen (d -> 4d und 4d -> d) werden als Matrix-Produkt-Operator (MPO = Tensor-Train-Matrix, TT) dargestellt, einem Tensornetz aus der Quanten-Vielteilchenphysik
  (Matrix-Produkt-Zustaende/-Operatoren, DMRG). Eine Matrix W mit Zeilenindex i = (i1,i2,i3) und Spaltenindex j = (j1,j2,j3) ist
      W[i,j] = sum_{a,b} G1[1,i1,j1,a] · G2[a,i2,j2,b] · G3[b,i3,j3,1]       (Bonddimension chi = a,b)
  Literatur (nur Verifiziertes): Oseledets 2011 „Tensor-Train Decomposition“ (SIAM J. Sci. Comput. 33(5)); Novikov, Podoprikhin, Osokin, Vetrov 2015 „Tensorizing Neural Networks“
  (NeurIPS 2015, arXiv:1509.06569; TT-Layer statt voll verbundener Schicht); zur Quantenphysik: Schollwoeck 2011 „The density-matrix renormalization group in the age of matrix
  product states“ (Ann. Phys. 326, 96). Weitere TT-/Tensornetz-Layer in Transformern (z. B. Ma et al. 2019 „A Tensorized Transformer for Language Modeling“, NeurIPS 2019,
  arXiv:1906.09777) werden nur als verwandt genannt, ihre Zahlen nicht uebernommen.
FRAGE: Ist eine TT-FFN (Breite 4d, kleine Bonddimension chi) bei GLEICHEM FFN-Rechenaufwand (FLOPs/Token as implemented) besser als eine schmale dichte FFN (Df, h=580) bzw. eine Niedrigrang-FFN (Lk, Rang 116)?
  Zusatzfrage (beschreibend): traegt das Born-Quadrat z² auch im Tensornetz (TT_z2 gegen TT_gelu)?

DATEN / MODELL / TRAINING wie I-045-v4 (Bausteine aus struktur_sprache importiert, nichts kopiert ausser den Arbeitern, die in diesem Modul liegen muessen, damit die Prozesse die TT-Arme kennen):
  UD_German-GSD, Byte-Ebene, fester Split; Decoder-Transformer d=512, 4 Lagen, 8 Koepfe, Kontext 128, Batch 8x128, Adam, Kosinus, 5 % Aufwaermen, Clip 1,0.
  T=16000 (Hauptpunkt, Lernrate nur bei T=4000 gesucht -- wie v4), T=4000 (beschreibend). Checkpoint alle 2000 Schritte.

FAKTORISIERUNG (d=512): Eingang/Ausgang der Residualbahn 512 = 8·8·8; verdeckte Breite 2048 = 16·16·8 (gleiche Faktoren wie 8·16·16, grosse Moden zuerst: bei gleichen FLOPs
  ca. 30 % weniger Zeit im Messtest als die Reihenfolge 8·16·16).
  W1: 512 -> 2048, Moden (8,8,8) -> (16,16,8);  W2: 2048 -> 512, Moden (16,16,8) -> (8,8,8). Kerne G_k[r_{k-1}, n_k, m_k, r_k], Raenge (1, chi, chi, 1), je Matrix ein Bias (dicht).
  Kontraktion von links nach rechts, ein Matmul je Kern: Schritt k kostet 2·(m_1..m_{k-1})·(m_k·r_k)·(r_{k-1}·n_k)·(n_{k+1}..n_K) FLOPs je Token (as implemented, im Vortest gegen
  torch.utils.flop_counter gezaehlt). chi=4 -> 1.146.880 FFN-FLOPs je Lage (Sz/Df: 1.187.840, also -3,4 %); chi=5 waeren +41 %, daher chi=4. chi=2 (TT_z2_klein) -> 0,317 der Sz-FLOPs.
  (Suche ueber andere Zerlegungen: bei gleichem FLOP-Budget landet jede 3-Kern-Zerlegung bei chi = 4..5; chi hoeher geht nur mit sehr ungleichen Moden, die wir nicht wollen.)
ARME (nur die FFN verschieden):
  TT_z2       TT-FFN chi=4 + z²      (geprueft, Haupt)
  TT_gelu     TT-FFN chi=4 + GELU    (H3: traegt das Quadrat?)
  TT_z2_klein TT-FFN chi=2 + z²      (beschreibend, ca. 1/3 der FLOPs)
  Df_z2       dicht h=580 + z²       (v4-Gegner, FLOP-gleich, beschreibend)
  Df_gelu     dicht h=580 + GELU     (bester v4-Gegner, H1)
  Lk_z2       Niedrigrang 116, Breite 2048 + z²  (v4-Gegner, H2)
  Sz          Fastfood + z², Breite 2048 (Bezug zu v4, beschreibend)
HYPOTHESEN (Holm ueber H1, H2; gepaarter einseitiger t, n=8, T=16000, d=512, Differenz Referenz − TT_z2 je Seed in bpc, positiv = TT besser):
  H1 TT_z2 < Df_gelu: TT_BESSER | H0 (Df_gelu sicher besser) | HU.
  H2 TT_z2 < Lk_z2:   TT_BESSER | H0 (Lk_z2 sicher besser)  | HU.
  H3 (beschreibend, zweiseitig, NICHT in Holm, alpha 0,05) TT_gelu − TT_z2 bei T=16000: QUADRAT_TRAEGT (z² besser) | GELU_BESSER | HU.
DIVERGENZ / KOLLAPS / NEGATIVKONTROLLE wie v4 (NK: TT_z2 und Df_gelu, d=512, 1000 Schritte, permutierte Trainingsbytes, Seeds 2688/2689, Schwelle 0,9 x Unigramm).
"""
from __future__ import annotations

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
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil, zusammenfassung

KARTE_TT = "I-047"
SEEDS_TT = tuple(range(2680, 2688))
NK_SEEDS_TT = (2688, 2689)
TUNING_SEED_TT = 2693
D_TT = 512
T_WERTE_TT = (4000, 16000)
BUDGET_TOLERANZ_TT = 0.05

# ---------------------------------------------------------------- Tensornetz-Bausteine
# Moden je Modellbreite: (Eingang der Residualbahn, verdeckte Breite). d=512 ist die Messbreite; die kleineren Eintraege nur fuer Vor-/Rauchtests.
MODEN = {512: ((8, 8, 8), (16, 16, 8)), 256: ((8, 8, 4), (16, 8, 8)), 64: ((4, 4, 4), (8, 8, 4))}
# Arm -> (Aktivierung, Bonddimension chi)
ARME_TT = {"TT_z2": (torch.square, 4), "TT_gelu": (F.gelu, 4), "TT_z2_klein": (torch.square, 2)}


def max_rang(ns, ms, k):
    """Groesste moegliche Bonddimension an der Bindung k (1..K-1): min(Zeilen-/Spaltenmoden links, rechts)."""
    return min(math.prod(ns[:k]) * math.prod(ms[:k]), math.prod(ns[k:]) * math.prod(ms[k:]))


def raenge(ns, ms, chi):
    """(r_0, ..., r_K) mit r_0 = r_K = 1 und chi, hoechstens max_rang."""
    return [1] + [min(chi, max_rang(ns, ms, k)) for k in range(1, len(ns))] + [1]


def tt_param(ns, ms, r) -> int:
    return int(sum(r[k] * ns[k] * ms[k] * r[k + 1] for k in range(len(ns))))


def tt_flops(ns, ms, r) -> int:
    """FLOPs je Token der Kontraktion von links nach rechts (1 Multiplikation oder Addition = 1 Flop; ein Matmul je Kern)."""
    n = 0
    for k in range(len(ns)):
        n += 2 * math.prod(ms[:k]) * (ms[k] * r[k + 1]) * (r[k] * ns[k]) * math.prod(ns[k + 1:])
    return int(n)


class TTMatrix(nn.Module):
    """Matrix-Produkt-Operator W (N_ein x N_aus) aus K Kernen G_k[r_{k-1}, n_k, m_k, r_k]. Forward: x (..., N_ein) -> (..., N_aus), OHNE Bias.
    Initialisierung: jeder Kern Normal(0, s²) mit s² so, dass die Eintraege der Gesamtmatrix die Varianz 1/(3·N_ein) haben (wie nn.Linear-Standard)."""

    def __init__(self, ns, ms, chi, gen=None):
        super().__init__()
        self.ns, self.ms = tuple(ns), tuple(ms)
        self.r = raenge(self.ns, self.ms, chi)
        K = len(self.ns)
        pfade = math.prod(self.r[1:-1])
        s = (1.0 / (3.0 * math.prod(self.ns) * pfade)) ** (1.0 / (2 * K))
        self.kerne = nn.ParameterList([
            nn.Parameter(torch.randn(self.r[k], self.ns[k], self.ms[k], self.r[k + 1], generator=gen) * s) for k in range(K)])

    @property
    def n_ein(self):
        return math.prod(self.ns)

    @property
    def n_aus(self):
        return math.prod(self.ms)

    def forward(self, x):
        lead = x.shape[:-1]
        B = x.numel() // self.n_ein
        K = len(self.ns)
        z = x.reshape(B, self.ns[0], math.prod(self.ns[1:]))       # (B', r_0·n_1, Rest)
        for k in range(K):
            g = self.kerne[k].permute(2, 3, 0, 1).reshape(self.ms[k] * self.r[k + 1], self.r[k] * self.ns[k])
            z = torch.matmul(g, z)                                  # (B', m_k·r_{k+1}, Rest)
            if k + 1 < K:
                rest = math.prod(self.ns[k + 2:])
                z = z.reshape(-1, self.ms[k], self.r[k + 1], self.ns[k + 1], rest).reshape(-1, self.r[k + 1] * self.ns[k + 1], rest)
        return z.reshape(*lead, self.n_aus)

    def dense(self) -> torch.Tensor:
        """Volle Matrix (N_ein, N_aus) -- nur fuer Tests."""
        # eindeutige Fassung: Einheitsvektoren durch das Netz schicken (Zeile i von W = Bild des i-ten Einheitsvektors)
        eye = torch.eye(self.n_ein)
        with torch.no_grad():
            return self.forward(eye)

    @staticmethod
    def aus_dicht(W: torch.Tensor, ns, ms, chi=None) -> "TTMatrix":
        """TT-SVD (Oseledets 2011) einer dichten Matrix W (N_ein, N_aus); chi=None: voller Rang (exakt), sonst abgeschnitten."""
        K = len(ns)
        T = W.reshape(*ns, *ms).double()
        perm = [x for k in range(K) for x in (k, K + k)]
        T = T.permute(*perm).contiguous()                            # (n1,m1,n2,m2,...)
        m = TTMatrix.__new__(TTMatrix)
        nn.Module.__init__(m)
        m.ns, m.ms = tuple(ns), tuple(ms)
        r, kerne = [1], []
        rest = T.reshape(1, -1)
        for k in range(K - 1):
            M = rest.reshape(r[-1] * ns[k] * ms[k], -1)
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            rk = len(S) if chi is None else min(chi, len(S))
            kerne.append(U[:, :rk].reshape(r[-1], ns[k], ms[k], rk))
            rest = S[:rk, None] * Vh[:rk]
            r.append(rk)
        kerne.append(rest.reshape(r[-1], ns[-1], ms[-1], 1))
        r.append(1)
        m.r = r
        m.kerne = nn.ParameterList([nn.Parameter(g.float()) for g in kerne])
        return m


class FFNTT(nn.Module):
    """FFN d -> 4d -> d, beide Matrizen als TT (Moden aus MODEN), dichte Biase, Aktivierung akt."""

    def __init__(self, d, akt, chi, gen):
        super().__init__()
        e, h = MODEN[d]
        self.a, self.b = TTMatrix(e, h, chi, gen), TTMatrix(h, e, chi, gen)
        self.bias_a, self.bias_b = nn.Parameter(torch.zeros(math.prod(h))), nn.Parameter(torch.zeros(math.prod(e)))
        self.akt = akt

    def forward(self, x):
        return self.b(self.akt(self.a(x) + self.bias_a)) + self.bias_b


def ffn_tt_param_lage(d, chi) -> int:
    e, h = MODEN[d]
    return tt_param(e, h, raenge(e, h, chi)) + tt_param(h, e, raenge(h, e, chi)) + math.prod(h) + math.prod(e)


def ffn_tt_flops_lage(d, chi) -> int:
    e, h = MODEN[d]
    return tt_flops(e, h, raenge(e, h, chi)) + tt_flops(h, e, raenge(h, e, chi))


# Anbindung an struktur_sprache.baue_ffn (Block ruft den Modul-globalen Namen auf), ohne diese Datei zu veraendern.
if not getattr(ss.baue_ffn, "_tt", False):
    _orig_baue_ffn = ss.baue_ffn

    def _baue_ffn_tt(arm, d, gen):
        if arm in ARME_TT:
            akt, chi = ARME_TT[arm]
            return FFNTT(d, akt, chi, gen)
        return _orig_baue_ffn(arm, d, gen)

    _baue_ffn_tt._tt = True
    ss.baue_ffn = _baue_ffn_tt


# ---------------------------------------------------------------- Rechenaufwand as implemented
def ffn_flops_modul(ffn) -> int:
    """FLOPs einer FFN je Token: TT = gezaehlte Kontraktion, sonst wie struktur_sprache (Linear, Fastfood)."""
    if isinstance(ffn, FFNTT):
        return tt_flops(ffn.a.ns, ffn.a.ms, ffn.a.r) + tt_flops(ffn.b.ns, ffn.b.ms, ffn.b.r)
    return ss._ffn_flops_modul(ffn)


def flops_je_token(arm, d=D_TT, lagen=ss.LAGEN, ctx=ss.KONTEXT) -> dict:
    netz = ss.Sprachmodell(arm, 0, d=d, h=d // 64, lagen=1)
    ffn = ffn_flops_modul(netz.bl[0].ffn) * lagen
    att = lagen * (2 * d * 3 * d + 2 * d * d + 2 * ctx * d)
    return {"ffn": int(ffn), "gesamt": int(ffn + att + 2 * 256 * d)}


def flops_gemessen(arm, d=D_TT) -> int:
    """FFN-FLOPs je Token einer Lage, vom PyTorch-Zaehler (torch.utils.flop_counter) gezaehlt (Matmuls) -- unabhaengige Pruefung der Formeln."""
    from torch.utils.flop_counter import FlopCounterMode
    netz = ss.Sprachmodell(arm, 0, d=d, h=d // 64, lagen=1)
    x = torch.randn(1, 4, d)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        netz.bl[0].ffn(x)
    return int(fc.get_total_flops() // 4)


# ---------------------------------------------------------------- Arme
ARME_MESS_TT = ("TT_z2", "TT_gelu", "TT_z2_klein", "Df_z2", "Df_gelu", "Lk_z2", "Sz")
NK_ARME_TT = ("TT_z2", "Df_gelu")
PROFILE_TT = {"standard": {"T": T_WERTE_TT, "d": (D_TT,), "tuning_T": 4000, "nk_T": 1000, "nk_d": D_TT, "ckpt_alle": ss.CKPT_ALLE,
                           "seeds": SEEDS_TT, "nk_seeds": NK_SEEDS_TT}}
_BELEGT_TT = ss._BELEGT_V4 | set(range(2620, 2630)) | set(range(2640, 2650)) | set(range(2660, 2670)) | set(range(2694, 2700))
HYPOTHESEN_TT = {
    "H1": "TT_z2 < Df_gelu (bpc, gleiche FFN-FLOPs/Token, +-5 %) bei T=16000, d=512 (einseitig gepaart t, n=8, Holm H1-H2). TT_BESSER | H0 (Df_gelu sicher besser) | HU.",
    "H2": "TT_z2 < Lk_z2 (bpc, gleiche FFN-FLOPs/Token) bei T=16000, d=512 (einseitig gepaart t, n=8, Holm H1-H2). TT_BESSER | H0 (Lk_z2 sicher besser) | HU.",
    "H3": "beschreibend, zweiseitig, nicht in Holm (alpha 0,05): TT_gelu − TT_z2 bei T=16000 (gepaart t). QUADRAT_TRAEGT (z² besser) | GELU_BESSER | HU.",
    "DIVERGENZ": ss.HYPOTHESEN_V3["DIVERGENZ"],
    "NK": f"Negativkontrolle (TT_z2, Df_gelu, d=512, 1000 Schritte, permutierte Trainingsbytes, Seeds 2688/2689): mittlerer Test-bpc >= {ss.NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc aller Arme x T, Abstand zu Sz und Df_gelu, Pareto bpc x FFN-Parameter x Zeit/Token, FLOPs/Token, Inferenzzeit/Token, Trainingszeit, Divergenzrate, Lernraten.",
}
# Sekunden je Trainingsschritt (1 Thread, Batch 8x128, d=512), Messung im Container (Datum und Werte im LABORBUCH).
# Gemessen 27.10.2026 im Container (4 Kerne, 1 Thread, Median ueber 15 Schritte, Maximum zweier Messungen, Rechner zeitweise durch andere Prozesse belastet = konservativ).
S_JE_SCHRITT_TT = {"TT_z2": 1.41, "TT_gelu": 1.23, "TT_z2_klein": 0.81, "Df_z2": 0.59, "Df_gelu": 0.59, "Lk_z2": 0.74, "Sz": 0.90}


def form_text(arm, d=D_TT) -> dict:
    if arm in ARME_TT:
        e, h = MODEN[d]
        chi = ARME_TT[arm][1]
        return {"art": "tt", "moden_ein": list(e), "moden_hidden": list(h), "chi": chi, "raenge_W1": raenge(e, h, chi), "raenge_W2": raenge(h, e, chi)}
    if arm in ss.ARME_V4:
        art, h, r = ss.form_v4(arm, d)
        return {"art": art, "h": h, "rang": r}
    return {"art": "fastfood" if arm == "Sz" else "dicht", "h": 4 * d}


# ---------------------------------------------------------------- Arbeiter (muessen in DIESEM Modul liegen: spawn-Prozesse importieren es und kennen so die TT-Arme)
def _schl_tt(a):
    return ss._schl_v3(a)


def _arbeiter_tt(a):
    teil, arm, d, T, seed, lr, klein, ck_alle, ckordner = a
    key = ss._schl_v3(a)
    r = ss.trainiere_v3(arm, lr, seed, T, d=d, klein=klein, permutiert=(teil == "nk"), ckpt=ss._ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"teil": teil, "arm": arm, "d": d, "T": T, "seed": seed, "lr": lr, **r}


_ZAEHLER_TT = {"lr": 0}


def _lr_arbeiter_tt(a):
    schl, seed, lr, d, T, klein, ck_alle, ckordner, arm = a
    _ZAEHLER_TT["lr"] += 1
    key = ss._schl_lr(a)
    r = ss.trainiere_v3(arm, lr, seed, T, d=d, klein=klein, val=True, ckpt=ss._ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"acc": 1 - r["bpc"] / 8, "bpc": r["bpc"], "instabil": r["instabil"]}


def tuning_tt(P, klein, tseed, pfad, kennung, prozesse, ckordner, arme=None):
    """Lernrate je Arm bei T = tuning_T auf dem Tuning-Seed (gittersuche_stabil); jede Aufgabe einzeln wiederaufnehmbar."""
    schl = [f"{a}|{d}" for a in (arme or ARME_MESS_TT) for d in P["d"]]

    def aufgaben_fuer(s, lr):
        a, d = s.split("|")
        return [(s, tseed, lr, int(d), P["tuning_T"], klein, P["ckpt_alle"], str(ckordner), a)]

    def auswerten(aufg):
        lang = sorted(aufg, key=lambda a: -a[3])
        res = _rechne(lang, pfad, kennung, fn=_lr_arbeiter_tt, schluessel=ss._schl_lr, prozesse=prozesse)
        return [res[ss._schl_lr(a)] for a in aufg]

    info, _ = gittersuche_stabil(schl, aufgaben_fuer, auswerten, gitter=ss.LR_GITTER, faktor=ss.LR_FAKTOR, kollaps=ss.KOLLAPS_SCORE)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _zeit_arbeiter_tt(paare):
    torch.set_num_threads(1)
    return {f"{a}|{d}": ss.zeit_je_token(ss.Sprachmodell(a, 0, d=d, h=d // 64)) for a, d in paare}


def zeit_im_ein_thread_tt(paare):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter_tt, (tuple(paare),))


def schritt_zeiten_tt(arme=ARME_MESS_TT, d=D_TT, n=20, warm=3):
    """Messung: Sekunden je Trainingsschritt (1 Thread, Batch 8x128, Vorwaerts + Rueckwaerts + Clip + Adam), Median ueber n Schritte."""
    torch.set_num_threads(1)
    tr, _, _ = ss.daten(True)
    offs = torch.arange(ss.KONTEXT)
    g = torch.Generator().manual_seed(1)
    aus = {}
    for a in arme:
        netz = ss.Sprachmodell(a, 0, d=d, h=d // 64)
        opt = torch.optim.Adam(netz.parameters(), lr=1e-4)
        ts = []
        for k in range(warm + n):
            i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
            t0 = time.perf_counter()
            l = F.cross_entropy(netz(tr[i[:, None] + offs].long()).reshape(-1, 256), tr[i[:, None] + offs + 1].long().reshape(-1))
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
            opt.step()
            if k >= warm:
                ts.append(time.perf_counter() - t0)
        aus[a] = float(np.median(ts))
    return aus


# ---------------------------------------------------------------- Aufgaben und Kosten
def aufgaben_tt(hp, P, klein, ckordner):
    """Mess- und NK-Aufgaben, die langen (T = 16000, teuerste Arme) zuerst."""
    d = P["d"][0]
    aufg = [("mess", a, d, T, s, hp[f"{a}|{d}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
            for a in ARME_MESS_TT for T in P["T"] for s in P["seeds"]]
    aufg += [("nk", a, P["nk_d"], P["nk_T"], s, hp[f"{a}|{P['nk_d']}"]["lr"], klein, P["ckpt_alle"], str(ckordner))
             for a in NK_ARME_TT for s in P["nk_seeds"]]
    aufg.sort(key=lambda a: -a[3] * (S_JE_SCHRITT_TT.get(a[1], 1.0) if S_JE_SCHRITT_TT else 1.0))
    return aufg


def kosten_schaetzung_tt(s_je_schritt=None, prozesse=72, aufschlag=1.3, tuning_raten=4.5, tuning_runden=2, ueberhang=30.0):
    """Kernstunden und Wandzeit (LPT-Planung auf `prozesse` Kerne); aufschlag = Parallel-Aufschlag bei vollem Knoten (I-045 beobachtete 1,2-1,5)."""
    z = s_je_schritt or S_JE_SCHRITT_TT
    P = PROFILE_TT["standard"]
    dauer = lambda a, T: T * z[a] * aufschlag + ueberhang
    t_aufg = [dauer(a, P["tuning_T"]) for a in ARME_MESS_TT]
    kern_tuning = sum(t_aufg) * tuning_raten
    wand_tuning = tuning_runden * max(t_aufg)
    m_aufg = [dauer(a, T) for a in ARME_MESS_TT for T in P["T"] for _ in P["seeds"]]
    m_aufg += [dauer(a, P["nk_T"]) for a in NK_ARME_TT for _ in P["nk_seeds"]]
    last = [0.0] * prozesse
    heapq.heapify(last)
    for t in sorted(m_aufg, reverse=True):
        heapq.heappush(last, heapq.heappop(last) + t)
    wand_mess = max(last)
    return {"aufgaben_mess": len(m_aufg), "kernstunden_mess": sum(m_aufg) / 3600, "kernstunden_tuning": kern_tuning / 3600,
            "kernstunden_gesamt": (sum(m_aufg) + kern_tuning) / 3600, "laengste_aufgabe_h": max(m_aufg) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand_mess / 3600, "wand_gesamt_h": (wand_tuning + wand_mess) / 3600,
            "prozesse": prozesse, "aufschlag": aufschlag}


# ---------------------------------------------------------------- Auswertung (bpc: kleiner ist besser)
def _zname(a, T, d):
    return f"{a}|T{T}|d{d}"


def auswertung_tt(zeilen: list, params: dict, unigramm: float, t_lo=T_WERTE_TT[0], t_hi=T_WERTE_TT[-1], d=D_TT, flops=None, zeit=None) -> dict:
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
        r["kollaps"][n] = [s for s in seeds if not (v[s] <= med + ss.KOLLAPS_BPC)]
        r["divergiert"][n] = [s for s in seeds if inst[k][s] or not (v[s] < unigramm)]
        r["divergenzrate"][n] = len(r["divergiert"][n]) / len(seeds)
        ok = [v[s] for s in seeds if s not in r["divergiert"][n]]
        r["bpc_ohne_divergente"][n] = float(np.mean(ok)) if ok else None
        r["training_s_mittel"][n] = float(np.nanmean(list(tt[k].values())))
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def vorsprung(T, ref, pruef="TT_z2"):
        """Je Seed: bpc(ref) − bpc(pruef); positiv = pruef besser. {} wenn eine Zelle fehlt."""
        a, b = bpc.get((ref, T, d)), bpc.get((pruef, T, d))
        if not a or not b:
            return {}
        return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}

    for h, ref in (("H1", "Df_gelu"), ("H2", "Lk_z2")):
        v = vorsprung(t_hi, ref)
        if v:
            r["diff"][h] = ss._stat(np.array(list(v.values())))
    v = vorsprung(t_hi, "TT_gelu", pruef="TT_z2")          # H3: TT_gelu − TT_z2, positiv = z² besser
    if v:
        r["diff"]["H3"] = ss._stat(np.array(list(v.values())))
    for ref in ("Df_gelu", "Df_z2", "Lk_z2", "Sz", "TT_gelu", "TT_z2_klein"):
        for T in (t_lo, t_hi):
            v = vorsprung(T, ref)
            if v:
                r["beschreibend"].setdefault(f"Vorsprung_TT_z2_gegen_{ref}", {})[f"T{T}"] = ss._stat(np.array(list(v.values())))
    for a in ARME_MESS_TT:        # Abstand zum besten v4-Gegner Df_gelu: bpc(Arm) − bpc(Df_gelu), positiv = Arm schlechter
        if a == "Df_gelu":
            continue
        for T in (t_lo, t_hi):
            v = vorsprung(T, a, pruef="Df_gelu")
            if v:
                r["beschreibend"].setdefault(f"Abstand_{a}_zu_Df_gelu", {})[f"T{T}"] = ss._stat(np.array(list(v.values())))
    pt = {}
    for a in ARME_MESS_TT:
        n = _zname(a, t_hi, d)
        pk = f"{a}|{d}"
        if n in r["zellen"] and pk in params and (zeit is None or pk in zeit):
            pt[a] = (r["zellen"][n]["mittel"], params[pk]["ffn"], (zeit or {}).get(pk, 0.0))
    r["beschreibend"]["pareto_T_hi"] = {"punkte": {a: list(p) for a, p in pt.items()}, "nicht_dominiert": ss.pareto_nicht_dominiert(pt),
                                        "achsen": ["bpc", "ffn_parameter", "s_je_token" if zeit else "(Zeit nicht gemessen: 0)"]}
    if flops:
        r["beschreibend"]["flops_je_token"] = flops
    if zeit:
        r["beschreibend"]["zeit_s_je_token"] = zeit
    prim = {h: r["diff"][h]["p_gr"] for h in ("H1", "H2") if h in r["diff"] and "p_gr" in r["diff"][h]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _vorab_tt(r, nk, h, zellen):
    if h not in r["diff"] or "mittel" not in r["diff"][h]:
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if h in ("H1", "H2") and h not in r["p_holm"]:
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


def _zahlen_h3(r):
    d = r["diff"]["H3"]
    return (f"TT_gelu−TT_z2 {d['mittel']:+.4f} bpc, 90-%-KI [{d['ki'][0]:+.4f}; {d['ki'][1]:+.4f}], n {d['n']}, p_zweiseitig {d['p_zwei']:.3g}").replace(".", ",")


def urteil_tt(r: dict) -> dict:
    nk = ss._nk(r)
    u = {"NK": f"{nk} (Mittel permutiert TT_z2/Df_gelu: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {ss.NK_FAKTOR}x)"}
    t_hi, d = r["t_hi"], r["d"]
    for h, ref in (("H1", "Df_gelu"), ("H2", "Lk_z2")):
        v = _vorab_tt(r, nk, h, [_zname(a, t_hi, d) for a in (ref, "TT_z2")])
        if v:
            u[h] = v
            continue
        z = ss._zahlen(r, h)
        if r["p_holm"][h] < ss.ALPHA:
            u[h] = f"TT_BESSER ({ref}-TT_z2 bei T={t_hi}, d={d}, gleiche FFN-FLOPs: {z})"
        elif r["diff"][h]["p_kl"] < ss.ALPHA / 2:
            u[h] = f"H0 ({ref} sicher besser als TT_z2 bei T={t_hi}, d={d}; {z})"
        else:
            u[h] = f"HU ({z})"
    v = _vorab_tt(r, nk, "H3", [_zname(a, t_hi, d) for a in ("TT_gelu", "TT_z2")])
    if v:
        u["H3"] = v
    else:
        z = _zahlen_h3(r)
        p = r["diff"]["H3"]["p_zwei"]
        if p < ss.ALPHA:
            u["H3"] = f"QUADRAT_TRAEGT ({z})" if r["diff"]["H3"]["mittel"] > 0 else f"GELU_BESSER ({z})"
        else:
            u["H3"] = f"HU ({z})"
    teile = []
    for T in sorted({int(k.split("|T")[1].split("|")[0]) for k in r["zellen"]}):
        ze = ", ".join(f"{a} {r['zellen'][_zname(a, T, d)]['mittel']:.3f}".replace(".", ",") for a in ARME_MESS_TT if _zname(a, T, d) in r["zellen"])
        teile.append(f"T{T}: {ze}")
    u["B"] = "beschreibend: Test-bpc " + "; ".join(teile) + "; Rest in auswertung"
    return u


# ---------------------------------------------------------------- Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I047_PROFIL", "standard")
    P = dict(PROFILE_TT[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte (Checkpoint alle 15 Schritte); NICHT ueber `direktor.py lauf --schnell` (Vorregistrierung)
        P.update({"T": (20, 40), "tuning_T": 20, "nk_T": 10, "ckpt_alle": 15, "seeds": (TUNING_SEED_TT,), "nk_seeds": (TUNING_SEED_TT,)})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT_TT or set(P["nk_seeds"]) & _BELEGT_TT or TUNING_SEED_TT in P["seeds"] or TUNING_SEED_TT in _BELEGT_TT:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_TT}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE_TT}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE_TT, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": ss._json_sicher(P),
               "lr_gitter": list(ss.LR_GITTER), "tuning_seed": TUNING_SEED_TT, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        hp0 = tuning_tt(P, klein, TUNING_SEED_TT, pfad, kennung, prozesse, ckordner, arme=ARME_MESS_TT)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning"] = hp0
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = aufgaben_tt(hp, P, klein, ckordner)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_tt, schluessel=_schl_tt, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    d = P["d"][0]
    paare = [(a, d) for a in ARME_MESS_TT]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread_tt(paare)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {f"{a}|{d}": {"ffn": ss.param_ffn(n), "gesamt": ss.param_gesamt(n), "form": form_text(a, d)}
              for a, d in paare for n in [ss.Sprachmodell(a, 0, d=d, h=d // 64)]}
    fl = {f"{a}|{d}": flops_je_token(a, d) for a, d in paare}
    tr, _, te = ss.daten(klein)
    e = {"abteilung": "Struktur", "modul": "struktur_tt", "karte": KARTE_TT, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_TT,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": TUNING_SEED_TT, "tuning": hp,
         "einstellungen": {"T": list(P["T"]), "d": d, "tuning_T": P["tuning_T"], "nk_T": P["nk_T"], "nk_d": P["nk_d"], "ckpt_alle": P["ckpt_alle"],
                           "batch": ss.BATCH, "kontext": ss.KONTEXT, "lagen": ss.LAGEN, "form": {a: params[f"{a}|{d}"]["form"] for a in ARME_MESS_TT},
                           "lr_gitter": list(ss.LR_GITTER), "daten_bytes": [len(x) for x in ss.daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_tt(zeilen, params, ss.unigramm_entropie(tr, te), t_lo=min(P["T"]), t_hi=max(P["T"]), d=d, flops=fl, zeit=zm)
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
    e["urteil"] = urteil_tt(e["auswertung"])
    return e


# ---------------------------------------------------------------- Vortest
def _synth_tt(rng, adv1=0.10, adv2=0.10, adv3=0.05, sd=0.004, n=8, koll=None, nk=5.0, div=None, fehlt=None):
    """Synthetische Zeilen: Df_gelu = TT_z2 + adv1, Lk_z2 = TT_z2 + adv2 (positiv = TT besser), TT_gelu = TT_z2 + adv3 (positiv = z² besser)."""
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for T in T_WERTE_TT:
            basis = {"TT_z2": 2.4, "TT_gelu": 2.4 + adv3, "TT_z2_klein": 2.7, "Df_z2": 2.5, "Df_gelu": 2.4 + adv1, "Lk_z2": 2.4 + adv2, "Sz": 2.6}
            for a, v in basis.items():
                if fehlt == (a, T):
                    continue
                x, ist = v + gem + rng.normal(0, sd), False
                if koll == (a, T) and i == 0:
                    x += 1.0
                if div == (a, T) and i == 1:
                    x, ist = 8.0, True
                zl.append({"teil": "mess", "arm": a, "d": D_TT, "T": T, "seed": 2680 + i, "bpc": x, "instabil": ist, "sekunden_training": 10.0})
        if i < 2:
            for a in NK_ARME_TT:
                zl.append({"teil": "nk", "arm": a, "d": D_TT, "T": 1000, "seed": 2688 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u_tt(rng, **kw):
    p = {f"{a}|{D_TT}": {"ffn": 1, "gesamt": 2} for a in ARME_MESS_TT}
    return urteil_tt(auswertung_tt(_synth_tt(rng, **kw), p, 5.0))


def _fn_test_tt(a):
    return ss._schl_v3(a), {"v": a[4] * 2}


def _vortest_modelle() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-047): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    d = D_TT
    # --- TT-Rekonstruktion: TT-SVD mit vollem Rang == dichte Matrix (W1 und W2 in Originalgroesse), Forward == Linear
    for (ein, aus) in ((MODEN[d][0], MODEN[d][1]), (MODEN[d][1], MODEN[d][0])):
        lin = nn.Linear(math.prod(ein), math.prod(aus), bias=False)
        W = lin.weight.detach().T.contiguous()                      # (N_ein, N_aus), y = x @ W
        tt = TTMatrix.aus_dicht(W, ein, aus, chi=None)
        x = torch.randn(3, 5, math.prod(ein))
        pruefe(f"TT mit vollem Rang == dichte Matrix {ein}->{aus} (Forward, Raenge {tt.r})",
               torch.allclose(tt(x), lin(x), atol=2e-4, rtol=1e-3) and tt(x).shape == (3, 5, math.prod(aus)))
        pruefe(f"TT.dense() == W {ein}->{aus}", torch.allclose(tt.dense(), W, atol=2e-4, rtol=1e-3))
        t4 = TTMatrix.aus_dicht(W, ein, aus, chi=4)
        pruefe(f"TT-SVD chi=4: Raenge (1,4,4,1), Forward == dense()", t4.r == [1, 4, 4, 1] and torch.allclose(t4(x), x @ t4.dense(), atol=1e-4, rtol=1e-3))
    # Gesamtform der Kerne und Index-Konvention an einem kleinen Beispiel gegen explizite Einsum-Summe
    kl = TTMatrix((2, 3, 2), (3, 2, 2), 3, torch.Generator().manual_seed(1))
    G1, G2, G3 = [g.detach() for g in kl.kerne]
    # explizit: W[(i1,i2,i3),(j1,j2,j3)] = sum_ab G1[0,i1,j1,a] G2[a,i2,j2,b] G3[b,i3,j3,0]
    ref = torch.zeros(12, 12)
    for i1 in range(2):
        for i2 in range(3):
            for i3 in range(2):
                for j1 in range(3):
                    for j2 in range(2):
                        for j3 in range(2):
                            ref[(i1 * 3 + i2) * 2 + i3, (j1 * 2 + j2) * 2 + j3] = sum(
                                G1[0, i1, j1, a] * G2[a, i2, j2, b] * G3[b, i3, j3, 0] for a in range(3) for b in range(3))
    pruefe("Index-Konvention: dense() == explizite Summe sum_ab G1·G2·G3", torch.allclose(kl.dense(), ref, atol=1e-5))
    xs = torch.randn(4, 12)
    pruefe("Forward == x @ explizite Matrix", torch.allclose(kl(xs), xs @ ref, atol=1e-5))
    # --- Modelle
    nz = {a: ss.Sprachmodell(a, 0, d=d, h=d // 64) for a in ARME_MESS_TT}
    x = torch.randint(0, 256, (2, ss.KONTEXT))
    for a, m in nz.items():
        m.eval()
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 256
        pruefe(f"Form/Kausalitaet {a}", m(x).shape == (2, ss.KONTEXT, 256) and torch.allclose(m(x)[:, :60], m(x2)[:, :60], atol=1e-5)
               and not torch.allclose(m(x)[:, 60:], m(x2)[:, 60:]))
        pruefe(f"Koepfe d/64 {a}", m.bl[0].h == d // 64)
    pruefe("Aktivierungen", nz["TT_z2"].bl[0].ffn.akt is torch.square and nz["TT_gelu"].bl[0].ffn.akt is F.gelu and nz["TT_z2_klein"].bl[0].ffn.akt is torch.square
           and nz["Df_gelu"].bl[0].ffn.akt is F.gelu and nz["Df_z2"].bl[0].ffn.akt is torch.square and nz["Lk_z2"].bl[0].ffn.akt is torch.square)
    pruefe("Faktorisierung 512 = 8·8·8, 2048 = 16·16·8", MODEN[512] == ((8, 8, 8), (16, 16, 8)) and math.prod(MODEN[512][0]) == 512 and math.prod(MODEN[512][1]) == 2048)
    pruefe("Raenge (1,chi,chi,1) beider Matrizen",
           all(nz[a].bl[0].ffn.a.r == [1, c, c, 1] and nz[a].bl[0].ffn.b.r == [1, c, c, 1] for a, c in (("TT_z2", 4), ("TT_gelu", 4), ("TT_z2_klein", 2))))
    # --- Parameter exakt
    pf = {a: ss.param_ffn(n) for a, n in nz.items()}
    for a, c in (("TT_z2", 4), ("TT_gelu", 4), ("TT_z2_klein", 2)):
        pruefe(f"Parameter {a} = 4·Formel ({pf[a]})", pf[a] == ss.LAGEN * ffn_tt_param_lage(d, c))
    pruefe("Parameter TT_z2 = 4·(192·4+128·16 + 192·4+128·16 + 2048 + 512) von Hand (= 22.528 + 10.240 = 32.768)",
           pf["TT_z2"] == 4 * (2 * (192 * 4 + 128 * 16) + 2048 + 512) and pf["TT_z2"] == 32768)
    pruefe("Parameter TT_gelu = TT_z2", pf["TT_gelu"] == pf["TT_z2"])
    pruefe("TT hat weniger FFN-Parameter als Df (2,38 M) und Sz (43 k)", pf["TT_z2"] < pf["Sz"] < pf["Df_z2"] and pf["TT_z2"] < pf["Lk_z2"])
    # --- FLOPs as implemented: Formel == Zaehlung des Moduls == PyTorch-FlopCounter
    fl = {a: flops_je_token(a, d) for a in ARME_MESS_TT}
    for a, c in (("TT_z2", 4), ("TT_gelu", 4), ("TT_z2_klein", 2)):
        pruefe(f"FLOP-Formel {a} == Modulzaehlung", fl[a]["ffn"] == ss.LAGEN * ffn_tt_flops_lage(d, c))
        pruefe(f"FLOP-Modulzaehlung {a} == torch.utils.flop_counter", flops_gemessen(a, d) == fl[a]["ffn"] // ss.LAGEN)
    for a in ("Df_z2", "Df_gelu", "Lk_z2"):
        pruefe(f"FLOP-Zaehler {a} == Modulzaehlung ({flops_gemessen(a, d)})", flops_gemessen(a, d) == fl[a]["ffn"] // ss.LAGEN)
    for a in ("TT_z2", "TT_gelu"):
        pruefe(f"FLOP-Gleichheit +-5 % {a} ({fl[a]['ffn']} gegen Sz {fl['Sz']['ffn']})", abs(fl[a]["ffn"] / fl["Sz"]["ffn"] - 1) <= BUDGET_TOLERANZ_TT)
        pruefe(f"FLOP-Gleichheit +-5 % {a} gegen Df_gelu", abs(fl[a]["ffn"] / fl["Df_gelu"]["ffn"] - 1) <= BUDGET_TOLERANZ_TT)
        pruefe(f"FLOP-Gleichheit +-5 % {a} gegen Lk_z2", abs(fl[a]["ffn"] / fl["Lk_z2"]["ffn"] - 1) <= BUDGET_TOLERANZ_TT)
    pruefe("Df/Lk exakt FLOP-gleich Sz (v4)", fl["Df_z2"]["ffn"] == fl["Sz"]["ffn"] == fl["Df_gelu"]["ffn"] == fl["Lk_z2"]["ffn"] == 4_751_360)
    pruefe("TT_z2_klein ca. 1/4 bis 1/3 der FLOPs (0,2-0,35)", 0.2 <= fl["TT_z2_klein"]["ffn"] / fl["Sz"]["ffn"] <= 0.35)
    pruefe("TT_z2: 1.146.880 FLOPs je Lage (chi=4), chi=5 wuerde +5 % ueberschreiten", fl["TT_z2"]["ffn"] == 4 * 1_146_880 and ffn_tt_flops_lage(d, 5) > 1.05 * 1_187_840)
    # --- Initialisierung und Gradienten
    for a in ("TT_z2", "TT_z2_klein"):
        f0 = nz[a].bl[0].ffn
        Wd = f0.a.dense()
        pruefe(f"Init-Varianz der Gesamtmatrix {a} ~ 1/(3·512) (Faktor 2)", 0.5 < float(Wd.var()) * 3 * 512 < 2.0)
    m = ss.Sprachmodell("TT_z2", 0, d=d, h=8, lagen=1)
    l = F.cross_entropy(m(x[:, :32]).reshape(-1, 256), x[:, 1:33].reshape(-1))
    l.backward()
    pruefe("Gradient auf allen TT-Kernen und Biasen != 0", all(float(p.grad.abs().sum()) > 0 for p in m.bl[0].ffn.parameters()))
    # --- Trainierbarkeit aller Arme
    unit = torch.arange(32).repeat(8)[None].repeat(4, 1)[:, :65]
    for a in ARME_MESS_TT:
        mm = ss.Sprachmodell(a, 0, d=d, h=8, lagen=2)
        opt = torch.optim.Adam(mm.parameters(), lr=2e-3)
        l0 = None
        for _ in range(25):
            opt.zero_grad()
            l = F.cross_entropy(mm(unit[:, :-1]).reshape(-1, 256), unit[:, 1:].reshape(-1))
            l0 = l0 or float(l.detach())
            l.backward()
            opt.step()
        pruefe(f"Trainierbar {a}", float(l.detach()) < 0.7 * l0)
    return bool(ok)


def _vortest_lauf() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-047): {name}", flush=True)
            ok = False

    d = D_TT
    pruefe("Seeds frisch (8 Mess, 2 NK, Tuning 2693)", len(SEEDS_TT) == 8 and len(NK_SEEDS_TT) == 2 and not ((set(SEEDS_TT) | set(NK_SEEDS_TT)) & _BELEGT_TT)
           and TUNING_SEED_TT not in _BELEGT_TT and TUNING_SEED_TT not in SEEDS_TT and not (set(SEEDS_TT) & set(NK_SEEDS_TT)))
    hp = {f"{a}|{d}": {"lr": 1e-3} for a in ARME_MESS_TT}
    P = PROFILE_TT["standard"]
    aufg = aufgaben_tt(hp, P, False, "x")
    schl = [_schl_tt(a) for a in aufg]
    pruefe("Aufgabenzahl 7·2·8 + 4 NK = 116, Schluessel eindeutig", len(aufg) == 116 and len(set(schl)) == 116 and sum(a[0] == "nk" for a in aufg) == 4)
    pruefe("Lange zuerst (T=16000), NK d=512 T=1000", aufg[0][3] == 16000 and all(a[3] == 16000 for a in aufg[:56]) and all(a[2] == 512 and a[3] == 1000 for a in aufg if a[0] == "nk"))
    k = kosten_schaetzung_tt()
    pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (mit Aufschlag 1,3), laengste Aufgabe < 24 h", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    k15 = kosten_schaetzung_tt(aufschlag=1.5)
    pruefe("Laufzeitplan auch mit Aufschlag 1,5 < 24 h", k15["wand_gesamt_h"] < 24)
    pruefe("Pareto: dominierter Arm wird erkannt", ss.pareto_nicht_dominiert({"a": (1, 1, 1), "b": (2, 2, 2), "c": (0.5, 5, 1)}) == ["a", "c"])
    t = ss.trainiere_v3("TT_z2", 1e9, 0, 5, d=256, klein=True, val=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        ck = str(Path(tmp) / "a.pt")
        ref = ss.trainiere_v3("TT_z2", 2e-3, 3, 30, d=256, klein=True, val=True)
        try:
            ss.trainiere_v3("TT_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=15)
            pruefe("Abbruch wurde ausgeloest", False)
        except ss._Abbruch:
            pass
        pruefe("Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        fort = ss.trainiere_v3("TT_z2", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Fortsetzung ab Schritt 10", fort["fortgesetzt_ab"] == 10)
        pruefe("Fortsetzung = ungestoerter Lauf (bpc identisch)", abs(fort["bpc"] - ref["bpc"]) < 1e-9)
        pruefe("Checkpoint nach Ende geloescht", not Path(ck).exists())
        try:
            ss.trainiere_v3("TT_gelu", 2e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10, _stopp_bei=12)
        except ss._Abbruch:
            pass
        anders = ss.trainiere_v3("TT_gelu", 1e-3, 3, 30, d=256, klein=True, val=True, ckpt=ck, ckpt_alle=10)
        pruefe("Checkpoint mit anderer Lernrate wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [("mess", a, 512, 10, s, 1e-3, True, 5, tmp) for a in ("TT_z2", "Df_z2", "Lk_z2") for s in (1, 2, 3)]
        z1 = _rechne(a4[:4], pfad, kennung, fn=_fn_test_tt, prozesse=1, schluessel=_schl_tt)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test_tt, prozesse=1, schluessel=_schl_tt)
        pruefe("Wiederaufnahme je Aufgabe (Arm, d, T, Seed)", len(z1) == 4 and len(z2) == 9 and all(z2[k_] == z1[k_] for k_ in z1))
        Pk = {**P, "d": (256,), "tuning_T": 3, "ckpt_alle": 2}
        pt = Path(tmp) / "t.json"
        kt = {"karte": "tt"}
        arme_k = ("TT_z2", "TT_gelu")
        n0 = _ZAEHLER_TT["lr"]
        t1 = tuning_tt(Pk, True, TUNING_SEED_TT, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        n1 = _ZAEHLER_TT["lr"]
        t2 = tuning_tt(Pk, True, TUNING_SEED_TT, pt, kt, 1, Path(tmp) / "ck", arme=arme_k)
        pruefe("Tuning: Raten je Arm gewaehlt", set(t1) == {f"{a}|256" for a in arme_k} and all(v["lr"] > 0 for v in t1.values()))
        pruefe("Tuning-Wiederaufnahme rechnet nichts neu und liefert dasselbe", n1 > n0 and _ZAEHLER_TT["lr"] == n1 and t1 == t2)
        # Prozesspool (spawn): ein Kindprozess muss die TT-Arme kennen (Anbindung wird beim Import dieses Moduls gesetzt)
        pp = Path(tmp) / "p.json"
        ap = [("mess", "TT_z2", 64, 4, 1, 1e-3, True, 2, tmp)]
        try:
            zp = _rechne(ap, pp, {"k": 2}, fn=_arbeiter_tt, prozesse=2, schluessel=_schl_tt)
            pruefe("Prozesspool (spawn) kennt die TT-Arme", len(zp) == 1 and math.isfinite(list(zp.values())[0]["bpc"]))
        except Exception as ex:
            pruefe(f"Prozesspool (spawn) kennt die TT-Arme: {ex!r}", False)
    # --- Urteilszweige erzwungen
    rng = np.random.default_rng(13)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k_, w in soll.items():
            gesehen.add((k_, w.split(" (")[0]))
            if not u[k_].startswith(w):
                print(f"  Vortest FEHLER (I-047) {name}: {k_} soll {w}, ist {u[k_]}", flush=True)
                ok = False

    erwarte("TT besser", _u_tt(rng), **{"NK": "BESTANDEN", "H1": "TT_BESSER", "H2": "TT_BESSER", "H3": "QUADRAT_TRAEGT"})
    erwarte("Referenzen besser", _u_tt(rng, adv1=-0.08, adv2=-0.08, adv3=-0.05), **{"H1": "H0", "H2": "H0", "H3": "GELU_BESSER"})
    erwarte("H1 gemischt", _u_tt(rng, adv1=0.10, adv2=-0.08), **{"H1": "TT_BESSER", "H2": "H0"})
    erwarte("Kein Unterschied", _u_tt(rng, adv1=0.0, adv2=0.0, adv3=0.0, sd=0.03), **{"H1": "HU", "H2": "HU", "H3": "HU"})
    erwarte("NK faellt", _u_tt(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle", "H3": "HU (Negativkontrolle"})
    erwarte("Divergenz Df_gelu T=16000", _u_tt(rng, div=("Df_gelu", 16000)), **{"H1": "HU (Divergenz", "H2": "TT_BESSER", "H3": "QUADRAT_TRAEGT"})
    erwarte("Divergenz TT_gelu T=16000", _u_tt(rng, div=("TT_gelu", 16000)), **{"H1": "TT_BESSER", "H3": "HU (Divergenz"})
    erwarte("Kollaps Lk_z2 T=16000", _u_tt(rng, koll=("Lk_z2", 16000)), **{"H1": "TT_BESSER", "H2": "HU (Kollaps"})
    erwarte("Kollaps TT_z2 T=16000", _u_tt(rng, koll=("TT_z2", 16000)), **{"H1": "HU (Kollaps", "H2": "HU (Kollaps", "H3": "HU (Kollaps"})
    erwarte("Divergenz bei T=4000 stoert nicht", _u_tt(rng, div=("TT_z2", 4000)), **{"H1": "TT_BESSER", "H2": "TT_BESSER", "H3": "QUADRAT_TRAEGT"})
    erwarte("Zelle fehlt", _u_tt(rng, fehlt=("Lk_z2", 16000)), **{"H1": "TT_BESSER", "H2": "HU (Zellen fehlen"})
    erwarte("TT_gelu fehlt", _u_tt(rng, fehlt=("TT_gelu", 16000)), **{"H1": "TT_BESSER", "H3": "HU (Zellen fehlen"})
    p = {f"{a}|{D_TT}": {"ffn": 1, "gesamt": 2} for a in ARME_MESS_TT}
    a_ = auswertung_tt(_synth_tt(rng, div=("TT_z2", 16000)), p, 5.0)
    pruefe("Divergenzrate 1/8, Seed bleibt in den Zahlen (n = 8)", abs(a_["divergenzrate"]["TT_z2|T16000|d512"] - 0.125) < 1e-9
           and a_["zellen"]["TT_z2|T16000|d512"]["n"] == 8 and a_["diff"]["H1"]["n"] == 8)
    pruefe("Holm nur ueber H1, H2 (H3 nicht in Holm)", set(a_["p_holm"]) == {"H1", "H2"} and "H3" in a_["diff"])
    pruefe("Beschreibend: Abstand zu Df_gelu, Vorsprung gegen Sz und Pareto vorhanden",
           "Abstand_TT_z2_zu_Df_gelu" in a_["beschreibend"] and "Vorsprung_TT_z2_gegen_Sz" in a_["beschreibend"] and "pareto_T_hi" in a_["beschreibend"])
    erlaubt = ("TT_BESSER", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend", "QUADRAT_TRAEGT", "GELU_BESSER")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u_tt(rng).values()))
    soll = {"H1": {"TT_BESSER", "H0", "HU"}, "H2": {"TT_BESSER", "H0", "HU"}, "H3": {"QUADRAT_TRAEGT", "GELU_BESSER", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k_, w in gesehen if k_ == h}
        pruefe(f"Zweige {fehlt} fuer {h} erzwungen", not fehlt)
    return bool(ok)


def vortest() -> bool:
    ok = _vortest_modelle()
    ok = _vortest_lauf() and ok
    print(f"  Vortest I-047: {'bestanden' if ok else 'DURCHGEFALLEN'}", flush=True)
    return bool(ok)
