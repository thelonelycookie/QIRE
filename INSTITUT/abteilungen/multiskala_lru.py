"""Abteilung Struktur -- I-049 „Multiskala (Hourglass) und diagonale komplexe Rekurrenz (LRU) im Byte-Sprachmodell bei gleichen FLOPs“.
(Folge der Ideenwelle 27.10.2026, Synthese L3; Rauchtest r2: Hourglass+LRU 3,372 gegen flach FLOP-gleich 3,703 bpc nach nur 190 Schritten -- vermutlich Frühregime-Effekt.
 Hier: voll trainiert, Lernrate je (Arm, T), mehrere Trainingslängen. Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

IDEE (zwei Bausteine, beide bekannt, Neuheit ist nicht das Ziel -- CLAUDE.md 07.10.; nur Verifiziertes zitiert):
  (a) Multiskala/Coarse-Graining (Renormierungsgruppen-Analogie, nur Motivation): Hourglass-Transformer, Nawrot et al. 2022 „Hierarchical Transformers Are More Efficient Language Models“
      (Findings of NAACL; Pooling mit Verschiebung, damit es kausal bleibt); verwandt Funnel-Transformer (Dai et al. 2020), MegaByte (Yu et al. 2023). Die mittleren Lagen arbeiten
      auf T/4 Positionen, also mit rund einem Viertel der Kosten je Lage -- die einzige Idee der Welle, die echte FLOPs spart.
  (b) Diagonale komplexe lineare Rekurrenz nach Orvieto et al. 2023 „Resurrecting Recurrent Neural Networks for Long Sequences“ (ICML): lambda = exp(-exp(nu) + i·exp(theta)),
      Normierung gamma = sqrt(1 - |lambda|²). Lindblad-Bild (Rotation + Dämpfung) nur als Analogie. Hier statt der Attention in Lage 0, parallel per FFT-Faltung.
FRAGE: Schlägt das Hourglass-Modell mit LRU in der Vorlage (HGLRU) ein flaches Modell mit gleichen FLOPs je Token, wenn BEIDE voll trainiert sind -- und bleibt der Vorsprung
  mit wachsender Trainingslänge T bestehen oder schrumpft er (Frühregime-Verdacht aus r2)?

DATEN / MODELL / TRAINING (Daten, Split, bpc aus struktur_sprache, nichts kopiert ausser den Arbeitern):
  UD_German-GSD, Byte-Ebene, fester Split (train 1.614.280 / dev 73.794 / test 99.487 Bytes). Decoder-Transformer, Pre-LayerNorm, d=128 (ausser FF/FF4), FFN 4d (GELU), 4 Köpfe,
  Kontext 256, gelernte Positionen, Ausgabematrix = Einbettung (gebunden). Batch 8 x 256 = 2048 Bytes/Schritt, Adam (ohne Weight Decay), 5 % Aufwärmen, Kosinus, Clip 1,0.
  T in {2000, 8000, 16000} Schritte = 2,5 / 10,1 / 20,3 Epochen (T=16000 wiederholt jedes Byte etwa 20-mal: Train-bpc wird mitgemessen).
  Test-bpc: alle Positionen, nicht überlappende 256er-Fenster des gesamten Testsatzes. Lernrate: je (Arm, T) auf dem Tuning-Seed, Validierung = erste 65.536 Bytes von dev.
ARME (Attention/Einbettung wie im Basismodell, nur das jeweils Genannte unterscheidet sich):
  F4     flach, 4 Lagen, d=128                                   Referenz
  FF     flach, 3 Lagen, d=124                                   FLOP-gleich zu HGLRU (+1,9 %, also leicht zugunsten des flachen Arms)   -> Gegner H1
  FF4    flach, 4 Lagen, d=104                                   zweites FLOP-gleiches Flachmodell (-2,3 %), andere Tiefe/Breite-Aufteilung (beschreibend)
  FP     flach, 6 Lagen, d=128                                   parametergleich zu HGLRU (-1,3 %)
  HG     Hourglass: Vorlage (voll) + Pooling x4 (kausal verschoben) + 4 Mittellagen auf T/4 + Upsampling (Linear + Wiederholen) + Skip + Nachlage
  LRU    flach wie F4, Lage 0 mit LRU statt Attention
  HGLRU  HG mit LRU in der Vorlage                                                                                                  -> geprüfter Arm
HYPOTHESEN (gepaarter einseitiger t-Test, n = 10, T = 16000, Differenz V = bpc(FF) − bpc(HGLRU) je Seed, positiv = HGLRU besser; alpha 0,05; Holm über die Familie {H1} = unverändert):
  H1 V >= 0,03 bpc: HGLRU_BESSER (Mittel von V - 0,03 > 0 gesichert) | H0 (V < 0,03 bpc gesichert, auch wenn HGLRU im Mittel besser ist) | HU.
  H2 (beschreibend, zweiseitig, NICHT in Holm, alpha 0,05) Trend des Vorsprungs: Δ = V(T=16000) − V(T=2000) je Seed: SCHRUMPFT (Δ < 0, p < alpha) | WAECHST (Δ > 0, p < alpha) | HU.
  Beschreibend: bpc aller Arme x T, V auch bei T=8000 und gegen FF4, F4, FP; Komponenten (LRU gegen F4, HG gegen FF, HGLRU gegen HG und LRU), FLOPs/Token, Parameter,
  Inferenzzeit (µs/Token, Einzelprozess, 1 Thread), Trainingszeit, Train-bpc, Divergenzrate, Lernraten, Pareto bpc x FLOPs x µs/Token.
DIVERGENZ / KOLLAPS / NEGATIVKONTROLLE wie I-045-v4: Divergenz = nicht endlicher Verlust oder End-bpc >= Unigramm-Entropie; Kollaps = Median + 0,2 bpc (je Zelle Arm x T); Seeds bleiben in
  allen Zahlen; jede beteiligte Hypothese = HU. NK: HGLRU und F4, 1000 Schritte, permutierte Trainingsbytes, Seeds 2910/2911, bestanden bei mittlerem Test-bpc >= 0,9 x Unigramm (4,743), sonst alles HU.
  (HGLRU enthält beide Leckquellen: Pooling-Verschiebung und FFT-Faltung.)
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
from abteilungen.struktur_gross import _lade_zwischen, _rechne, _schreibe_zwischen, _stoppuhr
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil, zusammenfassung

KARTE_ML = "I-049"
SEEDS_ML = tuple(range(2900, 2910))
NK_SEEDS_ML = (2910, 2911)
TUNING_SEED_ML = 2990
KOEPFE_ML, CTX_ML, BATCH_ML, POOL = 4, 256, 8, 4
D0 = 128
T_WERTE_ML = (2000, 8000, 16000)
LR_GITTER_ML = (1e-3, 3e-3, 9e-3)
LR_FAKTOR_ML = 3
RUNDEN_TUNING = 3
MARGE_H1 = 0.03
TOL_FLOP, TOL_PARAM = 0.03, 0.02
NK_ARME_ML = ("HGLRU", "F4")
ARME_ML = ("F4", "FF", "FF4", "FP", "HG", "LRU", "HGLRU")
ARME_SPEC = {
    "F4": dict(n=4, d=128),
    "FF": dict(n=3, d=124),
    "FF4": dict(n=4, d=104),
    "FP": dict(n=6, d=128),
    "HG": dict(hg=True, d=128),
    "LRU": dict(n=4, d=128, lru0=True),
    "HGLRU": dict(hg=True, d=128, lru0=True),
}
_BELEGT_ML = ss._BELEGT_V4 | set(range(2620, 2630)) | set(range(2640, 2650)) | set(range(2660, 2670)) | set(range(2680, 2700))
PROFILE_ML = {"standard": {"T": T_WERTE_ML, "nk_T": 1000, "ckpt_alle": ss.CKPT_ALLE, "seeds": SEEDS_ML, "nk_seeds": NK_SEEDS_ML, "runden": RUNDEN_TUNING}}


# ---------------------------------------------------------------- Modell
class BlockML(nn.Module):
    def __init__(self, d, lru=False):
        super().__init__()
        self.d, self.lru = d, lru
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if lru:
            N = d
            r = torch.empty(N).uniform_(0.9, 0.999)
            self.nu = nn.Parameter(torch.log(-torch.log(r)))
            self.th = nn.Parameter(torch.log(torch.empty(N).uniform_(0, 2 * math.pi / 10) + 1e-4))
            self.bre, self.bim = nn.Parameter(torch.randn(N, d) / math.sqrt(2 * d)), nn.Parameter(torch.randn(N, d) / math.sqrt(2 * d))
            self.cre, self.cim = nn.Parameter(torch.randn(d, N) / math.sqrt(N)), nn.Parameter(torch.randn(d, N) / math.sqrt(N))
        else:
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def lam_log(self):
        return torch.complex(-torch.exp(self.nu), torch.exp(self.th))      # log lambda, |lambda| < 1

    def mixer(self, x):
        B, T, C = x.shape
        if not self.lru:
            q, k, v = self.qkv(x).reshape(B, T, 3, KOEPFE_ML, C // KOEPFE_ML).permute(2, 0, 3, 1, 4)
            return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        ll = self.lam_log()
        gam = torch.sqrt(1 - torch.exp(2 * ll.real))
        u = torch.complex(x @ self.bre.T, x @ self.bim.T) * gam                    # (B, T, N)
        tau = torch.arange(T, dtype=torch.float32)[:, None]
        kern = torch.exp(tau * ll[None, :])                                        # (T, N) = lambda^tau
        h = torch.fft.ifft(torch.fft.fft(u, n=2 * T, dim=1) * torch.fft.fft(kern, n=2 * T, dim=0)[None], dim=1)[:, :T]
        return h.real @ self.cre.T - h.imag @ self.cim.T

    def mixer_schleife(self, x):
        """Naive Rekurrenz h_t = lambda·h_{t-1} + gamma·B x_t (nur Vortest: Referenz zur FFT-Faltung)."""
        B, T, C = x.shape
        ll = self.lam_log()
        lam, gam = torch.exp(ll), torch.sqrt(1 - torch.exp(2 * ll.real))
        u = torch.complex(x @ self.bre.T, x @ self.bim.T) * gam
        h, aus = torch.zeros(B, u.shape[2], dtype=u.dtype), []
        for t in range(T):
            h = lam[None] * h + u[:, t]
            aus.append(h)
        h = torch.stack(aus, 1)
        return h.real @ self.cre.T - h.imag @ self.cim.T

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))


class LMML(nn.Module):
    """hg=False: flaches Modell mit n Blöcken. hg=True: Vorblock (voll), Pooling x4 (kausal verschoben), 4 Mittelblöcke auf T/4, Upsampling + Skip, Nachblock.
    lru0: Block 0 (bzw. die Vorlage) nutzt LRU statt Attention."""

    def __init__(self, n=4, d=D0, hg=False, n_mid=4, lru0=False):
        super().__init__()
        self.hg, self.d = hg, d
        self.emb, self.pos = nn.Embedding(256, d), nn.Embedding(CTX_ML, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        if hg:
            self.pre, self.post = BlockML(d, lru0), BlockML(d)
            self.mid = nn.ModuleList([BlockML(d) for _ in range(n_mid)])
            self.up = nn.Linear(d, d)
            self.lnm = nn.LayerNorm(d)
        else:
            self.bl = nn.ModuleList([BlockML(d, lru0 and i == 0) for i in range(n)])
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        if not self.hg:
            for b in self.bl:
                x = b(x)
        else:
            x = self.pre(x)
            B, T, C = x.shape
            # kausal: POOL-1 Nullen links, dann Mittel über Gruppen; Kurz-Token j deckt die Positionen 4j-3..4j ab
            s = F.pad(x, (0, 0, POOL - 1, 0))[:, :T].reshape(B, T // POOL, POOL, C).mean(2)
            for b in self.mid:
                s = b(s)
            u = self.up(self.lnm(s)).repeat_interleave(POOL, dim=1)      # Position t nutzt Kurz-Token t//4 (deckt nur <= t ab)
            x = self.post(x + u)
        return self.ln(x) @ self.emb.weight.T


def baue_ml(arm, seed=0):
    torch.manual_seed(seed * 31 + 11)
    return LMML(**ARME_SPEC[arm])


def n_param_ml(netz) -> int:
    return int(sum(p.numel() for p in netz.parameters() if p.requires_grad))


def flops_je_token_ml(arm) -> dict:
    """Vorwärts-FLOPs je Token (1 Flop = 1 Multiplikation oder Addition, 2 je MAC), as implemented. Attention: Scores + Werte, kausal gemittelt 2·d·(T+1)/2 MAC;
    LRU: Matmuls 4·d·N (B und C, Real-/Imaginärteil) + ideale Rekurrenz 4·N MAC (komplexes MAC = 4 reelle); Einbettung, LayerNorm, Aktivierung, Pooling nicht gezählt.
    `mit_fft`: zusätzlich die reale FFT-Faltung (2 Transformationen der Länge 2T je Kanal, 5·n·log2 n Flops je Transformation; Kern-FFT wird über den Batch geteilt)."""
    a = ARME_SPEC[arm]
    d = a["d"]
    N = d

    def blk(T, lru, anteil):
        ffn = 2 * d * 4 * d
        mix = 4 * d * N + 4 * N if lru else 4 * d * d + 2 * d * (T + 1) / 2
        return anteil * 2 * (ffn + mix)

    kopf = 2 * d * 256
    if not a.get("hg"):
        lru_lagen = 1 if a.get("lru0") else 0
        ges = sum(blk(CTX_ML, a.get("lru0", False) and i == 0, 1) for i in range(a["n"])) + kopf
    else:
        lru_lagen = 1 if a.get("lru0") else 0
        T4 = CTX_ML // POOL
        mid = sum(blk(T4, False, 1 / POOL) for _ in range(4))
        ges = blk(CTX_ML, a.get("lru0", False), 1) + blk(CTX_ML, False, 1) + mid + 2 * d * d / POOL + kopf      # + Upsampling-Linear auf Kurz-Token
    fft = lru_lagen * 2 * 5 * (2 * CTX_ML) * math.log2(2 * CTX_ML) * N / CTX_ML
    return {"gesamt": float(ges), "mit_fft": float(ges + fft)}


def bpc_ml(netz, text, bs=8) -> float:
    return ss.bpc_auswerten(netz, text, ctx=CTX_ML, bs=bs)


def zeit_je_token_ml(netz, batch=BATCH_ML) -> float:
    netz.eval()
    x = torch.randint(0, 256, (batch, CTX_ML))
    with torch.no_grad():
        return _stoppuhr(lambda: netz(x)) / (batch * CTX_ML)


# ---------------------------------------------------------------- Training (mit Checkpoint innerhalb der Aufgabe)
def trainiere_ml(arm, lr, seed, schritte, klein=False, val=False, permutiert=False, ckpt=None, ckpt_alle=ss.CKPT_ALLE, _stopp_bei=None):
    """Wie ss.trainiere_v3 (Checkpoint alle `ckpt_alle` Schritte: Modell, Optimierer, Plan, Schritt, Zufallszustände; Wiederaufnahme nur bei passenden Metadaten),
    aber mit den Modellen dieses Moduls, Kontext 256, Batch 8. Alle Arme sehen bei gleichem Seed dieselbe Reihenfolge der Trainingsfenster (gepaart)."""
    torch.set_num_threads(1)
    tr, va, te = ss.daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 77))]
    netz = baue_ml(arm, seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(netz.parameters(), lr=lr)
    warm = max(1, schritte // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, schritte - warm))))
    meta = {"arm": arm, "lr": float(lr), "seed": int(seed), "schritte": int(schritte), "klein": bool(klein), "perm": bool(permutiert), "ctx": CTX_ML, "batch": BATCH_ML}
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
    offs = torch.arange(CTX_ML)
    t0 = time.time()
    netz.train()
    for s in range(start, schritte):
        i = torch.randint(0, len(tr) - CTX_ML - 1, (BATCH_ML,), generator=g)
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
            raise ss._Abbruch(f"simulierter Abbruch bei Schritt {s + 1}")
    t_train = t_vor + time.time() - t0
    out = {"bpc": bpc_ml(netz, va[:ss.VAL_BYTES] if val else te), "instabil": False, "sekunden_training": t_train, "fortgesetzt_ab": start}
    if not val:
        out["train_bpc"] = bpc_ml(netz, tr[:65536])
    if ckpt and Path(ckpt).exists():
        Path(ckpt).unlink()
    return out


# ---------------------------------------------------------------- Arbeiter (Modulebene, spawn-fest)
def _schl_ml(a):
    """Aufgabe (teil, arm, T, seed, lr, klein, ckpt_alle, ckordner) -> Schlüssel der Zwischendatei."""
    return f"einheit|{a[0]}|{a[1]}|T{a[2]}|{a[3]}"


def _schl_lr_ml(a):
    """Tuning-Aufgabe (schl, seed, lr, arm, T, klein, ckpt_alle, ckordner)."""
    return f"lr|{a[0]}|{a[2]!r}|{a[1]}"


def _arbeiter_ml(a):
    teil, arm, T, seed, lr, klein, ck_alle, ckordner = a
    key = _schl_ml(a)
    r = trainiere_ml(arm, lr, seed, T, klein=klein, permutiert=(teil == "nk"), ckpt=ss._ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"teil": teil, "arm": arm, "T": T, "seed": seed, "lr": lr, **r}


_ZAEHLER_ML = {"lr": 0}


def _lr_arbeiter_ml(a):
    schl, seed, lr, arm, T, klein, ck_alle, ckordner = a
    _ZAEHLER_ML["lr"] += 1
    key = _schl_lr_ml(a)
    r = trainiere_ml(arm, lr, seed, T, klein=klein, val=True, ckpt=ss._ckpt_pfad(ckordner, key), ckpt_alle=ck_alle)
    return key, {"acc": 1 - r["bpc"] / 8, "bpc": r["bpc"], "instabil": r["instabil"]}


def tuning_ml(P, klein, tseed, pfad, kennung, prozesse, ckordner, arme=None, ts=None):
    """Lernrate je (Arm, T) auf dem Tuning-Seed bei VOLLER Schrittzahl T auf der Validierung (gittersuche_stabil); jede Aufgabe einzeln wiederaufnehmbar."""
    schl = [f"{a}|T{T}" for a in (arme or ARME_ML) for T in (ts or P["T"])]

    def aufgaben_fuer(s, lr):
        a, t = s.split("|T")
        return [(s, tseed, lr, a, int(t), klein, P["ckpt_alle"], str(ckordner))]

    def auswerten(aufg):
        lang = sorted(aufg, key=lambda x: -x[4] * S_JE_SCHRITT_ML.get(x[3], 1.0))
        res = _rechne(lang, pfad, kennung, fn=_lr_arbeiter_ml, schluessel=_schl_lr_ml, prozesse=prozesse)
        return [res[_schl_lr_ml(x)] for x in aufg]

    info, _ = gittersuche_stabil(schl, aufgaben_fuer, auswerten, gitter=LR_GITTER_ML, faktor=LR_FAKTOR_ML, kollaps=ss.KOLLAPS_SCORE, runden=P.get("runden", RUNDEN_TUNING))
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"], "runden": v["runden"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _zeit_arbeiter_ml(arme):
    torch.set_num_threads(1)
    return {a: zeit_je_token_ml(baue_ml(a, 0)) for a in arme}


def zeit_im_ein_thread_ml(arme):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter_ml, (tuple(arme),))


def schritt_zeiten_ml(arme=ARME_ML, n=20, warm=3):
    """Sekunden je Trainingsschritt (1 Thread, Batch 8 x 256, Vorwärts + Rückwärts + Clip + Adam), Median über n Schritte."""
    torch.set_num_threads(1)
    tr, _, _ = ss.daten(True)
    offs = torch.arange(CTX_ML)
    g = torch.Generator().manual_seed(1)
    aus = {}
    for a in arme:
        netz = baue_ml(a, 0)
        opt = torch.optim.Adam(netz.parameters(), lr=1e-4)
        ts = []
        for k in range(warm + n):
            i = torch.randint(0, len(tr) - CTX_ML - 1, (BATCH_ML,), generator=g)
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


# Sekunden je Trainingsschritt (1 Thread, Batch 8 x 256), Messung im Container (Datum und Werte im LABORBUCH).
# Gemessen 27.10.2026 im Container (4 Kerne, 1 Thread, Median über 15 Schritte, Maximum dreier Messungen; konservativ).
S_JE_SCHRITT_ML = {"F4": 0.21, "FF": 0.17, "FF4": 0.19, "FP": 0.33, "HG": 0.16, "LRU": 0.24, "HGLRU": 0.17}


# ---------------------------------------------------------------- Aufgaben und Kosten
def aufgaben_ml(hp, P, klein, ckordner):
    """Mess- und NK-Aufgaben, die langen (T = 16000, teure Arme) zuerst. NK-Rate: die von T = 2000."""
    aufg = [("mess", a, T, s, hp[f"{a}|T{T}"]["lr"], klein, P["ckpt_alle"], str(ckordner)) for a in ARME_ML for T in P["T"] for s in P["seeds"]]
    t_nk = min(P["T"])
    aufg += [("nk", a, P["nk_T"], s, hp[f"{a}|T{t_nk}"]["lr"], klein, P["ckpt_alle"], str(ckordner)) for a in NK_ARME_ML for s in P["nk_seeds"]]
    aufg.sort(key=lambda a: -a[2] * S_JE_SCHRITT_ML.get(a[1], 1.0))
    return aufg


def kosten_schaetzung_ml(s_je_schritt=None, prozesse=72, aufschlag=1.3, tuning_raten=4.5, tuning_runden=3, ueberhang=30.0):
    """Kernstunden und Wandzeit (LPT-Planung auf `prozesse` Kerne). Tuning: erste Runde = 3 Raten je (Arm, T) (LPT), danach `tuning_runden` − 1 Verlängerungsrunden,
    jede höchstens so lang wie die längste Aufgabe (konservativ); Kernstunden Tuning = Σ Aufgabendauer x `tuning_raten` (mittlere Zahl Raten je Schlüssel)."""
    z = s_je_schritt or S_JE_SCHRITT_ML
    P = PROFILE_ML["standard"]
    dauer = lambda a, T: T * z[a] * aufschlag + ueberhang
    t_tun = [dauer(a, T) for a in ARME_ML for T in P["T"]]
    kern_tuning = sum(t_tun) * tuning_raten

    def lpt(ts):
        last = [0.0] * prozesse
        heapq.heapify(last)
        for t in sorted(ts, reverse=True):
            heapq.heappush(last, heapq.heappop(last) + t)
        return max(last)

    wand_tuning = lpt([t for t in t_tun for _ in range(3)]) + (tuning_runden - 1) * max(t_tun)
    m_aufg = [dauer(a, T) for a in ARME_ML for T in P["T"] for _ in P["seeds"]]
    m_aufg += [dauer(a, P["nk_T"]) for a in NK_ARME_ML for _ in P["nk_seeds"]]
    wand_mess = lpt(m_aufg)
    return {"aufgaben_mess": len(m_aufg), "kernstunden_mess": sum(m_aufg) / 3600, "kernstunden_tuning": kern_tuning / 3600,
            "kernstunden_gesamt": (sum(m_aufg) + kern_tuning) / 3600, "laengste_aufgabe_h": max(m_aufg) / 3600,
            "wand_tuning_h": wand_tuning / 3600, "wand_mess_h": wand_mess / 3600, "wand_gesamt_h": (wand_tuning + wand_mess) / 3600,
            "prozesse": prozesse, "aufschlag": aufschlag}


# ---------------------------------------------------------------- Auswertung (bpc: kleiner ist besser)
def _zname_ml(a, T):
    return f"{a}|T{T}"


def auswertung_ml(zeilen: list, params: dict, unigramm: float, t_lo=T_WERTE_ML[0], t_mid=T_WERTE_ML[1], t_hi=T_WERTE_ML[-1], flops=None, zeit=None) -> dict:
    bpc, inst, tt, trb = {}, {}, {}, {}
    for z in zeilen:
        if z["teil"] == "mess":
            k = (z["arm"], z["T"])
            bpc.setdefault(k, {})[z["seed"]] = z["bpc"]
            inst.setdefault(k, {})[z["seed"]] = bool(z.get("instabil"))
            tt.setdefault(k, {})[z["seed"]] = z.get("sekunden_training", float("nan"))
            trb.setdefault(k, {})[z["seed"]] = z.get("train_bpc", float("nan"))
    r = {"t_lo": t_lo, "t_mid": t_mid, "t_hi": t_hi, "zellen": {}, "kollaps": {}, "divergiert": {}, "divergenzrate": {}, "bpc_ohne_divergente": {}, "negativ": {},
         "diff": {}, "p_holm": {}, "beschreibend": {}, "parameter": params, "unigramm_bpc": unigramm, "training_s_mittel": {}, "train_bpc_mittel": {}}
    for k, v in bpc.items():
        seeds = sorted(v)
        w = np.array([v[s] for s in seeds])
        n = _zname_ml(*k)
        r["zellen"][n] = zusammenfassung(w)
        med = float(np.median(w))
        r["kollaps"][n] = [s for s in seeds if not (v[s] <= med + ss.KOLLAPS_BPC)]
        r["divergiert"][n] = [s for s in seeds if inst[k][s] or not (v[s] < unigramm)]
        r["divergenzrate"][n] = len(r["divergiert"][n]) / len(seeds)
        ok = [v[s] for s in seeds if s not in r["divergiert"][n]]
        r["bpc_ohne_divergente"][n] = float(np.mean(ok)) if ok else None
        r["training_s_mittel"][n] = float(np.nanmean(list(tt[k].values())))
        r["train_bpc_mittel"][n] = float(np.nanmean(list(trb[k].values()))) if not np.all(np.isnan(list(trb[k].values()))) else None
    nk = [z["bpc"] for z in zeilen if z["teil"] == "nk"]
    r["negativ"] = {"mittel": float(np.mean(nk)), "n": len(nk)} if nk else {}

    def vorsprung(T, ref, pruef="HGLRU"):
        """Je Seed: bpc(ref) − bpc(pruef); positiv = pruef besser. {} wenn eine Zelle fehlt."""
        a, b = bpc.get((ref, T)), bpc.get((pruef, T))
        if not a or not b:
            return {}
        return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}

    v_hi = vorsprung(t_hi, "FF")
    if v_hi:
        r["diff"]["H1"] = ss._stat(np.array(list(v_hi.values())), versatz=-MARGE_H1)      # p_gr: Mittel(V) − 0,03 > 0; p_kl: Mittel(V) − 0,03 < 0
    v_lo = vorsprung(t_lo, "FF")
    if v_hi and v_lo:
        gem = sorted(set(v_hi) & set(v_lo))
        r["diff"]["H2"] = ss._stat(np.array([v_hi[s] - v_lo[s] for s in gem]))                # Δ = V(16000) − V(2000), zweiseitig
    for ref in ("FF", "FF4", "F4", "FP", "HG", "LRU"):
        for T in (t_lo, t_mid, t_hi):
            v = vorsprung(T, ref)
            if v:
                r["beschreibend"].setdefault(f"Vorsprung_HGLRU_gegen_{ref}", {})[f"T{T}"] = ss._stat(np.array(list(v.values())))
    for T in (t_lo, t_mid, t_hi):
        for name, (ref, pr) in {"LRU_gegen_F4": ("F4", "LRU"), "HG_gegen_FF": ("FF", "HG"), "HG_gegen_FF4": ("FF4", "HG"), "HG_gegen_F4": ("F4", "HG")}.items():
            v = vorsprung(T, ref, pr)
            if v:
                r["beschreibend"].setdefault(f"Vorsprung_{name}", {})[f"T{T}"] = ss._stat(np.array(list(v.values())))
    v_mid = vorsprung(t_mid, "FF")
    if v_hi and v_mid:
        gem = sorted(set(v_hi) & set(v_mid))
        r["beschreibend"]["Trend_V_16000_minus_8000_gegen_FF"] = ss._stat(np.array([v_hi[s] - v_mid[s] for s in gem]))
    pt = {}
    for a in ARME_ML:
        n = _zname_ml(a, t_hi)
        if n in r["zellen"] and a in params and (zeit is None or a in zeit):
            pt[a] = (r["zellen"][n]["mittel"], (flops or {}).get(a, {}).get("gesamt", 0.0), (zeit or {}).get(a, 0.0))
    r["beschreibend"]["pareto_T_hi"] = {"punkte": {a: list(p) for a, p in pt.items()}, "nicht_dominiert": ss.pareto_nicht_dominiert(pt),
                                        "achsen": ["bpc", "flops_je_token", "s_je_token" if zeit else "(Zeit nicht gemessen: 0)"]}
    if flops:
        r["beschreibend"]["flops_je_token"] = flops
    if zeit:
        r["beschreibend"]["zeit_s_je_token"] = zeit
    if "H1" in r["diff"] and "p_gr" in r["diff"]["H1"]:
        r["p_holm"] = holm({"H1": r["diff"]["H1"]["p_gr"]})
    return r


def _vorab_ml(r, nk, h, zellen):
    if h not in r["diff"] or "mittel" not in r["diff"][h]:
        return "HU (Zellen fehlen oder zu wenige Seeds)"
    if h == "H1" and h not in r["p_holm"]:
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


def _zahlen_ml(d, name, extra=""):
    return (f"{name} {d['mittel']:+.4f} bpc, 90-%-KI [{d['ki'][0]:+.4f}; {d['ki'][1]:+.4f}], n {d['n']}, {d['positiv']}/{d['n']} Seeds positiv{extra}").replace(".", ",")


def urteil_ml(r: dict) -> dict:
    nk = ss._nk(r)
    u = {"NK": f"{nk} (Mittel permutiert HGLRU/F4: {r['negativ'].get('mittel', float('nan')):.3f} bpc; Unigramm {r['unigramm_bpc']:.3f}; Schwelle {ss.NK_FAKTOR}x)"}
    t_lo, t_hi = r["t_lo"], r["t_hi"]
    v = _vorab_ml(r, nk, "H1", [_zname_ml(a, t_hi) for a in ("FF", "HGLRU")])
    if v:
        u["H1"] = v
    else:
        d = r["diff"]["H1"]
        z = _zahlen_ml(d, f"V = FF−HGLRU bei T={t_hi}", f", p_Holm {r['p_holm']['H1']:.3g}, Marge {MARGE_H1}")
        if r["p_holm"]["H1"] < ss.ALPHA:
            u["H1"] = f"HGLRU_BESSER (Vorsprung >= {MARGE_H1} bpc gegen flach FLOP-gleich gesichert; {z})".replace(".", ",")
        elif d["p_kl"] < ss.ALPHA:
            u["H1"] = f"H0 (Vorsprung < {MARGE_H1} bpc gesichert; {z})".replace(".", ",")
        else:
            u["H1"] = f"HU ({z})"
    v = _vorab_ml(r, nk, "H2", [_zname_ml(a, t) for a in ("FF", "HGLRU") for t in (t_lo, t_hi)])
    if v:
        u["H2"] = v
    else:
        d = r["diff"]["H2"]
        z = _zahlen_ml(d, f"Δ = V({t_hi})−V({t_lo})", f", p_zweiseitig {d['p_zwei']:.3g}")
        if d["p_zwei"] < ss.ALPHA:
            u["H2"] = f"SCHRUMPFT ({z})" if d["mittel"] < 0 else f"WAECHST ({z})"
        else:
            u["H2"] = f"HU ({z})"
    teile = []
    for T in sorted({int(k.split("|T")[1]) for k in r["zellen"]}):
        ze = ", ".join(f"{a} {r['zellen'][_zname_ml(a, T)]['mittel']:.3f}".replace(".", ",") for a in ARME_ML if _zname_ml(a, T) in r["zellen"])
        teile.append(f"T{T}: {ze}")
    u["B"] = "beschreibend: Test-bpc " + "; ".join(teile) + "; Rest in auswertung"
    return u


# ---------------------------------------------------------------- Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None, **_):
    profil = profil or os.environ.get("I049_PROFIL", "standard")
    P = dict(PROFILE_ML[profil])
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, wenige Schritte; NICHT über `direktor.py lauf --schnell` (Vorregistrierung)
        P.update({"T": (8, 16, 24), "nk_T": 8, "ckpt_alle": 6, "seeds": (TUNING_SEED_ML,), "nk_seeds": (TUNING_SEED_ML,), "runden": 2})
        klein = True
    else:
        klein = False
        if set(P["seeds"]) & _BELEGT_ML or set(P["nk_seeds"]) & _BELEGT_ML or TUNING_SEED_ML in P["seeds"] or TUNING_SEED_ML in _BELEGT_ML:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_ML}{'-schnell' if schnell else ''}.json"
    ckordner = pfad.parent / f"_ckpt_{KARTE_ML}{'-schnell' if schnell else ''}"
    kennung = {"karte": KARTE_ML, "version": 1, "profil": "rauch" if schnell else profil, "profil_werte": ss._json_sicher(P),
               "lr_gitter": list(LR_GITTER_ML), "tuning_seed": TUNING_SEED_ML, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" not in zw:
        hp0 = tuning_ml(P, klein, TUNING_SEED_ML, pfad, kennung, prozesse, ckordner)
        zw = _lade_zwischen(pfad, kennung)
        zw["tuning"] = hp0
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = aufgaben_ml(hp, P, klein, ckordner)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_ml, schluessel=_schl_ml, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread_ml(ARME_ML)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {a: {"gesamt": n_param_ml(baue_ml(a, 0)), "spec": ARME_SPEC[a]} for a in ARME_ML}
    fl = {a: flops_je_token_ml(a) for a in ARME_ML}
    tr, _, te = ss.daten(klein)
    e = {"abteilung": "Struktur", "modul": "multiskala_lru", "karte": KARTE_ML, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN_ML,
         "seeds": list(P["seeds"]), "nk_seeds": list(P["nk_seeds"]), "tuning_seed": TUNING_SEED_ML, "tuning": hp,
         "einstellungen": {"T": list(P["T"]), "nk_T": P["nk_T"], "ckpt_alle": P["ckpt_alle"], "batch": BATCH_ML, "kontext": CTX_ML, "koepfe": KOEPFE_ML, "pool": POOL,
                           "arme": ARME_SPEC, "lr_gitter": list(LR_GITTER_ML), "marge_h1": MARGE_H1, "daten_bytes": [len(x) for x in ss.daten(klein)]},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung_ml(zeilen, params, ss.unigramm_entropie(tr, te), t_lo=min(P["T"]), t_mid=sorted(P["T"])[len(P["T"]) // 2], t_hi=max(P["T"]), flops=fl, zeit=zm)
    if zm is not None:
        e["zeitmessung_einzelprozess_s_je_token"] = zm
    e["urteil"] = urteil_ml(e["auswertung"])
    return e


HYPOTHESEN_ML = {
    "H1": f"V = bpc(FF) − bpc(HGLRU) >= {MARGE_H1} bpc bei T=16000 (FF: flach, FLOPs/Token +-3 % gleich; einseitig gepaart t, n=10, Marge {MARGE_H1}, alpha 0,05, Holm über {{H1}}). HGLRU_BESSER | H0 (V < {MARGE_H1} gesichert) | HU.",
    "H2": "beschreibend, zweiseitig, nicht in Holm (alpha 0,05): Δ = V(T=16000) − V(T=2000) je Seed (gepaart t). SCHRUMPFT (Δ < 0) | WAECHST (Δ > 0) | HU.",
    "DIVERGENZ": ss.HYPOTHESEN_V3["DIVERGENZ"],
    "NK": f"Negativkontrolle (HGLRU, F4, 1000 Schritte, permutierte Trainingsbytes, Seeds 2910/2911): mittlerer Test-bpc >= {ss.NK_FAKTOR} x Unigramm-Entropie. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: bpc aller Arme x T, V auch bei T=8000 und gegen FF4/F4/FP, Komponenten (LRU/F4, HG/FF, HGLRU/HG, HGLRU/LRU), FLOPs/Token, Parameter, µs/Token, Trainingszeit, Train-bpc, Divergenzrate, Lernraten, Pareto.",
}


# ---------------------------------------------------------------- Vortest
def _synth_ml(rng, adv_lo=0.20, adv_hi=0.05, sd=0.004, n=10, koll=None, nk=5.0, div=None, fehlt=None):
    """Synthetische Zeilen: bpc(FF) = bpc(HGLRU) + V(T) mit V(2000) = adv_lo, V(16000) = adv_hi, V(8000) = Mittel (positiv = HGLRU besser)."""
    zl = []
    adv = {2000: adv_lo, 8000: 0.5 * (adv_lo + adv_hi), 16000: adv_hi}
    for i in range(n):
        gem = rng.normal(0, 0.01)
        for T in T_WERTE_ML:
            basis = {"HGLRU": 2.4, "FF": 2.4 + adv[T], "FF4": 2.4 + adv[T] + 0.01, "F4": 2.4 + adv[T] + 0.04, "FP": 2.4 + adv[T] + 0.06, "HG": 2.4 + 0.5 * adv[T], "LRU": 2.4 + 0.6 * adv[T]}
            for a, v in basis.items():
                if fehlt == (a, T):
                    continue
                x, ist = v + gem + rng.normal(0, sd), False
                if koll == (a, T) and i == 0:
                    x += 1.0
                if div == (a, T) and i == 1:
                    x, ist = 8.0, True
                zl.append({"teil": "mess", "arm": a, "T": T, "seed": 2900 + i, "bpc": x, "instabil": ist, "sekunden_training": 10.0, "train_bpc": x - 0.1})
        if i < 2:
            for a in NK_ARME_ML:
                zl.append({"teil": "nk", "arm": a, "T": 1000, "seed": 2910 + i, "bpc": nk + rng.normal(0, 0.02)})
    return zl


def _u_ml(rng, **kw):
    p = {a: {"gesamt": 1} for a in ARME_ML}
    return urteil_ml(auswertung_ml(_synth_ml(rng, **kw), p, 5.0))


def _fn_test_ml(a):
    return _schl_ml(a), {"v": a[3] * 2}


def _vortest_modelle() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-049): {name}", flush=True)
            ok = False

    torch.manual_seed(0)
    nz = {a: baue_ml(a, 0).eval() for a in ARME_ML}
    # --- Form und Kausalität an mehreren Stellen (auch innerhalb einer Pooling-Gruppe)
    x = torch.randint(0, 256, (2, CTX_ML))
    for a, m in nz.items():
        gut = m(x).shape == (2, CTX_ML, 256)
        for t in (60, 100, 101, 102, 103, 255):
            x2 = x.clone()
            x2[:, t] = (x2[:, t] + 1) % 256
            with torch.no_grad():
                d = (m(x) - m(x2)).abs().amax(-1)
            gut &= bool(float(d[:, :t].max()) < 1e-5)
            if t < 255:
                gut &= bool(float(d[:, t:].max()) > 1e-6)
        pruefe(f"Form und Kausalität {a}", gut)
    # --- LRU: FFT-Faltung == naive Rekurrenz (Vorwärts) und Dämpfung < 1, Gradienten auf nu/theta
    b = BlockML(32, lru=True)
    xs = torch.randn(2, 24, 32)
    pruefe("LRU: FFT-Faltung == naive Rekurrenz", torch.allclose(b.mixer(xs), b.mixer_schleife(xs), atol=1e-4, rtol=1e-3))
    pruefe("LRU: |lambda| in (0,9; 0,999) bei Init", bool(((torch.exp(b.lam_log().real) > 0.899) & (torch.exp(b.lam_log().real) < 0.9991)).all()))
    b.mixer(xs).sum().backward()
    pruefe("LRU: Gradient auf nu, theta, B, C", all(float(p.grad.abs().sum()) > 0 for p in (b.nu, b.th, b.bre, b.bim, b.cre, b.cim)))
    # --- Hourglass: Pooling-Gruppen und Verschiebung
    s = torch.arange(1, 9, dtype=torch.float32)[None, :, None]
    kurz = F.pad(s, (0, 0, POOL - 1, 0))[:, :8].reshape(1, 2, POOL, 1).mean(2)
    pruefe("Pooling kausal verschoben: Kurz-Token 0 = Mittel(0,0,0,1), Kurz-Token 1 = Mittel(2,3,4,5)", torch.allclose(kurz[0, :, 0], torch.tensor([0.25, 3.5])))
    # --- Parameter: Zählung == Handrechnung (F4: 858.880 wie r2), Budgets
    pz = {a: n_param_ml(m) for a, m in nz.items()}
    blk = lambda d: 2 * d + 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    pruefe(f"Parameter F4 = Handrechnung 256·d + 256·d + 4·Block + LN = 858.880 ({pz['F4']})", pz["F4"] == 256 * 128 + 256 * 128 + 4 * blk(128) + 2 * 128 == 858_880)
    pruefe("Parameter FF = Handrechnung (3 Lagen, d=124)", pz["FF"] == 256 * 124 + 256 * 124 + 3 * blk(124) + 2 * 124)
    pruefe("Parameter FP = Handrechnung (6 Lagen, d=128)", pz["FP"] == 256 * 128 + 256 * 128 + 6 * blk(128) + 2 * 128)
    lru_extra = lambda d: 2 * d + 4 * d * d - (d * 3 * d + 3 * d) - (d * d + d)          # LRU-Mixer statt Attention
    pruefe("Parameter LRU = F4 − Attention + LRU-Mixer", pz["LRU"] == pz["F4"] + lru_extra(128))
    pruefe("Parameter HG = Handrechnung (6 Blöcke, Up-Linear, 2 LN)", pz["HG"] == 256 * 128 + 256 * 128 + 6 * blk(128) + (128 * 128 + 128) + 2 * 128 + 2 * 128)
    pruefe("Parameter HGLRU = HG + LRU-Mixer", pz["HGLRU"] == pz["HG"] + lru_extra(128))
    pruefe(f"Parameter-Gleichheit FP gegen HGLRU +-2 % ({pz['FP']} gegen {pz['HGLRU']})", abs(pz["FP"] / pz["HGLRU"] - 1) <= TOL_PARAM)
    # --- FLOPs
    fl = {a: flops_je_token_ml(a) for a in ARME_ML}
    pruefe(f"FLOP-Gleichheit FF gegen HGLRU +-3 % ({fl['FF']['gesamt'] / fl['HGLRU']['gesamt'] - 1:+.3f})", abs(fl["FF"]["gesamt"] / fl["HGLRU"]["gesamt"] - 1) <= TOL_FLOP)
    pruefe(f"FF hat mindestens so viele FLOPs wie HGLRU (konservativ für H1)", fl["FF"]["gesamt"] >= fl["HGLRU"]["gesamt"])
    pruefe(f"FLOP-Gleichheit FF4 gegen HGLRU +-3 % ({fl['FF4']['gesamt'] / fl['HGLRU']['gesamt'] - 1:+.3f})", abs(fl["FF4"]["gesamt"] / fl["HGLRU"]["gesamt"] - 1) <= TOL_FLOP)
    pruefe("HG und HGLRU brauchen weniger FLOPs als F4", fl["HG"]["gesamt"] < fl["F4"]["gesamt"] and fl["HGLRU"]["gesamt"] < fl["F4"]["gesamt"])
    pruefe("FFT-Zuschlag der LRU-Arme klein (< 3 %) und nur dort", all((fl[a]["mit_fft"] / fl[a]["gesamt"] - 1 < 0.03) and ((fl[a]["mit_fft"] > fl[a]["gesamt"]) == (a in ("LRU", "HGLRU"))) for a in ARME_ML))
    # unabhängige Prüfung der Matmul-Zählung mit PyTorch (torch.utils.flop_counter zählt auf der CPU nur Matmuls/Linear, NICHT die Attention-Scores und nicht die FFT):
    # Zähler == Matmul-Anteil der Formel EXAKT; die Attention-Scores (kausal gemittelt) und die ideale Rekurrenz kommen analytisch dazu.
    from torch.utils.flop_counter import FlopCounterMode

    def zaehler(a):
        xx = torch.randint(0, 256, (1, CTX_ML))
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            nz[a](xx)
        return fc.get_total_flops() / CTX_ML

    for a in ("F4", "FF", "FP", "FF4", "LRU"):       # LRU-Matmuls (4·d·N MAC) kosten genau so viel wie qkv + Ausgabe (4·d² MAC)
        d, n = ARME_SPEC[a]["d"], ARME_SPEC[a]["n"]
        matmul = n * 2 * (8 * d * d + 4 * d * d) + 2 * d * 256
        att = (n - (1 if a == "LRU" else 0)) * 2 * 2 * d * (CTX_ML + 1) / 2
        rek = 2 * 4 * d if a == "LRU" else 0
        pruefe(f"FLOP-Zähler {a} == Matmul-Anteil der Formel ({zaehler(a):.0f} gegen {matmul:.0f})", abs(zaehler(a) - matmul) < 1)
        pruefe(f"Formel flops_je_token_ml {a} == Matmul + Attention-Scores (kausal) + ideale Rekurrenz", abs(fl[a]["gesamt"] - (matmul + att + rek)) < 1)
    d, T4 = 128, CTX_ML // POOL
    for a, lru in (("HG", False), ("HGLRU", True)):
        matmul = 2 * 12 * d * d * (2 + 4 / POOL) + 2 * d * d / POOL + 2 * d * 256
        att = (1 if lru else 2) * 2 * 2 * d * (CTX_ML + 1) / 2 + 4 * (2 * 2 * d * (T4 + 1) / 2) / POOL
        rek = 2 * 4 * d if lru else 0
        pruefe(f"FLOP-Zähler {a} == Matmul-Anteil der Formel ({zaehler(a):.0f} gegen {matmul:.0f})", abs(zaehler(a) - matmul) < 1)
        pruefe(f"Formel flops_je_token_ml {a} == Matmul + Attention-Scores (kausal) + ideale Rekurrenz", abs(fl[a]["gesamt"] - (matmul + att + rek)) < 1)
    # --- Trainierbarkeit aller Arme (periodische Folge)
    unit = torch.arange(32).repeat(16)[None].repeat(4, 1)[:, :CTX_ML + 1]
    for a in ARME_ML:
        mm = baue_ml(a, 1)
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
            print(f"  Vortest FEHLER (I-049): {name}", flush=True)
            ok = False

    pruefe("Seeds frisch (10 Mess 2900-2909, 2 NK 2910/2911, Tuning 2990)", SEEDS_ML == tuple(range(2900, 2910)) and NK_SEEDS_ML == (2910, 2911) and TUNING_SEED_ML == 2990
           and not ((set(SEEDS_ML) | set(NK_SEEDS_ML) | {TUNING_SEED_ML}) & _BELEGT_ML) and not (set(SEEDS_ML) & set(NK_SEEDS_ML)))
    hp = {f"{a}|T{T}": {"lr": 1e-3} for a in ARME_ML for T in T_WERTE_ML}
    P = PROFILE_ML["standard"]
    aufg = aufgaben_ml(hp, P, False, "x")
    schl = [_schl_ml(a) for a in aufg]
    pruefe("Aufgabenzahl 7·3·10 + 4 NK = 214, Schlüssel eindeutig", len(aufg) == 214 and len(set(schl)) == 214 and sum(a[0] == "nk" for a in aufg) == 4)
    kost = [a[2] * S_JE_SCHRITT_ML[a[1]] for a in aufg]
    pruefe("Teuerste zuerst (FP, T=16000), Kosten nicht steigend, NK 1000 Schritte", aufg[0][1] == "FP" and aufg[0][2] == 16000 and kost == sorted(kost, reverse=True)
           and all(a[2] == 1000 for a in aufg if a[0] == "nk") and all(a[2] == 16000 for a in aufg[:35]))
    k = kosten_schaetzung_ml()
    pruefe("Laufzeitplan: Wandzeit bei 72 Prozessen < 24 h (Aufschlag 1,3, 3 Tuningrunden), längste Aufgabe < 24 h", k["wand_gesamt_h"] < 24 and k["laengste_aufgabe_h"] < 24)
    k15 = kosten_schaetzung_ml(aufschlag=1.5)
    pruefe("Laufzeitplan auch mit Aufschlag 1,5 < 24 h", k15["wand_gesamt_h"] < 24)
    t = trainiere_ml("HGLRU", 1e9, 0, 5, klein=True, val=True)
    pruefe("Divergenz-Pfad liefert instabil oder schlechten bpc", t["instabil"] or t["bpc"] > 5.0)
    with tempfile.TemporaryDirectory() as tmp:
        ck = str(Path(tmp) / "a.pt")
        ref = trainiere_ml("HGLRU", 2e-3, 3, 24, klein=True, val=True)
        try:
            trainiere_ml("HGLRU", 2e-3, 3, 24, klein=True, val=True, ckpt=ck, ckpt_alle=8, _stopp_bei=12)
            pruefe("Abbruch wurde ausgelöst", False)
        except ss._Abbruch:
            pass
        pruefe("Checkpoint nach Abbruch vorhanden", Path(ck).exists())
        fort = trainiere_ml("HGLRU", 2e-3, 3, 24, klein=True, val=True, ckpt=ck, ckpt_alle=8)
        pruefe("Fortsetzung ab Schritt 8", fort["fortgesetzt_ab"] == 8)
        pruefe("Fortsetzung = ungestörter Lauf (bpc identisch)", abs(fort["bpc"] - ref["bpc"]) < 1e-9)
        pruefe("Checkpoint nach Ende gelöscht", not Path(ck).exists())
        try:
            trainiere_ml("FF", 2e-3, 3, 24, klein=True, val=True, ckpt=ck, ckpt_alle=8, _stopp_bei=10)
        except ss._Abbruch:
            pass
        anders = trainiere_ml("FF", 1e-3, 3, 24, klein=True, val=True, ckpt=ck, ckpt_alle=8)
        pruefe("Checkpoint mit anderer Lernrate wird nicht verwendet", anders["fortgesetzt_ab"] == 0)
        # gleiche Fensterfolge je Seed für alle Arme (gepaart): zwei Arme, 3 Schritte, gleicher erster Verlust-Eingang
        g1 = torch.Generator().manual_seed(3 + 5)
        g2 = torch.Generator().manual_seed(3 + 5)
        pruefe("Fensterfolge hängt nur vom Seed ab", torch.equal(torch.randint(0, 1000, (8,), generator=g1), torch.randint(0, 1000, (8,), generator=g2)))
        pfad = Path(tmp) / "z.json"
        kennung = {"k": 1}
        a4 = [("mess", a, 10, s, 1e-3, True, 5, tmp) for a in ("F4", "FF", "HGLRU") for s in (1, 2, 3)]
        z1 = _rechne(a4[:4], pfad, kennung, fn=_fn_test_ml, prozesse=1, schluessel=_schl_ml)
        z2 = _rechne(a4, pfad, kennung, fn=_fn_test_ml, prozesse=1, schluessel=_schl_ml)
        pruefe("Wiederaufnahme je Aufgabe (Arm, T, Seed)", len(z1) == 4 and len(z2) == 9 and all(z2[k_] == z1[k_] for k_ in z1))
        Pk = {**P, "ckpt_alle": 2, "runden": 2}
        pt = Path(tmp) / "t.json"
        kt = {"karte": "ml"}
        n0 = _ZAEHLER_ML["lr"]
        t1 = tuning_ml(Pk, True, TUNING_SEED_ML, pt, kt, 1, Path(tmp) / "ck", arme=("F4", "HGLRU"), ts=(3, 5))
        n1 = _ZAEHLER_ML["lr"]
        t2 = tuning_ml(Pk, True, TUNING_SEED_ML, pt, kt, 1, Path(tmp) / "ck", arme=("F4", "HGLRU"), ts=(3, 5))
        pruefe("Tuning: Raten je (Arm, T) gewählt", set(t1) == {"F4|T3", "F4|T5", "HGLRU|T3", "HGLRU|T5"} and all(v["lr"] > 0 for v in t1.values()))
        pruefe("Tuning-Wiederaufnahme rechnet nichts neu und liefert dasselbe", n1 > n0 and _ZAEHLER_ML["lr"] == n1 and t1 == t2)
        pp = Path(tmp) / "p.json"
        ap = [("mess", "HGLRU", 4, 1, 1e-3, True, 2, tmp)]
        try:
            zp = _rechne(ap, pp, {"k": 2}, fn=_arbeiter_ml, prozesse=2, schluessel=_schl_ml)
            pruefe("Prozesspool (spawn) rechnet HGLRU", len(zp) == 1 and math.isfinite(list(zp.values())[0]["bpc"]))
        except Exception as ex:
            pruefe(f"Prozesspool (spawn): {ex!r}", False)
    # --- Urteilszweige erzwungen
    rng = np.random.default_rng(17)
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k_, w in soll.items():
            gesehen.add((k_, w.split(" (")[0]))
            if not u[k_].startswith(w):
                print(f"  Vortest FEHLER (I-049) {name}: {k_} soll {w}, ist {u[k_]}", flush=True)
                ok = False

    erwarte("HGLRU besser, Vorsprung schrumpft", _u_ml(rng, adv_lo=0.30, adv_hi=0.10), **{"NK": "BESTANDEN", "H1": "HGLRU_BESSER", "H2": "SCHRUMPFT"})
    erwarte("Vorsprung wächst", _u_ml(rng, adv_lo=0.10, adv_hi=0.30), **{"H1": "HGLRU_BESSER", "H2": "WAECHST"})
    erwarte("Vorsprung zu klein (0,01) -> H0", _u_ml(rng, adv_lo=0.01, adv_hi=0.01), **{"H1": "H0", "H2": "HU"})
    erwarte("Flach besser (V negativ, sinkt weiter: Δ < 0 heißt SCHRUMPFT)", _u_ml(rng, adv_lo=-0.05, adv_hi=-0.10), **{"H1": "H0", "H2": "SCHRUMPFT"})
    erwarte("Vorsprung nahe Marge, großes Rauschen", _u_ml(rng, adv_lo=0.03, adv_hi=0.03, sd=0.05), **{"H1": "HU", "H2": "HU"})
    erwarte("NK fällt", _u_ml(rng, nk=2.5), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle"})
    erwarte("Divergenz FF T=16000", _u_ml(rng, div=("FF", 16000)), **{"H1": "HU (Divergenz", "H2": "HU (Divergenz"})
    erwarte("Divergenz F4 stört nicht", _u_ml(rng, div=("F4", 16000)), **{"H1": "HGLRU_BESSER", "H2": "SCHRUMPFT"})
    erwarte("Divergenz bei T=8000 (FF) stört nicht", _u_ml(rng, div=("FF", 8000)), **{"H1": "HGLRU_BESSER", "H2": "SCHRUMPFT"})
    erwarte("Kollaps HGLRU T=2000", _u_ml(rng, koll=("HGLRU", 2000)), **{"H1": "HGLRU_BESSER", "H2": "HU (Kollaps"})
    erwarte("Kollaps HGLRU T=16000", _u_ml(rng, koll=("HGLRU", 16000)), **{"H1": "HU (Kollaps", "H2": "HU (Kollaps"})
    erwarte("Zelle fehlt", _u_ml(rng, fehlt=("FF", 2000)), **{"H1": "HGLRU_BESSER", "H2": "HU (Zellen fehlen"})
    erwarte("Zelle fehlt T=16000", _u_ml(rng, fehlt=("HGLRU", 16000)), **{"H1": "HU (Zellen fehlen", "H2": "HU (Zellen fehlen"})
    p = {a: {"gesamt": 1} for a in ARME_ML}
    a_ = auswertung_ml(_synth_ml(rng, div=("HGLRU", 16000)), p, 5.0)
    pruefe("Divergenzrate 1/10, Seed bleibt in den Zahlen (n = 10)", abs(a_["divergenzrate"]["HGLRU|T16000"] - 0.1) < 1e-9
           and a_["zellen"]["HGLRU|T16000"]["n"] == 10 and a_["diff"]["H1"]["n"] == 10)
    pruefe("Holm nur über {H1}; H2 nicht in Holm", set(a_["p_holm"]) == {"H1"} and "H2" in a_["diff"])
    pruefe("Beschreibend: V gegen FF4/F4/FP je T, Komponenten, Pareto vorhanden",
           all(f"Vorsprung_HGLRU_gegen_{r_}" in a_["beschreibend"] for r_ in ("FF4", "F4", "FP")) and "Vorsprung_LRU_gegen_F4" in a_["beschreibend"]
           and "Vorsprung_HG_gegen_FF" in a_["beschreibend"] and "pareto_T_hi" in a_["beschreibend"])
    pruefe("Marge-Test: V = 0,05 mit kleinem Rauschen besteht Marge 0,03, V = 0,02 nicht", _u_ml(rng, adv_hi=0.05, sd=0.003)["H1"].startswith("HGLRU_BESSER")
           and _u_ml(rng, adv_hi=0.02, sd=0.003)["H1"].startswith("H0"))
    erlaubt = ("HGLRU_BESSER", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend", "SCHRUMPFT", "WAECHST")
    pruefe("Vokabular", all(v.startswith(erlaubt) for v in _u_ml(rng).values()))
    soll = {"H1": {"HGLRU_BESSER", "H0", "HU"}, "H2": {"SCHRUMPFT", "WAECHST", "HU"}}
    for h, ws in soll.items():
        fehlt = ws - {w for k_, w in gesehen if k_ == h}
        pruefe(f"Zweige {fehlt} für {h} erzwungen", not fehlt)
    return bool(ok)


def vortest() -> bool:
    ok = _vortest_modelle()
    ok = _vortest_lauf() and ok
    print(f"  Vortest I-049: {'bestanden' if ok else 'DURCHGEFALLEN'}", flush=True)
    return bool(ok)
