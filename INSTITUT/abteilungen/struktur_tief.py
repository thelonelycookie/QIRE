"""Abteilung Struktur -- I-044 „Strukturschicht auf CIFAR-10: tiefer und im CNN“.
(Folge von I-042: dort kostete die Strukturschicht auf CIFAR-10 bei N=16.384 rund 6 pp gegen die dichte Schicht (A 55,3 / B 49,1 / E 49,2).
 Die Wirkung ist klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

FRAGE: Schließt man die Lücke durch (a) mehrere Strukturschichten hintereinander oder (b) Struktur in einem kleinen CNN?
DATEN: CIFAR-10 wie I-042 (50.000/10.000, kanalweise standardisiert, keine Augmentierung), Wiederverwendung von struktur_gross.daten.
TEIL 1 (MLP auf Rohpixeln, Breite N=4096): A dicht, 1 Schicht (ReLU) | A2 dicht, 2 Schichten (Referenz) |
  S1/S2/S3 = 1/2/3 Fastfood-Schichten (H D2 H D1, Diagonalen TRAINIERT) je + Nichtlinearität + LayerNorm (ohne Parameter), dann linearer Kopf.
  Nichtlinearität je Schicht aus {z², |z|, ReLU}, GREEDY auf dem Tuning-Seed gewählt (S1: Schicht 1; S2: Schicht 1 aus S1 übernommen,
  Schicht 2 gewählt; S3: Schichten 1-2 übernommen, Schicht 3 gewählt; Auswahl bei fester Rate SEL_LR, Validierung = letzte 10 % Training).
TEIL 2 (kleines CNN: 3 Conv-Blöcke 3x3 mit BatchNorm+ReLU, Breiten 16/32/64, Pooling nach Block 1, 2 und (hinter dem Mix) 3 -> 1.024 Merkmale):
  Kd: Kopf = dichte Schicht 1024->4096 (ReLU) | Ks: Kopf = Fastfood-Strukturschicht 1024->4096 (trainierte Diagonalen, Nichtlinearität aus
  {z², |z|, ReLU} auf dem Tuning-Seed gewählt), je + LayerNorm + linearer Kopf 4096->10.
  Kcd / Kcs: Kanalmischung hinter Block 3 (64 Kanäle, 8x8): dicht = 1x1-Conv 64->64 | strukturiert = Hadamard·Diagonal über Kanäle,
  Fastfood-2 (H D2 H D1, 2·64 trainierte Parameter), je + ReLU; Kopf bei beiden direkt linear 1024->10.
TRAINING: Adam, Batch 256, Kosinus-Abfall, Epochen fest (PROFILE), Lernrate je Arm auf Tuning-Seed 2399 (gittersuche_stabil, beidseitig verlängerbar).
SEEDS: Mess 2300-2304 (5), Tuning 2399. Test nur am Ende (letzte Epoche).
HYPOTHESEN (Holm je Teil, gepaarter t-Test n=5 -- Begründung wie I-042):
  Teil 1: H1 S3 >= A - 1 pp (Nichtunterlegenheit) | H2 S3 - S1 > 0 (Tiefe hilft).
  Teil 2: H3 Ks >= Kd - 1 pp UND >= 10x weniger Kopfparameter | H4 Kcs >= Kcd - 1 pp.
MESSUNG: Genauigkeit, trainierte Parameter (Schicht und gesamt), Inferenzzeit (1 Thread, Batch 1 und 64, Median 7 Wdh., ganzes Netz).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abteilungen.clifford_suche import holm
from abteilungen.struktur_gross import (Fastfood, _arr, _f, _lade_zwischen, _rechne, _schreibe_zwischen, _stat, _stoppuhr, daten,
                                        kollaps_seeds, pad2, walsh_klein)
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil, parallel, zusammenfassung

KARTE = "I-044"
SEEDS = tuple(range(2300, 2305))
TUNING_SEED = 2399
_BELEGT = (set(range(2000, 2070)) | set(range(2090, 2120)) | set(range(2150, 2160)) | set(range(900, 1001)) | set(range(2200, 2205))
           | {2299})
N_BREIT = 4096
LR_GITTER = (1e-3, 3e-3, 1e-2)
LR_FAKTOR = 3
SEL_LR = 3e-3                    # feste Rate für die Auswahl der Nichtlinearität (nur Tuning-Seed)
BATCH = 256
ALPHA = 0.05
NI_GRENZE = 0.01
KOLLAPS = 0.05
KOPF_FAKTOR = 10                 # H3: >= 10x weniger Kopfparameter
NK_BAND = (0.05, 0.15)
ARME1 = ("A", "A2", "S1", "S2", "S3")
ARME2 = ("Kd", "Ks", "Kcd", "Kcs")
NK_ARME = ("S1", "Ks")           # Negativkontrolle: permutierte Trainingslabels
ZEIT_BATCH = (1, 64)
CAND = ("z2", "abs", "relu")
AKT = {"z2": torch.square, "abs": torch.abs, "relu": F.relu}

# ---- Laufzeitprotokoll (VOR dem Lauf festgelegt, aus Messung 06.10., siehe LABORBUCH) ----
PROFILE = {
    "standard": {"ep1": 4, "ep2": 4, "sel": 2, "nk": 2},
    "fallback": {"ep1": 3, "ep2": 3, "sel": 1, "nk": 1},
}

HYPOTHESEN = {
    "H1": "Teil 1, N=4096: S3 (3 Fastfood-Schichten, trainierte Diagonalen) >= A (dicht, 1 Schicht) - 1 pp (Nichtunterlegenheit, einseitig gepaart t, Holm je Teil). "
          "NICHT_UNTERLEGEN | H0 (S3 sicher unterlegen) | HU.",
    "H2": "Teil 1: S3 - S1 > 0 (Tiefe hilft; einseitig gepaart t, Holm je Teil). TIEFE_HILFT | H0 (S3 sicher schlechter als S1) | HU.",
    "H3": "Teil 2: Ks >= Kd - 1 pp (Nichtunterlegenheit) UND >= 10x weniger Kopfparameter (breite Schicht + Klassifikator). NICHT_UNTERLEGEN | H0 | HU.",
    "H4": "Teil 2: Kcs (Hadamard·Diagonal über Kanäle) >= Kcd (dichte 1x1-Conv) - 1 pp. NICHT_UNTERLEGEN | H0 | HU.",
    "NK": f"Negativkontrolle (S1 und Ks mit permutierten Trainingslabels, über Arme und Seeds gepoolt): Mittel in {NK_BAND}. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: alle Arm-Mittel, Parameter, Zeiten, Lernraten, gewählte Nichtlinearitäten, A2 gegen A und S3.",
}
FOLGEFRAGEN = {
    "H1 NICHT_UNTERLEGEN": "Tiefe schließt die Lücke: Strukturschicht tragen als Baustein im tiefen Netz; nächster Schritt Sprachmodell-Block.",
    "H1 H0": "Tiefe allein schließt die Lücke nicht: Struktur nur dort einsetzen, wo die Aufgabe leicht ist (Fashion, I-042).",
    "H3 NICHT_UNTERLEGEN": "Im CNN genügt eine Strukturschicht im Kopf: Kopfparameter um mehr als das 10-Fache sparen.",
}


# ------------------------------------------------------------------ Bausteine
class KanalFastfood(nn.Module):
    """Hadamard·Diagonal über Kanäle: x -> H D2 H D1 x je Pixel (C Zweierpotenz, H orthonormal), 2·C trainierte Parameter."""

    def __init__(self, C, gen):
        super().__init__()
        self.register_buffer("H", walsh_klein(int(math.log2(C))))
        self.d1 = nn.Parameter((torch.randint(0, 2, (C,), generator=gen) * 2 - 1).float())
        self.d2 = nn.Parameter((torch.randint(0, 2, (C,), generator=gen) * 2 - 1).float())

    def forward(self, x):                                   # (B,C,H,W)
        z = x.permute(0, 2, 3, 1)
        z = (z * self.d1) @ self.H
        z = (z * self.d2) @ self.H
        return z.permute(0, 3, 1, 2)


class Breit(nn.Module):
    """Breite Schicht(en): dicht oder Fastfood, je Linear/Fastfood -> Nichtlinearität -> LayerNorm ohne Parameter."""

    def __init__(self, din, N, dicht, acts, seed):
        super().__init__()
        self.acts = acts
        ls = []
        for i, a in enumerate(acts):
            ein = din if i == 0 else N
            ls.append(nn.Linear(ein, N) if dicht else Fastfood(ein, N, True, torch.Generator().manual_seed(seed * 31 + N + 7 * i)))
        self.ls = nn.ModuleList(ls)

    def forward(self, x):
        for l, a in zip(self.ls, self.acts):
            x = F.layer_norm(AKT[a](l(x)), (l.N if isinstance(l, Fastfood) else l.out_features,))
        return x


def _spez(arm, akt):
    """(Teil, dicht?, Aktivierungen der breiten Schichten)."""
    if arm == "A":
        return 1, True, ("relu",)
    if arm == "A2":
        return 1, True, ("relu", "relu")
    if arm in ("S1", "S2", "S3"):
        return 1, False, tuple(akt[arm])
    if arm == "Kd":
        return 2, True, ("relu",)
    if arm == "Ks":
        return 2, False, tuple(akt["Ks"])
    return 2, arm == "Kcd", ()


class Netz(nn.Module):
    def __init__(self, arm, seed, akt, din=3072, N=N_BREIT):
        super().__init__()
        teil, dicht, acts = _spez(arm, akt)
        self.arm, self.teil = arm, teil
        if teil == 1:
            self.breit = Breit(din, N, dicht, acts, seed)
            self.kopf = nn.Linear(N, 10)
        else:
            c = (16, 32, 64)
            self.conv = nn.Sequential(
                nn.Conv2d(3, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU())
            if arm in ("Kd", "Ks"):
                self.breit = Breit(c[2] * 16, N, dicht, acts, seed)
                self.kopf = nn.Linear(N, 10)
            else:
                gen = torch.Generator().manual_seed(seed * 31 + 5)
                self.breit = nn.Conv2d(c[2], c[2], 1) if dicht else KanalFastfood(c[2], gen)
                self.kopf = nn.Linear(c[2] * 16, 10)

    def forward(self, x):
        if self.teil == 1:
            return self.kopf(self.breit(x))
        z = self.conv(x.reshape(-1, 3, 32, 32))
        if self.arm in ("Kcd", "Kcs"):
            z = F.relu(self.breit(z))
            z = F.max_pool2d(z, 2).flatten(1)
            return self.kopf(F.layer_norm(z, z.shape[-1:]))
        z = F.max_pool2d(z, 2).flatten(1)
        return self.kopf(self.breit(z))


def param_breit(netz):
    return int(sum(p.numel() for p in netz.breit.parameters() if p.requires_grad))


def param_gesamt(netz):
    return int(sum(p.numel() for p in netz.parameters() if p.requires_grad))


def param_kopf(netz):
    """Kopfparameter für H3: breite Schicht + Klassifikator."""
    return param_breit(netz) + int(sum(p.numel() for p in netz.kopf.parameters()))


# ------------------------------------------------------------------ Training
def trainiere(arm, akt, lr, seed, epochen, klein=False, val=False, permutiert=False):
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = daten("cifar10", klein)
    if val:
        nv = len(ytr) // 10
        xtr, ytr, xte, yte = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 5 + 100 * ord(arm[0]) + ord(arm[-1])))]
    torch.manual_seed(seed)
    netz = Netz(arm, seed, akt)
    opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=lr)
    n = len(ytr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochen * math.ceil(n / BATCH))
    netz.train()
    for _ in range(epochen):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            l = F.cross_entropy(netz(xtr[j]), ytr[j])
            if not torch.isfinite(l):
                return {"acc": 0.1, "instabil": True}
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
    netz.eval()
    with torch.no_grad():
        richtig = sum(int((netz(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum()) for i in range(0, len(yte), 1000))
    return {"acc": richtig / len(yte), "instabil": False}


def _akt_arbeiter(a):
    arm, akt, lr, s, ep, klein = a
    return trainiere(arm, akt, lr, s, ep, klein=klein, val=True)


def waehle_akt(tseed, ep, klein):
    """Greedy-Auswahl der Nichtlinearität je Schicht (fester Lauf bei SEL_LR, Tuning-Seed, Validierung)."""
    akt, roh = {}, {}
    for i, arm in enumerate(("S1", "S2", "S3")):
        vor = akt[("S1", "S2", "S3")[i - 1]] if i > 0 else ()
        aufg = [(arm, {arm: vor + (c,)}, SEL_LR, tseed, ep, klein) for c in CAND]
        if i == 0:      # Ks gemeinsam mit S1 (je 3 Kandidaten)
            aufg += [("Ks", {"Ks": (c,)}, SEL_LR, tseed, ep, klein) for c in CAND]
        erg = parallel(_akt_arbeiter, aufg)
        e = {}
        for a, r in zip(aufg, erg):
            e.setdefault(a[0], []).append(r["acc"])
        for k, w in e.items():
            b = int(np.argmax(w))                       # Gleichstand: erster Kandidat in CAND
            akt[k] = (vor if k != "Ks" else ()) + (CAND[b],)
            roh[k] = {c: float(x) for c, x in zip(CAND, w)}
    return akt, roh


def _lr_arbeiter(a):
    arm, s, lr, akt, ep, klein = a
    return trainiere(arm, akt, lr, s, ep, klein=klein, val=True)


def tuning(arme, akt, ep, klein, tseed):
    info, _ = gittersuche_stabil(list(arme), lambda k, lr: [(k, tseed, lr, akt, ep[k], klein)],
                                 lambda aufg: parallel(_lr_arbeiter, aufg), gitter=LR_GITTER, faktor=LR_FAKTOR, kollaps=KOLLAPS)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _arbeiter(a):
    """Eine (Teil, Arm, Seed)-Einheit; Kontrolle: Teil 'nk' mit permutierten Labels."""
    teil, arm, s, lr, akt, ep, klein = a
    t0 = time.time()
    r = trainiere(arm, akt, lr, s, ep, klein=klein, permutiert=(teil == "nk"))
    return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "acc": r["acc"], "instabil": r["instabil"], "sekunden": time.time() - t0}


def _schl(a):
    return f"einheit|{a[0]}|{a[1]}|{a[2]}"


# ------------------------------------------------------------------ Zeitmessung (Inferenz, 1 Thread, ganzes Netz)
def zeit_messung(akt, arme=ARME1 + ARME2, batches=ZEIT_BATCH):
    torch.set_num_threads(1)
    aus = []
    for arm in arme:
        netz = Netz(arm, 0, akt).eval()
        z = {"arm": arm, "parameter_breit": param_breit(netz), "parameter_kopf": param_kopf(netz), "parameter_gesamt": param_gesamt(netz)}
        for B in batches:
            x = torch.randn(B, 3072)
            with torch.no_grad():
                z[f"batch{B}_s"] = _stoppuhr(lambda: netz(x))
        aus.append(z)
    return aus


def _zeit_arbeiter(a):
    return zeit_messung(*a)


def zeit_im_ein_thread(akt):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter, ((akt,),))


def zeit_text(zm) -> dict:
    """Beschreibend: Verhältnis Zeit(dicht)/Zeit(Struktur) je Batch für (A,S3) und (Kd,Ks), (Kcd,Kcs)."""
    z = {}
    for (d, s) in (("A", "S3"), ("A2", "S3"), ("Kd", "Ks"), ("Kcd", "Kcs")):
        a = next((x for x in zm if x["arm"] == d), None)
        b = next((x for x in zm if x["arm"] == s), None)
        if a and b:
            z[f"{d}/{s}"] = {f"batch{B}": round(a[f"batch{B}_s"] / b[f"batch{B}_s"], 3) for B in ZEIT_BATCH}
    return z


# ------------------------------------------------------------------ Auswertung
def auswertung(zeilen: list, params: dict) -> dict:
    """zeilen: Liste der Einheiten-Dicts. params[arm] = {'breit','kopf','gesamt'}."""
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] != "nk"})
    acc = {}
    for z in zeilen:
        acc.setdefault((z["teil"], z["arm"]), {})[z["seed"]] = z["acc"]
    r = {"arme": {}, "kollaps": {}, "negativ": {}, "diff": {}, "p_holm": {}, "parameter": params, "beschreibend": {}}
    zz = {}
    for teil, arme in (("t1", ARME1), ("t2", ARME2)):
        zeilen_t = [{"seed": s, **{a: acc[(teil, a)][s] for a in arme if (teil, a) in acc and s in acc[(teil, a)]}} for s in seeds]
        zz[teil] = zeilen_t
        for a in arme:
            if zeilen_t and a in zeilen_t[0]:
                r["arme"][a] = zusammenfassung(_arr(zeilen_t, a))
                r["kollaps"][a] = kollaps_seeds(zeilen_t, a)
    perm = [v for (t, a), d in acc.items() if t == "nk" for v in d.values()]
    r["negativ"] = {"mittel": float(np.mean(perm)), "n": len(perm)} if perm else {}
    t1, t2 = zz["t1"], zz["t2"]
    diff = lambda zl, x, y: _arr(zl, x) - _arr(zl, y)
    if t1 and "S3" in t1[0] and "A" in t1[0]:
        r["diff"]["H1"] = _stat(diff(t1, "S3", "A"), versatz=NI_GRENZE)
        if "S1" in t1[0]:
            r["diff"]["H2"] = _stat(diff(t1, "S3", "S1"))
        for x, y in (("S3", "A2"), ("A2", "A"), ("S2", "S1"), ("S3", "S2")):
            if x in t1[0] and y in t1[0]:
                r["beschreibend"][f"{x} - {y}"] = _stat(diff(t1, x, y))
    if t2 and "Ks" in t2[0] and "Kd" in t2[0]:
        r["diff"]["H3"] = _stat(diff(t2, "Ks", "Kd"), versatz=NI_GRENZE)
    if t2 and "Kcs" in t2[0] and "Kcd" in t2[0]:
        r["diff"]["H4"] = _stat(diff(t2, "Kcs", "Kcd"), versatz=NI_GRENZE)
    for fam in (("H1", "H2"), ("H3", "H4")):
        prim = {h: r["diff"][h]["p_gr"] if "p_gr" in r["diff"][h] else 1.0 for h in fam if h in r["diff"]}
        if prim:
            r["p_holm"].update(holm(prim))
    return r


# ------------------------------------------------------------------ Urteile (Kriterien im Code)
def _zahlen(r, h):
    d = r["diff"][h]
    return f"d {_f(d['mittel_pp'])} pp, 90-%-KI [{_f(d['ki_pp'][0])}; {_f(d['ki_pp'][1])}], n {d['n']}, p_Holm {r['p_holm'][h]:.3g}"


def _nk(r) -> str:
    n = r["negativ"]
    return "BESTANDEN" if n and NK_BAND[0] <= n["mittel"] <= NK_BAND[1] else "DURCHGEFALLEN"


def _koll(r, arme):
    return sorted({s for a in arme for s in r["kollaps"].get(a, [])})


def _vorab(r, nk, h, arme):
    if h not in r["diff"] or "mittel_pp" not in r["diff"][h]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    k = _koll(r, arme)
    if k:
        return f"HU (Instabilitaet/Kollaps in Seeds {k})"
    return None


def _ni(r, nk, h, arme, extra=None) -> str:
    v = _vorab(r, nk, h, arme)
    if v:
        return v
    z = _zahlen(r, h)
    if r["p_holm"][h] < ALPHA:
        if extra:
            m = extra(r)
            if m:
                return f"HU ({m}; {z})"
        return f"NICHT_UNTERLEGEN ({z})"
    if r["diff"][h]["p_kl"] < ALPHA / 2:
        return f"H0 (unterlegen um mehr als 1 pp; {z})"
    return f"HU ({z})"


def _kopf_pruefung(r):
    p = r["parameter"]
    if "Kd" not in p or "Ks" not in p:
        return "Kopfparameter fehlen"
    ratio = p["Kd"]["kopf"] / max(1, p["Ks"]["kopf"])
    return None if ratio >= KOPF_FAKTOR else f"Kopfparameter nur {ratio:.1f}x weniger (< {KOPF_FAKTOR}x)"


def _urteil_h2(r, nk) -> str:
    v = _vorab(r, nk, "H2", ("S1", "S3"))
    if v:
        return v
    z = _zahlen(r, "H2")
    if r["p_holm"]["H2"] < ALPHA:
        return f"TIEFE_HILFT ({z})"
    if r["diff"]["H2"]["p_kl"] < ALPHA / 2:
        return f"H0 (S3 sicher schlechter als S1; {z})"
    return f"HU ({z})"


def urteil(r: dict, zeit=None, akt=None, tuning_lr=None) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutierte Labels S1/Ks: {r['negativ'].get('mittel', float('nan')):.3f}; Band {NK_BAND})"}
    u["H1"] = _ni(r, nk, "H1", ("A", "S3"))
    u["H2"] = _urteil_h2(r, nk)
    u["H3"] = _ni(r, nk, "H3", ("Kd", "Ks"), _kopf_pruefung)
    u["H4"] = _ni(r, nk, "H4", ("Kcd", "Kcs"))
    g = lambda a: f"{r['arme'][a]['mittel'] * 100:.2f}".replace(".", ",") if a in r["arme"] else "n/a"
    u["B"] = "beschreibend: Genauigkeit % " + ", ".join(f"{a} {g(a)}" for a in ARME1 + ARME2) + "; Rest in auswertung"
    if zeit is not None:
        u["B-Zeit"] = "beschreibend: Zeit(dicht)/Zeit(Struktur), 1 Thread: " + "; ".join(
            f"{k} " + ", ".join(f"{b} {w}" for b, w in v.items()) for k, v in zeit_text(zeit).items())
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, off=None, sd=0.001, n=5, koll=None, nk=0.10, muster=None):
    base = {"A": 0.560, "A2": 0.580, "S1": 0.500, "S2": 0.530, "S3": 0.555, "Kd": 0.700, "Ks": 0.695, "Kcd": 0.690, "Kcs": 0.688}
    base.update(off or {})
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        for a, v in base.items():
            teil = "t1" if a in ARME1 else "t2"
            x = v + gem + rng.normal(0, sd)
            if koll == a and i == 0:
                x -= 0.2
            if muster and a in ("S3", "Ks", "Kcs"):
                x += muster[i]      # Mittel 0: Differenzen schwanken ohne Effekt
            zl.append({"teil": teil, "arm": a, "seed": 2300 + i, "acc": x})
        for a in NK_ARME:
            zl.append({"teil": "nk", "arm": a, "seed": 2300 + i, "acc": nk + rng.normal(0, 0.01)})
    return zl


def _synth_param(off=None):
    p = {a: {"breit": 0, "kopf": 0, "gesamt": 0} for a in ARME1 + ARME2}
    p["Kd"].update(breit=4_194_304, kopf=4_235_274)
    p["Ks"].update(breit=8192, kopf=49_162)
    p.update(off or {})
    return p


def _u(rng, param_off=None, **kw):
    return urteil(auswertung(_synth(rng, **kw), _synth_param(param_off)))


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER: {name}", flush=True)
            ok = False

    # KanalFastfood = Matrix H D2 H D1 je Pixel, orthogonal
    gen = torch.Generator().manual_seed(3)
    kf = KanalFastfood(64, gen)
    x = torch.randn(2, 64, 3, 3)
    H = walsh_klein(6)
    U = H @ torch.diag(kf.d2.detach()) @ H @ torch.diag(kf.d1.detach())
    soll = torch.einsum("ij,bjhw->bihw", U, x)
    pruefe("KanalFastfood = Matrix", torch.allclose(kf(x), soll, atol=1e-4))
    pruefe("KanalFastfood normerhaltend", torch.allclose(kf(x).norm(dim=1), x.norm(dim=1), rtol=1e-4))
    # Formen, Parameterzählung, Trainierbarkeit je Arm
    akt = {"S1": ("z2",), "S2": ("z2", "abs"), "S3": ("z2", "abs", "relu"), "Ks": ("abs",)}
    x = torch.randn(3, 3072)
    for arm in ARME1 + ARME2:
        netz = Netz(arm, 0, akt)
        y = netz(x)
        pruefe(f"Form {arm}", y.shape == (3, 10))
        y.sum().backward()
        pruefe(f"Gradient bei breiter Schicht {arm}", all(p.grad is not None for p in netz.breit.parameters()))
    pruefe("Parameter A = 3072·4096+4096", param_breit(Netz("A", 0, akt)) == 3072 * 4096 + 4096)
    pruefe("Parameter A2 = A + 4096²+4096", param_breit(Netz("A2", 0, akt)) == 3072 * 4096 + 4096 + 4096 * 4096 + 4096)
    for k, arm in ((1, "S1"), (2, "S2"), (3, "S3")):
        pruefe(f"Parameter {arm} = {k}·8192", param_breit(Netz(arm, 0, akt)) == k * 8192)
    pruefe("Parameter Ks = 8192 (1024->4096, 4 Blöcke)", param_breit(Netz("Ks", 0, akt)) == 2 * 4 * 1024)
    pruefe("Parameter Kd", param_breit(Netz("Kd", 0, akt)) == 1024 * 4096 + 4096)
    pruefe("Parameter Kcs = 128, Kcd = 64·64+64", param_breit(Netz("Kcs", 0, akt)) == 128 and param_breit(Netz("Kcd", 0, akt)) == 64 * 64 + 64)
    pruefe("H3-Kopfverhältnis >= 10", param_kopf(Netz("Kd", 0, akt)) / param_kopf(Netz("Ks", 0, akt)) >= KOPF_FAKTOR)
    # Ks gibt Fastfood mit 4 Blöcken 1024 -> 4096; S2/S3 zweite Schicht hat 1 Block
    pruefe("S3 Schicht 2/3 je 1 Block", all(l.m == 1 for l in Netz("S3", 0, akt).breit.ls[1:]) and Netz("S3", 0, akt).breit.ls[0].din == 3072)
    # Trainierbarkeit (Verlust fällt) auf synthetischer Aufgabe mit kleinem Netz
    torch.manual_seed(0)
    X = torch.randn(512, 3072)
    Y = (X[:, :10].sum(1) > 0).long()
    for arm in ("A", "S2", "Kd", "Ks", "Kcs"):
        netz = Netz(arm, 0, akt, N=256) if arm in ("A", "S2", "Kd", "Ks") else Netz(arm, 0, akt)
        opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=3e-3)
        l0 = float(F.cross_entropy(netz(X), Y))
        for _ in range(40):
            opt.zero_grad()
            F.cross_entropy(netz(X), Y).backward()
            opt.step()
        pruefe(f"Trainierbar {arm}", float(F.cross_entropy(netz(X), Y)) < 0.7 * l0)
    pruefe("Seeds frisch", len(SEEDS) == 5 and not (set(SEEDS) & _BELEGT) and TUNING_SEED not in SEEDS and TUNING_SEED not in _BELEGT)
    pruefe("Profile: Fallback kürzer", all(PROFILE["fallback"][k] <= PROFILE["standard"][k] for k in PROFILE["standard"]))
    d = np.array([-0.004, -0.006, -0.005, -0.004, -0.006])
    pruefe("NI: -0,5 pp ist nichtunterlegen", _stat(d, NI_GRENZE)["p_gr"] < 0.01)
    pruefe("NI: -2 pp ist es nicht", _stat(d - 0.015, NI_GRENZE)["p_gr"] > 0.9 and _stat(d - 0.015, NI_GRENZE)["p_kl"] < 0.05)
    zm = zeit_messung(akt, arme=("A", "S3"), batches=(1,))
    pruefe("Zeitmessung", all(z["batch1_s"] > 0 for z in zm))
    return ok


def _fn_test(a):
    return _schl(a), {"v": a[2] * 2}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as t:
        pfad = Path(t) / "z.json"
        kennung = {"k": 1}
        aufg = [("t1", a, s) for a in ("A", "S3") for s in (1, 2)]
        z1 = _rechne(aufg[:2], pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        z2 = _rechne(aufg, pfad, kennung, fn=_fn_test, prozesse=1, schluessel=_schl)
        ok &= len(z1) == 2 and len(z2) == 4 and z2["einheit|t1|A|2"] == z1["einheit|t1|A|2"]
        try:
            _rechne(aufg, pfad, {"k": 2}, fn=_fn_test, prozesse=1, schluessel=_schl)
            ok = False
        except ValueError:
            pass
    if not ok:
        print("  Vortest FEHLER: Wiederaufnahme", flush=True)
    return ok


def _vortest_urteile() -> bool:
    """Jeden Urteilszweig erzwingen (synthetisch, n=5)."""
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

    erwarte("Normalfall", _u(rng), **{"NK": "BESTANDEN", "H1": "NICHT_UNTERLEGEN", "H2": "TIEFE_HILFT", "H3": "NICHT_UNTERLEGEN",
                                      "H4": "NICHT_UNTERLEGEN"})
    erwarte("alle unterlegen", _u(rng, off={"S3": 0.50, "Ks": 0.65, "Kcs": 0.64}), **{"H1": "H0", "H3": "H0", "H4": "H0"})
    erwarte("Tiefe schadet", _u(rng, off={"S3": 0.48, "S1": 0.50}), **{"H2": "H0"})
    erwarte("Tiefe nicht erkennbar", _u(rng, off={"S3": 0.5005, "S1": 0.500}, sd=0.003), **{"H2": "HU"})
    erwarte("Rauschen", _u(rng, muster=(0.03, -0.03, 0.02, -0.02, 0.0), off={"S3": 0.56, "S1": 0.56, "Ks": 0.70, "Kcs": 0.70, "Kcd": 0.70}), **{"H1": "HU", "H2": "HU", "H3": "HU", "H4": "HU"})
    erwarte("Kopfparameter reichen nicht", _u(rng, param_off={"Ks": {"breit": 3_000_000, "kopf": 3_040_000, "gesamt": 3_050_000}}), **{"H3": "HU"})
    erwarte("NK faellt", _u(rng, nk=0.30), **{"NK": "DURCHGEFALLEN", "H1": "HU (Negativkontrolle", "H2": "HU (Negativkontrolle",
                                              "H3": "HU (Negativkontrolle", "H4": "HU (Negativkontrolle"})
    erwarte("Kollaps S3/Ks", _u(rng, koll="S3"), **{"H1": "HU (Instabilitaet", "H2": "HU (Instabilitaet", "H3": "NICHT_UNTERLEGEN"})
    erwarte("Kollaps Ks", _u(rng, koll="Ks"), **{"H3": "HU (Instabilitaet"})
    erwarte("Arm fehlt", urteil(auswertung([z for z in _synth(rng) if z["arm"] != "Kcs"], _synth_param())), **{"H4": "HU (Arme fehlen"})
    erwarte("Fehlender Nutzen von A2 egal", _u(rng, off={"A2": 0.7}), **{"H1": "NICHT_UNTERLEGEN"})
    a = _u(rng, off={"S3": 0.52})["H1"].split(" ")[0]
    b = _u(rng, off={"S3": 0.60})["H1"].split(" ")[0]
    if a == b:
        print("  Vortest FEHLER: H1 unterscheidet die Richtung nicht", flush=True)
        ok = False
    erlaubt = ("NICHT_UNTERLEGEN", "TIEFE_HILFT", "HU", "H0", "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    if not all(isinstance(v, str) and v.startswith(erlaubt) for v in _u(rng).values()):
        print("  Vortest FEHLER: Urteil ausserhalb des Vokabulars", flush=True)
        ok = False
    soll = {"H1": {"NICHT_UNTERLEGEN", "H0", "HU"}, "H2": {"TIEFE_HILFT", "H0", "HU"}, "H3": {"NICHT_UNTERLEGEN", "H0", "HU"},
            "H4": {"NICHT_UNTERLEGEN", "H0", "HU"}}
    for h, ws in soll.items():
        if not ws <= {w for k, w in gesehen if k == h}:
            print(f"  Vortest FEHLER: Zweige {ws - {w for k, w in gesehen if k == h}} fuer {h} nicht erzwungen", flush=True)
            ok = False
    return ok


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok &= _vortest_wiederaufnahme()
    ok &= _vortest_urteile()
    return bool(ok)


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, zeit=True, profil=None):
    profil = profil or os.environ.get("I044_PROFIL", "standard")
    P = PROFILE[profil]
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, 1 Epoche
        seeds, klein = (TUNING_SEED,), True
        P = {"ep1": 1, "ep2": 1, "sel": 1, "nk": 1}
    else:
        seeds, klein = SEEDS, False
        if set(seeds) & _BELEGT or TUNING_SEED in seeds:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    ep = {**{a: P["ep1"] for a in ARME1}, **{a: P["ep2"] for a in ARME2}}
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "profil": "rauch" if schnell else profil, "epochen": P, "lr_gitter": list(LR_GITTER),
               "sel_lr": SEL_LR, "tuning_seed": TUNING_SEED, "klein": klein}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "akt" not in zw:
        akt, roh = waehle_akt(TUNING_SEED, P["sel"], klein)
        zw["akt"], zw["akt_roh"] = {k: list(v) for k, v in akt.items()}, roh
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Nichtlinearitaeten gewaehlt: {zw['akt']}", flush=True)
    akt = {k: tuple(v) for k, v in zw["akt"].items()}
    if "tuning" not in zw:
        zw["tuning"] = tuning(ARME1 + ARME2, akt, ep, klein, TUNING_SEED)
        _schreibe_zwischen(pfad, kennung, zw)
        print(f"  Tuning fertig: {({a: v['lr'] for a, v in zw['tuning'].items()})}", flush=True)
    hp = zw["tuning"]
    aufg = [("t1", a, s, hp[a]["lr"], akt, ep[a], klein) for a in ARME1 for s in seeds]
    aufg += [("t2", a, s, hp[a]["lr"], akt, ep[a], klein) for a in ARME2 for s in seeds]
    aufg += [("nk", a, s, hp[a]["lr"], akt, P["nk"], klein) for a in NK_ARME for s in seeds]
    kosten = {"A2": 6, "A": 4, "S3": 4, "S2": 3, "Kd": 3, "Ks": 3, "S1": 2, "Kcd": 2, "Kcs": 2}
    aufg.sort(key=lambda a: -kosten.get(a[1], 1))        # teuerste zuerst (Lastverteilung)
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [z for k, z in alle.items() if k.startswith("einheit|")]
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread(akt)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    netze = {a: Netz(a, 0, akt) for a in ARME1 + ARME2}
    params = {a: {"breit": param_breit(n), "kopf": param_kopf(n), "gesamt": param_gesamt(n)} for a, n in netze.items()}
    e = {"abteilung": "Struktur", "modul": "struktur_tief", "karte": KARTE, "version": 1, "profil": profil, "hypothesen": HYPOTHESEN,
         "folgefragen": FOLGEFRAGEN, "seeds": list(seeds), "tuning_seed": TUNING_SEED, "nichtlinearitaeten": {k: list(v) for k, v in akt.items()},
         "akt_roh": zw["akt_roh"], "tuning": hp, "einstellungen": {"epochen": ep, "nk_epochen": P["nk"], "batch": BATCH, "lr_gitter": list(LR_GITTER)},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, params)
    if zm is not None:
        e["zeitmessung"] = {"roh": zm, "zusammenfassung": zeit_text(zm)}
    e["urteil"] = urteil(e["auswertung"], zm)
    return e
