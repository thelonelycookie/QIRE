"""Abteilung Skalierung -- I-041 „Skalierung n=8 -> 10“: bleibt der Vorsprung der festen Mischung mit Quadrat bei 32x32?
(Plan: PLAN_2026-10-06_gesamt.md §2 P5. Die Frage ist klassisch; Quantenstufe Q2.)

HEILMEIER:
  Was wollen wir?  Antwort auf „Effizienz im Großen“ ohne Quantenrhetorik: Wächst der Vorsprung Haar² − Pixel-Softmax
      und der Born-Vorsprung (Quadrat gegen |z|) mit n = 8 -> 10 (16x16 -> 32x32), und ist Fastfood² bei 32x32 noch
      gleichwertig zu Haar²? Dazu beschreibend: Zeit FWHT-Fastfood gegen dichte Matmul (numpy, 1 Thread) und Operationszählung.
  Wie heute?  I-038 (16x16): Haar² − Pixel +2,84 pp (MNIST) / +1,43 pp (Fashion); Born² − |z| +0,35 / 0,00 pp;
      Fastfood-2 ≡ Haar² (GLEICHWERTIG).
  Was ist neu?  Auflösungs-Differenz-in-Differenz (gepaart je Seed), dasselbe Verfahren und derselbe Kopf bei beiden Auflösungen.

DATEN (festgeschrieben): MNIST und Fashion-MNIST, 6.000 Training / 5.000 offizielle Test je Seed, gleiche Bildauswahl je Seed
  bei beiden Auflösungen (gleicher Generator -> gleiche Bilder, damit die Auflösungs-Differenzen gepaart sind).
  16x16: Flächenmittel aus 28x28 (labor.gemeinsam.bilder, unverändert wie I-038).
  32x32: bilineares Hochskalieren aus 28x28 (torch interpolate, mode=bilinear, align_corners=False), Werte auf [0,1] beschnitten.
  EHRLICH: 32x32 ist Resampling, kein neuer Bildinhalt. Skaliert wird die Merkmalszahl (1.024 gegen 256), nicht die Information.
ARME je Auflösung (alle außer Pixel: x -> x/‖x‖ -> z = U x -> f(z) -> DERSELBE Kopf, Softmax-Regression, standardisierte Merkmale,
  L-BFGS, L2 je (Datensatz, Auflösung, Arm) auf Tuning-Seed 2097 aus Gitter, Validierung 1/6 der Trainingsbilder, nie Test):
  pix (Pixel-Softmax), haar_sq (Haar-Orthogonale, Quadrat), fastfood2 ((H D2 H D1)², Quadrat), haar_abs (Haar, |z|; bester
  klassischer Gegner aus I-038, fest, nicht neu gewählt), ident_sq (Identität, Quadrat), haar_zufallslabel (Negativkontrolle).
HYPOTHESEN je Datensatz (Seeds 2060-2069, gepaart):
  H1 Haar² − Pixel bei 32x32 > 0 (einseitiger exakter Wilcoxon).
  H2 (Haar² − Pixel)_32 − (Haar² − Pixel)_16 > 0 (einseitig, Differenz je Seed).
  H3 (Haar² − |z|)_32 − (Haar² − |z|)_16 > 0 (einseitig, gleiche Differenz-in-Differenz).
  Holm über H1-H3 je Datensatz. H4 Fastfood² ≡ Haar² bei 32x32 (TOST ±0,5 pp), eigene Familie (nicht in Holm).
KONTROLLEN: NK Zufallslabel (Haar², beide Auflösungen) Mittel in NK_BAND, sonst alles HU. PK: Haar² − Pixel bei 16x16 (Replikation
  I-038 H-D) einseitig p < 0,05 und Mittel >= 0,5 pp, sonst H4 HU. Kollaps (Arm-Seed < Median − 5 pp) in einem beteiligten Arm -> HU.
BESCHREIBEND: Zeitmessung (numpy, 1 Thread) FWHT-Fastfood gegen dichte Matmul, N = 256..16.384, Batch 1 und 64, plus Zählmodell.
KRITERIEN stehen im Code (urteil()); Urteile sind Strings.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from abteilungen.clifford_suche import holm, kopf_logistisch, normiere
from abteilungen.clifford_suche_v2 import tost_p, waehle_l2
from abteilungen.struktur_aequivalenz import wilcoxon_einseitig
from abteilungen.walsh_kontrolle import wilcoxon_schnell
from labor.gemeinsam import ERGEBNISSE, _roh, bilder, zusammenfassung

KARTE = "I-041"
DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
AUFLOESUNGEN = (16, 32)
SEEDS = tuple(range(2060, 2070))
TUNING_SEED = 2097
_BELEGT = set(range(2000, 2060)) | set(range(900, 1001)) | {2090, 2091, 2092, 2095, 2098, 2099}
ARME = ("pix", "haar_sq", "fastfood2", "haar_abs", "ident_sq")
L2_GITTER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
L2_RUNDEN = 4                     # höchstens 4 Verlängerungen des Gitters (Faktor 10^0,5)
TOST_W = 0.005
ALPHA = 0.05
KOLLAPS = 0.05
NK_BAND = (0.09, 0.115)
PK_MIN = 0.005
FAMILIE = ("H1", "H2", "H3")
ZEIT_N = (256, 512, 1024, 2048, 4096, 8192, 16384)
ZEIT_BATCH = (1, 64)

HYPOTHESEN = {
    "H1": "Haar² − Pixel-Softmax bei 32x32 > 0 (einseitiger exakter Wilcoxon, Holm x3). HAAR_BESSER | H0 (Gegenrichtung) | HU.",
    "H2": "(Haar² − Pixel)_32 − (Haar² − Pixel)_16 > 0 (einseitig, Holm x3). VORSPRUNG_WAECHST | H0 (Vorsprung schrumpft) | HU.",
    "H3": "(Haar² − |z|)_32 − (Haar² − |z|)_16 > 0 (einseitig, Holm x3). BORN_VORSPRUNG_WAECHST | H0 (Born-Vorsprung schrumpft) | HU.",
    "H4": "Fastfood² ≡ Haar² bei 32x32 (TOST ±0,5 pp, eigene Familie). GLEICHWERTIG | FASTFOOD_BESSER | HAAR_BESSER | HU.",
    "NK": f"Zufallslabel (Haar², 16 und 32) Mittel in {NK_BAND}. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "PK": "Haar² − Pixel bei 16x16: einseitig p < 0,05 und Mittel >= 0,5 pp (Replikation I-038 H-D). Sonst H4 HU.",
    "B": "beschreibend: alle Arm-Mittel, weitere Differenzen, L2-Tuning, Zeitmessung und Zählmodell.",
}
FOLGEFRAGEN = {
    "H2 VORSPRUNG_WAECHST": "Mischung+Quadrat lohnt mit n: eine Stufe weiter (n=12) oder echte Daten mit mehr Merkmalen.",
    "H2 H0": "Vorsprung schrumpft: Kernaussage auf 16x16 begrenzen; Resampling-Effekt (kein neuer Inhalt) mitnennen.",
    "H4 GLEICHWERTIG": "Fastfood² genügt bei 1.024 Merkmalen: Zeitmessung bestimmt, ab wann das auch schneller ist.",
}


# ------------------------------------------------------------------ Daten
def bilder_aufl(name, seed, seite, n_train, n_test):
    """16: labor.gemeinsam.bilder (Flächenmittel). 32: gleiche Bildauswahl, bilineares Hochskalieren (siehe Kopf)."""
    if seite == 16:
        return bilder(name, seed, n_train=n_train, n_test=n_test, seite=16)
    x, y = _roh(name)
    g = torch.Generator().manual_seed(seed)
    tr = torch.randperm(60_000, generator=g)[:n_train]
    te = 60_000 + torch.randperm(10_000, generator=g)[:n_test]
    idx = torch.cat([tr, te])
    xs = torch.tensor(x)[idx].reshape(-1, 1, 28, 28)
    xs = torch.nn.functional.interpolate(xs, size=(seite, seite), mode="bilinear", align_corners=False)
    xs = xs.clamp(0.0, 1.0).reshape(len(idx), -1)
    ys = torch.tensor(y)[idx]
    return xs[:n_train], ys[:n_train], xs[n_train:], ys[n_train:]


def _daten(ds, seed, res, klein):
    return bilder_aufl(ds, seed, res, 600 if klein else 6_000, 500 if klein else 5_000)


# ------------------------------------------------------------------ Bausteine
def walsh(n):
    H = torch.ones(1, 1)
    for _ in range(n):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(2 ** n)


def haar(g, d):
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q * torch.sign(torch.diagonal(R))


def vorzeichen(g, d):
    return (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float()


def fastfood2(g, d):
    n = int(round(math.log2(d)))
    H = walsh(n)
    return H @ torch.diag(vorzeichen(g, d)) @ H @ torch.diag(vorzeichen(g, d))


def _g(seed, res):
    return torch.Generator().manual_seed(seed + 7000 + res)


def _matrizen(seed, res):
    """Haar-Matrix, dann Fastfood-Vorzeichen aus demselben Generator (feste Reihenfolge)."""
    d = res * res
    g = _g(seed, res)
    return {"haar": haar(g, d), "ff": fastfood2(g, d)}


def _merkmale(arm, xa, xb, M):
    """Merkmale (train, test) je Arm; Pixel roh, sonst normiert -> U -> Nichtlinearität."""
    if arm == "pix":
        return xa, xb
    na, nb = normiere(xa), normiere(xb)
    if arm == "ident_sq":
        return na ** 2, nb ** 2
    U = M["ff"] if arm == "fastfood2" else M["haar"]
    f = (lambda x: (x @ U.T).abs()) if arm == "haar_abs" else (lambda x: (x @ U.T) ** 2)
    return f(na), f(nb)


def _acc(arm, xa, ya, xb, yb, M, l2):
    fa, fb = _merkmale(arm, xa, xb, M)
    a = kopf_logistisch(fa, ya, fb, yb, l2=l2)[0]
    return a if math.isfinite(a) else 0.1


# ------------------------------------------------------------------ Tuning (Seed 2097, Validierung 1/6 der Trainingsbilder)
def tuning(datensaetze, seed, klein):
    hp = {}
    for ds in datensaetze:
        hp[ds] = {}
        for res in AUFLOESUNGEN:
            xtr, ytr, _, _ = _daten(ds, seed, res, klein)
            nv = len(ytr) // 6
            xa, ya, xb, yb = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
            M = _matrizen(seed, res)
            for arm in ARME:
                gitter = {l2: None for l2 in L2_GITTER}
                for _ in range(L2_RUNDEN + 1):
                    for l2 in gitter:
                        if gitter[l2] is None:
                            gitter[l2] = _acc(arm, xa, ya, xb, yb, M, l2)
                    w = waehle_l2({l2: [v] for l2, v in gitter.items()})
                    if w["erweitern"] == "unten":
                        gitter[min(gitter) / 10 ** 0.5] = None
                    elif w["erweitern"] == "oben":
                        gitter[max(gitter) * 10 ** 0.5] = None
                    else:
                        break
                fertig = {l2: v for l2, v in gitter.items() if v is not None}
                w = waehle_l2({l2: [v] for l2, v in fertig.items()})
                hp[ds][f"{arm}_{res}"] = {"l2": w["l2"], "am_rand": w["erweitern"] is not None,
                                          "val": {repr(k): v for k, v in sorted(fertig.items())}}
    return hp


# ------------------------------------------------------------------ Arbeiter je (Datensatz, Seed)
def _seed_arbeiter(a):
    ds, s, hp, klein = a
    t0 = time.time()
    z = {"seed": s, "datensatz": ds}
    for res in AUFLOESUNGEN:
        xtr, ytr, xte, yte = _daten(ds, s, res, klein)
        M = _matrizen(s, res)
        for arm in ARME:
            z[f"{arm}_{res}"] = _acc(arm, xtr, ytr, xte, yte, M, hp[ds][f"{arm}_{res}"]["l2"])
        p = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(s + 9000 + res))
        fa, fb = _merkmale("haar_sq", xtr, xte, M)
        z[f"haar_zufallslabel_{res}"] = kopf_logistisch(fa, ytr[p], fb, yte,
                                                        l2=hp[ds][f"haar_sq_{res}"]["l2"])[0]
    z["sekunden"] = time.time() - t0
    return f"seed|{ds}|{s}", z


# ------------------------------------------------------------------ Wiederaufnahme je Seed
def _lade_zwischen(pfad: Path, kennung: dict) -> dict:
    if not pfad.exists():
        return {}
    inhalt = json.loads(pfad.read_text())
    if inhalt.get("kennung") != kennung:
        raise ValueError(f"Zwischendatei {pfad} stammt aus anderen Einstellungen -- nicht mischen; umbenennen")
    return inhalt["zeilen"]


def _schreibe_zwischen(pfad: Path, kennung: dict, zeilen: dict):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"karte": KARTE, "kennung": kennung, "zeilen": zeilen}, ensure_ascii=False))
    os.replace(tmp, pfad)


def _rechne(aufgaben, pfad: Path, kennung: dict, fn=None, prozesse=None, schluessel=None) -> dict:
    """Jeder fertige (Datensatz, Seed) geht SOFORT in die Zwischendatei; fertige werden übersprungen."""
    import multiprocessing as mp
    from labor.gemeinsam import _init_arbeiter
    fn = fn or _seed_arbeiter
    schluessel = schluessel or (lambda a: f"seed|{a[0]}|{a[1]}")
    zeilen = _lade_zwischen(pfad, kennung)
    offen = [a for a in aufgaben if schluessel(a) not in zeilen]
    if len(offen) < len(aufgaben):
        print(f"  Wiederaufnahme: {len(aufgaben) - len(offen)} von {len(aufgaben)} aus {pfad.name}", flush=True)
    prozesse = prozesse or int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))

    def ablegen(erg):
        k, z = erg
        zeilen[k] = z
        _schreibe_zwischen(pfad, kennung, zeilen)
        print(f"  fertig: {k} ({len(zeilen)} in Zwischendatei)", flush=True)

    if prozesse <= 1:
        _init_arbeiter()
        for a in offen:
            ablegen(fn(a))
    elif offen:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
            for erg in pool.imap_unordered(fn, offen):
                ablegen(erg)
    return zeilen


# ------------------------------------------------------------------ Zeitmessung und Zählmodell (beschreibend)
def fwht(x):
    """Schnelle Walsh-Hadamard-Transformation (Sylvester-Ordnung, normiert) über die letzte Achse, x: (B, N)."""
    B, N = x.shape
    h = 1
    while h < N:
        x = x.reshape(B, -1, 2, h)
        x = np.stack((x[:, :, 0, :] + x[:, :, 1, :], x[:, :, 0, :] - x[:, :, 1, :]), axis=2)
        h *= 2
    return x.reshape(B, N) / np.float32(math.sqrt(N))


def fastfood_fwht(x, d1, d2):
    """(H D2 H D1) x per FWHT: 2 FWHT + 2 Vorzeichenmultiplikationen."""
    return fwht(fwht(x * d1) * d2)


def operationen(N):
    """Zählmodell je Vektor (Gleitkomma-Operationen, ohne Speicherzugriffe).
    dicht: N² Multiplikationen + N(N−1) Additionen = 2N² − N.
    Fastfood-2: 2 FWHT je (N/2)·log2N Schmetterlinge à 2 Additionen = 2·N·log2N, dazu 2N Vorzeichen und 1N Skalierung."""
    lg = int(round(math.log2(N)))
    dicht = 2 * N * N - N
    ff = 2 * N * lg + 3 * N
    return {"N": N, "dicht": dicht, "fastfood": ff, "verhaeltnis": dicht / ff,
            "parameter_dicht": N * N, "bits_fastfood": 2 * N}


def _stoppuhr(f, mindest=0.02, wdh=7):
    f()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        dt = time.perf_counter() - t0
        if dt >= mindest or n >= 4096:
            break
        n *= 4
    best = dt / n
    for _ in range(wdh - 1):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def zeit_messung(ns=ZEIT_N, batches=ZEIT_BATCH, seed=0):
    """Sekunden je Aufruf (Minimum über Wiederholungen): dichte Matmul gegen FWHT-Fastfood, float32."""
    rng = np.random.default_rng(seed)
    aus = []
    for N in ns:
        z = {"N": N, "ops": operationen(N)}
        try:
            W = rng.standard_normal((N, N), dtype=np.float32)
        except MemoryError:
            z["fehler"] = "Speicher"
            aus.append(z)
            continue
        d1 = (rng.integers(0, 2, N) * 2 - 1).astype(np.float32)
        d2 = (rng.integers(0, 2, N) * 2 - 1).astype(np.float32)
        WT = np.ascontiguousarray(W.T)
        for B in batches:
            x = rng.standard_normal((B, N), dtype=np.float32)
            td = _stoppuhr(lambda: x @ WT)
            tf = _stoppuhr(lambda: fastfood_fwht(x, d1, d2))
            z[f"batch{B}"] = {"dicht_s": td, "fastfood_s": tf, "verhaeltnis": td / tf}
        del W, WT
        aus.append(z)
    return aus


def _zeit_arbeiter(a):
    return zeit_messung(*a)


def zeit_im_ein_thread(ns=ZEIT_N, batches=ZEIT_BATCH):
    """Messung in einem frischen Prozess mit BLAS = 1 Thread (Umgebung wird vor dem numpy-Import des Kindes gesetzt)."""
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter, ((tuple(ns), tuple(batches)),))


def zeit_urteil(zm) -> dict:
    """Beschreibend: kleinstes N, ab dem FWHT je Batch schneller ist (gemessen), und kleinstes N mit weniger Operationen."""
    u = {}
    for B in ZEIT_BATCH:
        ns = [z["N"] for z in zm if f"batch{B}" in z and z[f"batch{B}"]["verhaeltnis"] > 1.0]
        # erstes N, ab dem FWHT für alle größeren N schneller bleibt
        alle = [z for z in zm if f"batch{B}" in z]
        ab = None
        for i, z in enumerate(alle):
            if all(w[f"batch{B}"]["verhaeltnis"] > 1.0 for w in alle[i:]):
                ab = z["N"]
                break
        u[f"batch{B}"] = {"fastfood_schneller_ab_N": ab, "n_schneller": ns,
                          "verhaeltnis": {str(z["N"]): round(z[f"batch{B}"]["verhaeltnis"], 3) for z in alle}}
    u["operationen_verhaeltnis"] = {str(z["N"]): round(z["ops"]["verhaeltnis"], 2) for z in zm}
    return u


# ------------------------------------------------------------------ Lauf
def _pruefe_seeds(seeds, datensaetze):
    alle = set(seeds)
    return alle == set(SEEDS) and not (alle & _BELEGT) and TUNING_SEED not in alle


def lauf(schnell=False, datensaetze=DATENSAETZE, prozesse=None, zwischen=None, zeit=True):
    datensaetze = tuple(datensaetze)
    if schnell:
        seeds, tseed, klein = (TUNING_SEED,), TUNING_SEED, True        # Rauchtest NUR auf dem Tuning-Seed
        zn, zb = (256, 512), ZEIT_BATCH
    else:
        seeds, tseed, klein = SEEDS, TUNING_SEED, False
        zn, zb = ZEIT_N, ZEIT_BATCH
        if not _pruefe_seeds(seeds, datensaetze):
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "l2_gitter": list(L2_GITTER), "tuning_seed": tseed, "klein": klein,
               "datensaetze": list(datensaetze), "arme": list(ARME)}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" in zw:
        hp = zw["tuning"]
        print("  Wiederaufnahme: Tuning aus Zwischendatei", flush=True)
    else:
        hp = tuning(datensaetze, tseed, klein)
        zw["tuning"] = hp
        _schreibe_zwischen(pfad, kennung, zw)
    aufg = [(ds, s, hp, klein) for ds in datensaetze for s in seeds]
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zeilen = {ds: [alle[f"seed|{ds}|{s}"] for s in seeds] for ds in datensaetze}
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread(zn, zb)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    e = {"abteilung": "Skalierung", "modul": "skalierung_n10", "karte": KARTE, "version": 1,
         "hypothesen": HYPOTHESEN, "folgefragen": FOLGEFRAGEN, "seeds": list(seeds), "tuning_seed": tseed,
         "tuning": hp, "einstellungen": {"l2_gitter": list(L2_GITTER), "tost": TOST_W, "klein": klein,
                                         "daten": "16: Flächenmittel; 32: bilinear (align_corners=False), geklemmt [0,1]"},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen)
    if zeit:
        e["zeitmessung"] = {"roh": zm, "zusammenfassung": zeit_urteil(zm),
                            "zaehlmodell": "dicht 2N²−N; Fastfood-2 2N·log2N+3N je Vektor; Parameter N² gegen 2N Bits"}
    e["urteil"] = urteil(e["auswertung"])
    if zeit:
        z = e["zeitmessung"]["zusammenfassung"]
        e["urteil"]["B-Zeit"] = ("beschreibend: FWHT-Fastfood schneller ab N = " +
                                 ", ".join(f"{z[f'batch{B}']['fastfood_schneller_ab_N']} (Batch {B})" for B in ZEIT_BATCH) +
                                 "; Operationsverhältnis dicht/Fastfood bei N=1024: " +
                                 str(z["operationen_verhaeltnis"].get("1024")))
    return e


# ------------------------------------------------------------------ Statistik und Auswertung
def _stat(d) -> dict:
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return {"n": int(len(d))}
    from scipy.stats import t as t_vert
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return {"mittel_pp": m * 100, "sd_pp": float(d.std(ddof=1)) * 100, "ki_pp": [(m - q * se) * 100, (m + q * se) * 100],
            "positiv": int((d > 0).sum()), "n": int(len(d)), "p_zweiseitig": wilcoxon_schnell(d),
            "p_gr": wilcoxon_einseitig(d, True), "p_kl": wilcoxon_einseitig(d, False), "p_tost": tost_p(d, TOST_W)}


def _arr(zz, k):
    return np.array([z[k] for z in zz], dtype=float)


def kollaps_seeds(zz, k):
    w = _arr(zz, k)
    med = float(np.median(w))
    return [z["seed"] for z in zz if not (z[k] >= med - KOLLAPS)]


# Welche Arm-Spalten gehen in welche Hypothese ein (für die Kollapsprüfung)
BETEILIGT = {"H1": ("haar_sq_32", "pix_32"), "H2": ("haar_sq_32", "pix_32", "haar_sq_16", "pix_16"),
             "H3": ("haar_sq_32", "haar_abs_32", "haar_sq_16", "haar_abs_16"), "H4": ("fastfood2_32", "haar_sq_32"),
             "PK": ("haar_sq_16", "pix_16")}


def _ds_auswertung(zz) -> dict:
    r = {"n": len(zz)}
    spalten = [f"{a}_{res}" for res in AUFLOESUNGEN for a in ARME] + [f"haar_zufallslabel_{res}" for res in AUFLOESUNGEN]
    r["arme"] = {k: zusammenfassung(_arr(zz, k)) for k in spalten if k in zz[0]}
    A = lambda k: _arr(zz, k)
    r["vorsprung_pix"] = {str(res): _stat(A(f"haar_sq_{res}") - A(f"pix_{res}")) for res in AUFLOESUNGEN}
    r["vorsprung_born"] = {str(res): _stat(A(f"haar_sq_{res}") - A(f"haar_abs_{res}")) for res in AUFLOESUNGEN}
    d32, d16 = A("haar_sq_32") - A("pix_32"), A("haar_sq_16") - A("pix_16")
    b32, b16 = A("haar_sq_32") - A("haar_abs_32"), A("haar_sq_16") - A("haar_abs_16")
    r["diff"] = {"H1": _stat(d32), "H2": _stat(d32 - d16), "H3": _stat(b32 - b16),
                 "H4": _stat(A("fastfood2_32") - A("haar_sq_32")), "PK": _stat(d16)}
    r["beschreibend"] = {
        "fastfood2 - haar_sq @16": _stat(A("fastfood2_16") - A("haar_sq_16")),
        "haar_sq - ident_sq @16": _stat(A("haar_sq_16") - A("ident_sq_16")),
        "haar_sq - ident_sq @32": _stat(A("haar_sq_32") - A("ident_sq_32")),
        "haar_sq - haar_abs @16": r["vorsprung_born"]["16"], "haar_sq - haar_abs @32": r["vorsprung_born"]["32"],
        "pix 32 - pix 16": _stat(A("pix_32") - A("pix_16")), "haar_sq 32 - haar_sq 16": _stat(A("haar_sq_32") - A("haar_sq_16")),
        "fastfood2 32 - fastfood2 16": _stat(A("fastfood2_32") - A("fastfood2_16"))}
    r["kollaps"] = {k: kollaps_seeds(zz, k) for k in spalten if k in zz[0] and "zufallslabel" not in k}
    prim = {h: (r["diff"][h]["p_gr"] if "p_gr" in r["diff"][h] else 1.0) for h in FAMILIE}
    r["p_holm"] = holm(prim)
    r["negativ"] = {str(res): r["arme"][f"haar_zufallslabel_{res}"]["mittel"] for res in AUFLOESUNGEN}
    return r


def auswertung(zeilen: dict) -> dict:
    return {"datensaetze": {ds: _ds_auswertung(zz) for ds, zz in zeilen.items()}}


# ------------------------------------------------------------------ Urteile (Kriterien im Code)
def _f(x):
    return f"{x:+.2f}".replace(".", ",")


def _nk(a) -> str:
    ok = all(NK_BAND[0] <= v <= NK_BAND[1] for r in a["datensaetze"].values() for v in r["negativ"].values())
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _pk(r) -> str:
    d = r["diff"]["PK"]
    if "mittel_pp" not in d:
        return "DURCHGEFALLEN"
    ok = d["p_gr"] < ALPHA and d["mittel_pp"] >= PK_MIN * 100
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _zahlen(r, h):
    d = r["diff"][h]
    z = f"d {_f(d['mittel_pp'])} pp, 90-%-KI [{_f(d['ki_pp'][0])}; {_f(d['ki_pp'][1])}], n {d['n']}"
    return z + (f", p_Holm {r['p_holm'][h]:.3g}" if h in FAMILIE else f", p_TOST {d['p_tost']:.3g}")


def _urteil_h(h, r, nk, pk) -> str:
    d = r["diff"][h]
    if "mittel_pp" not in d:
        return "HU (zu wenige gepaarte Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    koll = sorted({s for k in BETEILIGT[h] for s in r["kollaps"].get(k, [])})
    if koll:
        return f"HU (Instabilitaet/Kollaps in Seeds {koll})"
    z = _zahlen(r, h)
    if h == "H4":
        if d["p_tost"] < ALPHA:
            return f"GLEICHWERTIG ({z})" if pk == "BESTANDEN" else f"HU (Positivkontrolle PK-16 nicht gefeuert; {z})"
        if d["p_zweiseitig"] < ALPHA:
            return f"{'FASTFOOD' if d['mittel_pp'] > 0 else 'HAAR'}_BESSER ({z}, p_zweiseitig {d['p_zweiseitig']:.3g})"
        return f"HU ({z})"
    ja = {"H1": "HAAR_BESSER", "H2": "VORSPRUNG_WAECHST", "H3": "BORN_VORSPRUNG_WAECHST"}[h]
    nein = {"H1": "Gegenrichtung", "H2": "Vorsprung schrumpft", "H3": "Born-Vorsprung schrumpft"}[h]
    if r["p_holm"][h] < ALPHA:
        return f"{ja} ({z})"
    if d["p_zweiseitig"] < ALPHA and d["mittel_pp"] < 0:
        return f"H0 ({nein}, p_zweiseitig {d['p_zweiseitig']:.3g}; {z})"
    return f"HU ({z})"


def urteil(a: dict) -> dict:
    nk = _nk(a)
    u = {"NK": f"{nk} (Zufallslabel-Mittel: " + ", ".join(
        f"{NAME[ds]} {res} {v:.4f}" for ds, r in a["datensaetze"].items() for res, v in r["negativ"].items()) +
         f"; Band {NK_BAND})"}
    for ds, r in a["datensaetze"].items():
        pk = _pk(r)
        d = r["diff"]["PK"]
        u[f"PK-{NAME[ds]}"] = f"{pk} (Haar² − Pixel @16: {_f(d.get('mittel_pp', float('nan')))} pp)"
        for h in FAMILIE + ("H4",):
            u[f"{h}-{NAME[ds]}"] = _urteil_h(h, r, nk, pk)
        vp, vb = r["vorsprung_pix"], r["vorsprung_born"]
        g = lambda x: _f(x["mittel_pp"]) if "mittel_pp" in x else "n/a"
        u[f"B-{NAME[ds]}"] = (f"beschreibend: Haar² − Pixel 16/32 {g(vp['16'])}/{g(vp['32'])} pp; Haar² − |z| 16/32 "
                              f"{g(vb['16'])}/{g(vb['32'])} pp; Rest in auswertung.beschreibend")
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, off=None, sd=0.001, n=10, koll=None, nk=0.10):
    base = {f"{a}_{res}": 0.90 for res in AUFLOESUNGEN for a in ARME}
    base.update({"haar_sq_16": 0.935, "pix_16": 0.910, "haar_abs_16": 0.930, "fastfood2_16": 0.935,
                 "haar_sq_32": 0.945, "pix_32": 0.910, "haar_abs_32": 0.930, "fastfood2_32": 0.945})
    base.update(off or {})
    zz = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        z = {"seed": 2060 + i}
        for k, v in base.items():
            z[k] = v + gem + rng.normal(0, sd)
        for res in AUFLOESUNGEN:
            z[f"haar_zufallslabel_{res}"] = nk + rng.normal(0, 0.002)
        if koll and i == 0:
            z[koll] -= 0.2
        zz.append(z)
    return zz


def _u(rng, **kw):
    n = kw.pop("n", 10)
    return urteil(auswertung({"mnist": _synth(rng, n=n, **kw)}))


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER: {name}", flush=True)
            ok = False

    I8 = torch.eye(256)
    M = _matrizen(1, 16)
    pruefe("Haar orthogonal 256", float((M["haar"] @ M["haar"].T - I8).abs().max()) < 1e-4)
    pruefe("Fastfood orthogonal 256", float((M["ff"] @ M["ff"].T - I8).abs().max()) < 1e-4)
    pruefe("Fastfood dicht", float((M["ff"].abs() > 1e-6).float().mean()) > 0.5)
    M10 = _matrizen(1, 32)
    I10 = torch.eye(1024)
    pruefe("Haar orthogonal 1024", float((M10["haar"] @ M10["haar"].T - I10).abs().max()) < 1e-3)
    pruefe("Fastfood orthogonal 1024", float((M10["ff"] @ M10["ff"].T - I10).abs().max()) < 1e-3)
    # FWHT = Walsh-Matrix; Fastfood per FWHT = dichte Fastfood-Matrix
    x = np.random.default_rng(0).standard_normal((3, 256)).astype(np.float32)
    pruefe("FWHT = Walsh-Matrix", np.allclose(fwht(x), x @ walsh(8).numpy().T, atol=1e-4))
    g = _g(1, 16)
    haar(g, 256)
    torch.manual_seed(0)
    gg = torch.Generator().manual_seed(5)
    d1, d2 = vorzeichen(gg, 256), vorzeichen(gg, 256)
    Hm = walsh(8)
    Uff = Hm @ torch.diag(d2) @ Hm @ torch.diag(d1)
    pruefe("Fastfood-FWHT = Matrix", np.allclose(fastfood_fwht(x, d1.numpy(), d2.numpy()), x @ Uff.numpy().T, atol=1e-4))
    o = operationen(1024)
    pruefe("Zählmodell", o["dicht"] == 2 * 1024 ** 2 - 1024 and o["fastfood"] == 2 * 1024 * 10 + 3 * 1024
           and o["parameter_dicht"] == 1024 ** 2 and o["bits_fastfood"] == 2048)
    pruefe("Zählmodell monoton", operationen(16384)["verhaeltnis"] > operationen(256)["verhaeltnis"])
    zm = zeit_messung((256, 512), (1, 4))
    pruefe("Zeitmessung liefert Werte", all(z["batch1"]["dicht_s"] > 0 and z["batch4"]["fastfood_s"] > 0 for z in zm)
           and set(zeit_urteil([dict(z, **{"batch64": z["batch4"]}) for z in zm]).keys()) >= {"batch1", "batch64"})
    # Auflösungen: Formen, gleiche Bilder je Seed, Werte im Bereich
    a16 = bilder_aufl("mnist", 2097, 16, 50, 40)
    a32 = bilder_aufl("mnist", 2097, 32, 50, 40)
    pruefe("Datenformen/gleiche Bilder", a16[0].shape == (50, 256) and a32[0].shape == (50, 1024) and a32[2].shape == (40, 1024)
           and bool((a16[1] == a32[1]).all()) and bool((a16[3] == a32[3]).all()) and 0.0 <= float(a32[0].min())
           and float(a32[0].max()) <= 1.0)
    # Seeds
    pruefe("Seeds frisch", _pruefe_seeds(SEEDS, DATENSAETZE) and len(SEEDS) == 10 and TUNING_SEED not in SEEDS
           and not (set(SEEDS) & _BELEGT) and not _pruefe_seeds(tuple(range(2000, 2010)), DATENSAETZE))
    # Statistik
    dd = np.random.default_rng(1).normal(0.01, 0.003, 10)
    pruefe("Wilcoxon einseitig Richtung", wilcoxon_einseitig(dd, True) < 0.01 and wilcoxon_einseitig(dd, False) > 0.99)
    h = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    pruefe("Holm", abs(h["a"] - 0.03) < 1e-12 and abs(h["b"] - 0.04) < 1e-12)
    pruefe("Familie", FAMILIE == ("H1", "H2", "H3"))
    return ok


def _fn_test(a):
    return f"seed|{a[0]}|{a[1]}", {"v": a[0] * 2}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as t:
        pfad = Path(t) / "z.json"
        kennung = {"k": 1}
        aufg = [("mnist", i) for i in range(4)]
        z1 = _rechne(aufg[:2], pfad, kennung, fn=_fn_test, prozesse=1)
        z2 = _rechne(aufg, pfad, kennung, fn=_fn_test, prozesse=1)
        ok &= len(z1) == 2 and len(z2) == 4 and z2["seed|mnist|1"] == z1["seed|mnist|1"]
        try:
            _rechne(aufg, pfad, {"k": 2}, fn=_fn_test, prozesse=1)
            ok = False
        except ValueError:
            pass
        ok &= not pfad.with_name(pfad.name + ".tmp").exists()
    if not ok:
        print("  Vortest FEHLER: Wiederaufnahme", flush=True)
    return ok


def _vortest_urteile() -> bool:
    """Jeden Urteilszweig erzwingen (synthetische Daten)."""
    rng = np.random.default_rng(11)
    ok = True
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k.split("-")[0], w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    M = "-MNIST"
    erwarte("Normalfall", _u(rng), **{"NK": "BESTANDEN", "PK-MNIST": "BESTANDEN", "H1" + M: "HAAR_BESSER",
                                      "H2" + M: "VORSPRUNG_WAECHST", "H3" + M: "BORN_VORSPRUNG_WAECHST",
                                      "H4" + M: "GLEICHWERTIG"})
    erwarte("H2 schrumpft", _u(rng, off={"pix_32": 0.94}), **{"H1" + M: "HAAR_BESSER", "H2" + M: "H0 (Vorsprung schrumpft"})
    erwarte("H1 Gegenrichtung", _u(rng, off={"pix_32": 0.96}), **{"H1" + M: "H0 (Gegenrichtung",
                                                                 "H2" + M: "H0 (Vorsprung schrumpft"})
    erwarte("H3 schrumpft", _u(rng, off={"haar_abs_32": 0.96}), **{"H3" + M: "H0 (Born-Vorsprung schrumpft",
                                                                  "H1" + M: "HAAR_BESSER"})
    erwarte("Rauschen", _u(rng, sd=0.02, off={"pix_32": 0.944, "haar_abs_32": 0.944, "pix_16": 0.934, "haar_abs_16": 0.934}),
            **{"H1" + M: "HU", "H2" + M: "HU", "H3" + M: "HU", "H4" + M: "HU"})
    erwarte("H4 Fastfood besser", _u(rng, off={"fastfood2_32": 0.96}), **{"H4" + M: "FASTFOOD_BESSER"})
    erwarte("H4 Haar besser", _u(rng, off={"fastfood2_32": 0.93}), **{"H4" + M: "HAAR_BESSER"})
    erwarte("PK faellt", _u(rng, off={"pix_16": 0.935}), **{"PK-MNIST": "DURCHGEFALLEN", "H4" + M: "HU (Positivkontrolle"})
    erwarte("NK faellt", _u(rng, nk=0.30), **{"NK": "DURCHGEFALLEN", "H1" + M: "HU (Negativkontrolle",
                                              "H4" + M: "HU (Negativkontrolle"})
    erwarte("Kollaps", _u(rng, koll="haar_sq_32"), **{"H1" + M: "HU (Instabilitaet", "H2" + M: "HU (Instabilitaet",
                                                      "H3" + M: "HU (Instabilitaet", "H4" + M: "HU (Instabilitaet"})
    erwarte("Kollaps nur 16er-Pixel", _u(rng, koll="pix_16"), **{"H1" + M: "HAAR_BESSER", "H2" + M: "HU (Instabilitaet",
                                                                 "H3" + M: "BORN_VORSPRUNG_WAECHST"})
    erwarte("n=1", urteil(auswertung({"mnist": _synth(rng, n=1)})), **{"H1" + M: "HU (zu wenige"})
    # Mutationstest: Richtung der Differenz muss das Urteil ändern
    a = _u(rng, off={"pix_32": 0.90})["H2" + M].split(" ")[0]
    b = _u(rng, off={"pix_32": 0.95})["H2" + M].split(" ")[0]
    if a == b:
        print("  Vortest FEHLER: H2 unterscheidet die Richtung nicht", flush=True)
        ok = False
    erlaubt = ("GLEICHWERTIG", "FASTFOOD_BESSER", "HAAR_BESSER", "VORSPRUNG_WAECHST", "BORN_VORSPRUNG_WAECHST", "HU", "H0",
               "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    u = _u(rng)
    if not all(isinstance(v, str) and v.startswith(erlaubt) for v in u.values()):
        print("  Vortest FEHLER: Urteil ausserhalb des Vokabulars", flush=True)
        ok = False
    soll = {"H1": {"HAAR_BESSER", "H0", "HU"}, "H2": {"VORSPRUNG_WAECHST", "H0", "HU"},
            "H3": {"BORN_VORSPRUNG_WAECHST", "H0", "HU"}, "H4": {"GLEICHWERTIG", "FASTFOOD_BESSER", "HAAR_BESSER", "HU"}}
    for h, ws in soll.items():
        erreicht = {w for k, w in gesehen if k == h}
        if not ws <= erreicht:
            print(f"  Vortest FEHLER: Zweige {ws - erreicht} fuer {h} nicht erzwungen", flush=True)
            ok = False
    return ok


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok &= _vortest_wiederaufnahme()
    ok &= _vortest_urteile()
    return bool(ok)
