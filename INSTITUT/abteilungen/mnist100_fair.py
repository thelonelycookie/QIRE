"""Abteilung Clifford/Effizienz -- I-039-v2 „fairer Bit-Vergleich“ (Agenda I-039-v2; Folge von I-039).

HEILMEIER:
  Was wollen wir?  I-039: b_ff (Fastfood -> Born-Quadrat -> datengewaehlte Gruppensummen + 10x10-Kopf) 80,46 % mit
      110 reellen Parametern und 882 Strukturbits (formal), klassisch (d) 74,84 % OHNE Strukturbits. Bleibt der
      Vorsprung, wenn klassische Modelle DASSELBE Strukturbit-Budget (<= ~900 Bit formal) bekommen?
  Wie heute?  Offen: I-039 H2 verglich nur reelle Parameter; die ~900 Strukturbits des Born-Arms standen daneben.
  Was ist neu?  Klassische Arme mit 110 reellen Parametern (Kopf 10x10 + 10 Bias) und gleichem Bitbudget, damit
      die Bitzaehlung auf beiden Seiten gleich ist; Arm k3 isoliert die Born-Nichtlinearitaet bei GLEICHER Struktur.

ARME (gleiche Daten je Seed; alle haben Kopf 10x10 + 10 Bias = 110 reell, ausser d = 118):
  b_ff      Born-Arm EXAKT wie I-039 (Code aus mnist100 wiederverwendet): Fastfood-Mischung, (xn W^T)^2, Gruppierung
            datengewaehlt, Gruppensummen sqrt(10 G), Kopf. 850 Bit Gruppierung + 32 Bit Seed = 882 formal.
  k1_lin    Pixel (L2-normiert) direkt in 10 datengewaehlte Gruppen (je Pixel die Klasse mit groesstem standardisierten
            Klassenmittel-Kontrast), Gruppensummen, Kopf. 850 Bit, kein Seed.
  k1_quad   = Arm c aus I-039 (quadrierte Pixel, gruppiert): 850 Bit.
  k1_fisher Fisher-Top-79-Pixel (K1_M), diese datengewaehlt in 10 Gruppen, Summen, Kopf: 79 * (8 + log2 10) = 894,2 Bit.
  k2_relu / k2_abs  zufaellige +-1-Projektion Z = xn R^T / 16 (R 256x256, Seed = 32 Bit), Nichtlinearitaet ReLU bzw. |z|,
            datengewaehlte Gruppierung (850 Bit), Gruppensummen, Kopf: 882 Bit formal (streng: 65.536 + 850).
  k3_relu / k3_abs  GLEICHE Fastfood-Mischung und GLEICHE Gruppierung wie b_ff, aber |z| bzw. ReLU(z) statt z^2: 882 Bit.
  d         klassisch 5x5-Mittelung -> 3 tanh -> 10 (118 reell, 0 Bit); zulaessiger klassischer Arm (Bit <= Budget).
  Kontrollen: pk_haar (PK-1: Voll-Kopf 256x10 auf 16 sqrt(P) Haar, muss >= 85 % erreichen), PK-2: d >= 70 %,
            nk (NK: b_ff mit permutierten Trainingslabels, muss <= 15 % bleiben).
  Mutual Information als Pixelwahl wird NICHT verwendet (nur Fisher; vorab festgelegt).
  Merkmalsskala: Gruppensummen der Arme k1/k2/k3 werden mit der festen Konstante SKALA multipliziert (zaehlt nicht).

ZAEHLREGEL (verbindlich, wie I-039): reell = trainierte reelle Zahlen (Kopf + Bias); Strukturbits getrennt:
  bits_formal (Zufallsmatrix nur als Seed = 32 Bit; Gruppierung 256 log2 10 = 850; Fisher 8 Bit je Pixel + log2 10
  je Gruppenzuweisung) und bits_streng (Informationsgehalt der Zufallsmatrix: Fastfood ~2.196, Rademacher 65.536,
  Haar 16 Bit je Eintrag). Berichtet werden beide, streng GETRENNT. Budget-Kriterium BIT_BUDGET = 900 gilt formal.

DATEN/SEEDS/RATEN: wie I-039 (MNIST 16x16, 6.000 Train, 5.000 Test); Mess-Seeds 2100-2119, Tuning-Seeds 2190-2192
  (Validierung 1.200 der 6.000, nie Test; labor.gemeinsam.gittersuche_stabil, Gitter wie I-039).
  Der beste klassische Arm (H1) und der beste k3-Arm (H2) werden nach mittlerer Validierungsgenauigkeit der
  Tuning-Seeds festgelegt (Gleichstand: weniger reell, weniger Bit formal, Name), vor jeder Testzahl.

HYPOTHESEN (vorab; Familie {H1, H2}, Holm, alpha = 0,05):
  H1  b_ff - bester klassischer Arm mit gleichem Bitbudget > 0 (einseitiger exakter Wilcoxon, Mittel > 0).
      Kandidaten: k1_lin, k1_quad, k1_fisher, k2_relu, k2_abs, k3_relu, k3_abs, d.
  H2  b_ff - bester k3-Arm > 0 (einseitiger exakter Wilcoxon, Mittel > 0): traegt die Born-Nichtlinearitaet allein?
  Zusatz berichtet (ohne Holm): TOST +-1 pp fuer beide Differenzen; alle Arme mit Mittel/sd/reell/Bit/bit_pro_pp.

URTEILSZWEIGE (Strings; Kriterien im Code; Marke H1 = Alternative, H0 = verfehlt, HU = unentschieden):
  Vorbedingungen PK-1, PK-2, NK, Instabilitaet (> 1 Seed unter Median - 5 pp) -> HU.
  H1/H2: „H1 VORSPRUNG_BLEIBT" | „H1 GLEICHWERTIG" (TOST) | „H0 KLASSISCH_BESSER" | „HU NICHT_ENTSCHIEDEN"
         (H2: „H1 BORN_NICHTLINEARITAET_TRAEGT" statt VORSPRUNG_BLEIBT, „H0 K3_BESSER").

REALISTISCHE ERWARTUNG (vor dem Lauf): k3 (|z|) lag in I-038 nur ~0,35 pp unter dem Quadrat; daher ist H2 knapp bis
  GLEICHWERTIG zu erwarten, H1 haengt am besten k1/k2-Arm: ein Verlust des Vorsprungs ist ausdruecklich moeglich.

OFFENLEGUNG: Es lief kein Test-Seed-Lauf; Rauchtest NUR auf Tuning-Seeds (--schnell, 150 Schritte).
LAUFZEIT: siehe Rauchtest im LABORBUCH (INSTITUT_PROZESSE=4). WIEDERAUFNAHME: je Seed ergebnisse/_zwischen_I-039-v2/seed_<s>.json
  und tuning.json (Konfig-Hash).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labor import gemeinsam as gem  # noqa: E402
from abteilungen import mnist100 as m  # noqa: E402  (Born-Arm, Daten, Statistik wiederverwendet)

# ------------------------------------------------------------------ Konstanten (vorregistriert)
SEEDS = tuple(range(2100, 2120))
TUNING_SEEDS = (2190, 2191, 2192)
SCHNELL_SEEDS = (2190,)                  # Rauchtest NUR auf einem Tuning-Seed
SCHRITTE = m.SCHRITTE
ALPHA = 0.05
DELTA = 1.0                              # TOST +-1 pp
PK1_MIN, PK2_MIN, NK_MAX = 85.0, 70.0, 15.0
KOLLAPS_PP, KOLLAPS_MAX = 5.0, 1
BIT_BUDGET = 900.0
K1_M = 79                                # Fisher-Pixel: 79 * (8 + log2 10) = 894,2 Bit
SKALA = 4.0
LOG2_10 = math.log2(10)
K3 = ("k3_abs", "k3_relu")
KLASSISCH = ("k1_lin", "k1_quad", "k1_fisher", "k2_relu", "k2_abs", "k3_relu", "k3_abs", "d")
ARME_GRAD = ("b_ff",) + KLASSISCH + ("pk_haar",)
ARME = ARME_GRAD + ("nk",)
ZWISCHEN = gem.ERGEBNISSE / "_zwischen_I-039-v2"
HYPOTHESEN = {
    "H1": "b_ff - bester klassischer Arm mit gleichem Strukturbit-Budget (<= ~900 Bit formal) > 0.",
    "H2": "b_ff - bester k3-Arm (gleiche Struktur, |z|/ReLU statt Quadrat) > 0.",
}


def _info(arm: str) -> dict:
    """reell + Strukturbits (formal/streng) je Arm; getrennt."""
    if arm in ("b_ff", "nk", "pk_haar", "d"):
        i = m._info("b_ff" if arm == "nk" else arm)
        return {k: i[k] for k in ("reell", "bits_formal", "bits_streng")}
    g = m.GRUPPEN_BITS
    if arm in ("k1_lin", "k1_quad"):
        return {"reell": 110, "bits_formal": g, "bits_streng": g}
    if arm == "k1_fisher":
        b = K1_M * (8 + LOG2_10)
        return {"reell": 110, "bits_formal": b, "bits_streng": b}
    if arm in ("k2_relu", "k2_abs"):
        return {"reell": 110, "bits_formal": g + 32, "bits_streng": g + 256 * 256}
    if arm in K3:
        i = m._info("b_ff")
        return {"reell": 110, "bits_formal": i["bits_formal"], "bits_streng": i["bits_streng"]}
    raise KeyError(arm)


# ------------------------------------------------------------------ Merkmale
def _norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def rademacher(seed: int) -> torch.Tensor:
    rng = np.random.default_rng([seed, 3])
    return torch.tensor(rng.choice([-1.0, 1.0], (256, 256)) / 16.0, dtype=torch.float32)


def born_gruppe(xa, ya, seed):
    """Fastfood-Mischung W und Gruppierung grp EXAKT wie in b_ff (I-039)."""
    W = m.mischung("ff", seed)
    return W, m.gruppe_daten(m.born(xa, W), ya)


def _summen(F, grp):
    return SKALA * m.gruppen_summen(F, grp)


def eingaben(arm: str, xa, ya, xb, seed: int):
    """(Aa, Ab, Kopfart) fuer die neuen Arme; datenabhaengige Teile nur aus (xa, ya)."""
    if arm in K3:
        W, grp = born_gruppe(xa, ya, seed)
        f = (lambda x: (_norm(x) @ W.T).abs()) if arm == "k3_abs" else (lambda x: (_norm(x) @ W.T).clamp_min(0))
        return _summen(f(xa), grp), _summen(f(xb), grp), "linear"
    if arm in ("k2_relu", "k2_abs"):
        R = rademacher(seed)
        f = (lambda x: (_norm(x) @ R.T * 16).abs()) if arm == "k2_abs" else (lambda x: (_norm(x) @ R.T * 16).clamp_min(0))
        Fa, Fb = f(xa), f(xb)
        grp = m.gruppe_daten((Fa / 16) ** 2, ya)
        return _summen(Fa, grp), _summen(Fb, grp), "linear"
    if arm == "k1_lin":
        Fa, Fb = _norm(xa), _norm(xb)
        grp = m.gruppe_daten((Fa / 16) ** 2, ya)
        return _summen(Fa, grp), _summen(Fb, grp), "linear"
    if arm == "k1_fisher":
        s = m.fisher_top(xa, ya, K1_M)
        Fa, Fb = _norm(xa)[:, s], _norm(xb)[:, s]
        grp = m.gruppe_daten((Fa / 16) ** 2, ya)
        return _summen(Fa, grp), _summen(Fb, grp), "linear"
    raise KeyError(arm)


def arm_grad(arm, xa, ya, xb, yb, lr, seed, schritte=SCHRITTE):
    """Liefert (Genauigkeit auf (xb, yb), reelle Parameter). Alte Arme laufen ueber mnist100 (identischer Code)."""
    if arm in ("b_ff", "d", "pk_haar", "nk"):
        return _alt(arm, xa, ya, xb, yb, lr, seed, schritte)
    if arm == "k1_quad":
        return m.arm_grad("c", xa, ya, xb, yb, lr, seed, schritte)
    Aa, Ab, art = eingaben(arm, xa, ya, xb, seed)
    fn, params = m.trainiere(art, Aa, ya, lr, seed, schritte)
    return m._genauigkeit(fn, Ab, yb), m.zaehle_params(params)


def _alt(arm, xa, ya, xb, yb, lr, seed, schritte):
    if arm == "nk":                      # Negativkontrolle auf b_ff: Trainingslabels permutiert
        yp = ya[torch.randperm(len(ya), generator=torch.Generator().manual_seed(seed + 31337))]
        return m.arm_grad("b_ff", xa, yp, xb, yb, lr, seed, schritte)
    return m.arm_grad(arm, xa, ya, xb, yb, lr, seed, schritte)


# ------------------------------------------------------------------ Arbeiter
def _tune_arbeiter(a):
    arm, seed, lr, schritte = a
    x, y, _, _ = m.daten(seed)
    fit, val = m.val_split(len(y), seed)
    acc, _ = arm_grad(arm, x[fit], y[fit], x[val], y[val], lr, seed, schritte)
    return {"acc": acc}


def _konfig(schritte, n_train, n_test, tuning_seeds):
    roh = json.dumps([schritte, n_train, n_test, list(tuning_seeds), m.TAU, list(m.LR_GITTER), list(ARME), K1_M, SKALA])
    return hashlib.sha1(roh.encode()).hexdigest()[:12]


def _seed_arbeiter(a):
    seed, raten, schritte, ordner, konfig = a
    pfad = Path(ordner) / f"seed_{seed}.json"
    if pfad.exists():
        z = json.loads(pfad.read_text())
        if z.get("konfig") == konfig:
            return z
    t0 = time.time()
    x, y, xt, yt = m.daten(seed)
    acc, reell = {}, {}
    for arm in ARME:
        acc[arm], reell[arm] = arm_grad(arm, x, y, xt, yt, raten["b_ff" if arm == "nk" else arm], seed, schritte)
    z = {"seed": seed, "acc": acc, "reell": reell, "sekunden": time.time() - t0, "konfig": konfig}
    Path(ordner).mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(z))
    return z


# ------------------------------------------------------------------ Auswertung und Urteil
def waehle(val: dict) -> dict:
    """Festlegung auf Validierung (vor jeder Testzahl). Gleichstand: weniger reell, weniger Bit formal, Name."""
    def schl(a):
        i = _info(a)
        return (-round(val[a], 12), i["reell"], i["bits_formal"], a)
    return {"klass": min((a for a in KLASSISCH if a in val), key=schl),
            "k3": min((a for a in K3 if a in val), key=schl)}


def _paar(pa, pb):
    d = np.asarray(pa) - np.asarray(pb)
    p_ni, p_ob = m.t_einseitig(d, -DELTA), m.t_einseitig(-d, -DELTA)
    return {"diff_mittel": float(d.mean()), "diff_sd": float(d.std(ddof=1)), "p_besser": m.wilcoxon_einseitig(d),
            "p_schlechter": m.wilcoxon_einseitig(-d), "p_tost": max(p_ni, p_ob),
            "tost_aequivalent": bool(max(p_ni, p_ob) < ALPHA)}


def auswertung(zeilen: list, wahl: dict) -> dict:
    pp = {a: [100.0 * z["acc"][a] for z in zeilen] for a in ARME}
    arme = {}
    for a in ARME:
        i, s = _info(a), m._stat(pp[a])
        arme[a] = {**s, **i, "bit_pro_pp_formal": m.bit_pro_pp(i["reell"], i["bits_formal"], s["mittel"]),
                   "bit_pro_pp_streng": m.bit_pro_pp(i["reell"], i["bits_streng"], s["mittel"]),
                   "kollaps_seeds": [zeilen[k]["seed"] for k in m.kollaps_seeds(pp[a])]}
    kb, k3 = wahl["klass"], wahl["k3"]
    h1, h2 = _paar(pp["b_ff"], pp[kb]), _paar(pp["b_ff"], pp[k3])
    h = m.holm({"H1": h1["p_besser"], "H2": h2["p_besser"]})
    h1.update(arm=kb, p_holm=h["H1"], klass_mittel=arme[kb]["mittel"], b_mittel=arme["b_ff"]["mittel"])
    h2.update(arm=k3, p_holm=h["H2"], klass_mittel=arme[k3]["mittel"], b_mittel=arme["b_ff"]["mittel"])
    ge = lambda w: [a for a in w if len(arme[a]["kollaps_seeds"]) > KOLLAPS_MAX]
    return {"n_seeds": len(zeilen), "wahl": wahl, "arme": arme, "H1": h1, "H2": h2,
            "pk": {"pk1_mittel": arme["pk_haar"]["mittel"], "pk2_mittel": arme["d"]["mittel"], "nk_mittel": arme["nk"]["mittel"]},
            "instabil": {"H1": ge(["b_ff", kb]), "H2": ge(["b_ff", k3])},
            "alle_paare_vs_b_ff": {a: float(np.mean(np.asarray(pp["b_ff"]) - np.asarray(pp[a]))) for a in KLASSISCH}}


def urteil(a: dict) -> dict:
    """Frage -> String. Kriterien stehen HIER (vor dem Lauf festgelegt)."""
    u, pk, arme = {}, a["pk"], a["arme"]
    u["PK-1"] = (f"PK_OK Voll-Kopf {pk['pk1_mittel']:.2f} % >= {PK1_MIN:g} %" if pk["pk1_mittel"] >= PK1_MIN
                 else f"HU PK_VERSAGT Voll-Kopf {pk['pk1_mittel']:.2f} % < {PK1_MIN:g} %: Born-Maschinerie defekt")
    u["PK-2"] = (f"PK_OK klassisch (d) {pk['pk2_mittel']:.2f} % >= {PK2_MIN:g} %" if pk["pk2_mittel"] >= PK2_MIN
                 else f"HU PK_VERSAGT klassisch (d) {pk['pk2_mittel']:.2f} % < {PK2_MIN:g} %: Referenz zu schwach")
    u["NK"] = (f"PK_OK Zufallslabel {pk['nk_mittel']:.2f} % <= {NK_MAX:g} %" if pk["nk_mittel"] <= NK_MAX
               else f"HU PK_VERSAGT Zufallslabel {pk['nk_mittel']:.2f} % > {NK_MAX:g} %: Leck")
    ok = all(u[k].startswith("PK_OK") for k in ("PK-1", "PK-2", "NK"))
    bud = [k for k in KLASSISCH if arme[k]["bits_formal"] > BIT_BUDGET]
    for h, ja, nein, name in (("H1", "VORSPRUNG_BLEIBT", "KLASSISCH_BESSER", "bester klassischer Arm"),
                              ("H2", "BORN_NICHTLINEARITAET_TRAEGT", "K3_BESSER", "bester k3-Arm")):
        r = a[h]
        if not ok:
            u[h] = f"HU VORBEDINGUNG Kontrolle verfehlt (PK-1/PK-2/NK), {h} nicht lesbar"
        elif bud:
            u[h] = f"HU BUDGET Arm(e) {', '.join(bud)} ueber {BIT_BUDGET:g} Bit formal"
        elif a["instabil"][h]:
            u[h] = f"HU INSTABILITAET Arm(e) {', '.join(a['instabil'][h])}: > {KOLLAPS_MAX} Seeds unter Median - {KOLLAPS_PP:g} pp"
        else:
            kopf = f"b_ff - {r['arm']} = {r['diff_mittel']:+.2f} pp ({r['b_mittel']:.2f} vs. {r['klass_mittel']:.2f} %"
            if r["p_holm"] < ALPHA and r["diff_mittel"] > 0:
                u[h] = f"H1 {ja} {kopf}; Holm-p {r['p_holm']:.4f}; TOST p {r['p_tost']:.3f})"
            elif r["tost_aequivalent"]:
                u[h] = f"H1 GLEICHWERTIG {kopf}; TOST +-{DELTA:g} pp p {r['p_tost']:.4f}; Vorsprung nicht gezeigt)"
            elif r["p_schlechter"] < ALPHA and r["diff_mittel"] < 0:
                u[h] = f"H0 {nein} {kopf}; p {r['p_schlechter']:.4f})"
            else:
                u[h] = f"HU NICHT_ENTSCHIEDEN {kopf}; Holm-p {r['p_holm']:.3f}, TOST p {r['p_tost']:.3f})"
    best = max(KLASSISCH, key=lambda k: arme[k]["mittel"])
    u["Z"] = (f"Z beschreibend: b_ff {arme['b_ff']['mittel']:.2f} %; bester klassischer Arm im Test {best} {arme[best]['mittel']:.2f} %; "
              + ", ".join(f"{k} {arme[k]['mittel']:.1f}" for k in KLASSISCH))
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, ordner=None, **kw):
    if schnell:
        seeds, schritte = SCHNELL_SEEDS, 150
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-039-v2-schnell"
    ordner = Path(ordner or ZWISCHEN)
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    if len(tuning_seeds) < 3:
        raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
    if not schnell and set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    if schnell and not set(seeds) <= set(tuning_seeds):
        raise ValueError("Rauchtest nur auf Tuning-Seeds")
    konfig = _konfig(schritte, m.N_TRAIN, m.N_TEST, tuning_seeds)
    t0 = time.time()
    ordner.mkdir(parents=True, exist_ok=True)
    tp = ordner / "tuning.json"
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != konfig:
        info, _ = gem.gittersuche_stabil(list(ARME_GRAD), lambda arm, lr: [(arm, s, lr, schritte) for s in tuning_seeds],
                                         lambda aufg: gem.parallel(_tune_arbeiter, aufg), gitter=m.LR_GITTER, faktor=2)
        print(f"  Raten nach {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v['lr']:g}" for k, v in info.items()), flush=True)
        val = {arm: info[arm]["gitter"][str(info[arm]["lr"])]["mittel"] for arm in ARME_GRAD}
        tun = {"konfig": konfig, "info": info, "val": val, "wahl": waehle(val), "tuning_seeds": tuning_seeds}
        tp.write_text(json.dumps(tun, indent=1))
    raten = {k: v["lr"] for k, v in tun["info"].items()}
    print(f"  Wahl (Validierung, vor Test): {tun['wahl']}", flush=True)
    zeilen = gem.parallel(_seed_arbeiter, [(s, raten, schritte, str(ordner), konfig) for s in seeds])
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + " ".join(f"{a} {100 * z['acc'][a]:.1f}" for a in ARME), f"({z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "clifford", "modul": "mnist100_fair", "agenda": "I-039-v2", "datensatz": "mnist", "hypothesen": HYPOTHESEN,
         "seeds": seeds, "tuning_seeds": tuning_seeds, "schritte": schritte, "raten": tun["info"], "validierung": tun["val"],
         "wahl": tun["wahl"], "zeilen": zeilen, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, tun["wahl"])
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Vortest
def _synth(mittel, sd=0.3, n=20, seed=0, kollaps=None, sd_arm=None):
    rng = np.random.default_rng(seed)
    w = {}
    for a in ARME:
        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        v = mittel[a] + (sd_arm or {}).get(a, sd) * z
        for i in range((kollaps or {}).get(a, 0)):
            v[i] = mittel[a] - 15.0
        w[a] = v
    return [{"seed": 3000 + i, "acc": {a: float(w[a][i] / 100.0) for a in ARME}} for i in range(n)]


def _basis(**o):
    b = {"b_ff": 80.0, "k1_lin": 60.0, "k1_quad": 77.0, "k1_fisher": 62.0, "k2_relu": 74.0, "k2_abs": 75.0,
         "k3_relu": 78.0, "k3_abs": 79.0, "d": 75.0, "pk_haar": 92.0, "nk": 10.0}
    b.update(o)
    return b


def vortest() -> bool:
    global BIT_BUDGET
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)
    # --- Zaehlung numerisch und Budget
    for a in ("b_ff",) + tuple(k for k in KLASSISCH if k != "d"):
        if _info(a)["reell"] != 110:
            fehler(f"reell {a}")
        if _info(a)["bits_formal"] > BIT_BUDGET:
            fehler(f"Budget {a}")
    if _info("d")["reell"] != 118 or _info("d")["bits_formal"] != 0 or _info("b_ff")["bits_formal"] != 882:
        fehler("d/b_ff Zaehlung")
    if not (893 < _info("k1_fisher")["bits_formal"] < 895) or _info("k2_abs")["bits_streng"] != 850 + 65536:
        fehler("Bitzaehlung k1_fisher/k2")
    if _info("k3_abs") != _info("b_ff") or _info("b_ff")["bits_streng"] < 3000:
        fehler("k3 muss b_ff-Bitzaehlung teilen")
    # --- tatsaechliche Parameterzahlen der Modelle
    for art, d_, sp in (("linear", 10, 110), ("tanh3", 25, 118), ("linear", 256, 2570)):
        if m.zaehle_params(m._modell(art, d_, 0)[0]) != sp:
            fehler(f"Parameterzahl {art}")
    # --- Merkmale: Form, Gruppierung von k3 identisch mit b_ff, Rademacher
    g = torch.Generator().manual_seed(0)
    yb = torch.arange(300) % 10
    xb = torch.rand(300, 256, generator=g) * 0.2
    for c in range(10):
        xb[yb == c, c * 20:c * 20 + 20] += 1.0
    xa, ya, xv, yv = xb[:220], yb[:220], xb[220:], yb[220:]
    R = rademacher(1)
    if set(R.abs().flatten().tolist()) != {1 / 16} or not torch.equal(R, rademacher(1)) or torch.equal(R, rademacher(2)):
        fehler("Rademacher")
    W, grp = born_gruppe(xa, ya, 1)
    Pa = m.born(xa, m.mischung("ff", 1))
    if not torch.equal(grp, m.gruppe_daten(Pa, ya)):
        fehler("k3-Gruppierung != b_ff-Gruppierung")
    for arm in K3 + ("k1_lin", "k1_fisher", "k2_relu", "k2_abs"):
        Aa, Ab, art = eingaben(arm, xa, ya, xv, 1)
        if Aa.shape != (220, 10) or Ab.shape != (80, 10) or art != "linear" or float(Aa.min()) < 0:
            fehler(f"Merkmale {arm}")
    if len(m.fisher_top(xa, ya, K1_M)) != K1_M:
        fehler("Fisher-Top")
    # k3 verwendet exakt die b_ff-Mischung, nur andere Nichtlinearitaet
    A_abs = eingaben("k3_abs", xa, ya, xv, 1)[0]
    zz = (_norm(xa) @ W.T).abs()
    if not torch.allclose(A_abs, _summen(zz, grp)):
        fehler("k3_abs Merkmal")
    # --- Tuning-/Testtrennung, Seeds
    import inspect
    src = inspect.getsource(_tune_arbeiter)
    if "xt" in src or "_, _" not in src:
        fehler("Tuning-Arbeiter beruehrt Testdaten")
    if set(SEEDS) & set(TUNING_SEEDS) or SEEDS[0] != 2100 or SEEDS[-1] != 2119 or TUNING_SEEDS != (2190, 2191, 2192):
        fehler("Seeds")
    if not set(SCHNELL_SEEDS) <= set(TUNING_SEEDS):
        fehler("Rauchtest nicht auf Tuning-Seed")
    # --- Wahl nach Validierung (Gleichstand: weniger reell -> d vor k*, dann Name)
    v = {a: 0.7 for a in KLASSISCH}
    if waehle(v) != {"klass": "k1_lin", "k3": "k3_abs"}:
        fehler("Wahl Gleichstand")
    v = {a: 0.7 for a in KLASSISCH}
    v["k2_abs"] = 0.8
    v["k3_relu"] = 0.75
    if waehle(v) != {"klass": "k2_abs", "k3": "k3_relu"}:
        fehler(f"Wahl nach Validierung {waehle(v)}")
    # --- Urteilszweige erzwingen
    def lauf_u(mit, sd=0.3, koll=None, sd_arm=None):
        zl = _synth(mit, sd=sd, kollaps=koll, sd_arm=sd_arm)
        a = auswertung(zl, {"klass": "k3_abs", "k3": "k3_abs"})
        return a, urteil(a)

    def erwarte(txt, u, k, anf):
        if not u[k].startswith(anf):
            fehler(f"{txt}: {k} = {u[k][:60]!r} erwartet {anf!r}")
    a, u = lauf_u(_basis(b_ff=82.0, k3_abs=79.0))
    for h, an in (("H1", "H1 VORSPRUNG_BLEIBT"), ("H2", "H1 BORN_NICHTLINEARITAET_TRAEGT")):
        erwarte("Vorsprung", u, h, an)
    erwarte("PK-1 ok", u, "PK-1", "PK_OK")
    _, u = lauf_u(_basis(b_ff=79.0, k3_abs=79.0))
    erwarte("Gleichwertig", u, "H1", "H1 GLEICHWERTIG")
    erwarte("Gleichwertig H2", u, "H2", "H1 GLEICHWERTIG")
    _, u = lauf_u(_basis(b_ff=76.0, k3_abs=79.0))
    erwarte("Klassisch besser", u, "H1", "H0 KLASSISCH_BESSER")
    erwarte("k3 besser", u, "H2", "H0 K3_BESSER")
    _, u = lauf_u(_basis(b_ff=79.5, k3_abs=79.0), sd=2.0)
    erwarte("Unentschieden", u, "H1", "HU NICHT_ENTSCHIEDEN")
    _, u = lauf_u(_basis(pk_haar=70.0))
    erwarte("PK-1", u, "PK-1", "HU PK_VERSAGT")
    for k in ("H1", "H2"):
        erwarte("PK-1 sperrt", u, k, "HU VORBEDINGUNG")
    _, u = lauf_u(_basis(d=60.0))
    erwarte("PK-2", u, "PK-2", "HU PK_VERSAGT")
    erwarte("PK-2 sperrt", u, "H1", "HU VORBEDINGUNG")
    _, u = lauf_u(_basis(nk=30.0))
    erwarte("NK", u, "NK", "HU PK_VERSAGT")
    _, u = lauf_u(_basis(), koll={"b_ff": 3})
    for k in ("H1", "H2"):
        erwarte("Instabilitaet", u, k, "HU INSTABILITAET")
    erwarte("1 Kollaps erlaubt", lauf_u(_basis(b_ff=82.0), koll={"b_ff": 1})[1], "H1", "H1 VORSPRUNG_BLEIBT")
    # Budget-Zweig (Konstante kurz anheben ueberprueft die Pruefung)
    alt, BIT_BUDGET = BIT_BUDGET, 800.0
    _, u = lauf_u(_basis())
    BIT_BUDGET = alt
    erwarte("Budget", u, "H1", "HU BUDGET")
    if not all(isinstance(x, str) for x in u.values()) or not {"PK-1", "PK-2", "NK", "H1", "H2", "Z"} <= set(u):
        fehler("Urteil nicht Frage -> String")
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN})
    except (TypeError, ValueError) as ex:
        fehler(f"nicht JSON-serialisierbar: {ex}")
    # --- Wiederaufnahme
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, "seed_9.json").write_text(json.dumps({"seed": 9, "acc": {}, "reell": {}, "sekunden": 0.0, "konfig": "K"}))
        if _seed_arbeiter((9, {}, 1, td, "K")).get("sekunden") != 0.0:
            fehler("Wiederaufnahme liest Datei nicht")
    # --- Mini-Durchlauf aller Arme (synthetisch)
    for arm in ARME:
        acc, reell = arm_grad(arm, xa, ya, xv, yv, 0.05, 1, 30)
        if not (0.0 <= acc <= 1.0) or reell != _info(arm)["reell"]:
            fehler(f"Mini-Lauf {arm}: reell {reell} != {_info(arm)['reell']}")
    return ok


if __name__ == "__main__":
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(gem.speichere("I-039-v2" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
