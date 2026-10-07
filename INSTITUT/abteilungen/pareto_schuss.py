"""Abteilung Information -- I-046 „Pareto-Front Genauigkeit × Information pro Schuss“ (Quantenstufe Q2: exakte Simulation).
(Folge von I-040-v2 H4: feste dichte Mischung (Haar) verbessert die exakte Genauigkeit, verschlechtert die Information pro Schuss.
 Literatur (recherche/2026-10-07_volltext_2.md, Teil B): Sbaiz u. a. 2021 trainieren eine Unitäre nur auf ein Ziel
 (Einzelschuss), Hu u. a. 2023 trainieren die Schaltung nicht; eine Pareto-Front mit λ-Reihe wurde nicht gefunden.
 Urteil dort: neue Verbindung. Wirkung klassisch erklärbar; formuliert wird „quanten-inspiriert“.)

FRAGE: Ist der Zielkonflikt trainierbar? Liefert eine gelernte orthogonale Mischung U, trainiert auf
    L(λ) = (1 − λ) · CE(Kopf(p)) − λ · I_C(p) / log2(10),   p = (U ψ)², ψ = x/|x|,
  eine Front, die beide festen Mischungen (Identität: beste Information, Haar: beste Genauigkeit) übertrifft?
ARME (MNIST 16×16, 256 Ausgänge, Daten wie I-040-v2: 6.000 / 5.000):
  id (fest), haar (fest, Haar-zufällig je Seed), L0, L25, L50, L75, L90, L100 (gelernt, λ = 0 … 1).
  U = matrix_exp(A − Aᵀ), Start A = 0 (also U = Identität), Kopf linear 256 → 10 auf standardisierten p, gemeinsam trainiert,
  Adam Vollbatch, SCHRITTE Schritte. Lernrate je λ auf Tuning-Seed 2479 (gittersuche_stabil, Kriterium auf Validierung =
  letzte 10 % Training: s = (1 − λ)·acc + λ·I_C/log2(10)).
MESSUNG je Arm (nach dem Training, U fest): acc_inf = I-038-Kopf (Softmax-Regression, L2 3e-3, L-BFGS 500) auf p,
  I_C = I(Klasse; 1 Schuss) aus Test-Klassenmitteln (wie I-040-v2), beschreibend acc bei S = 30 / 100 Schüssen (exakter Kopf).
HYPOTHESEN (Holm über H1, H2a, H2b; 10 Seeds, einseitiger exakter Wilcoxon, kleinster p = 2⁻¹⁰):
  H1  ZIELKONFLIKT_TRAINIERBAR: I_C(L100) > I_C(L0) UND acc_inf(L0) > acc_inf(L100) (Schnitt-Vereinigung, p = max).
  H2a INFO_UEBER_FEST: I_C(L100) > I_C(id) (gelernt übertrifft die beste feste Information).
  H2b GENAUIGKEIT_UEBER_FEST: acc_inf(L0) > acc_inf(haar) (gelernt übertrifft die beste feste Genauigkeit).
  Je: Ja-Zweig | H0 (Gegenrichtung signifikant) | HU.
  B: Front (acc_inf, I_C) aller Arme, Dominanz eines gelernten Arms über BEIDE festen (Anteil Seeds), λ-Monotonie.
KONTROLLEN: PK: acc_inf(id) >= 0,85 (I-040-v2: ≈ 0,90). NK: L50 auf permutierten Labels (Training, Kopf und I_C-Messung
  permutiert): I_C < 0,03 Bit und acc_inf <= 0,15. NK fällt: alles HU; PK fällt: H2b HU. Kollaps (acc_inf < Median − 5 pp in einem gelernten Arm): dessen
  Hypothesen HU.
SEEDS: Mess 2460-2469 (10, frisch), Tuning 2479.
EHRLICHE ERWARTUNG: H1 ja (der Strafterm wirkt). H2a offen (die Identität ist schon gut; I-040-v2 Suche < id). H2b eher ja
  (gelernte Orthogonale ≈ Cayley +1,5 pp in v3). Ob ein Zwischenarm beide festen dominiert, ist die eigentliche Neuigkeit.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from abteilungen.holevo_wirkungsgrad import (daten, holm, info_C, klassenmittel, kopf_acc, kopf_fit, merkmale, mischung,
                                             schuesse, wilcoxon_einseitig, _gen)
from abteilungen.struktur_gross import _lade_zwischen, _rechne, _schreibe_zwischen
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil, parallel

KARTE = "I-046"
SEEDS = tuple(range(2460, 2470))
TUNING_SEED = 2479
LAMBDAS = {"L0": 0.0, "L25": 0.25, "L50": 0.5, "L75": 0.75, "L90": 0.9, "L100": 1.0}
FEST = ("id", "haar")
ARME = FEST + tuple(LAMBDAS)
SCHRITTE = 400
LR_GITTER = (3e-3, 1e-2, 3e-2)
L2_EXAKT, ITER_EXAKT = 3e-3, 500
ALPHA = 0.05
PK_MIN = 0.85
NK_I, NK_ACC = 0.03, 0.15
KOLLAPS = 0.05
LOG10 = math.log2(10)
_BELEGT = set(range(900, 1001)) | set(range(2000, 2070)) | set(range(2090, 2120)) | set(range(2150, 2160)) | set(range(2200, 2205)) \
    | set(range(2300, 2325)) | set(range(2330, 2345)) | set(range(2350, 2358)) | set(range(2410, 2420)) | {2198, 2299, 2399, 2498}

HYPOTHESEN = {
    "H1": "I_C(L100) > I_C(L0) UND acc_inf(L0) > acc_inf(L100) (exakter Wilcoxon einseitig, p = max; Holm über H1, H2a, H2b). "
          "ZIELKONFLIKT_TRAINIERBAR | H0 | HU.",
    "H2a": "I_C(L100) > I_C(id). INFO_UEBER_FEST | H0 | HU.",
    "H2b": "acc_inf(L0) > acc_inf(haar). GENAUIGKEIT_UEBER_FEST | H0 | HU.",
    "PK": f"acc_inf(id) >= {PK_MIN}.",
    "NK": f"L50 mit permutierten Labels: I_C < {NK_I} Bit und acc_inf <= {NK_ACC}.",
    "B": "Front, Dominanz gelernter Arme über beide festen, λ-Monotonie, Schussgenauigkeit S = 30/100.",
}


# ------------------------------------------------------------------ Training der Mischung
def _info_torch(P, y, K=10):
    """Differenzierbares I(Klasse; 1 Schuss) in Bit aus Klassenmitteln von P (Zeilen Summe 1)."""
    oh = F.one_hot(y, K).double()
    n = oh.sum(0)
    q = (oh.T @ P.double()) / n[:, None]
    pi = n / n.sum()
    Hf = lambda p: -(p * torch.log2(p.clamp_min(1e-30))).sum(-1)
    return Hf(pi @ q) - (pi * Hf(q)).sum()


def trainiere_U(xtr, ytr, lam, lr, seed, schritte=SCHRITTE):
    torch.manual_seed(seed)
    A = torch.zeros(256, 256, requires_grad=True)
    W = torch.zeros(256, 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([A, W, b], lr=lr)
    for _ in range(schritte):
        U = torch.linalg.matrix_exp(A - A.T)
        P = (xtr @ U.T) ** 2
        P = P / P.sum(1, keepdim=True)
        f = (P - P.mean(0)) / P.std(0).clamp_min(1e-6)
        ce = F.cross_entropy(f @ W + b, ytr)
        loss = (1 - lam) * ce - lam * _info_torch(P, ytr).float() / LOG10
        if not torch.isfinite(loss):
            return None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return torch.linalg.matrix_exp(A - A.T).detach()


def _U(arm, xtr, ytr, seed, lr, permutiert=False):
    if arm == "id":
        return torch.eye(256)
    if arm == "haar":
        return mischung("haar", seed)
    y = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 77))] if permutiert else ytr
    return trainiere_U(xtr, y, LAMBDAS[arm], lr, seed)


def messe(U, xtr, ytr, xte, yte, seed, arm, train_y=None):
    Ptr, Pte = merkmale(xtr, U), merkmale(xte, U)
    yk = ytr if train_y is None else train_y
    h = kopf_fit(Ptr, yk, L2_EXAKT, max_iter=ITER_EXAKT)
    pi = np.bincount(ytr.numpy(), minlength=10) / len(ytr)
    out = {"acc_inf": kopf_acc(h, Pte, yte), "I_C": info_C(klassenmittel(Pte.numpy().astype(np.float64), yte.numpy()), pi)}
    sch = schuesse(Pte, (30, 100), _gen(KARTE, seed, arm))
    for S, Pq in sch.items():
        out[f"acc_S{S}"] = kopf_acc(h, Pq, yte)
    return out


def _bereite(seed, klein):
    xtr, ytr, xte, yte = daten("mnist", seed, klein)
    n = lambda x: x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return n(xtr.float()), ytr.long(), n(xte.float()), yte.long()


# ------------------------------------------------------------------ Arbeiter
def _tune_arbeiter(a):
    arm, s, lr, klein = a
    torch.set_num_threads(1)
    xtr, ytr, _, _ = _bereite(s, klein)
    nv = len(ytr) // 10
    xa, ya, xv, yv = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
    U = _U(arm, xa, ya, s, lr)
    if U is None:
        return {"acc": 0.0}
    m = messe(U, xa, ya, xv, yv, s, arm)
    lam = LAMBDAS[arm]
    return {"acc": (1 - lam) * m["acc_inf"] + lam * m["I_C"] / LOG10}


def tuning(klein):
    lern = list(LAMBDAS)
    info, _ = gittersuche_stabil(lern, lambda k, lr: [(k, TUNING_SEED, lr, klein)], lambda aufg: parallel(_tune_arbeiter, aufg),
                                 gitter=LR_GITTER, faktor=3, kollaps=KOLLAPS)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}}
            for k, v in info.items()}


def _arbeiter(a):
    teil, arm, s, lr, klein = a
    torch.set_num_threads(1)
    t0 = time.time()
    xtr, ytr, xte, yte = _bereite(s, klein)
    perm = teil == "nk"
    U = _U(arm, xtr, ytr, s, lr, permutiert=perm)
    if U is None:
        return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "instabil": True, "acc_inf": 0.1, "I_C": 0.0}
    if perm:   # NK: Training UND Messung (Kopf, I_C) auf permutierten Labels -> I_C ~ 0, acc ~ 0,1
        g = lambda y, k: y[torch.randperm(len(y), generator=torch.Generator().manual_seed(s + k))]
        m = messe(U, xtr, g(ytr, 77), xte, g(yte, 78), s, arm)
    else:
        m = messe(U, xtr, ytr, xte, yte, s, arm)
    return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "instabil": False, **m, "sekunden": time.time() - t0}


def _schl(a):
    return f"einheit|{a[0]}|{a[1]}|{a[2]}"


# ------------------------------------------------------------------ Auswertung und Urteile
def auswertung(zeilen: list) -> dict:
    d = {}
    for z in zeilen:
        if z["teil"] == "t":
            d.setdefault(z["arm"], {})[z["seed"]] = z
    seeds = sorted(set.intersection(*[set(v) for v in d.values()])) if d else []
    w = lambda arm, k: np.array([d[arm][s][k] for s in seeds], float) if arm in d else None
    r = {"n": len(seeds), "arme": {}, "p": {}, "p_holm": {}, "diff": {}, "kollaps": {}, "nk": {}, "beschreibend": {}}
    for arm in ARME:
        if arm in d and seeds:
            r["arme"][arm] = {k: float(w(arm, k).mean()) for k in ("acc_inf", "I_C", "acc_S30", "acc_S100")}
            a = w(arm, "acc_inf")
            r["kollaps"][arm] = [s for s, x in zip(seeds, a) if x < np.median(a) - KOLLAPS] if arm in LAMBDAS else []
    nk = [z for z in zeilen if z["teil"] == "nk"]
    if nk:
        r["nk"] = {"I_C": float(np.mean([z["I_C"] for z in nk])), "acc_inf": float(np.mean([z["acc_inf"] for z in nk]))}
    if len(seeds) >= 2 and all(a in d for a in ("L0", "L100")):
        a1, a2 = w("L100", "I_C") - w("L0", "I_C"), w("L0", "acc_inf") - w("L100", "acc_inf")
        r["diff"]["H1"] = {"I": float(a1.mean()), "acc_pp": float(a2.mean()) * 100}
        r["p"]["H1"] = max(wilcoxon_einseitig(a1), wilcoxon_einseitig(a2))
        r["p"]["H1_gegen"] = min(wilcoxon_einseitig(-a1), wilcoxon_einseitig(-a2))
    if len(seeds) >= 2 and all(a in d for a in ("L100", "id")):
        x = w("L100", "I_C") - w("id", "I_C")
        r["diff"]["H2a"] = {"I": float(x.mean())}
        r["p"]["H2a"], r["p"]["H2a_gegen"] = wilcoxon_einseitig(x), wilcoxon_einseitig(-x)
    if len(seeds) >= 2 and all(a in d for a in ("L0", "haar")):
        x = w("L0", "acc_inf") - w("haar", "acc_inf")
        r["diff"]["H2b"] = {"acc_pp": float(x.mean()) * 100}
        r["p"]["H2b"], r["p"]["H2b_gegen"] = wilcoxon_einseitig(x), wilcoxon_einseitig(-x)
    prim = {h: r["p"][h] for h in ("H1", "H2a", "H2b") if h in r["p"]}
    if prim:
        r["p_holm"] = holm(prim)
    if seeds and all(a in d for a in FEST):
        dom = {}
        for arm in LAMBDAS:
            if arm in d:
                ok = (w(arm, "acc_inf") >= w("haar", "acc_inf")) & (w(arm, "I_C") >= w("id", "I_C"))
                dom[arm] = float(ok.mean())
        r["beschreibend"]["dominiert_beide_festen_anteil_seeds"] = dom
    return r


def _nk(r):
    n = r["nk"]
    return "BESTANDEN" if n and n["I_C"] < NK_I and n["acc_inf"] <= NK_ACC else "DURCHGEFALLEN"


def _pk(r):
    return "BESTANDEN" if "id" in r["arme"] and r["arme"]["id"]["acc_inf"] >= PK_MIN else "DURCHGEFALLEN"


def _zweig(r, h, ja, arme, nk, pk_noetig=False, pk="BESTANDEN"):
    if h not in r["p"]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    if pk_noetig and pk != "BESTANDEN":
        return "HU (Positivkontrolle durchgefallen)"
    k = sorted({s for a in arme for s in r["kollaps"].get(a, [])})
    if k:
        return f"HU (Instabilitaet/Kollaps in Seeds {k})"
    z = f"{r['diff'][h]}, n {r['n']}, p_Holm {r['p_holm'][h]:.3g}"
    if r["p_holm"][h] < ALPHA:
        return f"{ja} ({z})"
    if r["p"][f"{h}_gegen"] < ALPHA:
        return f"H0 ({z})"
    return f"HU ({z})"


def urteil(r: dict) -> dict:
    nk, pk = _nk(r), _pk(r)
    u = {"NK": f"{nk} ({r['nk']})", "PK": f"{pk} (acc_inf id {r['arme'].get('id', {}).get('acc_inf', float('nan')):.4f})"}
    u["H1"] = _zweig(r, "H1", "ZIELKONFLIKT_TRAINIERBAR", ("L0", "L100"), nk)
    u["H2a"] = _zweig(r, "H2a", "INFO_UEBER_FEST", ("L100",), nk)
    u["H2b"] = _zweig(r, "H2b", "GENAUIGKEIT_UEBER_FEST", ("L0",), nk, pk_noetig=True, pk=pk)
    u["B"] = "beschreibend: (acc_inf %, I_C Bit) " + "; ".join(
        f"{a} ({v['acc_inf'] * 100:.2f}, {v['I_C']:.3f})".replace(".", ",") for a, v in r["arme"].items()) + \
        f"; dominiert beide festen: {r['beschreibend'].get('dominiert_beide_festen_anteil_seeds')}"
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, n=10, off=None, sd=0.002, nk=(0.01, 0.10), koll=None):
    base = {"id": (0.900, 0.70), "haar": (0.935, 0.50), "L0": (0.950, 0.45), "L25": (0.948, 0.60), "L50": (0.940, 0.70),
            "L75": (0.930, 0.78), "L90": (0.915, 0.82), "L100": (0.880, 0.85)}
    base.update(off or {})
    zl = []
    for i in range(n):
        for a, (acc, ic) in base.items():
            x = acc + rng.normal(0, sd) - (0.2 if koll == a and i == 0 else 0)
            zl.append({"teil": "t", "arm": a, "seed": 2460 + i, "acc_inf": x, "I_C": ic + rng.normal(0, sd * 5),
                       "acc_S30": x - 0.2, "acc_S100": x - 0.1})
        zl.append({"teil": "nk", "arm": "L50", "seed": 2460 + i, "acc_inf": nk[1], "I_C": nk[0], "acc_S30": 0.1, "acc_S100": 0.1})
    return zl


def vortest() -> bool:
    ok = True
    rng = np.random.default_rng(46)

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-046): {name}", flush=True)
            ok = False
    pruefe("Seeds frisch", not (set(SEEDS) & _BELEGT) and TUNING_SEED not in _BELEGT and len(SEEDS) == 10)
    # Bausteine
    x = torch.rand(50, 256)
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.arange(50) % 10
    U = trainiere_U(x, y, 0.5, 1e-2, 0, schritte=3)
    pruefe("U orthogonal", U is not None and torch.allclose(U @ U.T, torch.eye(256), atol=1e-4))
    pruefe("U Start = Identität", torch.allclose(trainiere_U(x, y, 0.5, 1e-2, 0, schritte=0), torch.eye(256)))
    P = merkmale(x, torch.eye(256)).double()
    pi = np.bincount(y.numpy(), minlength=10) / len(y)
    pruefe("I differenzierbar = info_C", abs(float(_info_torch(P, y)) - info_C(klassenmittel(P.numpy(), y.numpy()), pi)) < 1e-9)
    # Zweige
    u = lambda **kw: urteil(auswertung(_synth(rng, **kw)))
    gesehen = set()

    def erwarte(name, ur, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not ur[k].startswith(w):
                print(f"  Vortest FEHLER (I-046) {name}: {k} soll {w}, ist {ur[k]}", flush=True)
                ok = False
    erwarte("Normalfall", u(), NK="BESTANDEN", PK="BESTANDEN", H1="ZIELKONFLIKT_TRAINIERBAR", H2a="INFO_UEBER_FEST",
            H2b="GENAUIGKEIT_UEBER_FEST")
    erwarte("Gegenrichtung", u(off={"L0": (0.87, 0.90), "L100": (0.95, 0.40), "haar": (0.96, 0.5), "id": (0.90, 0.95)}),
            H1="H0", H2a="H0", H2b="H0")
    erwarte("Rauschen", u(sd=0.05, off={"L0": (0.92, 0.70), "L100": (0.92, 0.70), "haar": (0.92, 0.7), "id": (0.92, 0.7)}),
            H1="HU", H2a="HU", H2b="HU")
    erwarte("NK fällt", u(nk=(0.2, 0.5)), NK="DURCHGEFALLEN", H1="HU (Negativkontrolle")
    erwarte("PK fällt", u(off={"id": (0.80, 0.70)}), PK="DURCHGEFALLEN", H2b="HU (Positivkontrolle", H1="ZIELKONFLIKT_TRAINIERBAR")
    erwarte("Kollaps L0", u(koll="L0"), H1="HU (Instabilitaet", H2a="INFO_UEBER_FEST")
    erwarte("Arm fehlt", urteil(auswertung([z for z in _synth(rng) if z["arm"] != "haar"])), H2b="HU (Arme fehlen")
    for h, ja in (("H1", "ZIELKONFLIKT_TRAINIERBAR"), ("H2a", "INFO_UEBER_FEST"), ("H2b", "GENAUIGKEIT_UEBER_FEST")):
        pruefe(f"Zweige {h}", {ja, "H0", "HU"} <= {w for k, w in gesehen if k == h})
    return ok


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, **_):
    if schnell:          # Rauchtest NUR Tuning-Seed, kleine Daten
        seeds, klein = (TUNING_SEED,), True
    else:
        seeds, klein = SEEDS, False
        if set(seeds) & _BELEGT:
            raise ValueError("Seeds nicht frisch")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "seeds": list(seeds), "schritte": SCHRITTE, "klein": klein}
    t0 = time.time()
    z = _lade_zwischen(pfad, kennung)
    if "tuning" not in z:
        z["tuning"] = tuning(klein)
        _schreibe_zwischen(pfad, kennung, z)
    lr = {a: z["tuning"][a]["lr"] for a in LAMBDAS}
    aufg = [("t", a, s, lr.get(a, 0.0), klein) for a in ARME for s in seeds] + [("nk", "L50", s, lr["L50"], klein) for s in seeds]
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [v for k, v in alle.items() if k.startswith("einheit|")]
    r = auswertung(zeilen)
    e = {"abteilung": "Information", "modul": "pareto_schuss", "karte": KARTE, "version": 1, "seeds": list(seeds),
         "tuning_seed": TUNING_SEED, "lernraten": lr, "tuning": z["tuning"], "schritte": SCHRITTE, "zwischendatei": str(pfad),
         "zeilen": zeilen, "sekunden": time.time() - t0, "auswertung": r}
    e["urteil"] = urteil(r)
    return e
