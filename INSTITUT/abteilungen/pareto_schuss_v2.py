"""Abteilung Information -- I-046-v2 „Pareto Genauigkeit × Information pro Schuss bei GLEICHEM Parameterbudget“
(Quantenstufe Q2: exakte Simulation; Wirkung klassisch erklärbar, formuliert wird „quanten-inspiriert“.)
Erweiterung von I-046 (`abteilungen/pareto_schuss.py`): gleiche Daten (MNIST 16×16, 6.000/5.000), gleiche Messung
(acc_inf mit I-038-Kopf, I_C aus Test-Klassenmitteln), gleicher Kopf, gleiche Verlustfunktion
    L = (1 − λ)·CE(Kopf(p)) − λ·I_C(p)/log2(10),   p = (U ψ)², ψ = x/|x|.

ANLASS: In I-046 hatte die gelernte Orthogonale U 32.640 freie Parameter, die festen Arme id/haar 0. Ein Teil des
  Vorteils kann schlicht „mehr Parameter“ sein. FRAGE: Hält die Front, wenn die gelernte Mischung nur ein kleines,
  strukturiertes Budget hat?
ARME (11): fest id, haar (wie I-046) · gelernt je λ ∈ {0, 0,25, 1} (λ = 0,25 war der Knick der Front):
  voll0/voll25/voll100  U_voll   U = exp(A − Aᵀ), Start Identität (wie I-046, Referenz). Freie Parameter: 256·255/2 = 32.640.
  ff0/ff25/ff100        U_ff     U = H·R3·H·R2·H·R1 (Fastfood/Hadamard-Struktur). H = normierte Walsh-Hadamard-Matrix (fest),
                                 R_l = 128 trainierte 2×2-Drehungen auf festen, seed-eigenen zufälligen Koordinatenpaaren
                                 (die reelle Form der trainierten Diagonalen: eine Drehung ist die orthogonale Verallgemeinerung
                                 der ±1-Diagonale; eine frei trainierte reelle Diagonale wäre nicht orthogonal).
                                 Freie Parameter: 3 × 128 = 384. Start: Winkel gleichverteilt in [0, 2π) (die Struktur kann
                                 die Identität nicht darstellen, daher kein Identitätsstart).
  gi0/gi25/gi100        U_givens U = G3·G2·G1, G_l = 128 Givens-Drehungen auf festen, seed-eigenen zufälligen Paaren (k = 384
                                 Givens-Rotationen, gleiche Paarordnung wie die R_l von U_ff, aber OHNE Hadamard). Freie
                                 Parameter: 384 (= U_ff exakt, also im Rahmen ±5 %). Start: Winkel 0 (Identität).
  Kopf (linear 256 → 10 auf standardisierten p, gemeinsam trainiert, 2.570 Parameter) ist in allen Armen gleich und
  zählt NICHT zum Mischungsbudget. 400 Adam-Schritte Vollbatch. Alle Arme orthogonal (Messung max|UUᵀ − I| je Lauf).
  Lernrate je Arm auf Tuning-Seed 2697 (gittersuche_stabil, Kriterium auf Validierung (1 − λ)·acc + λ·I_C/log2(10)),
  Gitter voll (3e-3, 1e-2, 3e-2) wie I-046, ff/givens (1e-2, 3e-2, 1e-1); Faktor 3 in beide Richtungen verlängerbar.
HYPOTHESEN (Holm über H1, H2, H3; 10 Seeds, exakter Wilcoxon einseitig gepaart, kleinster p = 2⁻¹⁰):
  H1  FF_DOMINIERT_FEST: acc_inf(ff25) > acc_inf(haar) UND I_C(ff25) > I_C(id) (Schnitt, p = max).  H0 = Gegenrichtung
      einer der beiden Teile signifikant.
  H2  FF_NICHT_UNTERLEGEN: ff25 − voll25 bei I_C > −0,05 Bit UND bei acc_inf > −0,5 pp (Nichtunterlegenheit, p = max der
      beiden Verschiebungstests).  H0 = Unterlegenheit jenseits der Grenze in einem der beiden Teile signifikant.
  H3  FF_BESSER_GIVENS: I_C(ff25) > I_C(givens25) (gleiches Budget).  H0 = Gegenrichtung signifikant.
  Je: Ja-Name | H0 | HU.
  B (beschreibend): ganze Front je Arm (acc_inf, I_C, acc bei S = 30/100), Parameterzahlen, Anteil Seeds, in denen ein
      gelernter Arm BEIDE festen dominiert (acc_inf ≥ haar UND I_C ≥ id), Anteil „genauer als haar“/„informativer als id“.
KONTROLLEN: PK acc_inf(id) >= 0,85 (sonst H1 HU). NK: voll25, ff25, gi25 mit permutierten Labels (Training, Kopf, I_C-Messung):
  je Arm Mittel I_C < 0,03 Bit und acc_inf <= 0,15, sonst alles HU. Orthogonalität: max|UUᵀ − I| > 1e-3 (float32; gemessen ≤ 3e-5) in einem beteiligten
  Arm -> dessen Hypothesen HU. Kollaps (acc_inf < Median − 5 pp in einem beteiligten gelernten Arm): HU.
SEEDS: Mess 2640-2649 (10, frisch; auch für die NK), Tuning 2697.
EHRLICHE ERWARTUNG (vor jeder Messzahl): Der Vorteil in I-046 ist zu einem guten Teil „mehr Parameter“. U_ff kann die
  Identität nicht darstellen und startet bei einer Haar-ähnlichen Mischung (I_C ≈ 0,16 Bit); 384 Winkel dürften I_C
  deutlich weniger heben als 32.640 Parameter. Erwartet: H1 eher H0/HU (I_C(ff25) unter id), H2 H0 (voll25 ≈ 0,57 Bit,
  ff25 deutlich darunter), H3 eher H0 (Givens startet bei der Identität mit 0,36 Bit und hat eine lokale Struktur, die zu
  Pixelmerkmalen passt). Möglich ist ein Zwischenergebnis „Front hält nur bei Genauigkeit, nicht bei Information“.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard

from abteilungen.holevo_wirkungsgrad import holm, merkmale, wilcoxon_einseitig, mischung
from abteilungen.pareto_schuss import (ALPHA, KOLLAPS, LOG10, NK_ACC, NK_I, PK_MIN, SCHRITTE, _bereite, _info_torch, messe,
                                       trainiere_U)
from abteilungen.pareto_schuss import _BELEGT as _BELEGT_V1
from abteilungen.struktur_gross import _lade_zwischen, _rechne, _schreibe_zwischen
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil

KARTE = "I-046-v2"
SEEDS = tuple(range(2640, 2650))
TUNING_SEED = 2697
N = 256
LAYER = 3                       # Drehschichten je gelernter Struktur
PAARE = N // 2                  # 128 Drehungen je Schicht
PARAM_FF = PARAM_GI = LAYER * PAARE        # 384
PARAM_VOLL = N * (N - 1) // 2              # 32.640
ORTH_TOL = 1e-3
MARGE_I, MARGE_ACC = 0.05, 0.005           # Nichtunterlegenheit H2: -0,05 Bit, -0,5 pp
LAMS = {"0": 0.0, "25": 0.25, "100": 1.0}
TYPEN = ("voll", "ff", "gi")
GELERNT = {f"{t}{k}": (t, lam) for t in TYPEN for k, lam in LAMS.items()}
FEST = ("id", "haar")
ARME = FEST + tuple(GELERNT)
NK_ARME = ("voll25", "ff25", "gi25")
GITTER = {"voll": (3e-3, 1e-2, 3e-2), "ff": (1e-2, 3e-2, 1e-1), "gi": (1e-2, 3e-2, 1e-1)}
PARAMETER = {"id": 0, "haar": 0, **{a: {"voll": PARAM_VOLL, "ff": PARAM_FF, "gi": PARAM_GI}[t] for a, (t, _) in GELERNT.items()}}
_BELEGT = set(_BELEGT_V1) | set(range(2460, 2470)) | {2479} | set(range(2600, 2630)) | {2698, 2699}

HYPOTHESEN = {
    "H1": "acc_inf(ff25) > acc_inf(haar) UND I_C(ff25) > I_C(id) (p = max, Holm über H1-H3). FF_DOMINIERT_FEST | H0 | HU.",
    "H2": f"ff25 − voll25: I_C > −{MARGE_I} Bit UND acc_inf > −{MARGE_ACC * 100} pp (Nichtunterlegenheit, p = max). "
          "FF_NICHT_UNTERLEGEN | H0 | HU.",
    "H3": "I_C(ff25) > I_C(givens25) bei gleichem Budget. FF_BESSER_GIVENS | H0 | HU.",
    "PK": f"acc_inf(id) >= {PK_MIN} (sonst H1 HU).",
    "NK": f"voll25, ff25, gi25 mit permutierten Labels: je Arm I_C < {NK_I} Bit und acc_inf <= {NK_ACC}.",
    "B": "Front je Arm, Parameterzahl, Anteil dominierender Seeds.",
}


# ------------------------------------------------------------------ Strukturen
def _paare(seed, art, schicht):
    """Feste Koordinatenpaare: Permutation, erste Hälfte = a, zweite = b; inv macht [a|b] wieder zur Koordinatenordnung."""
    p = np.random.default_rng([seed, 46, art, schicht]).permutation(N)
    return torch.tensor(p[:PAARE]), torch.tensor(p[PAARE:]), torch.tensor(np.argsort(p))


def _hadamard():
    return torch.tensor(hadamard(N) / math.sqrt(N), dtype=torch.float32)


def _dreh(z, th, ab):
    a, b, inv = ab
    c, s = torch.cos(th), torch.sin(th)
    za, zb = z[:, a], z[:, b]
    return torch.cat([c * za - s * zb, s * za + c * zb], 1)[:, inv]


def _wende(typ, theta, paare, z):
    """z (Zeilen) -> Zeilen von U·z. ff: H R3 H R2 H R1 · givens: G3 G2 G1."""
    H = _hadamard() if typ == "ff" else None
    for l in range(LAYER):
        z = _dreh(z, theta[l], paare[l])
        if H is not None:
            z = z @ H
    return z


def baue_U(typ, theta, paare):
    return _wende(typ, theta, paare, torch.eye(N)).T      # Zeile i von f(I) = U e_i  ->  f(I) = Uᵀ


def _start(typ, seed):
    if typ == "gi":
        return [torch.zeros(PAARE, requires_grad=True) for _ in range(LAYER)]
    g = torch.Generator().manual_seed(seed + 4600)
    return [(torch.rand(PAARE, generator=g) * 2 * math.pi).requires_grad_(True) for _ in range(LAYER)]


def trainiere_klein(typ, xtr, ytr, lam, lr, seed, schritte=SCHRITTE, mit_param=False):
    """U_ff / U_givens mit gleicher Verlustfunktion wie trainiere_U."""
    art = 3 if typ == "ff" else 2
    paare = [_paare(seed, art, l) for l in range(LAYER)]
    theta = _start(typ, seed)
    W = torch.zeros(N, 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam(theta + [W, b], lr=lr)
    for _ in range(schritte):
        z = _wende(typ, theta, paare, xtr)
        P = z ** 2
        P = P / P.sum(1, keepdim=True)
        f = (P - P.mean(0)) / P.std(0).clamp_min(1e-6)
        ce = F.cross_entropy(f @ W + b, ytr)
        loss = (1 - lam) * ce - lam * _info_torch(P, ytr).float() / LOG10
        if not torch.isfinite(loss):
            return None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        U = baue_U(typ, theta, paare).detach()
    return (U, sum(t.numel() for t in theta)) if mit_param else U


def _U(arm, xtr, ytr, seed, lr, permutiert=False):
    if arm == "id":
        return torch.eye(N)
    if arm == "haar":
        return mischung("haar", seed)
    typ, lam = GELERNT[arm]
    y = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 77))] if permutiert else ytr
    return trainiere_U(xtr, y, lam, lr, seed) if typ == "voll" else trainiere_klein(typ, xtr, y, lam, lr, seed)


def orth_fehler(U):
    return float((U.double() @ U.double().T - torch.eye(N, dtype=torch.float64)).abs().max())


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
    lam = GELERNT[arm][1]
    return {"acc": (1 - lam) * m["acc_inf"] + lam * m["I_C"] / LOG10}


def _tune_schl(a):
    return f"tune|{a[0]}|{a[1]}|{a[2]:.6g}"


def _tune_fn(a):
    return _tune_schl(a), _tune_arbeiter(a)


def tuning(klein, pfad, kennung, prozesse=None, seed=TUNING_SEED):
    """Lernrate je gelerntem Arm; jede Tuning-Aufgabe wird einzeln in der Zwischendatei abgelegt (Wiederaufnahme)."""
    def auswerten(aufg):
        zl = _rechne(aufg, pfad, kennung, fn=_tune_fn, schluessel=_tune_schl, prozesse=prozesse)
        return [zl[_tune_schl(a)] for a in aufg]
    info = {}
    for typ in TYPEN:       # voll zuerst (lang)
        arme = [a for a, (t, _) in GELERNT.items() if t == typ]
        i, _ = gittersuche_stabil(arme, lambda k, lr: [(k, seed, lr, klein)], auswerten, gitter=GITTER[typ], faktor=3,
                                  kollaps=KOLLAPS)
        info.update(i)
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
        return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "instabil": True, "acc_inf": 0.1, "I_C": 0.0,
                                             "orth": 0.0}
    if perm:
        g = lambda y, k: y[torch.randperm(len(y), generator=torch.Generator().manual_seed(s + k))]
        m = messe(U, xtr, g(ytr, 77), xte, g(yte, 78), s, arm)
    else:
        m = messe(U, xtr, ytr, xte, yte, s, arm)
    return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "instabil": False, "orth": orth_fehler(U), **m,
                                         "sekunden": time.time() - t0}


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
    r = {"n": len(seeds), "arme": {}, "p": {}, "p_holm": {}, "diff": {}, "kollaps": {}, "orth": {}, "nk": {}, "beschreibend": {},
         "parameter": dict(PARAMETER)}
    for arm in ARME:
        if arm in d and seeds:
            r["arme"][arm] = {k: float(w(arm, k).mean()) for k in ("acc_inf", "I_C", "acc_S30", "acc_S100")}
            a = w(arm, "acc_inf")
            r["kollaps"][arm] = [s for s, x in zip(seeds, a) if x < np.median(a) - KOLLAPS] if arm in GELERNT else []
            r["orth"][arm] = float(w(arm, "orth").max())
    for arm in NK_ARME:
        nk = [z for z in zeilen if z["teil"] == "nk" and z["arm"] == arm]
        if nk:
            r["nk"][arm] = {"I_C": float(np.mean([z["I_C"] for z in nk])), "acc_inf": float(np.mean([z["acc_inf"] for z in nk]))}
    if len(seeds) >= 2 and all(a in d for a in ("ff25", "haar", "id")):
        a, i = w("ff25", "acc_inf") - w("haar", "acc_inf"), w("ff25", "I_C") - w("id", "I_C")
        r["diff"]["H1"] = {"acc_pp": float(a.mean()) * 100, "I": float(i.mean())}
        r["p"]["H1"] = max(wilcoxon_einseitig(a), wilcoxon_einseitig(i))
        r["p"]["H1_gegen"] = min(wilcoxon_einseitig(-a), wilcoxon_einseitig(-i))
    if len(seeds) >= 2 and all(a in d for a in ("ff25", "voll25")):
        a, i = w("ff25", "acc_inf") - w("voll25", "acc_inf"), w("ff25", "I_C") - w("voll25", "I_C")
        r["diff"]["H2"] = {"acc_pp": float(a.mean()) * 100, "I": float(i.mean())}
        xa, xi = a + MARGE_ACC, i + MARGE_I
        r["p"]["H2"] = max(wilcoxon_einseitig(xa), wilcoxon_einseitig(xi))
        r["p"]["H2_gegen"] = min(wilcoxon_einseitig(-xa), wilcoxon_einseitig(-xi))
    if len(seeds) >= 2 and all(a in d for a in ("ff25", "gi25")):
        x = w("ff25", "I_C") - w("gi25", "I_C")
        r["diff"]["H3"] = {"I": float(x.mean()), "acc_pp": float((w("ff25", "acc_inf") - w("gi25", "acc_inf")).mean()) * 100}
        r["p"]["H3"], r["p"]["H3_gegen"] = wilcoxon_einseitig(x), wilcoxon_einseitig(-x)
    prim = {h: r["p"][h] for h in ("H1", "H2", "H3") if h in r["p"]}
    if prim:
        r["p_holm"] = holm(prim)
    if seeds and all(a in d for a in FEST):
        dom, gen, inf = {}, {}, {}
        for arm in GELERNT:
            if arm in d:
                g = w(arm, "acc_inf") >= w("haar", "acc_inf")
                i = w(arm, "I_C") >= w("id", "I_C")
                dom[arm], gen[arm], inf[arm] = float((g & i).mean()), float(g.mean()), float(i.mean())
        r["beschreibend"] = {"dominiert_beide_festen_anteil_seeds": dom, "genauer_als_haar_anteil": gen,
                             "informativer_als_id_anteil": inf}
    return r


def _nk(r):
    ok = all(a in r["nk"] and r["nk"][a]["I_C"] < NK_I and r["nk"][a]["acc_inf"] <= NK_ACC for a in NK_ARME)
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _pk(r):
    return "BESTANDEN" if "id" in r["arme"] and r["arme"]["id"]["acc_inf"] >= PK_MIN else "DURCHGEFALLEN"


def _zweig(r, h, ja, arme, nk, pk="BESTANDEN"):
    if h not in r["p"]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    if h == "H1" and pk != "BESTANDEN":
        return "HU (Positivkontrolle durchgefallen)"
    schlecht = [a for a in arme if r["orth"].get(a, 1.0) > ORTH_TOL]
    if schlecht:
        return f"HU (Orthogonalitaet verletzt: {schlecht})"
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
    u["H1"] = _zweig(r, "H1", "FF_DOMINIERT_FEST", ("ff25",), nk, pk)
    u["H2"] = _zweig(r, "H2", "FF_NICHT_UNTERLEGEN", ("ff25", "voll25"), nk)
    u["H3"] = _zweig(r, "H3", "FF_BESSER_GIVENS", ("ff25", "gi25"), nk)
    u["B"] = "beschreibend: (Parameter; acc_inf %, I_C Bit) " + "; ".join(
        f"{a} ({r['parameter'][a]}; {v['acc_inf'] * 100:.2f}, {v['I_C']:.3f})".replace(".", ",") for a, v in r["arme"].items()) + \
        f"; dominiert beide festen: {r['beschreibend'].get('dominiert_beide_festen_anteil_seeds')}"
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, n=10, off=None, sd=0.002, nk=(0.01, 0.10), koll=None, orth=0.0):
    base = {"id": (0.900, 0.36), "haar": (0.938, 0.16)}
    for t, (a0, i0) in {"voll": (0.962, 0.20), "ff": (0.950, 0.45), "gi": (0.940, 0.40)}.items():
        base[t + "0"] = (a0, i0)
        base[t + "25"] = (a0 - 0.011, i0 + 0.20)
        base[t + "100"] = (a0 - 0.012, i0 + 0.30)
    base["ff25"] = (0.955, 0.58)
    base["voll25"] = (0.952, 0.57)
    base["gi25"] = (0.945, 0.45)
    base.update(off or {})
    zl = []
    for i in range(n):
        for a, (acc, ic) in base.items():
            x = acc + rng.normal(0, sd) - (0.2 if koll == a and i == 0 else 0)
            zl.append({"teil": "t", "arm": a, "seed": 2640 + i, "acc_inf": x, "I_C": ic + rng.normal(0, sd * 5),
                       "acc_S30": x - 0.2, "acc_S100": x - 0.1, "orth": orth if a == "ff25" else 1e-6})
        for a in NK_ARME:
            zl.append({"teil": "nk", "arm": a, "seed": 2640 + i, "acc_inf": nk[1], "I_C": nk[0]})
    return zl


def vortest() -> bool:
    ok = True
    rng = np.random.default_rng(461)

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-046-v2): {name}", flush=True)
            ok = False
    pruefe("Seeds frisch", not (set(SEEDS) & _BELEGT) and TUNING_SEED not in _BELEGT and len(SEEDS) == 10
           and TUNING_SEED not in SEEDS)
    # Budget
    pruefe("Budget ff == givens == 384", PARAM_FF == PARAM_GI == 384 and abs(PARAM_GI / PARAM_FF - 1) <= 0.05)
    pruefe("Budget voll 32640", PARAM_VOLL == 32640)
    # Orthogonalität, Parameterzählung, Start, Verlust/Gradienten
    x = torch.rand(60, N)
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.arange(60) % 10
    for typ in ("ff", "gi"):
        for lam in (0.0, 0.5, 1.0):
            U, n = trainiere_klein(typ, x, y, lam, 3e-2, 0, schritte=5, mit_param=True)
            pruefe(f"{typ} lam {lam} orthogonal (max {orth_fehler(U):.2e})", orth_fehler(U) < ORTH_TOL)
            pruefe(f"{typ} Parameterzahl {n}", n == 384)
    Uv = trainiere_U(x, y, 0.5, 1e-2, 0, schritte=3)
    pruefe("voll orthogonal", orth_fehler(Uv) < ORTH_TOL)
    pruefe("givens Start = Identität", torch.allclose(trainiere_klein("gi", x, y, 0.5, 1e-2, 0, schritte=0), torch.eye(N), atol=1e-6))
    Uf0 = trainiere_klein("ff", x, y, 0.5, 1e-2, 0, schritte=0)
    pruefe("ff Start orthogonal, nicht Identität", orth_fehler(Uf0) < ORTH_TOL and not torch.allclose(Uf0, torch.eye(N), atol=0.1))
    pruefe("U Wirkung = Matrixprodukt", torch.allclose(_wende("ff", [torch.zeros(PAARE)] * LAYER, [_paare(0, 3, l) for l in range(LAYER)], x),
                                                          x @ baue_U("ff", [torch.zeros(PAARE)] * LAYER, [_paare(0, 3, l) for l in range(LAYER)]).T,
                                                          atol=1e-5))
    pruefe("Paare disjunkt, vollständig", all(set(a.tolist()) | set(b.tolist()) == set(range(N)) and not set(a.tolist()) & set(b.tolist())
                                               for a, b, _ in (_paare(3, 3, l) for l in range(LAYER))))
    # Gradienten: Verlust endlich, Gradient auf allen Parametern ungleich 0
    for typ in ("ff", "gi"):
        paare = [_paare(0, 3 if typ == "ff" else 2, l) for l in range(LAYER)]
        theta = _start(typ, 0)
        if typ == "gi":
            theta = [(t.detach() + 0.1).requires_grad_(True) for t in theta]
        P = _wende(typ, theta, paare, x) ** 2
        P = P / P.sum(1, keepdim=True)
        loss = -_info_torch(P, y).float() / LOG10 + 0.5 * P.std(0).mean()
        loss.backward()
        pruefe(f"{typ} Verlust endlich, Gradient != 0", bool(torch.isfinite(loss)) and all(float(t.grad.abs().sum()) > 0 for t in theta))
    # Lernen wirkt: I_C steigt unter lam = 1
    xx = torch.rand(300, N) ** 2
    xx = xx / xx.norm(dim=1, keepdim=True)
    yy = (xx[:, :20].sum(1).argsort().argsort() * 10 // 300).long()
    for typ in ("ff", "gi"):
        i0 = float(_info_torch(merkmale(xx, trainiere_klein(typ, xx, yy, 1.0, 3e-2, 0, schritte=0)).double(), yy))
        i1 = float(_info_torch(merkmale(xx, trainiere_klein(typ, xx, yy, 1.0, 3e-2, 0, schritte=60)).double(), yy))
        pruefe(f"{typ}: lam=1 erhöht I_C ({i0:.3f} -> {i1:.3f})", i1 > i0)
    # Zweige
    u = lambda **kw: urteil(auswertung(_synth(rng, **kw)))
    gesehen = set()

    def erwarte(name, ur, **soll):
        nonlocal ok
        for k, wert in soll.items():
            gesehen.add((k, wert.split(" (")[0]))
            if not ur[k].startswith(wert):
                print(f"  Vortest FEHLER (I-046-v2) {name}: {k} soll {wert}, ist {ur[k]}", flush=True)
                ok = False
    erwarte("Normalfall", u(), NK="BESTANDEN", PK="BESTANDEN", H1="FF_DOMINIERT_FEST", H2="FF_NICHT_UNTERLEGEN", H3="FF_BESSER_GIVENS")
    erwarte("Gegenrichtung", u(off={"ff25": (0.90, 0.30), "voll25": (0.95, 0.60), "gi25": (0.94, 0.50)}),
            H1="H0", H2="H0", H3="H0")
    # H2: nur Information unterlegen (acc gleich) -> H0 ; nur knapp über Grenze -> HU
    erwarte("H2 Info unterlegen", u(off={"ff25": (0.952, 0.40), "voll25": (0.952, 0.57)}), H2="H0")
    erwarte("H2 acc unterlegen", u(off={"ff25": (0.930, 0.57), "voll25": (0.952, 0.57)}), H2="H0")
    erwarte("H2 knapp", u(sd=0.02, off={"ff25": (0.952, 0.57), "voll25": (0.952, 0.57)}), H2="HU")
    erwarte("Rauschen", u(sd=0.05, off={"ff25": (0.92, 0.5), "id": (0.92, 0.5), "haar": (0.92, 0.5), "voll25": (0.92, 0.5),
                                         "gi25": (0.92, 0.5)}), H1="HU", H2="HU", H3="HU")
    erwarte("NK fällt", u(nk=(0.2, 0.5)), NK="DURCHGEFALLEN", H1="HU (Negativkontrolle", H3="HU (Negativkontrolle")
    erwarte("PK fällt", u(off={"id": (0.80, 0.36)}), PK="DURCHGEFALLEN", H1="HU (Positivkontrolle", H3="FF_BESSER_GIVENS")
    erwarte("Kollaps ff25", u(koll="ff25"), H1="HU (Instabilitaet", H3="HU (Instabilitaet", H2="HU (Instabilitaet")
    erwarte("Orthogonalität", u(orth=1e-2), H1="HU (Orthogonalitaet", H2="HU (Orthogonalitaet", H3="HU (Orthogonalitaet")
    erwarte("Arm fehlt", urteil(auswertung([z for z in _synth(rng) if not (z["arm"] == "gi25" and z["teil"] == "t")])), H3="HU (Arme fehlen", H1="FF_DOMINIERT_FEST")
    for h, ja in (("H1", "FF_DOMINIERT_FEST"), ("H2", "FF_NICHT_UNTERLEGEN"), ("H3", "FF_BESSER_GIVENS")):
        pruefe(f"Zweige {h}", {ja, "H0", "HU"} <= {wert for k, wert in gesehen if k == h})
    # Wiederaufnahme je Aufgabe (Tuning und Einheiten teilen _rechne)
    import tempfile
    global _ZAEHL
    _ZAEHL = []
    with tempfile.TemporaryDirectory() as td:
        pf, ken = Path(td) / "z.json", {"karte": "test"}
        aufg = [("t", "ff25", s, 0.1, True) for s in range(4)]
        _rechne(aufg[:2], pf, ken, fn=_dummy, schluessel=_schl, prozesse=1)
        pruefe("Wiederaufnahme: Teilstand", len(_ZAEHL) == 2)
        _rechne(aufg, pf, ken, fn=_dummy, schluessel=_schl, prozesse=1)
        pruefe("Wiederaufnahme: nur Rest gerechnet", len(_ZAEHL) == 4 and sorted(_ZAEHL) == [0, 1, 2, 3])
        _rechne(aufg, pf, ken, fn=_dummy, schluessel=_schl, prozesse=1)
        pruefe("Wiederaufnahme: fertig -> nichts", len(_ZAEHL) == 4)
        try:
            _rechne(aufg, pf, {"karte": "anders"}, fn=_dummy, schluessel=_schl, prozesse=1)
            pruefe("Zwischendatei anderer Einstellungen wird abgelehnt", False)
        except ValueError:
            pass
    return ok


_ZAEHL: list = []


def _dummy(a):
    _ZAEHL.append(a[2])
    return _schl(a), {"teil": a[0], "arm": a[1], "seed": a[2]}


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, **_):
    if schnell:          # Rauchtest NUR Tuning-Seed, kleine Daten; Ergebnis wird verworfen
        seeds, klein = (TUNING_SEED,), True
    else:
        seeds, klein = SEEDS, False
        if set(seeds) & _BELEGT:
            raise ValueError("Seeds nicht frisch")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "seeds": list(seeds), "schritte": SCHRITTE, "klein": klein, "tuning_seed": TUNING_SEED,
               "arme": list(ARME)}
    t0 = time.time()
    z = _lade_zwischen(pfad, kennung)
    if "tuning|fertig" not in z:
        info = tuning(klein, pfad, kennung, prozesse)
        z = _lade_zwischen(pfad, kennung)
        z["tuning|fertig"] = info
        _schreibe_zwischen(pfad, kennung, z)
    info = z["tuning|fertig"]
    lr = {a: info[a]["lr"] for a in GELERNT}
    aufg = [("t", a, s, lr.get(a, 0.0), klein) for a in GELERNT for s in seeds] + \
           [("nk", a, s, lr[a], klein) for a in NK_ARME for s in seeds] + \
           [("t", a, s, 0.0, klein) for a in FEST for s in seeds]
    aufg.sort(key=lambda a: GELERNT.get(a[1], ("",))[0] != "voll")       # lange Aufgaben (voll) zuerst; sort ist stabil
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [v for k, v in alle.items() if k.startswith("einheit|")]
    r = auswertung(zeilen)
    e = {"abteilung": "Information", "modul": "pareto_schuss_v2", "karte": KARTE, "version": 1, "seeds": list(seeds),
         "tuning_seed": TUNING_SEED, "lernraten": lr, "tuning": info, "schritte": SCHRITTE, "parameter": dict(PARAMETER),
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0, "auswertung": r}
    e["urteil"] = urteil(r)
    return e
