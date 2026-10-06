"""Abteilung Struktur -- I-038 „Struktur-Aequivalenz“: ist QIRE gleichwertig zu klassischer dichter Mischung plus Quadrat?
(Quantenstufe Q2: exakte Simulation; die Frage selbst ist klassisch. Plan: PLAN_2026-10-06_gesamt.md §2 P1;
Vorlage fuer Arme/Hyperparameter: recherche/2026-10-05_vertiefung/v3_kontrollen.md und zwischenstand/v3/kontrollen.py.)

HEILMEIER:
  Was wollen wir?  Pflichtbefund fuer die schriftliche Arbeit: Traegt die Quantenstruktur der QIRE-Zelle (gelernte
      Winkel, CNOT, Clifford-Suche) etwas, das eine feste zufaellige/klassische dichte Mischung plus Quadrat nicht auch
      leistet? v3 (Erkundung, 5 Seeds): nein. Hier vorregistriert mit 20 Seeds, TOST, Holm, Negativ- und Positivkontrolle.
  Wie heute?  Erkundung v3: Haar² 0,938/0,834, QIRE 0,938/0,837, QIRE-Zufallswinkel 0,937/0,834, Fastfood-2 0,938/0,835,
      Pixel-Softmax 0,910/0,818.
  Was ist neu?  Vorregistrierte Gleichwertigkeit (TOST ±0,5 pp), gepaarte Clifford-Suche, Zufallslabel-Negativkontrolle,
      und (Erweiterung des Direktors) klassische Nichtlinearitaeten (ReLU, tanh, |z|, Quadrat ohne Normierung, MLP) als
      staerkste Gegenkandidaten zum Quadrat (H-G).

AUFBAU (alle Arme bis auf Pixel-Softmax/MLP: x -> x/‖x‖ -> z = U x -> Merkmal f(z) -> DERSELBE Kopf kopf_logistisch,
  Softmax-Regression, standardisierte Merkmale, L2 = 3e-3, L-BFGS, 2.570 Parameter). 16x16, 6.000 Training/5.000 Test je Seed.
ARME: pix_softmax, ident_sq, haar_sq, haar_lin, fastfood2 ((H D2 H D1)²), qire_zufall (L=4, Zufallswinkel, ungelernt),
  qire (gelernt), qire_nocnot (gelernt), suche (Clifford-Suche, NUR Seeds 2000-2009, gepaart), cayley (gelernt),
  haar_relu, haar_tanh (a auf Tuning-Seed), haar_abs, haar_sq_ohnenorm (z² ohne Normierung von x), mlp9 (ReLU 256-9-10),
  haar_zufallslabel (Negativkontrolle: Trainingslabels permutiert, Test mit echten Labels).
HYPOTHESEN (Holm ueber H-A..H-G, je Datensatz, x7):
  H-A QIRE gelernt ≡ Haar² (TOST ±0,5 pp)        H-B QIRE gelernt ≡ QIRE Zufallswinkel (TOST)
  H-C Fastfood-2 ≡ Haar² (TOST)                     H-D Haar² − Pixel-Softmax > 0 (einseitig)
  H-E Haar² − Haar linear > 0 (einseitig)           H-F Suche ≤ Haar² (suche − haar < 0, einseitig; Seeds 2000-2009)
  H-G Haar² ≥ bester klassischer Nichtlinearitaets-Arm (Superioritaet einseitig, zusaetzlich TOST ±0,5 pp berichtet).
  Der „beste“ klassische Arm {haar_relu, haar_tanh, haar_abs, haar_sq_ohnenorm, mlp9} wird auf Tuning-Seed 2099
  (Validierung, nie Test) festgelegt. Cayley − QIRE nur berichtet (mit Parameterzaehlung).
KRITERIEN (im Code, vor dem Lauf): siehe urteil(); Urteile sind Strings.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen.clifford import LAGEN, QUBITS_JE_DATENSATZ, QireZelle
from abteilungen.clifford_suche import Bewerter, _bewerte, holm, kopf_logistisch, normiere, zufall_codes
from abteilungen.clifford_suche_v2 import tost_p
from abteilungen.walsh_kontrolle import wilcoxon_schnell
from labor.gemeinsam import ERGEBNISSE, bilder, gittersuche_stabil, parallel, zusammenfassung

KARTE = "I-038"
DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
N = 8
D = 256
L2 = 3e-3
SEEDS = tuple(range(2000, 2020))
SEEDS_SUCHE = tuple(range(2000, 2010))
TUNING_SEED = 2099
ENTWURF = {"mnist": (2090,), "fashion_mnist": (2091,)}
ENTWURF_SUCHE = (2090, 2091)
ENTWURF_TUNING = 2098
_BELEGT = set(range(900, 1001))
TOST_W = 0.005
ALPHA = 0.05
SCHRITTE = 300                      # Adam-Schritte der gelernten Arme (wie v3)
GITTER = {"qire": (0.02, 0.04, 0.08, 0.16, 0.32), "qire_nocnot": (0.02, 0.04, 0.08, 0.16, 0.32),
          "cayley": (0.0025, 0.005, 0.01, 0.02, 0.04), "mlp9": (0.001, 0.003, 0.01, 0.03)}
TANH_A = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
KLASSISCH = ("haar_relu", "haar_tanh", "haar_abs", "haar_sq_ohnenorm", "mlp9")
FAMILIE = ("H-A", "H-B", "H-C", "H-D", "H-E", "H-F", "H-G")
PAARE = {"H-A": ("qire", "haar_sq"), "H-B": ("qire", "qire_zufall"), "H-C": ("fastfood2", "haar_sq"),
         "H-D": ("haar_sq", "pix_softmax"), "H-E": ("haar_sq", "haar_lin"), "H-F": ("suche", "haar_sq"),
         "H-G": ("haar_sq", None)}                         # None -> bester klassischer Arm (hp)
GLEICH = ("H-A", "H-B", "H-C")
GELERNT = ("qire", "qire_nocnot", "cayley")
MIN_SUCHE = 10
NK_BAND = (0.09, 0.115)
PK_MIN = 0.005
KOLLAPS = 0.05
PARAMETER = {"pix_softmax": 2570, "ident_sq": 2570, "haar_sq": 2570, "haar_lin": 2570, "fastfood2": 2570,
             "qire_zufall": 2570, "qire": 2570 + 32, "qire_nocnot": 2570 + 32, "suche": 2570, "cayley": 2570 + 32640,
             "haar_relu": 2570, "haar_tanh": 2570, "haar_abs": 2570, "haar_sq_ohnenorm": 2570, "mlp9": 256 * 9 + 9 + 90 + 10}
FESTE_WAHL = {"haar_sq": "65.536 feste Zufallszahlen (Seed nicht ausgewaehlt)", "fastfood2": "512 Vorzeichenbits",
              "qire_zufall": "32 feste Zufallswinkel", "suche": "32 Gatterbits (kanonisch), Auswahl per LOO-Suche",
              "haar_tanh": "Skala a (1 Zahl, Tuning-Seed)"}

HYPOTHESEN = {
    "H-A": "QIRE gelernt ≡ Haar² (TOST ±0,5 pp, Holm x7). GLEICHWERTIG | QIRE_BESSER | HAAR_BESSER | HU.",
    "H-B": "QIRE gelernt ≡ QIRE mit Zufallswinkeln (TOST). GLEICHWERTIG | GELERNT_BESSER | ZUFALL_BESSER | HU.",
    "H-C": "Fastfood-2 ≡ Haar² (TOST). GLEICHWERTIG | FASTFOOD_BESSER | HAAR_BESSER | HU.",
    "H-D": "Haar² − Pixel-Softmax > 0 (einseitig exakter Wilcoxon, Holm x7). HAAR_BESSER | H0 (Gegenrichtung) | HU.",
    "H-E": "Haar² − Haar linear > 0 (einseitig). QUADRAT_TRAEGT | H0 (Gegenrichtung) | HU.",
    "H-F": "Suche ≤ Haar² (suche − haar < 0, einseitig, 10 gepaarte Seeds). SUCHE_NICHT_BESSER | H0 (Suche besser) | HU.",
    "H-G": "Haar² ≥ bester klassischer Nichtlinearitaets-Arm (auf Tuning-Seed 2099 bestimmt): Superioritaet einseitig; "
           "TOST ±0,5 pp zusaetzlich (Bonferroni x2). BORN_BESSER | GLEICHWERTIG | KLASSISCH_BESSER | UNKLAR.",
    "NK": "Negativkontrolle Zufallslabel (Haar², permutierte Trainingslabels): Mittel der Testgenauigkeit in "
          f"{NK_BAND}. BESTANDEN | DURCHGEFALLEN. Faellt sie durch, sind alle Urteile HU/UNKLAR.",
    "PK": "Positivkontrolle je Datensatz: H-D UND H-E je Holm-signifikant mit Mittel >= 0,5 pp. Sonst kein "
          "GLEICHWERTIG (H-A/B/C/G): die Messkette haette einen Unterschied nicht gezeigt.",
    "B": "beschreibend: Mittel/sd aller Arme, Cayley − QIRE (mit Parameterzaehlung), weitere Differenzen, "
         "Kollaps-Pruefung, Tuning-Ergebnisse, bester klassischer Arm.",
}
FOLGEFRAGEN = {
    "H-A GLEICHWERTIG": "QIRE-Struktur traegt nichts Messbares: Kernaussage 'quanten-inspiriert' (Plan §1.2) belegt.",
    "H-A QIRE_BESSER": "Struktur traegt doch: welche (Winkel, CNOT)? I-041 / Skalierung n=10 pruefen.",
    "H-G KLASSISCH_BESSER": "Quadrat (Born) ist nicht die beste Nichtlinearitaet: Kernaussage auf 'Mischung' kuerzen.",
    "H-G BORN_BESSER": "Born-Quadrat ist unter klassischen Nichtlinearitaeten ausgezeichnet: Erklaerung suchen.",
}


# ------------------------------------------------------------------ Bausteine
def walsh(n=N):
    H = torch.ones(1, 1)
    for _ in range(n):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(2 ** n)


def haar(g):
    Q, R = torch.linalg.qr(torch.randn(D, D, generator=g))
    return Q * torch.sign(torch.diagonal(R))


def vorzeichen(g):
    return (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()


def fastfood2(g):
    H = walsh()
    return H @ torch.diag(vorzeichen(g)) @ H @ torch.diag(vorzeichen(g))


def feat(x, U, nl="sq", a=1.0):
    z = x @ U.T
    return {"sq": lambda: z ** 2, "abs": lambda: z.abs(), "lin": lambda: z, "relu": lambda: torch.relu(z),
            "tanh": lambda: torch.tanh(a * z)}[nl]()


def ev(ftr, ytr, fte, yte):
    return kopf_logistisch(ftr, ytr, fte, yte, l2=L2)[0]


class Cayley(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(D, D))
        self.register_buffer("I", torch.eye(D))

    def U(self):
        S = self.A - self.A.T
        return torch.linalg.solve(self.I + S, self.I - S)


class Kopfmodell(torch.nn.Module):
    """Gelernte Zelle + eingebauter Linearkopf (nur zum Lernen von U; danach U eingefroren, Standardkopf neu)."""

    def __init__(self, zelle, kind):
        super().__init__()
        self.z, self.kind = zelle, kind
        self.kopf = torch.nn.Linear(D, 10)

    def U(self):
        return self.z.U() if self.kind == "cayley" else self.z.unitaer()

    def forward(self, xn):
        return self.kopf(((xn @ self.U().T) ** 2 - 1.0 / D) * 16)


def trainiere_u(kind, xn, y, lr, seed, steps=SCHRITTE):
    torch.manual_seed(seed)
    z = Cayley() if kind == "cayley" else QireZelle(n=N, lagen=LAGEN, cnot=(kind == "qire"))
    m = Kopfmodell(z, "cayley" if kind == "cayley" else "qire")
    opt = torch.optim.Adam([p for n, p in m.named_parameters() if not n.startswith("z.kopf")], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(xn), y).backward()
        opt.step()
    with torch.no_grad():
        return m.U().detach()


def mlp(xtr, ytr, xte, yte, lr, seed, h=9, epochen=60, wd=1e-4):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-3)
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(D, h), torch.nn.ReLU(), torch.nn.Linear(h, 10))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochen):
        p = torch.randperm(len(ytr))
        for i in range(0, len(p), 128):
            j = p[i:i + 128]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(a[j]), ytr[j]).backward()
            opt.step()
    with torch.no_grad():
        return float((net(b).argmax(1) == yte).float().mean())


def _daten(ds, seed, klein=False):
    if klein:
        return bilder(ds, seed, n_train=600, n_test=500, seite=16)
    return bilder(ds, seed, n_train=6_000, n_test=5_000, seite=16)


def _haar_seed(seed):
    return torch.Generator().manual_seed(seed + 7000)


# ------------------------------------------------------------------ Tuning (Seed 2099, Validierung 1.000 aus 6.000)
def _tune_teile(ds, seed, klein):
    xtr, ytr, _, _ = _daten(ds, seed, klein)
    nv = len(ytr) // 6
    return xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]


def _tune_arbeiter(a):
    """(schluessel 'ds|arm', seed, lr, schritte, klein) -> {'acc': Validierungsgenauigkeit}."""
    schl, seed, lr, schritte, klein = a
    ds, arm = schl.split("|")
    xa, ya, xb, yb = _tune_teile(ds, seed, klein)
    if arm == "mlp9":
        return {"acc": mlp(xa, ya, xb, yb, lr, seed, epochen=60 if not klein else 5)}
    U = trainiere_u(arm, normiere(xa), ya, lr, seed, schritte)
    f = lambda x: (normiere(x) @ U.T) ** 2
    acc = ev(f(xa), ya, f(xb), yb)
    return {"acc": acc if math.isfinite(acc) else 0.1}


def tuning_voll(datensaetze, seed, schritte, klein, prozesse=None):
    """Gittersuche je Arm (eigenes Gitter je Arm: Aufruf je Arm, damit die Gitter verschieden sein duerfen)."""
    hp = {}
    for ds in datensaetze:
        hp[ds] = {"lr": {}, "tuning": {}}
    for arm, gitter in GITTER.items():
        schl = [f"{ds}|{arm}" for ds in datensaetze]
        info, _ = gittersuche_stabil(
            schl, lambda s, lr: [(s, seed, lr, schritte, klein)],
            lambda aufg: parallel(_tune_arbeiter, aufg, prozesse=prozesse), gitter=gitter)
        for ds in datensaetze:
            hp[ds]["lr"][arm] = info[f"{ds}|{arm}"]["lr"]
            hp[ds]["tuning"][arm] = info[f"{ds}|{arm}"]
    for ds in datensaetze:
        xa, ya, xb, yb = _tune_teile(ds, seed, klein)
        na, nb = normiere(xa), normiere(xb)
        U = haar(_haar_seed(seed))
        val = {"haar_sq": ev(feat(na, U), ya, feat(nb, U), yb)}
        for nl in ("relu", "abs"):
            val["haar_" + nl] = ev(feat(na, U, nl), ya, feat(nb, U, nl), yb)
        val["haar_sq_ohnenorm"] = ev(feat(xa, U), ya, feat(xb, U), yb)
        ta = {a: ev(feat(na, U, "tanh", a), ya, feat(nb, U, "tanh", a), yb) for a in TANH_A}
        a_best = max(TANH_A, key=lambda a: (ta[a], -a))
        val["haar_tanh"] = ta[a_best]
        mlp_g = hp[ds]["tuning"]["mlp9"]["gitter"][str(hp[ds]["lr"]["mlp9"])]
        val["mlp9"] = max(mlp_g["je_seed"])
        best = max(KLASSISCH, key=lambda k: (val[k], k))
        hp[ds].update({"tanh_a": a_best, "tanh_val": {str(a): v for a, v in ta.items()}, "val": val,
                       "bester_klassisch": best})
    return hp


# ------------------------------------------------------------------ Arbeiter je (Datensatz, Seed)
def _arme_seed(a):
    ds, s, hp, schritte, klein = a
    t0 = time.time()
    xtr, ytr, xte, yte = _daten(ds, s, klein)
    ntr, nte = normiere(xtr), normiere(xte)
    g = _haar_seed(s)
    z = {"seed": s, "datensatz": ds, "n_train": len(ytr), "n_test": len(yte)}
    z["pix_softmax"] = kopf_logistisch(xtr, ytr, xte, yte, l2=L2)[0]
    I = torch.eye(D)
    z["ident_sq"] = ev(feat(ntr, I), ytr, feat(nte, I), yte)
    Ur = haar(g)
    for nl, name in (("sq", "haar_sq"), ("lin", "haar_lin"), ("relu", "haar_relu"), ("abs", "haar_abs")):
        z[name] = ev(feat(ntr, Ur, nl), ytr, feat(nte, Ur, nl), yte)
    z["haar_tanh"] = ev(feat(ntr, Ur, "tanh", hp["tanh_a"]), ytr, feat(nte, Ur, "tanh", hp["tanh_a"]), yte)
    z["haar_sq_ohnenorm"] = ev(feat(xtr, Ur), ytr, feat(xte, Ur), yte)
    z["mlp9"] = mlp(xtr, ytr, xte, yte, hp["lr"]["mlp9"], s, epochen=60 if not klein else 5)
    Uf = fastfood2(g)
    z["fastfood2"] = ev(feat(ntr, Uf), ytr, feat(nte, Uf), yte)
    torch.manual_seed(s)
    with torch.no_grad():
        Uq = QireZelle(n=N, lagen=LAGEN, cnot=True).unitaer()
    z["qire_zufall"] = ev(feat(ntr, Uq), ytr, feat(nte, Uq), yte)
    p = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(s + 9000))
    z["haar_zufallslabel"] = ev(feat(ntr, Ur), ytr[p], feat(nte, Ur), yte)
    for arm in GELERNT:
        U = trainiere_u(arm, ntr, ytr, hp["lr"][arm], s, schritte)
        orth = float((U @ U.T - torch.eye(D)).abs().max())
        z[arm] = ev(feat(ntr, U), ytr, feat(nte, U), yte)
        z[arm + "_orth_fehler"] = orth
    z["sekunden"] = time.time() - t0
    return z


def _suche_seed(a):
    """Clifford-Suche wie I-036-v3 (gleiche Daten und Kopf wie die anderen Arme -> gepaart)."""
    ds, s, schritte, politur_max, n_zufall, klein = a
    t0 = time.time()
    n = QUBITS_JE_DATENSATZ[ds]
    xtr, ytr, xte, yte = _daten(ds, s, klein)
    bew = Bewerter(xtr, ytr, n)
    xs = (normiere(xtr), ytr, normiere(xte), yte, L2)
    zv = [bew(c)[0] for c in zufall_codes(s, n_zufall, n)]
    skala = max(float(np.std(zv)), 1e-4)
    start = np.random.default_rng(v1.START_SEED + s).integers(0, 4, (LAGEN, n))
    best, s_best, _ = v1.suche(bew, start, np.random.default_rng(v1.SA_SEED + s), skala, schritte,
                               politur_max=politur_max)
    return {"seed": s, "datensatz": ds, "suche": _bewerte(best, n, xs)[0], "suche_val": s_best[0],
            "suche_codes": best.tolist(), "suche_bewertungen": bew.bewertungen, "sekunden": time.time() - t0}


def _arbeiter(a):
    art, arg = a
    return _schluessel(a), (_suche_seed(arg) if art == "suche" else _arme_seed(arg))


def _schluessel(a):
    art, arg = a
    return f"{art}|{arg[0]}|{arg[1]}"


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
    """Jeder fertige Seed geht SOFORT in die Zwischendatei (in ergebnisse/); fertige werden uebersprungen."""
    import multiprocessing as mp
    from labor.gemeinsam import _init_arbeiter
    fn = fn or _arbeiter
    schluessel = schluessel or _schluessel
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


def _pruefe_seeds(seeds, suche_seeds, datensaetze):
    alle = set().union(*[set(seeds[ds]) for ds in datensaetze]) | set(suche_seeds)
    return not (alle & _BELEGT) and all(2000 <= s <= 2019 for s in alle) and set(suche_seeds) <= set(seeds[datensaetze[0]])


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=20, datensaetze=DATENSAETZE, prozesse=None, zwischen=None):
    datensaetze = tuple(datensaetze)
    if schnell:
        seeds, suche_seeds, tseed = ENTWURF, ENTWURF_SUCHE, ENTWURF_TUNING
        schritte, sa, politur, nz, klein = 30, 80, 20, 5, True
    else:
        if n_seeds != 20:
            raise ValueError("n_seeds ist vorregistriert: 20")
        seeds = {ds: SEEDS for ds in datensaetze}
        suche_seeds, tseed = SEEDS_SUCHE, TUNING_SEED
        schritte, sa, politur, nz, klein = SCHRITTE, v1.SA_SCHRITTE, v1.POLITUR_MAX, v1.N_ZUFALL, False
        if not _pruefe_seeds(seeds, suche_seeds, datensaetze):
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "schritte": schritte, "sa": sa, "politur": politur, "n_zufall": nz,
               "l2": L2, "tuning_seed": tseed, "klein": klein, "datensaetze": list(datensaetze)}
    t0 = time.time()
    # Stufe 0: Tuning (einmal, dann in der Zwischendatei)
    zw = _lade_zwischen(pfad, kennung)
    if "tuning" in zw:
        hp = zw["tuning"]
        print("  Wiederaufnahme: Tuning aus Zwischendatei", flush=True)
    else:
        hp = tuning_voll(datensaetze, tseed, schritte, klein, prozesse=prozesse)
        zw["tuning"] = hp
        _schreibe_zwischen(pfad, kennung, zw)
    # Stufe 1: je Seed alle Arme; Suche als eigene Aufgabe
    aufg = [("suche", (ds, s, sa, politur, nz, klein)) for ds in datensaetze for s in suche_seeds]
    aufg += [("arme", (ds, s, hp[ds], schritte, klein)) for ds in datensaetze for s in seeds[ds]]
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zeilen = {}
    for ds in datensaetze:
        zeilen[ds] = []
        for s in seeds[ds]:
            z = dict(alle[f"arme|{ds}|{s}"])
            if f"suche|{ds}|{s}" in alle:
                z["suche"] = alle[f"suche|{ds}|{s}"]["suche"]
            zeilen[ds].append(z)
    e = {"abteilung": "Struktur", "modul": "struktur_aequivalenz", "karte": KARTE, "version": 1,
         "hypothesen": HYPOTHESEN, "folgefragen": FOLGEFRAGEN, "seeds": {ds: list(seeds[ds]) for ds in datensaetze},
         "seeds_suche": list(suche_seeds), "tuning_seed": tseed, "tuning": hp, "parameter": PARAMETER,
         "feste_wahl": FESTE_WAHL, "einstellungen": {"schritte": schritte, "sa": sa, "politur": politur,
                                                     "n_zufall": nz, "l2": L2, "tost": TOST_W, "klein": klein},
         "zwischendatei": str(pfad), "zeilen": zeilen, "suche_codes": {k: v["suche_codes"] for k, v in alle.items()
                                                                       if k.startswith("suche|")},
         "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, hp)
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Statistik und Auswertung
def wilcoxon_einseitig(d, groesser=True) -> float:
    """Exakter einseitiger Wilcoxon (Nullverteilung symmetrisch: p_zweiseitig/2 in Richtung der Daten, sonst 1 - p/2)."""
    d = np.asarray([v for v in np.asarray(d, dtype=float) if v != 0.0])
    if len(d) == 0:
        return 1.0
    p2 = wilcoxon_schnell(d)
    raenge = np.argsort(np.argsort(np.abs(d))) + 1.0
    w, mitte = raenge[d > 0].sum(), raenge.sum() / 2
    if w == mitte:
        return 1.0
    passt = (w > mitte) == bool(groesser)
    return float(p2 / 2 if passt else 1 - p2 / 2)


def _gepaart(zz, a, b):
    d = np.array([z[a] - z[b] for z in zz if a in z and b in z], dtype=float)
    if len(d) < 2:
        return None
    from scipy.stats import t as t_vert
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return {"mittel_pp": m * 100, "sd_pp": float(d.std(ddof=1)) * 100, "ki_pp": [(m - q * se) * 100, (m + q * se) * 100],
            "positiv": int((d > 0).sum()), "n": int(len(d)), "p_zweiseitig": wilcoxon_schnell(d),
            "p_gr": wilcoxon_einseitig(d, True), "p_kl": wilcoxon_einseitig(d, False), "p_tost": tost_p(d, TOST_W)}


def kollaps_seeds(zz, arm):
    w = np.array([z[arm] for z in zz], dtype=float)
    med = float(np.median(w))
    return [z["seed"] for z in zz if not (z[arm] >= med - KOLLAPS)]


def _ds_auswertung(zz, h) -> dict:
    best = h["bester_klassisch"]
    r = {"n": len(zz), "bester_klassisch": best}
    arme = [k for k in PARAMETER if k in zz[0]]
    r["arme"] = {k: zusammenfassung([z[k] for z in zz if k in z]) for k in arme + ["haar_zufallslabel"]}
    r["diff"] = {}
    for hy, (a, b) in PAARE.items():
        b = b or best
        r["diff"][hy] = {"paar": [a, b], **(_gepaart(zz, a, b) or {})}
    r["beschreibend"] = {f"{a} - {b}": _gepaart(zz, a, b) for a, b in
                         (("cayley", "qire"), ("haar_sq", "ident_sq"), ("qire", "qire_nocnot"),
                          ("fastfood2", "haar_sq"), ("qire", "pix_softmax"), ("haar_sq", "haar_relu"),
                          ("haar_sq", "haar_tanh"), ("haar_sq", "haar_abs"), ("haar_sq", "haar_sq_ohnenorm"),
                          ("haar_sq", "mlp9"), ("suche", "pix_softmax"))}
    r["kollaps"] = {k: kollaps_seeds(zz, k) for k in GELERNT + ("mlp9",)}
    prim = {}
    for hy in FAMILIE:
        d = r["diff"][hy]
        if "p_tost" not in d:
            prim[hy] = 1.0
        elif hy in GLEICH:
            prim[hy] = d["p_tost"]
        elif hy == "H-F":
            prim[hy] = d["p_kl"]
        else:
            prim[hy] = d["p_gr"]
    r["p_primaer"], r["p_holm"] = prim, holm(prim)
    zw = {hy: (r["diff"][hy].get("p_zweiseitig", 1.0)) for hy in GLEICH + ("H-G",)}
    r["p_holm_zweiseitig"] = holm(zw)
    r["negativ"] = r["arme"]["haar_zufallslabel"]["mittel"]
    return r


def auswertung(zeilen: dict, hp: dict) -> dict:
    return {"datensaetze": {ds: _ds_auswertung(zz, hp[ds]) for ds, zz in zeilen.items()}}


# ------------------------------------------------------------------ Urteile (Kriterien im Code)
def _f(x):
    return f"{x:+.2f}".replace(".", ",")


def _nk(a) -> str:
    ok = all(NK_BAND[0] <= r["negativ"] <= NK_BAND[1] for r in a["datensaetze"].values())
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _pk(r, ds) -> str:
    ok = all(r["p_holm"][h] < ALPHA and r["diff"][h].get("mittel_pp", 0) >= PK_MIN * 100 for h in ("H-D", "H-E"))
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _zahlen(r, h):
    d = r["diff"][h]
    return (f"d {_f(d['mittel_pp'])} pp, 90-%-KI [{_f(d['ki_pp'][0])}; {_f(d['ki_pp'][1])}], n {d['n']}, "
            f"p_Holm {r['p_holm'][h]:.3g}")


def _urteil_h(h, ds, r, nk, pk) -> str:
    d = r["diff"][h]
    if "mittel_pp" not in d:
        return "HU (zu wenige gepaarte Seeds)"
    if nk != "BESTANDEN":
        return "UNKLAR (Negativkontrolle durchgefallen)" if h == "H-G" else "HU (Negativkontrolle durchgefallen)"
    a, b = d["paar"]
    koll = sorted({s for arm in (a, b) if arm in r["kollaps"] for s in r["kollaps"][arm]})
    if koll:
        return ("UNKLAR" if h == "H-G" else "HU") + f" (Instabilitaet/Kollaps in Seeds {koll})"
    if h == "H-F" and d["n"] < MIN_SUCHE:
        return f"HU (nur {d['n']} Suche-Seeds, Vorbedingung >= {MIN_SUCHE})"
    z = _zahlen(r, h)
    if h in GLEICH:
        n1, n2 = {"H-A": ("QIRE", "HAAR"), "H-B": ("GELERNT", "ZUFALL"), "H-C": ("FASTFOOD", "HAAR")}[h]
        if r["p_holm"][h] < ALPHA:
            if pk != "BESTANDEN":
                return f"HU (Positivkontrolle H-D/H-E nicht gefeuert; Gleichwertigkeit nicht belegbar) | {z}"
            return f"GLEICHWERTIG ({z})"
        if r["p_holm_zweiseitig"][h] < ALPHA:
            return f"{n1 if d['mittel_pp'] > 0 else n2}_BESSER ({z}, p_zweiseitig,Holm {r['p_holm_zweiseitig'][h]:.3g})"
        return f"HU ({z})"
    if h == "H-G":
        best = r["bester_klassisch"]
        z = f"{z}, Gegner {best}, p_TOST {d['p_tost']:.3g}"
        if r["p_holm"][h] < ALPHA:
            return f"BORN_BESSER ({z})"
        if d["p_tost"] * 2 < ALPHA:
            return (f"GLEICHWERTIG ({z}, TOST Bonferroni x2)" if pk == "BESTANDEN"
                    else f"UNKLAR (Positivkontrolle nicht gefeuert; {z})")
        if r["p_holm_zweiseitig"][h] < ALPHA and d["mittel_pp"] < 0:
            return f"KLASSISCH_BESSER ({z}, p_zweiseitig,Holm {r['p_holm_zweiseitig'][h]:.3g})"
        return f"UNKLAR ({z})"
    # D, E, F einseitig
    ja = {"H-D": "HAAR_BESSER", "H-E": "QUADRAT_TRAEGT", "H-F": "SUCHE_NICHT_BESSER"}[h]
    if r["p_holm"][h] < ALPHA:
        return f"{ja} ({z})"
    if d["p_zweiseitig"] < ALPHA and ((d["mittel_pp"] < 0) if h != "H-F" else (d["mittel_pp"] > 0)):
        return f"H0 ({'Suche besser' if h == 'H-F' else 'Gegenrichtung'}, p_zweiseitig {d['p_zweiseitig']:.3g}; {z})"
    return f"HU ({z})"


def urteil(a: dict) -> dict:
    nk = _nk(a)
    u = {"NK": f"{nk} (Zufallslabel-Mittel: " + ", ".join(f"{NAME[ds]} {r['negativ']:.4f}" for ds, r in
                                                          a["datensaetze"].items()) + f"; Band {NK_BAND})"}
    for ds, r in a["datensaetze"].items():
        pk = _pk(r, ds)
        u[f"PK-{NAME[ds]}"] = f"{pk} (H-D p_Holm {r['p_holm']['H-D']:.3g}, H-E p_Holm {r['p_holm']['H-E']:.3g})"
        for h in FAMILIE:
            u[f"{h}-{NAME[ds]}"] = _urteil_h(h, ds, r, nk, pk)
        bs = r["beschreibend"]["cayley - qire"] or {"mittel_pp": float("nan")}
        u[f"B-{NAME[ds]}"] = (f"beschreibend: bester klassischer Arm {r['bester_klassisch']}; Cayley − QIRE "
                              f"{_f(bs['mittel_pp'])} pp bei {PARAMETER['cayley']} vs {PARAMETER['qire']} Parametern; "
                              "Rest in auswertung.beschreibend")
    return u


# ------------------------------------------------------------------ Vortest
def _synth_zeilen(rng, off=None, sd=0.001, n=20, n_suche=10, koll=None):
    """Synthetische Zeilen: Basis 0,90 + Offsets je Arm + Rauschen (gemeinsamer Seed-Effekt -> gepaart)."""
    base = {k: 0.90 for k in list(PARAMETER) + ["haar_zufallslabel"]}
    base["haar_zufallslabel"] = 0.10
    base.update(off or {})
    zz = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        z = {"seed": 2000 + i}
        for k, v in base.items():
            if k == "suche" and i >= n_suche:
                continue
            z[k] = v + (0.0 if k == "haar_zufallslabel" else gem) + rng.normal(0, sd)
        if koll and i == 0:
            z[koll] -= 0.2
        zz.append(z)
    return zz


_HP = {"bester_klassisch": "haar_relu", "tanh_a": 16.0, "lr": {}}
_BASIS = {"haar_sq": 0.935, "haar_lin": 0.915, "pix_softmax": 0.910, "ident_sq": 0.90, "qire": 0.935,
          "qire_zufall": 0.935, "fastfood2": 0.935, "suche": 0.935, "haar_relu": 0.920, "cayley": 0.95,
          "haar_tanh": 0.92, "haar_abs": 0.93, "haar_sq_ohnenorm": 0.92, "mlp9": 0.90, "qire_nocnot": 0.92}


def _urteile_synth(rng, off=None, sd=0.001, n_suche=10, koll=None, nk_wert=None, bester="haar_relu"):
    o = dict(_BASIS)
    o.update(off or {})
    zz = _synth_zeilen(rng, o, sd=sd, n_suche=n_suche, koll=koll)
    if nk_wert is not None:
        for z in zz:
            z["haar_zufallslabel"] = nk_wert
    hp = dict(_HP, bester_klassisch=bester)
    return urteil(auswertung({"mnist": zz}, {"mnist": hp}))


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER: {name}", flush=True)
            ok = False

    g = torch.Generator().manual_seed(1)
    I = torch.eye(D)
    U = haar(g)
    pruefe("Haar orthogonal (gleiche Matrix)", float((U @ U.T - I).abs().max()) < 1e-4)
    Uf = fastfood2(g)
    pruefe("Fastfood-2 orthogonal", float((Uf @ Uf.T - I).abs().max()) < 1e-4)
    pruefe("Fastfood-2 mischt dicht", float((Uf.abs() > 1e-6).float().mean()) > 0.5)
    pruefe("Walsh-Matrix orthogonal", float((walsh() @ walsh().T - I).abs().max()) < 1e-4)
    torch.manual_seed(0)
    with torch.no_grad():
        Uq = QireZelle(n=N, lagen=LAGEN, cnot=True).unitaer()
        Un = QireZelle(n=N, lagen=LAGEN, cnot=False).unitaer()
    pruefe("QIRE orthogonal", float((Uq @ Uq.T - I).abs().max()) < 1e-3)
    pruefe("QIRE mit CNOT != ohne CNOT", float((Uq - Un).abs().max()) > 1e-2)
    c = Cayley()
    pruefe("Cayley startet bei Identitaet", float((c.U().detach() - I).abs().max()) < 1e-6)
    c.A.data = 0.1 * torch.randn(D, D)
    pruefe("Cayley orthogonal", float((c.U() @ c.U().T - I).abs().max()) < 1e-3)
    x = torch.rand(7, D) * 3
    xn = normiere(x)
    pruefe("Normierung", float((xn.norm(dim=1) - 1).abs().max()) < 1e-5)
    pruefe("ohne Normierung != mit", float((feat(x, U) - feat(xn, U)).abs().max()) > 1e-3)
    pruefe("linear = z", torch.allclose(feat(xn, U, "lin"), xn @ U.T))
    pruefe("relu>=0, abs>=0, tanh<1", bool((feat(xn, U, 'relu') >= 0).all() and (feat(xn, U, 'abs') >= 0).all()
                                             and (feat(xn, U, 'tanh', 50.0).abs() <= 1).all()))
    pruefe("Identitaet: z^2 = x^2", torch.allclose(feat(xn, I), xn ** 2))
    # Zufallslabel: Permutation behaelt Klassenhaeufigkeit, aendert Zuordnung
    y = torch.randint(0, 10, (500,))
    p = torch.randperm(500, generator=torch.Generator().manual_seed(3))
    pruefe("Zufallslabel Haeufigkeiten", bool((torch.bincount(y[p], minlength=10) == torch.bincount(y, minlength=10)).all())
           and bool((y[p] != y).any()))
    # Lernen senkt den Verlust (QIRE und Cayley) auf lernbarer Synthetik
    gs = torch.Generator().manual_seed(5)
    xs = normiere(torch.rand(300, D, generator=gs))
    ys = (xs[:, :10].argmax(1)).long()
    for kind in ("qire", "cayley"):
        torch.manual_seed(0)
        z0 = Cayley() if kind == "cayley" else QireZelle(n=N, lagen=LAGEN)
        m = Kopfmodell(z0, kind)
        v0 = float(torch.nn.functional.cross_entropy(m(xs), ys))
        Ue = trainiere_u(kind, xs, ys, 0.05, 0, 25)
        pruefe(f"{kind}: Training liefert orthogonales U", float((Ue @ Ue.T - I).abs().max()) < 1e-3)
        pruefe(f"{kind}: Training aendert U", float((Ue - m.U().detach()).abs().max()) > 1e-4 or kind == "qire")
        del v0
    # Statistik
    pruefe("TOST: gleich -> p klein", tost_p(np.random.default_rng(0).normal(0, 0.001, 20), TOST_W) < 0.01)
    pruefe("TOST: Differenz 1 pp -> p gross", tost_p(np.random.default_rng(0).normal(0.01, 0.001, 20), TOST_W) > 0.5)
    dd = np.random.default_rng(1).normal(0.01, 0.003, 20)
    pruefe("Wilcoxon einseitig Richtung", wilcoxon_einseitig(dd, True) < 0.001 and wilcoxon_einseitig(dd, False) > 0.99)
    pruefe("Wilcoxon einseitig Nulldifferenz", wilcoxon_einseitig(np.zeros(5)) == 1.0)
    h = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    pruefe("Holm", abs(h["a"] - 0.03) < 1e-12 and abs(h["b"] - 0.04) < 1e-12 and abs(h["c"] - 0.5) < 1e-12)
    pruefe("Seeds-Pruefung", _pruefe_seeds({ds: SEEDS for ds in DATENSAETZE}, SEEDS_SUCHE, DATENSAETZE)
           and not _pruefe_seeds({ds: tuple(range(990, 1010)) for ds in DATENSAETZE}, (), DATENSAETZE))
    pruefe("Suche nur auf 10 Seeds, Rest 2000-2019", len(SEEDS) == 20 and len(SEEDS_SUCHE) == 10
           and set(SEEDS_SUCHE) <= set(SEEDS))
    pruefe("Familie 7 Hypothesen", len(FAMILIE) == 7 and set(PAARE) == set(FAMILIE))
    return ok


def _fn_test(a):
    return f"t|{a[0]}", {"v": a[0] * 2}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as t:
        pfad = Path(t) / "z.json"
        kennung = {"k": 1}
        aufg = [(i, 0) for i in range(4)]
        sk = lambda a: f"t|{a[0]}"
        z1 = _rechne(aufg[:2], pfad, kennung, fn=_fn_test, prozesse=1, schluessel=sk)
        z2 = _rechne(aufg, pfad, kennung, fn=_fn_test, prozesse=1, schluessel=sk)
        ok &= len(z1) == 2 and len(z2) == 4 and z2["t|1"] == z1["t|1"]
        try:
            _rechne(aufg, pfad, {"k": 2}, fn=_fn_test, prozesse=1, schluessel=sk)
            ok = False
        except ValueError:
            pass
        ok &= not pfad.with_name(pfad.name + ".tmp").exists()
    if not ok:
        print("  Vortest FEHLER: Wiederaufnahme", flush=True)
    return ok


def _vortest_urteile() -> bool:
    """Jeden Urteilszweig erzwingen (synthetische Daten) und die Pruefung gegen den erwarteten Wortanfang."""
    rng = np.random.default_rng(11)
    ok = True
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    M = "-MNIST"
    u = _urteile_synth(rng)                                           # alles wie erwartet
    erwarte("Normalfall", u, **{"NK": "BESTANDEN", "PK-MNIST": "BESTANDEN", "H-A" + M: "GLEICHWERTIG",
                                "H-B" + M: "GLEICHWERTIG", "H-C" + M: "GLEICHWERTIG", "H-D" + M: "HAAR_BESSER",
                                "H-E" + M: "QUADRAT_TRAEGT", "H-F" + M: "HU", "H-G" + M: "BORN_BESSER"})
    u = _urteile_synth(rng, off={"suche": 0.925})                     # Suche deutlich schlechter (-1 pp)
    erwarte("Suche schlechter", u, **{"H-F" + M: "SUCHE_NICHT_BESSER"})
    u = _urteile_synth(rng, off={"suche": 0.95})
    erwarte("Suche besser", u, **{"H-F" + M: "H0 (Suche besser"})
    u = _urteile_synth(rng, off={"qire": 0.95, "fastfood2": 0.92, "haar_relu": 0.95})
    erwarte("A,C,B ungleich", u, **{"H-A" + M: "QIRE_BESSER", "H-B" + M: "GELERNT_BESSER",
                                    "H-C" + M: "HAAR_BESSER", "H-G" + M: "KLASSISCH_BESSER"})
    u = _urteile_synth(rng, off={"qire": 0.92, "fastfood2": 0.95, "qire_zufall": 0.95})
    erwarte("A,C,B ungleich (andere Richtung)", u, **{"H-A" + M: "HAAR_BESSER", "H-B" + M: "ZUFALL_BESSER",
                                                       "H-C" + M: "FASTFOOD_BESSER"})
    u = _urteile_synth(rng, off={"haar_relu": 0.935}, sd=0.02)                                  # viel Rauschen: nichts zeigbar
    erwarte("Rauschen", u, **{"H-A" + M: "HU", "H-C" + M: "HU", "H-G" + M: "UNKLAR"})
    u = _urteile_synth(rng, off={"pix_softmax": 0.96, "haar_lin": 0.95})
    erwarte("Gegenrichtung D/E", u, **{"H-D" + M: "H0 (Gegenrichtung", "H-E" + M: "H0 (Gegenrichtung",
                                       "PK-MNIST": "DURCHGEFALLEN", "H-A" + M: "HU (Positivkontrolle",
                                       "H-G" + M: "BORN_BESSER"})
    u = _urteile_synth(rng, off={"pix_softmax": 0.9345, "haar_lin": 0.9345, "haar_relu": 0.935})   # PK kann nicht feuern
    erwarte("PK faellt", u, **{"PK-MNIST": "DURCHGEFALLEN", "H-A" + M: "HU (Positivkontrolle",
                               "H-D" + M: "HU", "H-E" + M: "HU", "H-G" + M: "UNKLAR (Positivkontrolle"})
    u = _urteile_synth(rng, nk_wert=0.30)
    erwarte("NK faellt", u, **{"NK": "DURCHGEFALLEN", "H-A" + M: "HU (Negativkontrolle", "H-D" + M: "HU (Negativ",
                               "H-G" + M: "UNKLAR (Negativkontrolle"})
    u = _urteile_synth(rng, koll="qire")
    erwarte("Kollaps", u, **{"H-A" + M: "HU (Instabilitaet", "H-B" + M: "HU (Instabilitaet", "H-D" + M: "HAAR_BESSER"})
    u = _urteile_synth(rng, n_suche=1)
    erwarte("zu wenige Suche-Seeds", u, **{"H-F" + M: "HU"})
    u = _urteile_synth(rng, off={"mlp9": 0.935, "haar_relu": 0.9355}, bester="mlp9")   # Gegner ~ gleich
    erwarte("H-G gleichwertig", u, **{"H-G" + M: "GLEICHWERTIG"})
    # Mutationstest: Vorzeichen der Differenz umkehren darf das Urteil nicht erhalten
    u1 = _urteile_synth(rng, off={"haar_relu": 0.925})
    u2 = _urteile_synth(rng, off={"haar_relu": 0.945})
    if u1["H-G" + M].split(" ")[0] == u2["H-G" + M].split(" ")[0]:
        print("  Vortest FEHLER: H-G unterscheidet die Richtung nicht", flush=True)
        ok = False
    # Jeder Urteilsstring beginnt mit dem registrierten Wort
    erlaubt = ("GLEICHWERTIG", "QIRE_BESSER", "HAAR_BESSER", "GELERNT_BESSER", "ZUFALL_BESSER", "FASTFOOD_BESSER",
               "QUADRAT_TRAEGT", "SUCHE_NICHT_BESSER", "BORN_BESSER", "KLASSISCH_BESSER", "UNKLAR", "HU", "H0",
               "BESTANDEN", "DURCHGEFALLEN", "beschreibend")
    if not all(isinstance(v, str) and v.startswith(erlaubt) for v in u.values()):
        print("  Vortest FEHLER: Urteil ausserhalb des Vokabulars", flush=True)
        ok = False
    # Alle Zweige je Hypothese erreicht
    soll = {"H-A": {"GLEICHWERTIG", "QIRE_BESSER", "HAAR_BESSER", "HU"}, "H-D": {"HAAR_BESSER", "H0", "HU"},
            "H-G": {"BORN_BESSER", "GLEICHWERTIG", "KLASSISCH_BESSER", "UNKLAR"},
            "H-F": {"SUCHE_NICHT_BESSER", "H0", "HU"}}
    erreicht = {}
    for k, w in gesehen:
        erreicht.setdefault(k[:3], set()).add(w.split(" ")[0])
    for h, ws in soll.items():
        if not ws <= erreicht.get(h, set()) | ({"HU"} if h in ("H-A", "H-D", "H-F") else set()):
            print(f"  Vortest FEHLER: Zweige {ws - erreicht.get(h, set())} fuer {h} nicht erzwungen", flush=True)
            ok = False
    return ok


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok &= _vortest_wiederaufnahme()
    ok &= _vortest_urteile()
    return bool(ok)
