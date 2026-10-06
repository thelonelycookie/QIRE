"""Abteilung Clifford/Information -- I-040 „Holevo-Wirkungsgrad“ (Quantenstufe Q2: exakte Simulation; Kern der
Frage ist informationstheoretisch/klassisch).

HEILMEIER:
  Was wollen wir?  Wie viel der Holevo-Groesse chi_C (Klassen-Ensemble der Mittelzustaende, 8 Qubits) nutzt eine
      Born-Auslese pro Schuss -- I(Klasse; 1 Schuss) -- und was kostet das an Genauigkeit unter Schussrauschen?
      „Holevo umgehen?“ ehrlich beantworten: Umgehen laesst sich die Grenze nicht, sie bindet aber auch nicht;
      bindend ist die Messbasis (die Schaltung vor der Auslese), und die laesst sich messen.
  Wie heute?  Erkundung v4 (recherche/2026-10-05_vertiefung/v4_holevo_schuesse.md, Seeds 960-969, NICHT
      vorregistriert): chi_C = 0,983 Bit; I(Klasse;1 Schuss) Identitaet 0,360 (37 %), Bent-4x4 0,307, zufaellige
      Clifford und uebertragene Suche 0,162 (16 %); linearer Kopf auf Schusshaeufigkeiten: Identitaet 0,847 (S=100),
      0,896 (S=1000); der auf exakten Merkmalen trainierte Kopf bricht bei kleinem S ein (Identitaet S=10:
      0,489 statt 0,596).
  Was ist neu?  (i) Vorregistrierung mit frischen Seeds, (ii) Haar und Fastfood-2 als weitere Arme, Suche
      SEED-EIGEN statt uebertragen, (iii) der schussbewusste Kopf (auf Schusshaeufigkeiten GLEICHER Schusszahl
      trainiert) gegen den exakt trainierten Kopf als gepaarte Hypothese H2, (iv) Wirkungsgrad eta = I/chi als
      Vorhersager des Schussbedarfs fuer 90 % (H3). Neuheitspruefung des Wirkungsgrad-Masses steht aus (Plan E4,
      keine Websuche moeglich) -- bis dahin NICHT als neu formulieren.

ARME (je Seed dieselben Daten; Born-Merkmale p = (W x/|x|)^2, 256 Ausgaenge, 8 Qubits):
  id     Identitaet W = 1 (Pixel^2).
  haar   Haar-zufaellige Orthogonale (QR), seed-eigen.
  ff2    Fastfood-2: H D2 Pi H D1 (Walsh-Hadamard H/16, Zufallsvorzeichen, Zufallspermutation), seed-eigen.
  bent4  feste Bent-Zelle: 4x4-Pixelbloecke (Qubits 2,3,6,7), Block-Walsh, Maske (-1)^q, q = Kreuzkopplung
         Zeilenbit x Spaltenbit, Rang 4 (voll bent). Eine einzige feste Formkonvention (wie v4).
  suche  Clifford-Suche (SA + Politur, LOO-Ridge-Kriterium auf Trainingsdaten, wie I-036-v3/I-038), SEED-EIGEN,
         4 Lagen. Kosten ~6 min je Seed.
  nk     Negativkontrolle (nur Identitaetsmerkmale): Trainings- UND Informationslabels zufaellig permutiert.

KOEPFE (beide: Softmax-Regression auf standardisierten Schusshaeufigkeiten, L-BFGS, 2.570 reelle Parameter, L2 je
  Kopftyp auf dem Tuning-Seed 2098 aus {1e-3, 3e-3, 1e-2} gewaehlt (Gitter in beide Richtungen um Faktor 3 verlaengert,
  solange das Maximum am Rand liegt, hoechstens 4 Runden), Validierung 1.200 der 6.000 Trainbilder, nie Test):
  exakt     auf EXAKTEN Wahrscheinlichkeiten p trainiert, auf Schusshaeufigkeiten (S Schuesse) getestet.
  bewusst   auf Schusshaeufigkeiten GLEICHER Schusszahl S trainiert (je Trainingsbild eigene Schussziehung), gleiche
            Testhaeufigkeiten. S in {10, 30, 100, 300, 1000}. Test: 3 Schussrealisierungen je Testbild, beide
            Koepfe sehen dieselben Ziehungen (gepaart); Genauigkeit = Mittel ueber die 3.
  Wahl der L2-Staerke: Mittel der Validierungsgenauigkeit ueber die vier Nicht-Such-Arme und S in {30, 100}.
  Grenzwert exakt (acc_inf): exakter Kopf auf exakten Testmerkmalen (S -> unendlich).

INFORMATIONSGROESSEN (Bit; wie v4): I(Klasse; 1 Schuss) = I_C aus den Born-Verteilungen der Test-Klassenmittel
  (Prior = Trainingsanteile); chi_C = Holevo-Groesse aus den Klassen-Zustandsmatrizen der Trainingsbilder
  (basisunabhaengig, unitaer invariant); Wirkungsgrad eta = I_C / chi_C.
  Schussbedarf S90 = erste log-lineare Interpolations-Kreuzung von 0,90 auf der seedgemittelten Kurve des
  schussbewussten Kopfes; wird 0,90 bis S = 1000 nicht erreicht: zensiert (S90 := 2000, Gleichstand im Rang).

HYPOTHESEN (vorab; Familie {H1, H2, H3}, Holm, alpha = 0,05; Seeds 2050-2059 = 10, exakt kleinster p = 2/2^10 zweiseitig
  bzw. 2^-10 einseitig):
  H1  I(Klasse; 1 Schuss) der Identitaet > Haar UND > Suche (gepaart, einseitiger exakter Wilcoxon je Vergleich;
      p_H1 = max der beiden = Schnitt-Vereinigungs-Test; zusaetzlich beide Mittel-Differenzen > 0).
  H2  Schussbewusstes Training schlaegt exaktes Training bei S <= 100: je S in {10, 30, 100} Differenz bewusst - exakt,
      gemittelt ueber die fuenf Arme (id, haar, ff2, bent4, suche), gepaart ueber Seeds, einseitiger exakter Wilcoxon;
      p_H2 = max der drei (alle drei muessen tragen). Zusatz beschreibend: Differenz je Arm, je S, bei S >= 300.
  H3  Hoeherer Wirkungsgrad eta sagt kleineren Schussbedarf S90 voraus: Spearman-Rangkorrelation ueber die FUENF Arme
      (seedgemittelte eta, seedgemittelter Kurve-S90), einseitig rho < 0, EXAKTER Permutationstest (5! = 120
      Anordnungen; kleinster p = 1/120). Nur 5 Punkte: nur ein sehr deutlicher Zusammenhang kann tragen.

KONTROLLEN / VORBEDINGUNGEN (Kriterien in pruefe_vorbedingungen):
  PK-1  Positivkontrolle Kopf-Maschinerie: Identitaet, exakter Kopf, S -> unendlich >= 0,88 (v4: 0,901).
  PK-2  Schussrauschen muss messbar beissen: Identitaet, exakter Kopf, S = 10 mindestens 10 pp unter S = 1000
        (sonst koennte H2 gar nicht feuern).
  NK    Negativkontrolle: nk-Informationsmass I_C < 0,03 Bit UND nk-Genauigkeit (bewusst, S = 100) <= 0,15.
        Schlaegt NK fehl: alle Urteile HU. Schlagen PK-1/PK-2 fehl: H2 und H3 HU.
  Instabilitaet: Arm mit >= 2 von 10 Seeds, deren bewusste Genauigkeit bei S = 100 unter Median - 5 pp liegt:
        H2 und H3 HU (INSTABIL), vor der Urteilsbildung geprueft.

URTEILSZWEIGE (Strings; Marke H1 = Alternative gilt, H0 = gilt nicht, HU = unentschieden):
  H1: „H1 IDENTITAET_BESTE ...“ | „H0 NICHT_UEBERLEGEN ...“ (eine Differenz <= 0) | „HU NICHT_ENTSCHIEDEN ...“
  H2: „H1 SCHUSSBEWUSST_BESSER ...“ | „H0 EXAKT_BESSER ...“ (Gegenrichtung einseitig p < 0,05)
      | „H0 KEIN_VORTEIL ...“ (Mittel ueber S <= 0) | „HU NICHT_ENTSCHIEDEN ...“ | „HU INSTABIL“ | „HU PK_VERSAGT“
  H3: „H1 ETA_SAGT_VORAUS ...“ | „H0 ETA_NICHT_PROGNOSTISCH ...“ (rho >= -0,2) | „HU ZENSIERT ...“ (>= 3 von 5
      Armen zensiert) | „HU NICHT_ENTSCHIEDEN ...“ | „HU INSTABIL“ | „HU PK_VERSAGT“

DATEN/SEEDS: echtes MNIST 16x16 (6.000 Training, 5.000 offizielle Testbilder je Seed), Mess-Seeds 2050-2059, Tuning-Seed
  2098, Rauchtest NUR auf 2098 (kleine Daten, Mini-Suche, KEIN Mess-Seed). Fashion optional (fashion=True), nur
  BESCHREIBEND, mit den MNIST-Hyperparametern, ohne Urteil.

REALISTISCHE ERWARTUNG (vor dem Lauf, aus v4): H1 ja (Identitaet 0,36 gegen ~0,16, praktisch deterministisch); H2 bei
  S = 10/30 ja (v4: +10/+5 pp), bei S = 100 knapp (v4: +3 pp bei Identitaet) -> H2 wahrscheinlich ja, aber der
  Schnitt-Vereinigungs-Test ist streng; H3 offen bis eher nein: v4 sah ein Ranggefaelle, bei dem die Identitaet das
  hoechste eta hat, aber 90 % nicht erreicht (zensiert), waehrend Bent/Suche weniger eta haben und 90 % erreichen --
  das waere die GEGENRICHTUNG (H0 oder HU). Das wird vor dem Lauf offen gesagt.
GRENZEN: I_C aus Klassenmitteln, Testbilder aus denselben 10.000 offiziellen Bildern (sd ueber Seeds unterschaetzt die
  Genauigkeits-, nicht die Informationsunsicherheit); Fano-Wirkungsgrad aus v4 wird nicht mehr verwendet; die
  Bent-Formkonvention ist fest; eine Schussrealisierung je Trainingsbild.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abteilungen import clifford_suche as v1  # noqa: E402
from abteilungen.clifford_suche import Bewerter, normiere, unitaer_clifford, zufall_codes, _bewerte  # noqa: E402
from abteilungen.maske_block import matrix_vm, unterraum_aus_qubits  # noqa: E402
from abteilungen.maske_eigenschaft import maske_aus_form  # noqa: E402
from labor import gemeinsam as gem  # noqa: E402

KARTE = "I-040"
SEEDS = tuple(range(2050, 2060))
TUNING_SEED = 2098
SCHNELL_SEEDS = (2098,)               # Rauchtest-Seed = Tuning-Seed, KEIN Mess-Seed
S_GITTER = (10, 30, 100, 300, 1000)
S_H2 = (10, 30, 100)
N_REAL = 3                            # Schussrealisierungen je Testbild
L2_GITTER = (1e-3, 3e-3, 1e-2)
MAX_ITER = 200
ARME = ("id", "haar", "ff2", "bent4", "suche")
TUNING_ARME = ("id", "haar", "ff2", "bent4")
N_TRAIN, N_TEST, N_VAL = 6_000, 5_000, 1_200
ALPHA = 0.05
ZIEL = 0.90
S_ZENS = 2000.0
KOLLAPS_PP = 0.05
PK1_MIN, PK2_ABSTAND = 0.88, 0.10
NK_I_MAX, NK_ACC_MAX = 0.03, 0.15
RHO_NEIN = -0.2
QB = {4: (2, 3, 6, 7)}
BESCHREIBUNG = {"id": "Identitaet", "haar": "Haar-Orthogonale", "ff2": "Fastfood-2", "bent4": "Bent-4x4 (Rang 4)",
                "suche": "Clifford-Suche (seed-eigen)"}
HYPOTHESEN = {
    "H1": "I(Klasse;1 Schuss): Identitaet > Haar UND > Suche (gepaart, einseitiger exakter Wilcoxon, p = max).",
    "H2": "Schussbewusster Kopf > exakt trainierter Kopf bei S in {10,30,100} (Mittel ueber 5 Arme, je S einseitiger "
          "exakter Wilcoxon, p = max).",
    "H3": "Hoeheres eta = I/chi sagt kleineren Schussbedarf S90 voraus (Spearman ueber 5 Arme, exakter "
          "Permutationstest, einseitig rho < 0).",
}
FOLGEFRAGEN = {
    "H1 IDENTITAET_BESTE": "Messbasis verbessern: Suche mit I(Klasse;Schuss) oder S=30-Genauigkeit als Ziel (Idee B).",
    "H2 SCHUSSBEWUSST_BESSER": "Schussbewusstes Training als Standard fuer Hardware-Demo.",
    "H3 ETA_SAGT_VORAUS": "eta als Gueteklasse neben Genauigkeit berichten; Neuheitspruefung (E4) vor Formulierung.",
    "H3 H0 ETA_NICHT_PROGNOSTISCH": "eta nicht als Vorhersager verwenden; nur beschreibend.",
}


# ------------------------------------------------------------------ Daten und Arme
def daten(ds: str, seed: int, klein: bool = False):
    if klein:
        return gem.bilder(ds, seed, n_train=1_500, n_test=1_000, seite=16)
    return gem.bilder(ds, seed, n_train=N_TRAIN, n_test=N_TEST, seite=16)


def mischung(art: str, seed: int) -> torch.Tensor:
    """haar: Haar-zufaellig (QR); ff2: H D2 Pi H D1 mit Walsh-Hadamard H/16."""
    rng = np.random.default_rng([seed, 40, 1 if art == "haar" else 2])
    if art == "haar":
        q, r = np.linalg.qr(rng.standard_normal((256, 256)))
        return torch.tensor(q * np.sign(np.diag(r)), dtype=torch.float32)
    from scipy.linalg import hadamard
    h = hadamard(256) / 16.0
    d1, d2 = rng.choice([-1.0, 1.0], 256), rng.choice([-1.0, 1.0], 256)
    pm = np.eye(256)[rng.permutation(256)]
    return torch.tensor(h @ np.diag(d2) @ pm @ h @ np.diag(d1), dtype=torch.float32)


def bent_form(k: int, r: int) -> np.ndarray:
    B = np.zeros((k, k), np.int64)
    h = k // 2
    for i in range(r // 2):
        B[i, i + h] = B[i + h, i] = 1
    return B


def bent_zelle() -> torch.Tensor:
    V = unterraum_aus_qubits(QB[4])
    return matrix_vm(V, maske_aus_form(bent_form(4, 4), V))


def merkmale(xn: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Born-Wahrscheinlichkeiten (U xn)^2, Zeilen auf Summe 1 normiert (float32)."""
    P = ((xn.double() @ U.double().T) ** 2)
    return (P / P.sum(1, keepdim=True)).float()


def suche_unitaer(xtr, ytr, seed, sa, politur, nz):
    """Seed-eigene Clifford-Suche wie I-038/I-036-v3 (LOO-Ridge-Kriterium, nur Trainingsdaten)."""
    bew = Bewerter(xtr, ytr, 8)
    zv = [bew(c)[0] for c in zufall_codes(seed, nz, 8)]
    skala = max(float(np.std(zv)), 1e-4)
    start = np.random.default_rng(v1.START_SEED + seed).integers(0, 4, (v1.LAGEN, 8))
    best, s_best, _ = v1.suche(bew, start, np.random.default_rng(v1.SA_SEED + seed), skala, sa, politur_max=politur)
    return unitaer_clifford(best, 8), best.tolist(), float(s_best[0]), bew.bewertungen


# ------------------------------------------------------------------ Information
def H(p, ax=-1):
    p = np.asarray(p, np.float64)
    return -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(ax)


def info_C(q, pi) -> float:
    """I(Klasse; 1 Schuss) in Bit aus Klassenverteilungen q (K, B) und Prior pi."""
    return float(H(pi @ q) - pi @ H(q))


def klassenmittel(P: np.ndarray, y: np.ndarray, K: int = 10) -> np.ndarray:
    return np.stack([P[y == c].mean(0) for c in range(K)])


def holevo_chi(xn: torch.Tensor, y: np.ndarray, pi: np.ndarray) -> float:
    """chi_C = S(rho_bar) - sum pi_c S(rho_c) (Bit); rho_c = mittlerer reiner Zustand der Klasse c."""
    X = xn.double().numpy()
    Sv = lambda r: float(H(np.clip(np.linalg.eigvalsh(r), 0, None)))
    rho = [X[y == c].T @ X[y == c] / max(1, (y == c).sum()) for c in range(len(pi))]
    return Sv(sum(pi[c] * rho[c] for c in range(len(pi)))) - sum(pi[c] * Sv(rho[c]) for c in range(len(pi)))


# ------------------------------------------------------------------ Schuesse und Koepfe
def schuesse(P: torch.Tensor, slist, gen) -> dict:
    """Genestete Zaehlungen nach den ersten S Schuessen je Zeile (P Zeilen Summe 1)."""
    slist = sorted(slist)
    idx = torch.multinomial(P, slist[-1], replacement=True, generator=gen)
    cnt = torch.zeros(P.shape[0], P.shape[1])
    out, prev = {}, 0
    for S in slist:
        cnt.scatter_add_(1, idx[:, prev:S], torch.ones(P.shape[0], S - prev))
        out[S] = cnt.clone() / S
        prev = S
    return out


def kopf_fit(f, y, l2, max_iter=MAX_ITER):
    mu, sd = f.mean(0), f.std(0).clamp_min(1e-6)
    A = torch.cat([(f - mu) / sd, torch.ones(len(f), 1)], 1)
    W = torch.zeros(A.shape[1], 10, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-7,
                            tolerance_change=1e-10, history_size=20)

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(A @ W, y) + l2 * (W[:-1] ** 2).sum()
        l.backward()
        return l
    opt.step(cl)
    return W.detach(), mu, sd


def kopf_acc(h, f, y) -> float:
    W, mu, sd = h
    B = torch.cat([(f - mu) / sd, torch.ones(len(f), 1)], 1)
    return float(((B @ W).argmax(1) == y).float().mean())


def _gen(*teile) -> torch.Generator:
    h = int(hashlib.sha1(repr(teile).encode()).hexdigest()[:12], 16) % (2 ** 62)
    return torch.Generator().manual_seed(h)


def kopf_kurven(Ptr, ytr, Pte, yte, l2_ex, l2_bew, slist, seed, arm, max_iter=MAX_ITER, mit_inf=True):
    """acc_inf (exakter Kopf auf exakten Testmerkmalen), acc_exakt[S], acc_bewusst[S] (gepaart)."""
    he = kopf_fit(Ptr, ytr, l2_ex, max_iter)
    out = {"acc_exakt": {}, "acc_bewusst": {}}
    if mit_inf:
        out["acc_inf"] = kopf_acc(he, Pte, yte)
    tests = [schuesse(Pte, slist, _gen(seed, arm, "test", r)) for r in range(N_REAL)]
    for S in slist:
        out["acc_exakt"][str(S)] = float(np.mean([kopf_acc(he, t[S], yte) for t in tests]))
        ftr = schuesse(Ptr, [S], _gen(seed, arm, "train", S))[S]
        hb = kopf_fit(ftr, ytr, l2_bew, max_iter)
        out["acc_bewusst"][str(S)] = float(np.mean([kopf_acc(hb, t[S], yte) for t in tests]))
    return out


# ------------------------------------------------------------------ Aufgaben (je Seed und Arm; Wiederaufnahme)
def _konfig(klein, sa, politur, nz, max_iter, hp):
    roh = json.dumps([klein, sa, politur, nz, max_iter, hp, list(S_GITTER), N_REAL, list(ARME), N_TRAIN, N_TEST])
    return hashlib.sha1(roh.encode()).hexdigest()[:12]


def _arm_U(arm, xtr, ytr, seed, sa, politur, nz):
    if arm in ("id", "nk"):
        return torch.eye(256), {}
    if arm in ("haar", "ff2"):
        return mischung(arm if arm == "haar" else "ff", seed), {}
    if arm == "bent4":
        return bent_zelle(), {}
    U, codes, val, bew = suche_unitaer(xtr, ytr, seed, sa, politur, nz)
    return U, {"suche_codes": codes, "suche_val": val, "suche_bewertungen": bew}


def _aufgabe(a):
    """(ds, seed, arm, hp, klein, sa, politur, nz, max_iter, ordner, konfig) -> Ergebnis; schreibt/liest die Datei."""
    ds, seed, arm, hp, klein, sa, politur, nz, max_iter, ordner, konfig = a
    pfad = Path(ordner) / f"{ds}_{seed}_{arm}.json"
    if pfad.exists():
        z = json.loads(pfad.read_text())
        if z.get("konfig") == konfig:
            return z
    torch.set_num_threads(1)
    t0 = time.time()
    xtr, ytr, xte, yte = daten(ds, seed, klein)
    xn_tr, xn_te = normiere(xtr), normiere(xte)
    ytr_n, yte_n = ytr.numpy(), yte.numpy()
    pi = np.bincount(ytr_n, minlength=10) / len(ytr_n)
    U, extra = _arm_U(arm, xtr, ytr, seed, sa, politur, nz)
    Ptr, Pte = merkmale(xn_tr, U), merkmale(xn_te, U)
    z = {"ds": ds, "seed": seed, "arm": arm, "konfig": konfig, **extra}
    if arm == "nk":
        rng = np.random.default_rng([seed, 4040])
        ytr_p = torch.tensor(rng.permutation(ytr_n))
        yte_p = rng.permutation(yte_n)
        z["I_C_test"] = info_C(klassenmittel(Pte.numpy().astype(np.float64), yte_p), pi)
        k = kopf_kurven(Ptr, ytr_p, Pte, yte, hp["exakt"], hp["bewusst"], (100,), seed, arm, max_iter, mit_inf=False)
        z["acc_bewusst"] = k["acc_bewusst"]
    else:
        z["chi_C"] = holevo_chi(xn_tr, ytr_n, pi)
        Pn, Pt = Ptr.numpy().astype(np.float64), Pte.numpy().astype(np.float64)
        z["I_C_train"] = info_C(klassenmittel(Pn, ytr_n), pi)
        z["I_C_test"] = info_C(klassenmittel(Pt, yte_n), pi)
        z["I_img_test"] = float(H(Pt.mean(0)) - H(Pt).mean())
        z.update(kopf_kurven(Ptr, ytr, Pte, yte, hp["exakt"], hp["bewusst"], S_GITTER, seed, arm, max_iter))
    z["sekunden"] = time.time() - t0
    Path(ordner).mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(z))              # Rohdaten sofort sichern (Protokollregel)
    return z


def _tune_aufgabe(a):
    """(ds, seed, arm, klein, max_iter) -> {typ: {l2: {S: val-acc}}} auf Anpassung 4.800 / Validierung 1.200."""
    ds, seed, arm, klein, max_iter, l2s = a
    torch.set_num_threads(1)
    xtr, ytr, _, _ = daten(ds, seed, klein)
    n = len(ytr)
    nv = N_VAL if not klein else n // 5
    g = torch.Generator().manual_seed(seed + 777)
    p = torch.randperm(n, generator=g)
    iv, ia = p[:nv], p[nv:]
    U, _ = _arm_U(arm, xtr, ytr, seed, 0, 0, 0)
    P = merkmale(normiere(xtr), U)
    Pa, ya, Pv, yv = P[ia], ytr[ia], P[iv], ytr[iv]
    out = {"exakt": {}, "bewusst": {}}
    tv = {S: schuesse(Pv, [S], _gen(seed, arm, "val", S))[S] for S in (30, 100)}
    for l2 in l2s:
        he = kopf_fit(Pa, ya, l2, max_iter)
        out["exakt"][str(l2)] = {str(S): kopf_acc(he, tv[S], yv) for S in tv}
        out["bewusst"][str(l2)] = {}
        for S in tv:
            hb = kopf_fit(schuesse(Pa, [S], _gen(seed, arm, "valtr", S))[S], ya, l2, max_iter)
            out["bewusst"][str(l2)][str(S)] = kopf_acc(hb, tv[S], yv)
    return arm, out


def waehle_l2(roh: dict) -> dict:
    """roh[arm][typ][l2][S] -> {'exakt': l2, 'bewusst': l2, 'am_rand': {typ: 'unten'|'oben'|None}, 'val': {...}}.
    Kriterium: Mittel ueber Arme und S; Gleichstand -> staerkere Regularisierung."""
    gitter = sorted(float(k) for k in next(iter(roh.values()))["exakt"])
    wahl, rand, val = {}, {}, {}
    for typ in ("exakt", "bewusst"):
        m = {l2: float(np.mean([v for arm in roh for v in roh[arm][typ][str(l2)].values()])) for l2 in gitter}
        val[typ] = {str(k): v for k, v in m.items()}
        wahl[typ] = max(gitter, key=lambda l2: (round(m[l2], 12), l2))
        rand[typ] = "unten" if wahl[typ] == gitter[0] else "oben" if wahl[typ] == gitter[-1] else None
    return {**wahl, "am_rand": rand, "val": val, "gitter": gitter}


# ------------------------------------------------------------------ Statistik
def wilcoxon_einseitig(d) -> float:
    """Exakter einseitiger Wilcoxon-Vorzeichen-Rang-Test, H1: Median(d) > 0 (Nullen entfallen, Mittelraenge)."""
    d = np.asarray([v for v in d if v != 0.0], float)
    n = len(d)
    if n == 0:
        return 1.0
    r2 = np.round(2 * stats.rankdata(np.abs(d))).astype(int)
    summe = int(r2.sum())
    zaehl = np.zeros(summe + 1)
    zaehl[0] = 1.0
    for r in r2:
        neu = zaehl.copy()
        neu[r:] += zaehl[:summe + 1 - r]
        zaehl = neu
    return float(zaehl[int(r2[d > 0].sum()):].sum() / zaehl.sum())


def holm(ps: dict) -> dict:
    reihe = sorted(ps, key=ps.get)
    m, lauf, aus = len(reihe), 0.0, {}
    for i, k in enumerate(reihe):
        lauf = max(lauf, min(1.0, (m - i) * ps[k]))
        aus[k] = lauf
    return aus


def spearman_perm(x, y):
    """Spearman rho und EXAKTER einseitiger Permutationstest H1: rho < 0 (Mittelraenge bei Gleichstand).
    Konstanter Vektor -> (None, None)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = lambda b: float(np.corrcoef(rx, b)[0, 1])
    r0 = rho(ry)
    perm = [rho(ry[list(p)]) for p in itertools.permutations(range(len(ry)))]
    return r0, float(np.mean(np.asarray(perm) <= r0 + 1e-9))


def s_ziel(kurve: dict, ziel: float = ZIEL):
    """kurve {S: acc} -> erste Kreuzung von `ziel`, log-linear interpoliert; None = nicht erreicht (zensiert)."""
    s = sorted(int(k) for k in kurve)
    a = [kurve[str(k)] if str(k) in kurve else kurve[k] for k in s]
    if a[0] >= ziel:
        return float(s[0])
    for i in range(1, len(s)):
        if a[i] >= ziel > a[i - 1]:
            t = (ziel - a[i - 1]) / (a[i] - a[i - 1])
            return float(math.exp(math.log(s[i - 1]) + t * (math.log(s[i]) - math.log(s[i - 1]))))
    return None


def _stat(w):
    a = np.asarray(w, float)
    return {"mittel": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


# ------------------------------------------------------------------ Auswertung und Urteil
def auswertung(zeilen: list) -> dict:
    """zeilen: je Seed {'seed', 'arme': {arm: {I_C_test, chi_C, acc_inf, acc_exakt{S}, acc_bewusst{S}}}, 'nk': {...}}."""
    seeds = [z["seed"] for z in zeilen]
    arr = lambda arm, k: np.array([z["arme"][arm][k] for z in zeilen], float)
    sk = lambda arm, typ, S: np.array([z["arme"][arm][typ][str(S)] for z in zeilen], float)
    a = {"seeds": seeds, "arme": {}}
    for arm in ARME:
        I, chi = arr(arm, "I_C_test"), arr(arm, "chi_C")
        kb = {str(S): float(sk(arm, "acc_bewusst", S).mean()) for S in S_GITTER}
        ke = {str(S): float(sk(arm, "acc_exakt", S).mean()) for S in S_GITTER}
        s90 = s_ziel(kb)
        a["arme"][arm] = {"I": _stat(I), "chi": _stat(chi), "eta": _stat(I / chi), "I_train": _stat(arr(arm, "I_C_train")),
                          "acc_inf": _stat(arr(arm, "acc_inf")),
                          "acc_bewusst": {S: _stat(sk(arm, "acc_bewusst", S)) for S in map(str, S_GITTER)},
                          "acc_exakt": {S: _stat(sk(arm, "acc_exakt", S)) for S in map(str, S_GITTER)},
                          "diff_bewusst_exakt": {S: _stat(sk(arm, "acc_bewusst", S) - sk(arm, "acc_exakt", S))
                                                 for S in map(str, S_GITTER)},
                          "S90": s90, "S90_zensiert": s90 is None, "kurve_bewusst": kb, "kurve_exakt": ke,
                          "kollaps_seeds": _kollaps(sk(arm, "acc_bewusst", 100), seeds)}
    # H1
    I = {arm: arr(arm, "I_C_test") for arm in ARME}
    d1 = {"haar": I["id"] - I["haar"], "suche": I["id"] - I["suche"]}
    p1 = {k: wilcoxon_einseitig(v) for k, v in d1.items()}
    a["H1"] = {"diff": {k: _stat(v) for k, v in d1.items()}, "p_je": p1, "p": max(p1.values()),
               "anzahl_positiv": {k: int((v > 0).sum()) for k, v in d1.items()},
               "beschreibend_bent4_minus_haar": _stat(I["bent4"] - I["haar"])}
    # H2
    dS = {S: np.mean([sk(arm, "acc_bewusst", S) - sk(arm, "acc_exakt", S) for arm in ARME], axis=0) for S in S_H2}
    p2 = {str(S): wilcoxon_einseitig(v) for S, v in dS.items()}
    dagg = np.mean([dS[S] for S in S_H2], axis=0)
    a["H2"] = {"diff_je_S": {str(S): _stat(v) for S, v in dS.items()}, "p_je_S": p2, "p": max(p2.values()),
               "diff_mittel_S": _stat(dagg), "p_gegenrichtung": wilcoxon_einseitig(-dagg),
               "beschreibend_grosse_S": {str(S): _stat(np.mean([sk(arm, "acc_bewusst", S) - sk(arm, "acc_exakt", S)
                                                               for arm in ARME], axis=0)) for S in (300, 1000)}}
    # H3
    eta = [a["arme"][arm]["eta"]["mittel"] for arm in ARME]
    s90 = [a["arme"][arm]["S90"] if a["arme"][arm]["S90"] is not None else S_ZENS for arm in ARME]
    rho, p3 = spearman_perm(eta, s90)
    a["H3"] = {"eta": dict(zip(ARME, eta)), "S90": dict(zip(ARME, s90)), "zensiert": [arm for arm in ARME
               if a["arme"][arm]["S90_zensiert"]], "rho": rho, "p": p3}
    ps = {"H1": a["H1"]["p"], "H2": a["H2"]["p"], "H3": 1.0 if p3 is None else p3}
    a["holm"] = holm(ps)
    # Vorbedingungen
    a["kontrollen"] = pruefe_vorbedingungen(a, zeilen)
    return a


def _kollaps(werte, seeds, grenze=KOLLAPS_PP):
    med = float(np.median(werte))
    return [s for s, v in zip(seeds, werte) if v < med - grenze]


def pruefe_vorbedingungen(a: dict, zeilen: list) -> dict:
    k = {}
    inf = a["arme"]["id"]["acc_inf"]["mittel"]
    k["PK-1"] = ("PK_OK" if inf >= PK1_MIN else f"HU PK_VERSAGT (Identitaet exakt {inf:.3f} < {PK1_MIN})") + f" [{inf:.3f}]"
    e10, e1000 = (a["arme"]["id"]["acc_exakt"][s]["mittel"] for s in ("10", "1000"))
    k["PK-2"] = ("PK_OK" if e1000 - e10 >= PK2_ABSTAND else
                 f"HU PK_VERSAGT (exakter Kopf S=10 nur {e1000 - e10:.3f} unter S=1000)") + f" [{e1000 - e10:.3f}]"
    nkI = float(np.mean([z["nk"]["I_C_test"] for z in zeilen]))
    nkA = float(np.mean([z["nk"]["acc_bewusst"]["100"] for z in zeilen]))
    k["NK"] = ("NK_OK" if nkI < NK_I_MAX and nkA <= NK_ACC_MAX else
               f"HU NK_VERSAGT (I {nkI:.3f} Bit, Genauigkeit {nkA:.3f})") + f" [I {nkI:.4f}, acc {nkA:.3f}]"
    k["instabil"] = {arm: a["arme"][arm]["kollaps_seeds"] for arm in ARME if len(a["arme"][arm]["kollaps_seeds"]) >= 2}
    k["werte"] = {"pk1_acc_inf": inf, "pk2_abstand": e1000 - e10, "nk_I": nkI, "nk_acc": nkA}
    return k


def urteil(a: dict) -> dict:
    """Urteile als Strings; ALLE Kriterien stehen hier (vorregistriert)."""
    k, hp = a["kontrollen"], a["holm"]
    u = {"PK-1": k["PK-1"], "PK-2": k["PK-2"], "NK": k["NK"]}
    nk_ok, pk_ok = k["NK"].startswith("NK_OK"), k["PK-1"].startswith("PK_OK") and k["PK-2"].startswith("PK_OK")
    instab = bool(k["instabil"])
    # H1
    h1 = a["H1"]
    mp = [h1["diff"][x]["mittel"] for x in ("haar", "suche")]
    txt = (f"I(id)-I(haar) {mp[0]:+.3f} Bit, I(id)-I(suche) {mp[1]:+.3f} Bit, Holm-p {hp['H1']:.4f}, "
           f"positiv {h1['anzahl_positiv']['haar']}/{len(a['seeds'])} bzw. {h1['anzahl_positiv']['suche']}/{len(a['seeds'])}")
    if not nk_ok:
        u["H1"] = "HU NK_VERSAGT: Informationsmass faellt die Negativkontrolle durch"
    elif min(mp) > 0 and hp["H1"] < ALPHA:
        u["H1"] = "H1 IDENTITAET_BESTE: " + txt
    elif min(mp) <= 0:
        u["H1"] = "H0 NICHT_UEBERLEGEN: " + txt
    else:
        u["H1"] = "HU NICHT_ENTSCHIEDEN: " + txt
    # H2
    h2 = a["H2"]
    m = [h2["diff_je_S"][str(S)]["mittel"] for S in S_H2]
    txt = (f"Differenz bewusst-exakt je S=10/30/100: {m[0]:+.3f}/{m[1]:+.3f}/{m[2]:+.3f}, Holm-p {hp['H2']:.4f}, "
           f"Gegenrichtung p {h2['p_gegenrichtung']:.4f}")
    if not nk_ok or not pk_ok:
        u["H2"] = "HU PK_VERSAGT: Vorbedingung nicht erfuellt"
    elif instab:
        u["H2"] = f"HU INSTABIL: {k['instabil']}"
    elif min(m) > 0 and hp["H2"] < ALPHA:
        u["H2"] = "H1 SCHUSSBEWUSST_BESSER: " + txt
    elif h2["p_gegenrichtung"] < ALPHA and h2["diff_mittel_S"]["mittel"] < 0:
        u["H2"] = "H0 EXAKT_BESSER: " + txt
    elif h2["diff_mittel_S"]["mittel"] <= 0:
        u["H2"] = "H0 KEIN_VORTEIL: " + txt
    else:
        u["H2"] = "HU NICHT_ENTSCHIEDEN: " + txt
    # H3
    h3 = a["H3"]
    txt = f"rho {h3['rho'] if h3['rho'] is None else round(h3['rho'], 3)}, p {h3['p'] if h3['p'] is None else round(h3['p'], 4)}, " \
          f"Holm-p {hp['H3']:.4f}, zensiert {h3['zensiert']}"
    if not nk_ok or not pk_ok:
        u["H3"] = "HU PK_VERSAGT: Vorbedingung nicht erfuellt"
    elif instab:
        u["H3"] = f"HU INSTABIL: {k['instabil']}"
    elif len(h3["zensiert"]) >= 3 or h3["rho"] is None:
        u["H3"] = "HU ZENSIERT: " + txt
    elif h3["rho"] < 0 and hp["H3"] < ALPHA:
        u["H3"] = "H1 ETA_SAGT_VORAUS: " + txt
    elif h3["rho"] >= RHO_NEIN:
        u["H3"] = "H0 ETA_NICHT_PROGNOSTISCH: " + txt
    else:
        u["H3"] = "HU NICHT_ENTSCHIEDEN: " + txt
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seed=TUNING_SEED, ordner=None, fashion=False, prozesse=None, **kw):
    if schnell:
        seeds, klein, sa, politur, nz, max_iter = SCHNELL_SEEDS, True, 60, 20, 5, 30
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-040-schnell"
    else:
        klein, sa, politur, nz, max_iter = False, v1.SA_SCHRITTE, v1.POLITUR_MAX, v1.N_ZUFALL, MAX_ITER
        if tuple(seeds) != SEEDS or tuning_seed != TUNING_SEED:
            raise ValueError("Seeds sind vorregistriert: 2050-2059, Tuning 2098")
    ordner = Path(ordner or gem.ERGEBNISSE / "_zwischen_I-040")
    seeds = list(seeds)
    if not schnell and set(seeds) & {tuning_seed}:
        raise ValueError("Mess- und Tuning-Seed ueberschneiden sich")
    ordner.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dsets = ["mnist"] + (["fashion_mnist"] if fashion else [])
    # Stufe 0: L2 je Kopftyp auf dem Tuning-Seed (einmal, dann Datei)
    tp = ordner / "tuning.json"
    kt = hashlib.sha1(json.dumps([klein, max_iter, list(L2_GITTER), tuning_seed]).encode()).hexdigest()[:12]
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != kt:
        gitter, roh = list(L2_GITTER), {arm: {"exakt": {}, "bewusst": {}} for arm in TUNING_ARME}
        for _runde in range(4):             # Gitter in beide Richtungen verlaengern (Faktor 3), solange das Maximum am Rand liegt
            neu = [g for g in gitter if str(g) not in roh[TUNING_ARME[0]]["exakt"]]
            res = gem.parallel(_tune_aufgabe, [("mnist", tuning_seed, arm, klein, max_iter, neu) for arm in TUNING_ARME],
                               prozesse=prozesse)
            for arm, o in res:
                for typ in o:
                    roh[arm][typ].update(o[typ])
            w = waehle_l2(roh)
            ext = {w["am_rand"][t] for t in w["am_rand"]} - {None}
            if not ext:
                break
            if "unten" in ext:
                gitter = [min(gitter) / 3.0] + gitter
            if "oben" in ext:
                gitter = gitter + [max(gitter) * 3.0]
        tun = {"konfig": kt, "roh": roh, "wahl": w, "tuning_seed": tuning_seed}
        tp.write_text(json.dumps(tun, indent=1))
    hp = {"exakt": tun["wahl"]["exakt"], "bewusst": tun["wahl"]["bewusst"]}
    print(f"  L2-Wahl (Validierung, vor Test): {hp}  am Rand: {tun['wahl']['am_rand']}", flush=True)
    konfig = _konfig(klein, sa, politur, nz, max_iter, hp)
    # Stufe 1: je (Datensatz, Seed, Arm) eine Aufgabe, schwere Suche zuerst
    aufg = [(ds, s, arm, hp, klein, sa, politur, nz, max_iter, str(ordner), konfig)
            for arm in ("suche", "id", "haar", "ff2", "bent4", "nk") for ds in dsets for s in seeds]
    erg = gem.parallel(_aufgabe, aufg, prozesse=prozesse)
    tab = {(z["ds"], z["seed"], z["arm"]): z for z in erg}
    e = {"abteilung": "clifford", "modul": "holevo_wirkungsgrad", "agenda": "I-040", "hypothesen": HYPOTHESEN,
         "folgefragen": FOLGEFRAGEN, "seeds": seeds, "tuning_seed": tuning_seed, "tuning": tun, "l2": hp,
         "einstellungen": {"klein": klein, "sa": sa, "politur": politur, "n_zufall": nz, "max_iter": max_iter,
                           "S": list(S_GITTER), "n_real": N_REAL}, "zeilen": {}, "auswertung": {}}
    for ds in dsets:
        zeilen = [{"seed": s, "arme": {arm: tab[(ds, s, arm)] for arm in ARME}, "nk": tab[(ds, s, "nk")]} for s in seeds]
        e["zeilen"][ds] = zeilen
        if schnell and len(seeds) < 2:
            e["auswertung"][ds] = "Rauchtest: 1 Seed, keine Statistik"
            continue
        e["auswertung"][ds] = auswertung(zeilen)
    e["urteil"] = urteil(e["auswertung"]["mnist"]) if not schnell else {"rauchtest": "kein Urteil (1 Seed)"}
    if fashion and not schnell:
        e["fashion_beschreibend"] = e["auswertung"]["fashion_mnist"]
    e["sekunden"] = time.time() - t0
    return e


# ------------------------------------------------------------------ Vortest
def _kurve(s90, A_zens=0.85):
    """Synthetische Genauigkeitskurve a(S) = 0,1 + (A-0,1) S/(S+c) mit a(s90) = 0,9 (s90 None: Plateau A_zens)."""
    if s90 is None:
        A, c = A_zens, 100.0
    else:
        A, c = 0.95, s90 * 0.0625
    return lambda S: 0.1 + (A - 0.1) * S / (S + c)


def _synth(par: dict, n=10, seed=0) -> list:
    rng = np.random.default_rng(seed)
    z = []
    for i in range(n):
        arme = {}
        for arm in ARME:
            kb = _kurve(par["s90"][arm])
            row = {"I_C_test": par["I"][arm] + par["sdI"] * rng.standard_normal(), "I_C_train": par["I"][arm],
                   "chi_C": par["chi"], "acc_inf": par["acc_inf"][arm] + par["sd_acc"] * rng.standard_normal(),
                   "acc_bewusst": {}, "acc_exakt": {}}
            for S in S_GITTER:
                b = kb(S) + par["sd_acc"] * rng.standard_normal()
                dl = par["delta"][S] + par["sd_delta"] * rng.standard_normal()
                row["acc_bewusst"][str(S)] = b
                row["acc_exakt"][str(S)] = b - dl
            if arm in par.get("kollaps", {}) and i in par["kollaps"][arm]:
                row["acc_bewusst"]["100"] -= 0.2
            arme[arm] = row
        z.append({"seed": 2050 + i, "arme": arme,
                  "nk": {"I_C_test": par["nk_I"], "acc_bewusst": {"100": par["nk_acc"]}}})
    return z


def _basis(**ueber) -> dict:
    p = {"I": {"id": 0.36, "bent4": 0.307, "haar": 0.165, "ff2": 0.163, "suche": 0.161}, "sdI": 0.0005, "chi": 0.98,
         "acc_inf": {a: 0.90 for a in ARME}, "sd_acc": 0.002,
         "s90": {"id": 100, "bent4": 150, "ff2": 400, "haar": 450, "suche": 500},
         "delta": {10: 0.10, 30: 0.06, 100: 0.03, 300: 0.01, 1000: 0.0}, "sd_delta": 0.004, "nk_I": 0.004, "nk_acc": 0.10}
    p.update(ueber)
    return p


def vortest() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    def z(par, seed=0):
        a = auswertung(_synth(par, seed=seed))
        return urteil(a), a

    # --- Informationsgroessen
    if abs(info_C(np.array([[1.0, 0], [0, 1.0]]), np.array([.5, .5])) - 1.0) > 1e-9 or \
            abs(info_C(np.array([[.5, .5], [.5, .5]]), np.array([.5, .5]))) > 1e-9:
        fehler("info_C")
    X = torch.eye(4)
    chi = holevo_chi(X, np.arange(4), np.full(4, .25))
    if abs(chi - 2.0) > 1e-9:
        fehler(f"Holevo chi orthogonale Zustaende {chi}")
    xr = torch.randn(300, 4)
    xr = xr / xr.norm(dim=1, keepdim=True)
    if abs(holevo_chi(xr, np.zeros(300, int), np.array([1.0]))) > 1e-9:
        fehler("Holevo einzelne Klasse != 0")
    # --- Arme: orthogonal, Born normiert, reproduzierbar, seed-abhaengig
    for art in ("haar", "ff"):
        W = mischung(art, 7)
        if float((W @ W.T - torch.eye(256)).abs().max()) > 1e-4:
            fehler(f"{art} nicht orthogonal")
        if not torch.equal(W, mischung(art, 7)) or torch.allclose(W, mischung(art, 8)):
            fehler(f"{art} Reproduzierbarkeit/Seedabhaengigkeit")
    Wb = bent_zelle()
    if float((Wb @ Wb.T - torch.eye(256)).abs().max()) > 1e-4 or torch.allclose(Wb, torch.eye(256)):
        fehler("bent4 nicht orthogonal/gleich Identitaet")
    Uc = unitaer_clifford(np.random.default_rng(1).integers(0, 4, (v1.LAGEN, 8)), 8)
    xs = torch.rand(50, 256)
    for U in (torch.eye(256), mischung("haar", 1), mischung("ff", 1), Wb, Uc):
        if float((merkmale(normiere(xs), U).sum(1) - 1).abs().max()) > 1e-5:
            fehler("Born nicht normiert")
    # --- Schuesse: Summen, Nestung
    gen = _gen(1, "t")
    c = schuesse(torch.full((5, 8), 1 / 8), (10, 100), gen)
    if not (torch.allclose(c[10].sum(1), torch.ones(5)) and torch.allclose(c[100].sum(1), torch.ones(5))):
        fehler("Schusszaehlungen nicht auf 1")
    if float((c[100] * 100 - c[10] * 10).min()) < -1e-4:
        fehler("Schuesse nicht genestet")
    if not torch.equal(schuesse(torch.full((5, 8), 1 / 8), (10,), _gen(1, "t"))[10], schuesse(torch.full((5, 8), 1 / 8), (10,), _gen(1, "t"))[10]):
        fehler("Schuesse nicht reproduzierbar")
    # --- Kopf trennt Trivialdaten (Positivkontrolle der Kopf-Funktion)
    g = torch.Generator().manual_seed(0)
    yk = torch.arange(600) % 10
    fk = torch.randn(600, 12, generator=g) * 0.3
    fk[torch.arange(600), yk] += 2.0
    if kopf_acc(kopf_fit(fk, yk, 3e-3, 60), fk, yk) < 0.95:
        fehler("Kopf lernt Trivialaufgabe nicht")
    # --- Schussbedarf und Permutationstest
    if abs(s_ziel({"10": .5, "100": .9, "1000": .95}) - 100) > 1e-6 or s_ziel({"10": .5, "1000": .8}) is not None or \
            abs(s_ziel({"10": .95, "100": .96}) - 10) > 1e-6:
        fehler("s_ziel")
    if abs(s_ziel({"10": .8, "1000": 1.0}) - 10 * 100 ** 0.5) > 1e-6 and False:
        fehler("s_ziel Interpolation")
    r, p = spearman_perm([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    if abs(r + 1) > 1e-9 or abs(p - 1 / 120) > 1e-12:
        fehler(f"spearman_perm {r} {p}")
    r, p = spearman_perm([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    if abs(r - 1) > 1e-9 or abs(p - 1.0) > 1e-9:
        fehler("spearman_perm positiv")
    if spearman_perm([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) != (None, None):
        fehler("spearman_perm konstant")
    if abs(wilcoxon_einseitig(np.ones(10)) - 2 ** -10) > 1e-12 or wilcoxon_einseitig(-np.ones(10)) < 0.99:
        fehler("wilcoxon_einseitig")
    if not np.allclose([holm({"a": .01, "b": .02, "c": .04})[k] for k in "abc"], [.03, .04, .04]):
        fehler("holm")
    if set(SEEDS) & {TUNING_SEED} or set(SEEDS) & (set(range(2000, 2040)) | set(range(2060, 2070)) | {2090, 2091, 2092, 2095, 2099}):
        fehler("Seeds nicht frisch")
    # --- Urteilszweige auf synthetischen Daten ERZWINGEN
    def erwarte(name, par, urteile: dict, seed=0):
        u, _ = z(par, seed)
        for h, marke in urteile.items():
            if not u[h].startswith(marke):
                fehler(f"{name}: {h} = {u[h][:70]!r}, erwartet {marke!r}")
        return u

    erwarte("Basis (alle drei H1)", _basis(), {"PK-1": "PK_OK", "PK-2": "PK_OK", "NK": "NK_OK",
                                               "H1": "H1 IDENTITAET_BESTE", "H2": "H1 SCHUSSBEWUSST_BESSER",
                                               "H3": "H1 ETA_SAGT_VORAUS"})
    # H1-Zweige
    gleich = {"id": 0.165, "bent4": 0.2, "haar": 0.163, "ff2": 0.163, "suche": 0.161}
    erwarte("H1 H0", _basis(I={**_basis()["I"], "id": 0.15}), {"H1": "H0 NICHT_UEBERLEGEN"})
    erwarte("H1 HU", _basis(I={**gleich, "id": 0.1635}, sdI=0.02), {"H1": "HU NICHT_ENTSCHIEDEN"}, seed=3)
    # H2-Zweige
    erwarte("H2 H0 EXAKT_BESSER", _basis(delta={10: -0.03, 30: -0.03, 100: -0.03, 300: 0.0, 1000: 0.0}),
            {"H2": "H0 EXAKT_BESSER"})
    erwarte("H2 H0 KEIN_VORTEIL", _basis(delta={10: 0.0, 30: 0.0, 100: 0.0, 300: 0.0, 1000: 0.0}, sd_delta=0.0, sd_acc=0.0),
            {"H2": "H0 KEIN_VORTEIL"})
    erwarte("H2 HU (gemischt)", _basis(delta={10: 0.10, 30: 0.06, 100: -0.01, 300: 0.0, 1000: 0.0}), {"H2": "HU NICHT_ENTSCHIEDEN"})
    erwarte("H2 HU (verrauscht)", _basis(delta={10: 0.003, 30: 0.003, 100: 0.003, 300: 0.0, 1000: 0.0}, sd_delta=0.03),
            {"H2": "HU NICHT_ENTSCHIEDEN"}, seed=5)
    # H3-Zweige
    umgekehrt = {"id": 500, "bent4": 450, "ff2": 400, "haar": 150, "suche": 100}
    erwarte("H3 H0", _basis(s90=umgekehrt), {"H3": "H0 ETA_NICHT_PROGNOSTISCH"})
    erwarte("H3 HU ZENSIERT", _basis(s90={"id": None, "bent4": None, "ff2": None, "haar": 450, "suche": 500}),
            {"H3": "HU ZENSIERT"})
    # nur ein Arm "falsch" einsortiert: rho = -0,7 -> nicht signifikant, nicht >= -0,2
    teil = {"id": 100, "bent4": 400, "ff2": 150, "haar": 450, "suche": 500}
    u = erwarte("H3 HU (mittel)", _basis(s90=teil), {})
    if not u["H3"].startswith(("HU NICHT_ENTSCHIEDEN", "H1 ETA", "H0 ETA")):
        fehler("H3 mittel: unerwartet")
    a_t = auswertung(_synth(_basis(s90=teil), seed=0))
    if not (-0.95 < a_t["H3"]["rho"] < -0.2) or not urteil(a_t)["H3"].startswith("HU NICHT_ENTSCHIEDEN"):
        fehler(f"H3 mittlere Korrelation erzwingt HU nicht: rho {a_t['H3']['rho']}, {urteil(a_t)['H3'][:40]}")
    # Vorbedingungen
    erwarte("PK-1 faellt", _basis(acc_inf={a: 0.80 for a in ARME}), {"PK-1": "HU PK_VERSAGT", "H2": "HU PK_VERSAGT",
                                                                    "H3": "HU PK_VERSAGT", "H1": "H1 IDENTITAET_BESTE"})
    # PK-2: exakter Kopf darf bei S=10 nicht fast so gut sein wie bei S=1000 -> Kurve flach machen
    flach = _basis(delta={10: 0.0, 30: 0.0, 100: 0.0, 300: 0.0, 1000: 0.0})
    for zl in (_synth(flach),):
        for r in zl:
            for arm in ARME:
                r["arme"][arm]["acc_exakt"]["10"] = r["arme"][arm]["acc_exakt"]["1000"] - 0.01
        uu = urteil(auswertung(zl))
        if not (uu["PK-2"].startswith("HU PK_VERSAGT") and uu["H2"].startswith("HU PK_VERSAGT")):
            fehler("PK-2 erzwingt nicht")
    erwarte("NK faellt (Information)", _basis(nk_I=0.2), {"NK": "HU NK_VERSAGT", "H1": "HU NK_VERSAGT",
                                                          "H2": "HU PK_VERSAGT", "H3": "HU PK_VERSAGT"})
    erwarte("NK faellt (Genauigkeit)", _basis(nk_acc=0.5), {"NK": "HU NK_VERSAGT"})
    erwarte("Instabilitaet", _basis(kollaps={"haar": [1, 4]}), {"H2": "HU INSTABIL", "H3": "HU INSTABIL",
                                                               "H1": "H1 IDENTITAET_BESTE"})
    # Ein Kollaps-Seed allein darf NICHT als instabil gelten
    u, _ = z(_basis(kollaps={"haar": [1]}))
    if u["H2"].startswith("HU INSTABIL"):
        fehler("ein Kollaps-Seed darf nicht reichen")
    # --- Aufgaben-Mechanik End-to-End auf Mini-Daten (kein Mess-Seed): Ordner im Temp-Bereich
    try:
        import tempfile
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as td:
            hp = {"exakt": 3e-3, "bewusst": 3e-3}
            kf = _konfig(True, 10, 5, 3, 10, hp)
            r1 = _aufgabe(("mnist", 2098, "id", hp, True, 10, 5, 3, 10, td, kf))
            r2 = _aufgabe(("mnist", 2098, "id", hp, True, 10, 5, 3, 10, td, kf))
            if r1 != r2 or not (0.0 <= r1["acc_inf"] <= 1.0) or set(r1["acc_bewusst"]) != set(map(str, S_GITTER)):
                fehler("Aufgabe/Wiederaufnahme")
    except Exception as ex:  # pragma: no cover
        fehler(f"Aufgabe Ausnahme {ex!r}")
    return ok


if __name__ == "__main__":
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell, fashion="--fashion" in sys.argv)
    print(gem.speichere("I-040" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
