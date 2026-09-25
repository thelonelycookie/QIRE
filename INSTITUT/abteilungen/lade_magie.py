"""Abteilung Clifford/Hardware (Karte I-033, Quantenstufe Q3) -- Magie-Konzentration im LADEN.

HEILMEIER:
- Was wollen wir?  C-kodierung und I-020: Das Laden des Bildes (Amplituden-
  kodierung) frisst 77-96 % der T-Gatter der ganzen Quanten-Inferenz; die duenn
  besetzte exakte Praeparation braucht im Mittel ~3.400 T (~110 magische Ry).
  Frage: Wie viele Winkel der Zustandspraeparation muessen je Bild EXAKT (nicht-
  Clifford) sein, damit die trainierte QIRE-Zelle nichts verliert? Gilt die
  Magie-Konzentration (M50 = 2 von 32 Winkeln in der Schaltung) auch fuer das Laden?
- Wie heute?  Approximative Zustandspraeparation (kleine Winkel weglassen,
  Araujo et al. 2021; „approximate amplitude encoding“) ist bekannt; die Kurve
  Lade-T gegen Klassifikationsgenauigkeit mit Rundung auf das Clifford-Gitter
  und grob-zu-fein ueber die Baumebenen haben wir nicht gefunden.
- Was ist neu?  Zwei Familien genaeherter Praeparationen desselben Baums:
    (a) grob-zu-fein: die oberen e Baumebenen exakt, alle tieferen Ebenen auf das
        pi/2-Gitter gerundet (e = 0 -> alles Clifford, e = 8 -> exakt);
    (b) Top-m je Bild: die m Winkel mit dem groessten Abstand zum pi/2-Gitter
        bleiben exakt, der Rest wird gerundet (m = 0 -> alles Clifford,
        m = 255 -> exakt).
  Kontrollen: exakt geladen mit gleichem Nachlauf (FAIRE REFERENZ, Lehre
  C-fashion/C-referenz), zufaellige m Winkel exakt statt Top-m (gleiches m),
  alle Winkel gerundet, Identitaet (ohne Schaltung) als Positivkontrolle.
- Woran merken wir, ob es klappt?  vortest(): Normierung exakt in jedem Arm,
  e = 8 und m = 255 reproduzieren den exakten Zustand, numpy-Zustand == qiskit-
  Statevector der expliziten Schaltung, T-Formel == kodierung.kosten() der
  qiskit-Schaltung, Rundung aller Winkel -> 0 T; jeder Urteilszweig erzwungen.

WELCHE WINKEL (wichtig, vorab festgelegt):
  Gerundet werden die WINKEL DER GATTER, nicht die Baumwinkel theta_c. Der
  Moettoenen-Baum setzt Qubit j mit einer gleichfoermig kontrollierten Ry (UCRY)
  mit 2^j Winkeln theta_c; als Gatter wird sie per Gray-Code in 2^j einfache
  Ry(phi_i) + 2^j CNOT zerlegt, theta = M phi (gesamtbudget.gray_matrix). Ein auf
  pi/2 gerundetes theta_c macht die UCRY NICHT zu Clifford (kontrolliertes
  Ry(pi/2) braucht T-Gatter: phi = +-pi/4). Nur Ry(phi) mit phi auf k·pi/2 ist
  Clifford (0 T). Also: Basis = die 255 Gray-Code-Winkel phi (1+2+...+128) der
  DUENN BESETZTEN exakten Praeparation aus I-020 (gesamtbudget.phi_duenn, OMP:
  nur besetzte Teilbaeume legen Gleichungen fest, ~110 der 255 phi != 0). Jeder
  Arm behaelt eine Teilmenge dieser Winkel exakt und rundet den Rest auf k·pi/2.
  Die Referenz „duenn exakt“ ist damit die billigste exakte Schaltung, die das
  Institut kennt, und jeder Arm ist eine Teilmenge ihrer Magie.
  Grob-zu-fein: Ebene j = Modell-Qubit j (0 = hoechstwertiges Bit = obere/untere
  Bildhaelfte; Ebenen 0-3 teilen Zeilen, 4-7 Spalten).
  Rundung kann Teilbaeume hart auf 0 oder 1 setzen (Ry(k·pi) verschiebt die ganze
  Masse); die Normierung bleibt exakt, weil jede Stufe eine Drehung ist (Vortest).

T-ZAEHLUNG wie kodierung.py: T_Laden(Bild) = Summe ueber alle Nicht-Clifford-
  Winkel phi der Ross-Selinger-Synthese t_je_rotation(phi, eps = 1e-3) (EPS wie
  C-hardware/I-020, je Rotation). CNOTs und Clifford-Ry kosten 0 T. Vortest
  vergleicht die Formel mit kodierung.kosten() auf der expliziten qiskit-Schaltung
  (gesamtbudget.gray_schaltung). T wird auf einer Stichprobe von N_KOSTEN Test-
  bildern je Seed gezaehlt (gridsynth ~1,6 ms je Winkel); die Zahl magischer
  Winkel zusaetzlich auf ALLEN Testbildern.

ABLAUF JE SEED (Phase 1 = C-mnist-Rezept, Phase 2 identisch fuer alle Arme):
  Phase 1: volle Zelle (clifford._baue("voll")) auf EXAKT geladenen Bildern,
  MNIST 16x16, 10.000 Trainings-/5.000 Testbilder, 1.500 Schritte, Rate
  C-mnist „voll“. Phase 2 je Arm: Trainings- UND Testbilder mit der Arm-Naeherung
  laden, clifford._gerundet(voll, k = 32) (alle 32 Winkel frei, Start beim
  trainierten Modell, FRISCHER Kopf), 1.500 Schritte, Rate C-mnist „rundung“,
  gleicher Trainings-Seed. Der Arm „referenz“ ist genau diese Phase 2 auf den
  exakten Zustaenden. Identitaet: Kopf auf |x|^2 der exakten Zustaende (ein
  frischer linearer Kopf nach Phase 1 waere dieselbe Rechnung), Rate C-mnist
  „ohne_schaltung“.
  Lernraten: aus ergebnisse/C-mnist__*.json (dort auf eigenen Tuning-Seeds mit
  Gitterverlaengerung gesucht, Optimum innen). ABWEICHUNG (offen benannt): C-mnist
  nutzte 6.000 Trainingsbilder, hier 10.000 (Vorgabe); fuer die genaeherten
  Zustaende ist die Rate nicht eigens gesucht -- sie MUSS fuer alle Arme gleich
  sein (identische zweite Phase), sonst ist der Vergleich zur Referenz nicht fair.

ARME (vorab festgelegt; reduziert, damit der Lauf ~60 min bei 4 Prozessen bleibt,
  gemessen: 1.500 Schritte auf 10.000 Bildern ~110-140 s je Training):
  referenz (= e 8 = m 255), alle_pi2 (= e 0 = m 0), ebene_4, ebene_5, ebene_6,
  top_4, top_16, top_32, top_48, zufall_16 (10 Trainings je Seed) + Phase 1, dazu
  identitaet (billig: Kopf auf festen Merkmalen). 10 Seeds x 11 Trainings x ~120 s
  / 4 Prozesse + OMP ~100 s je Seed -> geschaetzt ~60-65 min.
  Weggelassen und warum: ebene_1..3 (<= 7 magische Winkel, Kurve beginnt bei
  alle_pi2), ebene_7 und top_64 (~60 % der duennen T, ueber der 50-%-Grenze der
  Karte; nur ihre T wird gezaehlt), top_8, zufall bei anderen m, „alle Winkel
  pi/4“ (in I-020 schon gemessen: duenn_pi4 0,731). Die T-Zahl aller Ebenen
  e = 0..8 und m = 0,4,8,...,255 wird trotzdem beschreibend gezaehlt (billig).
  Vorab angeschaut (nur KOSTEN, keine Genauigkeit): auf 300 Trainingsbildern hat
  die duenne Praeparation je Ebene im Mittel 1 / 2 / 3,8 / 6,5 / 12,2 / 14,3 /
  26,5 / 43 von 1/2/4/.../128 Winkeln != 0 (Summe ~109). Erwartete T-Anteile:
  e4 ~12 %, e5 ~23 %, e6 ~36 %, e7 ~61 %; m32 ~29 %, m48 ~44 %, m64 ~57 %.

VORREGISTRIERUNG (Regel 7, steht vor jeder Zahl; delta = 1 pp, alpha = 0,05):
  Seeds 140-149 (frisch), 10 Seeds, exakter Wilcoxon (labor.wilcoxon_exakt,
  einseitig datenmenge.wilcoxon_einseitig, gleiche Rangbildung).
  L-0 Positivkontrolle: d = acc(referenz) - acc(identitaet); H1 wenn mittel > 0
      und zweiseitiger exakter Wilcoxon p < 0,05, sonst H0.
  L-1 Haupt (Abbruchkriterium der Karte): d_s = acc(Stufe) - acc(referenz).
      Zulaessig ist eine Stufe, wenn ihre mittlere Lade-T (Stichprobe, alle Seeds)
      <= 50 % der mittleren Lade-T von „duenn exakt“ (= referenz) ist.
      Vorbedingung (Grenze relativ zum Effekt): Ladegewinn acc(referenz) -
      acc(alle_pi2) >= 5·delta = 5 pp, sonst HU (dann ist 1 pp kein kleiner Teil
      des Effekts). HAUPTSTUFEN (Familie, Holm): ebene_6 und top_48 -- je Variante
      die teuerste Stufe, deren T-Anteil nach der Kostenvorschau sicher unter 50 %
      liegt (die genaueste zulaessige Stufe hat die meiste Aussagekraft fuer das
      Abbruchkriterium; zwei Varianten = zwei Fragen der Karte, deshalb Holm ueber
      beide statt Suche ueber alle Stufen). Faellt eine Hauptstufe gemessen ueber
      50 %, wird sie aus der Familie genommen (Holm ueber den Rest).
      H1: mindestens eine Hauptstufe ist per TOST (exakter Wilcoxon, beide
          einseitigen Tests gegen -1 pp und +1 pp, p_TOST = max) nach Holm-
          Korrektur innerhalb +-1 pp der Referenz.
      H0: JEDE zulaessige gemessene Stufe (alle Arme ausser referenz/zufall mit
          T <= 50 %, einschliesslich alle_pi2) liegt nachweislich > 1 pp unter
          der Referenz (einseitiger exakter Wilcoxon auf d + delta < 0, p < 0,05
          je Stufe; Intersection-Union-Test, keine Korrektur noetig).
      Sonst HU; ebenso HU, wenn keine Stufe zulaessig ist.
  L-2: d = acc(top_16) - acc(zufall_16) (vorab benanntes m = 16: gleiches m,
      gleiche Magiezahl, ~15 % der T; bei kleinem m zaehlt die Wahl am meisten).
      H1: mittel > 0 und exakter Wilcoxon zweiseitig p < 0,05. H0: Vorteil
      nachweislich < 1 pp (einseitig d - delta < 0, p < 0,05). Sonst HU.
  L-3 beschreibend: Kurven Lade-T gegen Genauigkeit fuer (a) und (b); Gewinn-
      anteil G(Stufe) = (acc - acc_alle_pi2) / (acc_referenz - acc_alle_pi2)
      (Verhaeltnis der Mittel); M50/M90 = kleinste Stufe (e bzw. m) und kleinste
      Lade-T mit G >= 0,5 bzw. 0,9.

GRENZEN (vorab): rauschfrei simuliert; eps je Rotation fest (nicht als Gesamt-
  budget optimiert); OMP-Duennloesung ist eine Heuristik (T obere Schranke);
  T aus einer Stichprobe von Testbildern; Rate fuer genaeherte Zustaende nicht
  eigens gesucht (fuer alle Arme gleich); Top-m waehlt nach Gitterabstand, nicht
  nach Wirkung (das waere I-001 fuers Laden).
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from abteilungen.clifford import LAGEN, _baue, _gerundet
from abteilungen.datenmenge import holm, wilcoxon_einseitig
from abteilungen.gesamtbudget import baum_zustand, gray_schaltung, phi_duenn
from abteilungen.kodierung import EPS, ist_clifford, kosten, normiere, t_je_rotation, treue, uebersetze
from labor.gemeinsam import (bilder, genauigkeit, neuestes, parallel, speichere, trainiere, wilcoxon_exakt,
                             zusammenfassung)

HYPOTHESEN = {
    "L-0": "Positivkontrolle: Referenz (exakt geladen, nachtrainiert) schlaegt die Identitaet (ohne "
           "Schaltung). H1: mittel > 0 und exakter Wilcoxon zweiseitig p < 0,05; sonst H0.",
    "L-1": "Haupt (Abbruchkriterium der Karte I-033): Unter den Hauptstufen ebene_6 und top_48 (nur wenn "
           "gemessene mittlere Lade-T <= 50 % von duenn exakt) liegt mindestens eine per TOST (exakter "
           "Wilcoxon, +-1 pp, Holm ueber die zulaessigen Hauptstufen) innerhalb 1 pp der Referenz -> H1. "
           "Liegt JEDE zulaessige gemessene Stufe nachweislich > 1 pp darunter (einseitig p < 0,05 je "
           "Stufe) -> H0. Sonst HU. Vorbedingung: Ladegewinn referenz - alle_pi2 >= 5 pp, sonst HU.",
    "L-2": "Top-m schlaegt zufaellige m bei m = 16 (gepaart, exakter Wilcoxon zweiseitig p < 0,05) -> H1; "
           "Vorteil nachweislich < 1 pp -> H0; sonst HU.",
    "L-3": "Beschreibend: Lade-T gegen Genauigkeit fuer (a) grob-zu-fein und (b) Top-m; Gewinnanteil "
           "G = (acc - acc_alle_pi2) / (acc_referenz - acc_alle_pi2), M50/M90 des Ladens.",
}

N = 8                                   # Qubits, 16x16 = 256 Amplituden
N_WINKEL = 2 ** N - 1                   # 255 Gray-Code-Winkel
GITTER = math.pi / 2
SEEDS = tuple(range(140, 150))
N_TRAIN, N_TEST, SCHRITTE = 10_000, 5_000, 1500
N_KOSTEN = 100                          # Testbilder je Seed fuer die gridsynth-T-Zaehlung
DELTA = 0.01                            # TOST-Grenze 1 pp (Karte)
ALPHA = 0.05
ANTEIL_T = 0.5                          # Karte: <= 50 % der duennen Lade-T
MIN_LADEGEWINN = 5 * DELTA              # Grenze relativ zum Effekt
HAUPTSTUFEN = ("ebene_6", "top_48")     # L-1-Familie (Holm)
L2_M = 16
EBENEN = (4, 5, 6)
TOP_M = (4, 16, 32, 48)
ARME = ("referenz", "alle_pi2", *[f"ebene_{e}" for e in EBENEN], *[f"top_{m}" for m in TOP_M],
        f"zufall_{L2_M}")
# nur gezaehlt (T), nicht trainiert -- fuer die Kostenkurve
KOSTEN_ARME = tuple(dict.fromkeys(ARME + tuple(f"ebene_{e}" for e in range(N + 1))
                                  + tuple(f"top_{m}" for m in (0, 4, 8, 16, 32, 48, 64, 96, 128, N_WINKEL))))
KURVE_A = [("alle_pi2", 0)] + [(f"ebene_{e}", e) for e in EBENEN] + [("referenz", N)]
KURVE_B = [("alle_pi2", 0)] + [(f"top_{m}", m) for m in TOP_M] + [("referenz", N_WINKEL)]
LR_ERSATZ = {"voll": 0.08, "rundung": 0.08, "ohne_schaltung": 0.08}
EBENE = np.concatenate([np.full(2 ** j, j) for j in range(N)])      # Ebene je Winkel (255)


# ------------------------------------------------------------------ 1. Bausteine
def flach(phis) -> np.ndarray:
    """Liste je Ebene (B, 2^j) -> (B, 255)."""
    return np.concatenate([np.atleast_2d(p) for p in phis], axis=1)


def ebenen(phi: np.ndarray):
    """(B, 255) -> Liste je Ebene (B, 2^j); Ebene j beginnt bei 2^j - 1."""
    phi = np.atleast_2d(phi)
    return [phi[:, 2 ** j - 1: 2 ** (j + 1) - 1] for j in range(N)]


def gerundet(phi: np.ndarray) -> np.ndarray:
    return np.round(phi / GITTER) * GITTER


def magisch(phi: np.ndarray) -> np.ndarray:
    """Nicht-Clifford-Winkel, gleiche Toleranz wie kodierung.ist_clifford."""
    r = phi / GITTER
    return np.abs(r - np.round(r)) >= 1e-9


def _rng(seed, teil, m):
    return np.random.default_rng([int(seed), 0 if teil == "train" else 1, int(m)])


def maske_exakt(phi: np.ndarray, arm: str, seed: int = 0, teil: str = "test") -> np.ndarray:
    """(B, 255) bool: welche Winkel bleiben exakt? Der Rest wird auf k·pi/2 gerundet."""
    B = phi.shape[0]
    mag = magisch(phi)
    if arm == "referenz":
        return np.ones_like(mag)
    if arm == "alle_pi2":
        return np.zeros_like(mag)
    art, zahl = arm.rsplit("_", 1)
    zahl = int(zahl)
    if art == "ebene":
        return np.broadcast_to(EBENE[None, :] < zahl, (B, N_WINKEL)).copy()
    if art == "top":
        wert = np.abs(phi - gerundet(phi))
    elif art == "zufall":
        wert = _rng(seed, teil, zahl).random((B, N_WINKEL))
        wert[~mag] = -1.0
    else:
        raise ValueError(arm)
    ordnung = np.argsort(-wert, axis=1, kind="stable")[:, :zahl]
    m = np.zeros((B, N_WINKEL), dtype=bool)
    np.put_along_axis(m, ordnung, True, axis=1)
    return m & mag


def arm_winkel(phi: np.ndarray, arm: str, seed: int = 0, teil: str = "test") -> np.ndarray:
    return np.where(maske_exakt(phi, arm, seed, teil), phi, gerundet(phi))


def arm_zustand(phi: np.ndarray, arm: str, seed: int = 0, teil: str = "test") -> np.ndarray:
    return baum_zustand(ebenen(arm_winkel(phi, arm, seed, teil)))


def t_formel(phi_zeile: np.ndarray, eps=EPS) -> int:
    """T eines Bildes: Summe der gridsynth-T ueber alle Nicht-Clifford-Winkel (wie kodierung.kosten)."""
    return int(sum(t_je_rotation(float(a), eps) for a in phi_zeile if not ist_clifford(float(a))))


def duenne_winkel(psis: np.ndarray) -> np.ndarray:
    """Gray-Code-Winkel der duenn besetzten exakten Praeparation (I-020), (B, 255)."""
    return flach(phi_duenn(psis))


# ------------------------------------------------------------------ 2. Lauf-Bausteine
def _raten():
    e = neuestes("C-mnist")
    if not e:
        return dict(LR_ERSATZ), "Ersatz (keine C-mnist-Rohdaten)"
    return {arm: float(e["lr"][arm]["lr"]) for arm in LR_ERSATZ}, "ergebnisse/C-mnist (neuestes)"


def _pfad(ordner, seed):
    return os.path.join(ordner, f"seed_{seed}.npz"), os.path.join(ordner, f"voll_{seed}.pt")


def _vorbereitung(a):
    """Je Seed: Daten, duenne Winkel (Cache auf Platte), Phase 1, Identitaet, T-Stichprobe."""
    seed, n_train, n_test, schritte, n_kosten, lr, ordner = a
    t0 = time.time()
    xtr, ytr, xte, yte = bilder("mnist", seed, n_train=n_train, n_test=n_test, seite=16)
    ptr, pte = normiere(xtr.numpy()), normiere(xte.numpy())
    phi_tr, phi_te = duenne_winkel(ptr), duenne_winkel(pte)
    fehler = max(float(np.abs(baum_zustand(ebenen(phi_tr[:50])) - ptr[:50]).max()),
                 float(np.abs(baum_zustand(ebenen(phi_te[:50])) - pte[:50]).max()))
    t_omp = time.time() - t0
    np.savez(_pfad(ordner, seed)[0], phi_tr=phi_tr, phi_te=phi_te, ytr=ytr.numpy(), yte=yte.numpy())
    ztr, zte = torch.tensor(ptr, dtype=torch.float32), torch.tensor(pte, dtype=torch.float32)
    voll, _ = _baue("voll", seed, N)
    trainiere(voll, ztr, ytr, schritte, lr["voll"], seed=seed)
    torch.save(voll.state_dict(), _pfad(ordner, seed)[1])
    with torch.no_grad():
        acc_voll = genauigkeit(voll(zte), yte)
    # Identitaet: Winkel fest (frei = 0, Gradient 0) -> Merkmale einmal rechnen und nur den Kopf
    # trainieren; mathematisch dasselbe wie trainiere(ident, ...), aber ohne 1.500x die Unitaere.
    ident, _ = _baue("ohne_schaltung", seed, N)
    with torch.no_grad():
        mtr, mte = ident.merkmale(ztr), ident.merkmale(zte)
    trainiere(ident.kopf, mtr, ytr, schritte, lr["ohne_schaltung"], seed=seed)
    with torch.no_grad():
        acc_ident = genauigkeit(ident.kopf(mte), yte)
    t1 = time.time()
    k = min(n_kosten, len(pte))
    T = {}
    for arm in KOSTEN_ARME:
        w = arm_winkel(phi_te, arm, seed, "test")[:k] if arm.startswith("zufall") else arm_winkel(phi_te[:k], arm)
        T[arm] = [t_formel(z) for z in w]
    return {"seed": seed, "voll_phase1": acc_voll, "identitaet": acc_ident, "rekonstruktion_fehler": fehler,
            "T_stichprobe": T, "nichtnull_mittel": float(np.mean(np.count_nonzero(pte > 0, axis=1))),
            "sekunden_omp": t_omp, "sekunden_kosten": time.time() - t1, "sekunden": time.time() - t0}


def _arm_lauf(a):
    """Phase 2 eines Arms: genaehert laden, _gerundet(voll, 32) (frischer Kopf), gleiches Training."""
    seed, arm, schritte, lr, ordner = a
    t0 = time.time()
    d = np.load(_pfad(ordner, seed)[0])
    phi_tr, phi_te = d["phi_tr"], d["phi_te"]
    ytr, yte = torch.tensor(d["ytr"]), torch.tensor(d["yte"])
    ztr = arm_zustand(phi_tr, arm, seed, "train")
    zte = arm_zustand(phi_te, arm, seed, "test")
    exakt_te = baum_zustand(ebenen(phi_te))
    voll, _ = _baue("voll", seed, N)
    voll.load_state_dict(torch.load(_pfad(ordner, seed)[1]))
    m = _gerundet(voll, LAGEN * N, seed)          # alle Winkel frei, frischer Kopf
    trainiere(m, torch.tensor(ztr, dtype=torch.float32), ytr, schritte, lr["rundung"], seed=seed)
    with torch.no_grad():
        acc = genauigkeit(m(torch.tensor(zte, dtype=torch.float32)), yte)
    return {"seed": seed, "arm": arm, "acc": acc,
            "treue_test": float(np.mean(np.sum(zte * exakt_te, axis=1) ** 2)),
            "norm_fehler": float(np.abs(np.linalg.norm(zte, axis=1) - 1).max()),
            "magisch_test": float(np.mean(magisch(arm_winkel(phi_te, arm, seed, "test")).sum(1))),
            "null_amplituden_test": float(np.mean(np.abs(zte) < 1e-12)),
            "sekunden": time.time() - t0}


# ------------------------------------------------------------------ 3. Statistik
def _gepaart(d) -> dict:
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    aus = {"mittel": float(d.mean()), "p": wilcoxon_exakt(d), "n": int(len(d))}
    if len(d) > 1 and d.std(ddof=1) > 0:
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        q = float(t_vert.ppf(0.95, len(d) - 1))
        aus.update(sd=float(d.std(ddof=1)), ki90_unten=aus["mittel"] - q * se, ki90_oben=aus["mittel"] + q * se)
    return aus


def _tost(d, delta=DELTA) -> dict:
    d = np.asarray(d, dtype=float)
    p_unten = wilcoxon_einseitig(d + delta, "groesser")     # H0: d <= -delta
    p_oben = wilcoxon_einseitig(d - delta, "kleiner")       # H0: d >= +delta
    return {"p_unten": p_unten, "p_oben": p_oben, "p_tost": max(p_unten, p_oben),
            "p_schlechter": wilcoxon_einseitig(d + delta, "kleiner")}   # H1: d < -delta


def auswertung(zeilen) -> dict:
    """zeilen: je Seed {'seed', 'acc': {arm: float}, 'T_stichprobe': {arm: [T je Bild]}, ...}."""
    zeilen = sorted(zeilen, key=lambda z: z["seed"])
    arme = [a for a in zeilen[0]["acc"]]
    acc = {a: np.array([z["acc"][a] for z in zeilen], dtype=float) for a in arme}
    T = {a: float(np.mean([t for z in zeilen for t in z["T_stichprobe"][a]]))
         for a in zeilen[0]["T_stichprobe"]}
    t_ref = T["referenz"]
    aus = {"n": len(zeilen), "seeds": [z["seed"] for z in zeilen],
           "acc": {a: zusammenfassung(v) for a, v in acc.items()},
           "T_laden": {a: {"mittel": T[a], "anteil_duenn": T[a] / t_ref if t_ref > 0 else None} for a in T},
           "T_duenn_exakt": t_ref, "grenze_T": ANTEIL_T * t_ref}
    if "magisch_test" in zeilen[0]:
        aus["magisch_test"] = {a: float(np.mean([z["magisch_test"][a] for z in zeilen])) for a in arme
                               if a in zeilen[0]["magisch_test"]}
    # L-0
    aus["L0"] = _gepaart(acc["referenz"] - acc["identitaet"])
    # L-1
    ladegewinn = float(np.mean(acc["referenz"] - acc["alle_pi2"]))
    stufen = {}
    for a in arme:
        if a in ("referenz", "identitaet") or a.startswith("zufall") or a not in T:
            continue
        d = acc[a] - acc["referenz"]
        stufen[a] = {**_gepaart(d), **_tost(d), "T": T[a], "anteil_T": T[a] / t_ref,
                     "zulaessig": bool(T[a] <= ANTEIL_T * t_ref)}
    familie = [h for h in HAUPTSTUFEN if h in stufen and stufen[h]["zulaessig"]]
    p_holm = holm({h: stufen[h]["p_tost"] for h in familie}) if familie else {}
    zulaessig = [a for a, s in stufen.items() if s["zulaessig"]]
    aus["L1"] = {"ladegewinn": ladegewinn, "vorbedingung": bool(ladegewinn >= MIN_LADEGEWINN),
                 "stufen": stufen, "familie": familie, "p_holm": p_holm,
                 "aequivalent": [h for h in familie if p_holm[h] < ALPHA],
                 "zulaessig": zulaessig,
                 "alle_zulaessigen_schlechter": bool(zulaessig) and all(stufen[a]["p_schlechter"] < ALPHA
                                                                        for a in zulaessig)}
    # L-2
    d2 = acc[f"top_{L2_M}"] - acc[f"zufall_{L2_M}"]
    aus["L2"] = {**_gepaart(d2), "p_vorteil_klein": wilcoxon_einseitig(d2 - DELTA, "kleiner"),
                 "T_top": T.get(f"top_{L2_M}"), "T_zufall": T.get(f"zufall_{L2_M}")}
    # L-3
    basis, ref = float(acc["alle_pi2"].mean()), float(acc["referenz"].mean())

    def kurve(liste):
        out = []
        for arm, stufe in liste:
            if arm not in acc:
                continue
            g = (float(acc[arm].mean()) - basis) / (ref - basis) if ref > basis else None
            out.append({"arm": arm, "stufe": stufe, "T": T[arm], "anteil_T": T[arm] / t_ref,
                        "acc": float(acc[arm].mean()), "G": g})
        return out

    def m_stufe(k, schwelle):
        treffer = [p for p in k if p["G"] is not None and p["G"] >= schwelle]
        return min(treffer, key=lambda p: p["T"]) if treffer else None

    ka, kb = kurve(KURVE_A), kurve(KURVE_B)
    aus["L3"] = {"kurve_a": ka, "kurve_b": kb, "T_kostenkurve": {a: T[a] for a in T}}
    for s in (50, 90):
        for name, k in (("a", ka), ("b", kb), ("beide", ka + kb)):
            p = m_stufe(k, s / 100)
            aus["L3"][f"M{s}_{name}"] = None if p is None else {"arm": p["arm"], "stufe": p["stufe"],
                                                                "T": p["T"], "anteil_T": p["anteil_T"]}
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum; jeder Wert ist ein String."""
    u = {}
    l0 = a["L0"]
    u["L-0"] = (f"H1: Referenz schlaegt Identitaet um {100 * l0['mittel']:+.2f} pp (p = {l0['p']:.4f})"
                if l0["mittel"] > 0 and l0["p"] < ALPHA else
                f"H0: Referenz nicht nachweisbar besser als Identitaet ({100 * l0['mittel']:+.2f} pp, "
                f"p = {l0['p']:.4f}) -- Positivkontrolle feuert nicht")
    l1 = a["L1"]
    st = l1["stufen"]
    if not l1["vorbedingung"]:
        u["L-1"] = (f"HU (Vorbedingung): Ladegewinn referenz - alle_pi2 = {100 * l1['ladegewinn']:.2f} pp "
                    f"< {100 * MIN_LADEGEWINN:.0f} pp -- 1-pp-Grenze nicht klein gegen den Effekt")
    elif not l1["zulaessig"]:
        u["L-1"] = f"HU: keine Stufe mit Lade-T <= {100 * ANTEIL_T:.0f} % von duenn exakt ({a['T_duenn_exakt']:.0f} T)"
    elif l1["aequivalent"]:
        b = min(l1["aequivalent"], key=lambda h: st[h]["T"])
        s = st[b]
        u["L-1"] = (f"H1: {b} mit {s['T']:.0f} T ({100 * s['anteil_T']:.1f} % von duenn exakt "
                    f"{a['T_duenn_exakt']:.0f} T) liegt per TOST innerhalb 1 pp der Referenz "
                    f"({100 * s['mittel']:+.2f} pp, p_TOST = {s['p_tost']:.4f}, Holm {l1['p_holm'][b]:.4f}; "
                    f"Familie {l1['familie']})")
    elif l1["alle_zulaessigen_schlechter"]:
        schlecht = max(l1["zulaessig"], key=lambda h: st[h]["mittel"])
        u["L-1"] = (f"H0: jede Stufe mit <= {100 * ANTEIL_T:.0f} % der Lade-T liegt nachweislich > 1 pp unter "
                    f"der Referenz (beste: {schlecht} {100 * st[schlecht]['mittel']:+.2f} pp, "
                    f"p = {st[schlecht]['p_schlechter']:.4f}) -- Abbruchkriterium der Karte erfuellt")
    else:
        teile = ", ".join(f"{h} {100 * st[h]['mittel']:+.2f} pp (p_TOST {st[h]['p_tost']:.3f})"
                          for h in (l1["familie"] or l1["zulaessig"]))
        u["L-1"] = f"HU: weder Aequivalenz (Holm) noch jede Stufe nachweislich > 1 pp darunter -- {teile}"
    l2 = a["L2"]
    if l2["mittel"] > 0 and l2["p"] < ALPHA:
        u["L-2"] = f"H1: top_{L2_M} schlaegt zufall_{L2_M} um {100 * l2['mittel']:+.2f} pp (p = {l2['p']:.4f})"
    elif l2["p_vorteil_klein"] < ALPHA:
        u["L-2"] = (f"H0: Vorteil von top_{L2_M} nachweislich < 1 pp ({100 * l2['mittel']:+.2f} pp, "
                    f"p = {l2['p_vorteil_klein']:.4f})")
    else:
        u["L-2"] = f"HU: kein Urteil ({100 * l2['mittel']:+.2f} pp, p = {l2['p']:.4f})"
    l3 = a["L3"]

    def txt(p):
        return "nicht erreicht" if p is None else f"{p['arm']} ({p['T']:.0f} T, {100 * p['anteil_T']:.0f} %)"

    kurve = lambda k: " ".join(f"{p['stufe']}:{p['acc']:.3f}/{p['T']:.0f}T" for p in k)
    u["L-3"] = (f"beschreibend: (a) e {kurve(l3['kurve_a'])}; (b) m {kurve(l3['kurve_b'])}; "
                f"M50 a {txt(l3['M50_a'])}, b {txt(l3['M50_b'])}; M90 a {txt(l3['M90_a'])}, "
                f"b {txt(l3['M90_b'])}")
    return u


# ------------------------------------------------------------------ 4. Lauf
def lauf(schnell=False, seeds=None, schritte=None, n_train=None, n_test=None, n_kosten=None, **kw) -> dict:
    t0 = time.time()
    seeds = tuple(seeds) if seeds is not None else (SEEDS[:2] if schnell else SEEDS)
    schritte = schritte or (150 if schnell else SCHRITTE)
    n_train = n_train or (600 if schnell else N_TRAIN)
    n_test = n_test or (300 if schnell else N_TEST)
    n_kosten = n_kosten or (5 if schnell else N_KOSTEN)
    lr, lr_quelle = _raten()
    ordner = tempfile.mkdtemp(prefix="lade_magie_")
    try:
        vor = parallel(_vorbereitung, [(s, n_train, n_test, schritte, n_kosten, lr, ordner) for s in seeds])
        print(f"  Vorbereitung fertig ({time.time() - t0:.0f} s)", flush=True)
        arme = parallel(_arm_lauf, [(s, arm, schritte, lr, ordner) for s in seeds for arm in ARME])
    finally:
        shutil.rmtree(ordner, ignore_errors=True)
    zeilen = []
    for v in sorted(vor, key=lambda v: v["seed"]):
        eig = [r for r in arme if r["seed"] == v["seed"]]
        z = {**v, "acc": {"identitaet": v["identitaet"], **{r["arm"]: r["acc"] for r in eig}},
             "treue_test": {r["arm"]: r["treue_test"] for r in eig},
             "norm_fehler": max(r["norm_fehler"] for r in eig),
             "magisch_test": {r["arm"]: r["magisch_test"] for r in eig},
             "null_amplituden_test": {r["arm"]: r["null_amplituden_test"] for r in eig},
             "sekunden_arme": {r["arm"]: r["sekunden"] for r in eig}}
        zeilen.append(z)
        print(f"  Seed {z['seed']}: ref {z['acc']['referenz']:.4f}  ident {z['identitaet']:.4f}  "
              f"pi2 {z['acc']['alle_pi2']:.4f}  e6 {z['acc']['ebene_6']:.4f}  m48 {z['acc']['top_48']:.4f}",
              flush=True)
    ergebnis = {"abteilung": "clifford/hardware", "karte": "I-033", "quantenstufe": "Q3 (lokal simuliert)",
                "hypothesen": HYPOTHESEN, "seeds": list(seeds), "schritte": schritte, "n_train": n_train,
                "n_test": n_test, "n_kosten_je_seed": n_kosten, "eps_je_rotation": EPS, "lr": lr,
                "lr_quelle": lr_quelle, "arme": list(ARME), "hauptstufen": list(HAUPTSTUFEN),
                "vortest_bestanden": bool(vortest(verbose=False)) if not kw.get("ohne_vortest") else None,
                "zeilen": zeilen}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


# ------------------------------------------------------------------ 5. Vortest
def _synth(rng, mittel: dict, T: dict, rausch=0.002, n=10):
    zeilen = []
    for s in range(n):
        gemeinsam_ = rng.normal(0, 0.003)            # Seed-Effekt fuer alle Arme
        z = {"seed": s, "acc": {a: m + gemeinsam_ + rng.normal(0, rausch) for a, m in mittel.items()},
             "T_stichprobe": {a: [t, t] for a, t in T.items()}}
        zeilen.append(z)
    return zeilen


def vortest(verbose=True) -> bool:
    ok, tests = True, []

    def pruefe(name, wert, bed):
        nonlocal ok
        tests.append((name, float(wert), bool(bed)))
        ok &= bool(bed)
        if verbose:
            print(f"  {name}: {float(wert):.3g}  {'ok' if bed else 'FEHLER'}")

    # ---- Bausteine auf echten MNIST-Bildern und synthetischen Vektoren
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(7)
    _, _, xte, _ = bilder("mnist", SEEDS[0], n_train=10, n_test=6, seite=16)
    duenn = np.zeros(256)
    duenn[rng.choice(256, 12, replace=False)] = rng.normal(size=12)
    psis = np.concatenate([normiere(xte.numpy()), normiere(duenn)[None], normiere(rng.normal(size=(2, 256)))])
    phi = duenne_winkel(psis)
    exakt = baum_zustand(ebenen(phi))
    pruefe("(a) duenne Winkel reproduzieren psi", np.abs(exakt - psis).max(), np.abs(exakt - psis).max() < 1e-10)
    for arm in ("ebene_8", f"top_{N_WINKEL}", "referenz"):
        z = arm_zustand(phi, arm)
        pruefe(f"(b) {arm} == exakt", np.abs(z - psis).max(), np.abs(z - psis).max() < 1e-10)
    nmag = magisch(phi).sum(1)
    z = arm_zustand(phi, f"top_{int(nmag.max())}")
    pruefe("(b) top_m mit m >= #magisch == exakt", np.abs(z - psis).max(), np.abs(z - psis).max() < 1e-10)
    z0 = [arm_zustand(phi, a) for a in ("alle_pi2", "ebene_0", "top_0", "zufall_0")]
    pruefe("(c) alle_pi2 == ebene_0 == top_0 == zufall_0", max(np.abs(z - z0[0]).max() for z in z0),
           max(np.abs(z - z0[0]).max() for z in z0) == 0)
    normf = max(float(np.abs(np.linalg.norm(arm_zustand(phi, a, 3), axis=1) - 1).max()) for a in KOSTEN_ARME)
    pruefe("(d) Normierung exakt in allen Armen", normf, normf < 1e-12)
    # harte Masken: Rundung setzt Teilbaeume auf 0 (nur zeigen, dass es vorkommt und Norm bleibt)
    pi2 = arm_zustand(phi, "alle_pi2")
    neu_null = int(np.sum((np.abs(pi2) < 1e-12) & (np.abs(psis) > 1e-12)))
    pruefe("(d) Rundung erzeugt harte Masken (Amplituden exakt 0, die vorher != 0 waren)", neu_null, neu_null > 0)
    # Winkelzahlen
    falsch = [e for e in range(N + 1) if not np.array_equal(magisch(arm_winkel(phi, f"ebene_{e}")),
                                                             magisch(phi) & (EBENE[None] < e))]
    pruefe(f"(e) ebene_e: magisch genau die magischen Winkel der Ebenen < e (e = 0..8), falsch {falsch}",
           len(falsch), not falsch)
    falsch = [f"{art}_{m}" for m in (0, 4, 16, 48, 255) for art in ("top", "zufall")
              if not np.array_equal(magisch(arm_winkel(phi, f"{art}_{m}", 5)).sum(1), np.minimum(m, nmag))]
    pruefe(f"(e) top_m/zufall_m: #magisch == min(m, #magisch), falsch {falsch}", len(falsch), not falsch)
    # Top-m == naive Auswahl, zufall != top, zufall reproduzierbar
    b, m = 0, 16
    dist = np.abs(phi[b] - gerundet(phi[b]))
    naiv = set(sorted(range(N_WINKEL), key=lambda i: -dist[i])[:m])
    pruefe("(f) top_16 waehlt die 16 Winkel mit groesstem Gitterabstand", 0,
           set(np.flatnonzero(maske_exakt(phi, "top_16")[b])) == naiv)
    mz = maske_exakt(phi, "zufall_16", 3)
    pruefe("(f) zufall_16 != top_16 und reproduzierbar", 0,
           (mz != maske_exakt(phi, "top_16")).any() and np.array_equal(mz, maske_exakt(phi, "zufall_16", 3))
           and (mz <= magisch(phi)).all())
    # numpy == qiskit, T-Formel == kodierung.kosten
    falsch = []
    for b in (0, 1, 6):
        for arm in ("referenz", "ebene_4", "top_16", "zufall_16", "alle_pi2"):
            w = arm_winkel(phi, arm, 3)[b]
            qc = gray_schaltung([p[0] for p in ebenen(w[None])])
            sv = Statevector(qc).data
            zn = baum_zustand(ebenen(w[None]))[0]
            k = kosten(qc)
            if np.abs(sv - zn).max() > 1e-8 or k["T"] != t_formel(w) or k["rotationen_magisch"] != int(magisch(w).sum()):
                falsch.append(f"{b}/{arm}")
    pruefe(f"(g) qiskit-Statevector == numpy und T-Formel == kodierung.kosten() (3 Bilder x 5 Arme), falsch {falsch}",
           len(falsch), not falsch)
    pruefe("(g) alle_pi2 -> 0 T", t_formel(arm_winkel(phi, "alle_pi2")[0]), t_formel(arm_winkel(phi, "alle_pi2")[0]) == 0)
    # Vergleich mit der transpilierten Schaltung (uebersetze, wie C-kodierung/I-020): nur berichtet
    w = phi[0]
    tk = kosten(uebersetze(gray_schaltung([p[0] for p in ebenen(w[None])])))["T"]
    pruefe(f"(g) Info: T-Formel {t_formel(w)} vs. uebersetze() {tk} (Bild 0)", tk - t_formel(w), True)
    tr = treue(arm_zustand(phi, "top_48")[0], psis[0])
    pruefe("(g) Info: Treue top_48 Bild 0", tr, 0 <= tr <= 1 + 1e-12)

    # ---- Urteilszweige auf synthetischen Zeilen
    T = {a: 3400.0 for a in KOSTEN_ARME}
    T.update({"alle_pi2": 0.0, "ebene_4": 420.0, "ebene_5": 800.0, "ebene_6": 1250.0, "top_4": 125.0,
              "top_16": 500.0, "top_32": 990.0, "top_48": 1480.0, "top_64": 1950.0, "zufall_16": 500.0})
    basis = {"referenz": 0.94, "identitaet": 0.90, "alle_pi2": 0.60, "ebene_4": 0.85, "ebene_5": 0.90,
             "ebene_6": 0.938, "top_4": 0.80, "top_16": 0.92, "top_32": 0.93, "top_48": 0.938,
             "zufall_16": 0.85}

    def fall(name, aend=None, T_aend=None, rausch=0.002, erwartet=None, ziehungen=4):
        mittel = {**basis, **(aend or {})}
        TT = {**T, **(T_aend or {})}
        for i in range(ziehungen):
            u = urteil(auswertung(_synth(np.random.default_rng(100 + i), mittel, TT, rausch)))
            for k, marke in erwartet.items():
                if not u[k].startswith(marke):
                    pruefe(f"(h) Fall {name}: {k} erwartet {marke}, bekam '{u[k][:60]}'", 1, False)
                    return u
            if not all(isinstance(v, str) for v in u.values()):
                pruefe(f"(h) Fall {name}: Urteile sind Strings", 1, False)
        pruefe(f"(h) Fall {name} -> {erwartet}", 0, True)
        return u

    fall("alles gut", erwartet={"L-0": "H1", "L-1": "H1", "L-2": "H1", "L-3": "beschreibend"})
    fall("nur top_48 aequivalent", {"ebene_6": 0.90}, erwartet={"L-1": "H1: top_48"})
    fall("nur ebene_6 aequivalent", {"top_48": 0.90}, erwartet={"L-1": "H1: ebene_6"})
    fall("alles > 1 pp schlechter", {"ebene_6": 0.915, "top_48": 0.92, "top_32": 0.91, "top_16": 0.90,
                                     "identitaet": 0.95, "zufall_16": 0.8995},
         erwartet={"L-0": "H0", "L-1": "H0", "L-2": "H0"})
    fall("knapp 1 pp darunter, verrauscht", {"ebene_6": 0.93, "top_48": 0.93, "top_32": 0.93, "zufall_16": 0.915},
         rausch=0.02, erwartet={"L-1": "HU:", "L-2": "HU"})
    fall("Ladegewinn zu klein", {"alle_pi2": 0.93}, erwartet={"L-1": "HU (Vorbedingung)"})
    fall("keine Stufe unter 50 %", T_aend={a: 3000.0 for a in KOSTEN_ARME if a != "referenz"},
         erwartet={"L-1": "HU: keine Stufe"})
    # Hauptstufe ueber 50 % -> faellt aus der Familie; top_48 gut, aber zu teuer -> nicht H1
    u = fall("top_48 zu teuer, ebene_6 schlecht", {"ebene_6": 0.90}, T_aend={"top_48": 1800.0},
             erwartet={"L-1": "H"})
    pruefe("(h) teure Hauptstufe zaehlt nicht fuer H1", 0, not u["L-1"].startswith("H1"))
    # Holm direkt: zwei Hauptstufen, p = 0,03 und 0,2 -> keine Ablehnung; eine Stufe mit 0,03 -> Ablehnung
    h2 = holm({"ebene_6": 0.03, "top_48": 0.2})
    h1 = holm({"ebene_6": 0.03})
    pruefe("(h) Holm: 0,03 bei zwei Tests nicht signifikant, bei einem schon", h2["ebene_6"],
           h2["ebene_6"] >= ALPHA and h1["ebene_6"] < ALPHA)
    # TOST-Richtungen
    d = np.full(10, 0.0) + np.linspace(-0.001, 0.001, 10)
    t = _tost(d)
    pruefe("(h) TOST: d ~ 0 -> p_TOST = 1/1024", t["p_tost"], abs(t["p_tost"] - 1 / 1024) < 1e-12)
    t = _tost(d - 0.02)
    pruefe("(h) TOST: d ~ -2 pp -> schlechter p = 1/1024", t["p_schlechter"], abs(t["p_schlechter"] - 1 / 1024) < 1e-12)
    import json
    json.dumps(urteil(auswertung(_synth(rng, basis, T))))
    json.dumps(auswertung(_synth(rng, basis, T)))
    pruefe("(i) Auswertung und Urteil JSON-serialisierbar", 0, True)
    vortest.tests = tests
    return ok


def drucke(e):
    a = e["auswertung"]
    print(f"\nI-033 Lade-Magie (MNIST 16x16, Seeds {e['seeds']}), duenn exakt {a['T_duenn_exakt']:.0f} T, "
          f"Grenze {a['grenze_T']:.0f} T")
    print(" Arm           |  T Laden | Anteil | Genauigkeit (±sd) | d zur Ref. pp | p_TOST")
    for arm, s in a["acc"].items():
        st = a["L1"]["stufen"].get(arm, {"mittel": float("nan")})
        t = a["T_laden"].get(arm, {"mittel": float("nan"), "anteil_duenn": float("nan")})
        print(f" {arm:13s} | {t['mittel']:8.0f} | {t['anteil_duenn'] or 0:6.3f} | {s['mittel']:.4f} ± {s['std']:.4f} | "
              f"{100 * st.get('mittel', 0):+8.2f} | {st.get('p_tost', float('nan')):.4f}")
    for k, v in e["urteil"].items():
        print(f" {k}: {v}")


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
    if "--vortest" in sys.argv:
        sys.exit()
    e = lauf(schnell=schnell)
    drucke(e)
    print(speichere("I-033" + ("-schnell" if schnell else ""), e))
