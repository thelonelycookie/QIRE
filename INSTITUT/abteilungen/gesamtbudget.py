"""Abteilung Clifford/Hardware (Karte I-020, Quantenstufe Q3) -- das GESAMT-Magie-Budget: Laden + Schaltung.

HEILMEIER:
- Was wollen wir?  Die kleinste T-Zahl der GANZEN Quanten-Inferenz (Bild laden +
  gelernte Schaltung) fuer eine gegebene Genauigkeit -- als Pareto-Kurve
  „T gesamt gegen Testgenauigkeit“ auf MNIST 16x16 (n = 8 Qubits).
- Wie heute?  C-kodierung: exakte Amplitudenkodierung (Moettoenen) ~7.636 T,
  volle Schaltung k = 32 ~989 T -> „exakt + k=32“ ~8.625 T. Billige Kodierungen
  wurden dort nur mit neu trainiertem KOPF getestet (MPS chi=2: -5,7 pp).
- Was ist neu?  (1) Die QIRE-Zelle wird DIREKT auf der jeweiligen (genaeherten)
  Kodierung trainiert: Eingang sind die Amplituden des Zustands, den die billige
  Praeparationsschaltung wirklich erzeugt (je Bild einmal vorab berechnet und im
  Speicher gehalten). Danach Magie-Rundung mit k in {0, 2, 4, 32} (frischer Kopf,
  wie clifford._gerundet). (2) Eine DUENN BESETZTE Zustandspraeparation (71 % der
  Pixel sind 0), siehe unten. (3) Kodierung auf dem pi/4-Gitter (je ungeradem
  Vielfachen genau 1 T), fuer exakt und duenn.
- Woran merken wir, ob es klappt?  vortest(): die duenne Praeparation reproduziert
  den Zustand bis 1e-6 (qiskit-Statevector UND numpy), braucht bei einem duennen
  Vektor weniger magische Rotationen als Moettoenen; gerundete Zustaende == qiskit;
  Pareto-Funktion auf synthetischen Punkten korrekt; jeder G-1-Zweig erzwungen.

DUENN BESETZTE PRAEPARATION („Gray-Code-Baum nur ueber besetzte Aeste“):
  Moettoenen-Baum: Stufe j setzt Qubit j mit einer gleichfoermig kontrollierten
  Ry (UCRY) mit 2^j Winkeln theta_c (c = Praefix). Die Gray-Code-Zerlegung
  (Moettoenen 2004) ersetzt die UCRY durch 2^j Ry(phi_i) und 2^j CNOT mit
      theta_c = sum_i (-1)^{popcount(c & g_i)} phi_i ,   g_i = i XOR (i >> 1).
  Fuer einen Praefix c, dessen Teilbaum LEER ist (Amplitude 0), ist theta_c
  beliebig („don't care“). Nur die f_j BESETZTEN Praefixe legen Gleichungen fest.
  Wir suchen ein phi mit moeglichst wenigen Nicht-Null-Eintraegen, das genau diese
  f_j Gleichungen erfuellt (Orthogonal Matching Pursuit auf den Walsh-Zeilen, bei
  Misserfolg Spalten-pivotisierte QR -> hoechstens f_j Rotationen). Nullwinkel
  werden weggelassen (Ry(0) = 1), CNOTs sind Clifford (0 T). Rotationen gesamt
  <= sum_j f_j statt 2^n - 1 = 255. Der Zustand ist EXAKT derselbe; die Zelle
  sieht also denselben Eingang wie bei „exakt“ -> gleiche Genauigkeit, nur das
  Laden ist billiger. (OMP ist eine Heuristik: die T-Zahl ist eine obere Schranke
  fuer diese Schaltungsfamilie, nicht das Optimum.)

T-ZAEHLUNG wie kodierung.py: Schaltung in {cx, ry, rz, Clifford} (uebersetze),
  jede Nicht-Clifford-Rotation einzeln mit gridsynth bei eps = 1e-3 (EPS);
  pi/4-Gitter: jede ungerade Vielfache genau 1 T. MPS chi <= 2: eigene reelle
  Konstruktion; chi = 4: qiskit.Isometry (obere Schranke; die Parameterzahl als
  optimistische Untergrenze wird mitberichtet). T_Schaltung(k) = Steigung·k +
  Achsenabschnitt aus der C-hardware-Anpassung (grid_1e-3, ~30,9 T je magischem
  Winkel, R² = 1,000), T_Schaltung(0) = 0 exakt (H-1 in C-hardware).
  T_Laden = Mittel ueber eine Stichprobe von Testbildern (N_KOSTEN).

ABWEICHUNG VON PROTOKOLLREGEL 1 (offen benannt): Es wird KEINE eigene
  Lernratensuche je Arm gemacht. Zelle: lr = 0,08 (in C-mnist fuer „voll“ auf
  MNIST gesucht, Optimum innen, nicht am Rand); Rundungskoepfe: lr = 0,08 (in
  C-mnist fuer „rundung“ gesucht, innen); linearer Klassifikator: lr = 0,005
  (C-mnist „dicht_linear“, innen). Fuer die genaeherten Kodierungen (MPS, pi/4)
  ist diese Rate NICHT gesucht -- ihre Genauigkeiten koennen dadurch leicht
  unterschaetzt sein (konservativ gegen G-1).

VORREGISTRIERUNG (Regel 7, steht vor jeder Zahl):
  G-1 (aus der Karte): Es gibt eine Kombination (Kodierung, k) mit
      T_gesamt <= 25 % von T_gesamt(„exakt + k=32“)   (Karte: ~8.625 T -> ~2.156 T)
      deren mittlere Testgenauigkeit hoechstens 2 pp unter „exakt + k=32“ liegt.
      Beschreibend ueber 3 Seeds (70, 71, 72), Mittelwerte; KEIN Signifikanztest
      (3 Seeds tragen keinen) -- das Urteil ist eine Beschreibung, kein Nachweis.
      Zweige: „H1“ (Kandidat mit >= 0,5 pp Abstand zur 2-pp-Grenze), „H1 (knapp)“
      (bester Kandidat liegt weniger als 0,5 pp ueber der Grenze), „H0“ (kein
      Kandidat -> Abbruchkriterium der Karte erfuellt). Zusaetzlich wird fuer jeden
      Kandidaten berichtet, ob die 2-pp-Grenze in ALLEN Seeds einzeln haelt.
  G-2 (beschreibend): Anteil der Laden-T an der Gesamt-T fuer jeden Punkt der
      Pareto-Front.
  Referenzlinien: „exakt + k=32“ (volle Schaltung, exakte Kodierung) und ein
  klassischer linearer Klassifikator auf den Pixeln (0 T, nur zur Einordnung).

GRENZEN (vorab): rauschfrei simuliert; eps je Rotation fest (nicht als
  Gesamtbudget optimiert, T_budget_gesamt mitberichtet); OMP-Duennloesung und
  SVD-MPS sind nicht optimal; chi = 4 ueber allgemeine qiskit-Isometrie (T eher zu
  hoch); T_Laden aus einer Stichprobe, nicht aus allen 5.000 Testbildern.
"""
from __future__ import annotations

import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from abteilungen.clifford import QireZelle, _gerundet, daten
from abteilungen.kodierung import (EPS, ist_clifford, kosten, moettoenen, mps_parameter, mps_schaltung,
                                   mps_zerlege, normiere, qk, runde, treue, uebersetze)
from labor.gemeinsam import ERGEBNISSE, genauigkeit, parallel, speichere, trainiere, zusammenfassung

HYPOTHESEN = {
    "G-1": "Vorregistriert (Karte I-020): Es gibt eine Kombination (Kodierung, k) mit hoechstens 25 % der "
           "Gesamt-T von 'exakt + k=32', deren mittlere Testgenauigkeit (Seeds 70-72) hoechstens 2 pp "
           "darunter liegt. Beschreibend, kein Signifikanztest.",
    "G-2": "Beschreibend: Anteil der Laden-T an der Gesamt-T auf der Pareto-Front.",
}

N = 8                          # Qubits (16x16 = 256 Amplituden)
SEEDS = (70, 71, 72)
K_STUFEN = (0, 2, 4, 32)
LR_ZELLE = 0.08                # C-mnist 'voll', innen
LR_RUND = 0.08                 # C-mnist 'rundung', innen
LR_LINEAR = 0.005              # C-mnist 'dicht_linear', innen
SCHRITTE = 1500
N_KOSTEN_JE_SEED = 10          # Stichprobe fuer T_Laden: 10 Testbilder je Seed = 30
ANTEIL_T = 0.25
TOLERANZ = 0.02
KNAPP = 0.005
PI4 = math.pi / 4
TOL_BESETZT = 1e-24            # Quadratnorm eines Teilbaums, ab der er „besetzt“ ist
T_STEIGUNG_FALLBACK, T_ACHSE_FALLBACK = 30.925, -0.346   # C-hardware grid_1e-3

# Kodierungen, fuer die eine Zelle trainiert wird (duenn == exakt als Zustand)
TRAIN_KODIERUNGEN = ("exakt", "mps1", "mps2", "mps4", "exakt_pi4", "duenn_pi4")
TABELLEN_KODIERUNGEN = ("exakt", "duenn", "mps1", "mps2", "mps4", "exakt_pi4", "duenn_pi4")
ZUSTAND_VON = {"duenn": "exakt"}          # gleiche Amplituden -> gleiche Zelle
NAMEN = {"exakt": "exakt (Moettoenen)", "duenn": "duenn besetzt (exakt)", "mps1": "MPS chi=1",
         "mps2": "MPS chi=2", "mps4": "MPS chi=4 (Isometrie)", "exakt_pi4": "Moettoenen auf pi/4",
         "duenn_pi4": "duenn auf pi/4"}


# ------------------------------------------------------------------ 1. Gray-Code-Baum
def _popcount(x: np.ndarray) -> np.ndarray:
    c = np.zeros_like(x)
    for b in range(16):
        c += (x >> b) & 1
    return c


_M = {}


def gray_matrix(k: int) -> np.ndarray:
    """M[c, i] = (-1)^{popcount(c & g_i)}: theta = M phi (UCRY mit k Kontrollen)."""
    if k not in _M:
        i = np.arange(2 ** k)
        _M[k] = (-1.0) ** _popcount(i[:, None] & (i ^ (i >> 1))[None, :])
    return _M[k]


def baum_winkel(psis: np.ndarray):
    """Moettoenen-Baumwinkel theta_j (B, 2^j) und Besetzungsmasken (B, 2^j), j = 0..n-1.
    Modell-Qubit 0 = hoechstwertiges Bit; letzte Stufe mit Vorzeichen."""
    psis = np.atleast_2d(np.asarray(psis, dtype=np.float64))
    B, D = psis.shape
    n = int(round(math.log2(D)))
    th, bes = [], []
    for j in range(n):
        v2 = psis.reshape(B, 2 ** j, 2, -1)
        if j < n - 1:
            a0, a1 = np.linalg.norm(v2[:, :, 0, :], axis=-1), np.linalg.norm(v2[:, :, 1, :], axis=-1)
        else:
            a0, a1 = v2[:, :, 0, 0], v2[:, :, 1, 0]
        th.append(2 * np.arctan2(a1, a0))
        bes.append(a0 ** 2 + a1 ** 2 > TOL_BESETZT)
    return th, bes


def phi_moettoenen(thetas):
    """Gray-Code-Winkel der vollen UCRY: phi = M^T theta / 2^j."""
    return [t @ gray_matrix(j) / 2 ** j for j, t in enumerate(thetas)]


def _omp(A: np.ndarray, t: np.ndarray, tol=1e-11):
    """Orthogonal Matching Pursuit: duennes x mit A x = t (exakt bis tol)."""
    S, x, r = [], np.zeros(0), t.copy()
    for _ in range(A.shape[0]):
        if np.linalg.norm(r) < tol:
            break
        c = np.abs(A.T @ r)
        c[S] = -1.0
        S.append(int(np.argmax(c)))
        x, *_ = np.linalg.lstsq(A[:, S], t, rcond=None)
        r = t - A[:, S] @ x
    return S, x, float(np.linalg.norm(r))


def phi_duenn_einzeln(theta_j: np.ndarray, besetzt_j: np.ndarray, j: int) -> np.ndarray:
    """Duennste gefundene Gray-Code-Winkel, die nur die besetzten Praefixe festlegen."""
    M = gray_matrix(j)
    F = np.flatnonzero(besetzt_j)
    if len(F) == len(theta_j):
        return theta_j @ M / 2 ** j
    phi = np.zeros(2 ** j)
    if len(F) == 0:
        return phi
    A, t = M[F], theta_j[F]
    S, x, res = _omp(A, t)
    if res > 1e-9:                          # Rueckfall: pivotisierte QR -> genau |F| Spalten
        import scipy.linalg as sl
        _, _, piv = sl.qr(A, pivoting=True, mode="economic")
        S = list(piv[:len(F)])
        x = np.linalg.solve(A[:, S], t)
    phi[S] = x
    phi[np.abs(phi) < 1e-12] = 0.0
    return phi


def phi_duenn(psis: np.ndarray):
    """Fuer B Bilder: Liste je Stufe j von (B, 2^j) duennen Gray-Code-Winkeln."""
    th, bes = baum_winkel(psis)
    return [np.stack([phi_duenn_einzeln(th[j][b], bes[j][b], j) for b in range(th[j].shape[0])])
            for j in range(len(th))]


def baum_zustand(phis) -> np.ndarray:
    """Zustand der Gray-Code-Schaltung (numpy, alle Bilder zugleich): Stufe j wendet
    auf Praefix c die Drehung Ry(theta_c), theta = M phi, an (bewiesen in der
    Kopfdoku; vortest vergleicht gatterweise mit qiskit)."""
    B = phis[0].shape[0]
    v = np.ones((B, 1))
    for j, phi in enumerate(phis):
        th = phi @ gray_matrix(j).T
        v = np.stack([v * np.cos(th / 2), v * np.sin(th / 2)], axis=-1).reshape(B, -1)
    return v


def runde_phis(phis, gitter=PI4):
    return [np.round(p / gitter) * gitter for p in phis]


def gray_schaltung(phis_einzeln, n=N) -> QuantumCircuit:
    """Explizite Schaltung (nur ry und cx) fuer EIN Bild: phis_einzeln[j] hat Laenge 2^j.
    Rotation phi = 0 wird weggelassen. Nach Ry(phi_i) folgt CNOT vom Kontrollbit, in dem
    sich g_i und g_{i+1} unterscheiden (zyklisch); Kontrollbit b des Praefix = Modell-
    Qubit j-1-b; Modell-Qubit q = Qiskit-Qubit n-1-q (wie hardware.py)."""
    qc = QuantumCircuit(n)
    for j, phi in enumerate(phis_einzeln):
        ziel = qk(j, n)
        D = 2 ** j
        for i in range(D):
            if abs(phi[i]) > 1e-12:
                qc.ry(float(phi[i]), ziel)
            if j == 0:
                continue
            bit = ((i + 1) & -(i + 1)).bit_length() - 1 if i < D - 1 else j - 1
            qc.cx(qk(j - 1 - bit, n), ziel)
    return qc


def _einzeln(phis, b):
    return [p[b] for p in phis]


# ------------------------------------------------------------------ 2. Zustaende (Cache)
def zustaende(psis: np.ndarray, kodierung: str) -> np.ndarray:
    """Amplituden des Zustands, den die Kodierungsschaltung WIRKLICH erzeugt (reell)."""
    if kodierung in ("exakt", "duenn"):
        return psis.copy()
    if kodierung.startswith("mps"):
        chi = int(kodierung[3:])
        return np.stack([mps_zerlege(p, N, chi)[1] for p in psis])
    if kodierung == "exakt_pi4":
        return baum_zustand(runde_phis(phi_moettoenen(baum_winkel(psis)[0])))
    if kodierung == "duenn_pi4":
        return baum_zustand(runde_phis(phi_duenn(psis)))
    raise ValueError(kodierung)


# ------------------------------------------------------------------ 3. T-Kosten
def _t_schaltung_anpassung():
    d = sorted(ERGEBNISSE.glob("C-hardware__*.json"))
    if not d:
        return T_STEIGUNG_FALLBACK, T_ACHSE_FALLBACK, "Fallback"
    lin = json.loads(d[-1].read_text())["datensaetze"]["mnist"]["linear"]["grid_1e-3"]
    return float(lin["steigung_T_je_winkel"]), float(lin["achsenabschnitt"]), d[-1].name


def t_schaltung(k: int, steigung=T_STEIGUNG_FALLBACK, achse=T_ACHSE_FALLBACK) -> float:
    """T der gelernten Schaltung mit k magischen Winkeln (eps = 1e-3); k = 0 -> exakt 0."""
    return 0.0 if k == 0 else max(0.0, steigung * k + achse)


def _kosten_bild(psi: np.ndarray) -> dict:
    """Alle Kodierungen fuer ein Bild: T (eps je Rotation), magische Rotationen, CNOT, Treue."""
    aus = {"nichtnull": int(np.count_nonzero(psi))}
    k = kosten(uebersetze(moettoenen(psi, N)))
    aus["exakt"] = {**k, "treue": 1.0}
    phm = [p[0] for p in phi_moettoenen(baum_winkel(psi)[0])]
    phd = _einzeln(phi_duenn(psi[None]), 0)
    qd = gray_schaltung(phd)
    kd = kosten(uebersetze(qd))
    aus["duenn"] = {**kd, "treue": treue(Statevector(qd).data.real, psi),
                    "rotationen_vorher": int(sum(np.count_nonzero(p) for p in phd))}
    for name, ph in (("exakt_pi4", phm), ("duenn_pi4", phd)):
        r = runde(gray_schaltung(ph), PI4)
        aus[name] = {**kosten(r), "treue": treue(Statevector(r).data.real, psi)}
    for chi in (1, 2, 4):
        B, v = mps_zerlege(psi, N, chi)
        kk = kosten(uebersetze(mps_schaltung(B, N, "eigen" if chi <= 2 else "qiskit")))
        kk["parameter_untergrenze"] = mps_parameter(B)
        kk["T_untergrenze"] = kk["parameter_untergrenze"] * kk["T"] / max(kk["rotationen_magisch"], 1e-9)
        aus[f"mps{chi}"] = {**kk, "treue": treue(v, psi)}
    return aus


# ------------------------------------------------------------------ 4. Training
def _daten(seed, schnell):
    xtr, ytr, xte, yte = daten("mnist", seed)
    if schnell:
        xtr, ytr, xte, yte = xtr[:600], ytr[:600], xte[:300], yte[:300]
    return xtr, ytr, xte, yte


def _training(seed, kodierung, schritte, schnell):
    xtr, ytr, xte, yte = _daten(seed, schnell)
    t0 = time.time()
    ztr = torch.tensor(zustaende(normiere(xtr.numpy()), kodierung), dtype=torch.float32)
    zte = torch.tensor(zustaende(normiere(xte.numpy()), kodierung), dtype=torch.float32)
    t_zust = time.time() - t0
    treue_te = float(np.mean(np.sum(zte.numpy().astype(np.float64) * normiere(xte.numpy()), 1) ** 2))
    torch.manual_seed(seed)
    voll = QireZelle(n=N)
    trainiere(voll, ztr, ytr, schritte, LR_ZELLE, seed=seed)
    with torch.no_grad():
        z = {"seed": seed, "kodierung": kodierung, "treue_test": treue_te, "sekunden_zustaende": t_zust,
             "voll_eigener_kopf": genauigkeit(voll(zte), yte), "k": {}}
    for k in K_STUFEN:
        m = _gerundet(voll, k, seed)
        trainiere(m, ztr, ytr, schritte, LR_RUND, seed=seed)
        with torch.no_grad():
            z["k"][k] = genauigkeit(m(zte), yte)
    z["sekunden"] = time.time() - t0
    return z


def _linear(seed, schritte, schnell):
    xtr, ytr, xte, yte = _daten(seed, schnell)
    torch.manual_seed(seed)
    m = torch.nn.Linear(xtr.shape[1], 10)
    trainiere(m, xtr, ytr, schritte, LR_LINEAR, seed=seed)
    with torch.no_grad():
        return {"seed": seed, "linear": genauigkeit(m(xte), yte)}


def _aufgabe(a):
    art = a[0]
    if art == "training":
        return _training(*a[1:])
    if art == "linear":
        return _linear(*a[1:])
    if art == "kosten":
        _, seed, i = a
        _, _, xte, _ = daten("mnist", seed)
        return {"seed": seed, "bild": i, **_kosten_bild(normiere(xte[i].numpy()))}
    raise ValueError(art)


# ------------------------------------------------------------------ 5. Auswertung
def pareto(punkte):
    """punkte: Liste von dicts mit 'T' und 'acc'. Rueckgabe: nicht dominierte Punkte
    (kein anderer Punkt hat T <= und acc >= mit mindestens einer echten Ungleichung),
    nach T aufsteigend."""
    s = sorted(punkte, key=lambda p: (p["T"], -p["acc"]))
    front, best = [], -math.inf
    for p in s:
        if p["acc"] > best:
            front.append(p)
            best = p["acc"]
    return front


def tabelle_bauen(kosten_mittel: dict, genau: dict, t_k) -> list:
    """kosten_mittel[kod] = mittlere T_Laden; genau[(kod, k)] = Liste der Genauigkeiten je Seed
    (Seed-Reihenfolge fest); t_k(k) = T_Schaltung."""
    zeilen = []
    for kod in TABELLEN_KODIERUNGEN:
        quelle = ZUSTAND_VON.get(kod, kod)
        for k in K_STUFEN:
            if (quelle, k) not in genau or kod not in kosten_mittel:
                continue
            accs = genau[(quelle, k)]
            tl, ts = kosten_mittel[kod], t_k(k)
            zeilen.append({"kodierung": kod, "k": k, "T_laden": tl, "T_schaltung": ts, "T": tl + ts,
                           "acc": float(np.mean(accs)), "acc_je_seed": list(accs),
                           "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                           "anteil_laden": tl / (tl + ts) if tl + ts > 0 else None})
    return zeilen


def urteil_g1(zeilen: list, ref: dict) -> dict:
    """ref: Zeile 'exakt + k=32'. Vorregistrierter Zweig, siehe Kopfdoku."""
    grenze_T = ANTEIL_T * ref["T"]
    grenze_acc = ref["acc"] - TOLERANZ
    kand = []
    for z in zeilen:
        if z["T"] <= grenze_T and z["acc"] >= grenze_acc:
            je_seed = [a - r for a, r in zip(z["acc_je_seed"], ref["acc_je_seed"])]
            kand.append({"kodierung": z["kodierung"], "k": z["k"], "T": z["T"], "acc": z["acc"],
                         "abstand_pp": 100 * (z["acc"] - ref["acc"]),
                         "haelt_in_allen_seeds": bool(all(d >= -TOLERANZ for d in je_seed))})
    kand.sort(key=lambda c: (-c["acc"], c["T"]))
    g = {"grenze_T": grenze_T, "grenze_acc": grenze_acc, "ref_T": ref["T"], "ref_acc": ref["acc"],
         "kandidaten": kand}
    if not kand:
        g["urteil"] = (f"H0: keine Kombination mit <= {grenze_T:.0f} T haelt die Genauigkeit innerhalb 2 pp "
                       f"von {ref['acc']:.4f} -- Abbruchkriterium der Karte erfuellt (beschreibend, "
                       f"{len(ref['acc_je_seed'])} Seed(s), kein Signifikanztest)")
    else:
        b = kand[0]
        rand = b["acc"] - grenze_acc
        marke = "H1 (knapp)" if rand < KNAPP else "H1"
        g["urteil"] = (f"{marke}: {NAMEN.get(b['kodierung'], b['kodierung'])} + k={b['k']} mit "
                       f"{b['T']:.0f} T ({100 * b['T'] / ref['T']:.1f} % von {ref['T']:.0f}) und "
                       f"{b['acc']:.4f} ({b['abstand_pp']:+.2f} pp; in allen Seeds innerhalb 2 pp: "
                       f"{'ja' if b['haelt_in_allen_seeds'] else 'nein'}); {len(kand)} Kandidat(en). "
                       f"Beschreibend, kein Signifikanztest")
    return g


def g2(front: list) -> dict:
    return {"front": [{"kodierung": p["kodierung"], "k": p["k"], "T": p["T"], "acc": p["acc"],
                       "anteil_laden": p["anteil_laden"]} for p in front],
            "anteil_laden_mittel": float(np.mean([p["anteil_laden"] for p in front
                                                  if p["anteil_laden"] is not None] or [float("nan")]))}


# ------------------------------------------------------------------ Vortest
def vortest(verbose=True) -> bool:
    ok, tests = True, []

    def pruefe(name, wert, bed):
        nonlocal ok
        tests.append((name, float(wert), bool(bed)))
        ok &= bool(bed)
        if verbose:
            print(f"  {name}: {wert:.3g}  {'ok' if bed else 'FEHLER'}")

    rng = np.random.default_rng(5)
    _, _, xte, _ = daten("mnist", SEEDS[0])
    bilder = [normiere(xte[i].numpy()) for i in range(3)]
    duenn = np.zeros(256)
    duenn[rng.choice(256, 10, replace=False)] = rng.normal(size=10)      # mit Vorzeichen
    zufall = normiere(rng.normal(size=256))
    for name, psi in [("MNIST0", bilder[0]), ("MNIST1", bilder[1]), ("MNIST2", bilder[2]),
                      ("duenn10", normiere(duenn)), ("dicht", zufall)]:
        th, _ = baum_winkel(psi)
        phm = _einzeln(phi_moettoenen(th), 0)
        phd = _einzeln(phi_duenn(psi[None]), 0)
        for art, ph in (("Moettoenen-Gray", phm), ("duenn", phd)):
            qc = gray_schaltung(ph)
            sv = Statevector(qc).data
            pruefe(f"(a) {art} {name}: qiskit-Zustand == psi", np.abs(sv - psi).max(),
                   np.abs(sv - psi).max() < 1e-6 and set(qc.count_ops()) <= {"ry", "cx"})
            pruefe(f"(a) {art} {name}: numpy-Baum == psi", np.abs(baum_zustand([p[None] for p in ph])[0] - psi).max(),
                   np.abs(baum_zustand([p[None] for p in ph])[0] - psi).max() < 1e-6)
        # (c) pi/4-Rundung: numpy == qiskit, T == #ungerade Vielfache
        for art, ph in (("Moettoenen-Gray", phm), ("duenn", phd)):
            r = runde(gray_schaltung(ph), PI4)
            vn = baum_zustand([p[None] for p in runde_phis(ph)])[0]
            pruefe(f"(c) {art} {name} pi/4: numpy == qiskit", np.abs(Statevector(r).data - vn).max(),
                   np.abs(Statevector(r).data - vn).max() < 1e-6)
            ungerade = sum(int(round(a / PI4)) % 2 for p in ph for a in p)
            pruefe(f"(c) {art} {name} pi/4: T - #ungerade", kosten(r)["T"] - ungerade, kosten(r)["T"] == ungerade)
        if name in ("duenn10", "MNIST0"):
            rot_m = kosten(uebersetze(moettoenen(psi, N)))["rotationen_magisch"]
            rot_d = kosten(uebersetze(gray_schaltung(phd)))["rotationen_magisch"]
            pruefe(f"(b) {name}: Rotationen duenn {rot_d} < Moettoenen {rot_m}", rot_d - rot_m, rot_d < rot_m)
    # (b') dicht besetzter Vektor: duenn darf nicht schlechter sein als Moettoenen (gleiche Loesung)
    phd = _einzeln(phi_duenn(zufall[None]), 0)
    phm = _einzeln(phi_moettoenen(baum_winkel(zufall)[0]), 0)
    pruefe("(b') dichter Vektor: duenn == Moettoenen-Gray", max(np.abs(a - b).max() for a, b in zip(phd, phm)),
           max(np.abs(a - b).max() for a, b in zip(phd, phm)) < 1e-12)
    # (d) Pareto auf synthetischen Punkten
    P = [{"T": 10, "acc": 0.5, "id": "a"}, {"T": 5, "acc": 0.6, "id": "b"}, {"T": 20, "acc": 0.9, "id": "c"},
         {"T": 20, "acc": 0.8, "id": "d"}, {"T": 0, "acc": 0.3, "id": "e"}, {"T": 30, "acc": 0.85, "id": "f"},
         {"T": 5, "acc": 0.6, "id": "g"}]
    front = [p["id"] for p in pareto(P)]
    pruefe("(d) Pareto-Front == [e, b, c]", 0 if front == ["e", "b", "c"] else 1, front == ["e", "b", "c"])
    # (e) G-1-Zweige erzwingen
    t_k = lambda k: 0.0 if k == 0 else 31.0 * k
    def genau_fall(acc_mps2_k4, acc_ref=0.93, streu=(0.0, 0.0, 0.0)):
        g = {}
        for kod in TRAIN_KODIERUNGEN:
            for k in K_STUFEN:
                g[(kod, k)] = [0.5 + s for s in streu]
        g[("exakt", 32)] = [acc_ref + s for s in (0.0, 0.0, 0.0)]
        g[("mps2", 4)] = [acc_mps2_k4 + s for s in streu]
        return g
    km = {"exakt": 7636.0, "duenn": 3500.0, "mps1": 225.0, "mps2": 1176.0, "mps4": 6329.0,
          "exakt_pi4": 14.0, "duenn_pi4": 12.0}
    faelle = [(0.925, (0, 0, 0), "H1"), (0.912, (0, 0, 0), "H1 (knapp)"), (0.90, (0, 0, 0), "H0"),
              (0.925, (0.01, 0.01, -0.03), "H1")]
    for acc, streu, erw in faelle:
        z = tabelle_bauen(km, genau_fall(acc, streu=streu), t_k)
        ref = next(r for r in z if r["kodierung"] == "exakt" and r["k"] == 32)
        u = urteil_g1(z, ref)
        treffer = u["urteil"].split(":")[0] == erw
        pruefe(f"(e) G-1 Fall acc={acc} streu={streu} -> {erw}", 0 if treffer else 1, treffer)
        if streu[2] < 0:
            pruefe("(e) Einzelseed-Pruefung erkennt Ausreisser", 0, not u["kandidaten"][0]["haelt_in_allen_seeds"])
    # H0, weil billig aber zu ungenau und genau aber zu teuer
    km2 = dict(km, mps2=5000.0)
    z = tabelle_bauen(km2, genau_fall(0.93), t_k)
    ref = next(r for r in z if r["kodierung"] == "exakt" and r["k"] == 32)
    pruefe("(e) G-1: genau, aber zu teuer -> H0", 0, urteil_g1(z, ref)["urteil"].startswith("H0"))
    # (f) T_Schaltung
    st, ac, _ = _t_schaltung_anpassung()
    pruefe("(f) T_Schaltung(0) == 0", t_schaltung(0, st, ac), t_schaltung(0, st, ac) == 0)
    pruefe("(f) T_Schaltung(32) ~ 989", t_schaltung(32, st, ac), 950 < t_schaltung(32, st, ac) < 1030)
    vortest.tests = tests
    return ok


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False) -> dict:
    t0 = time.time()
    seeds = SEEDS[:1] if schnell else SEEDS
    schritte = 150 if schnell else SCHRITTE
    n_kosten = 1 if schnell else N_KOSTEN_JE_SEED
    ok = vortest(verbose=False)
    st, ac, quelle = _t_schaltung_anpassung()
    t_k = lambda k: t_schaltung(k, st, ac)
    aufg = [("training", s, kod, schritte, schnell) for s in seeds for kod in TRAIN_KODIERUNGEN]
    aufg += [("kosten", s, i) for s in (SEEDS if schnell else seeds) for i in range(n_kosten)]
    aufg += [("linear", s, schritte, schnell) for s in seeds]
    erg = parallel(_aufgabe, aufg)
    trainings = [e for a, e in zip(aufg, erg) if a[0] == "training"]
    kost = [e for a, e in zip(aufg, erg) if a[0] == "kosten"]
    lin = [e for a, e in zip(aufg, erg) if a[0] == "linear"]

    kosten_mittel, kosten_info = {}, {}
    for kod in TABELLEN_KODIERUNGEN:
        ts = [c[kod]["T"] for c in kost]
        kosten_mittel[kod] = float(np.mean(ts))
        kosten_info[kod] = {"T_mittel": float(np.mean(ts)), "T_min": int(min(ts)), "T_max": int(max(ts)),
                            "rotationen_magisch": float(np.mean([c[kod]["rotationen_magisch"] for c in kost])),
                            "cnot": float(np.mean([c[kod]["cnot"] for c in kost])),
                            "T_budget_gesamt": float(np.mean([c[kod]["T_budget_gesamt"] for c in kost])),
                            "treue_stichprobe": float(np.mean([c[kod]["treue"] for c in kost]))}
        if "T_untergrenze" in kost[0][kod]:
            kosten_info[kod]["T_untergrenze"] = float(np.mean([c[kod]["T_untergrenze"] for c in kost]))
    genau = {}
    for kod in TRAIN_KODIERUNGEN:
        zs = sorted([z for z in trainings if z["kodierung"] == kod], key=lambda z: z["seed"])
        for k in K_STUFEN:
            genau[(kod, k)] = [z["k"][k] for z in zs]
    zeilen = tabelle_bauen(kosten_mittel, genau, t_k)
    ref = next(z for z in zeilen if z["kodierung"] == "exakt" and z["k"] == 32)
    front = pareto(zeilen)
    lin_acc = [l["linear"] for l in sorted(lin, key=lambda l: l["seed"])]
    ergebnis = {
        "abteilung": "clifford/hardware", "karte": "I-020", "quantenstufe": "Q3 (lokal simuliert)",
        "hypothesen": HYPOTHESEN, "vortest_bestanden": bool(ok),
        "vortest": vortest.tests, "seeds": list(seeds), "schritte": schritte,
        "lr": {"zelle": LR_ZELLE, "rundung": LR_RUND, "linear": LR_LINEAR,
               "hinweis": "aus C-mnist uebernommen, keine eigene Suche (Abweichung von Regel 1)"},
        "eps_je_rotation": EPS, "t_schaltung": {"steigung": st, "achse": ac, "quelle": quelle,
                                                **{f"k{k}": t_k(k) for k in K_STUFEN}},
        "kosten_stichprobe": {"bilder": len(kost), "je_kodierung": kosten_info,
                              "nichtnull_mittel": float(np.mean([c["nichtnull"] for c in kost])),
                              "beispiele": kost[:3]},
        "trainings": trainings, "tabelle": zeilen, "pareto_front": front,
        "referenz_exakt_k32": ref,
        "referenz_linear": {"acc_je_seed": lin_acc, "acc": float(np.mean(lin_acc)), "T": 0},
    }
    ergebnis["urteil"] = {"G-1": urteil_g1(zeilen, ref), "G-2": g2(front)}
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


def drucke(e):
    print(f"\nPareto-Tabelle (MNIST 16x16, n=8, Seeds {e['seeds']}, eps={e['eps_je_rotation']} je Rotation)")
    print(" Kodierung                 |  k | T Laden | T Schalt. | T gesamt |  Genauigkeit (±sd) | Laden-Anteil | Front")
    ids = {(p["kodierung"], p["k"]) for p in e["pareto_front"]}
    for z in e["tabelle"]:
        a = z["anteil_laden"]
        print(f" {NAMEN[z['kodierung']]:25s} | {z['k']:2d} | {z['T_laden']:7.0f} | {z['T_schaltung']:9.0f} | "
              f"{z['T']:8.0f} | {z['acc']:.4f} ± {z['acc_std']:.4f} | "
              f"{'-' if a is None else f'{a:.3f}':>12s} | {'*' if (z['kodierung'], z['k']) in ids else ''}")
    r, l = e["referenz_exakt_k32"], e["referenz_linear"]
    print(f"\n Referenz exakt + k=32: {r['T']:.0f} T, {r['acc']:.4f};  linearer Klassifikator: 0 T, {l['acc']:.4f}")
    ks = e["kosten_stichprobe"]
    print(f" Kosten-Stichprobe: {ks['bilder']} Bilder, Nicht-Null-Amplituden im Mittel {ks['nichtnull_mittel']:.1f}")
    for kod, k in ks["je_kodierung"].items():
        print(f"  {NAMEN[kod]:25s} T {k['T_mittel']:7.1f} [{k['T_min']}-{k['T_max']}]  mag. Rot. "
              f"{k['rotationen_magisch']:6.1f}  CNOT {k['cnot']:6.1f}  Treue {k['treue_stichprobe']:.3f}"
              + (f"  T-Untergrenze {k['T_untergrenze']:.0f}" if "T_untergrenze" in k else ""))
    print("\n G-1:", e["urteil"]["G-1"]["urteil"])
    print(" G-2:", [(p["kodierung"], p["k"], round(p["anteil_laden"], 3) if p["anteil_laden"] is not None else None)
                    for p in e["urteil"]["G-2"]["front"]])


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
    if "--vortest" in sys.argv:
        sys.exit()
    e = lauf(schnell=schnell)
    drucke(e)
    print(speichere("I-020" + ("-schnell" if schnell else ""), e))
