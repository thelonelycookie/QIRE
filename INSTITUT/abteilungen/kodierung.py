"""Abteilung Clifford/Hardware (Agenda C-kodierung, Quantenstufe Q3) -- was kostet das LADEN des Bildes?

HEILMEIER:
- Was wollen wir?  C-hardware hat die gelernte Schaltung in T-Gatter uebersetzt
  (~31 T je magischem Winkel bei eps = 1e-3, volle Schaltung k = 32 ~ 990 T).
  NICHT mitgezaehlt war die Amplitudenkodierung: der 256-dim. Bildvektor muss
  erst als Zustand |psi> = x/|x| auf 8 Qubits praepariert werden. Wir messen
  diese Kosten in Clifford+T -- und ob es billigere Kodierungen gibt, die die
  Klassifikation nicht zerstoeren.
- Wie heute?  Allgemeine Zustandspraeparation (Moettoenen 2004, Shende-Bullock-
  Markov 2006, Iten 2016) braucht ~2^n Rotationen -- jede ist ein magischer
  Winkel, also ~31 T bei eps = 1e-3.
- Was ist neu?  Die Rechnung fuer UNSERE Bilder und UNSERE Zelle, mit drei
  guenstigeren quantenmechanischen Kodierungen daneben:
    (a) MPS-Naeherung mit Bonddimension chi (Rangreduktion durch abgeschnittene
        SVD), sequentiell praeparierbar (Schoen et al. 2005): Kosten ~ n·chi²
        statt 2^n Rotationen;
    (b) Winkel-/Produktkodierung ein Qubit je Merkmal -- bei 256 Merkmalen 256
        Qubits breit, wird NICHT gerechnet (nur erwaehnt). Der Produktzustand
        auf 8 Qubits ist genau MPS chi = 1;
    (c) gerundete Kodierung: die Ry-Winkel der Praeparationsschaltung werden auf
        das Gitter k·pi/2 (-> 0 T, reine Clifford-Schaltung) bzw. k·pi/4 (-> je
        ungeradem Vielfachen genau 1 T) gerundet.
  Fuer jede Variante: T-Gatter (gridsynth, eps = 1e-3 je Rotation), CNOT, Tiefe,
  Treue |<psi|psi~>|² und die Testgenauigkeit der neu trainierten QIRE-Zelle.
- Woran merken wir, ob es klappt?  vortest(): die Bausteine reproduzieren den
  Zustand bis 1e-6, MPS mit chi = max ist exakt, ein Produktzustand braucht chi = 1.

SCHALTUNGEN (reelle Amplituden -> nur Ry und CNOT, keine Rz noetig):
  * exakt: Moettoenen-Baum aus gleichfoermig kontrollierten Ry (UCRY), Gray-Code-
    Zerlegung -> hoechstens 2^n - 1 Ry und 2^n - 2 CNOT. Zum Vergleich wird auch
    qiskit.StatePreparation gemessen (allgemein komplex, daher zusaetzlich Rz --
    teurer; berichtet, aber nicht als „exakt“-Zeile gewertet).
  * MPS chi <= 2: eigene reelle Konstruktion je Schritt (Isometrie 2 -> 4 aus drei
    UCRY-Lagen, 5 Winkel = Dimension der reellen Stiefel-Mannigfaltigkeit V_2(R^4)).
  * MPS chi = 4: qiskit.Isometry (Iten et al., allgemein komplex) -> obere Schranke;
    zusaetzlich die Parameterzahl der reellen MPS als (optimistische) untere
    Abschaetzung der noetigen Rotationen.

BITORDNUNG: wie hardware.py -- Modell-Qubit q <-> Qiskit-Qubit n-1-q, Basisindex
identisch; ein Amplitudenvektor x ist direkt Statevector(x).

GRENZEN (ehrlich, vorab): 1 Seed fuer die Klassifikation (keine Signifikanz);
eps = 1e-3 JE Rotation (wie C-hardware) -- bei 255 Rotationen summiert sich der
Fehler auf bis zu ~0,26, fair waere ein festes Gesamtbudget (wird zusaetzlich als
T bei eps_gesamt = 1e-2 berichtet); MPS-Abschneiden per SVD ist nicht die optimale
chi-Naeherung (variationell waere etwas besser); duenn besetzte Zustandspraeparation
(71 % der MNIST-Pixel sind 0) wird nicht gemessen.
"""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry, StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import gridsynth_rz

from abteilungen.clifford import QireZelle, daten
from labor.gemeinsam import ERGEBNISSE, bilder, genauigkeit, speichere, trainiere, ziffern

HYPOTHESEN = {
    "K-1": "Vorregistriert: exakte Amplitudenkodierung eines MNIST-16x16-Bildes kostet "
           "MEHR T-Gatter als die ganze gelernte Schaltung mit k = 32 magischen Winkeln "
           "(~990 T bei eps = 1e-3). Entschieden am Mittel ueber 20 Testbilder.",
    "K-2": "Beschreibend (1 Seed, kein Signifikanztest): unter den gemessenen Varianten gibt "
           "es eine Kodierung mit < 10 % der T-Kosten der exakten Kodierung, die weniger als "
           "2 pp Testgenauigkeit verliert (primaer: Zustand direkt in die trainierte Zelle "
           "eingespeist; sekundaer: Kopf auf der Kodierung nachtrainiert).",
}

EPS = 1e-3                 # je Rotation, wie C-hardware
EPS_GESAMT = 1e-2          # zusaetzlich: festes Fehlerbudget fuer die ganze Praeparation
BASIS = ["cx", "ry", "rz", "h", "x", "s", "sdg", "z"]
T_NAMEN = ("t", "tdg")
SEED = 10
LR, SCHRITTE = 0.08, 1500
CHIS = (1, 2, 4)
T_LERN_K32_FALLBACK = 990.0


def qk(q: int, n: int) -> int:
    """Modell-Qubit q -> Qiskit-Qubit (little endian), wie hardware.qk."""
    return n - 1 - q


def normiere(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def treue(a, b) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


# ------------------------------------------------------------------ 1. MPS
def mps_zerlege(psi: np.ndarray, n: int, chi: int):
    """Rechts-kanonische MPS durch SVD von rechts nach links, Bonddimension <= chi.
    B[k] hat Form (chi_links, 2, chi_rechts) mit sum_{s,b} B[a,s,b] B[a',s,b] = delta.
    Rueckgabe: (Liste B, psi_chi normiert). Model-Ordnung: Site 0 = hoechstwertiges Bit."""
    rest = np.asarray(psi, dtype=np.float64).reshape(-1, 1)   # (2^n, 1)
    B = [None] * n
    for k in range(n - 1, 0, -1):
        chi_r = rest.shape[1]
        C = rest.reshape(2 ** k, 2 * chi_r)
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        r = min(chi, len(S))
        B[k] = Vh[:r].reshape(r, 2, chi_r)
        rest = U[:, :r] * S[:r]
    B[0] = rest.reshape(1, 2, -1)
    B[0] = B[0] / np.linalg.norm(B[0])
    return B, mps_vektor(B)


def mps_vektor(B) -> np.ndarray:
    v = B[0].reshape(2, -1)
    for Bk in B[1:]:
        v = (v @ Bk.reshape(Bk.shape[0], -1)).reshape(-1, Bk.shape[2])
    return v.reshape(-1)


def mps_parameter(B) -> int:
    """Freie reelle Parameter der Isometrien (Dimension der Stiefel-Mannigfaltigkeiten):
    V_k(R^d) hat d·k - k(k+1)/2 Dimensionen. Optimistische Untergrenze der Rotationen."""
    p = 0
    for Bk in B:
        k, d = Bk.shape[0], 2 * Bk.shape[2]
        p += d * k - k * (k + 1) // 2
    return p


# ------------------------------------------------------------------ 2. Schaltungen
def _winkel_null(x0, x1):
    """Ry(theta)·(x0,x1) = (r, 0), r >= 0."""
    return -2.0 * math.atan2(x1, x0)


def moettoenen(v, n, start=0, qc=None, gesamt_n=None) -> QuantumCircuit:
    """Reelle Zustandspraeparation (Moettoenen-Baum) von v (Laenge 2^n) auf den
    Modell-Qubits start..start+n-1 (Modell-Qubit start = hoechstwertiges Bit).
    Nur Ry/UCRY -> nach Zerlegung nur ry und cx."""
    N = gesamt_n if gesamt_n is not None else n
    qc = qc if qc is not None else QuantumCircuit(N)
    v = np.asarray(v, dtype=np.float64)
    for j in range(n):
        v2 = v.reshape(2 ** j, 2, -1)
        if j < n - 1:
            a0, a1 = np.linalg.norm(v2[:, 0, :], axis=1), np.linalg.norm(v2[:, 1, :], axis=1)
        else:
            a0, a1 = v2[:, 0, 0], v2[:, 1, 0]           # letzte Stufe: Vorzeichen erlaubt
        ang = 2 * np.arctan2(a1, a0)
        ziel = qk(start + j, N)
        if j == 0:
            qc.ry(float(ang[0]), ziel)
        else:
            # UCRY: Kontrollwert little endian ueber die Liste -> Liste so, dass der
            # Kontrollwert = Praefix (Modell-Qubit start = MSB) ist.
            qc.append(UCRYGate(ang.tolist()), [ziel] + [qk(start + j - 1 - i, N) for i in range(j)])
    return qc


def _iso_2zu4(v0, v1):
    """Winkel fuer G = U3·U2·U1 mit G v_a = e_(a,0); Index i = 2·q1 + q2.
    U1: UCRY Ziel q2, Kontrolle q1 (alpha0, alpha1); U2: UCRY Ziel q1, Kontrolle q2
    (beta0, beta1); U3: UCRY Ziel q2, Kontrolle q1 (0, gamma1)."""
    def ucry_q2(v, a):   # Paare (00,01), (10,11)
        w = v.copy()
        for c in (0, 1):
            co, si = math.cos(a[c] / 2), math.sin(a[c] / 2)
            x0, x1 = w[2 * c], w[2 * c + 1]
            w[2 * c], w[2 * c + 1] = co * x0 - si * x1, si * x0 + co * x1
        return w

    def ucry_q1(v, b):   # Paare (00,10), (01,11)
        w = v.copy()
        for c in (0, 1):
            co, si = math.cos(b[c] / 2), math.sin(b[c] / 2)
            x0, x1 = w[c], w[2 + c]
            w[c], w[2 + c] = co * x0 - si * x1, si * x0 + co * x1
        return w
    v0, v1 = np.asarray(v0, float), np.asarray(v1, float)
    al = [_winkel_null(v0[0], v0[1]), _winkel_null(v0[2], v0[3])]
    v0, v1 = ucry_q2(v0, al), ucry_q2(v1, al)
    be = [_winkel_null(v0[0], v0[2]), 2.0 * math.atan2(v1[1], v1[3])]
    v0, v1 = ucry_q1(v0, be), ucry_q1(v1, be)
    ga = [0.0, _winkel_null(v1[2], v1[3])]
    v1 = ucry_q2(v1, ga)
    return al, be, ga, (v0, v1)


def mps_schaltung(B, n, art="eigen") -> QuantumCircuit:
    """Sequentielle Praeparation der rechts-kanonischen MPS B (Schoen et al. 2005).
    Schritt k: Isometrie |a_{k-1}> -> sum B[a,s,b] |s>|b>, Eingang auf Modell-Qubits
    k..k+b_in-1, Ausgang s auf Qubit k, b auf k+1..k+b_out (die noch in |0> sind).
    art = "eigen": reelle Konstruktion (nur fuer Bonddimensionen <= 2);
    art = "qiskit": qiskit.Isometry fuer jeden Schritt (allgemein, komplex)."""
    qc = QuantumCircuit(n)
    for k, Bk in enumerate(B):
        chi_l, _, chi_r = Bk.shape
        b_in, b_out = int(round(math.log2(chi_l))), int(round(math.log2(chi_r)))
        assert 2 ** b_in == chi_l and 2 ** b_out == chi_r
        if art == "eigen" and b_in <= 1 and b_out <= 1:
            if b_in == 0:          # Zustandspraeparation auf 1 + b_out Qubits
                moettoenen(Bk.reshape(-1), 1 + b_out, start=k, qc=qc, gesamt_n=n)
            elif b_out == 0:       # 2x2 orthogonal: Drehung oder Drehung·Z
                M = Bk[:, :, 0].T  # M[s, a]
                th = 2 * math.atan2(M[1, 0], M[0, 0])
                if np.linalg.det(M) < 0:
                    qc.z(qk(k, n))
                qc.ry(th, qk(k, n))
            else:                  # Isometrie 2 -> 4 auf (q1 = Modell k, q2 = Modell k+1)
                v0, v1 = Bk[0].reshape(-1), Bk[1].reshape(-1)
                al, be, ga, _ = _iso_2zu4(v0, v1)
                q1, q2 = qk(k, n), qk(k + 1, n)
                # W = G^T = U1^T U2^T U3^T  ->  Schaltung: U3^T zuerst
                qc.append(UCRYGate([-g for g in ga]), [q2, q1])
                qc.append(UCRYGate([-b for b in be]), [q1, q2])
                qc.append(UCRYGate([-a for a in al]), [q2, q1])
            continue
        m = b_out + 1
        # Gatter-Qubits: zuerst die Eingangsqubits (LSB = Modell k+b_in-1), dann der Rest
        modell = [k + b_in - 1 - i for i in range(b_in)] + [k + b_out - i for i in range(m - b_in)]
        M = np.zeros((2 ** m, chi_l))
        for r in range(2 ** m):
            bit = {modell[i]: (r >> i) & 1 for i in range(m)}
            s = bit[k]
            b = sum(bit[k + j] << (b_out - j) for j in range(1, b_out + 1))
            M[r, :] = Bk[:, s, b]
        qc.append(Isometry(M, 0, 0), [qk(q, n) for q in modell])
    return qc


def statepreparation_qiskit(psi, n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(np.asarray(psi, float).tolist()), qc.qubits)
    return qc


def _magisch(tq) -> int:
    return sum(1 for i in tq.data if i.operation.name in ("ry", "rz")
               and not ist_clifford(float(i.operation.params[0])))


def uebersetze(qc: QuantumCircuit) -> QuantumCircuit:
    """In {cx, ry, rz, Clifford} uebersetzen. Qiskit-Stufe 3 fasst 2-Qubit-Bloecke per
    KAK neu zusammen (weniger CNOT, aber MEHR magische Rotationen); fuer die T-Zahl
    zaehlt die Rotationszahl -> Stufe 1 und 3 rechnen, die mit weniger magischen
    Rotationen nehmen (bei Gleichstand weniger CNOT; Stufe 1 bei voelligem Gleichstand)."""
    kand = [transpile(qc, basis_gates=BASIS, optimization_level=o) for o in (1, 3)]
    return min(kand, key=lambda t: (_magisch(t), t.count_ops().get("cx", 0)))


# ------------------------------------------------------------------ 3. Kosten
_T_CACHE = {}


def ist_clifford(winkel: float, gitter=math.pi / 2, tol=1e-9) -> bool:
    r = winkel / gitter
    return abs(r - round(r)) < tol


def t_je_rotation(winkel: float, eps: float) -> int:
    """T-Zahl der Ross-Selinger-Synthese von Rz(winkel) (Ry = S H Rz H S^dagger, gleiche T)."""
    s = (round(winkel % (4 * math.pi), 11), eps)
    if s not in _T_CACHE:
        o = gridsynth_rz(float(winkel), float(eps)).count_ops()
        _T_CACHE[s] = int(sum(o.get(g, 0) for g in T_NAMEN))
    return _T_CACHE[s]


def kosten(tq: QuantumCircuit, eps=EPS, eps_gesamt=EPS_GESAMT) -> dict:
    """tq muss in BASIS vorliegen. Nicht-Clifford-Rotationen -> gridsynth je Rotation."""
    rot = [float(i.operation.params[0]) for i in tq.data if i.operation.name in ("ry", "rz")]
    magisch = [a for a in rot if not ist_clifford(a)]
    ops = tq.count_ops()
    ok = set(ops) <= set(BASIS)
    n_mag = len(magisch)
    return {"rotationen_magisch": n_mag,
            "T": int(sum(t_je_rotation(a, eps) for a in magisch)),
            "T_budget_gesamt": int(sum(t_je_rotation(a, eps_gesamt / max(n_mag, 1)) for a in magisch)),
            "cnot": int(ops.get("cx", 0)), "tiefe": int(tq.depth()), "nur_basis": bool(ok)}


def zustand(qc: QuantumCircuit) -> np.ndarray:
    """Statevector im Modell-Index (== Qiskit-Index wegen Umnummerierung)."""
    return Statevector(qc).data


def reell(v: np.ndarray) -> np.ndarray:
    """Globale Phase entfernen (Zustand ist bis auf e^{i phi} reell) -> reeller Vektor."""
    j = int(np.argmax(np.abs(v)))
    w = v * np.exp(-1j * np.angle(v[j]))
    assert np.abs(w.imag).max() < 1e-8, "Zustand nicht reell"
    return w.real


def runde(tq: QuantumCircuit, gitter: float) -> QuantumCircuit:
    """Alle ry/rz-Winkel auf das Gitter runden (Kodierung (c))."""
    aus = QuantumCircuit(tq.num_qubits, global_phase=tq.global_phase)
    for ins in tq.data:
        op = ins.operation
        if op.name in ("ry", "rz"):
            a = round(float(op.params[0]) / gitter) * gitter
            getattr(aus, op.name)(a, ins.qubits[0]) if abs(a) > 1e-12 else None
        else:
            aus.append(op, ins.qubits)
    return aus


def produkt_gerundet(psi, n, gitter) -> np.ndarray:
    """MPS chi = 1 = Produktzustand (Ry je Qubit); Winkel auf das Gitter gerundet (numpy)."""
    B, _ = mps_zerlege(psi, n, 1)
    v = np.ones(1)
    for Bk in B:
        th = 2 * math.atan2(Bk[0, 1, 0], Bk[0, 0, 0])
        th = round(th / gitter) * gitter
        v = np.kron(v, [math.cos(th / 2), math.sin(th / 2)])
    return v


# ------------------------------------------------------------------ Vortest
def vortest(verbose=True) -> bool:
    """(a) Moettoenen reproduziert x bis 1e-6 (auch mit Vorzeichen), (b) qiskit
    StatePreparation reproduziert x bis 1e-6, (c) MPS chi = max ist exakt (Vektor und
    Schaltung, eigene und qiskit-Isometrie), (d) MPS chi <= 2 / chi = 4: Schaltung ==
    MPS-Vektor bis 1e-6, (e) Produktzustand hat Treue 1 bei chi = 1 (und die
    Bonddimension 1 genuegt), (f) Rundung: k·pi/2-Gitter -> T = 0; k·pi/4 -> T =
    Zahl der ungeraden Vielfachen; (g) exakte Moettoenen-Schaltung hat nur Ry/CX."""
    rng = np.random.default_rng(3)
    _, _, xte, _ = bilder("mnist", 1, n_train=10, n_test=20, seite=16)
    tests, ok = [], True

    def pruefe(name, wert, bedingung):
        nonlocal ok
        tests.append((name, wert, bool(bedingung)))
        ok &= bool(bedingung)
        if verbose:
            print(f"  {name}: {wert:.2e}  {'ok' if bedingung else 'FEHLER'}")

    for n, x in ((8, xte[0].numpy()), (6, rng.normal(size=64)), (8, rng.normal(size=256))):
        psi = normiere(x)
        tq = uebersetze(moettoenen(psi, n))
        pruefe(f"(a) Moettoenen n={n}", 1 - treue(zustand(tq), psi),
               np.abs(zustand(tq) - psi).max() < 1e-6 and set(tq.count_ops()) <= {"ry", "cx"})
        sp = zustand(statepreparation_qiskit(psi, n))
        pruefe(f"(b) StatePreparation n={n}", float(np.abs(np.abs(sp) - np.abs(psi)).max()),
               1 - treue(sp, psi) < 1e-6)
        chimax = 2 ** (n // 2)
        B, v = mps_zerlege(psi, n, chimax)
        pruefe(f"(c) MPS chi=max={chimax} Vektor n={n}", float(np.abs(v - psi).max()),
               np.abs(v - psi).max() < 1e-6)
        vq = zustand(mps_schaltung(B, n, "qiskit"))
        pruefe(f"(c) MPS chi=max qiskit-Isometrie n={n}", 1 - treue(vq, psi), 1 - treue(vq, psi) < 1e-6)
        for chi in CHIS:
            B, v = mps_zerlege(psi, n, chi)
            for art in (("eigen", "qiskit") if chi <= 2 else ("qiskit",)):
                vq = zustand(mps_schaltung(B, n, art))
                pruefe(f"(d) MPS chi={chi} {art} Schaltung==Vektor n={n}", 1 - treue(vq, v),
                       1 - treue(vq, v) < 1e-6)
    # (e) Produktzustand
    th = rng.uniform(-math.pi, math.pi, 8)
    prod = np.ones(1)
    for t in th:
        prod = np.kron(prod, [math.cos(t / 2), math.sin(t / 2)])
    B, v = mps_zerlege(prod, 8, 1)
    pruefe("(e) Produktzustand chi=1 Treue-Defekt", 1 - treue(v, prod), 1 - treue(v, prod) < 1e-10)
    B4, _ = mps_zerlege(prod, 8, 4)
    _, S, _ = np.linalg.svd(prod.reshape(16, 16))
    pruefe("(e) Produktzustand: 2. Singulaerwert (Schnitt Mitte)", float(S[1]), S[1] < 1e-10)
    # (f) Rundung
    psi = normiere(xte[1].numpy())
    tq = uebersetze(moettoenen(psi, 8))
    k2 = kosten(runde(tq, math.pi / 2))
    r4 = runde(tq, math.pi / 4)
    k4 = kosten(r4)
    ungerade = sum(1 for i in r4.data if i.operation.name == "ry"
                   and not ist_clifford(float(i.operation.params[0])))
    pruefe("(f) Rundung pi/2 -> T", k2["T"], k2["T"] == 0)
    pruefe("(f) Rundung pi/4 -> T - #ungerade", k4["T"] - ungerade, k4["T"] == ungerade)
    vortest.tests = tests
    return ok


# ------------------------------------------------------------------ 3. Lauf
def _t_lernschaltung():
    """T-Kosten der vollen gelernten Schaltung (k = 32) aus C-hardware (lineare Anpassung)."""
    import json
    d = sorted(ERGEBNISSE.glob("C-hardware__*.json"))
    if not d:
        return T_LERN_K32_FALLBACK, "Fallback"
    h = json.loads(d[-1].read_text())
    lin = h["datensaetze"]["mnist"]["linear"]["grid_1e-3"]
    return (float(lin["steigung_T_je_winkel"] * 32 + lin["achsenabschnitt"]), d[-1].name)


def _kostenmessung(name, psis, n):
    """Kosten und Treue aller Varianten fuer eine Liste normierter Bilder."""
    varianten = {}

    def add(v, k, f):
        varianten.setdefault(v, {"kosten": [], "treue": []})
        varianten[v]["kosten"].append(k)
        varianten[v]["treue"].append(f)

    for i, psi in enumerate(psis):
        tq = uebersetze(moettoenen(psi, n))
        add("exakt (Moettoenen, reell)", kosten(tq), treue(zustand(tq), psi))
        for g, gname in ((math.pi / 2, "pi/2"), (math.pi / 4, "pi/4")):
            r = runde(tq, g)
            add(f"exakt, Winkel gerundet {gname}", kosten(r), treue(zustand(r), psi))
        sp = uebersetze(statepreparation_qiskit(psi, n))
        add("qiskit StatePreparation (komplex)", kosten(sp), treue(zustand(sp), psi))
        for chi in CHIS:
            B, v = mps_zerlege(psi, n, chi)
            f = treue(v, psi)
            arten = ("eigen", "qiskit") if chi <= 2 else ("qiskit",)
            for art in arten:
                t = uebersetze(mps_schaltung(B, n, art))
                k = kosten(t)
                k["parameter_untergrenze"] = mps_parameter(B)
                add(f"MPS chi={chi} ({art})", k, treue(zustand(t), psi))
            if chi == 1:
                for g, gname in ((math.pi / 2, "pi/2"), (math.pi / 4, "pi/4")):
                    t = uebersetze(mps_schaltung(B, n, "eigen"))
                    r = runde(t, g)
                    add(f"MPS chi=1, Winkel gerundet {gname}", kosten(r), treue(zustand(r), psi))
            if chi == 2:
                t = uebersetze(mps_schaltung(B, n, "eigen"))
                r = runde(t, math.pi / 4)
                add("MPS chi=2, Winkel gerundet pi/4", kosten(r), treue(zustand(r), psi))
        if i % 5 == 4:
            print(f"  {name}: {i + 1}/{len(psis)} Bilder", flush=True)
    tabelle = {}
    for v, d in varianten.items():
        ks = d["kosten"]
        tabelle[v] = {sch: float(np.mean([k[sch] for k in ks])) for sch in ks[0] if sch != "nur_basis"}
        tabelle[v]["T_min"] = int(min(k["T"] for k in ks))
        tabelle[v]["T_max"] = int(max(k["T"] for k in ks))
        tabelle[v]["treue_mittel"] = float(np.mean(d["treue"]))
        tabelle[v]["treue_min"] = float(np.min(d["treue"]))
        tabelle[v]["nur_basis"] = all(k["nur_basis"] for k in ks)
    return tabelle


# ---- Zustaende fuer die Klassifikation (viele Bilder -> moeglichst ohne qiskit)
def _zustaende(psis, variante, n):
    """Genaeherte Zustaende (reell) fuer alle Bilder einer Variante."""
    aus = np.empty_like(psis)
    for i, psi in enumerate(psis):
        if variante == "exakt":
            aus[i] = psi
        elif variante.startswith("mps"):
            aus[i] = mps_zerlege(psi, n, int(variante[3:]))[1]
        elif variante == "produkt_pi2":
            aus[i] = produkt_gerundet(psi, n, math.pi / 2)
        elif variante == "produkt_pi4":
            aus[i] = produkt_gerundet(psi, n, math.pi / 4)
        else:
            raise ValueError(variante)
    return aus


def _zustaende_schaltung(psis, n):
    """Gerundete Schaltungen (brauchen qiskit): exakt pi/2, exakt pi/4, MPS chi=2 pi/4."""
    aus = {"exakt_pi2": np.empty_like(psis), "exakt_pi4": np.empty_like(psis),
           "mps2_pi4": np.empty_like(psis)}
    for i, psi in enumerate(psis):
        # eigene Schaltungen: Stufe 1 (nur ry/cx) ist genau die Schaltung, die auch
        # uebersetze() fuer die Kostentabelle waehlt -> hier direkt, spart die Haelfte
        tq = transpile(moettoenen(psi, n), basis_gates=BASIS, optimization_level=1)
        aus["exakt_pi2"][i] = reell(zustand(runde(tq, math.pi / 2)))
        aus["exakt_pi4"][i] = reell(zustand(runde(tq, math.pi / 4)))
        B, _ = mps_zerlege(psi, n, 2)
        aus["mps2_pi4"][i] = reell(zustand(runde(transpile(mps_schaltung(B, n, "eigen"), basis_gates=BASIS,
                                                                  optimization_level=1), math.pi / 4)))
        if i % 1000 == 999:
            print(f"    gerundete Schaltungen: {i + 1}/{len(psis)}", flush=True)
    return aus


def _merkmale(U, zust):
    """Born-Wahrscheinlichkeiten |U psi|^2 (U reell, float64)."""
    return torch.tensor((zust @ U.T) ** 2, dtype=torch.float32)


def _klassifikation(schnell=False):
    n = 8
    xtr, ytr, xte, yte = daten("mnist", SEED)
    if schnell:
        xtr, ytr, xte, yte = xtr[:600], ytr[:600], xte[:300], yte[:300]
    t0 = time.time()
    torch.manual_seed(SEED)
    m = QireZelle(n=n)
    trainiere(m, xtr, ytr, 150 if schnell else SCHRITTE, LR, seed=SEED)
    acc_modell = genauigkeit(m(xte), yte)
    # Abgleich mit den Rohdaten C-mnist (winkel_voll dieses Seeds)
    import json
    roh = json.loads(sorted(ERGEBNISSE.glob("C-mnist__*.json"))[-1].read_text())
    z = next(z for z in roh["zeilen"] if z["seed"] == SEED)
    abw = float(np.abs(np.array(z["winkel_voll"]) - m.theta.detach().numpy().ravel()).max())
    alt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        U = m.unitaer().numpy()
    torch.set_default_dtype(alt)
    print(f"  Zelle Seed {SEED}: Genauigkeit {acc_modell:.4f} (Rohdaten {z['voll']:.4f}), "
          f"Winkelabweichung {abw:.1e}, {time.time() - t0:.0f} s", flush=True)

    ptr, pte = normiere(xtr.numpy()), normiere(xte.numpy())
    zust_te = {v: _zustaende(pte, v, n) for v in
               ("exakt", "mps1", "mps2", "mps4", "produkt_pi2", "produkt_pi4")}
    zust_tr = {v: _zustaende(ptr, v, n) for v in zust_te}
    t1 = time.time()
    zust_te.update(_zustaende_schaltung(pte, n))
    zust_tr.update(_zustaende_schaltung(ptr, n))
    print(f"  gerundete Schaltungen: {time.time() - t1:.0f} s", flush=True)

    def kopf_neu(merk_tr):
        torch.manual_seed(SEED)
        k = torch.nn.Linear(2 ** n, 10)
        trainiere(k, merk_tr, ytr, 150 if schnell else SCHRITTE, LR, seed=SEED)
        return k

    erg = {"seed": SEED, "genauigkeit_modell": acc_modell, "genauigkeit_rohdaten": z["voll"],
           "winkel_abweichung_zu_rohdaten": abw, "varianten": {}}
    for v in zust_te:
        f = _merkmale(U, zust_te[v])
        k = kopf_neu(_merkmale(U, zust_tr[v]))
        with torch.no_grad():
            direkt = genauigkeit(m.kopf(f), yte)
            nach = genauigkeit(k(f), yte)
        tr = float(np.mean(np.sum(zust_te[v] * pte, axis=1) ** 2))
        erg["varianten"][v] = {"genauigkeit_direkt": direkt, "genauigkeit_kopf_nachtrainiert": nach,
                               "treue_test_mittel": tr}
        print(f"  {v:12s} Treue {tr:.3f}  direkt {direkt:.4f}  Kopf neu {nach:.4f}", flush=True)
    return erg


# Zuordnung Kostentabelle <-> Klassifikationsvariante
ZUORDNUNG = {
    "exakt (Moettoenen, reell)": "exakt",
    "exakt, Winkel gerundet pi/2": "exakt_pi2",
    "exakt, Winkel gerundet pi/4": "exakt_pi4",
    "MPS chi=1 (eigen)": "mps1",
    "MPS chi=1, Winkel gerundet pi/2": "produkt_pi2",
    "MPS chi=1, Winkel gerundet pi/4": "produkt_pi4",
    "MPS chi=2 (eigen)": "mps2",
    "MPS chi=2, Winkel gerundet pi/4": "mps2_pi4",
    "MPS chi=4 (qiskit)": "mps4",
}


def lauf(schnell=False) -> dict:
    t0 = time.time()
    ok = vortest(verbose=not schnell)
    t_lern, t_lern_quelle = _t_lernschaltung()
    ergebnis = {"abteilung": "clifford/hardware", "agenda": "C-kodierung", "quantenstufe": "Q3 (lokal simuliert)",
                "hypothesen": HYPOTHESEN, "vortest_bestanden": bool(ok),
                "vortest": [(a, float(b), c) for a, b, c in vortest.tests],
                "eps_je_rotation": EPS, "eps_gesamt_zusatz": EPS_GESAMT,
                "t_lernschaltung_k32": t_lern, "t_lernschaltung_quelle": t_lern_quelle,
                "bitordnung": "Modell-Qubit q <-> Qiskit-Qubit n-1-q; Basisindex identisch",
                "kosten": {}}
    anzahl = 3 if schnell else 20
    _, _, xte, _ = bilder("mnist", SEED, n_train=10, n_test=5_000, seite=16)
    ergebnis["kosten"]["mnist_16x16_n8"] = _kostenmessung("mnist", normiere(xte[:anzahl].numpy()), 8)
    _, _, zte, _ = ziffern(SEED)
    ergebnis["kosten"]["ziffern_8x8_n6"] = _kostenmessung("ziffern", normiere(zte[:anzahl].numpy()), 6)
    ergebnis["sekunden_kosten"] = time.time() - t0
    ergebnis["klassifikation_mnist"] = _klassifikation(schnell)

    # Gesamttabelle MNIST
    km, kl = ergebnis["kosten"]["mnist_16x16_n8"], ergebnis["klassifikation_mnist"]["varianten"]
    t_exakt = km["exakt (Moettoenen, reell)"]["T"]
    acc_d, acc_n = kl["exakt"]["genauigkeit_direkt"], kl["exakt"]["genauigkeit_kopf_nachtrainiert"]
    tabelle = []
    for name, v in ZUORDNUNG.items():
        k = km[name]
        tabelle.append({"kodierung": name, "T": k["T"], "T_anteil_von_exakt": k["T"] / t_exakt,
                        "T_budget_gesamt": k["T_budget_gesamt"], "cnot": k["cnot"], "tiefe": k["tiefe"],
                        "rotationen_magisch": k["rotationen_magisch"],
                        # optimistisch: Parameterzahl der reellen MPS x mittlere T je Rotation
                        "T_untergrenze_parameter": (k["parameter_untergrenze"] * k["T"]
                                                    / max(k["rotationen_magisch"], 1e-9)
                                                    if "parameter_untergrenze" in k else None),
                        "treue_20": k["treue_mittel"], "treue_test": kl[v]["treue_test_mittel"],
                        "genauigkeit_direkt": kl[v]["genauigkeit_direkt"],
                        "verlust_pp_direkt": 100 * (acc_d - kl[v]["genauigkeit_direkt"]),
                        "genauigkeit_kopf_neu": kl[v]["genauigkeit_kopf_nachtrainiert"],
                        "verlust_pp_kopf_neu": 100 * (acc_n - kl[v]["genauigkeit_kopf_nachtrainiert"])})
    ergebnis["tabelle_mnist"] = tabelle

    k1 = t_exakt > t_lern
    kand_d = [z["kodierung"] for z in tabelle if z["T_anteil_von_exakt"] < 0.10 and z["verlust_pp_direkt"] < 2]
    kand_n = [z["kodierung"] for z in tabelle if z["T_anteil_von_exakt"] < 0.10 and z["verlust_pp_kopf_neu"] < 2]
    ergebnis["urteil"] = {
        "K-1": (f"H1: exakte Kodierung {t_exakt:.0f} T > Lernschaltung k=32 {t_lern:.0f} T "
                f"(Faktor {t_exakt / t_lern:.1f})" if k1 else
                f"H0: exakte Kodierung {t_exakt:.0f} T <= Lernschaltung {t_lern:.0f} T"),
        "K-2": ("beschreibend, 1 Seed: primaer (direkt eingespeist) "
                + (f"JA: {kand_d}" if kand_d else "NEIN, keine Variante")
                + "; sekundaer (Kopf nachtrainiert) "
                + (f"JA: {kand_n}" if kand_n else "NEIN, keine Variante")),
    }
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


def drucke(e):
    print(f"\nMNIST 16x16 (n=8), eps={EPS} je Rotation; Lernschaltung k=32: {e['t_lernschaltung_k32']:.0f} T")
    print(" Kodierung | T | Anteil | T(eps_ges) | CNOT | Tiefe | Treue | Gen. direkt | Verlust pp | Gen. Kopf neu | Verlust pp")
    for z in e["tabelle_mnist"]:
        print(f" {z['kodierung']:34s} | {z['T']:7.0f} | {z['T_anteil_von_exakt']:5.3f} | {z['T_budget_gesamt']:7.0f} | "
              f"{z['cnot']:5.0f} | {z['tiefe']:5.0f} | {z['treue_test']:.3f} | {z['genauigkeit_direkt']:.4f} | "
              f"{z['verlust_pp_direkt']:6.1f} | {z['genauigkeit_kopf_neu']:.4f} | {z['verlust_pp_kopf_neu']:6.1f}")
    for ds, t in e["kosten"].items():
        print(f"\nKosten {ds} (Mittel ueber Bilder):")
        for v, k in t.items():
            print(f"  {v:36s} T {k['T']:7.1f} [{k['T_min']}-{k['T_max']}]  magische Rot. {k['rotationen_magisch']:6.1f}"
                  f"  CNOT {k['cnot']:6.1f}  Tiefe {k['tiefe']:6.1f}  Treue {k['treue_mittel']:.4f}"
                  + (f"  Param.-Untergrenze {k['parameter_untergrenze']:.0f}" if "parameter_untergrenze" in k else ""))
    print("\n", e["urteil"])


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    if "--vortest" in sys.argv:
        print("vortest:", vortest())
        sys.exit()
    e = lauf(schnell=schnell)
    drucke(e)
    print(speichere("C-kodierung" + ("-schnell" if schnell else ""), e))
