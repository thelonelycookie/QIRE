"""Abteilung Clifford, Q3-Bruecke (Agenda C-hardware) -- Magie-Budget in T-Gatter.

HEILMEIER:
- Was wollen wir?  Die Zahl „k freie Nicht-Clifford-Winkel“ aus C-clifford /
  C-mnist in die echte Kostengroesse eines fehlertoleranten Quantencomputers
  uebersetzen: die Zahl der T-Gatter im Gatterset Clifford+T.
- Wie heute?  Die Magie-Kurve war bisher nur „Winkel, die nicht auf k·pi/2
  liegen“. Das ist eine Zaehlung im Simulator, keine Hardwarekosten.
- Was ist neu?  Dieselbe Schaltung (Amplitudenkodierung -> L x [Ry je Qubit,
  CNOT-Ring] -> Born) wird als qiskit.QuantumCircuit gebaut, gegen die
  Matrix des Modells verifiziert und in Clifford+T uebersetzt:
    * gerundete Winkel k·pi/2  -> EXAKT durch h, x, z (keine T-Gatter),
    * magische Winkel          -> Naeherung durch Solovay-Kitaev (Rekursion
      1-3) bzw. Ross-Selinger „gridsynth“ (optimal fuer Rz, Genauigkeit eps).
- Woran merken wir, ob es klappt?  k = 0 muss T-Zahl 0 und exakten Operator
  liefern (Positivkontrolle), und die Hardware-Schaltung muss fuer echte
  Testbilder dieselben Born-Wahrscheinlichkeiten liefern wie das Modell.

BITORDNUNG (wichtig, Fehlerquelle Nr. 1 bei Qiskit):
  Modell (QireZelle.unitaer): torch.kron(K_0, K_1, ...), also ist
  Modell-Qubit 0 das HOECHSTWERTIGE Bit des Basisindex.
  Qiskit: Qubit j ist Bit j des Basisindex (little endian), Qubit 0 = NIEDRIGSTES Bit.
  Umnummerierung:  Modell-Qubit q  <->  Qiskit-Qubit n-1-q.
  Mit dieser Zuordnung ist der Basisindex in beiden Welten DERSELBE, d.h.
  Operator(schaltung).data == QireZelle.unitaer() ohne weitere Permutation,
  und ein Amplitudenvektor x kann unveraendert als Statevector(x) eingesetzt
  werden. (vortest() prueft genau das.)

GRENZE (ehrlich): In den Rohdaten ist nur „winkel_voll“ (das voll trainierte
Modell) gespeichert, nicht die nach dem Runden nachtrainierten Magiewinkel.
Die T-Zahl haengt aber praktisch nur von der ZAHL der magischen Winkel und
von eps ab, nicht vom genauen Wert (gridsynth: ~3·log2(1/eps) T je Winkel),
daher ist die Uebersetzung k -> T-Zahl davon nicht beruehrt.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, StatePreparation
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis import SolovayKitaevDecomposition, gridsynth_rz

from abteilungen.clifford import (GITTER, LAGEN, MAGIE_STUFEN, QireZelle, _magie_reihenfolge,
                                  auswertung, runde_clifford)
from labor.gemeinsam import ERGEBNISSE, bilder, speichere

HYPOTHESEN = {
    "H-1": "Pruefung (Positivkontrolle): k = 0 (alle Winkel auf k·pi/2) ergibt "
           "T-Zahl exakt 0 und einen EXAKTEN Operator (Fehler < 1e-5) -- "
           "Clifford braucht keine Magie. Fuer jeden Seed, jede Synthese.",
    "H-2": "Beschreibend: die T-Zahl waechst linear in k (T ≈ t_eps · k); "
           "berichtet werden Steigung t_eps je Genauigkeit und R².",
}

T_NAMEN = ("t", "tdg")
CLIFFORD_T_BASIS = {"h", "s", "sdg", "x", "y", "z", "cx", "t", "tdg"}
TOLERANZ_EXAKT = 1e-5

# Synthese-Verfahren fuer magische Ry-Winkel: (Name, Art, Parameter)
SYNTHESEN = [
    ("sk_r1", "sk", 1), ("sk_r2", "sk", 2), ("sk_r3", "sk", 3),
    ("grid_1e-2", "grid", 1e-2), ("grid_1e-3", "grid", 1e-3), ("grid_1e-4", "grid", 1e-4),
]
DATENSAETZE = {"mnist": "C-mnist", "ziffern": "C-clifford-v2"}


# ------------------------------------------------------------------ 1. Schaltung
def qk(q: int, n: int) -> int:
    """Modell-Qubit q -> Qiskit-Qubit (little endian)."""
    return n - 1 - q


def _clifford_ry(qc: QuantumCircuit, winkel: float, ziel: int):
    """Ry(m·pi/2) EXAKT aus h, x, z (inkl. globaler Phase):
       Ry(pi/2) = X·H,  Ry(pi) = X·Z,  Ry(3pi/2) = -H·X,  Ry(theta+2pi) = -Ry(theta).
    (Operator-Schreibweise; in der Schaltung steht das rechte Gatter zuerst.)"""
    m = int(round(winkel / GITTER))
    r, j = m % 4, (m - m % 4) // 4
    if r == 1:
        qc.h(ziel); qc.x(ziel)
    elif r == 2:
        qc.z(ziel); qc.x(ziel)
    elif r == 3:
        qc.x(ziel); qc.h(ziel)
        j += 1                     # H·X = Ry(-pi/2) = -Ry(3pi/2)
    if j % 2:
        qc.global_phase += math.pi


def baue_schaltung(winkel, frei=None, form="ry") -> QuantumCircuit:
    """QireZelle als Qiskit-Schaltung (ohne Kodierung/Auslese: nur die Unitaere).

    winkel: (L, n) -- die Winkel des Modells (= QireZelle.winkel()).
    frei:   (L, n) 0/1 -- 1 = magischer Winkel, 0 = liegt auf dem Gitter k·pi/2
            (wird hier auf das Gitter gerundet). None = alle frei.
    form:   "ry"       -> jeder Winkel als ry-Gatter (exakte Referenz),
            "clifford" -> gerundete Winkel als exakte h/x/z-Folge, magische als ry
                          (die magischen ry werden in zu_clifford_t ersetzt).
    Gatterfolge je Lage: Ry(theta[l,q]) auf allen Qubits, dann CNOT(q -> q+1 mod n)
    nacheinander fuer q = 0..n-1 (genau wie _cnot_ring_perm)."""
    w = torch.as_tensor(winkel, dtype=torch.float64)
    L, n = w.shape
    f = torch.ones(L, n) if frei is None else torch.as_tensor(frei).float()
    w = torch.where(f.bool(), w, runde_clifford(w))
    qc = QuantumCircuit(n)
    for l in range(L):
        for q in range(n):
            th = float(w[l, q])
            if form == "clifford" and not f[l, q]:
                _clifford_ry(qc, th, qk(q, n))
            else:
                qc.ry(th, qk(q, n))
        for q in range(n):
            qc.cx(qk(q, n), qk((q + 1) % n, n))
    return qc


def modell_aus_winkeln(winkel, frei=None) -> QireZelle:
    """QireZelle (float64) mit genau diesen Winkeln -- wie clifford._gerundet."""
    w = torch.as_tensor(winkel, dtype=torch.float64)
    L, n = w.shape
    f = torch.ones(L, n) if frei is None else torch.as_tensor(frei).float()
    m = QireZelle(n=n, lagen=L).double()
    m.fixiere(runde_clifford(w), f.double())
    with torch.no_grad():
        m.theta.copy_(torch.where(f.bool(), w, runde_clifford(w)))
    return m


@contextmanager
def _float64():
    """QireZelle.unitaer() baut torch.eye im Standard-dtype -> fuer float64 umstellen."""
    alt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.no_grad():
            yield
    finally:
        torch.set_default_dtype(alt)


def modell_unitaer(m: QireZelle) -> np.ndarray:
    """QireZelle.unitaer() in float64."""
    with _float64():
        return m.unitaer().numpy()


def operatorfehler(U: np.ndarray, V: np.ndarray) -> float:
    """Spektralnorm ||U - e^{i phi} V|| mit phi = arg tr(V^dagger U) (globale Phase
    ist physikalisch bedeutungslos und wird herausgerechnet)."""
    phi = np.angle(np.trace(V.conj().T @ U))
    return float(np.linalg.norm(U - np.exp(1j * phi) * V, 2))


# ------------------------------------------------------------------ 2. Clifford+T
_SK = None
_CACHE = {}


def _sk():
    global _SK
    if _SK is None:
        _SK = SolovayKitaevDecomposition()     # Basis h, t, tdg; Tiefe 12 (Qiskit-Standard)
    return _SK


def synthetisiere_ry(theta: float, art: str, param) -> QuantumCircuit:
    """Ein-Qubit-Schaltung in Clifford+T, die Ry(theta) naehert."""
    schluessel = (round(theta, 12), art, param)
    if schluessel not in _CACHE:
        if art == "sk":
            c = _sk().run(RYGate(theta), int(param))
        elif art == "grid":
            # Ry(theta) = S · H · Rz(theta) · H · S^dagger   (S X S^dagger = Y)
            c = QuantumCircuit(1)
            c.sdg(0); c.h(0)
            c.compose(gridsynth_rz(theta, float(param)), [0], inplace=True)
            c.h(0); c.s(0)
        else:
            raise ValueError(art)
        _CACHE[schluessel] = c
    return _CACHE[schluessel]


def zu_clifford_t(qc: QuantumCircuit, art: str, param) -> QuantumCircuit:
    """Ersetzt jedes verbliebene ry (= magischer Winkel) durch seine Clifford+T-Naeherung."""
    aus = QuantumCircuit(qc.num_qubits, global_phase=qc.global_phase)
    for instr in qc.data:
        op, qubits = instr.operation, instr.qubits
        if op.name == "ry":
            aus.compose(synthetisiere_ry(float(op.params[0]), art, param), qubits, inplace=True)
        else:
            aus.append(op, qubits)
    return aus


def kosten(qc: QuantumCircuit) -> dict:
    ops = qc.count_ops()
    return {"T": int(sum(ops.get(g, 0) for g in T_NAMEN)),
            "cnot": int(ops.get("cx", 0)),
            "gatter": int(sum(ops.values())),
            "tiefe": int(qc.depth()),
            "t_tiefe": int(qc.depth(filter_function=lambda i: i.operation.name in T_NAMEN)),
            "nur_clifford_t": set(ops) <= CLIFFORD_T_BASIS}


def maske_frei(winkel, k: int) -> torch.Tensor:
    """k freie Winkel wie clifford._magie_reihenfolge: groesster Abstand zum Gitter zuerst."""
    w = torch.as_tensor(winkel, dtype=torch.float64)
    f = torch.zeros(w.numel())
    f[_magie_reihenfolge(w)[:k]] = 1
    return f.reshape(w.shape)


# ------------------------------------------------------------------ Vortest
def vortest(ziehungen=3, verbose=True) -> bool:
    """Verifikation aus Teil 1 (+ Kontrollen fuer Teil 2):
    (a) Operator(Schaltung, form ry)       == QireZelle.unitaer()   bis 1e-5
    (b) Operator(Schaltung, form clifford) == QireZelle.unitaer()   bis 1e-5 (k-Masken)
    (c) Negativkontrolle: OHNE Umnummerierung (Qiskit-Qubit q = Modell-Qubit q)
        weicht der Operator ab -> der Test kann die Bitordnung ueberhaupt erkennen.
    (d) Clifford+T bei k=0: T-Zahl 0, nur Clifford+T-Gatter, Operator exakt.
    (e) Clifford+T bei k>0: T-Zahl > 0 und Fehler ~ eps (klein, aber nicht 0)."""
    rng = np.random.default_rng(1)
    ok, maxfehler = True, 0.0
    for n in (6, 8):
        for z in range(ziehungen):
            w = torch.tensor(rng.uniform(-2 * math.pi, 4 * math.pi, (LAGEN, n)))
            # (a) alle Winkel frei
            U = modell_unitaer(modell_aus_winkeln(w))
            V = Operator(baue_schaltung(w)).data
            fa = float(np.abs(U - V).max())
            # (a') dasselbe mit dem float32-Modell, wie es trainiert wird
            m32 = QireZelle(n=n, lagen=LAGEN)
            m32.fixiere(w.float(), torch.ones(LAGEN, n))
            with torch.no_grad():
                fa = max(fa, float(np.abs(m32.unitaer().numpy() - V).max()))
            # (b) gemischte Masken inkl. Winkel ausserhalb [0, 2pi) -> globale Phase
            fb = 0.0
            for k in (0, 3, LAGEN * n):
                f = maske_frei(w, k)
                U = modell_unitaer(modell_aus_winkeln(w, f))
                fb = max(fb, float(np.abs(U - Operator(baue_schaltung(w, f, "clifford")).data).max()),
                         float(np.abs(U - Operator(baue_schaltung(w, f, "ry")).data).max()))
            # (c) Negativkontrolle: Qubits nicht umnummeriert
            richtig, falsch = baue_schaltung(w), QuantumCircuit(n)
            for instr in richtig.data:
                falsch.append(instr.operation, [falsch.qubits[n - 1 - richtig.find_bit(b).index]
                                                for b in instr.qubits])
            U = modell_unitaer(modell_aus_winkeln(w))
            fc = float(np.abs(U - Operator(falsch).data).max())
            # (d) k=0 in Clifford+T
            f0 = maske_frei(w, 0)
            qc0 = zu_clifford_t(baue_schaltung(w, f0, "clifford"), "grid", 1e-3)
            U0 = modell_unitaer(modell_aus_winkeln(w, f0))
            k0 = kosten(qc0)
            fd = float(np.abs(U0 - Operator(qc0).data).max())
            # (e) k=2 in Clifford+T
            f2 = maske_frei(w, 2)
            qc2 = zu_clifford_t(baue_schaltung(w, f2, "clifford"), "grid", 1e-3)
            U2 = modell_unitaer(modell_aus_winkeln(w, f2))
            k2 = kosten(qc2)
            fe = operatorfehler(U2, Operator(qc2).data)
            maxfehler = max(maxfehler, fa, fb, fd)
            gut = (fa < TOLERANZ_EXAKT and fb < TOLERANZ_EXAKT and fc > 1e-2 and fd < TOLERANZ_EXAKT
                   and k0["T"] == 0 and k0["nur_clifford_t"] and k2["T"] > 0 and k2["nur_clifford_t"]
                   and 0 < fe < 3 * 2 * 1e-3)
            if verbose:
                print(f"  n={n} Ziehung {z}: (a) {fa:.1e} (b) {fb:.1e} (c) {fc:.2f} "
                      f"(d) {fd:.1e} T={k0['T']} (e) {fe:.1e} T={k2['T']}  {'ok' if gut else 'FEHLER'}")
            ok &= gut
    vortest.maxfehler = maxfehler
    return ok


# ------------------------------------------------------------------ 3. Lauf
def _rohdaten(hyp_id: str):
    import json
    dateien = sorted(ERGEBNISSE.glob(f"{hyp_id}__*.json"))
    return json.loads(dateien[-1].read_text()), dateien[-1].name


def _lineare_anpassung(ks, ts):
    ks, ts = np.asarray(ks, float), np.asarray(ts, float)
    a, b = np.polyfit(ks, ts, 1)
    r2 = 1 - ((ts - (a * ks + b)) ** 2).sum() / max(((ts - ts.mean()) ** 2).sum(), 1e-12)
    return {"steigung_T_je_winkel": float(a), "achsenabschnitt": float(b), "R2": float(r2)}


def _datensatz_lauf(name, hyp_id, seeds_max=None, synthesen=SYNTHESEN, stufen=MAGIE_STUFEN):
    roh, datei = _rohdaten(hyp_id)
    n = roh["qubits"]
    zeilen = roh["zeilen"][:seeds_max] if seeds_max else roh["zeilen"]
    # Gewinnanteile mit der AKTUELLEN Auswertung (v2-Rohdaten der Ziffern haben
    # C3_gewinnanteil noch nicht gespeichert -> aus den Zeilen neu berechnet)
    aus = auswertung(roh["zeilen"])
    anteil = {int(k): v for k, v in aus["C3_gewinnanteil"].items()}
    je_seed = []
    for z in zeilen:
        w = torch.tensor(z["winkel_voll"], dtype=torch.float64).reshape(LAGEN, n)
        eintrag = {"seed": z["seed"], "k": {}}
        for k in stufen:
            f = maske_frei(w, k)
            ref = baue_schaltung(w, f, "clifford")
            U = modell_unitaer(modell_aus_winkeln(w, f))
            e_ref = float(np.abs(U - Operator(ref).data).max())
            eintrag["k"][k] = {"fehler_exakt_vs_modell": e_ref}
            for sname, art, p in synthesen:
                qc = zu_clifford_t(ref, art, p)
                c = kosten(qc)
                c["operatorfehler"] = operatorfehler(U, Operator(qc).data)
                eintrag["k"][k][sname] = c
        je_seed.append(eintrag)
        print(f"  {name} Seed {z['seed']}: " + "  ".join(
            f"k{k}:T={eintrag['k'][k]['grid_1e-3']['T']}" for k in stufen), flush=True)

    tabelle = []
    for k in stufen:
        zeile = {"k": k, "gewinnanteil": anteil.get(k),
                 "fehler_exakt_max": max(e["k"][k]["fehler_exakt_vs_modell"] for e in je_seed)}
        for sname, _, _ in synthesen:
            T = [e["k"][k][sname]["T"] for e in je_seed]
            zeile[sname] = {"T_mittel": float(np.mean(T)), "T_min": int(min(T)), "T_max": int(max(T)),
                            "tiefe_mittel": float(np.mean([e["k"][k][sname]["tiefe"] for e in je_seed])),
                            "t_tiefe_mittel": float(np.mean([e["k"][k][sname]["t_tiefe"] for e in je_seed])),
                            "cnot": int(je_seed[0]["k"][k][sname]["cnot"]),
                            "operatorfehler_max": float(max(e["k"][k][sname]["operatorfehler"]
                                                            for e in je_seed)),
                            "nur_clifford_t": all(e["k"][k][sname]["nur_clifford_t"] for e in je_seed)}
        tabelle.append(zeile)
    linear = {s: _lineare_anpassung([z["k"] for z in tabelle], [z[s]["T_mittel"] for z in tabelle])
              for s, _, _ in synthesen}
    return {"rohdatei": datei, "qubits": n, "seeds": [e["seed"] for e in je_seed],
            "M50": aus.get("C3_M50"), "M90": aus.get("C3_M90"),
            "tabelle": tabelle, "linear": linear, "je_seed": je_seed}


def _born_vergleich(seed_index=0, k=2, bilder_anzahl=20, synthesen=SYNTHESEN):
    """Teil 4: Born-Wahrscheinlichkeiten der Clifford+T-Schaltung vs. Modell (MNIST)."""
    roh, _ = _rohdaten(DATENSAETZE["mnist"])
    n, z = roh["qubits"], roh["zeilen"][seed_index]
    w = torch.tensor(z["winkel_voll"], dtype=torch.float64).reshape(LAGEN, n)
    f = maske_frei(w, k)
    m = modell_aus_winkeln(w, f)
    _, _, xte, _ = bilder("mnist", z["seed"], n_train=6_000, n_test=5_000, seite=16)
    x = xte[:bilder_anzahl].double()
    with _float64():
        p_modell = m.merkmale(x).numpy()
    psi = (x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)).numpy()
    ref = baue_schaltung(w, f, "clifford")
    erg = {"seed": z["seed"], "k": k, "bilder": bilder_anzahl}
    for sname, art, p in [("exakt", None, None)] + list(synthesen):
        qc = ref if art is None else zu_clifford_t(ref, art, p)
        diffs, tv = [], []
        for i in range(bilder_anzahl):
            # Qiskit-Index == Modell-Index (Umnummerierung q -> n-1-q, s.o.)
            p_hw = Statevector(psi[i]).evolve(qc).probabilities()
            diffs.append(float(np.abs(p_hw - p_modell[i]).max()))
            tv.append(float(0.5 * np.abs(p_hw - p_modell[i]).sum()))
        erg[sname] = {"max_abs_diff": max(diffs), "totalvariation_max": max(tv)}
    # Kontrolle des „initialize“-Wegs: Zustandsvorbereitung als Gatter vor der Schaltung
    voll = QuantumCircuit(n)
    voll.append(StatePreparation(psi[0].tolist()), voll.qubits)
    voll.compose(ref, inplace=True)
    erg["statepreparation_diff"] = float(np.abs(Statevector(voll).probabilities() - p_modell[0]).max())
    return erg


def lauf(schnell=False) -> dict:
    t0 = time.time()
    ok = vortest(ziehungen=1 if schnell else 3)
    ergebnis = {"abteilung": "clifford/hardware", "agenda": "C-hardware", "quantenstufe": "Q3 (lokal simuliert)",
                "hypothesen": HYPOTHESEN, "vortest_bestanden": bool(ok),
                "vortest_maxfehler": getattr(vortest, "maxfehler", None),
                "bitordnung": "Modell-Qubit q <-> Qiskit-Qubit n-1-q; Basisindex identisch",
                "synthesen": {s: {"art": a, "parameter": p} for s, a, p in SYNTHESEN},
                "grenze": "Magiewinkel = winkel_voll (nachtrainierte Werte nicht gespeichert); "
                          "SK-Basis h,t,tdg (S zaehlt dort als 2 T).",
                "datensaetze": {}}
    for name, hyp_id in DATENSAETZE.items():
        ergebnis["datensaetze"][name] = _datensatz_lauf(name, hyp_id, seeds_max=2 if schnell else None)
    ergebnis["born_vergleich"] = _born_vergleich(bilder_anzahl=5 if schnell else 20)

    # Urteile
    h1 = all(z[s]["T_max"] == 0 and z[s]["operatorfehler_max"] < TOLERANZ_EXAKT and z[s]["nur_clifford_t"]
             for d in ergebnis["datensaetze"].values() for z in d["tabelle"] if z["k"] == 0
             for s, _, _ in SYNTHESEN)
    lin = {name: {s: d["linear"][s] for s in ("sk_r2", "grid_1e-3")}
           for name, d in ergebnis["datensaetze"].items()}
    ergebnis["urteil"] = {
        "H-1": ("H1: k=0 -> T-Zahl 0 und exakter Operator (alle Seeds, alle Synthesen)" if h1
                else "H0: k=0 braucht T-Gatter oder ist nicht exakt -- Bruecke fehlerhaft"),
        "H-2": "beschreibend: " + "; ".join(
            f"{name} {s}: T ≈ {v['steigung_T_je_winkel']:.1f}·k + {v['achsenabschnitt']:.1f} (R² {v['R2']:.4f})"
            for name, d in lin.items() for s, v in d.items()),
    }
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


def _drucke_tabelle(e):
    for name, d in e["datensaetze"].items():
        print(f"\n{name} (n={d['qubits']}, Rohdaten {d['rohdatei']}, M50={d['M50']}, M90={d['M90']})")
        print("  k | " + " | ".join(f"T {s}" for s, _, _ in SYNTHESEN) + " | Gewinnanteil | Fehler grid_1e-3")
        for z in d["tabelle"]:
            g = z["gewinnanteil"]
            print(f" {z['k']:2d} | " + " | ".join(f"{z[s]['T_mittel']:7.1f}" for s, _, _ in SYNTHESEN)
                  + f" | {g:.3f}" + f" | {z['grid_1e-3']['operatorfehler_max']:.1e}")
        print("  max. Operatorfehler (Spektralnorm, Phase ausgerichtet) je Synthese, alle k:",
              {s: f"{max(z[s]['operatorfehler_max'] for z in d['tabelle']):.1e}" for s, _, _ in SYNTHESEN})
        print("  Tiefe / T-Tiefe bei k=12 (grid_1e-3):",
              d["tabelle"][-1]["grid_1e-3"]["tiefe_mittel"], d["tabelle"][-1]["grid_1e-3"]["t_tiefe_mittel"],
              " CNOT:", d["tabelle"][-1]["grid_1e-3"]["cnot"])


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    e = lauf(schnell=schnell)
    _drucke_tabelle(e)
    print("\nBorn-Vergleich:", {k: v for k, v in e["born_vergleich"].items()})
    print(e["urteil"])
    print(speichere("C-hardware" + ("-schnell" if schnell else ""), e))
