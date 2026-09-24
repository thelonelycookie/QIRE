"""Abteilung Clifford, Karte I-002 -- Stabilisator-Renyi-Entropie als Magie-Vorhersager.

HEILMEIER:
- Was wollen wir? Aus der gelernten Schaltung allein (ohne Rundungsversuche,
  ohne Nachtraining) vorhersagen, wie viel ihre Magie der Aufgabe nuetzt.
- Wie macht man es heute? C-clifford/C-mnist runden die Winkel und trainieren
  den Kopf neu -- teuer, je Seed und Stufe ein ganzes Training.
- Was ist neu? Das Magie-Monoton der Quanteninformation, die Stabilisator-
  Renyi-Entropie M2 (Leone/Oliviero/Hamma, PRL 128, 050402, 2022), wird zur
  MESSGROESSE fuer den Rundungsverlust. Quantenstufe Q3: M2 ist auf Hardware
  messbar.
- Woran merken wir es? Rangkorrelation M2 <-> gemessener Rundungsverlust
  (voll - clifford_k0) ueber die gespeicherten Seeds (S-1), und M2 muss den
  naiven Vorhersager „mittlerer Abstand zum Clifford-Gitter“ schlagen (S-2).

MAGIE-MASSE (beide exakt, keine Stichprobe ueber Pauli-Strings):
1. Operator-SRE M2(U): SRE des Choi-Zustands |U>> = (U x 1)|Phi+> auf 2n
   Qubits, also ueber alle 4^(2n) Pauli-Paare (P, Q). Aequivalent:
   M2(U) = -log2( sum_{P,Q} R_PQ^4 / d^2 ) mit der Pauli-Transfermatrix
   R_PQ = Tr(P U Q U^T)/d. Fuer jede Clifford-Schaltung ist R eine Vorzeichen-
   Permutation -> M2 = 0 (Positivkontrolle). Invariant unter Clifford-Gattern
   links und rechts, additiv unter Tensorprodukten; fuer Ry(t) geschlossen:
   M2 = -log2((1 + cos^4 t + sin^4 t)/2), bei t = pi/4 also log2(4/3).
   (Die „Pauli-Spektrum-Entropie“ von U selbst ist KEIN Magiemass -- sie ist
   schon fuer Hadamard 1 Bit. Deshalb der Choi-Zustand.)
2. Zustands-SRE M2(U|x>) = -log2( sum_P <P>^4 / d ), gemittelt ueber
   N_BILDER Testbilder, und ihr Zuwachs gegenueber M2(|x>) (was die Schaltung
   an Magie HINZUFUEGT; fuer Clifford-U exakt 0, weil M2 Clifford-invariant).

SCHNELLE RECHNUNG: <psi| X^a Z^b |psi> = sum_x psi(x^a) psi(x) (-1)^(b.x) ist
fuer festes a eine Walsh-Hadamard-Transformation (= Multiplikation mit der
unnormierten Hadamard-Matrix H). Fuer den Choi-Zustand wird daraus
H F_a H mit F_a(i,j) = U(i^a1, j^a2) U(i,j): 4^n Matrixprodukte der Groesse
2^n statt 4^(2n) dichter Pauli-Erwartungswerte. n = 8: 65.536 x 2 Produkte
256 x 256 (~4 * 10^12 Flops, einige Minuten auf einem Kern). Nur fuer reelle
U und psi (Ry + CNOT ist reell) -- geprueft gegen Brute Force im Vortest.

Kein Training. Winkel aus den Rohdaten von C-clifford-v2 (Ziffern, n = 6) und
C-mnist (MNIST 16x16, n = 8), je Seed „winkel_voll“.
"""
from __future__ import annotations

import math
import sys
import time
from functools import lru_cache
from itertools import chain, permutations
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import rankdata

from abteilungen.clifford import GITTER, LAGEN, QUBITS_JE_DATENSATZ, QireZelle, daten, runde_clifford
from labor.gemeinsam import ERGEBNISSE, speichere, zusammenfassung

QUELLEN = {"ziffern": "C-clifford-v2", "mnist": "C-mnist"}
SCHWELLE_RHO = 0.3      # Abbruchkriterium der Karte I-002
ALPHA = 0.05
N_PERM = 200_000        # Monte-Carlo-Permutationen, wenn exakt zu teuer
N_BOOT = 10_000
N_BILDER = 200
N_ZUFALL = {6: 10, 8: 4}       # Schaltungen mit zufaelligen Winkeln je Qubitzahl
N_GERUNDET = {6: 10, 8: 3}     # gerundete gelernte Schaltungen (Positivkontrolle)
TOL_NULL = 1e-9         # „0 bis auf Rundungsfehler“ (float64)

HYPOTHESEN = {
    "S-1": "Die Operator-SRE M2(U) der gelernten Schaltung korreliert mit dem "
           "gemessenen Rundungsverlust (voll - clifford_k0) ueber alle Seeds beider "
           "Datensaetze (20 Punkte): Spearman |rho| >= 0,3 und Permutations-p < 0,05. "
           "Kontrollarm (vorregistriert): der Zusammenhang darf nicht nur der "
           "Versatz zwischen den Datensaetzen sein -- Permutation INNERHALB der "
           "Datensaetze muss ebenfalls p < 0,05 ergeben, sonst HU.",
    "S-2": "M2(U) schlaegt den naiven Vorhersager „mittlerer Winkelabstand zum "
           "Clifford-Gitter“: |rho_SRE| > |rho_naiv|; H1 nur, wenn das geschichtete "
           "Bootstrap-95-%-KI von |rho_SRE| - |rho_naiv| ganz ueber 0 liegt.",
    "S-3": "Beschreibend: M2 je Datensatz, gegen Schaltungen mit zufaelligen Winkeln "
           "und gegen die Haar-Referenz log2(D+3)-2; gerundete Winkel muessen M2 = 0 "
           "geben (Positivkontrolle, sonst sind S-1/S-2 ungueltig). Zustands-SRE und "
           "ihr Zuwachs durch die Schaltung werden beschreibend mitgefuehrt.",
}


# ------------------------------------------------------------------ SRE-Bausteine
@lru_cache(maxsize=None)
def _hadamard(n: int) -> torch.Tensor:
    """Unnormierte Walsh-Hadamard-Matrix H_{b,x} = (-1)^(b.x), 2^n x 2^n."""
    h1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    H = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(n):
        H = torch.kron(H, h1)
    return H


@lru_cache(maxsize=None)
def _xor_index(d: int) -> torch.Tensor:
    idx = torch.arange(d)
    return idx[:, None] ^ idx[None, :]


def sre_zustaende(psi: torch.Tensor, block: int = 50) -> torch.Tensor:
    """M2 fuer jede Zeile von psi (reell, normiert), exakt ueber alle 4^n Paulis."""
    psi = psi.to(torch.float64)
    s, d = psi.shape
    n = int(round(math.log2(d)))
    H, A = _hadamard(n), _xor_index(d)
    out = []
    for i in range(0, s, block):
        p = psi[i:i + block]
        F = p[:, A] * p[:, None, :]           # F[s,a,x] = psi(a^x) psi(x)
        W = F @ H                             # W[s,a,b] = <X^a Z^b> (bis auf Phase)
        out.append(W.square_().square_().sum((1, 2)))
    summe = torch.cat(out)
    return -torch.log2(summe / d)


def sre_operator(U: torch.Tensor) -> float:
    """Operator-SRE = SRE des Choi-Zustands U(i,j)/sqrt(d) auf 2n Qubits (reelles U)."""
    U = U.to(torch.float64)
    d = U.shape[0]
    n = int(round(math.log2(d)))
    H, A = _hadamard(n), _xor_index(d)
    idx = torch.arange(d)
    summe = 0.0
    for a1 in range(d):
        G = U[idx ^ a1]                              # G[i,j] = U(i^a1, j)
        F = G[:, A].permute(1, 0, 2) * U             # F[a2,i,j] = U(i^a1, j^a2) U(i,j)
        W = torch.matmul(torch.matmul(H, F), H)      # W[a2,b1,b2] = d * <Choi|P|Choi>
        summe += float(W.square_().square_().sum())
    return float(-math.log2(summe / d ** 6))


def sre_brute(psi: np.ndarray) -> float:
    """Definition woertlich: alle 4^n Pauli-Strings als dichte Matrizen (nur kleine n)."""
    psi = np.asarray(psi, dtype=complex)
    d = len(psi)
    n = int(round(math.log2(d)))
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    summe = 0.0
    for wahl in np.ndindex(*([4] * n)):
        P = np.ones((1, 1))
        for w in wahl:
            P = np.kron(P, paulis[w])
        summe += float(np.real(np.conj(psi) @ P @ psi)) ** 4
    return float(-math.log2(summe / d))


def sre_ry(theta: float) -> float:
    """Geschlossene Form fuer eine einzelne Ry(theta) (Operator-SRE)."""
    c, s = math.cos(theta) ** 4, math.sin(theta) ** 4
    return float(-math.log2((1 + c + s) / 2))


def haar_referenz(D: int) -> float:
    """Typische SRE eines Haar-zufaelligen Zustands der Dimension D (Leone et al. 2022)."""
    return float(math.log2(D + 3) - 2)


def unitaer_aus_winkeln(winkel, n: int, lagen: int = LAGEN, cnot: bool = True) -> torch.Tensor:
    """Genau die Schaltung aus clifford.QireZelle, aber in float64."""
    alt = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.set_default_dtype(torch.float64)
        try:
            m = QireZelle(n=n, lagen=lagen, cnot=cnot).double()
            th = torch.as_tensor(winkel, dtype=torch.float64).reshape(lagen, n)
            m.fixiere(th, torch.zeros(lagen, n))
            with torch.no_grad():
                U = m.unitaer().detach().clone()
        finally:
            torch.set_default_dtype(alt)
    return U


def _abstand(winkel) -> float:
    th = torch.as_tensor(winkel, dtype=torch.float64)
    return float((th - runde_clifford(th)).abs().mean())


# ------------------------------------------------------------------ Statistik
@lru_cache(maxsize=None)
def _alle_perm(n: int) -> np.ndarray:
    anzahl = math.factorial(n)
    return np.fromiter(chain.from_iterable(permutations(range(n))), dtype=np.int8,
                       count=anzahl * n).reshape(anzahl, n)


def _rho_raenge(rx, ry) -> float:
    rx, ry = rx - rx.mean(), ry - ry.mean()
    nenner = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / nenner) if nenner > 0 else 0.0


def spearman_perm(x, y, gruppen=None, n_perm=N_PERM, seed=0) -> dict:
    """Spearman-rho (Durchschnittsraenge bei Bindungen) mit zweiseitigem
    Permutationstest -- NICHT asymptotisch (Lehre aus Q-011).
    Ohne Gruppen und n <= 10: exakt (alle n! Permutationen). Sonst Monte Carlo,
    p = (1 + #extrem) / (1 + N). Mit Gruppen: y wird nur INNERHALB der Gruppen
    permutiert (haelt den Versatz zwischen Datensaetzen fest)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    rho = _rho_raenge(rx, ry)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    nenner = math.sqrt(float(rxc @ rxc) * float(ryc @ ryc))
    if nenner == 0:
        return {"rho": 0.0, "p": 1.0, "methode": "konstant"}
    n = len(x)
    if gruppen is None and n <= 10:
        perms = _alle_perm(n)
        treffer = 0
        for i in range(0, len(perms), 400_000):
            r = (ryc[perms[i:i + 400_000]] @ rxc) / nenner
            treffer += int(np.sum(np.abs(r) >= abs(rho) - 1e-12))
        return {"rho": rho, "p": treffer / len(perms), "methode": f"exakt ({len(perms)} Perm.)"}
    rng = np.random.default_rng(seed)
    gruppen = np.zeros(n, int) if gruppen is None else np.asarray(gruppen)
    perm = np.tile(np.arange(n), (n_perm, 1))
    for g in np.unique(gruppen):
        spalten = np.where(gruppen == g)[0]
        perm[:, spalten] = spalten[rng.permuted(np.tile(np.arange(len(spalten)), (n_perm, 1)), axis=1)]
    r = (ryc[perm] @ rxc) / nenner
    treffer = int(np.sum(np.abs(r) >= abs(rho) - 1e-12))
    return {"rho": rho, "p": (1 + treffer) / (1 + n_perm),
            "methode": f"Monte Carlo ({n_perm} Perm." + (", geschichtet)" if len(np.unique(gruppen)) > 1 else ")")}


def _rho_zeilenweise(X, Y) -> np.ndarray:
    rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    nenner = np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / nenner


def bootstrap_differenz(a, b, y, gruppen, n_boot=N_BOOT, seed=1) -> dict:
    """Geschichteter Bootstrap (Ziehen mit Zuruecklegen je Datensatz) fuer
    |rho(a,y)| - |rho(b,y)|; Perzentil-KI 95 %."""
    a, b, y, gruppen = map(np.asarray, (a, b, y, gruppen))
    rng = np.random.default_rng(seed)
    teile = [rng.choice(np.where(gruppen == g)[0], size=(n_boot, int(np.sum(gruppen == g))))
             for g in np.unique(gruppen)]
    idx = np.concatenate(teile, axis=1)
    delta = np.abs(_rho_zeilenweise(a[idx], y[idx])) - np.abs(_rho_zeilenweise(b[idx], y[idx]))
    delta = delta[np.isfinite(delta)]
    return {"ki_unten": float(np.percentile(delta, 2.5)), "ki_oben": float(np.percentile(delta, 97.5)),
            "gueltig": int(len(delta))}


# ------------------------------------------------------------------ Auswertung
PRAEDIKTOREN = ["sre_op", "abstand", "sre_zustand_zuwachs", "sre_zustand"]


def auswertung(zeilen, kontrollen) -> dict:
    ds = np.array([z["datensatz"] for z in zeilen])
    gruppen = np.unique(ds, return_inverse=True)[1]
    verlust = np.array([z["verlust"] for z in zeilen])
    aus = {"n": len(zeilen), "datensaetze": sorted(set(ds.tolist()))}
    for pr in PRAEDIKTOREN:
        if pr not in zeilen[0]:
            continue
        w = np.array([z[pr] for z in zeilen])
        e = {"gesamt": spearman_perm(w, verlust),
             "geschichtet": spearman_perm(w, verlust, gruppen=gruppen, seed=2)}
        for name in aus["datensaetze"]:
            m = ds == name
            e[name] = spearman_perm(w[m], verlust[m])
        aus[f"rho_{pr}"] = e
    aus["rho_sre_vs_abstand"] = spearman_perm([z["sre_op"] for z in zeilen], [z["abstand"] for z in zeilen])
    # S-1
    aus["S1_rho"] = aus["rho_sre_op"]["gesamt"]["rho"]
    aus["S1_p"] = aus["rho_sre_op"]["gesamt"]["p"]
    aus["S1_p_geschichtet"] = aus["rho_sre_op"]["geschichtet"]["p"]
    # S-2
    rs, rn = abs(aus["S1_rho"]), abs(aus["rho_abstand"]["gesamt"]["rho"])
    aus["S2_delta"] = rs - rn
    aus.update({f"S2_{k}": v for k, v in bootstrap_differenz(
        [z["sre_op"] for z in zeilen], [z["abstand"] for z in zeilen], verlust, gruppen).items()})
    # S-3
    for name in aus["datensaetze"]:
        m = ds == name
        aus[f"S3_{name}_gelernt"] = zusammenfassung(np.array([z["sre_op"] for z in zeilen])[m])
        aus[f"S3_{name}_verlust"] = zusammenfassung(verlust[m])
        zuf = kontrollen.get("zufall_op", {}).get(name)
        if zuf:
            aus[f"S3_{name}_zufall"] = zusammenfassung(zuf) if len(zuf) > 1 else {"mittel": float(zuf[0]), "n": 1}
            aus[f"S3_{name}_anteil_von_zufall"] = float(aus[f"S3_{name}_gelernt"]["mittel"] / np.mean(zuf))
    ger = list(kontrollen.get("gerundet_op", [])) + list(kontrollen.get("gerundet_zustand_zuwachs", []))
    aus["S3_gerundet_max_betrag"] = float(np.max(np.abs(ger))) if ger else None
    aus["kontrolle_ok"] = bool(ger) and aus["S3_gerundet_max_betrag"] < TOL_NULL and kontrollen.get("bausteine_ok", False)
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum. Jeder Zweig hat eigenen Text."""
    u = {}
    if not a["kontrolle_ok"]:
        text = (f"UNGUELTIG: Positivkontrolle/Bausteine durchgefallen (gerundet max |M2| = "
                f"{a['S3_gerundet_max_betrag']}) -- Messgeraet kaputt, kein Urteil")
        return {"S-1": text, "S-2": text, "S-3": text}
    rho, p, pg = a["S1_rho"], a["S1_p"], a["S1_p_geschichtet"]
    if abs(rho) >= SCHWELLE_RHO and p < ALPHA:
        if pg < ALPHA:
            richtung = "mehr Magie -> mehr Rundungsverlust" if rho > 0 else \
                "UMGEKEHRTES Vorzeichen: mehr Magie -> weniger Verlust"
            u["S-1"] = f"H1: rho = {rho:+.2f}, p = {p:.4f} (geschichtet {pg:.4f}); {richtung}"
        else:
            u["S-1"] = (f"HU: rho = {rho:+.2f}, p = {p:.4f}, aber innerhalb der Datensaetze "
                        f"p = {pg:.4f} -- nur Versatz zwischen Ziffern und MNIST")
    elif abs(rho) < SCHWELLE_RHO:
        u["S-1"] = f"H0: |rho| = {abs(rho):.2f} < {SCHWELLE_RHO} -- Abbruchkriterium der Karte erfuellt"
    else:
        u["S-1"] = f"HU: |rho| = {abs(rho):.2f} >= {SCHWELLE_RHO}, aber p = {p:.4f} >= {ALPHA} -- zu wenig Aufloesung"
    d, lo, hi = a["S2_delta"], a["S2_ki_unten"], a["S2_ki_oben"]
    if d > 0 and lo > 0:
        u["S-2"] = f"H1: SRE schlaegt Winkelabstand (Delta|rho| = {d:+.2f}, KI [{lo:+.2f}, {hi:+.2f}])"
    elif d <= 0:
        u["S-2"] = f"H0: SRE nicht besser als Winkelabstand (Delta|rho| = {d:+.2f}, KI [{lo:+.2f}, {hi:+.2f}])"
    else:
        u["S-2"] = f"HU: SRE vorn, aber nicht gesichert (Delta|rho| = {d:+.2f}, KI [{lo:+.2f}, {hi:+.2f}])"
    teile = []
    for name in a["datensaetze"]:
        g = a[f"S3_{name}_gelernt"]
        t = f"{name}: M2 = {g['mittel']:.3f} +- {g.get('std', 0):.3f}"
        if f"S3_{name}_zufall" in a:
            t += f" (zufaellige Winkel {a[f'S3_{name}_zufall']['mittel']:.3f}, Anteil {a[f'S3_{name}_anteil_von_zufall']:.2f})"
        teile.append(t)
    u["S-3"] = "; ".join(teile) + f"; gerundet max |M2| = {a['S3_gerundet_max_betrag']:.1e} (Positivkontrolle bestanden)"
    return u


# ------------------------------------------------------------------ Vortest
def _bausteine(ausgabe=True) -> bool:
    """Baustein-Pruefungen des Messgeraets."""
    ok = True
    rng = np.random.default_rng(7)

    def pruefe(name, ist, soll, tol=1e-9):
        nonlocal ok
        gut = abs(ist - soll) < tol
        ok &= gut
        if ausgabe and not gut:
            print(f"  BAUSTEIN FEHLER {name}: ist {ist:.3e}, soll {soll:.3e}")

    # 1) schnelle Zustands-SRE gegen die woertliche Definition (n = 1..3)
    for n in (1, 2, 3):
        psi = rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        pruefe(f"zustand brute n={n}", float(sre_zustaende(torch.tensor(psi)[None])[0]), sre_brute(psi))
    # 2) schnelle Operator-SRE gegen Brute Force auf dem Choi-Zustand (n = 1, 2, 3)
    for n in (1, 2, 3):
        U = unitaer_aus_winkeln(rng.uniform(0, 2 * math.pi, 2 * n), n, lagen=2)
        pruefe(f"operator brute n={n}", sre_operator(U), sre_brute(U.numpy().reshape(-1) / math.sqrt(2 ** n)))
    # 3) Clifford-Schaltungen -> 0 (Identitaet, CNOT-Ring, zufaellige Gitterwinkel)
    for n in (4, 6):
        pruefe(f"identitaet n={n}", sre_operator(unitaer_aus_winkeln(np.zeros(LAGEN * n), n)), 0.0)
        g = rng.integers(0, 4, LAGEN * n) * GITTER
        pruefe(f"clifford n={n}", sre_operator(unitaer_aus_winkeln(g, n)), 0.0)
    # 4) ein einzelner T-artiger Winkel pi/4 zwischen Clifford-Gattern -> log2(4/3)
    for n in (4, 6):
        g = rng.integers(0, 4, LAGEN * n) * GITTER
        g[int(rng.integers(0, LAGEN * n))] = math.pi / 4
        pruefe(f"T-winkel n={n}", sre_operator(unitaer_aus_winkeln(g, n)), math.log2(4 / 3))
    pruefe("geschlossene Form pi/4", sre_ry(math.pi / 4), math.log2(4 / 3), 1e-12)
    # 5) Additivitaet: eine Lage ohne CNOT = Tensorprodukt
    th = rng.uniform(0, 2 * math.pi, 4)
    pruefe("additiv", sre_operator(unitaer_aus_winkeln(th, 4, lagen=1, cnot=False)), sum(sre_ry(t) for t in th))
    # 6) Zustaende: |0...0> -> 0, Ry(pi/4)|0> -> log2(4/3), Clifford-invariant
    pruefe("stabilisatorzustand", float(sre_zustaende(torch.eye(64, dtype=torch.float64)[:1])[0]), 0.0)
    pruefe("T-zustand", float(sre_zustaende(torch.tensor([[math.cos(math.pi / 8), math.sin(math.pi / 8)]], dtype=torch.float64))[0]),
           math.log2(4 / 3))
    psi = torch.tensor(rng.normal(size=(3, 64)))
    psi = psi / psi.norm(dim=1, keepdim=True)
    Uc = unitaer_aus_winkeln(rng.integers(0, 4, LAGEN * 6) * GITTER, 6)
    pruefe("zustand clifford-invariant", float((sre_zustaende(psi @ Uc.T) - sre_zustaende(psi)).abs().max()), 0.0)
    # 7) Orthogonalitaet der float64-Unitaeren
    U = unitaer_aus_winkeln(rng.uniform(0, 7, LAGEN * 6), 6)
    pruefe("orthogonal", float((U @ U.T - torch.eye(64, dtype=torch.float64)).abs().max()), 0.0, 1e-12)
    return bool(ok)


def _synthetisch(rng, art):
    """20 Zeilen (10 Ziffern, 10 MNIST) mit erzwungener Struktur."""
    zeilen = []
    fest = np.random.default_rng(123).normal(size=20)       # fester Stoervektor (Raenge fix)
    for j, name in enumerate(["ziffern", "mnist"]):
        for s in range(10):
            i = 10 * j + s
            v = 0.01 + 0.004 * s + rng.normal(0, 1e-5)        # Verlust, steigt mit s
            if art == "versatz":
                v += 0.03 * j
            elif art == "maessig":                             # feste Reihenfolge zwischen Datensaetzen
                v += 0.001 * j
            elif art == "hu_s2":                               # Paare zwischen Datensaetzen stets konkordant
                v += 0.1 * j
            if art in ("h1", "hu_s2"):
                sre = 3 * v
            elif art == "negativ":
                sre = -3 * v
            elif art == "null":                                # U-Form -> rho ~ 0
                sre = (s - 4.5) ** 2 + 0.1 * (s % 2) + 2.0 * j
            elif art == "versatz":                             # innen U-Form, aussen Versatz
                sre = (s - 4.5) ** 2 + 0.1 * (s % 2) + 30.0 * j
            elif art == "maessig":                             # rho ~ 0,35
                sre = s / 9 + 1.0 * fest[i]
            sre += rng.normal(0, 1e-7)
            if art == "hu_s2":                                 # Abstand = SRE mit einem Rangtausch
                ab = 3 * (0.01 + 0.004 * {3: 4, 4: 3}.get(s, s) + 0.1 * j)
            elif art == "null":
                ab = v                                         # naiver Vorhersager perfekt
            else:
                ab = rng.normal(0.35, 0.05)
            zeilen.append({"datensatz": name, "seed": s, "verlust": v, "sre_op": sre,
                           "abstand": ab + rng.normal(0, 1e-7), "sre_zustand_zuwachs": rng.normal(1, 0.1)})
    return zeilen


def vortest(ziehungen=5, bausteine=True) -> bool:
    """Pflicht-Vortest: Bausteine des Messgeraets + jeden Urteilszweig erzwingen."""
    ok = _bausteine() if bausteine else True
    if not ok:
        print("  VORTEST: Bausteine durchgefallen")
    k_gut = {"gerundet_op": [0.0, 1e-13], "gerundet_zustand_zuwachs": [0.0], "zufall_op": {}, "bausteine_ok": True}
    k_schlecht = dict(k_gut, gerundet_op=[0.0, 0.01])
    faelle = [
        ("h1", k_gut, {"S-1": "H1", "S-2": "H1"}),
        ("negativ", k_gut, {"S-1": "H1: rho = -"}),
        ("null", k_gut, {"S-1": "H0", "S-2": "H0"}),
        ("maessig", k_gut, {"S-1": "HU: |rho|"}),
        ("versatz", k_gut, {"S-1": "HU: rho"}),
        ("hu_s2", k_gut, {"S-1": "H1", "S-2": "HU"}),
        ("h1", k_schlecht, {"S-1": "UNGUELTIG", "S-2": "UNGUELTIG"}),
    ]
    for z in range(ziehungen):
        rng = np.random.default_rng(1000 + z)
        for art, k, erwartet in faelle:
            u = urteil(auswertung(_synthetisch(rng, art), k))
            for h, marke in erwartet.items():
                if not u[h].startswith(marke):
                    ok = False
                    print("  VORTEST FEHLER", art, h, u[h])
    return bool(ok)


# ------------------------------------------------------------------ Lauf
def _rohdaten(name):
    dateien = sorted(ERGEBNISSE.glob(f"{QUELLEN[name]}__*.json"))
    import json
    return dateien[-1].name, json.loads(dateien[-1].read_text())


def _testbilder(name, seed, anzahl):
    _, _, xte, _ = daten(name, seed)
    x = xte[:anzahl].to(torch.float64)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def lauf(schnell=False) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    n_bilder = 20 if schnell else N_BILDER
    zeilen, quellen = [], {}
    kontrollen = {"gerundet_op": [], "gerundet_zustand_zuwachs": [], "zufall_op": {}, "zufall_winkel": {}}
    print("  Bausteine ...", flush=True)
    kontrollen["bausteine_ok"] = _bausteine()
    for name in QUELLEN:
        datei, roh = _rohdaten(name)
        quellen[name] = datei
        n = QUBITS_JE_DATENSATZ[name]
        roh_zeilen = roh["zeilen"][:2] if (schnell and n == 8) else roh["zeilen"]
        n_ger = 1 if schnell else N_GERUNDET[n]
        for i, r in enumerate(roh_zeilen):
            t1 = time.time()
            th = r["winkel_voll"]
            U = unitaer_aus_winkeln(th, n)
            x = _testbilder(name, r["seed"], n_bilder)
            m_ein = sre_zustaende(x)
            m_aus = sre_zustaende(x @ U.T)
            z = {"datensatz": name, "seed": r["seed"], "n": n,
                 "voll": r["voll"], "clifford_k0": r["clifford_k0"], "ohne_schaltung": r["ohne_schaltung"],
                 "verlust": r["voll"] - r["clifford_k0"], "abstand": _abstand(th),
                 "sre_op": sre_operator(U),
                 "sre_zustand": float(m_aus.mean()), "sre_eingang": float(m_ein.mean()),
                 "sre_zustand_zuwachs": float((m_aus - m_ein).mean())}
            if i < n_ger:          # Positivkontrolle an echten gelernten Winkeln
                Ur = unitaer_aus_winkeln(runde_clifford(torch.tensor(th, dtype=torch.float64)), n)
                kontrollen["gerundet_op"].append(sre_operator(Ur))
                kontrollen["gerundet_zustand_zuwachs"].append(
                    float((sre_zustaende(x @ Ur.T) - m_ein).abs().max()))
            zeilen.append(z)
            print(f"  {name} Seed {r['seed']}: M2(U) = {z['sre_op']:.4f}  Zuwachs = {z['sre_zustand_zuwachs']:+.4f}  "
                  f"Verlust = {z['verlust']:+.4f}  Abstand = {z['abstand']:.3f}  ({time.time() - t1:.0f} s)", flush=True)
        g = torch.Generator().manual_seed(4242 + n)
        anzahl = 1 if schnell else N_ZUFALL[n]
        kontrollen["zufall_winkel"][name] = []
        kontrollen["zufall_op"][name] = []
        for _ in range(anzahl):
            w = torch.rand(LAGEN * n, generator=g, dtype=torch.float64) * 2 * math.pi
            kontrollen["zufall_winkel"][name].append(w.tolist())
            kontrollen["zufall_op"][name].append(sre_operator(unitaer_aus_winkeln(w, n)))
        print(f"  {name} zufaellige Winkel: M2 = {np.round(kontrollen['zufall_op'][name], 3).tolist()}", flush=True)
    kontrollen["haar_referenz"] = {name: haar_referenz(4 ** QUBITS_JE_DATENSATZ[name]) for name in QUELLEN}
    ergebnis = {"abteilung": "clifford", "karte": "I-002", "version": 1, "quellen": quellen,
                "hypothesen": HYPOTHESEN, "bilder_je_seed": n_bilder,
                "methode": "Operator-SRE = SRE des Choi-Zustands (alle 4^(2n) Paulis, exakt, float64); "
                           "Zustands-SRE ueber alle 4^n Paulis; Spearman mit Permutationstest",
                "zeilen": zeilen, "kontrollen": kontrollen}
    ergebnis["auswertung"] = auswertung(zeilen, kontrollen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN", flush=True)
    schnell = "--schnell" in sys.argv
    e = lauf(schnell=schnell)
    print(speichere("I-002" + ("-schnell" if schnell else ""), e))
    for k, v in e["urteil"].items():
        print(f"  {k}: {v}")
