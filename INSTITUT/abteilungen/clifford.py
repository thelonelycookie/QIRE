"""Abteilung Clifford -- das Magie-Budget einer Aufgabe.

IDEE (aus der Quanteninformation, Ressourcentheorie der „Magie“):
In einem Quantencomputer sind Clifford-Gatter „billig“ (klassisch effizient
simulierbar, fehlerkorrigiert fast kostenlos), nur die Nicht-Clifford-Gatter
(T-Gatter, „Magie“) kosten. Die T-Zahl ist DIE Ressourcengroesse.

Uebertragen auf QIRE: Eine reelle Ry-Drehung mit Winkel k·pi/2 ist eine
Vorzeichen-Permutation oder eine (skalierte) Hadamard-Matrix mit Eintraegen
±1/sqrt(2). Ein Kronecker-Produkt solcher Drehungen plus CNOT-Ring rechnet
deshalb **ohne eine einzige Multiplikation** (nur Additionen/Subtraktionen,
der Faktor 1/sqrt(2)^k ist global und faellt vor dem Kopf heraus).
Nur die Winkel, die NICHT auf dem Clifford-Gitter liegen, kosten
Multiplikationen -- sie sind das „Magie-Budget“.

FRAGE: Wie viele Nicht-Clifford-Winkel braucht eine Aufgabe wirklich?
Wenn die Antwort „fast keine“ ist, laeuft die Schicht multiplikationsfrei.

Protokoll: Kriterien stehen VOR dem Lauf hier im Code (Regel 7), eigene
Lernraten je Arm auf eigenen Tuning-Seeds (Regel 1a), Gitter wird in beide
Richtungen verlaengert (1b), Suche mit vollem Budget (1c), exakter Wilcoxon,
Aufloesungspruefung (Regel 6), Rohdaten sofort sichern (Regel 5).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from labor.gemeinsam import (genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt,
                             ziffern, zusammenfassung)

N_QUBITS = 6            # 2^6 = 64 = 8x8-Pixel
LAGEN = 4
GITTER = math.pi / 2    # Clifford-Gitter fuer reelle Ry-Drehungen
TOLERANZ = 0.02         # 2 Prozentpunkte: „gleich gut“ im Sinne dieser Frage
MAGIE_STUFEN = [0, 1, 2, 4, 8, 12]

HYPOTHESEN = {
    "C-0": "Vorbedingung: die trainierte Schaltung schlaegt die Identitaet "
           "(Born-Auslese direkt auf x). Sonst ist jede Clifford-Aussage leer.",
    "C-1": "Clifford-Rundung: alle Winkel auf k·pi/2 runden (Magie 0), Kopf "
           "nachtrainieren -> hoechstens 2 pp Verlust gegenueber dem vollen Modell.",
    "C-2": "Die gerundete Schaltung ist besser als eine ZUFAELLIGE Clifford-"
           "Schaltung und besser als die feste Hadamard-Schaltung (sonst hat "
           "das Training nichts gelernt, was die Rundung ueberlebt).",
    "C-3": "Magie-Budget M*: kleinste Zahl freier Nicht-Clifford-Winkel, mit der "
           "der Verlust unter 2 pp liegt (beschreibend, Kurve ueber k).",
}


# ------------------------------------------------------------------ Modell
def _cnot_ring_perm(n: int) -> torch.Tensor:
    """CNOT(q -> q+1 mod n) nacheinander als eine einzige Indexpermutation."""
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1   # Bit q = Qubit q
    for q in range(n):
        t = (q + 1) % n
        bits[:, t] ^= bits[:, q]
    neu = (bits << (n - 1 - np.arange(n))).sum(1)
    perm = np.empty_like(neu)
    perm[neu] = idx          # neuer Zustand[neu] = alter Zustand[idx]
    return torch.tensor(perm)


class QireZelle(nn.Module):
    """Amplitudenkodierung -> L x (Ry je Qubit, CNOT-Ring) -> Born -> Kopf."""

    def __init__(self, n=N_QUBITS, lagen=LAGEN, auslese="born", cnot=True):
        super().__init__()
        self.n, self.lagen, self.auslese, self.cnot = n, lagen, auslese, cnot
        self.theta = nn.Parameter(torch.rand(lagen, n) * 2 * math.pi)
        self.register_buffer("perm", _cnot_ring_perm(n))
        self.register_buffer("frei", torch.ones(lagen, n))      # 1 = Winkel trainierbar
        self.register_buffer("fest", torch.zeros(lagen, n))     # Werte der fixierten Winkel
        breite = 2 ** n if auslese == "born" else n
        self.kopf = nn.Linear(breite, 10)

    def winkel(self):
        return self.frei * self.theta + (1 - self.frei) * self.fest

    def unitaer(self):
        """Die ganze Schaltung als 2^n x 2^n-Matrix (Kronecker-Produkte, CNOT als
        Zeilenpermutation). Mathematisch identisch zur Gatterfolge, aber auf der
        CPU ~100x schneller zu trainieren. Die Multiplikationszahl der INFERENZ
        wird analytisch fuer die Gatterform gezaehlt (multiplikationen())."""
        th = self.winkel()
        U = torch.eye(2 ** self.n)
        for l in range(self.lagen):
            K = torch.ones(1, 1)
            for q in range(self.n):
                c, s = torch.cos(th[l, q] / 2), torch.sin(th[l, q] / 2)
                K = torch.kron(K, torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
            U = K @ U
            if self.cnot:
                U = U[self.perm]
        return U

    def merkmale(self, x):
        b = x.shape[0]
        psi = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        psi = psi @ self.unitaer().T
        p = psi.reshape(b, -1) ** 2
        if self.auslese == "z":
            idx = torch.arange(2 ** self.n)
            vz = torch.stack([1 - 2 * ((idx >> (self.n - 1 - q)) & 1) for q in range(self.n)], 1)
            return p @ vz.float()
        return p

    def forward(self, x):
        return self.kopf(self.merkmale(x))

    def fixiere(self, werte: torch.Tensor, maske_frei: torch.Tensor):
        with torch.no_grad():
            self.fest.copy_(werte)
            self.frei.copy_(maske_frei.float())
            self.theta.copy_(werte)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.kopf = nn.Linear(64, 10)

    def forward(self, x):
        return self.kopf(x)


def runde_clifford(th: torch.Tensor) -> torch.Tensor:
    return torch.round(th / GITTER) * GITTER


def multiplikationen(n=N_QUBITS, lagen=LAGEN, magie=None, auslese="born") -> int:
    """Multiplikationen je Inferenz. Eine 2x2-Drehung auf 2^n Amplituden kostet
    2*2^n Mult.; auf dem Clifford-Gitter 0 (nur ±-Additionen)."""
    d = 2 ** n
    dreh = lagen * n if magie is None else magie
    kopf = (d if auslese == "born" else n) * 10
    return dreh * 2 * d + d + kopf


# ------------------------------------------------------------------ Arme
def _baue(arm, seed):
    torch.manual_seed(seed)
    if arm == "dicht_linear":
        return Linear(), None
    m = QireZelle(auslese="z" if arm == "voll_z" else "born")
    L, n = m.lagen, m.n
    if arm == "ohne_schaltung":
        m.fixiere(torch.zeros(L, n), torch.zeros(L, n))
    elif arm == "hadamard_fest":
        m.fixiere(torch.full((L, n), GITTER), torch.zeros(L, n))
    elif arm == "zufall_clifford":
        g = torch.Generator().manual_seed(10_000 + seed)
        m.fixiere(torch.randint(0, 4, (L, n), generator=g).float() * GITTER, torch.zeros(L, n))
    return m, None


ARME_FREI = ["voll", "voll_z", "dicht_linear", "ohne_schaltung", "hadamard_fest", "zufall_clifford"]


def _tuning_punkt(a):
    arm, lr, s, schritte = a
    xtr, ytr, _, _ = ziffern(s)
    n_val = len(ytr) // 4
    m, _ = _baue(arm, s)
    trainiere(m, xtr[n_val:], ytr[n_val:], schritte, lr, seed=s)
    return genauigkeit(m(xtr[:n_val]), ytr[:n_val])


def _lr_waehlen(arme, tuning_seeds, schritte, gitter=(0.005, 0.02, 0.08)):
    """Regel 1: eigene Rate je Arm, auf eigenen Tuning-Seeds, Validierung aus dem
    Trainingsteil, volles Budget (1c), Gitter wird verlaengert, wenn das Optimum
    am Rand liegt -- in BEIDE Richtungen (1b)."""
    gitter = {arm: list(gitter) for arm in arme}
    wert = {arm: {} for arm in arme}
    offen = set(arme)
    info = {}
    for _ in range(4):
        aufg = [(arm, lr, s, schritte) for arm in offen for lr in gitter[arm]
                if lr not in wert[arm] for s in tuning_seeds]
        erg = parallel(_tuning_punkt, aufg)
        tmp = {}
        for (arm, lr, s, _), acc in zip(aufg, erg):
            tmp.setdefault((arm, lr), []).append(acc)
        for (arm, lr), v in tmp.items():
            wert[arm][lr] = float(np.mean(v))
        for arm in list(offen):
            g = sorted(gitter[arm])
            best = max(g, key=lambda v: wert[arm][v])
            if best == g[0]:
                gitter[arm] = [g[0] / 4] + g
            elif best == g[-1]:
                gitter[arm] = g + [g[-1] * 4]
            else:
                info[arm] = {"lr": best, "gitter": wert[arm], "am_rand": False}
                offen.discard(arm)
        if not offen:
            break
    for arm in offen:
        best = max(wert[arm], key=wert[arm].get)
        info[arm] = {"lr": best, "gitter": wert[arm], "am_rand": True}
    return info


def _magie_reihenfolge(theta_voll):
    """Welche Winkel bleiben frei? Die mit dem groessten Abstand zum Gitter."""
    abstand = (theta_voll - runde_clifford(theta_voll)).abs().flatten()
    return torch.argsort(abstand, descending=True)


def _seed_lauf(a):
    s, schritte, lr = a
    xtr, ytr, xte, yte = ziffern(s)
    z = {"seed": s}
    for arm in ARME_FREI:
        m, _ = _baue(arm, s)
        trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
        z[arm] = genauigkeit(m(xte), yte)
        if arm == "voll":
            voll = m
    th = voll.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    z["abstand_zum_gitter"] = float((th - runde_clifford(th)).abs().mean())
    reihenfolge = _magie_reihenfolge(th)
    for k in MAGIE_STUFEN:
        m = QireZelle()
        m.load_state_dict(voll.state_dict())
        frei = torch.zeros(LAGEN * N_QUBITS)
        frei[reihenfolge[:k]] = 1
        m.fixiere(runde_clifford(th), frei.reshape(LAGEN, N_QUBITS))
        with torch.no_grad():   # freie Winkel starten beim trainierten Wert
            m.theta.copy_(torch.where(m.frei.bool(), th, runde_clifford(th)))
        z[f"clifford_k{k}_ohne_nachtraining"] = genauigkeit(m(xte), yte)
        trainiere(m, xtr, ytr, schritte, lr["kopf"], seed=s)
        z[f"clifford_k{k}"] = genauigkeit(m(xte), yte)
    return z


def lauf(seeds=range(10), schritte=1500, tuning_seeds=(900, 901), schnell=False):
    if schnell:
        seeds, schritte, tuning_seeds = range(2), 150, (900,)
    t0 = time.time()
    lr_info = _lr_waehlen(ARME_FREI, tuning_seeds, schritte)
    lr = {arm: v["lr"] for arm, v in lr_info.items()}
    # Rundungsarme trainieren nur Kopf (+k Magiewinkel): Rate von hadamard_fest
    # (gleiche Lage: feste Schaltung, trainierter Kopf).
    lr["kopf"] = lr["hadamard_fest"]
    zeilen = parallel(_seed_lauf, [(s, schritte, lr) for s in seeds])
    for z in zeilen:
        print(f"  Seed {z['seed']}: voll {z['voll']:.4f}  k0 {z['clifford_k0']:.4f}  "
              f"zufall {z['zufall_clifford']:.4f}  ohne {z['ohne_schaltung']:.4f}", flush=True)
    ergebnis = {"abteilung": "clifford", "hypothesen": HYPOTHESEN, "lr": lr_info,
                "schritte": schritte, "zeilen": zeilen, "sekunden": time.time() - t0,
                "multiplikationen": {"voll": multiplikationen(),
                                     **{f"clifford_k{k}": multiplikationen(magie=k) for k in MAGIE_STUFEN},
                                     "dicht_linear": 640, "voll_z": multiplikationen(auslese="z")}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def auswertung(zeilen) -> dict:
    a = lambda k: np.array([z[k] for z in zeilen])
    aus = {k: zusammenfassung(a(k)) for k in zeilen[0] if k not in ("seed", "winkel_voll")}
    d0 = a("voll") - a("ohne_schaltung")
    d1 = a("clifford_k0") - a("voll")
    d2z = a("clifford_k0") - a("zufall_clifford")
    d2h = a("clifford_k0") - a("hadamard_fest")
    aus["C0_diff"], aus["C0_p"] = float(d0.mean()), wilcoxon_exakt(d0)
    aus["C1_diff"] = float(d1.mean())
    aus["C1_aufloesung"] = float(d1.std(ddof=1) / math.sqrt(len(d1))) if len(d1) > 1 else 1.0
    aus["C2_diff_zufall"], aus["C2_p_zufall"] = float(d2z.mean()), wilcoxon_exakt(d2z)
    aus["C2_diff_hadamard"], aus["C2_p_hadamard"] = float(d2h.mean()), wilcoxon_exakt(d2h)
    kurve = {k: float((a(f"clifford_k{k}") - a("voll")).mean()) for k in MAGIE_STUFEN}
    aus["C3_kurve"] = kurve
    aus["C3_Mstern"] = next((k for k in MAGIE_STUFEN if kurve[k] >= -TOLERANZ), None)
    aus["n"] = len(zeilen)
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum. Jeder Zweig hat eigenen Text."""
    u = {}
    if a["C1_aufloesung"] > TOLERANZ / 2:
        u["C-1"] = "HU: Aufloesung reicht nicht (SE der Differenz > 1 pp) -- kein Urteil"
    elif a["C1_diff"] >= -TOLERANZ:
        u["C-1"] = "H1: Clifford-Rundung kostet < 2 pp"
    else:
        u["C-1"] = "H0: Rundung kostet > 2 pp -- Magie wird gebraucht (siehe C-3)"
    u["C-0"] = ("H1: Schaltung schlaegt Identitaet" if a["C0_diff"] > 0 and a["C0_p"] < 0.05
                else "H0: Schaltung nicht nachweisbar besser als Identitaet")
    sig_z = a["C2_diff_zufall"] > 0 and a["C2_p_zufall"] < 0.05
    sig_h = a["C2_diff_hadamard"] > 0 and a["C2_p_hadamard"] < 0.05
    if sig_z and sig_h:
        u["C-2"] = "H1: gelernte Clifford-Schaltung schlaegt Zufall und Hadamard"
    elif sig_z or sig_h:
        u["C-2"] = "HU: schlaegt nur " + ("Zufall" if sig_z else "Hadamard")
    else:
        u["C-2"] = "H0: gelernte Rundung nicht besser als feste Clifford-Mischung"
    u["C-3"] = f"M* = {a['C3_Mstern']} von {LAGEN * N_QUBITS} Winkeln"
    return u


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: jeden Ausgang auf synthetischen Zeilen erzwingen."""
    rng = np.random.default_rng(0)

    def zeilen(voll, k0, zuf, had, ohne, rausch=0.004):
        out = []
        for s in range(10):
            z = {"seed": s}
            for name, mu in [("voll", voll), ("zufall_clifford", zuf), ("hadamard_fest", had),
                             ("ohne_schaltung", ohne), ("voll_z", 0.5), ("dicht_linear", 0.95)]:
                z[name] = mu + rng.normal(0, rausch)
            for k in MAGIE_STUFEN:
                z[f"clifford_k{k}"] = k0 + (voll - k0) * min(1, k / 4) + rng.normal(0, rausch)
            z["abstand_zum_gitter"] = 0.3
            out.append(z)
        return out

    faelle = [
        ((0.95, 0.94, 0.90, 0.90, 0.90), {"C-1": "H1", "C-2": "H1", "C-0": "H1"}),
        ((0.95, 0.85, 0.80, 0.90, 0.90), {"C-1": "H0", "C-2": "HU"}),
        ((0.95, 0.94, 0.94, 0.94, 0.96), {"C-1": "H1", "C-2": "H0", "C-0": "H0"}),
    ]
    ok = True
    for (werte, erwartet) in faelle:
        for _ in range(ziehungen):
            u = urteil(auswertung(zeilen(*werte)))
            for k, marke in erwartet.items():
                if not u[k].startswith(marke):
                    ok = False
                    print("  VORTEST FEHLER", werte, k, u[k])
    u = urteil(auswertung(zeilen(0.95, 0.94, 0.9, 0.9, 0.9, rausch=0.08)))
    if not u["C-1"].startswith("HU"):
        ok = False
        print("  VORTEST FEHLER: Aufloesungspruefung feuert nicht", u["C-1"])
    return ok


if __name__ == "__main__":
    import sys
    print("Vortest:", "bestanden" if vortest() else "DURCHGEFALLEN")
    e = lauf(schnell="--schnell" in sys.argv)
    print(speichere("C-clifford" + ("-schnell" if "--schnell" in sys.argv else ""), e))
    print(e["urteil"])
