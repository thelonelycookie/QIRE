from __future__ import annotations

"""RESEARCH ID: Q-019W -- Walsh statt Fourier, und ueber die Datenmenge gemessen.

Spezifikation: ENTWURF_Q019_WALSH.md (Stand 28.09.2026, vor jeder Zahl).
Ersetzt den Lauf aus `zeitreihen.py`; jenes Modul bleibt unveraendert stehen
und liefert hier nur Bausteine (Butterfly-Arme, dichte Arme, Auslese).

DIE IDEE
Ohne CNOT ist die geteilte Butterfly ein Kronecker-Produkt R1 x ... x R8 aus
2x2-Drehungen. Mit Ry(pi/2) in jeder Lage ist das bis auf Vorzeichen die
Walsh-Hadamard-Transformation H^8 (`pruefe_theorie` rechnet es nach). Fuer
Rechteck-/Schaltsignale ist die Struktur also die richtige Basis. Die Frage
ist nicht, ob sie die Loesung DARSTELLEN kann (bekannt), sondern ob 40 Winkel
sie aus WENIG DATEN lernen, wo 65.536 freie Gewichte es nicht koennen.

DIE AUFGABE
d = 256. x = sum_k a_k w_k + Rauschen, ANTEILE Basisfunktionen mit Amplitude
0,3..1,0, zufaelligem Vorzeichen und Mindestvorsprung des staerksten Anteils.
Klasse = Sequency-Band (8 Baender a 32) des staerksten Anteils.
Basen: walsh (Hauptbedingung) | zufall (dieselben Koeffizienten in einer
festen Haar-zufaelligen Orthobasis Q*W -- Kontrolle) | permutiert (Walsh mit
fester Zeitachsen-Vertauschung, beschreibend) | fourier (Sinus wie Q-019,
nur N = 6.000, beschreibend).
Lernkurve: N in {50, 200, 1.000, 6.000}, klassenbalanciert; Test fest 2.000
(je Klasse 250) je Seed. Festes Budget SCHRITTE = 470 Gradientenschritte,
Batch 128 (bei N < 128 voller Batch), fuer ALLE N gleich. Kein Early Stopping.

DIE ARME (Mittelschicht; normieren -> Mitte -> <Z_q> (8) -> linear 8 -> 8)
  butterfly  geteilte Butterfly ohne CNOT, 5 Lagen            40   Kandidat
  dense      freie 256x256-Matrix                          65.536   Referenz
  lowrank    I + U V^T, Rang 1                                512   Kontrolle
             (40 sind mit Rang 1 nicht erreichbar, 512 ist das Minimum)
  wht_fest   feste H^8, natuerliche Hadamard-Ordnung            0   Anker
  untied     ungeteilte Butterfly                           5.120   beschreibend
  narrow     Engstelle 256 -> 64 -> 256                    32.768   Aufloesung, N = 6.000

VORREGISTRIERUNG (vor jeder Q-019W-Zahl; Aenderungen erst ins Tagebuch)
  Messgroesse balancierte Genauigkeit, 15 Seeds (19101-19115), Tuning-Seeds
  19151-19153. Lernrate je Arm x Basis x N, VOLLES Budget, Gitter
  0,001..0,03, Verlaengerung in BEIDE Richtungen bis 3-mal (Regel 1 a-c);
  "innen" erst, wenn hinter dem Optimum ein schlechterer Punkt gemessen ist.
  Familie bei N = 200, Wilcoxon ZWEISEITIG + TOST +-0,005, Holm ueber GENAU:
    PRIMAER  walsh: butterfly - dense
             STRUKTUR_BESSER / GLEICHWERTIG / DICHT_BESSER / UNENTSCHIEDEN
    P2       (butterfly - dense)_walsh - (butterfly - dense)_zufall, seedweise
             PASSUNG_TRAEGT / GLEICHWERTIG / PASSUNG_SCHADET / UNENTSCHIEDEN
    P3       walsh: butterfly - lowrank
             STRUKTUR_TRAEGT / GLEICHWERTIG / LOWRANK_BESSER / UNENTSCHIEDEN
  Rangfolge der Zweige: HU_UNVOLLSTAENDIG -> HU_AUFLOESUNG -> HU_BODEN ->
  HU_SAETTIGUNG -> TOST -> Holm -> Richtung -> UNENTSCHIEDEN.
  Kontrollen (urteilstragend, fuer alle drei Tests gemeinsam):
    Aufloesung  dense trennt sich von narrow, walsh, N = 6.000 (Wilcoxon
                zweiseitig p < 0,05; TRENNUNG, nicht Richtung)
    Boden       in jeder vorhandenen Familienzelle (walsh/zufall, N = 200)
                bester Arm >= 0,125 + 0,10
    Saettigung  kein Familienarm (butterfly, dense, lowrank) in einer
                Familienzelle ueber 0,95 im Mittel
  Erreichbarkeit (Regel 2): kleinstes p bei 15 Seeds 6,1e-5, nach Holm x3
  1,8e-4. TOST +-0,005 nur bei gepaarter sd < 0,011 -- bei N = 200
  vermutlich groesser; `macht` misst das VOR der Freigabe. Liegt sie darueber,
  kann GLEICHWERTIG nicht feuern.
  Erwartung vorher: PRIMAER STRUKTUR_BESSER, P2 PASSUNG_TRAEGT, P3
  STRUKTUR_TRAEGT; N* zwischen 1.000 und 6.000, dense bei 6.000 vorn.
  Scheitert PRIMAER schon bei N = 200, traegt das Dateneffizienz-Argument
  nicht -- ein klarer Negativbefund.
  Beschreibend, ohne Holm: Lernkurven aller Arme (walsh, zufall), Uebergangs-
  punkt N* (kleinstes gemessenes N, ab dem dense >= butterfly bleibt; linear
  in log N zwischen den Stuetzstellen interpoliert, als Bereich berichtet),
  butterfly gegen wht_fest, untied, permutiert, fourier, dense walsh gegen
  dense zufall (Plausibilitaet: muss ~0 sein).

WAS AUS DER VORLAGENPRUEFUNG FOLGT (CLAUDE.md §4, letzter Absatz)
  * `zeitreihen.sweep_lr` haengt an globalen Armen und `_zelle`; hier wird die
    parametrisierte Fassung aus `kernaussage` benutzt (dort geprueft).
  * `zeitreihen` testet EINSEITIG und hat H0 VOR H1 nur fuer die Wechsel-
    wirkung; hier zweiseitig + TOST fuer alle drei, wie der Entwurf verlangt.
  * `zeitreihen` sichert Rohdaten je Bedingung, aber ohne Wiederaufnahme;
    hier eigene Datei je Lauf-ID in eigenem Unterordner, nie `latest()`,
    Fingerabdruck der Konfiguration.
  * `zeitreihen._kunst` gab beiden Metrikfeldern denselben Wert -- ein Fehler
    im Metrikfeld waere unsichtbar gewesen. Hier: UMGEKEHRTE Versaetze und
    eigenes Rauschen im Feld `accuracy`.
  * `kernaussage` hat keine Mutationspruefung; sie ist hier neu.
  * ACHTUNG, Messgroesse: Die Testmenge ist exakt klassenbalanciert. Dann ist
    balancierte Genauigkeit == Genauigkeit; das Feld zaehlt nur im Vortest.

Run (aus dem Ordner A):
    python -m quantum_sim.walsh vortest           # Pflicht, ~1 min
    python -m quantum_sim.walsh saettigung        # vor dem Lauf
    python -m quantum_sim.walsh macht             # Regel 2(b), vor dem Lauf
    python -m quantum_sim.walsh zeit              # Zeitmessung + Hochrechnung
    python -m quantum_sim.walsh lauf              # der Messlauf (Malte entscheidet)
    python -m quantum_sim.walsh auswertung [ID|Pfad]
"""

import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import RESULTS_DIR, Recorder

from .kernaussage import sweep_lr
from .zeitreihen import (DIM, KLASSEN, LAGEN, QUBITS, DichteDrehung, OrthoButterfly,
                         SchmaleSchicht, UngeteilteButterfly, auslese)
from .zeitreihen import signale as fourier_signale

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt (Regel 7)
# ---------------------------------------------------------------------------
LAUF_ID = "Q-019W"
BASEN = ("walsh", "zufall", "permutiert", "fourier")
N_WERTE = (50, 200, 1000, 6000)
N_FAMILIE, N_AUFLOESUNG = 200, 6000
N_TEST = 2000
SCHRITTE, BATCH = 470, 128
ARME_KURVE = ("butterfly", "dense", "lowrank", "wht_fest", "untied")
FAMILIENARME = ("butterfly", "dense", "lowrank")
FAMILIE = ("PRIMAER", "P2", "P3")
BAND = DIM // KLASSEN                                  # 32 Sequency-Werte je Band

# Die Zellen des Laufs, in der Reihenfolge der Wichtigkeit (Familie zuerst,
# dann die Aufloesungszelle, dann der Rest der Lernkurve).
PLAN: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("walsh", 200, ARME_KURVE),
    ("zufall", 200, ARME_KURVE),
    ("walsh", 6000, ARME_KURVE + ("narrow",)),
    ("zufall", 6000, ARME_KURVE),
    ("walsh", 1000, ARME_KURVE), ("zufall", 1000, ARME_KURVE),
    ("walsh", 50, ARME_KURVE), ("zufall", 50, ARME_KURVE),
    ("fourier", 6000, ("butterfly", "dense")),
    ("permutiert", 6000, ("butterfly", "dense")),
)

EVAL_SEEDS = tuple(range(19101, 19116))
TUNING_SEEDS = (19151, 19152, 19153)
LR_GITTER = (0.001, 0.003, 0.01, 0.03)
ERWEITERUNGEN = 3

# Schwierigkeit -- darf nur VOR dem Lauf geaendert werden (Tagebuch zuerst).
ANTEILE = 3
RAUSCHEN = 0.45
AMPLITUDENABSTAND = 0.15
SEED_ZUFALLSBASIS = 20260928
SEED_PERMUTATION = 20260929

BODEN = 1.0 / KLASSEN + 0.10
SAETTIGUNG = 0.95
METRIK = "balanced_accuracy"


# ---------------------------------------------------------------------------
# Basen
# ---------------------------------------------------------------------------
def _bitumkehr(i: int, n: int = QUBITS) -> int:
    return int(format(i, f"0{n}b")[::-1], 2)


def hadamard() -> torch.Tensor:
    """H^8 in natuerlicher (Sylvester-)Ordnung, Eintraege +-1: H[i,j] = (-1)^|i&j|.
    Qubit 0 = hoechstwertiges Bit, wie `auslese`."""
    i = torch.arange(DIM)
    und = i[:, None] & i[None, :]
    bits = sum(((und >> b) & 1) for b in range(QUBITS))
    return (1 - 2 * (bits % 2)).double()


# Hadamard-Index der k-ten Walsh-Funktion (Sequency k): Gray-Code, dann Bitumkehr.
H_VON_SEQ = torch.tensor([_bitumkehr(k ^ (k >> 1)) for k in range(DIM)])
SEQ_VON_H = torch.empty(DIM, dtype=torch.long)
SEQ_VON_H[H_VON_SEQ] = torch.arange(DIM)


def walsh_basis() -> torch.Tensor:
    """Zeile k = Walsh-Funktion mit Sequency k (k Nulldurchgaenge), +-1."""
    return hadamard()[H_VON_SEQ]


def zufalls_basis() -> torch.Tensor:
    """Zeile k = Q * w_k, Q fest Haar-zufaellig orthogonal (ein Seed fuer alle
    Laeufe). Dieselbe Sparsamkeit, dieselben Etiketten, keine Kronecker-Struktur."""
    g = torch.Generator().manual_seed(SEED_ZUFALLSBASIS)
    q, r = torch.linalg.qr(torch.randn(DIM, DIM, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))              # Haar-Mass (Vorzeichen fixieren)
    return walsh_basis() @ q.T


_BASIS_CACHE: Dict[str, torch.Tensor] = {}


def _basis(name: str) -> torch.Tensor:
    if name not in _BASIS_CACHE:
        _BASIS_CACHE[name] = (zufalls_basis() if name == "zufall" else walsh_basis()).float()
    return _BASIS_CACHE[name]


def _permutation() -> torch.Tensor:
    return torch.randperm(DIM, generator=torch.Generator().manual_seed(SEED_PERMUTATION))


# ---------------------------------------------------------------------------
# Daten -- klassenbalanciert
# ---------------------------------------------------------------------------
def signale(seed: int, n: int, basis: str, anteile: int = ANTEILE,
            rauschen: float = RAUSCHEN, abstand: float = AMPLITUDENABSTAND
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """n Signale, exakt klassenbalanciert (Klasse c = i mod 8, dann gemischt).

    Der staerkste Anteil liegt im Band der Klasse; die uebrigen Indizes sind
    paarweise verschieden und verschieden vom staerksten -- sonst koennten sich
    zwei schwache Anteile zu einem staerkeren addieren und das Etikett waere
    falsch. Amplituden 0,3..1,0 mit Mindestvorsprung `abstand` (Verwerfen)."""
    if basis not in BASEN:
        raise ValueError(f"unbekannte Basis: {basis}")
    if basis == "fourier":
        return _fourier_balanciert(seed, n, anteile, rauschen, abstand)
    g = torch.Generator().manual_seed(seed)
    y = (torch.arange(n) % KLASSEN)[torch.randperm(n, generator=g)]
    amp = torch.empty(n, anteile)
    offen = torch.ones(n, dtype=torch.bool)
    while offen.any():
        m = int(offen.sum())
        a = (0.3 + 0.7 * torch.rand(m, anteile, generator=g)).sort(dim=1, descending=True)[0]
        amp[offen] = a
        if anteile > 1:
            neu = offen.clone()
            neu[offen] = (a[:, 0] - a[:, 1]) < abstand
            offen = neu
        else:
            offen[:] = False
    idx = [y * BAND + torch.randint(0, BAND, (n,), generator=g)]
    for j in range(1, anteile):
        r = torch.randint(0, DIM - j, (n,), generator=g)
        for e in torch.stack(idx, 1).sort(dim=1)[0].T:      # an Belegten vorbeischieben
            r = r + (r >= e).long()
        idx.append(r)
    idx = torch.stack(idx, 1)
    vz = torch.randint(0, 2, (n, anteile), generator=g).float() * 2 - 1
    x = ((amp * vz)[..., None] * _basis(basis)[idx]).sum(1)
    x = x + torch.randn(x.shape, generator=g) * rauschen
    if basis == "permutiert":
        x = x[:, _permutation()]
    return x, y


def _fourier_balanciert(seed, n, anteile, rauschen, abstand):
    """Q-019-Signale (`zeitreihen.signale`, unveraendert), balanciert gezogen."""
    je = [n // KLASSEN + (c < n % KLASSEN) for c in range(KLASSEN)]
    m = 2 * n + 2000
    while True:
        x, y = fourier_signale(seed, m, "periodisch", anteile, rauschen, abstand)
        wahl = [torch.nonzero(y == c).flatten()[:je[c]] for c in range(KLASSEN)]
        if all(len(w) == je[c] for c, w in enumerate(wahl)):
            idx = torch.cat(wahl)
            idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))]
            return x[idx], y[idx]
        m *= 2


# ---------------------------------------------------------------------------
# Die neuen Mittelschichten (die uebrigen kommen unveraendert aus zeitreihen)
# ---------------------------------------------------------------------------
class FesteWHT(nn.Module):
    """H^8 / 16, natuerliche Hadamard-Ordnung, untrainiert. Genau die Matrix, die
    `butterfly` mit Ry(pi/2) ueberall darstellt (bis auf Vorzeichen)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("h", (hadamard() / math.sqrt(DIM)).float())

    def forward(self, x):
        return x @ self.h.T


class NiedrigRang(nn.Module):
    """I + U V^T, Rang 1: 512 Parameter, parameterarm ohne Struktur."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim, 1) / math.sqrt(dim))
        self.v = nn.Parameter(torch.randn(dim, 1) / math.sqrt(dim))

    def forward(self, x):
        return x + (x @ self.v) @ self.u.T


MITTE = {"butterfly": OrthoButterfly, "untied": UngeteilteButterfly,
         "dense": DichteDrehung, "narrow": SchmaleSchicht,
         "wht_fest": FesteWHT, "lowrank": NiedrigRang}


class WalshModell(nn.Module):
    """normieren -> Mittelschicht -> <Z_q> -> linear 8 -> 8. Kein Encoder."""

    def __init__(self, arm: str):
        super().__init__()
        self.mitte = MITTE[arm]()
        self.kopf = nn.Linear(QUBITS, KLASSEN)

    def forward(self, x):
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.kopf(auslese(self.mitte(x)))


# ---------------------------------------------------------------------------
# Eine Zelle -- festes Schrittbudget
# ---------------------------------------------------------------------------
def zelle(basis: str, n: int, arm: str, seed: int, lr: float, threads: int = 6,
          schritte: int = SCHRITTE) -> Dict[str, object]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    xtr, ytr = signale(seed, n, basis)
    xte, yte = signale(seed + 500_000, N_TEST, basis)
    modell = WalshModell(arm)
    opt = torch.optim.Adam(modell.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    b, schritt = min(BATCH, n), 0
    while schritt < schritte:                 # Epochen, bis das Budget verbraucht ist
        ordnung = torch.randperm(n, generator=g)
        for s in range(0, n, b):
            if schritt >= schritte:
                break
            idx = ordnung[s:s + b]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(modell(xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 5.0)
            opt.step()
            schritt += 1
    modell.eval()
    with torch.no_grad():
        pred = modell(xte).argmax(-1)
    bal = float(np.mean([float((pred[yte == c] == c).float().mean()) for c in torch.unique(yte)]))
    return {"basis": basis, "n_train": n, "arm": arm, "seed": seed, "lr": lr,
            "balanced_accuracy": bal, "accuracy": float((pred == yte).float().mean()),
            "middle_parameters": sum(p.numel() for p in modell.mitte.parameters()),
            "total_parameters": sum(p.numel() for p in modell.parameters()),
            "schritte": schritt, "sekunden": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# Wiederaufnahme -- eigene Datei je Lauf-ID, eigener Unterordner, nie latest()
# ---------------------------------------------------------------------------
def _schl(*teile) -> str:
    return "|".join(str(t) for t in teile)


class Zustand:
    """Tuning-Zellen, Lernraten und Messzeilen, nach JEDER Zelle atomar
    geschrieben. Angenommen nur bei passendem Konfig-Fingerabdruck."""

    def __init__(self, lauf_id: str, konfig: Dict[str, object]):
        self.pfad = RESULTS_DIR / "q019w_zwischenstand" / f"{lauf_id}_wiederaufnahme.json"
        self.fingerabdruck = hashlib.sha256(
            json.dumps(konfig, sort_keys=True).encode()).hexdigest()[:16]
        self.tuning: Dict[str, float] = {}
        self.lr: Dict[str, Dict[str, object]] = {}
        self.records: List[Dict[str, object]] = []
        if self.pfad.exists():
            d = json.loads(self.pfad.read_text(encoding="utf-8"))
            if d.get("fingerabdruck") != self.fingerabdruck:
                raise SystemExit(f"{self.pfad.name} gehoert zu einer ANDEREN Konfiguration. "
                                 "Nicht ueberschreiben -- umbenennen oder Konfiguration pruefen.")
            self.tuning, self.lr, self.records = d["tuning"], d["lr"], d["records"]
            print(f"  WIEDERAUFNAHME: {len(self.tuning)} Tuning-Zellen, "
                  f"{len(self.records)} Messzeilen aus {self.pfad.name}")

    def speichern(self) -> None:
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerabdruck": self.fingerabdruck, "tuning": self.tuning,
                                   "lr": self.lr, "records": self.records}), encoding="utf-8")
        os.replace(tmp, self.pfad)

    def gemessen(self, basis: str, n: int, arm: str, seed: int) -> bool:
        return any(r["basis"] == basis and r["n_train"] == n and r["arm"] == arm
                   and r["seed"] == seed for r in self.records)


# ---------------------------------------------------------------------------
# Der Lauf
# ---------------------------------------------------------------------------
def konfiguration() -> Dict[str, object]:
    return {"plan": [[b, n, list(a)] for b, n, a in PLAN], "seeds": list(EVAL_SEEDS),
            "tuning_seeds": list(TUNING_SEEDS), "lr_gitter": list(LR_GITTER),
            "erweiterungen": ERWEITERUNGEN, "schritte": SCHRITTE, "batch": BATCH,
            "n_test": N_TEST, "anteile": ANTEILE, "rauschen": RAUSCHEN,
            "amplitudenabstand": AMPLITUDENABSTAND, "lagen": LAGEN,
            "seed_zufallsbasis": SEED_ZUFALLSBASIS, "seed_permutation": SEED_PERMUTATION}


def run(lauf_id: str = LAUF_ID, threads: int = 6) -> Recorder:
    konfig = konfiguration()
    z = Zustand(lauf_id, konfig)
    rec = Recorder(
        experiment_id=lauf_id,
        description=("Q-019W: geteilte Butterfly gegen dicht, Low-Rank und feste WHT auf "
                     "Walsh-Signalen ueber die Datenmenge; Familie bei N=200 "
                     "(PRIMAER b-d, P2 Passung walsh-zufall, P3 b-lowrank)."),
        config={**konfig, "margin": MARGIN, "test_metric": METRIK, "familie": list(FAMILIE),
                "n_familie": N_FAMILIE, "boden": BODEN, "saettigung": SAETTIGUNG,
                "aufloesungspruefung": "dense trennt sich von narrow, walsh, N=6000",
                "torch_threads": threads, "fingerabdruck": z.fingerabdruck})
    for basis, n, arme in PLAN:
        print(f"\n  {basis} N={n} -- Lernratensuche (volles Budget {SCHRITTE} Schritte)", flush=True)
        for arm in arme:
            k_lr = _schl(basis, n, arm)
            if k_lr in z.lr:
                continue

            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = _schl(basis, n, _arm, seed, f"{lr:.6g}")
                if k not in z.tuning:
                    z.tuning[k] = zelle(basis, n, _arm, seed, lr, threads)[METRIK]
                    z.speichern()
                return z.tuning[k]

            best, vermerk, mittel = sweep_lr(basis, arm, LR_GITTER, TUNING_SEEDS,
                                             ERWEITERUNGEN, messe)
            z.lr[k_lr] = {"lr": best, "rand": vermerk}
            z.speichern()
            print(f"    {arm:<10} -> lr {best:g}"
                  + (f"  ACHTUNG Rand {vermerk}" if vermerk != "innen" else "")
                  + "   " + "  ".join(f"{k:g}:{mittel[k]:.4f}" for k in sorted(mittel)), flush=True)
        for i, seed in enumerate(EVAL_SEEDS, 1):
            for arm in arme:
                if not z.gemessen(basis, n, arm, seed):
                    z.records.append(zelle(basis, n, arm, seed,
                                           float(z.lr[_schl(basis, n, arm)]["lr"]), threads))
                    z.speichern()
            if i % 5 == 0:
                print(f"    seed {seed} fertig ({i}/{len(EVAL_SEEDS)})", flush=True)
        rec.records, rec.config["lr"] = list(z.records), dict(z.lr)
        print("    Zwischenstand gesichert (Regel 5):")
        rec.save()
    return rec


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _reihe(records, basis: str, n: int, arm: str, feld: str) -> Dict[int, float]:
    return {int(r["seed"]): float(r[feld]) for r in records
            if r["basis"] == basis and int(r["n_train"]) == n and r["arm"] == arm}


def _wilcoxon(d: np.ndarray) -> float:
    from scipy import stats
    try:
        p = float(stats.wilcoxon(d).pvalue)                  # zweiseitig
    except ValueError:                                       # alle Differenzen null
        return 1.0
    return 1.0 if math.isnan(p) else p


def _vergleich(a: Dict[int, float], b: Dict[int, float]) -> Optional[Dict[str, object]]:
    if len(a) < 2 or set(a) != set(b):
        return None
    s = sorted(a)
    va, vb = np.array([a[k] for k in s]), np.array([b[k] for k in s])
    eq = tost_paired(va, vb, MARGIN)
    return {"delta": float(va.mean() - vb.mean()), "p": _wilcoxon(va - vb),
            "p_tost": eq["p_tost"], "erreicht": eq["achieved"],
            "siege": int((va > vb).sum()), "n": len(s),
            "mittel_a": float(va.mean()), "mittel_b": float(vb.mean())}


def _differenz(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
    return {k: a[k] - b[k] for k in a} if a and set(a) == set(b) else {}


def _familie(records, feld: str) -> Dict[str, Optional[Dict[str, object]]]:
    w = {a: _reihe(records, "walsh", N_FAMILIE, a, feld) for a in FAMILIENARME}
    z = {a: _reihe(records, "zufall", N_FAMILIE, a, feld) for a in FAMILIENARME}
    return {"PRIMAER": _vergleich(w["butterfly"], w["dense"]),
            "P2": _vergleich(_differenz(w["butterfly"], w["dense"]),
                             _differenz(z["butterfly"], z["dense"])),
            "P3": _vergleich(w["butterfly"], w["lowrank"])}


def _kontrollen(records, feld: str, richtung: bool = False) -> Dict[str, object]:
    """`richtung=True` ist die ABSICHTLICH FALSCHE Fassung fuer die Mutations-
    pruefung: die Aufloesung verlangt dann dense > narrow statt Trennung."""
    r = _vergleich(_reihe(records, "walsh", N_AUFLOESUNG, "dense", feld),
                   _reihe(records, "walsh", N_AUFLOESUNG, "narrow", feld))
    out: Dict[str, object] = {"aufloesung": {
        "bestanden": bool(r and r["p"] < 0.05 and (r["delta"] > 0 or not richtung)),
        **(r or {"fehlt": True})}}
    beste, satt = {}, []
    for basis in ("walsh", "zufall"):          # nur VORHANDENE Familienzellen
        m = {a: float(np.mean(list(v.values()))) for a in ARME_KURVE
             if (v := _reihe(records, basis, N_FAMILIE, a, feld))}
        if m:
            beste[basis] = max(m.values())
            satt += [f"{basis}/{a} {m[a]:.4f}" for a in FAMILIENARME if m.get(a, 0) > SAETTIGUNG]
    out["boden"] = {"bestanden": all(v >= BODEN for v in beste.values()), "bester": beste}
    out["saettigung"] = {"gesaettigt": satt}
    return out


_TEXTE = {
    "PRIMAER": ("STRUKTUR_BESSER", "DICHT_BESSER",
                "die geteilte Butterfly (40 Winkel) schlaegt die dichte Drehung (65.536) bei "
                "N = 200 -- Dateneffizienz belegt",
                "die dichte Drehung schlaegt die Butterfly schon bei N = 200; das "
                "Dateneffizienz-Argument traegt nicht -- klarer Negativbefund"),
    "P2": ("PASSUNG_TRAEGT", "PASSUNG_SCHADET",
           "der Vorsprung (butterfly - dense) ist in der Walsh-Basis groesser als in der "
           "zufaelligen Orthobasis -- er kommt aus der Passung, nicht nur aus der Parameterarmut",
           "der Vorsprung ist in der ZUFAELLIGEN Basis groesser -- Gegenrichtung der "
           "Hypothese, eigenes Ergebnis"),
    "P3": ("STRUKTUR_TRAEGT", "LOWRANK_BESSER",
           "die Butterfly schlaegt den Low-Rank-Arm (512) -- diese Struktur, nicht "
           "irgendeine Parameterarmut",
           "der Low-Rank-Arm ist besser -- Parameterarmut genuegt, die Struktur traegt nicht"),
}
_VERGLEICH = {"PRIMAER": "walsh: butterfly - dense",
              "P2": "(butterfly - dense) walsh - (butterfly - dense) zufall",
              "P3": "walsh: butterfly - lowrank"}


def _urteil(tag: str, v: Optional[Dict[str, object]], p_holm: float, k: Dict[str, object],
            tost_zuerst: bool = True) -> str:
    """Rangfolge vor dem Lauf festgelegt (Entwurf §5). `tost_zuerst=False` ist
    die ABSICHTLICH FALSCHE Fassung fuer die Mutationspruefung."""
    code_a, code_b, satz_a, satz_b = _TEXTE[tag]
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
    if not k["aufloesung"]["bestanden"]:
        return (f"HU_AUFLOESUNG -- {tag}: dense trennt sich bei N = {N_AUFLOESUNG} nicht von "
                f"narrow" + (" (Zelle fehlt)" if k["aufloesung"].get("fehlt") else "")
                + ". Der Aufbau ist fuer diese Frage blind; KEIN Nullbefund.")
    if not k["boden"]["bestanden"]:
        return (f"HU_BODEN -- {tag}: bester Arm unter {BODEN:.3f} in einer Familienzelle "
                f"{k['boden']['bester']}; die Zelle misst Trainingsversagen, nicht Architektur.")
    if k["saettigung"]["gesaettigt"]:
        return (f"HU_SAETTIGUNG -- {tag}: Familienarm ueber {SAETTIGUNG} "
                f"({', '.join(k['saettigung']['gesaettigt'])}); die Aufgabe ist gesaettigt "
                f"und trennt nicht mehr.")
    z = (f"{_VERGLEICH[tag]}: {v['delta']:+.4f}, {v['siege']}/{v['n']} Seeds, "
         f"p={v['p']:.4f}, Holm={p_holm:.4f}, TOST p={v['p_tost']:.4f}")
    gleich = (f"GLEICHWERTIG -- {tag}: gleichwertig innerhalb +-{MARGIN} (erreicht "
              f"+-{v['erreicht']:.4f}); {z}. Positiver Gleichwertigkeitsnachweis, nicht "
              f"blosses p>0,05." + (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber "
                                   "kleiner als die Erheblichkeitsgrenze; massgeblich ist "
                                   "die Grenze." if p_holm < 0.05 else ""))
    code, satz = (code_a, satz_a) if v["delta"] > 0 else (code_b, satz_b)
    signifikant = f"{code} -- {tag}: {satz}; {z}."
    reihenfolge = ((v["p_tost"] < 0.05, gleich), (p_holm < 0.05, signifikant))
    for bedingung, text in (reihenfolge if tost_zuerst else reihenfolge[::-1]):
        if bedingung:
            return text
    return (f"UNENTSCHIEDEN -- {tag}: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
            f"die Daten tragen erst ab +-{v['erreicht']:.4f}; {z}.")


def uebergang(ns: Sequence[int], delta: Sequence[float]) -> Dict[str, object]:
    """N* = kleinstes gemessenes N, ab dem dense >= butterfly BLEIBT (delta =
    butterfly - dense <= 0 dort und bei allen groesseren N). Bereich = das
    Intervall davor; Schaetzung linear in log N zwischen den Stuetzstellen
    (das Gitter 50..6.000 ist logarithmisch; festgelegt am 28.09., Tagebuch)."""
    for i in range(len(ns)):
        if all(d <= 0 for d in delta[i:]):
            if i == 0:
                return {"bereich": f"<= {ns[0]}", "n_stern": None}
            n0, n1, d0, d1 = ns[i - 1], ns[i], delta[i - 1], delta[i]
            anteil = d0 / (d0 - d1)
            return {"bereich": f"{n0}..{n1}", "gemessen": n1,
                    "n_stern": float(math.exp(math.log(n0)
                                              + anteil * (math.log(n1) - math.log(n0))))}
    return {"bereich": f"> {ns[-1]}", "n_stern": None}


def _lesart(v: Dict[str, object]) -> str:
    if v["p_tost"] < 0.05:
        return f"gleichwertig (+-{MARGIN})"
    if v["p"] < 0.05:
        return "Unterschied (a besser)" if v["delta"] > 0 else "Unterschied (b besser)"
    return f"unentschieden (+-{v['erreicht']:.4f})"


def _beschreibend(records, feld: str, say) -> Dict[str, object]:
    out: Dict[str, object] = {"kurven": {}, "uebergang": {}, "vergleiche": []}
    say("\nLERNKURVEN (beschreibend) -- Mittel der balancierten Genauigkeit")
    say(f"  {'Basis':<11}{'Arm':<11}" + "".join(f"{'N=' + str(n):>10}" for n in N_WERTE))
    for basis in BASEN:
        for arm in ARME_KURVE + ("narrow",):
            m = {n: float(np.mean(list(v.values()))) for n in N_WERTE
                 if (v := _reihe(records, basis, n, arm, feld))}
            if m:
                out["kurven"][f"{basis}|{arm}"] = m
                say(f"  {basis:<11}{arm:<11}" + "".join(
                    f"{m[n]:>10.4f}" if n in m else f"{'-':>10}" for n in N_WERTE))
    for basis in ("walsh", "zufall"):
        kb, kd = out["kurven"].get(f"{basis}|butterfly", {}), out["kurven"].get(f"{basis}|dense", {})
        ns = [n for n in N_WERTE if n in kb and n in kd]
        if ns:
            u = uebergang(ns, [kb[n] - kd[n] for n in ns])
            out["uebergang"][basis] = u
            say(f"  Uebergangspunkt N* ({basis}): Bereich {u['bereich']}"
                + (f", linear interpoliert ~{u['n_stern']:.0f}" if u["n_stern"] else ""))
    paare = [(b, n, "butterfly", b, n, "wht_fest", "erreicht das Lernen die feste WHT?")
             for b in ("walsh", "zufall") for n in N_WERTE]
    paare += [("walsh", n, "untied", "walsh", n, "butterfly", "Preis der Gewichtsteilung")
              for n in N_WERTE]
    paare += [("walsh", n, "dense", "zufall", n, "dense", "Plausibilitaet: dense sieht keine Basis")
              for n in N_WERTE]
    paare += [(b, 6000, "butterfly", b, 6000, "dense", f"Anschluss Q-019 ({b})")
              for b in ("fourier", "permutiert", "walsh")]
    say("\nBESCHREIBEND (ohne Holm, kein Urteil haengt daran)")
    for ba, na, a, bb, nb, b, was in paare:
        v = _vergleich(_reihe(records, ba, na, a, feld), _reihe(records, bb, nb, b, feld))
        if v:
            v.update(a=f"{ba}/{a}", b=f"{bb}/{b}", n_train=na, was=was, lesart=_lesart(v))
            out["vergleiche"].append(v)
            say(f"  N={na:<5} {v['a']:>17} - {v['b']:<17} {v['delta']:+.4f}  p={v['p']:.4f}  "
                f"TOST p={v['p_tost']:.4f}  {v['lesart']:<26} {was}")
    return out


def analyse(rec_or_id, still: bool = False, mut: Optional[Dict[str, object]] = None
            ) -> Dict[str, object]:
    """`mut` nur fuer die Mutationspruefung des Vortests -- im Lauf immer None."""
    def say(*a):
        if not still:
            print(*a)

    mut = mut or {}
    feld = str(mut.get("feld", METRIK))
    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    say(f"\nRESEARCH ID {LAUF_ID} | Walsh-Signale, Familie bei N = {N_FAMILIE}")
    fam = _familie(records, feld)
    k = _kontrollen(records, feld, bool(mut.get("richtung", False)))
    p_holm = dict(zip(FAMILIE, holm([fam[t]["p"] if fam[t] else 1.0 for t in FAMILIE])))
    out: Dict[str, object] = {"kontrollen": k}
    say("=" * 96)
    say(f"VORREGISTRIERTE FAMILIE (Holm ueber {len(FAMILIE)}), Grenze +-{MARGIN}")
    for t in FAMILIE:
        u = _urteil(t, fam[t], p_holm[t], k, bool(mut.get("tost_zuerst", True)))
        out[t] = {**(fam[t] or {}), "p_holm": p_holm[t], "verdict": u}
        say(f"  [{t}] {u}")
    a = k["aufloesung"]
    say(f"KONTROLLEN  Aufloesung {'bestanden' if a['bestanden'] else 'NICHT BESTANDEN'}"
        + (f" (dense-narrow {a['delta']:+.4f}, p={a['p']:.4f})" if "delta" in a else " (fehlt)")
        + f" | Boden {k['boden']} | Saettigung {k['saettigung']['gesaettigt'] or 'keine'}")
    if not still:
        out["beschreibend"] = _beschreibend(records, feld, say)
    raender = {s: v["rand"] for s, v in config.get("lr", {}).items() if v.get("rand") != "innen"}
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest (CLAUDE.md §5) -- Teil 1: Theorie, ohne Training
# ---------------------------------------------------------------------------
def _signtreu(m: torch.Tensor, h: torch.Tensor) -> float:
    """Abstand von m zu h bis auf Vorzeichen: min ueber H m, m H von
    max|P - D| mit D = diag(+-1). 0 heisst m = D h oder m = h D."""
    best = float("inf")
    for p in (h @ m, m @ h):
        d = torch.diagonal(p)
        best = min(best, float((p - torch.diag(torch.sign(d))).abs().max()))
    return best


def _linear_ablesbar(merkmale: torch.Tensor, y: torch.Tensor, schritte: int = 3000) -> float:
    """Trainingsgenauigkeit eines linearen Softmax-Kopfs 8 -> 8 (volle Charge)."""
    torch.manual_seed(0)
    kopf = nn.Linear(merkmale.shape[1], KLASSEN)
    opt = torch.optim.Adam(kopf.parameters(), lr=0.1)
    for _ in range(schritte):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(kopf(merkmale), y).backward()
        opt.step()
    with torch.no_grad():
        return float((kopf(merkmale).argmax(-1) == y).float().mean())


def pruefe_theorie() -> bool:
    print("=" * 96)
    print("THEORIEPRUEFUNG Q-019W")
    ok, zeilen = True, []
    h = hadamard() / math.sqrt(DIM)
    eye = torch.eye(DIM, dtype=torch.float64)
    with torch.no_grad():
        # (1) Ry(pi/2) = Drehung um pi/4 in der Konvention von OrthoButterfly
        #     (c = cos theta). 5 Lagen: R^5 = -R je Bit, (-1)^8 = +1.
        b = OrthoButterfly().double()
        b.theta.fill_(math.pi / 4)
        zeilen.append(("butterfly, Ry(pi/2) ueberall == H^8 bis auf Vorzeichen",
                       _signtreu(b(eye), h), "<", 1e-6))
        b.theta.copy_(torch.rand(LAGEN, QUBITS, dtype=torch.float64) * 2 * math.pi)
        zeilen.append(("Gegenprobe: zufaellige Winkel sind NICHT H^8", _signtreu(b(eye), h), ">", 0.1))
        zeilen.append(("wht_fest == H^8 / 16", float((FesteWHT().double()(eye) - h).abs().max()),
                       "<", 1e-6))
        # (2) Sequency-Ordnung
        wechsel = lambda m: (m[:, 1:] != m[:, :-1]).sum(1)          # noqa: E731
        zeilen.append(("Walsh-Zeile k hat genau k Nulldurchgaenge",
                       float((wechsel(walsh_basis()) != torch.arange(DIM)).sum()), "<", 0.5))
        zeilen.append(("wht_fest-Zeile h hat Sequency SEQ_VON_H[h]",
                       float((wechsel(hadamard()) != SEQ_VON_H).sum()), "<", 0.5))
        zeilen.append(("Gegenprobe: natuerliche Ordnung ist NICHT sequency-geordnet",
                       float((wechsel(hadamard()) != torch.arange(DIM)).sum()), ">", 0.5))
        zb = zufalls_basis()
        zeilen.append(("zufall: orthogonal, Norm 16",
                       float((zb @ zb.T - DIM * eye).abs().max()), "<", 1e-8))
        # (3) Etiketten: ein Anteil, kein Rauschen -> Band aus der Walsh-Analyse
        for basis in ("walsh", "zufall"):
            x, y = signale(5, 400, basis, anteile=1, rauschen=0.0)
            k = (x.double() @ _basis(basis).double().T).abs().argmax(1)
            zeilen.append((f"Etikett == Sequency-Band des Anteils ({basis})",
                           float((k // BAND != y).sum()), "<", 0.5))
        x, y = signale(6, 2000, "walsh")
        zeilen.append(("Klassenbalance (2000 -> je 250)",
                       float(torch.bincount(y, minlength=KLASSEN).sub(250).abs().max()), "<", 0.5))
    for name, wert, rel, grenze in zeilen:
        good = wert < grenze if rel == "<" else wert > grenze
        ok &= good
        print(f"  {name:<62} {wert:.2e} {rel} {grenze:g}  {'OK' if good else 'FEHLER'}")

    # (4) Ist das Sequency-Band ueber <Z> linear ablesbar? Ideale Signale: ein
    #     Anteil, kein Rauschen, alle 256 Indizes x beide Vorzeichen, durch
    #     wht_fest und die Auslese. Positivkontrolle: Baender auf den oberen
    #     3 Hadamard-Bits MUESSEN 1,0 geben.
    #     KORREKTUR 28.09. (Tagebuch, vor jeder Zahl): Der Entwurf behauptete
    #     urspruenglich NEIN (XOR von Hadamard-Bits). Das ist falsch: Das Band ist
    #     eine Bijektion von drei Hadamard-Bits (h7, h7^h6, h7^h6^h5), und ein
    #     Softmax-Kopf ueber 8 Klassen trennt JEDE solche Bijektion (Ecke des
    #     Wuerfels {+-1}^3 als Gewicht). Die Zeile ist deshalb nur noch ein
    #     HINWEIS; ob die echte Aufgabe (Anteile, Rauschen) saettigt, entscheidet
    #     allein `saettigung()` vor dem Lauf und HU_SAETTIGUNG im Urteil.
    with torch.no_grad():
        k = torch.arange(DIM).repeat(2)
        x = walsh_basis().float()[k] * torch.cat([torch.ones(DIM), -torch.ones(DIM)])[:, None]
        merk = auslese(FesteWHT()(x / x.norm(dim=1, keepdim=True)))
    pk = _linear_ablesbar(merk, H_VON_SEQ[k] >> 5)
    seq = _linear_ablesbar(merk, k // BAND)
    good_pk = pk > 0.999
    ok &= good_pk
    print(f"  {'Positivkontrolle: obere 3 Hadamard-Bits linear ablesbar':<62} {pk:.4f}  "
          f"{'OK' if good_pk else 'FEHLER (Anpassung blind)'}")
    print(f"  {'HINWEIS: Sequency-Band ueber <Z> ideal linear ablesbar':<62} {seq:.4f}  "
          "(kein Kriterium -- saettigung() entscheidet)")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return bool(ok)


def test_gitterverlaengerung() -> bool:
    """Regel 1(b) in beide Richtungen, gegen eine kuenstliche Glockenkurve.
    Reichweite: 0,001/27 = 3,7e-5 .. 0,03*27 = 0,81."""
    faelle = [("Optimum im Startgitter", 0.003, "innen", 0.003),
              ("eine Verlaengerung oben", 0.09, "innen", 0.09),
              ("eine Verlaengerung unten", 1e-3 / 3, "innen", 1e-3 / 3),
              ("ausserhalb oben", 300.0, "oben", 0.03 * 27),
              ("ausserhalb unten", 1e-9, "unten", 1e-3 / 27)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen")
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        best, rand, _ = sweep_lr("walsh", "butterfly", LR_GITTER, (1, 2), ERWEITERUNGEN,
                                 lambda lr, s: 1.0 / (1.0 + math.log10(lr / opt) ** 2))
        good = abs(math.log10(best / rate_soll)) < 1e-6 and rand == rand_soll
        ok &= good
        print(f"  {name:<28} -> lr {best:<10.4g} Rand {rand:<6} {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


# ---------------------------------------------------------------------------
# Pflichtvortest -- Teil 2: jeden Ausgang einzeln erzwingen
# ---------------------------------------------------------------------------
W, Z, G = ("walsh", 200), ("zufall", 200), ("walsh", 6000)
NIVEAU_K = {W: 0.60, Z: 0.55, G: 0.85}
VERSATZ_K = {W: {"butterfly": 0.05, "dense": 0.0, "lowrank": -0.03, "wht_fest": 0.08, "untied": 0.02},
             Z: {"butterfly": -0.02, "dense": 0.0, "lowrank": -0.03, "wht_fest": -0.10, "untied": -0.01},
             G: {"butterfly": -0.05, "dense": 0.0, "lowrank": -0.08, "wht_fest": 0.0,
                 "untied": -0.02, "narrow": -0.08}}


def _kunst(seed: int, streuung: float = 0.008, seeds: int = 15, versatz=None, niveau=None,
           weg: Sequence = ()) -> Recorder:
    """Synthetischer Recorder: Niveau + Versatz + gemeinsamer Seed-Anteil +
    EIGENES Rauschen je Arm und Zelle (Lehre Q-019). `accuracy` bekommt die
    Versaetze mit UMGEKEHRTEM Vorzeichen und ein eigenes Rauschen -- die beiden
    Felder unterscheiden sich also nicht konstant, und ein Fehler im Metrikfeld
    kippt jedes Richtungsurteil. Eigener Strom je Ziehung ueber crc32."""
    def rng(*t):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, t)).encode())])
    rec = Recorder("Q-019W-vortest", "synthetisch", config={"lr": {}})
    for zelle_, vs0 in VERSATZ_K.items():
        vs = {**vs0, **(versatz or {}).get(zelle_, {})}
        nv = (niveau or {}).get(zelle_, NIVEAU_K[zelle_])
        for s in range(19101, 19101 + seeds):
            grund = rng("grund", s).normal(0, 0.01)
            for arm, v in vs.items():
                if zelle_ in weg or (*zelle_, arm) in weg:
                    continue
                eig = rng(arm, *zelle_, s).normal(0, streuung)
                rec.add(basis=zelle_[0], n_train=zelle_[1], arm=arm, seed=s, lr=0.01,
                        balanced_accuracy=float(nv + v + grund + eig),
                        accuracy=float(nv - v + grund + eig + rng("acc", arm, *zelle_, s)
                                       .normal(0, streuung)), middle_parameters=1)
    return rec


FAELLE = [
    # (Name, Test, kwargs fuer _kunst, Code, Zweigtext, Ziehungen, Anteil)
    ("P-S   Butterfly +0,05", "PRIMAER", {}, "STRUKTUR_BESSER", "Dateneffizienz belegt", 5, 1.0),
    ("P-D   dense +0,05", "PRIMAER", dict(versatz={W: {"butterfly": -0.05}}),
     "DICHT_BESSER", "Negativbefund", 5, 1.0),
    ("P-G   gleich, kleines Rauschen", "PRIMAER",
     dict(streuung=0.0015, versatz={W: {"butterfly": 0.0}}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("P-Gs  +0,002 bei winzigem Rauschen: signifikant UND gleichwertig", "PRIMAER",
     dict(streuung=0.0004, versatz={W: {"butterfly": 0.002}}), "GLEICHWERTIG",
     "zugleich signifikant", 5, 1.0),
    ("P-U   Nulleffekt, grosses Rauschen", "PRIMAER",
     dict(streuung=0.025, versatz={W: {"butterfly": 0.0}}), "UNENTSCHIEDEN", "weder Unterschied",
     40, 0.80),
    ("P-A   Aufloesung faellt (narrow = dense)", "PRIMAER", dict(versatz={G: {"narrow": 0.0}}),
     "HU_AUFLOESUNG", "blind", 40, 0.80),
    ("P-Ar  narrow BESSER als dense: Trennung zaehlt, nicht Richtung", "PRIMAER",
     dict(versatz={G: {"narrow": 0.08}}), "STRUKTUR_BESSER", "Dateneffizienz belegt", 5, 1.0),
    ("P-An  narrow-Zelle fehlt", "PRIMAER", dict(weg=[("walsh", 6000, "narrow")]),
     "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
    ("P-B   Boden unterschritten (walsh)", "PRIMAER", dict(niveau={W: 0.12}), "HU_BODEN",
     "Trainingsversagen", 5, 1.0),
    ("P-Sa  butterfly gesaettigt (0,98)", "PRIMAER", dict(niveau={W: 0.93}), "HU_SAETTIGUNG",
     "gesaettigt", 5, 1.0),
    ("P-Sw  nur wht_fest ueber 0,95 -- kein Familienarm", "PRIMAER",
     dict(niveau={W: 0.85}, versatz={W: {"wht_fest": 0.12}}), "STRUKTUR_BESSER",
     "Dateneffizienz belegt", 5, 1.0),
    ("P-X   dense fehlt bei walsh/200", "PRIMAER", dict(weg=[("walsh", 200, "dense")]),
     "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("V-T   Passung traegt (+0,07)", "P2", {}, "PASSUNG_TRAEGT", "aus der Passung", 5, 1.0),
    ("V-G   gleicher Vorsprung in beiden Basen", "P2",
     dict(streuung=0.0015, versatz={Z: {"butterfly": 0.05}}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("V-S   Vorsprung nur in zufall", "P2",
     dict(versatz={W: {"butterfly": 0.0}, Z: {"butterfly": 0.06}}), "PASSUNG_SCHADET",
     "Gegenrichtung", 5, 1.0),
    ("V-U   Nulleffekt, grosses Rauschen", "P2",
     dict(streuung=0.025, versatz={W: {"butterfly": 0.0}, Z: {"butterfly": 0.0}}),
     "UNENTSCHIEDEN", "weder Unterschied", 40, 0.80),
    ("V-B   Boden unterschritten (zufall)", "P2", dict(niveau={Z: 0.10}), "HU_BODEN",
     "Trainingsversagen", 5, 1.0),
    ("V-X   zufall fehlt ganz", "P2", dict(weg=[Z]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("X-P   zufall fehlt -> PRIMAER trotzdem gerechnet", "PRIMAER", dict(weg=[Z]),
     "STRUKTUR_BESSER", "Dateneffizienz belegt", 5, 1.0),
    ("T-T   Butterfly vor lowrank (+0,08)", "P3", {}, "STRUKTUR_TRAEGT", "diese Struktur", 5, 1.0),
    ("T-G   lowrank = butterfly, kleines Rauschen", "P3",
     dict(streuung=0.0015, versatz={W: {"lowrank": 0.05}}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("T-L   lowrank +0,05 vor butterfly", "P3", dict(versatz={W: {"lowrank": 0.10}}),
     "LOWRANK_BESSER", "Parameterarmut genuegt", 5, 1.0),
    ("T-U   Nulleffekt, grosses Rauschen", "P3",
     dict(streuung=0.025, versatz={W: {"lowrank": 0.05}}), "UNENTSCHIEDEN",
     "weder Unterschied", 40, 0.80),
]


def _szenarien(mut: Optional[Dict[str, object]] = None, laut: bool = True) -> bool:
    alle_ok = True
    for i, (name, test, kw, code, zweig, ziehungen, anteil) in enumerate(FAELLE):
        treffer, marken = 0, []
        for zug in range(ziehungen):
            u = str(analyse(_kunst(seed=1000 * i + zug, **kw), still=True, mut=mut)[test]["verdict"])
            marken.append(u.split(" -- ")[0])
            treffer += int(u.startswith(code + " --") and zweig in u)
        ok = treffer / ziehungen >= anteil - 1e-9
        alle_ok &= ok
        if laut:
            print(f"  {'OK    ' if ok else 'FEHLER'} {name:<64} {treffer}/{ziehungen}  "
                  f"{dict(Counter(marken))}")
        elif not ok:
            print(f"      gefangen von {name.split()[0]}: {treffer}/{ziehungen} {dict(Counter(marken))}")
    return alle_ok


def pruefe_generator() -> bool:
    """Der Pruefstand selbst kann der Fehler sein (CLAUDE.md §5, vierter Punkt)."""
    rec = _kunst(seed=4242)
    r = lambda a, f: np.array(list(_reihe(rec.records, "walsh", 200, a, f).values()))  # noqa: E731
    diff_sd = [float(np.std(r(a, "accuracy") - r(a, METRIK), ddof=1)) for a in FAMILIENARME]
    d_bal, d_acc = r("butterfly", METRIK) - r("dense", METRIK), r("butterfly", "accuracy") - r("dense", "accuracy")
    soll = math.sqrt(2) * 0.008
    pruef = [("accuracy - balanced_accuracy NICHT konstant (sd je Arm > 0)", min(diff_sd) > 1e-4),
             ("gepaarte Differenz kippt mit dem Feld", d_bal.mean() > 0 > d_acc.mean()),
             (f"eigenes Rauschen: sd(butterfly - dense) {np.std(d_bal, ddof=1):.4f} ~ {soll:.4f}",
              0.5 * soll < np.std(d_bal, ddof=1) < 2 * soll)]
    for name, good in pruef:
        print(f"  {'OK    ' if good else 'FEHLER'} {name}")
    return all(g for _, g in pruef)


MUTATIONEN = [("Metrikfeld accuracy statt balanced_accuracy", {"feld": "accuracy"}),
              ("Signifikanz VOR TOST", {"tost_zuerst": False}),
              ("Aufloesung verlangt Richtung statt Trennung", {"richtung": True})]


def vortest() -> bool:
    print("=" * 96)
    print("VORTEST Q-019W -- jeder Ausgang einzeln, ueber mehrere Ziehungen, Marke UND Zweigtext")
    ok = _szenarien()
    print("GENERATOR")
    ok &= pruefe_generator()
    print("MUTATIONSPRUEFUNG -- jeder absichtliche Fehler MUSS den Vortest durchfallen lassen")
    for name, m in MUTATIONEN:
        faellt = not _szenarien(m, laut=False)
        ok &= faellt
        print(f"  {'OK    ' if faellt else 'FEHLER'} Mutation '{name}' "
              f"{'faellt durch' if faellt else 'BLEIBT UNENTDECKT'}")
    print("BESCHREIBENDER TEIL (Durchlauf auf synthetischen Daten) und N*")
    analyse(_kunst(seed=7), still=False)
    u = uebergang([50, 200, 1000, 6000], [0.08, 0.05, 0.01, -0.03])
    good = u["bereich"] == "1000..6000" and abs(u["n_stern"] - 1000 * 6 ** 0.25) < 1e-6 \
        and uebergang([50, 200], [-0.01, 0.02])["bereich"] == "> 200" \
        and uebergang([50, 200], [-0.01, -0.02])["bereich"] == "<= 50"
    ok &= good
    print(f"  {'OK    ' if good else 'FEHLER'} N*-Interpolation {u}")
    print("=" * 96)
    print("VORTEST BESTANDEN" if ok else "VORTEST FEHLGESCHLAGEN")
    return bool(ok)


# ---------------------------------------------------------------------------
# Vor der Freigabe: Saettigung, Macht, Zeit (NICHT waehrend Q-020 starten)
# ---------------------------------------------------------------------------
def saettigung(threads: int = 6, seeds: Sequence[int] = (19161, 19162), lr: float = 0.01) -> None:
    """Feste Rate, nur Groessenordnung. HU_SAETTIGUNG droht, wenn ein Familien-
    arm bei N = 200 ueber 0,95 liegt; dann Schwierigkeit erhoehen, VOR dem Lauf,
    Tagebuch zuerst."""
    print("=" * 96)
    print(f"SAETTIGUNG -- {ANTEILE} Anteile, Rauschen {RAUSCHEN}, Abstand {AMPLITUDENABSTAND}, lr {lr}")
    for basis in ("walsh", "zufall"):
        for n in (N_FAMILIE, N_AUFLOESUNG):
            for arm in ARME_KURVE:
                m = float(np.mean([zelle(basis, n, arm, s, lr, threads)[METRIK] for s in seeds]))
                marke = ("  <== GESAETTIGT (Familie!)" if m > SAETTIGUNG and n == N_FAMILIE
                         and arm in FAMILIENARME else "  <== ueber 0,95" if m > SAETTIGUNG
                         else "  <== unter dem Boden" if m < BODEN else "")
                print(f"  {basis:<7} N={n:<5} {arm:<10} {m:.4f}{marke}", flush=True)


def macht(threads: int = 6, seeds: Sequence[int] = tuple(range(19171, 19176)),
          lr: float = 0.01) -> Dict[str, float]:
    """Regel 2(b): gepaarte Streuung der drei Familientests bei N = 200 aus
    einem echten Vorlauf (feste Rate). TOST +-0,005 erreichbar nur bei
    sd < 0,005*sqrt(15)/1,7613 = 0,0110. Nachweisgrenze bei 80 % Macht,
    zweiseitig, Holm schlimmstenfalls x3: (2,638 + 0,842) * sd / sqrt(15)."""
    rec = Recorder("Q-019W-macht", "Vorlauf", config={"lr": {}})
    for basis, arme in (("walsh", FAMILIENARME), ("zufall", ("butterfly", "dense"))):
        for arm in arme:
            for s in seeds:
                rec.add(**zelle(basis, N_FAMILIE, arm, s, lr, threads))
    fam = _familie(rec.records, METRIK)
    w = {a: _reihe(rec.records, "walsh", N_FAMILIE, a, METRIK) for a in FAMILIENARME}
    z = {a: _reihe(rec.records, "zufall", N_FAMILIE, a, METRIK) for a in ("butterfly", "dense")}
    d = {"PRIMAER": _differenz(w["butterfly"], w["dense"]),
         "P3": _differenz(w["butterfly"], w["lowrank"])}
    dz = _differenz(z["butterfly"], z["dense"])
    d["P2"] = {k: d["PRIMAER"][k] - dz[k] for k in dz}
    tost_grenze = MARGIN * math.sqrt(len(EVAL_SEEDS)) / 1.7613
    print("=" * 96)
    print(f"MACHT Q-019W -- {len(seeds)} Seeds, lr {lr} fest, N = {N_FAMILIE}")
    out = {}
    for t in FAMILIE:
        sd = float(np.std(list(d[t].values()), ddof=1))
        grenze = 3.48 * sd / math.sqrt(len(EVAL_SEEDS))
        out[t] = sd
        print(f"  {t:<8} Mittel {fam[t]['delta']:+.4f}  sd {sd:.4f}  nachweisbar ab {grenze:.4f}  "
              f"TOST {'erreichbar' if sd < tost_grenze else 'NICHT erreichbar (sd > 0,011)'}")
    return out


def zeit(threads: int = 6, lr: float = 0.01) -> float:
    """Je Arm eine Zelle mit vollem Budget (N = 6.000, teuerster Fall), dann
    Hochrechnung: je Zelle des PLANS 3 Tuning-Seeds x ~5 Raten + 15 Mess-Seeds."""
    print("=" * 96)
    print(f"ZEITMESSUNG -- {threads} Threads, {SCHRITTE} Schritte")
    t = {}
    for arm in ARME_KURVE + ("narrow",):
        t[arm] = zelle("walsh", N_AUFLOESUNG, arm, 19199, lr, threads)["sekunden"]
        print(f"  {arm:<10} {t[arm]:>7.1f} s", flush=True)
    gesamt = sum(t[a] * (len(TUNING_SEEDS) * 5 + len(EVAL_SEEDS)) for _, _, arme in PLAN for a in arme)
    print(f"  GESAMT geschaetzt {gesamt / 3600:.1f} h")
    return gesamt


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        raise SystemExit(0 if (pruefe_theorie() & test_gitterverlaengerung() & vortest()) else 1)
    elif befehl == "saettigung":
        saettigung()
    elif befehl == "macht":
        macht()
    elif befehl == "zeit":
        zeit()
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else LAUF_ID)
    elif befehl == "lauf":
        if not (pruefe_theorie() and test_gitterverlaengerung() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print(f"  Nachholen: python -m quantum_sim.walsh auswertung {LAUF_ID}")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | saettigung | macht | zeit | lauf | auswertung [ID|Pfad]")
