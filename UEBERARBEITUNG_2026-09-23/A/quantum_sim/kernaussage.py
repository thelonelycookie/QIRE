from __future__ import annotations

"""RESEARCH ID: Q-020 -- Die Kernaussage von Projekt A, fair neu gemessen.

WARUM DIESER LAUF NOETIG IST (Review vom 23. September, BERICHT/00_REVIEW_BEFUNDE.md)
1. Die "starke klassische Referenz" aus Q-003 (`classical_norm_deep`, 67.848
   Parameter) ist SCHLECHTER als ein rein lineares Modell (Stichprobe 3 Seeds:
   0,8544 gegen 0,8873 auf MNIST). Normierung auf Einheitslaenge + Tanh-Engstelle
   laehmen sie. Der Abstand +0,0932 misst vor allem die Schwaeche der Kontrolle.
   Gegen ein gewoehnliches ReLU-Netz mit derselben 8-Kanal-Engstelle bleibt in
   der Stichprobe rund +0,01.
2. Die beiden Nullbefunde "Verschraenkung / Tiefe tragen nichts bei" stammen
   aus einem Aufbau MIT trainierbarem Encoder 784->256. Genau diese
   Konstellation hat Q-018 als Artefakt entlarvt: Ein Nullbefund an einem
   Bauteil ist wertlos, solange ein anderes Bauteil dessen Aufgabe uebernehmen
   kann. Ausserdem ist die Positivkontrolle jenes Laufs (5 gegen 1 Lage)
   durchgefallen.

ZWEI AUFBAUTEN
  (a) MIT Encoder wie Q-003: 784 -> 256 (trainierbar) -> Mittelschicht -> 8 Kanaele
      -> Kopf 8 -> 10. MNIST (primaer) und Fashion-MNIST (Wiederholung).
      Getestet wird auf dem OFFIZIELLEN Testsatz (10.000 Bilder), trainiert auf
      2.800 je Seed zufaellig gezogenen Bildern des offiziellen Trainingsteils
      (dieselbe Trainingsmenge wie Q-003). Begruendung in ENTWURF_Q020.md §4:
      bei 1.200 Testbildern (30-%-Split in Q-003) ist allein der
      Stichprobenfehler der Testmenge (~0,007) groesser als die
      Gleichwertigkeitsgrenze 0,005; bei 10.000 sind es ~0,0026. Und der
      offizielle Testsatz ist der, gegen den jede Literaturzahl steht.
  (b) OHNE Encoder wie Q-018: Haufen-Aufgabe aus `frozen_angles.make_data`,
      Eingabe 256 = 2^8 geht direkt in die Mittelschicht, 6 Klassen, 128
      Haufen. Die Stufe ist aus den Q-018-Rohdaten gewaehlt: bei 32 Haufen stand
      die 5-Lagen-Schaltung schon bei 0,9588 (8 von 15 Seeds exakt 1,0 -- Decke),
      bei 128 Haufen bei rund 0,65. `kalibrierung` prueft das vor dem Lauf.

DIE ARME (Mittelschicht; Parameterzahl der Mittelschicht)
  q1              Schaltung, 1 Lage, CNOT-Ring              16 Winkel (8 wirksam*)
  q1_ohne         Schaltung, 1 Lage, OHNE CNOT              16 Winkel (8 wirksam*)
  q5              Schaltung, 5 Lagen, CNOT-Ring             80 Winkel
  q5_ohne         Schaltung, 5 Lagen, OHNE CNOT             80 Winkel
  kron            KLASSISCH: Kronecker-Drehung R1 x ... x R8 aus reellen
                  2x2-Drehungen + Quadrieren + Z-Auslese      8 Winkel
  quad_free       normieren + blockweise freie 8x8-Matrix + Quadrieren (Q-003) 2.048
  mlp             ReLU-MLP, gleiche 8-Kanal-Engstelle     67.848
  linear          256 -> 8, keine Nichtlinearitaet        2.056  (Untergrenze)
  ohne_engstelle  keine 8-Kanal-Engstelle -- BESCHREIBEND (Obergrenze)
  norm_deep       die Kontrolle aus Q-003 -- nur (a), BESCHREIBEND (Anschluss an Q-003)

  * Nachgerechnet in `pruefe_theorie()`: In einer einzigen Lage stehen die
    Rz-Winkel VOR der Messung (ohne CNOT) bzw. vor einer reinen
    Basis-Vertauschung (CNOT ist eine Permutation). Beides aendert die
    Betragsquadrate nicht. Die 8 Rz-Winkel sind also wirkungslos; `q1_ohne`
    rechnet EXAKT dieselbe Funktion wie `kron`, und `q1` dieselbe wie `kron`
    mit einer vertauschten (Paritaets-)Auslese.

  `kron` ist die Aequivalenz-Positivkontrolle: mathematisch dieselbe Familie
  wie `q1_ohne` (bei reeller Eingabe ist die reduzierte Dichtematrix jedes
  Qubits reell, die y-Komponente fehlt; eine komplexe U(2) bringt gegenueber
  einer reellen Drehung nichts, was die Z-Auslese sieht), dieselbe
  Winkelparametrisierung, aber eigene Initialisierung und eigene Lernrate.
  Sie MUSS gleichwertig abschneiden. Tut sie es, ist (i) gezeigt, dass der
  Aufbau "gleichwertig" ueberhaupt liefern KANN, und (ii) die Deutung
  "der Mechanismus ist klassisch" an einer Stelle direkt gemessen.

VORREGISTRIERUNG -- vor jeder Q-020-Zahl festgelegt (23. September 2026)
  Messgroesse: balancierte Genauigkeit. 15 Seeds (20201-20215), Tuning-Seeds
  20251-20253, Lernrate je Arm UND Abschnitt mit dem VOLLEN Budget des
  Messlaufs, Gitter verlaengert in beide Richtungen (Regel 1 a-c).

  PRIMAER  (a) MNIST: q1 gegen mlp. Wilcoxon zweiseitig + TOST +-0,005.
           Ausgaenge: GLEICHWERTIG / SCHALTUNG_BESSER / MLP_BESSER / UNENTSCHIEDEN.
  P2       (b) 128 Haufen: q5 gegen q5_ohne -- traegt Verschraenkung bei, wenn
           kein Encoder aushelfen kann? Ausgaenge: VERSCHRAENKUNG_TRAEGT /
           GLEICHWERTIG / VERSCHRAENKUNG_SCHADET / UNENTSCHIEDEN.
  P3       (b) 128 Haufen: q5 gegen q1 -- traegt Tiefe bei (mit CNOT)?
           Ausgaenge: TIEFE_TRAEGT / GLEICHWERTIG / TIEFE_SCHADET / UNENTSCHIEDEN.
  Holm ueber GENAU diese drei (Regel 4). Alles andere ist beschreibend.

  RANGFOLGE DER ZWEIGE (wie in `zeitreihen.py`, dort begruendet): TOST vor
  Signifikanz. Ein Unterschied, der signifikant UND nachweislich kleiner als
  +-0,005 ist, heisst GLEICHWERTIG (mit Anmerkung). Davor stehen die
  Ausschluesse: HU_UNVOLLSTAENDIG, HU_AUFLOESUNG, HU_BODEN; im
  Gleichwertigkeitszweig zusaetzlich HU_DECKE (beide Arme > 0,95).

  AUFLOESUNGSPRUEFUNG (Regel 6, urteilstragend): mlp muss sich von linear
  TRENNEN (Wilcoxon zweiseitig p<0,05), in (a) auf MNIST und in (b) auf der
  Primaerstufe. Verlangt wird Trennung, nicht Richtung (Lehre Q-015).
  Stichprobe 23.09.: +0,037 auf MNIST -- sie KANN feuern.
  BODENPRUEFUNG (urteilstragend): bester Arm >= Rateniveau + 0,10.
  AEQUIVALENZ-POSITIVKONTROLLE (deutend, nicht urteilstragend): kron gegen
  q1_ohne. "bestanden" = TOST +-0,005 besteht. "nicht erreicht" = weder noch
  -> ein UNENTSCHIEDEN der Familie ist dann als Messgrenze zu lesen.
  "VERLETZT" = signifikanter Unterschied -> Warnung: Implementierung oder
  Deutung pruefen (unter der Nullhypothese in 5 % der Laeufe Fehlalarm).
  WIEDERHOLUNG: Fashion-MNIST wird fuer PRIMAER beschreibend mitberichtet,
  ohne Holm; sie entscheidet nichts allein (wie B-007, Q-003).

  WAS VORHER ERWARTET WIRD (damit es hinterher nicht umgedeutet wird):
  PRIMAER eher SCHALTUNG_BESSER oder UNENTSCHIEDEN (Stichprobe: 0,9337 aus
  Q-003 gegen 0,9246; andere Testmenge!). P2 VERSCHRAENKUNG_TRAEGT: ohne CNOT
  ist jede Tiefe wieder eine Kronecker-Drehung (Gruppe abgeschlossen), die
  8 Auslesekanaele sehen dann nur 8 feste quadratische Formen. P3
  TIEFE_TRAEGT aus demselben Grund. Beschreibend: q5_ohne gleichwertig q1_ohne.

Run (aus dem Ordner A):
    python -m quantum_sim.kernaussage vortest       # Pflicht, ~1 min
    python -m quantum_sim.kernaussage zeit          # Zeitmessung + Hochrechnung
    python -m quantum_sim.kernaussage kalibrierung  # Stufe (b) pruefen
    python -m quantum_sim.kernaussage pilot         # 3 Seeds, KEIN Befund
    python -m quantum_sim.kernaussage lauf          # der Messlauf (Malte entscheidet)
    python -m quantum_sim.kernaussage auswertung [Q-020|Pfad]
"""

import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import RESULTS_DIR, Recorder

from .butterfly_layer import ButterflyQuantumLayer
from .frozen_angles import make_data as haufen_daten
from .hybrid_final_control import QuadraticReadout

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt (Regel 7)
# ---------------------------------------------------------------------------
QUBITS = 8
DIM = 2 ** QUBITS
KANAELE = 8

ARME_A = ("q1", "q1_ohne", "q5", "q5_ohne", "kron", "quad_free", "mlp",
          "linear", "ohne_engstelle", "norm_deep")
ARME_B = ARME_A[:-1]                      # norm_deep nur als Anschluss an Q-003
SCHALTUNG = {"q1": (1, True), "q1_ohne": (1, False),
             "q5": (5, True), "q5_ohne": (5, False)}   # (Lagen, CNOT)

HAUFEN_B = 128
KLASSEN_B = 6
ABSCHNITTE = ("mnist", f"haufen{HAUFEN_B}", "fashion_mnist")   # Reihenfolge = Wichtigkeit
PRIMAER_A = "mnist"
PRIMAER_B = f"haufen{HAUFEN_B}"

N_TRAIN_A, N_OFFIZIELL_TRAIN = 2800, 60000
EPOCHS_A, BATCH_A = 12, 128
EPOCHS_B, BATCH_B = 12, 64                # N_TRAIN/N_TEST aus frozen_angles: 6000/3000

EVAL_SEEDS = tuple(range(20201, 20216))   # 15 Seeds -> kleinstes p zweiseitig 6,1e-5
TUNING_SEEDS = (20251, 20252, 20253)
LR_GITTER_A = (0.003, 0.01, 0.03, 0.1)    # Q-003: Schaltungen am Rand 0,1
LR_GITTER_B = (0.01, 0.03, 0.1, 0.3)      # Q-018: 0,1 / 0,3 / 0,003 gewaehlt
ERWEITERUNGEN = 3                         # je Richtung, Faktor 3

# Pilot: KEIN Befund. Eigene Seeds, damit er keinen Mess-Seed vorwegnimmt.
PILOT_EVAL_SEEDS = (20291, 20292, 20293)
PILOT_TUNING_SEEDS = (20299,)
# Zeitmessung 24.09.: 548 s je Seed ueber alle Arme und Abschnitte -> ein Pilot
# mit 3 Gitterpunkten und Fashion haette ~55 min gekostet. Deshalb nur die beiden
# Familienabschnitte und zwei Punkte, ohne Verlaengerung (~36 min).
PILOT_ABSCHNITTE = ("mnist", f"haufen{HAUFEN_B}")
PILOT_GITTER_A = (0.01, 0.1)
PILOT_GITTER_B = (0.03, 0.1)

FAMILIE = ("PRIMAER", "P2", "P3")
BODEN_ABSTAND = 0.10
SAETTIGUNG = 0.95


def _aufbau(abschnitt: str) -> str:
    return "b" if abschnitt.startswith("haufen") else "a"


def arme(abschnitt: str) -> Tuple[str, ...]:
    return ARME_B if _aufbau(abschnitt) == "b" else ARME_A


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------
def z_auslese(prob: torch.Tensor, n: int = QUBITS) -> torch.Tensor:
    """<Z_q> aus Betragsquadraten; Qubit 0 = hoechstwertiges Bit, wie in
    `butterfly_layer`."""
    out = []
    for q in range(n):
        p = prob.reshape(-1, 2 ** q, 2, 2 ** (n - 1 - q))
        out.append(p[:, :, 0, :].sum(dim=(1, 2)) - p[:, :, 1, :].sum(dim=(1, 2)))
    return torch.stack(out, dim=1)


class KroneckerDrehung(nn.Module):
    """KLASSISCH: normieren -> R(t_1) x ... x R(t_8) -> quadrieren -> Z-Auslese.

    Kein komplexer Zustand, keine Schaltung: der Eingabevektor wird als
    2x2x...x2-Tensor gelesen und auf jeder Achse mit einer reellen
    2x2-Drehung multipliziert. Das ist genau die Kronecker-Produkt-Matrix,
    nur ohne sie je aufzubauen. R(t) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
    ist dieselbe Parametrisierung wie Ry(t), damit Lernraten vergleichbar
    bleiben. 8 Parameter.
    """

    def __init__(self, n: int = QUBITS):
        super().__init__()
        self.n = n
        self.theta = nn.Parameter(torch.rand(n) * 2 * math.pi)

    def wahrscheinlichkeiten(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        h = h.reshape((b,) + (2,) * self.n)
        c, s = torch.cos(self.theta / 2), torch.sin(self.theta / 2)
        for q in range(self.n):
            r = torch.stack([torch.stack([c[q], -s[q]]), torch.stack([s[q], c[q]])])
            h = torch.movedim(torch.movedim(h, q + 1, -1) @ r.T, -1, q + 1)
        return (h ** 2).reshape(b, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return z_auslese(self.wahrscheinlichkeiten(x), self.n)


class NormDeep(nn.Module):
    """Die Mittelschicht von `ClassicalDeepNormed` (Q-003): normieren ->
    Linear/Tanh -> Linear/Tanh. 67.848 Parameter. Nur als Anschluss an Q-003."""

    def __init__(self):
        super().__init__()
        self.netz = nn.Sequential(nn.Linear(DIM, DIM), nn.Tanh(),
                                  nn.Linear(DIM, KANAELE), nn.Tanh())

    def forward(self, x):
        return self.netz(x / x.norm(dim=1, keepdim=True).clamp_min(1e-12))


def _mitte(arm: str, aufbau: str) -> nn.Module:
    if arm in SCHALTUNG:
        lagen, cnot = SCHALTUNG[arm]
        # Butterfly-Umsetzung statt Zustandsvektor-Simulator: `verify()`
        # zeigt dieselbe Funktion bis 1e-4, und sie ist um ein Vielfaches
        # schneller. `pruefe_theorie()` wiederholt die Pruefung fuer n=8.
        return ButterflyQuantumLayer(QUBITS, lagen, entangle=cnot)
    if arm == "kron":
        return KroneckerDrehung()
    if arm == "quad_free":
        return QuadraticReadout(DIM, KANAELE, orthogonal=False)
    if arm == "mlp":
        # (a): ReLU hinter dem Encoder wie in der Stichprobe vom 23.09.;
        # (b): keine ReLU direkt auf der Rohdateneingabe (die ist vorzeichenbehaftet)
        vorne = [nn.ReLU()] if aufbau == "a" else []
        return nn.Sequential(*vorne, nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, KANAELE))
    if arm == "linear":
        return nn.Linear(DIM, KANAELE)
    if arm == "ohne_engstelle":
        return nn.ReLU() if aufbau == "a" else nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
    if arm == "norm_deep":
        return NormDeep()
    raise ValueError(arm)


class Q020Modell(nn.Module):
    """[(a) Encoder 784->256] -> Mittelschicht -> Kopf. Die Mittelschicht wird
    ZUERST gebaut, wie in `HybridModel` -- gleiche Reihenfolge der Ziehungen."""

    def __init__(self, arm: str, aufbau: str, eingabe: int, klassen: int):
        super().__init__()
        self.mitte = _mitte(arm, aufbau)
        self.encoder = nn.Linear(eingabe, DIM) if aufbau == "a" else None
        self.kopf = nn.Linear(DIM if arm == "ohne_engstelle" else KANAELE, klassen)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.kopf(self.mitte(x))

    def mittel_parameter(self) -> int:
        return sum(p.numel() for p in self.mitte.parameters())


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
_ROH: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def daten(abschnitt: str, seed: int):
    """(x_train, y_train, x_test, y_test, eingabe, klassen)."""
    if _aufbau(abschnitt) == "b":
        xtr, ytr, xte, yte = haufen_daten(seed, HAUFEN_B if abschnitt == PRIMAER_B
                                          else int(abschnitt[6:]), KLASSEN_B)
        return xtr, ytr, xte, yte, DIM, KLASSEN_B
    if abschnitt not in _ROH:
        from qire_ai.data_loader import _load_raw
        _, x, y, *_ = _load_raw(abschnitt, 0)
        # OpenML haelt die Originalreihenfolge: erste 60.000 = Training, letzte
        # 10.000 = offizieller Testsatz. Geprueft am 23.09. an den Klassenzahlen
        # (MNIST 980/1135/1032/..., Fashion je 1000).
        _ROH[abschnitt] = (torch.from_numpy(x), torch.from_numpy(y))
    x, y = _ROH[abschnitt]
    idx = torch.from_numpy(np.random.RandomState(seed).choice(
        N_OFFIZIELL_TRAIN, size=N_TRAIN_A, replace=False))
    return x[idx], y[idx], x[N_OFFIZIELL_TRAIN:], y[N_OFFIZIELL_TRAIN:], x.shape[1], 10


# ---------------------------------------------------------------------------
# Eine Zelle
# ---------------------------------------------------------------------------
def zelle(abschnitt: str, arm: str, seed: int, lr: float, threads: int = 6) -> Dict[str, object]:
    t0 = time.perf_counter()
    set_seed(seed)
    torch.set_num_threads(threads)
    xtr, ytr, xte, yte, eingabe, klassen = daten(abschnitt, seed)
    aufbau = _aufbau(abschnitt)
    modell = Q020Modell(arm, aufbau, eingabe, klassen)
    epochs, batch = (EPOCHS_A, BATCH_A) if aufbau == "a" else (EPOCHS_B, BATCH_B)
    opt = torch.optim.Adam(modell.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = xtr.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(modell(xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 5.0)
            opt.step()
    modell.eval()
    with torch.no_grad():
        pred = torch.cat([modell(xte[i:i + 2000]).argmax(-1)
                          for i in range(0, xte.shape[0], 2000)])
    bal = float(np.mean([float((pred[yte == c] == c).float().mean())
                         for c in torch.unique(yte)]))
    return {"aufbau": aufbau, "abschnitt": abschnitt, "arm": arm, "seed": seed,
            "lr": lr, "balanced_accuracy": bal,
            "accuracy": float((pred == yte).float().mean()),
            "middle_parameters": modell.mittel_parameter(),
            "total_parameters": sum(p.numel() for p in modell.parameters()),
            "n_train": int(n), "n_test": int(xte.shape[0]),
            "sekunden": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# Wiederaufnahme -- eigene Datei je Lauf-ID, NICHT `result_store.latest`
# ---------------------------------------------------------------------------
class Zustand:
    """Haelt Tuning-Zellen, Lernraten und Messzeilen und schreibt sie nach
    JEDER Zelle atomar weg.

    Warum nicht `latest()`: T-002 hat gezeigt, dass eine Probe danach die
    neueste Datei ist und der Hauptlauf still bei null anfaengt. Hier hat jeder
    Lauf seine eigene Datei `<ID>_wiederaufnahme.json` (EIN Unterstrich, damit
    `latest("<ID>")`, das `<ID>__*.json` sucht, sie nie findet), und sie wird
    nur angenommen, wenn der Fingerabdruck der Konfiguration passt.
    """

    def __init__(self, lauf_id: str, konfiguration: Dict[str, object]):
        self.pfad = RESULTS_DIR / f"{lauf_id}_wiederaufnahme.json"
        self.fingerabdruck = hashlib.sha256(
            json.dumps(konfiguration, sort_keys=True).encode()).hexdigest()[:16]
        self.tuning: Dict[str, float] = {}
        self.lr: Dict[str, Dict[str, object]] = {}
        self.records: List[Dict[str, object]] = []
        if self.pfad.exists():
            d = json.loads(self.pfad.read_text(encoding="utf-8"))
            if d.get("fingerabdruck") != self.fingerabdruck:
                raise SystemExit(f"Wiederaufnahmedatei {self.pfad.name} gehoert zu einer "
                                 "ANDEREN Konfiguration. Nicht ueberschreiben -- umbenennen "
                                 "oder die Konfiguration zuruecksetzen.")
            self.tuning, self.lr, self.records = d["tuning"], d["lr"], d["records"]
            print(f"  WIEDERAUFNAHME: {len(self.tuning)} Tuning-Zellen, "
                  f"{len(self.records)} Messzeilen aus {self.pfad.name}")

    def speichern(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerabdruck": self.fingerabdruck,
                                   "tuning": self.tuning, "lr": self.lr,
                                   "records": self.records}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.pfad)

    def gemessen(self, abschnitt: str, arm: str, seed: int) -> bool:
        return any(r["abschnitt"] == abschnitt and r["arm"] == arm and r["seed"] == seed
                   for r in self.records)


# ---------------------------------------------------------------------------
# Lernratensuche -- volles Budget, Verlaengerung in BEIDE Richtungen
# ---------------------------------------------------------------------------
def sweep_lr(abschnitt: str, arm: str, gitter: Sequence[float], seeds: Sequence[int],
             erweiterungen: int, messe: Callable[[float, int], float]
             ) -> Tuple[float, str, Dict[float, float]]:
    """Regel 1 (a)-(c), Logik aus `zeitreihen.sweep_lr` uebernommen und gegen
    die aktuellen Regeln geprueft. `messe(lr, seed)` liefert die balancierte
    Genauigkeit EINER Zelle mit dem vollen Budget; im Test wird sie ersetzt.

    Randvermerk: "innen" nur, wenn hinter dem Optimum ein schlechterer Punkt
    gemessen wurde. Rueckgabe (beste Rate, Vermerk, Mittel je Rate)."""
    mittel: Dict[float, float] = {}
    raten, vermerk = list(gitter), "innen"
    for runde in range(erweiterungen + 1):
        for lr in raten:
            if lr not in mittel:
                mittel[lr] = float(np.mean([messe(lr, s) for s in seeds]))
        bester = max(mittel, key=mittel.get)
        am_rand = bester in (max(mittel), min(mittel))
        if not am_rand:
            break
        if runde == erweiterungen:
            vermerk = "oben" if bester == max(mittel) else "unten"
            break
        raten = [max(mittel) * 3] if bester == max(mittel) else [min(mittel) / 3]
    return max(mittel, key=mittel.get), vermerk, mittel


# ---------------------------------------------------------------------------
# Der Lauf
# ---------------------------------------------------------------------------
def run(lauf_id: str = "Q-020", abschnitte: Sequence[str] = ABSCHNITTE,
        seeds: Sequence[int] = EVAL_SEEDS, tuning_seeds: Sequence[int] = TUNING_SEEDS,
        gitter_a: Sequence[float] = LR_GITTER_A, gitter_b: Sequence[float] = LR_GITTER_B,
        erweiterungen: int = ERWEITERUNGEN, threads: int = 6,
        pilot: bool = False) -> Recorder:
    konfig = {"abschnitte": list(abschnitte), "seeds": list(seeds),
              "tuning_seeds": list(tuning_seeds), "gitter_a": list(gitter_a),
              "gitter_b": list(gitter_b), "erweiterungen": erweiterungen,
              "arme_a": list(ARME_A), "arme_b": list(ARME_B),
              "n_train_a": N_TRAIN_A, "epochs_a": EPOCHS_A, "batch_a": BATCH_A,
              "epochs_b": EPOCHS_B, "batch_b": BATCH_B, "haufen_b": HAUFEN_B,
              "klassen_b": KLASSEN_B}
    zustand = Zustand(lauf_id, konfig)
    rec = Recorder(
        experiment_id=lauf_id,
        description=(("PILOT -- KEIN BEFUND. Nur Durchlauf und Groessenordnungen. "
                      if pilot else "")
                     + "Q-020: Kernaussage von A fair neu gemessen. (a) mit Encoder, "
                       "offizieller Testsatz, Schaltung gegen ReLU-MLP gleicher "
                       "Engstelle; (b) ohne Encoder, Verschraenkung und Tiefe."),
        config={**konfig, "pilot": pilot, "torch_threads": threads, "margin": MARGIN,
                "test_metric": "balanced_accuracy", "familie": list(FAMILIE),
                "primaer": "(a) mnist: q1 gegen mlp, Wilcoxon zweiseitig + TOST",
                "p2": f"(b) {PRIMAER_B}: q5 gegen q5_ohne",
                "p3": f"(b) {PRIMAER_B}: q5 gegen q1",
                "aufloesungspruefung": "mlp trennt sich von linear (a: mnist, b: Primaerstufe)",
                "aequivalenzkontrolle": "kron gleichwertig q1_ohne (TOST)",
                "bodenpruefung": f"bester Arm >= Rateniveau + {BODEN_ABSTAND}",
                "saettigung": SAETTIGUNG, "testsatz_a": "offiziell, 10.000",
                "fingerabdruck": zustand.fingerabdruck},
    )
    for abschnitt in abschnitte:
        gitter = gitter_b if _aufbau(abschnitt) == "b" else gitter_a
        print(f"\n  {abschnitt} -- Lernratensuche (volles Budget, "
              f"{len(tuning_seeds)} Tuning-Seed(s))", flush=True)
        for arm in arme(abschnitt):
            schluessel = f"{abschnitt}|{arm}"
            if schluessel in zustand.lr:
                continue

            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = f"{abschnitt}|{_arm}|{seed}|{lr:.6g}"
                if k not in zustand.tuning:
                    zustand.tuning[k] = zelle(abschnitt, _arm, seed, lr, threads)["balanced_accuracy"]
                    zustand.speichern()
                return zustand.tuning[k]

            best, vermerk, mittel = sweep_lr(abschnitt, arm, gitter, tuning_seeds,
                                             erweiterungen, messe)
            zustand.lr[schluessel] = {"lr": best, "rand": vermerk}
            zustand.speichern()
            print(f"    {abschnitt:<13} / {arm:<15} -> lr {best:g}"
                  + (f"  ACHTUNG Rand {vermerk}" if vermerk != "innen" else "")
                  + "   " + "  ".join(f"{k:g}:{mittel[k]:.4f}" for k in sorted(mittel)),
                  flush=True)
        for i, seed in enumerate(seeds, 1):
            for arm in arme(abschnitt):
                if not zustand.gemessen(abschnitt, arm, seed):
                    zustand.records.append(zelle(abschnitt, arm, seed,
                                                 float(zustand.lr[f"{abschnitt}|{arm}"]["lr"]),
                                                 threads))
                    zustand.speichern()
            print(f"    {abschnitt} seed {seed} fertig ({i}/{len(seeds)})", flush=True)
        # Regel 5: nach JEDEM Abschnitt eine vollstaendige Rohdatendatei
        rec.records = list(zustand.records)
        rec.config["lr"] = dict(zustand.lr)
        print("    Zwischenstand gesichert:")
        rec.save()
    return rec


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _werte(records, abschnitt: str, arm: str, feld: str = "balanced_accuracy") -> np.ndarray:
    zeilen = sorted((r for r in records if r["abschnitt"] == abschnitt and r["arm"] == arm),
                    key=lambda r: r["seed"])
    return np.array([float(r[feld]) for r in zeilen])


def _wilcoxon(d: np.ndarray) -> float:
    from scipy import stats
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:                 # alle Differenzen null
        return 1.0
    return 1.0 if math.isnan(p) else p


def _vergleich(records, abschnitt: str, a: str, b: str) -> Optional[Dict[str, object]]:
    va, vb = _werte(records, abschnitt, a), _werte(records, abschnitt, b)
    if not len(va) or len(va) != len(vb):
        return None
    eq = tost_paired(va, vb, MARGIN)
    return {"a": a, "b": b, "abschnitt": abschnitt, "delta": float(va.mean() - vb.mean()),
            "p": _wilcoxon(va - vb), "p_tost": eq["p_tost"], "erreicht": eq["achieved"],
            "siege": int((va > vb).sum()), "n": len(va),
            "mittel_a": float(va.mean()), "mittel_b": float(vb.mean())}


def _lesart(v: Dict[str, object], p: float) -> str:
    if v["p_tost"] < 0.05:
        return f"gleichwertig (+-{MARGIN})"
    if p < 0.05:
        return "Unterschied" + (" (a besser)" if v["delta"] > 0 else " (b besser)")
    return f"unentschieden (+-{v['erreicht']:.4f})"


def _kontrollen(records, abschnitt: str, klassen: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    mittel = {a: float(_werte(records, abschnitt, a).mean())
              for a in arme(abschnitt) if len(_werte(records, abschnitt, a))}
    bester = max(mittel.values()) if mittel else float("nan")
    schwelle = 1.0 / klassen + BODEN_ABSTAND
    out["boden"] = {"bestanden": bool(mittel) and bool(bester >= schwelle),
                    "bester": bester, "schwelle": schwelle}
    r = _vergleich(records, abschnitt, "mlp", "linear")
    out["aufloesung"] = ({"bestanden": bool(r["p"] < 0.05), **r} if r else
                         {"bestanden": False, "fehlt": True})
    k = _vergleich(records, abschnitt, "kron", "q1_ohne")
    if k is None:
        out["aequivalenz"] = {"status": "fehlt"}
    else:
        status = ("bestanden" if k["p_tost"] < 0.05 else
                  "VERLETZT" if k["p"] < 0.05 else "nicht erreicht")
        out["aequivalenz"] = {"status": status, **k}
    return out


# Texte je Familientest: (Code a besser, Code b besser, Satz a besser, Satz b besser)
_TEXTE = {
    "PRIMAER": ("SCHALTUNG_BESSER", "MLP_BESSER",
                "die Schaltung mit 16 Winkeln (8 wirksam) schlaegt das ReLU-MLP mit 67.848 "
                "Parametern bei gleicher 8-Kanal-Engstelle",
                "das ReLU-MLP schlaegt die Schaltung; die Kernaussage in der Form "
                "'gleich gut mit Faktor ~4.000 weniger Parametern' faellt"),
    "P2": ("VERSCHRAENKUNG_TRAEGT", "VERSCHRAENKUNG_SCHADET",
           "ohne Encoder traegt der CNOT-Ring bei; der Nullbefund aus Q-003 war ein "
           "Encoder-Artefakt",
           "ohne CNOT ist die Schaltung BESSER -- Gegenrichtung der Theorie, eigenes Ergebnis"),
    "P3": ("TIEFE_TRAEGT", "TIEFE_SCHADET",
           "ohne Encoder tragen 5 Lagen (mit CNOT) gegenueber 1 Lage bei; der Nullbefund "
           "aus Q-003 war ein Encoder-Artefakt",
           "1 Lage ist BESSER als 5 -- Gegenrichtung, eigenes Ergebnis (Optimierung?)"),
}


def _urteil(tag: str, v: Optional[Dict[str, object]], p_holm: float,
            k: Dict[str, object]) -> str:
    """Rangfolge vor dem Lauf festgelegt: Ausschluesse -> TOST -> Signifikanz."""
    code_a, code_b, satz_a, satz_b = _TEXTE[tag]
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
    if not k["aufloesung"]["bestanden"]:
        return (f"HU_AUFLOESUNG -- {tag}: Aufloesungspruefung durchgefallen (mlp trennt sich "
                f"nicht von linear). Der Aufbau sieht nicht einmal diesen groben Unterschied; "
                f"KEIN Nullbefund, der Lauf ist fuer diese Frage blind.")
    if not k["boden"]["bestanden"]:
        return (f"HU_BODEN -- {tag}: Bodenpruefung durchgefallen (bester Arm "
                f"{k['boden']['bester']:.4f} < {k['boden']['schwelle']:.3f}); die Zelle misst "
                f"Trainingsversagen, nicht Architektur.")
    z = f"{v['a']} {v['mittel_a']:.4f} gegen {v['b']} {v['mittel_b']:.4f}, " \
        f"{v['delta']:+.4f}, {v['siege']}/{v['n']} Seeds, p={v['p']:.4f}, Holm={p_holm:.4f}"
    if v["p_tost"] < 0.05:
        if v["mittel_a"] > SAETTIGUNG and v["mittel_b"] > SAETTIGUNG:
            return (f"HU_DECKE -- {tag}: TOST besteht, aber beide Arme liegen ueber "
                    f"{SAETTIGUNG} ({z}). Gleichwertigkeit an der Decke ist keine Aussage.")
        t = (f"GLEICHWERTIG -- {tag}: gleichwertig innerhalb +-{MARGIN} "
             f"(TOST p={v['p_tost']:.4f}, erreicht +-{v['erreicht']:.4f}); {z}. "
             f"Positiver Gleichwertigkeitsnachweis, nicht blosses p>0,05.")
        if p_holm < 0.05:
            t += (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber kleiner als die "
                  "Erheblichkeitsgrenze; massgeblich ist die Grenze.")
        return t
    if p_holm < 0.05:
        code, satz = (code_a, satz_a) if v["delta"] > 0 else (code_b, satz_b)
        t = f"{code} -- {tag}: {satz}; {z}."
        if max(v["mittel_a"], v["mittel_b"]) > SAETTIGUNG:
            t += f" Ein Arm liegt ueber {SAETTIGUNG}: der Abstand ist eine Untergrenze."
        return t
    t = (f"UNENTSCHIEDEN -- {tag}: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
         f"die Daten tragen erst ab +-{v['erreicht']:.4f}; {z}.")
    if k["aequivalenz"].get("status") != "bestanden":
        t += (" Die Aequivalenz-Positivkontrolle ist hier ebenfalls nicht bestanden -- "
              "unentschieden ist dann als Messgrenze des Aufbaus zu lesen, nicht als Hinweis "
              "auf einen Unterschied.")
    return t


BESCHREIBEND = {
    "a": [("q1", "quad_free", "16 Winkel gegen 2.048 (Q-003-Vergleich)"),
          ("q1", "linear", "Schaltung gegen lineare Untergrenze"),
          ("q5", "q1", "Tiefe MIT Encoder (alte Q-003-Kontrolle)"),
          ("q5", "q5_ohne", "Verschraenkung MIT Encoder (alter Q-003-Nullbefund)"),
          ("q5_ohne", "q1_ohne", "Tiefe ohne CNOT -- Theorie: gleichwertig"),
          ("q1", "q1_ohne", "CNOT in 1 Lage = nur Auslese-Vertauschung"),
          ("kron", "q1_ohne", "AEQUIVALENZ-POSITIVKONTROLLE"),
          ("mlp", "linear", "AUFLOESUNGSPRUEFUNG"),
          ("mlp", "norm_deep", "ReLU-MLP gegen die Q-003-Kontrolle (Strohmann?)"),
          ("ohne_engstelle", "mlp", "was die 8-Kanal-Engstelle kostet")],
    "b": [("q5_ohne", "q1_ohne", "Tiefe ohne CNOT -- Theorie: gleichwertig"),
          ("q1", "q1_ohne", "CNOT in 1 Lage = nur Auslese-Vertauschung"),
          ("q5", "quad_free", "Schaltung 80 gegen 2.048"),
          ("q5", "mlp", "Schaltung 80 gegen ReLU-MLP"),
          ("q1", "mlp", "Schaltung 16 gegen ReLU-MLP"),
          ("kron", "q1_ohne", "AEQUIVALENZ-POSITIVKONTROLLE"),
          ("mlp", "linear", "AUFLOESUNGSPRUEFUNG"),
          ("ohne_engstelle", "mlp", "was die 8-Kanal-Engstelle kostet")],
}


def analyse(rec_or_id, still: bool = False) -> Dict[str, object]:
    def say(*a):
        if not still:
            print(*a)

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    pilot = bool(config.get("pilot", False))
    out: Dict[str, object] = {"pilot": pilot}
    vorhanden = [a for a in ABSCHNITTE if any(r["abschnitt"] == a for r in records)]
    say("\nRESEARCH ID Q-020" + ("  ***  PILOT -- KEIN BEFUND  ***" if pilot else ""))

    for ab in vorhanden:
        say(f"\n--- {ab} " + "-" * max(4, 60 - len(ab)))
        say(f'  {"Arm":<15}{"balanced":>10}{"sd":>8}{"Mitte-Param.":>14}{"lr":>8}{"n":>4}')
        tab = {}
        for arm in arme(ab):
            v = _werte(records, ab, arm)
            if not len(v):
                continue
            r0 = next(r for r in records if r["abschnitt"] == ab and r["arm"] == arm)
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            tab[arm] = {"mittel": float(v.mean()), "sd": sd, "n": len(v),
                        "parameter": r0["middle_parameters"], "lr": r0["lr"]}
            say(f"  {arm:<15}{v.mean():>10.4f}{sd:>8.4f}{r0['middle_parameters']:>14,}"
                f"{r0['lr']:>8g}{len(v):>4}")
        out[f"tabelle_{ab}"] = tab

    kontrollen = {PRIMAER_A: _kontrollen(records, PRIMAER_A, 10),
                  PRIMAER_B: _kontrollen(records, PRIMAER_B, KLASSEN_B)}
    out["kontrollen"] = kontrollen
    paare = {"PRIMAER": (PRIMAER_A, "q1", "mlp"), "P2": (PRIMAER_B, "q5", "q5_ohne"),
             "P3": (PRIMAER_B, "q5", "q1")}
    verg = {t: _vergleich(records, *paare[t]) for t in FAMILIE}
    # Holm ueber die GANZE angemeldete Familie. Fehlt ein Mitglied, geht es mit
    # p=1 ein -- konservativ, die Familiengroesse bleibt 3.
    p_holm = dict(zip(FAMILIE, holm([verg[t]["p"] if verg[t] else 1.0 for t in FAMILIE])))
    say("\n" + "=" * 96)
    say(f"VORREGISTRIERTE FAMILIE (Holm ueber {len(FAMILIE)}), Grenze +-{MARGIN}")
    for t in FAMILIE:
        k = kontrollen[paare[t][0]]
        u = _urteil(t, verg[t], p_holm[t], k)
        out[t] = {**(verg[t] or {}), "p_holm": p_holm[t], "verdict": u}
        say(f"  [{t}] {u}")

    # Wiederholung (beschreibend, ohne Holm)
    w = _vergleich(records, "fashion_mnist", "q1", "mlp")
    if w:
        wtxt = (f"Wiederholung fashion_mnist (beschreibend, ohne Holm): q1 gegen mlp "
                f"{w['delta']:+.4f}, p={w['p']:.4f}, TOST p={w['p_tost']:.4f} -> "
                f"{_lesart(w, w['p'])}")
        out["PRIMAER"]["wiederholung"] = {**w, "text": wtxt}
        say(f"  {wtxt}")

    say("\nKONTROLLEN")
    for ab, k in kontrollen.items():
        a = k["aufloesung"]
        say(f"  {ab}: Boden {'bestanden' if k['boden']['bestanden'] else 'NICHT BESTANDEN'} "
            f"(bester {k['boden']['bester']:.4f}, Schwelle {k['boden']['schwelle']:.3f}) | "
            f"Aufloesung {'bestanden' if a['bestanden'] else 'NICHT BESTANDEN'}"
            + (f" (mlp-linear {a['delta']:+.4f}, p={a['p']:.4f}"
               + (", UNERWARTETE Richtung: linear besser" if a["delta"] < 0 else "") + ")"
               if "delta" in a else "")
            + f" | Aequivalenzkontrolle kron~q1_ohne: {k['aequivalenz']['status']}"
            + (f" ({k['aequivalenz']['delta']:+.4f}, TOST p={k['aequivalenz']['p_tost']:.4f})"
               if "delta" in k["aequivalenz"] else ""))
        if k["aequivalenz"].get("status") == "VERLETZT":
            say("    WARNUNG: kron und q1_ohne sind dieselbe Funktionsfamilie mit derselben "
                "Parametrisierung. Ein signifikanter Unterschied heisst: Implementierung "
                "pruefen (pruefe_theorie) -- oder Fehlalarm, unter der Nullhypothese 5 %.")

    say("\nBESCHREIBEND (ohne Holm, kein Urteil haengt daran)")
    besch = []
    for ab in vorhanden:
        for a, b, was in BESCHREIBEND[_aufbau(ab)]:
            v = _vergleich(records, ab, a, b)
            if v:
                v["lesart"], v["was"] = _lesart(v, v["p"]), was
                besch.append(v)
                say(f"  {ab:<13} {a:>14} - {b:<14} {v['delta']:+.4f}  p={v['p']:.4f}  "
                    f"TOST p={v['p_tost']:.4f}  {v['lesart']:<28} {was}")
    out["beschreibend"] = besch

    raender = {k: v["rand"] for k, v in config.get("lr", {}).items() if v.get("rand") != "innen"}
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    if pilot:
        out["verdict"] = "PILOT, KEIN BEFUND: " + out["verdict"]
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest (CLAUDE.md §5)
# ---------------------------------------------------------------------------
def pruefe_theorie() -> bool:
    """Deterministische Pruefungen, ohne Training. Eine Deutung, die im Code
    steht, muss nachgerechnet sein, bevor sie ein Urteil traegt."""
    from .butterfly_layer import verify
    from .quantum_layer import QuantumLayer
    print("=" * 96)
    print("THEORIEPRUEFUNG Q-020")
    ok = verify(qubits=(QUBITS,), layers=(1, 5))
    torch.manual_seed(0)
    x = torch.randn(64, DIM)
    zeilen = []
    with torch.no_grad():
        for cnot in (False, True):
            s = ButterflyQuantumLayer(QUBITS, 1, entangle=cnot)
            k = KroneckerDrehung()
            k.theta.copy_(s.theta[0, :, 0])
            p = k.wahrscheinlichkeiten(x)
            for i in range(s.num_perms):          # CNOT = Vertauschung der Basis
                p = p[:, getattr(s, f"perm_{i}")]
            dev = float((z_auslese(p) - s(x)).abs().max())
            zeilen.append((f"1 Lage {'mit' if cnot else 'ohne'} CNOT == Kronecker"
                           f"{' + vertauschte Auslese' if cnot else ''}", dev))
            ref = s(x)
            s.theta[0, :, 1] = torch.rand(QUBITS) * 2 * math.pi     # Rz neu wuerfeln
            zeilen.append((f"1 Lage {'mit' if cnot else 'ohne'} CNOT: Rz-Winkel wirkungslos",
                           float((s(x) - ref).abs().max())))
        # Gegenprobe: bei 5 Lagen MIT CNOT duerfen die Rz-Winkel NICHT wirkungslos sein,
        # sonst prueft die Zeile oben nichts.
        s5 = ButterflyQuantumLayer(QUBITS, 5, entangle=True)
        ref = s5(x)
        s5.theta[:, :, 1] = torch.rand(5, QUBITS) * 2 * math.pi
        gegen = float((s5(x) - ref).abs().max())
    for name, dev in zeilen:
        good = dev < 1e-4
        ok &= good
        print(f"  {name:<52} max|Diff| {dev:.2e}  {'OK' if good else 'FEHLER'}")
    good = gegen > 1e-2
    ok &= good
    print(f"  {'Gegenprobe: 5 Lagen mit CNOT, Rz WIRKT':<52} max|Diff| {gegen:.2e}  "
          f"{'OK' if good else 'FEHLER (Pruefung blind)'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return bool(ok)


def test_gitterverlaengerung() -> bool:
    """Regel 1(b) in beide Richtungen, gegen eine kuenstliche Glockenkurve.
    Reichweite Startgitter 0,003..0,1 (a), 3 Schritte Faktor 3: 1,1e-4 .. 2,7."""
    faelle = [("Optimum im Startgitter", 0.01, "innen", 0.01),
              ("eine Verlaengerung oben", 0.3, "innen", 0.3),
              ("eine Verlaengerung unten", 0.001, "innen", 0.001),
              ("ausserhalb oben", 300.0, "oben", 0.1 * 27),
              ("ausserhalb unten", 1e-9, "unten", 0.003 / 27)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen")
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        best, rand, _ = sweep_lr("mnist", "q1", LR_GITTER_A, (1, 2), ERWEITERUNGEN,
                                 lambda lr, s: 1.0 / (1.0 + math.log10(lr / opt) ** 2))
        good = abs(math.log10(best / rate_soll)) < 1e-6 and rand == rand_soll
        ok &= good
        print(f"  {name:<28} -> lr {best:<10.4g} Rand {rand:<6} {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


NIVEAU = {"mnist": 0.90, PRIMAER_B: 0.60, "fashion_mnist": 0.80}
VERSATZ = {"a": {"q1": 0.0, "q1_ohne": -0.01, "q5": 0.0, "q5_ohne": -0.01, "kron": -0.01,
                 "quad_free": -0.01, "mlp": 0.0, "linear": -0.04, "ohne_engstelle": 0.01,
                 "norm_deep": -0.08},
           "b": {"q1": -0.20, "q1_ohne": -0.30, "q5": 0.0, "q5_ohne": -0.30, "kron": -0.30,
                 "quad_free": -0.10, "mlp": 0.20, "linear": -0.30, "ohne_engstelle": 0.25}}


def _kunst(seed: int, streuung: float = 0.008, n: int = 15,
           versatz: Optional[Dict[str, Dict[str, float]]] = None,
           niveau: Optional[Dict[str, float]] = None,
           abschnitte: Sequence[str] = ABSCHNITTE) -> Recorder:
    """Synthetischer Recorder. Gemeinsamer Seed-Anteil PLUS eigenes Rauschen je
    Arm (Lehre Q-019: ohne eigenes Rauschen kuerzt sich alles weg). Eigener
    Zufallsstrom je (Arm, Abschnitt, Seed) ueber crc32, damit ein zusaetzlicher
    Arm keine Ziehung verschiebt. `accuracy` bekommt die Versaetze mit
    UMGEKEHRTEM Vorzeichen: rechnet die Auswertung auf dem falschen Feld, kippt
    jedes Urteil -- das sieht der Vortest sofort."""
    def rng(*teile):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, teile)).encode())])
    rec = Recorder("Q-020-vortest", "synthetisch", config={"pilot": False, "lr": {}})
    for ab in abschnitte:
        au = _aufbau(ab)
        vs = dict(VERSATZ[au])
        vs.update((versatz or {}).get(au, {}))
        nv = (niveau or {}).get(ab, NIVEAU[ab])
        for s in range(20201, 20201 + n):
            grund = rng("grund", ab, s).normal(0, 0.01)
            for arm in arme(ab):
                eig = rng(arm, ab, s).normal(0, streuung)
                rec.add(aufbau=au, abschnitt=ab, arm=arm, seed=s, lr=0.01,
                        balanced_accuracy=float(nv + vs[arm] + grund + eig),
                        accuracy=float(nv - vs[arm] + grund + eig),
                        middle_parameters=1)
    return rec


def vortest() -> bool:
    """Jeden Ausgang einzeln erzwingen, ueber mehrere Ziehungen, und auf den
    ZWEIGTEXT pruefen, nicht nur auf die Marke."""
    kron0 = {"kron": -0.01}
    faelle = [
        # (Name, Test, kwargs, Code, Zweigtext, Ziehungen, Anteil)
        ("P-G   q1 = mlp, kleines Rauschen", "PRIMAER",
         dict(streuung=0.0015), "GLEICHWERTIG", "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
        ("P-Gs  +0,002 bei winzigem Rauschen: signifikant UND gleichwertig", "PRIMAER",
         dict(streuung=0.0004, versatz={"a": {"q1": 0.002}}), "GLEICHWERTIG",
         "zugleich signifikant", 5, 1.0),
        ("P-S   Schaltung +0,03", "PRIMAER", dict(versatz={"a": {"q1": 0.03}}),
         "SCHALTUNG_BESSER", "schlaegt das ReLU-MLP", 5, 1.0),
        ("P-M   MLP +0,03", "PRIMAER", dict(versatz={"a": {"q1": -0.03}}),
         "MLP_BESSER", "Kernaussage", 5, 1.0),
        ("P-U   Nulleffekt, grosses Rauschen", "PRIMAER", dict(streuung=0.025),
         "UNENTSCHIEDEN", "Messgrenze", 40, 0.80),
        ("P-A   Aufloesung faellt (mlp = linear)", "PRIMAER",
         dict(versatz={"a": {"q1": 0.03, "linear": 0.0}}), "HU_AUFLOESUNG", "blind", 40, 0.80),
        ("P-Ar  linear BESSER als mlp: Trennung zaehlt, nicht Richtung", "PRIMAER",
         dict(versatz={"a": {"q1": 0.03, "linear": 0.05}}), "SCHALTUNG_BESSER",
         "schlaegt das ReLU-MLP", 5, 1.0),
        ("P-B   Boden unterschritten", "PRIMAER",
         dict(versatz={"a": {"q1": 0.03}}, niveau={"mnist": 0.12}), "HU_BODEN",
         "Trainingsversagen", 5, 1.0),
        ("P-D   beide an der Decke", "PRIMAER",
         dict(streuung=0.0015, niveau={"mnist": 0.985}), "HU_DECKE", "Decke", 5, 1.0),
        ("V-T   Verschraenkung +0,2", "P2", {}, "VERSCHRAENKUNG_TRAEGT", "Encoder-Artefakt", 5, 1.0),
        ("V-G   Verschraenkung gleichwertig", "P2",
         dict(streuung=0.0015, versatz={"b": {"q5_ohne": 0.0}}), "GLEICHWERTIG",
         "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
        ("V-S   ohne CNOT besser", "P2", dict(versatz={"b": {"q5_ohne": 0.05}}),
         "VERSCHRAENKUNG_SCHADET", "Gegenrichtung", 5, 1.0),
        ("V-U   Nulleffekt, grosses Rauschen", "P2",
         dict(streuung=0.025, versatz={"b": {"q5_ohne": 0.0}}), "UNENTSCHIEDEN",
         "weder Unterschied", 40, 0.80),
        # P-A und V-A haengen an einem NICHT signifikanten Test unter wahrer
        # Nullhypothese -> in 5 % der Ziehungen trennt er doch. Wie Szenario G
        # in zeitreihen: 40 Ziehungen, Anteil statt 5/5.
        ("V-A   Aufloesung (b) faellt", "P2", dict(versatz={"b": {"linear": 0.20}}),
         "HU_AUFLOESUNG", "blind", 40, 0.80),
        ("V-D   Decke in (b)", "P2",
         dict(streuung=0.0015, versatz={"b": {"q5_ohne": 0.0}}, niveau={PRIMAER_B: 0.985}),
         "HU_DECKE", "Decke", 5, 1.0),
        ("T-T   Tiefe +0,2", "P3", {}, "TIEFE_TRAEGT", "5 Lagen", 5, 1.0),
        ("T-G   Tiefe gleichwertig", "P3",
         dict(streuung=0.0015, versatz={"b": {"q1": 0.0}}), "GLEICHWERTIG",
         "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
        ("T-S   1 Lage besser", "P3", dict(versatz={"b": {"q1": 0.05}}),
         "TIEFE_SCHADET", "Gegenrichtung", 5, 1.0),
        ("T-U   Nulleffekt, grosses Rauschen", "P3",
         dict(streuung=0.025, versatz={"b": {"q1": 0.0}}), "UNENTSCHIEDEN",
         "weder Unterschied", 40, 0.80),
        ("X-U   (b) fehlt ganz -> P2 unvollstaendig", "P2",
         dict(abschnitte=("mnist", "fashion_mnist")), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
        ("X-P   (b) fehlt -> PRIMAER trotzdem gerechnet", "PRIMAER",
         dict(abschnitte=("mnist",), versatz={"a": {"q1": 0.03}}), "SCHALTUNG_BESSER",
         "Holm", 5, 1.0),
    ]
    print("=" * 96)
    print("VORTEST Q-020 -- jeder Ausgang einzeln, ueber mehrere Rauschziehungen, Zweigtext geprueft")
    alle_ok = True
    for i, (name, test, kw, code, zweig, ziehungen, anteil) in enumerate(faelle):
        treffer, marken = 0, []
        for zug in range(ziehungen):
            u = str(analyse(_kunst(seed=1000 * i + zug, **kw), still=True)[test]["verdict"])
            marken.append(u.split(" -- ")[0])
            treffer += int(u.startswith(code + " --") and zweig in u)
        quote = treffer / ziehungen
        ok = quote >= anteil - 1e-9
        alle_ok &= ok
        print(f"  {'OK    ' if ok else 'FEHLER'} {name:<66} {treffer}/{ziehungen}  "
              f"{dict(Counter(marken))}")
    # Kontrollen, die kein Familienurteil tragen, aber gemeldet werden muessen
    extra = [
        ("K-V   kron weicht ab -> Aequivalenzkontrolle VERLETZT",
         dict(versatz={"a": {"kron": -0.06}}), "VERLETZT"),
        ("K-B   kron = q1_ohne, kleines Rauschen -> bestanden",
         dict(streuung=0.0015, versatz={"a": kron0}), "bestanden"),
    ]
    for name, kw, soll in extra:
        stati = [analyse(_kunst(seed=77 + z, **kw), still=True)["kontrollen"][PRIMAER_A]
                 ["aequivalenz"]["status"] for z in range(5)]
        ok = all(s == soll for s in stati)
        alle_ok &= ok
        print(f"  {'OK    ' if ok else 'FEHLER'} {name:<66} {dict(Counter(stati))}")
    print("  Holm-Familiengroesse:", len(FAMILIE))
    print("=" * 96)
    print("VORTEST BESTANDEN" if alle_ok else "VORTEST FEHLGESCHLAGEN")
    return alle_ok


# ---------------------------------------------------------------------------
# Zeitmessung, Kalibrierung, Pilot
# ---------------------------------------------------------------------------
def zeitmessung(threads: int = 6, lr: float = 0.03) -> Dict[str, float]:
    """Je Arm EINE Zelle mit vollem Budget, je Aufbau; dann Hochrechnung."""
    print("=" * 96)
    print(f"ZEITMESSUNG -- je Arm eine Zelle, volles Budget, {threads} Threads")
    zeiten: Dict[str, float] = {}
    for ab in ("mnist", PRIMAER_B):
        daten(ab, 1)                          # Laden nicht mitmessen
        for arm in arme(ab):
            r = zelle(ab, arm, 20999, lr, threads)
            zeiten[f"{ab}|{arm}"] = r["sekunden"]
            print(f"  {ab:<12} {arm:<15} {r['sekunden']:>7.1f} s   bal {r['balanced_accuracy']:.4f}",
                  flush=True)
    hochrechnung(zeiten)
    return zeiten


def hochrechnung(zeiten: Dict[str, float], punkte: float = 5.0) -> float:
    """Tuning: Arme x Tuning-Seeds x `punkte` Gitterpunkte (4 Start + im Mittel
    ~1 Verlaengerung, geschaetzt), Messung: Arme x 15 Seeds. Fashion wie MNIST."""
    gesamt = 0.0
    for ab in ABSCHNITTE:
        quelle = "mnist" if ab == "fashion_mnist" else ab
        summe = sum(v for k, v in zeiten.items() if k.startswith(quelle + "|"))
        t = summe * (len(TUNING_SEEDS) * punkte + len(EVAL_SEEDS))
        gesamt += t
        print(f"  {ab:<14} Summe je Seed ueber alle Arme {summe:>6.0f} s -> "
              f"{t / 3600:>5.1f} h  ({len(TUNING_SEEDS)}x{punkte:g} Tuning + "
              f"{len(EVAL_SEEDS)} Mess-Seeds)")
    print(f"  GESAMT geschaetzt {gesamt / 3600:.1f} h auf diesem Rechner")
    return gesamt


def kalibrierung(threads: int = 6, haufen: Sequence[int] = (64, 128),
                 seeds: Sequence[int] = (20297, 20298)) -> None:
    """Prueft die Stufe (b) vor dem Lauf: Boden, Decke, Aufloesung. Feste Raten
    (Schaltung 0,1 wie Q-018, klassisch 0,01), nur Groessenordnung."""
    global HAUFEN_B
    alt = HAUFEN_B
    print("=" * 96)
    print("KALIBRIERUNG (b) -- 6 Klassen, Rateniveau 0,167")
    for h in haufen:
        HAUFEN_B = h
        werte = {arm: float(np.mean([zelle(PRIMAER_B, arm, s, 0.1 if arm in SCHALTUNG
                                           else 0.01, threads)["balanced_accuracy"]
                                     for s in seeds]))
                 for arm in ("q1", "q5", "q5_ohne", "mlp", "linear")}
        print(f"  {h:>4} Haufen: " + "  ".join(f"{a}={v:.3f}" for a, v in werte.items()),
              flush=True)
    HAUFEN_B = alt


def pilot(threads: int = 6) -> None:
    rec = run(lauf_id="Q-020-pilot", abschnitte=PILOT_ABSCHNITTE,
              seeds=PILOT_EVAL_SEEDS, tuning_seeds=PILOT_TUNING_SEEDS,
              gitter_a=PILOT_GITTER_A, gitter_b=PILOT_GITTER_B, erweiterungen=0,
              threads=threads, pilot=True)
    try:
        auswertung = analyse(rec)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        auswertung = {"error": repr(exc)}
    rec.save(analysis=auswertung)


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        ok = pruefe_theorie() & test_gitterverlaengerung() & vortest()
        raise SystemExit(0 if ok else 1)
    elif befehl == "zeit":
        zeitmessung()
    elif befehl == "kalibrierung":
        kalibrierung()
    elif befehl == "pilot":
        pilot()
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else "Q-020")
    elif befehl == "lauf":
        if not (pruefe_theorie() and test_gitterverlaengerung() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                  # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print("  Nachholen: python -m quantum_sim.kernaussage auswertung Q-020")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | zeit | kalibrierung | pilot | lauf | auswertung [ID|Pfad]")
