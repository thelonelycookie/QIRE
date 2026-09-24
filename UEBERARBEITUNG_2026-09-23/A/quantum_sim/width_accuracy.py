from __future__ import annotations

"""RESEARCH ID: Q-008 -- Haelt die Genauigkeit dort, wo die Architektur schneller ist?

WARUM DIESER LAUF
Gemessen am 5. August: die Butterfly-Umsetzung der Quantenschaltung ist ab
16.384 Dimensionen (14 Qubits) etwa **doppelt so schnell** wie eine dichte
Schicht gleicher Breite -- Median 1,99x, in allen sieben Messreihen schneller.
Darunter verliert sie deutlich (0,21x bei 4.096).

Ein Geschwindigkeitsvorteil ohne Genauigkeitsnachweis ist wertlos. Genau dieser
Fehler stand beim Tensor-Train-Streaming schon einmal ausdruecklich als offener
Punkt im Tagebuch ("Die Genauigkeit bei 4096 Neuronen mit Rang 4 habe ich nicht
getestet -- bisher nur die Geschwindigkeit"). Dieser Lauf holt das nach, bevor
die Zahl irgendwo behauptet wird.

EIN FAIRNESSPROBLEM, DAS VORHER BEHOBEN WERDEN MUSSTE
`ButterflyQuantumLayer` bildet (batch, d) auf (batch, n) ab -- sie kollabiert am
Ende auf n Erwartungswerte. Gegen `Linear(d, d)` gemessen waere das kein fairer
Schichtvergleich, weil die dichte Schicht d Werte weiterreicht statt n. Hier
wird deshalb die echte d->d-Variante benutzt:

    ButterflyBlock(x) = | U(theta) . normalise(x) |^2

also der transformierte Zustand nach der Born-Regel. Das ist zugleich die
Nichtlinearitaet -- der faire Gegenpart ist `Linear(d, d)` gefolgt von ReLU.
Die Zeitmessung bleibt davon unberuehrt: die n Erwartungswerte zu bilden war
teurer als die d Wahrscheinlichkeiten, der d->d-Block ist also eher schneller.

DIE ARME
  butterfly  |U(theta) x|^2 mit geteilten Winkeln -- 2*n*L Parameter,
             O(d log d) Operationen
  dense      Linear(d, d) + ReLU -- d^2 Parameter, O(d^2) Operationen

BEWUSST GEWAEHLTE AUFGABE
MNIST braucht laut B-014 nur rund sieben effektive Richtungen; bei 16.384
Neuronen saettigen beide Arme und der Vergleich waere nichtssagend. Deshalb
synthetische Daten mit **einstellbarer intrinsischer Dimension** (derselbe
Generator wie in B-014): erst dort ist Breite ueberhaupt eine knappe Ressource,
und erst dort ist die Frage "haelt die Genauigkeit" beantwortbar.

RECHENBUDGET -- ehrlich benannt
Bei 14 Qubits kostet ein Vorwaertslauf der dichten Schicht 2,46 s bei Batch 64,
die Gewichtsmatrix belegt 1,07 GB und Adam noch einmal das Dreifache. Ein
vollstaendiger Lauf mit 15 Seeds ist auf diesem Rechner **nicht machbar**.
Dieser Lauf faehrt deshalb ein reduziertes Budget und wenige Seeds; die
erreichte Schranke wird berichtet, nicht ueberdeckt. Ab 15 Qubits laesst sich
der dichte Arm gar nicht mehr bauen -- das steht dann als Machbarkeitsgrenze
in der Tabelle, nicht als fehlender Wert.

VORREGISTRIERUNG
  PRIMAER  Bei 14 Qubits (d=16.384) ist `butterfly` nicht schlechter als
           `dense`: TOST-Gleichwertigkeit innerhalb +-0,02 balancierter
           Genauigkeit. Die Grenze ist bewusst weiter als die ueblichen +-0,005,
           weil bei wenigen Seeds nichts Schaerferes traegt -- die erreichte
           Schranke wird zusaetzlich berichtet.
  P2       Der Abstand zwischen den Armen waechst nicht mit der Breite
           (Spearman ueber die Breiten, Vorzeichen und p).

  H1 : Gleichwertigkeit bei 14 Qubits nachgewiesen -> der Geschwindigkeits-
       vorteil ist echt, weil er nichts kostet.
  H0 : `dense` ist signifikant besser -> der Vorteil ist erkauft, und der
       Abstand gehoert neben jede Zeitangabe.
  HU : weder noch -> die Datenlage traegt es nicht; erreichte Schranke wird
       berichtet und die Aussage bleibt aus der Arbeit heraus.

  AUFLOESUNGSPRUEFUNG (Regel 6): ein dritter Arm `dense_narrow` mit einem
  Viertel der Breite muss signifikant schlechter sein als `dense`. Trennt sich
  das nicht, kann der Aufbau Breitenunterschiede nicht aufloesen und kein
  Ergebnis ist interpretierbar.

Run: python -m quantum_sim.width_accuracy
"""

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.equivalence import holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import apply_1q, cnot_permutation

ARMS = ("butterfly", "dense", "dense_narrow")
QUBITS = (10, 11, 12, 13, 14)
LAYERS = 5
EVAL_SEEDS = (8101, 8102, 8103, 8104, 8105)
TUNING_SEEDS = (8201,)
LR_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03)   # nach unten erweitert: der
# dichte Arm hat bei 16,8 Mio. Parametern deutlich kleinere Raten noetig als
# die Schaltung mit 120, ein gemeinsames enges Gitter benachteiligt ihn.
MARGIN_ACC = 0.02
DENSE_LIMIT_GB = 2.0

INTRINSIC_DIM = 64          # echte Dimension der Daten -- Breite soll knapp sein
NUM_CLASSES = 10
N_CLUSTERS = 512            # viele Haufen, zufaellig auf die Klassen verteilt
N_TRAIN, N_TEST = 6000, 3000
EPOCHS = 6
BATCH = 64


# --------------------------------------------------------------------------
# Daten mit vorgegebener intrinsischer Dimension (Generator wie in B-014)
# --------------------------------------------------------------------------

def make_data(seed: int, input_dim: int, d_intrinsic: int = INTRINSIC_DIM,
              classes: int = NUM_CLASSES, clusters: int = N_CLUSTERS):
    """Viele kleine Haufen mit ZUFAELLIG zugewiesenen Klassen.

    Ein erster Entwurf mit einem Haufen je Klasse war unbrauchbar: beide Arme
    erreichten 1,0000 balancierte Genauigkeit schon bei 1.024 Neuronen, die
    Aufgabe war gesaettigt und der Breitenvergleich damit nichtssagend -- genau
    der Fall, den B-014 vorhersagt, weil die Zahl der noetigen Richtungen von
    der AUFGABE gesetzt wird und nicht von der Architektur.

    Werden dagegen viele Haufen zufaellig auf wenige Klassen verteilt, ist die
    Entscheidungsgrenze stueckweise und zerklueftet; das Netz muss viele Gebiete
    einzeln darstellen. Erst dann ist Breite eine knappe Ressource und die
    Frage "haelt die Genauigkeit bei dieser Breite" ueberhaupt beantwortbar.
    """
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(d_intrinsic, input_dim, generator=g) / math.sqrt(d_intrinsic)
    centres = torch.randn(clusters, d_intrinsic, generator=g) * 2.0
    labels = torch.randint(0, classes, (clusters,), generator=g)
    out = []
    for n in (N_TRAIN, N_TEST):
        c = torch.randint(0, clusters, (n,), generator=g)
        z = centres[c] + torch.randn(n, d_intrinsic, generator=g) * 0.55
        out += [z @ basis, labels[c]]
    return out


# --------------------------------------------------------------------------
# Die Bloecke
# --------------------------------------------------------------------------

class ButterflyBlock(nn.Module):
    """d -> d: |U(theta) . normalise(x)|^2, mit geteilten Winkeln je Qubit-Ebene.

    Die Born-Regel ist hier zugleich die Nichtlinearitaet -- deshalb steht dem
    Arm `dense` ein ReLU gegenueber und nicht eine rein lineare Schicht.
    """

    def __init__(self, num_qubits: int, num_layers: int = LAYERS,
                 entangle: bool = True):
        super().__init__()
        self.n, self.L = num_qubits, num_layers
        self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)
        if entangle and num_qubits > 1:
            for q in range(num_qubits):
                self.register_buffer(f"perm_{q}",
                                     cnot_permutation(num_qubits, q, (q + 1) % num_qubits))
            self.num_perms = num_qubits
        else:
            self.num_perms = 0

    def _gate(self, layer: int, q: int) -> torch.Tensor:
        a, b = self.theta[layer, q, 0], self.theta[layer, q, 1]
        ca, sa = torch.cos(a / 2).to(torch.complex64), torch.sin(a / 2).to(torch.complex64)
        eb = torch.exp(-0.5j * b.to(torch.complex64))
        ebc = torch.exp(0.5j * b.to(torch.complex64))
        return torch.stack([torch.stack([ca * eb, -sa * eb]),
                            torch.stack([sa * ebc, ca * ebc])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x.to(torch.complex64)
        state = state / state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for layer in range(self.L):
            for q in range(self.n):
                state = apply_1q(state, self._gate(layer, q), q, self.n)
            for q in range(self.num_perms):
                state = state[:, getattr(self, f"perm_{q}")]
        # Born-Regel: |Amplitude|^2, skaliert damit die Werte nicht mit d schrumpfen
        return (state.real ** 2 + state.imag ** 2) * float(2 ** self.n)


class Model(nn.Module):
    def __init__(self, arm: str, input_dim: int, width: int, num_qubits: int,
                 classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Linear(input_dim, width)
        # Die Butterfly-Schicht normiert ihre Eingabe -- das gehoert zur
        # Amplituden-Kodierung. Die klassischen Arme bekommen dieselbe
        # Normierung, sonst vergleicht man Struktur GEGEN Normierung statt
        # Struktur gegen Struktur. Ohne sie lag der dichte Arm im Vortest bei
        # 0,1295, also auf Rateniveau -- derselbe Fehler, den Q-002 im Projekt
        # schon einmal aufgedeckt hat.
        self.normalise = arm != "butterfly"
        if arm == "butterfly":
            self.block, out_dim = ButterflyBlock(num_qubits), width
        elif arm == "dense":
            self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU())
            out_dim = width
        elif arm == "dense_narrow":
            narrow = width // 4
            self.block = nn.Sequential(nn.Linear(width, narrow), nn.ReLU())
            out_dim = narrow
        else:
            raise ValueError(arm)
        self.head = nn.Linear(out_dim, classes)

    def block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters())

    def forward(self, x):
        h = self.encoder(x)
        if self.normalise:
            h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(h.shape[1])
        return self.head(self.block(h))


def _balanced(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(np.mean([float((pred[y == c] == c).float().mean())
                          for c in torch.unique(y)]))


def _cell(arm: str, num_qubits: int, seed: int, lr: float, threads: int,
          epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    width = 2 ** num_qubits
    input_dim = 256
    xtr, ytr, xte, yte = make_data(seed, input_dim)

    model = Model(arm, input_dim, width, num_qubits)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(xtr.shape[0], generator=g)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xte[i:i + 256]).argmax(-1)
                          for i in range(0, len(xte), 256)])
    return {"arm": arm, "num_qubits": num_qubits, "width": width, "seed": seed,
            "lr": lr, "balanced_accuracy": _balanced(pred, yte),
            "accuracy": float((pred == yte).float().mean()),
            "block_parameters": model.block_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def feasible(arm: str, num_qubits: int) -> bool:
    if arm == "butterfly":
        return True
    w = 2 ** num_qubits
    div = 4 if arm == "dense_narrow" else 1
    return (w * w / div) * 4 / 1e9 <= DENSE_LIMIT_GB


def run(qubits: Sequence[int] = QUBITS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-008",
        description=("Genauigkeit der Butterfly-Architektur ueber die Breiten, "
                     "bis an die Machbarkeitsgrenze der dichten Kontrolle. "
                     "Synthetische Daten mit intrinsischer Dimension "
                     f"{INTRINSIC_DIM}, damit Breite eine knappe Ressource ist."),
        config={"arms": list(ARMS), "qubits": list(qubits), "seeds": list(seeds),
                "layers": LAYERS, "lr_grid": list(LR_GRID),
                "intrinsic_dim": INTRINSIC_DIM, "n_train": N_TRAIN,
                "n_test": N_TEST, "epochs": EPOCHS, "batch_size": BATCH,
                "margin_accuracy": MARGIN_ACC, "torch_threads": threads,
                "test_metric": "balanced_accuracy",
                "primary": "butterfly gleichwertig zu dense bei 14 Qubits",
                "resolution_check": "dense_narrow muss schlechter sein als dense",
                "budget_note": ("reduziertes Budget und wenige Seeds, weil ein "
                                "dichter Vorwaertslauf bei 14 Qubits 2,46 s "
                                "kostet; erreichte Schranke wird berichtet")},
    )
    for n in qubits:
        print(f"\n=== {n} Qubits (Breite {2 ** n:,}) ===")
        for arm in ARMS:
            if not feasible(arm, n):
                print(f"  {arm}: nicht baubar "
                      f"({(2 ** n) ** 2 * 4 / 1e9:.2f} GB Gewichte)")
                rec.config.setdefault("infeasible", []).append(
                    {"arm": arm, "num_qubits": n})
                continue
            best, best_acc = LR_GRID[0], -1.0
            for lr in LR_GRID:
                a = _cell(arm, n, TUNING_SEEDS[0], lr, threads,
                          epochs=max(2, EPOCHS // 2))["balanced_accuracy"]
                if a > best_acc:
                    best, best_acc = lr, a
            print(f"  {arm}: lr {best}")
            for seed in seeds:
                rec.add(**_cell(arm, n, seed, best, threads))
            vals = [r["balanced_accuracy"] for r in rec.records
                    if r["arm"] == arm and r["num_qubits"] == n]
            print(f"    {arm:>13}  {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")
        print(f"  Zwischenstand nach {n} Qubits gesichert:")
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]

    qubits = sorted({int(r["num_qubits"]) for r in records})
    out: Dict[str, object] = {}

    def vals(arm: str, n: int) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm
                       and int(r["num_qubits"]) == n), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print("RESEARCH ID Q-008 | Genauigkeit ueber die Breiten")
    head = (f'  {"Qubits":>6} {"Breite":>8} {"butterfly":>20} {"dense":>20} '
            f'{"Differenz":>10} {"Block-Par. bf":>14}')
    print(); print(head); print("  " + "-" * (len(head) - 2))
    deltas, ns = [], []
    for n in qubits:
        b, d = vals("butterfly", n), vals("dense", n)
        bp = next((r["block_parameters"] for r in records
                   if r["arm"] == "butterfly" and int(r["num_qubits"]) == n), 0)
        btxt = f"{b.mean():.4f} +-{b.std(ddof=1):.4f}" if len(b) else "        --"
        if len(d):
            dtxt = f"{d.mean():.4f} +-{d.std(ddof=1):.4f}"
            delta = f"{b.mean() - d.mean():>+10.4f}" if len(b) else "        --"
            if len(b) == len(d) and len(b):
                deltas.append(float(b.mean() - d.mean())); ns.append(n)
        else:
            dtxt, delta = "   nicht baubar", "        --"
        print(f'  {n:>6} {2 ** n:>8,} {btxt:>20} {dtxt:>20} {delta:>10} {bp:>14,}')

    # ---- Primaertest bei der groessten gemeinsamen Breite ----
    common = [n for n in qubits if len(vals("butterfly", n)) and len(vals("dense", n))]
    print()
    print("=" * 92)
    if common:
        n0 = max(common)
        b, d = vals("butterfly", n0), vals("dense", n0)
        p = float(stats.wilcoxon(b, d).pvalue)
        eq = tost_paired(b, d, MARGIN_ACC)
        ctrl = vals("dense_narrow", n0)
        c_p = float(stats.wilcoxon(d, ctrl).pvalue) if len(ctrl) == len(d) and len(d) else 1.0
        c_ok = len(ctrl) == len(d) and len(d) > 0 and d.mean() > ctrl.mean() and c_p < 0.05
        p_holm = holm([p, c_p])
        print(f"PRIMAERTEST bei {n0} Qubits (Breite {2 ** n0:,}), n={len(b)} Seeds")
        print(f'  butterfly - dense = {b.mean() - d.mean():+.4f}   p={p:.4f}   '
              f'Holm={p_holm[0]:.4f}')
        print(f'  TOST bei +-{MARGIN_ACC}: p={eq["p_tost"]:.4f}   '
              f'erreichte Schranke +-{eq["achieved"]:.4f}')
        out["primary"] = {"num_qubits": n0, "delta": float(b.mean() - d.mean()),
                          "p": p, "p_holm": p_holm[0], "p_tost": eq["p_tost"],
                          "achieved": eq["achieved"], "n_seeds": int(len(b))}
        print()
        if c_ok:
            ctxt = (f"Aufloesungspruefung BESTANDEN: dense schlaegt dense_narrow "
                    f"({d.mean() - ctrl.mean():+.4f}, p={c_p:.4f}). Der Aufbau "
                    f"loest Breitenunterschiede auf.")
        else:
            ctxt = (f"Aufloesungspruefung NICHT BESTANDEN: dense trennt sich nicht "
                    f"von einem Viertel der Breite (p={c_p:.4f}). Dann ist der "
                    f"Aufbau fuer die Frage blind und der Primaertest oben ist "
                    f"nicht interpretierbar.")
        print(f"  {ctxt}")
        out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

        print()
        if not c_ok:
            v = "HU -- Aufloesungspruefung durchgefallen, kein Ergebnis verwertbar."
        elif p_holm[0] < 0.05 and b.mean() < d.mean():
            v = (f"H0 -- dense ist bei Breite {2 ** n0:,} signifikant besser "
                 f"({b.mean() - d.mean():+.4f}). Der Geschwindigkeitsvorteil ist "
                 f"erkauft; dieser Abstand gehoert neben jede Zeitangabe.")
        elif eq["p_tost"] < 0.05:
            v = (f"H1 -- Gleichwertigkeit innerhalb +-{MARGIN_ACC} nachgewiesen. "
                 f"Der gemessene Geschwindigkeitsvorteil von rund 2x kostet bei "
                 f"dieser Breite keine Genauigkeit.")
        else:
            v = (f"HU -- unentschieden. Weder Unterschied noch Gleichwertigkeit "
                 f"belegt; die Daten tragen erst ab +-{eq['achieved']:.4f}. Bei "
                 f"{len(b)} Seeds ist das erwartbar und gehoert so berichtet.")
        print(f"BEWERTUNG: {v}")
        out["verdict"] = v

    if len(deltas) >= 3:
        rho, pr = stats.spearmanr(ns, deltas)
        print()
        print(f"P2 -- waechst der Abstand mit der Breite? rho={rho:+.3f}, p={pr:.4f}")
        out["trend"] = {"rho": float(rho), "p": float(pr)}
    print("=" * 92)
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.width_accuracy '
              "import analyse; analyse('Q-008')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
