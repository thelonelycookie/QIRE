from __future__ import annotations

"""RESEARCH ID: Q-001 -- Laesst sich ein Hybridmodell mit echter Quantenschaltung
trainieren, und was leistet es bei gleicher Parameterzahl?

WAS HIER ANDERS IST ALS IN ALLEN VORIGEN VERSUCHEN
A-001 bis B-003 haben quanten*inspirierte* klassische Verfahren gegen
klassische Kontrollen getestet -- fuenfmal ohne quantenspezifischen Beitrag.
Das ist wenig ueberraschend: dort wurde durchgehend klassisch gerechnet, nur
die Struktur war quantenmotiviert.

Hier rechnet tatsaechlich Quantenmechanik: Superposition ueber 2^n Zustaende,
Verschraenkung durch CNOT-Gatter, Ausgabe durch Messung nach der Born-Regel.
Die Schaltung wird klassisch simuliert, aber die Physik ist nicht ersetzt --
sie wird exakt nachgerechnet.

FRAGESTELLUNG
1. Laesst sich die Schaltung end-to-end trainieren? (Gradienten durch
   Superposition und Verschraenkung hindurch)
2. Wie verhaelt sich die Genauigkeit gegen klassische Baselines mit
   *derselben* Parameterzahl?
3. Traegt die Verschraenkung etwas bei? (Kontrolle ohne CNOT-Gatter)

BASELINES
  hybrid            Encoder -> Quantenschaltung -> Kopf
  hybrid_noent      identisch, aber ohne CNOT: Superposition ohne Verschraenkung
                    -- isoliert genau den Beitrag der Verschraenkung
  classical_linear  Encoder -> lineare Abbildung 2^n -> n -> Kopf; die
                    natuerliche klassische Entsprechung (mit mehr Parametern)
  classical_dense   Encoder -> volle dichte Schicht -> Kopf (Obergrenze)

`hybrid_noent` ist die entscheidende Kontrolle: Verschraenkung ist die
Eigenschaft, die klassisch nicht effizient darstellbar ist. Bringt sie nichts,
ist der Quantenanteil auch hier nur Dekoration.

WAS NICHT BEHAUPTET WIRD
Kein Geschwindigkeitsvorteil. Die Simulation der Schaltung ist auf klassischer
Hardware teurer als eine Matrixmultiplikation. Gemessen wird Qualitaet pro
Parameter, nicht pro Sekunde.

Run: python -m quantum_sim.hybrid_experiment
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset

from .quantum_layer import HybridModel, QuantumLayer


class ClassicalLinear(nn.Module):
    """Encoder -> einfache lineare Abbildung 2^n -> n -> Kopf.

    Das ist die natuerliche klassische Entsprechung der Quantenschicht: sie
    leistet dieselbe Abbildung (2^n Merkmale auf n Ausgaben). Ein exakter
    Parameterabgleich ist nicht moeglich -- schon die kleinste lineare
    Abbildung 2^n -> n braucht mehr Parameter als die Schaltung mit 2*n*L
    Winkeln. Der klassische Arm bekommt hier also *mehr* Budget, der Vergleich
    ist damit konservativ gegen die Quantenschaltung.

    Ein Flaschenhals-Aufbau mit exakt passender Parameterzahl waere unfair aus
    dem umgekehrten Grund: er landet bei Rang 1, und A-003 hat gezeigt, dass
    solche Schichten das Signal vollstaendig zerstoeren -- das waere eine
    kaputte Kontrolle, kein strenger Vergleich.
    """

    def __init__(self, input_dim: int, encoded_dim: int, out_features: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoded_dim)
        self.middle = nn.Sequential(nn.Linear(encoded_dim, out_features), nn.Tanh())
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x):
        return self.head(self.middle(self.encoder(x)))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


class ClassicalDense(nn.Module):
    def __init__(self, input_dim: int, encoded_dim: int, out_features: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoded_dim)
        self.middle = nn.Sequential(nn.Linear(encoded_dim, encoded_dim), nn.Tanh(),
                                    nn.Linear(encoded_dim, out_features), nn.Tanh())
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x):
        return self.head(self.middle(self.encoder(x)))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


def build(arm: str, input_dim: int, num_classes: int, num_qubits: int, num_layers: int):
    encoded = 2 ** num_qubits
    if arm == "hybrid":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes, entangle=True)
    if arm == "hybrid_noent":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes, entangle=False)
    if arm == "classical_linear":
        return ClassicalLinear(input_dim, encoded, num_qubits, num_classes)
    if arm == "classical_dense":
        return ClassicalDense(input_dim, encoded, num_qubits, num_classes)
    raise ValueError(arm)


def middle_params(model) -> int:
    if isinstance(model, HybridModel):
        return model.quantum.num_circuit_parameters()
    return model.middle_parameters()


def train(model, x, y, epochs: int, lr: float, batch_size: int, seed: int) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    start = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return time.perf_counter() - start


ARMS = ["hybrid", "hybrid_noent", "classical_linear", "classical_dense"]


def run(
    num_qubits: int = 8,
    num_layers: int = 5,
    seeds: Sequence[int] = tuple(range(1301, 1309)),
    max_samples: int = 4000,
    epochs: int = 12,
    lr: float = 0.01,
    batch_size: int = 128,
    threads: int = 6,
) -> Dict[str, object]:
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    secs: Dict[str, List[float]] = {}

    for seed in seeds:
        for arm in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = build(arm, bundle.input_dim, bundle.num_classes, num_qubits, num_layers)
            duration = train(model, bundle.x_train, bundle.y_train, epochs, lr, batch_size, seed)
            model.eval()
            with torch.no_grad():
                pred = model(bundle.x_test).argmax(-1)
                a = float((pred == bundle.y_test).float().mean())
            acc.setdefault(arm, []).append(a)
            secs.setdefault(arm, []).append(duration)
            params[arm] = middle_params(model)
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params, "secs": secs,
            "qubits": num_qubits, "layers": num_layers}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]     # type: ignore[assignment]
    params: Dict[str, int] = result["params"]       # type: ignore[assignment]
    secs: Dict[str, List[float]] = result["secs"]   # type: ignore[assignment]
    n = len(acc["hybrid"])
    nq, nl = result["qubits"], result["layers"]

    print()
    print(f"RESEARCH ID Q-001 | MNIST, {nq} Qubits, {nl} Lagen, {n} Seeds")
    header = f'{"arm":>20} {"accuracy":>9} {"std":>7} {"Mittelschicht":>14} {"Zeit/Lauf":>11}'
    print(header); print("-" * len(header))
    for arm in ARMS:
        a = np.array(acc[arm]); s = np.array(secs[arm])
        print(f'{arm:>20} {a.mean():>9.4f} {a.std(ddof=1):>7.4f} {params[arm]:>14,} {s.mean():>10.1f}s')

    hyb = np.array(acc["hybrid"])
    noent = np.array(acc["hybrid_noent"])
    linear = np.array(acc["classical_linear"])
    dense = np.array(acc["classical_dense"])

    print()
    print("Traegt die Verschraenkung etwas bei?")
    p = stats.wilcoxon(hyb, noent).pvalue
    print(f'  hybrid vs hybrid_noent: {hyb.mean()-noent.mean():+.4f}, p={p:.4f}'
          + (" *signifikant*" if p < 0.05 else " (nicht signifikant)"))

    print()
    print("Gegen klassische Verfahren:")
    for label, other in (("classical_linear", linear), ("classical_dense", dense)):
        p = stats.wilcoxon(hyb, other).pvalue
        print(f'  hybrid vs {label:>18}: {hyb.mean()-other.mean():+.4f}, p={p:.4f}'
              + (" *signifikant*" if p < 0.05 else " (nicht signifikant)"))

    print()
    print(f'Parameterverhaeltnis: Schaltung {params["hybrid"]:,} gegen dichte Entsprechung '
          f'{2**nq * 2**nq:,} -> Faktor {(2**nq * 2**nq)/params["hybrid"]:,.0f}')


if __name__ == "__main__":
    analyse(run())
