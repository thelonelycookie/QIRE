from __future__ import annotations

"""RESEARCH ID: Q-002 -- Erklaert die Amplituden-Normierung den Vorteil aus Q-001?

BEFUND AUS Q-001
Das Hybridmodell (80 Schaltungsparameter) schlug eine dichte klassische Schicht
(67.848 Parameter) signifikant: +0,0546, p=0,0078. Verschraenkung trug dabei
nichts Messbares bei (+0,0039, p=0,141).

DER VERDACHT
Die Amplituden-Kodierung normiert den Eingabevektor auf Laenge 1, bevor er in
die Schaltung geht -- die klassischen Arme in Q-001 bekamen diese Normierung
nicht. L2-Normierung ist eine gewoehnliche klassische Vorverarbeitung. Wenn sie
den Vorteil erklaert, ist er nicht quantenspezifisch.

Ein zweiter Unterschied kommt hinzu: die Schaltungsausgabe sind
Erwartungswerte <Z>, die konstruktionsbedingt in [-1, 1] liegen. Die
klassischen Arme erhielten zwar ein Tanh, aber keine vorgeschaltete Normierung.

KONTROLLBEDINGUNGEN
  hybrid              wie in Q-001 (Referenz)
  hybrid_noent        ohne CNOT (Verschraenkungskontrolle, wie in Q-001)
  classical_norm      klassische Schicht MIT L2-Normierung der kodierten Merkmale
  classical_norm_deep dichte Variante MIT L2-Normierung
  classical_plain     ohne Normierung (entspricht Q-001, zur Einordnung)

HYPOTHESEN
  H1 : hybrid schlaegt auch die normierten klassischen Arme -> der Vorteil
       liegt nicht an der Normierung
  H0 : classical_norm zieht gleich -> der Vorteil war Normierung, nicht
       Quantenmechanik
  H2 : normierte Arme liegen dazwischen -> Normierung erklaert einen Teil, der
       Rest bleibt offen

FALSIFIKATION
Zieht ein normierter klassischer Arm gleich, ist der Q-001-Befund nicht als
Quantenvorteil haltbar.

Run: python -m quantum_sim.hybrid_control
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset

from .quantum_layer import HybridModel


class L2Normalise(nn.Module):
    """Genau die Normierung, die die Amplituden-Kodierung implizit vornimmt."""

    def forward(self, x):
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


class ClassicalNormed(nn.Module):
    """Encoder -> [L2-Normierung] -> klassische Schicht(en) -> Kopf."""

    def __init__(self, input_dim: int, encoded_dim: int, out_features: int,
                 num_classes: int, deep: bool, normalise: bool):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoded_dim)
        self.norm = L2Normalise() if normalise else nn.Identity()
        if deep:
            self.middle = nn.Sequential(nn.Linear(encoded_dim, encoded_dim), nn.Tanh(),
                                        nn.Linear(encoded_dim, out_features), nn.Tanh())
        else:
            self.middle = nn.Sequential(nn.Linear(encoded_dim, out_features), nn.Tanh())
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x):
        return self.head(self.middle(self.norm(self.encoder(x))))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


ARMS = ["hybrid", "hybrid_noent", "classical_norm", "classical_norm_deep", "classical_plain"]


def build(arm: str, input_dim: int, num_classes: int, num_qubits: int, num_layers: int):
    encoded = 2 ** num_qubits
    if arm == "hybrid":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes, entangle=True)
    if arm == "hybrid_noent":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes, entangle=False)
    if arm == "classical_norm":
        return ClassicalNormed(input_dim, encoded, num_qubits, num_classes, deep=False, normalise=True)
    if arm == "classical_norm_deep":
        return ClassicalNormed(input_dim, encoded, num_qubits, num_classes, deep=True, normalise=True)
    if arm == "classical_plain":
        return ClassicalNormed(input_dim, encoded, num_qubits, num_classes, deep=True, normalise=False)
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
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return time.perf_counter() - start


def run(
    num_qubits: int = 8,
    num_layers: int = 5,
    seeds: Sequence[int] = tuple(range(1401, 1409)),
    max_samples: int = 4000,
    epochs: int = 12,
    lr: float = 0.01,
    batch_size: int = 128,
    threads: int = 6,
) -> Dict[str, object]:
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    for seed in seeds:
        for arm in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = build(arm, bundle.input_dim, bundle.num_classes, num_qubits, num_layers)
            train(model, bundle.x_train, bundle.y_train, epochs, lr, batch_size, seed)
            model.eval()
            with torch.no_grad():
                pred = model(bundle.x_test).argmax(-1)
                acc.setdefault(arm, []).append(float((pred == bundle.y_test).float().mean()))
            params[arm] = middle_params(model)
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params, "qubits": num_qubits, "layers": num_layers}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]   # type: ignore[assignment]
    params: Dict[str, int] = result["params"]     # type: ignore[assignment]
    n = len(acc["hybrid"])

    desc = {
        "hybrid": "Quantenschaltung",
        "hybrid_noent": "Schaltung ohne CNOT",
        "classical_norm": "klassisch + L2-Normierung",
        "classical_norm_deep": "klassisch tief + L2-Normierung",
        "classical_plain": "klassisch tief, ohne Normierung",
    }
    print()
    print(f"RESEARCH ID Q-002 | MNIST, {result['qubits']} Qubits, {n} Seeds")
    header = f'{"arm":>22} {"Beschreibung":>32} {"accuracy":>9} {"std":>7} {"Params":>9}'
    print(header); print("-" * len(header))
    for arm in ARMS:
        v = np.array(acc[arm])
        print(f'{arm:>22} {desc[arm]:>32} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} {params[arm]:>9,}')

    hyb = np.array(acc["hybrid"])
    print()
    print("ENTSCHEIDEND -- haelt der Vorteil gegen NORMIERTE klassische Arme?")
    survives = True
    for arm in ["classical_norm", "classical_norm_deep"]:
        other = np.array(acc[arm])
        p = stats.wilcoxon(hyb, other).pvalue
        if not (hyb.mean() > other.mean() and p < 0.05):
            survives = False
        print(f'  hybrid vs {arm:>22}: {hyb.mean()-other.mean():+.4f}, p={p:.4f}'
              + (" *signifikant*" if p < 0.05 else " (nicht signifikant)"))

    plain = np.array(acc["classical_plain"])
    norm_deep = np.array(acc["classical_norm_deep"])
    print()
    print("Wieviel erklaert die Normierung allein?")
    p_norm = stats.wilcoxon(norm_deep, plain).pvalue
    print(f'  classical_norm_deep vs classical_plain: {norm_deep.mean()-plain.mean():+.4f}, '
          f'p={p_norm:.4f}' + (" *signifikant*" if p_norm < 0.05 else " (nicht signifikant)"))

    print()
    if survives:
        print("  -> Der Vorteil haelt auch gegen normierte klassische Arme.")
        print("     Die Normierung erklaert ihn nicht.")
    else:
        print("  -> Mindestens ein normierter klassischer Arm zieht gleich.")
        print("     Der Q-001-Befund ist dann Normierung, nicht Quantenmechanik.")


if __name__ == "__main__":
    analyse(run())
