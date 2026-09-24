from __future__ import annotations

"""RESEARCH ID: Q-003 -- Laesst sich der Vorteil klassisch nachbauen?

STAND NACH Q-001 UND Q-002
Die Quantenschaltung (80 Parameter) schlaegt klassische Arme mit bis zu 67.848
Parametern signifikant (+0,0707, p=0,0078). L2-Normierung erklaert den Vorteil
nicht (p=0,55). Verschraenkung ebenfalls nicht -- die Variante ohne CNOT ist
fast gleichauf.

Was die Schaltung tatsaechlich tut, laesst sich in drei Schritte zerlegen:
  1. Amplituden-Kodierung   -> L2-Normierung des Eingabevektors
  2. unitaere Entwicklung   -> normerhaltende Drehung im 2^n-dim Raum,
                               parametrisiert durch nur 2*n*L Winkel
  3. <Z>-Messung            -> QUADRATISCHE Auslese, denn <Z>_q ist eine
                               Summe ueber |Amplitude|^2

Schritt 3 ist der bisher uebersehene Punkt: die klassischen Kontrollen
rechneten Linear -> Tanh, also eine ganz andere Funktionsklasse. Die
quadratische Auslese ist eine Nichtlinearitaet, die dort schlicht nicht
vorkam. Der Vorteil koennte daher schlicht daran liegen -- und Quadrieren ist
nichts spezifisch Quantenmechanisches.

KONTROLLBEDINGUNGEN
  hybrid              Quantenschaltung (Referenz)
  classical_quadratic normieren -> ORTHOGONALE Matrix -> quadrieren ->
                      gruppenweise summieren. Baut alle drei Schritte klassisch
                      nach, ohne Quantenformalismus.
  classical_quad_free wie oben, aber die Matrix ist FREI (nicht orthogonal) --
                      isoliert, ob die Normerhaltung noetig ist
  classical_norm_deep starke klassische Referenz aus Q-002 (Linear/Tanh)

Die orthogonale Matrix wird als expm(A - A^T) parametrisiert. Um den
Parametervergleich fair zu halten, ist sie blockweise aufgebaut: 2^n ist zu
gross fuer eine volle orthogonale Matrix (bei n=8 waeren das 32.640 freie
Parameter gegenueber 80 in der Schaltung). Die Bloecke werden so dimensioniert,
dass die Parameterzahl in derselben Groessenordnung liegt.

HYPOTHESEN
  H1 : hybrid schlaegt classical_quadratic -> die spezifische
       Schaltungsstruktur traegt ueber die drei Schritte hinaus
  H0 : classical_quadratic zieht gleich -> der Wirkmechanismus ist klassisch
       vollstaendig beschreibbar (normieren + orthogonal + quadrieren)

EINORDNUNG VORAB
Auch bei H0 bleibt ein belastbarer Befund: die *Architektur* stammt aus der
Quantenschaltungstheorie. 2*n*L Parameter fuer eine unitaere Abbildung auf
2^n Dimensionen ist keine Konstruktion, auf die man klassisch naheliegend
kaeme. Das waere dann ein quantenabgeleiteter Aufbau mit klassisch erklaerbarem
Mechanismus -- ein ehrliches und weiterhin verwertbares Ergebnis.

Run: python -m quantum_sim.hybrid_final_control
"""

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset

from .quantum_layer import HybridModel


class QuadraticReadout(nn.Module):
    """normieren -> blockweise (orthogonale) Drehung -> quadrieren -> summieren.

    Das ist die klassische Nachbildung dessen, was die Schaltung leistet:
    Schritt 1 die Normierung, Schritt 2 eine normerhaltende Transformation,
    Schritt 3 die quadratische Auslese mit gruppenweiser Summation, genau wie
    <Z>_q ueber die Amplituden summiert.
    """

    def __init__(self, dim: int, out_features: int, block_size: int = 8,
                 orthogonal: bool = True):
        super().__init__()
        assert dim % block_size == 0
        self.dim, self.block_size, self.out_features = dim, block_size, out_features
        self.blocks = dim // block_size
        self.orthogonal = orthogonal
        self.weight = nn.Parameter(torch.empty(self.blocks, block_size, block_size))
        nn.init.xavier_uniform_(self.weight)
        # Zuordnung der 2^n Quadrate auf out_features Gruppen, wie <Z> je Qubit
        self.register_buffer("group", torch.arange(dim) % out_features)

    def _matrices(self) -> torch.Tensor:
        if not self.orthogonal:
            return self.weight
        skew = self.weight - self.weight.transpose(-1, -2)
        return torch.matrix_exp(skew)

    def forward(self, x):
        batch = x.shape[0]
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)      # Schritt 1
        h = x.reshape(batch, self.blocks, self.block_size)
        h = torch.bmm(h.transpose(0, 1), self._matrices()).transpose(0, 1)
        h = h.reshape(batch, self.dim)                             # Schritt 2
        sq = h ** 2                                                # Schritt 3
        out = torch.zeros(batch, self.out_features, device=x.device, dtype=x.dtype)
        out.index_add_(1, self.group, sq)
        return out * 2.0 - 1.0   # auf [-1, 1] bringen, wie <Z>


class ClassicalQuadratic(nn.Module):
    def __init__(self, input_dim: int, encoded_dim: int, out_features: int,
                 num_classes: int, orthogonal: bool):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoded_dim)
        self.middle = QuadraticReadout(encoded_dim, out_features, orthogonal=orthogonal)
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x):
        return self.head(self.middle(self.encoder(x)))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


class ClassicalDeepNormed(nn.Module):
    def __init__(self, input_dim: int, encoded_dim: int, out_features: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoded_dim)
        self.middle = nn.Sequential(nn.Linear(encoded_dim, encoded_dim), nn.Tanh(),
                                    nn.Linear(encoded_dim, out_features), nn.Tanh())
        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x):
        e = self.encoder(x)
        e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.head(self.middle(e))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


ARMS = ["hybrid", "classical_quadratic", "classical_quad_free", "classical_norm_deep"]


def build(arm: str, input_dim: int, num_classes: int, num_qubits: int, num_layers: int):
    encoded = 2 ** num_qubits
    if arm == "hybrid":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes, entangle=True)
    if arm == "classical_quadratic":
        return ClassicalQuadratic(input_dim, encoded, num_qubits, num_classes, orthogonal=True)
    if arm == "classical_quad_free":
        return ClassicalQuadratic(input_dim, encoded, num_qubits, num_classes, orthogonal=False)
    if arm == "classical_norm_deep":
        return ClassicalDeepNormed(input_dim, encoded, num_qubits, num_classes)
    raise ValueError(arm)


def middle_params(model) -> int:
    if isinstance(model, HybridModel):
        return model.quantum.num_circuit_parameters()
    return model.middle_parameters()


def train(model, x, y, epochs: int, lr: float, batch_size: int, seed: int):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def run(
    num_qubits: int = 8,
    num_layers: int = 5,
    seeds: Sequence[int] = tuple(range(1501, 1509)),
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
    return {"acc": acc, "params": params, "qubits": num_qubits}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]   # type: ignore[assignment]
    params: Dict[str, int] = result["params"]     # type: ignore[assignment]
    n = len(acc["hybrid"])
    desc = {
        "hybrid": "Quantenschaltung",
        "classical_quadratic": "normieren+orthogonal+quadrieren",
        "classical_quad_free": "wie oben, Matrix frei",
        "classical_norm_deep": "Linear/Tanh + Normierung",
    }

    print()
    print(f"RESEARCH ID Q-003 | MNIST, {result['qubits']} Qubits, {n} Seeds")
    header = f'{"arm":>22} {"Beschreibung":>34} {"accuracy":>9} {"std":>7} {"Params":>9}'
    print(header); print("-" * len(header))
    for arm in ARMS:
        v = np.array(acc[arm])
        print(f'{arm:>22} {desc[arm]:>34} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} {params[arm]:>9,}')

    hyb = np.array(acc["hybrid"])
    print()
    print("ENTSCHEIDEND -- laesst sich die Schaltung klassisch nachbauen?")
    survives = True
    for arm in ["classical_quadratic", "classical_quad_free"]:
        other = np.array(acc[arm])
        p = stats.wilcoxon(hyb, other).pvalue
        if not (hyb.mean() > other.mean() and p < 0.05):
            survives = False
        print(f'  hybrid vs {arm:>22}: {hyb.mean()-other.mean():+.4f}, p={p:.4f}'
              + (" *signifikant*" if p < 0.05 else " (nicht signifikant)"))

    quad = np.array(acc["classical_quadratic"])
    deep = np.array(acc["classical_norm_deep"])
    p_q = stats.wilcoxon(quad, deep).pvalue
    print()
    print("Wieviel erklaert die quadratische Auslese allein?")
    print(f'  classical_quadratic vs classical_norm_deep: {quad.mean()-deep.mean():+.4f}, '
          f'p={p_q:.4f}' + (" *signifikant*" if p_q < 0.05 else " (nicht signifikant)"))

    print()
    if survives:
        print("  -> Die Schaltung schlaegt auch ihre klassische Nachbildung.")
        print("     Die spezifische Struktur traegt ueber die drei Schritte hinaus.")
    else:
        print("  -> Die klassische Nachbildung zieht gleich.")
        print("     Der Wirkmechanismus ist klassisch vollstaendig beschreibbar:")
        print("     normieren + normerhaltende Drehung + quadratische Auslese.")
        print("     Die ARCHITEKTUR bleibt quantenabgeleitet, der MECHANISMUS ist es nicht.")


if __name__ == "__main__":
    analyse(run())
