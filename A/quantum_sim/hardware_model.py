from __future__ import annotations

"""RESEARCH ID: Q-004 -- In Simulation trainiert, auf echter Quantenhardware ausgefuehrt.

ZIEL
Den Kreis schliessen: ein KI-Modell, dessen Quantenteil in Simulation gelernt
wird und dessen gelernte Winkel anschliessend auf einem echten
IBM-Quantenprozessor ausgefuehrt werden. Verglichen werden die Vorhersagen aus
Simulation und Hardware.

WARUM WINKEL- STATT AMPLITUDEN-KODIERUNG
Das Modell aus Q-001 bis Q-003 nutzt Amplituden-Kodierung: ein 256-dimensionaler
Vektor wird als Zustand von 8 Qubits praepariert. Auf echter Hardware braeuchte
diese Zustandspraeparation hunderte Gatter -- bei einer gemessenen Fehlerrate
von rund 2 % je drei Gatter (siehe Hardware-Messungen weiter oben) waere das
Ergebnis reines Rauschen.

Winkel-Kodierung ist flach: jedes Merkmal wird zu genau einem Drehwinkel, die
Kodierung kostet also n Gatter statt 2^n. Das ist der uebliche Weg fuer echte
Hardware und macht den Vergleich ueberhaupt erst moeglich.

AUFBAU
    784 Pixel -> klassischer Encoder -> n Merkmale
              -> Ry(Merkmal) je Qubit            (Winkel-Kodierung)
              -> L Lagen: Ry/Rz je Qubit + CNOT-Ring   (gelernte Winkel)
              -> <Z> je Qubit
              -> klassischer Kopf -> Klassen

Trainiert wird vollstaendig in Simulation ueber den differenzierbaren
Simulator. Danach werden ausschliesslich die gelernten Winkel uebernommen und
in eine Qiskit-Schaltung geschrieben.

WAS GEMESSEN WIRD
  1. Genauigkeit in Simulation (Referenz)
  2. <Z>-Werte aus echter Hardware gegen die exakt berechneten
  3. Vorhersagen aus Hardware-<Z> gegen Simulation und gegen die wahren Labels

WAS NICHT BEHAUPTET WIRD
Kein Geschwindigkeits- oder Genauigkeitsvorteil durch die Hardware. Erwartet
wird eine Verschlechterung durch Rauschen. Der Zweck ist der Nachweis, dass das
System auf echter Quantenhardware laeuft, und die Quantifizierung des
Unterschieds.

Run: python -m quantum_sim.hardware_model
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset

from .torch_simulator import CNOT, TorchStatevector, ry, rz


class AngleEncodedQuantumLayer(nn.Module):
    """Winkel-Kodierung + variationelle Schaltung, differenzierbar.

    Eingabe : (batch, n) klassische Merkmale -> je ein Ry-Winkel pro Qubit
    Ausgabe : (batch, n) Erwartungswerte <Z>
    """

    def __init__(self, num_qubits: int, num_layers: int = 2):
        super().__init__()
        self.n = num_qubits
        self.L = num_layers
        self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        state = TorchStatevector(self.n, batch=batch)
        # Winkel-Kodierung: jedes Merkmal wird ein Drehwinkel.
        # Pro Beispiel unterschiedlich, deshalb einzeln angewandt.
        for q in range(self.n):
            angles = features[:, q]
            state.state = self._apply_batched_ry(state.state, angles, q)
        for layer in range(self.L):
            for q in range(self.n):
                state.apply(ry(self.theta[layer, q, 0]), [q])
                state.apply(rz(self.theta[layer, q, 1]), [q])
            for q in range(self.n):
                state.apply(CNOT, [q, (q + 1) % self.n])
        return state.expectation_z()

    def _apply_batched_ry(self, state: torch.Tensor, angles: torch.Tensor, qubit: int) -> torch.Tensor:
        """Ry mit einem eigenen Winkel je Batch-Element."""
        c = torch.cos(angles / 2).to(torch.complex64)
        s = torch.sin(angles / 2).to(torch.complex64)
        axis = qubit + 1
        moved = state.movedim(axis, -1)          # (..., 2)
        a, b = moved[..., 0], moved[..., 1]
        shape = [-1] + [1] * (moved.dim() - 2)
        c_b, s_b = c.reshape(shape), s.reshape(shape)
        new_a = c_b * a - s_b * b
        new_b = s_b * a + c_b * b
        return torch.stack([new_a, new_b], dim=-1).movedim(-1, axis)

    def circuit_parameters(self) -> int:
        return self.theta.numel()


class HardwareReadyModel(nn.Module):
    """Encoder -> Winkel-Kodierung -> Quantenschaltung -> Kopf."""

    model_name = "HardwareReadyModel"

    def __init__(self, input_dim: int, num_qubits: int = 4, num_layers: int = 2,
                 num_classes: int = 10):
        super().__init__()
        self.n = num_qubits
        self.encoder = nn.Linear(input_dim, num_qubits)
        self.quantum = AngleEncodedQuantumLayer(num_qubits, num_layers)
        self.head = nn.Linear(num_qubits, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Die Merkmale, die zu Drehwinkeln werden -- auf [-pi, pi] begrenzt,
        damit sie auf Hardware direkt als Winkel taugen."""
        return torch.tanh(self.encoder(x)) * math.pi

    def forward(self, x):
        return self.head(self.quantum(self.encode(x)))

    def learned_angles(self) -> np.ndarray:
        return self.quantum.theta.detach().cpu().numpy()


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


def build_qiskit_circuit(encoding_angles: np.ndarray, learned: np.ndarray, num_qubits: int):
    """Baut exakt dieselbe Schaltung fuer Qiskit.

    encoding_angles: (n,) die Winkel dieses einen Beispiels
    learned: (L, n, 2) die trainierten Winkel
    """
    from qiskit import QuantumCircuit

    num_layers = learned.shape[0]
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.ry(float(encoding_angles[q]), q)
    for layer in range(num_layers):
        for q in range(num_qubits):
            qc.ry(float(learned[layer, q, 0]), q)
            qc.rz(float(learned[layer, q, 1]), q)
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)
    qc.measure_all()
    return qc


def expectation_z_from_counts(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    """<Z> je Qubit aus gemessenen Bitstrings.

    Qiskit schreibt Qubit 0 als RECHTESTES Zeichen -- diese Konvention war
    bereits beim Simulator-Vergleich aufgefallen und wird hier explizit
    behandelt, statt sie anzunehmen.
    """
    total = sum(counts.values())
    z = np.zeros(num_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(num_qubits):
            bit = bits[-(q + 1)]
            z[q] += count * (1.0 if bit == "0" else -1.0)
    return z / total


def run_simulation(num_qubits: int = 4, num_layers: int = 2, seed: int = 1601,
                   max_samples: int = 4000, epochs: int = 15, lr: float = 0.01,
                   batch_size: int = 128, threads: int = 6):
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
    model = HardwareReadyModel(bundle.input_dim, num_qubits, num_layers, bundle.num_classes)
    train(model, bundle.x_train, bundle.y_train, epochs, lr, batch_size, seed)
    model.eval()
    with torch.no_grad():
        pred = model(bundle.x_test).argmax(-1)
        accuracy = float((pred == bundle.y_test).float().mean())
    return model, bundle, accuracy


if __name__ == "__main__":
    print("Trainiere das hardwaretaugliche Modell in Simulation ...")
    model, bundle, accuracy = run_simulation()
    print(f"  Genauigkeit in Simulation: {accuracy:.4f}")
    print(f"  Schaltungsparameter: {model.quantum.circuit_parameters()}")
    print(f"  Qubits: {model.n}, Lagen: {model.quantum.L}")
    print()
    print("Gelernte Winkel (Bogenmass):")
    print(np.round(model.learned_angles(), 3))
