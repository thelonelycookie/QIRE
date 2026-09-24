from __future__ import annotations

"""Parametrisierte Quantenschaltung als trainierbare PyTorch-Schicht.

Dies ist der Punkt, an dem KI und Quantenmechanik tatsaechlich zusammenkommen
-- nicht als Analogie. In der Schaltung findet echte Superposition statt, die
CNOT-Gatter erzeugen echte Verschraenkung, und die Ausgabe entsteht durch
Messung im Sinne der Born-Regel.

AUFBAU
    klassischer Encoder  ->  Amplituden-Kodierung  ->  variationelle Schaltung
                         ->  <Z>-Messung je Qubit  ->  klassischer Kopf

Die Schaltung folgt dem ueblichen "hardware-efficient ansatz":
abwechselnd eine Lage einzelner Rotationen (Ry, Rz) auf jedem Qubit und eine
Lage CNOT-Gatter in Ringtopologie, die Verschraenkung aufbaut.

PARAMETERZAHL -- die nachrechenbare Effizienzaussage
Eine Schaltung mit n Qubits und L Lagen hat 2*n*L trainierbare Winkel, wirkt
aber auf einen 2^n-dimensionalen Zustandsraum. Bei n=10, L=5 sind das 100
Parameter fuer einen 1024-dimensionalen Raum; eine klassische dichte Schicht
gleicher Breite braeuchte 1024*1024 = 1.048.576. Das ist keine Analogie,
sondern eine Abzaehlung.

WAS DIESER ANSATZ NICHT LEISTET -- vorab und ausdruecklich
Auf klassischer Hardware ist die Simulation der Schaltung *teurer* als eine
gewoehnliche Matrixmultiplikation, weil 2^n Amplituden fortgeschrieben werden.
Ein Geschwindigkeitsvorteil ist hier ausgeschlossen und wird nicht behauptet.
Was gemessen wird, ist das Verhaeltnis von Modellqualitaet zu Parameterzahl.
"""

import math
from typing import List

import torch
from torch import nn

from .torch_simulator import CNOT, TorchStatevector, ry, rz


class QuantumLayer(nn.Module):
    """Variationelle Quantenschaltung, differenzierbar und end-to-end trainierbar.

    Eingabe : (batch, 2**num_qubits) klassische Merkmale -> Amplituden-Kodierung
    Ausgabe : (batch, num_qubits) Erwartungswerte <Z> je Qubit
    """

    def __init__(self, num_qubits: int, num_layers: int = 3, entangle: bool = True):
        super().__init__()
        self.n = num_qubits
        self.L = num_layers
        self.entangle = entangle
        # zwei Rotationswinkel (Ry, Rz) je Qubit und Lage
        self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)

    @property
    def input_dim(self) -> int:
        return 2 ** self.n

    def num_circuit_parameters(self) -> int:
        return self.theta.numel()

    def dense_equivalent_parameters(self) -> int:
        """Parameterzahl einer klassischen dichten Schicht gleicher Breite --
        die Vergleichsgroesse fuer die Kompressionsaussage."""
        d = 2 ** self.n
        return d * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = TorchStatevector.from_amplitudes(x, self.n)
        for layer in range(self.L):
            for q in range(self.n):
                state.apply(ry(self.theta[layer, q, 0]), [q])
                state.apply(rz(self.theta[layer, q, 1]), [q])
            if self.entangle and self.n > 1:
                # Ringtopologie: verschraenkt alle Qubits ueber die Lagen hinweg
                for q in range(self.n):
                    state.apply(CNOT, [q, (q + 1) % self.n])
        return state.expectation_z()


class HybridModel(nn.Module):
    """klassischer Encoder -> Quantenschaltung -> klassischer Kopf.

    Der Encoder bringt die Eingabe auf 2**num_qubits Dimensionen, damit sie
    als Amplituden kodiert werden kann. Der Kopf liest die n Erwartungswerte
    aus. Dazwischen rechnet echte Quantenmechanik.
    """

    model_name = "HybridQuantumModel"

    def __init__(self, input_dim: int, num_qubits: int = 8, num_layers: int = 3,
                 num_classes: int = 10, entangle: bool = True):
        super().__init__()
        self.quantum = QuantumLayer(num_qubits, num_layers, entangle=entangle)
        self.encoder = nn.Linear(input_dim, self.quantum.input_dim)
        self.head = nn.Linear(num_qubits, num_classes)
        self.method = f"hybrid_q{num_qubits}_l{num_layers}" + ("" if entangle else "_noent")

    def forward(self, x):
        encoded = self.encoder(x)
        measured = self.quantum(encoded)
        return self.head(measured)

    def parameter_breakdown(self) -> dict:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "quantum": self.quantum.num_circuit_parameters(),
            "head": sum(p.numel() for p in self.head.parameters()),
            "quantum_dense_equivalent": self.quantum.dense_equivalent_parameters(),
        }
