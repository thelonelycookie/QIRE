from __future__ import annotations

"""Differenzierbarer Zustandsvektor-Simulator in PyTorch.

Der bestehende Simulator (`simulator.py`) rechnet mit NumPy und ist damit
nicht differenzierbar -- gut zum Verstehen und Validieren, aber nicht
trainierbar. Diese Fassung nutzt komplexe PyTorch-Tensoren, sodass Autograd
durch die gesamte Quantenschaltung hindurch funktioniert. Damit lassen sich
Rotationswinkel als Modellparameter lernen.

Konventionen sind identisch zu `simulator.py`:
  - Qubit i entspricht Tensorachse i
  - Zustand als Tensor der Form (batch, 2, 2, ..., 2), also eine Achse je Qubit
  - Gatter werden per Tensorkontraktion nur auf den betroffenen Achsen
    angewandt, es wird nie eine volle 2^n x 2^n Matrix gebildet

Der Batch wird als fuehrende Achse mitgefuehrt, damit ein ganzer
Datenstapel gleichzeitig durch dieselbe Schaltung laeuft.

Verifiziert gegen `simulator.py` in `quantum_sim/validate_torch.py`.
"""

import math
from typing import Sequence

import torch


def _c(dtype: torch.dtype = torch.complex64) -> torch.dtype:
    return dtype


class TorchStatevector:
    """Batch von Quantenzustaenden, differenzierbar.

    state hat die Form (batch, 2, 2, ..., 2) mit n Zweier-Achsen.
    """

    def __init__(self, num_qubits: int, batch: int = 1, dtype: torch.dtype = torch.complex64,
                 device: torch.device | None = None):
        self.n = num_qubits
        self.dtype = dtype
        shape = (batch,) + (2,) * num_qubits
        state = torch.zeros(shape, dtype=dtype, device=device)
        state.reshape(batch, -1)[:, 0] = 1.0  # alle Qubits in |0>
        self.state = state

    @classmethod
    def from_amplitudes(cls, amplitudes: torch.Tensor, num_qubits: int) -> "TorchStatevector":
        """Amplituden-Kodierung: ein klassischer Vektor wird normiert und als
        Zustand interpretiert. Das ist die uebliche Art, klassische Daten in
        einen Quantenzustand zu ueberfuehren."""
        batch = amplitudes.shape[0]
        dim = 2 ** num_qubits
        if amplitudes.shape[1] < dim:
            pad = torch.zeros(batch, dim - amplitudes.shape[1],
                              dtype=amplitudes.dtype, device=amplitudes.device)
            amplitudes = torch.cat([amplitudes, pad], dim=1)
        amplitudes = amplitudes[:, :dim]
        norm = amplitudes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        normalised = (amplitudes / norm).to(torch.complex64)
        obj = cls.__new__(cls)
        obj.n = num_qubits
        obj.dtype = torch.complex64
        obj.state = normalised.reshape((batch,) + (2,) * num_qubits)
        return obj

    def apply(self, gate: torch.Tensor, targets: Sequence[int]) -> "TorchStatevector":
        """Gatter auf die angegebenen Qubits anwenden. Achse 0 ist der Batch,
        deshalb liegt Qubit i auf Achse i+1."""
        targets = list(targets)
        k = len(targets)
        axes = [t + 1 for t in targets]
        gate_tensor = gate.reshape([2] * k + [2] * k)
        result = torch.tensordot(gate_tensor, self.state, dims=(list(range(k, 2 * k)), axes))
        # result: (neue Ziel-Achsen..., batch, uebrige Achsen...)
        remaining = [ax for ax in range(1, self.n + 1) if ax not in axes]
        perm = [k]  # batch zuerst
        for ax in range(1, self.n + 1):
            perm.append(axes.index(ax) if ax in axes else k + 1 + remaining.index(ax))
        self.state = result.permute(*perm)
        return self

    def probabilities(self) -> torch.Tensor:
        """(batch, 2^n) Messwahrscheinlichkeiten nach der Born-Regel."""
        flat = self.state.reshape(self.state.shape[0], -1)
        return (flat.real ** 2 + flat.imag ** 2)

    def expectation_z(self) -> torch.Tensor:
        """(batch, n) Erwartungswerte <Z> je Qubit -- der uebliche Weg, aus
        einem Quantenzustand reelle Merkmale fuer eine klassische Schicht zu
        gewinnen, ohne den vollen Zustand auslesen zu muessen."""
        probs = self.probabilities()
        batch = probs.shape[0]
        probs = probs.reshape((batch,) + (2,) * self.n)
        out = []
        for q in range(self.n):
            axes = [a for a in range(1, self.n + 1) if a != q + 1]
            marginal = probs.sum(dim=axes) if axes else probs
            out.append(marginal[:, 0] - marginal[:, 1])   # P(0) - P(1)
        return torch.stack(out, dim=1)


# ---- Gatter als differenzierbare Funktionen der Parameter -------------------

def rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([c, zero]), torch.stack([zero, c]),
    ]).to(torch.complex64) + 1j * torch.stack([
        torch.stack([zero, -s]), torch.stack([-s, zero]),
    ]).to(torch.complex64)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([
        torch.stack([c, -s]), torch.stack([s, c]),
    ]).to(torch.complex64)


def rz(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2
    cos, sin = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(cos)
    real = torch.stack([torch.stack([cos, zero]), torch.stack([zero, cos])])
    imag = torch.stack([torch.stack([-sin, zero]), torch.stack([zero, sin])])
    return real.to(torch.complex64) + 1j * imag.to(torch.complex64)


CNOT = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=torch.complex64)
H = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64)
