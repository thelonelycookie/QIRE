from __future__ import annotations

from typing import Dict, Optional

from . import gates
from .simulator import StateVectorSimulator


class Circuit:
    """Fluent builder over StateVectorSimulator, e.g. Circuit(2).h(0).cnot(0, 1)."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.sim = StateVectorSimulator(num_qubits)

    def _apply(self, gate, targets) -> "Circuit":
        self.sim.apply_gate(gate, targets)
        return self

    def i(self, q: int) -> "Circuit":
        return self._apply(gates.I, [q])

    def x(self, q: int) -> "Circuit":
        return self._apply(gates.X, [q])

    def y(self, q: int) -> "Circuit":
        return self._apply(gates.Y, [q])

    def z(self, q: int) -> "Circuit":
        return self._apply(gates.Z, [q])

    def h(self, q: int) -> "Circuit":
        return self._apply(gates.H, [q])

    def s(self, q: int) -> "Circuit":
        return self._apply(gates.S, [q])

    def t(self, q: int) -> "Circuit":
        return self._apply(gates.T, [q])

    def rx(self, q: int, theta: float) -> "Circuit":
        return self._apply(gates.Rx(theta), [q])

    def ry(self, q: int, theta: float) -> "Circuit":
        return self._apply(gates.Ry(theta), [q])

    def rz(self, q: int, theta: float) -> "Circuit":
        return self._apply(gates.Rz(theta), [q])

    def phase(self, q: int, theta: float) -> "Circuit":
        return self._apply(gates.Phase(theta), [q])

    def cnot(self, control: int, target: int) -> "Circuit":
        return self._apply(gates.CNOT, [control, target])

    def cz(self, q1: int, q2: int) -> "Circuit":
        return self._apply(gates.CZ, [q1, q2])

    def swap(self, q1: int, q2: int) -> "Circuit":
        return self._apply(gates.SWAP, [q1, q2])

    def toffoli(self, control1: int, control2: int, target: int) -> "Circuit":
        return self._apply(gates.TOFFOLI, [control1, control2, target])

    def statevector(self):
        return self.sim.statevector()

    def probabilities(self):
        return self.sim.probabilities()

    def print_state(self) -> "Circuit":
        self.sim.print_state()
        return self

    def measure_all(self, shots: int = 1000, seed: Optional[int] = None) -> Dict[str, int]:
        return self.sim.measure_all(shots=shots, seed=seed)
