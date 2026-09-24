from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class StateVectorSimulator:
    """From-scratch state-vector quantum simulator.

    A state of n qubits is 2**n complex amplitudes, stored here as a rank-n
    tensor of shape (2,)*n rather than a flat vector -- axis i holds qubit i.
    Applying a k-qubit gate is a tensor contraction (np.tensordot) over just
    that gate's target axes, then moving the result's axes back to their
    qubit positions. This avoids ever building a 2**n x 2**n matrix, which is
    the standard technique real simulators (e.g. Qiskit's Statevector) use
    too. It reproduces the exact, noiseless mathematics of a quantum circuit;
    it does not and cannot show a quantum speedup, since it runs entirely on
    classical hardware.

    Qubit-index convention: qubit 0 is tensor axis 0. This does NOT
    necessarily match how other tools print bitstrings (Qiskit, for
    instance, prints qubit 0 as the rightmost bit) -- check convention
    explicitly before comparing amplitudes/counts against another framework.
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.n = num_qubits
        self.state = np.zeros((2,) * self.n, dtype=complex)
        self.state[(0,) * self.n] = 1.0  # |00...0>

    def apply_gate(self, gate: np.ndarray, targets: Sequence[int]) -> None:
        targets = list(targets)
        k = len(targets)
        expected_dim = 2 ** k
        if gate.shape != (expected_dim, expected_dim):
            raise ValueError(f"gate for {k} qubit(s) must be {expected_dim}x{expected_dim}, got {gate.shape}")
        if len(set(targets)) != k or any(t < 0 or t >= self.n for t in targets):
            raise ValueError(f"invalid/duplicate target qubits: {targets}")

        gate_tensor = gate.reshape([2] * k + [2] * k)
        input_axes = list(range(k, 2 * k))
        result = np.tensordot(gate_tensor, self.state, axes=(input_axes, targets))
        # result's axes: [0..k-1] are the new values for `targets` (in that order),
        # [k..n-1] are the untouched qubits, in their original relative order.
        remaining = [ax for ax in range(self.n) if ax not in targets]
        perm = [
            targets.index(i) if i in targets else k + remaining.index(i)
            for i in range(self.n)
        ]
        self.state = np.transpose(result, perm)

    def statevector(self) -> np.ndarray:
        return self.state.reshape(-1).copy()

    def probabilities(self) -> np.ndarray:
        return np.abs(self.statevector()) ** 2

    def norm(self) -> float:
        """Should stay 1.0 (up to floating point error) after any unitary gate sequence."""
        return float(np.sum(self.probabilities()))

    def bitstring(self, index: int) -> str:
        return format(index, f"0{self.n}b")

    def measure_all(self, shots: int = 1000, seed: Optional[int] = None) -> Dict[str, int]:
        """Sample `shots` measurements from the current state (Born rule), without
        collapsing the stored state -- useful for reading out a distribution
        without disturbing further circuit-building/inspection."""
        probs = self.probabilities()
        probs = probs / probs.sum()  # guard against float drift
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(len(probs), size=shots, p=probs)
        counts: Dict[str, int] = {}
        for idx in outcomes:
            key = self.bitstring(int(idx))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def print_state(self, tol: float = 1e-9) -> None:
        for idx, amp in enumerate(self.statevector()):
            if abs(amp) > tol:
                print(f"|{self.bitstring(idx)}>: {amp:.4f}  (p={abs(amp)**2:.4f})")
