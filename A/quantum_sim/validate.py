"""Checks the simulator against circuits whose theoretical outcome is known,
before trusting it for anything new. Run: python -m quantum_sim.validate
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit


def _assert_close(actual, expected, msg, tol=1e-9):
    if not np.allclose(actual, expected, atol=tol):
        raise AssertionError(f"FAILED: {msg}\n  expected={expected}\n  actual={actual}")
    print(f"  ok: {msg}")


def test_single_hadamard():
    print("Hadamard on |0>: expect 50/50 superposition")
    c = Circuit(1).h(0)
    _assert_close(c.probabilities(), [0.5, 0.5], "H|0> gives equal |0>/|1> probability")


def test_x_is_bitflip():
    print("X on |0>: expect exactly |1>")
    c = Circuit(1).x(0)
    _assert_close(c.probabilities(), [0.0, 1.0], "X|0> = |1>")


def test_double_hadamard_is_identity():
    print("H H on |0>: expect back to |0> (H^2 = I)")
    c = Circuit(1).h(0).h(0)
    _assert_close(c.probabilities(), [1.0, 0.0], "H.H|0> = |0>")


def test_bell_state():
    print("Bell state (H(0), CNOT(0,1)): expect 50/50 on |00>/|11> only")
    c = Circuit(2).h(0).cnot(0, 1)
    probs = c.probabilities()  # order: |00>, |01>, |10>, |11>
    _assert_close(probs, [0.5, 0.0, 0.0, 0.5], "Bell state probabilities")
    _assert_close(c.sim.norm(), 1.0, "Bell state stays normalized")


def test_ghz_state():
    print("GHZ state (3 qubits, H(0), CNOT(0,1), CNOT(1,2)): expect 50/50 on |000>/|111>")
    c = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
    probs = c.probabilities()
    expected = np.zeros(8)
    expected[0] = 0.5   # |000>
    expected[7] = 0.5   # |111>
    _assert_close(probs, expected, "GHZ state probabilities")


def test_uniform_superposition():
    print("H on every qubit of a 3-qubit register: expect uniform over all 8 states")
    c = Circuit(3).h(0).h(1).h(2)
    probs = c.probabilities()
    _assert_close(probs, np.full(8, 1 / 8), "uniform superposition over 8 basis states")


def test_toffoli():
    print("Toffoli: flips target only when both controls are 1")
    c = Circuit(3).x(0).x(1).toffoli(0, 1, 2)  # controls=1,1 -> target flips 0->1
    _assert_close(c.probabilities()[7], 1.0, "|110> -> |111> via Toffoli")
    c2 = Circuit(3).x(0).toffoli(0, 1, 2)  # only one control is 1 -> target unchanged
    _assert_close(c2.probabilities()[4], 1.0, "|100> stays |100> via Toffoli (one control off)")


def test_rz_full_turn_returns_to_start_probabilities():
    print("Rz(2*pi): global phase changes, but measurement probabilities must be unchanged")
    c = Circuit(1).h(0).rz(0, 2 * np.pi)
    _assert_close(c.probabilities(), [0.5, 0.5], "Rz(2pi) leaves probabilities at 50/50")


def test_measure_all_matches_probabilities():
    print("measure_all(shots=20000) on Bell state: sampled frequencies should match 50/50")
    c = Circuit(2).h(0).cnot(0, 1)
    counts = c.measure_all(shots=20000, seed=0)
    assert set(counts.keys()) <= {"00", "11"}, f"unexpected outcomes: {counts.keys()}"
    freq00 = counts.get("00", 0) / 20000
    assert 0.47 < freq00 < 0.53, f"sampled frequency of |00> = {freq00}, expected close to 0.5"
    print(f"  ok: sampled counts {counts}, freq(|00>)={freq00:.4f}")


def run_all():
    tests = [
        test_single_hadamard,
        test_x_is_bitflip,
        test_double_hadamard_is_identity,
        test_bell_state,
        test_ghz_state,
        test_uniform_superposition,
        test_toffoli,
        test_rz_full_turn_returns_to_start_probabilities,
        test_measure_all_matches_probabilities,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} validation checks passed.")


if __name__ == "__main__":
    run_all()
