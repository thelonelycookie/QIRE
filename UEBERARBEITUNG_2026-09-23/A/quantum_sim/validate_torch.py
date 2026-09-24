"""Prueft den differenzierbaren Simulator gegen den bereits validierten
NumPy-Simulator und gegen analytisch bekannte Faelle.

Der NumPy-Simulator (`simulator.py`) wurde seinerseits gegen neun bekannte
Schaltungen und gegen Qiskit geprueft. Stimmen beide ueberein, traegt die
Torch-Fassung dieselbe Absicherung.

Run: python -m quantum_sim.validate_torch
"""
from __future__ import annotations

import math

import torch

from .circuit import Circuit
from .torch_simulator import CNOT, H, TorchStatevector, rx, ry, rz


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(f"FEHLGESCHLAGEN: {msg}")
    print(f"  ok: {msg}")


def test_against_numpy_simulator():
    print("Torch-Fassung gegen den validierten NumPy-Simulator:")

    # Bell-Zustand
    sv = TorchStatevector(2, batch=1)
    sv.apply(H, [0]).apply(CNOT, [0, 1])
    torch_probs = sv.probabilities()[0].detach().numpy()
    numpy_probs = Circuit(2).h(0).cnot(0, 1).probabilities()
    _assert(abs(torch_probs - numpy_probs).max() < 1e-6, "Bell-Zustand identisch")

    # GHZ-Zustand
    sv = TorchStatevector(3, batch=1)
    sv.apply(H, [0]).apply(CNOT, [0, 1]).apply(CNOT, [1, 2])
    torch_probs = sv.probabilities()[0].detach().numpy()
    numpy_probs = Circuit(3).h(0).cnot(0, 1).cnot(1, 2).probabilities()
    _assert(abs(torch_probs - numpy_probs).max() < 1e-6, "GHZ-Zustand identisch")

    # Rotationen
    theta = torch.tensor(0.7)
    sv = TorchStatevector(2, batch=1)
    sv.apply(ry(theta), [0]).apply(rz(theta), [1]).apply(CNOT, [0, 1])
    torch_probs = sv.probabilities()[0].detach().numpy()
    numpy_probs = Circuit(2).ry(0, 0.7).rz(1, 0.7).cnot(0, 1).probabilities()
    _assert(abs(torch_probs - numpy_probs).max() < 1e-6, "Ry/Rz + CNOT identisch")


def test_normalisation():
    print("Norm bleibt unter unitaeren Gattern erhalten:")
    torch.manual_seed(0)
    sv = TorchStatevector(4, batch=8)
    for q in range(4):
        sv.apply(ry(torch.tensor(0.3 * (q + 1))), [q])
    for q in range(3):
        sv.apply(CNOT, [q, q + 1])
    total = sv.probabilities().sum(dim=1)
    _assert(float((total - 1.0).abs().max()) < 1e-5, "alle Batch-Zustaende normiert")


def test_gradients_flow():
    print("Gradienten fliessen durch die gesamte Schaltung:")
    theta = torch.tensor([0.4, 0.9, 1.3], requires_grad=True)
    sv = TorchStatevector(3, batch=4)
    for q in range(3):
        sv.apply(ry(theta[q]), [q])
    sv.apply(CNOT, [0, 1]).apply(CNOT, [1, 2])
    loss = sv.expectation_z().sum()
    loss.backward()
    _assert(theta.grad is not None and float(theta.grad.abs().sum()) > 0,
            f"Gradienten vorhanden, Norm={float(theta.grad.norm()):.4f}")


def test_expectation_known_values():
    print("Erwartungswerte <Z> gegen analytisch bekannte Faelle:")
    # |0> -> <Z> = +1
    sv = TorchStatevector(1, batch=1)
    _assert(abs(float(sv.expectation_z()[0, 0]) - 1.0) < 1e-6, "|0> liefert <Z> = +1")
    # Ry(pi) |0> = |1> -> <Z> = -1
    sv = TorchStatevector(1, batch=1).apply(ry(torch.tensor(math.pi)), [0])
    _assert(abs(float(sv.expectation_z()[0, 0]) + 1.0) < 1e-5, "Ry(pi)|0> liefert <Z> = -1")
    # Ry(pi/2)|0> = gleiche Ueberlagerung -> <Z> = 0
    sv = TorchStatevector(1, batch=1).apply(ry(torch.tensor(math.pi / 2)), [0])
    _assert(abs(float(sv.expectation_z()[0, 0])) < 1e-5, "Ry(pi/2)|0> liefert <Z> = 0")


def test_amplitude_encoding():
    print("Amplituden-Kodierung klassischer Daten:")
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    sv = TorchStatevector.from_amplitudes(x, num_qubits=2)
    probs = sv.probabilities()[0]
    # normiert: (3,4)/5 -> Wahrscheinlichkeiten 0.36 und 0.64
    _assert(abs(float(probs[0]) - 0.36) < 1e-5 and abs(float(probs[1]) - 0.64) < 1e-5,
            "Vektor korrekt normiert und kodiert")
    _assert(abs(float(probs.sum()) - 1.0) < 1e-6, "Wahrscheinlichkeiten summieren zu 1")


def test_batch_independence():
    print("Batch-Verarbeitung: Zustaende beeinflussen sich nicht gegenseitig:")
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    sv = TorchStatevector.from_amplitudes(x, 2).apply(H, [0])
    probs = sv.probabilities()
    single_a = TorchStatevector.from_amplitudes(x[:1], 2).apply(H, [0]).probabilities()
    single_b = TorchStatevector.from_amplitudes(x[1:], 2).apply(H, [0]).probabilities()
    _assert(float((probs[0] - single_a[0]).abs().max()) < 1e-6
            and float((probs[1] - single_b[0]).abs().max()) < 1e-6,
            "Batch liefert dasselbe wie Einzelrechnungen")


def run_all():
    tests = [
        test_against_numpy_simulator,
        test_normalisation,
        test_gradients_flow,
        test_expectation_known_values,
        test_amplitude_encoding,
        test_batch_independence,
    ]
    for t in tests:
        t()
    print(f"\nAlle {len(tests)} Pruefungen bestanden.")


if __name__ == "__main__":
    run_all()
