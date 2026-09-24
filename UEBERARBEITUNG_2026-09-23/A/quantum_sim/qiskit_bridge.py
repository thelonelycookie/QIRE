"""Bridge between the from-scratch simulator and Qiskit / real IBM hardware.

Two separate purposes, kept in separate functions:
1. Cross-validate quantum_sim against an independent implementation (Qiskit's
   local Statevector) -- if two independently-written simulators agree, that
   is much stronger evidence of correctness than either one's own unit tests.
2. Submit the same circuit to real IBM Quantum hardware. This needs an IBM
   Quantum account and API token, which the student must create themselves
   at https://quantum.ibm.com (free tier: 10 minutes/month on real 100+ qubit
   processors) -- not something this script can do on your behalf. Provide it
   either as the IBM_QUANTUM_TOKEN environment variable, or by writing it to
   `ibm_quantum_token.txt` in the project root (see `_load_token()`) -- this
   project has no git repo yet, but if one is added later, gitignore that
   file before ever committing.

Bit-order warning: Qiskit prints qubit 0 as the RIGHTMOST character in a
bitstring; quantum_sim's `bitstring()` prints qubit 0 as the LEFTMOST
character (matching its tensor-axis convention). `to_quantum_sim_order()`
below converts between the two so comparisons are on the same footing --
never compare the raw strings directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .circuit import Circuit


def to_quantum_sim_order(qiskit_bitstring: str) -> str:
    """Qiskit's qubit-0-is-rightmost bitstring -> quantum_sim's qubit-0-is-leftmost."""
    return qiskit_bitstring[::-1]


def bell_state_qiskit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def ghz_state_qiskit(n: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


def cross_validate(qc: QuantumCircuit, own_circuit: Circuit, tol: float = 1e-9) -> None:
    """Raises AssertionError if Qiskit's exact statevector and quantum_sim's
    exact statevector disagree on measurement probabilities (after
    correcting for the bit-order convention difference)."""
    qiskit_probs = Statevector.from_instruction(qc).probabilities_dict()
    qiskit_probs_converted = {to_quantum_sim_order(k): v for k, v in qiskit_probs.items()}

    own_probs = own_circuit.probabilities()
    own_probs_dict = {
        own_circuit.sim.bitstring(i): float(p) for i, p in enumerate(own_probs) if p > tol
    }

    all_keys = set(qiskit_probs_converted) | set(own_probs_dict)
    mismatches = []
    for key in sorted(all_keys):
        a = qiskit_probs_converted.get(key, 0.0)
        b = own_probs_dict.get(key, 0.0)
        if abs(a - b) > 1e-6:
            mismatches.append((key, a, b))
    if mismatches:
        raise AssertionError(f"Qiskit and quantum_sim disagree: {mismatches}")
    print(f"  ok: Qiskit and quantum_sim agree on all {len(all_keys)} outcomes: {own_probs_dict}")


TOKEN_FILE = Path(__file__).resolve().parent.parent / "ibm_quantum_token.txt"


def _load_token() -> str | None:
    """Env var first (works when the whole session sets it), otherwise a local
    token file (works across separate tool/shell invocations, since -- unlike
    an exported env var -- a file on disk is visible to any later process).
    Never printed or logged anywhere; keep TOKEN_FILE out of any repo you
    later push (add it to .gitignore once this project has a .git)."""
    import os

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        raw = TOKEN_FILE.read_bytes()
        # PowerShell's `>` redirection writes UTF-16LE with a BOM by default,
        # not UTF-8 -- try the common encodings rather than assuming one.
        for encoding in ("utf-16", "utf-8-sig", "utf-8"):
            try:
                return raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"could not decode {TOKEN_FILE} with utf-16/utf-8-sig/utf-8")
    return None


def run_on_real_hardware(qc: QuantumCircuit, shots: int = 1024, backend_name: str | None = None):
    """Submits `qc` to real IBM Quantum hardware. Requires:
      1. A free account at https://quantum.ibm.com (student creates this themselves)
      2. An API token from that account's dashboard, either set as the
         IBM_QUANTUM_TOKEN environment variable, or written to
         `ibm_quantum_token.txt` in the project root (e.g. from your own
         terminal: `echo "your-token" > ibm_quantum_token.txt`) -- the file
         path is fixed, contents are never printed by this module.

    Not called automatically by anything in this module -- only run this once
    you actually have a token, since it consumes real (rate-limited) quantum
    hardware time.
    """
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    token = _load_token()
    if not token:
        raise RuntimeError(
            "No IBM Quantum token found. Create a free account at "
            "https://quantum.ibm.com, copy your API token from the dashboard, "
            f"then either set IBM_QUANTUM_TOKEN or write it to {TOKEN_FILE}."
        )
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    print(f"Submitting to real backend: {backend.name}")
    with_measurement = qc.copy()
    with_measurement.measure_all()
    # Real hardware only executes its own native gate set + qubit connectivity
    # (e.g. RZ/SX/X + one two-qubit gate) -- transpile() rewrites H/CNOT/etc.
    # into an equivalent circuit the backend can actually run.
    transpiled = transpile(with_measurement, backend=backend, optimization_level=1)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"Job ID: {job.job_id()} (check status at https://quantum.ibm.com)")
    result = job.result()
    counts = result[0].data.meas.get_counts()
    return dict(counts)


if __name__ == "__main__":
    print("Cross-validating Bell state (Qiskit vs. quantum_sim):")
    cross_validate(bell_state_qiskit(), Circuit(2).h(0).cnot(0, 1))

    print("Cross-validating GHZ state (Qiskit vs. quantum_sim):")
    cross_validate(ghz_state_qiskit(3), Circuit(3).h(0).cnot(0, 1).cnot(1, 2))

    print("\nBoth circuits agree between two independent implementations.")
    print("Real-hardware run is available via run_on_real_hardware() once you")
    print("have set IBM_QUANTUM_TOKEN -- not run automatically here.")
