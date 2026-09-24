from __future__ import annotations

"""RESEARCH ID: Q-004b -- Das trainierte Modell auf echter IBM-Hardware ausfuehren.

ABLAUF
  1. Modell in Simulation trainieren (differenzierbarer Simulator)
  2. Gelernte Winkel in eine Qiskit-Schaltung uebertragen
     -- vorab verifiziert: Torch- und Qiskit-Schaltung liefern identische
        <Z>-Werte (Abweichung 4,6e-06)
  3. Je Testbeispiel eine Schaltung, alle zusammen als ein Job an die Hardware
  4. <Z> aus den gemessenen Bitstrings rekonstruieren
  5. Diese Hardware-<Z> durch den klassischen Kopf schicken
  6. Vergleich: Vorhersage aus Simulation vs. aus Hardware vs. wahres Label

ZWEI KONFIGURATIONEN
Bewusst zwei Schaltungstiefen, weil eine fruehere Messung in diesem Projekt
gezeigt hat, dass die Fehlerrate mit der Tiefe signifikant zunimmt (1,57 % bei
2 Gattern, 2,06 % bei 3, p=0,043). Hier laesst sich derselbe Effekt an einer
echten Anwendung pruefen:

  flach : 4 Qubits, 2 Lagen -> 33 Gatter
  tief  : 6 Qubits, 3 Lagen -> 67 Gatter

ERWARTUNG, VORAB FESTGEHALTEN
Die Hardware-Ergebnisse werden schlechter sein als die Simulation. <Z>-Werte
sollten durch Dekohaerenz Richtung 0 gedaempft werden, und zwar bei der tiefen
Schaltung staerker. Das ist kein Fehlschlag, sondern die Messgroesse.

Run: python -m quantum_sim.run_on_hardware
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from .hardware_model import (
    build_qiskit_circuit,
    expectation_z_from_counts,
    run_simulation,
)
from .qiskit_bridge import _load_token


def prepare(num_qubits: int, num_layers: int, num_test: int, seed: int):
    """Trainiert und liefert alles, was fuer den Hardware-Lauf noetig ist."""
    model, bundle, accuracy = run_simulation(num_qubits=num_qubits, num_layers=num_layers,
                                             seed=seed, max_samples=3000, epochs=12)
    model.eval()
    with torch.no_grad():
        x = bundle.x_test[:num_test]
        y = bundle.y_test[:num_test]
        encoding = model.encode(x)                    # (num_test, n) Winkel
        sim_z = model.quantum(encoding)               # exakte <Z>
        sim_logits = model.head(sim_z)
        sim_pred = sim_logits.argmax(-1)
    return {
        "model": model, "bundle": bundle, "sim_accuracy": accuracy,
        "x": x, "y": y,
        "encoding": encoding.numpy(),
        "sim_z": sim_z.numpy(),
        "sim_pred": sim_pred.numpy(),
        "learned": model.learned_angles(),
    }


def execute_on_hardware(prep: Dict, num_qubits: int, shots: int = 1024,
                        backend_name: str | None = None) -> Dict:
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    token = _load_token()
    if not token:
        raise RuntimeError("Kein IBM-Quantum-Token gefunden.")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = (service.backend(backend_name) if backend_name
               else service.least_busy(operational=True, simulator=False))
    print(f"  Backend: {backend.name}")

    circuits = [build_qiskit_circuit(prep["encoding"][i], prep["learned"], num_qubits)
                for i in range(len(prep["encoding"]))]
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    print(f"  Job-ID: {job.job_id()}")
    result = job.result()

    hw_z = np.zeros((len(circuits), num_qubits))
    for i in range(len(circuits)):
        counts = result[i].data.meas.get_counts()
        hw_z[i] = expectation_z_from_counts(counts, num_qubits)
    return {"hw_z": hw_z, "job_id": job.job_id(), "backend": backend.name}


def analyse(prep: Dict, hw: Dict, label: str) -> Dict:
    model = prep["model"]
    sim_z, hw_z = prep["sim_z"], hw["hw_z"]
    y = prep["y"].numpy()

    with torch.no_grad():
        hw_pred = model.head(torch.tensor(hw_z, dtype=torch.float32)).argmax(-1).numpy()
    sim_pred = prep["sim_pred"]

    z_error = np.abs(sim_z - hw_z)
    damping = np.abs(hw_z).mean() / max(np.abs(sim_z).mean(), 1e-9)

    return {
        "label": label,
        "job_id": hw["job_id"],
        "backend": hw["backend"],
        "sim_accuracy_full": prep["sim_accuracy"],
        "sim_acc_subset": float((sim_pred == y).mean()),
        "hw_acc_subset": float((hw_pred == y).mean()),
        "agreement": float((sim_pred == hw_pred).mean()),
        "z_mean_abs_error": float(z_error.mean()),
        "z_max_abs_error": float(z_error.max()),
        "damping": float(damping),
    }


def run(num_test: int = 20, shots: int = 1024) -> List[Dict]:
    configs = [("flach (4 Qubits, 2 Lagen)", 4, 2, 1601),
               ("tief  (6 Qubits, 3 Lagen)", 6, 3, 1602)]
    results = []
    for label, nq, nl, seed in configs:
        print(f"\n{label}")
        print("  trainiere in Simulation ...")
        prep = prepare(nq, nl, num_test, seed)
        print(f"  Simulation: {prep['sim_accuracy']:.4f} Genauigkeit, "
              f"{model_gates(prep, nq)} Gatter")
        print("  sende an echte Hardware ...")
        hw = execute_on_hardware(prep, nq, shots=shots)
        results.append(analyse(prep, hw, label))
    return results


def model_gates(prep: Dict, num_qubits: int) -> int:
    qc = build_qiskit_circuit(prep["encoding"][0], prep["learned"], num_qubits)
    return sum(qc.count_ops().values())


def report(results: List[Dict]) -> None:
    print()
    print("=" * 78)
    print("RESEARCH ID Q-004b | In Simulation trainiert, auf echter Hardware ausgefuehrt")
    print("=" * 78)
    for r in results:
        print()
        print(f"{r['label']}   Backend {r['backend']}, Job {r['job_id']}")
        print(f"  Genauigkeit Simulation (Teilmenge) : {r['sim_acc_subset']:.3f}")
        print(f"  Genauigkeit Hardware   (Teilmenge) : {r['hw_acc_subset']:.3f}")
        print(f"  Uebereinstimmung der Vorhersagen   : {r['agreement']:.3f}")
        print(f"  mittlerer <Z>-Fehler               : {r['z_mean_abs_error']:.4f}")
        print(f"  groesster <Z>-Fehler               : {r['z_max_abs_error']:.4f}")
        print(f"  Daempfung |<Z>|_hw / |<Z>|_sim      : {r['damping']:.3f}")
    print()
    if len(results) == 2:
        flach, tief = results
        print("Nimmt die Abweichung mit der Schaltungstiefe zu?")
        print(f"  <Z>-Fehler flach {flach['z_mean_abs_error']:.4f} -> "
              f"tief {tief['z_mean_abs_error']:.4f}")
        print(f"  Daempfung  flach {flach['damping']:.3f} -> tief {tief['damping']:.3f}")
        print("  (Daempfung unter 1 bedeutet: Dekohaerenz zieht <Z> Richtung 0)")


if __name__ == "__main__":
    report(run())
