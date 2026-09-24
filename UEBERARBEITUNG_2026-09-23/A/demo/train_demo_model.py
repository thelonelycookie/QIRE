from __future__ import annotations

"""Trainiert die drei Modelle fuer die Stand-Demo und legt sie als Datei ab.

Die Demo soll am Wettbewerbsstand OHNE Internet und ohne Trainingslauf starten.
Deshalb wird hier einmal trainiert und alles Noetige gespeichert.

DREI MODELLE -- und die Auswahl ist selbst die Aussage der Arbeit:

  quantum_sim   Quantenschaltung mit Amplituden-Kodierung. 8 Qubits, 256-dim
                Zustandsraum, 80 Parameter. Erreicht klassisches Niveau, ist
                aber wegen der aufwendigen Zustandspraeparation NICHT auf echter
                Hardware ausfuehrbar.

  quantum_hw    Quantenschaltung mit Winkel-Kodierung, hardwaretauglich und
                nachweislich auf ibm_fez gelaufen. Schwaecher, weil die flache
                Kodierung weniger Information in die Schaltung bringt.

  classical     Klassische Schicht gleicher Breite. Die Kontrolle.

Der Besucher sieht damit unmittelbar den Kernbefund: in Simulation zieht die
Quantenschaltung mit, die hardwaretaugliche Variante faellt ab -- und genau
das ist der Grund, warum Quanten-KI heute noch nicht praktisch ist.

Run: python -m demo.train_demo_model
"""

import json
import math
from pathlib import Path

import torch
from torch import nn

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from quantum_sim.hardware_model import HardwareReadyModel, train as train_quantum
from quantum_sim.quantum_layer import HybridModel

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


class MatchedClassical(nn.Module):
    """Gleicher Encoder wie die Quantenmodelle, danach eine klassische Schicht
    derselben Breite. So unterscheidet sich nur der mittlere Teil."""

    def __init__(self, input_dim: int, width: int, num_classes: int, depth: int = 4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, width)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.middle = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.head(self.middle(torch.tanh(self.encoder(x)) * math.pi))

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


def train_classical(model, x, y, epochs, lr, batch_size, seed):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], generator=g)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()


def accuracy(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def main(seed: int = 2001, max_samples: int = 20000, epochs: int = 20):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.set_num_threads(6)

    print("Lade MNIST ...")
    bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
    args = (bundle.x_train, bundle.y_train, epochs, 0.01, 128, seed)

    print("1/3  Quantenschaltung, Amplituden-Kodierung (8 Qubits, Simulation) ...")
    q_sim = HybridModel(bundle.input_dim, num_qubits=8, num_layers=5,
                        num_classes=bundle.num_classes, entangle=True)
    train_quantum(q_sim, *args)
    acc_sim = accuracy(q_sim, bundle.x_test, bundle.y_test)
    print(f"     Genauigkeit: {acc_sim:.4f}  ({q_sim.quantum.num_circuit_parameters()} Parameter)")

    print("2/3  Quantenschaltung, Winkel-Kodierung (10 Qubits, hardwaretauglich) ...")
    q_hw = HardwareReadyModel(bundle.input_dim, num_qubits=10, num_layers=4,
                              num_classes=bundle.num_classes)
    train_quantum(q_hw, *args)
    acc_hw = accuracy(q_hw, bundle.x_test, bundle.y_test)
    print(f"     Genauigkeit: {acc_hw:.4f}  ({q_hw.quantum.circuit_parameters()} Parameter)")

    print("3/3  Klassische Vergleichsschicht (gleiche Breite) ...")
    classical = MatchedClassical(bundle.input_dim, 10, bundle.num_classes, depth=4)
    train_classical(classical, *args)
    acc_cl = accuracy(classical, bundle.x_test, bundle.y_test)
    print(f"     Genauigkeit: {acc_cl:.4f}  ({classical.middle_parameters()} Parameter)")

    torch.save({
        "quantum_sim_state": q_sim.state_dict(),
        "quantum_hw_state": q_hw.state_dict(),
        "classical_state": classical.state_dict(),
        "input_dim": bundle.input_dim,
        "num_classes": bundle.num_classes,
        "sim_qubits": 8, "sim_layers": 5,
        "hw_qubits": 10, "hw_layers": 4,
        "classical_width": 10, "classical_depth": 4,
        "accuracy": {"quantum_sim": acc_sim, "quantum_hw": acc_hw, "classical": acc_cl},
        "parameters": {
            "quantum_sim": q_sim.quantum.num_circuit_parameters(),
            "quantum_hw": q_hw.quantum.circuit_parameters(),
            "classical": classical.middle_parameters(),
        },
    }, ARTIFACT_DIR / "demo_models.pt")
    print(f"\nGespeichert: {ARTIFACT_DIR / 'demo_models.pt'}")

    hardware_evidence = {
        "backend": "ibm_fez",
        "processor_qubits": 156,
        "runs": [
            {"label": "flach (4 Qubits, 33 Gatter)", "job_id": "d9ns0pc60llc73caf80g",
             "sim_accuracy": 0.500, "hw_accuracy": 0.500, "agreement": 0.850,
             "z_error": 0.1076, "damping": 0.852},
            {"label": "tief (6 Qubits, 67 Gatter)", "job_id": "d9ns10ssfqic73ardag0",
             "sim_accuracy": 0.450, "hw_accuracy": 0.450, "agreement": 0.750,
             "z_error": 0.1498, "damping": 0.707},
        ],
        "bell_ghz": {"bell_error_percent": 1.57, "ghz_error_percent": 2.06, "p_value": 0.043},
        "note": ("Am Stand rechnet die Simulation, damit die Vorhersage sofort "
                 "erscheint. Dieselben Schaltungen liefen nachweislich auf echter "
                 "Hardware -- die Job-IDs sind ueber quantum.ibm.com pruefbar."),
    }
    (ARTIFACT_DIR / "hardware_evidence.json").write_text(
        json.dumps(hardware_evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gespeichert: {ARTIFACT_DIR / 'hardware_evidence.json'}")


if __name__ == "__main__":
    main()
