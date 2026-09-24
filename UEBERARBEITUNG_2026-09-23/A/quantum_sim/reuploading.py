from __future__ import annotations

"""RESEARCH ID: Q-017 -- Data Re-Uploading: Ausdruckskraft ohne mehr Qubits.

DIE LUECKE
Das Projekt kennt genau eine Art, Daten in die Schaltung zu bringen: einmal
kodieren, dann variationelle Lagen. Amplituden-Kodierung braucht dafuer 2^n
Werte und hunderte Gatter -- auf echter Hardware unbrauchbar. Deshalb lief auf
`ibm_fez` die Winkel-Kodierung, und die kostet: 0,958 in Simulation gegen
**0,808** hardwaretauglich.

Data Re-Uploading (Perez-Salinas et al. 2020) ist genau der Gegenentwurf: Die
Daten werden **in JEDER Lage erneut eingespeist**, statt nur am Anfang. Das
zentrale Ergebnis der Arbeit: ein EINZELNES Qubit mit genuegend
Wiedereinspeisungen ist ein universeller Funktionsapproximator.

Warum das das Projektziel trifft: Es erzeugt Ausdruckskraft aus der TIEFE statt
aus der BREITE. Mehr Qubits kosten exponentiell (Simulation) oder sind gar nicht
verfuegbar (Hardware); mehr Lagen kosten linear. Und die Schaltung bleibt flach
genug fuer echte Hardware, weil je Lage nur n Rotationen und n CNOTs anfallen.

DER ZUSAMMENHANG ZU DEN EIGENEN BEFUNDEN
Q-003-replication hat gemessen, dass **Tiefe nichts beitraegt**: 1 Lage ist
gleichwertig zu 5 (TOST, +-0,005). Das ist kein Widerspruch, sondern der
Anlass -- ohne Wiedereinspeisung ist eine tiefere Schaltung nur eine andere
Drehung derselben Gruppe (siehe `theory.py`, Teil 2: Butterfly(t1) o
Butterfly(t2) = Butterfly(t3)). Zusaetzliche Lagen koennen dann per Konstruktion
nichts Neues. **Mit** Wiedereinspeisung gilt das nicht mehr: zwischen den
Drehungen steht jedesmal ein datenabhaengiger Schritt, und die Verkettung
verlaesst die Gruppe.

Damit ist eine scharfe Vorhersage moeglich:
    ohne Re-Uploading  ->  Tiefe bringt nichts   (gemessen, TOST-belegt)
    mit  Re-Uploading  ->  Tiefe bringt etwas    (offen, hier getestet)
Trifft das zu, ist die Gruppenstruktur der Schaltung nicht nur eine Erklaerung
im Nachhinein, sondern hat korrekt vorhergesagt, wann Tiefe hilft.

DIE ARME (alle mit demselben klassischen Encoder auf n Merkmale)
  angle_once     Winkel-Kodierung einmal, dann L variationelle Lagen.
                 Das ist die heutige hardwaretaugliche Variante.
  reupload       Winkel-Kodierung in JEDER Lage: Ry(a_l * x_q + b_l), dann
                 variationelles Rz, dann CNOT-Ring. Parameter je Lage: 3n
                 (Skala, Verschiebung, Phase) statt 2n.
  reupload_deep  dasselbe mit doppelt so vielen Lagen -- prueft, ob Tiefe MIT
                 Wiedereinspeisung etwas bringt.
  amplitude      Amplituden-Kodierung wie im bisherigen Hybridmodell.
                 Referenz "was in Simulation moeglich ist", auf Hardware nicht
                 ausfuehrbar.

VORREGISTRIERUNG
  Balancierte Genauigkeit, Lernrate je Arm, 15 Seeds, zwei Datensaetze.

  PRIMAER  reupload schlaegt angle_once. Das ist die Frage, ob
           Wiedereinspeisung die hardwaretaugliche Variante rettet.
  P2       reupload_deep schlaegt reupload -- traegt Tiefe MIT
           Wiedereinspeisung, anders als ohne?
  Holm ueber genau diese beiden.

  H1 : beide halten -> Wiedereinspeisung erzeugt Ausdruckskraft aus Tiefe, und
       zwar messbar dort, wo blosse Tiefe nichts brachte. Das waere zugleich
       eine bestaetigte Vorhersage der Gruppenstruktur-Ueberlegung UND ein
       hardwaretauglicher Gewinn.
  H0 : PRIMAER faellt -> Wiedereinspeisung hilft hier nicht; die Luecke zur
       Amplituden-Kodierung liegt woanders.
  HU : unentschieden.

  AUFLOESUNGSPRUEFUNG (Regel 6): `amplitude` muss `angle_once` deutlich
  schlagen -- der Abstand 0,958 zu 0,808 aus der Demo laesst das erwarten.
  Trennt sich das nicht, misst der Aufbau den Kodierungsunterschied nicht und
  kein Ergebnis oben ist interpretierbar.

  ERWARTUNG, vorher notiert: H1 fuer PRIMAER (die Literatur ist da deutlich),
  offen fuer P2. Sollte P2 halten, ist das der interessantere Teil, weil er
  eine Vorhersage aus der eigenen Theorie bestaetigt.

Run: python -m quantum_sim.reuploading
"""

import math
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .torch_simulator import CNOT, TorchStatevector, ry, rz
from .quantum_layer import QuantumLayer

ARMS = ("angle_once", "reupload", "reupload_deep", "amplitude")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
NUM_QUBITS = 8
LAYERS = 5
DEEP_LAYERS = 10
EVAL_SEEDS = tuple(range(9901, 9916))       # 15 Seeds
TUNING_SEEDS = (9951, 9952)
LR_GRID = (0.003, 0.01, 0.03, 0.1)
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


class AngleCircuit(nn.Module):
    """Winkel-Kodierung, wahlweise mit Wiedereinspeisung in jeder Lage.

    Ohne Wiedereinspeisung (reupload=False) ist das die hardwaretaugliche
    Variante: n Rotationen zur Kodierung, dann L variationelle Lagen.

    Mit Wiedereinspeisung traegt jede Lage den Winkel a_l * x_q + b_l, das
    Merkmal x_q geht also L-mal ein. Genau das verlaesst die Gruppe der reinen
    Drehungen -- ohne Daten dazwischen waere die Verkettung wieder nur eine
    Drehung, und mehr Lagen koennten per Konstruktion nichts Neues.
    """

    def __init__(self, num_qubits: int, num_layers: int, reupload: bool):
        super().__init__()
        self.n, self.L, self.reupload = num_qubits, num_layers, reupload
        # Skala und Verschiebung je Qubit und Lage (nur bei Wiedereinspeisung
        # wirklich je Lage; sonst einmal am Anfang)
        k = num_layers if reupload else 1
        self.scale = nn.Parameter(torch.ones(k, num_qubits))
        self.shift = nn.Parameter(torch.zeros(k, num_qubits))
        # variationelle Phase je Qubit und Lage
        self.phase = nn.Parameter(torch.rand(num_layers, num_qubits) * 2 * math.pi)
        if not reupload:
            # ohne Wiedereinspeisung braucht es zusaetzlich freie Ry-Winkel,
            # sonst haette der Arm weniger Parameter und der Vergleich waere
            # nicht ueber die Kodierung, sondern ueber die Parameterzahl
            self.free = nn.Parameter(torch.rand(num_layers, num_qubits) * 2 * math.pi)

    def num_circuit_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        state = TorchStatevector(self.n, batch=batch)
        for layer in range(self.L):
            k = layer if self.reupload else 0
            for q in range(self.n):
                if self.reupload or layer == 0:
                    angle = self.scale[k, q] * x[:, q] + self.shift[k, q]
                    # datenabhaengige Rotation: je Beispiel ein eigener Winkel
                    state = _apply_ry_batch(state, angle, q, self.n)
                if not self.reupload:
                    state.apply(ry(self.free[layer, q]), [q])
                state.apply(rz(self.phase[layer, q]), [q])
            for q in range(self.n):
                state.apply(CNOT, [q, (q + 1) % self.n])
        return state.expectation_z()


def _apply_ry_batch(state: TorchStatevector, angle: torch.Tensor, q: int,
                    n: int) -> TorchStatevector:
    """Ry mit einem EIGENEN Winkel je Beispiel im Batch.

    `TorchStatevector.apply` erwartet ein festes Gatter fuer den ganzen Batch;
    bei datenabhaengigen Winkeln geht das nicht. Deshalb hier direkt auf den
    Amplituden, in derselben Konvention (Qubit q = Achse q+1).
    """
    flat = state.state.reshape(state.state.shape[0], -1)
    left, right = 2 ** q, 2 ** (n - 1 - q)
    v = flat.reshape(-1, left, 2, right)
    c = torch.cos(angle / 2).to(torch.complex64).reshape(-1, 1, 1)
    s = torch.sin(angle / 2).to(torch.complex64).reshape(-1, 1, 1)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    out = torch.stack((c * a - s * b, s * a + c * b), dim=2)
    state.state = out.reshape((-1,) + (2,) * n)
    return state


class ReuploadModel(nn.Module):
    def __init__(self, arm: str, input_dim: int, num_classes: int):
        super().__init__()
        if arm == "amplitude":
            self.encoder = nn.Linear(input_dim, 2 ** NUM_QUBITS)
            self.circuit = QuantumLayer(NUM_QUBITS, LAYERS, entangle=True)
        else:
            self.encoder = nn.Linear(input_dim, NUM_QUBITS)
            layers = DEEP_LAYERS if arm == "reupload_deep" else LAYERS
            self.circuit = AngleCircuit(NUM_QUBITS, layers,
                                        reupload=arm != "angle_once")
        self.head = nn.Linear(NUM_QUBITS, num_classes)
        self.arm = arm

    def circuit_parameters(self) -> int:
        return self.circuit.num_circuit_parameters()

    def forward(self, x):
        return self.head(self.circuit(self.encoder(x)))


def _cell(arm: str, dataset: str, seed: int, lr: float, threads: int,
          max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = ReuploadModel(arm, bundle.input_dim, bundle.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = bundle.x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bundle.x_train[idx]), bundle.y_train[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(bundle.x_test[i:i + 512]).argmax(-1)
                          for i in range(0, bundle.x_test.shape[0], 512)])
        classes = torch.unique(bundle.y_test)
        bal = float(np.mean([float((pred[bundle.y_test == c] == c).float().mean())
                             for c in classes]))
    return {"arm": arm, "dataset": dataset, "seed": seed, "lr": lr,
            "balanced_accuracy": bal,
            "accuracy": float((pred == bundle.y_test).float().mean()),
            "circuit_parameters": model.circuit_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "hardware_ready": arm != "amplitude"}


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["balanced_accuracy"]
                                    for s in TUNING_SEEDS])) for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<14} -> lr {best[arm]}")
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-017",
        description=("Data Re-Uploading: rettet Wiedereinspeisung der Daten die "
                     "hardwaretaugliche Winkel-Kodierung? Und traegt Tiefe MIT "
                     "Wiedereinspeisung, anders als ohne?"),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "num_qubits": NUM_QUBITS,
                "layers": LAYERS, "deep_layers": DEEP_LAYERS,
                "lr_grid": list(LR_GRID), "max_samples": MAX_SAMPLES,
                "epochs": EPOCHS, "batch_size": BATCH, "margin": MARGIN,
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "reupload schlaegt angle_once",
                "p2": "reupload_deep schlaegt reupload (Tiefe mit Wiedereinspeisung)",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "amplitude muss angle_once deutlich schlagen",
                "vorhersage": ("ohne Wiedereinspeisung ist Tiefe wirkungslos "
                               "(TOST-belegt in Q-003-replication), weil die "
                               "Verkettung von Drehungen wieder eine Drehung "
                               "ist; mit Wiedereinspeisung sollte sie wirken")},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, threads)
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_{dataset}"] = edge
        if edge:
            print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                rec.add(**_cell(arm, dataset, seed, lr_map[arm], threads))
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
        print(f"    Zwischenstand nach {dataset} gesichert:")
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    datasets = [str(d) for d in config["datasets"]]
    out: Dict[str, object] = {}

    def vals(arm: str, ds: str) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print(f"RESEARCH ID Q-017 | Data Re-Uploading, {NUM_QUBITS} Qubits")
    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 62 - len(ds)))
        head = (f'  {"Arm":>14} {"balanced":>10} {"std":>8} {"Schaltung":>10} '
                f'{"hardwaretauglich":>18} {"lr":>6}')
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, ds)
            if not len(v):
                continue
            r0 = next(r for r in records if r["arm"] == arm and r["dataset"] == ds)
            print(f'  {arm:>14} {v.mean():>10.4f} {v.std(ddof=1):>8.4f} '
                  f'{r0["circuit_parameters"]:>10,} '
                  f'{str(r0["hardware_ready"]):>18} {r0["lr"]:>6}')

    fam = [("reupload", "angle_once", "PRIMAER"),
           ("reupload_deep", "reupload", "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, PRIMARY_DATASET), vals(b, PRIMARY_DATASET)
        p = float(stats.wilcoxon(va, vb).pvalue)
        raw.append(p)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN), int((va > vb).sum()), len(va)))
    adj = holm(raw)

    print()
    print("=" * 92)
    print(f"VORREGISTRIERTE FAMILIE auf {PRIMARY_DATASET}")
    for (a, b, tag, delta, p, eq, w, nn), p_h in zip(cells, adj):
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} vs {b}: {delta:>+8.4f}  {w}/{nn}  '
              f'p={p:.4f}  Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"]}

    am, ao = vals("amplitude", PRIMARY_DATASET), vals("angle_once", PRIMARY_DATASET)
    c_p = float(stats.wilcoxon(am, ao).pvalue)
    c_ok = am.mean() > ao.mean() and c_p < 0.05
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"Amplituden- gegen Winkel-Kodierung {am.mean() - ao.mean():+.4f}, "
            f"p={c_p:.4f}." + ("" if c_ok else " Der Aufbau misst den "
            "Kodierungsunterschied nicht -- kein Ergebnis oben ist verwertbar."))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    prim, p2 = out["PRIMAER"], out["P2"]
    ru, ao_m, am_m = (vals("reupload", PRIMARY_DATASET).mean(), ao.mean(), am.mean())
    closed = (ru - ao_m) / max(am_m - ao_m, 1e-9) * 100
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        v = (f"H1 -- Wiedereinspeisung verbessert die hardwaretaugliche Variante "
             f"um {prim['delta']:+.4f} und schliesst damit {closed:.0f} % der "
             f"Luecke zur Amplituden-Kodierung -- bei einer Schaltung, die auf "
             f"echter Hardware ausfuehrbar bleibt.")
        if p2["p_holm"] < 0.05 and p2["delta"] > 0:
            v += (" Und Tiefe traegt MIT Wiedereinspeisung etwas bei, waehrend "
                  "sie ohne TOST-gleichwertig war -- die Gruppenstruktur-"
                  "Ueberlegung hat das korrekt vorhergesagt.")
    elif prim["p_holm"] < 0.05:
        v = (f"H0 -- Wiedereinspeisung ist SCHLECHTER ({prim['delta']:+.4f}). "
             f"Die Luecke zur Amplituden-Kodierung liegt woanders.")
    else:
        v = f"HU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f}."
    print(f"BEWERTUNG: {v}")
    print("=" * 92)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.reuploading '
              "import analyse; analyse('Q-017')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
