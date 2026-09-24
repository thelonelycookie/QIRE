from __future__ import annotations

"""RESEARCH ID: Q-012 -- Laesst sich auch der Encoder quantenabgeleitet bauen?

DAS PROBLEM, DAS DIESER LAUF ANGREIFT
Die Kernaussage von Projekt A lautet: 80 Parameter schlagen 67.848. Sie gilt
aber fuer die MITTLERE SCHICHT. Die Gesamtmodelle liegen bei 201.130 gegen
268.898 Parametern, weil in jedem Arm ein klassischer Encoder steckt, der die
784 Pixel auf 256 Amplituden bringt -- und der allein kostet 200.704 Parameter,
also 99,96 % des Hybridmodells.

Solange das so ist, ist die Aussage angreifbar: "Ihr habt die kleine Schicht
ersetzt und die grosse stehen lassen." Dieser Lauf prueft, ob der Encoder
mitkomprimierbar ist.

EIN ENTWURF, DER BEIM NACHDENKEN SOFORT GESCHEITERT IST -- und warum das zaehlt
Der naheliegende Ansatz waere: den Encoder als Butterfly mit geteilten Winkeln
bauen, so wie die Schaltung selbst. Das funktioniert nicht, und der Grund ist
lehrreich.

Die Butterfly-Abbildung ist **normerhaltend und quadratisch**, also ein Element
derselben Gruppe wie die Quantenschaltung. Zwei davon hintereinander ergeben
wieder genau so ein Element:

    Butterfly(theta_1) . Butterfly(theta_2) = Butterfly(theta_3)

Ein Butterfly-Encoder vor der Schaltung waere also **nichts anderes als mehr
Lagen** in derselben Schaltung. Und Q-003-replication hat gemessen, dass Tiefe
nichts beitraegt: 16 Parameter (1 Lage) sind gleichwertig zu 80 (5 Lagen),
TOST-belegt. Der naive Butterfly-Encoder waere also garantiert wirkungslos --
das steht vor dem Lauf fest und muss nicht gemessen werden.

WAS DER ENCODER LEISTEN MUSS, DAMIT ER MEHR IST ALS TIEFE
Er muss etwas tun, das die Schaltung nicht kann. Die Schaltung ist unitaer,
also dimensionserhaltend und umkehrbar. Der Encoder dagegen **reduziert
Information**: 784 Pixel auf 256 Amplituden. Genau diese Reduktion ist der
nicht-unitaere Anteil.

Daraus der Aufbau, der hier getestet wird:

    784 Pixel
      -> mit Nullen auf 1024 auffuellen        (parameterfrei)
      -> Butterfly ueber 10 Qubits, geteilt    (2*10*L Winkel)
      -> auf die ersten 256 Komponenten kuerzen  (parameterfrei, NICHT unitaer)
      -> Quantenschaltung ueber 8 Qubits       (2*8*L Winkel)
      -> klassischer Kopf                      (8*10 + 10)

Die Kuerzung ist der Punkt: Sie macht den Encoder zu etwas anderem als einer
weiteren Lage. Und weil die Butterfly-Abbildung global mischt, liegt vor der
Kuerzung Information aus allen 784 Pixeln in jeder Komponente.

Parameterbilanz bei L=5:
    Butterfly-Encoder   100
    Quantenschaltung     80
    Kopf                 90
    ------------------------
    GESAMT              270   gegen 201.130 des jetzigen Hybridmodells

DIE ARME
  full_butterfly   der Aufbau oben, rund 270 Parameter insgesamt
  dense_encoder    heutiges Hybridmodell: klassischer Encoder + Schaltung
  dense_all        klassischer Encoder + Linear/Tanh -- der bekannte Bezugspunkt
  frozen_encoder   AUFLOESUNGSPRUEFUNG: klassischer Encoder, aber eingefroren
                   auf der Initialisierung. Muss deutlich schlechter sein als
                   `dense_encoder` -- sonst lernt der Encoder ohnehin nichts und
                   der ganze Vergleich waere gegenstandslos.

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm, 15 Seeds, zwei
  Datensaetze.

  PRIMAER  full_butterfly gegen dense_encoder, TOST auf Gleichwertigkeit
           innerhalb **+-0,02**. Die Grenze ist bewusst weiter als die sonst
           benutzten +-0,005: hier stehen 270 gegen 201.130 Parameter, und die
           Frage ist nicht "genau gleich", sondern "nah genug, dass der Faktor
           745 den Abstand wert ist".
  P2       full_butterfly gegen dense_all (Linear/Tanh, 268.898 Parameter).

  H1 : Gleichwertigkeit innerhalb +-0,02 nachgewiesen -> das gesamte Modell
       laesst sich mit rund 270 statt 201.130 Parametern bauen. Das waere die
       Aussage ueber das GANZE Netz, die bisher fehlt.
  H0 : dense_encoder ist signifikant besser -> der Encoder traegt echt etwas
       bei, die 848x-Aussage bleibt auf die mittlere Schicht beschraenkt, und
       das gehoert so berichtet.
  HU : weder noch -> erreichte Schranke berichten.

  AUFLOESUNGSPRUEFUNG: `frozen_encoder` muss signifikant schlechter sein als
  `dense_encoder`. Ist er es nicht, lernt der Encoder nichts Nuetzliches, und
  dann ist "der Encoder laesst sich ersetzen" trivial statt interessant.

Run: python -m quantum_sim.butterfly_encoder
"""

import math
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import apply_1q, cnot_permutation
from .hybrid_final_control import build as build_classical
from .quantum_layer import QuantumLayer

ARMS = ("full_butterfly", "dense_encoder", "dense_all", "frozen_encoder")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(4401, 4416))       # 15 Seeds
TUNING_SEEDS = (4451, 4452)
# 0,3 ergaenzt: in Q-003-replication sass die Hybrid-Familie auch bei 0,1 noch
# am oberen Rand des Gitters, das Optimum lag also moeglicherweise darueber.
LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
ENC_QUBITS = 10                              # 1024 Amplituden vor der Kuerzung
NUM_QUBITS = 8                               # 256 danach
LAYERS = 5
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128
MARGIN_WIDE = 0.02


class ButterflyEncoder(nn.Module):
    """784 -> auffuellen auf 2^ENC_QUBITS -> geteiltes Butterfly -> kuerzen.

    Die Kuerzung ist der nicht-unitaere Anteil und der Grund, warum diese
    Schicht mehr ist als zusaetzliche Tiefe der nachfolgenden Schaltung.
    """

    def __init__(self, input_dim: int, enc_qubits: int = ENC_QUBITS,
                 out_dim: int = 2 ** NUM_QUBITS, num_layers: int = LAYERS):
        super().__init__()
        self.n, self.L = enc_qubits, num_layers
        self.dim, self.out_dim, self.input_dim = 2 ** enc_qubits, out_dim, input_dim
        assert self.dim >= input_dim and self.dim >= out_dim
        self.theta = nn.Parameter(torch.rand(num_layers, enc_qubits, 2) * 2 * math.pi)
        for q in range(enc_qubits):
            self.register_buffer(f"perm_{q}",
                                 cnot_permutation(enc_qubits, q, (q + 1) % enc_qubits))

    def _gate(self, layer: int, q: int) -> torch.Tensor:
        a, b = self.theta[layer, q, 0], self.theta[layer, q, 1]
        ca, sa = torch.cos(a / 2).to(torch.complex64), torch.sin(a / 2).to(torch.complex64)
        eb, ebc = torch.exp(-0.5j * b.to(torch.complex64)), torch.exp(0.5j * b.to(torch.complex64))
        return torch.stack([torch.stack([ca * eb, -sa * eb]),
                            torch.stack([sa * ebc, ca * ebc])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        pad = torch.zeros(batch, self.dim - x.shape[1], dtype=x.dtype, device=x.device)
        state = torch.cat([x, pad], dim=1).to(torch.complex64)
        state = state / state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for layer in range(self.L):
            for q in range(self.n):
                state = apply_1q(state, self._gate(layer, q), q, self.n)
            for q in range(self.n):
                state = state[:, getattr(self, f"perm_{q}")]
        # Kuerzen: der informationsreduzierende, nicht-unitaere Schritt
        return state[:, :self.out_dim].real * math.sqrt(self.dim)


class FullButterfly(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ButterflyEncoder(input_dim)
        self.quantum = QuantumLayer(NUM_QUBITS, LAYERS, entangle=True)
        self.head = nn.Linear(NUM_QUBITS, num_classes)

    def middle_parameters(self) -> int:
        return self.quantum.num_circuit_parameters()

    def forward(self, x):
        return self.head(self.quantum(self.encoder(x)))


class FrozenEncoderHybrid(nn.Module):
    """Wie das Hybridmodell, aber der Encoder bleibt auf der Initialisierung."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 2 ** NUM_QUBITS)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.quantum = QuantumLayer(NUM_QUBITS, LAYERS, entangle=True)
        self.head = nn.Linear(NUM_QUBITS, num_classes)

    def middle_parameters(self) -> int:
        return self.quantum.num_circuit_parameters()

    def forward(self, x):
        return self.head(self.quantum(self.encoder(x)))


def build(arm: str, input_dim: int, num_classes: int):
    if arm == "full_butterfly":
        return FullButterfly(input_dim, num_classes)
    if arm == "frozen_encoder":
        return FrozenEncoderHybrid(input_dim, num_classes)
    if arm == "dense_encoder":
        return build_classical("hybrid", input_dim, num_classes, NUM_QUBITS, LAYERS)
    if arm == "dense_all":
        return build_classical("classical_norm_deep", input_dim, num_classes,
                               NUM_QUBITS, LAYERS)
    raise ValueError(arm)


def _cell(arm: str, dataset: str, seed: int, lr: float, threads: int,
          max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = build(arm, bundle.input_dim, bundle.num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = bundle.x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bundle.x_train[idx]), bundle.y_train[idx]).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(bundle.x_test).argmax(-1)
        classes = torch.unique(bundle.y_test)
        bal = float(np.mean([float((pred[bundle.y_test == c] == c).float().mean())
                             for c in classes]))
    trainable = sum(p.numel() for p in params)
    return {"arm": arm, "dataset": dataset, "seed": seed, "lr": lr,
            "balanced_accuracy": bal,
            "accuracy": float((pred == bundle.y_test).float().mean()),
            "trainable_parameters": trainable,
            "total_parameters": sum(p.numel() for p in model.parameters())}


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["balanced_accuracy"]
                                    for s in TUNING_SEEDS])) for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<16} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-012",
        description=("Laesst sich auch der Encoder quantenabgeleitet bauen? "
                     "Butterfly mit geteilten Winkeln plus Kuerzung statt einer "
                     "dichten 784x256-Schicht. 15 Seeds, zwei Datensaetze, "
                     "Lernrate je Arm, TOST bei +-0,02."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "enc_qubits": ENC_QUBITS,
                "num_qubits": NUM_QUBITS, "layers": LAYERS,
                "lr_grid": list(LR_GRID), "max_samples": MAX_SAMPLES,
                "epochs": EPOCHS, "batch_size": BATCH, "margin": MARGIN_WIDE,
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "full_butterfly gleichwertig zu dense_encoder (+-0,02)",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "frozen_encoder muss schlechter sein als dense_encoder",
                "verworfener_entwurf": ("reiner Butterfly-Encoder ohne Kuerzung "
                                        "waere nur zusaetzliche Tiefe derselben "
                                        "Gruppe; Tiefe ist TOST-gleichwertig")},
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
    print(f"RESEARCH ID Q-012 | {len(vals(ARMS[0], datasets[0]))} Seeds")
    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 58 - len(ds)))
        head = f'  {"Arm":>16} {"balanced":>10} {"std":>8} {"trainierbar":>13} {"lr":>6}'
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, ds)
            if not len(v):
                continue
            par = next(r["trainable_parameters"] for r in records
                       if r["arm"] == arm and r["dataset"] == ds)
            lr = next(r["lr"] for r in records
                      if r["arm"] == arm and r["dataset"] == ds)
            print(f'  {arm:>16} {v.mean():>10.4f} {v.std(ddof=1):>8.4f} '
                  f'{par:>13,} {lr:>6}')

    fam = [("full_butterfly", "dense_encoder", "PRIMAER"),
           ("full_butterfly", "dense_all", "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, PRIMARY_DATASET), vals(b, PRIMARY_DATASET)
        p = float(stats.wilcoxon(va, vb).pvalue)
        raw.append(p)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN_WIDE)))
    adj = holm(raw)

    print()
    print("=" * 90)
    print(f"VORREGISTRIERTE FAMILIE auf {PRIMARY_DATASET}, Grenze +-{MARGIN_WIDE}")
    for (a, b, tag, delta, p, eq), p_h in zip(cells, adj):
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN_WIDE})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} vs {b}: {delta:>+8.4f}  p={p:.4f}  '
              f'Holm={p_h:.4f}  TOST={eq["p_tost"]:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"]}

    fe, de = vals("frozen_encoder", PRIMARY_DATASET), vals("dense_encoder", PRIMARY_DATASET)
    c_p = float(stats.wilcoxon(de, fe).pvalue)
    c_ok = de.mean() > fe.mean() and c_p < 0.05
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"gelernter gegen eingefrorenen Encoder {de.mean() - fe.mean():+.4f}, "
            f"p={c_p:.4f}." + ("" if c_ok else " Der Encoder lernt nichts "
            "Nuetzliches -- dann ist 'er laesst sich ersetzen' trivial."))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    prim = out["PRIMAER"]
    par_full = next(r["trainable_parameters"] for r in records
                    if r["arm"] == "full_butterfly")
    par_dense = next(r["trainable_parameters"] for r in records
                     if r["arm"] == "dense_encoder")
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_tost"] < 0.05:
        v = (f"H1 -- das GESAMTE Modell laesst sich mit {par_full:,} statt "
             f"{par_dense:,} trainierbaren Parametern bauen "
             f"(Faktor {par_dense / par_full:.0f}), gleichwertig innerhalb "
             f"+-{MARGIN_WIDE}. Damit gilt die Effizienzaussage nicht mehr nur "
             f"fuer die mittlere Schicht.")
    elif prim["p_holm"] < 0.05 and prim["delta"] < 0:
        v = (f"H0 -- der klassische Encoder traegt echt bei ({prim['delta']:+.4f}). "
             f"Die 848x-Aussage bleibt auf die mittlere Schicht beschraenkt und "
             f"muss so berichtet werden.")
    else:
        v = f"HU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f}."
    print(f"BEWERTUNG: {v}")
    print("=" * 90)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.butterfly_encoder '
              "import analyse; analyse('Q-012')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
