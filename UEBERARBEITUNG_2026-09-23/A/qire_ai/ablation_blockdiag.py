from __future__ import annotations

"""RESEARCH ID: B-003 -- Traegt die Kernaussage von Projekt B ueberhaupt?

WARUM DIESES EXPERIMENT NOETIG IST
Im Tagebuch steht als klassisches Hauptergebnis:

    "spektral-normalisierte Blockschicht: 819x weniger Parameter, besser als
     dicht"

Diese Aussage ist nie in einem Experiment gemessen worden. Sie setzt zwei
getrennte Laeufe zusammen:

  A-002 (ablation_structure.py, Seeds 701-710, depth 1)
        block_diag OHNE Beschraenkung  >  dense        (20.480 vs 16.781.312)
  B-002 (ablation_orthogonality.py, Seeds 1001-1010, depth 3)
        spectral ~ orthogonal  >  plain                (KEIN dense-Arm)

B-002 enthaelt ueberhaupt keinen dichten Vergleichsarm, und die beiden Laeufe
haben verschiedene Tiefen (1 vs 3) und verschiedene Seeds. Aus "plain schlaegt
dicht" und "spektral schlaegt plain" folgt rechnerisch nichts Belastbares: die
Vertrauensbereiche zweier gepaarter Tests auf verschiedenen Seed-Mengen setzen
sich nicht zusammen. Die Aussage ist damit eine Vermutung, keine Messung.

Zusaetzlich: die Konfigurationen wurden nach ihrem Abschneiden in frueheren
Laeufen ausgewaehlt. Ein Nachtest auf denselben Seeds waere zirkulaer -- genau
das Rauschen, das sie auffaellig gemacht hat, wuerde den Effekt reproduzieren.
Deshalb frische Seeds ab 1101, die noch kein Experiment dieses Projekts
beruehrt hat.

DESIGN
Ein Lauf, eine Seed-Menge, alle Arme direkt gegeneinander:

  arm                 Mittelschicht                       Parameter
  ------------------------------------------------------------------
  dense               volle 4096x4096-Matrix              16.781.312
  block_plain         1024 unbeschraenkte 4x4-Bloecke         20.480
  block_spectral      dieselben Bloecke, groesster
                      Singulaerwert = 1                       20.480
  block_orthogonal    dieselben Bloecke, alle
                      Singulaerwerte = 1                      20.480

Alle drei Blockarme haben identischen Speicherbedarf (819x weniger als dicht).
Der orthogonale Arm hat bei gleichem Speicher WENIGER freie Parameter
(d(d-1)/2 = 6 statt 16 je Block).

Beide Tiefen, weil die beiden Vorlaeufer sich genau darin unterscheiden:
  depth 1 -- die Architektur, aus der die 819x-und-besser-als-dicht-Aussage
             stammt (A-002)
  depth 3 -- die Architektur, in der die Beschraenkung gewirkt hat (B-002)

HYPOTHESEN (vor dem Lauf festgelegt)
  H1 : block_spectral schlaegt dense signifikant bei beiden Tiefen
       -> die Kernaussage von B haelt in genau der Form, in der sie im
          Tagebuch steht
  H2 : block_spectral schlaegt dense, aber nicht besser als block_plain
       -> die Blockstruktur traegt, die Spektralnormalisierung ist Beiwerk;
          die Aussage muss auf "Blockschicht" gekuerzt werden
  H0 : kein Blockarm schlaegt dense signifikant
       -> die Kernaussage faellt; B haette kein Hauptergebnis mehr

FALSIFIKATION
H1 verlangt BEIDE Vergleiche signifikant in derselben Seed-Menge:
block_spectral > dense UND block_spectral > block_plain. Faellt der zweite,
gilt H2 und die Formulierung im Tagebuch muss geaendert werden.

Run: python -m qire_ai.ablation_blockdiag
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .ablation_orthogonality import ConstrainedBlock
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .result_store import Recorder

ARMS = ["dense", "block_plain", "block_spectral", "block_orthogonal"]
DEPTHS = (1, 3)
BLOCK_SIZE = 4


def build_middle(label: str, dim: int) -> nn.Module:
    if label == "dense":
        return nn.Linear(dim, dim)
    if label == "block_plain":
        return ConstrainedBlock(dim, BLOCK_SIZE, "plain")
    if label == "block_spectral":
        return ConstrainedBlock(dim, BLOCK_SIZE, "spectral")
    if label == "block_orthogonal":
        return ConstrainedBlock(dim, BLOCK_SIZE, "orthogonal")
    raise ValueError(label)


class MiddleNet(nn.Module):
    """784 -> width -> [Mittelschicht] x depth -> classes."""

    model_name = "MiddleNet"

    def __init__(self, input_dim: int, width: int, num_classes: int, label: str, depth: int):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.middles = nn.ModuleList([build_middle(label, width) for _ in range(depth)])
        self.fc_out = nn.Linear(width, num_classes)
        self.method = f"{label}_d{depth}"

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        for layer in self.middles:
            h = torch.relu(layer(h))
        return self.fc_out(h), {"active_model_parts": 1.0, "gate_entropy": 0.0,
                                "router_ops": 0, "expert_ops": 0}

    def middle_parameters(self) -> int:
        return sum(p.numel() for m in self.middles for p in m.parameters())


def _forward_ms(model: nn.Module, x: torch.Tensor, repeats: int = 30) -> float:
    """Median einer Vorwaertsrechnung in ms. Nur als Nebenangabe gedacht --
    die Thread-Zahl steht in der Konfiguration und ist fuer alle Arme gleich."""
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for _ in range(3):                      # aufwaermen
            model(x)
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def run(
    width: int = 4096,
    seeds: Sequence[int] = tuple(range(1101, 1113)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
    timing_batch: int = 256,
    depths: Sequence[int] = DEPTHS,
    arms: Sequence[str] = tuple(ARMS),
) -> Recorder:
    rec = Recorder(
        experiment_id="B-003",
        description=(
            "Blockdiagonale Mittelschicht (unbeschraenkt / spektral / orthogonal) "
            "direkt gegen eine dichte Schicht, gleiche Seeds, beide Tiefen. "
            "Prueft die Tagebuch-Aussage '819x weniger Parameter, besser als dicht'."
        ),
        config={
            "dataset": "mnist", "width": width, "block_size": BLOCK_SIZE,
            "seeds": list(seeds), "depths": list(depths), "arms": list(arms),
            "max_samples": max_samples, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "torch_threads": threads,
            "timing_batch": timing_batch,
            "seed_note": "frische Seeds ab 1101, in keinem frueheren Experiment benutzt",
        },
    )
    total = len(seeds) * len(depths) * len(arms)
    done = 0
    for seed in seeds:
        for depth in depths:
            for label in arms:
                set_seed(seed)
                torch.set_num_threads(threads)
                bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
                model = MiddleNet(bundle.input_dim, width, bundle.num_classes, label, depth)
                info = train_model(model, bundle.x_train, bundle.y_train, epochs=epochs,
                                   lr=lr, seed=seed, batch_size=batch_size)
                _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
                ms = _forward_ms(model, bundle.x_test[:timing_batch])
                rec.add(
                    arm=label, depth=depth, seed=seed,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=model.middle_parameters(),
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    train_time_sec=float(info["training_time_sec"]),
                    forward_ms_median=ms,
                    nan_detected=str(info.get("nan_detected", "False")),
                )
                done += 1
        print(f"  seed {seed} fertig ({done}/{total} Laeufe)")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records = rec_or_id.records
    else:
        from .result_store import records_of
        records = records_of(rec_or_id)

    def vals(arm: str, depth: int) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["depth"] == depth),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    def params(arm: str, depth: int) -> int:
        for r in records:
            if r["arm"] == arm and r["depth"] == depth:
                return int(r["middle_parameters"])
        return 0

    def ms(arm: str, depth: int) -> float:
        rows = [r["forward_ms_median"] for r in records if r["arm"] == arm and r["depth"] == depth]
        return float(np.median(rows)) if rows else float("nan")

    depths = sorted({int(r["depth"]) for r in records})
    arms = [a for a in ARMS if any(r["arm"] == a for r in records)]
    n = len(vals(arms[0], depths[0]))
    summary: Dict[str, object] = {"n_seeds": n, "depths": depths, "per_depth": {}}

    print()
    print(f"RESEARCH ID B-003 | MNIST, Breite 4096, {n} frische Seeds")

    for depth in depths:
        print()
        print(f"--- depth {depth} " + "-" * 58)
        header = (f'{"arm":>18} {"accuracy":>9} {"std":>7} {"params":>12} '
                  f'{"vs dicht":>10} {"fwd ms":>8}')
        print(header)
        print("-" * len(header))
        dense_p = params("dense", depth)
        for arm in arms:
            v = vals(arm, depth)
            p_arm = params(arm, depth)
            ratio = f"{dense_p / p_arm:.0f}x" if p_arm and arm != "dense" else "-"
            print(f'{arm:>18} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} {p_arm:>12,} '
                  f'{ratio:>10} {ms(arm, depth):>8.1f}')

        dense_v = vals("dense", depth)
        plain_v = vals("block_plain", depth)
        depth_out: Dict[str, object] = {}

        print()
        print("  Kernvergleich -- jeder Blockarm gegen die dichte Schicht:")
        for arm in arms:
            if arm == "dense":
                continue
            v = vals(arm, depth)
            p = float(stats.wilcoxon(v, dense_v).pvalue)
            wins = int((v > dense_v).sum())
            mark = "*signifikant*" if p < 0.05 else "(nicht signifikant)"
            print(f'    {arm:>18} vs dense: {v.mean()-dense_v.mean():>+8.4f}  '
                  f'{wins:>2}/{n} Siege  p={p:.4f} {mark}')
            depth_out[f"{arm}_vs_dense"] = {"delta": float(v.mean()-dense_v.mean()),
                                            "p": p, "wins": wins}

        print()
        print("  Traegt die Beschraenkung zusaetzlich etwas bei?")
        for arm in ("block_spectral", "block_orthogonal"):
            if arm not in arms:
                continue
            v = vals(arm, depth)
            p = float(stats.wilcoxon(v, plain_v).pvalue)
            mark = "*signifikant*" if p < 0.05 else "(nicht signifikant)"
            print(f'    {arm:>18} vs block_plain: {v.mean()-plain_v.mean():>+8.4f}  '
                  f'p={p:.4f} {mark}')
            depth_out[f"{arm}_vs_plain"] = {"delta": float(v.mean()-plain_v.mean()), "p": p}

        summary["per_depth"][str(depth)] = depth_out  # type: ignore[index]

    print()
    print("=" * 72)
    print("BEWERTUNG (Kriterien vor dem Lauf festgelegt)")
    for depth in depths:
        d = summary["per_depth"][str(depth)]      # type: ignore[index]
        spec_dense = d.get("block_spectral_vs_dense", {})
        spec_plain = d.get("block_spectral_vs_plain", {})
        beats_dense = spec_dense.get("delta", 0) > 0 and spec_dense.get("p", 1) < 0.05
        beats_plain = spec_plain.get("delta", 0) > 0 and spec_plain.get("p", 1) < 0.05
        if beats_dense and beats_plain:
            verdict = "H1 -- Aussage haelt vollstaendig (spektral > dicht UND > unbeschraenkt)"
        elif beats_dense:
            verdict = ("H2 -- Blockstruktur traegt, Spektralnormalisierung nicht nachweisbar. "
                       "Aussage auf 'Blockschicht' kuerzen.")
        else:
            verdict = "H0 -- kein signifikanter Vorteil gegenueber dicht. Kernaussage faellt."
        print(f"  depth {depth}: {verdict}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    recorder = run()
    result = analyse(recorder)
    recorder.save(analysis=result)
