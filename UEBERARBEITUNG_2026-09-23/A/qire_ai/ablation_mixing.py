from __future__ import annotations

"""RESEARCH ID: A-003 -- Does block-diagonal only win because dense neighbours
do its mixing for it?

BACKGROUND (A-002)
Block-diagonal beat Tensor-Train at an identical 20,480-parameter budget
(-0.0168, p=0.004), which falsified the claim that TT's bond chain is what
helps. But A-002 has a weakness I need to test before retracting anything:
information never crosses a block boundary in a block-diagonal layer, yet the
test network puts a *dense* fc_in before it and a *dense* fc_out after it.
Both mix globally. Block-diagonal may simply be riding on its neighbours.

TT does not have that dependency: its bonds carry correlation across the whole
index range inside the layer itself.

DESIGN
Two architectures, same compressed layer type in each:

  shallow  784 -> [dense] -> 4096 -> [COMPRESSED] -> 4096 -> [dense] -> 10
           (the A-002 setup: dense neighbours available to mix)

  deep     784 -> [dense] -> 4096 -> [COMPRESSED] x3 -> 4096 -> [dense] -> 10
           (three compressed layers in a row; mixing must happen inside them)

If block-diagonal was relying on its neighbours, stacking three of them should
hurt it disproportionately: with block size 4, three stacked block-diagonal
layers still never mix beyond a 4-dimensional group, so the network cannot
build any feature spanning more than 4 of the 4096 units. TT's bonds, by
contrast, propagate across the full index range within each layer.

HYPOTHESES
  H1 (main) : block-diagonal degrades sharply in `deep`, TT holds -> bonds do
              provide the global mixing that pure locality cannot, and A-002's
              verdict was an artefact of the shallow architecture.
  H0 (null) : both behave the same in `deep` -> A-002 stands, bonds are not
              needed, and the quantum-inspired claim does not survive.
  H2 (alt)  : both degrade -> neither structure supports deep stacking, and the
              comparison says nothing about bonds either way.

FALSIFICATION
H1 requires an interaction: block-diagonal must lose *more* than TT when going
from shallow to deep. A uniform drop for both supports H2, not H1.

Run: python -m qire_ai.ablation_mixing
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .ablation_structure import BlockDiagonal, LowRankLinear, build_middle
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate


class StackedNet(nn.Module):
    """784 -> width -> [compressed] x depth -> classes."""

    model_name = "StackedNet"

    def __init__(self, input_dim: int, width: int, num_classes: int,
                 label: str, depth: int):
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

    def compressed_parameters(self) -> int:
        return sum(p.numel() for m in self.middles for p in m.parameters())


ARMS = ["tt_c2_r2", "block_diag", "lowrank_r2", "dense"]
DEPTHS = (1, 3)


def run(
    width: int = 4096,
    seeds: Sequence[int] = tuple(range(801, 809)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, object]:
    acc: Dict[tuple, List[float]] = {}
    params: Dict[tuple, int] = {}
    for seed in seeds:
        for depth in DEPTHS:
            for label in ARMS:
                set_seed(seed)
                torch.set_num_threads(threads)
                bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
                model = StackedNet(bundle.input_dim, width, bundle.num_classes, label, depth)
                train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                            seed=seed, batch_size=batch_size)
                _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
                acc.setdefault((label, depth), []).append(float(ex["balanced_accuracy"]))
                params[(label, depth)] = model.compressed_parameters()
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[tuple, List[float]] = result["acc"]      # type: ignore[assignment]
    params: Dict[tuple, int] = result["params"]        # type: ignore[assignment]
    n = len(next(iter(acc.values())))

    print()
    print(f"RESEARCH ID A-003 | MNIST, width 4096, {n} seeds")
    header = (f'{"arm":>14} {"depth 1":>18} {"depth 3":>18} {"Aenderung":>11}')
    print(header); print("-" * len(header))
    change: Dict[str, float] = {}
    for label in ARMS:
        a1 = np.array(acc[(label, 1)])
        a3 = np.array(acc[(label, 3)])
        change[label] = a3.mean() - a1.mean()
        print(f'{label:>14} {a1.mean():>9.4f} +/-{a1.std(ddof=1):<6.4f} '
              f'{a3.mean():>9.4f} +/-{a3.std(ddof=1):<6.4f} {change[label]:>+11.4f}')

    print()
    print("Interaktionstest -- verliert Block-Diagonal beim Stapeln mehr als TT?")
    tt1, tt3 = np.array(acc[("tt_c2_r2", 1)]), np.array(acc[("tt_c2_r2", 3)])
    bd1, bd3 = np.array(acc[("block_diag", 1)]), np.array(acc[("block_diag", 3)])
    tt_delta, bd_delta = tt3 - tt1, bd3 - bd1
    p_inter = stats.wilcoxon(bd_delta, tt_delta).pvalue
    print(f'  TT Aenderung           : {tt_delta.mean():+.4f}')
    print(f'  Block-Diagonal Aenderung: {bd_delta.mean():+.4f}')
    print(f'  Differenz der Aenderungen: {bd_delta.mean()-tt_delta.mean():+.4f}, p={p_inter:.4f}'
          + (" *signifikant*" if p_inter < 0.05 else " (nicht signifikant)"))

    print()
    print("Direktvergleich bei depth 3 (dort muss die Mischung intern passieren):")
    p_deep = stats.wilcoxon(tt3, bd3).pvalue
    winner = "TT" if tt3.mean() > bd3.mean() else "Block-Diagonal"
    print(f'  TT {tt3.mean():.4f} vs Block-Diagonal {bd3.mean():.4f} -> {winner} vorn, '
          f'p={p_deep:.4f}' + (" *signifikant*" if p_deep < 0.05 else " (nicht signifikant)"))

    print()
    print("Auswertung:")
    print("  Block-Diagonal bricht ein, TT haelt -> Bonds leisten die globale Mischung (H1)")
    print("  beide gleich              -> A-002 steht, Bonds nicht noetig (H0)")
    print("  beide brechen ein         -> Aussage ueber Bonds nicht moeglich (H2)")


if __name__ == "__main__":
    analyse(run())
