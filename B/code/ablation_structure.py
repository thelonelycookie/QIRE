from __future__ import annotations

"""RESEARCH ID: A-002 -- Which structural property gives Tensor-Train its edge?

BACKGROUND (A-001)
At an identical 20,480-parameter budget, TT beat a single low-rank bottleneck
by +0.0393 (p=0.002). That rules out "any compression works", but not the
weaker explanation that *several sequential bottlenecks* beat one, regardless
of tensor structure. A-001 compared a chain against a single cut, so two
properties were confounded.

TT has two distinguishable structural properties:
  (a) LOCALITY -- each core acts on one index of a factorised input, not on
      the whole vector at once
  (b) BONDS -- correlation is carried between those local factors by a shared
      bond index

This experiment separates them with controls that have one property but not
the other, all at the same parameter budget:

  arm                property               params
  ------------------------------------------------
  dense              neither (full matrix)  16,781,312
  tt_c2_r2           local + bonds              20,480
  lowrank_r2         global, single cut         20,480
  deep_bottleneck    global, chained          ~20,494
  block_diag         local, no bonds            20,480

Block-diagonal uses 1024 independent 4x4 blocks: strictly local, no
information crosses blocks -- locality without bonds. Deep bottleneck is
4096 -> r -> r -> r -> 4096: sequential composition of global maps -- chaining
without locality.

HYPOTHESES
  H1 (main) : TT beats all three controls -> the *combination* of locality and
              bonds matters, which is exactly the MPS structure.
  H0 (null) : TT matches deep_bottleneck -> the advantage is sequential
              composition, not tensor structure. The quantum framing weakens
              substantially.
  H2 (alt)  : TT matches block_diag -> the advantage is locality alone; bonds
              are decoration and a cheaper block-diagonal layer would do.

FALSIFICATION
If either control matches TT at equal budget, the specific claim "the bond
chain is what helps" does not survive in its current form.

Run: python -m qire_ai.ablation_structure
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .tensor_train import TTLinear


class DeepBottleneck(nn.Module):
    """4096 -> r -> r -> ... -> 4096. Sequential composition of global maps:
    chaining WITHOUT locality. Controls for 'several bottlenecks beat one'."""

    def __init__(self, dim: int, rank: int, stages: int = 2):
        super().__init__()
        layers = [nn.Linear(dim, rank)]
        for _ in range(stages):
            layers.append(nn.Linear(rank, rank))
        layers.append(nn.Linear(rank, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BlockDiagonal(nn.Module):
    """`blocks` independent square blocks along the diagonal. Strictly local --
    no information crosses a block boundary: locality WITHOUT bonds."""

    def __init__(self, dim: int, block_size: int):
        super().__init__()
        assert dim % block_size == 0, "block_size must divide dim"
        self.dim, self.block_size = dim, block_size
        self.blocks = dim // block_size
        self.weight = nn.Parameter(torch.empty(self.blocks, block_size, block_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        batch = x.shape[0]
        h = x.reshape(batch, self.blocks, self.block_size)
        h = torch.bmm(h.transpose(0, 1), self.weight).transpose(0, 1)
        return h.reshape(batch, self.dim) + self.bias


class SquareMiddleNet(nn.Module):
    model_name = "SquareMiddleNet"

    def __init__(self, input_dim: int, width: int, num_classes: int, middle: nn.Module, label: str):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.middle = middle
        self.fc_out = nn.Linear(width, num_classes)
        self.method = label

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        h = torch.relu(self.middle(h))
        return self.fc_out(h), {"active_model_parts": 1.0, "gate_entropy": 0.0,
                                "router_ops": 0, "expert_ops": 0}

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


class LowRankLinear(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim)

    def forward(self, x):
        return self.u(self.v(x))


def build_middle(label: str, dim: int) -> nn.Module:
    if label == "dense":
        return nn.Linear(dim, dim)
    if label == "tt_c2_r2":
        return TTLinear(dim, dim, num_cores=2, tt_rank=2)
    if label == "lowrank_r2":
        return LowRankLinear(dim, rank=2)
    if label == "deep_bottleneck":
        return DeepBottleneck(dim, rank=2, stages=2)
    if label == "block_diag":
        return BlockDiagonal(dim, block_size=4)
    raise ValueError(label)


ARMS = ["dense", "tt_c2_r2", "lowrank_r2", "deep_bottleneck", "block_diag"]


def run(
    width: int = 4096,
    seeds: Sequence[int] = tuple(range(701, 711)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, object]:
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    for seed in seeds:
        for label in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = SquareMiddleNet(bundle.input_dim, width, bundle.num_classes,
                                    build_middle(label, width), label)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            params[label] = model.middle_parameters()
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]  # type: ignore[assignment]
    params: Dict[str, int] = result["params"]  # type: ignore[assignment]
    n = len(acc["dense"])

    props = {
        "dense": "voll",
        "tt_c2_r2": "lokal + Bonds",
        "lowrank_r2": "global, ein Schnitt",
        "deep_bottleneck": "global, verkettet",
        "block_diag": "lokal, ohne Bonds",
    }
    print()
    print(f"RESEARCH ID A-002 | MNIST, 4096x4096 middle layer, {n} seeds")
    header = f'{"arm":>16} {"Struktur":>22} {"accuracy":>9} {"std":>7} {"params":>12}'
    print(header); print("-" * len(header))
    for label in ARMS:
        vals = np.array(acc[label])
        print(f'{label:>16} {props[label]:>22} {vals.mean():>9.4f} '
              f'{vals.std(ddof=1):>7.4f} {params[label]:>12,}')

    tt = np.array(acc["tt_c2_r2"])
    print()
    print("TT gegen jede Kontrollbedingung (gepaarter Wilcoxon, gleiches Budget):")
    for label in ["lowrank_r2", "deep_bottleneck", "block_diag"]:
        other = np.array(acc[label])
        p = stats.wilcoxon(tt, other).pvalue
        flag = " *signifikant*" if p < 0.05 else " (nicht signifikant)"
        print(f'  vs {label:>16} ({params[label]:>7,}): {tt.mean()-other.mean():+.4f}, p={p:.4f}{flag}')

    print()
    print("Auswertung:")
    print("  TT schlaegt alle drei  -> Lokalitaet UND Bonds noetig (MPS-Struktur bestaetigt)")
    print("  TT ~ deep_bottleneck   -> Vorteil ist Verkettung, nicht Tensorstruktur")
    print("  TT ~ block_diag        -> Vorteil ist Lokalitaet, Bonds sind Dekoration")


if __name__ == "__main__":
    analyse(run())
