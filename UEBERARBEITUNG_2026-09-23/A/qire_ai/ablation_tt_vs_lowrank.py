from __future__ import annotations

"""RESEARCH ID: A-001 -- Does the tensor-network structure beat plain low-rank
at a matched parameter budget?

RESEARCH QUESTION
Every result so far compared Tensor-Train against a *dense* layer. That
comparison cannot separate "tensor networks help" from "any low-rank
compression helps". If a plain rank-r factorisation W ~ U V matches TT at the
same parameter count, then the tensor-network structure -- and with it the
quantum-inspired framing of the headline result -- contributes nothing beyond
generic compression.

QUANTUM PRINCIPLE
None, deliberately. This is the Abgrenzung test: the non-quantum-inspired
baseline that the tensor-network claim has to survive.

PARAMETER BUDGETS (4096x4096 layer)
  TT, 3 cores rank 2   :  6,144    | low-rank r=1 : 8,192  (33% MORE)
  TT, 2 cores rank 2   : 20,480    | low-rank r=2 : 16,384 (20% FEWER)
  dense                : 16,781,312
Note that TT's budget is not reachable exactly by low-rank: a rank-1
factorisation of a 4096x4096 matrix already costs 8,192 parameters. Where the
budgets cannot be matched exactly, low-rank is given the *larger* budget, so
the comparison is conservative against TT rather than in its favour.

HYPOTHESES
  H1 (main)  : TT reaches higher accuracy than low-rank at equal-or-smaller budget.
  H0 (null)  : No measurable difference -- the tensor structure adds nothing
               beyond low-rank compression.
  H2 (alt)   : Any difference is explained by parameter count alone, not by
               structure, and vanishes once budgets are matched.

FALSIFICATION CRITERION
If low-rank matches or beats TT at its own (larger) budget, the tensor-network
framing of the headline result does not survive and must be restated.

MEASURED
Balanced accuracy (paired over seeds), parameter count, single-threaded
inference latency at several batch sizes.

Run: python -m qire_ai.ablation_tt_vs_lowrank
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


class LowRankLinear(nn.Module):
    """W ~ U @ V with inner rank r. The non-quantum-inspired control: it is
    ordinary low-rank factorisation, the standard method TT has to beat."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.v = nn.Linear(in_features, rank, bias=False)
        self.u = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.u(self.v(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SquareMiddleNet(nn.Module):
    """784 -> width -> width -> classes; only the square middle layer varies."""

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


def _build_middle(kind: str, width: int, a: int, b: int | None = None) -> nn.Module:
    if kind == "dense":
        return nn.Linear(width, width)
    if kind == "tt":
        return TTLinear(width, width, num_cores=a, tt_rank=b)
    if kind == "lowrank":
        return LowRankLinear(width, width, rank=a)
    raise ValueError(kind)


ARMS = [
    ("dense", "dense", None, None),
    ("tt_c3_r2", "tt", 3, 2),
    ("tt_c2_r2", "tt", 2, 2),
    ("lowrank_r1", "lowrank", 1, None),
    ("lowrank_r2", "lowrank", 2, None),
    ("lowrank_r5", "lowrank", 5, None),
]


def measure_latency(width: int = 4096, batches: Sequence[int] = (1, 8, 256)) -> Dict[str, Dict[int, float]]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    out: Dict[str, Dict[int, float]] = {}
    for label, kind, a, b in ARMS:
        module = _build_middle(kind, width, a, b).eval()
        fn = module.forward_streaming if kind == "tt" else module.forward
        per_batch = {}
        for batch in batches:
            x = torch.randn(batch, width)
            with torch.no_grad():
                for _ in range(5):
                    fn(x)
                reps = 100 if batch <= 8 else 20
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn(x)
            per_batch[batch] = (time.perf_counter() - t0) / reps * 1000
        out[label] = per_batch
    return out


def run(
    width: int = 4096,
    seeds: Sequence[int] = tuple(range(601, 611)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, List[float]]:
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    for seed in seeds:
        for label, kind, a, b in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            middle = _build_middle(kind, width, a, b)
            model = SquareMiddleNet(bundle.input_dim, width, bundle.num_classes, middle, label)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            params[label] = model.middle_parameters()
        print(f"  seed {seed} done")
    acc["__params__"] = params  # type: ignore[assignment]
    return acc


def analyse(acc: Dict[str, List[float]], latency: Dict[str, Dict[int, float]]) -> None:
    from scipy import stats

    params = acc.pop("__params__")  # type: ignore[arg-type]
    dense = np.array(acc["dense"])
    n = len(dense)
    batches = sorted(next(iter(latency.values())).keys())

    print()
    print(f"RESEARCH ID A-001 | MNIST, square {4096}x{4096} middle layer, {n} seeds")
    header = (f'{"arm":>12} {"accuracy":>9} {"std":>7} {"params":>12} '
              + " ".join(f'{f"b={b} ms":>10}' for b in batches))
    print(header); print("-" * len(header))
    for label in acc:
        vals = np.array(acc[label])
        lat = " ".join(f'{latency[label][b]:>10.3f}' for b in batches)
        print(f'{label:>12} {vals.mean():>9.4f} {vals.std(ddof=1):>7.4f} {params[label]:>12,} {lat}')

    print()
    print("Decisive comparisons (paired Wilcoxon, budget noted):")
    pairs = [("tt_c3_r2", "lowrank_r1"), ("tt_c2_r2", "lowrank_r2"), ("tt_c2_r2", "lowrank_r5")]
    for tt_label, lr_label in pairs:
        if tt_label not in acc or lr_label not in acc:
            continue
        a, b = np.array(acc[tt_label]), np.array(acc[lr_label])
        p = stats.wilcoxon(a, b).pvalue
        budget = "TT has FEWER params" if params[tt_label] < params[lr_label] else "TT has MORE params"
        verdict = "TT better" if a.mean() > b.mean() else "low-rank better"
        flag = " *significant*" if p < 0.05 else " (not significant)"
        print(f'  {tt_label} ({params[tt_label]:,}) vs {lr_label} ({params[lr_label]:,}): '
              f'{a.mean()-b.mean():+.4f} -> {verdict}, p={p:.4f}{flag}  [{budget}]')

    print()
    print("Falsification criterion: if low-rank matches or beats TT at its own")
    print("budget, the tensor-network framing of the headline result does not")
    print("survive and has to be restated as generic low-rank compression.")


if __name__ == "__main__":
    lat = measure_latency()
    analyse(run(), lat)
