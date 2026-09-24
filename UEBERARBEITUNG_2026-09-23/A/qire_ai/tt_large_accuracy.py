from __future__ import annotations

"""Does Tensor-Train keep its accuracy at the size where streaming finally wins?

benchmark_tt_streaming.py found streaming contraction beating a dense layer at
4096x4096 (1.54x) using only ~0.04% of the parameters -- but measured speed
only. Whether a rank-4 chain still has the capacity to learn at that width was
untested, so the speedup meant nothing on its own yet.

Two things are deliberately kept together here:

  1. Accuracy and speed are measured on the *same* configuration. The 1.54x
     figure came from a square 4096x4096 layer; MNIST's first layer is
     784x4096, a different shape, so quoting the square number for an MNIST
     model would be extrapolation. Both are measured directly instead.

  2. Training uses forward() (reconstruction) regardless of which path is
     faster at inference. The two paths were verified numerically identical,
     so the contraction path cannot affect accuracy -- only speed. Training
     via the cheaper path and reporting inference speed separately is
     therefore sound, not a shortcut.

Run: python -m qire_ai.tt_large_accuracy
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .models_baseline import DenseMLP, TensorTrainMLP
from .tensor_train import TTLinear


def _time(fn, x, repeats: int = 10) -> float:
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(x)
    return (time.perf_counter() - t0) / repeats * 1000


def speed_at_shape(in_features: int, out_features: int, cores: int, ranks: Sequence[int], batch: int = 256) -> None:
    """Measure the actual MNIST-shaped layer, rather than reusing the square number."""
    torch.manual_seed(0)
    dense = nn.Linear(in_features, out_features).eval()
    x = torch.randn(batch, in_features)
    t_dense = _time(dense, x)
    print(f"Layer {in_features}x{out_features}, batch {batch} (single-threaded)")
    print(f'{"rank":>6} {"dense ms":>9} {"recon ms":>9} {"stream ms":>10} {"stream/dense":>13} {"param frac":>11}')
    print("-" * 62)
    for rank in ranks:
        layer = TTLinear(in_features, out_features, num_cores=cores, tt_rank=rank).eval()
        t_recon = _time(layer.forward, x)
        t_stream = _time(layer.forward_streaming, x)
        frac = layer.num_tt_parameters() / (in_features * out_features + out_features)
        mark = "  <-- faster" if t_stream < t_dense else ""
        print(f'{rank:>6} {t_dense:>9.2f} {t_recon:>9.2f} {t_stream:>10.2f} '
              f'{t_stream/t_dense:>12.2f}x {frac:>11.5f}{mark}')


class SquareMiddleMLP(nn.Module):
    """784 -> width -> width -> classes, with the square middle layer either dense
    or Tensor-Train. The square layer is the one where streaming contraction was
    measured to beat dense (2.08x at 4096); MNIST's 784x4096 input layer is a
    different shape and does *not* benefit (1.66x slower), so compressing the
    input layer instead would not pair with that speed result."""

    model_name = "SquareMiddleMLP"

    def __init__(self, input_dim: int, width: int, num_classes: int,
                 tt_rank: int | None = None, num_cores: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        if tt_rank is None:
            self.middle = nn.Linear(width, width)
            self.method = f"square_dense_w{width}"
        else:
            self.middle = TTLinear(width, width, num_cores=num_cores, tt_rank=tt_rank)
            self.method = f"square_tt_r{tt_rank}_w{width}"
        self.fc_out = nn.Linear(width, num_classes)
        self.width = width

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        h = torch.relu(self.middle(h))
        logits = self.fc_out(h)
        return logits, {"active_model_parts": 1.0, "gate_entropy": 0.0,
                        "router_ops": 0, "expert_ops": 0}

    def middle_parameters(self) -> int:
        return sum(p.numel() for p in self.middle.parameters())


def run_square(
    width: int = 4096,
    ranks: Sequence[int] = (4, 16),
    seeds: Sequence[int] = (401, 402, 403),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> List[Dict[str, object]]:
    """Accuracy at exactly the configuration where streaming wins on speed."""
    acc: Dict[str, List[float]] = {}
    mid: Dict[str, int] = {}
    for seed in seeds:
        for label, rank in [("dense", None)] + [(f"tt_rank{r}", r) for r in ranks]:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = SquareMiddleMLP(bundle.input_dim, width, bundle.num_classes, tt_rank=rank)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            mid[label] = model.middle_parameters()
        print(f"  seed {seed} done")

    base = float(np.mean(acc["dense"]))
    return [{
        "arm": label,
        "accuracy": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "drop": base - float(np.mean(vals)),
        "middle_params": mid[label],
        "middle_fraction": mid[label] / mid["dense"],
    } for label, vals in acc.items()]


def run(
    hidden: int = 4096,
    ranks: Sequence[int] = (4, 16),
    seeds: Sequence[int] = (401, 402, 403),
    max_samples: int = 10000,
    epochs: int = 15,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> List[Dict[str, object]]:
    acc: Dict[str, List[float]] = {}
    par: Dict[str, int] = {}

    for seed in seeds:
        set_seed(seed)
        torch.set_num_threads(threads)
        bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)

        dense = DenseMLP(bundle.input_dim, hidden, bundle.num_classes)
        train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                    seed=seed, batch_size=batch_size)
        _, _, _, ex = evaluate(dense, bundle.x_test, bundle.y_test)
        acc.setdefault("dense", []).append(float(ex["balanced_accuracy"]))
        par["dense"] = sum(p.numel() for p in dense.parameters())

        for rank in ranks:
            label = f"tt_rank{rank}"
            set_seed(seed)
            torch.set_num_threads(threads)
            model = TensorTrainMLP(bundle.input_dim, hidden, bundle.num_classes,
                                   tt_rank=rank, num_cores=4)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            par[label] = sum(p.numel() for p in model.parameters())
        print(f"  seed {seed} done")

    base = float(np.mean(acc["dense"]))
    rows = []
    for label in acc:
        rows.append({
            "arm": label,
            "accuracy": float(np.mean(acc[label])),
            "std": float(np.std(acc[label], ddof=1)) if len(acc[label]) > 1 else 0.0,
            "drop": base - float(np.mean(acc[label])),
            "params": par[label],
            "param_fraction": par[label] / par["dense"],
        })
    return rows


def analyse(rows: List[Dict[str, object]]) -> None:
    print()
    header = f'{"arm":>12} {"accuracy":>9} {"std":>7} {"drop":>8} {"params":>12} {"param frac":>11}'
    print(header); print("-" * len(header))
    for r in rows:
        print(f'{r["arm"]:>12} {r["accuracy"]:>9.4f} {r["std"]:>7.4f} {r["drop"]:>+8.4f} '
              f'{r["params"]:>12,} {r["param_fraction"]:>11.5f}')


def run_configs(
    configs: Sequence[tuple] = ((None, None), (2, 2), (3, 2), (4, 4)),
    width: int = 4096,
    seeds: Sequence[int] = tuple(range(501, 513)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, List[float]]:
    """Per-seed accuracies for several (cores, rank) settings on the square layer.

    Returns raw per-seed values rather than summaries, so a paired test can be run
    afterwards -- the headline claim rests on this, and 3 seeds (as used in the
    first pass) is not enough to support it. Seeds are fresh (501+), disjoint from
    the 401-403 used before, so this is an out-of-sample check rather than a
    re-description of the run that produced the hypothesis.
    """
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    for seed in seeds:
        for cores, rank in configs:
            label = "dense" if rank is None else f"tt_c{cores}_r{rank}"
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = SquareMiddleMLP(bundle.input_dim, width, bundle.num_classes,
                                    tt_rank=rank, num_cores=cores or 4)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            params[label] = model.middle_parameters()
        print(f"  seed {seed} done")
    acc["_params"] = params  # type: ignore[assignment]
    return acc


def analyse_configs(acc: Dict[str, List[float]]) -> None:
    from scipy import stats

    params = acc.pop("_params")  # type: ignore[arg-type]
    dense = np.array(acc["dense"])
    n = len(dense)
    print()
    print(f"MNIST, square 4096x4096 middle layer, {n} fresh seeds")
    header = (f'{"arm":>14} {"accuracy":>9} {"std":>7} {"vs dense":>10} '
              f'{"params":>12} {"fraction":>10} {"wins":>7} {"p":>8}')
    print(header); print("-" * len(header))
    for label, values in acc.items():
        vals = np.array(values)
        if label == "dense":
            print(f'{label:>14} {vals.mean():>9.4f} {vals.std(ddof=1):>7.4f} {"—":>10} '
                  f'{params[label]:>12,} {1.0:>10.5f} {"—":>7} {"—":>8}')
            continue
        diff = vals - dense
        p = stats.wilcoxon(vals, dense).pvalue
        flag = " *" if p < 0.05 else ""
        print(f'{label:>14} {vals.mean():>9.4f} {vals.std(ddof=1):>7.4f} {diff.mean():>+10.4f} '
              f'{params[label]:>12,} {params[label]/params["dense"]:>10.5f} '
              f'{int((diff>0).sum()):>3}/{n:<3} {p:>8.4f}{flag}')
    print()
    print("* = significant at p<0.05. A non-significant difference is the claim that")
    print("  matters here: compression at no measurable accuracy cost. Claiming the")
    print("  compressed model is *better* would need a significant positive result,")
    print("  which an earlier apparent effect of this kind failed to reproduce.")


def analyse_square(rows: List[Dict[str, object]]) -> None:
    print()
    header = f'{"arm":>12} {"accuracy":>9} {"std":>7} {"drop":>8} {"middle params":>14} {"fraction":>10}'
    print(header); print("-" * len(header))
    for r in rows:
        print(f'{r["arm"]:>12} {r["accuracy"]:>9.4f} {r["std"]:>7.4f} {r["drop"]:>+8.4f} '
              f'{r["middle_params"]:>14,} {r["middle_fraction"]:>10.5f}')


if __name__ == "__main__":
    print("=== Speed at the shape MNIST actually uses (784 -> 4096) ===")
    speed_at_shape(784, 4096, cores=4, ranks=(4, 16))
    print()
    print("=== Speed at the square shape where streaming wins (4096 -> 4096) ===")
    speed_at_shape(4096, 4096, cores=4, ranks=(4, 16))
    print()
    print("=== Accuracy at that same square configuration ===")
    analyse_square(run_square())
