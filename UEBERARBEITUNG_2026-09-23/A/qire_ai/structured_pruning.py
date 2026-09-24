from __future__ import annotations

"""Structured pruning: does removing whole neurons finally buy real wall-clock
time, and does fine-tuning rescue the post-hoc regime?

Two things make structured pruning different from everything tried so far in
this project:

  1. Removing whole hidden neurons makes the weight matrices *literally
     smaller*. Unlike magnitude masking (which leaves a same-size matrix full
     of zeros) or Tensor-Train (which reconstructs a full matrix each call),
     there is no orchestration overhead added -- the same dense matmul just
     runs on a smaller matrix. If any technique in this project can convert a
     resource saving into measured time, it is this one.

  2. regime_generalization.py found `width` had the *largest* post-hoc penalty
     (17.4x). But that test pruned and stopped. The standard practical recipe
     is train -> prune -> fine-tune, and the fine-tuning step is exactly what
     the post-hoc arm was missing. So this also probes a boundary of the
     regime principle: does a short fine-tune close a gap that large?

Part A compares three arms on accuracy:
    dense        hidden=32, the baseline
    scratch      hidden=16 trained from initialization (compressed training)
    prune+ft     hidden=32 trained, pruned to 16, then fine-tuned

Part B measures inference wall-clock across model sizes, to find where (if
anywhere) halving the neuron count halves the time. Single-threaded to match
qire_ai.benchmark.set_seed(), so the numbers are comparable to every other
timing figure in this project.

Run: python -m qire_ai.structured_pruning
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .models_baseline import DenseMLP


def prune_to_width(dense: DenseMLP, keep: int) -> DenseMLP:
    """Keep the `keep` most important hidden neurons, scored by
    ||incoming|| * ||outgoing||, and return a genuinely smaller model."""
    w1 = dense.net[0].weight.detach()
    b1 = dense.net[0].bias.detach()
    w2 = dense.net[2].weight.detach()
    b2 = dense.net[2].bias.detach()
    importance = w1.norm(dim=1) * w2.norm(dim=0)
    idx = torch.argsort(importance, descending=True)[:keep]
    small = DenseMLP(dense.input_dim, keep, dense.num_classes)
    with torch.no_grad():
        small.net[0].weight.copy_(w1[idx])
        small.net[0].bias.copy_(b1[idx])
        small.net[2].weight.copy_(w2[:, idx])
        small.net[2].bias.copy_(b2)
    return small


def _test_accuracy(model, bundle) -> float:
    _, _, _, extras = evaluate(model, bundle.x_test, bundle.y_test)
    return float(extras.get("balanced_accuracy", 0.0))


def part_a(
    datasets: Sequence[str] = ("wine", "breast_cancer", "digits"),
    seeds: Sequence[int] = tuple(range(201, 216)),
    hidden_dim: int = 32,
    keep: int = 16,
    epochs: int = 80,
    finetune_epochs: int = 20,
    lr: float = 0.01,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            set_seed(seed)
            bundle = load_dataset(dataset, seed=seed)

            dense = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
            train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
            dense_acc = _test_accuracy(dense, bundle)

            set_seed(seed)
            narrow = DenseMLP(bundle.input_dim, keep, bundle.num_classes)
            train_model(narrow, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
            scratch_acc = _test_accuracy(narrow, bundle)

            pruned = prune_to_width(dense, keep)
            pruned_acc_before = _test_accuracy(pruned, bundle)
            train_model(pruned, bundle.x_train, bundle.y_train, epochs=finetune_epochs, lr=lr, seed=seed)
            pruned_acc = _test_accuracy(pruned, bundle)

            records.append({
                "dataset": dataset, "seed": seed,
                "dense": dense_acc, "scratch": scratch_acc,
                "pruned_no_ft": pruned_acc_before, "pruned_ft": pruned_acc,
            })
        print(f"done: {dataset}")
    return records


def analyse_a(records: List[Dict[str, object]]) -> None:
    from scipy import stats

    datasets = []
    for r in records:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    print()
    header = f'{"dataset":>15} {"dense":>8} {"no-ft":>8} {"prune+ft":>9} {"scratch":>8} {"ft vs dense":>12} {"p":>8}'
    print(header); print("-" * len(header))
    for dataset in datasets:
        sub = [r for r in records if r["dataset"] == dataset]
        dense = np.array([r["dense"] for r in sub])
        noft = np.array([r["pruned_no_ft"] for r in sub])
        ft = np.array([r["pruned_ft"] for r in sub])
        scr = np.array([r["scratch"] for r in sub])
        p = stats.wilcoxon(ft, dense).pvalue
        print(f'{dataset:>15} {dense.mean():>8.4f} {noft.mean():>8.4f} {ft.mean():>9.4f} '
              f'{scr.mean():>8.4f} {ft.mean()-dense.mean():>+12.4f} {p:>8.4f}')

    dense = np.array([r["dense"] for r in records])
    noft = np.array([r["pruned_no_ft"] for r in records])
    ft = np.array([r["pruned_ft"] for r in records])
    scr = np.array([r["scratch"] for r in records])

    print()
    print("Does fine-tuning rescue the post-hoc arm?")
    print(f'  drop without fine-tuning : {(dense-noft).mean():.4f}')
    print(f'  drop with fine-tuning    : {(dense-ft).mean():.4f}')
    print(f'  drop training narrow     : {(dense-scr).mean():.4f}')
    p_ft = stats.wilcoxon(ft, noft).pvalue
    p_vs = stats.wilcoxon(ft, scr).pvalue
    print(f'  fine-tune vs no fine-tune: p={p_ft:.6f}')
    print(f'  fine-tune vs from-scratch: p={p_vs:.6f}')


def part_b(
    input_dim: int = 256,
    num_classes: int = 10,
    hidden_sizes: Sequence[int] = (32, 128, 512, 2048),
    batch_sizes: Sequence[int] = (32, 1000, 10000),
    repeats: int = 30,
) -> List[Dict[str, object]]:
    """Measured inference time when the hidden layer is halved. No masking, no
    reconstruction -- just a smaller dense matmul."""
    torch.set_num_threads(1)
    rows = []
    print()
    print("Part B -- measured inference time, hidden h vs h/2 (single-threaded)")
    print(f'{"hidden":>8} {"batch":>8} {"full ms":>10} {"half ms":>10} {"speedup":>9}')
    print("-" * 50)
    for h in hidden_sizes:
        torch.manual_seed(0)
        full = DenseMLP(input_dim, h, num_classes).eval()
        half = DenseMLP(input_dim, h // 2, num_classes).eval()
        for batch in batch_sizes:
            x = torch.randn(batch, input_dim)
            times = {}
            for name, model in (("full", full), ("half", half)):
                with torch.no_grad():
                    for _ in range(5):
                        model(x)
                    t0 = time.perf_counter()
                    for _ in range(repeats):
                        model(x)
                    times[name] = (time.perf_counter() - t0) / repeats * 1000
            speedup = times["full"] / times["half"]
            rows.append({"hidden": h, "batch": batch, "full_ms": times["full"],
                         "half_ms": times["half"], "speedup": speedup})
            print(f'{h:>8} {batch:>8} {times["full"]:>10.4f} {times["half"]:>10.4f} {speedup:>8.2f}x')
    print()
    print("A speedup of 2.00x would be the ideal for halving the neurons;")
    print("1.00x means the saving did not materialise in measured time at all.")
    return rows


if __name__ == "__main__":
    analyse_a(part_a())
    part_b()
