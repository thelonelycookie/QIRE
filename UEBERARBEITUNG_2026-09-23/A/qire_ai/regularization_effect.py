from __future__ import annotations

"""Is a structural constraint sometimes *better* than the dense model, and if
so, is that a regularization effect?

In compression_regimes.py and regime_generalization.py several constrained
models beat the dense baseline rather than merely matching it (negative
accuracy "drops"). The effects were tiny (0.1-0.7 percentage points) and each
came from 5 seeds, so they could easily be noise. Two questions follow:

  1. Does the effect survive proper statistical testing?
  2. If it is real, is the mechanism regularization? A regularizer should help
     exactly where the dense model overfits, and hurt where capacity is
     genuinely needed.

Methodological note: the configurations tested here were picked by looking at
which ones looked promising in the earlier runs. Re-testing them on the same
seeds would be circular -- the apparent effect could just be the noise that
made them stand out. This module therefore uses a disjoint set of seeds
(101-125) that no earlier experiment touched, making it an out-of-sample
replication rather than a re-description of the data that generated the
hypothesis.

Train accuracy is recorded alongside test accuracy so the overfitting
mechanism can be tested directly, rather than assumed.

Run: python -m qire_ai.regularization_effect
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .models_baseline import DenseMLP, TensorTrainMLP

# (label, builder) -- chosen from earlier runs, now re-tested on fresh seeds
CONFIGS = {
    "tensor_train_r8": lambda b, h: TensorTrainMLP(b.input_dim, h, b.num_classes, tt_rank=8, num_cores=3),
    "width_16": lambda b, h: DenseMLP(b.input_dim, 16, b.num_classes),
}

FRESH_SEEDS = tuple(range(101, 126))  # disjoint from every earlier experiment


def _accuracies(model, bundle) -> Tuple[float, float]:
    _, _, _, extras_test = evaluate(model, bundle.x_test, bundle.y_test)
    _, _, _, extras_train = evaluate(model, bundle.x_train, bundle.y_train)
    return (
        float(extras_test.get("balanced_accuracy", 0.0)),
        float(extras_train.get("balanced_accuracy", 0.0)),
    )


def run(
    datasets: Sequence[str] = ("wine", "breast_cancer", "iris", "digits"),
    seeds: Sequence[int] = FRESH_SEEDS,
    hidden_dim: int = 32,
    epochs: int = 80,
    lr: float = 0.01,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            set_seed(seed)
            bundle = load_dataset(dataset, seed=seed)

            dense = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
            train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
            dense_test, dense_train = _accuracies(dense, bundle)

            for label, builder in CONFIGS.items():
                set_seed(seed)
                model = builder(bundle, hidden_dim)
                train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
                test_acc, train_acc = _accuracies(model, bundle)
                records.append({
                    "dataset": dataset, "config": label, "seed": seed,
                    "dense_test": dense_test, "dense_train": dense_train,
                    "model_test": test_acc, "model_train": train_acc,
                    "benefit": test_acc - dense_test,              # >0 means constrained wins
                    "dense_overfit": dense_train - dense_test,     # how much the dense model overfits
                    "model_overfit": train_acc - test_acc,
                })
        print(f"done: {dataset}")
    return records


def analyse(records: List[Dict[str, object]]) -> None:
    from scipy import stats

    datasets = sorted({r["dataset"] for r in records}, key=lambda d: [x["dataset"] for x in records].index(d))
    configs = sorted({r["config"] for r in records})

    print()
    print("Q1: Does the constrained model beat the dense one? (paired over fresh seeds)")
    header = f'{"dataset":>15} {"config":>16} {"dense":>8} {"model":>8} {"benefit":>9} {"wins":>7} {"p":>9}'
    print(header); print("-" * len(header))
    for dataset in datasets:
        for config in configs:
            sub = [r for r in records if r["dataset"] == dataset and r["config"] == config]
            dense = np.array([r["dense_test"] for r in sub])
            model = np.array([r["model_test"] for r in sub])
            benefit = model - dense
            try:
                p = stats.wilcoxon(model, dense).pvalue
            except ValueError:
                p = float("nan")
            wins = int((benefit > 0).sum())
            flag = " *" if p < 0.05 else ""
            print(f'{dataset:>15} {config:>16} {dense.mean():>8.4f} {model.mean():>8.4f} '
                  f'{benefit.mean():>+9.4f} {wins:>3}/{len(sub):<3} {p:>9.4f}{flag}')

    print()
    print("Q2: Is the mechanism regularization? A regularizer should help where")
    print("    the dense model overfits most.")
    overfit = np.array([r["dense_overfit"] for r in records])
    benefit = np.array([r["benefit"] for r in records])
    r_s, p_s = stats.spearmanr(overfit, benefit)
    r_p, p_p = stats.pearsonr(overfit, benefit)
    print(f'    Spearman(dense overfit, benefit) = {r_s:+.3f} (p={p_s:.5f})')
    print(f'    Pearson (dense overfit, benefit) = {r_p:+.3f} (p={p_p:.5f})')
    print()
    print("    WARNING -- this seed-level correlation is NOT interpretable as evidence.")
    print("    dense_overfit = dense_train - dense_test and benefit = model_test -")
    print("    dense_test both contain -dense_test, so a seed with a low dense_test")
    print("    scores high on both by construction. Simulating independent models")
    print("    with no true relationship still yields correlations near +0.5 to +0.7")
    print("    (depending on how little dense_train varies). A measured value below")
    print("    that null is evidence against the hypothesis, not for it. Read the")
    print("    per-dataset table below instead -- averaging over seeds removes the")
    print("    shared term, so the between-dataset comparison is the honest test.")
    print()
    print(f'{"dataset":>15} {"dense overfit":>14} {"mean benefit":>13}')
    print("-" * 45)
    for dataset in datasets:
        sub = [r for r in records if r["dataset"] == dataset]
        o = np.mean([r["dense_overfit"] for r in sub])
        b = np.mean([r["benefit"] for r in sub])
        print(f'{dataset:>15} {o:>14.4f} {b:>+13.4f}')
    print()
    print("    A positive correlation supports regularization: the constraint helps")
    print("    where the dense model was overfitting, and hurts where it was not.")


if __name__ == "__main__":
    analyse(run())
