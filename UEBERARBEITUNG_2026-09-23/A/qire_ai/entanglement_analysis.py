from __future__ import annotations

"""Does the entanglement entropy of a trained weight matrix predict where
Tensor-Train compression works?

Motivation (see ROADMAP.md): TTLinear with a fixed tt_rank=4 preserved accuracy
on wine and breast_cancer but caused a statistically significant accuracy loss
on digits. In quantum many-body physics, whether a state can be represented
compactly as a Matrix-Product-State is governed by its entanglement entropy:
low entanglement -> a small bond dimension suffices, high entanglement -> it
does not. If weight matrices behave the same way, the entropy should have
predicted the digits failure *before* running the experiment.

This module measures that. For each dataset it trains the same dense MLP the
comparison used, reshapes its first-layer weight into the same higher-order
tensor TTLinear would use, and computes at every bond:

  - the Schmidt spectrum (singular values of the bipartition)
  - the entanglement entropy S = -sum p_i ln p_i, with p_i = sigma_i^2 / sum sigma^2
  - the effective rank exp(S)
  - `captured`, the fraction of spectral weight the actual bond dimension keeps

`captured` is the operative quantity: it is exactly the part of the matrix a
rank-r truncation can still represent. A random matrix is included as a
control -- if trained matrices show lower entanglement than random ones, the
structure is something training produces, not an artifact of the dimensions.

Run: python -m qire_ai.entanglement_analysis
"""

from typing import Dict, List, Sequence

import numpy as np
import torch

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .models_baseline import DenseMLP
from .tensor_train import factorize_balanced


def bond_spectra(weight: torch.Tensor, in_factors: Sequence[int], out_factors: Sequence[int]) -> List[np.ndarray]:
    """Singular-value spectrum at each TT bond of `weight` (out_features x in_features).

    Mirrors TTLinear's index structure: the matrix is zero-padded up to
    (prod(out_factors) x prod(in_factors)), reshaped to
    (m_1..m_d, n_1..n_d), permuted to the interleaved (m_1,n_1,...,m_d,n_d)
    order the cores use, and cut between core k and core k+1.
    """
    d = len(in_factors)
    padded_out = int(np.prod(out_factors))
    padded_in = int(np.prod(in_factors))
    w = torch.zeros(padded_out, padded_in, dtype=weight.dtype)
    w[: weight.shape[0], : weight.shape[1]] = weight

    tensor = w.reshape(*out_factors, *in_factors)
    # (m_1..m_d, n_1..n_d) -> (m_1, n_1, m_2, n_2, ...)
    perm: List[int] = []
    for k in range(d):
        perm.extend([k, d + k])
    tensor = tensor.permute(*perm).contiguous()

    spectra = []
    for k in range(1, d):  # bonds sit between cores, so d-1 of them
        left = int(np.prod([out_factors[i] * in_factors[i] for i in range(k)]))
        matrix = tensor.reshape(left, -1)
        singular_values = torch.linalg.svdvals(matrix)
        spectra.append(singular_values.detach().cpu().numpy())
    return spectra


def spectrum_metrics(singular_values: np.ndarray, bond_dimension: int) -> Dict[str, float]:
    weights = singular_values ** 2
    total = weights.sum()
    if total <= 0:
        return {"entropy": 0.0, "effective_rank": 1.0, "captured": 1.0, "max_rank": len(singular_values)}
    probabilities = weights / total
    nonzero = probabilities[probabilities > 1e-12]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    captured = float(weights[:bond_dimension].sum() / total)
    return {
        "entropy": entropy,
        "effective_rank": float(np.exp(entropy)),
        "captured": captured,
        "max_rank": int(len(singular_values)),
    }


def analyse_dataset(
    dataset: str,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    hidden_dim: int = 32,
    epochs: int = 80,
    lr: float = 0.01,
    num_cores: int = 3,
    tt_rank: int = 4,
) -> Dict[str, object]:
    trained_metrics: List[List[Dict[str, float]]] = []
    random_metrics: List[List[Dict[str, float]]] = []

    for seed in seeds:
        set_seed(seed)
        bundle = load_dataset(dataset, seed=seed)
        input_dim = bundle.input_dim
        num_classes = bundle.num_classes

        in_factors = factorize_balanced(input_dim, num_cores)
        out_factors = factorize_balanced(hidden_dim, num_cores)

        model = DenseMLP(input_dim, hidden_dim, num_classes)
        random_weight = model.net[0].weight.detach().clone()  # before any training
        train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
        trained_weight = model.net[0].weight.detach().clone()

        trained_metrics.append(
            [spectrum_metrics(s, tt_rank) for s in bond_spectra(trained_weight, in_factors, out_factors)]
        )
        random_metrics.append(
            [spectrum_metrics(s, tt_rank) for s in bond_spectra(random_weight, in_factors, out_factors)]
        )

    def summarise(runs: List[List[Dict[str, float]]], key: str) -> Dict[str, float]:
        # average each bond over seeds, then take the worst (most demanding) bond,
        # since a single under-resourced bond is enough to bottleneck the chain
        per_bond = np.array([[bond[key] for bond in run] for run in runs])  # (seeds, bonds)
        bond_means = per_bond.mean(axis=0)
        return {
            "per_bond": [float(v) for v in bond_means],
            "worst_bond": float(bond_means.min() if key == "captured" else bond_means.max()),
        }

    return {
        "dataset": dataset,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "in_factors": in_factors,
        "out_factors": out_factors,
        "trained_entropy": summarise(trained_metrics, "entropy"),
        "trained_captured": summarise(trained_metrics, "captured"),
        "trained_effective_rank": summarise(trained_metrics, "effective_rank"),
        "random_entropy": summarise(random_metrics, "entropy"),
        "random_captured": summarise(random_metrics, "captured"),
        "n_seeds": len(seeds),
        "tt_rank": tt_rank,
    }


def run(datasets: Sequence[str] = ("wine", "breast_cancer", "digits"), tt_rank: int = 4) -> List[Dict[str, object]]:
    results = []
    for dataset in datasets:
        print(f"analysing {dataset} ...")
        results.append(analyse_dataset(dataset, tt_rank=tt_rank))
    print()
    header = f'{"dataset":>16} {"dims":>10} {"S (trained)":>12} {"eff.rank":>9} {"captured@r":>11} {"S (random)":>11} {"capt.random":>12}'
    print(header)
    print("-" * len(header))
    for r in results:
        dims = f'{r["input_dim"]}x{r["num_classes"]}'
        print(
            f'{r["dataset"]:>16} {dims:>10} '
            f'{r["trained_entropy"]["worst_bond"]:>12.4f} '
            f'{r["trained_effective_rank"]["worst_bond"]:>9.2f} '
            f'{r["trained_captured"]["worst_bond"]:>11.4f} '
            f'{r["random_entropy"]["worst_bond"]:>11.4f} '
            f'{r["random_captured"]["worst_bond"]:>12.4f}'
        )
    print()
    print(f"captured@r = fraction of spectral weight a bond dimension of {tt_rank} can represent")
    print("(higher = Tensor-Train truncation loses less; the prediction is that this")
    print(" tracks where TT preserved accuracy: wine/breast_cancer high, digits low)")
    return results


if __name__ == "__main__":
    run()
