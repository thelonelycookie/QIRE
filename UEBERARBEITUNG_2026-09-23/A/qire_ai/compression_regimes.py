from __future__ import annotations

"""Does the entanglement/truncation criterion predict accuracy loss in one
compression regime but not the other?

Background (ROADMAP.md, 2026-08-02): the Schmidt-truncation criterion from
MPS theory said a bond dimension of 8-13 was needed, yet training a TT layer
from scratch at rank 4 preserved accuracy on 2 of 3 datasets. The proposed
explanation is that the theory describes *approximating a given matrix*,
while training-in-compressed-form lets the network find a different solution
inside the constrained family, so it never has to reproduce the dense matrix
at all.

That gives a falsifiable prediction, tested here:

  Regime A -- post-hoc: train dense, then replace its weight with a rank-r
              TT-SVD approximation, no retraining. The criterion SHOULD
              predict the accuracy drop here, since this is exactly the
              "approximate a fixed target" setting the theory covers.

  Regime B -- compressed training: train a TTLinear of rank r from scratch.
              The criterion should predict the drop much more weakly, if at
              all.

Reported as the correlation between (1 - captured@r), i.e. the spectral
weight truncation throws away, and the measured accuracy drop, per regime.

Run: python -m qire_ai.compression_regimes
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .entanglement_analysis import bond_spectra, spectrum_metrics
from .metrics import evaluate
from .models_baseline import DenseMLP, TensorTrainMLP
from .tensor_train import factorize_balanced


def _to_interleaved(weight: torch.Tensor, in_factors: Sequence[int], out_factors: Sequence[int]) -> torch.Tensor:
    """(out, in) matrix -> tensor with one mode per core, each of size m_k*n_k."""
    d = len(in_factors)
    padded_out, padded_in = int(np.prod(out_factors)), int(np.prod(in_factors))
    w = torch.zeros(padded_out, padded_in, dtype=weight.dtype)
    w[: weight.shape[0], : weight.shape[1]] = weight
    tensor = w.reshape(*out_factors, *in_factors)
    perm: List[int] = []
    for k in range(d):
        perm.extend([k, d + k])
    tensor = tensor.permute(*perm).contiguous()
    return tensor.reshape(*[out_factors[k] * in_factors[k] for k in range(d)])


def _from_interleaved(tensor: torch.Tensor, in_factors, out_factors, out_features, in_features) -> torch.Tensor:
    d = len(in_factors)
    interleaved: List[int] = []
    for k in range(d):
        interleaved.extend([out_factors[k], in_factors[k]])
    tensor = tensor.reshape(*interleaved)
    inverse_perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
    tensor = tensor.permute(*inverse_perm).contiguous()
    w = tensor.reshape(int(np.prod(out_factors)), int(np.prod(in_factors)))
    return w[:out_features, :in_features]


def tt_svd_compress(weight: torch.Tensor, in_factors, out_factors, rank: int) -> torch.Tensor:
    """Standard TT-SVD: sequential SVDs, each truncated to `rank`, then
    contracted back into a dense matrix. Returns the rank-r approximation of
    `weight`. At full rank this reconstructs the input exactly."""
    tensor = _to_interleaved(weight, in_factors, out_factors)
    mode_sizes = list(tensor.shape)
    d = len(mode_sizes)

    cores = []
    remainder = tensor.reshape(mode_sizes[0], -1)
    left_rank = 1
    for k in range(d - 1):
        matrix = remainder.reshape(left_rank * mode_sizes[k], -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        keep = min(rank, s.numel())
        u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
        cores.append(u.reshape(left_rank, mode_sizes[k], keep))
        remainder = torch.diag(s) @ vh
        left_rank = keep
    cores.append(remainder.reshape(left_rank, mode_sizes[-1], 1))

    result = cores[0]  # (1, m_0, r)
    for k in range(1, d):
        result = torch.tensordot(result, cores[k], dims=([-1], [0]))
    result = result.squeeze(0).squeeze(-1)
    return _from_interleaved(result, in_factors, out_factors, weight.shape[0], weight.shape[1])


def _balanced_accuracy(model, x_test, y_test) -> float:
    _, _, _, extras = evaluate(model, x_test, y_test)
    return float(extras.get("balanced_accuracy", 0.0))


def run(
    datasets: Sequence[str] = ("wine", "breast_cancer", "digits"),
    ranks: Sequence[int] = (2, 4, 8, 12),
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    hidden_dim: int = 32,
    epochs: int = 80,
    lr: float = 0.01,
    num_cores: int = 3,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        in_factors = out_factors = None
        dense_acc: List[float] = []
        posthoc: Dict[int, List[float]] = {r: [] for r in ranks}
        captured: Dict[int, List[float]] = {r: [] for r in ranks}
        scratch: Dict[int, List[float]] = {r: [] for r in ranks}

        for seed in seeds:
            set_seed(seed)
            bundle = load_dataset(dataset, seed=seed)
            in_factors = factorize_balanced(bundle.input_dim, num_cores)
            out_factors = factorize_balanced(hidden_dim, num_cores)

            # ---- Regime A: train dense, then approximate its weight ----
            dense = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
            train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
            dense_acc.append(_balanced_accuracy(dense, bundle.x_test, bundle.y_test))
            trained_weight = dense.net[0].weight.detach().clone()
            spectra = bond_spectra(trained_weight, in_factors, out_factors)

            for r in ranks:
                approx = tt_svd_compress(trained_weight, in_factors, out_factors, r)
                with torch.no_grad():
                    dense.net[0].weight.copy_(approx)
                posthoc[r].append(_balanced_accuracy(dense, bundle.x_test, bundle.y_test))
                with torch.no_grad():
                    dense.net[0].weight.copy_(trained_weight)  # restore for next rank
                captured[r].append(min(spectrum_metrics(s, r)["captured"] for s in spectra))

            # ---- Regime B: train the constrained model from scratch ----
            for r in ranks:
                set_seed(seed)
                tt_model = TensorTrainMLP(bundle.input_dim, hidden_dim, bundle.num_classes, tt_rank=r, num_cores=num_cores)
                train_model(tt_model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
                scratch[r].append(_balanced_accuracy(tt_model, bundle.x_test, bundle.y_test))

        baseline = float(np.mean(dense_acc))
        for r in ranks:
            rows.append({
                "dataset": dataset,
                "rank": r,
                "dense_accuracy": baseline,
                "posthoc_accuracy": float(np.mean(posthoc[r])),
                "scratch_accuracy": float(np.mean(scratch[r])),
                "posthoc_drop": baseline - float(np.mean(posthoc[r])),
                "scratch_drop": baseline - float(np.mean(scratch[r])),
                "captured": float(np.mean(captured[r])),
                "discarded": 1.0 - float(np.mean(captured[r])),
            })
        print(f"done: {dataset}")

    return rows


def analyse(rows: List[Dict[str, object]]) -> None:
    from scipy import stats

    print()
    header = (
        f'{"dataset":>15} {"rank":>5} {"discarded":>10} {"dense":>8} '
        f'{"post-hoc":>9} {"drop":>8} {"scratch":>9} {"drop":>8}'
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f'{r["dataset"]:>15} {r["rank"]:>5} {r["discarded"]:>10.4f} {r["dense_accuracy"]:>8.4f} '
            f'{r["posthoc_accuracy"]:>9.4f} {r["posthoc_drop"]:>8.4f} '
            f'{r["scratch_accuracy"]:>9.4f} {r["scratch_drop"]:>8.4f}'
        )

    discarded = np.array([r["discarded"] for r in rows])
    posthoc_drop = np.array([r["posthoc_drop"] for r in rows])
    scratch_drop = np.array([r["scratch_drop"] for r in rows])

    print()
    print("Does discarded spectral weight predict the accuracy drop?")
    for label, drops in (("post-hoc (Regime A)", posthoc_drop), ("from scratch (Regime B)", scratch_drop)):
        r_p, p_p = stats.pearsonr(discarded, drops)
        r_s, p_s = stats.spearmanr(discarded, drops)
        print(f"  {label:>24}: Pearson r={r_p:+.3f} (p={p_p:.4f}) | Spearman rho={r_s:+.3f} (p={p_s:.4f})")
    print()
    print(f"  mean accuracy drop, post-hoc:    {posthoc_drop.mean():.4f}")
    print(f"  mean accuracy drop, from scratch: {scratch_drop.mean():.4f}")
    print()
    print("Prediction under test: the criterion should track the post-hoc drop")
    print("(approximating a fixed matrix) far better than the from-scratch drop")
    print("(the network can find a different solution inside the constraint).")


if __name__ == "__main__":
    analyse(run())
