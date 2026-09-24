from __future__ import annotations

"""Is the post-hoc vs. trained-in regime gap specific to Tensor-Train, or a
general property of imposing structure on a network?

compression_regimes.py found that for Tensor-Train, imposing a rank-r
constraint after training costs ~12x more accuracy than training under the
same constraint from the start. If that is a general principle rather than a
quirk of TT, the same gap should appear for other structural constraints.

Three constraint families are compared, all applied to the same first layer
of the same 2-layer MLP, so the only thing that varies is the *kind* of
structure imposed:

  tensor_train  weight is a chain of TT cores (quantum-inspired, MPS-style)
  low_rank      weight factorizes as U @ V through an r-dimensional bottleneck
  width         hidden layer simply has fewer neurons

Each is measured in both regimes:
  post-hoc  train unconstrained, then project onto the constraint, no retraining
  scratch   train under the constraint from initialization

Note on the post-hoc width case: removing neurons needs a choice of *which*.
Neurons are scored by ||incoming|| * ||outgoing||, the standard magnitude
heuristic -- a better criterion would narrow the gap, so this is a
conservative test of the hypothesis, not an inflated one.

Run: python -m qire_ai.regime_generalization
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .compression_regimes import tt_svd_compress
from .data_loader import load_dataset
from .metrics import evaluate
from .models_baseline import DenseMLP, TensorTrainMLP
from .tensor_train import factorize_balanced


class LowRankMLP2(nn.Module):
    """Same shape as DenseMLP, but the first weight matrix is forced through an
    r-dimensional bottleneck (W = U @ V). Deliberately not the existing
    LowRankMLP, which has an extra layer and would confound the comparison."""

    model_name = "LowRankMLP2"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, rank: int):
        super().__init__()
        self.input_dim, self.hidden_dim, self.num_classes, self.rank = input_dim, hidden_dim, num_classes, rank
        self.v = nn.Linear(input_dim, rank, bias=False)
        self.u = nn.Linear(rank, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.method = f"lowrank2_r{rank}_h{hidden_dim}"

    def forward(self, x):
        h = torch.relu(self.u(self.v(x)))
        logits = self.fc2(h)
        return logits, {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0,
                        "expert_ops": self.estimated_ops_per_sample()}

    def estimated_ops_per_sample(self) -> int:
        return int(2 * (self.input_dim * self.rank + self.rank * self.hidden_dim + self.hidden_dim * self.num_classes))


def _balanced_accuracy(model, x_test, y_test) -> float:
    _, _, _, extras = evaluate(model, x_test, y_test)
    return float(extras.get("balanced_accuracy", 0.0))


def _first_layer_params(kind: str, strength: int, input_dim: int, hidden_dim: int, num_cores: int = 3) -> int:
    """Intrinsic parameter count of the constrained first layer -- identical in
    both regimes at a given strength, which is what makes them comparable."""
    if kind == "width":
        return input_dim * strength + strength
    if kind == "low_rank":
        return hidden_dim * strength + strength * input_dim + hidden_dim
    if kind == "tensor_train":
        in_f, out_f = factorize_balanced(input_dim, num_cores), factorize_balanced(hidden_dim, num_cores)
        ranks = [1] + [strength] * (num_cores - 1) + [1]
        return sum(ranks[k] * out_f[k] * in_f[k] * ranks[k + 1] for k in range(num_cores)) + hidden_dim
    raise ValueError(kind)


def _posthoc_accuracy(kind, strength, dense, bundle, hidden_dim, num_cores=3) -> float:
    """Project the trained dense model onto the constraint, then evaluate.
    No retraining, no fine-tuning."""
    w = dense.net[0].weight.detach().clone()
    b = dense.net[0].bias.detach().clone()
    w2 = dense.net[2].weight.detach().clone()
    b2 = dense.net[2].bias.detach().clone()

    if kind == "tensor_train":
        in_f, out_f = factorize_balanced(bundle.input_dim, num_cores), factorize_balanced(hidden_dim, num_cores)
        approx = tt_svd_compress(w, in_f, out_f, strength)
        model = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
        with torch.no_grad():
            model.net[0].weight.copy_(approx); model.net[0].bias.copy_(b)
            model.net[2].weight.copy_(w2); model.net[2].bias.copy_(b2)
    elif kind == "low_rank":
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        k = min(strength, s.numel())
        approx = u[:, :k] @ torch.diag(s[:k]) @ vh[:k, :]
        model = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
        with torch.no_grad():
            model.net[0].weight.copy_(approx); model.net[0].bias.copy_(b)
            model.net[2].weight.copy_(w2); model.net[2].bias.copy_(b2)
    elif kind == "width":
        # keep the `strength` most important hidden neurons
        importance = w.norm(dim=1) * w2.norm(dim=0)
        keep = torch.argsort(importance, descending=True)[:strength]
        model = DenseMLP(bundle.input_dim, strength, bundle.num_classes)
        with torch.no_grad():
            model.net[0].weight.copy_(w[keep]); model.net[0].bias.copy_(b[keep])
            model.net[2].weight.copy_(w2[:, keep]); model.net[2].bias.copy_(b2)
    else:
        raise ValueError(kind)
    return _balanced_accuracy(model, bundle.x_test, bundle.y_test)


def _scratch_model(kind, strength, bundle, hidden_dim, num_cores=3):
    if kind == "tensor_train":
        return TensorTrainMLP(bundle.input_dim, hidden_dim, bundle.num_classes, tt_rank=strength, num_cores=num_cores)
    if kind == "low_rank":
        return LowRankMLP2(bundle.input_dim, hidden_dim, bundle.num_classes, rank=strength)
    if kind == "width":
        return DenseMLP(bundle.input_dim, strength, bundle.num_classes)
    raise ValueError(kind)


CONSTRAINTS = {
    "tensor_train": (2, 4, 8),
    "low_rank": (2, 4, 8),
    "width": (8, 16, 24),
}


def run(
    datasets: Sequence[str] = ("wine", "breast_cancer", "digits"),
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    hidden_dim: int = 32,
    epochs: int = 80,
    lr: float = 0.01,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        acc: Dict[tuple, List[float]] = {}
        dense_acc: List[float] = []
        for seed in seeds:
            set_seed(seed)
            bundle = load_dataset(dataset, seed=seed)
            dense = DenseMLP(bundle.input_dim, hidden_dim, bundle.num_classes)
            train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
            dense_acc.append(_balanced_accuracy(dense, bundle.x_test, bundle.y_test))

            for kind, strengths in CONSTRAINTS.items():
                for strength in strengths:
                    acc.setdefault((kind, strength, "posthoc"), []).append(
                        _posthoc_accuracy(kind, strength, dense, bundle, hidden_dim)
                    )
                    set_seed(seed)
                    model = _scratch_model(kind, strength, bundle, hidden_dim)
                    train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr, seed=seed)
                    acc.setdefault((kind, strength, "scratch"), []).append(
                        _balanced_accuracy(model, bundle.x_test, bundle.y_test)
                    )

            del bundle
        baseline = float(np.mean(dense_acc))
        dense_params = None
        for kind, strengths in CONSTRAINTS.items():
            for strength in strengths:
                b = load_dataset(dataset, seed=seeds[0])
                dense_params = b.input_dim * hidden_dim + hidden_dim
                rows.append({
                    "dataset": dataset, "constraint": kind, "strength": strength,
                    "param_fraction": _first_layer_params(kind, strength, b.input_dim, hidden_dim) / dense_params,
                    "dense": baseline,
                    "posthoc_drop": baseline - float(np.mean(acc[(kind, strength, "posthoc")])),
                    "scratch_drop": baseline - float(np.mean(acc[(kind, strength, "scratch")])),
                })
        print(f"done: {dataset}")
    return rows


def analyse(rows: List[Dict[str, object]]) -> None:
    from scipy import stats

    print()
    header = f'{"dataset":>14} {"constraint":>13} {"str":>4} {"params":>7} {"post-hoc":>9} {"scratch":>9} {"gap":>8}'
    print(header); print("-" * len(header))
    for r in rows:
        print(f'{r["dataset"]:>14} {r["constraint"]:>13} {r["strength"]:>4} {r["param_fraction"]:>7.3f} '
              f'{r["posthoc_drop"]:>9.4f} {r["scratch_drop"]:>9.4f} {r["posthoc_drop"]-r["scratch_drop"]:>8.4f}')

    print()
    print("Per constraint family: is the post-hoc drop larger than the trained-in drop?")
    for kind in CONSTRAINTS:
        sub = [r for r in rows if r["constraint"] == kind]
        post = np.array([r["posthoc_drop"] for r in sub])
        scr = np.array([r["scratch_drop"] for r in sub])
        wins = int((scr < post).sum())
        w = stats.wilcoxon(post, scr)
        ratio = post.mean() / scr.mean() if abs(scr.mean()) > 1e-9 else float("inf")
        print(f'  {kind:>13}: post-hoc {post.mean():.4f} vs trained-in {scr.mean():.4f} '
              f'| ratio {ratio:>6.1f}x | trained-in better in {wins}/{len(sub)} | Wilcoxon p={w.pvalue:.4f}')

    post_all = np.array([r["posthoc_drop"] for r in rows])
    scr_all = np.array([r["scratch_drop"] for r in rows])
    w = stats.wilcoxon(post_all, scr_all)
    print()
    print(f'  overall ({len(rows)} cells): trained-in better in {int((scr_all < post_all).sum())}/{len(rows)}, '
          f'Wilcoxon p={w.pvalue:.6f}, mean ratio {post_all.mean()/scr_all.mean():.1f}x')


if __name__ == "__main__":
    analyse(run())
