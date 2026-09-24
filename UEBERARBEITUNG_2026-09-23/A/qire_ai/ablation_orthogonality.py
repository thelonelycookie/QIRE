from __future__ import annotations

"""RESEARCH ID: B-002 -- Is it orthogonality specifically, or just norm control?

BACKGROUND (B-001)
Constraining block-diagonal blocks to be orthogonal gave +0.0139 balanced
accuracy at an identical parameter budget (p=0.002), while the brickwork
light-cone idea gave nothing. Orthogonality is on the accepted list of quantum
principles (unitary / norm-preserving transformations), and a real orthogonal
matrix is the classical analogue of a unitary gate.

But orthogonal weight constraints are a well-established classical ML technique
(orthogonal init, unitary RNNs). Before calling this a quantum-inspired result
it has to survive the same Abgrenzung test that Tensor-Train failed: does the
*specific* property of orthogonality matter, or would any norm-controlling
regularisation do the same?

WHAT DISTINGUISHES THE CONTROLS
  plain        unconstrained blocks (baseline)
  orthogonal   ALL singular values = 1; norm- and angle-preserving, condition
               number exactly 1. The unitary analogue.
  spectral     only the LARGEST singular value = 1. Bounds scale, but the map
               is contractive, not norm-preserving.
  frobenius    total "energy" fixed to that of an orthogonal block
               (||B||_F = sqrt(d)), spectrum otherwise unconstrained.
  weight_decay unconstrained blocks, Adam weight_decay=1e-4. The standard
               classical regulariser.

If the benefit is really about unitarity, `orthogonal` should beat all three
norm-control arms. If `spectral` or `frobenius` match it, the effect is scale
control and the quantum framing does not hold. If `weight_decay` matches it,
it is ordinary regularisation.

HYPOTHESES
  H1 (main) : orthogonal > spectral, frobenius, weight_decay
  H0 (null) : one or more controls match orthogonal -> known technique, no
              quantum-specific contribution
  H2 (alt)  : all constrained arms beat plain equally -> any constraint helps,
              the specific form is irrelevant

Note: an orthogonal d x d block has d(d-1)/2 free parameters instead of d^2,
so the orthogonal arm carries FEWER effective degrees of freedom at identical
storage. Winning despite that is the signature of a real structural constraint.

Run: python -m qire_ai.ablation_orthogonality
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate


class ConstrainedBlock(nn.Module):
    """Block-diagonal layer with a selectable constraint on each block."""

    def __init__(self, dim: int, block_size: int = 4, mode: str = "plain"):
        super().__init__()
        assert dim % block_size == 0
        self.dim, self.block_size, self.mode = dim, block_size, mode
        self.blocks = dim // block_size
        self.weight = nn.Parameter(torch.empty(self.blocks, block_size, block_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(dim))

    def _blocks(self) -> torch.Tensor:
        w = self.weight
        if self.mode in ("plain", "weight_decay"):
            return w
        if self.mode == "orthogonal":
            # expm of an antisymmetric matrix is orthogonal: all singular values 1
            return torch.matrix_exp(w - w.transpose(-1, -2))
        if self.mode == "spectral":
            sigma_max = torch.linalg.matrix_norm(w, ord=2).clamp_min(1e-8)
            return w / sigma_max[:, None, None]
        if self.mode == "frobenius":
            # match the Frobenius norm an orthogonal d x d block has: sqrt(d)
            fro = torch.linalg.matrix_norm(w, ord="fro").clamp_min(1e-8)
            return w * (self.block_size ** 0.5) / fro[:, None, None]
        raise ValueError(self.mode)

    def forward(self, x):
        batch = x.shape[0]
        h = x.reshape(batch, self.blocks, self.block_size)
        h = torch.bmm(h.transpose(0, 1), self._blocks()).transpose(0, 1)
        return h.reshape(batch, self.dim) + self.bias


class StackedNet(nn.Module):
    model_name = "StackedNet"

    def __init__(self, input_dim: int, width: int, num_classes: int, mode: str, depth: int):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.middles = nn.ModuleList([ConstrainedBlock(width, 4, mode) for _ in range(depth)])
        self.fc_out = nn.Linear(width, num_classes)
        self.method = f"{mode}_d{depth}"

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        for layer in self.middles:
            h = torch.relu(layer(h))
        return self.fc_out(h), {"active_model_parts": 1.0, "gate_entropy": 0.0,
                                "router_ops": 0, "expert_ops": 0}

    def compressed_parameters(self) -> int:
        return sum(p.numel() for m in self.middles for p in m.parameters())


ARMS = ["plain", "orthogonal", "spectral", "frobenius", "weight_decay"]
WEIGHT_DECAY = 1e-4


def run(
    width: int = 4096,
    depth: int = 3,
    seeds: Sequence[int] = tuple(range(1001, 1011)),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, object]:
    acc: Dict[str, List[float]] = {}
    params: Dict[str, int] = {}
    for seed in seeds:
        for mode in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = StackedNet(bundle.input_dim, width, bundle.num_classes, mode, depth)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size,
                        weight_decay=WEIGHT_DECAY if mode == "weight_decay" else 0.0)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(mode, []).append(float(ex["balanced_accuracy"]))
            params[mode] = model.compressed_parameters()
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]    # type: ignore[assignment]
    params: Dict[str, int] = result["params"]      # type: ignore[assignment]
    n = len(acc["plain"])
    desc = {
        "plain": "unbeschraenkt (Basis)",
        "orthogonal": "alle Singulaerwerte = 1",
        "spectral": "groesster Singulaerwert = 1",
        "frobenius": "Gesamtenergie fixiert",
        "weight_decay": "Adam weight_decay 1e-4",
    }

    print()
    print(f"RESEARCH ID B-002 | MNIST, width 4096, depth 3, {n} seeds")
    header = f'{"arm":>14} {"Beschraenkung":>28} {"accuracy":>9} {"std":>7} {"vs plain":>10}'
    print(header); print("-" * len(header))
    base = np.array(acc["plain"])
    for mode in ARMS:
        vals = np.array(acc[mode])
        delta = "" if mode == "plain" else f'{vals.mean()-base.mean():>+10.4f}'
        print(f'{mode:>14} {desc[mode]:>28} {vals.mean():>9.4f} {vals.std(ddof=1):>7.4f} {delta:>10}')

    print()
    print("Jede Beschraenkung gegen unbeschraenkt (gepaarter Wilcoxon):")
    for mode in ARMS[1:]:
        vals = np.array(acc[mode])
        p = stats.wilcoxon(vals, base).pvalue
        flag = " *signifikant*" if p < 0.05 else " (nicht signifikant)"
        print(f'  {mode:>14}: {vals.mean()-base.mean():+.4f}, p={p:.4f}{flag}')

    print()
    print("ENTSCHEIDEND -- orthogonal gegen jede Norm-Kontrolle:")
    ortho = np.array(acc["orthogonal"])
    beaten_all = True
    for mode in ["spectral", "frobenius", "weight_decay"]:
        vals = np.array(acc[mode])
        p = stats.wilcoxon(ortho, vals).pvalue
        wins = ortho.mean() > vals.mean()
        if not (wins and p < 0.05):
            beaten_all = False
        flag = " *signifikant*" if p < 0.05 else " (nicht signifikant)"
        print(f'  orthogonal vs {mode:>12}: {ortho.mean()-vals.mean():+.4f}, p={p:.4f}{flag}')

    print()
    if beaten_all:
        print("  -> orthogonal schlaegt ALLE Norm-Kontrollen signifikant.")
        print("     Der Effekt haengt an der Unitaritaets-Eigenschaft, nicht an blosser")
        print("     Skalenkontrolle. Quanteninspirierter Bezug haelt dem Test stand.")
    else:
        print("  -> mindestens eine Norm-Kontrolle zieht gleich.")
        print("     Der Effekt ist dann Skalen-/Regularisierungskontrolle, nicht spezifisch")
        print("     unitaer -- der quanteninspirierte Anspruch waere nicht haltbar.")


if __name__ == "__main__":
    analyse(run())
