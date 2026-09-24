from __future__ import annotations

"""RESEARCH ID: B-001 -- Local quantum circuit architecture: brickwork blocks.

MOTIVATION FROM OUR OWN DATA
A-002/A-003 showed locality helps and bond chains do not. I first read that as
the quantum-inspired part failing. That was too quick: locality of interaction
is itself the central structural principle of quantum many-body physics -- the
area law, and with it the whole reason tensor networks work, follows *from*
interactions being local.

What plain block-diagonal lacks is the structure that local quantum circuits
actually have. In a real local quantum circuit the blocks are OFFSET between
layers (the "brickwork" pattern, also the structure of MERA). Correlations then
spread in a light cone with depth (Lieb-Robinson), instead of staying trapped
inside fixed blocks forever.

QUANTUM PRINCIPLE (per the definition in use)
  - locality of interaction: each gate acts on a few neighbouring sites
  - brickwork/light-cone propagation: staggered layers spread correlation at a
    bounded rate with circuit depth
  - unitarity: quantum gates are norm-preserving; the classical analogue is an
    orthogonal block

CLASSICAL TRANSLATION
  block_diag  : W = blockdiag(B_1..B_k), fixed partition every layer
  brickwork   : same, but every other layer shifts the partition by half a
                block (implemented as a roll), so block boundaries interleave
  brickwork_o : brickwork with each block constrained orthogonal via
                B = expm(A - A^T), the classical analogue of a unitary gate

Note on effective capacity: an orthogonal d x d block has only d(d-1)/2 free
parameters, not d^2. The orthogonal arm therefore has FEWER effective degrees
of freedom at identical storage -- a harder test, not an easier one.

WHY IT COULD BE MORE EFFICIENT
Fixed blocks of size 4 can never build a feature spanning more than 4 of 4096
units, no matter the depth. Brickwork at depth L reaches ~L * blocksize/2
units. If that extra reach matters, brickwork should beat block_diag at equal
parameter count -- with no extra parameters, only a different arrangement.

WHEN IT SHOULD *NOT* HELP
If the surrounding dense layers already supply all needed global mixing (which
A-003 suggests is largely the case), the extra reach buys nothing, and
brickwork will match block_diag.

HYPOTHESES
  H1 (main) : brickwork > block_diag at depth 3, equal budget
  H0 (null) : no difference -- the offset arrangement is irrelevant here
  H2 (alt)  : orthogonality alone drives any difference, independent of offset
              (separated by the brickwork-without-orthogonality arm)

FALSIFICATION
If brickwork does not beat fixed blocks at equal budget and depth, the
light-cone argument does not apply at this scale and the approach is dropped.

Run: python -m qire_ai.brickwork
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate


class BrickworkBlock(nn.Module):
    """Block-diagonal layer, optionally offset by half a block and/or with
    orthogonal blocks. `offset=True` implements the staggered brickwork layer."""

    def __init__(self, dim: int, block_size: int = 4, offset: bool = False, orthogonal: bool = False):
        super().__init__()
        assert dim % block_size == 0
        self.dim, self.block_size = dim, block_size
        self.blocks = dim // block_size
        self.shift = block_size // 2 if offset else 0
        self.orthogonal = orthogonal
        self.weight = nn.Parameter(torch.empty(self.blocks, block_size, block_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(dim))

    def _blocks(self) -> torch.Tensor:
        if not self.orthogonal:
            return self.weight
        # B = expm(A - A^T) is orthogonal for any A: the classical analogue of a
        # unitary gate. Batched over blocks.
        skew = self.weight - self.weight.transpose(-1, -2)
        return torch.matrix_exp(skew)

    def forward(self, x):
        batch = x.shape[0]
        if self.shift:
            x = torch.roll(x, shifts=self.shift, dims=1)
        h = x.reshape(batch, self.blocks, self.block_size)
        h = torch.bmm(h.transpose(0, 1), self._blocks()).transpose(0, 1)
        h = h.reshape(batch, self.dim)
        if self.shift:
            h = torch.roll(h, shifts=-self.shift, dims=1)
        return h + self.bias


class StackedNet(nn.Module):
    """784 -> width -> [compressed] x depth -> classes. For brickwork arms the
    offset alternates between layers, which is what creates the light cone."""

    model_name = "StackedNet"

    def __init__(self, input_dim: int, width: int, num_classes: int, label: str, depth: int,
                 block_size: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        layers = []
        for i in range(depth):
            if label == "block_diag":
                layers.append(BrickworkBlock(width, block_size, offset=False, orthogonal=False))
            elif label == "brickwork":
                layers.append(BrickworkBlock(width, block_size, offset=(i % 2 == 1), orthogonal=False))
            elif label == "brickwork_ortho":
                layers.append(BrickworkBlock(width, block_size, offset=(i % 2 == 1), orthogonal=True))
            elif label == "block_diag_ortho":
                layers.append(BrickworkBlock(width, block_size, offset=False, orthogonal=True))
            elif label == "dense":
                layers.append(nn.Linear(width, width))
            else:
                raise ValueError(label)
        self.middles = nn.ModuleList(layers)
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


ARMS = ["block_diag", "brickwork", "block_diag_ortho", "brickwork_ortho", "dense"]


def run(
    width: int = 4096,
    depth: int = 3,
    seeds: Sequence[int] = tuple(range(901, 911)),
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
            model = StackedNet(bundle.input_dim, width, bundle.num_classes, label, depth)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            acc.setdefault(label, []).append(float(ex["balanced_accuracy"]))
            params[label] = model.compressed_parameters()
        print(f"  seed {seed} done")
    return {"acc": acc, "params": params, "depth": depth}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]    # type: ignore[assignment]
    params: Dict[str, int] = result["params"]      # type: ignore[assignment]
    n = len(acc["block_diag"])

    labels = {
        "block_diag": "feste Bloecke",
        "brickwork": "versetzt (Lichtkegel)",
        "block_diag_ortho": "feste Bloecke, orthogonal",
        "brickwork_ortho": "versetzt + orthogonal",
        "dense": "dicht (Referenz)",
    }
    print()
    print(f"RESEARCH ID B-001 | MNIST, width 4096, depth {result['depth']}, {n} seeds")
    header = f'{"arm":>18} {"Struktur":>28} {"accuracy":>9} {"std":>7} {"params":>12}'
    print(header); print("-" * len(header))
    for label in ARMS:
        vals = np.array(acc[label])
        print(f'{label:>18} {labels[label]:>28} {vals.mean():>9.4f} '
              f'{vals.std(ddof=1):>7.4f} {params[label]:>12,}')

    base = np.array(acc["block_diag"])
    print()
    print("Gegen feste Bloecke (gepaarter Wilcoxon, identisches Budget):")
    for label in ["brickwork", "block_diag_ortho", "brickwork_ortho"]:
        vals = np.array(acc[label])
        p = stats.wilcoxon(vals, base).pvalue
        flag = " *signifikant*" if p < 0.05 else " (nicht signifikant)"
        print(f'  {label:>18}: {vals.mean()-base.mean():+.4f}, p={p:.4f}{flag}')

    print()
    print("Trennung der beiden Effekte:")
    bw, bwo = np.array(acc["brickwork"]), np.array(acc["brickwork_ortho"])
    bdo = np.array(acc["block_diag_ortho"])
    print(f'  Versatz allein   (brickwork vs block_diag)      : {bw.mean()-base.mean():+.4f}')
    print(f'  Orthogonal allein(block_diag_ortho vs block_diag): {bdo.mean()-base.mean():+.4f}')
    print(f'  beides zusammen  (brickwork_ortho vs block_diag) : {bwo.mean()-base.mean():+.4f}')
    print()
    print("Falsifikation: Schlaegt Versatz die festen Bloecke bei gleichem Budget")
    print("nicht, gilt das Lichtkegel-Argument auf dieser Skala nicht.")


if __name__ == "__main__":
    analyse(run())
