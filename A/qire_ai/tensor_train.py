from __future__ import annotations

"""Tensor-Train (TT) factorized linear layer.

Quantum-physics connection (see sources/literature_cards/tensor_networks.md):
Tensor-Train / Matrix-Product-State decompositions were developed to
compactly represent quantum many-body states, whose parameter count would
otherwise grow exponentially with the number of particles. Applied to a
neural network weight matrix, the same idea reduces an m x n dense matrix
(m*n parameters) to a chain of small "core" tensors -- if that chain has few
enough parameters relative to a dense matrix, the layer represents the same
map with a fraction of the parameters. Unlike QIRE's Born-rule gate, this is
not a relabeled classical operation: it's a genuinely different parameter
structure, with a real literature track record (Novikov et al. 2015 report
up to ~200,000x compression on a single large layer with no accuracy loss).

Whether that benefit survives at *this* project's small matrix sizes (input
dims of 13-64, hidden dims of ~32) is an open, testable question -- not
assumed. See tensor_train_benchmark.py for where the crossover point is.
"""

from typing import List, Sequence, Tuple

import torch
from torch import nn


def factorize_balanced(dim: int, num_factors: int) -> List[int]:
    """Splits `dim` into `num_factors` roughly-equal integer factors whose
    product is >= dim (padding up if dim isn't exactly factorizable that way).
    E.g. factorize_balanced(13, 2) -> [4, 4] (4*4=16 >= 13)."""
    if num_factors < 1:
        raise ValueError("num_factors must be >= 1")
    target = max(1, round(dim ** (1.0 / num_factors)))
    factors = [target] * num_factors
    while _product(factors) < dim:
        factors[factors.index(min(factors))] += 1
    return factors


def _product(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


class TTLinear(nn.Module):
    """Drop-in replacement for nn.Linear(in_features, out_features) whose
    weight matrix is represented as a tensor-train chain of cores instead of
    a dense (out_features x in_features) matrix.

    in_features is factored into in_factors = [n_1..n_d], out_features into
    out_factors = [m_1..m_d] (padding up if needed, extra input columns are
    zero, extra output rows are sliced off). Core k has shape
    (rank[k-1], out_factors[k], in_factors[k], rank[k]), rank[0]=rank[d]=1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_cores: int = 3,
        tt_rank: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores
        self.in_factors = factorize_balanced(in_features, num_cores)
        self.out_factors = factorize_balanced(out_features, num_cores)
        self.padded_in = _product(self.in_factors)
        self.padded_out = _product(self.out_factors)
        ranks = [1] + [tt_rank] * (num_cores - 1) + [1]
        self.ranks = ranks

        self.cores = nn.ParameterList()
        for k in range(num_cores):
            core = torch.empty(ranks[k], self.out_factors[k], self.in_factors[k], ranks[k + 1])
            nn.init.xavier_uniform_(core.reshape(ranks[k], -1))
            self.cores.append(nn.Parameter(core.reshape(ranks[k], self.out_factors[k], self.in_factors[k], ranks[k + 1])))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def reconstruct_weight(self) -> torch.Tensor:
        """Contracts the TT-cores into the full (out_features x in_features)
        dense matrix. Fine for the small matrices this project uses; a
        production TT-layer would instead contract cores directly against the
        input without ever forming the dense matrix."""
        # Running tensor starts as core[0] with the leading rank-1 axis squeezed.
        result = self.cores[0][0]  # shape (m_1, n_1, r_1)
        for k in range(1, self.num_cores):
            core_k = self.cores[k]  # shape (r_k, m_k, n_k, r_{k+1})
            # Contract result's last axis (r_k) with core_k's first axis (r_k).
            result = torch.tensordot(result, core_k, dims=([-1], [0]))
            # result shape before this step: (..., m_1*..*m_{k-1} axes, n_1*..*n_{k-1} axes, r_k)
            # after tensordot: (..., m_k, n_k, r_{k+1})
        result = result.squeeze(-1)  # drop the trailing rank-1 axis (r_d=1)
        # `result` axes are interleaved as (m_1,...,m_d, n_1,...,n_d) only for
        # num_cores<=1; for >=2 cores tensordot appends each core's (m_k, n_k)
        # pair at the end, so axes are already in (m_1,n_1,m_2,n_2,...) order --
        # separate the m's and n's, then flatten each group.
        axes = list(range(result.dim()))
        m_axes = axes[0::2]
        n_axes = axes[1::2]
        result = result.permute(*m_axes, *n_axes)
        weight = result.reshape(self.padded_out, self.padded_in)
        return weight[: self.out_features, : self.in_features]

    def num_tt_parameters(self) -> int:
        return sum(c.numel() for c in self.cores) + (self.bias.numel() if self.bias is not None else 0)

    def num_dense_equivalent_parameters(self) -> int:
        return self.in_features * self.out_features + (self.out_features if self.bias is not None else 0)

    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Contract the cores directly against the input, never forming the dense
        matrix. Mathematically identical to forward(), different cost profile.

        State carries shape (batch, *remaining_n, *accumulated_m, rank): axis 1 is
        always the next input factor and the last axis is always the current bond,
        so each step is one tensordot over exactly those two.

        Not automatically cheaper. Dense costs prod(n)*prod(m) per sample; this
        costs about d * q^(d+1) * r^2 for roughly square factorisations of mode
        size q, so the ratio to dense is ~ d * r^2 * q^(1-d). It only wins with
        many cores and small rank -- with d=3, q~8, r=16 it is about 12x *worse*.
        benchmark_tt_streaming.py measures where the crossover actually falls.
        """
        batch = x.shape[0]
        if self.padded_in > self.in_features:
            x = torch.nn.functional.pad(x, (0, self.padded_in - self.in_features))
        state = x.reshape(batch, *self.in_factors).unsqueeze(-1)  # trailing bond r_0 = 1
        for core in self.cores:
            state = torch.tensordot(state, core, dims=([1, -1], [2, 0]))
        out = state.squeeze(-1).reshape(batch, self.padded_out)[:, : self.out_features]
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.reconstruct_weight()
        out = x @ weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out
