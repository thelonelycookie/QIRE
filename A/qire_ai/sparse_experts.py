from __future__ import annotations

import torch
from torch import nn


class SparseExpertLayer(nn.Module):
    """Computes only the experts selected by top-k routing for each sample.

    Dispatch is grouped-GEMM style: tokens are sorted by assigned expert so each
    expert runs one contiguous batched forward pass, instead of a per-expert
    boolean gather/scatter. A prior per-expert `.nonzero()` + advanced-indexing
    loop carried enough Python/tensor overhead that it was never faster than a
    dense baseline in measured wall-clock time, even at large batch sizes and
    high sparsity, despite doing fewer real FLOPs. This version is numerically
    identical (verified against that implementation) but amortizes better: it
    measurably beats a dense baseline at large model/batch scale, though small
    models still don't have enough per-expert compute to outrun the fixed
    sort/gather/scatter cost.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor, top_idx: torch.Tensor, top_vals: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        k = top_idx.shape[1]
        flat_expert_ids = top_idx.reshape(-1)
        flat_weights = top_vals.reshape(-1)
        flat_row_ids = torch.arange(batch_size, device=x.device).repeat_interleave(k)
        sort_order = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_row_ids = flat_row_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        x_gathered = x.index_select(0, sorted_row_ids)
        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts).tolist()
        chunks = []
        offset = 0
        for expert_id, expert in enumerate(self.experts):
            count = counts[expert_id]
            if count == 0:
                continue
            chunks.append(expert(x_gathered[offset:offset + count]))
            offset += count
        outputs_sorted = torch.cat(chunks, dim=0) if chunks else x_gathered.new_zeros(0, self.num_classes)
        weighted = outputs_sorted * sorted_weights.unsqueeze(-1)
        out = x.new_zeros(batch_size, self.num_classes)
        out.index_add_(0, sorted_row_ids, weighted)
        return out
