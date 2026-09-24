from __future__ import annotations

"""Standalone microbenchmark: does sparse expert routing ever measure faster
than a dense baseline in real wall-clock time, and at what scale?

Context: on the project's actual datasets (wine, breast_cancer -- only 30-115
test rows), estimated_ops always claims a saving for QIRE/StandardMoE, but
measured inference time is always far worse (see ROADMAP.md, "vektorisierter
Fix"). This script isolates the question from any specific dataset: it varies
only batch size and model size, with torch.set_num_threads(1) matching
qire_ai/benchmark.py:set_seed() exactly, so results are on the same footing as
every other timing number this project reports (a single dense baseline
matmul otherwise benefits unfairly from multi-core BLAS that the per-expert
matmuls in a sparse layer do not get the same benefit from).

Run: python -m qire_ai.benchmark_sparse_dispatch
"""

import time

import torch
from torch import nn

from .models_baseline import DenseMLP
from .sparse_experts import SparseExpertLayer


class _NaiveSparseExpertLayer(nn.Module):
    """The original (pre-fix) per-expert nonzero()/gather/index_add_ dispatch.
    Kept here only so this benchmark can show the before/after; it is not used
    anywhere else. See git history / ROADMAP.md for why it was replaced."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_experts):
        super().__init__()
        self.num_classes = num_classes
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)) for _ in range(num_experts)]
        )

    def forward(self, x, top_idx, top_vals):
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
        for expert_id, expert in enumerate(self.experts):
            positions = (top_idx == expert_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            rows = positions[:, 0]
            slots = positions[:, 1]
            expert_out = expert(x[rows])
            weights = top_vals[rows, slots].unsqueeze(-1)
            out.index_add_(0, rows, expert_out * weights)
        return out


def _make_routing(batch: int, num_experts: int, top_k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    probs = torch.rand(batch, num_experts, generator=g)
    probs = probs / probs.sum(-1, keepdim=True)
    top_vals, top_idx = torch.topk(probs, k=top_k, dim=-1)
    top_vals = top_vals / top_vals.sum(-1, keepdim=True)
    return top_idx, top_vals


def _bench(module: nn.Module, x: torch.Tensor, routing, reps: int = 50) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(5):
            module(x, *routing) if routing else module(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            module(x, *routing) if routing else module(x)
        t1 = time.perf_counter()
    return (t1 - t0) / reps * 1000.0


def run(configs=None, batch_sizes=None) -> list[dict]:
    """Returns a list of result rows; also prints a table. Forces single-thread
    torch to match qire_ai.benchmark.set_seed(), the setting every tracked
    experiment in this project actually runs under."""
    torch.set_num_threads(1)
    if configs is None:
        configs = [
            ("small: hidden=32, 4 experts, top_k=1", 13, 32, 3, 4, 1, 12),
            ("large: hidden=256, 16 experts, top_k=1 (6.25% active)", 128, 256, 10, 16, 1, 64),
        ]
    if batch_sizes is None:
        batch_sizes = [30, 1000, 10000, 100000]
    results = []
    for label, input_dim, mlp_hidden, num_classes, num_experts, top_k, expert_hidden in configs:
        print(f"\n-- {label} --")
        torch.manual_seed(0)
        mlp = DenseMLP(input_dim, mlp_hidden, num_classes)
        torch.manual_seed(0)
        naive = _NaiveSparseExpertLayer(input_dim, expert_hidden, num_classes, num_experts)
        torch.manual_seed(0)
        vec = SparseExpertLayer(input_dim, expert_hidden, num_classes, num_experts)
        vec.load_state_dict(naive.state_dict())
        print(f'{"batch":>8} {"MLP ms":>10} {"naive ms":>10} {"vec ms":>10} {"naive/MLP":>10} {"vec/MLP":>10}')
        for batch in batch_sizes:
            x = torch.randn(batch, input_dim)
            routing = _make_routing(batch, num_experts, top_k, seed=1)
            t_mlp = _bench(mlp, x, None)
            t_naive = _bench(naive, x, routing)
            t_vec = _bench(vec, x, routing)
            row = {
                "config": label, "batch": batch, "mlp_ms": t_mlp, "naive_ms": t_naive, "vec_ms": t_vec,
                "naive_over_mlp": t_naive / t_mlp, "vec_over_mlp": t_vec / t_mlp,
            }
            results.append(row)
            print(f"{batch:>8} {t_mlp:>10.4f} {t_naive:>10.4f} {t_vec:>10.4f} {row['naive_over_mlp']:>9.2f}x {row['vec_over_mlp']:>9.2f}x")
    return results


def plot(results: list[dict], output_path: str = "results/figures/sparse_dispatch_crossover.png") -> None:
    import matplotlib.pyplot as plt
    from pathlib import Path

    configs = sorted(set(r["config"] for r in results))
    fig, axes = plt.subplots(1, len(configs), figsize=(6.5 * len(configs), 5), squeeze=False)
    for ax, config in zip(axes[0], configs):
        rows = [r for r in results if r["config"] == config]
        rows.sort(key=lambda r: r["batch"])
        batches = [r["batch"] for r in rows]
        ax.plot(batches, [r["naive_over_mlp"] for r in rows], marker="o", label="naive loop / MLP")
        ax.plot(batches, [r["vec_over_mlp"] for r in rows], marker="s", label="vectorized / MLP")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity with dense MLP")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("sparse layer time / dense MLP time")
        ax.set_title(config, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Sparse expert routing: measured wall-clock ratio vs dense baseline (single-threaded)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved crossover plot to {output_path}")


if __name__ == "__main__":
    results = run()
    plot(results)
