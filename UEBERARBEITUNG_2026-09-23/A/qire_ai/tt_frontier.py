from __future__ import annotations

"""Map the streaming-TT speed frontier at the size where it actually wins.

Two gaps in what has been measured so far:

  1. The (cores, rank) sweep in benchmark_tt_streaming.py ran at 784x512 --
     a shape where *no* configuration beat dense. Concluding that (4 cores,
     rank 4) is the right choice from a sweep where everything lost is not
     sound. This re-runs the sweep at 4096x4096, where streaming does win, so
     the optimum is chosen inside the winning regime rather than outside it.

  2. Every timing so far used batch 256. That matters more than it looks:
     reconstruction builds the full matrix once per call and then amortises it
     over the batch, while streaming's cost scales with the batch. At batch 1
     -- real-time inference on constrained hardware, which is the actual
     resource-efficiency use case -- reconstruction has to build a 16.7M-entry
     matrix to process a single sample, so the ranking may invert entirely.

Pure timing, no training, single-threaded to stay comparable with every other
timing figure in the project.

Run: python -m qire_ai.tt_frontier
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .tensor_train import TTLinear


def _time(fn, x, repeats: int = 10) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(x)
    return (time.perf_counter() - t0) / repeats * 1000


def frontier(
    dim: int = 4096,
    core_counts: Sequence[int] = (2, 3, 4, 5, 6),
    ranks: Sequence[int] = (2, 4, 8, 16),
    batch: int = 256,
) -> List[Dict[str, object]]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dense = nn.Linear(dim, dim).eval()
    x = torch.randn(batch, dim)
    t_dense = _time(dense, x)
    print(f"=== (cores, rank) frontier at {dim}x{dim}, batch {batch} ===")
    print(f"dense reference: {t_dense:.2f} ms")
    print(f'{"cores":>6} {"rank":>5} {"params":>10} {"frac":>9} {"stream ms":>10} {"vs dense":>10}')
    print("-" * 54)
    rows = []
    for cores in core_counts:
        for rank in ranks:
            layer = TTLinear(dim, dim, num_cores=cores, tt_rank=rank).eval()
            t_stream = _time(layer.forward_streaming, x)
            ratio = t_stream / t_dense
            frac = layer.num_tt_parameters() / (dim * dim + dim)
            rows.append({"cores": cores, "rank": rank, "params": layer.num_tt_parameters(),
                         "fraction": frac, "stream_ms": t_stream, "ratio": ratio, "batch": batch})
            mark = f"  <-- {1/ratio:.2f}x faster" if ratio < 1 else ""
            print(f'{cores:>6} {rank:>5} {layer.num_tt_parameters():>10,} {frac:>9.5f} '
                  f'{t_stream:>10.2f} {ratio:>9.2f}x{mark}')
    return rows


def batch_sweep(
    dim: int = 4096,
    cores: int = 4,
    rank: int = 4,
    batches: Sequence[int] = (1, 8, 64, 256, 1024),
) -> List[Dict[str, object]]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dense = nn.Linear(dim, dim).eval()
    layer = TTLinear(dim, dim, num_cores=cores, tt_rank=rank).eval()
    print()
    print(f"=== batch-size dependence at {dim}x{dim}, {cores} cores, rank {rank} ===")
    print(f'{"batch":>7} {"dense ms":>10} {"recon ms":>10} {"stream ms":>10} '
          f'{"stream/dense":>13} {"recon/dense":>12}')
    print("-" * 66)
    rows = []
    for batch in batches:
        x = torch.randn(batch, dim)
        reps = 50 if batch <= 64 else 10
        t_dense = _time(dense, x, reps)
        t_recon = _time(layer.forward, x, reps)
        t_stream = _time(layer.forward_streaming, x, reps)
        rows.append({"batch": batch, "dense_ms": t_dense, "recon_ms": t_recon,
                     "stream_ms": t_stream, "stream_ratio": t_stream / t_dense,
                     "recon_ratio": t_recon / t_dense})
        mark = f"  <-- {t_dense/t_stream:.1f}x faster" if t_stream < t_dense else ""
        print(f'{batch:>7} {t_dense:>10.3f} {t_recon:>10.3f} {t_stream:>10.3f} '
              f'{t_stream/t_dense:>12.2f}x {t_recon/t_dense:>11.2f}x{mark}')
    return rows


def analyse(frontier_rows, batch_rows) -> None:
    winners = [r for r in frontier_rows if r["ratio"] < 1.0]
    print()
    if winners:
        best = min(winners, key=lambda r: r["ratio"])
        print(f"Best configuration found: {best['cores']} cores, rank {best['rank']} -- "
              f"{1/best['ratio']:.2f}x faster than dense at {best['fraction']*100:.3f}% of parameters.")
        print(f"({len(winners)} of {len(frontier_rows)} configurations beat dense.)")
    else:
        print("No configuration beat dense in this sweep.")

    best_batch = min(batch_rows, key=lambda r: r["stream_ratio"])
    print()
    print(f"Streaming is most favourable at batch {best_batch['batch']}: "
          f"{1/best_batch['stream_ratio']:.2f}x faster than dense.")
    small = [r for r in batch_rows if r["batch"] <= 8]
    if small:
        print("Small-batch behaviour matters for edge inference, where a model runs on")
        print("one sample at a time and reconstruction has to rebuild the whole matrix")
        print("for each single prediction.")


if __name__ == "__main__":
    f = frontier()
    b = batch_sweep()
    analyse(f, b)
