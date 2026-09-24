from __future__ import annotations

"""Where does streaming TT contraction actually beat a dense matmul?

scale_up.py showed Tensor-Train compressing MNIST layers to 3.5% of their
parameters while still being *slower* than dense (0.74-0.85x), because
TTLinear.forward() rebuilds the full matrix on every call. forward_streaming()
removes that step by contracting the cores straight against the input.

That does not automatically win. Dense costs prod(n)*prod(m) per sample. The
streaming contraction carries both the not-yet-consumed input factors and the
already-accumulated output factors at once, costing roughly d*q^(d+1)*r^2 for a
roughly square factorisation of mode size q. So:

    streaming / dense  ~  d * r^2 * q^(1-d)

which predicts streaming only wins with many cores and small rank -- and that
the current setting (d=3, q~8, r=16) should be about 12x *worse*, not better.

This script measures the real ratio across (cores, rank) and checks that
prediction. Single-threaded, matching every other timing figure in the project.

Run: python -m qire_ai.benchmark_tt_streaming
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .tensor_train import TTLinear, factorize_balanced


def _time(fn, x, repeats: int = 20) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(x)
    return (time.perf_counter() - t0) / repeats * 1000


def predicted_ratio(in_features: int, out_features: int, cores: int, rank: int) -> float:
    """d * r^2 * q^(1-d), with q the geometric-mean mode size."""
    in_f = factorize_balanced(in_features, cores)
    out_f = factorize_balanced(out_features, cores)
    q = float(np.exp(np.mean(np.log(in_f + out_f))))
    return cores * (rank ** 2) * q ** (1 - cores)


def run(
    in_features: int = 784,
    out_features: int = 512,
    core_counts: Sequence[int] = (3, 4, 5, 6),
    ranks: Sequence[int] = (2, 4, 8, 16),
    batch: int = 1000,
) -> List[Dict[str, object]]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dense = nn.Linear(in_features, out_features).eval()
    x = torch.randn(batch, in_features)
    dense_ms = _time(dense, x)
    print(f"dense reference: {in_features}x{out_features}, batch {batch}: {dense_ms:.3f} ms")
    print()
    header = (f'{"cores":>6} {"rank":>5} {"params":>9} {"param frac":>11} '
              f'{"recon ms":>9} {"stream ms":>10} {"stream/dense":>13} {"predicted":>10}')
    print(header); print("-" * len(header))

    rows = []
    for cores in core_counts:
        for rank in ranks:
            layer = TTLinear(in_features, out_features, num_cores=cores, tt_rank=rank).eval()
            recon_ms = _time(layer.forward, x)
            stream_ms = _time(layer.forward_streaming, x)
            ratio = stream_ms / dense_ms
            pred = predicted_ratio(in_features, out_features, cores, rank)
            frac = layer.num_tt_parameters() / (in_features * out_features + out_features)
            rows.append({
                "cores": cores, "rank": rank, "params": layer.num_tt_parameters(),
                "param_fraction": frac, "recon_ms": recon_ms, "stream_ms": stream_ms,
                "stream_over_dense": ratio, "predicted": pred,
            })
            marker = "  <-- schneller als dicht" if ratio < 1.0 else ""
            print(f'{cores:>6} {rank:>5} {layer.num_tt_parameters():>9,} {frac:>11.4f} '
                  f'{recon_ms:>9.3f} {stream_ms:>10.3f} {ratio:>12.2f}x {pred:>9.2f}x{marker}')
    return rows


def analyse(rows: List[Dict[str, object]]) -> None:
    from scipy import stats

    measured = np.array([r["stream_over_dense"] for r in rows])
    predicted = np.array([r["predicted"] for r in rows])
    r_s, p_s = stats.spearmanr(predicted, measured)
    print()
    print(f"Does the cost model predict the measurement? Spearman rho={r_s:+.3f} (p={p_s:.6f})")

    winners = [r for r in rows if r["stream_over_dense"] < 1.0]
    print()
    if winners:
        best = min(winners, key=lambda r: r["stream_over_dense"])
        print(f"{len(winners)} of {len(rows)} configurations beat dense in measured time.")
        print(f"Best: {best['cores']} cores, rank {best['rank']} -- "
              f"{1/best['stream_over_dense']:.2f}x faster than dense, "
              f"at {best['param_fraction']*100:.2f}% of the parameters.")
    else:
        print("No configuration beat dense. Streaming removes the reconstruction cost")
        print("but the contraction itself stays more expensive at this shape.")


def scale_sweep(
    dims: Sequence[int] = (512, 1024, 2048, 4096),
    cores: int = 4,
    rank: int = 4,
    batch: int = 256,
) -> List[Dict[str, object]]:
    """Square layers of growing size, to locate the crossover. The (cores, rank)
    sweep above fixes the shape and varies the factorisation; this fixes a good
    factorisation and varies the size, which is where the crossover lives."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    print()
    print(f"Scale sweep (cores={cores}, rank={rank}, batch={batch})")
    header = f'{"layer":>13} {"dense ms":>9} {"recon ms":>9} {"stream ms":>10} {"stream/dense":>13} {"param frac":>11}'
    print(header); print("-" * len(header))
    rows = []
    for dim in dims:
        dense = nn.Linear(dim, dim).eval()
        layer = TTLinear(dim, dim, num_cores=cores, tt_rank=rank).eval()
        x = torch.randn(batch, dim)
        t_dense = _time(dense, x, repeats=10)
        t_recon = _time(layer.forward, x, repeats=10)
        t_stream = _time(layer.forward_streaming, x, repeats=10)
        frac = layer.num_tt_parameters() / (dim * dim + dim)
        rows.append({"dim": dim, "dense_ms": t_dense, "recon_ms": t_recon,
                     "stream_ms": t_stream, "stream_over_dense": t_stream / t_dense,
                     "param_fraction": frac})
        marker = "  <-- faster than dense" if t_stream < t_dense else ""
        print(f'{dim:>6}x{dim:<6} {t_dense:>9.2f} {t_recon:>9.2f} {t_stream:>10.2f} '
              f'{t_stream/t_dense:>12.2f}x {frac:>11.5f}{marker}')
    print()
    print("Reconstruction can never beat dense: it performs dense's matmul *plus*")
    print("building the matrix first. Streaming avoids that matmul entirely, so it")
    print("is the only route to a genuine speedup -- once the layer is big enough")
    print("that its many small contractions stop being overhead-dominated.")
    return rows


if __name__ == "__main__":
    analyse(run())
    scale_sweep()
