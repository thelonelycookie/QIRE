"""How does TT-Linear's parameter-count compression ratio scale with matrix
size? Mirrors benchmark_sparse_dispatch.py's crossover analysis, but for
parameter count rather than wall-clock time. Run:
python -m qire_ai.tensor_train_benchmark
"""
from __future__ import annotations

from .tensor_train import TTLinear


def run(sizes=None, tt_rank: int = 4, num_cores: int = 3):
    if sizes is None:
        sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    rows = []
    print(f'{"size":>8} {"dense params":>14} {"TT params":>12} {"ratio":>10}')
    for size in sizes:
        layer = TTLinear(in_features=size, out_features=size, num_cores=num_cores, tt_rank=tt_rank, bias=False)
        dense = layer.num_dense_equivalent_parameters()
        tt = layer.num_tt_parameters()
        ratio = tt / dense
        rows.append({"size": size, "dense_params": dense, "tt_params": tt, "ratio": ratio})
        print(f"{size:>8} {dense:>14} {tt:>12} {ratio:>10.4f}")
    return rows


def plot(rows, output_path: str = "results/figures/tensor_train_crossover.png") -> None:
    import matplotlib.pyplot as plt
    from pathlib import Path

    sizes = [r["size"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, ratios, marker="o")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity with dense (no savings)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matrix size (n x n)")
    ax.set_ylabel("TT params / dense params")
    ax.set_title("Tensor-Train compression ratio vs. matrix size")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
