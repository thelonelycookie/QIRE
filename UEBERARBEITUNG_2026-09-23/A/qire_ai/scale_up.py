from __future__ import annotations

"""Do the project's three findings still hold above the measured scale threshold?

Every experiment so far ran at 32 hidden neurons on 13-80 input features.
structured_pruning.py then measured that resource savings only turn into
measured wall-clock time above roughly 512 neurons -- which means all previous
experiments sat *below* the point where the project's own question becomes
practically relevant. This module tests the findings on MNIST (784 features,
real image data) at 512 and 2048 neurons, i.e. on both sides of that threshold.

Three arms, all on the same task and hidden width:
    dense       unconstrained baseline
    tensor_train  quantum-inspired constraint (finding 1)
    pruned+ft   train wide, halve the neurons, fine-tune (finding 2 + 3)

Both accuracy and measured inference time are recorded, because the whole
point is that those two have repeatedly disagreed in this project.

Threading note: training uses all cores, because accuracy does not depend on
thread count and MNIST is otherwise too slow on CPU. The timing measurements
are forced back to a single thread, matching benchmark_sparse_dispatch.py and
structured_pruning.py so the numbers stay comparable to every earlier timing
figure here.

Run: python -m qire_ai.scale_up
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .metrics import evaluate
from .models_baseline import DenseMLP, TensorTrainMLP
from .structured_pruning import prune_to_width


def _accuracy(model, bundle) -> float:
    _, _, _, extras = evaluate(model, bundle.x_test, bundle.y_test)
    return float(extras.get("balanced_accuracy", 0.0))


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _time_inference(model, x: torch.Tensor, repeats: int = 20) -> float:
    """Single-threaded, to stay comparable with earlier timing work."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(3):
                model(x)
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(x)
            return (time.perf_counter() - t0) / repeats * 1000
    finally:
        torch.set_num_threads(previous)


def run(
    hidden_sizes: Sequence[int] = (512, 2048),
    tt_ranks: Sequence[int] = (16, 32),
    seeds: Sequence[int] = (301, 302, 303),
    max_samples: int = 20000,
    epochs: int = 30,
    finetune_epochs: int = 8,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    timing_batch = 1000

    for hidden in hidden_sizes:
        acc: Dict[str, List[float]] = {}
        par: Dict[str, int] = {}
        tim: Dict[str, float] = {}

        for seed in seeds:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            x_time = bundle.x_test[:timing_batch]

            dense = DenseMLP(bundle.input_dim, hidden, bundle.num_classes)
            train_model(dense, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                        seed=seed, batch_size=batch_size)
            acc.setdefault("dense", []).append(_accuracy(dense, bundle))
            par["dense"] = _params(dense)
            tim["dense"] = _time_inference(dense, x_time)

            for rank in tt_ranks:
                label = f"tensor_train_r{rank}"
                set_seed(seed)
                torch.set_num_threads(threads)
                model = TensorTrainMLP(bundle.input_dim, hidden, bundle.num_classes,
                                       tt_rank=rank, num_cores=3)
                train_model(model, bundle.x_train, bundle.y_train, epochs=epochs, lr=lr,
                            seed=seed, batch_size=batch_size)
                acc.setdefault(label, []).append(_accuracy(model, bundle))
                par[label] = _params(model)
                tim[label] = _time_inference(model, x_time)

            pruned = prune_to_width(dense, hidden // 2)
            torch.set_num_threads(threads)
            train_model(pruned, bundle.x_train, bundle.y_train, epochs=finetune_epochs,
                        lr=lr, seed=seed, batch_size=batch_size)
            acc.setdefault("pruned_half+ft", []).append(_accuracy(pruned, bundle))
            par["pruned_half+ft"] = _params(pruned)
            tim["pruned_half+ft"] = _time_inference(pruned, x_time)

            print(f"  hidden={hidden} seed={seed} done")

        base_acc = float(np.mean(acc["dense"]))
        base_par = par["dense"]
        base_time = tim["dense"]
        for label in acc:
            rows.append({
                "hidden": hidden, "arm": label,
                "accuracy": float(np.mean(acc[label])),
                "acc_drop": base_acc - float(np.mean(acc[label])),
                "params": par[label],
                "param_fraction": par[label] / base_par,
                "infer_ms": tim[label],
                "speedup": base_time / tim[label],
            })
        print(f"hidden={hidden} complete")
    return rows


def analyse(rows: List[Dict[str, object]]) -> None:
    print()
    header = (f'{"hidden":>7} {"arm":>16} {"accuracy":>9} {"drop":>8} '
              f'{"params":>10} {"param frac":>11} {"infer ms":>9} {"speedup":>8}')
    print(header); print("-" * len(header))
    for r in rows:
        print(f'{r["hidden"]:>7} {r["arm"]:>16} {r["accuracy"]:>9.4f} {r["acc_drop"]:>+8.4f} '
              f'{r["params"]:>10,} {r["param_fraction"]:>11.3f} {r["infer_ms"]:>9.3f} '
              f'{r["speedup"]:>7.2f}x')
    print()
    print("speedup > 1.00x means the saving finally shows up in measured time.")
    print("Below the ~512-neuron threshold nothing in this project ever managed that,")
    print("so the question is whether parameter savings and time savings agree here.")


if __name__ == "__main__":
    analyse(run())
