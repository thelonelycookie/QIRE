from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    name: str
    input_dim: int
    num_classes: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_names: Tuple[str, ...]
    n_train: int
    n_test: int
    source: str
    notes: str = ""


DATASET_ALIASES: Dict[str, str] = {
    "iris": "iris",
    "wine": "wine",
    "breast_cancer": "breast_cancer",
    "breast-cancer": "breast_cancer",
    "cancer": "breast_cancer",
    "digits": "digits",
    "mnist": "mnist",
    "mnist_784": "mnist",
    "fashion_mnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "synthetic_linear": "synthetic_linear",
    "synthetic_nonlinear": "synthetic_nonlinear",
    "synthetic_moons": "synthetic_moons",
    "synthetic_circles": "synthetic_circles",
    "synthetic_noisy": "synthetic_noisy",
    "synthetic_imbalanced": "synthetic_imbalanced",
    "synthetic_high_dim": "synthetic_high_dim",
    "synthetic_blobs": "synthetic_blobs",
    "synthetic_large": "synthetic_large",
    "synthetic_wide": "synthetic_wide",
    "synthetic_sparse_signal": "synthetic_sparse_signal",
    "synthetic_routing_challenge": "synthetic_routing_challenge",
    "synthetic_expert_specialization": "synthetic_expert_specialization",
    "synthetic_many_classes": "synthetic_many_classes",
    "irrelevant_features_90": "irrelevant_features_90",
    "clustered_subtasks": "clustered_subtasks",
    "noisy_subtasks": "noisy_subtasks",
    "feature_group_switching": "feature_group_switching",
    "conditional_rules": "conditional_rules",
    "budget_sensitive_task": "budget_sensitive_task",
}


DEFAULT_DATASETS = ("wine", "breast_cancer")
EXTENDED_DATASETS = (
    "wine",
    "breast_cancer",
    "iris",
    "digits",
    "synthetic_linear",
    "synthetic_nonlinear",
    "synthetic_noisy",
    "synthetic_imbalanced",
    "synthetic_high_dim",
    "synthetic_large",
    "synthetic_wide",
    "synthetic_sparse_signal",
    "synthetic_routing_challenge",
    "synthetic_expert_specialization",
    "synthetic_many_classes",
    "irrelevant_features_90",
    "clustered_subtasks",
    "noisy_subtasks",
    "feature_group_switching",
    "conditional_rules",
    "budget_sensitive_task",
)


def available_datasets() -> Dict[str, str]:
    return {
        "iris": "sklearn toy multiclass dataset; useful as smoke test only",
        "wine": "sklearn multiclass tabular dataset; primary small benchmark",
        "breast_cancer": "sklearn binary tabular dataset; primary small benchmark",
        "digits": "sklearn 8x8 image tabularized dataset; medium local benchmark",
        "synthetic_linear": "make_classification, low nonlinearity",
        "synthetic_nonlinear": "make_classification with redundant features and clusters",
        "synthetic_moons": "two-moons nonlinear synthetic dataset",
        "synthetic_circles": "concentric circles nonlinear synthetic dataset",
        "synthetic_noisy": "classification with noisy labels/features",
        "synthetic_imbalanced": "class-imbalanced classification",
        "synthetic_high_dim": "higher-dimensional sparse-informative synthetic task",
        "synthetic_blobs": "cluster-based multiclass synthetic task",
        "synthetic_large": "larger local synthetic benchmark for longer CPU training",
        "synthetic_wide": "wide-feature benchmark for router-overhead tests",
        "synthetic_sparse_signal": "many irrelevant features; tests feature/router selectivity",
        "synthetic_routing_challenge": "mixture of subproblems designed for routing/expert tests",
        "synthetic_expert_specialization": "feature groups with different class rules; tests expert specialization",
        "synthetic_many_classes": "multiclass synthetic task with many classes",
        "irrelevant_features_90": "90 percent irrelevant feature benchmark",
        "clustered_subtasks": "clustered subtask benchmark for expert routing",
        "noisy_subtasks": "routing benchmark with different noise levels per subtask",
        "feature_group_switching": "samples rely on different feature groups",
        "conditional_rules": "rule-switching benchmark for conditional computation",
        "budget_sensitive_task": "task designed to test accuracy under operation budgets",
    }


def _load_raw(name: str, seed: int):
    name = DATASET_ALIASES.get(name.lower().strip(), name.lower().strip())
    rng = np.random.RandomState(seed)
    if name == "iris":
        raw = load_iris()
        return name, raw.data.astype(np.float32), raw.target.astype(np.int64), tuple(str(c) for c in raw.target_names), "sklearn.load_iris", "smoke-test dataset; too easy for final claims"
    if name == "wine":
        raw = load_wine()
        return name, raw.data.astype(np.float32), raw.target.astype(np.int64), tuple(str(c) for c in raw.target_names), "sklearn.load_wine", "primary small tabular benchmark"
    if name == "breast_cancer":
        raw = load_breast_cancer()
        return name, raw.data.astype(np.float32), raw.target.astype(np.int64), tuple(str(c) for c in raw.target_names), "sklearn.load_breast_cancer", "primary binary tabular benchmark"
    if name == "digits":
        raw = load_digits()
        x = (raw.data / 16.0).astype(np.float32)
        return name, x, raw.target.astype(np.int64), tuple(str(c) for c in raw.target_names), "sklearn.load_digits", "8x8 image dataset flattened to 64 features"
    if name in {"mnist", "fashion_mnist"}:
        # Real, large benchmark. Every other dataset here sits below the ~512-neuron
        # scale threshold measured in structured_pruning.py, where resource savings
        # stop translating into measured time; this one is big enough to test above it.
        openml_name = "mnist_784" if name == "mnist" else "Fashion-MNIST"
        cache = Path(__file__).resolve().parent.parent / "results" / "datasets"
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / f"{name}.npz"
        if cached.exists():
            blob = np.load(cached)
            x, y = blob["x"], blob["y"]
        else:
            from sklearn.datasets import fetch_openml

            raw = fetch_openml(openml_name, version=1, as_frame=False, parser="auto")
            x = (raw.data / 255.0).astype(np.float32)
            y = raw.target.astype(np.int64)
            np.savez_compressed(cached, x=x, y=y)
        return (
            name, x, y, tuple(str(c) for c in range(10)),
            f"openml.{openml_name}",
            "70k x 784 image benchmark; large enough to test above the measured scale threshold",
        )
    if name == "synthetic_linear":
        x, y = make_classification(
            n_samples=900, n_features=20, n_informative=12, n_redundant=0, n_repeated=0,
            n_classes=2, class_sep=1.8, flip_y=0.01, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1"), "sklearn.make_classification", "mostly linear synthetic benchmark"
    if name == "synthetic_nonlinear":
        x, y = make_classification(
            n_samples=1100, n_features=24, n_informative=10, n_redundant=6, n_classes=3,
            n_clusters_per_class=2, class_sep=1.05, flip_y=0.03, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1", "2"), "sklearn.make_classification", "moderately nonlinear multiclass synthetic benchmark"
    if name == "synthetic_noisy":
        x, y = make_classification(
            n_samples=1200, n_features=30, n_informative=8, n_redundant=8, n_classes=2,
            class_sep=0.75, flip_y=0.12, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1"), "sklearn.make_classification", "noisy labels/features stress test"
    if name == "synthetic_imbalanced":
        x, y = make_classification(
            n_samples=1000, n_features=22, n_informative=8, n_redundant=4, n_classes=2,
            weights=[0.86, 0.14], class_sep=1.1, flip_y=0.02, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("majority", "minority"), "sklearn.make_classification", "imbalanced-class stress test; use balanced accuracy"
    if name == "synthetic_high_dim":
        x, y = make_classification(
            n_samples=900, n_features=96, n_informative=12, n_redundant=12, n_classes=3,
            class_sep=1.0, flip_y=0.04, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1", "2"), "sklearn.make_classification", "high-dimensional sparse-informative stress test"
    if name == "synthetic_moons":
        x, y = make_moons(n_samples=900, noise=0.18, random_state=seed)
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1"), "sklearn.make_moons", "nonlinear 2D toy task"
    if name == "synthetic_circles":
        x, y = make_circles(n_samples=900, noise=0.12, factor=0.45, random_state=seed)
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1"), "sklearn.make_circles", "nonlinear radial toy task"
    if name == "synthetic_blobs":
        x, y = make_blobs(n_samples=900, centers=4, n_features=12, cluster_std=2.2, random_state=seed)
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(4)), "sklearn.make_blobs", "cluster-based multiclass synthetic task"

    if name == "synthetic_large":
        x, y = make_classification(
            n_samples=5000, n_features=80, n_informative=28, n_redundant=18, n_classes=4,
            n_clusters_per_class=2, class_sep=1.0, flip_y=0.035, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(4)), "sklearn.make_classification", "larger local synthetic benchmark; useful for multi-hour runs"
    if name == "synthetic_wide":
        x, y = make_classification(
            n_samples=2600, n_features=240, n_informative=18, n_redundant=28, n_classes=3,
            n_clusters_per_class=2, class_sep=0.95, flip_y=0.04, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(3)), "sklearn.make_classification", "wide-feature benchmark; stresses router/state mapper overhead"
    if name == "synthetic_sparse_signal":
        x, y = make_classification(
            n_samples=3000, n_features=160, n_informative=10, n_redundant=10, n_classes=2,
            class_sep=0.85, flip_y=0.06, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), ("0", "1"), "sklearn.make_classification", "many irrelevant features; useful for pruning and routing selectivity"

    if name == "synthetic_routing_challenge":
        # Three subproblems with distinct feature blocks; useful for testing whether routers specialize.
        n = 1800
        blocks = []
        labels = []
        for k in range(3):
            xk, yk = make_classification(
                n_samples=n // 3, n_features=18, n_informative=5, n_redundant=2, n_classes=2,
                class_sep=1.0 + 0.25 * k, flip_y=0.03 + 0.01 * k, random_state=seed + k,
            )
            prefix = np.zeros((xk.shape[0], 3), dtype=np.float32)
            prefix[:, k] = 1.0
            blocks.append(np.concatenate([prefix, xk], axis=1))
            labels.append(yk + 2 * k)
        x = np.vstack(blocks)
        y = np.concatenate(labels)
        perm = rng.permutation(len(y))
        return name, x[perm].astype(np.float32), y[perm].astype(np.int64), tuple(str(i) for i in range(6)), "synthetic generated", "routing challenge: separate subproblems should encourage expert specialization"
    if name == "synthetic_expert_specialization":
        n = 2200
        x, y = make_classification(
            n_samples=n, n_features=72, n_informative=18, n_redundant=12, n_classes=4,
            n_clusters_per_class=1, class_sep=0.9, flip_y=0.04, random_state=seed,
        )
        # Add blockwise nonlinear interactions to make different feature groups useful.
        x[:, 0:6] += (y[:, None] == 0).astype(np.float32) * 0.9
        x[:, 18:24] += (y[:, None] == 1).astype(np.float32) * -0.8
        x[:, 36:42] += (y[:, None] == 2).astype(np.float32) * 0.7
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(4)), "synthetic generated", "expert specialization challenge with feature-block signals"
    if name == "synthetic_many_classes":
        x, y = make_classification(
            n_samples=2400, n_features=64, n_informative=24, n_redundant=12, n_classes=8,
            n_clusters_per_class=1, class_sep=1.05, flip_y=0.03, random_state=seed,
        )
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(8)), "sklearn.make_classification", "many-class local stress test"


    if name == "irrelevant_features_90":
        x, y = make_classification(
            n_samples=2400, n_features=120, n_informative=12, n_redundant=0, n_classes=3,
            class_sep=1.0, flip_y=0.04, random_state=seed,
        )
        noise = rng.normal(0, 1, size=(x.shape[0], 80))
        x = np.concatenate([x[:, :40], noise], axis=1)
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(3)), "synthetic generated", "90-percent irrelevant-feature benchmark for feature selection/router selectivity"
    if name == "clustered_subtasks":
        blocks=[]; labels=[]
        for k in range(4):
            xk, yk = make_blobs(n_samples=450, centers=2, n_features=16, cluster_std=1.2 + 0.2*k, random_state=seed+k)
            prefix = np.zeros((xk.shape[0], 4), dtype=np.float32); prefix[:, k] = 1.0
            blocks.append(np.concatenate([prefix, xk], axis=1)); labels.append(yk + 2*k)
        x=np.vstack(blocks); y=np.concatenate(labels); perm=rng.permutation(len(y))
        return name, x[perm].astype(np.float32), y[perm].astype(np.int64), tuple(str(i) for i in range(8)), "synthetic generated", "clustered subtasks; routing/expert models should specialize"
    if name == "noisy_subtasks":
        blocks=[]; labels=[]
        for k, flip in enumerate([0.01, 0.05, 0.12]):
            xk, yk = make_classification(n_samples=600, n_features=24, n_informative=8, n_redundant=4, n_classes=2, class_sep=1.0, flip_y=flip, random_state=seed+11*k)
            prefix=np.zeros((xk.shape[0],3), dtype=np.float32); prefix[:,k]=1.0
            blocks.append(np.concatenate([prefix,xk], axis=1)); labels.append(yk+2*k)
        x=np.vstack(blocks); y=np.concatenate(labels); perm=rng.permutation(len(y))
        return name, x[perm].astype(np.float32), y[perm].astype(np.int64), tuple(str(i) for i in range(6)), "synthetic generated", "subtasks have different noise levels; tests adaptive computation"
    if name == "feature_group_switching":
        n=2100; x=rng.normal(0,1,(n,60)).astype(np.float32); group=rng.randint(0,3,n)
        score=np.zeros(n)
        score[group==0]=x[group==0,0:8].sum(axis=1)
        score[group==1]=x[group==1,20:28].sum(axis=1)
        score[group==2]=x[group==2,40:48].sum(axis=1)
        y=(score>0).astype(np.int64)+2*group
        prefix=np.eye(3,dtype=np.float32)[group]
        x=np.concatenate([prefix,x],axis=1)
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(6)), "synthetic generated", "different samples depend on different feature groups"
    if name == "conditional_rules":
        n=2200; x=rng.normal(0,1,(n,36)).astype(np.float32); switch=(x[:,0]>0).astype(np.int64)
        y=np.where(switch==0, (x[:,1]+x[:,2]-x[:,3] > 0), (x[:,10]*x[:,11]+x[:,12] > 0)).astype(np.int64) + 2*switch
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(4)), "synthetic generated", "conditional rule switching; tests whether routing learns rule-dependent paths"
    if name == "budget_sensitive_task":
        x, y = make_classification(n_samples=2600, n_features=72, n_informative=16, n_redundant=16, n_classes=4, class_sep=0.95, flip_y=0.05, random_state=seed)
        x[:, :8] += (y[:, None] == 0).astype(np.float32) * 0.3
        return name, x.astype(np.float32), y.astype(np.int64), tuple(str(i) for i in range(4)), "synthetic generated", "budget-sensitive benchmark for low-op models"

    raise ValueError(f"Unsupported dataset: {name}. Available: {', '.join(available_datasets())}")


# Datasets that are already on a sensible common scale and must NOT be run through
# per-feature standardisation. StandardScaler divides each feature by its own std,
# which is fine for tabular data but harmful for images: MNIST's quietest border
# pixel has std 0.000119, so standardising amplifies it 8368x, to >200 standard
# deviations. Measured on MNIST (784x512 net, 3 seeds): raw /255 scaling reaches
# 0.9518, a 1% variance floor 0.9391, plain StandardScaler 0.9311. Tabular sets
# genuinely need standardising -- on breast_cancer a variance floor *costs*
# 0.0198 (p=0.002), because its low-variance features carry real signal -- so this
# is opt-in per dataset and every existing dataset keeps its original behaviour.
PRESCALED_DATASETS = {"mnist", "fashion_mnist"}


def load_dataset(
    name: str,
    seed: int,
    test_size: float = 0.30,
    max_samples: int | None = None,
    standardise: bool | None = None,
) -> DatasetBundle:
    canonical, x, y, class_names, source, notes = _load_raw(name, seed)
    if standardise is None:
        standardise = canonical not in PRESCALED_DATASETS
    if max_samples is not None and max_samples > 0 and x.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(np.arange(x.shape[0]), size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    if standardise:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
    else:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
    return DatasetBundle(
        name=canonical,
        input_dim=int(x_train.shape[1]),
        num_classes=int(np.max(y) + 1),
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.long),
        class_names=tuple(class_names),
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        source=source,
        notes=notes,
    )
