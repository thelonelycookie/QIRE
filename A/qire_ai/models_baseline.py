from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseMLP(nn.Module):
    """Small dense MLP baseline for tabular classification."""

    model_name = "MLP"
    method = "dense_mlp_baseline"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.method = f"dense_mlp_baseline_h{self.hidden_dim}"
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        return int(2 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.num_classes))

    def estimated_active_params(self) -> int:
        return int(self.input_dim * self.hidden_dim + self.hidden_dim * self.num_classes)


class LogisticRegressionTorch(nn.Module):
    """Linear softmax classifier baseline. Useful sanity check for tabular tasks."""

    model_name = "LogisticRegression"
    method = "linear_softmax_classifier"

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        logits = self.linear(x)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        return int(2 * self.input_dim * self.num_classes)


class DeepMLP(nn.Module):
    model_name = "DeepMLP"
    method = "two_hidden_layer_mlp"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.method = f"two_hidden_layer_mlp_h{self.hidden_dim}"
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        return int(2 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim + self.hidden_dim * self.num_classes))


class LowRankMLP(nn.Module):
    """Low-rank factorized first layer baseline."""

    model_name = "LowRankMLP"
    method = "low_rank_factorized_mlp"

    def __init__(self, input_dim: int, rank_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.rank_dim = int(max(2, rank_dim))
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.method = f"low_rank_factorized_mlp_r{self.rank_dim}_h{self.hidden_dim}"
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.rank_dim), nn.ReLU(),
            nn.Linear(self.rank_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        return int(2 * (self.input_dim * self.rank_dim + self.rank_dim * self.hidden_dim + self.hidden_dim * self.num_classes))


class TensorTrainMLP(nn.Module):
    """MLP whose first (largest) layer is Tensor-Train factorized instead of
    dense -- see tensor_train.py. Only the input->hidden layer is factorized
    (hidden->num_classes is normally tiny already, e.g. 32x3, not worth it).

    Important honesty note, same pattern as the SparseExpertLayer finding
    earlier in this project: TTLinear.forward() reconstructs the full dense
    matrix from its cores every call, then does an ordinary matmul against
    it. That means this model gets a real, measurable reduction in
    *parameter_count* (see `count_parameters()` in metrics.py, already
    tracked per experiment row) -- but *not* a reduction in estimated_ops,
    since the reconstruction itself costs compute and the matmul afterwards
    is still full-size. Getting a real per-sample compute reduction would
    need a "streaming" TT-contraction that applies the cores directly to the
    input without ever forming the dense matrix -- a further, separate
    implementation step, not assumed to already be true here.
    """

    model_name = "TensorTrainMLP"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, tt_rank: int = 4, num_cores: int = 3):
        super().__init__()
        from .tensor_train import TTLinear

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.tt_rank = int(tt_rank)
        self.num_cores = int(num_cores)
        self.tt1 = TTLinear(self.input_dim, self.hidden_dim, num_cores=self.num_cores, tt_rank=self.tt_rank)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.method = f"tensor_train_mlp_r{self.tt_rank}_c{self.num_cores}_h{self.hidden_dim}"

    def forward(self, x: torch.Tensor):
        h = self.relu(self.tt1(x))
        logits = self.fc2(h)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        # TT-layer ops: reconstructing the weight (core contractions) + the matmul itself.
        tt_reconstruct_ops = 2 * sum(c.numel() for c in self.tt1.cores)
        tt_matmul_ops = 2 * self.input_dim * self.hidden_dim
        fc2_ops = 2 * self.hidden_dim * self.num_classes
        return int(tt_reconstruct_ops + tt_matmul_ops + fc2_ops)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EarlyExitMLP(nn.Module):
    """Adaptive-inference baseline with a cheap early classifier.

    During evaluation, confident samples use the early head. During training, both
    heads are trained via a combined loss in benchmark.train_model.
    """

    model_name = "EarlyExitMLP"
    method = "adaptive_early_exit"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, confidence_threshold: float = 0.86):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.method = f"adaptive_early_exit_h{self.hidden_dim}_c{self.confidence_threshold:g}"
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.early = nn.Linear(self.hidden_dim, self.num_classes)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.final = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        early_logits = self.early(h)
        if self.training:
            h2 = F.relu(self.fc2(h))
            final_logits = self.final(h2)
            aux = {"early_logits": early_logits, "active_model_parts": 2.0, "gate_entropy": 0.0}
            return final_logits, aux
        probs = torch.softmax(early_logits, dim=-1)
        conf = probs.max(dim=-1).values
        use_early = conf >= self.confidence_threshold
        if bool(use_early.all()):
            logits = early_logits
        else:
            h2 = F.relu(self.fc2(h))
            final_logits = self.final(h2)
            logits = torch.where(use_early.unsqueeze(-1), early_logits, final_logits)
        active = 1.0 + float((~use_early).float().mean().detach().cpu())
        aux = {"active_model_parts": active, "gate_entropy": 0.0, "early_exit_rate": float(use_early.float().mean().detach().cpu())}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        early_ops = 2 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.num_classes)
        late_ops = 2 * (self.hidden_dim * self.hidden_dim + self.hidden_dim * self.num_classes)
        # Static estimate assumes half samples require full path; dynamic measured active parts logged separately.
        return int(early_ops + 0.5 * late_ops)


class SklearnClassifierWrapper(nn.Module):
    """scikit-learn baseline adapter used for real non-neural classical controls.

    It is intentionally handled by benchmark.py through fit_sklearn/predict_logits.
    The output logits are confidence/probability-like scores, not neural activations.
    """

    def __init__(self, kind: str, input_dim: int, num_classes: int, seed: int = 0):
        super().__init__()
        self.kind = kind
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.model_name = {
            "random_forest": "RandomForest",
            "svm_rbf": "SVM_RBF",
            "svm_linear": "SVM_Linear",
        }.get(kind, kind)
        self.method = f"sklearn_{kind}_baseline"
        self.estimator = None

    def fit_sklearn(self, x_train, y_train):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC, LinearSVC
        x = x_train.detach().cpu().numpy()
        y = y_train.detach().cpu().numpy()
        if self.kind == "random_forest":
            self.estimator = RandomForestClassifier(n_estimators=120, max_depth=None, random_state=self.seed, n_jobs=1)
        elif self.kind == "svm_rbf":
            self.estimator = SVC(kernel="rbf", C=2.0, gamma="scale", probability=False, random_state=self.seed)
        elif self.kind == "svm_linear":
            self.estimator = LinearSVC(C=1.0, random_state=self.seed, max_iter=5000)
        else:
            raise ValueError(f"unknown sklearn baseline kind: {self.kind}")
        self.estimator.fit(x, y)

    def predict_logits(self, x):
        import numpy as np
        import torch
        if self.estimator is None:
            raise RuntimeError("sklearn baseline has not been fitted yet")
        x_np = x.detach().cpu().numpy()
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(x_np)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
        elif hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(x_np)
        else:
            pred = self.estimator.predict(x_np)
            scores = np.zeros((len(pred), self.num_classes), dtype=np.float32)
            scores[np.arange(len(pred)), pred] = 1.0
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor):
        logits = self.predict_logits(x)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        if self.kind == "random_forest":
            # proxy: number of trees times approximate feature checks; not exact FLOPs
            return int(120 * min(32, self.input_dim))
        if self.kind == "svm_rbf":
            return int(2 * self.input_dim * 200)  # proxy for support-vector comparisons
        if self.kind == "svm_linear":
            return int(2 * self.input_dim * self.num_classes)
        return int(2 * self.input_dim * self.num_classes)

    def parameters(self, recurse: bool = True):
        return iter(())
