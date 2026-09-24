from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .data_loader import DEFAULT_DATASETS, EXTENDED_DATASETS


@dataclass(frozen=True)
class ExperimentJob:
    dataset: str
    model_key: str
    seed: int
    epochs: int = 80
    lr: float = 0.01
    hidden_dim: int = 32
    expert_hidden_dim: int = 16
    num_experts: int = 4
    top_k: int = 2
    prune_amount: float = 0.50
    gate_type: str = "baseline"
    state_dim: int | None = None
    state_mapper: str = "trig"
    lambda_resource: float = 0.0
    mu_activation: float = 0.0
    batch_size: int = 0
    quant_bits: int = 8
    temperature: float = 1.0
    confidence_threshold: float = 0.86
    shared_backbone: bool = False
    tt_rank: int = 4
    tt_cores: int = 3
    max_samples: int | None = None
    search_mode: str = "manual"
    hypothesis_id: str = "H000"
    experiment_id: str = "E000"
    parent_job_id: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def stable_config(self) -> Dict[str, Any]:
        data = self.as_dict().copy()
        data.pop("notes", None)
        return data

    @property
    def job_id(self) -> str:
        payload = json.dumps(self.stable_config(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self.dataset}_{self.model_key}_s{self.seed}_{digest}"


MODEL_KEYS_CORE = ("logreg", "random_forest", "svm_linear", "svm_rbf", "mlp", "pruned_mlp", "standard_moe", "qire")
MODEL_KEYS_EXTENDED = (
    "logreg", "random_forest", "svm_linear", "svm_rbf", "mlp", "deep_mlp", "lowrank_mlp", "early_exit_mlp", "pruned_mlp", "quantized_mlp",
    "student_mlp", "standard_moe", "full_moe", "random_gate_moe", "qire",
)
QIRE_GATES = ("softmax_state", "born", "interference_born", "gumbel_softmax", "hard_threshold", "entropy_adaptive", "budget_aware", "hybrid_softmax_born")
STATE_MAPPERS = ("linear", "low_rank", "trig", "compact_trig", "poly2", "normalized", "cluster_based")


def baseline_jobs(
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    seeds: Sequence[int] = (1, 2, 3),
    epochs: int = 80,
    lr: float = 0.01,
) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed in itertools.product(datasets, seeds):
        jobs.extend([
            ExperimentJob(dataset, "logreg", seed, epochs, lr, search_mode="baseline", hypothesis_id="H001", experiment_id="E01", notes="Linear sanity baseline"),
            ExperimentJob(dataset, "random_forest", seed, epochs, lr, search_mode="baseline", hypothesis_id="H001RF", experiment_id="E01RF", notes="Real scikit-learn Random Forest baseline"),
            ExperimentJob(dataset, "svm_linear", seed, epochs, lr, search_mode="baseline", hypothesis_id="H001S", experiment_id="E01S", notes="Real scikit-learn linear SVM baseline"),
            ExperimentJob(dataset, "mlp", seed, epochs, lr, hidden_dim=32, search_mode="baseline", hypothesis_id="H001", experiment_id="E01"),
            ExperimentJob(dataset, "pruned_mlp", seed, epochs, lr, hidden_dim=32, prune_amount=0.50, search_mode="baseline", hypothesis_id="H002", experiment_id="E02"),
            ExperimentJob(dataset, "quantized_mlp", seed, epochs, lr, hidden_dim=32, quant_bits=8, search_mode="baseline", hypothesis_id="H002Q", experiment_id="E02Q"),
            ExperimentJob(dataset, "standard_moe", seed, epochs, lr, expert_hidden_dim=16, num_experts=4, top_k=2, gate_type="softmax", search_mode="baseline", hypothesis_id="H003", experiment_id="E03"),
            ExperimentJob(dataset, "qire", seed, epochs, lr, expert_hidden_dim=16, num_experts=4, top_k=2, gate_type="interference_born", state_mapper="trig", search_mode="baseline", hypothesis_id="H004", experiment_id="E04"),
        ])
    return dedupe_jobs(jobs)


def quick_jobs(*, epochs: int = 40, seeds: Sequence[int] = (1, 2, 3)) -> List[ExperimentJob]:
    jobs = baseline_jobs(seeds=seeds, epochs=epochs)
    for dataset, seed in itertools.product(DEFAULT_DATASETS, seeds):
        jobs.append(ExperimentJob(dataset, "qire", seed, epochs, 0.01, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="born", state_mapper="linear", search_mode="quick_ablation", hypothesis_id="H005", experiment_id="E05"))
        jobs.append(ExperimentJob(dataset, "qire", seed, epochs, 0.01, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="softmax_state", state_mapper="linear", search_mode="quick_ablation", hypothesis_id="H006", experiment_id="E06"))
    return dedupe_jobs(jobs)


def ablation_jobs(*, epochs: int = 60, seeds: Sequence[int] = (1, 2, 3, 4, 5), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed, gate, top_k, n_exp, exp_hid, mapper, temp in itertools.product(
        datasets, seeds, ("softmax_state", "born", "interference_born"), (1, 2), (2, 4, 8), (8, 16, 32), ("linear", "compact_trig", "trig"), (0.7, 1.0)
    ):
        if top_k <= n_exp:
            jobs.append(ExperimentJob(
                dataset=dataset, model_key="qire", seed=seed, epochs=epochs, lr=0.01, expert_hidden_dim=exp_hid,
                num_experts=n_exp, top_k=top_k, gate_type=gate, state_mapper=mapper, temperature=temp,
                search_mode="ablation", hypothesis_id="H010", experiment_id="E10",
                notes="QIRE ablation: gate, top_k, experts, expert size, state mapper, temperature",
            ))
    jobs.extend(baseline_jobs(datasets=datasets, seeds=seeds, epochs=epochs))
    return dedupe_jobs(jobs)


def overhead_jobs(*, epochs: int = 50, seeds: Sequence[int] = (1, 2, 3), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed, mapper, gate in itertools.product(datasets, seeds, STATE_MAPPERS, QIRE_GATES):
        jobs.append(ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=8, num_experts=4, top_k=1, gate_type=gate, state_mapper=mapper, search_mode="router_overhead", hypothesis_id="H011", experiment_id="E11", notes="Router-overhead decomposition via router_ops/expert_ops"))
    jobs.extend([ExperimentJob(ds, "standard_moe", s, epochs=epochs, expert_hidden_dim=8, num_experts=4, top_k=1, search_mode="router_overhead", hypothesis_id="H011C", experiment_id="E11C") for ds, s in itertools.product(datasets, seeds)])
    return dedupe_jobs(jobs)


def classical_baseline_jobs(*, epochs: int = 80, seeds: Sequence[int] = (1, 2, 3, 4, 5), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed, model_key, hidden in itertools.product(datasets, seeds, ("logreg", "random_forest", "svm_linear", "svm_rbf", "mlp", "deep_mlp", "lowrank_mlp", "early_exit_mlp", "pruned_mlp", "quantized_mlp", "student_mlp"), (16, 32, 64)):
        if model_key == "logreg" and hidden != 16:
            continue
        jobs.append(ExperimentJob(dataset, model_key, seed, epochs=epochs, hidden_dim=hidden, prune_amount=0.3 if hidden == 16 else 0.5, quant_bits=8, search_mode="classical_baselines", hypothesis_id="H030", experiment_id="E30"))
    return dedupe_jobs(jobs)


def dataset_scaling_jobs(*, epochs: int = 60, seeds: Sequence[int] = (1, 2, 3)) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    datasets = EXTENDED_DATASETS
    for dataset, seed in itertools.product(datasets, seeds):
        jobs.extend([
            ExperimentJob(dataset, "mlp", seed, epochs=epochs, hidden_dim=32, search_mode="dataset_scaling", hypothesis_id="H040", experiment_id="E40"),
            ExperimentJob(dataset, "pruned_mlp", seed, epochs=epochs, hidden_dim=32, prune_amount=0.5, search_mode="dataset_scaling", hypothesis_id="H041", experiment_id="E41"),
            ExperimentJob(dataset, "standard_moe", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=1, search_mode="dataset_scaling", hypothesis_id="H042", experiment_id="E42"),
            ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="born", state_mapper="linear", search_mode="dataset_scaling", hypothesis_id="H043", experiment_id="E43"),
        ])
    return dedupe_jobs(jobs)


def seed_stability_jobs(*, epochs: int = 70, seeds: Sequence[int] = tuple(range(1, 11)), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed in itertools.product(datasets, seeds):
        jobs.extend([
            ExperimentJob(dataset, "mlp", seed, epochs=epochs, hidden_dim=32, search_mode="seed_stability", hypothesis_id="H050", experiment_id="E50"),
            ExperimentJob(dataset, "pruned_mlp", seed, epochs=epochs, hidden_dim=32, prune_amount=0.5, search_mode="seed_stability", hypothesis_id="H051", experiment_id="E51"),
            ExperimentJob(dataset, "standard_moe", seed, epochs=epochs, expert_hidden_dim=16, num_experts=4, top_k=1, search_mode="seed_stability", hypothesis_id="H052", experiment_id="E52"),
            ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="born", state_mapper="linear", search_mode="seed_stability", hypothesis_id="H053", experiment_id="E53"),
        ])
    return dedupe_jobs(jobs)


def grid_search_jobs(*, epochs: int = 50, seeds: Sequence[int] = (1, 2, 3), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for dataset, seed, hidden, expert_h, n_exp, top_k in itertools.product(datasets, seeds, (16, 32, 64), (8, 16, 32), (2, 4, 8), (1, 2)):
        if top_k <= n_exp:
            jobs.append(ExperimentJob(dataset, "qire", seed, epochs=epochs, hidden_dim=hidden, expert_hidden_dim=expert_h, num_experts=n_exp, top_k=top_k, gate_type="born", state_mapper="linear", search_mode="grid_search", hypothesis_id="H060", experiment_id="E60"))
    return dedupe_jobs(jobs)


def random_search_jobs(*, epochs: int = 60, seeds: Sequence[int] = tuple(range(1, 8)), rng_seed: int = 42, max_random: int = 180, datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    rng = random.Random(rng_seed)
    jobs: List[ExperimentJob] = []
    for _ in range(max_random):
        dataset = rng.choice(list(datasets))
        seed = rng.choice(list(seeds))
        model_key = rng.choice(["mlp", "deep_mlp", "pruned_mlp", "quantized_mlp", "standard_moe", "qire", "qire", "qire"])
        n_exp = rng.choice([2, 3, 4, 6, 8])
        top_k = min(rng.choice([1, 2, 3]), n_exp)
        jobs.append(ExperimentJob(
            dataset=dataset, model_key=model_key, seed=seed, epochs=epochs, lr=rng.choice([0.003, 0.005, 0.01, 0.02]),
            hidden_dim=rng.choice([8, 16, 32, 48, 64, 96]), expert_hidden_dim=rng.choice([4, 8, 12, 16, 24, 32, 48]),
            num_experts=n_exp, top_k=top_k, prune_amount=rng.choice([0.25, 0.40, 0.50, 0.65, 0.80]),
            gate_type=rng.choice(QIRE_GATES), state_mapper=rng.choice(STATE_MAPPERS), lambda_resource=rng.choice([0.0, 1e-5, 3e-5]),
            mu_activation=rng.choice([0.0, 1e-4, 3e-4]), batch_size=rng.choice([0, 16, 32, 64]), quant_bits=rng.choice([4, 8, 16]),
            temperature=rng.choice([0.5, 0.7, 1.0, 1.5]), shared_backbone=rng.choice([False, False, True]),
            search_mode="random_search", hypothesis_id="H061", experiment_id="E61",
        ))
    return dedupe_jobs(jobs)


def successive_halving_jobs(*, epochs: int = 25, seeds: Sequence[int] = (1, 2), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    # First rung only; runner can later be used to reproduce winners with higher epochs.
    jobs = random_search_jobs(epochs=epochs, seeds=seeds, rng_seed=7, max_random=96, datasets=datasets)
    return [replace_search_mode(j, "successive_halving_rung1", "H062", "E62") for j in jobs]


def constrained_search_jobs(*, epochs: int = 55, seeds: Sequence[int] = (1, 2, 3), datasets: Sequence[str] = DEFAULT_DATASETS) -> List[ExperimentJob]:
    jobs = []
    for dataset, seed, n_exp, top_k, mapper in itertools.product(datasets, seeds, (2, 4), (1,), ("linear", "compact_trig")):
        jobs.append(ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=8, num_experts=n_exp, top_k=top_k, gate_type="born", state_mapper=mapper, lambda_resource=1e-5, mu_activation=1e-4, search_mode="constrained_low_ops", hypothesis_id="H063", experiment_id="E63", notes="Constrained search: small router and top-1 to reduce overhead"))
    jobs.extend([ExperimentJob(ds, "mlp", s, epochs=epochs, hidden_dim=16, search_mode="constrained_low_ops", hypothesis_id="H063B", experiment_id="E63B") for ds, s in itertools.product(datasets, seeds)])
    return dedupe_jobs(jobs)


def research8h_seed_jobs(*, epochs: int = 70, seeds: Sequence[int] = tuple(range(1, 11)), rng_seed: int = 42, max_random: int = 220) -> List[ExperimentJob]:
    jobs = []
    jobs.extend(baseline_jobs(seeds=seeds, epochs=epochs))
    jobs.extend(overhead_jobs(epochs=max(30, epochs // 2), seeds=seeds[:3] if isinstance(seeds, tuple) else list(seeds)[:3]))
    jobs.extend(random_search_jobs(epochs=epochs, seeds=seeds, rng_seed=rng_seed, max_random=max_random))
    jobs.extend(constrained_search_jobs(epochs=max(40, epochs - 20), seeds=seeds[:5] if isinstance(seeds, tuple) else list(seeds)[:5]))
    return dedupe_jobs(jobs)


def replace_search_mode(job: ExperimentJob, mode: str, hypothesis_id: str, experiment_id: str) -> ExperimentJob:
    data = job.as_dict()
    data.update({"search_mode": mode, "hypothesis_id": hypothesis_id, "experiment_id": experiment_id})
    return ExperimentJob(**data)


def dedupe_jobs(jobs: Sequence[ExperimentJob]) -> List[ExperimentJob]:
    seen = set()
    unique: List[ExperimentJob] = []
    for job in jobs:
        if job.job_id not in seen:
            unique.append(job)
            seen.add(job.job_id)
    return unique


def limit_jobs(jobs: Sequence[ExperimentJob], max_runs: int | None) -> List[ExperimentJob]:
    if max_runs is None or max_runs <= 0:
        return list(jobs)
    return list(jobs)[:max_runs]


HEAVY_DATASETS = (
    "wine", "breast_cancer", "digits", "synthetic_linear", "synthetic_nonlinear",
    "synthetic_noisy", "synthetic_high_dim", "synthetic_large", "synthetic_wide", "synthetic_sparse_signal", "synthetic_routing_challenge", "synthetic_expert_specialization", "synthetic_many_classes", "irrelevant_features_90", "clustered_subtasks", "noisy_subtasks", "feature_group_switching", "conditional_rules", "budget_sensitive_task",
)


def deep_research_jobs(*, epochs: int = 100, seeds: Sequence[int] = tuple(range(1, 8))) -> List[ExperimentJob]:
    """Large but bounded research preset. Designed for hours, not smoke tests."""
    jobs: List[ExperimentJob] = []
    datasets = ("wine", "breast_cancer", "digits", "synthetic_nonlinear", "synthetic_high_dim", "synthetic_sparse_signal")
    jobs.extend(classical_baseline_jobs(epochs=epochs, seeds=seeds, datasets=datasets[:4]))
    jobs.extend(ablation_jobs(epochs=max(70, epochs), seeds=seeds[:5] if isinstance(seeds, tuple) else list(seeds)[:5], datasets=datasets[:5]))
    jobs.extend(overhead_jobs(epochs=max(60, epochs // 2), seeds=seeds[:4] if isinstance(seeds, tuple) else list(seeds)[:4], datasets=datasets))
    jobs.extend(grid_search_jobs(epochs=max(70, epochs), seeds=seeds[:4] if isinstance(seeds, tuple) else list(seeds)[:4], datasets=datasets[:5]))
    jobs.extend(random_search_jobs(epochs=epochs, seeds=seeds, rng_seed=144, max_random=700, datasets=datasets))
    jobs.extend(constrained_search_jobs(epochs=max(70, epochs), seeds=seeds[:5] if isinstance(seeds, tuple) else list(seeds)[:5], datasets=datasets))
    return dedupe_jobs(jobs)


def day_research_jobs(*, epochs: int = 110, seeds: Sequence[int] = tuple(range(1, 13))) -> List[ExperimentJob]:
    """Very large local research queue intended to be stopped by a time budget.

    This intentionally plans far more jobs than a typical school PC can finish in a day.
    Use --time-budget-hours to stop cleanly and export partial results.
    """
    jobs: List[ExperimentJob] = []
    datasets = HEAVY_DATASETS
    jobs.extend(baseline_jobs(epochs=epochs, seeds=seeds[:8] if isinstance(seeds, tuple) else list(seeds)[:8], datasets=datasets[:6]))
    jobs.extend(classical_baseline_jobs(epochs=epochs, seeds=seeds[:6] if isinstance(seeds, tuple) else list(seeds)[:6], datasets=datasets[:7]))
    jobs.extend(seed_stability_jobs(epochs=max(90, epochs), seeds=seeds, datasets=("wine", "breast_cancer", "digits")))
    jobs.extend(ablation_jobs(epochs=epochs, seeds=seeds[:6] if isinstance(seeds, tuple) else list(seeds)[:6], datasets=datasets[:7]))
    jobs.extend(overhead_jobs(epochs=max(70, epochs // 2), seeds=seeds[:6] if isinstance(seeds, tuple) else list(seeds)[:6], datasets=datasets))
    jobs.extend(random_search_jobs(epochs=epochs, seeds=seeds, rng_seed=404, max_random=1600, datasets=datasets))
    jobs.extend(successive_halving_jobs(epochs=max(45, epochs // 3), seeds=seeds[:5] if isinstance(seeds, tuple) else list(seeds)[:5], datasets=datasets))
    jobs.extend(router_lab_jobs(epochs=max(60, epochs // 2), seeds=seeds[:4] if isinstance(seeds, tuple) else list(seeds)[:4]))
    jobs.extend(synthetic_benchmark_jobs(epochs=max(70, epochs // 2), seeds=seeds[:4] if isinstance(seeds, tuple) else list(seeds)[:4]))
    jobs.extend(constrained_search_jobs(epochs=max(80, epochs), seeds=seeds[:8] if isinstance(seeds, tuple) else list(seeds)[:8], datasets=datasets))
    # extra explicit QIRE heavy variants
    for dataset, seed, gate, mapper, top_k, n_exp, exp_hid, lam, mu in itertools.product(
        datasets, seeds[:5] if isinstance(seeds, tuple) else list(seeds)[:5],
        QIRE_GATES, STATE_MAPPERS, (1, 2, 3), (2, 4, 8, 12), (8, 16, 32, 64), (0.0, 1e-5, 3e-5), (0.0, 1e-4)
    ):
        if top_k <= n_exp:
            jobs.append(ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=exp_hid, num_experts=n_exp, top_k=top_k, gate_type=gate, state_mapper=mapper, lambda_resource=lam, mu_activation=mu, search_mode="day_research_qire", hypothesis_id="H200", experiment_id="E200", notes="Day research QIRE expansion"))
    return dedupe_jobs(jobs)




def router_lab_jobs(*, epochs: int = 60, seeds: Sequence[int] = (1, 2, 3), datasets: Sequence[str] = ("synthetic_routing_challenge", "synthetic_expert_specialization", "wine")) -> List[ExperimentJob]:
    """Isolated QIRE routing diagnostics: born/softmax/random controls on datasets where routing can matter."""
    jobs: List[ExperimentJob] = []
    for dataset, seed, gate, mapper, top_k in itertools.product(datasets, seeds, QIRE_GATES, STATE_MAPPERS, (1, 2)):
        jobs.append(ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=top_k, gate_type=gate, state_mapper=mapper, search_mode="router_lab", hypothesis_id="H300", experiment_id="E300", notes="Router Lab: gate/state mapper behavior diagnostics."))
    for dataset, seed in itertools.product(datasets, seeds):
        jobs.append(ExperimentJob(dataset, "standard_moe", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="softmax", search_mode="router_lab_control", hypothesis_id="H301", experiment_id="E301"))
        jobs.append(ExperimentJob(dataset, "random_gate_moe", seed, epochs=epochs, expert_hidden_dim=12, num_experts=4, top_k=1, gate_type="random", search_mode="router_lab_control", hypothesis_id="H302", experiment_id="E302"))
    return dedupe_jobs(jobs)


def synthetic_benchmark_jobs(*, epochs: int = 70, seeds: Sequence[int] = (1, 2, 3)) -> List[ExperimentJob]:
    datasets = ("synthetic_routing_challenge", "synthetic_expert_specialization", "synthetic_sparse_signal", "synthetic_many_classes", "synthetic_high_dim", "irrelevant_features_90", "clustered_subtasks", "noisy_subtasks", "feature_group_switching", "conditional_rules", "budget_sensitive_task")
    jobs: List[ExperimentJob] = []
    for dataset, seed in itertools.product(datasets, seeds):
        jobs.extend([
            ExperimentJob(dataset, "mlp", seed, epochs=epochs, hidden_dim=64, search_mode="synthetic_benchmark", hypothesis_id="H310", experiment_id="E310"),
            ExperimentJob(dataset, "pruned_mlp", seed, epochs=epochs, hidden_dim=64, prune_amount=0.5, search_mode="synthetic_benchmark", hypothesis_id="H311", experiment_id="E311"),
            ExperimentJob(dataset, "lowrank_mlp", seed, epochs=epochs, hidden_dim=64, state_dim=16, search_mode="synthetic_benchmark", hypothesis_id="H312", experiment_id="E312"),
            ExperimentJob(dataset, "standard_moe", seed, epochs=epochs, expert_hidden_dim=20, num_experts=4, top_k=1, gate_type="softmax", search_mode="synthetic_benchmark", hypothesis_id="H313", experiment_id="E313"),
            ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=16, num_experts=4, top_k=1, gate_type="born", state_mapper="low_rank", search_mode="synthetic_benchmark", hypothesis_id="H314", experiment_id="E314"),
            ExperimentJob(dataset, "qire", seed, epochs=epochs, expert_hidden_dim=16, num_experts=4, top_k=1, gate_type="interference_born", state_mapper="low_rank", search_mode="synthetic_benchmark", hypothesis_id="H315", experiment_id="E315"),
        ])
    return dedupe_jobs(jobs)



def discovery_proposal_jobs(*, epochs: int = 80, seeds: Sequence[int] = (1, 2, 3), max_jobs: int = 200) -> List[ExperimentJob]:
    """Jobs generated from the v0.6 Discovery Engine proposal queue."""
    from .discovery.discovery_engine import discovery_jobs
    return dedupe_jobs(discovery_jobs(seeds=list(seeds), epochs=epochs, max_jobs=max_jobs))


def hyperband_discovery_jobs(*, epochs: int = 12, seeds: Sequence[int] = (1, 2, 3), datasets: Sequence[str] = ("wine", "breast_cancer", "digits")) -> List[ExperimentJob]:
    from .discovery.hyperband_scheduler import hyperband_candidate_jobs
    return dedupe_jobs(hyperband_candidate_jobs(datasets=datasets, seeds=seeds, max_candidates=180))


def innovation_day_jobs(*, max_new_jobs: int = 2000, epochs: int = 90) -> List[ExperimentJob]:
    from .auto_innovator import innovation_jobs_if_enabled
    base = innovation_jobs_if_enabled(True, max_new_jobs=max_new_jobs)
    if base:
        # Convert weak short proposals into real research jobs.
        out = []
        for j in base:
            d = j.as_dict()
            d["epochs"] = max(int(d.get("epochs") or 0), epochs)
            d["search_mode"] = "innovation_day"
            d["notes"] = (d.get("notes") or "") + " | Innovation-day expanded from measured parents"
            out.append(ExperimentJob(**d))
        return dedupe_jobs(out)
    # If there are no prior results yet, fall back to a broad but safe search.
    return random_search_jobs(epochs=epochs, seeds=tuple(range(1, 8)), rng_seed=909, max_random=min(max_new_jobs, 600), datasets=("wine", "breast_cancer", "digits", "synthetic_high_dim", "synthetic_sparse_signal"))
