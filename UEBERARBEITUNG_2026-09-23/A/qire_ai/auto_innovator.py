from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

from .config import LOGS_DIR
from .research_store import ResearchStore
from .search_spaces import ExperimentJob, QIRE_GATES, STATE_MAPPERS, dedupe_jobs

POLICY_PATH = LOGS_DIR / "innovation_policy.json"

DEFAULT_POLICY = {
    "enabled_by_default": False,
    "requires_explicit_flag": True,
    "may_modify_source_code": False,
    "may_install_packages": False,
    "may_delete_results": False,
    "may_write_only_inside": ["results/logs", "results/raw", "results/sandbox"],
    "allowed_actions": [
        "read previous Raw Results",
        "rank configurations by measured multi-objective score",
        "mutate hyperparameters within bounded search space",
        "enqueue new ExperimentJob configurations",
        "write innovation proposals to logs/innovation_proposals.jsonl",
        "optionally generate sandbox-only candidate model files when code mutation is explicitly enabled",
    ],
    "scientific_warning": "Innovation mode proposes experiments; it does not prove claims, does not rewrite main code, and requires human review.",
}


def ensure_policy() -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not POLICY_PATH.exists():
        POLICY_PATH.write_text(json.dumps(DEFAULT_POLICY, indent=2, ensure_ascii=False), encoding="utf-8")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("may_modify_source_code"):
        raise RuntimeError("Unsafe policy: may_modify_source_code must remain false.")
    return policy


def _f(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def result_score(row: Dict[str, object]) -> float:
    """Conservative search heuristic, not a scientific claim."""
    acc = _f(row, "accuracy")
    bal = _f(row, "balanced_accuracy", acc)
    ops = max(1.0, _f(row, "estimated_ops", 1.0))
    infer = max(0.0001, _f(row, "inference_time_ms", 0.0001))
    instability_penalty = abs(acc - bal)
    return 0.65 * acc + 0.35 * bal - 0.000018 * math.sqrt(ops) - 0.002 * infer - 0.08 * instability_penalty


class EvolutionaryInnovationEngine:
    """Bounded self-improvement-by-experimentation engine.

    It evolves experiment configurations, not its own source code.
    """

    def __init__(self, seed: int = 2026):
        self.rng = random.Random(seed)
        self.proposal_log = LOGS_DIR / "innovation_proposals.jsonl"

    def propose(self, *, max_new_jobs: int = 48, base_seeds: tuple[int, ...] = (1, 2, 3)) -> List[ExperimentJob]:
        store = ResearchStore()
        rows = [r for r in store.load_raw_results(include_failed=False) if str(r.get("status", "ok")) == "ok"]
        if not rows:
            return self._cold_start(max_new_jobs=max_new_jobs, base_seeds=base_seeds)
        ranked = sorted(rows, key=result_score, reverse=True)[:24]
        proposals: List[ExperimentJob] = []
        # Exploit: mutate around strong configurations.
        for row in ranked[:14]:
            for seed in base_seeds:
                proposals.append(self._mutate_from_row(row, seed=seed, explore=False))
        # Explore: deliberately create small/novel QIRE configurations.
        datasets = sorted({str(r.get("dataset", "wine")) for r in ranked}) or ["wine", "breast_cancer"]
        for _ in range(max_new_jobs):
            dataset = self.rng.choice(datasets)
            seed = self.rng.choice(base_seeds)
            proposals.append(self._random_novel_qire(dataset, seed))
        sanitized = [self._sanitize_job(j) for j in proposals]
        unique = dedupe_jobs(sanitized)
        self._log(unique[:max_new_jobs], ranked)
        return unique[:max_new_jobs]


    def _sanitize_job(self, job: ExperimentJob) -> ExperimentJob:
        data = job.as_dict()
        if data["model_key"] == "qire":
            if data.get("gate_type") not in QIRE_GATES:
                data["gate_type"] = "born"
        elif data["model_key"] in {"standard_moe", "full_moe", "random_gate_moe"}:
            data["gate_type"] = "softmax" if data["model_key"] != "random_gate_moe" else "random"
        else:
            data["gate_type"] = "baseline"
            data["top_k"] = 1
            data["num_experts"] = max(1, int(data.get("num_experts", 1)))
        return ExperimentJob(**data)

    def _cold_start(self, *, max_new_jobs: int, base_seeds: tuple[int, ...]) -> List[ExperimentJob]:
        jobs: List[ExperimentJob] = []
        for dataset in ("wine", "breast_cancer", "digits"):
            for seed in base_seeds:
                for gate, mapper, top_k in [("born", "linear", 1), ("interference_born", "linear", 1), ("softmax_state", "linear", 1), ("born", "compact_trig", 1)]:
                    jobs.append(ExperimentJob(dataset, "qire", seed, epochs=55, expert_hidden_dim=8, num_experts=4, top_k=top_k, gate_type=gate, state_mapper=mapper, search_mode="innovation_cold_start", hypothesis_id="H090", experiment_id="E90", notes="Cold-start innovation: small bounded QIRE variant."))
        return dedupe_jobs(jobs)[:max_new_jobs]

    def _mutate_from_row(self, row: Dict[str, object], seed: int, explore: bool) -> ExperimentJob:
        def intv(key: str, default: int) -> int:
            try:
                return int(float(row.get(key, default) or default))
            except Exception:
                return default
        def fltv(key: str, default: float) -> float:
            try:
                return float(row.get(key, default) or default)
            except Exception:
                return default
        dataset = str(row.get("dataset", self.rng.choice(["wine", "breast_cancer"])))
        parent = str(row.get("job_id", ""))
        # Prefer QIRE and fair controls; not only the current winner.
        model_key = self.rng.choice(["qire", "qire", "standard_moe", "pruned_mlp", "quantized_mlp", "lowrank_mlp"])
        num_experts = self._mutate_choice(intv("num_experts", 4), [2, 3, 4, 6, 8])
        top_k = min(num_experts, self._mutate_choice(intv("top_k", 1), [1, 2, 3]))
        return ExperimentJob(
            dataset=dataset,
            model_key=model_key,
            seed=seed,
            epochs=self._mutate_choice(intv("epochs", 60), [35, 50, 70, 90, 120]),
            lr=self._mutate_choice(fltv("lr", 0.01), [0.003, 0.005, 0.01, 0.02]),
            hidden_dim=self._mutate_choice(intv("hidden_dim", 32), [8, 16, 32, 48, 64, 96]),
            expert_hidden_dim=self._mutate_choice(intv("expert_hidden_dim", 16), [4, 8, 12, 16, 24, 32, 48]),
            num_experts=num_experts,
            top_k=top_k,
            prune_amount=self._mutate_choice(fltv("prune_amount", 0.5), [0.25, 0.4, 0.5, 0.65, 0.8]),
            gate_type=self._mutate_choice(str(row.get("gate_type", "born")), list(QIRE_GATES)),
            state_mapper=self._mutate_choice(str(row.get("state_mapper", "linear")), list(STATE_MAPPERS)),
            lambda_resource=self._mutate_choice(fltv("lambda_resource", 0.0), [0.0, 1e-5, 3e-5, 1e-4]),
            mu_activation=self._mutate_choice(fltv("mu_activation", 0.0), [0.0, 1e-4, 3e-4, 1e-3]),
            batch_size=self._mutate_choice(intv("batch_size", 0), [0, 16, 32, 64]),
            quant_bits=self._mutate_choice(intv("quant_bits", 8), [4, 8, 16]),
            temperature=self._mutate_choice(fltv("temperature", 1.0), [0.5, 0.7, 1.0, 1.5]),
            shared_backbone=self.rng.choice([False, False, True]),
            search_mode="innovation_mutation",
            hypothesis_id="H091",
            experiment_id="E91",
            parent_job_id=parent,
            notes="Generated by disabled-by-default innovation engine; bounded hyperparameter mutation only, no main-code self-modification.",
        )

    def _random_novel_qire(self, dataset: str, seed: int) -> ExperimentJob:
        n_exp = self.rng.choice([2, 3, 4, 6, 8])
        return ExperimentJob(
            dataset=dataset, model_key="qire", seed=seed, epochs=self.rng.choice([35, 50, 70]), lr=self.rng.choice([0.005, 0.01, 0.02]),
            hidden_dim=self.rng.choice([16, 32, 64]), expert_hidden_dim=self.rng.choice([4, 8, 12, 16, 24]),
            num_experts=n_exp, top_k=min(self.rng.choice([1, 2]), n_exp), gate_type=self.rng.choice(list(QIRE_GATES)),
            state_mapper=self.rng.choice(list(STATE_MAPPERS)), lambda_resource=self.rng.choice([0.0, 1e-5, 3e-5]),
            mu_activation=self.rng.choice([0.0, 1e-4, 3e-4]), temperature=self.rng.choice([0.5, 0.7, 1.0, 1.5]),
            shared_backbone=self.rng.choice([False, False, True]), search_mode="innovation_novelty", hypothesis_id="H092", experiment_id="E92",
            notes="Novelty-search QIRE candidate within bounded safe search space.",
        )

    def _mutate_choice(self, current, choices):
        choices = list(choices)
        if current not in choices:
            choices.append(current)
        if self.rng.random() < 0.58:
            return self.rng.choice(choices)
        return current

    def _log(self, jobs: List[ExperimentJob], parents: List[Dict[str, object]]) -> None:
        with self.proposal_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "proposal_count": len(jobs),
                "parent_count": len(parents),
                "parents": [{"job_id": p.get("job_id"), "dataset": p.get("dataset"), "model": p.get("model_name"), "score": result_score(p)} for p in parents[:12]],
                "jobs": [j.as_dict() for j in jobs],
                "warning": DEFAULT_POLICY["scientific_warning"],
            }, ensure_ascii=False, sort_keys=True) + "\n")


def innovation_jobs_if_enabled(enable_innovation: bool, max_new_jobs: int = 48) -> List[ExperimentJob]:
    policy = ensure_policy()
    if not enable_innovation:
        print("Innovation mode is disabled. No self-evolving experiment proposals will be generated.")
        return []
    if policy.get("may_modify_source_code"):
        raise RuntimeError("Unsafe innovation policy: source-code modification must remain disabled.")
    engine = EvolutionaryInnovationEngine()
    return engine.propose(max_new_jobs=max_new_jobs)
