from __future__ import annotations
from typing import Sequence
from qire_ai.search_spaces import ExperimentJob
from .cost_model import job_priority


def order_jobs_for_oracle(jobs: Sequence[ExperimentJob]) -> list[ExperimentJob]:
    """CPU/free-tier friendly order: cheap baselines, compact QIRE, then heavier models."""
    return sorted(jobs, key=lambda j: (-job_priority(j), j.dataset, j.model_key, j.seed, j.job_id))
