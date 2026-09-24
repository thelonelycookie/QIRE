from __future__ import annotations
from dataclasses import asdict
from typing import List, Sequence
from qire_ai.search_spaces import ExperimentJob, random_search_jobs, dedupe_jobs


def hyperband_candidate_jobs(datasets: Sequence[str] = ("wine", "breast_cancer", "digits"), seeds: Sequence[int] = (1, 2, 3), max_candidates: int = 120) -> List[ExperimentJob]:
    # Rung 1: many candidates with few epochs. Later rungs are scheduled by reproduction_planner from results.
    jobs = random_search_jobs(epochs=12, seeds=seeds, rng_seed=606, max_random=max_candidates, datasets=datasets)
    out = []
    for j in jobs:
        data = j.as_dict(); data.update({"search_mode": "hyperband_rung1", "hypothesis_id": "H-HYPERBAND", "experiment_id": "E-HYPERBAND", "notes": "Hyperband rung 1: short trial; winners must be promoted manually or by discovery."})
        out.append(ExperimentJob(**data))
    return dedupe_jobs(out)
