from __future__ import annotations
from dataclasses import replace
from typing import Sequence
from qire_ai.search_spaces import ExperimentJob


def adapt_epochs_for_run_type(jobs: Sequence[ExperimentJob], run_type: str, profile: str = "local") -> list[ExperimentJob]:
    """Return jobs with epochs adapted to research stage.

    Smoke/exploration are deliberately cheaper; verification/final evaluation
    are allowed to be more expensive. This preserves reproducibility because the
    modified epoch count is part of each job hash/config.
    """
    out = []
    for job in jobs:
        e = job.epochs
        if run_type == "smoke":
            e = min(e, 8)
        elif run_type == "exploration":
            e = min(e, 80 if profile == "oracle_free" else 100)
        elif run_type == "verification":
            e = max(e, 100)
        elif run_type == "final_evaluation":
            e = max(e, 140)
        out.append(replace(job, epochs=e))
    return out
