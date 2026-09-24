"""Zweiter Durchlauf fuer Projekt B -- Vorhersage pruefen und Aequivalenz absichern.

Der erste Durchlauf (run_project_b.py) hat zwei Aussagen zurueckgezogen und eine
neue hinterlassen, die bei 10 Seeds nur eine Schranke von +-0,008 hergab. Dieser
Lauf holt beides nach:

    Stufe 1  B-008 : Rangreihe 4..256 gegen dicht auf MNIST UND Fashion-MNIST,
                     30 Seeds, plus das exakte Spektrum der gelernten dichten
                     Matrix. Prueft die aus B-005 abgeleitete Vorhersage, dass
                     die Spektralkonzentration den noetigen Rang bestimmt.
    Stufe 2  B-009 : Monarch gegen dicht in der lokalen Architektur, 30 Seeds,
                     beide Datensaetze -- der beste Aequivalenzkandidat aus
                     B-004 (dort -0,0008 bei nur 10 Seeds).

Beide Stufen speichern ihre Rohdaten, bevor die naechste beginnt.

Aufruf:
    python run_project_b2.py          # beide Stufen
    python run_project_b2.py 2        # nur Stufe 2
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Sequence, Tuple

from qire_ai import ablation_locality as loc
from qire_ai import equivalence, rank_prediction

SEEDS = tuple(range(1601, 1631))
DATASETS = ("mnist", "fashion_mnist")
B009_ARMS = ("dense", "monarch_64x64", "block_diag_128")


def stage_1_rank() -> None:
    print("\n" + "#" * 74)
    print("# Stufe 1 -- B-008: Rangreihe, Spektrum, zwei Datensaetze, 30 Seeds")
    print("#" * 74)
    rec = rank_prediction.run(seeds=SEEDS, datasets=DATASETS)
    rec.save(analysis=rank_prediction.analyse(rec))


def stage_2_monarch() -> None:
    print("\n" + "#" * 74)
    print("# Stufe 2 -- B-009: Monarch gegen dicht, lokale Architektur, 30 Seeds")
    print("#" * 74)
    for dataset in DATASETS:
        print(f"\n  Lernratensuche fuer {dataset} (local_ends)")
        best = rank_prediction.sweep_lr(
            dataset, B009_ARMS, width=loc.WIDTH, groups=loc.GROUPS,
            max_samples=6000, epochs=10, batch_size=256, threads=6,
            arch="local_ends")
        lr_map: Dict[Tuple[str, str], float] = {
            (arm, "local_ends"): lr for arm, lr in best.items()}

        rec = loc.run(
            lr_map=lr_map, seeds=SEEDS, arms=B009_ARMS, archs=("local_ends",),
            dataset=dataset, experiment_id=f"B-009-{dataset}",
            description=(
                f"Monarch und Blockdiagonale gegen eine dichte Mittelschicht in der "
                f"Architektur ohne globale Mischschichten, {dataset}, 30 Seeds. "
                f"Zweck ist der Aequivalenznachweis, nicht der Unterschiedsnachweis "
                f"-- deshalb die hohe Seed-Zahl."),
        )
        rec.save()
        print(f"\n  Aequivalenz gegen die dichte Schicht ({dataset}):")
        equivalence.family(rec.records, "dense",
                           [a for a in B009_ARMS if a != "dense"],
                           f"B-009 / {dataset} / local_ends")


def main(stages: Sequence[int]) -> None:
    start = time.perf_counter()
    if 1 in stages:
        stage_1_rank()
    if 2 in stages:
        stage_2_monarch()
    print(f"\nFertig in {(time.perf_counter()-start)/3600:.2f} h.")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]} or {1, 2})
