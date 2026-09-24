"""Vollstaendiger Durchlauf fuer Projekt B -- eine Datei, eine Reihenfolge.

Fuehrt alle Experimente aus, auf denen Projekt B beruht, und speichert jede
Stufe einzeln ab, bevor die naechste beginnt. Ein Abbruch mittendrin kostet
also nur die laufende Stufe.

    Stufe 0  Verifikation der Monarch-/BTT-Schicht und der lokalen Architektur.
             Schlaegt sie fehl, bricht alles ab -- ein Implementierungsfehler
             saehe sonst aus wie ein Ergebnis.
    Stufe 1  B-004-lr : Lernratensuche je Arm und Architektur (eigene Seeds)
    Stufe 2  B-004    : Hauptvergleich, gleiches Budget, drei Architekturen
    Stufe 3  B-006    : Konvergenzkontrolle -- haelt die Rangfolge, wenn
                        deutlich laenger und auf mehr Daten trainiert wird?
    Stufe 4  B-005    : Struktur der gelernten dichten Schicht

Aufruf:
    python run_project_b.py            # alles
    python run_project_b.py 2 3        # nur die Stufen 2 und 3
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Tuple

from qire_ai import ablation_locality as loc
from qire_ai import validate_structured, weight_structure
from qire_ai.result_store import load

# Konvergenzkontrolle: deutlich mehr Daten und Epochen als der Hauptlauf.
# Bewusst nur die vier aussagekraeftigsten Arme, sonst waere die Rechenzeit
# nicht vertretbar. Sechs Seeds, weil der Wilcoxon-Test darunter p<0,05 gar
# nicht erreichen kann (bei n=5 ist 0,0625 das Minimum) -- derselbe Fehler,
# der in diesem Projekt bei den Quantenkernen schon einmal passiert ist.
CONVERGENCE = dict(
    experiment_id="B-006",
    seeds=(1401, 1402, 1403, 1404, 1405, 1406),
    arms=("dense", "block_diag_128", "monarch_64x64", "tt_c2_r64"),
    archs=("dense_ends", "local_ends"),
    max_samples=15000,
    epochs=30,
    description=(
        "Konvergenzkontrolle zu B-004: dieselben Arme mit 15.000 Beispielen und "
        "30 Epochen statt 6.000 und 10. Prueft, ob die Rangfolge der Strukturen "
        "ein Artefakt des knappen Trainingsbudgets war. Lernraten aus B-004-lr "
        "uebernommen -- die dort gewaehlte Rate galt fuer das kuerzere Budget, "
        "das ist eine dokumentierte Annahme."
    ),
)


def stage_0_validate() -> None:
    print("\n" + "#" * 74)
    print("# Stufe 0 -- Verifikation der Schichten und der lokalen Architektur")
    print("#" * 74)
    if not validate_structured.main():
        raise SystemExit("Verifikation fehlgeschlagen -- Abbruch vor jedem Experiment.")


def stage_1_lr() -> Dict[Tuple[str, str], float]:
    print("\n" + "#" * 74)
    print("# Stufe 1 -- B-004-lr: Lernratensuche")
    print("#" * 74)
    best, rec = loc.sweep_learning_rate()
    loc.report_sweep(best, rec)
    rec.save(analysis={f"{a}|{b}": v for (a, b), v in best.items()})
    return best


def load_lr_map() -> Dict[Tuple[str, str], float]:
    """Lernraten aus einem frueheren Lauf holen, damit einzelne Stufen auch
    getrennt startbar sind."""
    payload = load("B-004-lr")
    out: Dict[Tuple[str, str], float] = {}
    for key, value in payload["analysis"].items():
        arm, arch = key.split("|")
        out[(arm, arch)] = float(value)
    print(f"  Lernraten aus {payload['created']} uebernommen ({len(out)} Zellen)")
    return out


def stage_2_main(lr_map: Dict[Tuple[str, str], float]) -> None:
    print("\n" + "#" * 74)
    print("# Stufe 2 -- B-004: Hauptvergleich bei gleichem Budget")
    print("#" * 74)
    rec = loc.run(lr_map=lr_map)
    rec.save(analysis=loc.analyse(rec))


def stage_3_convergence(lr_map: Dict[Tuple[str, str], float]) -> None:
    print("\n" + "#" * 74)
    print("# Stufe 3 -- B-006: Konvergenzkontrolle")
    print("#" * 74)
    rec = loc.run(lr_map=lr_map, **CONVERGENCE)     # type: ignore[arg-type]
    rec.save(analysis=loc.analyse(rec))


def stage_4_structure() -> None:
    print("\n" + "#" * 74)
    print("# Stufe 4 -- B-005: Struktur der gelernten dichten Schicht")
    print("#" * 74)
    rec = weight_structure.run()
    rec.save(analysis=weight_structure.analyse(rec))


def main(stages: set[int]) -> None:
    start = time.perf_counter()
    lr_map: Dict[Tuple[str, str], float] = {}

    if 0 in stages:
        stage_0_validate()
    if 1 in stages:
        lr_map = stage_1_lr()
    if (2 in stages or 3 in stages) and not lr_map:
        lr_map = load_lr_map()
    if 2 in stages:
        stage_2_main(lr_map)
    if 3 in stages:
        stage_3_convergence(lr_map)
    if 4 in stages:
        stage_4_structure()

    total = time.perf_counter() - start
    print(f"\nAlle gewaehlten Stufen fertig in {total/3600:.2f} h.")


if __name__ == "__main__":
    selected = {int(a) for a in sys.argv[1:]} or {0, 1, 2, 3, 4}
    main(selected)
