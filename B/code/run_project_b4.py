"""B-013 (Rang-Curriculum) und B-014 (intrinsische Dimension), seriell.

Beide speichern nach jedem Datensatz bzw. am Ende, sodass ein Abbruch nur die
laufende Stufe kostet.

B-012 liefert fuer B-013 eine Vorhersage, die vorher festgehalten sei: Der
Effektivrang der Mittelschicht liegt schon nach EINER Epoche bei rund 5, und
der Vorteil des spaeteren Beschraenkens ist nach etwa drei Epochen
ausgeschoepft. Wenn das stimmt, sollte ein ueber acht Epochen verteiltes
Curriculum NICHT besser sein als ein einzelner Schnitt bei Epoche 5 -- die
Kontrolle `single_cut` ist damit der eigentliche Gegner, nicht `fixed`.

Aufruf:
    python run_project_b4.py          # beide
    python run_project_b4.py 14       # nur B-014
"""

from __future__ import annotations

import sys
import time

from qire_ai import intrinsic_dimension, rank_curriculum


def main(stages: set[int]) -> None:
    start = time.perf_counter()
    if 13 in stages:
        for ds in rank_curriculum.DATASETS:
            print("\n" + "#" * 74)
            print(f"# B-013 -- Rang-Curriculum -- {ds}")
            print("#" * 74)
            rec = rank_curriculum.run_dataset(ds)
            rec.save(analysis=rank_curriculum.analyse(rec))
    if 14 in stages:
        print("\n" + "#" * 74)
        print("# B-014 -- intrinsische Dimension gegen Klassenzahl")
        print("#" * 74)
        rec = intrinsic_dimension.run()
        rec.save(analysis=intrinsic_dimension.analyse(rec))
    print(f"\nFertig in {(time.perf_counter() - start) / 3600:.2f} h.")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]} or {13, 14})
