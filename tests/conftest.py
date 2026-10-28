"""Gemeinsame pytest-Einstellungen fuer tests/ (Repo-Wurzel).

- INSTITUT in den Suchpfad (wie bei `python direktor.py ...`)
- BLAS/OpenMP-Threads auf 1 (FEHLERBUCH 24.09.: Ueberbuchung in Parallelprozessen)
"""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

WURZEL = Path(__file__).resolve().parent.parent
INSTITUT = WURZEL / "INSTITUT"
WERKZEUGE = WURZEL / "werkzeuge"
for _p in (INSTITUT, WERKZEUGE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
