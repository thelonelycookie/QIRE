"""pytest-Einstellungen fuer labor/: INSTITUT in den Suchpfad, BLAS-Threads = 1 (Fehlerbuch: Ueberbuchung)."""
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
INSTITUT = Path(__file__).resolve().parent.parent
if str(INSTITUT) not in sys.path:
    sys.path.insert(0, str(INSTITUT))
