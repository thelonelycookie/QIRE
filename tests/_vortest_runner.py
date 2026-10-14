"""Startet vortest() EINES Moduls in einem eigenen Prozess (Isolation, Zeitlimit von aussen).

Aufruf: python tests/_vortest_runner.py <modulname>
Rueckgabecodes: 0 bestanden, 1 durchgefallen, 77 braucht Daten/Netz (-> pytest ueberspringt),
                78 Modul hat keinen vortest.
Der Netz-Zugriff der Daten-Ladefunktion (urllib.request.urlretrieve) wird gesperrt: Tests
laden NIE Datensaetze herunter, sie nutzen nur, was in INSTITUT/daten (oder QIRE_DATEN) liegt.
"""
import importlib
import os
import sys
import urllib.request
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("INSTITUT_PROZESSE", "1")
INSTITUT = Path(__file__).resolve().parent.parent / "INSTITUT"
sys.path.insert(0, str(INSTITUT))
os.chdir(INSTITUT)


class _KeinNetz(Exception):
    pass


def _gesperrt(*a, **k):
    raise _KeinNetz("Download gesperrt")


urllib.request.urlretrieve = _gesperrt
urllib.request.urlopen = _gesperrt

name = sys.argv[1]
modul = importlib.import_module(f"abteilungen.{name}")
if not hasattr(modul, "vortest"):
    sys.exit(78)
try:
    ergebnis = modul.vortest()
except _KeinNetz:
    print("DATEN_FEHLEN: Datensatz nicht in INSTITUT/daten vorhanden (Download in Tests gesperrt)")
    sys.exit(77)
except (FileNotFoundError, OSError) as fehler:       # z.B. fehlende gespeicherte Rohdaten
    if "daten" in str(fehler).lower() or "npz" in str(fehler).lower():
        print("DATEN_FEHLEN:", fehler)
        sys.exit(77)
    raise
print("VORTEST", "bestanden" if ergebnis else "DURCHGEFALLEN")
sys.exit(0 if ergebnis else 1)
