import sys
from pathlib import Path

import pytest

ORDNER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ORDNER))

from schmiede.betrieb import lade_config  # noqa: E402


@pytest.fixture
def cfg():
    c = lade_config()
    c["betrieb"].update(ideen_pro_runde=10, herzschlag_intervall_s=0.0, backup_intervall_s=0.0,
                        fehler_pause_s=0.0, checkpoint_alle_ideen=1)
    c["waechter"].update(intervall_s=0.0, drossel_pause_s=0.01, pause_pruefschritt_s=0.05)
    c["rubrik"]["mindest_score"] = 0.0
    return c


@pytest.fixture
def leeres_repo(tmp_path):
    r = tmp_path / "repo"
    (r / "INSTITUT").mkdir(parents=True)
    return r
