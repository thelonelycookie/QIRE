"""Konsistenz der Agenda (INSTITUT/agenda.json) -- nur strukturelle Pruefungen, kein Rechnen."""
import json
from pathlib import Path

import pytest

WURZEL = Path(__file__).resolve().parent.parent
AGENDA = WURZEL / "INSTITUT" / "agenda.json"
STATUS = {"aktiv", "offen", "wartet", "erledigt", "verworfen_papier", "idee"}


@pytest.fixture(scope="module")
def punkte():
    return json.loads(AGENDA.read_text(encoding="utf-8"))["punkte"]


def test_ids_eindeutig(punkte):
    ids = [p["id"] for p in punkte]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_status_gueltig(punkte):
    schlecht = [(p["id"], p["status"]) for p in punkte if p["status"] not in STATUS]
    assert not schlecht, schlecht


def test_module_existieren(punkte):
    fehlt = []
    for p in punkte:
        if p.get("modul"):
            datei = WURZEL / "INSTITUT" / (p["modul"].replace(".", "/") + ".py")
            if not datei.exists():
                fehlt.append((p["id"], p["modul"]))
    assert not fehlt, fehlt


def test_folgen_verweisen_auf_existierende_punkte(punkte):
    ids = {p["id"] for p in punkte}
    # "folgen" ist meist {"Frage:Marke": [IDs]}; vereinzelt Freitext (str) -> nur Listen werden geprueft
    schlecht = [(p["id"], b, f) for p in punkte for b, fs in p.get("folgen", {}).items()
                if isinstance(fs, list) for f in fs if f not in ids]
    assert not schlecht, schlecht


def test_erledigte_punkte_haben_ergebnis(punkte):
    """Jeder erledigte Punkt mit Modul hat mindestens eine Ergebnisdatei <ID>__*.json (Rohdaten gesichert)."""
    fehlt = [p["id"] for p in punkte if p["status"] == "erledigt" and p.get("modul")
             and not list((WURZEL / "INSTITUT" / "ergebnisse").glob(f"{p['id']}__*.json"))]
    assert not fehlt, fehlt
