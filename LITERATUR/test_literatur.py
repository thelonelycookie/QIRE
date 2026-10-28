# -*- coding: utf-8 -*-
"""Konsistenz der Literaturdatenbank (R5.3).  Aufruf:  python -m pytest LITERATUR -q"""
import json
import re
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import erzeuge as E  # noqa: E402
import quellen_daten as Q  # noqa: E402


def test_keine_fehler_und_alle_repo_ids_abgedeckt():
    fehler, fehlend = E.pruefe(E.scanne_fundstellen())
    assert not fehler, fehler
    assert not fehlend, f"arXiv-IDs im Repo, aber nicht in quellen_daten.py: {fehlend}"


def test_index_verweise_und_pflichtfelder():
    d = json.loads((HIER / "neuheit_index.json").read_text(encoding="utf-8"))
    assert d["format"] == "qire-neuheit-index" and d["version"] == 1
    keys = set(d["arbeiten"])
    for b, v in d["begriffe"].items():
        assert set(v["arbeiten"]) <= keys, b
        assert v["status"] in {"bekannt", "teilweise_bekannt", "gegenbefund", "neue_verbindung", "nicht_gefunden", "offen", "ungeprueft"}
    for p in d["neuheitspruefungen"]:
        assert set(p["arbeiten"]) <= keys and set(p["begriffe"]) <= set(d["begriffe"])
    for k, a in d["arbeiten"].items():
        assert a["status"] in {"Volltext", "Abstract", "Auszug", "nur Titel"}, k
        assert isinstance(a["unvollstaendig"], list)


def test_erzeugte_dateien_sind_aktuell():
    """Wer quellen_daten.py aendert, muss `python LITERATUR/erzeuge.py` laufen lassen."""
    fund = E.scanne_fundstellen()
    idx = E.baue_index(fund)
    gespeichert = json.loads((HIER / "neuheit_index.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(idx, ensure_ascii=False)) == gespeichert
    bib = (HIER / "literatur.bib").read_text(encoding="utf-8")
    assert len(re.findall(r"^@", bib, re.M)) == len(Q.ALLE)


def test_nichts_erfunden_markierung():
    """Jeder Eintrag ohne Autoren/Jahr/ID muss im Index als unvollstaendig stehen."""
    d = json.loads((HIER / "neuheit_index.json").read_text(encoding="utf-8"))
    for k, a in d["arbeiten"].items():
        if not a["autoren"]:
            assert "Autoren" in a["unvollstaendig"], k
        if not a["arxiv"]:
            assert "arXiv-ID" in a["unvollstaendig"], k


def test_bib_klammern_balanciert():
    for e in re.split(r"\n(?=@)", (HIER / "literatur.bib").read_text(encoding="utf-8")):
        if e.startswith("@"):
            assert e.count("{") == e.count("}"), e[:50]
