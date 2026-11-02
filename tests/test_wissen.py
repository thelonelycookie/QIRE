"""Tests für WISSEN/werkzeuge/pruefen.py (nur tmp-Dateien)."""
import copy, importlib.util, json
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "WISSEN" / "werkzeuge" / "pruefen.py"
_s = importlib.util.spec_from_file_location("pruefen", _p)
pr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pr)

OK = {"id": "arXiv:2401.12345", "titel": "Ein Titel", "jahr": 2024, "autoren": "A. B.",
      "feld": ["K3"], "kernaussage": "x", "zahlen": [{"wert": "1", "kontext": "k", "geprueft": True}],
      "baseline_fuer_qire": None, "quantenbezug": "analogie", "bezug_qire_befund": ["I-038"],
      "stuetzt_widerspricht": "stuetzt", "volltext_gelesen": True, "gefunden_am": "2026-11-02",
      "welle": 1, "herkunft": "test"}


def eintrag(**kw):
    e = copy.deepcopy(OK)
    e.update(kw)
    return e


def test_gueltig():
    assert pr.pruefe_alles([OK]) == []


def test_fehlendes_feld():
    e = copy.deepcopy(OK)
    del e["kernaussage"]
    assert any("kernaussage" in x for x in pr.pruefe_alles([e]))


def test_falsche_feld_id():
    assert any("K9" in x for x in pr.pruefe_alles([eintrag(feld=["K9"])]))


def test_duplikat_id_und_titel():
    f = pr.pruefe_alles([OK, eintrag()])
    assert any("Duplikat-id" in x for x in f)
    f = pr.pruefe_alles([OK, eintrag(id="arXiv:2402.00001", titel="EIN  titel!")])
    assert any("Titel" in x for x in f)


def test_saettigung():
    es = [eintrag(id=f"arXiv:2401.{i:05d}", titel=f"T{i}x", welle=1) for i in range(9)]
    es.append(eintrag(id="arXiv:2401.99999", titel="Neu", welle=2))
    neu, ges, an = pr.saettigung(es, 2)["K3"]
    assert (neu, ges) == (1, 10) and abs(an - 0.1) < 1e-9
    es.append(eintrag(id="arXiv:2401.99998", titel="Neu2", welle=1))
    assert pr.saettigung(es, 2)["K3"][2] < 0.10


def test_cli_exit_und_index(tmp_path):
    q = tmp_path / "q.json"
    idx = tmp_path / "INDEX.md"
    q.write_text(json.dumps([OK]))
    assert pr.main(["--quellen", str(q), "--index", "--index-pfad", str(idx), "--saettigung", "1"]) == 0
    assert "| K3 |" in idx.read_text()
    q.write_text(json.dumps([eintrag(feld=["X"])]))
    assert pr.main(["--quellen", str(q)]) == 1
