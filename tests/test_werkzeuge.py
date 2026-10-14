"""Tests fuer werkzeuge/: Seed-Pruefung, FEHLERBUCH-Lint, Zahlen-Audit, Vorregistrierungs-Hook.

Jedes Werkzeug muss feuern KOENNEN (Positivkontrolle): synthetische Fehler werden erkannt,
der echte Repo-Stand ist sauber.
"""
import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import fehlerbuch_lint as L
import seeds_pruefen as SP
import zahlen_audit as ZA

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL / "werkzeuge" / "hooks"))
import vorregistrierung_pruefen as VP  # noqa: E402


# ------------------------------------------------------------------ Seed-Register
@pytest.fixture(scope="module")
def register():
    return json.loads((WURZEL / "INSTITUT" / "seeds.json").read_text(encoding="utf-8"))


def test_seed_register_repo_stand_ohne_fehler():
    assert SP.main(["--still"]) == 0


def test_seed_register_altlasten_werden_ausgegeben_nicht_verschwiegen(capsys):
    SP.main([])
    aus = capsys.readouterr().out
    assert "Altlasten" in aus and "I-015-hybrid + I-024" in aus and "rauchtest_auf_messseeds" in aus


def test_seed_doppelnutzung_wird_erkannt(register):
    r = copy.deepcopy(register)
    r["eintraege"].append({"id": "X-neu", "datensatz": ["mnist"], "zweck": "messung", "seeds": [[2020, 2021]],
                           "modul": "x", "notiz": ""})
    fehler, _, _ = SP.pruefe_register(r)
    assert any("Doppelnutzung" in f and "2020" in f for f in fehler)


def test_seed_anderer_datensatz_ist_nur_hinweis(register):
    r = copy.deepcopy(register)
    r["eintraege"].append({"id": "X-neu", "datensatz": ["cifar10"], "zweck": "messung", "seeds": [[2020, 2020]],
                           "modul": "x", "notiz": ""})
    fehler, hinweise, _ = SP.pruefe_register(r)
    assert not any("X-neu" in f for f in fehler)
    assert any("X-neu" in h for h in hinweise)


def test_rauchtest_auf_messseed_wird_erkannt(register):
    r = copy.deepcopy(register)
    r["eintraege"].append({"id": "X-neu", "datensatz": ["mnist"], "zweck": "rauchtest", "seeds": [[2000, 2000]],
                           "modul": "x", "notiz": ""})
    fehler, _, _ = SP.pruefe_register(r)
    assert any("Rauchtest X-neu nutzt Mess-Seed 2000" in f for f in fehler)


def test_blockregel_messung_in_erkundungsblock(register):
    r = copy.deepcopy(register)
    r["eintraege"].append({"id": "X-neu", "datensatz": ["mnist"], "zweck": "messung", "seeds": [[950, 951]],
                           "modul": "x", "notiz": ""})
    fehler, _, _ = SP.pruefe_register(r)
    assert any("Blockregel" in f and "X-neu" in f for f in fehler)


def test_neu_pruefung_frei_und_belegt(register, capsys):
    assert SP.main(["--neu", "2500", "2519"]) == 0
    assert SP.main(["--neu", "2000", "2005"]) == 1
    assert SP.main(["--neu", "2490", "2492", "--tuning"]) == 0
    assert SP.main(["--neu", "2492", "2495"]) == 1            # Messung auf Tuning-Endziffern


def test_jeder_seed_in_ergebnissen_ist_registriert():
    fehler, _, n = SP.pruefe_ergebnisse(json.loads((WURZEL / "INSTITUT" / "seeds.json").read_text(encoding="utf-8")))
    assert n > 30 and not fehler, fehler


# ------------------------------------------------------------------ FEHLERBUCH-Lint
def lint(tmp_path, code, name="abteilungen_x.py", mess=None):
    d = tmp_path / "abteilungen"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text(textwrap.dedent(code), encoding="utf-8")
    return {k for _, k, _ in L.pruefe_datei(f, mess or {})}


def test_lint_blas_threads(tmp_path):
    schlecht = """
        import multiprocessing as mp
        def lauf(seeds): 
            with mp.get_context("spawn").Pool(4) as p: return p.map(abs, seeds)
        def vortest(): return True
    """
    assert "blas-threads" in lint(tmp_path, schlecht)
    gut = schlecht.replace("def lauf", 'import os\nos.environ["OMP_NUM_THREADS"] = "1"\nos.environ["OPENBLAS_NUM_THREADS"] = "1"\n'
                           'os.environ["MKL_NUM_THREADS"] = "1"\nimport torch\ntorch.set_num_threads(1)\ndef lauf')
    assert "blas-threads" not in lint(tmp_path, gut)


def test_lint_ratensuche(tmp_path):
    schlecht = """
        TUNING_SEEDS = (900, 901)
        GITTER = (0.02, 0.08)
        def lauf(): 
            best = max(GITTER, key=lambda lr: 1)   # beste lr nach Mittel
        def vortest(): return True
    """
    assert "ratensuche" in lint(tmp_path, schlecht)
    assert "ratensuche" not in lint(tmp_path, schlecht + "\nfrom labor.gemeinsam import gittersuche_stabil\n")
    assert "ratensuche" not in lint(tmp_path, schlecht + "\n# Begruendung: einfache Suche aus Zeitgruenden, Ausreisser-Pruefung vor dem Urteil\n")


def test_lint_urteil_kein_string(tmp_path):
    schlecht = """
        def lauf():
            urteil = {"A-1": 0.3 > 0.2, "A-2": "H1: ok"}
            return {"urteil": urteil}
        def vortest(): return True
    """
    assert "urteil-kein-string" in lint(tmp_path, schlecht)
    assert "urteil-kein-string" not in lint(tmp_path, schlecht.replace("0.3 > 0.2", "'H1: ok'"))


def test_lint_arm_erbt_kopf(tmp_path):
    schlecht = """
        import copy
        def lauf(): 
            kopf = copy.deepcopy(voll.kopf)
        def vortest(): return True
    """
    assert "arm-erbt-kopf" in lint(tmp_path, schlecht)
    assert "arm-erbt-kopf" not in lint(tmp_path, schlecht + "\n# frischer Kopf je Arm\n")


def test_lint_wiederaufnahme_und_vortest(tmp_path):
    k = lint(tmp_path, """
        def lauf(seeds=range(3)):
            for seed in seeds:
                pass
    """)
    assert "keine-wiederaufnahme" in k and "kein-vortest" in k
    k2 = lint(tmp_path, """
        def lauf(seeds=range(3)):
            zwischen = {}
            for seed in seeds:
                pass
        def vortest(): return True
    """)
    assert "keine-wiederaufnahme" not in k2 and "kein-vortest" not in k2


def test_lint_kontrolle_ohne_matrixcheck(tmp_path):
    schlecht = """
        ARME = ("hadamard_fest", "voll")
        def lauf(): pass
        def vortest(): return True
    """
    assert "kontrolle-ohne-matrixcheck" in lint(tmp_path, schlecht)
    gut = schlecht.replace("return True", "import numpy as np\n    return np.count_nonzero(np.eye(2)) == 2")
    assert "kontrolle-ohne-matrixcheck" not in lint(tmp_path, gut)


def test_lint_rauchtest_auf_messseeds(tmp_path):
    code = """
        SEEDS = tuple(range(100, 110))
        def lauf(schnell=False, seeds=SEEDS):
            if schnell:
                seeds = SEEDS[:2]
        def vortest(): return True
    """
    assert "rauchtest-messseed" in lint(tmp_path, code, mess={100: {"I-024"}, 101: {"I-024"}})
    assert "rauchtest-messseed" not in lint(tmp_path, code, mess={})
    entwurf = code.replace("seeds = SEEDS[:2]", "seeds = (990, 991)")
    assert "rauchtest-messseed" not in lint(tmp_path, entwurf, mess={100: {"I-024"}})


def test_lint_doppelte_definition_und_ausnahmekommentar(tmp_path):
    code = "def vortest(): return True\ndef f(): pass\ndef f(): pass\n"
    assert "doppelte-definition" in lint(tmp_path, code)
    assert "doppelte-definition" not in lint(tmp_path, code + "# lint: ok doppelte-definition Spleiss bekannt\n")


def test_lint_repo_neue_module_sind_sauber():
    assert L.main(["--streng", "--zusammenfassung"]) == 0


def test_lint_findet_bekannte_altlast_mnist100_doppelt():
    ws = L.pruefe_datei(WURZEL / "INSTITUT" / "abteilungen" / "mnist100.py", {})
    assert any(k == "doppelte-definition" for _, k, _ in ws)


# ------------------------------------------------------------------ Zahlen-Audit
def test_audit_bestaetigt_und_erkennt_abweichung(tmp_path):
    md = tmp_path / "LB.md"
    md.write_text(textwrap.dedent("""\
        ## 24.09.2026, 16:30 — C-clifford-v2 (frische Seeds 10–19)

        | Arm | Genauigkeit |
        |---|---|
        | voll | 0,9683 ± 0,0101 |
        | ohne_schaltung | 0,9999 ± 0,0062 |
        | clifford_k4 | 0,9642 ± 0,0101 |
        """), encoding="utf-8")
    zellen = ZA.pruefe_datei(md)
    st = {z["label"]: z["status"] for z in zellen if z["zahl"] in ("0,9683", "0,9999", "0,9642")}
    assert st["voll"] == "bestaetigt (spezifisch)"
    assert st["ohne_schaltung"] == "abweichend"
    assert st["clifford_k4"] == "bestaetigt (spezifisch)"


def test_audit_ohne_abschnitts_id_ist_nicht_zuordenbar(tmp_path):
    md = tmp_path / "LB.md"
    md.write_text("## Irgendwas\n\n| Arm | Wert |\n|---|---|\n| voll | 0,9683 ± 0,0101 |\n", encoding="utf-8")
    assert {z["status"] for z in ZA.pruefe_datei(md)} == {"nicht zuordenbar"}


def test_audit_laborbuch_hat_keine_abweichung_und_viele_treffer():
    zellen = ZA.pruefe_datei(WURZEL / "INSTITUT" / "LABORBUCH.md")
    abw = [z for z in zellen if z["status"] == "abweichend"]
    best = [z for z in zellen if z["status"].startswith("bestaetigt")]
    assert not abw, [(z["zeile"], z["label"], z["zahl"], z["info"]) for z in abw]
    assert len(best) > 100


# ------------------------------------------------------------------ Hook (Logik mit Attrappe)
class Attrappe:
    def __init__(self, head=None, index=None, geaendert=(), hat_head=True):
        self._head, self._index, self._geaendert, self._hat = head or {}, index or {}, list(geaendert), hat_head

    def head(self, p):
        return self._head.get(p)

    def index(self, p):
        return self._index.get(p)

    def geaendert(self):
        return self._geaendert

    def hat_head(self):
        return self._hat


AGENDA = json.dumps({"punkte": [{"id": "I-900", "modul": "abteilungen.neu", "status": "offen"}]})
SEEDS = json.dumps({"eintraege": [
    {"id": "I-900", "zweck": "messung", "seeds": [[3000, 3009]], "datensatz": ["mnist"], "modul": "neu", "notiz": ""},
    {"id": "I-001", "zweck": "messung", "seeds": [[30, 39]], "datensatz": ["ziffern"], "modul": "m", "notiz": ""}],
    "ausnahmen": []})
ERG = "INSTITUT/ergebnisse/I-900__20261201_120000.json"
MOD = "INSTITUT/abteilungen/neu.py"


def hook(head, index, geaendert):
    return VP.pruefe(Attrappe(head, index, geaendert), geaendert)


def test_hook_laesst_vorregistrierten_lauf_durch():
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS, MOD: "x"}
    f, _ = hook(head, {ERG: json.dumps({"seeds": list(range(3000, 3010))})}, [("A", ERG)])
    assert f == []


def test_hook_blockiert_wenn_modul_nicht_frueher_committet():
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS}
    f, _ = hook(head, {ERG: json.dumps({"seeds": [3000]})}, [("A", ERG), ("A", MOD)])
    assert any("nicht in einem fruehren Commit" in x for x in f)


def test_hook_blockiert_wenn_modul_im_selben_commit_geaendert_wird():
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS, MOD: "alt"}
    f, _ = hook(head, {ERG: json.dumps({"seeds": [3000]})}, [("A", ERG), ("M", MOD)])
    assert any("SELBEN Commit" in x for x in f)


def test_hook_blockiert_nicht_registrierte_seeds():
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS, MOD: "x"}
    f, _ = hook(head, {ERG: json.dumps({"seeds": [3500]})}, [("A", ERG)])
    assert any("nicht als messung" in x for x in f)


def test_hook_blockiert_rauchtest_auf_messseeds():
    pfad = "INSTITUT/ergebnisse/I-900-schnell__20261201_120000.json"
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS}
    f, _ = hook(head, {pfad: json.dumps({"seeds": [3000, 3001]})}, [("A", pfad)])
    assert any("Rauchtest benutzt Mess-Seeds" in x for x in f)
    ok, _ = hook(head, {pfad: json.dumps({"seeds": [3090]})}, [("A", pfad)])
    assert ok == []


def test_hook_ignoriert_zwischendateien_und_erste_commits():
    z = "INSTITUT/ergebnisse/_zwischen_I-900.json"
    assert hook({}, {z: "{}"}, [("A", z)])[0] == []
    f, w = VP.pruefe(Attrappe(hat_head=False), [("A", ERG)])
    assert f == [] and w


def test_hook_blockiert_neues_modul_mit_rauchtest_auf_messseeds():
    code = "SEEDS = tuple(range(30, 40))\ndef lauf(schnell=False):\n    if schnell:\n        seeds = SEEDS[:2]\ndef vortest(): return True\n"
    pfad = "INSTITUT/abteilungen/neu2.py"
    head = {"INSTITUT/agenda.json": AGENDA, "INSTITUT/seeds.json": SEEDS}
    f, _ = hook(head, {pfad: code}, [("A", pfad)])
    assert any("rauchtest" in x.lower() and "Mess-Seeds" in x for x in f)


def test_hook_integration_echtes_git_repo(tmp_path):
    """Echter git commit mit core.hooksPath: Ergebnis ohne vorheriges Modul wird blockiert, --no-verify umgeht."""
    git = lambda *a, **k: subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, **k)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(WURZEL / "werkzeuge" / "hooks"))
    (tmp_path / "INSTITUT" / "ergebnisse").mkdir(parents=True)
    (tmp_path / "INSTITUT" / "agenda.json").write_text(AGENDA, encoding="utf-8")
    (tmp_path / "INSTITUT" / "seeds.json").write_text(SEEDS, encoding="utf-8")
    assert git("add", "-A").returncode == 0
    assert git("commit", "-q", "-m", "start", "--no-verify").returncode == 0
    (tmp_path / "INSTITUT" / "ergebnisse" / "I-900__20261201_120000.json").write_text(
        json.dumps({"seeds": [3000]}), encoding="utf-8")
    git("add", "-A")
    p = git("commit", "-q", "-m", "Rohdaten")
    assert p.returncode != 0 and "BLOCKIERT" in p.stderr and "--no-verify" in p.stderr
    p = git("commit", "-q", "-m", "Rohdaten (umgangen: Test)", "--no-verify")
    assert p.returncode == 0
