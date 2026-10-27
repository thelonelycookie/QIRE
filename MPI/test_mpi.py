# -*- coding: utf-8 -*-
"""Leichte Tests fuer MPI/: Skript-Syntax, Datenpruefung der QLoRA-Vorlage, starte.sh --trocken.
Aufruf:  python -m pytest MPI -q     (kein Cluster, kein torch noetig)"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MPI = Path(__file__).resolve().parent
sys.path.insert(0, str(MPI / "training"))
import qlora_train as Q  # noqa: E402

BASH = shutil.which("bash")


@pytest.mark.skipif(not BASH, reason="bash fehlt")
@pytest.mark.parametrize("skript", ["slurm/cpu_lauf.sbatch", "slurm/gpu_qlora.sbatch", "slurm/starte.sh", "uebertragen.sh"])
def test_bash_syntax(skript):
    r = subprocess.run([BASH, "-n", str(MPI / skript)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sbatch_kopf_hat_requeue_und_signal():
    for s in ("slurm/cpu_lauf.sbatch", "slurm/gpu_qlora.sbatch"):
        t = (MPI / s).read_text(encoding="utf-8")
        assert "#SBATCH --requeue" in t and "#SBATCH --signal=B:USR1@" in t and "#SBATCH --output=logs/" in t


def test_keine_zugangsdaten_im_text():
    import re
    for p in list(MPI.rglob("*.sh")) + list(MPI.rglob("*.sbatch")) + list(MPI.rglob("*.md")) + list(MPI.rglob("*.yml")) + list(MPI.rglob("*.beispiel")):
        t = p.read_text(encoding="utf-8")
        assert not re.search(r"BEGIN (RSA|OPENSSH|PRIVATE)|password\s*=|passwort\s*=\s*\S|ssh-rsa AAAA", t, re.I), p
    gi = (MPI / ".gitignore").read_text()
    assert "lokal.env" in gi and "*.key" in gi


def _daten(tmp_path, n=200):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": f"Idee {i}", "antwort": f"Note {i % 5}"}) for i in range(n)), encoding="utf-8")
    return p


def test_daten_split_deterministisch(tmp_path):
    d = Q.lade_jsonl(_daten(tmp_path))
    t1, v1 = Q.teile(d, 0.1)
    t2, v2 = Q.teile(list(reversed(d)), 0.1)
    assert {x["prompt"] for x in v1} == {x["prompt"] for x in v2}          # Reihenfolge egal
    assert 5 < len(v1) < 40 and len(t1) + len(v1) == 200
    assert not ({x["prompt"] for x in t1} & {x["prompt"] for x in v1})


def test_daten_fehler(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"prompt": "a"}\nkein json\n')
    with pytest.raises(ValueError) as e:
        Q.lade_jsonl(p)
    assert "Zeile 1" in str(e.value) and "Zeile 2" in str(e.value)


def test_daten_pruefen_cli(tmp_path, capsys):
    assert Q.main(["--daten-pruefen", "--daten", str(_daten(tmp_path))]) == 0
    assert "200 Beispiele" in capsys.readouterr().out


def test_letzter_checkpoint(tmp_path):
    for n in (100, 20, 300):
        (tmp_path / f"checkpoint-{n}").mkdir()
    (tmp_path / "checkpoint-abc").mkdir()
    assert Q.letzter_checkpoint(tmp_path).endswith("checkpoint-300")
    assert Q.letzter_checkpoint(tmp_path / "leer") is None


@pytest.mark.skipif(not BASH, reason="bash fehlt")
def test_starte_trocken(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([BASH, str(MPI / "slurm" / "starte.sh"), "cpu", "I-045-v3", "--schnell", "--kette", "2", "--trocken"],
                       capture_output=True, text=True, env=env, cwd=str(MPI.parent))
    assert r.returncode == 0, r.stderr
    zeilen = r.stdout.strip().splitlines()
    assert len(zeilen) == 2 and "cpu_lauf.sbatch I-045-v3 --schnell" in zeilen[0] and "--dependency=afterany:JOB1" in zeilen[1]
