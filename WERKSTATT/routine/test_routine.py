# -*- coding: utf-8 -*-
"""Tests fuer Routine-Worker und GPU-Sperre. Aufruf:  python -m pytest WERKSTATT/routine -q
Kein Netz ausser einem lokalen Attrappen-Server auf 127.0.0.1 (Zufallsport), kein echtes Modell."""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sperre as G  # noqa: E402
import routine as R  # noqa: E402


# ------------------------------------------------------------------ GPU-Sperre
def test_exklusiv_und_freigabe(tmp_path):
    d = tmp_path / "gpu.lock"
    a, b = G.GpuSperre("routine-worker", "x", d), G.GpuSperre("ideen-ki", "y", d)
    assert a.versuche()
    assert not b.versuche()
    a.freigeben()
    assert b.versuche()
    b.freigeben()
    assert not d.exists()


def test_fremde_sperre_wird_nicht_geloescht(tmp_path):
    d = tmp_path / "gpu.lock"
    a, b = G.GpuSperre("a", "", d), G.GpuSperre("b", "", d)
    assert a.versuche()
    b.freigeben()                      # b besitzt nichts -> darf a nicht entfernen
    assert d.exists()
    a.freigeben()


def test_veraltet_durch_toten_prozess(tmp_path):
    d = tmp_path / "gpu.lock"
    import platform
    d.write_text(json.dumps({"besitzer": "ideen-ki", "pid": 2 ** 22 + 12345, "host": platform.node(),
                             "start": time.time(), "herzschlag": time.time(), "token": "alt"}))
    assert G.lese_sperre(d)["veraltet"]
    assert G.GpuSperre("routine-worker", "", d).versuche()      # uebernimmt


def test_veraltet_durch_herzschlag_anderer_rechner(tmp_path):
    d = tmp_path / "gpu.lock"
    d.write_text(json.dumps({"besitzer": "x", "pid": os.getpid(), "host": "anderer-rechner",
                             "start": 1000.0, "herzschlag": 1000.0, "token": "t"}))
    assert G.lese_sperre(d, tot_nach_s=60)["veraltet"]
    frisch = json.dumps({"besitzer": "x", "pid": os.getpid(), "host": "anderer-rechner",
                         "start": time.time(), "herzschlag": time.time(), "token": "t"})
    d.write_text(frisch)
    assert not G.lese_sperre(d, tot_nach_s=60)["veraltet"]
    assert not G.GpuSperre("w", "", d).versuche()


def test_lebende_sperre_bleibt(tmp_path):
    d = tmp_path / "gpu.lock"
    a = G.GpuSperre("ideen-ki", "Runde", d)
    assert a.versuche()
    s = G.lese_sperre(d)
    assert s["besitzer"] == "ideen-ki" and not s["veraltet"]
    assert not G.GpuSperre("routine-worker", "", d).versuche()
    a.freigeben()


def test_anfrage_hat_vorrang_und_wird_aufgeraeumt(tmp_path):
    d = tmp_path / "gpu.lock"
    ki = G.GpuSperre("ideen-ki", "", d)
    assert ki.versuche()
    w = G.GpuSperre("routine-worker", "log", d)

    def gebe_frei_wenn_anfrage():       # so verhaelt sich die Ideen-KI (siehe GPU_SPERRE.md)
        for _ in range(100):
            if G.anfrage_offen(d):
                ki.freigeben_nur_sperre = True
                ki._gib_frei_ohne_anfrage()
                return
            time.sleep(0.05)
    t = threading.Thread(target=gebe_frei_wenn_anfrage)
    t.start()
    assert w.erwerbe(warte_max_s=5, pause_s=0.05)
    t.join()
    assert G.anfrage_offen(d) is not None         # waehrend wir halten, bleibt die Flagge stehen
    w.freigeben()
    assert G.anfrage_offen(d) is None and not d.exists()


def test_erwerbe_zeitgrenze(tmp_path):
    d = tmp_path / "gpu.lock"
    a = G.GpuSperre("ideen-ki", "", d)
    assert a.versuche()
    t0 = time.time()
    assert not G.GpuSperre("w", "", d).erwerbe(warte_max_s=0.3, pause_s=0.05)
    assert time.time() - t0 < 3
    assert G.anfrage_offen(d) is None             # Anfrage nach Misserfolg zurueckgenommen
    a.freigeben()


def test_vram_quittung(tmp_path):
    d = tmp_path / "gpu.lock"
    w = G.GpuSperre("routine-worker", "", d)
    assert not w.erwerbe(warte_max_s=0.3, vram_frei=True, pause_s=0.05)    # keine Quittung -> kein Zugriff
    assert not d.exists()
    (tmp_path / "gpu.vram_frei").write_text(json.dumps({"zeit": time.time(), "pid": 1}))
    assert w.erwerbe(warte_max_s=1, vram_frei=True, pause_s=0.05)
    w.freigeben()
    assert not (tmp_path / "gpu.vram_frei").exists()


def test_herzschlag(tmp_path):
    d = tmp_path / "gpu.lock"
    t = [1000.0]
    a = G.GpuSperre("w", "", d, uhr=lambda: t[0])
    assert a.versuche()
    t[0] = 1100.0
    a.herzschlag()
    assert json.loads(d.read_text())["herzschlag"] == 1100.0
    a.freigeben()


# ------------------------------------------------------------------ Attrappen-Server
class _H(BaseHTTPRequestHandler):
    antwort = "- Lauf fertig, 10 Seeds\n- keine Fehler"
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _H.letzte = json.loads(self.rfile.read(n))
        body = json.dumps({"choices": [{"message": {"content": _H.antwort}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    s = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=s.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


@pytest.fixture
def cfgdatei(tmp_path, server):
    c = R.lade_konfig()
    c["llm"]["url"] = server
    c["gpu_sperre"]["datei"] = str(tmp_path / "gpu.lock")
    c["gpu_sperre"]["warte_max_s"] = 1
    p = tmp_path / "konfig.json"
    p.write_text(json.dumps(c))
    return p


@pytest.fixture(autouse=True)
def ergebnisse(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ERGEBNISSE", tmp_path / "erg")
    return tmp_path / "erg"


def test_chat_openai_format(server):
    cfg = R.lade_konfig()
    assert R.chat(server, "hallo", cfg).startswith("- Lauf fertig")
    assert _H.letzte["messages"][1]["content"] == "hallo" and _H.letzte["messages"][0]["role"] == "system"


def test_chat_fehler():
    with pytest.raises(R.LLMFehler):
        R.chat("http://127.0.0.1:9", "x", R.lade_konfig())


def test_url_aus_ideen_ki_konfig():
    c = R.lade_konfig()
    c["llm"]["url"] = "auto"
    assert R.ermittle_llm_url(c).startswith("http://127.0.0.1:")


def test_ruhezeit():
    c = R.lade_konfig()
    assert R.in_ruhezeit(c, 0) and R.in_ruhezeit(c, 5) and not R.in_ruhezeit(c, 6) and not R.in_ruhezeit(c, 23)


def test_ruhezeit_blockiert_start(cfgdatei, capsys):
    assert R.main(["--konfig", str(cfgdatei), "bericht"], stunde=3) == 3
    assert "Ruhezeit" in capsys.readouterr().err


def test_zahlen_ohne_beleg():
    assert R.zahlen_ohne_beleg("10 Seeds, 99 Fehler, 3,5 s", "Lauf mit 10 Seeds und 3.5 s") == ["99"]


def test_log_notiz(cfgdatei, tmp_path, ergebnisse, capsys):
    log = tmp_path / "lauf.log"
    log.write_text("start\nSeed 1/10\nTraceback (most recent call last)\nValueError\nSeed 10/10\nfertig\n")
    assert R.main(["--konfig", str(cfgdatei), "--id", "t-log", "log", str(log)], stunde=12) == 0
    notiz = (ergebnisse / "t-log" / "notiz.md").read_text()
    assert "Zusammenfassung des lokalen Modells" in notiz and "Fehler/Traceback/OOM: 2" in notiz
    b = (ergebnisse / "t-log" / "bericht.md").read_text()
    assert "STATUS: erledigt" in b and "COMMIT: keiner" in b
    assert not (tmp_path / "gpu.lock").exists()           # Sperre wieder frei


def test_log_zahl_ohne_beleg_wird_gemeldet(cfgdatei, tmp_path, ergebnisse, monkeypatch):
    monkeypatch.setattr(_H, "antwort", "- 4242 Seeds gelaufen")
    log = tmp_path / "l.log"; log.write_text("Seed 1/10\n")
    R.main(["--konfig", str(cfgdatei), "--id", "t-z", "log", str(log)], stunde=12)
    assert "4242" in (ergebnisse / "t-z" / "notiz.md").read_text()
    assert "Zahlen ohne Beleg" in (ergebnisse / "t-z" / "bericht.md").read_text()


def test_ohne_modell_faellt_auf_rohbericht_zurueck(tmp_path, ergebnisse):
    c = R.lade_konfig()
    c["llm"]["url"] = "http://127.0.0.1:9"
    c["gpu_sperre"]["datei"] = str(tmp_path / "gpu.lock")
    p = tmp_path / "k.json"; p.write_text(json.dumps(c))
    log = tmp_path / "l.log"; log.write_text("Seed 1/10\n")
    assert R.main(["--konfig", str(p), "--id", "t-f", "log", str(log)], stunde=12) == 0
    b = (ergebnisse / "t-f" / "bericht.md").read_text()
    assert "STATUS: teilweise" in b and "nicht erreichbar" in b
    assert not (tmp_path / "gpu.lock").exists()


def test_gpu_belegt_gibt_rohbericht(cfgdatei, tmp_path, ergebnisse):
    ki = G.GpuSperre("ideen-ki", "Runde", tmp_path / "gpu.lock")
    assert ki.versuche()
    log = tmp_path / "l.log"; log.write_text("Seed 1/10\n")
    R.main(["--konfig", str(cfgdatei), "--id", "t-b", "log", str(log)], stunde=12)
    b = (ergebnisse / "t-b" / "bericht.md").read_text()
    assert "STATUS: teilweise" in b and "Sperre" in b and "ideen-ki" in b
    ki.freigeben()


def test_statusbericht(cfgdatei, tmp_path, ergebnisse, monkeypatch):
    inst = tmp_path / "INSTITUT"; (inst / "laeufe").mkdir(parents=True)
    (inst / "agenda.json").write_text(json.dumps({"punkte": [
        {"id": "I-001", "status": "erledigt"}, {"id": "I-002", "status": "offen", "wartet_auf": "malte", "notiz": "Freigabe?"}]}))
    jetzt = time.time()
    (inst / "laeufe" / "I-050.status.json").write_text(json.dumps({"id": "I-050", "zustand": "laeuft", "seed": 3, "gesamt": 10, "herzschlag": jetzt - 5000}))
    (inst / "laeufe" / "I-051.status.json").write_text(json.dumps({"id": "I-051", "zustand": "fertig", "herzschlag": jetzt}))
    monkeypatch.setattr(R, "INSTITUT", inst)
    assert R.main(["--konfig", str(cfgdatei), "--id", "t-s", "bericht"], stunde=12) == 0
    md = (ergebnisse / "t-s" / "status.md").read_text()
    assert "I-050 | tot | 3/10" in md and "I-002" in md
    k = json.loads((ergebnisse / "t-s" / "kennzahlen.json").read_text())
    assert k["tot"] == 1 and k["wartet_auf_malte"] == 1 and k["punkte"] == 2


def test_vortest_parser():
    e = R.parse_vortest("I-001 Vortest: bestanden\nI-002 Vortest: DURCHGEFALLEN\n")
    assert e == {"bestanden": ["I-001"], "durchgefallen": ["I-002"], "traceback": False}


def test_vortest_nicht_bei_aktivem_lauf(cfgdatei, tmp_path, monkeypatch, capsys):
    inst = tmp_path / "INSTITUT"; (inst / "laeufe").mkdir(parents=True)
    (inst / "laeufe" / "I-050.status.json").write_text(json.dumps({"id": "I-050", "zustand": "laeuft", "herzschlag": time.time()}))
    monkeypatch.setattr(R, "INSTITUT", inst)
    assert R.main(["--konfig", str(cfgdatei), "vortest"], stunde=12) == 5
    assert "Lauf aktiv" in capsys.readouterr().err


def test_vortest_lauf_mit_attrappen_direktor(cfgdatei, tmp_path, ergebnisse, monkeypatch):
    inst = tmp_path / "INSTITUT"; inst.mkdir()
    (inst / "direktor.py").write_text("print('I-001 Vortest: bestanden')\nprint('I-002 Vortest: DURCHGEFALLEN')\n")
    monkeypatch.setattr(R, "INSTITUT", inst)
    assert R.main(["--konfig", str(cfgdatei), "--id", "t-v", "vortest"], stunde=12) == 1     # durchgefallen -> Exit 1
    b = (ergebnisse / "t-v" / "bericht.md").read_text()
    assert "1 bestanden, 1 durchgefallen" in b and "STATUS: teilweise" in b
    assert "Einordnung des lokalen Modells" in (ergebnisse / "t-v" / "notiz.md").read_text()


def test_kein_git_im_code():
    code = (Path(R.__file__).read_text(encoding="utf-8") + Path(G.__file__).read_text(encoding="utf-8"))
    assert '"git"' not in code and "'git'" not in code
