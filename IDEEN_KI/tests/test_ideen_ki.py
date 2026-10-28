"""Tests der Ideen-KI -- nur mit AttrappenLLM, ohne GPU und Netz."""
import csv
import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from schmiede import filter as flt
from schmiede import karten as kart
from schmiede import operatoren as ops
from schmiede import pruefer as pf
from schmiede.betrieb import Betrieb, bewerte_waechter, lies_zeilen
from schmiede.llm import AttrappenLLM
from schmiede.rueckmeldung import Rueckmeldung

ORDNER = Path(__file__).resolve().parent.parent


def baue(cfg, tmp_path, repo, llm=None, **kw):
    return Betrieb(cfg, llm or AttrappenLLM(), ki_ordner=tmp_path / "ki", repo=repo, seed=1, **kw)


# ---------- Stopp-Zeit ----------
def test_stopp_zeit_unter_5_minuten(cfg, tmp_path, leeres_repo):
    """Dauerbetrieb mit trager Attrappe; STOP-Datei -> Ende deutlich unter 5 min (hier: wenige Sekunden)."""
    b = baue(cfg, tmp_path, leeres_repo, AttrappenLLM(verzoegerung=0.05))
    t = threading.Thread(target=b.run)
    t.start()
    time.sleep(1.0)
    assert t.is_alive() and len(b.ideen) > 0
    t0 = time.time()
    b.stop_datei.write_text("stop")
    t.join(timeout=60)
    assert not t.is_alive()
    assert time.time() - t0 < 5.0          # simuliert: Schritte dauern 0,05 s statt ~60 s
    status = json.loads(b.status_datei.read_text())
    assert status["laeuft"] is False
    assert b.checkpoint_pfad.exists()


def test_stopp_vor_jedem_llm_aufruf(cfg, tmp_path, leeres_repo):
    b = baue(cfg, tmp_path, leeres_repo)
    b.stop_datei.parent.mkdir(parents=True, exist_ok=True)
    b.stop_datei.write_text("x")
    b.run()
    assert b.llm.aufrufe == 0


def test_llm_timeout_obergrenze():
    from schmiede.llm import LlamaServerLLM
    l = LlamaServerLLM(timeout=1000)
    assert l.timeout <= 90


# ---------- Wiederaufnahme nach hartem Abbruch ----------
def test_wiederaufnahme_nach_hartem_abbruch(cfg, tmp_path, leeres_repo):
    import yaml
    ki = tmp_path / "ki"
    cfgpfad = tmp_path / "cfg.yaml"
    cfgpfad.write_text(yaml.safe_dump(cfg))
    skript = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ORDNER)!r})
        from pathlib import Path
        from schmiede.betrieb import Betrieb, lade_config
        from schmiede.llm import AttrappenLLM
        class Hart(AttrappenLLM):
            def chat(self, *a, **k):
                if self.aufrufe >= 9:
                    os._exit(3)          # harter Abbruch ohne finally/Abschluss
                return super().chat(*a, **k)
        Betrieb(lade_config(Path({str(cfgpfad)!r})), Hart(), ki_ordner=Path({str(ki)!r}),
                repo=Path({str(leeres_repo)!r}), seed=1).run()
    """)
    r = subprocess.run([sys.executable, "-c", skript], timeout=120)
    assert r.returncode == 3
    gesichert = {z["id"]: z for z in lies_zeilen(ki / "daten" / "ideen.jsonl") if z.get("text")}
    assert len(gesichert) >= 2
    # Abgeschnittene Schlusszeile simulieren (Strom weg mitten im Schreiben)
    with open(ki / "daten" / "ideen.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "KI-I999999", "text": "halb')
    b = Betrieb(cfg, AttrappenLLM(), ki_ordner=ki, repo=leeres_repo, seed=2, max_ideen=3)
    for i, z in gesichert.items():
        assert b.ideen[i]["text"] == z["text"]            # keine gespeicherte Idee verloren
    assert "KI-I999999" not in b.ideen
    offen = b.offene_idee_ids()
    b.run()
    for i in offen:                                        # angefangene Ideen wurden fortgesetzt
        assert b.ideen[i]["stufe"] not in ("roh", "gefiltert", "geprueft")
    neue = set(b.ideen) - set(gesichert)
    assert all(int(n.split("-I")[1]) > max(int(g.split("-I")[1]) for g in gesichert) for n in neue)  # keine ID-Kollision


def test_backup_rotation_und_status(cfg, tmp_path, leeres_repo):
    cfg["betrieb"]["backup_anzahl"] = 3
    b = baue(cfg, tmp_path, leeres_repo, max_ideen=4)
    for k in range(6):
        b.schreibe_idee({"id": f"KI-I{k:06d}", "text": "x" * 20, "t": time.time()})
        p = b.backup_wenn_faellig(erzwingen=True)
        if p is None:
            time.sleep(1.1)
    time.sleep(1.1)
    b.backup_wenn_faellig(erzwingen=True)
    assert 1 <= len(list(b.backup_ordner.glob("backup_*.zip"))) <= 3
    b.schreibe_status()
    s = json.loads(b.status_datei.read_text())
    assert {"zeit", "ideen_pro_stunde", "fehler", "letzte_aktivitaet_vor_s"} <= set(s)


# ---------- Duplikatfilter ----------
def test_duplikatfilter_tfidf():
    idx = flt.AehnlichkeitsIndex(None)
    idx.hinzufuegen("a", "Teste, ob die Magie-Konzentration bei Ziffern sinkt, wenn Winkel zufaellig eingefroren werden.")
    idx.hinzufuegen("b", "Kreuze den Quanten-Zeno-Effekt mit dem Immunsystem fuer adaptives Pruning von Gewichten.")
    dup, s, nah = idx.ist_duplikat("Teste ob die Magie-Konzentration bei Ziffern sinkt wenn Winkel zufaellig eingefroren werden!")
    assert dup and nah == "a"
    dup2, s2, _ = idx.ist_duplikat("Ein voellig anderer Gedanke ueber Musik, Harmonielehre und Verkehrsfluss in Staedten.")
    assert not dup2 and s2 < s


def test_duplikatfilter_embedding():
    idx = flt.AehnlichkeitsIndex(AttrappenLLM(embeddings=True), schwelle_embedding=0.9)
    idx.hinzufuegen("a", "Winkel einfrieren senkt Magie bei Ziffern")
    assert idx.ist_duplikat("Winkel einfrieren senkt Magie bei Ziffern")[0]
    assert not idx.ist_duplikat("Oekologie Nahrungsnetze Pruning Gewichte")[0]
    assert idx.naechster("Winkel einfrieren senkt Magie bei Ziffern")[2] == "embedding"


def test_dubletten_im_betrieb(cfg, tmp_path, leeres_repo):
    b = baue(cfg, tmp_path, leeres_repo)
    b.initialisiere_index()
    t = "Teste ob Winkelmuster 5 die Magie-Konzentration bei Ziffern senkt mit Zufallswinkel als Kontrolle."
    for k in (1, 2):
        b.schreibe_idee({"id": f"KI-I00000{k}", "t": time.time(), "kanal": "Neuland", "operator": "axiom",
                         "meta": {}, "text": t, "stufe": "roh"})
        b.verarbeite(b.ideen[f"KI-I00000{k}"])
    assert b.ideen["KI-I000001"]["stufe"] in ("bewertet", "unbewertet", "physik_abgelehnt")
    assert b.ideen["KI-I000002"]["stufe"] == "duplikat"


def test_agenda_ideen_json_im_duplikatindex(cfg, tmp_path, leeres_repo):
    (leeres_repo / "INSTITUT" / "ideen.json").write_text(json.dumps(
        {"karten": [{"id": "I-001", "titel": "Magie-Pruning nach Verlustanstieg der Winkel", "frage": "x"}]}))
    b = baue(cfg, tmp_path, leeres_repo)
    b.initialisiere_index()
    assert b.index.ist_duplikat("Magie-Pruning nach Verlustanstieg der Winkel x")[0]


# ---------- Lueckenkarte / Operatoren / Bandit ----------
def test_konzeptlisten_und_operatoren():
    qm, fremd = ops.lade_liste("konzepte_qm.txt"), ops.lade_liste("konzepte_fremd.txt")
    assert len(qm) >= 40 and len(fremd) >= 40
    rng = random.Random(0)
    lk = flt.Lueckenkarte(qm, fremd)
    for name, o in ops.OPERATOREN.items():
        e = ops.Eingabe(o.kanaele[0], "ctx", rng, lk, qm, fremd, ["Befund A", "Befund B"])
        prompt, meta = ops.erzeuge_prompt(name, e)
        assert "Aufgabe" in prompt
    x, y = lk.ziehe_paar(rng)
    lk.zaehle(x, y, 5)
    assert lk.bonus(x, y) < 1.0


def test_banditen_update(tmp_path):
    b = ops.Bandit(tmp_path / "b.json")
    k = ops.KANAL_NEULAND
    b.aktualisiere(k, "axiom", 1.0)
    b.aktualisiere(k, "axiom", 1.0)
    b.aktualisiere(k, "kreuzung", 0.0)
    assert b.mittel(k, "axiom") > 0.5 > b.mittel(k, "kreuzung")
    rng = random.Random(0)
    for _ in range(40):
        b.aktualisiere(k, "axiom", 1.0)
        b.aktualisiere(k, "luecke", 0.0)
    wahl = [b.waehle(k, rng) for _ in range(200)]
    assert wahl.count("axiom") > 100                      # erfolgreicher Operator wird oefter gezogen
    assert ops.Bandit(tmp_path / "b.json").mittel(k, "axiom") == b.mittel(k, "axiom")   # persistiert


# ---------- JSON-Parsen ----------
@pytest.mark.parametrize("text", [
    '{"neuheit": 3, "pruefbarkeit": 4, "erkenntniswert": 2, "ehrlichkeit": 5, "quantenstufe": "Q2"}',
    'Hier ist die Bewertung:\n```json\n{"neuheit": 3, "pruefbarkeit": 4, "erkenntniswert": 2, "ehrlichkeit": 5, "quantenstufe": "Q2",}\n```\nViel Erfolg!',
    "<think>hmm {kein json}</think>{'neuheit': '3', 'pruefbarkeit': 4.2, 'erkenntniswert': 2, 'ehrlichkeit': 5, 'quantenstufe': 'q2'}",
    '{"neuheit": 3, "pruefbarkeit": 4, "erkenntniswert": 2, "ehrlichkeit": 5, "quantenstufe": "Q2"',
])
def test_json_parsen_robust(text):
    r = pf.normalisiere_rubrik(pf.parse_json(text))
    assert r and r["neuheit"] == 3 and r["pruefbarkeit"] in (4,) and r["quantenstufe"] == "Q2"


def test_json_parsen_muell():
    assert pf.parse_json("") is None
    assert pf.parse_json("kein json hier") is None
    assert pf.normalisiere_rubrik(pf.parse_json('{"neuheit": 3}')) is None
    assert pf.normalisiere_rubrik({"neuheit": 9, "pruefbarkeit": 0, "erkenntniswert": 3, "ehrlichkeit": 3})["neuheit"] == 5


def test_rubrik_neuland_gewichtet_neuheit(cfg):
    r = {"neuheit": 5, "pruefbarkeit": 1, "erkenntniswert": 1, "ehrlichkeit": 1, "quantenstufe": "Q1"}
    assert pf.score(r, "Neuland", cfg["rubrik"]) > pf.score(r, "QIRE-nah", cfg["rubrik"])


def test_physik_pruefer():
    ok, v = pf.vorfilter("Ein Signal schneller als Licht ueber Verschraenkung senden.")
    assert not ok and v
    ok2, _ = pf.vorfilter("Gedankenexperiment: ein Signal schneller als Licht ueber Verschraenkung.")
    assert ok2
    assert pf.vorfilter("Teste Magie-Pruning auf Ziffern.")[0]
    res = pf.pruefe_physik(AttrappenLLM(), "Teste Magie-Pruning auf Ziffern mit Kontrolle.")
    assert res["ok"]


# ---------- Waechter ----------
def test_waechter_entscheidungen(cfg):
    w = cfg["waechter"]
    assert bewerte_waechter(w, {"temp": 60, "vram_anteil": 0.5}, {"ram_anteil": 0.5})[0] == "ok"
    assert bewerte_waechter(w, {"temp": 80, "vram_anteil": 0.5}, None)[0] == "drosseln"
    assert bewerte_waechter(w, {"temp": 90, "vram_anteil": 0.5}, None)[0] == "pausieren"
    assert bewerte_waechter(w, None, {"ram_anteil": 0.97})[0] == "pausieren"
    assert bewerte_waechter(w, None, None)[0] == "ok"


def test_pause_laesst_sich_stoppen(cfg, tmp_path, leeres_repo):
    b = baue(cfg, tmp_path, leeres_repo, waechter_quelle=lambda: ({"temp": 95, "vram_anteil": 0.1}, None))
    t = threading.Thread(target=b.run)
    t.start()
    time.sleep(0.5)
    b.stop_datei.write_text("x")
    t.join(10)
    assert not t.is_alive()


# ---------- Karten ----------
def test_kartenformat(cfg, tmp_path, leeres_repo):
    b = baue(cfg, tmp_path, leeres_repo, max_ideen=20)
    b.run()
    dateien = list((b.ausgabe).glob("karten_*.json"))
    assert dateien and (b.ausgabe / "TOP.md").exists()
    karten = json.loads(dateien[0].read_text(encoding="utf-8"))["karten"]
    assert len(karten) >= 1
    for k in karten:
        assert kart.validiere_karte(k) == []
        for f in ("hypothese", "kontrolle", "erwarteter_effekt", "abbruch", "neuheitsbegruendung", "kanal", "operator"):
            assert k[f]
        assert set(kart.ZIEL_FELDER) <= set(k)       # kompatibel zum Format von INSTITUT/ideen.json
    assert (b.ausgabe / "blindnoten.csv").exists()
    n_bew = sum(1 for r in b.ideen.values() if r.get("stufe") in ("bewertet", "ausgearbeitet"))
    assert len(karten) <= max(2, int(0.2 * n_bew) + 2)   # nur die besten ~10 %


def test_karte_validierung_findet_luecken():
    assert kart.validiere_karte({"id": "x"})


# ---------- Rueckmeldung ----------
def test_rueckmeldung_urteile_und_noten(tmp_path):
    daten, ausgabe = tmp_path / "daten", tmp_path / "ausgabe"
    ausgabe.mkdir()
    karte = {"id": "KI-20261014-001", "frage": "Idee X", "kanal": "Neuland", "operator": "axiom", "rubrik": {}}
    (ausgabe / "karten_20261014.json").write_text(json.dumps({"karten": [karte]}))
    agenda = tmp_path / "agenda.json"
    erg = tmp_path / "ergebnisse"
    erg.mkdir()
    agenda.write_text(json.dumps({"punkte": [{"id": "I-100", "quelle_ideen_ki": "KI-20261014-001", "status": "erledigt"}]}))
    (erg / "I-100__20261015_000000.json").write_text(json.dumps({"urteil": {"A-1": "H1: Effekt", "A-2": "H0: nix"}}))
    bandit = ops.Bandit(tmp_path / "b.json")
    r = Rueckmeldung(daten, ausgabe, bandit)
    vor = bandit.mittel("Neuland", "axiom")
    assert r.verarbeite_urteile(agenda, erg) == 1
    assert bandit.mittel("Neuland", "axiom") > vor
    assert r.verarbeite_urteile(agenda, erg) == 0          # nicht doppelt zaehlen
    pfad = r.erzeuge_vorlage()
    rows = list(csv.reader(open(pfad, encoding="utf-8"), delimiter=";"))
    assert rows[0][0] == "karten_id" and rows[1][0] == karte["id"]
    rows[1][2] = "4,5"
    csv.writer(open(pfad, "w", encoding="utf-8", newline=""), delimiter=";").writerows(rows)
    assert r.lies_blindnoten() == 1
    assert "Idee X" in r.fewshot("Neuland")


# ---------- Kontext ----------
def test_kontext_rag(tmp_path):
    from schmiede import kontext as kx
    repo = tmp_path / "repo"
    (repo / "INSTITUT" / "recherche").mkdir(parents=True)
    (repo / "INSTITUT" / "UEBERGABE.md").write_text(
        "## 4. Die Hauptlinie\n\n| Befund | Stand | Zahl |\n|---|---|---|\n| Magie-Konzentration in Winkeln | H1 | G(4)=0,7 |\n\n## 5. Anderes\ntext\n")
    (repo / "INSTITUT" / "FEHLERBUCH.md").write_text("## Fehler 1\nArm erbt Kopf, unfair gegen Kontrollen.\n\n## Fehler 2\nBLAS-Threads nicht auf 1.\n")
    (repo / "INSTITUT" / "recherche" / "a.md").write_text("# Notiz\nZeno-Effekt und Quantenwanderung.\n" * 5)
    idx = kx.baue_index(repo, tmp_path / "ki", tmp_path / "cache.json")
    treffer = idx.suche("Magie Konzentration Winkel", token_budget=200)
    assert treffer and "Magie" in treffer[0].text
    assert sum(kx.schaetze_token(a.text) + 8 for a in treffer) <= 200
    idx2 = kx.baue_index(repo, tmp_path / "ki", tmp_path / "cache.json")      # aus Cache
    assert len(idx2.abschnitte) == len(idx.abschnitte)
