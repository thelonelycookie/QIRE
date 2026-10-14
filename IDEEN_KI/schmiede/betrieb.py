"""Hauptschleife der Ideen-KI: Kontext -> erzeugen -> filtern -> pruefen/bewerten -> ausarbeiten.
Dauerbetrieb ohne Ruhezeit. Stopp ueber Datei STOP (nach jedem Schritt geprueft).
Jede Idee wird sofort als Zeile in daten/ideen.jsonl gesichert (append + flush + fsync).

Aufruf (im Ordner IDEEN_KI):  python -m schmiede.betrieb [--attrappe] [--max-ideen N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import filter as flt
from . import karten as kart
from . import kontext as kx
from . import operatoren as ops
from . import pruefer as pf
from .llm import LLM, AttrappenLLM, LLMFehler
from .rueckmeldung import Rueckmeldung

ORDNER = Path(__file__).resolve().parent.parent


class Stopp(Exception):
    """Stopp angefordert (Datei STOP) -> sauber beenden."""


def lade_config(pfad: Optional[Path] = None) -> dict:
    return yaml.safe_load(Path(pfad or ORDNER / "config.yaml").read_text(encoding="utf-8"))


# ---------------- Waechter ----------------
def gpu_status() -> Optional[dict]:
    """Temperatur/VRAM ueber nvidia-smi; None, wenn nicht verfuegbar."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        t, u, g = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"temp": t, "vram_anteil": u / g if g else 0.0}
    except Exception:
        return None


def ram_status() -> Optional[dict]:
    try:
        import psutil
        return {"ram_anteil": psutil.virtual_memory().percent / 100.0}
    except Exception:
        return None


def bewerte_waechter(cfg_w: dict, gpu: Optional[dict], ram: Optional[dict]) -> tuple[str, str]:
    """("ok"|"drosseln"|"pausieren", Grund)."""
    if gpu:
        if gpu["temp"] >= cfg_w["gpu_temp_pause_c"]:
            return "pausieren", f"GPU {gpu['temp']:.0f} Grad"
        if gpu["vram_anteil"] >= cfg_w["vram_anteil_pause"]:
            return "pausieren", f"VRAM {gpu['vram_anteil']:.0%}"
    if ram and ram["ram_anteil"] >= cfg_w["ram_anteil_pause"]:
        return "pausieren", f"RAM {ram['ram_anteil']:.0%}"
    if gpu and gpu["temp"] >= cfg_w["gpu_temp_drosseln_c"]:
        return "drosseln", f"GPU {gpu['temp']:.0f} Grad"
    if ram and ram["ram_anteil"] >= cfg_w["ram_anteil_drosseln"]:
        return "drosseln", f"RAM {ram['ram_anteil']:.0%}"
    return "ok", ""


# ---------------- Hilfen ----------------
def atomar_schreiben(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pfad)


def lies_zeilen(pfad: Path) -> list[dict]:
    """JSON-Lines lesen; eine abgebrochene letzte Zeile wird ueberlesen."""
    out = []
    if not pfad.exists():
        return out
    with open(pfad, encoding="utf-8", errors="replace") as f:
        for z in f:
            z = z.strip()
            if not z:
                continue
            try:
                out.append(json.loads(z))
            except Exception:
                continue
    return out


class Betrieb:
    def __init__(self, cfg: dict, llm: LLM, ki_ordner: Path = ORDNER, repo: Optional[Path] = None,
                 server=None, seed: Optional[int] = None, uhr: Callable[[], float] = time.time,
                 waechter_quelle: Optional[Callable[[], tuple]] = None,
                 max_ideen: Optional[int] = None):
        self.cfg, self.llm, self.server = cfg, llm, server
        self.ordner = Path(ki_ordner)
        p = cfg["pfade"]
        self.daten = self.ordner / p["daten"]
        self.ausgabe = self.ordner / p["ausgabe"]
        self.backup_ordner = self.ordner / p["backup"]
        self.stop_datei = self.ordner / p["stop_datei"]
        self.status_datei = self.ordner / p["status"]
        self.repo = Path(repo) if repo else (self.ordner / p["repo_wurzel"]).resolve()
        self.uhr = uhr
        self.waechter_quelle = waechter_quelle or (lambda: (gpu_status(), ram_status()))
        self.max_ideen = max_ideen
        self.rng = random.Random(seed)
        self.b = cfg["betrieb"]
        self.fehler: list[dict] = []
        self.fehler_in_folge = 0
        self.wachter_zustand = ("ok", "")
        self._letzter_waechter = 0.0
        self._letzter_herzschlag = 0.0
        self._letzte_aktivitaet = uhr()
        self.neue_ideen_dieser_lauf = 0
        self.laeuft = False
        # Zustand laden
        self.ideen_pfad = self.daten / "ideen.jsonl"
        self.checkpoint_pfad = self.daten / "checkpoint.json"
        self.ideen: dict[str, dict] = {}
        self.laden()
        self.bandit = ops.Bandit(self.daten / "bandit.json")
        self.qm = ops.lade_liste("konzepte_qm.txt")
        self.fremd = ops.lade_liste("konzepte_fremd.txt")
        self.luecken = flt.Lueckenkarte(self.qm, self.fremd, self.daten / "lueckenkarte.json")
        self.rueck = Rueckmeldung(self.daten, self.ausgabe, self.bandit)
        self.index = flt.AehnlichkeitsIndex(llm, self.b["duplikat_schwelle"], self.b["duplikat_schwelle_tfidf"])
        self.kontext_index: Optional[kx.KontextIndex] = None

    # ---------- Zustand ----------
    def laden(self) -> None:
        self.ideen = {}
        for z in lies_zeilen(self.ideen_pfad):
            if "id" in z:
                self.ideen.setdefault(z["id"], {}).update(z)
        cp = {}
        if self.checkpoint_pfad.exists():
            try:
                cp = json.loads(self.checkpoint_pfad.read_text(encoding="utf-8"))
            except Exception:
                cp = {}
        nr = max([int(i.split("-I")[-1]) for i in self.ideen if "-I" in i] + [0])
        self.cp = {"runde": 0, "in_runde": 0, "runde_ids": [], "kartennr": 0, **cp}
        self.cp["naechste_nr"] = max(int(cp.get("naechste_nr", 0)), nr + 1)

    def speichere_checkpoint(self) -> None:
        self.cp["zeit"] = self.uhr()
        atomar_schreiben(self.checkpoint_pfad, json.dumps(self.cp, ensure_ascii=False))

    def schreibe_idee(self, rec: dict) -> None:
        """Sofort sichern: append + flush + fsync. Updates tragen dieselbe id."""
        self.ideen_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ideen_pfad, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.ideen.setdefault(rec["id"], {}).update(rec)

    # ---------- Stopp / Takt ----------
    def stop_angefordert(self) -> bool:
        return self.stop_datei.exists()

    def pruefe_stopp(self) -> None:
        if self.stop_angefordert():
            raise Stopp()

    def schlafe(self, sek: float) -> None:
        ende = self.uhr() + sek
        while True:
            self.pruefe_stopp()
            rest = ende - self.uhr()
            if rest <= 0:
                return
            time.sleep(min(1.0, rest))

    def takt(self) -> None:
        """Nach jedem Schritt: Stopp pruefen, Herzschlag, Backup, Waechter."""
        self.pruefe_stopp()
        jetzt = self.uhr()
        if jetzt - self._letzter_herzschlag >= self.b["herzschlag_intervall_s"]:
            self.schreibe_status()
        self.backup_wenn_faellig()
        w = self.cfg["waechter"]
        if jetzt - self._letzter_waechter >= w["intervall_s"]:
            self._letzter_waechter = jetzt
            self.wachter_zustand = bewerte_waechter(w, *self.waechter_quelle())
        zustand, grund = self.wachter_zustand
        if zustand == "drosseln":
            self.schlafe(w["drossel_pause_s"])
        while zustand == "pausieren":
            self.schreibe_status(extra={"pause": grund})
            self.schlafe(w["pause_pruefschritt_s"])
            self._letzter_waechter = self.uhr()
            self.wachter_zustand = bewerte_waechter(w, *self.waechter_quelle())
            zustand, grund = self.wachter_zustand
        if self.b.get("schritt_pause_s"):
            self.schlafe(self.b["schritt_pause_s"])

    # ---------- Status / Backup ----------
    def ideen_pro_stunde(self) -> float:
        grenze = self.uhr() - 3600
        n = sum(1 for i in self.ideen.values() if i.get("t", 0) >= grenze)
        return float(n)

    def schreibe_status(self, extra: Optional[dict] = None, laeuft: Optional[bool] = None) -> None:
        self._letzter_herzschlag = self.uhr()
        s = {"zeit": datetime.now().isoformat(timespec="seconds"),
             "laeuft": self.laeuft if laeuft is None else laeuft,
             "runde": self.cp["runde"], "ideen_gesamt": len(self.ideen),
             "ideen_pro_stunde": self.ideen_pro_stunde(),
             "karten_gesamt": self.cp["kartennr"],
             "letzte_aktivitaet_vor_s": round(self.uhr() - self._letzte_aktivitaet, 1),
             "waechter": list(self.wachter_zustand), "fehler": self.fehler[-10:]}
        if extra:
            s.update(extra)
        atomar_schreiben(self.status_datei, json.dumps(s, ensure_ascii=False, indent=1))

    def backup_wenn_faellig(self, erzwingen: bool = False) -> Optional[Path]:
        self.backup_ordner.mkdir(parents=True, exist_ok=True)
        alt = sorted(self.backup_ordner.glob("backup_*.zip"))
        if not erzwingen and alt and self.uhr() - alt[-1].stat().st_mtime < self.b["backup_intervall_s"]:
            return None
        name = self.backup_ordner / f"backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
        if name.exists():
            return None
        tmp = name.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for ordner in (self.daten, self.ausgabe):
                if ordner.exists():
                    for f in sorted(ordner.rglob("*")):
                        if f.is_file() and not f.name.endswith(".tmp"):
                            z.write(f, f.relative_to(self.ordner).as_posix())
        os.replace(tmp, name)
        for veraltet in sorted(self.backup_ordner.glob("backup_*.zip"))[:-self.b["backup_anzahl"]]:
            veraltet.unlink()
        return name

    def fehler_melden(self, wo: str, e: BaseException) -> None:
        self.fehler.append({"zeit": datetime.now().isoformat(timespec="seconds"), "wo": wo, "fehler": str(e)[:200]})
        self.fehler = self.fehler[-50:]
        self.fehler_in_folge += 1

    # ---------- LLM-Aufruf mit Stopp-Pruefung ----------
    def frage(self, rolle: str, system: str, user: str) -> str:
        self.pruefe_stopp()
        self.warte_auf_gpu()
        t = self.cfg["temperaturen"][rolle]
        mt = self.cfg["max_tokens"][rolle]
        antwort = self.llm.chat(system, user, t, mt, self.cfg["server"]["llm_timeout_s"])
        self._letzte_aktivitaet = self.uhr()
        return antwort

    def warte_auf_gpu(self) -> None:
        """GPU-Sperre der Routine-Worker respektieren (WERKSTATT/routine/GPU_SPERRE.md):
        solange eine fremde gpu.anfrage offen ist, pausieren; Stopp wird weiter geprueft."""
        datei = self.repo / "WERKSTATT" / "gpu.lock"
        try:
            sys.path.insert(0, str(self.repo / "WERKSTATT" / "routine"))
            from gpu_sperre import anfrage_offen
        except Exception:
            return
        while anfrage_offen(datei):
            self.pruefe_stopp()
            time.sleep(1)

    # ---------- Kontext & Index ----------
    def baue_kontext(self) -> None:
        cache = self.daten / "kontext_index.json"
        self.kontext_index = kx.baue_index(self.repo, self.ordner, cache)

    def initialisiere_index(self) -> None:
        """Alle bisherigen Ideen + agenda/ideen.json in den Duplikatindex."""
        for rec in self.ideen.values():
            if rec.get("stufe") != "duplikat" and rec.get("text"):
                self.index.hinzufuegen(rec["id"], rec["text"])
        for pfad, feld in ((self.repo / "INSTITUT" / "ideen.json", "karten"),
                           (self.repo / "INSTITUT" / "agenda.json", "punkte")):
            try:
                d = json.loads(pfad.read_text(encoding="utf-8"))
            except Exception:
                continue
            for x in d.get(feld, []):
                t = (x.get("titel") or "") + " " + (x.get("frage") or "")
                if t.strip():
                    self.index.hinzufuegen(f"{pfad.name}:{x.get('id', '?')}", t)
        if not (self.daten / "lueckenkarte.json").exists() and self.kontext_index:
            self.luecken.fuelle_aus_texten([a.text for a in self.kontext_index.abschnitte])
            self.luecken.speichern()

    # ---------- Eine Idee ----------
    def erzeuge_idee(self) -> dict:
        kanal = ops.waehle_kanal(self.cfg["kanaele"]["anteil_neuland"], self.rng)
        op = self.bandit.waehle(kanal, self.rng)
        ctx = ""
        befunde = []
        if self.kontext_index:
            befunde = [a.text for a in self.kontext_index.abschnitte if a.quelle == "UEBERGABE.md#4"]
            anfrage = " ".join(self.rng.sample(self.qm, 2)) + " QIRE Befund"
            ctx = kx.formatiere(self.kontext_index.suche(anfrage, self.b["kontext_token_budget"]))
        e = ops.Eingabe(kanal, ctx, self.rng, self.luecken, self.qm, self.fremd, befunde)
        prompt, meta = ops.erzeuge_prompt(op, e)
        shots = self.rueck.fewshot(kanal)
        text = self.frage("erfinder", ops.SYSTEM_ERFINDER, (shots + "\n\n" if shots else "") + prompt).strip()
        nr = self.cp["naechste_nr"]
        self.cp["naechste_nr"] = nr + 1
        rec = {"id": f"KI-I{nr:06d}", "t": self.uhr(), "zeit": datetime.now().isoformat(timespec="seconds"),
               "kanal": kanal, "operator": op, "meta": meta, "text": text, "stufe": "roh"}
        self.schreibe_idee(rec)             # SOFORT sichern
        self.neue_ideen_dieser_lauf += 1
        if "paar" in meta:
            self.luecken.zaehle(*meta["paar"])
        return self.ideen[rec["id"]]

    def verarbeite(self, rec: dict) -> None:
        """Filtern -> Physik -> Rubrik. Jede Stufe wird sofort als Update gesichert; ein Stopp dazwischen
        laesst die Idee in ihrer Stufe stehen und sie wird beim naechsten Start fortgesetzt."""
        i = rec["id"]
        if rec["stufe"] in ("roh",):
            if not rec["text"] or len(rec["text"]) < 15:
                self.schreibe_idee({"id": i, "stufe": "verworfen", "grund": "leer/zu kurz"})
                return
            dup, sim, nah = self.index.ist_duplikat(rec["text"])
            if dup:
                self.schreibe_idee({"id": i, "stufe": "duplikat", "aehnlich_zu": nah, "aehnlichkeit": round(sim, 3)})
                return
            self.index.hinzufuegen(i, rec["text"])
            self.schreibe_idee({"id": i, "stufe": "gefiltert", "aehnlichkeit": round(sim, 3)})
            self.takt()
        if self.ideen[i]["stufe"] == "gefiltert":
            self.pruefe_stopp()
            ph = pf.pruefe_physik(self.llm, rec["text"], self.cfg["temperaturen"]["pruefer"],
                                  self.cfg["max_tokens"]["pruefer"])
            self._letzte_aktivitaet = self.uhr()
            self.schreibe_idee({"id": i, "physik": ph, "stufe": "geprueft" if ph["ok"] else "physik_abgelehnt"})
            self.takt()
        if self.ideen[i]["stufe"] == "geprueft":
            self.pruefe_stopp()
            r = pf.bewerte(self.llm, rec["text"], rec["kanal"], self.cfg["rubrik"],
                           self.rueck.fewshot(rec["kanal"]), self.cfg["temperaturen"]["bewerter"],
                           self.cfg["max_tokens"]["bewerter"])
            self._letzte_aktivitaet = self.uhr()
            if r is None:
                self.schreibe_idee({"id": i, "stufe": "unbewertet"})
            else:
                self.schreibe_idee({"id": i, "rubrik": r, "score": r["score"], "stufe": "bewertet"})
                self.bandit.aktualisiere(rec["kanal"], rec["operator"], r["score"], gewicht=0.5)
            self.takt()

    # ---------- Runde ----------
    def offene_idee_ids(self) -> list[str]:
        return [i for i, r in self.ideen.items() if r.get("stufe") in ("roh", "gefiltert", "geprueft")]

    def arbeite_offene_ab(self) -> None:
        for i in self.offene_idee_ids():
            self.sicher("wiederaufnahme", lambda i=i: self.verarbeite(self.ideen[i]))

    def sicher(self, wo: str, fn: Callable[[], None]) -> bool:
        """Fuehrt einen Schritt aus; Fehler werden protokolliert, Stopp wird durchgereicht."""
        try:
            fn()
            self.fehler_in_folge = 0
            return True
        except Stopp:
            raise
        except Exception as e:
            self.fehler_melden(wo, e)
            if self.fehler_in_folge >= self.b["max_fehler_in_folge"] or isinstance(e, LLMFehler):
                self.server_pruefen()
                self.schlafe(self.b["fehler_pause_s"])
            return False

    def server_pruefen(self) -> None:
        if self.server and not self.server.gesund():
            self.server.beenden()
            self.server.starten()
            self.server.warte_bereit(self.cfg["server"]["start_timeout_s"], self.stop_angefordert)

    def runde(self) -> None:
        self.baue_kontext()
        n_runde = self.b["ideen_pro_runde"]
        while self.cp["in_runde"] < n_runde:
            self.pruefe_stopp()
            if self.max_ideen is not None and self.neue_ideen_dieser_lauf >= self.max_ideen:
                break

            def schritt():
                rec = self.erzeuge_idee()
                self.cp["runde_ids"].append(rec["id"])
                self.cp["in_runde"] += 1
                if self.cp["in_runde"] % max(1, self.b["checkpoint_alle_ideen"]) == 0:
                    self.speichere_checkpoint()
                self.takt()
                self.verarbeite(rec)
            self.sicher("erzeugen", schritt)
        self.beende_runde()

    def beende_runde(self) -> None:
        """Beste ~10 % der Runde zu Karten ausarbeiten, Rueckmeldungen einlesen, Runde abschliessen."""
        runde = [self.ideen[i] for i in self.cp["runde_ids"] if i in self.ideen]
        bewertet = [r for r in runde if r.get("stufe") == "bewertet"]
        bewertet.sort(key=lambda r: -r["score"])
        n_top = kart.anzahl_top(len(bewertet), self.b["ausarbeiten_anteil"])
        neue = []
        for r in bewertet[:n_top]:
            if r["score"] < self.cfg["rubrik"]["mindest_score"]:
                continue

            def ausarbeiten(r=r):
                self.pruefe_stopp()
                karte = kart.arbeite_aus(self.llm, r, self.cp["kartennr"] + 1,
                                         self.cfg["temperaturen"]["ausarbeiter"], self.cfg["max_tokens"]["ausarbeiter"])
                if karte:
                    self.cp["kartennr"] += 1
                    neue.append(karte)
                    kart.schreibe_karten([karte], self.ausgabe)     # sofort sichern
                    self.schreibe_idee({"id": r["id"], "stufe": "ausgearbeitet", "karte": karte["id"]})
                else:
                    self.schreibe_idee({"id": r["id"], "karte_fehlgeschlagen": True})
                self._letzte_aktivitaet = self.uhr()
                self.takt()
            self.sicher("ausarbeiten", ausarbeiten)
        if neue:
            self.rueck.erzeuge_vorlage()
        self.rueckmeldungen_lesen()
        self.luecken.speichern()
        self.cp["runde"] += 1
        self.cp["in_runde"] = 0
        self.cp["runde_ids"] = []
        self.speichere_checkpoint()

    def rueckmeldungen_lesen(self) -> None:
        try:
            self.rueck.verarbeite_urteile(self.repo / "INSTITUT" / "agenda.json", self.repo / "INSTITUT" / "ergebnisse")
            self.rueck.lies_blindnoten()
        except Exception as e:
            self.fehler_melden("rueckmeldung", e)

    # ---------- Lauf ----------
    def abschluss(self) -> None:
        try:
            self.luecken.speichern()
            self.bandit.speichern()
            self.speichere_checkpoint()
        finally:
            self.laeuft = False
            self.schreibe_status(laeuft=False)
            if self.server:
                self.server.beenden()
            try:
                (self.daten / "betrieb.pid").unlink()
            except OSError:
                pass

    def run(self) -> None:
        self.daten.mkdir(parents=True, exist_ok=True)
        (self.daten / "betrieb.pid").write_text(str(os.getpid()))
        self.laeuft = True
        try:
            if self.server:
                self.server.starten()
                if not self.server.warte_bereit(self.cfg["server"]["start_timeout_s"], self.stop_angefordert):
                    self.pruefe_stopp()
                    raise RuntimeError("llama.cpp-Server wurde nicht bereit")
            self.schreibe_status()
            self.baue_kontext()
            self.initialisiere_index()
            self.arbeite_offene_ab()
            while True:
                self.pruefe_stopp()
                if self.max_ideen is not None and self.neue_ideen_dieser_lauf >= self.max_ideen:
                    break
                self.runde()
        except Stopp:
            pass
        finally:
            self.abschluss()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QIRE-Ideen-KI Dauerbetrieb")
    ap.add_argument("--config", default=str(ORDNER / "config.yaml"))
    ap.add_argument("--attrappe", action="store_true", help="Attrappen-LLM, kein Server (Trockenlauf)")
    ap.add_argument("--ohne-server", action="store_true", help="Server laeuft schon extern")
    ap.add_argument("--max-ideen", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = lade_config(Path(a.config))
    server = None
    if a.attrappe:
        llm: LLM = AttrappenLLM()
    else:
        from .llm import LlamaServerLLM
        from .server import ServerProzess, baue_befehl
        s = cfg["server"]
        modell = next(m for m in cfg["modelle"]["kandidaten"] if m["name"] == cfg["modelle"]["aktiv"])
        url = f"http://{s['host']}:{s['port']}"
        emb = cfg["modelle"]["embedding"]
        embed_url = None
        if emb["modus"] == "haupt":
            embed_url = url
        elif emb["modus"] == "server":
            embed_url = f"http://{s['host']}:{emb['port']}"
        llm = LlamaServerLLM(url, modell["name"], embed_url, s["llm_timeout_s"])
        if not a.ohne_server:
            server = ServerProzess(baue_befehl(cfg, modell), s["host"], s["port"], ORDNER / "daten" / "server.log")
    b = Betrieb(cfg, llm, server=server, max_ideen=a.max_ideen)
    b.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
