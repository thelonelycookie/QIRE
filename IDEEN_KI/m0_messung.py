"""Schritt M0: Kandidatenmodelle messen (Tokens/s, VRAM/RAM, 20 Probeaufgaben -> Blindbewertung).

    python m0_messung.py --minuten 15              # alle Kandidaten aus config.yaml
    python m0_messung.py --modell gpt-oss-20b      # nur einer
    python m0_messung.py --attrappe --minuten 0.05 # Trockenlauf ohne GPU

Ergebnis in ausgabe/m0/:  messung.json (Zahlen), blindbewertung.csv (Malte traegt Noten 1-5 ein),
schluessel.json (welches Modell hinter A/B/C steckt -- erst NACH der Bewertung ansehen!).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
from pathlib import Path

from schmiede.betrieb import gpu_status, lade_config, ram_status
from schmiede.llm import AttrappenLLM, LlamaServerLLM

ORDNER = Path(__file__).resolve().parent


class Peaks(threading.Thread):
    """Misst im Hintergrund Spitzen von VRAM-Anteil, GPU-Temperatur und RAM."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.vram = self.temp = self.ram = 0.0

    def run(self):
        while not self.stop:
            g, r = gpu_status(), ram_status()
            if g:
                self.vram, self.temp = max(self.vram, g["vram_anteil"]), max(self.temp, g["temp"])
            if r:
                self.ram = max(self.ram, r["ram_anteil"])
            time.sleep(2)


def messe(llm, aufgaben, minuten, max_tokens=400):
    """Schleife ueber Probeaufgaben fuer `minuten`; gibt Zahlen + Antworten der ersten Runde."""
    ende = time.time() + minuten * 60
    antworten, tps, n_tok, t_ges = {}, [], 0, 0.0
    i = 0
    while True:
        a = aufgaben[i % len(aufgaben)]
        t0 = time.time()
        try:
            text = llm.chat("Du bist ein hilfreicher Forschungsassistent. Antworte knapp auf Deutsch.", a, 0.7, max_tokens, 90)
        except Exception as e:
            text = f"[FEHLER: {e}]"
        dt = time.time() - t0
        tim = getattr(llm, "letzte_timings", {}) or {}
        tok = tim.get("predicted_n") or max(1, len(text.split()) * 2)
        rate = tim.get("predicted_per_second") or tok / max(dt, 1e-6)
        tps.append(rate); n_tok += tok; t_ges += dt
        antworten.setdefault(i % len(aufgaben), text)
        i += 1
        # mindestens alle Aufgaben einmal, danach bis die Zeit um ist (Obergrenze 10 Durchgaenge)
        if (time.time() >= ende and i >= len(aufgaben)) or i >= 10 * len(aufgaben):
            break
    return {"aufrufe": i, "tokens_pro_s_mittel": sum(tps) / len(tps), "tokens_pro_s_min": min(tps),
            "tokens_gesamt": n_tok}, antworten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minuten", type=float, default=15)
    ap.add_argument("--modell", default=None)
    ap.add_argument("--attrappe", action="store_true")
    a = ap.parse_args()
    cfg = lade_config()
    aufgaben = [z.strip() for z in (ORDNER / "daten_fest" / "m0_aufgaben.txt").read_text(encoding="utf-8").splitlines() if z.strip()]
    kand = [m for m in cfg["modelle"]["kandidaten"] if not a.modell or m["name"] == a.modell]
    out = ORDNER / "ausgabe" / "m0"
    out.mkdir(parents=True, exist_ok=True)
    messung, alle_antworten = {}, {}
    for m in kand:
        print(f"== {m['name']} ({a.minuten} min)")
        peaks = Peaks(); peaks.start()
        server = None
        if a.attrappe:
            llm = AttrappenLLM()
        else:
            from schmiede.server import ServerProzess, baue_befehl
            s = cfg["server"]
            server = ServerProzess(baue_befehl(cfg, m), s["host"], s["port"], out / f"server_{m['name']}.log")
            server.starten()
            if not server.warte_bereit(s["start_timeout_s"]):
                print("  Server nicht bereit (Modelldatei/Pfad pruefen) -> uebersprungen")
                server.beenden(); peaks.stop = True
                messung[m["name"]] = {"fehler": "Server nicht bereit"}
                continue
            llm = LlamaServerLLM(server.url, m["name"])
        try:
            zahlen, antw = messe(llm, aufgaben, a.minuten)
        finally:
            peaks.stop = True
            if server:
                server.beenden()
        zahlen.update({"vram_spitze_anteil": peaks.vram, "gpu_temp_spitze_c": peaks.temp, "ram_spitze_anteil": peaks.ram,
                       "konfig": {k: m.get(k) for k in ("n_cpu_moe", "ngl", "ctx")}})
        messung[m["name"]] = zahlen
        alle_antworten[m["name"]] = antw
        print("  ", json.dumps(zahlen, ensure_ascii=False))
    (out / "messung.json").write_text(json.dumps(messung, indent=1, ensure_ascii=False), encoding="utf-8")
    # Blinde Zuordnung: Modelle -> Buchstaben, pro Aufgabe zufaellig gemischt
    namen = list(alle_antworten)
    rng = random.Random(42)
    schluessel = {}
    with open(out / "blindbewertung.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["aufgabe_nr", "aufgabe", "variante", "antwort", "note_1_bis_5"])
        for i, auf in enumerate(aufgaben):
            ordnung = namen[:]; rng.shuffle(ordnung)
            for buchstabe, n in zip("ABCDEF", ordnung):
                schluessel[f"{i+1}{buchstabe}"] = n
                w.writerow([i + 1, auf, buchstabe, alle_antworten[n].get(i, "").replace("\n", " "), ""])
    (out / "schluessel.json").write_text(json.dumps(schluessel, indent=1), encoding="utf-8")
    print("Fertig: ausgabe/m0/ -- Malte bewertet blindbewertung.csv (Spalte note_1_bis_5).")


if __name__ == "__main__":
    main()
