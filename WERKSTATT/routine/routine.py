# -*- coding: utf-8 -*-
"""Routine-Worker (R5.1): kleine, wiederkehrende Aufgaben ueber den lokalen llama.cpp-Server,
damit das Claude-Kontingent geschont wird. Reine Standardbibliothek.

    python WERKSTATT/routine/routine.py log <logdatei> [--zeilen N]   # (a) Lauf-Log -> kurze Markdown-Notiz
    python WERKSTATT/routine/routine.py bericht                       # (b) Status-Bericht aus agenda.json + laeufe/*.status.json
    python WERKSTATT/routine/routine.py vortest [--trotzdem]          # (c) `direktor.py vortest` anstossen + zusammenfassen
    python WERKSTATT/routine/routine.py sperre [--erzwinge-frei]      # Zustand der GPU-Sperre zeigen
    Gemeinsame Optionen: --ohne-llm (nur deterministischer Rohbericht), --konfig <datei>, --id <Auftrags-ID>

Regeln (WERKSTATT/CLAUDE.md):
- Ergebnis nach WERKSTATT/ergebnisse/<ID>/ (bericht.md im Berichtsformat, notiz.md, rohdaten). Sonst schreibt der Worker nichts.
- KEIN git: kein add/commit/push. Der Planer (Claude/Malte) entscheidet, was ins Repo kommt.
- Ruhezeit 00:00-06:00 gilt fuer den Routine-Worker (fuer die Ideen-KI ist sie ausgesetzt): kein Start in dieser Zeit.
- GPU-Zugriff nur unter der Sperre WERKSTATT/gpu.lock (siehe GPU_SPERRE.md). Sperre wird nur fuer die
  LLM-Aufrufe gehalten (Daten sammeln und Vortest laufen ohne Sperre).
- Ehrlich: Die Modell-Zusammenfassung steht immer UNTER den Rohfakten und ist als "ungeprueft" markiert;
  Zahlen, die nicht in der Eingabe stehen, werden ausdruecklich gemeldet. Faellt das Modell aus, entsteht
  trotzdem der Rohbericht (STATUS: teilweise).
Exit-Codes: 0 ok (auch bei Rohbericht ohne Modell, dann STATUS: teilweise), 1 Fehler/Vortest durchgefallen, 2 Aufruf,
3 Ruhezeit, 5 Vortest nicht gestartet (anderer Lauf aktiv).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

HIER = Path(__file__).resolve().parent
WERKSTATT = HIER.parent
REPO = WERKSTATT.parent
INSTITUT = REPO / "INSTITUT"
ERGEBNISSE = WERKSTATT / "ergebnisse"
sys.path.insert(0, str(HIER))
from gpu_sperre import GpuSperre, anfrage_offen, lese_sperre  # noqa: E402

SYSTEM = ("Du bist der Routine-Worker des QIRE-Projekts (Jugend forscht). Antworte auf Deutsch, knapp, sachlich, "
          "in Stichpunkten. Verwende NUR Fakten und Zahlen aus der Eingabe; erfinde nichts, runde nichts, "
          "schaetze nichts. Wenn etwas in der Eingabe fehlt, schreibe 'nicht erkennbar'.")


# ------------------------------------------------------------------ Konfiguration
def lade_konfig(pfad: Optional[Path] = None) -> dict:
    return json.loads(Path(pfad or HIER / "konfig.json").read_text(encoding="utf-8"))


def ermittle_llm_url(cfg: dict) -> str:
    """'auto' -> host/port aus IDEEN_KI/config.yaml (Abschnitt server:). Ohne PyYAML genuegt eine einfache Zeilensuche."""
    url = cfg["llm"].get("url", "auto")
    if url != "auto":
        return url.rstrip("/")
    pfad = (HIER / cfg["llm"].get("ideen_ki_konfig", "../../IDEEN_KI/config.yaml")).resolve()
    host, port = "127.0.0.1", 8080
    try:
        text = pfad.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
            s = yaml.safe_load(text).get("server", {})
            host, port = s.get("host", host), int(s.get("port", port))
        except ImportError:
            block = re.search(r"^server:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
            if block:
                h = re.search(r"^\s+host:\s*([\w.\-]+)", block.group(1), re.M)
                p = re.search(r"^\s+port:\s*(\d+)", block.group(1), re.M)
                host = h.group(1) if h else host
                port = int(p.group(1)) if p else port
    except OSError:
        pass
    return f"http://{host}:{port}"


def in_ruhezeit(cfg: dict, stunde: Optional[int] = None) -> bool:
    ab, bis = cfg["ruhezeit"]["von"], cfg["ruhezeit"]["bis"]
    h = time.localtime().tm_hour if stunde is None else stunde
    return (ab <= h < bis) if ab < bis else (h >= ab or h < bis)


# ------------------------------------------------------------------ LLM (OpenAI-kompatibel)
class LLMFehler(RuntimeError):
    pass


def chat(url: str, user: str, cfg: dict, system: str = SYSTEM) -> str:
    """POST /v1/chat/completions (llama.cpp-Server). Harte Zeitgrenze aus der Konfig (<= 90 s)."""
    nutz = {"model": cfg["llm"].get("modell", "lokal"), "temperature": cfg["llm"].get("temperatur", 0.2),
            "max_tokens": cfg["llm"].get("max_tokens", 700),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(nutz).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=min(float(cfg["llm"].get("timeout_s", 90)), 90.0)) as r:
            d = json.loads(r.read().decode("utf-8"))
        return (d["choices"][0]["message"]["content"] or "").strip()
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
        raise LLMFehler(f"chat fehlgeschlagen: {e}") from e


def zahlen_ohne_beleg(antwort: str, eingabe: str) -> list[str]:
    """Zahlen (>= 2 Ziffern oder mit Dezimalstelle), die in der Modellantwort stehen, aber nicht in der Eingabe."""
    def norm(t):
        return {z.replace(",", ".") for z in re.findall(r"\d+(?:[.,]\d+)?", t)}
    ein = norm(eingabe)
    return sorted(z for z in norm(antwort) if (len(z) >= 2 or "." in z) and z not in ein)


class Zusammenfasser:
    """Haelt (falls gewuenscht) die GPU-Sperre nur fuer die Dauer der LLM-Aufrufe."""

    def __init__(self, cfg: dict, ohne_llm: bool = False, sperre_datei: Optional[Path] = None,
                 llm_fn: Optional[Callable[[str, str], str]] = None):
        self.cfg, self.ohne_llm = cfg, ohne_llm
        self.url = ermittle_llm_url(cfg)
        sp = cfg["gpu_sperre"]
        self.sperre = GpuSperre("routine-worker", "Routine-Zusammenfassung",
                                datei=sperre_datei or (HIER / sp["datei"]).resolve(),
                                tot_nach_s=sp["tot_nach_s"], max_halten_s=sp["max_halten_s"])
        self.llm_fn = llm_fn or (lambda user, system: chat(self.url, user, cfg, system))
        self.probleme: list[str] = []

    def __call__(self, user: str, system: str = SYSTEM) -> Optional[str]:
        """Modell fragen. None = nicht verfuegbar (Grund steht in self.probleme)."""
        if self.ohne_llm:
            return None
        sp = self.cfg["gpu_sperre"]
        if not self.sperre.erwerbe(warte_max_s=sp["warte_max_s"], vram_frei=(sp["modus"] == "eigener_server")):
            b = lese_sperre(self.sperre.datei) or {}
            self.probleme.append(f"GPU-Sperre nach {sp['warte_max_s']} s nicht frei (Besitzer: {b.get('besitzer', '?')}, "
                                 f"{b.get('zweck', '')}) -- ohne Modell weitergemacht")
            return None
        try:
            return chat_mit_wiederholung(self.llm_fn, user, system, self.probleme)
        finally:
            self.sperre.freigeben()


def chat_mit_wiederholung(fn, user, system, probleme, versuche: int = 2) -> Optional[str]:
    letzter = ""
    for _ in range(versuche):
        try:
            return fn(user, system)
        except LLMFehler as e:
            letzter = str(e)
            time.sleep(1)
    probleme.append(f"Modell nicht erreichbar ({letzter}) -- nur Rohbericht")
    return None


# ------------------------------------------------------------------ Ausgabe im Berichtsformat
def _rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def neue_id(art: str) -> str:
    return f"routine-{time.strftime('%Y%m%d-%H%M%S')}-{art}"


def schreibe_ergebnis(auftrag: str, status: str, ergebnis: str, zahlen: str, dateien: dict[str, str],
                      probleme: list[str], ergebnisse_dir: Optional[Path] = None) -> Path:
    ziel = (ergebnisse_dir or ERGEBNISSE) / auftrag
    ziel.mkdir(parents=True, exist_ok=True)
    for name, inhalt in dateien.items():
        (ziel / name).write_text(inhalt, encoding="utf-8")
    bericht = (f"AUFTRAG: {auftrag}\nSTATUS: {status}\nERGEBNIS: {ergebnis}\nZAHLEN: {zahlen}\n"
               f"DATEIEN: {', '.join(['bericht.md'] + list(dateien))} (in {_rel(ziel)})\n"
               f"COMMIT: keiner (Routine-Worker committet nie)\nPROBLEME: {'; '.join(probleme) or 'keine'}\n")
    (ziel / "bericht.md").write_text(bericht, encoding="utf-8")
    return ziel


# ------------------------------------------------------------------ (a) Log zusammenfassen
WICHTIG = re.compile(r"traceback|error|fehler|exception|warn|oom|memory|killed|abbruch|h0|h1|hu\b|urteil|fertig|"
                     r"seed|gesamt|ergebnis|nan\b|durchgefallen|bestanden", re.I)


def log_auszug(text: str, max_zeichen: int) -> tuple[str, dict]:
    """Anfang + wichtige Zeilen + Ende, gekuerzt auf max_zeichen. Rueckgabe (Auszug, Kennzahlen)."""
    zeilen = text.splitlines()
    n = len(zeilen)
    kenn = {"zeilen": n, "zeichen": len(text),
            "fehlerzeilen": sum(bool(re.search(r"traceback|error|exception|fehler|oom|killed", z, re.I)) for z in zeilen),
            "warnzeilen": sum(bool(re.search(r"warn", z, re.I)) for z in zeilen)}
    if len(text) <= max_zeichen:
        return text, kenn
    kopf, ende = zeilen[:15], zeilen[-40:]
    mitte = [z for z in zeilen[15:-40] if WICHTIG.search(z)]
    teile = ["\n".join(kopf), "[...]", "\n".join(mitte[: max(0, 120)]), "[...]", "\n".join(ende)]
    auszug = "\n".join(teile)
    if len(auszug) > max_zeichen:
        auszug = auszug[: max_zeichen // 3] + "\n[... gekuerzt ...]\n" + auszug[-(2 * max_zeichen // 3):]
    return auszug, kenn


def cmd_log(a, cfg, zus: Zusammenfasser) -> int:
    pfad = Path(a.logdatei)
    if not pfad.exists():
        print(f"Logdatei fehlt: {pfad}", file=sys.stderr)
        return 1
    with pfad.open(encoding="utf-8", errors="replace") as f:
        text = f.read()
    if a.zeilen:
        text = "\n".join(text.splitlines()[-a.zeilen:])
    auszug, kenn = log_auszug(text, cfg["llm"]["eingabe_zeichen"])
    antwort = zus(f"Fasse dieses Lauf-Log in 5-8 Stichpunkten zusammen: Was lief, wie weit, Fehler/Warnungen, "
                  f"Ergebnis/Urteil falls erkennbar, naechster sinnvoller Schritt.\n\nLOG ({pfad.name}):\n{auszug}")
    fremd = zahlen_ohne_beleg(antwort, auszug) if antwort else []
    notiz = [f"# Log-Notiz: {pfad.name}", "",
             f"*Erzeugt {time.strftime('%d.%m.%Y %H:%M')} vom Routine-Worker. Quelle: `{pfad}`.*", "",
             "## Rohfakten (deterministisch)",
             f"- Zeilen: {kenn['zeilen']}, Zeichen: {kenn['zeichen']}",
             f"- Zeilen mit Fehler/Traceback/OOM: {kenn['fehlerzeilen']}, mit Warnung: {kenn['warnzeilen']}",
             "- Letzte Zeilen:", "```", *text.splitlines()[-6:], "```", ""]
    if antwort:
        notiz += ["## Zusammenfassung des lokalen Modells (ungeprüft)", antwort, ""]
        if fremd:
            notiz += [f"> **Achtung:** Diese Zahlen stehen in der Zusammenfassung, aber nicht im Log-Auszug: {', '.join(fremd)}. Nicht übernehmen, ohne im Log zu prüfen.", ""]
    else:
        notiz += ["## Zusammenfassung", "nicht erzeugt (Modell/GPU nicht verfügbar oder --ohne-llm).", ""]
    aid = a.id or neue_id("log")
    ziel = schreibe_ergebnis(aid, "erledigt" if antwort else "teilweise",
                             f"Log {pfad.name}: {kenn['zeilen']} Zeilen, {kenn['fehlerzeilen']} Fehlerzeilen"
                             + ("; Modell-Zusammenfassung in notiz.md." if antwort else "; nur Rohfakten."),
                             f"{kenn['zeilen']} Zeilen / {kenn['fehlerzeilen']} Fehlerzeilen (Quelle: {pfad})",
                             {"notiz.md": "\n".join(notiz)}, zus.probleme + ([f"Zahlen ohne Beleg im Log: {', '.join(fremd)}"] if fremd else []))
    print(ziel / "notiz.md")
    return 0


# ------------------------------------------------------------------ (b) Status-Bericht
def lies_laeufe(laeufe_dir: Path, tot_nach_s: float, jetzt: Optional[float] = None) -> list[dict]:
    t = time.time() if jetzt is None else jetzt
    out = []
    for p in sorted(laeufe_dir.glob("*.status.json")) if laeufe_dir.exists() else []:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append({"id": p.name.replace(".status.json", ""), "zustand": "unlesbar"})
            continue
        d.setdefault("id", p.name.replace(".status.json", ""))
        d["alter_s"] = t - float(d.get("herzschlag", 0) or 0)
        if d.get("zustand") == "laeuft" and d["alter_s"] > tot_nach_s:
            d["zustand"] = "tot"
        out.append(d)
    return out


def baue_statusbericht(agenda: dict, laeufe: list[dict]) -> tuple[str, dict]:
    ps = agenda.get("punkte", [])
    nach = {}
    for p in ps:
        nach[p.get("status", "?")] = nach.get(p.get("status", "?"), 0) + 1
    malte = [p for p in ps if p.get("wartet_auf") == "malte"]
    offen = [p for p in ps if p.get("status") == "offen"]
    z = ["## Agenda", "", f"- Punkte gesamt: {len(ps)} (" + ", ".join(f"{k}: {v}" for k, v in sorted(nach.items())) + ")", ""]
    z += [f"**Wartet auf Malte ({len(malte)}):**"] + ([f"- {p['id']} [{p.get('status')}] {(p.get('notiz') or p.get('frage') or '')[:110]}" for p in malte] or ["- nichts"])
    z += ["", f"**Offen/rechenbar ({len(offen)}):** " + (", ".join(p["id"] for p in offen) or "keine"), "", "## Läufe (laeufe/*.status.json)", ""]
    if laeufe:
        z += ["| Lauf | Zustand | Fortschritt | Rechner | Herzschlag vor |", "|---|---|---|---|---|"]
        for d in laeufe:
            fort = f"{d.get('seed', 0)}/{d['gesamt']}" if d.get("gesamt") else str(d.get("seed", "–"))
            z.append(f"| {d.get('id')} | {d.get('zustand')} | {fort} | {d.get('rechner', '?')} | {int(d.get('alter_s', 0))} s |")
    else:
        z.append("Keine Statusdateien vorhanden (nichts läuft oder R1.4-Wächter noch nicht benutzt).")
    kenn = {"punkte": len(ps), "nach_status": nach, "wartet_auf_malte": len(malte), "offen": len(offen),
            "laeufe": len(laeufe), "laeuft": sum(d.get("zustand") == "laeuft" for d in laeufe),
            "tot": sum(d.get("zustand") == "tot" for d in laeufe), "fehler": sum(d.get("zustand") == "fehler" for d in laeufe)}
    return "\n".join(z), kenn


def cmd_bericht(a, cfg, zus: Zusammenfasser) -> int:
    ag = INSTITUT / "agenda.json"
    probleme = []
    try:
        agenda = json.loads(ag.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        agenda = {"punkte": []}
        probleme.append(f"agenda.json nicht lesbar: {e}")
    laeufe = lies_laeufe(INSTITUT / "laeufe", cfg["laeufe_tot_nach_s"])
    roh, kenn = baue_statusbericht(agenda, laeufe)
    antwort = zus("Schreibe aus diesem Status eine Kurzbewertung (max. 5 Stichpunkte): was läuft, was ist tot/fehlerhaft, "
                  "was wartet auf Malte, was ist als nächstes dran. Keine Zahlen außer denen im Text.\n\n" + roh)
    fremd = zahlen_ohne_beleg(antwort, roh) if antwort else []
    md = [f"# Status-Bericht {time.strftime('%d.%m.%Y %H:%M')}", "", "*Routine-Worker. Rohfakten unten stammen direkt aus agenda.json und laeufe/*.status.json.*", ""]
    if antwort:
        md += ["## Kurzbewertung des lokalen Modells (ungeprüft)", antwort, ""]
        if fremd:
            md += [f"> **Achtung:** Zahlen ohne Beleg im Status: {', '.join(fremd)}.", ""]
    md += [roh, ""]
    aid = a.id or neue_id("status")
    ziel = schreibe_ergebnis(aid, "erledigt" if not probleme and (antwort or a.ohne_llm) else "teilweise",
                             f"{kenn['punkte']} Agenda-Punkte, {kenn['wartet_auf_malte']} warten auf Malte, {kenn['laeuft']} Lauf/Läufe aktiv, {kenn['tot']} tot, {kenn['fehler']} Fehler.",
                             json.dumps(kenn, ensure_ascii=False) + f" (Quelle: {_rel(ag)}, INSTITUT/laeufe/)",
                             {"status.md": "\n".join(md), "kennzahlen.json": json.dumps(kenn, ensure_ascii=False, indent=1)},
                             probleme + zus.probleme + ([f"Zahlen ohne Beleg: {', '.join(fremd)}"] if fremd else []))
    print(ziel / "status.md")
    return 0


# ------------------------------------------------------------------ (c) Vortests
def python_befehl(cfg: dict) -> list[str]:
    p = cfg.get("python", "auto")
    return [sys.executable] if p == "auto" else ([p] if isinstance(p, str) else list(p))


def parse_vortest(ausgabe: str) -> dict:
    ok, schlecht = [], []
    for z in ausgabe.splitlines():
        m = re.match(r"\s*(\S+)\s+Vortest:\s*(bestanden|DURCHGEFALLEN)", z)
        if m:
            (ok if m.group(2) == "bestanden" else schlecht).append(m.group(1))
    return {"bestanden": ok, "durchgefallen": schlecht,
            "traceback": "Traceback (most recent call last)" in ausgabe}


def cmd_vortest(a, cfg, zus: Zusammenfasser) -> int:
    laeufe = lies_laeufe(INSTITUT / "laeufe", cfg["laeufe_tot_nach_s"])
    aktiv = [d["id"] for d in laeufe if d.get("zustand") == "laeuft"]
    if aktiv and not a.trotzdem:
        print(f"Vortest nicht gestartet: Lauf aktiv ({', '.join(aktiv)}). Nur ein Rechenlauf zur Zeit (4 Kerne, T-006-OOM). "
              "Mit --trotzdem erzwingen.", file=sys.stderr)
        return 5
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    try:
        r = subprocess.run(python_befehl(cfg) + ["direktor.py", "vortest"], cwd=str(INSTITUT), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=cfg["vortest_timeout_s"])
        ausgabe, code, zeitueberschreitung = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else ""), r.returncode, False
    except subprocess.TimeoutExpired as e:
        ausgabe = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n[ZEITUEBERSCHREITUNG]"
        code, zeitueberschreitung = -1, True
    dauer = time.time() - t0
    erg = parse_vortest(ausgabe)
    kenn = dict(erg, exitcode=code, dauer_s=round(dauer, 1), zeitueberschreitung=zeitueberschreitung)
    auszug, _ = log_auszug(ausgabe, cfg["llm"]["eingabe_zeichen"])
    antwort = zus("Fasse das Ergebnis dieser Vortests in max. 5 Stichpunkten zusammen. Nenne durchgefallene Vortests beim Namen "
                  "und die wahrscheinliche Ursache, falls im Text erkennbar.\n\n" + auszug) if (erg["durchgefallen"] or erg["traceback"] or code != 0) else None
    md = [f"# Vortest-Ergebnis {time.strftime('%d.%m.%Y %H:%M')}", "",
          f"- Befehl: `direktor.py vortest` in `INSTITUT/` (OMP_NUM_THREADS=1), Exitcode {code}, Dauer {dauer:.0f} s",
          f"- bestanden: {len(erg['bestanden'])} ({', '.join(erg['bestanden']) or '–'})",
          f"- DURCHGEFALLEN: {len(erg['durchgefallen'])} ({', '.join(erg['durchgefallen']) or '–'})",
          f"- Traceback in der Ausgabe: {'ja' if erg['traceback'] else 'nein'}", ""]
    if antwort:
        md += ["## Einordnung des lokalen Modells (ungeprüft)", antwort, ""]
    elif erg["durchgefallen"] or erg["traceback"]:
        md += ["## Einordnung", "nicht erzeugt (Modell/GPU nicht verfügbar oder --ohne-llm); siehe Rohausgabe.", ""]
    ok_gesamt = code == 0 and not erg["durchgefallen"] and not erg["traceback"] and not zeitueberschreitung
    aid = a.id or neue_id("vortest")
    ziel = schreibe_ergebnis(aid, "erledigt" if ok_gesamt else "teilweise",
                             f"Vortests: {len(erg['bestanden'])} bestanden, {len(erg['durchgefallen'])} durchgefallen"
                             + (", Zeitüberschreitung" if zeitueberschreitung else "") + ".",
                             json.dumps(kenn, ensure_ascii=False) + " (Quelle: rohausgabe.txt)",
                             {"notiz.md": "\n".join(md), "rohausgabe.txt": ausgabe, "kennzahlen.json": json.dumps(kenn, ensure_ascii=False, indent=1)},
                             zus.probleme + ([f"durchgefallen: {', '.join(erg['durchgefallen'])}"] if erg["durchgefallen"] else []))
    print(ziel / "notiz.md")
    return 0 if ok_gesamt else 1


def cmd_sperre(a, cfg, zus) -> int:
    datei = (HIER / cfg["gpu_sperre"]["datei"]).resolve()
    s = lese_sperre(datei, cfg["gpu_sperre"]["tot_nach_s"])
    print(json.dumps({"sperre": s, "anfrage": anfrage_offen(datei)}, ensure_ascii=False, indent=1) if s or anfrage_offen(datei) else "GPU frei (keine gpu.lock).")
    if a.erzwinge_frei and s and s.get("veraltet"):
        datei.unlink(missing_ok=True)
        print("Veraltete Sperre entfernt.")
    elif a.erzwinge_frei and s:
        print("Sperre ist NICHT veraltet -- nicht entfernt (Besitzer lebt).", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------ main
def main(argv=None, stunde: Optional[int] = None, llm_fn=None, sperre_datei: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description="QIRE Routine-Worker (R5.1)")
    ap.add_argument("--konfig")
    ap.add_argument("--ohne-llm", action="store_true")
    ap.add_argument("--id", help="Auftrags-ID (Standard: routine-<Zeit>-<art>)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("log"); pl.add_argument("logdatei"); pl.add_argument("--zeilen", type=int, default=0)
    sub.add_parser("bericht")
    pv = sub.add_parser("vortest"); pv.add_argument("--trotzdem", action="store_true")
    ps = sub.add_parser("sperre"); ps.add_argument("--erzwinge-frei", action="store_true")
    a = ap.parse_args(argv)
    cfg = lade_konfig(Path(a.konfig) if a.konfig else None)
    if a.cmd != "sperre" and in_ruhezeit(cfg, stunde):
        print(f"Ruhezeit ({cfg['ruhezeit']['von']}-{cfg['ruhezeit']['bis']} Uhr): Routine-Worker startet nicht.", file=sys.stderr)
        return 3
    zus = Zusammenfasser(cfg, a.ohne_llm, sperre_datei, llm_fn)
    try:
        return {"log": cmd_log, "bericht": cmd_bericht, "vortest": cmd_vortest, "sperre": cmd_sperre}[a.cmd](a, cfg, zus)
    except KeyboardInterrupt:
        zus.sperre.freigeben()
        return 1


if __name__ == "__main__":
    sys.exit(main())
