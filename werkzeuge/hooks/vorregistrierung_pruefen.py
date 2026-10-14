#!/usr/bin/env python3
"""Vorregistrierungs-Pruefung fuer den pre-commit-Hook (R3.3).

Regeln (Protokoll: "Kriterien vor dem Lauf in den Code, Vorregistrierung vor den Zahlen committen"):

  V1  Neue Datei INSTITUT/ergebnisse/<ID>__*.json (Rohdaten eines Laufs) nur, wenn das Modul der ID
      schon in HEAD liegt (frueherer Commit) UND im selben Commit unveraendert bleibt.
      Das Modul wird ermittelt aus INSTITUT/agenda.json (Stand HEAD), sonst aus dem Feld "modul" der Ergebnisdatei.
  V2  Die ID muss in der Agenda (Stand HEAD) stehen -- sonst nur Warnung (alte Laeufe ohne Eintrag).
  V3  Seeds eines Mess-Ergebnisses muessen in INSTITUT/seeds.json (Stand HEAD) unter dieser ID als "messung"
      stehen (Seed-Vergabe vor dem Lauf, siehe werkzeuge/seeds_pruefen.py).
  V4  Rauchtest-Ergebnisse (ID endet auf "-schnell") duerfen keine Mess-Seeds irgendeiner ID benutzen.
  V5  Neue Module (INSTITUT/abteilungen/*.py, neu hinzugefuegt): statische Pruefung auf Rauchtest-Zweige
      mit Mess-Seeds (werkzeuge/fehlerbuch_lint.py, Klasse rauchtest-messseed).
      Alle uebrigen Lint-Klassen erscheinen nur als Hinweis.

Nicht erkennbar (bewusst nicht geraten): ein Rauchtest unter normaler ID, dessen Seeds zufaellig Mess-Seeds
sind -- dafuer gibt es V3 (nicht registrierte Seeds werden abgelehnt).

UMGEHEN:  git commit --no-verify     oder     QIRE_HOOK_UEBERSPRINGEN=1 git commit ...
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WURZEL_SKRIPT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WURZEL_SKRIPT / "werkzeuge"))

ERG_RE = re.compile(r"^INSTITUT/ergebnisse/([^/_][^/]*?)__[^/]*\.json$")
MESS_KEYS = ("seeds", "seeds_mnist", "seeds_struktur", "seeds_suche")


def _git(*args, cwd=None, check=False):
    p = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p


def _seeds(v):
    if isinstance(v, bool):
        return set()
    if isinstance(v, int):
        return {v}
    if isinstance(v, list):
        return {x for x in v if isinstance(x, int) and not isinstance(x, bool)}
    if isinstance(v, dict):
        return {s for x in v.values() for s in _seeds(x)}
    return set()


def _bereiche(b):
    return {s for a, c in b for s in range(a, c + 1)}


class GitQuelle:
    """Liest Stand HEAD und Stand 'staged' (Index)."""

    def __init__(self, wurzel):
        self.wurzel = wurzel

    def head(self, pfad):
        p = _git("show", f"HEAD:{pfad}", cwd=self.wurzel)
        return p.stdout if p.returncode == 0 else None

    def index(self, pfad):
        p = _git("show", f":{pfad}", cwd=self.wurzel)
        return p.stdout if p.returncode == 0 else None

    def geaendert(self):
        p = _git("diff", "--cached", "--name-status", "--diff-filter=AMR", cwd=self.wurzel)
        out = []
        for z in p.stdout.splitlines():
            teile = z.split("\t")
            if len(teile) >= 2:
                out.append((teile[0][0], teile[-1]))
        return out

    def hat_head(self):
        return _git("rev-parse", "--verify", "-q", "HEAD", cwd=self.wurzel).returncode == 0


def modul_der_id(id_, agenda_text, ergebnis):
    """-> (Dateipfad des Moduls relativ zur Wurzel | None, Quelle)"""
    if agenda_text:
        try:
            for p in json.loads(agenda_text).get("punkte", []):
                if p.get("id") == id_ and p.get("modul"):
                    return "INSTITUT/" + p["modul"].replace(".", "/") + ".py", "agenda"
        except Exception:
            pass
    if isinstance(ergebnis, dict) and isinstance(ergebnis.get("modul"), str):
        m = ergebnis["modul"]
        m = m if m.startswith("abteilungen.") else "abteilungen." + m
        return "INSTITUT/" + m.replace(".", "/") + ".py", "ergebnisdatei"
    return None, ""


def pruefe(quelle, geaendert=None):
    """-> (fehler[list[str]], warnungen[list[str]])"""
    fehler, warn = [], []
    geaendert = geaendert if geaendert is not None else quelle.geaendert()
    if not quelle.hat_head():
        return fehler, ["Erster Commit - Vorregistrierungspruefung uebersprungen."]
    agenda_head = quelle.head("INSTITUT/agenda.json")
    seeds_head_text = quelle.head("INSTITUT/seeds.json")
    seeds_reg = json.loads(seeds_head_text) if seeds_head_text else None
    modul_in_commit = {p for s, p in geaendert}
    mess_alle, rauch_ok = {}, {}
    if seeds_reg:
        for e in seeds_reg["eintraege"]:
            if e["zweck"] == "messung":
                for s in _bereiche(e["seeds"]):
                    mess_alle.setdefault(s, set()).add(e["id"])

    for status, pfad in geaendert:
        m = ERG_RE.match(pfad)
        if not m:
            continue
        id_ = m.group(1)
        inhalt = quelle.index(pfad)
        try:
            e = json.loads(inhalt) if inhalt else None
        except Exception:
            e = None
        ist_rauch = id_.endswith("-schnell")
        basis = id_[: -len("-schnell")] if ist_rauch else id_
        mess = set()
        if isinstance(e, dict):
            for k in MESS_KEYS:
                mess |= _seeds(e.get(k))

        # V4 Rauchtest auf Mess-Seeds
        if ist_rauch:
            if mess:
                bad = sorted(s for s in mess if s in mess_alle)
                if bad:
                    ids = sorted({i for s in bad for i in mess_alle[s]})
                    fehler.append(f"{pfad}: Rauchtest benutzt Mess-Seeds {bad[:8]} (Mess-ID: {', '.join(ids)}). "
                                  "Rauchtests nur auf Tuning-/Entwurfs-Seeds (900-999 bzw. xx90-99).")
            continue                                  # V1-V3 gelten fuer Mess-Laeufe

        # V1 Modul vorher committet
        if status != "A":
            continue                                  # nachtraegliche Aenderung an alter Rohdatei: nicht Gegenstand
        modul, wo = modul_der_id(id_, agenda_head, e)
        if modul is None:
            warn.append(f"{pfad}: Modul zu ID {id_} nicht ermittelbar (weder Agenda noch Feld 'modul') - V1 nicht geprueft.")
        else:
            if quelle.head(modul) is None:
                fehler.append(f"{pfad}: Modul {modul} (laut {wo}) liegt nicht in einem fruehren Commit. "
                              "Erst Modul + Kriterien + Agenda-Eintrag committen (Vorregistrierung), dann die Rohdaten.")
            elif modul in modul_in_commit:
                fehler.append(f"{pfad}: Modul {modul} wird im SELBEN Commit wie die Ergebnisse geaendert - "
                              "Kriterien duerfen nicht zusammen mit den Zahlen eingereicht werden.")
        # V2 Agenda-Eintrag
        if agenda_head:
            try:
                ids = {p.get("id") for p in json.loads(agenda_head).get("punkte", [])}
            except Exception:
                ids = set()
            if id_ not in ids:
                warn.append(f"{pfad}: ID {id_} steht nicht in INSTITUT/agenda.json (HEAD).")
        # V3 Seeds registriert
        if seeds_reg is None:
            warn.append(f"{pfad}: INSTITUT/seeds.json fehlt in HEAD - Seed-Register nicht geprueft.")
        elif mess:
            reg = set()
            for en in seeds_reg["eintraege"]:
                if en["id"] == id_ and en["zweck"] in ("messung", "auswertung"):
                    reg |= _bereiche(en["seeds"])
            fehlt = sorted(mess - reg)
            if fehlt:
                fehler.append(f"{pfad}: Seeds {fehlt[:8]} sind nicht als messung von {id_} in INSTITUT/seeds.json (HEAD) "
                              "eingetragen. Erst vergeben (werkzeuge/seeds_pruefen.py --neu VON BIS), committen, dann rechnen.")

    # V5 neue Module statisch
    try:
        import fehlerbuch_lint as L
        import tempfile
        mess_reg = L.mess_seeds_register()
        for status, pfad in geaendert:
            if status == "A" and re.match(r"^INSTITUT/abteilungen/[^/]+\.py$", pfad):
                text = quelle.index(pfad)
                if text is None:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    pf = Path(td) / "abteilungen" / Path(pfad).name
                    pf.parent.mkdir()
                    pf.write_text(text, encoding="utf-8")
                    for z, k, t in L.pruefe_datei(pf, mess_reg):
                        if k == "rauchtest-messseed":
                            fehler.append(f"{pfad}:{z}: {t}")
                        else:
                            warn.append(f"{pfad}:{z}: [lint {k}] {t}")
    except Exception as ex:                                # Lint darf den Commit nie wegen eigener Fehler sperren
        warn.append(f"Lint nicht ausgefuehrt: {ex!r}")
    return fehler, warn


def main():
    top = _git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return 0
    wurzel = top.stdout.strip()
    fehler, warn = pruefe(GitQuelle(wurzel))
    for w in warn:
        print("pre-commit Hinweis:", w, file=sys.stderr)
    if fehler:
        print("\npre-commit BLOCKIERT (Vorregistrierung/Seeds):", file=sys.stderr)
        for f in fehler:
            print("  -", f, file=sys.stderr)
        print("\nBewusst umgehen: git commit --no-verify  (Grund in die Commit-Nachricht schreiben)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
