#!/usr/bin/env python3
"""FEHLERBUCH-Lint (R3.5): statische, heuristische Pruefung der bekannten Fehlerklassen.

    python werkzeuge/fehlerbuch_lint.py                    # alle Module in INSTITUT/abteilungen + labor
    python werkzeuge/fehlerbuch_lint.py INSTITUT/abteilungen/neu.py
    python werkzeuge/fehlerbuch_lint.py --streng           # Exit 1 bei Warnungen in NEUEN Modulen
    python werkzeuge/fehlerbuch_lint.py --json

Klassen (Quelle: INSTITUT/FEHLERBUCH.md, UEBERGABE.md §3/§6):
  arm-erbt-kopf       Arm uebernimmt Kopf/Zustand eines anderen Arms (deepcopy/load_state_dict), ohne
                      Hinweis auf frischen Kopf oder eigene Lernrate (C-clifford v1, 24.09.)
  blas-threads        Parallelverarbeitung ohne OMP/OPENBLAS/MKL_NUM_THREADS=1 bzw. set_num_threads(1)
                      und ohne labor.gemeinsam.parallel (Ueberbuchung, 24.09.)
  ratensuche          einfache Lernratensuche mit <= 2 Tuning-Seeds ohne gittersuche_stabil/waehle_rate
                      und ohne Begruendungskommentar (C-referenz, I-003, I-035)
  urteil-kein-string  Urteile muessen Strings sein (Module liefern 'H1: ...'/'HU: ...'; I-020-Absturz)
  keine-wiederaufnahme  lauf() ohne Zwischenstand je Seed (Wiederaufnahme ist Pflicht)
  kontrolle-ohne-matrixcheck  Kontrollarm (hadamard/identitaet/zufall...) ohne Matrix-/Entartungspruefung im vortest
  rauchtest-messseed  Rauchtest-Zweig (if schnell) nutzt Seeds, die in INSTITUT/seeds.json Mess-Seeds sind
  kein-vortest        Modul ohne vortest()
  doppelte-definition dieselbe Funktion mehrfach auf Modulebene definiert (nur die letzte zaehlt; Spleiss-Artefakt,
                      Teile davon koennen tot oder kaputt sein -- z.B. mnist100.py)

Es sind HEURISTIKEN: Ausgabe sind Warnungen, keine Beweise. Altmodule (schon gerechnet, Ergebnisse
festgeschrieben) werden nur berichtet; --streng bewertet nur Module ohne Ergebnisdatei.
Bewusste Ausnahme im Quelltext: Kommentar `# lint: ok <klasse> <Begruendung>` in der Datei.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
INSTITUT = WURZEL / "INSTITUT"
SUCHPFADE = [INSTITUT / "abteilungen", INSTITUT / "labor"]
BEGRUENDUNG = re.compile(r"begr(ue|ü)nd|ausrei(ss|ß)er|einfache(n)? (suche|ratensuche)|aus zeitgr(ue|ü)nden", re.I)


# ------------------------------------------------------------------ Hilfen
# Module mit Ergebnisdatei, die weder in der Agenda noch im Feld 'modul' vorkommen (Ergebnis liegt vor)
ALT_ZUSATZ = {"clifford_diagnose", "kodierung", "hardware", "sre", "schuesse", "ideenschmiede", "clifford"}


def altmodule() -> set:
    """Module, die schon in Ergebnisdateien (Feld 'modul'/'abteilung') oder in der Agenda als
    erledigt auftauchen -> Ergebnisse sind festgeschrieben, nur berichten."""
    namen = set(ALT_ZUSATZ)
    try:
        agenda = json.loads((INSTITUT / "agenda.json").read_text(encoding="utf-8"))
        for p in agenda.get("punkte", []):
            if p.get("modul") and p.get("status") == "erledigt":
                namen.add(p["modul"].split(".")[-1])
    except Exception:
        pass
    for f in (INSTITUT / "ergebnisse").glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("modul"), str):
            namen.add(d["modul"].split(".")[-1])
    return namen


def mess_seeds_register() -> dict:
    """seed -> Liste (id) aus INSTITUT/seeds.json (zweck messung)."""
    try:
        reg = json.loads((INSTITUT / "seeds.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict = {}
    for e in reg.get("eintraege", []):
        if e.get("zweck") != "messung":
            continue
        for a, b in e["seeds"]:
            for s in range(a, b + 1):
                out.setdefault(s, set()).add(e["id"])
    return out


class Konstanten:
    """Wertet einfache Seed-Ausdruecke aus (Zahlen, Tupel/Listen, range, tuple, Namen auf Modulebene)."""

    def __init__(self, baum: ast.Module):
        self.tab: dict = {}
        for n in baum.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self.tab[n.targets[0].id] = n.value

    def wert(self, n, tiefe=0):
        if tiefe > 8:
            raise ValueError
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, (ast.Tuple, ast.List)):
            return [self.wert(x, tiefe + 1) for x in n.elts]
        if isinstance(n, ast.Name) and n.id in self.tab:
            return self.wert(self.tab[n.id], tiefe + 1)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == "range":
                a = [self.wert(x, tiefe + 1) for x in n.args]
                return list(range(*a))
            if n.func.id in ("tuple", "list", "sorted") and len(n.args) == 1:
                return list(self.wert(n.args[0], tiefe + 1))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return list(self.wert(n.left, tiefe + 1)) + list(self.wert(n.right, tiefe + 1))
        if isinstance(n, ast.Subscript):
            basis = self.wert(n.value, tiefe + 1)
            sl = n.slice
            if isinstance(sl, ast.Slice):
                lo = self.wert(sl.lower, tiefe + 1) if sl.lower else None
                hi = self.wert(sl.upper, tiefe + 1) if sl.upper else None
                return list(basis)[lo:hi]
        raise ValueError

    def seeds(self, n) -> set | None:
        try:
            w = self.wert(n)
        except Exception:
            return None
        if isinstance(w, int):
            return {w}
        flach = set()
        for x in w:
            if isinstance(x, int):
                flach.add(x)
            else:
                return None
        return flach


def ausnahmen_im_quelltext(quelle: str) -> set:
    return {m.group(1) for m in re.finditer(r"#\s*lint:\s*ok\s+([\w-]+)", quelle)}


# ------------------------------------------------------------------ Pruefungen
def pruefe_datei(pfad: Path, mess: dict) -> list:
    quelle = pfad.read_text(encoding="utf-8")
    try:
        baum = ast.parse(quelle)
    except SyntaxError as ex:
        return [(ex.lineno or 1, "syntax", f"Syntaxfehler: {ex.msg}")]
    zeilen = quelle.splitlines()
    ok = ausnahmen_im_quelltext(quelle)
    w: list = []

    def add(zeile, klasse, text):
        if klasse not in ok:
            w.append((zeile, klasse, text))

    def zeile_von(regex, default=1):
        for i, z in enumerate(zeilen, 1):
            if re.search(regex, z):
                return i
        return default

    funktionen = {n.name: n for n in ast.walk(baum) if isinstance(n, ast.FunctionDef)}
    hat_lauf = "lauf" in funktionen or any(re.match(r"lauf_?\w*$", n) for n in funktionen)

    # --- blas-threads
    pool_direkt = re.search(r"\bPool\(|ProcessPoolExecutor|get_context\(", quelle)
    nutzt_parallel = re.search(r"\bparallel\(", quelle) and re.search(r"from labor\.gemeinsam import[^\n]*\bparallel|gemeinsam\.parallel|import gemeinsam", quelle)
    if pool_direkt and not nutzt_parallel:
        kind_eins = re.search(r"set_num_threads\(\s*1\s*\)|initializer\s*=\s*_init_arbeiter", quelle)
        gesetzt = all(v in quelle for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) and kind_eins
        omp_nur = re.search(r"OMP_NUM_THREADS['\"]?\]?\s*(=|,)\s*['\"]1['\"]", quelle)
        if not (gesetzt or (omp_nur and kind_eins)):
            add(zeile_von(r"\bPool\(|get_context\("), "blas-threads",
                "Prozess-Pool ohne OMP/OPENBLAS/MKL_NUM_THREADS=1 und torch.set_num_threads(1) im Kindprozess")

    # --- ratensuche
    tun_seeds = None
    for n in baum.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and re.fullmatch(r"TUNING_SEEDS?|TUNING", n.targets[0].id):
            tun_seeds = Konstanten(baum).seeds(n.value)
    for fn in funktionen.values():
        if fn.name.startswith("lauf"):
            for a, d in zip(fn.args.args[-len(fn.args.defaults):] if fn.args.defaults else [], fn.args.defaults):
                if "tuning" in a.arg:
                    s = Konstanten(baum).seeds(d)
                    if s is not None and tun_seeds is None:
                        tun_seeds = s
    sucht_rate = re.search(r"(?i)\b(lr_?gitter|gitter|raten?gitter|GITTER)\b", quelle) and re.search(r"(?i)\blr\b|\brate\b|lernrate", quelle)
    nutzt_stabil = re.search(r"gittersuche_stabil|waehle_rate", quelle)
    if sucht_rate and not nutzt_stabil and (tun_seeds is None or len(tun_seeds) <= 2):
        if not BEGRUENDUNG.search(quelle):
            n_txt = "unbekannt" if tun_seeds is None else str(len(tun_seeds))
            add(zeile_von(r"(?i)gitter"), "ratensuche",
                f"Lernratengitter-Suche (Tuning-Seeds: {n_txt}) ohne gittersuche_stabil/waehle_rate und ohne Begruendungskommentar")

    # --- arm-erbt-kopf
    erbt = [i for i, z in enumerate(zeilen, 1)
            if re.search(r"deepcopy\(|load_state_dict\(", z) and re.search(r"(?i)kopf|head|voll|basis", z)]
    if erbt and not re.search(r"(?i)frisch\w*\s+kopf|eigene\w*\s+(lern)?rate|kopf\s+frisch|erbt\s+nicht|fresh head", quelle):
        add(erbt[0], "arm-erbt-kopf",
            "Zustand/Kopf eines anderen Arms wird uebernommen (deepcopy/load_state_dict) - Kontrollen brauchen denselben Start "
            "oder eigene Lernratensuche; im Quelltext kein Hinweis auf frischen Kopf/eigene Rate")

    # --- urteil-kein-string
    def nicht_string(v) -> bool:
        if isinstance(v, ast.Constant):
            return not isinstance(v.value, str)
        if isinstance(v, (ast.Compare, ast.BoolOp, ast.List, ast.Tuple, ast.Set, ast.ListComp)):
            return True
        if isinstance(v, ast.UnaryOp):
            return True
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("bool", "float", "int", "round"):
            return True
        if isinstance(v, ast.IfExp):
            return nicht_string(v.body) or nicht_string(v.orelse)
        return False

    for n in ast.walk(baum):
        # urteil["X"] = ...   /  urteil = {...}  /  "urteil": {...}
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "urteil" \
                        and nicht_string(n.value):
                    add(n.lineno, "urteil-kein-string", "urteil[...] bekommt keinen String (Urteile sind Strings: 'H1: ...')")
                if isinstance(t, ast.Name) and t.id == "urteil" and isinstance(n.value, ast.Dict):
                    for k, v in zip(n.value.keys, n.value.values):
                        if nicht_string(v):
                            add(v.lineno, "urteil-kein-string", f"urteil[{ast.unparse(k) if k else '**'}] ist kein String")
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value == "urteil" and isinstance(v, ast.Dict):
                    for k2, v2 in zip(v.keys, v.values):
                        if nicht_string(v2):
                            add(v2.lineno, "urteil-kein-string", f"urteil[{ast.unparse(k2) if k2 else '**'}] ist kein String")

    # --- keine-wiederaufnahme
    if hat_lauf and re.search(r"(?i)seed", quelle):
        lange = re.search(r"\bparallel\(|\bPool\(|for\s+\w*seed\w*\s+in\b|for\s+s\s+in\s+seeds", quelle)
        if lange and not re.search(r"(?i)zwischen|wiederaufnahme|wieder aufnehm|resume|checkpoint", quelle):
            add(zeile_von(r"def lauf"), "keine-wiederaufnahme",
                "lauf() rechnet ueber Seeds, aber im Quelltext ist keine Wiederaufnahme/Zwischendatei je Seed erkennbar")

    # --- kontrolle-ohne-matrixcheck
    kontrolle = re.search(r"(?i)hadamard|identit|kontrollarm|\"kontrolle|zufall_clifford|ohne_schaltung", quelle)
    if kontrolle and "vortest" in funktionen:
        vt = ast.get_source_segment(quelle, funktionen["vortest"]) or ""
        # auch Hilfsfunktionen, die der Vortest aufruft (_vortest_*)
        for nm, f in funktionen.items():
            if nm.startswith("_vortest") or nm.startswith("_pruefe"):
                vt += ast.get_source_segment(quelle, f) or ""
        if not re.search(r"(?i)matrix|nonzero|count_nonzero|entartet|allclose|isclose|torch\.eye|np\.eye|permutation", vt):
            add(funktionen["vortest"].lineno, "kontrolle-ohne-matrixcheck",
                "Kontrollarm erkannt, aber der Vortest prueft keine Matrix/Entartung (z.B. Hadamard bei gerader Lagenzahl = Identitaet)")

    # --- doppelte-definition
    gesehen: dict = {}
    for n in baum.body:
        if isinstance(n, ast.FunctionDef):
            if n.name in gesehen:
                add(n.lineno, "doppelte-definition", f"{n.name}() steht schon in Zeile {gesehen[n.name]} - die fruehere Fassung ist tot")
            gesehen[n.name] = n.lineno

    # --- kein-vortest
    if hat_lauf and "vortest" not in funktionen and pfad.parent.name == "abteilungen":
        add(1, "kein-vortest", "Modul hat lauf(), aber keinen vortest() (Pflicht vor jedem Lauf)")

    # --- rauchtest-messseed
    if mess:
        konst = Konstanten(baum)
        modname = pfad.stem
        for n in ast.walk(baum):
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "schnell":
                for m in ast.walk(ast.Module(body=n.body, type_ignores=[])):
                    ziele_werte = []
                    if isinstance(m, ast.Assign):
                        if isinstance(m.targets[0], ast.Tuple) and isinstance(m.value, ast.Tuple):
                            ziele_werte = list(zip(m.targets[0].elts, m.value.elts))
                        else:
                            ziele_werte = [(m.targets[0], m.value)]
                    for ziel, wert in ziele_werte:
                        if isinstance(ziel, ast.Name) and re.search(r"(?i)seed", ziel.id) and "tuning" not in ziel.id.lower():
                            sd = konst.seeds(wert)
                            if sd:
                                treffer = {s: mess[s] for s in sd if s in mess}
                                if treffer:
                                    ids = sorted({i for v in treffer.values() for i in v})
                                    add(m.lineno, "rauchtest-messseed",
                                        f"Rauchtest (if schnell) nutzt Mess-Seeds {sorted(treffer)[:6]} (Mess-IDs: {', '.join(ids)}) - "
                                        "Rauchtests nur auf Tuning-/Entwurfs-Seeds")
    return sorted(w)


def sammle(pfade) -> list:
    out = []
    for p in pfade:
        p = Path(p)
        if p.is_dir():
            out += sorted(x for x in p.glob("*.py") if x.name != "__init__.py")
        elif p.suffix == ".py":
            out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pfade", nargs="*")
    ap.add_argument("--streng", action="store_true", help="Exit 1, wenn NEUE Module (ohne Ergebnisse) Warnungen haben")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--zusammenfassung", action="store_true", help="nur Zaehlung je Klasse")
    a = ap.parse_args(argv)
    dateien = sammle(a.pfade or SUCHPFADE)
    alt = altmodule()
    mess = mess_seeds_register()
    bericht = []
    for f in dateien:
        ws = pruefe_datei(f, mess)
        bericht.append({"datei": str(f.relative_to(WURZEL)) if f.is_absolute() and WURZEL in f.parents else str(f),
                        "altmodul": f.stem in alt, "warnungen": [{"zeile": z, "klasse": k, "text": t} for z, k, t in ws]})
    neu_mit_w = [b for b in bericht if b["warnungen"] and not b["altmodul"]]
    if a.json:
        print(json.dumps(bericht, ensure_ascii=False, indent=1))
    else:
        zaehl = Counter()
        zaehl_neu = Counter()
        for b in bericht:
            for w in b["warnungen"]:
                zaehl[w["klasse"]] += 1
                if not b["altmodul"]:
                    zaehl_neu[w["klasse"]] += 1
            if b["warnungen"] and not a.zusammenfassung:
                print(f"{b['datei']}  [{'Altmodul - nur Bericht' if b['altmodul'] else 'NEUES Modul'}]")
                for w in b["warnungen"]:
                    print(f"  {w['zeile']}: [{w['klasse']}] {w['text']}")
        print(f"\nFEHLERBUCH-Lint: {len(dateien)} Dateien, {sum(zaehl.values())} Warnungen "
              f"({sum(zaehl_neu.values())} in neuen Modulen, {len(neu_mit_w)} neue Module betroffen).")
        for k, n in zaehl.most_common():
            print(f"  {k:30s} {n:3d}  (neu: {zaehl_neu.get(k, 0)})")
    if a.streng and neu_mit_w:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
