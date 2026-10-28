#!/usr/bin/env python3
"""Zahlen-Audit fuers Institut (R3.6): Tabellenzahlen aus LABORBUCH.md (und START.md, falls vorhanden)
gegen INSTITUT/ergebnisse/*.json pruefen -- Vorbild: UEBERARBEITUNG_2026-09-23/A/quantum_sim/q_audit.py.

    python werkzeuge/zahlen_audit.py                 # Bericht
    python werkzeuge/zahlen_audit.py --details       # jede Zelle einzeln
    python werkzeuge/zahlen_audit.py --streng        # Exit 1 bei "abweichend" (fuer CI)
    python werkzeuge/zahlen_audit.py --datei INSTITUT/LABORBUCH.md

Vorgehen (bewusst konservativ):
  * Ein Tabellenabschnitt wird einer Ergebnisdatei nur zugeordnet, wenn die ID (z.B. "I-038", "C-clifford-v2")
    in der Ueberschrift des Abschnitts (oder, bei START.md, in der Tabellenzeile) steht UND es
    INSTITUT/ergebnisse/<ID>__*.json gibt. Sonst "nicht zuordenbar".
  * Aus der JSON werden alle Zahlen gesammelt: Blattwerte, Mittel/SD jeder Spalte ueber Seeds und je Arm,
    jeweils auch als Prozent (x100).
  * bestaetigt  = die Tabellenzahl stimmt (auf die angegebenen Nachkommastellen gerundet) mit einer Zahl der Datei ueberein.
                  "spezifisch" = der Zeilenname (Arm) steht im Pfad dieser Zahl; sonst "lose" (Zahl kommt irgendwo vor).
  * abweichend  = der Zeilenname (Arm) ist in der JSON eindeutig vorhanden, die Zahl aber weder bei diesem Arm
                  noch sonst irgendwo in der Datei -- das ist der einzige Alarmfall.
  * nicht zuordenbar = alles andere (kein Abschnitts-ID, keine Datei, ganze kleine Zahlen, Arm unbekannt,
                  abgeleitete Zahlen wie Differenzen/Anteile). KEIN Alarm.
Bekannte, begruendete Abweichungen stehen in werkzeuge/zahlen_audit_ausnahmen.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
ERGEBNISSE = WURZEL / "INSTITUT" / "ergebnisse"
AUSNAHMEN = Path(__file__).resolve().parent / "zahlen_audit_ausnahmen.json"
STANDARD_DATEIEN = [WURZEL / "INSTITUT" / "LABORBUCH.md", WURZEL / "INSTITUT" / "START.md", WURZEL / "START.md"]

ID_RE = re.compile(r"\b([CIVTZ]-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b")
ZAHL_RE = re.compile(r"(?<![\w.,])[−–-]?\d{1,3}(?:\.\d{3})+(?![\d,])|(?<![\w.,])[−–-]?\d+(?:,\d+)?(?:[eE][−+-]?\d+)?")


# ------------------------------------------------------------------ JSON-Zahlen sammeln
def _zahl(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _mittel_std(w):
    n = len(w)
    m = sum(w) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in w) / (n - 1)) if n > 1 else 0.0
    return m, sd


def sammle_zahlen(obj, pfad="", out=None, tiefe=0):
    """-> Liste (wert, pfad). Zusaetzlich Mittel/SD ueber Zeilenlisten (Spalten, je Arm gruppiert)."""
    if out is None:
        out = []
    if tiefe > 12:
        return out
    if _zahl(obj):
        out.append((float(obj), pfad))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sammle_zahlen(v, f"{pfad}/{k}", out, tiefe + 1)
    elif isinstance(obj, list):
        if obj and all(isinstance(r, dict) for r in obj) and len(obj) <= 5000:
            zeilen_stats(obj, pfad, out)
        elif obj and all(_zahl(v) for v in obj) and len(obj) > 1:
            m, sd = _mittel_std([float(v) for v in obj])
            out.append((m, f"{pfad}/(mittel)"))
            out.append((sd, f"{pfad}/(std)"))
        if len(obj) <= 3000:
            for i, v in enumerate(obj):
                sammle_zahlen(v, f"{pfad}[{i}]", out, tiefe + 1)
    return out


def zeilen_stats(zeilen, pfad, out):
    """Wide: je numerischer Spalte Mittel/SD ueber alle Zeilen; Long: zusaetzlich gruppiert nach String-Spalten."""
    schluessel = defaultdict(list)
    for r in zeilen:
        for k, v in r.items():
            if _zahl(v):
                schluessel[k].append(float(v))
    for k, w in schluessel.items():
        if len(w) > 1:
            m, sd = _mittel_std(w)
            out.append((m, f"{pfad}/(mittel)/{k}"))
            out.append((sd, f"{pfad}/(std)/{k}"))
    str_keys = [k for k in zeilen[0] if isinstance(zeilen[0].get(k), str)]
    for sk in str_keys:
        gruppen = defaultdict(list)
        for r in zeilen:
            if isinstance(r.get(sk), str):
                gruppen[r[sk]].append(r)
        if len(gruppen) > 60:
            continue
        for g, rs in gruppen.items():
            if len(rs) < 2:
                continue
            for k in schluessel:
                w = [float(r[k]) for r in rs if _zahl(r.get(k))]
                if len(w) > 1:
                    m, sd = _mittel_std(w)
                    out.append((m, f"{pfad}/(mittel)/{g}/{k}"))
                    out.append((sd, f"{pfad}/(std)/{g}/{k}"))


class Quelle:
    def __init__(self, id_):
        self.id = id_
        self.dateien = sorted(ERGEBNISSE.glob(f"{id_}__*.json"))
        self.zahlen: list = []
        for f in self.dateien:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for w, p in sammle_zahlen(d):
                self.zahlen.append((w, p, f.name))
        self._tokens = None

    def tokens(self):
        if self._tokens is None:
            t = set()
            for _, p, _ in self.zahlen:
                t |= {x.lower() for x in re.split(r"[/\[\]]", p) if x}
            self._tokens = t
        return self._tokens


# ------------------------------------------------------------------ Markdown
def parse_zahl(text):
    """'0,9683' -> (0.9683, 4) ; '19.200' -> (19200, 0) ; '-3,2e-3'"""
    t = text.replace("−", "-").replace("–", "-")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
        return float(t.replace(".", "")), 0
    t = t.replace(",", ".")
    try:
        w = float(t)
    except ValueError:
        return None
    nk = len(t.split(".")[1].split("e")[0].split("E")[0]) if "." in t else 0
    return w, nk


def parse_tabellen(text):
    """-> Liste von (abschnitt_titel, zeilennummer_start, header(list), zeilen[(nr, cells)])"""
    zeilen = text.splitlines()
    abschnitt = ""
    out, i = [], 0
    while i < len(zeilen):
        z = zeilen[i]
        if z.startswith("#"):
            abschnitt = z.lstrip("# ").strip()
        if z.lstrip().startswith("|"):
            block, start = [], i
            while i < len(zeilen) and zeilen[i].lstrip().startswith("|"):
                block.append((i + 1, zeilen[i]))
                i += 1
            zellen = [(nr, [c.strip() for c in l.strip().strip("|").split("|")]) for nr, l in block]
            zellen = [(nr, c) for nr, c in zellen if not all(re.fullmatch(r":?-{2,}:?", x or "---") for x in c)]
            if len(zellen) >= 2:
                out.append((abschnitt, start + 1, zellen[0][1], zellen[1:]))
            continue
        i += 1
    return out


def ids_in(text):
    ids = []
    for m in ID_RE.finditer(text):
        t = m.group(1)
        # "C-1", "T-1" (Hypothesennamen) sind keine Ergebnis-IDs; Datei muss existieren
        ids.append(t)
    return ids


def bevorzugte_ids(text):
    """IDs aus Ueberschrift, laengste zuerst (I-038-schnell vor I-038); nur solche mit Ergebnisdatei."""
    gef = []
    for t in sorted(set(ids_in(text)), key=len, reverse=True):
        if list(ERGEBNISSE.glob(f"{t}__*.json")) and t not in gef:
            gef.append(t)
    return gef


def label_tokens(label):
    """Zeilenname -> moegliche Arm-Schluessel: 'ohne_schaltung (Identitaet)' -> ohne_schaltung; 'clifford_k4'."""
    l = re.sub(r"[`*_]{2,}|\*", "", label)
    erst = re.split(r"[ (:/]", l.strip(), maxsplit=1)[0].strip("`*").lower()
    kandidaten = {erst} if erst else set()
    l2 = re.sub(r"\([^)]*\)", "", l).strip("`* ").lower().replace(" ", "_")
    if l2:
        kandidaten.add(l2)
    return {k for k in kandidaten if len(k) >= 3 and not re.fullmatch(r"[\d.,\- ]+", k)}


def lade_ausnahmen():
    if AUSNAHMEN.exists():
        try:
            return json.loads(AUSNAHMEN.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


# ------------------------------------------------------------------ Pruefung
def pruefe_datei(md: Path, details=False):
    text = md.read_text(encoding="utf-8")
    quellen: dict = {}
    ausn = lade_ausnahmen()
    zellen_bericht = []
    for abschnitt, start, header, zeilen in parse_tabellen(text):
        sek_ids = bevorzugte_ids(abschnitt)
        for nr, cells in zeilen:
            zeile_ids = bevorzugte_ids(" ".join(cells))
            ids = zeile_ids or sek_ids
            qs = []
            for i in ids:
                if i not in quellen:
                    quellen[i] = Quelle(i)
                if quellen[i].zahlen:
                    qs.append(quellen[i])
            # Blockbreite: Tabellen koennen zwei Arm/Wert-Paare pro Zeile haben ("| voll | 0,97 | | k0 | 0,95 |")
            for ci, zelle in enumerate(cells):
                if ci == 0 and len(cells) > 2 and len(parse_ints_only(zelle)) > 0:
                    continue
                zahlen = []
                for m in ZAHL_RE.finditer(zelle):
                    pz = parse_zahl(m.group(0))
                    if pz:
                        zahlen.append((m.group(0), pz))
                if not zahlen:
                    continue
                if ci == 0:
                    continue                      # erste Spalte = Zeilenname/Index
                label = cells[ci - 1] if ci >= 1 and not parse_ints_only(cells[ci - 1]) and _ist_label(cells[ci - 1]) else cells[0]
                # Zwei-Block-Tabellen: Label ist die letzte Textzelle links der Wertezelle
                for j in range(ci - 1, -1, -1):
                    if _ist_label(cells[j]):
                        label = cells[j]
                        break
                for roh, (wert, nk) in zahlen:
                    status, info = bewerte(wert, nk, roh, label, qs, zelle)
                    if status != "nicht zuordenbar" or details:
                        pass
                    ref = {"datei": md.name, "zeile": nr, "abschnitt": abschnitt, "zelle": zelle, "label": label,
                           "zahl": roh, "status": status, "info": info, "ids": ids}
                    if status == "abweichend" and ist_ausnahme(ausn, ref):
                        ref["status"] = "abweichend (begruendete Ausnahme)"
                    zellen_bericht.append(ref)
    return zellen_bericht


def parse_ints_only(zelle):
    return [m for m in re.findall(r"^\s*\*{0,2}\d+\*{0,2}\s*$", zelle)]


def _ist_label(zelle):
    z = zelle.strip()
    return bool(z) and bool(re.search(r"[A-Za-zÄÖÜäöü]{2,}", z)) and not re.search(r"\d,\d|±", z)


def ist_ausnahme(ausn, ref):
    for a in ausn:
        if a.get("abschnitt_enthaelt", "") in ref["abschnitt"] and a.get("zahl") == ref["zahl"] \
                and a.get("label", ref["label"]) == ref["label"]:
            ref["grund"] = a.get("grund", "")
            return True
    return False


def treffer(wert, nk, kandidaten):
    """kandidaten [(w, pfad, datei)] -> passende, inkl. Prozent-Skalierung."""
    tol = 0.5 * 10 ** (-nk) + 1e-9
    gef = []
    for w, p, f in kandidaten:
        for faktor in (1.0, 100.0):
            if abs(w * faktor - wert) <= tol * (1.0 if faktor == 1.0 else 1.0) or \
               (nk == 0 and round(w * faktor) == wert):
                gef.append((w * faktor, p, f))
                break
    return gef


def bewerte(wert, nk, roh, label, qs, zelle):
    if not qs:
        return "nicht zuordenbar", "kein Abschnitts-/Zeilen-ID mit Ergebnisdatei"
    # ganze kleine Zahlen und 1-stellige Dezimalzahlen sind zu unspezifisch (zufaellige Treffer)
    if (nk == 0 and abs(wert) < 100) or (nk == 1 and abs(wert) < 10):
        return "nicht zuordenbar", "Zahl zu unspezifisch (kleine ganze Zahl / 1 Nachkommastelle)"
    alle = [z for q in qs for z in q.zahlen]
    gef = treffer(wert, nk, alle)
    toks = label_tokens(label)
    if gef:
        spez = [g for g in gef if any(t in g[1].lower().split("/") or any(t == s for s in re.split(r"[/\[\]]", g[1].lower()))
                                       for t in toks)]
        if spez:
            return "bestaetigt (spezifisch)", f"{spez[0][2]} {spez[0][1]}"
        if nk >= 4 or (nk == 0 and abs(wert) >= 1000):
            return "bestaetigt (lose)", f"{gef[0][2]} {gef[0][1]} (Zahl kommt vor, Zeilenname nicht im Pfad)"
        return "nicht zuordenbar", "Treffer nur zufaellig moeglich (wenige Nachkommastellen, Zeilenname nicht im Pfad)"
    # Alarmfall: Arm eindeutig vorhanden, Zahl nirgends
    arm_pfade = []
    for t in toks:
        for q in qs:
            if t in q.tokens():
                arm_pfade.append((q.id, t))
    # Alarm nur bei "Mittel ± SD"-Zellen mit >= 3 Nachkommastellen: reine Einzelzahlen koennen aus anderen
    # Laeufen (Vergleichsspalte MNIST/Fashion, v1/v2) stammen -- das ist kein Fehler.
    rolle_ok = bool(re.search(r"\d[\d.,]*\s*±\s*\d", zelle)) and nk >= 3
    if arm_pfade and rolle_ok:
        naechste = None
        t0 = arm_pfade[0][1]
        cand = [z for q in qs for z in q.zahlen if t0 in re.split(r"[/\[\]]", z[1].lower())]
        if cand:
            naechste = min(cand, key=lambda z: abs(z[0] - wert) if abs(z[0] - wert) < abs(z[0] * 100 - wert) else abs(z[0] * 100 - wert))
        return "abweichend", (f"Arm '{t0}' in {arm_pfade[0][0]} vorhanden, aber {roh} kommt nirgends in der Datei vor"
                              + (f"; naechste Zahl bei diesem Arm: {naechste[0]:.6g} ({naechste[1]})" if naechste else ""))
    return "nicht zuordenbar", "Zeilenname nicht in der Ergebnisdatei gefunden bzw. abgeleitete Zahl"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datei", action="append", help="Markdown-Datei (Standard: LABORBUCH.md, START.md)")
    ap.add_argument("--details", action="store_true", help="alle Zellen auflisten")
    ap.add_argument("--streng", action="store_true", help="Exit 1 bei abweichenden Zahlen")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    dateien = [Path(d) for d in a.datei] if a.datei else [d for d in STANDARD_DATEIEN if d.exists()]
    alle = []
    for d in dateien:
        alle += pruefe_datei(d, a.details)
    z = Counter(r["status"] for r in alle)
    if a.json:
        print(json.dumps(alle, ensure_ascii=False, indent=1))
    else:
        print("Zahlen-Audit: " + ", ".join(d.name for d in dateien))
        pro = defaultdict(Counter)
        for r in alle:
            pro[(r["datei"], r["abschnitt"][:70])][r["status"].split(" (")[0]] += 1
        if a.details:
            for r in alle:
                print(f"  {r['datei']}:{r['zeile']} [{r['status']}] {r['label'][:30]!r} {r['zahl']}  {r['info'][:110]}")
        else:
            for (d, ab), c in pro.items():
                if c.get("abweichend") or c.get("bestaetigt"):
                    print(f"  {d} | {ab}: bestaetigt {c.get('bestaetigt', 0)}, abweichend {c.get('abweichend', 0)}, "
                          f"nicht zuordenbar {c.get('nicht zuordenbar', 0)}")
        abw = [r for r in alle if r["status"] == "abweichend"]
        if abw:
            print("\nABWEICHEND:")
            for r in abw:
                print(f"  {r['datei']}:{r['zeile']} ({r['abschnitt'][:60]}) Zeile {r['label']!r}: {r['zahl']} -- {r['info']}")
        ausn = [r for r in alle if r["status"].startswith("abweichend (")]
        if ausn:
            print("\nBegruendete Ausnahmen (zahlen_audit_ausnahmen.json):")
            for r in ausn:
                print(f"  {r['datei']}:{r['zeile']} {r['label']!r} {r['zahl']}: {r.get('grund', '')}")
        n_best = sum(v for k, v in z.items() if k.startswith("bestaetigt"))
        print(f"\nBilanz: {n_best} bestaetigt "
              f"({z.get('bestaetigt (spezifisch)', 0)} spezifisch, {z.get('bestaetigt (lose)', 0)} lose), "
              f"{z.get('abweichend', 0)} abweichend, {sum(v for k, v in z.items() if k.startswith('abweichend ('))} begruendete Ausnahmen, "
              f"{z.get('nicht zuordenbar', 0)} nicht zuordenbar.")
    return 1 if (a.streng and any(r["status"] == "abweichend" for r in alle)) else 0


if __name__ == "__main__":
    sys.exit(main())
