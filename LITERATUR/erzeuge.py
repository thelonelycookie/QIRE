# -*- coding: utf-8 -*-
"""Erzeugt literatur.bib, notizen.md und neuheit_index.json aus quellen_daten.py (R5.3).

    python LITERATUR/erzeuge.py            # schreibt die drei Dateien neu
    python LITERATUR/erzeuge.py --pruefen  # nur Konsistenz + Abdeckung gegen das Repo pruefen

Der Lauf liest zusaetzlich INSTITUT/ und A/sources/ (nur lesen), sucht dort alle arXiv-IDs und
meldet IDs, die in quellen_daten.py noch fehlen (Abdeckungspruefung). Kein Netzwerk noetig.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

HIER = Path(__file__).resolve().parent
WURZEL = HIER.parent
sys.path.insert(0, str(HIER))
import quellen_daten as Q  # noqa: E402

STAND = "2026-10-14"


# ------------------------------------------------------------------ Fundstellen im Repo
def scanne_fundstellen():
    """arXiv-ID -> Liste (Datei, Zeile). Nur Zeilen, die 'arxiv' enthalten (case-insensitive)."""
    dateien = [f for f in glob.glob(str(WURZEL / "INSTITUT" / "**" / "*.md"), recursive=True)
               if "zwischenstand" not in f and "/ergebnisse/" not in f]
    dateien += glob.glob(str(WURZEL / "A" / "sources" / "literature_cards" / "*.md"))
    pat = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d)?|quant-ph/\d{7}")
    out = defaultdict(list)
    for f in sorted(dateien):
        rel = str(Path(f).relative_to(WURZEL))
        for i, zeile in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
            if not re.search("arxiv", zeile, re.I):
                continue
            for m in pat.finditer(zeile):
                k = m.group(0)
                k = re.sub(r"v\d$", "", k)
                if re.match(r"\d{4}\.", k) and not (1 <= int(k[2:4]) <= 12 and 7 <= int(k[:2]) <= 27):
                    continue
                if (rel, i) not in out[k]:
                    out[k].append((rel, i))
    return out


def esc(s):
    """BibTeX/LaTeX-sicher: Sonderzeichen maskieren (Backslashes entfernen)."""
    s = (s or "").replace("\\", "")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("^", r"\textasciicircum{}"),
                 ("~", r"\textasciitilde{}"), ("|", r"\textbar{}"), ("<", r"\textless{}"), (">", r"\textgreater{}"),
                 ("{", "("), ("}", ")")):
        s = s.replace(a, b)
    return s


def bib_autoren(a):
    """'Bowles, Ahmed, Schuld' -> '{Bowles} and {Ahmed} and {Schuld}' (nur Nachnamen: Vornamen stehen nicht in den Quellen)."""
    if not a:
        return "{unvollstaendig}"
    a = re.sub(r"\([^)]*\)", "", a)
    andere = bool(re.search(r"et al\.?|u\. ?a\.?", a))
    a = re.sub(r"et al\.?|u\. ?a\.?", "", a)
    teile = [t.strip() for t in re.split(r",|;| und | and |/", a) if t.strip()]
    s = " and ".join("{" + esc(t) + "}" for t in teile) or "{unvollstaendig}"
    return s + (" and others" if andere else "")


def unvollstaendig(e):
    fehlt = []
    if not e["autoren"]:
        fehlt.append("Autoren")
    if not e["jahr"]:
        fehlt.append("Jahr")
    if e["titel"].startswith("(") or e["key"] in Q.TITEL_AUS_WISSEN:
        fehlt.append("Titel unverifiziert" if e["key"] in Q.TITEL_AUS_WISSEN else "Titel")
    if not e["venue"]:
        fehlt.append("Venue")
    if not e["arxiv"]:
        fehlt.append("arXiv-ID")
    return fehlt


def pruefe(fund):
    keys = [e["key"] for e in Q.ALLE]
    doppelt = {k for k in keys if keys.count(k) > 1}
    ids = [e["arxiv"] for e in Q.ALLE if e["arxiv"]]
    doppelt_id = {i for i in ids if ids.count(i) > 1}
    kset = set(keys)
    fehler = []
    if doppelt:
        fehler.append(f"doppelte Keys: {sorted(doppelt)}")
    if doppelt_id:
        fehler.append(f"doppelte arXiv-IDs: {sorted(doppelt_id)}")
    for p in Q.PRUEFUNGEN:
        for a in p["arbeiten"]:
            if a not in kset:
                fehler.append(f"{p['id']}: unbekannter Key {a}")
    fehlend = sorted(i for i in fund if i not in set(ids))
    return fehler, fehlend


# ------------------------------------------------------------------ bib
def schreibe_bib(fund):
    z = ["% LITERATUR/literatur.bib -- erzeugt von LITERATUR/erzeuge.py (nicht von Hand aendern;",
         "% Daten stehen in quellen_daten.py). Stand " + STAND + ".",
         "% Jeder Eintrag mit 'unvollstaendig' im note-Feld hat Luecken: nichts wurde erfunden.",
         "% Quelle aller Angaben: INSTITUT/recherche/*, NEUHEIT_BORN_MAGIE.md (nur Suchauszuege/Abstracts,",
         "% sofern status nicht 'Volltext'). Vor dem Zitieren in der Arbeit pruefen.",
         "% Autoren: nur Nachnamen (Vornamen stehen in den Quellen nicht); 'and others' = Liste im Original laenger.", ""]
    for e in sorted(Q.ALLE, key=lambda x: x["key"]):
        luecken = unvollstaendig(e)
        felder = OrderedDict()
        felder["title"] = esc(e["titel"])
        felder["author"] = bib_autoren(e["autoren"])
        if e["jahr"]:
            felder["year"] = str(e["jahr"])
        if e["venue"]:
            felder["journal" if re.search(r"Phys|Nat|Quantum \d|PRL|PRX|J\. ACM|JMLR", e["venue"] or "") else "howpublished"] = esc(e["venue"])
        if e["arxiv"]:
            felder["eprint"] = e["arxiv"]
            felder["archivePrefix"] = "arXiv"
            felder["url"] = "https://arxiv.org/abs/" + e["arxiv"]
        n = f"Lesestatus: {e['status']}."
        if luecken:
            n += " unvollstaendig: " + ", ".join(luecken) + "."
        if e["key"] in Q.TITEL_AUS_WISSEN or "aus Wissen" in e["kern"]:
            n += " Titel/Angaben teils vom Katalogisierer ergaenzt, nicht in der Recherche belegt - vor Zitat pruefen."
        n += " QIRE-Bezug: " + e["qire"] + "."
        felder["note"] = esc(n)
        typ = "article" if "journal" in felder else "misc"
        z.append("@" + typ + "{" + e["key"] + ",")
        for k, v in felder.items():
            z.append(f"  {k} = {{{v}}}," if k != "author" else f"  author = {{{v}}},")
        z.append("}")
        z.append("")
    (HIER / "literatur.bib").write_text("\n".join(z), encoding="utf-8")


# ------------------------------------------------------------------ neuheit_index.json
def baue_index(fund):
    nach_key = {e["key"]: e for e in Q.ALLE}
    arbeiten = OrderedDict()
    for e in sorted(Q.ALLE, key=lambda x: x["key"]):
        arbeiten[e["key"]] = OrderedDict(
            arxiv=e["arxiv"], autoren=e["autoren"], jahr=e["jahr"], titel=e["titel"], status=e["status"],
            kernaussage=e["kern"], qire_bezug=e["qire"], neuheit_hinweis=e["neuheit"],
            begriffe=e["begriffe"], unvollstaendig=unvollstaendig(e),
            fundstellen=[f"{d}:{n}" for d, n in fund.get(e["arxiv"], [])[:6]] if e["arxiv"] else [])
    begriffe = OrderedDict()
    pr_von = defaultdict(list)
    for p in Q.PRUEFUNGEN:
        for b in p["begriffe"]:
            pr_von[b].append(p["id"])
    alle_b = sorted({b for e in Q.ALLE for b in e["begriffe"]} | {b for p in Q.PRUEFUNGEN for b in p["begriffe"]})
    rang = {"bekannt": 0, "teilweise_bekannt": 1, "gegenbefund": 2, "neue_verbindung": 3, "nicht_gefunden": 4, "offen": 5}
    for b in alle_b:
        ks = sorted(k for k, e in nach_key.items() if b in e["begriffe"])
        urteile = sorted({p["urteil"] for p in Q.PRUEFUNGEN if b in p["begriffe"]}, key=lambda u: rang[u])
        beschr, syn = Q.BEGRIFFE.get(b, (b.replace("-", " "), []))
        begriffe[b] = OrderedDict(beschreibung=beschr, synonyme=syn, arbeiten=ks,
                                  pruefungen=pr_von.get(b, []),
                                  neuheitsurteile=urteile,
                                  status=("ungeprueft" if not urteile else urteile[0]))
    pruef = [OrderedDict(id=p["id"], datum=p["datum"], datei=p["datei"], aussage=p["aussage"], urteil=p["urteil"],
                         begriffe=p["begriffe"], arbeiten=p["arbeiten"], hinweis=p["hinweis"], grenzen=p["grenzen"])
             for p in Q.PRUEFUNGEN]
    return OrderedDict(
        format="qire-neuheit-index", version=1, stand=STAND,
        beschreibung="Begriffe -> Arbeiten fuer Lueckenkarte und Neuheitsfilter der Ideen-KI. Siehe LITERATUR/notizen.md (Kopf).",
        urteil_skala=OrderedDict(
            bekannt="Idee/Baustein ist publiziert; nicht als eigene Neuheit behaupten.",
            teilweise_bekannt="Bausteine bekannt, die konkrete Kombination/Richtung nicht gefunden.",
            nicht_gefunden="In begrenzter Suche nichts Vergleichbares gefunden: Neuheit plausibel, nicht bewiesen.",
            neue_verbindung="Bausteine einzeln bekannt, direkter Vergleich/Gesetz nicht gefunden.",
            gegenbefund="Literatur widerspricht oder relativiert (muss diskutiert werden).",
            offen="Pruefung duenn/unvollstaendig; nicht als bestaetigt behandeln."),
        hinweis_ehrlich="'nicht_gefunden' heisst: in wenigen Websuchen (meist nur Abstracts) nichts gefunden - kein Beweis der Abwesenheit.",
        begriffe=begriffe, neuheitspruefungen=pruef, arbeiten=arbeiten)


# ------------------------------------------------------------------ notizen.md
KOPF = """# Literaturnotizen QIRE (R5.3)

*Stand {stand}. Erzeugt von `LITERATUR/erzeuge.py` aus `LITERATUR/quellen_daten.py` (nicht von Hand ändern, sondern dort ergänzen und neu erzeugen).*
*Quelle ausschließlich Repo-Texte: `INSTITUT/recherche/*`, `INSTITUT/NEUHEIT_BORN_MAGIE.md`, Handover/LABORBUCH, `A/sources/literature_cards/`. Nichts wurde im Internet nachgeschlagen. Wo etwas fehlt, steht **unvollständig**; Titel mit `*` hat der Katalogisierer ergänzt und sind vor dem Zitieren zu prüfen.*

**Ehrliche Grenze:** Das meiste sind Suchauszüge/Abstracts. Nur Einträge mit Lesestatus „Volltext“ hat das Institut im PDF gelesen. „nicht gefunden“ in einer Neuheitsprüfung heißt: in wenigen Suchen nichts gefunden, kein Beweis.

## Dateien
| Datei | Inhalt |
|---|---|
| `literatur.bib` | BibTeX aller {n} Arbeiten (Key = `nachname+jahr+kurz`), `note`-Feld mit Lesestatus, Lücken, QIRE-Bezug |
| `notizen.md` | diese Datei: Prüfungsübersicht, Begriffe, alle Arbeiten (alphabetisch nach Key) |
| `neuheit_index.json` | maschinenlesbarer Index für die Ideen-KI (Format unten) |
| `quellen_daten.py` / `erzeuge.py` | Datenquelle und Generator (`python LITERATUR/erzeuge.py [--pruefen]`) |

## Format `neuheit_index.json` (für Lückenkarte und Neuheitsfilter der Ideen-KI)
```json
{{
  "format": "qire-neuheit-index", "version": 1, "stand": "...",
  "urteil_skala": {{ "<urteil>": "Bedeutung", ... }},
  "begriffe": {{
    "<begriff-schluessel>": {{
      "beschreibung": "Klartext", "synonyme": ["..."],
      "arbeiten": ["<bib-key>", ...],        // alle Arbeiten, die den Begriff berühren
      "pruefungen": ["N-B", ...],            // Neuheitsprüfungen, in denen der Begriff vorkommt
      "neuheitsurteile": ["teilweise_bekannt", ...],
      "status": "bekannt|teilweise_bekannt|gegenbefund|neue_verbindung|nicht_gefunden|offen|ungeprueft"
    }}
  }},
  "neuheitspruefungen": [ {{ "id", "datum", "datei", "aussage", "urteil", "begriffe", "arbeiten", "hinweis", "grenzen" }} ],
  "arbeiten": {{ "<bib-key>": {{ "arxiv", "autoren", "jahr", "titel", "status", "kernaussage", "qire_bezug", "neuheit_hinweis", "begriffe", "unvollstaendig", "fundstellen" }} }}
}}
```
**Benutzung durch die Ideen-KI**
- *Neuheitsfilter:* Idee auf Begriffe abbilden (Schlüssel oder `synonyme`). Hat ein Begriff `status` `bekannt` oder `teilweise_bekannt`, dann `arbeiten` und `neuheitspruefungen[].hinweis` in die Karte als Abgrenzung schreiben und die Neuheitsbewertung deckeln (kein „erstmals“). Bei `nicht_gefunden`/`neue_verbindung` normale Bewertung, aber „plausibel“ statt „neu“. `gegenbefund` muss in der Karte diskutiert werden. Schon geprüfte Aussagen (`neuheitspruefungen[].aussage`) nicht erneut als Recherche beauftragen.
- *Lückenkarte:* Zellen = Paare von Begriffen. Ein Paar ist **gefüllt**, wenn eine Arbeit beide Begriffe in `begriffe` trägt; **leer**, wenn keine Arbeit beide trägt (Bonus für leere Felder, aber „leer“ = nur „in unseren Notizen nicht belegt“). Begriffe mit wenigen Arbeiten und `status: ungeprueft` sind Suchlücken der Literaturarbeit, nicht Neuland.
- Der Index wird von `erzeuge.py` neu erzeugt; nie von Hand editieren.

Lesestatus: **Volltext** (PDF gelesen) · **Abstract** (Abstract gelesen) · **Auszug** (Suchtreffer/Snippet) · **nur Titel**.

## 1. Neuheitsprüfungen des Instituts (nicht wiederholen)
| ID | Datum | Aussage | Urteil | Datei |
|---|---|---|---|---|
"""


def schreibe_notizen(fund, index):
    z = [KOPF.format(stand=STAND, n=len(Q.ALLE)).rstrip("\n")]
    for p in Q.PRUEFUNGEN:
        z.append(f"| {p['id']} | {p['datum']} | {p['aussage'].replace('|', chr(92) + '|')} | **{p['urteil']}** | `{p['datei']}` |")
    z.append("")
    z.append("### Details je Prüfung")
    for p in Q.PRUEFUNGEN:
        z += ["", f"**{p['id']}: {p['aussage']}** — Urteil: *{p['urteil']}* ({p['datum']})",
              f"- Hinweis: {p['hinweis']}", f"- Grenzen: {p['grenzen']}",
              "- Arbeiten: " + ", ".join(f"`{a}`" for a in p["arbeiten"])]
    z += ["", "## 2. Begriffe (Kurzfassung des Index)", "", "| Begriff | Status | Arbeiten | Prüfungen |", "|---|---|---|---|"]
    for b, d in index["begriffe"].items():
        z.append(f"| `{b}` | {d['status']} | {len(d['arbeiten'])} | {', '.join(d['pruefungen']) or '–'} |")
    z += ["", "## 3. Arbeiten (alphabetisch nach Key)", "",
          "Felder: arXiv · Autoren · Jahr · Lesestatus · Kernaussage · Bezug zu QIRE-IDs · Neuheitsprüfung-Ergebnis · Fundstellen (Datei:Zeile).", ""]
    for e in sorted(Q.ALLE, key=lambda x: x["key"]):
        luecken = unvollstaendig(e)
        titel = e["titel"] + (" \\*" if e["key"] in Q.TITEL_AUS_WISSEN else "")
        z.append(f"### `{e['key']}` — {titel}")
        z.append(f"- **arXiv:** {e['arxiv'] or 'unvollständig (keine ID)'} · **Autoren:** {e['autoren'] or 'unvollständig'} · **Jahr:** {e['jahr'] or 'unvollständig'}"
                 + (f" · **Venue:** {e['venue']}" if e["venue"] else ""))
        z.append(f"- **Lesestatus:** {e['status']}" + (f" · **unvollständig:** {', '.join(luecken)}" if luecken else ""))
        z.append(f"- **Kernaussage:** {e['kern']}")
        z.append(f"- **QIRE-Bezug:** {e['qire']}")
        z.append(f"- **Neuheitsprüfung:** {e['neuheit']}")
        fs = fund.get(e["arxiv"], []) if e["arxiv"] else []
        if fs:
            z.append("- **Fundstellen:** " + ", ".join(f"`{d}:{n}`" for d, n in fs[:5]) + (" …" if len(fs) > 5 else ""))
        z.append("- **Begriffe:** " + ", ".join(e["begriffe"]))
        z.append("")
    (HIER / "notizen.md").write_text("\n".join(z), encoding="utf-8")


def main():
    fund = scanne_fundstellen()
    fehler, fehlend = pruefe(fund)
    for f in fehler:
        print("FEHLER:", f)
    print(f"Arbeiten: {len(Q.ALLE)} · Prüfungen: {len(Q.PRUEFUNGEN)} · IDs im Repo gefunden: {len(fund)}")
    if fehlend:
        print("IDs im Repo, aber nicht in quellen_daten.py:", ", ".join(fehlend))
    if "--pruefen" in sys.argv:
        sys.exit(1 if fehler else 0)
    if fehler:
        sys.exit(1)
    index = baue_index(fund)
    (HIER / "neuheit_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    schreibe_bib(fund)
    schreibe_notizen(fund, index)
    print("geschrieben: literatur.bib, notizen.md, neuheit_index.json")


if __name__ == "__main__":
    main()
