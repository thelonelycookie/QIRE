#!/usr/bin/env python3
"""Bereinigung 0b: schwache Einträge in quellen.json ersetzen (arXiv-Metadaten + eigene Kurzfassung).

Eingaben (WISSEN/werkzeuge/daten_0b/):
  arxiv_neu.tsv    id|feld,feld|quantenbezug|kernaussage   (schwache arXiv-Einträge, von Hand aus Titel+Abstract)
  nichtarxiv.json  Ersatzdaten für schwache doi/url/bib-Einträge, oder {"entfernen": Grund}
  korrekturen.json Korrekturen von quantenbezug/feld bei nicht schwachen arXiv-Einträgen
Cache: WISSEN/volltexte/arxiv_abstracts.json (erzeugt von holen_0b.py).
Ausgaben: quellen.json (überschrieben), werkzeuge/entfernt_0b.json, werkzeuge/unbekannte_ids.json.
Lauf: python3 WISSEN/werkzeuge/holen_0b.py && python3 WISSEN/werkzeuge/bereinigen_0b.py
Idempotent: bereits bereinigte Einträge sind nicht mehr schwach.
"""
import json, re
from pathlib import Path

W = Path(__file__).resolve().parent.parent
D = W / "werkzeuge" / "daten_0b"
FRAG = ["Siehe ", "@article", "@inproceedings", "org/abs", "org/pdf", "org/html", "nicht näher", "nicht naeher", "Kontext aus"]


def schwach(e):
    t, k = e["titel"], e["kernaussage"]
    return ("nicht genannt" in t or t.startswith("(") or not t.strip() or t.startswith("|")
            or not e["feld"] or any(s in k for s in FRAG) or len(k) < 30)


def autoren_kurz(namen):
    nach = [n.split()[-1] if False else n for n in namen]
    return ", ".join(nach) if len(nach) <= 3 else ", ".join(nach[:3]) + " u. a."


ZAHL = re.compile(r"(\d[\d.,]*\s?(?:%|x\b|×|-fold|fold|bits?\b|qubits?\b|times\b|Mio|million|billion|B\b|M\b|k\b|GB|ms\b)|\b\d+(?:\.\d+)?\s?(?:x|×)\b)", re.I)


def zahlen_aus_abstract(abstract, vorhanden):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", abstract):
        m = ZAHL.search(s)
        if m and not re.search(r"arXiv|doi", s):
            wert = m.group(0).strip()
            if any(wert == z["wert"] for z in vorhanden + out):
                continue
            out.append({"wert": wert, "kontext": s[:220] + " [arXiv-Abstract]", "geprueft": False})
        if len(out) >= 2:
            break
    return out


def main():
    q = json.load(open(W / "quellen.json"))
    cache = json.load(open(W / "volltexte" / "arxiv_abstracts.json"))
    neu = {}
    for z in open(D / "arxiv_neu.tsv", encoding="utf-8"):
        if z.strip():
            i, f, qb, k = z.rstrip("\n").split("|", 3)
            neu[i] = (f.split(","), qb, k)
    nicht = json.load(open(D / "nichtarxiv.json"))
    korr = json.load(open(D / "korrekturen.json"))
    entfernt, unbekannt, ergebnis = [], [], []
    stat = {"bereinigt": 0, "korrigiert": 0, "kommentar": 0}
    for e in q:
        eid = e["id"]
        if eid.startswith("arXiv:"):
            a = eid[6:]
            m = cache.get(a)
            if m is None:
                unbekannt.append(eid)
                if "arXiv-ID nicht gefunden" not in e["kernaussage"]:
                    e["kernaussage"] = "arXiv-ID nicht gefunden (mögliche Tippfehler), Quelle prüfen. " + e["kernaussage"]
                ergebnis.append(e); continue
            war_schwach = schwach(e)
            if not e["autoren"].strip() or war_schwach:
                e["autoren"] = autoren_kurz(m["autoren"])
            if e["jahr"] is None or war_schwach:
                e["jahr"] = m["jahr"]
            if war_schwach:
                f, qb, k = neu[a]
                e["titel"], e["feld"], e["quantenbezug"], e["kernaussage"] = m["titel"], f, qb, k
                e["zahlen"] = e["zahlen"] + zahlen_aus_abstract(m["abstract"], e["zahlen"])
                stat["bereinigt"] += 1
            elif a in korr:
                for key, v in korr[a].items():
                    e[key] = v
                stat["korrigiert"] += 1
        elif schwach(e):
            n = nicht.get(eid)
            if n is None:
                e["kernaussage"] = "Inhalt unbekannt, Quelle prüfen. " + e["kernaussage"]
                stat["kommentar"] += 1
            elif "entfernen" in n:
                entfernt.append({"id": eid, "titel": e["titel"], "kernaussage": e["kernaussage"], "grund": n["entfernen"]})
                continue
            else:
                for key in ("titel", "jahr", "autoren", "feld", "quantenbezug", "kernaussage"):
                    if n.get(key) is not None:
                        e[key] = n[key]
                stat["bereinigt"] += 1
        ergebnis.append(e)
    # Duplikat bib:lu2023qasbench -> url:...lu23f.pdf (gleiche Arbeit QAS-Bench)
    bib = next((e for e in ergebnis if e["id"] == "bib:lu2023qasbench"), None)
    ziel = next((e for e in ergebnis if e["id"].endswith("lu23f.pdf")), None)
    if bib and ziel:
        ziel["bezug_qire_befund"] = sorted(set(ziel["bezug_qire_befund"]) | set(bib["bezug_qire_befund"]))
        ziel["herkunft"] = "; ".join(sorted(set(ziel["herkunft"].split("; ")) | set(bib["herkunft"].split("; "))))
        entfernt.append({"id": bib["id"], "titel": bib["titel"], "kernaussage": bib["kernaussage"],
                         "grund": "Duplikat von " + ziel["id"] + " (Befund-Bezüge und Herkunft übernommen)"})
        ergebnis.remove(bib)
    # Duplikate: nicht-arXiv-Eintrag mit gleichem Titel wie ein arXiv-Eintrag -> in arXiv-Eintrag zusammenführen
    def nt(t):
        return re.sub(r"[^a-z0-9]", "", t.lower())
    arx = {nt(e["titel"]): e for e in ergebnis if e["id"].startswith("arXiv:") or e["id"].startswith("bib:")}
    for e in list(ergebnis):
        z = arx.get(nt(e["titel"]))
        if z is not None and e is not z and not e["id"].startswith(("arXiv:", "bib:")):
            z["bezug_qire_befund"] = sorted(set(z["bezug_qire_befund"]) | set(e["bezug_qire_befund"]))
            z["herkunft"] = "; ".join(sorted(set(z["herkunft"].split("; ")) | set(e["herkunft"].split("; "))))
            z["volltext_gelesen"] = z["volltext_gelesen"] or e["volltext_gelesen"]
            known = {x["wert"] for x in z["zahlen"]}
            z["zahlen"] += [x for x in e["zahlen"] if x["wert"] not in known]
            entfernt.append({"id": e["id"], "titel": e["titel"], "kernaussage": e["kernaussage"],
                             "grund": "Duplikat von " + z["id"] + " (Bezüge, Herkunft, Zahlen übernommen)"})
            ergebnis.remove(e)
    json.dump(ergebnis, open(W / "quellen.json", "w"), ensure_ascii=False, indent=1)
    json.dump(entfernt, open(W / "werkzeuge" / "entfernt_0b.json", "w"), ensure_ascii=False, indent=1)
    json.dump(unbekannt, open(W / "werkzeuge" / "unbekannte_ids.json", "w"), ensure_ascii=False, indent=1)
    print(stat, "entfernt", len(entfernt), "unbekannt", len(unbekannt), "gesamt", len(ergebnis))


main()
