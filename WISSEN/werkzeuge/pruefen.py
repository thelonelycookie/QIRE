#!/usr/bin/env python3
"""Prüft WISSEN/quellen.json: Schema, Duplikate, Sättigung, Index. Nur Standardbibliothek."""
import argparse, json, re, sys
from datetime import date
from pathlib import Path

WISSEN = Path(__file__).resolve().parent.parent
FELDER = {
    "K1": "Gewichtsquantisierung PTQ/QAT", "K2": "Aktivierungs- und KV-Quantisierung",
    "K3": "Rotationen (Hadamard, Butterfly, gelernt)", "K4": "Gitter/Codebücher/Vektorquantisierung",
    "K5": "Tensornetze (MPO/MPS/TT)", "K6": "Quanten-inspirierte ML",
    "K7": "Strukturierte Matrizen", "K8": "Niedrigbit-/ternäres Training",
    "N1": "Skalengesetze", "N2": "Effiziente Architekturen", "N3": "Distillation",
    "N4": "Sparsity und Pruning", "N5": "Energie/Hardware", "N6": "Zufalls-/Seed-Gewichte",
    "N7": "Aktivierungsfunktionen", "N8": "Rate-Distortion/Informationsgrenzen",
    "F1": "Quantenfehlerkorrektur/Codierungstheorie", "F2": "Renormierungsgruppe/Multiskala",
    "F3": "Signalverarbeitung/Kompression", "F4": "Neurobiologie", "F5": "Offene Quantensysteme",
    "F6": "Physik-Simulation",
}
QB = {"direkt", "analogie", "keiner"}
SW = {"stuetzt", "widerspricht", "neutral", "unklar"}
ID_RE = re.compile(r"^(arXiv:\d{4}\.\d{4,5}|doi:.+|url:.+)$")
DATUM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def _ist_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def pruefe_eintrag(e, nr):
    """Gibt Liste von Fehlertexten für einen Eintrag zurück."""
    f = []
    if not isinstance(e, dict):
        return [f"#{nr}: kein Objekt"]
    w = f"#{nr} ({e.get('id', '?')})"
    pflicht = ["id", "titel", "jahr", "autoren", "feld", "kernaussage", "zahlen",
               "baseline_fuer_qire", "quantenbezug", "bezug_qire_befund",
               "stuetzt_widerspricht", "volltext_gelesen", "gefunden_am", "welle", "herkunft"]
    for k in pflicht:
        if k not in e:
            f.append(f"{w}: Pflichtfeld '{k}' fehlt")
    if "id" in e and not (isinstance(e["id"], str) and ID_RE.match(e["id"])):
        f.append(f"{w}: id ungültig (arXiv:xxxx.xxxxx, doi:… oder url:…)")
    for k in ("titel", "autoren", "kernaussage", "herkunft"):
        if k in e and not isinstance(e[k], str):
            f.append(f"{w}: '{k}' muss str sein")
    if "titel" in e and isinstance(e["titel"], str) and not e["titel"].strip():
        f.append(f"{w}: titel leer")
    if "jahr" in e and e["jahr"] is not None and not _ist_int(e["jahr"]):
        f.append(f"{w}: jahr muss int oder null sein")
    if "feld" in e:
        if not isinstance(e["feld"], list) or not e["feld"]:
            f.append(f"{w}: feld muss nichtleere Liste sein")
        else:
            for x in e["feld"]:
                if x not in FELDER:
                    f.append(f"{w}: Feld-ID '{x}' ungültig")
    if "zahlen" in e:
        if not isinstance(e["zahlen"], list):
            f.append(f"{w}: zahlen muss Liste sein")
        else:
            for z in e["zahlen"]:
                if not (isinstance(z, dict) and {"wert", "kontext", "geprueft"} <= set(z)
                        and isinstance(z["geprueft"], bool)):
                    f.append(f"{w}: zahlen-Eintrag braucht wert, kontext, geprueft(bool)")
    if "baseline_fuer_qire" in e and e["baseline_fuer_qire"] is not None \
            and not isinstance(e["baseline_fuer_qire"], str):
        f.append(f"{w}: baseline_fuer_qire muss str oder null sein")
    if "quantenbezug" in e and e["quantenbezug"] not in QB:
        f.append(f"{w}: quantenbezug '{e['quantenbezug']}' ungültig")
    if "bezug_qire_befund" in e and not (isinstance(e["bezug_qire_befund"], list)
                                         and all(isinstance(x, str) for x in e["bezug_qire_befund"])):
        f.append(f"{w}: bezug_qire_befund muss Liste von str sein")
    if "stuetzt_widerspricht" in e and e["stuetzt_widerspricht"] not in SW:
        f.append(f"{w}: stuetzt_widerspricht '{e['stuetzt_widerspricht']}' ungültig")
    if "volltext_gelesen" in e and not isinstance(e["volltext_gelesen"], bool):
        f.append(f"{w}: volltext_gelesen muss bool sein")
    if "gefunden_am" in e:
        ok = isinstance(e["gefunden_am"], str) and DATUM_RE.match(e["gefunden_am"])
        if ok:
            try:
                date.fromisoformat(e["gefunden_am"])
            except ValueError:
                ok = False
        if not ok:
            f.append(f"{w}: gefunden_am muss YYYY-MM-DD sein")
    if "welle" in e and not _ist_int(e["welle"]):
        f.append(f"{w}: welle muss int sein")
    return f


def finde_duplikate(eintraege):
    f, ids, titel = [], {}, {}
    for i, e in enumerate(eintraege):
        if not isinstance(e, dict):
            continue
        if isinstance(e.get("id"), str):
            if e["id"] in ids:
                f.append(f"Duplikat-id '{e['id']}': #{ids[e['id']]} und #{i}")
            else:
                ids[e["id"]] = i
        if isinstance(e.get("titel"), str) and norm(e["titel"]):
            n = norm(e["titel"])
            if n in titel and e.get("id") != eintraege[titel[n]].get("id"):
                f.append(f"Ähnlicher Titel: #{titel[n]} und #{i} ('{e['titel']}')")
            elif n in titel:
                pass  # gleiche id schon als Duplikat-id gemeldet
            else:
                titel[n] = i
    return f


def pruefe_alles(eintraege):
    if not isinstance(eintraege, list):
        return ["quellen.json muss eine Liste sein"]
    f = []
    for i, e in enumerate(eintraege):
        f += pruefe_eintrag(e, i)
    return f + finde_duplikate(eintraege)


def gueltige(eintraege):
    return [e for e in eintraege if isinstance(e, dict)]


def zaehle_je_feld(eintraege):
    c = {k: 0 for k in FELDER}
    for e in gueltige(eintraege):
        fl = e.get("feld")
        for x in set(fl) if isinstance(fl, list) else []:
            if x in c:
                c[x] += 1
    return c


def saettigung(eintraege, welle):
    """Je Feld: (neu in Welle, gesamt, Anteil|None)."""
    erg = {}
    for k in FELDER:
        mit = [e for e in gueltige(eintraege) if isinstance(e.get("feld"), list) and k in e["feld"]]
        neu = sum(1 for e in mit if e.get("welle") == welle)
        erg[k] = (neu, len(mit), neu / len(mit) if mit else None)
    return erg


def volltext_je_feld(eintraege):
    c = {k: 0 for k in FELDER}
    for e in gueltige(eintraege):
        if e.get("volltext_gelesen") is True and isinstance(e.get("feld"), list):
            for x in set(e["feld"]):
                if x in c:
                    c[x] += 1
    return c


def anteile(eintraege):
    es = gueltige(eintraege)
    vt = sum(1 for e in es if e.get("volltext_gelesen") is True)
    z = [z for e in es if isinstance(e.get("zahlen"), list) for z in e["zahlen"] if isinstance(z, dict)]
    zg = sum(1 for x in z if x.get("geprueft") is True)
    return (vt, len(es)), (zg, len(z))


def schreibe_index(eintraege, pfad):
    zj, vj = zaehle_je_feld(eintraege), volltext_je_feld(eintraege)
    wellen = [e["welle"] for e in gueltige(eintraege) if _ist_int(e.get("welle"))]
    letzte = max(wellen) if wellen else None
    sat = saettigung(eintraege, letzte) if letzte is not None else {}
    z = ["# INDEX der Wissensbasis", "",
         f"Automatisch erzeugt von `werkzeuge/pruefen.py --index` am {date.today().isoformat()}. "
         f"Einträge gesamt: {len(gueltige(eintraege))}. Letzte Welle: {letzte if letzte is not None else '–'}.", "",
         "| Feld | Name | Einträge | davon Volltext | Sättigung letzte Welle |", "|---|---|---|---|---|"]
    for k, n in FELDER.items():
        a = sat.get(k, (0, 0, None))[2]
        s = "–" if a is None else f"{a:.0%}" + (" (gesättigt-verdächtig)" if a < 0.10 else "")
        z.append(f"| {k} | {n} | {zj[k]} | {vj[k]} | {s} |")
    Path(pfad).write_text("\n".join(z) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quellen", default=str(WISSEN / "quellen.json"))
    p.add_argument("--saettigung", type=int, metavar="WELLE")
    p.add_argument("--index", action="store_true", help="WISSEN/INDEX.md neu schreiben")
    p.add_argument("--index-pfad", default=str(WISSEN / "INDEX.md"))
    a = p.parse_args(argv)
    try:
        daten = json.loads(Path(a.quellen).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        print(f"FEHLER: {a.quellen} nicht lesbar: {ex}")
        return 1
    fehler = pruefe_alles(daten)
    print(f"Fehler: {len(fehler)}")
    for x in fehler:
        print("  -", x)
    if isinstance(daten, list):
        print("Einträge je Feld:")
        for k, n in zaehle_je_feld(daten).items():
            print(f"  {k} {FELDER[k]}: {n}")
        (vt, n), (zg, zn) = anteile(daten)
        print(f"Volltext gelesen: {vt}/{n}" + (f" ({vt/n:.0%})" if n else ""))
        print(f"Geprüfte Zahlen: {zg}/{zn}" + (f" ({zg/zn:.0%})" if zn else ""))
        if a.saettigung is not None:
            print(f"Sättigung Welle {a.saettigung} (neu/alle des Feldes):")
            for k, (neu, ges, an) in saettigung(daten, a.saettigung).items():
                if an is None:
                    print(f"  {k}: keine Einträge")
                else:
                    print(f"  {k}: {neu}/{ges} = {an:.0%}" + ("  < 10 %  GESÄTTIGT-VERDÄCHTIG" if an < 0.10 else ""))
        if a.index:
            schreibe_index(daten, a.index_pfad)
            print(f"Index geschrieben: {a.index_pfad}")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
