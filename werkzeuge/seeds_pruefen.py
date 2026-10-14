#!/usr/bin/env python3
"""Seed-Register pruefen (R3.4).

    python werkzeuge/seeds_pruefen.py              # Register + Ergebnisdateien pruefen
    python werkzeuge/seeds_pruefen.py --neu 2500 2519 [--tuning|--rauch]
                                                   # geplanten Block VOR der Vorregistrierung pruefen
    python werkzeuge/seeds_pruefen.py --still      # nur Fehler ausgeben

Prueft gegen INSTITUT/seeds.json:
  1. Blockregeln (Messung 0-899 / >=2000 mit Endziffern < 90; Tuning/Rauchtest 900-999 bzw. Endziffern >= 90)
  2. Doppelnutzung von Mess-Seeds zwischen verschiedenen IDs (gleicher Datensatz = Fehler,
     anderer Datensatz = Hinweis)
  3. Rauchtests/Tuning auf Mess-Seeds anderer IDs
  4. Jeder Seed in INSTITUT/ergebnisse/*.json ist im Register eingetragen
Bekannte Altlasten stehen in seeds.json unter "ausnahmen" MIT Begruendung: Sie werden
immer mit ausgegeben (nicht verschwiegen), loesen aber keinen Fehlerstatus aus.
Rueckgabewert 1 bei mindestens einem neuen Fehler.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
REGISTER = WURZEL / "INSTITUT" / "seeds.json"
ERGEBNISSE = WURZEL / "INSTITUT" / "ergebnisse"

MESS_KEYS = ("seeds", "seeds_mnist", "seeds_struktur", "seeds_suche")
TUNING_KEYS = ("tuning_seeds", "tuning_seed")


def expandiere(bereiche):
    return {s for a, b in bereiche for s in range(a, b + 1)}


def als_menge(v):
    """Seeds aus beliebig verschachtelten Ergebnisfeldern (Liste, Dict je Datensatz, Zahl)."""
    if isinstance(v, bool):
        return set()
    if isinstance(v, int):
        return {v}
    if isinstance(v, list):
        return {x for x in v if isinstance(x, int) and not isinstance(x, bool)}
    if isinstance(v, dict):
        return {s for x in v.values() for s in als_menge(x)}
    return set()


def blockregel_ok(zweck: str, s: int) -> bool:
    if zweck == "messung":
        return s < 900 or (s >= 2000 and s % 100 < 90)
    return 900 <= s <= 999 or (s >= 2000 and s % 100 >= 90)


def passt_ausnahme(ausnahmen, art, ids, seed):
    """Gibt die passende Ausnahme (mit Begruendung) zurueck oder None."""
    for a in ausnahmen:
        if a["art"] != art:
            continue
        lo, hi = a["seeds"]
        if not lo <= seed <= hi:
            continue
        pass_ids = a["ids"]
        if "*-schnell (Altlast)" in pass_ids and any(i.endswith("-schnell") for i in ids):
            return a
        if all(i in pass_ids for i in ids):
            return a
    return None


def pruefe_register(reg):
    fehler, hinweise, altlasten = [], [], {}
    eintraege = reg["eintraege"]
    ausn = reg["ausnahmen"]
    for e in eintraege:
        e["_menge"] = expandiere(e["seeds"])

    # 1. Blockregeln ("auswertung" hat keine eigenen Seeds, "-schnell"-Altlast siehe 3.)
    for e in eintraege:
        if e["zweck"] == "auswertung" or (e["zweck"] == "rauchtest" and e["id"].endswith("-schnell")):
            continue
        schlecht = sorted(s for s in e["_menge"] if not blockregel_ok(e["zweck"], s))
        offen = [s for s in schlecht if not passt_ausnahme(ausn, "blockregel", [e["id"]], s)]
        if offen:
            fehler.append(f"Blockregel verletzt: {e['id']} ({e['zweck']}) Seeds {offen[:6]}")
        elif schlecht:
            a = passt_ausnahme(ausn, "blockregel", [e["id"]], schlecht[0])
            altlasten.setdefault("blockregel", []).append(
                f"{e['id']} ({e['zweck']}, {schlecht[0]}-{schlecht[-1]}): {a['grund']}")

    # 2. Doppelnutzung Messung
    mess = [e for e in eintraege if e["zweck"] == "messung"]
    nutz = defaultdict(list)       # (seed, datensatz) -> ids
    nutz_seed = defaultdict(set)   # seed -> ids
    for e in mess:
        for s in e["_menge"]:
            nutz_seed[s].add(e["id"])
            for ds in e["datensatz"]:
                if e["id"] not in nutz[(s, ds)]:
                    nutz[(s, ds)].append(e["id"])
    gemeldet = set()
    for (s, ds), ids in sorted(nutz.items()):
        if len(ids) > 1:
            a = passt_ausnahme(ausn, "doppelnutzung_messung", ids, s)
            if a:
                gemeldet.add(("a", tuple(sorted(ids)), a["grund"]))
            else:
                fehler.append(f"Doppelnutzung Mess-Seed {s} auf {ds}: {', '.join(ids)}")
    for s, ids in sorted(nutz_seed.items()):
        if len(ids) > 1:
            ds_paare = [(i, j) for i in ids for j in ids if i < j]
            for i, j in ds_paare:
                di = {d for e in mess if e["id"] == i and s in e["_menge"] for d in e["datensatz"]}
                dj = {d for e in mess if e["id"] == j and s in e["_menge"] for d in e["datensatz"]}
                if di & dj:
                    continue
                a = passt_ausnahme(ausn, "seed_zahl_ueber_datensaetze", [i, j], s)
                if a:
                    gemeldet.add(("b", (i, j), a["grund"]))
                else:
                    hinweise.append(f"Seed {s} in {i} ({'/'.join(sorted(di))}) und {j} ({'/'.join(sorted(dj))}): "
                                    "andere Datensaetze, nicht als Ausnahme eingetragen")
    gruppen = defaultdict(set)
    for art, ids, grund in gemeldet:
        gruppen[("doppelnutzung" if art == "a" else "seed_zahl_ueber_datensaetze", grund)] |= set(ids)
    for (art, grund), ids in sorted(gruppen.items()):
        altlasten.setdefault(art, []).append(f"{' + '.join(sorted(ids))}: {grund}")

    # 3. Rauchtest/Tuning auf Mess-Seeds anderer IDs
    for e in eintraege:
        if e["zweck"] in ("messung", "auswertung"):
            continue
        basis = e["id"][:-len("-schnell")] if e["id"].endswith("-schnell") else e["id"]
        for s in sorted(e["_menge"]):
            gegner = sorted(nutz_seed.get(s, set()) - {basis})
            eigene = basis in nutz_seed.get(s, set())
            if not gegner and not eigene:
                continue
            a = passt_ausnahme(ausn, "rauchtest_auf_messseeds", [e["id"]], s) if e["zweck"] == "rauchtest" else None
            if a:
                altlasten.setdefault("rauchtest_auf_messseeds", {})
                altlasten["rauchtest_auf_messseeds"][e["id"]] = a["grund"]
            else:
                fehler.append(f"{e['zweck'].capitalize()} {e['id']} nutzt Mess-Seed {s} "
                              f"(Mess-ID: {', '.join(gegner or [basis])})")
    return fehler, hinweise, altlasten


def pruefe_ergebnisse(reg):
    """Jeder Seed in den Ergebnisdateien muss im Register stehen."""
    fehler, hinweise = [], []
    nach_id = defaultdict(lambda: {"messung": set(), "tuning": set(), "rauchtest": set(), "auswertung": set()})
    for e in reg["eintraege"]:
        nach_id[e["id"]][e["zweck"]] |= expandiere(e["seeds"])
    n = 0
    for f in sorted(ERGEBNISSE.glob("*.json")):
        if f.name.startswith("_"):
            continue
        id_ = f.name.split("__")[0]
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as ex:
            fehler.append(f"{f.name}: nicht lesbar ({ex})")
            continue
        if not isinstance(d, dict):
            continue
        mess = set().union(*(als_menge(d.get(k)) for k in MESS_KEYS))
        tun = set().union(*(als_menge(d.get(k)) for k in TUNING_KEYS))
        if not mess and not tun:
            continue
        n += 1
        basis = id_[:-len("-schnell")] if id_.endswith("-schnell") else id_
        reg_ = nach_id.get(id_) or nach_id.get(basis)
        if reg_ is None:
            fehler.append(f"{f.name}: ID {id_} fehlt im Register")
            continue
        if id_.endswith("-schnell"):
            erlaubt = reg_["rauchtest"] | reg_["tuning"] | nach_id.get(basis, reg_)["rauchtest"] | nach_id.get(basis, reg_)["tuning"] \
                | nach_id.get(id_, {"messung": set()})["messung"]
            fehlt = (mess | tun) - erlaubt - nach_id.get(basis, reg_)["messung"]
            auf_mess = (mess | tun) & nach_id.get(basis, reg_)["messung"]
            if fehlt:
                fehler.append(f"{f.name}: Rauchtest-Seeds {sorted(fehlt)[:6]} nicht als rauchtest/tuning registriert")
            if auf_mess:
                hinweise.append(f"{f.name}: Rauchtest laeuft auf Mess-Seeds {sorted(auf_mess)[:6]} der eigenen ID")
            continue
        for s in sorted(mess - reg_["messung"] - reg_["auswertung"]):
            fehler.append(f"{f.name}: Mess-Seed {s} nicht als messung von {id_} registriert")
            break
        for s in sorted(tun - reg_["tuning"] - reg_["rauchtest"]):
            fehler.append(f"{f.name}: Tuning-Seed {s} nicht als tuning von {id_} registriert")
            break
        extra = reg_["messung"] - mess
        if extra and mess:
            hinweise.append(f"{f.name}: im Register mehr Mess-Seeds als in der Datei ({sorted(extra)[:4]}...) - Modul-Teilmenge/Option?")
    return fehler, hinweise, n


def pruefe_neu(reg, von, bis, zweck):
    for e in reg["eintraege"]:
        e["_menge"] = expandiere(e["seeds"])
    wunsch = set(range(von, bis + 1))
    probleme = []
    schlecht = sorted(s for s in wunsch if not blockregel_ok(zweck, s))
    if schlecht:
        probleme.append(f"{zweck}: Seeds {schlecht[:8]} liegen im falschen Block (Messung <900 oder >=2000 mit Endziffern <90; "
                        "Tuning/Rauchtest 900-999 bzw. Endziffern >=90)")
    belegt = defaultdict(set)
    for e in reg["eintraege"]:
        for s in e["_menge"] & wunsch:
            belegt[s].add(f"{e['id']}[{e['zweck']}]")
    if zweck == "messung":
        for s, ids in sorted(belegt.items())[:10]:
            probleme.append(f"Seed {s} schon vergeben: {', '.join(sorted(ids))}")
    else:
        for s, ids in sorted(belegt.items()):
            if any("[messung]" in i for i in ids):
                probleme.append(f"Seed {s} ist Mess-Seed: {', '.join(sorted(ids))}")
    return probleme


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neu", nargs=2, type=int, metavar=("VON", "BIS"), help="geplanten Seed-Block pruefen")
    ap.add_argument("--tuning", action="store_true", help="mit --neu: Block ist Tuning")
    ap.add_argument("--rauch", action="store_true", help="mit --neu: Block ist Rauchtest")
    ap.add_argument("--still", action="store_true", help="nur Fehler ausgeben")
    ap.add_argument("--register", default=str(REGISTER))
    a = ap.parse_args(argv)
    reg = json.loads(Path(a.register).read_text(encoding="utf-8"))

    if a.neu:
        zweck = "tuning" if a.tuning else "rauchtest" if a.rauch else "messung"
        p = pruefe_neu(reg, a.neu[0], a.neu[1], zweck)
        if p:
            print("Seed-Block NICHT frei:")
            for x in p:
                print("  -", x)
            return 1
        print(f"Seeds {a.neu[0]}-{a.neu[1]} sind als {zweck} frei. Jetzt in INSTITUT/seeds.json eintragen und committen.")
        return 0

    f1, h1, alt = pruefe_register(reg)
    f2, h2, n = pruefe_ergebnisse(reg)
    fehler, hinweise = f1 + f2, h1 + h2
    print(f"Seed-Register: {len(reg['eintraege'])} Eintraege, {n} Ergebnisdateien mit Seeds geprueft.")
    if not a.still:
        if alt:
            print("\nBekannte Altlasten (in seeds.json begruendet, nicht verschwiegen):")
            for art, v in alt.items():
                print(f"  [{art}]")
                if isinstance(v, dict):
                    gruende = defaultdict(list)
                    for i, g in v.items():
                        gruende[g].append(i)
                    for g, ids in gruende.items():
                        print(f"    {len(ids)} Rauchtests ({', '.join(sorted(ids))})\n      Grund: {g}")
                else:
                    for x in v:
                        print("    -", x)
        if hinweise:
            print("\nHinweise:")
            for x in hinweise:
                print("  -", x)
    if fehler:
        print(f"\nFEHLER ({len(fehler)}):")
        for x in fehler:
            print("  -", x)
        return 1
    print("\nOK: keine neuen Fehler." if not a.still else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
