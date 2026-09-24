"""Abteilung Ideenschmiede -- erfindet Ideen und leitet sie weiter.

Drei Stufen, jede mit hartem Tor (nichts wird weitergeleitet, was ein Feld leer hat):

  roh ──(Papierpruefung)──> geprueft ──(Neuheitspruefung)──> neu ──(Tor)──> weitergeleitet
                 │                              │
                 └──> verworfen (Grund)  <──────┘   (bekannt / widerlegt -- bleibt stehen!)

1. ERZEUGEN (lokal, 0 Tokens): morphologischer Kasten -- Quantenkonzept x
   KI-Engpass x Denkzug (Uebertragen, Umkehren, Diskretisieren, Messbar machen,
   ...). Konzepte, die in Befunden des Instituts schon getragen haben (H1),
   werden haeufiger gezogen; schon gezogene Kombinationen nie doppelt.
   Das ergibt ROHLINGE -- Fragen, keine Ideen. Ein Mensch oder Claude macht
   aus einem Rohling eine Idee (oder verwirft ihn in einer Zeile).
2. PRUEFEN: Papierpruefung (widerlegt die Rechnung die Idee schon?),
   Neuheitspruefung (Websuche, Quellen eintragen).
3. WEITERLEITEN: Nur Karten mit Abbruchkriterium, billigstem Test, Kontrolle
   und Zielabteilung. Die Karte erscheint dann in agenda.json als Punkt der
   Zielabteilung (Status „idee“); der Direktor zeigt sie in STAND.md.

    python -m abteilungen.ideenschmiede neu 8       # 8 Rohlinge ziehen
    python -m abteilungen.ideenschmiede liste       # Trichter anzeigen
    python -m abteilungen.ideenschmiede leiten I-007  # Tor pruefen, weiterleiten
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
IDEEN = WURZEL / "ideen.json"
AGENDA = WURZEL / "agenda.json"

# Quantenkonzept -> (Kernmechanismus, Abteilung, die es pruefen wuerde)
KONZEPTE = {
    "Magie / Clifford-Hierarchie": ("nur Nicht-Clifford-Anteile kosten; Clifford ist klassisch billig", "Clifford"),
    "Stabilisator-Renyi-Entropie": ("billig messbares Mass fuer Magie", "Clifford"),
    "Verschraenkungsentropie / Flaechengesetz": ("Korrelationen begrenzen den noetigen Bondrang", "Verschraenkung"),
    "Schmidt-Zerlegung / Operator-Verschraenkung": ("Rang ueber einen Schnitt = Kompressionsgrenze", "Verschraenkung"),
    "Born-Regel": ("quadratische Auslese liefert Nichtlinearitaet ohne Parameter", "Auslese"),
    "Klassische Schatten": ("viele Observablen aus wenigen Zufallsmessungen", "Auslese"),
    "Holevo-Schranke": ("n Qubits tragen hoechstens n Bit auslesbare Information", "Auslese"),
    "Zero-Noise-Extrapolation": ("Rauschen verstaerken und auf null extrapolieren", "Theorie"),
    "Probabilistische Fehlerauslöschung": ("Quasi-Wahrscheinlichkeiten heben Fehler im Mittel auf", "Theorie"),
    "Quanten-Zeno-Effekt": ("haeufiges Messen friert Dynamik ein", "Theorie"),
    "Adiabatisches Theorem": ("langsame Aenderung haelt den Grundzustand", "Training"),
    "Imaginaerzeit-Evolution": ("exp(-tH) filtert zum Grundzustand, Normierung inklusive", "Training"),
    "Trotter-Zerlegung": ("exp(A+B) als Produkt kleiner Schritte", "Clifford"),
    "Matchgates / freie Fermionen": ("klassisch simulierbar trotz Verschraenkung", "Verschraenkung"),
    "Stabilisator-Codes": ("wenige strukturierte Pruefungen schuetzen viel Information", "Theorie"),
    "Renormierungsgruppe / MERA": ("Grobkoernung Schicht fuer Schicht", "Verschraenkung"),
    "Dekohaerenz / Lindblad-Dynamik": ("Umgebung loescht Nichtdiagonalelemente", "Training"),
    "Quantenlauf (Quantum Walk)": ("Interferenz verbreitet schneller als Diffusion", "Theorie"),
    "Twirling / Symmetrie-Mittelung": ("Mitteln ueber eine Gruppe projiziert auf Invarianten", "Theorie"),
    "Lieb-Robinson-Lichtkegel": ("Information breitet sich lokal begrenzt aus", "Theorie"),
    "Messbasierte Quantenrechnung": ("Rechnung = Verschraenkung vorab + Messungen", "Theorie"),
    "Pauli-Propagation (Heisenberg-Bild)": ("Observablen rueckwaerts statt Zustaende vorwaerts", "Auslese"),
    "Magische Zustaende / Injektion": ("Nicht-Clifford-Anteil als Ressource von aussen einspeisen", "Clifford"),
    "Katalyse": ("ein Hilfszustand ermoeglicht etwas und bleibt erhalten", "Theorie"),
}
ENGPAESSE = [
    "Multiplikationen/Energie je Inferenz", "Speicher der Gewichte", "Einbettungstabelle des Chatbots",
    "Aufmerksamkeit O(L^2)", "Trainingsschritte bis Ziel", "benoetigte Datenmenge", "Bits je Gewicht",
    "Latenz / Fruehausstieg", "KV-Cache beim Generieren", "Hyperparameter-Suche",
]
DENKZUEGE = [
    "die Voraussetzung der Methode mit uebertraegt und zuerst prueft",
    "es umkehrt (was dort teuer ist, hier billig macht)",
    "kontinuierliche Parameter auf ein Gitter zwingt",
    "es zur Messgroesse macht (vorhersagt statt ausprobiert)",
    "es mit einem Befund des Instituts kombiniert",
    "es vom Inferenz- ins Trainingsproblem verschiebt (oder zurueck)",
    "nur einen kleinen Teil quantisch macht und den Rest klassisch laesst",
    "den Grenzfall sucht, in dem es exakt wird",
]
PFLICHT = ["frage", "quantenkern", "quantenstufe", "papierpruefung", "neuheit", "abbruch",
           "billigster_test", "kontrolle", "abteilung"]

# Maltes Grundsatz (24.09.): Das Institut arbeitet quantenphysikalisch, nicht mit
# „optimierten klassischen Algorithmen“. Jede Karte nennt ihren Quantenkern:
#   Q3 = laeuft als echte Quantenschaltung (Qiskit/IBM-Hardware moeglich), der
#        Effekt ist eine Quantenressource (Magie, Verschraenkung, Messung ...)
#   Q2 = exakte Quantenmechanik (Zustandsvektor, Unitaere, Born-Regel), im
#        Simulator gerechnet, auf Hardware uebertragbar
#   Q1 = nur mathematische Analogie (z. B. Tensor-Train auf klassischen Gewichten)
# Q1 wird NICHT aussortiert (Malte, 24.09.), bekommt aber die niedrigste
# Prioritaet: Q3 -> Prioritaet 1, Q2 -> 2, Q1 -> 3. Der Direktor rechnet
# offene Punkte in dieser Reihenfolge.
PRIORITAET = {3: 1, 2: 2, 1: 3}


def lade():
    return json.loads(IDEEN.read_text(encoding="utf-8")) if IDEEN.exists() else {"karten": []}


def sichere(d):
    IDEEN.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def _befund_gewichte():
    """Abteilungen mit H1-Befunden bekommen mehr Rohlinge (Gewinner vertiefen)."""
    gew = {}
    try:
        from direktor import neuestes_ergebnis
        for p in json.loads(AGENDA.read_text(encoding="utf-8"))["punkte"]:
            e = neuestes_ergebnis(p["id"])
            treffer = sum(v.startswith("H1") for v in (e or {}).get("urteil", {}).values())
            gew[p["abteilung"]] = gew.get(p["abteilung"], 0) + treffer
    except Exception:
        pass
    return gew


def neu(anzahl=8, seed=None):
    d = lade()
    schon = {(k.get("konzept"), k.get("engpass"), k.get("denkzug")) for k in d["karten"]}
    rng = random.Random(seed if seed is not None else time.time_ns())
    gew = _befund_gewichte()
    namen = list(KONZEPTE)
    # Gewinner-Abteilungen bevorzugen, aber schon oft gezogene Konzepte daempfen
    # (sonst zog die Schmiede am 24.09. fuenfmal „Trotter“).
    schon_konzept = {n: sum(k.get("konzept") == n for k in d["karten"]) for n in namen}
    w = [(1 + gew.get(KONZEPTE[n][1], 0)) / (1 + schon_konzept[n]) ** 2 for n in namen]
    neu_k = []
    for _ in range(anzahl * 50):
        if len(neu_k) == anzahl:
            break
        kz, ep, dz = rng.choices(namen, w)[0], rng.choice(ENGPAESSE), rng.choice(DENKZUEGE)
        if (kz, ep, dz) in schon:
            continue
        schon.add((kz, ep, dz))
        karte = {"id": f"I-{len(d['karten']) + 1:03d}", "status": "roh", "erzeugt": time.strftime("%Y-%m-%d"),
                 "konzept": kz, "engpass": ep, "denkzug": dz, "abteilung": KONZEPTE[kz][1],
                 "frage": f"Wie senkt „{kz}“ ({KONZEPTE[kz][0]}) den Engpass „{ep}“, wenn man {dz}?"}
        d["karten"].append(karte)
        neu_k.append(karte)
    sichere(d)
    for k in neu_k:
        print(k["id"], k["frage"])


def liste():
    d = lade()
    stufen = ["roh", "geprueft", "neu", "weitergeleitet", "verworfen"]
    print("Trichter:", " -> ".join(f"{s} {sum(k['status'] == s for k in d['karten'])}" for s in stufen))
    for k in d["karten"]:
        if k["status"] != "roh":
            print(f"  {k['id']} [{k['status']}] -> {k.get('abteilung')}: {k.get('titel', k['frage'])[:90]}")


def leiten(kid):
    """Das Tor: alle Pflichtfelder gefuellt, Neuheit nicht 'BEKANNT' -> in die Agenda."""
    d = lade()
    k = next(x for x in d["karten"] if x["id"] == kid)
    fehlt = [f for f in PFLICHT if not k.get(f)]
    if fehlt:
        print("Tor zu -- es fehlt:", ", ".join(fehlt))
        return False
    prio = PRIORITAET[int(str(k["quantenstufe"]).lstrip("Qq"))]
    if str(k["neuheit"]).upper().startswith("BEKANNT"):
        k["status"] = "verworfen"
        k["grund"] = "bekannt: " + k["neuheit"]
        sichere(d)
        print("Verworfen (bekannt). Bleibt im Trichter stehen.")
        return False
    agenda = json.loads(AGENDA.read_text(encoding="utf-8"))
    if not any(p["id"] == kid for p in agenda["punkte"]):
        agenda["punkte"].append({"id": kid, "abteilung": k["abteilung"], "status": "idee", "modul": None,
                                 "frage": k.get("titel", k["frage"]), "prioritaet": prio,
                                 "notiz": f"Aus der Ideenschmiede. Abbruch: {k['abbruch']} Test: {k['billigster_test']}"})
        AGENDA.write_text(json.dumps(agenda, indent=1, ensure_ascii=False), encoding="utf-8")
    k["status"] = "weitergeleitet"
    sichere(d)
    print(f"{kid} weitergeleitet an Abteilung {k['abteilung']} (Prioritaet {prio}).")
    return True


if __name__ == "__main__":
    sys.path.insert(0, str(WURZEL))
    a = sys.argv[1:] or ["liste"]
    if a[0] == "neu":
        neu(int(a[1]) if len(a) > 1 else 8)
    elif a[0] == "leiten":
        leiten(a[1])
    else:
        liste()
