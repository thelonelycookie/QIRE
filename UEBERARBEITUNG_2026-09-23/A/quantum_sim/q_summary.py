from __future__ import annotations

"""Sammelbericht ueber die Q-Serie -- was gilt, was nicht, und was fehlt.

WARUM ES DAS GIBT
Die Q-Laeufe schreiben je ein eigenes Log und eine eigene Rohdatendatei. Bei
sechs Laeufen hintereinander heisst das: sechs Logs lesen, in jedem die
Auswertung suchen, und in jedem daran denken, dass **die Auflösungspruefung
darueber entscheidet, ob das Urteil ueberhaupt gilt**. Genau dieser letzte
Schritt geht beim Durchblaettern verloren -- und dann steht ein Urteil in der
Arbeit, dessen Aufbau seine eigene Frage nicht auflösen konnte. Das ist in
diesem Projekt schon zweimal passiert (B-004 und Q-009).

Dieses Werkzeug liest die gespeicherten Rohdaten (nicht die Logs, die sind
fluechtig), zieht aus jedem Lauf Urteil und Auflösungspruefung, und sortiert
alles in drei Toepfe:

  GILT              Urteil steht, Auflösungspruefung bestanden
  NICHT VERWERTBAR  Auflösungspruefung durchgefallen -> das Urteil darf nicht
                    zitiert werden, egal wie gut es klingt
  OFFEN             noch nicht gelaufen oder Auswertung fehlgeschlagen

ERGAENZT AM 23. SEPTEMBER
  * Das gespeicherte Urteil ist der Stand am Laufende. Was SPAETER dazu
    entschieden wurde -- Q-015 auf Fashion-MNIST zurueckgezogen, Q-003 auf
    Fashion nur mit Vorbehalt, die Q-003-Gleichwertigkeiten nicht zitierbar --
    stand nur im Tagebuch. Wer nur diesen Bericht las, zitierte die alten
    Zahlen. Jetzt steht es unter NACHTRAEGE und wird bei jedem Lauf gedruckt.
  * RANDLAGEN (Regel 1b) werden aus den Rohdaten gelesen, nicht abgeschrieben:
    fuer jeden Arm, ob seine Lernrate am unteren oder oberen Gitterrand sitzt,
    und fuer das Primaerpaar, in welche Richtung das den Befund verzerrt. Ein
    untertunierter KONTROLLARM blaeht den Abstand auf (fuer die Hypothese),
    ein untertunierter MECHANISMUSARM drueckt ihn (gegen sie).
  * Q-019 und T-000 fehlten in QUESTIONS (Regel aus CLAUDE.md §6).

Ergaenzt `qire_ai.audit`: der prueft Tagebuchzahlen gegen Rohdaten, dieser hier
prueft die Urteile gegen ihre eigenen Vorbedingungen.

Run: python -m quantum_sim.q_summary
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "experiments"

# Die Q-Serie mit der Frage, die jeder Lauf beantworten soll. Wird ein Lauf
# ergaenzt, gehoert er hierher -- sonst faellt er im Bericht unter "unbekannt".
QUESTIONS: Dict[str, str] = {
    "Q-003-replication": "Schlaegt die Schaltung ein klassisches Netz mit 848x mehr Parametern?",
    "Q-005": "Faellt die Gradientenvarianz exponentiell mit der Qubitzahl?",
    "Q-006": "Schlaegt Ising-Pruning die Magnitude-Heuristik?",
    "Q-007": "Auf wie vielen Aufgabentypen haelt der Vorteil?",
    "Q-009": "Halten 140 Parameter bei gleichem Zeitbudget gegen 222 Millionen?",
    "Q-010": "Schlaegt die Schaltung ihre faire klassische Nachbildung?",
    "Q-011": "Waechst der Vorteil der Orthogonalitaet mit der Tiefe?",
    "Q-012": "Laesst sich auch der Encoder quantenabgeleitet bauen?",
    "Q-013": "Haelt der Vorteil in einem Sprachmodell?",
    "Q-014": "Kostet die quantenabgeleitete Gewichtsteilung Qualitaet?",
    "Q-015": "Funktionsklasse oder Parametrisierung -- woher kommt der Vorteil?",
    "Q-016": "Braucht die Schaltung weniger Daten und Schritte?",
    "Q-017": "Rettet Data Re-Uploading die hardwaretaugliche Kodierung?",
    "Q-018": "Traegt das Training der Winkel etwas bei, ohne Encoder der aushilft?",
    "Q-019": ("Traegt die Butterfly-Struktur auf periodischen Signalen einen "
              "inhaltlichen Vorteil (Wechselwirkung periodisch x permutiert)?"),
    "Q-019W": ("Walsh statt Fourier: schlaegt die geteilte Butterfly (40) bei wenig "
               "Daten (N = 200) die dichte Schicht, und kommt das aus der Bitstruktur?"),
    "Q-020": ("Kernaussage fair neu: schlagen 16 Winkel ein ReLU-MLP gleicher "
              "Engstelle, und tragen Verschraenkung und Tiefe ohne Encoder bei?"),
    "T-000": "Kalibrierung: welche Modellbreite ist auf dieser Hardware trainierbar?",
    "T-001": "Verschiebt der Umbau R1/R5/R6 den Uebergangspunkt?",
    "T-004": ("Chatbot: senkt eine 192-Winkel-QIRE-Drehung als Adapter auf "
              "LLaeMmlein-120M den Verlust, wie weit kommt sie an LoRA heran, und "
              "traegt der CNOT-Ring?"),
    "T-005": ("Chatbot: wie faellt der Verlust mit der Lagenzahl des QIRE-Adapters, "
              "kommt L=10 an LoRA heran, und ergaenzt QIRE ein LoRA?"),
    "T-006": ("Chatbot: wie waechst der Gewinn des QIRE-Adapters von L=10 ueber 20 "
              "auf 40 Lagen (Monotonie, Form gegen sqrt(L)), und kommt L=40 an LoRA heran?"),
    "T-007": ("Chatbot: schlaegt oder erreicht ein Fastfood-Adapter (H D2 H D1, O(d) Diagonalen) "
              "bei GLEICHER Parameterzahl (30.720) LoRA r=1, und liegt er vor dem QIRE-Adapter; "
              "schlaegt ein Born-Zweig (Mischung, Quadrat, Rueckmischung) denselben Zweig mit GELU?"),
    "T-002": "Haelt die Struktur in einem Sprachmodell auf deutschem Text?",
}
# T-001 steht hier, obwohl es kein Q-Lauf ist: Der Zweck dieses Werkzeugs ist,
# dass kein fertiger Lauf im Bericht verschwindet, und T-001 hat alles, was der
# Bericht liest -- Vorregistrierung, Urteil, Auflösungspruefung.
# T-000 stand bis zum 23. September bewusst NICHT hier: eine Kalibrierung ohne
# Hypothese und ohne Urteil wuerde dauerhaft als "kein Urteil gespeichert"
# erscheinen und einen unfertigen Lauf vortaeuschen. Das Argument gilt weiter
# -- deshalb bekommt sie einen eigenen Topf KALIBRIERUNG statt OFFEN. Nur ganz
# zu fehlen war die falsche Loesung: dann taucht der Lauf nirgends auf.
KALIBRIERUNGEN = {"T-000"}
# Merkposten: Q-015 bis Q-017 fehlten hier zunaechst, weil sie nach dem Bau
# dieses Werkzeugs entstanden. Ein fertiger Lauf tauchte dadurch im Bericht
# ueberhaupt nicht auf -- schlimmer als eine falsche Einstufung, weil nichts
# darauf hinweist. Neue Laeufe gehoeren beim Anlegen sofort hierher. Dasselbe
# ist mit Q-019 noch einmal passiert.

# Was NACH dem Lauf entschieden wurde. Jede Zeile mit Tagebuchstelle. Das
# gespeicherte Urteil wird davon nicht veraendert -- es bleibt, was der Lauf
# damals sagte --, aber es wird nie mehr ohne diese Zeilen gedruckt.
NACHTRAEGE: Dict[str, List[str]] = {
    "Q-003-replication": [
        "MNIST (+0,0932) zitierbar: 7,2-facher Abstand zum Lernraten-Artefakt "
        "(+-0,013); `hybrid` am OBEREN Gitterrand -> Vorsprung eher zu klein. "
        "(Tagebuch 27.08., 'Budget-Fehler in neun Modulen')",
        "FASHION (+0,0599) NUR MIT VORBEHALT: Kontrollarm `classical_norm_deep` "
        "am UNTEREN Gitterrand -> Vorsprung aufgeblaeht; haelt bei 4,6-fachem "
        "Abstand zum Artefakt. (Tagebuch 27.08.)",
        "GLEICHWERTIGKEITEN Tiefe (+0,0002) und Verschraenkung (-0,0015) NICHT "
        "ZITIERBAR: (1) die Positivkontrolle ist durchgefallen, (2) der Encoder "
        "784->256 ist trainierbar und kann die Aufgabe der Schaltung uebernehmen "
        "-- genau das Artefakt, das Q-018 bei Q-014 gezeigt hat. Nachtest ohne "
        "Encoder noetig. (Review 23.09., BERICHT/00_REVIEW_BEFUNDE.md §2)",
        "Die klassische Kontrolle `classical_norm_deep` ist schwaecher als ein "
        "lineares Modell (Stichprobe 23.09., 3 Seeds, KEIN Befund) -- der "
        "Abstand +0,0932 misst vor allem die Schwaeche der Kontrolle. "
        "(Review 23.09. §1)",
    ],
    "Q-015": [
        "Nur MNIST (+0,0674, beide Arme innen, 5,2x Artefakt) zitierbar.",
        "FASHION (+0,0201) ZURUECKGEZOGEN: Kontrollarm `global_orth` am UNTEREN "
        "Gitterrand und nur 1,5-facher Abstand zum Lernraten-Artefakt. "
        "(Tagebuch 27.08.)",
    ],
    "Q-018": [
        "Primaerpaar beide innen, 31x Artefakt -- unberuehrt. P2 nicht "
        "zitierbar (`untied_trained` gesaettigt und bei 32 Haufen am unteren "
        "Rand). (Tagebuch 27.08.)",
    ],
    "Q-011": [
        "Dritter Anlauf am 27.08. um 23:58 gestorben (mnist Tiefe 16, Seed "
        "1810). Gerettet: 225 Zeilen, Tiefen 1-8 auf mnist. Tiefe 16 und "
        "fashion_mnist fehlen. Wiederaufnahme ist seit 23.09. eingebaut "
        "(`unitarity_depth.py`, `weiter=True`).",
        "Die Tuning-Abstaende (+0,0232 bei Tiefe 16) sind NICHT zitierbar -- "
        "drei Tuning-Seeds, abgebrochener Lauf. (Tagebuch 30.08.)",
        "`plain` bei Tiefe 8 am unteren Rand ist KOLLAPS (alle Raten 0,1000), "
        "keine Untertunierung. (HANDOFF_AKTUELL §0)",
    ],
    "Q-019": [
        "Vorlauf (2 Seeds, nicht zitierbar): P2 wird fallen (butterfly 0,686 "
        "gegen dense 0,913). Der Primaertest ist fast per Konstruktion wahr "
        "(feste Permutation ist fuer eine dichte Schicht lernbar, fuer eine "
        "Butterfly nicht) -- er belegt Ordnungsabhaengigkeit, keinen Vorteil. "
        "(Tagebuch 27.08. nachts; Review 23.09. §4)",
    ],
}

# Wer ist Mechanismus, wer Kontrolle? Nur damit kann die Richtung einer
# Randlage gedeutet werden. (Mechanismus, Kontrolle) des Primaervergleichs.
PRIMAERPAAR: Dict[str, Tuple[str, str]] = {
    "Q-003-replication": ("hybrid", "classical_norm_deep"),
    "Q-011": ("orthogonal", "spectral"),
    "Q-014": ("tied_L5", "untied_L5"),
    "Q-015": ("hybrid", "global_orth"),
    "Q-018": ("tied_trained", "tied_frozen"),
    "Q-019": ("butterfly", "dense"),
}


def latest_file(experiment_id: str) -> Optional[Path]:
    if not RESULTS.exists():
        return None
    files = sorted(RESULTS.glob(f"{experiment_id}__*.json"))
    return files[-1] if files else None


def _resolution(analysis: Dict) -> Tuple[Optional[bool], str]:
    """Auflösungspruefung finden -- die Laeufe legen sie unter verschiedenen
    Schluesseln ab, je nachdem wie die Kontrolle heisst."""
    for key in ("resolution_check", "positive_control"):
        block = analysis.get(key)
        if isinstance(block, dict) and "passed" in block:
            return bool(block["passed"]), str(block.get("text", ""))
    # Q-020 (kernaussage.py): je Abschnitt `kontrollen[<abschnitt>]["aufloesung"]`.
    # Bestanden nur, wenn ALLE gespeicherten Abschnitte bestanden haben.
    kontrollen = analysis.get("kontrollen")
    # Q-019W (walsh.py): `kontrollen["aufloesung"]` direkt, ein Abschnitt.
    if isinstance(kontrollen, dict) and isinstance(kontrollen.get("aufloesung"), dict):
        t = kontrollen["aufloesung"]
        return bool(t.get("bestanden")), (f"dense-narrow {float(t.get('delta', 0)):+.4f}, "
                                          f"p={float(t.get('p', 1)):.4f}")
    if isinstance(kontrollen, dict):
        teile = {ab: k["aufloesung"] for ab, k in kontrollen.items()
                 if isinstance(k, dict) and isinstance(k.get("aufloesung"), dict)}
        if teile:
            ok = all(bool(t.get("bestanden")) for t in teile.values())
            text = "; ".join(f"{ab}: {t.get('a')}-{t.get('b')} "
                             f"{float(t.get('delta', 0)):+.4f}, p={float(t.get('p', 1)):.4f}"
                             for ab, t in teile.items())
            return ok, text
    return None, ""


# ---------------------------------------------------------------------------
# Randlagen aus den Rohdaten
# ---------------------------------------------------------------------------
def randlagen(payload: Dict) -> List[Dict[str, object]]:
    """Fuer jede gespeicherte Lernratenwahl: sitzt der Arm am Gitterrand?

    Gelesen werden die Schluessel `lr_map_<bedingung>` (Q-003, Q-011, Q-014,
    Q-015, Q-018, Q-019) gegen `lr_grid` / `lr_gitter`. Liegt eine Rate
    AUSSERHALB des Startgitters, wurde verlaengert; dann zaehlt der Vermerk
    `lr_rand_<bedingung>`, falls der Lauf ihn gespeichert hat (Q-019, Q-011 ab
    23.09.), sonst "verlaengert, Rand unbekannt".

    KOLLAPS: Sind alle Messwerte eines Arms in dieser Bedingung gleich und
    liegen auf Rateniveau, war die Suche flach -- `max()` nimmt dann die erste
    Rate, und die Randmeldung ist bedeutungslos (Q-011, `plain` bei Tiefe 8).
    """
    cfg = payload.get("config") or {}
    gitter = cfg.get("lr_grid") or cfg.get("lr_gitter")
    if not gitter:
        return []
    # T-004 speichert je Arm ein eigenes Gitter ({arm: [..]}); sonst eine Liste.
    if isinstance(gitter, dict):
        je_arm = {a: (min(map(float, g)), max(map(float, g)))
                  for a, g in gitter.items() if g}
        if not je_arm:
            return []
        lo = min(v[0] for v in je_arm.values())
        hi = max(v[1] for v in je_arm.values())
    else:
        je_arm = {}
        lo, hi = min(gitter), max(gitter)
    records = payload.get("records") or []
    out: List[Dict[str, object]] = []
    karten = {k: v for k, v in cfg.items()
              if k.startswith("lr_map") and isinstance(v, dict)}
    if not karten:
        # Q-009 speichert keine lr_map, nur die Rate je Messzeile. Ohne diesen
        # Rueckgriff waere ausgerechnet der Lauf unsichtbar, in dem ALLE Arme
        # am unteren Rand sitzen.
        for r in records:
            if "lr" not in r or "arm" not in r:
                continue
            bed = next((str(r[f]) for f in ("dataset", "bedingung", "clusters")
                        if f in r), "")
            karten.setdefault(f"lr_map_{bed}", {})[r["arm"]] = r["lr"]
    for key, lr_map in karten.items():
        bed = key[len("lr_map_"):] if key != "lr_map" else ""
        vermerke = cfg.get(f"lr_rand_{bed}") or {}
        for arm, lr in lr_map.items():
            lr = float(lr)
            if arm in je_arm:
                lo, hi = je_arm[arm]
            if arm in vermerke:
                lage = str(vermerke[arm])
            elif lr <= lo * (1 + 1e-9):
                lage = "unten"
            elif lr >= hi * (1 - 1e-9) and lr <= hi * (1 + 1e-9):
                lage = "oben"
            elif lr < lo or lr > hi:
                lage = "verlaengert, Rand unbekannt"
            else:
                lage = "innen"
            werte = _werte_der_bedingung(records, arm, bed)
            if (lage != "innen" and len(werte) > 1 and np.ptp(werte) == 0.0
                    and float(werte[0]) <= 0.2):
                lage = "Kollaps (Suche flach)"
            out.append({"bedingung": bed, "arm": arm, "lr": lr, "lage": lage})
    return out


def _werte_der_bedingung(records, arm: str, bed: str) -> np.ndarray:
    """Messwerte eines Arms in einer Bedingung. Bedingungen heissen in den
    Laeufen verschieden: 'mnist', 'mnist_d8', '32' (Haufen), 'periodisch'."""
    def passt(r) -> bool:
        if r.get("arm") != arm:
            return False
        teile = bed.rsplit("_d", 1)
        if len(teile) == 2 and teile[1].isdigit() and "depth" in r:
            return r.get("dataset") == teile[0] and int(r["depth"]) == int(teile[1])
        for feld in ("dataset", "bedingung", "clusters"):
            if feld in r:
                return str(r[feld]) == bed
        return False
    return np.array([float(r["balanced_accuracy"]) for r in records
                     if passt(r) and "balanced_accuracy" in r])


def randbewertung(exp_id: str, lagen: List[Dict[str, object]]) -> List[str]:
    """Deutet die Randlagen des Primaerpaars. Regel (Tagebuch 27.08.): Nicht
    wer die kleinere Rate hat entscheidet, sondern AN WELCHEM RAND ein Arm
    sitzt -- untertuniert ist er an beiden."""
    paar = PRIMAERPAAR.get(exp_id)
    zeilen: List[str] = []
    for lage in lagen:
        if lage["lage"] in ("innen",):
            continue
        rolle = ""
        if paar and lage["arm"] == paar[0]:
            rolle = "MECHANISMUS untertuniert -> Abstand eher zu KLEIN (konservativ)"
        elif paar and lage["arm"] == paar[1]:
            rolle = "KONTROLLE untertuniert -> Abstand AUFGEBLAEHT (fuer die Hypothese)"
        if str(lage["lage"]).startswith("Kollaps"):
            rolle = "kein Tuningfehler, Arm auf Rateniveau"
        zeilen.append(f"{lage['bedingung'] or '-':<14} {lage['arm']:<22} "
                      f"lr {lage['lr']:<8g} {lage['lage']:<10}"
                      + (f"  {rolle}" if rolle else ""))
    return zeilen


def _vollstaendigkeit(payload: Dict) -> str:
    """Fuer Laeufe ohne Urteil: was ist schon gerechnet?"""
    rec = payload.get("records") or []
    if rec and "depth" in rec[0]:
        zellen: Dict[Tuple[str, int], int] = {}
        for r in rec:
            k = (str(r.get("dataset")), int(r["depth"]))
            zellen[k] = zellen.get(k, 0) + 1
        return "gerechnet: " + ", ".join(f"{ds} Tiefe {d} ({n})"
                                        for (ds, d), n in sorted(zellen.items()))
    return ""


def collect() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for exp_id, question in QUESTIONS.items():
        path = latest_file(exp_id)
        basis = {"id": exp_id, "question": question,
                 "nachtraege": NACHTRAEGE.get(exp_id, [])}
        if path is None:
            rows.append({**basis, "state": "OFFEN", "detail": "noch keine Rohdaten"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        analysis = payload.get("analysis") or {}
        n = payload.get("n_records", 0)
        created = str(payload.get("created", ""))[:16].replace("T", " ")
        basis.update({"n": n, "created": created, "file": path.name,
                      "rand": randbewertung(exp_id, randlagen(payload))})

        if exp_id in KALIBRIERUNGEN:
            rows.append({**basis, "state": "KALIBRIERUNG",
                         "detail": "Messung ohne Hypothese und ohne Urteil -- "
                                   "gehoert nicht zu OFFEN"})
            continue
        if "error" in analysis:
            rows.append({**basis, "state": "OFFEN",
                         "detail": f"Auswertung fehlgeschlagen: {analysis['error'][:60]}"})
            continue
        verdict = str(analysis.get("verdict", "")).strip()
        if not verdict:
            detail = "Rohdaten da, aber kein Urteil gespeichert (Lauf evtl. noch nicht durch)"
            voll = _vollstaendigkeit(payload)
            rows.append({**basis, "state": "OFFEN",
                         "detail": detail + (f"; {voll}" if voll else "")})
            continue
        passed, text = _resolution(analysis)
        state = _classify(verdict, passed)
        rows.append({**basis, "state": state, "verdict": verdict,
                     "resolution": passed, "resolution_text": text})
    return rows


def _classify(verdict: str, passed: Optional[bool]) -> str:
    """Wofuer eine durchgefallene Auflösungspruefung zaehlt -- und wofuer nicht.

    Eine Positivkontrolle sichert die Deutung von NULLBEFUNDEN ab. Findet ein
    Lauf dagegen einen grossen, signifikanten Unterschied, hat er damit selbst
    bewiesen, dass sein Aufbau aufloest -- der durchgefallene Test sagt dann
    nur, dass er KLEINE Unterschiede nicht trennt.

    Konkret an Q-003-replication: die Positivkontrolle (5 Lagen gegen 1 Lage)
    fiel durch, weil beide gleich gut sind -- der Primaereffekt ist mit +0,0932
    aber das 400-fache davon und mit Holm p=0,0001 belegt. Ihn deswegen zu
    verwerfen waere falsch. Umgekehrt darf im selben Lauf keine
    Gleichwertigkeitsaussage stehen bleiben.

    Die Laeufe kodieren das im Urteil: H1 = Unterschied gefunden,
    H0 = Gleichwertigkeit behauptet, HU = unentschieden.
    """
    if passed is not False:
        return "GILT"
    head = verdict.strip()[:3].upper()
    if head.startswith("H1"):
        return "GILT MIT VORBEHALT"
    return "NICHT VERWERTBAR"


def _wrap(text: str, width: int, indent: str) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return f"\n{indent}".join(lines)


ZUSTAENDE = ("NICHT VERWERTBAR", "GILT MIT VORBEHALT", "GILT", "KALIBRIERUNG", "OFFEN")


def main() -> None:
    rows = collect()
    order = {s: i for i, s in enumerate(ZUSTAENDE)}
    rows.sort(key=lambda r: (order[str(r["state"])], str(r["id"])))

    print("=" * 92)
    print("SAMMELBERICHT Q-SERIE")
    print("Gelesen werden die gespeicherten Rohdaten, nicht die Konsolenlogs.")
    print("NACHTRAG = spaeter entschieden (Tagebuch), RAND = aus den Rohdaten gelesen.")
    print("=" * 92)

    for state in ZUSTAENDE:
        sub = [r for r in rows if r["state"] == state]
        if not sub:
            continue
        print()
        print(f"### {state} ({len(sub)})")
        if state == "NICHT VERWERTBAR":
            print("### Auflösungspruefung durchgefallen UND der Lauf behauptet")
            print("### Gleichwertigkeit oder bleibt unentschieden. Solche Urteile")
            print("### duerfen nicht zitiert werden: der Aufbau haette einen")
            print("### Unterschied gar nicht finden koennen. Fehler wie in B-004.")
        elif state == "GILT MIT VORBEHALT":
            print("### Auflösungspruefung durchgefallen, ABER der Lauf hat einen")
            print("### signifikanten Unterschied gefunden -- damit hat er selbst")
            print("### bewiesen, dass er aufloest. Der gefundene Unterschied")
            print("### steht; jede GLEICHWERTIGKEITS-Aussage aus demselben Lauf")
            print("### steht nicht.")
        elif state == "GILT":
            print("### Urteil steht -- aber die NACHTRAEGE darunter gelten mit.")
        print("-" * 92)
        for r in sub:
            print(f"  {r['id']}   {r.get('created', '')}"
                  + (f"   {r.get('n', 0)} Zeilen" if r.get("n") else ""))
            print(f"    Frage:  {_wrap(str(r['question']), 78, ' ' * 12)}")
            if "verdict" in r:
                print(f"    Urteil: {_wrap(str(r['verdict']), 78, ' ' * 12)}")
                if r.get("resolution") is False:
                    print(f"    !!      {_wrap(str(r['resolution_text']), 78, ' ' * 12)}")
                elif r.get("resolution") is None:
                    print("    Hinweis: keine Auflösungspruefung gespeichert -- "
                          "Regel 6 nicht erfuellt.")
            else:
                print(f"    Stand:  {_wrap(str(r.get('detail', '')), 78, ' ' * 12)}")
            for z in r.get("rand") or []:
                print(f"    RAND    {z}")
            for z in r.get("nachtraege") or []:
                print(f"    NACHTRAG {_wrap(z, 77, ' ' * 13)}")
            print()

    tally = {s: sum(1 for r in rows if r["state"] == s) for s in ZUSTAENDE}
    nicht = tally["NICHT VERWERTBAR"]
    n_rand = sum(len(r.get("rand") or []) for r in rows)
    print("=" * 92)
    print(f"BILANZ: {tally['GILT']} verwertbar, "
          f"{tally['GILT MIT VORBEHALT']} mit Vorbehalt, "
          f"{nicht} nicht verwertbar, {tally['OFFEN']} offen, "
          f"{tally['KALIBRIERUNG']} Kalibrierung")
    print(f"        {n_rand} Lernratenwahl(en) am Gitterrand oder kollabiert, "
          f"{sum(len(r.get('nachtraege') or []) for r in rows)} Nachtraege")
    if nicht:
        print()
        print("Ein durchgefallener Auflösungstest ist KEIN Nullbefund. Er heisst,")
        print("dass der Aufbau nichts haette finden koennen -- der Lauf muss mit")
        print("einer schwereren Aufgabe oder groesserem Kontrast wiederholt werden.")
    print("=" * 92)


if __name__ == "__main__":
    main()
