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

Ergaenzt `qire_ai.audit`: der prueft Tagebuchzahlen gegen Rohdaten, dieser hier
prueft die Urteile gegen ihre eigenen Vorbedingungen.

Run: python -m quantum_sim.q_summary
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "T-001": "Verschiebt der Umbau R1/R5/R6 den Uebergangspunkt?",
    "T-002": "Haelt die Struktur in einem Sprachmodell auf deutschem Text?",
}
# T-001 steht hier, obwohl es kein Q-Lauf ist: Der Zweck dieses Werkzeugs ist,
# dass kein fertiger Lauf im Bericht verschwindet, und T-001 hat alles, was der
# Bericht liest -- Vorregistrierung, Urteil, Auflösungspruefung.
# T-000 steht bewusst NICHT hier. Es ist eine Kalibrierung ohne Hypothese und
# ohne Urteil; es wuerde dauerhaft als "kein Urteil gespeichert" erscheinen und
# damit einen unfertigen Lauf vortaeuschen.
# Merkposten: Q-015 bis Q-017 fehlten hier zunaechst, weil sie nach dem Bau
# dieses Werkzeugs entstanden. Ein fertiger Lauf tauchte dadurch im Bericht
# ueberhaupt nicht auf -- schlimmer als eine falsche Einstufung, weil nichts
# darauf hinweist. Neue Laeufe gehoeren beim Anlegen sofort hierher.


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
    return None, ""


def collect() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for exp_id, question in QUESTIONS.items():
        path = latest_file(exp_id)
        if path is None:
            rows.append({"id": exp_id, "question": question, "state": "OFFEN",
                         "detail": "noch keine Rohdaten"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        analysis = payload.get("analysis") or {}
        n = payload.get("n_records", 0)
        created = str(payload.get("created", ""))[:16].replace("T", " ")

        if "error" in analysis:
            rows.append({"id": exp_id, "question": question, "state": "OFFEN",
                         "detail": f"Auswertung fehlgeschlagen: {analysis['error'][:60]}",
                         "n": n, "created": created, "file": path.name})
            continue
        verdict = str(analysis.get("verdict", "")).strip()
        if not verdict:
            rows.append({"id": exp_id, "question": question, "state": "OFFEN",
                         "detail": "Rohdaten da, aber kein Urteil gespeichert "
                                   "(Lauf evtl. noch nicht durch)",
                         "n": n, "created": created, "file": path.name})
            continue
        passed, text = _resolution(analysis)
        state = _classify(verdict, passed)
        rows.append({"id": exp_id, "question": question, "state": state,
                     "verdict": verdict, "resolution": passed,
                     "resolution_text": text, "n": n, "created": created,
                     "file": path.name})
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


def main() -> None:
    rows = collect()
    order = {"NICHT VERWERTBAR": 0, "GILT MIT VORBEHALT": 1, "GILT": 2, "OFFEN": 3}
    rows.sort(key=lambda r: (order[str(r["state"])], str(r["id"])))

    print("=" * 92)
    print("SAMMELBERICHT Q-SERIE")
    print("Gelesen werden die gespeicherten Rohdaten, nicht die Konsolenlogs.")
    print("=" * 92)

    for state in ("NICHT VERWERTBAR", "GILT MIT VORBEHALT", "GILT", "OFFEN"):
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
                print(f"    Stand:  {r.get('detail', '')}")
            print()

    tally = {s: sum(1 for r in rows if r["state"] == s)
             for s in ("GILT", "GILT MIT VORBEHALT", "NICHT VERWERTBAR", "OFFEN")}
    nicht = tally["NICHT VERWERTBAR"]
    print("=" * 92)
    print(f"BILANZ: {tally['GILT']} verwertbar, "
          f"{tally['GILT MIT VORBEHALT']} mit Vorbehalt, "
          f"{nicht} nicht verwertbar, {tally['OFFEN']} offen")
    if nicht:
        print()
        print("Ein durchgefallener Auflösungstest ist KEIN Nullbefund. Er heisst,")
        print("dass der Aufbau nichts haette finden koennen -- der Lauf muss mit")
        print("einer schwereren Aufgabe oder groesserem Kontrast wiederholt werden.")
    print("=" * 92)


if __name__ == "__main__":
    main()
