"""Ausarbeitung der besten ~10 % zu Karten im Stil von INSTITUT/ideen.json."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .pruefer import parse_json

SYSTEM_KARTE = (
    "ROLLE:KARTE. Arbeite die Idee zu einer Forschungskarte aus. Antworte NUR als JSON mit den Feldern: "
    "titel, hypothese, kontrolle (nicht entartet!), erwarteter_effekt (mit Groesse), abbruch (Abbruchkriterium, "
    "messbar), neuheitsbegruendung (ehrlich: was ist bekannt?), billigster_test (unter 2 h PC), konzept, "
    "engpass, quantenstufe (Q1|Q2|Q3). Auf Deutsch, knapp."
)

PFLICHT = ["id", "titel", "hypothese", "kontrolle", "erwarteter_effekt", "abbruch",
           "neuheitsbegruendung", "kanal", "operator", "billigster_test", "status"]
# Felder wie in INSTITUT/ideen.json
ZIEL_FELDER = ["titel", "konzept", "engpass", "denkzug", "abteilung", "papierpruefung", "neuheit", "abbruch",
               "billigster_test", "kontrolle", "id", "status", "erzeugt", "herkunft", "frage", "quantenkern",
               "quantenstufe"]


def validiere_karte(k: dict) -> list[str]:
    """Liste der Maengel (leer = ok)."""
    m = [f"Feld fehlt/leer: {f}" for f in PFLICHT if not str(k.get(f, "")).strip()]
    if k.get("kanal") not in ("QIRE-nah", "Neuland"):
        m.append("Kanal ungueltig")
    if k.get("quantenstufe") not in ("Q1", "Q2", "Q3"):
        m.append("Quantenstufe ungueltig")
    return m


def arbeite_aus(llm, idee: dict, laufnr: int, temperatur=0.6, max_tokens=900,
                heute: Optional[datetime] = None) -> Optional[dict]:
    """idee: Datensatz aus ideen.jsonl (id, text, kanal, operator, rubrik, ...). Gibt Karte oder None."""
    heute = heute or datetime.now()
    d = None
    for _ in range(2):
        d = parse_json(llm.chat(SYSTEM_KARTE, f"Kanal: {idee['kanal']}\nIdee: {idee['text']}", temperatur, max_tokens))
        if d:
            break
    if not d:
        return None
    rub = idee.get("rubrik") or {}
    q = d.get("quantenstufe") or rub.get("quantenstufe") or "Q1"
    kid = f"KI-{heute:%Y%m%d}-{laufnr:03d}"
    text = idee["text"]
    karte = {
        "id": kid,
        "titel": str(d.get("titel") or text[:100]),
        "konzept": str(d.get("konzept", "")),
        "engpass": str(d.get("engpass", "")),
        "denkzug": idee["operator"],
        "abteilung": "Ideenschmiede-KI",
        "papierpruefung": str(rub.get("begruendung", "")),
        "neuheit": str(d.get("neuheitsbegruendung", "")),
        "abbruch": str(d.get("abbruch", "")),
        "billigster_test": str(d.get("billigster_test", "")),
        "kontrolle": str(d.get("kontrolle", "")),
        "status": "roh_ki",
        "erzeugt": f"{heute:%Y-%m-%d}",
        "herkunft": "Lokale Ideen-KI (nicht von Claude geprueft)",
        "frage": text,
        "quantenkern": str(d.get("hypothese", "")),
        "quantenstufe": q if q in ("Q1", "Q2", "Q3") else "Q1",
        # Zusatzfelder der Ideen-KI
        "hypothese": str(d.get("hypothese", "")),
        "erwarteter_effekt": str(d.get("erwarteter_effekt", "")),
        "neuheitsbegruendung": str(d.get("neuheitsbegruendung", "")),
        "kanal": idee["kanal"],
        "operator": idee["operator"],
        "quelle_idee": idee["id"],
        "rubrik": {k: rub.get(k) for k in ("neuheit", "pruefbarkeit", "erkenntniswert", "ehrlichkeit",
                                           "quantenstufe", "score")},
        "gedankenexperiment": bool(idee.get("physik", {}).get("gedankenexperiment", False)),
    }
    return karte if not validiere_karte(karte) else None


def anzahl_top(n_bewertet: int, anteil: float) -> int:
    return max(1, math.ceil(n_bewertet * anteil)) if n_bewertet else 0


def _atomar(pfad: Path, text: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_suffix(pfad.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, pfad)


def schreibe_karten(karten: list[dict], ausgabe: Path, heute: Optional[datetime] = None) -> Path:
    """Haengt Karten an ausgabe/karten_JJJJMMTT.json an (ohne doppelte IDs) und erneuert TOP.md."""
    heute = heute or datetime.now()
    pfad = ausgabe / f"karten_{heute:%Y%m%d}.json"
    vorhanden = []
    if pfad.exists():
        try:
            vorhanden = json.loads(pfad.read_text(encoding="utf-8")).get("karten", [])
        except Exception:
            vorhanden = []
    ids = {k["id"] for k in vorhanden}
    alle = vorhanden + [k for k in karten if k["id"] not in ids]
    _atomar(pfad, json.dumps({"karten": alle}, ensure_ascii=False, indent=1))
    schreibe_top(ausgabe)
    return pfad


def schreibe_top(ausgabe: Path, n: int = 15) -> Path:
    """TOP.md: beste Karten aller karten_*.json nach Score."""
    karten = []
    for p in sorted(ausgabe.glob("karten_*.json")):
        try:
            karten += json.loads(p.read_text(encoding="utf-8")).get("karten", [])
        except Exception:
            pass
    karten.sort(key=lambda k: -(k.get("rubrik", {}).get("score") or 0))
    z = ["# Top-Ideen der lokalen Ideen-KI", "",
         "*Nur Vorschlaege, nicht von Claude geprueft. Bitte blind bewerten: `ausgabe/blindnoten.csv`.*", ""]
    for k in karten[:n]:
        r = k.get("rubrik", {})
        z += [f"## {k['id']} ({k['kanal']}, {k['operator']}) -- {k['titel']}",
              f"- Score {r.get('score', 0):.2f} | Neuheit {r.get('neuheit')} | Pruefbarkeit {r.get('pruefbarkeit')} | "
              f"Erkenntnis {r.get('erkenntniswert')} | Ehrlichkeit {r.get('ehrlichkeit')} | {k.get('quantenstufe')}",
              f"- Hypothese: {k['hypothese']}", f"- Kontrolle: {k['kontrolle']}",
              f"- Erwarteter Effekt: {k['erwarteter_effekt']}", f"- Abbruch: {k['abbruch']}",
              f"- Neuheit: {k['neuheitsbegruendung']}", ""]
    pfad = ausgabe / "TOP.md"
    _atomar(pfad, "\n".join(z) + "\n")
    return pfad
