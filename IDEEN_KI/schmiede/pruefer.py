"""Physik-Pruefer + Rubrik-Bewertung (1-5). Robustes Parsen von LLM-Ausgaben."""
from __future__ import annotations

import ast
import json
import re
from typing import Optional

SYSTEM_PHYSIK = (
    "ROLLE:PHYSIK. Du pruefst eine Forschungsidee auf Verletzungen physikalischer Grundregeln: "
    "Unitaritaet, No-Cloning, kein ueberlichtschnelles Signal (No-Signaling), Energieerhaltung. "
    "Absichtliche Axiom-Brueche sind erlaubt, wenn sie ausdruecklich als Gedankenexperiment markiert sind. "
    'Antworte NUR als JSON: {"verstoesse": ["..."], "gedankenexperiment": true|false, "begruendung": "..."}'
)
SYSTEM_RUBRIK = (
    "ROLLE:RUBRIK. Du bewertest eine Forschungsidee streng und ehrlich mit ganzen Zahlen 1-5: "
    "neuheit, pruefbarkeit (5 = unter 2 Stunden auf einem PC testbar), erkenntniswert, "
    "ehrlichkeit (5 = Effekt waere NICHT bloss klassisch erklaerbar), und quantenstufe Q1 (nur Name), "
    "Q2 (Struktur uebernommen), Q3 (echter Quantenmechanismus). "
    'Antworte NUR als JSON: {"neuheit":n,"pruefbarkeit":n,"erkenntniswert":n,"ehrlichkeit":n,'
    '"quantenstufe":"Q1|Q2|Q3","begruendung":"..."}'
)

# Regelbasierter Vorfilter (billig, vor dem LLM)
_VERBOTEN = [
    (r"schneller als (das )?licht|ueberlicht|überlicht|superluminal|instantan\w* (signal|nachricht|kommunikation)",
     "ueberlichtschnelles Signal"),
    (r"(klon\w*|kopier\w*)\s+(eines |von )?(unbekannten )?(quanten)?zustand", "No-Cloning"),
    (r"perpetuum|energie aus nichts|mehr energie heraus|energieerhaltung (verletz|umgeh)", "Energieerhaltung"),
    (r"nicht-?unit(a|ä)er\w*\s+(schicht|gatter|zeitentwicklung)\s+(ohne|bei erhaltener)", "Unitaritaet"),
]
_MARKE = re.compile(r"gedankenexperiment", re.I)


def vorfilter(text: str) -> tuple[bool, list[str]]:
    """(ok, Verstoesse). Gedankenexperiment-Markierung hebt den Verstoss auf."""
    if _MARKE.search(text):
        return True, []
    v = [name for pat, name in _VERBOTEN if re.search(pat, text, re.I)]
    return (not v), v


# ---------- robustes JSON-Parsen ----------
def _balanciert(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        tiefe, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    tiefe += 1
                elif c == "}":
                    tiefe -= 1
                    if tiefe == 0:
                        return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_json(text: str) -> Optional[dict]:
    """Extrahiert das erste JSON-Objekt aus LLM-Text (Codezaeune, Vorreden, haengende Kommas,
    einfache Anfuehrungszeichen, abgeschnittene Klammern). None, wenn nichts zu retten ist."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"```(?:json|JSON)?", "", t)
    kand = _balanciert(t)
    if kand is None and "{" in t:                       # abgeschnitten: Klammern ergaenzen
        teil = t[t.index("{"):].rstrip().rstrip(",")
        if teil.count('"') % 2:
            teil += '"'
        kand = teil + "}" * max(0, teil.count("{") - teil.count("}"))
    if kand is None:
        return None
    for versuch in (kand, re.sub(r",\s*([}\]])", r"\1", kand)):
        try:
            d = json.loads(versuch)
            return d if isinstance(d, dict) else None
        except Exception:
            pass
    try:
        d = ast.literal_eval(re.sub(r"\b(true|false|null)\b",
                                    lambda m: {"true": "True", "false": "False", "null": "None"}[m.group(1)], kand))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _zahl(x, lo=1, hi=5) -> Optional[int]:
    try:
        m = re.search(r"-?\d+(?:[.,]\d+)?", str(x))
        return None if not m else int(min(hi, max(lo, round(float(m.group(0).replace(",", "."))))))
    except Exception:
        return None


def normalisiere_rubrik(d: Optional[dict]) -> Optional[dict]:
    if not d:
        return None
    out = {}
    for k in ("neuheit", "pruefbarkeit", "erkenntniswert", "ehrlichkeit"):
        z = _zahl(d.get(k))
        if z is None:
            return None
        out[k] = z
    m = re.search(r"[Qq]\s*([123])", str(d.get("quantenstufe", "")))
    out["quantenstufe"] = f"Q{m.group(1)}" if m else "Q1"
    out["begruendung"] = str(d.get("begruendung", ""))[:400]
    return out


def score(rubrik: dict, kanal: str, cfg_rubrik: dict) -> float:
    """Gewichteter Score in 0..1. Neuland: Neuheit x Faktor."""
    g = dict(cfg_rubrik["gewichte"])
    if kanal == "Neuland":
        g["neuheit"] = g.get("neuheit", 1.0) * cfg_rubrik.get("neuland_neuheit_faktor", 2.0)
    summe = sum(g.values())
    s = sum(g[k] * (rubrik[k] - 1) / 4.0 for k in g)
    s = s / summe
    bonus = cfg_rubrik.get("quantenstufe_bonus", {}).get(rubrik.get("quantenstufe", "Q1"), 0.0)
    return float(max(0.0, min(1.0, 0.9 * s + 0.1 * bonus)))


def pruefe_physik(llm, text: str, temperatur=0.2, max_tokens=300) -> dict:
    ok, regeln = vorfilter(text)
    if not ok:
        return {"ok": False, "verstoesse": regeln, "gedankenexperiment": False, "quelle": "regel"}
    antwort = llm.chat(SYSTEM_PHYSIK, "Idee: " + text, temperatur, max_tokens)
    d = parse_json(antwort) or {}
    verstoesse = [str(v) for v in d.get("verstoesse", []) or []]
    ge = bool(d.get("gedankenexperiment", False)) or bool(_MARKE.search(text))
    return {"ok": (not verstoesse) or ge, "verstoesse": verstoesse, "gedankenexperiment": ge,
            "quelle": "llm" if d else "llm_unlesbar"}


def bewerte(llm, text: str, kanal: str, cfg_rubrik: dict, beispiele: str = "",
            temperatur=0.2, max_tokens=300) -> Optional[dict]:
    """Rubrik 1-5; bei unlesbarer Ausgabe ein zweiter Versuch, sonst None."""
    user = (beispiele + "\n" if beispiele else "") + f"Kanal: {kanal}\nIdee: {text}"
    for _ in range(2):
        r = normalisiere_rubrik(parse_json(llm.chat(SYSTEM_RUBRIK, user, temperatur, max_tokens)))
        if r:
            r["score"] = score(r, kanal, cfg_rubrik)
            return r
    return None
