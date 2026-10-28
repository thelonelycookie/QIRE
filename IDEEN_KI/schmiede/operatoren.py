"""Kreativ-Operatoren (Plan §3), zwei Kanaele, Banditen-Auswahl (Thompson-Sampling)."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

FEST = Path(__file__).resolve().parent.parent / "daten_fest"

KANAL_NAH = "QIRE-nah"
KANAL_NEULAND = "Neuland"

SYSTEM_ERFINDER = (
    "ROLLE:ERFINDER. Du bist ein kreativer Forschungs-Ideengeber fuer das Projekt QIRE "
    "(quanten-inspirierte, ressourceneffiziente KI; Simulation auf einem normalen PC). "
    "Antworte NUR mit 1-2 Saetzen auf Deutsch: eine konkrete, pruefbare Idee mit "
    "Hypothese und Kontrolle. Keine Einleitung, keine Aufzaehlung."
)


def lade_liste(name: str) -> list[str]:
    p = FEST / name
    return [z.strip() for z in p.read_text(encoding="utf-8").splitlines() if z.strip() and not z.startswith("#")]


@dataclass
class Operator:
    name: str
    beschreibung: str
    kanaele: tuple[str, ...]
    baue: Callable[["Eingabe"], tuple[str, dict]]


@dataclass
class Eingabe:
    kanal: str
    kontext: str                   # RAG-Ausschnitte (Text)
    rng: random.Random
    luecken: object = None         # Lueckenkarte
    konzepte_qm: list = field(default_factory=list)
    konzepte_fremd: list = field(default_factory=list)
    befunde: list = field(default_factory=list)   # Kurztexte einzelner Befunde


def _kopf(e: Eingabe) -> str:
    ziel = ("Bleibe nah an QIRE: die Idee muss mit den vorhandenen Modulen in unter 2 Stunden auf einem PC testbar sein."
            if e.kanal == KANAL_NAH else
            "Kanal NEULAND: sei radikal neu, auch ausserhalb des bisherigen QIRE-Rahmens, aber simulierbar und ehrlich pruefbar.")
    return f"{ziel}\nKontext aus dem Projekt:\n{e.kontext or '(kein Kontext)'}\n\n"


def _kreuzung(e: Eingabe):
    if e.luecken is not None:
        x, y = e.luecken.ziehe_paar(e.rng)
    else:
        x, y = e.rng.choice(e.konzepte_qm), e.rng.choice(e.konzepte_fremd)
    return (_kopf(e) + f"Aufgabe (Begriffs-Kreuzung): Verbinde das Quantenkonzept '{x}' mit dem Fremdgebiet '{y}' "
            "zu einer neuen, pruefbaren Idee fuer ressourceneffizientes maschinelles Lernen.",
            {"paar": [x, y]})


def _luecke(e: Eingabe):
    x, y = e.luecken.ziehe_paar(e.rng, versuche=24) if e.luecken is not None else (
        e.rng.choice(e.konzepte_qm), e.rng.choice(e.konzepte_fremd))
    return (_kopf(e) + f"Aufgabe (Luecke): Das Paar '{x}' x '{y}' wurde im Projekt noch nie verbunden. "
            "Fuelle diese Luecke mit einer konkreten, pruefbaren Idee.", {"paar": [x, y], "luecke": True})


_AXIOME = ["Die Born-Regel lautet |psi|^3 statt |psi|^2", "Die Zeitentwicklung ist leicht nicht-unitaer",
           "Messungen kollabieren nicht, sondern dekohaerieren nur teilweise",
           "Es gibt eine dritte Art von Teilchen zwischen Boson und Fermion",
           "Verschraenkung ist begrenzt auf Paare (Monogamie strenger)",
           "Die Hilbert-Raum-Dimension waechst nur polynomiell mit der Teilchenzahl",
           "Amplituden sind reell (keine komplexen Phasen)", "Superposition ist nur fuer zwei Zustaende erlaubt"]


def _axiom(e: Eingabe):
    a = e.rng.choice(_AXIOME)
    return (_kopf(e) + f"Aufgabe (Axiom-Spiel): Angenommen: {a}. Was waere simulier- oder messbar, und was "
            "folgt fuer lernende Schaltungen? Markiere die Idee ausdruecklich als Gedankenexperiment.",
            {"axiom": a, "gedankenexperiment": True})


def _h0(e: Eingabe):
    b = e.rng.choice(e.befunde) if e.befunde else "Der Vorteil von QIRE ist rein klassisch erklaerbar."
    return (_kopf(e) + f"Aufgabe (H0-Angriff): Greife diesen Befund an und schlage einen Test vor, der ihn "
            f"widerlegen koennte: {b}", {"befund": b})


def _umkehr(e: Eingabe):
    b = e.rng.choice(e.befunde) if e.befunde else "Winkel werden auf ein Gitter gerundet."
    return (_kopf(e) + f"Aufgabe (Umkehrung): Kehre diese Annahme um und beschreibe das Experiment zum Gegenentwurf: {b}",
            {"befund": b})


def _kombi(e: Eingabe):
    if len(e.befunde) >= 2:
        b1, b2 = e.rng.sample(e.befunde, 2)
    else:
        b1, b2 = "Magie konzentriert sich in wenigen Winkeln", "Die Wirkung von QIRE ist klassisch"
    return (_kopf(e) + f"Aufgabe (Kombination): Verbinde diese zwei Befunde zu einer neuen Frage:\n(1) {b1}\n(2) {b2}",
            {"befunde": [b1, b2]})


def _extrem(e: Eingabe):
    f = e.rng.choice(["Qubitzahl auf 2", "Qubitzahl auf 20", "Lagenzahl auf 1", "Lagenzahl auf 100",
                      "Datenmenge auf 50 Beispiele", "Rauschen auf 30 %", "Parameterzahl auf 1 pro Schicht"])
    return (_kopf(e) + f"Aufgabe (Extrem-Skalierung): Was passiert, wenn man {f} setzt? "
            "Formuliere eine pruefbare Vorhersage.", {"extrem": f})


def _kontrolle(e: Eingabe):
    return (_kopf(e) + "Aufgabe (Kontrolle als Hauptsache): Waehle eine Kontrollbedingung aus dem Projekt "
            "(z. B. Zufallsschaltung, Identitaet, klassische Baseline) und mache sie zum Hauptgegenstand eines "
            "neuen Experiments.", {})


OPERATOREN: dict[str, Operator] = {o.name: o for o in [
    Operator("kreuzung", "Begriffs-Kreuzung", (KANAL_NEULAND, KANAL_NAH), _kreuzung),
    Operator("luecke", "Luckenkarte", (KANAL_NEULAND,), _luecke),
    Operator("axiom", "Axiom-Spiel", (KANAL_NEULAND,), _axiom),
    Operator("h0_angriff", "H0-Angriff", (KANAL_NAH, KANAL_NEULAND), _h0),
    Operator("umkehrung", "Umkehrung", (KANAL_NAH, KANAL_NEULAND), _umkehr),
    Operator("kombination", "Kombination zweier Befunde", (KANAL_NAH,), _kombi),
    Operator("extrem", "Extrem-Skalierung", (KANAL_NAH, KANAL_NEULAND), _extrem),
    Operator("kontrolle", "Kontrolle-als-Hauptsache", (KANAL_NAH,), _kontrolle),
]}


def operatoren_fuer(kanal: str) -> list[str]:
    return [n for n, o in OPERATOREN.items() if kanal in o.kanaele]


class Bandit:
    """Thompson-Sampling: je (Kanal, Operator) eine Beta(alpha, beta)-Verteilung.
    Belohnung r in [0,1] (zuerst Rubrik-Score, spaeter Urteile H1/H0/HU und Blindnoten)."""

    def __init__(self, pfad: Optional[Path] = None):
        self.pfad = pfad
        self.arme: dict[str, list[float]] = {}
        if pfad and pfad.exists():
            try:
                self.arme = json.loads(pfad.read_text(encoding="utf-8"))
            except Exception:
                self.arme = {}

    @staticmethod
    def schluessel(kanal: str, op: str) -> str:
        return f"{kanal}::{op}"

    def _ab(self, kanal, op):
        return self.arme.setdefault(self.schluessel(kanal, op), [1.0, 1.0])

    def waehle(self, kanal: str, rng: random.Random) -> str:
        namen = operatoren_fuer(kanal)
        return max(namen, key=lambda n: rng.betavariate(*self._ab(kanal, n)))

    def aktualisiere(self, kanal: str, op: str, belohnung: float, gewicht: float = 1.0) -> None:
        r = min(1.0, max(0.0, float(belohnung)))
        ab = self._ab(kanal, op)
        ab[0] += gewicht * r
        ab[1] += gewicht * (1.0 - r)
        self.speichern()

    def mittel(self, kanal: str, op: str) -> float:
        a, b = self._ab(kanal, op)
        return a / (a + b)

    def speichern(self) -> None:
        if not self.pfad:
            return
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.arme, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.pfad)


def waehle_kanal(anteil_neuland: float, rng: random.Random) -> str:
    return KANAL_NEULAND if rng.random() < anteil_neuland else KANAL_NAH


def erzeuge_prompt(op_name: str, e: Eingabe) -> tuple[str, dict]:
    return OPERATOREN[op_name].baue(e)
