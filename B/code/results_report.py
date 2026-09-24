from __future__ import annotations

"""Erzeugt RESULTS.md -- die Zuordnung jeder Aussage zu ihrer Rohdatei.

WARUM GENERIERT UND NICHT GESCHRIEBEN
Eine von Hand gepflegte Ergebnisuebersicht veraltet beim ersten Lauf, der etwas
umwirft -- und in diesem Projekt sind an einem einzigen Tag vier Hauptaussagen
gefallen. Diese Datei liest stattdessen die gespeicherten Rohdaten und schreibt
daraus RESULTS.md. Damit kann die Uebersicht nicht von den Daten abweichen; sie
IST die Daten.

Fuer eine Jury ist das der Unterschied zwischen "ich habe das gemessen" und
"hier ist die Datei, aus der die Zahl stammt, mit Seed-Liste, Konfiguration und
Softwareumgebung".

Run: python -m qire_ai.results_report
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .result_store import RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "RESULTS.md"
DIARY = ROOT / "PROJEKTTAGEBUCH.md"

# Kurzbeschreibung je Experiment. Was hier nicht steht, wird trotzdem
# aufgefuehrt -- nur ohne Einordnung.
TITLES: Dict[str, str] = {
    "B-003": "Blockschicht gegen dicht, mit und ohne Beschraenkung, zwei Tiefen",
    "B-004-lr": "Lernratensuche je Arm und Architektur",
    "B-004": "Gleiches Budget: Blockdiagonale, Monarch, TT, Low-Rank; drei Architekturen",
    "B-005": "Struktur der gelernten dichten Schicht (Blockmasse, Spektrum)",
    "B-006": "Konvergenzkontrolle zu B-004",
    "B-007": "Orthogonalitaet gegen Spektralnorm, vorregistrierte Wiederholung",
    "B-008": "Rangreihe 4..256 plus exaktes Gewichtsspektrum, zwei Datensaetze",
    "B-009": "Monarch gegen dicht in der lokalen Architektur, 30 Seeds",
    "B-010": "Datenmenge gegen Struktur, sechs Trainingsgroessen",
    "B-011": "Darstellungs- oder Optimierungsluecke; datengewichtetes Spektrum",
    "B-012": "Dosis-Wirkungs-Kurve des Trainingswegs, konstantes Budget",
    "B-013": "Rang-Curriculum gegen festen Rang und einzelnen Schnitt",
    "B-014": "Intrinsische Datendimension gegen Klassenzahl, mit untrainierter Kontrolle",
    "Q-003-replication": "Q-001 bis Q-003 unter dem neuen Protokoll",
}


def _latest_per_experiment() -> Dict[str, Path]:
    """Neueste Datei je Experiment-ID."""
    out: Dict[str, Path] = {}
    if not RESULTS_DIR.exists():
        return out
    for path in sorted(RESULTS_DIR.glob("*.json")):
        exp = path.name.split("__")[0]
        out[exp] = path            # sortiert -> letzte gewinnt
    return out


def _base_id(exp: str) -> str:
    """`B-012-mnist` -> `B-012`; `Q-003-replication` bleibt."""
    m = re.match(r"^([ABCQ]-\d{3})(?:-(mnist|fashion_mnist))?$", exp)
    return m.group(1) if m and m.group(2) else exp


def _fmt_config(cfg: dict) -> str:
    keys = ["dataset", "datasets", "seeds", "arms", "archs", "ranks", "depths",
            "k_values", "train_sizes", "dimensions", "class_counts", "width",
            "epochs", "lr", "max_samples"]
    parts: List[str] = []
    for k in keys:
        if k not in cfg:
            continue
        v = cfg[k]
        if isinstance(v, list):
            v = f"{len(v)} Werte" if len(v) > 6 else ", ".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def _verdicts(analysis) -> List[str]:
    """Urteile aus der gespeicherten Auswertung ziehen, egal wie verschachtelt."""
    found: List[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "verdict":
                    if isinstance(value, list):
                        found.extend(str(x) for x in value)
                    else:
                        found.append(str(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(analysis)
    return found


def build() -> str:
    files = _latest_per_experiment()
    lines: List[str] = []
    lines.append("# Ergebnisübersicht — jede Aussage mit ihrer Rohdatei")
    lines.append("")
    lines.append("**Diese Datei wird erzeugt, nicht geschrieben.** Sie entsteht aus den")
    lines.append("gespeicherten Rohdaten unter `results/experiments/` und kann deshalb")
    lines.append("nicht von ihnen abweichen. Neu erzeugen mit:")
    lines.append("")
    lines.append("```")
    lines.append("python -m qire_ai.results_report")
    lines.append("```")
    lines.append("")
    lines.append(f"Erzeugt am {datetime.now().strftime('%d.%m.%Y, %H:%M')} Uhr aus "
                 f"{len(files)} gespeicherten Läufen.")
    lines.append("")
    lines.append("Zu jedem Lauf liegen neben der JSON-Datei die Messzeilen als CSV im")
    lines.append("selben Verzeichnis. Jede Datei enthält außerdem die vollständige")
    lines.append("Konfiguration und die Softwareumgebung (Python-, torch-, numpy-,")
    lines.append("scipy-Version, CPU, Thread-Zahl).")
    lines.append("")
    lines.append("---")
    lines.append("")

    total_rows = 0
    for exp in sorted(files):
        payload = json.loads(files[exp].read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        n = payload.get("n_records", 0)
        total_rows += n
        title = TITLES.get(_base_id(exp)) or TITLES.get(exp) or ""
        lines.append(f"## {exp}" + (f" — {title}" if title else ""))
        lines.append("")
        lines.append(f"- **Rohdaten:** `results/experiments/{files[exp].name}`")
        lines.append(f"- **Messzeilen:** {n}")
        lines.append(f"- **Erzeugt:** {payload.get('created', '?')}")
        conf = _fmt_config(cfg)
        if conf:
            lines.append(f"- **Konfiguration:** {conf}")
        for key in ("primary", "primary_comparisons", "prediction", "control",
                    "caveat", "fairness_note", "design"):
            if key in cfg:
                lines.append(f"- **{key}:** {cfg[key]}")
        verdicts = _verdicts(payload.get("analysis", {}))
        if verdicts:
            lines.append("- **Urteil nach den vorher festgelegten Kriterien:**")
            for v in verdicts:
                lines.append(f"  - {v}")
        lines.append("")

    # --- Was NICHT belegt ist ---
    lines.append("---")
    lines.append("")
    lines.append("## Ohne Rohdaten — nur durch einen Neulauf belegbar")
    lines.append("")
    if DIARY.exists():
        text = DIARY.read_text(encoding="utf-8")
        mentioned = sorted(set(re.findall(r"\b([ABCQ]-\d{3})\b", text)))
        have = set(files)
        missing = [m for m in mentioned
                   if not any(h == m or h.startswith(m + "-") for h in have)]
        if missing:
            lines.append("Diese im Tagebuch genannten Experimente haben **keine**")
            lines.append("gespeicherten Rohdaten. Ihre Zahlen stammen aus Konsolenausgaben,")
            lines.append("die nicht mehr existieren:")
            lines.append("")
            for m in missing:
                lines.append(f"- `{m}`")
            lines.append("")
            lines.append("Für die schriftliche Arbeit gilt: entweder neu rechnen oder")
            lines.append("ausdrücklich als nicht mehr reproduzierbar kennzeichnen.")
        else:
            lines.append("Keine — alle im Tagebuch genannten Experimente haben Rohdaten.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Insgesamt **{total_rows} einzelne Messungen** in "
                 f"{len(files)} Läufen.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    text = build()
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"RESULTS.md geschrieben: {OUTPUT}")
    print(f"  {len(text.splitlines())} Zeilen")


if __name__ == "__main__":
    main()
