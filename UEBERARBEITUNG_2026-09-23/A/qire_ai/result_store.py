from __future__ import annotations

"""Persistenz fuer die Ablations-Experimente.

WARUM ES DIESE DATEI GIBT
Die Experimentskripte (ablation_*, regime_*, structured_pruning, ...) haben ihre
Ergebnisse bisher nur auf die Konsole gedruckt. Nach dem Schliessen des Fensters
war jede Zahl weg -- reproduzierbar nur durch einen kompletten Neulauf, und ohne
Rohdaten, die man einer Jury vorlegen oder spaeter neu auswerten koennte.

Diese Datei speichert pro Lauf:
  * jede einzelne Messung als eigene Zeile (Arm, Seed, Genauigkeit, Parameter,
    Trainingszeit) -- nicht nur den Mittelwert
  * die vollstaendige Konfiguration des Laufs
  * die Softwareumgebung (Python-, torch-, numpy-, scipy-Version, CPU, Threads)

Damit ist eine Auswertung ohne Neulauf moeglich, und ein spaeterer Neulauf ist
gegen die alte Datei vergleichbar.

Ablage: results/experiments/<ID>__<zeitstempel>.json  (vollstaendig)
        results/experiments/<ID>__<zeitstempel>.csv   (nur die Messzeilen)
"""

import csv
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "experiments"


def _versions() -> Dict[str, str]:
    """Versionen der Pakete, von denen die Zahlen abhaengen."""
    out: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unbekannt",
    }
    for name in ("torch", "numpy", "scipy", "sklearn"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unbekannt"))
        except Exception:
            out[name] = "nicht installiert"
    try:
        import torch

        out["torch_num_threads"] = str(torch.get_num_threads())
    except Exception:
        pass
    return out


@dataclass
class Recorder:
    """Sammelt Messzeilen waehrend eines Laufs und schreibt sie am Ende weg.

    Eine Zeile pro (Arm, Seed) -- die Rohdaten, nicht die Zusammenfassung.
    """

    experiment_id: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **fields: Any) -> None:
        self.records.append(dict(fields))

    def save(self, analysis: Optional[Dict[str, Any]] = None) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = RESULTS_DIR / f"{self.experiment_id}__{stamp}"
        # Zwei Sicherungen in derselben Sekunde haetten sich frueher still
        # ueberschrieben (gefunden im T-002-Rauchtest, 24. September). Jetzt
        # bekommt die zweite ein Suffix; "_02" sortiert nach dem Original, also
        # findet `latest()` weiterhin die neueste Datei.
        zaehler = 2
        while base.with_suffix(".json").exists():
            base = RESULTS_DIR / f"{self.experiment_id}__{stamp}_{zaehler:02d}"
            zaehler += 1

        payload = {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config": self.config,
            "environment": _versions(),
            "n_records": len(self.records),
            "records": self.records,
            "analysis": analysis or {},
        }
        base.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if self.records:
            keys: List[str] = []
            for rec in self.records:
                for k in rec:
                    if k not in keys:
                        keys.append(k)
            with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.records)

        print(f"\n  Rohdaten gespeichert: {base.with_suffix('.json')}")
        print(f"  Messzeilen ({len(self.records)}): {base.with_suffix('.csv')}")
        return base.with_suffix(".json")


def latest(experiment_id: str) -> Optional[Path]:
    """Neueste gespeicherte Datei zu einer Experiment-ID, oder None."""
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob(f"{experiment_id}__*.json"))
    return files[-1] if files else None


def load(experiment_id_or_path: str | Path) -> Dict[str, Any]:
    """Gespeicherten Lauf laden -- per Pfad oder per Experiment-ID (neueste)."""
    path = Path(experiment_id_or_path)
    if not path.exists():
        found = latest(str(experiment_id_or_path))
        if found is None:
            raise FileNotFoundError(
                f"Kein gespeicherter Lauf fuer '{experiment_id_or_path}' in {RESULTS_DIR}"
            )
        path = found
    return json.loads(path.read_text(encoding="utf-8"))


def records_of(experiment_id_or_path: str | Path) -> List[Dict[str, Any]]:
    return load(experiment_id_or_path)["records"]


def group(records: Iterable[Dict[str, Any]], *keys: str) -> Dict[tuple, List[Dict[str, Any]]]:
    """Messzeilen nach einem oder mehreren Feldern gruppieren."""
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for rec in records:
        out.setdefault(tuple(rec[k] for k in keys), []).append(rec)
    return out


def series(records: Iterable[Dict[str, Any]], value: str, *keys: str) -> Dict[tuple, List[float]]:
    """Wie `group`, gibt aber direkt die Messwerte in Seed-Reihenfolge zurueck.

    Die Reihenfolge ist nach `seed` sortiert, damit gepaarte Tests ueber
    verschiedene Arme hinweg wirklich dieselben Seeds paaren.
    """
    grouped = group(records, *keys)
    return {
        k: [float(r[value]) for r in sorted(rows, key=lambda r: r["seed"])]
        for k, rows in grouped.items()
    }
