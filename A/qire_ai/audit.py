from __future__ import annotations

"""Bestandspruefung: welche Behauptung im Tagebuch hat Rohdaten hinter sich?

Nach dem Fund vom 3. August (kein Experimentskript hat je gespeichert) ist die
wichtigste Frage nicht mehr "stimmen die Zahlen", sondern "kann ich sie
ueberhaupt noch belegen". Dieses Modul beantwortet das ueberpruefbar:

  1. Inventar aller gespeicherten Rohdatensaetze
  2. Nachrechnen der Kernzahlen aus den Rohdaten und Abgleich mit dem, was im
     Tagebuch steht -- Abweichungen werden angezeigt, nicht stillschweigend
     hingenommen
  3. Liste aller Experiment-IDs, die das Tagebuch nennt, aber fuer die keine
     Rohdaten existieren. Das sind die Zahlen, die bei einer Nachfrage nur durch
     einen kompletten Neulauf zu belegen waeren.
  4. Umgebung (GPU vorhanden? Threads?)

Run: python -m qire_ai.audit
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .result_store import RESULTS_DIR, load

ROOT = Path(__file__).resolve().parent.parent
DIARY = ROOT / "PROJEKTTAGEBUCH.md"

# (Beschreibung, Experiment, Rechenvorschrift, im Tagebuch behaupteter Wert, Toleranz)
CHECKS: List[Tuple[str, str, str, float, float]] = [
    ("B-003 d1: Block unbeschraenkt vs dicht", "B-003",
     "mean:block_plain:depth=1 - mean:dense:depth=1", +0.0125, 0.0002),
    ("B-003 d1: dichte Schicht", "B-003", "mean:dense:depth=1", 0.9237, 0.0002),
    ("B-003 d3: dichte Schicht", "B-003", "mean:dense:depth=3", 0.9035, 0.0002),
    ("B-004 dense_ends: dicht", "B-004", "mean:dense:arch=dense_ends", 0.9347, 0.0002),
    ("B-004 local_ends: Monarch vs Block", "B-004",
     "mean:monarch_64x64:arch=local_ends - mean:block_diag_128:arch=local_ends",
     +0.0115, 0.0002),
    ("B-004 Permutation: Monarch verliert", "B-004",
     "mean:monarch_64x64:arch=local_ends_perm - mean:monarch_64x64:arch=local_ends",
     -0.0133, 0.0002),
    ("B-004 lowrank_r64 vs dicht (dense_ends)", "B-004",
     "mean:lowrank_r64:arch=dense_ends - mean:dense:arch=dense_ends", +0.0003, 0.0002),
    ("B-006 dense_ends: dicht", "B-006", "mean:dense:arch=dense_ends", 0.9605, 0.0002),
    # ab hier die Laeufe vom 4. August
    ("B-012 mnist: dichte Referenz", "B-012-mnist", "mean:dense:rank=0", 0.9377, 0.0002),
    ("B-012 mnist: Zweig k=0, Rang 64", "B-012-mnist",
     "mean:branch:rank=64:k=0", 0.9297, 0.0002),
    ("B-012 mnist: Zweig k=3, Rang 64", "B-012-mnist",
     "mean:branch:rank=64:k=3", 0.9408, 0.0002),
    ("B-013 mnist: ein Schnitt bei Epoche 5", "B-013-mnist",
     "mean:single_cut", 0.9399, 0.0002),
    ("B-013 mnist: fester Rang 64", "B-013-mnist", "mean:fixed", 0.9303, 0.0002),
]

# Zusaetzliche Pruefungen, die nicht auf `balanced_accuracy` gehen
EXTRA_CHECKS = [
    ("B-012 mnist: Effektivrang nach Epoche 1", "B-012-mnist",
     ("dense_trajectory", "k", 1, "effective_rank"), 5.1, 0.05),
    ("B-012 mnist: Effektivrang nach Epoche 10", "B-012-mnist",
     ("dense_trajectory", "k", 10, "effective_rank"), 6.7, 0.05),
]


def _mean(records, arm: str, **filt) -> float:
    rows = [r for r in records if r["arm"] == arm
            and all(str(r.get(k)) == str(v) for k, v in filt.items())]
    if not rows:
        return float("nan")
    return float(np.mean([r["balanced_accuracy"] for r in rows]))


def _evaluate(expr: str, records) -> float:
    """Winzige Auswertung fuer 'mean:ARM:key=val [- mean:ARM:key=val]'."""
    total, sign = 0.0, +1.0
    for part in re.split(r"\s+-\s+", expr):
        bits = part.split(":")
        arm = bits[1]
        filt = dict(b.split("=") for b in bits[2:] if "=" in b)
        total += sign * _mean(records, arm, **filt)
        sign = -1.0
    return total


def inventory() -> Dict[str, dict]:
    print("=" * 90)
    print("1. INVENTAR DER GESPEICHERTEN ROHDATEN")
    print("=" * 90)
    found: Dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        print("  results/experiments existiert nicht -- nichts gespeichert.")
        return found
    for path in sorted(RESULTS_DIR.glob("*.json")):
        payload = load(path)
        exp = payload["experiment_id"]
        cfg = payload["config"]
        seeds = cfg.get("seeds", [])
        print(f"  {exp:<16} {payload['created']}  {payload['n_records']:>4} Zeilen  "
              f"{len(seeds):>2} Seeds  {cfg.get('dataset') or cfg.get('datasets', '')}")
        found[exp] = payload
    return found


def verify(found: Dict[str, dict]) -> int:
    print()
    print("=" * 90)
    print("2. KERNZAHLEN AUS DEN ROHDATEN GEGEN DIE ANGABEN IM TAGEBUCH")
    print("=" * 90)
    problems = 0
    for label, exp, expr, claimed, tol in CHECKS:
        if exp not in found:
            print(f"  [FEHLT ] {label:<44} -- keine Rohdaten fuer {exp}")
            problems += 1
            continue
        got = _evaluate(expr, found[exp]["records"])
        ok = np.isfinite(got) and abs(got - claimed) <= tol
        problems += 0 if ok else 1
        print(f"  [{'OK    ' if ok else 'ABWEIC'}] {label:<44} "
              f"Tagebuch {claimed:+.4f} | Rohdaten {got:+.4f}")

    for label, exp, (arm, key, val, field), claimed, tol in EXTRA_CHECKS:
        if exp not in found:
            print(f"  [FEHLT ] {label:<44} -- keine Rohdaten fuer {exp}")
            problems += 1
            continue
        rows = [r[field] for r in found[exp]["records"]
                if r["arm"] == arm and r.get(key) == val]
        got = float(np.mean(rows)) if rows else float("nan")
        ok = np.isfinite(got) and abs(got - claimed) <= tol
        problems += 0 if ok else 1
        print(f"  [{'OK    ' if ok else 'ABWEIC'}] {label:<44} "
              f"Tagebuch {claimed:>7.2f} | Rohdaten {got:>7.2f}")
    return problems


def unbacked() -> None:
    print()
    print("=" * 90)
    print("3. IM TAGEBUCH GENANNTE EXPERIMENTE OHNE ROHDATEN")
    print("=" * 90)
    if not DIARY.exists():
        print("  Tagebuch nicht gefunden.")
        return
    text = DIARY.read_text(encoding="utf-8")
    mentioned = sorted(set(re.findall(r"\b([ABCQ]-\d{3})\b", text)))
    have = {p.name.split("__")[0] for p in RESULTS_DIR.glob("*.json")} if RESULTS_DIR.exists() else set()
    # Laeufe, die je Datensatz gespeichert werden, heissen "B-012-mnist" o.ae.
    # Ein exakter Abgleich meldet die Basis-ID faelschlich als unbelegt.
    missing = [m for m in mentioned
               if not any(h == m or h.startswith(m + "-") for h in have)]
    print(f"  genannt: {', '.join(mentioned) if mentioned else '-'}")
    print(f"  mit Rohdaten: {', '.join(sorted(have)) if have else '-'}")
    print()
    if missing:
        print(f"  OHNE ROHDATEN ({len(missing)}): {', '.join(missing)}")
        print("  -> Diese Zahlen sind nur durch einen kompletten Neulauf belegbar.")
        print("     Fuer die schriftliche Arbeit entweder neu rechnen oder als")
        print("     'nicht mehr reproduzierbar' kennzeichnen.")
    else:
        print("  Alle genannten Experimente haben gespeicherte Rohdaten.")


def environment() -> None:
    print()
    print("=" * 90)
    print("4. UMGEBUNG")
    print("=" * 90)
    import torch
    print(f"  torch {torch.__version__}, Threads {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"  GPU vorhanden: {torch.cuda.get_device_name(0)} "
              f"-- Zeitmessungen auf GPU sind moeglich")
    else:
        print("  KEINE GPU verfuegbar. Alle Zeitaussagen gelten nur fuer CPU;")
        print("  Monarchs eigentliches Argument (Hardware-Effizienz auf GPUs)")
        print("  ist auf diesem Rechner nicht pruefbar und muss als Grenze")
        print("  der Arbeit benannt werden.")


def main() -> int:
    found = inventory()
    problems = verify(found)
    unbacked()
    environment()
    print()
    print("=" * 90)
    if problems == 0:
        print("ERGEBNIS: Alle geprueften Kernzahlen stimmen mit den Rohdaten ueberein.")
    else:
        print(f"ERGEBNIS: {problems} Abweichung(en) oder fehlende Rohdaten -- oben nachsehen.")
    print("=" * 90)
    return problems


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
