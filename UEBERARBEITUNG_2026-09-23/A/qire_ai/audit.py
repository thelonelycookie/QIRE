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

ERGAENZT AM 23. SEPTEMBER
  Bis dahin standen hier 15 B-Zahlen -- und ausgerechnet der tragende Befund
  B-014 fehlte, ebenso B-007 bis B-011. Ein "alle Kernzahlen stimmen" sagte
  also nichts ueber die Zahlen, auf denen Projekt B steht. Jetzt geprueft:
  B-014 (Zellmittel UND die pseudoreplizierten Altwerte, getrennt markiert),
  B-011 (TOST), B-010, B-009, B-008, B-007 samt Machtanalyse vom 27. August.
  Jede Zahl stammt aus PROJEKTTAGEBUCH.md oder B/HANDOFF_PROJEKT_B.md.

  Und das Werkzeug lief bisher nur im Archiv: In A fehlen die B-Rohdaten, in B
  fehlte ein Aufruf. Weitere Rohdatenordner lassen sich jetzt angeben -- der
  Standard (results/experiments) bleibt unveraendert:

    python -m qire_ai.audit                                  wie bisher
    python -m qire_ai.audit --daten ..\\B\\daten              aus A, mit B-Daten
    $env:QIRE_B_DATEN = "..\\B\\daten"; python -m qire_ai.audit   dasselbe
    python pruefen_audit.py                                  aus dem Buendel B

Run: python -m qire_ai.audit [--daten ORDNER ...] [--tagebuch DATEI]
"""

import argparse
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .result_store import RESULTS_DIR, load

ROOT = Path(__file__).resolve().parent.parent
DIARY = ROOT / "PROJEKTTAGEBUCH.md"
# Zusaetzliche Rohdatenordner. Der Standardordner wird immer mitgelesen, falls
# er existiert; diese hier kommen dazu. Gesetzt von main().
DATA_DIRS: List[Path] = [RESULTS_DIR]
UMGEBUNGSVARIABLE = "QIRE_B_DATEN"

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


def _json_files() -> List[Path]:
    """Alle Rohdatendateien aus allen Ordnern, nach Dateiname sortiert -- bei
    gleicher ID gewinnt damit die neueste (Zeitstempel im Namen)."""
    files: Dict[str, Path] = {}
    for d in DATA_DIRS:
        if d.exists():
            for p in d.glob("*.json"):
                files.setdefault(p.name, p)
    return [files[k] for k in sorted(files)]


def inventory() -> Dict[str, dict]:
    print("=" * 90)
    print("1. INVENTAR DER GESPEICHERTEN ROHDATEN")
    print("=" * 90)
    for d in DATA_DIRS:
        print(f"  Ordner: {d}{'' if d.exists() else '   (existiert nicht)'}")
    found: Dict[str, dict] = {}
    files = _json_files()
    if not files:
        print("  keine Rohdaten gefunden -- nichts gespeichert.")
        return found
    for path in files:
        payload = load(path)
        if "experiment_id" not in payload or "records" not in payload:
            # z. B. Wiederaufnahme-Zustaende anderer Laeufe
            # (Q-020-pilot_wiederaufnahme.json, 24.09.). Keine Rohdatendatei --
            # bis dahin brach das ganze Audit hier mit KeyError ab.
            print(f"  {path.name:<40} keine Rohdatendatei (ohne experiment_id) "
                  f"-- uebersprungen")
            continue
        exp = payload["experiment_id"]
        cfg = payload["config"]
        seeds = cfg.get("seeds", [])
        print(f"  {exp:<16} {payload['created']}  {payload['n_records']:>4} Zeilen  "
              f"{len(seeds):>2} Seeds  {cfg.get('dataset') or cfg.get('datasets', '')}")
        payload["_datei"] = path.name
        found[exp] = payload
    return found


Zeile = Tuple[str, bool, str]


# ---------------------------------------------------------------------------
# Die tragenden B-Zahlen, ergaenzt am 23. September
# ---------------------------------------------------------------------------
class _Liste:
    """Sammelt Pruefzeilen. Drei Vergleichsarten, weil das Tagebuch Zahlen in
    drei Formen nennt: gerundet auf Stellen, als Zehnerpotenz, als Schranke."""

    def __init__(self) -> None:
        self.zeilen: List[Zeile] = []

    def abs(self, label: str, behauptet: float, ist: float, tol: float,
            fmt: str = "+.4f") -> None:
        ok = bool(np.isfinite(ist) and abs(ist - behauptet) <= tol)
        self.zeilen.append((label, ok, f"Tagebuch {behauptet:{fmt}} | "
                                       f"Rohdaten {ist:{fmt}}"))

    def rel(self, label: str, behauptet: float, ist: float,
            tol: float = 0.06) -> None:
        """p-Werte als Zehnerpotenz ("1,7e-3"): relative Toleranz."""
        ok = bool(np.isfinite(ist) and ist > 0
                  and abs(np.log10(ist / behauptet)) <= np.log10(1 + tol))
        self.zeilen.append((label, ok, f"Tagebuch {behauptet:.2g} | Rohdaten {ist:.2g}"))

    def unter(self, label: str, schranke: float, ist: float) -> None:
        self.zeilen.append((label, bool(ist < schranke),
                            f"Tagebuch <{schranke:g} | Rohdaten {ist:.2g}"))

    def gleich(self, label: str, behauptet, ist) -> None:
        self.zeilen.append((label, behauptet == ist,
                            f"Tagebuch {behauptet} | Rohdaten {ist}"))

    def wahr(self, label: str, ok: bool, detail: str) -> None:
        self.zeilen.append((label, bool(ok), detail))


def _reihe(records, arm: Optional[str] = None, feld: str = "balanced_accuracy",
           **filt) -> np.ndarray:
    rows = [r for r in records if (arm is None or r.get("arm") == arm)
            and all(str(r.get(k)) == str(v) for k, v in filt.items())]
    return np.array([float(r[feld]) for r in sorted(rows, key=lambda r: r["seed"])])


def _equivalence():
    # relativ wie der Rest des Pakets; im Buendel B laedt pruefen_audit.py den
    # Ordner code/ als Paket qire_ai, dann greift derselbe Import
    from .equivalence import holm, tost_paired
    return holm, tost_paired


def check_b014(found) -> List[Zeile]:
    """Tagebuch 4. August (B-014) und 27. August (Pseudoreplikation).

    ZITIERBAR sind nur die Zellmittel-Zeilen. Die pseudoreplizierten Werte
    werden mitgeprueft, damit die Korrektur selbst belegt ist -- sie stehen als
    "Altwert" markiert und duerfen nicht zitiert werden."""
    from scipy import stats

    rec = found["B-014"]["records"]
    L = _Liste()
    dims = sorted({int(r["d"]) for r in rec})
    cls = sorted({int(r["num_classes"]) for r in rec})

    def mittel(key: str) -> Dict[Tuple[int, int], float]:
        return {(d, c): float(np.mean([r[key] for r in rec if int(r["d"]) == d
                                       and int(r["num_classes"]) == c]))
                for d in dims for c in cls}

    tr, un = mittel("effective_rank"), mittel("effective_rank_untrained")
    L.gleich("Gitter: 6 d-Stufen x 4 Klassenzahlen = 24 Zellen",
             24, len(tr))
    L.gleich("10 Seeds je Zelle", 10,
             len([r for r in rec if int(r["d"]) == 2 and int(r["num_classes"]) == 2]))

    def zell_spearman(werte, achse):
        x = [k[achse] for k in werte]
        return stats.spearmanr(x, [werte[k] for k in werte])

    for name, werte, achse, rho, p in (
            ("untrainiert, d -> Rang", un, 0, 0.987, 6.5e-19),
            ("untrainiert, C -> Rang", un, 1, 0.027, 0.90),
            ("trainiert, d -> Rang", tr, 0, 0.606, 1.7e-3),
            ("trainiert, C -> Rang", tr, 1, 0.673, 3.1e-4)):
        s = zell_spearman(werte, achse)
        L.abs(f"Zellmittel {name}: rho", rho, float(s.statistic), 0.0006, "+.3f")
        if p >= 0.01:
            L.abs(f"Zellmittel {name}: p", p, float(s.pvalue), 0.005, ".2f")
        else:
            L.rel(f"Zellmittel {name}: p", p, float(s.pvalue))

    # Die Altwerte ueber 240 Einzelmessungen -- NUR zum Beleg der Korrektur.
    for name, key, feld, rho, p in (
            ("untrainiert, d", "effective_rank_untrained", "d", 0.986, 3.1e-187),
            ("untrainiert, C", "effective_rank_untrained", "num_classes", 0.002, 0.98),
            ("trainiert, d", "effective_rank", "d", 0.565, 1.1e-21),
            ("trainiert, C", "effective_rank", "num_classes", 0.575, 1.7e-22)):
        s = stats.spearmanr([r[feld] for r in rec], [r[key] for r in rec])
        L.abs(f"ALTWERT pseudorepliziert {name}: rho", rho, float(s.statistic),
              0.0006, "+.3f")
        if p >= 0.01:
            L.abs(f"ALTWERT pseudorepliziert {name}: p", p, float(s.pvalue), 0.005, ".2f")
        else:
            L.rel(f"ALTWERT pseudorepliziert {name}: p", p, float(s.pvalue))

    # Innerhalb-Effekte (rechnen schon immer auf Zellmitteln)
    within_c = np.mean([stats.spearmanr(dims, [tr[(d, c)] for d in dims]).statistic
                        for c in cls])
    within_d = np.mean([stats.spearmanr(cls, [tr[(d, c)] for c in cls]).statistic
                        for d in dims])
    L.abs("d-Effekt bei fester Klassenzahl", 0.957, float(within_c), 0.0006, "+.3f")
    L.abs("Klassen-Effekt bei festem d", 0.833, float(within_d), 0.0006, "+.3f")

    # Die beiden Tabellen des Tagebuchs (eine Nachkommastelle)
    for (d, c), wert in {(2, 2): 1.4, (2, 5): 1.7, (2, 10): 1.7, (2, 20): 1.7,
                         (10, 2): 1.5, (10, 5): 2.4, (10, 10): 2.1, (10, 20): 2.7,
                         (100, 2): 1.7, (100, 5): 3.0, (100, 10): 3.7,
                         (100, 20): 4.6}.items():
        L.abs(f"Tabelle trainiert d={d}, C={c}", wert, tr[(d, c)], 0.051, ".2f")
    for (d, c), wert in {(2, 2): 3.7, (2, 5): 3.7, (2, 10): 3.7, (2, 20): 3.7,
                         (10, 2): 19.1, (10, 5): 19.2, (10, 10): 19.1, (10, 20): 19.1,
                         (100, 2): 165.4, (100, 5): 165.6, (100, 10): 165.5,
                         (100, 20): 165.6}.items():
        L.abs(f"Tabelle untrainiert d={d}, C={c}", wert, un[(d, c)], 0.051, ".2f")
    L.abs("Spanne trainiert: kleinster Wert", 1.4, min(tr.values()), 0.051, ".2f")
    L.abs("Spanne trainiert: groesster Wert", 4.7, max(tr.values()), 0.051, ".2f")
    L.abs("Spanne untrainiert: kleinster Wert", 3.7, min(un.values()), 0.051, ".2f")
    L.abs("Spanne untrainiert: groesster Wert (\"165\")", 165.0, max(un.values()),
          1.0, ".1f")
    L.abs("10 Klassen, gemittelt ueber d (\"2,7 statt 9\")", 2.7,
          float(np.mean([tr[(d, 10)] for d in dims])), 0.051, ".2f")
    return L.zeilen


def check_b011(found) -> List[Zeile]:
    """Tagebuch 4. August (B-011) und HANDOFF_PROJEKT_B.md Abschnitt 3."""
    from scipy import stats

    holm, tost = _equivalence()
    L = _Liste()
    for ds in ("mnist", "fashion_mnist"):
        exp = f"B-011-{ds}"
        if exp not in found:
            L.wahr(f"{exp} vorhanden", False, "keine Rohdaten")
            continue
        rec = found[exp]["records"]
        dense = _reihe(rec, "dense")
        L.gleich(f"{ds}: 15 Seeds", 15, len(dense))

        def arm(a, rang):
            return _reihe(rec, a, rank=rang)

        if ds == "mnist":
            L.abs("mnist: dichte Referenz", 0.9403, float(dense.mean()), 0.00005)
            for a, acc, delta in (("scratch", 0.9290, -0.0114),
                                  ("svd_frozen", 0.9407, +0.0003),
                                  ("distill", 0.9387, -0.0016)):
                v = arm(a, 64)
                L.abs(f"mnist Rang 64 {a}: Genauigkeit", acc, float(v.mean()), 0.00005)
                L.abs(f"mnist Rang 64 {a}: vs. dicht", delta,
                      float(v.mean() - dense.mean()), 0.00005)
            L.abs("mnist Rang 64 scratch: 'schlechter (p=0,0067)'", 0.0067,
                  float(stats.wilcoxon(arm("scratch", 64), dense).pvalue), 0.00005)
            for a in ("svd_frozen", "distill"):
                L.unter(f"mnist Rang 64 {a}: TOST +-0,005", 0.0001,
                        tost(arm(a, 64), dense, 0.005)["p_tost"])
            for a, delta in (("svd_frozen", -0.0321), ("distill", -0.0226),
                             ("scratch", -0.0059)):
                L.abs(f"mnist Rang 8 {a}: vs. dicht", delta,
                      float(arm(a, 8).mean() - dense.mean()), 0.00005)
            energie = {"w_energy_8": 0.459, "w_energy_64": 0.646,
                       "d_energy_8": 0.923, "d_energy_64": 0.998}
        else:
            v = arm("svd_frozen", 64)
            L.abs("fashion Rang 64 SVD-Projektion: vs. dicht", 0.0011,
                  float(v.mean() - dense.mean()), 0.00005)
            L.abs("fashion Rang 64 SVD-Projektion: TOST p (\"0,0001\")", 0.0001,
                  tost(v, dense, 0.005)["p_tost"], 0.00005)
            s = arm("scratch", 64)
            L.abs("fashion Rang 64 scratch: vs. dicht", -0.0011,
                  float(s.mean() - dense.mean()), 0.00005)
            L.abs("fashion Rang 64 scratch: p", 0.80,
                  float(stats.wilcoxon(s, dense).pvalue), 0.005, ".2f")
            energie = {"d_energy_8": 0.963, "d_energy_64": 0.999}
        for feld, wert in energie.items():
            L.abs(f"{ds}: {feld} (Spektrum {'W' if feld[0] == 'w' else 'Ausgaben'})",
                  wert, float(np.mean([r[feld] for r in rec if feld in r
                                       and np.isfinite(r[feld])])), 0.0005, ".3f")
    return L.zeilen


def check_b010(found) -> List[Zeile]:
    """Tagebuch 4. August (B-010). Die groesste Stufe (angefordert 14.000)
    wird wie in sample_efficiency.analyse auf die angeforderte Groesse
    zurueckgerundet -- ohne das wurde die Statistik dort am 4. August NaN."""
    from scipy import stats

    payload = found["B-010"]
    rec, cfg = payload["records"], payload["config"]
    L = _Liste()
    sizes = sorted(int(s) for s in cfg["train_sizes"])

    def bucket(n) -> int:
        return min(sizes, key=lambda s: abs(s - int(n)))

    def vals(arm, n, ds):
        rows = sorted((r for r in rec if r["arm"] == arm and r["dataset"] == ds
                       and bucket(r["n_train"]) == n), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    L.gleich("sechs Trainingsgroessen 200 .. 14.000", [200, 14000],
             [sizes[0], sizes[-1]] if len(sizes) == 6 else sizes)
    L.gleich("15 Seeds", 15, len(vals("dense", 200, "mnist")))
    for ds, delta, rho in (("mnist", 0.0149, -0.600), ("fashion_mnist", 0.0139, -0.257)):
        b, d = vals("block_diag_128", 200, ds), vals("dense", 200, ds)
        L.abs(f"{ds} n=200: Blockdiagonale vs. dicht", delta, float(b.mean() - d.mean()),
              0.00005)
        p = float(stats.wilcoxon(b, d).pvalue)
        L.wahr(f"{ds} n=200: 'signifikant' (Wilcoxon p<0,05)", p < 0.05, f"p={p:.4f}")
        deltas = [float(vals("block_diag_128", n, ds).mean() - vals("dense", n, ds).mean())
                  for n in sizes]
        r = float(stats.spearmanr(np.log(sizes), deltas).statistic)
        L.abs(f"{ds}: Trend Blockdiagonale (Spearman, 6 Punkte)", rho, r, 0.0006, "+.3f")
    lr8 = vals("lowrank_r8", 200, "mnist").mean() - vals("dense", 200, "mnist").mean()
    L.abs("mnist n=200: lowrank_r8 vs. dicht", -0.0204, float(lr8), 0.00005)
    verdict = str((payload.get("analysis") or {}).get("verdict", ""))
    L.wahr("gespeichertes Urteil H0", verdict.startswith("H0"), verdict[:60])
    return L.zeilen


def check_b009(found) -> List[Zeile]:
    """Tagebuch 3. August (B-009) und HANDOFF Abschnitt 4. Die Holm-Familie ist
    die aus equivalence.family: Monarch und Blockdiagonale gegen dicht."""
    from scipy import stats

    holm, tost = _equivalence()
    L = _Liste()
    for ds, d_md, p_md, d_mb, siege, p_mb in (
            ("mnist", -0.0045, 0.0010, +0.0094, 27, None),
            ("fashion_mnist", -0.0112, 0.0001, -0.0028, 11, 0.077)):
        exp = f"B-009-{ds}"
        if exp not in found:
            L.wahr(f"{exp} vorhanden", False, "keine Rohdaten")
            continue
        rec = found[exp]["records"]
        dense, mon, blk = (_reihe(rec, "dense"), _reihe(rec, "monarch_64x64"),
                           _reihe(rec, "block_diag_128"))
        L.gleich(f"{ds}: 30 Seeds", 30, len(dense))
        p_raw = [float(stats.wilcoxon(mon, dense).pvalue),
                 float(stats.wilcoxon(blk, dense).pvalue)]
        p_holm = holm(p_raw)[0]
        L.abs(f"{ds}: Monarch vs. dicht", d_md, float(mon.mean() - dense.mean()), 0.00005)
        L.abs(f"{ds}: Monarch vs. dicht, p (Holm)", p_md, p_holm, 0.00005)
        L.abs(f"{ds}: Monarch - Block", d_mb, float(mon.mean() - blk.mean()), 0.00005)
        L.gleich(f"{ds}: Siege Monarch gegen Block", siege, int((mon > blk).sum()))
        p = float(stats.wilcoxon(mon, blk).pvalue)
        if p_mb is None:
            L.unter(f"{ds}: Monarch - Block, p", 0.0001, p)
        else:
            L.abs(f"{ds}: Monarch - Block, p", p_mb, p, 0.0005, ".3f")
            L.abs(f"{ds}: TOST-Schranke Monarch - Block", 0.0052,
                  tost(mon, blk, 0.005)["achieved"], 0.00005)
    return L.zeilen


def check_b008(found) -> List[Zeile]:
    """Tagebuch 3. August (B-008): Rangreihe, 30 Seeds, exakte SVD."""
    holm, tost = _equivalence()
    rec = found["B-008"]["records"]
    L = _Liste()
    dense = _reihe(rec, "dense", dataset="mnist")
    L.gleich("mnist: 30 Seeds", 30, len(dense))
    L.abs("mnist: dicht", 0.9379, float(dense.mean()), 0.00005)
    L.gleich("Parameter dicht", 16781312,
             int(next(r["middle_parameters"] for r in rec if r["arm"] == "dense")))
    energie = {r_: float(np.mean([r[f"energy_at_{r_}"] for r in rec
                                  if r["arm"] == "dense" and r["dataset"] == "mnist"]))
               for r_ in (8, 64, 256)}
    for rang, acc, delta, p_t, par, en in ((8, 0.9332, -0.0047, 0.46, 69632, 0.4598),
                                           (64, 0.9313, -0.0066, 0.72, 528384, 0.6455),
                                           (256, 0.9269, -0.0110, 1.00, 2101248, 0.7416)):
        v = _reihe(rec, f"lowrank_r{rang}", dataset="mnist")
        L.abs(f"mnist Rang {rang}: Genauigkeit", acc, float(v.mean()), 0.00005)
        L.abs(f"mnist Rang {rang}: vs. dicht", delta, float(v.mean() - dense.mean()),
              0.00005)
        L.abs(f"mnist Rang {rang}: TOST p", p_t, tost(v, dense, 0.005)["p_tost"],
              0.005, ".2f")
        L.gleich(f"Parameter Rang {rang}", par,
                 int(next(r["middle_parameters"] for r in rec
                          if r["arm"] == f"lowrank_r{rang}")))
        L.abs(f"kumulierte Energie der Gewichte bei Rang {rang}", en, energie[rang],
              0.00005)
    # "Kein einziger Rang erreicht Gleichwertigkeit innerhalb +-0,005, auf
    # keinem der beiden Datensaetze."
    for ds in ("mnist", "fashion_mnist"):
        d_ = _reihe(rec, "dense", dataset=ds)
        pt = [tost(_reihe(rec, f"lowrank_r{r_}", dataset=ds), d_, 0.005)["p_tost"]
              for r_ in (4, 8, 16, 32, 64, 128, 256)]
        L.wahr(f"{ds}: kein Rang gleichwertig (alle TOST p >= 0,05)",
               min(pt) >= 0.05, f"kleinster TOST p={min(pt):.3f}")
    return L.zeilen


def check_b007(found) -> List[Zeile]:
    """Tagebuch 4./5. August (B-007) und 27. August (Machtanalyse). Die neueste
    Datei (beide Datensaetze, 20 Seeds) -- die aeltere enthaelt nur mnist."""
    from scipy import stats

    holm, tost = _equivalence()
    rec = found["B-007"]["records"]
    L = _Liste()

    def v(arm, ds, depth):
        return _reihe(rec, arm, dataset=ds, depth=depth)

    L.gleich("mnist: 20 Seeds je Zelle", 20, len(v("orthogonal", "mnist", 1)))
    diffs = {(ds, t): v("orthogonal", ds, t) - v("spectral", ds, t)
             for ds in ("mnist", "fashion_mnist") for t in (1, 3)}
    p_m = [float(stats.wilcoxon(v("orthogonal", "mnist", t),
                                v("spectral", "mnist", t)).pvalue) for t in (1, 3)]
    p_h = holm(p_m)
    for i, (t, delta, siege, ph, cd, pt) in enumerate(((1, 0.0025, 17, 0.0020, 0.84, 0.0007),
                                                        (3, 0.0051, 17, 0.0001, 1.18, 0.54))):
        d_ = diffs[("mnist", t)]
        L.abs(f"mnist Tiefe {t}: orthogonal - spectral", delta, float(d_.mean()), 0.00005)
        L.gleich(f"mnist Tiefe {t}: Siege", siege, int((d_ > 0).sum()))
        L.abs(f"mnist Tiefe {t}: p (Holm)", ph, p_h[i], 0.00005)
        L.abs(f"mnist Tiefe {t}: Cohen's d", cd, float(d_.mean() / d_.std(ddof=1)),
              0.005, "+.2f")
        tp = tost(v("orthogonal", "mnist", t), v("spectral", "mnist", t), 0.005)["p_tost"]
        if pt < 0.01:
            L.abs(f"mnist Tiefe {t}: TOST p", pt, tp, 0.00005)
        else:
            L.abs(f"mnist Tiefe {t}: TOST p", pt, tp, 0.005, ".2f")
    for arm, a1, a3 in (("plain", 0.9409, 0.9344), ("orthogonal", 0.9497, 0.9500)):
        L.abs(f"mnist {arm}: Tiefe 1", a1, float(v(arm, "mnist", 1).mean()), 0.00005)
        L.abs(f"mnist {arm}: Tiefe 3", a3, float(v(arm, "mnist", 3).mean()), 0.00005)
    for t, delta, p in ((1, 0.0048, 0.114), (3, -0.0045, 0.133)):
        d_ = diffs[("fashion_mnist", t)]
        L.abs(f"fashion Tiefe {t}: orthogonal - spectral", delta, float(d_.mean()), 0.00005)
        L.abs(f"fashion Tiefe {t}: p", p, float(stats.wilcoxon(
            v("orthogonal", "fashion_mnist", t), v("spectral", "fashion_mnist", t)).pvalue),
            0.0005, ".3f")
    for ds in ("mnist", "fashion_mnist"):
        w = v("weight_decay", ds, 3)
        L.wahr(f"{ds} Tiefe 3: weight_decay exakt 0,1000, Streuung 0",
               bool(len(w) and np.all(w == w[0]) and abs(w[0] - 0.1) < 1e-12),
               f"Mittel {w.mean():.4f}, Streuung {w.std():.4f}")
    verdict = str((found["B-007"].get("analysis") or {}).get("verdict", ""))
    L.wahr("gespeichertes Urteil HU", verdict.startswith("HU"), verdict[:60])

    # Machtanalyse vom 27. August: Nachweisgrenze bei 80 % Macht, alpha=0,05,
    # gepaart, Normalnaeherung (z_0,975 + z_0,8) * SD / sqrt(n).
    z = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
    for (ds, t), sd, grenze in ((("mnist", 1), 0.0030, 0.0022),
                                (("mnist", 3), 0.0043, 0.0031),
                                (("fashion_mnist", 1), 0.0123, 0.0089),
                                (("fashion_mnist", 3), 0.0126, 0.0091)):
        s = float(diffs[(ds, t)].std(ddof=1))
        L.abs(f"Machtanalyse {ds} Tiefe {t}: SD der Paardifferenz", sd, s, 0.00005)
        L.abs(f"Machtanalyse {ds} Tiefe {t}: Nachweisgrenze (15 Seeds)", grenze,
              z * s / np.sqrt(15), 0.00005)
    L.abs("Machtanalyse fashion Tiefe 1: Grenze mit 20 Seeds", 0.0077,
          z * float(diffs[("fashion_mnist", 1)].std(ddof=1)) / np.sqrt(20), 0.00005)
    return L.zeilen


def check_b004_positivkontrolle(found) -> List[Zeile]:
    """HANDOFF_PROJEKT_B.md Abschnitt 3, Messfehlerquelle 1: im blinden Aufbau
    trennt sich ein 32-fach groesseres Budget nicht (p=0,23)."""
    from scipy import stats

    rec = found["B-004"]["records"]
    L = _Liste()
    b = _reihe(rec, "block_diag_128", arch="dense_ends")
    d = _reihe(rec, "dense", arch="dense_ends")
    L.abs("dense_ends: Blockdiagonale vs. dicht, p", 0.23,
          float(stats.wilcoxon(b, d).pvalue), 0.005, ".2f")
    return L.zeilen


B_PRUEFUNGEN: Dict[str, Tuple[Sequence[str], Callable[[Dict[str, dict]], List[Zeile]]]] = {
    "B-014 -- der tragende Befund": (("B-014",), check_b014),
    "B-011 -- Gleichwertigkeit der SVD-Projektion (TOST)":
        (("B-011-mnist", "B-011-fashion_mnist"), check_b011),
    "B-010 -- Datenmenge gegen Struktur": (("B-010",), check_b010),
    "B-009 -- Monarch, 30 Seeds": (("B-009-mnist", "B-009-fashion_mnist"), check_b009),
    "B-008 -- Rangreihe": (("B-008",), check_b008),
    "B-007 -- Orthogonalitaet gegen Spektralnorm, mit Machtanalyse":
        (("B-007",), check_b007),
    "B-004 -- Positivkontrolle des blinden Aufbaus": (("B-004",), check_b004_positivkontrolle),
}


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

    # --- die tragenden B-Zahlen (seit 23. September) ---
    for titel, (ids, fn) in B_PRUEFUNGEN.items():
        print()
        print(f"  ### {titel}")
        fehlend = [i for i in ids if i not in found]
        if len(fehlend) == len(ids):
            print(f"  [FEHLT ] keine Rohdaten fuer {', '.join(ids)}")
            problems += 1
            continue
        try:
            zeilen = fn(found)
        except Exception as exc:                  # noqa: BLE001
            print(f"  [FEHLER] Pruefung fehlgeschlagen: {exc!r}")
            problems += 1
            continue
        for label, ok, detail in zeilen:
            problems += 0 if ok else 1
            print(f"  [{'OK    ' if ok else 'ABWEIC'}] {label:<52} {detail}")
    return problems


def unbacked() -> None:
    print()
    print("=" * 90)
    print("3. IM TAGEBUCH GENANNTE EXPERIMENTE OHNE ROHDATEN")
    print("=" * 90)
    if not DIARY.exists():
        print(f"  Tagebuch nicht gefunden: {DIARY}")
        return
    print(f"  Tagebuch: {DIARY}")
    text = DIARY.read_text(encoding="utf-8")
    mentioned = sorted(set(re.findall(r"\b([ABCQ]-\d{3})\b", text)))
    have = {p.name.split("__")[0] for p in _json_files()}
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
        if not any(h.startswith("B-") for h in have):
            print(f"  HINWEIS: keine B-Rohdaten in den gelesenen Ordnern. Aus A heraus")
            print(f"  mit --daten ..\\B\\daten oder ${UMGEBUNGSVARIABLE} aufrufen.")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    global DIARY, DATA_DIRS
    ap = argparse.ArgumentParser(description="Tagebuchzahlen der B-Serie gegen "
                                             "die Rohdaten nachrechnen.")
    ap.add_argument("--daten", action="append", default=[],
                    help="zusaetzlicher Rohdatenordner (mehrfach moeglich), "
                         "z. B. ..\\B\\daten")
    ap.add_argument("--tagebuch", default=None,
                    help=f"Tagebuchdatei fuer Abschnitt 3 (Standard: {DIARY.name})")
    args = ap.parse_args(argv)
    extra = list(args.daten)
    if os.environ.get(UMGEBUNGSVARIABLE):
        extra += os.environ[UMGEBUNGSVARIABLE].split(os.pathsep)
    DATA_DIRS = [RESULTS_DIR] + [Path(p).resolve() for p in extra if p]
    if args.tagebuch:
        DIARY = Path(args.tagebuch).resolve()

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
