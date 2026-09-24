from __future__ import annotations

"""Aequivalenztests und Korrektur fuer multiples Testen -- auf gespeicherten Daten.

WARUM
Meine Kernaussage lautet inzwischen: "Rang 64 ersetzt eine dichte
4096x4096-Schicht ohne messbaren Verlust." Das ist eine AEQUIVALENZ-Behauptung.
Ein Wilcoxon-Test mit p=0,85 belegt sie nicht -- er sagt nur "Unterschied nicht
nachgewiesen". Das ist nicht dasselbe, und es steht als Vorbehalt schon in
meinem eigenen Tagebuch.

Das richtige Werkzeug ist TOST (two one-sided tests): zwei einseitige Tests
gegen eine vorher festgelegte Grenze. Erst wenn BEIDE die Grenze ausschliessen,
darf man "gleichwertig innerhalb dieser Grenze" sagen.

Zweitens: ueber alle Vergleiche eines Laufs habe ich bisher nicht korrigiert.
B-004 rechnet rund 15 Tests je Lauf; bei alpha=0,05 ist knapp ein falsches
"signifikant" allein durch die Anzahl zu erwarten. Holm-Korrektur ist die
konservative Standardantwort und wird hier je Testfamilie angewandt.

DIE GRENZE
Vorab festgelegt: +-0,005 balancierte Genauigkeit. Begruendung: das ist kleiner
als die Seed-zu-Seed-Streuung des dichten Arms (std 0,0065 bis 0,0109 ueber die
Laeufe), also strenger als das Rauschen, gegen das verglichen wird. Zusaetzlich
wird die ERREICHTE Grenze berichtet -- die kleinste Schranke, bei der der Test
noch bestanden haette. Damit haengt die Aussage nicht an einer einzelnen
willkuerlichen Zahl.

Run: python -m qire_ai.equivalence
"""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from .result_store import load

MARGIN = 0.005
ALPHA = 0.05


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni: gibt die korrigierten p-Werte in Eingabereihenfolge."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)          # Monotonie erzwingen
        adjusted[idx] = min(1.0, running)
    return adjusted


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float = MARGIN,
                alpha: float = ALPHA) -> Dict[str, float]:
    """Gepaarter Aequivalenztest.

    Gibt den TOST-p-Wert, das 90-%-Intervall der Differenz und die kleinste
    Grenze zurueck, bei der Aequivalenz noch nachgewiesen waere. Fuer TOST auf
    Niveau alpha ist genau das der groessere Betrag der 1-2*alpha-Grenzen.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        return {"mean": mean, "p_tost": 0.0 if abs(mean) < margin else 1.0,
                "ci_low": mean, "ci_high": mean, "achieved": abs(mean), "n": n}
    df = n - 1
    # zwei einseitige Tests gegen -margin und +margin
    p_lower = stats.t.sf((mean + margin) / se, df)     # H0: d <= -margin
    p_upper = stats.t.cdf((mean - margin) / se, df)    # H0: d >= +margin
    p_tost = float(max(p_lower, p_upper))
    crit = stats.t.ppf(1 - alpha, df)                  # 90-%-Intervall bei alpha=0,05
    lo, hi = mean - crit * se, mean + crit * se
    return {"mean": mean, "p_tost": p_tost, "ci_low": float(lo), "ci_high": float(hi),
            "achieved": float(max(abs(lo), abs(hi))), "n": n}


def _series(records, arm: str, **filters) -> np.ndarray:
    rows = [r for r in records if r["arm"] == arm
            and all(r.get(k) == v for k, v in filters.items())]
    return np.array([r["balanced_accuracy"] for r in sorted(rows, key=lambda r: r["seed"])])


def family(records, baseline: str, arms: Iterable[str], label: str,
           margin: float = MARGIN, **filters) -> List[Dict[str, object]]:
    """Alle Arme gegen eine Basis, mit Holm-Korrektur und Aequivalenztest."""
    base = _series(records, baseline, **filters)
    rows: List[Dict[str, object]] = []
    raw_p: List[float] = []
    for arm in arms:
        vals = _series(records, arm, **filters)
        if len(vals) != len(base) or len(vals) == 0:
            continue
        p_diff = float(stats.wilcoxon(vals, base).pvalue)
        eq = tost_paired(vals, base, margin)
        rows.append({"arm": arm, "p_diff": p_diff, **eq})
        raw_p.append(p_diff)
    for row, adj in zip(rows, holm(raw_p)):
        row["p_diff_holm"] = adj

    print(f"\n  {label}  (Basis: {baseline}, n={rows[0]['n'] if rows else 0}, "
          f"Grenze +-{margin})")
    head = (f'    {"Arm":>26} {"Differenz":>10} {"p":>8} {"p Holm":>8} '
            f'{"90%-KI":>20} {"TOST p":>8}  Lesart')
    print(head)
    print("    " + "-" * (len(head) - 4))
    for row in rows:
        ci = f'[{row["ci_low"]:+.4f},{row["ci_high"]:+.4f}]'
        if row["p_diff_holm"] < ALPHA:
            reading = "Unterschied belegt"
        elif row["p_tost"] < ALPHA:
            reading = f'gleichwertig (+-{margin})'
        else:
            reading = f'unentschieden, reicht erst ab +-{row["achieved"]:.4f}'
        print(f'    {row["arm"]:>26} {row["mean"]:>+10.4f} {row["p_diff"]:>8.4f} '
              f'{row["p_diff_holm"]:>8.4f} {ci:>20} {row["p_tost"]:>8.4f}  {reading}')
    return rows


def main() -> None:
    print("=" * 100)
    print("AEQUIVALENZ UND MULTIPLES TESTEN -- nachtraeglich auf gespeicherten Rohdaten")
    print(f"Vorab festgelegte Grenze: +-{MARGIN} balancierte Genauigkeit, alpha={ALPHA}")
    print("=" * 100)

    b004 = load("B-004")["records"]
    equal = ["block_diag_128", "block_diag_128_spectral", "monarch_64x64",
             "tt_c2_r64", "lowrank_r64", "block_diag_4"]

    print("\n### B-004 -- jede Struktur gegen die DICHTE Schicht")
    print("### (das ist die Familie, die die Kernaussage traegt)")
    for arch in ["dense_ends", "local_ends", "local_ends_perm"]:
        family(b004, "dense", equal, f"B-004 / {arch}", arch=arch)

    print("\n\n### B-006 -- dieselbe Frage nach laengerem Training")
    b006 = load("B-006")["records"]
    for arch in ["dense_ends", "local_ends"]:
        family(b006, "dense", ["block_diag_128", "monarch_64x64", "tt_c2_r64"],
               f"B-006 / {arch}", arch=arch)

    print("\n\n### B-003 -- traegt die Beschraenkung etwas bei?")
    b003 = load("B-003")["records"]
    for depth in (1, 3):
        family(b003, "block_plain", ["block_spectral", "block_orthogonal"],
               f"B-003 / depth {depth}", depth=depth)

    print("\n" + "=" * 100)
    print("LESEHILFE")
    print('  "Unterschied belegt"  -- haelt auch nach Holm-Korrektur')
    print('  "gleichwertig"        -- TOST schliesst beide Grenzen aus; das ist ein')
    print('                           positiver Nachweis, nicht blosses p>0,05')
    print('  "unentschieden"       -- weder Unterschied noch Gleichwertigkeit belegt.')
    print('                           Die genannte Schranke sagt, wie grob die Aussage')
    print('                           bei dieser Seed-Zahl ueberhaupt sein kann.')
    print("=" * 100)


if __name__ == "__main__":
    main()
