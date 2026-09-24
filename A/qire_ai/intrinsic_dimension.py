from __future__ import annotations

"""RESEARCH ID: B-014 -- Setzen die Daten den Rang, oder das Netz?

DIE FRAGE
Eine trainierte 4096er-Schicht arbeitet in rund 64 Richtungen. Woher kommt
diese Zahl? Zwei Erklaerungen, und sie sind vollkommen verschieden:

  (a) das NETZ erzeugt sie -- eine Eigenschaft von Architektur und Optimierer
  (b) die DATEN erzwingen sie -- MNIST liegt bekanntlich auf einer
      niedrigdimensionalen Mannigfaltigkeit im 784-dimensionalen Pixelraum

Bisher steht dazu nichts fest. Alle Datensaetze des Projekts waren VORGEFUNDEN;
ihre intrinsische Dimension ist unbekannt und nicht einstellbar. Damit laesst
sich die Frage prinzipiell nicht beantworten -- man kann nur korrelieren.

DER AUFBAU: EINE ECHTE UNABHAENGIGE VARIABLE
Hier werden die Daten selbst gebaut, mit vorgegebener intrinsischer Dimension d:

    z ~ N(0, I_d)                     latenter Raum, Dimension d
    x = A z + sigma * eps             Einbettung in 784 Dimensionen
    y = argmax(C z)                   Label haengt NUR von z ab, nicht vom Rauschen

A ist eine feste zufaellige 784xd-Matrix mit orthonormalen Spalten, C eine feste
zufaellige Klassenmatrix. Damit ist d die einzige Variable, die sich aendert --
Stichprobenzahl, Eingangsdimension, Klassenzahl und Rauschstaerke bleiben
gleich.

Das ist das erste Experiment in diesem Projekt mit einer wirklich kontrollierten
unabhaengigen Variable. Alles bisherige war ein Vergleich zwischen vorgefundenen
Bedingungen.

VORREGISTRIERUNG
  H1 : Der gemessene Effektivrang der Mittelschicht waechst monoton mit d.
       Spearman(d, Effektivrang) > 0, p < 0,05.
       -> Die Daten setzen den Rang. Dann ist die noetige Schichtbreite eine
          Eigenschaft der AUFGABE und im Prinzip vor dem Training bestimmbar.
  H0 : kein Zusammenhang -- der Rang ist eine Eigenschaft des Netzes und sagt
       ueber die Aufgabe nichts aus.

  ZUSATZ, schaerfer: Wenn der Effektivrang nicht nur mit d waechst, sondern
  ungefaehr GLEICH d ist, waere der Zusammenhang nicht nur monoton, sondern
  quantitativ. Berichtet wird deshalb auch das Verhaeltnis Effektivrang/d.

  KONTROLLE: dieselbe Messung an einem UNTRAINIERTEN Netz. Findet sich der
  Zusammenhang schon dort, kaeme er aus der Einbettung der Daten und nicht aus
  dem Lernen. Ohne diese Kontrolle waere der Befund nicht interpretierbar --
  genau der Fehler, der in diesem Projekt bei der Verschraenkungsentropie
  schon einmal passiert ist.

Run: python -m qire_ai.intrinsic_dimension
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder
from .spectral_probe import middle_outputs, output_spectrum, train_with_probe

WIDTH = 4096
INPUT_DIM = 784
NUM_CLASSES = 10
N_SAMPLES = 6000
TEST_FRACTION = 0.30
DIMENSIONS: Tuple[int, ...] = (2, 5, 10, 20, 50, 100)
# Zweite kontrollierte Variable. Im Vortest lag der Effektivrang bei d=50 nicht
# bei 50, sondern bei 8,5 -- auffaellig nah an Klassenzahl minus eins. Ob die
# Datendimension oder die Klassenzahl den Rang setzt, laesst sich nur trennen,
# wenn man BEIDE variiert. Mit nur einer waere jede Antwort eine Vermutung.
CLASS_COUNTS: Tuple[int, ...] = (2, 5, 10, 20)
ENERGY_RANKS = (8, 32, 64, 256)
EVAL_SEEDS = tuple(range(2201, 2211))       # 10 Seeds
EPOCHS = 10
LR = 0.003
BATCH = 256
NOISE = 0.10


def make_manifold_data(d: int, seed: int, n: int = N_SAMPLES,
                       input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES,
                       noise: float = NOISE):
    """Daten mit exakt bekannter intrinsischer Dimension d."""
    g = torch.Generator().manual_seed(seed * 1000 + d)
    z = torch.randn(n, d, generator=g)
    # orthonormale Einbettung: die Mannigfaltigkeit ist genau d-dimensional
    A, _ = torch.linalg.qr(torch.randn(input_dim, d, generator=g))
    x = z @ A.T
    x = x / x.std() * 0.5
    x = x + noise * torch.randn(n, input_dim, generator=g)
    C = torch.randn(num_classes, d, generator=g)
    y = (z @ C.T).argmax(dim=1)
    cut = int(n * (1 - TEST_FRACTION))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def run(dimensions: Sequence[int] = DIMENSIONS, seeds: Sequence[int] = EVAL_SEEDS,
        class_counts: Sequence[int] = CLASS_COUNTS, width: int = WIDTH,
        epochs: int = EPOCHS, lr: float = LR,
        batch_size: int = BATCH, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="B-014",
        description=("Synthetische Daten mit vorgegebener intrinsischer Dimension d. "
                     "Prueft, ob der Effektivrang der Mittelschicht von den Daten "
                     "gesetzt wird. Mit untrainierter Kontrolle."),
        config={"dimensions": list(dimensions), "seeds": list(seeds), "width": width,
                "class_counts": list(class_counts),
                "input_dim": INPUT_DIM, "num_classes": NUM_CLASSES,
                "n_samples": N_SAMPLES, "noise": NOISE, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "torch_threads": threads,
                "energy_ranks": list(ENERGY_RANKS),
                "primary": "Spearman(d, Effektivrang) > 0",
                "control": "dieselbe Messung am untrainierten Netz"},
    )
    grid = [(d, c) for d in dimensions for c in class_counts]
    for i, seed in enumerate(seeds, 1):
        for d, classes in grid:
            set_seed(seed)
            torch.set_num_threads(threads)
            x_tr, y_tr, x_te, y_te = make_manifold_data(d, seed, num_classes=classes)
            model = MiddleNet(INPUT_DIM, width, classes,
                              lambda: nn.Linear(width, width), f"d{d}c{classes}",
                              "dense_ends", depth=1)
            before = output_spectrum(middle_outputs(model, x_tr, max_rows=2048),
                                     ENERGY_RANKS)
            train_with_probe(model, x_tr, y_tr, epochs=epochs, lr=lr,
                             batch_size=batch_size, seed=seed)
            after = output_spectrum(middle_outputs(model, x_tr, max_rows=2048),
                                    ENERGY_RANKS)
            _, _, _, ex = evaluate(model, x_te, y_te)
            # Spektrum der Rohdaten als weitere Bezugsgroesse
            raw = output_spectrum(x_tr[:2048], ENERGY_RANKS)
            rec.add(d=d, num_classes=classes, seed=seed,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    effective_rank=after["effective_rank"],
                    effective_rank_untrained=before["effective_rank"],
                    effective_rank_raw_data=raw["effective_rank"],
                    entropy_rank=after["entropy_rank"],
                    energy_64=after["energy_64"],
                    energy_64_untrained=before["energy_64"])
        print(f"  seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    dims = [int(d) for d in config["dimensions"]]
    out: Dict[str, object] = {}

    classes = sorted({int(r["num_classes"]) for r in records})

    def cell(d, c, key):
        return np.array([r[key] for r in records
                         if r["d"] == d and r["num_classes"] == c])

    n_seeds = len(cell(dims[0], classes[0], "effective_rank"))
    print()
    print(f"RESEARCH ID B-014 | synthetische Mannigfaltigkeiten, {n_seeds} Seeds, "
          f"Breite {config['width']}")
    print("Effektivrang der Mittelschicht nach dem Training:")
    label = "d / Klassen"
    head = f'  {label:>12} ' + " ".join(f'{c:>9}' for c in classes)
    print(head); print("  " + "-" * (len(head) - 2))
    for d in dims:
        line = f'  {d:>12} '
        for c in classes:
            line += f'{float(cell(d, c, "effective_rank").mean()):>9.1f} '
        print(line)

    print()
    print("Dieselbe Tabelle am UNTRAINIERTEN Netz (Kontrolle):")
    print(head); print("  " + "-" * (len(head) - 2))
    for d in dims:
        line = f'  {d:>12} '
        for c in classes:
            line += f'{float(cell(d, c, "effective_rank_untrained").mean()):>9.1f} '
        print(line)

    # --- welche Variable erklaert den Rang? ---
    d_all, c_all, r_all, u_all = [], [], [], []
    for d in dims:
        for c in classes:
            for v_, u_ in zip(cell(d, c, "effective_rank"),
                              cell(d, c, "effective_rank_untrained")):
                d_all.append(d); c_all.append(c); r_all.append(float(v_))
                u_all.append(float(u_))
    rho_d, p_d = stats.spearmanr(d_all, r_all)
    rho_c, p_c = stats.spearmanr(c_all, r_all)
    rho_du, p_du = stats.spearmanr(d_all, u_all)
    rho_cu, p_cu = stats.spearmanr(c_all, u_all)

    print()
    print("Welche Variable sagt den Rang vorher?")
    print(f'  trainiert  : Spearman(d, Rang)       rho={rho_d:+.3f}  p={p_d:.2e}')
    print(f'               Spearman(Klassen, Rang) rho={rho_c:+.3f}  p={p_c:.2e}')
    print(f'  untrainiert: Spearman(d, Rang)       rho={rho_du:+.3f}  p={p_du:.2e}')
    print(f'               Spearman(Klassen, Rang) rho={rho_cu:+.3f}  p={p_cu:.2e}')

    # partiell: Zusammenhang mit d INNERHALB jeder Klassenzahl und umgekehrt
    within_c = [stats.spearmanr(dims, [float(cell(d, c, "effective_rank").mean())
                                       for d in dims]).statistic for c in classes]
    within_d = [stats.spearmanr(classes, [float(cell(d, c, "effective_rank").mean())
                                          for c in classes]).statistic for d in dims]
    print()
    print(f'  d-Effekt innerhalb fester Klassenzahl : rho im Mittel '
          f'{np.nanmean(within_c):+.3f}')
    print(f'  Klassen-Effekt innerhalb festem d     : rho im Mittel '
          f'{np.nanmean(within_d):+.3f}')

    # Vergleich gegen die Vorhersage "Rang = Klassen - 1"
    print()
    print('  Vergleich mit der Vorhersage "Rang = Klassenzahl - 1":')
    for c in classes:
        got = float(np.mean([float(cell(d, c, "effective_rank").mean()) for d in dims]))
        print(f'    Klassen {c:>3}: vorhergesagt {c-1:>3}, gemessen {got:>6.1f} '
              f'(gemittelt ueber alle d)')

    out.update({"rho_d": float(rho_d), "p_d": float(p_d),
                "rho_classes": float(rho_c), "p_classes": float(p_c),
                "rho_d_untrained": float(rho_du),
                "rho_classes_untrained": float(rho_cu),
                "within_class_d_effect": float(np.nanmean(within_c)),
                "within_d_class_effect": float(np.nanmean(within_d))})

    print()
    print("=" * 78)
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    d_wins = abs(np.nanmean(within_c)) > abs(np.nanmean(within_d))
    if p_d < 0.05 and np.nanmean(within_c) > 0 and d_wins:
        v = ("H1 -- die intrinsische Datendimension setzt den Rang; sie erklaert "
             "mehr als die Klassenzahl.")
    elif p_c < 0.05 and np.nanmean(within_d) > 0 and not d_wins:
        v = ("H1' -- nicht die Datendimension, sondern die KLASSENZAHL setzt den "
             "Rang. Das ist eine andere und schaerfere Aussage als die "
             "vorregistrierte: die noetige Breite haengt an der Aufgabe, nicht an "
             "der Geometrie der Eingaben.")
    else:
        v = ("H0 -- keine der beiden Variablen erklaert den Rang monoton. Der Rang "
             "ist keine einfache Funktion von Datendimension oder Klassenzahl.")
    if rho_du > 0 and p_du < 0.05:
        v += (" Vorbehalt: das UNTRAINIERTE Netz zeigt den d-Zusammenhang bereits; "
              "der Anteil, der aus dem Lernen stammt, ist nur die Differenz zur "
              "Kontrolle.")
    print(f"  {v}")
    print("=" * 78)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    recorder.save(analysis=analyse(recorder))
