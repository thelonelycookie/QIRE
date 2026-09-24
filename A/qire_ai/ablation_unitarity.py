from __future__ import annotations

"""RESEARCH ID: B-007 -- Ist es wirklich die Unitaritaet, oder nur Skalenkontrolle?

WARUM DIESER LAUF NOETIG IST
Befund 4 meiner Gesamtbilanz lautet: "Unitaritaet gegen Spektralnorm, kein
Unterschied (p=0,32)". Er stammt aus B-002. Am 3. August habe ich denselben
Vergleich auf den Rohdaten von B-003 nachgerechnet -- gleiche Architektur,
gleiche Blockgroesse, gleiche Tiefe, gleiche Epochenzahl, ebenfalls 10 Seeds,
nur andere Seeds -- und bekam +0,0068 mit p=0,037 statt p=0,32.

Zwei Laeufe desselben Experiments, gegenteilige Schlussfolgerung. Damit ist
Befund 4 nicht belastbar, und er ist kein Nebenschauplatz: Orthogonalitaet ist
das klassische Gegenstueck zu unitaeren Quantengattern, also der
quanteninspirierte Mechanismus selbst. Faellt Befund 4, aendert das die
Kernaussage von Projekt A.

WAS HIER ANDERS GEMACHT WIRD
1. **20 Seeds statt 10.** Bei der beobachteten Effektstaerke (d ~ 0,7) liegt die
   Teststaerke damit bei etwa 80 %; mit 10 Seeds war es ein Muenzwurf, ob der
   Effekt gefunden wird. Genau daran sind beide Vorlaeufer gescheitert.
2. **Lernrate je Arm gesucht**, nicht fest auf 0,01. In B-004-lr hat sich
   gezeigt, dass ein fester Wert einzelne Arme systematisch benachteiligt --
   die dichte Schicht verlor dadurch rund 0,006 Genauigkeit.
3. **Zweiter Datensatz.** Fashion-MNIST als unabhaengige Wiederholung. MNIST ist
   und bleibt der vorregistrierte Primaerdatensatz; Fashion-MNIST entscheidet
   nichts allein, sondern zeigt, ob der Befund ueberhaupt uebertraegt.
4. **Aequivalenztest, wenn nichts signifikant wird.** "p > 0,05" ist kein Beleg
   fuer Gleichheit. Nur ein bestandener TOST erlaubt es, Befund 4 positiv zu
   bestaetigen statt ihn bloss nicht zu widerlegen.

DIE ARME
  plain        unbeschraenkte Bloecke (Basis)
  orthogonal   ALLE Singulaerwerte = 1; norm- und winkeltreu. Das unitaere
               Gegenstueck. Ueber matrix_exp einer antisymmetrischen Matrix.
  spectral     nur der GROESSTE Singulaerwert = 1. Begrenzt die Skala, ist aber
               kontrahierend statt normerhaltend.
  frobenius    Gesamtenergie auf die eines orthogonalen Blocks fixiert,
               Spektrum sonst frei.
  weight_decay unbeschraenkt, Adam weight_decay=1e-4 -- der gewoehnliche
               klassische Regularisierer.

Wichtig fuer die Einordnung: ein orthogonaler d x d-Block hat nur d(d-1)/2 = 6
echte Freiheitsgrade statt 16, bei identischem Speicher. Gewinnt er trotzdem,
ist das die Signatur einer echten strukturellen Einschraenkung -- nicht die
Folge von mehr Kapazitaet.

VORREGISTRIERUNG
  PRIMAER : orthogonal gegen spectral, auf MNIST, bei Tiefe 1 und Tiefe 3.
            Zwei Tests, Holm-korrigiert innerhalb dieser Familie.
  Alles andere (gegen plain, gegen frobenius, weight_decay, Fashion-MNIST)
  ist sekundaer und wird nicht als Beleg gefuehrt, sondern zur Einordnung.

  H1 : orthogonal schlaegt spectral auf MNIST nach Holm-Korrektur, und
       Fashion-MNIST zeigt dieselbe Richtung.
       -> Befund 4 ist falsch. Der Effekt haengt an der Unitaritaet, nicht an
          blosser Skalenkontrolle. Das ist ein POSITIVER Befund fuer den
          quanteninspirierten Mechanismus -- der erste im ganzen Projekt.
  H0 : kein signifikanter Unterschied UND TOST weist Gleichwertigkeit
       innerhalb +-0,005 nach.
       -> Befund 4 ist bestaetigt, und zwar erstmals positiv statt nur
          "nicht widerlegt".
  HU : weder noch.
       -> unentschieden; die Datenlage reicht auch mit 20 Seeds nicht. Dann
          wird die erreichte Schranke berichtet und Befund 4 bleibt als
          unsicher markiert.

FALSIFIKATION
H1 verlangt beides: Holm-korrigierte Signifikanz auf MNIST *und* gleiche
Richtung auf Fashion-MNIST. Ein einzelner knapper p-Wert auf einem Datensatz
reicht ausdruecklich nicht -- genau daran ist der Vorlaeufer gescheitert.

Run: python -m qire_ai.ablation_unitarity
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .ablation_orthogonality import ConstrainedBlock
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .equivalence import MARGIN, holm, tost_paired
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder

WIDTH = 4096
BLOCK_SIZE = 4
ARMS = ("plain", "orthogonal", "spectral", "frobenius", "weight_decay")
DEPTHS = (1, 3)
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(1701, 1721))       # 20 Seeds
TUNING_SEEDS = (1301, 1302, 1303)
LR_GRID = (0.003, 0.01)
WEIGHT_DECAY = 1e-4


def _cell(mode: str, depth: int, seed: int, dataset: str, lr: float,
          width: int, max_samples: int, epochs: int, batch_size: int,
          threads: int) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                      lambda: ConstrainedBlock(width, BLOCK_SIZE, mode),
                      mode, "dense_ends", depth=depth)
    info = train_model(model, bundle.x_train, bundle.y_train, epochs=epochs,
                       lr=lr, seed=seed, batch_size=batch_size,
                       weight_decay=WEIGHT_DECAY if mode == "weight_decay" else 0.0)
    _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
    return {"arm": mode, "depth": depth, "seed": seed, "dataset": dataset, "lr": lr,
            "balanced_accuracy": float(ex["balanced_accuracy"]),
            "middle_parameters": model.middle_parameters(),
            "train_time_sec": float(info["training_time_sec"]),
            "train_best_loss": float(info["train_best_loss"]),
            "nan_detected": str(info.get("nan_detected", "False"))}


def sweep_lr(dataset: str, width: int, max_samples: int, epochs: int,
             batch_size: int, threads: int,
             depths: Sequence[int] = DEPTHS) -> Dict[Tuple[str, int], float]:
    best: Dict[Tuple[str, int], float] = {}
    for depth in depths:
        for arm in ARMS:
            means = {}
            for lr in LR_GRID:
                vals = [_cell(arm, depth, s, dataset, lr, width, max_samples,
                              epochs, batch_size, threads)["balanced_accuracy"]
                        for s in TUNING_SEEDS]
                means[lr] = float(np.mean(vals))
            best[(arm, depth)] = max(means, key=means.get)
            print(f"    {dataset:>14} d{depth} / {arm:<13} -> lr {best[(arm, depth)]}  "
                  + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(width: int = WIDTH, seeds: Sequence[int] = EVAL_SEEDS,
        datasets: Sequence[str] = DATASETS, depths: Sequence[int] = DEPTHS,
        max_samples: int = 6000, epochs: int = 10, batch_size: int = 256,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="B-007",
        description=("Orthogonalitaet gegen Spektralnormalisierung und weitere "
                     "Norm-Kontrollen, 20 Seeds, zwei Datensaetze, Lernrate je Arm. "
                     "Vorregistrierte Wiederholung von Befund 4 der Gesamtbilanz."),
        config={"datasets": list(datasets), "depths": list(depths), "arms": list(ARMS),
                "width": width, "block_size": BLOCK_SIZE, "seeds": list(seeds),
                "max_samples": max_samples, "epochs": epochs, "batch_size": batch_size,
                "torch_threads": threads, "weight_decay": WEIGHT_DECAY,
                "lr_grid": list(LR_GRID), "margin": MARGIN,
                "primary": "orthogonal vs spectral auf mnist, Tiefe 1 und 3, Holm",
                "primary_dataset": PRIMARY_DATASET,
                "seed_note": "Auswertungs-Seeds ab 1701, Lernraten auf 1301-1303"},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, width, max_samples, epochs, batch_size, threads,
                          depths=depths)
        rec.config[f"lr_map_{dataset}"] = {f"{a}|d{d}": v for (a, d), v in lr_map.items()}
        for i, seed in enumerate(seeds, 1):
            for depth in depths:
                for arm in ARMS:
                    rec.add(**_cell(arm, depth, seed, dataset, lr_map[(arm, depth)],
                                    width, max_samples, epochs, batch_size, threads))
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
        # Nach JEDEM Datensatz sichern. Der erste Anlauf dieses Laufs ist in
        # Fashion-MNIST abgebrochen und hat die komplette, fertige MNIST-Phase
        # mitgerissen -- 200 Zellen, weil erst ganz am Ende gespeichert wurde.
        # Mehrfaches Speichern legt je einen Zwischenstand ab; die letzte Datei
        # enthaelt alles.
        print(f"    Zwischenstand nach {dataset} gesichert:")
        rec.save()
    return rec


def _vals(records, arm: str, depth: int, dataset: str) -> np.ndarray:
    rows = sorted((r for r in records if r["arm"] == arm and r["depth"] == depth
                   and r["dataset"] == dataset), key=lambda r: r["seed"])
    return np.array([r["balanced_accuracy"] for r in rows])


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    datasets = [str(d) for d in config["datasets"]]
    depths = [int(d) for d in config["depths"]]
    out: Dict[str, object] = {"per_dataset": {}}

    n = len(_vals(records, "plain", depths[0], datasets[0]))
    print()
    print(f"RESEARCH ID B-007 | Breite {config['width']}, Blockgroesse "
          f"{config['block_size']}, {n} Seeds, Datensaetze: {', '.join(datasets)}")

    # ---- Uebersicht ----
    for dataset in datasets:
        for depth in depths:
            print()
            print(f"--- {dataset}, Tiefe {depth} " + "-" * max(4, 44 - len(dataset)))
            head = f'{"Arm":>14} {"accuracy":>9} {"std":>7} {"lr":>7} {"vs plain":>10}'
            print(head); print("-" * len(head))
            base = _vals(records, "plain", depth, dataset)
            for arm in ARMS:
                v = _vals(records, arm, depth, dataset)
                if len(v) == 0:
                    continue
                lr = next(r["lr"] for r in records if r["arm"] == arm
                          and r["depth"] == depth and r["dataset"] == dataset)
                delta = "" if arm == "plain" else f"{v.mean()-base.mean():>+10.4f}"
                flag = ""
                if v.std(ddof=1) == 0.0:
                    flag = "  <- Streuung exakt 0, Arm kollabiert"
                print(f'{arm:>14} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} {lr:>7} '
                      f'{delta:>10}{flag}')

    # ---- der vorregistrierte Primaertest ----
    print()
    print("=" * 82)
    print("PRIMAERTEST -- orthogonal gegen spectral auf "
          f"{PRIMARY_DATASET}, Holm ueber beide Tiefen")
    primary: Dict[str, object] = {}
    raw_p, cells = [], []
    for depth in depths:
        o = _vals(records, "orthogonal", depth, PRIMARY_DATASET)
        s = _vals(records, "spectral", depth, PRIMARY_DATASET)
        raw_p.append(float(stats.wilcoxon(o, s).pvalue))
        cells.append((depth, o, s))
    for (depth, o, s), p, p_h in zip(cells, raw_p, holm(raw_p)):
        eq = tost_paired(o, s, MARGIN)
        diff = o - s
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden, reicht erst ab +-{eq['achieved']:.4f}")
        print(f'  Tiefe {depth}: {o.mean()-s.mean():+.4f}  {int((o>s).sum())}/{len(o)}  '
              f'p={p:.4f}  Holm={p_h:.4f}  TOST p={eq["p_tost"]:.4f}  d={d:+.2f}  -> {state}')
        primary[f"depth{depth}"] = {"delta": float(o.mean()-s.mean()), "p": p,
                                    "p_holm": p_h, "p_tost": eq["p_tost"],
                                    "achieved": eq["achieved"]}

    # ---- Wiederholung auf dem zweiten Datensatz ----
    print()
    print("WIEDERHOLUNG (sekundaer, entscheidet nichts allein):")
    replication: Dict[str, object] = {}
    for dataset in [d for d in datasets if d != PRIMARY_DATASET]:
        for depth in depths:
            o = _vals(records, "orthogonal", depth, dataset)
            s = _vals(records, "spectral", depth, dataset)
            if len(o) == 0:
                continue
            p = float(stats.wilcoxon(o, s).pvalue)
            print(f'  {dataset}, Tiefe {depth}: {o.mean()-s.mean():+.4f}  p={p:.4f}')
            replication[f"{dataset}_depth{depth}"] = {"delta": float(o.mean()-s.mean()),
                                                      "p": p}

    # ---- Urteil nach den vorher festgelegten Kriterien ----
    same_direction = all(v["delta"] > 0 for v in replication.values()) or \
                     all(v["delta"] < 0 for v in replication.values())
    holm_sig = any(v["p_holm"] < 0.05 for v in primary.values())       # type: ignore[index]
    primary_dir = [v["delta"] for v in primary.values()]               # type: ignore[index]
    repl_dir = [v["delta"] for v in replication.values()]              # type: ignore[index]
    consistent = same_direction and repl_dir and primary_dir and \
        (np.sign(np.mean(primary_dir)) == np.sign(np.mean(repl_dir)))
    all_equivalent = all(v["p_tost"] < 0.05 for v in primary.values()) # type: ignore[index]

    print()
    if holm_sig and consistent:
        verdict = ("H1 -- Befund 4 der Gesamtbilanz ist FALSCH. Orthogonalitaet "
                   "schlaegt blosse Spektralkontrolle, auf beiden Datensaetzen in "
                   "derselben Richtung. Das ist ein positiver Befund fuer den "
                   "quanteninspirierten Mechanismus.")
    elif holm_sig and not consistent:
        verdict = ("HU -- auf mnist signifikant, aber die Wiederholung zeigt nicht "
                   "dieselbe Richtung. Nach dem vorregistrierten Kriterium reicht "
                   "das nicht. Befund 4 bleibt unsicher.")
    elif all_equivalent:
        verdict = ("H0 -- Befund 4 ist BESTAETIGT, und zwar positiv: der "
                   f"Aequivalenztest weist Gleichwertigkeit innerhalb +-{MARGIN} "
                   "nach. Der Effekt haengt am Spektralradius, nicht an Unitaritaet.")
    else:
        worst = max(v["achieved"] for v in primary.values())           # type: ignore[index]
        verdict = (f"HU -- unentschieden. Weder Unterschied noch Gleichwertigkeit "
                   f"nachgewiesen; die Daten tragen erst ab +-{worst:.4f}. Befund 4 "
                   f"bleibt als unsicher markiert.")
    print(f"  {verdict}")
    print("=" * 82)
    out["primary"] = primary
    out["replication"] = replication
    out["verdict"] = verdict
    return out


if __name__ == "__main__":
    recorder = run()
    result = analyse(recorder)
    recorder.save(analysis=result)
