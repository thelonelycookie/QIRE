from __future__ import annotations

"""RESEARCH ID: B-008 -- Sagt das Spektrum der dichten Schicht den noetigen Rang
vorher?

WORAUS DAS FOLGT
B-005 hat gemessen, dass das Training die Spektralenergie der dichten
4096x4096-Schicht von 5,4 % auf 71 % in die 64 groessten von 4096 Richtungen
presst. Das war eine Beobachtung. Sie macht aber eine Vorhersage, und die laesst
sich pruefen:

    Wenn die dichte Schicht effektiv nur wenige Richtungen nutzt, muss die
    Genauigkeit einer Low-Rank-Schicht ungefaehr dort saettigen, wo die
    kumulierte Spektralenergie saettigt.

Damit wird aus einer Messung eine Regel -- und eine Regel ist ueberpruefbar,
uebertragbar und deutlich mehr wert als noch ein Vergleich.

AUFBAU
Fuer jeden Datensatz und jeden Seed:
  1. dichte Mittelschicht trainieren; Genauigkeit messen UND das vollstaendige
     Singulaerwertspektrum der gelernten Matrix berechnen (exakte SVD, nicht
     genaehert -- an dieser Zahl haengt die ganze Aussage)
  2. fuer jeden Rang r eine Low-Rank-Schicht mit demselben Budget trainieren und
     die Genauigkeit messen

Zwei Datensaetze: MNIST und Fashion-MNIST. Gleiche Bildgroesse, gleiche
Klassenzahl, deutlich verschiedene Schwierigkeit. 30 Seeds, weil der
Aequivalenztest bei 10 Seeds nur eine Schranke von +-0,008 hergab und die
Aussage +-0,005 tragen soll.

DIE VORHERSAGE, VORHER FESTGELEGT
  r_acc    kleinster Rang, der der dichten Schicht innerhalb +-0,005
           nachweislich gleichwertig ist (TOST, alpha=0,05)
  r_E(t)   kleinster Rang, dessen mittlere kumulierte Spektralenergie >= t ist

  H1 (Vorhersage haelt): Es gibt EINE Schwelle t aus {0,90; 0,95; 0,99}, fuer
      die r_E(t) bei BEIDEN Datensaetzen mit r_acc uebereinstimmt -- hoechstens
      um einen Gitterschritt daneben (Faktor 2).
  H0: Keine einzelne Schwelle passt fuer beide Datensaetze. Dann sagt das
      Spektrum den noetigen Rang nicht vorher und B-005 bleibt eine
      Beschreibung.

Die Bedingung "dieselbe Schwelle fuer beide Datensaetze" ist der Kern. Ohne sie
koennte ich t im Nachhinein je Datensatz passend waehlen, und die "Vorhersage"
waere wertlos.

Run: python -m qire_ai.rank_prediction
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .ablation_locality import _one_cell, middle_factory
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .equivalence import MARGIN, tost_paired
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder

WIDTH = 4096
RANKS: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
DATASETS = ("mnist", "fashion_mnist")
EVAL_SEEDS = tuple(range(1601, 1631))
TUNING_SEEDS = (1301, 1302, 1303)
# 0,03 ist in B-004-lr bei JEDEM Arm um rund 0,05 abgefallen und wird deshalb
# hier nicht erneut vermessen. Das ist eine dokumentierte Entscheidung auf Basis
# vorhandener Daten, keine stillschweigende Auslassung.
LR_GRID = (0.003, 0.01)
THRESHOLDS = (0.90, 0.95, 0.99)


def spectrum_energy(weight: torch.Tensor, ranks: Sequence[int]) -> Dict[int, float]:
    """Anteil der Spektralenergie in den r groessten Singulaerwerten, exakt.

    Bewusst die volle SVD statt `svd_lowrank`: die Naeherung wird bei den
    kleineren der Top-256-Werte ungenau, und genau diese Zahlen tragen hier die
    Aussage.
    """
    s = torch.linalg.svdvals(weight.double())
    energy = (s ** 2)
    total = float(energy.sum())
    cumulative = torch.cumsum(energy, dim=0)
    return {r: float(cumulative[min(r, len(s)) - 1] / total) for r in ranks}


def sweep_lr(dataset: str, arms: Sequence[str], width: int, groups: int,
             max_samples: int, epochs: int, batch_size: int, threads: int,
             arch: str = "dense_ends") -> Dict[str, float]:
    """Beste Lernrate je Arm auf Tuning-Seeds, getrennt fuer jeden Datensatz."""
    best: Dict[str, float] = {}
    for arm in arms:
        means = {}
        for lr in LR_GRID:
            vals = [_one_cell(arm, arch, seed, width, groups, max_samples, epochs,
                              lr, batch_size, threads, measure_time=False,
                              dataset=dataset)["balanced_accuracy"]
                    for seed in TUNING_SEEDS]
            means[lr] = float(np.mean(vals))
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<14} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(
    width: int = WIDTH,
    seeds: Sequence[int] = EVAL_SEEDS,
    datasets: Sequence[str] = DATASETS,
    ranks: Sequence[int] = RANKS,
    max_samples: int = 6000,
    epochs: int = 10,
    batch_size: int = 256,
    threads: int = 6,
    groups: int = 16,
    arch: str = "dense_ends",
) -> Recorder:
    arms = ["dense"] + [f"lowrank_r{r}" for r in ranks]
    rec = Recorder(
        experiment_id="B-008",
        description=("Rangreihe gegen dichte Schicht auf zwei Datensaetzen, plus "
                     "das exakte Spektrum der gelernten dichten Matrix. Prueft, ob "
                     "die Spektralkonzentration den noetigen Rang vorhersagt."),
        config={"datasets": list(datasets), "ranks": list(ranks), "width": width,
                "seeds": list(seeds), "arch": arch, "max_samples": max_samples,
                "epochs": epochs, "batch_size": batch_size, "torch_threads": threads,
                "groups": groups, "margin": MARGIN, "thresholds": list(THRESHOLDS),
                "lr_grid": list(LR_GRID),
                "prediction": ("eine Schwelle t fuer beide Datensaetze; r_E(t) darf "
                               "hoechstens einen Gitterschritt von r_acc abweichen"),
                "seed_note": "Auswertungs-Seeds ab 1601, Lernraten auf 1301-1303"},
    )

    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, arms, width, groups, max_samples, epochs,
                          batch_size, threads, arch)
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)

        for i, seed in enumerate(seeds, 1):
            # dichte Schicht: Genauigkeit UND Spektrum aus demselben Modell
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
            model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                              lambda: nn.Linear(width, width), "dense", arch,
                              groups=groups, depth=1)
            info = train_model(model, bundle.x_train, bundle.y_train, epochs=epochs,
                               lr=lr_map["dense"], seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            weight = model.middles[0].weight.detach()
            energies = spectrum_energy(weight, ranks)
            rec.add(arm="dense", dataset=dataset, seed=seed, rank=0,
                    lr=lr_map["dense"],
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=model.middle_parameters(),
                    train_time_sec=float(info["training_time_sec"]),
                    **{f"energy_at_{r}": energies[r] for r in ranks})

            for r in ranks:
                arm = f"lowrank_r{r}"
                row = _one_cell(arm, arch, seed, width, groups, max_samples, epochs,
                                lr_map[arm], batch_size, threads, measure_time=False,
                                dataset=dataset)
                row["rank"] = r
                rec.add(**row)
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    ranks = [int(r) for r in config["ranks"]]
    datasets = [str(d) for d in config["datasets"]]
    width = int(config["width"])
    out: Dict[str, object] = {"per_dataset": {}}

    print()
    print(f"RESEARCH ID B-008 | Rangreihe und Spektrum, Grenze +-{MARGIN}")

    r_acc: Dict[str, object] = {}
    r_energy: Dict[str, Dict[float, object]] = {}

    for dataset in datasets:
        def acc(arm):
            rows = sorted((r for r in records
                           if r["dataset"] == dataset and r["arm"] == arm),
                          key=lambda r: r["seed"])
            return np.array([r["balanced_accuracy"] for r in rows])

        dense = acc("dense")
        energies = {r: float(np.mean([rec[f"energy_at_{r}"] for rec in records
                                      if rec["dataset"] == dataset and rec["arm"] == "dense"]))
                    for r in ranks}

        print()
        print(f"--- {dataset} (n={len(dense)} Seeds) " + "-" * max(4, 40 - len(dataset)))
        header = (f'{"Rang":>7} {"Parameter":>12} {"accuracy":>9} {"vs dicht":>10} '
                  f'{"TOST p":>8} {"Schranke":>10} {"kum. Energie":>13}')
        print(header); print("-" * len(header))
        print(f'{"dicht":>7} {width*width+width:>12,} {dense.mean():>9.4f} {"":>10} '
              f'{"":>8} {"":>10} {1.0:>13.4f}')
        equivalent: List[int] = []
        for r in ranks:
            v = acc(f"lowrank_r{r}")
            if len(v) != len(dense):
                continue
            eq = tost_paired(v, dense, MARGIN)
            ok = eq["p_tost"] < 0.05
            if ok:
                equivalent.append(r)
            params = 2 * width * r + width
            print(f'{r:>7} {params:>12,} {v.mean():>9.4f} {eq["mean"]:>+10.4f} '
                  f'{eq["p_tost"]:>8.4f} {eq["achieved"]:>10.4f} {energies[r]:>13.4f}'
                  + ("  <- gleichwertig" if ok else ""))

        r_acc[dataset] = min(equivalent) if equivalent else None
        r_energy[dataset] = {}
        for t in THRESHOLDS:
            hit = [r for r in ranks if energies[r] >= t]
            r_energy[dataset][t] = min(hit) if hit else None
        print()
        print(f'  kleinster gleichwertiger Rang r_acc          : {r_acc[dataset]}')
        for t in THRESHOLDS:
            print(f'  kleinster Rang mit Energie >= {t:.2f}  r_E     : {r_energy[dataset][t]}')
        out["per_dataset"][dataset] = {          # type: ignore[index]
            "r_acc": r_acc[dataset],
            "r_energy": {str(t): r_energy[dataset][t] for t in THRESHOLDS},
            "energies": energies,
            "dense_accuracy": float(dense.mean()),
        }

    # ---- die vorher festgelegte Pruefung ----
    print()
    print("=" * 78)
    print("VORHERSAGE-PRUEFUNG (Kriterium vor dem Lauf festgelegt)")
    print("Eine einzige Schwelle muss fuer BEIDE Datensaetze passen, hoechstens")
    print("einen Gitterschritt (Faktor 2) daneben.")
    winner = None
    for t in THRESHOLDS:
        ok_all, detail = True, []
        for dataset in datasets:
            ra, re_ = r_acc[dataset], r_energy[dataset][t]
            if ra is None or re_ is None:
                ok_all = False
                detail.append(f"{dataset}: r_acc={ra}, r_E={re_}")
                continue
            ratio = max(ra, re_) / min(ra, re_)
            ok_all &= ratio <= 2.0
            detail.append(f"{dataset}: r_acc={ra} vs r_E={re_} (Faktor {ratio:.1f})")
        mark = "PASST" if ok_all else "passt nicht"
        print(f"  Schwelle {t:.2f}: {mark:<12} " + " | ".join(detail))
        if ok_all and winner is None:
            winner = t
    print()
    if winner is not None:
        verdict = (f"H1 -- die Spektralkonzentration sagt den noetigen Rang vorher. "
                   f"Eine Schwelle von {winner:.2f} kumulierter Energie passt auf "
                   f"beiden Datensaetzen.")
    else:
        verdict = ("H0 -- keine einzelne Schwelle passt auf beide Datensaetze. Das "
                   "Spektrum sagt den noetigen Rang nicht vorher; B-005 bleibt eine "
                   "Beschreibung ohne Vorhersagekraft.")
    print(f"  {verdict}")
    print("=" * 78)
    out["verdict"] = verdict
    out["threshold"] = winner
    return out


if __name__ == "__main__":
    recorder = run()
    result = analyse(recorder)
    recorder.save(analysis=result)
