from __future__ import annotations

"""RESEARCH ID: B-010 -- Gewinnt Struktur, wenn die Daten knapp werden?

DIE OFFENE ACHSE
Alle bisherigen Vergleiche liefen mit derselben Datenmenge (4.200
Trainingsbeispiele). Damit ist die Frage nie gestellt worden, fuer die weniger
Parameter klassischerweise sprechen: Ein kleineres Modell ueberanpasst weniger
und sollte gerade dann vorne liegen, wenn wenig Daten da sind. Datenbedarf steht
seit Beginn auf meiner Ressourcenliste und ist die einzige Achse, die ich fuer
Strukturen nie angefasst habe.

ZWEI FALLEN, DIE HIER GESCHLOSSEN WERDEN

1. **Der Testsatz darf nicht mitschrumpfen.** `load_dataset` teilt 30 % ab; mit
   `max_samples=250` waeren das 75 Testbeispiele, und die Genauigkeit haette
   eine Streuung, gegen die jeder Effekt untergeht. Hier wird deshalb EINMAL
   gross geladen (20.000) und nur der TRAININGSsatz verkleinert. Der Testsatz
   ist in allen Bedingungen derselbe und gross.

2. **Gleiche Epochenzahl waere kein fairer Vergleich.** Bei 200 Beispielen und
   10 Epochen sieht das Modell 2.000 Beispiele, bei 14.000 Beispielen 140.000 --
   die kleine Bedingung waere schlicht untertrainiert, und "kleines Modell
   gewinnt bei wenig Daten" waere in Wahrheit "alle sind untertrainiert".
   Deshalb wird die Epochenzahl so gesetzt, dass die Zahl der
   Gradientenschritte ueber alle Datenmengen etwa gleich bleibt (Ziel: 170
   Schritte, das entspricht der bisherigen Einstellung 4.200 Beispiele /
   10 Epochen / Batch 256).

   Das ist eine Entscheidung mit Nebenwirkung, die ich benenne: bei gleicher
   Schrittzahl laeuft ein kleiner Datensatz sehr viel oefter durch dieselben
   Beispiele, Ueberanpassung wird also beguenstigt. Genau darum geht es aber --
   die Frage ist, welches Modell damit besser umgeht.

STRATIFIZIERT wird ausserdem: die Teilmenge zieht aus jeder Klasse gleich viele
Beispiele. Bei 200 Beispielen und 10 Klassen waere eine zufaellige Ziehung sonst
merklich unbalanciert, und balanced_accuracy wuerde das bestrafen.

VORREGISTRIERUNG
  PRIMAER: fuer jeden strukturierten Arm die Spearman-Korrelation zwischen
           log(Trainingsgroesse) und dem Abstand zur dichten Schicht, auf
           MNIST. Holm-korrigiert ueber die Arme.
           Vorhersage: NEGATIV -- der Vorteil der Struktur schrumpft, je mehr
           Daten da sind.
  SEKUNDAER: bei der kleinsten Datenmenge -- ist der Abstand positiv und
           signifikant?

  H1 : mindestens ein Arm zeigt eine signifikant negative Korrelation UND
       liegt bei der kleinsten Datenmenge vorn, mit gleicher Richtung auf
       Fashion-MNIST.
       -> Struktur zahlt sich bei knappen Daten aus. Das waere der erste
          positive Effizienzbefund von Projekt B, auf einer Achse, die noch
          niemand geprueft hat.
  H0 : keine Korrelation ueberlebt, oder die dichte Schicht liegt ueberall
       vorn.
       -> auch die Datenachse gibt nichts her.

Ein einzelner Gewinnpunkt bei einer Datenmenge reicht ausdruecklich nicht --
verlangt ist ein TREND ueber die ganze Reihe. Genau daran ist heute schon
mehrfach ein Befund gescheitert, der an einem einzelnen Punkt gut aussah.

Run: python -m qire_ai.sample_efficiency
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .ablation_locality import middle_factory
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder

WIDTH = 4096
POOL_SAMPLES = 20000          # einmal gross laden -> grosser, fester Testsatz
TRAIN_SIZES: Tuple[int, ...] = (200, 400, 800, 1600, 4200, 14000)
TARGET_STEPS = 170            # 4.200 / Batch 256 * 10 Epochen
ARMS = ("dense", "lowrank_r8", "monarch", "block_diag_128")
STRUCTURED = tuple(a for a in ARMS if a != "dense")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(1801, 1816))       # 15 Seeds
TUNING_SEEDS = (1301, 1302)
LR_GRID = (0.003, 0.01)


def stratified_subset(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Indizes einer klassenweise gleich grossen Teilmenge der Groesse ~n."""
    g = np.random.default_rng(seed)
    classes = torch.unique(y).tolist()
    per_class = max(1, n // len(classes))
    picked: List[int] = []
    for c in classes:
        idx = torch.nonzero(y == c, as_tuple=False).flatten().numpy()
        take = min(per_class, len(idx))
        picked.extend(g.choice(idx, size=take, replace=False).tolist())
    g.shuffle(picked)
    return torch.tensor(picked[:n], dtype=torch.long)


def epochs_for(n_train: int, batch_size: int) -> int:
    batches = max(1, math.ceil(n_train / batch_size))
    return max(1, round(TARGET_STEPS / batches))


def _cell(arm: str, n_train: int, seed: int, dataset: str, lr: float, width: int,
          batch_size: int, threads: int, groups: int = 16) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=POOL_SAMPLES)
    idx = stratified_subset(bundle.y_train, n_train, seed)
    x_tr, y_tr = bundle.x_train[idx], bundle.y_train[idx]
    epochs = epochs_for(len(idx), batch_size)
    model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                      middle_factory(arm, width), arm, "dense_ends",
                      groups=groups, depth=1)
    info = train_model(model, x_tr, y_tr, epochs=epochs, lr=lr, seed=seed,
                       batch_size=batch_size)
    _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
    return {"arm": arm, "dataset": dataset, "n_train": int(len(idx)),
            # angeforderte Groesse mitschreiben: die stratifizierte Ziehung
            # erreicht sie nicht immer exakt, und ohne diese Spalte laesst sich
            # nachtraeglich nicht sauber gruppieren
            "n_requested": int(n_train), "seed": seed,
            "lr": lr, "epochs": epochs,
            "steps": epochs * max(1, math.ceil(len(idx) / batch_size)),
            "n_test": int(len(bundle.y_test)),
            "balanced_accuracy": float(ex["balanced_accuracy"]),
            "middle_parameters": model.middle_parameters(),
            "train_time_sec": float(info["training_time_sec"]),
            "train_best_loss": float(info["train_best_loss"])}


def sweep_lr(dataset: str, width: int, batch_size: int, threads: int,
             sizes: Sequence[int]) -> Dict[Tuple[str, int], float]:
    """Lernrate je (Arm, Datenmenge) -- die optimale Rate haengt auch an der
    Datenmenge, ein fester Wert waere derselbe Fehler wie in B-004."""
    best: Dict[Tuple[str, int], float] = {}
    for n in sizes:
        for arm in ARMS:
            means = {lr: float(np.mean([_cell(arm, n, s, dataset, lr, width,
                                              batch_size, threads)["balanced_accuracy"]
                                        for s in TUNING_SEEDS]))
                     for lr in LR_GRID}
            best[(arm, n)] = max(means, key=means.get)
        print(f"    {dataset:>14} n={n:<6} -> "
              + ", ".join(f"{a}:{best[(a, n)]}" for a in ARMS))
    return best


def run(width: int = WIDTH, seeds: Sequence[int] = EVAL_SEEDS,
        datasets: Sequence[str] = DATASETS, sizes: Sequence[int] = TRAIN_SIZES,
        batch_size: int = 256, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="B-010",
        description=("Strukturierte Schichten gegen dicht ueber sechs Datenmengen, "
                     "fester grosser Testsatz, angeglichene Zahl der "
                     "Gradientenschritte, Lernrate je Arm und Datenmenge."),
        config={"datasets": list(datasets), "train_sizes": list(sizes),
                "arms": list(ARMS), "width": width, "seeds": list(seeds),
                "pool_samples": POOL_SAMPLES, "target_steps": TARGET_STEPS,
                "batch_size": batch_size, "torch_threads": threads,
                "lr_grid": list(LR_GRID),
                "primary": ("Spearman(log Trainingsgroesse, Abstand zu dicht) je Arm "
                            "auf mnist, Holm-korrigiert; Vorhersage negativ"),
                "primary_dataset": PRIMARY_DATASET,
                "note": ("Testsatz fest und gross; nur der Trainingssatz variiert, "
                         "stratifiziert gezogen")},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, width, batch_size, threads, sizes)
        rec.config[f"lr_map_{dataset}"] = {f"{a}|{n}": v for (a, n), v in lr_map.items()}
        for i, seed in enumerate(seeds, 1):
            for n in sizes:
                for arm in ARMS:
                    rec.add(**_cell(arm, n, seed, dataset, lr_map[(arm, n)],
                                    width, batch_size, threads))
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    from .equivalence import holm

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    # Die stratifizierte Ziehung erreicht die angeforderte Groesse nicht immer
    # exakt: pro Klasse werden n/Klassenzahl Beispiele gezogen, und wenn eine
    # Klasse im jeweiligen Split weniger hat, faellt die Summe knapp darunter.
    # Bei 14.000 kam so pro Seed eine andere Zahl heraus (13676, 13687, ...),
    # wodurch jeder Seed in einer eigenen Gruppe landete und die Statistik NaN
    # wurde. Deshalb wird auf die angeforderte Groesse zurueckgerundet.
    requested = sorted(int(s) for s in config["train_sizes"])
    sizes = requested
    datasets = [str(d) for d in config["datasets"]]

    def bucket(n: int) -> int:
        return min(requested, key=lambda r: abs(r - int(n)))
    out: Dict[str, object] = {"per_dataset": {}}

    def vals(arm, n, ds):
        rows = sorted((r for r in records if r["arm"] == arm
                       and bucket(r["n_train"]) == n and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print(f"RESEARCH ID B-010 | Datenmenge gegen Struktur, "
          f"{len(vals(ARMS[0], sizes[0], datasets[0]))} Seeds")

    trends: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 60 - len(ds)))
        head = f'{"n_train":>8} {"Epochen":>8} ' + " ".join(f'{a:>15}' for a in ARMS)
        print(head); print("-" * len(head))
        for n in sizes:
            ep = next((r["epochs"] for r in records if bucket(r["n_train"]) == n
                       and r["dataset"] == ds), 0)
            line = f'{n:>8} {ep:>8} '
            for arm in ARMS:
                v = vals(arm, n, ds)
                line += f'{v.mean():>15.4f} '
            print(line)

        print()
        print("  Abstand zur dichten Schicht (positiv = Struktur vorn):")
        head2 = f'{"n_train":>8} ' + " ".join(f'{a:>17}' for a in STRUCTURED)
        print(head2); print("  " + "-" * (len(head2) - 2))
        deltas: Dict[str, List[float]] = {a: [] for a in STRUCTURED}
        for n in sizes:
            d = vals("dense", n, ds)
            line = f'{n:>8} '
            for arm in STRUCTURED:
                v = vals(arm, n, ds)
                p = float(stats.wilcoxon(v, d).pvalue)
                deltas[arm].append(float(v.mean() - d.mean()))
                star = "*" if p < 0.05 else " "
                line += f'{v.mean()-d.mean():>+15.4f}{star} '
            print(line)

        ds_out: Dict[str, float] = {}
        print()
        print("  PRIMAER -- Trend ueber die Datenmenge (Spearman, log n):")
        raw = []
        for arm in STRUCTURED:
            rho, p = stats.spearmanr(np.log(sizes), deltas[arm])
            raw.append(float(p))
            ds_out[f"{arm}_rho"] = float(rho)
        for arm, p_adj in zip(STRUCTURED, holm(raw)):
            rho = ds_out[f"{arm}_rho"]
            ds_out[f"{arm}_p_holm"] = p_adj
            mark = "*signifikant*" if p_adj < 0.05 else "(nicht signifikant)"
            reading = ("Vorteil schrumpft mit mehr Daten" if rho < 0
                       else "Vorteil waechst mit mehr Daten")
            print(f'    {arm:>17}: rho={rho:>+6.3f}  Holm p={p_adj:.4f}  {mark}  {reading}')
        trends[ds] = ds_out
        out["per_dataset"][ds] = ds_out       # type: ignore[index]

    # ---- Urteil ----
    print()
    print("=" * 78)
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    smallest = sizes[0]
    winners = []
    for arm in STRUCTURED:
        prim = trends[PRIMARY_DATASET]
        neg_trend = prim[f"{arm}_rho"] < 0 and prim[f"{arm}_p_holm"] < 0.05
        d = vals("dense", smallest, PRIMARY_DATASET)
        v = vals(arm, smallest, PRIMARY_DATASET)
        p_small = float(stats.wilcoxon(v, d).pvalue)
        wins_small = v.mean() > d.mean() and p_small < 0.05
        other = [t for t in datasets if t != PRIMARY_DATASET]
        same_dir = all(trends[o][f"{arm}_rho"] < 0 for o in other) if other else False
        if neg_trend and wins_small and same_dir:
            winners.append(arm)
        print(f'  {arm:>17}: Trend negativ&sig={neg_trend}, bei n={smallest} vorn&sig='
              f'{wins_small}, zweiter Datensatz gleiche Richtung={same_dir}')
    if winners:
        verdict = ("H1 -- Struktur zahlt sich bei knappen Daten aus: "
                   + ", ".join(winners) + ". Erster positiver Effizienzbefund "
                   "von Projekt B, auf einer bisher ungeprueften Achse.")
    else:
        verdict = ("H0 -- kein Arm erfuellt alle drei Bedingungen. Auch die "
                   "Datenachse gibt keinen Vorteil fuer Struktur her.")
    print()
    print(f"  {verdict}")
    print("=" * 78)
    out["verdict"] = verdict
    return out


if __name__ == "__main__":
    recorder = run()
    result = analyse(recorder)
    recorder.save(analysis=result)
