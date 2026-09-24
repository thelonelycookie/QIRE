from __future__ import annotations

"""RESEARCH ID: B-004 -- Blockdiagonale gegen Monarch bei exakt gleichem Budget,
mit und ohne dichte Mischschichten, mit und ohne Bildstruktur.

WAS DIESES EXPERIMENT KLAEREN SOLL
Meine Aussage aus A-002/A-003 lautet: "Lokalitaet traegt, Bonds nicht." Drei
Gruende, warum sie so nicht haltbar ist:

1. FALSCHER GEGNER. Verglichen wurde gegen `dense`, Low-Rank und Tensor-Train.
   Blockdiagonale Schichten sind aber seit Nesky & Stout (ICANN 2018) bekannt;
   gegen `dense` zu gewinnen ist kein Ergebnis. Der Stand sind Monarch (Dao et
   al., ICML 2022 -- zwei blockdiagonale Matrizen mit Permutation dazwischen)
   und Block-Tensor-Train (Qiu et al. 2024). Monarch existiert genau deshalb,
   WEIL eine reine Blockdiagonale nicht ueber Blockgrenzen mischt.

2. FALSCHE ARCHITEKTUR. In allen bisherigen Laeufen standen dichte Schichten
   vor und hinter der Mittelschicht. In `validate_structured.py` (Pruefung 6)
   ist gemessen: im alten Aufbau erreicht ein Eingang auch mit blockdiagonaler
   Mitte fremde Gruppen (groesster Jacobi-Eintrag 6,9e-02). A-003 konnte seine
   eigene Frage also gar nicht beantworten.

3. MOEGLICHE VERWECHSLUNG VON LOKALITAET MIT BILDSTRUKTUR. Wenn die Gruppen
   zusammenhaengende Pixelbloecke sind, ist "Gruppenstruktur hilft" womoeglich
   nur "benachbarte Pixel gehoeren zusammen" -- eine Aussage ueber MNIST, nicht
   ueber die Schicht. Deshalb hier eine dritte Architekturvariante mit einer
   FESTEN ZUFAELLIGEN PIXELPERMUTATION vor dem Netz. Die Permutation ist fuer
   alle Arme und Seeds dieselbe, also ein kontrollierter Faktor und kein
   Rauschen. Bricht der Vorsprung der Blockdiagonale dort ein, war es
   Bildstruktur.

FAIRE LERNRATE
Alle bisherigen Experimente dieses Projekts trainieren jeden Arm mit lr=0,01.
Monarch, Tensor-Train, Low-Rank und die Blockdiagonale haben aber voellig
verschiedene Konditionierung -- ein fester Wert misst womoeglich, welche
Struktur zufaellig mit 0,01 zurechtkommt. Deshalb laeuft vorab eine
Lernratensuche je (Arm, Architektur) auf EIGENEN Tuning-Seeds (1301-1303).
Der Hauptlauf benutzt dann fuer jeden Arm dessen beste Lernrate und frische
Auswertungs-Seeds (1201-1210). Getrennte Seeds, damit die Auswahl nicht auf
denselben Daten stattfindet, ueber die spaeter berichtet wird.

Die permutierte Variante uebernimmt die Lernraten von `local_ends`: identische
Architektur, nur die Eingangsreihenfolge unterscheidet sich, die
Konditionierung also nicht. Das ist eine Annahme und steht so in der Konfig.

BUDGET
Fuenf Arme haben exakt dieselbe Parameterzahl (524.288 + 4.096 Bias):

    Monarch(64x64)               4096 * (64+64)      = 524.288
    Blockdiagonale(128)          32 * 128^2          = 524.288
    Blockdiagonale(128) spektral 32 * 128^2          = 524.288
    Tensor-Train(2 Kerne, 64)    2 * 64*64*64        = 524.288
    Low-Rank(Rang 64)            2 * 4096*64         = 524.288

Bezugspunkte: `dense` (16.781.312) und die urspruengliche Blockdiagonale mit
Blockgroesse 4 (20.480) aus A-002 -- das ist das Budget, auf das sich die
819x-Aussage im Tagebuch bezieht.

PRIMAERE VERGLEICHE (vorher festgelegt, alles andere ist explorativ)
    P1: monarch_64x64 gegen block_diag_128 in `local_ends`
    P2: derselbe Vergleich in `local_ends_perm`
Alle weiteren Tests dienen der Einordnung und werden nicht als Beleg gefuehrt.

HYPOTHESEN
  H1 : Der Vorsprung der Blockdiagonale haengt an den dichten Enden oder an
       der Bildstruktur. In `local_ends` oder `local_ends_perm` schlaegt
       Monarch sie signifikant.
       -> A-002/A-003 waren ein Artefakt, die Aussage muss zurueckgezogen
          werden.
  H0 : Die Blockdiagonale haelt in beiden lokalen Varianten gegen Monarch.
       -> der Befund ueberlebt gegen den Stand der Technik.
  H2 : In den lokalen Varianten trennt sich kein Arm.
       -> Aufbau zu schwach, keine Aussage moeglich.

FALSIFIKATION
H0 verlangt: kein signifikanter Rueckstand in P1 UND P2. Zusaetzlich muss der
Aufbau nachweislich empfindlich sein. Positivkontrolle ist der `dense`-Arm
(32x mehr Parameter, mischt global); trennt er sich in der jeweiligen
Architektur nicht von der Blockdiagonale, gilt H2 und das Ergebnis wird
verworfen statt als Nullbefund berichtet.

Run: python -m qire_ai.ablation_locality
"""

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .ablation_orthogonality import ConstrainedBlock
from .ablation_structure import BlockDiagonal, LowRankLinear
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder
from .structured_layers import MonarchLinear
from .tensor_train import TTLinear

WIDTH = 4096
GROUPS = 16
PERM_SEED = 12345          # feste Pixelpermutation, fuer alle Arme/Seeds dieselbe
LR_GRID = (0.003, 0.01, 0.03)
TUNING_SEEDS = (1301, 1302, 1303)
EVAL_SEEDS = tuple(range(1201, 1211))

EQUAL_BUDGET_ARMS = ["block_diag_128", "block_diag_128_spectral",
                     "monarch_64x64", "tt_c2_r64", "lowrank_r64"]
ARMS = ["dense", "block_diag_128", "block_diag_128_spectral", "monarch_64x64",
        "tt_c2_r64", "lowrank_r64", "block_diag_4"]
ARCHS = ["dense_ends", "local_ends", "local_ends_perm"]
PRIMARY = [("monarch_64x64", "local_ends"), ("monarch_64x64", "local_ends_perm")]


def middle_factory(label: str, dim: int) -> Callable[[], nn.Module]:
    # Allgemeine Formen zuerst, damit beliebige Raenge und Blockgroessen ohne
    # neue Zweige funktionieren: "lowrank_r16", "block_diag_64", ...
    if label.startswith("lowrank_r"):
        rank = int(label.split("_r")[1])
        return lambda: LowRankLinear(dim, rank=rank)
    if label.startswith("block_diag_") and label.split("_")[-1].isdigit():
        size = int(label.split("_")[-1])
        return lambda: BlockDiagonal(dim, block_size=size)
    if label == "monarch":
        # Breitenunabhaengig: die ausgewogenste Zerlegung dim = m1*m2. Noetig,
        # sobald die Breite variiert wird -- "monarch_64x64" gilt nur fuer 4096.
        return lambda: MonarchLinear(dim)
    if label.startswith("monarch_") and "x" in label:
        m1, m2 = (int(v) for v in label.split("_")[1].split("x"))
        return lambda: MonarchLinear(dim, m1, m2)
    if label == "dense":
        return lambda: nn.Linear(dim, dim)
    if label == "block_diag_128_spectral":
        # B-003 prueft die Spektralnormalisierung nur bei Blockgroesse 4. Hier
        # dieselbe Beschraenkung auf dem Monarch-Budget, sonst laesst sich die
        # Aussage "spektral-normalisierte Blockschicht gegen Monarch" nicht
        # belegen.
        return lambda: ConstrainedBlock(dim, 128, "spectral")
    if label == "tt_c2_r64":
        return lambda: TTLinear(dim, dim, num_cores=2, tt_rank=64)
    raise ValueError(label)


def base_arch(arch: str) -> str:
    """Architektur ohne den Permutations-Zusatz."""
    return "local_ends" if arch == "local_ends_perm" else arch


def _permutation(dim: int) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(PERM_SEED).permutation(dim))


def _forward_ms(model: nn.Module, x: torch.Tensor, repeats: int = 25) -> float:
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for _ in range(3):
            model(x)
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def _one_cell(arm: str, arch: str, seed: int, width: int, groups: int,
              max_samples: int, epochs: int, lr: float, batch_size: int,
              threads: int, measure_time: bool = True,
              dataset: str = "mnist") -> Dict[str, object]:
    """Ein Training plus Auswertung. Gibt eine fertige Messzeile zurueck."""
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    x_train, x_test = bundle.x_train, bundle.x_test
    if arch == "local_ends_perm":
        perm = _permutation(bundle.input_dim)
        x_train, x_test = x_train[:, perm], x_test[:, perm]
    model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                      middle_factory(arm, width), arm, base_arch(arch),
                      groups=groups, depth=1)
    info = train_model(model, x_train, bundle.y_train, epochs=epochs, lr=lr,
                       seed=seed, batch_size=batch_size)
    _, _, _, ex = evaluate(model, x_test, bundle.y_test)
    return {
        "arm": arm, "arch": arch, "seed": seed, "lr": lr, "dataset": dataset,
        "balanced_accuracy": float(ex["balanced_accuracy"]),
        "middle_parameters": model.middle_parameters(),
        "end_parameters": model.end_parameters(),
        "train_time_sec": float(info["training_time_sec"]),
        # Traegt die Information, ob das Modell ueberhaupt auskonvergiert ist --
        # ohne sie laesst sich der Konvergenzlauf (B-006) nicht einordnen.
        "train_best_loss": float(info["train_best_loss"]),
        "forward_ms_median": _forward_ms(model, x_test[:256]) if measure_time else float("nan"),
        "nan_detected": str(info.get("nan_detected", "False")),
    }


# --------------------------------------------------------------------------
# Stufe 1 -- Lernratensuche auf eigenen Seeds
# --------------------------------------------------------------------------

def sweep_learning_rate(
    width: int = WIDTH,
    seeds: Sequence[int] = TUNING_SEEDS,
    lr_grid: Sequence[float] = LR_GRID,
    arms: Sequence[str] = tuple(ARMS),
    archs: Sequence[str] = ("dense_ends", "local_ends"),
    max_samples: int = 6000,
    epochs: int = 10,
    batch_size: int = 256,
    threads: int = 6,
    groups: int = GROUPS,
) -> Tuple[Dict[Tuple[str, str], float], Recorder]:
    """Beste Lernrate je (Arm, Architektur), gemessen auf Tuning-Seeds.

    `local_ends_perm` wird nicht eigens vermessen und uebernimmt die Werte von
    `local_ends` -- identische Architektur, nur andere Eingangsreihenfolge.
    """
    rec = Recorder(
        experiment_id="B-004-lr",
        description=("Lernratensuche je Arm und Architektur auf eigenen Seeds. "
                     "Verhindert, dass der Hauptvergleich misst, welche Struktur "
                     "zufaellig mit einer festen Lernrate zurechtkommt."),
        config={"lr_grid": list(lr_grid), "tuning_seeds": list(seeds),
                "arms": list(arms), "archs": list(archs), "width": width,
                "max_samples": max_samples, "epochs": epochs,
                "batch_size": batch_size, "torch_threads": threads,
                "note": "local_ends_perm uebernimmt die Lernraten von local_ends"},
    )
    total = len(seeds) * len(archs) * len(arms) * len(lr_grid)
    done = 0
    for arch in archs:
        for arm in arms:
            for lr in lr_grid:
                for seed in seeds:
                    rec.add(**_one_cell(arm, arch, seed, width, groups, max_samples,
                                        epochs, lr, batch_size, threads,
                                        measure_time=False))
                    done += 1
            print(f"  {arch:>16} / {arm:<24} fertig ({done}/{total})")

    best: Dict[Tuple[str, str], float] = {}
    for arch in archs:
        for arm in arms:
            means = {}
            for lr in lr_grid:
                vals = [r["balanced_accuracy"] for r in rec.records
                        if r["arm"] == arm and r["arch"] == arch and r["lr"] == lr]
                means[lr] = float(np.mean(vals)) if vals else -1.0
            best[(arm, arch)] = max(means, key=means.get)
            if arch == "local_ends":
                best[(arm, "local_ends_perm")] = best[(arm, arch)]
    return best, rec


def report_sweep(best: Dict[Tuple[str, str], float], rec: Recorder) -> None:
    print()
    print("Gewaehlte Lernraten (bestes Mittel ueber die Tuning-Seeds):")
    header = f'{"arm":>26} {"dense_ends":>12} {"local_ends":>12}'
    print(header); print("-" * len(header))
    for arm in ARMS:
        d = best.get((arm, "dense_ends"), float("nan"))
        l = best.get((arm, "local_ends"), float("nan"))
        flag = "" if d == l else "   <- unterschiedlich"
        print(f'{arm:>26} {d:>12} {l:>12}{flag}')
    spread = len({v for v in best.values()})
    if spread > 1:
        print("\n  Die Arme bevorzugen verschiedene Lernraten -- ein fester Wert "
              "haette den Vergleich verzerrt.")
    else:
        print("\n  Alle Arme bevorzugen dieselbe Lernrate -- der feste Wert der "
              "frueheren Experimente war hier unkritisch.")


# --------------------------------------------------------------------------
# Stufe 2 -- Hauptlauf
# --------------------------------------------------------------------------

def run(
    lr_map: Optional[Dict[Tuple[str, str], float]] = None,
    width: int = WIDTH,
    seeds: Sequence[int] = EVAL_SEEDS,
    max_samples: int = 6000,
    epochs: int = 10,
    batch_size: int = 256,
    threads: int = 6,
    groups: int = GROUPS,
    arms: Sequence[str] = tuple(ARMS),
    archs: Sequence[str] = tuple(ARCHS),
    experiment_id: str = "B-004",
    description: Optional[str] = None,
    default_lr: float = 0.01,
    dataset: str = "mnist",
) -> Recorder:
    rec = Recorder(
        experiment_id=experiment_id,
        description=description or (
            "Blockdiagonale gegen Monarch, Tensor-Train und Low-Rank bei exakt "
            "gleichem Budget (524.288); drei Architekturen: dichte Enden, "
            "gruppenweise Enden, gruppenweise Enden mit fester Pixelpermutation. "
            "Lernrate je Arm aus B-004-lr."
        ),
        config={
            "dataset": dataset, "width": width, "groups": groups,
            "seeds": list(seeds), "arms": list(arms), "archs": list(archs),
            "max_samples": max_samples, "epochs": epochs, "batch_size": batch_size,
            "torch_threads": threads, "depth": 1,
            "equal_budget_arms": EQUAL_BUDGET_ARMS,
            "equal_budget_middle_params": 524288 + width,
            "permutation_seed": PERM_SEED,
            "primary_comparisons": [f"{a} vs block_diag_128 in {b}" for a, b in PRIMARY],
            "lr_map": {f"{a}|{b}": v for (a, b), v in (lr_map or {}).items()},
            "seed_note": "Auswertungs-Seeds ab 1201; Lernraten auf 1301-1303 gewaehlt",
            "architecture_check": ("local_ends hat nachweislich keine Querverbindung "
                                   "zwischen Gruppen ausser der Mittelschicht -- "
                                   "validate_structured.py Pruefung 6"),
        },
    )
    total = len(seeds) * len(archs) * len(arms)
    done = 0
    for seed in seeds:
        for arch in archs:
            for arm in arms:
                lr = (lr_map or {}).get((arm, arch), default_lr)
                rec.add(**_one_cell(arm, arch, seed, width, groups, max_samples,
                                    epochs, lr, batch_size, threads,
                                    dataset=dataset))
                done += 1
        print(f"  seed {seed} fertig ({done}/{total} Laeufe)")
    return rec


# --------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------

def _vals(records, arm: str, arch: str) -> np.ndarray:
    rows = sorted((r for r in records if r["arm"] == arm and r["arch"] == arch),
                  key=lambda r: r["seed"])
    return np.array([r["balanced_accuracy"] for r in rows])


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, exp_id, config = rec_or_id.records, rec_or_id.experiment_id, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, exp_id, config = (payload["records"], payload["experiment_id"],
                                   payload["config"])

    arms = [a for a in ARMS if any(r["arm"] == a for r in records)]
    archs = [a for a in ARCHS if any(r["arch"] == a for r in records)]
    n = len(_vals(records, arms[0], archs[0]))
    out: Dict[str, object] = {"n_seeds": n, "per_arch": {}}

    print()
    print(f"RESEARCH ID {exp_id} | MNIST, Breite {config.get('width', WIDTH)}, "
          f"{n} Auswertungs-Seeds, {config.get('epochs')} Epochen, "
          f"{config.get('max_samples')} Beispiele")

    def params(arm, arch):
        for r in records:
            if r["arm"] == arm and r["arch"] == arch:
                return int(r["middle_parameters"])
        return 0

    for arch in archs:
        print()
        print(f"--- {arch} " + "-" * max(4, 62 - len(arch)))
        header = (f'{"Mittelschicht":>26} {"accuracy":>9} {"std":>7} '
                  f'{"params":>12} {"lr":>7} {"fwd ms":>8}')
        print(header); print("-" * len(header))
        for arm in arms:
            v = _vals(records, arm, arch)
            rows = [r for r in records if r["arm"] == arm and r["arch"] == arch]
            ms = np.median([r["forward_ms_median"] for r in rows])
            lr = rows[0]["lr"] if rows else float("nan")
            print(f'{arm:>26} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} '
                  f'{params(arm, arch):>12,} {lr:>7} {ms:>8.1f}')

        base = _vals(records, "block_diag_128", arch)
        arch_out: Dict[str, object] = {}
        print()
        print("  Gleiches Budget -- jeder Gegner gegen die Blockdiagonale(128):")
        for arm in EQUAL_BUDGET_ARMS + ["dense"]:
            if arm == "block_diag_128" or arm not in arms:
                continue
            v = _vals(records, arm, arch)
            p = float(stats.wilcoxon(v, base).pvalue)
            mark = "*signifikant*" if p < 0.05 else "(nicht signifikant)"
            lead = "Gegner vorn" if v.mean() > base.mean() else "Blockdiagonale vorn"
            star = "  [PRIMAER]" if (arm, arch) in PRIMARY else ""
            note = "  [Positivkontrolle]" if arm == "dense" else ""
            print(f'    {arm:>26}: {v.mean()-base.mean():>+8.4f}  p={p:.4f}  '
                  f'{mark:<20} {lead}{star}{note}')
            arch_out[arm] = {"delta_vs_block": float(v.mean()-base.mean()), "p": p}
        out["per_arch"][arch] = arch_out       # type: ignore[index]

    # ---- Interaktion: haengt das Bild an den Enden bzw. an der Bildstruktur? ----
    print()
    print("=" * 76)
    print("INTERAKTION -- veraendert sich der Abstand zur Blockdiagonale,")
    print("wenn die dichten Enden wegfallen bzw. die Pixel permutiert werden?")
    inter: Dict[str, object] = {}
    if "dense_ends" in archs:
        for other in [a for a in archs if a != "dense_ends"]:
            print(f"\n  dense_ends  ->  {other}")
            for arm in EQUAL_BUDGET_ARMS + ["dense"]:
                if arm == "block_diag_128" or arm not in arms:
                    continue
                d0 = _vals(records, arm, "dense_ends") - _vals(records, "block_diag_128", "dense_ends")
                d1 = _vals(records, arm, other) - _vals(records, "block_diag_128", other)
                p = float(stats.wilcoxon(d1, d0).pvalue)
                mark = "*signifikant*" if p < 0.05 else "(nicht signifikant)"
                print(f'    {arm:>26}: {d0.mean():>+8.4f} -> {d1.mean():>+8.4f}  '
                      f'(Aenderung {d1.mean()-d0.mean():>+8.4f}, p={p:.4f}) {mark}')
                inter[f"{arm}|{other}"] = {"delta_dense_ends": float(d0.mean()),
                                           "delta_other": float(d1.mean()),
                                           "change": float(d1.mean()-d0.mean()), "p": p}
    if "local_ends" in archs and "local_ends_perm" in archs:
        print("\n  Bildstruktur -- local_ends gegen local_ends_perm "
              "(dieselbe Architektur, nur Pixel gemischt):")
        for arm in ["block_diag_128", "monarch_64x64", "dense"]:
            if arm not in arms:
                continue
            a0 = _vals(records, arm, "local_ends")
            a1 = _vals(records, arm, "local_ends_perm")
            p = float(stats.wilcoxon(a1, a0).pvalue)
            mark = "*signifikant*" if p < 0.05 else "(nicht signifikant)"
            print(f'    {arm:>26}: {a0.mean():.4f} -> {a1.mean():.4f}  '
                  f'({a1.mean()-a0.mean():>+8.4f}, p={p:.4f}) {mark}')
            inter[f"perm_effect|{arm}"] = {"local": float(a0.mean()),
                                           "permuted": float(a1.mean()),
                                           "change": float(a1.mean()-a0.mean()), "p": p}
    out["interaction"] = inter

    # ---- Bewertung nach den vorher festgelegten Kriterien ----
    print()
    print("BEWERTUNG (Kriterien vor dem Lauf festgelegt)")
    verdicts = []
    for arm, arch in PRIMARY:
        if arch not in archs:
            continue
        cell = out["per_arch"][arch]                       # type: ignore[index]
        mono = cell.get(arm, {})
        dense_ctrl = cell.get("dense", {})
        sensitive = dense_ctrl.get("delta_vs_block", 0) > 0 and dense_ctrl.get("p", 1) < 0.05
        monarch_wins = mono.get("delta_vs_block", 0) > 0 and mono.get("p", 1) < 0.05
        if not sensitive:
            v = (f"{arch}: H2 -- selbst die dichte Mittelschicht trennt sich nicht "
                 f"von der Blockdiagonale. Aufbau nicht empfindlich genug, aus "
                 f"dieser Architektur folgt nichts.")
        elif monarch_wins:
            v = (f"{arch}: H1 -- Monarch schlaegt die Blockdiagonale. Der Befund "
                 f"aus A-002/A-003 haelt hier nicht.")
        else:
            v = (f"{arch}: H0 -- die Blockdiagonale haelt gegen Monarch, und der "
                 f"Aufbau ist nachweislich empfindlich.")
        verdicts.append(v)
        print(f"  {v}")
    print("=" * 76)
    out["verdict"] = verdicts
    return out


if __name__ == "__main__":
    print("Stufe 1 -- Lernratensuche")
    lr_best, lr_rec = sweep_learning_rate()
    report_sweep(lr_best, lr_rec)
    lr_rec.save(analysis={f"{a}|{b}": v for (a, b), v in lr_best.items()})

    print("\nStufe 2 -- Hauptlauf")
    recorder = run(lr_map=lr_best)
    result = analyse(recorder)
    recorder.save(analysis=result)
