from __future__ import annotations

"""RESEARCH ID: B-011 -- Ist die Luecke zu dicht eine Kapazitaets- oder eine
Optimierungsluecke?

DIE BEOBACHTUNG, AUS DER DAS FOLGT
In B-008 ist die Genauigkeit ueber die Raenge flach: Rang 8 (69.632 Parameter)
ist so gut wie Rang 256 (2,1 Mio.). Dreissigfache Kapazitaet aendert nichts.
Trotzdem liegen alle Raenge etwa 0,005 unter der dichten Schicht.

Wenn zusaetzliche Kapazitaet nichts bringt, kann die Luecke keine Frage der
Ausdruckskraft sein. Dann bleiben zwei andere Erklaerungen, und die trennt
dieses Experiment:

  (a) DARSTELLBARKEIT -- die Familie enthaelt ueberhaupt keine Funktion, die so
      gut ist wie die dichte. Mehr Rang hilft nicht, weil Rang die falsche
      Achse ist.
  (b) OPTIMIERUNG -- die Familie enthaelt so eine Funktion, aber das Training
      von zufaelliger Initialisierung aus findet sie nicht.

Das ist eine andere Frage als alles, was Projekt B bisher gestellt hat. Die
Literatur zu strukturierten Matrizen rahmt den Vergleich fast immer als
Ausdruckskraft-Kompromiss; dass es ein Optimierungsproblem sein koennte, wird
selten geprueft.

DIE ARME (alle mit derselben strukturierten Schicht, nur anders erreicht)
  dense        volle Schicht -- die Referenz
  scratch      strukturierte Schicht, zufaellig initialisiert, auf Labels
               trainiert. Das ist genau das bisherige Vorgehen.
  svd_frozen   Rang-r-Trunkierung der FERTIG TRAINIERTEN dichten Matrix,
               eingesetzt, nicht nachtrainiert. Zeigt, wieviel der dichten
               Funktion die Trunkierung ueberhaupt ueberlebt.
  svd_finetune wie svd_frozen, danach mit demselben Schrittbudget wie `scratch`
               nachtrainiert.
  distill      strukturierte Schicht wird darauf trainiert, die AUSGABE der
               trainierten dichten Schicht nachzubilden (kleinste Quadrate auf
               den echten Aktivierungen), nicht die Labels. Das ist der direkte
               Test auf Darstellbarkeit: kann die Familie diese Funktion?

LESART, VORHER FESTGELEGT
  distill ~ dense  und  scratch < dense
        -> (b): die Familie KANN es, das Training findet es nur nicht.
           Ein Optimierungs- bzw. Zielfunktionsproblem. Praktisch verwertbar:
           strukturierte Schichten sollte man aus einem dichten Lauf
           initialisieren statt von null zu trainieren.
  distill < dense  ebenfalls deutlich
        -> (a): echte Darstellungsgrenze. Zusammen mit der flachen Rangkurve
           hiesse das: Rang ist schlicht die falsche Struktur fuer das, was die
           dichte Schicht tut.
  svd_finetune ~ dense, scratch < dense
        -> Initialisierung ist der entscheidende Hebel, nicht die Zielfunktion.

FAIRNESS -- ausdruecklich benannt
`svd_frozen`, `svd_finetune` und `distill` setzen eine bereits fertig
trainierte dichte Schicht voraus. Sie kosten also MEHR Gesamtrechenzeit als
`scratch`, nicht weniger. Dieses Experiment beantwortet deshalb ausdruecklich
NICHT die Frage "was ist effizienter", sondern nur "ist die Funktion in der
Familie erreichbar". Ein Effizienzvorteil darf daraus nicht abgeleitet werden.
Damit das Nachtrainieren fair bleibt, bekommt `svd_finetune` genau dasselbe
Schrittbudget wie `scratch`.

ZUSAETZLICH: DAS RICHTIGE SPEKTRUM
B-008 hat das Spektrum der Gewichtsmatrix W gemessen und keine Vorhersagekraft
gefunden. Moeglicherweise war es das falsche Spektrum. Fuer die Frage, wieviel
eine Trunkierung kostet, zaehlt nicht W allein, sondern W gewichtet mit der
Verteilung der tatsaechlichen Eingaben -- also das Spektrum der AUSGABEN der
Schicht auf echten Daten. Richtungen, die die Daten nie ansprechen, sind
gleichgueltig, egal wie gross ihr Singulaerwert ist.

Beides wird hier gemessen und gegen die Rangkurve gehalten. Faellt die
Vorhersage auch mit dem datengewichteten Spektrum durch, ist das ein
staerkerer Negativbefund als der aus B-008.

Run: python -m qire_ai.optimization_gap
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ablation_locality import middle_factory
from .ablation_structure import LowRankLinear
from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder

WIDTH = 4096
RANKS: Tuple[int, ...] = (8, 64)
DATASETS = ("mnist", "fashion_mnist")
EVAL_SEEDS = tuple(range(1901, 1916))       # 15 Seeds
MAX_SAMPLES = 6000
EPOCHS = 10
LR = 0.003                                   # aus B-004-lr fuer dense/lowrank
BATCH = 256
DISTILL_STEPS = 400
ARMS = ("dense", "scratch", "svd_frozen", "svd_finetune", "distill")


# --------------------------------------------------------------------------
# SVD-Initialisierung
# --------------------------------------------------------------------------

def svd_factors(svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                rank: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rang-r-Zerlegung W ~ Wu @ Wv mit ausgewogen verteilten Singulaerwerten.

    Die Wurzel des Singulaerwerts auf beide Faktoren zu legen ist die
    konditionsguenstigste Aufteilung; alles auf eine Seite zu legen erzeugt
    einen schlecht skalierten Faktor.

    Bekommt die fertige Zerlegung uebergeben: eine volle SVD einer
    4096x4096-Matrix kostet rund eine Minute und haengt nicht vom Rang ab, wird
    also einmal je Seed berechnet und fuer alle Raenge wiederverwendet.
    """
    U, S, Vh = svd
    r = min(rank, S.numel())
    root = torch.sqrt(S[:r])
    Wu = (U[:, :r] * root).to(dtype)                   # (dim, r)
    Wv = (root[:, None] * Vh[:r, :]).to(dtype)         # (r, dim)
    return Wu, Wv


def load_lowrank_from_dense(layer: LowRankLinear,
                            svd: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                            bias: torch.Tensor, rank: int) -> None:
    Wu, Wv = svd_factors(svd, rank, layer.u.weight.dtype)
    with torch.no_grad():
        layer.v.weight.copy_(Wv)
        layer.u.weight.copy_(Wu)
        layer.u.bias.copy_(bias)


def replace_middle(model: MiddleNet, new_middle: nn.Module) -> MiddleNet:
    model.middles = nn.ModuleList([new_middle])
    return model


# --------------------------------------------------------------------------
# Spektren
# --------------------------------------------------------------------------

def cumulative_energy(singular: torch.Tensor, ranks: Sequence[int]) -> Dict[int, float]:
    energy = singular ** 2
    total = float(energy.sum())
    cum = torch.cumsum(energy, dim=0)
    return {r: float(cum[min(r, len(energy)) - 1] / total) for r in ranks}


def weight_spectrum(singular: torch.Tensor, ranks: Sequence[int]) -> Dict[int, float]:
    return cumulative_energy(singular, ranks)


def data_spectrum(model: MiddleNet, x: torch.Tensor, ranks: Sequence[int],
                  max_rows: int = 4096) -> Dict[int, float]:
    """Spektrum der AUSGABEN der Mittelschicht auf echten Daten.

    Das ist die Groesse, die zaehlt: eine Richtung mit grossem Singulaerwert in
    W ist gleichgueltig, wenn die Daten sie nie ansprechen.
    """
    model.eval()
    with torch.no_grad():
        h = torch.relu(model.fc_in(x[:max_rows]))
        y = model.middles[0](h)
        y = y - y.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(y.double())
    return cumulative_energy(s, ranks)


# --------------------------------------------------------------------------
# Destillation
# --------------------------------------------------------------------------

def distill_middle(student: nn.Module, teacher: nn.Module, h: torch.Tensor,
                   steps: int, lr: float, seed: int) -> float:
    """Bringt `student` dazu, die Funktion von `teacher` auf echten
    Aktivierungen nachzubilden. Gibt den erreichten relativen Fehler zurueck."""
    torch.manual_seed(seed)
    with torch.no_grad():
        target = teacher(h)
        scale = float(target.pow(2).mean())
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n = h.shape[0]
    g = torch.Generator().manual_seed(seed)
    last = float("nan")
    for step in range(steps):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(student(h[idx]), target[idx])
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return last / scale if scale > 0 else float("nan")


# --------------------------------------------------------------------------
# Lauf
# --------------------------------------------------------------------------

def run_dataset(dataset: str, seeds: Sequence[int] = EVAL_SEEDS,
                ranks: Sequence[int] = RANKS, width: int = WIDTH,
                max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS,
                lr: float = LR, batch_size: int = BATCH,
                threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id=f"B-011-{dataset}",
        description=("Ist die Luecke zwischen strukturierter und dichter Schicht "
                     "eine Darstellungs- oder eine Optimierungsluecke? Vergleicht "
                     "Training von null, SVD-Initialisierung, Nachtrainieren und "
                     "Funktionsdestillation. Misst zusaetzlich das datengewichtete "
                     "Spektrum."),
        config={"dataset": dataset, "ranks": list(ranks), "width": width,
                "seeds": list(seeds), "max_samples": max_samples, "epochs": epochs,
                "lr": lr, "batch_size": batch_size, "torch_threads": threads,
                "distill_steps": DISTILL_STEPS, "arms": list(ARMS),
                "fairness_note": ("svd_* und distill setzen ein fertig trainiertes "
                                  "dichtes Modell voraus und kosten daher MEHR "
                                  "Gesamtrechenzeit als scratch. Dieses Experiment "
                                  "beantwortet Erreichbarkeit, nicht Effizienz.")},
    )
    for i, seed in enumerate(seeds, 1):
        set_seed(seed)
        torch.set_num_threads(threads)
        bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)

        # 1. dichte Referenz
        dense_model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                                lambda: nn.Linear(width, width), "dense",
                                "dense_ends", depth=1)
        train_model(dense_model, bundle.x_train, bundle.y_train, epochs=epochs,
                    lr=lr, seed=seed, batch_size=batch_size)
        _, _, _, ex = evaluate(dense_model, bundle.x_test, bundle.y_test)
        dense_acc = float(ex["balanced_accuracy"])
        W = dense_model.middles[0].weight.detach().clone()
        b = dense_model.middles[0].bias.detach().clone()
        teacher = dense_model.middles[0]
        # einmal je Seed -- teuer und rangunabhaengig
        svd = torch.linalg.svd(W.double(), full_matrices=False)
        w_spec = weight_spectrum(svd[1], ranks)
        d_spec = data_spectrum(dense_model, bundle.x_train, ranks)
        rec.add(arm="dense", rank=0, seed=seed, dataset=dataset,
                balanced_accuracy=dense_acc, middle_parameters=W.numel() + b.numel(),
                distill_rel_error=float("nan"),
                **{f"w_energy_{r}": w_spec[r] for r in ranks},
                **{f"d_energy_{r}": d_spec[r] for r in ranks})

        with torch.no_grad():
            h_train = torch.relu(dense_model.fc_in(bundle.x_train)).detach()

        for rank in ranks:
            label = f"lowrank_r{rank}"

            # 2. von null trainiert
            set_seed(seed)
            torch.set_num_threads(threads)
            scratch = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                                middle_factory(label, width), label, "dense_ends",
                                depth=1)
            train_model(scratch, bundle.x_train, bundle.y_train, epochs=epochs,
                        lr=lr, seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(scratch, bundle.x_test, bundle.y_test)
            rec.add(arm="scratch", rank=rank, seed=seed, dataset=dataset,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=scratch.middle_parameters(),
                    distill_rel_error=float("nan"))

            # 3. SVD-Trunkierung des trainierten dichten Modells, eingefroren
            import copy
            frozen = copy.deepcopy(dense_model)
            low = LowRankLinear(width, rank)
            load_lowrank_from_dense(low, svd, b, rank)
            replace_middle(frozen, low)
            _, _, _, ex = evaluate(frozen, bundle.x_test, bundle.y_test)
            rec.add(arm="svd_frozen", rank=rank, seed=seed, dataset=dataset,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=sum(p.numel() for p in low.parameters()),
                    distill_rel_error=float("nan"))

            # 4. dieselbe Initialisierung, mit gleichem Budget nachtrainiert
            tuned = copy.deepcopy(frozen)
            train_model(tuned, bundle.x_train, bundle.y_train, epochs=epochs,
                        lr=lr, seed=seed, batch_size=batch_size)
            _, _, _, ex = evaluate(tuned, bundle.x_test, bundle.y_test)
            rec.add(arm="svd_finetune", rank=rank, seed=seed, dataset=dataset,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=sum(p.numel() for p in tuned.middles[0].parameters()),
                    distill_rel_error=float("nan"))

            # 5. Destillation: die Funktion der dichten Schicht nachbilden
            student_net = copy.deepcopy(dense_model)
            student = LowRankLinear(width, rank)
            err = distill_middle(student, teacher, h_train, DISTILL_STEPS, lr, seed)
            replace_middle(student_net, student)
            _, _, _, ex = evaluate(student_net, bundle.x_test, bundle.y_test)
            rec.add(arm="distill", rank=rank, seed=seed, dataset=dataset,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    middle_parameters=sum(p.numel() for p in student.parameters()),
                    distill_rel_error=float(err))

        print(f"  {dataset} seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    from .equivalence import MARGIN, tost_paired

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    ranks = [int(r) for r in config["ranks"]]
    dataset = config["dataset"]
    out: Dict[str, object] = {"dataset": dataset, "per_rank": {}}

    def vals(arm, rank):
        rows = sorted((r for r in records if r["arm"] == arm and r["rank"] == rank),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    dense = vals("dense", 0)
    n = len(dense)
    print()
    print(f"RESEARCH ID B-011 | {dataset}, {n} Seeds, dichte Referenz {dense.mean():.4f}")

    print()
    print("Spektren der trainierten dichten Schicht (kumulierte Energie):")
    head = f'  {"Rang":>6} {"Gewichte W":>12} {"Daten (Ausgaben)":>18}'
    print(head); print("  " + "-" * (len(head) - 2))
    for r in ranks:
        w = float(np.mean([rec[f"w_energy_{r}"] for rec in records if rec["arm"] == "dense"]))
        d = float(np.mean([rec[f"d_energy_{r}"] for rec in records if rec["arm"] == "dense"]))
        print(f'  {r:>6} {w:>12.4f} {d:>18.4f}')
        out[f"w_energy_{r}"] = w
        out[f"d_energy_{r}"] = d

    for rank in ranks:
        print()
        print(f"--- Rang {rank} " + "-" * 52)
        head = (f'{"Arm":>14} {"accuracy":>9} {"vs dicht":>10} {"p":>8} '
                f'{"TOST p":>8} {"Lesart":>28}')
        print(head); print("-" * len(head))
        rank_out: Dict[str, object] = {}
        for arm in ARMS[1:]:
            v = vals(arm, rank)
            if len(v) != n:
                continue
            p = float(stats.wilcoxon(v, dense).pvalue)
            eq = tost_paired(v, dense, MARGIN)
            reading = ("gleichwertig" if eq["p_tost"] < 0.05 else
                       "Unterschied belegt" if p < 0.05 else
                       f"unentschieden (+-{eq['achieved']:.4f})")
            print(f'{arm:>14} {v.mean():>9.4f} {v.mean()-dense.mean():>+10.4f} '
                  f'{p:>8.4f} {eq["p_tost"]:>8.4f} {reading:>28}')
            rank_out[arm] = {"delta": float(v.mean() - dense.mean()), "p": p,
                             "p_tost": eq["p_tost"]}
        err = [r["distill_rel_error"] for r in records
               if r["arm"] == "distill" and r["rank"] == rank]
        if err:
            print(f'  relativer Nachbildungsfehler der Destillation: {np.mean(err):.4f}')
            rank_out["distill_rel_error"] = float(np.mean(err))
        out["per_rank"][str(rank)] = rank_out       # type: ignore[index]

        # Lesart nach den vorher festgelegten Kriterien
        d_ok = rank_out.get("distill", {}).get("p_tost", 1) < 0.05
        s_bad = rank_out.get("scratch", {}).get("p", 1) < 0.05 and \
            rank_out.get("scratch", {}).get("delta", 0) < 0
        f_ok = rank_out.get("svd_finetune", {}).get("p_tost", 1) < 0.05
        if d_ok and s_bad:
            verdict = ("OPTIMIERUNGSLUECKE -- die Familie kann die Funktion "
                       "darstellen, das Training von null findet sie nicht.")
        elif not d_ok and s_bad:
            verdict = ("DARSTELLUNGSGRENZE -- auch die Destillation erreicht die "
                       "dichte Schicht nicht. Rang ist die falsche Struktur.")
        else:
            verdict = "kein klares Bild bei diesem Rang."
        if f_ok and s_bad:
            verdict += " Zusaetzlich: SVD-Initialisierung schliesst die Luecke."
        print(f'  -> {verdict}')
        rank_out["verdict"] = verdict               # type: ignore[assignment]

    return out


if __name__ == "__main__":
    for ds in DATASETS:
        print("\n" + "#" * 74)
        print(f"# B-011 -- {ds}")
        print("#" * 74)
        recorder = run_dataset(ds)
        result = analyse(recorder)
        recorder.save(analysis=result)      # nach JEDEM Datensatz speichern
