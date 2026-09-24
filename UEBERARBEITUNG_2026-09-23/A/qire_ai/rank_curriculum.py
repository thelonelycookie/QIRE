from __future__ import annotations

"""RESEARCH ID: B-013 -- Rang-Curriculum: hoch anfangen, absenken.

+-------------------------------------------------------------------------+
| ERGEBNIS (nachgetragen):                                                |
|                                                                         |
| H0 auf BEIDEN Datensaetzen -- das Curriculum schlaegt festen Rang nicht. |
| Gegen einen einzelnen Schnitt ist es auf Fashion-MNIST sogar            |
| signifikant SCHLECHTER (-0,0106, Holm p=0,0012). Vermutlich, weil jede  |
| Projektion den Adam-Zustand zuruecksetzt: vier Stoerungen statt einer.  |
|                                                                         |
| Ausserdem beruht die Begruendung unten auf einer Annahme, die B-012     |
| widerlegt hat: der Rang kollabiert NICHT waehrend des Trainings,        |
| sondern ist schon nach einer Epoche niedrig. Damit war fuer ein         |
| allmaehliches Absenken von vornherein wenig Raum -- was die vorab ins   |
| Skript geschriebene Vorhersage auch so gesagt hat.                       |
|                                                                         |
| Verwertbar bleibt: ein einzelner Schnitt bei Epoche 5 schlaegt festen   |
| Rang (MNIST +0,0097, Holm p=0,0215). Gegen die dichte Schicht liegt er  |
| numerisch vorn (+0,0032 / +0,0049), aber bei 15 Seeds ist weder         |
| Ueberlegenheit noch Gleichwertigkeit nachgewiesen (traegt nur +-0,007). |
+-------------------------------------------------------------------------+

DIE IDEE, UND WARUM SIE ETWAS ANDERES IST ALS ALLES BISHERIGE
Bisher hat Projekt B nur gemessen und verglichen. Hier wird zum ersten Mal ein
VERFAHREN vorgeschlagen und gegen den Stand geprueft.

Die Ueberlegung folgt direkt aus B-011 und B-012: Das Ziel ist niedrigrangig,
aber der Weg dorthin fuehrt durch hohen Rang. Wenn das stimmt, ist die uebliche
Praxis -- von Anfang an mit der Zielstruktur zu trainieren -- genau der Fehler.
Besser waere: mit voller Freiheit anfangen und die Beschraenkung waehrend des
Trainings schrittweise anziehen.

Das ist eine Trainingsmethode, kein Befund. Sie kann funktionieren oder nicht,
und beides ist berichtenswert.

DIE ARME (alle mit identischem Gesamtbudget von 10 Epochen und identischem
Endrang, damit der Vergleich nicht ueber Kosten oder Modellgroesse laeuft)

  dense        keine Beschraenkung -- obere Bezugsgroesse
  fixed        Endrang von Epoche 0 an. Das ist die uebliche Praxis und der
               eigentliche Gegner.
  single_cut   bis Epoche c dicht, dann EINMAL auf den Endrang. Das ist die
               Kontrolle, die klaert, ob das Schrittweise ueberhaupt noetig ist
               oder ob ein einziger spaeter Schnitt genauso gut waere.
  curriculum   mehrere Stufen: 512 -> 256 -> 128 -> Endrang, verteilt ueber das
               Training.

Der Vergleich `curriculum` gegen `single_cut` ist der interessantere: `fixed`
zu schlagen zeigt nur, dass spaeteres Beschraenken hilft (das wissen wir aus
B-012). Erst `single_cut` zu schlagen zeigt, dass das ALLMAEHLICHE etwas
beitraegt.

VORREGISTRIERUNG
  PRIMAER  curriculum gegen fixed, gepaart, Holm ueber die Datensaetze.
  SEKUNDAER curriculum gegen single_cut.

  H1 : curriculum schlaegt fixed signifikant auf beiden Datensaetzen
       -> Rang-Curriculum ist ein brauchbares Verfahren.
  H2 : zusaetzlich curriculum > single_cut
       -> das Allmaehliche traegt selbst bei, nicht nur der spaete Zeitpunkt.
  H0 : kein signifikanter Unterschied zu fixed
       -> die Idee traegt nicht; der Befund aus B-012 laesst sich nicht in ein
          Verfahren uebersetzen.

EHRLICH DAZU: Waehrend der frueheren Epochen rechnet das Curriculum mit mehr
Parametern als `fixed` und ist deshalb pro Epoche teurer. Gleich ist die Zahl
der Epochen und der Endrang, nicht die Rechenzeit. Ein Effizienzvorteil gegen
`fixed` darf daraus nicht abgeleitet werden -- der Anspruch ist ausschliesslich
bessere Genauigkeit bei gleichem Endmodell.

Run: python -m qire_ai.rank_curriculum
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .ablation_structure import LowRankLinear
from .benchmark import set_seed
from .data_loader import load_dataset
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder
from .spectral_probe import train_with_probe
from .trajectory_dose import project_to_lowrank

WIDTH = 4096
TOTAL_EPOCHS = 10
FINAL_RANK = 64
SINGLE_CUT_EPOCH = 5
CURRICULUM: Tuple[Tuple[int, int], ...] = ((2, 512), (4, 256), (6, 128), (8, FINAL_RANK))
DATASETS = ("mnist", "fashion_mnist")
EVAL_SEEDS = tuple(range(2101, 2116))      # 15 Seeds
LR = 0.003
BATCH = 256
ARMS = ("dense", "fixed", "single_cut", "curriculum")


def _schedule_for(arm: str, final_rank: int) -> List[Tuple[int, int]]:
    """(Epoche, Rang) -- wann auf welchen Rang projiziert wird."""
    if arm == "dense":
        return []
    if arm == "fixed":
        return [(0, final_rank)]
    if arm == "single_cut":
        return [(SINGLE_CUT_EPOCH, final_rank)]
    if arm == "curriculum":
        return [(e, r) for e, r in CURRICULUM[:-1]] + [(CURRICULUM[-1][0], final_rank)]
    raise ValueError(arm)


def _current_weight(model: MiddleNet, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Effektive Gewichtsmatrix der Mittelschicht, egal in welcher Form sie
    gerade vorliegt (dicht oder faktorisiert)."""
    layer = model.middles[0]
    if isinstance(layer, nn.Linear):
        return layer.weight.detach().clone(), layer.bias.detach().clone()
    if isinstance(layer, LowRankLinear):
        w = (layer.u.weight @ layer.v.weight).detach().clone()
        return w, layer.u.bias.detach().clone()
    raise TypeError(type(layer))


def run_dataset(dataset: str, seeds: Sequence[int] = EVAL_SEEDS,
                final_rank: int = FINAL_RANK, width: int = WIDTH,
                total_epochs: int = TOTAL_EPOCHS, max_samples: int = 6000,
                lr: float = LR, batch_size: int = BATCH, threads: int = 6,
                arms: Sequence[str] = ARMS) -> Recorder:
    rec = Recorder(
        experiment_id=f"B-013-{dataset}",
        description=("Rang-Curriculum gegen festen Rang und gegen einen einzelnen "
                     "spaeten Schnitt. Gleiches Epochenbudget, gleicher Endrang."),
        config={"dataset": dataset, "final_rank": final_rank, "width": width,
                "total_epochs": total_epochs, "seeds": list(seeds),
                "max_samples": max_samples, "lr": lr, "batch_size": batch_size,
                "torch_threads": threads, "arms": list(arms),
                "curriculum": [list(x) for x in CURRICULUM],
                "single_cut_epoch": SINGLE_CUT_EPOCH,
                "primary": "curriculum gegen fixed",
                "secondary": "curriculum gegen single_cut",
                "caveat": ("gleiche Epochenzahl und gleicher Endrang, aber nicht "
                           "gleiche Rechenzeit -- das Curriculum rechnet frueh mit "
                           "mehr Parametern. Kein Effizienzanspruch.")},
    )
    for i, seed in enumerate(seeds, 1):
        for arm in arms:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
            model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                              lambda: nn.Linear(width, width), arm, "dense_ends",
                              depth=1)
            schedule = dict(_schedule_for(arm, final_rank))

            # Projektion vor Epoche 0, falls vorgesehen
            if 0 in schedule:
                w, b = _current_weight(model, width)
                model.middles = nn.ModuleList([project_to_lowrank(w, b, schedule[0])])

            optimizer = None
            for epoch in range(total_epochs):
                optimizer, _ = train_with_probe(
                    model, bundle.x_train, bundle.y_train, epochs=1, lr=lr,
                    batch_size=batch_size, seed=seed + epoch, optimizer=optimizer)
                nxt = epoch + 1
                if nxt in schedule and nxt < total_epochs:
                    w, b = _current_weight(model, width)
                    model.middles = nn.ModuleList([project_to_lowrank(w, b, schedule[nxt])])
                    # Parameter haben sich geaendert -> Adam neu aufsetzen.
                    # Betrifft jeden Arm mit Zeitplan gleichermassen.
                    optimizer = None

            # Endmodell muss wirklich den Endrang haben, sonst waere der
            # Vergleich ueber die Modellgroesse unfair
            if arm != "dense" and not isinstance(model.middles[0], LowRankLinear):
                w, b = _current_weight(model, width)
                model.middles = nn.ModuleList([project_to_lowrank(w, b, final_rank)])

            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)
            rec.add(arm=arm, seed=seed, dataset=dataset,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    final_middle_parameters=sum(p.numel() for p in model.middles.parameters()),
                    schedule=str(_schedule_for(arm, final_rank)))
        print(f"  {dataset} seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    from .equivalence import MARGIN, holm, tost_paired

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    dataset = config["dataset"]
    arms = [a for a in ARMS if any(r["arm"] == a for r in records)]

    def vals(arm):
        rows = sorted((r for r in records if r["arm"] == arm), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    n = len(vals(arms[0]))
    print()
    print(f"RESEARCH ID B-013 | {dataset}, {n} Seeds, Endrang {config['final_rank']}")
    head = f'  {"Arm":>12} {"accuracy":>9} {"std":>7} {"Parameter":>12}'
    print(head); print("  " + "-" * (len(head) - 2))
    for arm in arms:
        v = vals(arm)
        par = next((r["final_middle_parameters"] for r in records if r["arm"] == arm), 0)
        print(f'  {arm:>12} {v.mean():>9.4f} {v.std(ddof=1):>7.4f} {par:>12,}')

    out: Dict[str, object] = {"dataset": dataset}
    comparisons = [("curriculum", "fixed", "PRIMAER"),
                   ("curriculum", "single_cut", "sekundaer"),
                   ("single_cut", "fixed", "sekundaer"),
                   ("curriculum", "dense", "Bezug")]
    raw = []
    rows_out = []
    for a, b, tag in comparisons:
        if a not in arms or b not in arms:
            continue
        va, vb = vals(a), vals(b)
        p = float(stats.wilcoxon(va, vb).pvalue)
        eq = tost_paired(va, vb, MARGIN)
        raw.append(p)
        rows_out.append((a, b, tag, float(va.mean() - vb.mean()), p, eq))
    print()
    for (a, b, tag, delta, p, eq), p_h in zip(rows_out, holm(raw)):
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  {a:>12} vs {b:<12} {delta:>+8.4f}  p={p:.4f}  Holm={p_h:.4f}  '
              f'{state}   [{tag}]')
        out[f"{a}_vs_{b}"] = {"delta": delta, "p": p, "p_holm": p_h,
                              "p_tost": eq["p_tost"]}

    prim = out.get("curriculum_vs_fixed", {})
    sec = out.get("curriculum_vs_single_cut", {})
    print()
    if prim.get("delta", 0) > 0 and prim.get("p_holm", 1) < 0.05:
        v = "H1 -- das Rang-Curriculum schlaegt festen Rang bei gleichem Endmodell."
        if sec.get("delta", 0) > 0 and sec.get("p_holm", 1) < 0.05:
            v += " H2 ebenfalls: auch gegen einen einzelnen spaeten Schnitt."
        else:
            v += (" H2 nicht: gegen einen einzelnen spaeten Schnitt kein Vorteil -- "
                  "es zaehlt der Zeitpunkt, nicht das Allmaehliche.")
    else:
        v = ("H0 -- kein signifikanter Vorteil gegenueber festem Rang. Der Befund "
             "aus B-012 laesst sich nicht in ein Verfahren uebersetzen.")
    print(f"  {v}")
    out["verdict"] = v
    return out


if __name__ == "__main__":
    for ds in DATASETS:
        print("\n" + "#" * 74)
        print(f"# B-013 -- {ds}")
        print("#" * 74)
        recorder = run_dataset(ds)
        recorder.save(analysis=analyse(recorder))
