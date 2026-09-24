from __future__ import annotations

"""RESEARCH ID: B-012 -- Wieviel ist der Weg wert? Eine Dosis-Wirkungs-Kurve.

+-------------------------------------------------------------------------+
| ERGEBNIS (nachgetragen, damit die Hypothese unten nicht als Befund       |
| gelesen wird):                                                          |
|                                                                         |
| Die Vermutung "das Ziel ist niedrigrangig, aber der Weg fuehrt durch     |
| hohen Rang" ist WIDERLEGT. Der Effektivrang faellt nicht, er STEIGT      |
| (MNIST 5,1 -> 6,7 ueber 10 Epochen, rho=+0,988). Schon nach EINER Epoche |
| liegen 99,9 % der Ausgabeenergie in 64 Richtungen -- die                 |
| niedrigdimensionale Struktur ist praktisch sofort da.                    |
|                                                                         |
| Was bleibt: Beschraenken ab Epoche 0 kostet rund 0,010 gegenueber        |
| spaeterem Beschraenken, aber der Vorteil ist nach zwei bis drei Epochen  |
| ausgeschoepft. Deshalb ist der Zusammenhang nur auf Fashion-MNIST        |
| monoton signifikant (rho=+0,857, p=0,0137), auf MNIST nicht -> H0.       |
| H2 (optimales k waechst mit dem Rang) ist widerspruechlich zwischen den  |
| Datensaetzen und damit nicht etabliert.                                  |
+-------------------------------------------------------------------------+

DIE FRAGE
B-011 hat zwei Endpunkte gezeigt: eine Rang-64-Schicht, von null trainiert, ist
schlechter als dicht (-0,0114); dieselbe Schicht, aus der fertig trainierten
dichten Matrix projiziert, ist ihr gleichwertig. Die Erklaerung, die ich dafuer
vermute: Das Ziel ist niedrigrangig, aber der WEG dorthin fuehrt durch hohen
Rang. Wer von Anfang an beschraenkt, versperrt den Weg.

Das war bisher eine Behauptung mit zwei Datenpunkten. Hier wird daraus eine
Dosis-Wirkungs-Kurve -- der Goldstandard fuer eine kausale Aussage, und etwas,
das dieses Projekt noch nie hatte.

DER AUFBAU
Fuer jeden Seed EINE dichte Trainingsbahn ueber 10 Epochen. An sieben Stellen
k = 0, 1, 2, 3, 5, 7, 10 wird abgezweigt: Mittelschicht auf Rang r projizieren,
danach die restlichen 10-k Epochen beschraenkt weitertrainieren.

    k = 0   : sofort projiziert -> praktisch von null beschraenkt trainiert
    k = 10  : voll dicht trainiert, dann projiziert, kein Nachlauf (= B-011)

**Das Gesamtbudget ist in jedem Arm exakt 10 Epochen.** Damit faellt der
Einwand weg, der bei B-011 noch galt: dort brauchte die Projektion vorher ein
komplett trainiertes dichtes Modell und kostete deshalb mehr. Hier kostet jeder
Punkt der Kurve dasselbe. Was sich unterscheidet, ist ausschliesslich, WANN die
Beschraenkung einsetzt.

Alle Abzweigungen eines Seeds teilen sich dieselbe dichte Bahn. Damit ist der
Verzweigungszeitpunkt die einzige Variable -- kein zusaetzliches Rauschen durch
verschiedene dichte Laeufe.

VORHERSAGE, VORHER FESTGELEGT
  H1 : Die Genauigkeit steigt monoton mit k. Spearman(k, Genauigkeit) > 0 und
       signifikant, bei Rang 64 auf beiden Datensaetzen.
       -> Der Weg hat einen messbaren Wert, und zwar bei gleichen Kosten.
          Das ist die kausale Fassung des Befunds aus B-011.
  H0 : Kein monotoner Zusammenhang.
       -> Der Vorteil in B-011 kam nicht vom Weg, sondern von der zusaetzlichen
          Rechenzeit, die dort ungleich verteilt war.

  ZUSATZ (loest einen Widerspruch in meinen eigenen Ergebnissen):
  Mein Faktor-6,4-Befund sagt, nachtraeglich aufgezwungene Struktur koste mehr
  als mittrainierte. B-011 sagt bei Rang 64 das Gegenteil. Die Trajektorien-
  Erklaerung sagt vorher, WANN welches gilt: Enthaelt die Familie die dichte
  Loesung (hoher Rang), gewinnt spaetes Projizieren; ist sie zu eng (niedriger
  Rang), gewinnt frueh beschraenktes Training, weil die Familie dann eine
  eigene Loesung suchen muss. Also:

  H2 : Das optimale k waechst mit dem Rang. Bei Rang 8 liegt es niedrig, bei
       Rang 64 hoch.
       -> Der Widerspruch ist aufgeloest, beide frueheren Befunde sind
          Spezialfaelle einer Regel.

Run: python -m qire_ai.trajectory_dose
"""

import copy
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
from .spectral_probe import (middle_outputs, output_spectrum, per_class_spectra,
                             train_with_probe)

WIDTH = 4096
TOTAL_EPOCHS = 10
K_VALUES: Tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10)
RANKS: Tuple[int, ...] = (8, 32, 64)
ENERGY_RANKS = (8, 32, 64, 256)
DATASETS = ("mnist", "fashion_mnist")
EVAL_SEEDS = tuple(range(2001, 2013))      # 12 Seeds
LR = 0.003
BATCH = 256
OVERSAMPLE = 64                            # Reserve fuer die genaeherte SVD


def top_svd(weight: torch.Tensor, rank: int, oversample: int = OVERSAMPLE,
            niter: int = 8):
    """Fuehrende Singulaertripel. Genaehert, aber mit reichlich Reserve.

    Eine volle SVD einer 4096x4096-Matrix kostet knapp eine Minute und muesste
    hier hunderte Male gerechnet werden. `svd_lowrank` mit grosszuegiger
    Ueberabtastung ist um Groessenordnungen billiger; wie gross der Fehler
    tatsaechlich ist, wird im Smoke-Test gegen die exakte SVD gemessen statt
    angenommen.
    """
    q = min(weight.shape[0], rank + oversample)
    U, S, V = torch.svd_lowrank(weight, q=q, niter=niter)
    return U[:, :rank], S[:rank], V[:, :rank]


def project_to_lowrank(weight: torch.Tensor, bias: torch.Tensor,
                       rank: int) -> LowRankLinear:
    """Rang-r-Naeherung als eigenstaendige Low-Rank-Schicht."""
    U, S, V = top_svd(weight, rank)
    root = torch.sqrt(S.clamp_min(0))
    layer = LowRankLinear(weight.shape[0], rank)
    with torch.no_grad():
        layer.v.weight.copy_((root[:, None] * V.T))     # (r, dim)
        layer.u.weight.copy_(U * root)                  # (dim, r)
        layer.u.bias.copy_(bias)
    return layer


def run_dataset(dataset: str, seeds: Sequence[int] = EVAL_SEEDS,
                ranks: Sequence[int] = RANKS, ks: Sequence[int] = K_VALUES,
                width: int = WIDTH, total_epochs: int = TOTAL_EPOCHS,
                max_samples: int = 6000, lr: float = LR, batch_size: int = BATCH,
                threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id=f"B-012-{dataset}",
        description=("Dosis-Wirkungs-Kurve fuer den Trainingsweg: nach k von "
                     "insgesamt 10 Epochen wird die Mittelschicht auf Rang r "
                     "projiziert und beschraenkt weitertrainiert. Gesamtbudget "
                     "in jedem Arm identisch."),
        config={"dataset": dataset, "k_values": list(ks), "ranks": list(ranks),
                "width": width, "total_epochs": total_epochs, "seeds": list(seeds),
                "max_samples": max_samples, "lr": lr, "batch_size": batch_size,
                "torch_threads": threads, "energy_ranks": list(ENERGY_RANKS),
                "design": ("eine dichte Bahn je Seed, an sieben Stellen abgezweigt; "
                           "Gesamtbudget konstant 10 Epochen in jedem Arm"),
                "caveat": ("beim Wechsel auf die beschraenkte Schicht wird der "
                           "Adam-Zustand neu aufgesetzt -- gilt fuer jedes k>0 "
                           "gleichermassen")},
    )
    for i, seed in enumerate(seeds, 1):
        set_seed(seed)
        torch.set_num_threads(threads)
        bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)

        # --- eine dichte Bahn, an den k-Stellen Zwischenstaende sichern ---
        dense = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                          lambda: nn.Linear(width, width), "dense", "dense_ends",
                          depth=1)
        snapshots: Dict[int, Tuple[torch.Tensor, torch.Tensor, dict]] = {}
        if 0 in ks:
            snapshots[0] = (dense.middles[0].weight.detach().clone(),
                            dense.middles[0].bias.detach().clone(),
                            copy.deepcopy(dense.state_dict()))

        def hook(epoch: int, model: nn.Module) -> None:
            done = epoch + 1
            if done in ks:
                snapshots[done] = (model.middles[0].weight.detach().clone(),
                                   model.middles[0].bias.detach().clone(),
                                   copy.deepcopy(model.state_dict()))
            # Spektrum der Mittelschicht auf echten Daten, je Epoche.
            # 2048 Zeilen genuegen: der Effektivrang liegt bei rund 64, und eine
            # SVD ueber 4096 Zeilen wuerde allein hierfuer gut eine Stunde kosten.
            spec = output_spectrum(middle_outputs(model, bundle.x_train, max_rows=2048),
                                   ENERGY_RANKS)
            rec.add(arm="dense_trajectory", rank=0, k=done, seed=seed,
                    dataset=dataset, balanced_accuracy=float("nan"), **spec)

        train_with_probe(dense, bundle.x_train, bundle.y_train, epochs=total_epochs,
                         lr=lr, batch_size=batch_size, seed=seed, on_epoch=hook)

        _, _, _, ex = evaluate(dense, bundle.x_test, bundle.y_test)
        dense_acc = float(ex["balanced_accuracy"])
        outputs = middle_outputs(dense, bundle.x_train)
        cls = per_class_spectra(outputs, bundle.y_train[:outputs.shape[0]])
        rec.add(arm="dense", rank=0, k=total_epochs, seed=seed, dataset=dataset,
                balanced_accuracy=dense_acc,
                **output_spectrum(outputs, ENERGY_RANKS), **cls)

        # --- Abzweigungen ---
        for rank in ranks:
            for k in ks:
                w, b, state = snapshots[k]
                set_seed(seed + 10_000 * k + rank)
                torch.set_num_threads(threads)
                branch = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                                   lambda: nn.Linear(width, width), "dense",
                                   "dense_ends", depth=1)
                branch.load_state_dict(state)
                branch.middles = nn.ModuleList([project_to_lowrank(w, b, rank)])
                remaining = total_epochs - k
                if remaining > 0:
                    train_with_probe(branch, bundle.x_train, bundle.y_train,
                                     epochs=remaining, lr=lr, batch_size=batch_size,
                                     seed=seed + k)
                _, _, _, ex = evaluate(branch, bundle.x_test, bundle.y_test)
                rec.add(arm="branch", rank=rank, k=k, seed=seed, dataset=dataset,
                        balanced_accuracy=float(ex["balanced_accuracy"]),
                        dense_reference=dense_acc,
                        middle_parameters=sum(p.numel() for p in branch.middles.parameters()))
        print(f"  {dataset} seed {seed} fertig ({i}/{len(seeds)})")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    ks = [int(k) for k in config["k_values"]]
    ranks = [int(r) for r in config["ranks"]]
    dataset = config["dataset"]
    out: Dict[str, object] = {"dataset": dataset}

    def branch(rank, k):
        rows = sorted((r for r in records if r["arm"] == "branch"
                       and r["rank"] == rank and r["k"] == k), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    dense = np.array([r["balanced_accuracy"] for r in
                      sorted((r for r in records if r["arm"] == "dense"),
                             key=lambda r: r["seed"])])
    print()
    print(f"RESEARCH ID B-012 | {dataset}, {len(dense)} Seeds, "
          f"dichte Referenz {dense.mean():.4f}")

    # --- Rangkollaps waehrend des Trainings ---
    print()
    print("Effektivrang der Mittelschicht waehrend des dichten Trainings:")
    head = f'  {"Epoche":>7} {"Effektivrang":>13} {"Entropie-Rang":>14} {"Energie@64":>11}'
    print(head); print("  " + "-" * (len(head) - 2))
    traj = []
    for k in sorted({int(r["k"]) for r in records if r["arm"] == "dense_trajectory"}):
        rows = [r for r in records if r["arm"] == "dense_trajectory" and r["k"] == k]
        er = float(np.mean([r["effective_rank"] for r in rows]))
        en = float(np.mean([r["entropy_rank"] for r in rows]))
        e64 = float(np.mean([r["energy_64"] for r in rows]))
        print(f'  {k:>7} {er:>13.1f} {en:>14.1f} {e64:>11.4f}')
        traj.append((k, er))
        out[f"effective_rank_epoch_{k}"] = er
    if len(traj) > 2:
        rho, p = stats.spearmanr([t[0] for t in traj], [t[1] for t in traj])
        verdict = ("faellt waehrend des Trainings" if rho < 0 else "steigt")
        print(f'  -> Effektivrang {verdict}: Spearman rho={rho:+.3f}, p={p:.4f}')
        out["rank_trend_rho"] = float(rho)
        out["rank_trend_p"] = float(p)

    # --- Klassenweise Unterraeume ---
    cls_rows = [r for r in records if r["arm"] == "dense" and "mean_principal_cosine" in r]
    if cls_rows:
        pc = float(np.mean([r["per_class_effective_rank"] for r in cls_rows]))
        mc = float(np.mean([r["mean_principal_cosine"] for r in cls_rows]))
        print()
        print(f"Klassenweise Unterraeume: Effektivrang je Klasse {pc:.1f}, "
              f"mittlerer Hauptkosinus {mc:.3f}")
        print("  (nahe 1 = alle Klassen teilen denselben Unterraum;")
        print("   nahe 0 = jede Klasse hat ihren eigenen -> implizites MoE)")
        out["per_class_effective_rank"] = pc
        out["mean_principal_cosine"] = mc

    # --- die Dosis-Wirkungs-Kurve ---
    print()
    print("Dosis-Wirkungs-Kurve: Genauigkeit nach Verzweigungszeitpunkt k")
    head = f'  {"Rang":>6} ' + " ".join(f'{"k="+str(k):>9}' for k in ks) + f' {"Spearman":>10} {"p":>8}'
    print(head); print("  " + "-" * (len(head) - 2))
    per_rank: Dict[str, object] = {}
    for rank in ranks:
        means = [branch(rank, k).mean() for k in ks]
        rho, p = stats.spearmanr(ks, means)
        best_k = ks[int(np.argmax(means))]
        line = f'  {rank:>6} ' + " ".join(f'{m:>9.4f}' for m in means)
        print(line + f' {rho:>10.3f} {p:>8.4f}   bestes k={best_k}')
        per_rank[str(rank)] = {"means": [float(m) for m in means], "rho": float(rho),
                               "p": float(p), "best_k": int(best_k)}
    out["per_rank"] = per_rank

    print()
    print("=" * 78)
    print("BEWERTUNG (Kriterien vor dem Lauf festgelegt)")
    top = per_rank[str(max(ranks))]
    if top["rho"] > 0 and top["p"] < 0.05:                    # type: ignore[index]
        v1 = (f"H1 -- bei Rang {max(ranks)} steigt die Genauigkeit monoton mit dem "
              f"Verzweigungszeitpunkt (rho={top['rho']:+.3f}, p={top['p']:.4f}). "  # type: ignore[index]
              f"Der Weg hat einen messbaren Wert, bei identischem Budget.")
    else:
        v1 = (f"H0 -- kein monotoner Zusammenhang bei Rang {max(ranks)}. Der "
              f"Vorteil in B-011 kam nicht vom Weg.")
    print(f"  {v1}")
    best_ks = [per_rank[str(r)]["best_k"] for r in ranks]      # type: ignore[index]
    rho2, p2 = stats.spearmanr(ranks, best_ks)
    if rho2 > 0:
        v2 = (f"H2 gestuetzt -- das optimale k waechst mit dem Rang "
              f"({dict(zip(ranks, best_ks))}, rho={rho2:+.3f}). Damit sind der "
              f"Faktor-6,4-Befund und B-011 Spezialfaelle derselben Regel.")
    else:
        v2 = (f"H2 nicht gestuetzt -- das optimale k waechst nicht mit dem Rang "
              f"({dict(zip(ranks, best_ks))}).")
    print(f"  {v2}")
    print("=" * 78)
    out["verdict"] = [v1, v2]
    return out


if __name__ == "__main__":
    for ds in DATASETS:
        print("\n" + "#" * 74)
        print(f"# B-012 -- {ds}")
        print("#" * 74)
        recorder = run_dataset(ds)
        recorder.save(analysis=analyse(recorder))
