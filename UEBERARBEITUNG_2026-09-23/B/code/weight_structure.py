from __future__ import annotations

"""RESEARCH ID: B-005 -- Lernt eine dichte Schicht ueberhaupt etwas, das keine
Blockstruktur ist?

WARUM DIESE FRAGE
Alle meine Strukturexperimente sind Pferderennen: zwei Architekturen, wer hat am
Ende die hoehere Genauigkeit. Keines schaut hinein. Wenn die Blockdiagonale bei
0,12 % der Parameter mit einer dichten Schicht mithaelt, ist die naheliegende
Erklaerung, dass die dichte Schicht ihre Freiheit gar nicht nutzt -- dass das,
was sie lernt, ohnehin fast blockartig ist.

Das waere eine Erklaerung statt einer weiteren Messung, und Erklaerungen sind
das, was diesem Projekt bisher fehlt.

WAS GEMESSEN WIRD
Die gelernte 4096x4096-Mittelschicht wird in 32x32 Bloecke der Groesse 128
zerlegt und der Betrag der Gewichte je Block aufsummiert. Dann:

  diag_identity   Anteil der Gesamtmasse in den 32 Diagonalbloecken, in der
                  Reihenfolge, in der die Schicht tatsaechlich arbeitet
  diag_random     derselbe Anteil nach zufaelliger Permutation von Zeilen und
                  Spalten -- die Kontrolle. Erwartungswert 32/1024 = 3,125 %
  diag_clustered  derselbe Anteil, nachdem Zeilen und Spalten per
                  Co-Clustering optimal gruppiert wurden: "gibt es IRGENDEINE
                  Anordnung, in der die Schicht blockartig ist?"
  top64_energy    Anteil der Spektralenergie in den 64 groessten
                  Singulaerwerten -- die entsprechende Frage fuer Low-Rank
                  (der lowrank_r64-Arm aus B-004 hat genau Rang 64)

Alles jeweils fuer die trainierte Schicht UND fuer dieselbe Schicht bei
Initialisierung. Die Differenz ist der eigentliche Befund: was das Training
hinzufuegt, nicht was die Initialisierung ohnehin mitbringt.

INTERPRETATION, VORHER FESTGELEGT
  diag_identity ~ diag_random   -> die dichte Schicht lernt nichts Blockartiges.
                                   Dass die Blockdiagonale trotzdem mithaelt,
                                   braucht dann eine andere Erklaerung.
  diag_identity >> diag_random  -> die dichte Schicht konzentriert sich von
                                   selbst auf Bloecke; die Blockbeschraenkung
                                   nimmt ihr kaum etwas weg. Das erklaert den
                                   Befund.
  diag_clustered >> diag_identity -> blockartig, aber in einer anderen
                                   Anordnung als der Index-Reihenfolge. Dann
                                   waere eine gelernte Permutation der
                                   naechste sinnvolle Schritt -- und genau das
                                   ist Monarchs Konstruktionsprinzip.

Kein Signifikanztest ueber Arme, sondern eine Beschreibung mit Kontrolle: die
zufaellige Permutation ist die Nullverteilung, gegen die alles gehalten wird.

Run: python -m qire_ai.weight_structure
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed, train_model
from .data_loader import load_dataset
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder

WIDTH = 4096
BLOCK = 128


def block_mass(weight: torch.Tensor, block: int = BLOCK,
               row_perm: torch.Tensor | None = None,
               col_perm: torch.Tensor | None = None) -> float:
    """Anteil der Betragsmasse, der in den Diagonalbloecken liegt."""
    w = weight.abs()
    if row_perm is not None:
        w = w[row_perm]
    if col_perm is not None:
        w = w[:, col_perm]
    n = w.shape[0]
    groups = n // block
    # zu (groups, block, groups, block) falten und ueber die Blockinhalte summieren
    grid = w.reshape(groups, block, groups, block).sum(dim=(1, 3))
    total = grid.sum()
    diag = grid.diagonal().sum()
    return float(diag / total) if total > 0 else 0.0


def top_energy(weight: torch.Tensor, k: int = 64) -> float:
    """Anteil der Spektralenergie in den k groessten Singulaerwerten."""
    # svd_lowrank braucht nur die fuehrenden Werte -- eine volle SVD einer
    # 4096x4096-Matrix waere hier unnoetig teuer.
    _, s_top, _ = torch.svd_lowrank(weight, q=min(k + 32, weight.shape[0] - 1), niter=4)
    s_top = s_top[:k]
    total = float((weight ** 2).sum())
    return float((s_top ** 2).sum() / total) if total > 0 else 0.0


def clustered_mass(weight: torch.Tensor, groups: int) -> float:
    """Beste erreichbare Blockmasse nach Co-Clustering von Zeilen und Spalten.

    Beantwortet: ist die Matrix blockartig in IRGENDEINER Anordnung? Wenn das
    Verfahren nicht verfuegbar ist oder scheitert, wird NaN zurueckgegeben --
    lieber eine fehlende Zahl als eine erfundene.
    """
    try:
        from sklearn.cluster import SpectralCoclustering
        w = weight.abs().double().numpy()
        model = SpectralCoclustering(n_clusters=groups, svd_method="randomized",
                                     random_state=0, n_init=5)
        model.fit(w)
        row_perm = torch.from_numpy(np.argsort(model.row_labels_))
        col_perm = torch.from_numpy(np.argsort(model.column_labels_))
        # Nach dem Sortieren sind die Cluster zusammenhaengend, aber nicht
        # zwingend gleich gross -- deshalb wird hier ueber die tatsaechlichen
        # Clustergrenzen summiert statt ueber feste Bloecke.
        w_sorted = torch.from_numpy(w)[row_perm][:, col_perm]
        rl = np.sort(model.row_labels_)
        cl = np.sort(model.column_labels_)
        total = float(w_sorted.sum())
        diag = 0.0
        for g in range(groups):
            rows = torch.from_numpy(np.where(rl == g)[0])
            cols = torch.from_numpy(np.where(cl == g)[0])
            if len(rows) and len(cols):
                diag += float(w_sorted[rows][:, cols].sum())
        return diag / total if total > 0 else float("nan")
    except Exception as exc:      # noqa: BLE001 -- bewusst breit, siehe Docstring
        print(f"    (Co-Clustering nicht verfuegbar: {exc})")
        return float("nan")


def middle_weight(model: MiddleNet) -> torch.Tensor:
    """Die Gewichtsmatrix der dichten Mittelschicht."""
    layer = model.middles[0]
    assert isinstance(layer, nn.Linear), "B-005 analysiert nur die dichte Mittelschicht"
    return layer.weight.detach().clone()


def run(
    width: int = WIDTH,
    seeds: Sequence[int] = (1501, 1502, 1503, 1504, 1505),
    archs: Sequence[str] = ("dense_ends", "local_ends"),
    max_samples: int = 6000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
    groups: int = 16,
    do_clustering: bool = True,
) -> Recorder:
    n_blocks = width // BLOCK
    rec = Recorder(
        experiment_id="B-005",
        description=("Struktur der gelernten dichten Mittelschicht: wieviel "
                     "Betragsmasse liegt in Diagonalbloecken, verglichen mit "
                     "zufaelliger Permutation als Nullverteilung, vor und nach "
                     "dem Training."),
        config={"dataset": "mnist", "width": width, "block": BLOCK,
                "n_blocks": n_blocks, "seeds": list(seeds), "archs": list(archs),
                "max_samples": max_samples, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "torch_threads": threads,
                "chance_level_diag": n_blocks / (n_blocks ** 2),
                "note": "nur der Arm mit dichter Mittelschicht wird analysiert"},
    )
    for seed in seeds:
        for arch in archs:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            model = MiddleNet(bundle.input_dim, width, bundle.num_classes,
                              lambda: nn.Linear(width, width), "dense", arch,
                              groups=groups, depth=1)
            w_init = middle_weight(model)
            train_model(model, bundle.x_train, bundle.y_train, epochs=epochs,
                        lr=lr, seed=seed, batch_size=batch_size)
            w_trained = middle_weight(model)
            _, _, _, ex = evaluate(model, bundle.x_test, bundle.y_test)

            g = torch.Generator().manual_seed(seed)
            rp = torch.randperm(width, generator=g)
            cp = torch.randperm(width, generator=g)

            row = {
                "seed": seed, "arch": arch,
                "balanced_accuracy": float(ex["balanced_accuracy"]),
                "diag_identity_trained": block_mass(w_trained),
                "diag_identity_init": block_mass(w_init),
                "diag_random_trained": block_mass(w_trained, row_perm=rp, col_perm=cp),
                "diag_random_init": block_mass(w_init, row_perm=rp, col_perm=cp),
                "top64_energy_trained": top_energy(w_trained),
                "top64_energy_init": top_energy(w_init),
            }
            if do_clustering:
                row["diag_clustered_trained"] = clustered_mass(w_trained, n_blocks)
                row["diag_clustered_init"] = clustered_mass(w_init, n_blocks)
            rec.add(**row)
            print(f"  seed {seed} / {arch} fertig")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    n_blocks, width, block = (int(config["n_blocks"]), int(config["width"]),
                              int(config["block"]))

    chance = n_blocks / (n_blocks ** 2)
    archs = sorted({r["arch"] for r in records})
    out: Dict[str, object] = {"chance_level": chance, "per_arch": {}}

    print()
    print(f"RESEARCH ID B-005 | dichte {width}x{width}-Mittelschicht, "
          f"{n_blocks}x{n_blocks} Bloecke der Groesse {block}")
    print(f"Zufallserwartung fuer die Diagonalmasse: {chance:.4f} ({chance*100:.2f} %)")

    for arch in archs:
        rows = [r for r in records if r["arch"] == arch]
        print()
        print(f"--- {arch} " + "-" * max(4, 56 - len(arch)))
        header = f'{"Groesse":>26} {"init":>10} {"trainiert":>10} {"Aenderung":>11}'
        print(header); print("-" * len(header))
        arch_out: Dict[str, float] = {}
        for key, label in [("diag_identity", "Diagonalmasse (Index)"),
                           ("diag_random", "Diagonalmasse (zufaellig)"),
                           ("diag_clustered", "Diagonalmasse (geclustert)"),
                           ("top64_energy", "Energie in Top-64-Rang")]:
            init_vals = [r[f"{key}_init"] for r in rows if f"{key}_init" in r] or [float("nan")]
            tr_vals = [r[f"{key}_trained"] for r in rows if f"{key}_trained" in r]
            if not tr_vals or np.all(np.isnan(tr_vals)):
                continue
            i_m, t_m = float(np.nanmean(init_vals)), float(np.nanmean(tr_vals))
            print(f'{label:>26} {i_m:>10.4f} {t_m:>10.4f} {t_m-i_m:>+11.4f}')
            arch_out[f"{key}_init"] = i_m
            arch_out[f"{key}_trained"] = t_m

        ident = arch_out.get("diag_identity_trained", float("nan"))
        rand = arch_out.get("diag_random_trained", float("nan"))
        clus = arch_out.get("diag_clustered_trained", float("nan"))
        print()
        ratio = ident / rand if rand else float("nan")
        print(f'  Index-Anordnung gegen Zufallsanordnung: Faktor {ratio:.2f}')
        if not np.isnan(clus):
            print(f'  Beste gefundene Anordnung gegen Index-Anordnung: '
                  f'Faktor {clus/ident:.2f}' if ident else "")
        if abs(ident - rand) < 0.1 * rand:
            reading = ("Die dichte Schicht ist NICHT blockartig in ihrer eigenen "
                       "Reihenfolge -- sie liegt auf Zufallsniveau. Dass die "
                       "Blockdiagonale mithaelt, ist damit nicht dadurch erklaert, "
                       "dass die dichte Schicht ihre Freiheit nicht nutzt.")
        elif ident > rand:
            reading = ("Die dichte Schicht konzentriert Masse auf Diagonalbloecke, "
                       "deutlich ueber Zufallsniveau -- die Blockbeschraenkung "
                       "nimmt ihr also wenig weg. Das ist eine Erklaerung fuer "
                       "den Befund aus A-002/B-004.")
        else:
            reading = ("Die dichte Schicht meidet Diagonalbloecke -- unerwartet, "
                       "gehoert so berichtet und nicht wegerklaert.")
        print(f'  Lesart: {reading}')
        arch_out["reading"] = reading      # type: ignore[assignment]
        out["per_arch"][arch] = arch_out   # type: ignore[index]

    return out


if __name__ == "__main__":
    recorder = run()
    result = analyse(recorder)
    recorder.save(analysis=result)
