from __future__ import annotations

"""RESEARCH ID: Q-006 -- Ising/QUBO-Pruning: macht ein quantenabgeleitetes
Auswahlverfahren das Netz messbar effizienter?

WARUM DIESER LAUF -- und warum die Vorhersage aus eigenen Daten stammt
In `ROADMAP.md` steht seit dem 2. August als "Weg 2": Sparsity-Muster als
kombinatorisches Optimierungsproblem finden statt per Magnitude-Heuristik. Nie
ausgefuehrt. `regime_generalization.py` hat aber genau die Zahl geliefert, die
vorhersagt, dass es wirken muesste:

    Guete des nachtraeglichen Verfahrens        gemessene Luecke
    beweisbar optimal (Eckart-Young/SVD)        2,8x
    quasi-optimal (gierig, TT-SVD)             11,8x
    reine Heuristik (Magnitude) -> width       17,4x

Die Luecke ist exakt invers zur Guete des Auswahlverfahrens geordnet. Und
ausgerechnet beim strukturierten Pruning -- dem EINZIGEN Verfahren im ganzen
Projekt, bei dem die Ersparnis in gemessener Zeit ankommt (1,89x bei 512 und
1,95x bei 2048 Neuronen) -- steht das schlechteste Auswahlverfahren.

DER QUANTENBEZUG IST KEINE ANALOGIE, SONDERN DIESELBE MATHEMATIK
Welche Neuronen behalten? Sei H die Aktivierungsmatrix der versteckten Schicht
auf echten Daten und W2 die Ausgangsgewichte. Der Beitrag der Schicht ist

    Y = sum_i H[:,i] (x) W2[:,i]

Entfernt man die Menge U, ist der Rekonstruktionsfehler

    E(u) = || sum_i u_i H[:,i] W2[:,i]^T ||_F^2
         = sum_ij u_i u_j (H[:,i].H[:,j]) (W2[:,i].W2[:,j])
         = u^T Q u        mit  Q = G (*) K, G = Gram(H), K = Gram(W2)

u ist binaer, Q symmetrisch, dazu eine Kardinalitaetsbedingung. Das IST ein
QUBO, also ein Ising-Modell -- dieselbe Form, die Quantenannealer minimieren.
Kein Vokabelaustausch: die Zielfunktion des Prunings ist buchstaeblich eine
Ising-Energie mit Kopplungsmatrix Q.

**Und hier steckt die inhaltliche Aussage:** Die Magnitude-Heuristik benutzt nur
die DIAGONALE von Q -- jedes Neuron wird fuer sich bewertet. Der Ising-Ansatz
benutzt die Wechselwirkungen Q_ij: zwei redundante Neuronen haben beide grosse
Diagonaleintraege, aber beide zu behalten ist Verschwendung. Genau diese
Kopplung ist das, was die Heuristik nicht sehen kann.

DIE ARME -- die Leiter ist der Abgrenzungstest
  magnitude     ||w_in|| * ||w_out||, das Verfahren aus `structured_pruning.py`.
                Die Basis, gegen die der Anspruch antritt.
  diagonal      sqrt(Q_ii), also datenbewusst, aber ohne Wechselwirkungen.
                **Trennt "Daten benutzen" von "Kopplungen benutzen".**
  qubo_anneal   volles Q, minimiert per simuliertem Annealing mit
                kardinalitaetserhaltenden Tauschzuegen. Das Ising-Verfahren.
  random        zufaellige Maske gleicher Groesse. Positivkontrolle.

Die Leiter ist bewusst so gebaut, dass der eigene Anspruch daran scheitern
kann: Holt `diagonal` schon den ganzen Gewinn, dann traegt der Ising-Teil
NICHTS, und der Befund lautet "datenbewusstes Pruning hilft", nicht
"quantenabgeleitetes Pruning hilft". Das waere dann so zu berichten.

VORREGISTRIERUNG
  Gemessen wird balancierte Genauigkeit **unmittelbar nach dem Pruning**
  (primaer) und nach 1, 5 und 20 Nachtrainings-Epochen (Ressourcenaussage:
  braucht ein besserer Start weniger Nachtraining?).

  PRIMAER  qubo_anneal schlaegt magnitude ohne Nachtraining auf dem
           Primaerdatensatz.
  P2       qubo_anneal schlaegt diagonal -- der Abgrenzungstest. Nur wenn
           dieser haelt, traegt der Ising-Anteil selbst etwas bei.
  Holm ueber genau diese beiden. Alles andere ist Einordnung.

  H1 : beide halten, auf beiden Datensaetzen gleiche Richtung
       -> ein quantenabgeleitetes Verfahren macht das Netz messbar besser bei
          identischem Parameterbudget. Das waere der erste Positivbefund dieser
          Art im Projekt.
  H0 : PRIMAER haelt, P2 nicht (TOST weist Gleichwertigkeit zu `diagonal` nach)
       -> der Gewinn kommt daher, die Daten zu benutzen, nicht aus der
          Ising-Struktur. Ehrlich zu berichten als klassischer Befund.
  HU : PRIMAER haelt nicht -> kein Gewinn gegenueber der Heuristik.

  AUFLOESUNGSPRUEFUNG (Regel 6): `random` MUSS signifikant schlechter sein als
  `magnitude`. Kann der Aufbau nicht einmal eine zufaellige von einer
  heuristischen Maske trennen, ist er fuer die eigentliche Frage blind und kein
  Ergebnis dieses Laufs ist interpretierbar.

Run: python -m quantum_sim.qubo_pruning
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from qire_ai.benchmark import set_seed, train_model
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.metrics import evaluate
from qire_ai.models_baseline import DenseMLP
from qire_ai.result_store import Recorder

ARMS = ("magnitude", "diagonal", "qubo_anneal", "random")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(6101, 6116))       # 15 Seeds
TUNING_SEEDS = (6201, 6202)
LR_GRID = (0.003, 0.01, 0.03)
HIDDEN = 512                                # oberhalb der 512-Neuronen-Schwelle
KEEP = 256                                  # 50 % Sparsity
MAX_SAMPLES = 6000
DENSE_EPOCHS = 10
FT_CHECKPOINTS = (0, 1, 5, 20)
GRAM_BATCH = 2000                           # Beispiele fuer die Gram-Matrizen
ANNEAL_SWEEPS = 4000


# --------------------------------------------------------------------------
# Die Ising-Kopplungsmatrix
# --------------------------------------------------------------------------

def coupling_matrix(model: DenseMLP, x: torch.Tensor) -> np.ndarray:
    """Q = Gram(H) (*) Gram(W2) -- die Kopplungsmatrix des Pruning-Ising-Modells.

    Q_ij = (H[:,i].H[:,j]) * (W2[:,i].W2[:,j]). Der Rekonstruktionsfehler beim
    Entfernen der Menge u ist exakt u^T Q u.
    """
    with torch.no_grad():
        h = torch.relu(model.net[0](x))           # (batch, hidden)
        w2 = model.net[2].weight                  # (classes, hidden)
        g = (h.T @ h).cpu().numpy()               # Gram der Aktivierungen
        k = (w2.T @ w2).cpu().numpy()             # Gram der Ausgangsgewichte
    q = g * k
    return 0.5 * (q + q.T)                        # exakte Symmetrie erzwingen


def anneal_mask(q: np.ndarray, remove: int, seed: int,
                sweeps: int = ANNEAL_SWEEPS) -> np.ndarray:
    """Minimiert u^T Q u unter |u| = remove per simuliertem Annealing.

    Zuege sind **Tausche** (ein entferntes Neuron gegen ein behaltenes), damit
    die Kardinalitaet exakt erhalten bleibt -- keine Strafterme noetig, die die
    Bedingung nur weich durchsetzen wuerden.
    """
    rng = np.random.default_rng(seed)
    d = q.shape[0]
    scale = float(np.abs(q).max()) or 1.0
    qs = q / scale

    # Start bei der Diagonallosung: die Neuronen mit der kleinsten Eigenenergie
    # entfernen. Damit startet das Annealing nie schlechter als `diagonal`.
    u = np.zeros(d, dtype=bool)
    u[np.argsort(np.diag(qs))[:remove]] = True

    s = qs @ u.astype(float)                      # s_i = sum_j Q_ij u_j
    energy = float(u.astype(float) @ s)
    removed = np.flatnonzero(u)
    kept = np.flatnonzero(~u)

    t0, t1 = 1.0, 1e-4
    for sweep in range(sweeps):
        t = t0 * (t1 / t0) ** (sweep / max(1, sweeps - 1))
        ia = int(rng.integers(len(removed)))
        ib = int(rng.integers(len(kept)))
        a, b = int(removed[ia]), int(kept[ib])

        d_out = -2.0 * s[a] + qs[a, a]            # a: entfernt -> behalten
        s_tmp = s - qs[:, a]
        d_in = 2.0 * s_tmp[b] + qs[b, b]          # b: behalten -> entfernt
        delta = d_out + d_in

        if delta <= 0 or rng.random() < np.exp(-delta / max(t, 1e-12)):
            s = s_tmp + qs[:, b]
            u[a], u[b] = False, True
            removed[ia], kept[ib] = b, a
            energy += delta
    return u


def select_removed(arm: str, model: DenseMLP, x: torch.Tensor, remove: int,
                   seed: int) -> np.ndarray:
    """Boolesche Maske: True = dieses Neuron wird entfernt."""
    d = model.net[0].weight.shape[0]
    if arm == "magnitude":
        with torch.no_grad():
            imp = (model.net[0].weight.norm(dim=1)
                   * model.net[2].weight.norm(dim=0)).cpu().numpy()
        u = np.zeros(d, dtype=bool)
        u[np.argsort(imp)[:remove]] = True
        return u
    if arm == "random":
        rng = np.random.default_rng(seed)
        u = np.zeros(d, dtype=bool)
        u[rng.choice(d, size=remove, replace=False)] = True
        return u
    q = coupling_matrix(model, x)
    if arm == "diagonal":
        u = np.zeros(d, dtype=bool)
        u[np.argsort(np.diag(q))[:remove]] = True
        return u
    if arm == "qubo_anneal":
        return anneal_mask(q, remove, seed)
    raise ValueError(arm)


def prune(model: DenseMLP, keep_idx: np.ndarray) -> DenseMLP:
    """Echt kleineres Modell -- keine Maskierung, keine Nullen."""
    idx = torch.as_tensor(np.sort(keep_idx), dtype=torch.long)
    small = DenseMLP(model.input_dim, int(len(idx)), model.num_classes)
    with torch.no_grad():
        small.net[0].weight.copy_(model.net[0].weight[idx])
        small.net[0].bias.copy_(model.net[0].bias[idx])
        small.net[2].weight.copy_(model.net[2].weight[:, idx])
        small.net[2].bias.copy_(model.net[2].bias)
    return small


def _bal_acc(model, bundle) -> float:
    _, _, _, extras = evaluate(model, bundle.x_test, bundle.y_test)
    return float(extras.get("balanced_accuracy", 0.0))


# --------------------------------------------------------------------------
# Eine Zelle
# --------------------------------------------------------------------------

def _cell(arm: str, dataset: str, seed: int, ft_lr: float, threads: int,
          max_samples: int = MAX_SAMPLES,
          checkpoints: Sequence[int] = FT_CHECKPOINTS) -> List[Dict[str, object]]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)

    dense = DenseMLP(bundle.input_dim, HIDDEN, bundle.num_classes)
    train_model(dense, bundle.x_train, bundle.y_train,
                epochs=DENSE_EPOCHS, lr=0.003, seed=seed)
    dense_acc = _bal_acc(dense, bundle)

    x_gram = bundle.x_train[:GRAM_BATCH]
    u = select_removed(arm, dense, x_gram, HIDDEN - KEEP, seed)
    small = prune(dense, np.flatnonzero(~u))

    # Die Ising-Energie der gewaehlten Maske mitschreiben: sie ist die Groesse,
    # die das Verfahren minimiert, und macht nachpruefbar, ob das Annealing
    # ueberhaupt besser optimiert hat als die Diagonallosung.
    q = coupling_matrix(dense, x_gram)
    uf = u.astype(float)
    energy = float(uf @ q @ uf) / (float(np.abs(q).max()) or 1.0)

    rows: List[Dict[str, object]] = []
    done = 0
    for cp in sorted(checkpoints):
        if cp > done:
            train_model(small, bundle.x_train, bundle.y_train,
                        epochs=cp - done, lr=ft_lr, seed=seed + cp)
            done = cp
        rows.append({"arm": arm, "dataset": dataset, "seed": seed,
                     "finetune_epochs": cp, "lr": ft_lr,
                     "balanced_accuracy": _bal_acc(small, bundle),
                     "dense_balanced_accuracy": dense_acc,
                     "ising_energy": energy,
                     "hidden": HIDDEN, "kept": KEEP,
                     "parameters": sum(p.numel() for p in small.parameters())})
    return rows


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    """Nachtrainings-Lernrate je Arm, auf eigenen Tuning-Seeds (Regel 1)."""
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {}
        for lr in LR_GRID:
            vals = []
            for s in TUNING_SEEDS:
                rows = _cell(arm, dataset, s, lr, threads,
                             max_samples=3000, checkpoints=(5,))
                vals.append(float(rows[-1]["balanced_accuracy"]))
            means[lr] = float(np.mean(vals))
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<12} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-006",
        description=("Ising/QUBO-Pruning gegen Magnitude-Heuristik. Die "
                     "Maskenwahl ist buchstaeblich eine Ising-Energie; getestet "
                     "wird, ob die Kopplungen Q_ij ueber die Diagonale hinaus "
                     "etwas beitragen. 15 Seeds, Lernrate je Arm, zwei "
                     "Datensaetze, Holm, TOST, Positivkontrolle."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "hidden": HIDDEN, "keep": KEEP,
                "max_samples": MAX_SAMPLES, "dense_epochs": DENSE_EPOCHS,
                "finetune_checkpoints": list(FT_CHECKPOINTS),
                "lr_grid": list(LR_GRID), "anneal_sweeps": ANNEAL_SWEEPS,
                "gram_batch": GRAM_BATCH, "torch_threads": threads,
                "margin": MARGIN, "test_metric": "balanced_accuracy",
                "primary": "qubo_anneal vs magnitude, 0 Epochen Nachtraining, mnist",
                "primary_dataset": PRIMARY_DATASET,
                "holm_family": list(REGISTERED),
                "resolution_check": "random muss signifikant schlechter sein als magnitude"},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, threads)
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                for row in _cell(arm, dataset, seed, lr_map[arm], threads):
                    rec.add(**row)
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
        # Nach JEDEM Datensatz sichern (Regel 5).
        print(f"    Zwischenstand nach {dataset} gesichert:")
        rec.save()
    return rec


# --------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------

def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    datasets = [str(d) for d in config["datasets"]]
    out: Dict[str, object] = {}

    def vals(arm: str, ds: str, cp: int, field: str = "balanced_accuracy") -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds
                       and int(r["finetune_epochs"]) == cp), key=lambda r: r["seed"])
        return np.array([r[field] for r in rows])

    n = len(vals(ARMS[0], datasets[0], 0))
    print()
    print(f"RESEARCH ID Q-006 | {HIDDEN} -> {KEEP} Neuronen, {n} Seeds")

    for ds in datasets:
        print()
        print(f"=== {ds} " + "=" * max(4, 60 - len(ds)))
        dense_ref = float(np.mean([r["dense_balanced_accuracy"] for r in records
                                   if r["dataset"] == ds]))
        print(f"  dichte Referenz ({HIDDEN} Neuronen): {dense_ref:.4f}")

        head = (f'  {"Arm":>12} ' + " ".join(f'{"ft=" + str(c):>9}' for c in FT_CHECKPOINTS)
                + f' {"Ising-E":>10}')
        print(); print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            cells = " ".join(f"{vals(arm, ds, c).mean():>9.4f}" for c in FT_CHECKPOINTS)
            e = vals(arm, ds, 0, "ising_energy")
            print(f'  {arm:>12} {cells} {e.mean():>10.4f}')

        # ---- vorregistrierte Familie, ohne Nachtraining ----
        comparisons = [("qubo_anneal", "magnitude", "PRIMAER"),
                       ("qubo_anneal", "diagonal", "P2"),
                       ("diagonal", "magnitude", "Daten statt Gewichte"),
                       ("magnitude", "random", "KONTROLLE")]
        rows_out = []
        for a, b, tag in comparisons:
            va, vb = vals(a, ds, 0), vals(b, ds, 0)
            p = float(stats.wilcoxon(va, vb).pvalue)
            eq = tost_paired(va, vb, MARGIN)
            rows_out.append([a, b, tag, float(va.mean() - vb.mean()), p, eq, None])
        fam = [r for r in rows_out if r[2] in REGISTERED]
        for row, p_h in zip(fam, holm([r[4] for r in fam])):
            row[6] = p_h

        print()
        print(f"  ohne Nachtraining (die Groesse, auf der die Auswahl wirkt):")
        for a, b, tag, delta, p, eq, p_h in rows_out:
            if p_h is None:
                state = ("Unterschied (unkorrigiert)" if p < 0.05 else
                         f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                         f"unentschieden (+-{eq['achieved']:.4f})")
                htxt = "--"
            else:
                state = ("Unterschied belegt" if p_h < 0.05 else
                         f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                         f"unentschieden (+-{eq['achieved']:.4f})")
                htxt = f"{p_h:.4f}"
            print(f'    {a:>12} vs {b:<12} {delta:>+8.4f}  p={p:.4f}  '
                  f'Holm={htxt:>6}  {state}   [{tag}]')
            out[f"{ds}|{a}_vs_{b}"] = {"delta": delta, "p": p, "p_holm": p_h,
                                       "p_tost": eq["p_tost"],
                                       "achieved": eq["achieved"],
                                       "registered": tag in REGISTERED}

        # ---- Ressourcenaussage ----
        print()
        print("  Ressourcenfrage: braucht ein besserer Start weniger Nachtraining?")
        for cp in FT_CHECKPOINTS:
            if cp == 0:
                continue
            va, vb = vals("qubo_anneal", ds, cp), vals("magnitude", ds, cp)
            p = float(stats.wilcoxon(va, vb).pvalue)
            print(f'    nach {cp:>2} Epochen: {va.mean() - vb.mean():>+8.4f}  p={p:.4f}')
            out[f"{ds}|ft{cp}|qubo_vs_magnitude"] = {
                "delta": float(va.mean() - vb.mean()), "p": p}

    # ---- Auflösungsprüfung ----
    ctrl = out.get(f"{PRIMARY_DATASET}|magnitude_vs_random", {})
    ctrl_ok = ctrl.get("delta", 0) > 0 and ctrl.get("p", 1) < 0.05

    # ---- Urteil ----
    prim = out.get(f"{PRIMARY_DATASET}|qubo_anneal_vs_magnitude", {})
    p2 = out.get(f"{PRIMARY_DATASET}|qubo_anneal_vs_diagonal", {})
    others = [k for k in out if k.endswith("|qubo_anneal_vs_magnitude")
              and "|ft" not in k and not k.startswith(PRIMARY_DATASET)]
    same_dir = all(out[k]["delta"] > 0 for k in others) if others else False

    print()
    print("=" * 82)
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    if (prim.get("delta", 0) > 0 and (prim.get("p_holm") or 1) < 0.05 and same_dir
            and p2.get("delta", 0) > 0 and (p2.get("p_holm") or 1) < 0.05):
        v = ("H1 -- das Ising-Verfahren schlaegt sowohl die Magnitude-Heuristik "
             "als auch die rein diagonale, datenbewusste Auswahl. Die Kopplungen "
             "Q_ij tragen selbst etwas bei; ein quantenabgeleitetes Verfahren "
             "macht das Netz bei identischem Parameterbudget messbar besser.")
    elif prim.get("delta", 0) > 0 and (prim.get("p_holm") or 1) < 0.05:
        v = ("H0 -- der Gewinn gegenueber der Heuristik ist belegt, aber "
             "`diagonal` zieht gleich. Dann kommt er daher, die DATEN zu "
             "benutzen, nicht aus der Ising-Struktur. Ehrlich als klassischer "
             "Befund zu berichten, nicht als quantenabgeleiteter.")
    else:
        v = (f"HU -- kein belegter Gewinn gegenueber der Magnitude-Heuristik. "
             f"Erreichte Schranke +-{prim.get('achieved', float('nan')):.4f}.")
    print(f"  {v}")

    if ctrl_ok:
        c = (f"Aufloesungspruefung BESTANDEN: magnitude schlaegt random "
             f"({ctrl.get('delta', float('nan')):+.4f}, p={ctrl.get('p', 1):.4f}). "
             f"Der Aufbau kann Maskenqualitaet aufloesen.")
    else:
        c = (f"Aufloesungspruefung NICHT BESTANDEN: magnitude trennt sich nicht "
             f"von einer ZUFAELLIGEN Maske ({ctrl.get('delta', float('nan')):+.4f}, "
             f"p={ctrl.get('p', 1):.4f}). Dann ist der Aufbau fuer die Frage blind "
             f"und kein Ergebnis oben ist interpretierbar -- derselbe Fehler wie "
             f"in B-004.")
    print()
    print(f"  {c}")
    print("=" * 82)

    out["verdict"] = v
    out["resolution_check"] = {"passed": bool(ctrl_ok), "text": c}
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001 -- bewusst breit
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print("  Rohdaten werden trotzdem gespeichert. Auswertung nachholen mit:")
        print('    python -c "from quantum_sim.qubo_pruning import analyse; '
              "analyse('Q-006')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
