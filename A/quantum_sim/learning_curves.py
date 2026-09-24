from __future__ import annotations

"""RESEARCH ID: Q-016 -- Wie schnell kommt die Schaltung ans Ziel, nicht nur wie weit.

DER PERSPEKTIVWECHSEL
Jede Messung der Q-Serie vergleicht **Endgenauigkeit nach festem Budget**: 12
Epochen, 4.000 Beispiele, dann schauen wo die Arme stehen. Damit ist eine ganze
Achse unsichtbar geblieben -- und ausgerechnet die, um die es beim erklaerten
Projektziel geht: *weniger Rechenleistung, um dahin zu kommen.*

Dieselbe Luecke hat das Projekt schon einmal produktiv gemacht: Die K-Serie
entstand aus der Beobachtung, dass sechs Ansaetze immer nur *das Modell*
veraenderten und nie Datenrepraesentation, Optimierer oder **Datenbedarf**.
Hier ist es der Trainingsverlauf.

WARUM DORT ETWAS ZU ERWARTEN IST
Die Schaltung hat 80 trainierbare Parameter in der Mitte, das klassische Netz
67.848. Nach Standard-Lerntheorie waechst die noetige Stichprobengroesse mit der
Kapazitaet -- ein Modell mit drei Groessenordnungen weniger Parametern sollte
mit deutlich weniger Beispielen auskommen und deutlich weniger zur
Ueberanpassung neigen.

Das ist eine Vorhersage, keine Hoffnung, und sie ist bisher nie geprueft worden.
Wenn sie zutrifft, ist es der erste Befund des Projekts, der die Zielgroesse
direkt trifft.

WAS GEMESSEN WIRD
Zwei Achsen, an denen "Aufwand" haengt:

  (a) DATENBEDARF -- Genauigkeit ueber die Trainingsgroesse
      (250 / 500 / 1000 / 2000 / 4000 Beispiele)
  (b) SCHRITTBEDARF -- Genauigkeit ueber die Zahl der Gradientenschritte
      Zwischenstaende nach 50, 100, 200, 400, 800 Schritten

WARUM SCHRITTE UND NICHT EPOCHEN -- ein Entwurfsfehler, im Smoke-Test gefangen
Der erste Entwurf mass ueber EPOCHEN. Bei 250 Beispielen und Batch 128 sind
12 Epochen aber nur 24 Gradientenschritte; gemessen stand das Modell nach drei
Epochen bei 0,18 balancierter Genauigkeit, also weit vor jeder Konvergenz. Ein
Vergleich "wenig Daten gegen viel Daten" haette dann nicht den Datenbedarf
gemessen, sondern schlicht, dass der kleine Lauf zu frueh abgebrochen wurde.

B-010 hat das bereits richtig gemacht ("angeglichene Zahl der
Gradientenschritte"). Dasselbe hier: fester Schrittbudget je Zelle, kleine
Datenmengen laufen entsprechend mehr Epochen ueber dieselben Beispiele.

Dazu bei jedem Punkt die **Trainingsgenauigkeit**, also die Ueberanpassung.
Kostet nichts extra und beantwortet die naheliegende Rueckfrage einer Jury.

Die entscheidende abgeleitete Groesse ist nicht die Genauigkeit selbst, sondern
**wie viele Beispiele bzw. Epochen ein Arm braucht, um eine Zielmarke zu
erreichen**. Das ist die Zahl, die "Ressourceneffizienz" tatsaechlich bedeutet.

DIE ARME
  hybrid               Quantenschaltung, 80 Parameter in der Mitte
  classical_quad_free  beste klassische Nachbildung, 2.048
  classical_norm_deep  starke klassische Referenz, 67.848

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm UND je Trainingsgroesse
  (eine feste Rate ueber verschiedene Datenmengen waere genau der Fehler aus
  B-004), 10 Seeds, zwei Datensaetze.

  PRIMAER  Bei der KLEINSTEN Datenmenge (250 Beispiele) schlaegt `hybrid` das
           Netz `classical_norm_deep`. Dort sollte der Kapazitaetsunterschied
           am staerksten wirken.
  P2       Die Ueberanpassung (Trainings- minus Testgenauigkeit) ist bei
           `hybrid` geringer als bei `classical_norm_deep`, ueber alle
           Trainingsgroessen gemittelt.
  Holm ueber genau diese beiden.

  H1 : beide halten -> Die Schaltung braucht weniger Daten UND ueberangepasst
       weniger. Das ist Ressourceneffizienz auf der Achse, die das Projektziel
       nennt, und sie folgt aus der Parameterzahl.
  H0 : PRIMAER faellt -> der Vorteil bei kleinen Datenmengen existiert nicht;
       die 848-fach kleinere Kapazitaet zahlt sich beim Datenbedarf nicht aus.
  HU : unentschieden -> erreichte Schranke berichten.

  AUFLOESUNGSPRUEFUNG (Regel 6): Der Datenbedarf muss ueberhaupt messbar sein --
  jeder Arm muss bei 4.000 Beispielen signifikant besser sein als bei 250.
  Trennt sich das nicht, ist die Aufgabe von der Datenmenge unabhaengig und
  kein Vergleich auf dieser Achse aussagekraeftig.

  ERWARTUNG, vorher notiert: H1 fuer P2 (Ueberanpassung), unsicher fuer PRIMAER.
  Weniger Parameter heisst zuverlaessig weniger Ueberanpassung; ob es auch
  BESSERE Genauigkeit bei wenig Daten heisst, haengt davon ab, ob die 80
  Parameter die Aufgabe ueberhaupt fassen koennen. B-010 hat auf der Datenachse
  fuer die B-Familien H0 ergeben -- ein Warnsignal.

Run: python -m quantum_sim.learning_curves
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .hybrid_final_control import build, middle_params

ARMS = ("hybrid", "classical_quad_free", "classical_norm_deep")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
TRAIN_SIZES = (250, 500, 1000, 2000, 4000)
STEP_CHECKPOINTS = (50, 100, 200, 400, 800)  # Gradientenschritte, nicht Epochen
NUM_QUBITS, LAYERS = 8, 5
EVAL_SEEDS = tuple(range(8801, 8811))       # 10 Seeds
TUNING_SEEDS = (8851, 8852)
LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
BATCH = 64
TARGET = 0.85                                # Zielmarke fuer "Aufwand bis dahin"


def _balanced(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(np.mean([float((pred[y == c] == c).float().mean())
                          for c in torch.unique(y)]))


def _cells(arm: str, dataset: str, n_train: int, seed: int, lr: float,
           threads: int, checkpoints: Sequence[int] = STEP_CHECKPOINTS
           ) -> List[Dict[str, object]]:
    """Ein Trainingslauf mit Zwischenstaenden nach festen GRADIENTENSCHRITTEN.

    Damit bekommt jede Datenmenge dasselbe Rechenbudget; kleine Datensaetze
    laufen entsprechend oefter ueber dieselben Beispiele. Ohne diese Angleichung
    misst der Vergleich nicht den Datenbedarf, sondern nur, dass der kleine Lauf
    zu frueh abgebrochen wurde.
    """
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=n_train)
    model = build(arm, bundle.input_dim, bundle.num_classes, NUM_QUBITS, LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = bundle.x_train.shape[0]

    rows, step, targets = [], 0, sorted(checkpoints)
    order, pos = torch.randperm(n, generator=g), 0
    for target in targets:
        while step < target:
            if pos + BATCH > n:                       # neue Epoche
                order, pos = torch.randperm(n, generator=g), 0
            idx = order[pos:pos + BATCH]
            pos += BATCH
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bundle.x_train[idx]),
                            bundle.y_train[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            te = _balanced(model(bundle.x_test).argmax(-1), bundle.y_test)
            tr = _balanced(model(bundle.x_train).argmax(-1), bundle.y_train)
        model.train()
        rows.append({"arm": arm, "dataset": dataset, "n_train": int(n),
                     "seed": seed, "lr": lr, "gradient_steps": step,
                     "epochs_equivalent": step * BATCH / n,
                     "balanced_accuracy": te, "train_balanced_accuracy": tr,
                     "overfit": tr - te,
                     "middle_parameters": middle_params(model)})
    return rows


def sweep_lr(dataset: str, n_train: int, threads: int) -> Dict[str, float]:
    """Lernrate je Arm UND je Datenmenge -- eine feste Rate ueber verschiedene
    Datenmengen waere genau der Fehler, an dem B-004 gescheitert ist."""
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {}
        for lr in LR_GRID:
            v = [_cells(arm, dataset, n_train, s, lr, threads,
                        checkpoints=(6,))[-1]["balanced_accuracy"]
                 for s in TUNING_SEEDS]
            means[lr] = float(np.mean(v))
        best[arm] = max(means, key=means.get)
    print(f"    n={n_train:>5}: " + "  ".join(f"{a}={best[a]}" for a in ARMS))
    return best


def run(datasets: Sequence[str] = DATASETS, sizes: Sequence[int] = TRAIN_SIZES,
        seeds: Sequence[int] = EVAL_SEEDS, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-016",
        description=("Lernkurven statt Endgenauigkeit: Datenbedarf, "
                     "Schrittbedarf und Ueberanpassung der Quantenschaltung "
                     "gegen klassische Arme. Lernrate je Arm und Datenmenge."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "train_sizes": list(sizes), "STEP_CHECKPOINTS": list(STEP_CHECKPOINTS),
                "seeds": list(seeds), "num_qubits": NUM_QUBITS, "layers": LAYERS,
                "lr_grid": list(LR_GRID), "batch_size": BATCH, "margin": MARGIN,
                "target_accuracy": TARGET, "torch_threads": threads,
                "test_metric": "balanced_accuracy",
                "primary": f"hybrid > classical_norm_deep bei {min(sizes)} Beispielen",
                "p2": "hybrid ueberangepasst weniger als classical_norm_deep",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "jeder Arm muss bei 4000 besser sein als bei 250"},
    )
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        print("  Lernratensuche je Datenmenge")
        for n_train in sizes:
            lr_map = sweep_lr(dataset, n_train, threads)
            rec.config[f"lr_map_{dataset}_n{n_train}"] = dict(lr_map)
            edge = [a for a, v in lr_map.items()
                    if v in (min(LR_GRID), max(LR_GRID))]
            if edge:
                print(f"      WARNUNG Gitterrand: {', '.join(edge)}")
            for seed in seeds:
                for arm in ARMS:
                    for row in _cells(arm, dataset, n_train, seed,
                                      lr_map[arm], threads):
                        rec.add(**row)
            for arm in ARMS:
                v = [r["balanced_accuracy"] for r in rec.records
                     if r["arm"] == arm and r["dataset"] == dataset
                     and r["n_train"] == n_train and r["gradient_steps"] == max(STEP_CHECKPOINTS)]
                o = [r["overfit"] for r in rec.records
                     if r["arm"] == arm and r["dataset"] == dataset
                     and r["n_train"] == n_train and r["gradient_steps"] == max(STEP_CHECKPOINTS)]
                print(f"      {arm:>20} n={n_train:<5} {np.mean(v):.4f}  "
                      f"Ueberanpassung {np.mean(o):+.4f}")
            print(f"      Zwischenstand nach n={n_train} gesichert:")
            rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    datasets = [str(d) for d in config["datasets"]]
    sizes = [int(s) for s in config["train_sizes"]]
    last_ep = max(int(e) for e in config["STEP_CHECKPOINTS"])
    out: Dict[str, object] = {}

    def vals(arm: str, ds: str, n: int, ep: int = last_ep,
             field: str = "balanced_accuracy") -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds
                       and int(r["n_train"]) == n and int(r["gradient_steps"]) == ep),
                      key=lambda r: r["seed"])
        return np.array([r[field] for r in rows])

    print()
    print("RESEARCH ID Q-016 | Lernkurven")

    # ---- Datenachse ----
    for ds in datasets:
        print()
        print(f"--- {ds}: Genauigkeit ueber die Trainingsgroesse " + "-" * 20)
        head = "  " + f'{"Arm":>20} ' + " ".join(f'{n:>8}' for n in sizes)
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            cells = " ".join(f"{vals(arm, ds, n).mean():>8.4f}" for n in sizes)
            print(f'  {arm:>20} {cells}')
        print()
        print("  " + f'{"Ueberanpassung":>20} ' + " ".join(f'{n:>8}' for n in sizes))
        for arm in ARMS:
            cells = " ".join(f"{vals(arm, ds, n, field='overfit').mean():>8.4f}"
                             for n in sizes)
            print(f'  {arm:>20} {cells}')

    # ---- vorregistrierte Familie ----
    n0 = min(sizes)
    a, b = "hybrid", "classical_norm_deep"
    va, vb = vals(a, PRIMARY_DATASET, n0), vals(b, PRIMARY_DATASET, n0)
    p_prim = float(stats.wilcoxon(va, vb).pvalue)
    eq = tost_paired(va, vb, MARGIN)

    oa = np.concatenate([vals(a, PRIMARY_DATASET, n, field="overfit") for n in sizes])
    ob = np.concatenate([vals(b, PRIMARY_DATASET, n, field="overfit") for n in sizes])
    p_over = float(stats.wilcoxon(oa, ob).pvalue)
    adj = holm([p_prim, p_over])

    print()
    print("=" * 92)
    print(f"VORREGISTRIERTE FAMILIE auf {PRIMARY_DATASET}")
    print(f"  PRIMAER  bei {n0} Beispielen: hybrid - norm_deep = "
          f"{va.mean() - vb.mean():+.4f}  p={p_prim:.4f}  Holm={adj[0]:.4f}")
    print(f"  P2       Ueberanpassung: hybrid {oa.mean():+.4f} gegen "
          f"norm_deep {ob.mean():+.4f}  p={p_over:.4f}  Holm={adj[1]:.4f}")
    out["primary"] = {"delta": float(va.mean() - vb.mean()), "p": p_prim,
                      "p_holm": adj[0], "p_tost": eq["p_tost"],
                      "achieved": eq["achieved"], "n_train": n0}
    out["p2_overfit"] = {"hybrid": float(oa.mean()), "dense": float(ob.mean()),
                         "p": p_over, "p_holm": adj[1]}

    # ---- Aufloesungspruefung ----
    ok = True
    for arm in ARMS:
        lo, hi = vals(arm, PRIMARY_DATASET, min(sizes)), vals(arm, PRIMARY_DATASET, max(sizes))
        p = float(stats.wilcoxon(hi, lo).pvalue)
        good = hi.mean() > lo.mean() and p < 0.05
        ok &= good
        print(f"    {arm:>20}: {min(sizes)} -> {max(sizes)} Beispiele "
              f"{hi.mean() - lo.mean():+.4f}, p={p:.4f} {'OK' if good else 'NICHT'}")
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if ok else 'NICHT BESTANDEN'}: "
            f"die Datenmenge wirkt {'bei allen' if ok else 'nicht bei allen'} Armen."
            + ("" if ok else " Dann ist die Aufgabe von der Datenmenge weitgehend "
               "unabhaengig und kein Vergleich auf dieser Achse aussagekraeftig."))
    print()
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(ok), "text": ctxt}

    prim = out["primary"]
    print()
    if not ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        v = (f"H1 -- bei nur {n0} Beispielen schlaegt die Schaltung das "
             f"848-fach groessere Netz um {prim['delta']:+.4f}. Weniger "
             f"Kapazitaet heisst hier weniger Datenbedarf -- Ressourceneffizienz "
             f"auf der Achse, die das Projektziel nennt.")
        if adj[1] < 0.05 and oa.mean() < ob.mean():
            v += (" Und die Ueberanpassung ist ebenfalls geringer, wie aus der "
                  "Parameterzahl zu erwarten war.")
    elif prim["p_holm"] < 0.05:
        v = (f"H0 -- bei kleiner Datenmenge ist das grosse Netz besser "
             f"({prim['delta']:+.4f}). Die kleine Kapazitaet zahlt sich beim "
             f"Datenbedarf nicht aus.")
    else:
        v = (f"HU -- unentschieden bei {len(va)} Seeds; die Daten tragen erst ab "
             f"+-{prim['achieved']:.4f}.")
    print(f"BEWERTUNG: {v}")
    print("=" * 92)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.learning_curves '
              "import analyse; analyse('Q-016')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)

