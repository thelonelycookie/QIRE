from __future__ import annotations

"""RESEARCH ID: Q-007 -- Wo zahlt sich die Quanten-Parametrisierung aus?

WARUM DIESER LAUF
Q-003-replication hat auf zwei Datensaetzen gezeigt: die Schaltung mit 80
Parametern in der Mitte schlaegt ein klassisches Netz mit 67.848 deutlich
(+0,085 / +0,069, Holm p=0,0001), und sie ist von der besten klassischen
Nachbildung mit 2.048 Parametern nicht zu unterscheiden.

Zwei Datensaetze sind aber keine Verallgemeinerung. Die naheliegende Reaktion --
"auf moeglichst vielen Aufgaben testen und dann sagen, es sei ueberall besser" --
ist genau das Muster, an dem in diesem Projekt sieben Mechanismen gescheitert
sind: wer auf zwoelf Aufgaben testet und die Gewinner berichtet, findet bei
alpha=0,05 im Schnitt einen Treffer allein durch die Anzahl.

Dieser Lauf stellt deshalb die staerkere Frage. Nicht "ist es ueberall besser",
sondern **wovon haengt es ab**. `data_loader` liefert Datensaetze mit bekannter,
gezielt variierter Struktur -- Eingabedimension, Klassenzahl, Nichtlinearitaet,
Anteil irrelevanter Merkmale. Damit ist "wo wirkt der Mechanismus" messbar statt
behauptbar, und ein Vorteil, der ueber viele Aufgabentypen mit Holm-Korrektur
haelt, ist eine Aussage, die eine Kontrolle uebersteht.

DIE ARME (unveraendert aus `hybrid_final_control` / `quantum_layer`)
  hybrid              Quantenschaltung, 8 Qubits, 5 Lagen -> 80 Winkel
  hybrid_1layer       dieselbe Schaltung, 1 Lage -> 16 Winkel. Positivkontrolle
                      UND eigenstaendige Frage: in Q-003-replication war sie auf
                      Fashion-MNIST genauso gut wie die tiefe Variante.
  classical_quad_free normieren + freie Matrix + quadrieren, 2.048 Parameter.
                      Die faire Kontrolle: dieselbe Funktionsklasse, klassisch.
  classical_norm_deep Linear/Tanh mit Normierung, 67.848 Parameter. Die starke
                      klassische Referenz aus Q-002.

  `classical_quadratic` (orthogonale Matrix) faellt weg -- in beiden bisherigen
  Laeufen der schlechteste und instabilste Arm; er kostet Rechenzeit, ohne eine
  offene Frage zu beantworten.

WICHTIGE EINSCHRAENKUNG, DIE MITGEMESSEN WIRD
Der Encoder bildet die Eingabe auf 2^8 = 256 Amplituden ab. Bei MNIST (784
Merkmale) ist das eine Verdichtung, bei `wine` (13 Merkmale) eine Aufblaehung --
dort KOSTET die Architektur Parameter, statt welche zu sparen. Die Zahl "80
gegen 67.848" gilt fuer die MITTE; die Gesamtmodelle liegen bei rund 201.000
gegen 269.000. Beides wird je Zelle gespeichert, damit die Aussage nicht
groesser wird als die Messung.

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm und Datensatz auf
  eigenen Tuning-Seeds, 15 Seeds je Zelle.

  PRIMAER  hybrid gegen classical_norm_deep, **Holm ueber alle Datensaetze**.
           Kriterium: der Vorteil haelt auf der Mehrheit der Datensaetze nach
           Holm, und auf keinem dreht er signifikant um.
  P2       hybrid gegen classical_quad_free, ebenfalls Holm ueber alle
           Datensaetze. Das ist der eigentliche Abgrenzungstest: quad_free hat
           dieselbe Funktionsklasse ohne Quantenformalismus.

  H1 : PRIMAER haelt auf der Mehrheit -> die Parametereffizienz der
       Quanten-Parametrisierung ist kein Zwei-Datensatz-Zufall.
  H2 : zusaetzlich haelt P2 -> die Schaltung schlaegt auch ihre faire
       klassische Nachbildung. Das waere der erste echte Vorteil des
       Quantenanteils selbst.
  H0 : PRIMAER haelt nicht durchgehend -> der Vorteil ist datensatzabhaengig,
       und die Strukturanalyse sagt, wovon er abhaengt.

  AUFLOESUNGSPRUEFUNG (Regel 6) je Datensatz: `classical_norm_deep` hat das
  848-fache Parameterbudget von `hybrid`. Trennt sich auf einem Datensatz gar
  nichts zwischen irgendwelchen Armen, ist die Zelle fuer ihre Frage blind und
  wird aus der Hauptauswertung ausgeschlossen -- nicht stillschweigend, sondern
  ausgewiesen.

STRUKTURANALYSE (der eigentliche Ertrag)
  Der Vorteil je Datensatz wird gegen die Eigenschaften des Datensatzes
  korreliert: Eingabedimension, Klassenzahl, Trainingsgroesse. Spearman, und
  ausdruecklich als explorativ gekennzeichnet -- diese Korrelationen sind keine
  vorregistrierten Tests, sondern erzeugen die Hypothese fuer einen naechsten
  Lauf.

Run: python -m quantum_sim.breadth_battery
"""

from typing import Dict, List, Sequence

import numpy as np
import torch

from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .hybrid_replication import _cell

ARMS = ("hybrid", "hybrid_1layer", "classical_quad_free", "classical_norm_deep")
REGISTERED = ("PRIMAER", "P2")

# Bewusst gemischt: echte kleine Tabellendaten, echte Bilddaten, und
# synthetische mit gezielt variierter Struktur.
DATASETS = (
    "wine", "breast_cancer", "digits",              # echt, klein
    "mnist", "fashion_mnist",                       # echt, Bilder
    "synthetic_linear", "synthetic_nonlinear",      # Nichtlinearitaet
    "synthetic_high_dim", "synthetic_many_classes",  # Dimension / Klassenzahl
    "irrelevant_features_90", "synthetic_sparse_signal",  # redundante Merkmale
    "clustered_subtasks", "conditional_rules",      # Struktur in der Aufgabe
)

EVAL_SEEDS = tuple(range(7101, 7116))     # 15 Seeds
TUNING_SEEDS = (7201, 7202)
LR_GRID = (0.003, 0.01, 0.03, 0.1)
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    """Lernrate je Arm auf eigenen Tuning-Seeds (Regel 1)."""
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {}
        for lr in LR_GRID:
            vals = [_cell(arm, dataset, s, lr, threads,
                          max_samples=2000, epochs=6)["balanced_accuracy"]
                    for s in TUNING_SEEDS]
            means[lr] = float(np.mean(vals))
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>26} / {arm:<20} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-007",
        description=("Breitentest der Quanten-Parametrisierung ueber "
                     f"{len(datasets)} Aufgaben mit je eigener fairer Kontrolle. "
                     "Holm ueber die Datensaetze, Lernrate je Arm und Datensatz, "
                     "15 Seeds, Rohdaten nach jedem Datensatz gesichert."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "lr_grid": list(LR_GRID),
                "max_samples": MAX_SAMPLES, "epochs": EPOCHS,
                "batch_size": BATCH, "torch_threads": threads,
                "margin": MARGIN, "test_metric": "balanced_accuracy",
                "holm_family": list(REGISTERED),
                "primary": "hybrid vs classical_norm_deep, Holm ueber Datensaetze",
                "p2": "hybrid vs classical_quad_free, Holm ueber Datensaetze",
                "note": ("Encoder bildet auf 2^8=256 Amplituden ab; bei kleinen "
                         "Eingabedimensionen ist das eine Aufblaehung, keine "
                         "Verdichtung. middle_parameters und total_parameters "
                         "stehen je Zelle in den Rohdaten.")},
    )
    for di, dataset in enumerate(datasets, 1):
        print(f"\n[{di}/{len(datasets)}] {dataset} -- Lernratensuche")
        try:
            lr_map = sweep_lr(dataset, threads)
        except Exception as exc:                  # noqa: BLE001
            # Ein Datensatz, der nicht laedt, darf den ganzen Lauf nicht kosten.
            print(f"    UEBERSPRUNGEN ({exc!r})")
            rec.config.setdefault("skipped", []).append({"dataset": dataset,
                                                         "error": repr(exc)})
            continue
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_{dataset}"] = edge
        if edge:
            print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")

        info = load_dataset(dataset, seed=seeds[0], max_samples=MAX_SAMPLES)
        rec.config[f"shape_{dataset}"] = {"input_dim": int(info.input_dim),
                                          "num_classes": int(info.num_classes),
                                          "n_train": int(info.x_train.shape[0])}
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                row = _cell(arm, dataset, seed, lr_map[arm], threads,
                            max_samples=MAX_SAMPLES, epochs=EPOCHS, batch=BATCH)
                row["input_dim"] = int(info.input_dim)
                row["num_classes"] = int(info.num_classes)
                rec.add(**row)
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
        # Nach JEDEM Datensatz sichern (Regel 5).
        print(f"    Zwischenstand nach {dataset} gesichert:")
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

    datasets = sorted({str(r["dataset"]) for r in records})
    out: Dict[str, object] = {}

    def vals(arm: str, ds: str) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print(f"RESEARCH ID Q-007 | {len(datasets)} Aufgaben, "
          f"{len(vals(ARMS[0], datasets[0]))} Seeds")

    # ---- Uebersicht je Datensatz ----
    head = (f'  {"Datensatz":>24} {"dim":>5} {"cls":>4} ' +
            " ".join(f'{a[:14]:>14}' for a in ARMS))
    print(); print(head); print("  " + "-" * (len(head) - 2))
    shapes: Dict[str, Dict[str, int]] = {}
    for ds in datasets:
        sh = config.get(f"shape_{ds}", {})
        shapes[ds] = sh
        cells = " ".join(f"{vals(a, ds).mean():>14.4f}" for a in ARMS)
        print(f'  {ds:>24} {sh.get("input_dim", 0):>5} '
              f'{sh.get("num_classes", 0):>4} {cells}')

    # ---- die beiden vorregistrierten Familien, Holm ueber Datensaetze ----
    families = [("hybrid", "classical_norm_deep", "PRIMAER"),
                ("hybrid", "classical_quad_free", "P2")]
    for a, b, tag in families:
        raw, cells = [], []
        for ds in datasets:
            va, vb = vals(a, ds), vals(b, ds)
            if len(va) != len(vb) or len(va) == 0:
                continue
            p = float(stats.wilcoxon(va, vb).pvalue)
            raw.append(p)
            cells.append((ds, float(va.mean() - vb.mean()), p,
                          tost_paired(va, vb, MARGIN), int((va > vb).sum()), len(va)))
        adj = holm(raw)
        print()
        print("=" * 96)
        print(f"{tag}: {a} gegen {b}  --  Holm ueber {len(cells)} Datensaetze")
        h = (f'  {"Datensatz":>24} {"Differenz":>10} {"Siege":>8} {"p":>8} '
             f'{"Holm":>8}  Lesart')
        print(h); print("  " + "-" * (len(h) - 2))
        wins = losses = 0
        for (ds, delta, p, eq, w, n), p_h in zip(cells, adj):
            if p_h < 0.05 and delta > 0:
                reading, wins = "Vorteil belegt", wins + 1
            elif p_h < 0.05 and delta < 0:
                reading, losses = "NACHTEIL belegt", losses + 1
            elif eq["p_tost"] < 0.05:
                reading = f"gleichwertig (+-{MARGIN})"
            else:
                reading = f"unentschieden (+-{eq['achieved']:.4f})"
            print(f'  {ds:>24} {delta:>+10.4f} {w:>5}/{n:<2} {p:>8.4f} '
                  f'{p_h:>8.4f}  {reading}')
            out[f"{tag}|{ds}"] = {"delta": delta, "p": p, "p_holm": p_h,
                                  "p_tost": eq["p_tost"], "achieved": eq["achieved"]}
        print(f'  -> Vorteil belegt auf {wins}/{len(cells)}, '
              f'Nachteil belegt auf {losses}/{len(cells)}')
        out[f"{tag}_summary"] = {"wins": wins, "losses": losses, "n": len(cells)}

    # ---- Strukturanalyse: wovon haengt der Vorteil ab? ----
    print()
    print("=" * 96)
    print("STRUKTURANALYSE -- explorativ, NICHT vorregistriert, erzeugt nur Hypothesen")
    deltas, dims, clss = [], [], []
    for ds in datasets:
        key = f"PRIMAER|{ds}"
        if key not in out:
            continue
        deltas.append(out[key]["delta"])
        dims.append(shapes.get(ds, {}).get("input_dim", 0))
        clss.append(shapes.get(ds, {}).get("num_classes", 0))
    if len(deltas) >= 4:
        for name, xs in (("Eingabedimension", dims), ("Klassenzahl", clss)):
            rho, p = stats.spearmanr(xs, deltas)
            print(f'  Vorteil gegen {name:>18}: rho={rho:>+6.3f}  p={p:.4f}')
            out[f"struct|{name}"] = {"rho": float(rho), "p": float(p)}
        print()
        print("  Lesart: ein positiver rho bei der Eingabedimension hiesse, dass")
        print("  sich die Amplituden-Kodierung erst lohnt, wenn sie wirklich")
        print("  verdichtet -- bei 13 Merkmalen auf 256 Amplituden blaeht sie auf.")

    # ---- Urteil ----
    prim = out.get("PRIMAER_summary", {})
    p2 = out.get("P2_summary", {})
    n = prim.get("n", 0) or 1
    print()
    print("=" * 96)
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    if prim.get("wins", 0) > n / 2 and prim.get("losses", 0) == 0:
        if p2.get("wins", 0) > n / 2 and p2.get("losses", 0) == 0:
            v = (f"H2 -- die Schaltung schlaegt auf {prim['wins']}/{n} Aufgaben das "
                 f"grosse klassische Netz UND auf {p2['wins']}/{n} ihre eigene faire "
                 f"klassische Nachbildung. Das waere ein Vorteil des Quantenanteils "
                 f"selbst, nicht nur der Funktionsklasse.")
        else:
            v = (f"H1 -- Parametereffizienz belegt auf {prim['wins']}/{n} Aufgaben "
                 f"gegen 67.848 Parameter, aber gegen die faire klassische "
                 f"Nachbildung nur auf {p2.get('wins', 0)}/{n}. Der Vorteil liegt "
                 f"in der FUNKTIONSKLASSE (normieren, drehen, quadrieren), nicht "
                 f"im Quantenformalismus -- so zu berichten.")
    else:
        v = (f"H0 -- der Vorteil ist datensatzabhaengig: belegt auf "
             f"{prim.get('wins', 0)}/{n}, widerlegt auf {prim.get('losses', 0)}/{n}. "
             f"Die Strukturanalyse oben sagt, wovon er abhaengt; das ist die "
             f"eigentliche Aussage dieses Laufs.")
    print(f"  {v}")
    print("=" * 96)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.breadth_battery '
              "import analyse; analyse('Q-007')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
