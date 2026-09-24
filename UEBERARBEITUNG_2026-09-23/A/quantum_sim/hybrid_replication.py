from __future__ import annotations

"""RESEARCH ID: Q-003-replication -- Q-001 bis Q-003 unter dem neuen Protokoll.

WARUM DIESER LAUF NOETIG IST
Die Kernaussagen von Projekt A stammen aus Q-001 bis Q-003:

    Die Quantenschaltung mit 80 Parametern schlaegt ein klassisches Netz mit
    67.848 Parametern (+0,0707, p=0,0078) -- und der klassische Nachbau aus
    normieren + orthogonaler Drehung + quadratischer Auslese schlaegt
    wiederum die Schaltung.

Der Audit vom 3. August hat gezeigt: fuer keinen dieser Laeufe existieren
Rohdaten. Sie sind nur durch einen kompletten Neulauf belegbar. Ausserdem
haben sie genau das Profil, an dem in diesem Projekt inzwischen VIER Befunde
gescheitert sind:

  * 8 Seeds -- bei einem gepaarten Wilcoxon-Test ist p=0,0078 dort das
    rechnerische Minimum, es gibt also keinen Spielraum nach unten
  * feste Lernrate 0,01 fuer alle Arme, obwohl B-004-lr gezeigt hat, dass ein
    fester Wert einzelne Arme systematisch benachteiligt
  * kein Aequivalenztest -- "nicht signifikant" wurde als "gleich" gelesen
  * nur ein Datensatz
  * plain accuracy statt balanced_accuracy, anders als im ganzen Rest

WAS HIER ANDERS IST
  15 Seeds, Lernrate je Arm auf eigenen Tuning-Seeds gesucht, zwei Datensaetze,
  Holm-Korrektur, TOST-Aequivalenztest, und alles wird gespeichert -- nach JEDEM
  Datensatz, nicht erst am Ende.

  Getestet wird auf BALANCED accuracy. Die Grenze MARGIN aus `equivalence` ist
  ausdruecklich in balancierter Genauigkeit definiert und gegen deren
  Seed-Streuung begruendet; auf plain accuracy angewandt waere sie eine Grenze
  fuer eine andere Groesse als die, fuer die sie hergeleitet wurde.

Die Modelldefinitionen werden UNVERAENDERT aus `hybrid_final_control` und
`quantum_layer` uebernommen. Nur das Protokoll aendert sich -- sonst waere
unklar, ob ein abweichendes Ergebnis am Protokoll oder am Modell liegt.

DIE ARME
  hybrid              Quantenschaltung mit Verschraenkung (Q-001)
  hybrid_no_entangle  dieselbe Schaltung ohne CNOT (Q-001: traegt
                      Verschraenkung etwas bei?)
  hybrid_1layer       POSITIVKONTROLLE: dieselbe Schaltung mit 1 statt 5 Lagen,
                      also 16 statt 80 Winkeln. Dieser Unterschied MUSS sich
                      zeigen; tut er es nicht, kann der Aufbau die Frage nicht
                      aufloesen und kein Nullbefund daraus ist etwas wert.
  classical_quadratic normieren + orthogonale Drehung + quadrieren (Q-003)
  classical_quad_free wie oben, Matrix frei -- isoliert die Normerhaltung
  classical_norm_deep Linear/Tanh mit Normierung, 67.848 Parameter (Q-002)

VORREGISTRIERUNG
  PRIMAER  hybrid gegen classical_norm_deep auf MNIST -- das ist die Aussage,
           die in der Kurzfassung stehen soll.
  P2       classical_quadratic gegen hybrid -- laesst sich die Schaltung
           klassisch nachbauen?
  Holm ueber diese beiden -- und nur ueber diese beiden. Alles andere ist
  Einordnung und wird unkorrigiert, als deskriptiv gekennzeichnet, berichtet.

  KONTROLLE hybrid gegen hybrid_1layer. Kein Test einer Hypothese, sondern die
           Pruefung, ob der Aufbau ueberhaupt aufloest (Regel 6). Faellt sie
           durch, wird jeder Nullbefund oben als "nicht aufloesbar" berichtet,
           nicht als "gleichwertig".

  H1 : hybrid schlaegt classical_norm_deep auf dem Primaerdatensatz nach Holm,
       und die Wiederholung auf dem zweiten Datensatz zeigt dieselbe Richtung
       -> die Kernaussage von A haelt. (Dasselbe Kriterium wie in B-007: der
       zweite Datensatz ist Wiederholung, er entscheidet nichts allein.)
  H0 : kein signifikanter Unterschied, und TOST weist Gleichwertigkeit nach
       -> die Kernaussage faellt und muss zurueckgezogen werden.
  HU : weder noch -> die Datenlage traegt die Aussage nicht; erreichte
       Schranke wird berichtet.

Run: python -m quantum_sim.hybrid_replication
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.metrics import evaluate
from qire_ai.result_store import Recorder

from .hybrid_final_control import build as build_arm
from .hybrid_final_control import middle_params, train
from .quantum_layer import HybridModel

ARMS = ("hybrid", "hybrid_no_entangle", "hybrid_1layer", "classical_quadratic",
        "classical_quad_free", "classical_norm_deep")
CONTROL_LAYERS = 1          # Lagen im Positivkontroll-Arm (statt NUM_LAYERS)
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
# Die vorregistrierte Testfamilie -- nur ueber diese wird Holm gerechnet.
REGISTERED = ("PRIMAER", "P2")
EVAL_SEEDS = tuple(range(2301, 2316))       # 15 Seeds
TUNING_SEEDS = (2401, 2402)
# 0,1 ergaenzt nach einem Fehlstart am 5. August: im Gitter (0,003 / 0,01 / 0,03)
# waehlten FUENF der sechs Arme den Randwert 0,03 -- das Optimum lag also
# moeglicherweise ausserhalb. Nur `classical_norm_deep` lag mit 0,01 im Inneren,
# womit ausgerechnet der Primaervergleich asymmetrisch geworden waere.
LR_GRID = (0.003, 0.01, 0.03, 0.1)
NUM_QUBITS = 8
NUM_LAYERS = 5
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


def build(arm: str, input_dim: int, num_classes: int, num_qubits: int,
          num_layers: int):
    """Wie `hybrid_final_control.build`, plus Arm ohne Verschraenkung und
    Positivkontrolle. Beide nutzen dieselbe `HybridModel`-Klasse -- es aendert
    sich nur ein dokumentierter Hyperparameter, kein Modellcode."""
    if arm == "hybrid_no_entangle":
        return HybridModel(input_dim, num_qubits, num_layers, num_classes,
                           entangle=False)
    if arm == "hybrid_1layer":
        return HybridModel(input_dim, num_qubits, CONTROL_LAYERS, num_classes,
                           entangle=True)
    return build_arm(arm, input_dim, num_classes, num_qubits, num_layers)


def _cell(arm: str, dataset: str, seed: int, lr: float, threads: int,
          max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS,
          batch: int = BATCH) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = build(arm, bundle.input_dim, bundle.num_classes, NUM_QUBITS, NUM_LAYERS)
    train(model, bundle.x_train, bundle.y_train, epochs, lr, batch, seed)
    model.eval()
    with torch.no_grad():
        logits = model(bundle.x_test)
        pred = logits.argmax(-1)
        acc = float((pred == bundle.y_test).float().mean())
        # balanced accuracy zusaetzlich, damit die Zahl mit dem Rest des
        # Projekts vergleichbar ist -- Q-003 hat nur plain accuracy berichtet
        classes = torch.unique(bundle.y_test)
        per = [float((pred[bundle.y_test == c] == c).float().mean()) for c in classes]
    return {"arm": arm, "dataset": dataset, "seed": seed, "lr": lr,
            "accuracy": acc, "balanced_accuracy": float(np.mean(per)),
            "middle_parameters": middle_params(model),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["accuracy"]
                                    for s in TUNING_SEEDS]))
                 for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<20} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-003-replication",
        description=("Q-001 bis Q-003 unter dem Protokoll vom August 2026: "
                     "15 Seeds, Lernrate je Arm, zwei Datensaetze, Holm, TOST, "
                     "Rohdaten gespeichert. Modelle unveraendert uebernommen."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "num_qubits": NUM_QUBITS,
                "num_layers": NUM_LAYERS, "max_samples": MAX_SAMPLES,
                "epochs": EPOCHS, "batch_size": BATCH, "lr_grid": list(LR_GRID),
                "torch_threads": threads, "margin": MARGIN,
                "primary": "hybrid vs classical_norm_deep auf mnist",
                "primary_dataset": PRIMARY_DATASET,
                # Vorregistrierung, mit den Rohdaten zusammen abgelegt, damit
                # sie spaeter nicht aus dem Code rekonstruiert werden muss
                "test_metric": "balanced_accuracy",
                "holm_family": list(REGISTERED),
                "positive_control": (f"hybrid vs hybrid_1layer "
                                     f"({NUM_LAYERS} gegen {CONTROL_LAYERS} Lagen)"),
                "original": ("Q-003 lief mit 8 Seeds, fester lr 0,01, einem "
                             "Datensatz und ohne Aequivalenztest")},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, threads)
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)
        # Randlagen melden: waehlt ein Arm den kleinsten oder groessten Wert des
        # Gitters, liegt sein Optimum moeglicherweise ausserhalb und der
        # Vergleich ist zu seinen Ungunsten verzerrt. Genau daran ist der erste
        # Anlauf dieses Laufs gescheitert.
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_{dataset}"] = edge
        if edge:
            print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")
            print(f"    Das Optimum kann ausserhalb {LR_GRID} liegen.")
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                rec.add(**_cell(arm, dataset, seed, lr_map[arm], threads))
            print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
        # Nach JEDEM Datensatz sichern (Regel 5). B-007 hat an genau dieser
        # Stelle 200 fertige Zellen verloren, weil erst am Ende gespeichert
        # wurde und der Lauf in Fashion-MNIST abbrach. Mehrfaches Speichern
        # legt je einen Zwischenstand ab; die letzte Datei enthaelt alles.
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
    datasets = [str(d) for d in config["datasets"]]
    out: Dict[str, object] = {}

    # Alle Tests laufen auf balancierter Genauigkeit -- das ist die Groesse, in
    # der MARGIN definiert und begruendet ist, und die Groesse, auf der der
    # Rest des Projekts (equivalence, B-007, B-008) rechnet. Q-003 hatte plain
    # accuracy berichtet; das steht oben als einer der fuenf Maengel.
    METRIC = "balanced_accuracy"

    def vals(arm, ds, field=METRIC):
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r[field] for r in rows])

    n = len(vals(ARMS[0], datasets[0]))
    print()
    print(f"RESEARCH ID Q-003-replication | {NUM_QUBITS} Qubits, {n} Seeds")

    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 56 - len(ds)))
        head = (f'  {"Arm":>20} {"accuracy":>9} {"balanced":>9} {"std":>7} '
                f'{"Params":>9} {"lr":>6}')
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v, b = vals(arm, ds, "accuracy"), vals(arm, ds, METRIC)
            par = next((r["middle_parameters"] for r in records
                        if r["arm"] == arm and r["dataset"] == ds), 0)
            lr = next((r["lr"] for r in records
                       if r["arm"] == arm and r["dataset"] == ds), float("nan"))
            # std der GETESTETEN Groesse, nicht der nur mitberichteten
            print(f'  {arm:>20} {v.mean():>9.4f} {b.mean():>9.4f} '
                  f'{b.std(ddof=1):>7.4f} {par:>9,} {lr:>6}')

        comparisons = [("hybrid", "classical_norm_deep", "PRIMAER"),
                       ("classical_quadratic", "hybrid", "P2"),
                       ("hybrid", "hybrid_1layer", "KONTROLLE"),
                       ("hybrid", "hybrid_no_entangle", "Verschraenkung"),
                       ("classical_quadratic", "classical_quad_free", "Normerhaltung")]
        rows_out = []
        for a, b, tag in comparisons:
            va, vb = vals(a, ds), vals(b, ds)
            p = float(stats.wilcoxon(va, vb).pvalue)
            eq = tost_paired(va, vb, MARGIN)
            rows_out.append([a, b, tag, float(va.mean() - vb.mean()), p, eq, None])

        # Holm NUR ueber die vorregistrierte Familie (PRIMAER, P2). Die beiden
        # anderen Vergleiche sind laut Vorregistrierung Einordnung; sie
        # mitzukorrigieren wuerde die primaere Aussage schwerer machen, als
        # angemeldet war, und wird darum nicht getan. Sie werden unkorrigiert
        # und als deskriptiv gekennzeichnet berichtet.
        family = [row for row in rows_out if row[2] in REGISTERED]
        for row, p_h in zip(family, holm([row[4] for row in family])):
            row[6] = p_h

        print()
        for a, b, tag, delta, p, eq, p_h in rows_out:
            if p_h is None:                       # deskriptiv, nicht korrigiert
                state = ("Unterschied (unkorrigiert)" if p < 0.05 else
                         f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                         f"unentschieden (+-{eq['achieved']:.4f})")
                holm_txt = "--"
            else:
                state = ("Unterschied belegt" if p_h < 0.05 else
                         f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                         f"unentschieden (+-{eq['achieved']:.4f})")
                holm_txt = f"{p_h:.4f}"
            print(f'  {a:>20} vs {b:<20} {delta:>+8.4f}  p={p:.4f}  '
                  f'Holm={holm_txt:>6}  {state}   [{tag}]')
            out[f"{ds}|{a}_vs_{b}"] = {"delta": delta, "p": p, "p_holm": p_h,
                                       "p_tost": eq["p_tost"],
                                       "achieved": eq["achieved"],
                                       "registered": tag in REGISTERED,
                                       "metric": METRIC}

    print()
    print("=" * 82)
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    prim = out.get(f"{PRIMARY_DATASET}|hybrid_vs_classical_norm_deep", {})
    others = [k for k in out if k.endswith("|hybrid_vs_classical_norm_deep")
              and not k.startswith(PRIMARY_DATASET)]
    same_dir = all(out[k]["delta"] > 0 for k in others) if others else False

    # Positivkontrolle (Regel 6): loest dieser Aufbau einen Unterschied, den es
    # sicher gibt, ueberhaupt auf? 5 Lagen gegen 1 Lage, sonst alles gleich.
    # Faellt die Kontrolle durch, ist ein Nullbefund oben nicht als
    # "kein Unterschied" lesbar, sondern als "nicht aufloesbar" -- genau der
    # Fehler, an dem in B-004 der ganze fruehere Aufbau gescheitert ist.
    ctrl = out.get(f"{PRIMARY_DATASET}|hybrid_vs_hybrid_1layer", {})
    ctrl_delta, ctrl_p = ctrl.get("delta", float("nan")), ctrl.get("p", 1.0)
    ctrl_ok = ctrl_delta > 0 and ctrl_p < 0.05
    if prim.get("delta", 0) > 0 and prim.get("p_holm", 1) < 0.05 and same_dir:
        v = ("H1 -- die Kernaussage von Projekt A haelt: die Quantenschaltung "
             "schlaegt das klassische Netz auch unter dem strengeren Protokoll, "
             "auf beiden Datensaetzen.")
    elif prim.get("p_tost", 1) < 0.05:
        v = ("H0 -- die Kernaussage FAELLT. Der Aequivalenztest weist "
             f"Gleichwertigkeit innerhalb +-{MARGIN} nach; der berichtete Vorteil "
             "von +0,0707 aus Q-003 haelt dem Protokoll nicht stand.")
    else:
        v = (f"HU -- unentschieden. Weder Vorteil noch Gleichwertigkeit belegt; "
             f"die Daten tragen erst ab +-{prim.get('achieved', float('nan')):.4f}. "
             f"Die Aussage darf so nicht in die Kurzfassung.")
    print(f"  {v}")

    if ctrl_ok:
        c = (f"Positivkontrolle BESTANDEN: {NUM_LAYERS} Lagen schlagen "
             f"{CONTROL_LAYERS} Lage ({ctrl_delta:+.4f}, p={ctrl_p:.4f}). Der "
             f"Aufbau loest Unterschiede dieser Groessenordnung auf, das Urteil "
             f"oben ist also belastbar.")
    else:
        c = (f"Positivkontrolle NICHT BESTANDEN: {NUM_LAYERS} Lagen trennen sich "
             f"nicht von {CONTROL_LAYERS} Lage ({ctrl_delta:+.4f}, p={ctrl_p:.4f}). "
             f"Ein Nullbefund oben darf NICHT als Gleichwertigkeit gelesen werden "
             f"-- der Aufbau ist fuer seine eigene Frage moeglicherweise blind. "
             f"Das ist derselbe Fehler wie in B-004 und muss so berichtet werden.")
    print()
    print(f"  {c}")
    print("=" * 82)
    out["verdict"] = v
    out["positive_control"] = {"passed": bool(ctrl_ok), "delta": ctrl_delta,
                               "p": ctrl_p, "text": c}
    return out


if __name__ == "__main__":
    recorder = run()
    # Die Auswertung darf die Rohdaten nicht mitreissen: `save(analysis=analyse(...))`
    # wertet erst aus und speichert danach -- ein Fehler in analyse() haette
    # zwei Stunden Rechenzeit gekostet. Die Zwischenstaende aus run() liegen
    # zwar schon auf der Platte, die Auswertung wird hier aber trotzdem
    # abgefangen, damit die letzte Datei vollstaendig ist.
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001 -- bewusst breit
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print("  Die Rohdaten werden trotzdem gespeichert. Auswertung nachholen mit:")
        print('    python -c "from quantum_sim.hybrid_replication import analyse; '
              "analyse('Q-003-replication')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
