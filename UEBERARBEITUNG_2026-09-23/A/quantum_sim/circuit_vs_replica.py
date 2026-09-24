from __future__ import annotations

"""RESEARCH ID: Q-010 -- Schlaegt die Quantenschaltung ihre eigene klassische Nachbildung?

WARUM DIESER LAUF, UND WARUM ER VORREGISTRIERT SEIN MUSS
Q-003 hatte berichtet, der klassische Nachbau schlage die Schaltung (-0,0147,
p=0,0078). Q-003-replication hat das am 5./6. August auf gespeicherten Rohdaten
nachgerechnet -- und das Vorzeichen dreht sich:

    mnist          +0,0126 zugunsten der Schaltung, 15/15 Seeds, p=0,0001
    fashion_mnist  +0,0051, 10/15, p=0,2293 (gleiche Richtung, nicht signifikant)

Das ist verlockend, aber es ist **kein Befund**: Der Vergleich stand nicht in
der vorregistrierten Testfamilie jenes Laufs, und er repliziert nicht auf dem
zweiten Datensatz. Aus genau diesen zwei Gruenden wurde am 3. August der Fund
`block_orthogonal` gegen `block_spectral` nicht als Ergebnis gefuehrt, obwohl er
ebenso verlockend war. Wer diese Regel einmal beugt, hat sie nicht mehr.

Dieser Lauf stellt die Frage deshalb sauber: vorher angemeldet, mit mehr Seeds,
und auf drei Datensaetzen.

WAS AUF DEM SPIEL STEHT
`classical_quad_free` ist nicht irgendeine Kontrolle, sondern die **faire
Nachbildung derselben Funktionsklasse**: normieren -> freie lineare Abbildung ->
quadrieren -> gruppenweise summieren. Genau die drei Schritte, die die Schaltung
leistet, nur klassisch und ohne Quantenformalismus. Haelt der Vorteil gegen
diese Kontrolle, dann traegt die **Quanten-Parametrisierung selbst** etwas bei --
2*n*L Winkel fuer eine Drehung im 2^n-dimensionalen Raum -- und nicht bloss die
Funktionsform. Das waere der erste Mechanismus im Projekt, der seine
Abgrenzung besteht.

Faellt er, ist die Aussage weiterhin gut, aber bescheidener: Die Funktionsform
wirkt, und die Schaltung realisiert sie nur besonders sparsam.

WAS GEGENUEBER Q-003-replication GEAENDERT WURDE, UND WARUM
  * **30 statt 15 Seeds.** Der Effekt auf Fashion-MNIST ist mit +0,0051 klein;
    bei 15 Seeds war er nicht aufloesbar. In diesem Projekt haben drei Befunde
    an einem Tag zwischen 10 und 30 Seeds die Richtung gewechselt -- 30 ist die
    Zahl, ab der hier Aussagen gehalten haben.
  * **Drei Datensaetze statt zwei.** Mit zwei ist "repliziert" eine
    Ja/Nein-Frage; mit drei laesst sich unterscheiden, ob ein Effekt allgemein
    ist oder an einem Datensatz haengt.
  * **Lernratengitter bis 0,3.** In Q-003-replication sass `hybrid` auch bei
    0,1 noch am Rand des Gitters; das Optimum lag moeglicherweise darueber, was
    den Vorteil unterschaetzt haette.
  * **Nur vier Arme.** `classical_quadratic` (orthogonale Matrix) war in beiden
    bisherigen Laeufen der schlechteste und instabilste Arm und beantwortet
    keine offene Frage mehr. `hybrid_no_entangle` ebenfalls nicht -- die
    Gleichwertigkeit ist gezeigt.

DIE ARME (Modelldefinitionen unveraendert aus `hybrid_final_control`)
  hybrid               Schaltung, 5 Lagen -> 80 Winkel
  hybrid_1layer        Schaltung, 1 Lage -> 16 Winkel. Im Nachtest lag auch sie
                       vor dem Nachbau; wenn 16 Parameter reichen, ist das die
                       staerkere Fassung derselben Aussage.
  classical_quad_free  die faire Nachbildung, 2.048 Parameter
  classical_norm_deep  67.848 Parameter -- dient hier als AUFLOESUNGSPRUEFUNG

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm und Datensatz auf
  eigenen Tuning-Seeds, gepaarter Wilcoxon-Test, Holm ueber die angemeldete
  Familie, TOST gegen +-0,005.

  PRIMAER  hybrid gegen classical_quad_free auf **mnist**.
  P2       hybrid_1layer gegen classical_quad_free auf **mnist**.
  Holm ueber genau diese beiden. Die beiden anderen Datensaetze sind
  WIEDERHOLUNG und entscheiden nichts allein -- dasselbe Kriterium wie in B-007.

  H1 : PRIMAER haelt nach Holm, UND beide Wiederholungen zeigen dieselbe
       Richtung -> die Quanten-Parametrisierung traegt ueber die Funktionsform
       hinaus etwas bei. Erster bestandener Abgrenzungstest des Projekts.
  H0 : TOST weist Gleichwertigkeit innerhalb +-0,005 nach -> Schaltung und
       Nachbildung sind gleich gut; die Schaltung erreicht das mit 16 bis 80
       statt 2.048 Parametern. Das bleibt ein Effizienz-, aber kein
       Mechanismusbefund.
  HU : weder noch -> erreichte Schranke berichten, nichts behaupten.

  AUFLOESUNGSPRUEFUNG (Regel 6): `hybrid` muss `classical_norm_deep` deutlich
  schlagen -- in Q-003-replication waren das +0,0932 bei Holm p=0,0001. Tritt
  das nicht ein, stimmt etwas am Aufbau nicht und kein Ergebnis oben ist
  interpretierbar.

  ABBRUCHREGEL: Keine. Der Lauf geht ueber alle drei Datensaetze, auch wenn der
  Primaertest frueh eindeutig aussieht.

Run: python -m quantum_sim.circuit_vs_replica
"""

from typing import Dict, List, Sequence

import numpy as np
import torch

from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .hybrid_replication import _cell

ARMS = ("hybrid", "hybrid_1layer", "classical_quad_free", "classical_norm_deep")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist", "digits")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(3301, 3331))          # 30 Seeds
TUNING_SEEDS = (3401, 3402)
LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["balanced_accuracy"]
                                    for s in TUNING_SEEDS]))
                 for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<20} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-010",
        description=("Vorregistriert: schlaegt die Quantenschaltung ihre faire "
                     "klassische Nachbildung (normieren + freie Matrix + "
                     "quadrieren)? 30 Seeds, drei Datensaetze, Lernrate je Arm, "
                     "Holm ueber die zwei angemeldeten Vergleiche, TOST."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "lr_grid": list(LR_GRID),
                "max_samples": MAX_SAMPLES, "epochs": EPOCHS,
                "batch_size": BATCH, "torch_threads": threads, "margin": MARGIN,
                "test_metric": "balanced_accuracy",
                "primary": "hybrid vs classical_quad_free auf mnist",
                "p2": "hybrid_1layer vs classical_quad_free auf mnist",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "hybrid muss classical_norm_deep deutlich schlagen",
                "anlass": ("nachtraeglich auf Q-003-replication gerechnet: mnist "
                           "+0,0126 (15/15, p=0,0001), fashion +0,0051 (p=0,23); "
                           "nicht vorregistriert, deshalb dieser Lauf")},
    )
    for dataset in datasets:
        print(f"\n  Lernratensuche fuer {dataset}")
        lr_map = sweep_lr(dataset, threads)
        rec.config[f"lr_map_{dataset}"] = dict(lr_map)
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_{dataset}"] = edge
        if edge:
            print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                rec.add(**_cell(arm, dataset, seed, lr_map[arm], threads,
                                max_samples=MAX_SAMPLES, epochs=EPOCHS, batch=BATCH))
            if i % 5 == 0 or i == len(seeds):
                print(f"    {dataset} seed {seed} fertig ({i}/{len(seeds)})")
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

    def vals(arm: str, ds: str) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    n = len(vals(ARMS[0], datasets[0]))
    print()
    print(f"RESEARCH ID Q-010 | {n} Seeds, {len(datasets)} Datensaetze")

    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 58 - len(ds)))
        head = f'  {"Arm":>20} {"balanced":>10} {"std":>8} {"Mitte":>9} {"lr":>6}'
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, ds)
            if not len(v):
                continue
            par = next(r["middle_parameters"] for r in records
                       if r["arm"] == arm and r["dataset"] == ds)
            lr = next(r["lr"] for r in records
                      if r["arm"] == arm and r["dataset"] == ds)
            print(f'  {arm:>20} {v.mean():>10.4f} {v.std(ddof=1):>8.4f} '
                  f'{par:>9,} {lr:>6}')

    # ---- vorregistrierte Familie: beide auf dem Primaerdatensatz ----
    ctrl_arm, replica = "classical_quad_free", "classical_quad_free"
    fam = [("hybrid", replica, "PRIMAER"), ("hybrid_1layer", replica, "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, PRIMARY_DATASET), vals(b, PRIMARY_DATASET)
        p = float(stats.wilcoxon(va, vb).pvalue)
        raw.append(p)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN), int((va > vb).sum()), len(va)))
    adj = holm(raw)

    print()
    print("=" * 90)
    print(f"VORREGISTRIERTE FAMILIE -- Holm ueber genau zwei Tests, {PRIMARY_DATASET}")
    for (a, b, tag, delta, p, eq, w, nn), p_h in zip(cells, adj):
        state = ("Vorteil belegt" if p_h < 0.05 and delta > 0 else
                 "NACHTEIL belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a:>14} vs {b:<20} {delta:>+8.4f}  {w:>2}/{nn}  '
              f'p={p:.4f}  Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"],
                    "wins": w, "n": nn}

    # ---- Wiederholung auf den uebrigen Datensaetzen ----
    print()
    print("WIEDERHOLUNG (sekundaer, entscheidet nichts allein)")
    repl: Dict[str, Dict[str, float]] = {}
    for ds in [d for d in datasets if d != PRIMARY_DATASET]:
        for a, b, tag in fam:
            va, vb = vals(a, ds), vals(b, ds)
            if not len(va):
                continue
            p = float(stats.wilcoxon(va, vb).pvalue)
            d0 = float(va.mean() - vb.mean())
            print(f'  {ds:>14} [{tag:>7}] {a:>14}: {d0:>+8.4f}  '
                  f'{int((va > vb).sum()):>2}/{len(va)}  p={p:.4f}')
            repl[f"{ds}|{tag}"] = {"delta": d0, "p": p}
    out["replication"] = repl

    # ---- Aufloesungspruefung ----
    hv, nv = vals("hybrid", PRIMARY_DATASET), vals("classical_norm_deep", PRIMARY_DATASET)
    c_p = float(stats.wilcoxon(hv, nv).pvalue)
    c_ok = hv.mean() > nv.mean() and c_p < 0.05
    print()
    if c_ok:
        ctxt = (f"Aufloesungspruefung BESTANDEN: hybrid schlaegt "
                f"classical_norm_deep ({hv.mean() - nv.mean():+.4f}, p={c_p:.4f}).")
    else:
        ctxt = (f"Aufloesungspruefung NICHT BESTANDEN (p={c_p:.4f}): der Aufbau "
                f"reproduziert nicht einmal den grossen, mehrfach belegten "
                f"Unterschied. Kein Ergebnis oben ist interpretierbar.")
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    # ---- Urteil ----
    prim = out["PRIMAER"]
    same_dir = all(v["delta"] > 0 for k, v in repl.items() if k.endswith("PRIMAER"))
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0 and same_dir:
        v = (f"H1 -- die Quantenschaltung schlaegt ihre faire klassische "
             f"Nachbildung ({prim['delta']:+.4f}, Holm p={prim['p_holm']:.4f}), "
             f"und die Wiederholungen zeigen dieselbe Richtung. Damit traegt die "
             f"Quanten-Parametrisierung ueber die Funktionsform hinaus etwas bei "
             f"-- der erste Mechanismus des Projekts, der seine Abgrenzung "
             f"besteht.")
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        v = (f"HU -- auf {PRIMARY_DATASET} belegt ({prim['delta']:+.4f}), aber die "
             f"Wiederholung zeigt nicht durchgehend dieselbe Richtung. Nach dem "
             f"vorregistrierten Kriterium reicht das nicht.")
    elif prim["p_tost"] < 0.05:
        v = (f"H0 -- Schaltung und Nachbildung sind gleichwertig innerhalb "
             f"+-{MARGIN}. Die Schaltung erreicht dasselbe mit 16 bis 80 statt "
             f"2.048 Parametern: ein Effizienzbefund, kein Mechanismusbefund.")
    else:
        v = (f"HU -- unentschieden; die Daten tragen erst ab "
             f"+-{prim['achieved']:.4f}.")
    print(f"BEWERTUNG: {v}")
    print("=" * 90)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.circuit_vs_replica '
              "import analyse; analyse('Q-010')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
