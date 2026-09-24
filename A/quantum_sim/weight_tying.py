from __future__ import annotations

"""RESEARCH ID: Q-014 -- Kostet die quantenabgeleitete Gewichtsteilung etwas?

DIE FRAGE NACH DEM ALLEINSTELLUNGSMERKMAL
Dass sich die Wirkung der Schaltung klassisch erklaeren laesst, ist kein
Problem -- eine Entdeckung darf klassisch erklaerbar sein. Ein Alleinstellungs-
merkmal braucht aber mehr als Herkunft. Geschwindigkeit und Groesse allein
reichen nicht: Sobald die Butterfly-Struktur beschrieben ist, kann sie jeder
klassisch nachbauen, ganz ohne Quantenmechanik.

Der Anspruch muss dort liegen, wo die **Quantenherleitung etwas gibt, das man
sonst nachweislich nicht haette.** Genau ein solcher Kandidat steckt in der
Architektur, und er ist bisher nie geprueft worden.

DIE GEWICHTSTEILUNG
Ein Ein-Qubit-Gatter wirkt physikalisch auf **ein Qubit** -- gleichgueltig, wie
gross der Zustandsraum ist. Uebersetzt in die Butterfly-Struktur heisst das:
derselbe Winkel wirkt auf **alle d/2 Paare einer Ebene**. Bei 8 Qubits sind das
80 statt 10.240 Winkel, Faktor 128; bei 12 Qubits 120 statt 245.760.

Klassisch wuerde niemand so bauen. Ein Butterfly-Netz bekommt ueblicherweise je
Paar einen eigenen Winkel -- es gibt keinen klassischen Grund, sie zu koppeln.
Der Grund kommt aus der Physik.

WAS BISHER FEHLT, UND WARUM ES DIE ENTSCHEIDENDE FRAGE IST
Bisher ist nur gezaehlt, dass die Teilung Parameter spart. Das ist trivial --
jede Einschraenkung spart Parameter. Die eigentliche Frage lautet:

    Kostet die Teilung Qualitaet?

  * Kostet sie nichts, dann ist sie eine **kostenlose Kompression um Faktor
    128**, auf die klassisch niemand kommt. Das waere ein Alleinstellungsmerkmal.
  * Kostet sie etwas, weiss man genau wie viel, und die Aussage wird ehrlich
    bezifferbar statt vage.

Und die schaerfere Fassung: Ist bei **gleichem Parameterbudget** ein geteiltes
Netz mit vielen Lagen besser als ein ungeteiltes mit wenigen? Dann waere die
Beschraenkung nicht bloss sparsam, sondern eine *bessere Verwendung* von
Parametern.

DIE ARME (8 Qubits, 256 Amplituden; Architektur wie im Hybridmodell)
  tied_L1          16 Parameter    geteilt, 1 Lage
  tied_L5          80 Parameter    geteilt, 5 Lagen -- die Schaltung selbst
  tied_L80      1.280 Parameter    geteilt, 80 Lagen
  untied_L1     2.048 Parameter    ungeteilt, 1 Lage
  untied_L5    10.240 Parameter    ungeteilt, 5 Lagen
  tied_L5_frozen   0 trainierbar   geteilt, 5 Lagen, Winkel EINGEFROREN

DIE ZWEI VERGLEICHE, UM DIE ES GEHT
  A) GLEICHE RECHENZEIT: tied_L5 gegen untied_L5. Dieselbe Tiefe, dieselbe Zahl
     an Operationen, dieselbe Struktur -- nur die Teilung unterscheidet sich.
     80 gegen 10.240 Parameter.
  B) GLEICHES BUDGET: tied_L80 (1.280) gegen untied_L1 (2.048). Hier hat das
     geteilte Netz sogar WENIGER Parameter, dafuer mehr Tiefe.

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm, 15 Seeds, zwei
  Datensaetze.

  PRIMAER  tied_L5 gegen untied_L5: TOST auf Gleichwertigkeit innerhalb
           +-0,005. Gleichwertig hiesse: die Teilung kostet nichts.
  P2       tied_L80 gegen untied_L1 bei angeglichenem Budget.
  Holm ueber genau diese beiden.

  H1 : PRIMAER gleichwertig -> die Teilung ist eine kostenlose Kompression um
       Faktor 128. Zusammen mit P2 (geteilt besser bei gleichem Budget) waere
       das der gesuchte eigenstaendige Beitrag der Quantenherleitung.
  H1+: tied_L5 ist signifikant BESSER als untied_L5 -> die Teilung kostet
       nicht nur nichts, sie nuetzt. Da beide Familien bei gleicher
       Parameterzahl gleich gut approximieren (zweifach geprueft), kann das
       nicht an der Ausdruckskraft liegen, sondern nur an der Trainierbarkeit.
       ERGAENZT AM 10. AUGUST, VOR DEM ERSTEN LAUF -- keine Rohdaten
       existierten zu diesem Zeitpunkt. Grund: dieser Ausgang war nicht
       vorgesehen, obwohl Q-015 genau ihn gemessen hat (80 geteilte Winkel
       schlagen 32.640 freie um +0,0674). Ohne den Zweig waere ein solches
       Ergebnis als "unentschieden" gemeldet worden -- der Fehler, der in
       Q-015 schon einmal ein Ergebnis mit p=0,0001 verworfen hat. Ein
       synthetischer Testfall vor dem Lauf hat das reproduziert.
  H0 : untied_L5 ist signifikant besser -> die Teilung kostet, und zwar
       messbar. Dann ist die Architektur eine Kompressionstechnik mit Preis,
       und der Preis gehoert neben jede Parameterzahl.
  HU : weder noch -> erreichte Schranke berichten.

  Faellt ein Ergebnis in H1 UND H1+ zugleich (signifikant, aber kleiner als
  +-0,005), gilt die Gleichwertigkeitsaussage; die Richtung wird nur
  beschreibend genannt. So festgelegt vor dem Lauf.

  AUFLOESUNGSPRUEFUNG (Regel 6): `tied_L5_frozen` -- dieselbe Schaltung mit
  eingefrorenen Zufallswinkeln -- muss signifikant schlechter sein als
  `tied_L5`. Trennt sich das nicht, lernt der Block ueberhaupt nichts, und kein
  Vergleich oben ist interpretierbar.

  ERWARTUNG, vorher notiert: Q-003-replication hat gezeigt, dass Tiefe nichts
  beitraegt (16 Parameter gleichwertig zu 80). Das spricht dafuer, dass auch
  mehr Parameter durch Aufhebung der Teilung wenig bringen -- also fuer H1. Es
  koennte aber auch bedeuten, dass die Aufgabe so wenig Kapazitaet braucht,
  dass gar nichts trennt; genau dafuer ist die Aufloesungspruefung da.

Run: python -m quantum_sim.weight_tying
"""

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import ButterflyQuantumLayer

# (Name, geteilt, Lagen, eingefroren)
SPECS = {
    "tied_L1":        (True, 1, False),
    "tied_L5":        (True, 5, False),
    "tied_L80":       (True, 80, False),
    "untied_L1":      (False, 1, False),
    "untied_L5":      (False, 5, False),
    "tied_L5_frozen": (True, 5, True),
}
ARMS = tuple(SPECS)
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
NUM_QUBITS = 8
EVAL_SEEDS = tuple(range(6601, 6616))       # 15 Seeds
TUNING_SEEDS = (6651, 6652)
# 0,3 ergaenzt (oberer Rand in Q-003-replication) und 0,001 (tied_L80 hat
# 80 Lagen und duerfte deutlich kleinere Raten brauchen als tied_L5).
LR_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


class TyingModel(nn.Module):
    """Encoder -> Butterfly-Block -> Kopf. Identisch zum Hybridmodell, nur dass
    der Block wahlweise geteilt oder ungeteilt parametrisiert ist."""

    def __init__(self, arm: str, input_dim: int, num_classes: int):
        super().__init__()
        tie, layers, frozen = SPECS[arm]
        self.encoder = nn.Linear(input_dim, 2 ** NUM_QUBITS)
        self.block = ButterflyQuantumLayer(NUM_QUBITS, layers, entangle=True,
                                           tie_weights=tie)
        if frozen:
            for p in self.block.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(NUM_QUBITS, num_classes)

    def block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters())

    def trainable_block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters() if p.requires_grad)

    def forward(self, x):
        return self.head(self.block(self.encoder(x)))


def _cell(arm: str, dataset: str, seed: int, lr: float, threads: int,
          max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = TyingModel(arm, bundle.input_dim, bundle.num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = bundle.x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bundle.x_train[idx]), bundle.y_train[idx]).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(bundle.x_test).argmax(-1)
        classes = torch.unique(bundle.y_test)
        bal = float(np.mean([float((pred[bundle.y_test == c] == c).float().mean())
                             for c in classes]))
    tie, layers, frozen = SPECS[arm]
    return {"arm": arm, "dataset": dataset, "seed": seed, "lr": lr,
            "tied": tie, "layers": layers, "frozen": frozen,
            "balanced_accuracy": bal,
            "accuracy": float((pred == bundle.y_test).float().mean()),
            "block_parameters": model.block_parameters(),
            "trainable_block_parameters": model.trainable_block_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["balanced_accuracy"]
                                    for s in TUNING_SEEDS])) for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<16} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-014",
        description=("Kostet die quantenabgeleitete Gewichtsteilung Qualitaet? "
                     "Geteilt gegen ungeteilt bei gleicher Rechenzeit und bei "
                     "angeglichenem Parameterbudget. 15 Seeds, zwei Datensaetze."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "num_qubits": NUM_QUBITS,
                "specs": {k: {"tied": v[0], "layers": v[1], "frozen": v[2]}
                          for k, v in SPECS.items()},
                "lr_grid": list(LR_GRID), "max_samples": MAX_SAMPLES,
                "epochs": EPOCHS, "batch_size": BATCH, "margin": MARGIN,
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "tied_L5 gleichwertig zu untied_L5 (gleiche Rechenzeit)",
                "p2": "tied_L80 gegen untied_L1 (angeglichenes Budget)",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "tied_L5 muss tied_L5_frozen schlagen",
                "erwartung_vorher": ("H1 -- Tiefe traegt laut Q-003-replication "
                                     "nichts bei, also duerften auch mehr "
                                     "Parameter wenig bringen")},
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
                rec.add(**_cell(arm, dataset, seed, lr_map[arm], threads))
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

    print()
    print(f"RESEARCH ID Q-014 | Gewichtsteilung, {NUM_QUBITS} Qubits")
    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 62 - len(ds)))
        head = (f'  {"Arm":>16} {"geteilt":>8} {"Lagen":>6} {"Block-Par.":>11} '
                f'{"balanced":>10} {"std":>8} {"lr":>6}')
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, ds)
            if not len(v):
                continue
            r0 = next(r for r in records if r["arm"] == arm and r["dataset"] == ds)
            print(f'  {arm:>16} {str(r0["tied"]):>8} {r0["layers"]:>6} '
                  f'{r0["block_parameters"]:>11,} {v.mean():>10.4f} '
                  f'{v.std(ddof=1):>8.4f} {r0["lr"]:>6}')

    fam = [("tied_L5", "untied_L5", "PRIMAER"), ("tied_L80", "untied_L1", "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, PRIMARY_DATASET), vals(b, PRIMARY_DATASET)
        p = float(stats.wilcoxon(va, vb).pvalue)
        raw.append(p)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN), int((va > vb).sum()), len(va)))
    adj = holm(raw)

    print()
    print("=" * 94)
    print(f"VORREGISTRIERTE FAMILIE auf {PRIMARY_DATASET}, Grenze +-{MARGIN}")
    for (a, b, tag, delta, p, eq, w, nn), p_h in zip(cells, adj):
        pa = next(r["block_parameters"] for r in records if r["arm"] == a)
        pb = next(r["block_parameters"] for r in records if r["arm"] == b)
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} ({pa:,}) vs {b} ({pb:,}): {delta:>+8.4f}  '
              f'{w}/{nn}  p={p:.4f}  Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"],
                    "params_a": pa, "params_b": pb}

    tv, fv = vals("tied_L5", PRIMARY_DATASET), vals("tied_L5_frozen", PRIMARY_DATASET)
    c_p = float(stats.wilcoxon(tv, fv).pvalue)
    c_ok = tv.mean() > fv.mean() and c_p < 0.05
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"trainiert gegen eingefroren {tv.mean() - fv.mean():+.4f}, p={c_p:.4f}."
            + ("" if c_ok else " Der Block lernt nichts -- kein Vergleich oben "
               "ist interpretierbar."))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    prim, p2 = out["PRIMAER"], out["P2"]
    ratio = prim["params_b"] / max(prim["params_a"], 1)
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        # Zweig ergaenzt am 10. August, VOR dem ersten Lauf. Er fehlte, und die
        # Vorregistrierung kannte nur "gleichwertig" oder "ungeteilt besser".
        # Ein synthetischer Testfall (geteilt +0,0510, 15/15 Seeds, Holm
        # p=0,0001) landete damit bei "unentschieden" -- exakt der Fehler, der
        # in Q-015 ein Ergebnis mit p=0,0001 als nicht verwertbar abgestempelt
        # hat. Q-015 macht diesen Ausgang sogar zum wahrscheinlichen: dort
        # schlugen 80 geteilte Winkel 32.640 freie Freiheitsgrade um +0,0674.
        v = (f"H1+ -- die Teilung kostet nicht nur nichts, sie NUETZT: geteilt "
             f"schlaegt ungeteilt um {prim['delta']:+.4f} bei Faktor "
             f"{ratio:.0f} weniger Parametern und identischer Rechenzeit. "
             f"Beide Familien approximieren bei gleicher Parameterzahl gleich "
             f"gut (zweifach geprueft, 3 und 4 Qubits) -- der Unterschied kann "
             f"also nicht an der Ausdruckskraft liegen, sondern nur an der "
             f"TRAINIERBARKEIT. Das ist dieselbe Richtung wie in Q-015 "
             f"(+0,0674 gegen die volle globale Drehung) und damit eine "
             f"unabhaengige Bestaetigung an einer zweiten Kontrolle. "
             f"Vorbehalt: der ungeteilte Arm hat mit {prim['params_b']:,} "
             f"Parametern auf {MAX_SAMPLES:,} Beispielen auch mehr Gelegenheit "
             f"zur Ueberanpassung.")
        if prim["p_tost"] < 0.05:
            v += (f" ACHTUNG: der Unterschied ist zugleich TOST-gleichwertig "
                  f"innerhalb +-{MARGIN} -- er ist also konsistent, aber "
                  f"kleiner als die vor dem Lauf als bedeutsam erklaerte "
                  f"Schranke. Dann gilt die Gleichwertigkeitsaussage, und die "
                  f"Richtung ist nur beschreibend zu nennen.")
    elif prim["p_tost"] < 0.05:
        v = (f"H1 -- die Gewichtsteilung kostet nichts: gleichwertig innerhalb "
             f"+-{MARGIN} bei Faktor {ratio:.0f} weniger Parametern und "
             f"identischer Rechenzeit. Auf diese Kopplung fuehrt der klassische "
             f"Weg nicht -- sie folgt daraus, dass ein Gatter auf ein Qubit "
             f"wirkt.")
        if p2["p_holm"] < 0.05 and p2["delta"] > 0:
            v += (" Und bei angeglichenem Budget ist das geteilte Netz sogar "
                  "BESSER -- die Beschraenkung ist dann nicht nur sparsam, "
                  "sondern eine bessere Verwendung der Parameter.")
    elif prim["p_holm"] < 0.05 and prim["delta"] < 0:
        v = (f"H0 -- die Teilung kostet {abs(prim['delta']):.4f} balancierte "
             f"Genauigkeit. Das ist der Preis fuer Faktor {ratio:.0f} weniger "
             f"Parameter und gehoert neben jede Parameterangabe.")
    else:
        v = f"HU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f}."
    print(f"BEWERTUNG: {v}")
    print("=" * 94)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.weight_tying '
              "import analyse; analyse('Q-014')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
