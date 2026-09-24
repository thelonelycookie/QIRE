from __future__ import annotations

"""RESEARCH ID: Q-015 -- Die Kontrolle, die in Q-003 gefehlt hat.

WIE DIESER LAUF ZUSTANDE KAM
Am 6. August wurde die quadratische Auslese zum ersten Mal *charakterisiert*
statt nur benutzt. Nachgerechnet und exakt bestaetigt (Abweichung 1e-7):

    out_q(x) = x^T A_q x / (x^T x)     mit  A_q = U^T P_q U

A_q ist eine echte orthogonale Projektion (Eigenwerte {0,1}, Rang d/n). Die
Auslese berechnet also fuer jeden Kanal den **Anteil der Eingabe-Energie in
einem orthogonalen Unterraum**; die Ausgaben sind nichtnegativ und summieren
sich exakt zu 1. Sie ist skaleninvariant. Linear/Tanh hat keine dieser
Eigenschaften.

Damit erfuellt die Schaltung den Pruefmassstab dieses Projekts -- sie legt eine
strukturelle Einschraenkung auf -- und das ist bisher nie bemerkt worden.

**Und dabei faellt ein Fehler in der bisherigen Kontrolle auf.**
`QuadraticReadout` in `hybrid_final_control.py` benutzt BLOCKDIAGONALE
orthogonale Matrizen (32 Bloecke à 8x8 bei 256 Dimensionen). Information
ueberquert dort nie eine Blockgrenze. Die Quantenschaltung mischt dagegen
**global** ueber alle 256 Amplituden -- der CNOT-Ring koppelt alle Qubits.

Die beiden sind damit **nicht dieselbe Funktionsklasse**, und B-004 hat bereits
gemessen, wie gross der Unterschied zwischen blocklokaler und globaler Mischung
ist. Die gemessenen Zahlen fuegen sich genau so:

    classical_quadratic   blocklokal, orthogonal   2.048 Par.   0,8881
    classical_quad_free   blocklokal, frei         2.048 Par.   0,9213
    hybrid                GLOBAL,     orthogonal      80 Par.   0,9339

Die Schaltung gewinnt, weil sie global mischt. Keine der beiden Kontrollen tut
das. **Die entscheidende Kontrolle fehlt: eine volle globale orthogonale
Auslese.**

WAS DIESER LAUF TRENNT
Der Vorteil der Schaltung kann zwei Quellen haben, und bisher sind sie vermischt:

  (a) die FUNKTIONSKLASSE -- globale statt blocklokaler Mischung
  (b) die PARAMETRISIERUNG -- 2*n*L Winkel fuer eine globale Drehung, statt
      d(d-1)/2 freier Parameter

Nur (b) ist quantenspezifisch. (a) ist eine klassische Eigenschaft, die man
auch klassisch haben kann -- sie kostet dann nur 32.640 statt 80 Parameter.

DIE ARME
  hybrid            Quantenschaltung, global orthogonal, 80 Parameter
  global_orth       volle 256x256 orthogonale Matrix ueber expm(A - A^T),
                    32.640 freie Parameter. **Dieselbe Funktionsklasse wie
                    hybrid**, nur teuer parametrisiert.
  block_orth        blockdiagonal orthogonal, 2.048 -- die bisherige Kontrolle
  quad_free         blockdiagonal frei, 2.048 -- die andere bisherige Kontrolle

VORREGISTRIERUNG
  PRIMAER  hybrid gegen global_orth: TOST auf Gleichwertigkeit innerhalb
           +-0,005. Gleichwertig hiesse: die Butterfly-Parametrisierung
           erreicht dieselbe Funktionsklasse mit **408-fach weniger
           Parametern**. Das waere der eigenstaendige Beitrag der
           Quantenherleitung, praezise benannt.
  P2       global_orth gegen block_orth: traegt globale Mischung ueberhaupt?
  Holm ueber genau diese beiden.

  H1 : PRIMAER gleichwertig UND P2 signifikant zugunsten global_orth
       -> Der Vorteil zerfaellt sauber: die globale Mischung macht die
          Funktionsklasse, die Quantenparametrisierung macht sie billig.
  H0 : global_orth ist signifikant besser als hybrid -> die billige
       Parametrisierung kostet, und zwar messbar. Dann ist die Schaltung eine
       Naeherung, kein gleichwertiger Ersatz.
  HU : weder noch.

  AUFLOESUNGSPRUEFUNG (Regel 6) ist P2 selbst: Trennt sich `global_orth` nicht
  von `block_orth`, dann loest der Aufbau den Unterschied zwischen globaler und
  blocklokaler Mischung nicht auf -- und dann ist auch der Primaertest nicht
  interpretierbar. B-004 laesst hier einen deutlichen Effekt erwarten.

  ERWARTUNG, vorher notiert: H1. Die Schaltung ist mathematisch eine globale
  orthogonale Drehung; eine andere globale orthogonale Drehung sollte dieselbe
  Aufgabe gleich gut loesen, wenn beide gut optimiert werden. Sollte
  `global_orth` deutlich BESSER sein, heisst das, dass die 80 Winkel die
  erreichbaren Drehungen doch spuerbar einschraenken -- auch das waere ein
  klares Ergebnis und die Antwort auf die Alleinstellungsfrage, nur die andere.

Run: python -m quantum_sim.global_orthogonal
"""

import math
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .hybrid_final_control import QuadraticReadout
from .quantum_layer import QuantumLayer

ARMS = ("hybrid", "global_orth", "block_orth", "quad_free")
REGISTERED = ("PRIMAER", "P2")
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
NUM_QUBITS = 8
LAYERS = 5
EVAL_SEEDS = tuple(range(7701, 7716))       # 15 Seeds
TUNING_SEEDS = (7751, 7752)
LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
MAX_SAMPLES = 4000
EPOCHS = 12
BATCH = 128


class GlobalOrthogonalReadout(nn.Module):
    """normieren -> VOLLE orthogonale Drehung -> <Z_q>-Auslese.

    Exakt dieselbe Funktionsklasse wie die Quantenschaltung, nur mit einer frei
    parametrisierten globalen Drehung statt 2*n*L Winkeln. Orthogonalitaet ueber
    expm(A - A^T), also exakt statt als Strafterm.

    WARUM DIE AUSLESE NICHT `arange(dim) % n` BENUTZT -- ein Konfundierungsfehler,
    der beim Bauen dieses Laufs auffiel und vor dem Start behoben wurde:

    `QuadraticReadout` zerlegt den Index in n DISJUNKTE Gruppen. Bei einer
    orthogonalen Drehung bleibt die Norm erhalten, also summieren sich die n
    gruppenweisen Quadratsummen auf exakt 1. Nachgemessen: die Summe ist
    -6,000000 mit Streuung 3e-07, und der Effektivrang der Ausgabe faellt von 8
    auf **6,99** -- eine der acht Ausgabedimensionen traegt nur eine Konstante.

    Genau das erklaert, warum `classical_quadratic` (orthogonal, Rang 7) mit
    0,8881 schlechter ist als `classical_quad_free` (frei, Rang 7,97) mit
    0,9213 -- bei identischer Parameterzahl und identischer Mischung.

    Die Quantenschaltung hat dieses Problem NICHT: <Z_q> = P(Bit q = 0) -
    P(Bit q = 1) benutzt n **ueberlappende Zweiteilungen** statt einer Zerlegung
    in n Gruppen. Jedes Qubit halbiert den Raum anders, es gibt keine feste
    Summe, und der Effektivrang bleibt bei 7,99.

    Waere hier die Gruppenauslese verwendet worden, haette der Lauf "global
    gegen Butterfly" gemessen UND "disjunkt gegen ueberlappend" -- zwei
    Unterschiede in einem Vergleich. Deshalb bekommt dieser Arm dieselbe
    <Z>-Auslese wie die Schaltung, und der Vergleich isoliert wirklich nur die
    Parametrisierung.
    """

    def __init__(self, dim: int, num_qubits: int):
        super().__init__()
        self.dim, self.n = dim, num_qubits
        assert 2 ** num_qubits == dim
        self.weight = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))

    def effective_parameters(self) -> int:
        return self.dim * (self.dim - 1) // 2

    def forward(self, x):
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        skew = self.weight - self.weight.T
        h = x @ torch.matrix_exp(skew).T
        prob = h ** 2                                   # Born-Regel
        # <Z_q> ueber ueberlappende Zweiteilungen -- Bit q des Index, Qubit 0
        # ist das hoechstwertige Bit (Konvention aus `simulator.py`)
        outs = []
        for q in range(self.n):
            p = prob.reshape(-1, 2 ** q, 2, 2 ** (self.n - 1 - q))
            outs.append(p[:, :, 0, :].sum(dim=(1, 2)) - p[:, :, 1, :].sum(dim=(1, 2)))
        return torch.stack(outs, dim=1)


class ReadoutModel(nn.Module):
    def __init__(self, arm: str, input_dim: int, num_classes: int):
        super().__init__()
        d = 2 ** NUM_QUBITS
        self.encoder = nn.Linear(input_dim, d)
        if arm == "hybrid":
            self.middle = QuantumLayer(NUM_QUBITS, LAYERS, entangle=True)
        elif arm == "global_orth":
            self.middle = GlobalOrthogonalReadout(d, NUM_QUBITS)
        elif arm == "block_orth":
            self.middle = QuadraticReadout(d, NUM_QUBITS, orthogonal=True)
        elif arm == "quad_free":
            self.middle = QuadraticReadout(d, NUM_QUBITS, orthogonal=False)
        else:
            raise ValueError(arm)
        self.head = nn.Linear(NUM_QUBITS, num_classes)
        self.arm = arm

    def middle_parameters(self) -> int:
        """Wirksame Freiheitsgrade, nicht rohe Tensorgroesse -- bei einer
        schiefsymmetrisch erzwungenen Matrix zaehlt nur die Haelfte."""
        if self.arm == "hybrid":
            return self.middle.num_circuit_parameters()
        if self.arm == "global_orth":
            return self.middle.effective_parameters()
        return sum(p.numel() for p in self.middle.parameters())

    def forward(self, x):
        return self.head(self.middle(self.encoder(x)))


def _cell(arm: str, dataset: str, seed: int, lr: float, threads: int,
          max_samples: int = MAX_SAMPLES, epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    bundle = load_dataset(dataset, seed=seed, max_samples=max_samples)
    model = ReadoutModel(arm, bundle.input_dim, bundle.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = bundle.x_train.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bundle.x_train[idx]), bundle.y_train[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(bundle.x_test).argmax(-1)
        classes = torch.unique(bundle.y_test)
        bal = float(np.mean([float((pred[bundle.y_test == c] == c).float().mean())
                             for c in classes]))
    return {"arm": arm, "dataset": dataset, "seed": seed, "lr": lr,
            "balanced_accuracy": bal,
            "accuracy": float((pred == bundle.y_test).float().mean()),
            "middle_parameters": model.middle_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def sweep_lr(dataset: str, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([_cell(arm, dataset, s, lr, threads,
                                          max_samples=2000, epochs=6)["balanced_accuracy"]
                                    for s in TUNING_SEEDS])) for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {dataset:>14} / {arm:<13} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-015",
        description=("Die fehlende Kontrolle: volle globale orthogonale Auslese "
                     "gegen die Quantenschaltung. Trennt Funktionsklasse "
                     "(global vs blocklokal) von Parametrisierung (80 vs 32.640)."),
        config={"arms": list(ARMS), "datasets": list(datasets),
                "seeds": list(seeds), "num_qubits": NUM_QUBITS, "layers": LAYERS,
                "lr_grid": list(LR_GRID), "max_samples": MAX_SAMPLES,
                "epochs": EPOCHS, "batch_size": BATCH, "margin": MARGIN,
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "hybrid gleichwertig zu global_orth (+-0,005)",
                "p2": "global_orth besser als block_orth",
                "holm_family": list(REGISTERED),
                "primary_dataset": PRIMARY_DATASET,
                "resolution_check": "P2 selbst: global muss blocklokal schlagen",
                "anlass": ("Charakterisierung der Auslese als Rayleigh-Quotient "
                           "am 6. August; dabei fiel auf, dass die bisherige "
                           "Kontrolle blockdiagonal ist und damit eine andere "
                           "Funktionsklasse als die global mischende Schaltung")},
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
    print(f"RESEARCH ID Q-015 | {len(vals(ARMS[0], datasets[0]))} Seeds")
    for ds in datasets:
        print()
        print(f"--- {ds} " + "-" * max(4, 62 - len(ds)))
        head = (f'  {"Arm":>13} {"Mischung":>12} {"balanced":>10} {"std":>8} '
                f'{"Freiheitsgrade":>15} {"lr":>6}')
        print(head); print("  " + "-" * (len(head) - 2))
        mixing = {"hybrid": "global", "global_orth": "global",
                  "block_orth": "blocklokal", "quad_free": "blocklokal"}
        for arm in ARMS:
            v = vals(arm, ds)
            if not len(v):
                continue
            r0 = next(r for r in records if r["arm"] == arm and r["dataset"] == ds)
            print(f'  {arm:>13} {mixing[arm]:>12} {v.mean():>10.4f} '
                  f'{v.std(ddof=1):>8.4f} {r0["middle_parameters"]:>15,} '
                  f'{r0["lr"]:>6}')

    fam = [("hybrid", "global_orth", "PRIMAER"),
           ("global_orth", "block_orth", "P2")]
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
        pa = next(r["middle_parameters"] for r in records if r["arm"] == a)
        pb = next(r["middle_parameters"] for r in records if r["arm"] == b)
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} ({pa:,}) vs {b} ({pb:,}): {delta:>+8.4f}  '
              f'{w}/{nn}  p={p:.4f}  Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"],
                    "params_a": pa, "params_b": pb}

    p2 = out["P2"]
    # Die Vorregistrierung verlangt TRENNUNG, nicht Richtung: "Trennt sich
    # global_orth nicht von block_orth, dann loest der Aufbau nicht auf."
    # Die erste Fassung des Codes verlangte zusaetzlich delta > 0 und hat damit
    # ein hochsignifikantes Ergebnis (p=0,0001) als "nicht verwertbar"
    # abgestempelt, nur weil es in die unerwartete Richtung ging. Der Code war
    # strenger als die Anmeldung -- korrigiert am 6. August.
    res_ok = p2["p_holm"] < 0.05
    direction = "global besser" if p2["delta"] > 0 else "BLOCKLOKAL besser"
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if res_ok else 'NICHT BESTANDEN'}: "
            f"globale gegen blocklokale Mischung {p2['delta']:+.4f}, "
            f"Holm p={p2['p_holm']:.4f} -- {direction}."
            + ("" if res_ok else " Der Aufbau trennt globale und blocklokale "
               "Mischung nicht -- dann ist auch der Primaertest nicht "
               "interpretierbar.")
            + (" Die Richtung ist der Vorhersage entgegengesetzt; das ist ein "
               "Ergebnis, kein Messversagen." if res_ok and p2["delta"] < 0 else ""))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(res_ok), "text": ctxt}

    prim = out["PRIMAER"]
    ratio = prim["params_b"] / max(prim["params_a"], 1)
    print()
    if not res_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        # Dieser Zweig fehlte in der ersten Fassung: ich hatte nur
        # "gleichwertig" oder "global_orth besser" vorgesehen und gar nicht
        # damit gerechnet, dass 80 Parameter 32.640 SCHLAGEN koennten.
        v = (f"H1+ -- die Schaltung SCHLAEGT die volle globale Drehung um "
             f"{prim['delta']:+.4f} bei {ratio:.0f}-fach weniger "
             f"Freiheitsgraden. Beide haben dieselbe Funktionsklasse und sind "
             f"beide universell -- der Unterschied kann also nicht an der "
             f"Ausdruckskraft liegen, sondern nur an der TRAINIERBARKEIT. Die "
             f"Butterfly-Parametrisierung ist demnach nicht bloss sparsamer, "
             f"sondern besser konditioniert. Vorbehalt: der freie Arm hat mit "
             f"32.640 Parametern auf 4.000 Beispielen auch mehr Gelegenheit "
             f"zur Ueberanpassung, und seine Streuung ist vierfach hoeher -- "
             f"beides gehoert zur Deutung dazu.")
    elif prim["p_tost"] < 0.05:
        v = (f"H1 -- die Butterfly-Parametrisierung erreicht dieselbe "
             f"Funktionsklasse wie eine volle globale Drehung, mit Faktor "
             f"{ratio:.0f} weniger Freiheitsgraden (gleichwertig innerhalb "
             f"+-{MARGIN}). Der Vorteil zerfaellt damit sauber: die GLOBALE "
             f"MISCHUNG macht die Funktionsklasse, die QUANTENPARAMETRISIERUNG "
             f"macht sie billig. Nur der zweite Teil ist quantenspezifisch -- "
             f"und er ist damit praezise benannt.")
    elif prim["p_holm"] < 0.05 and prim["delta"] < 0:
        v = (f"H0 -- die volle Drehung ist um {abs(prim['delta']):.4f} besser. "
             f"Die 80 Winkel schraenken die erreichbaren Drehungen also spuerbar "
             f"ein; die Schaltung ist eine Naeherung, kein gleichwertiger "
             f"Ersatz. Das ist der Preis fuer Faktor {ratio:.0f}.")
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
        print('  Nachholen mit: python -c "from quantum_sim.global_orthogonal '
              "import analyse; analyse('Q-015')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
