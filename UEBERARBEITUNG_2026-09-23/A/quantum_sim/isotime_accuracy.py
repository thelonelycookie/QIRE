from __future__ import annotations

"""RESEARCH ID: Q-009 -- Bei gleichem Zeitbudget: wann gewinnt die Quantenarchitektur?

WARUM DIESER LAUF, UND WARUM DER VORHERIGE ENTWURF VERWORFEN WURDE
Der erste Entwurf (`width_accuracy.py`) verglich Butterfly gegen dicht bei
**gleicher Breite**. Im Vortest zeigte sich, dass diese Frage ihre Antwort schon
kennt: bei Breite 16.384 stehen **140 Parameter gegen 268 Millionen**. Auf einer
Aufgabe, die Kapazitaet verlangt, kann die Schaltung das nicht gewinnen -- nicht
weil die Architektur schlecht waere, sondern weil sie hundert Parameter hat.

Die Aussage, um die es geht, lautet aber nicht "gleich gut bei gleicher Breite",
sondern **weniger Rechenleistung bei gleicher Leistung**. Die passende Frage ist
deshalb:

    Was erreicht die Butterfly-Architektur in DERSELBEN ZEIT, die der dichte
    Arm bei seiner eigenen bestmoeglichen Breite braucht?

Gemessen am 5. August ist Butterfly bei 16.384 Dimensionen rund doppelt so
schnell wie dicht gleicher Breite (Median 1,99x, 7/7 Messreihen). Bei gleichem
Zeitbudget darf der dichte Arm deshalb nur eine kleinere Breite benutzen. Genau
diese Breite wird hier **empirisch einkalibriert**, nicht geschaetzt.

Das ist dieselbe Disziplin wie in B-004, wo vier Familien auf exakt 524.288
Parameter angeglichen wurden, statt sie bei gleicher Breite antreten zu lassen.

DIE ZWEITE ACHSE -- und der eigentliche Ertrag
Ob die Schaltung bei gleicher Zeit gewinnt, haengt davon ab, wie viel Kapazitaet
die Aufgabe wirklich verlangt. Deshalb wird die Schwierigkeit mitvariiert: viele
kleine Haufen mit zufaellig zugewiesenen Klassen, und die Zahl der Haufen wird
durchgefahren. Erwartung, vorher festgelegt:

  * wenige Haufen  -> die Aufgabe braucht wenig Kapazitaet, die 140 Parameter
                      reichen, und der Zeitvorteil schlaegt durch
  * viele Haufen   -> die Aufgabe braucht Kapazitaet, der dichte Arm gewinnt
                      trotz kleinerer Breite

Der interessante Wert ist die **Grenze dazwischen**. Das ist dieselbe Form von
Ergebnis wie die 512-Neuronen-Schwelle des Projekts: kein "besser" oder
"schlechter", sondern eine bezifferte Bedingung.

VORREGISTRIERUNG
  PRIMAER  Bei der kleinsten Aufgabenschwierigkeit (`CLUSTERS[0]`) ist
           `butterfly` bei gleichem Zeitbudget besser als `dense_isotime`.
  P2       Es existiert eine Grenze: das Vorzeichen der Differenz dreht ueber
           die Schwierigkeitsstufen (Spearman ueber die Stufen, negativ).
  Holm ueber diese beiden.

  H1 : beide halten -> "Die quantenabgeleitete Architektur liefert bei gleichem
       Rechenbudget mehr Genauigkeit, solange die Aufgabe hoechstens X Gebiete
       verlangt." Das ist die Aussage in verteidigbarer Form.
  H0 : PRIMAER faellt -> auch bei gleicher Zeit und der einfachsten Aufgabe
       gewinnt der dichte Arm. Dann ist der Zeitvorteil erkauft und die These
       in dieser Form widerlegt.
  HU : unentschieden -> erreichte Schranke berichten, nichts behaupten.

  AUFLOESUNGSPRUEFUNG (Regel 6): `dense_quarter` mit einem Viertel der
  isochronen Breite muss signifikant schlechter sein als `dense_isotime`.
  Trennt sich das nicht, kann der Aufbau Kapazitaetsunterschiede nicht
  aufloesen und kein Ergebnis ist interpretierbar. Im Vortest des verworfenen
  Entwurfs ist genau das passiert (1,0000 gegen 0,9997) -- deshalb sind die
  Aufgaben hier deutlich schwerer angesetzt.

Run: python -m quantum_sim.isotime_accuracy
"""

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.equivalence import holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import apply_1q, cnot_permutation

ARMS = ("butterfly", "dense_isotime", "dense_quarter")
NUM_QUBITS = 14                 # Breite 16.384 -- dort ist Butterfly schneller
LAYERS = 5
CLUSTERS = (64, 256, 1024)      # Schwierigkeitsachse
NUM_CLASSES = 10
INTRINSIC_DIM = 64
INPUT_DIM = 256
N_TRAIN, N_TEST = 6000, 3000
EPOCHS = 3
BATCH = 64
EVAL_SEEDS = (9101, 9102, 9103)
TUNING_SEED = 9201
LR_GRID = (0.0003, 0.001, 0.003, 0.01)
MARGIN_ACC = 0.02
CALIB_BATCHES = 3


def make_data(seed: int, clusters: int):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(INTRINSIC_DIM, INPUT_DIM, generator=g) / math.sqrt(INTRINSIC_DIM)
    centres = torch.randn(clusters, INTRINSIC_DIM, generator=g) * 2.0
    labels = torch.randint(0, NUM_CLASSES, (clusters,), generator=g)
    out = []
    for n in (N_TRAIN, N_TEST):
        c = torch.randint(0, clusters, (n,), generator=g)
        z = centres[c] + torch.randn(n, INTRINSIC_DIM, generator=g) * 0.55
        out += [z @ basis, labels[c]]
    return out


class ButterflyBlock(nn.Module):
    """d -> d: |U(theta) . normalise(x)|^2, geteilte Winkel je Qubit-Ebene."""

    def __init__(self, num_qubits: int, num_layers: int = LAYERS):
        super().__init__()
        self.n, self.L = num_qubits, num_layers
        self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)
        for q in range(num_qubits):
            self.register_buffer(f"perm_{q}",
                                 cnot_permutation(num_qubits, q, (q + 1) % num_qubits))

    def _gate(self, layer: int, q: int) -> torch.Tensor:
        a, b = self.theta[layer, q, 0], self.theta[layer, q, 1]
        ca, sa = torch.cos(a / 2).to(torch.complex64), torch.sin(a / 2).to(torch.complex64)
        eb, ebc = torch.exp(-0.5j * b.to(torch.complex64)), torch.exp(0.5j * b.to(torch.complex64))
        return torch.stack([torch.stack([ca * eb, -sa * eb]),
                            torch.stack([sa * ebc, ca * ebc])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x.to(torch.complex64)
        state = state / state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for layer in range(self.L):
            for q in range(self.n):
                state = apply_1q(state, self._gate(layer, q), q, self.n)
            for q in range(self.n):
                state = state[:, getattr(self, f"perm_{q}")]
        return (state.real ** 2 + state.imag ** 2) * float(2 ** self.n)


class Model(nn.Module):
    """Encoder -> Block -> Kopf. Die Normierung bekommen ALLE Arme, sonst
    vergleicht man Struktur gegen Normierung statt Struktur gegen Struktur."""

    def __init__(self, arm: str, width: int):
        super().__init__()
        self.encoder = nn.Linear(INPUT_DIM, width)
        if arm == "butterfly":
            self.block, out_dim = ButterflyBlock(NUM_QUBITS), width
        else:
            self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU())
            out_dim = width
        self.head = nn.Linear(out_dim, NUM_CLASSES)
        self.width = width

    def block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters())

    def forward(self, x):
        h = self.encoder(x)
        h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(h.shape[1])
        return self.head(self.block(h))


def _step_time(arm: str, width: int, threads: int) -> float:
    """Zeit eines vollstaendigen Trainingsschritts (vorwaerts + rueckwaerts)."""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = Model(arm, width)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(BATCH, INPUT_DIM), torch.randint(0, NUM_CLASSES, (BATCH,))
    for _ in range(2):                       # aufwaermen
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(CALIB_BATCHES):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
    dt = (time.perf_counter() - t0) / CALIB_BATCHES
    del model, opt
    return dt


def calibrate(threads: int) -> Tuple[int, float, float]:
    """Die dichte Breite finden, die genauso lange braucht wie Butterfly.

    Empirisch statt aus der O(d^2)-Formel geschaetzt: dieses Projekt hat
    fuenfmal gezeigt, dass Operationszaehlungen die Zeit nicht vorhersagen.
    """
    target = _step_time("butterfly", 2 ** NUM_QUBITS, threads)
    print(f"  Butterfly bei Breite {2 ** NUM_QUBITS:,}: {target * 1000:.0f} ms je Schritt")
    lo, hi = 512, 2 ** NUM_QUBITS
    best, best_err = lo, float("inf")
    for _ in range(6):                       # binaere Suche
        mid = (lo + hi) // 2
        t = _step_time("dense_isotime", mid, threads)
        err = abs(t - target) / target
        print(f"    dicht bei Breite {mid:>6,}: {t * 1000:>7.0f} ms  "
              f"({t / target:.2f}x des Ziels)")
        if err < best_err:
            best, best_err = mid, err
        if t > target:
            hi = mid
        else:
            lo = mid
        if err < 0.05:
            break
    print(f"  -> isochrone dichte Breite: {best:,} "
          f"(Abweichung {best_err * 100:.1f} %)")
    return best, target, best_err


def _cell(arm: str, width: int, clusters: int, seed: int, lr: float,
          threads: int, epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    xtr, ytr, xte, yte = make_data(seed, clusters)
    model = Model(arm, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(xtr.shape[0], generator=g)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xte[i:i + 256]).argmax(-1)
                          for i in range(0, len(xte), 256)])
    bal = float(np.mean([float((pred[yte == c] == c).float().mean())
                         for c in torch.unique(yte)]))
    return {"arm": arm, "width": width, "clusters": clusters, "seed": seed,
            "lr": lr, "balanced_accuracy": bal,
            "accuracy": float((pred == yte).float().mean()),
            "train_seconds": train_s, "block_parameters": model.block_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def run(clusters: Sequence[int] = CLUSTERS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    print("KALIBRIERUNG -- gleiches Zeitbudget")
    iso_width, target, err = calibrate(threads)
    widths = {"butterfly": 2 ** NUM_QUBITS, "dense_isotime": iso_width,
              "dense_quarter": max(256, iso_width // 4)}

    rec = Recorder(
        experiment_id="Q-009",
        description=("Butterfly gegen dicht bei GLEICHEM ZEITBUDGET statt "
                     "gleicher Breite, ueber drei Schwierigkeitsstufen. Die "
                     "isochrone dichte Breite wird empirisch einkalibriert."),
        config={"arms": list(ARMS), "num_qubits": NUM_QUBITS, "layers": LAYERS,
                "widths": widths, "clusters": list(clusters),
                "seeds": list(seeds), "lr_grid": list(LR_GRID),
                "epochs": EPOCHS, "batch_size": BATCH, "n_train": N_TRAIN,
                "intrinsic_dim": INTRINSIC_DIM, "margin_accuracy": MARGIN_ACC,
                "calibration": {"target_step_s": target, "rel_error": err},
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "butterfly > dense_isotime bei der leichtesten Aufgabe",
                "resolution_check": "dense_quarter muss schlechter sein als dense_isotime"},
    )
    print(f"  Breiten: {widths}")
    for c in clusters:
        print(f"\n=== {c} Haufen ===")
        for arm in ARMS:
            best, best_acc = LR_GRID[0], -1.0
            for lr in LR_GRID:
                a = _cell(arm, widths[arm], c, TUNING_SEED, lr, threads,
                          epochs=max(1, EPOCHS - 1))["balanced_accuracy"]
                if a > best_acc:
                    best, best_acc = lr, a
            print(f"  {arm}: lr {best}")
            for seed in seeds:
                rec.add(**_cell(arm, widths[arm], c, seed, best, threads))
            v = [r["balanced_accuracy"] for r in rec.records
                 if r["arm"] == arm and r["clusters"] == c]
            t = [r["train_seconds"] for r in rec.records
                 if r["arm"] == arm and r["clusters"] == c]
            print(f"    {arm:>14}  {np.mean(v):.4f} +-{np.std(v, ddof=1):.4f}  "
                  f"({np.mean(t):.0f}s Training)")
        print(f"  Zwischenstand nach {c} Haufen gesichert:")
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

    clusters = sorted({int(r["clusters"]) for r in records})
    out: Dict[str, object] = {}

    def vals(arm: str, c: int, field: str = "balanced_accuracy") -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm
                       and int(r["clusters"]) == c), key=lambda r: r["seed"])
        return np.array([r[field] for r in rows])

    print()
    print("RESEARCH ID Q-009 | gleiches Zeitbudget, Breiten "
          f"{config.get('widths')}")
    head = (f'  {"Haufen":>7} {"butterfly":>18} {"dense_isotime":>18} '
            f'{"Differenz":>10} {"Zeit bf/dicht":>14}')
    print(); print(head); print("  " + "-" * (len(head) - 2))
    deltas = []
    for c in clusters:
        b, d = vals("butterfly", c), vals("dense_isotime", c)
        tb, td = vals("butterfly", c, "train_seconds"), vals("dense_isotime", c, "train_seconds")
        delta = float(b.mean() - d.mean())
        deltas.append(delta)
        print(f'  {c:>7} {b.mean():>10.4f} +-{b.std(ddof=1):.4f} '
              f'{d.mean():>10.4f} +-{d.std(ddof=1):.4f} {delta:>+10.4f} '
              f'{tb.mean():>6.0f}s/{td.mean():.0f}s')
        out[f"clusters_{c}"] = {"delta": delta,
                                "butterfly": float(b.mean()),
                                "dense": float(d.mean()),
                                "time_ratio": float(tb.mean() / td.mean())}

    c0 = clusters[0]
    b, d = vals("butterfly", c0), vals("dense_isotime", c0)
    p = float(stats.wilcoxon(b, d).pvalue) if len(b) > 2 else float("nan")
    eq = tost_paired(b, d, MARGIN_ACC)
    rho, p_tr = stats.spearmanr(clusters, deltas) if len(clusters) >= 3 else (float("nan"),) * 2
    p_holm = holm([p, float(p_tr)])

    q = vals("dense_quarter", c0)
    c_p = float(stats.wilcoxon(d, q).pvalue) if len(q) == len(d) and len(d) > 2 else 1.0
    c_ok = len(q) == len(d) and len(d) > 0 and d.mean() > q.mean() and c_p < 0.05

    print()
    print("=" * 92)
    print(f"PRIMAER bei {c0} Haufen: butterfly - dense = {b.mean() - d.mean():+.4f}  "
          f"p={p:.4f}  Holm={p_holm[0]:.4f}")
    print(f"P2 Grenze:  rho={rho:+.3f} ueber die Schwierigkeit  p={p_tr:.4f}  "
          f"Holm={p_holm[1]:.4f}")
    print(f"TOST +-{MARGIN_ACC}: p={eq['p_tost']:.4f}, erreichte Schranke "
          f"+-{eq['achieved']:.4f}")
    out["primary"] = {"delta": float(b.mean() - d.mean()), "p": p,
                      "p_holm": p_holm[0], "p_tost": eq["p_tost"],
                      "achieved": eq["achieved"]}
    out["trend"] = {"rho": float(rho), "p": float(p_tr), "p_holm": p_holm[1]}

    print()
    if c_ok:
        ctxt = (f"Aufloesungspruefung BESTANDEN: dense_isotime schlaegt ein "
                f"Viertel der Breite ({d.mean() - q.mean():+.4f}, p={c_p:.4f}).")
    else:
        ctxt = (f"Aufloesungspruefung NICHT BESTANDEN (p={c_p:.4f}): der Aufbau "
                f"trennt Kapazitaetsunterschiede nicht. Kein Ergebnis oben ist "
                f"interpretierbar -- die Aufgabe muss schwerer werden.")
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif p_holm[0] < 0.05 and b.mean() > d.mean():
        v = (f"H1 -- bei gleichem Rechenbudget liefert die quantenabgeleitete "
             f"Architektur bei {c0} Gebieten mehr Genauigkeit "
             f"({b.mean() - d.mean():+.4f}).")
        if p_holm[1] < 0.05 and rho < 0:
            v += (" Und es gibt eine Grenze: der Vorteil faellt mit der "
                  "Aufgabenschwierigkeit (rho={:+.2f}).".format(rho))
    elif p_holm[0] < 0.05:
        v = (f"H0 -- der dichte Arm gewinnt auch bei gleichem Zeitbudget und der "
             f"leichtesten Aufgabe ({b.mean() - d.mean():+.4f}). Der Zeitvorteil "
             f"ist erkauft.")
    else:
        v = (f"HU -- unentschieden bei {len(b)} Seeds; die Daten tragen erst ab "
             f"+-{eq['achieved']:.4f}.")
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
        print('  Nachholen mit: python -c "from quantum_sim.isotime_accuracy '
              "import analyse; analyse('Q-009')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
