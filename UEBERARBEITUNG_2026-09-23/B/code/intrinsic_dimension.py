from __future__ import annotations

"""RESEARCH ID: B-014 -- Setzen die Daten den Rang, oder das Netz?

DIE FRAGE
Eine trainierte 4096er-Schicht arbeitet in rund 64 Richtungen. Woher kommt
diese Zahl? Zwei Erklaerungen, und sie sind vollkommen verschieden:

  (a) das NETZ erzeugt sie -- eine Eigenschaft von Architektur und Optimierer
  (b) die DATEN erzwingen sie -- MNIST liegt bekanntlich auf einer
      niedrigdimensionalen Mannigfaltigkeit im 784-dimensionalen Pixelraum

Bisher steht dazu nichts fest. Alle Datensaetze des Projekts waren VORGEFUNDEN;
ihre intrinsische Dimension ist unbekannt und nicht einstellbar. Damit laesst
sich die Frage prinzipiell nicht beantworten -- man kann nur korrelieren.

DER AUFBAU: EINE ECHTE UNABHAENGIGE VARIABLE
Hier werden die Daten selbst gebaut, mit vorgegebener intrinsischer Dimension d:

    z ~ N(0, I_d)                     latenter Raum, Dimension d
    x = A z + sigma * eps             Einbettung in 784 Dimensionen
    y = argmax(C z)                   Label haengt NUR von z ab, nicht vom Rauschen

A ist eine feste zufaellige 784xd-Matrix mit orthonormalen Spalten, C eine feste
zufaellige Klassenmatrix. Damit ist d die einzige Variable, die sich aendert --
Stichprobenzahl, Eingangsdimension, Klassenzahl und Rauschstaerke bleiben
gleich.

Das ist das erste Experiment in diesem Projekt mit einer wirklich kontrollierten
unabhaengigen Variable. Alles bisherige war ein Vergleich zwischen vorgefundenen
Bedingungen.

VORREGISTRIERUNG
  H1 : Der gemessene Effektivrang der Mittelschicht waechst monoton mit d.
       Spearman(d, Effektivrang) > 0, p < 0,05.
       -> Die Daten setzen den Rang. Dann ist die noetige Schichtbreite eine
          Eigenschaft der AUFGABE und im Prinzip vor dem Training bestimmbar.
  H0 : kein Zusammenhang -- der Rang ist eine Eigenschaft des Netzes und sagt
       ueber die Aufgabe nichts aus.

  ZUSATZ, schaerfer: Wenn der Effektivrang nicht nur mit d waechst, sondern
  ungefaehr GLEICH d ist, waere der Zusammenhang nicht nur monoton, sondern
  quantitativ. Berichtet wird deshalb auch das Verhaeltnis Effektivrang/d.

  KONTROLLE: dieselbe Messung an einem UNTRAINIERTEN Netz. Findet sich der
  Zusammenhang schon dort, kaeme er aus der Einbettung der Daten und nicht aus
  dem Lernen. Ohne diese Kontrolle waere der Befund nicht interpretierbar --
  genau der Fehler, der in diesem Projekt bei der Verschraenkungsentropie
  schon einmal passiert ist.

KORREKTUR VOM 27. AUGUST, IM CODE NACHGEZOGEN AM 23. SEPTEMBER
  Die Korrelation lief ueber alle 240 Einzelmessungen. Das Gitter hat aber nur
  24 Zellen; die zehn Seeds einer Zelle tragen dasselbe d und dasselbe C und
  sind keine unabhaengigen Beobachtungen des Zusammenhangs -- PSEUDOREPLIKATION
  (Tagebuch 27. August, p=3e-187 statt 6e-19). Korrigiert war bis heute nur das
  Tagebuch; wer die Auswertung neu startete, bekam die alten Zahlen. Jetzt
  gilt: korreliert wird ueber die 24 ZELLMITTEL. Die alte Rechnung steht
  daneben, ausdruecklich als "pseudorepliziert, nur zum Vergleich".

  Der Urteilstext behauptete ausserdem "d erklaert mehr als die Klassenzahl".
  Das folgte aus dem Vergleich der Innerhalb-Effekte (d bei fester Klassenzahl
  +0,96 gegen Klassen bei festem d +0,83) und widerspricht den Zellmittel-
  Korrelationen (d +0,61 gegen Klassen +0,67). Beide Rangfolgen messen etwas
  anderes; KEINE sagt, welche Variable "mehr erklaert". Der Text sagt das
  jetzt so. Die vorregistrierte Hypothese H1 (Spearman(d, Rang) > 0) ist davon
  unberuehrt.

  Neu dazu: ein MODELLVERGLEICH auf den Zellmitteln (additiv, multiplikativ,
  Minimum, Produkt, Wechselwirkung; kleinste Quadrate, Leave-one-out) und die
  Korrelationen fuer jedes RANGMASS, das die Rohdaten hergeben. Beides ist
  NACHTRAEGLICH und beschreibend, nicht vorregistriert.

Run:  python -m qire_ai.intrinsic_dimension              der Lauf (Stunden!)
      python -m qire_ai.intrinsic_dimension auswerten [DATEI]
                                                          nur Auswertung
      python -m qire_ai.intrinsic_dimension vortest       Pruefung auf
                                                          synthetischen Daten
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .benchmark import set_seed
from .local_net import MiddleNet
from .metrics import evaluate
from .result_store import Recorder
from .spectral_probe import middle_outputs, output_spectrum, train_with_probe

WIDTH = 4096
INPUT_DIM = 784
NUM_CLASSES = 10
N_SAMPLES = 6000
TEST_FRACTION = 0.30
DIMENSIONS: Tuple[int, ...] = (2, 5, 10, 20, 50, 100)
# Zweite kontrollierte Variable. Im Vortest lag der Effektivrang bei d=50 nicht
# bei 50, sondern bei 8,5 -- auffaellig nah an Klassenzahl minus eins. Ob die
# Datendimension oder die Klassenzahl den Rang setzt, laesst sich nur trennen,
# wenn man BEIDE variiert. Mit nur einer waere jede Antwort eine Vermutung.
CLASS_COUNTS: Tuple[int, ...] = (2, 5, 10, 20)
ENERGY_RANKS = (8, 32, 64, 256)
EVAL_SEEDS = tuple(range(2201, 2211))       # 10 Seeds
EPOCHS = 10
LR = 0.003
BATCH = 256
NOISE = 0.10


def make_manifold_data(d: int, seed: int, n: int = N_SAMPLES,
                       input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES,
                       noise: float = NOISE):
    """Daten mit exakt bekannter intrinsischer Dimension d."""
    g = torch.Generator().manual_seed(seed * 1000 + d)
    z = torch.randn(n, d, generator=g)
    # orthonormale Einbettung: die Mannigfaltigkeit ist genau d-dimensional
    A, _ = torch.linalg.qr(torch.randn(input_dim, d, generator=g))
    x = z @ A.T
    x = x / x.std() * 0.5
    x = x + noise * torch.randn(n, input_dim, generator=g)
    C = torch.randn(num_classes, d, generator=g)
    y = (z @ C.T).argmax(dim=1)
    cut = int(n * (1 - TEST_FRACTION))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def energierang(outputs: torch.Tensor, schwelle: float) -> int:
    """Kleinste Zahl von Richtungen, die `schwelle` der Ausgabeenergie tragen.

    Drittes Rangmass neben Teilnahmezahl und Entropie-Rang (Roy/Vetterli).
    Es braucht eine willkuerliche Schwelle -- genau deshalb wird es mit ZWEI
    Schwellen (90 und 99 %) berichtet und nur zur Robustheitspruefung benutzt.
    """
    y = outputs - outputs.mean(dim=0, keepdim=True)
    e = torch.linalg.svdvals(y.double()) ** 2
    total = float(e.sum())
    if total <= 0:
        return 0
    anteil = torch.cumsum(e, dim=0) / total
    return int(torch.searchsorted(anteil, torch.tensor(schwelle - 1e-12,
                                                       dtype=anteil.dtype))) + 1


def run(dimensions: Sequence[int] = DIMENSIONS, seeds: Sequence[int] = EVAL_SEEDS,
        class_counts: Sequence[int] = CLASS_COUNTS, width: int = WIDTH,
        epochs: int = EPOCHS, lr: float = LR,
        batch_size: int = BATCH, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="B-014",
        description=("Synthetische Daten mit vorgegebener intrinsischer Dimension d. "
                     "Prueft, ob der Effektivrang der Mittelschicht von den Daten "
                     "gesetzt wird. Mit untrainierter Kontrolle."),
        config={"dimensions": list(dimensions), "seeds": list(seeds), "width": width,
                "class_counts": list(class_counts),
                "input_dim": INPUT_DIM, "num_classes": NUM_CLASSES,
                "n_samples": N_SAMPLES, "noise": NOISE, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "torch_threads": threads,
                "energy_ranks": list(ENERGY_RANKS),
                "primary": "Spearman(d, Effektivrang) > 0",
                "control": "dieselbe Messung am untrainierten Netz"},
    )
    grid = [(d, c) for d in dimensions for c in class_counts]
    for i, seed in enumerate(seeds, 1):
        for d, classes in grid:
            set_seed(seed)
            torch.set_num_threads(threads)
            x_tr, y_tr, x_te, y_te = make_manifold_data(d, seed, num_classes=classes)
            model = MiddleNet(INPUT_DIM, width, classes,
                              lambda: nn.Linear(width, width), f"d{d}c{classes}",
                              "dense_ends", depth=1)
            h_vorher = middle_outputs(model, x_tr, max_rows=2048)
            before = output_spectrum(h_vorher, ENERGY_RANKS)
            train_with_probe(model, x_tr, y_tr, epochs=epochs, lr=lr,
                             batch_size=batch_size, seed=seed)
            h_nachher = middle_outputs(model, x_tr, max_rows=2048)
            after = output_spectrum(h_nachher, ENERGY_RANKS)
            _, _, _, ex = evaluate(model, x_te, y_te)
            # Spektrum der Rohdaten als weitere Bezugsgroesse
            raw = output_spectrum(x_tr[:2048], ENERGY_RANKS)
            rec.add(d=d, num_classes=classes, seed=seed,
                    balanced_accuracy=float(ex["balanced_accuracy"]),
                    effective_rank=after["effective_rank"],
                    effective_rank_untrained=before["effective_rank"],
                    effective_rank_raw_data=raw["effective_rank"],
                    entropy_rank=after["entropy_rank"],
                    entropy_rank_untrained=before["entropy_rank"],
                    # Schwellenraenge, ergaenzt am 23. September: B-014 hatte
                    # nur die Teilnahmezahl und den Entropie-Rang; die
                    # Robustheit ueber mehrere Rangmasse war deshalb nicht
                    # pruefbar. Kuenftige Laeufe speichern beide mit.
                    rank_90=energierang(h_nachher, 0.90),
                    rank_99=energierang(h_nachher, 0.99),
                    rank_90_untrained=energierang(h_vorher, 0.90),
                    rank_99_untrained=energierang(h_vorher, 0.99),
                    energy_64=after["energy_64"],
                    energy_64_untrained=before["energy_64"])
        print(f"  seed {seed} fertig ({i}/{len(seeds)})")
    return rec


# ---------------------------------------------------------------------------
# Auswertung -- ueber ZELLMITTEL, nicht ueber Einzelmessungen
# ---------------------------------------------------------------------------
# Rangmasse, die ein Lauf speichern kann. Aeltere Laeufe (B-014 vom 4. August)
# haben nur die ersten beiden; die uebrigen werden dann als fehlend gemeldet,
# nicht stillschweigend weggelassen.
RANGMASSE: Tuple[Tuple[str, str], ...] = (
    ("effective_rank", "Teilnahmezahl (Hauptmass)"),
    ("entropy_rank", "Entropie-Rang (Roy/Vetterli)"),
    ("rank_90", "Energierang 90 %"),
    ("rank_99", "Energierang 99 %"),
)


def zellmittel(records, dims: Sequence[int], classes: Sequence[int],
               key: str) -> np.ndarray:
    """Matrix der Zellmittel, Zeilen = d, Spalten = Klassenzahl. NaN, wo die
    Zelle leer ist oder das Feld fehlt."""
    m = np.full((len(dims), len(classes)), np.nan)
    for i, d in enumerate(dims):
        for j, c in enumerate(classes):
            v = [float(r[key]) for r in records
                 if int(r["d"]) == d and int(r["num_classes"]) == c and key in r]
            if v:
                m[i, j] = float(np.mean(v))
    return m


def korrelation_zellmittel(dims, classes, m: np.ndarray) -> Dict[str, float]:
    """Spearman ueber die Zellmittel -- eine Beobachtung je Gitterzelle."""
    from scipy import stats

    dd, cc = np.meshgrid(dims, classes, indexing="ij")
    ok = np.isfinite(m)
    rd = stats.spearmanr(dd[ok], m[ok])
    rc = stats.spearmanr(cc[ok], m[ok])
    return {"rho_d": float(rd.statistic), "p_d": float(rd.pvalue),
            "rho_c": float(rc.statistic), "p_c": float(rc.pvalue),
            "n": int(ok.sum())}


def korrelation_pseudorepliziert(records, key: str) -> Dict[str, float]:
    """DIE ALTE RECHNUNG, nur zum Vergleich: Spearman ueber alle
    Einzelmessungen. Pseudorepliziert -- die Seeds einer Zelle tragen
    dasselbe d und C. Ihre p-Werte sind um Groessenordnungen zu klein und
    duerfen nicht zitiert werden."""
    from scipy import stats

    rows = [r for r in records if key in r]
    d = [int(r["d"]) for r in rows]
    c = [int(r["num_classes"]) for r in rows]
    v = [float(r[key]) for r in rows]
    rd, rc = stats.spearmanr(d, v), stats.spearmanr(c, v)
    return {"rho_d": float(rd.statistic), "p_d": float(rd.pvalue),
            "rho_c": float(rc.statistic), "p_c": float(rc.pvalue), "n": len(v)}


def _modelle():
    """Kandidaten fuer Rang = f(d, C). Jeder liefert (Anpassung, Parameterzahl).
    `ld`, `lc` sind natuerliche Logarithmen; `r` sind die Zellmittel."""
    from scipy import optimize

    def linear(spalten, log_ziel=False):
        def anpassen(ld, lc, r):
            X = np.column_stack([np.ones_like(ld)] + [f(ld, lc) for f in spalten])
            y = np.log(r) if log_ziel else r
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

            def vorhersage(ld2, lc2):
                X2 = np.column_stack([np.ones_like(ld2)]
                                     + [f(ld2, lc2) for f in spalten])
                v = X2 @ beta
                return np.exp(v) if log_ziel else v
            return vorhersage
        return anpassen, 1 + len(spalten)

    def minimum(mit_achse: bool):
        def f(p, ld, lc):
            v = np.minimum(p[0] * ld, p[1] * lc)
            return v + p[2] if mit_achse else v

        def anpassen(ld, lc, r):
            best = None
            # mehrere Startpunkte: min() ist nicht glatt, ein einzelner Start
            # kann in einem Ast haengen bleiben
            for a0 in (0.3, 1.0, 3.0):
                for b0 in (0.3, 1.0, 3.0):
                    p0 = [a0, b0] + ([1.0] if mit_achse else [])
                    res = optimize.least_squares(lambda p: f(p, ld, lc) - r, p0)
                    if best is None or res.cost < best.cost:
                        best = res
            p = best.x
            return lambda ld2, lc2: f(p, ld2, lc2)
        return anpassen, 3 if mit_achse else 2

    return {
        "konstant": linear([]),
        "additiv       c + a·ln d + b·ln C": linear([lambda a, b: a, lambda a, b: b]),
        "multiplikativ c · d^a · C^b": linear([lambda a, b: a, lambda a, b: b],
                                              log_ziel=True),
        "Minimum       min(a·ln d, b·ln C)": minimum(False),
        "Minimum       c + min(a·ln d, b·ln C)": minimum(True),
        "Produkt       c + e·ln d·ln C": linear([lambda a, b: a * b]),
        "Wechselwirk.  c + a·ln d + b·ln C + e·ln d·ln C":
            linear([lambda a, b: a, lambda a, b: b, lambda a, b: a * b]),
    }


def modellvergleich(dims, classes, m: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Kleinste Quadrate auf den Zellmitteln, drei Fehlermasse:

      RMSE          in der Stichprobe -- belohnt jeden zusaetzlichen Parameter
      LOO           Leave-one-out ueber die Zellen -- der faire Vergleich
      ohne Stufe    eine ganze d- bzw. C-Stufe weglassen und vorhersagen --
                    prueft, ob das Modell zwischen Stufen traegt, statt nur
                    Nachbarzellen zu interpolieren

    NACHTRAEGLICH und beschreibend: kein Modell war vorregistriert, und 24
    Zellen tragen keinen Test zwischen sieben Modellen.
    """
    dd, cc = np.meshgrid(dims, classes, indexing="ij")
    ok = np.isfinite(m)
    ld, lc, r = np.log(dd[ok].astype(float)), np.log(cc[ok].astype(float)), m[ok]
    dstufe, cstufe = dd[ok], cc[ok]
    alle = np.arange(len(r))
    out: Dict[str, Dict[str, float]] = {}
    for name, (anpassen, k) in _modelle().items():
        voll = anpassen(ld, lc, r)
        rmse = float(np.sqrt(np.mean((voll(ld, lc) - r) ** 2)))
        loo = []
        for i in alle:
            t = alle != i
            loo.append(float(anpassen(ld[t], lc[t], r[t])(ld[i:i + 1], lc[i:i + 1])[0]
                             - r[i]))

        def block(stufen):
            fehler = []
            for s in np.unique(stufen):
                t, v = stufen != s, stufen == s
                fehler.extend(anpassen(ld[t], lc[t], r[t])(ld[v], lc[v]) - r[v])
            return float(np.sqrt(np.mean(np.square(fehler))))
        out[name] = {"k": k, "rmse": rmse,
                     "loo": float(np.sqrt(np.mean(np.square(loo)))),
                     "ohne_d_stufe": block(dstufe), "ohne_c_stufe": block(cstufe)}
    return out


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from .result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    dims = [int(d) for d in config["dimensions"]]
    classes = sorted({int(r["num_classes"]) for r in records})
    out: Dict[str, object] = {}

    def cell(d, c, key):
        return np.array([r[key] for r in records
                         if int(r["d"]) == d and int(r["num_classes"]) == c])

    n_seeds = len(cell(dims[0], classes[0], "effective_rank"))
    rang = zellmittel(records, dims, classes, "effective_rank")
    rang_u = zellmittel(records, dims, classes, "effective_rank_untrained")
    print()
    print(f"RESEARCH ID B-014 | synthetische Mannigfaltigkeiten, {n_seeds} Seeds, "
          f"Breite {config['width']}, {len(dims) * len(classes)} Gitterzellen")
    label = "d / Klassen"
    head = f'  {label:>12} ' + " ".join(f'{c:>9}' for c in classes)
    for titel, m in (("Effektivrang der Mittelschicht nach dem Training "
                      "(Zellmittel):", rang),
                     ("Dieselbe Tabelle am UNTRAINIERTEN Netz (Kontrolle):", rang_u)):
        print()
        print(titel)
        print(head); print("  " + "-" * (len(head) - 2))
        for i, d in enumerate(dims):
            print(f'  {d:>12} ' + "".join(f'{m[i, j]:>9.1f} ' for j in range(len(classes))))

    # --- welche Variable sagt den Rang vorher? UEBER ZELLMITTEL ---
    kz, kzu = (korrelation_zellmittel(dims, classes, rang),
               korrelation_zellmittel(dims, classes, rang_u))
    ps, psu = (korrelation_pseudorepliziert(records, "effective_rank"),
               korrelation_pseudorepliziert(records, "effective_rank_untrained"))
    print()
    print(f"Welche Variable sagt den Rang vorher?  Spearman ueber die "
          f"{kz['n']} ZELLMITTEL (gilt)")
    for name, k in (("trainiert  ", kz), ("untrainiert", kzu)):
        print(f"  {name}: d -> Rang  rho={k['rho_d']:+.3f}  p={k['p_d']:.1e}    "
              f"Klassen -> Rang  rho={k['rho_c']:+.3f}  p={k['p_c']:.1e}")
    print(f"  zum Vergleich, PSEUDOREPLIZIERT ueber {ps['n']} Einzelmessungen "
          f"(NICHT zitieren):")
    for name, k in (("trainiert  ", ps), ("untrainiert", psu)):
        print(f"  {name}: d -> Rang  rho={k['rho_d']:+.3f}  p={k['p_d']:.1e}    "
              f"Klassen -> Rang  rho={k['rho_c']:+.3f}  p={k['p_c']:.1e}")

    # partiell: Zusammenhang mit d INNERHALB jeder Klassenzahl und umgekehrt.
    # Das rechnet schon immer auf Zellmitteln und war nie pseudorepliziert.
    within_c = [stats.spearmanr(dims, rang[:, j]).statistic for j in range(len(classes))]
    within_d = [stats.spearmanr(classes, rang[i, :]).statistic for i in range(len(dims))]
    print()
    print(f'  d-Effekt innerhalb fester Klassenzahl : rho im Mittel '
          f'{np.nanmean(within_c):+.3f}')
    print(f'  Klassen-Effekt innerhalb festem d     : rho im Mittel '
          f'{np.nanmean(within_d):+.3f}')

    print()
    print('  Vergleich mit der Vorhersage "Rang = Klassenzahl - 1":')
    for j, c in enumerate(classes):
        print(f'    Klassen {c:>3}: vorhergesagt {c-1:>3}, gemessen '
              f'{float(np.nanmean(rang[:, j])):>6.1f} (gemittelt ueber alle d)')

    # --- Robustheit ueber die Rangmasse ---
    print()
    print("Robustheit ueber die Rangmasse (Zellmittel, trainiert):")
    masse: Dict[str, object] = {}
    for key, name in RANGMASSE:
        m = zellmittel(records, dims, classes, key)
        if not np.isfinite(m).any():
            print(f"  {name:<30} -- nicht in den Rohdaten (offen, braucht Neulauf)")
            masse[key] = None
            continue
        k = korrelation_zellmittel(dims, classes, m)
        masse[key] = k
        print(f"  {name:<30} d: rho={k['rho_d']:+.3f} p={k['p_d']:.1e}   "
              f"Klassen: rho={k['rho_c']:+.3f} p={k['p_c']:.1e}")
    out["rangmasse"] = masse

    # --- Modellvergleich ---
    print()
    print("Modellvergleich auf den Zellmitteln (NACHTRAEGLICH, beschreibend):")
    vergleiche: Dict[str, object] = {}
    for key, name in RANGMASSE[:2]:
        m = zellmittel(records, dims, classes, key)
        if not np.isfinite(m).all():
            continue
        mv = modellvergleich(dims, classes, m)
        vergleiche[key] = mv
        print(f"  {name}:")
        print(f"    {'Modell':<50} {'k':>2} {'RMSE':>7} {'LOO':>7} "
              f"{'ohne d':>7} {'ohne C':>7}")
        bester = min(mv, key=lambda n: mv[n]["loo"])
        for n, v in mv.items():
            print(f"    {n:<50} {v['k']:>2} {v['rmse']:>7.3f} {v['loo']:>7.3f} "
                  f"{v['ohne_d_stufe']:>7.3f} {v['ohne_c_stufe']:>7.3f}"
                  + ("   <- kleinster LOO" if n == bester else ""))
    out["modellvergleich"] = vergleiche

    out.update({"rho_d": kz["rho_d"], "p_d": kz["p_d"],
                "rho_classes": kz["rho_c"], "p_classes": kz["p_c"],
                "rho_d_untrained": kzu["rho_d"], "p_d_untrained": kzu["p_d"],
                "rho_classes_untrained": kzu["rho_c"],
                "p_classes_untrained": kzu["p_c"],
                "korrelation_ueber": "24 Zellmittel",
                "pseudorepliziert_nur_vergleich": {"trainiert": ps,
                                                   "untrainiert": psu},
                "within_class_d_effect": float(np.nanmean(within_c)),
                "within_d_class_effect": float(np.nanmean(within_d))})

    # --- Urteil ---
    # Vorregistriert ist nur H1: Spearman(d, Rang) > 0 bei p < 0,05. Die
    # Unterscheidung "d erklaert mehr" / H1' stand nur im Code und benutzte die
    # Innerhalb-Effekte. Am 23. September ersetzt: sind BEIDE Zellmittel-
    # Korrelationen signifikant, sagt das Urteil genau das -- und nicht, welche
    # Variable mehr erklaert. Das ist eine Korrektur der FORMULIERUNG nach dem
    # Blick auf die Daten und gehoert so ins Tagebuch; an H1 aendert sie nichts.
    print()
    print("=" * 78)
    print("BEWERTUNG (H1 vor dem Lauf festgelegt, Formulierung am 23.09. korrigiert)")
    sig_d = kz["p_d"] < 0.05 and kz["rho_d"] > 0
    sig_c = kz["p_c"] < 0.05 and kz["rho_c"] > 0
    zahlen = (f"(Zellmittel: d rho={kz['rho_d']:+.2f}, p={kz['p_d']:.1e}; "
              f"Klassen rho={kz['rho_c']:+.2f}, p={kz['p_c']:.1e})")
    if sig_d and sig_c:
        v = (f"H1 -- der Rang waechst mit der Datendimension, aber ebenso mit der "
             f"Klassenzahl {zahlen}. Beide setzen den Rang gemeinsam; WELCHE "
             f"mehr erklaert, entscheidet dieser Lauf nicht")
        if (kz["rho_d"] > kz["rho_c"]) != (np.nanmean(within_c) > np.nanmean(within_d)):
            v += (f" -- Randkorrelation und Innerhalb-Effekte (d {np.nanmean(within_c):+.2f}, "
                  f"Klassen {np.nanmean(within_d):+.2f}) ordnen die beiden sogar "
                  f"verschieden")
        v += "."
    elif sig_d:
        v = (f"H1 -- die Datendimension setzt den Rang, die Klassenzahl nicht "
             f"nachweisbar {zahlen}.")
    elif sig_c:
        v = (f"H1' -- nicht die Datendimension, sondern die KLASSENZAHL setzt den "
             f"Rang {zahlen}. Vorregistriert war H1; H1 haelt damit nicht.")
    else:
        v = (f"H0 -- keine der beiden Variablen erklaert den Rang monoton {zahlen}.")
    if kzu["rho_d"] > 0 and kzu["p_d"] < 0.05:
        v += (f" Vorbehalt: das UNTRAINIERTE Netz zeigt den d-Zusammenhang bereits "
              f"(rho={kzu['rho_d']:+.3f}), aber keinen Klassenzusammenhang "
              f"(rho={kzu['rho_c']:+.3f}, p={kzu['p_c']:.2f}); die "
              f"Klassenabhaengigkeit entsteht erst beim Lernen.")
    if "effective_rank" in vergleiche:
        mv = vergleiche["effective_rank"]
        bester = min(mv, key=lambda n: mv[n]["loo"])
        v += (f" Nachtraeglicher Modellvergleich: kleinster Leave-one-out-Fehler bei "
              f"'{bester.split('  ')[0].strip()}' -- beschreibend, nicht getestet.")
    v += (" Architektur wurde NICHT variiert (eine MLP, Breite "
          f"{config['width']}); 'nicht von der Architektur' folgt hieraus nicht.")
    print(f"  {v}")
    print("=" * 78)
    out["verdict"] = v
    return out


# ---------------------------------------------------------------------------
# Vortest -- die Korrektur ist eine neue Behauptung und wird selbst geprueft
# ---------------------------------------------------------------------------
def vortest() -> bool:
    """Prueft die neue Auswertung auf synthetischen Daten mit bekannter Antwort.

    1. Zellmittel-Korrelation = Spearman ueber explizit gebildete Mittelwerte,
       und der pseudoreplizierte p-Wert ist bei denselben Daten viel kleiner.
    2. Modellvergleich erkennt das WAHRE Modell (je 5 Rauschziehungen):
       Minimum-Daten -> ein Minimum-Modell hat den kleinsten LOO-Fehler,
       additive Daten -> ein Modell ohne Minimum gewinnt, und das Minimum-
       Modell mit Achse verliert gegen das additive.
    3. energierang: bekannte Spektren.
    """
    from scipy import stats

    rng = np.random.default_rng(1401)
    dims, classes = list(DIMENSIONS), list(CLASS_COUNTS)
    ok_alle = True

    def synth(f, rauschen, seeds=10):
        rec = []
        for d in dims:
            for c in classes:
                for s in range(seeds):
                    rec.append({"d": d, "num_classes": c, "seed": s,
                                "effective_rank": f(d, c) + rng.normal(0, rauschen)})
        return rec

    print("=" * 78)
    print("VORTEST intrinsic_dimension -- Zellmittel und Modellvergleich")
    print("=" * 78)
    rec = synth(lambda d, c: 1 + 0.5 * np.log(d) + 0.3 * np.log(c), 0.5)
    m = zellmittel(rec, dims, classes, "effective_rank")
    k = korrelation_zellmittel(dims, classes, m)
    hand = stats.spearmanr([d for d in dims for _ in classes], m.ravel())
    p = korrelation_pseudorepliziert(rec, "effective_rank")
    ok1 = abs(k["rho_d"] - hand.statistic) < 1e-12 and k["n"] == 24
    ok2 = p["p_d"] < k["p_d"] / 100 and p["n"] == 240
    print(f"  Zellmittel = Hand-Spearman ueber 24 Mittel        "
          f"{'OK' if ok1 else 'FEHLER'}")
    print(f"  pseudorepliziert p={p['p_d']:.1e} << Zellmittel p={k['p_d']:.1e}  "
          f"{'OK' if ok2 else 'FEHLER'}")
    ok_alle &= ok1 and ok2

    faelle = [
        ("Minimum-Daten", lambda d, c: 0.5 + min(0.9 * np.log(d), 1.2 * np.log(c)),
         lambda b: b.startswith("Minimum")),
        ("additive Daten", lambda d, c: 1 + 0.5 * np.log(d) + 0.6 * np.log(c),
         lambda b: not b.startswith("Minimum")),
    ]
    for name, f, erwartet in faelle:
        treffer = 0
        for _ in range(5):
            m = zellmittel(synth(f, 0.3), dims, classes, "effective_rank")
            mv = modellvergleich(dims, classes, m)
            bester = min(mv, key=lambda n: mv[n]["loo"])
            treffer += int(erwartet(bester))
        ok = treffer == 5
        ok_alle &= ok
        print(f"  {name:<16} richtige Modellklasse in {treffer}/5 Ziehungen      "
              f"{'OK' if ok else 'FEHLER'}")

    # deterministisch, Energie 25:1 -> 96,2 % in der ersten Richtung. (Erster
    # Entwurf nahm Zufallsspalten im Verhaeltnis 100:1; dort liegen 99,0 % in
    # EINER Richtung, der erwartete 99-%-Rang 2 war also falsch erwartet.)
    y = torch.zeros(100, 10)
    y[:, 0] = 5.0 * torch.tensor([1.0, -1.0] * 50)
    y[:, 1] = torch.tensor([1.0, 1.0, -1.0, -1.0] * 25)
    r90, r99 = energierang(y, 0.90), energierang(y, 0.99)
    ok3 = r90 == 1 and r99 == 2 and energierang(torch.eye(8) - 1 / 8, 0.99) == 7
    print(f"  energierang: dominant 90%={r90}, 99%={r99}, Simplex 99%="
          f"{energierang(torch.eye(8) - 1 / 8, 0.99)} (erwartet 1, 2, 7)   "
          f"{'OK' if ok3 else 'FEHLER'}")
    ok_alle &= ok3
    print("=" * 78)
    print("VORTEST BESTANDEN" if ok_alle else "VORTEST FEHLGESCHLAGEN")
    print("=" * 78)
    return bool(ok_alle)


if __name__ == "__main__":
    import sys

    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        raise SystemExit(0 if vortest() else 1)
    if befehl == "auswerten":
        analyse(sys.argv[2] if len(sys.argv) > 2 else "B-014")
        raise SystemExit(0)
    if befehl:
        raise SystemExit(f"unbekannter Befehl {befehl!r} -- vortest | auswerten [DATEI]")
    recorder = run()
    recorder.save(analysis=analyse(recorder))
