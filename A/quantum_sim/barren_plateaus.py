from __future__ import annotations

"""RESEARCH ID: Q-005 -- Warum die Quantenschaltung nicht skaliert: Barren Plateaus.

WARUM DIESER LAUF
Die Q-Serie hat gezeigt, dass die Hybridschaltung ihren Vorteil verliert, sobald
eine faire klassische Kontrolle danebensteht. Was sie NICHT gezeigt hat, ist
*warum* die Schaltung nicht einfach mit mehr Qubits besser wird -- der
naheliegendste Ausweg aus jedem der bisherigen Negativbefunde.

Die Theorie kennt den Grund: In zufaellig initialisierten, hinreichend tiefen
parametrisierten Quantenschaltungen faellt die Varianz des Gradienten
**exponentiell mit der Qubitzahl** (McClean et al. 2018, "Barren plateaus in
quantum neural network training landscapes"). Bei n Qubits ist das Trainingssignal
also um den Faktor 2^-n gedaempft; ab einer gewissen Groesse ist es numerisch
nicht mehr von null zu unterscheiden, und die Schaltung ist prinzipiell nicht
mehr trainierbar.

Das ist im Projekt noch nie gemessen worden, obwohl der differenzierbare
Simulator genau das kann. Und es ist kein neuer Abgrenzungstest -- es ist die
Erklaerung fuer die Decke, an die alle bisherigen anstossen.

DER ZUSAMMENHANG ZUM KONZENTRATIONSPRINZIP
Aus den Quantenkernen stammt der Satz: in einem 2^n-dimensionalen Hilbertraum
sind zufaellige Zustaende fast immer orthogonal, gemessen als Halbierung der
mittleren Kernaehnlichkeit je Qubit (0,349 -> 0,162 -> 0,082 -> 0,024). Barren
Plateaus sind dieselbe Aussage, von der anderen Seite: beide folgen daraus, dass
zufaellige Zustaende im exponentiell grossen Raum keine gemeinsame Struktur mehr
teilen. Faellt die Gradientenvarianz hier ebenfalls wie 2^-n, ist das eine
**zweite, unabhaengige Bestaetigung desselben Prinzips an einem anderen
Mechanismus** -- und damit deutlich mehr wert als ein weiterer Einzelbefund.

DIE ARME
  deep_entangling     die Schaltung aus `quantum_layer.QuantumLayer`, mit
                      CNOT-Ring, feste Tiefe. Hier wird der Abfall erwartet.
  deep_no_entangle    dieselbe Schaltung ohne CNOT. Ohne Verschraenkung bleibt
                      der Zustand ein Produktzustand; <Z_0> haengt dann nur von
                      den Rotationen auf Qubit 0 ab, und die Varianz muss
                      **konstant** in n sein. Das ist die Abgrenzungskontrolle:
                      sie isoliert die Verschraenkung als Ursache.
  shallow_entangling  mit CNOT, aber nur eine Lage. Flache Schaltungen mit
                      lokaler Kostenfunktion sind nach Cerezo et al. 2021 vom
                      Effekt weitgehend verschont; erwartet wird ein deutlich
                      flacherer Abfall als bei `deep_entangling`.

MESSGROESSE
Kostenfunktion C(theta) = <Z_0> auf dem Zustand, den die Schaltung aus |0...0>
erzeugt. Gemessen wird die Varianz von dC/dtheta[0,0,0] ueber R unabhaengige
Zufallsinitialisierungen der Winkel. Kein Training, keine Daten -- nur
Vorwaerts- und Rueckwaertslauf. Deshalb ist der Lauf billig.

VORREGISTRIERUNG (vor dem Lauf festgelegt)
  Ausgewertet wird die Steigung von log2(Var) gegen n, je Arm, mit
  Bootstrap-Konfidenzintervall ueber die R Ziehungen.

  PRIMAER  Steigung von `deep_entangling` ist negativ und ihr 95-%-Intervall
           schliesst 0 aus.
  P2       Steigung von `deep_entangling` ist signifikant steiler als die von
           `deep_no_entangle`.
  Holm ueber diese beiden. Alles andere ist Einordnung.

  H1 : beide halten, und die Steigung von deep_entangling liegt nahe -1
       -> Var ~ 2^-n bestaetigt. Die Schaltung ist ab wenigen Qubits nicht mehr
          trainierbar, und zwar wegen der Verschraenkung.
  H0 : die Steigung ist nicht von 0 zu unterscheiden (TOST gegen +-0,1)
       -> kein Barren Plateau in diesem Aufbau; die Decke der Q-Serie hat eine
          andere Ursache und muss anders gesucht werden.
  HU : weder noch -> Datenlage traegt die Aussage nicht, erreichte Schranke
       wird berichtet.

  AUFLOESUNGSPRUEFUNG (Regel 6): `deep_no_entangle` MUSS eine Steigung nahe 0
  zeigen. Faellt dort ebenfalls die Varianz exponentiell, misst der Aufbau
  etwas anderes als Verschraenkung -- dann ist kein Ergebnis dieses Laufs
  interpretierbar, und das wird so berichtet.

Run: python -m quantum_sim.barren_plateaus
"""

import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from qire_ai.result_store import Recorder

from .quantum_layer import QuantumLayer

ARMS = ("deep_entangling", "deep_no_entangle", "shallow_entangling")
QUBITS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DRAWS = 400                 # Zufallsinitialisierungen je (Arm, Qubitzahl)
DEEP_LAYERS = 6
SHALLOW_LAYERS = 1
BOOTSTRAP = 2000
SLOPE_MARGIN = 0.1          # Grenze fuer "keine exponentielle Daempfung"
BASE_SEED = 5100


def _build(arm: str, n: int) -> QuantumLayer:
    if arm == "deep_entangling":
        return QuantumLayer(n, DEEP_LAYERS, entangle=True)
    if arm == "deep_no_entangle":
        return QuantumLayer(n, DEEP_LAYERS, entangle=False)
    if arm == "shallow_entangling":
        return QuantumLayer(n, SHALLOW_LAYERS, entangle=True)
    raise ValueError(arm)


def _basis_state(n: int) -> torch.Tensor:
    """|0...0> als Amplitudenvektor, Form (1, 2**n)."""
    x = torch.zeros(1, 2 ** n)
    x[0, 0] = 1.0
    return x


def gradients(arm: str, n: int, draws: int = DRAWS,
              seed: int = BASE_SEED) -> np.ndarray:
    """dC/dtheta[0,0,0] ueber `draws` unabhaengige Zufallsinitialisierungen."""
    torch.manual_seed(seed)
    x = _basis_state(n)
    out: List[float] = []
    for _ in range(draws):
        layer = _build(arm, n)          # __init__ zieht theta ~ U[0, 2pi)
        cost = layer(x)[0, 0]           # <Z> auf Qubit 0, lokale Kostenfunktion
        cost.backward()
        assert layer.theta.grad is not None
        out.append(float(layer.theta.grad[0, 0, 0]))
    return np.asarray(out, dtype=float)


def _slope(ns: Sequence[int], variances: Sequence[float]) -> float:
    """Steigung von log2(Var) gegen n, kleinste Quadrate."""
    y = np.log2(np.maximum(np.asarray(variances, float), 1e-300))
    return float(np.polyfit(np.asarray(ns, float), y, 1)[0])


def _bootstrap_slope(per_n: Dict[int, np.ndarray], rounds: int = BOOTSTRAP,
                     seed: int = 0) -> np.ndarray:
    """Bootstrap-Verteilung der Steigung, indem je Qubitzahl die Gradienten
    neu gezogen und die Varianz daraus neu berechnet wird."""
    rng = np.random.default_rng(seed)
    ns = sorted(per_n)
    out = np.empty(rounds)
    for r in range(rounds):
        v = [float(rng.choice(per_n[n], size=len(per_n[n]), replace=True).var(ddof=1))
             for n in ns]
        out[r] = _slope(ns, v)
    return out


def run(arms: Sequence[str] = ARMS, qubits: Sequence[int] = QUBITS,
        draws: int = DRAWS, threads: int = 6) -> Recorder:
    torch.set_num_threads(threads)
    rec = Recorder(
        experiment_id="Q-005",
        description=("Barren Plateaus: Varianz von dC/dtheta gegen die "
                     "Qubitzahl, drei Arme, kein Training. Prueft, ob die "
                     "Decke der Q-Serie an exponentiell gedaempften Gradienten "
                     "haengt und ob die Verschraenkung die Ursache ist."),
        config={"arms": list(arms), "qubits": list(qubits), "draws": draws,
                "deep_layers": DEEP_LAYERS, "shallow_layers": SHALLOW_LAYERS,
                "cost": "<Z_0> auf |0...0>", "bootstrap": BOOTSTRAP,
                "slope_margin": SLOPE_MARGIN, "torch_threads": threads,
                "base_seed": BASE_SEED,
                "primary": "Steigung log2(Var) vs n fuer deep_entangling < 0",
                "holm_family": ["PRIMAER", "P2"],
                "resolution_check": "deep_no_entangle muss Steigung ~ 0 zeigen"},
    )
    for arm in arms:
        for n in qubits:
            g = gradients(arm, n, draws, seed=BASE_SEED + 17 * n)
            rec.add(arm=arm, num_qubits=n, draws=int(len(g)),
                    grad_var=float(g.var(ddof=1)), grad_mean=float(g.mean()),
                    grad_absmax=float(np.abs(g).max()),
                    layers=DEEP_LAYERS if arm != "shallow_entangling" else SHALLOW_LAYERS,
                    gradients=[float(v) for v in g])
            print(f"    {arm:>18}  n={n:>2}  Var={g.var(ddof=1):.3e}  "
                  f"|g|max={np.abs(g).max():.3e}")
        # Nach jedem Arm sichern (Regel 5), nicht erst am Ende.
        print(f"    Zwischenstand nach {arm} gesichert:")
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from qire_ai.equivalence import holm

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]

    arms = [str(a) for a in config["arms"]]
    per_arm: Dict[str, Dict[int, np.ndarray]] = {}
    for arm in arms:
        rows = sorted((r for r in records if r["arm"] == arm),
                      key=lambda r: r["num_qubits"])
        per_arm[arm] = {int(r["num_qubits"]): np.asarray(r["gradients"], float)
                        for r in rows}

    out: Dict[str, object] = {}
    print()
    print(f"RESEARCH ID Q-005 | Barren Plateaus, {config['draws']} Ziehungen je Zelle")

    # ---- Varianz gegen Qubitzahl ----
    for arm in arms:
        print()
        print(f"--- {arm} " + "-" * max(4, 52 - len(arm)))
        head = f'  {"Qubits":>6} {"Var[dC/dtheta]":>16} {"log2(Var)":>10}'
        print(head); print("  " + "-" * (len(head) - 2))
        for n in sorted(per_arm[arm]):
            v = float(per_arm[arm][n].var(ddof=1))
            print(f'  {n:>6} {v:>16.4e} {math.log2(max(v, 1e-300)):>10.2f}')

    # ---- Steigungen mit Bootstrap-Intervall ----
    print()
    print("=" * 82)
    print("STEIGUNG von log2(Var) gegen n  (-1 entspricht exakt Var ~ 2^-n)")
    stats: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        ns = sorted(per_arm[arm])
        obs = _slope(ns, [float(per_arm[arm][n].var(ddof=1)) for n in ns])
        boot = _bootstrap_slope(per_arm[arm], seed=abs(hash(arm)) % (2 ** 31))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # zweiseitiger Bootstrap-p-Wert gegen die Nullhypothese Steigung = 0
        p0 = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
        p0 = float(min(1.0, max(p0, 1.0 / len(boot))))
        stats[arm] = {"slope": obs, "ci_low": float(lo), "ci_high": float(hi),
                      "p_vs_zero": p0}
        print(f'  {arm:>18}  Steigung {obs:>+7.3f}   95%-KI '
              f'[{lo:>+7.3f}, {hi:>+7.3f}]   p(=0) {p0:.4f}')
        out[f"slope|{arm}"] = stats[arm]

    # ---- vorregistrierte Familie ----
    deep, ctrl = "deep_entangling", "deep_no_entangle"
    b_deep = _bootstrap_slope(per_arm[deep], seed=1)
    b_ctrl = _bootstrap_slope(per_arm[ctrl], seed=2)
    diff = b_deep - b_ctrl
    p_diff = 2.0 * min((diff >= 0).mean(), (diff <= 0).mean())
    p_diff = float(min(1.0, max(p_diff, 1.0 / len(diff))))
    raw = [stats[deep]["p_vs_zero"], p_diff]
    p_holm = holm(raw)

    print()
    print("VORREGISTRIERTE FAMILIE (Holm ueber genau diese zwei)")
    print(f'  PRIMAER  {deep} Steigung < 0            p={raw[0]:.4f}  Holm={p_holm[0]:.4f}')
    print(f'  P2       {deep} steiler als {ctrl}      p={raw[1]:.4f}  Holm={p_holm[1]:.4f}')
    out["primary"] = {"p": raw[0], "p_holm": p_holm[0]}
    out["p2"] = {"p": raw[1], "p_holm": p_holm[1], "delta_slope":
                 float(stats[deep]["slope"] - stats[ctrl]["slope"])}

    # ---- Aufloesungspruefung ----
    c = stats[ctrl]
    ctrl_flat = abs(c["slope"]) < SLOPE_MARGIN and c["ci_low"] > -SLOPE_MARGIN
    print()
    if ctrl_flat:
        ctrl_txt = (f"Aufloesungspruefung BESTANDEN: der Arm ohne Verschraenkung "
                    f"zeigt Steigung {c['slope']:+.3f}, also keine exponentielle "
                    f"Daempfung. Was bei {deep} gemessen wird, haengt damit an der "
                    f"Verschraenkung und nicht an der Qubitzahl an sich.")
    else:
        ctrl_txt = (f"Aufloesungspruefung NICHT BESTANDEN: auch ohne Verschraenkung "
                    f"faellt die Varianz (Steigung {c['slope']:+.3f}, KI "
                    f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]). Dann misst der "
                    f"Aufbau nicht die Verschraenkung, und kein Ergebnis dieses "
                    f"Laufs darf ihr zugeschrieben werden.")
    print(f"  {ctrl_txt}")
    out["resolution_check"] = {"passed": bool(ctrl_flat), **c, "text": ctrl_txt}

    # ---- Urteil ----
    d = stats[deep]
    print()
    print("BEWERTUNG (Kriterium vor dem Lauf festgelegt)")
    if p_holm[0] < 0.05 and d["slope"] < 0 and p_holm[1] < 0.05:
        v = (f"H1 -- die Gradientenvarianz faellt exponentiell mit der Qubitzahl "
             f"(Steigung {d['slope']:+.3f}, 95%-KI [{d['ci_low']:+.3f}, "
             f"{d['ci_high']:+.3f}]), und die Verschraenkung ist die Ursache. "
             f"Damit ist die Decke der Q-Serie erklaert: die Schaltung wird mit "
             f"mehr Qubits nicht besser, sondern untrainierbar.")
    elif abs(d["slope"]) < SLOPE_MARGIN and d["ci_low"] > -SLOPE_MARGIN:
        v = (f"H0 -- kein Barren Plateau in diesem Aufbau (Steigung "
             f"{d['slope']:+.3f} innerhalb +-{SLOPE_MARGIN}). Die Decke der "
             f"Q-Serie hat eine andere Ursache.")
    else:
        v = (f"HU -- unentschieden. Steigung {d['slope']:+.3f}, 95%-KI "
             f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]; weder exponentieller "
             f"Abfall noch Flachheit belegt. Mehr Ziehungen oder mehr "
             f"Qubitzahlen noetig.")
    print(f"  {v}")

    # ---- Bezug zum Konzentrationsprinzip ----
    print()
    print("BEZUG ZUM KONZENTRATIONSPRINZIP")
    print("  Die Quantenkerne zeigten eine Halbierung der mittleren Aehnlichkeit")
    print("  je Qubit, also ebenfalls einen Faktor 2^-n. Liegt die Steigung hier")
    print(f"  nahe -1 (gemessen {d['slope']:+.3f}), ist das dieselbe Skalierung an")
    print("  einem voellig anderen Mechanismus -- zwei unabhaengige Messungen")
    print("  desselben Prinzips statt zweier Einzelbefunde.")
    print("=" * 82)

    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001 -- bewusst breit
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print("  Rohdaten werden trotzdem gespeichert. Auswertung nachholen mit:")
        print('    python -c "from quantum_sim.barren_plateaus import analyse; '
              "analyse('Q-005')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
