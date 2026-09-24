from __future__ import annotations

"""RESEARCH ID: T-001 -- Verschiebt der Umbau den Uebergangspunkt?

DIE FRAGE
Am 5. August wurde gemessen: Die Butterfly-Architektur ist unterhalb von rund
8.000 Dimensionen deutlich langsamer als eine dichte Schicht gleicher Breite,
und erst ab 16.384 schneller (Faktor 1,99, Median aus sieben Messreihen, 7/7).
Bei d=4096 braucht sie 340-fach weniger Arithmetik und ist trotzdem 5-fach
langsamer.

`butterfly_fast.py` wendet die drei verhaltenserhaltenden Umschreibungen R1
(Permutationen zusammenfassen), R5 (Gatter gesammelt berechnen) und R6 (Auslese
in einem Durchlauf) an. Verifiziert auf 4,0e-7 ueber 30 Konfigurationen,
Gradienten identisch.

VORREGISTRIERUNG
  PRIMAER  Der Uebergangspunkt -- die kleinste Breite, ab der die Schaltung
           schneller ist als eine dichte Schicht gleicher Breite -- sinkt von
           16.384 auf <= 4.096.
  KONTROLLE `ButterflyQuantumLayer` unveraendert, gleiche Hardware, gleiche
           Methodik, im selben Lauf abwechselnd gemessen.
  AUFLOESUNGSPRUEFUNG  Bei d=16.384 muss `ButterflyFast` mindestens so schnell
           sein wie `ButterflyQuantumLayer`. Ist er dort langsamer, misst der
           Aufbau nicht Fusion, sondern etwas anderes -- dann ist kein Ergebnis
           interpretierbar.

  T1  Uebergang <= 4.096  -> die Luecke war handwerklich; die Architektur ist im
      Bereich echter FFN-Breiten einsetzbar.
  T0  Uebergang > 8.192   -> die Fusion traegt nicht weit genug; die Luecke ist
      naeher an fundamental als an handwerklich. Auch das ist ein Ergebnis, und
      es beendet die T-Serie nach dem Ausstiegskriterium der Roadmap.
  TU  dazwischen          -> erreichten Uebergangspunkt berichten, kein Urteil.

  ERWARTUNG, vorher notiert: R1/R5/R6 senken die Zahl der Durchlaeufe ueber den
  Zustand von rund 89 auf rund 46, also etwa Faktor zwei. Die L*n
  Gatteranwendungen bleiben unfusioniert -- sie brauchen einen echten Kernel.
  Faktor zwei verschiebt den Uebergang um etwa eine Verdopplung der Breite,
  also von 16.384 auf ungefaehr 8.192. **Damit ist T1 nach eigener Rechnung
  eher unwahrscheinlich und TU der erwartete Ausgang.** Das wird vorher
  aufgeschrieben, damit hinterher niemand die Erwartung nachjustiert.

METHODIK -- uebernommen vom 5. August, unveraendert
Sieben unabhaengige Messreihen. Innerhalb jeder Reihe wird ABWECHSELND
gemessen, damit Drift oder Fremdlast beide Arme gleich trifft. Der Faktor wird
JE REIHE gepaart gebildet, nicht als Verhaeltnis zweier Mediane -- so entstand
am 5. August der Unterschied zwischen 1,57x (Einzelmessung) und 1,99x (sieben
Reihen). Einfaedig, wie damals.

NICHT GLEICHZEITIG MIT ANDEREN SCHWEREN LAEUFEN STARTEN. Q-009s Zeitspalte ist
genau daran unbrauchbar geworden.

Run: python -m quantum_sim.kernel_messung
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from qire_ai.benchmark import set_seed
from qire_ai.result_store import Recorder

from .butterfly_fast import ButterflyFast
from .butterfly_layer import ButterflyQuantumLayer

QUBITS = (10, 11, 12, 13, 14)      # d = 1.024 bis 16.384
LAGEN = 5
BATCH = 64
REIHEN = 7
WIEDERHOLUNGEN = 3
WARMUP = 2
ARME = ("referenz", "schnell", "dicht")


def _baue(arm: str, n: int):
    d = 2 ** n
    if arm == "referenz":
        return ButterflyQuantumLayer(n, LAGEN, entangle=True, tie_weights=True)
    if arm == "schnell":
        return ButterflyFast(n, LAGEN, entangle=True)
    if arm == "dicht":
        return nn.Linear(d, d, bias=False)
    raise ValueError(arm)


def _zeit(modell, x: torch.Tensor) -> float:
    """Median mehrerer Vorwaertslaeufe. Nur Inferenz -- der 1,99x-Vorteil vom
    5. August galt fuer Inferenz, im Training schrumpfte er auf 1,18x."""
    with torch.no_grad():
        for _ in range(WARMUP):
            modell(x)
        zeiten = []
        for _ in range(WIEDERHOLUNGEN):
            t0 = time.perf_counter()
            modell(x)
            zeiten.append(time.perf_counter() - t0)
    return float(np.median(zeiten))


def run(qubits: Sequence[int] = QUBITS) -> Recorder:
    rec = Recorder(
        experiment_id="T-001",
        description=("Verschiebt der Umbau R1/R5/R6 den Uebergangspunkt der "
                     "Butterfly-Architektur? Sieben Messreihen, abwechselnd, "
                     "gepaart je Reihe. Inferenz, einfaedig."),
        config={"qubits": list(qubits), "lagen": LAGEN, "batch": BATCH,
                "reihen": REIHEN, "wiederholungen": WIEDERHOLUNGEN,
                "warmup": WARMUP, "arme": list(ARME), "threads": 1,
                "messgroesse": "inferenz_sekunden",
                "primary": "Uebergangspunkt sinkt von 16384 auf <= 4096",
                "resolution_check": "schnell darf bei d=16384 nicht langsamer "
                                    "sein als referenz",
                "methodik": "wie 5. August: 7 Reihen, abwechselnd, Faktor je "
                            "Reihe gepaart",
                "erwartung_vorher": ("TU -- rund 89 auf rund 46 Durchlaeufe, also "
                                     "etwa Faktor 2, was den Uebergang eher auf "
                                     "8192 als auf 4096 schiebt")},
    )
    print("\n" + "=" * 94)
    print("T-001 -- ZEITMESSUNG")
    print(f"{REIHEN} Reihen, abwechselnd, Batch {BATCH}, {LAGEN} Lagen, einfaedig")
    print("=" * 94)
    for n in qubits:
        d = 2 ** n
        set_seed(5000 + n)                      # setzt auch num_threads(1)
        try:
            modelle = {arm: _baue(arm, n) for arm in ARME}
        except (RuntimeError, MemoryError) as exc:
            print(f"  d={d:>6}: nicht baubar ({type(exc).__name__})")
            rec.add(qubits=n, dim=d, arm="dicht", baubar=False,
                    reihe=-1, sekunden=float("nan"))
            rec.save()
            continue
        x = torch.randn(BATCH, d)
        print(f"\n  d = {d:,} ({n} Qubits)")
        for reihe in range(REIHEN):
            # ABWECHSELND innerhalb der Reihe: jede Reihe misst alle Arme
            # unmittelbar nacheinander, damit Drift alle gleich trifft.
            for arm in ARME:
                s = _zeit(modelle[arm], x)
                rec.add(qubits=n, dim=d, arm=arm, baubar=True, reihe=reihe,
                        sekunden=s)
            zeilen = [r for r in rec.records
                      if r["qubits"] == n and r["reihe"] == reihe]
            werte = {r["arm"]: r["sekunden"] for r in zeilen}
            print(f"    Reihe {reihe + 1}/{REIHEN}: "
                  + "  ".join(f"{a}={werte[a] * 1000:.1f}ms" for a in ARME))
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    dims = sorted({int(r["dim"]) for r in records if r.get("baubar", True)})
    out: Dict[str, object] = {}

    def reihen_faktoren(d: int, zaehler: str, nenner: str) -> np.ndarray:
        """Faktor JE REIHE gepaart -- nicht Verhaeltnis zweier Mediane."""
        f = []
        for reihe in range(REIHEN):
            z = [r["sekunden"] for r in records if int(r["dim"]) == d
                 and r["arm"] == zaehler and r["reihe"] == reihe]
            nn_ = [r["sekunden"] for r in records if int(r["dim"]) == d
                   and r["arm"] == nenner and r["reihe"] == reihe]
            if z and nn_ and nn_[0] > 0:
                f.append(z[0] / nn_[0])
        return np.array(f)

    print()
    print("=" * 94)
    print("T-001 | Vorteil gegen die dichte Schicht (>1 heisst schneller als dicht)")
    head = (f'  {"d":>8} {"referenz":>10} {"schnell":>10} {"Gewinn":>8} '
            f'{"Reihen schnell>dicht":>21}')
    print(head); print("  " + "-" * (len(head) - 2))
    uebergang = {"referenz": None, "schnell": None}
    for d in dims:
        zeilen = {}
        for arm in ("referenz", "schnell"):
            f = reihen_faktoren(d, "dicht", arm)      # dicht/arm: >1 = arm schneller
            zeilen[arm] = float(np.median(f)) if len(f) else float("nan")
            if zeilen[arm] > 1.0 and uebergang[arm] is None:
                uebergang[arm] = d
        siege = int((reihen_faktoren(d, "dicht", "schnell") > 1.0).sum())
        gewinn = zeilen["schnell"] / zeilen["referenz"] if zeilen["referenz"] else float("nan")
        print(f'  {d:>8,} {zeilen["referenz"]:>10.3f} {zeilen["schnell"]:>10.3f} '
              f'{gewinn:>8.2f}x {f"{siege}/{REIHEN}":>21}')
        out[f"d{d}"] = {"faktor_referenz": zeilen["referenz"],
                        "faktor_schnell": zeilen["schnell"],
                        "gewinn": gewinn, "reihen_schneller": siege}

    # --- Auflösungspruefung: schnell darf bei der groessten Breite nicht
    #     langsamer sein als referenz. Verlangt wird das, was angemeldet ist. ---
    groesste = max(dims)
    rel = reihen_faktoren(groesste, "referenz", "schnell")   # >1 = schnell besser
    res_median = float(np.median(rel)) if len(rel) else float("nan")
    res_ok = res_median >= 1.0
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if res_ok else 'NICHT BESTANDEN'}: "
            f"bei d={groesste:,} ist schnell um Faktor {res_median:.2f} gegenueber "
            f"referenz."
            + ("" if res_ok else " Der Umbau macht die Schicht dort LANGSAMER -- "
               "dann misst der Aufbau nicht Fusion, und kein Ergebnis oben ist "
               "interpretierbar."))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(res_ok), "median": res_median,
                               "text": ctxt}

    u_neu, u_alt = uebergang["schnell"], uebergang["referenz"]
    print()
    if not res_ok:
        v = "TU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif u_neu is None:
        v = (f"T0 -- der Uebergangspunkt liegt auch nach dem Umbau oberhalb der "
             f"groessten vermessenen Breite ({groesste:,}). Die Fusion traegt "
             f"nicht weit genug; nach dem Ausstiegskriterium der Roadmap endet "
             f"die T-Serie hier, und der Befund lautet: die Luecke ist nicht "
             f"allein handwerklich.")
    elif u_neu <= 4096:
        v = (f"T1 -- der Uebergangspunkt sinkt auf {u_neu:,} (Kontrolle: "
             f"{u_alt if u_alt else '>' + str(groesste)}). Die Luecke war "
             f"handwerklich, und die Architektur ist damit im Bereich echter "
             f"FFN-Breiten einsetzbar. Das ist die Vorbedingung fuer Phase 3.")
    elif u_neu > 8192:
        v = (f"T0 -- der Uebergang liegt bei {u_neu:,}, also ueber 8.192. Die "
             f"Fusion allein traegt nicht weit genug. Ausstiegskriterium der "
             f"Roadmap greift.")
    else:
        v = (f"TU -- der Uebergang sinkt auf {u_neu:,} (Kontrolle: "
             f"{u_alt if u_alt else '>' + str(groesste)}), bleibt aber ueber "
             f"4.096. Verbesserung belegt, Ziel verfehlt. Das ist der vorher "
             f"erwartete Ausgang: R1/R5/R6 lassen die Gatteranwendungen "
             f"unfusioniert, und die sind der Hauptposten.")
    print(f"BEWERTUNG: {v}")
    print("=" * 94)
    out["uebergang_schnell"], out["uebergang_referenz"] = u_neu, u_alt
    out["verdict"] = v
    return out


if __name__ == "__main__":
    from .butterfly_fast import gradientenpruefung, verifiziere
    if not (verifiziere() and gradientenpruefung()):
        print("\nVerifikation durchgefallen -- keine Zeitmessung.")
        raise SystemExit(1)
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
