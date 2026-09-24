"""RESEARCH ID: T-003-vorlauf -- Zeitmessung des Rechen-Buendlers, CPU und GPU.

Ein Skript fuer beide Geraete, damit die Methodik nicht auseinanderlaeuft.
Vorregistrierung: `ENTWURF_T003_BUENDLER.md`. Methodik von T-001 unveraendert:
sieben Reihen, innerhalb jeder Reihe alle Arme unmittelbar nacheinander in
fester Reihenfolge, je Messung Median aus 3 Laeufen nach 2 Aufwaermlaeufen,
Faktor JE REIHE gepaart. Auf der GPU wird vor und nach jedem Lauf
`torch.cuda.synchronize()` gerufen -- ohne das misst man nur den Kernelstart.

    Aufruf (aus dem Ordner A):
      python server_setup/gpu_benchmark_buendler.py --vortest
      python server_setup/gpu_benchmark_buendler.py --geraet cpu --batch 64 --modus vorwaerts
      python server_setup/gpu_benchmark_buendler.py --geraet cuda --batch 256 --modus training

VOR JEDER MESSUNG: Verifikation auf dem Zielgeraet (alle Arme gegen
ButterflyFast auf der CPU). Faellt sie durch, bricht das Skript ab.
NICHT PARALLEL ZU ANDEREN SCHWEREN LAEUFEN. Das Skript prueft das selbst
(psutil) und bricht ab, ausser `--fremdlast-erlaubt` -- dann werden die Zeilen
als nicht verwertbar markiert.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qire_ai.result_store import Recorder  # noqa: E402
from quantum_sim.buendler import Buendler  # noqa: E402
from quantum_sim.butterfly_fast import ButterflyFast  # noqa: E402

ID = "T-003-vorlauf"
LAGEN, REIHEN, WIEDERHOLUNGEN, WARMUP = 5, 7, 3, 2
# Reihenfolge = Messreihenfolge innerhalb jeder Reihe (fest, wie T-001)
ARME = ("referenz", "kron2", "kron3", "kron2_gefaltet", "kron2_ohne", "reell_kron2",
        "vorkompiliert", "dicht")
PRIMAER_ARM = "kron2"


def baue(arm: str, n: int) -> nn.Module:
    if arm == "dicht":
        return nn.Linear(2 ** n, 2 ** n, bias=False)
    if arm == "referenz":
        return ButterflyFast(n, LAGEN, entangle=True)
    if arm == "kron2_ohne":
        return Buendler(n, LAGEN, entangle=False, strategie="kron2")
    return Buendler(n, LAGEN, entangle=True, strategie=arm)


def zulaessig(arm: str, n: int, a) -> bool:
    if arm == "dicht":
        return n <= a.max_n_dicht
    if arm == "referenz":
        return n <= a.max_n_referenz
    if arm == "vorkompiliert":
        return a.modus == "vorwaerts" and n <= a.max_n_vorkompiliert
    return True


def sync(geraet: str) -> None:
    if geraet == "cuda":
        torch.cuda.synchronize()


def zeit(modell: nn.Module, x: torch.Tensor, modus: str, geraet: str) -> float:
    """Median aus WIEDERHOLUNGEN Laeufen nach WARMUP Aufwaermlaeufen."""
    def lauf():
        if modus == "vorwaerts":
            with torch.no_grad():
                modell(x)
        else:
            modell.zero_grad(set_to_none=True)
            modell(x).sum().backward()
    for _ in range(WARMUP):
        lauf()
    zeiten = []
    for _ in range(WIEDERHOLUNGEN):
        sync(geraet)
        t0 = time.perf_counter()
        lauf()
        sync(geraet)
        zeiten.append(time.perf_counter() - t0)
    return float(np.median(zeiten))


def fremde_last() -> List[str]:
    """Andere Python-Prozesse mit nennenswerter CPU-Last (psutil, falls da)."""
    try:
        import psutil
    except ImportError:
        return ["psutil fehlt -- Fremdlast NICHT geprueft"]
    eigen, gefunden = os.getpid(), []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == eigen or "python" not in (p.info["name"] or "").lower():
                continue
            last = p.cpu_percent(interval=0.5)
            if last > 20:
                gefunden.append(f"{p.info['pid']}: {' '.join(p.info['cmdline'] or [])[:100]} ({last:.0f}%)")
        except Exception:                                   # noqa: BLE001
            continue
    return gefunden


def geraete_pruefung(geraet: str, qubits, tol: float = 1e-5) -> bool:
    """Jeder Arm auf dem Zielgeraet gegen ButterflyFast auf der CPU.
    Die volle Verifikation (n=4..14, Gradienten, float64) steht in
    `python -m quantum_sim.buendler`; hier wird nur geprueft, dass die
    Geraetebibliothek (cuBLAS/MKL) dieselbe Funktion rechnet."""
    ok, schlimmst = True, 0.0
    for n in qubits:
        if n > 12:
            continue
        torch.manual_seed(n)
        ref = ButterflyFast(n, LAGEN, entangle=True)
        ref_ohne = ButterflyFast(n, LAGEN, entangle=False)
        x = torch.randn(9, 2 ** n)
        for arm in ARME:
            if arm == "dicht":
                continue
            m = baue(arm, n)
            r = ref_ohne if arm == "kron2_ohne" else ref
            with torch.no_grad():
                if arm == "reell_kron2":
                    r = ButterflyFast(n, LAGEN, entangle=True)
                    r.theta.copy_(ref.theta)
                    r.theta[..., 1] = 0.0
                    m.theta.copy_(r.theta[..., :1])
                else:
                    m.theta.copy_(r.theta)
                erwartet = r(x)
                m = m.to(geraet)
                erhalten = m(x.to(geraet)).cpu()
            dev = float((erwartet - erhalten).abs().max())
            schlimmst = max(schlimmst, dev)
            if dev > tol:
                ok = False
                print(f"  GERAETEPRUEFUNG FEHLER n={n} {arm}: {dev:.2e}")
    print(f"  Geraetepruefung auf {geraet}: groesste Abweichung {schlimmst:.2e} -> "
          f"{'OK' if ok else 'DURCHGEFALLEN'}")
    return ok


def run(a) -> Recorder:
    geraet = a.geraet
    if geraet == "cpu":
        torch.set_num_threads(a.threads)
    info = torch.cuda.get_device_name(0) if geraet == "cuda" else f"cpu, {a.threads} Threads"
    rec = Recorder(experiment_id=ID, description=(
        "T-003-vorlauf: Rechen-Buendler (kron2/kron3/gefaltet/vorkompiliert/reell) gegen "
        "ButterflyFast und dicht. 7 Reihen, fest abwechselnd, Faktor je Reihe gepaart."),
        config={"geraet": geraet, "geraet_info": info, "threads": a.threads,
                "batch": a.batch, "modus": a.modus, "lagen": LAGEN, "reihen": REIHEN,
                "wiederholungen": WIEDERHOLUNGEN, "warmup": WARMUP, "arme": list(ARME),
                "qubits": list(a.qubits), "max_n_dicht": a.max_n_dicht,
                "max_n_referenz": a.max_n_referenz,
                "max_n_vorkompiliert": a.max_n_vorkompiliert,
                "winkel": "eingefroren (Inferenz)" if a.modus == "vorwaerts" else "trainierbar",
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "fremdlast": a.fremdlast_liste, "verwertbar": not a.fremdlast_liste,
                "vorregistrierung": "ENTWURF_T003_BUENDLER.md", "torch": torch.__version__})
    start = time.time()
    print(f"\nT-003-vorlauf | {info} | Batch {a.batch} | {a.modus} | L={LAGEN}")
    for n in a.qubits:
        if (time.time() - start) / 60 > a.max_minuten:
            print(f"  Zeitgrenze {a.max_minuten} min erreicht -- Abbruch vor n={n}.")
            rec.config["abgebrochen_vor_n"] = n
            break
        d = 2 ** n
        torch.manual_seed(5000 + n)
        arme = [arm for arm in ARME if zulaessig(arm, n, a)]
        modelle = {}
        for arm in arme:
            try:
                m = baue(arm, n).to(geraet)
                if a.modus == "vorwaerts":
                    m.requires_grad_(False)                  # R5: Winkel eingefroren
                modelle[arm] = m
            except (RuntimeError, MemoryError) as exc:
                print(f"  d={d}: {arm} nicht baubar ({type(exc).__name__})")
                rec.add(qubits=n, dim=d, arm=arm, baubar=False, reihe=-1, sekunden=float("nan"))
        x = torch.randn(a.batch, d, device=geraet)
        print(f"\n  d = {d:,} ({n} Qubits)")
        for reihe in range(REIHEN):
            werte = {}
            for arm, m in modelle.items():
                werte[arm] = zeit(m, x, a.modus, geraet)
                rec.add(qubits=n, dim=d, arm=arm, baubar=True, reihe=reihe,
                        sekunden=werte[arm])
            print(f"    Reihe {reihe + 1}/{REIHEN}: "
                  + " ".join(f"{k}={v * 1000:.2f}ms" for k, v in werte.items()), flush=True)
        del modelle
        rec.save()                                           # Regel 5: nach jedem Abschnitt
    return rec


# ================================================================ Auswertung
def faktoren(records, d: int, zaehler: str, nenner: str) -> np.ndarray:
    """zaehler/nenner JE REIHE gepaart; dicht/arm > 1 heisst: arm schneller."""
    f = []
    for reihe in range(REIHEN):
        z = [r["sekunden"] for r in records if r["dim"] == d and r["arm"] == zaehler
             and r["reihe"] == reihe]
        nn_ = [r["sekunden"] for r in records if r["dim"] == d and r["arm"] == nenner
               and r["reihe"] == reihe]
        if z and nn_ and nn_[0] > 0:
            f.append(z[0] / nn_[0])
    return np.array(f)


def analyse(records, config) -> Dict[str, object]:
    dims = sorted({r["dim"] for r in records if r.get("baubar", True)})
    arme = [a for a in ARME if a != "dicht" and any(r["arm"] == a for r in records)]
    out: Dict[str, object] = {"faktor_gegen_dicht": {}, "uebergang": {}}
    print("\n" + "=" * 100)
    print("Faktor gegen dicht (>1 = schneller als dicht), Median der gepaarten Reihenfaktoren")
    print("  " + f'{"d":>7}' + "".join(f"{x:>15}" for x in arme))
    for d in dims:
        zeile = f"  {d:>7}"
        for arm in arme:
            f = faktoren(records, d, "dicht", arm)
            med = float(np.median(f)) if len(f) else float("nan")
            out["faktor_gegen_dicht"].setdefault(arm, {})[d] = med
            if len(f) and med > 1.0 and out["uebergang"].get(arm) is None:
                out["uebergang"][arm] = d
            zeile += f"{med:>15.3f}" if len(f) else f"{'-':>15}"
        print(zeile)
    print("\nGewinn gegen referenz (ButterflyFast), >1 = schneller")
    print("  " + f'{"d":>7}' + "".join(f"{x:>15}" for x in arme if x != "referenz"))
    for d in dims:
        zeile = f"  {d:>7}"
        for arm in arme:
            if arm == "referenz":
                continue
            f = faktoren(records, d, "referenz", arm)
            zeile += f"{float(np.median(f)):>15.3f}" if len(f) else f"{'-':>15}"
            if len(f):
                out.setdefault("gewinn_gegen_referenz", {}).setdefault(arm, {})[d] = float(np.median(f))
        print(zeile)
    for arm in arme:
        out["uebergang"].setdefault(arm, None)
    out["verdict"] = urteil(records, config, out)
    print("\nBEWERTUNG: " + out["verdict"])
    print("=" * 100)
    return out


def urteil(records, config, out) -> str:
    """Die in ENTWURF_T003_BUENDLER.md angemeldeten Ausgaenge, woertlich kodiert."""
    primaerlauf = (config.get("geraet") == "cpu" and config.get("batch") == 64
                   and config.get("modus") == "vorwaerts" and config.get("threads") == 1)
    if not config.get("verwertbar", True):
        return "NICHT VERWERTBAR -- unter Fremdlast gemessen, nur Rauchtest."
    groesste = 16384
    rel = faktoren(records, groesste, "referenz", PRIMAER_ARM)
    if not len(rel):
        return "TU -- Aufloesungspruefung nicht durchfuehrbar (keine Messung bei d=16384)."
    res = float(np.median(rel))
    if res < 1.0:
        return (f"TU -- Aufloesungspruefung NICHT BESTANDEN: kron2 ist bei d=16384 um Faktor "
                f"{res:.2f} gegenueber referenz. Der Aufbau misst dann nicht Buendelung; "
                f"kein Ergebnis dieses Laufs ist interpretierbar.")
    u = out["uebergang"].get(PRIMAER_ARM)
    tag = "" if primaerlauf else " [Nebenlauf -- Urteil nur beschreibend, nicht die Primaerhypothese]"
    if u is None:
        return (f"T0 -- kron2 schlaegt dicht bis d=16384 nirgends (Aufloesung bestanden, "
                f"{res:.2f}x gegen referenz). Buendelung zu GEMMs genuegt nicht.{tag}")
    if u <= 4096:
        return (f"T1 -- Uebergang kron2 gegen dicht bei d={u:,} (T-001: 16.384). Die Luecke "
                f"war Ausfuehrung, nicht Struktur. Aufloesung {res:.2f}x.{tag}")
    if u > 8192:
        return (f"T0 -- Uebergang kron2 erst bei d={u:,}, ueber 8.192. Buendelung verschiebt "
                f"den Schnittpunkt nicht genug. Aufloesung {res:.2f}x.{tag}")
    return (f"TU -- Uebergang kron2 bei d={u:,}: besser als T-001, Ziel 4.096 verfehlt. "
            f"Aufloesung {res:.2f}x.{tag}")


def vortest() -> bool:
    """Jeden Ausgang auf synthetischen Daten erzwingen (CLAUDE.md Abschnitt 5),
    je 5 Rauschziehungen; geprueft wird der Verzweigungstext, nicht nur die Marke."""
    cfg = {"geraet": "cpu", "batch": 64, "modus": "vorwaerts", "threads": 1, "verwertbar": True}
    dims = [2 ** n for n in range(8, 15)]

    def daten(uebergang_kron2: Optional[int], kron2_zu_ref: float, seed: int):
        # Erster Vortest (24.09.) fiel durch -- am GENERATOR, nicht an urteil():
        # kron2 lag bei 16384 langsamer als referenz (loeste die Aufloesungspruefung
        # aus), und im Fall "nirgends" war referenz dort schneller als dicht.
        rng = np.random.default_rng(seed)
        recs = []
        for d in dims:
            dicht = 1e-9 * d * d
            ref_vorteil = 2.0 if uebergang_kron2 is not None or kron2_zu_ref < 1 else 0.5
            ref = dicht / (0.3 if d < 16384 else ref_vorteil)
            k2 = ref / kron2_zu_ref
            if uebergang_kron2 is not None:                 # Faktor dicht/kron2 kreuzt 1 genau dort
                k2 = dicht / (1.3 if d >= uebergang_kron2 else 0.7)
                if d == 16384:
                    k2 = min(k2, ref / 1.2)                 # Aufloesungspruefung besteht
            for reihe in range(REIHEN):
                for arm, t in (("referenz", ref), ("kron2", k2), ("dicht", dicht)):
                    recs.append({"dim": d, "arm": arm, "reihe": reihe, "baubar": True,
                                 "sekunden": t * float(np.exp(rng.normal(0, 0.05)))})
        return recs

    faelle = [("T1", 2048, 1.0, "T1 -- Uebergang kron2 gegen dicht bei d=2,048"),
              ("T1", 4096, 1.0, "T1 -- Uebergang kron2 gegen dicht bei d=4,096"),
              ("TU", 8192, 1.0, "TU -- Uebergang kron2 bei d=8,192"),
              ("T0", 16384, 1.0, "T0 -- Uebergang kron2 erst bei d=16,384"),
              ("T0", None, 1.5, "T0 -- kron2 schlaegt dicht bis d=16384 nirgends"),
              ("TU", None, 0.5, "Aufloesungspruefung NICHT BESTANDEN")]
    ok = True
    for marke, u, k2ref, text in faelle:
        for seed in range(5):
            recs = daten(u, k2ref, seed)
            if u is not None and k2ref == 1.0:
                pass
            res_out = {"uebergang": {}}
            for d in dims:
                f = faktoren(recs, d, "dicht", "kron2")
                if np.median(f) > 1 and "kron2" not in res_out["uebergang"]:
                    res_out["uebergang"]["kron2"] = d
            v = urteil(recs, cfg, res_out)
            if not (v.startswith(marke) and text in v):
                ok = False
                print(f"  VORTEST FEHLER ({marke}, u={u}, seed {seed}): {v}")
    v = urteil([], {**cfg, "verwertbar": False}, {"uebergang": {}})
    ok &= v.startswith("NICHT VERWERTBAR")
    print(f"  Vortest: {len(faelle)} Ausgaenge x 5 Ziehungen + Fremdlastfall -> "
          f"{'BESTANDEN' if ok else 'DURCHGEFALLEN'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geraet", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--modus", default="vorwaerts", choices=("vorwaerts", "training"))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--qubits", default="8-16")
    p.add_argument("--max-n-dicht", type=int, default=14)
    p.add_argument("--max-n-referenz", type=int, default=14)
    p.add_argument("--max-n-vorkompiliert", type=int, default=13)
    p.add_argument("--max-minuten", type=float, default=15.0)
    p.add_argument("--fremdlast-erlaubt", action="store_true")
    p.add_argument("--vortest", action="store_true")
    a = p.parse_args()
    if a.vortest:
        return 0 if vortest() else 1
    lo, hi = (int(v) for v in a.qubits.split("-"))
    a.qubits = list(range(lo, hi + 1))
    a.fremdlast_liste = fremde_last() if a.geraet == "cpu" else []
    if a.fremdlast_liste:
        print("FREMDLAST:", *a.fremdlast_liste, sep="\n  ")
        if not a.fremdlast_erlaubt:
            print("Abbruch. Messung nur ohne Fremdlast (oder --fremdlast-erlaubt fuer Rauchtest).")
            return 2
    if a.geraet == "cuda" and not torch.cuda.is_available():
        print("CUDA nicht verfuegbar."); return 2
    if not (vortest() and geraete_pruefung(a.geraet, a.qubits)):
        print("Vortest oder Geraetepruefung durchgefallen -- keine Zeitmessung."); return 1
    rec = run(a)
    try:
        analysis = analyse(rec.records, rec.config)
    except Exception as exc:                                # noqa: BLE001
        print(f"Auswertung fehlgeschlagen: {exc!r}")
        analysis = {"error": repr(exc)}
    rec.save(analysis=analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
