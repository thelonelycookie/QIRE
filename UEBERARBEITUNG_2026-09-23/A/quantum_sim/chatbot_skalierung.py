"""RESEARCH ID: T-006 -- Adapter-Skalierung L = 20/40 (Nachfolger von T-005).

Baut auf T-005 (`chatbot_kurve.py`) und T-004 (`chatbot_adapter.py`) auf; dort wird
NICHTS geaendert. Modell, Daten, Aufteilung, Drehung, Drehbank, Verlust, Auswertung
und die Zelle sind dieselben (`chatbot_kurve.zelle`). Hier kommen nur neue Arme
(L20, L40, L10 mit 600 Schritten), Seeds, Hypothesen und Auswertung hinzu.
Vorregistrierung ausfuehrlich: ENTWURF_T006_SKALIERUNG.md (maßgeblich ist dieser
Modulkopf, Stand Commit). Herleitung: INSTITUT/recherche/2026-10-05_vertiefung/v5_lie_skalierung.md §6.

FRAGE  Wie waechst der Gewinn des QIRE-Adapters bei L = 10 -> 20 -> 40 Lagen, und
  kommt qire_L40 (7.680 Winkel) an lora_r1 (30.720) heran? Die Lie-Algebra ist so(256)
  (Jacobi-Rang min(8 L, 32.640), L* = 4.080): bei L <= 40 begrenzt NICHT die Gruppe.

ARME (q_proj und v_proj der 12 Lagen = 24 Matrizen; CNOT-Ring; SCHRITTE = 300)
    basis         kein Training                                            0
    qire_L10      Bruecke und POSITIVKONTROLLE (T-005: 2,4115)          1.920
    qire_L20      20 Lagen                                              3.840
    qire_L40      40 Lagen                                              7.680
    lora_r1       Kontrolle (wie T-004/T-005)                          30.720
    qire_L10_600  qire_L10 mit 600 statt 300 Schritten (OPTIONAL,       1.920
                  eigener Arm, ausserhalb der Holm-Familie, beschreibend)
VORREGISTRIERUNG (vor jeder T-006-Zahl; Aenderungen erst ins Tagebuch)
  Seeds FRISCH: Messung 60001-60008, Tuning 60401/60402 (disjunkt zu T-004 40001-40008/
  40401-40402 und T-005 50001-50008/50401-50402; `pruefe_theorie` prueft das).
  Lernraten: qire_L10, lora_r1 uebernommen aus T-005 (1e-3, innen, bitgleiche Zelle);
  qire_L10_600 uebernimmt die Rate von qire_L10; qire_L20/L40 getunt, Gitter 3e-4/1e-3/3e-3,
  Verlaengerung (x3 / :3, max. 3 Schritte) wie T-005 (`sweep_lr`).
  Familie, Holm ueber GENAU 3:
    K1  Monotonie L10 -> L20 -> L40: exakter Page-Trendtest (k = 3, n = 8) + IUT ueber
        beide Nachbarstufen (Wilcoxon einseitig, max p < 0,05/3).
          STRENG_MONOTON / WAECHST_MIT_PLATEAU / KEIN_TREND
    K2  qire_L40 - lora_r1, zweiseitig + TOST, Grenze +-20 % x |Gewinn lora_r1 gg. basis|
          QIRE_BESSER / GLEICHWERTIG / LORA_BESSER / UNENTSCHIEDEN
    K3  Form des Gewinns G(L) = Verlust(basis) - Verlust(L), Exponent
        b = log(G40 / G10) / log 4 (Potenzgesetz G ~ L^b; b = 0,5 = Wurzelgesetz aus
        isotropem linearem Regime). H0: b = 0,5, zweiseitig, Bootstrap ueber Seeds
        (B = 10.000, rng-Seed 20261006, gepaart: gleiche Seed-Ziehung in allen Armen,
        p = 2 min(P(b* <= 0,5), P(b* >= 0,5)) mit (k+1)/(B+1)). Beschreibend dazu:
        G20/G10 und G40/G20 gegen 2^0,5 = 1,414 mit 95 %-Bootstrap-Intervall.
          SAETTIGT_UNTER_WURZEL / STEIGT_UEBER_WURZEL / OFFEN_ZU_WURZEL
        (OFFEN_ZU_WURZEL heisst nur: Wurzelgesetz nicht verworfen, KEINE Gleichwertigkeit.)
  Positivkontrolle: Mittel qire_L10 (8 Seeds) = 2,4115 +- 0,003 (T-005, andere Seeds).
  Rangfolge: HU_UNVOLLSTAENDIG -> HU_POSKONTROLLE -> HU_AUFLOESUNG -> Urteil.
    Aufloesung wie T-004/T-005 (lora_r1 trennt sich von basis, Wilcoxon zweiseitig p < 0,05).
  Erreichbarkeit (Regel 2): Page min (1/3!)^8 = 6e-7; IUT min 0,0039 < 0,0167; Bootstrap
    min p = 1/(B+1) = 1e-4; Wilcoxon zweiseitig min 0,0078 x 3 (Holm) <= 0,023.
  Optional beschreibend (ohne Holm): qire_L10_600 - qire_L10 gegen qire_L20 - qire_L10:
    BUDGETGRENZE (L10_600 <= L20 im Mittel) / STRUKTURGRENZE_EHER (Zuwachs < Haelfte des
    L20-Zuwachses) / GEMISCHT.

Run (aus dem Ordner A):
    python -m quantum_sim.chatbot_skalierung vortest          # Pflicht (Modell noetig)
    python -m quantum_sim.chatbot_skalierung vortest_statistik  # ohne Modell
    python -m quantum_sim.chatbot_skalierung zeit [threads] [schritte]
    python -m quantum_sim.chatbot_skalierung lauf             # Malte entscheidet
    python -m quantum_sim.chatbot_skalierung auswertung [ID|Pfad]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zlib
from collections import Counter
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from qire_ai.equivalence import holm
from qire_ai.result_store import RESULTS_DIR, Recorder

from . import chatbot_adapter as ca
from . import chatbot_kurve as ck
from .chatbot_adapter import BREITE, N_ABSTIMM, N_VAL, _reihe, _schl, _vergleich, _wilcoxon, daten
from .kernaussage import sweep_lr

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt
# ---------------------------------------------------------------------------
LAUF_ID = "T-006"
LAGEN = (10, 20, 40)
MIT_L10_600 = True
SCHRITTE = ca.SCHRITTE                               # 300
SCHRITTE_LANG = 600
QIRE = {f"qire_L{l}": l for l in LAGEN}
QIRE_LANG = {"qire_L10_600": 10} if MIT_L10_600 else {}
ARME = ("basis", *QIRE, "lora_r1", *QIRE_LANG)
# Reihenfolge = Wichtigkeit: Positivkontrolle zuerst (scheitert sie, ist der Lauf frueh erkennbar),
# dann K2 (L40, lora), dann K1 (L20), zuletzt der optionale Arm.
TRAINIERT = tuple(a for a in ("qire_L10", "lora_r1", "qire_L40", "qire_L20", "qire_L10_600") if a in ARME)
SCHRITTE_ARM = {a: (SCHRITTE_LANG if a in QIRE_LANG else SCHRITTE) for a in ARME}
ERWARTETE_PARAMETER = {"basis": 0, **{a: 24 * 8 * l for a, l in {**QIRE, **QIRE_LANG}.items()},
                       "lora_r1": ca.ERWARTETE_PARAMETER["lora_r1"]}
EVAL_SEEDS = tuple(range(60001, 60009))
TUNING_SEEDS = (60401, 60402)
LR_START = (3e-4, 1e-3, 3e-3)
T005_ROH_PATTERN = "T-005__20260930_084358_02.json"
T005_ROH = RESULTS_DIR / T005_ROH_PATTERN
LR_UEBERNOMMEN = {"qire_L10": 0.001, "lora_r1": 0.001}      # aus T-005 (config["lr"], innen)
LR_GEKOPPELT = {"qire_L10_600": "qire_L10"}                  # gleiche Rate wie der 300-Schritt-Arm
LR_GITTER = {a: LR_START for a in TRAINIERT if a not in LR_UEBERNOMMEN and a not in LR_GEKOPPELT}
ERWEITERUNGEN = 3
REL_GRENZE = ca.REL_GRENZE                           # 0,20
FAMILIE = ("K1", "K2", "K3")
ALPHA = 0.05
IUT_SCHWELLE = ALPHA / len(FAMILIE)
METRIK = "val_verlust"
B_NULL = 0.5                                         # Wurzelgesetz
BOOT_B, BOOT_SEED = 10000, 20261006
POS_SOLL, POS_TOL = 2.4115, 0.003                    # Positivkontrolle qire_L10
ADAPTER_DIR = RESULTS_DIR.parent / "t006_adapter"
ZWISCHEN_DIR = RESULTS_DIR.parent / "t006_zwischenstand"

# T-005-Zelle wiederverwenden: sie liest QIRE / ERWARTETE_PARAMETER des Moduls chatbot_kurve.
# Eintraege nur ERGAENZT (L20, L40, L10_600); alle T-005-Eintraege bleiben, T-005 laeuft in
# einem eigenen Prozess und liest seine Tupel LAGEN/ARME, nicht diese Dicts.
for _a, _l in {**QIRE, **QIRE_LANG}.items():
    ck.QIRE.setdefault(_a, _l)
    ck.ERWARTETE_PARAMETER.setdefault(_a, 24 * 8 * _l)


def zelle(arm: str, seed: int, lr: float, menge: str = "validierung", threads: int = 6,
          schritte: Optional[int] = None, zeitmessung: bool = False) -> Dict[str, object]:
    return ck.zelle(arm, seed, lr, menge, threads, SCHRITTE_ARM.get(arm, SCHRITTE) if schritte is None else schritte,
                    zeitmessung)


# ---------------------------------------------------------------------------
# Wiederaufnahme je Zelle
# ---------------------------------------------------------------------------
class Zustand(ca.Zustand):
    def __init__(self, lauf_id: str, konfig: Dict[str, object]):   # noqa: D107 -- nur anderer Pfad
        self.pfad = ZWISCHEN_DIR / f"{lauf_id}_wiederaufnahme.json"
        self.fingerabdruck = hashlib.sha256(json.dumps(konfig, sort_keys=True).encode()).hexdigest()[:16]
        self.tuning, self.lr, self.records = {}, {}, []
        if self.pfad.exists():
            d = json.loads(self.pfad.read_text(encoding="utf-8"))
            if d.get("fingerabdruck") != self.fingerabdruck:
                raise SystemExit(f"{self.pfad.name} gehoert zu einer ANDEREN Konfiguration. "
                                 "Nicht ueberschreiben -- umbenennen oder Konfiguration pruefen.")
            self.tuning, self.lr, self.records = d["tuning"], d["lr"], d["records"]
            print(f"  WIEDERAUFNAHME: {len(self.tuning)} Tuning-Zellen, {len(self.records)} Messzeilen")


def konfiguration() -> Dict[str, object]:
    return {"basis_t004": ca.konfiguration(), "arme": list(ARME), "lagen": list(LAGEN), "cnot": True,
            "trainiert": list(TRAINIERT), "schritte": SCHRITTE, "schritte_arm": SCHRITTE_ARM,
            "seeds": list(EVAL_SEEDS), "tuning_seeds": list(TUNING_SEEDS),
            "lr_gitter": {k: list(v) for k, v in LR_GITTER.items()}, "erweiterungen": ERWEITERUNGEN,
            "lr_uebernommen": LR_UEBERNOMMEN, "lr_gekoppelt": LR_GEKOPPELT, "lr_quelle": T005_ROH_PATTERN,
            "rel_grenze": REL_GRENZE, "familie": list(FAMILIE), "iut_schwelle": IUT_SCHWELLE,
            "b_null": B_NULL, "boot": [BOOT_B, BOOT_SEED], "pos": [POS_SOLL, POS_TOL],
            "parameter": ERWARTETE_PARAMETER}


def _adapter_pfad(lauf_id: str, arm: str, seed):
    return ADAPTER_DIR / lauf_id / f"{arm}_{seed}.pt"


def run(lauf_id: str = LAUF_ID, threads: int = 6) -> Recorder:
    konfig = konfiguration()
    z = Zustand(lauf_id, konfig)
    rec = Recorder(experiment_id=lauf_id,
                   description=("T-006: Adapter-Skalierung qire_L10/L20/L40 (CNOT-Ring) gegen lora_r1 auf "
                                "LLaeMmlein-120M; K1 Page-Trend, K2 qire_L40-lora_r1 TOST, K3 Exponent b "
                                "gegen 0,5 (Bootstrap); Positivkontrolle qire_L10 = 2,4115."),
                   config={**konfig, "metrik": METRIK, "torch_threads": threads,
                           "aufloesungspruefung": "lora_r1 trennt sich von basis (zweiseitig)",
                           "fingerabdruck": z.fingerabdruck})

    def sichern():
        rec.records, rec.config["lr"] = list(z.records), dict(z.lr)
        rec.save()

    if not z.gemessen("basis", EVAL_SEEDS[0]):
        r = zelle("basis", 0, 0.0, "validierung", threads)
        r.pop("_adapter")
        z.records += [{**r, "seed": s} for s in EVAL_SEEDS]    # deterministisch
        z.speichern()
        print(f"  basis {r[METRIK]:.4f} nats/Token ({r['tokens']} Antwort-Tokens)", flush=True)
    for arm in TRAINIERT:
        if arm not in z.lr and arm in LR_UEBERNOMMEN:
            z.lr[arm] = {"lr": LR_UEBERNOMMEN[arm], "rand": "innen", "quelle": T005_ROH_PATTERN}
            z.speichern()
            print(f"  {arm:<12} -> lr {LR_UEBERNOMMEN[arm]:g}  (aus T-005 uebernommen)", flush=True)
        if arm not in z.lr and arm in LR_GEKOPPELT:
            z.lr[arm] = {**z.lr[LR_GEKOPPELT[arm]], "quelle": LR_GEKOPPELT[arm]}
            z.speichern()
            print(f"  {arm:<12} -> lr {z.lr[arm]['lr']:g}  (wie {LR_GEKOPPELT[arm]})", flush=True)
        if arm not in z.lr:
            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = _schl(_arm, seed, f"{lr:.6g}")
                if k not in z.tuning:
                    z.tuning[k] = zelle(_arm, seed, lr, "abstimmung", threads)[METRIK]
                    z.speichern()
                return -z.tuning[k]
            best, vermerk, mittel = sweep_lr("t006", arm, LR_GITTER[arm], TUNING_SEEDS, ERWEITERUNGEN, messe)
            z.lr[arm] = {"lr": best, "rand": vermerk}
            z.speichern()
            print(f"  {arm:<12} -> lr {best:g}" + (f"  ACHTUNG Rand {vermerk}" if vermerk != "innen" else "")
                  + "   " + "  ".join(f"{k:g}:{-v:.4f}" for k, v in sorted(mittel.items())), flush=True)
        for seed in EVAL_SEEDS:
            if not z.gemessen(arm, seed):
                r = zelle(arm, seed, float(z.lr[arm]["lr"]), "validierung", threads)
                pfad = _adapter_pfad(lauf_id, arm, seed)
                pfad.parent.mkdir(parents=True, exist_ok=True)
                torch.save(r.pop("_adapter"), pfad)           # erst Gewichte, dann Zeile
                z.records.append(r)
                z.speichern()
                print(f"    {arm} seed {seed}: {r[METRIK]:.4f}  ({r['sekunden']:.0f} s)", flush=True)
        eigene = [r for r in z.records if r["arm"] == arm]
        bester = min(eigene, key=lambda r: r[METRIK])
        bp = ADAPTER_DIR / lauf_id / "beste.json"
        beste = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
        beste[arm] = {"seed": bester["seed"], METRIK: bester[METRIK], "lr": bester["lr"],
                      "datei": _adapter_pfad(lauf_id, arm, bester["seed"]).name}
        bp.write_text(json.dumps(beste, indent=2), encoding="utf-8")
        print(f"  Zwischenstand gesichert, bester {arm}-Seed {bester['seed']}", flush=True)
        sichern()
    return rec


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _gewinne(records, feld: str):
    """(seeds, G10, G20, G40) -- Gewinn je Seed = Mittel basis - Verlust; None bei Luecken."""
    b = _reihe(records, "basis", feld)
    t = ck._tabelle(records, feld, list(QIRE))
    if t is None or not b:
        return None
    seeds, v = t
    b0 = float(np.mean(list(b.values())))
    return seeds, b0 - v[:, 0], b0 - v[:, 1], b0 - v[:, 2]


def exponent_test(g10: np.ndarray, g20: np.ndarray, g40: np.ndarray, b_null: float = B_NULL) -> Dict[str, object]:
    """b = log(G40/G10)/log 4 mit G = Mittel der Seed-Gewinne; gepaarter Bootstrap ueber Seeds."""
    n = len(g10)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT_B, n))
    m10, m20, m40 = g10[idx].mean(1), g20[idx].mean(1), g40[idx].mean(1)
    ok = (m10 > 0) & (m20 > 0) & (m40 > 0)
    with np.errstate(all="ignore"):
        b_s = np.log(m40 / m10) / math.log(4.0)
        r1, r2 = m20 / m10, m40 / m20
    bs = b_s[ok]
    unten, oben = int((bs <= b_null).sum()), int((bs >= b_null).sum())
    p = min(1.0, 2 * min(unten + 1, oben + 1) / (len(bs) + 1)) if len(bs) else 1.0
    a, c, d = float(g10.mean()), float(g20.mean()), float(g40.mean())
    ci = lambda x: [float(np.nanpercentile(x[ok], 2.5)), float(np.nanpercentile(x[ok], 97.5))]  # noqa: E731
    return {"gewinne": [a, c, d], "b": math.log(d / a) / math.log(4.0) if min(a, c, d) > 0 else float("nan"),
            "b_ci": ci(b_s), "r1": c / a if a > 0 else float("nan"), "r1_ci": ci(r1),
            "r2": d / c if c > 0 else float("nan"), "r2_ci": ci(r2), "p": float(p),
            "def": bool(min(a, c, d) > 0 and ok.mean() > 0.95), "wurzel_verhaeltnis": 2.0 ** b_null}


def _monotonie(records, feld: str, mut: Dict[str, object]):
    arme = list(QIRE)[::-1] if mut.get("reihenfolge") else list(QIRE)
    t = ck._tabelle(records, feld, arme)
    if t is None:
        return None
    _, v = t
    stufen = [{"stufe": f"L{QIRE[arme[i]]}->L{QIRE[arme[i + 1]]}",
               "delta": float((v[:, i + 1] - v[:, i]).mean()), "siege": int((v[:, i + 1] < v[:, i]).sum()),
               "p": _wilcoxon(v[:, i + 1] - v[:, i], "less")} for i in range(len(arme) - 1)]
    schwach = (min if mut.get("iut_min") else max)(stufen, key=lambda s: s["p"])
    return {**ck.page_test(v), "p_iut": schwach["p"], "schwaechste": schwach["stufe"], "stufen": stufen,
            "mittel": {a: float(v[:, i].mean()) for i, a in enumerate(arme)}}


def _k3(records, feld: str, mut: Dict[str, object]):
    g = _gewinne(records, feld)
    if g is None or len(g[0]) < 2:
        return None
    vorz = -1.0 if mut.get("b_vorzeichen") else 1.0
    r = exponent_test(vorz * g[1], vorz * g[2], vorz * g[3], 1.0 if mut.get("b_eins") else B_NULL)
    r["p"] = r["p"] if r["def"] else 1.0
    return r


def _familie(records, feld: str, grenze: float, mut: Dict[str, object]):
    r = {a: _reihe(records, a, feld) for a in ("qire_L40", "lora_r1")}
    return {"K1": _monotonie(records, feld, mut), "K2": _vergleich(r["qire_L40"], r["lora_r1"], grenze),
            "K3": _k3(records, feld, mut)}


def poskontrolle(records, feld: str, mut: Dict[str, object]) -> Dict[str, object]:
    w = _reihe(records, "qire_L10", feld)
    m = float(np.mean(list(w.values()))) if w else float("nan")
    ok = bool(w) and (abs(m - POS_SOLL) <= POS_TOL or bool(mut.get("pos_aus")))
    return {"bestanden": ok, "mittel": m, "soll": POS_SOLL, "tol": POS_TOL}


_VERGLEICH = {"K2": "qire_L40 - lora_r1"}


def _urteil(tag: str, v, p_holm: float, k, pos, tost_zuerst: bool = True) -> str:
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
    if not pos["bestanden"]:
        return (f"HU_POSKONTROLLE -- {tag}: qire_L10 liegt bei {pos['mittel']:.4f}, erwartet "
                f"{pos['soll']:.4f} +-{pos['tol']:g} (T-005). Aufbau nicht reproduziert; KEINE Aussage.")
    if not k["aufloesung"]["bestanden"]:
        return (f"HU_AUFLOESUNG -- {tag}: lora_r1 trennt sich nicht von basis"
                + (" (Zelle fehlt)" if k["aufloesung"].get("fehlt") else "")
                + ". Der Aufbau sieht keine Anpassung; KEIN Nullbefund.")
    if tag == "K1":
        z = (f"Page L={v['L']:.0f}, rho-Mittel {v['rho_mittel']:+.3f}, p={v['p']:.2e}, Holm={p_holm:.4f}; "
             f"schwaechste Stufe {v['schwaechste']} p={v['p_iut']:.4f} (Schwelle {IUT_SCHWELLE:.4f})")
        if p_holm < ALPHA and v["p_iut"] < IUT_SCHWELLE:
            return f"STRENG_MONOTON -- K1: jede weitere Lagenstufe senkt den Verlust (Trend und jede Stufe belegt); {z}."
        if p_holm < ALPHA:
            return (f"WAECHST_MIT_PLATEAU -- K1: der Gewinn waechst im Trend mit L, aber nicht an jeder "
                    f"Stufe belegt (Stufe {v['schwaechste']}); {z}.")
        return (f"KEIN_TREND -- K1: kein belegter Anstieg des Gewinns mit L; {z}."
                + (" Beschreibend FAELLT der Gewinn mit L." if v["rho_mittel"] < 0 else ""))
    if tag == "K3":
        if not v["def"]:
            return (f"HU_UNVOLLSTAENDIG -- K3: ein Gewinn ist nicht positiv, Exponent nicht definiert "
                    f"(G10/G20/G40 = {', '.join(f'{x:+.4f}' for x in v['gewinne'])}).")
        z = (f"b={v['b']:.3f} [{v['b_ci'][0]:.3f}; {v['b_ci'][1]:.3f}] gegen {B_NULL}, p={v['p']:.4f}, "
             f"Holm={p_holm:.4f}; G20/G10={v['r1']:.3f}, G40/G20={v['r2']:.3f} (Wurzel: {v['wurzel_verhaeltnis']:.3f})")
        if p_holm < ALPHA and v["b"] < B_NULL:
            return f"SAETTIGT_UNTER_WURZEL -- K3: der Gewinn waechst langsamer als sqrt(L); {z}."
        if p_holm < ALPHA:
            return f"STEIGT_UEBER_WURZEL -- K3: der Gewinn waechst schneller als sqrt(L); {z}."
        return (f"OFFEN_ZU_WURZEL -- K3: Wurzelgesetz nicht verworfen (KEINE Gleichwertigkeit, nur nicht "
                f"unterscheidbar); {z}.")
    z = (f"{_VERGLEICH[tag]}: {v['delta']:+.4f} nats, {v['siege']}/{v['n']} Seeds besser, "
         f"p={v['p']:.4f}, Holm={p_holm:.4f}, TOST p={v['p_tost']:.4f} (Grenze +-{v['grenze']:.4f})")
    gleich = (f"GLEICHWERTIG -- K2: gleichwertig innerhalb +-{v['grenze']:.4f} nats "
              f"({REL_GRENZE:.0%} des LoRA-Gewinns; erreicht +-{v['erreicht']:.4f}); {z}. Positiver "
              "Gleichwertigkeitsnachweis, nicht blosses p>0,05."
              + (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber kleiner als die Grenze."
                 if p_holm < ALPHA else ""))
    code, satz = (("QIRE_BESSER", "qire_L40 (7.680 Winkel) senkt den Verlust staerker als LoRA r=1")
                  if v["delta"] < 0 else
                  ("LORA_BESSER", "LoRA r=1 senkt den Verlust staerker; die 4-fach kleinere Drehung "
                                  "kommt nicht gleich weit"))
    reihenfolge = ((v["p_tost"] < ALPHA, gleich), (p_holm < ALPHA, f"{code} -- K2: {satz}; {z}."))
    for bedingung, text in (reihenfolge if tost_zuerst else reihenfolge[::-1]):
        if bedingung:
            return text
    return (f"UNENTSCHIEDEN -- K2: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
            f"die Daten tragen erst ab +-{v['erreicht']:.4f}; {z}.")


def budget_urteil(records, feld: str) -> Optional[Dict[str, object]]:
    """Beschreibend, ausserhalb Holm: qire_L10_600 gegen qire_L10 und qire_L20."""
    if "qire_L10_600" not in ARME:
        return None
    a, b, c = (_reihe(records, x, feld) for x in ("qire_L10_600", "qire_L10", "qire_L20"))
    v1, v2 = _vergleich(a, b, 0.0), _vergleich(c, b, 0.0)
    if v1 is None or v2 is None:
        return None
    zuwachs, struktur = -v1["delta"], -v2["delta"]
    if v1["mittel_a"] <= v2["mittel_a"]:
        marke = "BUDGETGRENZE"
    elif struktur > 0 and zuwachs < 0.5 * struktur:
        marke = "STRUKTURGRENZE_EHER"
    else:
        marke = "GEMISCHT"
    return {"marke": marke, "zuwachs_600": zuwachs, "zuwachs_L20": struktur, "p_600": v1["p"], "p_L20": v2["p"],
            "mittel_600": v1["mittel_a"], "mittel_L20": v2["mittel_a"]}


def analyse(rec_or_id, still: bool = False, mut: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """`mut` nur fuer die Mutationspruefung des Vortests -- im Lauf immer None."""
    def say(*a):
        if not still:
            print(*a)

    mut = mut or {}
    feld = str(mut.get("feld", METRIK))
    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    k = ca._kontrollen(records, feld, mut)
    pos = poskontrolle(records, feld, mut)
    fam = _familie(records, feld, k["grenze"], mut)
    p_holm = dict(zip(FAMILIE, holm([fam[t]["p"] if fam[t] else 1.0 for t in FAMILIE])))
    out: Dict[str, object] = {"kontrollen": k, "poskontrolle": pos}
    say(f"\nRESEARCH ID {LAUF_ID} | {METRIK} (nats/Token, nur Antwort-Tokens), kleiner = besser")
    say("=" * 96)
    say(f"POSITIVKONTROLLE qire_L10 {pos['mittel']:.4f} (soll {POS_SOLL} +-{POS_TOL}): "
        f"{'bestanden' if pos['bestanden'] else 'NICHT BESTANDEN'}")
    say(f"VORREGISTRIERTE FAMILIE (Holm ueber 3), TOST-Grenze {REL_GRENZE:.0%} x LoRA-Gewinn "
        f"{k['gewinn_lora']:+.4f} = +-{k['grenze']:.4f}")
    for t in FAMILIE:
        u = _urteil(t, fam[t], p_holm[t], k, pos, bool(mut.get("tost_zuerst", True)))
        out[t] = {**(fam[t] or {}), "p_holm": p_holm[t], "verdict": u}
        say(f"  [{t}] {u}")
    a = k["aufloesung"]
    say(f"KONTROLLE  Aufloesung {'bestanden' if a['bestanden'] else 'NICHT BESTANDEN'}"
        + (f" (lora_r1-basis {a['delta']:+.4f}, p={a['p']:.4f})" if "delta" in a else " (fehlt)"))
    bu = budget_urteil(records, feld)
    if bu:
        out["budget"] = bu
        say(f"OPTIONAL (ohne Holm) {bu['marke']}: L10_600 {bu['mittel_600']:.4f} (Zuwachs {bu['zuwachs_600']:+.4f}, "
            f"p={bu['p_600']:.4f}) gegen L20 {bu['mittel_L20']:.4f} (Zuwachs {bu['zuwachs_L20']:+.4f}, p={bu['p_L20']:.4f})")
    if not still:
        say("\nBESCHREIBEND (ohne Holm)")
        b = _reihe(records, "basis", feld)
        b0 = float(np.mean(list(b.values()))) if b else float("nan")
        for arm in ARME:
            w = np.array(list(_reihe(records, arm, feld).values()))
            if len(w):
                n_p, g = ERWARTETE_PARAMETER[arm], b0 - w.mean()
                say(f"  {arm:<12} {len(w)} Seeds  Mittel {w.mean():.4f}  sd {w.std(ddof=1) if len(w) > 1 else 0:.4f}"
                    f"  Gewinn {g:+.4f}" + (f" ({g / k['gewinn_lora']:.0%} von LoRA)" if k["gewinn_lora"] else "")
                    + (f"  {g / n_p:.2e}/Param" if n_p else "") + f"  trainierbar {n_p}"
                    + f"  lr {config.get('lr', {}).get(arm, {}).get('lr', '-')}")
        if fam["K1"]:
            for s in fam["K1"]["stufen"]:
                say(f"  Stufe {s['stufe']:<8} {s['delta']:+.4f}  {s['siege']}/{fam['K1']['n']} Seeds  p={s['p']:.4f}")
    raender = {s: v["rand"] for s, v in config.get("lr", {}).items() if v.get("rand") != "innen"}
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest, Teil 1: Theorie (Modellteil nur, wenn das Modell vorliegt)
# ---------------------------------------------------------------------------
def modell_vorhanden() -> bool:
    try:
        ca.modell("basis")
        return True
    except Exception as exc:                          # noqa: BLE001
        print(f"  Modell nicht ladbar ({type(exc).__name__}): Modellteil uebersprungen")
        return False


def pruefe_theorie(threads: int = 3, mit_modell: bool = True) -> bool:
    torch.set_num_threads(threads)
    zeilen, eye = [], torch.eye(BREITE, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        for lagen in LAGEN:
            d = ca.QireDrehung(lagen, True)
            zeilen.append((f"L{lagen}: Winkel 0 -> R = I (mit Rueckpermutation)",
                           float((ca._matrix(d) - eye).abs().max()), "<", 1e-12))
            d = ca.QireDrehung(lagen, True)
            d.theta.uniform_(-math.pi, math.pi)
            r = ca._matrix(d)
            zeilen.append((f"L{lagen}: R^T R = I (zufaellige Winkel)", float((r.T @ r - eye).abs().max()), "<", 1e-5))
            zeilen.append((f"L{lagen}: Parameter {24 * 8 * lagen}", float(d.theta.numel() * 24 - 24 * 8 * lagen), "<", 0.5))
            bank = ca.Drehbank(24, lagen, True).double()
            bank.theta.uniform_(-math.pi, math.pi)
            bank.aktualisieren()
            d = ca.QireDrehung(lagen, True).double()
            d.theta.copy_(bank.theta[11])
            zeilen.append((f"L{lagen}: Drehbank R_k^T == QireDrehung",
                           float((torch.block_diag(*[bank.rt[11]] * 3) - ca._matrix(d).T).abs().max()), "<", 1e-10))
        # Gradient-/Skalierungsannahme: Winkel winzig -> R ~ I + A (lineares Regime)
        d = ca.QireDrehung(40, True)
        d.theta.uniform_(-0.02, 0.02)
        zeilen.append(("L40: kleine Winkel (|theta| <= 0,02) -> ||R - I|| < 0,5 (lineares Regime)",
                       float((ca._matrix(d) - eye).abs().max()), "<", 0.5))
        if mit_modell:
            d_ = daten()
            ids, maske, lab = ca.stapel(d_["validierung"][:3])
            ref = ck.modell("basis")
            ref.eval()
            l_ref = ref(input_ids=ids, attention_mask=maske).logits
            for arm in TRAINIERT:
                m = ck.modell(arm)
                m.eval()
                zeilen.append((f"{arm}: Startwert -> Logits == Basismodell",
                               float((m(input_ids=ids, attention_mask=maske).logits - l_ref).abs().max()), "<", 1e-5))
                zeilen.append((f"{arm}: trainierbar {ca.trainierbar(m)} == {ERWARTETE_PARAMETER[arm]}",
                               abs(ca.trainierbar(m) - ERWARTETE_PARAMETER[arm]), "<", 0.5))
            # Bruecke: T-006-qire_L10 == T-005-qire_L10 (3 Schritte) und lora_r1
            for arm in ("qire_L10", "lora_r1"):
                a = zelle(arm, 7, 1e-3, "validierung", threads, 3, True)
                b = ck.zelle(arm, 7, 1e-3, "validierung", threads, 3, True)
                zeilen.append((f"Bruecke: T-006-Zelle {arm} == T-005-Zelle (3 Schritte)",
                               max(abs(a[METRIK] - b[METRIK]), abs(a["train_verlust_ende"] - b["train_verlust_ende"])),
                               "<", 1e-9))
            # L10_600: 600 Schritte laufen wirklich (Zaehler), ohne 600 zu rechnen: Schrittzahl je Arm
            zeilen.append(("Schritte je Arm: L10_600 == 2 x SCHRITTE, alle anderen == SCHRITTE",
                           float(sum(SCHRITTE_ARM[a] != (2 * SCHRITTE if a in QIRE_LANG else SCHRITTE) for a in ARME)),
                           "<", 0.5))
    # Lernraten == T-005-Rohdaten
    lr_alt = json.loads(T005_ROH.read_text(encoding="utf-8"))["config"] if T005_ROH.exists() else None
    if lr_alt is not None:
        abw = sum(lr_alt.get("lr", {}).get(a, {}).get("lr") != v or lr_alt["lr"][a].get("rand") != "innen"
                  for a, v in LR_UEBERNOMMEN.items()) + (lr_alt.get("schritte") != SCHRITTE)
        zeilen.append(("LR-Uebernahme == T-005-Rohdaten (lr, innen, Schritte)", float(abw), "<", 0.5))
    else:
        print(f"  HINWEIS: {T005_ROH_PATTERN} fehlt, LR-Uebernahme nicht gegengeprueft")
    zeilen.append(("Rest getunt: jeder trainierte Arm hat Rate (uebernommen/gekoppelt/Gitter)",
                   float(sum(a not in LR_GITTER and a not in LR_UEBERNOMMEN and a not in LR_GEKOPPELT
                             for a in TRAINIERT)), "<", 0.5))
    alt = set(ca.EVAL_SEEDS) | set(ca.TUNING_SEEDS) | set(ck.EVAL_SEEDS) | set(ck.TUNING_SEEDS)
    zeilen.append(("Seeds frisch: disjunkt zu T-004/T-005, Mess/Tuning disjunkt (Ueberschneidungen)",
                   float(len((set(EVAL_SEEDS) | set(TUNING_SEEDS)) & alt) + len(set(EVAL_SEEDS) & set(TUNING_SEEDS))
                         + (len(EVAL_SEEDS) != 8)), "<", 0.5))
    zeilen.append(("Parameter qire_L10 / lora_r1 == T-005",
                   float(sum(ERWARTETE_PARAMETER[a] != ck.ERWARTETE_PARAMETER[a] for a in ("qire_L10", "lora_r1"))),
                   "<", 0.5))
    return ck._zeilen_drucken("THEORIEPRUEFUNG T-006", zeilen)


def pruefe_page() -> bool:
    zeilen = []
    for k, n in ((3, 8), (3, 3)):
        dist, gesamt = ck.page_verteilung(k, n)
        mu = sum(v * c for v, c in dist.items()) / gesamt
        zeilen.append((f"k={k}, n={n}: E[L] == n k (k+1)^2 / 4", abs(mu - n * k * (k + 1) ** 2 / 4), "<", 1e-6))
    from scipy.stats import page_trend_test
    rng = np.random.default_rng(5)
    for i in range(3):
        v = rng.normal(size=(len(EVAL_SEEDS), 3)) - 0.3 * i * np.arange(3)
        zeilen.append((f"scipy page_trend_test(exact) == eigene Faltung (k=3, Trend {0.3 * i:.1f})",
                       abs(page_trend_test(-v, method="exact").pvalue - ck.page_test(v)["p"]), "<", 1e-9))
    perfekt = -np.tile(np.arange(3, dtype=float), (len(EVAL_SEEDS), 1)) + rng.normal(0, 1e-3, (len(EVAL_SEEDS), 1))
    zeilen.append(("perfekt fallend -> p == (1/3!)^8 (relativ)",
                   abs(ck.page_test(perfekt)["p"] * 6 ** len(EVAL_SEEDS) - 1.0), "<", 1e-9))
    zeilen.append(("Erreichbarkeit: Page-min x 3 < 0,05", 3 * 6.0 ** -len(EVAL_SEEDS), "<", 0.05))
    zeilen.append(("Erreichbarkeit: IUT-min (Wilcoxon einseitig, 8) < 0,05/3",
                   _wilcoxon(-np.arange(1, 9, dtype=float), "less"), "<", IUT_SCHWELLE))
    zeilen.append(("Erreichbarkeit: Bootstrap-min x 3 < 0,05", 3.0 / (BOOT_B + 1), "<", 0.05))
    # Exponent-Test: bekanntes b wird erkannt
    for b_wahr, soll in ((0.5, "hoch"), (0.2, "niedrig"), (0.9, "niedrig")):
        g = np.array([0.15 * (1 + 0.003 * rng.normal()) for _ in range(8)])
        g40 = np.array([0.15 * 4 ** b_wahr * (1 + 0.003 * rng.normal()) for _ in range(8)])
        p = exponent_test(g, np.sqrt(g * g40), g40)["p"]
        zeilen.append((f"Exponent b={b_wahr}: p {'hoch (> 0,05)' if soll == 'hoch' else 'niedrig (< 0,05)'}",
                       p, ">" if soll == "hoch" else "<", 0.05))
    return ck._zeilen_drucken("PRUEFUNG PAGE- UND EXPONENT-TEST", zeilen)


def test_gitterverlaengerung() -> bool:
    g = LR_START
    faelle = [("Optimum im Startgitter", 1e-3, "innen", 1e-3), ("Verlaengerung oben", 9e-3, "innen", 9e-3),
              ("Verlaengerung unten", 1e-4, "innen", 1e-4),
              ("ausserhalb oben", 300.0, "oben", g[-1] * 27), ("ausserhalb unten", 1e-9, "unten", g[0] / 27)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG -- beide Richtungen, Startgitter", g)
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        best, rand, _ = sweep_lr("t006", "qire_L40", g, (1, 2), ERWEITERUNGEN,
                                 lambda lr, s: -(3.0 + math.log10(lr / opt) ** 2))
        good = abs(math.log10(best / rate_soll)) < 1e-6 and rand == rand_soll
        ok &= good
        print(f"  {name:<28} -> lr {best:<10.4g} Rand {rand:<6} {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


# ---------------------------------------------------------------------------
# Pflichtvortest, Teil 2: jeden Ausgang einzeln erzwingen (synthetische Werte)
# ---------------------------------------------------------------------------
NIVEAU, NIVEAU_ALLE = 2.5616, 3.50
G10, G_LORA = 0.1501, 0.2741                         # T-005: 2,4115 bzw. 2,2875
_V = {"basis": 0.0, "qire_L10": -G10, "qire_L20": -G10 * math.sqrt(2), "qire_L40": -G10 * 2.0,
      "lora_r1": -G_LORA, "qire_L10_600": -0.19}
VERSATZ = {a: _V[a] for a in ARME}
GLEICH_L = {"qire_L20": -G10, "qire_L40": -G10}
FALLEND_L = {"qire_L20": -0.10, "qire_L40": -0.05}


def _kunst(seed: int, streuung: float = 0.003, versatz=None, weg: Sequence[str] = (),
           weg_seed: Optional[Dict[str, int]] = None, s_arm: Optional[Dict[str, float]] = None) -> Recorder:
    def rng(*t):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, t)).encode())])
    vs = {**VERSATZ, **(versatz or {})}
    rec = Recorder("T-006-vortest", "synthetisch", config={"lr": {}})
    for s in EVAL_SEEDS:
        grund = rng("grund", s).normal(0, 0.0015)
        for arm, v in vs.items():
            if arm in weg or (weg_seed or {}).get(arm) == s:
                continue
            sd = (s_arm or {}).get(arm, streuung)
            eig = 0.0 if arm == "basis" else grund + rng(arm, s).normal(0, sd)
            alle = 0.0 if arm == "basis" else eig + rng("alle", arm, s).normal(0, sd)
            rec.add(arm=arm, seed=s, lr=0.001, menge="validierung", val_verlust=NIVEAU + v + eig,
                    val_verlust_alle=NIVEAU_ALLE - v + alle, trainierbar=ERWARTETE_PARAMETER[arm])
    return rec


def _g(g20: float, g40: float) -> dict:
    return {"qire_L20": -g20, "qire_L40": -g40}


FAELLE = [
    # (Name, Test, kwargs, Code, Zweigtext, Ziehungen, Anteil)
    ("M-S   streng monoton (b=0,5)", "K1", {}, "STREN" "G_MONOTON", "jede weitere", 5, 1.0),
    ("M-P   Plateau: L40 = L20", "K1", dict(versatz={"qire_L40": -G10 * math.sqrt(2)}),
     "WAECHST_MIT_PLATEAU", "Stufe L20->L40", 20, 0.9),
    ("M-R   L40 schlechter als L20", "K1", dict(versatz=_g(0.25, 0.20)), "WAECHST_MIT_PLATEAU", "Stufe L20->L40", 5, 1.0),
    ("M-N   alle L gleich, Rauschen 0,01", "K1", dict(versatz=GLEICH_L, s_arm={"qire_L20": 0.01, "qire_L40": 0.01}),
     "KEIN_TREND", "kein belegter Anstieg", 40, 0.8),
    ("M-F   Gewinn FAELLT mit L", "K1", dict(versatz=FALLEND_L), "KEIN_TREND", "Beschreibend FAELLT", 5, 1.0),
    ("M-A   Aufloesung faellt (lora = basis)", "K1", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG",
     "KEIN Nullbefund", 40, 0.8),
    ("M-An  lora-Zelle fehlt", "K1", dict(weg=["lora_r1"]), "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
    ("M-X   qire_L20 fehlt", "K1", dict(weg=["qire_L20"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("M-Xs  ein Seed fehlt in qire_L40", "K1", dict(weg_seed={"qire_L40": EVAL_SEEDS[2]}), "HU_UNVOLLSTAENDIG",
     "Seeds ungleich", 5, 1.0),
    ("M-Ar  lora SCHLECHTER als basis: Trennung zaehlt", "K1", dict(versatz={"lora_r1": 0.10}),
     "STRENG_MONOTON", "jede weitere", 5, 1.0),
    ("P-K   Positivkontrolle faellt (L10 bei 2,45)", "K1", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE",
     "nicht reproduziert", 5, 1.0),
    ("P-K3  Positivkontrolle faellt -> auch K3", "K3", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE",
     "nicht reproduziert", 5, 1.0),
    ("Z-L   LoRA besser als qire_L40 (0,074)", "K2", dict(versatz={"qire_L40": -0.20}), "LORA_BESSER",
     "nicht gleich weit", 5, 1.0),
    ("Z-Q   qire_L40 besser als LoRA", "K2", dict(versatz={"qire_L40": -0.45}), "QIRE_BESSER", "staerker als LoRA", 5, 1.0),
    ("Z-G   gleich, kleines Rauschen", "K2", dict(versatz={"qire_L40": -G_LORA}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("Z-Gs  +0,005 bei winzigem Rauschen: signifikant UND gleichwertig", "K2",
     dict(streuung=0.0005, versatz={"qire_L40": -G_LORA + 0.005}, s_arm={"qire_L10": 0.003}), "GLEICHWERTIG",
     "zugleich signifikant", 5, 1.0),
    ("Z-U   gleich, grosses Rauschen", "K2",
     dict(versatz={"qire_L40": -G_LORA}, s_arm={"qire_L40": 0.08, "lora_r1": 0.08}), "UNENTSCHIEDEN",
     "weder Unterschied", 40, 0.8),
    ("Z-X   qire_L40 fehlt", "K2", dict(weg=["qire_L40"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("Z-A   Aufloesung faellt", "K2", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.8),
    ("B-W   Wurzelgesetz b=0,5", "K3", {}, "OFFEN_ZU_WURZEL", "KEINE Gleichwertigkeit", 5, 1.0),
    ("B-S   Saettigung (G20 0,17, G40 0,19)", "K3", dict(versatz=_g(0.17, 0.19)), "SAETTIGT_UNTER_WURZEL",
     "langsamer als sqrt", 5, 1.0),
    ("B-T   steiler als Wurzel (b=1)", "K3", dict(versatz=_g(0.30, 0.60)), "STEIGT_UEBER_WURZEL",
     "schneller als sqrt", 5, 1.0),
    ("B-U   b=0,5, grosses Rauschen", "K3", dict(s_arm={"qire_L20": 0.05, "qire_L40": 0.05}), "OFFEN_ZU_WURZEL",
     "nicht verworfen", 40, 0.8),
    ("B-X   qire_L40 fehlt", "K3", dict(weg=["qire_L40"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("B-A   Aufloesung faellt", "K3", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.8),
    ("X-K   qire_L40 und K3-Daten ok, lora fehlt -> K1 liefert Aufloesung", "K3", dict(weg=["lora_r1"]),
     "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
]
FAELLE[0] = (FAELLE[0][0], "K1", {}, "STRENG_MONOTON", "jede weitere", 5, 1.0)


def _szenarien(mut: Optional[Dict[str, object]] = None, laut: bool = True) -> bool:
    alle_ok = True
    for i, (name, test, kw, code, zweig, ziehungen, anteil) in enumerate(FAELLE):
        treffer, marken = 0, []
        for zug in range(ziehungen):
            u = str(analyse(_kunst(seed=1000 * i + zug, **kw), still=True, mut=mut)[test]["verdict"])
            marken.append(u.split(" -- ")[0])
            treffer += int(u.startswith(code + " --") and zweig in u)
        ok = treffer / ziehungen >= anteil - 1e-9
        alle_ok &= ok
        if laut:
            print(f"  {'OK    ' if ok else 'FEHLER'} {name:<64} {treffer}/{ziehungen}  {dict(Counter(marken))}")
        elif not ok:
            print(f"      gefangen von {name.split()[0]}: {treffer}/{ziehungen} {dict(Counter(marken))}")
    return alle_ok


def pruefe_budget() -> bool:
    ok = True
    for name, v600, soll in (("L10_600 besser als L20", -0.25, "BUDGETGRENZE"),
                             ("L10_600 kaum besser", -0.155, "STRUKTURGRENZE_EHER"),
                             ("L10_600 halbwegs", -0.19, "GEMISCHT")):
        got = budget_urteil(_kunst(seed=11, versatz={"qire_L10_600": v600}).records, METRIK)["marke"]
        ok &= got == soll
        print(f"  {'OK    ' if got == soll else 'FEHLER'} Budgetmarke {name}: {got}")
    return ok


def pruefe_generator() -> bool:
    rec = _kunst(seed=4242)
    r = lambda a, f: np.array(list(_reihe(rec.records, a, f).values()))  # noqa: E731
    pruef = [("val_verlust_alle - val_verlust NICHT konstant", min(float(np.std(r(a, "val_verlust_alle") - r(a, METRIK), ddof=1))
                                                               for a in TRAINIERT) > 1e-4),
             ("basis deterministisch, trainierte Arme streuen",
              float(np.std(r("basis", METRIK))) == 0 and float(np.std(r("qire_L40", METRIK))) > 1e-4),
             ("Standardfall: Gewinne steigen mit L und b = 0,5 (G40 = 2 G10)",
              abs(-_V["qire_L40"] / -_V["qire_L10"] - 2.0) < 1e-9 and _V["qire_L20"] < _V["qire_L10"]),
             ("Positivkontrolle im Standardfall erfuellt", poskontrolle(rec.records, METRIK, {})["bestanden"])]
    for name, good in pruef:
        print(f"  {'OK    ' if good else 'FEHLER'} {name}")
    return all(g for _, g in pruef)


MUTATIONEN = [("Metrikfeld val_verlust_alle statt val_verlust", {"feld": "val_verlust_alle"}),
              ("Signifikanz VOR TOST (K2)", {"tost_zuerst": False}),
              ("Aufloesung verlangt Richtung statt Trennung", {"richtung": True}),
              ("TOST-Grenze absolut 0,2 statt relativ", {"grenze_absolut": True}),
              ("K1 Arme in falscher Reihenfolge (L absteigend)", {"reihenfolge": True}),
              ("K1 IUT mit min statt max (eine Stufe reicht)", {"iut_min": True}),
              ("K3 gegen b = 1 statt 0,5", {"b_eins": True}),
              ("K3 Vorzeichen der Gewinne vertauscht", {"b_vorzeichen": True}),
              ("Positivkontrolle abgeschaltet", {"pos_aus": True})]


def vortest(mit_modell: bool = False) -> bool:
    print("=" * 96)
    print("VORTEST T-006 -- jeder Ausgang einzeln, ueber mehrere Ziehungen, Marke UND Zweigtext")
    ok = _szenarien()
    print("GENERATOR")
    ok &= pruefe_generator()
    print("OPTIONALER ARM (Budgetmarke)")
    ok &= pruefe_budget()
    print("MUTATIONSPRUEFUNG -- jeder absichtliche Fehler MUSS den Vortest durchfallen lassen")
    for name, m in MUTATIONEN:
        faellt = not _szenarien(m, laut=False)
        ok &= faellt
        print(f"  {'OK    ' if faellt else 'FEHLER'} Mutation '{name}' {'faellt durch' if faellt else 'BLEIBT UNENTDECKT'}")
    print("BESCHREIBENDER TEIL (Durchlauf auf synthetischen Daten)")
    analyse(_kunst(seed=7), still=False)
    print("=" * 96)
    print("VORTEST BESTANDEN" if ok else "VORTEST FEHLGESCHLAGEN")
    return bool(ok)


# ---------------------------------------------------------------------------
# Zeitmessung (Modell noetig) und Prognose aus T-005-Rohdaten (v5 §6)
# ---------------------------------------------------------------------------
def prognose() -> None:
    s = lambda l: 1.05 + 0.06 * l                    # s/Schritt bei 6 Threads, Fit an T-005  # noqa: E731
    zelle_min = lambda arm, st: (st * (0.87 if arm == "lora_r1" else s(ck.ERWARTETE_PARAMETER[arm] // 192)) + 130) / 60  # noqa: E731
    print("PROGNOSE (Extrapolation, unsicher): s/Schritt = 1,05 + 0,06 L; Zelle = Schritte x s + 130 s")
    tab = {a: zelle_min(a, SCHRITTE_ARM[a]) for a in TRAINIERT}
    for a, t in tab.items():
        print(f"  {a:<13} {t:5.1f} min/Zelle")
    mess = sum(8 * t for a, t in tab.items() if a != "qire_L10_600") / 60
    tune = sum(6 * (tab[a] - 0.5) for a in LR_GITTER) / 60
    print(f"  Messung (ohne optionalen Arm) {mess:.1f} h, Tuning {tune:.1f} h, Kern {mess + tune:.1f} h; "
          f"optionaler Arm {8 * tab.get('qire_L10_600', 0) / 60:.1f} h")


def zeit(threads: int = 6, schritte: int = 4) -> None:
    prognose()
    print("=" * 96)
    print(f"ZEITMESSUNG T-006 -- {threads} Threads, {schritte} Schritte je Arm")
    daten()
    for arm in ("qire_L10", "qire_L20", "qire_L40", "lora_r1"):
        r = zelle(arm, 1, 1e-3, "validierung", threads, schritte, True)
        print(f"  {arm:<9} {r['s_je_schritt']:>6.3f} s/Schritt   trainierbar {r['trainierbar']}", flush=True)


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        ok = pruefe_theorie(mit_modell=modell_vorhanden())
        raise SystemExit(0 if (ok & pruefe_page() & test_gitterverlaengerung() & vortest()) else 1)
    elif befehl == "vortest_statistik":            # ohne Modell
        raise SystemExit(0 if (pruefe_theorie(mit_modell=False) & pruefe_page() & test_gitterverlaengerung()
                               & vortest()) else 1)
    elif befehl == "prognose":
        prognose()
    elif befehl == "zeit":
        zeit(*(int(a) for a in sys.argv[2:4]))
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else LAUF_ID)
    elif befehl == "lauf":
        if not (pruefe_theorie(mit_modell=True) and pruefe_page() and test_gitterverlaengerung() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print(f"  Nachholen: python -m quantum_sim.chatbot_skalierung auswertung {LAUF_ID}")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | vortest_statistik | prognose | zeit [threads] [schritte] | lauf | auswertung [ID|Pfad]")
