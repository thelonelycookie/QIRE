"""RESEARCH ID: T-005 -- Lagenkurve des QIRE-Adapters und QIRE + LoRA kombiniert.

Baut auf T-004 auf (`chatbot_adapter.py`, dort NICHTS geaendert; Modell, Daten,
Aufteilung, Maske, Drehung, Drehbank, Verlust und Auswertung werden importiert).
T-004 (29.09.): basis 2,5616; qire_kron (192) 2,5172; qire_5 (960, CNOT-Ring)
2,4437; lora_r1 (30.720) 2,2893 nats/Token; Seed-sd 0,0005-0,0024.

FRAGE  Wie waechst der Gewinn mit der Zahl L der QIRE-Lagen (mit CNOT-Ring),
  und ergaenzen sich QIRE und LoRA?

ARME (alle auf q_proj und v_proj der 12 Lagen = 24 Matrizen)
    Arm        Aufbau                                              trainierbar
    basis      kein Training, nur Auswertung                                0
    qire_L<L>  L Lagen: Ry je Qubit, dann CNOT-Ring; am Ende P^-L        192 L
               L in LAGEN = (1, 2, 5, 10): 192, 384, 960, 1.920
               (L3 = 576 gestrichen, s. BUDGET)
    lora_r1    wie T-004 (r=1, alpha=2)                                30.720
    kombi      qire_L5 UND lora_r1 gemeinsam trainiert:                31.680
               y = W R x + (alpha/r) B A x, also W' = W R + B A
  qire_L1 hat den CNOT-Ring, ist aber mathematisch GENAU qire_kron (P^-1 P K = K;
  `pruefe_theorie` rechnet es nach) -- alle L-Arme sind eine einzige Familie.
  qire_L5 ist genau qire_5 aus T-004. Beide Gleichheiten prueft der Vortest
  zusaetzlich am echten Training (3 Schritte, T-004-Zelle gegen T-005-Zelle).
  kombi: LoRA sieht dieselbe unrotierte Eingabe wie im Arm lora_r1, die Drehung
  wirkt wie in qire_L5 auf W. EINE gemeinsame Lernrate fuer alle Parameter
  (T-004: beide Einzeloptima 0,001), mit eigener Suche wie jeder Arm.

VORREGISTRIERUNG (vor jeder T-005-Zahl; Aenderungen erst ins Tagebuch)
  Daten, Aufteilung, SCHRITTE = 300, Batch 4, Adam, Clip 1,0: wie T-004.
  Seeds FRISCH: Messung 50001-50008, Tuning 50401/50402 (disjunkt zu T-004
  40001-40008 / 40401-40402; `pruefe_theorie` prueft das).
  Lernrate je Arm mit VOLLEM Budget auf der Abstimmmenge (Regel 1):
  - NEU gesucht fuer qire_L2, qire_L10, kombi: Startgitter (3e-4, 1e-3, 3e-3),
    Tuning-Seeds 50401/50402, Verlaengerung in BEIDE Richtungen bis 3-mal.
    3-Punkt-Gitter, weil in T-004 das Optimum aller drei Arme bei 1e-3 lag und
    auf beiden Seiten schlechter war (qire: 3,3e-4 und 3e-3 gemessen, lora:
    3e-4 und 3e-3); faellt ein Optimum an den Rand, verlaengert `sweep_lr`.
  - UEBERNOMMEN aus T-004 fuer qire_L1, qire_L5, lora_r1 (alle 1e-3, "innen"):
    diese drei Zellen sind BITGLEICH zu T-004 qire_kron / qire_5 / lora_r1
    (Vortest: 3 Trainingsschritte, Abweichung 0,00), dieselbe Abstimmmenge,
    dasselbe Budget, Tuning-Seeds 40401/40402 disjunkt zu den neuen Mess-
    Seeds. Regel 1 ist damit fuer sie schon erfuellt; eine zweite Suche
    kostete 1,8 h und wiederholte nur dieselbe Rechnung mit anderen Seeds.
  Familie, Holm ueber GENAU 3:
    K1  Monotonie ueber L in LAGEN. Teststatistik: exakter Page-Trendtest
        (Bloecke = Mess-Seeds, Rang des Verlusts je Seed; H1 einseitig: der
        Verlust faellt mit L). Page ist die exakte Aggregation der seedweisen
        Spearman-Korrelation gegen L (L_Page linear in rho-Mittel); Verteilung
        exakt durch Faltung der k!-Permutationsverteilung ueber die Bloecke --
        KEINE Asymptotik. Holm-p von K1 = Page-p.
        Strenge Monotonie (Zusatzurteil, Intersection-Union): JEDE Nachbarstufe
        L_i -> L_i+1 einseitig Wilcoxon (Verlust sinkt), p_IUT = max der Stufen,
        muss < 0,05/3 (Bonferroni ueber die Familie, konservativ) sein.
          STRENG_MONOTON / WAECHST_MIT_PLATEAU / KEIN_TREND
    K2  qire_L10 - lora_r1 (1.920 gegen 30.720), zweiseitig + TOST, Grenze wie
        T-004: +-20 % x |Gewinn lora_r1 gegen basis|
          QIRE_BESSER / GLEICHWERTIG / LORA_BESSER / UNENTSCHIEDEN
    K3  kombi - lora_r1, Wilcoxon EINSEITIG (ergaenzt QIRE LoRA?)
          ERGAENZT / KEIN_NACHWEIS
  Rangfolge: HU_UNVOLLSTAENDIG -> HU_AUFLOESUNG -> Urteil. Aufloesung
    (urteilstragend fuer alle drei) wie T-004: lora_r1 trennt sich von basis,
    Wilcoxon zweiseitig p < 0,05.
  Erreichbarkeit (Regel 2): 8 Seeds -> Wilcoxon einseitig min 0,0039, zweiseitig
    0,0078, x3 (Holm) <= 0,023; Page min (1/4!)^8 = 9e-12; IUT min 0,0039 < 0,0167.
    Die T-004-Streuung (sd <= 0,0024) ist klein gegen die dort gemessenen
    Stufen (0,07) -- der Aufbau kann Unterschiede > ~0,003 aufloesen.
  Beschreibend (ohne Holm): Gewinn je L, Gewinn je Parameter, Anteil am
    LoRA-Gewinn, jede Nachbarstufe mit p; kombi - qire_L5.

BUDGET (29.09., `zeit 3 4`, zweimal; kalibriert an den Istzeiten des T-004-
  Laufs aus T004_ROH: Faktor Training x0,57 bzw. x0,52, Auswertung x0,54 bzw.
  x0,50 -- der Rechner war verschieden belastet, die kalibrierte Prognose
  blieb gleich; Restabweichung je Referenzarm <= 6 %). s/Schritt bei 3
  Threads (1. Messung): L1 1,74, L2 1,82, L3 1,95, L5 2,14, L10 2,65, lora
  1,54, kombi 2,14. Messzelle bei 6 Threads: L1 6,7, L2 7,0, L3 7,4, L5 8,0,
  L10 9,6, lora 6,0, kombi 8,0 min (T-004-Ist: 6,8 / 8,3 / 5,8 min).
    alle 5 L, alle Arme getunt (3 Raten)           11,7 h  (4 Raten 13,2 h)
    ohne L3, alle getunt                            10,0 h  (11,4 h)
    PLAN: ohne L3, Tuning nur L2/L10/kombi           8,2 h  (9,0 h, falls
          jede der drei Suchen einmal verlaengert; Messung allein 6,1 h)
  Entscheidung: L3 gestrichen (die Kurve 1-2-5-10 ist etwa log-gleichmaessig;
  L3 kostet 1,6 h) und T-004-Lernraten fuer die drei bitgleichen Arme
  uebernommen (s. o.). Beides vor jeder T-005-Zahl.

Run (aus dem Ordner A):
    python -m quantum_sim.chatbot_kurve vortest          # Pflicht
    python -m quantum_sim.chatbot_kurve zeit [threads] [schritte]
    python -m quantum_sim.chatbot_kurve lauf             # Malte entscheidet
    python -m quantum_sim.chatbot_kurve auswertung [ID|Pfad]
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from qire_ai.equivalence import holm
from qire_ai.result_store import RESULTS_DIR, Recorder

from . import chatbot_adapter as ca
from .chatbot_adapter import (BATCH, BLOCK, BREITE, CLIP, N_ABSTIMM, N_VAL, ZIELE, Drehbank,
                              GedrehteLinear, QireDrehung, _kron_explizit, _matrix, _reihe, _schl,
                              _vergleich, _verlust, _wilcoxon, auswerten, daten, stapel, trainierbar)
from .kernaussage import sweep_lr

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt (Regel 7)
# ---------------------------------------------------------------------------
LAUF_ID = "T-005"
LAGEN = (1, 2, 5, 10)                                  # L3 gestrichen, s. BUDGET
QIRE = {f"qire_L{l}": l for l in LAGEN}
KOMBI_LAGEN = 5
ARME = ("basis", *QIRE, "lora_r1", "kombi")
# Reihenfolge = Wichtigkeit (Regel 5: bricht der Lauf ab, stehen K2/K3 zuerst)
TRAINIERT = ("lora_r1", "qire_L10", "kombi", "qire_L1", "qire_L5", "qire_L2", "qire_L3")
TRAINIERT = tuple(a for a in TRAINIERT if a in ARME)
LORA_PARAMETER = ca.ERWARTETE_PARAMETER["lora_r1"]
ERWARTETE_PARAMETER = {"basis": 0, **{a: 24 * 8 * l for a, l in QIRE.items()},
                       "lora_r1": LORA_PARAMETER, "kombi": LORA_PARAMETER + 24 * 8 * KOMBI_LAGEN}
SCHRITTE = ca.SCHRITTE                                 # 300, wie T-004
EVAL_SEEDS = tuple(range(50001, 50009))
TUNING_SEEDS = (50401, 50402)
LR_START = (3e-4, 1e-3, 3e-3)
# Uebernommen aus T-004 (config["lr"] in T004_ROH, alle "innen"; `pruefe_theorie`
# vergleicht mit der Datei und belegt die bitgleiche Zelle). Siehe BUDGET.
LR_UEBERNOMMEN = {"qire_L1": 0.001, "qire_L5": 0.001, "lora_r1": 0.001}
LR_GITTER = {a: LR_START for a in TRAINIERT if a not in LR_UEBERNOMMEN}
ERWEITERUNGEN = 3
REL_GRENZE = ca.REL_GRENZE                             # 0,20, wie T-004
FAMILIE = ("K1", "K2", "K3")
ALPHA = 0.05
IUT_SCHWELLE = ALPHA / len(FAMILIE)
METRIK = "val_verlust"
ADAPTER_DIR = RESULTS_DIR.parent / "t005_adapter"
ZWISCHEN_DIR = RESULTS_DIR.parent / "t005_zwischenstand"
T004_ROH = RESULTS_DIR / "T-004__20260929_132842_02.json"
T004_PAARE = {"qire_L1": "qire_kron", "qire_L5": "qire_5", "lora_r1": "lora_r1"}


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------
def _kausal(m, arm: str):
    """Das LlamaForCausalLM hinter dem Arm (bei kombi im PeftModel)."""
    return m.base_model.model if arm == "kombi" else m


def _bank_einbauen(kausal, lagen: int, in_lora: bool) -> Drehbank:
    """Drehbank mit CNOT-Ring auf alle 24 Ziele. Der Hook sitzt am LlamaModel,
    das immer ueber __call__ laeuft (auch hinter peft)."""
    bank = Drehbank(len(kausal.model.layers) * len(ZIELE), lagen, True)
    kausal.drehbank = bank
    kausal.model.register_forward_pre_hook(bank.aktualisieren)
    for i, schicht in enumerate(kausal.model.layers):
        for j, name in enumerate(ZIELE):
            ziel, k = getattr(schicht.self_attn, name), len(ZIELE) * i + j
            if in_lora:                                # lora.Linear: y = base(x) + s B A x
                ziel.base_layer = GedrehteLinear(ziel.base_layer, bank=bank, index=k)
            else:
                setattr(schicht.self_attn, name, GedrehteLinear(ziel, bank=bank, index=k))
    return bank


def modell(arm: str):
    if arm in ("basis", "lora_r1"):
        return ca.modell(arm)
    if arm in QIRE:
        m = ca.modell("basis")
        _bank_einbauen(m, QIRE[arm], False)
        return m
    if arm == "kombi":
        m = ca.modell("lora_r1")                       # peft friert alles ausser LoRA ein ...
        _bank_einbauen(_kausal(m, arm), KOMBI_LAGEN, True)   # ... die Winkel entstehen danach
        return m
    raise ValueError(arm)


def adapter_zustand(m, arm: str) -> Dict[str, torch.Tensor]:
    z = ca.adapter_zustand(m, "lora_r1") if arm in ("lora_r1", "kombi") else {}
    if arm != "lora_r1":
        z["drehbank.theta"] = _kausal(m, arm).drehbank.theta.detach().clone()
    return z


# ---------------------------------------------------------------------------
# Eine Zelle -- Trainingsschleife wie `chatbot_adapter.zelle` (dort fest an die
# T-004-Arme gebunden); die Gleichheit prueft `pruefe_theorie` am echten Lauf.
# ---------------------------------------------------------------------------
def zelle(arm: str, seed: int, lr: float, menge: str = "validierung", threads: int = 6,
          schritte: int = SCHRITTE, zeitmessung: bool = False) -> Dict[str, object]:
    t0 = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    d = daten()
    m = modell(arm)
    zahl = trainierbar(m)
    if zahl != ERWARTETE_PARAMETER[arm]:
        raise SystemExit(f"{arm}: {zahl} trainierbare Parameter, erwartet {ERWARTETE_PARAMETER[arm]}")
    verlauf, t_train = [], 0.0
    if arm != "basis" and schritte > 0:
        m.train()
        param = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(param, lr=lr)
        g = torch.Generator().manual_seed(seed)
        train, schritt = d["train"], 0
        while schritt < schritte:
            ordnung = torch.randperm(len(train), generator=g).tolist()
            for k in range(0, len(ordnung) - BATCH + 1, BATCH):
                if schritt >= schritte:
                    break
                if schritt == 1:
                    t1 = time.perf_counter()          # Schritt 0 = Aufwaermen
                s, n, _, _ = _verlust(m, *stapel([train[i] for i in ordnung[k:k + BATCH]]))
                opt.zero_grad(set_to_none=True)
                (s / n).backward()
                torch.nn.utils.clip_grad_norm_(param, CLIP)
                opt.step()
                verlauf.append(float(s.detach()) / n)
                schritt += 1
        t_train = (time.perf_counter() - t1) / max(1, schritte - 1) if schritte > 1 else 0.0
    t2 = time.perf_counter()
    e = auswerten(m, d[menge][:40] if zeitmessung else d[menge])
    out = {"arm": arm, "seed": seed, "lr": lr, "menge": menge, "schritte": len(verlauf),
           METRIK: e["verlust"], "val_verlust_alle": e["verlust_alle"], "tokens": e["tokens"],
           "train_verlust_ende": float(np.mean(verlauf[-20:])) if verlauf else None,
           "trainierbar": zahl, "lagen": QIRE.get(arm, KOMBI_LAGEN if arm == "kombi" else 0),
           "s_je_schritt": round(t_train, 3), "s_auswertung": round(time.perf_counter() - t2, 2),
           "sekunden": round(time.perf_counter() - t0, 2), "threads": threads}
    out["_adapter"] = adapter_zustand(m, arm) if arm != "basis" else None
    return out


# ---------------------------------------------------------------------------
# Wiederaufnahme -- eigene Datei, eigener Ordner, Fingerabdruck der Konfiguration
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
    return {"basis_t004": ca.konfiguration(), "arme": list(ARME), "lagen": list(LAGEN),
            "kombi_lagen": KOMBI_LAGEN, "cnot": True, "trainiert": list(TRAINIERT),
            "schritte": SCHRITTE, "seeds": list(EVAL_SEEDS), "tuning_seeds": list(TUNING_SEEDS),
            "lr_gitter": {k: list(v) for k, v in LR_GITTER.items()}, "erweiterungen": ERWEITERUNGEN,
            "lr_uebernommen": LR_UEBERNOMMEN, "lr_quelle": T004_ROH.name,
            "rel_grenze": REL_GRENZE, "familie": list(FAMILIE), "iut_schwelle": IUT_SCHWELLE,
            "parameter": ERWARTETE_PARAMETER}


def _adapter_pfad(lauf_id: str, arm: str, seed):
    return ADAPTER_DIR / lauf_id / f"{arm}_{seed}.pt"


def run(lauf_id: str = LAUF_ID, threads: int = 6) -> Recorder:
    konfig = konfiguration()
    z = Zustand(lauf_id, konfig)
    rec = Recorder(experiment_id=lauf_id,
                   description=("T-005: QIRE-Lagenkurve L in " + str(list(LAGEN)) + " (CNOT-Ring) und "
                                "kombi = qire_L5 + lora_r1 auf LLaeMmlein-120M; K1 Page-Trend exakt, "
                                "K2 qire_L10-lora_r1 TOST, K3 kombi-lora_r1 einseitig."),
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
            z.lr[arm] = {"lr": LR_UEBERNOMMEN[arm], "rand": "innen", "quelle": T004_ROH.name}
            z.speichern()
            print(f"  {arm:<10} -> lr {LR_UEBERNOMMEN[arm]:g}  (aus T-004 uebernommen)", flush=True)
        if arm not in z.lr:
            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = _schl(_arm, seed, f"{lr:.6g}")
                if k not in z.tuning:
                    z.tuning[k] = zelle(_arm, seed, lr, "abstimmung", threads)[METRIK]
                    z.speichern()
                return -z.tuning[k]                       # sweep_lr maximiert
            best, vermerk, mittel = sweep_lr("t005", arm, LR_GITTER[arm], TUNING_SEEDS,
                                             ERWEITERUNGEN, messe)
            z.lr[arm] = {"lr": best, "rand": vermerk}
            z.speichern()
            print(f"  {arm:<10} -> lr {best:g}" + (f"  ACHTUNG Rand {vermerk}" if vermerk != "innen" else "")
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
        print(f"  Zwischenstand gesichert (Regel 5), bester {arm}-Seed {bester['seed']}")
        sichern()
    return rec


# ---------------------------------------------------------------------------
# Auswertung -- exakter Page-Trendtest und Intersection-Union ueber die Stufen
# ---------------------------------------------------------------------------
_PAGE_CACHE: Dict[tuple, tuple] = {}


def page_verteilung(k: int, n: int):
    """Exakte Nullverteilung von L = sum_Bloecke sum_j j * Rang_j: je Block ist
    jede der k! Rangfolgen gleich wahrscheinlich, die Bloecke unabhaengig ->
    n-fache Faltung. Ganzzahlig, keine Asymptotik. (Verteilung, k!^n)."""
    if (k, n) not in _PAGE_CACHE:
        block = Counter(sum((j + 1) * (p + 1) for j, p in enumerate(perm))
                        for perm in itertools.permutations(range(k)))
        dist = Counter({0: 1})
        for _ in range(n):
            neu: Counter = Counter()
            for a, na in dist.items():
                for b, nb in block.items():
                    neu[a + b] += na * nb
            dist = neu
        _PAGE_CACHE[(k, n)] = (dist, math.factorial(k) ** n)
    return _PAGE_CACHE[(k, n)]


def page_test(verlust: np.ndarray) -> Dict[str, float]:
    """verlust (n Seeds x k Arme), Spalten nach L aufsteigend. H1: der Verlust
    FAELLT mit L (= Gewinn steigt). Rang 1 = hoechster Verlust im Seed."""
    from scipy.stats import rankdata
    raenge = np.vstack([rankdata(-zeile) for zeile in verlust])
    n, k = raenge.shape
    j = np.arange(1, k + 1)
    stat = float((raenge * j).sum())
    dist, gesamt = page_verteilung(k, n)
    p = sum(c for v, c in dist.items() if v >= stat - 1e-9) / gesamt
    rho = 1 - 6 * ((raenge - j) ** 2).sum(axis=1) / (k * (k * k - 1))
    return {"L": stat, "p": float(p), "rho_mittel": float(rho.mean()),
            "bindungen": bool(any(len(set(z)) < k for z in verlust)), "n": n, "k": k}


def _tabelle(records, feld: str, arme: Sequence[str]):
    """(Seeds, n x k) oder None, wenn ein Arm fehlt oder die Seeds ungleich sind."""
    reihen = [_reihe(records, a, feld) for a in arme]
    if any(len(r) < 2 for r in reihen) or len({frozenset(r) for r in reihen}) != 1:
        return None
    seeds = sorted(reihen[0])
    return seeds, np.array([[r[s] for r in reihen] for s in seeds])


def _monotonie(records, feld: str, mut: Dict[str, object]) -> Optional[Dict[str, object]]:
    arme = list(QIRE)[::-1] if mut.get("reihenfolge") else list(QIRE)
    t = _tabelle(records, feld, arme)
    if t is None:
        return None
    _, v = t
    stufen = [{"stufe": f"L{QIRE[arme[i]]}->L{QIRE[arme[i + 1]]}",
               "delta": float((v[:, i + 1] - v[:, i]).mean()),
               "siege": int((v[:, i + 1] < v[:, i]).sum()),
               "p": _wilcoxon(v[:, i + 1] - v[:, i], "less")} for i in range(len(arme) - 1)]
    waehle = min if mut.get("iut_min") else max
    schwach = waehle(stufen, key=lambda s: s["p"])
    return {**page_test(v), "p_iut": schwach["p"], "schwaechste": schwach["stufe"],
            "stufen": stufen, "mittel": {a: float(v[:, i].mean()) for i, a in enumerate(arme)}}


def _familie(records, feld: str, grenze: float, mut: Dict[str, object]):
    r = {a: _reihe(records, a, feld) for a in ("qire_L10", "lora_r1", "kombi")}
    return {"K1": _monotonie(records, feld, mut),
            "K2": _vergleich(r["qire_L10"], r["lora_r1"], grenze),
            "K3": _vergleich(r["kombi"], r["lora_r1"], grenze, str(mut.get("einseitig", "less")))}


_VERGLEICH = {"K2": "qire_L10 - lora_r1", "K3": "kombi - lora_r1"}


def _urteil(tag: str, v, p_holm: float, k, tost_zuerst: bool = True) -> str:
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
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
    z = (f"{_VERGLEICH[tag]}: {v['delta']:+.4f} nats, {v['siege']}/{v['n']} Seeds besser, "
         f"p={v['p']:.4f}, Holm={p_holm:.4f}")
    if tag == "K3":
        if p_holm < ALPHA:
            return f"ERGAENZT -- K3: QIRE ergaenzt LoRA, kombi senkt den Verlust unter lora_r1; {z}."
        return (f"KEIN_NACHWEIS -- K3: keine belegte Ergaenzung (einseitig); {z}."
                + (" Beschreibend liegt kombi sogar HOEHER als lora_r1." if v["delta"] > 0 else ""))
    z += f", TOST p={v['p_tost']:.4f} (Grenze +-{v['grenze']:.4f})"
    gleich = (f"GLEICHWERTIG -- K2: gleichwertig innerhalb +-{v['grenze']:.4f} nats "
              f"({REL_GRENZE:.0%} des LoRA-Gewinns; erreicht +-{v['erreicht']:.4f}); {z}. Positiver "
              "Gleichwertigkeitsnachweis, nicht blosses p>0,05."
              + (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber kleiner als die Grenze."
                 if p_holm < ALPHA else ""))
    code, satz = (("QIRE_BESSER", "qire_L10 (1.920 Winkel) senkt den Verlust staerker als LoRA r=1")
                  if v["delta"] < 0 else
                  ("LORA_BESSER", "LoRA r=1 senkt den Verlust staerker; die 16-fach kleinere Drehung "
                                  "kommt nicht gleich weit"))
    reihenfolge = ((v["p_tost"] < ALPHA, gleich), (p_holm < ALPHA, f"{code} -- K2: {satz}; {z}."))
    for bedingung, text in (reihenfolge if tost_zuerst else reihenfolge[::-1]):
        if bedingung:
            return text
    return (f"UNENTSCHIEDEN -- K2: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
            f"die Daten tragen erst ab +-{v['erreicht']:.4f}; {z}.")


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
    fam = _familie(records, feld, k["grenze"], mut)
    p_holm = dict(zip(FAMILIE, holm([fam[t]["p"] if fam[t] else 1.0 for t in FAMILIE])))
    out: Dict[str, object] = {"kontrollen": k}
    say(f"\nRESEARCH ID {LAUF_ID} | {METRIK} (nats/Token, nur Antwort-Tokens), kleiner = besser")
    say("=" * 96)
    say(f"VORREGISTRIERTE FAMILIE (Holm ueber 3), TOST-Grenze {REL_GRENZE:.0%} x LoRA-Gewinn "
        f"{k['gewinn_lora']:+.4f} = +-{k['grenze']:.4f}")
    for t in FAMILIE:
        u = _urteil(t, fam[t], p_holm[t], k, bool(mut.get("tost_zuerst", True)))
        out[t] = {**(fam[t] or {}), "p_holm": p_holm[t], "verdict": u}
        say(f"  [{t}] {u}")
    a = k["aufloesung"]
    say(f"KONTROLLE  Aufloesung {'bestanden' if a['bestanden'] else 'NICHT BESTANDEN'}"
        + (f" (lora_r1-basis {a['delta']:+.4f}, p={a['p']:.4f})" if "delta" in a else " (fehlt)"))
    if not still:
        say("\nBESCHREIBEND (ohne Holm)")
        b = _reihe(records, "basis", feld)
        b0 = float(np.mean(list(b.values()))) if b else float("nan")
        for arm in ARME:
            w = np.array(list(_reihe(records, arm, feld).values()))
            if len(w):
                n_p, g = ERWARTETE_PARAMETER[arm], b0 - w.mean()
                say(f"  {arm:<9} {len(w)} Seeds  Mittel {w.mean():.4f}  sd {w.std(ddof=1) if len(w) > 1 else 0:.4f}"
                    f"  Gewinn {g:+.4f}" + (f" ({g / k['gewinn_lora']:.0%} von LoRA)" if k["gewinn_lora"] else "")
                    + (f"  {g / n_p:.2e}/Param" if n_p else "") + f"  trainierbar {n_p}"
                    + f"  lr {config.get('lr', {}).get(arm, {}).get('lr', '-')}")
        if fam["K1"]:
            for s in fam["K1"]["stufen"]:
                say(f"  Stufe {s['stufe']:<8} {s['delta']:+.4f}  {s['siege']}/{fam['K1']['n']} Seeds  p={s['p']:.4f}")
        v = _vergleich(_reihe(records, "kombi", feld), _reihe(records, "qire_L5", feld), 0.0)
        if v:
            say(f"  kombi - qire_L5 {v['delta']:+.4f}  p={v['p']:.4f}")
    raender = {s: v["rand"] for s, v in config.get("lr", {}).items() if v.get("rand") != "innen"}
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest (CLAUDE.md §5) -- Teil 1: Theorie, echtes Modell, Referenzgleichheit
# ---------------------------------------------------------------------------
def _ordnung_ring() -> int:
    p, _ = ca._ring(1, True)
    q, m = p.clone(), 1
    while not torch.equal(q, torch.arange(BLOCK)):
        q, m = q[p], m + 1
    return m


def _zeilen_drucken(titel: str, zeilen) -> bool:
    print("=" * 96)
    print(titel)
    ok = True
    for name, wert, rel, grenze in zeilen:
        good = wert < grenze if rel == "<" else wert > grenze
        ok &= good
        print(f"  {name:<66} {wert:.2e} {rel} {grenze:g}  {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return bool(ok)


def pruefe_theorie(threads: int = 3) -> bool:
    torch.set_num_threads(threads)
    zeilen, eye = [], torch.eye(BREITE, dtype=torch.float64)
    ordnung = _ordnung_ring()
    torch.manual_seed(0)
    with torch.no_grad():
        for lagen in LAGEN:
            d = QireDrehung(lagen, True)
            zeilen.append((f"L{lagen}: Winkel 0 -> R = I (mit Rueckpermutation)", float((_matrix(d) - eye).abs().max()), "<", 1e-12))
            if lagen % ordnung:
                d.rueck = torch.arange(BLOCK)
                zeilen.append((f"L{lagen}: Gegenprobe OHNE Rueckpermutation: Winkel 0 -> R = P^L != I",
                               float((_matrix(d) - eye).abs().max()), ">", 0.5))
            d = QireDrehung(lagen, True)
            d.theta.uniform_(-math.pi, math.pi)
            r = _matrix(d)
            zeilen.append((f"L{lagen}: R^T R = I (zufaellige Winkel)", float((r.T @ r - eye).abs().max()), "<", 1e-5))
            zeilen.append((f"L{lagen}: R blockdiagonal (3 x 256)", float((r - torch.block_diag(
                *[r[i:i + BLOCK, i:i + BLOCK] for i in range(0, BREITE, BLOCK)])).abs().max()), "<", 1e-12))
            abstand = float((r - _kron_explizit(d.theta.double().sum(0))).abs().max())
            zeilen.append((f"L{lagen}: " + ("== Kronecker (= qire_kron)" if lagen == 1 else "Gegenprobe: KEIN Kronecker-Produkt"),
                           abstand, "<" if lagen == 1 else ">", 1e-10 if lagen == 1 else 0.1))
            bank = Drehbank(24, lagen, True).double()
            bank.theta.uniform_(-math.pi, math.pi)
            bank.aktualisieren()
            abw = 0.0
            for k in (0, 11, 23):
                d = QireDrehung(lagen, True).double()
                d.theta.copy_(bank.theta[k])
                abw = max(abw, float((torch.block_diag(*[bank.rt[k]] * 3) - _matrix(d).T).abs().max()))
            zeilen.append((f"L{lagen}: Drehbank R_k^T == QireDrehung", abw, "<", 1e-10))
        zeilen.append((f"Ordnung des CNOT-Rings = {ordnung} (Info; P^L = I nur fuer L Vielfaches)", float(ordnung), ">", 0.5))
        # Echtes Modell
        d_ = daten()
        ids, maske, lab = stapel(d_["validierung"][:3])
        ref = modell("basis")
        ref.eval()
        l_ref = ref(input_ids=ids, attention_mask=maske).logits
        for arm in TRAINIERT:
            m = modell(arm)
            m.eval()
            zeilen.append((f"{arm}: Startwert -> Logits == Basismodell",
                           float((m(input_ids=ids, attention_mask=maske).logits - l_ref).abs().max()), "<", 1e-5))
            zeilen.append((f"{arm}: trainierbar {trainierbar(m)} == {ERWARTETE_PARAMETER[arm]}",
                           abs(trainierbar(m) - ERWARTETE_PARAMETER[arm]), "<", 0.5))
        # kombi: W' = W R + s B A an einer Schicht explizit
        m = modell("kombi")
        bank = _kausal(m, "kombi").drehbank
        bank.theta.uniform_(-1, 1)
        bank.aktualisieren()
        lin = _kausal(m, "kombi").model.layers[3].self_attn.v_proj
        a_, b_ = lin.lora_A["default"].weight, lin.lora_B["default"].weight
        b_.normal_(0, 0.1)
        d = QireDrehung(KOMBI_LAGEN, True)
        d.theta.copy_(bank.theta[lin.base_layer.index])
        x = torch.randn(5, BREITE, dtype=torch.float64)
        w = lin.base_layer.basis.weight.double()
        s_ = lin.scaling["default"]
        soll = x @ (w @ _matrix(d)).T + s_ * x @ a_.double().T @ b_.double().T
        ist = lin(x.float()).double()
        zeilen.append((f"kombi: v_proj Lage 3 == x (W R)^T + s x A^T B^T (Index {lin.base_layer.index})",
                       float((ist - soll).abs().max()), "<", 1e-4))
        zeilen.append(("kombi Gegenprobe: Drehung wirkt (Abstand zu x W^T + LoRA)",
                       float((ist - x @ w.T - s_ * x @ a_.double().T @ b_.double().T).abs().max()), ">", 1e-3))
        zeilen.append(("kombi Gegenprobe: LoRA wirkt (Abstand zu x (W R)^T)",
                       float((ist - x @ (w @ _matrix(d)).T).abs().max()), ">", 1e-3))
    # Gradient fliesst in Winkel UND LoRA, nicht in W
    m = modell("kombi")
    s, n, _, _ = _verlust(m, ids, maske, lab)
    (s / n).backward()
    g = _kausal(m, "kombi").drehbank.theta.grad
    zeilen.append(("kombi: Gradient an allen 24 Winkelsaetzen != 0 (Anzahl ohne)",
                   24.0 if g is None else float((g.abs().amax(dim=(1, 2)) == 0).sum()), "<", 0.5))
    zeilen.append(("kombi: Gradient an LoRA-B (alle 24) != 0 (Anzahl ohne)",
                   float(sum(p.grad is None or float(p.grad.abs().max()) == 0
                             for n_, p in m.named_parameters() if "lora_B" in n_)), "<", 0.5))
    zeilen.append(("kombi: Basisgewichte ohne Gradient",
                   float(sum(p.grad is not None for n_, p in m.named_parameters()
                             if "drehbank" not in n_ and "lora_" not in n_)), "<", 0.5))
    # Referenzgleichheit: T-005-Zelle == T-004-Zelle (3 Trainingsschritte, 40 Beispiele)
    for neu, alt in T004_PAARE.items():
        a = zelle(neu, 7, 1e-3, "validierung", threads, 3, True)
        b = ca.zelle(alt, 7, 1e-3, "validierung", threads, 3, True)
        zeilen.append((f"Referenz: zelle {neu} == T-004 {alt} (val und train, 3 Schritte)",
                       max(abs(a[METRIK] - b[METRIK]), abs(a["train_verlust_ende"] - b["train_verlust_ende"])),
                       "<", 1e-6))
    # Uebernommene Lernraten == T-004-Rohdaten (gleiche Zelle, gleiches Budget, Rand innen)
    lr_alt = json.loads(T004_ROH.read_text(encoding="utf-8"))["config"] if T004_ROH.exists() else {}
    abw = sum(lr_alt.get("lr", {}).get(T004_PAARE[a], {}) != {"lr": v, "rand": "innen"}
              for a, v in LR_UEBERNOMMEN.items())
    abw += (lr_alt.get("schritte") != SCHRITTE) + (lr_alt.get("seed_aufteilung") != ca.SEED_AUFTEILUNG)
    abw += sum(a not in T004_PAARE or a in LR_GITTER for a in LR_UEBERNOMMEN)
    abw += sum(a not in LR_GITTER and a not in LR_UEBERNOMMEN for a in TRAINIERT)
    zeilen.append(("LR-Uebernahme == T-004-Rohdaten (lr, innen, Schritte, Aufteilung), Rest getunt",
                   float(abw), "<", 0.5))
    # Seeds und Parameter
    alt = set(ca.EVAL_SEEDS) | set(ca.TUNING_SEEDS)
    zeilen.append(("Seeds frisch: disjunkt zu T-004 und Mess/Tuning disjunkt (Ueberschneidungen)",
                   float(len((set(EVAL_SEEDS) | set(TUNING_SEEDS)) & alt) + len(set(EVAL_SEEDS) & set(TUNING_SEEDS))
                         + (len(EVAL_SEEDS) != 8)), "<", 0.5))
    zeilen.append(("Parameter L1 == qire_kron, L5 == qire_5, lora == T-004 (Abweichungen)",
                   float(sum(ERWARTETE_PARAMETER[n_] != ca.ERWARTETE_PARAMETER[a_] for n_, a_ in T004_PAARE.items())),
                   "<", 0.5))
    return _zeilen_drucken("THEORIEPRUEFUNG T-005", zeilen)


def pruefe_page() -> bool:
    """Die exakte Page-Verteilung gegen Formel, Vollaufzaehlung und scipy."""
    zeilen = []
    for k, n in ((len(LAGEN), len(EVAL_SEEDS)), (4, 8), (5, 3)):
        dist, gesamt = page_verteilung(k, n)
        mu = sum(v * c for v, c in dist.items()) / gesamt
        var = sum((v - mu) ** 2 * c for v, c in dist.items()) / gesamt
        zeilen.append((f"k={k}, n={n}: Summe == k!^n", float(abs(sum(dist.values()) - gesamt)), "<", 0.5))
        zeilen.append((f"k={k}, n={n}: E[L] == n k (k+1)^2 / 4", abs(mu - n * k * (k + 1) ** 2 / 4), "<", 1e-6))
        zeilen.append((f"k={k}, n={n}: Var[L] == n k^2 (k+1)(k^2-1) / 144",
                       abs(var - n * k * k * (k + 1) * (k * k - 1) / 144), "<", 1e-6))
    rng = np.random.default_rng(5)
    v = rng.normal(size=(2, 4)) - 0.4 * np.arange(4)
    stat = page_test(v)["L"]
    treffer = sum(page_test(np.vstack([v[0][list(p1)], v[1][list(p2)]]))["L"] >= stat - 1e-9
                  for p1 in itertools.permutations(range(4)) for p2 in itertools.permutations(range(4)))
    zeilen.append(("Vollaufzaehlung n=2, k=4 (576 Umordnungen) == exakt",
                   abs(treffer / 576 - page_test(v)["p"]), "<", 1e-12))
    from scipy.stats import page_trend_test
    for i in range(3):
        v = rng.normal(size=(len(EVAL_SEEDS), len(LAGEN))) - 0.3 * i * np.arange(len(LAGEN))
        zeilen.append((f"scipy page_trend_test(exact) == eigene Faltung (Trend {0.3 * i:.1f})",
                       abs(page_trend_test(-v, method="exact").pvalue - page_test(v)["p"]), "<", 1e-9))
    k, n = len(LAGEN), len(EVAL_SEEDS)
    perfekt = -np.tile(np.arange(k, dtype=float), (n, 1)) + rng.normal(0, 1e-3, (n, 1))
    zeilen.append(("perfekt fallend -> p == (1/k!)^n (relativ)",
                   abs(page_test(perfekt)["p"] * math.factorial(k) ** n - 1.0), "<", 1e-9))
    zeilen.append(("perfekt steigend -> p == 1", abs(page_test(-perfekt)["p"] - 1.0), "<", 1e-12))
    zeilen.append(("Erreichbarkeit: Page-min x 3 < 0,05", 3 * math.factorial(k) ** -n, "<", 0.05))
    zeilen.append(("Erreichbarkeit: IUT-min (Wilcoxon einseitig, 8) < 0,05/3",
                   _wilcoxon(-np.arange(1, 9, dtype=float), "less"), "<", IUT_SCHWELLE))
    return _zeilen_drucken("PRUEFUNG PAGE-TEST (exakt, keine Asymptotik)", zeilen)


def test_gitterverlaengerung() -> bool:
    """Regel 1(b) in beide Richtungen auf dem T-005-Startgitter."""
    g = LR_START
    faelle = [("Optimum im Startgitter", 1e-3, "innen", 1e-3),
              ("Verlaengerung oben", 9e-3, "innen", 9e-3),
              ("Verlaengerung unten", 1e-4, "innen", 1e-4),
              ("ausserhalb oben", 300.0, "oben", g[-1] * 27),
              ("ausserhalb unten", 1e-9, "unten", g[0] / 27)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen, Startgitter", g)
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        best, rand, _ = sweep_lr("t005", "qire_L10", g, (1, 2), ERWEITERUNGEN,
                                 lambda lr, s: -(3.0 + math.log10(lr / opt) ** 2))
        good = abs(math.log10(best / rate_soll)) < 1e-6 and rand == rand_soll
        ok &= good
        print(f"  {name:<28} -> lr {best:<10.4g} Rand {rand:<6} {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


# ---------------------------------------------------------------------------
# Pflichtvortest -- Teil 2: jeden Ausgang einzeln erzwingen
# ---------------------------------------------------------------------------
NIVEAU, NIVEAU_ALLE = 3.00, 3.50
_V = {"basis": 0.0, "qire_L1": -0.04, "qire_L2": -0.07, "qire_L3": -0.09, "qire_L5": -0.12,
      "qire_L10": -0.15, "lora_r1": -0.27, "kombi": -0.30}
VERSATZ = {a: _V[a] for a in ARME}
GLEICH_L = {a: -0.08 for a in QIRE}
FALLEND_L = dict(zip(QIRE, [VERSATZ[a] for a in QIRE][::-1]))


def _kunst(seed: int, streuung: float = 0.01, versatz=None, weg: Sequence[str] = (),
           weg_seed: Optional[Dict[str, int]] = None) -> Recorder:
    """basis deterministisch (ein Wert fuer alle Seeds, wie im Lauf). Trainierte
    Arme: gemeinsamer Seed-Anteil + EIGENES Rauschen. `val_verlust_alle` (plau-
    sibles Falschfeld) mit UMGEKEHRTEN Versaetzen und eigenem Rauschen."""
    def rng(*t):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, t)).encode())])
    vs = {**VERSATZ, **(versatz or {})}
    rec = Recorder("T-005-vortest", "synthetisch", config={"lr": {}})
    for s in EVAL_SEEDS:
        grund = rng("grund", s).normal(0, 0.005)
        for arm, v in vs.items():
            if arm in weg or (weg_seed or {}).get(arm) == s:
                continue
            eig = 0.0 if arm == "basis" else grund + rng(arm, s).normal(0, streuung)
            alle = 0.0 if arm == "basis" else eig + rng("alle", arm, s).normal(0, streuung)
            rec.add(arm=arm, seed=s, lr=0.001, menge="validierung", val_verlust=NIVEAU + v + eig,
                    val_verlust_alle=NIVEAU_ALLE - v + alle, trainierbar=ERWARTETE_PARAMETER[arm])
    return rec


FAELLE = [
    # (Name, Test, kwargs, Code, Zweigtext, Ziehungen, Anteil)
    ("M-S   streng monoton, kleines Rauschen", "K1", dict(streuung=0.003), "STRENG_MONOTON", "jede weitere", 5, 1.0),
    ("M-P   Plateau: L10 = L5", "K1", dict(streuung=0.003, versatz={"qire_L10": -0.12}),
     "WAECHST_MIT_PLATEAU", "Stufe L5->L10", 20, 0.9),
    ("M-R   L10 schlechter als L5 (Umkehr am Ende)", "K1", dict(streuung=0.003, versatz={"qire_L10": -0.10}),
     "WAECHST_MIT_PLATEAU", "Stufe L5->L10", 5, 1.0),
    ("M-N   alle L gleich, Rauschen 0,01", "K1", dict(versatz=GLEICH_L), "KEIN_TREND", "kein belegter Anstieg", 40, 0.8),
    ("M-F   Gewinn FAELLT mit L", "K1", dict(versatz=FALLEND_L), "KEIN_TREND", "Beschreibend FAELLT", 5, 1.0),
    ("M-A   Aufloesung faellt (lora = basis)", "K1", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG",
     "KEIN Nullbefund", 40, 0.8),
    ("M-An  lora-Zelle fehlt", "K1", dict(weg=["lora_r1"]), "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
    ("M-X   qire_L2 fehlt", "K1", dict(weg=["qire_L2"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("M-Xs  ein Seed fehlt in qire_L5", "K1", dict(weg_seed={"qire_L5": EVAL_SEEDS[2]}), "HU_UNVOLLSTAENDIG",
     "Seeds ungleich", 5, 1.0),
    ("Z-L   LoRA 0,12 besser als qire_L10", "K2", {}, "LORA_BESSER", "nicht gleich weit", 5, 1.0),
    ("Z-Q   qire_L10 besser als LoRA", "K2", dict(versatz={"qire_L10": -0.40}), "QIRE_BESSER",
     "staerker als LoRA", 5, 1.0),
    ("Z-G   gleich, kleines Rauschen", "K2", dict(streuung=0.003, versatz={"qire_L10": -0.27}),
     "GLEICHWERTIG", "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("Z-Gs  +0,005 bei winzigem Rauschen: signifikant UND gleichwertig", "K2",
     dict(streuung=0.0005, versatz={"qire_L10": -0.265}), "GLEICHWERTIG", "zugleich signifikant", 5, 1.0),
    ("Z-U   gleich, grosses Rauschen", "K2", dict(streuung=0.08, versatz={"qire_L10": -0.27}),
     "UNENTSCHIEDEN", "weder Unterschied", 40, 0.8),
    ("Z-X   qire_L10 fehlt", "K2", dict(weg=["qire_L10"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("Z-A   Aufloesung faellt", "K2", dict(versatz={"lora_r1": 0.0, "qire_L10": 0.0}), "HU_AUFLOESUNG",
     "KEIN Nullbefund", 40, 0.8),
    ("E-J   kombi 0,03 vor lora", "K3", dict(streuung=0.005), "ERGAENZT", "QIRE ergaenzt LoRA", 5, 1.0),
    ("E-N   kombi 0,05 HINTER lora", "K3", dict(versatz={"kombi": -0.22}), "KEIN_NACHWEIS", "sogar HOEHER", 5, 1.0),
    ("E-U   kombi = lora, grosses Rauschen", "K3", dict(streuung=0.05, versatz={"kombi": -0.27}),
     "KEIN_NACHWEIS", "keine belegte Ergaenzung", 40, 0.8),
    ("E-Ar  lora SCHLECHTER als basis: Trennung zaehlt", "K3", dict(versatz={"lora_r1": 0.10, "kombi": 0.05}),
     "ERGAENZT", "QIRE ergaenzt LoRA", 5, 1.0),
    ("E-X   kombi fehlt", "K3", dict(weg=["kombi"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("X-K   qire_L2 fehlt -> K3 trotzdem gerechnet", "K3", dict(weg=["qire_L2"], streuung=0.005),
     "ERGAENZT", "QIRE ergaenzt LoRA", 5, 1.0),
    ("X-Z   kombi fehlt -> K1 trotzdem gerechnet", "K1", dict(weg=["kombi"], streuung=0.003),
     "STRENG_MONOTON", "jede weitere", 5, 1.0),
]


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


def pruefe_generator() -> bool:
    rec = _kunst(seed=4242)
    r = lambda a, f: np.array(list(_reihe(rec.records, a, f).values()))  # noqa: E731
    diff_sd = [float(np.std(r(a, "val_verlust_alle") - r(a, METRIK), ddof=1)) for a in TRAINIERT]
    d_m = r("qire_L10", METRIK) - r("qire_L1", METRIK)
    d_a = r("qire_L10", "val_verlust_alle") - r("qire_L1", "val_verlust_alle")
    sd_kl = float(np.std(r("qire_L10", METRIK) - r("lora_r1", METRIK), ddof=1))
    soll = math.sqrt(2) * 0.01
    pruef = [("val_verlust_alle - val_verlust NICHT konstant (sd je Arm > 0)", min(diff_sd) > 1e-4),
             ("gepaarte Differenz L10 - L1 kippt mit dem Feld", d_m.mean() < 0 < d_a.mean()),
             ("basis deterministisch (sd 0), trainierte Arme streuen",
              float(np.std(r("basis", METRIK))) == 0 and float(np.std(d_m)) > 1e-3),
             (f"eigenes Rauschen: sd(qire_L10 - lora) {sd_kl:.4f} ~ {soll:.4f}", 0.5 * soll < sd_kl < 2 * soll),
             ("Standardfall ist streng monoton geordnet (Versaetze fallen mit L)",
              all(VERSATZ[a] > VERSATZ[b] for a, b in zip(list(QIRE), list(QIRE)[1:])))]
    for name, good in pruef:
        print(f"  {'OK    ' if good else 'FEHLER'} {name}")
    return all(g for _, g in pruef)


MUTATIONEN = [("Metrikfeld val_verlust_alle statt val_verlust", {"feld": "val_verlust_alle"}),
              ("Signifikanz VOR TOST (K2)", {"tost_zuerst": False}),
              ("Aufloesung verlangt Richtung statt Trennung", {"richtung": True}),
              ("K3 einseitig in FALSCHER Richtung", {"einseitig": "greater"}),
              ("TOST-Grenze absolut 0,2 statt relativ", {"grenze_absolut": True}),
              ("K1 Arme in falscher Reihenfolge (L absteigend)", {"reihenfolge": True}),
              ("K1 IUT mit min statt max (eine Stufe reicht)", {"iut_min": True})]


def vortest() -> bool:
    print("=" * 96)
    print("VORTEST T-005 -- jeder Ausgang einzeln, ueber mehrere Ziehungen, Marke UND Zweigtext")
    ok = _szenarien()
    print("GENERATOR")
    ok &= pruefe_generator()
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
# Zeitmessung -- kurz gemessen, an den Istzeiten des T-004-Laufs kalibriert
# ---------------------------------------------------------------------------
def zeit(threads: int = 3, schritte: int = 4) -> Dict[str, float]:
    print("=" * 96)
    print(f"ZEITMESSUNG T-005 -- {threads} Threads, {schritte} Schritte je Arm")
    daten()
    t, a, ov = {}, {}, {}
    for arm in ARME:
        r = zelle(arm, 1, 1e-3, "validierung", threads, 0 if arm == "basis" else schritte, True)
        t[arm], a[arm] = r["s_je_schritt"], r["s_auswertung"] / 40
        ov[arm] = max(0.0, r["sekunden"] - r["s_auswertung"] - t[arm] * r["schritte"])
        print(f"  {arm:<9} {t[arm]:>6.3f} s/Schritt   Auswertung {a[arm]:.3f} s/Beispiel   trainierbar "
              f"{r['trainierbar']}", flush=True)
    f_t = f_a = 1.0
    ov_ist = float(np.mean(list(ov.values())))
    if T004_ROH.exists():
        roh = json.loads(T004_ROH.read_text(encoding="utf-8"))["records"]
        q_t, q_a, o = [], [], []
        for neu, alt in T004_PAARE.items():
            rs = [x for x in roh if x["arm"] == alt]
            it, ia = np.mean([x["s_je_schritt"] for x in rs]), np.mean([x["s_auswertung"] for x in rs]) / N_VAL
            q_t.append(it / t[neu])
            q_a.append(ia / a[neu])
            o += [x["sekunden"] - x["s_je_schritt"] * SCHRITTE - x["s_auswertung"] for x in rs]
            print(f"  T-004-Ist {alt:<9} {it:.3f} s/Schritt, {ia:.3f} s/Beispiel ({rs[0]['threads']} Threads)"
                  f"  -> Faktor {it / t[neu]:.2f} / {ia / a[neu]:.2f}")
        f_t, f_a, ov_ist = float(np.exp(np.mean(np.log(q_t)))), float(np.exp(np.mean(np.log(q_a)))), float(np.mean(o))
        print(f"  Kalibrierung (geom. Mittel): Training x{f_t:.2f}, Auswertung x{f_a:.2f}, Nebenzeit {ov_ist:.0f} s/Zelle")

    def zelle_s(arm, n_ausw):
        return SCHRITTE * t[arm] * f_t + n_ausw * a[arm] * f_a + ov_ist

    print(f"  Prognose Messzelle (SCHRITTE={SCHRITTE}, 1.000 Val.): "
          + ", ".join(f"{x} {zelle_s(x, N_VAL) / 60:.1f} min" for x in TRAINIERT))
    mess = sum(len(EVAL_SEEDS) * zelle_s(x, N_VAL) for x in TRAINIERT) + N_VAL * a["basis"] * f_a + ov_ist
    for name, getunt in (("PLAN (Tuning nur " + ", ".join(LR_GITTER) + ")", tuple(LR_GITTER)),
                         ("zum Vergleich: alle Arme getunt", TRAINIERT)):
        for raten in (3, 4):
            tune = sum(len(TUNING_SEEDS) * raten * zelle_s(x, N_ABSTIMM) for x in getunt)
            print(f"  {name:<50} {raten} Raten/Arm: GESAMT {(mess + tune) / 3600:.1f} h "
                  f"(Messung {mess / 3600:.1f} h, Tuning {tune / 3600:.1f} h)")
    print("  [kalibriert auf T-004-Bedingungen: 6 Threads, freier Rechner]")
    return {"f_t": f_t, "f_a": f_a, **{f"{k}_t": v for k, v in t.items()}}


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        raise SystemExit(0 if (pruefe_theorie() & pruefe_page() & test_gitterverlaengerung() & vortest()) else 1)
    elif befehl == "vortest_statistik":            # schneller Teil ohne Modell
        raise SystemExit(0 if (pruefe_page() & test_gitterverlaengerung() & vortest()) else 1)
    elif befehl == "zeit":
        zeit(*(int(a) for a in sys.argv[2:4]))
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else LAUF_ID)
    elif befehl == "lauf":
        if not (pruefe_theorie() and pruefe_page() and test_gitterverlaengerung() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print(f"  Nachholen: python -m quantum_sim.chatbot_kurve auswertung {LAUF_ID}")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | vortest_statistik | zeit [threads] [schritte] | lauf | auswertung [ID|Pfad]")
