"""RESEARCH ID: T-007 (Institut: I-043) -- Fastfood-Adapter im Sprachmodell.

Baut auf T-004/T-005/T-006 auf (`chatbot_adapter.py`, `chatbot_kurve.py`, `chatbot_skalierung.py`;
dort wird NICHTS geaendert). Modell (LLaeMmlein-120M), Daten, Aufteilung, Schritte (300), Batch 4,
Adam, Clip 1,0, Verlust und Metrik `val_verlust` sind dieselben. Neu: der Fastfood-Adapter, Arme,
Seeds, Hypothesen, Auswertung. Vorregistrierung ausfuehrlich: ENTWURF_T007_FASTFOOD_ADAPTER.md
(massgeblich ist dieser Modulkopf, Stand Commit). Herleitung: INSTITUT/LABORBUCH.md I-044 ERGEBNIS
(Struktur traegt dort, wo die Eingabe schon gute Merkmale hat).

FRAGE  (1) Ersetzt eine strukturierte Fastfood-Abbildung mit O(d) trainierbaren Parametern den
  niedrigen Rang (LoRA r=1) als Adapter -- bei GLEICHER Parameterzahl?

ADAPTER  Je Projektionsmatrix W (q_proj 768->768, v_proj 768->256; 24 Matrizen) y = W x + dW x mit
    dW x = c * (H D_aus ... H D_ein) x     (H = Walsh-Hadamard 256 x 256, orthonormal, blockweise;
    die 768 Eingangsdimensionen sind 3 Bloecke zu 256; wie die QIRE-Bloecke)
  Stufen je Seite s (ff_adapter s=1, ff_adapter_breit s=2): s trainierte Diagonalen D_ein (Laenge d_in),
  je gefolgt von H; ist d_aus < d_in (v_proj), werden die d_in/d_aus Bloecke summiert (Faktor
  1/sqrt(Bloecke)); dann s trainierte Diagonalen D_aus (Laenge d_aus), je gefolgt von H.
  c = FF_SKALA = 2,0 fest (analog LoRA alpha/r = 2; nicht trainiert).
  Init: alle Diagonalen Zufallsvorzeichen +-1 (Generator je Seed), ausser der LETZTEN Diagonale = 0
  => dW = 0 exakt, Start = Basismodell. Gradient fliesst erst nach dem ersten Schritt auch in die
  vorderen Diagonalen (Vortest).
  PARAMETER  s * (d_in + d_aus) je Matrix. ff_adapter = 12 * ((768+768) + (768+256)) = 30.720
  == lora_r1 EXAKT (A: d_in, B: d_aus je Matrix) -> fairer Paritaetsvergleich.
  ff_adapter_breit = 61.440 (2 x lora_r1; Tiefe/Breite-Test, kein Paritaetsarm).

BORN-ZWEIG (Aenderung 07.10.2026, vor jeder Zahl; Grund: I-045-v2 zeigt Wechselwirkung Struktur x z^2)
  ff_born  dW x = c * D3 * H( D2 * fold( (H(D1 * x))^2 ) )     c = 2,0 fest, H blockweise 256, fold wie oben
  ff_gelu  dasselbe mit GELU statt z^2 (Kontrolle der Wechselwirkung, gleiche Struktur, gleiche Parameter)
  Init D1, D2 = Zufallsvorzeichen, D3 = 0 -> dW = 0, Start = Basismodell. Parameter d_in + 2 d_aus je Matrix
  = 12 * ((768+1536) + (768+512)) = 43.008 = 1,40 x lora_r1 (30.720), 0,70 x lora_r2 (61.440): NICHT
  parametergleich zu LoRA; lora_r1 und lora_r2 klammern den Born-Zweig ein (offen gelegt). Nahe Verwandte:
  Needle-2-Hadamard-MLP (SiLU), siehe INSTITUT/recherche/2026-10-07_literatur_struktur_quadrat.md.
  ff_adapter_breit (61.440) GESTRICHEN (mit ihm Kern ca. 11 h > 10 h); H3 entfaellt.

ARME
    basis             kein Training                                         0
    qire_L10          Bruecke und POSITIVKONTROLLE (T-006: 2,4111)      1.920
    lora_r1           Referenz (wie T-004/T-005/T-006)                 30.720
    ff_adapter        Fastfood, 2 Diagonalstufen (1 je Seite)          30.720
    ff_born           Born-Zweig, z^2                                  43.008
    ff_gelu           Born-Zweig, GELU (Kontrolle)                     43.008
    lora_r2           obere LoRA-Klammer, nur beschreibend, lr = 1e-3
                      von lora_r1 uebernommen (NICHT getunt)           61.440

VORREGISTRIERUNG (vor jeder T-007-Zahl; Aenderungen erst ins Tagebuch)
  Seeds FRISCH: Messung 70001-70008, Tuning 70401/70402 (disjunkt zu T-004 40001-40008/40401-40402,
  T-005 50001-50008/50401-50402, T-006 60001-60008/60401-60402; `pruefe_theorie` prueft das).
  SCHRITTE = 300 fuer alle trainierten Arme.
  Lernraten: qire_L10, lora_r1 uebernommen aus T-005 (1e-3, innen; wie T-006). ff_adapter, ff_born, ff_gelu
  getunt (gleiches Gitter): Startgitter 1e-4/3e-4/1e-3/3e-3/1e-2, Verlaengerung x3 bzw. :3 in BEIDE
  Richtungen, hoechstens 3 Schritte (`sweep_lr`), Tuning-Seeds. Randpruefung: Optimum nie am Rand, sonst
  "ACHTUNG Rand". ALLE Tuningzellen und Mittel je Rate in den ROHDATEN (config["tuning"]); `rand_aus_tuning`
  rechnet den Randvermerk unabhaengig nach (q_audit).
  Familie, Holm ueber GENAU 4 (alpha 0,05):
    H1  ff_adapter - lora_r1, zweiseitig (Wilcoxon) + TOST +-20 % x |Gewinn lora_r1|; TOST vor Signifikanz
          FF_BESSER / GLEICHWERTIG / LORA_BESSER / UNENTSCHIEDEN
    H2  ff_adapter < qire_L10 (einseitig). NICHT parametergleich: Bruecke.
          FF_BESSER_ALS_QIRE / KEIN_NACHWEIS
    H4  ff_born < ff_gelu (einseitig; Wechselwirkung Struktur x Quadrat, gleiche Parameter)
          BORN_BESSER / KEIN_NACHWEIS
    H5  ff_born vs lora_r1, zweiseitig + TOST +-20 % x |Gewinn lora_r1| (lora_r1 = parametergleichste
        Referenz, Born hat 1,4 x so viele Parameter; lora_r2 als Gegenstueck nur beschreibend)
          BORN_BESSER / GLEICHWERTIG / LORA_BESSER / UNENTSCHIEDEN
  Rangfolge: HU_UNVOLLSTAENDIG -> HU_DIVERGENZ -> HU_POSKONTROLLE (qire_L10 Mittel 2,4111 +- 0,003)
    -> HU_AUFLOESUNG (lora_r1 trennt sich von basis, Wilcoxon zweiseitig p < 0,05) -> Urteil.
  Erreichbarkeit: 8 Seeds, Wilcoxon einseitig min 0,0039, zweiseitig 0,0078, je x4 (Holm) <= 0,031 < 0,05.
  Beschreibend (ohne Holm): Gewinn je Arm, Anteil am LoRA-Gewinn, Gewinn je Parameter, s/Schritt,
    ff_born - lora_r2, ff_born - ff_adapter, ff_gelu - lora_r1, Randvermerke.
  Laufzeit: Ziel Kern <= 10 h (`prognose`: 9,0 h, 9,7 h mit Verlaengerung); misst `zeit` mehr, entfaellt lora_r2 (letzter Arm).

Run (aus dem Ordner A):
    python -m quantum_sim.chatbot_fastfood vortest            # Pflicht (Modell noetig)
    python -m quantum_sim.chatbot_fastfood vortest_statistik  # ohne Modell
    python -m quantum_sim.chatbot_fastfood zeit [threads] [schritte]
    python -m quantum_sim.chatbot_fastfood lauf               # Malte entscheidet
    python -m quantum_sim.chatbot_fastfood auswertung [ID|Pfad]
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
import torch.nn as nn

from qire_ai.equivalence import holm
from qire_ai.result_store import RESULTS_DIR, Recorder

from . import chatbot_adapter as ca
from . import chatbot_kurve as ck
from . import chatbot_skalierung as cs
from .chatbot_adapter import (BATCH, BLOCK, BREITE, CLIP, ZIELE, _reihe, _schl, _verlust, _vergleich,
                              auswerten, daten, stapel, trainierbar)
from .kernaussage import sweep_lr

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt
# ---------------------------------------------------------------------------
LAUF_ID = "T-007"
FF_ARME = {"ff_adapter": 1}                                 # lineare Fastfood-Arme: Diagonalstufen JE SEITE
BORN_ARME = {"ff_born": "z2", "ff_gelu": "gelu"}            # Mischung -> Nichtlinearitaet -> Rueckmischung
FF_ALLE = (*FF_ARME, *BORN_ARME)
# ff_adapter_breit (4 Stufen, 61.440) ist seit 07.10. (Aenderung der Vorregistrierung) GESTRICHEN: mit ihm laege
# die Kernzeit bei ca. 11 h > 10 h. H3 entfaellt; Familie = H1, H2, H4, H5.
ARME = ("basis", "qire_L10", "lora_r1", "lora_r2", *FF_ALLE)
# Reihenfolge = Wichtigkeit: Positivkontrolle zuerst, dann die Referenz, dann H1/H2, H4/H5; lora_r2 (nur
# beschreibend) ZULETZT -- bei Zeitnot ist es der Arm, der wegfaellt.
TRAINIERT = ("qire_L10", "lora_r1", "ff_adapter", "ff_born", "ff_gelu", "lora_r2")
SCHRITTE = ca.SCHRITTE                                      # 300
FF_SKALA = 2.0                                              # c, fest (wie LoRA alpha/r)
WHT_A, WHT_B = 16, 16                                       # H_256 = H_16 (x) H_16
assert WHT_A * WHT_B == BLOCK
N_MATRIZEN = 24
_DIMS = (("q_proj", 768, 768), ("v_proj", 768, 256))        # (Name, d_in, d_aus)
ERWARTETE_PARAMETER = {"basis": 0, "qire_L10": ck.ERWARTETE_PARAMETER["qire_L10"],
                       "lora_r1": ca.ERWARTETE_PARAMETER["lora_r1"],
                       "lora_r2": 2 * ca.ERWARTETE_PARAMETER["lora_r1"],
                       **{a: 12 * s * sum(di + do for _, di, do in _DIMS) for a, s in FF_ARME.items()},
                       # Born-Zweig: D1 (d_in) + D2 (d_aus) + D3 (d_aus) je Matrix
                       **{a: 12 * sum(di + 2 * do for _, di, do in _DIMS) for a in BORN_ARME}}
EVAL_SEEDS = tuple(range(70001, 70009))
TUNING_SEEDS = (70401, 70402)
LR_START = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
T005_ROH_PATTERN = cs.T005_ROH_PATTERN
T005_ROH = cs.T005_ROH
LR_UEBERNOMMEN = {"qire_L10": 0.001, "lora_r1": 0.001}      # aus T-005 (config["lr"], innen)
LR_ABGELEITET = {"lora_r2": 0.001}                          # NICHT getunt: von lora_r1 uebernommen (nur beschreibend)
LR_GITTER = {a: LR_START for a in TRAINIERT if a not in LR_UEBERNOMMEN and a not in LR_ABGELEITET}
ERWEITERUNGEN = 3
REL_GRENZE = ca.REL_GRENZE                                  # 0,20
FAMILIE = ("H1", "H2", "H4", "H5")
ALPHA = 0.05
METRIK = "val_verlust"
POS_SOLL, POS_TOL = 2.4111, 0.003                           # Positivkontrolle qire_L10 (T-006, 8 Seeds)
ADAPTER_DIR = RESULTS_DIR.parent / "t007_adapter"
ZWISCHEN_DIR = RESULTS_DIR.parent / "t007_zwischenstand"


# ---------------------------------------------------------------------------
# Der Fastfood-Adapter
# ---------------------------------------------------------------------------
def walsh(n: int) -> torch.Tensor:
    """Orthonormale Sylvester-Hadamard-Matrix 2^n x 2^n."""
    h = torch.ones(1, 1)
    for _ in range(n):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(2 ** n)


class FFAdapter(nn.Module):
    """dW x = c * (H D_aus_s ... H D_aus_1) fold (H D_ein_s ... H D_ein_1) x, blockweise H_256.
    `stufen` = Diagonalen JE SEITE; Parameter = stufen * (d_in + d_aus). Letzte Diagonale 0 -> dW = 0."""

    def __init__(self, d_in: int, d_aus: int, stufen: int, gen: torch.Generator, skala: float = FF_SKALA):
        super().__init__()
        if d_in % BLOCK or d_aus % BLOCK or d_in % d_aus:
            raise ValueError(f"d_in {d_in}, d_aus {d_aus}: Vielfache von {BLOCK}, d_aus teilt d_in verlangt")
        self.d_in, self.d_aus, self.stufen, self.skala = d_in, d_aus, stufen, skala
        self.fold = d_in // d_aus
        self.register_buffer("ha", walsh(int(math.log2(WHT_A))))
        self.register_buffer("hb", walsh(int(math.log2(WHT_B))))

        def zeichen(n):
            return (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()
        self.ein = nn.ParameterList([nn.Parameter(zeichen(d_in)) for _ in range(stufen)])
        aus = [zeichen(d_aus) for _ in range(stufen)]
        aus[-1] = torch.zeros(d_aus)
        self.aus = nn.ParameterList([nn.Parameter(a) for a in aus])

    def _wht(self, z: torch.Tensor) -> torch.Tensor:
        form = z.shape
        return (self.ha @ z.reshape(-1, WHT_A, WHT_B) @ self.hb).reshape(form)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for d in self.ein:
            z = self._wht(z * d)
        if self.fold > 1:
            z = z.reshape(*z.shape[:-1], self.fold, self.d_aus).sum(-2) / math.sqrt(self.fold)
        for d in self.aus:
            z = self._wht(z * d)
        return self.skala * z


class FFBorn(nn.Module):
    """Mischung -> Quadrat (bzw. GELU) -> Rueckmischung, blockweise H_256, trainierte Diagonalen:
        dW x = c * D3 * H( D2 * fold( act( H( D1 * x ) ) ) )         act = z^2 (ff_born) oder GELU (ff_gelu)
    D1 (d_in), D2 (d_aus), D3 (d_aus); Parameter d_in + 2 d_aus je Matrix. Ist d_aus < d_in (v_proj), werden
    nach act die d_in/d_aus Bloecke summiert (Faktor 1/sqrt). Init: D1, D2 = Zufallsvorzeichen, D3 = 0 -> dW = 0.
    z^2 ist Born-Regel-artig (Betragsquadrat der gemischten Amplituden); klassisch gerechnet (I-038)."""

    def __init__(self, d_in: int, d_aus: int, art: str, gen: torch.Generator, skala: float = FF_SKALA):
        super().__init__()
        if d_in % BLOCK or d_aus % BLOCK or d_in % d_aus:
            raise ValueError(f"d_in {d_in}, d_aus {d_aus}: Vielfache von {BLOCK}, d_aus teilt d_in verlangt")
        if art not in ("z2", "gelu"):
            raise ValueError(art)
        self.d_in, self.d_aus, self.art, self.skala = d_in, d_aus, art, skala
        self.fold = d_in // d_aus
        self.register_buffer("ha", walsh(int(math.log2(WHT_A))))
        self.register_buffer("hb", walsh(int(math.log2(WHT_B))))

        def zeichen(n):
            return (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()
        self.d1 = nn.Parameter(zeichen(d_in))
        self.d2 = nn.Parameter(zeichen(d_aus))
        self.d3 = nn.Parameter(torch.zeros(d_aus))

    _wht = FFAdapter._wht

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._wht(x * self.d1)
        z = z * z if self.art == "z2" else torch.nn.functional.gelu(z)
        if self.fold > 1:
            z = z.reshape(*z.shape[:-1], self.fold, self.d_aus).sum(-2) / math.sqrt(self.fold)
        return self.skala * self.d3 * self._wht(z * self.d2)


class FFLinear(nn.Module):
    """y = W x + b + dW x; W eingefroren."""

    def __init__(self, basis: nn.Linear, ff: FFAdapter):
        super().__init__()
        self.basis, self.ff = basis, ff

    def forward(self, x):
        return self.basis(x) + self.ff(x)


def ff_einbauen(m, stufen, seed: int):
    """Adapter auf q_proj und v_proj aller Lagen. Zeichen aus eigenem Generator (Seed), feste Reihenfolge.
    `stufen`: int -> FFAdapter (linear); "z2" / "gelu" -> FFBorn."""
    gen = torch.Generator().manual_seed(seed)
    for schicht in m.model.layers:
        for name in ZIELE:
            lin = getattr(schicht.self_attn, name)
            ad = (FFBorn(lin.in_features, lin.out_features, stufen, gen) if isinstance(stufen, str)
                  else FFAdapter(lin.in_features, lin.out_features, stufen, gen))
            setattr(schicht.self_attn, name, FFLinear(lin, ad))
    return m


def modell(arm: str, seed: int = 0):
    if arm in FF_ALLE:
        return ff_einbauen(ca.modell("basis"), {**FF_ARME, **BORN_ARME}[arm], seed)
    if arm == "lora_r2":                                    # parametergleiche Obergrenze, alpha/r = 2 wie lora_r1
        from peft import LoraConfig, get_peft_model
        m = ca.modell("basis")
        return get_peft_model(m, LoraConfig(r=2, lora_alpha=2 * ca.LORA_ALPHA // ca.LORA_R, lora_dropout=0.0,
                                            target_modules=list(ZIELE), bias="none"))
    if arm in ("basis", "lora_r1"):
        return ca.modell(arm)
    if arm == "qire_L10":
        return ck.modell(arm)
    raise ValueError(arm)


def adapter_zustand(m, arm: str) -> Dict[str, torch.Tensor]:
    if arm in FF_ALLE or arm == "lora_r2":
        return {k: v.detach().clone() for k, v in m.named_parameters() if v.requires_grad}
    return ck.adapter_zustand(m, arm)


# ---------------------------------------------------------------------------
# Eine Zelle -- Trainingsschleife Zeile fuer Zeile wie `chatbot_kurve.zelle`
# (Gleichheit mit der T-005-Zelle fuer qire_L10/lora_r1 prueft der Vortest bitgenau)
# ---------------------------------------------------------------------------
def zelle(arm: str, seed: int, lr: float, menge: str = "validierung", threads: int = 6,
          schritte: int = SCHRITTE, zeitmessung: bool = False) -> Dict[str, object]:
    t0 = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    d = daten()
    m = modell(arm, seed)
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
           "trainierbar": zahl, "stufen": FF_ARME.get(arm, 0), "art": BORN_ARME.get(arm, "linear"),
           "s_je_schritt": round(t_train, 3), "s_auswertung": round(time.perf_counter() - t2, 2),
           "sekunden": round(time.perf_counter() - t0, 2), "threads": threads}
    out["_adapter"] = adapter_zustand(m, arm) if arm != "basis" else None
    return out


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
    return {"basis_t004": ca.konfiguration(), "arme": list(ARME), "trainiert": list(TRAINIERT),
            "ff_stufen": FF_ARME, "born_arme": BORN_ARME, "ff_skala": FF_SKALA,
            "lr_abgeleitet": LR_ABGELEITET, "schritte": SCHRITTE,
            "seeds": list(EVAL_SEEDS), "tuning_seeds": list(TUNING_SEEDS),
            "lr_gitter": {k: list(v) for k, v in LR_GITTER.items()}, "erweiterungen": ERWEITERUNGEN,
            "lr_uebernommen": LR_UEBERNOMMEN, "lr_quelle": T005_ROH_PATTERN,
            "rel_grenze": REL_GRENZE, "familie": list(FAMILIE), "pos": [POS_SOLL, POS_TOL],
            "parameter": ERWARTETE_PARAMETER}


def _adapter_pfad(lauf_id: str, arm: str, seed):
    return ADAPTER_DIR / lauf_id / f"{arm}_{seed}.pt"


def rand_aus_tuning(tuning: Dict[str, float], arm: str, seeds: Sequence[int] = TUNING_SEEDS):
    """Randvermerk UNABHAENGIG aus den Rohdaten (config["tuning"]): (beste Rate, Vermerk, Mittel je Rate).
    Wie `sweep_lr`: "innen" nur, wenn hinter dem Optimum eine schlechtere Rate gemessen ist."""
    je: Dict[float, list] = {}
    for k, v in tuning.items():
        a, s, lr = k.split("|")
        if a == arm and int(s) in seeds:
            je.setdefault(float(lr), []).append(float(v))
    mittel = {lr: float(np.mean(v)) for lr, v in je.items() if len(v) == len(seeds)}
    if not mittel:
        return None, "fehlt", {}
    best = min(mittel, key=mittel.get)                       # Verlust: klein = gut
    return best, ("innen" if min(mittel) < best < max(mittel) else ("unten" if best == min(mittel) else "oben")), mittel


def run(lauf_id: str = LAUF_ID, threads: int = 6) -> Recorder:
    konfig = konfiguration()
    z = Zustand(lauf_id, konfig)
    rec = Recorder(experiment_id=lauf_id,
                   description=("T-007 (I-043): Fastfood-Adapter ff_adapter (30.720), Born-Zweig ff_born (43.008, z^2), "
                                "ff_gelu (43.008, GELU-Kontrolle) gegen lora_r1, lora_r2 (61.440) und qire_L10 auf "
                                "LLaeMmlein-120M; H1 ff-lora TOST, H2 ff<qire_L10, H4 born<gelu, H5 born-lora TOST; "
                                "Positivkontrolle qire_L10 = 2,4111."),
                   config={**konfig, "metrik": METRIK, "torch_threads": threads,
                           "aufloesungspruefung": "lora_r1 trennt sich von basis (zweiseitig)",
                           "fingerabdruck": z.fingerabdruck})

    def sichern():
        rec.records, rec.config["lr"] = list(z.records), dict(z.lr)
        rec.config["tuning"] = dict(z.tuning)               # ALLE Tuningzellen in die Rohdaten (Lehre T-006)
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
            print(f"  {arm:<17} -> lr {LR_UEBERNOMMEN[arm]:g}  (aus T-005 uebernommen)", flush=True)
        if arm not in z.lr and arm in LR_ABGELEITET:
            z.lr[arm] = {"lr": LR_ABGELEITET[arm], "rand": "nicht getunt", "quelle": "von lora_r1 uebernommen"}
            z.speichern()
            print(f"  {arm:<17} -> lr {LR_ABGELEITET[arm]:g}  (von lora_r1 uebernommen, NICHT getunt)", flush=True)
        if arm not in z.lr:
            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = _schl(_arm, seed, f"{lr:.6g}")
                if k not in z.tuning:
                    z.tuning[k] = zelle(_arm, seed, lr, "abstimmung", threads)[METRIK]
                    z.speichern()
                return -z.tuning[k]
            best, vermerk, mittel = sweep_lr("t007", arm, LR_GITTER[arm], TUNING_SEEDS, ERWEITERUNGEN, messe)
            z.lr[arm] = {"lr": best, "rand": vermerk, "mittel": {f"{k:.6g}": -v for k, v in sorted(mittel.items())}}
            z.speichern()
            print(f"  {arm:<17} -> lr {best:g}" + (f"  ACHTUNG Rand {vermerk}" if vermerk != "innen" else "")
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
def poskontrolle(records, feld: str, mut: Dict[str, object]) -> Dict[str, object]:
    w = _reihe(records, "qire_L10", feld)
    m = float(np.mean(list(w.values()))) if w else float("nan")
    ok = bool(w) and (abs(m - POS_SOLL) <= POS_TOL or bool(mut.get("pos_aus")))
    return {"bestanden": ok, "mittel": m, "soll": POS_SOLL, "tol": POS_TOL}


def divergiert(records, feld: str):
    """Arme mit nicht endlichen Werten (Divergenz)."""
    return sorted({r["arm"] for r in records if not math.isfinite(float(r[feld]))})


def _familie(records, feld: str, grenze: float, mut: Dict[str, object]):
    r = {a: _reihe(records, a, feld) for a in ARME}
    h4 = (_vergleich(r["ff_gelu"], r["ff_born"], 0.0, "less") if mut.get("h4_vertauscht")
          else _vergleich(r["ff_born"], r["ff_gelu"], 0.0, "two-sided" if mut.get("h4_zweiseitig") else "less"))
    return {"H1": _vergleich(r["ff_adapter"], r["lora_r1"], grenze),
            "H2": _vergleich(r["ff_adapter"], r["qire_L10"], 0.0, "two-sided" if mut.get("h2_zweiseitig") else "less"),
            "H4": h4,
            "H5": _vergleich(r["ff_born"], r["lora_r1"], grenze)}


_VERGLEICH = {"H1": "ff_adapter - lora_r1", "H2": "ff_adapter - qire_L10", "H4": "ff_born - ff_gelu",
              "H5": "ff_born - lora_r1"}


def _urteil(tag: str, v, p_holm: float, k, pos, div, tost_zuerst: bool = True) -> str:
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
    if div:
        return f"HU_DIVERGENZ -- {tag}: nicht endliche Werte in {', '.join(div)}; Training divergiert, KEINE Aussage."
    if not pos["bestanden"]:
        return (f"HU_POSKONTROLLE -- {tag}: qire_L10 liegt bei {pos['mittel']:.4f}, erwartet "
                f"{pos['soll']:.4f} +-{pos['tol']:g} (T-006). Aufbau nicht reproduziert; KEINE Aussage.")
    if not k["aufloesung"]["bestanden"]:
        return (f"HU_AUFLOESUNG -- {tag}: lora_r1 trennt sich nicht von basis"
                + (" (Zelle fehlt)" if k["aufloesung"].get("fehlt") else "")
                + ". Der Aufbau sieht keine Anpassung; KEIN Nullbefund.")
    z = (f"{_VERGLEICH[tag]}: {v['delta']:+.4f} nats, {v['siege']}/{v['n']} Seeds besser, "
         f"p={v['p']:.4f}, Holm={p_holm:.4f}")
    if tag in ("H1", "H5"):
        z += f", TOST p={v['p_tost']:.4f} (Grenze +-{v['grenze']:.4f})"
        gleich = (f"GLEICHWERTIG -- {tag}: gleichwertig innerhalb +-{v['grenze']:.4f} nats "
                  f"({REL_GRENZE:.0%} des LoRA-Gewinns; erreicht +-{v['erreicht']:.4f}); {z}. Positiver "
                  "Gleichwertigkeitsnachweis"
                  + (" bei GLEICHER Parameterzahl" if tag == "H1" else " (Parameter 43.008 gegen 30.720, siehe Entwurf Abschn. 3)")
                  + ", nicht blosses p>0,05."
                  + (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber kleiner als die Grenze."
                     if p_holm < ALPHA else ""))
        if tag == "H1":
            code, satz = (("FF_BESSER", "der Fastfood-Adapter senkt den Verlust staerker als LoRA r=1 bei gleicher Parameterzahl")
                          if v["delta"] < 0 else
                          ("LORA_BESSER", "LoRA r=1 senkt den Verlust staerker als der Fastfood-Adapter gleicher Parameterzahl"))
        else:
            code, satz = (("BORN_BESSER", "der Born-Zweig (43.008 Parameter, 1,4 x lora_r1) senkt den Verlust staerker als LoRA r=1")
                          if v["delta"] < 0 else
                          ("LORA_BESSER", "LoRA r=1 senkt den Verlust staerker als der Born-Zweig (der 1,4 x so viele Parameter hat)"))
        reihenfolge = ((v["p_tost"] < ALPHA, gleich), (p_holm < ALPHA, f"{code} -- {tag}: {satz}; {z}."))
        for bedingung, text in (reihenfolge if tost_zuerst else reihenfolge[::-1]):
            if bedingung:
                return text
        return (f"UNENTSCHIEDEN -- {tag}: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
                f"die Daten tragen erst ab +-{v['erreicht']:.4f}; {z}.")
    if tag == "H2":
        if p_holm < ALPHA:
            return (f"FF_BESSER_ALS_QIRE -- H2: ff_adapter (30.720) liegt vor qire_L10 (1.920); NICHT "
                    f"parametergleich, kein Effizienzbeweis; {z}.")
        return (f"KEIN_NACHWEIS -- H2: ff_adapter ist nicht belegt besser als qire_L10; {z}."
                + (" Beschreibend liegt qire_L10 im Mittel VORN." if v["delta"] > 0 else ""))
    if p_holm < ALPHA:
        return (f"BORN_BESSER -- H4: bei GLEICHER Struktur und Parameterzahl (43.008) senkt z^2 den Verlust staerker "
                f"als GELU (Wechselwirkung Struktur x Quadrat, vgl. I-045-v2); {z}.")
    return (f"KEIN_NACHWEIS -- H4: ff_born ist nicht belegt besser als ff_gelu; {z}."
            + (" Beschreibend liegt ff_gelu im Mittel VORN." if v["delta"] > 0 else ""))


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
    div = divergiert(records, feld)
    fam = _familie(records, feld, k["grenze"], mut)
    roh_p = [fam[t]["p"] if fam[t] else 1.0 for t in FAMILIE]
    p_holm = dict(zip(FAMILIE, roh_p if mut.get("kein_holm") else holm(roh_p)))
    out: Dict[str, object] = {"kontrollen": k, "poskontrolle": pos}
    say(f"\nRESEARCH ID {LAUF_ID} | {METRIK} (nats/Token, nur Antwort-Tokens), kleiner = besser")
    say("=" * 96)
    say(f"POSITIVKONTROLLE qire_L10 {pos['mittel']:.4f} (soll {POS_SOLL} +-{POS_TOL}): "
        f"{'bestanden' if pos['bestanden'] else 'NICHT BESTANDEN'}")
    say(f"VORREGISTRIERTE FAMILIE (Holm ueber 4), TOST-Grenze H1/H5 {REL_GRENZE:.0%} x LoRA-Gewinn "
        f"{k['gewinn_lora']:+.4f} = +-{k['grenze']:.4f}")
    for t in FAMILIE:
        u = _urteil(t, fam[t], p_holm[t], k, pos, div, bool(mut.get("tost_zuerst", True)))
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
                s = [float(r["s_je_schritt"]) for r in records if r["arm"] == arm and r.get("s_je_schritt")]
                say(f"  {arm:<17} {len(w)} Seeds  Mittel {w.mean():.4f}  sd {w.std(ddof=1) if len(w) > 1 else 0:.4f}"
                    f"  Gewinn {g:+.4f}" + (f" ({g / k['gewinn_lora']:.0%} von LoRA)" if k["gewinn_lora"] else "")
                    + (f"  {g / n_p:.2e}/Param" if n_p else "") + f"  trainierbar {n_p}"
                    + f"  lr {config.get('lr', {}).get(arm, {}).get('lr', '-')}"
                    + (f"  {np.mean(s):.2f} s/Schritt" if s else ""))
        rr = {a: _reihe(records, a, feld) for a in ARME}
        for x, y, hinweis in (("ff_born", "lora_r2", "lora_r2 = 61.440 Parameter, Rate NICHT getunt"),
                              ("ff_born", "ff_adapter", "Born-Zweig gegen linearen Fastfood-Adapter"),
                              ("ff_gelu", "lora_r1", "Kontrolle gegen LoRA")):
            bl = _vergleich(rr[x], rr[y], 0.0)
            if bl:
                say(f"  {x} - {y}: {bl['delta']:+.4f} nats, {bl['siege']}/{bl['n']} Seeds besser, "
                    f"p={bl['p']:.4f} (beschreibend; {hinweis})")
    # Randpruefung: gespeicherter Vermerk UND unabhaengige Nachrechnung aus den Rohdaten
    raender = {s: v["rand"] for s, v in config.get("lr", {}).items()
               if v.get("rand") != "innen" and s in LR_GITTER}
    tun = config.get("tuning", {})
    inkons = {}
    for arm in LR_GITTER:
        best, rand, _ = rand_aus_tuning(tun, arm)
        gesp = config.get("lr", {}).get(arm)
        if gesp and (best is None or abs(math.log10(best / float(gesp["lr"]))) > 1e-6 or rand != gesp.get("rand")):
            inkons[arm] = (best, rand, gesp.get("lr"), gesp.get("rand"))
        if rand in ("oben", "unten"):
            raender.setdefault(arm, rand)
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    if inkons:
        say(f"\n  ACHTUNG, Randvermerk passt nicht zu den Tuning-Rohdaten: {inkons}")
        out["lr_inkonsistent"] = inkons
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest, Teil 1: Theorie (Adapter ohne Modell; Modellteil nur, wenn das Modell vorliegt)
# ---------------------------------------------------------------------------
def _explizit(d_in: int, d_aus: int, ein, aus, skala: float = FF_SKALA) -> torch.Tensor:
    """Referenz: dW als EXPLIZITE Matrix (d_aus x d_in) aus scipy-Hadamard, blockdiagonal."""
    from scipy.linalg import hadamard
    h = torch.tensor(hadamard(BLOCK), dtype=torch.float64) / math.sqrt(BLOCK)
    n_in, n_aus = d_in // BLOCK, d_aus // BLOCK
    hb = lambda n: torch.block_diag(*[h] * n)                                   # noqa: E731
    m = torch.eye(d_in, dtype=torch.float64)
    for d in ein:
        m = hb(n_in) @ torch.diag(d.double()) @ m
    if n_in > n_aus:
        fold = torch.cat([torch.eye(d_aus, dtype=torch.float64)] * (n_in // n_aus), 1) / math.sqrt(n_in // n_aus)
        m = fold @ m
    for d in aus:
        m = hb(n_aus) @ torch.diag(d.double()) @ m
    return skala * m


def _explizit_born(d_in: int, d_aus: int, d1, d2, d3, x, art: str, skala: float = FF_SKALA) -> torch.Tensor:
    """Referenz fuer FFBorn: Zeile fuer Zeile mit expliziten scipy-Hadamard-Matrizen (blockdiagonal), float64."""
    from scipy.linalg import hadamard
    h = torch.tensor(hadamard(BLOCK), dtype=torch.float64) / math.sqrt(BLOCK)
    n_in, n_aus = d_in // BLOCK, d_aus // BLOCK
    hb = lambda n: torch.block_diag(*[h] * n)                                   # noqa: E731
    z = (hb(n_in) @ (d1.double() * x.double()).T).T
    z = z * z if art == "z2" else torch.nn.functional.gelu(z)
    if n_in > n_aus:
        z = z.reshape(z.shape[0], n_in // n_aus, d_aus).sum(1) / math.sqrt(n_in // n_aus)
    z = (hb(n_aus) @ (d2.double() * z).T).T
    return skala * d3.double() * z


def pruefe_theorie(threads: int = 3, mit_modell: bool = True) -> bool:
    torch.set_num_threads(threads)
    zeilen = []
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        # Hadamard: orthonormal, == scipy
        from scipy.linalg import hadamard
        h = torch.kron(walsh(4), walsh(4)).double()
        zeilen.append(("H_256 = H_16 (x) H_16 == scipy.hadamard/16",
                       float((h - torch.tensor(hadamard(BLOCK), dtype=torch.float64) / 16).abs().max()), "<", 1e-6))
        zeilen.append(("H orthonormal (H^T H = I)", float((h.T @ h - torch.eye(BLOCK, dtype=torch.float64)).abs().max()),
                       "<", 1e-10))
        for stufen in (1, 2):
            for name, di, do in _DIMS:
                ff = FFAdapter(di, do, stufen, gen).double()
                x = torch.randn(5, di, dtype=torch.float64)
                zeilen.append((f"s={stufen} {name}: dW = 0 beim Start (letzte Diagonale 0)",
                               float(ff(x).abs().max()), "<", 1e-12))
                for d in ff.ein[:]:
                    d.copy_(torch.randn_like(d))
                for d in ff.aus:
                    d.copy_(torch.randn_like(d))
                ref = _explizit(di, do, [p.data for p in ff.ein], [p.data for p in ff.aus])
                zeilen.append((f"s={stufen} {name}: Vorwaertslauf == explizite Matrix (scipy-Hadamard)",
                               float((ff(x) - x @ ref.T).abs().max()), "<", 1e-9))
                zeilen.append((f"s={stufen} {name}: Parameter {stufen * (di + do)}",
                               float(sum(p.numel() for p in ff.parameters()) - stufen * (di + do)), "<", 0.5))
                # Normerhaltung bei Vorzeichen-Diagonalen (kein Skalenfehler): ||dW x|| = c ||x|| (d_aus == d_in)
                ff2 = FFAdapter(di, do, stufen, gen).double()
                for d in ff2.aus:
                    d.copy_(torch.ones_like(d))
                if di == do:
                    zeilen.append((f"s={stufen} {name}: alle Diagonalen +-1 -> ||dW x|| = c ||x||",
                                   float(abs(ff2(x).norm() / (FF_SKALA * x.norm()) - 1)), "<", 1e-9))
        # Born-Zweig (ff_born / ff_gelu): Start 0, Vorwaertslauf == explizite Referenz, Homogenitaet, Parameter
        for art in ("z2", "gelu"):
            for name, di, do in _DIMS:
                fb = FFBorn(di, do, art, gen).double()
                x = torch.randn(5, di, dtype=torch.float64)
                zeilen.append((f"Born {art} {name}: dW = 0 beim Start (D3 = 0)", float(fb(x).abs().max()), "<", 1e-12))
                for p in (fb.d1, fb.d2, fb.d3):
                    p.copy_(torch.randn_like(p))
                ref = _explizit_born(di, do, fb.d1.data, fb.d2.data, fb.d3.data, x, art)
                zeilen.append((f"Born {art} {name}: Vorwaertslauf == explizite Referenz (scipy-Hadamard)",
                               float((fb(x) - ref).abs().max()), "<", 1e-9))
                zeilen.append((f"Born {art} {name}: Parameter d_in + 2 d_aus = {di + 2 * do}",
                               float(sum(p.numel() for p in fb.parameters()) - (di + 2 * do)), "<", 0.5))
                if art == "z2":
                    zeilen.append((f"Born z2 {name}: quadratisch homogen, dW(2x) = 4 dW(x)",
                                   float((fb(2 * x) - 4 * fb(x)).abs().max()), "<", 1e-9))
                    zeilen.append((f"Born z2 {name}: gerade, dW(-x) = dW(x) (Vorzeichen der Eingabe egal)",
                                   float((fb(-x) - fb(x)).abs().max()), "<", 1e-9))
        for art in ("z2", "gelu"):
            fb = FFBorn(768, 256, art, gen)
            x, w = torch.randn(7, 768), torch.randn(7, 256)
            with torch.enable_grad():
                (fb(x) * w).sum().backward()
            zeilen.append((f"Born {art} Start: Gradient von D3 != 0", float(fb.d3.grad.abs().max()), ">", 0.0))
            zeilen.append((f"Born {art} Start: Gradient von D1, D2 == 0 (dW = 0)",
                           max(float(fb.d1.grad.abs().max()), float(fb.d2.grad.abs().max())), "<", 1e-12))
            with torch.no_grad():
                fb.d3.add_(0.01)
            fb.zero_grad()
            with torch.enable_grad():
                (fb(x) * w).sum().backward()
            zeilen.append((f"Born {art}: nach kleinem Schritt Gradient in D1, D2, D3 != 0",
                           min(float(p.grad.abs().max()) for p in (fb.d1, fb.d2, fb.d3)), ">", 0.0))
        # Gradientenfluss: Start nur in die letzte Diagonale, nach einem Schritt auch in die vorderen
        ff = FFAdapter(768, 256, 2, gen)
        x = torch.randn(7, 768)
    w = torch.randn(7, 256)
    ff.zero_grad()
    with torch.enable_grad():
        (ff(x) * w).sum().backward()
    zeilen.append(("Start: Gradient der letzten Diagonale != 0", float(ff.aus[-1].grad.abs().max()), ">", 0.0))
    zeilen.append(("Start: Gradient der vorderen Diagonalen == 0 (dW = 0 -> Kettenregel)",
                   max(float(p.grad.abs().max()) for p in list(ff.ein) + [ff.aus[0]]), "<", 1e-12))
    with torch.no_grad():
        ff.aus[-1].add_(0.01)
    ff.zero_grad()
    with torch.enable_grad():
        (ff(x) * w).sum().backward()
    zeilen.append(("nach kleinem Schritt: Gradient in ALLEN Diagonalen != 0",
                   min(float(p.grad.abs().max()) for p in list(ff.ein) + list(ff.aus)), ">", 0.0))
    # Parametergleichheit -- das Herzstueck des fairen Vergleichs
    zeilen.append(("Parametergleichheit: ff_adapter == lora_r1 (30.720)",
                   float(abs(ERWARTETE_PARAMETER["ff_adapter"] - ERWARTETE_PARAMETER["lora_r1"])), "<", 0.5))
    zeilen.append(("lora_r2 == 2 x lora_r1 (61.440)",
                   float(abs(ERWARTETE_PARAMETER["lora_r2"] - 2 * ERWARTETE_PARAMETER["lora_r1"])), "<", 0.5))
    zeilen.append(("ff_born == ff_gelu == 12 x ((768+2x768) + (768+2x256)) = 43.008 (Offenlegung: 1,40 x lora_r1)",
                   float(abs(ERWARTETE_PARAMETER["ff_born"] - 43008) + abs(ERWARTETE_PARAMETER["ff_gelu"] - 43008)
                         + abs(ERWARTETE_PARAMETER["ff_born"] - 12 * ((768 + 1536) + (768 + 512)))), "<", 0.5))
    zeilen.append(("lora_r1 < ff_born < lora_r2 (die zwei LoRA-Arme klammern den Born-Zweig ein)",
                   float(not (ERWARTETE_PARAMETER["lora_r1"] < ERWARTETE_PARAMETER["ff_born"]
                              < ERWARTETE_PARAMETER["lora_r2"])), "<", 0.5))
    zeilen.append(("ff_adapter_breit gestrichen (nicht in ARME/TRAINIERT/Familie), H3 entfaellt",
                   float("ff_adapter_breit" in ARME or "ff_adapter_breit" in TRAINIERT or "H3" in FAMILIE), "<", 0.5))
    zeilen.append(("lora_r1 == 12 x ((768+768) + (768+256)) (A: d_in, B: d_aus)",
                   float(abs(ERWARTETE_PARAMETER["lora_r1"] - 12 * ((768 + 768) + (768 + 256)))), "<", 0.5))
    zeilen.append(("Breite 768 == BREITE des T-004-Aufbaus", float(abs(BREITE - 768)), "<", 0.5))
    if mit_modell:
        d_ = daten()
        ids, maske, lab = ca.stapel(d_["validierung"][:3])
        ref = ca.modell("basis")
        ref.eval()
        l_ref = ref(input_ids=ids, attention_mask=maske).logits
        for arm in TRAINIERT:
            m = modell(arm, 1)
            m.eval()
            zeilen.append((f"{arm}: Startwert -> Logits == Basismodell",
                           float((m(input_ids=ids, attention_mask=maske).logits - l_ref).abs().max()), "<", 1e-5))
            zeilen.append((f"{arm}: trainierbar {trainierbar(m)} == {ERWARTETE_PARAMETER[arm]}",
                           abs(trainierbar(m) - ERWARTETE_PARAMETER[arm]), "<", 0.5))
        # Bruecke: T-007-Zelle == T-005-Zelle (3 Schritte) fuer qire_L10 und lora_r1
        for arm in ("qire_L10", "lora_r1"):
            with torch.enable_grad():
                a = zelle(arm, 7, 1e-3, "validierung", threads, 3, True)
                b = ck.zelle(arm, 7, 1e-3, "validierung", threads, 3, True)
            zeilen.append((f"Bruecke: T-007-Zelle {arm} == T-005-Zelle (3 Schritte)",
                           max(abs(a[METRIK] - b[METRIK]), abs(a["train_verlust_ende"] - b["train_verlust_ende"])),
                           "<", 1e-9))
        # ff-Zellen laufen (Tuning-Seed-Bereich, nie Messseed), Werte endlich, Verlust nach 3 Schritten ~ Basis
        basis = ca.auswerten(ref, d_["validierung"][:40])["verlust"]
        for arm in FF_ALLE:
            with torch.enable_grad():
                a = zelle(arm, TUNING_SEEDS[0], 1e-3, "validierung", threads, 3, True)
            zeilen.append((f"{arm}: 3 Schritte laufen, Verlust endlich und |Verlust - Basis| < 0,05",
                           float(abs(a[METRIK] - basis)) if math.isfinite(a[METRIK]) else 9.0, "<", 0.05))
            print(f"  INFO {arm}: {a['s_je_schritt']} s/Schritt (3 Schritte, unsicher), Verlust {a[METRIK]:.4f}, Basis {basis:.4f}")
    lr_alt = json.loads(T005_ROH.read_text(encoding="utf-8"))["config"] if T005_ROH.exists() else None
    if lr_alt is not None:
        abw = sum(lr_alt.get("lr", {}).get(a, {}).get("lr") != v or lr_alt["lr"][a].get("rand") != "innen"
                  for a, v in LR_UEBERNOMMEN.items()) + (lr_alt.get("schritte") != SCHRITTE)
        zeilen.append(("LR-Uebernahme == T-005-Rohdaten (lr, innen, Schritte)", float(abw), "<", 0.5))
    else:
        print(f"  HINWEIS: {T005_ROH_PATTERN} fehlt, LR-Uebernahme nicht gegengeprueft")
    zeilen.append(("Rest getunt: jeder trainierte Arm hat Rate (uebernommen, abgeleitet oder Gitter)",
                   float(sum(a not in LR_GITTER and a not in LR_UEBERNOMMEN and a not in LR_ABGELEITET
                             for a in TRAINIERT)), "<", 0.5))
    zeilen.append(("Wechselwirkungs-Kontrolle: ff_born und ff_gelu getunt mit GLEICHEM Gitter und gleichen Tuning-Seeds",
                   float(LR_GITTER["ff_born"] != LR_GITTER["ff_gelu"] or "ff_born" not in LR_GITTER), "<", 0.5))
    zeilen.append(("Lernrategitter reicht bis 1e-4 nach unten (Lehre T-006)",
                   float(min(min(g) for g in LR_GITTER.values()) > 1.0001e-4), "<", 0.5))
    alt = (set(ca.EVAL_SEEDS) | set(ca.TUNING_SEEDS) | set(ck.EVAL_SEEDS) | set(ck.TUNING_SEEDS)
           | set(cs.EVAL_SEEDS) | set(cs.TUNING_SEEDS))
    zeilen.append(("Seeds frisch: disjunkt zu T-004/T-005/T-006, Mess/Tuning disjunkt (Ueberschneidungen)",
                   float(len((set(EVAL_SEEDS) | set(TUNING_SEEDS)) & alt) + len(set(EVAL_SEEDS) & set(TUNING_SEEDS))
                         + (len(EVAL_SEEDS) != 8)), "<", 0.5))
    zeilen.append(("Parameter qire_L10 / lora_r1 == T-005",
                   float(sum(ERWARTETE_PARAMETER[a] != ck.ERWARTETE_PARAMETER[a] for a in ("qire_L10", "lora_r1"))),
                   "<", 0.5))
    return ck._zeilen_drucken("THEORIEPRUEFUNG T-007", zeilen)


def pruefe_mini_modell() -> bool:
    """Logik am eingebauten Mini-Llama (Zufallsgewichte, KEIN Download): gleiche Form wie LLaeMmlein
    (Breite 768, v_proj 768->256), 2 Lagen. Prueft Einbau, Parameterzahl, Start == Basis, Lernfaehigkeit."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:                                     # noqa: BLE001
        print(f"  Mini-Modell uebersprungen ({type(exc).__name__}); auf Maltes PC laeuft transformers")
        return True
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=768, intermediate_size=256, num_hidden_layers=2, num_attention_heads=12,
                      num_key_value_heads=4, vocab_size=300, max_position_embeddings=64)
    ids = torch.randint(0, 300, (4, 24))
    zeilen = []
    for arm, s in {**FF_ARME, **BORN_ARME}.items():
        m = LlamaForCausalLM(cfg).float()
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        l0 = m(input_ids=ids).logits.detach()
        ff_einbauen(m, s, 5)
        with torch.no_grad():
            dl = float((m(input_ids=ids).logits - l0).abs().max())
        zeilen.append((f"Mini {arm}: Start-Logits == Basis (dW = 0)", dl, "<", 1e-6))
        zeilen.append((f"Mini {arm}: trainierbar == 2 Lagen x {ERWARTETE_PARAMETER[arm] // 12}",
                       float(abs(trainierbar(m) - 2 * ERWARTETE_PARAMETER[arm] // 12)), "<", 0.5))
        zeilen.append((f"Mini {arm}: q_proj und v_proj ersetzt, Rest unveraendert",
                       float(sum(not isinstance(getattr(l.self_attn, n), FFLinear) for l in m.model.layers for n in ZIELE)
                             + sum(isinstance(getattr(l.self_attn, n), FFLinear) for l in m.model.layers
                                   for n in ("k_proj", "o_proj"))), "<", 0.5))
        m.train()
        start = [p.detach().clone() for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        ziel = torch.randint(0, 300, (4, 24))
        v = []
        for _ in range(30):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(input_ids=ids).logits.reshape(-1, 300), ziel.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], CLIP)
            opt.step()
            v.append(float(loss.detach()))
        zeilen.append((f"Mini {arm}: Verlust faellt in 30 Schritten (Lernfaehigkeit, fester Batch)", v[0] - v[-1], ">", 0.01))
        zeilen.append((f"Mini {arm}: alle Diagonalen haben nach dem Training Gradient/Bewegung (nicht eingefroren)",
                       float(sum(float((p.detach() - q).abs().max()) == 0.0
                                 for p, q in zip([p for p in m.parameters() if p.requires_grad], start))), "<", 0.5))
        zeilen.append((f"Mini {arm}: Basisgewichte eingefroren (kein Gradient, requires_grad aus)",
                       float(sum(p.requires_grad for n, p in m.named_parameters() if ".basis." in n or "embed" in n)), "<", 0.5))
    return ck._zeilen_drucken("MINI-MODELL T-007 (Zufallsgewichte, gleiche Form)", zeilen)


def pruefe_rand() -> bool:
    """Randpruefung aus Rohdaten und Gitterverlaengerung (beide Richtungen), unabhaengig von sweep_lr."""
    g = LR_START
    faelle = [("Optimum im Startgitter", 1e-3, "innen", 1e-3), ("Verlaengerung unten", 3e-5, "innen", 3.33333e-5),
              ("Verlaengerung oben", 3e-2, "innen", 3e-2),
              ("ausserhalb oben", 300.0, "oben", g[-1] * 27), ("ausserhalb unten", 1e-9, "unten", g[0] / 27),
              ("Optimum am unteren Startrand 1e-4 -> wird verlaengert", 1e-4, "innen", 1e-4)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG UND RANDPRUEFUNG AUS ROHDATEN -- Startgitter", g)
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        tuning = {}

        def messe(lr, s, _o=opt):
            v = 3.0 + math.log10(lr / _o) ** 2
            tuning[_schl("ff_adapter", 70400 + s, f"{lr:.6g}")] = v
            return -v
        best, rand, _ = sweep_lr("t007", "ff_adapter", g, (1, 2), ERWEITERUNGEN, messe)
        tuning = {k.replace("|70401|", "|70401|"): v for k, v in tuning.items()}
        b2, r2, _m = rand_aus_tuning({k: v for k, v in tuning.items()}, "ff_adapter", (1 + 70400, 2 + 70400))
        good = (abs(math.log10(best / rate_soll)) < 1e-3 and rand == rand_soll
                and b2 is not None and abs(math.log10(b2 / best)) < 1e-6 and r2 == rand)
        ok &= good
        print(f"  {name:<52} -> lr {best:<10.4g} Rand {rand:<6} Rohdaten: {b2:.4g}/{r2:<6} {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return ok


# ---------------------------------------------------------------------------
# Pflichtvortest, Teil 2: jeden Ausgang einzeln erzwingen (synthetische Werte)
# ---------------------------------------------------------------------------
NIVEAU, NIVEAU_ALLE = 2.5616, 3.50
G_Q, G_LORA = 0.1505, 0.2730                          # T-006: qire_L10 2,4111, lora_r1 2,2886
_V = {"basis": 0.0, "qire_L10": -G_Q, "lora_r1": -G_LORA, "lora_r2": -0.30, "ff_adapter": -0.20,
      "ff_born": -0.20, "ff_gelu": -0.15}
VERSATZ = {a: _V[a] for a in ARME}


def _kunst(seed: int, streuung: float = 0.003, versatz=None, weg: Sequence[str] = (),
           weg_seed: Optional[Dict[str, int]] = None, s_arm: Optional[Dict[str, float]] = None,
           werte: Optional[Dict[str, Sequence[float]]] = None) -> Recorder:
    def rng(*t):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, t)).encode())])
    vs = {**VERSATZ, **(versatz or {})}
    rec = Recorder("T-007-vortest", "synthetisch", config={"lr": {}})
    for i, s in enumerate(EVAL_SEEDS):
        grund = rng("grund", s).normal(0, 0.0015)
        for arm, v in vs.items():
            if arm in weg or (weg_seed or {}).get(arm) == s:
                continue
            sd = (s_arm or {}).get(arm, streuung)
            eig = 0.0 if arm == "basis" else grund + rng(arm, s).normal(0, sd)
            alle = 0.0 if arm == "basis" else eig + rng("alle", arm, s).normal(0, sd)
            wert = NIVEAU + v + eig if not (werte and arm in werte) else float(werte[arm][i])
            rec.add(arm=arm, seed=s, lr=0.001, menge="validierung", val_verlust=wert,
                    val_verlust_alle=NIVEAU_ALLE - v + alle, trainierbar=ERWARTETE_PARAMETER[arm],
                    s_je_schritt=1.0)
    return rec


# Deterministisches Holm-Szenario: H1 zweiseitig raw p = 10/256 x 2 = 0,039 (< 0,05, aber Holm x3 > 0,05);
# H2, H3 weit weg von signifikant. Faengt die Mutation "kein_holm".
_E = np.array([0.001, -0.001, 0.0005, -0.0005, 0.0015, -0.0015, 0.0, 0.0])
_DH1 = np.array([0.001, 0.002, -0.003, 0.10, 0.11, 0.12, 0.13, 0.14])
_LORA_W = 2.2886 + _E
_WIGGLE = np.array([0.004, -0.003, 0.002, -0.001, 0.001, -0.002, 0.003, -0.004])
# ff_born = ff_adapter + Wackeln (H5 ~ H1: raw p ~ 0,039, gekippt), ff_gelu = ff_born + Wackeln' (H4 nicht signifikant)
HOLM_WERTE = {"lora_r1": _LORA_W, "qire_L10": 2.4111 + _E[::-1], "ff_adapter": _LORA_W + _DH1,
              "ff_born": _LORA_W + _DH1 + _WIGGLE, "ff_gelu": _LORA_W + _DH1 + _WIGGLE[::-1] * 0.5,
              "lora_r2": _LORA_W - 0.03}

FAELLE = [
    # (Name, Test, kwargs, Code, Zweigtext, Ziehungen, Anteil)
    ("H1-Q  ff besser als LoRA (-0,40)", "H1", dict(versatz={"ff_adapter": -0.40}), "FF_BESSER", "staerker als LoRA", 5, 1.0),
    ("H1-L  LoRA besser (ff -0,20)", "H1", {}, "LORA_BESSER", "staerker als der Fastfood", 5, 1.0),
    ("H1-G  gleich, kleines Rauschen", "H1", dict(versatz={"ff_adapter": -G_LORA}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("H1-Gs +0,005 bei winzigem Rauschen: signifikant UND gleichwertig", "H1",
     dict(streuung=0.0005, versatz={"ff_adapter": -G_LORA + 0.005}, s_arm={"qire_L10": 0.003}), "GLEICHWERTIG",
     "zugleich signifikant", 5, 1.0),
    ("H1-U  gleich, grosses Rauschen", "H1",
     dict(versatz={"ff_adapter": -G_LORA}, s_arm={"ff_adapter": 0.08, "lora_r1": 0.08}), "UNENTSCHIEDEN",
     "weder Unterschied", 40, 0.8),
    ("H1-H  Holm kippt raw p 0,039 -> UNENTSCHIEDEN", "H1", dict(werte=HOLM_WERTE), "UNENTSCHIEDEN", "weder Unterschied", 1, 1.0),
    ("H1-X  ff_adapter fehlt", "H1", dict(weg=["ff_adapter"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H1-Xs ein Seed fehlt in ff_adapter", "H1", dict(weg_seed={"ff_adapter": EVAL_SEEDS[2]}), "HU_UNVOLLSTAENDIG",
     "Seeds ungleich", 5, 1.0),
    ("H1-D  ff_adapter NaN (Divergenz)", "H1", dict(versatz={"ff_adapter": float("nan")}), "HU_DIVERGENZ",
     "nicht endliche Werte", 5, 1.0),
    ("H1-P  Positivkontrolle faellt (qire_L10 bei 2,45)", "H1", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE",
     "nicht reproduziert", 5, 1.0),
    ("H1-A  Aufloesung faellt (lora = basis)", "H1", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG",
     "KEIN Nullbefund", 40, 0.8),
    ("H1-An lora-Zelle fehlt (H1 braucht lora)", "H1", dict(weg=["lora_r1"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H2-An lora-Zelle fehlt (Aufloesung nicht pruefbar)", "H2", dict(weg=["lora_r1"]), "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
    ("H1-Ar lora SCHLECHTER als basis: Trennung zaehlt, ff vorn", "H1", dict(versatz={"lora_r1": 0.10}),
     "FF_BESSER", "staerker als LoRA", 5, 1.0),
    ("H2-J  ff besser als qire_L10 (-0,20 gegen -0,15)", "H2", {}, "FF_BESSER_ALS_QIRE", "NICHT parametergleich", 5, 1.0),
    ("H2-N  ff schlechter als qire_L10", "H2", dict(versatz={"ff_adapter": -0.10}), "KEIN_NACHWEIS", "qire_L10 im Mittel VORN", 5, 1.0),
    ("H2-G  ff = qire_L10, Rauschen", "H2", dict(versatz={"ff_adapter": -G_Q}), "KEIN_NACHWEIS", "nicht belegt besser", 40, 0.8),
    ("H2-X  qire_L10 fehlt", "H2", dict(weg=["qire_L10"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H2-P  Positivkontrolle faellt -> auch H2", "H2", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE",
     "nicht reproduziert", 5, 1.0),
    ("H2-A  Aufloesung faellt", "H2", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.8),
    ("H4-J  born besser als gelu", "H4", {}, "BORN_BESSER", "z^2 den Verlust staerker", 5, 1.0),
    ("H4-N  born = gelu, Rauschen", "H4", dict(versatz={"ff_born": -0.15}), "KEIN_NACHWEIS", "nicht belegt besser", 40, 0.8),
    ("H4-S  born schlechter als gelu", "H4", dict(versatz={"ff_born": -0.10}), "KEIN_NACHWEIS", "ff_gelu im Mittel VORN", 5, 1.0),
    ("H4-X  ff_born fehlt", "H4", dict(weg=["ff_born"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H4-Xg ff_gelu fehlt", "H4", dict(weg=["ff_gelu"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H4-Xs ein Seed fehlt in ff_gelu", "H4", dict(weg_seed={"ff_gelu": EVAL_SEEDS[3]}), "HU_UNVOLLSTAENDIG",
     "Seeds ungleich", 5, 1.0),
    ("H4-A  Aufloesung faellt", "H4", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.8),
    ("H4-P  Positivkontrolle faellt", "H4", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE", "nicht reproduziert", 5, 1.0),
    ("H4-D  gelu NaN", "H4", dict(versatz={"ff_gelu": float("nan")}), "HU_DIVERGENZ", "nicht endliche Werte", 5, 1.0),
    ("H5-Q  born besser als LoRA (-0,40)", "H5", dict(versatz={"ff_born": -0.40}), "BORN_BESSER", "staerker als LoRA", 5, 1.0),
    ("H5-L  LoRA besser (born -0,20)", "H5", {}, "LORA_BESSER", "staerker als der Born-Zweig", 5, 1.0),
    ("H5-G  gleich, kleines Rauschen", "H5", dict(versatz={"ff_born": -G_LORA}, s_arm={"qire_L10": 0.001}), "GLEICHWERTIG",
     "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("H5-U  gleich, grosses Rauschen", "H5",
     dict(versatz={"ff_born": -G_LORA}, s_arm={"ff_born": 0.08, "lora_r1": 0.08}), "UNENTSCHIEDEN", "weder Unterschied", 40, 0.8),
    ("H5-H  Holm kippt raw p 0,039 -> UNENTSCHIEDEN", "H5", dict(werte=HOLM_WERTE), "UNENTSCHIEDEN", "weder Unterschied", 1, 1.0),
    ("H5-X  ff_born fehlt", "H5", dict(weg=["ff_born"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("H5-P  Positivkontrolle faellt", "H5", dict(versatz={"qire_L10": -0.11}), "HU_POSKONTROLLE", "nicht reproduziert", 5, 1.0),
    ("H5-A  Aufloesung faellt", "H5", dict(versatz={"lora_r1": 0.0}), "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.8),
    ("H5-D  born NaN", "H5", dict(versatz={"ff_born": float("nan")}), "HU_DIVERGENZ", "nicht endliche Werte", 5, 1.0),
    ("H5-r2 lora_r2 fehlt: H5 unberuehrt (lora_r2 nur beschreibend)", "H5", dict(weg=["lora_r2"]), "LORA_BESSER",
     "staerker als der Born-Zweig", 5, 1.0),
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
    p_holm_szenario = analyse(_kunst(seed=1, werte=HOLM_WERTE), still=True)
    pruef = [("val_verlust_alle - val_verlust NICHT konstant", min(float(np.std(r(a, "val_verlust_alle") - r(a, METRIK), ddof=1))
                                                               for a in TRAINIERT) > 1e-4),
             ("basis deterministisch, trainierte Arme streuen",
              float(np.std(r("basis", METRIK))) == 0 and float(np.std(r("ff_adapter", METRIK))) > 1e-4),
             ("Positivkontrolle im Standardfall erfuellt", poskontrolle(rec.records, METRIK, {})["bestanden"]),
             ("Holm-Szenario (4 Tests): Positivkontrolle und Aufloesung bestanden, H1 raw p in (0,0125; 0,05)",
              p_holm_szenario["poskontrolle"]["bestanden"] and p_holm_szenario["kontrollen"]["aufloesung"]["bestanden"]
              and 0.0125 < p_holm_szenario["H1"]["p"] < 0.05)]
    for name, good in pruef:
        print(f"  {'OK    ' if good else 'FEHLER'} {name}")
    return all(g for _, g in pruef)


MUTATIONEN = [("Metrikfeld val_verlust_alle statt val_verlust", {"feld": "val_verlust_alle"}),
              ("Signifikanz VOR TOST (H1)", {"tost_zuerst": False}),
              ("Aufloesung verlangt Richtung statt Trennung", {"richtung": True}),
              ("TOST-Grenze absolut 0,2 statt relativ", {"grenze_absolut": True}),
              ("H2 zweiseitig statt einseitig", {"h2_zweiseitig": True}),
              ("H4 Arme vertauscht", {"h4_vertauscht": True}),
              ("H4 zweiseitig statt einseitig", {"h4_zweiseitig": True}),
              ("Holm abgeschaltet", {"kein_holm": True}),
              ("Positivkontrolle abgeschaltet", {"pos_aus": True})]


def vortest(mit_modell: bool = False) -> bool:
    print("=" * 96)
    print("VORTEST T-007 -- jeder Ausgang einzeln, ueber mehrere Ziehungen, Marke UND Zweigtext")
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
# Zeitmessung (Modell noetig) und Prognose aus T-006-Rohdaten
# ---------------------------------------------------------------------------
# T-006-Ist (Mittel je Arm, 6 Threads, Maltes PC): qire_L10 1,64 s/Schritt, 10,4 min/Zelle; lora_r1 0,84 s,
# 5,9 min/Zelle. Zelle = 300 x s + ~100 s (Laden + Auswertung 1.000 Beispiele). ff-Zeiten sind ANNAHMEN
# (Hadamard als 2 kleine Matmuls je Block: ca. +10-20 % gegen LoRA; breit ca. +40 %); `zeit` misst sie.
S_ANNAHME = {"qire_L10": 1.64, "lora_r1": 0.84, "lora_r2": 0.95, "ff_adapter": 1.0, "ff_born": 1.1, "ff_gelu": 1.1}
ZELLE_FIX_S = 100.0


def prognose() -> None:
    zelle_min = lambda arm: (SCHRITTE * S_ANNAHME[arm] + ZELLE_FIX_S) / 60                  # noqa: E731
    print("PROGNOSE (Extrapolation aus T-006, ff-Zeiten Annahme): Zelle = 300 x s/Schritt + 100 s")
    for a in TRAINIERT:
        print(f"  {a:<17} {S_ANNAHME[a]:.2f} s/Schritt  {zelle_min(a):5.1f} min/Zelle")
    mess = 8 * sum(zelle_min(a) for a in TRAINIERT) / 60 + 1.6 / 60
    n_roh = len(LR_START) * len(TUNING_SEEDS)
    tune = sum(n_roh * (zelle_min(a) - 0.5) for a in LR_GITTER) / 60
    tune_verl = sum(len(TUNING_SEEDS) * (zelle_min(a) - 0.5) for a in LR_GITTER) / 60
    kern = mess + tune
    print(f"  Messung {mess:.1f} h, Tuning {tune:.1f} h (+ je Verlaengerung {tune_verl:.1f} h fuer {len(LR_GITTER)} Arme), "
          f"Kern {kern:.1f} h; mit einer Verlaengerung je Arm {kern + tune_verl:.1f} h")
    breit = (8 * (SCHRITTE * 1.2 + ZELLE_FIX_S) + n_roh * (SCHRITTE * 1.2 + ZELLE_FIX_S - 30)) / 3600
    print(f"  (gestrichener Arm ff_adapter_breit haette +{breit:.1f} h gekostet: Kern {kern + breit:.1f} h > 10 h)")
    print("  Regel vorab: Ziel Kern <= 10 h; misst `zeit` > 10 h, entfaellt lora_r2 (letzter Arm, nur beschreibend).")


def zeit(threads: int = 6, schritte: int = 4) -> None:
    prognose()
    print("=" * 96)
    print(f"ZEITMESSUNG T-007 -- {threads} Threads, {schritte} Schritte je Arm (Seed 1, kein Mess-/Tuning-Seed)")
    daten()
    for arm in TRAINIERT:
        r = zelle(arm, 1, 1e-3, "validierung", threads, schritte, True)
        print(f"  {arm:<17} {r['s_je_schritt']:>6.3f} s/Schritt   trainierbar {r['trainierbar']}", flush=True)


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        ok = pruefe_theorie(mit_modell=cs.modell_vorhanden())
        raise SystemExit(0 if (ok & pruefe_mini_modell() & pruefe_rand() & vortest()) else 1)
    elif befehl == "vortest_statistik":            # ohne Modell
        raise SystemExit(0 if (pruefe_theorie(mit_modell=False) & pruefe_mini_modell() & pruefe_rand() & vortest())
                         else 1)
    elif befehl == "prognose":
        prognose()
    elif befehl == "zeit":
        zeit(*(int(a) for a in sys.argv[2:4]))
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else LAUF_ID)
    elif befehl == "lauf":
        if not (pruefe_theorie(mit_modell=True) and pruefe_mini_modell() and pruefe_rand() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print(f"  Nachholen: python -m quantum_sim.chatbot_fastfood auswertung {LAUF_ID}")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | vortest_statistik | prognose | zeit [threads] [schritte] | lauf | auswertung [ID|Pfad]")
