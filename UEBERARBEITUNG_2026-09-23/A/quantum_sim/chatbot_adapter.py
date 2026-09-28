from __future__ import annotations

"""RESEARCH ID: T-004 -- Der Chatbot: QIRE-Drehung als Adapter auf LLaeMmlein-120M.

Spezifikation: ENTWURF_T004_CHATBOT_ADAPTER.md (28.09.2026, vor jeder Zahl),
massgeblich Abschnitt 5b ("kein Kaggle, CPU mit LLaeMmlein-120M").

BASISMODELL  LSX-UniWue/LLaMmlein_120M (Llama, 12 Lagen, Breite 768 = 3 x 256,
  12 Q-Koepfe, 4 KV-Koepfe a 64 -> q_proj 768x768, v_proj 256x768).
  LIZENZ: LLaeMmlein RESEARCH-ONLY RAIL-M. Forschung (Jugend forscht) ist
  erlaubt; ein oeffentlich gehosteter Chatbot waere "Distribution" -- vor jeder
  Veroeffentlichung (auch der Adapter-Gewichte) die Lizenz pruefen.
DATEN  FreedomIntelligence/alpaca-gpt4-deutsch (Apache-2.0, 54 MB, 49.969 Zeilen).

DER ADAPTER  W (d_out x 768) bleibt eingefroren, W' = W . R, R orthogonal.
  Umgesetzt als y = W (R x): R wirkt auf die EINGABE der Schicht. Die Winkel
  werden als Tensor-Umformung wie `zeitreihen.OrthoButterfly` angewandt,
  EINMAL je Vorwaertslauf fuer alle 24 Matrizen auf I_256 (`Drehbank`, s.
  ANWENDUNG), dann 256x256-Blockprodukt -- nie eine dichte 768x768-Matrix,
  nie W neu berechnet. 768 ist keine Zweierpotenz: R = Blockdiagonale aus 3 GLEICHEN
  8-Qubit-Drehungen (Winkel ueber die 3 Bloecke geteilt) -- ehrlich eine
  Block-Variante wie Q-020 (a).
  Zielschichten q_proj und v_proj aller 12 Lagen = 24 Matrizen.
    Arm        R                                              je Matrix   gesamt
    basis      kein Training, nur Auswertung                         0        0
    qire_kron  1 Lage, R1 x ... x R8 (reelle 2x2-Drehungen)          8      192
    qire_5     5 Lagen: Ry je Qubit, dann CNOT-Ring (Permutation);
               am Ende die feste Rueckpermutation P^-5 (s. u.)       40      960
    lora_r1    peft LoraConfig r=1, alpha=2, auf dieselben Module 1.536/1.024 30.720
  Gezaehlt wird im Lauf (`trainierbar` je Zelle), nicht nur hier behauptet.
  RUECKPERMUTATION (Widerspruch im Entwurf, 28.09.): "Winkel 0 -> R = I" gilt
  fuer qire_5 MIT CNOT-Ring nicht, denn bei Winkel 0 bleibt P^5 != I stehen.
  Deshalb R = P^-5 . prod_l (P . K_l): bei Winkel 0 exakt I, und sobald ein
  Winkel != 0 ist, wirkt der Ring ZWISCHEN den Lagen (Verschraenkung).
  Ohne CNOT waeren 5 geteilte Kronecker-Lagen dasselbe wie EINE (Winkel
  addieren sich) -- P3 misst also allein den Beitrag des CNOT-Rings
  (`pruefe_theorie` rechnet beides nach).

AUFGABE UND MESSGROESSE  Chatformat "### Instruktion:\\n{frage}\\n\\n### Antwort:\\n
  {antwort}</s>", Verlust NUR auf Antwort-Tokens (+ </s>), max. 128 Tokens,
  Batch 4. Feste Aufteilung (Seed 20260928, alle Seeds gleich): Training 5.000,
  Validierung 1.000, Abstimmmenge 500 (NUR fuer die Lernratensuche -- sonst
  waehlte die Suche auf der Messmenge), 50 Vorzeige-Fragen. Nur einteilige
  Gespraeche; Frage <= 96 Tokens, damit >= 31 Antwort-Tokens bleiben.
  Messgroesse `val_verlust`: Kreuzentropie (nats/Token) ueber ALLE Antwort-
  Tokens der Validierung (tokengewichtet). Gleiches Budget SCHRITTE fuer alle
  Arme. Adam, kein Weight Decay, Gradienten-Clipping 1,0.

VORREGISTRIERUNG (vor jeder T-004-Zahl; Aenderungen erst ins Tagebuch)
  8 Seeds (40001-40008), Tuning-Seeds 40401, 40402. Lernrate je Arm mit VOLLEM
  Budget auf der Abstimmmenge, Startgitter je Arm, Verlaengerung in BEIDE
  Richtungen bis 3-mal (`kernaussage.sweep_lr`, Regel 1).
  Familie, Holm ueber GENAU 3:
    PRIMAER  qire_kron - basis, Wilcoxon EINSEITIG (senkt qire_kron den
             Verlust?)             SENKT_VERLUST / KEIN_NACHWEIS
    P2       qire_kron - lora_r1, zweiseitig + TOST
             QIRE_BESSER / GLEICHWERTIG / LORA_BESSER / UNENTSCHIEDEN
    P3       qire_5 - qire_kron, zweiseitig + TOST
             VERSCHRAENKUNG_TRAEGT / GLEICHWERTIG / VERSCHRAENKUNG_SCHADET /
             UNENTSCHIEDEN
  TOST-GRENZE (relativ, vorab): +-REL_GRENZE x |Gewinn von lora_r1 gegen basis|
    = +-20 % der Anpassung, die die Aufloesungspruefung ueberhaupt misst.
    Begruendung: Die absolute Groesse eines Verlustgewinns in nats ist vor dem
    Lauf unbekannt (ein Vorlauf ist wegen des parallelen Laufs nicht erlaubt);
    eine feste Zahl waere geraten. "Gleich weit wie LoRA" heisst hier: der
    Abstand ist kleiner als ein Fuenftel dessen, was LoRA ueberhaupt bewegt.
    Die ERREICHTE Grenze wird immer mitberichtet.
  Rangfolge: HU_UNVOLLSTAENDIG -> HU_AUFLOESUNG -> (TOST -> Holm -> Richtung
    -> UNENTSCHIEDEN) bzw. fuer PRIMAER (Holm einseitig -> KEIN_NACHWEIS).
  Aufloesung (urteilstragend fuer alle drei): lora_r1 trennt sich von basis,
    Wilcoxon zweiseitig p < 0,05 (Trennung, nicht Richtung).
  Erreichbarkeit (Regel 2): kleinstes p bei 8 Seeds einseitig 0,0039,
    zweiseitig 0,0078; nach Holm x3 hoechstens 0,023 -- erreichbar.
    basis ist deterministisch (keine Seeds); sein Wert steht in jeder Seed-
    Zeile gleich, PRIMAER ist damit ein Ein-Stichproben-Test auf qire_kron.

Run (aus dem Ordner A):
    python -m quantum_sim.chatbot_adapter vortest          # Pflicht
    python -m quantum_sim.chatbot_adapter zeit [threads] [schritte]
    python -m quantum_sim.chatbot_adapter lauf             # Malte entscheidet
    python -m quantum_sim.chatbot_adapter auswertung [ID|Pfad]
    python -m quantum_sim.chatbot_adapter chat [arm] ["Frage"]
"""

import copy
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.equivalence import holm, tost_paired
from qire_ai.result_store import RESULTS_DIR, Recorder

from .butterfly_fast import komponiere
from .butterfly_layer import cnot_permutation
from .kernaussage import sweep_lr

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt (Regel 7)
# ---------------------------------------------------------------------------
LAUF_ID = "T-004"
MODELL = "LSX-UniWue/LLaMmlein_120M"
DATENSATZ = "FreedomIntelligence/alpaca-gpt4-deutsch"
BREITE, QUBITS = 768, 8
BLOCK = 2 ** QUBITS                                    # 256, 3 Bloecke
ZIELE = ("q_proj", "v_proj")
ARME = ("basis", "qire_kron", "qire_5", "lora_r1")
TRAINIERT = ("qire_kron", "lora_r1", "qire_5")         # Reihenfolge = Wichtigkeit
SCHALTUNG = {"qire_kron": (1, False), "qire_5": (5, True)}   # (Lagen, CNOT-Ring)
LORA_R, LORA_ALPHA = 1, 2
ERWARTETE_PARAMETER = {"basis": 0, "qire_kron": 24 * 8, "qire_5": 24 * 40,
                       "lora_r1": 12 * ((768 + 768) + (768 + 256))}

N_TRAIN, N_VAL, N_ABSTIMM, N_VORZEIGE = 5000, 1000, 500, 50
MAX_TOKENS, MAX_FRAGE, BATCH, BATCH_AUSWERTUNG = 128, 96, 4, 8
SEED_AUFTEILUNG = 20260928
SCHRITTE = 200        # vorlaeufig aus `zeit` (28.09., 3 Threads); mit 6 Threads bestaetigen, Tagebuch
CLIP = 1.0
# Anwendung von R: "kette" = O(d log d)-Umformung direkt auf den Aktivierungen
# (je Matrix, Tokens*3 Zeilen). "bank" = dieselbe Kette EINMAL je Vorwaertslauf
# fuer alle 24 Matrizen zugleich auf I_256 (liefert R^T), dann je Schicht eine
# 256x256-Blockmultiplikation; keine 768x768-Matrix, keine Neuberechnung von W.
# Mathematisch identisch (pruefe_theorie). "bank" gewaehlt, weil `zeit` die
# Kette auf der CPU als Engpass misst (28.09.: qire_5 5,8 s/Schritt statt 1,8
# bei LoRA -- elementweise Operationen, nicht Rechenleistung).
ANWENDUNG = "bank"

EVAL_SEEDS = tuple(range(40001, 40009))
TUNING_SEEDS = (40401, 40402)
LR_GITTER = {"qire_kron": (0.001, 0.003, 0.01, 0.03), "qire_5": (0.001, 0.003, 0.01, 0.03),
             "lora_r1": (1e-4, 3e-4, 1e-3, 3e-3)}
ERWEITERUNGEN = 3
REL_GRENZE = 0.20
FAMILIE = ("PRIMAER", "P2", "P3")
METRIK = "val_verlust"
ADAPTER_DIR = RESULTS_DIR.parent / "t004_adapter"
FRAGE_FORMAT = "### Instruktion:\n{frage}\n\n### Antwort:\n"


# ---------------------------------------------------------------------------
# Die QIRE-Drehung
# ---------------------------------------------------------------------------
def _ring(lagen: int, cnot: bool):
    """CNOT-Ring als Permutation (Konvention butterfly_layer) und P^-L."""
    if not cnot:
        return None, None
    p = komponiere([cnot_permutation(QUBITS, q, (q + 1) % QUBITS) for q in range(QUBITS)])
    return p, torch.argsort(komponiere([p] * lagen))


def _kette(h: torch.Tensor, theta: torch.Tensor, perm=None, rueck=None) -> torch.Tensor:
    """h (K, Zeilen, 256), theta (K, L, 8): zeilenweise h -> R_k h.
    Je Lage reelle 2x2-Drehung [[c,-s],[s,c]] auf jedem Qubit (Qubit 0 = hoechstes
    Bit, wie OrthoButterfly), dann optional der CNOT-Ring; am Ende P^-L, damit
    Winkel 0 exakt R = I ergibt. Auf I_256 angewandt liefert das R_k^T."""
    k, n = h.shape[0], QUBITS
    c, s = torch.cos(theta).to(h.dtype), torch.sin(theta).to(h.dtype)
    for lage in range(theta.shape[1]):
        for q in range(n):
            v = h.reshape(k, -1, 2 ** q, 2, 2 ** (n - 1 - q))
            a, b = v[:, :, :, 0, :], v[:, :, :, 1, :]
            cq, sq = c[:, lage, q].view(k, 1, 1, 1), s[:, lage, q].view(k, 1, 1, 1)
            h = torch.stack([cq * a - sq * b, sq * a + cq * b], dim=3).reshape(k, -1, BLOCK)
        if perm is not None:
            h = h[..., perm]
    return h[..., rueck] if perm is not None else h


class QireDrehung(nn.Module):
    """EINE Matrix R auf R^768 = Blockdiagonale aus 3 gleichen 8-Qubit-Drehungen.
    forward(x) = R x je Zeile, O(d log d) je Token. Referenz fuer die Pruefung
    und Anwendung "kette"."""

    def __init__(self, lagen: int, cnot: bool):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(lagen, QUBITS))
        self.perm, self.rueck = _ring(lagen, cnot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _kette(x.reshape(1, -1, BLOCK), self.theta[None], self.perm, self.rueck).reshape(x.shape)


class Drehbank(nn.Module):
    """Alle K = 24 Matrizen eines Arms: theta (K, L, 8). `aktualisieren` baut
    EINMAL je Vorwaertslauf alle R_k^T (K x 256 x 256) in einem Kettenaufruf."""

    def __init__(self, anzahl: int, lagen: int, cnot: bool):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(anzahl, lagen, QUBITS))
        self.perm, self.rueck = _ring(lagen, cnot)
        self.rt: Optional[torch.Tensor] = None

    def aktualisieren(self, *_):
        k = self.theta.shape[0]
        eye = torch.eye(BLOCK, dtype=self.theta.dtype).expand(k, BLOCK, BLOCK)
        self.rt = _kette(eye, self.theta, self.perm, self.rueck)


class GedrehteLinear(nn.Module):
    """y = W (R x) + b = (W . R) x + b; W eingefroren. R aus einer eigenen
    QireDrehung ("kette") oder aus Zeile `index` der Drehbank ("bank")."""

    def __init__(self, basis: nn.Linear, drehung: Optional[QireDrehung] = None,
                 bank: Optional[Drehbank] = None, index: int = 0):
        super().__init__()
        self.basis, self.drehung, self.index = basis, drehung, index
        self._bank = [bank]                  # Liste: nicht als Untermodul doppelt zaehlen

    def forward(self, x):
        if self.drehung is not None:
            return self.basis(self.drehung(x))
        rt = self._bank[0].rt[self.index]
        return self.basis((x.reshape(-1, BLOCK) @ rt).reshape(x.shape))


# ---------------------------------------------------------------------------
# Modell und Daten
# ---------------------------------------------------------------------------
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELL, local_files_only=True)


def modell(arm: str, modus: Optional[str] = None):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODELL, local_files_only=True, dtype=torch.float32)
    for p in m.parameters():
        p.requires_grad_(False)
    if arm in SCHALTUNG:
        lagen, cnot = SCHALTUNG[arm]
        modus = modus or ANWENDUNG
        if modus == "bank":
            m.drehbank = Drehbank(len(m.model.layers) * len(ZIELE), lagen, cnot)
            m.register_forward_pre_hook(m.drehbank.aktualisieren)
        for i, schicht in enumerate(m.model.layers):
            for j, name in enumerate(ZIELE):
                lin = getattr(schicht.self_attn, name)
                neu = (GedrehteLinear(lin, bank=m.drehbank, index=len(ZIELE) * i + j) if modus == "bank"
                       else GedrehteLinear(lin, QireDrehung(lagen, cnot)))
                setattr(schicht.self_attn, name, neu)
    elif arm == "lora_r1":
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                                         target_modules=list(ZIELE), bias="none"))
    elif arm != "basis":
        raise ValueError(arm)
    return m


def trainierbar(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def kodiere(tok, frage: str, antwort: Optional[str]) -> Tuple[List[int], List[int]]:
    """(ids, labels); labels = -100 auf <s> und Frage, sonst die Antwort-Tokens + </s>."""
    f = [tok.bos_token_id] + tok(FRAGE_FORMAT.format(frage=frage), add_special_tokens=False)["input_ids"]
    if antwort is None:
        return f, [-100] * len(f)
    a = tok(antwort, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (f + a)[:MAX_TOKENS]
    return ids, ([-100] * len(f) + a)[:MAX_TOKENS]


_DATEN: Dict[str, object] = {}


def daten() -> Dict[str, object]:
    """Feste Aufteilung, fuer alle Seeds gleich. Einmal je Prozess."""
    if _DATEN:
        return _DATEN
    from datasets import load_dataset
    tok = tokenizer()
    ds = load_dataset(DATENSATZ, split="train")
    paare = [(z["conversations"][0]["value"].strip(), z["conversations"][1]["value"].strip())
             for z in ds if len(z["conversations"]) == 2
             and z["conversations"][0]["from"] == "human" and z["conversations"][1]["from"] == "gpt"]
    ordnung = np.random.default_rng(SEED_AUFTEILUNG).permutation(len(paare))
    brauche = N_TRAIN + N_VAL + N_ABSTIMM
    beispiele, vorzeige = [], []
    for i in ordnung:
        frage, antwort = paare[i]
        if not frage or not antwort:
            continue
        ids, lab = kodiere(tok, frage, antwort)
        if len(ids) - sum(1 for x in lab if x != -100) > MAX_FRAGE + 1:
            continue
        if len(beispiele) < brauche:
            beispiele.append((ids, lab))
        else:
            vorzeige.append(frage)
            if len(vorzeige) == N_VORZEIGE:
                break
    _DATEN.update(train=beispiele[:N_TRAIN], validierung=beispiele[N_TRAIN:N_TRAIN + N_VAL],
                  abstimmung=beispiele[N_TRAIN + N_VAL:], vorzeige=vorzeige, tok=tok,
                  n_paare=len(paare))
    return _DATEN


def stapel(beispiele) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lang = max(len(i) for i, _ in beispiele)
    ids = torch.zeros(len(beispiele), lang, dtype=torch.long)
    maske = torch.zeros_like(ids)
    lab = torch.full_like(ids, -100)
    for k, (i, l) in enumerate(beispiele):
        ids[k, :len(i)], maske[k, :len(i)], lab[k, :len(l)] = torch.tensor(i), 1, torch.tensor(l)
    return ids, maske, lab


def _verlust(m, ids, maske, lab, alle: bool = False) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
    """Summen der Kreuzentropie: (Antwort-Tokens, Anzahl, alle Nicht-Pad-Tokens, Anzahl)."""
    logits = m(input_ids=ids, attention_mask=maske).logits[:, :-1].float()
    ziel = lab[:, 1:]
    flach = logits.reshape(-1, logits.shape[-1])
    s = F.cross_entropy(flach, ziel.reshape(-1), ignore_index=-100, reduction="sum")
    if not alle:
        return s, int((ziel != -100).sum()), s, 0
    z_alle = torch.where(maske[:, 1:] == 1, ids[:, 1:], torch.full_like(ziel, -100))
    s_alle = F.cross_entropy(flach, z_alle.reshape(-1), ignore_index=-100, reduction="sum")
    return s, int((ziel != -100).sum()), s_alle, int((z_alle != -100).sum())


@torch.no_grad()
def auswerten(m, beispiele) -> Dict[str, float]:
    m.eval()
    s = n = s_a = n_a = 0.0
    for k in range(0, len(beispiele), BATCH_AUSWERTUNG):
        a, b, c, d = _verlust(m, *stapel(beispiele[k:k + BATCH_AUSWERTUNG]), alle=True)
        s, n, s_a, n_a = s + float(a), n + b, s_a + float(c), n_a + d
    return {"verlust": s / n, "verlust_alle": s_a / n_a, "tokens": int(n)}


def adapter_zustand(m, arm: str) -> Dict[str, torch.Tensor]:
    if arm == "lora_r1":
        from peft import get_peft_model_state_dict
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(m).items()}
    return {k: v.detach().clone() for k, v in m.named_parameters() if v.requires_grad}


# ---------------------------------------------------------------------------
# Eine Zelle -- festes Schrittbudget
# ---------------------------------------------------------------------------
def zelle(arm: str, seed: int, lr: float, menge: str = "validierung", threads: int = 6,
          schritte: int = SCHRITTE, zeitmessung: bool = False, modus: Optional[str] = None
          ) -> Dict[str, object]:
    t0 = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    d = daten()
    m = modell(arm, modus)
    zahl = trainierbar(m)
    if zahl != ERWARTETE_PARAMETER[arm]:
        raise SystemExit(f"{arm}: {zahl} trainierbare Parameter, erwartet {ERWARTETE_PARAMETER[arm]}")
    verlauf, t_train = [], 0.0
    if arm != "basis" and schritte > 0:
        m.train()
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
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
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], CLIP)
                opt.step()
                verlauf.append(float(s.detach()) / n)
                schritt += 1
        t_train = (time.perf_counter() - t1) / max(1, schritte - 1) if schritte > 1 else 0.0
    t2 = time.perf_counter()
    e = auswerten(m, d[menge][:40] if zeitmessung else d[menge])
    out = {"arm": arm, "seed": seed, "lr": lr, "menge": menge, "schritte": len(verlauf),
           METRIK: e["verlust"], "val_verlust_alle": e["verlust_alle"], "tokens": e["tokens"],
           "train_verlust_ende": float(np.mean(verlauf[-20:])) if verlauf else None,
           "trainierbar": zahl, "s_je_schritt": round(t_train, 3),
           "s_auswertung": round(time.perf_counter() - t2, 2),
           "sekunden": round(time.perf_counter() - t0, 2), "threads": threads}
    out["_adapter"] = adapter_zustand(m, arm) if arm != "basis" else None
    return out


# ---------------------------------------------------------------------------
# Wiederaufnahme -- eigene Datei je Lauf-ID, eigener Unterordner, nie latest()
# ---------------------------------------------------------------------------
def _schl(*teile) -> str:
    return "|".join(str(t) for t in teile)


class Zustand:
    def __init__(self, lauf_id: str, konfig: Dict[str, object]):
        self.pfad = RESULTS_DIR.parent / "t004_zwischenstand" / f"{lauf_id}_wiederaufnahme.json"
        self.fingerabdruck = hashlib.sha256(json.dumps(konfig, sort_keys=True).encode()).hexdigest()[:16]
        self.tuning: Dict[str, float] = {}
        self.lr: Dict[str, Dict[str, object]] = {}
        self.records: List[Dict[str, object]] = []
        if self.pfad.exists():
            d = json.loads(self.pfad.read_text(encoding="utf-8"))
            if d.get("fingerabdruck") != self.fingerabdruck:
                raise SystemExit(f"{self.pfad.name} gehoert zu einer ANDEREN Konfiguration. "
                                 "Nicht ueberschreiben -- umbenennen oder Konfiguration pruefen.")
            self.tuning, self.lr, self.records = d["tuning"], d["lr"], d["records"]
            print(f"  WIEDERAUFNAHME: {len(self.tuning)} Tuning-Zellen, {len(self.records)} Messzeilen")

    def speichern(self) -> None:
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerabdruck": self.fingerabdruck, "tuning": self.tuning,
                                   "lr": self.lr, "records": self.records}), encoding="utf-8")
        os.replace(tmp, self.pfad)

    def gemessen(self, arm: str, seed: int) -> bool:
        return any(r["arm"] == arm and r["seed"] == seed for r in self.records)


def konfiguration() -> Dict[str, object]:
    return {"modell": MODELL, "datensatz": DATENSATZ, "ziele": list(ZIELE), "arme": list(ARME),
            "schaltung": {k: list(v) for k, v in SCHALTUNG.items()}, "lora": [LORA_R, LORA_ALPHA],
            "n": [N_TRAIN, N_VAL, N_ABSTIMM, N_VORZEIGE], "max_tokens": MAX_TOKENS,
            "max_frage": MAX_FRAGE, "batch": BATCH, "seed_aufteilung": SEED_AUFTEILUNG,
            "schritte": SCHRITTE, "clip": CLIP, "seeds": list(EVAL_SEEDS),
            "tuning_seeds": list(TUNING_SEEDS), "lr_gitter": {k: list(v) for k, v in LR_GITTER.items()},
            "erweiterungen": ERWEITERUNGEN, "rel_grenze": REL_GRENZE, "format": FRAGE_FORMAT,
            "anwendung": ANWENDUNG}


def _adapter_pfad(lauf_id: str, arm: str, seed) -> "os.PathLike":
    return ADAPTER_DIR / lauf_id / f"{arm}_{seed}.pt"


def run(lauf_id: str = LAUF_ID, threads: int = 6) -> Recorder:
    konfig = konfiguration()
    z = Zustand(lauf_id, konfig)
    rec = Recorder(experiment_id=lauf_id,
                   description=("T-004: QIRE-Drehung (W' = W R) als Adapter auf q/v von "
                                "LLaeMmlein-120M, alpaca-gpt4-deutsch; PRIMAER qire_kron-basis "
                                "einseitig, P2 qire_kron-lora_r1 TOST, P3 qire_5-qire_kron."),
                   config={**konfig, "metrik": METRIK, "familie": list(FAMILIE),
                           "aufloesungspruefung": "lora_r1 trennt sich von basis (zweiseitig)",
                           "parameter": ERWARTETE_PARAMETER, "torch_threads": threads,
                           "fingerabdruck": z.fingerabdruck})

    def sichern():
        rec.records, rec.config["lr"] = list(z.records), dict(z.lr)
        rec.save()

    if not z.gemessen("basis", EVAL_SEEDS[0]):
        r = zelle("basis", 0, 0.0, "validierung", threads)
        r.pop("_adapter")
        z.records += [{**r, "seed": s} for s in EVAL_SEEDS]    # deterministisch, keine Seeds
        z.speichern()
        print(f"  basis {r[METRIK]:.4f} nats/Token ({r['tokens']} Antwort-Tokens)", flush=True)
    for arm in TRAINIERT:
        if arm not in z.lr:
            def messe(lr: float, seed: int, _arm=arm) -> float:
                k = _schl(_arm, seed, f"{lr:.6g}")
                if k not in z.tuning:
                    z.tuning[k] = zelle(_arm, seed, lr, "abstimmung", threads)[METRIK]
                    z.speichern()
                return -z.tuning[k]                       # sweep_lr maximiert
            best, vermerk, mittel = sweep_lr("t004", arm, LR_GITTER[arm], TUNING_SEEDS,
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
        beste[arm] = {"seed": bester["seed"], METRIK: bester[METRIK], "lr": bester["lr"]}
        bp.write_text(json.dumps(beste, indent=2), encoding="utf-8")
        print(f"  Zwischenstand gesichert (Regel 5), bester {arm}-Seed {bester['seed']}")
        sichern()
    return rec


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _reihe(records, arm: str, feld: str) -> Dict[int, float]:
    return {int(r["seed"]): float(r[feld]) for r in records
            if r["arm"] == arm and r.get("menge", "validierung") == "validierung"}


def _wilcoxon(d: np.ndarray, alternative: str = "two-sided") -> float:
    from scipy import stats
    try:
        p = float(stats.wilcoxon(d, alternative=alternative).pvalue)
    except ValueError:
        return 1.0
    return 1.0 if math.isnan(p) else p


def _vergleich(a, b, grenze: float, alternative: str = "two-sided") -> Optional[Dict[str, object]]:
    """delta = a - b im Verlust; negativ heisst: a besser."""
    if len(a) < 2 or set(a) != set(b):
        return None
    s = sorted(a)
    va, vb = np.array([a[k] for k in s]), np.array([b[k] for k in s])
    eq = tost_paired(va, vb, grenze) if grenze > 0 else {"p_tost": 1.0, "achieved": float("nan")}
    return {"delta": float(va.mean() - vb.mean()), "p": _wilcoxon(va - vb, alternative),
            "p_tost": eq["p_tost"], "erreicht": eq["achieved"], "grenze": grenze,
            "siege": int((va < vb).sum()), "n": len(s),
            "mittel_a": float(va.mean()), "mittel_b": float(vb.mean())}


def _kontrollen(records, feld: str, mut: Dict[str, object]) -> Dict[str, object]:
    lora, basis = _reihe(records, "lora_r1", feld), _reihe(records, "basis", feld)
    r = _vergleich(lora, basis, 0.0)
    bestanden = bool(r and r["p"] < 0.05 and (r["delta"] < 0 or not mut.get("richtung")))
    gewinn = -r["delta"] if r else 0.0
    grenze = REL_GRENZE * (1.0 if mut.get("grenze_absolut") else abs(gewinn))
    return {"aufloesung": {"bestanden": bestanden, **(r or {"fehlt": True})},
            "gewinn_lora": gewinn, "grenze": grenze}


def _familie(records, feld: str, grenze: float, mut: Dict[str, object]):
    r = {a: _reihe(records, a, feld) for a in ARME}
    return {"PRIMAER": _vergleich(r["qire_kron"], r["basis"], grenze,
                                  str(mut.get("einseitig", "less"))),
            "P2": _vergleich(r["qire_kron"], r["lora_r1"], grenze),
            "P3": _vergleich(r["qire_5"], r["qire_kron"], grenze)}


_TEXTE = {"P2": ("QIRE_BESSER", "LORA_BESSER",
                 "qire_kron (192 Winkel) senkt den Verlust staerker als LoRA r=1 (30.720)",
                 "LoRA r=1 senkt den Verlust staerker; die 160-fach kleinere Drehung kommt nicht "
                 "gleich weit"),
          "P3": ("VERSCHRAENKUNG_TRAEGT", "VERSCHRAENKUNG_SCHADET",
                 "5 Lagen mit CNOT-Ring schlagen die reine Kronecker-Drehung -- der Ring traegt",
                 "die reine Kronecker-Drehung ist besser -- der CNOT-Ring schadet hier")}
_VERGLEICH = {"PRIMAER": "qire_kron - basis", "P2": "qire_kron - lora_r1", "P3": "qire_5 - qire_kron"}


def _urteil(tag, v, p_holm, k, tost_zuerst: bool = True) -> str:
    if v is None:
        return f"HU_UNVOLLSTAENDIG -- {tag}: Daten fehlen oder Seeds ungleich, keine Aussage."
    if not k["aufloesung"]["bestanden"]:
        return (f"HU_AUFLOESUNG -- {tag}: lora_r1 trennt sich nicht von basis"
                + (" (Zelle fehlt)" if k["aufloesung"].get("fehlt") else "")
                + ". Der Aufbau sieht keine Anpassung; KEIN Nullbefund.")
    z = (f"{_VERGLEICH[tag]}: {v['delta']:+.4f} nats, {v['siege']}/{v['n']} Seeds besser, "
         f"p={v['p']:.4f}, Holm={p_holm:.4f}")
    if tag == "PRIMAER":
        if p_holm < 0.05:
            return f"SENKT_VERLUST -- PRIMAER: die 192-Winkel-Drehung senkt den Verlust; {z}."
        return (f"KEIN_NACHWEIS -- PRIMAER: keine belegte Senkung (einseitig); {z}."
                + (" Beschreibend liegt qire_kron sogar HOEHER als basis." if v["delta"] > 0 else ""))
    z += f", TOST p={v['p_tost']:.4f} (Grenze +-{v['grenze']:.4f})"
    code_a, code_b, satz_a, satz_b = _TEXTE[tag]
    gleich = (f"GLEICHWERTIG -- {tag}: gleichwertig innerhalb +-{v['grenze']:.4f} nats "
              f"({REL_GRENZE:.0%} des LoRA-Gewinns; erreicht +-{v['erreicht']:.4f}); {z}. Positiver "
              "Gleichwertigkeitsnachweis, nicht blosses p>0,05."
              + (" ANMERKUNG: zugleich signifikant -- nachweisbar, aber kleiner als die Grenze."
                 if p_holm < 0.05 else ""))
    code, satz = (code_a, satz_a) if v["delta"] < 0 else (code_b, satz_b)
    reihenfolge = ((v["p_tost"] < 0.05, gleich), (p_holm < 0.05, f"{code} -- {tag}: {satz}; {z}."))
    for bedingung, text in (reihenfolge if tost_zuerst else reihenfolge[::-1]):
        if bedingung:
            return text
    return (f"UNENTSCHIEDEN -- {tag}: weder Unterschied (Holm) noch Gleichwertigkeit belegt; "
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
    k = _kontrollen(records, feld, mut)
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
        for arm in ARME:
            r = [x for x in records if x["arm"] == arm and x.get("menge", "validierung") == "validierung"]
            if r:
                w = np.array([x[feld] for x in r])
                say(f"  {arm:<10} {len(r)} Seeds  Mittel {w.mean():.4f}  sd {w.std(ddof=1) if len(w) > 1 else 0:.4f}"
                    f"  trainierbar {r[0].get('trainierbar', '?')}  lr {r[0].get('lr')}")
        v = _vergleich(_reihe(records, "qire_5", feld), _reihe(records, "basis", feld), k["grenze"])
        if v:
            say(f"  qire_5 - basis {v['delta']:+.4f}  p={v['p']:.4f}")
    raender = {s: v["rand"] for s, v in config.get("lr", {}).items() if v.get("rand") != "innen"}
    if raender:
        say(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {raender}")
        out["lr_rand_warnung"] = raender
    out["verdict"] = " | ".join(out[t]["verdict"].split(" -- ")[0] for t in FAMILIE)
    say(f"\nBEWERTUNG: {out['verdict']}")
    say("=" * 96)
    return out


# ---------------------------------------------------------------------------
# Pflichtvortest (CLAUDE.md §5) -- Teil 1: Theorie am echten Modell, ohne Training
# ---------------------------------------------------------------------------
def _matrix(d: QireDrehung) -> torch.Tensor:
    """R explizit: Spalte j = R e_j (nur fuer die Pruefung, nie im Training)."""
    return copy.deepcopy(d).double()(torch.eye(BREITE, dtype=torch.float64)).T


def _kron_explizit(theta: torch.Tensor) -> torch.Tensor:
    r = torch.ones(1, 1, dtype=torch.float64)
    for t in theta.double():
        c, s = torch.cos(t), torch.sin(t)
        r = torch.kron(r, torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
    return torch.block_diag(r, r, r)


def pruefe_theorie(threads: int = 3) -> bool:
    torch.set_num_threads(threads)
    print("=" * 96)
    print("THEORIEPRUEFUNG T-004")
    zeilen, eye = [], torch.eye(BREITE, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        for arm, (lagen, cnot) in SCHALTUNG.items():
            d = QireDrehung(lagen, cnot)
            zeilen.append((f"{arm}: Winkel 0 -> R = I", float((_matrix(d) - eye).abs().max()), "<", 1e-12))
            d.theta.uniform_(-math.pi, math.pi)
            r = _matrix(d)
            zeilen.append((f"{arm}: R^T R = I (zufaellige Winkel)", float((r.T @ r - eye).abs().max()), "<", 1e-5))
            zeilen.append((f"{arm}: R ist blockdiagonal (3 x 256)",
                           float((r - torch.block_diag(*[r[i:i + BLOCK, i:i + BLOCK]
                                                         for i in range(0, BREITE, BLOCK)])).abs().max()), "<", 1e-12))
        d = QireDrehung(1, False)
        d.theta.uniform_(-math.pi, math.pi)
        zeilen.append(("qire_kron == blockdiag(R1 x ... x R8) explizit",
                       float((_matrix(d) - _kron_explizit(d.theta[0])).abs().max()), "<", 1e-10))
        for lagen, cnot in SCHALTUNG.values():
            bank = Drehbank(24, lagen, cnot).double()
            bank.theta.uniform_(-math.pi, math.pi)
            bank.aktualisieren()
            abw = 0.0
            for k in (0, 7, 23):
                d = QireDrehung(lagen, cnot).double()
                d.theta.copy_(bank.theta[k])
                abw = max(abw, float((torch.block_diag(*[bank.rt[k]] * 3) - _matrix(d).T).abs().max()))
            zeilen.append((f"Drehbank R_k^T == QireDrehung ({lagen} Lagen, CNOT {cnot})", abw, "<", 1e-10))
        # Ohne CNOT sind 5 geteilte Lagen EINE (Winkel addieren sich) -> P3 misst den Ring.
        d5o, d5 = QireDrehung(5, False), QireDrehung(5, True)
        d5o.theta.uniform_(-math.pi, math.pi)
        d5.theta.copy_(d5o.theta)
        zeilen.append(("5 Lagen OHNE CNOT == 1 Lage mit Winkelsumme",
                       float((_matrix(d5o) - _kron_explizit(d5o.theta.double().sum(0))).abs().max()),
                       "<", 1e-10))
        zeilen.append(("Gegenprobe: 5 Lagen MIT CNOT ist KEIN Kronecker-Produkt",
                       float((_matrix(d5) - _kron_explizit(d5.theta.double().sum(0))).abs().max()), ">", 0.1))
        # Am echten Modell
        d_ = daten()
        tok = d_["tok"]
        ids, maske, lab = stapel(d_["validierung"][:3])
        ref = modell("basis")
        ref.eval()
        l_ref = ref(input_ids=ids, attention_mask=maske).logits
        for arm in ("qire_kron", "qire_5", "lora_r1"):
            m = modell(arm)
            m.eval()
            zeilen.append((f"{arm}: Startwert -> Logits == Basismodell",
                           float((m(input_ids=ids, attention_mask=maske).logits - l_ref).abs().max()), "<", 1e-5))
            zeilen.append((f"{arm}: trainierbar {trainierbar(m)} == {ERWARTETE_PARAMETER[arm]}",
                           abs(trainierbar(m) - ERWARTETE_PARAMETER[arm]), "<", 0.5))
        m = modell("qire_5")
        m.drehbank.theta.uniform_(-1, 1)
        m.drehbank.aktualisieren()
        gl = m.model.layers[3].self_attn.v_proj                  # Index 7 der Bank
        d = QireDrehung(5, True)
        d.theta.copy_(m.drehbank.theta[gl.index])
        x = torch.randn(5, BREITE)
        w_r = gl.basis.weight.double() @ _matrix(d)
        zeilen.append((f"GedrehteLinear == x (W . R)^T (W' = W R, v_proj Lage 3, Index {gl.index})",
                       float((gl(x).double() - x.double() @ w_r.T).abs().max()), "<", 1e-4))
        zeilen.append(("Gegenprobe: Ausgabe != x W^T (die Drehung wirkt)",
                       float((gl(x).double() - x.double() @ gl.basis.weight.double().T).abs().max()), ">", 1e-3))
        # Ganzes Modell: bank == kette bei denselben (zufaelligen) Winkeln
        mk = modell("qire_5", "kette")
        for i, schicht in enumerate(mk.model.layers):
            for j, name in enumerate(ZIELE):
                getattr(schicht.self_attn, name).drehung.theta.copy_(m.drehbank.theta[len(ZIELE) * i + j])
        mk.eval()
        m.eval()
        zeilen.append(("qire_5: Logits bank == kette (zufaellige Winkel)",
                       float((m(input_ids=ids, attention_mask=maske).logits
                              - mk(input_ids=ids, attention_mask=maske).logits).abs().max()), "<", 1e-3))
        # Maske: nur Antwort-Tokens tragen den Verlust
        i, l = kodiere(tok, "Wie heisst die Hauptstadt?", "Berlin.")
        n_frage = len(kodiere(tok, "Wie heisst die Hauptstadt?", None)[0])
        n_ant = len(tok("Berlin.", add_special_tokens=False)["input_ids"]) + 1
        zeilen.append(("Maske: Frage -100, Antwort + </s> Ziel",
                       float(sum(x != -100 for x in l[:n_frage]) + abs(sum(x != -100 for x in l) - n_ant)
                             + (l[-1] != tok.eos_token_id)), "<", 0.5))
        zeilen.append((f"Aufteilung {len(d_['train'])}/{len(d_['validierung'])}/{len(d_['abstimmung'])}"
                       f"/{len(d_['vorzeige'])}, max. {max(len(x) for x, _ in d_['train'])} Tokens",
                       float(abs(len(d_['train']) - N_TRAIN) + abs(len(d_['validierung']) - N_VAL)
                             + abs(len(d_['vorzeige']) - N_VORZEIGE)
                             + (max(len(x) for x, _ in d_['train']) > MAX_TOKENS)), "<", 0.5))
    # Gradient fliesst in die Winkel, nicht in W
    m = modell("qire_5")
    s, n, _, _ = _verlust(m, ids, maske, lab)
    (s / n).backward()
    g = m.drehbank.theta.grad
    zeilen.append(("qire_5: Gradient an allen 24 Winkelsaetzen != 0 (Anzahl ohne)",
                   24.0 if g is None else float((g.abs().amax(dim=(1, 2)) == 0).sum()), "<", 0.5))
    zeilen.append(("Basisgewichte ohne Gradient",
                   float(sum(p.grad is not None for n_, p in m.named_parameters() if "drehbank" not in n_)),
                   "<", 0.5))
    ok = True
    for name, wert, rel, grenze in zeilen:
        good = wert < grenze if rel == "<" else wert > grenze
        ok &= good
        print(f"  {name:<62} {wert:.2e} {rel} {grenze:g}  {'OK' if good else 'FEHLER'}")
    print("  " + ("BESTANDEN" if ok else "FEHLGESCHLAGEN"))
    return bool(ok)


def test_gitterverlaengerung() -> bool:
    """Regel 1(b) in beide Richtungen; sweep_lr maximiert -> messe gibt -Verlust."""
    faelle = [("Optimum im Startgitter", 0.003, "innen", 0.003),
              ("eine Verlaengerung oben", 0.09, "innen", 0.09),
              ("eine Verlaengerung unten", 1e-3 / 3, "innen", 1e-3 / 3),
              ("ausserhalb oben", 300.0, "oben", 0.03 * 27),
              ("ausserhalb unten", 1e-9, "unten", 1e-3 / 27)]
    print("=" * 96)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen, Verlust (kleiner = besser)")
    ok = True
    for name, opt, rand_soll, rate_soll in faelle:
        best, rand, _ = sweep_lr("t004", "qire_kron", LR_GITTER["qire_kron"], (1, 2), ERWEITERUNGEN,
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
VERSATZ = {"basis": 0.0, "qire_kron": -0.10, "qire_5": -0.15, "lora_r1": -0.20}


def _kunst(seed: int, streuung: float = 0.01, versatz=None, weg: Sequence[str] = ()) -> Recorder:
    """basis deterministisch (wie im Lauf: ein Wert fuer alle Seeds). Trainierte
    Arme: gemeinsamer Seed-Anteil (gleiche Datenreihenfolge) + EIGENES Rauschen.
    `val_verlust_alle` (plausibles Falschfeld) bekommt UMGEKEHRTE Versaetze und
    eigenes Rauschen -- ein Fehler im Metrikfeld kippt jedes Richtungsurteil."""
    def rng(*t):
        return np.random.default_rng([seed, zlib.crc32("|".join(map(str, t)).encode())])
    vs = {**VERSATZ, **(versatz or {})}
    rec = Recorder("T-004-vortest", "synthetisch", config={"lr": {}})
    for s in EVAL_SEEDS:
        grund = rng("grund", s).normal(0, 0.005)
        for arm, v in vs.items():
            if arm in weg:
                continue
            eig = 0.0 if arm == "basis" else grund + rng(arm, s).normal(0, streuung)
            alle = 0.0 if arm == "basis" else eig + rng("alle", arm, s).normal(0, streuung)
            rec.add(arm=arm, seed=s, lr=0.01, menge="validierung", val_verlust=NIVEAU + v + eig,
                    val_verlust_alle=NIVEAU_ALLE - v + alle, trainierbar=ERWARTETE_PARAMETER[arm])
    return rec


FAELLE = [
    # (Name, Test, kwargs, Code, Zweigtext, Ziehungen, Anteil)
    ("P-S   qire_kron -0,10", "PRIMAER", {}, "SENKT_VERLUST", "192-Winkel-Drehung senkt", 5, 1.0),
    ("P-N   qire_kron +0,05 (schlechter)", "PRIMAER", dict(versatz={"qire_kron": 0.05}),
     "KEIN_NACHWEIS", "sogar HOEHER", 5, 1.0),
    ("P-U   Nulleffekt, grosses Rauschen", "PRIMAER", dict(streuung=0.05, versatz={"qire_kron": 0.0}),
     "KEIN_NACHWEIS", "keine belegte Senkung", 40, 0.80),
    ("P-A   Aufloesung faellt (lora = basis)", "PRIMAER", dict(versatz={"lora_r1": 0.0}),
     "HU_AUFLOESUNG", "KEIN Nullbefund", 40, 0.80),
    ("P-Ar  lora SCHLECHTER als basis: Trennung zaehlt", "PRIMAER", dict(versatz={"lora_r1": 0.10}),
     "SENKT_VERLUST", "192-Winkel-Drehung senkt", 5, 1.0),
    ("P-An  lora-Zelle fehlt", "PRIMAER", dict(weg=["lora_r1"]), "HU_AUFLOESUNG", "Zelle fehlt", 5, 1.0),
    ("P-X   qire_kron fehlt", "PRIMAER", dict(weg=["qire_kron"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("V-L   LoRA +0,10 besser", "P2", {}, "LORA_BESSER", "nicht gleich weit", 5, 1.0),
    ("V-Q   qire_kron besser als LoRA", "P2", dict(versatz={"qire_kron": -0.30}), "QIRE_BESSER",
     "staerker als LoRA", 5, 1.0),
    ("V-G   gleich, kleines Rauschen", "P2", dict(streuung=0.003, versatz={"qire_kron": -0.20}),
     "GLEICHWERTIG", "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("V-Gs  +0,005 bei winzigem Rauschen: signifikant UND gleichwertig", "P2",
     dict(streuung=0.0005, versatz={"qire_kron": -0.195}), "GLEICHWERTIG", "zugleich signifikant", 5, 1.0),
    ("V-U   gleich, grosses Rauschen", "P2", dict(streuung=0.08, versatz={"qire_kron": -0.20}),
     "UNENTSCHIEDEN", "weder Unterschied", 40, 0.80),
    ("V-X   lora fehlt", "P2", dict(weg=["lora_r1"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("T-T   qire_5 -0,05 vor qire_kron", "P3", {}, "VERSCHRAENKUNG_TRAEGT", "der Ring traegt", 5, 1.0),
    ("T-G   qire_5 = qire_kron, kleines Rauschen", "P3", dict(streuung=0.003, versatz={"qire_5": -0.10}),
     "GLEICHWERTIG", "Positiver Gleichwertigkeitsnachweis", 5, 1.0),
    ("T-S   qire_5 +0,05 hinter qire_kron", "P3", dict(versatz={"qire_5": -0.05}),
     "VERSCHRAENKUNG_SCHADET", "Ring schadet", 5, 1.0),
    ("T-U   gleich, grosses Rauschen", "P3", dict(streuung=0.08, versatz={"qire_5": -0.10}),
     "UNENTSCHIEDEN", "weder Unterschied", 40, 0.80),
    ("T-X   qire_5 fehlt", "P3", dict(weg=["qire_5"]), "HU_UNVOLLSTAENDIG", "Daten fehlen", 5, 1.0),
    ("X-P   qire_5 fehlt -> PRIMAER trotzdem gerechnet", "PRIMAER", dict(weg=["qire_5"]),
     "SENKT_VERLUST", "192-Winkel-Drehung senkt", 5, 1.0),
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
    d_m, d_a = r("qire_kron", METRIK) - r("basis", METRIK), r("qire_kron", "val_verlust_alle") - r("basis", "val_verlust_alle")
    sd_kl = float(np.std(r("qire_kron", METRIK) - r("lora_r1", METRIK), ddof=1))
    soll = math.sqrt(2) * 0.01
    pruef = [("val_verlust_alle - val_verlust NICHT konstant (sd je Arm > 0)", min(diff_sd) > 1e-4),
             ("gepaarte Differenz kippt mit dem Feld", d_m.mean() < 0 < d_a.mean()),
             ("basis deterministisch (sd 0), trainierte Arme streuen",
              float(np.std(r("basis", METRIK))) == 0 and float(np.std(d_m)) > 1e-3),
             (f"eigenes Rauschen: sd(qire_kron - lora) {sd_kl:.4f} ~ {soll:.4f}", 0.5 * soll < sd_kl < 2 * soll)]
    for name, good in pruef:
        print(f"  {'OK    ' if good else 'FEHLER'} {name}")
    return all(g for _, g in pruef)


MUTATIONEN = [("Metrikfeld val_verlust_alle statt val_verlust", {"feld": "val_verlust_alle"}),
              ("Signifikanz VOR TOST", {"tost_zuerst": False}),
              ("Aufloesung verlangt Richtung statt Trennung", {"richtung": True}),
              ("PRIMAER einseitig in FALSCHER Richtung", {"einseitig": "greater"}),
              ("TOST-Grenze absolut 0,2 statt relativ", {"grenze_absolut": True})]


def vortest() -> bool:
    print("=" * 96)
    print("VORTEST T-004 -- jeder Ausgang einzeln, ueber mehrere Ziehungen, Marke UND Zweigtext")
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
# Zeitmessung und Chat
# ---------------------------------------------------------------------------
def zeit(threads: int = 3, schritte: int = 6) -> Dict[str, float]:
    """s/Schritt je Arm mit kleinem Budget (Schritt 0 = Aufwaermen, nicht gezaehlt)
    und Auswertungszeit auf 40 Beispielen, dann Hochrechnung fuer den Plan."""
    print("=" * 96)
    print(f"ZEITMESSUNG T-004 -- {threads} Threads, {schritte} Schritte je Arm, Batch {BATCH} x <= {MAX_TOKENS}")
    daten()
    t: Dict[str, float] = {}
    laeufe = [(a, None) for a in ARME] + [(a, "kette" if ANWENDUNG == "bank" else "bank")
                                          for a in SCHALTUNG]
    for arm, modus in laeufe:
        r = zelle(arm, 1, 0.01 if arm != "lora_r1" else 1e-3, "validierung", threads, schritte, True, modus)
        name = arm if modus is None else f"{arm}[{modus}]"
        t[name], t[f"{name}_ausw"] = r["s_je_schritt"], r["s_auswertung"] / 40
        print(f"  {name:<17} {r['s_je_schritt']:>6.3f} s/Schritt   Auswertung {t[f'{name}_ausw']:.3f} s/Beispiel"
              f"   trainierbar {r['trainierbar']}" + ("   (Vergleich, nicht im Lauf)" if modus else ""),
              flush=True)
    for s in (200, 250, 300, 400):
        zellen = {a: (s * t[a] + N_VAL * t[f"{a}_ausw"], s * t[a] + N_ABSTIMM * t[f"{a}_ausw"]) for a in TRAINIERT}
        gesamt = N_VAL * t["basis_ausw"] + sum(len(EVAL_SEEDS) * v + len(TUNING_SEEDS) * 5 * w
                                               for v, w in zellen.values())
        print(f"  SCHRITTE={s}: Messzelle " + ", ".join(f"{a} {v / 60:.1f} min" for a, (v, _) in zellen.items())
              + f" | GESAMT (5 Raten je Arm) {gesamt / 3600:.1f} h bei {threads} Threads")
    return t


def chat(arm: str = "qire_kron", frage: Optional[str] = None, lauf_id: str = LAUF_ID) -> str:
    beste = ADAPTER_DIR / lauf_id / "beste.json"
    m = modell(arm)
    if arm != "basis":
        if not beste.exists() or arm not in json.loads(beste.read_text(encoding="utf-8")):
            raise SystemExit(f"Kein gespeicherter Adapter fuer {arm} in {beste.parent} -- erst `lauf`.")
        seed = json.loads(beste.read_text(encoding="utf-8"))[arm]["seed"]
        gewichte = torch.load(_adapter_pfad(lauf_id, arm, seed), weights_only=True)
        if arm == "lora_r1":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(m, gewichte)
        else:
            if m.load_state_dict(gewichte, strict=False).unexpected_keys:
                raise SystemExit("Adapter passt nicht zum Modell (unerwartete Schluessel).")
        print(f"  Adapter {arm}, Seed {seed} ({sum(v.numel() for v in gewichte.values())} Werte)")
    tok = tokenizer()
    frage = frage or "Erkläre in zwei Sätzen, was ein Quantencomputer ist."
    ids, _ = kodiere(tok, frage, None)
    m.eval()
    with torch.no_grad():
        aus = m.generate(torch.tensor([ids]), max_new_tokens=150, do_sample=False,
                         repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=0)
    antwort = tok.decode(aus[0, len(ids):], skip_special_tokens=True).strip()
    print(f"FRAGE: {frage}\nANTWORT ({arm}): {antwort}")
    return antwort


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "vortest":
        raise SystemExit(0 if (pruefe_theorie() & test_gitterverlaengerung() & vortest()) else 1)
    elif befehl == "zeit":
        zeit(*(int(a) for a in sys.argv[2:4]))
    elif befehl == "auswertung":
        analyse(sys.argv[2] if len(sys.argv) > 2 else LAUF_ID)
    elif befehl == "chat":
        chat(sys.argv[2] if len(sys.argv) > 2 else "qire_kron", sys.argv[3] if len(sys.argv) > 3 else None)
    elif befehl == "lauf":
        if not (pruefe_theorie() and test_gitterverlaengerung() and vortest()):
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print(f"  Nachholen: python -m quantum_sim.chatbot_adapter auswertung {LAUF_ID}")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
    else:
        print("Befehle: vortest | zeit [threads] [schritte] | lauf | auswertung [ID|Pfad] | chat [arm] [Frage]")
