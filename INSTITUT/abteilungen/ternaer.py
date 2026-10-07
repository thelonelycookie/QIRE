"""Abteilung Clifford -- Clifford-Zelle gegen ternaere Schicht (Agenda T-ternaer, Quantenstufe Q2).

VORREGISTRIERUNG, 07.10.2026, VOR JEDER ZAHL (Plan: Opus, Umsetzung: Sonnet).
KORREKTUR 07.10. vor jeder Messzahl (nur Schnelllauf mit 40 Schritten gelaufen, verworfen):
die erste Fassung verglich die Zelle (256x256-Operator, Kopf sieht 256 Merkmale) mit einer
ternaeren Schicht 256 -> 8 -> 10, also mit 8-Kanal-Flaschenhals -- unfair. Jetzt ist der
primaere ternaere Gegenpart ein 256x256-Operator mit demselben Kopf 256 -> 10; die kleinen
Varianten bleiben als ternaer8/ternaer8_quad (nur beschreibend). V-2 geaendert: dicht_quad
statt linear_voll, zweiseitige Trennung (Lehre Q-015).

FRAGE: Die Clifford-gerundete QIRE-Zelle rechnet ohne Multiplikation in der
Schaltung. BitNet b1.58 (Ma u. a. 2024, arXiv:2402.17764) erreicht Multiplikations-
freiheit klassisch mit ternaeren Gewichten {-1, 0, +1}·alpha. Ist die quanten-
abgeleitete Clifford-Zelle bei gleichem Budget mindestens gleich gut wie eine
ternaere Schicht?

NEUHEIT (Stand 07.10.2026, Suche des Direktors): ternaere Netze (BitNet) und
Clifford-/Stabilisator-QNN sind je einzeln bekannt; der direkte Vergleich
Clifford-Butterfly-Zelle gegen ternaere Schicht (Genauigkeit gegen Multiplikationen
und Bit) wurde nicht gefunden. Kein Befund ueber Quantenvorteil -- Q2 heisst: exakte
QM im Simulator, Vergleich klassisch gerechnet.

ARME (MNIST 16x16 = 8 Qubits = 256 Dim., Daten wie clifford.py fuer mnist):
  voll          QIRE-Zelle trainiert, freie Winkel (Referenz)
  cliff0        voll trainiert, alle Winkel auf k·pi/2 gerundet (M = 0), frischer Kopf
                nachtrainiert -- clifford._gerundet, nicht kopiert
  cliff2/cliff4 wie cliff0, die 2 bzw. 4 wirksamsten Winkel bleiben frei
                (clifford._magie_reihenfolge)
  ternaer_quad  PRIMAERER Gegenpart: ternaere Matrix 256x256 nach BitNet b1.58 (Latentgewichte in
                Gleitkomma, Vorwaerts W_q = clip(round(W/g), -1, 1)·g mit g = mean(|W|), Straight-
                Through-Estimator, ohne Bias), danach QUADRIERT (gleiche Nichtlinearitaet wie die
                Born-Auslese), dann Kopf 256 -> 10 wie bei der Zelle
  ternaer       wie ternaer_quad ohne Quadrierung (beschreibend)
  dicht_quad    Gleitkomma 256x256 -> quadriert -> Kopf 256 -> 10 (Obergrenze: gleiche Struktur
                ohne Quantisierung)
  ternaer8      ternaere Schicht 256 -> 8, Kopf 8 -> 10 (beschreibend, Pareto)
  ternaer8_quad wie ternaer8, quadriert (beschreibend, Pareto)
  identitaet    Born-Auslese direkt auf x (Untergrenze, wie C-0 in clifford.py)
Eingabe fuer alle Arme gleich normiert: psi = x/||x||. Alle Arme ausser ternaer8* haben den
Kopf 256 -> 10 mit je Seed IDENTISCHEM Startzustand (der von clifford._baue; der Vortest prueft
das); ternaer8* haben Kopf 8 -> 10. Kopf-Kosten werden separat ausgewiesen.

HYPOTHESEN siehe HYPOTHESEN (T-1, T-2, T-3, V-1, V-2). Entscheidungsbaum (alles vor
dem Lauf festgelegt):
  d = Genauigkeit(Clifford-Arm) - Genauigkeit(ternaer_quad), gepaart je Seed.
  Nichtunterlegenheitsgrenze DELTA = 1 pp. Einseitiges 95-%-KI der Differenz
  (t-Verteilung, wie clifford.py). H1: KI-Untergrenze > -DELTA. H0: KI-Obergrenze
  < -DELTA. Sonst HU. Holm (alpha = 0,05) ueber {T-1, T-2}: wirksam ist die Korrektur
  ueber einseitige t-Tests (H1: d <= -DELTA verworfen; H0: d >= -DELTA verworfen), je
  Richtung eigene Holm-Familie. H1/H0 gelten nur, wenn der Holm-korrigierte Test
  verworfen wird, sonst HU.
  Vorbedingungen (urteilstragend, sonst T-1/T-2 = HU):
    V-1 voll schlaegt identitaet (exakter Wilcoxon p < 0,05, voll > identitaet).
    V-2 dicht_quad trennt sich von identitaet (exakter Wilcoxon zweiseitig p < 0,05, Trennung
    statt Richtung, Lehre Q-015).
  T-3 beschreibend: Pareto-Front Genauigkeit gegen (Multiplikationen, Bit) aller Arme,
  einmal fuer die Mittelschicht, einmal mit Kopf.

EHRLICHE VORAB-ERWARTUNG: ternaer_quad (65.536 ternaere Gewichte) ist vermutlich genauer als
cliff0 (32 Winkel), dafuer ~1.600x mehr Speicherbits (~104.000 gegen 64) und ~8x mehr
Additionen (65.280 gegen 8.192). Beide haben dieselbe Quadrierung (256 Mult.). Der erwartete
Befund ist die Pareto-Lage (T-3), nicht ein Sieg; T-1 geht wahrscheinlich H0 oder HU aus.

KOSTENMODELL (kosten(arm), im Ergebnis gespeichert; Details in der Funktion):
Multiplikationen je Bild (Quadrierungen der Born-Auslese zaehlen als Multiplikationen,
Kopf separat), Additionen je Bild (obere Schranke), Speicher in Bit der Mittelschicht
(Gleitkomma 32 Bit; Clifford-Winkel 2 Bit; ternaer log2(3) Bit je Gewicht + 32 Bit g;
freie Winkel 32 Bit). Globale Skalare (1/sqrt(2)^m der Hadamard-Faktoren, g) werden in
die Kopfgewichte gefaltet und kosten nichts; die Eingangsnormierung (d + 11 Mult.) ist
fuer alle Arme gleich und steht separat. clifford.multiplikationen() zaehlt diese
Normierung NICHT (Pruefung 07.10.: sonst stimmt die Formel fuer die Gatterform).

SEEDS (frisch, kein Treffer in abteilungen/*.py und ergebnisse/*.json; der Vortest
prueft das): Mess 310-319, Tuning 960-962. Lernrate je Arm ueber
labor.gemeinsam.gittersuche_stabil (Faktor 2, Stabilitaetsabstand, Gitter beidseitig
verlaengert) auf den Tuning-Seeds, Validierung aus dem Trainingsteil. Cliff-Arme werden
auf den Voll-Modellen der Tuning-Seeds getunt (wie skala_v2, Phase 2), je Arm eigene Rate.
Budget wie clifford.py mnist: 6.000 Trainingsbilder, 1.500 Vollbatch-Adam-Schritte,
5.000 Testbilder.

BEDENKEN (offen):
 1. Die ternaere Matrix hat volle 256x256-Struktur (65.536 Gewichte), die Zelle nur 32 Winkel in
    Butterfly-Struktur: gleiche Ein-/Ausgangsbreite, nicht gleiche Parameterzahl. Gleiche
    Parameterzahl prueft die Folgefrage T-ternaer-butterfly.
 2. BitNet quantisiert zusaetzlich Aktivierungen (8 Bit) und nutzt LayerNorm; hier nur die
    Gewichtsternaerisierung (Aktivierungen bleiben Gleitkomma). Fuer echte Multiplikations-
    freiheit der ternaeren Schicht waere ein ganzzahliger Eingang noetig -- nicht Teil der Frage.
 3. Die Zelle lernt frei und wird nachtrainiert gerundet; ternaer lernt ternaer-bewusst (STE von
    Anfang an): verschiedene Verfahren, gleiche Schrittzahl.
 4. Rechenzeit: die ternaeren 256x256-Arme sind teurer als die 256 -> 8-Arme.
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from abteilungen import clifford as cl
from labor.gemeinsam import (ERGEBNISSE, bilder, genauigkeit, gittersuche_stabil, parallel, speichere,
                             trainiere, wilcoxon_exakt, zusammenfassung)
from labor.gemeinsam import _roh

N_QUBITS = 8                      # 2^8 = 256 = 16x16 Pixel
D = 2 ** N_QUBITS
LAGEN = cl.LAGEN
N_WINKEL = LAGEN * N_QUBITS       # 32
KANAELE_KLEIN = 8                 # Breite der Arme ternaer8/ternaer8_quad
DELTA = 0.01                      # Nichtunterlegenheitsgrenze: 1 pp
ALPHA = 0.05
SEEDS = tuple(range(310, 320))    # frisch
TUNING_SEEDS = (960, 961, 962)    # frisch
ENTWURF_SEEDS = (990, 991)        # nur Rauchtest
ENTWURF_TUNING = (992, 993, 994)  # nur Rauchtest
SCHRITTE, N_TRAIN, N_TEST = 1500, 6_000, 5_000
LR_GITTER = (0.02, 0.04, 0.08, 0.16, 0.32)
FAKTOR = 2
KOLLAPS = 0.05

ARME = ["voll", "cliff0", "cliff2", "cliff4", "ternaer_quad", "ternaer", "dicht_quad", "ternaer8_quad", "ternaer8",
        "identitaet"]
CLIFF_K = {"cliff0": 0, "cliff2": 2, "cliff4": 4}
ZELLEN_ARME = ["voll", "cliff0", "cliff2", "cliff4", "identitaet"]
MITTEL_ARME = ["ternaer_quad", "ternaer", "dicht_quad", "ternaer8_quad", "ternaer8"]
FREIE_WINKEL = {"voll": N_WINKEL, "cliff0": 0, "cliff2": 2, "cliff4": 4, "identitaet": 0}
RATEN_PHASE1 = ["voll", "identitaet"] + MITTEL_ARME
RATEN_PHASE2 = ["cliff0", "cliff2", "cliff4"]
BIT_TERNAER = math.log2(3)


def _breite(arm):
    return KANAELE_KLEIN if arm.startswith("ternaer8") else D

HYPOTHESEN = {
    "T-1": "Primaer: cliff0 ist nicht schlechter als ternaer_quad (Nichtunterlegenheit, Grenze 1 pp): "
           "einseitiges 95-%-KI der gepaarten Differenz cliff0 - ternaer_quad > -0,01 -> H1; KI ganz "
           "unter -0,01 -> H0; sonst HU. Holm ueber {T-1, T-2}. Erwartung: eher H0/HU "
           "(65.536 ternaere Gewichte gegen 32 Winkel).",
    "T-2": "cliff4 (vier freie Magiewinkel) ist nicht schlechter als ternaer_quad (gleiche Logik, Holm "
           "ueber {T-1, T-2}).",
    "T-3": "Beschreibend: Pareto-Front Genauigkeit gegen (Multiplikationen, Bit) aller Arme "
           "(Mittelschicht und mit Kopf). Erwartung: ternaer_quad genauer, Clifford ~1.600x kleiner im Speicher "
           "-- der Befund ist die Pareto-Lage, nicht ein Sieg.",
    "V-1": "Vorbedingung: voll schlaegt identitaet (exakter Wilcoxon p < 0,05). Sonst T-1/T-2 = HU.",
    "V-2": "Vorbedingung/Aufloesung: dicht_quad (Gleitkomma 256x256, quadriert) trennt sich von identitaet "
           "(exakter Wilcoxon zweiseitig p < 0,05, Trennung statt Richtung). Sonst T-1/T-2 = HU.",
}


# ------------------------------------------------------------------ Kostenmodell
def kosten(arm: str) -> dict:
    """Kosten je Bild und Arm (n = 8 Qubits, d = 256, L = 4 Lagen, 32 Winkel).

    mult_mitte      Multiplikationen der Mittelschicht INKLUSIVE Quadrierung (Born / ternaer_quad)
    mult_quadrierung davon die Quadrierungen (Born: d = 256; ternaer_quad: 8)
    mult_kopf       Multiplikationen des linearen Kopfes (separat)
    mult_norm       Eingangsnormierung, fuer alle Arme gleich: d Quadrate + 1 Kehrwertwurzel + 10
                    Skalierungen der Logits = 267 (steht NICHT in mult_mitte/mult_gesamt)
    adds_mitte      Additionen der Mittelschicht, obere Schranke (jede Drehung = Hadamard-artig
                    oder frei: d Additionen; Permutationen/Vorzeichen kosten real weniger)
    adds_kopf       Additionen des Kopfes inkl. Bias
    bit_mitte       Speicher der Mittelschicht; bit_kopf Speicher des Kopfes (Gleitkomma, 32 Bit)
    Globale Skalare (Hadamard-Faktoren 1/sqrt(2), gamma) werden in die Kopfgewichte gefaltet.
    Zelle: eine 2x2-Drehung mit freiem Winkel auf 2^n Amplituden = 2d Mult. + d Add.; auf dem
    Clifford-Gitter 0 Mult. (Mult. aus clifford.multiplikationen(), dort Drehungen*2d + d + 10d).
    Ternaer (Breite w = 256 oder 8): w Ausgaben * (d - 1) Additionen von ±x (Nullen entfallen),
    0 Mult. (gamma gefaltet), plus w Quadrierungen bei *_quad. dicht_quad: w*d Mult. + w*(d-1) Add."""
    if arm in ZELLEN_ARME:
        f = FREIE_WINKEL[arm]
        gesamt = cl.multiplikationen(N_QUBITS, magie=f)         # Drehungen*2d + d + 10 d
        kopf = D * 10
        bit = (0 if arm == "identitaet"
               else (N_WINKEL - f) * 2 + f * 32)
        k = {"mult_mitte": gesamt - kopf, "mult_quadrierung": D, "mult_kopf": kopf,
             "adds_mitte": 0 if arm == "identitaet" else N_WINKEL * D, "adds_kopf": D * 10,
             "bit_mitte": float(bit), "bit_kopf": float((D * 10 + 10) * 32)}
    elif arm in MITTEL_ARME:
        w = _breite(arm)
        quad = w if arm.endswith("_quad") else 0
        dicht = arm == "dicht_quad"
        k = {"mult_mitte": (w * D if dicht else 0) + quad, "mult_quadrierung": quad,
             "mult_kopf": w * 10, "adds_mitte": w * (D - 1), "adds_kopf": w * 10,
             "bit_mitte": float(w * D * 32) if dicht else w * D * BIT_TERNAER + 32,
             "bit_kopf": float((w * 10 + 10) * 32)}
    else:
        raise ValueError(arm)
    k["mult_norm"] = D + 1 + 10
    k["mult_gesamt"] = k["mult_mitte"] + k["mult_kopf"]
    k["bit_gesamt"] = k["bit_mitte"] + k["bit_kopf"]
    k["freie_winkel"] = FREIE_WINKEL.get(arm)
    return k


# ------------------------------------------------------------------ ternaere Schicht (BitNet b1.58)
def ternarisiere(w: torch.Tensor):
    """BitNet b1.58: gamma = mean(|W|), W_q = clip(round(W/gamma), -1, 1)·gamma, Vorwaerts mit dem
    Straight-Through-Estimator (Rueckwaerts: Gradient laeuft unveraendert auf die Latentgewichte).
    Rueckgabe (W_q mit STE, gamma)."""
    gamma = w.abs().mean().clamp_min(1e-8)
    wq = torch.clamp(torch.round(w / gamma), -1, 1) * gamma
    return w + (wq - w).detach(), gamma


class MittelModell(nn.Module):
    """psi = x/||x|| -> Schicht 256 -> w (ternaer | Gleitkomma) [-> quadriert] -> Kopf w -> 10.
    w = 256 (ternaer, ternaer_quad, dicht_quad) oder 8 (ternaer8*). Kopf und Latentgewichte haengen
    NUR vom Seed ab, nicht vom Arm. Der 256er-Kopf ist exakt der Startkopf der Zellenarme."""

    def __init__(self, art: str, seed: int):
        super().__init__()
        assert art in MITTEL_ARME
        self.art, self.quad = art, art.endswith("_quad")
        self.tern = art.startswith("ternaer")
        w = _breite(art)
        if w == D:
            ref, _ = cl._baue("voll", seed, N_QUBITS)
            self.kopf = nn.Linear(D, 10)
            self.kopf.load_state_dict(ref.kopf.state_dict())
        else:
            torch.manual_seed(seed)
            self.kopf = nn.Linear(w, 10)
        torch.manual_seed(seed + 7919)
        self.gewicht = nn.Parameter(nn.Linear(D, w, bias=False).weight.detach().clone())

    def schicht(self):
        return ternarisiere(self.gewicht)[0] if self.tern else self.gewicht

    def forward(self, x):
        psi = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        z = psi @ self.schicht().T
        if self.quad:
            z = z ** 2
        return self.kopf(z)


# ------------------------------------------------------------------ Daten, Modelle, Arbeiter
@lru_cache(maxsize=8)
def _daten(seed, n_train=N_TRAIN, n_test=N_TEST):
    if (n_train, n_test) == (N_TRAIN, N_TEST):
        return cl.daten("mnist", seed)               # genau wie clifford.py fuer mnist
    return bilder("mnist", seed, n_train=n_train, n_test=n_test, seite=16)


def _baue_zelle(arm, seed):
    m, _ = cl._baue("ohne_schaltung" if arm == "identitaet" else "voll", seed, N_QUBITS)
    return m


def _zelle_aus_zustand(zustand):
    m = cl.QireZelle(n=N_QUBITS)
    m.load_state_dict(zustand)
    return m


def _tuning_arbeit(a):
    """a = (arm, seed, lr, (schritte, n_train, n_test), zustand_voll | None); Seed an Stelle 1."""
    arm, s, lr, (schritte, n_train, n_test), zustand = a
    xtr, ytr, _, _ = _daten(s, n_train, n_test)
    n_val = len(ytr) // 4
    xt, yt, xv, yv = xtr[n_val:], ytr[n_val:], xtr[:n_val], ytr[:n_val]
    if arm in CLIFF_K:
        m = cl._gerundet(_zelle_aus_zustand(zustand), CLIFF_K[arm], s)
    elif arm in MITTEL_ARME:
        m = MittelModell(arm, s)
    else:
        m = _baue_zelle(arm, s)
    trainiere(m, xt, yt, schritte, lr, seed=s)
    e = {"acc": genauigkeit(m(xv), yv)}
    if arm == "voll":
        e["zustand"] = {k: v.detach().clone() for k, v in m.state_dict().items()}
    return e


def _mess_zelle(a):
    """Ein Seed, Zellenarme: voll, cliff0/2/4 (aus demselben voll), identitaet."""
    s, lr, (schritte, n_train, n_test) = a
    xtr, ytr, xte, yte = _daten(s, n_train, n_test)
    z = {"seed": s}
    voll = _baue_zelle("voll", s)
    trainiere(voll, xtr, ytr, schritte, lr["voll"], seed=s)
    z["voll"] = genauigkeit(voll(xte), yte)
    th = voll.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    z["abstand_zum_gitter"] = float((th - cl.runde_clifford(th)).abs().mean())
    for arm, k in CLIFF_K.items():
        m = cl._gerundet(voll, k, s)
        trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
        z[arm] = genauigkeit(m(xte), yte)
    m = _baue_zelle("identitaet", s)
    trainiere(m, xtr, ytr, schritte, lr["identitaet"], seed=s)
    z["identitaet"] = genauigkeit(m(xte), yte)
    return z


def _mess_mitte(a):
    """Ein Seed, klassische Mittelarme (ternaer*, dicht_quad)."""
    s, lr, (schritte, n_train, n_test) = a
    xtr, ytr, xte, yte = _daten(s, n_train, n_test)
    z = {"seed": s}
    for arm in MITTEL_ARME:
        m = MittelModell(arm, s)
        trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
        z[arm] = genauigkeit(m(xte), yte)
        if arm.startswith("ternaer"):
            with torch.no_grad():
                z[f"{arm}_nullanteil"] = float((m.schicht() == 0).float().mean())
    return z


# ------------------------------------------------------------------ Lauf
def _belegte_seeds() -> set:
    """Alle dreistelligen Zahlen in den anderen Modulen + Seeds in ergebnisse/*.json (ohne -schnell)."""
    belegt = set()
    for f in Path(__file__).parent.glob("*.py"):
        if f.name != Path(__file__).name:
            belegt |= {int(x) for x in re.findall(r"\b\d{3}\b", f.read_text(encoding="utf-8"))}
    for f in ERGEBNISSE.glob("*.json"):
        if "schnell" in f.name or f.name.startswith("T-ternaer"):
            continue
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for key in ("seeds", "tuning_seeds"):
            if isinstance(e, dict) and isinstance(e.get(key), list):
                belegt |= {int(x) for x in e[key] if isinstance(x, (int, float))}
    return belegt


def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, n_train=N_TRAIN,
         n_test=N_TEST, **kw):
    gitter = LR_GITTER
    if schnell:
        seeds, tuning_seeds, schritte, n_train, n_test = ENTWURF_SEEDS, ENTWURF_TUNING, 40, 600, 500
        gitter = (0.02, 0.08, 0.32)
    else:
        if len(tuning_seeds) < 3:
            raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
        if set(seeds) & set(tuning_seeds):
            raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
        if (set(seeds) | set(tuning_seeds)) & _belegte_seeds():
            raise ValueError("Seeds in anderen Modulen/Ergebnissen belegt -- frische waehlen")
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    t0 = time.time()
    bud = (schritte, n_train, n_test)
    _roh("mnist")      # Daten EINMAL im Hauptprozess laden (sonst laden alle Arbeiter gleichzeitig -> kaputte Dateien)

    def auswerten(aufg):
        return parallel(_tuning_arbeit, aufg)

    # Phase 1: Raten voll, identitaet und alle klassischen Mittelarme
    info1, roh1 = gittersuche_stabil(
        RATEN_PHASE1, lambda arm, lr: [(arm, s, lr, bud, None) for s in tuning_seeds], auswerten,
        gitter=gitter, faktor=FAKTOR, kollaps=KOLLAPS)
    lr = {arm: v["lr"] for arm, v in info1.items()}
    print(f"  Phase 1 nach {time.time() - t0:.0f} s: " + ", ".join(f"{a} {lr[a]:g}" for a in RATEN_PHASE1), flush=True)

    # Phase 2: Raten der Clifford-Arme auf den Voll-Modellen der Tuning-Seeds (mit der Voll-Rate)
    zustaende = {s: e["zustand"] for s, e in roh1[("voll", lr["voll"])].items()}
    info2, _ = gittersuche_stabil(
        RATEN_PHASE2, lambda arm, r: [(arm, s, r, bud, zustaende[s]) for s in tuning_seeds], auswerten,
        gitter=gitter, faktor=FAKTOR, kollaps=KOLLAPS)
    info = {**info1, **info2}
    lr.update({arm: v["lr"] for arm, v in info2.items()})
    for arm, v in info.items():
        print(f"    {arm}: lr {v['lr']:g} (argmax min {v['argmax']:g}, Mittel {v['argmax_mittel']:g}, "
              f"abgestuft {v['abgestuft']}{', AM RAND' if v['am_rand'] else ''})", flush=True)
    print(f"  Phase 2 nach {time.time() - t0:.0f} s", flush=True)

    # Phase 3: Mess-Seeds (Zellen zuerst: laengste Aufgaben vorn)
    aufg = [(_mess_zelle, (s, lr, bud)) for s in seeds] + [(_mess_mitte, (s, lr, bud)) for s in seeds]
    erg = parallel(_dispatch, aufg)
    zeilen = {s: {} for s in seeds}
    for e in erg:
        zeilen[e["seed"]].update(e)
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + "  ".join(f"{a} {z[a]:.4f}" for a in ARME), flush=True)

    ergebnis = {"abteilung": "clifford", "modul": "ternaer", "agenda": "T-ternaer", "version": 1,
                "datensatz": "mnist", "qubits": N_QUBITS, "hypothesen": HYPOTHESEN, "lr": info,
                "schritte": schritte, "n_train": n_train, "n_test": n_test, "seeds": seeds,
                "tuning_seeds": tuning_seeds, "zeilen": zeilen, "sekunden": time.time() - t0,
                "kosten": {a: kosten(a) for a in ARME}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


def _dispatch(a):
    fn, arg = a
    return fn(arg)


# ------------------------------------------------------------------ Auswertung
def _holm(p: dict, alpha=ALPHA) -> dict:
    """Holm-Bonferroni ueber die Familie p {name: p-Wert} -> {name: verworfen}."""
    rest, verworfen = sorted(p, key=p.get), {}
    stop = False
    for i, name in enumerate(rest):
        if not stop and p[name] <= alpha / (len(rest) - i):
            verworfen[name] = True
        else:
            stop = True
            verworfen[name] = False
    return verworfen


def _vergleich(d: np.ndarray) -> dict:
    """Gepaarte Differenz d: Mittel, einseitiges 95-%-KI (t), einseitige p-Werte gegen -DELTA."""
    from scipy.stats import t as t_vert
    n = len(d)
    m = float(d.mean())
    se = max(float(d.std(ddof=1)) / math.sqrt(n), 1e-12)
    q = float(t_vert.ppf(1 - ALPHA, n - 1))
    tw = (m + DELTA) / se
    return {"mittel": m, "se": se, "ki_unten": m - q * se, "ki_oben": m + q * se,
            "p_ni": float(t_vert.sf(tw, n - 1)),      # H0': d <= -DELTA   (Nichtunterlegenheit)
            "p_inf": float(t_vert.cdf(tw, n - 1)),    # H0'': d >= -DELTA  (Unterlegenheit)
            "p_wilcoxon": wilcoxon_exakt(d)}


def _pareto(punkte: dict) -> list:
    """punkte {arm: (acc, mult, bit)} -> nichtdominierte Arme (hoehere acc, weniger mult, weniger bit)."""
    front = []
    for a, (acc, m, b) in punkte.items():
        dom = any(oa >= acc and om <= m and ob <= b and (oa > acc or om < m or ob < b)
                  for o, (oa, om, ob) in punkte.items() if o != a)
        if not dom:
            front.append(a)
    return front


def auswertung(zeilen) -> dict:
    a = lambda k: np.array([z[k] for z in zeilen], dtype=float)
    aus = {"genauigkeit": {arm: zusammenfassung(a(arm)) for arm in ARME}, "n": len(zeilen)}
    for name, (x, y) in {"V1": ("voll", "identitaet"), "V2": ("dicht_quad", "identitaet")}.items():
        d = a(x) - a(y)
        aus[name] = {"diff": float(d.mean()), "p": wilcoxon_exakt(d)}
    for name, arm in {"T1": "cliff0", "T2": "cliff4"}.items():
        aus[name] = _vergleich(a(arm) - a("ternaer_quad"))
    holm_ni = _holm({k: aus[k]["p_ni"] for k in ("T1", "T2")})
    holm_inf = _holm({k: aus[k]["p_inf"] for k in ("T1", "T2")})
    for k in ("T1", "T2"):
        aus[k]["holm_ni"], aus[k]["holm_inf"] = holm_ni[k], holm_inf[k]
    aus["kontraste"] = {f"{x}-{y}": float((a(x) - a(y)).mean()) for x, y in
                        [("cliff0", "ternaer"), ("cliff0", "voll"), ("ternaer_quad", "ternaer"),
                         ("ternaer_quad", "dicht_quad"), ("voll", "dicht_quad"), ("cliff4", "cliff0"),
                         ("cliff0", "ternaer8_quad"), ("ternaer8_quad", "ternaer8")]}
    k = {arm: kosten(arm) for arm in ARME}
    mw = {arm: aus["genauigkeit"][arm]["mittel"] for arm in ARME}
    aus["T3"] = {"pareto_mitte": _pareto({r: (mw[r], k[r]["mult_mitte"], k[r]["bit_mitte"]) for r in ARME}),
                 "pareto_gesamt": _pareto({r: (mw[r], k[r]["mult_gesamt"], k[r]["bit_gesamt"]) for r in ARME})}
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum. Alle Werte sind Strings."""
    u = {}
    v1 = a["V1"]["diff"] > 0 and a["V1"]["p"] < ALPHA
    v2 = a["V2"]["p"] < ALPHA
    u["V-1"] = (f"H1: voll schlaegt identitaet ({a['V1']['diff']:+.4f}, p = {a['V1']['p']:.4f})" if v1 else
                f"HU: voll trennt sich nicht von identitaet ({a['V1']['diff']:+.4f}, p = {a['V1']['p']:.4f}) -- "
                "T-1/T-2 ohne Urteil")
    u["V-2"] = (f"H1: dicht_quad trennt sich von identitaet ({a['V2']['diff']:+.4f}, p = {a['V2']['p']:.4f})" if v2 else
                f"HU: dicht_quad trennt sich nicht von identitaet ({a['V2']['diff']:+.4f}, "
                f"p = {a['V2']['p']:.4f}) -- T-1/T-2 ohne Urteil")
    for name, schl, arm in (("T-1", "T1", "cliff0"), ("T-2", "T2", "cliff4")):
        t = a[schl]
        zahlen = (f"{arm} - ternaer_quad = {t['mittel']:+.4f}, einseitiges 95-%-KI "
                  f"[{t['ki_unten']:+.4f}, {t['ki_oben']:+.4f}], Grenze {-DELTA:+.2f}, "
                  f"p_NI = {t['p_ni']:.4f}, p_Unterleg. = {t['p_inf']:.4f} (Holm ueber T-1/T-2)")
        if t["holm_ni"] and t["ki_unten"] > -DELTA:
            text = f"H1: {arm} nicht schlechter als ternaer_quad -- {zahlen}"
        elif t["holm_inf"] and t["ki_oben"] < -DELTA:
            text = f"H0: {arm} nachweislich um mehr als 1 pp schlechter als ternaer_quad -- {zahlen}"
        else:
            text = f"HU: kein Urteil -- {zahlen}"
        if not (v1 and v2):
            fehlt = "+".join(n for n, ok in (("V-1", v1), ("V-2", v2)) if not ok)
            text = f"HU (Vorbedingung {fehlt} nicht erfuellt): " + text
        u[name] = text
    kont = ", ".join(f"{k} {v:+.4f}" for k, v in a["kontraste"].items())
    u["T-3"] = ("beschreibend: Pareto-Front Mitte (Mult., Bit): " + ", ".join(a["T3"]["pareto_mitte"])
                + "; mit Kopf: " + ", ".join(a["T3"]["pareto_gesamt"]) + f"; Kontraste: {kont}")
    return {k: str(v) for k, v in u.items()}


# ------------------------------------------------------------------ Vortest
def _synth(rng, mu: dict, rausch=0.004, n=10, abw: dict | None = None) -> list:
    """Synthetische Zeilen. Rauschen je Arm auf Mittel 0 zentriert (exakte Mittel, deterministische
    KI). abw {arm: (m, sigma)}: Arm = ternaer_quad + m + zentriertes Rauschen mit Std sigma."""
    def zentriert(sd):
        e = rng.normal(0, 1, n)
        e -= e.mean()
        return e / e.std(ddof=1) * sd
    z = [{"seed": s} for s in range(n)]
    for arm in ARME:
        e = zentriert(rausch)
        for i in range(n):
            z[i][arm] = mu[arm] + e[i]
    for arm, (m, sigma) in (abw or {}).items():
        e = zentriert(sigma)
        for i in range(n):
            z[i][arm] = z[i]["ternaer_quad"] + m + e[i]
    return z


def _vortest_ste() -> bool:
    ok = True
    w = torch.tensor([[0.9, -0.8, 0.05, -0.02], [0.4, 0.0, -0.5, 0.6]], requires_grad=True)
    wq, g = ternarisiere(w)
    if abs(float(g.detach()) - float(w.detach().abs().mean())) > 1e-7:
        ok = False
        print("  VORTEST FEHLER STE: gamma != mean|W|")
    erlaubt = torch.stack([-g, torch.zeros(()), g])
    if not bool(((wq.detach()[..., None] - erlaubt).abs().min(-1).values < 1e-6).all()):
        ok = False
        print("  VORTEST FEHLER STE: Werte nicht in {-g, 0, g}")
    soll = torch.clamp(torch.round(w.detach() / g), -1, 1) * g
    if not torch.allclose(wq.detach(), soll):
        ok = False
        print("  VORTEST FEHLER STE: Rundung falsch")
    c = torch.arange(1.0, 9.0).reshape(2, 4)
    (wq * c).sum().backward()
    if w.grad is None or not torch.allclose(w.grad, c):
        ok = False
        print("  VORTEST FEHLER STE: Gradient fliesst nicht (Straight-Through)")
    # Modell: Gewichte der Schicht ternaer, Gradient erreicht Latentgewichte, gleicher Start je Seed
    m1, m2, m3 = (MittelModell(a, 3) for a in ("ternaer_quad", "ternaer", "dicht_quad"))
    m4, m5 = MittelModell("ternaer8_quad", 3), MittelModell("ternaer8", 3)
    zelle, _ = cl._baue("voll", 3, N_QUBITS)
    if not (m1.gewicht.shape == (D, D) and torch.equal(m1.gewicht, m2.gewicht) and torch.equal(m1.gewicht, m3.gewicht)
            and torch.equal(m1.kopf.weight, m2.kopf.weight) and torch.equal(m1.kopf.weight, m3.kopf.weight)
            and torch.equal(m1.kopf.weight, zelle.kopf.weight) and torch.equal(m1.kopf.bias, zelle.kopf.bias)
            and m1.kopf.weight.shape == (10, D)
            and m4.gewicht.shape == (8, D) and torch.equal(m4.gewicht, m5.gewicht)
            and torch.equal(m4.kopf.weight, m5.kopf.weight)):
        ok = False
        print("  VORTEST FEHLER: Startzustand/Form der Mittelarme (Kopf = Zellenkopf)")
    mb = m1
    x = torch.rand(5, D)
    loss = nn.functional.cross_entropy(mb(x), torch.arange(5))
    loss.backward()
    wq_m = mb.schicht().detach()
    if len(torch.unique(wq_m)) > 3 or mb.gewicht.grad.shape != (D, D) or not mb.gewicht.grad.abs().sum() > 0:
        ok = False
        print("  VORTEST FEHLER: Modell nicht ternaer oder kein Gradient")
    return ok


def _vortest_kosten() -> bool:
    ok = True

    def gleich(name, ist, soll):
        nonlocal ok
        if abs(ist - soll) > 1e-6:
            ok = False
            print(f"  VORTEST FEHLER Kosten {name}: {ist} statt {soll}")
    # Handbeispiel n = 8, d = 256, 32 Winkel: freie Drehung 2d = 512 Mult.
    gleich("voll mitte", kosten("voll")["mult_mitte"], 32 * 512 + 256)               # 16.640
    gleich("voll gesamt", kosten("voll")["mult_gesamt"], 19_200)
    gleich("voll == clifford.multiplikationen", kosten("voll")["mult_gesamt"], cl.multiplikationen(8))
    gleich("cliff0 mitte", kosten("cliff0")["mult_mitte"], 256)                      # nur Born
    gleich("cliff0 gesamt", kosten("cliff0")["mult_gesamt"], 2_816)
    gleich("cliff2 mitte", kosten("cliff2")["mult_mitte"], 2 * 512 + 256)            # 1.280
    gleich("cliff4 mitte", kosten("cliff4")["mult_mitte"], 4 * 512 + 256)            # 2.304
    gleich("cliff4 == clifford.multiplikationen", kosten("cliff4")["mult_gesamt"], cl.multiplikationen(8, magie=4))
    gleich("identitaet mitte", kosten("identitaet")["mult_mitte"], 256)
    gleich("identitaet adds", kosten("identitaet")["adds_mitte"], 0)
    gleich("cliff0 adds", kosten("cliff0")["adds_mitte"], 32 * 256)
    gleich("ternaer mitte", kosten("ternaer")["mult_mitte"], 0)
    gleich("ternaer_quad mitte", kosten("ternaer_quad")["mult_mitte"], 256)
    gleich("ternaer adds", kosten("ternaer")["adds_mitte"], 256 * 255)               # 65.280
    gleich("ternaer kopf", kosten("ternaer")["mult_kopf"], 2560)
    gleich("dicht_quad mitte", kosten("dicht_quad")["mult_mitte"], 65536 + 256)
    gleich("dicht_quad adds", kosten("dicht_quad")["adds_mitte"], 65_280)
    gleich("ternaer8 mitte", kosten("ternaer8")["mult_mitte"], 0)
    gleich("ternaer8_quad mitte", kosten("ternaer8_quad")["mult_mitte"], 8)
    gleich("ternaer8 adds", kosten("ternaer8")["adds_mitte"], 8 * 255)
    gleich("ternaer8 kopf", kosten("ternaer8")["mult_kopf"], 80)
    gleich("bit ternaer8", kosten("ternaer8")["bit_mitte"], 2048 * math.log2(3) + 32)
    gleich("bit voll", kosten("voll")["bit_mitte"], 32 * 32)
    gleich("bit cliff0", kosten("cliff0")["bit_mitte"], 64)
    gleich("bit cliff2", kosten("cliff2")["bit_mitte"], 30 * 2 + 2 * 32)
    gleich("bit cliff4", kosten("cliff4")["bit_mitte"], 28 * 2 + 4 * 32)
    gleich("bit ternaer", kosten("ternaer")["bit_mitte"], 65536 * math.log2(3) + 32)   # ~103.872
    gleich("bit ternaer_quad", kosten("ternaer_quad")["bit_mitte"], 65536 * math.log2(3) + 32)
    gleich("bit dicht_quad", kosten("dicht_quad")["bit_mitte"], 65536 * 32)
    gleich("bit identitaet", kosten("identitaet")["bit_mitte"], 0)
    gleich("norm", kosten("voll")["mult_norm"], 267)
    # Pareto: bekannte Dominanz
    p = _pareto({"a": (0.9, 10, 10), "b": (0.8, 20, 20), "c": (0.95, 30, 5), "d": (0.9, 10, 10)})
    if set(p) != {"a", "c", "d"}:
        ok = False
        print("  VORTEST FEHLER Pareto:", p)
    return ok


def vortest(ziehungen=3) -> bool:
    """Pflicht-Vortest: jeden Urteilszweig auf synthetischen Zeilen erzwingen, Kostenmodell am
    Handbeispiel, STE-Ternaerisierung, Seeds, Schnittstelle."""
    ok = _vortest_ste()
    ok = _vortest_kosten() and ok
    rng = np.random.default_rng(7)

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    basis = {"voll": 0.95, "cliff0": 0.90, "cliff2": 0.92, "cliff4": 0.93, "ternaer": 0.94,
             "ternaer_quad": 0.93, "dicht_quad": 0.94, "ternaer8_quad": 0.90, "ternaer8": 0.91,
             "identitaet": 0.88}
    mu = lambda **kw: {**basis, **kw}
    faelle = [   # (Bezeichnung, Zeilen-Argumente, erwartete Praefixe)
        ("H1/H1", dict(mu=mu(cliff0=0.93, cliff4=0.935)), {"T-1": "H1:", "T-2": "H1:", "V-1": "H1:", "V-2": "H1:"}),
        ("H0/H0", dict(mu=mu(cliff0=0.85, cliff4=0.88)), {"T-1": "H0:", "T-2": "H0:"}),
        ("H1/H0", dict(mu=mu(cliff0=0.93, cliff4=0.85)), {"T-1": "H1:", "T-2": "H0:"}),
        ("HU/H1 (Grenze)", dict(mu=mu(cliff0=0.92, cliff4=0.935)), {"T-1": "HU:", "T-2": "H1:"}),
        ("HU (Aufloesung)", dict(mu=mu(), abw={"cliff0": (0.0, 0.05), "cliff4": (0.0, 0.05)}),
         {"T-1": "HU:", "T-2": "HU:"}),
        ("V-2 Trennung nach unten", dict(mu=mu(dicht_quad=0.80)), {"V-1": "H1:", "V-2": "H1:"}),
        ("V-1 HU", dict(mu=mu(voll=0.88)), {"V-1": "HU:", "V-2": "H1:", "T-1": "HU (Vorbedingung V-1", "T-2": "HU (Vorbedingung V-1"}),
        ("V-2 HU", dict(mu=mu(dicht_quad=0.88)), {"V-1": "H1:", "V-2": "HU:", "T-1": "HU (Vorbedingung V-2", "T-2": "HU (Vorbedingung V-2"}),
        ("beide V HU", dict(mu=mu(voll=0.88, dicht_quad=0.88)),
         {"V-1": "HU:", "V-2": "HU:", "T-1": "HU (Vorbedingung V-1+V-2", "T-2": "HU (Vorbedingung V-1+V-2"}),
    ]
    for name, par, erwartet in faelle:
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth(rng, **par)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehler(f"{name}: {h} -> {u[h][:110]} (erwartet {praefix})")
    # Holm: beide einzeln knapp H1 (p_NI ~ 0,03 > 0,025), zusammen korrigiert HU
    sigma = 0.01
    m = -DELTA + 2.15 * sigma / math.sqrt(10)
    z = _synth(rng, mu(), abw={"cliff0": (m, sigma), "cliff4": (m, sigma)})
    a = auswertung(z)
    if not (0.025 < a["T1"]["p_ni"] < 0.05 and a["T1"]["ki_unten"] > -DELTA and a["T2"]["ki_unten"] > -DELTA):
        fehler(f"Holm-Aufbau: p_NI {a['T1']['p_ni']:.4f}, KI-Untergrenze {a['T1']['ki_unten']:.4f}")
    u = urteil(a)
    if not (u["T-1"].startswith("HU:") and u["T-2"].startswith("HU:")):
        fehler(f"Holm greift nicht: {u['T-1'][:80]}")
    # Holm greift nicht, wenn nur einer knapp ist und der andere klar: beide H1
    z = _synth(rng, mu(), abw={"cliff0": (m, sigma), "cliff4": (0.0, 0.004)})
    u = urteil(auswertung(z))
    if not (u["T-1"].startswith("H1:") and u["T-2"].startswith("H1:")):
        fehler(f"Holm zu streng: {u['T-1'][:80]} | {u['T-2'][:80]}")
    # Schnittstelle: Strings, JSON, Pareto-Lage, Kostenmodell im Ergebnis serialisierbar
    a = auswertung(_synth(rng, mu()))
    u = urteil(a)
    if not all(isinstance(v, str) for v in u.values()) or set(u) != {"V-1", "V-2", "T-1", "T-2", "T-3"}:
        fehler("Urteil nicht Frage -> String")
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN, "kosten": {r: kosten(r) for r in ARME}})
    except (TypeError, ValueError) as e:
        fehler(f"nicht JSON-serialisierbar: {e}")
    if set(HYPOTHESEN) != {"T-1", "T-2", "T-3", "V-1", "V-2"} or not all(isinstance(v, str) for v in HYPOTHESEN.values()):
        fehler("HYPOTHESEN")
    # Pareto-Erwartung: ternaer schlaegt ternaer_quad (gleiche Bits, weniger Mult.) -> quad nur bei besserer acc
    if "ternaer_quad" in _pareto({"ternaer": (0.94, 0, 3277), "ternaer_quad": (0.93, 8, 3277)}):
        fehler("Pareto: ternaer_quad haette dominiert sein muessen")
    # Seeds frisch, Schnittstelle prueft sie
    if set(SEEDS) & set(TUNING_SEEDS) or (set(SEEDS) | set(TUNING_SEEDS)) & _belegte_seeds() or len(SEEDS) != 10 \
            or len(TUNING_SEEDS) < 3:
        fehler("Seeds nicht frisch / zu wenige")
    for kw in ({"tuning_seeds": (960, 961)}, {"seeds": (310, 960)}, {"seeds": (120, 121)}):
        try:
            lauf(**kw)
            fehler(f"lauf nimmt unzulaessige Seeds an: {kw}")
        except ValueError:
            pass
        except Exception as e:      # noqa: BLE001
            fehler(f"lauf mit {kw}: {type(e).__name__} statt ValueError")
    # Struktur des Laufs: kanonische Ratensuche mit Stabilitaetsabstand
    import inspect
    q = inspect.getsource(lauf)
    if q.count("gittersuche_stabil(") != 2 or "kollaps=KOLLAPS" not in q:
        fehler("lauf nutzt nicht gittersuche_stabil fuer alle Arme")
    if sorted(RATEN_PHASE1 + RATEN_PHASE2) != sorted(ARME):
        fehler("Ratenschluessel decken nicht alle Arme ab")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("T-ternaer" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
