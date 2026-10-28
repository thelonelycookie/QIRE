"""Abteilung Clifford -- ternaere Butterfly gegen Clifford-Zelle (Agenda T-ternaer-butterfly, Quantenstufe Q2).

VORREGISTRIERUNG, 27.10.2026, VOR JEDER MESSZAHL (Plan: Opus, Umsetzung: Sonnet). Baut auf dem Modul
`ternaer.py` auf (Folgefrage nach T-ternaer, Bedingung T-1:H0); `ternaer.py` selbst bleibt unveraendert,
weil T-ternaer gerechnet und festgeschrieben ist. Wirkung klassisch (I-038): Q2 = exakte QM im Simulator,
Vergleich klassisch gerechnet; Formulierung "quanten-inspirierter Vorteil", nicht "Quantenvorteil".

FRAGE: T-ternaer fand cliff0 (Clifford-Zelle, 32 Winkel, 64 Bit) 4,4 pp unter ternaer_quad (ternaere
256x256-Schicht, 65.536 Gewichte). Liegt der Abstand an der Clifford-/Rotationsgeometrie oder an der
Parameterzahl/Struktur? Test: dieselbe Butterfly-Struktur wie die QIRE-Zelle (8 Qubits x 4 Lagen = 32
Bloecke, je Block eine 2x2-Matrix auf einer Qubit-Achse, danach CNOT-Ring), aber die Blockeintraege sind
ternaer {-1,0,+1}·gamma statt Ry-Winkel -- bei GLEICHER Blockzahl.

ARME (MNIST 16x16 = 8 Qubits = 256 Dim., Daten, Budget, Training und Kopf 256 -> 10 wie T-ternaer;
Eingabe psi = x/||x||; der Kopf-Startzustand ist je Seed IDENTISCH, der von clifford._baue):
  voll            QIRE-Zelle, freie Ry-Winkel (32 Parameter), Born-Auslese (Quadrat)   [Referenz]
  cliff0          voll trainiert, Winkel auf k·pi/2 gerundet, Kopf frisch nachtrainiert [Referenz]
  identitaet      Born-Auslese direkt auf x                                              [Untergrenze, V-1/V-2]
  ternaer_quad    ternaere 256x256 (BitNet b1.58, STE) + Quadrat + Kopf                  [Referenz aus T-ternaer]
  dicht_quad      Gleitkomma 256x256 + Quadrat + Kopf                                    [Kontrolle V-2]
  bfly_tern_quad  Butterfly-Struktur, 128 Eintraege (32 Bloecke x 4), ternaer nach BitNet (gamma = mean|W| ueber
                  alle 128 Eintraege, W_q = clip(round(W/gamma),-1,1)·gamma, STE), Quadrat -> PRIMAERER Arm.
                  Quadrat, weil cliff0/voll die Born-Auslese (Quadrat) haben: gleiche Auslese = gleiche Frage.
  bfly_tern       wie bfly_tern_quad ohne Quadrat (beschreibend)
  bfly_real_quad  Butterfly-Struktur, 128 reelle Eintraege in Gleitkomma OHNE Orthogonalitaetszwang,
                  Quadrat (Obergrenze der Struktur)
  bfly_real       wie bfly_real_quad ohne Quadrat (beschreibend)
Butterfly-Modell: U = Produkt ueber 4 Lagen von [CNOT-Ring-Permutation · kron(B[l,0], ..., B[l,7])], dazu
Ausgangsnormierung psi' / ||psi'|| und Merkmalsskala D·psi'^2 (linear: sqrt(D)·psi'), eine globale Konstante, im
Kostenmodell in den Kopf gefaltet (kostet nichts). Die Skala ist KEINE Schoenung: im Rauchtest auf dem Tuning-Seed
2694 (300 Schritte, 2.000 Bilder) lernte bfly_tern_quad ohne Skala nicht (Merkmale ~1/256, Kopf-Rate und Block-Rate
sind dieselbe Adam-Rate: zu kleine Rate -> Kopf lernt nicht, zu grosse -> Bloecke kippen zufaellig), mit Skala
bei lr 0.005 schon 83 % (Rauchtest verworfen, kein Mess-Seed). Die Zellenarme (voll, cliff0, identitaet) bleiben
wie in T-ternaer ohne Skala; ihre Rate wird je Arm getunt (kompensiert die Skala teilweise) -- Bedenken 2.
Zur Normierung: Bei der Zelle ist die Norm durch Unitaritaet 1; bei nicht-orthogonalen
und ternaeren Bloecken muss sie explizit sein (sie kuerzt auch gamma). Startwert der Latentgewichte: die
Ry-Bloecke [[c,-s],[s,c]] der Start-Winkel der Zelle (gleicher Start wie voll, nichts Zufaelliges, keine
Nullbloecke). Klassenbeziehung (im Vortest nachgewiesen): die ternaere Butterfly ENTHAELT die
Clifford-Zelle als Spezialfall (Ry(k·pi/2) ist bis auf einen globalen Skalar ein ternaerer Block; der
Skalar kuerzt sich in der Normierung). Gleiche Funktionsklasse + mehr, 128 Eintraege gegen 32 Winkel.

HYPOTHESEN (gepaarte Differenz je Seed, einseitiges 95-%-KI wie T-ternaer, Nichtunterlegenheitsgrenze
DELTA = 1 pp, Holm ueber {B-1, B-2} je Richtung eigene Familie, alpha = 0,05):
  B-1 (primaer) bfly_tern_quad nicht schlechter als cliff0 (d = bfly_tern_quad - cliff0 > -1 pp).
       H1: KI-Untergrenze > -DELTA und Holm. H0: KI-Obergrenze < -DELTA und Holm. Sonst HU.
       Lesart: H1 = Geometrie (Winkel gegen ternaere Eintraege) allein erklaert die Luecke nicht.
  B-2 bfly_tern_quad nicht schlechter als voll (gleiche Logik): kostet die Ternaerisierung in der Butterfly
       gegenueber freien Ry-Winkeln mehr als 1 pp?
  B-3 beschreibend: Pareto-Front Genauigkeit gegen (Multiplikationen, Bit) aller Arme (Mitte und mit Kopf),
       Kontraste (bfly_real_quad - bfly_tern_quad = Kosten der Ternaerisierung in der Struktur;
       bfly_real_quad - voll = Wert der Nicht-Orthogonalitaet; ternaer_quad - bfly_real_quad = Wert der
       dichten Struktur; bfly_tern_quad - bfly_tern = Quadrat), und die Lueckenanteile
       G_tern = (bfly_tern_quad - cliff0)/(ternaer_quad - cliff0)   (Anteil der T-ternaer-Luecke, der mit der
                                                                     Ternaerisierung der Eintraege schliesst)
       G_struktur = (ternaer_quad - bfly_real_quad)/(ternaer_quad - cliff0)  (Anteil, der auch mit freien
                                                                     reellen Eintraegen in Butterfly bleibt).
       Lesart (beschreibend, ohne Urteil): G_tern klein UND G_struktur gross = Luecke liegt an Parameterzahl/
       dichter Struktur, nicht an der Clifford-Geometrie; G_tern gross = Geometrie/Rotationszwang.
  V-1 voll schlaegt identitaet (exakter Wilcoxon p < 0,05, voll > identitaet).
  V-2 dicht_quad trennt sich von identitaet (exakter Wilcoxon zweiseitig p < 0,05).
  Sonst sind B-1/B-2 = HU (Vorbedingung nicht erfuellt), wie T-1/T-2.

EHRLICHE VORAB-ERWARTUNG (keine Befunde): bfly_tern_quad liegt zwischen cliff0 und ternaer_quad, am
ehesten bei voll oder leicht darueber (die Klasse enthaelt cliff0, hat 4x so viele Eintraege); B-1 eher H1,
B-2 eher HU oder H1. bfly_real_quad >= bfly_tern_quad. ternaer_quad bleibt vermutlich am genauesten
(65.536 Gewichte), d. h. der Hauptteil der T-ternaer-Luecke liegt an Parameterzahl und dichter Struktur
(G_struktur gross), nicht an der Clifford-Geometrie. Speicher: bfly_tern_quad ~235 Bit gegen cliff0 64 Bit
und ternaer_quad ~103.900 Bit; der Befund ist die Pareto-Lage.

KOSTEN/PARAMETER: Siehe kosten(). Zusatz gegenueber T-ternaer: Ausgangsnormierung der Butterfly-Arme
(256 Quadrate + 1 Kehrwertwurzel + 10 Skalierungen = 267 Mult., obere Schranke) steht in mult_mitte der
Butterfly-Arme (cliff0 braucht sie nicht, die Unitaritaet erledigt es). Parameter: mittlere Schicht ohne
Kopf (Kopf 2.570 bei allen Armen ausser dem Kopf-Vergleich identisch).

BEDENKEN (offen):
 1. Die Butterfly-Arme haben 128 Eintraege gegen 32 Winkel (gleiche Blockzahl, 4x so viele Zahlen); Parameterzahl
    ist je Arm exakt gezaehlt, aber nicht gleich. Gleich ist die Struktur (Blockzahl, Verdrahtung, Auslese).
 2. Merkmalsskala der Butterfly-Arme (siehe oben) und Ausgangsnormierung gelten nur dort; die Zellenarme sind
    unveraendert aus T-ternaer. Ein Rest-Vorteil durch die Skala kann nicht ausgeschlossen werden (Rate je Arm getunt).
 3. Ternaere Arme lernen STE-bewusst von Anfang an, cliff0 durch nachtrainiertes Runden (wie T-ternaer).
 4. Nur Gewichts-, keine Aktivierungsquantisierung (wie T-ternaer).

SEEDS (frisch, in seeds.json eingetragen, vom Vortest geprueft): Mess 2660-2669, Tuning 2694-2696 (auch
Rauchtest nur auf Tuning-Seeds). Lernrate je Arm ueber labor.gemeinsam.gittersuche_stabil (Faktor 2,
Stabilitaetsabstand, Gitter beidseitig verlaengert) auf den Tuning-Seeds, Validierung aus dem Trainingsteil;
cliff0 auf den Voll-Modellen der Tuning-Seeds (Phase 2). Alle Referenzarme werden mit demselben Protokoll NEU
gemessen (kein Wiederverwenden der T-ternaer-Zahlen).

WIEDERAUFNAHME: jede Tuning-Aufgabe und jede Mess-Aufgabe (Seed x Arm) geht sofort atomar in
ergebnisse/_zwischen_T-ternaer-butterfly.json; ein Neustart ueberspringt Fertiges. Zwischendatei mit anderer
Kennung (Seeds, Schritte ...) -> Fehler (nicht mischen). Prozesse ueber INSTITUT_PROZESSE (bis 72).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from abteilungen import clifford as cl
from abteilungen import ternaer as tn
from labor.gemeinsam import (ERGEBNISSE, genauigkeit, gittersuche_stabil, trainiere, wilcoxon_exakt,
                             zusammenfassung)
from labor.gemeinsam import _roh

ID = "T-ternaer-butterfly"
N_QUBITS, D, LAGEN = tn.N_QUBITS, tn.D, cl.LAGEN
N_BLOECKE = LAGEN * N_QUBITS            # 32
N_EINTRAEGE = N_BLOECKE * 4             # 128
DELTA, ALPHA = tn.DELTA, tn.ALPHA
SEEDS = tuple(range(2660, 2670))
TUNING_SEEDS = (2694, 2695, 2696)
SCHRITTE, N_TRAIN, N_TEST = tn.SCHRITTE, tn.N_TRAIN, tn.N_TEST
LR_GITTER, FAKTOR, KOLLAPS = tn.LR_GITTER, tn.FAKTOR, tn.KOLLAPS
BIT_TERNAER = math.log2(3)
KOPF_PARAM = D * 10 + 10                # 2.570
MERKMAL_SKALA = float(D)                # Butterfly-Arme: Kopf sieht D·psi'^2 (bzw. sqrt(D)·psi'), Merkmale O(1)

BFLY = ["bfly_tern_quad", "bfly_tern", "bfly_real_quad", "bfly_real"]
ARME = ["voll", "cliff0", "identitaet", "ternaer_quad", "dicht_quad"] + BFLY
RATEN_PHASE1 = [a for a in ARME if a != "cliff0"]
RATEN_PHASE2 = ["cliff0"]

HYPOTHESEN = {
    "B-1": "Primaer: bfly_tern_quad (ternaere Butterfly, Quadrat) ist nicht schlechter als cliff0 "
           "(Nichtunterlegenheit, Grenze 1 pp): einseitiges 95-%-KI der gepaarten Differenz > -0,01 -> H1; "
           "KI ganz unter -0,01 -> H0; sonst HU. Holm ueber {B-1, B-2}. Erwartung: eher H1 "
           "(die ternaere Butterfly enthaelt die Clifford-Zelle als Spezialfall).",
    "B-2": "bfly_tern_quad ist nicht schlechter als voll (gleiche Logik, Holm ueber {B-1, B-2}): Ternaerisierung der "
           "Butterfly-Eintraege kostet gegen freie Ry-Winkel hoechstens 1 pp. Erwartung: eher HU oder H1.",
    "B-3": "Beschreibend: Pareto-Front Genauigkeit gegen (Multiplikationen, Bit) aller Arme (Mitte und mit Kopf), "
           "Kontraste und Lueckenanteile G_tern, G_struktur. Erwartung: ternaer_quad am genauesten, "
           "Luecke liegt ueberwiegend an Parameterzahl/dichter Struktur, nicht an der Clifford-Geometrie.",
    "V-1": "Vorbedingung: voll schlaegt identitaet (exakter Wilcoxon p < 0,05). Sonst B-1/B-2 = HU.",
    "V-2": "Vorbedingung: dicht_quad trennt sich von identitaet (exakter Wilcoxon zweiseitig p < 0,05). "
           "Sonst B-1/B-2 = HU.",
}


# ------------------------------------------------------------------ Modell
class BflyModell(nn.Module):
    """psi = x/||x|| -> 4 Lagen [CNOT-Ring · kron(B[l,0..7])] -> psi/||psi|| -> [quadriert] -> Kopf 256 -> 10.
    B[l,q] 2x2, ternaer (STE, BitNet) oder reell. Start = Ry-Bloecke der Start-Winkel von cl._baue("voll", seed);
    der Kopf ist exakt der Startkopf der Zellenarme."""

    def __init__(self, art: str, seed: int):
        super().__init__()
        assert art in BFLY
        self.art, self.quad, self.tern = art, art.endswith("_quad"), art.startswith("bfly_tern")
        ref, _ = cl._baue("voll", seed, N_QUBITS)
        self.kopf = nn.Linear(D, 10)
        self.kopf.load_state_dict(ref.kopf.state_dict())
        th = ref.theta.detach()
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        blk = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)    # (L, n, 2, 2)
        self.block = nn.Parameter(blk.clone())
        self.register_buffer("perm", ref.perm.clone())

    def bloecke(self):
        return tn.ternarisiere(self.block)[0] if self.tern else self.block

    def unitaer(self):
        B = self.bloecke()
        U = torch.eye(D)
        for l in range(LAGEN):
            K = torch.ones(1, 1)
            for q in range(N_QUBITS):
                K = torch.kron(K, B[l, q])
            U = (K @ U)[self.perm]
        return U

    def forward(self, x):
        psi = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        psi = psi @ self.unitaer().T
        psi = psi / psi.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.kopf(MERKMAL_SKALA * psi ** 2 if self.quad else math.sqrt(D) * psi)


# ------------------------------------------------------------------ Kosten und Parameter
def kosten(arm: str) -> dict:
    """Kosten je Bild (Konventionen wie ternaer.kosten). Zusatz param_mitte (Eintraege/Gewichte der Mittelschicht
    ohne Kopf), param_kopf, param_diskret (davon diskret). Butterfly-Arme:
    Block auf 2^n Amplituden: ternaer 1 Addition je Amplitude und Block (±a ±b), 0 Mult. (gamma kuerzt die
    Normierung); reell 2 Mult. + 1 Add. je Amplitude (2d Mult. je Block, wie eine freie Drehung);
    + Quadrate (d) bei *_quad + Ausgangsnormierung 267 (in mult_mitte)."""
    if arm in BFLY:
        quad = D if arm.endswith("_quad") else 0
        tern = arm.startswith("bfly_tern")
        norm = D + 1 + 10
        k = {"mult_mitte": (0 if tern else N_BLOECKE * 2 * D) + quad + norm, "mult_quadrierung": quad,
             "mult_kopf": D * 10, "adds_mitte": N_BLOECKE * D, "adds_kopf": D * 10,
             "bit_mitte": N_EINTRAEGE * BIT_TERNAER + 32 if tern else float(N_EINTRAEGE * 32),
             "bit_kopf": float(KOPF_PARAM * 32), "mult_ausgangsnorm": norm,
             "param_mitte": N_EINTRAEGE, "param_diskret": N_EINTRAEGE if tern else 0}
        k["mult_norm"] = D + 1 + 10
        k["mult_gesamt"] = k["mult_mitte"] + k["mult_kopf"]
        k["bit_gesamt"] = k["bit_mitte"] + k["bit_kopf"]
        k["freie_winkel"] = None
    else:
        k = dict(tn.kosten(arm))
        k["param_mitte"] = {"voll": N_BLOECKE, "cliff0": N_BLOECKE, "identitaet": 0,
                            "ternaer_quad": D * D, "dicht_quad": D * D}[arm]
        k["param_diskret"] = {"voll": 0, "cliff0": N_BLOECKE, "identitaet": 0,
                              "ternaer_quad": D * D, "dicht_quad": 0}[arm]
    k["param_kopf"] = KOPF_PARAM
    return k


# ------------------------------------------------------------------ Arbeiter
def _modell(arm, s):
    if arm in BFLY:
        return BflyModell(arm, s)
    if arm in ("ternaer_quad", "dicht_quad"):
        return tn.MittelModell(arm, s)
    return tn._baue_zelle(arm, s)           # voll, identitaet


def _zustand(m) -> dict:
    return {"theta": m.theta.detach().flatten().tolist(), "kopf_w": m.kopf.weight.detach().flatten().tolist(),
            "kopf_b": m.kopf.bias.detach().tolist()}


def _zelle_aus(z) -> cl.QireZelle:
    m = cl.QireZelle(n=N_QUBITS)
    with torch.no_grad():
        m.theta.copy_(torch.tensor(z["theta"]).reshape(LAGEN, N_QUBITS))
        m.kopf.weight.copy_(torch.tensor(z["kopf_w"]).reshape(10, D))
        m.kopf.bias.copy_(torch.tensor(z["kopf_b"]))
    return m


def _nullanteil(m):
    with torch.no_grad():
        if isinstance(m, BflyModell):
            return float((m.bloecke() == 0).float().mean())
        return float((m.schicht() == 0).float().mean())


def _tuning_arbeit(a):
    """a = (arm, seed, lr, (schritte, n_train, n_test), zustand_voll | None); Seed an Stelle 1."""
    arm, s, lr, (schritte, n_train, n_test), zustand = a
    xtr, ytr, _, _ = tn._daten(s, n_train, n_test)
    n_val = len(ytr) // 4
    xt, yt, xv, yv = xtr[n_val:], ytr[n_val:], xtr[:n_val], ytr[:n_val]
    m = cl._gerundet(_zelle_aus(zustand), 0, s) if arm == "cliff0" else _modell(arm, s)
    trainiere(m, xt, yt, schritte, lr, seed=s)
    e = {"acc": genauigkeit(m(xv), yv)}
    if arm == "voll":
        e["zustand"] = _zustand(m)
    return e


def _mess_zelle(a):
    """Ein Seed: voll, cliff0 (aus demselben voll), identitaet."""
    s, lr, (schritte, n_train, n_test) = a
    xtr, ytr, xte, yte = tn._daten(s, n_train, n_test)
    z = {"seed": s}
    voll = tn._baue_zelle("voll", s)
    trainiere(voll, xtr, ytr, schritte, lr["voll"], seed=s)
    z["voll"] = genauigkeit(voll(xte), yte)
    th = voll.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    z["abstand_zum_gitter"] = float((th - cl.runde_clifford(th)).abs().mean())
    m = cl._gerundet(voll, 0, s)
    trainiere(m, xtr, ytr, schritte, lr["cliff0"], seed=s)
    z["cliff0"] = genauigkeit(m(xte), yte)
    m = tn._baue_zelle("identitaet", s)
    trainiere(m, xtr, ytr, schritte, lr["identitaet"], seed=s)
    z["identitaet"] = genauigkeit(m(xte), yte)
    return z


def _mess_arm(a):
    arm, s, lr, (schritte, n_train, n_test) = a
    xtr, ytr, xte, yte = tn._daten(s, n_train, n_test)
    m = _modell(arm, s)
    trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
    z = {"seed": s, arm: genauigkeit(m(xte), yte)}
    if arm == "ternaer_quad" or arm.startswith("bfly_tern"):
        z[f"{arm}_nullanteil"] = _nullanteil(m)
    return z


# ------------------------------------------------------------------ Wiederaufnahme
def _lade_zwischen(pfad: Path, kennung: dict) -> dict:
    if not pfad.exists():
        return {}
    inhalt = json.loads(pfad.read_text())
    if inhalt.get("kennung") != kennung:
        raise ValueError(f"Zwischendatei {pfad} stammt aus anderen Einstellungen ({inhalt.get('kennung')}) -- "
                         "nicht mischen; Datei sichern/umbenennen und neu starten")
    return inhalt["zeilen"]


def _schreibe_zwischen(pfad: Path, kennung: dict, zeilen: dict):
    """Atomar: .tmp vollstaendig schreiben, dann os.replace."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"kennung": kennung, "zeilen": zeilen}, ensure_ascii=False))
    os.replace(tmp, pfad)


def _aufgabe(a):
    """a = (schluessel, art, arg) -> (schluessel, ergebnis). Top-level (spawn)."""
    schl, art, arg = a
    t0 = time.time()
    e = {"lr": _tuning_arbeit, "zelle": _mess_zelle, "arm": _mess_arm}[art](arg)
    if art != "lr":
        e["sekunden"] = time.time() - t0
    return schl, e


def _rechne(aufgaben, pfad: Path, kennung: dict, prozesse=None, fn=None) -> dict:
    """aufgaben [(schluessel, art, arg)]; fertige Schluessel (aus pfad) werden uebersprungen, jedes Ergebnis
    geht sofort atomar in die Zwischendatei. Rueckgabe: {schluessel: ergebnis} aller Aufgaben."""
    import multiprocessing as mp
    from labor.gemeinsam import _init_arbeiter
    fn = fn or _aufgabe
    zeilen = _lade_zwischen(pfad, kennung)
    offen = [a for a in aufgaben if a[0] not in zeilen]
    if len(offen) < len(aufgaben):
        print(f"  Wiederaufnahme: {len(aufgaben) - len(offen)} von {len(aufgaben)} Aufgaben aus {pfad.name}", flush=True)
    prozesse = prozesse or int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))

    def ablegen(erg):
        schl, e = erg
        zeilen[schl] = e
        _schreibe_zwischen(pfad, kennung, zeilen)
        print(f"  fertig: {schl} ({len(zeilen)} in Zwischendatei)", flush=True)

    if prozesse <= 1 or len(offen) <= 1:
        _init_arbeiter()
        for a in offen:
            ablegen(fn(a))
    elif offen:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with mp.get_context("spawn").Pool(min(prozesse, len(offen)), initializer=_init_arbeiter) as pool:
            for erg in pool.imap_unordered(fn, offen):
                ablegen(erg)
    return {a[0]: zeilen[a[0]] for a in aufgaben}


# ------------------------------------------------------------------ Seeds
def _seeds_anderer() -> set:
    """Seeds anderer IDs aus seeds.json und aus ergebnisse/*.json (ohne ID, ohne Schnelllaeufe)."""
    belegt = set()
    reg = ERGEBNISSE.parent / "seeds.json"
    for e in json.loads(reg.read_text(encoding="utf-8")).get("eintraege", []):
        if e["id"].startswith(ID):
            continue
        for a, b in e["seeds"]:
            belegt |= set(range(a, b + 1))
    for f in ERGEBNISSE.glob("*.json"):
        if f.name.startswith(ID) or f.name.startswith("_") or "schnell" in f.name:
            continue
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for key in ("seeds", "tuning_seeds"):
            if isinstance(e, dict) and isinstance(e.get(key), list):
                belegt |= {int(x) for x in e[key] if isinstance(x, (int, float))}
    return belegt


def _pruefe_seeds(seeds, tuning_seeds):
    if len(tuning_seeds) < 3:
        raise ValueError("Ratenregel braucht >= 3 Tuning-Seeds")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    if (set(seeds) | set(tuning_seeds)) & _seeds_anderer():
        raise ValueError("Seeds in anderen IDs/Ergebnissen belegt -- frische waehlen")


# ------------------------------------------------------------------ Lauf
def lauf_butterfly(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, n_train=N_TRAIN,
                   n_test=N_TEST, zwischen=None, **kw):
    """schnell: Rauchtest NUR auf Tuning-Seeds (Mess = erster Tuning-Seed, Tuning = die uebrigen), 40 Schritte,
    Zwischendatei getrennt; Ergebnis wird verworfen."""
    gitter = LR_GITTER
    if schnell:
        seeds, tuning_seeds, schritte, n_train, n_test = (TUNING_SEEDS[0],), TUNING_SEEDS[1:], 40, 600, 500
        gitter = (0.02, 0.08, 0.32)
    else:
        _pruefe_seeds(seeds, tuning_seeds)
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    t0 = time.time()
    bud = (schritte, n_train, n_test)
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{ID}{'-schnell' if schnell else ''}.json"
    kennung = {"modul": "ternaer_butterfly", "version": 1, "seeds": seeds, "tuning_seeds": tuning_seeds,
               "schritte": schritte, "n_train": n_train, "n_test": n_test, "gitter": list(gitter),
               "faktor": FAKTOR, "kollaps": KOLLAPS}
    _roh("mnist")      # Daten EINMAL im Hauptprozess laden

    def auswerten(aufg):
        schl = [f"lr|{a[0]}|{a[2]!r}|{a[1]}" for a in aufg]
        erg = _rechne([(k, "lr", a) for k, a in zip(schl, aufg)], pfad, kennung)
        return [erg[k] for k in schl]

    info1, roh1 = gittersuche_stabil(
        RATEN_PHASE1, lambda arm, lr: [(arm, s, lr, bud, None) for s in tuning_seeds], auswerten,
        gitter=gitter, faktor=FAKTOR, kollaps=KOLLAPS)
    lr = {arm: v["lr"] for arm, v in info1.items()}
    print(f"  Phase 1 nach {time.time() - t0:.0f} s: " + ", ".join(f"{a} {lr[a]:g}" for a in RATEN_PHASE1), flush=True)
    zustaende = {s: e["zustand"] for s, e in roh1[("voll", lr["voll"])].items()}
    info2, _ = gittersuche_stabil(
        RATEN_PHASE2, lambda arm, r: [(arm, s, r, bud, zustaende[s]) for s in tuning_seeds], auswerten,
        gitter=gitter, faktor=FAKTOR, kollaps=KOLLAPS)
    info = {**info1, **info2}
    lr.update({arm: v["lr"] for arm, v in info2.items()})
    for arm, v in info.items():
        print(f"    {arm}: lr {v['lr']:g} (argmax min {v['argmax']:g}, Mittel {v['argmax_mittel']:g}, "
              f"abgestuft {v['abgestuft']}{', AM RAND' if v['am_rand'] else ''})", flush=True)
    t_tuning = time.time() - t0

    # Messung: Zellen und dichte Arme zuerst (laengste Aufgaben vorn)
    auf = [(f"mess|zelle|{s}", "zelle", (s, lr, bud)) for s in seeds]
    auf += [(f"mess|{arm}|{s}", "arm", (arm, s, lr, bud)) for arm in ["ternaer_quad", "dicht_quad"] + BFLY for s in seeds]
    erg = _rechne(auf, pfad, kennung)
    zeilen = {s: {} for s in seeds}
    for k, e in erg.items():
        zeilen[e["seed"]].update({kk: vv for kk, vv in e.items() if kk != "sekunden"})
        zeilen[e["seed"]].setdefault("sekunden_je_aufgabe", {})[k.split("|")[1]] = e["sekunden"]
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + "  ".join(f"{a} {z[a]:.4f}" for a in ARME), flush=True)

    ergebnis = {"abteilung": "clifford", "modul": "ternaer_butterfly", "agenda": ID, "version": 1,
                "datensatz": "mnist", "qubits": N_QUBITS, "hypothesen": HYPOTHESEN, "lr": info,
                "schritte": schritte, "n_train": n_train, "n_test": n_test, "seeds": seeds,
                "tuning_seeds": tuning_seeds, "zeilen": zeilen, "sekunden": time.time() - t0,
                "sekunden_tuning": t_tuning, "kosten": {a: kosten(a) for a in ARME}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


def lauf(**kw):
    return lauf_butterfly(**kw)


# ------------------------------------------------------------------ Auswertung
def auswertung(zeilen) -> dict:
    a = lambda k: np.array([z[k] for z in zeilen], dtype=float)
    aus = {"genauigkeit": {arm: zusammenfassung(a(arm)) for arm in ARME}, "n": len(zeilen)}
    for name, (x, y) in {"V1": ("voll", "identitaet"), "V2": ("dicht_quad", "identitaet")}.items():
        d = a(x) - a(y)
        aus[name] = {"diff": float(d.mean()), "p": wilcoxon_exakt(d)}
    for name, ref in {"B1": "cliff0", "B2": "voll"}.items():
        aus[name] = tn._vergleich(a("bfly_tern_quad") - a(ref))
    holm_ni = tn._holm({k: aus[k]["p_ni"] for k in ("B1", "B2")})
    holm_inf = tn._holm({k: aus[k]["p_inf"] for k in ("B1", "B2")})
    for k in ("B1", "B2"):
        aus[k]["holm_ni"], aus[k]["holm_inf"] = holm_ni[k], holm_inf[k]
    paare = [("bfly_tern_quad", "cliff0"), ("bfly_tern_quad", "voll"), ("bfly_tern_quad", "ternaer_quad"),
             ("bfly_tern_quad", "bfly_tern"), ("bfly_real_quad", "bfly_tern_quad"), ("bfly_real_quad", "voll"),
             ("ternaer_quad", "bfly_real_quad"), ("ternaer_quad", "cliff0"), ("voll", "cliff0"),
             ("bfly_real_quad", "bfly_real"), ("bfly_tern_quad", "identitaet")]
    aus["kontraste"] = {f"{x}-{y}": float((a(x) - a(y)).mean()) for x, y in paare}
    nenner = float((a("ternaer_quad") - a("cliff0")).mean())
    aus["luecke"] = {"ternaer_quad-cliff0": nenner,
                     "G_tern": float((a("bfly_tern_quad") - a("cliff0")).mean() / nenner) if nenner > 0.005 else None,
                     "G_struktur": float((a("ternaer_quad") - a("bfly_real_quad")).mean() / nenner)
                     if nenner > 0.005 else None}
    k = {arm: kosten(arm) for arm in ARME}
    mw = {arm: aus["genauigkeit"][arm]["mittel"] for arm in ARME}
    aus["B3"] = {"pareto_mitte": tn._pareto({r: (mw[r], k[r]["mult_mitte"], k[r]["bit_mitte"]) for r in ARME}),
                 "pareto_gesamt": tn._pareto({r: (mw[r], k[r]["mult_gesamt"], k[r]["bit_gesamt"]) for r in ARME})}
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum. Alle Werte sind Strings."""
    u = {}
    v1 = a["V1"]["diff"] > 0 and a["V1"]["p"] < ALPHA
    v2 = a["V2"]["p"] < ALPHA
    u["V-1"] = (f"H1: voll schlaegt identitaet ({a['V1']['diff']:+.4f}, p = {a['V1']['p']:.4f})" if v1 else
                f"HU: voll trennt sich nicht von identitaet ({a['V1']['diff']:+.4f}, p = {a['V1']['p']:.4f}) -- "
                "B-1/B-2 ohne Urteil")
    u["V-2"] = (f"H1: dicht_quad trennt sich von identitaet ({a['V2']['diff']:+.4f}, p = {a['V2']['p']:.4f})" if v2 else
                f"HU: dicht_quad trennt sich nicht von identitaet ({a['V2']['diff']:+.4f}, "
                f"p = {a['V2']['p']:.4f}) -- B-1/B-2 ohne Urteil")
    for name, schl, ref in (("B-1", "B1", "cliff0"), ("B-2", "B2", "voll")):
        t = a[schl]
        zahlen = (f"bfly_tern_quad - {ref} = {t['mittel']:+.4f}, einseitiges 95-%-KI "
                  f"[{t['ki_unten']:+.4f}, {t['ki_oben']:+.4f}], Grenze {-DELTA:+.2f}, "
                  f"p_NI = {t['p_ni']:.4f}, p_Unterleg. = {t['p_inf']:.4f} (Holm ueber B-1/B-2)")
        if t["holm_ni"] and t["ki_unten"] > -DELTA:
            text = f"H1: bfly_tern_quad nicht schlechter als {ref} -- {zahlen}"
        elif t["holm_inf"] and t["ki_oben"] < -DELTA:
            text = f"H0: bfly_tern_quad nachweislich um mehr als 1 pp schlechter als {ref} -- {zahlen}"
        else:
            text = f"HU: kein Urteil -- {zahlen}"
        if not (v1 and v2):
            fehlt = "+".join(n for n, ok in (("V-1", v1), ("V-2", v2)) if not ok)
            text = f"HU (Vorbedingung {fehlt} nicht erfuellt): " + text
        u[name] = text
    L = a["luecke"]
    g = lambda x: "n/a" if x is None else f"{x:.2f}"
    kont = ", ".join(f"{k} {v:+.4f}" for k, v in a["kontraste"].items())
    u["B-3"] = ("beschreibend: Pareto-Front Mitte (Mult., Bit): " + ", ".join(a["B3"]["pareto_mitte"])
                + "; mit Kopf: " + ", ".join(a["B3"]["pareto_gesamt"])
                + f"; Luecke ternaer_quad-cliff0 {L['ternaer_quad-cliff0']:+.4f}, G_tern {g(L['G_tern'])}, "
                  f"G_struktur {g(L['G_struktur'])}; Kontraste: {kont}")
    return {k: str(v) for k, v in u.items()}


# ------------------------------------------------------------------ Vortest
def _synth(rng, mu: dict, rausch=0.004, n=10, abw: dict | None = None) -> list:
    """Synthetische Zeilen mit exakten Armmitteln (Rauschen zentriert). abw {arm: (m, sigma)}:
    Arm = bfly_tern_quad + m + zentriertes Rauschen (Std sigma)."""
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
            z[i][arm] = z[i]["bfly_tern_quad"] + m + e[i]
    return z


_ZAEHLER = {"n": 0}


def _spielzeug(a):
    _ZAEHLER["n"] += 1
    return a[0], {"wert": a[2][2]}


def _vortest_modell() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)
    s = 3
    zelle, _ = cl._baue("voll", s, N_QUBITS)
    x = torch.rand(6, D)
    # 1. Form, Start: bfly_real_quad == Zelle (gleicher Start, Unitaritaet), gleicher Kopf
    ms = {a: BflyModell(a, s) for a in BFLY}
    r = ms["bfly_real_quad"]
    erw = zelle.merkmale(x) @ (MERKMAL_SKALA * zelle.kopf.weight).T + zelle.kopf.bias    # Zelle mit Skala im Kopf
    if r.block.shape != (LAGEN, N_QUBITS, 2, 2) or not torch.allclose(r(x), erw, atol=1e-3):
        fehler("bfly_real_quad zum Start nicht gleich der Zelle (bis auf Kopfskala)")
    if not all(torch.equal(m.kopf.weight, zelle.kopf.weight) and torch.equal(m.kopf.bias, zelle.kopf.bias)
               for m in ms.values()) or not all(torch.equal(ms[a].block, r.block) for a in BFLY):
        fehler("Start/Kopf der Butterfly-Arme nicht identisch mit der Zelle")
    U = r.unitaer()
    if not torch.allclose(U @ U.T, torch.eye(D), atol=1e-4):
        fehler("Start-Butterfly nicht orthogonal")
    # 2. Ternaerisierung: nur {-g, 0, g}, Gradient erreicht Latentgewichte, Form 32 Bloecke
    t = ms["bfly_tern_quad"]
    B = t.bloecke().detach()
    g = float(t.block.detach().abs().mean())
    if len(torch.unique(B)) > 3 or not bool(((B[..., None] - torch.tensor([-g, 0.0, g])).abs().min(-1).values < 1e-6).all()):
        fehler("bfly_tern nicht ternaer")
    nn.functional.cross_entropy(t(x), torch.arange(6)).backward()
    if t.block.grad is None or not t.block.grad.abs().sum() > 0 or t.block.numel() != N_EINTRAEGE:
        fehler("kein Gradient auf den Latentgewichten oder falsche Eintragszahl")
    if t(x).shape != (6, 10) or ms["bfly_tern"](x).shape != (6, 10):
        fehler("Ausgabeform")
    if len(torch.unique(ms["bfly_real_quad"].bloecke())) <= 3:
        fehler("bfly_real ist nicht reell/frei")
    # 3. Klassenbeziehung: ternaere Butterfly enthaelt Clifford-Zelle (Ry(k pi/2)-Bloecke) als Spezialfall
    th = zelle.theta.detach()
    for k in (0, 1):
        thk = cl.runde_clifford(th) if k == 0 else torch.round(th / (math.pi / 2)) * (math.pi / 2)
        c, sn = torch.cos(thk / 2), torch.sin(thk / 2)
        blk = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)
        bt = BflyModell("bfly_tern_quad", s)
        with torch.no_grad():
            bt.block.copy_(blk)
        zc = cl.QireZelle(n=N_QUBITS)
        zc.load_state_dict(zelle.state_dict())
        zc.fixiere(thk, torch.zeros(LAGEN, N_QUBITS))
        if not torch.allclose(bt(x), zc.merkmale(x) @ (MERKMAL_SKALA * zc.kopf.weight).T + zc.kopf.bias, atol=1e-3):
            fehler("ternaere Butterfly reproduziert die Clifford-Zelle nicht")
    # 4. Parameterzaehlung gegen echte Modelle
    zaehle = lambda m: sum(p.numel() for n_, p in m.named_parameters() if not n_.startswith("kopf"))
    for a in BFLY:
        kk = kosten(a)
        if zaehle(ms[a]) != kk["param_mitte"] or kk["param_mitte"] != 128:
            fehler(f"Parameterzahl {a}")
    for a, soll in (("ternaer_quad", D * D), ("dicht_quad", D * D)):
        if zaehle(tn.MittelModell(a, s)) != soll or kosten(a)["param_mitte"] != soll:
            fehler(f"Parameterzahl {a}")
    if zaehle(zelle) != N_BLOECKE or kosten("voll")["param_mitte"] != N_BLOECKE or kosten("cliff0")["param_diskret"] != 32:
        fehler("Parameterzahl Zelle")
    if sum(p.numel() for p in ms["bfly_tern"].kopf.parameters()) != KOPF_PARAM or any(kosten(a)["param_kopf"] != KOPF_PARAM for a in ARME):
        fehler("Kopfparameter")
    # Kosten/Bit exakt
    gl = lambda n_, i, so: (None if abs(i - so) < 1e-6 else fehler(f"Kosten {n_}: {i} statt {so}"))
    gl("bit tern", kosten("bfly_tern_quad")["bit_mitte"], 128 * BIT_TERNAER + 32)
    gl("bit tern lin", kosten("bfly_tern")["bit_mitte"], 128 * BIT_TERNAER + 32)
    gl("bit real", kosten("bfly_real_quad")["bit_mitte"], 128 * 32)
    gl("bit cliff0", kosten("cliff0")["bit_mitte"], 64)
    gl("bit voll", kosten("voll")["bit_mitte"], 1024)
    gl("mult tern_quad", kosten("bfly_tern_quad")["mult_mitte"], 256 + 267)
    gl("mult tern", kosten("bfly_tern")["mult_mitte"], 267)
    gl("mult real_quad", kosten("bfly_real_quad")["mult_mitte"], 32 * 512 + 256 + 267)
    gl("adds tern", kosten("bfly_tern_quad")["adds_mitte"], 32 * 256)
    gl("mult cliff0", kosten("cliff0")["mult_mitte"], 256)
    gl("mult ternaer_quad", kosten("ternaer_quad")["mult_mitte"], 256)
    gl("param ternaer_quad", kosten("ternaer_quad")["param_mitte"], 65536)
    gl("gesamt bit", kosten("bfly_tern_quad")["bit_gesamt"], 128 * BIT_TERNAER + 32 + KOPF_PARAM * 32)
    # 5. Minitraining Schnittstelle (3 Schritte, echte Daten, nur Tuning-Seed)
    try:
        for a in BFLY:
            e = _tuning_arbeit((a, TUNING_SEEDS[0], 0.05, (3, 400, 200), None))
            if not 0.0 <= e["acc"] <= 1.0:
                fehler(f"Minitraining {a}")
        ez = _tuning_arbeit(("voll", TUNING_SEEDS[0], 0.05, (3, 400, 200), None))
        ec = _tuning_arbeit(("cliff0", TUNING_SEEDS[0], 0.05, (3, 400, 200), ez["zustand"]))
        json.dumps(ez)
        if not 0.0 <= ec["acc"] <= 1.0:
            fehler("cliff0 aus JSON-Zustand")
        zv = _zelle_aus(ez["zustand"])
        if not torch.equal(torch.tensor(ez["zustand"]["theta"]).reshape(LAGEN, N_QUBITS), zv.theta.detach()):
            fehler("Zustand Rundreise")
    except FileNotFoundError:
        raise
    return ok


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    kennung = {"a": 1}
    with tempfile.TemporaryDirectory() as td:
        pfad = Path(td) / "z.json"
        auf = [(f"k{i}", "arm", (i, 0, i * i)) for i in range(5)]
        _ZAEHLER["n"] = 0
        e1 = _rechne(auf[:2], pfad, kennung, prozesse=1, fn=_spielzeug)         # simulierter Abbruch nach 2
        e2 = _rechne(auf, pfad, kennung, prozesse=1, fn=_spielzeug)
        if _ZAEHLER["n"] != 5 or {k: v["wert"] for k, v in e2.items()} != {f"k{i}": i * i for i in range(5)}:
            ok = False
            print("  VORTEST FEHLER: Wiederaufnahme rechnet Fertiges erneut oder verliert Ergebnisse", _ZAEHLER)
        _ZAEHLER["n"] = 0
        _rechne(auf, pfad, kennung, prozesse=1, fn=_spielzeug)
        if _ZAEHLER["n"] != 0:
            ok = False
            print("  VORTEST FEHLER: vollstaendige Zwischendatei wird neu gerechnet")
        try:
            _rechne(auf, pfad, {"a": 2}, prozesse=1, fn=_spielzeug)
            ok = False
            print("  VORTEST FEHLER: fremde Zwischendatei wird vermischt")
        except ValueError:
            pass
        if list(Path(td).glob("*.tmp")):
            ok = False
            print("  VORTEST FEHLER: .tmp-Datei bleibt liegen")
    return ok


def vortest(ziehungen=3) -> bool:
    """Pflicht-Vortest: Form, Ternaerisierung, Klassenbeziehung, Parameterzaehlung, alle Urteilszweige,
    Wiederaufnahme, Seeds frisch, Schnittstelle."""
    ok = _vortest_modell()
    ok = _vortest_wiederaufnahme() and ok
    rng = np.random.default_rng(11)

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    basis = {"voll": 0.94, "cliff0": 0.91, "identitaet": 0.90, "ternaer_quad": 0.955, "dicht_quad": 0.953,
             "bfly_tern_quad": 0.935, "bfly_tern": 0.90, "bfly_real_quad": 0.94, "bfly_real": 0.91}
    mu = lambda **kw: {**basis, **kw}
    faelle = [   # B-2 vergleicht mit voll, B-1 mit cliff0, bfly_tern_quad Basis 0,935
        ("H1/H1", mu(), {"B-1": "H1:", "B-2": "H1:", "V-1": "H1:", "V-2": "H1:"}),
        ("H0/H0", mu(cliff0=0.96, voll=0.96), {"B-1": "H0:", "B-2": "H0:"}),
        ("H1/H0", mu(cliff0=0.91, voll=0.96), {"B-1": "H1:", "B-2": "H0:"}),
        ("H0/H1", mu(cliff0=0.96, voll=0.935), {"B-1": "H0:", "B-2": "H1:"}),
        ("HU/H1 (Grenze)", mu(cliff0=0.945, voll=0.935), {"B-1": "HU:", "B-2": "H1:"}),
        ("V-1 HU", mu(voll=0.90), {"V-1": "HU:", "V-2": "H1:", "B-1": "HU (Vorbedingung V-1", "B-2": "HU (Vorbedingung V-1"}),
        ("V-2 HU", mu(dicht_quad=0.90), {"V-1": "H1:", "V-2": "HU:", "B-1": "HU (Vorbedingung V-2", "B-2": "HU (Vorbedingung V-2"}),
        ("beide V HU", mu(voll=0.90, dicht_quad=0.90),
         {"V-1": "HU:", "V-2": "HU:", "B-1": "HU (Vorbedingung V-1+V-2", "B-2": "HU (Vorbedingung V-1+V-2"}),
        ("V-2 Trennung nach unten", mu(dicht_quad=0.80), {"V-1": "H1:", "V-2": "H1:"}),
    ]
    for name, m_, erwartet in faelle:
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth(rng, m_)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehler(f"{name}: {h} -> {u[h][:110]} (erwartet {praefix})")
    # HU durch Aufloesung (grosse Streuung)
    u = urteil(auswertung(_synth(rng, mu(), abw={"cliff0": (0.0, 0.05), "voll": (0.0, 0.05)})))
    if not (u["B-1"].startswith("HU:") and u["B-2"].startswith("HU:")):
        fehler(f"HU durch Streuung: {u['B-1'][:60]}")
    # Holm: beide einzeln knapp (p_NI ~ 0,03 > 0,025), zusammen korrigiert HU; einer knapp + einer klar: beide H1
    sigma = 0.01
    m = -DELTA + 2.15 * sigma / math.sqrt(10)
    zz = _synth(rng, mu(), abw={"cliff0": (-m, sigma), "voll": (-m, sigma)})
    a = auswertung(zz)
    if not (0.025 < a["B1"]["p_ni"] < 0.05 and a["B1"]["ki_unten"] > -DELTA and a["B2"]["ki_unten"] > -DELTA):
        fehler(f"Holm-Aufbau: p_NI {a['B1']['p_ni']:.4f}, KI-Untergrenze {a['B1']['ki_unten']:.4f}")
    u = urteil(a)
    if not (u["B-1"].startswith("HU:") and u["B-2"].startswith("HU:")):
        fehler(f"Holm greift nicht: {u['B-1'][:80]}")
    zz = _synth(rng, mu(), abw={"cliff0": (-m, sigma), "voll": (0.0, 0.004)})
    u = urteil(auswertung(zz))
    if not (u["B-1"].startswith("H1:") and u["B-2"].startswith("H1:")):
        fehler(f"Holm zu streng: {u['B-1'][:80]} | {u['B-2'][:80]}")
    # Schnittstelle, Luecke, JSON
    a = auswertung(_synth(rng, mu()))
    u = urteil(a)
    if set(u) != {"V-1", "V-2", "B-1", "B-2", "B-3"} or not all(isinstance(v, str) for v in u.values()):
        fehler("Urteil nicht Frage -> String")
    soll_gt = (0.935 - 0.91) / (0.955 - 0.91)
    if abs(a["luecke"]["G_tern"] - soll_gt) > 1e-6 or abs(a["luecke"]["G_struktur"] - (0.955 - 0.94) / 0.045) > 1e-6:
        fehler(f"Lueckenanteile: {a['luecke']}")
    a2 = auswertung(_synth(rng, mu(cliff0=0.96)))
    if a2["luecke"]["G_tern"] is not None:
        fehler("Lueckenanteil bei Nenner <= 0 muss None sein")
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN, "kosten": {r: kosten(r) for r in ARME}})
    except (TypeError, ValueError) as e:
        fehler(f"nicht JSON-serialisierbar: {e}")
    if set(HYPOTHESEN) != {"B-1", "B-2", "B-3", "V-1", "V-2"}:
        fehler("HYPOTHESEN")
    if sorted(RATEN_PHASE1 + RATEN_PHASE2) != sorted(ARME):
        fehler("Ratenschluessel decken nicht alle Arme ab")
    # Seeds frisch (gegen seeds.json anderer IDs und Ergebnisdateien) und Registereintrag vorhanden
    try:
        _pruefe_seeds(SEEDS, TUNING_SEEDS)
    except ValueError as e:
        fehler(f"Seeds nicht frisch: {e}")
    if len(SEEDS) != 10 or len(TUNING_SEEDS) < 3:
        fehler("Seedanzahl")
    reg = json.loads((ERGEBNISSE.parent / "seeds.json").read_text(encoding="utf-8"))["eintraege"]
    eigene = {s for e in reg if e["id"] == ID and e["zweck"] == "messung" for a_, b_ in e["seeds"] for s in range(a_, b_ + 1)}
    tuning = {s for e in reg if e["id"] == ID and e["zweck"] == "tuning" for a_, b_ in e["seeds"] for s in range(a_, b_ + 1)}
    if eigene != set(SEEDS) or tuning != set(TUNING_SEEDS):
        fehler("Seeds nicht (oder abweichend) in seeds.json eingetragen")
    for kw in ({"tuning_seeds": (2694, 2695)}, {"seeds": (2660, 2694)}, {"seeds": (120, 121)}):
        try:
            lauf_butterfly(**kw)
            fehler(f"lauf nimmt unzulaessige Seeds an: {kw}")
        except ValueError:
            pass
        except Exception as e:      # noqa: BLE001
            fehler(f"lauf mit {kw}: {type(e).__name__} statt ValueError")
    import inspect
    q = inspect.getsource(lauf_butterfly)
    if q.count("gittersuche_stabil(") != 2 or "kollaps=KOLLAPS" not in q:
        fehler("lauf nutzt nicht gittersuche_stabil fuer alle Arme")
    if "TUNING_SEEDS[0]" not in q or "schnell" not in q:
        fehler("Rauchtest nicht auf Tuning-Seeds begrenzt")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf_butterfly(schnell=schnell)
    if schnell:
        print("Rauchtest (verworfen):", {a: round(z, 3) for a, z in e["auswertung"]["genauigkeit"].items()} if False else "ok")
    else:
        from labor.gemeinsam import speichere
        print(speichere(ID, e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
