"""Abteilung Clifford -- Magie-Skalierung v2 (Agenda C-skala-v2, Quantenstufe Q3).

FRAGE wie C-skala (abteilungen/skala.py): Waechst der Magiebedarf M50 LANGSAMER
als die Winkelzahl n*L -- und zwar auch dann, wenn Breite und Bildaufloesung
getrennt werden (Arm fest10 = 8x8-Information in 10 Qubits)? Hypothesen S-0..S-3,
Grenzen (DELTA = 0,08 G, alpha = 0,05), Referenz (nachtrainiert), Basis
(Identitaet), Pseudowerte (Delta-Methode), Holm-Familie {S-1, S-3} (zweiseitig)
und Entscheidungsbaum sind UNVERAENDERT aus skala.py uebernommen -- per import,
nicht kopiert. S-3 bleibt der saubere Test.

WIEDERVERWENDET aus skala.py (per import): SkalaZelle und Simulation, Arbeiter
_arbeit (voll / ohne_schaltung / rund mit k freien Magiewinkeln, frischer Kopf,
zweite Phase), Parallelisierung _parallel (laengste zuerst), Daten (echt,
fest = |x_8x8> (x) |+>^4), stufen(), auswertung(), urteil(), Statistik,
synthetische Daten _synth und der Struktur-Vortest. NEU ist nur: (1) die
Ratenwahl, (2) die Gruppierung der Magie-Arme, (3) die Instabilitaets- und
Kollapszaehlung im Urteil, (4) Seeds und Konfigurationen.

WARUM v2 (C-skala, 25.09., Rohdaten ergebnisse/C-skala__20260925_044543.json):
Die Ratensuche auf 2 Tuning-Seeds (argmax des Mittels) waehlte fuer
nachtrainiert ueberall 0,32; der naechste Gitterpunkt 1,28 kollabiert (0,53-0,57
bei echt8/fest10). Auf den Mess-Seeds kippte nachtrainiert bei echt8 in 2/10,
bei fest10 in 5/10 Seeds (bis 0,17) -> S-0 fest10 H0 -> S-3 „HU (Vorbedingung)“.
Dieselbe Kante traf auch andere Arme mit freien Winkeln: voll echt6 (0,32,
1,28 kollabiert auf 0,48) kippte in 2/10 Seeds (0,69; 0,86), der Magie-Arm fest10
(1,28, getunt bei k = 5) bei k5 in 1/10 und bei k20 in 3/10 Seeds (bis 0,09).

AENDERUNG DIREKTOR 25.09. NACH C-SKALA: RATENWAHL MIT STABILITAETSABSTAND
(vor jeder v2-Zahl). Fuer JEDEN Arm (Schluessel Konfiguration x Arm), gleiches
Gitter mit FAKTOR 2 statt 4 wie in C-skala (0,02; 0,04; 0,08; 0,16; 0,32, in beide
Richtungen um je Faktor 2 verlaengert, volles Budget auf dem Tuning-Split; Aenderung
Direktor 25.09. vor dem Lauf: „eine Stufe tiefer“ soll nur Faktor 2 kosten, damit
die Referenz nicht untertrainiert und G nicht aufblaeht):
  (1) >= 3 Tuning-Seeds (frisch: 930, 931, 932; lauf() verweigert weniger).
  (2) Kriterium = SCHLECHTESTER Tuning-Seed (min ueber Seeds), nicht das Mittel.
      Begruendung: Ein Kollaps ist ein Ereignis auf einzelnen Seeds; das Mittel
      verduennt ihn (1 von 2 Seeds bei 0,55 kostet im Mittel ~19 pp, bei 1 von 10
      Mess-Seeds aber das ganze S-0). Das Minimum bestraft genau die Rate, die auf
      irgendeinem Seed einbricht. Gleichstand -> kleinere Rate.
  (3) Stabilitaetsabstand: Eine Rate r ist nur waehlbar, wenn ihr oberer
      Gitternachbar 2r auf KEINEM Tuning-Seed kollabiert; Kollaps heisst
      acc(2r, s) < max_r' acc(r', s) - 5 pp (beste Rate DESSELBEN Seeds). Sonst
      eine Stufe tiefer (r/2; wiederholt, bis die Bedingung gilt; unter dem
      Gitterende wird das Gitter nach unten verlaengert). Liegt das Kriterium-
      Maximum am oberen Gitterrand, wird ohnehin nach oben verlaengert -- der
      obere Nachbar der gewaehlten Rate ist also immer auf allen Tuning-Seeds
      gemessen. Begruendung: Die Kollapsneigung steigt mit der Rate steil an
      (C-skala: 0,32 stabil auf den Tuning-Seeds, 1,28 kollabiert, 0,32 kippt
      auf 20-50 % der Mess-Seeds). Wer direkt unter der Kante waehlt, waehlt eine
      Rate, die auf einem Teil neuer Seeds schon jenseits der Kante liegt; mit 3
      Tuning-Seeds sieht man einen Kollaps mit Rate 20 % nur zu 49 %. Eine
      Gitterstufe Abstand zu einer auf den Tuning-Seeds kollabierenden Rate ist
      die billigste Sicherung. Mit Faktor 2 kostet sie wenig Genauigkeit (C-skala
      echt6 voll: 0,08 statt 0,32 = -1,8 pp bei Faktor 4); der Preis sind mehr
      Gitterpunkte (~6 statt ~4 je Arm, siehe LAUFZEIT).
  Die Regel gilt fuer alle Arme mit freien Winkeln (voll, nachtrainiert,
  Magie-Arme) und -- einheitlich -- auch fuer die Kopf-Arme (Identitaet, k0):
  dort ist das Problem konvex, ein Kollaps kommt nicht vor (C-skala: 1,28 kostet
  < 1 pp), die Regel reduziert sich auf das min-Kriterium. Eine Regel fuer alle
  vermeidet, dass Referenz und Basis nach verschiedenen Verfahren getunt werden.

MAGIE-ARME in zwei Ratengruppen (statt einer gemeinsamen Rate bei n*L/8 wie in
C-skala; fest10 k20 kippte dort mit der bei k5 getunten Rate in 3/10 Seeds):
  kklein  k in {1, 2, n*L/8}   getunt bei k = n*L/8
  kgross  k in {n*L/4, n*L/2}  getunt bei k = n*L/2
Jede Gruppe wird an ihrem GROESSTEN k getunt (meiste freie Winkel = instabilste
Stelle der Gruppe, Lehre C-referenz); fuer die kleineren k der Gruppe ist die
Rate damit eher zu vorsichtig als zu mutig. Weiterhin eine gemeinsame Rate je
Gruppe statt je k (Abweichung von Regel 1a aus Budgetgruenden, offen benannt).
nachtrainiert (k = n*L) und voll haben je eine eigene Rate wie in C-skala.

INSTABILITAET (im Urteil genannt, nicht entscheidungsrelevant -- der
Entscheidungsbaum bleibt der vorregistrierte aus skala.py):
  instabil  Seeds mit nachtrainiert < voll - 1 pp (wie C-skala, je Konfiguration;
            erfasst auch eine durch die vorsichtige Rate UNTERtrainierte
            Referenz, die den Nenner von G verkleinern und G aufblaehen wuerde).
  Kollaps   je Arm mit freien Winkeln (voll, nachtrainiert, k > 0): Seeds mit
            acc < Median ueber die Mess-Seeds dieses Arms - 5 pp.
  Robustheit (beschreibend): S-1/S-3 ohne die Seeds, die in einer der beiden
            Konfigurationen instabil sind oder kollabieren.
  Warnung   G(k) > 1,05 bei irgendeinem k (Referenz evtl. untertrainiert).

KONFIGURATIONEN: echt6, echt10, fest10 (fest6 = echt6). echt8 WEGGELASSEN:
S-1 (echt6 -> echt10) und S-3 (echt6 -> fest10) brauchen es nicht, es war in
C-skala nur Stuetzpunkt fuer den beschreibenden Fit (S-2) und Bruecke zu
C-mnist. Mit 3 Tuning-Seeds und 4 Ratenschluesseln mit freien Winkeln je
Konfiguration kostet echt8 ~0,8 Kern-h (+~12 min) und schoebe den Lauf ueber
das Ziel von ~110 min; der Fit hat dann 2 Punkte je Aufgabe (beschreibend).
Per lauf(breiten=(6, 8, 10)) zuschaltbar.

SEEDS: Mess 220-229 (frisch, nicht 120-129 aus C-skala), Tuning 930-932
(frisch), Rauchtest nur Entwurfs-Seeds 990/991 (Mess) und 992-994 (Tuning).
Budget wie C-skala: 2.500 Trainingsbilder, 500 Schritte, 5.000 Testbilder.

LAUFZEIT (geschaetzt, INSTITUT_PROZESSE=4; C-skala: ~100 s je volles Training
bei n = 10 und 4 Prozessen, Tuning-Split 3/4 davon): je n = 10-Konfiguration
Tuning 4 Schluessel x ~6 Raten (5 + eine Verlaengerung) x 3 Seeds = ~72
Trainings (~54 volle Aequivalente) + Messung 10 x 7 = 70 -> ~124 x 100 s =
3,4 Kern-h; echt10 + fest10 ~6,9 Kern-h, echt6 ~0,35, Kopf-Arme ~0,15 ->
~7,4 Kern-h / 4 = ~111 min, mit Phasengrenzen ~115-125 min; jede weitere
Verlaengerung eines n = 10-Schluessels +~1 min.

BEDENKEN (offen): Eine Stufe tiefer heisst Faktor 2 in der Rate; bei 500
Schritten kann ein Arm dadurch leicht untertrainiert sein (bei Faktor 4 kostete
das in C-skala echt6 voll 1,8 pp). Trifft das die Referenz staerker als die Magie-Arme,
wird G aufgeblaeht -- sichtbar als instabil (nachtrainiert < voll - 1 pp), als
d_nach < 0 und als Warnung G > 1,05. Trifft es echt6 und fest10 verschieden,
ist S-3 verzerrt; die Raten stehen deshalb im S-2-Text.
"""
from __future__ import annotations

import json
import time

import numpy as np

from abteilungen import skala as v1
from abteilungen.skala import (ALPHA, ANTEILE_ALLE, DELTA, INSTABIL_GRENZE, N_TEST, N_TRAIN,
                               NACH, PAAR_FEST, PAAR_HAUPT, SCHRITTE, _cfg, _cfg_zerlege, _k_anteil, stufen)

LAGEN = v1.LAGEN
BREITEN = (6, 10)                  # echt8 weggelassen (Begruendung im Kopf)
FEST_BREITEN = (10,)
SEEDS = tuple(range(220, 230))     # frisch (C-skala: 120-129)
TUNING_SEEDS = (930, 931, 932)     # frisch, >= 3 (C-skala: 920/921)
ENTWURF_SEEDS = (990, 991)         # nur Rauchtest
ENTWURF_TUNING = (992, 993, 994)   # nur Rauchtest
MIN_TUNING_SEEDS = 3
FAKTOR = 2                         # Gitterabstand (C-skala: 4)
LR_GITTER = (0.02, 0.04, 0.08, 0.16, 0.32)
KOLLAPS = 0.05                     # Kollaps = 5 pp unter der besten Rate desselben Seeds
WARN_G = 1.05
ANTEIL_KLEIN = 1 / 8               # Gruppe kklein getunt bei k = n*L/8
ANTEIL_GROSS = 1 / 2               # Gruppe kgross getunt bei k = n*L/2
RATEN_ARME_GRUND = ("voll", "ohne_schaltung")
RATEN_ARME_RUND = ("k0", "kklein", "kgross", NACH)
_ALT_SEEDS = set(v1.SEEDS) | set(v1.TUNING_SEEDS)

HYPOTHESEN = {
    **v1.HYPOTHESEN,
    "S-0": v1.HYPOTHESEN["S-0"] + " Instabile Seeds (nachtrainiert < voll - 1 pp) und Kollapse "
                                  "(< Median - 5 pp) je Arm werden genannt.",
    "S-2": v1.HYPOTHESEN["S-2"] + " Dazu die gewaehlten Raten (Stabilitaetsregel) und je Konfiguration "
                                  "die Zahl instabiler/kollabierter Seeds; S-1/S-3 ohne diese Seeds.",
    "S-3": v1.HYPOTHESEN["S-3"] + " (v2: alle Raten mit Stabilitaetsabstand gewaehlt, damit die "
                                  "Vorbedingung nicht an der Lernrate scheitert.)",
}
RATENREGEL = ("min ueber >= 3 Tuning-Seeds; waehlbar nur, wenn der obere Gitternachbar auf keinem "
              "Tuning-Seed kollabiert (acc < beste Rate desselben Seeds - 5 pp), sonst eine Stufe tiefer; "
              "Gitter 0,02/0,04/0,08/0,16/0,32 (Faktor 2) beidseitig verlaengert")


# ------------------------------------------------------------------ Ratenwahl
def waehle_rate(werte: dict, kollaps=KOLLAPS, notfall=False) -> dict:
    """werte {lr: [acc je Tuning-Seed]} (gleiche Seed-Reihenfolge je lr).
    Rueckgabe {'lr', 'erweitern' (None | 'unten' | 'oben'), 'argmax' (min-Kriterium),
    'argmax_mittel', 'abgestuft' (Stufen unter argmax), 'kollabiert' {lr: [Seed-Index]}}.
    Mit notfall=True (Suche ohne weitere Runden) wird nie erweitert: ein Maximum am
    oberen Rand gilt als 'oberer Nachbar ungeprueft' -> eine Stufe tiefer."""
    g = sorted(werte)
    n_s = {len(v) for v in werte.values()}
    if len(n_s) != 1:
        raise ValueError("ungleich viele Tuning-Seeds je Rate")
    n_s = n_s.pop()
    krit = {lr: float(min(werte[lr])) for lr in g}
    mittel = {lr: float(np.mean(werte[lr])) for lr in g}
    beste = [max(werte[lr][s] for lr in g) for s in range(n_s)]
    koll = {lr: [s for s in range(n_s) if werte[lr][s] < beste[s] - kollaps] for lr in g}
    i = max(range(len(g)), key=lambda j: (krit[g[j]], -j))       # Gleichstand -> kleinere Rate
    aus = {"argmax": g[i], "argmax_mittel": max(g, key=lambda lr: (mittel[lr], -lr)),
           "kriterium": {str(lr): krit[lr] for lr in g}, "mittel": {str(lr): mittel[lr] for lr in g},
           "kollabiert": {str(lr): koll[lr] for lr in g}, "erweitern": None, "lr": None,
           "abgestuft": 0, "am_rand": False}
    if not notfall:
        if i == 0:
            aus["erweitern"] = "unten"
            return aus
        if i == len(g) - 1:
            aus["erweitern"] = "oben"
            return aus
    j = i
    if j == len(g) - 1:          # nur im Notfall: oberer Nachbar nie gemessen
        j -= 1
        aus["am_rand"] = True
    while j >= 0 and koll[g[j + 1]]:
        j -= 1
    if j < 0:
        if not notfall:
            aus["erweitern"] = "unten"
            return aus
        j, aus["am_rand"] = 0, True
    if notfall and i == 0:
        aus["am_rand"] = True
    aus["lr"], aus["abgestuft"] = g[j], i - j
    return aus


def _gittersuche_stabil(schluessel, aufgaben_fuer, gitter=LR_GITTER, auswerten=None, runden=6):
    """Eigene Rate je Schluessel mit Stabilitaetsabstand (waehle_rate). aufgaben_fuer(schl, lr)
    -> Liste von Aufgaben, Seed an Stelle 1; auswerten(aufgaben) -> [{'acc', ...}] (Standard:
    skala._parallel). Alle offenen Schluessel einer Runde laufen parallel.
    Rueckgabe: info {schl: {...}}, roh {(schl, lr): {seed: Ergebnis}}."""
    auswerten = auswerten or v1._parallel
    gitter = {s: sorted(gitter) for s in schluessel}
    roh, info, offen = {}, {}, set(schluessel)

    def werte(s):
        return {lr: [roh[(s, lr)][sd]["acc"] for sd in sorted(roh[(s, lr)])] for lr in gitter[s]}

    def eintrag(s, w, runde):
        return {"lr": w["lr"], "argmax": w["argmax"], "argmax_mittel": w["argmax_mittel"],
                "abgestuft": w["abgestuft"], "am_rand": w["am_rand"], "runden": runde,
                "gitter": {str(lr): {"min": w["kriterium"][str(lr)], "mittel": w["mittel"][str(lr)],
                                     "je_seed": [roh[(s, lr)][sd]["acc"] for sd in sorted(roh[(s, lr)])],
                                     "kollabiert_seeds": [sorted(roh[(s, lr)])[i] for i in w["kollabiert"][str(lr)]]}
                           for lr in gitter[s]}}

    for runde in range(1, runden + 1):
        paare = [(s, lr) for s in sorted(offen) for lr in gitter[s] if (s, lr) not in roh]
        aufg, zu = [], []
        for s, lr in paare:
            for a in aufgaben_fuer(s, lr):
                aufg.append(a)
                zu.append((s, lr))
        for a, (s, lr), e in zip(aufg, zu, auswerten(aufg)):
            roh.setdefault((s, lr), {})[a[1]] = e
        for s in sorted(offen):
            w = waehle_rate(werte(s))
            if w["erweitern"] == "unten":
                gitter[s] = [gitter[s][0] / FAKTOR] + gitter[s]
            elif w["erweitern"] == "oben":
                gitter[s] = gitter[s] + [gitter[s][-1] * FAKTOR]
            else:
                info[s] = eintrag(s, w, runde)
                offen.discard(s)
        if not offen:
            break
    for s in sorted(offen):      # Notfall: Gitter erschoepft, nur mit gemessenen Punkten
        gitter[s] = [lr for lr in gitter[s] if (s, lr) in roh]
        info[s] = eintrag(s, waehle_rate(werte(s), notfall=True), runden)
    return info, roh


# ------------------------------------------------------------------ Arme
def _arm_fuer_k(n, k):
    """Ratengruppe der Stufe k bei Breite n: k0 | kklein (k <= n*L/8) | kgross (< n*L) | nachtrainiert."""
    nl = n * LAGEN
    if k == 0:
        return "k0"
    if k == nl:
        return NACH
    return "kklein" if k <= _k_anteil(n, ANTEIL_KLEIN) else "kgross"


def _k_tuning(n, arm):
    return {"k0": 0, "kklein": _k_anteil(n, ANTEIL_KLEIN), "kgross": _k_anteil(n, ANTEIL_GROSS),
            NACH: n * LAGEN}[arm]


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, n_train=N_TRAIN,
         n_test=N_TEST, datensatz="mnist", breiten=BREITEN, fest_breiten=FEST_BREITEN, **kw):
    if schnell:
        seeds, tuning_seeds, schritte, n_train, n_test = ENTWURF_SEEDS, ENTWURF_TUNING, 60, 400, 1000
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    if len(tuning_seeds) < MIN_TUNING_SEEDS:
        raise ValueError(f"Ratenregel v2 braucht >= {MIN_TUNING_SEEDS} Tuning-Seeds, nicht {tuning_seeds}")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Mess- und Tuning-Seeds ueberschneiden sich")
    if (set(seeds) | set(tuning_seeds)) & _ALT_SEEDS:
        raise ValueError("Seeds aus C-skala (120-129, 920/921) nicht wiederverwenden")
    t0 = time.time()
    budget = (schritte, n_train, n_test, datensatz)
    cfgs = [_cfg("echt", n) for n in breiten] + [_cfg("fest", n) for n in fest_breiten if n != 6]

    # Phase 1: Raten fuer voll und Identitaet (Tuning-Split, Stabilitaetsregel)
    def auf_grund(sch, lr):
        cfg, arm = sch
        return [(cfg, s, arm, lr, "tuning", 0, None, budget) for s in tuning_seeds]
    info1, roh1 = _gittersuche_stabil([(c, a) for c in cfgs for a in RATEN_ARME_GRUND], auf_grund)
    lr_voll = {c: info1[(c, "voll")]["lr"] for c in cfgs}
    print(f"  Phase 1 (Raten voll/Identitaet) nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{c} {lr_voll[c]:g}" for c in cfgs), flush=True)

    # Phase 2: Raten der Rundungsarme auf den Voll-Modellen der Tuning-Seeds (mit deren Rate)
    winkel_tuning = {c: {s: e["winkel"] for s, e in roh1[((c, "voll"), lr_voll[c])].items()} for c in cfgs}

    def auf_rund(sch, lr):
        cfg, arm = sch
        k = _k_tuning(_cfg_zerlege(cfg)[1], arm)
        return [(cfg, s, "rund", lr, "tuning", k, winkel_tuning[cfg][s], budget) for s in tuning_seeds]
    info2, _ = _gittersuche_stabil([(c, a) for c in cfgs for a in RATEN_ARME_RUND], auf_rund)
    info = {**info1, **info2}
    lr = {sch: v["lr"] for sch, v in info.items()}
    print(f"  Phase 2 (Raten Rundung) nach {time.time() - t0:.0f} s", flush=True)
    for (c, a), v in sorted(info.items()):
        print(f"    {c}/{a}: lr {v['lr']:g} (argmax min {v['argmax']:g}, argmax Mittel {v['argmax_mittel']:g}, "
              f"abgestuft {v['abgestuft']}{', AM RAND' if v['am_rand'] else ''})", flush=True)

    # Phase 3a: Mess-Seeds, voll und Identitaet
    aufg = [(c, s, arm, lr[(c, arm)], "mess", 0, None, budget)
            for c in cfgs for s in seeds for arm in RATEN_ARME_GRUND]
    zeilen = {c: {s: {"seed": s} for s in seeds} for c in cfgs}
    for (c, s, arm, *_), e in zip(aufg, v1._parallel(aufg)):
        zeilen[c][s][arm] = e["acc"]
        if arm == "voll":
            zeilen[c][s]["winkel_voll"] = e["winkel"]
            zeilen[c][s]["abstand_zum_gitter"] = e["abstand"]
    print(f"  Phase 3a (voll/Identitaet) nach {time.time() - t0:.0f} s", flush=True)

    # Phase 3b: Rundungsarme je Stufe + nachtrainiert (k = n*L), Rate je Gruppe
    aufg, namen = [], []
    for c in cfgs:
        n = _cfg_zerlege(c)[1]
        for s in seeds:
            for k in stufen(n) + [n * LAGEN]:
                aufg.append((c, s, "rund", lr[(c, _arm_fuer_k(n, k))], "mess", k,
                             zeilen[c][s]["winkel_voll"], budget))
                namen.append(NACH if k == n * LAGEN else f"k{k}")
    for (c, s, *_), name, e in zip(aufg, namen, v1._parallel(aufg)):
        zeilen[c][s][name] = e["acc"]
    zeilen = {c: [zeilen[c][s] for s in seeds] for c in cfgs}
    for c in cfgs:
        for z in zeilen[c]:
            print(f"  {c} Seed {z['seed']}: voll {z['voll']:.4f}  nach {z[NACH]:.4f}  ohne {z['ohne_schaltung']:.4f}  "
                  + "  ".join(f"k{k} {z[f'k{k}']:.4f}" for k in stufen(_cfg_zerlege(c)[1])), flush=True)

    raten = {f"{c}/{a}": v for (c, a), v in info.items()}
    ergebnis = {"abteilung": "clifford", "modul": "skala_v2", "agenda": "C-skala-v2", "version": 2,
                "datensatz": datensatz, "hypothesen": HYPOTHESEN, "ratenregel": RATENREGEL,
                "konfigurationen": cfgs, "lr": raten,
                "ratengruppen": {c: {f"k{k}": _arm_fuer_k(_cfg_zerlege(c)[1], k)
                                     for k in stufen(_cfg_zerlege(c)[1]) + [_cfg_zerlege(c)[1] * LAGEN]}
                                 for c in cfgs},
                "stufen": {c: stufen(_cfg_zerlege(c)[1]) for c in cfgs},
                "schritte": schritte, "n_train": n_train, "n_test": n_test,
                "seeds": seeds, "tuning_seeds": tuning_seeds, "zeilen": zeilen,
                "sekunden": time.time() - t0,
                "multiplikationen": {c: {f"k{k}": v1.multiplikationen(_cfg_zerlege(c)[1], magie=k)
                                         for k in stufen(_cfg_zerlege(c)[1])} | {
                    "voll": v1.multiplikationen(_cfg_zerlege(c)[1])} for c in cfgs}}
    ergebnis["auswertung"] = auswertung(zeilen, raten={k: v["lr"] for k, v in raten.items()},
                                        abgestuft={k: v["abgestuft"] for k, v in raten.items()})
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def _instabil(zz):
    return [z["seed"] for z in zz if z[NACH] < z["voll"] - INSTABIL_GRENZE]


def _kollapse(zz, n):
    """{Arm: [Seeds mit acc < Median(Arm) - 5 pp]} fuer alle Arme mit freien Winkeln."""
    aus = {}
    for arm in ["voll", NACH] + [f"k{k}" for k in stufen(n) if k > 0]:
        v = np.array([z[arm] for z in zz], dtype=float)
        aus[arm] = [z["seed"] for z, x in zip(zz, v) if x < float(np.median(v)) - KOLLAPS]
    return aus


def auswertung(zeilen: dict, raten=None, abgestuft=None) -> dict:
    """skala.auswertung (vorregistriert, unveraendert) + Instabilitaet, Kollapse,
    Robustheit (beschreibend) und die gewaehlten Raten."""
    a = v1.auswertung(zeilen)
    for c, zz in zeilen.items():
        n = _cfg_zerlege(c)[1]
        info = a["konfig"][c]
        koll = _kollapse(zz, n)
        info["instabil_liste"] = _instabil(zz)
        info["kollaps"] = {arm: len(s) for arm, s in koll.items()}
        info["kollaps_seeds"] = {arm: s for arm, s in koll.items() if s}
        gmax = [(k, v) for k, v in info["G"].items() if v is not None and v > WARN_G]
        info["warnung_G_ueber_1"] = [int(k) for k, _ in gmax]
    for name, (ca, cb) in (("S1", PAAR_HAUPT), ("S3", PAAR_FEST)):
        if name not in a:
            continue
        raus = set()
        for c in (ca, cb):
            raus |= set(a["konfig"][c]["instabil_liste"])
            for s in a["konfig"][c]["kollaps_seeds"].values():
                raus |= set(s)
        behalten = sorted({z["seed"] for z in zeilen[ca]} & {z["seed"] for z in zeilen[cb]} - raus)
        r = {"ausgeschlossen": sorted(raus), "n": len(behalten)}
        if raus and len(behalten) >= 2:
            sub = v1.auswertung({c: [z for z in zeilen[c] if z["seed"] in behalten] for c in (ca, cb)})[name]
            r.update({k: sub[k] for k in ("mittel", "p_rechts", "p_links", "ki_unten", "ki_oben", "vorbedingung")})
        a[name]["ohne_instabile"] = r
    a["raten"] = dict(raten or {})
    a["raten_abgestuft"] = dict(abgestuft or {})
    return a


def _instab_text(info):
    n = info["genauigkeit"]["voll"]["n"]
    k = ", ".join(f"{arm} {len(s)}" for arm, s in info["kollaps_seeds"].items()) or "keiner"
    return f"instabil {len(info['instabil_liste'])}/{n} (nachtrainiert < voll - 1 pp), Kollaps (< Median - 5 pp): {k}"


def urteil(a: dict) -> dict:
    """skala.urteil (vorregistrierter Entscheidungsbaum, Praefixe unveraendert) + Instabilitaet
    je Konfiguration in S-0, S-1, S-3 und die Raten in S-2. Alle Werte sind Strings."""
    u = v1.urteil(a)
    for c, info in a["konfig"].items():
        u[f"S-0 {c}"] += "; " + _instab_text(info)
    for name, key in (("S1", "S-1"), ("S3", "S-3")):
        if key not in u:
            continue
        v = a[name]
        teile = [f"{c} {len(a['konfig'][c]['instabil_liste'])}/{a['konfig'][c]['genauigkeit']['voll']['n']}"
                 for c in v["paar"]]
        txt = "; Instabilitaet (nachtrainiert < voll - 1 pp): " + ", ".join(teile)
        r = v.get("ohne_instabile", {})
        if not r.get("ausgeschlossen"):
            txt += "; keine instabilen/kollabierten Seeds"
        elif "mittel" in r:
            txt += (f"; beschreibend ohne Seeds {r['ausgeschlossen']}: d = {r['mittel']:+.3f} G "
                    f"(KI {r['ki_unten']:+.3f} .. {r['ki_oben']:+.3f}, p_rechts = {r['p_rechts']:.4f}, "
                    f"n = {r['n']})")
        else:
            txt += f"; ohne Seeds {r['ausgeschlossen']} bleiben {r['n']} -- keine Robustheitspruefung"
        u[key] += txt
    zus = []
    for c, info in a["konfig"].items():
        r = ", ".join(f"{arm} {a['raten'][f'{c}/{arm}']:g}"
                      + (f" (-{a['raten_abgestuft'][f'{c}/{arm}']} Stufe)" if a.get("raten_abgestuft", {}).get(f"{c}/{arm}") else "")
                      for arm in RATEN_ARME_GRUND + RATEN_ARME_RUND if f"{c}/{arm}" in a.get("raten", {}))
        w = f", WARNUNG G > {WARN_G} bei k = {info['warnung_G_ueber_1']}" if info["warnung_G_ueber_1"] else ""
        zus.append(f"{c}: {_instab_text(info)}{w}" + (f"; Raten {r}" if r else ""))
    u["S-2"] += " | v2: " + "; ".join(zus)
    return {k: str(v) for k, v in u.items()}


# ------------------------------------------------------------------ Vortest
def _vortest_ratenwahl() -> bool:
    ok = True

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER (Ratenwahl):", text)

    # a) argmax an der Kante: 0,32 bestes min, 0,64 kollabiert auf Seed 1 -> 0,16
    kante = {0.02: [0.85, 0.86, 0.85], 0.04: [0.87, 0.88, 0.87], 0.08: [0.89, 0.90, 0.89],
             0.16: [0.91, 0.905, 0.91], 0.32: [0.92, 0.915, 0.92], 0.64: [0.93, 0.55, 0.92]}
    w = waehle_rate(kante)
    if not (w["erweitern"] is None and w["argmax"] == 0.32 and w["lr"] == 0.16 and w["abgestuft"] == 1
            and w["kollabiert"]["0.64"] == [1]):
        fehler(f"Kante: erwartet 0,16 (eine Stufe unter 0,32), erhalten {w}")
    # b) oberer Nachbar stabil -> argmax bleibt
    w = waehle_rate({0.08: [0.85] * 3, 0.16: [0.89] * 3, 0.32: [0.92, 0.91, 0.92], 0.64: [0.90, 0.89, 0.90]})
    if not (w["lr"] == 0.32 and w["abgestuft"] == 0):
        fehler(f"stabiler Nachbar: erwartet 0,32, erhalten {w['lr']}")
    # c) min statt Mittel: Mittel waehlte 0,64, schlechtester Seed spricht fuer 0,32
    w = waehle_rate({0.16: [0.89] * 3, 0.32: [0.92, 0.91, 0.92], 0.64: [0.95, 0.88, 0.95], 1.28: [0.90] * 3})
    if not (w["argmax_mittel"] == 0.64 and w["argmax"] == 0.32 and w["lr"] == 0.32):
        fehler(f"min-Kriterium: erwartet 0,32 (Mittel 0,64), erhalten {w}")
    # d) Raender -> erweitern
    if waehle_rate({0.02: [0.9] * 3, 0.08: [0.8] * 3, 0.32: [0.7] * 3})["erweitern"] != "unten":
        fehler("Maximum am unteren Rand erweitert nicht nach unten")
    if waehle_rate({0.02: [0.7] * 3, 0.08: [0.8] * 3, 0.32: [0.9] * 3})["erweitern"] != "oben":
        fehler("Maximum am oberen Rand erweitert nicht nach oben")
    # e) mehrfach abstufen bis unter das Gitter -> nach unten erweitern
    w = waehle_rate({0.16: [0.96, 0.50, 0.50], 0.32: [0.90] * 3, 0.64: [0.40] * 3, 1.28: [0.30] * 3})
    if w["erweitern"] != "unten":
        fehler(f"Kaskade der Abstufung: erwartet 'unten', erhalten {w}")
    # f) Notfall (Gitter erschoepft, Maximum oben): oberer Nachbar ungeprueft -> eine Stufe tiefer
    w = waehle_rate({0.16: [0.8] * 3, 0.32: [0.85] * 3, 0.64: [0.9] * 3}, notfall=True)
    if not (w["lr"] == 0.32 and w["am_rand"]):
        fehler(f"Notfall am oberen Rand: erwartet 0,32 + am_rand, erhalten {w}")
    # g) ganze Suche mit synthetischem Gitter (Seed-Reihenfolge, Verlaengerung, Kante)
    tseeds = (930, 931, 932)

    def landschaft(sch, s, lr):
        l2 = np.log2(lr)
        if sch == "kante":      # Optimum 0,32, ab 0,64 Kollaps auf Seed 931
            if lr >= 0.64 and s == 931:
                return 0.30
            return 0.93 - 0.01 * (l2 - np.log2(0.32)) ** 2
        if sch == "hoch":       # Optimum 1,28, stabil
            return 0.93 - 0.01 * (l2 - np.log2(1.28)) ** 2 + 0.001 * (s - 931)
        return 0.93 - 0.01 * (l2 - np.log2(0.005)) ** 2          # "tief": Optimum 0,005

    aufrufe = []

    def auswerten(aufg):
        aufrufe.append(len(aufg))
        return [{"acc": float(landschaft(sch, s, lr))} for sch, s, lr in aufg]

    info, roh = _gittersuche_stabil(["kante", "hoch", "tief"],
                                    lambda sch, lr: [(sch, s, lr) for s in tseeds], auswerten=auswerten)
    soll = {"kante": 0.16, "hoch": 1.28, "tief": 0.005}
    for sch, lr_soll in soll.items():
        if abs(info[sch]["lr"] - lr_soll) > 1e-12 or info[sch]["am_rand"]:
            fehler(f"Suche {sch}: erwartet {lr_soll}, erhalten {info[sch]['lr']} ({info[sch]})")
        oben = info[sch]["lr"] * FAKTOR
        if not any(abs(oben - lr) < 1e-12 and len(roh[(sch, lr)]) == 3 for (s2, lr) in roh if s2 == sch):
            fehler(f"Suche {sch}: oberer Nachbar nicht auf allen Tuning-Seeds gemessen")
    if info["kante"]["abgestuft"] != 1 or info["kante"]["gitter"]["0.64"]["kollabiert_seeds"] != [931] \
            or "0.04" not in info["kante"]["gitter"] or "0.01" not in info["tief"]["gitter"]:
        fehler(f"Suche kante: Abstufung/Kollaps-Seed falsch {info['kante']}")
    json.dumps(info)
    g = sorted(LR_GITTER)
    if FAKTOR != 2 or any(abs(y / x - FAKTOR) > 1e-9 for x, y in zip(g, g[1:])) or 0.32 not in g \
            or _gittersuche_stabil.__defaults__[0] != LR_GITTER:
        fehler(f"Gitter nicht Faktor 2: {LR_GITTER}")
    # h) Schnittstelle: >= 3 Tuning-Seeds, frische Seeds, Gruppen der Magie-Arme
    for kw in ({"tuning_seeds": (930, 931)}, {"seeds": v1.SEEDS}, {"tuning_seeds": (920, 931, 932)},
               {"seeds": (930, 1), "tuning_seeds": (930, 931, 932)}):
        try:        # breiten leer: falls die Pruefung versagt, wird trotzdem nichts gerechnet
            lauf(breiten=(), fest_breiten=(), **kw)
            fehler(f"lauf nimmt unzulaessige Seeds an: {kw}")
        except ValueError:
            pass
        except Exception as e:      # noqa: BLE001
            fehler(f"lauf mit {kw}: {type(e).__name__} statt ValueError")
    if set(SEEDS) & _ALT_SEEDS or set(TUNING_SEEDS) & _ALT_SEEDS or set(SEEDS) & set(TUNING_SEEDS) \
            or len(TUNING_SEEDS) < MIN_TUNING_SEEDS or len(SEEDS) != 10:
        fehler("Seeds nicht frisch / zu wenige")
    gruppen = {n: {k: _arm_fuer_k(n, k) for k in stufen(n) + [n * LAGEN]} for n in (6, 10)}
    if gruppen[6] != {0: "k0", 1: "kklein", 2: "kklein", 3: "kklein", 6: "kgross", 12: "kgross", 24: NACH} or \
            gruppen[10] != {0: "k0", 1: "kklein", 2: "kklein", 5: "kklein", 10: "kgross", 20: "kgross", 40: NACH}:
        fehler(f"Ratengruppen falsch: {gruppen}")
    if _k_tuning(10, "kgross") != 20 or _k_tuning(6, "kklein") != 3 or _k_tuning(10, NACH) != 40:
        fehler("Tuning-k der Gruppen falsch")
    import inspect
    q = inspect.getsource(lauf)
    if q.count("_gittersuche_stabil(") != 2 or "v1._gittersuche(" in q or "_arm_fuer_k(n, k)" not in q:
        fehler("lauf nutzt nicht fuer alle Arme die Stabilitaetsregel")
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Struktur (skala), neue Ratenwahl, jeden Urteilszweig synthetisch."""
    ok = v1._vortest_struktur()
    ok = _vortest_ratenwahl() and ok
    rng = np.random.default_rng(25)
    drei = lambda f6, f10, ff10: {"echt6": f6, "echt10": f10, "fest10": ff10}
    konst = v1._linear_bis(2)
    prop6, prop10 = v1._linear_bis(3), v1._linear_bis(5)

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    faelle = [
        (dict(g_kurven=drei(konst, konst, konst)), {"S-1": "H1:", "S-3": "H1:", "S-0 fest10": "H1:"}, 0),
        (dict(g_kurven=drei(prop6, v1._linear_bis(10), v1._linear_bis(10))),
         {"S-1": "H0 (niedriger):", "S-3": "H0 (niedriger):"}, 0),
        (dict(g_kurven=drei(prop6, prop10, prop10), sd=0.001), {"S-1": "H0 (gleich):", "S-3": "H0 (gleich):"}, 1),
        (dict(g_kurven=drei(prop6, prop10, prop10), sd_seed=0.02), {"S-1": "HU:", "S-3": "HU:"}, 2),
        (dict(g_kurven=drei(konst, konst, konst), gewinne={"echt10": -0.003}),
         {"S-0 echt10": "H0:", "S-0 echt6": "H1:", "S-1": "HU (Vorbedingung):", "S-3": "H1:"}, 0),
        (dict(g_kurven=drei(konst, konst, konst), gewinne={"fest10": -0.003}),
         {"S-0 fest10": "H0:", "S-3": "HU (Vorbedingung):", "S-1": "H1:"}, 0),
    ]
    for par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            u = urteil(auswertung(v1._synth(rng, **par)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest: {h} -> {u[h][:120]}")
        for h, f in fehl.items():
            if f > toleranz:
                fehler(f"{h}: {f} von {ziehungen} falsch (erwartet {erwartet[h]})")
    # Holm ueber {S-1, S-3}: S-1 knapp, S-3 klar nicht -> S-1 HU
    a = auswertung(v1._synth(rng, drei(prop6, prop10, prop10), sd=0.0005))
    a["S1"].update(mittel=0.05, p_rechts=0.03, p_links=0.97, ki_unten=0.001, ki_oben=0.2)
    a["S3"].update(mittel=0.01, p_rechts=0.40, p_links=0.60, ki_unten=-0.2, ki_oben=0.2)
    for k, v in v1._holm({"S1": 0.03, "S3": 0.40}).items():
        a[k]["holm_signifikant"] = v
    if not urteil(a)["S-1"].startswith("HU:"):
        fehler("Holm-Korrektur greift nicht")
    a = auswertung(v1._synth(rng, drei(konst, konst, konst)))
    if set(p for p in ("S1", "S3") if a[p]["p_richtung"] is not None) != {"S1", "S3"}:
        fehler("Holm-Familie nicht {S-1, S-3}")
    # Instabilitaet wie in C-skala (fest10 kippt auf 5/10 Seeds) -> S-0 fest10 H0, S-3 HU (Vorbedingung)
    # mit Zaehlung im Urteil; S-1 unberuehrt.
    zz = v1._synth(rng, drei(konst, konst, konst))
    for z, wert in zip(zz["fest10"][:5], (0.78, 0.17, 0.48, 0.81, 0.30)):
        z[NACH] = wert
    for z in zz["fest10"][:2]:
        z["k20"] = 0.09
    a = auswertung(zz, raten={"fest10/nachtrainiert": 0.08, "fest10/voll": 0.08},
                   abgestuft={"fest10/nachtrainiert": 1})
    u = urteil(a)
    if not (u["S-0 fest10"].startswith("H0:") and "instabil 5/10" in u["S-0 fest10"]
            and u["S-3"].startswith("HU (Vorbedingung):") and "fest10 5/10" in u["S-3"]
            and u["S-1"].startswith("H1:") and "echt10 0/10" in u["S-1"]
            and "keine instabilen" in u["S-1"]):
        fehler(f"Instabilitaet nicht im Urteil: {u['S-0 fest10'][-120:]} | {u['S-3'][-120:]}")
    if a["konfig"]["fest10"]["kollaps"]["k20"] != 2 or a["konfig"]["fest10"]["kollaps"][NACH] < 3 \
            or a["konfig"]["echt6"]["kollaps"]["voll"] != 0:
        fehler(f"Kollapszaehlung falsch: {a['konfig']['fest10']['kollaps']}")
    if "nachtrainiert 0.08 (-1 Stufe)" not in u["S-2"]:
        fehler(f"Raten nicht im S-2-Text: {u['S-2'][-200:]}")
    # Robustheit: 3 instabile Seeds in echt10 -> S-1 bleibt H1, beschreibend ohne 3 Seeds (n = 7)
    zz = v1._synth(rng, drei(konst, konst, konst))
    for z in zz["echt10"][:3]:
        z[NACH] = z["voll"] - 0.02
    a = auswertung(zz)
    u = urteil(a)
    if a["S1"]["ohne_instabile"]["n"] != 7 or "ohne Seeds [0, 1, 2]" not in u["S-1"] \
            or a["S3"]["ohne_instabile"]["n"] != 10 or "instabil 3/10" not in u["S-0 echt10"]:
        fehler(f"Robustheit/Instabilitaet: {a['S1']['ohne_instabile']} | {u['S-1'][-160:]}")
    # untertrainierte Referenz -> Warnung G > 1,05
    zz = v1._synth(rng, drei(konst, konst, konst), sd=1e-5)
    for z in zz["echt6"]:
        z[NACH] = z["ohne_schaltung"] + 0.03          # Gewinn 3 statt 4 pp -> G(12) = 1,33
    a = auswertung(zz)
    if 12 not in a["konfig"]["echt6"]["warnung_G_ueber_1"] or "WARNUNG G > 1.05" not in urteil(a)["S-2"]:
        fehler("Warnung G > 1 fehlt")
    # Schnittstelle: Strings, JSON
    a = auswertung(v1._synth(rng, drei(konst, konst, konst)), raten={"echt6/voll": 0.08}, abgestuft={})
    u = urteil(a)
    if not all(isinstance(v, str) for v in u.values()) or not {"S-1", "S-2", "S-3", "S-0 echt6"} <= set(u):
        fehler("Urteil nicht Frage -> String")
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN})
    except (TypeError, ValueError) as e:
        fehler(f"nicht JSON-serialisierbar: {e}")
    if not all(isinstance(v, str) for v in HYPOTHESEN.values()) or set(HYPOTHESEN) != {"S-0", "S-1", "S-2", "S-3"}:
        fehler("HYPOTHESEN")
    if DELTA != 0.08 or ALPHA != 0.05 or PAAR_FEST != ("echt6", "fest10") or ANTEILE_ALLE[-1] != ANTEIL_GROSS:
        fehler("Grenzen weichen von C-skala ab")
    return ok


if __name__ == "__main__":
    import sys
    from labor.gemeinsam import speichere
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("C-skala-v2" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
