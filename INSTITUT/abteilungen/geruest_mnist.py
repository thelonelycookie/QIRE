"""Abteilung Clifford -- Clifford-Geruest + k Magiewinkel von Anfang an, auf MNIST
(Karte I-004-mnist, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  Wissen, ob man eine wenig-magische QIRE-Zelle (T ~ 31*k,
      C-hardware) mit EINEM Training bekommt: zufaelliges Clifford-Geruest,
      k Nicht-Clifford-Winkel von Anfang an trainiert -- statt voll trainieren,
      runden, nachtrainieren (zwei Trainings). Und ob dieses Training IM
      eingeschraenkten Raum bei wenig Daten regularisiert (I-024 zeigte: das
      nachtraegliche Runden tut es nicht, dort sank G(2) bei n = 500).
  Wie heute?  I-004 (geruest.py) stellte die Frage auf den Ziffern 8x8: alle
      Urteile HU, weil der Gewinn (2,1 pp = ~11 Testbilder) 0,2-0,3 G nicht
      aufloest; beschreibend G(4) = 0,71 fuer das zufaellige Geruest.
  Was ist neu?  MNIST 16x16 (8 Qubits, L = 4, 32 Winkel), Testmenge 5.000
      Bilder, zwei Datenmengen n_train = 500 / 10.000 (geschachtelt, wie
      I-024), faire Referenz nachtrainiert, Positivkontrolle fuer das
      Lernverfahren (Gradient aus Zufallsstart im bekannten guten Geruest).

WIEDERVERWENDUNG (per import, nichts kopiert):
  geruest.py      -- zufalls_geruest, magie_start, baue_geruest, _gitter_suche
                     (Geruest-Arme bitgleich zu I-004).
  magie_pruning   -- rangfolgen (Rangfolge „verlust“), _gerundet_mit.
  clifford.py     -- _baue (voll/ohne), _gerundet (nachtrainiert = alle 32 frei).
  datenmenge.py   -- _teile (geschachtelte Mengen, 20 % Stopp-Validierung,
                     Tuning-Bewertung auf Pool-Bildern 10.000-11.999, nie Test),
                     trainiere_mit_stopp (Vollbatch-Adam + fruehes Stoppen alle
                     25 Schritte, gleiche Regel in allen Armen), _gepaart,
                     holm, wilcoxon_einseitig. Damit sind voll, nachtrainiert
                     und ohne_schaltung exakt die Arme aus I-024.

ARME (je Mess-Seed und je n_train):
  voll               -- alle 32 Winkel frei, EIN Training.
  ohne_schaltung     -- Identitaet (Basis von G).
  nachtrainiert      -- REFERENZ: voll + zweite Phase, alle 32 Winkel frei,
                        frischer Kopf (Lehre C-fashion/C-referenz).
  nachtraeglich_k{k} -- I-001-Weg: voll runden, k Winkel nach Rangfolge
                        „verlust“ frei (Start beim trainierten Wert), Kopf
                        frisch. ZWEI Trainings.
  geruest_zufall_k{k}-- zufaelliges Clifford-Geruest (Generator 40.000 + seed),
                        k zufaellige Positionen, Magie zufaellig gestartet
                        (60.000 + seed), Kopf frisch. EIN Training. Geruest,
                        Positionen und Start je Seed fuer beide n gleich
                        (gepaart ueber n).
  geruest_fest_k2    -- POSITIVKONTROLLE fuer das Lernverfahren: Geruest und
                        Positionen von nachtraeglich_k2, aber Magie zufaellig
                        neu gestartet. Eine gute Loesung in diesem Geruest ist
                        BEKANNT (nachtraeglich_k2); die Kontrolle prueft, ob
                        der Gradient aus Zufallsstart sie erreicht (Lehre
                        I-003: „loesbar“ ist nicht „lernbar“).
  k in {2, 4}.  Kopf-Initialisierung aller Arme ausser voll/ohne je Seed
  identisch (torch.manual_seed(seed) + reset_parameters), vom Vortest geprueft.

KUERZUNGEN (Budget <= ~75 min bei 4 Prozessen, begruendet):
  * k nur 2 und 4 (k = 1 liegt auf MNIST an der Rauschgrenze einzelner Arme,
    k = 8 ist fuer „wenig Magie“ uninteressant; M50 liegt bei 2-4).
  * n nur 500 und 10.000 (die Enden aus I-024; 2.000 lag dort dazwischen).
  * geruest_fest nur bei k = 2 (Positivkontrolle, keine eigene Frage).
  * I-001 auf MNIST (Genauigkeitsarme nach Abstand/Zufall) NICHT gebaut: je
    Rangfolge ein eigener Arm mit eigener Rate kostet ~+10 min. Billig faellt
    nur ab (R-5, beschreibend): Ueberlappung der Rangfolgen „verlust“ und
    „abstand“ und Spearman(Abstand, Verlustanstieg) aus den voll-Modellen.

VORREGISTRIERT (vor dem ersten Lauf dieses Moduls, Regel 7):
  Bezeichnungen: hi = 10.000, lo = 500; ref(n) = nachtrainiert, AUSSER die
  Referenz ist instabil: zaehlt man bei n >= 2 Seeds mit acc_nachtrainiert <
  acc_voll - 1 pp, wird fuer dieses n voll die Referenz (Urteil erhaelt den
  Zusatz „(Referenz voll)“). Lehre C-referenz: eine Referenz selbst auf
  Einzelseed-Ausreisser pruefen, bevor sie Nenner wird.
  Gewinn(n) = mittel_s(acc_ref - acc_ohne); Skala: Differenzen werden durch
  Gewinn(n) geteilt (positive Konstante, aendert den Wilcoxon nicht) -> G-Einheiten.
  R-0 Vorbedingung: bei hi schlaegt ref die Identitaet (Gewinn > 0, exakter
      Wilcoxon p < 0,05). Sonst R-1/R-2 „HU (Vorbedingung)“.
  R-P Positivkontrolle (Verfahren), bei hi: d_s = (acc_fest2 - acc_nach2)/Gewinn.
        nachtraeglich_k2 schlaegt ohne nicht (mittel > 0, p < 0,05)
                                            -> HU (Vorbedingung: keine bekannte Loesung)
        einseitiges 95-%-KI-unten(d) > -0,25  -> H1 (Gradient erreicht die bekannte Loesung)
        KI-oben(d) < -0,25                    -> H0 (Gradient findet sie nicht)
        sonst                                 -> HU
      Die Kontrolle entscheidet NICHT ueber R-1..R-3, sie beschriftet den
      H0-Zweig von R-1: „(Geruest)“ wenn R-P H1, sonst „(Verfahren
      ungeprueft)“. Bei lo nur beschreibend.
  R-1 (Frage 1) bei hi: e_s = (acc_gz4 - (acc_ohne + acc_ref)/2) / Gewinn,
      mittel(e) = G(4) - 0,5 (wie I-004 R-1).
        Vorbedingung verletzt                           -> HU (Vorbedingung)
        mittel(e) > 0 und Holm-p(Wilcoxon e) < 0,05      -> H1
        KI-oben(e) < 0: gz4 schlaegt ohne (mittel > 0, p < 0,05)
                                                        -> H0 (teilweise)
                        sonst                           -> H0
        sonst                                           -> HU
  R-2 (Frage 2) bei hi, k = 2: d_s = (acc_nach2 - acc_gz2) / Gewinn.
        Vorbedingung verletzt                           -> HU (Vorbedingung)
        mittel(d) > 0 und Holm-p < 0,05                  -> H1 (zwei Trainings lohnen)
        mittel(d) < 0 und Holm-p < 0,05                  -> H0 (umgekehrt)
        KI-oben(d) < 0,15                               -> H0 (billige Pipeline reicht)
        sonst                                           -> HU
      k = 4 beschreibend in R-4.
  R-3 (Frage 3) Regularisierung durch Training IM Geruest, k = 2:
      g_s(n) = (acc_gz2 - acc_ohne) / Gewinn(n), e_s = g_s(lo) - g_s(hi)
      (> 0: bei wenig Daten holt das Geruest einen groesseren Anteil = das
      eingeschraenkte Training regularisiert). Gegenstueck zu I-024 D-2 (dort
      fuer nachtraegliches Runden -0,275, p = 0,042).
        Gewinn(lo) oder Gewinn(hi) nicht > 0 mit p < 0,05
                                                        -> HU (Vorbedingung)
        einseitig (e > 0) Holm-p < 0,05 UND mittel(e) >= 0,15 -> H1
        einseitig (e < 0) p < 0,05 UND mittel(e) <= -0,15     -> H0 (Gegenrichtung)
        90-%-KI(e) ganz in (-0,25; +0,25)                    -> H0 (TOST: kein Einfluss)
        sonst                                                -> HU
  HOLM: Familie = {R-1, R-2, R-3} (nur Tests mit erfuellter Vorbedingung;
      R-1/R-2 zweiseitiger exakter Wilcoxon, R-3 einseitig). R-0, R-P
      Kontrollen, R-4/R-5 beschreibend.
  R-4 beschreibend: G(k)-Kurven je n und Weg (auch gegen voll), gz - voll bei
      lo (starke Regularisierung: Geruest schlaegt volles Modell?),
      Wechselwirkung (gz2 - nach2)(lo) - (gz2 - nach2)(hi), Generalisierungs-
      luecke, Stopp-Schritte (Budget-Warnung, wenn >= 30 % der Seeds eines Arms
      erst nach 90 % der Schritte stoppen), Positivkontrolle bei lo,
      Einzelseed-Instabilitaet je Arm und n: Zahl der Seeds mit acc < acc_ref
      - 1 pp (fuer k-Arme erwartbar, fuer voll/nachtrainiert ein Warnsignal)
      und mit acc < acc_ohne - 1 pp (Einbruch unter die Identitaet), Trainings
      je Pipeline, Multiplikationen, T-Schaetzung.
  R-5 beschreibend (I-001-Vorblick): Ueberlappung verlust|abstand bei k = 2/4
      (Zufallserwartung k/32), Spearman(Abstand, Verlustanstieg), je n.

AUFLOESUNG (vor dem Bau gerechnet, aus den I-024-Rohdaten, MNIST, 10 Seeds):
  Test = 5.000 Bilder, 1 Bild = 0,02 pp. Gewinn(10.000) = 3,53 pp,
  Gewinn(500) = 3,80 pp. SD je Seed einer Differenz zweier k-Arme bei 10.000:
  0,5-0,8 pp = 0,15-0,23 G -> einseitige 95-%-Halbbreite 0,09-0,14 G; bei 500:
  0,9-1,3 pp = 0,24-0,35 G -> 0,14-0,20 G. Schwellen daran ausgerichtet:
  R-2 0,15 G (~0,5 pp), R-P 0,25 G (Stufe G(2) -> G(4)), R-1 prueft G = 0,5
  gegen eine Halbbreite ~0,1-0,15 G. R-3 vergleicht ueber n: SD 0,38 G,
  Halbbreite 0,22 G -- H1 braucht einen Effekt >~ 0,25 G (Groesse des I-024-
  Gegeneffekts), der TOST (+-0,25) ist praktisch unerreichbar; HU ist dort ein
  ehrlicher Ausgang. Zufallsgerueste streuen je Seed staerker als gerundete
  (andere Schaltung je Seed) -- die Halbbreiten koennen groesser ausfallen.

PROTOKOLL: MNIST 16x16, Mess-Seeds 160-169 (frisch), Tuning-Seeds 912/913
  (nur Tuning, nie gemessen; dieselben wie I-024, damit die Referenzarme
  vergleichbar getunt sind). Regel 1: eigene Rate je (Arm, n) -- auch fuer
  voll, nachtrainiert und ohne --, Startgitter 0,08/0,32/1,28 (I-024 fand die
  Optima zwischen 0,08 und 5,12), am Rand in BEIDE Richtungen verlaengert
  (Faktor 4, bis 6 Runden), volles Budget 1.500 Schritte mit fruehem
  Stoppen. Die Rundungs-/Positivkontroll-Raten werden auf den voll-Modellen
  der Tuning-Seeds mit der gewaehlten Voll-Rate je n gesucht. BLAS-Threads 1
  (labor.parallel). Rohdaten sofort sichern (Regel 5).

LAUFZEIT (gemessen 25.09. auf dem belasteten 4-Kern-Container, 1 Thread: ein
  Training mit freien Winkeln ~24 s bei n = 500 und ~105 s bei n = 10.000;
  ohne_schaltung laeuft auf festen Merkmalen und ist fast gratis):
  Messung 7 Arme mit freien Winkeln x 10 Seeds x ~130 s  ~  9.100 CPU-s
  Tuning  7 Arme x ~3,7 Raten x 2 Seeds x ~130 s          ~  6.700 CPU-s
  Summe ~15.800 CPU-s -> ~66 min bei INSTITUT_PROZESSE=4 ohne Fremdlast; mit
  Fremdlast oder schlechter Lastverteilung (pool.map-Bloecke) eher 75-85 min.
  Rauchtest (schnell, 1 Prozess, n = 200/1.000, 40 Schritte): 78 s.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from abteilungen.clifford import GITTER, LAGEN, QireZelle, _baue, _gerundet, multiplikationen, runde_clifford
from abteilungen.datenmenge import (N_TUNING_EVAL, T_JE_WINKEL, _daten, _gepaart, _innen,
                                    _ist_vorzeichen_permutation, _teile, _voll_aus_winkeln, holm,
                                    trainiere_mit_stopp, wilcoxon_einseitig)
from abteilungen.geruest import _gitter_suche, baue_geruest, magie_start, zufalls_geruest
from abteilungen.magie_pruning import _gerundet_mit, rangfolgen
from labor.gemeinsam import genauigkeit, parallel, speichere, wilcoxon_exakt, zusammenfassung

N = 8                                   # 16x16 = 256 = 2^8 Amplituden
ANZAHL = LAGEN * N                      # 32 Winkel
N_WERTE = (500, 10_000)
SCHRITTE = 1_500
STUFEN = (2, 4)
K_R1, K_R2 = 4, 2
MESS_SEEDS = tuple(range(160, 170))
TUNING_SEEDS = (912, 913)
START_GITTER = (0.08, 0.32, 1.28)
MAX_RUNDEN = 6
ALPHA = 0.05
HALB = 0.5
SCHWELLE_R2 = 0.15
MARGE_PK = 0.25
MARGE_R3 = 0.15
TOST_R3 = 0.25
EINBRUCH = 0.01                         # 1 pp: Einzelseed-Instabilitaet
MIN_INSTABIL = 2                        # ab so vielen Seeds gilt nachtrainiert als instabil
BUDGET_RAND = 0.9                       # Stopp nach 90 % der Schritte ...
BUDGET_ANTEIL = 0.3                     # ... bei >= 30 % der Seeds -> Budget-Warnung

REF = "nachtrainiert"
GRUNDARME = ["voll", "ohne_schaltung"]
ARME_GZ = [f"geruest_zufall_k{k}" for k in STUFEN]
ARME_NACH = [f"nachtraeglich_k{k}" for k in STUFEN]
PK = "geruest_fest_k2"
ARME_OHNE_VOLL = GRUNDARME + ARME_GZ            # brauchen kein trainiertes volles Modell
ARME_MIT_VOLL = [REF] + ARME_NACH + [PK]
ARME = ARME_OHNE_VOLL + ARME_MIT_VOLL
TRAININGS_JE_PIPELINE = {"voll": 1, "geruest_zufall": 1, "nachtraeglich": 2, "geruest_fest": 2, REF: 2}

HYPOTHESEN = {
    "R-0": "Vorbedingung: bei n_train = 10.000 schlaegt die Referenz nachtrainiert die Identitaet "
           "(Gewinn > 0, exakter Wilcoxon p < 0,05).",
    "R-P": "Positivkontrolle (Lernverfahren): im bekannten guten Geruest (Geruest + Positionen von "
           "nachtraeglich_k2) erreicht der Gradient aus zufaelligem Magie-Start die Nachtrain-Loesung "
           "bis auf 0,25 G (einseitiges 95-%-KI-unten > -0,25), n = 10.000.",
    "R-1": "Ein zufaelliges Clifford-Geruest mit 4 von Anfang an trainierten Magiewinkeln (EIN "
           "Training) erreicht bei n = 10.000 mindestens den halben Gewinn (G(4) >= 0,5; Wilcoxon "
           "auf acc_gz4 - (ohne + ref)/2, Holm ueber R-1..R-3).",
    "R-2": "Nachtraegliches Runden (zwei Trainings, Rangfolge verlust) schlaegt das zufaellige "
           "Geruest bei gleichem k = 2 (n = 10.000, gepaart, Holm); H0 wenn KI-oben < 0,15 G.",
    "R-3": "Training IM Clifford-Geruest regularisiert: G_500(gz2) - G_10000(gz2) >= 0,15 "
           "(einseitig, Holm); Gegenrichtung <= -0,15; TOST +-0,25 fuer 'kein Einfluss'.",
    "R-4": "beschreibend: G(k)-Kurven je n und Weg, gz - voll bei 500, Wechselwirkung mit "
           "nachtraeglich, Luecke, Stopp, Einzelseed-Instabilitaet, Kosten.",
    "R-5": "beschreibend (I-001-Vorblick): Ueberlappung Rangfolge verlust|abstand, Spearman.",
}


# ------------------------------------------------------------------ Arme
def _k(arm: str) -> int:
    return int(arm.rsplit("_k", 1)[1])


def baue_arm(arm: str, seed: int, winkel=None, rang=None) -> QireZelle:
    """arm -> untrainiertes Modell. winkel/rang: trainiertes voll-Modell desselben
    Seeds und DERSELBEN Datenmenge (nur fuer ARME_MIT_VOLL)."""
    if arm in GRUNDARME:
        return _baue(arm, seed, N)[0]
    if arm == REF:
        return _gerundet(_voll_aus_winkeln(winkel), ANZAHL, seed)
    k = _k(arm)
    if arm.startswith("nachtraeglich_k"):
        return _gerundet_mit(_voll_aus_winkeln(winkel), rang["verlust"][:k], seed)
    if arm.startswith("geruest_fest_k"):
        return baue_geruest(N, runde_clifford(torch.tensor(winkel, dtype=torch.float32)), rang["verlust"][:k], seed)
    if arm.startswith("geruest_zufall_k"):
        geruest, perm = zufalls_geruest(seed, ANZAHL)
        return baue_geruest(N, geruest, perm[:k], seed)
    raise ValueError(arm)


def _arbeit(a):
    """(arm, n, lr, seed, teil, schritte, winkel_voll|None, rang|None) -> Messzeile."""
    arm, n, lr, seed, teil, schritte, winkel, rang = a
    xtr, ytr, xva, yva, xev, yev = _teile(seed, n, teil)
    m = baue_arm(arm, seed, winkel, rang)
    info = trainiere_mit_stopp(m, xtr, ytr, xva, yva, schritte, lr, seed)
    with torch.no_grad():
        z = {"arm": arm, "n": n, "lr": lr, "seed": seed, "acc": genauigkeit(m(xev), yev),
             "acc_train": genauigkeit(m(xtr), ytr), **info}
    frei = m.frei.flatten().bool()
    if arm == "voll":
        z["winkel"] = m.theta.detach().flatten().tolist()
        z["rang"] = rangfolgen(m, xtr, ytr, seed)
    elif 0 < int(frei.sum()) < ANZAHL:
        z["magie"] = m.winkel().detach().flatten()[frei].tolist()
    return z


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=MESS_SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE,
         n_werte=N_WERTE, **kw):
    runden = MAX_RUNDEN
    if schnell:
        seeds, tuning_seeds, schritte, n_werte, runden = tuple(seeds)[:2], tuple(tuning_seeds)[:1], 40, (200, 1_000), 2
    seeds, tuning_seeds, n_werte = tuple(seeds), tuple(tuning_seeds), tuple(sorted(n_werte))
    t0 = time.time()
    zaehler = {"tuning": 0, "mess": 0}
    ablage = {}          # (n, lr, seed) -> voll-Zeile auf dem Tuning-Teil

    def bewerte(paare, basis=None):
        aufg = [(arm, n, lr, s, "tuning", schritte,
                 basis[(s, n)]["winkel"] if basis else None, basis[(s, n)]["rang"] if basis else None)
                for (arm, n), lr in paare for s in tuning_seeds]
        zaehler["tuning"] += len(aufg)
        tmp = {}
        for z in parallel(_arbeit, aufg):
            tmp.setdefault(((z["arm"], z["n"]), z["lr"]), []).append(z["acc"])
            if z["arm"] == "voll":
                ablage[(z["n"], z["lr"], z["seed"])] = z
        return {k: float(np.mean(v)) for k, v in tmp.items()}

    # Tuning 1: Arme ohne Abhaengigkeit (voll, ohne, geruest_zufall), je n
    lr_info = _gitter_suche([(arm, n) for arm in ARME_OHNE_VOLL for n in n_werte], bewerte,
                            start=START_GITTER, runden=runden)
    # Tuning 2: Arme aus dem voll-Modell, auf den voll-Modellen der Tuning-Seeds mit der gewaehlten Rate
    basis = {(s, n): ablage[(n, lr_info[("voll", n)]["lr"], s)] for s in tuning_seeds for n in n_werte}
    lr_info.update(_gitter_suche([(arm, n) for arm in ARME_MIT_VOLL for n in n_werte],
                                 lambda p: bewerte(p, basis), start=START_GITTER, runden=runden))
    lr = {k: v["lr"] for k, v in lr_info.items()}
    print(f"  Raten nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{a}@{n}={lr[(a, n)]:g}" for a in ARME for n in n_werte), flush=True)

    # Messung: erst die unabhaengigen Arme, dann die aus dem voll-Modell
    aufg = [(arm, n, lr[(arm, n)], s, "mess", schritte, None, None)
            for s in seeds for n in n_werte for arm in ARME_OHNE_VOLL]
    stufe1 = parallel(_arbeit, aufg)
    voll = {(z["seed"], z["n"]): z for z in stufe1 if z["arm"] == "voll"}
    aufg2 = [(arm, n, lr[(arm, n)], s, "mess", schritte, voll[(s, n)]["winkel"], voll[(s, n)]["rang"])
             for s in seeds for n in n_werte for arm in ARME_MIT_VOLL]
    stufe2 = parallel(_arbeit, aufg2)
    zaehler["mess"] = len(aufg) + len(aufg2)

    zeilen = {s: {"seed": s, "acc": {}, "acc_train": {}, "stopp": {}, "val_verlust": {},
                  "winkel_voll": {}, "rang": {}, "magie": {}} for s in seeds}
    for z in stufe1 + stufe2:
        r, n = zeilen[z["seed"]], str(z["n"])
        for feld in ("acc", "acc_train", "stopp", "val_verlust"):
            r[feld].setdefault(n, {})[z["arm"]] = z[feld]
        if z["arm"] == "voll":
            r["winkel_voll"][n], r["rang"][n] = z["winkel"], z["rang"]
        if "magie" in z:
            r["magie"].setdefault(n, {})[z["arm"]] = z["magie"]
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + "  |  ".join(
            f"n={n}: voll {z['acc'][str(n)]['voll']:.4f} ref {z['acc'][str(n)][REF]:.4f} "
            f"ohne {z['acc'][str(n)]['ohne_schaltung']:.4f} nach2 {z['acc'][str(n)]['nachtraeglich_k2']:.4f} "
            f"gz2 {z['acc'][str(n)]['geruest_zufall_k2']:.4f} gz4 {z['acc'][str(n)]['geruest_zufall_k4']:.4f} "
            f"fest2 {z['acc'][str(n)][PK]:.4f}" for n in n_werte), flush=True)
    entartet = {str(s): _ist_geruest_permutation(s) for s in seeds}
    ergebnis = {
        "abteilung": "clifford", "modul": "geruest_mnist", "karte": "I-004-mnist", "version": 1,
        "datensatz": "mnist", "qubits": N, "lagen": LAGEN, "n_werte": list(n_werte),
        "hypothesen": HYPOTHESEN, "referenz": REF, "schritte": schritte,
        "lr": {f"{a}@{n}": {**v, "gitter": {str(q): w for q, w in v["gitter"].items()}}
               for (a, n), v in lr_info.items()},
        "seeds": list(seeds), "tuning_seeds": list(tuning_seeds), "trainingslaeufe": zaehler,
        "trainings_je_pipeline": TRAININGS_JE_PIPELINE, "zeilen": zeilen,
        "zufallsgeruest_ist_permutation": entartet,
        "multiplikationen": {"voll": multiplikationen(N), **{f"k{k}": multiplikationen(N, magie=k) for k in STUFEN}},
        "t_gatter_schaetzung": {"voll": T_JE_WINKEL * ANZAHL, **{f"k{k}": T_JE_WINKEL * k for k in STUFEN}},
        "sekunden": time.time() - t0,
    }
    ergebnis["auswertung"] = auswertung(zeilen, schritte=schritte)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


def _ist_geruest_permutation(seed: int) -> bool:
    """Ist das Zufallsgeruest (alle 32 Winkel auf dem Gitter) eine Vorzeichen-Permutation?
    Beschreibend: dann ist das Geruest ohne Magie gleichwertig zur Identitaet."""
    geruest, _ = zufalls_geruest(seed, ANZAHL)
    m = QireZelle(n=N)
    m.fixiere(geruest.reshape(LAGEN, N), torch.zeros(LAGEN, N))
    return _ist_vorzeichen_permutation(m)


# ------------------------------------------------------------------ Auswertung
def auswertung(zeilen, schritte=SCHRITTE) -> dict:
    n_werte = sorted(int(n) for n in zeilen[0]["acc"])
    lo, hi = n_werte[0], n_werte[-1]
    A = lambda arm, n: np.array([z["acc"][str(n)][arm] for z in zeilen], dtype=float)
    aus = {"n": len(zeilen), "n_werte": n_werte, "lo": lo, "hi": hi, "je_n": {}}
    gewinn, ref = {}, {}
    for n in n_werte:
        e = {arm: zusammenfassung(A(arm, n)) for arm in ARME}
        # Referenz-Pruefung auf Einzelseed-Ausreisser (Lehre C-referenz)
        e["ref_instabil_seeds"] = int((A(REF, n) < A("voll", n) - EINBRUCH).sum())
        e["voll_instabil_seeds"] = int((A("voll", n) < A(REF, n) - EINBRUCH).sum())
        ref[n] = REF if e["ref_instabil_seeds"] < MIN_INSTABIL else "voll"
        e["ref"] = ref[n]
        gew = A(ref[n], n) - A("ohne_schaltung", n)
        gewinn[n] = float(gew.mean())
        e["gewinn"], e["gewinn_p"] = gewinn[n], wilcoxon_exakt(gew)
        e["gewinn_ok"] = bool(gewinn[n] > 0 and e["gewinn_p"] < ALPHA)
        g = gewinn[n]
        Gf = lambda arm, basis=g: (float((A(arm, n) - A("ohne_schaltung", n)).mean()) / basis if basis > 0 else None)
        e["G"] = {arm: Gf(arm) for arm in ARME if arm != "ohne_schaltung"}
        gv = float((A("voll", n) - A("ohne_schaltung", n)).mean())
        e["G_gegen_voll"] = {arm: Gf(arm, gv) for arm in ARME if arm != "ohne_schaltung"}
        skala = g if g > 0 else 1.0
        e["d_nach"] = _gepaart(A(REF, n) - A("voll", n))
        # Einzelseed-Instabilitaet je Arm
        e["unter_ref_1pp"] = {arm: int((A(arm, n) < A(ref[n], n) - EINBRUCH).sum()) for arm in ARME}
        e["unter_ohne_1pp"] = {arm: int((A(arm, n) < A("ohne_schaltung", n) - EINBRUCH).sum()) for arm in ARME}
        # Positivkontrolle
        e["pk_vorbed"] = _gepaart(A("nachtraeglich_k2", n) - A("ohne_schaltung", n))
        e["pk"] = _gepaart((A(PK, n) - A("nachtraeglich_k2", n)) / skala)
        # Frage 1 und 2 (bei hi entscheidend, bei lo beschreibend)
        mitte = (A("ohne_schaltung", n) + A(ref[n], n)) / 2
        e["r1"] = _gepaart((A(f"geruest_zufall_k{K_R1}", n) - mitte) / skala)
        e["r1_gegen_ohne"] = _gepaart((A(f"geruest_zufall_k{K_R1}", n) - A("ohne_schaltung", n)) / skala)
        e["nach_minus_gz"] = {str(k): _gepaart((A(f"nachtraeglich_k{k}", n) - A(f"geruest_zufall_k{k}", n)) / skala)
                              for k in STUFEN}
        e["gz_minus_voll_pp"] = {str(k): _gepaart(A(f"geruest_zufall_k{k}", n) - A("voll", n)) for k in STUFEN}
        e["gz_minus_ref_pp"] = {str(k): _gepaart(A(f"geruest_zufall_k{k}", n) - A(ref[n], n)) for k in STUFEN}
        if all("acc_train" in z for z in zeilen):
            T = lambda arm: np.array([z["acc_train"][str(n)][arm] for z in zeilen], dtype=float)
            e["luecke"] = {arm: float((T(arm) - A(arm, n)).mean()) for arm in ARME}
        if all("stopp" in z for z in zeilen):
            st = {arm: np.array([z["stopp"][str(n)][arm] for z in zeilen], dtype=float) for arm in ARME}
            e["stopp_median"] = {arm: float(np.median(v)) for arm, v in st.items()}
            e["budget_warnung"] = sorted(arm for arm, v in st.items()
                                         if float((v >= BUDGET_RAND * schritte).mean()) >= BUDGET_ANTEIL)
        aus["je_n"][str(n)] = e
    ehi, elo = aus["je_n"][str(hi)], aus["je_n"][str(lo)]
    aus["R0"] = {"gewinn": ehi["gewinn"], "p": ehi["gewinn_p"], "ok": ehi["gewinn_ok"], "ref": ref[hi]}
    aus["vorbedingung"] = ehi["gewinn_ok"]
    aus["R1"], aus["R1_gegen_ohne"] = ehi["r1"], ehi["r1_gegen_ohne"]
    aus["R2"] = ehi["nach_minus_gz"][str(K_R2)]
    aus["PK"] = {"vorbed": ehi["pk_vorbed"], "d": ehi["pk"]}
    # Frage 3: ueber n gepaart
    aus["R3_vorbedingung"] = bool(elo["gewinn_ok"] and ehi["gewinn_ok"])
    if gewinn[lo] > 0 and gewinn[hi] > 0:
        g = lambda n: (A(f"geruest_zufall_k{K_R2}", n) - A("ohne_schaltung", n)) / gewinn[n]
        aus["R3"] = _gepaart(g(lo) - g(hi))
        g4 = lambda n: (A(f"geruest_zufall_k{K_R1}", n) - A("ohne_schaltung", n)) / gewinn[n]
        aus["R4_regularisierung_k4"] = _gepaart(g4(lo) - g4(hi))
        gn = lambda n: (A("nachtraeglich_k2", n) - A("ohne_schaltung", n)) / gewinn[n]
        aus["R4_regularisierung_nach2"] = _gepaart(gn(lo) - gn(hi))
        w = lambda n: (A("geruest_zufall_k2", n) - A("nachtraeglich_k2", n)) / gewinn[n]
        aus["R4_wechselwirkung"] = _gepaart(w(lo) - w(hi))
    else:
        aus["R3"] = aus["R4_regularisierung_k4"] = aus["R4_regularisierung_nach2"] = aus["R4_wechselwirkung"] = None
    fam = {}
    if aus["vorbedingung"]:
        fam["R-1"], fam["R-2"] = aus["R1"]["p"], aus["R2"]["p"]
    if aus["R3_vorbedingung"] and aus["R3"] is not None:
        fam["R-3"] = aus["R3"]["p_groesser"]
    aus["holm"] = holm(fam) if fam else {}
    # R-5: Rangfolgen der voll-Modelle (I-001-Vorblick)
    if all(z.get("rang") for z in zeilen):
        from scipy.stats import spearmanr
        r5 = {}
        for n in n_werte:
            rz = [z["rang"][str(n)] for z in zeilen]
            r5[str(n)] = {"ueberlappung_verlust_abstand": {str(k): float(np.mean(
                [len(set(r["verlust"][:k]) & set(r["abstand"][:k])) / k for r in rz])) for k in STUFEN},
                "zufallserwartung": {str(k): k / ANZAHL for k in STUFEN},
                "spearman_abstand_verlust": float(np.nanmean(
                    [spearmanr(r["abstand_je_winkel"], r["verlust_delta"])[0] for r in rz]))}
        aus["R5"] = r5
    else:
        aus["R5"] = None
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (Kopfkommentar). Alle Werte sind Strings.
    Praefixe: 'H1:', 'H0:', 'H0 (teilweise):', 'H0 (umgekehrt):', 'H0 (Gegenrichtung):',
    'H0 (Geruest)', 'HU:', 'HU (Vorbedingung):', 'beschreibend:'."""
    lo, hi = a["lo"], a["hi"]
    ehi = a["je_n"][str(hi)]
    zusatz = " (Referenz voll)" if ehi["ref"] != REF else ""
    pp = lambda v: f"{100 * v:+.2f} pp"
    r0 = a["R0"]
    u = {"R-0": (f"H1: {r0['ref']} schlaegt Identitaet bei n={hi} (Gewinn {pp(r0['gewinn'])}, p = {r0['p']:.4f}){zusatz}"
                 if r0["ok"] else
                 f"H0: Gewinn ueber Identitaet bei n={hi} nicht nachweisbar ({pp(r0['gewinn'])}, p = {r0['p']:.4f}){zusatz}")}
    # Positivkontrolle
    pv, pd = a["PK"]["vorbed"], a["PK"]["d"]
    ki = f"KI {pd['ki_unten']:+.3f} .. {pd['ki_oben']:+.3f}"
    if not (pv["mittel"] > 0 and pv["p"] < ALPHA):
        u["R-P"] = (f"HU (Vorbedingung): nachtraeglich_k2 schlaegt die Identitaet nicht ({pp(pv['mittel'])}, "
                    f"p = {pv['p']:.4f}) -- keine bekannte Loesung im Geruest")
    elif pd["ki_unten"] > -MARGE_PK:
        u["R-P"] = (f"H1: Gradient aus Zufallsstart erreicht im bekannten Geruest die Nachtrain-Loesung bis auf "
                    f"{MARGE_PK} G (fest2 - nach2 = {pd['mittel']:+.3f} G, {ki})")
    elif pd["ki_oben"] < -MARGE_PK:
        u["R-P"] = (f"H0: Gradient findet die bekannte Geruest-Loesung nicht (fest2 - nach2 = {pd['mittel']:+.3f} G, "
                    f"{ki}) -- Training im Geruest scheitert am Verfahren")
    else:
        u["R-P"] = f"HU: Positivkontrolle unentschieden (fest2 - nach2 = {pd['mittel']:+.3f} G, {ki})"
    pk_ok = u["R-P"].startswith("H1")
    # R-1, R-2
    if not a["vorbedingung"]:
        for h in ("R-1", "R-2"):
            u[h] = f"HU (Vorbedingung): kein nachweisbarer Gewinn bei n={hi} -- G undefiniert"
    else:
        r1, r1o = a["R1"], a["R1_gegen_ohne"]
        g4 = r1["mittel"] + HALB
        ph1 = a["holm"]["R-1"]
        grund = "Geruest" if pk_ok else "Verfahren ungeprueft"
        if r1["mittel"] > 0 and ph1 < ALPHA:
            u["R-1"] = (f"H1: zufaelliges Geruest + {K_R1} Magiewinkel erreicht nachweislich mehr als den halben "
                        f"Gewinn (G = {g4:.2f}, Holm-p = {ph1:.4f}) -- ein Training genuegt{zusatz}")
        elif r1["ki_oben"] < 0:
            if r1o["mittel"] > 0 and r1o["p"] < ALPHA:
                u["R-1"] = (f"H0 (teilweise; {grund}): zufaelliges Geruest hilft (G = {g4:.2f}, gegen Identitaet "
                            f"p = {r1o['p']:.4f}), bleibt aber nachweislich unter dem halben Gewinn{zusatz}")
            else:
                u["R-1"] = (f"H0 ({grund}): zufaelliges Geruest bleibt nachweislich unter dem halben Gewinn und "
                            f"ist nicht nachweisbar besser als die Identitaet (G = {g4:.2f}){zusatz}")
        else:
            u["R-1"] = (f"HU: KI ueberdeckt G = 0,5 (G = {g4:.2f}, KI {r1['ki_unten'] + HALB:.2f} .. "
                        f"{r1['ki_oben'] + HALB:.2f}, Holm-p = {ph1:.4f}){zusatz}")
        r2, ph2 = a["R2"], a["holm"]["R-2"]
        if r2["mittel"] > 0 and ph2 < ALPHA:
            u["R-2"] = (f"H1: nachtraegliches Runden schlaegt das zufaellige Geruest bei k={K_R2} "
                        f"({r2['mittel']:+.3f} G, Holm-p = {ph2:.4f}) -- das zweite Training lohnt{zusatz}")
        elif r2["mittel"] < 0 and ph2 < ALPHA:
            u["R-2"] = (f"H0 (umgekehrt): zufaelliges Geruest schlaegt nachtraegliches Runden "
                        f"({r2['mittel']:+.3f} G, Holm-p = {ph2:.4f}){zusatz}")
        elif r2["ki_oben"] < SCHWELLE_R2:
            u["R-2"] = (f"H0: nachtraeglich nachweislich nicht um >= {SCHWELLE_R2} G besser ({r2['mittel']:+.3f} G, "
                        f"KI oben {r2['ki_oben']:+.3f}) -- ein Training reicht{zusatz}")
        else:
            u["R-2"] = (f"HU: unentschieden ({r2['mittel']:+.3f} G, Holm-p = {ph2:.4f}, "
                        f"KI oben {r2['ki_oben']:+.3f}){zusatz}")
    # R-3
    r3 = a["R3"]
    if not a["R3_vorbedingung"] or r3 is None:
        u["R-3"] = f"HU (Vorbedingung): Gewinn bei n={lo} oder n={hi} nicht nachweisbar > 0 -- G undefiniert"
    else:
        ki3 = f"90-%-KI {r3['ki_unten']:+.3f} .. {r3['ki_oben']:+.3f}"
        ph3 = a["holm"]["R-3"]
        if ph3 < ALPHA and r3["mittel"] >= MARGE_R3:
            u["R-3"] = (f"H1: Training im Clifford-Geruest regularisiert -- bei n={lo} holt das Geruest (k={K_R2}) "
                        f"einen um {r3['mittel']:+.3f} G groesseren Anteil (Holm-p = {ph3:.4f}; {ki3})")
        elif r3["p_kleiner"] < ALPHA and r3["mittel"] <= -MARGE_R3:
            u["R-3"] = (f"H0 (Gegenrichtung): auch im Geruest traegt Magie bei wenig Daten WENIGER "
                        f"({r3['mittel']:+.3f} G, p = {r3['p_kleiner']:.4f}; {ki3})")
        elif _innen(r3, TOST_R3):
            u["R-3"] = f"H0: Anteil des Geruests unabhaengig von der Datenmenge (TOST +-{TOST_R3}; {r3['mittel']:+.3f} G, {ki3})"
        else:
            u["R-3"] = (f"HU: unentschieden (G_{lo}(gz2) - G_{hi}(gz2) = {r3['mittel']:+.3f}, "
                        f"p einseitig = {r3['p_groesser']:.4f}; {ki3})")
    # R-4 beschreibend
    teile = []
    for n in a["n_werte"]:
        e = a["je_n"][str(n)]
        G = e["G"]
        f = lambda arm: "-" if G.get(arm) is None else f"{G[arm]:.2f}"
        teile.append(
            f"n={n} (ref {e['ref']}, Gewinn {pp(e['gewinn'])}): G nach k2/k4 {f('nachtraeglich_k2')}/{f('nachtraeglich_k4')}, "
            f"gz k2/k4 {f('geruest_zufall_k2')}/{f('geruest_zufall_k4')}, fest k2 {f(PK)}, voll {f('voll')}; "
            f"gz4 - voll {pp(e['gz_minus_voll_pp']['4']['mittel'])} (p = {e['gz_minus_voll_pp']['4']['p']:.3f}); "
            f"nach4 - gz4 {e['nach_minus_gz']['4']['mittel']:+.3f} G; d_nach {pp(e['d_nach']['mittel'])}; "
            f"Instabil nach<voll-1pp {e['ref_instabil_seeds']}, voll<nach-1pp {e['voll_instabil_seeds']}; "
            f"unter Identitaet-1pp {sum(e['unter_ohne_1pp'].values())} Arm-Seeds; "
            f"Budget-Warnung {e.get('budget_warnung', '-') or 'keine'}")
    ww = a.get("R4_wechselwirkung")
    if ww is not None:
        teile.append(f"Wechselwirkung (gz2 - nach2)({lo}) - ({hi}) = {ww['mittel']:+.3f} G (p = {ww['p']:.3f}); "
                     f"Regularisierung gz4 {a['R4_regularisierung_k4']['mittel']:+.3f} G, "
                     f"nach2 {a['R4_regularisierung_nach2']['mittel']:+.3f} G")
    elo = a["je_n"][str(lo)]
    teile.append(f"Positivkontrolle bei n={lo}: fest2 - nach2 = {elo['pk']['mittel']:+.3f} G "
                 f"(KI unten {elo['pk']['ki_unten']:+.3f})")
    teile.append(f"Trainings je Pipeline {TRAININGS_JE_PIPELINE}")
    u["R-4"] = "beschreibend: " + "; ".join(teile)
    r5 = a.get("R5")
    if r5 is None:
        u["R-5"] = "beschreibend: keine Rangdaten (nur Genauigkeiten) -- I-001-Vorblick entfaellt"
    else:
        u["R-5"] = "beschreibend: I-001-Vorblick (keine Genauigkeitsarme) -- " + "; ".join(
            f"n={n}: Ueberlappung verlust|abstand k2 {v['ueberlappung_verlust_abstand']['2']:.2f}, "
            f"k4 {v['ueberlappung_verlust_abstand']['4']:.2f} (Zufall {v['zufallserwartung']['2']:.3f}/"
            f"{v['zufallserwartung']['4']:.3f}), Spearman {v['spearman_abstand_verlust']:+.2f}"
            for n, v in r5.items())
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_struktur() -> bool:
    """Tun die Bausteine, was ihr Name sagt? Billig: untrainiertes „volles“ Modell als
    Stellvertreter, 300 MNIST-Bilder, hoechstens 5 Schritte."""
    ok = True
    torch.set_num_threads(1)

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    # (a) Mengen geschachtelt, Tuning-Bewertung nicht aus den Trainingsbildern, Testmenge fest
    # (jede Menge = Praefix desselben Pools -> die 500er ist Teil der 10.000er)
    s0 = MESS_SEEDS[0]
    klein = _teile(s0, N_WERTE[0], "mess")
    x_pool = _daten(s0)[0]
    for n in N_WERTE:
        xtr, _, xva, _, xev, _ = _teile(s0, n, "tuning")
        if len(xtr) + len(xva) != n or len(xev) != N_TUNING_EVAL:
            fehler(f"Aufteilung bei n = {n}")
        if not torch.equal(torch.cat([xva, xtr]), x_pool[:n]):
            fehler(f"Menge bei n = {n} ist nicht das Praefix des Pools")
        if not torch.equal(_teile(s0, n, "mess")[4], klein[4]):
            fehler("Testmenge haengt von n ab")
    # (b) Arme
    torch.manual_seed(7)
    voll = QireZelle(n=N)
    winkel = voll.theta.detach().flatten().tolist()
    x, y = x_pool[:300], _daten(s0)[1][:300]
    rang = rangfolgen(voll, x, y, seed=1)
    s = 3
    _, perm = zufalls_geruest(s, ANZAHL)
    start = magie_start(s, ANZAHL)
    koepfe = {}
    for arm in ARME:
        m = baue_arm(arm, s, winkel, rang)
        wk, frei = m.winkel().detach().flatten(), m.frei.flatten().bool()
        if arm not in GRUNDARME:
            koepfe[arm] = (m.kopf.weight.detach().clone(), m.kopf.bias.detach().clone())
            if torch.equal(m.kopf.weight, voll.kopf.weight):
                fehler(f"{arm}: Kopf geerbt statt frisch")
        if arm == "ohne_schaltung":
            if int(frei.sum()) != 0 or not torch.equal(wk, torch.zeros(ANZAHL)):
                fehler("ohne_schaltung ist nicht die Identitaet")
            continue
        if arm in ("voll", REF):
            if int(frei.sum()) != ANZAHL:
                fehler(f"{arm}: nicht alle Winkel frei")
            if arm == REF and not torch.equal(wk, torch.tensor(winkel)):
                fehler("nachtrainiert startet nicht beim trainierten voll")
            continue
        k = _k(arm)
        soll = perm[:k] if arm.startswith("geruest_zufall") else rang["verlust"][:k]
        if int(frei.sum()) != k or sorted(torch.nonzero(frei).flatten().tolist()) != sorted(soll):
            fehler(f"{arm}: falsche freie Positionen")
        g = wk[~frei]
        if bool(((g / GITTER - torch.round(g / GITTER)).abs() > 1e-6).any()):
            fehler(f"{arm}: Geruest nicht auf dem Gitter")
        if arm.startswith(("nachtraeglich", "geruest_fest")) and \
                not torch.equal(g, runde_clifford(torch.tensor(winkel))[~frei]):
            fehler(f"{arm}: Geruest ist nicht das gerundete volle Modell")
        if arm.startswith("geruest") and not torch.equal(wk[frei], start[frei]):
            fehler(f"{arm}: Magie nicht zufaellig (magie_start) gestartet")
        if arm.startswith("nachtraeglich") and not torch.equal(wk[frei], torch.tensor(winkel)[frei]):
            fehler(f"{arm}: Magie startet nicht beim trainierten Wert")
        m.zero_grad()
        F.cross_entropy(m(x[:64]), y[:64]).backward()
        gr = m.theta.grad.flatten()
        if not torch.all(gr[~frei] == 0) or torch.any(gr[frei] == 0):
            fehler(f"{arm}: Gradientenmaske falsch")
        xtr, ytr, xva, yva = x[60:], y[60:], x[:60], y[:60]
        trainiere_mit_stopp(m, xtr, ytr, xva, yva, 5, 0.1, s, takt=5)
        wk2 = m.winkel().detach().flatten()
        if not torch.equal(wk2[~frei], wk[~frei]):
            fehler(f"{arm}: Training bewegt das Geruest")
    ref0 = koepfe[REF]
    if any(not (torch.equal(w, ref0[0]) and torch.equal(b, ref0[1])) for w, b in koepfe.values()):
        fehler("Koepfe der abgeleiteten Arme nicht identisch initialisiert")
    # geruest_zufall haengt nicht vom vollen Modell ab
    a1, a2 = baue_arm("geruest_zufall_k2", s, winkel, rang), baue_arm("geruest_zufall_k2", s)
    if any(not torch.equal(a1.state_dict()[q], a2.state_dict()[q]) for q in a1.state_dict()):
        fehler("geruest_zufall benutzt Information aus dem vollen Modell")
    ents = [sd for sd in MESS_SEEDS if _ist_geruest_permutation(sd)]
    if len(ents) > len(MESS_SEEDS) // 2:
        fehler(f"Zufallsgerueste meist Vorzeichen-Permutationen: {ents}")
    # (c) Gittersuche mit Tupel-Schluesseln, beide Richtungen
    for spitze in (0.08 / 16, 1.28 * 16, 0.32):
        info = _gitter_suche([("a", 1)], lambda p: {(k, lr): -abs(math.log(lr / spitze)) for k, lr in p},
                             start=START_GITTER, runden=MAX_RUNDEN)
        if info[("a", 1)]["am_rand"] or not math.isclose(info[("a", 1)]["lr"], spitze):
            fehler(f"Gittersuche findet {spitze} nicht: {info}")
    # (d) Statistik
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    if not (math.isclose(h["a"], 0.03) and math.isclose(h["c"], 0.06) and math.isclose(h["b"], 0.06)):
        fehler(f"Holm: {h}")
    if abs(wilcoxon_einseitig(np.arange(1, 11) / 100) - 1 / 1024) > 1e-12:
        fehler("einseitiger Wilcoxon")
    return ok


def _synth(rng, G, gewinn=None, sd=0.002, streu=None, n_seeds=10, ref_kollaps=0, voll_versatz=0.0):
    """Synthetische Zeilen. G: {n: {arm: wahrer Anteil}} oder {arm: Anteil} fuer alle n
    (fehlende Arme: 0,5). gewinn: {n: ref - ohne}. ref_kollaps: so viele Seeds, in denen
    nachtrainiert um 4 pp einbricht (voll bleibt gesund)."""
    gewinn = gewinn or {n: 0.035 for n in N_WERTE}
    streu = streu or {}
    je_n = {n: (G[n] if n in G else G) for n in N_WERTE}
    out = []
    for s in range(n_seeds):
        z = {"seed": s, "acc": {}, "acc_train": {}, "stopp": {}}
        for n in N_WERTE:
            ohne = 0.84 + 0.02 * math.log10(n / 500) + rng.normal(0, sd)
            refw = ohne + gewinn[n] + rng.normal(0, sd)
            acc = {"ohne_schaltung": ohne, REF: refw - (0.04 if s < ref_kollaps else 0.0),
                   "voll": refw - voll_versatz + rng.normal(0, sd)}
            for arm in ARME_GZ + ARME_NACH + [PK]:
                acc[arm] = ohne + gewinn[n] * je_n[n].get(arm, 0.5) + rng.normal(0, sd) \
                    + rng.normal(0, streu.get(arm, 0.0) + 1e-12)
            z["acc"][str(n)] = acc
            z["acc_train"][str(n)] = {arm: min(1.0, v + 0.03) for arm, v in acc.items()}
            z["stopp"][str(n)] = {arm: 500 for arm in acc}
        out.append(z)
    return out


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Bausteine pruefen, dann jeden Urteilszweig auf synthetischen
    Zeilen erzwingen. Null- und Grenzfaelle kippen mit Rate ~alpha -- dort Toleranz."""
    ok = _vortest_struktur()
    rng = np.random.default_rng(160)
    lo, hi = N_WERTE[0], N_WERTE[-1]
    gut = {"geruest_zufall_k4": 0.9, "geruest_zufall_k2": 0.2, "nachtraeglich_k2": 0.8, PK: 0.8}
    alle_h1 = {hi: gut, lo: {**gut, "geruest_zufall_k2": 0.8}}
    faelle = [
        # 1) alle H1
        (dict(G=alle_h1), {"R-0": "H1:", "R-P": "H1:", "R-1": "H1:", "R-2": "H1:", "R-3": "H1:"}, 0),
        # 2) teilweise / umgekehrt / Gegenrichtung / Positivkontrolle H0 -> R-1 mit Zusatz „Verfahren ungeprueft“
        (dict(G={hi: {"geruest_zufall_k4": 0.25, "geruest_zufall_k2": 0.9, "nachtraeglich_k2": 0.5, PK: 0.1},
                 lo: {"geruest_zufall_k2": 0.3, "nachtraeglich_k2": 0.5, PK: 0.1}}),
         {"R-P": "H0:", "R-1": "H0 (teilweise; Verfahren ungeprueft):", "R-2": "H0 (umgekehrt):",
          "R-3": "H0 (Gegenrichtung):"}, 0),
        # 3) gz nutzlos, PK gut -> „H0 (Geruest)“; kein Unterschied nach/gz (H0, Nullfall);
        #    Anteil unabhaengig von n bei kleiner Streuung (TOST-H0)
        (dict(G={"geruest_zufall_k4": 0.0, "geruest_zufall_k2": 0.5, "nachtraeglich_k2": 0.5, PK: 0.5}, sd=0.001),
         {"R-P": "H1:", "R-1": "H0 (Geruest):", "R-2": "H0:", "R-3": "H0:"}, 1),
        # 4) Grenzfaelle genau auf den Schwellen, grosse Streuung -> HU.
        #    Kalibriert (300 Ziehungen, 25.09.): Kipprate je Hypothese in Fall 4 <= 11 %, in Fall 8
        #    <= 7 %, im Nullfall 3 <= 3 %. Mit Toleranz 2 (bzw. 1) von 5 ist ein Fehlalarm des
        #    Vortests ~1 %; ein echter Fehler im Baum kippt fast immer alle 5.
        (dict(G={"geruest_zufall_k4": 0.5, "geruest_zufall_k2": 0.4, "nachtraeglich_k2": 0.5, PK: 0.25},
              streu={"geruest_zufall_k4": 0.03, "geruest_zufall_k2": 0.03, PK: 0.03}),
         {"R-P": "HU:", "R-1": "HU:", "R-2": "HU:", "R-3": "HU:"}, 2),
        # 5) Vorbedingung verletzt (kein Gewinn bei hi, nach2 schlaegt ohne nicht)
        (dict(G={"geruest_zufall_k4": 0.9, "nachtraeglich_k2": 0.8}, gewinn={lo: 0.035, hi: -0.005}),
         {"R-0": "H0:", "R-P": "HU (Vorbedingung):", "R-1": "HU (Vorbedingung):",
          "R-2": "HU (Vorbedingung):", "R-3": "HU (Vorbedingung):"}, 0),
        # 6) kein Gewinn bei lo -> nur R-3 HU (Vorbedingung), R-1/R-2 entscheiden weiter
        (dict(G=alle_h1, gewinn={lo: 0.0, hi: 0.035}),
         {"R-0": "H1:", "R-1": "H1:", "R-2": "H1:", "R-3": "HU (Vorbedingung):"}, 0),
        # 7) Referenz nachtrainiert kollabiert auf 4 Seeds -> voll wird Referenz, Urteile wie Fall 1
        (dict(G=alle_h1, ref_kollaps=4), {"R-0": "H1:", "R-1": "H1:", "R-2": "H1:", "R-3": "H1:"}, 0),
        # 8) Aufloesungspruefung: alles verrauscht -> nur HU darf feuern
        (dict(G={"geruest_zufall_k4": 0.6, "nachtraeglich_k2": 0.55, "geruest_zufall_k2": 0.5, PK: 0.5},
              streu={a: 0.05 for a in ARME_GZ + ARME_NACH + [PK]}),
         {"R-1": "HU:", "R-2": "HU:", "R-3": "HU:"}, 2),
    ]
    for par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            a = auswertung(_synth(rng, **par))
            u = urteil(a)
            try:
                json.dumps({"a": a, "u": u})
            except (TypeError, ValueError) as ex:
                ok = False
                print("  VORTEST FEHLER: nicht JSON-serialisierbar:", ex)
            if not all(isinstance(v, str) for v in u.values()):
                ok = False
                print("  VORTEST FEHLER: Urteil ist kein String")
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest: {h} erwartet '{praefix}' -> {u[h]}")
            if par.get("ref_kollaps") and not (a["je_n"][str(hi)]["ref"] == "voll" and "(Referenz voll)" in u["R-1"]):
                ok = False
                print("  VORTEST FEHLER: instabile Referenz nicht erkannt")
        for h, f in fehl.items():
            if f > toleranz:
                ok = False
                print(f"  VORTEST FEHLER: {h} bei Fall {erwartet}: {f} von {ziehungen} falsch")
    # stabile Referenz bleibt nachtrainiert
    a = auswertung(_synth(rng, G=alle_h1))
    if a["je_n"][str(hi)]["ref"] != REF:
        ok = False
        print("  VORTEST FEHLER: stabile Referenz faelschlich ersetzt")
    # Holm wird im Urteil benutzt: rohe p klein, Holm-p gross -> kein H1
    a["holm"] = {h: 0.06 for h in a["holm"]}
    u = urteil(a)
    if any(u[h].startswith("H1") for h in ("R-1", "R-2", "R-3")):
        ok = False
        print("  VORTEST FEHLER: Urteil ignoriert Holm", u)
    # beschreibende Punkte sind als solche markiert
    if not (u["R-4"].startswith("beschreibend") and u["R-5"].startswith("beschreibend")):
        ok = False
        print("  VORTEST FEHLER: R-4/R-5 nicht als beschreibend markiert")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-004-mnist" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
