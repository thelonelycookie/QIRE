"""Abteilung Clifford -- Diskrete Clifford-Suche auf zwei Datensaetzen gegen volles Training
(Karte I-005-v2, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  Pruefen, ob eine reine Clifford-Schaltung (0 Magiewinkel, 0 T-Gatter),
      gefunden durch die diskrete Suche aus I-005, GLEICH GUT ist wie das volle Gradienten-
      training mit allen 32 Winkeln frei -- auf MNIST 16x16 UND Fashion-MNIST 16x16.
  Wie heute?  I-005 (alle H1): MNIST suche 0,9352 ~ voll 0,9344, aber nur beschreibend,
      ohne TOST, und `voll` lief mit EINER festen Rate 0,08 (Verstoss gegen Regel 1 „Lernrate
      je Arm“). Fashion ungeprueft; dort war die Magie-Konzentration schwaecher (G(4) ~ 1,0
      aus C-fashion, M90 = 4), also ist „keine Magie noetig“ dort NICHT selbstverstaendlich.
  Was ist neu?  Vorregistrierte Aequivalenzfrage (TOST) mit fair getuntem Gegenarm:
      Lernrate von `voll` je Datensatz mit Stabilitaetsabstand (labor.gemeinsam.
      gittersuche_stabil / waehle_rate), eigene Tuning-Seeds; L2 des gemeinsamen Kopfes je
      Datensatz nach fester Regel auf den Tuning-Seeds (I-005 hatte nur fuer MNIST
      Entwurfswerte).
  Woran merken wir es?  V-1 (MNIST) und V-2 (Fashion): Suche = voll im Rahmen +-0,5 pp
      (GLEICHWERTIG) oder einer besser; V-3: Suche schlaegt auf Fashion den Median von 100
      zufaelligen Clifford-Schaltungen (Replikation von Q-1).

VERFAHREN (unveraendert aus clifford_suche.py importiert, nicht kopiert):
  Kriterium Ridge-LOO (lambda 1), SA auf kanonischen Bits (6.000 Schritte, tau 1,4 -> 0,14
  relativ zur sd der Zufalls-Clifford), gierige Politur (<= 300), Zwischenspeicher; alle Arme
  mit DEMSELBEN logistischen L-BFGS-Kopf, trainiert auf allen 6.000 Nicht-Test-Bildern,
  gemessen auf 5.000 Testbildern (clifford.daten, 16x16, wie I-005/C-mnist/C-fashion).
  Arme je Seed: suche, zufall_i (100; Median, best100), identitaet, voll (Winkel per Adam,
  1.500 Schritte, GETUNTE Rate, danach gemeinsamer Kopf), voll_eigener_kopf, k0 (voll
  gerundet), zufallssuche (gleich viele neue Bewertungen wie die Suche). Weggelassen
  gegenueber I-005: die Politur ab k0 (beschreibend, fuer V-1..V-3 ohne Belang).

TUNING (nur Tuning-Seeds, nie Mess-Seeds; Validierung = erstes Viertel der Nicht-Test-Bilder,
  trainiert auf den anderen drei Vierteln):
  1. Kopf-L2 je Datensatz: Gitter 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 (Faktor ~3,16,
     bis zu 2 Verlaengerungen, wenn das Optimum am Rand liegt); Kriterium = mittlere
     Validierungsgenauigkeit ueber Identitaet + 8 zufaellige Clifford-Schaltungen je
     Tuning-Seed; Gleichstand -> groessere L2. Gilt fuer ALLE Arme gleich (fair). I-005 nahm
     fuer MNIST 1e-2 aus zwei Entwurfs-Seeds; die gewaehlte MNIST-L2 wird berichtet.
  2. Lernrate von voll je Datensatz: gittersuche_stabil (Gitter 0,02..0,32, Faktor 2,
     Kollaps 5 pp, Kriterium schlechtester Tuning-Seed, oberer Nachbar darf auf keinem
     Tuning-Seed kollabieren), 3 Tuning-Seeds, volles Budget 1.500 Schritte. Kriterium =
     Validierungsgenauigkeit des EIGENEN Adam-Kopfes (reagiert auf Kollaps; ein neu
     angepasster Kopf wuerde einen Kollaps der Winkel verdecken). Die Genauigkeit mit dem
     gemeinsamen Kopf wird je Punkt mitgeschrieben (beschreibend).
  k0 bekommt keine eigene Rate: es ist das gerundete voll-Modell mit gemeinsamem Kopf.

SEEDS (28.09. geprueft gegen ergebnisse/*.json, agenda.json und abteilungen/*.py -- alle
  frisch; lauf() und Vortest pruefen Disjunktheit gegen die dort gefundenen belegten Seeds):
  Mess-Seeds       MNIST 300-309, Fashion 310-319   (Option n_seeds=20: + MNIST 330-339,
                                                      Fashion 340-349)
  Strukturaufgabe  360-369 (Q-0)
  Tuning           MNIST 950-952, Fashion 953-955
  Rauchtest        MNIST 970/971, Fashion 972/973, Tuning 974-979, Struktur 980/981
                   (Entwurfs-Seeds; 970/971/980/981 schon in I-005-Entwurf/Rauchtest benutzt,
                   duerfen nie Mess-Seeds werden)

VORREGISTRIERT (vor der ersten Zahl der Mess-Seeds; Holm ueber GENAU die Familie
  {V-1, V-2, V-3}, getrennt fuer Unterschiedstest und Aequivalenztest):
  Je Hypothese gepaart je Seed d_s; Unterschied: exakter zweiseitiger Wilcoxon -> Holm-p;
  Aequivalenz: TOST (zwei einseitige t-Tests, p_TOST = max) -> Holm-p_TOST; dazu das
  unadjustierte 90-%-KI (beschreibend).
  V-1 MNIST:   d_s = acc_test(suche) - acc_test(voll), Grenze delta = 0,5 pp.
  V-2 Fashion: dasselbe auf Fashion.
      Verzweigung V-1/V-2 (in dieser Reihenfolge; Aequivalenz ist die Hauptfrage):
        Holm-p_TOST < 0,05:
          Q-0 nicht H1                          -> HU (Positivkontrolle)
          Sensitivitaet verfehlt                -> HU (Sensitivitaet)
          voll auf >= 1 Mess-Seed kollabiert    -> HU (Instabilitaet)
          sonst                                 -> GLEICHWERTIG (ist zusaetzlich Holm-p < 0,05,
                                                   steht im Text: Unterschied nachweisbar, aber
                                                   kleiner als die Grenze)
        Holm-p < 0,05 und mittel(d) < 0         -> VOLL_BESSER (die Magiewinkel bringen etwas,
                                                   das die Clifford-Suche nicht erreicht)
        Holm-p < 0,05 und mittel(d) > 0         -> SUCHE_BESSER; bei kollabiertem voll-Seed
                                                   -> HU (Instabilitaet)
        sonst                                   -> HU
      Sensitivitaet (assay sensitivity, nicht in der Familie): mittel(voll - identitaet)
        >= 2*delta = 1 pp und exakter Wilcoxon p < 0,05. Sonst hiesse „gleich“ nur, dass
        die Schaltung auf diesem Datensatz ueberhaupt nichts ausmacht (Grenze relativ zum
        Effekt: delta hoechstens halber Gewinn der Schaltung). I-005 MNIST: 3,75 pp.
      Kollaps (vor der Urteilsbildung geprueft, Regel aus UEBERGABE §6): Mess-Seed mit
        voll_eigener_kopf < Median(voll_eigener_kopf) - 5 pp (oder nicht endlich).
      delta = 0,5 pp: halb so gross wie die Streuung einer einzelnen Zufalls-Clifford
        (C-mnist ~1 pp), ~1/7 des Schaltungsgewinns auf MNIST, 25 Testbilder je Seed.
  V-3 Fashion: d_s = acc_test(suche) - median_i acc_test(zufall_i) (Replikation von I-005
      Q-1), Verzweigung wie Q-1 dort, aber TOST +-0,3 pp mit Holm-p_TOST:
        Holm-p < 0,05, mittel > 0 -> H1; Holm-p < 0,05, mittel < 0 -> H0 (Gegenrichtung);
        Holm-p_TOST < 0,05 und Q-0 = H1 -> H0 (gleich); TOST ja, Q-0 nicht H1 -> HU
        (Positivkontrolle); sonst HU.
  Fehlt ein Datensatz (Teillauf), zaehlt sein Glied mit p = 1 in Holm (Familie bleibt 3).
  Q-0 Positivkontrolle (Strukturaufgabe, Verfahren UNVERAENDERT aus I-005, frische Seeds
      360-369, NICHT in der Familie): exakt I-005 Q-0 (G_L >= 0,80 und Suche > Identitaet ->
      H1 ...). Laeuft mit, weil (a) sie billig ist (~50 s je Seed) und (b) die Regel
      „Positivkontrolle muss feuern koennen“ fuer JEDEN Lauf gilt, der ein „gleich“-Urteil
      faellen darf (der Code-Pfad kann sich seit I-005 geaendert haben).
  V-B beschreibend je Datensatz: alle Arme, G_voll = (suche - id)/(voll - id), suche - k0,
      suche - zufallssuche, suche - best100 (unadjustiert), Siegerfluch, Aufwand, Kaskade,
      gewaehlte Rate und L2, Tuning-Gitter.

ERREICHBARKEIT DER TOST-GRENZE (vor dem Lauf gerechnet, aus I-005-Rohdaten):
  I-005 MNIST, d = suche - voll je Seed (190-199): +0,16 +0,62 -1,22 +0,20 +0,44 +0,82
  +0,74 -0,30 -0,18 -0,48 pp -> Mittel +0,08 pp, sd 0,64 pp, se (n=10) 0,20 pp.
  TOST besteht, wenn |mittel| + t * sd/sqrt(n) < 0,5 pp. Holm ueber 3: das kleinste
  p_TOST muss < 0,05/3 = 0,0167 sein (t_0,983;9 ~ 2,52), das zweite < 0,025 (t ~ 2,26).
    n = 10: noetig sd <= (0,5 - |m|) * 3,16 / 2,52 = 0,63 pp (m = 0) bzw. 0,53 pp
            (m = +0,08). I-005-Streuung 0,64 pp -> GRENZWERTIG: Macht fuer GLEICHWERTIG bei
            wahrem d = 0 grob 10-20 % (unadjustiert, t = 1,83, waeren es ~50 %).
            Mit den I-005-Zahlen: 0,08 + 2,52 * 0,20 = 0,59 pp > 0,5 -> waere HU gewesen.
    n = 20: t_0,983;19 ~ 2,31, se 0,14 -> |m| < 0,17 pp noetig -> Macht ~75 % (d = 0).
  Das getunte voll kann die Streuung senken oder erhoehen (Rate 0,08 lag in I-005 fest).
  Fashion: Streuung unbekannt (keine Suche-Daten) -- eher groesser (schwerere Aufgabe).
  FOLGERUNG: Mit 10 Seeds ist HU fuer V-1/V-2 das wahrscheinlichste Ergebnis, selbst wenn
  beide Arme wirklich gleich sind. Deshalb gibt es n_seeds = 20 (Direktor entscheidet VOR dem
  Commit; die Wahl ist Teil der Vorregistrierung). Unterschiedstests: kleinstes exaktes
  Wilcoxon-p = 2/2^10 = 0,002, Holm x3 = 0,006 < 0,05 -> erreichbar.

LAUFZEIT (Schaetzung aus I-005: je MNIST-Seed 1.230-1.415 s = Suche ~550 s + Zufallssuche
  ~500 s + 100 Koepfe + voll ~130 s; Strukturseed ~50 s; INSTITUT_PROZESSE = 5):
  L2-Tuning 6 Aufgaben ~3-5 min; Ratensuche 30 Punkte x ~100 s (+ je Verlaengerung
  6 Punkte) ~12-20 min; Messung n=10: 20 x ~1.330 s + 10 x 50 s = ~27.100 s / 5 = ~90 min.
  GESAMT n = 10: ~1 h 50 min - 2 h; n = 20: ~3 h 15 min - 3 h 30 min.

GRENZEN (vorab benannt): Test 5.000 Bilder (0,02 pp je Bild). Die Suche sieht den Test nie
  (I-005-Vortest (h)). voll bekommt fuer das Urteil den gemeinsamen Kopf (fair zur Suche),
  sein eigener Kopf steht beschreibend daneben. Clifford-Schaltungen sind nach Gottesman-
  Knill klassisch effizient simulierbar: GLEICHWERTIG ist ein Befund ueber Hardware-Kosten
  (0 T-Gatter), nicht ueber einen Quantenvorteil.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen.clifford import LAGEN, QUBITS_JE_DATENSATZ
from abteilungen.clifford_suche import (_bewerte, _gemeinsame_arme, _gepaart, codes_aus_winkeln, holm,
                                        kanonisch, kaskaden_kennzahl, normiere, unitaer_winkel, zufallssuche)
from labor.gemeinsam import gittersuche_stabil, parallel, speichere, wilcoxon_exakt, zusammenfassung

DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
SEEDS_LANG = {"mnist": tuple(range(300, 310)) + tuple(range(330, 340)),
              "fashion_mnist": tuple(range(310, 320)) + tuple(range(340, 350))}
N_SEEDS = 20                                   # Vorregistrierung 28.09. (Direktor): 20 -- bei 10 Macht fuer GLEICHWERTIG nur ~10-20 %
SEEDS_STRUKTUR = tuple(range(360, 370))
TUNING_SEEDS = {"mnist": (950, 951, 952), "fashion_mnist": (953, 954, 955)}
ENTWURF_SEEDS = {"mnist": (970, 971), "fashion_mnist": (972, 973)}
ENTWURF_TUNING = {"mnist": (974, 975, 976), "fashion_mnist": (977, 978, 979)}
ENTWURF_STRUKTUR = (980, 981)
# belegt (28.09., aus ergebnisse/*.json, agenda.json, abteilungen/*.py) -- nie als Mess-/Tuning-Seed
_BELEGT = (set(range(0, 48)) | {50, 51, 52} | set(range(60, 73)) | set(range(80, 95)) | set(range(100, 130))
           | set(range(140, 150)) | set(range(160, 172)) | set(range(180, 182)) | set(range(190, 240))
           | set(range(250, 256)) | {900, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913,
                                     920, 921, 930, 931, 932, 940, 941}
           | set(range(970, 972)) | set(range(980, 995)))

SCHRITTE_VOLL = 1500
LR_GITTER = (0.02, 0.04, 0.08, 0.16, 0.32)
LR_FAKTOR, KOLLAPS = 2, 0.05
L2_GITTER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
L2_FAKTOR, L2_RUNDEN, L2_ZUFALL, L2_SEED = math.sqrt(10), 3, 8, 90_000
ALPHA = 0.05
TOST_V = 0.005        # V-1/V-2: +-0,5 pp
TOST_V3 = 0.003       # V-3: +-0,3 pp wie I-005 Q-1
SENS_MIN = 2 * TOST_V
FAMILIE = {"V-1": ("mnist", "suche_minus_voll"), "V-2": ("fashion_mnist", "suche_minus_voll"),
           "V-3": ("fashion_mnist", "suche_minus_median")}

HYPOTHESEN = {
    "Q-0": "Positivkontrolle wie I-005 Q-0 (Strukturaufgabe, gleiches Verfahren, frische Seeds 360-369): "
           "G_L >= 0,80 und Suche > Identitaet (p < 0,05) -> H1; G_L < 0,50 -> H0; sonst HU. Nicht in der Familie; "
           "Voraussetzung fuer GLEICHWERTIG (V-1/V-2) und H0 (gleich) (V-3).",
    "V-1": "MNIST 16x16: Suche (0 Magiewinkel) vs. voll (alle Winkel, Rate je Datensatz mit Stabilitaetsabstand), "
           "gemeinsamer Kopf, gepaart je Seed. Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (wenn Q-0 H1, voll - id "
           ">= 1 pp und p < 0,05, kein voll-Kollaps; sonst HU mit Grund); sonst Holm-p(Wilcoxon) < 0,05 -> "
           "VOLL_BESSER / SUCHE_BESSER; sonst HU. Holm ueber V-1..V-3, getrennt fuer TOST und Wilcoxon.",
    "V-2": "Fashion-MNIST 16x16: wie V-1.",
    "V-3": "Fashion-MNIST: Suche vs. Median von 100 zufaelligen Clifford-Schaltungen (Replikation I-005 Q-1): "
           "Holm-p < 0,05 -> H1 / H0 (Gegenrichtung); Holm-p_TOST(+-0,3 pp) < 0,05 und Q-0 H1 -> H0 (gleich); sonst HU.",
    "V-B": "beschreibend je Datensatz: Arme, G_voll, suche - k0/zufallssuche/best100, Siegerfluch, Aufwand, "
           "Kaskade, gewaehlte Rate/L2.",
}


# ------------------------------------------------------------------ Statistik
def tost_p(d, delta) -> float:
    """TOST: H0 |mu| >= delta; p = max der beiden einseitigen t-Test-p. p < 0,05 <=> 90-%-KI in (-delta, delta)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return 1.0
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    if se == 0.0:
        return 0.0 if abs(m) < delta else 1.0
    p_unten = float(t_vert.sf((m + delta) / se, len(d) - 1))    # gegen mu <= -delta
    p_oben = float(t_vert.cdf((m - delta) / se, len(d) - 1))    # gegen mu >= +delta
    return max(p_unten, p_oben)


def waehle_l2(werte: dict) -> dict:
    """werte {l2: [Validierungs-acc]} -> Kriterium Mittel, Gleichstand -> groessere L2;
    Optimum am Rand -> 'erweitern' ('unten'/'oben')."""
    g = sorted(werte)
    krit = {l2: float(np.mean(werte[l2])) for l2 in g}
    i = max(range(len(g)), key=lambda j: (round(krit[g[j]], 12), j))
    aus = {"l2": g[i], "erweitern": None, "kriterium": {repr(l2): krit[l2] for l2 in g}}
    if i == 0:
        aus["erweitern"] = "unten"
    elif i == len(g) - 1:
        aus["erweitern"] = "oben"
    return aus


# ------------------------------------------------------------------ Arbeiter
def _daten(ds, s):
    from abteilungen.clifford import daten
    return daten(ds, s)                              # 6.000 Nicht-Test / 5.000 Test, 16x16


def _l2_punkt(a):
    """(ds, tuning_seed, l2-Liste, n_zufall) -> {l2: [val-acc je Schaltung]} (nur Nicht-Test-Daten)."""
    ds, s, l2s, n_zufall = a
    n = QUBITS_JE_DATENSATZ[ds]
    xtr, ytr, _, _ = _daten(ds, s)
    nv = len(ytr) // 4
    g = np.random.default_rng(L2_SEED + s)
    codes = [np.zeros((LAGEN, n), np.int64)] + [g.integers(0, 4, (LAGEN, n)) for _ in range(n_zufall)]
    basis = (normiere(xtr[nv:]), ytr[nv:], normiere(xtr[:nv]), ytr[:nv])
    return {l2: [_bewerte(c, n, (*basis, l2))[0] for c in codes] for l2 in l2s}


def _tuning_voll(a):
    """(ds, tuning_seed, lr, schritte, l2) -> {'acc' (eigener Kopf, Validierung), 'acc_kopf'}."""
    from abteilungen.clifford import _baue
    from labor.gemeinsam import trainiere
    ds, s, lr, schritte, l2 = a
    n = QUBITS_JE_DATENSATZ[ds]
    xtr, ytr, _, _ = _daten(ds, s)
    nv = len(ytr) // 4
    m, _ = _baue("voll", s, n)
    trainiere(m, xtr[nv:], ytr[nv:], schritte, lr, seed=s)
    with torch.no_grad():
        acc = float((m(xtr[:nv]).argmax(1) == ytr[:nv]).float().mean())
        th = m.theta.detach().clone()
    acc_kopf = (_bewerte(unitaer_winkel(th, n), n, (normiere(xtr[nv:]), ytr[nv:], normiere(xtr[:nv]), ytr[:nv], l2))[0]
                if bool(torch.isfinite(th).all()) else 0.0)
    return {"acc": acc, "acc_kopf": acc_kopf}


def _mess_seed(a):
    """(ds, seed, schritte, politur_max, n_zufall, schritte_voll, lr, l2, zufallssuche?) -> Zeile."""
    from abteilungen.clifford import _baue, runde_clifford
    from labor.gemeinsam import trainiere
    ds, s, schritte, politur_max, n_zufall, schritte_voll, lr, l2, mit_zsuche = a
    n, t0 = QUBITS_JE_DATENSATZ[ds], time.time()
    xtr, ytr, xte, yte = _daten(ds, s)
    z, bew, xs, _ = _gemeinsame_arme(s, n, (xtr, ytr, xte, yte), schritte, politur_max, n_zufall, l2)
    z["datensatz"], z["lr_voll"], z["kopf_l2"] = ds, lr, l2
    tv = time.time()
    m, _ = _baue("voll", s, n)
    trainiere(m, xtr, ytr, schritte_voll, lr, seed=s)
    z["voll_sekunden"] = time.time() - tv
    with torch.no_grad():
        z["voll_eigener_kopf"] = float((m(xte).argmax(1) == yte).float().mean())
    th = m.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    endlich = bool(torch.isfinite(th).all())
    z["voll"] = _bewerte(unitaer_winkel(th, n), n, xs)[0] if endlich else 0.0
    k0 = kanonisch(codes_aus_winkeln(runde_clifford(th))) if endlich else np.zeros((LAGEN, n), np.int64)
    z["k0_codes"] = k0.tolist()
    z["k0"], konf, richtig = _bewerte(k0, n, xs)
    z["k0_kaskade"] = kaskaden_kennzahl(konf, richtig)
    z["k0_val"] = bew(k0)[0]
    if mit_zsuche:
        bz, sz = zufallssuche(bew, np.random.default_rng(v1.ZSUCHE_SEED + s), z["suche_bewertungen"], LAGEN, n)
        z["zufallssuche"], z["zufallssuche_val"] = _bewerte(bz, n, xs)[0], sz[0]
    z["sekunden"] = time.time() - t0
    return z


def _arbeiter(a):
    art, arg = a
    return art, (v1._struktur_seed(arg) if art == "struktur" else _mess_seed(arg))


# ------------------------------------------------------------------ Tuning
def _l2_suche(datensaetze, tuning, n_zufall, gitter=L2_GITTER, runden=L2_RUNDEN):
    gitter = {ds: sorted(gitter) for ds in datensaetze}
    werte = {ds: {} for ds in datensaetze}
    info, offen = {}, set(datensaetze)
    for runde in range(1, runden + 1):
        aufg = [(ds, s, [l for l in gitter[ds] if l not in werte[ds]], n_zufall)
                for ds in sorted(offen) for s in tuning[ds]]
        for (ds, *_), e in zip(aufg, parallel(_l2_punkt, aufg)):
            for l2, v in e.items():
                werte[ds].setdefault(l2, []).extend(v)
        for ds in sorted(offen):
            w = waehle_l2(werte[ds])
            if w["erweitern"] and runde < runden:
                g = gitter[ds]
                gitter[ds] = [g[0] / L2_FAKTOR] + g if w["erweitern"] == "unten" else g + [g[-1] * L2_FAKTOR]
            else:
                info[ds] = {**w, "am_rand": w["erweitern"] is not None, "runden": runde}
                offen.discard(ds)
        if not offen:
            break
    return info


def _lr_aufgaben(tuning, schritte_voll, l2):
    def aufgaben_fuer(ds, lr):
        return [(ds, s, lr, schritte_voll, l2[ds]) for s in tuning[ds]]     # Seed an Stelle 1
    return aufgaben_fuer


def _pruefe_seeds(seeds, tuning, struktur, datensaetze):
    alle = [s for ds in datensaetze for s in list(seeds[ds]) + list(tuning[ds])] + list(struktur)
    return len(set(alle)) == len(alle) and not (set(alle) & _BELEGT)


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=N_SEEDS, seeds=None, seeds_struktur=SEEDS_STRUKTUR, tuning_seeds=None,
         datensaetze=DATENSAETZE, schritte=v1.SA_SCHRITTE, politur_max=v1.POLITUR_MAX, n_zufall=v1.N_ZUFALL,
         schritte_voll=SCHRITTE_VOLL, zufallssuche_arm=True, mit_struktur=True):
    datensaetze = tuple(datensaetze)
    l2_gitter, l2_zufall = L2_GITTER, L2_ZUFALL
    if schnell:                                                   # nur Entwurfs-Seeds
        seeds, tuning_seeds, seeds_struktur = ENTWURF_SEEDS, ENTWURF_TUNING, ENTWURF_STRUKTUR
        schritte, politur_max, n_zufall, schritte_voll = 80, 20, 5, 40
        l2_gitter, l2_zufall = (1e-3, 1e-2, 1e-1), 2
    else:
        if n_seeds not in (10, 20):
            raise ValueError("n_seeds ist vorregistriert: 10 oder 20")
        seeds = seeds or {ds: SEEDS_LANG[ds][:n_seeds] for ds in datensaetze}
        tuning_seeds = tuning_seeds or TUNING_SEEDS
        if not _pruefe_seeds(seeds, tuning_seeds, seeds_struktur if mit_struktur else (), datensaetze):
            raise ValueError("Seeds nicht frisch oder nicht disjunkt")
    seeds = {ds: list(seeds[ds]) for ds in datensaetze}
    t0 = time.time()

    # Phase 1: Kopf-L2 je Datensatz (Tuning-Seeds)
    l2_info = _l2_suche(datensaetze, tuning_seeds, l2_zufall, gitter=l2_gitter)
    l2 = {ds: l2_info[ds]["l2"] for ds in datensaetze}
    print(f"  Phase 1 (Kopf-L2) nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{NAME[ds]} {l2[ds]:.2g}{' AM RAND' if l2_info[ds]['am_rand'] else ''}" for ds in datensaetze),
          flush=True)

    # Phase 2: Lernrate von voll je Datensatz mit Stabilitaetsabstand (Regel 1)
    lr_info, lr_roh = gittersuche_stabil(list(datensaetze), _lr_aufgaben(tuning_seeds, schritte_voll, l2),
                                         lambda aufg: parallel(_tuning_voll, aufg), gitter=LR_GITTER,
                                         faktor=LR_FAKTOR, kollaps=KOLLAPS)
    lr = {ds: lr_info[ds]["lr"] for ds in datensaetze}
    for (ds, lr_k), je_seed in lr_roh.items():
        eintrag = lr_info[ds]["gitter"].get(str(lr_k))
        if eintrag is not None:
            eintrag["acc_kopf"] = [je_seed[s]["acc_kopf"] for s in sorted(je_seed)]
    print(f"  Phase 2 (Rate voll) nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{NAME[ds]} {lr[ds]:g} (abgestuft {lr_info[ds]['abgestuft']}"
                      f"{', AM RAND' if lr_info[ds]['am_rand'] else ''})" for ds in datensaetze), flush=True)

    # Phase 3: Messung (lange Seeds zuerst, Strukturseeds fuellen die Luecken)
    aufg = [("mess", (ds, s, schritte, politur_max, n_zufall, schritte_voll, lr[ds], l2[ds], zufallssuche_arm))
            for ds in datensaetze for s in seeds[ds]]
    if mit_struktur:
        aufg += [("struktur", (s, schritte, politur_max, n_zufall)) for s in seeds_struktur]
    erg = parallel(_arbeiter, aufg)
    zs = sorted((z for art, z in erg if art == "struktur"), key=lambda z: z["seed"])
    zeilen = {ds: sorted((z for art, z in erg if art == "mess" and z["datensatz"] == ds), key=lambda z: z["seed"])
              for ds in datensaetze}
    for z in zs:
        print(f"  Struktur {z['seed']}: suche {z['suche']:.4f} lehrer {z['lehrer']:.4f} id {z['identitaet']:.4f} "
              f"aequivalent {z['lehrer_aequivalent']}", flush=True)
    for ds in datensaetze:
        for z in zeilen[ds]:
            print(f"  {NAME[ds]} {z['seed']}: suche {z['suche']:.4f} voll {z['voll']:.4f} (eigener Kopf "
                  f"{z['voll_eigener_kopf']:.4f}) zufall-median {z['zufall_median']:.4f} id {z['identitaet']:.4f} "
                  f"k0 {z['k0']:.4f} ({z['suche_bewertungen']} Bew., {z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "Clifford", "modul": "clifford_suche_v2", "karte": "I-005-v2", "version": 1,
         "hypothesen": HYPOTHESEN, "seeds": seeds, "seeds_struktur": list(seeds_struktur) if mit_struktur else [],
         "tuning_seeds": {ds: list(tuning_seeds[ds]) for ds in datensaetze}, "n_seeds": len(seeds[datensaetze[0]]),
         "lr_voll": lr_info, "kopf_l2": l2_info,
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "tau_start": v1.TAU_START, "tau_ende": v1.TAU_ENDE, "ridge_lambda": v1.RIDGE_LAMBDA,
                           "schritte_voll": schritte_voll, "lr_gitter": LR_GITTER, "kollaps": KOLLAPS,
                           "l2_gitter": list(l2_gitter), "tost_v": TOST_V, "tost_v3": TOST_V3, "sens_min": SENS_MIN},
         "zeilen": zeilen, "zeilen_struktur": zs, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, zs)
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Auswertung
def _ds_auswertung(zz) -> dict:
    arr = lambda k: np.array([z[k] for z in zz], dtype=float)
    su, vo, id_, med, k0 = arr("suche"), arr("voll"), arr("identitaet"), arr("zufall_median"), arr("k0")
    eig = arr("voll_eigener_kopf")
    grenze = float(np.nanmedian(eig)) - KOLLAPS
    r = {"n": len(zz),
         "suche_minus_voll": {**_gepaart(su - vo), "p_tost": tost_p(su - vo, TOST_V)},
         "suche_minus_median": {**_gepaart(su - med), "p_tost": tost_p(su - med, TOST_V3)},
         "sensitivitaet": {"mittel": float((vo - id_).mean()), "p": wilcoxon_exakt(vo - id_)},
         "kollaps_voll": [z["seed"] for z, v in zip(zz, eig) if not v >= grenze],
         "lr_voll": zz[0].get("lr_voll"), "kopf_l2": zz[0].get("kopf_l2")}
    r["sensitiv"] = bool(r["sensitivitaet"]["mittel"] >= SENS_MIN and r["sensitivitaet"]["p"] < ALPHA)
    b = {arm: zusammenfassung(arr(arm)) for arm in ("suche", "voll", "voll_eigener_kopf", "identitaet",
                                                    "zufall_median", "zufall_best100", "k0", "zufallssuche")
         if arm in zz[0]}
    gv = float((vo - id_).mean())
    b["G_voll_suche"] = float((su - id_).mean()) / gv if gv > 0 else None
    b["G_voll_k0"] = float((k0 - id_).mean()) / gv if gv > 0 else None
    b["suche_minus_k0"] = _gepaart(su - k0)
    b["suche_minus_best100"] = _gepaart(su - arr("zufall_best100"))
    if "zufallssuche" in zz[0]:
        b["suche_minus_zufallssuche"] = _gepaart(su - arr("zufallssuche"))
    b["siegerfluch"] = float(np.mean([(z["suche_val"] - float(np.median(z["zufall_val"])))
                                      - (z["suche"] - z["zufall_median"]) for z in zz]))
    for k in ("suche_bewertungen", "suche_sekunden", "voll_sekunden", "sekunden"):
        b[k] = float(np.mean([z[k] for z in zz]))
    for arm in ("suche", "k0"):
        b[f"{arm}_anteil_sicher"] = float(np.mean([z[f"{arm}_kaskade"]["anteil_sicher"] for z in zz]))
    r["beschreibend"] = b
    return r


def auswertung(zeilen: dict, zs) -> dict:
    a = {"datensaetze": {ds: _ds_auswertung(zz) for ds, zz in zeilen.items() if zz}, "n_struktur": len(zs)}
    if zs:
        a["Q0"] = v1.auswertung([], zs)["Q0"]
    # Familie immer mit genau drei Gliedern: fehlender Datensatz -> p = 1 (konservativ)
    p, pt = {}, {}
    for h, (ds, k) in FAMILIE.items():
        d = a["datensaetze"].get(ds, {}).get(k)
        p[h], pt[h] = (d["p"], d["p_tost"]) if d else (1.0, 1.0)
    a["holm"], a["holm_tost"] = holm(p), holm(pt)
    return a


def _zahlen(ds, d, ph, pth, delta):
    return (f"{NAME[ds]}: d = {d['mittel'] * 100:+.2f} pp, 90-%-KI [{d['ki_unten'] * 100:+.2f}; "
            f"{d['ki_oben'] * 100:+.2f}], TOST(+-{delta * 100:.1f} pp) p = {d['p_tost']:.4f}, Holm-p_TOST = "
            f"{pth:.4f}, Wilcoxon p = {d['p']:.4f}, Holm-p = {ph:.4f}, {d.get('positiv', 0)}/{d['n']} Seeds positiv")


def _urteil_voll(h, ds, a, q0_h1) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    d, ph, pth = r["suche_minus_voll"], a["holm"][h], a["holm_tost"][h]
    s = r["sensitivitaet"]
    t = (_zahlen(ds, d, ph, pth, TOST_V) + f"; Sensitivitaet voll - id {s['mittel'] * 100:+.2f} pp (p = {s['p']:.4f}); "
         f"lr_voll {r['lr_voll']}, Kopf-L2 {r['kopf_l2']}; voll kollabiert: {r['kollaps_voll'] or 'keiner'}")
    if pth < ALPHA:
        if not q0_h1:
            return "HU (Positivkontrolle): gleich im Rahmen +-0,5 pp, aber Q-0 nicht H1 -- " + t
        if not r["sensitiv"]:
            return (f"HU (Sensitivitaet): gleich im Rahmen +-0,5 pp, aber voll schlaegt die Identitaet nicht um "
                    f">= {SENS_MIN * 100:.1f} pp nachweisbar -- " + t)
        if r["kollaps_voll"]:
            return "HU (Instabilitaet): gleich im Rahmen +-0,5 pp, aber voll auf Mess-Seeds kollabiert -- " + t
        extra = " (Unterschied zusaetzlich nachweisbar, aber kleiner als die Grenze)" if ph < ALPHA else ""
        return f"GLEICHWERTIG: Clifford-Suche (0 Magiewinkel) = volles Training im Rahmen +-0,5 pp{extra} -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return "VOLL_BESSER: die Magiewinkel bringen etwas, das die Clifford-Suche nicht erreicht -- " + t
    if ph < ALPHA and d["mittel"] > 0:
        if r["kollaps_voll"]:
            return "HU (Instabilitaet): Suche besser, aber voll auf Mess-Seeds kollabiert -- " + t
        return "SUCHE_BESSER: die Clifford-Suche schlaegt das volle Training -- " + t
    return "HU: " + t


def _urteil_zufall(h, ds, a, q0_h1) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    d, ph, pth = r["suche_minus_median"], a["holm"][h], a["holm_tost"][h]
    t = _zahlen(ds, d, ph, pth, TOST_V3)
    if ph < ALPHA and d["mittel"] > 0:
        return "H1: Suche besser als der Median von 100 Zufalls-Clifford (Q-1 repliziert) -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return "H0 (Gegenrichtung): Zufalls-Clifford besser als die Suche -- " + t
    if pth < ALPHA:
        if q0_h1:
            return "H0 (gleich): Suche = Median der Zufalls-Clifford im Rahmen +-0,3 pp (TOST) -- " + t
        return "HU (Positivkontrolle): gleich im Rahmen +-0,3 pp, aber Q-0 nicht H1 -- " + t
    return "HU: " + t


def urteil(a: dict) -> dict:
    u = {"Q-0": (v1.urteil({"Q0": a["Q0"], "n_struktur": a["n_struktur"]})["Q-0"] if "Q0" in a
                 else "HU (Vorbedingung): keine Strukturseeds")}
    q0_h1 = u["Q-0"].startswith("H1")
    u["V-1"] = _urteil_voll("V-1", "mnist", a, q0_h1)
    u["V-2"] = _urteil_voll("V-2", "fashion_mnist", a, q0_h1)
    u["V-3"] = _urteil_zufall("V-3", "fashion_mnist", a, q0_h1)
    teile = []
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for ds, r in a["datensaetze"].items():
        b = r["beschreibend"]
        t = (f"{NAME[ds]}: suche {b['suche']['mittel']:.4f}, voll {b['voll']['mittel']:.4f} (eigener Kopf "
             f"{b['voll_eigener_kopf']['mittel']:.4f}), k0 {b['k0']['mittel']:.4f}, zufall-median "
             f"{b['zufall_median']['mittel']:.4f}, best100 {b['zufall_best100']['mittel']:.4f}, id "
             f"{b['identitaet']['mittel']:.4f}")
        if "zufallssuche" in b:
            t += (f", Zufallssuche {b['zufallssuche']['mittel']:.4f} (suche - zs "
                  f"{b['suche_minus_zufallssuche']['mittel'] * 100:+.2f} pp, p = {b['suche_minus_zufallssuche']['p']:.4f})")
        t += (f"; G_voll suche {fmt(b['G_voll_suche'])}, k0 {fmt(b['G_voll_k0'])}; suche - median "
              f"{r['suche_minus_median']['mittel'] * 100:+.2f} pp (p = {r['suche_minus_median']['p']:.4f}); suche - k0 "
              f"{b['suche_minus_k0']['mittel'] * 100:+.2f} pp; Siegerfluch {b['siegerfluch'] * 100:+.2f} pp; "
              f"{b['suche_bewertungen']:.0f} Bew., Suche {b['suche_sekunden']:.0f} s vs voll {b['voll_sekunden']:.0f} s; "
              f"konf > 0,9: suche {b['suche_anteil_sicher'] * 100:.1f} %, k0 {b['k0_anteil_sicher'] * 100:.1f} %; "
              f"lr_voll {r['lr_voll']}, Kopf-L2 {r['kopf_l2']}")
        teile.append(t)
    u["V-B"] = "beschreibend: " + (" | ".join(teile) if teile else "keine Mess-Seeds")
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, d_voll=0.0, d_med=0.02, sd=0.001, sens=0.035, kollaps=0, n=10):
    """Synthetische Mess-Zeilen: d_voll = suche - voll, d_med = suche - zufall_median, sens = voll - id."""
    zeilen = []
    for i in range(n):
        vo = 0.934 + rng.normal(0, 0.002)
        su = vo + d_voll + rng.normal(0, sd)
        med = su - d_med + rng.normal(0, sd)
        zeilen.append({"seed": i, "suche": su, "voll": vo, "voll_eigener_kopf": vo - (0.2 if i < kollaps else 0.0),
                       "identitaet": vo - sens + rng.normal(0, 0.001), "zufall_median": med,
                       "zufall_best100": med + 0.004, "k0": vo - 0.03, "zufallssuche": su - 0.01,
                       "suche_val": su + 0.003, "zufall_val": [med] * 3, "suche_bewertungen": 5000,
                       "suche_sekunden": 500.0, "voll_sekunden": 130.0, "sekunden": 1300.0,
                       "suche_kaskade": {"anteil_sicher": 0.6, "acc_sicher": 0.98},
                       "k0_kaskade": {"anteil_sicher": 0.5, "acc_sicher": 0.97}, "lr_voll": 0.08, "kopf_l2": 0.01})
    return zeilen


def _vortest_urteile(ziehungen=5) -> bool:
    import json
    ok = True
    rng = np.random.default_rng(7)
    gut = lambda: v1._synth_struktur(rng, 0.925)
    schlecht = lambda: v1._synth_struktur(rng, 0.79)
    # (Name, Struktur, MNIST-Parameter, Fashion-Parameter, {Hypothese: (Marke, [Textteile])}, Toleranz)
    faelle = [
        ("alles gleich", gut, {}, {},
         {"Q-0": ("H1", []), "V-1": ("GLEICHWERTIG", ["MNIST", "0,5 pp", "Holm-p_TOST"]),
          "V-2": ("GLEICHWERTIG", ["Fashion", "Holm-p_TOST"]), "V-3": ("H1", ["Fashion", "Holm-p"])}, 0),
        ("Richtungen", gut, {"d_voll": -0.01}, {"d_voll": 0.01, "d_med": -0.01},
         {"V-1": ("VOLL_BESSER", ["MNIST", "Magiewinkel"]), "V-2": ("SUCHE_BESSER", ["Fashion"]),
          "V-3": ("H0 (Gegenrichtung)", ["Fashion"])}, 0),
        ("Streuung", gut, {"sd": 0.02}, {"sd": 0.02, "d_med": 0.0},
         {"V-1": ("HU:", ["MNIST"]), "V-2": ("HU:", ["Fashion"]), "V-3": ("HU:", ["Fashion"])}, 1),
        ("V-3 gleich", gut, {}, {"d_med": 0.0, "sd": 0.0005},
         {"V-2": ("GLEICHWERTIG", ["Fashion"]), "V-3": ("H0 (gleich)", ["0,3 pp", "Fashion"])}, 1),
        ("Positivkontrolle", schlecht, {}, {"d_med": 0.0, "sd": 0.0005},
         {"Q-0": ("H0", []), "V-1": ("HU (Positivkontrolle)", ["MNIST", "Q-0"]),
          "V-2": ("HU (Positivkontrolle)", ["Fashion"]), "V-3": ("HU (Positivkontrolle)", ["Fashion"])}, 1),
        ("Sensitivitaet", gut, {"sens": 0.003}, {"sens": 0.003},
         {"V-1": ("HU (Sensitivitaet)", ["MNIST", "Identitaet"]), "V-2": ("HU (Sensitivitaet)", ["Fashion"]),
          "V-3": ("H1", [])}, 0),
        ("Kollaps", gut, {"kollaps": 1}, {"kollaps": 1, "d_voll": 0.01},
         {"V-1": ("HU (Instabilitaet)", ["MNIST", "kollabiert: [0]"]),
          "V-2": ("HU (Instabilitaet)", ["Suche besser", "Fashion"])}, 0),
        ("klein, aber gleich", gut, {"d_voll": 0.002, "sd": 0.0003}, {"d_voll": -0.002, "sd": 0.0003},
         {"V-1": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"]), "V-2": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"])}, 0),
        ("nur MNIST", gut, {}, None,
         {"V-1": ("GLEICHWERTIG", ["MNIST"]), "V-2": ("HU: keine Fashion", []), "V-3": ("HU: keine Fashion", [])}, 0),
        ("ohne Struktur", None, {}, {},
         {"Q-0": ("HU (Vorbedingung)", []), "V-1": ("HU (Positivkontrolle)", ["MNIST"])}, 0),
    ]
    for name, mach_s, pm, pf, erwartet, tol in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            zeilen = {"mnist": _synth(rng, **pm)}
            if pf is not None:
                zeilen["fashion_mnist"] = _synth(rng, **pf)
            a = auswertung(zeilen, mach_s() if mach_s else [])
            u = urteil(a)
            for h, (marke, teile) in erwartet.items():
                if not u[h].startswith(marke) or any(t not in u[h] for t in teile):
                    fehl[h] += 1
                    print(f"  Vortest [{name}]: {h} -> {u[h][:140]}")
            if not u["V-B"].startswith("beschreibend") or not all(isinstance(v, str) for v in u.values()):
                ok = False
                print(f"  VORTEST FEHLER [{name}]: V-B oder Urteil kein String")
            try:
                json.dumps({"auswertung": a, "urteil": u})
            except TypeError as err:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: nicht JSON-serialisierbar", err)
        for h, f in fehl.items():
            if f > tol:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: {h} {f}/{ziehungen} falsch")
    # grosse Streuung, wahre Differenz 0 -> NIE GLEICHWERTIG (Toleranz 0)
    for _ in range(ziehungen):
        u = urteil(auswertung({"mnist": _synth(rng, sd=0.02), "fashion_mnist": _synth(rng, sd=0.02)}, gut()))
        if u["V-1"].startswith("GLEICHWERTIG") or u["V-2"].startswith("GLEICHWERTIG"):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung")
    return ok


def _vortest_bausteine() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(3)
    # (a) TOST-p < 0,05 <=> unadjustiertes 90-%-KI in (-delta, delta)
    for _ in range(200):
        d = rng.normal(rng.normal(0, 0.003), rng.uniform(0.0005, 0.008), 10)
        g, p = _gepaart(d), tost_p(d, TOST_V)
        if (p < ALPHA) != (-TOST_V < g["ki_unten"] and g["ki_oben"] < TOST_V):
            fehler(f"TOST-p passt nicht zum KI ({p}, {g['ki_unten']}, {g['ki_oben']})")
            break
    if not tost_p(np.full(10, 0.0) + rng.normal(0, 1e-5, 10), TOST_V) < 1e-6 \
            or not tost_p(TOST_V + rng.normal(0, 1e-4, 10), TOST_V) > 0.3:
        fehler("TOST-p an den Enden falsch")
    # (b) Holm (aus I-005) und Familie genau drei Glieder
    h = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    if abs(h["a"] - 0.03) > 1e-12 or abs(h["b"] - 0.08) > 1e-12 or set(FAMILIE) != {"V-1", "V-2", "V-3"}:
        fehler(f"Holm/Familie {h}")
    # (c) L2-Regel: inneres Maximum, Rand -> erweitern, Gleichstand -> groessere L2
    if waehle_l2({1e-3: [0.5], 1e-2: [0.6], 1e-1: [0.55]})["l2"] != 1e-2 \
            or waehle_l2({1e-3: [0.7], 1e-2: [0.6], 1e-1: [0.55]})["erweitern"] != "unten" \
            or waehle_l2({1e-3: [0.5], 1e-2: [0.6], 1e-1: [0.65]})["erweitern"] != "oben" \
            or waehle_l2({1e-4: [0.5], 1e-3: [0.6], 1e-2: [0.6], 1e-1: [0.5]})["l2"] != 1e-2:
        fehler("L2-Regel")
    # (d) Seeds frisch, disjunkt, I-005-Seeds und Entwurfs-Seeds belegt
    for n in (10, 20):
        if not _pruefe_seeds({ds: SEEDS_LANG[ds][:n] for ds in DATENSAETZE}, TUNING_SEEDS, SEEDS_STRUKTUR, DATENSAETZE):
            fehler(f"Seeds (n = {n}) nicht frisch/disjunkt")
    if not (set(v1.SEEDS_MNIST) | set(v1.SEEDS_STRUKTUR)) <= _BELEGT \
            or _pruefe_seeds({"mnist": (190,), "fashion_mnist": (310,)}, TUNING_SEEDS, (), DATENSAETZE) \
            or _pruefe_seeds({"mnist": (300,), "fashion_mnist": (300,)}, TUNING_SEEDS, (), DATENSAETZE):
        fehler("Seed-Pruefung laesst belegte/doppelte Seeds durch")
    entwurf = {s for d in (ENTWURF_SEEDS, ENTWURF_TUNING) for v in d.values() for s in v} | set(ENTWURF_STRUKTUR)
    mess = {s for v in SEEDS_LANG.values() for s in v} | {s for v in TUNING_SEEDS.values() for s in v} | set(SEEDS_STRUKTUR)
    if entwurf & mess:
        fehler("Entwurfs- und Mess-/Tuning-Seeds ueberschneiden sich")
    # (e) Ratensuche: Seed an Stelle 1 (gittersuche_stabil), L2 je Datensatz durchgereicht
    auf = _lr_aufgaben(TUNING_SEEDS, 1500, {"mnist": 0.01, "fashion_mnist": 0.003})("fashion_mnist", 0.08)
    if [a[1] for a in auf] != list(TUNING_SEEDS["fashion_mnist"]) or any(a[4] != 0.003 or a[2] != 0.08 for a in auf):
        fehler(f"Tuning-Aufgaben falsch: {auf}")
    # (f) Kollaps: nicht endlich oder < Median - 5 pp wird erkannt, sonst nicht
    z = _synth(rng)
    z[3]["voll_eigener_kopf"] = float("nan")
    z[5]["voll_eigener_kopf"] -= 0.06
    z[6]["voll_eigener_kopf"] -= 0.03
    if _ds_auswertung(z)["kollaps_voll"] != [3, 5]:
        fehler(f"Kollaps-Erkennung {_ds_auswertung(z)['kollaps_voll']}")
    # (g) Rauchtest-Seeds werden in lauf(schnell=False) nie benutzt; n_seeds ausserhalb 10/20 abgelehnt
    try:
        lauf(n_seeds=15)
        fehler("n_seeds = 15 nicht abgelehnt")
    except ValueError:
        pass
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Statistik-Bausteine (TOST, Holm, L2-Regel, Seeds, Tuning-Aufgaben, Kollaps) +
    jeden Urteilszweig synthetisch erzwingen (Marke UND Text, mehrere Ziehungen). Die Such-Bausteine
    selbst prueft der Vortest von clifford_suche (I-005), aus dem sie unveraendert importiert sind."""
    ok = _vortest_bausteine()
    return _vortest_urteile(ziehungen) and ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-005-v2" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
