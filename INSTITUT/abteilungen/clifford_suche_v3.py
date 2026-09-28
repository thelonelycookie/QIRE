"""Abteilung Clifford -- Wie viele Magiewinkel AUF der gesuchten Clifford-Schaltung schliessen die
Fashion-Luecke? (Karte I-005-v3, Quantenstufe Q3)

HEILMEIER:
  Was wollen wir?  I-005-v2: auf Fashion-MNIST 16x16 holt die beste Clifford-Schaltung aus der
      diskreten Suche (0 Magiewinkel) nur G_voll = 0,65 des Schaltungsgewinns; volles Training ist
      1,31 pp besser (VOLL_BESSER, 20/20). Frage: wie viele Nicht-Clifford-Winkel braucht man
      ZUSAETZLICH auf dieser Schaltung, um die Luecke zu schliessen?
  Wie heute?  Magie-Konzentration (C-fashion/C-referenz: M90 = 4) wurde immer vom GRADIENTEN-
      Startpunkt aus gemessen (voll trainieren, dann runden). I-004/I-004-mnist: zufaelliges
      Clifford-Geruest + k Winkel. Ein GESUCHTES Geruest + k polierte Winkel gab es noch nicht.
  Was ist neu?  M50_S/M90_S -- Magiebedarf ueber der besten Clifford-Schaltung, unabhaengig vom
      Gradienten-Startpunkt: G_S(k) = (acc(k) - acc(suche)) / (acc(voll) - acc(suche)).
  Woran merken wir es?  M-1: k = 8 (nach Verlust) = voll (TOST +-0,5 pp) oder nicht; M-2: die
      Verlust-Auswahl schlaegt die zufaellige bei k = 2; M-3: voll schlaegt suche (Sensitivitaet).

VERFAHREN (Suche, voll, Kopf unveraendert importiert aus clifford_suche / clifford_suche_v2):
  Nur Fashion-MNIST 16x16 (8 Qubits, 4 Lagen = 32 Winkel), 6.000 Nicht-Test / 5.000 Test je Seed.
  MNIST weggelassen (Laufzeit; auf MNIST gibt es keine Luecke zu schliessen, V-1 GLEICHWERTIG).
  Je Mess-Seed: v2._mess_seed ohne Zufallssuche -> suche (SA + Politur, Ridge-LOO), 100 Zufalls-
  Clifford, identitaet, voll (Adam, 1.500 Schritte, getunte Rate), voll_eigener_kopf, k0. Dann:
  NACHTRAINING (Arm k{k}_{r}): QireZelle mit allen Winkeln auf den Suche-Werten (codes * pi/2),
    genau k Winkel frei. Phase A: Kopf allein (Adam, auf den festen Born-Merkmalen der Suche-
    Schaltung, gleiche Rate und Schrittzahl) -- sonst zoege ein zufaelliger Kopf die freien Winkel
    in der ersten Phase irgendwohin; Phase B: k Winkel + Kopf gemeinsam (Adam, 1.500 Schritte wie
    voll). Urteil mit DEMSELBEN logistischen L-BFGS-Kopf wie suche und voll (fair); der Adam-Kopf
    steht als *_eigener_kopf daneben (Kollaps-Pruefung).
  RANGFOLGE „verlust“ (Anpassung von I-001 an ein Clifford-Geruest): Arm k32_alle = alle 32
    Winkel frei ab der Suche-Schaltung (sonst wie oben). Je Winkel i: Anstieg des Trainings-
    verlusts (Nicht-Testdaten, trainierter Adam-Kopf fest), wenn NUR Winkel i auf seinen Suche-
    Wert zurueckgesetzt wird (magie_pruning._ce). Groesster Anstieg zuerst. In I-001 wurde auf das
    naechste Clifford-Gitter gerundet; hier ist der Suche-Wert das Geruest, auf dem die anderen
    Winkel in den k-Armen stehen bleiben (nach Politur kann der naechste Gitterpunkt ein anderer
    sein). Kontrolle „zufall“: torch.randperm(32), Generator RANG_SEED + seed. Die Mengen sind
    geschachtelt (die k freien Winkel = erste k der Rangfolge). k32_alle ist beschreibend (G_S(32)).
  k in {1, 2, 4, 8}, Rangfolgen {verlust, zufall} -> 8 Nachtrain-Arme + k32_alle je Seed.

TUNING (nur Tuning-Seeds; Validierung = erstes Viertel der Nicht-Test-Bilder, trainiert/gesucht
  auf den anderen drei Vierteln; Lernrate JE ARM, Regel 1):
  1. Kopf-L2: v2._l2_suche (feste Regel aus I-005-v2), Tuning-Seeds dieses Moduls.
  2. Suche auf den Tuning-Seeds (gleiches Verfahren, nur Trainingsteil) -> Geruest je Tuning-Seed.
  3. Rate von voll UND k32_alle: labor.gemeinsam.gittersuche_stabil (Gitter 0,02..0,32, Faktor 2,
     Kollaps 5 pp, Kriterium schlechtester Tuning-Seed, Stabilitaetsabstand); Kriterium =
     Validierungsgenauigkeit des EIGENEN Adam-Kopfes (reagiert auf Kollaps).
  4. Verlust-Rangfolge je Tuning-Seed aus dem k32_alle-Tuninglauf mit der GEWAEHLTEN Rate.
  5. Rate je Nachtrain-Arm (8 Schluessel k{k}_{r}) mit gittersuche_stabil wie in 3. Verlust- und
     Zufalls-Arm bekommen je EIGENE Rate (M-2 fair: keine Rate auf die Verlust-Auswahl zugeschnitten).

SEEDS (28.09. geprueft gegen ergebnisse/*.json, agenda.json, ideen.json und abteilungen/*.py --
  frisch; I-005-v2 belegt 300-349, 360-369, 950-955, 970-981; lauf() und Vortest pruefen es):
  Mess-Seeds Fashion 420-439 (n_seeds = 20; Option 10 -> 420-429)
  Tuning              960-962
  Rauchtest           Mess 963/964, Tuning 965/966 (Entwurfs-Seeds, nie Mess-Seeds)

SEED-ZAHL (vor dem Lauf aus den v2-Rohdaten ergebnisse/I-005-v2__20260928_203403.json abgeschaetzt):
  v2 Fashion d = suche - voll: Mittel -1,31 pp, sd 0,24 pp (n = 20; ohne die kollabierten voll-Seeds
  314/315: sd 0,22 pp). sd(k8 - voll) ist unbekannt: k8 bringt eine eigene Trainingsstreuung dazu,
  plausibel 0,24-0,5 pp. TOST +-0,5 pp, Holm ueber 3 (kleinstes p_TOST muss < 0,0167 sein;
  t_0,983;9 ~ 2,52, t_0,983;19 ~ 2,33). Simulierte Macht fuer GLEICHWERTIG (20.000 Ziehungen):
       sd(d)   wahres d      n = 10   n = 20
       0,24    0 / -0,2 / -0,3 pp   1,00 / 0,90 / 0,57   1,00 / 1,00 / 0,92
       0,35    0 / -0,2 / -0,3 pp   0,93 / 0,60 / 0,29   1,00 / 0,93 / 0,61
       0,50    0 / -0,2 / -0,3 pp   0,50 / 0,29 / 0,15   0,96 / 0,65 / 0,33
  -> N_SEEDS = 20: bei realistischer Streuung (0,35 pp) und einer Restluecke von 0,2 pp (k8 schliesst
  85 % der Luecke) Macht 0,93 statt 0,60. Unterschiedstests: kleinstes exaktes Wilcoxon-p bei n = 20
  ~1e-6, Holm x3 unkritisch. Kosten: +1 h Laufzeit gegenueber n = 10.

VORREGISTRIERT (vor der ersten Zahl der Mess-Seeds; Holm ueber GENAU die Familie {M-1, M-2, M-3},
  getrennt fuer Unterschied (exakter zweiseitiger Wilcoxon) und Aequivalenz (TOST, p = max der zwei
  einseitigen t-Tests); dazu das unadjustierte 90-%-KI, beschreibend):
  M-1 d_s = acc(k8_verlust) - acc(voll), delta = 0,5 pp.
        Holm-p_TOST < 0,05: Sensitivitaet verfehlt -> HU (Sensitivitaet); ein beteiligter Arm auf
          >= 1 Mess-Seed kollabiert -> HU (Instabilitaet); sonst GLEICHWERTIG.
        Holm-p < 0,05, mittel < 0 -> VOLL_BESSER (8 Magiewinkel auf der Suche schliessen die Luecke
          NICHT); mittel > 0 -> K8_BESSER.
        sonst HU.
      Sensitivitaet (nicht in der Familie): mittel(voll - suche) >= 2*delta = 1 pp und exakter
        Wilcoxon p < 0,05 -- sonst gibt es keine Luecke, die k8 schliessen koennte (v2: 1,31 pp).
  M-2 d_s = acc(k2_verlust) - acc(k2_zufall). Holm-p < 0,05: mittel > 0 -> H1, < 0 -> H0
      (Gegenrichtung); Holm-p_TOST(+-0,3 pp) < 0,05 -> H0 (gleich); sonst HU. k = 2 wie I-001 P-2
      (dort H1). delta 0,3 pp ~ 23 % der erwarteten Luecke, wie I-005 Q-1/V-3.
  M-3 d_s = acc(voll) - acc(suche) (Positivkontrolle/Sensitivitaet, repliziert V-2 auf frischen
      Seeds). Holm-p < 0,05: mittel > 0 -> H1, < 0 -> H0 (Gegenrichtung); Holm-p_TOST(+-0,5 pp)
      < 0,05 -> H0 (gleich); sonst HU.
  KOLLAPS IN JEDEM ZWEIG (Lehre I-005-v2: VOLL_BESSER pruefte ihn nicht): Arm kollabiert auf einem
    Mess-Seed, wenn sein eigener Adam-Kopf < Median(eigener Kopf dieses Arms) - 5 pp liegt oder die
    Winkel nicht endlich sind (suche ist diskret und kann nicht kollabieren). Beteiligte Arme:
    M-1 {k8_verlust, voll}, M-2 {k2_verlust, k2_zufall}, M-3 {voll}.
      Aequivalenz-Zweige (GLEICHWERTIG, H0 (gleich)): jeder Kollaps -> HU (Instabilitaet) (wie v2).
      Unterschieds-Zweige (VOLL_BESSER, K8_BESSER, H1, H0 (Gegenrichtung)): das Urteil gilt nur, wenn
        die Auswertung OHNE die kollabierten Seeds (gepaart ausgeschlossen, Holm mit den bereinigten
        p aller drei Glieder) denselben Zweig ergibt; sonst HU (Instabilitaet). Der Text nennt die Seeds.
  M-B beschreibend: G_S(k) je Rangfolge (k = 0, 1, 2, 4, 8, 32; Seed-Mittel ueber die Seeds, auf
    denen voll nicht kollabiert ist -- Nenner geschuetzt), M50_S/M90_S = kleinstes k in {0,1,2,4,8}
    mit G_S >= 0,5 / 0,9 (sonst „> 8“), T-Gatter nach C-hardware (kaskade.T_JE_K), Arme, eigene
    Koepfe, Abstand der freien Winkel zum Gitter, Ueberlappung verlust/zufall, G_voll(suche),
    gewaehlte Raten/L2, Tuning-Gitter, Kollaps-Listen, Aufwand.

LAUFZEIT (Schaetzung aus v2 Fashion: je Seed 1.254 s = Suche 442 s + Zufallssuche ~440 s + 100
  Koepfe/k0 ~250 s + voll 120 s; INSTITUT_PROZESSE = 5):
  je Mess-Seed: v2-Seed ohne Zufallssuche ~815 s + 9 Nachtrain-Arme (k32 + 8) x ~125 s ~1.125 s
    -> ~1.950 s; 20 Seeds / 5 Prozesse = 4 Wellen -> ~7.800 s ~ 2 h 10 min.
  Tuning: L2 ~3 min; Suche auf 3 Tuning-Seeds ~7 min; Rate voll + k32 (30 Laeufe x ~95 s / 5)
    ~10-13 min; Rate der 8 Nachtrain-Arme (120 Laeufe x ~100 s / 5) ~40-50 min mit Verlaengerungen.
  GESAMT n = 20: ~3 h 10 min - 3 h 40 min (n = 10: ~2 h 10 min).

GRENZEN (vorab benannt): Test 5.000 Bilder (0,02 pp je Bild). Suche und Rangfolge sehen den Test
  nie. Die Verlust-Rangfolge ist eine Einzelwinkel-Naeherung (Wechselwirkungen ignoriert, wie I-001).
  Die k freien Winkel landen nicht zwingend weit weg vom Gitter (Abstand wird berichtet); T-Gatter
  nach C-hardware unterstellen echte Nicht-Clifford-Winkel. Nur Fashion: M50_S ist ein Befund ueber
  EINE Aufgabe.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen import clifford_suche_v2 as v2
from abteilungen.clifford import GITTER, LAGEN, QUBITS_JE_DATENSATZ, QireZelle
from abteilungen.clifford_suche import (Bewerter, _bewerte, _gepaart, holm, kanonisch, normiere, suche,
                                        unitaer_clifford, unitaer_winkel, zufall_codes)
from labor.gemeinsam import gittersuche_stabil, parallel, speichere, trainiere, zusammenfassung

DS = "fashion_mnist"
N = QUBITS_JE_DATENSATZ[DS]
K_ALLE = LAGEN * N                                     # 32
SEEDS_LANG = tuple(range(420, 440))
N_SEEDS = 20                                           # Vorregistrierung: 20 (Macht-Tabelle im Kopf)
TUNING_SEEDS = (960, 961, 962)
ENTWURF_SEEDS, ENTWURF_TUNING = (963, 964), (965, 966)
_BELEGT = (v2._BELEGT | set(range(300, 350)) | set(range(360, 370)) | set(range(950, 956))
           | set(range(970, 982)))

STUFEN = (1, 2, 4, 8)
RANGFOLGEN = ("verlust", "zufall")
ARM_ALLE = "k32_alle"
NACH_ARME = tuple(f"k{k}_{r}" for r in RANGFOLGEN for k in STUFEN)
RANG_SEED = 120_000
SCHRITTE = v2.SCHRITTE_VOLL                            # 1.500 fuer voll UND jeden Nachtrain-Arm
LR_GITTER, LR_FAKTOR, KOLLAPS = v2.LR_GITTER, v2.LR_FAKTOR, v2.KOLLAPS
ALPHA = 0.05
TOST_M1, TOST_M2, TOST_M3 = 0.005, 0.003, 0.005
SENS_MIN = 2 * TOST_M1
G50, G90 = 0.5, 0.9
# Hypothese -> (Arm a, Arm b, delta): d_s = acc(a) - acc(b)
FAMILIE = {"M-1": ("k8_verlust", "voll", TOST_M1), "M-2": ("k2_verlust", "k2_zufall", TOST_M2),
           "M-3": ("voll", "suche", TOST_M3)}

HYPOTHESEN = {
    "M-1": "Fashion 16x16: Suche + 8 Magiewinkel (Auswahl nach Verlustanstieg, nachtrainiert) vs. voll, gemeinsamer "
           "Kopf, gepaart. Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (wenn voll - suche >= 1 pp und p < 0,05 und "
           "kein Kollaps von k8_verlust/voll; sonst HU mit Grund); sonst Holm-p(Wilcoxon) < 0,05 -> VOLL_BESSER / "
           "K8_BESSER (nur, wenn es ohne kollabierte Seeds haelt); sonst HU. Holm ueber M-1..M-3.",
    "M-2": "Fashion: k = 2 nach Verlust vs. k = 2 zufaellig (je eigene Rate): Holm-p < 0,05 -> H1 / H0 "
           "(Gegenrichtung); Holm-p_TOST(+-0,3 pp) < 0,05 -> H0 (gleich); sonst HU; Kollaps-Pruefung wie M-1.",
    "M-3": "Positivkontrolle: voll schlaegt suche auf Fashion (repliziert I-005-v2 V-2, frische Seeds): Holm-p < 0,05 "
           "-> H1 / H0 (Gegenrichtung); Holm-p_TOST(+-0,5 pp) < 0,05 -> H0 (gleich); sonst HU; Kollaps von voll geprueft.",
    "M-B": "beschreibend: G_S(k) = (acc(k) - acc(suche)) / (acc(voll) - acc(suche)) je Rangfolge, M50_S/M90_S, "
           "T-Gatter, Arme, eigene Koepfe, Winkelabstand zum Gitter, Ueberlappung, Raten/L2, Kollaps, Aufwand.",
}


# ------------------------------------------------------------------ Bausteine
def _stufe(arm) -> int:
    return int(arm.split("_")[0][1:])


def _nachtrain(codes, frei, x, y, lr, schritte, seed) -> QireZelle:
    """Geruest = Suche-Schaltung (codes * pi/2), genau die Winkel `frei` (flache Indizes) trainierbar.
    Phase A: Kopf allein auf den festen Born-Merkmalen; Phase B: freie Winkel + Kopf (Adam)."""
    torch.manual_seed(seed)
    m = QireZelle(n=N, lagen=LAGEN)
    maske = torch.zeros(LAGEN * N)
    maske[list(frei)] = 1.0
    m.fixiere(torch.as_tensor(np.asarray(codes), dtype=torch.float32).reshape(LAGEN, N) * GITTER,
              maske.reshape(LAGEN, N))
    with torch.no_grad():
        f = m.merkmale(x)
    trainiere(m.kopf, f, y, schritte, lr, seed=seed)
    if len(frei):
        trainiere(m, x, y, schritte, lr, seed=seed)
    return m


def _endlich(m) -> bool:
    return bool(torch.isfinite(m.winkel()).all()) and all(bool(torch.isfinite(p).all()) for p in m.kopf.parameters())


def _acc_eigen(m, x, y) -> float:
    if not _endlich(m):
        return 0.0
    with torch.no_grad():
        acc = float((m(x).argmax(1) == y).float().mean())
    return acc if math.isfinite(acc) else 0.0


def _acc_gemeinsam(m, xs) -> float:
    th = m.winkel().detach()
    return _bewerte(unitaer_winkel(th, N), N, xs)[0] if bool(torch.isfinite(th).all()) else 0.0


def _rang_verlust(m, codes, x, y):
    """I-001-Verlust-Rangfolge auf dem Suche-Geruest: Anstieg des Trainingsverlusts (Kopf fest), wenn NUR
    Winkel i auf seinen Suche-Wert zurueckgesetzt wird. -> (Rangfolge, delta je Winkel)."""
    from abteilungen.magie_pruning import _ce
    th = m.winkel().detach().clone().flatten()
    ziel = torch.as_tensor(np.asarray(codes), dtype=torch.float32).flatten() * GITTER
    basis = _ce(m, th, x, y)
    delta = torch.zeros(len(th))
    for i in range(len(th)):
        w = th.clone()
        w[i] = ziel[i]
        delta[i] = _ce(m, w, x, y) - basis
    delta = torch.nan_to_num(delta, nan=-math.inf)
    return torch.argsort(delta, descending=True, stable=True).tolist(), delta.tolist()


def _rang_zufall(seed):
    return torch.randperm(K_ALLE, generator=torch.Generator().manual_seed(RANG_SEED + seed)).tolist()


def _suche_codes(x, y, s, schritte, politur_max, n_zufall):
    """Die Suche aus v1._gemeinsame_arme OHNE Test-Koepfe (fuer Tuning-Seeds): gleiche Generatoren,
    gleiche Temperatur-Skala (sd des Kriteriums ueber n_zufall Zufalls-Clifford). Vortest: identisch."""
    bew = Bewerter(x, y, N)
    skala = max(float(np.std([bew(c)[0] for c in zufall_codes(s, n_zufall, N)])), 1e-4)
    start = np.random.default_rng(v1.START_SEED + s).integers(0, 4, (LAGEN, N))
    best, _, _ = suche(bew, start, np.random.default_rng(v1.SA_SEED + s), skala, schritte, politur_max=politur_max)
    return best.tolist()


# ------------------------------------------------------------------ Arbeiter
def _tuning_suche(a):
    """(seed, schritte, politur_max, n_zufall) -> (seed, Codes) -- nur auf dem Trainingsteil (3/4)."""
    s, schritte, politur_max, n_zufall = a
    xtr, ytr, _, _ = v2._daten(DS, s)
    nv = len(ytr) // 4
    return s, _suche_codes(xtr[nv:], ytr[nv:], s, schritte, politur_max, n_zufall)


def _tuning_arbeiter(a):
    """(arm, seed, lr, schritte, codes, frei, l2) -> {'acc': Validierung eigener Kopf, ...}.
    voll -> v2._tuning_voll; k32_alle liefert zusaetzlich die Verlust-Rangfolge (Trainingsteil)."""
    arm, s, lr, schritte, codes, frei, l2 = a
    if arm == "voll":
        return v2._tuning_voll((DS, s, lr, schritte, l2))
    xtr, ytr, _, _ = v2._daten(DS, s)
    nv = len(ytr) // 4
    frei = list(range(K_ALLE)) if frei is None else list(frei)
    m = _nachtrain(codes, frei, xtr[nv:], ytr[nv:], lr, schritte, s)
    aus = {"acc": _acc_eigen(m, xtr[:nv], ytr[:nv])}
    if arm == ARM_ALLE:
        aus["rang"], aus["delta"] = _rang_verlust(m, codes, xtr[nv:], ytr[nv:])
    return aus


def _mess_seed(a):
    """(seed, schritte, politur_max, n_zufall, schritte_nach, lr {arm: lr}, l2) -> Zeile."""
    s, schritte, politur_max, n_zufall, schritte_nach, lr, l2 = a
    t0 = time.time()
    z = v2._mess_seed((DS, s, schritte, politur_max, n_zufall, schritte_nach, lr["voll"], l2, False))
    xtr, ytr, xte, yte = v2._daten(DS, s)
    xs = (normiere(xtr), ytr, normiere(xte), yte, l2)
    codes = z["suche_codes"]
    tn = time.time()
    m = _nachtrain(codes, range(K_ALLE), xtr, ytr, lr[ARM_ALLE], schritte_nach, s)
    z[ARM_ALLE], z[ARM_ALLE + "_eigener_kopf"] = _acc_gemeinsam(m, xs), _acc_eigen(m, xte, yte)
    rang = {"zufall": _rang_zufall(s)}
    rang["verlust"], z["verlust_delta"] = _rang_verlust(m, codes, xtr, ytr)
    z["rang"] = rang
    for arm in NACH_ARME:
        k, r = _stufe(arm), arm.split("_")[1]
        frei = rang[r][:k]
        m = _nachtrain(codes, frei, xtr, ytr, lr[arm], schritte_nach, s)
        z[arm], z[arm + "_eigener_kopf"] = _acc_gemeinsam(m, xs), _acc_eigen(m, xte, yte)
        th = m.winkel().detach().flatten()[frei]
        z[arm + "_abstand"] = float((th - torch.round(th / GITTER) * GITTER).abs().mean())
        z[arm + "_frei"] = list(frei)
    z["nach_sekunden"] = time.time() - tn
    z["sekunden_v3"] = time.time() - t0
    for k in ("zufall_test", "zufall_val", "winkel_voll"):          # Rohdaten schlank halten (in v2 gesichert)
        z.pop(k, None)
    return z


# ------------------------------------------------------------------ Tuning
def _aufgaben(tuning, schritte, l2, codes, rang):
    def aufgaben_fuer(arm, lr):                                 # Seed an Stelle 1 (gittersuche_stabil)
        if arm == "voll":
            return [("voll", s, lr, schritte, None, None, l2) for s in tuning]
        if arm == ARM_ALLE:
            return [(arm, s, lr, schritte, codes[s], None, l2) for s in tuning]
        k, r = _stufe(arm), arm.split("_")[1]
        return [(arm, s, lr, schritte, codes[s], rang[r][s][:k], l2) for s in tuning]
    return aufgaben_fuer


def _raten(arme, aufgaben_fuer, gitter):
    info, roh = gittersuche_stabil(list(arme), aufgaben_fuer, lambda aufg: parallel(_tuning_arbeiter, aufg),
                                   gitter=gitter, faktor=LR_FAKTOR, kollaps=KOLLAPS)
    return info, roh


def _pruefe_seeds(seeds, tuning):
    alle = list(seeds) + list(tuning)
    return len(set(alle)) == len(alle) and not (set(alle) & _BELEGT)


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=N_SEEDS, seeds=None, tuning_seeds=None, schritte=v1.SA_SCHRITTE,
         politur_max=v1.POLITUR_MAX, n_zufall=v1.N_ZUFALL, schritte_nach=SCHRITTE):
    l2_gitter, l2_zufall, lr_gitter = v2.L2_GITTER, v2.L2_ZUFALL, LR_GITTER
    if schnell:                                                   # nur Entwurfs-Seeds
        seeds, tuning_seeds = ENTWURF_SEEDS, ENTWURF_TUNING
        schritte, politur_max, n_zufall, schritte_nach = 80, 20, 5, 40
        l2_gitter, l2_zufall, lr_gitter = (1e-3, 1e-2, 1e-1), 2, (0.04, 0.08, 0.16)
    else:
        if n_seeds not in (10, 20):
            raise ValueError("n_seeds ist vorregistriert: 10 oder 20")
        seeds = seeds or SEEDS_LANG[:n_seeds]
        tuning_seeds = tuning_seeds or TUNING_SEEDS
        if not _pruefe_seeds(seeds, tuning_seeds):
            raise ValueError("Seeds nicht frisch oder nicht disjunkt")
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    t0 = time.time()
    zeit = lambda: f"{time.time() - t0:.0f} s"

    # T1: Kopf-L2 (Regel aus v2) und Suche auf den Tuning-Seeds (nur Trainingsteil)
    l2_info = v2._l2_suche((DS,), {DS: tuning_seeds}, l2_zufall, gitter=l2_gitter)[DS]
    l2 = l2_info["l2"]
    codes = dict(parallel(_tuning_suche, [(s, schritte, politur_max, n_zufall) for s in tuning_seeds]))
    print(f"  T1 (L2 {l2:.2g}{' AM RAND' if l2_info['am_rand'] else ''}, Suche Tuning-Seeds) nach {zeit()}", flush=True)

    # T2: Raten von voll und k32_alle; T3: Verlust-Rangfolge je Tuning-Seed bei der gewaehlten k32-Rate
    lr_info, roh = _raten(("voll", ARM_ALLE), _aufgaben(tuning_seeds, schritte_nach, l2, codes, {}), lr_gitter)
    lr_alle = lr_info[ARM_ALLE]["lr"]
    rang = {"verlust": {s: roh[(ARM_ALLE, lr_alle)][s]["rang"] for s in tuning_seeds},
            "zufall": {s: _rang_zufall(s) for s in tuning_seeds}}
    print(f"  T2 (Rate voll {lr_info['voll']['lr']:g}, k32_alle {lr_alle:g}) nach {zeit()}", flush=True)

    # T4: Rate je Nachtrain-Arm
    info2, _ = _raten(NACH_ARME, _aufgaben(tuning_seeds, schritte_nach, l2, codes, rang), lr_gitter)
    lr_info.update(info2)
    lr = {arm: lr_info[arm]["lr"] for arm in ("voll", ARM_ALLE) + NACH_ARME}
    print(f"  T4 (Raten {', '.join(f'{a} {v:g}' for a, v in lr.items())}) nach {zeit()}", flush=True)

    # Messung
    zz = sorted(parallel(_mess_seed, [(s, schritte, politur_max, n_zufall, schritte_nach, lr, l2) for s in seeds]),
                key=lambda z: z["seed"])
    for z in zz:
        print(f"  Fashion {z['seed']}: suche {z['suche']:.4f} voll {z['voll']:.4f} (eig. {z['voll_eigener_kopf']:.4f}) "
              + " ".join(f"{a} {z[a]:.4f}" for a in NACH_ARME + (ARM_ALLE,)) + f" ({z['sekunden_v3']:.0f} s)", flush=True)
    e = {"abteilung": "Clifford", "modul": "clifford_suche_v3", "karte": "I-005-v3", "version": 1,
         "hypothesen": HYPOTHESEN, "datensatz": DS, "seeds": seeds, "tuning_seeds": tuning_seeds, "n_seeds": len(seeds),
         "lr": lr, "lr_info": lr_info, "kopf_l2": l2_info, "tuning_codes": {str(s): c for s, c in codes.items()},
         "tuning_rang": {r: {str(s): v for s, v in d.items()} for r, d in rang.items()},
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "schritte_voll": schritte_nach, "schritte_nach": schritte_nach, "stufen": STUFEN,
                           "lr_gitter": list(lr_gitter), "kollaps": KOLLAPS, "l2_gitter": list(l2_gitter),
                           "tost": {h: f[2] for h, f in FAMILIE.items()}, "sens_min": SENS_MIN, "rang_seed": RANG_SEED},
         "zeilen": zz, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zz)
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Auswertung
def _eigen_schluessel(arm):
    return None if arm == "suche" else arm + "_eigener_kopf"


def kollabiert(zz, arm) -> list:
    """Mess-Seeds, auf denen `arm` kollabiert ist: eigener Kopf < Median - 5 pp oder nicht endlich."""
    k = _eigen_schluessel(arm)
    if k is None:
        return []
    eig = np.array([z[k] for z in zz], dtype=float)
    grenze = float(np.nanmedian(eig)) - KOLLAPS if np.isfinite(eig).any() else math.inf
    return [z["seed"] for z, v in zip(zz, eig) if not v >= grenze]


def _test(zz, a, b, delta, ohne=()):
    d = np.array([z[a] - z[b] for z in zz if z["seed"] not in ohne], dtype=float)
    return {**_gepaart(d), "p_tost": v2.tost_p(d, delta)}


def _g_kurve(zz, rangfolge, sauber):
    """G_S(k) = (acc(k) - acc(suche)) / (acc(voll) - acc(suche)), Seed-Mittel ueber `sauber`."""
    zs = [z for z in zz if z["seed"] in sauber]
    if not zs:
        return {}
    m = lambda k: float(np.mean([z[k] for z in zs]))
    nenner = m("voll") - m("suche")
    arme = {0: "suche", **{k: f"k{k}_{rangfolge}" for k in STUFEN}, K_ALLE: ARM_ALLE}
    return {str(k): ((m(a) - m("suche")) / nenner if nenner > 0 else None) for k, a in arme.items()}


def m_schwelle(kurve, g):
    """Kleinstes k in {0, 1, 2, 4, 8} mit G_S(k) >= g, sonst None (= mehr als 8)."""
    for k in (0,) + STUFEN:
        v = kurve.get(str(k))
        if v is not None and v >= g:
            return k
    return None


def auswertung(zz) -> dict:
    arme = ("suche", "voll", ARM_ALLE) + NACH_ARME
    koll = {arm: kollabiert(zz, arm) for arm in arme}
    a = {"n": len(zz), "kollaps": koll, "tests": {}, "tests_sauber": {}, "ausgeschlossen": {}}
    for h, (x, y, delta) in FAMILIE.items():
        ohne = sorted(set(koll[x]) | set(koll[y]))
        a["ausgeschlossen"][h] = ohne
        a["tests"][h] = _test(zz, x, y, delta)
        a["tests_sauber"][h] = _test(zz, x, y, delta, ohne)
    for schl, holm_schl in (("tests", "holm"), ("tests_sauber", "holm_sauber")):
        a[holm_schl] = {"p": holm({h: t["p"] for h, t in a[schl].items()}),
                        "p_tost": holm({h: t["p_tost"] for h, t in a[schl].items()})}
    s = a["tests"]["M-3"]
    a["sensitivitaet"] = {"mittel": s["mittel"], "p": s["p"],
                          "erfuellt": bool(s["mittel"] >= SENS_MIN and s["p"] < ALPHA)}
    # beschreibend
    sauber = {z["seed"] for z in zz} - set(koll["voll"])
    b = {"arme": {arm: zusammenfassung([z[arm] for z in zz]) for arm in arme + ("identitaet", "k0", "zufall_median")},
         "eigene_koepfe": {arm: zusammenfassung([z[_eigen_schluessel(arm)] for z in zz]) for arm in arme[1:]},
         "G_S": {r: _g_kurve(zz, r, sauber) for r in RANGFOLGEN}, "G_S_seeds": sorted(sauber),
         "G_S_alle_seeds": {r: _g_kurve(zz, r, {z["seed"] for z in zz}) for r in RANGFOLGEN}}
    from abteilungen.kaskade import T_JE_K
    b["M50_S"] = {r: m_schwelle(b["G_S"][r], G50) for r in RANGFOLGEN}
    b["M90_S"] = {r: m_schwelle(b["G_S"][r], G90) for r in RANGFOLGEN}
    b["T_M50_S"] = {r: (T_JE_K.get(k) if k is not None else None) for r, k in b["M50_S"].items()}
    b["T_M90_S"] = {r: (T_JE_K.get(k) if k is not None else None) for r, k in b["M90_S"].items()}
    b["abstand_frei"] = {arm: float(np.mean([z[arm + "_abstand"] for z in zz])) for arm in NACH_ARME}
    b["ueberlappung"] = {str(k): float(np.mean([len(set(z["rang"]["verlust"][:k]) & set(z["rang"]["zufall"][:k])) / k
                                               for z in zz])) for k in STUFEN}
    mid = lambda k: float(np.mean([z[k] for z in zz]))
    gv = mid("voll") - mid("identitaet")
    b["G_voll_suche"] = (mid("suche") - mid("identitaet")) / gv if gv > 0 else None
    b["k8_verlust_minus_k8_zufall"] = _gepaart([z["k8_verlust"] - z["k8_zufall"] for z in zz])
    b["k32_minus_voll"] = _gepaart([z[ARM_ALLE] - z["voll"] for z in zz])
    for k in ("suche_sekunden", "voll_sekunden", "nach_sekunden", "sekunden_v3", "suche_bewertungen"):
        b[k] = float(np.mean([z[k] for z in zz]))
    a["beschreibend"] = b
    return a


# ------------------------------------------------------------------ Urteil
def _marke(h, t, ph, pth) -> str:
    if h == "M-1":
        if pth < ALPHA:
            return "GLEICHWERTIG"
        if ph < ALPHA:
            return "VOLL_BESSER" if t["mittel"] < 0 else "K8_BESSER"
        return "HU"
    if ph < ALPHA:
        return "H1" if t["mittel"] > 0 else "H0 (Gegenrichtung)"
    if pth < ALPHA:
        return "H0 (gleich)"
    return "HU"


_AEQUIVALENZ = ("GLEICHWERTIG", "H0 (gleich)")
_TEXT = {"GLEICHWERTIG": "Suche + 8 Magiewinkel (nach Verlust) = volles Training im Rahmen +-0,5 pp",
         "VOLL_BESSER": "8 Magiewinkel auf der Suche-Schaltung schliessen die Luecke zu voll NICHT",
         "K8_BESSER": "Suche + 8 Magiewinkel schlaegt das volle Training",
         ("M-2", "H1"): "Verlust-Auswahl schlaegt zufaellige Auswahl bei k = 2",
         ("M-2", "H0 (Gegenrichtung)"): "zufaellige Auswahl besser als Verlust-Auswahl bei k = 2",
         ("M-2", "H0 (gleich)"): "Verlust- = Zufalls-Auswahl bei k = 2 im Rahmen +-0,3 pp",
         ("M-3", "H1"): "voll schlaegt suche (I-005-v2 V-2 repliziert, Sensitivitaet gegeben)",
         ("M-3", "H0 (Gegenrichtung)"): "suche schlaegt voll",
         ("M-3", "H0 (gleich)"): "voll = suche im Rahmen +-0,5 pp -- keine Luecke"}


def _zahlen(h, t, ph, pth, ohne) -> str:
    x, y, delta = FAMILIE[h]
    return (f"d = {x} - {y} = {t['mittel'] * 100:+.2f} pp, 90-%-KI [{t['ki_unten'] * 100:+.2f}; "
            f"{t['ki_oben'] * 100:+.2f}], TOST(+-{delta * 100:.1f} pp) p = {t['p_tost']:.4f}, Holm-p_TOST = "
            f"{pth:.4f}, Wilcoxon p = {t['p']:.4g}, Holm-p = {ph:.4g}, {t.get('positiv', 0)}/{t['n']} Seeds positiv; "
            f"kollabiert ({x}/{y}): {ohne or 'keiner'}")


def _urteil_h(h, a) -> str:
    t, ph, pth = a["tests"][h], a["holm"]["p"][h], a["holm"]["p_tost"][h]
    ohne = a["ausgeschlossen"][h]
    txt = _zahlen(h, t, ph, pth, ohne)
    if h == "M-1":
        s = a["sensitivitaet"]
        txt += f"; Sensitivitaet voll - suche {s['mittel'] * 100:+.2f} pp (p = {s['p']:.4g})"
    marke = _marke(h, t, ph, pth)
    if marke == "HU":
        return "HU: " + txt
    if h == "M-1" and marke == "GLEICHWERTIG" and not a["sensitivitaet"]["erfuellt"]:
        return (f"HU (Sensitivitaet): gleich im Rahmen +-0,5 pp, aber voll schlaegt suche nicht um mindestens "
                f"{SENS_MIN * 100:.1f} pp nachweisbar -- " + txt)
    inhalt = _TEXT.get(marke) or _TEXT[(h, marke)]
    if ohne:
        if marke in _AEQUIVALENZ:
            return f"HU (Instabilitaet): {marke} ({inhalt}), aber beteiligte Arme auf Mess-Seeds kollabiert -- " + txt
        ts = a["tests_sauber"][h]
        m2 = _marke(h, ts, a["holm_sauber"]["p"][h], a["holm_sauber"]["p_tost"][h])
        if m2 != marke:
            return (f"HU (Instabilitaet): {marke} nur mit den kollabierten Seeds {ohne}; ohne sie {m2} "
                    f"(d = {ts['mittel'] * 100:+.2f} pp, n = {ts['n']}) -- " + txt)
        txt += f"; haelt ohne kollabierte Seeds {ohne} (d = {ts['mittel'] * 100:+.2f} pp, n = {ts['n']})"
    return f"{marke}: {inhalt} -- " + txt


def urteil(a: dict) -> dict:
    u = {h: _urteil_h(h, a) for h in FAMILIE}
    b = a["beschreibend"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    mk = lambda v: "mehr als 8" if v is None else str(v)
    teile = []
    for r in RANGFOLGEN:
        teile.append(f"G_S {r}: " + ", ".join(f"k={k} {fmt(v)}" for k, v in b["G_S"][r].items())
                     + f" -> M50_S {mk(b['M50_S'][r])} (T {b['T_M50_S'][r]}), M90_S {mk(b['M90_S'][r])} "
                       f"(T {b['T_M90_S'][r]})")
    teile.append("Test: " + ", ".join(f"{k} {v['mittel']:.4f}" for k, v in b["arme"].items()))
    teile.append(f"G_voll(suche) {fmt(b['G_voll_suche'])}; k8 verlust - zufall "
                 f"{b['k8_verlust_minus_k8_zufall']['mittel'] * 100:+.2f} pp; k32_alle - voll "
                 f"{b['k32_minus_voll']['mittel'] * 100:+.2f} pp")
    teile.append("Abstand freier Winkel zum Gitter (rad): " + ", ".join(f"{k} {v:.3f}" for k, v in b["abstand_frei"].items()))
    teile.append("Ueberlappung verlust/zufall: " + ", ".join(f"k={k} {v:.2f}" for k, v in b["ueberlappung"].items()))
    teile.append("kollabiert: " + (", ".join(f"{k} {v}" for k, v in a["kollaps"].items() if v) or "keiner")
                 + f"; G_S auf {len(b['G_S_seeds'])} Seeds ohne voll-Kollaps")
    teile.append(f"Aufwand je Seed: Suche {b['suche_sekunden']:.0f} s, voll {b['voll_sekunden']:.0f} s, "
                 f"Nachtraining {b['nach_sekunden']:.0f} s, gesamt {b['sekunden_v3']:.0f} s")
    u["M-B"] = "beschreibend: " + " | ".join(teile)
    return u


# ------------------------------------------------------------------ Vortest
_G_VERLUST = {1: 0.35, 2: 0.55, 4: 0.8}
_G_ZUFALL = {1: 0.1, 4: 0.4, 8: 0.6}


def _synth(rng, luecke=0.013, g8=1.0, d2=0.004, sd=0.001, kollaps=None, tief=0.0, n=10):
    """Synthetische Mess-Zeilen (n = 10: der exakte Wilcoxon zaehlt 2^n Vorzeichen, n = 20 dauert ~40 s je
    Aufruf -- die Verzweigungslogik haengt nicht von n ab). luecke = voll - suche, g8 = G_S(8) des Verlust-Arms, d2 = k2_verlust -
    k2_zufall; kollaps {arm: Anzahl Seeds}: eigener Kopf -0,2 (und gemeinsamer Kopf -`tief`)."""
    kollaps = kollaps or {}
    zz = []
    for i in range(n):
        su = 0.824 + rng.normal(0, 0.003)
        z = {"seed": 1000 + i, "suche": su, "voll": su + luecke + rng.normal(0, sd), "identitaet": 0.7995,
             "k0": su - 0.008, "zufall_median": su - 0.007, "suche_sekunden": 440.0, "voll_sekunden": 120.0,
             "nach_sekunden": 1100.0, "sekunden_v3": 1950.0, "suche_bewertungen": 4600,
             "rang": {"verlust": list(range(K_ALLE)), "zufall": list(range(K_ALLE))[::-1]}}
        for k in STUFEN:
            z[f"k{k}_verlust"] = su + (g8 if k == 8 else _G_VERLUST[k]) * luecke + rng.normal(0, sd)
        for k in STUFEN:
            z[f"k{k}_zufall"] = (z["k2_verlust"] - d2 if k == 2 else su + _G_ZUFALL[k] * luecke) + rng.normal(0, sd)
        z[ARM_ALLE] = z["voll"] + rng.normal(0, sd)
        for arm in ("voll", ARM_ALLE) + NACH_ARME:
            z[arm + "_eigener_kopf"] = z[arm] + 0.002
            if arm in NACH_ARME:
                z[arm + "_abstand"] = 0.3
            if i < kollaps.get(arm, 0):
                z[arm + "_eigener_kopf"] -= 0.2
                z[arm] -= tief
        zz.append(z)
    return zz


def _vortest_urteile(ziehungen=5) -> bool:
    import json
    ok = True
    rng = np.random.default_rng(11)
    # (Name, _synth-Parameter, {Hypothese: (Marke, [Textteile])}, Toleranz)
    faelle = [
        ("k8 schliesst", {}, {"M-1": ("GLEICHWERTIG", ["Holm-p_TOST", "k8_verlust - voll", "Sensitivitaet"]),
                              "M-2": ("H1", ["k = 2", "k2_verlust - k2_zufall"]), "M-3": ("H1", ["repliziert"])}, 0),
        ("Luecke bleibt", {"g8": 0.5}, {"M-1": ("VOLL_BESSER", ["NICHT", "0/10"]), "M-3": ("H1", [])}, 0),
        ("k8 besser", {"g8": 1.6}, {"M-1": ("K8_BESSER", ["schlaegt"])}, 0),
        ("Streuung", {"luecke": 0.0, "d2": 0.0, "sd": 0.02},
         {"M-1": ("HU:", ["k8_verlust"]), "M-2": ("HU:", []), "M-3": ("HU:", [])}, 1),
        ("keine Luecke", {"luecke": 0.004}, {"M-1": ("HU (Sensitivitaet)", ["1.0 pp"]), "M-3": ("H1", [])}, 0),
        ("M-3 gleich", {"luecke": 0.0, "sd": 0.0005},
         {"M-1": ("HU (Sensitivitaet)", []), "M-3": ("H0 (gleich)", ["keine Luecke"])}, 0),
        ("M-3 Gegenrichtung", {"luecke": -0.01},
         {"M-3": ("H0 (Gegenrichtung)", ["suche schlaegt voll"]), "M-1": ("HU (Sensitivitaet)", [])}, 0),
        ("M-2 gleich", {"d2": 0.0, "sd": 0.0005}, {"M-2": ("H0 (gleich)", ["0,3 pp"])}, 0),
        ("M-2 Gegenrichtung", {"d2": -0.004}, {"M-2": ("H0 (Gegenrichtung)", ["zufaellige Auswahl besser"])}, 0),
        ("Kollaps voll", {"kollaps": {"voll": 2}},
         {"M-1": ("HU (Instabilitaet)", ["GLEICHWERTIG", "kollabiert (k8_verlust/voll): [1000, 1001]"]),
          "M-3": ("H1", ["haelt ohne kollabierte Seeds [1000, 1001]"])}, 0),
        ("Kollaps k8 VOLL_BESSER", {"g8": 0.5, "kollaps": {"k8_verlust": 1}},
         {"M-1": ("VOLL_BESSER", ["haelt ohne kollabierte Seeds [1000]"])}, 0),
        # 4 von 10 k8-Seeds kollabiert und 5 pp schlechter: mit ihnen VOLL_BESSER, ohne sie GLEICHWERTIG
        ("Kollaps kippt", {"kollaps": {"k8_verlust": 4}, "tief": 0.05, "g8": 0.97, "sd": 0.0001},
         {"M-1": ("HU (Instabilitaet)", ["VOLL_BESSER nur mit den kollabierten Seeds", "ohne sie GLEICHWERTIG"])}, 0),
        ("Kollaps M-2 gleich", {"d2": 0.0, "sd": 0.0005, "kollaps": {"k2_zufall": 1}},
         {"M-2": ("HU (Instabilitaet)", ["H0 (gleich)", "[1000]"])}, 0),
        ("Kollaps M-2 H1", {"kollaps": {"k2_verlust": 3}}, {"M-2": ("H1", ["haelt ohne"])}, 0),
    ]
    for name, par, erwartet, tol in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            a = auswertung(_synth(rng, **par))
            u = urteil(a)
            for h, (marke, teile) in erwartet.items():
                if not u[h].startswith(marke) or any(t not in u[h] for t in teile):
                    fehl[h] += 1
                    print(f"  Vortest [{name}]: {h} -> {u[h][:170]}")
            if not u["M-B"].startswith("beschreibend") or not all(isinstance(v, str) for v in u.values()) \
                    or set(u) != {"M-1", "M-2", "M-3", "M-B"}:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: Urteile unvollstaendig oder kein String")
            try:
                json.dumps({"auswertung": a, "urteil": u})
            except (TypeError, ValueError) as err:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: nicht JSON-serialisierbar", err)
        for h, f in fehl.items():
            if f > tol:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: {h} {f}/{ziehungen} falsch")
    # grosse Streuung, wahre Differenz 0 -> NIE GLEICHWERTIG / H0 (gleich) (Toleranz 0)
    for _ in range(ziehungen):
        u = urteil(auswertung(_synth(rng, g8=1.0, d2=0.0, sd=0.02)))
        if u["M-1"].startswith("GLEICHWERTIG") or u["M-2"].startswith("H0 (gleich)"):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung")
    # G_S-Kurve und M50/M90 auf dem konstruierten Fall (sd 0): verlust 0,35/0,55/0,8/1,0 -> M50 2, M90 8
    a = auswertung(_synth(np.random.default_rng(0), sd=0.0))
    b = a["beschreibend"]
    g = b["G_S"]["verlust"]
    if abs(g["0"]) > 1e-9 or abs(g["2"] - 0.55) > 1e-6 or abs(g["8"] - 1.0) > 1e-6 or abs(g["32"] - 1.0) > 1e-6 \
            or b["M50_S"]["verlust"] != 2 or b["M90_S"]["verlust"] != 8 or b["M90_S"]["zufall"] is not None \
            or b["T_M50_S"]["verlust"] != 62 or "mehr als 8" not in urteil(a)["M-B"]:
        ok = False
        print(f"  VORTEST FEHLER: G_S/M50/M90 {g} {b['M50_S']} {b['M90_S']} {b['T_M50_S']}")
    return ok


def _vortest_bausteine() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(3)
    # (a) TOST-p < 0,05 <=> 90-%-KI in (-delta, delta); Holm; Familie genau {M-1, M-2, M-3}
    for _ in range(100):
        d = rng.normal(rng.normal(0, 0.003), rng.uniform(0.0005, 0.008), 10)
        g, p = _gepaart(d), v2.tost_p(d, TOST_M1)
        if (p < ALPHA) != (-TOST_M1 < g["ki_unten"] and g["ki_oben"] < TOST_M1):
            fehler("TOST-p passt nicht zum KI")
            break
    h = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    if abs(h["a"] - 0.03) > 1e-12 or abs(h["b"] - 0.08) > 1e-12 or set(FAMILIE) != {"M-1", "M-2", "M-3"}:
        fehler(f"Holm/Familie {h}")
    # (b) Seeds frisch und disjunkt; v2-Seeds belegt; Entwurf getrennt; n_seeds nur 10/20
    for n in (10, 20):
        if not _pruefe_seeds(SEEDS_LANG[:n], TUNING_SEEDS):
            fehler(f"Seeds (n = {n}) nicht frisch/disjunkt")
    v2_seeds = ({s for v in v2.SEEDS_LANG.values() for s in v} | set(v2.SEEDS_STRUKTUR)
                | {s for v in v2.TUNING_SEEDS.values() for s in v} | {s for v in v2.ENTWURF_SEEDS.values() for s in v}
                | {s for v in v2.ENTWURF_TUNING.values() for s in v} | set(v2.ENTWURF_STRUKTUR)
                | set(v1.SEEDS_MNIST) | set(v1.SEEDS_STRUKTUR))
    if not v2_seeds <= _BELEGT or _pruefe_seeds((310,), TUNING_SEEDS) or _pruefe_seeds((420, 420), TUNING_SEEDS) \
            or _pruefe_seeds(SEEDS_LANG, (953,)):
        fehler("Seed-Pruefung laesst belegte/doppelte Seeds durch")
    if (set(ENTWURF_SEEDS) | set(ENTWURF_TUNING)) & (set(SEEDS_LANG) | set(TUNING_SEEDS) | _BELEGT) \
            or len(SEEDS_LANG) != 20 or N_SEEDS != 20:
        fehler("Entwurfs-Seeds ueberschneiden sich oder Seed-Zahl falsch")
    try:
        lauf(n_seeds=15)
        fehler("n_seeds = 15 nicht abgelehnt")
    except ValueError:
        pass
    # (c) Kollaps: nicht endlich oder < Median - 5 pp wird erkannt, -3 pp nicht; suche nie
    zz = _synth(rng)
    zz[3]["voll_eigener_kopf"] = float("nan")
    zz[5]["voll_eigener_kopf"] -= 0.06
    zz[6]["voll_eigener_kopf"] -= 0.03
    if kollabiert(zz, "voll") != [1003, 1005] or kollabiert(zz, "suche") != [] or kollabiert(zz, "k8_verlust") != []:
        fehler(f"Kollaps-Erkennung {kollabiert(zz, 'voll')}")
    # (d) Nachtraining: feste Winkel bleiben exakt auf dem Suche-Geruest, freie bewegen sich; k = 0 = Suche-Schaltung
    torch.manual_seed(0)
    x, y = torch.rand(96, 2 ** N), torch.randint(0, 10, (96,))
    codes = rng.integers(0, 2, (LAGEN, N)).tolist()
    ziel = torch.as_tensor(np.asarray(codes), dtype=torch.float32).flatten() * GITTER
    frei = [3, 17]
    m = _nachtrain(codes, frei, x, y, 0.1, 5, 7)
    w = m.winkel().detach().flatten()
    fest = [i for i in range(K_ALLE) if i not in frei]
    if not torch.equal(w[fest], ziel[fest]) or not bool((w[frei] != ziel[frei]).all()) or int(m.frei.sum()) != 2:
        fehler("Nachtraining: feste Winkel bewegt oder freie Winkel unbewegt")
    m0 = _nachtrain(codes, [], x, y, 0.1, 5, 7)
    with torch.no_grad():
        if not torch.equal(m0.winkel().detach().flatten(), ziel) \
                or not torch.allclose(m0.unitaer(), unitaer_clifford(codes, N), atol=1e-5):
            fehler("k = 0 ist nicht die Suche-Schaltung")
    if not (0.0 <= _acc_eigen(m, x, y) <= 1.0):
        fehler("eigener Kopf ausserhalb [0, 1]")
    # (e) Verlust-Rangfolge: nur der vom Geruest abweichende Winkel hat einen Verlustanstieg != 0
    ma = QireZelle(n=N, lagen=LAGEN)
    th = ziel.clone()
    th[11] += 0.7
    ma.fixiere(th.reshape(LAGEN, N), torch.ones(LAGEN, N))
    r, delta = _rang_verlust(ma, codes, x, y)
    andere = [abs(v) for i, v in enumerate(delta) if i != 11]
    if sorted(r) != list(range(K_ALLE)) or max(andere) > 1e-7 or abs(delta[11]) < 1e-6 \
            or any(delta[r[i]] < delta[r[i + 1]] for i in range(K_ALLE - 1)):
        fehler(f"Verlust-Rangfolge: delta[11] = {delta[11]}, max andere {max(andere)}")
    # (f) Zufalls-Rangfolge: Permutation, deterministisch, je Seed verschieden
    if sorted(_rang_zufall(420)) != list(range(K_ALLE)) or _rang_zufall(420) != _rang_zufall(420) \
            or _rang_zufall(420) == _rang_zufall(421):
        fehler("Zufalls-Rangfolge")
    # (g) Suche fuer Tuning-Seeds = Suche aus v1._gemeinsame_arme (gleiche Generatoren, gleiche Skala)
    xs_, ys_ = torch.rand(160, 2 ** N), torch.randint(0, 10, (160,))
    z, *_ = v1._gemeinsame_arme(5, N, (xs_, ys_, xs_[:40], ys_[:40]), 25, 10, 4, 1e-2)
    if _suche_codes(xs_, ys_, 5, 25, 10, 4) != z["suche_codes"]:
        fehler("Tuning-Suche weicht von v1._gemeinsame_arme ab")
    # (h) Tuning-Aufgaben: Seed an Stelle 1, richtige freie Winkel, L2 durchgereicht
    cd = {s: codes for s in (1, 2)}
    rg = {"verlust": {1: list(range(K_ALLE)), 2: list(range(K_ALLE))[::-1]}, "zufall": {1: [5] * 32, 2: [6] * 32}}
    af = _aufgaben((1, 2), 99, 0.003, cd, rg)
    a4, av, a32 = af("k4_verlust", 0.08), af("voll", 0.16), af(ARM_ALLE, 0.04)
    if [a[1] for a in a4] != [1, 2] or a4[1][5] != [31, 30, 29, 28] or a4[0][2] != 0.08 or a4[0][3] != 99 \
            or av[0][0] != "voll" or av[0][6] != 0.003 or a32[0][5] is not None or af("k2_zufall", 0.1)[1][5] != [6, 6]:
        fehler("Tuning-Aufgaben falsch")
    if m_schwelle({}, 0.5) is not None or m_schwelle({"0": 0.0, "1": 0.6}, 0.5) != 1 or _stufe("k8_zufall") != 8:
        fehler("m_schwelle/_stufe")
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Bausteine (TOST, Holm, Seeds, Kollaps, Nachtraining, Rangfolgen, Tuning-Suche,
    Tuning-Aufgaben) + jeden Urteilszweig synthetisch erzwingen (Marke UND Text, mehrere Ziehungen).
    Die Such-Bausteine selbst prueft der Vortest von clifford_suche (I-005), aus dem sie importiert sind."""
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
    print(speichere("I-005-v3" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
