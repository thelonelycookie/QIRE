"""Abteilung Clifford -- Ist die Magie-Konzentration eine Eigenschaft der AUFGABE?
(Karte I-003, Quantenstufe Q3)

HEILMEIER:
  Was wollen wir?  Wissen, ob die Magie-Konzentration (MNIST: G(1) = 0,49,
      M50 = 2 von 32 Winkeln, LABORBUCH „C-mnist“) etwas ueber die AUFGABE
      sagt -- oder nur ueber Architektur und Training (dann ist sie keine
      Kennzahl einer Aufgabe, und der Arbeitstitel „Magie-Budget einer
      Aufgabe“ faellt).
  Wie heute?  clifford.py v3 misst G(k) auf je einem Datensatz. Ein einzelner
      Datensatz kann nicht zwischen „Aufgabe“ und „Architektur“ entscheiden.
  Was ist neu?  Dieselbe Pipeline (voll, ohne_schaltung, clifford_k{0,1,2,4,8,
      12} mit FRISCHEM Kopf, _gerundet aus clifford.py) auf drei Aufgaben mit
      denselben Bildern, derselben Architektur, denselben Seeds:
        echt     -- die echten Ziffern-Etiketten (Messgroesse: Testgenauigkeit)
        zufall   -- ZUFAELLIGE Etiketten: die Etiketten des Trainingsteils
                    werden je Seed mit einer festen Permutation ueber die
                    BILDER gemischt (Klassenhaeufigkeiten bleiben gleich,
                    jede Struktur Bild -> Klasse ist weg).
        struktur -- Aufgabe mit bekannter Loesung OHNE Magie: Etikett =
                    argmax eines festen linearen Lehrers auf den Born-
                    Wahrscheinlichkeiten einer festen zufaelligen CLIFFORD-
                    Schaltung (gleiche Architektur, alle Winkel auf k*pi/2).
                    Die Aufgabe ist per Konstruktion mit Magie 0 exakt
                    loesbar -> Positivkontrolle: dort MUSS G(0) hoch sein
                    koennen, sonst kann das Mass gar nicht variieren.
  Woran merken wir es?  Unterscheidet sich G(k) zwischen echt und zufall,
      traegt die Aufgabe; ist es gleich (TOST), ist es die Architektur.

MESSGROESSE BEI ZUFALLSETIKETTEN (Designentscheidung, vor dem Lauf):
  Auf Zufallsetiketten ist die Testgenauigkeit fuer JEDEN Arm ~0,10 -- der
  Gewinn der Schaltung waere 0, G undefiniert. Die einzige Leistung, die es
  dort gibt, ist MEMORIERUNG: wie viele der 1.257 Trainingsbilder bekommen
  nach dem Training ihr (zufaelliges) Etikett? Messgroesse fuer „zufall“ ist
  deshalb die Trainingsgenauigkeit auf genau der Menge, auf der trainiert
  wurde (volles Budget, gleiche Schritte). G(k) heisst dort: Anteil der
  zusaetzlichen MEMORIERUNGSkapazitaet der Schaltung (ueber die Identitaet),
  den k Magiewinkel erhalten. Fuer echt/struktur ist die Testgenauigkeit die
  Messgroesse (auf den echten Ziffern waere die Trainingsgenauigkeit fuer
  alle Arme ~1,0 und G wieder undefiniert). G ist dimensionslos (Anteil am
  eigenen Gewinn) und damit zwischen den Aufgaben vergleichbar, auch wenn die
  Rohgroessen verschieden sind. Die jeweils andere Genauigkeit wird
  mitgeschrieben („neben“; bei zufall muss der Test ~0,10 sein -- Pruefung,
  dass die Etiketten wirklich zufaellig sind).
  Die Lernraten-Suche benutzt dieselbe Messgroesse: bei zufall die
  Trainingsgenauigkeit auf den Tuning-Seeds (Validierung waere Zufall).

STRUKTURAUFGABE (Positivkontrolle), feste Wahl vor dem Lauf:
  Lehrer-Winkel: randint(0, 4) * pi/2 (Generator LEHRER_SEED), 4 Lagen x 6
  Qubits, CNOT-Ring wie die Schueler. Lehrer-Kopf W ~ N(0, 1) (10 x 64).
  Etikett = argmax_j (W p_C(x) - mu_j) / sd_j, mu/sd ueber alle 1.797
  Ziffern (etwa gleich grosse Klassen). Das ist affin in p_C -> ein linearer
  Kopf auf den richtigen Clifford-Merkmalen trennt exakt.
  LEHRER MIT MARGE (Standard im Lehrer-Schueler-Aufbau): Zur Aufgabe gehoeren
  nur Bilder, deren bester Lehrer-Punktwert den zweitbesten um >= MARGE = 0,25
  uebertrifft (fest je Bild, unabhaengig vom Seed; ~65 % der Ziffern, also ~820
  Trainings-/~350 Testbilder je Seed). Grund (Designphase, vor jeder Messung):
  ohne Marge generalisiert selbst ein linearer Kopf auf den EXAKTEN Lehrer-
  Merkmalen nur auf 0,86 (Training 1,00) -- 16 % der Testbilder liegen
  weniger als 0,1 an einer Klassengrenze, 1.257 Bilder reichen nicht, sie in
  64 Dimensionen festzulegen. Mit Marge: 0,93 gegen 0,74 auf der Identitaet.
  Die Strukturaufgabe hat damit andere (weniger) Bilder als echt/zufall; sie
  geht nur in T-1 und beschreibend in T-3 ein, nicht in den Paarvergleich T-2.
  Bausteinpruefung im
  Vortest: (a) Lehrer ist keine (Vorzeichen-)Permutation (Lehre „entartete
  Kontrolle“), (b) QireZelle mit Lehrer-Winkeln liefert exakt p_C,
  (c) linearer Kopf auf p_C erreicht >= LOESBAR Testgenauigkeit, (d) auf den
  Identitaets-Merkmalen mindestens ABSTAND_ID darunter (sonst gaebe es keinen
  Gewinn der Schaltung und G waere undefiniert), (e) keine Klasse < 3 %.

VORREGISTRIERT (vor dem ersten Lauf dieses Moduls, Regel 7):
  Gewinn(t) = mittel_s(acc_voll - acc_ohne) je Aufgabe t;
  G_t(k)   = mittel_s(acc_k - acc_ohne) / Gewinn(t)   (wie clifford.py v3).
  Vorbedingung(t): Gewinn(t) > 0 und exakter Wilcoxon p < 0,05.
  T-1 Positivkontrolle (struktur):
        Vorbedingung(struktur) verletzt        -> HU (Vorbedingung)
        G_struktur(0) >= 0,80                   -> H1 (Mass kann variieren)
        sonst                                   -> H0 (Mass blind: selbst eine
                                                   magiefreie Aufgabe verliert
                                                   beim Runden)
  T-2 echt vs. zufall bei k = 2, gepaart ueber die Seeds:
        SEED-WEISE GROESSE: g_t,s = (acc_k2,s - acc_ohne,s) / Gewinn(t),
        d_s = g_echt,s - g_zufall,s. Mittel(d) = G_echt(2) - G_zufall(2)
        exakt. BEGRUENDUNG: ein Nenner JE SEED (acc_voll,s - acc_ohne,s)
        liegt auf den Ziffern bei ~1,7 pp ± 1 pp, also fuer einzelne Seeds
        nahe 0 -> der Quotient explodiert (Vorzeichen kippt, Betrag > 10) und
        ein einziger Seed bestimmt das Mittel. Der Aufgaben-Nenner Gewinn(t)
        ist ein Mittel ueber 10 Seeds und stabil; innerhalb einer Aufgabe ist
        er eine Konstante (aendert den Wilcoxon nicht), zwischen den
        Aufgaben skaliert er richtig auf „Anteil am eigenen Gewinn“.
        (Preis: der Nenner ist aus denselben Seeds geschaetzt -- eine
        gemeinsame Konstante, keine Seed-Kopplung; offen benannt.) Der
        Vortest erzwingt einen Seed mit Nenner ~0 und prueft, dass der
        Seed-Quotient dort explodiert, die gewaehlte Groesse aber nicht.
        T-1 nicht H1                            -> HU (Positivkontrolle)
        Vorbedingung(echt) oder (zufall) verl.  -> HU (Vorbedingung)
        Wilcoxon p < 0,05 und |mittel(d)| >= 0,15 -> H1 (Aufgabe praegt G)
        zweiseitiges 90-%-KI(d) in (-0,15; 0,15) (TOST) -> H0 (Architektur)
        sonst                                   -> HU
        Schwelle 0,15: kleiner als der Schritt G(2) -> G(4) auf MNIST (0,53
        -> 0,68), d. h. ein Unterschied < 0,15 verschiebt M50/M90 um weniger
        als eine Stufe.
  T-3 beschreibend: G_t(k), M50/M90 je Aufgabe; Kurvenmittel k = 1, 2, 4
      (echt - zufall); G(2) echt - struktur; Nenner-Diagnose (kleinster
      Seed-Gewinn je Aufgabe).
  AUFLOESUNG (offen benannt): Ziffern-Test = 539 Bilder (0,19 pp je Bild).
  Auf echt war der Gewinn in C-clifford-v2 1,7 pp -> g_echt,s streut je Seed
  grob ± 0,7; bei 10 Seeds ist das KI von d etwa ± 0,3 breit. T-2 kann damit
  grosse Unterschiede (H1) finden, Gleichheit (H0) nur, wenn die Streuung
  kleiner ausfaellt. Ein HU ist hier ein ehrliches Ergebnis. Strukturaufgabe:
  ~350 Testbilder (0,29 pp je Bild); zufall: 1.257 Trainingsbilder (0,08 pp).

PROTOKOLL: Ziffern (n = 6, 24 Winkel), Mess-Seeds 60-69 (frisch), Tuning-
Seeds 908/909. Regel 1: eigene Rate je (Aufgabe, Arm) -- voll, ohne_schaltung
UND jede Stufe clifford_k{k} einzeln (wie I-001, nicht nur k = 0 wie
clifford.py), Gitter in beide Richtungen verlaengert (1b), volles Budget
1.500 Schritte (1c). Echt/struktur: Validierung aus dem Trainingsteil; zufall:
Trainingsgenauigkeit (s. o.). Die Voll-Modelle der Tuning-Seeds werden je
Aufgabe einmal mit der gewaehlten Voll-Rate trainiert und fuer die Suche der
Rundungs-Raten geteilt. Keine Abweichung von Regel 1. Rohdaten sofort
sichern (Regel 5).
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from abteilungen.clifford import GITTER, LAGEN, QireZelle, _baue, _gerundet, multiplikationen, runde_clifford
from labor.gemeinsam import genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, ziffern, zusammenfassung

N = 6
AUFGABEN = ["echt", "zufall", "struktur"]
MESSGROESSE = {"echt": "test", "zufall": "training", "struktur": "test"}
MAGIE_STUFEN = [0, 1, 2, 4, 8, 12]
ARME = ["voll", "ohne_schaltung"] + [f"clifford_k{k}" for k in MAGIE_STUFEN]
K_ENTSCHEID = 2
POSITIV_G0 = 0.80
SCHWELLE_T2 = 0.15
ALPHA = 0.05
LR_GITTER = (0.02, 0.08, 0.32)   # C-clifford-v2 fand 0,08 fuer voll/ohne/rundung
LEHRER_SEED = 30_000
ZUFALL_SEED = 40_000
LOESBAR = 0.90
MARGE = 0.25       # Lehrer mit Marge (s. Kopf): Bild gehoert zur Strukturaufgabe, wenn Marge >= 0,25
ABSTAND_ID = 0.10

HYPOTHESEN = {
    "T-1": "Positivkontrolle: auf der magiefreien Strukturaufgabe (Clifford-Lehrer) "
           "erhaelt die Rundung G(0) >= 0,80 -- sonst ist das Mass blind.",
    "T-2": "G(2) unterscheidet sich zwischen echten und zufaelligen Etiketten "
           "(gepaart, exakter Wilcoxon, |Differenz| >= 0,15; TOST fuer Gleichheit).",
    "T-3": "beschreibend: G(k), M50, M90 je Aufgabe.",
}


# ------------------------------------------------------------------ Aufgaben
@lru_cache(maxsize=1)
def _lehrer():
    """Feste Clifford-Schaltung + fester linearer Kopf + Standardisierung."""
    from sklearn.datasets import load_digits
    g = torch.Generator().manual_seed(LEHRER_SEED)
    winkel = torch.randint(0, 4, (LAGEN, N), generator=g).float() * GITTER
    W = torch.randn(10, 2 ** N, generator=g)
    m = QireZelle(n=N)
    m.fixiere(winkel, torch.zeros(LAGEN, N))
    x = torch.tensor(load_digits().data, dtype=torch.float32) / 16.0
    with torch.no_grad():
        s = m.merkmale(x) @ W.T
    return winkel, W, s.mean(0), s.std(0)


def lehrer_merkmale(x):
    winkel, *_ = _lehrer()
    m = QireZelle(n=N)
    m.fixiere(winkel, torch.zeros(LAGEN, N))
    with torch.no_grad():
        return m.merkmale(x)


def _lehrer_punkte(x):
    _, W, mu, sd = _lehrer()
    return (lehrer_merkmale(x) @ W.T - mu) / sd


def struktur_etiketten(x):
    return _lehrer_punkte(x).argmax(1)


def lehrer_marge(x):
    """Abstand bester - zweitbester standardisierter Lehrer-Punktwert je Bild."""
    s = _lehrer_punkte(x).sort(1, descending=True).values
    return s[:, 0] - s[:, 1]


def aufgabe_daten(aufgabe, seed):
    """Dieselben Bilder und derselbe Split wie ziffern(seed); nur die Etiketten
    haengen von der Aufgabe ab."""
    xtr, ytr, xte, yte = ziffern(seed)
    if aufgabe == "zufall":
        g = torch.Generator().manual_seed(ZUFALL_SEED + seed)
        ytr = ytr[torch.randperm(len(ytr), generator=g)]
        yte = yte[torch.randperm(len(yte), generator=g)]
    elif aufgabe == "struktur":
        mtr, mte = lehrer_marge(xtr) >= MARGE, lehrer_marge(xte) >= MARGE
        xtr, xte = xtr[mtr], xte[mte]
        ytr, yte = struktur_etiketten(xtr), struktur_etiketten(xte)
    elif aufgabe != "echt":
        raise ValueError(aufgabe)
    return xtr, ytr, xte, yte


def _teile(aufgabe, s, teil):
    """-> (x_train, y_train, x_mess, y_mess, x_neben, y_neben).
    Messmenge = Validierung (tuning) bzw. Test (mess); bei zufall stets die
    Trainingsmenge selbst (Memorierung). „neben“ = die jeweils andere Menge."""
    xtr, ytr, xte, yte = aufgabe_daten(aufgabe, s)
    if teil == "tuning" and aufgabe != "zufall":
        nv = len(ytr) // 4
        xtr, ytr, xte, yte = xtr[nv:], ytr[nv:], xtr[:nv], ytr[:nv]
    if MESSGROESSE[aufgabe] == "training":
        return xtr, ytr, xtr, ytr, xte, yte
    return xtr, ytr, xte, yte, xtr, ytr


# ------------------------------------------------------------------ Arbeiter
def _voll_aus_winkeln(winkel):
    voll = QireZelle(n=N)
    with torch.no_grad():
        voll.theta.copy_(torch.tensor(winkel).reshape(voll.theta.shape))
    return voll


def _tuning_punkt(a):
    """(aufgabe, arm, lr, seed, schritte, winkel_voll|None) -> Messgroesse (tuning)."""
    aufgabe, arm, lr, s, schritte, winkel = a
    xtr, ytr, xev, yev, _, _ = _teile(aufgabe, s, "tuning")
    if arm.startswith("clifford_k"):
        m = _gerundet(_voll_aus_winkeln(winkel), int(arm[len("clifford_k"):]), s)
    else:
        m, _ = _baue(arm, s, N)
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    return genauigkeit(m(xev), yev)


def _voll_tuning(a):
    """(aufgabe, seed, schritte, lr) -> trainierte Voll-Winkel auf dem Tuning-Teil."""
    aufgabe, s, schritte, lr = a
    xtr, ytr, *_ = _teile(aufgabe, s, "tuning")
    m, _ = _baue("voll", s, N)
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    return m.theta.detach().flatten().tolist()


def _gittersuche(schluessel, aufgabe_fuer, tuning_seeds, gitter=LR_GITTER):
    """Regel 1 (Muster clifford._lr_waehlen): je Schluessel (aufgabe, arm) eigene
    Rate, Gitter in BEIDE Richtungen verlaengert, bis das Optimum innen liegt
    (hoechstens 4 Runden), volles Budget. aufgabe_fuer(schluessel, lr, seed)
    liefert das Argument fuer _tuning_punkt."""
    gitter = {k: sorted(gitter) for k in schluessel}
    wert = {k: {} for k in schluessel}
    offen, info = set(schluessel), {}
    for _ in range(4):
        paare = [(k, lr) for k in sorted(offen) for lr in gitter[k] if lr not in wert[k]]
        erg = parallel(_tuning_punkt, [aufgabe_fuer(k, lr, s) for k, lr in paare for s in tuning_seeds])
        t = len(tuning_seeds)
        for i, (k, lr) in enumerate(paare):
            wert[k][lr] = float(np.mean(erg[i * t:(i + 1) * t]))
        for k in list(offen):
            g = sorted(gitter[k])
            best = max(g, key=lambda v: wert[k][v])
            if best == g[0]:
                gitter[k] = [g[0] / 4] + g
            elif best == g[-1]:
                gitter[k] = g + [g[-1] * 4]
            else:
                info[k] = {"lr": best, "gitter": wert[k], "am_rand": False}
                offen.discard(k)
        if not offen:
            break
    for k in offen:
        info[k] = {"lr": max(wert[k], key=wert[k].get), "gitter": wert[k], "am_rand": True}
    return info


def lernraten(tuning_seeds, schritte):
    """-> {aufgabe: {arm: lr}}, info. Erst voll/ohne je Aufgabe, dann die
    Voll-Modelle der Tuning-Seeds, dann jede Rundungsstufe je Aufgabe."""
    grund = [(t, arm) for t in AUFGABEN for arm in ("voll", "ohne_schaltung")]
    info = _gittersuche(grund, lambda k, lr, s: (k[0], k[1], lr, s, schritte, None), tuning_seeds)
    lr = {t: {arm: info[(t, arm)]["lr"] for arm in ("voll", "ohne_schaltung")} for t in AUFGABEN}
    aufg = [(t, s, schritte, lr[t]["voll"]) for t in AUFGABEN for s in tuning_seeds]
    winkel = {(t, s): w for (t, s, *_), w in zip(aufg, parallel(_voll_tuning, aufg))}
    rund = [(t, f"clifford_k{k}") for t in AUFGABEN for k in MAGIE_STUFEN]
    info.update(_gittersuche(rund, lambda k, lr, s: (k[0], k[1], lr, s, schritte, winkel[(k[0], s)]),
                             tuning_seeds))
    for t, arm in rund:
        lr[t][arm] = info[(t, arm)]["lr"]
    return lr, {f"{t}:{arm}": v for (t, arm), v in info.items()}


def _seed_lauf(a):
    """(aufgabe, seed, schritte, lr{arm}) -> Zeilenteil mit allen Armen."""
    aufgabe, s, schritte, lr = a
    xtr, ytr, xev, yev, xne, yne = _teile(aufgabe, s, "mess")
    z = {"seed": s, "aufgabe": aufgabe, "neben": {}}
    m, _ = _baue("ohne_schaltung", s, N)
    trainiere(m, xtr, ytr, schritte, lr["ohne_schaltung"], seed=s)
    z["ohne_schaltung"], z["neben"]["ohne_schaltung"] = genauigkeit(m(xev), yev), genauigkeit(m(xne), yne)
    voll, _ = _baue("voll", s, N)
    trainiere(voll, xtr, ytr, schritte, lr["voll"], seed=s)
    z["voll"], z["neben"]["voll"] = genauigkeit(voll(xev), yev), genauigkeit(voll(xne), yne)
    th = voll.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    z["abstand_zum_gitter"] = float((th - runde_clifford(th)).abs().mean())
    for k in MAGIE_STUFEN:
        arm = f"clifford_k{k}"
        m = _gerundet(voll, k, s)
        trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
        z[arm], z["neben"][arm] = genauigkeit(m(xev), yev), genauigkeit(m(xne), yne)
    return z


# ------------------------------------------------------------------ Lauf
def lauf(seeds=range(60, 70), schritte=1500, tuning_seeds=(908, 909), schnell=False):
    if schnell:
        seeds, schritte, tuning_seeds = range(2), 150, (908,)
    t0 = time.time()
    lr, lr_info = lernraten(tuning_seeds, schritte)
    print(f"  Raten gewaehlt nach {time.time() - t0:.0f} s", flush=True)
    aufg = [(t, s, schritte, lr[t]) for t in AUFGABEN for s in seeds]
    teile = parallel(_seed_lauf, aufg)
    zeilen = {s: {"seed": s} for s in seeds}
    for z in teile:
        zeilen[z["seed"]][z.pop("aufgabe")] = {k: v for k, v in z.items() if k != "seed"}
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + "  ".join(
            f"{t}: voll {z[t]['voll']:.4f} ohne {z[t]['ohne_schaltung']:.4f} k2 {z[t]['clifford_k2']:.4f}"
            for t in AUFGABEN), flush=True)
    ergebnis = {"abteilung": "clifford", "modul": "magie_aufgabe", "karte": "I-003", "version": 1,
                "datensatz": "ziffern", "qubits": N, "hypothesen": HYPOTHESEN, "messgroesse": MESSGROESSE,
                "lr": lr_info, "schritte": schritte, "seeds": list(seeds), "tuning_seeds": list(tuning_seeds),
                "lehrer_winkel": _lehrer()[0].flatten().tolist(), "zeilen": zeilen,
                "sekunden": time.time() - t0,
                "multiplikationen": {"voll": multiplikationen(N),
                                     **{f"clifford_k{k}": multiplikationen(N, magie=k) for k in MAGIE_STUFEN}}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def _gepaart(d, einseitig_q=0.95):
    """Mittel, exakter zweiseitiger Wilcoxon, KI aus t-Quantil einseitig_q
    (0,95 -> zweiseitiges 90-%-KI = zwei einseitige 95-%-Tests, TOST)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    if len(d) < 2:
        return {"mittel": m, "p": 1.0, "ki_unten": -math.inf, "ki_oben": math.inf}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(einseitig_q, len(d) - 1))
    return {"mittel": m, "p": wilcoxon_exakt(d), "ki_unten": m - q * se, "ki_oben": m + q * se}


def auswertung(zeilen) -> dict:
    aus, g_seed = {}, {}
    for t in AUFGABEN:
        a = lambda arm: np.array([z[t][arm] for z in zeilen], dtype=float)
        d0 = a("voll") - a("ohne_schaltung")
        gewinn = float(d0.mean())
        e = {arm: zusammenfassung(a(arm)) for arm in ARME}
        e["gewinn"], e["p_gewinn"] = gewinn, wilcoxon_exakt(d0)
        e["vorbedingung"] = bool(gewinn > 0 and e["p_gewinn"] < ALPHA)
        skala = gewinn if gewinn > 0 else 1.0          # bei verletzter Vorbedingung nur beschreibend
        g_seed[t] = {k: (a(f"clifford_k{k}") - a("ohne_schaltung")) / skala for k in MAGIE_STUFEN}
        e["G"] = {str(k): (float(g_seed[t][k].mean()) if gewinn > 0 else None) for k in MAGIE_STUFEN}
        for stufe in (50, 90):
            e[f"M{stufe}"] = next((k for k in MAGIE_STUFEN
                                   if e["G"][str(k)] is not None and e["G"][str(k)] >= stufe / 100), None)
        e["G0_ki"] = _gepaart(g_seed[t][0])
        # Nenner-Diagnose: Seed-Quotient (NICHT die Teststatistik, nur zur Offenlegung)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (a(f"clifford_k{K_ENTSCHEID}") - a("ohne_schaltung")) / d0
        e["seed_gewinn_min"] = float(d0.min())
        e["seed_quotient_G2_max_betrag"] = float(np.nanmax(np.abs(q))) if np.isfinite(q).any() else None
        if "neben" in zeilen[0][t]:
            e["neben"] = {arm: zusammenfassung([z[t]["neben"][arm] for z in zeilen]) for arm in ARME}
        if "abstand_zum_gitter" in zeilen[0][t]:
            e["abstand_zum_gitter"] = zusammenfassung([z[t]["abstand_zum_gitter"] for z in zeilen])
        aus[t] = e
    kk = K_ENTSCHEID
    aus["T2"] = _gepaart(g_seed["echt"][kk] - g_seed["zufall"][kk])
    kurve = lambda t: np.mean([g_seed[t][k] for k in (1, 2, 4)], axis=0)
    aus["T3_kurvenmittel_echt_zufall"] = _gepaart(kurve("echt") - kurve("zufall"))
    aus["T3_G2_echt_struktur"] = _gepaart(g_seed["echt"][kk] - g_seed["struktur"][kk])
    aus["n"] = len(zeilen)
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Eindeutige Praefixe:
    'H1:', 'H0:', 'HU:', 'HU (Vorbedingung):', 'HU (Positivkontrolle):'."""
    u = {}
    s = a["struktur"]
    if not s["vorbedingung"]:
        u["T-1"] = (f"HU (Vorbedingung): Schaltung schlaegt die Identitaet auf der Strukturaufgabe "
                    f"nicht nachweisbar (Gewinn {s['gewinn']:+.4f}, p = {s['p_gewinn']:.4f})")
    elif s["G"]["0"] >= POSITIV_G0:
        u["T-1"] = (f"H1: Positivkontrolle bestanden -- G_struktur(0) = {s['G']['0']:.2f} >= {POSITIV_G0} "
                    f"(Gewinn {s['gewinn']:+.4f}); das Mass kann M50 = 0 anzeigen")
    else:
        u["T-1"] = (f"H0: Mass blind -- auch die magiefreie Aufgabe verliert beim Runden "
                    f"(G_struktur(0) = {s['G']['0']:.2f} < {POSITIV_G0})")
    d = a["T2"]
    if not u["T-1"].startswith("H1:"):
        u["T-2"] = "HU (Positivkontrolle): T-1 nicht bestanden -- Vergleich echt/zufall nicht auswertbar"
    elif not (a["echt"]["vorbedingung"] and a["zufall"]["vorbedingung"]):
        fehlt = [t for t in ("echt", "zufall") if not a[t]["vorbedingung"]]
        u["T-2"] = (f"HU (Vorbedingung): kein nachweisbarer Gewinn auf {', '.join(fehlt)} -- G undefiniert")
    elif d["p"] < ALPHA and abs(d["mittel"]) >= SCHWELLE_T2:
        u["T-2"] = (f"H1: Aufgabe praegt G(2) -- echt - zufall = {d['mittel']:+.3f} "
                    f"(p = {d['p']:.4f}; G echt {a['echt']['G']['2']:.2f}, zufall {a['zufall']['G']['2']:.2f})")
    elif -SCHWELLE_T2 < d["ki_unten"] and d["ki_oben"] < SCHWELLE_T2:
        u["T-2"] = (f"H0: G(2) gleich im Rahmen ±{SCHWELLE_T2} (TOST, 90-%-KI [{d['ki_unten']:+.3f}; "
                    f"{d['ki_oben']:+.3f}]) -- Konzentration ist Architektur, nicht Aufgabe")
    else:
        u["T-2"] = (f"HU: echt - zufall = {d['mittel']:+.3f}, p = {d['p']:.4f}, "
                    f"90-%-KI [{d['ki_unten']:+.3f}; {d['ki_oben']:+.3f}]")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    u["T-3"] = "; ".join(
        f"{t}: G(k) " + ", ".join(fmt(a[t]["G"][str(k)]) for k in MAGIE_STUFEN)
        + f" -> M50 = {a[t]['M50']}, M90 = {a[t]['M90']}" for t in AUFGABEN) + f" (k = {MAGIE_STUFEN})"
    return u


# ------------------------------------------------------------------ Vortest
def _kopf_genauigkeit(ftr, ytr, fte, yte, schritte=1500):
    """Bester linearer Kopf (wie QireZelle.kopf) auf festen Merkmalen, Raten 0,02-1,28."""
    beste = 0.0
    for lr in (0.02, 0.08, 0.32, 1.28):
        torch.manual_seed(0)
        kopf = nn.Linear(ftr.shape[1], 10)
        trainiere(kopf, ftr, ytr, schritte, lr, seed=0)
        with torch.no_grad():
            beste = max(beste, genauigkeit(kopf(fte), yte))
    return beste


def _vortest_struktur() -> tuple[bool, dict]:
    """Bausteine: Lehrer echt Clifford und nicht entartet, Strukturaufgabe loesbar
    und nicht schon von der Identitaet geloest, Zufallsetiketten wirklich zufaellig."""
    ok, info = True, {}
    winkel, *_ = _lehrer()
    if not torch.allclose(winkel / GITTER, torch.round(winkel / GITTER)):
        ok = False
        print("  VORTEST FEHLER: Lehrer-Winkel nicht auf dem Clifford-Gitter")
    m = QireZelle(n=N)
    m.fixiere(winkel, torch.zeros(LAGEN, N))
    U = m.unitaer().detach()
    nichtnull = (U.abs() > 1e-6).sum(1).float()
    info["lehrer_nichtnull_je_zeile"] = float(nichtnull.mean())
    if float(nichtnull.mean()) <= 1.0:
        ok = False
        print("  VORTEST FEHLER: Lehrer ist eine (Vorzeichen-)Permutation -- entartet")
    xtr, ytr, xte, yte = aufgabe_daten("struktur", 0)
    if not (torch.allclose(m.merkmale(xtr).detach(), lehrer_merkmale(xtr))
            and bool((lehrer_marge(xtr) >= MARGE).all()) and bool((lehrer_marge(xte) >= MARGE).all())):
        ok = False
        print("  VORTEST FEHLER: Lehrer-Merkmale oder Margenfilter falsch")
    from sklearn.datasets import load_digits
    x_alle = torch.tensor(load_digits().data, dtype=torch.float32) / 16.0
    alle = struktur_etiketten(x_alle[lehrer_marge(x_alle) >= MARGE])
    info["struktur_anteil_bilder"] = round(len(alle) / len(x_alle), 3)
    anteil = torch.bincount(alle, minlength=10).float() / len(alle)
    info["klassenanteile"] = [round(float(v), 3) for v in anteil]
    if float(anteil.min()) < 0.03:
        ok = False
        print("  VORTEST FEHLER: Strukturaufgabe mit fast leerer Klasse", info["klassenanteile"])
    acc_lehrer = _kopf_genauigkeit(lehrer_merkmale(xtr), ytr, lehrer_merkmale(xte), yte)
    ohne, _ = _baue("ohne_schaltung", 0, N)
    with torch.no_grad():
        ftr_id, fte_id = ohne.merkmale(xtr), ohne.merkmale(xte)
    acc_id = _kopf_genauigkeit(ftr_id, ytr, fte_id, yte)
    info["kopf_auf_lehrer_merkmalen"], info["kopf_auf_identitaet"] = acc_lehrer, acc_id
    if acc_lehrer < LOESBAR:
        ok = False
        print(f"  VORTEST FEHLER: Strukturaufgabe mit richtigen Merkmalen nicht loesbar ({acc_lehrer:.3f})")
    if acc_lehrer - acc_id < ABSTAND_ID:
        ok = False
        print(f"  VORTEST FEHLER: Identitaet loest die Strukturaufgabe fast so gut ({acc_id:.3f} vs "
              f"{acc_lehrer:.3f}) -- kein Raum fuer einen Gewinn der Schaltung")
    # Identitaets-Kontrolle ist eine reine Permutation (Kopf gleicht sie exakt aus)
    Uo = ohne.unitaer().detach()
    if not (torch.allclose(Uo.abs().sum(1), torch.ones(len(Uo))) and int((Uo.abs() > 1e-6).sum()) == len(Uo)):
        ok = False
        print("  VORTEST FEHLER: Identitaets-Kontrolle ist keine Permutation")
    # Zufallsetiketten: gleiche Klassenhaeufigkeiten, Bilder gemischt, je Seed fest und verschieden
    _, ye, _, _ = aufgabe_daten("echt", 0)
    _, yz, _, _ = aufgabe_daten("zufall", 0)
    _, yz2, _, _ = aufgabe_daten("zufall", 0)
    _, yz1, _, _ = aufgabe_daten("zufall", 1)
    info["zufall_gleich_echt"] = float((ye == yz).float().mean())
    if not torch.equal(torch.bincount(ye, minlength=10), torch.bincount(yz, minlength=10)) \
            or info["zufall_gleich_echt"] > 0.2 or not torch.equal(yz, yz2) or torch.equal(yz[:len(yz1)], yz1[:len(yz)]):
        ok = False
        print("  VORTEST FEHLER: Zufallsetiketten nicht wie beabsichtigt", info["zufall_gleich_echt"])
    # Messgroessen: zufall misst auf der Trainingsmenge, echt/struktur auf Test/Validierung
    for teil in ("tuning", "mess"):
        xa, ya, xm, ym, _, _ = _teile("zufall", 0, teil)
        if not (torch.equal(xa, xm) and torch.equal(ya, ym)):
            ok = False
            print("  VORTEST FEHLER: zufall misst nicht auf der Trainingsmenge", teil)
        xa, _, xm, _, _, _ = _teile("echt", 0, teil)
        if len(xa) + len(xm) > 1797 or any(torch.equal(xa[0], r) for r in xm[:5]):
            ok = False
            print("  VORTEST FEHLER: echt misst nicht auf getrennter Menge", teil)
    # _gerundet: k freie Winkel, Rest auf dem Gitter, frischer Kopf
    torch.manual_seed(1)
    voll = QireZelle(n=N)
    for k in MAGIE_STUFEN:
        g = _gerundet(voll, k, 7)
        w, frei = g.winkel().detach().flatten(), g.frei.flatten().bool()
        if int(frei.sum()) != k or not torch.allclose(w[~frei] / GITTER, torch.round(w[~frei] / GITTER)) \
                or torch.equal(g.kopf.weight, voll.kopf.weight):
            ok = False
            print("  VORTEST FEHLER: _gerundet verhaelt sich nicht wie erwartet, k =", k)
    return ok, info


def _synth(rng, g, gewinn=None, sd=0.003, n=10, nenner_null=None):
    """Synthetische Zeilen. g = {aufgabe: (G0, G2)}, gewinn = {aufgabe: Gewinn}.
    Kurve: G(1) = Mitte, G(4) / G(8) / G(12) naehern sich 1. nenner_null = Aufgabe,
    deren Seed 0 fast keinen Gewinn hat (voll = ohne + 0,0005)."""
    gewinn = {t: 0.03 for t in AUFGABEN} | (gewinn or {})
    zeilen = []
    for s in range(n):
        z = {"seed": s}
        for t in AUFGABEN:
            g0, g2 = g[t]
            kurve = {0: g0, 1: (g0 + g2) / 2, 2: g2, 4: g2 + 0.4 * (1 - g2), 8: g2 + 0.8 * (1 - g2), 12: 1.0}
            ohne = 0.5 + rng.normal(0, sd)
            e = {"ohne_schaltung": ohne, "voll": ohne + gewinn[t] + rng.normal(0, sd)}
            if nenner_null == t and s == 0:
                e["voll"] = ohne + 0.0005
            for k, gk in kurve.items():
                e[f"clifford_k{k}"] = ohne + gewinn[t] * gk + rng.normal(0, sd)
            z[t] = e
        zeilen.append(z)
    return zeilen


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Bausteine + jeden Zweig auf synthetischen Zeilen erzwingen.
    Null- und Grenzfaelle kippen mit Rate ~alpha -- dort Toleranz."""
    ok, info = _vortest_struktur()
    print(f"  Bausteine: Kopf auf Lehrer-Merkmalen {info.get('kopf_auf_lehrer_merkmalen', 0):.3f}, "
          f"auf Identitaet {info.get('kopf_auf_identitaet', 0):.3f}, Lehrer {info.get('lehrer_nichtnull_je_zeile', 0):.1f} "
          f"Nicht-Null je Zeile, Klassen {info.get('klassenanteile')}, Bildanteil {info.get('struktur_anteil_bilder')}", flush=True)
    rng = np.random.default_rng(3)
    gut = {"echt": (0.1, 0.5), "zufall": (0.1, 0.5), "struktur": (0.95, 0.97)}
    faelle = [
        (dict(g=gut | {"zufall": (0.3, 0.95)}), {"T-1": "H1:", "T-2": "H1:"}, 0),
        (dict(g=gut | {"zufall": (0.0, 0.05)}), {"T-1": "H1:", "T-2": "H1:"}, 0),
        # Gleichheit: TOST-KI (~±0,06) liegt klar in ±0,15, selten Kipper
        (dict(g=gut, sd=0.002), {"T-1": "H1:", "T-2": "H0:"}, 1),
        (dict(g=gut | {"struktur": (0.40, 0.60)}), {"T-1": "H0:", "T-2": "HU (Positivkontrolle):"}, 0),
        (dict(g=gut, gewinn={"struktur": -0.005}), {"T-1": "HU (Vorbedingung):", "T-2": "HU (Positivkontrolle):"}, 0),
        (dict(g=gut, gewinn={"zufall": -0.005}), {"T-1": "H1:", "T-2": "HU (Vorbedingung):"}, 0),
        (dict(g=gut, gewinn={"echt": 0.0}), {"T-2": "HU (Vorbedingung):"}, 1),
        # Grenzfall: wahre Differenz genau 0,15, grosse Streuung -> meist HU; H1/H0
        # je mit Rate ~alpha, daher Toleranz 2 von 5.
        (dict(g=gut | {"zufall": (0.1, 0.65)}, sd=0.015, gewinn={"struktur": 0.3}), {"T-1": "H1:", "T-2": "HU:"}, 2),
    ]
    for par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth(rng, **par)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest: {h} -> {u[h]}")
        for h, f in fehl.items():
            if f > toleranz:
                ok = False
                print(f"  VORTEST FEHLER: {h}: {f} von {ziehungen} falsch bei {par}")
    # Nenner-Robustheit: ein Seed mit Gewinn ~0 auf echt, sonst gleiche G. Der
    # Seed-Quotient muss dort explodieren (Szenario wirklich erzeugt), die gewaehlte
    # Groesse darf KEIN H1 liefern.
    fehl = 0
    for _ in range(ziehungen):
        a = auswertung(_synth(rng, g=gut, sd=0.002, nenner_null="echt"))
        if not (a["echt"]["seed_quotient_G2_max_betrag"] or 0) > 5:
            ok = False
            print("  VORTEST FEHLER: Nenner-Szenario nicht erzeugt", a["echt"]["seed_quotient_G2_max_betrag"])
        if urteil(a)["T-2"].startswith("H1:"):
            fehl += 1
    if fehl > 1:
        ok = False
        print(f"  VORTEST FEHLER: Seed mit Nenner ~0 erzeugt H1 ({fehl} von {ziehungen})")
    # Aufloesung: echte Differenz 0, aber riesige Streuung -> darf nicht H0 sein
    for _ in range(ziehungen):
        u = urteil(auswertung(_synth(rng, g=gut, sd=0.03, gewinn={"struktur": 0.5})))
        if u["T-2"].startswith("H0:") or not u["T-1"].startswith("H1:"):
            ok = False
            print("  VORTEST FEHLER: Aufloesungspruefung -- H0 trotz riesiger Streuung", u["T-2"])
    # M50/M90 aus der Kurve
    a = auswertung(_synth(rng, g=gut, sd=0.0))
    if not (a["echt"]["M50"] == 2 and a["struktur"]["M50"] == 0 and a["struktur"]["M90"] == 0):
        ok = False
        print("  VORTEST FEHLER: M50/M90 falsch", a["echt"]["M50"], a["struktur"]["M50"], a["struktur"]["M90"])
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-003" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
