"""Abteilung Clifford -- Magie gegen Datenmenge (Karte I-024, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  Wissen, ob der Magiebedarf der QIRE-Zelle (Zahl der
      Nicht-Clifford-Winkel = T-Gatter, C-hardware: T ~ 31*k) von der
      DATENLAGE abhaengt. Wenn die Zelle mit wenig Trainingsdaten weniger
      Magie braucht, wirkt das Clifford-Gitter (k = 0: 0 T-Gatter) als
      Regularisierer, und auf Hardware hiesse das: wenig Daten -> fast T-frei.
  Wie heute?  C-mnist misst den Abstand voll - k=0 (3,2 pp) und M50 = 2 nur
      bei EINER Datenmenge (6.000 Bilder). I-003 fragt Aufgabe gegen
      Architektur, nicht die Datenmenge.
  Was ist neu?  Dieselbe Pipeline (clifford.py: QireZelle, _baue, _gerundet
      mit Rundung nach Abstand und FRISCHEM Kopf) bei n_train = 500 / 2.000 /
      10.000 auf GESCHACHTELTEN Trainingsmengen (je Seed ist die 500er-Menge
      Teil der 2.000er, diese Teil der 10.000er: bilder() zieht eine feste
      Permutation je Seed und schneidet vorne ab) und IDENTISCHER Testmenge
      (5.000 Bilder, unabhaengig von n_train). Gepaart wird ueber den
      Seed-Index: gleicher Start der Winkel, gleiche zufaellige Clifford-
      Kontrolle, gleiche Testbilder -- nur die Datenmenge aendert sich.
  Woran merken wir es?  Abstand Delta(n) = acc(voll) - acc(k=0) und
      Gewinnanteil G(2) bzw. M50 als Funktion von n.

ARME (je Seed und je n_train, MNIST 16x16, 8 Qubits, L = 4, 32 Winkel):
  voll            -- alle 32 Winkel frei (Magie 32).
  clifford_k{k}   -- k in MAGIE_STUFEN = 0, 1, 2, 4, 8: Winkel des voll-Modells
                     (desselben Seeds und DERSELBEN Datenmenge) auf k*pi/2
                     gerundet, die k mit dem groessten Abstand zum Gitter
                     bleiben frei (Start beim trainierten Wert), Kopf FRISCH
                     (Lehre „geerbter Kopf“). Die Karte nennt k = 0/2/8; k = 1
                     und 4 kommen dazu, weil M50 auf MNIST zwischen 1 und 2
                     liegt (C-mnist: G(1) = 0,49, G(2) = 0,53) -- mit
                     {0, 2, 8} koennte eine Verschiebung von M50 um eine Stufe
                     gar nicht sichtbar werden.
  ohne_schaltung  -- Identitaet (alle Winkel 0 + CNOT-Ring = reine
                     Permutation, der Kopf gleicht sie exakt aus). Kontrolle je n.
  zufall_clifford -- feste zufaellige Clifford-Schaltung (clifford._baue,
                     Generator 10.000 + seed, je Seed fuer alle n gleich).
  Der Vortest prueft, dass die Identitaet eine Permutation ist und dass die
  Zufallskontrolle nicht in allen Mess-Seeds zur (Vorzeichen-)Permutation
  entartet (Lehre „Kontrollen muessen tun, was ihr Name sagt“).

BUDGET UND KOPF-REGULARISIERUNG (Designentscheidung vor dem Lauf):
  * Vollbatch-Adam, SCHRITTE = 1.500 in allen Armen und bei allen n. Im
    Vollbatch ist ein Schritt eine Epoche -- „gleiche Schritte“ und „gleiche
    Epochen“ fallen zusammen. Gleiche Epochen im Minibatch-Sinn haetten bei
    n = 500 nur 1/20 der Updates -> Unteranpassung statt Datenmangel.
  * n_train ist das GESAMTE Etikettenbudget: davon 20 % (ANTEIL_VAL) als
    Validierung fuer fruehes Stoppen, 80 % Training (n = 500 -> 400 / 100).
    Fruehes Stoppen: alle TAKT = 25 Schritte (und bei Schritt 0) wird der
    Validierungs-Kreuzentropieverlust gemessen, das Modell mit dem kleinsten
    Wert wird genommen. Gleiche Regel in ALLEN Armen, keine Gewichtsabnahme.
    Grund: Der lineare Kopf (2.570 Parameter) ueberanpasst bei 400 Bildern
    ohnehin; eine Gewichtsabnahme muesste je n (und fair je Arm) getunt
    werden, fruehes Stoppen hat bei fester Aufteilung keinen freien Parameter
    und benutzt nur Etiketten innerhalb des Budgets.
  * Arme mit festen Winkeln (ohne_schaltung, zufall_clifford, clifford_k0)
    trainieren mathematisch nur den Kopf (Gradient nach theta ist exakt 0,
    Adam-Schritt 0). Sie laufen deshalb auf einmal berechneten Merkmalen --
    der Vortest prueft Gleichheit mit dem vollen Weg.

VORREGISTRIERT (vor dem ersten Lauf dieses Moduls, Regel 7):
  Gewinn(n) = mittel_s(acc_voll - acc_ohne); Delta_s(n) = acc_voll,s - acc_k0,s;
  G_n(k) = mittel_s(acc_k - acc_ohne) / Gewinn(n) (wie clifford.py v3);
  M50/M90(n) = kleinste Stufe mit G_n(k) >= 0,5 / 0,9.
  D-0 Positivkontrolle: bei n = 10.000 schlaegt voll die Identitaet
      (Gewinn > 0, exakter Wilcoxon zweiseitig p < 0,05). Sonst D-1/D-2
      „HU (Vorbedingung)“.
  D-1 (Haupt) d_s = Delta_s(10.000) - Delta_s(500)  (> 0: Abstand schrumpft).
      Vorbedingung: D-0 H1 UND Delta(10.000) > 0 mit p < 0,05 (Magie wird bei
      vollen Daten gebraucht -- sonst gibt es nichts, was schrumpfen kann).
        Vorbedingung verletzt                               -> HU (Vorbedingung)
        exakter Wilcoxon EINSEITIG (d > 0), Holm-p < 0,05 UND
        mittel(d) >= 0,5 pp                                 -> H1
        einseitig (d < 0) p < 0,05 UND mittel(d) <= -0,5 pp -> H0 (Gegenrichtung:
                                                               wenig Daten brauchen MEHR Magie)
        TOST: zweiseitiges 90-%-KI von Delta(10.000) - Delta(n) liegt fuer
        n = 500 UND n = 2.000 ganz in (-0,5; +0,5) pp      -> H0 (Abbruch der Karte)
        sonst                                               -> HU
      Die Mindestgroesse 0,5 pp fuer H1 ist dieselbe Grenze wie im TOST: ein
      signifikantes, aber kleineres Schrumpfen ist „praktisch gleich“.
      Grenze 0,5 pp aus der Karte = ~1/6 des Abstands bei 6.000 Bildern.
  D-2 G(2) als Funktion von n: e_s = g_s(500) - g_s(10.000),
      g_s(n) = (acc_k2,s - acc_ohne,s) / Gewinn(n) (Nenner je n ueber alle
      Seeds gemittelt, Begruendung wie I-003 T-2: ein Nenner je Seed
      explodiert bei kleinem Seed-Gewinn). e > 0: bei wenig Daten erhalten 2
      Magiewinkel einen groesseren Teil des Gewinns = weniger Magie noetig.
      Vorbedingung: Gewinn(500) und Gewinn(10.000) je > 0 mit p < 0,05.
        Vorbedingung verletzt                               -> HU (Vorbedingung)
        einseitig (e > 0) Holm-p < 0,05 UND mittel(e) >= 0,15 -> H1
        einseitig (e < 0) p < 0,05 UND mittel(e) <= -0,15   -> H0 (Gegenrichtung)
        TOST: 90-%-KI von g(n) - g(10.000) fuer n = 500 und 2.000 in
        (-0,15; 0,15) (fuer n mit Gewinn(n) <= 0 nicht erfuellt) -> H0
        sonst                                               -> HU
      Schwelle 0,15 wie I-003: kleiner als der Schritt G(2) -> G(4) auf MNIST.
  HOLM: Familie = {D-1, D-2} (nur die einseitigen H1-Tests; nur die, deren
      Vorbedingung erfuellt ist). D-0 ist Vorbedingung, D-3 beschreibend.
  D-3 beschreibend: je n Gewinn, Delta, G(k)-Kurve, M50/M90, Regularisierer-
      Sicht acc(k0) - acc(voll) (> 0 hiesse: Gitter schlaegt volles Modell),
      acc(k0) - acc(zufall_clifford), Generalisierungsluecke (Training - Test)
      von voll und k0, Stopp-Schritt des fruehen Stoppens (Median).

AUFLOESUNG / TESTSTAERKE (offen benannt): In C-mnist (6.000 Bilder) streut
  Delta je Seed mit SD 1,4 pp (k0 allein 1,6 pp). Mit 10 Seeds ist das
  90-%-KI einer Differenz zweier Abstaende grob +-1 pp breit -- der TOST mit
  +-0,5 pp wird nur gelingen, wenn die Paarung ueber die geschachtelten
  Mengen die Streuung stark senkt. H1 findet ein Schrumpfen um >~1,5 pp.
  Ein HU ist hier ein ehrliches Ergebnis; mehr Seeds ueber `seeds=`.

PROTOKOLL: MNIST 16x16, Mess-Seeds 100-109 (frisch), Tuning-Seeds 912/913.
  Regel 1: eigene Rate je (Arm, n) -- auch jede Stufe clifford_k{k} einzeln
  --, Gitter 0,02/0,08/0,32 in beide Richtungen verlaengert (1b), volles
  Budget (1c). Tuning: Training/Stopp-Validierung wie im Messlauf, bewertet
  auf 2.000 weiteren Trainingsbildern (Pool-Index 10.000-11.999 des Tuning-
  Seeds), nie auf Testbildern. Die Voll-Modelle der Tuning-Seeds werden je n
  einmal mit der gewaehlten Voll-Rate trainiert und fuer die Suche der
  Rundungs-Raten geteilt. BLAS-Threads 1 (labor.parallel). Rohdaten sofort
  sichern (Regel 5).

LAUFZEIT (gemessen auf belastetem 4-Kern-Rechner, 1 Thread): ein Schritt
  voll ~16 / 24 / 60 ms bei 400 / 1.600 / 8.000 Trainingsbildern -> ~150 s
  je (Arm mit freien Winkeln, Seed) ueber alle drei n; 5 solche Arme x 10
  Seeds + Lernratensuche ~ 14.000 CPU-s -> mit INSTITUT_PROZESSE=4 ~40-60 min.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from abteilungen.clifford import (GITTER, LAGEN, QireZelle, _baue, _gerundet, multiplikationen,
                                  runde_clifford)
from labor.gemeinsam import bilder, genauigkeit, parallel, speichere, wilcoxon_exakt, zusammenfassung

DATENSATZ = "mnist"
N = 8                                   # 16x16 = 256 = 2^8 Amplituden
N_WERTE = (500, 2_000, 10_000)
N_TUNING_EVAL = 2_000                   # Bewertungsbilder der Tuning-Seeds (aus dem Trainingspool)
N_TEST = 5_000
ANTEIL_VAL = 0.2
SCHRITTE = 1_500
TAKT = 25
MAGIE_STUFEN = [0, 1, 2, 4, 8]
K_ENTSCHEID = 2
ARME_FEST = ["voll", "ohne_schaltung", "zufall_clifford"]
ARME_RUND = [f"clifford_k{k}" for k in MAGIE_STUFEN]
ARME = ARME_FEST + ARME_RUND
MESS_SEEDS = tuple(range(100, 110))
TUNING_SEEDS = (912, 913)
LR_GITTER = (0.02, 0.08, 0.32)          # C-mnist fand 0,08 fuer voll/ohne/rundung
ALPHA = 0.05
MARGE_D1 = 0.005                        # 0,5 pp (Karte)
MARGE_D2 = 0.15                         # Anteil G (wie I-003)
T_JE_WINKEL = 31                        # C-hardware: T ~ 31*k bei eps = 1e-3

HYPOTHESEN = {
    "D-0": "Positivkontrolle: bei n_train = 10.000 schlaegt voll die Identitaet "
           "(Gewinn > 0, exakter Wilcoxon p < 0,05). Sonst kein Urteil in D-1/D-2.",
    "D-1": "Der Magie-Abstand Delta(n) = acc(voll) - acc(k=0) ist bei n_train = 500 "
           "um >= 0,5 pp kleiner als bei 10.000 (gepaart ueber den Seed-Index, "
           "exakter Wilcoxon einseitig, Holm ueber D-1/D-2). H0 (Abbruch der Karte): "
           "Delta(500) und Delta(2.000) liegen per TOST innerhalb +-0,5 pp von Delta(10.000).",
    "D-2": "Mit wenig Daten erhalten 2 Magiewinkel einen groesseren Anteil des "
           "Gewinns: G_500(2) - G_10000(2) >= 0,15 (einseitig, Holm); TOST +-0,15 fuer H0.",
    "D-3": "beschreibend: je n Gewinn, Delta, G(k), M50/M90, acc(k0) - acc(voll) "
           "(Regularisierer-Sicht), k0 gegen zufaellige Clifford, Generalisierungsluecke.",
}


# ------------------------------------------------------------------ Daten
@lru_cache(maxsize=16)
def _daten(seed):
    """Pool N_max + N_TUNING_EVAL Trainingsbilder (Praefixe sind die kleineren
    Mengen) und die feste Testmenge des Seeds."""
    return bilder(DATENSATZ, seed, n_train=max(N_WERTE) + N_TUNING_EVAL, n_test=N_TEST, seite=16)


def _teile(seed, n, teil):
    """-> Training, Stopp-Validierung (beide aus den ersten n Bildern), Bewertung
    (teil 'tuning': Pool-Bilder hinter N_max; 'mess': Testmenge)."""
    x, y, xte, yte = _daten(seed)
    nv = int(round(n * ANTEIL_VAL))
    xs, ys = x[:n], y[:n]
    if teil == "tuning":
        n_max = max(N_WERTE)
        xev, yev = x[n_max:n_max + N_TUNING_EVAL], y[n_max:n_max + N_TUNING_EVAL]
    else:
        xev, yev = xte, yte
    return xs[nv:], ys[nv:], xs[:nv], ys[:nv], xev, yev


# ------------------------------------------------------------------ Training
def trainiere_mit_stopp(m, xtr, ytr, xva, yva, schritte, lr, seed, takt=TAKT, abkuerzung=True):
    """Vollbatch-Adam wie labor.trainiere, dazu fruehes Stoppen: bei Schritt 0,
    alle `takt` Schritte und am Ende Validierungsverlust messen, bestes Modell
    zurueckladen. Bei festen Winkeln (frei == 0) auf einmal berechneten
    Merkmalen (identisch, da der theta-Gradient exakt 0 ist)."""
    fest = abkuerzung and float(m.frei.sum()) == 0.0
    if fest:
        with torch.no_grad():
            ftr, fva = m.merkmale(xtr), m.merkmale(xva)
        vor_tr, vor_va, params = (lambda: m.kopf(ftr)), (lambda: m.kopf(fva)), list(m.kopf.parameters())
    else:
        vor_tr, vor_va, params = (lambda: m(xtr)), (lambda: m(xva)), list(m.parameters())
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    bester = {"val_verlust": math.inf, "stopp": -1, "zustand": None}

    def pruefe(schritt):
        with torch.no_grad():
            v = F.cross_entropy(vor_va(), yva).item()
        if v < bester["val_verlust"]:
            bester.update(val_verlust=v, stopp=schritt,
                          zustand={k: t.detach().clone() for k, t in m.state_dict().items()})

    pruefe(0)
    for i in range(1, schritte + 1):
        opt.zero_grad()
        F.cross_entropy(vor_tr(), ytr).backward()
        opt.step()
        if i % takt == 0 or i == schritte:
            pruefe(i)
    m.load_state_dict(bester["zustand"])
    return {"val_verlust": bester["val_verlust"], "stopp": bester["stopp"]}


def _voll_aus_winkeln(winkel):
    voll = QireZelle(n=N)
    with torch.no_grad():
        voll.theta.copy_(torch.tensor(winkel, dtype=torch.float32).reshape(voll.theta.shape))
    return voll


def _baue_arm(arm, seed, winkel=None):
    if arm.startswith("clifford_k"):
        return _gerundet(_voll_aus_winkeln(winkel), int(arm[len("clifford_k"):]), seed)
    m, _ = _baue(arm, seed, N)
    return m


def _arbeit(a):
    """(arm, n, lr, seed, teil, schritte, winkel_voll|None) -> Messzeile."""
    arm, n, lr, seed, teil, schritte, winkel = a
    xtr, ytr, xva, yva, xev, yev = _teile(seed, n, teil)
    m = _baue_arm(arm, seed, winkel)
    info = trainiere_mit_stopp(m, xtr, ytr, xva, yva, schritte, lr, seed)
    with torch.no_grad():
        z = {"arm": arm, "n": n, "lr": lr, "seed": seed, "acc": genauigkeit(m(xev), yev),
             "acc_train": genauigkeit(m(xtr), ytr), **info}
    if arm == "voll":
        z["winkel"] = m.theta.detach().flatten().tolist()
    return z


def _lr_suche(schluessel, seeds, schritte, winkel_von=None, gitter=LR_GITTER, runden=4):
    """Regel 1: eigene Rate je (arm, n) auf den Tuning-Seeds; liegt das Optimum am
    Rand, wird das Gitter in diese Richtung verlaengert (beide Richtungen, 1b)."""
    winkel_von = winkel_von or {}
    gitter = {s: sorted(gitter) for s in schluessel}
    wert = {s: {} for s in schluessel}
    offen, info = set(schluessel), {}
    for _ in range(runden):
        aufg = [(arm, n, lr, s, "tuning", schritte, winkel_von.get((s, n)))
                for (arm, n) in sorted(offen) for lr in gitter[(arm, n)] if lr not in wert[(arm, n)]
                for s in seeds]
        tmp = {}
        for z in parallel(_arbeit, aufg):
            tmp.setdefault((z["arm"], z["n"], z["lr"]), []).append(z["acc"])
        for (arm, n, lr), v in tmp.items():
            wert[(arm, n)][lr] = float(np.mean(v))
        for s in list(offen):
            g = sorted(gitter[s])
            best = max(g, key=lambda v: wert[s][v])
            if best == g[0]:
                gitter[s] = [g[0] / 4] + g
            elif best == g[-1]:
                gitter[s] = g + [g[-1] * 4]
            else:
                info[s] = {"lr": best, "gitter": {str(k): v for k, v in wert[s].items()}, "am_rand": False}
                offen.discard(s)
        if not offen:
            break
    for s in offen:
        info[s] = {"lr": max(wert[s], key=wert[s].get),
                   "gitter": {str(k): v for k, v in wert[s].items()}, "am_rand": True}
    return info


def _ist_vorzeichen_permutation(m) -> bool:
    with torch.no_grad():
        U = m.unitaer()
    return bool(int((U.abs() > 1e-6).sum()) == len(U))


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=MESS_SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, **kw):
    if schnell:
        seeds, tuning_seeds, schritte = tuple(seeds)[:2], tuple(tuning_seeds)[:1], 60
    seeds, tuning_seeds = tuple(seeds), tuple(tuning_seeds)
    t0 = time.time()
    # 1) Raten fuer die Arme ohne Abhaengigkeit, je n
    lr_info = _lr_suche([(arm, n) for arm in ARME_FEST for n in N_WERTE], tuning_seeds, schritte)
    lr = {k: v["lr"] for k, v in lr_info.items()}
    print(f"  Raten voll/ohne/zufall nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{a}@{n}={lr[(a, n)]:g}" for a, n in sorted(lr)), flush=True)
    # 2) Voll-Modelle der Tuning-Seeds je n (geteilt fuer die Rundungs-Suche)
    basis = parallel(_arbeit, [("voll", n, lr[("voll", n)], s, "tuning", schritte, None)
                               for s in tuning_seeds for n in N_WERTE])
    winkel_tuning = {(z["seed"], z["n"]): z["winkel"] for z in basis}
    # 3) Raten der Rundungsarme, je (k, n) eigen
    lr_rund = _lr_suche([(arm, n) for arm in ARME_RUND for n in N_WERTE], tuning_seeds, schritte,
                        winkel_von=winkel_tuning)
    lr_info.update(lr_rund)
    lr.update({k: v["lr"] for k, v in lr_rund.items()})
    print(f"  alle Raten nach {time.time() - t0:.0f} s", flush=True)
    # 4) Messlauf: erst voll/ohne/zufall, dann die Rundungsarme aus den voll-Winkeln
    stufe1 = parallel(_arbeit, [(arm, n, lr[(arm, n)], s, "mess", schritte, None)
                                for s in seeds for n in N_WERTE for arm in ARME_FEST])
    winkel = {(z["seed"], z["n"]): z["winkel"] for z in stufe1 if z["arm"] == "voll"}
    stufe2 = parallel(_arbeit, [(arm, n, lr[(arm, n)], s, "mess", schritte, winkel[(s, n)])
                                for s in seeds for n in N_WERTE for arm in ARME_RUND])
    zeilen = {s: {"seed": s, "acc": {}, "acc_train": {}, "stopp": {}, "val_verlust": {}, "winkel_voll": {}}
              for s in seeds}
    for z in stufe1 + stufe2:
        r, n = zeilen[z["seed"]], str(z["n"])
        for feld in ("acc", "acc_train", "stopp", "val_verlust"):
            r[feld].setdefault(n, {})[z["arm"]] = z[feld]
        if z["arm"] == "voll":
            r["winkel_voll"][n] = z["winkel"]
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: " + "  ".join(
            f"n={n}: voll {z['acc'][str(n)]['voll']:.4f} k0 {z['acc'][str(n)]['clifford_k0']:.4f} "
            f"ohne {z['acc'][str(n)]['ohne_schaltung']:.4f}" for n in N_WERTE), flush=True)
    perm = {s: _ist_vorzeichen_permutation(_baue_arm("zufall_clifford", s)) for s in seeds}
    ergebnis = {
        "abteilung": "clifford", "modul": "datenmenge", "karte": "I-024", "version": 1,
        "datensatz": DATENSATZ, "qubits": N, "lagen": LAGEN, "n_werte": list(N_WERTE),
        "anteil_val": ANTEIL_VAL, "takt": TAKT, "n_test": N_TEST, "hypothesen": HYPOTHESEN,
        "lr": {f"{a}@{n}": v for (a, n), v in lr_info.items()}, "schritte": schritte,
        "seeds": list(seeds), "tuning_seeds": list(tuning_seeds), "zeilen": zeilen,
        "zufall_ist_permutation": {str(s): v for s, v in perm.items()},
        "multiplikationen": {"voll": multiplikationen(N),
                             **{f"clifford_k{k}": multiplikationen(N, magie=k) for k in MAGIE_STUFEN}},
        "t_gatter_schaetzung": {"voll": T_JE_WINKEL * LAGEN * N,
                                **{f"clifford_k{k}": T_JE_WINKEL * k for k in MAGIE_STUFEN}},
        "sekunden": time.time() - t0,
    }
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Statistik
def wilcoxon_einseitig(diff, richtung="groesser") -> float:
    """Exakter einseitiger Wilcoxon-Vorzeichen-Rang-Test, gleiche Rangbildung wie
    labor.wilcoxon_exakt (Nullen weg, Raenge 1..n). P(W+ >= w) bzw. P(W+ <= w),
    exakte Verteilung per dynamischer Programmierung (auch fuer 20+ Seeds)."""
    d = np.asarray([v for v in diff if v != 0.0], dtype=float)
    n = len(d)
    if n == 0:
        return 1.0
    raenge = (np.argsort(np.argsort(np.abs(d))) + 1).astype(int)
    w = int(raenge[d > 0].sum())
    zahl = np.zeros(n * (n + 1) // 2 + 1)
    zahl[0] = 1.0
    for r in range(1, n + 1):
        zahl[r:] = zahl[r:] + zahl[:-r].copy()
    vert = zahl / zahl.sum()
    return float(vert[w:].sum() if richtung == "groesser" else vert[:w + 1].sum())


def holm(p: dict) -> dict:
    """Holm-adjustierte p-Werte (monoton, gedeckelt bei 1)."""
    namen = sorted(p, key=p.get)
    aus, laufend = {}, 0.0
    for i, h in enumerate(namen):
        laufend = max(laufend, min(1.0, (len(namen) - i) * p[h]))
        aus[h] = laufend
    return aus


def _gepaart(d) -> dict:
    """Mittel, SD, exakter Wilcoxon (zweiseitig + beide einseitig), zweiseitiges
    90-%-KI (= zwei einseitige 95-%-Tests, TOST)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    aus = {"mittel": m, "p": wilcoxon_exakt(d), "p_groesser": wilcoxon_einseitig(d, "groesser"),
           "p_kleiner": wilcoxon_einseitig(d, "kleiner")}
    if len(d) < 2:
        return {**aus, "sd": 0.0, "ki_unten": -math.inf, "ki_oben": math.inf}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return {**aus, "sd": float(d.std(ddof=1)), "ki_unten": m - q * se, "ki_oben": m + q * se}


def _innen(g, marge) -> bool:
    return bool(g["ki_unten"] > -marge and g["ki_oben"] < marge)


# ------------------------------------------------------------------ Auswertung
def auswertung(zeilen) -> dict:
    n_werte = sorted(int(n) for n in zeilen[0]["acc"])
    lo, hi = n_werte[0], n_werte[-1]
    A = lambda arm, n: np.array([z["acc"][str(n)][arm] for z in zeilen], dtype=float)
    aus = {"n": len(zeilen), "n_werte": n_werte, "je_n": {}}
    delta, g2 = {}, {}
    for n in n_werte:
        e = {arm: zusammenfassung(A(arm, n)) for arm in ARME}
        gew = A("voll", n) - A("ohne_schaltung", n)
        e["gewinn"], e["gewinn_p"] = float(gew.mean()), wilcoxon_exakt(gew)
        e["gewinn_ok"] = bool(e["gewinn"] > 0 and e["gewinn_p"] < ALPHA)
        delta[n] = A("voll", n) - A("clifford_k0", n)
        e["delta"] = _gepaart(delta[n])
        gw = e["gewinn"]
        e["G"] = {str(k): (float((A(f"clifford_k{k}", n) - A("ohne_schaltung", n)).mean()) / gw
                           if gw > 0 else None) for k in MAGIE_STUFEN}
        for stufe in (50, 90):
            e[f"M{stufe}"] = next((k for k in MAGIE_STUFEN
                                   if e["G"][str(k)] is not None and e["G"][str(k)] >= stufe / 100), None)
        g2[n] = (A(f"clifford_k{K_ENTSCHEID}", n) - A("ohne_schaltung", n)) / gw if gw > 0 else None
        e["k0_minus_voll"] = _gepaart(-delta[n])
        e["k0_minus_zufall"] = _gepaart(A("clifford_k0", n) - A("zufall_clifford", n))
        if all("acc_train" in z for z in zeilen):
            T = lambda arm: np.array([z["acc_train"][str(n)][arm] for z in zeilen], dtype=float)
            e["luecke"] = {arm: float((T(arm) - A(arm, n)).mean()) for arm in ("voll", "clifford_k0", "ohne_schaltung")}
        if all("stopp" in z for z in zeilen):
            e["stopp_median"] = {arm: float(np.median([z["stopp"][str(n)][arm] for z in zeilen])) for arm in ARME}
        aus["je_n"][str(n)] = e
    ehi = aus["je_n"][str(hi)]
    # D-0
    aus["D0"] = {"gewinn": ehi["gewinn"], "p": ehi["gewinn_p"], "ok": ehi["gewinn_ok"]}
    # D-1
    aus["D1_vorbedingung"] = bool(aus["D0"]["ok"] and ehi["delta"]["mittel"] > 0 and ehi["delta"]["p"] < ALPHA)
    aus["D1"] = _gepaart(delta[hi] - delta[lo])
    aus["D1_tost"] = {str(n): _gepaart(delta[hi] - delta[n]) for n in n_werte if n != hi}
    aus["D1_tost_gleich"] = all(_innen(g, MARGE_D1) for g in aus["D1_tost"].values())
    # D-2
    aus["D2_vorbedingung"] = bool(aus["je_n"][str(lo)]["gewinn_ok"] and ehi["gewinn_ok"])
    if g2[lo] is not None and g2[hi] is not None:
        aus["D2"] = _gepaart(g2[lo] - g2[hi])
    else:
        aus["D2"] = None
    aus["D2_tost"] = {str(n): (_gepaart(g2[n] - g2[hi]) if g2[n] is not None and g2[hi] is not None else None)
                      for n in n_werte if n != hi}
    aus["D2_tost_gleich"] = all(g is not None and _innen(g, MARGE_D2) for g in aus["D2_tost"].values())
    # Holm ueber die angemeldete Familie (nur Tests mit erfuellter Vorbedingung)
    fam = {}
    if aus["D1_vorbedingung"]:
        fam["D-1"] = aus["D1"]["p_groesser"]
    if aus["D2_vorbedingung"] and aus["D2"] is not None:
        fam["D-2"] = aus["D2"]["p_groesser"]
    aus["holm"] = holm(fam) if fam else {}
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (Kopfkommentar). Praefixe: 'H1:', 'H0:',
    'H0 (Gegenrichtung):', 'HU:', 'HU (Vorbedingung):'."""
    lo, hi = a["n_werte"][0], a["n_werte"][-1]
    pp = lambda v: f"{100 * v:+.2f} pp"
    d0 = a["D0"]
    u = {"D-0": (f"H1: voll schlaegt Identitaet bei n={hi} (Gewinn {pp(d0['gewinn'])}, p = {d0['p']:.4f})"
                 if d0["ok"] else
                 f"H0: Gewinn ueber Identitaet bei n={hi} nicht nachweisbar ({pp(d0['gewinn'])}, p = {d0['p']:.4f})")}
    # D-1
    d1 = a["D1"]
    dhi = a["je_n"][str(hi)]["delta"]
    if not a["D1_vorbedingung"]:
        u["D-1"] = (f"HU (Vorbedingung): D-0 oder Delta({hi}) = {pp(dhi['mittel'])} (p = {dhi['p']:.4f}) "
                    f"nicht nachweisbar > 0")
    elif a["holm"]["D-1"] < ALPHA and d1["mittel"] >= MARGE_D1:
        u["D-1"] = (f"H1: Abstand voll - k0 schrumpft mit weniger Daten um {pp(d1['mittel'])} "
                    f"(Delta({lo}) = {pp(a['je_n'][str(lo)]['delta']['mittel'])}, Delta({hi}) = {pp(dhi['mittel'])}, "
                    f"Holm-p = {a['holm']['D-1']:.4f})")
    elif d1["p_kleiner"] < ALPHA and d1["mittel"] <= -MARGE_D1:
        u["D-1"] = (f"H0 (Gegenrichtung): mit weniger Daten wird MEHR Magie gebraucht "
                    f"(Delta({lo}) - Delta({hi}) = {pp(-d1['mittel'])}, p = {d1['p_kleiner']:.4f})")
    elif a["D1_tost_gleich"]:
        u["D-1"] = (f"H0: Abstand unabhaengig von der Datenmenge (TOST +-0,5 pp; Differenz {lo}: "
                    f"{pp(d1['mittel'])}, KI [{pp(d1['ki_unten'])}; {pp(d1['ki_oben'])}])")
    else:
        u["D-1"] = (f"HU: unentschieden (Delta({hi}) - Delta({lo}) = {pp(d1['mittel'])}, "
                    f"p einseitig = {d1['p_groesser']:.4f}, KI [{pp(d1['ki_unten'])}; {pp(d1['ki_oben'])}])")
    # D-2
    d2 = a["D2"]
    if not a["D2_vorbedingung"] or d2 is None:
        u["D-2"] = f"HU (Vorbedingung): Gewinn bei n={lo} oder n={hi} nicht nachweisbar > 0 -- G undefiniert"
    elif a["holm"]["D-2"] < ALPHA and d2["mittel"] >= MARGE_D2:
        u["D-2"] = (f"H1: mit wenig Daten tragen 2 Magiewinkel mehr: G_{lo}(2) - G_{hi}(2) = "
                    f"{d2['mittel']:+.3f} (Holm-p = {a['holm']['D-2']:.4f})")
    elif d2["p_kleiner"] < ALPHA and d2["mittel"] <= -MARGE_D2:
        u["D-2"] = (f"H0 (Gegenrichtung): mit wenig Daten tragen 2 Magiewinkel WENIGER "
                    f"({d2['mittel']:+.3f}, p = {d2['p_kleiner']:.4f})")
    elif a["D2_tost_gleich"]:
        u["D-2"] = (f"H0: G(2) unabhaengig von der Datenmenge (TOST +-{MARGE_D2}; Differenz "
                    f"{d2['mittel']:+.3f}, KI [{d2['ki_unten']:+.3f}; {d2['ki_oben']:+.3f}])")
    else:
        u["D-2"] = (f"HU: unentschieden (G_{lo}(2) - G_{hi}(2) = {d2['mittel']:+.3f}, "
                    f"p einseitig = {d2['p_groesser']:.4f}, KI [{d2['ki_unten']:+.3f}; {d2['ki_oben']:+.3f}])")
    # D-3
    teile = []
    for n in a["n_werte"]:
        e = a["je_n"][str(n)]
        g = e["G"][str(K_ENTSCHEID)]
        teile.append(f"n={n}: Gewinn {pp(e['gewinn'])}, Delta {pp(e['delta']['mittel'])}, "
                     f"G(2) {'-' if g is None else f'{g:.2f}'}, M50 {e['M50']}, M90 {e['M90']}, "
                     f"k0-voll {pp(e['k0_minus_voll']['mittel'])} (p = {e['k0_minus_voll']['p']:.3f})")
    u["D-3"] = "; ".join(teile)
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_struktur() -> bool:
    """Tun die Bausteine, was ihr Name sagt? Billig: 300 Bilder, wenige Schritte."""
    ok = True
    # 1 Thread: der Vortest laeuft oft neben grossen Laeufen; mehrere Threads je
    # Prozess ueberbuchen die Kerne (Lehre „BLAS-Threads 1“, 24.09.).
    torch.set_num_threads(1)
    # (a) geschachtelte Mengen und identische Testmenge
    x, y, xte, yte = _daten(MESS_SEEDS[0])
    x5, y5, xte5, yte5 = bilder(DATENSATZ, MESS_SEEDS[0], n_train=500, n_test=N_TEST, seite=16)
    if not (torch.equal(x[:500], x5) and torch.equal(y[:500], y5) and torch.equal(xte, xte5)):
        ok = False
        print("  VORTEST FEHLER: Trainingsmengen nicht geschachtelt oder Testmenge haengt von n ab")
    for n in N_WERTE:
        xtr, _, xva, _, xev, _ = _teile(MESS_SEEDS[0], n, "tuning")
        if len(xtr) + len(xva) != n or len(xev) != N_TUNING_EVAL:
            ok = False
            print("  VORTEST FEHLER: falsche Aufteilung bei n =", n)
    # (b) Identitaet = Permutation; Zufallskontrolle nicht durchweg entartet
    if not _ist_vorzeichen_permutation(_baue_arm("ohne_schaltung", 0)):
        ok = False
        print("  VORTEST FEHLER: Identitaets-Kontrolle ist keine Permutation")
    entartet = [s for s in MESS_SEEDS if _ist_vorzeichen_permutation(_baue_arm("zufall_clifford", s))]
    if len(entartet) > len(MESS_SEEDS) // 2:
        ok = False
        print("  VORTEST FEHLER: zufaellige Clifford-Kontrolle meist entartet", entartet)
    # (c) Rundungsarme: k freie Winkel beim trainierten Wert, Rest auf dem Gitter, Kopf frisch
    torch.manual_seed(3)
    voll = QireZelle(n=N)
    th = voll.theta.detach().flatten().clone()
    for k in MAGIE_STUFEN:
        m = _baue_arm(f"clifford_k{k}", 5, th.tolist())
        w, frei = m.winkel().detach().flatten(), m.frei.flatten().bool()
        if int(frei.sum()) != k or not torch.equal(w[frei], th[frei]):
            ok = False
            print("  VORTEST FEHLER: falsche freie Winkel, k =", k)
        if not torch.allclose(w[~frei] / GITTER, torch.round(w[~frei] / GITTER)):
            ok = False
            print("  VORTEST FEHLER: gerundete Winkel nicht auf dem Gitter, k =", k)
        if torch.equal(m.kopf.weight, voll.kopf.weight):
            ok = False
            print("  VORTEST FEHLER: Kopf geerbt statt frisch, k =", k)
    if not torch.equal(runde_clifford(th).reshape(LAGEN, N), _baue_arm("clifford_k0", 5, th.tolist()).winkel().detach()):
        ok = False
        print("  VORTEST FEHLER: k0 ist nicht die volle Rundung")
    # (d) Abkuerzung fuer feste Winkel == voller Weg; (e) fruehes Stoppen liefert das beste Modell
    xtr, ytr, xva, yva = x[60:300], y[60:300], x[:60], y[:60]
    for arm in ("ohne_schaltung", "clifford_k0"):
        m1, m2 = _baue_arm(arm, 1, th.tolist()), _baue_arm(arm, 1, th.tolist())
        i1 = trainiere_mit_stopp(m1, xtr, ytr, xva, yva, 40, 0.08, 1, takt=5, abkuerzung=True)
        i2 = trainiere_mit_stopp(m2, xtr, ytr, xva, yva, 40, 0.08, 1, takt=5, abkuerzung=False)
        if not (torch.allclose(m1.kopf.weight, m2.kopf.weight, atol=1e-5) and i1["stopp"] == i2["stopp"]
                and torch.equal(m1.theta, m2.theta)):
            ok = False
            print("  VORTEST FEHLER: Merkmals-Abkuerzung weicht vom vollen Weg ab", arm, i1, i2)
    m = _baue_arm("voll", 1)
    with torch.no_grad():
        v0 = F.cross_entropy(m(xva), yva).item()
    info = trainiere_mit_stopp(m, xtr, ytr, xva, yva, 400, 0.32, 1, takt=10)
    with torch.no_grad():
        v = F.cross_entropy(m(xva), yva).item()
    if not (abs(v - info["val_verlust"]) < 1e-5 and v <= v0 + 1e-9):
        ok = False
        print("  VORTEST FEHLER: fruehes Stoppen laedt nicht das beste Modell", v, info, v0)
    if not (0 < info["stopp"] < 400):
        print(f"  Vortest-Hinweis: Stopp bei {info['stopp']} (240 Bilder, 400 Schritte) -- kein Fehler")
    # (f) Statistik: einseitiger Wilcoxon, Holm
    if abs(wilcoxon_einseitig(np.arange(1, 11) / 100) - 1 / 1024) > 1e-12:
        ok = False
        print("  VORTEST FEHLER: einseitiger Wilcoxon, alle positiv")
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = rng.normal(0.3, 1, 10)
        pg, pk = wilcoxon_einseitig(d, "groesser"), wilcoxon_einseitig(d, "kleiner")
        if abs(min(1.0, 2 * min(pg, pk)) - wilcoxon_exakt(d)) > 1e-9 or pg + pk < 1 - 1e-12:
            ok = False
            print("  VORTEST FEHLER: einseitiger Wilcoxon passt nicht zum zweiseitigen", pg, pk, wilcoxon_exakt(d))
            break
    h = holm({"a": 0.03, "b": 0.04, "c": 0.001})
    if not (abs(h["c"] - 0.003) < 1e-12 and abs(h["a"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12):
        ok = False
        print("  VORTEST FEHLER: Holm", h)
    return ok


def _synth(rng, delta, gewinn, g2, sd=0.002, sd_delta=0.0, sd_k=None, n_seeds=10):
    """Synthetische Zeilen: je n Gewinn (voll - ohne), Delta (voll - k0), G(2)."""
    sd_k = sd if sd_k is None else sd_k
    out = []
    for s in range(n_seeds):
        z = {"seed": s, "acc": {}, "acc_train": {}, "stopp": {}}
        for n in N_WERTE:
            ohne = 0.85 + 0.03 * math.log10(n / 500) + rng.normal(0, sd)
            voll = ohne + gewinn[n] + rng.normal(0, sd)
            G = {1: g2[n] - 0.05, 2: g2[n], 4: g2[n] + 0.4 * (1 - g2[n]), 8: 0.95}
            acc = {"voll": voll, "ohne_schaltung": ohne, "zufall_clifford": ohne + rng.normal(0, sd),
                   "clifford_k0": voll - delta[n] + rng.normal(0, sd) + rng.normal(0, sd_delta)}
            for k in MAGIE_STUFEN[1:]:
                acc[f"clifford_k{k}"] = ohne + gewinn[n] * G[k] + rng.normal(0, sd_k)
            z["acc"][str(n)] = acc
            z["acc_train"][str(n)] = {arm: min(1.0, v + 0.05) for arm, v in acc.items()}
            z["stopp"][str(n)] = {arm: 1000 for arm in acc}
        out.append(z)
    return out


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Bausteine pruefen, dann jeden Urteilszweig auf
    synthetischen Zeilen erzwingen. Null- und Grenzfaelle kippen mit Rate
    ~alpha -- dort je Fall und Hypothese eine Abweichung erlaubt."""
    ok = _vortest_struktur()
    rng = np.random.default_rng(24)
    gl = lambda v: {n: v for n in N_WERTE}
    faelle = [
        # 1) Abstand schrumpft stark, G(2) steigt bei wenig Daten
        (dict(delta={500: 0.005, 2000: 0.02, 10000: 0.035}, gewinn=gl(0.04), g2={500: 0.85, 2000: 0.65, 10000: 0.5}),
         {"D-0": "H1:", "D-1": "H1:", "D-2": "H1:"}, 0),
        # 2) alles gleich, kleine Streuung -> TOST greift (Abbruch der Karte)
        (dict(delta=gl(0.03), gewinn=gl(0.04), g2=gl(0.5), sd=0.001),
         {"D-0": "H1:", "D-1": "H0:", "D-2": "H0:"}, 1),
        # 3) signifikant, aber kleiner als die Grenze -> praktisch gleich (H0)
        (dict(delta={500: 0.028, 2000: 0.029, 10000: 0.03}, gewinn=gl(0.04), g2=gl(0.5), sd=0.0003),
         {"D-1": "H0:"}, 0),
        # 4) Gegenrichtung: wenig Daten brauchen mehr Magie
        (dict(delta={500: 0.06, 2000: 0.045, 10000: 0.03}, gewinn=gl(0.04), g2={500: 0.2, 2000: 0.35, 10000: 0.5}),
         {"D-1": "H0 (Gegenrichtung):", "D-2": "H0 (Gegenrichtung):"}, 0),
        # 5) grosse Streuung, kein wahrer Unterschied -> HU (TOST kann nicht greifen).
        #    Simuliert (200 Ziehungen): kippt mit ~5-9 % (H1 oder Gegenrichtung, je ~alpha)
        #    -> Toleranz 2 (Fehlalarm des Vortests < 1 %).
        (dict(delta=gl(0.03), gewinn=gl(0.04), g2=gl(0.5), sd_delta=0.012, sd_k=0.02),
         {"D-1": "HU:", "D-2": "HU:"}, 2),
        # 6) Positivkontrolle scheitert -> beide HU (Vorbedingung)
        (dict(delta=gl(0.03), gewinn={500: 0.04, 2000: 0.04, 10000: -0.005}, g2=gl(0.5)),
         {"D-0": "H0:", "D-1": "HU (Vorbedingung):", "D-2": "HU (Vorbedingung):"}, 0),
        # 7) keine Magie noetig bei vollen Daten (Delta(10000) = 0) -> D-1 HU (Vorbedingung)
        (dict(delta=gl(0.0), gewinn=gl(0.04), g2=gl(0.5), sd=0.001),
         {"D-0": "H1:", "D-1": "HU (Vorbedingung):", "D-2": "H0:"}, 1),
        # 8) kein Gewinn bei n = 500 -> D-2 HU (Vorbedingung), D-1 entscheidet weiter
        (dict(delta={500: 0.0, 2000: 0.02, 10000: 0.035}, gewinn={500: 0.0, 2000: 0.03, 10000: 0.04}, g2=gl(0.5)),
         {"D-1": "H1:", "D-2": "HU (Vorbedingung):"}, 1),
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
                print(f"  VORTEST FEHLER: {h} erwartet {erwartet[h]} ({f} von {ziehungen} falsch)")
    # Holm wird im Urteil tatsaechlich benutzt: roher p-Wert klein, Holm-p gross -> kein H1
    a = auswertung(_synth(rng, **faelle[0][0]))
    a["holm"] = {**a["holm"], "D-1": 0.06, "D-2": 0.06}
    u = urteil(a)
    if u["D-1"].startswith("H1") or u["D-2"].startswith("H1"):
        ok = False
        print("  VORTEST FEHLER: Urteil ignoriert Holm", u["D-1"], u["D-2"])
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-024" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
