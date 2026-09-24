"""Abteilung Clifford -- Magie-Skalierung (Agenda C-skala, Quantenstufe Q3).

FRAGE: Waechst der Magiebedarf M50 (kleinste Zahl k freier Nicht-Clifford-
Winkel, mit der die Schaltung >= 50 % ihres Nutzens behaelt) LANGSAMER als die
Winkelzahl n*L? Breiten n = 6, 8, 10 Qubits, L = 4 Lagen, MNIST amplituden-
kodiert auf 2^(n/2) x 2^(n/2) Pixeln (8x8, 16x16, 32x32). Wenn der relative
Magiebedarf M50/(n*L) mit n faellt, wird die Schicht mit wachsender Breite
fast multiplikationsfrei (fast T-frei).

WIEDERVERWENDET aus clifford.py (per import): QIRE-Zelle (Parameter, Puffer,
CNOT-Ring, fixiere), Rundung runde_clifford, Rangfolge _magie_reihenfolge
(groesster Abstand zum Gitter zuerst -- wie C-mnist, damit M50 = 2 dort
vergleichbar bleibt), multiplikationen(), Aufbau von _baue/_gerundet (v2:
frischer Kopf). NEU ist nur die SIMULATION: SkalaZelle rechnet dieselbe
Schaltung auf den Zustaenden (jede Lage als Kronecker-Produkt zweier Haelften
A (x) B, also A * Psi * B^T) statt die 2^n x 2^n-Matrix zu bauen. Bei n = 10
ist das ~5x schneller; der Vortest prueft, dass Merkmale, Startzustaende und
Rundungsarme mit clifford.py uebereinstimmen (Merkmale bis 1e-5, state_dict
bitgleich). Arme ohne freien Winkel (Identitaet, k = 0) trainieren nur den
Kopf auf einmal berechneten Merkmalen -- mathematisch identisch (Adam bewegt
Winkel mit Gradient 0 nicht), ebenfalls im Vortest geprueft.

ARME je Konfiguration (Konfiguration = Aufgabe + Breite n):
  voll            alle n*L Winkel frei trainiert (Start wie clifford._baue)
  ohne_schaltung  Identitaet: Ry(0) + CNOT-Ring = reine Permutation (Vortest)
  nachtrainiert   = Rundungsarm mit k = n*L: alle Winkel frei, Start beim
                  trainierten voll, frischer Kopf, zweite Trainingsphase mit
                  derselben Rate/demselben Budget wie die Magie-Arme k > 0.
  k<k>            Rundung aller Winkel auf k*pi/2, die k Winkel mit dem
                  groessten Abstand zum Gitter bleiben frei (Start beim
                  trainierten Wert), Kopf FRISCH (Lehre „geerbter Kopf“).
  Stufen k je n: {0, 1, 2} und die Anteile 1/8, 1/4, 1/2 von n*L:
     n = 6 (24 Winkel): 0 1 2 3 6 12 | n = 8 (32): 0 1 2 4 8 16 | n = 10 (40): 0 1 2 5 10 20.
  Warum keine Anteile 1/32, 1/16: n*L = 24, 32, 40 haben den ggT 8; nur
  Vielfache von 1/8 ergeben bei ALLEN drei Breiten ganze k. Kleinere Anteile
  deckt das absolute k = 1, 2 beschreibend ab (keine Interpolation im Test).

AUFGABEN:
  echt  MNIST mit Bildseite 2^(n/2) (labor.gemeinsam.bilder, seite=...).
  fest  (zusaetzlicher, sauberer Arm) Bild erst auf 8x8 verkleinert, dann
        per naechstem Nachbarn auf 2^(n/2) hochskaliert. Quantenmechanisch
        ist das |x_8x8> (x) |+>^(n-6): die Zusatzqubits (untere Zeilen- und
        Spaltenbits) tragen KEINE Information, die Aufgabe ist exakt die von
        n = 6, nur die Schaltung ist breiter. fest6 = echt6.

KONFUNDIERUNG (offen benannt): Im Arm „echt“ waechst mit n nicht nur die
Breite, sondern auch die Aufgabe: mehr Bildaufloesung (8x8 -> 16x16 -> 32x32;
32x32 ist aus 28x28 HOCHinterpoliert, enthaelt also keine Information ueber
das Original hinaus), ein 2^n x 10 grosser Kopf und eine andere Identitaets-
Basislinie. Ein Unterschied in G kann deshalb von der Aufgabe kommen, nicht
von der Breite. Der Arm „fest“ haelt die Information fest (8x8) und aendert
nur die Breite -- er hat seine eigene Verzerrung (4 informationsfreie Qubits
in |+>, Born-Auslese 1024-dim fuer 64-dim Information). Beide zusammen
klammern die Frage ein; nur wenn S-1 und S-3 gleich ausgehen, ist die
Aussage „Breite“ belastbar. Gleiches Budget fuer alle n (n_train, Schritte);
die Rate ist je Breite, Aufgabe und Arm getunt, damit n = 10 nicht nur
schlechter konvergiert. Das Budget ist kleiner als in C-mnist (2.500 statt
6.000 Bilder, 500 statt 1.500 Schritte), sonst waere n = 10 nicht in ~1,5 h
zu schaffen (ein volles Training bei n = 10 kostet ~0,2 s je Schritt und
Kern). Die Zahlen sind daher NICHT direkt mit C-mnist (M50 = 2) vergleichbar;
echt8 ist die Bruecke.

AENDERUNG DIREKTOR 24.09. VOR JEDER ZAHL: faire Referenz. Die Rundungsarme
bekommen eine ZWEITE Trainingsphase (Start bei voll, frischer Kopf, volles
Budget), voll nicht; auf Fashion schlug k8 dadurch voll um +0,53 pp (p = 0,01).
G gegen voll waere also durch das Nachtraining aufgeblaeht. Referenz im Nenner
von G und fuer M50/M90 in S-0/S-1/S-3 ist daher „nachtrainiert“ (gleiche
zweite Phase, nur ohne Rundung); die Basis bleibt die Identitaet. Die alten
voll-basierten G_voll/M50_voll und d_nach = acc(nachtrainiert) - acc(voll)
stehen beschreibend unter S-2. S-0 prueft NUR nachtrainiert gegen Identitaet:
die Vorbedingung muss genau die Groesse absichern, die im Nenner steht; ein
zusaetzlicher Test von voll schuetzt G nicht und wuerde nur einen Weg zu
„HU (Vorbedingung)“ ohne Bezug zur Messgroesse oeffnen. voll gegen
Identitaet wird im S-0-Text mitberichtet.

VORREGISTRIERT (vor dem ersten Lauf, Regel 7):
  Gewinnanteil wie clifford.py v3, aber mit fairer Referenz:
  G_n(k) = Mittel(acc_k - acc_ohne) / Gewinn_n, Gewinn_n = Mittel ueber Seeds
  von (acc_nachtrainiert - acc_ohne) bei dieser Breite.
  Fuer den gepaarten Test je Seed ein Pseudowert (Delta-Methode, _je_seed_G):
  G_n,s(k) = (N_s - G_n(k) * (D_s - Gewinn_n)) / Gewinn_n, N_s = acc_k - acc_ohne,
  D_s = acc_nachtrainiert - acc_ohne. Mittel = G_n(k); die Streuung enthaelt auch den
  Schaetzfehler des Nenners (mit festem Nenner war der Test zu liberal).
  (Die Basis ist die Identitaet, NICHT die gerundete Schaltung k = 0: deren
  Genauigkeit streut je Seed stark -- in C-mnist sd 1,5 pp bei 3,2 pp Gewinn
  gegenueber k = 0 -- und wuerde den gepaarten Test verrauschen. Die auf k = 0
  bezogene Kurve G_mag wird unter S-2 beschreibend mitberichtet.)
  S-0 (je Konfiguration) Positivkontrolle: nachtrainiert schlaegt Identitaet (Gewinn
      > 0, exakter Wilcoxon zweiseitig p < 0,05). Sonst ist G/M50 dort
      bedeutungslos -> S-1/S-3 mit dieser Breite „HU (Vorbedingung)“.
  S-1 (Haupt, Arm echt): d_s = Mittel ueber f in {1/8, 1/4} von
      G_10,s(f*40) - G_6,s(f*24), gepaart ueber den Seed (gleiche Seeds =
      gleiche Bilder bei allen Breiten).
        Vorbedingung S-0 bei echt6 und echt10 verletzt -> HU (Vorbedingung)
        mittel(d) > 0, einseitiger exakter Wilcoxon p < alpha_Holm -> H1
        mittel(d) < 0, einseitiger exakter Wilcoxon p < alpha_Holm -> H0 (niedriger)
        TOST: einseitige 95-%-KI von d ganz in (-0,08; +0,08)     -> H0 (gleich)
        sonst                                                     -> HU
      GRENZE RELATIV ZUM EFFEKT: Auf C-mnist steigt G um ~0,2 je Verdopplung
      von k (G(4) = 0,68 -> G(8) = 0,89). Waere M50 von n unabhaengig
      (staerkster H1-Fall), laege G_10 beim gleichen Anteil um ~0,2 *
      log2(40/24) = 0,15 hoeher; waechst M50 proportional zu n*L, ist d = 0.
      Die Aequivalenzgrenze ist die Haelfte des staerksten Effekts: 0,08.
      AUFLOESUNG: bei sd(G je Seed) ~ 0,2 ist das KI von d etwa +-0,15 breit;
      „H0 (gleich)“ ist damit kaum erreichbar, HU der wahrscheinliche Ausgang
      bei kleinem Effekt. Die Richtung wird erst an den Daten abgelesen,
      deshalb geht der einseitige p-Wert verdoppelt (= zweiseitig) in Holm
      ein (Direktor 24.09., vor dem Lauf) -- Gesamtniveau 0,05.
  S-3 (Arm fest, sauberere Variante): wie S-1 mit fest10 gegen echt6 (=fest6).
      Holm ueber die Familie {S-1, S-3} (nur Mitglieder mit erfuellter
      Vorbedingung), jeweils auf den einseitigen p der beobachteten Richtung.
  S-2 beschreibend: M50, M90 absolut je Konfiguration (kleinste gemessene
      Stufe und linear zwischen Stufen interpoliert), m50 = M50/(n*L), dasselbe
      bezogen auf voll (G_voll, M50_voll), d_nach = nachtrainiert - voll, Fit
      M50 ~ a * n^b (log-log, echt: 3 Punkte, fest: 2 Punkte), G und G_mag je
      Stufe, d je Anteil (0, 1/8, 1/4, 1/2) und fuer k = 1, 2 absolut.

PROTOKOLL: Mess-Seeds 120-129 (frisch, 10 Seeds -> kleinster einseitiger
p = 1/1024), Tuning-Seeds 920/921 (Validierung = erstes Viertel des
Trainingsteils). Raten je Konfiguration eigen fuer voll, ohne_schaltung, k0
und fuer die Magie-Arme k > 0 (getunt bei k = n*L/8, dem kleineren Haupt-
Anteil; eine gemeinsame Rate fuer alle k > 0 wie in clifford.py v2 --
Abweichung von Regel 1a aus Budgetgruenden, offen benannt). Gitter
(0,02; 0,08; 0,32), in beide Richtungen verlaengert (1b), volles Budget (1c).
BLAS-Threads 1 je Prozess; Aufgaben mit chunksize 1, laengste zuerst.
LAUFZEIT (geschaetzt, 4 Prozesse): ~82 volle Trainings je Konfiguration
(inkl. nachtrainiert); n = 10: ~100 s je Training -> echt10 + fest10
~ 4,6 Kern-h, echt8 ~ 0,6, echt6 ~ 0,2 -> gesamt ~ 80-100 min.
"""
from __future__ import annotations

import functools
import math
import os
import time
from itertools import product

import numpy as np
import torch

from abteilungen.clifford import (GITTER, LAGEN, QireZelle, _magie_reihenfolge, multiplikationen,
                                  runde_clifford)
from labor.gemeinsam import (_init_arbeiter, bilder, genauigkeit, parallel, speichere, trainiere,
                             wilcoxon_exakt, zusammenfassung)

BREITEN = (6, 8, 10)
FEST_BREITEN = (10,)          # Arm „fest“ (8x8-Information in n Qubits); fest6 = echt6
SEEDS = tuple(range(120, 130))
TUNING_SEEDS = (920, 921)
SCHRITTE = 500
N_TRAIN = 2_500
N_TEST = 5_000
LR_GITTER = (0.02, 0.08, 0.32)
ANTEILE_HAUPT = (1 / 8, 1 / 4)
ANTEILE_ALLE = (1 / 8, 1 / 4, 1 / 2)
K_ABSOLUT = (1, 2)
ANTEIL_TUNING = 1 / 8         # Magie-Arme k > 0 werden bei k = n*L/8 getunt
ALPHA = 0.05
DELTA = 0.08                  # Aequivalenzgrenze in G (Herleitung im Kopf)
PAAR_HAUPT = ("echt6", "echt10")
PAAR_FEST = ("echt6", "fest10")
NACH = "nachtrainiert"        # faire Referenz (Aenderung Direktor 24.09.)
REFERENZ = NACH               # Nenner von G fuer S-0/S-1/S-3; "voll" nur beschreibend

HYPOTHESEN = {
    "S-0": "Positivkontrolle je Konfiguration: nachtrainiert (faire Referenz, gleiche zweite "
           "Trainingsphase wie die Rundungsarme) schlaegt die Identitaet (Gewinn > 0, "
           "exakter Wilcoxon p < 0,05); sonst ist M50 bei dieser Breite bedeutungslos.",
    "S-1": "Haupt (echt): Gewinnanteil G (Referenz nachtrainiert, Basis Identitaet) bei gleichem Anteil k/(n*L) in {1/8, 1/4} ist fuer "
           "n = 10 hoeher als fuer n = 6 (gepaart ueber Seeds, exakter Wilcoxon einseitig, Holm "
           "ueber S-1/S-3) -> H1: Magiebedarf waechst langsamer als n*L. Nachweislich niedriger "
           "-> H0 (niedriger); TOST +-0,08 -> H0 (gleich); sonst HU.",
    "S-2": "beschreibend: M50, M90 absolut und relativ je n, Fit M50 ~ a*n^b, G-, G_voll- und "
           "G_mag-Kurven, d_nach = nachtrainiert - voll, Differenzen je Anteil und fuer k = 1, 2.",
    "S-3": "Wie S-1, aber Aufgabe festgehalten (8x8-Information, hochskaliert auf 10 Qubits = "
           "|x> (x) |+>^4): trennt Breite von Bildaufloesung.",
}


# ------------------------------------------------------------------ Konfigurationen
def _cfg(aufgabe, n):
    return f"{aufgabe}{n}"


def _cfg_zerlege(name):
    for a in ("echt", "fest"):
        if name.startswith(a):
            return a, int(name[len(a):])
    raise ValueError(name)


def stufen(n, lagen=LAGEN):
    """Magie-Stufen k fuer Breite n: 0, 1, 2 und die Anteile 1/8, 1/4, 1/2 von n*L."""
    nl = n * lagen
    ks = {0, *K_ABSOLUT}
    for f in ANTEILE_ALLE:
        k = f * nl
        if abs(k - round(k)) > 1e-9:
            raise ValueError(f"Anteil {f} von {nl} Winkeln ist keine ganze Zahl")
        ks.add(int(round(k)))
    return sorted(ks)


def _k_anteil(n, f):
    return int(round(f * n * LAGEN))


# ------------------------------------------------------------------ Modell (schnelle Simulation)
def _kron(mats):
    K = torch.ones(1, 1)
    for m in mats:
        K = torch.kron(K, m)
    return K


class SkalaZelle(QireZelle):
    """Dieselbe QIRE-Zelle wie clifford.QireZelle (gleiche Parameter, Puffer,
    Initialisierung), aber die Schaltung wird auf den Zustaenden simuliert:
    Lage = Ry-Kronecker-Produkt = A (x) B mit A ueber die oberen, B ueber die
    unteren Qubits, angewandt als A * Psi * B^T, danach CNOT-Ring als
    Indexpermutation. Kosten je Bild und Lage 2^n * (2^(n/2) + 2^(n/2)) statt
    (2^n)^2 fuer die ganze Matrix."""

    def zustand(self, x):
        b, n = x.shape[0], self.n
        h = n // 2
        da, db = 2 ** h, 2 ** (n - h)
        psi = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        th = self.winkel()
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # (L, n, 2, 2)
        for l in range(self.lagen):
            A, B = _kron(list(R[l, :h])), _kron(list(R[l, h:]))
            p = (psi.reshape(b * da, db) @ B.T).reshape(b, da, db).transpose(1, 2).reshape(b * db, da) @ A.T
            psi = p.reshape(b, db, da).transpose(1, 2).reshape(b, -1)
            if self.cnot:
                psi = psi[:, self.perm]
        return psi

    def merkmale(self, x):
        if self.auslese != "born":
            raise ValueError("SkalaZelle kennt nur die Born-Auslese")
        return self.zustand(x) ** 2


def _baue_skala(arm, seed, n):
    """Wie clifford._baue fuer 'voll' und 'ohne_schaltung' (Vortest: bitgleich)."""
    torch.manual_seed(seed)
    m = SkalaZelle(n=n)
    if arm == "ohne_schaltung":
        m.fixiere(torch.zeros(LAGEN, n), torch.zeros(LAGEN, n))
    elif arm != "voll":
        raise ValueError(arm)
    return m


def _gerundet_skala(voll, k, seed):
    """Wie clifford._gerundet (v2): runden, k Magiewinkel (groesster Abstand) frei,
    Start beim trainierten Wert, Kopf FRISCH. Vortest: state_dict bitgleich."""
    n = voll.n
    th = voll.theta.detach().clone()
    m = SkalaZelle(n=n)
    m.load_state_dict(voll.state_dict())
    torch.manual_seed(seed)
    m.kopf.reset_parameters()
    frei = torch.zeros(LAGEN * n)
    frei[_magie_reihenfolge(th)[:k]] = 1
    m.fixiere(runde_clifford(th), frei.reshape(LAGEN, n))
    with torch.no_grad():
        m.theta.copy_(torch.where(m.frei.bool(), th, runde_clifford(th)))
    return m


def _trainiere(m, x, y, schritte, lr, seed):
    """Ohne freien Winkel ist die Schaltung fest: Merkmale einmal berechnen und
    nur den Kopf trainieren (identisch zu trainiere(m, ...), Vortest)."""
    if not bool(m.frei.bool().any()):
        with torch.no_grad():
            p = m.merkmale(x)
        return trainiere(m.kopf, p, y, schritte, lr, seed=seed)
    return trainiere(m, x, y, schritte, lr, seed=seed)


def _bewerte(m, x, y):
    with torch.no_grad():
        return genauigkeit(m(x), y)


# ------------------------------------------------------------------ Daten
def _hochskalieren(x, seite, von=8):
    """Naechster Nachbar von von x von auf seite x seite: |x> (x) |+>^(Zusatzbits)."""
    if seite == von:
        return x
    return torch.nn.functional.interpolate(x.reshape(-1, 1, von, von), size=(seite, seite),
                                           mode="nearest").reshape(len(x), -1)


@functools.lru_cache(maxsize=4)
def _daten(aufgabe, n, seed, n_train, n_test, datensatz):
    seite = 2 ** (n // 2)
    if aufgabe == "echt":
        d = bilder(datensatz, seed, n_train=n_train, n_test=n_test, seite=seite)
    else:
        xtr, ytr, xte, yte = bilder(datensatz, seed, n_train=n_train, n_test=n_test, seite=8)
        d = (_hochskalieren(xtr, seite), ytr, _hochskalieren(xte, seite), yte)
    assert d[0].shape[1] == 2 ** n, (aufgabe, n, d[0].shape)
    return d


def _teile(aufgabe, n, seed, n_train, n_test, datensatz, teil):
    xtr, ytr, xte, yte = _daten(aufgabe, n, seed, n_train, n_test, datensatz)
    if teil == "tuning":
        nv = len(ytr) // 4
        return xtr[nv:], ytr[nv:], xtr[:nv], ytr[:nv]
    return xtr, ytr, xte, yte


# ------------------------------------------------------------------ Arbeiter
def _arbeit(a):
    """(cfg, seed, arm, lr, teil, k, winkel_voll, budget) -> {'acc', ...}.
    arm: 'voll' | 'ohne_schaltung' | 'rund' (mit k freien Magiewinkeln)."""
    cfg, s, arm, lr, teil, k, winkel, (schritte, n_train, n_test, ds) = a
    aufgabe, n = _cfg_zerlege(cfg)
    xtr, ytr, xev, yev = _teile(aufgabe, n, s, n_train, n_test, ds, teil)
    if arm == "rund":
        voll = SkalaZelle(n=n)
        with torch.no_grad():
            voll.theta.copy_(torch.tensor(winkel).reshape(voll.theta.shape))
        m = _gerundet_skala(voll, k, s)
    else:
        m = _baue_skala(arm, s, n)
    _trainiere(m, xtr, ytr, schritte, lr, s)
    z = {"acc": _bewerte(m, xev, yev)}
    if arm == "voll":
        th = m.theta.detach().clone()
        z["winkel"] = th.flatten().tolist()
        z["abstand"] = float((th - runde_clifford(th)).abs().mean())
    return z


def _kosten(a):
    cfg, _, arm, _, _, k, *_ = a
    n = _cfg_zerlege(cfg)[1]
    voll_training = arm == "voll" or (arm == "rund" and k > 0)
    return 4.0 ** n * (1.0 if voll_training else 0.04)


def _parallel(aufgaben):
    """Wie labor.gemeinsam.parallel, aber chunksize 1 und laengste Aufgaben zuerst
    (die Trainings sind bei n = 10 ~60x teurer als bei n = 6)."""
    import multiprocessing as mp
    prozesse = int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))
    if prozesse <= 1 or len(aufgaben) <= 1:
        return parallel(_arbeit, aufgaben, prozesse=1)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ordnung = sorted(range(len(aufgaben)), key=lambda i: -_kosten(aufgaben[i]))
    with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
        erg = pool.map(_arbeit, [aufgaben[i] for i in ordnung], chunksize=1)
    aus = [None] * len(aufgaben)
    for i, e in zip(ordnung, erg):
        aus[i] = e
    return aus


def _gittersuche(schluessel, aufgaben_fuer, gitter=LR_GITTER):
    """Regel 1: eigene Rate je Schluessel (Konfiguration, Arm) auf den Tuning-Seeds,
    volles Budget, Gitter in BEIDE Richtungen verlaengert, alle Schluessel einer
    Runde parallel. aufgaben_fuer(schluessel, lr) -> Liste von Aufgaben.
    Rueckgabe: info {schluessel: {lr, gitter, am_rand}}, roh {(schluessel, lr): [Ergebnis]}."""
    gitter = {s: sorted(gitter) for s in schluessel}
    wert = {s: {} for s in schluessel}
    roh, info, offen = {}, {}, set(schluessel)
    for _ in range(4):
        paare = [(s, lr) for s in sorted(offen) for lr in gitter[s] if lr not in wert[s]]
        aufg, zu = [], []
        for s, lr in paare:
            for a in aufgaben_fuer(s, lr):
                aufg.append(a)
                zu.append((s, lr))
        for (s, lr), e in zip(zu, _parallel(aufg)):
            roh.setdefault((s, lr), []).append(e)
        for s, lr in paare:
            wert[s][lr] = float(np.mean([e["acc"] for e in roh[(s, lr)]]))
        for s in list(offen):
            g = sorted(gitter[s])
            best = max(g, key=lambda v: wert[s][v])
            if best == g[0]:
                gitter[s] = [g[0] / 4] + g
            elif best == g[-1]:
                gitter[s] = g + [g[-1] * 4]
            else:
                info[s] = {"lr": best, "gitter": wert[s], "am_rand": False}
                offen.discard(s)
        if not offen:
            break
    for s in offen:
        info[s] = {"lr": max(wert[s], key=wert[s].get), "gitter": wert[s], "am_rand": True}
    return info, roh


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seeds=TUNING_SEEDS, schritte=SCHRITTE, n_train=N_TRAIN,
         n_test=N_TEST, datensatz="mnist", breiten=BREITEN, fest_breiten=FEST_BREITEN, **kw):
    if schnell:
        seeds, tuning_seeds, schritte, n_train, n_test = range(2), (920,), 60, 400, 1000
    seeds, tuning_seeds = list(seeds), list(tuning_seeds)
    t0 = time.time()
    budget = (schritte, n_train, n_test, datensatz)
    cfgs = [_cfg("echt", n) for n in breiten] + [_cfg("fest", n) for n in fest_breiten if n != 6]

    # Phase 1: Raten fuer voll und Identitaet (Tuning-Split)
    def auf_grund(sch, lr):
        cfg, arm = sch
        return [(cfg, s, arm, lr, "tuning", 0, None, budget) for s in tuning_seeds]
    info1, roh1 = _gittersuche([(c, a) for c in cfgs for a in ("voll", "ohne_schaltung")], auf_grund)
    lr_voll = {c: info1[(c, "voll")]["lr"] for c in cfgs}
    print(f"  Phase 1 (Raten voll/Identitaet) nach {time.time() - t0:.0f} s", flush=True)

    # Phase 2: Raten der Rundungsarme (k = 0 und Magie-Arme bei k = n*L/8),
    # auf den Voll-Modellen der Tuning-Seeds mit deren gewaehlter Rate.
    winkel_tuning = {c: [e["winkel"] for e in roh1[((c, "voll"), lr_voll[c])]] for c in cfgs}

    def auf_rund(sch, lr):
        cfg, arm = sch
        n = _cfg_zerlege(cfg)[1]
        k = 0 if arm == "k0" else _k_anteil(n, ANTEIL_TUNING)
        return [(cfg, s, "rund", lr, "tuning", k, w, budget) for s, w in zip(tuning_seeds, winkel_tuning[cfg])]
    info2, _ = _gittersuche([(c, a) for c in cfgs for a in ("k0", "kmagie")], auf_rund)
    info = {**info1, **info2}
    lr = {sch: v["lr"] for sch, v in info.items()}
    print(f"  Phase 2 (Raten Rundung) nach {time.time() - t0:.0f} s", flush=True)

    # Phase 3a: Mess-Seeds, voll und Identitaet
    aufg = [(c, s, arm, lr[(c, arm)], "mess", 0, None, budget)
            for c in cfgs for s in seeds for arm in ("voll", "ohne_schaltung")]
    zeilen = {c: {s: {"seed": s} for s in seeds} for c in cfgs}
    for (c, s, arm, *_), e in zip(aufg, _parallel(aufg)):
        zeilen[c][s][arm] = e["acc"]
        if arm == "voll":
            zeilen[c][s]["winkel_voll"] = e["winkel"]
            zeilen[c][s]["abstand_zum_gitter"] = e["abstand"]
    print(f"  Phase 3a (voll/Identitaet) nach {time.time() - t0:.0f} s", flush=True)

    # Phase 3b: Rundungsarme je Stufe
    # + Arm nachtrainiert (Aenderung Direktor 24.09.): k = n*L, alle Winkel frei,
    # Start beim trainierten voll, frischer Kopf, dieselbe Rate wie die Magie-Arme.
    aufg, namen = [], []
    for c in cfgs:
        n = _cfg_zerlege(c)[1]
        for s in seeds:
            for k in stufen(n) + [n * LAGEN]:
                aufg.append((c, s, "rund", lr[(c, "k0" if k == 0 else "kmagie")], "mess", k,
                             zeilen[c][s]["winkel_voll"], budget))
                namen.append(NACH if k == n * LAGEN else f"k{k}")
    for (c, s, *_), name, e in zip(aufg, namen, _parallel(aufg)):
        zeilen[c][s][name] = e["acc"]
    zeilen = {c: [zeilen[c][s] for s in seeds] for c in cfgs}
    for c in cfgs:
        n = _cfg_zerlege(c)[1]
        kh = [_k_anteil(n, f) for f in ANTEILE_HAUPT]
        for z in zeilen[c]:
            print(f"  {c} Seed {z['seed']}: voll {z['voll']:.4f}  nach {z[NACH]:.4f}  ohne {z['ohne_schaltung']:.4f}  "
                  + "  ".join(f"k{k} {z[f'k{k}']:.4f}" for k in [0, 1, 2] + kh), flush=True)

    ergebnis = {"abteilung": "clifford", "modul": "skala", "agenda": "C-skala", "version": 1,
                "datensatz": datensatz, "hypothesen": HYPOTHESEN, "konfigurationen": cfgs,
                "lr": {f"{c}/{a}": v for (c, a), v in info.items()},
                "stufen": {c: stufen(_cfg_zerlege(c)[1]) for c in cfgs},
                "schritte": schritte, "n_train": n_train, "n_test": n_test,
                "seeds": seeds, "tuning_seeds": tuning_seeds, "zeilen": zeilen,
                "sekunden": time.time() - t0,
                "multiplikationen": {c: {f"k{k}": multiplikationen(_cfg_zerlege(c)[1], magie=k)
                                         for k in stufen(_cfg_zerlege(c)[1])} | {
                    "voll": multiplikationen(_cfg_zerlege(c)[1])} for c in cfgs}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Statistik
def wilcoxon_einseitig(d):
    """Exakter einseitiger Wilcoxon-Vorzeichen-Rang-Test (Durchschnittsraenge bei
    Bindungen, exakte Verteilung ueber alle 2^n Vorzeichen). -> (p_rechts, p_links):
    p_rechts = P(W+ >= w | H0) (Alternative: d > 0), p_links = P(W+ <= w | H0)."""
    from scipy.stats import rankdata
    d = np.asarray([v for v in d if v != 0.0], dtype=float)
    n = len(d)
    if n == 0:
        return 1.0, 1.0
    if n > 20:
        raise ValueError("exakte Aufzaehlung nur bis 20 Werte")
    r = rankdata(np.abs(d))
    w = r[d > 0].sum()
    alle = np.array(list(product([0.0, 1.0], repeat=n))) @ r
    return float(np.mean(alle >= w - 1e-9)), float(np.mean(alle <= w + 1e-9))


def _gepaart(d):
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    p_r, p_l = wilcoxon_einseitig(d)
    if len(d) < 2:
        return {"mittel": m, "p_rechts": p_r, "p_links": p_l, "ki_unten": -math.inf, "ki_oben": math.inf,
                "n": int(len(d))}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return {"mittel": m, "p_rechts": p_r, "p_links": p_l, "ki_unten": m - q * se, "ki_oben": m + q * se,
            "n": int(len(d))}


def _holm(p: dict, alpha=ALPHA) -> dict:
    """Holm-Stufenverfahren: {name: p} -> {name: signifikant}."""
    sig = {k: False for k in p}
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        if v <= alpha / (len(p) - i):
            sig[k] = True
        else:
            break
    return sig


def _m_schwelle(kurve: dict, schwelle):
    """kurve {k: G}: (kleinste Stufe mit G >= schwelle, linear interpoliertes k)."""
    ks = sorted(kurve)
    stufe = next((k for k in ks if kurve[k] is not None and kurve[k] >= schwelle), None)
    if stufe is None:
        return None, None
    i = ks.index(stufe)
    if i == 0:
        return stufe, float(stufe)
    k0, g0, g1 = ks[i - 1], kurve[ks[i - 1]], kurve[stufe]
    if g1 <= g0:
        return stufe, float(stufe)
    return stufe, float(k0 + (schwelle - g0) / (g1 - g0) * (stufe - k0))


def _potenzfit(punkte):
    """[(n, M)] mit M > 0 -> {a, b} fuer M ~ a * n^b (kleinste Quadrate in log-log)."""
    p = [(n, m) for n, m in punkte if m is not None and m > 0]
    if len(p) < 2:
        return None
    b, loga = np.polyfit(np.log([n for n, _ in p]), np.log([m for _, m in p]), 1)
    return {"a": float(math.exp(loga)), "b": float(b), "b_relativ": float(b - 1), "punkte": len(p)}


def _je_seed_G(zeilen, gewinn, ref=REFERENZ):
    """{seed: {k: G_s(k)}} als Pseudowerte (Delta-Methode): G_s(k) = (N_s(k) -
    G(k) * (D_s - D)) / D mit N_s(k) = acc_k - acc_ohne, D_s = acc_ref - acc_ohne,
    D = Mittel(D_s), G(k) = Mittel(N)/D. Das Mittel der Pseudowerte ist genau G(k),
    ihre Streuung enthaelt aber auch die Unsicherheit des Nenners D. Mit festem
    Nenner (wie in magie_pruning.py) war der gepaarte Test ueber zwei Breiten zu
    liberal: der Schaetzfehler von D verschiebt alle Seeds gemeinsam (Vortest
    24.09.: „H0 (gleich)“-Fall kippte signifikant)."""
    if gewinn <= 0:          # Vorbedingung verletzt: nur beschreibend, ohne Korrektur
        return {z["seed"]: {int(k[1:]): z[k] - z["ohne_schaltung"]
                            for k in z if k.startswith("k") and k[1:].isdigit()} for z in zeilen}
    ks = sorted(int(k[1:]) for k in zeilen[0] if k.startswith("k") and k[1:].isdigit())
    G = {k: float(np.mean([z[f"k{k}"] - z["ohne_schaltung"] for z in zeilen])) / gewinn for k in ks}
    out = {}
    for z in zeilen:
        dd = (z[ref] - z["ohne_schaltung"]) - gewinn
        out[z["seed"]] = {k: (z[f"k{k}"] - z["ohne_schaltung"] - G[k] * dd) / gewinn for k in ks}
    return out


def auswertung(zeilen: dict) -> dict:
    """zeilen {cfg: [ {seed, voll, ohne_schaltung, k<k>...} ]}."""
    aus = {"konfig": {}}
    Gs = {}
    for c, zz in zeilen.items():
        n = _cfg_zerlege(c)[1]
        a = lambda k: np.array([z[k] for z in zz], dtype=float)
        ks = stufen(n)
        d0 = a(REFERENZ) - a("ohne_schaltung")
        gewinn = float(d0.mean())
        p0 = wilcoxon_exakt(d0)
        dv = a("voll") - a("ohne_schaltung")
        gewinn_voll = float(dv.mean())
        dn = a(NACH) - a("voll")
        gm = float((a(REFERENZ) - a("k0")).mean())
        G = {k: (float((a(f"k{k}") - a("ohne_schaltung")).mean()) / gewinn if gewinn > 0 else None) for k in ks}
        Gvoll = {k: (float((a(f"k{k}") - a("ohne_schaltung")).mean()) / gewinn_voll if gewinn_voll > 0 else None)
                 for k in ks}
        Gmag = {k: (float((a(f"k{k}") - a("k0")).mean()) / gm if gm > 0 else None) for k in ks}
        info = {"n": n, "winkel": n * LAGEN, "referenz": REFERENZ, "gewinn": gewinn, "S0_p": p0,
                "vorbedingung": bool(gewinn > 0 and p0 < ALPHA),
                "gewinn_voll": gewinn_voll, "S0_p_voll": wilcoxon_exakt(dv),
                "d_nach": float(dn.mean()), "d_nach_p": wilcoxon_exakt(dn),
                "gewinn_ueber_k0": gm, "G": {str(k): v for k, v in G.items()},
                "G_voll": {str(k): v for k, v in Gvoll.items()},
                "G_mag": {str(k): v for k, v in Gmag.items()},
                "genauigkeit": {arm: zusammenfassung(a(arm)) for arm in ["voll", NACH, "ohne_schaltung"]
                                + [f"k{k}" for k in ks]},
                "abstand_zum_gitter": float(a("abstand_zum_gitter").mean()) if "abstand_zum_gitter" in zz[0] else None}
        for name, kurve in (("", G), ("_voll", Gvoll), ("_mag", Gmag)):
            if all(v is not None for v in kurve.values()):
                for sw in (50, 90):
                    st, ip = _m_schwelle(kurve, sw / 100)
                    info[f"M{sw}{name}_stufe"], info[f"M{sw}{name}_interp"] = st, ip
                    info[f"m{sw}{name}_relativ"] = None if ip is None else ip / (n * LAGEN)
        aus["konfig"][c] = info
        Gs[c] = _je_seed_G(zz, gewinn)

    def vergleich(ca, cb, paare_k):
        """d_s = Mittel ueber (ka, kb) von G_b,s(kb) - G_a,s(ka), gepaart ueber Seeds."""
        gemeinsam = sorted(set(Gs[ca]) & set(Gs[cb]))
        d = [np.mean([Gs[cb][s][kb] - Gs[ca][s][ka] for ka, kb in paare_k]) for s in gemeinsam]
        return _gepaart(d)

    def k_paare(ca, cb, anteile):
        na, nb = _cfg_zerlege(ca)[1], _cfg_zerlege(cb)[1]
        return [(_k_anteil(na, f), _k_anteil(nb, f)) for f in anteile]

    for name, (ca, cb) in (("S1", PAAR_HAUPT), ("S3", PAAR_FEST)):
        if ca in zeilen and cb in zeilen:
            aus[name] = vergleich(ca, cb, k_paare(ca, cb, ANTEILE_HAUPT))
            aus[name]["paar"] = [ca, cb]
            aus[name]["vorbedingung"] = bool(aus["konfig"][ca]["vorbedingung"] and aus["konfig"][cb]["vorbedingung"])
    # beschreibend: je Anteil, absolutes k, benachbarte Breiten
    beschr = {}
    namen = list(zeilen)
    for i, ca in enumerate(namen):
        for cb in namen[i + 1:]:
            e = {f"anteil_{f:.4g}": vergleich(ca, cb, k_paare(ca, cb, [f])) for f in (0.0,) + ANTEILE_ALLE}
            e.update({f"k{k}_absolut": vergleich(ca, cb, [(k, k)]) for k in K_ABSOLUT})
            beschr[f"{ca}|{cb}"] = {k: {"mittel": v["mittel"], "p_rechts": v["p_rechts"],
                                        "p_links": v["p_links"]} for k, v in e.items()}
    aus["S2_vergleiche"] = beschr
    for aufgabe in ("echt", "fest"):
        pkt = [(c, info) for c, info in aus["konfig"].items() if _cfg_zerlege(c)[0] == aufgabe
               or (aufgabe == "fest" and c == "echt6")]
        if aufgabe == "fest" and len(pkt) < 2:
            continue
        aus[f"S2_fit_{aufgabe}"] = _potenzfit([(info["n"], info.get("M50_interp")) for _, info in pkt])
        aus[f"S2_fit90_{aufgabe}"] = _potenzfit([(info["n"], info.get("M90_interp")) for _, info in pkt])
    # Holm ueber die Familie {S-1, S-3} (nur mit erfuellter Vorbedingung)
    p_fam = {}
    for name in ("S1", "S3"):
        if name in aus and aus[name]["vorbedingung"]:
            v = aus[name]
            # Richtung erst nach dem Blick auf die Daten -> zweiseitig (x2), sonst
            # waeren H1 und H0 (niedriger) zusammen ein Test auf Niveau 0,10.
            p_fam[name] = min(1.0, 2 * (v["p_rechts"] if v["mittel"] > 0 else v["p_links"]))
    sig = _holm(p_fam)
    for name in ("S1", "S3"):
        if name in aus:
            aus[name]["p_richtung"] = p_fam.get(name)
            aus[name]["holm_signifikant"] = sig.get(name, False)
    aus["n_seeds"] = {c: len(zz) for c, zz in zeilen.items()}
    return aus


def _urteil_vergleich(v, was):
    if not v["vorbedingung"]:
        return f"HU (Vorbedingung): Schaltung schlaegt die Identitaet nicht nachweisbar bei {v['paar']} -- G undefiniert"
    txt = (f"d = {v['mittel']:+.3f} G (KI {v['ki_unten']:+.3f} .. {v['ki_oben']:+.3f}, "
           f"p_rechts = {v['p_rechts']:.4f}, p_links = {v['p_links']:.4f}, Holm {'ja' if v['holm_signifikant'] else 'nein'})")
    if v["mittel"] > 0 and v["holm_signifikant"]:
        return f"H1: {was}: bei gleichem Winkelanteil hoeherer Gewinnanteil fuer n = 10 -- Magiebedarf waechst langsamer als n*L; {txt}"
    if v["mittel"] < 0 and v["holm_signifikant"]:
        return f"H0 (niedriger): {was}: Gewinnanteil fuer n = 10 nachweislich niedriger -- Magiebedarf waechst mindestens wie n*L; {txt}"
    if v["ki_unten"] > -DELTA and v["ki_oben"] < DELTA:
        return f"H0 (gleich): {was}: TOST +-{DELTA} -- relativer Magiebedarf M50/(n*L) bleibt gleich; {txt}"
    return f"HU: {was}: kein Urteil; {txt}"


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Praefixe: 'H1:',
    'H0 (niedriger):', 'H0 (gleich):', 'HU:', 'HU (Vorbedingung):'."""
    u = {}
    for c, info in a["konfig"].items():
        zus = (f"; beschreibend voll-Identitaet {info['gewinn_voll']:+.4f} (p = {info['S0_p_voll']:.4f}), "
               f"d_nach = {info['d_nach']:+.4f} (p = {info['d_nach_p']:.4f})")
        u[f"S-0 {c}"] = (f"H1: nachtrainiert schlaegt Identitaet (Gewinn {info['gewinn']:+.4f}, p = {info['S0_p']:.4f})"
                         if info["vorbedingung"] else
                         f"H0: Gewinn ueber Identitaet nicht nachweisbar ({info['gewinn']:+.4f}, p = {info['S0_p']:.4f})") + zus
    if "S1" in a:
        u["S-1"] = _urteil_vergleich(a["S1"], "Breite+Aufloesung (echt6 -> echt10)")
    if "S3" in a:
        u["S-3"] = _urteil_vergleich(a["S3"], "nur Breite (echt6 -> fest10)")
    teile = []
    for c, info in a["konfig"].items():
        f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
        teile.append(f"{c}: M50 {f(info.get('M50_stufe'))} (interp. {f(info.get('M50_interp'))}, "
                     f"relativ {f(info.get('m50_relativ'))}), M90 {f(info.get('M90_stufe'))}, "
                     f"G(1) {f(info['G'].get('1'))}; bezogen auf voll: M50 {f(info.get('M50_voll_stufe'))} "
                     f"(interp. {f(info.get('M50_voll_interp'))}), d_nach {info['d_nach']:+.4f}")
    for aufgabe in ("echt", "fest"):
        fit = a.get(f"S2_fit_{aufgabe}")
        if fit:
            teile.append(f"Fit {aufgabe}: M50 ~ {fit['a']:.2f} * n^{fit['b']:.2f} ({fit['punkte']} Punkte)")
    u["S-2"] = "; ".join(teile)
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_struktur() -> bool:
    """Tun die Bausteine, was ihr Name sagt? (Lehren „entartete Kontrolle“,
    „geerbter Kopf“.) Billig: untrainierte Modelle, Zufallsdaten."""
    from abteilungen.clifford import _baue, _gerundet
    ok = True

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    # 1) schnelle Simulation == ganze Matrix aus clifford.py
    for n in (6, 8, 10):
        torch.manual_seed(n)
        x = torch.rand(12, 2 ** n)
        alt, neu = QireZelle(n=n), SkalaZelle(n=n)
        neu.load_state_dict(alt.state_dict())
        with torch.no_grad():
            if not torch.allclose(alt.merkmale(x), neu.merkmale(x), atol=1e-5):
                fehler(f"SkalaZelle weicht von QireZelle ab, n = {n}")
            if not torch.allclose(neu.merkmale(x).sum(1), torch.ones(12), atol=1e-4):
                fehler(f"Born-Wahrscheinlichkeiten nicht normiert, n = {n}")
    # 2) Startzustaende und Rundungsarme bitgleich zu clifford.py
    for arm in ("voll", "ohne_schaltung"):
        a, _ = _baue(arm, 5, 6)
        b = _baue_skala(arm, 5, 6)
        if any(not torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()):
            fehler(f"_baue_skala weicht von clifford._baue ab ({arm})")
    torch.manual_seed(3)
    voll_q = QireZelle(n=6)
    voll_s = SkalaZelle(n=6)
    voll_s.load_state_dict(voll_q.state_dict())
    th = voll_s.theta.detach().clone()
    for k in (0, 1, 3, 12):
        a, b = _gerundet(voll_q, k, 7), _gerundet_skala(voll_s, k, 7)
        if any(not torch.equal(a.state_dict()[n_], b.state_dict()[n_]) for n_ in a.state_dict()):
            fehler(f"_gerundet_skala weicht von clifford._gerundet ab, k = {k}")
        w = b.winkel().detach().flatten()
        frei = b.frei.flatten().bool()
        if int(frei.sum()) != k or not torch.equal(w[frei], th.flatten()[frei]):
            fehler(f"falsche freie Winkel, k = {k}")
        if not torch.allclose(w[~frei] / GITTER, torch.round(w[~frei] / GITTER)):
            fehler(f"gerundete Winkel nicht auf dem Gitter, k = {k}")
        if torch.equal(b.kopf.weight, voll_s.kopf.weight):
            fehler(f"Kopf geerbt statt frisch, k = {k}")
    # 2b) Arm nachtrainiert: k = n*L -> alle Winkel frei, Winkel == voll vor dem
    #     Nachtraining, Kopf frisch, Kosten als volles Training eingestuft.
    for n in (6, 10):
        torch.manual_seed(11)
        v = SkalaZelle(n=n)
        m = _gerundet_skala(v, n * LAGEN, 7)
        if not (bool(m.frei.bool().all()) and torch.equal(m.winkel().detach(), v.theta.detach())
                and torch.equal(m.theta.detach(), v.theta.detach())):
            fehler(f"nachtrainiert: Winkel != voll oder nicht alle frei, n = {n}")
        if torch.equal(m.kopf.weight, v.kopf.weight):
            fehler(f"nachtrainiert: Kopf geerbt statt frisch, n = {n}")
        with torch.no_grad():
            x = torch.rand(5, 2 ** n)
            if not torch.allclose(m.merkmale(x), v.merkmale(x), atol=1e-6):
                fehler(f"nachtrainiert: Schaltung vor dem Nachtraining != voll, n = {n}")
        if _kosten(("echt10", 0, "rund", 0.1, "mess", 40, None, None)) != 4.0 ** 10:
            fehler("nachtrainiert nicht als volles Training eingestuft")
    # 3) Identitaet ist eine reine Permutation (auch bei n = 10)
    for n in (6, 10):
        ohne = _baue_skala("ohne_schaltung", 0, n)
        with torch.no_grad():
            U = ohne.zustand(torch.eye(2 ** n))
        if not (torch.allclose(U.abs().sum(1), torch.ones(2 ** n), atol=1e-5)
                and int((U.abs() > 1e-4).sum()) == 2 ** n):
            fehler(f"Identitaets-Kontrolle ist keine Permutation, n = {n}")
    # 4) Kopf-Training auf festen Merkmalen == volles Training
    torch.manual_seed(0)
    x, y = torch.rand(150, 64), torch.randint(0, 10, (150,))
    for bau in (lambda: _baue_skala("ohne_schaltung", 2, 6), lambda: _gerundet_skala(voll_s, 0, 2)):
        m1, m2 = bau(), bau()
        _trainiere(m1, x, y, 25, 0.08, 2)
        trainiere(m2, x, y, 25, 0.08, seed=2)
        if not (torch.allclose(m1.kopf.weight, m2.kopf.weight, atol=1e-5)
                and torch.equal(m1.theta, m2.theta)):
            fehler("Kopf-Training auf festen Merkmalen weicht vom vollen Training ab")
    # 5) Arm „fest“: hochskaliertes 8x8-Bild = |x> (x) |+>^4, Identitaet sieht nur x
    x8 = torch.rand(3, 64)
    x32 = _hochskalieren(x8, 32)
    v = (x32 / x32.norm(dim=1, keepdim=True)).reshape(3, 8, 4, 8, 4)
    plus = v[:, :, :1, :, :1].expand_as(v)
    if not torch.allclose(v, plus) or not torch.allclose(v[:, :, 0, :, 0] * 4, x8.reshape(3, 8, 8) / x8.norm(dim=1).reshape(3, 1, 1), atol=1e-6):
        fehler("Hochskalierung ist kein Produktzustand |x> (x) |+>^4")
    # 6) Stufen ganzzahlig, Anteile vergleichbar
    for n in BREITEN:
        ks = stufen(n)
        if not {0, 1, 2} <= set(ks) or any(_k_anteil(n, f) not in ks for f in ANTEILE_ALLE):
            fehler(f"Stufen unvollstaendig, n = {n}: {ks}")
    if stufen(6) != [0, 1, 2, 3, 6, 12] or stufen(10) != [0, 1, 2, 5, 10, 20]:
        fehler("Stufen nicht wie vorregistriert")
    # 7) einseitiger Wilcoxon konsistent mit dem zweiseitigen aus labor
    rng = np.random.default_rng(3)
    for _ in range(5):
        d = rng.normal(0.2, 1, 10)
        p_r, p_l = wilcoxon_einseitig(d)
        if abs(min(1.0, 2 * min(p_r, p_l)) - wilcoxon_exakt(d)) > 1e-9:
            fehler("einseitiger Wilcoxon passt nicht zum zweiseitigen")
    if abs(wilcoxon_einseitig(np.arange(1, 11))[0] - 1 / 1024) > 1e-12:
        fehler("kleinster einseitiger p bei 10 Seeds ist nicht 1/1024")
    if _holm({"a": 0.04, "b": 0.30}) != {"a": False, "b": False} or \
            _holm({"a": 0.04, "b": 0.01}) != {"a": True, "b": True} or \
            _holm({"a": 0.03}) != {"a": True}:
        fehler("Holm falsch")
    return ok


def _synth(rng, g_kurven, gewinne=None, sd=0.002, sd_seed=0.0, n_seeds=10):
    """Synthetische Zeilen. g_kurven {cfg: Funktion k -> G}; gewinne {cfg: Gewinn};
    sd = Rauschen je Arm, sd_seed = zusaetzliches Rauschen je Seed auf allen k>0
    (streut d, ohne die Mittel zu verschieben)."""
    gewinne = gewinne or {}
    out = {}
    for c, g in g_kurven.items():
        n = _cfg_zerlege(c)[1]
        gw = gewinne.get(c, 0.04)
        zz = []
        for s in range(n_seeds):
            ohne = 0.88 + rng.normal(0, sd)
            z = {"seed": s, "ohne_schaltung": ohne, NACH: ohne + gw + rng.normal(0, sd),
                 "voll": ohne + gw - 0.005 + rng.normal(0, sd),
                 "abstand_zum_gitter": 0.35}
            extra = rng.normal(0, sd_seed)
            for k in stufen(n):
                z[f"k{k}"] = ohne + gw * g(k) + rng.normal(0, sd) + (extra if k > 0 else 0.0)
            zz.append(z)
        out[c] = zz
    return out


def _linear_bis(m50):
    """G(k) = 0,5 * k / M50, bei 1 gekappt: M50 exakt bekannt."""
    return lambda k: min(1.0, 0.5 * k / m50)


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: jeden Urteilszweig auf synthetischen Daten erzwingen.
    Grenzfaelle kippen mit Rate ~alpha -- je Fall und Hypothese begrenzt erlaubt."""
    ok = _vortest_struktur()
    rng = np.random.default_rng(17)
    alle4 = lambda f6, f8, f10, ff10: {"echt6": f6, "echt8": f8, "echt10": f10, "fest10": ff10}
    # Kurven: G = 0,5 * k / M50 mit M50 je Breite. Anteil 1/8, 1/4 bei n = 6: k = 3, 6;
    # bei n = 10: k = 5, 10. M50 konstant 2 -> G_6 = 0,75/1,0, G_10 = 1,0/1,0 -> d = +0,125.
    konst = _linear_bis(2)
    prop6, prop8, prop10 = _linear_bis(3), _linear_bis(4), _linear_bis(5)   # M50 ~ n*L/8
    faelle = [
        # H1: Magiebedarf konstant in n (echt und fest)
        (dict(g_kurven=alle4(konst, konst, konst, konst)), {"S-1": "H1:", "S-3": "H1:", "S-0 echt10": "H1:"}, 0),
        # H0 (niedriger): n = 10 braucht relativ MEHR Magie (M50 = 10 statt 3)
        (dict(g_kurven=alle4(prop6, prop8, _linear_bis(10), _linear_bis(10))),
         {"S-1": "H0 (niedriger):", "S-3": "H0 (niedriger):"}, 0),
        # H0 (gleich): M50 proportional zu n*L, wenig Rauschen
        (dict(g_kurven=alle4(prop6, prop8, prop10, prop10), sd=0.001),
         {"S-1": "H0 (gleich):", "S-3": "H0 (gleich):"}, 1),
        # HU: proportional, aber starke Streuung je Seed (sd(d) ~ 0,5 G)
        (dict(g_kurven=alle4(prop6, prop8, prop10, prop10), sd_seed=0.02),
         {"S-1": "HU:", "S-3": "HU:"}, 2),
        # Vorbedingung verletzt bei echt10 -> S-1 HU (Vorbedingung), S-3 unberuehrt (H1)
        (dict(g_kurven=alle4(konst, konst, konst, konst), gewinne={"echt10": -0.003}),
         {"S-0 echt10": "H0:", "S-0 echt6": "H1:", "S-1": "HU (Vorbedingung):", "S-3": "H1:"}, 0),
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
                print(f"  VORTEST FEHLER: {h}: {f} von {ziehungen} falsch (erwartet {erwartet[h]})")
    # Holm greift: S-1 knapp (p_rechts ~ 0,03), S-3 klar nicht -> S-1 darf nicht H1 sein.
    a = auswertung(_synth(rng, alle4(prop6, prop8, prop10, prop10), sd=0.0005))
    a["S1"].update(mittel=0.05, p_rechts=0.03, p_links=0.97, ki_unten=0.001, ki_oben=0.2)
    a["S3"].update(mittel=0.01, p_rechts=0.40, p_links=0.60, ki_unten=-0.2, ki_oben=0.2)
    p_fam = {"S1": 0.03, "S3": 0.40}
    for k, v in _holm(p_fam).items():
        a[k]["holm_signifikant"] = v
    if not urteil(a)["S-1"].startswith("HU:"):
        ok = False
        print("  VORTEST FEHLER: Holm-Korrektur greift nicht", urteil(a)["S-1"])
    # Referenz: G wird auf nachtrainiert bezogen, G_voll nur beschreibend, d_nach berichtet
    a = auswertung(_synth(rng, alle4(konst, konst, konst, konst), sd=1e-5))
    for c, info in a["konfig"].items():
        if abs(info["d_nach"] - 0.005) > 1e-3 or abs(info["G"]["20" if c.endswith("10") else "12" if c == "echt6" else "16"] - 1) > 1e-3:
            ok = False
            print("  VORTEST FEHLER: Referenz nachtrainiert / d_nach falsch", c, info["d_nach"])
        if not info["G_voll"]["2"] > info["G"]["2"]:
            ok = False
            print("  VORTEST FEHLER: G_voll nicht auf voll bezogen", c)
    # Vorbedingung haengt an nachtrainiert, nicht an voll
    zz = _synth(rng, alle4(konst, konst, konst, konst))
    for z in zz["echt10"]:
        z["voll"] = z["ohne_schaltung"] - 0.01
    u = urteil(auswertung(zz))
    if not (u["S-0 echt10"].startswith("H1:") and u["S-1"].startswith("H1:")):
        ok = False
        print("  VORTEST FEHLER: S-0 nicht auf nachtrainiert bezogen", u["S-0 echt10"][:80])
    # S-2: Fit erkennt b = 0 (M50 konstant) und b = 1 (M50 ~ n)
    for kurven, b_soll in ((alle4(konst, konst, konst, konst), 0.0), (alle4(prop6, prop8, prop10, prop10), 1.0)):
        a = auswertung(_synth(rng, kurven, sd=1e-5))
        fit = a["S2_fit_echt"]
        if fit is None or abs(fit["b"] - b_soll) > 0.05:
            ok = False
            print("  VORTEST FEHLER: Potenzfit", fit, "soll b =", b_soll)
        if not urteil(a)["S-2"].startswith("echt6: M50"):
            ok = False
            print("  VORTEST FEHLER: S-2-Text", urteil(a)["S-2"][:80])
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("C-skala" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
