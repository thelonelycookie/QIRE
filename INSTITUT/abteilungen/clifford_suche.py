"""Abteilung Clifford -- Diskrete Clifford-Suche (Karte I-005, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  Die Clifford-Stufe (Magie 0, 0 T-Gatter, klassisch mit
      Additionen simulierbar -- Gottesman-Knill) so gut machen wie moeglich,
      OHNE Gradienten: nur die diskrete Wahl der Winkel aus {0, pi/2, pi,
      3pi/2} wird gelernt; jede Magie steckt dann allein im Eingangszustand
      (den Daten, Amplitudenkodierung).
  Wie heute?  Die Clifford-Stufe entsteht durch Runden eines per Gradient
      trainierten Modells (k0; C-mnist: k0 ~ zufaellige Clifford, C-2 H0).
      I-003: Gradientenabstieg findet eine BEKANNTE Clifford-Loesung nicht
      (Strukturaufgabe: voll 0,726 < Identitaet 0,784).
  Was ist neu?  Ein eigener, diskreter Lerner: simulated annealing ueber die
      Clifford-Wahlen mit KONVEXEM Kopf (je Kandidat nur ein linearer Kopf,
      geschlossen geloest). Nutzen fuer I-015-hybrid: je besser die klassische
      Stufe 0, desto weniger Bilder muessen auf den Quantenrechner.
  Woran merken wir es?  (a) Positivkontrolle: auf der Strukturaufgabe aus
      I-003 findet die Suche die Lehrer-Loesung; (b) auf MNIST 16x16 ist die
      gefundene Clifford-Schaltung besser als der Median zufaelliger Clifford-
      Schaltungen (Karte), als k0 (Q-2) und als reine Zufallssuche mit
      gleichem Aufwand (Q-4).

STRUKTUR DES SUCHRAUMS (Rechnung + Vortest, vor dem Lauf):
  Ry(t + pi) = Ry(t) Ry(pi), und Ry(pi) = -iY ist eine (reelle) Pauli-Matrix. Eine
  Pauli-Matrix wandert durch jede Clifford-Schaltung bis ans Ende und wird dort eine
  Vorzeichen-Permutation -> die Born-Merkmale werden nur PERMUTIERT, der lineare
  Kopf gleicht das exakt aus. Also ist c gleichwertig zu c+2 in JEDER Position:
  von den „2 Bit je Gatter“ der Karte traegt nur EINES Information (c mod 2:
  Pauli-artig vs. Hadamard-artig). Effektiver Raum 2^(L n): MNIST 2^32, Ziffern
  2^24. Die Suche arbeitet auf diesen kanonischen Bits; Zufalls-Clifford werden
  weiter aus {0,1,2,3} gezogen (wie clifford.py „zufall_clifford“) -- gleichwertig.

VERFAHREN (fest vor dem Lauf):
  Merkmale einer Wahl c: Winkel c*pi/2 in die QireZelle (clifford.py, fixiere),
    U = unitaer() einmal je Kandidat, Born p = ((x/|x|) U^T)^2. U hat nur
    Eintraege 0 oder +-2^(-h/2) (Vortest) -- klassisch Vorzeichen-Permutation
    plus Hadamard-Additionen; hier genuegt die dichte Matrix.
  SUCHKRITERIUM (nur Nicht-Test-Daten, NIE Test): Ridge-Klassifikator (One-hot,
    Achsenabschnitt, standardisierte Merkmale, lambda = 1) mit EXAKTER Leave-one-
    out-Kreuzvalidierung ueber die Hutmatrix (LOO-Genauigkeit, bei Gleichstand
    kleinerer LOO-MSE). Begruendung: konvex, geschlossen, keine Lernrate (Regel 1
    kann nicht verletzt werden), ~50 ms je MNIST-Kandidat -- erst das macht
    Tausende Kandidaten je Seed bezahlbar. LOO statt fester Validierung: im
    Entwurf war die Validierung der Strukturaufgabe (204 Bilder) so verrauscht
    (sd ~3,4 pp) wie die ganze Streuung zufaelliger Schaltungen (3,6 pp).
  SUCHE = simulated annealing + gierige Politur: Start zufaellig (Generator
    START_SEED + seed). Je Schritt kippt EIN kanonisches Bit (ein Gatter Pauli- <->
    Hadamard-artig); Annahme bei Verbesserung/Gleichstand, sonst mit exp(D/T);
    T = tau * sd(Kriterium der 100 Zufalls-Clifford dieses Seeds), tau geometrisch
    von TAU_START = 1,4 auf TAU_ENDE = 0,14 ueber SA_SCHRITTE = 6.000 Vorschlaege
    (bereits bewertete Wahlen kommen aus dem Zwischenspeicher). Danach gierige
    Politur (alle L*n Einzel-Flips, erster Gewinn zaehlt, bis ein voller Durchgang
    nichts bringt oder POLITUR_MAX = 300 neue Bewertungen). Ergebnis = beste
    gesehene Wahl. Temperatur relativ zur Streuung (Lehre „Grenzen relativ zum
    Effekt“): dieselbe Regel auf beiden Aufgaben, obwohl die Skalen verschieden sind.
  BEWERTUNG (alle Arme GLEICH): jede Schaltung (Suche, jede der 100 Zufalls-
    Clifford, k0, Identitaet, Lehrer, Zufallssuche, voll) bekommt denselben Kopf:
    multinomiale logistische Regression (Softmax wie der QireZelle-Kopf und die
    Kaskaden-Konfidenz), standardisierte Merkmale, L2 = KOPF_L2 (je Aufgabe),
    L-BFGS bis zur Konvergenz (konvex, eindeutig, keine Lernrate), trainiert auf
    allen Nicht-Test-Bildern, gemessen auf dem TEST.

ENTWURFSPHASE (nur Entwurfs-Seeds MNIST 970/971, Struktur 980-989, vor jeder
  Messzahl; Mess-Seeds unberuehrt):
  - Ridge-lambda 0,1/1/10: LOO-Genauigkeit gleich (+-0,05 pp) -> 1.
  - Kopf-L2 (Gitter bis innen verlaengert): MNIST 1e-3/1e-2/3e-2/1e-1 ->
    0,900/0,904/0,894/0,879 -> 1e-2; Struktur 1e-4/1e-3/1e-2 -> 1e-3 (im Mittel
    ueber Lehrer, Identitaet, Zufall gleichauf mit 1e-4, besser als 1e-2).
  - Erste Fassung (feste Validierung, Zuege c -> c+1/2/3, T absolut 0,003) fand
    den Lehrer in 0/2 Seeds; Landschaft um den Lehrer: 1 Bit daneben -12 pp, ab 3
    Bit auf Zufallsniveau (enges Becken, kein Trichter aus der Ferne).
  - Mit kanonischen Bits + LOO, 6.000 Schritte, Lehrer (bis auf Aequivalenz)
    gefunden: tau 0,55 -> 6/10, 0,85 -> 8/10, 1,4 -> 9/10 (der zehnte Lauf fand
    eine Wahl mit HOEHEREM Kriterium als der Lehrer), ~2.900 Bewertungen;
    10.000 Schritte bei tau 0,85 -> 7/10. Gewaehlt: tau 1,4 -> 0,14.
  - MNIST 970 (tau 1,4): 5.344 Bewertungen, LOO 0,887 -> 0,911; Test 0,935
    gegen beste-von-100-Zufall 0,925 (Kopf L2 1e-3). Einzelwert, kein Befund.

ARME
  MNIST 16x16 (8 Qubits, 4 Lagen = 32 Winkel); je Seed 6.000 Nicht-Test-Bilder
    (Kriterium, Kopf, Gradiententraining von voll) und 5.000 Testbilder (nur
    Bewertung) -- wie C-mnist.
    suche          SA + Politur ab zufaelliger Clifford-Wahl (Hauptarm)
    zufall_i       N_ZUFALL = 100 zufaellige Clifford-Wahlen (Generator
                   ZUFALL_SEED + seed); Kennzahl: MEDIAN der Testgenauigkeit
    k0             volles Modell (QireZelle, lr 0,08, 1.500 Schritte wie C-mnist
                   Arm voll -- Rate aus C-mnist, dort je Arm gesucht, nie am Rand;
                   Abweichung von Regel 1 wie in kaskade.py offen benannt), alle
                   Winkel auf das Gitter gerundet, DERSELBE Kopf wie alle Arme
    zufallssuche   beste von genau so vielen NEUEN Zufallswahlen, wie die Suche
                   in diesem Seed bewertet hat (gleicher Aufwand, Q-4)
    identitaet     alle Winkel 0 (Born-Auslese direkt auf x)
    beschreibend:  zufall_best100 (beste der 100 nach Kriterium), k0_politur
                   (gierige Politur ab k0, <= 600 Bewertungen; Feld suche_ab_k0),
                   voll (trainierte Winkel, NICHT Clifford, derselbe Kopf),
                   voll_eigener_kopf (Adam-Kopf aus dem Training).
  Strukturaufgabe (magie_aufgabe.py: Ziffern 8x8, 6 Qubits, 24 Winkel, Clifford-
    Lehrer mit Marge; ~820 Nicht-Test / ~350 Testbilder je Seed): suche, lehrer
    (Lehrer-Winkel), identitaet, zufall_i, zufallssuche (beschreibend). Dazu je
    Seed: ist die gefundene Schaltung zum Lehrer AEQUIVALENT (gleiche Zeilen von U
    bis auf Vorzeichen und Reihenfolge -> gleiche Born-Merkmale bis auf
    Permutation)?

SEEDS: MNIST 190-199, Strukturaufgabe 200-209 (frisch). Direktor 25.09., vor dem Lauf: die
urspruenglichen Mess-Seeds 170/171/180/181 waren in einem fruehen Rauchtest (80 Schritte)
beruehrt -- deshalb ersetzt, Kriterien unveraendert.

VORREGISTRIERT (Regel 7, vor der ersten Zahl der Mess-Seeds):
  Q-0 Positivkontrolle (Strukturaufgabe, mit dem VERFAHREN -- Lehre I-003):
        G_L = mittel_s(acc_suche - acc_id) / mittel_s(acc_lehrer - acc_id)
      (Aufgaben-Nenner wie magie_aufgabe.py; Test, gleicher Kopf).
        Vorbedingung: Lehrer - Identitaet >= 5 pp und Wilcoxon p < 0,05,
          sonst                                  -> HU (Vorbedingung)
        G_L >= 0,80 und Suche > Identitaet (exakter Wilcoxon p < 0,05)
                                                 -> H1 (Verfahren findet die
                                                    bekannte Loesung)
        G_L < 0,50                               -> H0 (Suche blind)
        sonst                                    -> HU
      Schaerfer als „>= 90 % der Lehrer-Genauigkeit“: mit Identitaet ~0,78 und
      Lehrer ~0,93 laege 90 % bei 0,84, also schon bei 40 % des Gewinns.
  Q-1 Hauptfrage (Abbruchkriterium der Karte), MNIST, gepaart je Seed:
        d_s = acc_test(suche) - median_i acc_test(zufall_i)
  Q-2 Suche vs. k0: d_s = acc_test(suche) - acc_test(k0). Als TEST, weil es die
      fuer I-015-hybrid entscheidende Frage ist (ersetzt die gradientenfreie
      Suche das Runden als Weg zur klassischen Stufe 0?). Fair: beide Clifford
      (0 T), gleicher Kopf, gleiche Daten.
  Q-4 Suche vs. Zufallssuche gleichen Aufwands: d_s = acc_test(suche) -
      acc_test(zufallssuche). Trennt „Suche“ von blosser „Auswahl“: Q-1 H1 koennte
      schon aus der Auswahl unter vielen Kandidaten folgen.
      Verzweigung fuer Q-1, Q-2, Q-4 (exakter zweiseitiger Wilcoxon, Holm ueber
      die Familie {Q-1, Q-2, Q-4}, 10 Seeds, kleinstes p = 0,002):
        Holm-p < 0,05 und mittel(d) > 0          -> H1
        Holm-p < 0,05 und mittel(d) < 0          -> H0 (Gegenrichtung)
        zweiseitiges 90-%-KI(d) in (-0,3; +0,3) pp (TOST) und Q-0 = H1
                                                 -> H0 (gleich)  [Q-1: Karte
                                                    abbrechen]
        TOST erfuellt, aber Q-0 nicht H1         -> HU (Positivkontrolle)
        sonst                                    -> HU
      Grenze 0,3 pp wie C-referenz: kleiner als die Streuung einer einzelnen
      Zufalls-Clifford (C-mnist sd 1,0 pp) und als k0 - Identitaet (~1 pp);
      15 Testbilder je Seed.
  Q-3 beschreibend: Suchkurve (beste LOO-Genauigkeit ueber neue Bewertungen,
      Seed-Mittel), Rechenaufwand (Bewertungen, Sekunden; gegen das
      Gradiententraining von voll), Siegerfluch (Kriterium - Test), zufall_best100,
      k0_politur, voll, G_voll = (suche - id)/(voll - id), und fuer die Kaskade:
      Anteil der Testbilder mit Konfidenz > 0,9 und deren Genauigkeit (suche, k0).

AUFLOESUNG / GRENZEN (vorab benannt):
  MNIST-Test 5.000 Bilder (0,02 pp je Bild); Strukturaufgabe ~350 Testbilder
  (0,29 pp). Suchkriterium (Ridge-LOO) und Bewertungskopf (logistisch) sind beide
  linear auf denselben Born-Merkmalen; die Suche kann den Test nie sehen (Vortest:
  andere Test-Etiketten -> identische Suche). Das Kriterium wird durch Tausende
  Bewertungen ueberangepasst (Siegerfluch) -- betrifft nur die Kriteriumszahl,
  nicht den Test (Q-3 berichtet die Luecke). Die Aequivalenzpruefung ist
  hinreichend, nicht notwendig (andere Schaltungen koennen gleich gut sein).
  LAUFZEIT (Schaetzung, 4 Prozesse): je MNIST-Seed ~5.000 Bewertungen Suche +
  ebenso viele Zufallssuche (~50-80 ms) + 105 Koepfe (~0,3 s) + Gradienten-
  training voll -> ~12-15 min; 10 Seeds in 3 Wellen + Strukturseeds ~40-50 min.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen.clifford import GITTER, LAGEN, QireZelle, runde_clifford
from labor.gemeinsam import parallel, speichere, wilcoxon_exakt, zusammenfassung

N_MNIST, N_STRUKTUR = 8, 6
SEEDS_MNIST = range(190, 200)
SEEDS_STRUKTUR = range(200, 210)
N_ZUFALL = 100
SA_SCHRITTE = 6000         # SA-Vorschlaege je Suche (bereits bewertete Wahlen kosten nichts)
POLITUR_MAX = 300          # hoechstens so viele neue Bewertungen in der gierigen Politur
TAU_START, TAU_ENDE = 1.4, 0.14   # Temperatur in Vielfachen der sd des Kriteriums ueber Zufalls-Clifford
POLITUR_AB_K0 = 600        # beschreibend: gierige Politur ab k0, hoechstens so viele neue Bewertungen
RIDGE_LAMBDA = 1.0         # auf standardisierten Merkmalen
KOPF_L2 = {"mnist": 1e-2, "struktur": 1e-3}   # L2 des Bewertungskopfes je Aufgabe (Entwurfs-Seeds)
LR_VOLL, SCHRITTE_VOLL = 0.08, 1500   # wie kaskade.py (aus C-mnist)
START_SEED, ZUFALL_SEED, ZSUCHE_SEED, SA_SEED = 50_000, 60_000, 70_000, 80_000
ALPHA = 0.05
TOST = 0.003               # +-0,3 pp
G_H1, G_H0 = 0.80, 0.50
MIN_GEWINN_LEHRER = 0.05   # Q-0-Vorbedingung: Lehrer >= 5 pp ueber Identitaet (I-003: ~15 pp), sonst ist G_L instabil
KONF_KASKADE = 0.9
KURVE_PUNKTE = [0, 25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200]

HYPOTHESEN = {
    "Q-0": "Positivkontrolle (Strukturaufgabe aus I-003, Clifford-Lehrer, gleiches Verfahren): "
           "G_L = mittel(suche - id)/mittel(lehrer - id) >= 0,80 und Suche > Identitaet (exakter Wilcoxon, "
           "p < 0,05) -> H1; G_L < 0,50 -> H0; sonst HU; Vorbedingung Lehrer - id >= 5 pp und p < 0,05.",
    "Q-1": "Hauptfrage/Abbruch: auf MNIST 16x16 ist die gefundene Clifford-Schaltung (Auswahl nur nach "
           "LOO-Kriterium auf Nicht-Test-Daten, Test) besser als der Median von 100 zufaelligen Clifford-"
           "Schaltungen (gepaart je Seed, exakter Wilcoxon, Holm ueber Q-1/Q-2/Q-4) -> H1; besser Zufall -> "
           "H0 (Gegenrichtung); TOST +-0,3 pp bei Q-0 = H1 -> H0 (gleich, Karte abbrechen); sonst HU.",
    "Q-2": "Suche vs. k0 (gerundetes, per Gradient trainiertes Modell), gleicher Kopf, gepaart, "
           "Verzweigung wie Q-1.",
    "Q-4": "Suche vs. Zufallssuche mit gleich vielen Bewertungen (Suche oder nur Auswahl?), gepaart, "
           "Verzweigung wie Q-1.",
    "Q-3": "beschreibend: Suchkurve, Rechenaufwand, Siegerfluch, beste von 100, Politur ab k0, voll, "
           "G_voll, Kaskaden-Konfidenz (Anteil konf > 0,9).",
}


# ------------------------------------------------------------------ Bausteine
def unitaer_clifford(codes, n) -> torch.Tensor:
    """codes (L x n) in {0,1,2,3} -> U der QireZelle mit Winkeln codes*pi/2."""
    m = QireZelle(n=n, lagen=LAGEN)
    m.fixiere(torch.as_tensor(np.asarray(codes), dtype=torch.float32) * GITTER, torch.zeros(LAGEN, n))
    with torch.no_grad():
        return m.unitaer()


def unitaer_winkel(theta, n) -> torch.Tensor:
    m = QireZelle(n=n, lagen=LAGEN)
    m.fixiere(torch.as_tensor(theta, dtype=torch.float32).reshape(LAGEN, n), torch.zeros(LAGEN, n))
    with torch.no_grad():
        return m.unitaer()


def normiere(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def born(xn, U):
    """xn: normierte Eingaben; Born-Wahrscheinlichkeiten |U x|^2."""
    return (xn @ U.T) ** 2


def codes_aus_winkeln(theta) -> np.ndarray:
    """Gerundete Winkel -> Codes mod 4 (Ry(t + 2 pi) = -Ry(t): globales Vorzeichen, Born gleich)."""
    th = torch.as_tensor(theta, dtype=torch.float32).reshape(LAGEN, -1)
    return (torch.round(th / GITTER).long() % 4).numpy().astype(np.int64)


def _standard(ftr):
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    return mu, sd


def _mit_eins(a):
    return torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1)


def ridge_loo(f, y, lam=RIDGE_LAMBDA):
    """Ridge-Klassifikator (One-hot, Achsenabschnitt unbestraft, Merkmale standardisiert)
    mit EXAKTER Leave-one-out-Kreuzvalidierung ueber die Hutmatrix:
    P_loo,i = (P_i - h_ii Y_i) / (1 - h_ii). -> (LOO-Genauigkeit, LOO-MSE).
    Jedes Bild wird von einem Kopf bewertet, der es nicht gesehen hat (die
    Standardisierung ueber alle ist eine feste affine Abbildung, die ein linearer Kopf
    ohnehin ausgleicht). Grosse Produkte in float32, Inversion in float64 (im Entwurf
    identische Genauigkeiten wie ganz float64, 1,5x schneller); Vortest prueft die
    Formel gegen explizites Weglassen."""
    mu, sd = _standard(f)
    A = _mit_eins((f - mu) / sd).float()
    Y = torch.nn.functional.one_hot(y, 10).float()
    G = (A.T @ A).double()
    idx = torch.arange(A.shape[1] - 1)
    G[idx, idx] += lam
    M = torch.linalg.inv(G).float()
    AM = A @ M
    P = AM @ (A.T @ Y)
    h = (AM * A).sum(1, keepdim=True).clamp_max(1 - 1e-6)
    Pl = (P - h * Y) / (1 - h)
    return float((Pl.argmax(1) == y).float().mean()), float(((Pl - Y) ** 2).sum(1).mean())


def kopf_logistisch(ftr, ytr, fte, yte, l2=1e-3, max_iter=500):
    """Bewertungskopf fuer ALLE Arme: Softmax-Regression, standardisierte Merkmale,
    L2 (ohne Achsenabschnitt), L-BFGS (konvex, keine Lernrate).
    -> (acc_test, Konfidenz je Testbild, richtig je Testbild)."""
    mu, sd = _standard(ftr)
    A = _mit_eins((ftr - mu) / sd)
    B = _mit_eins((fte - mu) / sd)
    W = torch.zeros(A.shape[1], 10, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20)

    def schritt():
        opt.zero_grad()
        verlust = torch.nn.functional.cross_entropy(A @ W, ytr) + l2 * (W[:-1] ** 2).sum()
        verlust.backward()
        return verlust

    opt.step(schritt)
    with torch.no_grad():
        konf, pred = torch.softmax(B @ W, 1).max(1)
    richtig = (pred == yte).numpy()
    return float(richtig.mean()), konf.numpy(), richtig


def kaskaden_kennzahl(konf, richtig, schwelle=None):
    """Anteil der Bilder, die eine klassische Stufe 0 mit Konfidenz > Schwelle selbst
    entscheiden wuerde, und deren Genauigkeit (beschreibend fuer I-015-hybrid)."""
    schwelle = KONF_KASKADE if schwelle is None else schwelle
    sicher = konf > schwelle
    return {"anteil_sicher": float(sicher.mean()),
            "acc_sicher": float(richtig[sicher].mean()) if sicher.any() else None}


def kanonisch(codes) -> np.ndarray:
    """Ry(t + pi) = Ry(t) Ry(pi), und Ry(pi) ist eine reelle Pauli-Matrix (-iY). Eine Pauli-
    Matrix wandert durch jede Clifford-Schaltung bis ans Ende und wird dort zu einer
    Vorzeichen-Permutation -> Born-Merkmale nur PERMUTIERT, der Kopf gleicht das exakt
    aus. Also sind c und c+2 in JEDER Position gleichwertig: effektiv 1 Bit je Gatter
    (c mod 2: Pauli-artig 0 oder Hadamard-artig 1). Vortest prueft das."""
    return (np.asarray(codes, dtype=np.int64) % 2)


class Bewerter:
    """Suchkriterium: exakte LOO-Kreuzvalidierung des Ridge-Kopfes auf den Nicht-Test-
    Daten. Der Test ist nicht einmal Attribut. Zwischenspeicher je kanonischer Wahl."""

    def __init__(self, x, y, n):
        self.x, self.y, self.n = normiere(x), y, n
        self.cache, self.bewertungen, self.sekunden = {}, 0, 0.0

    def __call__(self, codes):
        schluessel = kanonisch(codes).astype(np.int8).tobytes()
        if schluessel in self.cache:
            return self.cache[schluessel]
        t0 = time.time()
        r = ridge_loo(born(self.x, unitaer_clifford(kanonisch(codes), self.n)), self.y)
        self.cache[schluessel] = r
        self.bewertungen += 1
        self.sekunden += time.time() - t0
        return r


def _besser(a, b):
    return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])


def suche(bew, start, rng, skala, schritte=SA_SCHRITTE, tau_start=TAU_START, tau_ende=TAU_ENDE,
          politur_max=POLITUR_MAX):
    """Simulated annealing auf den kanonischen Bits (je Schritt EIN Gatter Pauli- <->
    Hadamard-artig; der neutrale Zug c -> c+2 entfaellt) + gierige Politur.
    Temperatur T = tau * skala, skala = Streuung (sd) des Kriteriums ueber die
    Zufalls-Clifford-Schaltungen DIESES Seeds (Grenzen relativ zum Effekt).
    -> (beste Codes, bester Punktwert, Kurve [(neue Bewertungen, beste LOO-acc)])."""
    b0 = bew.bewertungen
    cur = kanonisch(start).copy()
    L, n = cur.shape
    s_cur = bew(cur)
    best, s_best = cur.copy(), s_cur
    kurve = [(0, s_best[0])]
    t0, t1 = tau_start * skala, tau_ende * skala
    for i in range(schritte):
        T = t0 * (t1 / t0) ** (i / max(1, schritte - 1)) if t0 > 0 else 0.0
        kand = cur.copy()
        kand[int(rng.integers(L)), int(rng.integers(n))] ^= 1
        s = bew(kand)
        d = s[0] - s_cur[0]
        if d >= 0 or (T > 0 and rng.random() < math.exp(d / T)):
            cur, s_cur = kand, s
        if _besser(s, s_best):
            best, s_best = kand.copy(), s
            kurve.append((bew.bewertungen - b0, s_best[0]))
    # gierige Politur ab der besten Wahl: alle L*n Einzel-Flips, erster Gewinn zaehlt
    zuege, verbessert, start_pol = L * n, True, bew.bewertungen
    while verbessert and bew.bewertungen - start_pol < politur_max:
        verbessert = False
        for j in rng.permutation(zuege):
            if bew.bewertungen - start_pol >= politur_max:
                break
            kand = best.copy()
            kand[j // n, j % n] ^= 1
            s = bew(kand)
            if _besser(s, s_best):
                best, s_best, verbessert = kand, s, True
                kurve.append((bew.bewertungen - b0, s_best[0]))
    kurve.append((bew.bewertungen - b0, s_best[0]))
    return best, s_best, kurve


def zufallssuche(bew, rng, n_kand, L, n):
    """Beste von n_kand NEUEN zufaelligen Wahlen nach demselben Kriterium."""
    best, s_best, b0 = None, None, bew.bewertungen
    for _ in range(50 * n_kand):
        if bew.bewertungen - b0 >= n_kand:
            break
        c = kanonisch(rng.integers(0, 4, (L, n)))
        s = bew(c)
        if s_best is None or _besser(s, s_best):
            best, s_best = c, s
    return best, s_best


def zeilen_kanonisch(U, tol=1e-5):
    """Zeilen von U bis auf Vorzeichen, sortiert -> Vergleich 'gleiche Born-Merkmale bis auf Permutation'."""
    U = U.double().numpy()
    out = []
    for r in U:
        nz = np.flatnonzero(np.abs(r) > tol)
        r = r * (np.sign(r[nz[0]]) if len(nz) else 1.0)
        out.append(tuple(np.round(r / tol).astype(np.int64)))
    return sorted(out)


def aequivalent(U1, U2):
    return zeilen_kanonisch(U1) == zeilen_kanonisch(U2)


def kurve_an_punkten(kurve, punkte=KURVE_PUNKTE):
    """Beste LOO-Genauigkeit nach p neuen Bewertungen (Treppenfunktion)."""
    out, ende = {}, max(b for b, _ in kurve)
    for p in punkte:
        vor = [v for b, v in kurve if b <= p]
        out[str(p)] = vor[-1] if vor and p <= ende else None
    return out


# ------------------------------------------------------------------ Arbeiter
def _bewerte(codes_oder_U, n, xs):
    """xs = (xtr, ytr, xte, yte, l2), x normiert -> logistischer Kopf, Test."""
    xtr, ytr, xte, yte, l2 = xs
    U = codes_oder_U if isinstance(codes_oder_U, torch.Tensor) else unitaer_clifford(codes_oder_U, n)
    return kopf_logistisch(born(xtr, U), ytr, born(xte, U), yte, l2=l2)


def zufall_codes(seed, anzahl, n):
    g = np.random.default_rng(ZUFALL_SEED + seed)
    return [g.integers(0, 4, (LAGEN, n)) for _ in range(anzahl)]


def _gemeinsame_arme(s, n, teile, schritte, politur_max, n_zufall, l2):
    """Zufalls-Clifford, Suche (ab Zufallsstart), Identitaet. Kriterium und Kopf sehen
    NUR die Nicht-Test-Daten (xtr, ytr); der Test geht nur in _bewerte."""
    xtr, ytr, xte, yte = teile
    bew = Bewerter(xtr, ytr, n)
    xs = (normiere(xtr), ytr, normiere(xte), yte, l2)
    z = {"seed": s, "n_train": len(ytr), "n_test": len(yte)}
    zuf = zufall_codes(s, n_zufall, n)
    z["zufall_val"] = [bew(c)[0] for c in zuf]
    z["zufall_test"] = [_bewerte(c, n, xs)[0] for c in zuf]
    z["zufall_median"] = float(np.median(z["zufall_test"]))
    z["zufall_best100"] = z["zufall_test"][int(np.argmax(z["zufall_val"]))]
    z["skala"] = skala = max(float(np.std(z["zufall_val"])), 1e-4)
    t0, b0 = time.time(), bew.bewertungen
    start = np.random.default_rng(START_SEED + s).integers(0, 4, (LAGEN, n))
    best, s_best, kurve = suche(bew, start, np.random.default_rng(SA_SEED + s), skala, schritte,
                                politur_max=politur_max)
    z["suche_sekunden"], z["suche_bewertungen"] = time.time() - t0, bew.bewertungen - b0
    z["suche_codes"], z["suche_val"] = best.tolist(), s_best[0]
    z["suche_kurve"] = kurve_an_punkten(kurve)
    z["suche"], konf, richtig = _bewerte(best, n, xs)
    z["suche_kaskade"] = kaskaden_kennzahl(konf, richtig)
    null = np.zeros((LAGEN, n), np.int64)
    z["identitaet"], z["identitaet_val"] = _bewerte(null, n, xs)[0], bew(null)[0]
    return z, bew, xs, best


def _mnist_seed(a):
    """(seed, schritte, politur_max, n_zufall, schritte_voll, zufallssuche?) -> Zeile."""
    from abteilungen.clifford import _baue, daten
    from labor.gemeinsam import trainiere
    s, schritte, politur_max, n_zufall, schritte_voll, mit_zsuche = a
    n, t0 = N_MNIST, time.time()
    xtr, ytr, xte, yte = daten("mnist", s)            # 6.000 Nicht-Test / 5.000 Test (wie C-mnist)
    z, bew, xs, best = _gemeinsame_arme(s, n, (xtr, ytr, xte, yte), schritte, politur_max, n_zufall,
                                     KOPF_L2["mnist"])
    # k0 = volles Modell per Gradient (wie C-mnist: Arm voll, lr 0,08), gerundet, DERSELBE Kopf
    tv = time.time()
    m, _ = _baue("voll", s, n)
    trainiere(m, xtr, ytr, schritte_voll, LR_VOLL, seed=s)
    z["voll_sekunden"] = time.time() - tv
    with torch.no_grad():
        z["voll_eigener_kopf"] = float((m(xte).argmax(1) == yte).float().mean())
    th = m.theta.detach().clone()
    z["winkel_voll"] = th.flatten().tolist()
    z["voll"] = _bewerte(unitaer_winkel(th, n), n, xs)[0]
    k0 = kanonisch(codes_aus_winkeln(runde_clifford(th)))
    z["k0_codes"] = k0.tolist()
    z["k0"], konf, richtig = _bewerte(k0, n, xs)
    z["k0_kaskade"] = kaskaden_kennzahl(konf, richtig)
    z["k0_val"] = bew(k0)[0]
    # beschreibend: gierige Politur ab k0 (ohne SA -- eine heisse SA liefe von k0 weg)
    b0 = bew.bewertungen
    b2, s2, _ = suche(bew, k0, np.random.default_rng(SA_SEED + 1_000 + s), z["skala"], 0,
                      politur_max=POLITUR_AB_K0 if schritte >= 1000 else politur_max)
    z["suche_ab_k0"], z["suche_ab_k0_val"] = _bewerte(b2, n, xs)[0], s2[0]
    z["suche_ab_k0_bewertungen"] = bew.bewertungen - b0
    # Zufallssuche mit genau so vielen NEUEN Bewertungen wie die Hauptsuche (fuer Q-4)
    if mit_zsuche:
        bz, sz = zufallssuche(bew, np.random.default_rng(ZSUCHE_SEED + s), z["suche_bewertungen"], LAGEN, n)
        z["zufallssuche"], z["zufallssuche_val"] = _bewerte(bz, n, xs)[0], sz[0]
    z["sekunden"] = time.time() - t0
    return z


def _struktur_seed(a):
    """(seed, schritte, politur_max, n_zufall) -> Zeile der Positivkontrolle."""
    from abteilungen.magie_aufgabe import _lehrer, aufgabe_daten
    s, schritte, politur_max, n_zufall = a
    n, t0 = N_STRUKTUR, time.time()
    teile = aufgabe_daten("struktur", s)             # Lehrer mit Marge: ~820 Nicht-Test / ~350 Test
    z, bew, xs, best = _gemeinsame_arme(s, n, teile, schritte, politur_max, n_zufall, KOPF_L2["struktur"])
    lc = kanonisch(codes_aus_winkeln(_lehrer()[0]))
    z["lehrer"], z["lehrer_val"] = _bewerte(lc, n, xs)[0], bew(lc)[0]
    z["lehrer_aequivalent"] = bool(aequivalent(unitaer_clifford(best, n), unitaer_clifford(lc, n)))
    bz, sz = zufallssuche(bew, np.random.default_rng(ZSUCHE_SEED + s), z["suche_bewertungen"], LAGEN, n)
    z["zufallssuche"], z["zufallssuche_val"] = _bewerte(bz, n, xs)[0], sz[0]
    z["zufallssuche_aequivalent"] = bool(aequivalent(unitaer_clifford(bz, n), unitaer_clifford(lc, n)))
    z["sekunden"] = time.time() - t0
    return z


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds_mnist=SEEDS_MNIST, seeds_struktur=SEEDS_STRUKTUR, schritte=SA_SCHRITTE,
         politur_max=POLITUR_MAX, n_zufall=N_ZUFALL, schritte_voll=SCHRITTE_VOLL, zufallssuche_arm=True):
    if schnell:
        seeds_mnist, seeds_struktur = (970, 971), (980, 981)      # nur Entwurfs-Seeds
        schritte, politur_max, n_zufall, schritte_voll = 80, 20, 5, 40
    t0 = time.time()
    seeds_mnist, seeds_struktur = list(seeds_mnist), list(seeds_struktur)
    # MNIST-Seeds (lang) zuerst verteilen, Strukturseeds (kurz) fuellen die Luecken
    aufg = ([("mnist", (s, schritte, politur_max, n_zufall, schritte_voll, zufallssuche_arm)) for s in seeds_mnist]
            + [("struktur", (s, schritte, politur_max, n_zufall)) for s in seeds_struktur])
    erg = parallel(_arbeiter, aufg)
    zs = sorted((z for art, z in erg if art == "struktur"), key=lambda z: z["seed"])
    zm = sorted((z for art, z in erg if art == "mnist"), key=lambda z: z["seed"])
    for z in zs:
        print(f"  Struktur {z['seed']}: suche {z['suche']:.4f} lehrer {z['lehrer']:.4f} id {z['identitaet']:.4f} "
              f"zufall-median {z['zufall_median']:.4f} aequivalent {z['lehrer_aequivalent']} "
              f"({z['suche_bewertungen']} Bew.)", flush=True)
    for z in zm:
        print(f"  MNIST {z['seed']}: suche {z['suche']:.4f} k0 {z['k0']:.4f} zufall-median {z['zufall_median']:.4f} "
              f"id {z['identitaet']:.4f} voll {z['voll']:.4f} ({z['suche_bewertungen']} Bew., {z['sekunden']:.0f} s)",
              flush=True)
    e = {"abteilung": "Clifford", "modul": "clifford_suche", "karte": "I-005", "version": 1,
         "hypothesen": HYPOTHESEN, "seeds_mnist": seeds_mnist, "seeds_struktur": seeds_struktur,
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "tau_start": TAU_START, "tau_ende": TAU_ENDE, "ridge_lambda": RIDGE_LAMBDA,
                           "kopf_l2": KOPF_L2, "lr_voll": LR_VOLL, "schritte_voll": schritte_voll, "tost": TOST,
                           "konf_kaskade": KONF_KASKADE},
         "zeilen_struktur": zs, "zeilen_mnist": zm, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zm, zs)
    e["urteil"] = urteil(e["auswertung"])
    return e


def _arbeiter(a):
    art, arg = a
    return art, (_struktur_seed(arg) if art == "struktur" else _mnist_seed(arg))


# ------------------------------------------------------------------ Auswertung
def _gepaart(d, einseitig_q=0.95):
    """Mittel, exakter zweiseitiger Wilcoxon, t-KI (0,95 einseitig = 90-%-KI zweiseitig, TOST)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    if len(d) < 2:
        return {"mittel": m, "p": 1.0, "ki_unten": -math.inf, "ki_oben": math.inf, "n": len(d)}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(einseitig_q, len(d) - 1))
    return {"mittel": m, "p": wilcoxon_exakt(d), "ki_unten": m - q * se, "ki_oben": m + q * se,
            "positiv": int((d > 0).sum()), "n": len(d)}


def holm(p: dict) -> dict:
    namen = sorted(p, key=p.get)
    out, lauf_max = {}, 0.0
    for i, k in enumerate(namen):
        lauf_max = max(lauf_max, min(1.0, (len(namen) - i) * p[k]))
        out[k] = lauf_max
    return out


def auswertung(zm, zs) -> dict:
    a = {"n_mnist": len(zm), "n_struktur": len(zs)}
    arr = lambda zeilen, k: np.array([z[k] for z in zeilen], dtype=float)
    # Q-0 Strukturaufgabe
    if zs:
        su, le, id_ = arr(zs, "suche"), arr(zs, "lehrer"), arr(zs, "identitaet")
        gewinn = float((le - id_).mean())
        q0 = {"suche": zusammenfassung(su), "lehrer": zusammenfassung(le), "identitaet": zusammenfassung(id_),
              "zufall_median": zusammenfassung(arr(zs, "zufall_median")),
              "zufall_best100": zusammenfassung(arr(zs, "zufall_best100")),
              "gewinn_lehrer": gewinn, "p_lehrer": wilcoxon_exakt(le - id_),
              "suche_minus_id": _gepaart(su - id_),
              "G_L": float((su - id_).mean()) / gewinn if gewinn > 0 else None,
              "aequivalent": int(sum(z["lehrer_aequivalent"] for z in zs)),
              "val_suche_minus_lehrer": float(np.mean([z["suche_val"] - z["lehrer_val"] for z in zs])),
              "zufallssuche": zusammenfassung(arr(zs, "zufallssuche")),
              "zufallssuche_aequivalent": int(sum(z["zufallssuche_aequivalent"] for z in zs)),
              "bewertungen": float(np.mean([z["suche_bewertungen"] for z in zs]))}
        q0["vorbedingung"] = bool(gewinn >= MIN_GEWINN_LEHRER and q0["p_lehrer"] < ALPHA)
        a["Q0"] = q0
    # Q-1 / Q-2 MNIST
    if zm:
        su, zmed, k0 = arr(zm, "suche"), arr(zm, "zufall_median"), arr(zm, "k0")
        id_, vo = arr(zm, "identitaet"), arr(zm, "voll")
        a["Q1"] = _gepaart(su - zmed)
        a["Q2"] = _gepaart(su - k0)
        fam = {"Q-1": a["Q1"]["p"], "Q-2": a["Q2"]["p"]}
        if "zufallssuche" in zm[0]:
            a["Q4"] = _gepaart(su - arr(zm, "zufallssuche"))
            fam["Q-4"] = a["Q4"]["p"]
        a["holm"] = holm(fam)
        besch = {arm: zusammenfassung(arr(zm, arm)) for arm in
                 ("suche", "zufall_median", "zufall_best100", "k0", "identitaet", "voll", "voll_eigener_kopf",
                  "suche_ab_k0", "zufallssuche") if arm in zm[0]}
        besch["G_voll_suche"] = (float((su - id_).mean()) / float((vo - id_).mean())
                                 if float((vo - id_).mean()) > 0 else None)
        besch["G_voll_k0"] = (float((k0 - id_).mean()) / float((vo - id_).mean())
                              if float((vo - id_).mean()) > 0 else None)
        besch["suche_minus_best100"] = _gepaart(su - arr(zm, "zufall_best100"))
        if "suche_ab_k0" in zm[0]:
            besch["suche_ab_k0_minus_k0"] = _gepaart(arr(zm, "suche_ab_k0") - k0)
        # Siegerfluch: Vorsprung der Suche vor dem Zufallsmedian im KRITERIUM minus im TEST
        # (Differenz der Differenzen -- Kriterium und Bewertungskopf sind verschieden skaliert)
        besch["siegerfluch"] = float(np.mean([(z["suche_val"] - float(np.median(z["zufall_val"])))
                                              - (z["suche"] - z["zufall_median"]) for z in zm]))
        besch["kurve_val"] = {p: float(np.mean([z["suche_kurve"][p] for z in zm]))
                              for p in zm[0]["suche_kurve"] if all(z["suche_kurve"][p] is not None for z in zm)}
        besch["bewertungen"] = float(np.mean([z["suche_bewertungen"] for z in zm]))
        besch["sekunden_suche"] = float(np.mean([z["suche_sekunden"] for z in zm]))
        besch["bewertungen_ab_k0"] = float(np.mean([z["suche_ab_k0_bewertungen"] for z in zm]))
        besch["sekunden_voll"] = float(np.mean([z["voll_sekunden"] for z in zm]))
        besch["sekunden_seed"] = float(np.mean([z["sekunden"] for z in zm]))
        for arm in ("suche", "k0"):
            besch[f"{arm}_anteil_sicher"] = float(np.mean([z[f"{arm}_kaskade"]["anteil_sicher"] for z in zm]))
            besch[f"{arm}_acc_sicher"] = float(np.mean([z[f"{arm}_kaskade"]["acc_sicher"] or 0.0 for z in zm]))
        a["Q3"] = besch
    return a


def _vergleich(d, p_holm, q0_h1, name_gegen, zweck):
    """Gemeinsame Verzweigung fuer Q-1/Q-2 (siehe Kopfkommentar)."""
    t = (f"d = {d['mittel'] * 100:+.2f} pp, 90-%-KI [{d['ki_unten'] * 100:+.2f}; {d['ki_oben'] * 100:+.2f}], "
         f"Wilcoxon p = {d['p']:.4f}, Holm-p = {p_holm:.4f}, {d.get('positiv', 0)}/{d['n']} Seeds positiv")
    if p_holm < ALPHA and d["mittel"] > 0:
        return f"H1: Suche besser als {name_gegen} -- " + t
    if p_holm < ALPHA and d["mittel"] < 0:
        return f"H0 (Gegenrichtung): {name_gegen} besser als die Suche -- " + t
    if -TOST < d["ki_unten"] and d["ki_oben"] < TOST:
        if q0_h1:
            return f"H0 (gleich): Suche = {name_gegen} im Rahmen +-{TOST * 100:.1f} pp (TOST){zweck} -- " + t
        return f"HU (Positivkontrolle): gleich im Rahmen +-{TOST * 100:.1f} pp, aber Q-0 nicht H1 -- " + t
    return "HU: " + t


def urteil(a: dict) -> dict:
    u = {}
    q0 = a.get("Q0")
    if q0 is None:
        u["Q-0"] = "HU (Vorbedingung): keine Strukturseeds"
    elif not q0["vorbedingung"]:
        u["Q-0"] = (f"HU (Vorbedingung): Lehrer schlaegt Identitaet nicht nachweisbar "
                    f"(Gewinn {q0['gewinn_lehrer'] * 100:+.2f} pp, p = {q0['p_lehrer']:.4f})")
    else:
        d = q0["suche_minus_id"]
        t = (f"G_L = {q0['G_L']:.2f} (suche {q0['suche']['mittel']:.4f}, lehrer {q0['lehrer']['mittel']:.4f}, "
             f"id {q0['identitaet']['mittel']:.4f}), suche - id {d['mittel'] * 100:+.2f} pp, p = {d['p']:.4f}, "
             f"lehrer-aequivalent in {q0['aequivalent']}/{a['n_struktur']} Seeds (Zufallssuche gleichen "
             f"Aufwands: {q0['zufallssuche_aequivalent']}), {q0['bewertungen']:.0f} Bewertungen")
        if q0["G_L"] >= G_H1 and d["mittel"] > 0 and d["p"] < ALPHA:
            u["Q-0"] = "H1: Positivkontrolle bestanden -- die diskrete Suche findet die Lehrer-Loesung; " + t
        elif q0["G_L"] < G_H0:
            u["Q-0"] = "H0: Suche findet die bekannte Clifford-Loesung nicht; " + t
        else:
            u["Q-0"] = "HU: " + t
    q0_h1 = u["Q-0"].startswith("H1")
    if "Q1" in a:
        u["Q-1"] = _vergleich(a["Q1"], a["holm"]["Q-1"], q0_h1, "Median von Zufalls-Clifford",
                              " -- Karte I-005 abbrechen")
        u["Q-2"] = _vergleich(a["Q2"], a["holm"]["Q-2"], q0_h1, "k0 (gerundetes Modell)",
                              " -- Suche ersetzt das Runden, ohne Gradient")
        if "Q4" in a:
            u["Q-4"] = _vergleich(a["Q4"], a["holm"]["Q-4"], q0_h1, "Zufallssuche gleichen Aufwands",
                                  " -- der Nutzen ist reine Auswahl, nicht Suche")
        b = a["Q3"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        teile = [f"Test: suche {b['suche']['mittel']:.4f}, k0 {b['k0']['mittel']:.4f}, "
                 f"zufall-median {b['zufall_median']['mittel']:.4f}, best100 {b['zufall_best100']['mittel']:.4f}, "
                 f"id {b['identitaet']['mittel']:.4f}, voll {b['voll']['mittel']:.4f}",
                 f"G_voll suche {fmt(b['G_voll_suche'])}, k0 {fmt(b['G_voll_k0'])}"]
        if "zufallssuche" in b:
            teile.append(f"Zufallssuche gleichen Aufwands {b['zufallssuche']['mittel']:.4f}")
        if "suche_ab_k0" in b:
            teile.append(f"Politur ab k0 {b['suche_ab_k0']['mittel']:.4f} "
                         f"({b['suche_ab_k0_minus_k0']['mittel'] * 100:+.2f} pp ueber k0)")
        teile.append("LOO-Kurve (neue Bewertungen: bestes Kriterium) " + ", ".join(f"{k}: {v:.4f}" for k, v in b["kurve_val"].items()))
        teile.append(f"Siegerfluch (Vorsprung vor Zufallsmedian: Kriterium - Test) {b['siegerfluch'] * 100:+.2f} pp")
        teile.append(f"Aufwand {b['bewertungen']:.0f} Bewertungen (ab k0 {b['bewertungen_ab_k0']:.0f}), Suche {b['sekunden_suche']:.0f} s, "
                     f"Gradiententraining voll {b['sekunden_voll']:.0f} s je Seed")
        teile.append(f"Kaskade: konf > {KONF_KASKADE}: suche {b['suche_anteil_sicher'] * 100:.1f} % der Bilder "
                     f"(acc {b['suche_acc_sicher']:.4f}), k0 {b['k0_anteil_sicher'] * 100:.1f} % "
                     f"(acc {b['k0_acc_sicher']:.4f})")
        u["Q-3"] = "beschreibend: " + "; ".join(teile)
    else:
        u["Q-1"] = u["Q-2"] = u["Q-4"] = "HU: keine MNIST-Seeds"
        u["Q-3"] = "beschreibend: keine MNIST-Seeds"
    return u


# ------------------------------------------------------------------ Vortest
def _synth_mnist(rng, d_med, d_k0, d_zs, sd=0.001, n=10):
    zeilen = []
    for s in range(n):
        med = 0.905 + rng.normal(0, 0.003)
        su = med + d_med + rng.normal(0, sd)
        zeilen.append({
            "seed": s, "suche": su, "zufall_median": med, "zufall_best100": med + 0.004, "k0": su - d_k0 + rng.normal(0, sd),
            "zufallssuche": su - d_zs + rng.normal(0, sd), "identitaet": 0.90, "voll": 0.938, "voll_eigener_kopf": 0.938,
            "suche_ab_k0": su, "suche_val": su + 0.003, "zufall_val": [med] * 3, "zufall_test": [med] * 3,
            "suche_kurve": {str(p): med + 0.01 for p in KURVE_PUNKTE}, "suche_bewertungen": 1000,
            "suche_ab_k0_bewertungen": 900, "suche_sekunden": 60.0, "voll_sekunden": 90.0, "sekunden": 300.0,
            "suche_kaskade": {"anteil_sicher": 0.8, "acc_sicher": 0.97}, "k0_kaskade": {"anteil_sicher": 0.78, "acc_sicher": 0.97}})
    return zeilen


def _synth_struktur(rng, suche, lehrer=0.93, ident=0.78, sd=0.01, n=10):
    return [{"seed": s, "suche": suche + rng.normal(0, sd), "lehrer": lehrer + rng.normal(0, sd),
             "identitaet": ident + rng.normal(0, sd), "zufall_median": ident, "zufall_best100": ident + 0.02,
             "lehrer_aequivalent": suche > 0.9, "zufallssuche_aequivalent": False, "suche_val": suche,
             "lehrer_val": lehrer, "zufallssuche": ident + 0.03, "suche_bewertungen": 1000} for s in range(n)]


def _vortest_urteile(ziehungen=5) -> bool:
    ok = True
    rng = np.random.default_rng(5)
    gut = lambda: _synth_struktur(rng, 0.925)
    faelle = [
        # (struktur, mnist(d_med, d_k0, d_zs, sd), erwartet, toleranz)
        (gut, (0.015, 0.012, 0.008, 0.001), {"Q-0": "H1", "Q-1": "H1", "Q-2": "H1", "Q-4": "H1"}, 0),
        (gut, (0.0, 0.0, 0.0, 0.001), {"Q-0": "H1", "Q-1": "H0 (gleich)", "Q-2": "H0 (gleich)", "Q-4": "H0 (gleich)"}, 1),
        (gut, (-0.01, -0.01, -0.01, 0.001), {"Q-1": "H0 (Gegenrichtung)", "Q-2": "H0 (Gegenrichtung)",
                                              "Q-4": "H0 (Gegenrichtung)"}, 0),
        (lambda: _synth_struktur(rng, 0.79), (0.0, 0.0, 0.0, 0.001),
         {"Q-0": "H0", "Q-1": "HU (Positivkontrolle)", "Q-2": "HU (Positivkontrolle)"}, 1),
        (lambda: _synth_struktur(rng, 0.88), (0.015, 0.0, 0.0, 0.001), {"Q-0": "HU:", "Q-1": "H1"}, 1),
        (lambda: _synth_struktur(rng, 0.80, lehrer=0.78), (0.003, 0.003, 0.003, 0.01),
         {"Q-0": "HU (Vorbedingung)", "Q-1": "HU:", "Q-2": "HU:"}, 1),
    ]
    for mach_s, (dm, dk, dz, sd), erwartet, tol in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth_mnist(rng, dm, dk, dz, sd), mach_s()))
            for h, pr in erwartet.items():
                if not u[h].startswith(pr):
                    fehl[h] += 1
                    print(f"  Vortest: {h} -> {u[h][:110]}")
            if not u["Q-3"].startswith("beschreibend"):
                fehl.setdefault("Q-3", 0)
                fehl["Q-3"] += 1
            if not all(isinstance(v, str) for v in u.values()):
                ok = False
                print("  VORTEST FEHLER: Urteil nicht als String")
        for h, f in fehl.items():
            if f > tol:
                ok = False
                print(f"  VORTEST FEHLER: {h} {f}/{ziehungen} falsch bei {erwartet}")
    # Aufloesung: grosse Streuung, wahre Differenz 0 -> nie H0 (gleich)
    for _ in range(ziehungen):
        u = urteil(auswertung(_synth_mnist(rng, 0.0, 0.0, 0.0, 0.02), gut()))
        if u["Q-1"].startswith("H0 (gleich)"):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung", u["Q-1"][:100])
    # Holm
    h = holm({"a": 0.01, "b": 0.04})
    h2 = holm({"a": 0.03, "b": 0.02, "c": 0.5})
    if not (abs(h["a"] - 0.02) < 1e-12 and abs(h["b"] - 0.04) < 1e-12 and abs(h2["b"] - 0.06) < 1e-12
            and abs(h2["a"] - 0.06) < 1e-12 and abs(h2["c"] - 0.5) < 1e-12):
        ok = False
        print("  VORTEST FEHLER: Holm", h, h2)
    return ok


def _mini_daten(seed, n=3, anzahl=700, marge=0.3):
    """Kuenstliche Mini-Aufgabe mit bekannter Clifford-Loesung (n = 3, L = 4 -> 4^12 Wahlen,
    2^12 kanonische): Etikett = argmax eines festen linearen Lehrers auf den Born-
    Merkmalen einer festen Clifford-Wahl, nur Bilder mit Marge."""
    g = np.random.default_rng(seed)
    lehrer = g.integers(0, 4, (LAGEN, n))
    x = torch.tensor(g.random((anzahl, 2 ** n)) ** 3, dtype=torch.float32)
    W = torch.tensor(g.normal(size=(10, 2 ** n)), dtype=torch.float32)
    f = born(normiere(x), unitaer_clifford(lehrer, n))
    pkt = f @ W.T
    pkt = (pkt - pkt.mean(0)) / pkt.std(0)
    srt = pkt.sort(1, descending=True).values
    behalte = (srt[:, 0] - srt[:, 1]) >= marge
    x, y = x[behalte], pkt[behalte].argmax(1)
    nt = len(y) * 2 // 3
    return lehrer, (x[:nt], y[:nt], x[nt:], y[nt:])


def _vortest_bausteine() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(11)
    n = 4
    x = torch.tensor(rng.random((200, 2 ** n)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 10, 200))
    bew = Bewerter(x, y, n)
    # (a) c und c+2 gleichwertig (Pauli wandert ans Ende): gleiche Zeilen bis auf Vorzeichen/Reihenfolge
    for _ in range(10):
        c = rng.integers(0, 4, (LAGEN, n))
        c2 = c.copy()
        c2[rng.integers(LAGEN), rng.integers(n)] += 2
        U1, U2 = unitaer_clifford(c, n), unitaer_clifford(c2 % 4, n)
        f1, f2 = born(normiere(x), U1), born(normiere(x), U2)
        if not aequivalent(U1, U2) or not torch.allclose(f1.sort(1).values, f2.sort(1).values, atol=1e-6):
            fehler("c und c+2 nicht gleichwertig")
            break
        if abs(ridge_loo(f1, y)[0] - ridge_loo(f2, y)[0]) > 1e-9:
            fehler("LOO-Kriterium nicht permutationsinvariant")
            break
    # (b) U orthogonal, Eintraege nur 0 oder +-2^(-h/2) (Clifford: Additionen + globaler Faktor)
    for _ in range(5):
        U = unitaer_clifford(rng.integers(0, 4, (LAGEN, n)), n).double()
        if not torch.allclose(U @ U.T, torch.eye(2 ** n, dtype=torch.float64), atol=1e-5):
            fehler("U nicht orthogonal")
        nz = U.abs()[U.abs() > 1e-6]
        h = -2 * torch.log2(nz)
        if not torch.allclose(h, torch.round(h), atol=1e-4):
            fehler("U hat Nicht-Clifford-Eintraege")
    # (c) Identitaet = CNOT-Ring-Permutation: Born = permutierte x^2
    f0 = born(normiere(x), unitaer_clifford(np.zeros((LAGEN, n), np.int64), n))
    if not torch.allclose(f0.sort(1).values, (normiere(x) ** 2).sort(1).values, atol=1e-6):
        fehler("Identitaet ist keine Permutation")
    # (d) LOO ueber die Hutmatrix = explizites Weglassen
    f = born(normiere(x[:40]), unitaer_clifford(rng.integers(0, 4, (LAGEN, n)), n))
    yy = y[:40]
    acc_h, mse_h = ridge_loo(f, yy, lam=0.7)
    mu, sd = _standard(f)
    A = _mit_eins((f - mu) / sd).double()
    Y = torch.nn.functional.one_hot(yy, 10).double()
    reg = 0.7 * torch.eye(A.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0
    treffer, mse = 0, 0.0
    for i in range(len(yy)):
        m = torch.ones(len(yy), dtype=torch.bool)
        m[i] = False
        W = torch.linalg.solve(A[m].T @ A[m] + reg, A[m].T @ Y[m])
        p = A[i] @ W
        treffer += int(p.argmax() == yy[i])
        mse += float(((p - Y[i]) ** 2).sum())
    if abs(treffer / len(yy) - acc_h) > 1e-6 or abs(mse / len(yy) - mse_h) > 1e-4 * max(1.0, mse_h):
        fehler(f"LOO-Formel falsch ({acc_h} vs {treffer / len(yy)}, {mse_h} vs {mse / len(yy)})")
    # (e) gerundete Winkel -> Codes mod 4 liefern exakt die Born-Merkmale der gerundeten Zelle
    th = torch.rand(LAGEN, n) * 4 * math.pi - 2 * math.pi
    fa = born(normiere(x), unitaer_winkel(runde_clifford(th), n))
    fb = born(normiere(x), unitaer_clifford(kanonisch(codes_aus_winkeln(runde_clifford(th))), n))
    if not torch.allclose(fa.sort(1).values, fb.sort(1).values, atol=1e-5):
        fehler("Codes aus gerundeten Winkeln stimmen nicht")
    # (f) Kopf deterministisch und auf trennbaren Daten fast perfekt
    xs = torch.randn(300, 5)
    ys = (xs[:, 0] > 0).long() + 2 * (xs[:, 1] > 0).long()
    a1, _, _ = kopf_logistisch(xs[:200], ys[:200], xs[200:], ys[200:])
    a2, _, _ = kopf_logistisch(xs[:200], ys[:200], xs[200:], ys[200:])
    if a1 != a2 or a1 < 0.9:
        fehler(f"logistischer Kopf ({a1}, {a2})")
    # (g) Mini-Aufgabe: die Suche findet die bekannte Clifford-Wahl; erschoepfend geprueft
    gefunden = 0
    for sm in (1, 2, 3):
        lehrer, (xtr, ytr, xte, yte) = _mini_daten(sm)
        b = Bewerter(xtr, ytr, 3)
        alle = [np.array([(k >> j) & 1 for j in range(LAGEN * 3)]).reshape(LAGEN, 3) for k in range(2 ** 12)]
        punkte = [b(c) for c in alle]
        bestes = alle[max(range(len(alle)), key=lambda i: (punkte[i][0], -punkte[i][1]))]
        if not aequivalent(unitaer_clifford(bestes, 3), unitaer_clifford(lehrer, 3)):
            fehler(f"Mini-Aufgabe {sm}: erschoepfendes Optimum ist nicht der Lehrer")
        b2 = Bewerter(xtr, ytr, 3)
        sk = float(np.std([b2(c)[0] for c in zufall_codes(sm, 50, 3)]))
        best, _, _ = suche(b2, np.random.default_rng(sm).integers(0, 4, (LAGEN, 3)), np.random.default_rng(sm + 7),
                           sk, 1500)
        gefunden += aequivalent(unitaer_clifford(best, 3), unitaer_clifford(lehrer, 3))
    if gefunden < 2:
        fehler(f"Suche findet den Lehrer der Mini-Aufgabe nur {gefunden}/3 mal")
    # (h) Test strikt getrennt: andere Test-Etiketten -> identische Suche und Auswahl
    lehrer, (xtr, ytr, xte, yte) = _mini_daten(4)
    z1, *_ = _gemeinsame_arme(4, 3, (xtr, ytr, xte, yte), 300, 50, 5, 1e-3)
    z2, *_ = _gemeinsame_arme(4, 3, (xtr, ytr, xte, yte[torch.randperm(len(yte))]), 300, 50, 5, 1e-3)
    if z1["suche_codes"] != z2["suche_codes"] or z1["zufall_val"] != z2["zufall_val"] or z1["suche_val"] != z2["suche_val"]:
        fehler("Suche/Auswahl haengt vom Test ab")
    # (i) Zufallssuche macht genau die verlangte Zahl neuer Bewertungen
    b3 = Bewerter(xtr, ytr, 3)
    b3(np.zeros((LAGEN, 3), np.int64))
    k = b3.bewertungen
    zufallssuche(b3, np.random.default_rng(0), 40, LAGEN, 3)
    if b3.bewertungen - k != 40:
        fehler(f"Zufallssuche: {b3.bewertungen - k} statt 40 Bewertungen")
    # (j) Strukturaufgabe: Lehrer ist Clifford, keine Permutation, und schlaegt die Identitaet im Kriterium
    from abteilungen.magie_aufgabe import _lehrer, aufgabe_daten
    lc = kanonisch(codes_aus_winkeln(_lehrer()[0]))
    Ul = unitaer_clifford(lc, N_STRUKTUR)
    if float((Ul.abs() > 1e-6).sum(1).float().mean()) <= 1.0:
        fehler("Lehrer ist eine Permutation")
    xtr, ytr, _, _ = aufgabe_daten("struktur", 0)
    bs = Bewerter(xtr, ytr, N_STRUKTUR)
    if not bs(lc)[0] > bs(np.zeros((LAGEN, N_STRUKTUR), np.int64))[0] + 0.05:
        fehler("Lehrer im Kriterium nicht klar besser als Identitaet")
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Bausteine (Clifford-Gleichwertigkeit, LOO-Formel, Kopf, Mini-Aufgabe
    mit bekannter Loesung, Testtrennung) + jeden Urteilszweig synthetisch erzwingen."""
    import json
    ok = _vortest_bausteine()
    ok = _vortest_urteile(ziehungen) and ok
    rng = np.random.default_rng(1)
    a = auswertung(_synth_mnist(rng, 0.01, 0.01, 0.01), _synth_struktur(rng, 0.92))
    try:
        json.dumps({"auswertung": a, "urteil": urteil(a)})
    except TypeError as err:
        ok = False
        print("  VORTEST FEHLER: nicht JSON-serialisierbar", err)
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-005" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
