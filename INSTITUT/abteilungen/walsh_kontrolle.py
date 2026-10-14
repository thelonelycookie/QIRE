"""Abteilung Clifford -- Ist die gesuchte Clifford-Zelle ein klassischer Walsh-Klassifikator?
(Karte I-036, Quantenstufe Q2: exakte Simulation; Kern der Frage ist klassisch.)

HEILMEIER:
  Was wollen wir?  Pruefen, ob die diskrete Clifford-Suche (I-005/I-005-v2, 0 Magiewinkel) im Kern
      nur einen FESTEN klassischen Walsh-Hadamard-Klassifikator nachbaut: x -> (H^8 x/|x|)^2 -> derselbe
      L-BFGS-Kopf. Wenn ja, ist „0 T-Gatter, gleich gut wie voll“ (I-005-v2 V-1) ein Befund ueber eine
      bekannte klassische Transformation, nicht ueber die Suche.
  Wie heute?  Recherche 29.09. (clifford_klassifikatoren.md, Abschnitt 3.2 [EIGEN]) vermutet die
      Aequivalenz, ungeprueft. C-mnist hatte einen Arm hadamard_fest (= walsh_fest, siehe unten), aber mit
      Adam-Kopf und ohne Suche daneben: MNIST 0,883 < Identitaet 0,901; Fashion 0,807 > 0,796.
  Was ist neu?  (1) Exakte Beschreibung dessen, was die Zelle bei Clifford-Winkeln rechnet (unten,
      im Vortest geprueft). (2) Vorregistrierter gepaarter Vergleich Suche vs. feste Walsh-Merkmale mit
      demselben Kopf auf frischen Seeds, dazu eine Dosis-Kurve fester Block-Walsh-Transformationen.

THEORIE (vor dem Festlegen gerechnet; Vortest (T1)-(T6) prueft jeden Satz):
  T1  walsh_fest liegt im Suchraum: Codes Lage 0 = alle 1 (Ry(pi/2) auf jedem Qubit), Lagen 1-3 = 0.
      Ry(pi/2) = H·Z, also Ry(pi/2)^8 = H^8 Z^8; Z^8 ist ein Walsh-Charakter und permutiert nur die
      Zeilen von H^8; die CNOT-Ringe danach sind Basispermutationen. -> U hat bis auf Vorzeichen und
      Zeilenreihenfolge GENAU die Zeilen der Sylvester-Hadamard-Matrix H^8 (numerisch: aequivalent()).
      Die Born-Merkmale sind eine Permutation von (H^8 x)^2; der standardisierte Kopf gleicht das exakt
      aus. Es ist genau der Arm hadamard_fest aus clifford.py. Die kanonischen Bits (32 Bit, Lage 0 alle 1)
      sind ein Punkt des SA-Raums -> W-1 ist ERREICHBAR. Jede Lage allein mit allen 8 H ist ebenfalls
      aequivalent; H in zwei Lagen hebt sich auf (k = 0, Lehre „Hadamard bei gerader Lagenzahl“).
  T2  Was die Zelle allgemein rechnet (Bruhat-Zerlegung der Clifford-Gruppe; fuer reelle Clifford
      U = S1 · H_Q · S2 mit S1, S2 vorzeichenbehafteten affinen Permutationen mit quadratischer Phase):
        p = ( Blockweiser Walsh-Hadamard von (m ⊙ x) )^2  (bis auf Permutation der 256 Merkmale),
      d. h. es gibt einen Unterraum V ⊆ GF(2)^8 der Dimension k (= Zahl der wirksamen Hadamards nach
      Zusammenfassen), die 256 Pixel zerfallen in 2^(8-k) Nebenklassen („Bloecke“ zu 2^k Pixeln), jedes
      Pixel bekommt ein festes Vorzeichen m(y) = (-1)^(quadratische Form), und in jedem Block wird die
      Walsh-Transformation genommen und quadriert. Geprueft wird das je Schaltung (struktur()):
      alle Zeilen haben Traeger der Groesse 2^k, alle Traeger sind Nebenklassen DESSELBEN V, und je zwei
      Zeilen auf derselben Nebenklasse unterscheiden sich um einen affin-linearen Charakter
      (-> gemeinsame Maske m je Block). Das ist hinreichend fuer die Formel oben.
      Sonderfaelle: Identitaet k = 0; walsh_fest k = 8, m linear (= keine Maske).
  T3  Befund auf den VORHANDENEN Suchergebnissen (I-005 MNIST 10, I-005-v2 MNIST 20 + Fashion 20,
      I-005-v3 Fashion 20; nur gelesen, nichts neu gerechnet): 70/70 in Bruhat-Form, 0/70 Walsh-
      aequivalent, 69/70 mit echter (quadratischer) Vorzeichenmaske. Blockdimension k: MNIST 4-7
      (Mittel 5,4 bzw. 5,7; nie 8), Fashion 5-8 (Mittel 6,85 bzw. 6,6; k = 8 in 5/20 bzw. 3/20).
      -> Die Suche findet NICHT den globalen Walsh-Klassifikator, sondern Block-Walsh mit Maske.
         Auf MNIST kleinere (lokalere) Bloecke als auf Fashion (beschreibend, ungeprueft).
  FOLGERUNG FUER DAS DESIGN: Die Karte fragt, ob walsh_fest die Suche ERREICHT. Nach T3 und C-mnist ist
      SUCHE_BESSER auf MNIST das erwartete Ergebnis (Vorhersage vor dem Lauf, hier festgehalten);
      interessant ist, (a) ob das auch auf Fashion gilt, wo die Suche naeher an k = 8 liegt, und (b) wie
      weit FESTE Block-Walsh-Transformationen ohne Suche kommen (Dosis-Kurve, beschreibend). Die Familie
      bleibt wie in der Karte; GLEICHWERTIG heisst: die Suche leistet nicht mehr als feste Walsh-
      Merkmale (±0,5 pp) -- NICHT „sie findet dieselbe Schaltung“ (das misst W-S direkt: 0/70 bisher).

ARME (alle mit DEMSELBEN logistischen L-BFGS-Kopf aus clifford_suche, standardisierte Merkmale,
  trainiert auf 6.000 Nicht-Test-Bildern, gemessen auf 5.000 Testbildern; Daten clifford.daten 16x16):
  suche        SA + Politur, unveraendert aus I-005 (v1._gemeinsame_arme), 6.000 Schritte
  walsh_fest   Codes T1, OHNE Suche
  identitaet   alle Winkel 0 (Pixelquadrate)
  zufall_i     100 zufaellige Clifford (Median, best100 = beste nach LOO-Kriterium)
  beschreibend feste Block-Walsh ohne Suche (H nur in Lage 0 auf einer Qubit-Menge; Pixelindex =
               16·Zeile + Spalte, Qubits 0-3 = Zeilenbits, 4-7 = Spaltenbits, Qubit 0 = hoechstes Bit):
               block2x2 {3,7} k=2, zeilen16 {4,5,6,7} k=4 (1D-Walsh je Bildzeile), block4x4 {2,3,6,7}
               k=4, block8x8 {1,2,3,5,6,7} k=6, walsh_fest k=8. Dazu: LOO-Kriterium von walsh_fest vs.
               suche (hat die Suche Walsh uebersehen oder bewusst verworfen?), ob die Suche walsh_fest
               bewertet hat (Zwischenspeicher), struktur() der gefundenen Schaltung, L2-Empfindlichkeit
               (suche, walsh, identitaet mit L2 1e-3 und 1e-2 statt 3e-3).
  Weggelassen gegenueber I-005-v2: voll, k0, Zufallssuche (fuer W-1..W-4 ohne Belang; spart ~650 s je
  Seed und die ganze Ratensuche). Kopf-L2 = 0,003 fuer beide Datensaetze, uebernommen aus I-005-v2
  (dort nach fester Regel auf den Tuning-Seeds 950-955 gewaehlt, die hier nicht gemessen werden) --
  kein neues Tuning; der Einfluss wird beschreibend ueber die L2-Empfindlichkeit berichtet.

SEEDS (29.09. geprueft gegen ergebnisse/*.json, agenda.json, abteilungen/*.py, labor/*.py, direktor.py:
  frei im Block 637-697 und 942-949; _pruefe_seeds prueft zusaetzlich gegen die I-005-v2-Liste):
  Mess-Seeds  MNIST 640-659, Fashion 660-679 (je 20)     Strukturaufgabe (Q-0) 680-689
  Rauchtest   MNIST 942/943, Fashion 944/945, Struktur 946/947 (nie Mess-Seeds)
  Seed-Zahl 20, begruendet aus der Streuung in I-005-v2 (Unterschied suche - fester Arm, dort
  suche - identitaet): sd MNIST 0,67 pp, Fashion 0,35 pp (I-005: 0,63; I-005-v3: 0,40). TOST ±0,5 pp mit
  Holm ueber 3 (kleinstes p < 0,0167): n = 10 braucht |m| + 2,52·sd/3,16 < 0,5 -> MNIST unmoeglich
  (0,53 > 0,5), GLEICHWERTIG waere dort gar nicht erreichbar; n = 20: |m| < 0,5 - 2,31·0,67/4,47 = 0,15 pp
  (MNIST) bzw. 0,32 pp (Fashion). Unterschiedstests: kleinstes Wilcoxon-p 2/2^20, trivial erreichbar.
  NEU GERECHNET mit Holm x4 (W-4 aufgenommen, Direktor 29.09.): kleinstes p_TOST < 0,05/4 = 0,0125
  (t_0,9875;19 ~ 2,43), zweites < 0,0167 (t ~ 2,35), drittes < 0,025 (t ~ 2,21). Annahme sd = 0,67 pp
  (MNIST, fuer W-1 und W-4; die Streuung von suche - block2x2 ist unbekannt, sd(suche - fester Arm)
  ist die beste Naeherung) bzw. 0,35 pp (Fashion), n = 20, se = sd/4,47:
    MNIST als kleinstes p: |m| < 0,5 - 2,43·0,150 = 0,14 pp; Fashion: |m| < 0,5 - 2,43·0,078 = 0,31 pp.
    Macht fuer GLEICHWERTIG bei wahrem d = 0 auf MNIST damit ~70 % (vorher ~75 %); ist die Streuung
    von suche - block2x2 groesser als 0,9 pp, wird GLEICHWERTIG in W-4 unerreichbar (dann HU).
  Wilcoxon-Glieder: 2/2^20 · 4 weit unter 0,05 -> Holm x4 kostet dort nichts.

VORREGISTRIERT (Holm ueber GENAU {W-1, W-2, W-3, W-4}, getrennt fuer Wilcoxon und TOST, wie I-005-v2):
  W-1 MNIST:   d_s = acc(suche) - acc(walsh_fest), delta = 0,5 pp.
  W-2 Fashion: dasselbe.
      Holm-p_TOST < 0,05: Q-0 nicht H1 -> HU (Positivkontrolle); Sensitivitaet verfehlt -> HU
                          (Sensitivitaet); sonst GLEICHWERTIG (bei Holm-p < 0,05 zusaetzlich im Text)
      Holm-p < 0,05, d > 0 -> SUCHE_BESSER (die gesuchte Struktur ist mehr als feste Walsh-Merkmale)
      Holm-p < 0,05, d < 0 -> WALSH_BESSER (Walsh liegt im Suchraum, die Suche verfehlt ihn; der Text
                              nennt, auf wie vielen Seeds Walsh schon im LOO-Kriterium besser war)
      sonst HU.
      Sensitivitaet (nicht in der Familie): mittel(suche - identitaet) >= 2·delta = 1 pp, Wilcoxon p <
      0,05 (I-005-v2: MNIST 3,77, Fashion 2,43 pp). Ohne sie hiesse „gleich“ nur „Schaltung egal“.
  W-3 MNIST:   d_s = acc(walsh_fest) - acc(identitaet) (bringt die Walsh-Struktur ueberhaupt etwas?):
      Holm-p < 0,05, d > 0 -> H1; Holm-p < 0,05, d < 0 -> H0 (Gegenrichtung: globale Walsh-Merkmale
      schlechter als Pixelquadrate); Holm-p_TOST(±0,5 pp) < 0,05 und Sensitivitaet -> H0 (gleich);
      TOST ja, Sensitivitaet nein -> HU (Sensitivitaet); sonst HU. (Vorhersage aus C-mnist: H0 Gegenr.)
  W-4 MNIST:   d_s = acc(suche) - acc(fest_block2x2) (FESTER lokaler 2x2-Block-Walsh, k = 2, Qubits
      {3,7}, ohne Suche; ist die Suche im Kern ein lokaler Block-Walsh-Klassifikator?), delta = 0,5 pp.
      Verzweigung wie W-1: Holm-p_TOST < 0,05 -> Q-0 nicht H1 -> HU (Positivkontrolle); Sensitivitaet
      verfehlt -> HU (Sensitivitaet); sonst GLEICHWERTIG. Holm-p < 0,05, d > 0 -> SUCHE_BESSER;
      d < 0 -> BLOCK_BESSER (block2x2 liegt ebenfalls im Suchraum); sonst HU.
      WAHL VON block2x2 (offen benannt): Sie stuetzt sich NUR auf die Rauchtest-Seeds 942-947 (MNIST
      942/943: block2x2 0,926 vs. kurze Suche 0,929; 80 SA-Schritte), NICHT auf Mess-Seeds; nach dem
      Rauchtest eingefuegt, vor jeder Messzahl (Direktor 29.09.). Die anderen festen Bloecke bleiben
      beschreibend; kein Durchsuchen mehrerer Bloecke im Urteil (sonst Auswahlverzerrung).
  Fehlt ein Datensatz, zaehlt sein Glied mit p = 1 (Familie bleibt 4).
  Q-0 Positivkontrolle der Suche (Strukturaufgabe, Verfahren unveraendert aus I-005, frische Seeds,
      NICHT in der Familie): Voraussetzung fuer GLEICHWERTIG; ~50 s je Seed.
  W-S beschreibend: struktur() je Suchergebnis (k, Maske, Walsh-aequivalent), suche_sah_walsh.
  W-B beschreibend: alle Arme, Dosis-Kurve fester Block-Walsh, G = (walsh - id)/(suche - id),
      L2-Empfindlichkeit, walsh_val vs. suche_val, Aufwand.

LAUFZEIT (aus I-005-v2 gemessen, 5 Prozesse: Suche 522 s MNIST / 442 s Fashion je Seed, Rest der
  gemeinsamen Arme ~140 s, Strukturseed ~50 s; walsh_fest + Block-Arme + L2: 11 Koepfe, ~10 s):
  MNIST ~670 s, Fashion ~590 s je Seed -> 20·670 + 20·590 + 10·50 = ~25.700 CPU-s.
  INSTITUT_PROZESSE = 5: ~1 h 30 min (bis ~1 h 45 min mit Lastungleichgewicht); = 2: ~3 h 35 min.

RAUCHTEST 29.09. (Entwurfs-Seeds 942-947, 80 SA-Schritte, INSTITUT_PROZESSE = 2, 101 s; KEIN Befund):
  MNIST walsh_fest 0,880/0,881 < id 0,900 < suche 0,927/0,931; Fashion walsh 0,789/0,800 ~ id. Auffaellig:
  der FESTE block2x2 (k = 2) lag auf MNIST bei 0,926 -- nahe an der (kurzen) Suche. Beschreibend, siehe
  Daraufhin hat der Direktor W-4 (suche vs. block2x2) in die Familie aufgenommen, vor jeder Messzahl.

GRENZEN: Test 5.000 Bilder (0,02 pp je Bild). Alles hier ist klassisch simulierbar; die Frage ist, wie
  viel Struktur die Suche findet, nicht Quantenvorteil. walsh_fest nimmt die natuerliche Pixelordnung;
  fuer k = 8 ist jede lineare Bit-Permutation der Pixel gleichwertig (T1), eine quadratische Maske nicht.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen import clifford_suche_v2 as v2
from abteilungen.clifford import LAGEN, QUBITS_JE_DATENSATZ
from abteilungen.clifford_suche import (_bewerte, _gemeinsame_arme, aequivalent, holm, kanonisch,
                                        kaskaden_kennzahl, unitaer_clifford)
from abteilungen.clifford_suche_v2 import tost_p
from labor.gemeinsam import parallel, speichere, wilcoxon_exakt, zusammenfassung

DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
N_SEEDS = 20
SEEDS = {"mnist": tuple(range(640, 660)), "fashion_mnist": tuple(range(660, 680))}
SEEDS_STRUKTUR = tuple(range(680, 690))
ENTWURF_SEEDS = {"mnist": (942, 943), "fashion_mnist": (944, 945)}
ENTWURF_STRUKTUR = (946, 947)
# belegt laut Karte/Direktor 29.09. (zusaetzlich zu v2._BELEGT)
_BELEGT = (v2._BELEGT | set(range(190, 240)) | set(range(300, 370)) | set(range(420, 440))
           | set(range(950, 967)) | set(range(970, 982)))
KOPF_L2 = {"mnist": 0.003, "fashion_mnist": 0.003}      # aus I-005-v2 (Tuning-Seeds 950-955)
L2_BESCHREIBEND = (1e-3, 1e-2)
ALPHA, TOST_W = 0.05, 0.005
SENS_MIN = 2 * TOST_W
FAMILIE = {"W-1": ("mnist", "suche_minus_walsh"), "W-2": ("fashion_mnist", "suche_minus_walsh"),
           "W-3": ("mnist", "walsh_minus_id"), "W-4": ("mnist", "suche_minus_block2x2")}
N = 8
BLOECKE = {"block2x2": (3, 7), "zeilen16": (4, 5, 6, 7), "block4x4": (2, 3, 6, 7),
           "block8x8": (1, 2, 3, 5, 6, 7), "walsh_fest": tuple(range(N))}

HYPOTHESEN = {
    "Q-0": "Positivkontrolle der Suche wie I-005 Q-0 (Strukturaufgabe, frische Seeds 680-689). Nicht in der Familie; "
           "Voraussetzung fuer GLEICHWERTIG.",
    "W-1": "MNIST 16x16: Suche (0 Magiewinkel) vs. FESTE quadrierte Walsh-Hadamard-Merkmale (walsh_fest, liegt im "
           "Suchraum), gleicher Kopf, gepaart. Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (wenn Q-0 H1 und suche - id "
           ">= 1 pp, p < 0,05; sonst HU mit Grund); sonst Holm-p < 0,05 -> SUCHE_BESSER / WALSH_BESSER; sonst HU.",
    "W-2": "Fashion-MNIST 16x16: wie W-1.",
    "W-3": "MNIST: walsh_fest vs. Identitaet. Holm-p < 0,05 -> H1 / H0 (Gegenrichtung); Holm-p_TOST(+-0,5 pp) < 0,05 "
           "und Sensitivitaet -> H0 (gleich); sonst HU. Holm ueber W-1..W-4, getrennt fuer Wilcoxon und TOST.",
    "W-4": "MNIST: Suche vs. FESTER block2x2 (lokaler 2x2-Block-Walsh, k = 2, ohne Suche; Wahl nur aus Rauchtest-"
           "Seeds 942-947). Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (wenn Q-0 H1 und suche - id >= 1 pp, p < 0,05; "
           "sonst HU mit Grund); sonst Holm-p < 0,05 -> SUCHE_BESSER / BLOCK_BESSER; sonst HU.",
    "W-S": "beschreibend: Struktur der gefundenen Schaltungen (Blockdimension k, Vorzeichenmaske, Walsh-aequivalent).",
    "W-B": "beschreibend: alle Arme, Dosis-Kurve fester Block-Walsh (k = 0, 2, 4, 4, 6, 8), L2-Empfindlichkeit, "
           "LOO-Kriterium walsh vs. suche, Aufwand.",
}


# ------------------------------------------------------------------ Theorie
def codes_block(qubits, n=N) -> np.ndarray:
    """Ry(pi/2) nur in Lage 0 auf den genannten Qubits, sonst Ry(0)."""
    c = np.zeros((LAGEN, n), np.int64)
    c[0, list(qubits)] = 1
    return c


WALSH_CODES = codes_block(BLOECKE["walsh_fest"])


def _affin_linear(g, V) -> bool:
    """g: Vorzeichen (+-1) auf den Elementen v von V (Index v), V: Unterraum (Indexliste, enthaelt 0).
    Affin-linear <=> g(v1) g(v2) g(v1^v2) g(0) = 1 fuer alle v1, v2 in V."""
    x = np.bitwise_xor.outer(V, V)
    return bool(np.all(g[V][:, None] * g[V][None, :] * g[x] * g[0] > 0))


def struktur(U, tol=1e-6) -> dict:
    """Pruefung von T2 fuer eine Clifford-Unitaere U (Zeilen = Ausgaenge, Spalten = Pixel):
    k, gemeinsamer Unterraum, Bruhat-Form (je Nebenklasse gemeinsame Maske), Maske noetig?, Walsh-aequivalent."""
    A = U.double().numpy() if isinstance(U, torch.Tensor) else np.asarray(U, dtype=float)
    d = A.shape[0]
    traeger = [np.flatnonzero(np.abs(r) > tol) for r in A]
    groesse = {len(t) for t in traeger}
    k = int(round(math.log2(len(traeger[0]))))
    aus = {"k": k, "einheitlich": len(groesse) == 1 and 2 ** k == len(traeger[0])}
    V = np.sort(traeger[0] ^ traeger[0][0])
    maske = np.zeros(d, bool)
    maske[V] = True
    aus["unterraum"] = bool(aus["einheitlich"] and maske[np.bitwise_xor.outer(V, V)].all())
    aus["gemeinsamer_unterraum"] = bool(aus["unterraum"] and all(
        len(t) == len(V) and maske[t ^ t[0]].all() for t in traeger))
    if not aus["gemeinsamer_unterraum"]:
        aus.update(bruhat_form=False, maske_noetig=None, walsh_aequivalent=False)
        return aus
    gruppen = {}
    for i, t in enumerate(traeger):
        gruppen.setdefault(int(t.min()), []).append(i)
    lin_zeilen, bruhat = 0, True
    for a0, zeilen in gruppen.items():
        sig = []
        for i in zeilen:
            g = np.zeros(d)
            g[V] = np.sign(A[i, V ^ a0])
            sig.append(g)
            lin_zeilen += _affin_linear(g, V)
        bruhat &= len(zeilen) == len(V) and all(_affin_linear(sig[0] * s, V) for s in sig[1:])
    aus["bruhat_form"] = bool(bruhat)
    aus["maske_noetig"] = bool(lin_zeilen < d)
    aus["walsh_aequivalent"] = bool(k == N and aequivalent(torch.as_tensor(A), sylvester(N)))
    return aus


def sylvester(n=N) -> torch.Tensor:
    import scipy.linalg
    return torch.tensor(scipy.linalg.hadamard(2 ** n), dtype=torch.float64) / 2 ** (n / 2)


# ------------------------------------------------------------------ Arbeiter
def _mess_seed(a):
    """(ds, seed, schritte, politur_max, n_zufall, l2) -> Zeile."""
    ds, s, schritte, politur_max, n_zufall, l2 = a
    n, t0 = QUBITS_JE_DATENSATZ[ds], time.time()
    teile = v2._daten(ds, s)
    z, bew, xs, best = _gemeinsame_arme(s, n, teile, schritte, politur_max, n_zufall, l2)
    z["datensatz"], z["kopf_l2"] = ds, l2
    z["suche_sah_walsh"] = kanonisch(WALSH_CODES).astype(np.int8).tobytes() in bew.cache
    for name, q in BLOECKE.items():
        c = codes_block(q, n)
        arm = "walsh" if name == "walsh_fest" else f"fest_{name}"
        z[arm], konf, richtig = _bewerte(c, n, xs)
        z[f"{arm}_val"] = bew(c)[0]
        if arm == "walsh":
            z["walsh_kaskade"] = kaskaden_kennzahl(konf, richtig)
    for l2b in L2_BESCHREIBEND:
        xsb = (*xs[:4], l2b)
        for arm, c in (("suche", best), ("walsh", WALSH_CODES), ("identitaet", np.zeros((LAGEN, n), np.int64))):
            z[f"{arm}_l2_{l2b:g}"] = _bewerte(c, n, xsb)[0]
    z["suche_struktur"] = struktur(unitaer_clifford(best, n))
    z["suche_hbits"] = int(kanonisch(best).sum())
    z["sekunden"] = time.time() - t0
    return z


def _arbeiter(a):
    art, arg = a
    return art, (v1._struktur_seed(arg) if art == "struktur" else _mess_seed(arg))


def _pruefe_seeds(seeds, struktur_seeds, datensaetze):
    alle = [s for ds in datensaetze for s in seeds[ds]] + list(struktur_seeds)
    entwurf = {s for v in ENTWURF_SEEDS.values() for s in v} | set(ENTWURF_STRUKTUR)
    return len(set(alle)) == len(alle) and not (set(alle) & (_BELEGT | entwurf))


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=N_SEEDS, seeds=None, seeds_struktur=SEEDS_STRUKTUR, datensaetze=DATENSAETZE,
         schritte=v1.SA_SCHRITTE, politur_max=v1.POLITUR_MAX, n_zufall=v1.N_ZUFALL, mit_struktur=True):
    datensaetze = tuple(datensaetze)
    if schnell:                                                          # nur Entwurfs-Seeds
        seeds, seeds_struktur = ENTWURF_SEEDS, ENTWURF_STRUKTUR
        schritte, politur_max, n_zufall = 80, 20, 5
    else:
        if n_seeds not in (10, 20):
            raise ValueError("n_seeds ist vorregistriert: 20 (10 nur als Teillauf)")
        seeds = seeds or {ds: SEEDS[ds][:n_seeds] for ds in datensaetze}
        if not _pruefe_seeds(seeds, seeds_struktur if mit_struktur else (), datensaetze):
            raise ValueError("Seeds nicht frisch oder nicht disjunkt")
    seeds = {ds: list(seeds[ds]) for ds in datensaetze}
    t0 = time.time()
    aufg = [("mess", (ds, s, schritte, politur_max, n_zufall, KOPF_L2[ds])) for ds in datensaetze for s in seeds[ds]]
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
            st = z["suche_struktur"]
            print(f"  {NAME[ds]} {z['seed']}: suche {z['suche']:.4f} walsh {z['walsh']:.4f} id {z['identitaet']:.4f} "
                  f"median {z['zufall_median']:.4f} | k {st['k']} Maske {st['maske_noetig']} Walsh-aequiv "
                  f"{st['walsh_aequivalent']} | val suche {z['suche_val']:.4f} walsh {z['walsh_val']:.4f} "
                  f"({z['sekunden']:.0f} s)", flush=True)
    e = {"abteilung": "Clifford", "modul": "walsh_kontrolle", "karte": "I-036", "version": 1,
         "hypothesen": HYPOTHESEN, "seeds": seeds, "seeds_struktur": list(seeds_struktur) if mit_struktur else [],
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "tau_start": v1.TAU_START, "tau_ende": v1.TAU_ENDE, "ridge_lambda": v1.RIDGE_LAMBDA,
                           "kopf_l2": KOPF_L2, "l2_beschreibend": list(L2_BESCHREIBEND), "tost": TOST_W,
                           "sens_min": SENS_MIN, "bloecke": {k: list(v) for k, v in BLOECKE.items()}},
         "zeilen": zeilen, "zeilen_struktur": zs, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, zs)
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Auswertung
def wilcoxon_schnell(diff) -> float:
    """Derselbe exakte zweiseitige Wilcoxon wie labor.gemeinsam.wilcoxon_exakt (gleiche Raenge, gleiche
    Zweiseitigkeit), aber die Nullverteilung der Rangsumme per Zaehl-Rekursion statt 2^n Aufzaehlung:
    bei n = 20 Mikrosekunden statt ~10 s je Aufruf. Vortest prueft Gleichheit fuer n <= 12."""
    d = np.asarray([v for v in diff if v != 0.0])
    n = len(d)
    if n == 0:
        return 1.0
    raenge = np.argsort(np.argsort(np.abs(d))) + 1
    w = float(raenge[d > 0].sum())
    zahl = np.zeros(n * (n + 1) // 2 + 1)
    zahl[0] = 1.0
    for r in raenge:
        zahl[r:] = zahl[r:] + zahl[:-r].copy()
    mitte = raenge.sum() / 2
    summen = np.arange(len(zahl))
    return float(zahl[np.abs(summen - mitte) >= abs(w - mitte) - 1e-9].sum() / 2.0 ** n)


def _gepaart(d, einseitig_q=0.95):
    """Wie clifford_suche._gepaart (Mittel, Wilcoxon, t-KI 90 %), nur mit wilcoxon_schnell (n = 20)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(einseitig_q, len(d) - 1))
    return {"mittel": m, "p": wilcoxon_schnell(d), "ki_unten": m - q * se, "ki_oben": m + q * se,
            "positiv": int((d > 0).sum()), "n": len(d)}


def _ds_auswertung(zz) -> dict:
    arr = lambda k: np.array([z[k] for z in zz], dtype=float)
    su, wa, id_ = arr("suche"), arr("walsh"), arr("identitaet")
    r = {"n": len(zz),
         "suche_minus_walsh": {**_gepaart(su - wa), "p_tost": tost_p(su - wa, TOST_W)},
         "walsh_minus_id": {**_gepaart(wa - id_), "p_tost": tost_p(wa - id_, TOST_W)},
         "sensitivitaet": {"mittel": float((su - id_).mean()), "p": wilcoxon_schnell(su - id_)},
         "walsh_val_besser": int(sum(z["walsh_val"] > z["suche_val"] for z in zz))}
    bl = arr("fest_block2x2")
    r["suche_minus_block2x2"] = {**_gepaart(su - bl), "p_tost": tost_p(su - bl, TOST_W)}
    r["block_val_besser"] = int(sum(z["fest_block2x2_val"] > z["suche_val"] for z in zz))
    r["sensitiv"] = bool(r["sensitivitaet"]["mittel"] >= SENS_MIN and r["sensitivitaet"]["p"] < ALPHA)
    b = {arm: zusammenfassung(arr(arm)) for arm in ("suche", "walsh", "identitaet", "zufall_median", "zufall_best100")}
    b["dosis_fest"] = {f"{name} (k={len(q)})": float(arr("walsh" if name == "walsh_fest" else f"fest_{name}").mean())
                       for name, q in BLOECKE.items()}
    gs = float((su - id_).mean())
    b["G_walsh"] = float((wa - id_).mean()) / gs if gs > 0 else None
    b["suche_minus_median"] = _gepaart(su - arr("zufall_median"))
    b["walsh_minus_median"] = _gepaart(wa - arr("zufall_median"))
    b["l2"] = {f"{arm}_l2_{l:g}": float(arr(f"{arm}_l2_{l:g}").mean()) for l in L2_BESCHREIBEND
               for arm in ("suche", "walsh", "identitaet") if f"{arm}_l2_{l:g}" in zz[0]}
    b["val_suche_minus_walsh"] = float((arr("suche_val") - arr("walsh_val")).mean())
    st = [z["suche_struktur"] for z in zz]
    b["struktur"] = {"k": [s["k"] for s in st], "k_mittel": float(np.mean([s["k"] for s in st])),
                     "bruhat_form": int(sum(bool(s["bruhat_form"]) for s in st)),
                     "maske_noetig": int(sum(bool(s["maske_noetig"]) for s in st)),
                     "walsh_aequivalent": int(sum(bool(s["walsh_aequivalent"]) for s in st)),
                     "suche_sah_walsh": int(sum(bool(z["suche_sah_walsh"]) for z in zz))}
    for k in ("suche_bewertungen", "suche_sekunden", "sekunden"):
        b[k] = float(np.mean([z[k] for z in zz]))
    r["beschreibend"] = b
    return r


def auswertung(zeilen: dict, zs) -> dict:
    a = {"datensaetze": {ds: _ds_auswertung(zz) for ds, zz in zeilen.items() if zz}, "n_struktur": len(zs)}
    if zs:
        a["Q0"] = v1.auswertung([], zs)["Q0"]
    p, pt = {}, {}
    for h, (ds, k) in FAMILIE.items():
        d = a["datensaetze"].get(ds, {}).get(k)
        p[h], pt[h] = (d["p"], d["p_tost"]) if d else (1.0, 1.0)
    a["holm"], a["holm_tost"] = holm(p), holm(pt)
    return a


def _zahlen(ds, d, ph, pth):
    return (f"{NAME[ds]}: d = {d['mittel'] * 100:+.2f} pp, 90-%-KI [{d['ki_unten'] * 100:+.2f}; "
            f"{d['ki_oben'] * 100:+.2f}], TOST(+-{TOST_W * 100:.1f} pp) p = {d['p_tost']:.4f}, Holm-p_TOST = "
            f"{pth:.4f}, Wilcoxon p = {d['p']:.4f}, Holm-p = {ph:.4f}, {d.get('positiv', 0)}/{d['n']} Seeds positiv")


def _sens_text(r):
    s = r["sensitivitaet"]
    return f"; Sensitivitaet suche - id {s['mittel'] * 100:+.2f} pp (p = {s['p']:.4f})"


def _urteil_walsh(h, ds, a, q0_h1) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    d, ph, pth = r["suche_minus_walsh"], a["holm"][h], a["holm_tost"][h]
    st = r["beschreibend"]["struktur"]
    t = (_zahlen(ds, d, ph, pth) + _sens_text(r) + f"; Suche Walsh-aequivalent in {st['walsh_aequivalent']}/{r['n']}, "
         f"Blockdimension k Mittel {st['k_mittel']:.2f}; Walsh im LOO-Kriterium besser auf {r['walsh_val_besser']}/{r['n']}")
    if pth < ALPHA:
        if not q0_h1:
            return "HU (Positivkontrolle): gleich im Rahmen +-0,5 pp, aber Q-0 nicht H1 -- " + t
        if not r["sensitiv"]:
            return (f"HU (Sensitivitaet): gleich im Rahmen +-0,5 pp, aber die Suche schlaegt die Identitaet nicht um "
                    f">= {SENS_MIN * 100:.1f} pp nachweisbar -- " + t)
        extra = " (Unterschied zusaetzlich nachweisbar, aber kleiner als die Grenze)" if ph < ALPHA else ""
        return ("GLEICHWERTIG: die Clifford-Suche leistet nicht mehr als feste quadrierte Walsh-Merkmale "
                f"(+-0,5 pp){extra} -- " + t)
    if ph < ALPHA and d["mittel"] > 0:
        return "SUCHE_BESSER: die gesuchte Clifford-Struktur ist mehr als der feste Walsh-Klassifikator -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return ("WALSH_BESSER: feste Walsh-Merkmale schlagen die Suche, obwohl sie im Suchraum liegen "
                "(Suchdefizit oder Siegerfluch) -- " + t)
    return "HU: " + t


def _urteil_block(h, ds, a, q0_h1) -> str:
    """W-4: Suche vs. fester block2x2; Verzweigung wie W-1 (_urteil_walsh)."""
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    d, ph, pth = r["suche_minus_block2x2"], a["holm"][h], a["holm_tost"][h]
    t = (_zahlen(ds, d, ph, pth) + _sens_text(r) + f"; Blockdimension k der Suche Mittel "
         f"{r['beschreibend']['struktur']['k_mittel']:.2f}; block2x2 im LOO-Kriterium besser auf "
         f"{r['block_val_besser']}/{r['n']}; Wahl von block2x2 nur aus Rauchtest-Seeds 942-947")
    if pth < ALPHA:
        if not q0_h1:
            return "HU (Positivkontrolle): gleich im Rahmen +-0,5 pp, aber Q-0 nicht H1 -- " + t
        if not r["sensitiv"]:
            return (f"HU (Sensitivitaet): gleich im Rahmen +-0,5 pp, aber die Suche schlaegt die Identitaet nicht um "
                    f">= {SENS_MIN * 100:.1f} pp nachweisbar -- " + t)
        extra = " (Unterschied zusaetzlich nachweisbar, aber kleiner als die Grenze)" if ph < ALPHA else ""
        return ("GLEICHWERTIG: die Clifford-Suche leistet nicht mehr als der feste lokale 2x2-Block-Walsh "
                f"(+-0,5 pp){extra} -- " + t)
    if ph < ALPHA and d["mittel"] > 0:
        return "SUCHE_BESSER: die gesuchte Clifford-Struktur ist mehr als der feste 2x2-Block-Walsh -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return ("BLOCK_BESSER: der feste 2x2-Block-Walsh schlaegt die Suche, obwohl er im Suchraum liegt "
                "(Suchdefizit oder Siegerfluch) -- " + t)
    return "HU: " + t


def _urteil_id(h, ds, a) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    d, ph, pth = r["walsh_minus_id"], a["holm"][h], a["holm_tost"][h]
    t = _zahlen(ds, d, ph, pth) + _sens_text(r)
    if ph < ALPHA and d["mittel"] > 0:
        return "H1: feste Walsh-Merkmale schlagen die Pixelquadrate (Identitaet) -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return "H0 (Gegenrichtung): globale Walsh-Merkmale sind schlechter als die Pixelquadrate -- " + t
    if pth < ALPHA:
        if r["sensitiv"]:
            return "H0 (gleich): Walsh = Identitaet im Rahmen +-0,5 pp -- " + t
        return "HU (Sensitivitaet): gleich im Rahmen +-0,5 pp, aber keine Schaltung wirkt nachweisbar -- " + t
    return "HU: " + t


def urteil(a: dict) -> dict:
    u = {"Q-0": (v1.urteil({"Q0": a["Q0"], "n_struktur": a["n_struktur"]})["Q-0"] if "Q0" in a
                 else "HU (Vorbedingung): keine Strukturseeds")}
    q0_h1 = u["Q-0"].startswith("H1")
    u["W-1"] = _urteil_walsh("W-1", "mnist", a, q0_h1)
    u["W-2"] = _urteil_walsh("W-2", "fashion_mnist", a, q0_h1)
    u["W-3"] = _urteil_id("W-3", "mnist", a)
    u["W-4"] = _urteil_block("W-4", "mnist", a, q0_h1)
    ts, tb = [], []
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for ds, r in a["datensaetze"].items():
        b, st = r["beschreibend"], r["beschreibend"]["struktur"]
        ts.append(f"{NAME[ds]}: Bruhat-Form {st['bruhat_form']}/{r['n']}, Maske noetig {st['maske_noetig']}/{r['n']}, "
                  f"Walsh-aequivalent {st['walsh_aequivalent']}/{r['n']}, k {st['k']} (Mittel {st['k_mittel']:.2f}), "
                  f"Suche hat walsh_fest bewertet in {st['suche_sah_walsh']}/{r['n']}")
        tb.append(f"{NAME[ds]}: suche {b['suche']['mittel']:.4f}, walsh {b['walsh']['mittel']:.4f}, id "
                  f"{b['identitaet']['mittel']:.4f}, median {b['zufall_median']['mittel']:.4f}, best100 "
                  f"{b['zufall_best100']['mittel']:.4f}; G_walsh {fmt(b['G_walsh'])}; fest ohne Suche: "
                  + ", ".join(f"{k} {v:.4f}" for k, v in b["dosis_fest"].items())
                  + f"; L2: " + ", ".join(f"{k} {v:.4f}" for k, v in b["l2"].items())
                  + f"; LOO suche - walsh {b['val_suche_minus_walsh'] * 100:+.2f} pp; {b['suche_bewertungen']:.0f} Bew., "
                  f"{b['sekunden']:.0f} s je Seed")
    u["W-S"] = "beschreibend: " + (" | ".join(ts) if ts else "keine Mess-Seeds")
    u["W-B"] = "beschreibend: " + (" | ".join(tb) if tb else "keine Mess-Seeds")
    return u


# ------------------------------------------------------------------ Vortest: Theorie und Bausteine
def _basis(V):
    """GF(2)-Basis eines Unterraums (Indexliste)."""
    basis, span = [], {0}
    for v in V:
        if int(v) not in span:
            basis.append(int(v))
            span |= {s ^ int(v) for s in span}
    return basis


def merkmale_bruhat(U, x):
    """Unabhaengige Nachrechnung von T2: aus U nur V, Nebenklassen und Maske (Vorzeichen der ersten Zeile je
    Nebenklasse) ablesen, dann Merkmale = (Sylvester-Walsh je Block von m * x)^2 -- ohne U auf x anzuwenden."""
    import scipy.linalg
    A = U.double().numpy()
    t0 = np.flatnonzero(np.abs(A[0]) > 1e-6)
    b = _basis(t0 ^ t0[0])
    k = len(b)
    koord = np.array([int(np.bitwise_xor.reduce([b[j] for j in range(k) if (t >> (k - 1 - j)) & 1] or [0]))
                      for t in range(2 ** k)])
    Hk = scipy.linalg.hadamard(2 ** k) / 2 ** (k / 2)
    gesehen, feats = set(), []
    for r in A:
        a0 = int(np.flatnonzero(np.abs(r) > 1e-6).min())
        if a0 in gesehen:
            continue
        gesehen.add(a0)
        idx = a0 ^ koord
        feats.append(((x[:, idx] * np.sign(r[idx])[None, :]) @ Hk.T) ** 2)
    return np.concatenate(feats, 1)


def _vortest_bausteine() -> bool:
    from abteilungen.clifford_suche import Bewerter, born, normiere, ridge_loo
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(29)
    H8 = sylvester(N)
    # (T1) walsh_fest = H^8 bis auf Vorzeichen/Reihenfolge; Born-Merkmale = Permutation von (H x)^2
    Uw = unitaer_clifford(WALSH_CODES, N)
    xn = normiere(torch.tensor(rng.random((64, 2 ** N)) ** 3, dtype=torch.float32))
    f_zelle = np.sort(born(xn, Uw).double().numpy(), 1)
    f_walsh = np.sort(((xn.double() @ H8.T) ** 2).numpy(), 1)
    if not aequivalent(Uw, H8) or np.abs(f_zelle - f_walsh).max() > 1e-5:
        fehler("walsh_fest ist nicht H^8 bis auf Permutation/Vorzeichen")
    if not set(np.unique(kanonisch(WALSH_CODES)).tolist()) <= {0, 1} or kanonisch(WALSH_CODES).shape != (LAGEN, N):
        fehler("walsh_fest ist kein Punkt des kanonischen Suchraums")
    for lage in range(LAGEN):                      # jede Lage allein: ebenfalls Walsh
        c = np.zeros((LAGEN, N), np.int64)
        c[lage] = 1
        if not aequivalent(unitaer_clifford(c, N), H8):
            fehler(f"alle H in Lage {lage} nicht Walsh-aequivalent")
    c = np.zeros((LAGEN, N), np.int64)
    c[0] = c[1] = 1
    if struktur(unitaer_clifford(c, N))["k"] != 0:
        fehler("H in zwei Lagen hebt sich nicht auf (k != 0)")
    # (T2) Bruhat-Form fuer zufaellige Clifford; Merkmale unabhaengig nachgerechnet; Maske meist noetig
    masken = 0
    for _ in range(12):
        c = rng.integers(0, 4, (LAGEN, N))
        U = unitaer_clifford(c, N)
        s = struktur(U)
        masken += bool(s["maske_noetig"])
        f1 = np.sort(born(xn, U).double().numpy(), 1)
        f2 = np.sort(merkmale_bruhat(U, xn.double().numpy()), 1)
        if not s["bruhat_form"] or np.abs(f1 - f2).max() > 1e-5:
            fehler(f"T2 verletzt fuer zufaellige Codes {c.tolist()}: {s}")
            break
    if masken == 0:
        fehler("keine zufaellige Clifford mit Maske -- struktur() unterscheidet nicht")
    # (T3) struktur() erkennt Gegenbeispiele: Haar-orthogonal, Walsh mit zeilenweise verschiedener Maske
    Q, _ = np.linalg.qr(rng.normal(size=(2 ** N, 2 ** N)))
    if struktur(torch.tensor(Q))["bruhat_form"]:
        fehler("Haar-orthogonale Matrix als Bruhat-Form erkannt")
    y = np.arange(2 ** N)
    quad = (-1.0) ** (((y >> 7) & 1) * ((y >> 6) & 1))
    Hm = H8.numpy().copy()
    Hm[::2] *= quad[None, :]                       # halbe Zeilen mit quadratischer Maske, halbe ohne
    s = struktur(torch.tensor(Hm))
    if s["bruhat_form"] or not s["gemeinsamer_unterraum"]:
        fehler(f"verschiedene Masken je Zeile nicht erkannt: {s}")
    s = struktur(torch.tensor(H8.numpy() * quad[None, :]))
    if not (s["bruhat_form"] and s["maske_noetig"] and not s["walsh_aequivalent"]):
        fehler(f"gemeinsame quadratische Maske falsch eingeordnet: {s}")
    # (T4) Block-Arme: k = Zahl der H-Qubits, keine Maske, nur walsh_fest Walsh-aequivalent
    for name, q in BLOECKE.items():
        s = struktur(unitaer_clifford(codes_block(q), N))
        if s["k"] != len(q) or s["maske_noetig"] or s["walsh_aequivalent"] != (name == "walsh_fest"):
            fehler(f"Block-Arm {name}: {s}")
    s = struktur(unitaer_clifford(np.zeros((LAGEN, N), np.int64), N))
    if s["k"] != 0 or s["maske_noetig"] or s["walsh_aequivalent"]:
        fehler(f"Identitaet: {s}")
    # (T5) Suchkriterium sieht walsh_fest als klassisches Walsh-Modell; Zwischenspeicher-Schluessel passt
    xy = torch.tensor(rng.random((300, 2 ** N)) ** 3, dtype=torch.float32)
    yy = torch.tensor(rng.integers(0, 10, 300))
    bew = Bewerter(xy, yy, N)
    r1 = bew(WALSH_CODES)
    r2 = ridge_loo(((normiere(xy).double() @ H8.T) ** 2).float(), yy)
    if abs(r1[0] - r2[0]) > 1e-9 or abs(r1[1] - r2[1]) > 1e-3:
        fehler(f"Bewerter(walsh_fest) {r1} != Ridge-LOO auf (Hx)^2 {r2}")
    if kanonisch(WALSH_CODES).astype(np.int8).tobytes() not in bew.cache:
        fehler("Cache-Schluessel fuer suche_sah_walsh passt nicht zu Bewerter")
    # (S1) schneller Wilcoxon = wilcoxon_exakt, _gepaart = v1._gepaart; TOST/Holm/Familie
    for n in (1, 5, 8, 12):
        for _ in range(10):
            d = rng.normal(rng.normal(0, 0.003), 0.004, n)
            if abs(wilcoxon_schnell(d) - wilcoxon_exakt(d)) > 1e-12:
                fehler(f"wilcoxon_schnell != wilcoxon_exakt (n = {n}): {wilcoxon_schnell(d)} vs {wilcoxon_exakt(d)}")
                break
    d = rng.normal(0.002, 0.004, 10)
    g1, g2 = _gepaart(d), v1._gepaart(d)
    if any(abs(g1[k] - g2[k]) > 1e-12 for k in ("mittel", "p", "ki_unten", "ki_oben", "positiv", "n")):
        fehler(f"_gepaart weicht von clifford_suche._gepaart ab: {g1} vs {g2}")
    if abs(wilcoxon_schnell(np.full(20, 0.01)) - 2 / 2 ** 20) > 1e-15:
        fehler("kleinstes Wilcoxon-p bei n = 20 nicht 2/2^20")
    p0, p1 = tost_p(rng.normal(0.0, 0.001, 20), TOST_W), tost_p(TOST_W + rng.normal(0, 1e-4, 20), TOST_W)
    if not (p0 < 1e-6 and p1 > 0.05):
        fehler(f"TOST-p an den Enden falsch ({p0}, {p1})")
    h = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    if abs(h["a"] - 0.03) > 1e-12 or abs(h["b"] - 0.08) > 1e-12 or set(FAMILIE) != {"W-1", "W-2", "W-3", "W-4"}:
        fehler(f"Holm/Familie {h}")
    h4 = holm({"a": 0.01, "b": 0.012, "c": 0.02, "d": 0.5})
    if abs(h4["a"] - 0.04) > 1e-12 or abs(h4["b"] - 0.04) > 1e-12 or abs(h4["c"] - 0.04) > 1e-12 or h4["d"] != 0.5:
        fehler(f"Holm x4 {h4}")
    if FAMILIE["W-4"] != ("mnist", "suche_minus_block2x2") or BLOECKE["block2x2"] != (3, 7):
        fehler("W-4 zeigt nicht auf MNIST suche - fest_block2x2 (Qubits 3, 7)")
    # (S2) Seeds frisch, disjunkt; belegte und Entwurfs-Seeds werden abgelehnt; n_seeds nur 10/20
    for n in (10, 20):
        if not _pruefe_seeds({ds: SEEDS[ds][:n] for ds in DATENSAETZE}, SEEDS_STRUKTUR, DATENSAETZE):
            fehler(f"Seeds (n = {n}) nicht frisch/disjunkt")
    for schlecht in ({"mnist": (300,), "fashion_mnist": (661,)}, {"mnist": (420,), "fashion_mnist": (661,)},
                     {"mnist": (942,), "fashion_mnist": (661,)}, {"mnist": (660,), "fashion_mnist": (660,)},
                     {"mnist": (955,), "fashion_mnist": (661,)}):
        if _pruefe_seeds(schlecht, (), DATENSAETZE):
            fehler(f"belegte/doppelte Seeds durchgelassen: {schlecht}")
    try:
        lauf(n_seeds=15)
        fehler("n_seeds = 15 nicht abgelehnt")
    except ValueError:
        pass
    return ok


# ------------------------------------------------------------------ Vortest: Urteile
def _synth(rng, d_sw=0.0, sens=0.035, sd=0.001, n=20, walsh_val_besser=False, d_sb=None, sd_b=0.001,
           block_val_besser=False):
    """d_sw = suche - walsh, sens = suche - identitaet (Mittel); walsh - identitaet = sens - d_sw.
    d_sb = suche - fest_block2x2 (None: alle festen Bloecke = identitaet + 0,5 pp)."""
    zeilen = []
    for i in range(n):
        su = 0.935 + rng.normal(0, 0.002)
        z = {"seed": i, "suche": su, "walsh": su - d_sw + rng.normal(0, sd), "identitaet": su - sens + rng.normal(0, 0.001),
             "zufall_median": su - 0.025, "zufall_best100": su - 0.015, "suche_val": su,
             "walsh_val": su + (0.01 if walsh_val_besser else -0.01), "suche_bewertungen": 5000,
             "suche_sekunden": 500.0, "sekunden": 650.0, "suche_sah_walsh": False,
             "suche_struktur": {"k": 6, "bruhat_form": True, "maske_noetig": True, "walsh_aequivalent": False}}
        for name in BLOECKE:
            if name != "walsh_fest":
                z[f"fest_{name}"] = z["identitaet"] + 0.005
                z[f"fest_{name}_val"] = su - 0.02
        if d_sb is not None:
            z["fest_block2x2"] = su - d_sb + rng.normal(0, sd_b)
        z["fest_block2x2_val"] = su + (0.01 if block_val_besser else -0.01)
        for l in L2_BESCHREIBEND:
            for arm in ("suche", "walsh", "identitaet"):
                z[f"{arm}_l2_{l:g}"] = z[arm]
        zeilen.append(z)
    return zeilen


def _vortest_urteile(ziehungen=5) -> bool:
    import json
    ok = True
    rng = np.random.default_rng(36)
    gut = lambda: v1._synth_struktur(rng, 0.925)
    schlecht = lambda: v1._synth_struktur(rng, 0.79)
    # (Name, Struktur, MNIST-Parameter, Fashion-Parameter, {Hypothese: (Marke, [Textteile])}, Toleranz)
    faelle = [
        ("Walsh gleich", gut, {}, {},
         {"Q-0": ("H1", []), "W-1": ("GLEICHWERTIG", ["MNIST", "Walsh-Merkmale", "Holm-p_TOST"]),
          "W-2": ("GLEICHWERTIG", ["Fashion"]), "W-3": ("H1", ["Pixelquadrate", "MNIST"])}, 0),
        ("Suche besser, Walsh < id", gut, {"d_sw": 0.045}, {"d_sw": 0.02},
         {"W-1": ("SUCHE_BESSER", ["MNIST", "Walsh-aequivalent in 0/20"]), "W-2": ("SUCHE_BESSER", ["Fashion"]),
          "W-3": ("H0 (Gegenrichtung)", ["MNIST"])}, 0),
        ("Walsh besser", gut, {"d_sw": -0.01, "walsh_val_besser": True}, {"d_sw": -0.01},
         {"W-1": ("WALSH_BESSER", ["Suchraum", "LOO-Kriterium besser auf 20/20"]),
          "W-2": ("WALSH_BESSER", ["Fashion", "LOO-Kriterium besser auf 0/20"]), "W-3": ("H1", [])}, 0),
        ("Streuung W-1/W-2", gut, {"sd": 0.03}, {"sd": 0.03},
         {"W-1": ("HU:", ["MNIST"]), "W-2": ("HU:", ["Fashion"])}, 1),
        ("Streuung W-3", gut, {"d_sw": 0.035, "sd": 0.03}, {},
         {"W-3": ("HU:", ["MNIST"])}, 1),
        ("Positivkontrolle", schlecht, {}, {},
         {"Q-0": ("H0", []), "W-1": ("HU (Positivkontrolle)", ["MNIST", "Q-0"]),
          "W-2": ("HU (Positivkontrolle)", ["Fashion"]), "W-3": ("H1", [])}, 1),
        ("Sensitivitaet", gut, {"sens": 0.0}, {"sens": 0.0},
         {"W-1": ("HU (Sensitivitaet)", ["MNIST", "Identitaet"]), "W-2": ("HU (Sensitivitaet)", ["Fashion"]),
          "W-3": ("HU (Sensitivitaet)", ["MNIST"])}, 0),
        ("Walsh = id", gut, {"d_sw": 0.035}, {"d_sw": 0.035},
         {"W-1": ("SUCHE_BESSER", []), "W-3": ("H0 (gleich)", ["0,5 pp", "MNIST"])}, 1),
        ("klein, aber gleich", gut, {"d_sw": 0.002, "sd": 0.0003}, {"d_sw": -0.002, "sd": 0.0003},
         {"W-1": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"]), "W-2": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"])}, 0),
        ("nur MNIST", gut, {}, None,
         {"W-1": ("GLEICHWERTIG", ["MNIST"]), "W-2": ("HU: keine Fashion", [])}, 0),
        ("ohne Struktur", None, {}, {},
         {"Q-0": ("HU (Vorbedingung)", []), "W-1": ("HU (Positivkontrolle)", ["MNIST"])}, 0),
        # W-4 (Suche vs. fester block2x2)
        ("W-4 Suche besser (Standard)", gut, {}, {},
         {"W-4": ("SUCHE_BESSER", ["MNIST", "2x2-Block-Walsh", "Rauchtest-Seeds 942-947"])}, 0),
        ("W-4 gleich", gut, {"d_sw": 0.03, "d_sb": 0.0}, {},
         {"W-1": ("SUCHE_BESSER", []), "W-4": ("GLEICHWERTIG", ["lokale 2x2-Block-Walsh", "MNIST", "Holm-p_TOST"])}, 0),
        ("W-4 Block besser", gut, {"d_sb": -0.01, "block_val_besser": True}, {},
         {"W-4": ("BLOCK_BESSER", ["Suchraum", "LOO-Kriterium besser auf 20/20"])}, 0),
        ("W-4 Streuung", gut, {"d_sb": 0.0, "sd_b": 0.03}, {},
         {"W-4": ("HU:", ["MNIST"])}, 1),
        ("W-4 Positivkontrolle", schlecht, {"d_sb": 0.0}, {},
         {"W-4": ("HU (Positivkontrolle)", ["MNIST", "Q-0"])}, 1),
        ("W-4 Sensitivitaet", gut, {"sens": 0.0, "d_sb": 0.0}, {"sens": 0.0},
         {"W-4": ("HU (Sensitivitaet)", ["MNIST", "Identitaet"])}, 0),
        ("W-4 klein, aber gleich", gut, {"d_sb": 0.002, "sd_b": 0.0003}, {},
         {"W-4": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"])}, 0),
        ("W-4 ohne Struktur", None, {"d_sb": 0.0}, {},
         {"W-4": ("HU (Positivkontrolle)", ["MNIST"])}, 0),
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
                    print(f"  Vortest [{name}]: {h} -> {u[h][:160]}")
            if not (u["W-S"].startswith("beschreibend") and u["W-B"].startswith("beschreibend")
                    and all(isinstance(v, str) for v in u.values())):
                ok = False
                print(f"  VORTEST FEHLER [{name}]: W-S/W-B oder Urteil kein String")
            try:
                json.dumps({"auswertung": a, "urteil": u})
            except TypeError as err:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: nicht JSON-serialisierbar", err)
        for h, f in fehl.items():
            if f > tol:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: {h} {f}/{ziehungen} falsch")
    for _ in range(ziehungen):                     # grosse Streuung, wahre Differenz 0 -> NIE GLEICHWERTIG
        u = urteil(auswertung({ds: _synth(rng, sd=0.03, d_sb=0.0, sd_b=0.03) for ds in DATENSAETZE}, gut()))
        if any(u[h].startswith("GLEICHWERTIG") for h in ("W-1", "W-2", "W-4")):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung")
        u = urteil(auswertung({"fashion_mnist": _synth(rng)}, gut()))
        if not (u["W-4"].startswith("HU: keine MNIST") and u["W-1"].startswith("HU: keine MNIST")):
            ok = False
            print("  VORTEST FEHLER: fehlendes MNIST nicht als HU erkannt", u["W-4"][:80])
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: Theorie (T1-T5: walsh_fest = H^8 im Suchraum, Bruhat-Form mit unabhaengiger
    Nachrechnung, Gegenbeispiele, Block-Arme, Suchkriterium) + Statistik/Seeds + jeder Urteilszweig
    synthetisch (Marke UND Text, mehrere Ziehungen). Die Such-Bausteine prueft der Vortest von I-005."""
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
    print(speichere("I-036" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
