"""Abteilung Clifford -- Traegt die gesuchte MASKE oder die gesuchte BLOCKWAHL den Vorsprung der Suche?
(Karte I-036-v2, Quantenstufe Q2: exakte Simulation; Kern der Frage ist klassisch.)

HEILMEIER:
  Was wollen wir?  I-036 (05.10.) hat gezeigt: Die Clifford-Suche (0 Magiewinkel) schlaegt den FESTEN 2x2-Block-
      Walsh (block2x2) auf MNIST um +0,93 pp (19/20, W-4 SUCHE_BESSER), auf Fashion um +0,34 pp (beschreibend).
      Jede gefundene Schaltung hat Bruhat-Form: Merkmale = (blockweiser Walsh von (m ⊙ x))^2 mit Unterraum V_s
      (k = 4-7 auf MNIST) und echter quadratischer Vorzeichenmaske m_s. Welcher der beiden gesuchten Bestandteile
      traegt den Vorsprung vor block2x2: die Maske m_s oder die Blockwahl V_s?
  Wie heute?  Unbekannt. I-036 vergleicht nur ganze Schaltungen; feste Bloecke OHNE Maske (Dosis-Kurve) liegen
      alle unter block2x2 (block8x8 k=6: 0,9256, block4x4: 0,9107), sagen also nichts ueber V_s MIT/ohne m_s.
  Was ist neu?  Eine 2x2-Zerlegung JE SEED direkt aus der gefundenen Schaltung (keine weitere Suche): dieselbe
      (V, m)-Merkmalsabbildung mit ausgetauschtem V bzw. m, derselbe L-BFGS-Kopf, dieselben Daten.
      Neuheitspruefung: recherche/2026-10-05_neuheit_I-036-v2.md (nichts Vergleichbares gefunden).

THEORIE (vor dem Festlegen gerechnet; der Vortest (B1)-(B8) prueft jeden Satz):
  T1  (aus I-036 T2) Fuer eine Schaltung U in Bruhat-Form (struktur()["bruhat_form"]) gibt es einen Unterraum
      V ⊆ GF(2)^8 der Dimension k, die 256 Pixel zerfallen in 2^(8-k) Nebenklassen a0 + V, und jede Zeile von U ist
      auf ihrer Nebenklasse  ± m(y) · (-1)^(w·y) / 2^(k/2)  mit EINER Maske m je Nebenklasse. Die Born-Merkmale sind
      (bis auf Permutation) die quadrierten Walsh-Koeffizienten von (m ⊙ x) je Nebenklasse. Hier wird daraus eine
      Matrix W(V, m) gebaut (matrix_vm); W(V_s, m_s) hat GENAU die Zeilen von U bis auf Vorzeichen und Reihenfolge
      (aequivalent(), im Vortest und in JEDEM Mess-Seed geprueft). Die Arme sind also nur verschiedene W(V, m).
  T2  ABLESEN VON (V_s, m_s) aus U (zerlege): V_s = Traeger der ersten Zeile, verschoben auf 0; m_roh(y) =
      Vorzeichen der ersten Zeile, die auf der Nebenklasse von y lebt. m_roh ist nur bis auf einen AFFIN-LINEAREN
      Charakter je Nebenklasse bestimmt (andere Zeile derselben Nebenklasse = m_roh mal Charakter), und genau diese
      Charaktere aendern die Merkmale nicht: m·χ mit χ(a0+v) = ±(-1)^(w·v) vertauscht nur die Walsh-Zeilen des
      Blocks und dreht Vorzeichen; nach dem Quadrieren ist es eine Permutation der Merkmale, die der standardisierte
      Kopf exakt ausgleicht. Wirksam ist nur der QUADRATISCHE Anteil von m auf jeder Nebenklasse.
  T3  ARM ohne_maske = W(V_s, 1). WAHL m ≡ +1 statt „nur linearer Charakter“: nach T2 sind beide merkmalsgleich
      (der lineare Anteil ist wirkungslos), m ≡ +1 ist der eindeutige, einfachste Vertreter. ohne_maske heisst
      also genau: die Bloecke der Suche behalten, den quadratischen Teil der Maske weglassen. Vortest (B4): W(V_s, 1)
      ist aequivalent zu W(V_s, χ) fuer zufaellige affine χ je Nebenklasse und zu einer unabhaengig (ueber
      Charaktere w statt Basiskoordinaten) konstruierten Matrix; mit echter Maske ist W(V_s, m_s) NICHT aequivalent.
  T4  ARM block2x2_maske = W(V_b, m_s) mit V_b = <Qubit 3, Qubit 7> = {0, 1, 16, 17} (2x2-Pixelbloecke,
      Pixelindex 16·Zeile + Spalte, Qubit 0 = hoechstes Bit; = struktur von codes_block((3, 7)), Vortest).
      MEHRDEUTIGKEIT (offen benannt): Die Freiheit aus T2 ist bezueglich V_s harmlos, bezueglich V_b NICHT -- ein
      auf V_s-Nebenklassen affiner Faktor ist auf 2x2-Bloecken im Allgemeinen quadratisch. „Die Maske der Suche“
      ist fuer einen fremden Unterraum also erst nach einer Normierung definiert. VORAB FESTGELEGT (kanonische
      Maske): je V_s-Nebenklasse a0 + V_s (a0 = kleinster Pixelindex) wird m so mit einem affinen Charakter
      multipliziert, dass m(a0) = +1 und m(a0 ⊕ b_j) = +1 fuer die Basis b_1..b_k von V_s in REDUZIERTER
      ZEILENSTUFENFORM (eindeutig fuer V_s). Das ist eine Konvention, keine Naturtatsache; deshalb beschreibend
      zusaetzlich block2x2_maske mit zwei zufaelligen anderen Basen von V_s (alt1, alt2, je Seed fest gezogen):
      liegen sie weit von der kanonischen Wahl, gilt M-2 nur „fuer diese Konvention“ (so im Text zu lesen).
  T5  SONDERFAELLE: block2x2_maske mit m ≡ +1 = block2x2 (Vortest gegen codes_block((3, 7))). Ist die kanonische
      Maske auf jedem 2x2-Block affin, gilt block2x2_maske = block2x2 exakt (Anteil „maske_wirksam_2x2“ der
      2x2-Bloecke mit quadratischer Maske wird je Seed berichtet). Bei k = 0 (V = {0}) ist jedes W(V, m) die
      Identitaet (Pixelquadrate), unabhaengig von m; eine Suchschaltung mit k_s = 0 hat die kanonische Maske ≡ +1,
      also suche = ohne_maske = Identitaet und block2x2_maske = block2x2.
  T6  BEFUND AUF DEN VORHANDENEN I-036-SUCHERGEBNISSEN (05.10., nur gelesen, nichts neu gerechnet; beim Bau
      gefunden): die kanonische Maske ist auf 2x2-Bloecken ALLES ODER NICHTS -- auf jedem 2x2-Block affin
      (maske_wirksam_2x2 = 0, dann block2x2_maske = block2x2 EXAKT) in 12/20 MNIST- und 16/20 Fashion-Seeds,
      sonst auf jedem 2x2-Block quadratisch (= 1). Unabhaengig von der Basiskonvention (alt-Basen: dieselbe
      Wirksamkeit in 40/40). Folge: M-2 ist zu ~60 % (MNIST) mechanisch derselbe Vergleich wie I-036 W-4
      (suche - block2x2). ENTSCHEIDUNG DES DIREKTORS (05.10., vor jeder Messzahl): M-2 und M-4 sind deshalb
      BESCHREIBEND, ohne Familienurteil -- in ~60 % der Seeds gilt block2x2_maske = block2x2 exakt, M-2 wuerde dort
      nur I-036 W-4 wiederholen, und diese Replikation deckt P-0 bereits ab. Berichtet werden immer der Anteil der
      Seeds mit wirksamer 2x2-Maske und suche - block2x2_maske / block2x2_maske - block2x2 auf der Teilmenge
      „nur_wirksam“.
  ZERLEGUNG (beschreibend):            Maske m_s           ohne Maske
                         Bloecke V_s   suche               ohne_maske
                         Bloecke V_b   block2x2_maske      block2x2
      suche - block2x2 = (suche - ohne_maske) + (ohne_maske - block2x2)
                       = (suche - block2x2_maske) + (block2x2_maske - block2x2);
      Wechselwirkung I = suche - ohne_maske - block2x2_maske + block2x2 (traegt die Maske nur auf ihren eigenen
      Bloecken?). Haupteffekte als Mittel der beiden Wege.

ARME (alle mit DEMSELBEN logistischen L-BFGS-Kopf aus clifford_suche, standardisierte Merkmale, Kopf-L2 0,003
  wie I-036, 6.000 Nicht-Test-Bilder / 5.000 Testbilder, clifford.daten 16x16):
  suche           SA + Politur, unveraendert aus I-005/I-036 (gleiche Start-/SA-Seeds je Seed relativ, 100 Zufalls-
                  Clifford NUR fuer die Temperaturskala und den Zwischenspeicher -- ihre Test-Koepfe entfallen, sie
                  gingen in I-036 nur in beschreibende Arme; der Suchlauf selbst ist bitgleich zu I-036).
  ohne_maske      W(V_s, 1)                    block2x2_maske   W(V_b, m_s kanonisch)
  block2x2        codes_block((3, 7)) wie I-036 (Born-Merkmale der Schaltung)
  beschreibend    identitaet (Sensitivitaet), suche_rekonstruiert = W(V_s, m_s) (muss = suche sein, Numerik),
                  block2x2_maske_alt1/alt2 (T4), LOO-Kriterium (Ridge, Nicht-Test) aller Arme.

AUSSCHLUSSREGEL (vorab): Ein Mess-Seed, dessen Suchschaltung keine Bruhat-Form hat (struktur()) oder dessen
  Rekonstruktion W(V_s, m_s) nicht aequivalent zu U_s ist, wird GEZAEHLT und aus ALLEN Vergleichen dieses
  Datensatzes ausgeschlossen (auch aus P-0). Bleiben weniger als 10 gueltige Seeds, sind die Hypothesen dieses
  Datensatzes HU (Vorbedingung). (I-036: 40/40, I-005*: 70/70 in Bruhat-Form -- erwartet: 0 Ausschluesse.)

SEEDS (05.10. geprueft gegen ergebnisse/*.json (jede ganze Zahl 700-759 mit Pfad: nur Gatterzahlen/Schrittzahlen,
  keine Seeds), agenda.json, ideen.json, abteilungen/*.py, labor/*.py, direktor.py; _pruefe_seeds prueft
  zusaetzlich gegen walsh_kontrolle._BELEGT und die I-036-Seeds 640-689, 942-947):
  Mess-Seeds  MNIST 700-719, Fashion 720-739 (je 20)       Strukturaufgabe (Q-0) 740-749
  Rauchtest   MNIST 750/751, Fashion 752/753, Struktur 754/755 (nie Mess-Seeds)
  Seed-Zahl 20 aus der Streuung von suche - block2x2 in I-036: sd MNIST 0,68 pp, Fashion 0,43 pp (fuer M-1/M-2
  unbekannt, dies ist die beste Naeherung). TOST ±0,5 pp, je Familie EIN Glied (Holm x1 = kein Holm):
  p_TOST < 0,05 (t_0,95;19 = 1,73); se = sd/4,47:
    MNIST  GLEICHWERTIG erreichbar bei |d| < 0,5 - 1,73·0,152 = 0,24 pp;
    Fashion |d| < 0,5 - 1,73·0,096 = 0,33 pp. Wilcoxon: kleinstes p 2/2^20.

FAMILIEN (Direktor 05.10., vor jeder Zahl): MNIST {M-1} ist die Hauptfamilie (dort ist der zu zerlegende Effekt
  nachgewiesen). M-2/M-4 beschreibend (T6). Fashion {M-3} ist die REPLIKATION in EIGENER Familie, nicht nur
  beschreibend: der
  Vorsprung vor block2x2 ist dort zwar klein (+0,34 pp), aber die Teilvergleiche koennen gross sein (ohne_maske
  liegt nicht zwischen suche und block2x2: feste Bloecke ohne Maske mit k = 5-8 liegen auf Fashion 1-3 pp unter
  der Suche), also ist eine Richtungsaussage moeglich und als Replikation wertvoll. Eigene Familie, weil es eine
  eigene Frage auf eigenen Seeds ist und die MNIST-Hauptaussage nicht durch die schwaechere Replikation verduennt
  werden soll. Gleichwertigkeit auf Fashion braucht die eigene Positivkontrolle P-0F (siehe unten), die nach I-036
  voraussichtlich FEHLSCHLAEGT -- dann gibt es auf Fashion nur *_BESSER oder HU (so vorhergesagt).

VORREGISTRIERT (Familien {M-1} und {M-3}, je ein Glied, Holm x1; delta = 0,5 pp; nur gueltige Seeds):
  P-0  Positivkontrolle MNIST (NICHT in der Familie, Voraussetzung fuer GLEICHWERTIG in M-1): suche - block2x2
       reproduziert den I-036-Effekt: einseitiger exakter Wilcoxon (suche > block2x2) p < 0,05 UND Mittel >= 0,5 pp
       (= delta: ist der Gesamteffekt kleiner als die Gleichwertigkeitsgrenze, folgt „Teil gleichwertig“ schon aus
       der Kleinheit des Ganzen -- Grenzen relativ zum Effekt). -> BESTANDEN / DURCHGEFALLEN.
  P-0F dasselbe auf Fashion (Voraussetzung fuer GLEICHWERTIG in M-3).
  Q-0  Positivkontrolle der Suche wie I-005/I-036 (Strukturaufgabe, Seeds 740-749), ebenfalls Voraussetzung fuer
       GLEICHWERTIG (das Suchverfahren muss eine bekannte Clifford-Loesung finden koennen).
  M-1  MNIST d_s = acc(suche) - acc(ohne_maske): TRAEGT DIE MASKE?
       Holm-p_TOST < 0,05: Q-0 nicht H1 -> HU (Positivkontrolle Q-0); P-0 durchgefallen -> HU (Positivkontrolle);
                           sonst GLEICHWERTIG (Maske traegt nicht; bei Holm-p < 0,05 zusaetzlich im Text)
       sonst Holm-p < 0,05, d > 0 -> SUCHE_BESSER (die quadratische Maske traegt)
             Holm-p < 0,05, d < 0 -> OHNE_BESSER (dieselben Bloecke ohne Maske sind besser: Suchdefizit/Siegerfluch;
                                     W(V_s, 1) liegt nicht notwendig im Suchraum der Zelle)
       sonst HU.   Weniger als 10 gueltige Seeds -> HU (Vorbedingung).
  M-3  wie M-1 auf Fashion (Familie {M-3}, Gleichwertigkeit braucht P-0F).
  M-2  BESCHREIBEND (kein Urteil, T6): MNIST suche - block2x2_maske (gepaart, TOST-p, Wilcoxon-p ohne Schwelle),
       Anteil Seeds mit wirksamer 2x2-Maske, dasselbe auf „nur_wirksam“, block2x2_maske - block2x2.
  M-4  dasselbe auf Fashion.
  Fehlt ein Datensatz, zaehlt sein Glied mit p = 1.
  M-B  beschreibend: 2x2-Tabelle der Mittel, Wechselwirkung I und Haupteffekte (gepaart), Verteilung von k,
       Anteil der Seeds in Bruhat-Form, maske_wirksam_2x2, Konventions-Empfindlichkeit (alt1/alt2), LOO-Kriterium
       der Arme, Sensitivitaet suche - identitaet, Rekonstruktionsabweichung, Aufwand.

VORHERSAGE (vor dem Lauf, hier festgehalten):
  P-0 BESTANDEN (I-036: +0,93 pp, KI [+0,67; +1,20]); P-0F DURCHGEFALLEN (I-036: +0,34 pp < 0,5 pp).
  M-1 SUCHE_BESSER, d ≈ +1 bis +2 pp: ohne Maske sind die Suchbloecke (k = 4-7) feste mittelgrosse Block-Walsh-
      Transformationen, und solche lagen in I-036 alle UNTER block2x2 (block8x8 0,9256, block4x4/zeilen16 0,911);
      die Maske ist das, was grosse Bloecke brauchbar macht (I-036 T3: 69/70 Suchergebnisse mit echter Maske).
  M-2 (beschreibend) d ≈ +0,9 pp oder mehr: nach T6 ist block2x2_maske in ~60 % der Seeds exakt block2x2, im Rest
      wirkt die uebertragene Maske wie ein festes quadratisches 2x2-Vorzeichenmuster (Erwartung: ≈ block2x2 oder
      schlechter, auch auf „nur_wirksam“).
  => Wechselwirkung deutlich positiv: Maske und Blockwahl tragen NUR GEMEINSAM (die Maske ist an ihre Bloecke
     gebunden). M-3 SUCHE_BESSER (Fashion, k naeher an 8, Bloecke ohne Maske schwach).

FOLGEFRAGEN je Ausgang (fuer agenda.json „folgen“, vom Direktor einzutragen):
  M-1 SUCHE_BESSER  -> Welche Maskeneigenschaft traegt (Rang der quadratischen Form je Block, Transfer der Maske
                       eines Seeds auf andere Seeds mit gleichem V)?
  M-1 GLEICHWERTIG  -> Reicht eine Suche NUR ueber Unterraeume V (ohne Maske, klassisch billig) fuer das Niveau
                       der Clifford-Suche?
  M-1 OHNE_BESSER   -> Suchdefizit: W(V_s, 1) als Start/Politur der Suche; liegt es im Suchraum der Zelle?
  M-1 HU            -> Wiederholung mit 40 Seeds bzw. groesserer Testmenge, falls die Streuung die Ursache ist.
  M-3 (je Ausgang)  -> wie M-1, auf Fashion.
  M-2 beschreibend deutet Blockwahl an (suche - block2x2_maske deutlich > 0, auch auf „nur_wirksam“) -> eigene
                       Karte: Suche NUR ueber Unterraeume V bei fester (kanonischer) Maske.
  P-0 DURCHGEFALLEN -> I-036 W-4 repliziert nicht: Siegerfluch/Seed-Satz pruefen, bevor zerlegt wird.

LAUFZEIT (I-036 gemessen, 4 Prozesse: Suche 378 s MNIST / 339 s Fashion je Seed, ganzer Seed 465 / 492 s mit
  100 Zufalls-Testkoepfen und 17 weiteren Koepfen; Strukturseed ~41 s). Hier entfallen die 100 Zufalls-Testkoepfe
  und die L2-/Dosis-Arme; dazu kommen 7 Koepfe + 7 LOO-Kriterien + 2 Aequivalenzpruefungen. Rauchtest 05.10.
  (1 Prozess, 80 SA-Schritte): alle Koepfe/Kriterien/Pruefungen 4,5 s (MNIST) / 8,5 s (Fashion) je Seed, ganzer
  Rauchtest 65 s. 100 Zufalls-Kriterien ~7 s (0,072 s je Bewertung, I-036). Schaetzung je Seed ~400 s MNIST /
  ~370 s Fashion -> 20·400 + 20·370 + 10·41 ≈ 15.800 CPU-s; INSTITUT_PROZESSE = 4: ~1 h 05 min (bis ~1 h 20 min
  mit Lastungleichgewicht); = 2: ~2 h 15 min.

WIEDERAUFNAHME (Pflicht nach dem Verlust des ersten I-036-Laufs): jeder fertige Seed wird SOFORT in
  ergebnisse/_zwischen_I-036-v2.json geschrieben (atomar: .tmp + os.replace, vom Hauptprozess). Ein Neustart mit
  denselben Einstellungen ueberspringt alle fertigen Seeds; andere Einstellungen -> Fehler (nie mischen). Die
  Datei bleibt nach dem Lauf liegen (der Direktor sichert erst danach) und darf nach dem Sichern geloescht werden.

GRENZEN: Test 5.000 Bilder (0,02 pp je Bild). block2x2_maske haengt an der Normierungskonvention (T4); M-2 ist
  „Blockwahl bei kanonisch uebertragener Maske“. ohne_maske/block2x2_maske sind nicht notwendig als Schaltung der
  4-Lagen-Zelle erreichbar (sie sind Clifford, aber nicht unbedingt im Suchraum) -- die Frage ist, welcher
  Bestandteil der gefundenen Merkmalsabbildung traegt, nicht, was die Zelle sonst noch darstellen kann.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen import clifford_suche_v2 as v2
from abteilungen import walsh_kontrolle as wk
from abteilungen.clifford import LAGEN, QUBITS_JE_DATENSATZ
from abteilungen.clifford_suche import (Bewerter, _bewerte, aequivalent, born, holm, normiere, ridge_loo,
                                        unitaer_clifford, zufall_codes)
from abteilungen.clifford_suche_v2 import tost_p
from abteilungen.walsh_kontrolle import (_affin_linear, _gepaart, codes_block, struktur, wilcoxon_schnell)
from labor.gemeinsam import ERGEBNISSE, speichere, wilcoxon_exakt, zusammenfassung

KARTE = "I-036-v2"
DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
N_SEEDS = 20
SEEDS = {"mnist": tuple(range(700, 720)), "fashion_mnist": tuple(range(720, 740))}
SEEDS_STRUKTUR = tuple(range(740, 750))
ENTWURF_SEEDS = {"mnist": (750, 751), "fashion_mnist": (752, 753)}
ENTWURF_STRUKTUR = (754, 755)
# belegt: alles aus walsh_kontrolle._BELEGT + I-036 (Mess 640-679, Struktur 680-689, Rauchtest 942-947)
_BELEGT = wk._BELEGT | set(range(640, 690)) | set(range(942, 948))
KOPF_L2 = dict(wk.KOPF_L2)                           # 0,003 wie I-036 (aus I-005-v2), kein neues Tuning
ALPHA, TOST_W = 0.05, 0.005
MIN_GUELTIG = 10
N = 8
QUBITS_BLOCK2X2 = wk.BLOECKE["block2x2"]           # (3, 7)
ALT_SEED = 110_000                                   # zufaellige Basen fuer die Konventions-Empfindlichkeit
FAMILIEN = {"MNIST": {"M-1": ("mnist", "suche_minus_ohne_maske")},          # Direktor 05.10.: je ein Glied
            "Fashion": {"M-3": ("fashion_mnist", "suche_minus_ohne_maske")}}
BESCHREIBEND_BLOCK = {"M-2": "mnist", "M-4": "fashion_mnist"}                 # T6: ohne Familienurteil
ARME = ("suche", "ohne_maske", "block2x2_maske", "block2x2")

HYPOTHESEN = {
    "P-0": "Positivkontrolle MNIST (nicht in der Familie): suche - block2x2 reproduziert I-036 W-4: einseitiger exakter "
           "Wilcoxon p < 0,05 und Mittel >= 0,5 pp -> BESTANDEN, sonst DURCHGEFALLEN. Voraussetzung fuer GLEICHWERTIG "
           "in M-1.",
    "P-0F": "wie P-0 auf Fashion; Voraussetzung fuer GLEICHWERTIG in M-3 (Vorhersage: DURCHGEFALLEN, I-036 +0,34 pp).",
    "Q-0": "Positivkontrolle der Suche wie I-005/I-036 Q-0 (Strukturaufgabe, frische Seeds 740-749). Nicht in der "
           "Familie; Voraussetzung fuer GLEICHWERTIG.",
    "M-1": "MNIST: suche (V_s, m_s) vs. ohne_maske (V_s, m = +1), gleicher Kopf, gepaart je Seed, nur Seeds in "
           "Bruhat-Form. Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (wenn Q-0 H1 und P-0 BESTANDEN; sonst HU mit Grund); "
           "sonst Holm-p(Wilcoxon) < 0,05 -> SUCHE_BESSER (Maske traegt) / OHNE_BESSER; sonst HU. Familie {M-1} (ein Glied, "
           "Holm x1). < 10 gueltige Seeds -> HU (Vorbedingung).",
    "M-2": "BESCHREIBEND (Direktor 05.10., vor jeder Zahl; kein Familienurteil): MNIST suche (V_s, m_s) vs. "
           "block2x2_maske (V_b = 2x2-Bloecke, m_s kanonisch normiert), mit Anteil Seeds mit wirksamer 2x2-Maske und "
           "Ergebnis auf 'nur_wirksam'. Grund: in ~60 % der Seeds gilt block2x2_maske = block2x2 exakt, M-2 wiederholte "
           "dort nur I-036 W-4, und das deckt P-0 ab.",
    "M-3": "Fashion (Replikation, eigene Familie {M-3}, Gleichwertigkeit braucht P-0F): wie M-1.",
    "M-4": "BESCHREIBEND: wie M-2 auf Fashion.",
    "M-B": "beschreibend: 2x2-Zerlegung (Wechselwirkung, Haupteffekte), Verteilung von k, Anteil Bruhat-Form, "
           "maske_wirksam_2x2, Konventions-Empfindlichkeit von block2x2_maske, LOO-Kriterium, Aufwand.",
}


# ------------------------------------------------------------------ Theorie: (V, m) -> Merkmalsmatrix
def rref_basis(V) -> list:
    """Basis eines GF(2)-Unterraums (Elemente als Bitmuster) in reduzierter Zeilenstufenform: eindeutig fuer V,
    absteigend sortiert, kein fuehrendes Bit eines Basisvektors kommt in einem anderen vor."""
    basis = []
    for v in (int(u) for u in V):
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    for i in range(len(basis)):
        hb = 1 << (basis[i].bit_length() - 1)
        for j in range(len(basis)):
            if j != i and basis[j] & hb:
                basis[j] ^= basis[i]
    return sorted(basis, reverse=True)


def span(basis) -> np.ndarray:
    V = {0}
    for b in basis:
        V |= {v ^ int(b) for v in V}
    return np.array(sorted(V), dtype=np.int64)


def unterraum_aus_qubits(qubits, n=N) -> np.ndarray:
    """H-Qubits -> Unterraum der Pixelindizes (Qubit q = Bit n-1-q)."""
    return span([1 << (n - 1 - q) for q in qubits])


def _koord(basis) -> np.ndarray:
    """Walsh-Index t (k Bit, hoechstes Bit zu basis[0]) -> Element von V."""
    k = len(basis)
    t = np.arange(2 ** k)
    koord = np.zeros(2 ** k, np.int64)
    for j, b in enumerate(basis):
        koord ^= ((t >> (k - 1 - j)) & 1) * int(b)
    return koord


def nebenklassen(V, d) -> np.ndarray:
    """Kleinster Pixelindex jeder Nebenklasse y + V."""
    return np.unique(np.bitwise_xor.outer(np.arange(d), np.asarray(V)).min(1))


def matrix_vm(V, m, basis=None) -> torch.Tensor:
    """W(V, m): je Nebenklasse a0 + V die 2^k Zeilen  H_k[j, t] · m(a0 ⊕ koord[t]) / 2^(k/2)  auf den Pixeln
    a0 ⊕ koord[t]. Born-Merkmale (x W^T)^2 = (blockweiser Walsh von m ⊙ x)^2. Orthogonal, float32 wie die Zelle."""
    import scipy.linalg
    m = np.asarray(m, dtype=float)
    d = len(m)
    basis = rref_basis(V) if basis is None else list(basis)
    k = len(basis)
    koord = _koord(basis)
    Hk = scipy.linalg.hadamard(2 ** k) / 2 ** (k / 2)
    W = np.zeros((d, d))
    for i, a0 in enumerate(nebenklassen(V, d)):
        idx = int(a0) ^ koord
        W[i * 2 ** k:(i + 1) * 2 ** k, idx] = Hk * m[idx][None, :]
    return torch.tensor(W, dtype=torch.float32)


def kanonische_maske(m_roh, V, basis=None) -> np.ndarray:
    """T4: je Nebenklasse a0 + V den affinen Charakter herausteilen, so dass m(a0) = +1 und m(a0 ⊕ b_j) = +1 fuer
    die Basis b_j (Standard: reduzierte Zeilenstufenform). Uebrig bleibt der quadratische Anteil."""
    m_roh = np.asarray(m_roh, dtype=float)
    basis = rref_basis(V) if basis is None else list(basis)
    k = len(basis)
    koord = _koord(basis)
    bits = (np.arange(2 ** k)[:, None] >> (k - 1 - np.arange(k))[None, :]) & 1
    m = m_roh.copy()
    for a0 in nebenklassen(V, len(m_roh)):
        a0 = int(a0)
        s = np.array([m_roh[a0] * m_roh[a0 ^ int(b)] for b in basis])
        chi = m_roh[a0] * np.prod(np.where(bits == 1, s[None, :], 1.0), axis=1) if k else np.array([m_roh[a0]])
        m[a0 ^ koord] = m_roh[a0 ^ koord] * chi
    return m


def zerlege(U):
    """Suchschaltung in Bruhat-Form -> (V_s, m_s kanonisch, m_roh). Voraussetzung: struktur(U)['bruhat_form']."""
    A = U.double().numpy() if isinstance(U, torch.Tensor) else np.asarray(U, dtype=float)
    d = A.shape[0]
    t0 = np.flatnonzero(np.abs(A[0]) > 1e-6)
    V = np.array(sorted(set((t0 ^ t0[0]).tolist())), dtype=np.int64)
    m_roh = np.zeros(d)
    for r in A:
        t = np.flatnonzero(np.abs(r) > 1e-6)
        if not m_roh[t].any():
            m_roh[t] = np.sign(r[t])
    return V, kanonische_maske(m_roh, V), m_roh


def zufalls_basis(V, rng) -> list:
    """Zufaellige (nicht reduzierte) Basis von V: zufaellige invertierbare Kombination der Stufenbasis."""
    b = rref_basis(V)
    k = len(b)
    while True:
        M = rng.integers(0, 2, (k, k))
        neu = [int(np.bitwise_xor.reduce([b[j] for j in range(k) if M[i, j]] or [0])) for i in range(k)]
        if len(rref_basis(neu)) == k:
            return neu


def maske_wirksam(m, V) -> float:
    """Anteil der Nebenklassen von V, auf denen m NICHT affin-linear ist (dort wirkt die Maske)."""
    d = len(m)
    reps = nebenklassen(V, d)
    g = np.zeros(d)
    wirk = 0
    for a0 in reps:
        g[:] = 0
        g[V] = m[int(a0) ^ V]
        wirk += not _affin_linear(g, V)
    return wirk / len(reps)


# ------------------------------------------------------------------ Arbeiter
def _mess_seed(a):
    """(ds, seed, schritte, politur_max, n_zufall, l2) -> Zeile. Suche bitgleich zu I-036 (_gemeinsame_arme)."""
    ds, s, schritte, politur_max, n_zufall, l2 = a
    n, t0 = QUBITS_JE_DATENSATZ[ds], time.time()
    xtr, ytr, xte, yte = v2._daten(ds, s)
    bew = Bewerter(xtr, ytr, n)
    xs = (normiere(xtr), ytr, normiere(xte), yte, l2)
    z = {"seed": s, "datensatz": ds, "kopf_l2": l2, "n_train": len(ytr), "n_test": len(yte)}
    # wie _gemeinsame_arme: Zufalls-Clifford NUR fuer Skala und Zwischenspeicher (gleicher Cache-Zustand)
    zuf = zufall_codes(s, n_zufall, n)
    zv = [bew(c)[0] for c in zuf]
    z["skala"] = skala = max(float(np.std(zv)), 1e-4)
    ts, b0 = time.time(), bew.bewertungen
    start = np.random.default_rng(v1.START_SEED + s).integers(0, 4, (LAGEN, n))
    best, s_best, _ = v1.suche(bew, start, np.random.default_rng(v1.SA_SEED + s), skala, schritte,
                               politur_max=politur_max)
    z["suche_sekunden"], z["suche_bewertungen"] = time.time() - ts, bew.bewertungen - b0
    z["suche_codes"], z["suche_val"] = best.tolist(), s_best[0]
    tk = time.time()
    z["suche"] = _bewerte(best, n, xs)[0]
    null = np.zeros((LAGEN, n), np.int64)
    z["identitaet"], z["identitaet_val"] = _bewerte(null, n, xs)[0], bew(null)[0]
    cb = codes_block(QUBITS_BLOCK2X2, n)
    z["block2x2"], z["block2x2_val"] = _bewerte(cb, n, xs)[0], bew(cb)[0]
    Us = unitaer_clifford(best, n)
    st = struktur(Us)
    z["suche_struktur"] = st
    z["gueltig"] = bool(st["bruhat_form"])
    if z["gueltig"]:
        V_s, m_s, _ = zerlege(Us)
        V_b = unterraum_aus_qubits(QUBITS_BLOCK2X2, n)
        W_s = matrix_vm(V_s, m_s)
        z["rekonstruktion_ok"] = bool(aequivalent(Us, W_s))
        z["gueltig"] = z["rekonstruktion_ok"]
        z["V_s"], z["maske_s"] = V_s.tolist(), m_s.astype(int).tolist()
        z["maske_anteil_minus"] = float((m_s < 0).mean())
        z["maske_wirksam_vs"] = maske_wirksam(m_s, V_s)
        z["maske_wirksam_2x2"] = maske_wirksam(m_s, V_b)
        rng = np.random.default_rng(ALT_SEED + s)
        arme = {"ohne_maske": (V_s, np.ones(len(m_s))), "block2x2_maske": (V_b, m_s), "suche_rekonstruiert": (V_s, m_s)}
        for j in (1, 2):
            arme[f"block2x2_maske_alt{j}"] = (V_b, kanonische_maske(m_s, V_s, zufalls_basis(V_s, rng)))
        for arm, (V, m) in arme.items():
            W = matrix_vm(V, m)
            z[arm] = _bewerte(W, n, xs)[0]
            z[f"{arm}_val"] = ridge_loo(born(bew.x, W), bew.y)[0]
        for j in (1, 2):
            z[f"alt{j}_gleich_kanonisch"] = bool(aequivalent(matrix_vm(*arme[f"block2x2_maske_alt{j}"]),
                                                             matrix_vm(V_b, m_s)))
    z["kopf_sekunden"] = time.time() - tk
    z["sekunden"] = time.time() - t0
    return z


def _arbeiter(a):
    art, arg = a
    if art == "struktur":
        z = v1._struktur_seed(arg)
        return f"struktur|{z['seed']}", art, z
    z = _mess_seed(arg)
    return f"mess|{z['datensatz']}|{z['seed']}", art, z


def _schluessel(a) -> str:
    art, arg = a
    return f"struktur|{arg[0]}" if art == "struktur" else f"mess|{arg[0]}|{arg[1]}"


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
    """Atomar: erst .tmp vollstaendig schreiben, dann os.replace (ein Abbruch hinterlaesst die alte ODER die neue
    Datei, nie eine halbe)."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"karte": KARTE, "kennung": kennung, "zeilen": zeilen}, ensure_ascii=False))
    os.replace(tmp, pfad)


def _rechne(aufgaben, pfad: Path, kennung: dict, fn=None, prozesse=None) -> dict:
    """Wie labor.gemeinsam.parallel, aber jeder fertige Seed geht SOFORT (im Hauptprozess) in die Zwischendatei;
    bereits fertige Seeds werden uebersprungen. -> {schluessel: zeile}."""
    import multiprocessing as mp
    from labor.gemeinsam import _init_arbeiter
    fn = fn or _arbeiter
    zeilen = _lade_zwischen(pfad, kennung)
    offen = [a for a in aufgaben if _schluessel(a) not in zeilen]
    if len(offen) < len(aufgaben):
        print(f"  Wiederaufnahme: {len(aufgaben) - len(offen)} von {len(aufgaben)} Aufgaben aus {pfad.name}", flush=True)
    prozesse = prozesse or int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))

    def ablegen(erg):
        schl, art, z = erg
        zeilen[schl] = z
        _schreibe_zwischen(pfad, kennung, zeilen)
        print(f"  fertig: {schl} ({len(zeilen)} in Zwischendatei)", flush=True)

    if prozesse <= 1:
        _init_arbeiter()
        for a in offen:
            ablegen(fn(a))
    elif offen:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
            for erg in pool.imap_unordered(fn, offen):
                ablegen(erg)
    return zeilen


def _pruefe_seeds(seeds, struktur_seeds, datensaetze):
    alle = [s for ds in datensaetze for s in seeds[ds]] + list(struktur_seeds)
    entwurf = {s for v in ENTWURF_SEEDS.values() for s in v} | set(ENTWURF_STRUKTUR)
    return len(set(alle)) == len(alle) and not (set(alle) & (_BELEGT | entwurf))


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=N_SEEDS, seeds=None, seeds_struktur=SEEDS_STRUKTUR, datensaetze=DATENSAETZE,
         schritte=v1.SA_SCHRITTE, politur_max=v1.POLITUR_MAX, n_zufall=v1.N_ZUFALL, mit_struktur=True,
         zwischen=None, prozesse=None):
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
    seeds_struktur = list(seeds_struktur) if mit_struktur else []
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "sa_schritte": schritte, "politur_max": politur_max,
               "n_zufall": n_zufall, "kopf_l2": KOPF_L2, "block2x2": list(QUBITS_BLOCK2X2), "alt_seed": ALT_SEED}
    t0 = time.time()
    aufg = [("mess", (ds, s, schritte, politur_max, n_zufall, KOPF_L2[ds])) for ds in datensaetze for s in seeds[ds]]
    aufg += [("struktur", (s, schritte, politur_max, n_zufall)) for s in seeds_struktur]
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zs = [alle[f"struktur|{s}"] for s in seeds_struktur]
    zeilen = {ds: [alle[f"mess|{ds}|{s}"] for s in seeds[ds]] for ds in datensaetze}
    for z in zs:
        print(f"  Struktur {z['seed']}: suche {z['suche']:.4f} lehrer {z['lehrer']:.4f} id {z['identitaet']:.4f} "
              f"aequivalent {z['lehrer_aequivalent']}", flush=True)
    for ds in datensaetze:
        for z in zeilen[ds]:
            st = z["suche_struktur"]
            if z["gueltig"]:
                print(f"  {NAME[ds]} {z['seed']}: suche {z['suche']:.4f} ohne_maske {z['ohne_maske']:.4f} block2x2_maske "
                      f"{z['block2x2_maske']:.4f} block2x2 {z['block2x2']:.4f} | k {st['k']} wirksam_2x2 "
                      f"{z['maske_wirksam_2x2']:.2f} ({z['sekunden']:.0f} s)", flush=True)
            else:
                print(f"  {NAME[ds]} {z['seed']}: AUSGESCHLOSSEN (Bruhat-Form {st['bruhat_form']}, Rekonstruktion "
                      f"{z.get('rekonstruktion_ok')})", flush=True)
    e = {"abteilung": "Clifford", "modul": "maske_block", "karte": KARTE, "version": 1,
         "hypothesen": HYPOTHESEN, "seeds": seeds, "seeds_struktur": seeds_struktur,
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "tau_start": v1.TAU_START, "tau_ende": v1.TAU_ENDE, "ridge_lambda": v1.RIDGE_LAMBDA,
                           "kopf_l2": KOPF_L2, "tost": TOST_W, "min_gueltig": MIN_GUELTIG,
                           "block2x2": list(QUBITS_BLOCK2X2), "alt_seed": ALT_SEED},
         "zwischendatei": str(pfad), "zeilen": zeilen, "zeilen_struktur": zs, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, zs)
    e["urteil"] = urteil(e["auswertung"])
    return e


# ------------------------------------------------------------------ Auswertung
def wilcoxon_einseitig(diff) -> float:
    """Exakter einseitiger Wilcoxon (Alternative: Median > 0), P(W+ >= w) ueber die Zaehl-Rekursion wie
    wilcoxon_schnell. Vortest prueft gegen vollstaendige Aufzaehlung."""
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
    return float(zahl[np.arange(len(zahl)) >= w - 1e-9].sum() / 2.0 ** n)


def _ds_auswertung(zz_alle) -> dict:
    zz = [z for z in zz_alle if z.get("gueltig")]
    r = {"n_alle": len(zz_alle), "n": len(zz), "ausgeschlossen": [z["seed"] for z in zz_alle if not z.get("gueltig")],
         "bruhat_form": int(sum(bool(z["suche_struktur"]["bruhat_form"]) for z in zz_alle))}
    r["genug"] = r["n"] >= MIN_GUELTIG
    if r["n"] < 2:
        return r
    arr = lambda k: np.array([z[k] for z in zz], dtype=float)
    su, om, bm, b2, id_ = arr("suche"), arr("ohne_maske"), arr("block2x2_maske"), arr("block2x2"), arr("identitaet")
    r["suche_minus_ohne_maske"] = {**_gepaart(su - om), "p_tost": tost_p(su - om, TOST_W)}
    r["suche_minus_block2x2_maske"] = {**_gepaart(su - bm), "p_tost": tost_p(su - bm, TOST_W)}
    pk = _gepaart(su - b2)
    pk["p_einseitig"] = wilcoxon_einseitig(su - b2)
    r["positivkontrolle"] = pk
    r["p0_bestanden"] = bool(r["genug"] and pk["p_einseitig"] < ALPHA and pk["mittel"] >= TOST_W)
    b = {arm: zusammenfassung(arr(arm)) for arm in ARME + ("identitaet", "suche_rekonstruiert",
                                                           "block2x2_maske_alt1", "block2x2_maske_alt2")}
    b["wechselwirkung"] = _gepaart(su - om - bm + b2)
    b["haupteffekt_maske"] = _gepaart(((su - om) + (bm - b2)) / 2)
    b["haupteffekt_bloecke"] = _gepaart(((su - bm) + (om - b2)) / 2)
    b["ohne_maske_minus_block2x2"] = _gepaart(om - b2)
    b["block2x2_maske_minus_block2x2"] = _gepaart(bm - b2)
    for j in (1, 2):
        b[f"suche_minus_alt{j}"] = float((su - arr(f"block2x2_maske_alt{j}")).mean())
        b[f"alt{j}_minus_kanonisch"] = float((arr(f"block2x2_maske_alt{j}") - bm).mean())
        b[f"alt{j}_gleich_kanonisch"] = int(sum(z[f"alt{j}_gleich_kanonisch"] for z in zz))
    wirksam = np.array([z["maske_wirksam_2x2"] > 0 for z in zz])
    b["nur_wirksam"] = {"n": int(wirksam.sum())}
    if wirksam.sum() >= 2:
        b["nur_wirksam"]["suche_minus_block2x2_maske"] = _gepaart((su - bm)[wirksam])
        b["nur_wirksam"]["block2x2_maske_minus_block2x2"] = _gepaart((bm - b2)[wirksam])
    b["rekonstruktion_abweichung_max"] = float(np.abs(arr("suche_rekonstruiert") - su).max())
    b["sensitivitaet_suche_minus_id"] = {"mittel": float((su - id_).mean()), "p": wilcoxon_schnell(su - id_)}
    b["val"] = {arm: float(arr(f"{arm}_val").mean()) for arm in ARME}
    b["val_ohne_maske_besser"] = int(sum(z["ohne_maske_val"] > z["suche_val"] for z in zz))
    b["val_block2x2_maske_besser"] = int(sum(z["block2x2_maske_val"] > z["suche_val"] for z in zz))
    ks = [z["suche_struktur"]["k"] for z in zz_alle]
    b["struktur"] = {"k": ks, "k_verteilung": {str(k): ks.count(k) for k in sorted(set(ks))},
                     "k_mittel": float(np.mean(ks)), "bruhat_anteil": r["bruhat_form"] / max(1, r["n_alle"]),
                     "maske_noetig": int(sum(bool(z["suche_struktur"]["maske_noetig"]) for z in zz)),
                     "maske_wirksam_2x2": float(np.mean([z["maske_wirksam_2x2"] for z in zz])),
                     "maske_wirksam_vs": float(np.mean([z["maske_wirksam_vs"] for z in zz])),
                     "maske_anteil_minus": float(np.mean([z["maske_anteil_minus"] for z in zz]))}
    for k in ("suche_bewertungen", "suche_sekunden", "kopf_sekunden", "sekunden"):
        b[k] = float(np.mean([z[k] for z in zz]))
    r["beschreibend"] = b
    return r


def auswertung(zeilen: dict, zs) -> dict:
    a = {"datensaetze": {ds: _ds_auswertung(zz) for ds, zz in zeilen.items() if zz}, "n_struktur": len(zs)}
    if zs:
        a["Q0"] = v1.auswertung([], zs)["Q0"]
    a["holm"], a["holm_tost"] = {}, {}
    for fam, glieder in FAMILIEN.items():
        p, pt = {}, {}
        for h, (ds, k) in glieder.items():
            r = a["datensaetze"].get(ds, {})
            d = r.get(k) if r.get("genug") else None
            p[h], pt[h] = (d["p"], d["p_tost"]) if d else (1.0, 1.0)
        a["holm"].update(holm(p))
        a["holm_tost"].update(holm(pt))
    return a


def _zahlen(ds, d, ph, pth):
    return (f"{NAME[ds]}: d = {d['mittel'] * 100:+.2f} pp, 90-%-KI [{d['ki_unten'] * 100:+.2f}; "
            f"{d['ki_oben'] * 100:+.2f}], TOST(+-{TOST_W * 100:.1f} pp) p = {d['p_tost']:.4f}, Holm-p_TOST = "
            f"{pth:.4f}, Wilcoxon p = {d['p']:.4f}, Holm-p = {ph:.4f}, {d.get('positiv', 0)}/{d['n']} Seeds positiv")


SPEZ = {
    "suche_minus_ohne_maske": {
        "gleich": "GLEICHWERTIG: die gesuchte Vorzeichenmaske traegt nicht -- dieselben Bloecke V_s ohne Maske leisten "
                  "dasselbe",
        "besser": "SUCHE_BESSER: die gesuchte Vorzeichenmaske traegt -- ohne ihren quadratischen Anteil verlieren "
                  "dieselben Bloecke V_s",
        "schlechter": "OHNE_BESSER: dieselben Bloecke V_s ohne Maske sind besser als die Suche (Suchdefizit oder "
                      "Siegerfluch; W(V_s, 1) liegt nicht notwendig im Suchraum)",
        "val": "val_ohne_maske_besser", "val_name": "ohne_maske"},
}


def _gueltig_text(r):
    aus = r["ausgeschlossen"]
    return (f"{r['n']}/{r['n_alle']} Seeds gueltig (Bruhat-Form {r['bruhat_form']}/{r['n_alle']}), "
            f"{len(aus)} ausgeschlossen{' ' + str(aus) if aus else ''}")


def _urteil_m(h, ds, k, a, q0_h1) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    if not r["genug"]:
        return (f"HU (Vorbedingung): {NAME[ds]} nur {r['n']} gueltige Seeds (< {MIN_GUELTIG}) -- " + _gueltig_text(r))
    sp, d, ph, pth = SPEZ[k], r[k], a["holm"][h], a["holm_tost"][h]
    b, pk = r["beschreibend"], r["positivkontrolle"]
    t = (_zahlen(ds, d, ph, pth) + f"; Positivkontrolle suche - block2x2 {pk['mittel'] * 100:+.2f} pp (einseitig p = "
         f"{pk['p_einseitig']:.4f}); Wechselwirkung {b['wechselwirkung']['mittel'] * 100:+.2f} pp; {sp['val_name']} im "
         f"LOO-Kriterium besser auf {b[sp['val']]}/{r['n']}; k Mittel {b['struktur']['k_mittel']:.2f}; " + _gueltig_text(r))
    if pth < ALPHA:
        if not q0_h1:
            return "HU (Positivkontrolle Q-0): gleich im Rahmen +-0,5 pp, aber Q-0 nicht H1 -- " + t
        if not r["p0_bestanden"]:
            return ("HU (Positivkontrolle): gleich im Rahmen +-0,5 pp, aber suche - block2x2 reproduziert den I-036-"
                    "Effekt nicht (einseitig p < 0,05 und >= 0,5 pp) -- " + t)
        extra = " (Unterschied zusaetzlich nachweisbar, aber kleiner als die Grenze)" if ph < ALPHA else ""
        return sp["gleich"] + f" (+-0,5 pp){extra} -- " + t
    if ph < ALPHA and d["mittel"] > 0:
        return sp["besser"] + " -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return sp["schlechter"] + " -- " + t
    return "HU: " + t


def _beschreibung_block(ds, a) -> str:
    """M-2/M-4: beschreibend (T6), Zahlen ohne Schwelle und ohne Familienurteil."""
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"beschreibend: keine {NAME[ds]}-Seeds"
    if "beschreibend" not in r:
        return f"beschreibend: {NAME[ds]} zu wenige gueltige Seeds -- " + _gueltig_text(r)
    d, b = r["suche_minus_block2x2_maske"], r["beschreibend"]
    nw = b["nur_wirksam"]
    g = lambda x: (f"{x['mittel'] * 100:+.2f} pp, 90-%-KI [{x['ki_unten'] * 100:+.2f}; {x['ki_oben'] * 100:+.2f}], "
                   f"Wilcoxon p = {x['p']:.4f}, {x['positiv']}/{x['n']} positiv")
    t = (f"beschreibend (kein Familienurteil, T6): {NAME[ds]}: suche - block2x2_maske {g(d)}, TOST p = {d['p_tost']:.4f}; "
         f"wirksame 2x2-Maske in {nw['n']}/{r['n']} Seeds (sonst block2x2_maske = block2x2 exakt); ")
    if nw["n"] >= 2:
        t += (f"nur_wirksam: suche - block2x2_maske {g(nw['suche_minus_block2x2_maske'])}, block2x2_maske - block2x2 "
              f"{g(nw['block2x2_maske_minus_block2x2'])}")
    else:
        t += "nur_wirksam: zu wenige Seeds"
    return t + "; " + _gueltig_text(r)


def _urteil_p0(ds, a) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"DURCHGEFALLEN (Vorbedingung): keine {NAME[ds]}-Seeds"
    if not r["genug"]:
        return f"DURCHGEFALLEN (Vorbedingung): nur {r['n']} gueltige Seeds -- " + _gueltig_text(r)
    pk = r["positivkontrolle"]
    t = (f"{NAME[ds]}: suche - block2x2 = {pk['mittel'] * 100:+.2f} pp, 90-%-KI [{pk['ki_unten'] * 100:+.2f}; "
         f"{pk['ki_oben'] * 100:+.2f}], einseitiger Wilcoxon p = {pk['p_einseitig']:.4f}, {pk['positiv']}/{pk['n']} "
         f"positiv (I-036: MNIST +0,93 pp 19/20, Fashion +0,34 pp 14/20)")
    if r["p0_bestanden"]:
        return "BESTANDEN: der I-036-Effekt (Suche vor festem 2x2-Block-Walsh) ist reproduziert -- " + t
    return "DURCHGEFALLEN: der I-036-Effekt ist nicht reproduziert (einseitig p < 0,05 und >= 0,5 pp) -- " + t


def urteil(a: dict) -> dict:
    u = {"Q-0": (v1.urteil({"Q0": a["Q0"], "n_struktur": a["n_struktur"]})["Q-0"] if "Q0" in a
                 else "HU (Vorbedingung): keine Strukturseeds")}
    q0_h1 = u["Q-0"].startswith("H1")
    u["P-0"] = _urteil_p0("mnist", a)
    u["P-0F"] = _urteil_p0("fashion_mnist", a)
    for glieder in FAMILIEN.values():
        for h, (ds, k) in glieder.items():
            u[h] = _urteil_m(h, ds, k, a, q0_h1)
    for h, ds in BESCHREIBEND_BLOCK.items():
        u[h] = _beschreibung_block(ds, a)
    tb = []
    for ds, r in a["datensaetze"].items():
        if "beschreibend" not in r:
            tb.append(f"{NAME[ds]}: zu wenige gueltige Seeds ({_gueltig_text(r)})")
            continue
        b, st = r["beschreibend"], r["beschreibend"]["struktur"]
        g = lambda x: f"{x['mittel'] * 100:+.2f} pp ({x['positiv']}/{x['n']}, p = {x['p']:.4f})"
        nw = b["nur_wirksam"]
        tw = (f"nur Seeds mit wirksamer 2x2-Maske ({nw['n']}/{r['n']}): suche - block2x2_maske "
              f"{g(nw['suche_minus_block2x2_maske'])}, block2x2_maske - block2x2 {g(nw['block2x2_maske_minus_block2x2'])}"
              if nw["n"] >= 2 else f"wirksame 2x2-Maske nur in {nw['n']}/{r['n']} Seeds")
        tb.append(f"{NAME[ds]}: 2x2-Tabelle suche {b['suche']['mittel']:.4f} | ohne_maske {b['ohne_maske']['mittel']:.4f} "
                  f"| block2x2_maske {b['block2x2_maske']['mittel']:.4f} | block2x2 {b['block2x2']['mittel']:.4f} "
                  f"(id {b['identitaet']['mittel']:.4f}); Wechselwirkung {g(b['wechselwirkung'])}; Haupteffekt Maske "
                  f"{g(b['haupteffekt_maske'])}, Bloecke {g(b['haupteffekt_bloecke'])}; ohne_maske - block2x2 "
                  f"{g(b['ohne_maske_minus_block2x2'])}; block2x2_maske - block2x2 {g(b['block2x2_maske_minus_block2x2'])}; "
                  f"Konvention: alt1/alt2 - kanonisch {b['alt1_minus_kanonisch'] * 100:+.2f}/"
                  f"{b['alt2_minus_kanonisch'] * 100:+.2f} pp (gleiche Merkmale in {b['alt1_gleich_kanonisch']}/"
                  f"{b['alt2_gleich_kanonisch']} Seeds); k {st['k_verteilung']} (Mittel {st['k_mittel']:.2f}), Bruhat-"
                  f"Anteil {st['bruhat_anteil']:.2f}, Maske noetig {st['maske_noetig']}/{r['n']}, Maske wirksam auf "
                  f"2x2-Bloecken {st['maske_wirksam_2x2']:.2f}, auf V_s-Bloecken {st['maske_wirksam_vs']:.2f}; {tw}; "
                  f"Rekonstruktion max |Abweichung| {b['rekonstruktion_abweichung_max'] * 100:.2f} pp; Sensitivitaet "
                  f"suche - id {b['sensitivitaet_suche_minus_id']['mittel'] * 100:+.2f} pp; "
                  f"{b['suche_bewertungen']:.0f} Bew., {b['sekunden']:.0f} s je Seed")
    u["M-B"] = "beschreibend: " + (" | ".join(tb) if tb else "keine Mess-Seeds")
    return u


# ------------------------------------------------------------------ Vortest: Bausteine
def matrix_explizit(V, m) -> torch.Tensor:
    """Unabhaengige Konstruktion fuer den Vortest (keine Basis, keine Koordinaten): je Nebenklasse alle 256
    Charaktere (-1)^(w·y) mal m(y) auf der Nebenklasse, normiert; doppelte Zeilen (bis auf Vorzeichen) entfernt."""
    m = np.asarray(m, dtype=float)
    d = len(m)
    pop = np.array([bin(i).count("1") for i in range(d)])
    V = np.asarray(V)
    zeilen, gesehen = [], set()
    for y in range(d):
        nk = np.sort(y ^ V)
        if int(nk[0]) in gesehen:
            continue
        gesehen.add(int(nk[0]))
        R = np.zeros((d, d))
        R[:, nk] = (-1.0) ** pop[np.bitwise_and.outer(np.arange(d), nk)] * m[nk][None, :] / math.sqrt(len(V))
        R *= np.sign(R[:, nk[0]])[:, None]
        zeilen.append(np.unique(np.round(R, 9), axis=0))
    return torch.tensor(np.concatenate(zeilen), dtype=torch.float64)


def _vortest_bausteine() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(3602)
    d = 2 ** N
    ident = torch.eye(d)
    # (B1) Stufenbasis eindeutig; V_b = {0, 1, 16, 17} = Unterraum der block2x2-Schaltung
    for _ in range(20):
        V = span(rng.integers(1, d, rng.integers(0, 6)))
        b = rref_basis(V)
        if rref_basis(rng.permutation(V)) != b or not np.array_equal(span(b), V) or len(b) != round(math.log2(len(V))):
            fehler(f"rref_basis nicht eindeutig fuer {V.tolist()}")
            break
    V_b = unterraum_aus_qubits(QUBITS_BLOCK2X2)
    U_b = unitaer_clifford(codes_block(QUBITS_BLOCK2X2), N)
    if V_b.tolist() != [0, 1, 16, 17] or zerlege(U_b)[0].tolist() != V_b.tolist():
        fehler(f"V_b falsch: {V_b.tolist()} vs. Schaltung {zerlege(U_b)[0].tolist()}")
    # (B2) matrix_vm = unabhaengige Konstruktion, orthogonal, fuer zufaellige V (k = 0..8) und Masken
    for k_ziel in range(N + 1):
        V = np.array([0])
        while len(V) < 2 ** k_ziel:
            V = span(rref_basis(list(V) + [int(rng.integers(1, d))]))
        m = rng.choice([-1.0, 1.0], d)
        W, E = matrix_vm(V, m), matrix_explizit(V, m)
        if E.shape != (d, d) or not aequivalent(W, E) or (W.double() @ W.double().T - torch.eye(d)).abs().max() > 1e-5:
            fehler(f"matrix_vm != explizite Konstruktion (k = {k_ziel})")
    # (B3) Zufalls-Clifford in Bruhat-Form: W(V_s, m_s) = U bis auf Vorzeichen/Reihenfolge; Maske kanonisch
    #      eindeutig (Zeilenreihenfolge egal); m_s ≡ 1 <=> keine Maske noetig; alt. Basen ebenfalls = U
    xn = normiere(torch.tensor(rng.random((64, d)) ** 3, dtype=torch.float32))
    mit_maske = 0
    for _ in range(12):
        U = unitaer_clifford(rng.integers(0, 4, (LAGEN, N)), N)
        s = struktur(U)
        if not s["bruhat_form"]:
            fehler(f"Zufalls-Clifford nicht in Bruhat-Form: {s}")
            continue
        V_s, m_s, m_roh = zerlege(U)
        W_s = matrix_vm(V_s, m_s)
        f1, f2 = np.sort(born(xn, U).numpy(), 1), np.sort(born(xn, W_s).numpy(), 1)
        if not aequivalent(U, W_s) or np.abs(f1 - f2).max() > 1e-5 or len(V_s) != 2 ** s["k"]:
            fehler("W(V_s, m_s) nicht aequivalent zur Schaltung")
        V2, m2, _ = zerlege(U[torch.tensor(rng.permutation(d))])
        if not (np.array_equal(V2, V_s) and np.array_equal(m2, m_s)):
            fehler("kanonische Maske haengt von der Zeilenreihenfolge ab")
        if bool(np.all(m_s == 1)) != (not s["maske_noetig"]):
            fehler(f"m_s ≡ 1 passt nicht zu maske_noetig = {s['maske_noetig']}")
        alt = kanonische_maske(m_s, V_s, zufalls_basis(V_s, rng))
        if not aequivalent(U, matrix_vm(V_s, alt)):
            fehler("Maske mit anderer Basis normiert ist auf V_s nicht merkmalsgleich")
        # (B4) ohne_maske = W(V_s, 1) = W(V_s, affiner Charakter) = explizit; mit echter Maske verschieden
        pop = np.array([bin(i).count("1") for i in range(d)])
        chi = np.ones(d)
        for a0 in nebenklassen(V_s, d):
            w = int(rng.integers(d))
            chi[int(a0) ^ V_s] = rng.choice([-1.0, 1.0]) * (-1.0) ** pop[w & V_s]
        W_om = matrix_vm(V_s, np.ones(d))
        if not (aequivalent(W_om, matrix_vm(V_s, chi)) and aequivalent(W_om, matrix_explizit(V_s, np.ones(d)))
                and aequivalent(W_om, matrix_vm(V_s, kanonische_maske(chi, V_s)))):
            fehler("ohne_maske != Weglassen des quadratischen Anteils (affiner Charakter)")
        if np.any(kanonische_maske(chi, V_s) != 1):
            fehler("kanonische Maske eines affinen Charakters ist nicht ≡ 1")
        if s["maske_noetig"]:
            mit_maske += 1
            if aequivalent(W_om, W_s):
                fehler("mit echter Maske ist ohne_maske trotzdem = suche")
        # (B5) block2x2_maske = explizit konstruiert
        if not aequivalent(matrix_vm(V_b, m_s), matrix_explizit(V_b, m_s)):
            fehler("block2x2_maske != explizite Konstruktion")
    if mit_maske == 0:
        fehler("keine Zufalls-Clifford mit Maske -- (B4) nicht scharf geprueft")
    # (B5) block2x2_maske mit m ≡ 1 = block2x2-Schaltung; Maske affin auf 2x2-Bloecken -> ebenfalls block2x2;
    #      quadratische Maske auf 2x2 -> verschieden, maske_wirksam erkennt sie
    if not aequivalent(matrix_vm(V_b, np.ones(d)), U_b):
        fehler("block2x2_maske mit m ≡ 1 ist nicht der block2x2-Arm")
    y = np.arange(d)
    lin = (-1.0) ** (((y >> 3) & 1) + (y & 1) * ((y >> 6) & 1))      # auf jedem 2x2-Block affin
    quad = (-1.0) ** ((y & 1) * ((y >> 4) & 1))                        # auf jedem 2x2-Block quadratisch
    if not aequivalent(matrix_vm(V_b, lin), U_b) or maske_wirksam(lin, V_b) != 0.0:
        fehler("auf 2x2 affine Maske veraendert block2x2")
    if aequivalent(matrix_vm(V_b, quad), U_b) or maske_wirksam(quad, V_b) != 1.0:
        fehler("quadratische 2x2-Maske nicht erkannt")
    # (B6) k = 0: jedes W({0}, m) = Identitaet; Schaltung mit k = 0 -> suche = ohne_maske = id, block2x2_maske = block2x2
    if not aequivalent(matrix_vm(np.array([0]), rng.choice([-1.0, 1.0], d)), ident):
        fehler("k = 0 ergibt nicht die Identitaet")
    c = np.zeros((LAGEN, N), np.int64)
    c[0] = c[1] = 1                                                    # H in zwei Lagen hebt sich auf
    U0 = unitaer_clifford(c, N)
    V0, m0, _ = zerlege(U0)
    if not (len(V0) == 1 and np.all(m0 == 1) and aequivalent(U0, ident) and aequivalent(matrix_vm(V0, m0), ident)
            and aequivalent(matrix_vm(V0, np.ones(d)), ident) and aequivalent(matrix_vm(V_b, m0), U_b)):
        fehler("Sonderfall k_s = 0 falsch")
    # (B7) Kopf auf W(V_s, m_s) = Kopf auf der Schaltung (Merkmalspermutation; Numerik)
    xtr = torch.tensor(rng.random((400, d)) ** 3, dtype=torch.float32)
    xte = torch.tensor(rng.random((300, d)) ** 3, dtype=torch.float32)
    ytr, yte = torch.tensor(rng.integers(0, 10, 400)), torch.tensor(rng.integers(0, 10, 300))
    ytr[:200] = (xtr[:200, :10].argmax(1))                             # etwas lernbare Struktur
    yte[:150] = (xte[:150, :10].argmax(1))
    xs = (normiere(xtr), ytr, normiere(xte), yte, 0.003)
    codes = rng.integers(0, 4, (LAGEN, N))
    Uc = unitaer_clifford(codes, N)
    a1, a2 = _bewerte(codes, N, xs)[0], _bewerte(matrix_vm(*zerlege(Uc)[:2]), N, xs)[0]
    if abs(a1 - a2) > 2 / 300:
        fehler(f"Kopf auf W(V_s, m_s) {a2:.4f} != Kopf auf Schaltung {a1:.4f}")
    # (B8) Statistik: einseitiger Wilcoxon = Aufzaehlung; Familien; Seeds
    from itertools import product
    for n in (1, 4, 7, 10):
        for _ in range(8):
            dd = rng.normal(rng.normal(0, 0.003), 0.004, n)
            r = np.argsort(np.argsort(np.abs(dd))) + 1
            w = r[dd > 0].sum()
            p_auf = np.mean([sum(rr for rr, sg in zip(r, v) if sg) >= w for v in product([0, 1], repeat=n)])
            if abs(wilcoxon_einseitig(dd) - p_auf) > 1e-12:
                fehler(f"wilcoxon_einseitig != Aufzaehlung (n = {n})")
                break
    if abs(wilcoxon_einseitig(np.full(20, 0.01)) - 1 / 2 ** 20) > 1e-15 or wilcoxon_einseitig(np.full(20, -0.01)) != 1.0:
        fehler("einseitiger Wilcoxon an den Enden falsch")
    dd = rng.normal(0.001, 0.004, 11)
    if abs(wilcoxon_schnell(dd) - wilcoxon_exakt(dd)) > 1e-12:
        fehler("wilcoxon_schnell != wilcoxon_exakt")
    if (set(FAMILIEN["MNIST"]) != {"M-1"} or set(FAMILIEN["Fashion"]) != {"M-3"}
            or set(BESCHREIBEND_BLOCK) != {"M-2", "M-4"}
            or {ds for g in FAMILIEN.values() for ds, _ in g.values()} != set(DATENSAETZE)):
        fehler("Familien falsch")
    for n in (10, 20):
        if not _pruefe_seeds({ds: SEEDS[ds][:n] for ds in DATENSAETZE}, SEEDS_STRUKTUR, DATENSAETZE):
            fehler(f"Seeds (n = {n}) nicht frisch/disjunkt")
    for schlecht in ({"mnist": (640,), "fashion_mnist": (721,)}, {"mnist": (942,), "fashion_mnist": (721,)},
                     {"mnist": (750,), "fashion_mnist": (721,)}, {"mnist": (720,), "fashion_mnist": (720,)},
                     {"mnist": (300,), "fashion_mnist": (721,)}, {"mnist": (685,), "fashion_mnist": (721,)}):
        if _pruefe_seeds(schlecht, (), DATENSAETZE):
            fehler(f"belegte/doppelte Seeds durchgelassen: {schlecht}")
    try:
        lauf(n_seeds=15)
        fehler("n_seeds = 15 nicht abgelehnt")
    except ValueError:
        pass
    return ok


# ------------------------------------------------------------------ Vortest: Wiederaufnahme
def _test_arbeiter(a):
    """Ersatz-Arbeiter fuer den Wiederaufnahme-Test (Modulebene, damit spawn ihn findet)."""
    art, arg = a
    if arg[-1] == "abbruch":
        raise RuntimeError("simulierter Abbruch")
    if art == "struktur":
        return f"struktur|{arg[0]}", art, {"seed": arg[0], "pid": os.getpid()}
    return f"mess|{arg[0]}|{arg[1]}", art, {"seed": arg[1], "datensatz": arg[0], "wert": arg[1] * 0.5}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER (Wiederaufnahme):", t)

    kennung = {"karte": KARTE, "test": 1}
    with tempfile.TemporaryDirectory() as tmp:
        pfad = Path(tmp) / "_zwischen_test.json"
        aufg = [("mess", ("mnist", s, "x")) for s in (1, 2, 3)] + [("struktur", (9, "x"))]
        kaputt = aufg[:2] + [("mess", ("mnist", 3, "abbruch"))] + aufg[3:]
        try:
            _rechne(kaputt, pfad, kennung, fn=_test_arbeiter, prozesse=1)
            fehler("simulierter Abbruch nicht ausgeloest")
        except RuntimeError:
            pass
        roh = json.loads(pfad.read_text())
        if set(roh["zeilen"]) != {"mess|mnist|1", "mess|mnist|2"} or Path(str(pfad) + ".tmp").exists():
            fehler(f"nach Abbruch falscher Inhalt: {sorted(roh['zeilen'])}")
        gerufen = []

        def zaehler(a):
            gerufen.append(_schluessel(a))
            return _test_arbeiter(a)

        alle = _rechne(aufg, pfad, kennung, fn=zaehler, prozesse=1)
        if sorted(gerufen) != ["mess|mnist|3", "struktur|9"] or len(alle) != 4 or alle["mess|mnist|2"]["wert"] != 1.0:
            fehler(f"Neustart rechnet {gerufen} statt nur der offenen Seeds")
        gerufen.clear()
        _rechne(aufg, pfad, kennung, fn=zaehler, prozesse=1)
        if gerufen:
            fehler("vollstaendige Zwischendatei rechnet erneut")
        try:
            _rechne(aufg, pfad, {"karte": KARTE, "test": 2}, fn=_test_arbeiter, prozesse=1)
            fehler("andere Einstellungen nicht abgelehnt")
        except ValueError:
            pass
        # Parallelpfad (spawn, imap_unordered): Abbruch -> Teilergebnisse gesichert, Neustart ergaenzt
        pfad2 = Path(tmp) / "_zwischen_par.json"
        par = [("mess", ("fashion_mnist", s, "x")) for s in range(6)]
        alle = _rechne(par, pfad2, kennung, fn=_test_arbeiter, prozesse=2)
        if len(alle) != 6 or len(json.loads(pfad2.read_text())["zeilen"]) != 6:
            fehler("Parallelpfad schreibt nicht alle Seeds")
    return ok


# ------------------------------------------------------------------ Vortest: Urteile
def _synth(rng, d_om=0.0, d_bm=0.0, d_b2=0.009, sd=0.001, n=20, ungueltig=0):
    """d_om = suche - ohne_maske, d_bm = suche - block2x2_maske, d_b2 = suche - block2x2 (Positivkontrolle)."""
    zeilen = []
    for i in range(n):
        su = 0.935 + rng.normal(0, 0.002)
        k = int(rng.integers(4, 8))
        gut = i >= ungueltig
        z = {"seed": i, "datensatz": "x", "suche": su, "block2x2": su - d_b2 + rng.normal(0, 0.001),
             "identitaet": su - 0.035, "suche_val": su, "block2x2_val": su - 0.01, "suche_bewertungen": 5000,
             "suche_sekunden": 380.0, "kopf_sekunden": 20.0, "sekunden": 420.0, "gueltig": gut,
             "suche_struktur": {"k": k, "bruhat_form": gut, "maske_noetig": gut, "walsh_aequivalent": False}}
        if gut:
            z.update({"ohne_maske": su - d_om + rng.normal(0, sd), "block2x2_maske": su - d_bm + rng.normal(0, sd),
                      "suche_rekonstruiert": su, "rekonstruktion_ok": True, "maske_wirksam_2x2": float(i % 2),
                      "maske_wirksam_vs": 1.0, "maske_anteil_minus": 0.5, "alt1_gleich_kanonisch": False,
                      "alt2_gleich_kanonisch": True})
            z["block2x2_maske_alt1"] = z["block2x2_maske"] - 0.001
            z["block2x2_maske_alt2"] = z["block2x2_maske"]
            for arm in ARME[1:]:
                z[f"{arm}_val"] = su - 0.01
            z["ohne_maske_val"] = z["block2x2_maske_val"] = su - 0.01
        zeilen.append(z)
    return zeilen


def _vortest_urteile(ziehungen=5) -> bool:
    ok = True
    rng = np.random.default_rng(3603)
    gut = lambda: v1._synth_struktur(rng, 0.925)
    schlecht = lambda: v1._synth_struktur(rng, 0.79)
    # (Name, Struktur, MNIST-Parameter, Fashion-Parameter, {Hypothese: (Marke, [Textteile])}, Toleranz)
    B = "beschreibend (kein Familienurteil"
    faelle = [
        ("Maske traegt", gut, {"d_om": 0.012}, {"d_om": 0.012},
         {"P-0": ("BESTANDEN", ["MNIST", "I-036"]), "P-0F": ("BESTANDEN", ["Fashion"]), "Q-0": ("H1", []),
          "M-1": ("SUCHE_BESSER", ["Vorzeichenmaske traegt", "MNIST", "20/20 Seeds gueltig", "Holm-p_TOST"]),
          "M-3": ("SUCHE_BESSER", ["Vorzeichenmaske", "Fashion"]),
          "M-2": (B, ["MNIST", "wirksame 2x2-Maske in 10/20", "nur_wirksam: suche - block2x2_maske", "TOST p"]),
          "M-4": (B, ["Fashion", "nur_wirksam"])}, 0),
        ("Maske traegt nicht", gut, {"d_bm": 0.012}, {"d_bm": 0.012},
         {"M-1": ("GLEICHWERTIG", ["Vorzeichenmaske traegt nicht", "MNIST"]), "M-3": ("GLEICHWERTIG", ["Fashion"]),
          "M-2": (B, ["+1.2"]), "M-4": (B, ["+1.2"])}, 0),
        ("ohne Maske besser", gut, {"d_om": -0.01}, {"d_om": -0.01},
         {"M-1": ("OHNE_BESSER", ["Suchdefizit", "MNIST"]), "M-3": ("OHNE_BESSER", ["Fashion"]),
          "M-2": (B, [])}, 0),
        ("Streuung", gut, {"sd": 0.03}, {"sd": 0.03},
         {"M-1": ("HU:", ["MNIST"]), "M-3": ("HU:", ["Fashion"]), "M-2": (B, [])}, 1),
        ("Positivkontrolle fehlt (d = 0)", gut, {"d_b2": 0.0}, {"d_b2": 0.0},
         {"P-0": ("DURCHGEFALLEN:", ["MNIST"]), "P-0F": ("DURCHGEFALLEN:", ["Fashion"]),
          "M-1": ("HU (Positivkontrolle)", ["reproduziert den I-036-Effekt nicht", "MNIST"]),
          "M-3": ("HU (Positivkontrolle)", ["Fashion"]), "M-2": (B, [])}, 0),
        ("Positivkontrolle zu klein (+0,3 pp)", gut, {"d_b2": 0.003}, {"d_b2": 0.003},
         {"P-0": ("DURCHGEFALLEN:", ["+0.3"]), "M-1": ("HU (Positivkontrolle)", ["I-036"]),
          "M-3": ("HU (Positivkontrolle)", ["Fashion"])}, 0),
        ("Positivkontrolle fehlt, Unterschied bleibt", gut, {"d_b2": 0.0, "d_om": 0.012}, {"d_b2": 0.0, "d_om": -0.01},
         {"P-0": ("DURCHGEFALLEN:", []), "M-1": ("SUCHE_BESSER", ["Vorzeichenmaske traegt"]),
          "M-3": ("OHNE_BESSER", ["Fashion"])}, 0),
        ("Q-0 faellt", schlecht, {}, {},
         {"Q-0": ("H0", []), "M-1": ("HU (Positivkontrolle Q-0)", ["MNIST"]),
          "M-3": ("HU (Positivkontrolle Q-0)", ["Fashion"]), "M-2": (B, [])}, 1),
        ("ohne Strukturseeds", None, {}, {},
         {"Q-0": ("HU (Vorbedingung)", []), "M-1": ("HU (Positivkontrolle Q-0)", []),
          "M-3": ("HU (Positivkontrolle Q-0)", [])}, 0),
        ("3 ausgeschlossen", gut, {"d_om": 0.012, "ungueltig": 3}, {},
         {"M-1": ("SUCHE_BESSER", ["17/20 Seeds gueltig", "3 ausgeschlossen [0, 1, 2]"]),
          "M-2": (B, ["17/20 Seeds gueltig", "wirksame 2x2-Maske in 9/17"])}, 0),
        ("zu wenige gueltig", gut, {"ungueltig": 12}, {"ungueltig": 19},
         {"M-1": ("HU (Vorbedingung)", ["nur 8 gueltige Seeds", "MNIST"]), "P-0": ("DURCHGEFALLEN (Vorbedingung)", []),
          "M-2": (B, ["nur_wirksam: suche"]), "M-3": ("HU (Vorbedingung)", ["Fashion", "nur 1"]),
          "M-4": ("beschreibend: Fashion zu wenige gueltige Seeds", [])}, 0),
        ("klein, aber gleich", gut, {"d_om": 0.002, "sd": 0.0003}, {"d_om": -0.002, "sd": 0.0003},
         {"M-1": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"]), "M-3": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"])}, 0),
        ("nur MNIST", gut, {"d_om": 0.012}, None,
         {"M-1": ("SUCHE_BESSER", []), "M-3": ("HU: keine Fashion", []), "M-4": ("beschreibend: keine Fashion", []),
          "P-0F": ("DURCHGEFALLEN (Vorbedingung)", ["keine Fashion"])}, 0),
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
                    print(f"  Vortest [{name}]: {h} -> {u[h][:200]}")
            if not (u["M-B"].startswith("beschreibend") and all(isinstance(v, str) for v in u.values())):
                ok = False
                print(f"  VORTEST FEHLER [{name}]: M-B oder Urteil kein String")
            try:
                json.dumps({"auswertung": a, "urteil": u, "zeilen": zeilen})
            except TypeError as err:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: nicht JSON-serialisierbar", err)
        for h, f in fehl.items():
            if f > tol:
                ok = False
                print(f"  VORTEST FEHLER [{name}]: {h} {f}/{ziehungen} falsch")
    for _ in range(ziehungen):
        # grosse Streuung, wahre Differenz 0 -> NIE GLEICHWERTIG
        u = urteil(auswertung({ds: _synth(rng, sd=0.03) for ds in DATENSAETZE}, gut()))
        if any(u[h].startswith("GLEICHWERTIG") for h in ("M-1", "M-3")) or not all(
                u[h].startswith("beschreibend") for h in ("M-2", "M-4", "M-B")):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung")
        # Familien getrennt: Fashion-p veraendert die MNIST-Holm-Werte nicht
        zm = _synth(rng, d_om=0.004, sd=0.006)
        a1 = auswertung({"mnist": zm}, gut())
        a2 = auswertung({"mnist": zm, "fashion_mnist": _synth(rng, d_om=0.02)}, gut())
        if any(abs(a1["holm"][h] - a2["holm"][h]) > 1e-15 for h in ("M-1",)):
            ok = False
            print("  VORTEST FEHLER: Holm-Familien nicht getrennt")
        pm ={h: a1["datensaetze"]["mnist"][k]["p"] for h, (_, k) in FAMILIEN["MNIST"].items()}
        d1 = a1["datensaetze"]["mnist"]["suche_minus_ohne_maske"]
        if holm(pm) != {h: a1["holm"][h] for h in pm} or a1["holm"]["M-1"] != d1["p"] \
                or a1["holm_tost"]["M-1"] != d1["p_tost"] or set(a1["holm"]) != {"M-1", "M-3"}:
            ok = False
            print("  VORTEST FEHLER: Holm x1 (M-1 = rohes p) falsch")
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: (B1-B8) Bausteine -- W(V, m) gegen unabhaengige Konstruktion, Rekonstruktion der Schaltung,
    ohne_maske = Weglassen des quadratischen Anteils, block2x2_maske, k = 0, Kopf, Statistik, Seeds;
    Wiederaufnahme mit simuliertem Abbruch; jeder Urteilszweig synthetisch (Marke UND Text, mehrere Ziehungen)."""
    t0 = time.time()
    ok = _vortest_bausteine()
    ok = _vortest_wiederaufnahme() and ok
    ok = _vortest_urteile(ziehungen) and ok
    print(f"  Vortest {KARTE}: {'bestanden' if ok else 'DURCHGEFALLEN'} ({time.time() - t0:.0f} s)", flush=True)
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere(KARTE + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
