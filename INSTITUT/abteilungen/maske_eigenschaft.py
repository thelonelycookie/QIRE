"""Abteilung Clifford -- WELCHE Eigenschaft der gesuchten Vorzeichenmaske traegt: ihr Rang oder die konkrete Form?
(Karte I-036-v3, Quantenstufe Q2: exakte Simulation; Kern der Frage ist klassisch.)

HEILMEIER:
  Was wollen wir?  I-036-v2 (05.10., ergebnisse/I-036-v2__20261005_182417.json) hat gezeigt: Die Clifford-Suche
      findet Paare (V_s, m_s) aus Walsh-Unterraum und quadratischer Vorzeichenmaske m_s = (-1)^{q_s}; ohne den
      quadratischen Anteil verlieren dieselben Bloecke 1,93 pp (MNIST, M-1) bzw. 2,36 pp (Fashion, M-3). Welche
      Eigenschaft von q_s traegt: reicht ihr RANG (jede Form gleichen Rangs auf denselben Bloecken tut es), oder
      braucht es die KONKRETE, gesuchte Form?
  Wie heute?  Unbekannt. Beschreibend aus den I-036-v2-Rohdaten (beim Bau gelesen, nichts neu gerechnet): Rang und
      Gewinn haengen ueber die Seeds zusammen (Spearman MNIST 0,70, Fashion 0,46; Rang 0/2/4/6 -> +0,0/+0,7/+2,0/
      +3,7 pp auf MNIST), aber innerhalb eines Rangs streut der Gewinn stark (sd 1,3 pp; Rang 2: -0,1 bis +3,4 pp).
      Ob das an der Form oder an V_s liegt, trennt erst ein Austausch der Form bei festem V_s.
  Was ist neu?  Je Seed aus der gefundenen Schaltung (keine weitere Suche): dieselben Bloecke V_s mit (a) der
      gesuchten Form, (b) 20 zufaelligen Formen GLEICHEN RANGS, (c) der Form eines anderen Seeds, (d) ohne Maske;
      derselbe L-BFGS-Kopf, dieselben Daten. Neuheitspruefung: recherche/2026-10-05_neuheit_I-036-v3.md (nichts
      Vergleichbares; naechster Nachbar 2604.25480, optische gelernte Phasenmaske, kein Rang, kein Clifford).

THEORIE (vor dem Festlegen gerechnet; der Vortest (E1)-(E9) prueft jeden Satz):
  T1  (aus I-036-v2 T1/T2) Eine Suchschaltung in Bruhat-Form ist bis auf Vorzeichen/Reihenfolge der Zeilen
      W(V_s, m_s) (maske_block.matrix_vm); m_s ist je Nebenklasse a0 + V_s nur bis auf einen affinen Charakter
      bestimmt, und genau dieser aendert die Merkmale nicht. Merkmalswirksam ist also q_s MODULO AFFINER FUNKTIONEN.
  T2  INVARIANTE. Eine quadratische Funktion q auf V = GF(2)^k ist modulo affiner Funktionen genau durch ihre
      alternierende Bilinearform B(u, v) = q(u+v) + q(u) + q(v) + q(0) bestimmt (symmetrisch, Diagonale 0; ueber
      GF(2) ist v_i^2 = v_i linear). In Basiskoordinaten: B_ij = [m(a0) m(a0^b_i) m(a0^b_j) m(a0^b_i^b_j) = -1],
      unabhaengig vom affinen Charakter (form_aus_maske). GL(V) wirkt durch B -> P^T B P; die Bahnen sind GENAU die
      Raenge 2h (h = 0..k/2; Satz ueber alternierende Formen). GEWAEHLTE INVARIANTE: rang(B) ueber GF(2).
      Begruendung: (i) Sie ist die VOLLSTAENDIGE GL(V)-Invariante des merkmalswirksamen Teils -- „gleicher Rang“
      heisst genau „dieselbe Form bis auf Umbenennung der Koordinaten von V“; (ii) die Arf-Invariante ist KEINE
      Invariante modulo affiner Terme (ein linearer Term aendert sie, T1: merkmalsneutral) und scheidet deshalb aus;
      (iii) rang(B) = 2h ist die Zahl der CZ-gekoppelten Bitpaare nach Basiswechsel (q ~ x1x2 + ... + x_{2h-1}x_{2h}),
      also die „Verschraenkungstiefe“ der Phase.
  T3  EINHEITLICHKEIT (beim Bau auf I-036-v2 gefunden, 40/40 Seeds): die Form B ist auf ALLEN Nebenklassen von V_s
      dieselbe und die kanonische Maske ist auf jeder Nebenklasse genau (-1)^{sum_{i<j} B_ij t_i t_j} (ein globales
      Q auf GF(2)^8, eingeschraenkt: Q(a0+v) = Q(a0) + B(a0, v) + Q(v), der Mittelteil ist affin). Deshalb ist „die
      Form der Suche“ EIN k x k-B. Vorab festgelegt: Ein Seed mit uneinheitlicher oder nicht quadratischer Form wird
      gezaehlt und ausgeschlossen (erwartet 0; Vortest (E2): 12 Zufalls-Clifford, alle einheitlich).
  T4  ARM zufall_gleicher_rang: P gleichverteilt in GL(k, 2) (Verwerfungsziehung), B_r = P^T B_s P, dieselbe
      Form B_r auf allen Nebenklassen (wie T3), Maske maske_aus_form(B_r). Nach Bahn-Stabilisator ist B_r
      GLEICHVERTEILT auf der Bahn {B : rang B = rang B_s} (Vortest (E3): k = 3, Rang 2, alle 7 Formen gleich haeufig).
      Gleichwertig: die Koordinaten von V_s werden zufaellig linear umbenannt und die Maske folgt (Vortest: gleiche
      Merkmale wie die explizit umbenannte Maske). Je Seed 20 Ziehungen (eigener Seed FORM_SEED + s), Wert des Arms
      = MEDIAN der 20 Testgenauigkeiten (Rauschen der einzelnen Ziehung heraus; die Streuung wird berichtet). Mit
      Wahrscheinlichkeit 1/|Bahn| ist B_r = B_s (berichtet; |Bahn| = anzahl_formen(k, r), bei k >= 4 und r >= 2
      mindestens 28, bei k = 2 = r gleich 1 -- dann ist der Arm identisch zur Suche, berichtet).
  T5  ARM fremde_maske (beschreibend): W(V_s, m_t) mit der Maske m_t eines ANDEREN Mess-Seeds t desselben Datensatzes
      DIESES Laufs, NUR wenn V_s ⊆ V_t. Begruendung: m_t ist nur bis auf affine Charaktere je V_t-Nebenklasse
      bestimmt. Liegt jede V_s-Nebenklasse in einer V_t-Nebenklasse (<=> V_s ⊆ V_t), bleibt diese Mehrdeutigkeit auf
      V_s affin, also merkmalsneutral: die Uebertragung ist OHNE Konvention definiert, und die uebertragene Form ist
      die Einschraenkung B_t|V_s (Vortest (E5)). Fuer V_s ⊄ V_t wird die Mehrdeutigkeit quadratisch (I-036-v2 T4,
      Vortest (E5) zeigt ein Beispiel) -- eine Abbildung „ueber die kanonische Basis“ waere dann eine Konvention
      (I-036-v2 musste genau das mit alt1/alt2 absichern). „Nur gleiches V“ waere ebenfalls sauber, aber zu selten
      (I-036-v2: 1 bzw. 3 Paare von 190); V_s ⊆ V_t verallgemeinert es und deckte in I-036-v2 14/20 MNIST- und
      20/20 Fashion-Seeds mit >= 1 Spender. Wert je Seed = Median ueber seine Spender; der Rang der uebertragenen
      Form kann vom eigenen abweichen (berichtet, mit Teilmenge „gleicher Rang“). Rechnet erst, wenn ALLE Suchen
      fertig sind (zweite Stufe, eigene Wiederaufnahme-Schluessel mit Spenderliste).
  T6  SONDERFAELLE: q ≡ 0 ergibt exakt ohne_maske (Vortest (E4)); gleiches B ergibt gleiche Merkmale (Vortest (E6),
      und JE MESS-SEED: W(V_s, maske_aus_form(B_s)) muss aequivalent zur Schaltung sein, sonst Ausschluss). Rang 0
      (q_s affin; I-036-v2: 1/20 MNIST): suche = zufall = ohne_maske EXAKT nach Konstruktion; solche Seeds tragen keine
      Information ueber die Frage und wuerden TOST mechanisch zu „gleich“ schieben -> vorab aus R-1..R-4 UND aus
      P-0/P-0F herausgenommen (gezaehlt, Rangverteilung schliesst sie ein).
  ZERLEGUNG (beschreibend): suche - ohne_maske = (suche - zufall_gleicher_rang) + (zufall_gleicher_rang - ohne_maske)
      = Formanteil +Ranganteil; Anteil des Rangs = (zufall - ohne) / (suche - ohne) im Mittel.

ARME (alle mit DEMSELBEN logistischen L-BFGS-Kopf aus clifford_suche, standardisierte Merkmale, Kopf-L2 0,003 wie
  I-036/I-036-v2, 6.000 Nicht-Test-Bilder / 5.000 Testbilder, clifford.daten 16x16):
  suche                 SA + Politur, bitgleich zu I-036-v2 (_mess_seed dort: gleiche Start-/SA-Seeds relativ, 100
                        Zufalls-Clifford nur fuer Temperaturskala und Zwischenspeicher)
  zufall_gleicher_rang  W(V_s, (-1)^{q_r}), rang(B_r) = rang(B_s), Median ueber 20 Ziehungen (T4)
  ohne_maske            W(V_s, 1) wie I-036-v2
  fremde_maske          W(V_s, m_t), V_s ⊆ V_t, Median ueber Spender (T5, beschreibend)
  beschreibend          identitaet (Sensitivitaet), suche_aus_form = W(V_s, maske_aus_form(B_s)) (muss = suche sein),
                        LOO-Kriterium (Ridge, Nicht-Test) aller Arme -> Siegerfluch der Formwahl.

AUSSCHLUSSREGEL (vorab): Ein Mess-Seed ohne Bruhat-Form, mit nicht aequivalenter Rekonstruktion W(V_s, m_s), mit
  uneinheitlicher/nicht quadratischer Form (T3) oder mit W(V_s, maske_aus_form(B_s)) ≠ Schaltung wird gezaehlt und aus
  allen Vergleichen ausgeschlossen; Seeds mit rang(B_s) = 0 sind gueltig, aber nicht analysierbar (T6). Weniger als
  10 analysierbare Seeds -> Hypothesen des Datensatzes HU (Vorbedingung). Erwartet: 0 Ausschluesse, ~1 Rang-0-Seed.

SEEDS (05.10. geprueft gegen ergebnisse/*.json -- jede ganze Zahl 760-815 mit Pfad: nur T-Stichproben, Gatter-,
  Tiefen-, Schritt- und Trainingszahlen, keine Seeds --, agenda.json, ideen.json (keine Treffer), abteilungen/*.py,
  labor/*.py, direktor.py (nur Genauigkeiten/Kurvenpunkte); _pruefe_seeds prueft zusaetzlich gegen
  maske_block._BELEGT (walsh_kontrolle._BELEGT + I-036) und die I-036-v2-Seeds 700-759):
  Mess-Seeds  MNIST 760-779, Fashion 780-799 (je 20)       Strukturaufgabe (Q-0) 800-809
  Rauchtest   MNIST 810/811, Fashion 812/813, Struktur 814/815 (nie Mess-Seeds)
  Formziehungen: Generator FORM_SEED + Seed (120.000 + s), nicht mit anderen Generatoren geteilt.

FAMILIEN (vor jeder Zahl): MNIST {R-1, R-2} ist die Hauptfamilie (Holm x2): beide Glieder sind die zwei Summanden
  derselben Zerlegung auf denselben Seeds und beantworten zusammen die EINE Frage „Rang oder Form?“; eine Familie je
  Glied waere zu nachsichtig, eine gemeinsame Familie mit Fashion wuerde die Hauptaussage durch die Replikation
  verduennen. Fashion {R-3, R-4} ist die REPLIKATION in eigener Familie (Holm x2) -- anders als in I-036-v2 ist hier
  auch auf Fashion GLEICHWERTIG moeglich, weil die Positivkontrolle P-0F dort voraussichtlich besteht (+2,36 pp).
  Wilcoxon-p und TOST-p werden je getrennt Holm-korrigiert (wie I-036-v2).

ERREICHBARKEIT VON GLEICHWERTIG (Regel 2b, vorab gerechnet aus den I-036-v2-Rohdaten): TOST ±0,5 pp, n = 20 (19 bei
  einem Rang-0-Seed), Holm x2. Ist das andere Glied klar verschieden (TOST-p ~ 1), muss das kleinere TOST-p < 0,025
  sein: t_0,975;19 = 2,093, also GLEICHWERTIG nur bei |d| < 0,5 - 2,093 · sd/√20. Streuungsannahmen fuer
  d = suche - zufall (Median von 20):
    (a) „Rang reicht“: nur Testrauschen zweier verschiedener Merkmalsabbildungen gleicher Guete. I-036-v2: sd von
        suche - block2x2 0,50 / 0,45 pp, suche - block2x2_maske 0,51 / 0,66 pp (MNIST / Fashion); Median-Bildung
        senkt den zufall-Anteil. sd 0,50 pp -> erreichbar bei |d| < 0,27 pp; sd 0,66 pp -> |d| < 0,19 pp.
    (b) Form-Heterogenitaet wie der Gewinn innerhalb eines Rangs (I-036-v2 sd 1,31 pp MNIST): se 0,29 -> NICHT
        erreichbar (Grenze -0,11 pp). Hoechste sd, bei der GLEICHWERTIG bei d = 0 ueberhaupt moeglich ist:
        1,07 pp (erster Holm-Schritt), 1,29 pp (zweiter).
  Folge: GLEICHWERTIG ist GENAU in dem Szenario erreichbar, in dem es auch zutrifft (Rang reicht -> kleine Streuung),
  und im Gegenszenario sowieso nicht zu erwarten. HU bleibt moeglich, wenn die Form teilweise traegt (sd zwischen
  0,7 und 1,3 pp und |d| ~ 0,3-0,6 pp). Fuer R-2 gilt dieselbe Rechnung. Das ist die beste vorab verfuegbare
  Naeherung; die echte sd wird berichtet.

VORREGISTRIERT (Familien {R-1, R-2} und {R-3, R-4}, Holm x2; delta = 0,5 pp; nur analysierbare Seeds):
  P-0  Positivkontrolle MNIST (nicht in der Familie; Voraussetzung fuer GLEICHWERTIG in R-1/R-2): suche - ohne_maske
       reproduziert I-036-v2 M-1: einseitiger exakter Wilcoxon (suche > ohne_maske) p < 0,05 UND Mittel >= 0,5 pp
       -> BESTANDEN / DURCHGEFALLEN. Ohne nachweisbaren Masken-Effekt ist „Teil gleichwertig“ keine Aussage.
  P-0F dasselbe auf Fashion (Voraussetzung fuer GLEICHWERTIG in R-3/R-4).
  Q-0  Positivkontrolle der Suche wie I-005/I-036 (Strukturaufgabe, Seeds 800-809), Voraussetzung fuer GLEICHWERTIG.
  R-1  MNIST d = suche - zufall_gleicher_rang: IST ES DIE KONKRETE FORM ODER REICHT IHR RANG?
       Holm-p_TOST < 0,05: Q-0 nicht H1 -> HU (Positivkontrolle Q-0); P-0 durchgefallen -> HU (Positivkontrolle);
                           sonst GLEICHWERTIG (der Rang reicht; bei Holm-p < 0,05 zusaetzlich im Text)
       sonst Holm-p < 0,05, d > 0 -> SUCHE_BESSER (die konkrete Form traegt ueber den Rang hinaus)
             Holm-p < 0,05, d < 0 -> ZUFALL_BESSER (zufaellige Formen gleichen Rangs sind besser: Siegerfluch der
                                     Formwahl auf dem LOO-Kriterium)
       sonst HU. Weniger als 10 analysierbare Seeds -> HU (Vorbedingung).
  R-2  MNIST d = zufall_gleicher_rang - ohne_maske: BRINGT SCHON IRGENDEINE MASKE GLEICHEN RANGS ETWAS?
       gleiche Logik; Ausgaenge GLEICHWERTIG (der Rang allein bringt nichts) / ZUFALL_BESSER (der Rang traegt) /
       OHNE_BESSER (eine beliebige Form gleichen Rangs schadet) / HU.
  R-3, R-4  wie R-1, R-2 auf Fashion (Replikation, Familie {R-3, R-4}, Gleichwertigkeit braucht P-0F und Q-0).
  Fehlt ein Datensatz, zaehlen seine Glieder mit p = 1.
  R-Z  beschreibend, aus R-1/R-2 (bzw. R-3/R-4) abgeleitet, kein eigenes Urteil: GLEICHWERTIG + ZUFALL_BESSER ->
       RANG_TRAEGT; SUCHE_BESSER + GLEICHWERTIG -> FORM_TRAEGT; SUCHE_BESSER + ZUFALL_BESSER -> RANG_UND_FORM;
       sonst OFFEN.
  R-B  beschreibend: Mittel der Arme, Zerlegung (Anteil des Rangs), Rangverteilung und (k, Rang)-Tabelle (alle
       gueltigen Seeds), Spearman ueber SEEDS (ein Wert je Seed, keine Pseudoreplikation, Regel 9): rang ~ (suche -
       ohne), rang ~ (zufall - ohne), rang ~ (suche - zufall), k ~ (suche - ohne) (Stoerfaktor k, rang <= k);
       fremde_maske (Abdeckung, fremde - ohne, suche - fremde, fremde - zufall, Anteil gleicher Rang, Teilmenge
       gleicher Rang), Streuung der 20 Ziehungen (IQR), Anteil Ziehungen = B_s, Bahngroessen, LOO-Kriterium und
       Siegerfluch (Vorsprung suche - zufall im Kriterium minus im Test), Sensitivitaet suche - id, Aufwand.

VORHERSAGE (vor dem Lauf, hier festgehalten):
  Q-0 H1, P-0 BESTANDEN (~+2 pp), P-0F BESTANDEN (~+2,3 pp).
  R-1 SUCHE_BESSER, d ≈ +0,7 pp (MNIST): Die Form ist nicht beliebig -- in I-036-v2 streut der Gewinn innerhalb
      eines Rangs um 1,3 pp, und die Suche waehlt q nach dem LOO-Kriterium unter allen Formen der Bahn; die
      gesuchte Form koppelt Zeilen- mit Spaltenbits, eine zufaellige Umbenennung von V mischt das.
  R-2 ZUFALL_BESSER, d ≈ +1,2 pp: Eine Form hohen Rangs „verschmiert“ die Walsh-Zeilen eines Blocks (bei vollem
      Rang ist (-1)^q bent, ihre Walsh-Transformierte flach) -- das hilft grossen Bloecken auch ohne Feinwahl.
  => R-Z RANG_UND_FORM, Anteil des Rangs ~ 60 %. Fashion: R-3 HU oder SUCHE_BESSER (+0,4 pp; Fashion-k naeher an 8,
     Raenge 4-8, weniger Spielraum), R-4 ZUFALL_BESSER (+1,9 pp). Spearman rang ~ (suche - ohne) positiv (≈ 0,6
     MNIST), rang ~ (suche - zufall) nahe 0. fremde_maske zwischen zufall und suche.

FOLGEFRAGEN je Ausgang (fuer agenda.json „folgen“, vom Direktor einzutragen; maschinenlesbar in FOLGEFRAGEN):
  R-1 SUCHE_BESSER  -> Welche Struktur der Form traegt (Kopplung Zeilen-/Spaltenbits; Form-Suche bei festem V_s
                       und festem Rang = Politur nur der CZ/S-Lage)?
  R-1 GLEICHWERTIG  -> Suche nur ueber (V, Rang) mit zufaelliger Form: reicht eine klassisch billige Zweistufen-
                       suche fuer das Niveau der Clifford-Suche?
  R-1 ZUFALL_BESSER -> Siegerfluch der Formwahl: groessere Nicht-Test-Menge oder Kriterium mit Hold-out pruefen.
  R-1 HU            -> 40 Seeds bzw. 10.000 Testbilder; Ziehungszahl erhoehen, falls die Ziehungs-IQR dominiert.
  R-2 ZUFALL_BESSER -> Dosis-Kurve ueber den Rang: feste Bloecke V_s mit zufaelligen Formen aller Raenge 0..k.
  R-2 GLEICHWERTIG  -> Der Rang allein ist wertlos: Was unterscheidet die gesuchte Form von zufaelligen (Abstand
                       in der Bahn, Lage zu den Bildachsen)?
  R-2 OHNE_BESSER   -> Zufaellige Masken schaden: Ist die Maske nur als Korrektur zu V_s nuetzlich (Paarbindung)?
  R-2 HU            -> wie R-1 HU.
  P-0 DURCHGEFALLEN -> I-036-v2 M-1 repliziert nicht: Seed-Satz/Siegerfluch pruefen, bevor zerlegt wird.
  R-3/R-4 (je Ausgang) wie R-1/R-2 auf Fashion; Widerspruch MNIST/Fashion -> datensatzabhaengige Formwahl (Regel
                       „Baustein je Aufgabe kalibrieren“, Uebergabe §3).

LAUFZEIT (I-036-v2 gemessen, 4 Prozesse: ganzer Seed 403 s MNIST / 367 s Fashion, davon Suche 387 / 348 s und
  5 Koepfe + 5 LOO + Pruefungen 5,2 / 8,9 s; Strukturseed ~41 s). Hier je Seed 24 Koepfe + 24 LOO (suche,
  ohne_maske, identitaet, suche_aus_form, 20 Formen). Rauchtest 05.10. (1 Prozess, 80 SA-Schritte, Seeds 810-815):
  alle Koepfe/Kriterien/Pruefungen 14-16 s (MNIST) / 27-30 s (Fashion) je Seed, zweite Stufe < 5 s je Seed, ganzer
  Rauchtest 143 s; Ziehungs-IQR der 20 Formen 0,5-0,9 pp (darum der Median). Schaetzung je Seed ~415 s MNIST /
  ~390 s Fashion -> 20·415 + 20·390 + 10·41 + 40·8 ≈ 16.800 CPU-s; INSTITUT_PROZESSE = 4: ~1 h 10 min (bis ~1 h 25
  min mit Lastungleichgewicht und langsameren Koepfen unter Last); = 2: ~2 h 20 min.

WIEDERAUFNAHME (Pflicht, wie I-036-v2): jeder fertige Seed (Stufe 1: Suche + eigene Arme; Stufe 2: fremde_maske mit
  Spenderliste im Schluessel) wird SOFORT in ergebnisse/_zwischen_I-036-v3.json geschrieben (atomar: .tmp +
  os.replace, vom Hauptprozess). Ein Neustart mit denselben Einstellungen ueberspringt alle fertigen Seeds; andere
  Einstellungen -> Fehler (nie mischen). Die Datei bleibt nach dem Lauf liegen (der Direktor sichert erst danach).

GRENZEN: Test 5.000 Bilder (0,02 pp je Bild). „Gleicher Rang“ heisst gleiche GL(V)-Bahn auf DENSELBEN Bloecken; die
  Lage der Form zu den Bildachsen (welche Pixelbits gekoppelt sind) ist genau das, was zufall_gleicher_rang
  zerstoert -- R-1 SUCHE_BESSER sagt also „die Lage traegt“, nicht welche. Die zufaelligen Formen sind Clifford, aber
  nicht notwendig im Suchraum der 4-Lagen-Zelle. fremde_maske ist nur fuer V_s ⊆ V_t definiert (Auswahl der Seeds
  nach V, beschreibend).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from abteilungen import clifford_suche as v1
from abteilungen import clifford_suche_v2 as v2
from abteilungen import maske_block as mb
from abteilungen.clifford import LAGEN, QUBITS_JE_DATENSATZ
from abteilungen.clifford_suche import (Bewerter, _bewerte, aequivalent, born, holm, normiere, ridge_loo,
                                        unitaer_clifford, zufall_codes)
from abteilungen.clifford_suche_v2 import tost_p
from abteilungen.maske_block import (_koord, kanonische_maske, matrix_explizit, matrix_vm, nebenklassen, rref_basis,
                                     span, wilcoxon_einseitig, zerlege, zufalls_basis)
from abteilungen.walsh_kontrolle import _gepaart, struktur, wilcoxon_schnell
from labor.gemeinsam import ERGEBNISSE, speichere, wilcoxon_exakt, zusammenfassung

KARTE = "I-036-v3"
DATENSAETZE = ("mnist", "fashion_mnist")
NAME = {"mnist": "MNIST", "fashion_mnist": "Fashion"}
N_SEEDS = 20
SEEDS = {"mnist": tuple(range(760, 780)), "fashion_mnist": tuple(range(780, 800))}
SEEDS_STRUKTUR = tuple(range(800, 810))
ENTWURF_SEEDS = {"mnist": (810, 811), "fashion_mnist": (812, 813)}
ENTWURF_STRUKTUR = (814, 815)
# belegt: maske_block._BELEGT (walsh_kontrolle + I-036) und alles aus I-036-v2 (Mess 700-739, Struktur 740-749,
# Rauchtest 750-755) samt Reserve bis 759
_BELEGT = mb._BELEGT | set(range(700, 760))
KOPF_L2 = dict(mb.KOPF_L2)                           # 0,003 wie I-036/I-036-v2, kein neues Tuning
ALPHA, TOST_W = 0.05, 0.005
MIN_GUELTIG = 10
N = 8
N_FORMEN = 20                                        # Zufallsformen gleichen Rangs je Seed (Median)
FORM_SEED = 120_000
FAMILIEN = {"MNIST": {"R-1": ("mnist", "suche_minus_zufall"), "R-2": ("mnist", "zufall_minus_ohne_maske")},
            "Fashion": {"R-3": ("fashion_mnist", "suche_minus_zufall"),
                        "R-4": ("fashion_mnist", "zufall_minus_ohne_maske")}}
P0_VON = {"mnist": "P-0", "fashion_mnist": "P-0F"}
ARME = ("suche", "zufall_gleicher_rang", "ohne_maske")

HYPOTHESEN = {
    "P-0": "Positivkontrolle MNIST (nicht in der Familie): suche - ohne_maske reproduziert I-036-v2 M-1: einseitiger "
           "exakter Wilcoxon p < 0,05 und Mittel >= 0,5 pp -> BESTANDEN, sonst DURCHGEFALLEN. Voraussetzung fuer "
           "GLEICHWERTIG in R-1/R-2.",
    "P-0F": "wie P-0 auf Fashion (I-036-v2 M-3 +2,36 pp); Voraussetzung fuer GLEICHWERTIG in R-3/R-4.",
    "Q-0": "Positivkontrolle der Suche wie I-005/I-036 Q-0 (Strukturaufgabe, frische Seeds 800-809). Nicht in der "
           "Familie; Voraussetzung fuer GLEICHWERTIG.",
    "R-1": "MNIST: suche (V_s, q_s) vs. zufall_gleicher_rang (V_s, Median ueber 20 Formen q_r mit rang B_r = rang "
           "B_s, B_r = P^T B_s P, P gleichverteilt in GL(k,2)), gleicher Kopf, gepaart je Seed, nur analysierbare "
           "Seeds (gueltig, Rang > 0). Holm-p_TOST(+-0,5 pp) < 0,05 -> GLEICHWERTIG (Rang reicht; wenn Q-0 H1 und P-0 "
           "BESTANDEN, sonst HU mit Grund); sonst Holm-p(Wilcoxon) < 0,05 -> SUCHE_BESSER (konkrete Form traegt) / "
           "ZUFALL_BESSER; sonst HU. Familie {R-1, R-2} (Holm x2). < 10 analysierbare Seeds -> HU (Vorbedingung).",
    "R-2": "MNIST: zufall_gleicher_rang vs. ohne_maske (V_s, 1). GLEICHWERTIG (Rang allein bringt nichts) / "
           "ZUFALL_BESSER (Rang traegt) / OHNE_BESSER / HU; Logik und Familie wie R-1.",
    "R-3": "Fashion (Replikation, eigene Familie {R-3, R-4}, Gleichwertigkeit braucht P-0F): wie R-1.",
    "R-4": "Fashion: wie R-2, Familie {R-3, R-4}.",
    "R-Z": "beschreibend, aus R-1/R-2 bzw. R-3/R-4 abgeleitet (kein eigenes Urteil): RANG_TRAEGT / FORM_TRAEGT / "
           "RANG_UND_FORM / OFFEN.",
    "R-B": "beschreibend: Mittel, Zerlegung (Anteil des Rangs), Rangverteilung, (k, Rang), Spearman ueber Seeds "
           "(rang/k gegen Gewinne, ein Wert je Seed), fremde_maske (V_s ⊆ V_t), Ziehungsstreuung, Bahngroessen, "
           "LOO-Kriterium und Siegerfluch, Sensitivitaet, Aufwand.",
}

FOLGEFRAGEN = {
    "R-1": {"SUCHE_BESSER": "Welche Struktur der Form traegt (Kopplung Zeilen-/Spaltenbits)? Form-Suche bei festem V_s "
                            "und festem Rang (Politur nur der CZ/S-Lage).",
            "GLEICHWERTIG": "Zweistufensuche nur ueber (V, Rang) mit zufaelliger Form: erreicht sie das Niveau der "
                            "Clifford-Suche, klassisch billiger?",
            "ZUFALL_BESSER": "Siegerfluch der Formwahl: groessere Nicht-Test-Menge oder Hold-out-Kriterium pruefen.",
            "HU": "Wiederholung mit 40 Seeds bzw. 10.000 Testbildern; mehr Ziehungen, falls die Ziehungs-IQR dominiert."},
    "R-2": {"ZUFALL_BESSER": "Dosis-Kurve ueber den Rang: feste Bloecke V_s mit zufaelligen Formen aller Raenge 0..k.",
            "GLEICHWERTIG": "Rang allein wertlos: Was unterscheidet die gesuchte Form von zufaelligen derselben Bahn "
                            "(Lage zu den Bildachsen)?",
            "OHNE_BESSER": "Zufaellige Masken schaden: ist die Maske nur als Korrektur zu V_s nuetzlich (Paarbindung)?",
            "HU": "wie R-1 HU."},
    "R-3/R-4": "wie R-1/R-2 auf Fashion; Widerspruch zu MNIST -> datensatzabhaengige Formwahl als eigene Karte.",
    "P-0 DURCHGEFALLEN": "I-036-v2 M-1 repliziert nicht: Seed-Satz und Siegerfluch pruefen, bevor zerlegt wird.",
}


# ------------------------------------------------------------------ Theorie: Form <-> Maske
def rang_gf2(B) -> int:
    B = (np.asarray(B, dtype=np.int64) % 2).copy()
    r, (zeilen, spalten) = 0, B.shape
    for c in range(spalten):
        p = [i for i in range(r, zeilen) if B[i, c]]
        if not p:
            continue
        B[[r, p[0]]] = B[[p[0], r]]
        for i in range(zeilen):
            if i != r and B[i, c]:
                B[i] ^= B[r]
        r += 1
    return r


def _bits(k) -> np.ndarray:
    """Walsh-Index t -> Koordinatenvektor (Spalte j gehoert zu basis[j], wie _koord)."""
    return (np.arange(2 ** k)[:, None] >> (k - 1 - np.arange(k))[None, :]) & 1


def form_aus_maske(m, V, basis=None):
    """T2/T3: alternierende Form B (k x k, GF(2)) der Maske m auf den Nebenklassen von V, in den Koordinaten der
    Basis (Standard: reduzierte Zeilenstufenform). B_ij = [m(a0) m(a0^b_i) m(a0^b_j) m(a0^b_i^b_j) = -1] ist
    unabhaengig vom affinen Charakter. -> (B der ersten Nebenklasse, einheitlich, quadratisch)."""
    m = np.asarray(m, dtype=float)
    basis = rref_basis(V) if basis is None else list(basis)
    k = len(basis)
    formen, quadratisch = [], True
    mk = kanonische_maske(m, V, basis)
    koord = _koord(basis)
    for a0 in nebenklassen(V, len(m)):
        a0 = int(a0)
        B = np.zeros((k, k), np.int64)
        for i in range(k):
            for j in range(i + 1, k):
                bi, bj = int(basis[i]), int(basis[j])
                B[i, j] = B[j, i] = int(m[a0] * m[a0 ^ bi] * m[a0 ^ bj] * m[a0 ^ bi ^ bj] < 0)
        formen.append(B)
        quadratisch &= bool(np.array_equal(mk[a0 ^ koord], _vorzeichen(B)))
    einheitlich = all(np.array_equal(B, formen[0]) for B in formen)
    return formen[0], einheitlich, quadratisch


def _vorzeichen(B) -> np.ndarray:
    """(-1)^{sum_{i<j} B_ij t_i t_j} fuer alle Walsh-Indizes t."""
    k = len(B)
    if k == 0:
        return np.ones(1)
    b = _bits(k)
    q = np.einsum("ti,ij,tj->t", b, np.triu(np.asarray(B, dtype=np.int64), 1), b) % 2
    return (-1.0) ** q


def maske_aus_form(B, V, d=2 ** N, basis=None) -> np.ndarray:
    """Kanonische Maske zur Form B: auf JEDER Nebenklasse a0 + V gleich (-1)^{q_B(t)} (T3), ohne lineare Terme."""
    basis = rref_basis(V) if basis is None else list(basis)
    koord = _koord(basis)
    s = _vorzeichen(B)
    m = np.ones(d)
    for a0 in nebenklassen(V, d):
        m[int(a0) ^ koord] = s
    return m


def zufall_gl(k, rng) -> np.ndarray:
    """P gleichverteilt in GL(k, 2) (Verwerfungsziehung)."""
    while True:
        P = rng.integers(0, 2, (k, k))
        if rang_gf2(P) == k:
            return P


def zufall_form(B, rng) -> np.ndarray:
    """T4: B_r = P^T B P (mod 2), gleichverteilt auf der GL-Bahn von B (= alle Formen gleichen Rangs)."""
    B = np.asarray(B, dtype=np.int64)
    if len(B) == 0:
        return B.copy()
    P = zufall_gl(len(B), rng)
    return (P.T @ B @ P) % 2


def anzahl_formen(k, r) -> int:
    """Zahl der alternierenden k x k-Formen ueber GF(2) vom Rang r (= Groesse der Bahn): fuer r = 2h
    prod_{i<2h}(2^{k-i}-1) / prod_{i=1..h}(2^{2i}-1) · 2^{h(h-1)}; ungerades r: 0."""
    if r % 2 or r > k:
        return 0
    h = r // 2
    z = 1
    for i in range(2 * h):
        z *= 2 ** (k - i) - 1
    n = 1
    for i in range(1, h + 1):
        n *= 2 ** (2 * i) - 1
    return z // n * 2 ** (h * (h - 1))


def einschraenkung(B_t, V_t, V_s) -> np.ndarray:
    """B_t (Koordinaten der Stufenbasis von V_t) eingeschraenkt auf V_s ⊆ V_t (Koordinaten der Stufenbasis von V_s)."""
    bt, bs = rref_basis(V_t), rref_basis(V_s)
    index = {int(v): t for t, v in enumerate(_koord(bt))}
    bits = _bits(len(bt))
    C = np.array([bits[index[int(b)]] for b in bs], dtype=np.int64).reshape(len(bs), len(bt))
    return (C @ np.asarray(B_t, dtype=np.int64) @ C.T) % 2


def teilraum(V_s, V_t) -> bool:
    return set(int(v) for v in V_s) <= set(int(v) for v in V_t)


# ------------------------------------------------------------------ Arbeiter
def _mess_seed(a):
    """(ds, seed, schritte, politur_max, n_zufall, l2, n_formen) -> Zeile. Suche bitgleich zu I-036-v2."""
    ds, s, schritte, politur_max, n_zufall, l2, n_formen = a
    n, t0 = QUBITS_JE_DATENSATZ[ds], time.time()
    xtr, ytr, xte, yte = v2._daten(ds, s)
    bew = Bewerter(xtr, ytr, n)
    xs = (normiere(xtr), ytr, normiere(xte), yte, l2)
    z = {"seed": s, "datensatz": ds, "kopf_l2": l2, "n_train": len(ytr), "n_test": len(yte)}
    zuf = zufall_codes(s, n_zufall, n)                  # wie I-036-v2: nur Skala und Zwischenspeicher
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
    Us = unitaer_clifford(best, n)
    st = struktur(Us)
    z["suche_struktur"] = st
    z["gueltig"], z["analysierbar"] = bool(st["bruhat_form"]), False
    if z["gueltig"]:
        V_s, m_s, _ = zerlege(Us)
        B_s, einh, quad = form_aus_maske(m_s, V_s)
        m_f = maske_aus_form(B_s, V_s)
        z["rekonstruktion_ok"] = bool(aequivalent(Us, matrix_vm(V_s, m_s)))
        z["form_einheitlich"], z["form_quadratisch"] = bool(einh), bool(quad)
        z["form_rekonstruktion_ok"] = bool(aequivalent(Us, matrix_vm(V_s, m_f)))
        z["gueltig"] = z["rekonstruktion_ok"] and einh and quad and z["form_rekonstruktion_ok"]
        k, r = len(rref_basis(V_s)), rang_gf2(B_s)
        z.update({"k": k, "rang": r, "bahn": anzahl_formen(k, r), "V_s": V_s.tolist(), "form_s": B_s.tolist(),
                  "maske_s": m_s.astype(int).tolist()})
        z["analysierbar"] = bool(z["gueltig"] and r > 0)
        arme = {"ohne_maske": np.ones(len(m_s)), "suche_aus_form": m_f}
        for arm, m in arme.items():
            W = matrix_vm(V_s, m)
            z[arm] = _bewerte(W, n, xs)[0]
            z[f"{arm}_val"] = ridge_loo(born(bew.x, W), bew.y)[0]
        if z["analysierbar"]:
            rng = np.random.default_rng(FORM_SEED + s)
            tests, vals, raenge, gleich = [], [], [], 0
            for _ in range(n_formen):
                B_r = zufall_form(B_s, rng)
                raenge.append(rang_gf2(B_r))
                gleich += bool(np.array_equal(B_r, B_s))
                W = matrix_vm(V_s, maske_aus_form(B_r, V_s))
                tests.append(_bewerte(W, n, xs)[0])
                vals.append(ridge_loo(born(bew.x, W), bew.y)[0])
            z["zufall_test"], z["zufall_val"] = tests, vals
            z["zufall_raenge_ok"] = bool(all(x == r for x in raenge))
            z["zufall_gleich_suche"] = gleich
            z["zufall_gleicher_rang"] = float(np.median(tests))
            z["zufall_gleicher_rang_val"] = float(np.median(vals))
            z["zufall_iqr"] = float(np.subtract(*np.percentile(tests, [75, 25])))
            if not z["zufall_raenge_ok"]:                  # darf nach (E3) nie passieren
                z["analysierbar"] = z["gueltig"] = False
    z["kopf_sekunden"] = time.time() - tk
    z["sekunden"] = time.time() - t0
    return z


def _fremd_seed(a):
    """Zweite Stufe (T5): (ds, seed, l2, V_s, rang_s, [(t, V_t, maske_t, rang_t), ...]) -> Zeile fremde_maske."""
    ds, s, l2, V_s, rang_s, spender = a
    n, t0 = QUBITS_JE_DATENSATZ[ds], time.time()
    xtr, ytr, xte, yte = v2._daten(ds, s)
    xs = (normiere(xtr), ytr, normiere(xte), yte, l2)
    xn = normiere(xtr)
    V_s = np.asarray(V_s, dtype=np.int64)
    z = {"seed": s, "datensatz": ds, "spender": []}
    for t, V_t, m_t, rang_t in spender:
        m_t = np.asarray(m_t, dtype=float)
        W = matrix_vm(V_s, m_t)
        B_u, einh, quad = form_aus_maske(m_t, V_s)
        z["spender"].append({"seed": t, "test": _bewerte(W, n, xs)[0], "val": ridge_loo(born(xn, W), ytr)[0],
                             "rang_uebertragen": rang_gf2(B_u), "rang_spender": rang_t,
                             "gleiches_V": bool(set(V_t) == set(V_s.tolist())), "einheitlich": bool(einh and quad)})
    if z["spender"]:
        z["fremde_maske"] = float(np.median([x["test"] for x in z["spender"]]))
        z["fremde_maske_val"] = float(np.median([x["val"] for x in z["spender"]]))
        gl = [x["test"] for x in z["spender"] if x["rang_uebertragen"] == rang_s]
        z["fremde_maske_gleicher_rang"] = float(np.median(gl)) if gl else None
    z["sekunden"] = time.time() - t0
    return z


def _arbeiter(a):
    art, arg = a
    if art == "struktur":
        z = v1._struktur_seed(arg)
        return _schluessel(a), art, z
    if art == "fremd":
        return _schluessel(a), art, _fremd_seed(arg)
    return _schluessel(a), art, _mess_seed(arg)


def _schluessel(a) -> str:
    art, arg = a
    if art == "struktur":
        return f"struktur|{arg[0]}"
    if art == "fremd":                                  # Spenderliste im Schluessel: andere Spender -> neu rechnen
        return f"fremd|{arg[0]}|{arg[1]}|" + ",".join(str(x[0]) for x in arg[5])
    return f"mess|{arg[0]}|{arg[1]}"


# ------------------------------------------------------------------ Wiederaufnahme (wie I-036-v2)
def _lade_zwischen(pfad: Path, kennung: dict) -> dict:
    if not pfad.exists():
        return {}
    inhalt = json.loads(pfad.read_text())
    if inhalt.get("kennung") != kennung:
        raise ValueError(f"Zwischendatei {pfad} stammt aus anderen Einstellungen ({inhalt.get('kennung')}) -- "
                         "nicht mischen; Datei sichern/umbenennen und neu starten")
    return inhalt["zeilen"]


def _schreibe_zwischen(pfad: Path, kennung: dict, zeilen: dict):
    """Atomar: erst .tmp vollstaendig schreiben, dann os.replace."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"karte": KARTE, "kennung": kennung, "zeilen": zeilen}, ensure_ascii=False))
    os.replace(tmp, pfad)


def _rechne(aufgaben, pfad: Path, kennung: dict, fn=None, prozesse=None) -> dict:
    """Jeder fertige Seed geht SOFORT (im Hauptprozess) in die Zwischendatei; fertige Seeds werden uebersprungen."""
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


def fremd_aufgaben(zeilen: dict) -> list:
    """T5: je gueltigem Seed s alle Spender t != s desselben Datensatzes mit V_s ⊆ V_t."""
    aufg = []
    for ds, zz in zeilen.items():
        gut = [z for z in zz if z.get("gueltig")]
        for z in gut:
            sp = [(t["seed"], t["V_s"], t["maske_s"], t["rang"]) for t in gut
                  if t["seed"] != z["seed"] and teilraum(z["V_s"], t["V_s"])]
            aufg.append(("fremd", (ds, z["seed"], z["kopf_l2"], z["V_s"], z["rang"], sp)))
    return aufg


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, n_seeds=N_SEEDS, seeds=None, seeds_struktur=SEEDS_STRUKTUR, datensaetze=DATENSAETZE,
         schritte=v1.SA_SCHRITTE, politur_max=v1.POLITUR_MAX, n_zufall=v1.N_ZUFALL, n_formen=N_FORMEN,
         mit_struktur=True, zwischen=None, prozesse=None):
    datensaetze = tuple(datensaetze)
    if schnell:                                                          # nur Entwurfs-Seeds
        seeds, seeds_struktur = ENTWURF_SEEDS, ENTWURF_STRUKTUR
        schritte, politur_max, n_zufall = 80, 20, 5
    else:
        if n_seeds not in (10, 20):
            raise ValueError("n_seeds ist vorregistriert: 20 (10 nur als Teillauf)")
        if n_formen != N_FORMEN:
            raise ValueError("n_formen ist vorregistriert: 20")
        seeds = seeds or {ds: SEEDS[ds][:n_seeds] for ds in datensaetze}
        if not _pruefe_seeds(seeds, seeds_struktur if mit_struktur else (), datensaetze):
            raise ValueError("Seeds nicht frisch oder nicht disjunkt")
    seeds = {ds: list(seeds[ds]) for ds in datensaetze}
    seeds_struktur = list(seeds_struktur) if mit_struktur else []
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "sa_schritte": schritte, "politur_max": politur_max,
               "n_zufall": n_zufall, "kopf_l2": KOPF_L2, "n_formen": n_formen, "form_seed": FORM_SEED,
               "fremd": "V_s<=V_t"}
    t0 = time.time()
    # Stufe 1: Suche + eigene Arme (MNIST/Fashion zuerst, Strukturseeds fuellen die Luecken)
    aufg = [("mess", (ds, s, schritte, politur_max, n_zufall, KOPF_L2[ds], n_formen))
            for ds in datensaetze for s in seeds[ds]]
    aufg += [("struktur", (s, schritte, politur_max, n_zufall)) for s in seeds_struktur]
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zs = [alle[f"struktur|{s}"] for s in seeds_struktur]
    zeilen = {ds: [alle[f"mess|{ds}|{s}"] for s in seeds[ds]] for ds in datensaetze}
    # Stufe 2: fremde_maske (braucht alle Suchen)
    aufg2 = fremd_aufgaben(zeilen)
    alle = _rechne(aufg2, pfad, kennung, prozesse=prozesse)
    fremd = {(a[1][0], a[1][1]): alle[_schluessel(a)] for a in aufg2}
    for ds in datensaetze:
        for z in zeilen[ds]:
            f = fremd.get((ds, z["seed"]))
            if f is not None:
                z["fremd"] = f
                for k in ("fremde_maske", "fremde_maske_val", "fremde_maske_gleicher_rang"):
                    z[k] = f.get(k)
    for z in zs:
        print(f"  Struktur {z['seed']}: suche {z['suche']:.4f} lehrer {z['lehrer']:.4f} id {z['identitaet']:.4f} "
              f"aequivalent {z['lehrer_aequivalent']}", flush=True)
    for ds in datensaetze:
        for z in zeilen[ds]:
            if z.get("analysierbar"):
                fm = z.get("fremde_maske")
                print(f"  {NAME[ds]} {z['seed']}: suche {z['suche']:.4f} zufall {z['zufall_gleicher_rang']:.4f} "
                      f"(IQR {z['zufall_iqr'] * 100:.2f} pp) ohne_maske {z['ohne_maske']:.4f} fremd "
                      f"{'-' if fm is None else f'{fm:.4f}'} | k {z['k']} rang {z['rang']} ({z['sekunden']:.0f} s)",
                      flush=True)
            elif z.get("gueltig"):
                print(f"  {NAME[ds]} {z['seed']}: Rang 0 (nicht analysierbar), k {z['k']}", flush=True)
            else:
                print(f"  {NAME[ds]} {z['seed']}: AUSGESCHLOSSEN ({_grund(z)})", flush=True)
    e = {"abteilung": "Clifford", "modul": "maske_eigenschaft", "karte": KARTE, "version": 1,
         "hypothesen": HYPOTHESEN, "folgefragen": FOLGEFRAGEN, "seeds": seeds, "seeds_struktur": seeds_struktur,
         "einstellungen": {"sa_schritte": schritte, "politur_max": politur_max, "n_zufall": n_zufall,
                           "tau_start": v1.TAU_START, "tau_ende": v1.TAU_ENDE, "ridge_lambda": v1.RIDGE_LAMBDA,
                           "kopf_l2": KOPF_L2, "tost": TOST_W, "min_gueltig": MIN_GUELTIG, "n_formen": n_formen,
                           "form_seed": FORM_SEED, "fremd": "V_s<=V_t"},
         "zwischendatei": str(pfad), "zeilen": zeilen, "zeilen_struktur": zs, "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, zs)
    e["urteil"] = urteil(e["auswertung"])
    return e


def _grund(z) -> str:
    st = z["suche_struktur"]
    return (f"Bruhat-Form {st['bruhat_form']}, Rekonstruktion {z.get('rekonstruktion_ok')}, Form einheitlich "
            f"{z.get('form_einheitlich')}/quadratisch {z.get('form_quadratisch')}, Form-Rekonstruktion "
            f"{z.get('form_rekonstruktion_ok')}")


# ------------------------------------------------------------------ Auswertung
def _spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def _ds_auswertung(zz_alle) -> dict:
    gueltig = [z for z in zz_alle if z.get("gueltig")]
    zz = [z for z in gueltig if z.get("analysierbar")]
    r = {"n_alle": len(zz_alle), "n_gueltig": len(gueltig), "n": len(zz),
         "ausgeschlossen": [z["seed"] for z in zz_alle if not z.get("gueltig")],
         "rang0": [z["seed"] for z in gueltig if not z.get("analysierbar")],
         "bruhat_form": int(sum(bool(z["suche_struktur"]["bruhat_form"]) for z in zz_alle))}
    r["genug"] = r["n"] >= MIN_GUELTIG
    raenge = [z["rang"] for z in gueltig]
    rv = {"rang_verteilung": {str(k): raenge.count(k) for k in sorted(set(raenge))},
          "k_rang": {f"{k}|{q}": sum(1 for z in gueltig if (z["k"], z["rang"]) == (k, q))
                     for k, q in sorted({(z["k"], z["rang"]) for z in gueltig})}}
    if r["n"] < 2:
        r["beschreibend"] = {"struktur": rv, "unvollstaendig": True}
        return r
    arr = lambda k: np.array([z[k] for z in zz], dtype=float)
    su, zu, om, id_ = arr("suche"), arr("zufall_gleicher_rang"), arr("ohne_maske"), arr("identitaet")
    r["suche_minus_zufall"] = {**_gepaart(su - zu), "p_tost": tost_p(su - zu, TOST_W)}
    r["zufall_minus_ohne_maske"] = {**_gepaart(zu - om), "p_tost": tost_p(zu - om, TOST_W)}
    pk = _gepaart(su - om)
    pk["p_einseitig"] = wilcoxon_einseitig(su - om)
    r["positivkontrolle"] = pk
    r["p0_bestanden"] = bool(r["genug"] and pk["p_einseitig"] < ALPHA and pk["mittel"] >= TOST_W)
    b = {arm: zusammenfassung(arr(arm)) for arm in ARME + ("identitaet", "suche_aus_form")}
    b["struktur"] = rv
    gesamt = float((su - om).mean())
    b["anteil_rang"] = float((zu - om).mean()) / gesamt if gesamt > 0 else None
    rg, ks = arr("rang"), arr("k")
    b["spearman"] = {"rang_suche_minus_ohne": _spearman(rg, su - om), "rang_zufall_minus_ohne": _spearman(rg, zu - om),
                     "rang_suche_minus_zufall": _spearman(rg, su - zu), "k_suche_minus_ohne": _spearman(ks, su - om),
                     "rang_durch_k_suche_minus_ohne": _spearman(rg / ks, su - om)}
    mit = [i for i, z in enumerate(zz) if z.get("fremde_maske") is not None]
    f = {"n_mit_spender": len(mit), "n_spender": [len(zz[i]["fremd"]["spender"]) if "fremd" in zz[i] else 0
                                                  for i in range(len(zz))]}
    if len(mit) >= 2:
        fm = np.array([zz[i]["fremde_maske"] for i in mit])
        f["fremde_minus_ohne"] = _gepaart(fm - om[mit])
        f["suche_minus_fremde"] = _gepaart(su[mit] - fm)
        f["fremde_minus_zufall"] = _gepaart(fm - zu[mit])
        sp = [x for i in mit for x in zz[i]["fremd"]["spender"]]
        f["anteil_gleicher_rang"] = float(np.mean([x["rang_uebertragen"] == zz[i]["rang"]
                                                   for i in mit for x in zz[i]["fremd"]["spender"]]))
        f["anteil_gleiches_V"] = float(np.mean([x["gleiches_V"] for x in sp]))
        gl = [i for i in mit if zz[i].get("fremde_maske_gleicher_rang") is not None]
        f["n_gleicher_rang"] = len(gl)
        if len(gl) >= 2:
            fg = np.array([zz[i]["fremde_maske_gleicher_rang"] for i in gl])
            f["gleicher_rang_suche_minus_fremde"] = _gepaart(su[gl] - fg)
            f["gleicher_rang_fremde_minus_zufall"] = _gepaart(fg - zu[gl])
    b["fremde_maske"] = f
    b["zufall_iqr_mittel"] = float(arr("zufall_iqr").mean())
    b["zufall_gleich_suche"] = int(sum(z["zufall_gleich_suche"] for z in zz))
    b["zufall_ziehungen"] = int(sum(len(z["zufall_test"]) for z in zz))
    b["bahn_min"] = int(min(z["bahn"] for z in zz))
    b["sd"] = {"suche_minus_zufall": float((su - zu).std(ddof=1)), "zufall_minus_ohne": float((zu - om).std(ddof=1)),
               "suche_minus_ohne": float((su - om).std(ddof=1))}
    sv, zv, ov = arr("suche_val"), arr("zufall_gleicher_rang_val"), arr("ohne_maske_val")
    b["val"] = {"suche": float(sv.mean()), "zufall_gleicher_rang": float(zv.mean()), "ohne_maske": float(ov.mean())}
    b["siegerfluch_form"] = float(((sv - zv) - (su - zu)).mean())
    b["val_zufall_besser"] = int(sum(zv > sv))
    b["form_abweichung_max"] = float(np.abs(arr("suche_aus_form") - su).max())
    b["sensitivitaet_suche_minus_id"] = {"mittel": float((su - id_).mean()), "p": wilcoxon_schnell(su - id_)}
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
    "suche_minus_zufall": {
        "gleich": "GLEICHWERTIG: der Rang reicht -- zufaellige quadratische Formen gleichen Rangs auf denselben Bloecken "
                  "V_s leisten dasselbe wie die gesuchte Form",
        "besser": "SUCHE_BESSER: die konkrete Form traegt -- zufaellige Formen gleichen Rangs auf denselben Bloecken V_s "
                  "sind schlechter als die gesuchte",
        "schlechter": "ZUFALL_BESSER: zufaellige Formen gleichen Rangs sind besser als die gesuchte (Siegerfluch der "
                      "Formwahl auf dem LOO-Kriterium)"},
    "zufall_minus_ohne_maske": {
        "gleich": "GLEICHWERTIG: der Rang allein bringt nichts -- zufaellige Formen gleichen Rangs leisten nicht mehr "
                  "als dieselben Bloecke ohne Maske",
        "besser": "ZUFALL_BESSER: der Rang traegt -- schon eine zufaellige Maske gleichen Rangs ist besser als dieselben "
                  "Bloecke ohne Maske",
        "schlechter": "OHNE_BESSER: zufaellige Formen gleichen Rangs schaden -- dieselben Bloecke ohne Maske sind "
                      "besser"},
}


def _gueltig_text(r):
    aus = r["ausgeschlossen"]
    return (f"{r['n']}/{r['n_alle']} Seeds analysierbar (Bruhat-Form {r['bruhat_form']}/{r['n_alle']}, "
            f"{len(aus)} ausgeschlossen{' ' + str(aus) if aus else ''}, Rang 0: {len(r['rang0'])}"
            f"{' ' + str(r['rang0']) if r['rang0'] else ''})")


def _urteil_r(h, ds, k, a, q0_h1) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"HU: keine {NAME[ds]}-Seeds"
    if not r["genug"]:
        return f"HU (Vorbedingung): {NAME[ds]} nur {r['n']} analysierbare Seeds (< {MIN_GUELTIG}) -- " + _gueltig_text(r)
    sp, d, ph, pth = SPEZ[k], r[k], a["holm"][h], a["holm_tost"][h]
    b, pk = r["beschreibend"], r["positivkontrolle"]
    sr = b["spearman"]["rang_suche_minus_ohne"]
    ant = "-" if b["anteil_rang"] is None else f"{b['anteil_rang'] * 100:.0f} %"
    rho = "-" if sr is None else f"{sr['rho']:+.2f}"
    t = (_zahlen(ds, d, ph, pth) + f"; Positivkontrolle {P0_VON[ds]} suche - ohne_maske {pk['mittel'] * 100:+.2f} pp "
         f"(einseitig p = {pk['p_einseitig']:.4f}); Anteil des Rangs {ant}; Spearman rang ~ "
         f"(suche - ohne) {rho}; Siegerfluch der Formwahl "
         f"{b['siegerfluch_form'] * 100:+.2f} pp; Rangverteilung {b['struktur']['rang_verteilung']}; " + _gueltig_text(r))
    if pth < ALPHA:
        if not q0_h1:
            return "HU (Positivkontrolle Q-0): gleich im Rahmen +-0,5 pp, aber Q-0 nicht H1 -- " + t
        if not r["p0_bestanden"]:
            return (f"HU (Positivkontrolle): gleich im Rahmen +-0,5 pp, aber suche - ohne_maske reproduziert den "
                    f"I-036-v2-Effekt nicht ({P0_VON[ds]}: einseitig p < 0,05 und >= 0,5 pp) -- " + t)
        extra = " (Unterschied zusaetzlich nachweisbar, aber kleiner als die Grenze)" if ph < ALPHA else ""
        return sp["gleich"] + f" (+-0,5 pp){extra} -- " + t
    if ph < ALPHA and d["mittel"] > 0:
        return sp["besser"] + " -- " + t
    if ph < ALPHA and d["mittel"] < 0:
        return sp["schlechter"] + " -- " + t
    return "HU: " + t


def _urteil_p0(ds, a) -> str:
    r = a["datensaetze"].get(ds)
    if r is None:
        return f"DURCHGEFALLEN (Vorbedingung): keine {NAME[ds]}-Seeds"
    if not r["genug"]:
        return f"DURCHGEFALLEN (Vorbedingung): nur {r['n']} analysierbare Seeds -- " + _gueltig_text(r)
    pk = r["positivkontrolle"]
    t = (f"{NAME[ds]}: suche - ohne_maske = {pk['mittel'] * 100:+.2f} pp, 90-%-KI [{pk['ki_unten'] * 100:+.2f}; "
         f"{pk['ki_oben'] * 100:+.2f}], einseitiger Wilcoxon p = {pk['p_einseitig']:.4f}, {pk['positiv']}/{pk['n']} "
         f"positiv (I-036-v2: MNIST +1,93 pp 17/20, Fashion +2,36 pp 20/20)")
    if r["p0_bestanden"]:
        return "BESTANDEN: der I-036-v2-Masken-Effekt (suche vor ohne_maske) ist reproduziert -- " + t
    return "DURCHGEFALLEN: der I-036-v2-Masken-Effekt ist nicht reproduziert (einseitig p < 0,05 und >= 0,5 pp) -- " + t


ZUSAMMENSCHAU = {("GLEICHWERTIG", "ZUFALL_BESSER"): "RANG_TRAEGT: der Rang der quadratischen Form traegt den Masken-"
                                                      "Effekt, die konkrete Form nicht",
                 ("SUCHE_BESSER", "GLEICHWERTIG"): "FORM_TRAEGT: nur die konkrete, gesuchte Form traegt; eine Form "
                                                     "gleichen Rangs bringt nichts",
                 ("SUCHE_BESSER", "ZUFALL_BESSER"): "RANG_UND_FORM: der Rang traegt einen Teil, die konkrete Form "
                                                      "den Rest"}


def _zusammenschau(u, h1, h2, ds, a) -> str:
    m1, m2 = u[h1].split(":")[0].split(" ")[0], u[h2].split(":")[0].split(" ")[0]
    r = a["datensaetze"].get(ds, {})
    ant = r.get("beschreibend", {}).get("anteil_rang")
    t = (f" ({NAME[ds]}: {h1} {m1}, {h2} {m2}; Anteil des Rangs "
         f"{'-' if ant is None else f'{ant * 100:.0f} %'})")
    return ZUSAMMENSCHAU.get((m1, m2), "OFFEN: keine eindeutige Zerlegung") + t


def _beschreibung(ds, r) -> str:
    b = r["beschreibend"]
    if b.get("unvollstaendig"):
        return f"{NAME[ds]}: zu wenige analysierbare Seeds ({_gueltig_text(r)}), Raenge {b['struktur']}"
    g = lambda x: (f"{x['mittel'] * 100:+.2f} pp ({x['positiv']}/{x['n']}, p = {x['p']:.4f})" if x else "-")
    sp = lambda x: "-" if x is None else f"{x['rho']:+.2f} (p = {x['p']:.3f}, n = {x['n']})"
    f, s = b["fremde_maske"], b["spearman"]
    tf = (f"fremde_maske (V_s ⊆ V_t) in {f['n_mit_spender']}/{r['n']} Seeds: fremde - ohne {g(f['fremde_minus_ohne'])}, "
          f"suche - fremde {g(f['suche_minus_fremde'])}, fremde - zufall {g(f['fremde_minus_zufall'])}, gleicher Rang "
          f"uebertragen {f['anteil_gleicher_rang'] * 100:.0f} %, gleiches V {f['anteil_gleiches_V'] * 100:.0f} %"
          + (f", nur gleicher Rang ({f['n_gleicher_rang']}): suche - fremde "
             f"{g(f['gleicher_rang_suche_minus_fremde'])}" if "gleicher_rang_suche_minus_fremde" in f else "")
          if "fremde_minus_ohne" in f else f"fremde_maske nur in {f['n_mit_spender']}/{r['n']} Seeds (zu wenige)")
    ant = "-" if b["anteil_rang"] is None else f"{b['anteil_rang'] * 100:.0f} %"
    return (f"{NAME[ds]}: suche {b['suche']['mittel']:.4f} | zufall_gleicher_rang {b['zufall_gleicher_rang']['mittel']:.4f}"
            f" | ohne_maske {b['ohne_maske']['mittel']:.4f} (id {b['identitaet']['mittel']:.4f}); Anteil des Rangs "
            f"{ant}; sd suche - zufall "
            f"{b['sd']['suche_minus_zufall'] * 100:.2f} pp, zufall - ohne {b['sd']['zufall_minus_ohne'] * 100:.2f} pp; "
            f"Raenge {b['struktur']['rang_verteilung']}, (k|Rang) {b['struktur']['k_rang']}; Spearman ueber Seeds: "
            f"rang ~ (suche - ohne) {sp(s['rang_suche_minus_ohne'])}, rang ~ (zufall - ohne) "
            f"{sp(s['rang_zufall_minus_ohne'])}, rang ~ (suche - zufall) {sp(s['rang_suche_minus_zufall'])}, k ~ (suche - "
            f"ohne) {sp(s['k_suche_minus_ohne'])}; {tf}; Ziehungen: IQR {b['zufall_iqr_mittel'] * 100:.2f} pp, = B_s in "
            f"{b['zufall_gleich_suche']}/{b['zufall_ziehungen']} (kleinste Bahn {b['bahn_min']}); LOO: suche "
            f"{b['val']['suche']:.4f}, zufall {b['val']['zufall_gleicher_rang']:.4f}, ohne {b['val']['ohne_maske']:.4f}, "
            f"Siegerfluch der Formwahl {b['siegerfluch_form'] * 100:+.2f} pp; Form-Rekonstruktion max |Abweichung| "
            f"{b['form_abweichung_max'] * 100:.2f} pp; Sensitivitaet suche - id "
            f"{b['sensitivitaet_suche_minus_id']['mittel'] * 100:+.2f} pp; {b['suche_bewertungen']:.0f} Bew., "
            f"{b['sekunden']:.0f} s je Seed; " + _gueltig_text(r))


def urteil(a: dict) -> dict:
    u = {"Q-0": (v1.urteil({"Q0": a["Q0"], "n_struktur": a["n_struktur"]})["Q-0"] if "Q0" in a
                 else "HU (Vorbedingung): keine Strukturseeds")}
    q0_h1 = u["Q-0"].startswith("H1")
    u["P-0"] = _urteil_p0("mnist", a)
    u["P-0F"] = _urteil_p0("fashion_mnist", a)
    for glieder in FAMILIEN.values():
        for h, (ds, k) in glieder.items():
            u[h] = _urteil_r(h, ds, k, a, q0_h1)
    u["R-Z"] = ("beschreibend (aus R-1/R-2 bzw. R-3/R-4, kein eigenes Urteil): MNIST "
                + _zusammenschau(u, "R-1", "R-2", "mnist", a) + " | Fashion " + _zusammenschau(u, "R-3", "R-4",
                                                                                              "fashion_mnist", a))
    tb = [_beschreibung(ds, r) for ds, r in a["datensaetze"].items()]
    u["R-B"] = "beschreibend: " + (" | ".join(tb) if tb else "keine Mess-Seeds")
    return u


# ------------------------------------------------------------------ Vortest: Bausteine
def _vortest_bausteine() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    rng = np.random.default_rng(3610)
    d = 2 ** N
    pop = np.array([bin(i).count("1") for i in range(d)])

    def zufalls_unterraum(k):
        V = np.array([0])
        while len(V) < 2 ** k:
            V = span(rref_basis(list(V) + [int(rng.integers(1, d))]))
        return V

    def affiner_faktor(V):
        chi = np.ones(d)
        for a0 in nebenklassen(V, d):
            w = int(rng.integers(d))
            chi[int(a0) ^ V] = rng.choice([-1.0, 1.0]) * (-1.0) ** pop[w & V]
        return chi

    # (E1) Rang, Bahngroesse: Summe ueber Raenge = 2^(k(k-1)/2); Aufzaehlung fuer k <= 4
    from itertools import product
    for k in range(N + 1):
        if sum(anzahl_formen(k, r) for r in range(k + 1)) != 2 ** (k * (k - 1) // 2):
            fehler(f"anzahl_formen summiert nicht (k = {k})")
    for k in range(1, 5):
        paare = [(i, j) for i in range(k) for j in range(i + 1, k)]
        zaehl = {}
        for bits in product([0, 1], repeat=len(paare)):
            B = np.zeros((k, k), np.int64)
            for (i, j), x in zip(paare, bits):
                B[i, j] = B[j, i] = x
            zaehl[rang_gf2(B)] = zaehl.get(rang_gf2(B), 0) + 1
        if any(zaehl.get(r, 0) != anzahl_formen(k, r) for r in range(k + 1)):
            fehler(f"Aufzaehlung der Formen != anzahl_formen (k = {k}): {zaehl}")
    # (E2) Zufalls-Clifford in Bruhat-Form: Form einheitlich und quadratisch (T3); maske_aus_form(B_s) = Schaltung
    #      (gleiches q -> gleiche Merkmale); affiner Faktor aendert B nicht; Zeilenreihenfolge egal
    faelle, mit_rang = [], 0
    for _ in range(14):
        U = unitaer_clifford(rng.integers(0, 4, (LAGEN, N)), N)
        if not struktur(U)["bruhat_form"]:
            fehler("Zufalls-Clifford nicht in Bruhat-Form")
            continue
        V_s, m_s, m_roh = zerlege(U)
        B_s, einh, quad = form_aus_maske(m_s, V_s)
        if not (einh and quad):
            fehler(f"Form der Suchmaske nicht einheitlich/quadratisch (k = {len(rref_basis(V_s))})")
        if not np.array_equal(form_aus_maske(m_roh, V_s)[0], B_s) or \
                not np.array_equal(form_aus_maske(m_s * affiner_faktor(V_s), V_s)[0], B_s):
            fehler("Form haengt vom affinen Charakter ab")
        if not np.array_equal(B_s, B_s.T) or np.any(np.diag(B_s)) or rang_gf2(B_s) % 2:
            fehler("Form nicht alternierend")
        m_f = maske_aus_form(B_s, V_s)
        if not (np.array_equal(m_f, m_s) and aequivalent(U, matrix_vm(V_s, m_f))):
            fehler("maske_aus_form(B_s) != kanonische Maske bzw. nicht aequivalent zur Schaltung")
        if not aequivalent(matrix_vm(V_s, m_f), matrix_explizit(V_s, m_f)):
            fehler("maske_aus_form: matrix_vm != explizite Konstruktion")
        # andere Basis: Form transformiert sich kongruent, Rang bleibt
        alt = zufalls_basis(V_s, rng)
        if rang_gf2(form_aus_maske(m_s, V_s, alt)[0]) != rang_gf2(B_s):
            fehler("Rang haengt von der Basis ab")
        mit_rang += rang_gf2(B_s) >= 2
        faelle.append((U, V_s, m_s, B_s))
    if mit_rang < 3:
        fehler(f"zu wenige Zufalls-Clifford mit Rang >= 2 ({mit_rang}) -- (E3) nicht scharf")
    # (E3) Zufallsformen: gleicher Rang (immer), Gleichverteilung auf der Bahn (k = 3, Rang 2: 7 Formen),
    #      = explizite lineare Umbenennung von V (Merkmale gleich); verschieden von B_s meist
    B3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    zaehl = {}
    for _ in range(3500):
        Br = zufall_form(B3, rng)
        zaehl[Br.tobytes()] = zaehl.get(Br.tobytes(), 0) + 1
        if rang_gf2(Br) != 2:
            fehler("zufall_form aendert den Rang (k = 3)")
            break
    if len(zaehl) != anzahl_formen(3, 2) or max(zaehl.values()) > 1.25 * 500 or min(zaehl.values()) < 0.75 * 500:
        fehler(f"zufall_form nicht gleichverteilt auf der Bahn: {sorted(zaehl.values())}")
    verschieden = 0
    for U, V_s, m_s, B_s in faelle:
        k, r = len(B_s), rang_gf2(B_s)
        for _ in range(4):
            P = zufall_gl(k, rng) if k else np.zeros((0, 0), np.int64)
            Br = (P.T @ B_s @ P) % 2
            if rang_gf2(Br) != r or np.any(np.diag(Br)) or not np.array_equal(Br, Br.T):
                fehler("P^T B P hat anderen Rang / ist nicht alternierend")
            # explizite Umbenennung: m_rel(a0 ^ koord[t]) = m_s(a0 ^ koord[P t])
            basis = rref_basis(V_s)
            koord, bits = _koord(basis), _bits(k)
            ziel = (bits @ P.T) % 2 if k else np.zeros((1, 0), np.int64)
            idx = (ziel * (1 << (k - 1 - np.arange(k)))[None, :]).sum(1) if k else np.zeros(1, np.int64)
            m_rel = np.ones(d)
            for a0 in nebenklassen(V_s, d):
                m_rel[int(a0) ^ koord] = m_s[int(a0) ^ koord[idx]]
            if not aequivalent(matrix_vm(V_s, m_rel), matrix_vm(V_s, maske_aus_form(Br, V_s))):
                fehler("Zufallsform != explizit umbenannte Maske")
            if not np.array_equal(form_aus_maske(maske_aus_form(Br, V_s), V_s)[0], Br):
                fehler("form_aus_maske(maske_aus_form(B)) != B")
            verschieden += bool(r >= 2 and not np.array_equal(Br, B_s))
    if verschieden == 0:
        fehler("Zufallsformen nie von B_s verschieden")
    # (E4) q ≡ 0 -> ohne_maske exakt; Rang 0 -> Zufallsform bleibt 0
    for U, V_s, m_s, B_s in faelle[:4]:
        Z = np.zeros_like(B_s)
        if not np.array_equal(maske_aus_form(Z, V_s), np.ones(d)) or not aequivalent(
                matrix_vm(V_s, maske_aus_form(Z, V_s)), matrix_vm(V_s, np.ones(d))):
            fehler("q ≡ 0 ergibt nicht ohne_maske")
        if np.any(zufall_form(Z, rng)):
            fehler("Zufallsform von q ≡ 0 nicht 0")
        if rang_gf2(B_s) >= 2 and aequivalent(matrix_vm(V_s, m_s), matrix_vm(V_s, np.ones(d))):
            fehler("echte Form (Rang >= 2) ist trotzdem = ohne_maske")
    # (E5) fremde_maske: V_s ⊆ V_t -> konventionsfrei (affiner Faktor je V_t-Nebenklasse wirkungslos), Form =
    #      Einschraenkung; V_s ⊄ V_t -> Mehrdeutigkeit quadratisch (Begruendung von T5)
    for _ in range(10):
        k_t = int(rng.integers(3, 8))
        V_t = zufalls_unterraum(k_t)
        bt = rref_basis(V_t)
        B_t = np.zeros((k_t, k_t), np.int64)
        iu = np.triu_indices(k_t, 1)
        B_t[iu] = rng.integers(0, 2, len(iu[0]))
        B_t = B_t + B_t.T
        m_t = maske_aus_form(B_t, V_t) * affiner_faktor(V_t)
        sub = span([bt[j] for j in sorted(rng.choice(k_t, int(rng.integers(1, k_t)), replace=False))]
                   + [0]) if k_t > 1 else np.array([0])
        sub = span(zufalls_basis(sub, rng)) if len(sub) > 1 else sub
        if not teilraum(sub, V_t):
            fehler("Teilraum-Konstruktion falsch")
        B_u, einh, quad = form_aus_maske(m_t, sub)
        if not (einh and quad) or not np.array_equal(B_u, einschraenkung(B_t, V_t, sub)):
            fehler("uebertragene Form != Einschraenkung B_t|V_s")
        if not aequivalent(matrix_vm(sub, m_t), matrix_vm(sub, m_t * affiner_faktor(V_t))):
            fehler("fremde_maske haengt bei V_s ⊆ V_t vom affinen Charakter des Spenders ab")
    mehrdeutig = 0
    for _ in range(10):
        V_t, V_s = zufalls_unterraum(4), zufalls_unterraum(4)
        if teilraum(V_s, V_t):
            continue
        m_t = maske_aus_form(np.zeros((4, 4), np.int64), V_t)
        mehrdeutig += not aequivalent(matrix_vm(V_s, m_t), matrix_vm(V_s, m_t * affiner_faktor(V_t)))
    if mehrdeutig == 0:
        fehler("Mehrdeutigkeit fuer V_s ⊄ V_t nie gesehen -- Begruendung von T5 nicht belegt")
    # fremd_aufgaben: nur Spender mit V_s ⊆ V_t, nie der Seed selbst, nur gueltige
    Va, Vb = span([0x40, 0x20]), span([0x40, 0x20, 0x01])
    zz = {"mnist": [{"seed": 1, "gueltig": True, "V_s": Va.tolist(), "maske_s": [1] * d, "rang": 2, "kopf_l2": 0.003},
                    {"seed": 2, "gueltig": True, "V_s": Vb.tolist(), "maske_s": [1] * d, "rang": 2, "kopf_l2": 0.003},
                    {"seed": 3, "gueltig": False, "V_s": Vb.tolist(), "maske_s": [1] * d, "rang": 2, "kopf_l2": 0.003},
                    {"seed": 4, "gueltig": True, "V_s": span([0x80]).tolist(), "maske_s": [1] * d, "rang": 0,
                     "kopf_l2": 0.003}]}
    fa = {a[1][1]: [x[0] for x in a[1][5]] for a in fremd_aufgaben(zz)}
    if fa != {1: [2], 2: [], 4: []}:
        fehler(f"fremd_aufgaben waehlt falsche Spender: {fa}")
    if _schluessel(("fremd", ("mnist", 1, 0.003, [], 2, [(2, [], [], 2), (7, [], [], 0)]))) != "fremd|mnist|1|2,7":
        fehler("Schluessel der zweiten Stufe ohne Spenderliste")
    # (E6) Kopf: W(V_s, maske_aus_form(B_s)) = Schaltung auch nach dem Kopf (Numerik)
    xtr = torch.tensor(rng.random((400, d)) ** 3, dtype=torch.float32)
    xte = torch.tensor(rng.random((300, d)) ** 3, dtype=torch.float32)
    ytr, yte = torch.tensor(rng.integers(0, 10, 400)), torch.tensor(rng.integers(0, 10, 300))
    ytr[:200] = xtr[:200, :10].argmax(1)
    yte[:150] = xte[:150, :10].argmax(1)
    xs = (normiere(xtr), ytr, normiere(xte), yte, 0.003)
    codes = rng.integers(0, 4, (LAGEN, N))
    Uc = unitaer_clifford(codes, N)
    V_c, m_c, _ = zerlege(Uc)
    a1, a2 = _bewerte(codes, N, xs)[0], _bewerte(matrix_vm(V_c, maske_aus_form(form_aus_maske(m_c, V_c)[0], V_c)),
                                                 N, xs)[0]
    if abs(a1 - a2) > 2 / 300:
        fehler(f"Kopf auf W(V_s, maske_aus_form(B_s)) {a2:.4f} != Kopf auf Schaltung {a1:.4f}")
    # (E7) Statistik: einseitiger Wilcoxon, Spearman-Waechter
    for n in (1, 4, 7, 10):
        for _ in range(6):
            dd = rng.normal(rng.normal(0, 0.003), 0.004, n)
            rr = np.argsort(np.argsort(np.abs(dd))) + 1
            w = rr[dd > 0].sum()
            p_auf = np.mean([sum(x for x, sg in zip(rr, v) if sg) >= w for v in product([0, 1], repeat=n)])
            if abs(wilcoxon_einseitig(dd) - p_auf) > 1e-12:
                fehler(f"wilcoxon_einseitig != Aufzaehlung (n = {n})")
                break
    dd = rng.normal(0.001, 0.004, 11)
    if abs(wilcoxon_schnell(dd) - wilcoxon_exakt(dd)) > 1e-12:
        fehler("wilcoxon_schnell != wilcoxon_exakt")
    if _spearman([2, 2, 2], [1, 2, 3]) is not None or abs(_spearman([1, 2, 3, 4], [2, 4, 6, 9])["rho"] - 1) > 1e-12:
        fehler("Spearman-Waechter falsch")
    # (E8) Familien und Seeds
    if (set(FAMILIEN["MNIST"]) != {"R-1", "R-2"} or set(FAMILIEN["Fashion"]) != {"R-3", "R-4"}
            or {ds for g in FAMILIEN.values() for ds, _ in g.values()} != set(DATENSAETZE)):
        fehler("Familien falsch")
    for n in (10, 20):
        if not _pruefe_seeds({ds: SEEDS[ds][:n] for ds in DATENSAETZE}, SEEDS_STRUKTUR, DATENSAETZE):
            fehler(f"Seeds (n = {n}) nicht frisch/disjunkt")
    for schlecht in ({"mnist": (700,), "fashion_mnist": (781,)}, {"mnist": (745,), "fashion_mnist": (781,)},
                     {"mnist": (759,), "fashion_mnist": (781,)}, {"mnist": (640,), "fashion_mnist": (781,)},
                     {"mnist": (942,), "fashion_mnist": (781,)}, {"mnist": (810,), "fashion_mnist": (781,)},
                     {"mnist": (780,), "fashion_mnist": (780,)}, {"mnist": (300,), "fashion_mnist": (781,)}):
        if _pruefe_seeds(schlecht, (), DATENSAETZE):
            fehler(f"belegte/doppelte Seeds durchgelassen: {schlecht}")
    for kw in ({"n_seeds": 15}, {"n_formen": 5}):
        try:
            lauf(**kw)
            fehler(f"{kw} nicht abgelehnt")
        except ValueError:
            pass
    return ok


# ------------------------------------------------------------------ Vortest: Wiederaufnahme
def _test_arbeiter(a):
    """Ersatz-Arbeiter (Modulebene, damit spawn ihn findet)."""
    art, arg = a
    if arg[-1] == "abbruch":
        raise RuntimeError("simulierter Abbruch")
    if art == "struktur":
        return _schluessel(a), art, {"seed": arg[0], "pid": os.getpid()}
    if art == "fremd":
        return _schluessel(a), art, {"seed": arg[1], "fremde_maske": 0.5}
    return _schluessel(a), art, {"seed": arg[1], "datensatz": arg[0], "wert": arg[1] * 0.5}


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
        # zweite Stufe in derselben Datei: Abbruch mitten in Stufe 2, Stufe 1 bleibt erhalten, Neustart ergaenzt;
        # andere Spenderliste -> neu gerechnet
        st2 = [("fremd", ("mnist", s, 0.003, [], 2, [(t, [], [], 2)], "x")) for s, t in ((1, 2), (2, 1), (3, 1))]
        st2_kaputt = st2[:1] + [("fremd", ("mnist", 2, 0.003, [], 2, [(1, [], [], 2)], "abbruch"))] + st2[2:]
        try:
            _rechne(st2_kaputt, pfad, kennung, fn=_test_arbeiter, prozesse=1)
            fehler("simulierter Abbruch (Stufe 2) nicht ausgeloest")
        except RuntimeError:
            pass
        roh = json.loads(pfad.read_text())["zeilen"]
        if set(roh) != {"mess|mnist|1", "mess|mnist|2", "mess|mnist|3", "struktur|9", "fremd|mnist|1|2"}:
            fehler(f"Stufe 2 nach Abbruch falsch: {sorted(roh)}")
        gerufen.clear()
        alle = _rechne(st2, pfad, kennung, fn=zaehler, prozesse=1)
        if sorted(gerufen) != ["fremd|mnist|2|1", "fremd|mnist|3|1"] or len(alle) != 7:
            fehler(f"Neustart Stufe 2 rechnet {gerufen}")
        gerufen.clear()
        _rechne([("fremd", ("mnist", 1, 0.003, [], 2, [(2, [], [], 2), (3, [], [], 2)], "x"))], pfad, kennung,
                fn=zaehler, prozesse=1)
        if gerufen != ["fremd|mnist|1|2,3"]:
            fehler("andere Spenderliste wird nicht neu gerechnet")
        try:
            _rechne(aufg, pfad, {"karte": KARTE, "test": 2}, fn=_test_arbeiter, prozesse=1)
            fehler("andere Einstellungen nicht abgelehnt")
        except ValueError:
            pass
        # Parallelpfad (spawn, imap_unordered)
        pfad2 = Path(tmp) / "_zwischen_par.json"
        par = [("mess", ("fashion_mnist", s, "x")) for s in range(6)]
        alle = _rechne(par, pfad2, kennung, fn=_test_arbeiter, prozesse=2)
        if len(alle) != 6 or len(json.loads(pfad2.read_text())["zeilen"]) != 6:
            fehler("Parallelpfad schreibt nicht alle Seeds")
    return ok


# ------------------------------------------------------------------ Vortest: Urteile
def _synth(rng, d1=0.0, d2=0.0, sd=0.001, n=20, ungueltig=0, rang0=0, fremd=True):
    """d1 = suche - zufall_gleicher_rang, d2 = zufall_gleicher_rang - ohne_maske (P-0: d1 + d2)."""
    zeilen = []
    for i in range(n):
        su = 0.94 + rng.normal(0, 0.002)
        k = int(rng.integers(4, 8))
        gut = i >= ungueltig
        anal = gut and i >= ungueltig + rang0
        r = int(2 * rng.integers(1, k // 2 + 1)) if anal else 0
        z = {"seed": i, "datensatz": "x", "suche": su, "identitaet": su - 0.035, "suche_val": su,
             "identitaet_val": su - 0.03, "suche_bewertungen": 5000, "suche_sekunden": 380.0, "kopf_sekunden": 20.0,
             "sekunden": 420.0, "gueltig": gut, "analysierbar": anal,
             "suche_struktur": {"k": k, "bruhat_form": gut, "maske_noetig": anal, "walsh_aequivalent": False}}
        if gut:
            z.update({"k": k, "rang": r, "bahn": anzahl_formen(k, r), "rekonstruktion_ok": True,
                      "form_einheitlich": True, "form_quadratisch": True, "form_rekonstruktion_ok": True,
                      "suche_aus_form": su, "suche_aus_form_val": su, "ohne_maske_val": su - 0.012})
            if anal:
                zu = su - d1 + rng.normal(0, sd)
                z["ohne_maske"] = zu - d2 + rng.normal(0, sd)
                z.update({"zufall_gleicher_rang": zu, "zufall_gleicher_rang_val": su - 0.004,
                          "zufall_test": [zu] * 3, "zufall_val": [su - 0.004] * 3, "zufall_iqr": 0.002,
                          "zufall_gleich_suche": 0, "zufall_raenge_ok": True})
                if fremd and i % 3:
                    fm = zu + rng.normal(0, sd)
                    z.update({"fremde_maske": fm, "fremde_maske_val": su - 0.005,
                              "fremde_maske_gleicher_rang": fm if i % 2 else None,
                              "fremd": {"spender": [{"seed": 99, "test": fm, "val": su, "rang_uebertragen": r if i % 2
                                                     else 0, "rang_spender": r, "gleiches_V": False,
                                                     "einheitlich": True}]}})
            else:
                z["ohne_maske"] = su
        zeilen.append(z)
    return zeilen


def _vortest_urteile(ziehungen=5) -> bool:
    ok = True
    rng = np.random.default_rng(3611)
    gut = lambda: v1._synth_struktur(rng, 0.925)
    schlecht = lambda: v1._synth_struktur(rng, 0.79)
    B = "beschreibend"
    faelle = [
        ("Form traegt, Rang nicht", gut, {"d1": 0.015}, {"d1": 0.015},
         {"P-0": ("BESTANDEN", ["MNIST", "I-036-v2"]), "P-0F": ("BESTANDEN", ["Fashion"]), "Q-0": ("H1", []),
          "R-1": ("SUCHE_BESSER", ["konkrete Form traegt", "MNIST", "20/20 Seeds analysierbar", "Holm-p_TOST"]),
          "R-2": ("GLEICHWERTIG", ["Rang allein bringt nichts", "MNIST"]),
          "R-3": ("SUCHE_BESSER", ["Fashion"]), "R-4": ("GLEICHWERTIG", ["Fashion"]),
          "R-Z": (B, ["MNIST FORM_TRAEGT", "Fashion FORM_TRAEGT"]),
          "R-B": (B, ["Spearman ueber Seeds", "fremde_maske (V_s ⊆ V_t)", "Siegerfluch", "Raenge"])}, 0),
        ("Rang reicht", gut, {"d2": 0.015}, {"d2": 0.015},
         {"R-1": ("GLEICHWERTIG", ["der Rang reicht", "MNIST"]), "R-2": ("ZUFALL_BESSER", ["der Rang traegt"]),
          "R-3": ("GLEICHWERTIG", ["Fashion"]), "R-4": ("ZUFALL_BESSER", ["Fashion"]),
          "R-Z": (B, ["MNIST RANG_TRAEGT", "Fashion RANG_TRAEGT", "Anteil des Rangs"])}, 0),
        ("Rang und Form", gut, {"d1": 0.008, "d2": 0.01}, {"d1": 0.008, "d2": 0.01},
         {"R-1": ("SUCHE_BESSER", []), "R-2": ("ZUFALL_BESSER", []), "R-Z": (B, ["MNIST RANG_UND_FORM"])}, 0),
        ("Zufall besser als Suche", gut, {"d1": -0.01, "d2": 0.02}, {"d1": -0.01, "d2": 0.02},
         {"R-1": ("ZUFALL_BESSER", ["Siegerfluch der Formwahl", "MNIST"]), "R-2": ("ZUFALL_BESSER", []),
          "R-3": ("ZUFALL_BESSER", ["Fashion"]), "R-Z": (B, ["OFFEN"])}, 0),
        ("Zufallsform schadet", gut, {"d1": 0.02, "d2": -0.008}, {"d1": 0.02, "d2": -0.008},
         {"R-1": ("SUCHE_BESSER", []), "R-2": ("OHNE_BESSER", ["schaden", "MNIST"]), "R-4": ("OHNE_BESSER", [])}, 0),
        ("Streuung", gut, {"sd": 0.03}, {"sd": 0.03},
         {"R-1": ("HU:", ["MNIST"]), "R-2": ("HU:", []), "R-3": ("HU:", ["Fashion"]), "R-4": ("HU:", [])}, 1),
        ("Positivkontrolle fehlt (d = 0)", gut, {}, {},
         {"P-0": ("DURCHGEFALLEN:", ["MNIST"]), "P-0F": ("DURCHGEFALLEN:", ["Fashion"]),
          "R-1": ("HU (Positivkontrolle)", ["reproduziert den I-036-v2-Effekt nicht", "P-0", "MNIST"]),
          "R-2": ("HU (Positivkontrolle)", []), "R-3": ("HU (Positivkontrolle)", ["P-0F", "Fashion"]),
          "R-4": ("HU (Positivkontrolle)", [])}, 0),
        ("Positivkontrolle zu klein (+0,3 pp)", gut, {"d1": 0.0015, "d2": 0.0015, "sd": 0.0003},
         {"d1": 0.0015, "d2": 0.0015, "sd": 0.0003},
         {"P-0": ("DURCHGEFALLEN:", ["+0.3"]), "R-1": ("HU (Positivkontrolle)", ["I-036-v2"]),
          "R-2": ("HU (Positivkontrolle)", []), "R-3": ("HU (Positivkontrolle)", [])}, 0),
        ("Positivkontrolle fehlt, Unterschied bleibt", gut, {"d1": 0.012, "d2": -0.012}, {},
         {"P-0": ("DURCHGEFALLEN:", []), "R-1": ("SUCHE_BESSER", []), "R-2": ("OHNE_BESSER", [])}, 0),
        ("Q-0 faellt", schlecht, {"d2": 0.015}, {"d2": 0.015},
         {"Q-0": ("H0", []), "R-1": ("HU (Positivkontrolle Q-0)", ["MNIST"]), "R-2": ("ZUFALL_BESSER", []),
          "R-3": ("HU (Positivkontrolle Q-0)", ["Fashion"])}, 1),
        ("ohne Strukturseeds", None, {"d2": 0.015}, {"d2": 0.015},
         {"Q-0": ("HU (Vorbedingung)", []), "R-1": ("HU (Positivkontrolle Q-0)", []),
          "R-3": ("HU (Positivkontrolle Q-0)", [])}, 0),
        ("Ausschluss und Rang 0", gut, {"d1": 0.012, "ungueltig": 2, "rang0": 1}, {},
         {"R-1": ("SUCHE_BESSER", ["17/20 Seeds analysierbar", "2 ausgeschlossen [0, 1]", "Rang 0: 1 [2]"]),
          "R-B": (B, ["17/20 Seeds analysierbar"])}, 0),
        ("zu wenige analysierbar", gut, {"ungueltig": 6, "rang0": 6}, {"ungueltig": 19},
         {"R-1": ("HU (Vorbedingung)", ["nur 8 analysierbare Seeds", "MNIST"]),
          "R-2": ("HU (Vorbedingung)", []), "P-0": ("DURCHGEFALLEN (Vorbedingung)", []),
          "R-3": ("HU (Vorbedingung)", ["Fashion", "nur 1"]), "R-B": (B, ["Fashion: zu wenige analysierbare"])}, 0),
        ("klein, aber gleich", gut, {"d1": 0.002, "d2": 0.012, "sd": 0.0003}, {"d1": -0.002, "d2": 0.012, "sd": 0.0003},
         {"R-1": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"]), "R-3": ("GLEICHWERTIG", ["zusaetzlich nachweisbar"])},
         0),
        ("nur MNIST, ohne fremde", gut, {"d1": 0.012, "fremd": False}, None,
         {"R-1": ("SUCHE_BESSER", []), "R-3": ("HU: keine Fashion", []), "R-4": ("HU: keine Fashion", []),
          "P-0F": ("DURCHGEFALLEN (Vorbedingung)", ["keine Fashion"]), "R-B": (B, ["fremde_maske nur in 0/20"])}, 0),
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
                    print(f"  Vortest [{name}]: {h} -> {u[h][:240]}")
            if not (u["R-B"].startswith("beschreibend") and u["R-Z"].startswith("beschreibend")
                    and all(isinstance(v, str) for v in u.values())):
                ok = False
                print(f"  VORTEST FEHLER [{name}]: R-B/R-Z oder Urteil kein String")
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
        u = urteil(auswertung({ds: _synth(rng, d2=0.015, sd=0.03) for ds in DATENSAETZE}, gut()))
        if any(u[h].startswith("GLEICHWERTIG") for h in ("R-1", "R-3")):
            ok = False
            print("  VORTEST FEHLER: TOST feuert trotz riesiger Streuung")
        # Familien getrennt; Holm x2 innerhalb der Familie
        zm = _synth(rng, d1=0.004, d2=0.006, sd=0.006)
        a1 = auswertung({"mnist": zm}, gut())
        a2 = auswertung({"mnist": zm, "fashion_mnist": _synth(rng, d1=0.02)}, gut())
        if any(abs(a1["holm"][h] - a2["holm"][h]) > 1e-15 for h in ("R-1", "R-2")):
            ok = False
            print("  VORTEST FEHLER: Holm-Familien nicht getrennt")
        r = a1["datensaetze"]["mnist"]
        pm = {h: r[k]["p"] for h, (_, k) in FAMILIEN["MNIST"].items()}
        pt = {h: r[k]["p_tost"] for h, (_, k) in FAMILIEN["MNIST"].items()}
        if (holm(pm) != {h: a1["holm"][h] for h in pm} or holm(pt) != {h: a1["holm_tost"][h] for h in pt}
                or min(a1["holm"]["R-1"], a1["holm"]["R-2"]) != min(1.0, 2 * min(pm.values()))
                or set(a1["holm"]) != {"R-1", "R-2", "R-3", "R-4"}):
            ok = False
            print("  VORTEST FEHLER: Holm x2 falsch")
    return ok


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: (E1-E8) Bausteine -- Rang/Bahngroesse, Form aus Suchmaske einheitlich, gleiche Form = gleiche
    Merkmale, Zufallsformen gleichen Rangs und gleichverteilt, q ≡ 0 = ohne_maske, fremde_maske konventionsfrei,
    Kopf, Statistik, Familien, Seeds; Wiederaufnahme mit simuliertem Abbruch (beide Stufen); jeder Urteilszweig
    synthetisch (Marke UND Text, mehrere Ziehungen), inkl. fehlschlagender Positivkontrolle."""
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
