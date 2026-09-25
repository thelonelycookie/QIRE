"""Abteilung Clifford -- Magiebedarf als Funktion der Aufgaben-Schwierigkeit (I-035).

WARUM DIESE FRAGE (Vorgabe 7, UEBERGABE.md, 25.09.): nach einer Nacht voller
sauber gemessener, aber teils schon publizierter Genres soll das Institut
Fragen bevorzugen, zu denen die Literatursuche nichts Vergleichbares findet.
Karte I-035 (ideen.json) wurde nach genau so einer Pruefung direkt formuliert
(nicht aus dem Zufallsgenerator): 'Retrieving non-stabilizerness with Neural
Networks' (2403.00919) und 'Quantum resources of quantum and classical
variational methods' (2409.13008) sind VERWANDT -- beide verbinden
Nicht-Stabilisierbarkeit/Magie mit Messdaten bzw. Aufgabenleistung -- aber
eine ANDERE Frage: 2403.00919 schaetzt die Magie EINES GEGEBENEN Zustands aus
Messungen (nicht: wie viel Magie eine Aufgabe als Funktion ihrer Schwierigkeit
braucht); 2409.13008 verbindet Magie mit Leistung bei Quanten-Reservoir-
Computing (QELM), nicht bei ueberwachter Bildklassifikation mit Schwierigkeit
als Stellgroesse. Laut Karte I-035 (Feld 'neuheit') ist die hier gestellte
Frage nach Literaturpruefung TEILWEISE/OFFEN -- kein Vergleichsbefund
gefunden, damit nach Vorgabe 7 bevorzugt gegenueber einer weiteren
Nachrechnung bekannter Tricks.

FRAGE (Karte I-035, Felder 'frage'/'quantenkern'): Sagt die AUFGABEN-
Schwierigkeit (Anteil zufaellig vertauschter Trainingsetiketten, abgestuft in
10 Stufen 0/10/20/30/40/50/60/70/85/100 % -- Nachschlag 25.09., urspruenglich
6 Stufen, s. u.) den Magiebedarf M50 (Anteil Nicht-Clifford-Winkel fuer
50 % des Schaltungsnutzens, Definition wie C-mnist) einer MNIST-Aufgabe
voraus -- und gibt es einen BILLIGEN Kennwert AM VOLLEN TRAINIERTEN MODELL
(Generalisierungsluecke = Training- minus Testgenauigkeit; Entropie der
Born-Verteilung), der mit M50 korreliert, OHNE den teuren Rundungs-Sweep?
Vorarbeit im Institut: I-003 zeigte nur ZWEI Punkte (M50 = 1 echt vs. 8
Zufallsetiketten, explorativ, nicht belastbar -- Positivkontrolle scheiterte
dort am Lernverfahren, siehe FEHLERBUCH 24.09. 22:10); I-002 zeigte, dass die
Stabilizer-Renyi-Entropie der GELERNTEN Schicht M50 NICHT vorhersagt (H0,
rho = 0,26) -- ein naheliegender Kennwert kann also durchaus versagen, die
Generalisierungsluecke ist ein anderer, noch nicht getesteter Kandidat.

BAUSTEIN-WIEDERVERWENDUNG (per import, keine gemeinsame Datei veraendert):
  abteilungen/clifford.py: QireZelle, _baue (Arme 'voll'/'ohne_schaltung'),
    _gerundet (Rundung nach Abstand zum Gitter, frischer Kopf -- wie C-mnist),
    runde_clifford, multiplikationen, GITTER, LAGEN.
  labor/gemeinsam.py: bilder (MNIST 16x16), trainiere, genauigkeit,
    wilcoxon_exakt, parallel, speichere, zusammenfassung.
Die Etiketten-Verrauschung folgt magie_aufgabe.py (I-003) 'zufall': je Seed
EINE feste Permutation -- hier aber nur ueber eine TEILMENGE der
Trainingsetiketten (Groesse = Rausch-Anteil * n_train), nicht ueber alle.
Weil nur innerhalb der gewaehlten Teilmenge permutiert wird, bleiben die
Klassenhaeufigkeiten EXAKT erhalten (nicht nur 'weitgehend' wie in der Karte
formuliert). Die interpolierte, stetige M50-Definition folgt demselben
Prinzip wie skala.py/skala_v2.py (kleinste gemessene Stufe mit G >= 0,5,
linear zwischen den Nachbarn interpoliert) -- hier lokal neu geschrieben
(keine Kopplung an skala.py, das parallel von anderen Laeufen veraendert
werden koennte).

RAUSCH-STUFEN, NACHSCHLAG 25.09. (Direktor, nach der ersten Fassung): ZEHN
Stufen (0/10/20/30/40/50/60/70/85/100 %) statt sechs, dafuer NUR 6 statt 10
Seeds je Stufe (Gesamt-Budget Seeds x Stufen bleibt bei 60 Seed-Stufen-
Kombinationen, Laufzeit soll gleich bleiben). BEGRUENDUNG: Die Aufloesung
EINER Rangkorrelation (Spearman) haengt in erster Linie an der Zahl der
PUNKTE auf der x-Achse (hier: Rausch-Stufen), nicht an der Praezision jedes
einzelnen Punkts -- die Praezision je Punkt (wie eng die Bootstrap-KI um den
wahren Stufenmittelwert liegt) hat der Bootstrap ohnehin schon im Griff
(breite KI bei viel Streuung, s. u.); was der Bootstrap NICHT beheben kann,
ist zu WENIGE Punkte fuer eine stabile RANGORDNUNG: mit 6 Stufen ist der
kleinste erreichbare zweiseitige exakte Spearman-p bereits 2/6! ~ 0,0028,
aber die Bootstrap-KI aus nur 6 Punkten ist strukturell breit (siehe Vortest-
Erfahrung unten); mit 9-10 Punkten hat die Rangordnung deutlich mehr moegliche
Zustaende, und ein monotoner Trend ueber 10 Punkte ist robuster gegen den
Ausfall/die Verschiebung eines einzelnen Punkts als ueber 6. 6 statt 10 Seeds
je Stufe verbreitert zwar die KI JEDES einzelnen Stufenmittels etwas (Fehler
der Mittelwertschaetzung skaliert mit 1/sqrt(n): sqrt(10/6) ~= 1,29 -- ein
mittlerer Effekt), das wird aber durch die zusaetzliche Aufloesung (10 statt 6
Ordnungs-Positionen) mehr als ausgeglichen: mit 6 Seeds bleibt wilcoxon_exakt
(2^6 = 64 Vorzeichen) weiterhin trivial billig, und der Vortest (unten) prueft
explizit, dass die Bootstrap-Logik mit 6 Seeds je Stufe noch stabil arbeitet
(Faelle 'monoton'/'flach'/'verrauscht' erneut mit n_seeds=6 erzwungen).
Nebenwirkung, ehrlich benannt: die LERNRATEN-SUCHE braucht eine EIGENE Suche
je STUFE (nicht je Seed), ihre Kosten skalieren also mit der STUFENZAHL, nicht
mit der Seedzahl -- bei zehn statt sechs Stufen steigt sie um den Faktor 10/6,
waehrend die Messphase (Seeds x Stufen = const.) gleich teuer bleibt (s.
Laufzeitschaetzung unten: die Gesamtzeit steigt dadurch leicht, bleibt aber im
Zielkorridor).

RUNDUNGS-SWEEP, BUDGET-ABWAEGUNG (vor dem Lauf, Regel 7):
  10 Rausch-Stufen x >= 6 frische Seeds x eigene Lernrate je (Stufe, Arm) x
  vollen MAGIE_STUFEN-Sweep (clifford.py: 6 Rundungsstufen) waere bei
  C-mnist-Budget (6.000 Trainingsbilder, 1.500 Schritte) weit ueber dem
  Laufzeitziel von ~90 Minuten (C-mnist selbst: EIN Datensatz, EIN
  Rundungssweep, 71 Minuten). Drei Vereinfachungen, hier begruendet:
  (1) REDUZIERTER RUNDUNGS-SWEEP K_STUFEN = (0, 2, 8) statt aller sechs
      MAGIE_STUFEN aus clifford.py. Auf MNIST liegt M50 nach C-mnist/
      C-referenz bei 2-4 von 32 Winkeln und G(8) bereits bei 0,83-0,89 -- drei
      Stuetzstellen (0, 2, 8) reichen, um eine stetige M50-Schaetzung zu
      interpolieren, ohne die Schaltstellen 1/4/12 mitzumessen. Wird die
      Schwelle 0,5 innerhalb von k <= 8 nie erreicht, bleibt M50 fuer diesen
      Seed/diese Stufe UNDEFINIERT (rechtszensiert) -- offen ausgewiesen
      (M50_seeds_undefiniert je Stufe), nicht durch Extrapolation geraten.
  (2) BUDGET STATT C-MNIST-BUDGET: n_train = 2.500, Schritte = 600 (statt
      6.000/1.500) -- wie skala.py fuer denselben Qubit-/Aufloesungsbereich
      (dort mit Begruendung: sonst ist n = 10 nicht in ~1,5 h zu schaffen;
      hier: 6 Stufen statt 1 Datensatz). Das macht absolute Genauigkeiten
      NICHT direkt mit C-mnist vergleichbar; die Groessenordnung von M50 (im
      Bereich 0-8) sollte erhalten bleiben, wird aber nicht vorausgesetzt.
  (3) LERNRATE JE (STUFE, ARM), OHNE SKALA_V2-STABILITAETSABSTAND: Regel 1a
      und die Lehre aus C-referenz/C-skala gelten -- je (Stufe, Arm) eine
      eigene Rate auf eigenen Tuning-Seeds, Gitter beidseitig verlaengert
      (wie clifford.py/magie_aufgabe.py). Der volle Stabilitaetsabstand aus
      skala_v2.py (mehrere Tuning-Seeds, Kriterium = schlechtester Seed,
      Gitterabstand zur naechsten kollabierenden Rate) wird aus Zeitgruenden
      NICHT uebernommen -- er wuerde die Zahl der Trainings je Stufe
      versechsfachen (skala_v2.py: ~50 % laenger als skala.py bei GLEICHER
      Konfigurationszahl, weil jede Rate mehrfach gegen ihren oberen
      Nachbarn geprueft wird). Stattdessen: EINFACHE Ratensuche wie
      clifford.py (2 Tuning-Seeds, Mittelwert-Kriterium) PLUS ein billiger,
      rein deskriptiver Kollaps-Hinweis nach dem Lauf (ein Arm gilt als
      'instabil' auf einem Seed, wenn seine Genauigkeit mehr als
      INSTABIL_GRENZE unter der Identitaet liegt) -- er geht NICHT in die
      Urteile P-0..P-4 ein, macht aber sichtbar, wenn hohe Rauschanteile die
      Rate destabilisieren (genau das Risiko, das die Karte fuer viel
      Rauschen benennt). Wer mehr Zeit hat: skala_v2.waehle_rate ist ein
      fertiger Ersatz fuer lernrate_je_stufe().
  LAUFZEITSCHAETZUNG (INSTITUT_PROZESSE=4, grob; aktualisiert fuer 10 Stufen x
  6 Seeds, Nachschlag 25.09.): C-mnist brauchte fuer ~120 volle Trainings
  (6.000 Bilder, 1.500 Schritte, N=8, Matrixmethode) 71 min, also ~2,4 Kern-
  Minuten je Training. Bei n_train=2.500, Schritte=600 skaliert die Kosten je
  Trainingsschritt kaum, die Gesamtkosten aber ~ (600/1500) * (2500/6000) ~=
  0,17 -> ~0,4 Kern-Minuten je Training. Messphase: 5 Arme (voll,
  ohne_schaltung, k0, k2, k8) x 6 Seeds x 10 Stufen = weiterhin 300 Trainings
  (Seeds x Stufen bewusst konstant gehalten) ~= 120 Kern-Minuten. Ratensuche:
  3 Schluessel (voll, ohne_schaltung, rundung) x 10 Stufen (statt 6) x ~2
  Tuning-Seeds x ~6 Gitterpunkte ~= 360 Trainings ~= 144 Kern-Minuten -- HIER
  steigen die Kosten, weil die Ratensuche je STUFE laeuft, nicht je Seed (s.
  Nachschlag oben): 10/6 so viele Stufen kosten 10/6 so viel Ratensuche, auch
  wenn Seeds x Stufen insgesamt gleich blieb. Summe ~= 264 Kern-Minuten / 4
  Prozesse ~= 66 min, plus Datenladen/Overhead -> GESCHAETZT 65-95 min (statt
  60-90 min in der ersten Fassung mit 6 Stufen). Grobe Schaetzung; haengt von
  der Zahl der Gitter-Erweiterungen ab -- bei knappem Budget zuerst
  tuning_seeds auf ein einzelnes Paar beschraenken (bereits Standard) oder
  notfalls RAUSCH_STUFEN auf 8 Punkte kuerzen, bevor Seeds weiter gesenkt
  werden (Aufloesung vor Praezision, s. o.).

HYPOTHESEN (P-0 .. P-5, 'schaerfen erlaubt' -- UEBERGABE.md Vorgabe 7):
  Die Karte nennt als Abbruchkriterium fuer den billigen Kennwert
  '|Spearman| < 0,5, KI ueberdeckt 0'. Hier GESCHAERFT auf eine einheitliche,
  vorregistrierte Statistik fuer P-1 UND P-2 (dieselbe Bootstrap-KI-Logik,
  TOST-Grenze +-0,3 statt einer festen Rho-Schwelle) -- eine KI, die 0
  ueberdeckt, ist so oder so kein Beleg; die Schaerfung ersetzt nur die
  zusaetzliche Rho-Schwelle durch EINE symmetrische Grenze GRENZE_TOST = 0,3
  auf BEIDEN Seiten (wie clifford.py C-1/skala.py 'H0 (gleich)'): H1 nur, wenn
  die GANZE Bootstrap-KI ausserhalb von +-0,3 liegt, H0, wenn sie GANZ
  innerhalb liegt, sonst HU. Ohne diese Schaerfung koennte eine sehr praezise
  (enge KI), aber winzige Korrelation (z. B. rho = 0,03 mit KI [0,03; 0,03])
  als 'H1, KI deckt 0 nicht' durchgehen, obwohl der Effekt praktisch
  bedeutungslos ist -- ein Randfall, der beim Bau des Vortests auftrat (siehe
  _urteil_korrelation).

  P-0 Positivkontrolle: bei Rauschanteil 0 schlaegt die volle Schaltung die
      Identitaet (Born-Auslese direkt auf x) -- wie C-mnist. Sonst ist jede
      Aussage ueber diese Aufgabe leer.
  P-1 HAUPT (Abbruchkriterium 1 der Karte): M50 (stetig interpoliert wie
      C-mnist/skala.py) zeigt ueber die 10 Rausch-Stufen eine monotone
      Tendenz. Statistik: Spearman-Korrelation zwischen Rausch-Anteil und
      M50-Stufenmittel; Bootstrap-90-%-KI (Resampling des Seed-Index je
      Stufe); Kontrolle aus der Karte ('zufaellige Zuordnung Stufe<->
      Ergebnis') als Permutations-Nullverteilung. GANZE KI ausserhalb +-0,3
      und Richtung eindeutig -> H1; GANZE KI innerhalb +-0,3 (TOST) -> H0;
      sonst HU.
  P-2 HAUPT (Abbruchkriterium 2 der Karte): der billige Kennwert
      Generalisierungsluecke (Trainings- minus Testgenauigkeit des VOLLEN
      Modells, keine zusaetzliche Rundung) korreliert mit M50 ueber dieselben
      Stufen -- gleiche Statistik wie P-1 (Spearman ueber Stufenmittel,
      Bootstrap-KI, Permutations-Kontrolle).
  P-3 beschreibend (zweiter Kandidat, NICHT in der Holm-Familie): Born-
      Entropie des vollen Modells (Test) korreliert mit M50 -- dieselbe
      Statistik, aber nur beschreibend, weil die Karte die Generalisierungs-
      luecke als Hauptkandidaten fuehrt.
  P-4 Kontrolle: Generalisierungsluecke des KOPFS OHNE Schaltung
      (Identitaets-Arm) korreliert mit M50 schwaecher/nicht -- zeigt, dass
      die Schaltung (nicht nur der Kopf) die Quelle eines etwaigen P-2-
      Befunds ist. Gleiche Statistik, als Kontrolle deklariert.
  P-5 beschreibend: M50-Kurve, Generalisierungsluecke-Kurve (voll + Kopf),
      Born-Entropie-Kurve je Stufe, alle G(k)-Kurven.
  Holm-Korrektur NUR ueber die angemeldete Hauptfamilie {P-1, P-2} (auf den
  zweiseitigen Permutations-p-Werten), P-3/P-4 bleiben unkorrigiert
  (beschreibend/Kontrolle).

PROTOKOLL: MNIST 16x16 (8 Qubits, 256 Amplituden, 4 Lagen = 32 Winkel wie
C-mnist), 10 Rausch-Stufen, Mess-Seeds 250-255 (6, frisch) je Stufe (Nachschlag
25.09.: mehr Stufen statt mehr Seeds, s. o.), Tuning-Seeds 940/941 (frisch,
disjunkt von den Mess-Seeds). Regel 5: Rohdaten sofort sichern.
Regel 6 (Aufloesung): auch mit 10 Stufen bleibt ein Spearman-Test ueber
Stufenmittel SCHWACH (kleinster zweiseitiger exakter p bei perfekter
Rangfolge = 2/10! ~ 5,5*10^-7, aber die Bootstrap-KI aus nur 10
Stufenmitteln -- mit je 6 statt 10 Seeds dahinter -- bleibt strukturell
breiter als bei einem klassischen Experiment mit vielen unabhaengigen
Beobachtungen; ein HU-Ausgang bleibt der wahrscheinlichste, ehrliche Fall bei
einem mittleren Effekt (Erfahrung aus der ersten Fassung mit 6 Stufen, siehe
Bericht: nur ein sehr starker oder ein sehr sauberer Nulleffekt liefert ein
eindeutiges H1/H0). Mehr Stufen helfen der AUFLOESUNG (Zahl moeglicher
Rangordnungen), nicht direkt der KI-BREITE jedes Punkts -- das war genau der
Grund fuer den Nachschlag.
"""
from __future__ import annotations

import math
import time
import warnings
from functools import lru_cache

import numpy as np
import torch

from abteilungen.clifford import QireZelle, _baue, _gerundet, multiplikationen, runde_clifford
from labor.gemeinsam import bilder, genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, zusammenfassung

N = 8                                   # Qubits (MNIST 16x16, wie C-mnist)
K_STUFEN = (0, 2, 8)                    # reduzierter Rundungs-Sweep (Budget, s. Kopf)
RAUSCH_STUFEN = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)   # 10 Stufen (s. Kopf, Nachschlag 25.09.)
SEEDS = tuple(range(250, 256))          # 6 frische Mess-Seeds je Stufe (s. Kopf)
TUNING_SEEDS = (940, 941)               # frisch, disjunkt von SEEDS
SCHRITTE = 600
N_TRAIN = 2_500
N_TEST = 5_000
LR_GITTER = (0.02, 0.08, 0.32)
RAUSCH_SEED = 70_000
ALPHA = 0.05
GRENZE_TOST = 0.3                       # TOST-Grenze fuer Spearman-rho (geschaerft, s. Kopf)
N_BOOT = 3000
INSTABIL_GRENZE = 0.10                  # rein deskriptiver Kollaps-Hinweis (Budget-Vereinfachung, s. Kopf)

HYPOTHESEN = {
    "P-0": "Positivkontrolle: bei Rauschanteil 0 schlaegt die volle Schaltung die "
           "Identitaet (wie C-mnist). Sonst ist jede weitere Aussage leer.",
    "P-1": "Haupt (Abbruchkriterium 1): M50 zeigt ueber die 10 Rausch-Stufen eine "
           "monotone Tendenz -- Spearman ueber Stufenmittel, Bootstrap-90-%-KI "
           "(Kontrolle: zufaellige Zuordnung Stufe<->Ergebnis als Permutations-"
           "Nullverteilung). Ganze KI ausserhalb +-0,3, Richtung eindeutig -> H1; "
           "ganze KI innerhalb +-0,3 (TOST) -> H0; sonst HU.",
    "P-2": "Haupt (Abbruchkriterium 2): die Generalisierungsluecke des vollen "
           "Modells (ohne Rundungs-Sweep) korreliert mit M50 ueber dieselben "
           "Stufen -- gleiche Statistik wie P-1.",
    "P-3": "beschreibend, nicht in der Holm-Familie: Born-Entropie des vollen "
           "Modells korreliert mit M50 -- gleiche Statistik, zweiter Kandidat.",
    "P-4": "Kontrolle: Generalisierungsluecke des Kopfs OHNE Schaltung "
           "(Identitaets-Arm) korreliert mit M50 schwaecher/nicht -- zeigt, ob "
           "die Schaltung (nicht nur der Kopf) die Quelle eines P-2-Befunds ist.",
    "P-5": "beschreibend: M50-, Generalisierungsluecke- und Born-Entropie-Kurve "
           "je Stufe, alle G(k)-Kurven.",
}


# ------------------------------------------------------------------ Aufgabe: Etiketten-Rauschen
def _verrauschen(y: torch.Tensor, anteil: float, seed: int) -> torch.Tensor:
    """Feste Permutation EINES ANTEILS der Trainingsetiketten (wie magie_aufgabe.py
    'zufall', aber nur ueber eine Teilmenge der Groesse anteil*n): die Klassen-
    haeufigkeiten bleiben EXAKT erhalten, weil nur innerhalb der gewaehlten
    Teilmenge permutiert wird. anteil = 0 -> unveraendert; anteil = 1 -> wie
    magie_aufgabe.py 'zufall' (ganze Menge permutiert)."""
    if anteil <= 0:
        return y.clone()
    g = torch.Generator().manual_seed(RAUSCH_SEED + seed)
    n = len(y)
    m = int(round(anteil * n))
    if m < 2:
        return y.clone()
    idx = torch.randperm(n, generator=g)[:m]
    perm = idx[torch.randperm(m, generator=g)]
    y2 = y.clone()
    y2[idx] = y[perm]
    return y2


@lru_cache(maxsize=256)
def _daten(stufe: float, seed: int, n_train: int = N_TRAIN, n_test: int = N_TEST):
    """MNIST 16x16 (labor.gemeinsam.bilder) mit verrauschten TRAININGS-Etiketten;
    die Test-Etiketten bleiben sauber (Generalisierung wird gegen die echte
    Aufgabe gemessen, das Rauschen sitzt nur im Trainingssignal)."""
    xtr, ytr, xte, yte = bilder("mnist", seed, n_train=n_train, n_test=n_test, seite=16)
    ytr = _verrauschen(ytr, stufe, seed)
    return xtr, ytr, xte, yte


def _born_entropie(m: QireZelle, x: torch.Tensor) -> float:
    """Mittlere Shannon-Entropie (bit) der Born-Wahrscheinlichkeitsverteilung
    |psi(x)|^2 -- ein Kennwert, der am vollen Modell abfaellt, ohne jede
    zusaetzliche Rundung oder Training (billiger Kandidat, Karte I-035)."""
    with torch.no_grad():
        p = m.merkmale(x).clamp_min(1e-12)
        ent = -(p * p.log2()).sum(1)
    return float(ent.mean())


# ------------------------------------------------------------------ Lernrate je (Stufe, Arm)
def _tuning_punkt(a):
    """(schluessel, lr, seed, schritte, stufe, n_train, n_test, extra) -> Genauigkeit
    auf der Validierung (erstes Viertel des Trainingsteils). schluessel = 'voll' |
    'ohne_schaltung' | 'rundung' (rundet den mit extra=voll_lr trainierten voll-Arm
    auf k=0 und trainiert mit lr weiter -- EINE gemeinsame Rate fuer den ganzen
    Rundungs-Sweep, wie clifford.py; Abweichung von Regel 1a aus Budgetgruenden,
    offen benannt im Kopf)."""
    schluessel, lr, s, schritte, stufe, n_train, n_test, extra = a
    xtr, ytr, _, _ = _daten(stufe, s, n_train, n_test)
    n_val = len(ytr) // 4
    xt, yt, xv, yv = xtr[n_val:], ytr[n_val:], xtr[:n_val], ytr[:n_val]
    if schluessel == "rundung":
        m, _ = _baue("voll", s, N)
        trainiere(m, xt, yt, schritte, extra, seed=s)
        m = _gerundet(m, 0, s)
    else:
        m, _ = _baue(schluessel, s, N)
    trainiere(m, xt, yt, schritte, lr, seed=s)
    return genauigkeit(m(xv), yv)


def _gittersuche(schluessel, aufgabe_fuer, tuning_seeds, gitter=LR_GITTER):
    """Regel 1: eigene Rate je Schluessel, Gitter beidseitig verlaengert (hoechstens
    4 Runden), volles Budget (Muster magie_aufgabe.py/clifford.py)."""
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


def lernrate_je_stufe(stufe, tuning_seeds, schritte, n_train, n_test):
    """-> ({arm: lr}, info). Erst voll/ohne_schaltung, dann eine gemeinsame Rate
    fuer den Rundungs-Sweep (rundung), auf dem mit der voll-Rate trainierten
    voll-Modell der Tuning-Seeds -- wie clifford.py."""
    grund = ["voll", "ohne_schaltung"]
    info = _gittersuche(grund, lambda k, lr, s: (k, lr, s, schritte, stufe, n_train, n_test, None), tuning_seeds)
    voll_lr = info["voll"]["lr"]
    info.update(_gittersuche(["rundung"],
                             lambda k, lr, s: (k, lr, s, schritte, stufe, n_train, n_test, voll_lr), tuning_seeds))
    lr = {a: info[a]["lr"] for a in info}
    return lr, info


# ------------------------------------------------------------------ Messung
def _seed_lauf(a):
    stufe, s, schritte, lr, k_stufen, n_train, n_test = a
    xtr, ytr, xte, yte = _daten(stufe, s, n_train, n_test)
    z = {"seed": s, "stufe": stufe}
    m0, _ = _baue("ohne_schaltung", s, N)
    trainiere(m0, xtr, ytr, schritte, lr["ohne_schaltung"], seed=s)
    with torch.no_grad():
        z["ohne_schaltung"] = genauigkeit(m0(xte), yte)
        z["train_ohne_schaltung"] = genauigkeit(m0(xtr), ytr)
    voll, _ = _baue("voll", s, N)
    trainiere(voll, xtr, ytr, schritte, lr["voll"], seed=s)
    with torch.no_grad():
        z["voll"] = genauigkeit(voll(xte), yte)
        z["train_voll"] = genauigkeit(voll(xtr), ytr)
        z["entropie_born"] = _born_entropie(voll, xte)
    th = voll.theta.detach().clone()
    z["abstand_zum_gitter"] = float((th - runde_clifford(th)).abs().mean())
    for k in k_stufen:
        mk = _gerundet(voll, k, s)
        trainiere(mk, xtr, ytr, schritte, lr["rundung"], seed=s)
        with torch.no_grad():
            z[f"clifford_k{k}"] = genauigkeit(mk(xte), yte)
    return z


def lauf(schnell: bool = False, stufen=RAUSCH_STUFEN, seeds=SEEDS, tuning_seeds=TUNING_SEEDS,
         schritte=SCHRITTE, n_train=N_TRAIN, n_test=N_TEST, k_stufen=K_STUFEN, **kw) -> dict:
    if schnell:
        stufen, seeds, tuning_seeds = (0.0, 1.0), range(2), (940,)
        schritte, n_train, n_test, k_stufen = 60, 300, 500, (0, 2)
    stufen, seeds, tuning_seeds, k_stufen = list(stufen), list(seeds), list(tuning_seeds), list(k_stufen)
    t0 = time.time()
    lr_je_stufe, lr_info = {}, {}
    for stufe in stufen:
        lr, info = lernrate_je_stufe(stufe, tuning_seeds, schritte, n_train, n_test)
        lr_je_stufe[stufe] = lr
        lr_info[stufe] = info
        print(f"  Stufe {stufe:.2f}: Raten " + ", ".join(f"{a} {v:g}" for a, v in lr.items())
              + f" nach {time.time() - t0:.0f} s", flush=True)
    aufg = [(stufe, s, schritte, lr_je_stufe[stufe], k_stufen, n_train, n_test)
            for stufe in stufen for s in seeds]
    erg = parallel(_seed_lauf, aufg)
    zeilen = {stufe: [] for stufe in stufen}
    for (stufe, *_), z in zip(aufg, erg):
        zeilen[stufe].append(z)
    for stufe in stufen:
        for z in zeilen[stufe]:
            print(f"  Stufe {stufe:.2f} Seed {z['seed']}: voll {z['voll']:.4f}  ohne {z['ohne_schaltung']:.4f}  "
                  + "  ".join(f"k{k} {z[f'clifford_k{k}']:.4f}" for k in k_stufen), flush=True)
    ergebnis = {
        "abteilung": "clifford", "modul": "magie_schwierigkeit", "karte": "I-035", "version": 1,
        "datensatz": "mnist", "qubits": N, "hypothesen": HYPOTHESEN,
        "rausch_stufen": stufen, "k_stufen": k_stufen, "seeds": seeds, "tuning_seeds": tuning_seeds,
        "schritte": schritte, "n_train": n_train, "n_test": n_test,
        "lr": {f"{s:g}": v for s, v in lr_info.items()}, "zeilen": {f"{s:g}": z for s, z in zeilen.items()},
        "sekunden": time.time() - t0,
        "multiplikationen": {f"clifford_k{k}": multiplikationen(N, magie=k) for k in k_stufen}
                            | {"voll": multiplikationen(N)},
    }
    ergebnis["auswertung"] = auswertung(zeilen, k_stufen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung: M50, Korrelationen
def _schwelle_interp(kurve: dict, schwelle: float = 0.5):
    """kurve {k(int): G(float)} -> kleinste gemessene Stufe mit G >= schwelle,
    linear zwischen den benachbarten Stufen interpoliert (wie skala.py
    _m_schwelle, hier lokal, ohne Kopplung an skala.py). None, wenn die
    Schwelle innerhalb des gemessenen k-Bereichs nie erreicht wird (M50
    rechtszensiert, wird NICHT extrapoliert)."""
    ks = sorted(kurve)
    stufe = next((k for k in ks if kurve[k] is not None and kurve[k] >= schwelle), None)
    if stufe is None:
        return None
    i = ks.index(stufe)
    if i == 0:
        return float(stufe)
    k0, g0, g1 = ks[i - 1], kurve[ks[i - 1]], kurve[stufe]
    if g1 <= g0:
        return float(stufe)
    return float(k0 + (schwelle - g0) / (g1 - g0) * (stufe - k0))


def _rang_mittel(v: np.ndarray) -> np.ndarray:
    """Durchschnittsraenge (wie scipy.stats.rankdata), lokal statt scipy fuer
    Geschwindigkeit -- diese Funktion laeuft tausende Male je Korrelation."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rang_mittel(a), _rang_mittel(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _korrelation(x: dict, y: dict, n_boot: int = N_BOOT, rng=None) -> dict:
    """x, y: {stufe: np.array ueber Seeds, gleiche Laenge je Stufe}. Beobachtung:
    Spearman-Korrelation ueber die STUFENMITTEL (ein Punkt je Stufe). Bootstrap-
    90-%-KI: gemeinsames Resampling des Seed-INDEX innerhalb jeder Stufe (haelt
    die Paarung x<->y je Seed und Stufe fest -- ein 'Pseudopaar ueber die Seed-
    Position', da die Seeds selbst je Stufe frisch gezogen wurden und sich nicht
    ueber Stufen hinweg entsprechen); die Resampling-Ziehung ist unabhaengig
    zwischen den Stufen (pro Stufe vektorisiert: n_boot Ziehungen auf einmal).
    Kontrolle aus der Karte I-035 ('zufaellige Zuordnung Stufe<->Ergebnis'):
    Permutations-Nullverteilung, die die y-Stufenmittel zufaellig den
    x-Stufenmitteln zuordnet; p = Anteil |rho_perm| >= |rho_obs|.
    HINWEIS (Vortest 25.09.): mit nur 6 Stufen kann die KI bei sehr sauber
    getrennten, rauscharmen Stufenmitteln auf einen (fast) einzigen Punkt
    kollabieren -- das ist eine korrekte Bootstrap-Aussage ueber die RANG-
    ORDNUNG, sagt aber nichts ueber die Groesse des Effekts; deshalb prueft
    _urteil_korrelation die GANZE KI gegen die TOST-Grenze, nicht nur, ob sie
    die Null beruehrt (s. dort)."""
    rng = rng or np.random.default_rng(0)
    stufen = sorted(x)
    if sorted(y) != stufen:
        raise ValueError("x und y muessen dieselben Stufen haben")
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mx = np.array([float(np.nanmean(x[s])) for s in stufen])
        my = np.array([float(np.nanmean(y[s])) for s in stufen])
    gueltig = np.isfinite(mx) & np.isfinite(my)
    aus = {"stufenmittel_x": mx.tolist(), "stufenmittel_y": my.tolist(), "n_stufen": int(gueltig.sum())}
    if gueltig.sum() < 3 or np.unique(mx[gueltig]).size < 2 or np.unique(my[gueltig]).size < 2:
        return {**aus, "rho": float("nan"), "ki_unten": float("nan"), "ki_oben": float("nan"), "p_permutation": 1.0}
    rho_obs = _spearman(mx[gueltig], my[gueltig])
    bx_mat = np.tile(mx, (n_boot, 1))
    by_mat = np.tile(my, (n_boot, 1))
    for j, s in enumerate(stufen):
        xs, ys = np.atleast_1d(x[s]).astype(float), np.atleast_1d(y[s]).astype(float)
        n_s = len(xs)
        idx = rng.integers(0, n_s, (n_boot, n_s))
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bx_mat[:, j] = np.nanmean(xs[idx], axis=1)
            by_mat[:, j] = np.nanmean(ys[idx], axis=1)
    boot = []
    for i in range(n_boot):
        bx, by = bx_mat[i], by_mat[i]
        g = np.isfinite(bx) & np.isfinite(by)
        if g.sum() >= 3 and np.unique(bx[g]).size > 1 and np.unique(by[g]).size > 1:
            r = _spearman(bx[g], by[g])
            if np.isfinite(r):
                boot.append(r)
    boot = np.array(boot)
    ki_unten, ki_oben = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95))) if len(boot) >= 50 \
        else (float("nan"), float("nan"))
    idx_g = np.flatnonzero(gueltig)
    perm = []
    for _ in range(n_boot):
        p = rng.permutation(idx_g)
        r = _spearman(mx[idx_g], my[p])
        if np.isfinite(r):
            perm.append(r)
    perm = np.array(perm)
    p_perm = float(np.mean(np.abs(perm) >= abs(rho_obs) - 1e-9)) if len(perm) else 1.0
    return {**aus, "rho": rho_obs, "ki_unten": ki_unten, "ki_oben": ki_oben, "p_permutation": p_perm}


def _holm(p: dict, alpha=ALPHA) -> dict:
    sig = {k: False for k in p}
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        if v <= alpha / (len(p) - i):
            sig[k] = True
        else:
            break
    return sig


def auswertung(zeilen: dict, k_stufen=K_STUFEN, n_boot: int = N_BOOT) -> dict:
    """zeilen {stufe(float): [ {seed, stufe, voll, ohne_schaltung, train_voll,
    train_ohne_schaltung, entropie_born, abstand_zum_gitter, clifford_k<k>...} ]}."""
    stufen = sorted(zeilen)
    aus = {"je_stufe": {}, "stufen": stufen, "k_stufen": list(k_stufen)}
    m50, gap_voll, gap_kopf, entropie = {}, {}, {}, {}
    for stufe in stufen:
        zz = zeilen[stufe]
        a = lambda k: np.array([z[k] for z in zz], dtype=float)
        d0 = a("voll") - a("ohne_schaltung")
        gewinn = float(d0.mean())
        p0 = wilcoxon_exakt(d0)
        vorbedingung = bool(gewinn > 0 and p0 < ALPHA)
        G = {k: (float((a(f"clifford_k{k}") - a("ohne_schaltung")).mean()) / gewinn if gewinn > 0 else None)
             for k in k_stufen}
        m50_stufe = _schwelle_interp(G, 0.5) if gewinn > 0 else None
        m50_seeds = []
        for z in zz:
            if gewinn <= 0:
                m50_seeds.append(float("nan"))
                continue
            kurve_s = {k: (z[f"clifford_k{k}"] - z["ohne_schaltung"]) / gewinn for k in k_stufen}
            val = _schwelle_interp(kurve_s, 0.5)
            m50_seeds.append(float("nan") if val is None else val)
        m50[stufe] = np.array(m50_seeds)
        gap_voll[stufe] = a("train_voll") - a("voll")
        gap_kopf[stufe] = a("train_ohne_schaltung") - a("ohne_schaltung")
        entropie[stufe] = a("entropie_born")
        instabil = int((np.minimum(a("voll"), np.minimum.reduce([a(f"clifford_k{k}") for k in k_stufen]))
                        < a("ohne_schaltung").mean() - INSTABIL_GRENZE).sum())
        aus["je_stufe"][f"{stufe:g}"] = {
            "n": len(zz), "gewinn": gewinn, "p_gewinn": p0, "vorbedingung": vorbedingung,
            "G": {str(k): v for k, v in G.items()}, "M50_stufenmittel": m50_stufe,
            "M50_seeds_undefiniert": int(np.isnan(m50[stufe]).sum()),
            "generalisierungsluecke_voll": zusammenfassung(gap_voll[stufe]),
            "generalisierungsluecke_kopf": zusammenfassung(gap_kopf[stufe]),
            "entropie_born": zusammenfassung(entropie[stufe]),
            "abstand_zum_gitter": float(a("abstand_zum_gitter").mean()) if "abstand_zum_gitter" in zz[0] else None,
            "instabile_seeds_deskriptiv": instabil,
        }
    if 0.0 in zeilen:
        zz0 = zeilen[0.0]
        d0 = np.array([z["voll"] for z in zz0]) - np.array([z["ohne_schaltung"] for z in zz0])
        aus["P0_diff"], aus["P0_p"] = float(d0.mean()), wilcoxon_exakt(d0)
    else:
        aus["P0_diff"], aus["P0_p"] = None, None
    x_stufen = {s: np.full(len(zeilen[s]), s) for s in stufen}
    aus["P1"] = _korrelation(x_stufen, m50, n_boot=n_boot, rng=np.random.default_rng(2035))
    aus["P2"] = _korrelation(gap_voll, m50, n_boot=n_boot, rng=np.random.default_rng(2036))
    aus["P3"] = _korrelation(entropie, m50, n_boot=n_boot, rng=np.random.default_rng(2037))
    aus["P4"] = _korrelation(gap_kopf, m50, n_boot=n_boot, rng=np.random.default_rng(2038))
    return aus


def _urteil_korrelation(erg: dict, name: str) -> str:
    """Dreiwertige TOST-Entscheidung mit EINER Grenze GRENZE_TOST auf beiden
    Seiten (Muster wie clifford.py C-1/skala.py H0(gleich), NICHT zwei
    verschiedene Begriffe 'deckt 0' und 'TOST-Band'): das vermeidet einen
    Randfall, den der erste Entwurf hatte (Bootstrap-Vortest, 25.09.): bei
    sehr vielen, sauber getrennten Seeds kann die KI auf einen PUNKT knapp
    ueber 0 (z. B. rho = 0,03, KI [0,03; 0,03]) kollabieren -- 'deckt 0 nicht'
    waere dann technisch wahr, obwohl der Effekt praktisch bedeutungslos ist.
    Schaerfung (UEBERGABE Vorgabe 7, 'schaerfen erlaubt'): H1 verlangt, dass
    die GANZE KI ausserhalb von +-GRENZE_TOST liegt (klar UND relevant groesser
    als die Aequivalenzgrenze), H0 verlangt die ganze KI innerhalb, sonst HU."""
    rho, ku, ko = erg["rho"], erg["ki_unten"], erg["ki_oben"]
    if math.isnan(rho) or math.isnan(ku) or math.isnan(ko):
        return f"HU: {name} nicht auswertbar (zu wenige gueltige Stufen, n = {erg['n_stufen']})"
    txt = (f"rho = {rho:+.2f}, Bootstrap-90-%-KI [{ku:+.2f}; {ko:+.2f}], "
           f"Permutations-p = {erg['p_permutation']:.4f} (n = {erg['n_stufen']} Stufen)")
    if ku > GRENZE_TOST or ko < -GRENZE_TOST:
        richtung = "positiv (mehr Rauschen/hoeherer Wert -> mehr Magiebedarf)" if rho > 0 \
            else "negativ (mehr Rauschen/hoeherer Wert -> weniger Magiebedarf)"
        return f"H1: {name} zeigt eine {richtung} Tendenz -- {txt}"
    if ku > -GRENZE_TOST and ko < GRENZE_TOST:
        return f"H0: kein Zusammenhang (TOST +-{GRENZE_TOST}) -- {txt}"
    return f"HU: kein eindeutiges Urteil -- {txt}"


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Alle Werte STRINGS,
    keine verschachtelten dicts. Praefixe: 'H1:', 'H0:', 'HU:'."""
    u = {}
    if a["P0_diff"] is None:
        u["P-0"] = "HU (Vorbedingung): kein Rauschanteil 0 gemessen"
    else:
        txt = f"(Differenz {a['P0_diff']:+.4f}, p = {a['P0_p']:.4f})"
        u["P-0"] = (f"H1: Schaltung schlaegt Identitaet bei Rauschanteil 0 {txt}"
                    if a["P0_diff"] > 0 and a["P0_p"] < ALPHA else
                    f"H0: Schaltung nicht nachweisbar besser als Identitaet bei Rauschanteil 0 {txt}")

    p_fam = {"P-1": a["P1"]["p_permutation"], "P-2": a["P2"]["p_permutation"]}
    sig = _holm(p_fam, ALPHA)
    holm_txt = lambda k: f"Holm ueber {{P-1,P-2}}: {'signifikant' if sig[k] else 'nicht signifikant'} (p = {p_fam[k]:.4f})"
    u["P-1"] = _urteil_korrelation(a["P1"], "M50 ueber die Rausch-Stufen") + "; " + holm_txt("P-1")
    u["P-2"] = _urteil_korrelation(a["P2"], "Generalisierungsluecke (volles Modell) vs. M50") + "; " + holm_txt("P-2")
    u["P-3"] = "beschreibend, nicht in der Holm-Familie: " + _urteil_korrelation(a["P3"], "Born-Entropie vs. M50")

    r2, r4 = a["P2"]["rho"], a["P4"]["rho"]
    if math.isnan(r2) or math.isnan(r4):
        vergleich = ""
    elif abs(r4) < abs(r2):
        vergleich = " -- schwaecher als P-2 (Hinweis: die Schaltung traegt den Zusammenhang)"
    else:
        vergleich = " -- NICHT schwaecher als P-2 (Kontrolle unerwartet stark, Vorsicht bei der Deutung)"
    u["P-4"] = "Kontrolle: " + _urteil_korrelation(a["P4"], "Generalisierungsluecke (Kopf ohne Schaltung) vs. M50") \
               + vergleich

    teile = []
    for stufe in a["stufen"]:
        info = a["je_stufe"][f"{stufe:g}"]
        g_txt = ", ".join(f"k{k}={info['G'][str(k)]:.2f}" if info['G'][str(k)] is not None else f"k{k}=-"
                          for k in a["k_stufen"])
        teile.append(f"{stufe:.2f}: M50={info['M50_stufenmittel']}, "
                     f"Luecke_voll={info['generalisierungsluecke_voll']['mittel']:.3f}, "
                     f"Luecke_kopf={info['generalisierungsluecke_kopf']['mittel']:.3f}, "
                     f"Entropie={info['entropie_born']['mittel']:.2f}, G({g_txt})")
    u["P-5"] = "; ".join(teile)
    return {k: str(v) for k, v in u.items()}


# ------------------------------------------------------------------ Vortest
def _kurve_synth(k, m50):
    """Monoton steigende, bei k=m50 exakt 0,5 kreuzende Kurve (nur fuer den
    Vortest -- kein Anspruch auf physikalische Form)."""
    if m50 <= 0:
        return 1.0 if k > 0 else 0.0
    return k / (k + m50)


def _synth_zeilen(m50_fn, gap_fn, gap_kopf_fn=lambda s: 0.0, entropie_fn=lambda s: 2.0, sd=0.01,
                  sd_m50_seed=0.0, n_seeds=len(SEEDS), stufen=RAUSCH_STUFEN, k_stufen=K_STUFEN, gewinn=0.20,
                  rng=None):
    """Synthetische zeilen fuer auswertung()/urteil() -- ohne jedes Training."""
    rng = rng or np.random.default_rng(0)
    zeilen = {}
    for stufe in stufen:
        ziel = m50_fn(stufe)
        zz = []
        for i in range(n_seeds):
            ohne = 0.5 + rng.normal(0, sd)
            voll = ohne + gewinn + rng.normal(0, sd)
            z = {"seed": i, "stufe": stufe, "ohne_schaltung": ohne, "voll": voll}
            m50_seed = max(1e-6, ziel + rng.normal(0, sd_m50_seed))
            for k in k_stufen:
                z[f"clifford_k{k}"] = ohne + gewinn * _kurve_synth(k, m50_seed) + rng.normal(0, sd)
            z["train_voll"] = float(np.clip(voll + gap_fn(stufe) + rng.normal(0, sd), 0, 1))
            z["train_ohne_schaltung"] = float(np.clip(ohne + gap_kopf_fn(stufe) + rng.normal(0, sd), 0, 1))
            z["entropie_born"] = entropie_fn(stufe) + rng.normal(0, sd)
            z["abstand_zum_gitter"] = 0.3
            zz.append(z)
        zeilen[stufe] = zz
    return zeilen


def vortest(ziehungen: int = 5) -> bool:
    """Pflicht-Vortest: Bausteine (Verrauschung, Schwellen-Interpolation, Korrelation)
    + jeden Urteilszweig (H1/H0/HU) auf synthetischen Zeilen erzwingen."""
    ok = True

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    # -- Baustein: Verrauschung erhaelt Klassenhaeufigkeiten, verrauscht einen ANTEIL,
    #    ist reproduzierbar und seed-abhaengig.
    y = torch.randint(0, 10, (1000,))
    for anteil in (0.0, 0.1, 0.25, 0.5, 1.0):
        y2 = _verrauschen(y, anteil, seed=1)
        if not torch.equal(torch.bincount(y2, minlength=10), torch.bincount(y, minlength=10)):
            fehler(f"Verrauschung aendert Klassenhaeufigkeiten, anteil={anteil}")
        veraendert = float((y2 != y).float().mean())
        if anteil == 0.0 and veraendert != 0.0:
            fehler("Anteil 0 veraendert Etiketten")
        if anteil == 1.0 and veraendert < 0.85:
            fehler(f"Anteil 1,0 veraendert zu wenige Etiketten ({veraendert:.2f})")
        if anteil == 0.5 and not (0.30 < veraendert < 0.55):
            fehler(f"Anteil 0,5 veraendert einen unplausiblen Anteil ({veraendert:.2f})")
        if not torch.equal(y2, _verrauschen(y, anteil, seed=1)):
            fehler(f"Verrauschung nicht reproduzierbar, anteil={anteil}")
    if torch.equal(_verrauschen(y, 0.3, seed=1), _verrauschen(y, 0.3, seed=2)):
        fehler("verschiedene Seeds ergeben dieselbe Verrauschung")

    # -- Baustein: Schwellen-Interpolation
    if _schwelle_interp({0: 0.0, 2: 0.4, 8: 0.6}, 0.5) is None:
        fehler("_schwelle_interp findet die Schwelle nicht")
    elif abs(_schwelle_interp({0: 0.0, 2: 0.4, 8: 0.6}, 0.5) - 5.0) > 1e-9:
        fehler(f"_schwelle_interp interpoliert falsch: {_schwelle_interp({0: 0.0, 2: 0.4, 8: 0.6}, 0.5)}")
    if _schwelle_interp({0: 0.6, 2: 0.7}, 0.5) != 0.0:
        fehler("_schwelle_interp: Randfall k=0 falsch")
    if _schwelle_interp({0: 0.1, 2: 0.2}, 0.5) is not None:
        fehler("_schwelle_interp findet eine Schwelle, die nie erreicht wird")

    # -- Baustein: _korrelation erkennt perfekte positive/negative Beziehung und
    #    liefert bei echtem Rauschen kein extrem kleines Permutations-p. Mit
    #    n = 6 (wie die echten Mess-Seeds je Stufe, Nachschlag 25.09.), nicht 8.
    stufen = RAUSCH_STUFEN
    x_lin = {s: np.full(len(SEEDS), s) for s in stufen}
    y_lin = {s: np.full(len(SEEDS), s) for s in stufen}
    erg = _korrelation(x_lin, y_lin, n_boot=400, rng=np.random.default_rng(1))
    if not (erg["rho"] > 0.99 and erg["ki_unten"] > 0.9):
        fehler(f"_korrelation erkennt perfekte positive Beziehung nicht: {erg}")
    y_neg = {s: np.full(len(SEEDS), -s) for s in stufen}
    erg_neg = _korrelation(x_lin, y_neg, n_boot=400, rng=np.random.default_rng(1))
    if not (erg_neg["rho"] < -0.99 and erg_neg["ki_oben"] < -0.9):
        fehler(f"_korrelation erkennt perfekte negative Beziehung nicht: {erg_neg}")
    rng_z = np.random.default_rng(4)
    treffer = 0
    for _ in range(ziehungen):
        y_zufall = {s: rng_z.normal(0, 1, len(SEEDS)) for s in stufen}
        erg_z = _korrelation(x_lin, y_zufall, n_boot=400, rng=np.random.default_rng(int(rng_z.integers(1e6))))
        if erg_z["p_permutation"] < 0.05:
            treffer += 1
    if treffer > 2:      # < 0,05 sollte bei echtem Rauschen selten (~alpha) vorkommen
        fehler(f"Permutations-p bei Zufallsdaten zu oft signifikant ({treffer}/{ziehungen})")

    # -- Hauptpruefung: jeden Urteilszweig erzwingen (klar monoton -> H1, flach -> H0,
    #    verrauscht -> HU), fuer P-1 UND P-2 gemeinsam, MIT n_seeds = len(SEEDS) = 6
    #    (Nachschlag 25.09.: die echte Seedzahl je Stufe, nicht mehr 10 -- prueft
    #    genau die vom Direktor angefragte Stabilitaet der Bootstrap-Logik bei
    #    weniger Seeds je Stufe). Die drei Faelle sind so konstruiert, dass die
    #    STUFENMITTEL klar (nicht nur knapp) getrennt bzw. ueberlappend sind
    #    (Lehre beim Bau dieses Vortests, 25.09.: bei fast identischen
    #    Stufenmitteln kippt die Rangordnung durch reines Rundungsrauschen
    #    zufaellig hin und her -- das ergibt KEIN stabiles H0, sondern ein
    #    instabiles Zufalls-Ergebnis; ein echtes H0 braucht STUFENMITTEL, die
    #    sich zwischen den Stufen unterscheiden, aber in einer zur Rausch-
    #    Reihenfolge unkorrelierten -- nicht: identischen -- Anordnung).
    #    'flach' nutzt eine mit der Rausch-Reihenfolge nahezu unkorrelierte
    #    Permutation der 10 Stufen (rho_wahr ~ 0,006 zur Rausch-Reihenfolge UND
    #    zur Generalisierungsluecke -- durch Zufallssuche unter 200.000
    #    Permutationen von 1..10 gefunden) statt eines wortwoertlich konstanten
    #    M50. 'verrauscht' nutzt eine Permutation mit einem ECHTEN, aber nur
    #    mittelstarken Trend (rho_wahr = -0,45 zur Rausch-Reihenfolge) plus
    #    kraeftigem Seed- und Messrauschen -- ein Effekt, der bei 10 Stufen x 6
    #    Seeds ehrlicherweise nicht sicher zu entscheiden ist.
    # Ranggleiche Permutationen von 1..10 (Zufallssuche, 200.000 Ziehungen), auf
    # 1..7 SKALIERT (rangordnungserhaltend), damit kein Zielwert nahe/ueber
    # K_STUFEN-Maximum 8 liegt -- sonst wird M50 in _kurve_synth zu oft zensiert
    # (G(8) < 0,5) und die Zahl gueltiger Stufen faellt unkontrolliert (Lehre
    # beim Nachschlag 25.09.: erste Fassung mit rohen Raengen 1..10 verlor 2 von
    # 10 Stufen durch Zensur und aendert damit ungewollt die Ziel-Korrelation).
    _skala = lambda v: 1.0 + (v - 1) / 9 * 6                   # 1..10 -> 1..7
    _r = [8, 4, 1, 6, 7, 9, 5, 2, 10, 3]                       # rho(., Stufe) ~ 0,006
    _g = [6, 9, 2, 7, 8, 5, 1, 4, 3, 10]                       # rho(., _r)    ~ -0,006
    zero_perm = dict(zip(RAUSCH_STUFEN, (_skala(v) for v in _r)))
    gap_flach = dict(zip(RAUSCH_STUFEN, (0.01 * v for v in _g)))
    _j = [9, 5, 6, 10, 3, 7, 1, 4, 8, 2]                       # rho(., Stufe) = -0,45 (echter, aber unsicherer Trend)
    _gj = [5, 3, 1, 10, 4, 8, 6, 7, 2, 9]                      # rho(., _j)    ~ -0,006
    jum = dict(zip(RAUSCH_STUFEN, (_skala(v) for v in _j)))
    gap_jum = dict(zip(RAUSCH_STUFEN, (0.015 * v for v in _gj)))
    faelle = [
        ("monoton", dict(m50_fn=lambda s: 1 + 6 * s, gap_fn=lambda s: 0.02 + 0.30 * s, sd=0.003, sd_m50_seed=0.15,
                        n_seeds=len(SEEDS)), {"P-1": "H1:", "P-2": "H1:"}, 0),
        ("flach", dict(m50_fn=lambda s: zero_perm[s], gap_fn=lambda s: gap_flach[s], sd=0.0005, sd_m50_seed=0.5,
                      n_seeds=len(SEEDS)), {"P-1": "H0:", "P-2": "H0:"}, 1),
        ("verrauscht", dict(m50_fn=lambda s: jum[s], gap_fn=lambda s: gap_jum[s], sd=0.02, sd_m50_seed=1.5,
                            n_seeds=len(SEEDS)), {"P-1": "HU:", "P-2": "HU:"}, 2),
    ]
    for name, par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for z in range(ziehungen):
            zeilen = _synth_zeilen(rng=np.random.default_rng(100 + z), **par)
            u = urteil(auswertung(zeilen))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest ({name}): {h} -> {u[h][:140]}")
        for h, f in fehl.items():
            if f > toleranz:
                fehler(f"Fall '{name}', {h}: {f}/{ziehungen} falsch (erwartet {erwartet[h]})")

    # -- P-0: klarer Sieg -> H1, kein Sieg -> H0
    zeilen = _synth_zeilen(m50_fn=lambda s: 2.0, gap_fn=lambda s: 0.1, sd=0.003)
    if not urteil(auswertung(zeilen))["P-0"].startswith("H1:"):
        fehler("P-0 feuert nicht H1 bei klarem Sieg der Schaltung")
    zeilen0 = _synth_zeilen(m50_fn=lambda s: 2.0, gap_fn=lambda s: 0.1, sd=0.003, gewinn=0.0)
    if not urteil(auswertung(zeilen0))["P-0"].startswith("H0:"):
        fehler("P-0 feuert nicht H0 ohne Gewinn der Schaltung")

    # -- P-4 Kontrolle: Kopf-Luecke ohne Zusammenhang -> schwaecher als P-2, wenn
    #    P-2 selbst einen klaren (monotonen) Zusammenhang zeigt.
    zeilen = _synth_zeilen(m50_fn=lambda s: 1 + 6 * s, gap_fn=lambda s: 0.02 + 0.30 * s,
                          gap_kopf_fn=lambda s: 0.1, sd=0.003, sd_m50_seed=0.1)
    u = urteil(auswertung(zeilen))
    if "schwaecher als P-2" not in u["P-4"]:
        fehler(f"P-4-Kontrollvergleich feuert nicht wie erwartet: {u['P-4']}")

    # -- Holm-Familie: nur {P-1, P-2}, P-3 bleibt ausserhalb
    a = auswertung(_synth_zeilen(m50_fn=lambda s: 1 + 6 * s, gap_fn=lambda s: 0.02 + 0.30 * s, sd=0.003))
    u = urteil(a)
    if "Holm ueber {P-1,P-2}" not in u["P-1"] or "Holm ueber {P-1,P-2}" not in u["P-2"] \
            or "Holm" in u["P-3"].split("Born-Entropie")[0].replace("beschreibend, nicht in der Holm-Familie: ", ""):
        fehler("Holm-Familie nicht auf {P-1, P-2} beschraenkt")

    # -- Schnittstelle: Strings, keine verschachtelten dicts in urteil, JSON-serialisierbar
    a = auswertung(_synth_zeilen(m50_fn=lambda s: 2.0, gap_fn=lambda s: 0.1))
    u = urteil(a)
    if not all(isinstance(v, str) for v in u.values()):
        fehler("urteil() liefert nicht nur Strings")
    if set(u) != set(HYPOTHESEN):
        fehler(f"urteil()-Schluessel != HYPOTHESEN-Schluessel: {set(u)} vs {set(HYPOTHESEN)}")
    import json
    try:
        json.dumps({"auswertung": a, "urteil": u, "hypothesen": HYPOTHESEN})
    except (TypeError, ValueError) as e:
        fehler(f"nicht JSON-serialisierbar: {e}")
    if not all(isinstance(v, str) for v in HYPOTHESEN.values()) or set(HYPOTHESEN) != {"P-0", "P-1", "P-2", "P-3",
                                                                                        "P-4", "P-5"}:
        fehler("HYPOTHESEN-Schluessel falsch")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-035" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
