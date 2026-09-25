# Fehlerbuch — was schiefging und was wir daraus lernen

Ein gescheiterter Ausgang wird nie gelöscht. Jeder Eintrag endet mit der
Lehre und der Folgefrage. Der Direktor trägt H0/HU-Urteile automatisch ein;
die Lehre schreibt ein Mensch (oder Claude) dazu.

---

### 24.09.2026 — Z-zne (auf Papier)
- **Was:** Zero-Noise-Extrapolation über verschachtelte Bitbreiten.
- **Warum gescheitert:** Voraussetzung „Fehler glatt in der Rauschstärke“
  gilt für Runden nicht. Extrapolation macht den Fehler 7× größer.
- **Lehre:** Bei jeder Übertragung aus der Physik die *Gültigkeitsbedingung*
  der Methode mit übertragen und zuerst prüfen.
- **Folgefrage:** nur bei echtem Zufallsrauschen (analoge Hardware) — geparkt.

### 24.09.2026 — Rauchtest C-clifford
- **Was:** Gatterweise Simulation war 0,1 s/Schritt → ganzer Lauf ~8 h.
- **Lehre:** Training in der schnellsten *äquivalenten* Form (ganze Matrix),
  Kosten der Inferenz getrennt und analytisch zählen. Gleiche Trennung wie in
  Projekt A (Qualität bei trainierbarer Breite, Effizienz bei eigener Breite).

### 24.09.2026 — C-clifford (v1): der geerbte Kopf
- **Was:** Rundung kostete scheinbar 9,7 pp und war *schlechter* als eine
  zufällige Clifford-Schaltung (C-2: H0, p = 0,004).
- **Diagnose (explorativ, `clifford_diagnose.py`):** Die Rundungsarme erbten
  den Kopf des vollen Modells (auf andere Merkmale eingestellt, groß,
  übersättigt) und lernten mit der kleinen Rate 0,02 nach. Die Kontrollarme
  starteten mit frischem Kopf und eigener Rate. Mit frischem Kopf: k=0
  0,950 statt 0,870. **Der Befund war ein Artefakt des Aufbaus.**
- **Fehlerklasse (Regel „nach der Klasse suchen“):** *Arme mit
  unterschiedlichem Ausgangszustand* — jeder Arm, der von einem anderen Arm
  erbt, braucht denselben Start wie seine Kontrollen oder eine eigene
  Lernratensuche. Gesucht in `kompass.py`: dort erbt nichts, nicht betroffen.
- **Zweiter Fehler:** Auflösungsprüfung als absoluter SE-Schwellwert (1 pp)
  statt Konfidenzintervall relativ zur 2-pp-Grenze. Sie hätte bei einem
  Verlust von 10 ± 1,4 pp „kein Urteil“ gesagt, obwohl das eindeutig ist.
  v2: Nichtunterlegenheit mit einseitigem 95-%-KI.
- **Lehre:** Ein überraschender Befund ist zuerst ein Verdacht gegen den
  eigenen Aufbau. Die langweiligste Erklärung zuerst prüfen — hat 4 Minuten
  gekostet.
- **Folge:** `C-clifford-v2` auf frischen Seeds 10–19 (die Seeds 0–9 sind
  durch die Diagnose „verbraucht“), Design vor dem Lauf committet.

### 24.09.2026 — V-kompass Rauchtest: Bits ohne Bedeutung
- **Was:** Erster Entwurf komprimierte die *versteckte* 256×256-Schicht.
  Deren Neuronen-Reihenfolge ist willkürlich — eine Bitordnung *kann* dort
  nichts bedeuten. Außerdem wurde die ganze Matrix komprimiert, deren
  Zufallsstart unkomprimierbar ist (Fehler 0,94, Genauigkeit 19 %).
- **Lehre:** Eine Struktur-Methode nur dort testen, wo die Indizes Struktur
  *haben* (Eingang = Pixel: Zeilen- und Spaltenbits). Das Gelernte (ΔW)
  vom Zufälligen (W₀) trennen, wie LoRA es tut.
- **Folge:** erste Schicht, ΔW = W − W₀, zusätzliche menschliche Ordnung
  „Morton“ (Z-Kurve der Pixel) als harte Konkurrenz. Vor dem Lauf korrigiert.

### 24.09.2026 16:28 — C-clifford-v2 (automatisch)
- **C-1**: HU: KI ueberdeckt die 2-pp-Grenze -- kein Urteil
- **C-2**: H0: gelernte Rundung nicht besser als feste Clifford-Mischung
- **Lehre:** siehe manuellen Eintrag „C-clifford-v2: zwei Fehler im eigenen Design“ unten.

### 24.09.2026 — C-clifford-v2: zwei Fehler im eigenen Design
- **Entartete Kontrolle.** `hadamard_fest` lieferte in *jedem* Seed exakt
  dieselbe Genauigkeit wie die Identität (0,9516 ± 0,0062). Grund: Bei gerader
  Lagenzahl heben sich Ry(π/2)-Lagen mit CNOT-Ring zu einer
  Vorzeichen-Permutation auf (geprüft: 1 Nicht-Null-Eintrag je Zeile bei 2
  und 4 Lagen, 64 bei 1 und 3). **Lehre:** Jede Kontrolle wird vor dem Lauf
  darauf geprüft, dass sie *tatsächlich* das tut, was ihr Name sagt — hier:
  Matrix ansehen. v3: Hadamard nur in Lage 1.
- **Falsch skalierte Grenze.** C-1 fragte „Verlust ≤ 2 pp?“ — aber der
  *gesamte* Gewinn der Schaltung über die Identität beträgt auf den Ziffern
  nur 1,7 pp. Eine Grenze, die größer ist als der Effekt, kann nichts
  entscheiden. **Lehre:** Grenzen relativ zum Effekt setzen, den man erhalten
  will. v3: Gewinnanteil G(k) und M50/M90 zusätzlich vorregistriert
  (vor der ersten MNIST-Zahl).

### 24.09.2026 17:02 — V-kompass (automatisch)
- **V-1**: HU: besser, aber < 10 %
- **V-2**: HU: kein Unterschied nachweisbar
- **Lehre (Claude, 24.09.):** Die DMRG-Ordnung findet die Struktur, die schon
  da ist: Sie trennt Ausgangs- von Eingangsbits — das ist die natürliche
  Ordnung, deren Mittelschnitt genau der Matrixrang ist. Innerhalb der Blöcke
  sind die gelernten Filter der ersten Schicht *nicht* niedrig verschränkt,
  deshalb kostet jede weitere TT-Zerlegung nur Parameter: **gewöhnliche
  Niedrigrang-SVD schlägt alle Tensor-Train-Ordnungen deutlich** (Fehler
  0,78 gegen 0,92 bei 2.048 Parametern). Dasselbe Bild wie B-014: die
  Aufgabe setzt den Rang, nicht die Architektur.
- **Folgefrage (Idee, nicht vorregistriert):** nur die Pixelseite als
  Quanten-Tensor-Train (QTT, wie in der Bildkompression), die Rangseite als
  SVD — „Rang außen, Tensornetz innen“. Und: Kompression immer mit kurzem
  Nachtraining vergleichen, ohne Nachtraining fällt *jede* Methode bei
  ≤ 6 % der Parameter auf 45–73 %.

### 24.09.2026 18:13 — C-mnist (automatisch)
- **C-1**: H0: Rundung kostet nachweislich > 2 pp -- Magie wird gebraucht (siehe C-3)
- **C-2**: HU: schlaegt nur Hadamard
- **Lehre (Claude):** C-1: H0 ist hier der *erwartete* Befund — Magie wird gebraucht; Auswertung im LABORBUCH 19:15 (Magie-Konzentration M50 = 2).

### 24.09.2026 18:38 — I-008 (automatisch)
- **A-1**: HU: KI ueberdeckt die 1-pp-Grenze
- **A-2**: H0: born signifikant besser
- **Lehre (Claude):** A-2 war falsch gedacht. „Gemittelte Merkmale sind
  robuster“ stimmt nicht, wenn die Schaltung für die Born-Auslese *trainiert*
  wurde — die Paritätsköpfe saßen nachträglich auf einer fremd optimierten
  Schaltung (und mit ungesuchter Rate). Außerdem hat jede Parität dieselbe
  Schussvarianz ~1/S wie eine Wahrscheinlichkeit; Mitteln hilft nur, wenn die
  Information in wenigen Paritäten *steckt*. Fairer Nachtest: Schaltung
  gemeinsam mit dem Paritätskopf trainieren.

### 24.09.2026 18:53 — C-bewusst (automatisch)
- **B-1**: H0: kein nachweisbarer Vorteil des bewussten Trainings
- **B-2**: HU: KI ueberdeckt die Grenze
- **Lehre:** Konsistent mit C-2: Clifford-Schaltungen allein bringen nichts. Wer die Winkel waehrend des Trainings zum Clifford-Gitter zieht (Abstand 0,23 statt 0,40 rad), zieht sie zum nutzlosen Bereich -- das volle Modell verliert schon vor dem Runden 1,4 pp (0,947 gegen 0,961), und die Rundung wird nicht besser (B-1: -0,4 pp, p=1,0). Magie laesst sich nicht wegtrainieren, sie IST der Nutzen. Folgerung: nicht ALLE Winkel zum Gitter ziehen, sondern nur die unwichtigen -- das ist genau I-001 (Magie-Pruning) bzw. I-004 (Gerüst + k Winkel).

### 24.09.2026 19:44 — I-001 (automatisch)
- **P-1**: HU: KI ueberdeckt 0.1 (Differenz +0.073, p = 0.3750, KI oben +0.217)
- **Lehre:** P-1 HU: Verlust-Rangfolge hebt G(2) um +0,07 gegenueber Abstand (p=0,375), bei ~1,7 pp Gesamtgewinn auf den Ziffern ist 0,1 G nur ~1 Testbild -- Aufloesung zu klein (vom Forscher vorab gewarnt). P-2 H1: Wirkung schlaegt Zufall (+0,29 G, p=0,027). Folge: auf MNIST (Gewinn 3,8 pp) wiederholen, wenn Rechenzeit da ist.

### 24.09.2026 19:35 — I-002 (Forscher-Agent, nicht über den Direktor gestartet)
- **S-1**: H0: Operator-SRE sagt den Rundungsverlust nicht vorher (ρ = 0,26, p = 0,26)
- **Lehre:** Vorhandene Magie ist nicht genutzte Magie (Details LABORBUCH 19:45).
  Zweite Lehre, organisatorisch: Läufe von Agenten auch über den Direktor
  starten, sonst fehlt der automatische Eintrag.

### 24.09.2026 21:42 — C-fashion (automatisch)
- **C-1**: HU: KI ueberdeckt die 2-pp-Grenze -- kein Urteil
- **C-2**: HU: schlaegt nur Hadamard
- **Lehre:** C-1 HU nur knapp (KI [-2,05; -1,02] pp streift die 2-pp-Grenze), C-2 wie immer: k0 = zufaellige Clifford (+0,17 pp, p=0,77). Die Replikation traegt: 4 von 32 Winkeln holen 100 % des Magie-Nutzens (MNIST 62 %), M90 = 4 statt 12. DESIGNFEHLER entdeckt, weil k8 den vollen Arm SCHLUG (+0,53 pp, p=0,01): die Rundungsarme (_gerundet) bekommen eine zweite volle Trainingsphase (Start beim trainierten voll, frischer Kopf, Rate 0,32), der Arm voll nicht. G(k) gegen voll kann dadurch aufgeblaeht sein -- betrifft C-clifford/-v2/-mnist/-fashion, I-001, I-020, I-015, I-004. Lehre: Ein Arm, der aus einem anderen hervorgeht, braucht als Referenz denselben Nachlauf ohne den Eingriff (hier: alle Winkel frei, gleiche zweite Phase). Folge: C-referenz (faire Referenz, frische Seeds, MNIST+Fashion) vorregistriert; C-skala und I-024 vor jeder Zahl um den Arm nachtrainiert ergaenzt. Ein Befund, der ueber 100 % geht, ist ein Pruefauftrag, kein Erfolg.
- **Nachtrag 23:50 (nach C-referenz):** Der vermutete Designfehler hat sich auf MNIST nicht bestaetigt (d_nach = -0,03 pp, TOST +-0,3); auf Fashion verdeckt eine instabile Lernrate die Frage (siehe C-referenz). Die Pruefung war richtig, der Verdacht falsch -- G(k) aus C-mnist/C-fashion bleibt gueltig.

### 24.09.2026 22:04 — I-003 (automatisch)
- **T-1**: HU (Vorbedingung): Schaltung schlaegt die Identitaet auf der Strukturaufgabe nicht nachweisbar (Gewinn -0.0573, p = 0.0098)
- **T-2**: HU (Positivkontrolle): T-1 nicht bestanden -- Vergleich echt/zufall nicht auswertbar
- **Lehre:** Positivkontrolle gescheitert, aber lehrreich: Die Strukturaufgabe ist per Konstruktion mit einer Clifford-Schaltung (Magie 0) loesbar (Kopf auf Lehrer-Merkmalen 0,934), doch die per Gradient trainierte volle Zelle landet UNTER der Identitaet (0,726 gegen 0,784, sd 5,5 pp, lr 0,32). Gradientenabstieg aus Zufallswinkeln findet die diskrete Clifford-Loesung nicht -- 'exakt loesbar' ist nicht 'lernbar'. Zweitens bestaetigt sich der C-fashion-Designfehler drastisch: der nachtrainierte k12-Arm erreicht 0,786, also +6 pp ueber voll, nur durch die zweite Trainingsphase. Lehre: Eine Positivkontrolle muss mit dem VERFAHREN feuern koennen, nicht nur mit der Architektur -- vorher mit genau dem Lernverfahren pruefen, ob es die bekannte Loesung findet. Folgen: (1) C-referenz klaert die Referenz; (2) I-003-v2 mit Positivkontrolle, die das Verfahren erreicht (diskrete Clifford-Suche wie I-005 oder Lehrer = gerundete Schuelerschaltung); (3) I-005 bekommt Gewicht: diskrete Suche statt Gradient. Explorativ (nicht vorregistriert als Hauptfrage): G-Kurve echt liegt ueber zufall (+0,51 G, p = 0,002); Memorieren braucht mehr Magie (M50 = 8) als Generalisieren (M50 = 1).

### 24.09.2026 23:01 — I-020 (automatisch)
- **G-1**: H0: keine Kombination mit <= 2194 T haelt die Genauigkeit innerhalb 2 pp von 0.9397 -- Abbruchkriterium der Karte erfuellt (beschreibend, 3 Seed(s), kein Signifikanztest)
- **Lehre:** Die 4-fach-Ersparnis gibt es nicht, die 2-fach-Ersparnis gratis: duenn besetzte exakte Praeparation + volle Schaltung 4.372 T bei 0,940 = Referenz (8.776 T, dicht). duenn + k=4: 3.506 T (40 %), -0,7 pp. Billigere Kodierungen (MPS chi=2: 2.178 T, 0,881; pi/4-gerundet: < 1.000 T, 0,73) kosten 6-20 pp -- Kodierungs-Verlust ist nicht durch Schaltungs-Magie aufzuholen. Mit k=4 macht das Laden 96 % der T aus: die Magie-Konzentration hat ihren Teil getan, der Rest ist Datenladen. Lehre: Das Abbruchkriterium (25 %) war an der Hoffnung ausgerichtet, nicht an einer Schranke -- vorher die Untergrenze des Ladens (duenn exakt ~3.400 T = 39 %) ausrechnen, dann waere klar gewesen, dass 25 % nur mit verlustbehafteter Kodierung geht. Organisatorisch: Modul lieferte Urteile als dict, direktor.py stuerzte im Fehlerbuch ab (Daten waren gesichert) -- direktor._text macht das jetzt robust. Folge: Ladekosten selbst senken (amplitudenarme Kodierung, die ZUR Aufgabe gehoert, oder Winkelkodierung ~ Clifford+wenige T) -- neue Karte in der Ideenschmiede.

### 24.09.2026 23:42 — C-referenz (automatisch)
- **R-1**: HU: Verzerrung weder belegt noch ausgeschlossen -- mnist: d = -0.03 pp, KI [-0.24, +0.18] in +-0,3; fashion_mnist: d = -2.72 pp, KI [-5.08, -0.36], p = 0.1055
- **R-2**: HU (Vorbedingung): nachtrainiert schlaegt k0 nicht nachweisbar -- mnist: Gm*(4) = 0.59 KI [0.3, 0.77], Magiegewinn +2.72 pp (p = 0.0020); fashion_mnist: kein Magiegewinn
- **Lehre:** Der Verdacht aus C-fashion (zweite Trainingsphase blaeht G(k) auf) ist auf MNIST WIDERLEGT: nachtrainiert - voll = -0,03 pp, 90-%-KI [-0,24; +0,18] ganz in +-0,3 pp. Auf Fashion ist die faire Referenz selbst kaputt: Rate 0,32 (an k0 = nur Kopf getunt) laesst das Training mit allen 32 freien Winkeln auf 4 von 10 Seeds kollabieren (0,725-0,812 statt ~0,83); auf den 6 stabilen Seeds d_nach = +0,03 pp. Der wahre Stoerfaktor ist also die LERNRATE, nicht das Nachtraining: eine Rate, die fuer wenige freie Parameter getunt wurde, gilt nicht fuer viele (Regel 1a gilt je Arm, auch fuer Referenzarme). Dasselbe erklaert I-003 struktur (voll bei 0,32 instabil, sd 5,5 pp). R-3 'H1 Runden regularisiert' ist ein Artefakt dieser instabilen Referenz und wird NICHT verwendet; k8 - voll auf Fashion repliziert nicht (+0,24 pp, p = 0,08). Lehre 2: Eine Referenz, die ich zur Kontrolle einer anderen einfuehre, braucht selbst eine Pruefung auf Instabilitaet (Einzelseed-Ausreisser), bevor sie Nenner wird. Folgen: C-skala bekommt eine eigene Rate fuer nachtrainiert; I-024 tunt sie bereits je Arm.

### 24.09.2026 23:57 — I-004 (automatisch)
- **R-1**: HU: KI ueberdeckt G = 0,5 (G = 0.71, KI 0.47 .. 0.96, Holm-p = 0.1934)
- **R-2**: HU: unentschieden (+0.279 G, Holm-p = 0.1484, KI oben +0.532)
- **R-3**: HU: Aequivalenz weder gezeigt noch widerlegt (-0.063 G; 90-%-KI -0.230 .. +0.104)
- **Lehre:** Alle vier Fragen HU -- dieselbe Lehre wie I-001: auf den Ziffern (Gewinn 2,1 pp = ~11 Testbilder) ist 0,15-0,3 G nicht aufloesbar. Beschreibend aber bemerkenswert: ein ZUFAELLIGES Clifford-Geruest mit 4 von Anfang an trainierten Magiewinkeln erreicht G = 0,71 (KI 0,47-0,96) mit EINEM Training statt zwei -- man braucht das volle Modell gar nicht, um wenig-magische Schaltungen zu bekommen. Positionen aus dem vollen Modell (geruest_fest) sind bei k=2 besser als zufaellige (+0,22 G, p = 0,10). Lehre: Fragen, deren Effekt in G-Einheiten < 0,3 liegt, nicht auf den Ziffern stellen -- Aufloesung VOR dem Bau gegen den Gesamtgewinn rechnen (Regel 6). Folge: I-004 auf MNIST (Gewinn 3,8 pp) als Idee vormerken, zusammen mit I-001 auf MNIST.

### 25.09.2026 01:44 — I-024 (automatisch)
- **D-1**: H0 (Gegenrichtung): mit weniger Daten wird MEHR Magie gebraucht (Delta(500) - Delta(10000) = +1.26 pp, p = 0.0186)
- **D-2**: H0 (Gegenrichtung): mit wenig Daten tragen 2 Magiewinkel WENIGER (-0.275, p = 0.0420)
- **Lehre:** Meine Hypothese (Clifford-Gitter regularisiert, wenig Daten -> wenig Magie) ist in die GEGENRICHTUNG widerlegt: Delta(500) = 4,18 pp > Delta(10.000) = 2,91 pp (p = 0,019), G(2) 0,34 -> 0,43 -> 0,61 bei 500/2.000/10.000 (Gegenrichtung einseitig ohne Holm, also nur ein deutlicher Hinweis). Die Magie-Konzentration SCHAERFT sich mit mehr Daten; bei 500 Bildern faellt die gerundete Clifford-Schaltung sogar unter die Identitaet (G(0) = -0,10). Denkfehler im Kartenentwurf: nachtraegliches Runden eines trainierten Modells ist keine Regularisierung -- regularisieren wuerde nur ein Training IM eingeschraenkten Raum (Clifford-Geruest + k Winkel, I-004). 'Quantisierung regularisiert' gilt fuer quantisierungsbewusstes Training, nicht fuer Rundung danach; eine Analogie aus der Literatur vor der Uebernahme auf den eigenen Eingriff pruefen. Dritter Beleg gegen den Nachtrain-Verdacht: d_nach = +0,04 / +0,20 / +0,07 pp. Folge: in I-004-mnist das Geruest-Training auch bei n = 500 pruefen (dort koennte die Regularisierung sichtbar werden).

### 25.09.2026 04:45 — C-skala (automatisch)
- **S-0 echt8**: H0: Gewinn ueber Identitaet nicht nachweisbar (-0.0131, p = 0.4316); beschreibend voll-Identitaet +0.0394 (p = 0.0020), d_nach = -0.0525 (p = 0.4922)
- **S-0 fest10**: H0: Gewinn ueber Identitaet nicht nachweisbar (-0.0882, p = 0.8457); beschreibend voll-Identitaet +0.0535 (p = 0.0020), d_nach = -0.1417 (p = 0.2324)
- **S-3**: HU (Vorbedingung): Schaltung schlaegt die Identitaet nicht nachweisbar bei ['echt6', 'fest10'] -- G undefiniert
- **Lehre:** Die eigene Rate je Arm (Lehre C-referenz) reichte NICHT: Die Suche auf 2 Tuning-Seeds waehlte fuer nachtrainiert ueberall 0,32 -- der naechste Gitterpunkt 1,28 kollabiert (0,53-0,57). Die beste Rate liegt also direkt an der Stabilitaetskante; auf den Mess-Seeds kippte sie bei echt8 in 2/10, bei fest10 in 5/10 Seeds. S-3 (die saubere Trennung Breite/Aufloesung) faellt dadurch als HU (Vorbedingung) aus. Lehre: Lernrate nicht als argmax auf wenigen Seeds waehlen, sondern mit Stabilitaetsabstand -- z. B. die beste Rate, deren oberer Nachbar NICHT kollabiert, oder Kriterium = schlechtester Tuning-Seed statt Mittel. Gilt fuer alle Arme mit vielen freien Winkeln. S-1 H1 (echt6 -> echt10, beide stabil: +0,14 G, p = 0,0098, Holm) ist formal gueltig, aber wegen der Aufloesungs-Konfundierung nur ein Hinweis, bis S-3 steht. Folge: C-skala-v2 (nur Rundungs-/Referenzarme neu, Rate mit Stabilitaetsabstand, fest10 + echt6/8/10).

### 25.09.2026 05:20 — I-034 (automatisch)
- **G-2**: H0: kein relevanter Verlust (TOST +-0,5 pp) -- d(kombi_m48 - hybrid_dicht) = -0.03 pp (Wilcoxon p = 0.7344, p_TOST 0.0049); Verlust feste Stufe k2 durch Top-48 -0.93 pp, Wechselwirkung +0.90 pp (p = 0.0020)
- *Was lernen wir daraus? — vom Menschen/Claude nachzutragen.*

### 25.09.2026 07:11 — C-skala-v2 (automatisch)
- **S-1**: HU: Breite+Aufloesung (echt6 -> echt10): kein Urteil; d = -0.079 G (KI -0.161 .. +0.003, p_rechts = 0.9346, p_links = 0.0801, Holm nein); Instabilitaet (nachtrainiert < voll - 1 pp): echt6 0/10, echt10 0/10; keine instabilen/kollabierten Seeds
- **S-3**: HU: nur Breite (echt6 -> fest10): kein Urteil; d = +0.125 G (KI -0.039 .. +0.289, p_rechts = 0.0654, p_links = 0.9473, Holm nein); Instabilitaet (nachtrainiert < voll - 1 pp): echt6 0/10, fest10 1/10; beschreibend ohne Seeds [229]: d = +0.083 G (KI -0.080 .. +0.245, p_rechts = 0.1250, n = 9)
- **Lehre:** Mit stabiler Referenz (Ratenwahl min ueber 3 Tuning-Seeds, Stabilitaetsabstand, Gitter Faktor 2) verschwindet der Hinweis aus C-skala v1 (S-1 H1, +0,14 G): S-1 ist jetzt HU und dreht sogar leicht in die Gegenrichtung (d = -0,08 G, n.s., p_rechts = 0,93). S-3 (saubere Trennung Breite/Aufloesung mit fest10) ebenfalls HU, diesmal aber MIT sauberem Vergleich (0/10 bzw. 1/10 instabile Seeds, kein Kollaps) -- die vorherige Nichtaufloesung war also KEIN Aufloesungsproblem, sondern echte Unsicherheit ueber die Richtung. Lehre: Ein instabiler Effekt bei grober Ratenwahl war ein Artefakt der Instabilitaet, nicht ein echtes, nur schlecht gemessenes Signal -- Vorsicht vor der Versuchung, ein HU nach einer Methodenkorrektur automatisch als 'jetzt richtig gemessenes H1' zu lesen; hier war es umgekehrt (das H1 loeste sich auf). Die Magie-Skalierungsfrage (waechst M50 langsamer als n*L?) bleibt damit vorerst OFFEN, nicht beantwortet -- weder in die eine noch die andere Richtung.

### 25.09.2026 07:54 — I-034-fashion (automatisch)
- **G-1**: H0: jede Konfiguration > 10 % oder nachweislich > 0,5 pp schlechter -- Abbruchkriterium der Karte erfuellt -- kombi_m48: 958 T = 12.44 % von 7698 T, acc 81.54 % (ref 81.88 %), d -0.34 pp, KI95 einseitig [-0.65; -0.03], p_NI 0.1817 (Holm 0.3633), Quantenanteil 31.2 %, Ausweich 2; kombi_m32: 792 T = 10.28 % von 7698 T, acc 81.23 % (ref 81.88 %), d -0.64 pp, KI95 einseitig [-1.11; -0.17], p_NI 0.7045 (Holm 0.7045), Quantenanteil 36.4 %, Ausweich 3
- **Lehre:** Kombination repliziert nicht: Top-48-Laden verliert auf Fashion mehr (I-033 war nur MNIST getestet), und mehr Bilder erreichen ueberhaupt die Quantenstufe (Fashion braucht laut C-fashion mehr Magie als MNIST). Lehre: ein kombinierter Befund, der aus mehreren datensatzspezifisch kalibrierten Bausteinen besteht, generalisiert nicht automatisch -- jeder Baustein (hier: Top-m-Groesse) braucht eine eigene Kalibrierung je Aufgabe. Positiv: zwei_0_8_m48 (direkter Sprung 0->8 ohne k2-Zwischenstufe) liegt mit 6,5 % nahe am Ziel -- naechster Versuch: Top-m und Kaskadenstufen je Datensatz neu waehlen statt MNIST-Werte zu uebernehmen.

### 25.09.2026 10:23 — I-035 (automatisch)
- **P-0**: H0: Schaltung nicht nachweisbar besser als Identitaet bei Rauschanteil 0 (Differenz -0.0220, p = 0.4375)
- **P-1**: HU: kein eindeutiges Urteil -- rho = -0.31, Bootstrap-90-%-KI [-0.69; -0.10], Permutations-p = 0.4737 (n = 8 Stufen); Holm ueber {P-1,P-2}: nicht signifikant (p = 0.4737)
- **P-2**: HU: kein eindeutiges Urteil -- rho = -0.36, Bootstrap-90-%-KI [-0.57; +0.02], Permutations-p = 0.3930 (n = 8 Stufen); Holm ueber {P-1,P-2}: nicht signifikant (p = 0.3930)
- **Lehre:** Der Rohlauf hatte einen kollabierten Seed (253, Rate 0,32 an der Stabilitaetskante -- exakt das dritte Mal in dieser Nacht nach C-referenz und I-003-struktur). Das verzerrte P-0 zu einem falschen H0 UND liess die population-basierte M50-Rechnung die Stufe 0 % Rauschen (den wichtigsten Ankerpunkt) als undefiniert (gewinn<=0) fallen -- P-1/P-2 liefen nur mit 8 von 10 Stufen. Robuste Nachrechnung (Modul-eigenes Instabilitaets-Kriterium, dieselbe auswertung()-Funktion, nur 2 kollabierte Seeds ausgeschlossen: 253 bei 0 %, 252 bei 85 %) mit allen 10 Stufen: P-0 jetzt richtungsrichtig (+3,9 pp, p=0,0625 -- n=5 kann 0,05 nicht erreichen, aber konsistent mit C-mnist/C-referenz), P-1 rho=-0,44 (KI [-0,70;-0,30], durchgehend negativ, aber Permutations-p=0,196 -- HU bleibt), P-2 sauberes H0 (rho=-0,14, TOST): Generalisierungsluecke sagt M50 NICHT voraus. Lehre (generalisiert, dritte Wiederholung): Die 'einfache Ratensuche ohne Stabilitaetsabstand aus Zeitgruenden' ist ein wiederkehrendes Muster mit vorhersagbarem Fehler -- ab sofort soll JEDES Modul mit dieser Vereinfachung eine automatische, im urteil() eingebaute Ausreisser-Erkennung (Modul-eigenes Instabilitaets-Kriterium) VOR der Urteilsbildung anwenden, nicht nur deskriptiv mitzaehlen. Inhaltlich bemerkenswert (nicht signifikant, aber gegenlaeufig zur I-003-Vermutung): der Punktschaetzer zeigt eher WENIGER Magiebedarf bei MEHR Label-Rauschen, nicht mehr -- passt nicht zur exploративen I-003-Beobachtung (Memorieren M50=8). Beide Datenpunkte (I-003 zwei Extreme, I-035 zehn Stufen) sind einzeln nicht belastbar; der Widerspruch selbst ist die eigentliche Erkenntnis: die Beziehung ist nicht so einfach monoton wie I-003 nahelegte.
