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
