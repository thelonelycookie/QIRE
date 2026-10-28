# Kreativ-Methode des Instituts (Entwurf 27.10.2026)

Ziel: Ideen in großer Zahl erzeugen und dabei die Qualität halten. Menschliche Kreativtechniken werden mit der Breite paralleler Agenten verbunden. Malte liefert Gespür, Geschmack und Veto, die Agenten liefern Breite, Tempo und Prüfung.

## 1. Was die Welle 27.10. gelehrt hat
- Die wertvollsten Funde kamen nicht aus der Literatur. Sie kamen aus **eigenen Ungereimtheiten**: w9 und w12 fanden 10 Epochen statt „knappes Training“ und eine FLOP-Zählung um den Faktor 9–12.
- Der zweitbeste Weg war die **Analogie aus der Physik** (w10: RG → Hourglass, Lindblad → LRU). Daraus kam der stärkste Rauchtest, R2.
- **Billige Kontrollen** haben scheinbare Gewinne schnell entlarvt (R3: Phase = Quadrat, R1: Zufalls-Orthogonale = Hadamard).
- Die Vorab-Tipps der Agenten waren oft richtig. Das ist messbar, und daraus lässt sich lernen.

## 2. Der Kreislauf (je Welle ca. 1 Tag)
1. **Divergenz, breit (8–12 Agenten), mit verschiedenen Denkwerkzeugen statt verschiedener Themen:**
   - *Morphologischer Kasten:* Quantenprinzip (Superposition, Interferenz, Messung, Verschränkung, Unitarität, RG, Fehlerkorrektur, Dekohärenz) × KI-Baustein (Einbettung, Attention, FFN, Kopf, Optimierer, Daten, Quantisierung) × Ressource (FLOPs, Bytes, Daten, Energie, Latenz). Ein Agent füllt die ganze Tabelle, je Zelle ein Satz.
   - *Fehlerbuch-Bergbau:* jede „beschreibend“- oder „Nebenbefund“-Zeile im LABORBUCH, jede gescheiterte Hypothese → „Was müsste wahr sein, damit das kein Zufall war?“
   - *Umkehrung:* Annahmen umdrehen (statt FLOPs sparen: mehr FLOPs, dafür viel weniger Bytes; statt Struktur lernen: Struktur zufällig wählen).
   - *Perspektiven:* Experimentalphysikerin, Chip-Ingenieur, Jugend-forscht-Juror, Skeptiker, Zehnjährige. Jede Rolle liefert 3 Ideen.
   - *Zufallsimpuls:* ein zufällig gezogenes Paper aus einem fremden Gebiet (Biologie, Akustik, Kompression). Wie würde man es auf QIRE übertragen?
2. **Konvergenz, ein Agent:** Doppeltes zusammenführen, die langweiligste Erklärung je Idee benennen, je Idee den billigsten Test (≤ 10 min) und einen Vorab-Tipp mit Wahrscheinlichkeit festhalten.
3. **Turnier:** Je zwei Ideen werden paarweise von 3 unabhängigen Richter-Agenten verglichen. Kriterien: Aussicht bei gleichen FLOPs **und** Bytes, Kosten, Erklärbarkeit für die Jury, Quantenbezug. Daraus Elo-Wertung, die Top 5 kommen weiter.
4. **Malte-Schleife (5 min):** Die Top 10 als Einzeiler. Malte markiert Bauchgefühl ++/+/−, auch per Diktat. Seine ++ kommen immer in die Rauchtests, auch wenn sie im Turnier schwach waren. Menschliches Gespür bekommt so ein festes Kontingent.
5. **Rauchtests (Tuning-Seeds, parallel):** je Idee ≤ 30 min, immer mit der fiesesten Kontrolle (Zufallsvariante, gleiche FLOPs, gleiche Bytes).
6. **Vorregistrierung** nur für Grünes, nach Maltes Freigabe. Rechnen auf Raven.
7. **Inkubation:** Nach jedem Lauf liest ein Agent Ergebnis und alte Ideenliste neu. Welche verworfene Idee lohnt mit dem neuen Wissen wieder?

## 3. Wo die KI den Menschen übertreffen kann und wo nicht
- **Besser (quantitativ):** Breite (100 Ideen in 1 h), lückenloses Durchsuchen der eigenen Akten, konsequente Kontrollen, kein Liebling-Effekt bei fertigen Zahlen.
- **Besser (qualitativ), wenn erzwungen:** systematische Gegenhypothesen, Rechenprüfung jeder Zahl (z. B. w12), Kalibrierung über die Trefferquote der Vorab-Tipps.
- **Schwächer:** echte Überraschung, Geschmack, Sinn für „das ist eine schöne Geschichte für die Jury“. Der Agent neigt zur Mitte des Bekannten. Gegenmittel sind Zufallsimpuls, Umkehrung und Maltes festes Kontingent.
- **Gefahr:** Masse statt Klasse. Deshalb gilt die harte Regel aus I-045-v4: Gezählt wird nur ein Vorteil bei gleichen FLOPs oder gleichen Bytes, mit genug Training.

## 4. Messen, ob es wirkt
- **Trefferquote:** Anteil der Ideen, deren Rauchtest grün war bzw. deren vorregistrierter Lauf H1 bestätigt.
- **Kalibrierung:** Brier-Wert der Vorab-Tipps je Agent bzw. Denkwerkzeug. Werkzeuge mit schlechter Quote bekommen weniger Agenten.
- **Kosten je grüner Idee:** Agenten-Token und CPU-Minuten.
- Alles in `INSTITUT/ideen.json` (Feld `herkunft` = Denkwerkzeug, `vorab_p`, `rauchtest`, `ergebnis`).

## 5. Nächste Welle (Vorschlag)
Je ein Agent für Kasten, Fehlerbuch, Umkehrung, Zufallsimpuls und Perspektiven. Danach Konvergenz, Turnier und Maltes 5-Minuten-Schleife. Starten, sobald I-048, I-049 und I-045-v5 laufen. Die Rauchtests nutzen dann den Container, während Raven rechnet.
