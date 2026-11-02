# k3 — Denkwerkzeug Umkehrung (Welle 2, 28.10.2026)

Nur gedacht und gelesen (KREATIVITAET_2026-10-27, SYNTHESE, KERNAUSSAGE, LABORBUCH ab I-044, Vorregistrierungen I-047/48/49/45-v5). Nichts gerechnet, keine WebSearch. Literatur nur aus dem Gedächtnis, vor Verwendung prüfen. Wirkung überall klassisch (I-038), Formulierung „quanten-inspiriert“. Vorteil zählt nur bei gleichen FLOPs ODER gleichen Bytes, mit genug Training.

## Teil 1: 15 stille Annahmen und ihre Umkehrung

| # | Stille Annahme | Umkehrung | Idee |
|---|---|---|---|
| A1 | Gewichte werden gelernt | Gewichte stehen fest (Seed), gelernt werden nur Skalen oder die zweite Hälfte | 1, 2 |
| A2 | Mehr Training ist fairer (Kosten = Schritte) | Training beginnt klein und wächst; Kosten = gesamte Rechenmenge | 15 |
| A3 | Modell = eine Matrix je Schicht | eine Matrix für alle Schichten, Schichten unterscheiden sich nur durch einen Schlüssel | 3 |
| A4 | FLOPs sind die Kostengröße | Kosten = bewegte Bytes; Rechnen ist fast gratis, Speichern teuer (Rechnen statt Speichern) | 1, 2, 4 |
| A5 | Daten sind fest | Eingabe wird vor dem Modell im Spektrum beschnitten, Daten werden Teil der Architektur | 6 |
| A6 | Ausgabe ist ein Vektor (Softmax über Klassen) | Ausgabe ist ein Codewort eines Fehlerkorrektur-Codes | 5 |
| A7 | Gewichte sind zur Laufzeit statisch | Struktur hängt von der Eingabe ab | 7 |
| A8 | Jedes Gewicht hat gleich viele Bits | Bits fallen fraktional an (Entropiecode) | 8 |
| A9 | Ein Gewichtssatz je Aufgabe | viele Aufgaben überlagert in einem Satz, Schlüssel wählt aus (Superposition + Messung) | 9 |
| A10 | Aktivierungen werden fürs Rückwärtsrechnen gespeichert | Aktivierungen werden zurückgerechnet (Unitarität = Umkehrbarkeit) | 10 |
| A11 | Optimiererzustand in voller Genauigkeit | Zustand in rotierter Basis mit wenigen Bits | 11 |
| A12 | Der Zustand (KV-Cache) wird so gespeichert, wie er entsteht | Cache nach Rotation mit 1–2 Bit | 12 |
| A13 | Rauschen ist schädlich | Rauschen (Dither) ersetzt Bits: 1 Bit plus Mittel über Seeds | 13 |
| A14 | Eine Gewichtsmatrix ist eine Tabelle von Zahlen | Matrix = Summe weniger Pauli-Strings bzw. Funktion weniger Frequenzen | 4, 16 |
| A15 | Inferenz ist ein deterministischer Durchlauf | mehrere „Messungen“ mit verschiedenen Schlüsseln, dann Mittel | 14 |

Abgrenzung: Schon in SYNTHESE oder laufend (nicht wiederholt): Hadamard-Rotation vor 2/3-Bit (I-048), Zufalls-Orthogonale als Kontrolle, Hourglass/LRU (I-049), TT/MPS-FFN (I-047), Nur-Phase/Born (gestrichen), Rangbudget nach Entropie, Monarch, Fastfood+z² (I-045-v4/v5), frühes Aussteigen, Born-Attention.

---

## Teil 2: Ideen

### 1. Eingefrorene Seed-Aufwärtsschicht, nur Abwärtsschicht und Zeilenskalen lernen · Umkehrung (A1, A4) · Quantenbezug: eine feste Haar-zufällige „Einbettung“ in einen größeren Raum (Ancilla), nur die Auslese wird gelernt.
- **Mechanismus:** In jeder FFN ist W1 (d→h) eine Zufallsmatrix aus einem Seed (0 gespeicherte Bytes, bei Bedarf neu erzeugt), gelernt werden nur ein Skalenvektor je Zeile von W1 (h Zahlen) und W2 (h→d). Bytes ≈ halbe FFN. Bei gleichen FLOPs wie ein dichtes Modell gleicher Breite verliert es vermutlich leicht, aber bei gleichen Bytes darf es breiter sein (h doppelt) als die dichte Kontrolle. Gewinn nur, falls Breite mehr bringt als gelernte W1 (Zufallsmerkmale, „extreme learning“).
- **Langweiligste Erklärung:** Zufallsmerkmale + trainierte Auslese; eine breitere FFN hilft einfach, W1 ist egal. Dann ist es „eingefrorene Lagen sind fast gratis“ (bekannt).
- **Billigster Test:** Byte-LM d=128, 4 Lagen, 600 Schritte, Tuning-Seeds 970–972. Arme: (a) Zufalls-W1 + Skalen, h=4d·1,6; (b) dicht gelernt, h so schmal, dass Bytes gleich (≈ 0,6·4d). Kontrolle (c): gleich wie (a) ohne Zeilenskalen. Vergleich bpc bei gleichen Bytes; FLOPs separat ausweisen (a hat mehr).
- **Vorab-Tipp:** P(grün) 30 %, Jury 3.

### 2. Zufallscodebuch aus Seed (Gain-Shape-Quantisierung) · Umkehrung (A1, A4: Rechnen statt Speichern) · Quantenbezug: Shannons Zufallscodierung / Haar-zufällige Vektoren, das Codebuch ist ein Zustand, der nie gespeichert wird.
- **Mechanismus:** Jeder Block aus g=8 Gewichten eines trainierten Modells wird durch den besten von 256 Zufallsvektoren (aus Seed erzeugt) mal einer Skala ersetzt. Speicher: 8 Bit Index + ~4 Bit Skala je 8 Gewichte ≈ 1,5 Bit/Gewicht, Codebuch gratis. Zufallsvektoren in hoher Dimension füllen die Kugel gleichmäßiger als ein Würfelgitter (Skalar-2-Bit). Bei gleichen Bytes könnte der Fehler kleiner sein als bei gleichförmigen 1,5 Bit. Rechenaufwand: Regenerieren pro Block (Zusatz-FLOPs, ausweisen).
- **Langweiligste Erklärung:** Vektorquantisierung schlägt Skalarquantisierung bei niedrigen Bitraten (bekannt, SeedLM-artig; QuIP#/AQLM). Kein quantenspezifischer Teil.
- **Billigster Test:** MNIST-MLP 784-512-512-10 (5 Epochen, Seeds 970–972), danach je Schicht Quantisierung: (a) Seed-Codebuch g=8, (b) Skalar 2 Bit + Clip + Hadamard (I-048-Arm), (c) Skalar 1,5-Bit-äquivalent (Ternär + Skala). Bytes angleichen, Test-Genauigkeit ohne Nachtraining. Kontrolle (d): Codebuch aus Gittervektoren (E8) statt Zufall.
- **Vorab-Tipp:** P(grün gegen c) 60 %, gegen b 30 %; Jury 3.

### 3. Eine Matrix für alle Lagen, Lagenschlüssel durch Vorzeichen-Diagonalen · Umkehrung (A3) · Quantenbezug: ein festes „Hamiltonoperator“ H, jede Lage ist dieselbe Zeitentwicklung in einer anderen, durch zufällige Phasen D_l gedrehten Basis (D_l H D_l).
- **Mechanismus:** FFN-Gewichte W1, W2 werden über alle L Lagen geteilt; Lage l bekommt feste (Seed) oder gelernte Vorzeichen-/Skalendiagonalen vor und nach W (Fastfood-artig, d Zahlen je Lage). Bytes ≈ 1/L der FFN, FLOPs gleich wie ungeteilt. Bei gleichen Bytes kann die geteilte FFN L-fach breiter sein; ohne Schlüssel ist Teilung bekanntlich schwach (Universal Transformer/ALBERT), die Schlüssel sollen die Lagen unterscheidbar machen.
- **Langweiligste Erklärung:** Breitere Teil-FFN hilft, die Diagonalen tun nichts (Kontrolle ohne Diagonalen zeigt dasselbe).
- **Billigster Test:** Byte-LM d=128, L=4, 600 Schritte, Seeds 970–972. Arme bei gleichen Bytes: (a) geteilt, h=4d·4, mit gelernten Diagonalen; (b) geteilt ohne Diagonalen; (c) ungeteilt dicht h=4d. FLOPs: (a) hat 4× mehr als (c) – also zusätzlich (d) geteilt mit h=4d, gleiche FLOPs wie (c), 4× weniger Bytes. Ziel: (d) ≈ (c).
- **Vorab-Tipp:** P(grün, (d) gleichauf (c)) 35 %; Jury 4.

### 4. FFN als Summe weniger Pauli-Strings, Koeffizienten gelernt · Umkehrung (A4, A14) · Quantenbezug: Jeder Operator ist eine Linearkombination von Pauli-Strings (Hamiltonians aus wenigen Pauli-Termen).
- **Mechanismus:** W = Σ_{k=1..K} c_k · X^{a_k} Z^{b_k}, mit festen zufälligen (a_k, b_k) aus dem Seed; ein Pauli-String ist Permutation (Bit-XOR der Indizes) mal Vorzeichen, kostet O(N) statt O(N²), kein Speicher außer c_k. Parameter und FLOPs: K·N. Bei K≈16 und N=512 ist das dünn besetzte Zufallsmatrix mit Diagonalstruktur, aber (anders als Fastfood) mit vielen unabhängigen Koeffizienten. Mehr Ausdruck je FLOP als Niedrigrang? Offen.
- **Langweiligste Erklärung:** Es ist eine zufällige dünne Matrix mit K Einträgen je Zeile (Sparse-Training bekannt); die Pauli-Sprache ändert nichts.
- **Billigster Test:** MNIST-MLP und Byte-LM d=128. Arme gleicher FLOPs (K·N je Matrix) : Pauli-K, dichte schmale FFN, Niedrigrang, zufällige dünne Matrix mit gleich vielen Einträgen (Kontrolle, Entscheidend: Pauli = Zufallsdünn?). 600 Schritte, Seeds 970–972. Vorabzahl: gleiche FLOPs genau zählen.
- **Vorab-Tipp:** P(grün gegen Niedrigrang) 25 %, P(Pauli besser als Zufallsdünn) 10 %; Jury 3 (schöne Sprache, aber Kontrolle entlarvt es wohl).

### 5. Hadamard-Codewort-Ausgabe statt Softmax-Kopf · Umkehrung (A6) · Quantenbezug: Fehlerkorrektur: Klassen werden als Codewörter mit großem Abstand kodiert, Auslese per Messung der Bits.
- **Mechanismus:** Byte-LM: statt d→256 Logits wird d→32 Werte (Zeilen einer festen Hadamard-Matrix als Codebuch aller 256 Bytes, davon 32 Koordinaten), Klassenwahrscheinlichkeit aus Korrelation mit den Codewörtern (Softmax über 256 Skalarprodukte, die Codebuchmatrix ist fest und 0 Bytes). Spart Bytes und FLOPs im Kopf (d·32 statt d·256). Bei Byte-LM d=128 ist der Kopf ein beträchtlicher Teil der Gesamtparameter.
- **Langweiligste Erklärung:** Es ist ein Niedrigrang-Ausgabekopf (Rang 32, feste Rechtsmatrix); Niedrigrang-Köpfe sind bekannt, die Hadamard-Wahl ist egal (Zufallscodebuch gleich gut).
- **Billigster Test:** Byte-LM d=128, L=4, 600 Schritte, Seeds 970–972. Arme: (a) Hadamard-Code 32, (b) Zufalls-Code 32, (c) gelernter Niedrigrang 32 (gelernt beide Seiten), (d) voller Kopf (Bezug). Gleiche Bytes: (a),(b),(c) wie nahe; (a) hat mehr Bytes Ersparnis weil Codebuch frei, daher (e) dichter Kopf mit Rang 32+Bytes-Ausgleich in der FFN. Vergleich bpc.
- **Vorab-Tipp:** P(grün, a ≥ e) 35 %; Jury 3.

### 6. Eingabe im Walsh-Spektrum beschneiden, erste Schicht schrumpft · Umkehrung (A5) · Quantenbezug: man „misst“ nur die informativsten Walsh-Koeffizienten des Zustands (Eingabe).
- **Mechanismus:** MNIST 784→1024 (aufgefüllt), Hadamard-Transformation fest, danach nur die k größten (oder festen niedrigen) Koeffizienten als Eingabe, k=128. Erste Schicht 128·h statt 784·h: spart FLOPs und Bytes der größten Schicht. Bei gleicher Gesamt-FLOPzahl Breite erhöhen.
- **Langweiligste Erklärung:** PCA/DCT-Kompression der Eingabe; die Information liegt in wenigen Frequenzen.
- **Billigster Test:** MNIST/Fashion-MLP, 5 Epochen, Seeds 970–972; k ∈ {64,128,256}; Kontrollen: zufällige k Pixel, PCA-k (bester Gegner). Gleiche FLOPs durch Anpassen der Breite. Fashion ist härter für abgeschnittene Spektren (Rauchtest dort zusätzlich).
- **Vorab-Tipp:** P(Walsh ≥ zufällige Pixel) 90 %, P(Walsh ≥ PCA gleiche FLOPs) 15 %; Jury 2.

### 7. Fastfood mit eingabeabhängigen Diagonalen (schaltbare Struktur) · Umkehrung (A7) · Quantenbezug: eine von Vorzeichen/Phasen „gesteuerte“ Rotation, wie ein klassisch gesteuertes Gatter (bedingte Gatter).
- **Mechanismus:** Fastfood-Lage S·H·G·Π·H·B: die Diagonale G (oder S) wird aus der Eingabe x durch eine kleine lineare Projektion erzeugt (g = σ(Ux), U: d→r mit r klein, dann Aufweitung per fester Hadamard). Struktur bleibt billig (O(N log N)), Ausdruck steigt, da Gewicht von x abhängt (wie Gating/Mamba-Selektivität). In v4 verlor reines Sz bei gleichen FLOPs; hier soll Eingabeabhängigkeit die Lücke schließen.
- **Langweiligste Erklärung:** Multiplikative Gating-Wechselwirkung (GLU) bringt es, nicht die Fastfood-Struktur. GLU-Kontrolle mit dichter Matrix gleicher FLOPs erklärt es.
- **Billigster Test:** Byte-LM d=128, 600 Schritte, Seeds 970–972. Arme gleicher FLOPs: (a) Sz abhängig, (b) Sz statisch (v4-Arm), (c) dichte FFN schmal + GLU, (d) dichte FFN schmal ohne GLU. Ziel: (a) > (b) und (a) ≈ (d).
- **Vorab-Tipp:** P(a>b) 70 %, P(a≥d bei gleichen FLOPs) 15 %; Jury 3.

### 8. Entropiecodierte Rotationsgewichte (fraktionale Bits) · Umkehrung (A8) · Quantenbezug: Nach Hadamard-Rotation sind Gewichte Gauß-ähnlich (zentraler Grenzwertsatz); die Entropie eines Gauß-Quantisierers ist kleiner als seine feste Bitbreite.
- **Mechanismus:** Statt 2 Bit fest: Lloyd-Max-Quantisierung mit Huffman/arithmetischer Codierung, die mittlere Codelänge liegt bei gleicher Verzerrung ~0,25–0,4 Bit unter der festen Bitbreite. Bei gleichen Bytes ist die Verzerrung kleiner als bei I-048 „had 2 Bit“. Dekodierung kostet Zusatzoperationen (ausweisen).
- **Langweiligste Erklärung:** Bekannte Entropiecodierung (Deep Compression, Han et al. 2016); Gewinn ist reine Quellencodierung.
- **Billigster Test:** MNIST-MLP trainiert (Seeds 970–972, 5 Epochen), Hadamard-rotiert; Bytes: (a) gleichförmig 2 Bit, (b) Lloyd-Max mit berechneter Entropie als Bytes (mittlere Bits = 2,0 insgesamt bestimmt aus der Zahl der Level, z. B. 5 Level ≈ 2,0 Bit Entropie), (c) wie (a) ohne Rotation (Kontrolle). Test-Genauigkeit bei exakt gleichen Gesamtbytes.
- **Vorab-Tipp:** P(grün, b>a) 60 %; Jury 2 (kein Quantenneues).

### 9. Superposition zweier Aufgaben in einem Gewichtssatz (Schlüssel = Vorzeichenvektor) · Umkehrung (A9) · Quantenbezug: Zustände werden überlagert und durch Messung in einer Schlüsselbasis wieder getrennt (Superposition + Messung).
- **Mechanismus:** Ein MLP wird auf Aufgaben 1…n trainiert, jede Aufgabe hat einen festen zufälligen Vorzeichenvektor s_t, der Eingabe und/oder Zwischenschichten per Hadamard·Diag(s_t) dreht („Schlüssel“). Gewichte werden geteilt, Bytes ≈ 1/n gegenüber n getrennten Modellen. Bei gleichen Bytes (n kleine Modelle gegen ein n-fach größeres geteiltes) könnte das geteilte Modell wegen größerer Kapazität gewinnen, sofern die Aufgaben sich nicht stören (Cheung et al. 2019, „Superposition of many models into one“, aus dem Gedächtnis).
- **Langweiligste Erklärung:** Ein größeres Multitask-Netz ist besser als viele kleine; die Schlüssel sind überflüssig (Kontrolle ohne Schlüssel = Multitask mit Aufgaben-Etikett).
- **Billigster Test:** Aufgaben: MNIST, Fashion, MNIST mit fester Pixel-Permutation (3 Aufgaben), 5 Epochen, Seeds 970–972. Arme bei gleichen Bytes: (a) ein geteiltes Modell mit Schlüsseln, Breite 3h; (b) 3 getrennte Modelle Breite h; (c) geteiltes Modell ohne Schlüssel (nur Vorteil über Etikett, Ausgabeköpfe getrennt). Mittel Test-Genauigkeit.
- **Vorab-Tipp:** P(a>b) 55 %, P(a>c) 20 %; Jury 4 (starke Superpositions-Geschichte, auch für Adapter, Projekt A).

### 10. Umkehrbare orthogonale Blöcke: Aktivierungen zurückrechnen statt speichern · Umkehrung (A10) · Quantenbezug: Unitäre Entwicklung ist umkehrbar; Information geht nicht verloren, also muss nichts gespeichert werden.
- **Mechanismus:** Residualblöcke im RevNet-Stil (zwei Hälften, x1+F(x2)), F enthält Hadamard-Mischung; im Rückwärtsgang wird die Aktivierung aus der Ausgabe rekonstruiert statt gespeichert. Aktivierungsspeicher unabhängig von der Tiefe, Preis ≈ +33 % FLOPs im Training. Gleiche Bytes im Training erlauben größere Batches oder Modelle.
- **Langweiligste Erklärung:** RevNet (Gomez et al. 2017) tut genau das; Quantenbezug rein verbal.
- **Billigster Test:** Byte-LM d=128, L=8, 300 Schritte: (a) umkehrbar, (b) normal, (c) normal mit Gradient-Checkpointing (stärkster Gegner). Gemessen: Spitzenspeicher (Bytes) und bpc nach gleichen Schritten; erwartet gleiche bpc (zu zeigen: kein Verlust durch die Teilung in Hälften bei gleichen Parametern).
- **Vorab-Tipp:** P(grün: bpc gleich bei weniger Speicher) 55 %, P(besser als Checkpointing) 15 %; Jury 2.

### 11. Optimiererzustand in Hadamard-Basis mit 2–4 Bit · Umkehrung (A11) · Quantenbezug: Rotation vor Niedrigbit (wie I-048), aber auf Adam-Momenten statt Gewichten.
- **Mechanismus:** Adam speichert m und v (zwei Zahlen je Parameter, 8 Byte in fp32). Zustand wird blockweise Hadamard-rotiert und mit 4 Bit abgelegt, v in Log-Skala. Trainingsbytes sinken um ≈ 75 % beim Zustand, Gewichte bleiben fp32. Bei gleichen Trainingsbytes ließe sich das Modell größer machen. Rotation soll die Ausreißer in m (Spitzen) glätten (bekannt: 8-Bit-Adam, Dettmers 2022; Rotation neu).
- **Langweiligste Erklärung:** 4-Bit-Zustand klappt auch ohne Rotation (4-Bit-Optimierer, Li et al. 2023, aus dem Gedächtnis); Rotation nutzlos.
- **Billigster Test:** Byte-LM d=128, 600 Schritte, Seeds 970–972: (a) fp32-Adam, (b) 4-Bit ohne Rotation, (c) 4-Bit mit Hadamard, (d) 2 Bit mit Hadamard. Kennzahl: bpc gegen fp32 und gespeicherte Bytes. Entscheidend ist der Unterschied (c) − (b) bei 2 Bit.
- **Vorab-Tipp:** P(c≈a bei 4 Bit) 80 %, P(Rotation hilft bei 2 Bit) 35 %; Jury 3.

### 12. KV-Cache nach Rotation mit 1–2 Bit, mehr Kontext bei gleichen Cache-Bytes · Umkehrung (A12) · Quantenbezug: Der Zustand (Schlüssel/Werte) wird in einer Rotationsbasis „gemessen“, in der Ausreißerkanäle verschwinden.
- **Mechanismus:** Schlüssel und Werte jedes Tokens werden vor dem Ablegen hadamard-rotiert, mit 2 Bit gespeichert, bei Gebrauch dequantisiert (Rechnen statt Speichern). Gleiche Cache-Bytes erlauben 8× mehr Kontext bzw. mehr Köpfe. Bekannt für große Modelle (KIVI, QuaRot, aus dem Gedächtnis); neu für uns: Cache-Achse, bisher nur Gewichte (I-048).
- **Langweiligste Erklärung:** Rotation spielt bei Byte-LM mit d=128 keine Rolle, Rundung ohne Rotation reicht (kleine Modelle haben kaum Ausreißer, vgl. w4).
- **Billigster Test:** Vortrainiertes Byte-LM (aus Resten des Laufs, 1500 Schritte, Seed 970) nur Inferenz mit Cache-Quantisierung: (a) fp16, (b) 2 Bit roh, (c) 2 Bit + Hadamard, (d) 1 Bit + Hadamard. Test-bpc bei Kontext 128/256, Bytes des Caches. Kontrolle: Zufallsrotation.
- **Vorab-Tipp:** P(c≈a) 50 %, P(c besser als b) 30 %; Jury 3.

### 13. Dither statt Bits: 1-Bit-Gewichte, gemittelt über Seeds · Umkehrung (A13) · Quantenbezug: Wiederholte Messungen mit zufälligen Einstellungen mitteln die Auslese aus (Schussrauschen wird zur Ressource; Messstatistik).
- **Mechanismus:** Gewichte werden mit 1 Bit gespeichert; bei der Inferenz erzeugt ein Seed K verschiedene Dither-Rundungen (stochastische Rundung des gespeicherten Gewichts auf ±s), die Ausgaben werden über K Läufe gemittelt. Fehler sinkt ~1/√K, die Gewichtsbytes bleiben 1/16 von fp16. Preis K× FLOPs (bei gleichen Bytes ausgewiesen); die Regel „gleiche FLOPs ODER gleiche Bytes“ wird bei Bytes erfüllt und ehrlich gesagt, dass FLOPs steigen.
- **Langweiligste Erklärung:** Mittelung von K quantisierten Netzen ist ein Ensemble ohne Mehrinformation: bei Rundung auf ein festes Gitter fehlt die Information (Mittel konvergiert zum Rundungswert). Dither hilft nur, wenn die Quantisierung gespeicherte Zusatzinformation (Subbits) hat – möglicherweise bringt K nichts.
- **Billigster Test:** MNIST-MLP trainiert; gespeichert fp32; (a) 1 Bit deterministisch, (b) 1 Bit + Dither-Mittel K=8, (c) 2 Bit deterministisch (gleiche Bytes wie 2 Bit), (d) 1 Bit mit K=8 und stochastischer Rundung, aber **pro Gewicht** ein gespeicherter Rundungswert (Kontrolle: das wäre schon 2 Bit). Nur sinnvoll, wenn das Netz so trainiert wird (Rauschen im Training): QAT 300 Schritte Seeds 970–972.
- **Vorab-Tipp:** P(b>a) 70 %, P(b≥c bei gleichen Bytes) 15 %; Jury 3 (ehrliche Negativ-Erkenntnis möglich).

### 14. Geteilte Gewichte, K Rotationsschlüssel, Mittel („Messwiederholung“) · Umkehrung (A15) · Quantenbezug: Dieselbe Zustandsvorbereitung wird mehrfach in verschiedenen Messbasen ausgelesen; Ergebnisse werden gemittelt (Schussstatistik, vgl. I-040).
- **Mechanismus:** Ein Modell mit Breite w und Bytes B, K mal mit verschiedenen festen Hadamard·Diag(s_k)-Eingabe-Drehungen ausgeführt, Logits gemittelt (Training mit zufällig gewähltem k). Bei gleichen FLOPs: Breite w/√K. Bei gleichen Bytes: Ensemble ohne Zusatzbytes. Behauptet: gleiche Gewichte bilden K nicht-identische „Ansichten“.
- **Langweiligste Erklärung:** Test-Time-Augmentation / Ensemble mit gleichen Gewichten; bei gleichen FLOPs verliert man mehr durch schmaleres Modell als man durch Mittelung gewinnt.
- **Billigster Test:** MNIST-MLP, 5 Epochen, Seeds 970–972: (a) K=4 Rotationen, Breite h/2, (b) K=1 Breite h (gleiche FLOPs), (c) K=4 gleiche Gewichte ohne Rotation (identische Kopien = Kontrolle, muss = K=1 sein). Auch gleiche Bytes: (a) Breite h gegen (b) Breite h (nur Genauigkeit, FLOPs ×4).
- **Vorab-Tipp:** P(a>b bei gleichen FLOPs) 15 %, P(a>b bei gleichen Bytes) 65 %; Jury 3.

### 15. Wachsendes Modell: klein trainieren, isometrisch verbreitern · Umkehrung (A2) · Quantenbezug: Einbettung in einen größeren Hilbertraum durch Hinzufügen von Hilfsbits (Ancilla) ist isometrisch und erhält die Funktion.
- **Mechanismus:** Training beginnt mit Breite d/2 (4× billiger je Schritt), dann wird per funktionserhaltender Isometrie (Kopieren der Gewichte in einen Block, Hadamard-Mischung, kleines Rauschen für Symmetriebruch) auf Breite d verbreitert und weiter trainiert. Gesamt-FLOPs unter dem Direkttraining; bei gleichen FLOPs gewinnt es nur, wenn das große Modell das frühe Training wirklich unterbewertet. Bekannt: Net2Net (Chen et al. 2016), Stacking/„progressive growing“ (aus dem Gedächtnis). Berührt I-045-v5 (Trainingsbudget), aber andere Frage.
- **Langweiligste Erklärung:** Es reproduziert den Frühregime-Effekt (kleines Modell lernt anfangs schneller) und verliert bei langem Training; der Gewinn bei kleinem T ist nur Artefakt.
- **Billigster Test:** Byte-LM, 600 Schritte gesamt, Seeds 970–972: (a) d=64 für 300 Schritte, dann d=128 für 300; (b) d=128 für 600 Schritte mit gleichen Gesamt-FLOPs (kürzer: 380 Schritte-Äquivalent); (c) d=128 von Anfang an alle 600 (obere Schranke). Dritte Kontrolle: (a) ohne Funktionserhaltung. Zusätzlich T=2400 für Langzeitverhalten (Regime beachten).
- **Vorab-Tipp:** P(a>b bei gleichen FLOPs, kurz) 55 %, P(bei langem Training) 20 %; Jury 3.

### 16. Gewichte im Frequenzraum: zufällige Walsh-Koeffizienten-Teilmenge · Umkehrung (A14) · Quantenbezug: Der Operator wird in der Walsh-Hadamard-Basis (Quanten-Fourier-Ebene) dargestellt, nur eine feste Teilmenge der Koeffizienten wird gespeichert.
- **Mechanismus:** W = H_N · Ĉ · H_N mit Ĉ nur an K festen zufälligen Positionen (Seed) belegt, sonst 0 (Bytes ≈ K). Anwendung via Hadamard-Transformationen ist O(N log N) (+ dünner Anteil). Anders als Fastfood (nur Diagonale) liegt Information in beliebigen Spektralpositionen. Bei gleichen Bytes die Gewichtsmatrix einer fp16-Schicht ersetzen.
- **Langweiligste Erklärung:** Die ersten Nachbarn sind niedrige Frequenzen = glatte Matrix = untauglich; mit Zufallspositionen ist es eine unscharfe, dünne Zufallsmatrix im Spektrum – Pauli-Idee 4 in anderer Verkleidung.
- **Billigster Test:** MNIST-MLP (Seeds 970–972, 5 Epochen) mit W in Walsh-Koeffizienten-Form, K = 1/16 der Einträge; Arme bei gleichen Bytes: (a) zufällige Positionen, (b) niedrigste K Sequenzen (nach Walsh-Ordnung), (c) dünne Zufallsmatrix im Gewichtsraum (Kontrolle), (d) Niedrigrang. 
- **Vorab-Tipp:** P(a>d) 20 %, P(a>c) 10 %; Jury 2.

---

## Teil 3: Auswahl (meine Einschätzung)
- Beste Aussicht mit Kontrolle und Jury-Wirkung: **Idee 3** (geteilte Matrix mit Lagenschlüssel; gleiche FLOPs, 4× weniger Bytes, Schlüssel-Kontrolle trennt Effekt), **Idee 9** (Superposition der Aufgaben, starke Quantenerzählung, passt zu Adapter-Projekt A), **Idee 2** (Zufallscodebuch aus Seed: „Gewichte aus einem Seed erzeugen“, billig, Kontrolle klar).
- Ehrlich schwach, aber billig zum Streichen: 4/16 (Pauli/Walsh = Zufallsdünn?), 6, 13.
- Viele der Byte-Ideen (2, 8, 11, 12, 13) sind Fortsetzungen von I-048; sinnvoll erst nach dessen Ergebnis, um nicht dieselbe Frage doppelt zu prüfen.
