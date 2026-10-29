# Z2 Divergenz: Ideen, Neuheitsprüfung, Rangliste (29.10.2026)

Nichts gerechnet, kein git. Grundlage: ARCHIV.md (gleiches Verzeichnis), LABORBUCH bis 28.10., konv_kandidaten (K/E), Z1_kombinator (C), Z1_grenzen. P = Vorab-Schätzung, dass die Idee im Rauchtest (Tuning-Seeds 950–989, gleiche Bytes/FLOPs, zwei Trainingslängen) über dem stärksten Gegner liegt; keine Befunde. Literaturangaben aus dem Gedächtnis [G], vor Zitat prüfen. Alle Tests ≤ 30 min CPU; Trainingslängen T1 ≈ 600, T2 ≈ 2400 Schritte (Byte-LM d = 128) bzw. 3/12 Epochen (MNIST-MLP). Nachbehandlungstests (PTQ, Cache) sind Frühregime-immun, brauchen aber trotzdem zwei Trainingslängen des Ausgangsmodells.

## A. Mutationen des grünen Befunds („Rotation flacht PTQ-Schaden über Trainingslänge ab“)
| ID | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|
| M1 | **KV-Cache:** Steigung (Schaden_roh − Schaden_rot) je Verdopplung der Trainingslänge, wenn K und V mit 2 Bit (4 Stufen) abgelegt werden | Ausreißerkanäle wachsen beim Training; jede Kanalnormierung hilft. Kontrolle: KIVI-artige Skala je Kanal (K) bzw. je Token (V) ohne Rotation; Rotation nur mit Zufallsvorzeichen (ohne Hadamard) | Byte-LM d=128, T1/T2, danach K/V fake-quantisiert, bpc-Schaden; 5 Tuning-Seeds; ≈ 20 min | 55 % | neu (BK×Rotation leer) |
| M2 | **Aktivierungen:** gleiche Steigung bei 4-Bit-Aktivierung vor der Down-Projektion (FFN-Mitte, z² und GELU) | Aktivierungskurtosis wächst mit T, Rotation glättet nur. Kontrolle: Clipping nach Perzentil, Zufallsvorzeichen | wie M1, Hook an FFN-Mitte; ≈ 20 min | 55 % | neu (nur C05 Kand. mit anderem Fokus) |
| M3 | **Rotation schon im Training:** Gewichte als W = H·V parametrisiert (V trainiert, PTQ auf V ohne Zusatzrotation) gegen Standardtraining + Rotations-PTQ | Adam rechnet in anderer Basis (Teilduplikat K08/C15); Kontrolle: Standardtraining + Rotation nach dem Training; V mit Zufallsorthogonaler statt H | Byte-LM d=128, T1/T2, 2 Bit auf V; ≈ 25 min | 30 % | Teilduplikat K08 (Mechanismus), neu nur Messgröße PTQ-Steigung; behalten |
| M4 | **Skalengesetz-Fit:** Schaden(T) = a·T^b für roh und rot über 4 Längen (300…2400) und 2 Breiten, vorab eingetragene Vorhersage für T = 9600 | Schaden wächst ∝ Gewichtsnorm; Kontrolle: Schaden relativ zur Frobeniusnorm bzw. Kurtosis normiert, Fit nur mit festem b | Byte-LM d=96/128, 4 Längen × 2 Breiten, nur PTQ; ≈ 30 min | 60 % (b_rot < b_roh) | neu (Z1b hat nur 2 Längen) |
| M5 | **Bit-Achse:** Steigung bei echtem 2 Bit (4 Stufen), 3 Bit, 4 Bit gegen ternär 1,58 Bit | Befund war ternär (Vorbehalt I-048): Ternär hat schwere Fehler, der Effekt könnte mit mehr Bits verschwinden. Kontrolle: Rotation ohne Gitteranpassung (gleiche Stufenzahl roh) | Z1b-LM-Skript, Stufenzahl 3/4/8/16; ≈ 15 min | 60 % (Effekt bleibt bei 4 Stufen), 35 % (bleibt bei 8 Stufen) | neu; klärt Vorbehalt |

## B. Fernanalogien (je ein fremdes Fach)
| ID | Fach | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|---|
| F1 | Audio (Sigma-Delta / Rauschformung) | Quantisierungsfehler wird beim Runden in die nächsten Spalten zurückgeführt, gewichtet mit Eingangskovarianz (Rauschformung wie ΣΔ-Wandler), nach Rotation, 2 Bit | = GPTQ/OBQ/GPFQ bekannt [G]. Kontrolle: Rotation + einfaches Runden (bisher alle Archiv-Läufe), GPTQ ohne Rotation, klein-fp16 gleicher Bytes | MNIST-MLP (3/12 Epochen) und Byte-LM, Fehlerrückführung mit 256 Kalibrierproben; ≈ 15 min | 40 % (Gegner klein fp16 schlagen, Tendenz 10/10 in Z1b) | Archiv kennt nur Rundung ohne Rückführung: Baselinelücke, daher behalten |
| F2 | Genetik (alternatives Spleißen) | Ein Gewichtssatz, zur Laufzeit nur Präfix k der Breite benutzt (anytime), weil nach Hadamard die Koordinaten austauschbar sind | Slimmable/Dropout-Netze [G]. Kontrolle: gleiche Abschneidung ohne Rotation, Training mit Breiten-Dropout | MNIST-MLP h=256 Fastfood; Genauigkeit gegen k/h bei 3 und 12 Epochen; ≈ 10 min | 20 % | neu (FL×Rotation nur Zeit/Ops getestet) |
| F3 | Kristallographie (Superzelle + Modulation) | Gewichtsmatrix = eine gekachelte 2-Bit-Einheitszelle B (z. B. 32×32) mit gelernten Zeilen-/Spaltenskalen je Kachel | HashedNets/Gewichtsteilen [G]. Kontrolle: gleiche Bytes als schmale dichte Matrix, Kachel ohne Skalen | MNIST-MLP + Byte-LM FFN; Bytes gleich; ≈ 20 min | 12 % | neu, aber Nähe zu K04/K01 (Teilen); behalten, niedrige Priorität |
| F4 | Verkehr (Zufahrtsdosierung) | Mixture-of-Depths mit parameterfreiem Router: nur die Token mit größtem Zustand ‖x‖² (Born-Gewicht) laufen durch den Block, Rest Skip, FLOPs −50 % | Einfach weniger Rechnung = kleineres Modell [G, Raposo 2024]. Kontrolle: FLOP-gleiches flaches Modell (FF), Zufallsauswahl | Byte-LM d=128, T1/T2 (I-049-Lehre: Frühregime kippt); ≈ 30 min | 20 % | neu (FL×Superposition/Messung leer) |
| F5 | Kartografie (Douglas–Peucker) | KV-Cache wird wie eine Linie vereinfacht: nur Token behalten, deren Schlüssel stark vom Verlauf abweichen (Knickpunkte), Rest löschen | H2O/StreamingLLM-Eviction [G]. Kontrolle: Rezenz + Sinks, Zufall, aufmerksamkeitsbasiert | Nachbehandlung Byte-LM, Cache 25 %; bpc; ≈ 15 min | 20 % | neu |

## C. Ideen für LEERE Zellen
Zellen: S = Superposition, I = Interferenz, M = Messung, V = Verschränkung, U = Rotation, R = RG, K = Fehlerkorrektur, D = Dekohärenz; FL/BG/BK/DA/TZ wie im ARCHIV.

| ID | Zelle | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit / Urteil |
|---|---|---|---|---|---|---|
| L01 | S×FL | Zwei Eingabepositionen per Vorzeichenschlüssel in einen Vektor überlagert (Daten-Multiplexing), FLOPs/2 | Nur kürzere Folge; Kontrolle: Mittelung benachbarter Token ohne Schlüssel | Byte-LM, T1/T2 | 15 % | neu |
| L02 | S×BK | Zwei KV-Paare in einen Slot (K_t + D·K_t+1, Holographic Reduced Representation), Cache −50 % | Verlustbehaftetes Zusammenlegen; Kontrolle: Mittelung ohne Schlüssel, Löschen jedes zweiten Tokens | Nachbehandlung, Aufmerksamkeitsfehler | 15 % | neu |
| L03 | S×DA | Zwei Beispiele per Schlüssel überlagert, beide Labels verlangt, halbe Forward-FLOPs je Beispiel | Mixup-artig; Kontrolle: Standard-Mixup, gleiche Epochen | MNIST-MLP 3/12 Epochen | 12 % | neu |
| L04 | S×TZ | k Modelle in einem Gewichtssatz mit Vorzeichenschlüsseln (BatchEnsemble-artig) als billiges Ensemble | = BatchEnsemble [G]; Kontrolle: ein Modell mit k-facher Breite, FLOP-gleich | MNIST-MLP | 8 % | Duplikat der Literatur, nicht des Archivs; **streichen** (P zu klein) |
| L05 | I×BG | Zwei gegeneinander versetzte Gitter (Nonius): w = a·g1 + b·g2 mit je 1,58 Bit | = nichtuniformes 3,17-Bit-Gitter; Kontrolle: 3-Bit-Gleichgitter, gleiche Bytes | MNIST-MLP PTQ | 12 % | neu, aber schwach; Reserve |
| L06 | I×BK | Kohärente Laufsumme als Cache (lineare Aufmerksamkeit) | Entspricht LRU/Mamba, I-049 rot | – | – | **gestrichen** (Duplikat I-049/C13) |
| L07 | M×BG | Stochastisches Runden (Born-Wahrscheinlichkeit nach Abstand zum Gitterpunkt) mit Seed, Mittel über 4 Läufe | Ensemble von Rundungen kostet 4× FLOPs; Kontrolle: deterministisch | MNIST PTQ | 8 % | **gestrichen** (bricht Gleiche-FLOPs-Regel) |
| L08 | M×BK | Bits je Cache-Eintrag nach Aufmerksamkeitsgewicht (Born p = softmax) statt gleichmäßig, mittlere Bits fest | = Mixed-Precision-KV (MiKV) [G]; Kontrolle: gleichmäßig 2 Bit Rotation, H2O | Nachbehandlung Byte-LM, 2 Längen | 30 % | neu im Archiv; Kandidat für Kombination mit L22 |
| L09 | M×DA | Beispielauswahl ∝ Verlust² (Importance-Sampling), halbe Datenmenge | Bekanntes Coreset-Verfahren; Kontrolle: uniform, ∝ Verlust | Byte-LM halbe Daten (I-045-v5-Regime) | 20 % | neu |
| L10 | M×TZ | „Messkaskade“: Schichten werden nacheinander eingefroren und quantisiert (FreezeOut), Rückwärts-FLOPs sinken | FreezeOut [G]; Kontrolle: Training ohne Einfrieren, gleiche Gesamt-FLOPs | Byte-LM T1/T2 | 15 % | neu |
| L11 | V×BK | Schlüssel eines Tokens als d = a×b Matrix, Rang-1/2-Kronecker-Näherung, Speicher 2√d | Niedrigrang; Kontrolle: Niedrigrang über Token (Palu-artig), gleiche Bytes | Nachbehandlung | 15 % | neu |
| L12 | V×DA | Separierbare erste Schicht (a⊗b) | = K19 | – | – | **gestrichen** (Duplikat K19) |
| L13 | U×DA | Rotation als Augmentierung | = K15 rotiertes Dropout | – | – | **gestrichen** (Duplikat K15) |
| L14 | R×BG | Haar/Walsh-Basis, grobe Skalen mehr Bits | = K06 | – | – | **gestrichen** (Duplikat K06) |
| L15 | R×BK | Haar-Alterung des Caches | = K05 | – | – | **gestrichen** (Duplikat K05) |
| L16 | R×DA | Grobe Oberklassen-Label zuerst (Hierarchie-Hilfsverlust) | Curriculum/Hilfsverlust; Kontrolle: flaches Training | CIFAR-10-Kopf | 8 % | **gestrichen** (zu schwach, nur Reserve) |
| L17 | K×FL | Hadamard-Code-Ausgabekopf | = K12 | – | – | **gestrichen** |
| L18 | K×BK | Ausreißer-Parität: 2-Bit-Cache plus dünner Korrekturstrom nur für die größten Fehler (dense-and-sparse) | KVQuant [G]; Kontrolle: nur Rotation (Ausreißer verschwinden), gleiche Bytes | Nachbehandlung, nach Rotation | 12 % | neu im Archiv; Nebenarm von M1 |
| L19 | K×DA | ECOC-Etiketten | = K12 | – | – | **gestrichen** |
| L20 | K×TZ | Fehlerrückführung im QAT | im Straight-Through-Latentgewicht schon enthalten, E3 | – | – | **gestrichen** (Duplikat E3) |
| L21 | D×FL | Neuronen mit \|z\| < ε überspringen | = K10, E8 | – | – | **gestrichen** |
| L22 | D×BK | „Kohärenzzeit“: Bits je Cache-Eintrag fallen mit Alter (fp16 Fenster, dann 4, 2, 1 Bit), mittlere Bits fest | Rezenzfenster (KIVI) plus Treppe [G]; Kontrolle: gleichmäßige mittlere Bits, L08 | Nachbehandlung, 2 Längen | 30 % | neu (K05 Alterung ist Skalen-Haar, nicht Bitstufen); behalten |
| L23 | D×DA | Labelrauschen/Annealing | Label-Smoothing; E9/K15 | – | – | **gestrichen** |
| L24 | D×TZ | Quantisierungsrauschen mit fallender Stärke (Noise-QAT) | = E3 weicher Übergang | – | – | **gestrichen** (Duplikat E3) |

Bilanz: 24 Leerzellen, 14 Duplikate oder zu schwach gestrichen, 10 Ideen behalten (L01–L03, L05, L08–L11, L18, L22); dazu 5 Mutationen und 5 Fernanalogien. Dieser Befund an sich ist wichtig: Die meisten Leerzellen waren schon durch K/E/C-Kandidaten belegt, aber nie gerechnet.

## D. Rangliste Top 6 (Rang = P × Nutzen für die Arbeit, Kosten, Lücke im Raster)
| Rang | ID | Einzeiler | P | Test |
|---|---|---|---|---|
| 1 | **M5** | Gilt „Rotation flacht PTQ-Schaden ab“ auch bei echten 4 Stufen und 3/4 Bit, nicht nur ternär? Klärt den Vorbehalt des grünen Befunds. | 60 % | 15 min |
| 2 | **M1** | Rotation flacht auch den KV-Cache-Schaden (2 Bit) über die Trainingslänge ab. | 55 % | 20 min |
| 3 | **M4** | Skalengesetz-Fit Schaden(T)=a·T^b für roh/rot mit eingetragener Vorhersage bei T=9600. | 60 % | 30 min |
| 4 | **F1** | Audio-Rauschformung (Fehlerrückführung) nach Rotation: stärkste Baseline gegen „klein fp16 gleicher Bytes“. | 40 % | 15 min |
| 5 | **L22** | Cache-Bits fallen mit Alter (Treppe), gegen gleichmäßig 2 Bit und aufmerksamkeitsbasierte Bits (L08). | 30 % | 15 min |
| 6 | **M2** | Dieselbe Steigung für 4-Bit-Aktivierung vor der Down-Projektion (z² und GELU). | 55 % | 20 min |

Zusammensetzung: 4 Mutationen (M5, M1, M4, M2), 1 Fernanalogie (F1), 1 Leerzellenidee (L22). M1, M2 und M5 teilen das Z1b-LM-Skript (Hooks an K/V bzw. FFN-Mitte, Stufenzahl als Parameter): ein Skript, drei Aufträge.

### Top-6-Details und Kontrollen (Pflichtfeld „langweiligste Erklärung“)
1. **M5** Langweiligste Erklärung: Der Schaden fällt nur bei sehr grobem (ternärem) Gitter auf, Rotation behebt dort einfach die Ausreißer. Kontrolle: Stufenzahl 3/4/8/16 beider Arme; Auswertung = Steigung je Stufenzahl. Erfolg: Steigungsdifferenz bleibt bei 4 Stufen >0 (Seeds plus ≥ 8/10); Misserfolg ehrlich als „gilt nur ternär“ berichten.
2. **M1** Langweiligste Erklärung: Schlüsselkanäle haben Ausreißer, per-Kanal-Skala reicht (KIVI). Kontrolle: Skala je Kanal ohne Rotation und Zufallsvorzeichen-Rotation. Erfolg: Rotation flacher als beide Kontrollen bei T2, Rauschband aus 5 Seeds.
3. **M4** Langweiligste Erklärung: Schaden wächst mit der Gewichtsnorm. Kontrolle: Norm-normierter Schaden. Erfolg: b_rot < b_roh mit Intervall getrennt, Vorhersage bei T=9600 innerhalb des Streubands (später Raven, nur mit Maltes Zeile).
4. **F1** Langweiligste Erklärung: Das ist GPTQ; der Gewinn ist bekannt. Wert: Er ersetzt die bisherige schwache Rundungs-Baseline des Archivs. Kontrolle: Rotation + Runden, GPTQ ohne Rotation, klein-fp16. Erfolg: rot+Rückführung schlägt klein-fp16 gleicher Bytes bei beiden Trainingslängen.
5. **L22** Langweiligste Erklärung: Recent-Fenster + Eviction reicht. Kontrolle: gleichmäßig mittlere Bits, Rezenzfenster allein, L08. Erfolg: bpc besser als gleichmäßig bei gleichen Cache-Bytes, beide Längen.
6. **M2** Langweiligste Erklärung: Aktivierungsausreißer; Perzentil-Clipping hilft genauso. Kontrolle: Clipping, Zufallsvorzeichen. Erfolg wie M1.

Ehrlichkeitshinweis: Alles klassisch (I-038), Formulierung „quanten-inspirierter Vorteil“. Reserve bei Bedarf: F2, F4, F5, L08, L09, L10.
