# 05 – Extreme Parametereffizienz: Wie wenige Parameter braucht MNIST?

Stand 05.10.2026. Quellenlage: Das Websuch-Budget der Sitzung (200 Anfragen, geteilt) war nach rund 15 eigenen Anfragen aufgebraucht. arxiv.org, proceedings.mlr.press und greydanus.github.io sind für WebFetch gesperrt, daher stammen viele Zahlen nur aus **Suchauszügen** ([Auszug]). Alles ohne Quelle ist als [Lehrbuch], [EIGEN] oder [?] markiert. Eigene Rechnungen: Skripte im Scratchpad (`a.py`–`d.py`), volle 60k/10k-MNIST, ein Seed, keine Konfidenzintervalle.

## 1. Kurzfassung
- **Zählweise zuerst.** „Parameter“ ist eine schwache Maßzahl. Ehrlich ist die **Gesamtbeschreibungslänge in Bit**: trainierte Zahlen + feste Vorverarbeitung (Downsampling-Wahl, Maske, Partition, Seeds) + Architekturbeschreibung. Eine zufällige Projektionsmatrix ist nur dann „gratis“, wenn man ihren Seed nicht auswählt.
- **Kleinstes bekanntes 99-%-Modell** (Grimov-Repo): 1.936 Parameter, 99,1 % [Auszug]. Unter 1.000 Parametern mit ≥99 % habe ich nichts gefunden.
- **Eigene Messungen im Bereich ~100 Parameter:** 3×3-Mittelwerte + Softmax (100 Par.): 67,3 %. 5×5 → 3 tanh → 10 (118 Par.): 73,7 %. 7×7-Softmax (500 Par.): 90,2 %. Die Grenze „≈90 % braucht ≈500 Parameter“ stimmt mit Li et al. (d_int ≈ 750 für 90 % bei zufälligem Unterraum) überein. Strukturierte Unterräume (Mittelung) sind besser als zufällige.
- **Zufällige Projektion + Kopf ist nicht besser** als Mittelwert-Downsampling: k=100 → 89,9 % mit 1.010 Kopfparametern (7×7 erreicht 90,2 % mit 500).
- **[EIGEN] Born-Auslese mit gruppierten Basiszuständen** (8 Qubit, 0 reelle Parameter): 70,97 % Test mit nur einem gelernten 256-Bit-Vorzeichen-Maske plus fester Partition der 256 Ausgänge in 10 Gruppen. Zufällige Maske: 46,9 %, keine Maske: 25,6 %. Pro Bit aber **nicht besser** als quantisierte logistische Regression (siehe §3).
- Maltes Ziel „~100 Parameter für MNIST“ ist realistisch bei ~70–75 %. Für ≥90 % liegt die Untergrenze nach allen Belegen bei einigen hundert; ≥99 % bei ~2.000 (CNN) – dort hilft nur Faltungs-Gewichtsteilung, nicht Quantenphysik an sich [Befundlage, nicht bewiesen].

## 2. Stand der Forschung

### 2.1 Theorie
- **VC-Dimension** linearer Klassifikatoren in d Dimensionen = d+1 [Lehrbuch]. VC-Schranken sind für MNIST praktisch wertlos (Millionen Parameter generalisieren trotzdem); sie sagen nur, dass Parameterzahl keine harte Untergrenze liefert, sondern eine Kapazitätsobergrenze.
- **MDL / Kolmogorov:** Die Komplexität eines Modells ist seine Codelänge (Rissanen, Lehrbuch). Blier & Ollivier (NeurIPS 2018): beste *prequentielle* Codelänge für MNIST 4,10 kbit (Compression-Ratio 0,021) mit einem tiefen CNN bei 99,5 % [Auszug]; das ist 6-mal kürzer als die variationale Codelänge. Das liefert eine ehrliche, parameterunabhängige Vergleichsgröße: **~4 kbit ≈ 500 Byte für Daten + Modell zusammen**. https://arxiv.org/pdf/1802.07044
- **Intrinsische Dimension** (Li, Farkhoor, Liu, Yosinski, ICLR 2018): MLP 784-200-200-10 (199.210 Parameter) erreicht 90 % der Basisleistung mit d_int ≈ 750; Tiefe/Breite ändern das kaum [Auszug]. https://arxiv.org/abs/1804.08838 (nur Suchauszug, via semanticscholar/dblp). Aghajanyan et al. (2021) zeigen das für Sprachmodelle: Hunderte bis Tausende, größere Modelle niedriger [Auszug]. https://arxiv.org/pdf/2012.13255 Wichtig: der Unterraum ist *zufällig*; die Messung ist eine obere Schranke für strukturierte Parametrisierungen.
- **Lottery Ticket** (Frankle & Carbin): LeNet-300-100 auf MNIST mit 1,8 % der Gewichte (≈4.800) erreicht ~97 %; ≈3,6 % verhält sich wie dicht [Auszug]. https://arxiv.org/pdf/1803.03635 Die Topologie (Indexbits!) wird in der Parameterzählung meist nicht mitgezählt.
- **Hashing / Gewichtsteilung:** HashedNets (Chen et al. 2015): bei 1/64-Kompression 1,99 % Fehler auf MNIST (5 Lagen) vs. 2,69 % für gleich große Standardnetze [Auszug]. https://arxiv.org/abs/1504.04788 Die Hashfunktion zählt dort als frei – genau das ist die Zählfrage.
- **Binär/Ternär:** BinaryConnect 98,82 %, BNN 98,60 %, TWN 99,35 % [Auszug]. https://arxiv.org/pdf/1602.02830 , https://arxiv.org/pdf/1605.04711 Das spart Bit pro Gewicht (1–1,6 statt 32), nicht Gewichtsanzahl.
- **Tsetlin-Maschinen:** Fuzzy-Pattern-TM erreicht 98,56 % auf MNIST mit **20 Klauseln** [Auszug]; Literale/Klausel ≈ 40–48 ohne, <5–13 mit Beschränkung/Pruning [Auszug, anderes Papier]. https://arxiv.org/abs/2508.08350 , https://arxiv.org/pdf/2301.08190 Die Beschreibungslänge ist dann ≈ Klauseln × Literale × log₂(784·2) ; das ist die klassische Bit-Zählweise und sollte im Vergleich zu QIRE ausgewertet werden. [?] Genaue Klausel-/Literalzahl des 98,56-%-Modells nicht verifiziert (arXiv gesperrt).

### 2.2 Tensor-Netzwerke
- Stoudenmire & Schwab (NeurIPS 2016): MPS-Klassifikator auf MNIST (Pixel → 2-dim. Merkmalskarte, 784 Plätze), Bonddimension m=10: Fehler ~5 %, m=20: ~2 %, m=120: 0,97 % Test, 0,05 % Train [Auszug]. https://arxiv.org/abs/1605.05775 Parameterzahl ≈ N·d·m²·(Ausgabe an einem Platz): grob 784·2·m² → m=10 ≈ 1,6·10⁵, m=20 ≈ 6,3·10⁵ [EIGEN, Überschlag, nicht aus der Quelle]. Der MPS ist also **nicht** parameterklein; sein Gewinn ist die Gleichung „Bonddimension ↔ Verschränkung“, nicht die absolute Größe. Weitere MPS-Arbeiten: 98 % MNIST, 88 % Fashion [Auszug] https://arxiv.org/pdf/1905.01426 .

### 2.3 Kleinste bekannte MNIST-Modelle
| Modell | Parameter | Test-Acc. | Quelle |
|---|---|---|---|
| Grimov „mnist-minimal-model“ (Conv 8×3×3, SepConv 26, Dense 16/10, Rekursion) | 1.936 | 99,1 % | https://github.com/ruslangrimov/mnist-minimal-model [Auszug] |
| Effizientes CNN | 7.968 | ≥99 % | Medium-Artikel [Auszug] |
| ~3.000 freie Parameter | 3.000 | ~97 % | https://greydanus.github.io/2017/10/30/subspace-nn/ [nur Suchauszug] |
| LeNet-5 | 61.706 | 99,1 % | [Auszug] |
| Fuzzy-Pattern-TM | 20 Klauseln | 98,56 % | arxiv 2508.08350 [Auszug] |
Nichts gefunden: ein dokumentiertes Modell mit ≤100 Parametern und ≥90 %. [?] Das heißt nur, dass ich es nicht fand (Suche eingeschränkt).

### 2.4 Eigene Referenzmessungen (60k Train / 10k Test, ein Seed, Standard-MNIST)
Downsampling = Flächenmittel; Softmax-Regression (sklearn bzw. Adam). Kopf = Gewichte + Bias.
| Modell | trainierte Zahlen | Acc. |
|---|---|---|
| Softmax auf 2×2 | 50 | 46,6 % |
| Softmax auf 3×3 | 100 | 67,3 % |
| Softmax auf 4×4 | 170 | 75,1 % |
| Softmax auf 5×5 | 260 | 86,6 % |
| Softmax auf 7×7 | 500 | 90,2 % |
| Softmax auf 14×14 | 1.970 | 92,6 % |
| Softmax auf 28×28 | 7.850 | 92,7 % |
| Nächster Schwerpunkt, 3×3 / 4×4 / 7×7 / 28×28 | 90 / 160 / 490 / 7.840 gespeichert | 51,1 / 64,1 / 78,1 / 82,0 % |
| Zufallsprojektion k=10/16/25/49/100 + Softmax | 110/170/260/500/1.010 (+Seed) | 56,6/68,9/73,4/84,0/89,9 % |
| PCA k=10/16/25/49 + Softmax | 110/170/260/500 (+PCA-Matrix 784·k) | 80,9/85,7/89,2/91,3 % |
| Gepruntes Softmax (784×10), nnz=100/200/500/1000 | +Indexbits (~13 Bit/Eintrag) | 47,2/71,1/86,6/90,8 % |
| Nur Top-p-Pixel, dichter Kopf, p=10/20/40 | 110/210/410 (+Pixelindizes) | 44,6/63,6/77,9 % |
| 4×4→4 tanh→10 | 118 | 69,7 % |
| 5×5→3 tanh→10 | 118 | 73,7 % |
| 7×7→2 / 3 / 4 / 8 tanh→10 | 130 / 190 / 250 / 490 | 67,6 / 77,0 / 84,5 / 91,7 % |
| 7×7-Softmax (500), nachträglich quantisiert 3 / 7 / 15 Stufen | je 500 (1,6 / 2,8 / 3,9 Bit) | 15,3 / 71,7 / 86,3 % (Float 89,3 %) |
Warnung: Pruning-Zeile unter anfänglich falscher Wichtigkeit (|w| ohne Pixel-Std) lieferte ~12 % (Artefakt); korrigierte Version steht in der Tabelle. Tanh-Netze nur 1.500 Vollbatch-Schritte, vermutlich leicht untertrainiert (14×14→4: 84,8 % < 10×10→4: 85,5 %). Post-hoc-Quantisierung ist schwach; quantisierungsbewusstes Training würde besser sein [?].

## 3. Was das für QIRE bedeutet (ehrlich)
1. **Der lineare Kopf mit 2.570 Parametern ist nicht „99 % der Kosten“ in Bit-Zählung, sondern in Parameterzählung.** Ein reiner 16×16→Softmax hätte 2.570 Parameter und läge bei ~92,6 % (14×14: 92,6 %, 1.970 Par.). Die QIRE-Zelle (~0,94 mit Clifford) ist damit **nur wenige Prozentpunkte** über der Softmax-Basis; als Vergleichsmaß für „Quantenvorteil“ ist 7×7-Softmax (500 Par., 90,2 %) die erste Hürde, nicht „klassisches CNN“.
2. **Ehrliche Zählung (Vorschlag, vier Stufen):** (a) trainierte reelle Zahlen; (b) + feste, aber *ausgesuchte* Teile (Maske 256 Bit, Partition 256·log₂10 ≈ 850 Bit, gewählte Seeds); (c) + Hyperparameter/Architekturbeschreibung; (d) Gesamtbit (z. B. Float32 = 32 Bit, quantisiert = 2–4 Bit). Bei Stufe (b) mitzuzählen ist Pflicht: Die Clifford-Maske und die Blockwahl sind Information. Die Clifford-Zelle hat 0 T-Gatter, aber **nicht 0 Bit**: bei 8 Qubit, L Lagen: 256-Bit-Maske plus Blockstruktur plus Kopf.
3. **Random-Projektion zählt nicht gratis:** In meiner Messung ist sie schlechter als Mittelwertbildung bei gleicher Kopfgröße; „Zufall = frei“ ist deshalb kein Gewinn und Zufall mit gewähltem Seed = Information (log₂ der gesichteten Seeds).
4. **Born-Auslese ohne Kopf [EIGEN-Messung, ein Seed, 8 Qubit, 16×16, L2-normiert, Vorzeichenmaske s∈{±1}²⁵⁶, Walsh-Hadamard H⊗⁸, p=|H(s⊙x)|²; Klasse c = Summe der p-Werte in Gruppe G_c; G aus Klassenmitteln der p-Werte zugewiesen]:**
   - keine Maske 25,6 %; zufällige Maske 46,9 %; Maske per 20.156 Einzelbit-Flips (Hill-Climbing auf 5.000 Trainbildern, ~150 s) **70,97 %** Test (Trainsubset 72,6 %).
   - Mit gelerntem Kopf 10×10+10 auf √(Gruppensummen): 79,2 %.
   - Bit-Bilanz: 256 + ~850 = ~1.100 Bit, keine reelle Zahl. Gegenprobe gleiche Bitzahl: 5×5-Softmax (260 Zahlen) mit 4 Bit ≈ 1.040 Bit ergäbe ~86 % (Float-Wert, Quantisierung nicht gemessen [?]). **Die Gruppen-Born-Auslese ist pro Bit schlechter**, bleibt aber interessant, weil sie keinerlei klassische Multiplikationen im Kopf braucht und die Messung selbst die Klassifikation ist (Q3: Zählstatistik der Shots).
   - Befund passt zu I-036: Maske ist entscheidend (+45 Pkt gegen keine), Struktur der Maske trägt.
5. **Maltes ~100-Parameter-Ziel:** Mit den Mitteln der klassischen Basis bekommt man bei 100–120 Parametern 67–74 %. Wenn die QIRE-Zelle bei ≤100 Parametern (Clifford-fest + Kopf 20–100 Zahlen) >75 % erreichte, wäre das ein belastbarer, vergleichbarer Befund. 90 % unter 500 Parametern widerspräche sowohl der beobachteten Basis als auch d_int ≈ 750 (zufälliger Unterraum) – ausschließbar ist es nicht, weil die Clifford-Struktur ein *strukturierter* Unterraum ist. [Hypothese]
6. **Warnung:** MNIST-Genauigkeit allein sagt wenig; Fashion-MNIST (benötigt ~4 Magiewinkel) ist die härtere Testbasis für „Parametereffizienz ≠ Aufgabeneinfachheit“.

## 4. Neue Ideen / Forschungsfragen
(Neuheitsprüfung: wegen erschöpftem Suchbudget **nur aus Vorwissen und den ~15 Suchen**; „nicht gefunden“ heißt hier „keine Treffer in begrenzter Suche“. Vor Veröffentlichung in Etappe 2 gründlich nachsuchen.)

**4.1 [EIGEN] Bit-Budget-Pareto statt Parameterzahl.** Frage: Welche Gesamtbeschreibungslänge (Bit; Stufe d aus §3) braucht MNIST für 70/80/90/95 %, für klassische Softmax/MLP/TM und für QIRE-Clifford? Neu? Nicht gefunden (nur Blier & Ollivier für Codelänge, Binär/Ternär-Arbeiten für Bit pro Gewicht; kein QML-Vergleich auf Gesamtbitbasis gefunden). Test: bestehende Messungen aus §2.4 mit Bitzahl versehen (Quantisierung 2–8 Bit mit QAT), 1–2 h Skript, CPU. Kontrolle: Tsetlin-Maschine (pip `pyTsetlinMachine`), ternäre Softmax. Quantenstufe Q1 (Simulation). Das liefert Maltes „ehrliche Zählweise“ als Figur der schriftlichen Arbeit.

**4.2 [EIGEN] Feste Clifford-Zelle + winziger Kopf (≤100 Zahlen).** Frage: Kopf 16×10+10=170 aus 16 Gruppenmitteln der 256 Born-Wahrscheinlichkeiten (statt 2.570); oder 10 Skalare+10 Bias (20 Zahlen) auf Gruppensummen. Welche Genauigkeit bei Kopf ∈ {20, 50, 100, 170}? Neu? In der Literatur zu Clifford-/Stabilizer-QML fand ich im Auszug nichts zu „Kopf auf feste Clifford-Zelle“ (Suche nach Clifford-QML-Klassifikatoren wegen Budget nicht ausgeführt [?]). Test: Code aus §2.4 (b.py) plus Kopf-Varianten, bestehende Clifford-Zellen aus I-036 wiederverwenden; <5 min je Variante. Kontrolle: gleiche Kopfgröße auf 4×4/5×5-Downsampling (siehe Tabelle: 118 Par. → 73,7 %). Erfolgskriterium: >75 % bei ≤120 Parametern gesamt. Q1 (exakte Simulation) → Q3 (Shots, Kopf rechnet nur Zählhäufigkeiten).

**4.3 [EIGEN] Gruppierte Basiszustände als gelernte Zuordnung.** Frage: Statt Gruppen aus Klassenmitteln: Zuordnung 256→10 gemeinsam mit der Maske per Bit-/Zuordnungs-Hill-Climbing oder per Gumbel-Softmax lernen (2.560 Logits, Ergebnis diskret). Wie viel gewinnt man über 70,97 %? Nach 150 s Maskensuche stand ich bei 20.156 Flips noch auf dem Anstieg [?] – ist Suchzeit der Flaschenhals oder die Kapazität? Neu? Nicht gefunden: Treffer zu „Messausgänge in Klassengruppen zuordnen“ nur als allgemeine Q-Klassifikator-Idee [?]. Test: b.py auf 10 Min. erweitern (außerhalb der 5-Min-Regel nur in zwei Läufen), 6 Kerne parallel mit Seeds. Kontrolle: zufällige Partition + gelernte Maske; gelernte Partition + zufällige Maske. Q1/Q3.

**4.4 [EIGEN] Geteilte Maske / niedriger Rang.** Frage: Eine einzige Maske (quadratische Form vom Rang r über GF(2)) für alle Blöcke/Lagen statt je Block eine. Passt zu I-036 (Rang trägt 52–78 %, Formen übertragbar). Wie viele Bit (r·16 für r Rang statt 256) für dieselbe Genauigkeit? Neu? Aus I-036 abgeleitet; extern nicht gefunden. Test: Rang-1…8-Formen als Maskenraum in b.py (die Simulation nutzte bisher nur Z-Vorzeichen, nicht CZ-Phasen [?]), 10 Min. Kontrolle: Zufallsform gleichen Rangs. Q1.

**4.5 [EIGEN] Seed-Suche als Kolmogorov-Komprimierung.** Frage: Wie viele Bit Seed-Suche (k Kandidaten, Zufallsmasken oder Zufallsprojektionen, Auswahl per Trainfehler) ersetzt wie viele trainierte Zahlen? Zufallsmaske 46,9 % vs. gesucht 71 % zeigt: Suche ≈ Information. Neu? „Weight agnostic networks“ (Gaier & Ha 2019) und Supermasks (Zhou et al. 2019) liegen nahe [aus Gedächtnis, nicht geprüft, ?]. Test: Zufallsmasken-Auswahl aus 2^10…2^20 Kandidaten, 1–5 Min. Kontrolle: Hill-Climbing mit gleicher Evaluationszahl. Q1.

**4.6 Reihenfolge für Etappe 2:** 4.2 (Gewinn für Maltes Ziel 1), dann 4.1 (Zählweise für die Arbeit), dann 4.4/4.3.

## 5. Offene Fragen für Etappe 2
1. Verifizieren: Grimov-Repo (nur 1.936-Modell dokumentiert), Fuzzy-Pattern-TM (Klauselzahl, Literale), HashedNets/TWN-Zahlen, Greydanus-Unterraumtabelle (3/10/30/100/300/1000-Dim.) – arXiv/GitHub-Blogs per Mirror (Semantic Scholar, OpenAlex) lesen.
2. Gibt es veröffentlichte Modelle ≤1.000 Parameter mit ≥95 %? (Kaggle/GitHub „under 1000“, „MNIST-1D“-Familie, Sparse-Coding.)
3. Welche Parameterzahl hat der Stoudenmire-MPS bei m=10…20 exakt? (Eigene Zählung per tn4ml oder Papier.) [?]
4. Wie sieht der Zusammenhang Maskenrang ↔ Bit ↔ Genauigkeit quantitativ aus (I-036 und diese Born-Auslese)?
5. Hardware (Q3): Wie viele Shots braucht die Klassifikation per Gruppenzählung für 70 % (Binomialfehler der Gruppensummen bei p~1/256)? [EIGEN, rechenbar]
6. Quantisierungsbewusstes Training der Referenzen (die Post-hoc-Quantisierung 15,3 % bei 3 Stufen ist ein Artefakt des Verfahrens).
7. Mehrere Seeds und Konfidenzintervalle für alle Referenzen (hier ein Seed).

## 6. Quellenliste
- Li, Farkhoor, Liu, Yosinski 2018: https://arxiv.org/abs/1804.08838 (Auszug via dblp/semanticscholar/transferlab)
- Aghajanyan, Gupta, Zettlemoyer 2021: https://arxiv.org/pdf/2012.13255 (Auszug)
- Blier & Ollivier 2018: https://arxiv.org/pdf/1802.07044 (Auszug)
- Frankle & Carbin 2019: https://arxiv.org/pdf/1803.03635 (Auszug)
- Chen et al. 2015 (HashedNets): https://arxiv.org/abs/1504.04788 (Auszug)
- Hubara et al. (BNN): https://arxiv.org/pdf/1602.02830 ; Li & Liu (TWN): https://arxiv.org/pdf/1605.04711 (Auszug)
- Stoudenmire & Schwab 2016: https://arxiv.org/abs/1605.05775 ; MPS-Klassifikator (Autoren nicht verifiziert): https://arxiv.org/pdf/1905.01426 (Auszug)
- Fuzzy-Pattern-TM: https://arxiv.org/abs/2508.08350 ; Klauselgröße: https://arxiv.org/pdf/2301.08190 (Auszug)
- Grimov MNIST-Minimal-Model: https://github.com/ruslangrimov/mnist-minimal-model (Seite gelesen, nur ein Modell dokumentiert)
- Greydanus „Training Networks in Random Subspaces“: https://greydanus.github.io/2017/10/30/subspace-nn/ (nur Auszug)
- Gaier & Ha 2019, Zhou et al. 2019 (Supermasks): nur aus Gedächtnis, ungeprüft [?]
- Eigene Rechnungen: `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/{a,b,c2,d}.py`
