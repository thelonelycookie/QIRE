# Literatur-Neuheitsprüfung: strukturierte Mischung + Quadrat (06.10.2026)

**Wichtige Einschränkung (ehrlich):** WebFetch war für arxiv.org, ar5iv, mlr.press, thecvf.com, neurips.cc, openreview und eurecom im Container gesperrt (EGRESS_BLOCKED). **Kein Volltext gelesen.** Alle Aussagen stammen aus Suchtreffer-Auszügen (Abstract/Snippet). Alle „gelesen“-Spalten sind daher „Auszug“. Vor Zitat in der Arbeit: Volltext auf Maltes PC prüfen (Fragen siehe Ende).

## Kurzfassung
1. Strukturierte Mischung (Hadamard·Diagonal) als Ersatz dichter Schichten ist **bekannt und gut belegt** (Fastfood 2013, SORF 2016, ACDC 2016, Deep Fried Convnets 2015, Monarch/Butterfly 2019–22). Das ist kein neuer Befund von uns (I-038 H-C, I-041 H4 bestätigen es nur).
2. **Quadrat nach zufälliger Projektion ist ebenfalls bekannt:** OPU-Zufallsmerkmale (Saade 2016, Ohana 2020) berechnen |Wx|² mit komplexer Gauß-Matrix und verbinden es mit dem Polynomkern; Fastfood/TensorSRHT/Dot-Product-Kernel-Arbeiten behandeln Polynomkerne; Livni 2014 und „Quadratic Networks“ behandeln x² als Aktivierung; Primer (2021) nutzt quadriertes ReLU.
3. „Born-Regel“ in der klassischen Schicht gibt es als Namen/Idee (Text Classification with Born's Rule, NeurIPS 2022; Stoudenmire/Schwab 2016 Tensornetze). Eine Kombination **feste Hadamard-/Fastfood-Mischung + |z|² als Born-Auslese** mit Vergleich gegen |z|/ReLU auf Bildern habe ich in den Treffern **nicht gefunden** (Suchlücke möglich, keine Garantie).
4. Strukturierte Matrizen ersetzen dichte Schichten bei Bildern/Sprache ohne Genauigkeitsverlust (Monarch: 2x schneller bei ViT/GPT-2, Pixelated Butterfly bis 2,3x, Deep Fried: ~3,6x weniger Parameter bei ImageNet, ACDC, Kaleidoscope). Bekannt; die Geschwindigkeitsvorteile gelten für große Breiten und GPU-Training, nicht für unsere N≈1k-Skala.
5. Klassische Surrogate für Quantenmodelle (Landman 2022 RFF; Cerezo-Linie) und Quanten-ELM-Arbeiten stützen unsere Einordnung „Wirkung klassisch“ (I-038).
6. Zielkonflikt Schusszahl/Genauigkeit ist in der QML-Literatur qualitativ bekannt (Shot-frugal Kernel-Klassifikatoren 2022; „Few measurement shots…“ 2024; Shot-Noise-Studien). Eine **Information-pro-Schuss-Messung (I(Klasse;1 Schuss)) gegen exakte Genauigkeit für feste dichte Mischung** (I-040-v2 H4) fand ich nicht; Neuheit plausibel, aber unbewiesen.
7. Quadrat vs. |z|: bei uns Born ≡ |z| (Fashion, I-039-v2) bzw. +0,35 pp (MNIST). Ein expliziter Vergleich Quadrat vs. Betrag nach Hadamard-Mischung fehlte in den Treffern; Primer vergleicht Quadrat mit ReLU/GELU (andere Konstellation).
8. Vertretbar: „bekannte Bausteine, neu kombiniert und ehrlich vermessen“, nicht „neue Methode“.

## Quellentabelle
| Quelle | Jahr | Aussage | gelesen | Nähe zu uns |
|---|---|---|---|---|
| Le/Sarlós/Smola, Fastfood (arXiv 1408.3060, ICML) | 2013 | Hadamard·Diagonal ersetzt Gauß-Matrix; O(n log d) Zeit, O(n) Speicher; gilt für translationsinvariante und Dot-Product-/Polynomkerne; ~100x schneller, 1000x weniger Speicher | Auszug | sehr hoch: unsere Fastfood-Mischung; Polynomkern ≈ Quadrat |
| Yu et al., Orthogonal Random Features / SORF (NeurIPS) | 2016 | Orthogonale statt Gauß-Matrix senkt Fehler; SORF = binäre Diagonale × Walsh-Hadamard, O(d log d) | Auszug | hoch: Haar vs. Fastfood-Gleichwertigkeit (I-038 H-C) |
| Yang et al., Deep Fried Convnets (ICCV) | 2015 | Adaptive-Fastfood ersetzt alle FC-Schichten; ImageNet gleiche Güte mit ~halb so vielen Parametern (45,3 % Fehler, 16,4 M vs. 47,8 %, 58,6 M); MNIST 0,72 % Fehler | Auszug | hoch: strukturierter Ersatz dichter Schicht |
| Moczulski et al., ACDC (ICLR) | 2016 | Diagonal·DCT·Diagonal·DCT⁻¹, O(N) Parameter; mit ReLU verschachtelt in CNNs; Verbindung zu Fourier-Optik | Auszug | hoch (ACDC ist Ihre genannte Familie) |
| Sindhwani et al., Structured Transforms (NeurIPS) | 2015 | Toeplitz-artige Matrizen, >3,5x Kompression bei Sprach-Schlüsselworterkennung fast ohne Verlust | Auszug | mittel (Sprache) |
| Dao et al., Butterfly (ICML) / Kaleidoscope (ICLR) | 2019/2020 | Butterfly-Produkte lernen schnelle Transformationen; K-Matrizen fassen jede strukturierte Matrix; ShuffleNet +5 % ImageNet | Auszug | hoch |
| Dao et al., Monarch (ICML) | 2022 | Produkt zweier Blockdiagonalen; 2x Training bei ViT/GPT-2 ohne Qualitätsverlust; BERT 23 % schneller | Auszug | hoch (Bilder+Sprache, Geschwindigkeit) |
| Chen et al., Pixelated Butterfly (ICLR) | 2022 | feste Block-Butterfly+Low-Rank-Muster; bis 2,3x schneller als dichte ViT/MLP-Mixer/GPT-2 small ohne Genauigkeitsverlust | Auszug | hoch |
| Qiu et al., Compute Better Spent (ICML) | 2024 | strukturierte Matrizen ersetzen dichte Schichten; Skalierung bei gleichem Rechenaufwand | nur Titel/Snippet | mittel |
| Gonon et al., Fast Inference with Kronecker-Sparse Matrices (ICML) | 2025 | Butterfly/Monarch-Muster: Median 1,4x schneller (FP32), nicht immer schneller als dicht | Auszug | mittel: „ab welcher Breite“ |
| Saade et al. / Ohana et al., Optical Processing Units (ICASSP) | 2016/2020 | Zufallsmatrix komplex Gauß, dann elementweises |·|²; Dot-Product-Kern, Bezug zum Polynomkern; Kernel-Ridge, Transfer-Lernen Bilder | Auszug | **sehr hoch**: |Wx|² als Zufallsmerkmal, aber dichte Matrix und optisch, kein Vergleich mit |z| in den Treffern |
| Kar/Karnick; Wacker et al., Improved RF for Dot Product Kernels (JMLR); TensorSketch/TensorSRHT | 2012–24 | Zufallsmerkmale für Polynomkerne, strukturierte Sketches | Auszug | hoch: Quadrat = Grad-2-Polynomkern |
| Livni et al., Computational Efficiency of Training NNs (NeurIPS) | 2014 | Netze mit x² als Aktivierung („Polynomnetze“) sind so ausdrucksstark wie Schwellennetze, effizient lernbar | Auszug | mittel |
| So et al., Primer | 2021 | Quadriertes ReLU im Transformer-FFN: größter Einzelgewinn (~4x weniger Trainingsrechnung bei 500M) | Auszug | hoch: Quadrat vs. ReLU in der Praxis (Sprache) |
| Text Classification with Born's Rule (NeurIPS) | 2022 | Text als Überlagerung von Wörtern; Übergangswahrscheinlichkeit nach Born-Regel; einbettbar in NN | Auszug | hoch (Name), aber andere Konstruktion |
| Stoudenmire/Schwab, Tensor-Netze (NeurIPS) | 2016 | MPS als nichtlineares Kernmodell, MNIST <1 % Fehler | Auszug | mittel (quanteninspiriert, Born-Wahrscheinlichkeit) |
| Landman et al., RFF-Näherung variationaler QML (ICLR) | 2023 | VQCs mit Hamilton-Kodierung klassisch per Random Fourier Features approximierbar | Auszug | hoch: stützt „klassische Wirkung“ |
| Quanten-ELM für Bilder (arXiv 2409.00998) und „Entanglement and Classical Simulability in QELM“ (2509.06873) | 2024/25 | festes Quantenreservoir + klassischer Auslesekopf verbessert Klassifikation; Messwahrscheinlichkeiten als Merkmale | Auszug | mittel |
| Shot-frugal and Robust Quantum Kernel Classifiers (2210.06971) | 2022 | Klassifikation auch bei N=1 Messung zuverlässig, aber Genauigkeitseinbuße bei kleinem N | Auszug | mittel: Schuss-Zielkonflikt |
| Few measurement shots challenge generalization (2411.06600); Generalization Error with Sampling Noise (2410.14654) | 2024 | wenige Schüsse = dominante Fehlerquelle; Generalisierung leidet | Auszug | mittel |

## Was ist neu, was bekannt (ehrlich)
**Bekannt (nicht als eigenen Befund verkaufen):**
- Hadamard·Diagonal-Mischung ersetzt dichte Zufallsmatrix (Fastfood, SORF); strukturierte Schichten ersetzen dichte bei Bildern/Sprache (Deep Fried, Monarch, Butterfly, ACDC).
- Zufallsprojektion + |·|² als Merkmal (OPU-Linie) und Quadrat als Aktivierung (Livni, Primer).
- Polynomkern-Näherung mit strukturierten Zufallsmerkmalen.
- Klassische Surrogate für Quantenmodelle; Schusszahl-Genauigkeit-Abwägung qualitativ.
- „Struktur statt gelernter Parameter“ (I-038 H-A/B/C): das Konzept fester Zufallsmischung ist Standard (Random Kitchen Sinks, ELM).

**Möglicherweise neu (nur als „in den gesichteten Quellen nicht gefunden“ formulieren):**
- Der kontrollierte Vergleich Quadrat gegen |z| und ReLU nach **derselben** festen Hadamard-/Fastfood-Mischung, gepaart, mit TOST-Gleichwertigkeit (I-038 H-G, I-039-v2): Ergebnis „Quadrat ≈ |z|, +0,35 pp MNIST, 0 pp Fashion“.
- Der Befund, dass der Vorsprung gegen Pixel-Softmax aus Mischung + Gruppierung kommt, nicht aus dem Born-Quadrat, bei gleichem Strukturbit-Budget (I-039-v2).
- Skalierung 16→32 (I-041): Struktur-Vorsprung wächst, Born-Vorsprung nicht; Fastfood-Zeitgewinn ab N≈1–2k (konsistent mit allgemeiner Literatur, dort aber GPU/große Modelle).
- Information pro Schuss (I(Klasse;1 Schuss)) und Zielkonflikt „dichte Mischung +3,5 pp exakt, −0,20 Bit pro Schuss“ (I-040-v2 H4); η = I/χ als Vorhersager (H3 ist H0, also kein Befund).
- Alles nur MNIST/Fashion, kleine Skala; Neuheit gilt höchstens als „Messung“, nicht als Methode.

## Vertretbare Formulierung für die Arbeit
- „Wir untersuchen, ob eine feste strukturierte Mischung (Fastfood) mit quadratischer, an die Born-Regel angelehnter Auslese einer dichten Schicht gleichwertig ist. Strukturierte Mischungen und Zufallsmerkmale mit |·|² sind aus der Literatur bekannt (Fastfood, SORF, OPU-Merkmale); unser Beitrag ist ein kontrollierter, vorregistrierter Vergleich an kleinen Bilddatensätzen.“
- „Der Vorteil gegenüber einfachen Pixelmodellen entsteht durch Mischung und Nichtlinearität, nicht spezifisch durch das Quadrat: |z| liefert gleichwertige Ergebnisse (Differenz 0,0 bis 0,35 pp).“
- „Quanten-inspirierter Vorteil“ nur im Sinn „bekannte klassische Technik, quanteninspiriert motiviert“; kein Quantenvorteil, keine Neuheitsbehauptung für Struktur oder Quadrat.
- Für I-040: „Information pro Messung gegen exakte Genauigkeit“ als Beobachtung im Rahmen der bekannten Schuss-Abwägung; Neuheit nicht behaupten, bis Volltextsuche (z. B. „accessible information per shot“ in Quanten-Reservoir-Arbeiten) gemacht ist.
- Nicht vertretbar: „erstmals“, „neue Architektur“, „Born-Regel schlägt klassische Nichtlinearitäten“ (nur +0,35 pp auf MNIST, Fashion gleich, Skalierung wächst nicht).

## Offene Prüfpunkte (Volltext auf Maltes PC)
1. Ohana 2020 und Saade 2016: Vergleichen sie |·|² mit anderen Nichtlinearitäten oder strukturierten Matrizen?
2. Fastfood Abschn. Polynomkerne: wird Grad 2 mit Hadamard-Struktur auf MNIST gemessen?
3. Primer: Ablation Quadrat vs. |x|? Monarch/Pixelated Butterfly: ab welcher Breite schneller als dicht (Zahlen)?
4. Gezielte Suche: „squared activation random features Walsh-Hadamard image classification“, „accessible information per shot classifier“.
