# F3: Signalverarbeitung/Kompression (Wavelets, Rahmen)

*Welle 1, a2. Nur Abstracts gelesen (arXiv-API bestätigt), kein Volltext. Alle Zahlen ungeprüft.*

## Landkarte
- **Transformcodierung von Netzgewichten:** DCT/Transform + Clustering [arXiv:1805.07258], Transform Quantization im Rate-Distortion-Rahmen [arXiv:2009.01174], FreshNets (DCT + Hash) [arXiv:1506.04449].
- **Wavelets:** lernbare Wavelet-Schichten für RNN [arXiv:2004.09569, schon in Basis], HBLLM 1-Bit mit Haar [arXiv:2512.00862], WaveletGPT [arXiv:2409.12924].
- **Walsh-Hadamard in Netzen:** WHT-Schichten statt 1x1/3x3-Faltung [arXiv:2104.07085, arXiv:2201.02711]; FNet (feste Fourier-Mischung) [arXiv:2105.03824]; WHT statt Attention-Ausgabeprojektion [arXiv:2603.08343].
- **Rotation vor Quantisierung:** QuIP [arXiv:2307.13304]; "Any Orthogonal Transform Will Do" [arXiv:2601.21069].
- **DCT-Parametrisierung:** Transformer in Kosinus-Koeffizientenraum [arXiv:2604.04440], MaCP [arXiv:2410.09103].
- **Rahmen/Frames:** Deep Frame Approximation [arXiv:2103.05804], semi-diskrete Rahmen [arXiv:1504.05487]; Streunetze [arXiv:1203.1513].
- **Neuronale Kompression:** Benchmark [arXiv:2002.03711], Übersicht [arXiv:2503.01505]: gelernte, nichtlineare Transformcodierung.

## Stärkste Baselines/Gegner für QIRE
- Zufällige Orthonormalbasis/DCT-Parametrisierung [arXiv:2604.04440]: gleich gut wie DCT, bei halber Parameterzahl fast dicht.
- FNet und WHT-Projektion: feste Mischung ohne Parameter.
- QuIP-Rotation: Standardgegner bei Quantisierung.

## Wichtige Zahlen (mit geprüft/ungeprüft)
- DCT-Koeffizienten K=mn/2: Verlust 1,604 vs. dicht 1,580 vs. LoRA Rang 48 1,801 (ungeprüft, kleines Modell).
- FNet: 92-97 % BERT/GLUE, 80 % schneller auf GPU (ungeprüft).
- WHT-Schicht [arXiv:2104.07085]: 2,27 M auf 540 K Parameter, 95,21 % auf 92,98 % CIFAR-10 (ungeprüft, Suchauszug).
- Laude: Kompression 7,9-9,3 bei 1-2 % Verlust (ungeprüft).

## Quantenbezug
- **Echt quanten:** nur über Wavelet-MERA [arXiv:1602.01166]; sonst nichts in F3.
- **Analogie:** Hadamard-Gatter = Walsh-Hadamard-Transformation, Born-Quadrat = Betragsquadrat.
- **Hebel für QIRE:** Die Signalverarbeitung erklärt Hadamard vollständig klassisch (Dekorrelation, Wertebereichsreduktion). "Quanten-inspiriert" bleibt Namensgebung plus Bauanleitung (Hadamard-Gatter-Struktur).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3 (Rotation ≈ gleichwertig gegen faire Gegner): **stützt stark**, zwei unabhängige Arbeiten: jede orthogonale Transformation reicht [arXiv:2601.21069], zufällige Basis = DCT [arXiv:2604.04440].
- I-038: stützt.
- Möglicher Widerspruch: keiner gefunden; WHT-Schichten verlieren 2,2 Punkte (Pan) gegen volle Schichten, das ist die Grenze fester Transformationen.

## Runde 2 (Welle 3, c4; nur Abstracts, Zahlen ungeprüft)
- **Feste Mischer weiter belegt:** WHTMix ersetzt Attention durch festen Walsh-Hadamard-Mischer (2,46x weniger Rechnung, ungeprüft) [arXiv:2607.25234]; WaveMix mit Haar-DWT [arXiv:2203.03689].
- **FNet-Ablation:** Hartley war stärkste Alternative zu Fourier, Hadamard/DCT-Zahlen nicht aus Suchauszug zu bekommen; Anhang A.3 von [arXiv:2105.03824] müsste gelesen werden (Lücke bleibt).
- **DCT-Parametrisierung:** LoCA lernt Koeffizienten samt Orten im DCT-Spektrum [arXiv:2502.06820]; frühe DCT-Gewichtskodierung [arXiv:2112.15545]. Kein Vergleich mit Fastfood gefunden.
- **Rahmen:** Überkomplettheitsverhältnis d/m bestimmt lineare Rekonstruktionsrisiken bei straffem Rahmen [arXiv:2609.06862]; Kohärenz-Korrelation mit Generalisierung schon bekannt [arXiv:2103.05804]. Zur Betragsnichtlinearität nichts gefunden.

## Lücken/offene Fragen
- FNet-Ablation (Hadamard vs. Fourier vs. DCT) nicht gelesen; Zahl wäre direkt nützlich.
- Rahmen-Theorie (überkomplett) nicht mit Hadamard-Mischung verbunden; QIRE-Zelle mit Ausgangsdimension > Eingang könnte ein Rahmen sein.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1, a2 (F3, 7 Suchen) | 15 (F3) | hoch, nicht gesättigt |
| Welle 3, c4 Runde 2 (F3, 4 Suchen) | 4 neue (F3) | mittel; FNet-Ablation und Fastfood-Vergleich weiter Lücken |

## Schlüsse und neue Wege (Agent)
1. **Hadamard ist austauschbar:** Belege [arXiv:2601.21069, arXiv:2604.04440, arXiv:2307.13304]. Langweiligste Erklärung: jede gut mischende orthogonale Matrix tut es. Quantenbezug: keiner; der Name "Hadamard-Gatter" ist nur die billigste schnelle Wahl.
2. **Parametrisierungs-Gegner:** feste Basis + K trainierte Koeffizienten (DCT) als zusätzliche Baseline neben Fastfood. Belege [arXiv:2604.04440], [arXiv:1506.04449]. Langweiligste Erklärung: Rangflexibilität, nicht Struktur. Quantenbezug: keiner.
3. **Rahmen-Sicht:** überkomplette Hadamard-Mischung (Expansion plus Betrag) als Frame mit Rahmengrenzen messen [arXiv:2103.05804, arXiv:1504.05487]. Langweiligste Erklärung: Kohärenzmaß sagt nichts über Genauigkeit voraus. Quantenbezug: Analogie (Born-Quadrat als Phasen-Löschung).
4. **(R2) Fastfood-vs-DCT-Koeffizienten selbst messen:** Literatur vergleicht nie beide. Belege: [arXiv:2604.04440, arXiv:2502.06820]. Langweiligste Erklärung: bei gleicher Parameterzahl gleich. Quantenbezug: keiner.

## Runde 3 (Welle 5, e5; 1 Suche)
- **FNet-Ablation (Lücke aus R2) teilweise geschlossen:** laut Suchzusammenfassung zu [arXiv:2105.03824] Anhang: Hadamard etwas schneller als DFT, aber etwa -2 Punkte Genauigkeit; DCT etwa -4; Hartley gleich (GLUE 76,7 vs 76,7). Ungeprüft, Tabellen nicht gelesen. Heißt: Hadamard ist nicht gleichwertig zu Fourier bei Token-Mischung; Z3-Befund (Rotation gleichwertig) betrifft Gewichtsrotation, nicht Token-Mischung.
- Neu: [arXiv:2606.02631] (Haar-Frontend, Energieauswahl).
- Sättigung F3: 1 neue Quelle, Feld weitgehend gesättigt (ca. 10 %).
