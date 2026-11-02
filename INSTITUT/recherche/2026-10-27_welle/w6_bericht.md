# w6 — Adaptive Messung / frühes Aussteigen (Erkundung 27.10.2026, Seeds 950–952)

## 1. Kurzfassung
- **Ampel: GELB–ROT.** Frühes Aussteigen spart FLOPs, schlägt aber die statische Hülle (gleiche mittlere FLOPs) nur in einem schmalen Fenster und nur um ≤ 0,5 pp.
- MNIST 16×16, MLP 256-128-128-128-128, Kopf nach jeder Lage: Schwelle τ=0,7–0,9 → 95,1–95,2 % bei 74–83 kFLOPs, statisch gleich teuer 94,7–94,8 % (**+0,4 ± 0,2 pp**, n=3). Das ist 2,3× weniger FLOPs als der Vollnetz-Ausgang, bei −0,1 pp (Val-gewählt τ=0,7: 94,9–95,4 %, 73–75 kFLOPs gegen 174 k).
- Fashion: kein Vorteil (τ≤0,7: +0,1 pp, danach −0,2 bis −0,7 pp gegen statisch). Statisch größere Netze (d2_w256: 267 kFLOPs, 85,8 %) übertreffen alles Adaptive.
- Größtes Problem: Der Stamm sättigt ab Lage 2 (Lagen 2–4 ≈ gleich genau), späte Lagen bringen nichts; die Hülle der statischen Netze wächst mit Breite weiter.
- Kein quantenspezifischer Teil: SPRT/Chernoff-Gedanke nur als Analogie, Wirkung klassisch (I-038).

## 2. Idee, Bezug, Literatur
- Quanten-inspiriert: sequentielles Messen, bis Zustände unterscheidbar sind (Schussbudget, SPRT/Chernoff, I-040/I-046). Hier: „Messung“ = Auslesen eines Linearkopfs nach jeder Lage; Abbruch, sobald die Softmax-Marge (max p) ≥ τ.
- Literatur [Gedächtnis]: Wald-SPRT (1945); BranchyNet (Teerapittayanon 2016); Shallow-Deep Networks (Kaya 2019); MSDNet (Huang 2018); Early-exit/Konfidenz-Kaskaden (Viola–Jones 2001). Die Idee ist also bekannt, kein Neuheitsanspruch.
- Abweichung vom echten SPRT: τ ist eine Konfidenzschwelle auf Softmax, kein akkumuliertes Log-Likelihood-Verhältnis über Lagen; Lagen sind nicht unabhängig.

## 3. Aufbau und Ergebnisse
- Daten: MNIST und Fashion-MNIST auf 16×16, 10 000 Train (davon 2 000 Validierung für τ), 5 000 Test (offizieller Testsatz). Seeds 950, 951, 952 (Tuning-Seeds, drei je Datensatz).
- Adaptiv: 4 × Linear(→128)+ReLU, je ein Kopf 128→10; Summe der 4 Kreuzentropien; Adam 2e-3, Cosinus, 25 Epochen, Batch 128. Parameter gesamt 87 592 (gemessen); Lage 4 = 166 kFLOPs Hauptpfad, 174 kFLOPs mit allen Köpfen.
- FLOPs = 2·MAC, enthalten alle bis zum Ausstieg ausgewerteten Köpfe (ehrlich, Köpfe kosten 2,6 kFLOPs). Argmax-/Softmax-Kosten nicht gezählt, ebenso Verzweigung/Latenz (Stapelbetrieb spart real nichts, nur Einzelbeispiel-Betrieb).
- Statische Baselines (nur Endkopf, gleiche Trainingsschritte): 1 Schicht w=16…512, 2 Schichten w=16…256, 3 Schichten w=32…128, plus feste Lagen des Mehrkopf-Stamms. Vergleich: obere Hülle (Genauigkeit gegen FLOPs), linear interpoliert, je Seed. Parameter der Baselines: 4,3 k bis 137 k.
- MNIST (Test-Acc %, Mittel ± sd, n=3):

| τ | Acc adaptiv | mittl. FLOPs | mittl. Lagen | Acc statisch (gl. FLOPs) | Diff pp |
|---|---|---|---|---|---|
| 0 (nur Lage 1) | 94,36±0,34 | 68 096 | 1,00 | 94,60 | −0,24±0,21 |
| 0,7 | 95,11±0,27 | 73 953 | 1,17 | 94,69 | +0,42±0,17 |
| 0,8 | 95,21±0,33 | 77 291 | 1,26 | 94,74 | +0,47±0,22 |
| 0,9 | 95,23±0,34 | 82 908 | 1,42 | 94,82 | +0,41±0,23 |
| 0,98 | 95,23±0,34 | 99 630 | 1,89 | 95,08 | +0,15±0,25 |
| 0,99 | 95,23±0,34 | 108 255 | 2,14 | 95,39 | −0,16±0,13 |
| 1,01 (alle Lagen) | 95,23±0,34 | 174 080 | 4,00 | 95,69 | −0,45±0,25 |

  Statisch: d1_w256 95,15 (136 k), d1_w512 95,82 (272 k), d2_w128 95,21 (101 k), d2_w256 95,92 (267 k), d3_w128 95,59 (134 k).
- Fashion: τ=0,5: 84,43 % @ 74,6 k (+0,14 pp), τ=0,7: 84,62 @ 90,6 k (+0,08); τ≥0,8 negativ (−0,18…−0,68). Statisch d1_w256 84,78 @ 136 k, d2_w256 85,82 @ 267 k. Val-gewählte τ (Toleranz 0,5 pp): 0 / 0,5 / 0 → praktisch kein Ausstieg-Gewinn.
- Die Genauigkeit ist ab τ=0,9 konstant, weil die Marge dann nie mehr früher als nötig aussteigt und Lagen 2–4 die gleiche Genauigkeit liefern; die adaptive Kurve endet also bei 95,23 %, die statische Hülle läuft weiter hoch.

## 4. Deutung, langweiligste Erklärung
- Langweiligste Erklärung: **Konfidenzkaskade sortiert leichte und schwere Beispiele**; leichte (≈ 80 % bei τ=0,7) brauchen nur w=128, schwere bekommen eine zweite Lage. Der Gewinn von 0,4 pp ist dieser klassische Effekt, kein Quanten-Bezug; er ist so groß wie ca. eine Verdopplung von 128→~180 Breite im Hüllenbereich und liegt knapp über der Streuung (sd 0,2, n=3, nicht signifikant testbar).
- Der Vergleich ist für den Adaptiven leicht günstig: Köpfe + Mehrkopf-Training sind „gratis“ Regularisierung (Lage 2: 95,2 % vs. d2_w128 statisch 95,2 % — kein Unterschied), Hülle linear interpoliert (konservativ-konkav → unterschätzt statisch leicht), Baselines nur 25 Epochen wie adaptiv, größere Netze wurden nicht länger trainiert.
- Grenzen: Nur MLP, kein CNN, kein Byte-LM (Budget). Einzelbeispiel-FLOPs; auf Stapel-Hardware kein Zeitgewinn. Schwelle auf der Validierung gewählt (2 000 Bilder), Fashion-Ergebnis zeigt, dass die Val-Regel „−0,5 pp“ dort nichts spart.
- Drei Seeds, Tuning-Seeds, einzelne Architektur: Befund nur Erkundung.

## 5. Vorschlag für einen vorregistrierten Lauf (nur falls Malte will; Ampel gelb)
- H1: Bei gleicher mittlerer FLOPs-Zahl (Val-gewähltes τ, Ziel-FLOPs 75 k/100 k) schlägt Exit-MLP das beste statische Netz (Breite so gewählt, dass FLOPs gleich) um ≥ 0,3 pp auf MNIST und Fashion.
- Arme: Exit-MLP, Exit-MLP + Fastfood/Hadamard-Schichten (Parameter-Vorteil, I-045), statisch dicht, statisch Niedrigrang; je gleiche Parameter UND mittlere FLOPs; CNN-Variante (2 Blöcke, Köpfe nach Block); Kontrolle: zufälliges Aussteigen mit gleichem Anteil (Hat die Konfidenz Information?).
- Stärkere Baseline: längeres Training (≥ 60 Epochen), Wissensdestillation Endlage → frühe Köpfe.
- Mess-Seeds: 10, Effekt 0,3 pp bei sd 0,2 → n=10 reicht für Wilcoxon-Signifikanz bei 0,3/0,2.
- Laufzeit: ~80 s pro Seed und Arm im Container (1 Thread) → ca. 2 h Container (4 Prozesse), Raven 20 min. Realistischer Ausgang: H0 auf Fashion.

## 6. Skripte
- /home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w6/w6_frueh.py (Lauf: `python3 w6_frueh.py mnist 950 951 952`, Datensatzname `fashion_mnist`)
- /home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w6/w6_auswertung.py
- Rohdaten: w6/w6_roh_mnist.json, w6/w6_roh_fashion_mnist.json
- CPU-Verbrauch gesamt ≈ 10 min (1 Thread).
