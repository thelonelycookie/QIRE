# r1 — Hadamard + Niedrigbit bei GLEICHEN BYTES (Erkundung, Tuning-Seeds 970–972)

## 1. Kurzfassung
- **Ampel: GELB-GRÜN für MNIST, Sprachmodell nicht getestet.**
- Frage: Schlägt das große rotierte Niedrigbit-Modell das kleine fp16-Modell gleicher Bytes? **MNIST-MLP: ja, knapp.** 3 Bit: Hadamard+Clip 98,48 ± 0,17 % vs. klein dicht fp16 98,19 ± 0,07 % (+0,29 pp; gepaart +0,18/+0,48/+0,21). 2 Bit: 98,30 ± 0,11 vs. 97,90 ± 0,11 (+0,40 pp; +0,45/+0,37/+0,36).
- Niedrigrang fp16 gleicher Bytes ≈ klein dicht (98,18 bzw. 97,98 %). Großes fp32-Modell: 98,57 ± 0,10 %.
- **Kontrolle spricht gegen „Hadamard-Struktur“**: zufällige dichte Orthogonale + Clip ist gleich gut (3 Bit 98,51 ± 0,14; 2 Bit 98,21 ± 0,22). Der Gewinn kommt vom Basiswechsel (Mischen/Gauß-Machen), nicht von Hadamard. Hadamard ist nur billiger (n log n Additionen, n/8 B Vorzeichen statt dichtem Q).
- Roh+Clip verliert bei 2 Bit klar (97,10 ± 0,60), ist aber bei 3 Bit (98,27 ± 0,21) schon nahe am kleinen dichten Modell, dabei mit weniger Bytes (siehe 3, Padding-Hinweis).
- Sprachmodell: **nicht gerechnet** (Budget, siehe 4). Ausstehend.

## 2. Idee, Bezug, Literatur
- Wie w4: y = W x = (W Qᵀ)(Q x), Q = H·D/√n; je Zeile Skala per MSE-Gittersuche („Clip“), Fake-Quant. Neu hier: Vergleich bei gleichen Bytes. Quanten-inspiriert (Hadamard-Gatter als Basiswechsel), Wirkung klassisch (I-038).
- Literatur [Gedächtnis]: QuIP/QuIP#, QuaRot, SpinQuant (Rotationen vor Quantisierung); Zufalls-Orthogonale schon bei QuIP als Variante.

## 3. Aufbau und Ergebnisse
- MNIST-MLP, groß 784-512-512-10 (669 k Gewichte, fp32 trainiert, 10 Epochen Adam 2e-3 mit Kosinus, Batch 128, 55 000 Train / 5 000 Val / 10 000 offizieller Test), Quantisierung nachträglich (Code `w4_hadamard_quant.py` wiederverwendet). 3 Seeds, Mittel ± sd, lr nicht getunt (Val nur mitgeführt).
- Bytes = wie w4 (Bits·Gewichte/8 + 2 B Skala + 2 B Bias je Zeile + n/8 B Vorzeichen bei Rotation; Eingang 784 auf 1024 aufgefüllt, dadurch ist der rotierte Arm ≈ 18 % teurer als der rohe). Budget je Bitzahl = Bytes des rotierten Arms; Gegner passend darunter dimensioniert (nicht mehr Bytes als die Quant-Arme).
- Zufalls-Orthogonale: Haar-QR, **mit denselben Bytes angesetzt** (real wäre ein dichtes Q teurer oder per Seed erzeugt; hier Kontrolle für die Struktur, nicht für Kosten).

| Arm | 3 Bit Test % | Bytes | 2 Bit Test % | Bytes |
|---|---|---|---|---|
| groß roh+Clip | 98,27 ± 0,21 | 254 888 | 97,10 ± 0,60 | 171 304 |
| groß Hadamard+Clip | **98,48 ± 0,17** | 301 224 | **98,30 ± 0,11** | 202 280 |
| groß Zufalls-Orth.+Clip (Kontrolle) | 98,51 ± 0,14 | 301 224 | 98,21 ± 0,22 | 202 280 |
| klein dicht fp16 (h=157 / 111) | 98,19 ± 0,07 | 299 262 | 97,90 ± 0,11 | 201 374 |
| Niedrigrang fp16 (Breite 512, Rang 62 / 40) | 98,18 ± 0,06 | 300 188 | 97,98 ± 0,11 | 198 108 |
| groß fp32 (Referenz) | 98,57 ± 0,10 | 2 676 k (fp32) | | |

- Rotationskosten: ca. +14 k (Schicht 1: 1024·10) + 2·512·9 Additionen/Beispiel, einstelliger Prozentanteil der MACs; FLOPs des großen Modells (669 k MACs, Integer) sind **2,5–6× höher** als beim kleinen dichten (h=157: 0,25 M; h=111: 0,10 M). Es ist also eine **Bytes-, keine FLOP-Aussage**.

## 4. Deutung (inkl. langweiligster Erklärung)
- Langweiligste Erklärung: MNIST ist leicht und bei 98,2–98,5 % gesättigt; Abstände von 0,2–0,4 pp liegen bei 2–3 sd, n=3, kein Signifikanztest. Der größere trainierte Ausgangspunkt (98,57 %) gewinnt Breite „gratis“, Rundung kostet bei 3 Bit kaum etwas.
- Klein-dicht-Gegner wurde mit demselben Rezept (10 Ep., lr 2e-3) trainiert, nicht getunt; ein besser getuntes kleines Modell könnte die 0,3 pp teilweise schließen. Kein QAT, kein GPTQ.
- Die Zufallskontrolle zeigt: Struktur „Hadamard“ ist nicht nötig; es zählt irgendeine Rotation. Für die Arbeit also „Rotation vor Quantisierung“ (QuaRot/QuIP-Prinzip), Hadamard nur als billige Umsetzung.
- 3 Bit: roh+Clip (weniger Bytes, 255 k) liegt bei 98,27 und damit gleichauf mit klein dicht (299 k); der Rotationsvorteil bei 3 Bit ist klein. Echter Vorteil nur bei 2 Bit (+1,2 pp gegen roh, +0,4 pp gegen klein dicht).
- **Sprachmodell nicht gerechnet:** ein d=128-Modell mit 3000 Schritten braucht ca. 9 min CPU je Seed (w4: 1200 Schritte = 200–240 s), dazu zwei Gegner-Trainings; das hätte das 30-min-Budget (MNIST verbrauchte 8,3 min, plus Auswertung) um ein Mehrfaches gesprengt. Vorgeschlagen für Lauf (Raven/PC).
- Grenzen: nur MLP, Eingang MNIST (viele tote Pixel, siehe w4), Fake-Quant ohne Integer-Kernel, Einbettung/Ausgabe nicht betroffen (hier alle Linearschichten quantisiert, auch die Ausgabeschicht).

## 5. Vorschlag für einen vorregistrierten Lauf (nur mit Maltes Zustimmung)
- Hypothesen: H1 große rotierte Niedrigbit (2 und 3 Bit) ≥ kleines dichtes fp16 gleicher Bytes + 0,2 pp (MNIST) bzw. ≤ −0,02 bpc (Byte-LM, ≥ 8000 Schritte); H2 Hadamard ≈ Zufalls-Orthogonale (±0,15 pp / ±0,01 bpc, Äquivalenz); H3 Rotation > roh bei 2 Bit.
- Arme: wie r1, plus Fashion-MNIST, plus kleines dichtes Modell mit lr-Tuning (Val), plus Byte-LM d=128 vs. d=64-fp16 gleicher Bytes. Zusatz: Bytes-Rechnung ohne Auffüllung (784 → 800) und zusätzlich FLOP-Spalte.
- 10 Seeds, Effektgröße 0,2 pp / 0,02 bpc. Laufzeit MNIST ≈ 10 × 3 min = 30 min Container; LM ≈ 10 × 25 min → Raven/Maltes PC.

## 6. Skripte
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/r1/r1_gleiche_bytes.py` (Aufruf `OMP_NUM_THREADS=1 python3 r1_gleiche_bytes.py mnist`; importiert `../w4/w4_hadamard_quant.py`)
- Rohzahlen: `.../r1/r1_mnist.json`, Log `.../r1/mnist.log`. CPU ≈ 8,3 min.
