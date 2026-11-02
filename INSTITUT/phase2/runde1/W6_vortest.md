# W6 Vortest (Phase 2, Runde 1): Orbit-Lloyd, Kerdock-Winkelbuch, E8-Nachbau

Status: **Erkundung, kein Befund.** Skript `w6_vortest.py`, Daten `W6_vortest.json`. Synthetisch i.i.d. N(0,1) (Stand-in für hadamard-rotierte Gewichte), Seeds 0 bis 2, Test auf frischen Daten (Train/Test getrennt). SNR = -10 log10(MSE) bei Varianz 1; Shannon-Grenze = 6,02 dB je Bit.
Echte Gewichtsmatrix: nicht getestet (im Container keine Gewichte, kein torch).

## Bit-Buchhaltung
- Alle Arme: Index-Bits = Rate x Dim, eine globale Skala (Varianz bekannt, 0 Bit/Dim), also Skalenkosten für alle gleich und **Gruppenskalen (g16) nicht eingerechnet**.
- Codebuch-Zahlen (Float16): Lloyd-1D 2^R; Lloyd-2D 2^(2R) x 2 (32 bzw. 128); E8: 2 (Skala, Radius/Regel), Buch aus Regel; Orbit-Lloyd 8D: m x 8 Zahlen (m=64: 512) plus Gruppenregel; Winkelbuch: L Beträge; k-means 4D: 2^(4R) x 4 (Referenz, nicht buchfrei).

## Clifford/E8 (numerisch)
- reale_Stabilisatorvektoren_inkl_Vorzeichen: 480
- bis_auf_Vorzeichen: 240
- Gram_Werte: [-1.0, -0.707107, -0.5, -0.353553, 0.0, 0.353553, 0.5, 0.707107, 1.0]
- Rang: 8
- Stabilisatorvektor_Eintragswerte_|x|: [0.0, 0.35355, 0.5, 0.70711, 1.0]
- Skalarprodukt_Spektrum_(Wert:Anzahl): {'-1.0': 480, '-0.7071': 6720, '-0.5': 26880, '-0.3536': 30720, '0.0': 100800, '0.3536': 30720, '0.5': 26880, '0.7071': 6720, '1.0': 480}
- E8_Wurzelsystem_Gram_{0,+-1/2,+-1}?: False
- Spuren_von_200_Zufallselementen_alle_ganzzahlig: False
- Beispiel_nichtganzzahlige_Spuren: [1.4142, 1.4142, -1.4142, 1.4142, 1.4142]
- Generatoren_permutieren_die_Vektoren: True
- Ordnung_reale_Clifford_C3 (wirkt treu): 5160960
- Ordnung_W(E8): 696729600
- Index_in_W(E8): 135.0

Deutung: (1) 480 reale Stabilisatorvektoren (240 Geraden); Skalarprodukte auch ±1/√2, ±1/(2√2), also NICHT das E8-Wurzelsystem (nur 0, ±1/2, ±1). (2) Die Spuren von Gruppenelementen sind teils irrational (Vielfache von √2). Jede Untergruppe von W(E8) erhält ein Gitter, hat also ganzzahlige Spuren. **Die reale 3-Qubit-Clifford-Gruppe in der Qubit-Darstellung liegt daher NICHT in W(E8)** (hinreichend: irrationale Spur). Nur die Gruppenordnung 5160960 = |W(E8)|/135 stimmt mit einer abstrakten Untergruppe (2^(1+6).O6+(2)) von Index 135 überein, aber als andere 8-dim. Darstellung. Folge: Idee 1 ist nicht automatisch ein E8-Spezialfall; die Hypothese 'Clifford in W(E8)' aus W6_ideen ist in dieser Darstellung widerlegt.

## Gruppen (Untergruppen der realen Clifford-Gruppe, per BFS erzeugt)
- 2:D8^n lokal (H,X,Z je Qubit): Ordnung 128
- 2:Pauli+CZ: Ordnung 64
- 2:Pauli+H^n: Ordnung 64
- 2:Pauli+CZ+H1: Ordnung 256
- 3:D8^n lokal (H,X,Z je Qubit): Ordnung 1024
- 3:Pauli+CZ: Ordnung 1024
- 3:Pauli+H^n: Ordnung 256
- 3:Pauli+CZ+H1: Ordnung 8192

## Ergebnis 2 Bit/Dim (Mittel ± Std über Seeds 0-2)

| Arm | SNR dB | Gewinn vs Lloyd-1D dB | Lücke zu Shannon dB |
|---|---|---|---|
| Orbit-Lloyd 8D [D8^n lokal (H,X,Z je Qubit), |G|=1024, m=64] | 10.539 ± 0.012 | +1.240 ± 0.016 | 1.50 |
| E8 Kugelschnitt (nachgebaut) | 10.424 ± 0.007 | +1.125 ± 0.010 | 1.62 |
| Orbit-Lloyd 8D [Pauli+CZ, |G|=1024, m=64] | 10.398 ± 0.011 | +1.100 ± 0.013 | 1.64 |
| Orbit-Lloyd 8D [Pauli+CZ+H1, |G|=8192, m=8] | 10.190 ± 0.033 | +0.891 ± 0.035 | 1.85 |
| k-means 4D (unstrukturiert) | 10.037 ± 0.007 | +0.738 ± 0.009 | 2.00 |
| Orbit-Lloyd 4D [Pauli+H^n, |G|=64, m=4] | 9.760 ± 0.009 | +0.461 ± 0.009 | 2.28 |
| Lloyd-2D | 9.642 ± 0.036 | +0.344 ± 0.040 | 2.40 |
| Orbit-Lloyd 4D [Pauli+CZ, |G|=64, m=4] | 9.545 ± 0.016 | +0.246 ± 0.020 | 2.50 |
| Orbit-Lloyd 4D [D8^n lokal (H,X,Z je Qubit), |G|=128, m=2] | 9.393 ± 0.005 | +0.095 ± 0.009 | 2.65 |
| Lloyd-1D | 9.299 ± 0.004 | +0.000 ± 0.000 | 2.74 |
| Winkel W=8,L=2 | 9.233 ± 0.003 | -0.066 ± 0.002 | 2.81 |
| Kerdock-Winkelbuch (bester Split) | 9.233 ± 0.003 | -0.066 ± 0.002 | 2.81 |
| Orbit-Lloyd 4D [Pauli+CZ+H1, |G|=256, m=1] | 7.815 ± 0.009 | -1.484 ± 0.012 | 4.23 |
| Winkel W=4,L=4 | 6.987 ± 0.007 | -2.312 ± 0.010 | 5.05 |
| Winkel W=16,L=1 | 6.490 ± 0.004 | -2.808 ± 0.001 | 5.55 |
| Winkel W=2,L=8 | 2.978 ± 0.002 | -6.321 ± 0.005 | 9.06 |

## Ergebnis 3 Bit/Dim (Mittel ± Std über Seeds 0-2)

| Arm | SNR dB | Gewinn vs Lloyd-1D dB | Lücke zu Shannon dB |
|---|---|---|---|
| E8 Kugelschnitt (nachgebaut) | 15.834 ± 0.007 | +1.234 ± 0.011 | 2.23 |
| Orbit-Lloyd 4D [Pauli+H^n, |G|=64, m=64] | 15.776 ± 0.019 | +1.176 ± 0.014 | 2.29 |
| Orbit-Lloyd 4D [D8^n lokal (H,X,Z je Qubit), |G|=128, m=32] | 15.733 ± 0.020 | +1.133 ± 0.025 | 2.33 |
| Orbit-Lloyd 4D [Pauli+CZ, |G|=64, m=64] | 15.655 ± 0.027 | +1.055 ± 0.028 | 2.41 |
| k-means 4D (unstrukturiert) | 15.444 ± 0.004 | +0.844 ± 0.008 | 2.62 |
| Orbit-Lloyd 4D [Pauli+CZ+H1, |G|=256, m=16] | 15.340 ± 0.014 | +0.741 ± 0.017 | 2.72 |
| Lloyd-2D | 15.177 ± 0.023 | +0.577 ± 0.026 | 2.88 |
| Lloyd-1D | 14.600 ± 0.005 | +0.000 ± 0.000 | 3.46 |
| Winkel W=16,L=4 | 14.575 ± 0.013 | -0.025 ± 0.017 | 3.49 |
| Kerdock-Winkelbuch (bester Split) | 14.575 ± 0.013 | -0.025 ± 0.017 | 3.49 |
| Winkel W=8,L=8 | 12.523 ± 0.006 | -2.076 ± 0.010 | 5.54 |
| Winkel W=32,L=2 | 11.176 ± 0.009 | -3.424 ± 0.012 | 6.89 |
| Winkel W=4,L=16 | 7.355 ± 0.003 | -7.245 ± 0.008 | 10.71 |
| Winkel W=64,L=1 | 6.664 ± 0.009 | -7.936 ± 0.010 | 11.40 |
| Winkel W=2,L=32 | 3.004 ± 0.006 | -11.596 ± 0.007 | 15.06 |

## Einordnung (vorläufig, ehrlich)
- **Orbit-Lloyd 8D (Idee 2):** bei 2 Bit mit lokaler D8^3-Gruppe (|G|=1024, 64 gelernte Startvektoren = 512 Zahlen) +1,24 dB gegen Lloyd-1D und +0,12 dB gegen den E8-Nachbau (Std ≤ 0,03 über 3 Seeds). Aber: Dekodierung hier nur per Brute-Force über 65536 Punkte, kein schneller Dekoder gezeigt; die Gruppenwahl zählt (Pauli+CZ nur +1,10, größere Gruppe 8192 nur +0,89, also schlechter als E8). Der Vorsprung gegenüber E8 ist klein und hängt von meinem E8-Nachbau ab (kein E8P, Skala nur auf 20000 Blöcken getunt).
- **4D-Orbit-Lloyd:** bei 3 Bit (4096 Punkte) bis +1,18 dB, bei 2 Bit (256 Punkte) bis +0,46 dB gegen Lloyd-1D; liegt über Lloyd-2D. Der 'unstrukturierte' 4D-k-means (Obergrenze?) ist hier **schwächer** als der Orbit-Lloyd, ein Zeichen, dass k-means mit Zufallsinit, 15 Iterationen und 60000 Trainingspunkten untertrainiert ist; er ist keine verlässliche Obergrenze. Einzelner Orbit mit m=1 (|G|=256, 2 Bit) fällt auf -1,5 dB (lokales Optimum/zu wenig Freiheit).
- **Kerdock-Winkelbuch (Idee 3):** schlechter als Lloyd-1D (-0,07 dB bei 2 Bit, -0,03 dB bei 3 Bit) und weit unter Lloyd-2D (+0,34/+0,58 dB). Wie erwartet kein Gewinn; Kontrollgruppe bestätigt: Idee 3 tot.
- **E8-Nachbau:** +1,13 dB (2 Bit) bzw. +1,23 dB (3 Bit) gegen Lloyd-1D, passt zur Literaturgröße ≈1,2 dB (Nachbau plausibel).
- **Clifford/E8:** die reale 3-Qubit-Clifford-Gruppe (Ordnung 5160960) liegt in der Qubit-Darstellung nicht in W(E8) (irrationale Spuren); die 480 Stabilisatorvektoren (240 Geraden) sind kein E8-Wurzelsystem. Index 135 gilt nur abstrakt.
- Nicht enthalten: echte Gewichte, Gruppenskalen, Dekodierzeit, Ties/Rundung der Bits, fp16-Genauigkeit des Buches. Alles Erkundung, kein Befund.
- E8-Nachbau: Nächster-Punkt-Algorithmus (Conway/Sloane) + Schnitt auf Kugelschalen mit hashbasierter Teilschale, Skala auf Train getunt; Kandidaten-Suche (Schrumpfung + Wurzel-Nachbarn) ist nahezu, nicht garantiert exakt nächstgelegen im Buch (leichte Unterschätzung möglich). Kein E8P (Flip-Aufbau von QuIP#).
- Orbit-Lloyd: Lloyd mit gruppeninvariantem Buch (Schwerpunkte in kanonischer Lage gemittelt), Zufallsinit aus Trainingsvektoren, 12 (8D) bzw. 15 (4D) Iterationen, keine Mehrfach-Neustarts; lokales Optimum möglich.
- 8D-Orbit-Lloyd nur bei 2 Bit (2^24 Punkte bei 3 Bit sind per Brute-Force nicht in der Laufzeit); bei 3 Bit nur 4D.
- Winkelbuch: bester Split auf den Testdaten gewählt (leichter Optimismus, 2-6 Kandidaten).
- Clifford-Wirkung bleibt klassisch (I-038); Formulierung 'quanten-inspiriert'.
