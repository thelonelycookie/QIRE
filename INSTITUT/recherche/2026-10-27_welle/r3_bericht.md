# r3: Kontrolle zu w7, ist der Phasen-Vorsprung nur der Quadrat-Effekt? (Erkundung, kein Befund)

## 1. Kurzfassung
- Ampel: **gelb/rot für "Phase als Idee"**, grün nur für "Quadrat-Aktivierung ist ein billiger Gewinn bei knappem Training".
- MNIST 5k Param.: reell ReLU 93,74 -> reell z² 94,57 (gleiche Param. und FLOPs) -> Nur-Phase 95,44 (2x FLOPs). Etwa die Hälfte des w7-Vorsprungs (+1,7 pp) ist der Quadrat-Effekt, der Rest kommt mit doppelten FLOPs.
- Bei **gleichen FLOPs** (sq_F: Breite 37, 9889 Param.) schlägt reell z² die Nur-Phase-Variante: 95,83 gegen 95,44 (MNIST 5k), 96,71 gegen 96,20 (18k). Fashion: alles innerhalb ~1 sd.
- Zufallsphasen fest + gelernte Beträge = Nur-Phase (95,29 gegen 95,44; 96,50 gegen 96,20): Die gelernte Phase trägt nichts bei.
- Paarsumme (Born-Struktur ohne Phase) = reell z² (94,53 gegen 94,57), kein Zusatzgewinn der Paarung.
- Ausreichend Training (18k): alle Arme liegen innerhalb ~0,5 pp (MNIST) bzw. ~0,5 pp (Fashion), meist im Rauschen.

## 2. Idee, Bezug, Literatur
Kontrolle zu w7 (Nur-Phase-Gewichte e^{iθ}, Born-Auslese |z|²). Born-Auslese = Quadrat-Aktivierung; verwandt: Primer (So et al. 2021, z²-ReLU), Trabelsi 2018 (komplexe Netze) [Gedächtnis]. Quanten-inspiriert, Wirkung klassisch (I-038).

## 3. Aufbau
- Netz wie w7: Eingang 256 (16x16) -> Schicht 1 -> LayerNorm ohne Param. -> Linear 10. Adam, Kosinus, Batch 256, **20 Epochen** (w7: 30; wegen 20-min-Budget gekürzt, daher nicht direkt mit w7 vergleichbar). 10k Training, 5k Test (je Seed andere Teilmenge), Seeds 980-984 (n=5).
- Lernrate je Arm/Budget/Datensatz auf Validierung (Seed 980, 7,5k Training / 2,5k Val) aus {0,01; 0,03; 0,1; 0,3}. Ein Seed zur Wahl (Rauschen möglich).
- Arme: relu, sq (reell z²), sqpaar (z1²+z2², Bias vor Quadrat, = komplex dicht mit reellem Eingang; "kdicht" aus w7 deshalb weggelassen, mathematisch gleich), phase (Betrag fest, Phase gelernt), zufall (Phasen zufällig fest, Beträge r_jk gelernt, 1 Param./Eintrag). Parameter gleich zum Budget (5k/18k) gematcht. **relu_F / sq_F**: Breite so, dass FLOPs gleich Phase (dafür 2x Param.).
- FLOPs/Beispiel: 1 MAC = 2; reell 2nH, Paar/komplex/Phase 4nH (zwei Matmuls), Kopf 2·H·10. Bytes = 4·Param. (float32).

### Test-Genauigkeit % (Mittel ± sd, n=5)
| Daten | Budget | Arm | H | Param | FLOPs | Acc % (n=5) | lr |
|---|---|---|---|---|---|---|---|
| mnist | 5k | relu | 19 | 5083 | 10108 | 93.74 ± 0.21 | 0.1 |
| mnist | 5k | sq | 19 | 5083 | 10108 | 94.57 ± 0.15 | 0.03 |
| mnist | 5k | sqpaar | 10 | 5250 | 10440 | 94.53 ± 0.25 | 0.1 |
| mnist | 5k | phase | 19 | 5084 | 19836 | 95.44 ± 0.19 | 0.3 |
| mnist | 5k | zufall | 19 | 5083 | 19836 | 95.29 ± 0.23 | 0.03 |
| mnist | 5k | relu_F | 37 | 9889 | 19684 | 95.25 ± 0.29 | 0.01 |
| mnist | 5k | sq_F | 37 | 9889 | 19684 | 95.83 ± 0.19 | 0.03 |
| mnist | 18k | relu | 67 | 17899 | 35644 | 96.13 ± 0.22 | 0.01 |
| mnist | 18k | sq | 67 | 17899 | 35644 | 96.38 ± 0.26 | 0.01 |
| mnist | 18k | sqpaar | 34 | 17826 | 35496 | 96.43 ± 0.27 | 0.01 |
| mnist | 18k | phase | 67 | 17900 | 69948 | 96.20 ± 0.23 | 0.3 |
| mnist | 18k | zufall | 67 | 17899 | 69948 | 96.50 ± 0.25 | 0.03 |
| mnist | 18k | relu_F | 131 | 34987 | 69692 | 96.34 ± 0.26 | 0.03 |
| mnist | 18k | sq_F | 131 | 34987 | 69692 | 96.71 ± 0.07 | 0.01 |
| fashion_mnist | 5k | relu | 19 | 5083 | 10108 | 83.63 ± 0.61 | 0.03 |
| fashion_mnist | 5k | sq | 19 | 5083 | 10108 | 84.27 ± 0.51 | 0.01 |
| fashion_mnist | 5k | sqpaar | 10 | 5250 | 10440 | 84.27 ± 0.52 | 0.01 |
| fashion_mnist | 5k | phase | 19 | 5084 | 19836 | 84.65 ± 0.44 | 0.1 |
| fashion_mnist | 5k | zufall | 19 | 5083 | 19836 | 84.92 ± 0.47 | 0.03 |
| fashion_mnist | 5k | relu_F | 37 | 9889 | 19684 | 84.92 ± 0.32 | 0.01 |
| fashion_mnist | 5k | sq_F | 37 | 9889 | 19684 | 84.81 ± 0.20 | 0.03 |
| fashion_mnist | 18k | relu | 67 | 17899 | 35644 | 85.48 ± 0.36 | 0.01 |
| fashion_mnist | 18k | sq | 67 | 17899 | 35644 | 85.61 ± 0.40 | 0.01 |
| fashion_mnist | 18k | sqpaar | 34 | 17826 | 35496 | 85.40 ± 0.33 | 0.01 |
| fashion_mnist | 18k | phase | 67 | 17900 | 69948 | 85.49 ± 0.71 | 0.1 |
| fashion_mnist | 18k | zufall | 67 | 17899 | 69948 | 85.90 ± 0.43 | 0.01 |
| fashion_mnist | 18k | relu_F | 131 | 34987 | 69692 | 85.95 ± 0.27 | 0.01 |
| fashion_mnist | 18k | sq_F | 131 | 34987 | 69692 | 85.86 ± 0.43 | 0.01 |

## 4. Deutung
- Boring (und wahrscheinlich): Der Vorsprung ist (a) Quadrat-Aktivierung gegen ReLU bei knappem Training (+0,8 pp MNIST 5k, +0,6 Fashion 5k, +0,25 bei 18k) plus (b) mehr Rechenaufwand (doppelte FLOPs = breitere effektive Schicht). Eine echte Phasenstruktur-Wirkung gibt es nicht: zufällige feste Phasen mit gelernten Beträgen sind gleich gut, und reell z² mit gleichen FLOPs ist besser oder gleich.
- Nur-Phase spart Parameter (1 statt 2 je Eintrag), nicht FLOPs. Gegen sq bei gleichen Param. gewinnt phase auf MNIST 5k (+0,9 pp, 2x FLOPs), auf 18k nicht (96,20 gegen 96,38).
- Lernraten: phase braucht 0,3 (Gitterrand, MNIST 5k/18k), also Vorbehalt; der Gitterrand benachteiligt phase eher nicht (es gewinnt dort ohnehin nicht). relu_F/sq_F wählten oft 0,01 (unterer Rand), sodass sie womöglich mit kleinerem lr noch etwas gewönnen.
- Grenzen: nur Schicht 1 komplex, 20 Epochen, 10k Training, n=5, lr auf einem Seed gewählt, nur MLP. Unterschiede < 0,5 pp (außer relu gegen Rest bei 5k) liegen im Rauschen (sd 0,2-0,7).

## 5. Vorschlag (nur nach Maltes Zustimmung)
Kein eigener Lauf für Phase nötig. Wenn doch: Hypothese "phase ≥ sq_F bei gleichen FLOPs" ist hier schon widerlegt (MNIST). Besser in den ohnehin geplanten Lauf als Kontrollarm aufnehmen: reell z² (gleiche FLOPs), Zufallsphasen fest. Seeds 10, Effekt 0,5 pp, Container ~20 min. In der Arbeit Phasen/Born nur als Speicher-Aussage (Winkel quantisierbar) führen, nicht als Genauigkeits-Aussage.

## 6. Skripte
`INSTITUT/recherche/2026-10-27_welle/r3/`: `arme.py` (Schichten, Zählung, Training; nutzt `w7/komplex.py`), `lauf.py`, `auswerten.py`, Rohdaten `ergebnis.json`, `tabelle.md`, `lauf.log`. ~19 min CPU, 1 Thread.
