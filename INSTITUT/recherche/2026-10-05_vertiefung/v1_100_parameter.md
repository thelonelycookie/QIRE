# v1 – MNIST mit ≤ ~100 Parametern: Pareto-Kurve Clifford-Zelle gegen klassisch

**ERKUNDUNG, kein Befund, nichts vorregistriert.** Seeds 900–909 (10 Seeds; 910–919 blieben ungenutzt, Rechenzeit/Last). Echtes MNIST 16×16 (`abteilungen.clifford.daten`): je Seed 6.000 Trainbilder (zufällige Teilmenge) / 5.000 offizielle Testbilder. Hyperparameter (l2, τ, Adam-lr) auf 1.200 abgetrennten Trainbildern gewählt, dann Neufit auf allen 6.000; Test nur zur Auswertung. Alle Zahlen: Mittel ± sd über 10 Seeds, Test-Genauigkeit in %.

## 1. Kurzfassung
- Bei ≤ 100 reellen Parametern: beste Zelle = Born-Gruppierung mit nur 10 Bias (Gruppierung daten-gewählt, +850 Bit): **65,0 ± 0,6 %** (Bent 16×16) bzw. 70,5 ± 1,3 % (zufällige ±1-Maske, +256 Bit Maske). Beste klassische Referenz ≤ 100: Fisher-Top-9-Pixel (100 Par., +72 Bit) 60,1 ± 1,5 %; 3×3-Mittelung+Softmax (100) 50,9 ± 1,2 %.
- Bei 110–118 Parametern: Zelle (Gruppensummen + 10×10) bis **76,4 ± 0,9 %** (4×4-Bent), klassisch 5×5→3 tanh→10 (118): **75,1 ± 0,6 %**. Der Vorsprung ist ~1 pp und mit 850 Bit Gruppierung erkauft.
- **Die Bent-Maske ist nicht der Treiber.** Die „Identität“ (quadrierte Pixel, kein Walsh, gleicher Kopf) erreicht 76,3 % (H2) bzw. 63,8 % (H1) – gleich wie 4×4-Bent bzw. höchstens 1 pp darunter. Das Born-Feature trägt den Gewinn gegenüber Mittelwert-Downsampling, nicht die Clifford-Struktur.
- Bei Top-m-Fisher-Köpfen (m = 5/10/20) sind volle 16×16-Bent-Merkmale **schlechter** als „keine Maske“ und als Pixel-Basis (z. B. m=20: 68,3 vs. 73,5 vs. 71,6 Pixel); die kleinen Blöcke (2×2, 4×4) liegen 6–8 pp über Pixel-Fisher.
- **≥ 80 % mit ≤ 100 Parametern ist mit diesen Mitteln nicht erreichbar** (Maximum ≤ 100: 70,5 %; 80,5 % erst bei 210 Parametern). Realistisches vorregistriertes Ziel: ≥ 70 % bei ≤ 110 Parametern bzw. ≥ 75 % bei ≤ 120.
- Suchschaltung (c) weggelassen (siehe §2.4).

## 2. Aufbau und Ergebnisse
### 2.1 Zählweise
Reell = alle trainierten Zahlen (Kopf-Gewichte + Bias). Keine gelernte Standardisierung (feste Skalen: Born-Merkmal 16·√p, Gruppensumme √(10·Σp)); τ (Trainingstemperatur, nur für Bias-Kopf) ist Trainings-Hyperparameter, die Inferenz ist argmax(g + b/τ), also 10 reelle Zahlen. Diskret in Bit getrennt: Gruppierung der 256 Ausgänge auf 10 Klassen = 256·log₂10 ≈ 850 Bit (daten-gewählt: je Ausgang die Klasse mit größtem standardisierten Klassenmittel-Kontrast, nur Trainingsdaten); Top-m-Auswahl = 8 Bit je Merkmal; zufällige Form = k(k−1)/2 Bit; zufällige ±1-Maske = 256 Bit. Bent-Maske = feste Formel (0 Bit; Blockgröße ~2 Bit ignoriert).

### 2.2 Zelle
Merkmale = |Walsh_V(m ⊙ x̂)|² mit `matrix_vm(V, m)` aus `maske_block.py` (Unterraum V = Block aus den niederwertigen Zeilen-/Spaltenbits). Bent-Maske m = (−1)^{Σ r_j c_j} (xᵀy-Paarung Zeilenbit j ↔ Spaltenbit j im Block), „zufallsform“ = zufällige alternierende Form voller Rang k über GF(2) (2×2 identisch zu Bent), „keine“ = m ≡ 1, „zufallszeichen“ = unstrukturierte ±1 (nur 16×16), „identität“ = x²/‖x‖² (kein Walsh). Blöcke 2×2/4×4/8×8/16×16 (k = 2/4/6/8).

### 2.3 Tabelle (Auswahl; alle 79 Zeilen je Seed in `scratchpad/v1/erg_9xx.json`)
| Modell | reell | diskret Bit | Test-Acc. |
|---|---|---|---|
| **≤ 100 reell** | | | |
| 16×16 zufallszeichen, Gruppen+Bias (H1) | 10 | 1106 | 70,5 ± 1,3 |
| 16×16 zufallsform, H1 | 10 | 878 | 66,4 ± 3,4 |
| 16×16 Bent, H1 | 10 | 850 | 65,0 ± 0,6 |
| Identität (x²), H1 | 10 | 850 | 63,8 ± 1,0 |
| 8×8 / 2×2 / 4×4 Bent, H1 | 10 | 850 | 63,8 / 62,7 / 62,3 (±0,8–1,0) |
| 16×16 keine Maske, H1 | 10 | 850 | 60,3 ± 0,8 |
| Index-Gruppen (0 Bit) + Bias, beste Variante (Identität) | 10 | 0 | 16,1 ± 0,6 (fast Zufall: feste Gruppierung ohne Daten trägt nichts; Bent 16×16: 8,6) |
| Fisher-Top-5, 4×4 keine / 2×2 Bent / 16×16 Bent | 60 | 40 | 61,3 / 56,0 / 44,1 |
| Pixel-Fisher-Top-5 | 60 | 40 | 48,5 ± 1,5 |
| Pixel-Fisher-Top-9 | 100 | 72 | 60,1 ± 1,5 |
| 3×3-Mittel + Softmax | 100 | 0 | 50,9 ± 1,2 |
| 2×2-Mittel + Softmax | 50 | 0 | 28,9 ± 1,1 |
| **110–120 reell** | | | |
| 4×4 Bent, Gruppensummen+10×10 (H2) | 110 | 850 | 76,4 ± 0,9 |
| Identität, H2 | 110 | 850 | 76,3 ± 0,6 |
| 2×2 Bent / 16×16 Bent / 16×16 keine, H2 | 110 | 850 | 74,5 / 70,6 / 69,0 |
| Fisher-Top-10: 2×2 keine / 4×4 Bent / 16×16 Bent | 110 | 80 | 70,7 / 68,8 / 55,7 |
| Pixel-Fisher-Top-10 | 110 | 80 | 61,4 ± 1,4 |
| 5×5 → 3 tanh → 10 | 118 | 0 | **75,1 ± 0,6** |
| 4×4 → 4 tanh → 10 | 118 | 0 | 70,2 ± 0,9 |
| **> 120 (Kontext)** | | | |
| Fisher-Top-20: 2×2 keine / 4×4 Bent / 2×2 Bent / 16×16 Bent | 210 | 160 | 80,5 / 79,5 / 79,1 / 68,3 |
| Pixel-Fisher-Top-20 | 210 | 160 | 71,6 ± 1,3 |
| 4×4-Mittel + Softmax | 170 | 0 | 66,1 ± 0,8 |
| 5×5-Mittel + Softmax | 260 | 0 | 78,5 ± 0,4 |
| Zelle voll 256×10: Identität / 16×16 Bent / 16×16 keine | 2570 | 0 | 91,1 / 90,4 / 86,7 |

### 2.4 (c) Suchschaltung weggelassen
`clifford_suche.py` kostet laut I-036 ~380 s je Seed bei 4 freien Kernen mit dem 2.570-Parameter-Kopf als Kriterium. Hier liefen 5 Agenten parallel (Last ~14 auf 4 Kernen, 1 Thread je Prozess nötig), also Vielfaches von 5 min je Seed; zudem optimiert die Suche für den vollen Kopf, nicht für 10–110 Parameter. Offen bleibt, ob eine auf den kleinen Kopf optimierte Suche (z. B. LOO-Kriterium der Gruppensummen) mehr als die Zufalls-Zeichenmaske (70,5 %) bringt.

## 3. Deutung (ehrlich)
- **Pareto-Front ≤ 120 Parameter:** Punkte (Par., Acc.): (10, 70,5 mit 1.106 Bit), (10, 65,0 mit 850 Bit), (60, 61,3), (100, 60,1 Pixel), (110, 76,4 mit 850 Bit), (118, 75,1 ohne Bit). QIRE liegt nur dort auf der Front, wo man Bit nicht zählt (110: +1,3 pp über tanh-Netz, innerhalb ~2 sd). Zählt man Bit mit (850–1.100 Bit ≈ 27–35 float32-Zahlen-Äquivalente), ist 5×5→3→10 bei 118 Zahlen gleichwertig bei weniger Beschreibungslänge; nur im Bereich 10 reelle Parameter ist die Zelle konkurrenzlos (kein klassischer Punkt mit 10 Zahlen erreicht > 29 %), aber mit Bit-Kosten, die ein klassischer 5×5→Softmax-Kopf (260 Zahlen × 4 Bit ≈ 1.040 Bit, 78,5 % in Float) mindestens einholt.
- **Schlägt Bent die Klassik bei gleicher Parameterzahl?** Gegen Mittelwert-Downsampling und Pixel-Fisher: ja (z. B. H1 62–65 % vs. 3×3 50,9 %; Fisher-Top-10 2×2/4×4 68–71 % vs. Pixel 61 %). Gegen das tanh-Netz: nein/gleichauf. **Gegen die Kontrolle ohne Clifford (Identität, x²) und ohne Maske: nicht gewonnen** – die Bent-Maske hilft nur bei großem Block + Gruppenkopf (16×16 H1: Bent 65,0 vs. keine 60,3), schadet bei Fisher-Top-m (volle Bent 16×16 −5 bis −18 pp gegenüber „keine“, weil sie die Energie auf alle 256 Ausgänge breit verteilt und einzelne Ausgänge kaum informativ sind). Langweiligste Erklärung: Born-Quadrierung (Nichtlinearität |·|²) und Gruppenzuordnung bringen den Gewinn gegenüber dem linearen Mittelwert-Downsampling; Walsh/Bent trägt wenig.
- Die zufällige ±1-Maske schlägt Bent bei H1 (70,5 vs. 65,0) – Zufallsmasken wirken hier als Zufalls-Projektion. Bei H1 ist das ein Hinweis, dass nicht „Rang/Bent“ zählt, sondern die Verteilung der Klassenenergie auf Gruppen; mit n = 10 Seeds nicht weiter belastbar. Zufallsform hat deutlich höhere Streuung (±3,4) als Bent (±0,6).
- Warnungen: (i) Mittelwert-Referenzen sind hier schwächer als in Bericht 05 (3×3: 50,9 statt 67,3 %), weil adaptives Pooling 16→3 überlappende Fenster nutzt, nur 6.000 Trainbilder, ungestandardisierte Eingaben und 150 L-BFGS-Schritte; die Referenz ist vermutlich um bis zu ~10 pp zu pessimistisch – das 3×3-Ergebnis nicht als Gegenwert zitieren. Das tanh-Netz 5×5 (75,1 %) ist dagegen stimmig mit 73,7 % aus Bericht 05. (ii) Gruppierung daten-gewählt = Information; ohne Daten (Index) Zufallsniveau. (iii) Bit-Zählung der Bent-Maske vernachlässigt die Wahl der Blockgröße.

## 4. Vorschlag für einen vorregistrierten Lauf
- **Ziel:** ≥ 70 % bei ≤ 110 reellen Parametern UND Gesamtbeschreibung ≤ ~1.200 Bit; nicht ≥ 80 % (erst ≥ 210 Par. erreicht, 80,5 %).
- **Arme (reell ≤ 120):** Zelle H2 (4×4-Bent), Zelle H1 (16×16-Bent, 16×16 zufallszeichen), Zelle Fisher-Top-10 (2×2). **Kontrollen:** Identität x² mit gleichem Kopf (entscheidend, trennt Born von Clifford), „keine Maske“ gleichen Blocks, Zufallsform gleichen Rangs, tanh 5×5→3→10, Pixel-Fisher-Top-m, gestandardisiertes 3×3/4×4-Softmax (Referenzen fair abstimmen!). **Positivkontrolle:** Voll-Kopf 2.570: Identität 91,1 / Bent 90,4 / keine 86,7 muss reproduziert werden (Bent − keine ≈ +3,7 pp).
- **Hypothesen:** H-a: Bent-4×4-H2 ≥ tanh-5×5 + 0,5 pp (Erwartung: Δ ≈ +1,3 pp, sd je Seed ≈ 0,6–0,9 → mit 20 Seeds testbar); H-b: Bent ≥ Identität bei H2 (Erwartung Δ ≈ +0,05 pp → Gleichwertigkeit TOST ±1 pp bei 20 Seeds erreichbar, ±0,5 pp knapp). Bit-Zählung als Zusatzachse mitberichten.
- **Seeds:** ≥ 20 gepaarte (hier 10 gemessen). **Laufzeit:** ca. 3–6 min je Seed und Prozess bei 1 Thread (Last-abhängig); 20 Seeds auf 4–5 Kernen ≈ 25–30 min.

## 5. Offene Fragen
- Gruppierung verbessern (Hill-Climbing/Zuordnungslernen, vgl. Bericht 05 §4.3: 71 % mit gesuchter Maske ohne reelle Parameter) – hier nur Ein-Schritt-Zuordnung.
- Gruppierung als Bit zählen: 850 Bit ersetzen ≈ 100 klassische Zahlen; Pareto auf Bit-Achse neu rechnen mit quantisierten Referenzen.
- Suchschaltung mit kleinem Kopf als Kriterium; 8×8-Block (zeigte 72,1 / 68,1 %, kein Vorteil); Fashion-MNIST nicht gemessen.
- Referenzen mit Standardisierung/mehr Iterationen nachziehen; Paarstatistik (Wilcoxon) nicht berechnet, nur Mittel ± sd.

## 6. Skripte
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v1/` : `kern.py` (Masken, Merkmale, Köpfe), `lauf.py <seeds>` (79 Zeilen je Seed), `erg_900…909.json` (Rohdaten). Start: `cd scratchpad/v1; python lauf.py 900 902 …` (1 Thread/Prozess).
