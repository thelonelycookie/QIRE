# v4 — Information pro Schuss eines Born-Klassifikators (MNIST 16x16): Holevo, Chernoff, Schusszahl (ERKUNDUNG, kein Befund)

Alles unten ist **Erkundung** (nichts vorregistriert). Seeds 960–969 (n = 10, Datenaufteilung je Seed: 6.000 Trainings-, 5.000 offizielle Testbilder, `clifford.daten`). Mittel ± sd über Seeds. **Achtung:** die Testbilder sind Teilmengen derselben 10.000 offiziellen Bilder, die sd über Seeds unterschätzt daher die wahre Unsicherheit der Genauigkeiten (nicht der Informationsgrößen, die sind über Seeds fast deterministisch).

## 1. Kurzfassung
- **Holevo bindet nicht, aber die Messbasis lässt viel liegen.** Für das Klassen-Ensemble (Mittelzustände je Klasse, 8 Qubits) ist die Holevo-Größe χ_C = **0,983 ± 0,007 Bit** (Prior-Entropie 3,32 Bit); die Identitäts-Messung erreicht I(Klasse; 1 Schuss) = **0,360 Bit = 37 % von χ_C**, die feste Bent-Zelle 0,307 (31 %), zufällige Clifford 0,162 (16 %), gespeicherte Suchschaltung 0,162 (16 %). Die 8 Bit/Schuss der Rechenbasis sind nie der Engpass (realisiert < 0,4 Bit).
- **Rang der Maske hebt I je Schuss, aber nur innerhalb gleich großer Blöcke und nie über die Identität:** 4x4-Block (k=4): Rang 0/2/4 → 0,253 / 0,297 / 0,307 Bit (10/10 Seeds monoton). 16x16-global (k=8): 0,144 / 0,168 / 0,145 / 0,153 / 0,154 für Rang 0/2/4/6/8 — nicht monoton. Größere Blöcke senken I stark (2x2: 0,34 → 16x16: 0,14).
- **Die Suche optimiert nicht implizit I je Schuss.** Suchschaltung (übertragen) vs. zufällige Clifford: I_C −0,000 ± 0,009 (gleich), aber Kopf-Genauigkeit (exakt) 0,938 vs 0,909 (n=1 Ziehung C_rand0).
- **Schüsse mit linearem Kopf (auf Schuss-Häufigkeiten trainiert):** Identität 0,847 (S=100), 0,896 (S=1000). 90 % absolut erreichen nur 4x4-Bent (S ≈ 550) und die Suchschaltung (S ≈ 580) in allen 10 Seeds; Identität bleibt bei S=1000 knapp darunter (0,896, Grenzwert 0,901). ML-Entscheider mit Klassenmitteln erreicht 90 % nie (Plateau ≈ 0,82).
- **Hardware:** 90 % kosten ≈ 550 Schüsse/Bild → IonQ ≈ 44 USD je Bild, IBM ≈ 0,2–0,4 USD je Bild reine Schusszeit (Annahmen §2.7). Für 95 % des Grenzwerts genügen 130–440 Schüsse. Das Laden der Amplituden (≈ 250–380 CZ, Bericht 08) ist davon unabhängig und bleibt der größere Hardware-Engpass.
- Wirkungsgrad (Fano): η ≈ 0,03 bei 80 %, ≈ 0,004–0,006 bei 90 % — die Schranke ist allerdings sehr locker (§3).

## 2. Was gemessen wurde

### 2.1 Aufbau
Zellen (jeweils Matrix W, Merkmale p = (W x̂)², x̂ = normiertes 16x16-Bild; Code aus `maske_block.matrix_vm`, `maske_eigenschaft.maske_aus_form`):
- (a) **Identität** A_id.
- (b) **Bent-Maske + Block-WHT**: Unterraum V = Pixelblock (2x2, 4x4, 8x8, 16x16 = k 2/4/6/8 Bit), Walsh-Hadamard je Block, Maske m = (−1)^q mit q = *Kreuzkopplung Zeilenbit × Spaltenbit* (B[i,i+k/2]=1). "Rang r" = Rang der alternierenden Form über GF(2); r = k ist bent (flaches Walsh-Spektrum). Gerechnet: 2x2 r∈{0,2}; 4x4 r∈{0,2,4}; 8x8 r∈{0,6}; 16x16 r∈{0,2,4,6,8}. **Hauptzelle (b) = 4x4, Rang 4.** Die Formwahl (Kreuzkopplung) ist eine einzige feste Konvention, andere bent-Formen wurden nicht verglichen.
- (c) **Zufällige Clifford** (4 Lagen Ry(kπ/2) + CNOT-Ring, `unitaer_clifford`), 5 Ziehungen je Seed (Tabellen: Mittel der 5; Kopf nur für C_rand0).
- (d) **Suchschaltung:** NICHT neu gesucht (SA ≈ 400 s je Seed, zu teuer). Verwendet werden die gespeicherten MNIST-Suchcodes von I-036-v3 (Seeds 760–769, `ergebnisse/_zwischen_I-036-v3.json`), Seed 960+i ↔ 760+i, auf meine Aufteilung **übertragen**. Das ist ein fremder Seed: die Schaltung wurde auf anderen Trainingsbildern gewählt (Testbilder nicht zur Wahl benutzt, aber Überschneidung der Trainingsbilder möglich). Ehrlich lesen als "eine gute Clifford-Schaltung aus der Suche", nicht als seed-eigene Suche.
- Kein Kopf-/Hyperparameter auf Testdaten gewählt: Kopf = repo-Kopf (`kopf_logistisch`-Nachbau, L2 = 0,003 fest, L-BFGS; für Schuss-Häufigkeiten max. 100 Iterationen); ML-Glättung ε ∈ {1e-6…1e-2} auf Validierung (4.800/1.200 Teilung des Trainings, Mittel aus S = 10 und 100), fast immer 1e-6.

### 2.2 Informationsgrößen (Bit je Schuss; Test-Klassenmittel; Mittel ± sd, n = 10)
| Zelle | I(Klasse; Schuss) | I(Bild; Schuss) | Chernoff-Info (Mittel Paare) | Chernoff min |
|---|---|---|---|---|
| A Identität | 0,360 ± 0,004 | 1,302 | 0,232 ± 0,002 | 0,052 |
| 2x2 r0 / r2 | 0,338 / 0,355 | 1,162 / 1,254 | 0,214 / 0,225 | 0,051 / 0,053 |
| 4x4 r0 | 0,253 ± 0,003 | 0,884 | 0,147 | 0,048 |
| 4x4 r2 | 0,297 ± 0,003 | 1,014 | 0,173 | 0,048 |
| **4x4 r4 (b)** | **0,307 ± 0,003** | 1,070 | 0,180 ± 0,002 | 0,048 |
| 8x8 r0 / r6 | 0,169 / 0,189 | 0,691 / 0,824 | 0,091 / 0,103 | 0,032 |
| 16x16 r0 | 0,144 | 0,630 | 0,076 | 0,022 |
| 16x16 r2 / r4 | 0,168 / 0,145 | 0,707 / 0,650 | 0,091 / 0,076 | 0,027 / 0,020 |
| 16x16 r6 / r8 | 0,153 / 0,154 | 0,719 / 0,752 | 0,082 / 0,083 | 0,023 / 0,020 |
| **C zufällig (5 Zieh.)** | 0,162 ± 0,007 | 0,743 ± 0,015 | 0,087 ± 0,003 | 0,025 |
| **D Suche (übertragen)** | 0,162 ± 0,011 | 0,727 ± 0,033 | 0,087 ± 0,007 | 0,029 |

Basisunabhängige Obergrenzen (aus Zustandsmatrizen, Trainingsbilder): **Holevo χ_C = 0,983 ± 0,007 Bit** (Klassen-Ensemble), **S(ρ̄_Bild) = 3,539 ± 0,006 Bit** (Bild-Ensemble, reine Zustände), **Quanten-Chernoff** (Mittel über 45 Paare) **0,594 ± 0,004 Bit**, min 0,187. Alle drei sind unitär invariant, also für jede Zelle gleich. Verhältnisse: I_C/χ_C = 37 % (A), 31 % (b), 16 % (c, d); klassische/Quanten-Chernoff = 39 % (A), 30 % (b), 15 % (c, d). Hinweis: Eigenwert-Entropien aus 600 Bildern je Klasse in 256 Dimensionen sind leicht verzerrt (χ_C nach unten/oben unsicher, [?] nicht quantifiziert); I(Klasse;Schuss) Train vs. Test: Differenz ≈ 0,01 Bit (kleine Plug-in-Verzerrung).
Bedeutung: I(Klasse; Schuss) ist exakt aus den Born-Verteilungen der Klassenmittel; "pro Bild gemittelt" ergibt dieselbe Mischverteilung, deshalb die gleiche Zahl; I(Bild; Schuss) ist die Bild-Information (Fano-Kette §3).

### 2.3 Information bei Messung nur von m Qubits (I(Klasse; m Bit), Bit)
Beste m-Teilmenge (auf Training gewählt, auf Test gemessen) / Mittel über alle Teilmengen (Training):
| m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| A Identität | .035/.009 | .064/.028 | .124/.056 | .196/.094 | .265/.145 | .302/.206 | .330/.275 | .360/.350 |
| b 4x4 r4 | .035/.009 | .061/.025 | .105/.049 | .166/.082 | .218/.124 | .253/.174 | .283/.233 | .307/.297 |
| 16x16 r8 | .007/.002 | .022/.006 | .033/.013 | .053/.023 | .074/.039 | .109/.063 | .138/.098 | .154/.149 |
| C zufällig | .004/.001 | .014/.003 | .025/.006 | .038/.014 | .057/.028 | .084/.052 | .118/.093 | .162/.156 |
| D Suche | .007/.001 | .016/.004 | .027/.008 | .042/.016 | .060/.029 | .085/.053 | .124/.092 | .162/.156 |
Ablesung: Mit m = 5 von 8 Qubits (beste Teilmenge) hat Identität/Bent-4x4 bereits 74 % / 71 % der 8-Qubit-Information; bei zufälligen/Such-Schaltungen sind 5 Qubits nur 35 %, da die Information über alle Qubits verteilt ist. Bei Identität liegt die beste Teilmenge in den Bildmitten-Qubits (Zeilen-/Spaltenbits mittlerer Bedeutung, z. B. seed 960: Qubits {0,1,2,5,6}); mehr als m=6 hinzuzunehmen bringt wenig. Für Hardware: Identität und Bent-Block erlauben "weniger Qubits auslesen", Such-/Zufallsschaltungen nicht.

### 2.4 Genauigkeit gegen Schusszahl S (Mittel über 10 Seeds)
**ML-Entscheider** (log-Likelihood der S Zählungen unter Klassen-Born-Verteilungen + Prior; "0 trainierte Parameter" heißt hier nur: kein Kopf — die 10×256 = **2.560 Klassenmittel-Werte sind die Parameter**, so viele wie der lineare Kopf):
| Zelle | S=1 | 3 | 10 | 30 | 100 | 300 | 1000 | exakt (S→∞ mit P_i) |
|---|---|---|---|---|---|---|---|---|
| A Identität | .226 | .393 | .622 | .748 | .804 | .817 | .823 | .800 |
| 2x2 r2 | .233 | .396 | .620 | .750 | .801 | .818 | .825 | .801 |
| 4x4 r0 | .211 | .340 | .543 | .692 | .765 | .788 | .797 | .752 |
| **4x4 r4 (b)** | .230 | .375 | .586 | .733 | .798 | .818 | .826 | .789 |
| 16x16 r0 | .186 | .269 | .415 | .581 | .693 | .739 | .756 | .610 |
| 16x16 r8 | .197 | .282 | .450 | .620 | .737 | .777 | .793 | .691 |
| C zufällig | .195 | .287 | .458 | .630 | .743 | .778 | .792 | .690 |
| D Suche | .194 | .286 | .461 | .651 | .775 | .816 | .832 | .729 |
Streuung über Seeds bei S=100: 0,004–0,013. Der ML-Entscheider erreicht **nie 90 %** (Plateau ≈ 0,82–0,83): Klassenmittel-Modell ist falsch spezifiziert (Bilder einer Klasse streuen stark um den Mittelwert). Auffällig: ML mit exakten P_i (S→∞) ist *schlechter* als mit 1000 Schüssen (z. B. 0,800 vs 0,823). Vermutete Ursache: Log-Likelihood-Strafe für Bins mit q≈ε; Schussrauschen wirkt dort als Glättung [?, nicht geprüft].
Vorhersage per Chernoff/Union-Schranke (Klassenmittel-Welt): 90 % bei S ≈ 50–100, also **viel zu optimistisch** (misst nur das Rauschen *innerhalb* einer Klassenverteilung, nicht die Bild-Streuung) — die Chernoff-Information liefert hier keine brauchbare Schusszahl-Vorhersage.

**Linearer Kopf auf empirischen Häufigkeiten** (Kopf auf Trainingsbildern mit **gleichem S** neu trainiert = "angepasst"; Mittel ± sd):
| Zelle | S=1 | 3 | 10 | 30 | 100 | 300 | 1000 | exakte Merkmale |
|---|---|---|---|---|---|---|---|---|
| A Identität | .207±.006 | .366 | .596 | .754 | .847±.005 | .881 | .895±.003 | .901±.003 |
| 4x4 r0 | .195 | .310 | .500 | .673 | .794 | .853 | .885 | .908 |
| **4x4 r4 (b)** | .206 | .334 | .546 | .720 | .838±.006 | .887 | **.912±.003** | .926±.003 |
| 16x16 r8 | .165 | .239 | .395 | .592 | .761 | .844 | .888 | .914 |
| C zufällig (C_rand0) | .168 | .244 | .409 | .600 | .760 | .839 | .884 | .909±.011 |
| D Suche | .160 | .239 | .410 | .615 | .799 | .878 | **.917±.006** | .938±.006 |
Kopf nur einmal auf exakten Merkmalen trainiert und auf Häufigkeiten angewandt ("unangepasst") ist bei kleinem S deutlich schlechter (Identität S=10: 0,489 vs 0,596; 4x4-r4 S=30: 0,603 vs 0,720; ab S ≈ 1000 gleich). Für die Hardware heißt das: **der Kopf muss mit Schuss-Rauschen der gewählten Schusszahl trainiert werden.**
Kreuzung: bei S ≤ 30 ordnet sich der Kopf wie I(Klasse;Schuss) (Identität > 4x4-r4 > Suche/Zufall); ab S ≈ 300–1000 kehrt sich die Reihenfolge um (Suche 0,917 > 4x4-r4 0,912 > Identität 0,895). Der Kopf kann aus vielen schwach informativen Merkmalen (Suche/Bent) bei genug Schüssen mehr herausholen als aus den konzentrierten Identitäts-Merkmalen.

### 2.5 Schüsse für Zielgenauigkeit (Interpolation in log S; n = Seeds, die das Ziel erreichen)
| Zelle (Kopf angepasst) | 80 % | 90 % | 95 % des Grenzwerts (exakter Kopf) | 99 % des Grenzwerts |
|---|---|---|---|---|
| Identität | 54 | nicht erreicht (0/10) | 132 | 733 (8/10) |
| 4x4 r0 | 112 | nicht erreicht | 426 | nicht erreicht |
| **4x4 r4 (b)** | 68 | **552 (10/10)** | 255 | nicht erreicht (S≤1000) |
| 16x16 r8 | 167 | nicht erreicht | 574 | nicht erreicht |
| C zufällig (C_rand0) | 173 | nicht erreicht | 579 | nicht erreicht |
| D Suche | 104 | **579 (10/10)** | 443 | nicht erreicht |
ML-Entscheider: 80 % bei S = 90 (A), 106 (b), 174 (D), ≥ 570 oder gar nicht bei 4x4 r0, 16x16, zufällig; 90 % nie. S > 1000 wurde nicht simuliert (Grenzwert-Konvergenz: S=1000 ≈ 1–2 pp unter Grenzwert).

### 2.6 Wirkungsgrad η
Realisierte Information I(Klasse; Ĉ_S) (aus Verwechslungsmatrix, Kopf angepasst): Identität 0,14 / 1,24 / 2,34 / 2,59 Bit bei S = 1/10/100/1000; Bent-4x4: 0,13 / 1,06 / 2,29 / 2,69; Suche: 0,05 / 0,57 / 2,07 / 2,71; Maximum log₂10 = 3,32. Pro Schuss sinkt der Ertrag stark: Identität 0,14 → 0,12 → 0,023 → 0,0026 Bit/Schuss.
Fano: nötige Information für Genauigkeit A: I_F(0,8) = 1,96, I_F(0,9) = 2,53 Bit. Gültige Untergrenze S_min = I_F / I(Bild;Schuss) (Kette I(C;X^S) ≤ I(Bild;X^S) ≤ S·I(Bild;X¹)): Identität 1,5 (80 %) / 1,9 (90 %); 4x4-r4 1,8 / 2,4; Suche 2,7 / 3,5. Wirkungsgrad η = S_min / S_nötig (Kopf): Identität 0,028 (80 %); 4x4-r4 0,027 (80 %), 0,004 (90 %); Suche 0,026 / 0,006. **η ≈ 3 % bei 80 %, < 1 % bei 90 %** — Größenordnung wie die 3 % aus Bericht 01 (Digits). Der Holevo-Bit-Wirkungsgrad relativ zu den 8 Bit der Basis: 0,36/8 = 4,5 % (Identität) — nicht aussagekräftig, eher "weniger als 5 % des Kanals".

### 2.7 Hardware-Kosten (Preise Bericht 08; Schusszeit 0,25–0,5 ms ist dort als [?] markiert)
Annahmen: je Bild eine eigene Schaltung (Zustand pro Bild neu geladen), S Schüsse; IBM pay-as-you-go 96 USD/min = 1,6 USD/s; IonQ Forte (Braket) 0,30 USD/Task + 0,08 USD/Schuss. IBM-Zahlen sind **reine Schusszeit**, ohne Job-Overhead, Kompilierung und Datenladen.
| Ziel | S | IonQ je Bild | IonQ 1.000 Bilder | IBM je Bild (0,25–0,5 ms/Schuss) | IBM Open Plan (600 s frei): Bilder |
|---|---|---|---|---|---|
| 80 % (Kopf Identität) | 54 | 4,62 USD | 4.620 USD | 0,022–0,043 USD | 22.000–44.000 |
| 95 % des Grenzwerts (Identität) | 132 | 10,86 USD | 10.860 USD | 0,053–0,106 USD | 9.000–18.000 |
| **90 % (4x4-Bent / Suche)** | 550–580 | 44–47 USD | 44.000–46.700 USD | 0,22–0,46 USD | 2.000–4.400 |
| S=1000 (≈ Grenzwert) | 1000 | 80,30 USD | 80.300 USD | 0,40–0,80 USD | 750–1.500 |
Das ganze MNIST-Testset (10.000 Bilder) bei 90 %: IonQ ≈ 440.000 USD (ausgeschlossen), IBM ≈ 2.200–4.600 USD bezahlt oder ≈ 3–5 Open-Plan-Monate (bei reiner Schusszeit). Sinnvoll sind Demonstrationen mit 100–200 Bildern (Standardfehler eines Anteils bei n=200 ≈ 2,6 pp, Bericht 08). Der Faktor, der alles dominiert, bleibt das Laden (≈ 250–380 CZ je Bild): die Schussrechnung ist eine **Untergrenze der Kosten**.

## 3. Deutung (ehrlich, inkl. langweiligster Erklärung)
- **Steigt I mit dem Rang?** Ja, bei festem Block, in jedem getesteten Block mit k ≤ 6 (10/10 Seeds, aber ΔI nur 0,01–0,05 Bit). Langweilige Erklärung: Rang 0 = reiner WHT; jedes Hinzufügen von Vorzeichenmuster "verschmiert" die WHT-Zeilen wieder in Richtung der lokalisierten Pixelbasis, die laut Tabelle am informativsten ist (Identität 0,360 > alle Block-Zellen). Die Rang-Wirkung wäre also "weniger vom WHT-Schaden", nicht "Gewinn durch Struktur". Bei k = 8 nicht monoton (Rang 2 > 4, 6, 8): ein einfaches Rang-I-Gesetz gilt nicht. Mit der Genauigkeit des Kopfs bei exakten Merkmalen geht I nicht in gleicher Weise: Identität hat das höchste I, aber niedrigere Kopf-Genauigkeit (0,901) als 4x4-Bent (0,926) und Suche (0,938).
- **Optimiert die Suche I pro Schuss?** Nein (I_C gleich zufälliger Clifford, Differenz −0,000 ± 0,009; Identität liegt weit darüber). Sie optimiert die Trennbarkeit für den linearen Kopf bei exakter Auslese. Das ist kein Widerspruch zu "Information pro Schuss": ein linearer Kopf wertet 256 Merkmale kombiniert aus, I(Klasse; 1 Schuss) misst nur die Einzelschuss-Auflösung.
- **Chernoff ≠ Schusszahl.** Die klassische Chernoff-Information der Klassenmittel (0,08–0,23 Bit) sagt S ≈ 50–100 für 90 % voraus (Klassenmittel-Welt); real braucht es ≥ 550 mit Kopf, oder es wird nicht erreicht (ML). Gründe: Streuung innerhalb der Klasse (Bilder sind keine iid-Ziehungen aus dem Klassenmittel).
- **η klein, aber die Fano-Schranke ist locker** (I(Bild;Schuss) zählt auch irrelevante Bild-Details mit); η ≈ 3 % ist keine Aussage über Verbesserungspotenzial auf 100 %. Ein ehrlicheres Maß wäre I(Klasse; X^S) direkt — nicht berechnet.
- **Grenzen der Erkundung:** (i) Suchschaltung nur übertragen, nicht seed-eigen; (ii) Bent-Form nur eine Konvention (Kreuzkopplung); (iii) Kopf für S ≥ 100 mit max 100 Iterationen (nicht konvergiert → eventuell leicht unterschätzt); (iv) Kopf nur für 6 Zellen, C_rand nur 1 Ziehung (sd 0,011 zwischen Seeds); (v) ML-Entscheider: Glättungswahl fast immer minimal (1e-6), Effekt ε unklar; (vi) die S-Kurven nutzen je Zelle/Seed eine Realisierung des Schussrauschens (Zufallsfehler zusätzlich ≈ ±0,6 pp bei 5.000 Bildern); (vii) die Hardware-Rechnung nimmt ideale, rauschfreie Schüsse; reale Gatter-/Ausleseraten (~2e-3 je CZ, 1–2 % Auslese) kommen oben drauf.
- Parameterzählung: linearer Kopf 2.570 reelle Zahlen; ML: 2.560 (Klassenmittel); Zelle (b): 0 reelle, diskret ≈ 1 Wahl Blockgröße + 4–8 Bit Form (hier fest); Suche: 4×8 Codes à 2 Bit = 64 Bit.

## 4. Vorschlag für einen vorregistrierten Lauf (v4-Hauptlauf)
- **Hypothesen:** H1 (Schusseffizienz): Bei S = 30 liegt der angepasste Kopf auf Identität über 4x4-Bent (Differenz ≈ +3,4 pp, sd ≈ 1 pp, paarweise); bei S = 1000 liegt die Suchschaltung (seed-eigene Suche!) über Identität (≈ +2,2 pp). → Kreuzung einseitig testen. H2: Für 90 % genügt ein Kopf mit S ≤ 600 auf 4x4-Bent (Erwartung 552 ± ?). H3 (Rang): ΔI(Rang 4 − Rang 0) bei 4x4 ≥ 0,03 Bit (beobachtet 0,054; Streuung 0,001, praktisch deterministisch, Effekt klar).
- **Arme:** Identität; 4x4-Bent r=4; 4x4-WHT r=0; Suche (seed-eigen, 4 Lagen); Zufalls-Clifford (≥ 5 Ziehungen/Seed); eine zusätzliche bent-Form (Zeilen-Zeilen-Kopplung) zur Prüfung der Formkonvention.
- **Kontrollen/Positivkontrolle:** Positivkontrolle: Kopf bei S → ∞ reproduziert bekannte Genauigkeiten (Identität 0,90, Suche ≈ 0,94, aus I-036); Kontrolle "unangepasster Kopf" zeigt den erwarteten Einbruch bei kleinem S. Permutations-Labels: I → 0 und Genauigkeit ≈ 0,1.
- **Seeds/Statistik:** 20 Seeds (Aufteilung wie I-036); Verwende größere Testmenge (alle 10.000 offiziellen Bilder, mehrere Schuss-Realisierungen, mind. 3 je Bild) um die Pseudoreplikation der Testbilder zu mindern. Erwartete Effektgröße ≈ 2–4 pp, sd über Seeds ≈ 0,3–1 pp (C_rand ≈ 1,1–1,6 pp zwischen Ziehungen). TOST ±0,5 pp (Gleichwertigkeit Suche vs. Zufall bei I) wäre bei n = 20 erreichbar (sd ≈ 0,9 pp bei D−C_rand: Wert 0,009 Bit, TOST-Grenze in Bit noch festzulegen).
- **Laufzeit:** wie hier ≈ 15 s je (Zelle, Seed) ohne Kopf, ≈ 2 min mit Kopf-Training für 7 Schusszahlen; 20 Seeds × 6 Zellen ≈ 40 min auf 4–5 Kernen; Suchschaltung je Seed ≈ 400 s zusätzlich (4 Prozesse: ≈ 35 min).

## 5. Offene Fragen
1. I(Klasse; X^S) direkt (statt Fano-Kette) schätzen: Wie nahe an log₂10 bei S = 1000? Plug-in-Verzerrung ist bei 10 Klassen × S Zählungen groß.
2. Warum ist ML mit exakten P_i schlechter als ML mit 1000 Schüssen? Test: ε-Abhängigkeit, Log-Likelihood mit Kullback-Leibler statt Zählungen.
3. Optimiert eine Suche, die I(Klasse; Schuss) oder S=30-Genauigkeit als Ziel nimmt, andere Schaltungen als die Exakt-Auslese-Suche? (Idee B aus Bericht 01.) Erwartung nach diesen Daten: Ja, Richtung Identität/Bent-4x4.
4. Formkonvention der bent-Maske (Kreuz- vs. Zeilenkopplung) und der Block-Ausrichtung (hier 4x4 Pixelblöcke): Effekt unbekannt.
5. Hardware: reale Schusszeit je Schuss auf Heron/Nighthawk, Mindestdauer je Job, ob für 1 Bild = 1 Job oder Batch (Bericht 08 hat dazu nur [?]).
6. Sequentielle Schusszahl (SPRT): leichte Bilder könnten bei S ≈ 10 entschieden werden; Mittel-S bei 90 % könnte deutlich unter 550 liegen (nicht gerechnet).

## 6. Skripte
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v4/v4.py` (Lauf, je Seed `seed_96x.json`), `.../v4/agg.py` (Auswertung), `.../v4/run.log`. Quellen: Berichte `01_holevo.md`, `08_hardware.md` (Preise, Schusszeit [?], Ladekosten); Code `abteilungen/clifford.py`, `clifford_suche.py`, `maske_block.py`, `maske_eigenschaft.py`; gespeicherte Suchcodes `ergebnisse/_zwischen_I-036-v3.json`. Literatur nur aus Bericht 01 (Holevo, Fano, Audenaert Quanten-Chernoff, Hoeffding-Schusszahl [Gedächtnis]).
