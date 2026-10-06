# v6 — Dequantisierung: Wie „quanten“ ist die Magie? (ERKUNDUNG, kein Befund)

Stand 06.10.2026. Seeds 980–984 (5 Teilstichproben bzw. Gerüste). Skripte: Scratchpad `.../scratchpad/v6/`. Nichts hier ist vorregistriert. Literatur nur aus Etappe-1-Berichten 02/03/04 bzw. [Gedächtnis].

## 1. Kurzfassung
- **Surrogat exakt, aber nicht dünn.** Mit k freien Ry-Winkeln auf einem Clifford-Gerüst ist jeder Born-Wert ein trig. Polynom mit ≤ 3^k Frequenzen; gemessen (n=8, zufällige Clifford-Gerüste, 32 Bilder) sind **praktisch alle 3^k−1 nichtkonstanten Terme belegt**: k=1: 2/2, k=2: 7,2–8 von 8, k=4: 73–80 von 80, k=8: 2.600–4.400 von 6.560. Für 99 % der θ-Varianz braucht man k=4: ~70 Terme (87 %), k=8: ~2.000–3.000 (30–47 %). Die Energie ist kaum konzentriert; „99 % Energie“ = erst 10 % RMS-Fehler (Fehler ≈ √(Restenergie)).
- Exakte kompakte Form: **2^k Clifford-Zweige** (Ry(t)=cos(t/2)·I+sin(t/2)·Ry(π)); numerisch bis 4·10⁻¹⁶ bestätigt (n=8, 16).
- **Kippunkt Kosten** (Zählmodell, Abschn. 2.2): Exaktes Surrogat mit voller 2^n-Auslese lohnt nur, wenn die Zweige im Voraus berechnet sind (viele Winkelsätze je Bild), und dann bis k≈4 (n=8), 5 (n=16), 6 (n=32) — also ~log₂(L·n). Frisch gerechnet nie billiger als eine Clifford-bewusste Zustandsvektor-Simulation. Bei Abtast-/Einzelamplituden-Auslese (n groß): bis k≈n−3.
- **Geometrische Differenz (MNIST 16×16, 1.000 Bilder, 5 Teilstichproben):** g(quadratisch‖Q) ≈ 3,8 (λ=10⁻³, Tr K=N, √N=31,6), **für alle Schaltungen gleich** (Identität 3,90; Suche 3,80±0,29; Suche+k=4 3,87±0,12; generisch 32 Winkel 3,66±0,12; gepaarte Differenz k4−Suche +0,07±0,18). RBF 2,7–3,2, linear 44–76 (Rang-Defizit). **Magie ändert g nicht messbar.**
- Ursache ist exakt: jeder Born-Merkmalsvektor ist ein 256-dim Unterraum der quadratischen Formen in x/|x| (Dim. 32.896); K_Q ⪯ c·K_quad **für jedes Winkelbudget k**. Also ist der QIRE-Kern im Sinne von Huang et al. dequantisiert; der Magieanteil ändert nur *welchen* 256-dim Unterraum.
- Genauigkeit n=1.000: klassische Kernel-Ridge RBF 94,4 ± 0,7 %, quadratisch 93,7 ± 0,8 %, linear 83,7 %; QIRE-Kernel-Ridge 80–90 %, QIRE-Kopf (logistisch) 82–91 %. Klassisch holt ein und überholt.
- Was bleibt bei n=8: eine **parameter-/struktureffiziente klassische Merkmalsabbildung** (256 feste quadratische Formen mit Struktur O(N log N)), kein Rechenvorteil.

## 2. Was gemessen / hergeleitet wurde

### 2.1 Surrogat-Struktur (Herleitung)
Ry(θ)=cos(θ/2)I+sin(θ/2)J, J=Ry(π) (Pauli, Clifford). Mit k freien Winkeln: U(θ)=Σ_{s∈{0,1}^k} w_s(θ)·C_s, 2^k Cliffords. Amplitude a_b=Σ_s w_s(u_{b,s}·x̂), Born p_b=a_b². Produkte w_s w_s' liefern Basis (1, cosθ_j, sinθ_j)^k, also 3^k Terme; jeder Koeffizient ist quadratische Form in x̂ vom Rang ≤ 2^k. Daten: Surrogat in x̂ ist **polynomiell vom Grad 2** (nicht Fourier). Datenunabhängige Speicherung 3^k·N·N(N+1)/2 Zahlen (n=8, k=0: 8,4·10⁶; k=4: 6,8·10⁸; k=8: 5,5·10¹⁰): schon bei k=0 größer als die Schaltung (32 Winkel) — die kompakte Zweigform ist die sinnvolle.
Korollar: Die **Fidelity-Kern** |⟨x̂|x̂'⟩|² ist schaltungsunabhängig = quadratischer Kern; interessant ist nur der Born-Merkmals-Kern (projizierter Kern).

### 2.2 Termzahl gemessen (n=8, L=4, 5 Gerüste, je 32 MNIST-Bilder; Erkundung)
Aufbau: zufälliges Clifford-Gerüst (Codes 0..3), k Winkel auf 3-Punkt-Gitter je Winkel (exakt für Grad 1), 3^k Simulationen, Koeffizienten per Basiswechsel, Parseval-Energie (Gewichte 1, ½, ½) ohne konstanten Term. Mittel ± sd über Gerüste; Positionen „spread“ (zufällig), „layer0“, „qubit0“.

| k | 3^k−1 | belegt (spread) | n(90 %) | n(99 %) | eff. Rang |
|---|---|---|---|---|---|
| 0 | 0 | – | – | – | – |
| 1 | 2 | 2 | 2 | 2 | 1,95 |
| 2 | 8 | 7,2±1,8 | 6,0±2,2 | 7,2±1,8 | 5,9 |
| 4 | 80 | 80±0 | 46,8±9,1 | 72,4±5,9 | 35±11 |
| 8 | 6.560 | 2.643±878 | 862±347 | 1.950±653 | 157±84 |
| 8 (layer0) | 6.560 | 4.389±2.092 | 1.256±536 | 3.054±1.324 | 197±164 |

n=16 (nur 3 Gerüste, 4 Bilder, k≤4, Stichprobe): k=1: 2/2; k=2: 8 belegt, n99=5–8; k=4: 80 belegt, n99=42–72, n90=24–47. Kein Hinweis, dass die Termzahl mit n schrumpft oder wächst; k bestimmt sie. Vollständigkeit der Zweige: max. Fehler 10⁻¹⁶ (n=8), 10⁻¹⁷ (n=16).
Einschränkung: Gerüste zufällig, nicht gesucht; Bilder wenige; k=16, 32 nur gezählt (3^16=4,3·10⁷; 3^32=1,9·10¹⁵).

### 2.3 Kostenmodell (Flops je Bild und Winkelsatz, L=4; eigene Annahmen)
SV_gen=3LnN; SV_clif=(Ln+2k)N; Surrogat amortisiert (Zweige vorberechnet) 2^(k+1)N; Surrogat frisch 2^k·LnN+2^(k+1)N.
| n | Surrogat amortisiert noch ≤ SV_clif bis k = | ≤ SV_gen bis k = | frisch ≤ SV_gen bis k = | Abtast-Auslese (2^k n² ≤ L n 2^n) bis k = |
|---|---|---|---|---|
| 8 | 4 | 5 | 1 | 7 |
| 16 | 5 | 6 | 1 | 14 |
| 32 | 6 | 7 | 1 | 29 |
Aussage: bei voller 2^n-Auslese kippt es bei k ≈ log₂(Ln) (klein, wächst nur logarithmisch mit n). Bei n=32 ist die volle Auslese (4,3·10⁹ Werte) ohnehin nicht praktikabel; dort zählt die Abtast-Zeile: das Surrogat ist bis k≈n−3 billiger als Zustandsvektor — aber das ist Stabilizer-Zerlegung (Bravyi–Gosset-Typ, Bericht 04), keine neue Idee; die Auslese per Kopf mit 2^n Merkmalen bleibt das Problem.
Beachte: „Ab welchem k übersteigt das Surrogat die Zustandsvektorkosten“ hängt stark davon ab, was man zählt; die Antwort „k≈4–6 amortisiert, k=1 frisch, k≈n−3 bei Abtastung“ ist modellabhängig.

### 2.4 Geometrische Differenz
Aufbau: MNIST 16×16, je Seed 1.000 Trainingsbilder (Zufallsteilmenge), Test 2.000 offizielle Testbilder; Eingabe x̂=x/|x|. Q-Kern K_Q=⟨p(x),p(x')⟩ (Born-Merkmale, 256-dim). Klassisch: linear x̂·x̂'; quadratisch (x̂·x̂')²; RBF mit γ=1/median(d²) („med“) und 4× enger („eng“). Alle Kerne auf Tr K=N normiert. **Definition (eigene, [Gedächtnis]-Form nach Huang et al. 2021, deren regularisierte Formel ich nicht sicher kenne):** g_λ=√‖√K_Q (K_C+λI)^{-1} √K_Q‖_∞, ungeregelt λ→0 = Huangs g. Referenz „SELBST“ (K_C=K_Q) = 1,0 für alle λ. Obere Grenze √N≈31,6 bei vollem Rang. Schaltungen je Seed: Identität; Walsh voll (k=8, ohne Maske); **Bent-Maske** (k=8, Form Rang 8, Walsh über m⊙x); Bent 4×4-Block (Rang 4); zufälliges Clifford; **Clifford-Suche** (SA 500 Schritte+Politur, LOO-Ridge auf Trainingsbildern, ~35 % der Voll-Budgets; LOO 87,5±1,1 % gegen Zufallsmedian 83,8±1,2 %); **Suche+k=4** (4 Zufallspositionen, Koordinatensuche auf π/8-Gitter, LOO 89,2±0,8 %); generisch (32 zufällige Winkel, nur Kontrolle).

g (λ=10⁻³ / λ=0,1), Mittel ± sd über 5 Teilstichproben:
| Schaltung | quadratisch | RBF med | RBF eng | linear |
|---|---|---|---|---|
| Identität | 3,90±0,03 / 3,18 | 2,89 / 2,31 | 5,04 / 4,91 | 44,2 / 4,85 |
| Walsh voll | 2,29±0,01 / 2,16 | 2,33 / 2,20 | 6,41 / 6,32 | 72,2 / 7,76 |
| Bent-Maske k=8 | 3,89±0,03 / 3,20 | 3,09 / 2,61 | 5,84 / 5,72 | 75,7 / 8,10 |
| Bent 4×4 | 4,10±0,02 / 3,37 | 3,23 / 2,69 | 5,68 / 5,55 | 70,3 / 7,56 |
| zufälliges Clifford | 3,77±0,18 / 3,12 | 2,83 / 2,43 | 5,68 / 5,56 | 66,9 / 7,21 |
| Clifford-Suche | 3,80±0,29 / 3,14 | 2,81±0,10 / 2,42 | 5,73 / 5,61 | 66,5 / 7,14 |
| Suche + k=4 | 3,87±0,12 / 3,19 | 2,74±0,05 / 2,34 | 5,67 / 5,55 | 63,1 / 6,75 |
| generisch k=32 | 3,66±0,12 / 3,03 | 2,66±0,19 / 2,29 | 5,61 / 5,50 | 61,6±7,0 / 6,61 |
Gepaarte Differenzen g_quad (λ=10⁻³): k4−Suche +0,07±0,18; generisch−Suche −0,14±0,34; Suche−zufällig +0,03±0,32. λ=10⁻⁴ ändert g_quad kaum (3,7–4,1: Sättigung = endlicher Grenzwert, K_Q⪯cK_quad).

Zweite Messung (Funktionsebene): 20 Zufallsfunktionen aus dem RKHS von K_Q (Gauß-Gewichte auf den 256 Born-Merkmalen), Training 800, Test 200 Bilder, λ des klassischen Kerns **orakelhaft** auf dem Test gewählt (zugunsten klassisch); normierter MSE:
| Schaltung | quadr. | RBF med | RBF eng | linear |
|---|---|---|---|---|
| Identität | 0,043 | 0,048 | 0,179 | 0,067 |
| Suche | 0,089 | 0,102 | 0,217 | 0,298 |
| Suche+k=4 | 0,087 | 0,099 | 0,209 | 0,288 |
| Bent8 | 0,090 | 0,105 | 0,226 | 0,322 |
| generisch 32 | 0,074 | 0,085 | 0,194 | 0,260 |
(Mittel über 5 Seeds; Streuung nicht ausgewiesen.)

Klassifikation (Test 2.000, Mittel±sd): Kernel-Ridge klassisch: linear 83,7±1,4; quadratisch 93,7±0,8; RBF med 94,4±0,7; RBF eng 94,6±0,7. QIRE: logistischer Kopf auf Born-Merkmalen (L2=10⁻²): Identität 86,1±1,2; Suche 88,7±0,3; Suche+k4 89,9±0,5; generisch 91,0±1,0; Kernel-Ridge mit K_Q: 81,2 / 87,6 / 88,8 / 89,9. (Bei n=1.000 Training; keine Hyperparameterwahl auf dem Test außer dem orakelhaften λ im RKHS-Test, der nur die klassische Seite begünstigt.)

Parameter (ehrlich): QIRE-Kopf 2.570 reelle + Clifford-Wahl 32 Bit (SA ist diskret, effektiv 1 Bit je Gatter) + k=4 Winkel (4×4 Bit auf π/8-Gitter, zusätzlich 4 Positionen ≈ 20 Bit). Klassisch quadratisch: Dual-Kernel-Ridge hat 1.000 Koeffizienten (daten-abhängig), primal 32.896 Merkmale; RBF 1.000. Zeigt: der QIRE-Weg spart Parameter, nicht Rechnung.

## 3. Deutung (ehrlich)
1. **Dequantisierung ist exakt, nicht nur empirisch.** Weil ψ=Ux̂ und p_b=(u_b·x̂)², lebt jeder Born-Merkmalsvektor im Raum der quadratischen Formen. Das gilt für jede Ry/CNOT-Schaltung, jedes k. g(quadratisch‖Q) ist daher durch eine Konstante beschränkt (≈3,7–4,1, ≪√N), und Magie ändert sie nicht (sd 0,1–0,3 gegen Effekt ≤0,1).
2. **Langweiligste Erklärung für „g quadr. ≈ g Identität“:** K_Q ist ein Kern über Quadratmerkmalen; der Unterschied der Schaltungen ist eine Rotation innerhalb davon. g misst die Größe des Merkmalsraums, nicht Quantenartigkeit. Auch der RBF-Kern (n=1.000) hat g≈2,7–3,2: ein einfacher klassischer Kern ist ebenso „nah“.
3. **Wo QIRE klassisch unterscheidbar bleibt (Funktionsebene):** Mit begrenzten Daten (800) lernt der quadratische Kern QIRE-Zielfunktionen schlechter (NMSE 0,09 gegen 0,04 bei Identität). Das ist ein Stichproben-Effekt (256 von 32.896 Richtungen sind „bekannt“), keine Rechengrenze; ein Klassiker, der die Unterraum-Struktur kennt, **ist** die QIRE-Zelle (O(N log N)).
4. **Magie-Anteil „nicht klassisch“ nur in diesem Sinn:** (a) formal: k Nicht-Clifford-Winkel verlassen den Stabilizer-Formalismus (Gottesman–Knill) — Surrogatkosten 2^k, nicht polynomiell in k; (b) Termzahl: die Surrogatdichte 3^k ist real belegt; (c) praktisch: bei k=4 trivial, bei k≈n würde das Surrogat teuer. Dagegen: Rechengröße k=4 ≪ n=8, Zustandsvektor ist in 256 Werten trivial; Magie hat hier keine Beweiskraft für Quantenvorteil. Konsistent mit Bericht 02/04.
5. **Was bei n=8 übrig bleibt:** ein extrem günstiges Struktur-/Parametermodell: feste Walsh-/Maskenstruktur (Bruhat) + 256 quadratische Merkmale + kleiner Kopf. Der Befund „Clifford-Suche≈volles Training“ ist eine Aussage über **quadratische Merkmalsbasen**, nicht über Quantenressourcen. Ein klassisches Netz mit „signierte Maske + WHT + Quadrat“ muss geprüft werden (Bowles-Ablation, Bericht 02 I-2).
6. **Grenzen:** nur MNIST (Fashion nicht gerechnet); 1.000 Bilder, λ-abhängig (g_λ definiert wie oben; Normierung beeinflusst absolute Werte); Suche verkürzt (500 Schritte statt 6.000, LOO 87,5 % statt ~91 %); k=4 Positionen zufällig, nicht optimiert; Surrogat-Term-Messungen mit Zufallsgerüsten, nicht den gesuchten; n=16 nur 3 Gerüste/4 Bilder. Bent-Formen: eine feste (Paare benachbarter Bits).
7. Siehe Hinweis: ein Zähleffekt — viele Terme bei k=8 sind belegt, weil Gerüst zufällig und dicht; für gesuchte Bruhat-Gerüste (Blockstruktur) könnte die Zahl kleiner sein (offen).

## 4. Vorschlag für einen vorregistrierten Lauf
Titel: „Dequantisierungsprüfung des QIRE-Kerns“.
- **H1 (Dequantisierung):** g_λ(quadratisch‖K_Q) ≤ 1,5·g_λ(quadratisch‖K_Q,Identität) für Suche und Suche+k=4 (TOST-Gleichwertigkeit mit Rand ±0,5 in g; Erkundung: Differenz +0,07±0,18 mit n=5, sd ≈0,2 → bei 10 Seeds ist ±0,5 erreichbar, bei ±0,2 nicht).
- **H2 (kein k-Effekt):** g für k∈{0,4,8,32} gleich (Gleichwertigkeit ±0,3).
- **H3 (Surrogat):** Termzahl n(99 %) bei k=4, gesuchtes Gerüst, ≥ 60 von 80 (beschreibend), Positivkontrolle Gerüst mit gleichen Winkeln auf einem Qubit (Support kleiner).
- **H4 (Funktion):** NMSE klassischer quadratischer Kern auf K_Q-Zielfunktionen bei n=800/2.000/5.000 Trainingsbildern fällt gegen 0 (Stichprobenkurve).
- Arme: Identität, Walsh, Bent8, Suche (6.000 Schritte), Suche+k∈{1,2,4,8} (Positionen per Greedy statt zufällig), generisch. Kontrollen: zufälliges Clifford; **Positivkontrolle**: ein Kern, der nicht in der Quadratform liegt (z. B. Kern aus Grad-4-Merkmalen von x̂, oder Winkelkodierung mit Re-uploading), muss g_quad ≫ 4 zeigen.
- Seeds: ≥10 (z. B. 990–999 für Messung, nicht neu verwenden); Datensätze MNIST und Fashion; 1.000 und 2.000 Bilder. Erwartung: g_quad 3,7–4,1 (sd 0,1–0,3), Genauigkeit klassisch RBF ≈94 % ≥ QIRE.
- Laufzeit: je Seed ca. 3–4 min für 1.000 Bilder bei 500-SA-Schritten (belastete Maschine, 2 Threads); mit 6.000 Schritten ~15–20 min je Seed; 10 Seeds × 2 Datensätze ≈ 6–8 h auf 1 Kern, ~2 h auf 4–5 Kernen. Surrogat-Teil n=8, k bis 8: ~40 s je Gerüst und Platzierung (k=8), n=16 bis k=4: 30 s.

## 5. Offene Fragen
- Fashion (Magiebedarf dort höher): g-Messung steht aus.
- Regularisierte g-Formel nach Huang et al. im Wortlaut prüfen (hier eigene Fassung g_λ).
- Surrogat-Termzahl mit **gesuchten** Gerüsten und mit echten Magiewinkeln statt Zufallspositionen; Skalierung von n(99 %) mit n ≥ 12 bei k=8 (n=16, k=8 nicht gerechnet).
- Gilt K_Q⪯c·K_quad mit explizitem c? (Beweis aus Normgrenze der Merkmalsabbildung P: Vermutung c ≤ 1 für Tr-Normierung; nicht bewiesen.)
- Beispiel für einen echten Nicht-Dequantisierungs-Kern (Re-uploading, k ≫ n) auf Maltes Daten: Positivkontrolle.
- Bei Abtast-Auslese und n=32: Bravyi–Gosset-Rechnung konkretisieren (Konstanten, Bericht 04 E5).

## 6. Pfad der Skripte
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v6/`: `common.py` (Daten, Suche, Magie-Koordinatensuche, Bent-Matrizen), `geo.py` (g, RKHS-Test, Genauigkeit), `surr.py` (Frequenz-/Termzählung, Zweig-Check), `kosten.py` (Kostenmodell), `agg.py` (Auswertung); Rohdaten `geo_mnist_980_981_982_983_984.json`, `surr_n8_0_1_2_4_8_980_981_982_983_984.json`, `surr16.log`.
