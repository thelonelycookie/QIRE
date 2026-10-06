# v2 — Ist „Bent statt nicht-Bent“ der Kern des Rangeffekts? Kann eine feste Bent-Maske die Suche ersetzen?

Stand 06.10.2026. **ERKUNDUNG, kein Befund** (nichts vorregistriert). Alles Deutsch, Seeds 920–939 (Etappe-2-Block).

## 1. Kurzfassung
- **Nur global (k=8) ist Bent der Rangeffekt.** Dort steigt die Genauigkeit monoton mit dem Rang (MNIST 88,2 / 90,0 / 91,1 / 91,6 / 91,9 für Rang 0/2/4/6/8; Fashion 79,5 / 80,4 / 81,1 / 81,8 / 82,1). Bei festen Blöcken ist die Rang-Kurve flach (block8x8: alle Ränge 92,8–93,0 MNIST, ±0,2) oder sogar fallend (block2x2: Bent −0,9 pp MNIST, −1,7 pp Fashion gegenüber keiner Maske).
- **Keine Rang-Invariante:** Bei festem V und Rang streut die Genauigkeit zwischen Formen desselben Seeds um 0,4–1,1 pp (Spannen bis 5 pp, z. B. block4x4 Rang 2: 89,0–94,0 MNIST) — genauso viel wie der Rangunterschied selbst. Rang erklärt den Mittelwert (global), nicht die einzelne Form.
- **Feste Bent-Form ≠ Suche.** Die kanonische xᵀy-Bent-Form kommt global auf 91,5 (MNIST) / 81,5 (Fashion); I-036-v3-Suche: 93,6 / 82,3. Lücke ≈ −2,1 pp (MNIST), −0,8 pp (Fashion) — *Achtung:* anderer Seeds und anderer Vergleich (siehe §3), aber die Größenordnung ist klar. Beste feste V/Bent-Kombination MNIST: block4x4 + xᵀy-Bent 92,6 (aber block8x8 ohne Maske schon 92,65).
- **Die Wahl der Bent-Form zählt:** xᵀy-Paarung (i↔i+k/2) schlägt die „Nachbar“-Paarung (i↔i+1) um +2,3 pp (block4x4), +1,4 pp (global) auf MNIST — beide sind Bent, gleicher Rang. Die Paarung koppelt Zeilen- mit Spaltenbits (rref-Basis: hohe Bits = Zeilen); das ist plausibel der wirksame Teil, aber hier nicht isoliert.
- Theorie: Satz „Rang 2h ⇒ |W|=2^(n−h) auf 2^(2h) Stellen“ **bewiesen** und numerisch bestätigt (900 Funktionen, 0 Fehler). Folge für Born-Werte: jedes Merkmal ist das Quadrat einer ±2^(−h)-Mischung von 2^(2h) Walsh-Koeffizienten.

## 2. Aufbau und Zahlen
Kopf wie I-036-v3 (`kopf_logistisch`, Softmax, standardisiert, L2 0,003, 256 Born-Merkmale → 2.570 Parameter), 6.000 Trainings-/5.000 Testbilder aus dem offiziellen Testsatz (`clifford.daten`, 16×16), kein Hyperparameter am Testsatz gewählt (L2 aus Vorläufern übernommen). V: block2x2 (k=2), block4x4 (k=4), block8x8 (k=6), global (k=8) = Qubit-Unterräume. Maske = (−1)^{q_B}, gleiche Form auf allen Nebenklassen (`maske_aus_form`). Arme je (V, Seed): keine Maske; je Rang r=2..k 8 zufällige Formen (B=PᵀB₀P, P∈GL(k,2); bei k=2 nur 1 Form existiert); xᵀy-Bent (B_{i,i+k/2}=1); Nachbar-Bent (B_{i,i+1}); „Zufall Rang gleichverteilt“ = Mittel der Rang-Mittel inkl. Rang 0 (=keine Maske) — gleich im Erwartungswert zur gleichverteilten Ziehung.
**Seeds:** MNIST 20 (920–939), Fashion **14** (920–933; der Fashion-Prozess wurde nach Abbruch/Zeitlimit beendet; ≥5 Seeds erfüllt). Zahlen = Mittel ± sd über Seeds in %; Identität: MNIST 90,23 ± 0,31, Fashion 79,93 ± 0,33. Rechenaufwand wegen Last 4 Kerne ≈ 2,2 min je (Datensatz, Seed) auf 1 Thread.

**Mittel ± sd (%) je V** (M = MNIST, F = Fashion)

| V (k) | Arm | MNIST | Fashion |
|---|---|---|---|
| block2x2 (2) | keine Maske | 92,74 ± 0,31 | 82,30 ± 0,35 |
| | Bent (einzige Rang-2-Form) | 91,81 ± 0,31 | 80,59 ± 0,28 |
| | Zufall (Rang gl.) | 92,28 ± 0,27 | 81,45 ± 0,29 |
| block4x4 (4) | keine | 91,04 ± 0,34 | 80,74 ± 0,40 |
| | Rang 2 / Rang 4 (Zufall) | 91,38 / 91,76 | 81,47 / 81,08 |
| | Bent xᵀy | **92,63 ± 0,32** | 81,07 ± 0,34 |
| | Bent Nachbar | 90,30 ± 0,29 | 80,78 ± 0,33 |
| | Zufall (Rang gl.) | 91,39 ± 0,34 | 81,09 ± 0,30 |
| block8x8 (6) | keine | 92,65 ± 0,35 | 81,53 ± 0,45 |
| | Rang 2 / 4 / 6 | 92,97 / 92,81 / 92,85 | 81,71 / 81,86 / 81,82 |
| | Bent xᵀy / Nachbar | 92,62 ± 0,33 / 92,08 | 81,66 ± 0,51 / 81,34 |
| global (8) | keine | 88,23 ± 0,47 | 79,53 ± 0,36 |
| | Rang 2 / 4 / 6 / 8 | 89,95 / 91,10 / 91,57 / 91,86 | 80,43 / 81,14 / 81,81 / 82,08 |
| | Bent xᵀy / Nachbar | 91,53 ± 0,38 / 90,17 | 81,49 ± 0,42 / 81,25 |
| | Zufall (Rang gl.) | 90,54 ± 0,25 | 81,00 ± 0,28 |

**Gepaarte Differenzen (pp, Mittel ± se, positive Seeds):**
- global MNIST: xᵀy-Bent − keine +3,29 ± 0,11 (20/20); Zufalls-Bent − keine +3,62 (20/20); xᵀy-Bent − Zufalls-Bent −0,33 ± 0,08 (4/20); xᵀy − Nachbar +1,36 (20/20). Fashion: +1,96 (14/14); Zufalls-Bent −keine +2,55 (14/14); xᵀy − Zufalls-Bent −0,59 (1/14).
- block4x4 MNIST: xᵀy-Bent − keine +1,59 (20/20), − Zufalls-Bent +0,87 (20/20), − Nachbar +2,33 (20/20); Nachbar-Bent − keine −0,74 (0/20). Fashion: nur +0,33 (12/14), von Zufalls-Bent nicht unterscheidbar (−0,01).
- block8x8 MNIST: xᵀy-Bent − keine −0,03 (8/20); Zufalls-Bent − keine +0,20 (15/20) — praktisch null. Fashion +0,13 (8/14).
- block2x2: Bent ist **schlechter** als keine Maske (−0,94 / −1,71 pp, 0 positive Seeds).

## 3. Teil 2
**(a) Rang-Invariante? Nein.** Streuung zwischen Formen gleichen Rangs (gleicher Seed, 8 Formen): global 0,5–1,1 pp (MNIST), 0,4–0,6 (Fashion); block4x4 0,6–1,0; block8x8 0,7. Zwischen Rang 2 und 4 liegt global ≈ 1,1 pp, zwischen Rang 6 und 8 nur 0,3 pp — **kleiner als die Formstreuung**. Rang ist ein guter Prädiktor für den Mittelwert über Formen nur bei k=8 (und in Fashion-global), bei k≤6 ist er statistisch kaum von 0 verschieden. Bent-Anteil unter Zufallsformen (k=8) 0,42 (Bahnzählung, aus 09), sodass „zufällig“ meist schon nahe Bent liegt.
**(b) Feste Bent-Form vs. Suche.** Vergleich nur indikativ: Suchwerte (0,936 / 0,823) stammen aus I-036-v3 auf anderen Seeds (760–799), Suche optimierte V und Form. Beste Feste: MNIST 92,6 (block4x4/xᵀy, auch block8x8-keine 92,65 — also −0,9 bis −1,0 pp unter der Suche); Fashion global-Zufalls-Bent 82,1 (xᵀy 81,5, block8x8 81,7–81,9) — Fashion-Lücke zur Suche nur ≈ 0,2–0,8 pp. Eine einzige feste kanonische Form ersetzt die Suche auf MNIST nicht (≥ 0,9 pp, TOST ±0,5 pp klar verfehlt), auf Fashion nahe, aber nicht belegt (andere Seeds, 14 Seeds).
**(c) Cut-Rank der gespeicherten Suchformen (I-036-v3-JSON, `form_s`, `V_s`, 20+20 Seeds):** Gewinn = suche − ohne_maske. Rang vs Gewinn Spearman +0,79 (MNIST), +0,49 (Fashion) — aber **bei Kontrolle von k** (partiell) nur +0,34 / +0,16; k selbst +0,86 / +0,54. Cut-Rank über 100 zufällige halbe Koordinatenschnitte: ρ +0,72 / +0,51, partiell(k) +0,05 / +0,13. Zeile|Spalte-Schnitt nur in 9/20 (MNIST) bzw. 3/20 (Fashion) Seeds definierbar (V nicht Zeilen⊕Spalten-zerlegbar), Werte 2–3, ρ nur +0,14 (n=9). Bent-Suchformen (Rang=k): 2 (MNIST) bzw. 3 (Fashion) Seeds, Gewinn 2,9 vs 2,5 pp (MNIST), 2,5 vs 2,2 (Fashion) — Unterschied winzig. Rang-Defizit (k − Rang) ↔ Gewinn ρ −0,50 (MNIST), +0,01 (Fashion). Und: Rang ↔ Formvorteil (suche − Zufall gleichen Rangs) ρ −0,50 / −0,07 — hohe Ränge lassen der Suche weniger Formvorteil. **Langweilige Erklärung:** Rang ist stark mit k (Größe des Unterraums) konfundiert; Suchläufe mit großem k haben mehr Gewinn und mehr Rang.

## 4. Deutung (ehrlich)
- Der Rangeffekt (v3) kommt zum großen Teil von der **Größe des Unterraums bei zusammenhängender Maske**: Ohne Maske sind große Walsh-Blöcke schlecht (global 88,2 < Identität 90,2; Fashion 79,5 < 79,9), jede hochrangige Form behebt das. Bei k ≤ 6 ist der Walsh-Block ohne Maske schon gut, und Rang/Bent bringen ≈ 0 bis ±0,3 pp.
- „Bent statt nicht-Bent“ erklärt also nur, **warum global rangabhängig ist**; es erklärt nicht die Suche (die V und Form wählt, Zufalls-Bent global ist schon so gut wie xᵀy-Bent, aber 1,5–2 pp hinter der Suche).
- Bent ist nicht automatisch gut: block2x2-Bent ist schlechter als keine Maske; Nachbar-Bent in block4x4 schlechter als keine. Die Struktur der Kopplung (Zeile↔Spalte-Bits) zählt, nicht nur der Rang. Ob die Kopplungs-Struktur als Merkmal („welche Bits der Form verbunden“) die verbleibende Form-Info trägt, ist hier nicht getestet.
- Vorsicht: Fashion nur 14 Seeds; Zufallsformen 8 je Rang; Vergleich mit Suche über andere Seeds.

## 5. Theorie (Beweis)
**Satz.** q(x)=xᵀUx+l·x über GF(2)ⁿ, B=U+Uᵀ alternierend vom Rang 2h, f=(−1)^q, W(a)=Σ_x f(x)(−1)^{a·x}. Dann W(a)∈{0,±2^{n−h}}, mit genau 2^{2h} Stellen a ≠ 0.
**Beweis.** W(a)²=Σ_{x,z}(−1)^{q(x)+q(x+z)+a·z}. Wegen q(x+z)=q(x)+q(z)+B(x,z) folgt W(a)²=Σ_z(−1)^{q(z)+a·z}Σ_x(−1)^{B(x,z)}. Die innere Summe ist 2ⁿ für z im Radikal R=ker B (dim n−2h), sonst 0. Auf R ist B=0, also q|_R additiv-linear: q(z)=λ(z). Daher W(a)²=2ⁿ·Σ_{z∈R}(−1)^{(λ+a)·z}=2ⁿ·|R| wenn (a+λ)⊥R, sonst 0, d. h. |W|=2^{(n+n−2h)/2}=2^{n−h}. Der Träger {a:(a+λ)|_R=0} ist ein Nebenklasse von R^⊥ mit 2^{2h} Elementen. Parseval-Probe: 2^{2h}·2^{2n−2h}=2^{2n}. ∎ (Gilt für jede quadratische Funktion, auch mit linearem Anteil; Bent: R=0, h=n/2, |W|=2^{n/2} überall.) Numerisch: 900 Zufallsfunktionen (n=4,6,8), 0 Abweichungen (Skript `theorie.py`).
**Folge für Born-Werte** (Block, orthonormal H, c=H p Walsh-Koeffizienten der normierten Pixelwerte): Merkmal_j = (H(s⊙p))_j² = (2^{−h}·Σ_{a∈T_j} ε_j(a)·c_a)², T_j Nebenklasse mit 2^{2h} Elementen, ε=±1 (Faltung mit dem Maskenspektrum; numerisch bestätigt). Rang 0: Merkmal = einzelner c_j² (Leistungsspektrum). Rang 2h: Mischung von 2^{2h} Walsh-Koeffizienten mit gleichem Betrag 2^{−h}. Bent: alle 2^k Koeffizienten, Merkmal_j = 2^{−k}(Σ ε c_a)² = 2^{−k}Σc_a² + Kreuzterme; der Diagonalterm ist für **alle** j gleich (Energie/2^k), die ganze unterscheidende Information steckt in den **vorzeichenbehafteten Paarprodukten c_a c_b** (Autokorrelations-artig). Bent liefert somit „flache, nicht-Parity“-Merkmale zweiter Ordnung, aber ob das nützt, hängt von den Vorzeichenmustern ε (also der Form, nicht nur dem Rang) ab — passt zur gemessenen Formabhängigkeit.

## 6. Vorschlag für einen vorregistrierten Lauf
- **Hypothesen:** H-A (global k=8): Zufalls-Bent − xᵀy-Bent ≥ 0 (nicht xᵀy-spezifisch); H-B: Rang-Kurve bei k ≤ 6 gleichwertig (TOST ±0,5 pp: MNIST block8x8 Rang 6 − keine +0,2 ± 0,08 → erreichbar bei 20 Seeds); H-C: Kopplungsstruktur (Zeile↔Spalte vs Nachbar) trägt, bei gleichem Rang (Differenz +1,4–2,3 pp, se 0,1 → mit 10 Seeds sicher).
- **Arme:** feste Bent-Familien (xᵀy, Nachbar, Zeile↔Spalte-gemischt, 5 Zufalls-Bent), je V ∈ {block4x4, block8x8, global}; **Positivkontrolle:** global ohne Maske < alle Bent (Spreizung +3 pp); **Kontrolle:** Suche aus I-036-v3 auf denselben Seeds.
- **Seeds:** 20 (MNIST), 20 (Fashion), Streuung gepaart ≈ 0,3–0,5 pp. Laufzeit: ≈ 2 min je Seed·Datensatz·V-Satz auf 1 Thread; ≈ 45 min auf 4–5 Kernen ohne Suche.
- Nur wenn die Suche selbst mit auf denselben Seeds läuft (≈ 6 min/Seed), ist die Lücke feste Form↔Suche sauber messbar.

## 7. Offene Fragen
- Warum ist xᵀy-Kopplung (Zeile↔Spalte) besser als Nachbar-Kopplung? Isolieren (Form-Suche nur über die Paarung?).
- Reicht eine **gelernte** feste Form (Übertragung einer „Aufgabenform“, v3 fremde Maske +0,84 pp vs. Zufall) besser als jede feste Bent-Form? Hier nicht getestet.
- Cut-Rank bezogen auf Pixelschnitte (Zeile|Spalte) ist nur für 15–45 % der Suchformen definiert — braucht Basis-unabhängige Definition.
- Fashion-Lauf nicht vollständig (14/20 Seeds).

## 8. Skripte
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v2/` : `v2.py` (Messung), `run.py`, `ana.py` (+ `ana_out.txt`), `cut.py` (Cut-Rank aus I-036-v3-JSON), `theorie.py`, Rohdaten `roh_0.jsonl` (MNIST), `roh_1.jsonl` (Fashion).
**Parameter:** trainiert nur Kopf 2.570 reelle Zahlen; diskrete Wahl: V (Qubit-Teilmenge) + Formmatrix k(k−1)/2 Bit (≤ 28 Bit global); keine datengelernte Vorverarbeitung.
