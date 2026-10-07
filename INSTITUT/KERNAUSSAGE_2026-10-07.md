# Kernaussage-Entwurf für die Jugend-forscht-Arbeit (07.10.2026)

*Entwurf, ehrlich. Alle Zahlen aus `LABORBUCH.md` (06.10.), `PLAN_2026-10-06_gesamt.md`, `recherche/2026-10-06_literatur_struktur.md`. Literaturangaben stammen nur aus Suchauszügen (kein Volltext gelesen).*

## 1. Kernaussage in 3 Sätzen
1. Eine feste, billige Mischung (Hadamard·Diagonal, „Fastfood“, O(N log N), 0 gelernte Mischparameter) mit anschließender Born-artiger Quadrat-Auslese ersetzt auf MNIST/Fashion eine gelernte QIRE-Zelle und große dichte Schichten ohne messbaren Verlust. Das ist ein **quanten-inspirierter Vorteil bei Parametern und Rechenaufwand, kein Quantenvorteil**: Die Wirkung ist klassisch erklärbar (I-038).
2. Der Vorsprung gegenüber einfachen Pixelmodellen kommt aus Mischung plus Nichtlinearität, nicht spezifisch aus dem Born-Quadrat (|z| und ReLU wechseln sich ab). Die Struktur trägt dort, wo die Eingabe schon gute Merkmale hat (Fashion, CNN-Kopf), nicht auf Rohpixeln schwerer Aufgaben (CIFAR-10).
3. Strukturierte Mischungen und |·|² sind bekannt (Fastfood, SORF, OPU-Merkmale); unser Beitrag ist ein kontrollierter, vorregistrierter Vergleich mit Negativ- und Positivkontrollen sowie die Messung, was die Mischung an Information pro Schuss kostet (I-040-v2). Nicht behauptet: „erstmals“, „neue Architektur“, „Born schlägt klassische Nichtlinearitäten“.

## 2. Belegtabelle
| Aussage | Lauf | Zahl (Seeds; KI nur wo im Laborbuch angegeben) | Stärke |
|---|---|---|---|
| Gelernte QIRE-Zelle ≡ feste dichte Mischung (Haar) | I-038 H-A | TOST ±0,5 pp: +0,18 (MNIST) / +0,10 pp (Fashion); 20 Seeds 2000–2019 | belegt (gleichwertig) |
| QIRE gelernt ≡ Zufallswinkel | I-038 H-B | +0,22 / +0,07 pp; 20 Seeds | belegt |
| Fastfood-2 ≡ Haar² | I-038 H-C; I-041 H4 | +0,09 / +0,01 pp (16×16); bei 32×32 ebenfalls gleichwertig (10 Seeds 2060–2069) | belegt |
| Mischung+Quadrat schlägt Pixel-Softmax | I-038 H-D | +2,84 pp MNIST, +1,43 pp Fashion; 20 Seeds | belegt |
| Quadrat trägt gegenüber linear | I-038 H-E | +2,21 / +1,27 pp; 20 Seeds | belegt |
| Struktur-Vorsprung wächst mit Größe | I-041 H1/H2 | 32×32: +4,33 pp MNIST (Zuwachs +1,64 pp, wächst), Fashion +1,93 pp (Wachstum +0,44, nicht entschieden); 10 Seeds | belegt (MNIST) / Hinweis (Fashion) |
| MNIST mit 110 reellen Parametern ≥ 70 % | I-039 H1 | 80,46 ± 0,81 % (+882 Strukturbits formal, 3.046 streng); 20 Seeds 2020–2039 | belegt, mit Bitvorbehalt |
| Besser als klassisch gleicher reeller Parameterzahl | I-039 H2 | +5,61 pp gegen 5×5→3 tanh→10 (118 P., 0 Bit); Holm-p < 10⁻⁴ | belegt, aber unfair bzgl. Bits (s. u.) |
| Fastfood spart Rechenzeit | I-041; I-042 H4 | schneller ab N = 1.024 (Batch 1) / 2.048 (Batch 64); Operationsverhältnis 89× bei N = 1.024; 4–19× (Fashion), 7–54× (CIFAR) bei N ≥ 4.096 | belegt (CPU, ab N ≈ 1–2k) |
| Feste Struktur ersetzt 16,8-Mio-Schicht ohne Verlust (leichte Aufgabe) | I-042 H1 Fashion | A 89,0 / B 89,1 / E 89,1 %, +0,1 pp; 5 Seeds 2200–2204 | belegt (n = 5, kleine Stichprobe) |
| Struktur-Kopf im CNN ≥ dichter Kopf | I-044 H3 | 73,9 gegen 72,1 % (+1,8 pp), ≈ 49 k gegen 4,2 Mio Kopfparameter (≈ 85× weniger), 3× schneller (Batch 1); 5 Seeds 2300–2304 | belegt (Nichtunterlegenheit), n = 5 |
| Trainierte Diagonalen schlagen niedrigen Rang (gleiches Budget) | I-042 H3 | +29,7 pp CIFAR, +21,6 pp Fashion; 5 Seeds | belegt |
| Zielkonflikt: dichte Mischung verbessert exakte Auslese, verschlechtert Information pro Schuss | I-040-v2 H4 (H1) | Haar +3,5 pp Genauigkeit, −0,20 Bit/Schuss (Identität − Haar +0,201 Bit); 10/10 Seeds 2150–2159 | belegt |
| Schussbewusstes Training hilft bei wenigen Schüssen | I-040-v2 H2 | +1,5 / +1,2 / +0,3 pp bei S = 10/30/100 | belegt (klein bei S = 100) |
| Born-Quadrat knapp besser als beste klassische Nichtlinearität | I-038 H-G | MNIST +0,35 pp [0,27; 0,43]; Fashion 0,00 (gleichwertig) | nur Hinweis (klein, nicht übertragbar, s. §3) |
| Satz „Rang 2h ⇒ flaches Walsh-Spektrum“ | v2/09 (Erkundung) | bewiesen (Beweis, kein Lauf) | belegt (mathematisch) |
| Adapter-Algebra so(256), Jacobi-Rang min(8L, 32.640), L* = 4.080 | v5 (Erkundung) | Rechnung, nicht vorregistrierter Lauf | Hinweis (Erkundung) |

## 3. Widerlegt oder nicht belegt (mit Lauf-ID)
- **Quantenvorteil / Quanteneffekt der Zelle:** Verschränkung und gelernte Winkel tragen nichts Messbares (I-038 H-A/H-B; Zufallswinkel und Haar gleichwertig). Der Kern ist für jedes Magie-Budget dequantisiert (v6, Erkundung). Die Clifford-/Magie-Linie (I-005…I-036-v3) bleibt als Ressourcenaussage (T-Gatter) bestehen, nicht als Quantennutzen.
- **Born-Quadrat ist die bessere Nichtlinearität:** nicht belegt. I-038 H-G nur +0,35 pp (MNIST), Fashion 0,00; I-039-v2 H1/H2 GLEICHWERTIG (b_ff 80,26 % gegen k3_abs 80,65 %, −0,39 pp, TOST ±1 pp); I-041 H3: Born-Vorsprung wächst nicht (MNIST −0,07 HU, Fashion −0,57 schrumpft); I-042 H2: Wechsel je Aufgabe (z² − |z| +0,34 CIFAR / −0,36 Fashion; z² − ReLU −3,18 CIFAR / +0,57 Fashion).
- **Vorsprung von I-039 (+5,61 pp) ist ein Born-Effekt:** widerlegt. Mit gleichem Bitbudget und |z| erreicht ein klassisches Modell dasselbe (I-039-v2: k2_relu 79,8, k1_lin 79,4, k3_abs 80,65 %; klassisch d ohne Struktur 75,0 %). Er kommt aus Mischung plus Gruppierung (Struktur).
- **Struktur ersetzt dichte Schichten auf Rohpixeln schwerer Aufgaben:** widerlegt. CIFAR-10: −6,3 pp bei N = 16k (I-042 H1), 3 Strukturschichten −5,6 pp gegen dicht (I-044 H1), Tiefe hilft nicht (I-044 H2: −1,5 pp). Strukturierte Kanalmischung im CNN nicht entschieden (I-044 H4, −0,7 pp, HU).
- **Clifford-Suche besser als Haar²:** nicht belegt (I-038 H-F: MNIST HU −0,04, Fashion SUCHE_NICHT_BESSER −1,11 pp).
- **Holevo-Wirkungsgrad η = I/χ sagt den Schussbedarf voraus:** nicht belegt (I-040-v2 H3 H0, ρ 0,7, p 0,93, nur 5 Arme). I-040 (erster Lauf) war wegen PK-1-Versagens (0,868 < 0,88) nur für H1 gültig; H2/H3 HU, deshalb Wiederholung I-040-v2.
- **Plateau des Adapters beim Jacobi-Rang:** verworfen (v5: L* = 4.080, bei L = 20/40 begrenzt das Trainingsbudget).
- **Echter Zeitgewinn auf CPU unter N ≈ 1–2k:** nein (I-041). Die WHT-Zelle ist gegen BLAS auf CPU kleiner N langsamer (Plan §1.1, Erkundung 11).
- **Offene Offenlegungen:** I-039 Rauchtest einmal auf Mess-Seed 2020; I-038-Suche nur gepaart auf 10 Seeds (2000–2009); alle Bilder 16×16 bzw. resampeltes 32×32 (kein neuer Bildinhalt); I-042/I-044 nur 5 Seeds, 4–5 Epochen.

## 4. Lücken bis zur Anmeldung (30.11.)
| Fehlt | Zweck / Jurorenfrage | Stand |
|---|---|---|
| **T-006 Auswertung** (Adapter L = 20/40 gegen LoRA, K1–K3) | „Skaliert der Adapter, und wo bremst er?“ | Rohdaten Kern fertig (qire_L10, lora_r1, qire_L20, qire_L40 je 8 Seeds), qire_L10_600 läuft; Auswertung offen |
| **I-043 Sprache auf PC** (Fastfood-Schicht in FFN/Adapter von LLäMmlein-120M gegen LoRA und QIRE-Adapter) | „Gilt das auch für Sprachmodelle, nicht nur Bilder?“ Laut I-044 aussichtsreichster Fall (Schicht hinter gelernten Merkmalen) | nicht vorregistriert |
| **I-045 Container-Sprachmodell** (kleines Modell im Container, Wiederholung unabhängig vom PC) | „Ist das reproduzierbar und nicht ein Einzelergebnis auf einem PC?“ | im Handover nicht definiert, Entwurf nötig |
| **MPI-Skalierung** (Modelle ≥ 1 Mrd. Parameter) | „Gilt es bei realer Größe?“ | Zugang in Aussicht, Details offen |
| Fairer Bit-Vergleich auf weiteren Aufgaben, Neuheitsprüfung per Volltext | „Was ist neu?“ | Literaturcheck nur Auszüge (E4) |

## 5. Titel und Gliederung
**Titelvarianten**
1. „Feste Struktur statt gelernter Gewichte: Was eine quanten-inspirierte Mischung wirklich leistet“
2. „QIRE: Quanten-inspirierte Auslese als effiziente neuronale Schicht, ehrlich vermessen“
3. „Quanten-inspiriert, nicht Quanten-schneller: Hadamard-Mischung und Born-Regel in effizienter KI“

**Gliederung (11 Punkte)**
1. Fragestellung und Motivation (ressourceneffiziente KI)
2. Grundlagen: Born-Regel, Clifford-Gruppe, Hadamard-Mischung (kurz)
3. Stand der Literatur und was bekannt ist (Fastfood, OPU, Primer)
4. Methode: Vorregistrierung, Kontrollen, Parameter- und Bitzählung
5. Die QIRE-Zelle und die Gleichwertigkeit zu fester Mischung (I-038)
6. Was das Born-Quadrat wirklich beiträgt (I-038 H-G, I-039-v2, I-041, I-042)
7. Effizienz: 110 Parameter und Skalierung (I-039, I-041, I-042)
8. Grenzen der Struktur: CIFAR-10, Tiefe, CNN-Kopf (I-042, I-044)
9. Messphysik: Information pro Schuss und Zielkonflikt (I-040-v2)
10. Sprachmodelle und Adapter-Skalierung (T-006, I-043, I-045, MPI)
11. Diskussion, Grenzen, Ausblick (kein Quantenvorteil, Neuheitsprüfung, offene Fragen)
