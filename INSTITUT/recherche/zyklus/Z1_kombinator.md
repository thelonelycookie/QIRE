# Z1 Kombinator (Zyklus 1, 28.10.2026)

Nichts gerechnet, kein git. Gelesen: LOOP, Welle 3 (f1–f4), TURNIER, konv_kandidaten, SYNTHESE. 7 WebSearch. "(G)" = aus Gedächtnis, nicht geprüft. Vorab-P = Schätzung, dass die Kombination den **stärksten Einzelgegner** bei **beiden** Trainingslängen schlägt (über Rauschen); kein Befund. Wirkung klassisch (I-038), also "quanten-inspiriert".

## 0. Leitgedanke (aus den Funden abgeleitet)
- **Rotation macht Gewichte und Aktivierungen gaußförmig und isotrop.** Das hilft allem, was Gaußverteilung will: gleichmäßige Gitter, Zufallsbücher (Seed), 4-Bit-Aktivierungen. Es schadet allem, was schwere Ränder oder Struktur braucht: Beschneiden (2:4), Niedrigrang-Korrektur, koordinatenweises Adam. Daraus folgt, welche Paare sich addieren (A) und welche sich aufheben (X).
- **Frühregime-Regel trennt zwei Klassen.** Nachbehandlung fertiger Modelle (Quantisieren, Cache, Entwurf) ist immun. Alles, was beim Training mitläuft (QAT, MoE, geteilte Gewichte), braucht zwei Längen (hier: T1 ≈ 600, T2 ≈ 2400 Schritte, bei Byte-LM d = 128; bei MNIST 3 und 12 Epochen). Weiter nur, wenn der Vorteil bei T2 nicht unter die Hälfte von T1 fällt.
- Alle Tests: Tuning-Seeds 950–989, Gegner mit gleichen **Bytes**, und wo FLOPs sich ändern, auch mit gleichen FLOPs. Skalen-Bytes mitzählen. quant_rows(bits=2) ist ternär (1,58 Bit), als 2 Bit gezählt: offenlegen.

## Rangliste (Kurz)
| Rang | Nr. | Kombination | P | Kosten Test |
|---|---|---|---|---|
| 1 | C02 | Distillation in rotiertes Niedrigbit-Modell (QAT + KD) | 50 % | ≈ 15 min |
| 2 | C01 | Seed-Codebuch auf rotierten Blöcken (SeedLM + QuIP#-Idee) | 45 % | ≈ 10 min |
| 3 | C05 | z² + Rotation vor Down-Projektion + 4-Bit-Aktivierung | 40 % | ≈ 15 min |
| 4 | C04 | KV: Schicht-Teilung + Schlüssel an der Anfrage + Rotation 2 Bit | 40 % | ≈ 15 min |
| 5 | C03 | Ein Gewicht, zwei Auflösungen: rotierter 2-Bit-Kopf als Entwurf des vollen | 40 % | ≈ 10 min |
| 6 | C07 | z²→GELU-Übergabe (Frühregime nutzen, Spätregime meiden) | 20 % (≥ min der beiden: 60 %) | ≈ 20 min, billigster Test überhaupt |
| 7 | C06 | BitNet b1.58 mit Rotation vor der Ternärisierung (W1,58 A4) | 30 % | ≈ 20 min |
| 8 | C08 | Rotiertes 2-Bit-Grundgewicht + VeRA/Fastfood-Adapter (0-Byte-Matrizen) | 30 % | ≈ 15 min |
| 9 | C10 | MoE mit geteiltem 2-Bit-Grundgewicht, Experten = Diagonalpaare auf Seed-Matrizen | 25 % | ≈ 20 min |
| 10 | C11 | Fastfood-Sz-Student + Distillation (Datenregime) | 25 % | ≈ 20 min |
| 11 | C13 | Mamba/RWKV-Verwandte (LRU) + rotierte 2-Bit-Gewichte: Byte-Bilanz ohne Cache | 20 % | ≈ 20 min |
| 12 | C15 | Rotierter Trainingszustand: QAT-Latent und Adam-m in derselben Basis | 20 % | ≈ 20 min |
| 13 | C09 | MoE-Experten als Vorzeichen-/Dreh-Schlüssel eines geteilten Gewichts | 15 % | ≈ 20 min |
| 14 | C14 | Frühausstieg als Entwurf (Self-Speculation) mit rotiertem 2-Bit-Teilstapel | 15 % | ≈ 15 min |
| 15 | C12 | 2:4-Beschneiden in gedrehter Basis (Konfliktkombination) | 10 % | ≈ 3 min (Rechnung vorab) |

Empfehlung: C02, C01, C05, C04 bauen (Top 4), C07 als Nebenarm in C05 (gleiches Skript, kostet nichts). C12 nur als Papier-/Minutenrechnung, weil es das Leitgedanken-Gesetz (Gaußisierung gegen Sparsamkeit) testet.

---

## C02 Distillation in ein rotiertes 2-Bit-Modell
- **Bestehend:** BitNet Distillation (fp-Modell wird mit SubLN + Aufmerksamkeits-Distillation ins 1,58-Bit-Modell gebracht, [arXiv 2510.13998](https://arxiv.org/pdf/2510.13998)); BitDistiller (Selbst-Distillation unter 4 Bit, [arXiv 2402.10631](https://arxiv.org/pdf/2402.10631)). Beide ohne Rotation.
- **Mechanismus:** Schüler = Gewichte in rotierter Basis (H·D), Gitter 2 Bit, STE; Lehrer = fp-Modell gleicher Architektur. Verlust = KL(Lehrer‖Schüler) + kleiner Anteil Hartlabel.
- **Warum 1+1>2:** Rotation macht den Rundungsfehler isotrop und klein (E3/R1: +0,4 pp bei gleichen Bytes); KD liefert pro Schritt viel mehr Gradienteninformation als Hartlabels (Soft-Labels), und das Rundungsrauschen bleibt gleichmäßig verteilt, sodass der Schüler es als "weißes" Rauschen wegmitteln kann. Ohne Rotation lenken einzelne Ausreißer-Gewichte den STE-Gradienten (Instabilität in 2 Bit). Schwäche von KD (Lehrer-Kosten) und von QAT (Optimierbarkeit, vgl. T-ternär-butterfly-Kollaps) heben sich teilweise auf: KD stabilisiert, Rotation entschärft Ausreißer.
- **Härtester Gegner:** (a) QAT mit Rotation **ohne** KD bei gleicher Schrittzahl; (b) kleines dichtes fp16-Modell gleicher Bytes, **mit** KD (KD hilft jedem Schüler); (c) PTQ mit Rotation + Fehlerkompensation (GPTQ-artig) ohne Training.
- **Billigster Test:** Byte-LM d = 128 (oder MNIST-MLP). Lehrer fp auf T = 600 und 2400 Schritte trainiert. Schüler 300 / 1200 Schritte QAT. Arme: roh/rot × mit/ohne KD × klein-fp16-mit-KD. Maß: bpc gegen Bytes, Lehrer-Vorwärtsrechnung in Trainings-FLOPs ausweisen (E2-Pflicht). Kriterium: rot+KD schlägt rot (ohne KD) und klein+KD bei beiden Längen.
- **Vorab-P:** 50 %. Hauptrisiko: KD-Gewinn gilt für alle Schüler gleich und der Rotationsvorteil schrumpft im QAT (Latentgewichte passen sich dem Gitter an, analog zum Frühregime-Schrumpfen).

## C01 Seed-Codebuch auf rotierten Blöcken
- **Bestehend:** SeedLM (Block = Seed eines Pseudozufallsgenerators + wenige Koeffizienten, datenfrei, 3–4 Bit ≈ fp16 auf Llama-3-70B, ICLR 2025, [arXiv 2410.10714](https://arxiv.org/abs/2410.10714), aus f2); QuIP# (Hadamard + E8-Gitter); AQLM (gelernte Bücher).
- **Mechanismus:** Gewichtsblöcke (8 oder 16) zuerst mit Hadamard + Zufallsvorzeichen drehen, dann durch das nächste Wort eines Seed-Zufallsbuchs ersetzen (Seed = Bits, Buch = 0 Byte).
- **Warum 1+1>2:** Ein Zufallsbuch ist für **gaußverteilte** Quellen nahe der Raten-Verzerrungs-Grenze (zufällige Codes erreichen sie). Ohne Rotation sind Gewichtsblöcke schwerrandig und ein Zufallsbuch verschwendet Wörter auf leere Bereiche; mit Rotation passt die Quelle zum Buch. Rotation liefert die Gaußform, das Seed-Buch nutzt sie. SeedLM allein muss das mit mehr Koeffizienten bezahlen.
- **Härtester Gegner:** E8-Gitter (QuIP# ohne Feintuning, ebenfalls 0 Byte Buch); gelerntes k-Means-Buch (Bytes des Buchs mitzählen); skalar 4-stufig mit Rotation; SeedLM ohne Rotation.
- **Billigster Test:** K03-Skript plus zwei Arme (SeedLM roh/rotiert). MNIST-MLP, Basis auf 3 und 12 Epochen trainiert (schwerere Ränder nach längerem Training möglich). Gleiche Bytes. Kriterium: rotiertes Seed-Buch ≥ E8 − Rauschen und deutlich besser als nicht rotiertes.
- **Vorab-P:** 45 % (dass es E8 schlägt: nur 20 %; dass Rotation dem Seed-Buch hilft: 75 %). Beitrag ehrlich: Reproduktion + Byte-Buchhaltung. Nachbehandlung, also Frühregime-immun.

## C05 z² + Rotation + 4-Bit-Aktivierung
- **Bestehend:** BitNet v2 (H-BitLinear: Hadamard vor Aktivierungsquantisierung nur an Ausgangs- und Down-Projektion, W1,58 A4 gleichauf mit A8, [arXiv 2504.18415](https://arxiv.org/pdf/2504.18415)); BitNet a4.8 und Primer nutzen ReLU².
- **Mechanismus:** FFN-Mitte = z² (Born-Quadrat), vor der Down-Projektion Hadamard, Aktivierung 4 Bit.
- **Warum 1+1>2:** z² erzeugt Ausreißer (Quadrat verstärkt große Werte; f1 §3.6: quartisches Gewichtswachstum, Instabilität). Gerade deshalb ist z² in 4 Bit schwer. Rotation entschärft Ausreißer. Umgekehrt zeigt z² im Frühregime den Gewinn (−0,19 bpc bei d = 512, T = 1000). Schwäche von z² (Ausreißer) und Stärke der Rotation (Ausreißer) passen genau.
- **Härtester Gegner:** ReLU² + Hadamard (BitNet-v2-Rezept, hat das Gate); GELU + Hadamard; z²-Normierung (K09) ohne Rotation; z² mit 8 Bit.
- **Billigster Test:** Byte-LM d = 128 und 256, T = 600 und 2400 (zwei Längen sind Pflicht, weil z² gerade hier kippt). Arme: {gelu, relu², z²} × {fp, A4 roh, A4 rot}. Kurtosis der Mitte mitmessen. Kriterium: z²-rot-A4 verliert weniger gegen fp als relu²-rot-A4 **und** bleibt absolut vorn.
- **Vorab-P:** 40 % (absolut vorn bei T2: nur 20 %, f1: Gate schlägt reines z² bei langem Training). Vorteil realistisch nur "weniger Verlust durch Quantisierung", nicht Bestwert.

## C04 KV-Cache: Schicht-Teilung, Schlüssel an der Anfrage, Rotation 2 Bit
- **Bestehend:** KIVI (2 Bit, K pro Kanal, V pro Token); TurboQuant/QuaRot (Rotation im Cache); Cross-Layer-Attention (benachbarte Schichten teilen K/V-Cache, G); ALBERT-Befund (Teilen der Attention kostet wenig, der FFN viel, aus f2).
- **Mechanismus:** Schichtpaare teilen **einen** K/V-Cache. Jede Schicht liest ihn durch ihren eigenen festen Vorzeichen-Schlüssel auf der **Anfrage** q' = D_l·q (kostet 0 Bytes Cache, d Bit Schlüssel). Der gemeinsame Cache wird rotiert mit 2 Bit abgelegt, ein fp16-Fenster für die letzten Token.
- **Warum 1+1>2:** Die Einsparungen **multiplizieren sich** (Cache ÷ Teilungsfaktor × 2/16 Bit), und beide sind unabhängig (Teilung wirkt auf die Anzahl der Cache-Zeilen, Rotation auf Bits je Zeile). Der Schlüssel an der Anfrage ersetzt die Vielfalt, die die Teilung verliert. Zusätzlich: ALBERT zeigt, dass Attention-Teilung billig ist, die Einsparung kommt also dort, wo der Schaden klein ist (anders als bei K01 mit geteilter FFN).
- **Härtester Gegner:** Teilung ohne Schlüssel; gelernte Lagen-Skala statt Zufallsschlüssel; GQA/MQA gleicher Cache-Größe (weniger K/V-Köpfe, bekannt, trainiert von Grund auf); KIVI-Arm ohne Teilung bei gleichen Bytes.
- **Billigster Test:** Byte-LM mit 8 Lagen (4 sind zu flach, f2), d = 128, T = 600 / 2400. Arme: normal, Paar-Teilung ohne/mit Schlüssel, GQA gleicher Größe, jeweils × Cache {fp16, had-2Bit, KIVI-2Bit}. Maß: bpc gegen Cache-Bytes bei Kontext 128 und 512. Kriterium: Teilung+Schlüssel+had-2Bit liegt auf der Pareto-Linie gegen GQA+KIVI.
- **Vorab-P:** 40 % (Teilung braucht Training: Frühregime-Risiko, also Zwei-Längen-Regel; Cache-Quantisierung allein ist immun). Nur Byte-Achse, keine FLOP-Aussage.

## C03 Ein Gewicht, zwei Auflösungen: rotierter 2-Bit-Kopf als Entwurf des vollen Modells
- **Bestehend:** QSpec (Entwurf W4A4, Prüfung W4A16, gleiche Gewichte, Annahme ≈ 74 % auch bei Länge 6, [arXiv 2410.11305](https://arxiv.org/html/2410.11305v3)); ML-SpecQD (4-Bit-Direktkopie als Entwurf, 71 % Annahme gegen 42 % bei 68M-Entwurf, [arXiv 2503.13565](https://arxiv.org/pdf/2503.13565)); QuantSpec (4-Bit-Cache + Gewichte, > 90 %); EAGLE (Merkmals-Entwurf). Matryoshka-Quantisierung: höchstwertige Bits bilden die niedrige Auflösung (G).
- **Mechanismus:** Gewichte rotiert und in Bit-Ebenen gespeichert: die oberen 2 Bit sind der Entwurf, alle Bits der Prüfer. Gesamtbytes = ein Modell.
- **Warum 1+1>2:** Die Rotation macht den Fehler beim Abschneiden der unteren Bits gleichmäßig und klein, das erhöht die Annahmerate des Entwurfs; die Bit-Ebenen-Ordnung macht den Entwurf **kostenlos in Bytes** (keine Zweitkopie, anders als K18 mit eigenem Zwilling). Annahmerate steigt mit Ähnlichkeit zur Zielverteilung, und die Rotation erhöht genau diese Ähnlichkeit pro Bit.
- **Härtester Gegner:** n-Gramm/Prompt-Lookup (0 Parameter); EAGLE-artiger Kopf (wenige KB); Entwurf ohne Rotation; Layer-Skip (C14).
- **Billigster Test:** Fertig trainiertes Byte-LM (T = 600 und 2400), Annahmerate α bei Länge 4 für Entwurf = obere 2/3 Bit roh gegen rotiert; Wandzeit nur über Modell α, Aufwand-Verhältnis (nicht auf Maltes Ryzen behaupten). Kriterium: α_rot > α_roh und α_rot > α_n-Gramm um mehr als Rauschen.
- **Vorab-P:** 40 % (Rotation hilft α: 70 %; gegen n-Gramm bei Text: nur 40 %). Nachbehandlung, Frühregime-immun, aber "Zeit" bleibt unbelegt.

## C07 z²→GELU-Übergabe (Frühregime-Regel angewandt)
- **Bestehend:** Aktivierungs-Homotopien und Curriculum-Aktivierungen (G); Primer (ReLU²); f1-Daten: z² − GELU = −0,19 / −0,08 / **+0,05** (d = 512, T = 1000/4000/16000).
- **Mechanismus:** a(z) = (1−λ)·z² + λ·GELU(z), λ von 0 nach 1 über die ersten 20–30 % der Schritte; danach reines GELU. Kein Zusatz an FLOPs oder Bytes.
- **Warum 1+1>2 (Bedingung):** z² gewinnt früh, GELU gewinnt spät. Die Übergabe nimmt den Frühvorteil mit und vermeidet den Spätverlust. Echtes "1+1>2" nur, wenn der Frühvorteil Spuren hinterlässt (bessere Gewichte als Startpunkt), die GELU von Null nicht erreicht. Minimum "gleich dem besseren der beiden" ist fast sicher.
- **Härtester Gegner:** konstantes GELU und ReLU² (Primer, f1: ≈ GELU, Gate), mit **Lernrate je Arm und T** (f1: die v3-Rate wurde nur bei T = 4000 gesucht, das Kippen kann Artefakt sein); Lernraten-Aufwärmphase länger.
- **Billigster Test:** Byte-LM d = 256, T = 1000 / 4000 (T = 16000 nur nach Freigabe, wegen Kosten), 4 Arme, 3 Tuning-Seeds. Kriterium: Übergabe schlägt beide Konstanten bei T = 4000 um > 2 σ. Läuft als zusätzlicher Arm im Skript von C05.
- **Vorab-P:** 20 % für "schlägt beide", 60 % für "mindestens der bessere". Wert: macht aus dem inkonsistenten z²-Signal eine saubere Aussage ("z² als Anlauf").

## C06 BitNet b1.58 mit Rotation vor der Ternärisierung
- **Bestehend:** BitNet b1.58 (absmean-Ternär, QAT von Grund auf); BitNet v2 rotiert nur die **Aktivierungen**; QuaRot/SpinQuant rotieren fertige Modelle (PTQ).
- **Mechanismus:** Latentgewicht in rotierter Basis (W' = W·Hᵀ), ternär in W', Aktivierungen online durch Hx rotiert (n log n), A4. Ternär-Gitter 1,58 Bit trifft eine Gaußverteilung (Skala = Mittelwert).
- **Warum 1+1>2:** Ternärisierung ist am gerechtesten für gleichförmige Beträge; Rotation liefert sie. Gleiche Basis für Gewicht und Aktivierung ⇒ die Rotation wird einmal bezahlt und wirkt zweimal.
- **Härtester Gegner:** BitNet b1.58 und BitNet v2 im selben Skript (Standardrezept); kleines fp16-Modell gleicher Bytes; **T-ternär-butterfly kollabierte**, daher Pflicht: Latentgewichte + STE prüfen, sonst ist "QAT fehlt" der Befund.
- **Billigster Test:** Byte-LM d = 128, Arme ternär roh/rot, W1,58 A8/A4, T = 600 / 2400; Gegner klein-fp16. Kriterium: rot ≥ roh bei beiden Längen und ≥ klein bei gleichen Bytes.
- **Vorab-P:** 30 %. Risiko: Beim QAT von Grund auf passen sich Latentgewichte dem Gitter an; die PTQ-Gewinne der Rotation (R1) übertragen sich nur teilweise (Frühregime-Muster: Rotationsvorteil schrumpft mit mehr Training).

## C08 Rotiertes 2-Bit-Grundgewicht + VeRA-/Fastfood-Adapter
- **Bestehend:** VeRA (einmal Zufallsmatrizen A, B **geteilt über Schichten, eingefroren, aus Seed**, trainiert nur Skalenvektoren, ~10× weniger Parameter als LoRA, [arXiv 2310.11454](https://arxiv.org/abs/2310.11454)); QLoRA/LQ-LoRA (quantisiertes Grundgewicht + Adapter, G); K04.
- **Mechanismus:** Gewicht W ≈ Q(rot) (2 Bit, 0-Byte-Buch) + Adapter diag(b)·A_seed·diag(d)·B_seed, oder QIRE-Variante Fastfood: Adapter = Hᵀ·diag(b)·H·diag(d) (0 Byte Matrizen, 2d Zahlen).
- **Warum 1+1>2 (zweifelhaft):** Die Adapterzahlen sind winzig, also bleibt die Gesamtbilanz bei 2 Bit + ε. Sie soll Fehler korrigieren, den das Gitter lässt. **Aber Achtung:** Rotation macht den Rundungsfehler isotrop (weiß), und weißer Fehler ist **nicht niedrigrang korrigierbar**. Hier heben sich die Teile eher auf. Dafür gibt der Adapter Fein-Anpassung an die Aufgabe, nicht Fehlerkorrektur.
- **Härtester Gegner:** QLoRA-artig mit **nicht** rotiertem Grundgewicht + echtem LoRA gleicher Bytes (dort ist der Fehler strukturiert); rot ohne Adapter; E6 (nur Skalen und Bias nachtrainieren, 0 Zusatzbytes).
- **Billigster Test:** Byte-LM Basis auf T = 600 / 2400, quantisiert, dann 300 Schritte Anpassung an verschobene Teilmenge; Arme roh+LoRA, rot+VeRA, rot+Fastfood-Adapter, rot+E6. Messen: Fehlerrückgewinn pro Adapter-Byte. Kriterium: rot+Adapter > roh+LoRA bei gleichen Gesamtbytes.
- **Vorab-P:** 30 % (Sinn als Fehlerkorrektur: 15 %, als billige Domänenanpassung: 40 %).

## C10 MoE mit geteiltem 2-Bit-Grundgewicht, Experten als Diagonalpaare
- **Bestehend:** MoE; LoRA-MoE/geteilter Grundexperte + Niedrigrang je Experte (z. B. SLBF mit gemeinsamen Niedrigrang-Basen, [arXiv 2610.09342](https://arxiv.org/html/2610.09342)); DeepSeek-geteilte Experten; VeRA.
- **Mechanismus:** E Experten teilen ein rotiertes 2-Bit-Grundgewicht W; Experte e = W + diag(b_e)·A_seed·diag(d_e)·B_seed (A, B teilen alle, aus Seed). Router wählt top-1 oder top-2. Bytes ≈ 1 Experte + E·(2d+r) Zahlen; FLOPs wie ein dichter Experte plus kleiner Zusatz.
- **Warum 1+1>2:** MoE ist ein **Speicher**problem (alle Experten müssen im Speicher liegen); Rotation+2 Bit senkt den Grundpreis um ≈ 8×, die Seed-Adapter machen den Preis je Experte fast null. Der Router liefert den Zwang zur Spezialisierung, den K01 (Schichten) nicht hat.
- **Härtester Gegner:** Dichtes Modell gleicher Gesamtbytes; MoE mit E kleinen unabhängigen Experten (gleiche Bytes); LoRA-Experten; Experte ohne Adapter (Router sinnlos?).
- **Billigster Test:** Byte-LM d = 128, E = 4, T = 600 / 2400 (MoE-Router lernen spät: Zwei Längen entscheiden); Maß bpc gegen Gesamtbytes und aktive FLOPs. Lastverteilung mitmessen (Kollaps auf einen Experten ist die typische Fehlerform).
- **Vorab-P:** 25 %. Risiko: Experten fast gleich, Router zerfällt; 4-Lagen-Byte-LM zu klein für Spezialisierung.

## C11 Fastfood-Sz-Student + Distillation
- **Bestehend:** Sz (Fastfood+z², 0,5 % der FFN-Parameter, 0,015–0,024 bpc hinter voller FFN, f4); Distillation klein←groß (DistilBERT usw., G).
- **Mechanismus:** Lehrer D_gelu (volle FFN), Schüler Sz. KD auf Soft-Labels.
- **Warum 1+1>2:** Der Boden bei 1,6 MB/10 Epochen ist teils **Überanpassung** (f3 §3.3). Soft-Labels des Lehrers wirken wie Regularisierung/Datenvergrößerung und heben genau diesen Boden. Sz mit wenig Freiheitsgraden profitiert wegen geringer Kapazität stärker als ein dichter Schüler gleicher FLOPs.
- **Härtester Gegner:** Df (gleiche Matmul-FLOPs) **mit** KD; Sz ohne KD; Trainings-FLOP-Zählung mit Lehrer-Vorwärtsrechnung (≈ +33 %).
- **Billigster Test:** d = 256, T = 1000 / 4000, Lehrer auf gleicher Länge vortrainiert (separat zählen). Arme: Sz, Df, Dk (Parameter-gleich) × KD an/aus. Kriterium: Abstand Sz−Df schrumpft durch KD stärker als Abstand Dk−Df.
- **Vorab-P:** 25 %. Eher Bytes-Aussage.

## C13 Mamba/RWKV-Verwandte + rotierte 2-Bit-Gewichte
- **Bestehend:** Mamba/RWKV (konstanter Zustand statt wachsender KV-Cache); Quamba (statisch 8 Bit, Hadamard auf Ausgaben, 1,72× Latenz Orin Nano, −0,9 % Genauigkeit, [arXiv 2410.13229](https://arxiv.org/pdf/2410.13229)); Quamba2 (W4A8, Hadamard in Projektionen verschmolzen, [arXiv 2503.22879](https://arxiv.org/html/2503.22879)).
- **Mechanismus:** Rekurrentes Modell: über 90 % der Parameter sind Ein-/Ausgangsprojektionen (normale Matrizen, rotierbar), der Zustand ist klein und bleibt fp. Speicher bei Inferenz = Gewichte allein.
- **Warum 1+1>2:** Rotation+2 Bit senkt die **Gewichts**-Bytes, SSM beseitigt den **wachsenden** Cache; beide greifen verschiedene Speicherterme an (Gewicht, Kontext). Für lange Kontexte bleibt die Gesamtbilanz konstant.
- **Härtester Gegner:** Transformer + C04 (Cache-Rotation) bei gleichen Gesamtbytes und Kontext 4096; I-049 zeigt, dass LRU-Strukturen bei T = 16000 nicht vorn sind (−0,055).
- **Billigster Test:** Byte-LM flach vs. LRU-Block, T = 600 / 2400, je roh/rot 2 Bit; Maß: Bytes (Gewichte + Cache bei Kontext 128/1024/4096) gegen bpc. Kriterium: LRU-rot schlägt Transformer-rot ab einem Kontext N*, das aus der Rechnung kommt.
- **Vorab-P:** 20 %. Hauptgegner: SSM-Qualitätsboden; Quamba nennt die Rekurrenz als quantisierungsempfindlich, ihr Zustand muss fp bleiben.

## C15 Rotierter Trainingszustand: QAT-Latent und Adam-Momente in einer Basis
- **Bestehend:** 8-Bit-Adam (Dettmers), 4-Bit-Zustände (Li et al. 2023), K07/K08, E3 (QAT-Paket).
- **Mechanismus:** Gewichte, Gradienten und m leben in derselben rotierten Basis; ein Hadamard-Aufruf je Schritt liefert gleichzeitig Gitter für Latentgewicht (4 Bit) und für m (4 Bit). Trainings-Speicher ≈ 1/4.
- **Warum 1+1>2:** Eine Rotation (n log n) trägt zwei Quantisierer. Gradient in rotierter Basis ist ebenfalls gaußisch.
- **Härtester Gegner:** Adam ist koordinatenweise; Rotation zerstört die Bedeutung der Koordinaten (K08: Adam in gedrehter Basis ist **anderer Optimierer**, bekanntes Problem). 8-Bit-Adam roh + fp32-Master (Standard) ist stark und einfach.
- **Billigster Test:** Byte-LM d = 128, T = 600 / 2400; Arme: adam32, b8, b4roh, b4rot nur m (v roh), plus QAT-Latent 4 Bit rot/roh. Maß: bpc gegen Trainingsbytes.
- **Vorab-P:** 20 % (nur für m: 40 %, v rotiert: 10 %).

## C09 MoE-Experten als Vorzeichen-/Dreh-Schlüssel eines geteilten Gewichts
- **Bestehend:** K01 (Schlüssel-Superposition, Cheung 2019 PSP); MoE; ALBERT. **Kein Treffer** für "Experten = Vorzeichenvarianten eines Experten" in der Suche (eine Suche, G).
- **Mechanismus:** E Experten haben W_e = D1_e·W·D2_e (feste Zufallsvorzeichen) oder W_e = D1_e·H·D2_e·W; Router wählt. Bytes = 1 Experte + E·2d Bit.
- **Warum 1+1>2:** Wenn es klappt, ist es MoE zum Preis eines dichten Experten. Aber:
- **Härtester Gegner (sehr hart):** Vorzeichen-Diagonalen ändern nur die **Basis**, nicht die Funktion: y_e = D1_e W D2_e x ist bis auf gespiegelte Koordinaten dieselbe lineare Abbildung, ihre Unterschiedlichkeit liegt allein in der Nichtlinearität danach. Ohne Hadamard dazwischen kaum Vielfalt; ALBERT-Teilung verliert gerade bei FFN am meisten.
- **Billigster Test:** K01-Skript + Router (E = 4), T = 600 / 2400; Kontrollen: geteilt ohne Schlüssel mit Router (Router-Nutzen?), kleine unabhängige Experten gleicher Bytes. Kriterium: Router+Schlüssel > geteilt+Router (beides Bytes-gleich).
- **Vorab-P:** 15 %. Gut als Negativ-Befund für das Leitgedanken-Gesetz (Vorzeichen allein genügen nicht).

## C14 Frühausstieg als Entwurf (Self-Speculation) mit rotiertem 2-Bit-Teilstapel
- **Bestehend:** Frühausstieg (w6: nur +0,4 ± 0,2 pp, rot); LayerSkip/Self-Speculation (Entwurf = erste Schichten, Cache wiederverwendet, G); EAGLE.
- **Mechanismus:** Entwurf = erste L/2 Schichten, 2 Bit rotiert, eigener kleiner Ausgangskopf; Prüfer = ganze Tiefe, nutzt den K/V-Cache des Entwurfs.
- **Warum 1+1>2:** Frühausstieg ist als Qualitätsverfahren schwach, als **Entwurf** zählt nur die Annahmerate (Ausgabe bleibt exakt). Die 2-Bit-Teilschichten halbieren Entwurfsbytes nochmals; Cache-Wiederverwendung spart Prüfer-Rechnung.
- **Härtester Gegner:** n-Gramm-Entwurf (0 Byte); C03 (gleicher Entwurf aus Bit-Ebenen, kein Zusatzkopf); Medusa-/EAGLE-Kopf.
- **Billigster Test:** Byte-LM mit 8 Lagen, T = 600 / 2400, α für Entwurfstiefe 4; arme Kopf-Anpassung nur 200 Schritte. Maß α pro Entwurfs-Byte.
- **Vorab-P:** 15 % (Konkurrenz stark, Byte-LM mit Vokabular 256 hat leicht erratbare Token).

## C12 2:4-Beschneiden in gedrehter Basis (Konfliktkombination)
- **Bestehend:** SparseGPT/MaskLLM, SLiM (erst quantisieren, dann 2:4, dann Niedrigrang-Korrektur; +5,4–5,7 % Genauigkeit bei 2:4 + 4 Bit gegen Alternativen, [arXiv 2410.09615](https://arxiv.org/pdf/2410.09615)); PyTorch-Blog: 4 Bit + 50 % Spärlichkeit schlägt reines 2-Bit bei gleichen 8× ([pytorch.org](https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/)).
- **Mechanismus:** W' = W·Hᵀ speichern, 2:4 in W', Werte 3–4 Bit; Aktivierung online rotieren (Hardware-2:4 funktioniert in jeder festen Basis).
- **Warum es wahrscheinlich NICHT addiert:** Beschneiden gewinnt nur bei **schweren Rändern** (wenige große Gewichte tragen die Energie). Rotation erzeugt das Gegenteil (Gaußform, gleichmäßige Energie). Für Gaußquellen ist dichte Quantisierung bei gleicher Bitzahl nach der Raten-Verzerrungs-Theorie besser als Spärlichkeit; Rechnung: 2:4 mit 4-Bit-Werten ≈ 2 · 4 + 2,6 (Indexbits) = 10,6 Bit je 4 Gewichte ≈ 2,6 Bit/Gewicht, dichter Skalar mit 2,6 Bit ist für Gaußwerte besser. Rotation und 2:4 stehen im Konflikt; Reihenfolge SLiM (erst Beschneiden im Original, dann Quantisieren **ohne** Rotation) bleibt vermutlich überlegen.
- **Härtester Gegner:** Original-Basis-2:4 + 4 Bit (SLiM-Rezept); dichte Rotation 2,6 Bit (E10 mit Entropiecodierung).
- **Billigster Test (3 min):** Fertig trainiertes MNIST-MLP und Byte-LM (T = 600 / 2400), Fehler der Gewichtsmatrizen: Frobenius-Fehler und Genauigkeit bei gleichen Bytes für {2:4 roh, 2:4 rot, dicht 2,6 Bit rot, dicht 2,6 Bit roh}. Vorab als reine Verzerrungsrechnung auf Zufallsmatrizen (Gauß vs. Laplace-Ränder) möglich.
- **Vorab-P:** 10 %. Wert: schärft die Aussage der Arbeit ("Rotation passt zu Quantisierung, nicht zu Beschneiden") und ist eine gute Jury-Antwort auf "Warum nicht alles kombinieren?".

---

## Querbefunde
1. **Paare, die sich addieren (A):** Rotation × Quantisierer/Zufallsbuch/4-Bit-Aktivierung (C01, C05, C06), Rotation × Cache-Teilung (C04, multiplikativ), Rotation × Bit-Ebenen (C03). Gemeinsam: Zielverteilung gaußförmig und Nachbehandlung oder billige Mitlauf-Rotation.
2. **Paare, die sich aufheben (X):** Rotation × 2:4, Rotation × Niedrigrang-Korrektur (C08), Rotation × koordinatenweises Adam (C15), Vorzeichen-Schlüssel × MoE (C09). Gemeinsam: das zweite Verfahren nutzt Struktur, die Rotation löscht, oder die Schlüssel ändern nur die Basis.
3. **Bytes vor FLOPs:** Kein Eintrag verspricht FLOP-Gewinn ohne Gegenrechnung. C03/C14 sparen Zeit nur über Annahmerate (Modell). C10 spart aktive FLOPs, kostet aber Router.
4. **Frühregime-Regel greift bei:** C02, C04 (Teilung), C06, C07, C10, C11, C15 (Training mitlaufend). **Immun:** C01, C03, C05 (nur Aktivierungsteil), C12, C14.
5. **Neuheit:** Fast alle Teile sind zitiert (BitNet v2, QuIP#, SeedLM, VeRA, QSpec, SLiM). Beitrag = vorregistrierte Gleiche-Bytes-Messung der **Paarung** mit der Zufalls-Orthogonalen als Kontrolle. Das ist laut Malte (07.10.) genug.
6. **Nicht geprüft:** Zahlen aus Suchtreffern sind Abstract- oder Sekundärangaben; vor Verwendung in der Arbeit am Original prüfen. Eintrag "(G)" nicht belegt.

## Quellen (Suche 28.10.)
- [BitNet v2, arXiv 2504.18415](https://arxiv.org/pdf/2504.18415) · [Quamba, arXiv 2410.13229](https://arxiv.org/pdf/2410.13229) · [Quamba2, arXiv 2503.22879](https://arxiv.org/html/2503.22879)
- [VeRA, arXiv 2310.11454](https://arxiv.org/pdf/2310.11454) · [SLBF (MoE), arXiv 2610.09342](https://arxiv.org/html/2610.09342)
- [SLiM, arXiv 2410.09615](https://arxiv.org/pdf/2410.09615) · [PyTorch-Blog 2:4](https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/)
- [BitNet Distillation, arXiv 2510.13998](https://arxiv.org/pdf/2510.13998) · [BitDistiller, arXiv 2402.10631](https://arxiv.org/pdf/2402.10631)
- [QSpec, arXiv 2410.11305](https://arxiv.org/html/2410.11305v3) · [ML-SpecQD, arXiv 2503.13565](https://arxiv.org/pdf/2503.13565)
