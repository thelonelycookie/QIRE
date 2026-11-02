# Rohliste Schlüsse und Ideen aus WISSEN/felder (Querschnitt-Rohmaterial)

Quelle: Abschnitte „Schlüsse und neue Wege (Agent)“, „Runde 2“/„Nachtrag“, „Bezug zu eigenen Befunden“, „Lücken“ aller 22 Felddateien. Belege sind in den Feldern nur aus Abstracts, ungeprüft. Runde-2-Vorschläge sind mitgezählt. Quantenbezug: d = direkt, a = Analogie, k = keiner.

## 1. Tabelle (79 Punkte)

| ID | Feld | Kurzfassung | Belege | Langweiligste Erklärung | Q | Bezug QIRE-Befund |
|---|---|---|---|---|---|---|
| R-01 | F1 | Bitflip-Robustheit der Rotation als neue QIRE-Achse testen | 2603.16382, 2606.02430 | Rotation macht nur Ausreißer kleiner, Clipping leistet dasselbe | a | Z3 |
| R-02 | F1 | Vor jedem Rotationsanspruch festen Gegner Clipping + Gewichtsskalierung einplanen | 2006.13977, 2411.18993 | Gleichstand wie in Z3 | k | Z3 |
| R-03 | F1 | Hadamard-Zeilen (Walsh) als Codewörter für Ausgabe-Codierung (ECOC) | 2004.10700, NeurIPS-2024-ECOC (url) | Bekannte ECOC mit anderem Namen | a | – |
| R-04 | F2 | Mischung + Betrag ist strukturell ein Streunetz mit Hadamard statt Wavelet; als Rahmen nutzen | 1203.1513, 1312.5940, 2409.12924, 1602.01166 | Reine Faltungs-/Mischungs-Nichtlinearität | a | I-038 |
| R-05 | F2 | Haar-Pyramide statt Hadamard testen, gleiche Parameter, gleiche Auslese | WaveletGPT, Evenbly-White, 2203.03689 | Kein Unterschied (Z3-Muster) | a | Z3 |
| R-06 | F2 | MERA-Schicht und Hourglass in die Vergleichstabelle aufnehmen (stärkster Gegner) | – (Feldlücke) | Hourglass gewinnt durch Rechenverteilung, nicht Mischung | k | I-049 |
| R-07 | F2 | Hadamard mit lernbarer Diagonale (Fastfood-artig) als Mittelweg zwischen fest und gelernt | 2107.09539 | Bessere Initialisierung, nicht Struktur | a | I-045, I-038 |
| R-08 | F3 | Hadamard ist austauschbar: jede gut mischende orthogonale Matrix reicht | 2601.21069, 2604.04440, 2307.13304 | Jede orthogonale Mischung tut es | k | Z3, I-038 |
| R-09 | F3 | Parametrisierungs-Gegner: feste Basis + K trainierte DCT-Koeffizienten neben Fastfood | 2604.04440, 1506.04449 | Rangflexibilität, nicht Struktur | k | I-045 |
| R-10 | F3 | Überkomplette Hadamard-Mischung (Expansion + Betrag) als Frame mit Rahmengrenzen messen | 2103.05804, 1504.05487 | Kohärenzmaß sagt nichts über Genauigkeit | a | – |
| R-11 | F3 | Fastfood gegen DCT-Koeffizienten selbst messen, Literatur vergleicht beide nie | 2604.04440, 2502.06820 | Bei gleicher Parameterzahl gleich | k | I-045 |
| R-12 | F4 | FlyHash (spärliche Zufallsexpansion + Top-k) als Pflicht-Gegner zu Hadamard·Diagonal + Quadrat | 2001.04907 | Beides nur zufällige Merkmale | a | I-038 |
| R-13 | F4 | Effizienz in Parametern, FLOPs und Speicherzugriffen ausweisen, nicht nur Operationen | 2409.08290 | Zählfehler | k | I-045-v4, I-047 |
| R-14 | F4 | ca. 4–5 Bit als biologisch plausible Zielpräzision zitieren (Hinweis, kein Beleg) | eLife 10778 | Messgrenze statt Kapazität | k | – |
| R-15 | F4 | Bit-Budget als Spanne 1–5 Bit zitieren statt Punktzahl | 1602.04129, eLife 10778 | Unterschiedliche Messmodelle | k | – |
| R-16 | F5 | Rauschtraining (R1-Typ) als Regularisierer in QIRE-Training testen, Gegner Standard-QAT | 2306.08553, 2311.14114 | Übliche Regularisierung | a | Z3 (R1) |
| R-17 | F5 | Lindblad-artige Schicht (Mischung + Dephasierung) als formale Brücke zur Born-Auslese | 2307.12293 | Dämpfung/Dropout | d | I-038 |
| R-18 | F5 | Nicht behaupten, dass Rauschen hilft, ohne Gegner mit Clipping/Gruppenskalen | 2409.17734 vs 2207.07924 | Gewinne stammen aus schwachen Baselines | k | Z3 |
| R-19 | F6 | Negativbefund ist Befund: Physik-Simulationsmethoden kaum als ML-Effizienz übertragen | 2205.07408, NeurIPS-2024-HMC | SGD ist für große Netze schlicht optimal | k | I-045, I-047 |
| R-20 | F6 | Budgetprinzip „Kosten ∝ Magie/Verschränkung“ (Rang, T-Zahl je Schicht) als Messgröße | 1808.00128, 2409.01706 | Parameter-/Rang-Budgetkurve, aus K7 bekannt | d | I-038, I-045 |
| R-21 | F6 | DMRG-Sweep für adaptive Raten als einzige Übertragung mit Substanz, Vortest auf Tuning-Seeds | 2211.14657 | Generische Rangwahl | a | I-047 |
| R-22 | K1 | Rotation lohnt nur bei grober Skala (pro Kanal, g≥32 FP-Gitter) oder starken Ausreißern | 2509.23202, Z3 | Gruppenskalen isolieren Ausreißer selbst | a | Z3 |
| R-23 | K1 | Rotation gegen GPTQ-gerundete Gruppenquantisierung testen, nicht gegen RTN | 2505.11695, 2607.07964 | GPTQ-Rundung allein erklärt Gewinne | k | Z3 |
| R-24 | K1 | Gelernte Rotation (OptRot) vs feste Hadamard bei gleichen Bytes, billiger Test | 2512.24124 | Mehr Freiheitsgrade, nicht Struktur | a | Z3, I-038 |
| R-25 | K2 | Rotation gewinnt nur bei grober Granularität oder ungünstigem Format (Zweierpotenz-Skala), bei g16/g32 neutral bis schädlich | 2509.23202, 2511.04214 | Gruppenskala fängt dieselben Ausreißer lokal ab | k | Z3 |
| R-26 | K2 | Vermutlich neuer Beitrag: Lloyd-Gitter ohne Rotation vs Hadamard auf Aktivierungen/KV isolieren | – (Lücke) | Test klein, andere isolieren es nicht, weil uninteressant | a | Z3b |
| R-27 | K2 | Blockrotation (Block = Gruppengröße) als Kombi-Arm gegen g16-Lloyd | 2607.27694 | Ergebnis ≈ g16 | a | Z3 |
| R-28 | K2 | Great-Inversion-Theorie als Erklärungsrahmen, warum es keinen universellen Sieger gibt | 2608.25188 | Worst-Case-Theorie | k | Z3 |
| R-29 | K2 | (R2) Arm „g16-Lloyd-Aktivierungen“ mit LUT-Kosten statt nur PPL ausweisen | 2507.23035, 2604.04013 | LUT-Kosten fressen kleinen Gewinn auf | k | Z3b |
| R-30 | K3 | Rotation lohnt vor allem ohne feine Gruppenskalen | 2506.15689, 2608.25188, Z3 | Gruppenskalen machen Rotation redundant | k | Z3 |
| R-31 | K3 | Gelernte Rotation ≈ Hadamard-Start: Hadamard fest lassen, Budget in Skalen/Bias | ReSpinQuant, HARP, BASE-Q | Optimierung bleibt im Becken des Starts | a | I-038, Z3 |
| R-32 | K3 | Permutation + Hadamard (DuQuant-Zickzack) gegen reinen Hadamard, gleiche Bytes | DuQuant | Ohne massive Ausreißer ändert Permutation nichts | a | Z3 |
| R-33 | K4 | Hebelkraft liegt im Block-Gitter, nicht in der Rotation (bei festem Skalenformat) | Z3b, 2608.25188, 2603.11021 | Rotation glättet nur Ausreißer | a (gering) | Z3b |
| R-34 | K4 | E8-Gitter vs Lloyd-Skalar bei gleichen Bits, mit/ohne Hadamard (Erwartung ≤ ~0,1 Bit) | 2606.23406, 2402.04396 | Gewinn verschwindet gegen Gruppenskala + Entropiecodierung | k | Z3b |
| R-35 | K4 | Winkelcodebuch aus Clifford-/Kerdock-Phasen (Z4/Z8) für Paare nach Hadamard | 1904.07842, 2603.27467 | Gleichförmige Winkelquantisierung, Z4 nur 4 Winkel | d | I-038 |
| R-36 | K4 | Dekodierkosten (Bytes gelesen) als Bewertungsachse: Trellis/Gitter vs Tabelle | 2609.02652 | Reine Kernel-Eigenschaft | k | – |
| R-37 | K4 | (R2) Tok/s auf CPU und Bytes neben Bits; Gitter/Trellis nur mit LUT-/Bitshift-Kernel gegen Lloyd testen | 2407.00088, 2507.23035 | Skalar-Lloyd + Gruppenskala liegt auf CPU vorn | k | – |
| R-38 | K5 | I-047 mit muP-Skalierung wiederholen oder BTT als Arm ergänzen | 2406.06248 | I-047 hatte falsche LR/Init, 0,06 bpc = Rauschen + Fehlkalibrierung | a | I-047 |
| R-39 | K5 | Regel „viele Parameter pro FLOP, voller Rang“ als Entwurfsprinzip | 2410.02117 | Monarch/Kronecker-Sparsität, nichts Quantenhaftes | a | I-045, I-047 |
| R-40 | K5 | Adaptive Bonddimension per DMRG-Sweep statt fester χ | 2211.14657 | Generische Rangauswahl (AdaLoRA-artig) | d | I-047 |
| R-41 | K5 | Post-Training-Tensorkompression hat engen Nutzen, nicht als QIRE-Vorteil verkaufen | 2606.03465 | – | k | I-047 |
| R-42 | K6 | „Quanten-inspiriert“ ohne fair getunte Gegner hält nicht; Z3 in guter Gesellschaft | 2607.09905, 2606.18324, 2610.00841 | Nullergebnisse werden eher veröffentlicht, wenn spektakulär | a/d | Z3, I-038 |
| R-43 | K6 | SWave-Lehre: Ablationen offen zeigen, was nicht trägt | 2606.18324 | Einzelbericht eines Teams | k | I-039-v2, Z3 |
| R-44 | K6 | Unitäre Phasenrekurrenz (PMNet) als Gegner/Erweiterung für Sequenzaufgaben, mit gleich großem reellem Gegner | 2605.13370 | Gradientenstabilität, orthogonaler reeller Block reicht | a | I-049 |
| R-45 | K6 | Dequantisierung begrenzt Ehrlichkeitsformel: „quanten-inspirierter Vorteil“ polynomiell, nicht exponentiell | 2610.42069 | Worst-Case-Aussage | d | I-038 |
| R-46 | K7 | Fairer Test fehlt: Hadamard-MLP vs Low-Rank-FFN bei gleichen Parametern/FLOPs | 2407.09835, 2406.06248, 2406.11307 | Low-Rank gleich gut oder besser | a | I-045-v4 |
| R-47 | K7 | Compute-Gleichheit entscheidet; nur BTT-artig gewinnt bei Compute; Parameter ≠ FLOPs | 2406.06248 | Parameterzahl ≠ FLOPs | k | I-045-v4, I-047 |
| R-48 | K7 | WHT-Ausgabeprojektion als billigen Zusatz testen (Gewicht sparen, dann quantisieren) | 2603.08343 | Nur bei kleinen Modellen, Selbstbericht | a | – |
| R-49 | K8 | Rotation wirkt im Training am Wgrad (Ausreißergradienten), nicht am Gewichtsgitter | 2605.09825, 2509.25149 | Gradienten haben schwerere Ränder | a | Z3 |
| R-50 | K8 | Gitterform (E2M1 vs uniform) entscheidet mit; Rotationsgewinne können Gitterartefakte sein | 2606.20381 | Schlechtes Gitter | k | Z3b |
| R-51 | K8 | Byte-fairer Test ternär-von-Grund-auf gegen ParetoQ-4-Bit im Kleinmodell | ParetoQ | Skalierungsregime | k | – |
| R-52 | N1 | QIRE-Vergleiche über Reihe von Trainingslängen messen (QiD-Regime) statt eines Punktes | 2411.17691, 2401.00448 | Schaden folgt nur der Gewichtsverteilung | k | Z3, I-045-v3 |
| R-53 | N1 | Hadamard-Glättung könnte Token-Exponenten des Quantisierungsschadens senken | 2503.19206, 2605.23901 | Durch Gruppenscales ersetzbar | a | Z3 |
| R-54 | N1 | Bitzahl-Debatte: Größe in Bits auftragen, 2/3/4 Bit klein neu fitten | 2212.09720, 2502.02631 | Hängt am QAT-Budget | k | – |
| R-55 | N2 | I-049 mit Hybrid-Literatur vereinbar; nächster Test flach + wenig Attention/Gating (Griffin) | 2406.07887, 2402.19427 | Frühregime-Effekt, ungetunte LR | a | I-049 |
| R-56 | N2 | Gated DeltaNet als stärkster Rekurrenzgegner vor jeder LRU-Neuheitsbehauptung | 2412.06464 | Bessere Speicherregel/Optimierer | k | I-049 |
| R-57 | N2 | MoE nicht einplanen, solange kein Kleinmodellbeleg | 2402.07871 | Router-Overhead dominiert klein | k | – |
| R-58 | N3 | Rotationsvorteile bei QAT mit Destillation und gleichem Rechenbudget neu messen | 2509.22935, 2601.14888 | KD + mehr Tokens gleichen Gewinn aus | a | Z3 |
| R-59 | N3 | Gemeinsames Gesetz Destillation × Bits klein messen | 2502.08606, 2505.14302 | Effekte addieren sich | k | – |
| R-60 | N3 | Kanalsicht: Schüler-Kapazität in Bits begrenzt destillierbare Information, R(D) prüfen | 2411.07483 | Bekannte Kapazitätsgrenze neu formuliert | a | – |
| R-61 | N3 | Pruning+KD (Minitron) gegen Quantisierung+KD bei gleichen Bytes | 2407.14679 | Pruning gewinnt bei hoher, Quantisierung bei niedriger Bitbreite | k | – |
| R-62 | N3 | (R2) Kleinmodell-KD: Logits online, Kosten gegen CE-Training ausweisen | 2410.16215, 2503.16870 | – | k | – |
| R-63 | N4 | Wanda/SparseGPT+4-Bit-Gegner mit eingerechneter Maske aufnehmen | 2301.00774, 2410.09615 | Sparsität bei Kleinmodellen nicht billig | k | Z3 |
| R-64 | N4 | Rotation + Aktivierungssparsity testen, Rotation glättet Ausreißer | 2507.01299 | Klassischer Glättungseffekt | a | Z3 |
| R-65 | N4 | 2:4 nur als Bytes-Rechnung, nicht als CPU-Geschwindigkeit | 2102.04010 | Hardwareabhängigkeit | k | – |
| R-66 | N5 | Energie folgt Bytes, nicht Ops | 2403.14123, 2503.06862 | Decoding bandbreitengebunden | k | I-038 |
| R-67 | N5 | Hadamard-Betrieb ≈ 0,3 % der Ops, aber Speicherpässe; Wallclock auf Ryzen messen | eigene Rechnung, 2412.08832 | Fusioniert nicht messbar | a | Z3 |
| R-68 | N5 | Fairer Energievergleich nur gegen LUT-Kerne (T-MAC/bitnet.cpp) mit Joule + Wallclock + Bytes | 2407.00088, 2410.16144, 2505.06371 | Gewinne ggü. FP16 sind kleine Bitbreite | k | – |
| R-69 | N5 | Op-Zählmodelle (L-Mul) nur als Sekundärspalte | 2410.00907 | Modell ohne passende Hardware | k | I-045-v4, I-047 |
| R-70 | N6 | QIRE-Rotation als SeedLM-Variante testen (Seed-Struktur + wenige Koeffizienten) | 2410.10714, 1408.3060, 1502.07514 | Gleich gut wie SeedLM, beides Zufallsbasis | a | Z3, I-038 |
| R-71 | N6 | Zufallsbasis muss nicht gelernt werden (A fest in LoRA reicht) | 2402.16842, 2310.11454 | Überparametrisierung | a | Z3 |
| R-72 | N6 | Supermaske auf Zufallsgewichten (Seed + Bitmaske) als Kompressionsformat vergleichen | 2603.08914, 2101.06475 | Maskenbits kosten mehr als 4-Bit-Gewicht | a (gering) | – |
| R-73 | N7 | ReLU² statt z² in QIRE-Auslese testen, könnte CIFAR-Verlust erklären und Sparsität liefern | 2109.08668, 2503.16672 | Vorzeichenverlust bei z² = Symmetrie-Rauschen | a | I-042 |
| R-74 | N7 | SwiGLU als fairer Gegner: GLU bilinear, Born-Quadrat Spezialfall | 2002.05202 | Gleiche Funktionsklasse | a | I-039-v2 |
| R-75 | N7 | Aktivierungsgleichwertigkeit als Ergebnis zitieren statt Vorteil behaupten | 2102.11972 | Effekte unter Seed-Rauschen | k | I-039-v2, I-042 |
| R-76 | N7 | (R2) Test z² vs ReLU² vs \|z\| unter 4-Bit-Aktivierungen, Kurtosis messen; Init als Erklärung des z²-Verlusts | 2604.18128, 2601.22966, 2502.01247 | Init/Normalisierung | k | I-042 |
| R-77 | N8 | QIRE-Gewinne als Bit-Abstand zur Gauß-R(D)-Kurve angeben | 2602.05790, 2502.09720 | Gewichte nicht gaußisch | a | Z3 |
| R-78 | N8 | Kapazität 3,6 Bit/Parameter: Verlust bei 2–3 Bit erst bei ausgelasteten Modellen | 2505.24832, 2404.05405 | Kapazität betrifft Memorieren | k | – |
| R-79 | N8 | Holevo-Abschnitt in der Arbeit: warum QIRE keinen Quantenvorteil beansprucht | 1908.01364, 2101.02464, 2604.07639 | Zu allgemein ohne QIRE-Bezug | d | I-038 |

## 2. Cluster (feldübergreifend)

- **C-1 Rotation/Hadamard ersetzbar, lohnt nur bei grober Skala:** R-02, 08, 22, 25, 28, 30, 31, 53, 77. Neun Felder (F1, F3, K1–K3, N1, N6, N8) sagen dasselbe wie Z3: Gruppenskalen leisten, was Rotation leistet.
- **C-2 Pflicht zu starken fairen Gegnern pro Anspruch:** R-02, 06, 12, 18, 23, 42, 56, 61, 63, 68, 74, 46. Jedes Feld nennt seinen eigenen Gegner (Clipping, GPTQ, FlyHash, Wanda, Gated DeltaNet, SwiGLU, Low-Rank, LUT-Kerne). Ergibt eine Gegnerliste für die Arbeit.
- **C-3 Gitter-/Lloyd-Artefakt statt Rotationswirkung:** R-26, 29, 33, 34, 35, 37, 50. Z3b-Befund (Lloyd ohne Rotation ≈ Hadamard) wird von K2, K4, K8 gestützt; offen bleibt der isolierte Test auf Aktivierungen/KV.
- **C-4 Kostenachse: FLOPs, Bytes, Wallclock, CPU-Kernel statt Operationen:** R-13, 36, 37, 47, 62, 65, 66, 67, 68, 69. F4, K4, K7, N3–N5 verlangen dieselbe Messdisziplin (Joule + Wallclock + Bytes, LUT-Kerne).
- **C-5 Struktur-FFN (Fastfood/TT/Monarch/DCT) gegen dicht/Low-Rank bei gleichen FLOPs:** R-07, 09, 11, 38, 39, 40, 41, 46, 47, 48, 21. F3, K5, K7 liefern Gegner und Entwurfsregeln; DMRG-Rangwahl (R-21, 40) ist dieselbe Idee in F6 und K5.
- **C-6 Frühregime und Trainingslänge:** R-52, 53, 54, 55. N1 (QiD), N2 (I-049) und Methodenlehre des Laborbuchs ziehen dieselbe Linie: nur Vergleiche über mehrere Trainingslängen.
- **C-7 Born-Quadrat/Aktivierung:** R-12, 17, 73, 74, 75, 76. Quadrat als Squared-Activation/GLU-Spezialfall; Test z² vs ReLU² (I-042) und Quantisierungsrisiko (N7).
- **C-8 Ehrlichkeitsrahmen „klassische Wirkung“:** R-19, 20, 41, 43, 45, 75, 79. Dequantisierung, Holevo, Negativbefunde und offene Ablationen als eigener Abschnitt der Arbeit.
- **C-9 Bit-Budget und Grenzen:** R-14, 15, 54, 60, 77, 78. Bits statt Parameter als Achse, R(D)-Abstand, Kapazität, biologische Spanne 1–5 Bit.
- **C-10 Rauschen und Robustheit:** R-01, 03, 16, 17, 18. Bitflips, ECOC, Rauschtraining, Dephasierung; überall Vorbehalt „Gegner sind oft schwach“.
- **C-11 Fester/zufälliger Aufbau genügt (Gegner Fastfood, DCT, SeedLM, FlyHash):** R-07, 09, 11, 12, 31, 70, 71, 72. Ergänzt C-1 aus Sicht der Zufallsbasis.
- **C-12 Rotation im Trainings-/Gradientenpfad und mit KD:** R-24, 49, 58, 31. Anderer Einsatzort der Rotation als das Gewichtsgitter.
- **C-13 Rekurrenz/Phase/unitär:** R-44, 55, 56, 57. LRU, PMNet, Gated DeltaNet als Gegner für I-049-Linie.
- **C-14 Mehrskalen/Wavelet/Streunetz/Rahmen:** R-04, 05, 06, 10, 48. Haar-vs-Hadamard und MERA/Hourglass als offene Vergleiche.
- **C-15 Echte Quantenstruktur als Begründung (Clifford, Kerdock, Magie):** R-20, 35, 17. Einzige Punkte mit direktem Quantenbezug in der Struktur; Wirkung klassisch.

Stärkste Cluster (nach Zahl der Felder und Übereinstimmung mit Z3): C-1, C-2, C-4, C-5, C-3.

## 3. Stellen, an denen ein Feld einen eigenen Befund anzweifelt oder einschränkt

1. **K5, Z. 31 (I-047):** „Möglicher Widerspruch zu eigenen Einstellungen: Wurde die Lernrate/Initialisierung (muP-artig) für TT angepasst? Falls nicht, ist I-047 kein Beleg gegen TT, sondern gegen naive TT.“ Wiederholt in K5 Schluss 1 (Z. 47): „I-047 hatte schlicht falsche Lernrate/Init für TT, 0,06 bpc ist Rauschen plus Fehlkalibrierung.“
2. **N7, Z. 33 (I-042):** „Möglicher Widerspruch: QIRE fand z² schlechter als ReLU auf CIFAR (-3,18 pp, I-042), die Literatur (Primer, 2:4) sieht Squared-ReLU … als gut. Wahrscheinlich Unterschied zwischen z² (beidseitig) und ReLU² (einseitig …) … Hypothese, nicht geprüft.“ Dazu N7 Runde 2: „Init könnte den z²-Verlust (I-042) mit erklären (Hypothese, nicht geprüft).“
3. **N2, Z. 24 (I-049):** „Literatur stützt das nicht direkt, erklärt es aber … Im Frühregime (190 Schritte) sind Gewinne gewöhnlich Optimierungsvorteile.“ Schluss 1 (Z. 37): „Frühregime-Effekt und ungetunte Lernrate.“ (Einschränkung: I-049 gilt für reine LRU, nicht für Hybride.)
4. **K2, Z. 38 (Z3):** „Möglicher Widerspruch: Im Großmodell-Bereich mit massiven Ausreißern (LLaMA-3 70B) schlagen gelernte Transforms Gruppen-Baselines laut FlatQuant; Raven-Skala könnte das nicht abbilden.“
5. **K2 Z. 17 und Nachtrag:** „Unser Z3b (Lloyd ohne Rotation ≤ Hadamard bei 2 Bit) ist hier vermutlich neu, aber nur auf Raven-Skala.“
6. **K3, Z. 25 (Z3):** „Möglicher Widerspruch: SpinQuant/OSTQuant zeigen große Gewinne, aber gegen schwache Basis (zufällige Rotation, einfaches RTN); bei LLaMA-Skala, nicht Raven-Skala.“
7. **K4, Z. 40 (Z3/Z3b):** „Mögliche Spannung: QuIP#/HyperQuant zeigen Gewinn durch Hadamard+Gitter; unsere Z3 zeigt ihn gegen Lloyd/Gruppenskalen nicht. Unterschied: VQ im Block (8D+) nutzt Gitterform, nicht nur Rotation.“
8. **F5, Z. 31 (Z3):** „Mögliche Spannung: Rauschtraining-Gewinne (bis +9 %) stammen aus schwachen Baselines; unsere fairen Gegner sprechen für Vorsicht.“ (stützt eher, schränkt Übertragbarkeit ein.)
9. **F1, Z. 30 (Z3):** „… ein Rotationsvorteil gegen Bitflips [wäre] gegen Clipping/Gruppenskalen zu prüfen, nicht vorauszusetzen.“ (Grenze von Z3: Bitflips nicht gemessen.)
10. **K6, Z. 34 (I-038/Z3):** „Möglicher Gegenpol: [2605.13370] und [2608.03425] behaupten Vorteile unitärer Phasendynamik bei Langzeitgedächtnis … anderes Regime … kein direkter Widerspruch.“
11. **F3, Z. 33 (Z3/I-042):** „Möglicher Widerspruch: keiner gefunden; WHT-Schichten verlieren 2,2 Punkte (Pan) gegen volle Schichten, das ist die Grenze fester Transformationen.“ (Einschränkung der Aussage „feste Struktur reicht“, passt zu I-042 CIFAR −6,3 pp.)
12. **K1, Z. 22 und K8, Z. 19:** „Rotation hilft real bei grober Skala (g32 MXFP4) und im Gradientenpfad.“ bzw. „Rotation hilft im Gradienten-/Aktivierungspfad (Wgrad), nicht beim Gewichtsgitter“: schränkt Z3 ein (Rotation nicht überall wertlos).
13. **N1, Z. 26:** „QiD erklärt, warum Gewinne nur im Regime kleiner, lang trainierter Modelle sichtbar werden“ und Lücke „Wirkt Rotation auf den Token-Exponenten des Schadens? Unbekannt.“ (Z3 nur bei T=2400 gemessen.)
14. **N5, Z. 48:** „Hadamard-Betriebskosten kämen noch obendrauf, wären also ein reiner Nachteil, wenn die Güte gleich bleibt.“ (verschärft Z3.)
15. **K7, Z. 24:** „Z3-Muster (einfacher Gegner stark) passt … kein eigener Befund direkt zu K7 in dieser Welle geprüft.“ (Grenze der Abdeckung.)

## 4. Prüfung der Behauptungen über eigene Befunde

Zeilen: LABORBUCH = INSTITUT/LABORBUCH.md, Z3 = INSTITUT/recherche/zyklus/Z3_ergebnisse_raven.md.

| Behauptung (Feld) | Ergebnis | Beleg |
|---|---|---|
| I-042: z² 3,18 pp schlechter als ReLU auf CIFAR (N7) | **stimmt** mit Vorbehalt | LABORBUCH 1346: „H2 Born z² gegen ReLU: CIFAR KLASSISCH_BESSER −3,18“, Fashion BORN_BESSER +0,57; n=5. Replikation I-042-r (1443–1448) bestätigt nur H1/H3, H2 nicht repliziert. Einheit „pp“ steht in der Tabelle nicht, nur im Kontext von H1. Nebenbefund: I-047 H3 (2054) „Quadrat trägt“ gegen GELU, N7 erwähnt das nicht. |
| I-042: Nichtlinearitäten wechseln je Aufgabe (N7) | stimmt | LABORBUCH 1349 |
| I-047: TT-FFN 0,06 bpc schlechter als dicht (K5) | **stimmt** | LABORBUCH 2053: Df_gelu besser, −0,061 bpc [−0,065; −0,056]; Lk_z2 −0,059; Tabelle T=16000 TT_z2 2,148 vs Df_gelu 2,088 |
| I-047: „ohne angepasste Init/LR“ (K5, als Frage) | **teilweise falsch** | LR je Arm auf Tuning-Seed getunt (LABORBUCH 1855, 3 Raten, nur bei T=4000, übernommen für 16000; Vorbehalt dort selbst benannt). Init: Kerne Normal mit Varianz 1/(3·N_ein) wie nn.Linear (abteilungen/struktur_tt.py 97), also dichte-äquivalent, kein TT-spezifisches muP. Einwand „naive TT“ daher nur für Init/Skalierung berechtigt, nicht für LR. |
| I-045: Fastfood verliert bei gleichem Rechenaufwand (K5, F6) | **stimmt** | LABORBUCH 1691: Df_z2 besser als Sz −0,081 bpc; Lk_z2 −0,088; 1692: „Bei gleichem Rechenaufwand verliert die Fastfood-Struktur klar“ |
| I-045/I-047: keine Simulationsmethode mit FLOP-Vorteil (F6) | stimmt | LABORBUCH 1692, 1694 und 2056 („Kein Rechenvorteil bei gleichen FLOPs“) |
| I-049: Hourglass+LRU bei langem Training schlechter als flach (N2) | **stimmt** | LABORBUCH 2037: HGLRU − FF = −0,055 bpc bei T=16000, 0/10 Seeds; Tabelle 2028–2036 (HGLRU 2,132 vs FF 2,077) |
| I-049: Frühregime 190 Schritte (N2) | stimmt | LABORBUCH 1873 („nur 190 Schritte … Lernrate nicht getunt“), 2039 Deutung |
| I-049: Gewinne bei flach vs reine LRU (N2) | stimmt | LABORBUCH 2036: LRU 2,168, HG 2,194 beide schlechter als FF bei T=16000 |
| I-038: gelernte QIRE-Zelle ≡ feste Mischung (F6, K6) | stimmt | LABORBUCH 1234–1245: H-A/H-B/H-C GLEICHWERTIG; aber H-G Born² > \|z\| +0,35 pp auf MNIST (BORN_BESSER), von Feldern nicht erwähnt |
| I-039-v2: Gleichwertigkeit (N7) | stimmt | LABORBUCH 1309: Differenz −0,39 pp, TOST ±1 pp bestanden |
| Z3: Rotation ≈ gleichwertig gegen faire Gegner (viele Felder) | **stimmt, aber unscharf** | Z3 Kernaussage Z. 3–5: „fast nichts übrig, im Sprachmodell gar nichts“; Tabelle Z. 11–12: MLP +0,163 pp (0/20 grün), LM −0,115 bpc (Rotation schlechter, 0/20). Im LM also schlechter, nicht gleichwertig. Dateikopf Z. 1: Erkundung auf Tuning-Seeds, „kein Befund“. |
| Z3: Steigung 0,10 (rot) vs 0,91 (roh), g16 0,71 (F5, K2) | **stimmt** | Z3 Z. 16 |
| Z3b: Lloyd ohne Rotation 0,041 ≈ Hadamard 0,052 bei 2 Bit (K4, K2) | **stimmt** (nur 2 Bit, gelu) | Z3 Z. 25; bei 3 Bit Lloyd 0,015 vs Hadamard 0,013 (Hadamard leicht besser) |
| Z3b: Gitter ohne Nullstufe als Artefakt (K1, K2) | stimmt | Z3 Z. 5 (1,3–1,4 bpc ohne Rotation), Tabelle Z. 24 |
| Z3: Rotation macht Rauschen additiv, ohne Qualität (F5) | stimmt | Z3 Z. 16 |
| „Rotation hilft real bei g32 MXFP4 und im Gradientenpfad“ (K1, K8) | **nicht auffindbar** als eigener Befund | grep in LABORBUCH und Z*_ergebnisse_raven.md ohne Treffer zu MXFP4/Wgrad/Gradientenpfad; vermutlich Literaturaussage, im Feld aber unter „Bezug zu eigenen Befunden“ geführt |
| Z4/Z5-Bezüge in Feldern | nicht auffindbar | Felder nennen weder Z4 noch Z5 als Befund; Z5-Ergebnisdatei existiert nicht (nur Z4_ideen.md) |
