# Z3 Divergenz: Ideen, Neuheitsprüfung, Top 4 (29.10.2026)

Nichts gerechnet, kein git. Vier Werkzeuge (a Umkehrung, b Literatur 2025–2026, c Ende-zu-Ende-Kombination, d Fernanalogie), je 4 Ideen, 16 gesamt. P = Vorab-Schätzung, dass die Idee im Rauchtest (Tuning-Seeds 950–989, gleiche Bytes/FLOPs, zwei Trainingslängen T1 ≈ 600 / T2 ≈ 2400 Schritte Byte-LM d=128 bzw. 3/12 Epochen MNIST-MLP, ≤ 30 min CPU) den stärksten Gegner schlägt. Keine Befunde. Neuheit gegen ARCHIV.md (Stand Z3).

Ausgangslage (Maltes Stand): grün = Rotation vor 2-Bit-Aktivierungsquantisierung der FFN (1,5 → 0,05 bpc, 20/20; Kontrolle Z3_ffn_fair läuft) und rotiertes 2-Bit-Großmodell schlägt klein-fp16 bei gleichen Bytes (+0,54 pp, plus 20/20; Schwelle-grün nur 10/20). Rot: z² vs GELU lang, Bitzuteilung, Destillation, Seed-Codebuch, GPTQ-Rückführung, KV-Treppe, KV-Rotation.

## Literaturfunde (b), 6 WebSearch, nur Treffer-Snippets, vor Zitat Volltext prüfen
- **Qift** (arXiv 2606.02823, Jun 2026): Hadamard-rotierte W2A4/KV4; das Standard-2-Bit-Gitter {−2,−1,0,+1} kollabiert, no-zero-Gitter {±0,5, ±1,5} besser. Rotation Gaußisiert nur die Form. Warnt direkt vor unserem Verdacht (Gitter ohne Nullstufe/asymmetrisch).
- **Transforms for LLM Quantization: The Great Inversion** (arXiv 2608.25188, Aug 2026, Überblick ~200 Arbeiten): bei gruppenweiser gemeinsamer Skala hilft Flachmachen, bei frei zuteilbarem Code hilft Energiekonzentration; auf Fließkommagitter schwächt sich Flachmachen ab oder kehrt sich um. Das erklärt vermutlich unsere rote Bitzuteilung.
- **FlatQuant** (ICML 2025): gelernte affine Transformation je Schicht (Kronecker), W4A4 < 1 % Verlust bei LLaMA-3-70B; laut SERQ (2603.08185) stärkste W4A4-Referenz; rotationsbasierte Verfahren dominieren W4A4. **DuQuant++** (Apr 2026): eine 32×32-Rotation je MXFP4-Gruppe, halbiert Rotationskosten. **WUSH** (ICML 2026): Hadamard-Rückgrat plus datenabhängige Komponente, nah-optimal für INT/FP. **ButterflyQuant** (2509.09679), **ParoQuant**, **HARP**, **ReSpinQuant**, **ConQuR**: gelernte Butterfly-/Givens-/Cayley-Rotationen (nur Titel/Tabelle gesehen).
- **HALO** (NeurIPS 2025): Hadamard in Vorwärts- und Rückwärtspass für Feintuning mit Niedrigpräzision. Kein Treffer zu Rotation beim Vortraining von Grund auf.
- **Kein Treffer zu echtem W2A2 mit Rotation**: nächstes ist W2A4 (Qift, RCP als QAT). W2A2 ist daher Randgebiet; Neuheit wahrscheinlich, aber Qualität dort vermutlich schlecht.

## (a) Umkehrung
| ID | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|
| A1 | **Blockgröße umdrehen:** nicht volle Hadamard, sondern Blockdiagonal b ∈ {8, 16, 32, 128, voll}. Wie klein darf die Rotation sein (Kosten N·log b statt N·log N)? | Schon jede kleine Mischung glättet; Kontrolle: Zufallsvorzeichen-Permutation ohne Mischung, b=1 | Z2c-Skript FFN-Mitte, Block als Parameter, 4 Stufen und no-zero-Gitter, T1/T2, 5 Seeds; ≈ 15 min | 60 % (b=32 hält ≥ 80 % des Gewinns) | Archiv: nur volle Rotation getestet; Lit. DuQuant++/MXFP4-Block bekannt, für uns neu, behalten |
| A2 | **QAT statt PTQ als Gegner:** Aktivierungsquantisierung roh *im* Training (Straight-Through) gegen PTQ-Rotation | Training lernt das Rauschen; Rotation unnötig. Kontrolle: QAT roh, QAT rot, PTQ rot gleiche FLOPs | Byte-LM T1/T2, A2 in FFN-Mitte; ≈ 25 min | 40 % (QAT roh schließt ≥ 50 % der Lücke) | Archiv: TZ×Unitarität LEER, M3 verwandt; behalten, ehrlicher Gegner |
| A3 | **Asymmetriekarte W×A:** welche Seite braucht Rotation, Gewichte oder Aktivierung (W16A2, W2A16, W4A4 …)? | Folgt aus Z2c (Gewichte schwach, Aktivierung stark); Kontrolle: Gitter gleich | MLP 3×3-Karte; ≈ 15 min | 70 % bestätigt Z2c | Teil von C1, nicht separat |
| A4 | **Kurtosis-Strafe im Training** statt Rotation (Ausreißer verhindern statt verschieben), null Laufzeitkosten | Outlier-Safe-Pretraining u. ä. [G]. Kontrolle: Rotation ohne Strafe, beides | Byte-LM T1/T2, Strafe auf FFN-Mitte; ≈ 25 min | 35 % | neu im Archiv, Lit. verwandt; Reserve |

## (b) Literatur-Frontier
| ID | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|
| B1 | **Gitterachse (Great Inversion + Qift):** FFN-Aktivierung × Gitter {uniform mit Null, no-zero, Lloyd/NF, µ-Gesetz} × {roh, rot}. Wie viel vom 20/20-Gewinn bleibt, wenn der rohe Gegner ein faires Gitter hat? | Gewinn ist Gitterartefakt (Verdacht Z2). Kontrolle = Kern des Tests. Überschneidung mit Z3_ffn_fair prüfen, nur Lücken ausfüllen (Lloyd, µ-Gesetz, Gitter × Blockgröße) | Z2c-Skript, 8 Gitter-Arme, T1/T2, 5 Seeds; ≈ 20 min | 55 % (Rotationsvorteil bleibt ≥ 50 %); 30 % dass er verschwindet | Archiv: BG-Zelle nur Gewichte; für Aktivierung neu |
| B2 | **WUSH/FlatQuant-leicht:** Hadamard plus Kanalskala aus 256 Kalibrierproben (SmoothQuant-artig vor Rotation) | Skala allein reicht. Kontrolle: nur Skala, nur Hadamard | MLP + Byte-LM, 15 min | 35 % | Lit. bekannt, im Archiv neu |
| B3 | **Gelernte Butterfly-Rotation** (ButterflyQuant) statt fester Hadamard, Winkel auf Kalibrierdaten | Archiv: Hadamard ≡ Zufallsorthogonale (I-048 H2) spricht gegen Gewinn durch Struktur. Kontrolle: feste Hadamard, Zufallsrotation | MLP FFN-Mitte; ≈ 20 min | 20 % | neu, schwach; Reserve |
| B4 | **Rotation im Vortraining (HALO-artig):** A2-Quantisierung mit Hadamard in Vor- und Rückwärtspass, Training von Grund auf | Rotation ändert nur die Skala des Rauschens. Kontrolle: Zufallsvorzeichen, QAT roh (A2) | Byte-LM T1/T2; ≈ 30 min | 40 % | TZ×Unitarität LEER, Teilduplikat M3/A2; mit A2 vereinigen |

## (c) Kombination zu einem System
| ID | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|
| C1 | **W2A2-Bilanz Ende zu Ende:** Gewichte 2 Bit + FFN-Aktivierung 2 Bit, beide rotiert (Gitter aus B1), gegen (i) klein-fp16 gleicher Gewichtsbytes, (ii) W4A4 roh, (iii) W8A8. Ausgewiesen: Gewichtsbytes, Spitzen-Aktivierungsbytes, Ops (Additionen/Multiplikationen/Hadamard-Additionen), Energieschätzung per Energie-je-Operation-Tabelle (Horowitz 2014 [G]) | Es gewinnt nur die Gewichtsseite (Z2c); Aktivierung bringt Bytes, aber der Rest bricht. Kontrolle: A3-Karte, jede Seite einzeln | MLP 3/12 Epochen + Byte-LM T1/T2; Bytes- und Energiezählung rein analytisch; ≈ 25 min | 40 % (W2A2 rot ≥ klein-fp16); 70 % für W4A4 rot | Archiv: Kombination nie gerechnet; Lit. kein W2A2 mit Rotation gefunden; **behalten, Rang 1** |
| C2 | **Interaktion:** Schaden(W2 rot + A2 rot) gegen Summe der Einzelschäden | Unabhängige Fehler addieren sich; Kontrolle: Summe aus Einzelläufen | Teil von C1, 3 Zusatzarme | 50 % (unteradditiv) | in C1 aufgenommen |
| C3 | **Rotationsaufschlag analytisch:** Online-Hadamard nur vor Down-Projektion (Rest in Gewichte fusioniert), Aufschlag in % FLOPs und Bytes für d = 128…4096 | Bekannt (QuaRot); Kontrolle: Zeitmessung (w12: FWHT langsam bei Batch 1) | Reine Zählung, 0 min CPU, Zeitmessung getrennt als Randbefund | 80 % Aufschlag < 10 % ab d ≥ 512 | Archiv hat Ops-/Zeit-Widerspruch (FL), Zählung fehlt als Tabelle; in C1 einbauen |
| C4 | **Kreuzungspunkt:** ab welchem Bytebudget verliert rotiertes 2-Bit-Groß- gegen fp16-Kleinmodell (zwei Breiten, drei Budgets) | Gewinn nur in der Nähe der Kapazitätsgrenze; Kontrolle: Budget-Skalierung | MLP, 3 Budgets × 2 Breiten; ≈ 15 min | 55 % | neu (Z2 hat nur ein Budget); Reserve, in C1 als Zusatz |

## (d) Fernanalogie: Funk/OFDM
In OFDM spreizt IFFT/Hadamard die Energie auf viele Unterträger, das Signal wird Gaußähnlich und hat hohe Spitzen (PAPR); der Verstärker (= Quantisierer) leidet an Spitzen. Gegenmittel der Funktechnik: Clipping und Filtern, Companding (µ-Gesetz), Selected Mapping, Dither. Das ist dieselbe Kette wie bei uns, und ihre Gegenmittel sind unsere fairen Gegner.
| ID | Idee | Langweiligste Erklärung + Kontrolle | Billigster Test | P | Neuheit |
|---|---|---|---|---|---|
| D1 | **Companding statt Rotation:** µ-Gesetz-Gitter auf rohe Aktivierung (Funk/Telefonie: ungleichmäßiges Gitter ersetzt Glätten) | Ist nur ein nichtuniformes Gitter (Lloyd/NF). Kontrolle: Lloyd, Rotation | in B1 aufgenommen (µ-Arm) | 40 % schlägt roh-uniform, 20 % schlägt Rotation | Archiv: nicht getestet; in B1 |
| D2 | **Selected Mapping (SLM):** je Token K=4 Vorzeichenmuster vor Hadamard, das mit kleinstem Maximalbetrag gewählt, 2 Bit Zusatzindex | Entspricht Seed-Codebuch (Gewichte, rot Z1b) für Aktivierungen. Kontrolle: feste Rotation, gleiche Bytes inkl. Index | Byte-LM FFN-Mitte; ≈ 20 min | 25 % | Archiv: Seed-Codebuch rot für Gewichte; Aktivierung/dynamisch neu; Reserve |
| D3 | **Dither:** gemeinsamer Zufallsvorzeichen-/Offset-Dither (subtraktiv, aus Seed rekonstruierbar) entkorreliert den Fehler vom Signal, Rauschen weiß | Stochastisches Runden, Archiv L07 gestrichen (4× FLOPs); subtraktiver Dither kostet keine Zusatz-FLOPs. Kontrolle: ohne Dither, mit Rotation | MLP/FFN-Mitte; ≈ 10 min | 20 % | neu (Messung×BG/BK: Dekohärenz), Reserve |
| D4 | **Clipping-und-Filtern:** Aktivierung bei Perzentil p kappen, Restfehler (Kappung) dünn sparsam mitführen (dense-and-sparse) vor Rotation | = Perzentil-Clipping/KVQuant, Archiv L18. Kontrolle: Clip allein, Rotation allein | FFN-Mitte; ≈ 15 min | 30 % | Teilduplikat L18; als Kontrolle in B1 |

## Rangliste Top 4
| Rang | ID | Einzeiler | P | Test |
|---|---|---|---|---|
| 1 | **C1 (+C2, C3, A3, C4)** | W2A2-Ende-zu-Ende mit Hadamard: Bytes, Ops und Energieschätzung gegen klein-fp16, W4A4 roh, W8A8. | 40 % | 25 min |
| 2 | **B1 (+D1, D4)** | Gitterachse: uniform/no-zero/Lloyd/µ-Gesetz × roh/rot, wie viel vom FFN-Gewinn überlebt ein fairer roher Gegner (Great Inversion, Qift); nur Lücken zu Z3_ffn_fair. | 55 % | 20 min |
| 3 | **A2 + B4** | QAT-roh gegen PTQ-rot gegen QAT-rot: Rotation im Training von Grund auf (TZ×Unitarität, Lücke im Raster). | 40 % | 30 min |
| 4 | **A1** | Blockgröße der Hadamard-Rotation (8…voll): kleinster Aufschlag, der den Gewinn hält. | 60 % | 15 min |

Reserve: A4, B2, C4, D2, D3. Gestrichen: B3 (P 20 %, I-048 H2 spricht dagegen).

Reihenfolge: B1 vor C1 (C1 braucht das faire Gitter), A1 und B1 teilen das Z2c-FFN-Skript (ein Skript, Gitter und Block als Parameter). Erst Ergebnis Z3_ffn_fair abwarten: Fällt der Gewinn dort weg, werden C1 und A1 Nachweise über „kleineres Restproblem“, B1 wird dann Rang 1.

Ehrlichkeit: alles klassisch (I-038), Formulierung „quanten-inspirierter Vorteil“. Literaturangaben aus Suchtreffern/Gedächtnis, vor Zitat im Volltext prüfen. Läufe nur nach Maltes Zeile.
