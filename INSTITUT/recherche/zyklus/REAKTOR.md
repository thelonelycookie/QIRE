# REAKTOR (29.10.2026)

Nichts gerechnet, kein git, keine Websuche. Grundlage: STROM, ARCHIV, Z3_ideen, Z2_ergebnisse_raven. Alle Literaturzahlen sind ungeprüfte Snippets aus STROM. P = Vorab-Schätzung, dass der Test die Idee stützt. Alles klassisch (I-038), Wortwahl "quanten-inspirierter Vorteil".

## 1. Fund für Fund: (1) stützt/widerspricht, (2) neuer Gegner

| Fund | (1) Wirkung auf unsere Befunde | (2) Neuer stärkster Gegner |
|---|---|---|
| QAM-W (2605.26339) | Stützt: Hadamard + Lloyd-Gitter ist Standard. Widerspricht nichts. Zahlen bei ~5,5 Bit, für 2 Bit wertlos. | **G1: Hadamard + skalares Lloyd-Max** (statt uniform) als Gegner des rohen UND des rotierten uniformen Arms. Fair-Gitter-Frage, deckt B1 ab, aber Lloyd muss auf rotierten Daten gefittet sein. |
| PolarQuant (Vicentino) | Stützt Lloyd-Arm. ID fehlt, nicht zitierbar. | Gleicher Gegner wie G1; nicht mit PolarQuant/TurboQuant verwechseln. |
| RCP (2502.15779) | Stützt Aktivierungsachse (Rotation trägt A4). Widerspricht nicht. Verlust 2,84 PPL bei W2A4KV4 zeigt: gelernter nichtuniformer Quantisierer plus QAT ist schon das Standardrezept. | **G2: QAT-roh mit lernbarem nichtuniformem Gitter** (A2 muss dieses Gitter bekommen, sonst ist der QAT-Gegner zu schwach). |
| 1-Bit Wonder (2602.15563) | Berührt Gitterachse und Trainingslänge; k-Means-Gitter im QAT. Kein Widerspruch. | **G3: k-Means-Gitter roh** als Gegner der Rotation (deckt sich mit Lloyd, aber datenadaptiv im Training). |
| QAT-Skalengesetz (Chen, ICML 2026) | Stützt "Schaden wächst mit T" (Z1b). **Teilwiderspruch zur Methode:** unser "Rotation senkt a, nicht b" stützt sich auf zwei Trainingslängen (T1/T2) und Steigung je Verdopplung. Ein Exponent b ist mit zwei Punkten nicht bestimmt. Gesetz fittet 268 Läufe. | **G4: Gruppengröße.** Chen: Fehler steigt mit gröberer Gruppe. Roh mit kleiner Gruppenskala (g = 16/32/64) ist ein fairer Gegner, der Rotation teilweise ersetzt (Great Inversion: Flachmachen hilft nur bei gemeinsamer Skala). Bislang nur "Skala je Kanal" erwähnt, Gruppenachse fehlt. |
| Zhang et al. (2602.19241) | Passt als Theorierahmen zu "a nicht b", aber nur Hypothese (siehe 2.1). Kein Widerspruch. | **G5: Rauschtyp-Kontrolle:** Wenn Rotation nur den Rauschtyp (additiv/multiplikativ) ändert, muss ein roher Arm mit künstlich additivem Rauschen (Skala = Kanal-RMS) denselben Gewinn zeigen. |
| Low-Bit Favors Undertrained (2411.17691), ParetoQ (2502.02631) | Stützen Z1b (Schaden wächst mit Trainingslänge). ParetoQ **bedroht den "+0,54 pp"-Befund:** klein-fp16 ist ein schwacher Gegner, 2-Bit/ternär liegt auf der Pareto-Front nur mit QAT. Nachtragen ins ARCHIV (nicht 2026). | **G6 (wichtigster): Mittleres Modell mit 4 Bit (rotiert und roh, RTN) bei gleichen Bytes** zwischen Groß-2-Bit und Klein-fp16, plus **G7: Groß-ternär/2-Bit mit QAT** gleicher Bytes. Klein-fp16 allein ist kein fairer Endgegner. |
| Rattew (ICLR 2026) | Nur Abgrenzung: Quantenhardware, nicht unsere klassische Wirkung. | keiner |

## 2. Neue Ideen (Kombination mit unseren Befunden)

Neuheitsprüfung gegen ARCHIV und Z3_ideen. Test = Tuning-Seeds 950–989, gleiche Bytes/FLOPs, höchstens 30 min CPU.

### R1. Rauschtyp-Messung: multiplikativ oder additiv, mit und ohne Rotation
- Idee: Quantisierungsfehler der FFN-Aktivierung je Koordinate e_i gegen Koordinaten-Signalstärke s_i. Fit log Var(e_i) = c + k·log Var(x_i). k ≈ 1: multiplikativ (Fehler wächst mit Signal, Ausreißerkanäle). k ≈ 0: additiv (Fehler flach). Hypothese: roh k nahe 1 (Absmax-Skala gleicher Kanal-Ausreißer), rotiert k nahe 0. Dann ist Rotation ein Wechsel multiplikativ → additiv (Zhang), und "a statt b" folgt aus Theorie: additives Rauschen skaliert mit anderer Konstante, nicht mit anderem Exponenten.
- Langweiligste Erklärung: Rotation senkt nur das Verhältnis Max/RMS (Kurtosis), Skala des Rauschens sinkt um konstanten Faktor. Kontrolle: roh mit Gruppenskala g=16 (G4) und roh mit Kanal-Vorskalierung (G5) müssen k ebenfalls senken. Tun sie es, ist Rotation nur ein teurer Ersatz.
- Billigster Test: vorhandene Z2c-Checkpoints (FFN-Mitte, T1 und T2), nur Messung von Fehler und Varianz, kein Training, ca. 5 min, 5 Seeds.
- P 65 %, dass roh k > rot k um ≥ 0,3; P 40 %, dass Gruppenskala dasselbe leistet (dann Erklärung langweilig).
- Neuheit: Archiv "Dekohärenz: Quantisierung als Rauschen" ist offen und LEER bei Aktivierungen; Z3 hat keinen Rauschtyp-Test. Neu.

### R2. Exponent sauber messen: a·T^b mit ≥ 4 Trainingslängen, roh / rot / Gruppenskala
- Idee: Messlatte nach Chen: Schaden(T) = a·T^b mit T ∈ {300, 600, 1200, 2400} (vier Punkte, bis acht Dehnungen von T1 zu T2), Konfidenzband für b per Bootstrap über Seeds. Arme: roh, Hadamard, roh mit g=32. Befund "senkt a, nicht b" gilt nur, wenn die b-Intervalle überlappen und a-Intervalle nicht.
- Langweiligste Erklärung: b ist mit zwei Punkten nicht identifizierbar; scheinbar gleiche Steigung ist Rauschen. Oder Schaden sättigt (dann ist Potenzgesetz falsch, Modell mit Offset a·T^b + c nötig).
- Billigster Test: Byte-LM d=128, vier Längen (zusammen ≈ 3,2× die Rechenzeit von T2, daher nur 3 Seeds und nur FFN-Mitte GELU): ca. 28 min. Alternative zuerst: die Z1b-LM-Daten (10/10 grün) mit mehr als zwei Längen nachfitten, falls vorhanden (0 min Rechnung).
- P 55 %, dass b-Intervalle überlappen (Befund hält); 25 %, dass Rotation b tatsächlich senkt (dann wäre der Befund stärker); 20 %, dass Potenzgesetz nicht passt.
- Neuheit: ARCHIV nennt a·T^b, aber nur zwei Längen. Pflicht-Absicherung des grünen Befunds, nicht eigene Ideen-Neuheit.

### R3. 2D-Lloyd nach Hadamard bei 2 Bit (QAM-W-Arm)
- Idee: je zwei benachbarte rotierte Koordinaten gemeinsam mit 16 Punkten (4 Bit pro Paar = 2 Bit pro Koordinate) per Lloyd/k-Means quantisieren statt zwei mal 4 Stufen. Zusatz: Blockgröße b (A1) variieren: bei kleinem b sind rotierte Koordinaten nicht iid-Gauß, dort kann 2D mehr holen.
- Langweiligste Erklärung (und Erwartung): Nach voller Hadamard sind die Koordinaten nahezu unabhängig Gauß; dann bringt ein 2D-Gitter nur den Raumfüllgewinn: Obergrenze 1,53 dB in unendlich vielen Dimensionen, in 2D (Hexagon) nur ca. 0,17 dB, also etwa 0,03 Bit je Koordinate. Das ist klein gegen den Rotationsgewinn. Skalares Lloyd-Max (G1) hat den Großteil des Gewinns schon. Kontrolle: skalares Lloyd-Max, uniform mit Null, no-zero; 2D-Gitter mit gleich vielen Punkten auf Zufallsdaten.
- Billigster Test: Stufe 1 rein synthetisch: iid-Gauß, MSE von 4×4-Skalar gegen 16-Punkte-2D-Lloyd gegen gemessene rotierte Aktivierung (2 min, ohne Training, Vorabschätzung Differenz < 0,2 dB); Stufe 2 nur wenn Stufe 1 ≥ 0,15 dB zeigt: FFN-Mitte T2, 5 Seeds, ca. 15 min.
- P 15 %, dass 2D-Lloyd skalares Lloyd um ≥ 0,005 bpc schlägt bei voller Hadamard; P 40 % bei Block b = 8 (dort bleiben Korrelationen); Rechenaufwand im Betrieb (Suche im Buch, 16 Punkte) wahrscheinlich größer als Gewinn.
- Neuheit: Z3 hat Lloyd nur skalar (B1, D1), 2D-Gitter nirgends. QAM-W liefert es für ~5,5 Bit Gewichte, nicht für 2-Bit-Aktivierung. Neu, aber voraussichtlich schwach.

### R4. Gruppengröße als Rotationsersatz (Gesetz-Term nach Chen)
- Idee: Schaden(g, Rotation ∈ {roh, rot}) mit g ∈ {8, 16, 32, 64, voll}. Fragen: Ab welchem g holt roh den Rotationsgewinn ein (Kreuzung g*)? Mit Rotation: wie unabhängig ist Schaden von g? Ergibt direkt Messlatte "Niveau-Term hängt von g, Rotation macht ihn g-unabhängig". Bytes fair rechnen (Skala je Gruppe kostet 16 Bit / g je Wert).
- Langweiligste Erklärung: Kleine Gruppen lösen das Ausreißerproblem allein; Rotation spart nur Skalenbytes. Dann ist der Vorteil reiner Byte-Overhead (z. B. g=32 kostet 0,5 Bit je Wert bei fp16-Skala), und bei 2 Bit macht das 25 % mehr Bytes: Rotation wäre dann Byte-Ersparnis, kein Qualitätsgewinn.
- Billigster Test: Z2c-FFN-Mitte-Skript mit g als Parameter, 5 g-Werte × 2 Arme, T1/T2, 5 Seeds ≈ 20 min. Gemeinsames Skript mit A1 (Blockgröße) möglich (g und b als zwei Achsen, g = b erlaubt direkten Vergleich).
- P 55 %, dass roh mit g=32 mindestens 50 % der Lücke schließt; P 40 %, dass die Rotation bei Berücksichtigung der Skalenbytes weiter vorn liegt.
- Neuheit: ARCHIV/Z3 nennen Kanalskala (B2, Kontrolle) und Blockgröße der Rotation (A1), aber keine Gruppenskala als Gegner mit Byte-Verrechnung. Neu, hoher Nutzen als fairer Gegner.

### R5. Fairer Bytegegner: Mittelmodell 4 Bit zwischen Groß-2-Bit und Klein-fp16
- Idee: Drei Modelle gleicher Gewichtsbytes: Klein fp16, Mittel 4 Bit (rot und roh), Groß 2 Bit rotiert; zusätzlich Groß-2-Bit-QAT ohne Rotation (G7). Zeigt, ob "+0,54 pp" ein Bit-Breite-Effekt ist (alles unter 16 Bit gewinnt) oder spezifisch für Rotation + 2 Bit.
- Langweiligste Erklärung: Klein-fp16 ist ein Strohmann; schon Mittel-4-Bit-roh schlägt es, Groß-2-Bit-rotiert liegt auf derselben Bytelinie ("Bytes-pro-Parameter-Gesetz"). Kontrolle = Test selbst.
- Billigster Test: MLP, drei Breiten, Z2-Skript mit Bitzahl als Parameter; ca. 12 min.
- P 45 %, dass Groß-2-Bit-rot Mittel-4-Bit schlägt (±0,3 pp); 30 %, dass Gleichstand; 25 %, dass Mittel-4-Bit klar vorn. Falls Letzteres, fällt der "+0,54 pp"-Befund auf Teilbefund (Verdacht ParetoQ).
- Neuheit: Z3 C4 (Kreuzungspunkt) variiert Budget, aber nicht den Bitbreiten-Zwischenpunkt. Neu.

### R6 (kleinere Variante). Lloyd-Gitter im Training lernen (1-Bit-Wonder-artig)
- Idee: die vier Gitterpunkte als lernbare Parameter (4 Zahlen je Schicht) im QAT-roh gegen PTQ-rot. Kombination von G2/G3 mit A2.
- Langweiligste Erklärung: lernbares Gitter ≈ Lloyd-Max auf Daten, nichts Neues. Kontrolle: Lloyd offline.
- Test: in A2 integriert, 0 min Zusatz (Parameter in Skript). P 35 % schlägt PTQ-rot. Neuheit: Teilduplikat A2/B1, nur als Arm, keine eigene Idee.

## 3. Besondere Fragen

### Multiplikativ vs. additiv und Rotation
- Standardreading: Absmax-Quantisierung mit gemeinsamer Skala wirkt "multiplikativ", weil Fehlerleistung mit dem größten Wert skaliert (Ausreißerkanal bestimmt Schrittweite für alle). Rotation verteilt Ausreißerenergie, Fehler wird kanalunabhängig, also additiv. Das erklärt das riesige Aktivierungsniveau (1,5 → 0,05 bpc) ohne Neuheit, ist aber als Messung (R1) sauber prüfbar.
- Theoretische Folge für "a, nicht b": Wenn beide Rauscharten bei zunehmendem T gleich wachsen (Gewichte wachsen mit T), ändert nur die Konstante, passt zum Befund. Wenn Zhang zeigt, dass multiplikatives und additives Rauschen *verschiedene* Skalierungsexponenten geben, wäre ein Exponentenwechsel zu erwarten. Spannung: Das passt nicht zu "b gleich". Auflösung durch R1 plus R2 (mehr Längen).
- Warnung: Aussage ist nur Hypothese aus Snippet; Zhang untersucht lineare Regression, nicht Transformer-FFN.

### QAT-Skalengesetz als Messlatte für den Exponenten-Befund
- Wir vergleichen: unser a·T^b PTQ-Schaden gegen Chens QAT-Fehler mit (Modellgröße, Tokens, Gruppengröße). Unterschied: PTQ vs QAT, W4A4 vs unser 2 Bit, 268 Läufe vs unsere Kleinläufe. Nur Form (Potenzgesetz, Gruppeneinfluss) vergleichbar, nie Zahlenwerte.
- Pflicht vor jedem Schreiben: R2 (≥ 4 Längen), sonst darf "senkt a, nicht b" nicht im Text stehen (höchstens "Steigung je Verdopplung unverändert innerhalb Streuung").

### 2D-Lloyd nach Hadamard bei 2 Bit
- Erwartung gering (R3): Hadamard macht Koordinaten fast iid Gauß, Lloyd-Max skalar ist dort schon nahe der Grenze, 2D-Raumfüllgewinn nur ca. 0,17 dB. Chance nur bei kleinen Blöcken. Billiger Vortest rein synthetisch, 2 min.

## 4. Rangliste der neuen Ideen

| Rang | ID | Einzeiler | P | Test |
|---|---|---|---|---|
| 1 | **R5** | Mittel-4-Bit-Gegner bei gleichen Bytes prüft, ob "+0,54 pp" ein echter Rotationsbefund oder ein Bitbreiten-Strohmann ist. | 45 % | 12 min |
| 2 | **R4** | Gruppenskala g als fairer roher Gegner, mit Byte-Verrechnung; zeigt, ob Rotation Qualität oder nur Skalenbytes spart. | 55 % | 20 min |
| 3 | **R2** | Exponent b mit ≥ 4 Längen und Bootstrap: sichert oder kippt "Niveau a, nicht Exponent b". | 55 % | 28 min |
| 4 | **R1** | Rauschtyp-Messung (Steigung k): macht aus Hypothese Zhang eine Messung, 5 min ohne Training. | 65 % | 5 min |
| 5 | R3 | 2D-Lloyd nach Hadamard; erwartet schwach. | 15 % | 2 min synthetisch |
| Reserve | R6 | lernbares Gitter als Arm in A2. | 35 % | 0 min Zusatz |

## 5. Neue Pflicht-Gegner (in jeden relevanten Test)
1. **G6** Mittelmodell 4 Bit (roh und rot) bei gleichen Bytes, neben Klein-fp16 und Groß-2-Bit-rot.
2. **G7** Groß-2-Bit-QAT ohne Rotation (gleiche Bytes), nach ParetoQ/RCP.
3. **G1/G3** Hadamard + skalares Lloyd-Max sowie roh + k-Means/Lloyd-Gitter (auf der Verteilung gefittet, die der Arm sieht).
4. **G4** Roh mit Gruppenskala g ∈ {16, 32, 64}, Skalenbytes mitzählen.
5. **G5** Roh mit Kanal-/RMS-Vorskalierung (additives Rauschen erzwungen).
6. **G2** QAT-roh mit lernbarem nichtuniformem Gitter (nicht das uniforme Straight-Through-Gitter).
7. **Methodenpflicht:** ≥ 4 Trainingslängen und Bootstrap-Intervall für b vor Aussagen zum Exponenten.

## Nachträge fürs ARCHIV (nach Volltextprüfung)
- 2411.17691 und ParetoQ 2502.02631 stützen Z1b, nicht 2026, nachtragen.
- Zellen: Dekohärenz × BK (R1), Unitarität × BG (R5, R4), Unitarität × TZ (R2) bekommen Kand.-Einträge R1, R2, R4, R5.
- Reihenfolge: B1 und R4 teilen sich das Z2c-FFN-Skript; Ergebnis Z3_ffn_fair zuerst abwarten. Läufe nur nach Maltes Zeile.

---

# 29.10. Lauf 2

Nichts gerechnet, kein git, keine Websuche. Grundlage: STROM-Abschnitt "29.10. Lauf 2" (nur Snippets, alle Zahlen UNGEPRÜFT), bisheriger REAKTOR, Z2_ergebnisse_raven. ARCHIV nur gelesen/nicht angefasst. P = Vorab-Schätzung, dass der Test die Idee stützt. Alles klassisch (I-038): "quanten-inspirierter Vorteil". Zahlen aus meinem Gedächtnis (Horowitz 2014 usw.) sind als "Gedächtnis, vor Zitat prüfen" markiert.

## L2.1 Fund für Fund

| Fund | Wirkung | Neuer Gegner / Konsequenz |
|---|---|---|
| LightRot 2607.27704 | Widerspricht teilweise: Rotation ist nur dann ein Speicher-gegen-Rechnen-Gewinn, wenn ihre Kosten klein bleiben; Hardwarepapier zählt sie als eigenen Posten. Unser FWHT-Zeitbefund (w12) braucht dieselbe Buchführung. | **G8: Rotationskosten ehrlich mitzählen** (d·log2 d Additionen je Vektor, plus Speicherzugriffe). Kein neuer Qualitätsgegner, aber Pflichtspalte "Rotations-Ops". |
| FIGLUT 2503.06862 | Stützt: Niedrigbit-Gewinn kommt in Hardware real an, aber nur mit LUT-Kern (+59 bis +98 % TOPS/W, ungeprüft). Auf CPU/GPU ohne solchen Kern kommt er nicht an. | Grenze der Aussage: Energie nur als Modell, nicht als Messung auf Standardhardware. |
| 2508.06753 (INT2xINT8 auf Intel Xe2) | Stützt: 2-Bit-Kerne gibt es, bis 7x schneller als 16 Bit (ungeprüft). Zugleich Widerspruchsrisiko: das Modell dort braucht vermutlich QAT/spezielle Quantisierung, nicht PTQ-Rotation. | **G7 bleibt Pflicht** (Groß-2-Bit-QAT roh gleicher Bytes), jetzt mit Hardwarebeleg, dass der Gegner praxisrelevant ist. |
| 28-nm-Beschleuniger (INT4 27,4, INT8 22,9 TOPS/W) | Nur Größenordnung: 1 TOPS/W ~ 1 pJ/Op. Bedingungen verschieden, nicht vergleichbar. Zeigt: Rechnen liegt bei ~0,04 pJ/Op, also unter 1 pJ. | Dient nur als Beleg "Rechnen ist billig, Datenbewegung teuer" (siehe L2.2). |
| Quartet II 2601.22813 und HALO 2501.02625 | Stützen Hadamard im Training als etabliert (Vor- und Rückwärtspfad). Widersprechen unserem Befund nicht. Zahlen (>2x Fehler, 1,41x End-to-End) sind für Kernel/Feintuning, nicht für unsere PTQ-Stufe. | **G9: Hadamard im Training (HALO-Stil) + normales QAT** als Gegner zu A2, siehe L2.4. |
| QuEST 2502.05003 | Stützt Z1b indirekt (Trainingsbits hängen von Compute ab); Hadamard dort Bestandteil. | Gleicher Gegner wie G9/G2: QuEST-artiges QAT mit Hadamard ist das Rezept, das A2 schlagen oder ehrlich einordnen muss. |
| 2510.10964 "Not All Bits Are Equal" | Stützt R5, siehe L2.3. | Kippunkt ~8-Bit-4B liegt weit oberhalb unserer Kleinmodelle; Form des Gesetzes nutzbar, Zahl nie. |
| 2605.01188 Compute Optimal Tokenization (~60 B/Param) | Neutral. Datenachse DA bleibt leer. Für Byte-LM irrelevant (Tokenisierung fest = Bytes). | Nur Fußnote: unser Byte-LM ist der Randfall "Tokenisierung = 1 Byte"; ~60 B/Param nicht übertragbar. |
| 2605.13979 Lottery Tickets quanten-inspiriert | Zitierbarer Rahmen, siehe L2.5. | kein Gegner |
| 2307.06937 (VQML = MPS-Unterklasse) | Abgrenzung I-038. Zitat. | keiner |
| SORF 1610.09072 / 1605.09049 | Stützt, aber alt: Hadamard-Rotation ≡ strukturierte Zufallsorthogonale, kein Neuheitsanspruch. Kein 2025/26-Fastfood-Treffer (Lücke). | Zitatpflicht in der Arbeit (Fastfood, SORF, QuaRot-Linie). |

## L2.2 Energie "Speicher statt Rechenzeit": ehrliche Abschätzung

Grundproblem: Wir haben keine Leistungsmessung (Container, 4 Kerne, kein Zähler) und die Hardwarepapiere gelten für Sonderchips. Daher kein "wir sparen X J", sondern ein **parametrisches Zählmodell mit Bandbreite**, offen als Modell gekennzeichnet.

**Modell (pro Vorwärtsdurchlauf eines Tokens):**
E = E_Gew + E_Akt + E_Rechnen + E_Rotation
- E_Gew = Bytes Gewichte, aus Speicher gelesen × e_Speicher(Ebene). Bei Batch 1 / Erzeugung wird jedes Gewicht einmal pro Token gelesen: genau hier gewinnt "weniger Bytes".
- E_Rechnen = MACs × e_MAC(Bitbreite). Nur INT-/Entpack-Kosten; ohne Spezialkern wird 2-Bit zu INT8 entpackt, also e_MAC = INT8-Wert, **kein** Rechenvorteil (FIGLUT/Xe2 nur mit Kern).
- E_Rotation = d·log2 d Additionen pro Block je Aktivierungsvektor × e_Add, plus Aktivierungsbytes.
- Klein-fp16 vs Groß-2-Bit gleicher Bytes: Gewicht-E gleich (per Konstruktion), Rechen-E des Großen höher (mehr MACs), Rotation kommt dazu. Der Vorteil ist also **kein Energievorteil**, sondern ein Qualitätsvorteil bei gleichem Speichertraffic. Das ehrlich so schreiben.

**Konstanten (Größenordnung, aus dem Gedächtnis: Horowitz 2014, 45 nm; vor Zitat prüfen, Spanne ±3x, neuere Knoten kleiner):** DRAM-Zugriff ~100 bis 200 pJ je Byte, On-Chip-SRAM ~1 bis 5 pJ je Byte, INT8-MAC <1 pJ, FP16-MAC ~1 bis 2 pJ. Daraus: Datenbewegung aus DRAM kostet 100x und mehr als eine Rechenoperation. Stützt aus STROM nur grob: 28-nm-Chip ~0,04 pJ/Op bei INT4/INT8 (ungeprüft).

**Vorschlag für die Jugend-forscht-Arbeit (Pflichttabelle, 0 min Rechnung außer Zählen):**
1. Pro Arm zählen: Gewichtsbytes, Aktivierungsbytes, MACs, Rotations-Additionen (analytisch, aus Architektur, keine Messung).
2. Energie als **Intervall** [min, max] mit zwei Szenarien: (a) Gewichte im DRAM, Batch 1 (Speicherdominiert, 150 pJ/B ±3x), (b) Gewichte im Cache / Batch groß (Rechen dominiert). Ergebnis ehrlich: im Szenario (a) liegt der Vorteil von Bytegleich bei null (gleiche Bytes) und der Nutzen von 2-Bit nur im Qualitätsgewinn; der echte Energiehebel ist "gleiche Qualität mit weniger Bytes", also die Bytegleich-Achse **nach unten** (Kippunkt-Test R5/C4: wie viele Bytes spart Groß-2-Bit-rot bei gleicher Qualität wie Klein-fp16).
3. Zusätzlich CPU-Wandzeit der Rotation (FWHT) real messen und neben die Zählung stellen (Messung vs Modell, Abweichung nennen).
4. Keine "TOPS/W"-Zahlen aus Papieren in die eigene Bilanz mischen; nur als Plausibilitätsbeleg zitieren.

Langweiligste Gegenerklärung: Auf Standardhardware ist LLM-Erzeugung speicherbandbreitengebunden, daher spart jedes Byte fast linear Zeit/Energie; unser "Speicher statt Rechenzeit" ist dann nur die bekannte Bandbreiten-Rechnung, keine Neuheit. Neuheitsprüfung: bekannt (Roofline, Horowitz-Tabelle). Das als solide Grundlage zitieren, nicht als Ergebnis ausgeben.

## L2.3 Byte-Budget-Gesetz (2510.10964) als Messlatte für den Bytegleich-Test

- Aussage laut Snippet: Bei festem Gesamtbyte-Budget lohnt sich unterhalb eines Kippunkts (~8-Bit-4B, ca. 4,2 GB) mehr Gewichtsgenauigkeit/Größe, oberhalb mehr Tokens/KV. Unsere Modelle liegen weit darunter (Byte-LM, kleine MLP): wir sind im "Gewichts-Bit"-Regime, dort ist das Gesetz **Stütze** für Groß-niedrigbit gegen Klein-fp16, aber nur als Richtung.
- Nutzung als Messlatte: Unser Bytegleich-Test muss die dort getrennten Posten BG (Gewichte) und BK (KV/Aktivierung) getrennt führen. Messlatte = Mittel-4-Bit bei gleichen Bytes (G6/R5): ein Befund "Groß-2-Bit-rot schlägt Klein-fp16" zählt erst, wenn er auch Mittel-4-Bit bzw. die Pareto-Linie (ParetoQ) nicht verliert.
- **Widerspruchsrisiko:** Wenn das Gesetz für kleine Modelle 3 bis 4 Bit als Optimum liefert, ist 2 Bit unter dem Optimum; rot-2-Bit gewinnt dann nur, wenn Rotation das Optimum nach unten schiebt. Das ist selbst eine testbare, neue Aussage: **R7 (neu) "Rotation verschiebt das optimale Bit pro Gewicht nach unten".**
- R7: Bitbreiten {2, 3, 4, 6, 8, fp16} bei gleichen Gesamtbytes (Breite variiert), roh vs rot; Optimum-Bit pro Arm ablesen. Langweiligste Erklärung: Optimum liegt bei beiden bei 4 Bit, Rotation hebt nur das 2-Bit-Niveau, ohne es zum Optimum zu machen. Billigster Test: Z2-Skript mit Bit und Breite als Parameter, MLP, Tuning-Seeds 950-989, 5 Seeds, ~15 min (kann mit R5 geteilt werden: R5 ist die Teilmenge {2, 4, fp16}). P 35 %, dass Optimum-Bit rot < roh Optimum-Bit (ganze Stufe); P 60 %, dass die Kurve glatt genug ist, um es überhaupt zu sehen. Neuheit: Verallgemeinerung von R5 auf eine Kurve, eigenes Gesetz-Prüfen auf Rotation. Nicht in ARCHIV/Z3 (C4 variiert Budget, nicht Bitbreite). Gering-mittel neu; ParetoQ-Linie ist die nächste Arbeit, also abgrenzen.
- Bytegleich-Regeln aus dem Gesetz: Skalen, Codebuch (Lloyd-Gitter), Rotationsparameter (Vorzeichen-Diagonale D: 1 Bit je Koordinate, oder Seed = 0 Bytes) im Budget mitzählen.

## L2.4 Idee A2 (QAT-rot) schärfen: Hadamard im Training (HALO, Quartet II, QuEST)

Stand: A2 = QAT mit Rotation gegen QAT roh. Erkenntnis aus Lauf 2: Hadamard im Training ist Standard (HALO, Quartet II, QuEST), also **A2 darf nicht als "neu" verkauft werden**, sondern als Messung "bringt Rotation im QAT noch etwas, wenn QAT das Ausreißerproblem selbst lernen kann?".

**Schärfung (A2'):**
- Vier Arme bei gleichen Bytes/FLOPs: (i) PTQ roh, (ii) PTQ rot (bekannt: +0,5 pp), (iii) QAT roh, (iv) QAT rot (Hadamard in Vorwärts und, wie HALO, Rückwärts). Gegner (iii) bekommt ein lernbares nichtuniformes Gitter (G2) und Gruppenskala (G4), sonst ist es ein Strohmann.
- Kernfrage: **Wird der Rotationsgewinn (ii)-(i) durch QAT aufgefressen ((iv)-(iii) ≈ 0)?** Hypothese Z1b/Chen: Schaden wächst mit T; QAT lernt Gewichte, die ohne Rotation passen. Erwartung: Gewinn schrumpft, bleibt bei Aktivierungen eher (Aktivierungen sind datenabhängig, QAT kann sie schlechter vorab formen).
- Interessanter Zusatzarm (v): **Rotation nur auf Aktivierungen im Training** (billig, keine Gewichtsänderung), nach RCP-Linie; (vi) Rückwärts ohne Hadamard (nur Vorwärts) als Kostenreduktion (HALO-Variante).
- Langweiligste Erklärung: QAT mit gelerntem Gitter holt den Rotationsgewinn komplett ein (schon in 1-Bit Wonder/QuEST-Linie gesehen); dann ist der Befund "Rotation ist ein PTQ-Werkzeug", was ehrlich und für die Byte-/Energie-Geschichte (PTQ ohne Training) immer noch nutzbar ist.
- Billigster Test: Kleinmodell Byte-LM d=128, T1 nur, Tuning-Seeds, 3 Seeds, Arme (iii), (iv) + ein Gitterarm: ~25 min CPU (QAT ist teurer als PTQ; wenn >30 min, MLP statt LM). P 45 %, dass (iv)-(iii) ≥ 0,005 bpc bei Aktivierungen bleibt; P 20 % bei Gewichten. Neuheit: kombiniert mit Z1b-Befund (Schaden-Zeit) und Byte-Gleich, Methode selbst bekannt (HALO/Quartet II/QuEST zitieren). Als Messung wertvoll, nicht als Erfindung.

## L2.5 Quanten-inspirierte Dequantisierung (2605.13979) als Rahmen

- Was zitierbar ist (laut Snippet): quanten-inspirierter *klassischer* Algorithmus mit poly(D) statt exponentiell, nur Simulation. Dequantisierungslinie (Tang 2018, Stichprobenzugriff auf Zeilen/Spalten, Normen-Sampling) ist der etablierte Rahmen für "quanten-inspiriert": klassischer Algorithmus mit quantenähnlicher Datenstruktur (Sampling nach Norm), kein Quanten-Speedup.
- Nutzen für uns: **Rahmen/Begründung für die Wortwahl.** Wir schreiben: "quanten-inspiriert im Sinn der Dequantisierung: strukturierte unitäre/Hadamard-Transformationen und Norm-/Sampling-Ideen werden klassisch genutzt; Wirkung klassisch (I-038)". Das schützt vor Quantenvorteil-Behauptung und ist zitierbar (nur nach Volltextprüfung: Autoren, Inhalt, Zählung fehlen im Snippet).
- Widerspruchsrisiko: Hadamard-Rotation ist Fastfood/SORF-Linie (2013/2016), nicht Tang-Dequantisierung. Direkte Verbindung "unsere Rotation = Dequantisierung" wäre **überzogen**. Ehrliche Formulierung: Gemeinsamkeit nur im Muster "Quantenmathematik (unitäre Basiswechsel, Normenverteilung) -> klassisch billig".
- **R8 (neu, klein): Norm-Sampling als Arm.** Dequantisierungs-Trick: Spalten/Einträge mit Wahrscheinlichkeit ∝ Norm² auswählen. Anwendung: *Sparsifizieren nach Rotation*: nach Hadamard haben alle Koordinaten ähnliche Norm (flach), daher Norm-Sampling = gleichverteiltes Subsampling; ohne Rotation zerfällt Norm-Sampling auf Ausreißer. Test: Aktivierungsmatrix des FFN, Matrix-Produkt-Näherung mit k gesampelten Spalten (CRS/Drineas-Linie) roh (Norm-Sampling) vs rot (gleichverteilt), Fehler bei gleichem k. Langweiligste Erklärung: bekannte "Rotation macht Kohärenz niedrig"-Aussage der Randomized-NLA (Kohärenz/Leverage-Scores); nicht neu, nur als Verbindung nutzbar. Billigster Test: 5 min, nur Messung auf Z2c-Aktivierungen, kein Training. P 70 %, dass rot ≥ roh bei gleichem k; Neuheit gering (Randomized NLA, SRHT/Sketching, Drineas/Mahoney, Tropp); Wert: verknüpft sauber mit der Dequantisierungs-Zitierung und der Byte-Reduktion (weniger Spalten = weniger Bytes). Als Zitat-Brücke nutzen, nicht als Ergebnis.

## L2.6 Neue Pflicht-Gegner (Ergänzung zu Abschnitt 5)

8. **G8** Rotationskosten in Ops und Wandzeit mitzählen (LightRot-Linie); jede Energie-/Byte-Aussage nennt sie.
9. **G9** QAT roh *und* QAT mit Hadamard (HALO/QuEST-Stil) als Gegner/Referenz zu A2; ohne beides keine Aussage "Rotation im Training".
10. **G7 (verschärft)**: Groß-2-Bit-QAT roh gleicher Bytes, jetzt mit Hardwarebeleg (2508.06753), dass 2-Bit-Kerne real existieren.
11. **G6 als Kurve (R7)**: nicht nur 4 Bit, sondern Bitbreiten-Reihe bei gleichen Bytes.

## L2.7 Rangliste Zuwachs

| ID | Einzeiler | P | Test |
|---|---|---|---|
| R7 | Rotation verschiebt das optimale Bit pro Gewicht nach unten? (Kurve statt Punkt, Byte-Budget-Gesetz) | 35 % | 15 min (teilt R5-Skript) |
| A2' | QAT-rot gegen QAT-roh mit lernbarem Gitter: bleibt der Rotationsgewinn im Training? | 45 % Akt. / 20 % Gew. | 25 min |
| R8 | Norm-Sampling vs Rotation+Gleichsampling (Zitat-Brücke zur Dequantisierung) | 70 % (bekannt) | 5 min, nur Messung |
| Energie-Tabelle | Zählmodell mit Intervall statt Messung | n/a | 0 min Rechnung |

Gesamtreihenfolge (Gedanke, keine Freigabe): R5/R7 gemeinsames Skript zuerst (prüft den grünen +0,54-pp-Befund am stärksten), dann R4/R1 (Z2c-FFN-Skript), dann R2, A2'. R8 und Energie-Tabelle ohne Lauf, sofort machbar. Läufe nur nach Maltes Zeile.

## Nachträge fürs ARCHIV (nach Volltextprüfung)
- Neu: LightRot 2607.27704, FIGLUT 2503.06862, 2508.06753, Quartet II 2601.22813, HALO 2501.02625, QuEST 2502.05003, 2510.10964, 2605.13979, 2307.06937 (alle ungeprüft). Zellen: Unitarität x TZ (HALO, Quartet II, QuEST), Unitarität x BG (2510.10964, R7), Energie-Spalte (FIGLUT, Xe2, LightRot).
- Zahlen im Energieabschnitt (pJ/Byte, pJ/MAC) stammen aus meinem Gedächtnis, nicht aus STROM; vor Aufnahme in die Arbeit mit Primärquelle (Horowitz, ISSCC 2014) belegen.
