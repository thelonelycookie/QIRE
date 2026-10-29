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
