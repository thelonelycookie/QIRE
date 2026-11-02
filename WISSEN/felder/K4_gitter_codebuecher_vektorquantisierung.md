# K4: Gitter/Codebücher/Vektorquantisierung

*Stand Welle 1 (a6, 02.11.2026). Nur Abstracts/Suchauszüge, kein Volltext; Zahlen ungeprüft, sofern nicht „Abstract bestätigt“. Neue Quellen: `WISSEN/neu/w1_a6.json`.*

## Landkarte
- **Skalar, nichtuniform:** NF4 (Quantilgitter, [arXiv:2305.14314]), Lloyd-Max. HIGGS = Hadamard + MSE-optimale Gitter, datenfrei, schlägt NF4 [arXiv:2411.17525].
- **Gitter + Hadamard:** QuIP# (E8P, 8D, 2 Bit) [arXiv:2402.04396]; NestQuant (genestetes Gosset-Gitter, W/A/KV) [arXiv:2502.09720]; HyperQuant (E8/D4/A2 + Rice-Codierung) [arXiv:2606.23406]; Leech-Gitter 24D (LLVQ) [arXiv:2603.11021], Dekoder-Kosten [arXiv:2609.02652].
- **Gelernte Codebücher:** AQLM [arXiv:2401.06118], PV-Tuning [arXiv:2405.14852], GPTVQ [arXiv:2402.15319], VPTQ [arXiv:2409.17066].
- **Trellis:** QTIP [arXiv:2406.11235], BCJR-QAT [arXiv:2605.10655].
- **Winkel/Polar:** PCDVQ [arXiv:2506.05432], PolarQuant [arXiv:2502.02617], TurboAngle [arXiv:2603.27467], QAM-W (2D-Lloyd, schwächer als QTIP bei 4 bpw) [arXiv:2605.26339].
- **Gegen wen:** Buchfreie/regelmäßige Gitter (QuIP#, LLVQ) gegen gelernte Bücher (AQLM, GPTVQ) und Trellis (QTIP). Rotation wird als Voraussetzung gesehen; LLVQ meldet geringere Rotationsempfindlichkeit.

## Stärkste Baselines/Gegner für QIRE
1. QTIP/LLVQ bei 2 Bit (Selbstangaben der Autoren, LLVQ vorn).
2. QuIP# als Referenz „Hadamard + Gitter“.
3. HIGGS und NF4 als datenfreie Skalargegner (entsprechen unserem Lloyd-Gitter).
4. Gruppenskalen (eigene Z3-Befunde).

## Wichtige Zahlen
| Zahl | Quelle | geprüft |
|---|---|---|
| Granularer Gewinn max. 1,53 dB (≈0,25 Bit/Gewicht); E8 0,88 dB Abstand, d. h. ~0,65 dB geholt (42 %); Leech weitere ~0,38 dB | HyperQuant [arXiv:2606.23406] (klassische Hochratentheorie) | nein |
| QuIP# Llama-2-7B 2 Bit Wiki2: 8,22 (ohne FT) / 6,19 (mit FT), FP16 5,12 | Folienauszug [arXiv:2402.04396] | nein |
| NestQuant Llama-3-8B W/A/KV 4 Bit PPL 6,6 (FP 6,14; SpinQuant 7,3) | Abstract [arXiv:2502.09720] | ja (Abstract) |
| PCDVQ: Richtungs-Clustering −46,5 % vs Betrag −2,3 % | Abstract [arXiv:2506.05432] | ja (Abstract) |
| GPTVQ: 70B in 3–11 h auf H100; VQ-Dekodierung auf Mobil-CPU schneller als INT4 | Abstract [arXiv:2402.15319] | ja (Abstract) |
| Trellis liest 2,40x weniger Bytes, 2,27x schneller als Leech-Layout (Batch 1) | [arXiv:2609.02652] | ja (Abstract) |

## Dekodierkosten im Betrieb
Decode ist speicherbandbreitenbegrenzt; Gewinn nur, wenn Entpackung günstiger als eingesparte Bytes. E8P: 1 KiB-Buch. Zu große Bücher (Leech 301 Klassen) kosten Traffic; Trellis (Bitshift) ist schneller. AQLM: moderate Latenzgewinne. Tabellen wachsen exponentiell mit der Dimension; deshalb Gitter/Trellis.

## Quantenbezug
- **Echt quantenmathematisch:** Stabilisatorzustände = Orbits der Clifford-Gruppe; Kerdock-Codes liefern N+1 gegenseitig unverzerrte Basen (unitäres 2-Design) [arXiv:1904.07842]. Stabilisatorzustände bilden ein 3-Design (Suchauszug). Die Cliffordgruppe erzeugt auch Hadamard-artige Transformationen (Hadamard-Gatter, Walsh-Hadamard).
- **Analogie:** Winkelcodebücher (TurboAngle, PolarQuant) = Phasenquantisierung auf dem Kreis, ähnlich Qubit-Phasen/Z4-Phasen i^k; Born-Quadrat |z|² passt zu Polar/Shape-Gain.
- **Keine Arbeit gefunden**, die Stabilisator-/Clifford-Orbits als Gewichts-Codebuch nutzt (eine Suche, negativ; Lücke, nicht Beweis).
- **Quanten-inspirierter Hebel:** Clifford-Orbit/Kerdock-Winkelmenge als festes, buchloses, hardwarenahes Winkelbuch (Phasen in Z4/Z8).

## Bezug zu eigenen Befunden
- Z3b: Lloyd-Gitter ohne Rotation (0,041 bpc) ≈ Hadamard (0,052) bei 2 Bit. Stützt: Rotation wirkt vor allem über Gitterartefakte; LLVQ meldet geringere Rotationsempfindlichkeit, die Great-Inversion-Theorie [arXiv:2608.25188] nennt Gruppenskalen als Ursache.
- Mögliche Spannung: QuIP#/HyperQuant zeigen Gewinn durch Hadamard+Gitter; unsere Z3 zeigt ihn gegen Lloyd/Gruppenskalen nicht. Unterschied: VQ im Block (8D+) nutzt Gitterform, nicht nur Rotation.

## Lücken/offene Fragen
- Kein Volltext gelesen; Dekodierkosten pro Gitter nicht einheitlich verglichen.
- Reiner Gitter-Gewinn (E8 vs Lloyd-Skalar, gleiche Bits, mit/ohne Rotation) bei kleinen Modellen unbekannt.
- Clifford/Stabilisator-Codebuch ungetestet.

## Runde 2 (Welle 3, c6): Nachtrag
Nur Abstracts/Suchauszüge; Zahlen ungeprüft. 4 Suchen (Rest wegen Sättigung gespart), Einträge in `neu/w3_c6.json`.
- **Trellis-Dekodierkosten:** QTIP ersetzt gespeicherte Codebücher durch rechenbasierte Gauß-Codes auf Bitshift-Trellis, Kosten linear in der Dimension; ein 16D-Vektorquantisierer wäre laut Blog 256x teurer zu dekodieren als QuIP# (Suchauszug). Ableger: CCQ (Faltungscode, 2,0 bis 2,75 Bit, Bitshift-Dekodierung) [arXiv:2507.07145], Q-Palette (gebrochene Bits, Trellis, Latenzbewusst) [arXiv:2509.20214], Proteus (tabellenfrei, ICML 2026, Llama-2 7B bis 70B, GPU).
- **CPU:** T-MAC [arXiv:2407.00088]: LUT-Kernels statt Dequantisierung, Mittel 3,6x gegen llama.cpp, 2,8x Ende-zu-Ende Llama-2-7B 2 Bit. OASIS [arXiv:2507.23035] zeigt: nichtuniforme Gitter brauchen LUT-GEMM. Keine Arbeit zu Gitter/Trellis auf Kleinmodellen (0,5B bis 1B) mit CPU-Kernel gefunden; Trellis-Ergebnisse stammen fast nur von GPU und 7B+.
- **Stabilisator/Clifford-Codebuch:** zweite Suche negativ; nur Stabilisatorzustände als Netzdarstellung (RBM) oder Zielgröße, nie als Gewichtsbuch. Lücke bleibt, ist kein Beweis.
- **Bezug zu eigenen Befunden:** kein Widerspruch; stützt die Dekodierkosten-Achse (K4-Schluss 4).
- **Quantenbezug:** CCQ/Trellis stammen aus der Kanalcodierung (Analogie zu F1), Clifford-Buch bleibt ungetestet.
- **Neuer Vorschlag (Agent):** Für Maltes Ryzen-5-Ziel Dekodierkosten als gemessene Größe (Tok/s auf CPU, Bytes gelesen) neben Bits führen; Gitter/Trellis nur dann gegen Lloyd-Skalar testen, wenn ein T-MAC-artiger LUT-Kernel oder Bitshift-Dekoder vorhanden ist. Langweiligste Erklärung: Skalar-Lloyd + Gruppenskala liegt auf CPU vorn, weil der Kernel einfach ist.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1 (a6) | 11 | ~hoch (9 Suchen, Feld vorher fast leer von Klassikern) |
| Runde 2 (Welle 3, c6) | 3 (K4) | ~45 % der relevanten Treffer neu (T-MAC schon bekannt); 4 Suchen; Clifford-Suche zum zweiten Mal negativ; Lücke Kleinmodell+CPU bleibt |

## Schlüsse und neue Wege (Agent)
1. **Rotation wirkt nur mit festem Skalenformat; die verbleibende Hebelkraft liegt im Block-Gitter.** Belege: Z3b, [arXiv:2608.25188], [arXiv:2603.11021]. Langweiligste Erklärung: Rotation glättet nur Ausreißer, Gruppenskalen tun dasselbe. Quantenbezug: gering (Hadamard als Clifford-Element).
2. **Test: E8-Gitter vs. Lloyd-Skalar bei gleichen Bits, mit/ohne Hadamard, auf Tuning-Seeds.** Erwartung nach Theorie max. ~0,65 dB (~0,1 Bit). Belege: [arXiv:2606.23406], [arXiv:2402.04396]. Langweilig: Gewinn verschwindet gegen Gruppenskala+Entropiecodierung. Quantenbezug: keiner direkt.
3. **Winkelcodebuch aus Clifford-/Kerdock-Phasen (Z4/Z8) für Paare nach Hadamard.** Belege: [arXiv:1904.07842], [arXiv:2603.27467]. Langweilig: gleichförmige Winkelquantisierung, Z4 ist nur 4 Winkel. Quantenbezug: direkt in der Struktur, klassisch in der Wirkung (I-038).
4. **Dekodierkosten als Bewertungsachse** (Bytes gelesen, nicht nur Bits): Trellis/regelmäßiges Gitter vs. Tabelle. Belege: [arXiv:2609.02652]. Langweilig: reine Kernel-Eigenschaft.

## Runde 3 (Welle 5, e6): Nachtrag
1 Suche (Limit), Treffer nur über [arXiv:2608.25188]: Gaußquelle 2 Bit, Verzerrung etwa 0,118 Lloyd-Max, 0,089 E8P, 0,069 bei Dim 256 (Suchauszug, ungeprüft); E8P ist keine volle E8-Quantisierung. Keine Ablation E8 ohne Hadamard gefunden (QuIP# isoliert das Gitter nicht). Neue Quelle: NSNQuant [arXiv:2505.18231] (festes VQ-Buch nach Hadamard). Kein Widerspruch zu Z3b.

| Runde 3 (Welle 5, e6) | 1 (geteilt mit K2) | gering; Lücke "Gitter ohne Rotation" bestätigt, Feld gesättigt |
