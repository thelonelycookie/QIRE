# N8: Rate-Distortion- und Informationsgrenzen

## Landkarte
- Rate-Distortion-Sicht auf Gewichte: Radio [arXiv:2505.03031], Foundations Part 1 [arXiv:2409.02026]; Waterfilling-Untergrenze mit Aktivierungs-Kovarianz (MIT-MEng-Arbeit 2026, nur als Suchtreffer, nicht eingetragen); universelles Codebuch kostet ≤ 0,11 Bit/Dim [arXiv:2602.05790]; NestQuant (Gitter, informationstheoretisch optimal für Matrixprodukte) [arXiv:2502.09720].
- Verlustfreie Grenze: Entropie der Gewichtsverteilung [arXiv:2606.15789]; gelernte Codecs [arXiv:2510.11234].
- Kapazität: 2 Bit/Parameter [arXiv:2404.05405] (int8 ohne Verlust, int4 über 2x Kapazitätsverlust), 3,6 Bit/Parameter [arXiv:2505.24832]; klassische Gardner-Kapazität etwa 1-2 Bit/Gewicht (Suchauszug, ungeprüft, nicht eingetragen).
- Quantenbezug: Holevo (n Qubits liefern höchstens n klassische Bits); QNN-Kapazität ≤ Parameterzahl, kein Vorteil bei klassischen Gewichten [arXiv:1908.01364]; Informationsgrenzen des Quantenvorteils [arXiv:2101.02464]; Speichervorteil nur für SVM/PCA im Datenstrom [arXiv:2604.07639].

## Stärkste Baselines/Gegner für QIRE
- NestQuant/QuIP#/WaterSIC-artige Gitter: nahe an der Shannon-Grenze (Lücke etwa 0,1-0,25 Bit). Wenig Spielraum für Rotations-Tricks.
- Entropiekodierung (Neural Weight Compression, Lossless-Studie) als anderer Weg unter denselben Bits.

## Wichtige Zahlen (mit geprüft/ungeprüft)
- 3,6 Bit/Parameter, ungeprüft; 2 Bit/Parameter, ungeprüft.
- ≤ 0,11 Bit/Dim Universalitätspreis [arXiv:2602.05790], ungeprüft (Abstract).
- WaterSIC ≈ 0,255 Bit über Grenze: nur Suchauszug, nicht eingetragen.
- Kein Volltext gelesen.

## Quantenbezug
- Echt quanten: Holevo-Grenze und QNN-Kapazität. Folgerung: ein Quantenzustand speichert nicht mehr abrufbare klassische Information als ein Bit je Qubit; klassische Gewichte in QNN geben keinen Kapazitätsvorteil [arXiv:1908.01364].
- Analogie: Rate-Distortion mit Seiteninformation (Kovarianz) ähnlich Quantenmessung mit Vorwissen; Gitterkodierung (E8) als geometrische, nicht quantenmechanische Idee.
- Hebel für QIRE: ehrlicher Rahmen ("quanten-inspirierter Vorteil"): Holevo und I-038 stützen die Aussage, dass die Wirkung klassisch bleibt; Gewinn nur aus besserer Annäherung an die R(D)-Grenze.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3: Rotation ≈ gleichwertig; passt zu 0,11-Bit-Universalität: Rotation holt nur Restabstand zur Waterfilling-Grenze. I-038: Wirkung klassisch, von Holevo/Wright gestützt.

## Lücken/offene Fragen
- Keine verlässliche Bit/Gewicht-Untergrenze für reale Modelle (Annahme Gauß).
- Kein Volltext; Waterfilling-Thesis nicht verifiziert (nur Hochschul-PDF).

## Schlüsse und neue Wege (Agent)
1. **Grenzabstand als Messgröße.** Schluss: QIRE-Gewinne in Bit-Abstand zur Gauß-R(D)-Kurve angeben (bei 2 Bit Gauß: D = 2^-4). Belege: [arXiv:2602.05790], [arXiv:2502.09720]. Langweiligste Erklärung: Gewichte sind nicht gaußisch, Kurve nur Näherung. Quantenbezug: Analogie.
2. **Kapazität gegen Bits.** Schluss: Bei 3,6 Bit/Parameter Kapazität sollte Verlust bei 2-3 Bit erst ab ausgelasteten Modellen auftreten; passt zum QiD-Befund (siehe N1). Belege: [arXiv:2505.24832], [arXiv:2404.05405]. Langweiligste Erklärung: Kapazitätsmessung betrifft Memorieren, nicht Sprachaufgaben. Quantenbezug: keiner.
3. **Holevo als Ehrlichkeitsabschnitt für die schriftliche Arbeit.** Schluss: Abschnitt, warum QIRE keinen Quantenvorteil beansprucht (Holevo, QNN-Kapazität, Datenstrom-Vorteil nur für SVM/PCA). Belege: [arXiv:1908.01364], [arXiv:2101.02464], [arXiv:2604.07639]. Langweiligste Erklärung: zu allgemein, ohne QIRE-Bezug. Quantenbezug: direkt.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| W2 b5 | 10 | Anteil neu nicht berechnet |

## Runde 2 (Welle 4, d5)
- WaterSIC jetzt als Quelle: [arXiv:2603.04956] (Lücke ≤ 0,255 Bit zur Grenze, Spalten-Waterfilling; GPTQ kann beliebig weit entfernt sein), Theorie [arXiv:2601.17187], [arXiv:2605.13768]. Wichtig: GPTQ mit Zufallsrotation liegt je Schichttyp innerhalb 0,1 Bit von WaterSIC (Llama-3-8B-Σ), Abstract, ungeprüft. Das stützt Z3/S-001: Rotation plus GPTQ holt fast alles; für Rotations-Neuheit bleibt <= 0,1 Bit.
- Entropiekodierung praktisch: [arXiv:2505.18758], EntQuant [arXiv:2601.22787] (Snippets, ungeprüft): entropiebeschränkte Quantisierung als Gegner bei gleichen Bytes.
- Bits je Parameter bei QAT: kein neuer Befund; Kleinmodelle ternär gleichauf [arXiv:2411.05882].
- Lücke bleibt: keine Grenze für reale, nicht-gaußsche Gewichte kleiner Modelle; Skaleneffizienz der Waterfilling-Idee (Spaltenraten) für QIRE ungeprüft.
- Neuer Weg (Vorschlag): Spaltenraten nach Aktivierungsvarianz als billiger Gegner testen. Langweiligste Erklärung: entspricht gemischter Genauigkeit. Quantenbezug: Analogie (keine).
- Sättigung Runde 2: 3 neue Kernquellen bei 3 Suchen, N8 noch nicht gesättigt (Anteil neu hoch).
