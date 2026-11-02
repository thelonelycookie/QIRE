# N1: Skalengesetze (auch für Präzision)

## Landkarte
- Klassik: Chinchilla [arXiv:2203.15556], Replikation [arXiv:2404.10102]; mit Inferenzkosten kleiner und länger trainieren [arXiv:2401.00448].
- Präzision: Skalengesetze für Präzision [arXiv:2411.04330], k-Bit-Inferenz: 4 Bit meist optimal bei fester Bitzahl [arXiv:2212.09720], ParetoQ: 2/3/ternär ≥ 4 Bit [arXiv:2502.02631].
- Schaden hängt von der Trainingslänge ab: kleine, lange trainierte Modelle leiden am stärksten [arXiv:2411.17691]; QAT-Gesetz mit Tokens und Gruppengröße [arXiv:2505.14302]; compute-optimales QAT [arXiv:2509.22935]; Shannon-Rahmen (Kanalrauschen) [arXiv:2605.23901]; Overtraining macht Parameter empfindlicher [arXiv:2503.19206].
- Kapazität: 2 Bit/Parameter [arXiv:2404.05405], 3,6 Bit/Parameter [arXiv:2505.24832] (siehe N8).
- Streit: 4 Bit (Dettmers) gegen 2-3 Bit (ParetoQ); die Auflösung hängt an QAT gegen PTQ und am Gruppenscale.

## Stärkste Baselines/Gegner für QIRE
- ParetoQ (QAT, 2-3 Bit) und Dettmers (4 Bit, kleine Blöcke) als Größe-Genauigkeit-Maßstab.
- Das QiD-Gesetz sagt voraus: gerade QIREs Regime (kleines, voll trainiertes Modell) hat den größten PTQ-Schaden; jede Rotation muss dort gegen einfache Gruppenscales gemessen werden.

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Token-Exponent des QiD ≈ 0,53 [arXiv:2411.17691], ungeprüft (Sekundärquelle).
- Dettmers: über 35 000 Läufe, kleine Blockgröße ≈ +1 Bit, ungeprüft.
- Springer: OLMo-1B 3T gegen 2,3T über 2 % schlechter, ungeprüft.
- Alle Zahlen dieser Welle nur aus Abstract/Suchauszug, kein Volltext gelesen.

## Quantenbezug
- Echt quanten: keiner. Quantenmodelle haben eigene Skalenarbeiten [arXiv:2604.06007] (bereits in Basis), ohne Bezug zu klassischer Präzision.
- Analogie: Shannon-Kanalbild [arXiv:2605.23901]; Quantisierung als Rauschen eines Kanals.
- Hebel: Skalengesetze als Messrahmen (Schaden gegen Tokens/Größe) nutzen, um QIRE-Gewinne regimeabhängig zu prüfen.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3 (Rotation ≈ gleichwertig gegen faire Gegner), I-038 (Wirkung klassisch). QiD erklärt, warum Gewinne nur im Regime kleiner, lang trainierter Modelle sichtbar werden, ohne Rotation zu bevorzugen.

## Lücken/offene Fragen
- Kein Volltext gelesen; QiD-Gesetz bei Modellen unter 100M Parametern nicht belegt.
- Wirkt Rotation auf den Token-Exponenten des Schadens? Unbekannt.

## Schlüsse und neue Wege (Agent)
1. **QiD-Regime als Prüfstand.** Schluss: QIRE-Vergleiche über eine Reihe Trainingslängen (Tokens je Parameter) statt eines Punktes messen. Belege: [arXiv:2411.17691], [arXiv:2401.00448]. Langweiligste Erklärung: Schaden folgt nur der Gewichtsverteilung (Ausreißer), nicht der Methode. Quantenbezug: keiner.
2. **Rotation senkt vielleicht den Token-Exponenten.** Schluss: Hadamard-Glättung der Ausreißer könnte das Anwachsen des QiD bremsen; testbar billig. Belege: Z3 neutral, [arXiv:2503.19206] (Sensibilität wächst). Langweiligste Erklärung: Effekt nur durch Gruppenscales gleichwertig ersetzbar. Quantenbezug: Analogie (Rauschkanal, [arXiv:2605.23901]).
3. **Bitzahl-Debatte mit eigenen Zahlen schließen.** Schluss: Größe in Bits statt Parametern auftragen (Dettmers-Achse) und 2/3/4 Bit in kleinem Maßstab neu fitten. Belege: [arXiv:2212.09720], [arXiv:2502.02631]. Langweiligste Erklärung: Ergebnis hängt an QAT-Budget. Quantenbezug: keiner.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| W2 b5 | 8 (N1: 5 eigene) | Anteil neu nicht berechnet |

## Runde 2 (Welle 4, d5)
- Suchen: 5 von 8. Kein Gesetz speziell für Modelle < 100M gefunden; nächste Treffer: Compression Scaling Laws mit 100M-Konfiguration bei 2 Bit [arXiv:2502.16440], QAT-Gesetz 74M-595M [arXiv:2505.14302], Ternär-Training in MLP/Kleintransformern gleichauf [arXiv:2411.05882] (neu).
- Exponenten des Schadens: Precision-Scaling: Schaden wächst mit Tokens je Parameter, sinkt mit Größe [arXiv:2411.04330]; Sekundärquelle nennt etwa 1,1 (Daten) und 0,8 (Größe), ungeprüft; QiD-Tokenexponent ≈ 0,53 [arXiv:2411.17691]. Die Zahlen unterscheiden sich (anderes Maß), nicht vergleichbar.
- Rotation im Skalengesetz: nichts gefunden (nur Block-Hadamard-Schranken [arXiv:2601.22347], DFRot [arXiv:2412.00648]). Lücke bleibt, Schluss 2 bleibt ungeprüfte Vermutung.
- Bits je Parameter bei QAT: kein neuer Konsens; Apple-Gesetz [arXiv:2509.22935] und ParetoQ bleiben die Referenz.
- Sättigung Runde 2: nur 1 neue Quelle für N1 bei 5 Suchen, Feld nahezu gesättigt; offene Lücke ist empirisch (eigene Messreihe), nicht Literatur.
