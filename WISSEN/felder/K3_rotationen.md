# K3: Rotationen (Hadamard, Butterfly, gelernt)

## Landkarte
- Feste Rotation: QuaRot [arXiv:2404.00456] und QuIP# [arXiv:2402.04396] drehen mit (randomisiertem) Hadamard, damit Ausreißer verschwinden. Hadamard ist theoretisch besser als zufällige Rotation [arXiv:2504.13989].
- Gelernte Rotation: SpinQuant (Cayley auf Stiefel) [arXiv:2405.16406], OSTQuant (Rotation + diagonale Skalierung) [arXiv:2501.13987], ButterflyQuant [arXiv:2509.09679], HARP (butterfly-artig, Start bei RHT) [arXiv:2605.29843]. Billige Parametrisierung: Givens-Ketten, O(d) Parameter [arXiv:2404.04316] (Fine-Tuning, nicht Quantisierung).
- Rotation plus Permutation: DuQuant [arXiv:2406.01721] gegen massive Ausreißer.
- Theorie: Range Reduction, jede Orthogonale genügt [arXiv:2601.21069]; RHT-Beweis [arXiv:2605.06014]; Gaußisierung/Crest-Faktor O(sqrt(log d)). Gegenstimme: BASE-Q zeigt, dass Rotation Kanalmittel nicht ausgleicht und Weiteres-Lernen wenig bringt [arXiv:2506.15689]. Survey 2026 "Great Inversion": Rotation hilft bei gruppenskalierter Quantisierung, nicht bei Codierung mit freier Bitzuteilung [arXiv:2608.25188].
- Wer gegen wen: Rotation vs. Gruppenskalen/Smoothing/asymmetrisch; gelernt vs. fester Hadamard (SpinQuant: gelernt bis +16,2 Punkte vs. zufällige Rotation, aber Hadamard-Zufallsstreuung bis 6 Punkte; ReSpinQuant: gelernte Rotationen bleiben nah am Hadamard-Start [arXiv:2604.11080]).

## Stärkste Baselines/Gegner für QIRE
- Feine Gruppenskalen (g16/g32), asymmetrische Quantisierung mit Bias (BASE-Q), Lloyd-Gitter, OSTQuant (Skalierung gelernt).
- Fester randomisierter Hadamard als Gegner jeder "gelernten" QIRE-Rotation (kaum Gewinn laut BASE-Q, ReSpinQuant).

## Wichtige Zahlen (mit geprüft/ungeprüft)
- SpinQuant: bis +16,2 Punkte gelernt vs. zufällige Rotation; Hadamard-Streuung bis 6 Punkte; W4A4KV4 LLaMA-2-7B Lücke 2,9 Punkte. Ungeprüft (nur Abstract/Suchauszug).
- OSTQuant: W4-only 99,5 % FP-Genauigkeit; W4A4KV4 LLaMA-3-8B Lücke -32 % vs. SOTA. Ungeprüft.
- Keine Volltexte gelesen, alle Zahlen ungeprüft.

## Quantenbezug
- Echt quanten: nichts. Hadamard ist das Walsh-Hadamard-Gatter bzw. Clifford-Struktur; Stabilizer-Rényi-Entropie via XOR-FWH [arXiv:2512.24685] ist Mathematik-Verwandtschaft, kein Werkzeug.
- Analogie: Rotation = unitäre Basiswahl; Butterfly = Schaltkreis aus 2-Qubit-Gattern; Givens = elementare Rotation. Quanten-inspirierter Hebel: Schaltkreis-Parametrisierung (Butterfly/Givens) als billige gelernte Rotation mit O(d log d) bzw. O(d) Parametern.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3 (Raven): Rotation ≈ gleichwertig gegen faire Gegner. Stützt: BASE-Q (Lernen bringt wenig, Mittelwert-Fehler), ReSpinQuant, Great Inversion (Rotation nützt nur bei gruppenskalierter Quantisierung).
- Möglicher Widerspruch: SpinQuant/OSTQuant zeigen große Gewinne, aber gegen schwache Basis (zufällige Rotation, einfaches RTN); bei LLaMA-Skala, nicht Raven-Skala.

## Lücken/offene Fragen
- Kein sauberer Test "gelernte Rotation vs. fester RHT bei gleichem Gruppenskalen-Budget" gefunden.
- Givens/Cayley für Quantisierung nicht gefunden (nur Fine-Tuning); Permutation (DuQuant) in QIRE ungetestet.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2 (b2) | 6 (K3) | ca. 7 % von ~85 K3-Quellen |

## Schlüsse und neue Wege (Agent)
1. **Rotation lohnt vor allem ohne feine Gruppenskalen.** Belege: [arXiv:2506.15689], [arXiv:2608.25188], Z3. Langweiligste Erklärung: Gruppenskalen fangen Ausreißer schon ab, Rotation wird redundant. Quantenbezug: keiner.
2. **Gelernte Rotation ≈ Hadamard-Start.** Belege: ReSpinQuant, HARP-Initialisierung, BASE-Q. Langweiligste Erklärung: Optimierung bleibt im Becken des Starts. Konsequenz: Hadamard fest lassen, Budget in Skalen/Bias stecken. Quantenbezug: Analogie (feste Clifford-Gatter reichen).
3. **Test-Idee:** Permutation + Hadamard (DuQuant-Zickzack) gegen reinen Hadamard auf Raven, gleiche Bytes. Langweiligste Erklärung: Permutation ändert nichts bei kleinen Modellen ohne massive Ausreißer. Quantenbezug: Analogie (SWAP-Gatter).

## Runde 2 (Welle 4, d2)
Suchen: Online-Hadamard-Laufzeit, Rotation im Gradientenpfad, Rotation + Bitflips, Fester vs. gelernter Hadamard bei gleichem Budget (5 Suchen).
- Laufzeit: Fusion entscheidet (QuaRot: 3 Online-Hadamards ca. 0,26x Verlangsamung, fusioniert ca. 0,07x laut Survey [arXiv:2608.25188], ungeprüft); bei nicht-Zweierpotenz-Kopfzahl große Latenz [arXiv:2503.19353]. Kein sauberer End-to-End-W4A4-Wert gefunden.
- Gradientenpfad: nur Feintuning mit Hadamard in Vor- und Rückwärtspass (HALO [arXiv:2501.02625]); Rotation im Vortraining mit STE nicht gefunden.
- Bitflips: nur RoR (Householder, +9 % Latenz, Selbstbericht) [arXiv:2603.16382]; kein Hadamard-Test.
- Fest vs. gelernt: keine Quelle mit gleichem Gruppenskalen-Budget; InfoQuant zeigt ca. 2 PPL-Punkte für gelernt (LLaMA, ohne Budgetkontrolle) [arXiv:2605.26175], PolarQuant ca. 98 % Gewinn schon durch Hadamard allein [arXiv:2603.29078]. Lücke bleibt.
- Neu: QUAD [arXiv:2503.19353], Perplexity-Kritik [arXiv:2609.23125] (Fähigkeitstests ergänzen).
- Neue Idee: Rotationsvergleich zusätzlich mit Retrieval-/Induktionstest bewerten; Bitflip-Robustheit von Hadamard in QIRE billig testbar (Analogie: Fehlerverteilung wie Stabilizer-Code).
- Sättigung Runde 2: 2 neue Quellen / 5 Suchen, ca. 2 % von ~87; Feld weitgehend gesättigt.
