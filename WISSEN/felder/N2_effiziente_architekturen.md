# N2: Effiziente Architekturen (SSM, Hybrid, MoE, LRU)

## Landkarte
- Hauptlinien: (1) lineare Sequenzmodelle: SSM (Mamba [arXiv:2312.00752]), gated Rekurrenz (Griffin/RG-LRU [arXiv:2402.19427], LRU [arXiv:2303.06349]), RWKV [arXiv:2404.05892, arXiv:2503.14456], Delta-Regel [arXiv:2412.06464]; (2) Hybride mit wenig Attention [arXiv:2406.07887, arXiv:2408.15237]; (3) MoE [arXiv:1701.06538, arXiv:2402.07871]; (4) Hierarchie/Hourglass [arXiv:2110.13711, arXiv:2507.07955].
- Übersichten: [arXiv:2508.09834], [arXiv:2507.19595]: lineare Familien konvergieren zu einem Rahmen; Hybride sind der Trend.
- Wer gegen wen: Hybride (Mamba-2 + ~7 % Attention) schlagen reine Transformer und reine SSMs im 8B-Maßstab; reine SSMs schwach bei Copy/In-Context-Learning.
- Gesucht, nicht gefunden: belastbare Kleinmodell-/CPU-Arbeiten (unter 100M, Byte-Ebene). Fast alles ist 1B+ auf GPU.

## Stärkste Baselines/Gegner für QIRE
- Flaches Modell bei gleichen FLOPs (eigene Linie, I-049).
- Griffin/Hawk (RG-LRU + lokale Attention), Gated DeltaNet, Mamba-2-Hybrid: die Gegner für "Rekurrenz statt Attention".
- MoE: bei kleinen Modellen Vorteil nicht belegt [arXiv:2402.07871 zeigt Gewinn wachsend mit Größe].

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Hybrid +2,65 Punkte über 12 Aufgaben gegen 8B-Transformer, bis 8x schneller (ungeprüft, nur Abstract) [arXiv:2406.07887].
- Griffin gleich Llama-2 mit über 6x weniger Token (ungeprüft) [arXiv:2402.19427].
- Kein Volltext gelesen, alle Zahlen ungeprüft.

## Quantenbezug
- Echt quanten: keiner. LRU ist komplexe diagonale Rekurrenz (unitär verwandt, Analogie zu Phasen/Rotationen) [arXiv:1511.06464].
- Hebel für QIRE: Rotation/Phase im Zustand ist klassisch längst Stand der Technik (RG-LRU, LRU); kein Neuheitsanspruch, aber nutzbare Grundlage.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- I-049: Hourglass+LRU bei langem Training schlechter als flach. Literatur stützt das nicht direkt, erklärt es aber: Hourglass-Gewinne [arXiv:2110.13711] kommen bei großem Budget und gleichem Kontext; Hybride gewinnen erst mit etwas Attention [arXiv:2406.07887], reine Rekurrenz verliert bei Copy/ICL. Im Frühregime (190 Schritte) sind Gewinne gewöhnlich Optimierungsvorteile.
- Z3 (Rotation ≈ gleichwertig gegen faire Gegner): passt zum Bild, dass Rotation/LRU kein Alleinstellungsmerkmal ist.

## Lücken/offene Fragen
- Kein Beleg für Hourglass+LRU bei kleinen Byte-LMs mit langem Training.
- CPU-Kleinmodell-Literatur dünn; Quantisierung der Rekurrenz (siehe K-Felder, Quamba2) offen.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2, b6 | 7 | hoch (Felder fast leer); 6 von ~7 Suchen brachten Neues |

## Schlüsse und neue Wege (Agent)
1. Schluss: I-049 (flach gewinnt lang) ist mit der Hybrid-Literatur vereinbar; nächster fairer Test wäre flach + wenig Attention/Gating (Griffin-Stil) statt reiner LRU. Belege: [arXiv:2406.07887], [arXiv:2402.19427]. Langweiligste Erklärung: Frühregime-Effekt und ungetunte Lernrate. Quantenbezug: Analogie (Phase), kein Vorteil.
2. Schluss: Gated DeltaNet [arXiv:2412.06464] als stärkster neuer Rekurrenzgegner vor jeder LRU-Neuheitsbehauptung aufnehmen. Langweiligste Erklärung: es ist nur ein besserer Optimierer/Speicherregel. Quantenbezug: keiner.
3. Schluss: MoE für QIRE-Maßstab nicht einplanen, solange kein Kleinmodellbeleg vorliegt [arXiv:2402.07871, ungeprüft]. Langweiligste Erklärung: Router-Overhead dominiert bei kleinen Modellen. Quantenbezug: keiner.
