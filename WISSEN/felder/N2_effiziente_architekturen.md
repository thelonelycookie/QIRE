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
| Welle 4, d6 (Runde 2) | 12 | hoch; 7 Suchen, alle brachten Neues; Kleinmodell-Lücke bleibt aber (Daedalus 150M, PAM 5–100M sind die einzigen Treffer unter ~150M) |

## Schlüsse und neue Wege (Agent)
1. Schluss: I-049 (flach gewinnt lang) ist mit der Hybrid-Literatur vereinbar; nächster fairer Test wäre flach + wenig Attention/Gating (Griffin-Stil) statt reiner LRU. Belege: [arXiv:2406.07887], [arXiv:2402.19427]. Langweiligste Erklärung: Frühregime-Effekt und ungetunte Lernrate. Quantenbezug: Analogie (Phase), kein Vorteil.
2. Schluss: Gated DeltaNet [arXiv:2412.06464] als stärkster neuer Rekurrenzgegner vor jeder LRU-Neuheitsbehauptung aufnehmen. Langweiligste Erklärung: es ist nur ein besserer Optimierer/Speicherregel. Quantenbezug: keiner.
3. Schluss: MoE für QIRE-Maßstab nicht einplanen, solange kein Kleinmodellbeleg vorliegt [arXiv:2402.07871, ungeprüft]. Langweiligste Erklärung: Router-Overhead dominiert bei kleinen Modellen. Quantenbezug: keiner.

## Runde 2 (Welle 4, d6)
Nur Abstracts über arXiv-API bestätigt, kein Volltext, alle Zahlen ungeprüft.
- **Byte-Ebene:** MEGABYTE [arXiv:2305.07185], BLT [arXiv:2412.09871] (FLOP-kontrollierte Studie bis 8B), MambaByte [arXiv:2401.13660]. H-Net [arXiv:2507.07955] steht schon im Feld. Alles Großmodell; kein Beleg unter 100M.
- **CPU/Kleinmodell:** Daedalus-150M [arXiv:2608.20210]: Attention in 6 von 18 Blöcken, Rest Kurzfaltung, 1,76x schneller bei 2048 Kontext (Selbstangabe, Einzelautor); berichtet ehrlich 4-Bit-Verlust und tote Kanäle. Nemotron-Flash [arXiv:2511.18890]: reale Latenz entscheidet, Tiefe/Breite und Operatorwahl zählen mehr als Parameter. Hymba [arXiv:2411.13676]. Hybrid-Anteil-Scan 340M/1,3B [arXiv:2507.06457], Hybrid-Systematik [arXiv:2510.04800].
- **Phasen/komplexe Rekurrenz:** Mamba-3 [arXiv:2603.15569]: komplexe SSM = datenabhängige Rotation, löst Zustandsverfolgung (Stand der Technik, klassisch). PAM [arXiv:2604.05030]: komplex 5M–100M auf WikiText-103 bei jeder Größe schlechter als reelle Ablation, nur steilerer Exponent (−0,15 gegen −0,12; Selbstangabe). PRISM [arXiv:2512.01208] (Analogie, vorsichtig zitieren). Bereits im Bestand: [arXiv:2602.22255], [arXiv:1511.06464].
- **Quantisierung von SSM:** Mamba-PTQ [arXiv:2407.12397] (Ausreißerkanäle), im Bestand Quamba [arXiv:2410.13229], Quamba2 [arXiv:2503.22879], MambaQuant [arXiv:2501.13484], Gated DeltaNet 4-Bit [arXiv:2609.04098]. Ergebnis: Ausreißer und Zustandsquantisierung sind das Problem, Hadamard hilft bei den Ausgaben (Quamba).
- **Nicht gefunden (nach gezielter Suche):** Gated DeltaNet unter 100M auf Byte-Ebene; Hybridanteil-Scan unter 100M; CPU-Messung von Rekurrenz-Kernen bei Kleinmodellen mit Joule.

### Schlüsse und neue Wege (Runde 2, Agent)
1. Schluss: Phasen-/Rotations-Rekurrenz bringt im Kleinmodell keinen Vorteil: PAM ist bei 5–100M bei jeder Größe schlechter als reell [arXiv:2604.05030]; Mamba-3 nutzt Rotation nur für Zustandsverfolgung [arXiv:2603.15569]. Langweiligste Erklärung: Komplex kostet doppelt Zustand, Gewinn nur bei Aufgaben, die QIRE-Byte-LMs nicht prüfen. Quantenbezug: Analogie (Phase, Unitarität). Stützt Z3/S-001.
2. Schluss: Eine Messreihe „Hybridanteil und Rekurrenzvariante bei 1–50M, Byte-Ebene, CPU-Wallclock und Joule“ füllt eine echte Lücke (nächste Arbeiten: 340M [arXiv:2507.06457], 150M [arXiv:2608.20210]). Langweiligste Erklärung: Lücke besteht, weil Ergebnisse bei kleinem Maßstab nichts Neues zeigen. Quantenbezug: keiner; passt zu S-003.
3. Schluss: Gewinnkriterium vor dem Lauf festlegen, auch Misserfolge berichten, so wie Daedalus [arXiv:2608.20210] und Nemotron-Flash [arXiv:2511.18890] (Latenz statt Parameter). Langweiligste Erklärung: reine Methodik. Quantenbezug: keiner.
4. Schluss: Bei Quantisierung der Rekurrenz ist der Zustand die Schwachstelle; Hadamard auf den Ausgaben wirkt dort (Quamba [arXiv:2410.13229]). Das ist ein möglicher Ort, wo Rotation sinnvoll bleibt (S-007). Langweiligste Erklärung: wieder nur Ausreißerglättung. Quantenbezug: Analogie (Hadamard/Clifford), direkt als Baustein.
