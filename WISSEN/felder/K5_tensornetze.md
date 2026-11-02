# K5: Tensornetze (MPO/MPS/TT)

Stand 02.11.2026, Welle 2 (b3). Nur Abstracts gelesen, kein Volltext. Zahlen daher ungeprüft.

## Landkarte
- **Kompression vortrainierter LLMs:** CompactifAI [arXiv:2401.14109] (MPO je Gewichtsmatrix, Bonddimension χ, kurzes „Heilen“), TensorGPT [arXiv:2307.00526] (Einbettungen), MPO-Zerlegung für Transformer [arXiv:2603.28534], Minima [arXiv:2602.01613] (Tucker/TT/TR gemischt, eigene Kernel, Ziel: echter Serving-Gewinn). Fremdbewertung CompactifAI: [arXiv:2507.08836] (Herstellerbezug, Abstract: Genauigkeit erhalten).
- **Kritik:** [arXiv:2606.03465] (ICML 2026): Tensorzerlegung nach dem Training hat engen Nutzen, weil gemeinsame Unterräume nicht zu den heterogenen LLM-Darstellungen passen. Übersicht [arXiv:2608.30505] führt ρ_gap (Lücke Parameterersparnis vs. gemessene Beschleunigung) ein.
- **Training von Grund auf bei gleichem Rechenaufwand (für QIRE zentral):** [arXiv:2406.06248] und [arXiv:2410.02117]. Naives TT und Kronecker liegen unter dicht; Block-Tensor-Train (BTT, enthält Monarch) liegt darüber. Regel: großer Rang, wenig Parameterteilung, viele Parameter pro FLOP.
- **MPS/MPO-Klassifikatoren:** [arXiv:1605.05775], [arXiv:1905.01426], tiefe MPO-Netze [arXiv:2209.09098] (0,49 % MNIST-Fehler, 8,3 % Fashion; Abstract).
- **Tensor-Attention:** TPA [arXiv:2501.06425], Tensorized Attention [arXiv:2410.20926], Kronecker-Attention [arXiv:2412.02919]. Das sind meist Kronecker-/Tucker-Strukturen, keine echten Tensornetze.
- **Training mit Simulationsverfahren:** DMRG-artiger Zwei-Orts-Sweep für MPO-Netze [arXiv:2211.14657].

## Stärkste Baselines/Gegner für QIRE
1. **BTT/Monarch bei gleichen FLOPs** [arXiv:2406.06248]: der faire Gegner für jeden TT-FFN-Test. BTT-MoE [arXiv:2410.02117] schlägt dicht bei GPT-2 im Compute-Optimum.
2. Dichter FFN mit gleicher FLOP-Zahl (kleineres h) und Niedrigrang (wie I-045).
3. Für Kompression nach dem Training: gutes PTQ (K1) schlägt Tensorzerlegung häufig (Vermutung, [arXiv:2606.03465] deutet es an; nicht geprüft).

## Wichtige Zahlen (mit geprüft/ungeprüft)
| Zahl | Quelle | Status |
|---|---|---|
| BTT = dicht bei ViT-S/32 (ImageNet-1k) mit 3,8x weniger Rechenaufwand | [arXiv:2406.06248] | ungeprüft (Abstract) |
| MPO-Netz 0,49 % MNIST-, 8,3 % Fashion-Fehler | [arXiv:2209.09098] | ungeprüft |
| CompactifAI LLaMA-2 7B ca. 93 % Speicher, 2–3 % Genauigkeit | [arXiv:2401.14109] (Sekundärquellen) | ungeprüft, Hersteller |

## Quantenbezug
- **Echt Quanten:** nichts. MPS/MPO stammen aus der Vielteilchenphysik (DMRG), werden aber auf klassischer Hardware gerechnet; die Wirkung ist klassische Niedrigrang-/Strukturregularisierung (I-038).
- **Analogie:** Bonddimension χ ≙ Verschränkungsbudget; Truncation per SVD ≙ Verschränkungsschnitt.
- **Quanten-inspirierter Hebel für QIRE:** nicht „TT statt dicht“ (verliert, I-047), sondern (a) adaptive Bonddimension per Zwei-Orts-DMRG-Sweep [arXiv:2211.14657], (b) BTT/Monarch-artige Struktur mit großem Rang, die als „Schaltkreis aus Brickwork-Lagen“ gelesen werden kann (Verbindung zu Hadamard/Fastfood, K7).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- **I-047** (TT-FFN 0,06 bpc schlechter als dicht): stimmt mit [arXiv:2406.06248] überein (TT ohne strukturspezifische Skalierung unterliegt). Möglicher Widerspruch zu eigenen Einstellungen: Wurde die Lernrate/Initialisierung (muP-artig) für TT angepasst? Falls nicht, ist I-047 kein Beleg gegen TT, sondern gegen naive TT. Zu prüfen im Laborbuch.
- **I-045** (Fastfood verliert bei gleichem Rechenaufwand): passt zur Regel „kleiner Rang/geteilte Parameter schlecht“ [arXiv:2410.02117].
- Z3 (Rotation ≈ faire Gegner): gleichwertig, kein Widerspruch.

## Lücken/offene Fragen
- Keine Arbeit gefunden, die TT-FFN in Sprachmodellen bei gleichen FLOPs mit angepasster Skalierung gegen BTT vergleicht (nur Qiu et al. mit GPT-2 klein).
- Keine unabhängige Reproduktion der CompactifAI-Heilung (nur Herstellerstudie).
- 2025–2026 kaum neue MPS-Klassifikatoren-Arbeiten gefunden (eine Suche).
- Echte Tensornetz-Attention (MPS/PEPS) nicht belegt; alles Kronecker/Tucker.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2, b3 | 10 (K5 und F6 zusammen, 9 Suchen K5) | hoch, Feld nicht gesättigt |

## Schlüsse und neue Wege (Agent)
1. **I-047 mit muP-Skalierung wiederholen oder BTT als Arm ergänzen.** Beleg: [arXiv:2406.06248] (Kronecker/TT unter dicht nur ohne passende Init/LR; BTT darüber). Langweiligste Erklärung: I-047 hatte schlicht falsche Lernrate/Init für TT, 0,06 bpc ist Rauschen plus Fehlkalibrierung. Quantenbezug: Analogie (Bonddimension, Schaltkreis-Lagen).
2. **Regel „viele Parameter pro FLOP, voller Rang“ als Entwurfsprinzip für QIRE-Schichten.** Beleg: [arXiv:2410.02117]. Langweiligste Erklärung: es ist einfach Monarch/Kronecker-Sparsität, nichts Quantenhaftes. Quantenbezug: nur als Schaltkreis-Analogie (Brickwork).
3. **Adaptive Bonddimension per DMRG-Sweep** statt fester χ. Beleg: [arXiv:2211.14657] (nur MNIST/Regression). Langweiligste Erklärung: adaptive Rangwahl ist generische Rangauswahl (AdaLoRA-artig). Quantenbezug: direkt übertragenes Simulationsverfahren, aber klassische Wirkung.
4. **Ehrliche Aussage für die Arbeit:** Post-Training-Tensorkompression hat engen Nutzen [arXiv:2606.03465]; nicht als QIRE-Vorteil verkaufen.

## Runde 2 (Welle 4, d3)
8 Suchen, nur Abstracts (arXiv-API bestätigt), Zahlen ungeprüft.
- **Init/Skalierung für TT:** Standardinit liegt bei Tensornetzen falsch (Varianz des Produkts über Kerne) [arXiv:2205.15307]; Kernvarianz-Formel Var(w)=σ^(2n)·∏r_k [arXiv:2006.05442, aus Snippet]; Init über partielle Normen [arXiv:2309.06577]. Keine Arbeit zu muP/LR-Skalierung für TT gefunden (Suche ergab nur dichte muP). Die dichte-äquivalente Init in I-047 deckt den Varianzpunkt ab; fehlend bleibt die LR-Skalierung je Kern.
- **TT von Grund auf im kleinen LM:** GPT-2 mit TT-Matrix, ca. 40 % weniger Parameter bei ähnlicher Perplexität [arXiv:2306.02697]; gleiche FLOPs nicht belegt. Übersicht [arXiv:2608.30505] nennt nur zwei Methoden mit Perplexität auf zwei Größen (Snippet).
- **Tensornetz plus Quantisierung:** [arXiv:2306.01076] (QAT auf TT-Kernen, NLU), [arXiv:2512.09202] (PINN). Im LM-Pretraining nichts gefunden.
- **Adaptiver Rang:** MetaTT [arXiv:2506.09105] (hoher TT-Rang, dann DMRG-artige SVD-Kompression, Adam-Momente neu starten); bestätigt Idee aus Runde 1, nur für Feintuning.
- **MERA:** MERA ersetzt FC-Schichten in CNNs, ca. 1 % Genauigkeitsverlust [arXiv:1711.03357]; Baum-/Multiskalen-Klassifikatoren [arXiv:1710.04833], [arXiv:2001.08286]. Alles klein (CIFAR/MNIST), nichts für LMs.
- **Neuer Lückenbefund:** kein Vergleich TT/MERA gegen BTT bei gleichen FLOPs im LM.

Sättigung Runde 2: 8 Suchen, 8 neue Quellen von ca. 14 Treffern (ca. 55 % neu); Feld weiter nicht gesättigt.
Neuer Schluss: Für I-047-Wiederholung zuerst Init prüfen (Partialnorm), dann LR je Kern; BTT als Pflichtarm. Langweiligste Erklärung bleibt: bei kleinem d ist dicht optimal.

## Runde 3 (Welle 5, e2)
2 Suchen (LR-Skalierung je TT-Kern gegen BTT/Monarch; MPS/Baum-Tensornetz-Attention im LM). Kein neuer bestätigter Eintrag.
- Nichts Neues zu muP/LR je Kern: die Skalierungsregeln stehen weiter nur in [arXiv:2406.06248]; GPT-2-TT [arXiv:2306.02697] nutzt einen einzigen Zeitplan ohne Kernskalierung.
- Keine MPS-/Baum-Tensornetz-Attention für LMs gefunden (nur Kronecker-/Tucker-Arbeiten, schon im Feld; MPS-Sprachmodell von 2020 ist ein Wahrscheinlichkeitsmodell, nicht bestätigt).

Sättigung Runde 3: 2 Suchen, 0 neue Quellen (0 % neu); K5 gesättigt, Lücken bleiben echte Lücken (TT/MERA gegen BTT bei gleichen FLOPs, LR je Kern).
