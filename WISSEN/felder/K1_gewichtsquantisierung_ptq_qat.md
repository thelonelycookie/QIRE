# K1: Gewichtsquantisierung PTQ/QAT

## Landkarte
- Linien: (a) Rundung mit Fehlerkorrektur: GPTQ, GPTAQ, Qronos [arXiv:2505.11695]; (b) Skalen/Äquivalenztransformation: AWQ, OmniQuant [arXiv:2308.13137]; (c) Inkohärenz/Rotation + Gitter: QuIP, QuIP#, QTIP, OptRot [arXiv:2512.24124], KronQ [arXiv:2607.07964]; (d) rotationsfrei sub-4-Bit: D2Quant [arXiv:2602.02546]; (e) Skalengesetze: ParetoQ, Kumar et al. (Präzision), Apple-QAT-Compute [arXiv:2509.22935].
- Stand 2025–26: Bei 4 Bit liegen GPTQ/AWQ mit Gruppe 128 nahe beieinander (Blog, ungeprüft); bei 2–3 Bit trennen sich die Verfahren stark, Rotation+Gitter (QuIP#) vorn. Fast alle Vergleiche halten die nominale Bitzahl fest, nicht die Bytes inkl. Skalen.

## Stärkste Baselines/Gegner für QIRE
- Fairer Gegner bei gleichen Bytes: Gruppenskalen (g16/g32) + asymmetrisch/Lloyd-Gitter + GPTQ/Qronos-Rundung; Skalen-Metadaten in die Bitzahl einrechnen.
- Bei Bedarf Rotationsgegner: OptRot (gelernt), QuIP#/QTIP (Gitter), MR-GPTQ [arXiv:2509.23202].
- Einziger Befund zu fester Rotation in Gruppen: Hadamard hilft bei NVFP4 (Gruppe 16) nicht, bei MXFP4 (Gruppe 32)/INT4 schon [arXiv:2509.23202].

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Qronos 2 Bit, Llama3 1B/3B/8B Wiki2: 17,8/11,4/9,3 gegen OPTQ 24,6/13,2/10,4 (ungeprüft, Sekundärquelle).
- KronQ Llama-3-70B 2 Bit: 7,93 (Abstract) gegen 8,43 (Autorenkommentar), ungeklärt (ungeprüft).
- PolarQuant: Hadamard ≈ 98 % des Gewinns (Qwen3.5-9B PPL 6,90 → 6,40), ungeprüft.
- Keine Zahl im Volltext geprüft.

## Quantenbezug
- Echt quanten: keiner. Rotation/Inkohärenz ist klassische lineare Algebra (Johnson-Lindenstrauss, Konzentration); Analogie zu Quanten-Zufallsunitaries/Designs. Hebel für QIRE: Rotation als Struktur-Prior (Butterfly/Kronecker), Wirkung klassisch (I-038).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3: stützt, dass Rotation gegen faire Gruppenskalen/Lloyd wenig bringt [arXiv:2509.23202]; Z3b Gitterartefakt passt zu Shrinkage-Bias-Befund [arXiv:2606.20381]. Rotation hilft real bei grober Skala (g32 MXFP4) und im Gradientenpfad.

## Lücken/offene Fragen
- Keine Studie vergleicht Rotation vs. Gruppenskalen strikt bytegleich in kleinen Modellen (<1B). Genau diese Lücke füllt QIRE.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| 2 (b1) | 13 neu (K1 anteilig) | siehe Schätzung |

## Schlüsse und neue Wege (Agent)
1. **Schluss:** Rotation lohnt nur, wenn die Skalengranularität grob ist (pro Kanal, g≥32 mit FP-Gitter) oder Ausreißer stark sind; bei g16 oder gutem Lloyd-Gitter ist sie gleichwertig. Belege: [arXiv:2509.23202], Z3. Langweiligste Erklärung: Gruppenskalen leisten selbst schon Ausreißer-Isolation. Quantenbezug: Analogie (Zufallsunitär).
2. **Schluss:** Neben Rotation sind Rundung (Qronos/GPTQ) und Bitzuteilung (KronQ) eigenständige Hebel; QIRE sollte Rotation gegen GPTQ-gerundete Gruppenquantisierung testen, nicht gegen RTN. Belege: [arXiv:2505.11695], [arXiv:2607.07964]. Langweilig: GPTQ-Rundung allein erklärt die Gewinne. Quantenbezug: keiner.
3. **Schluss:** Gelernte Rotation (OptRot) vs. feste Hadamard bei gleichen Bytes ist ein offener, billiger Test. Beleg: [arXiv:2512.24124]. Langweilig: mehr Freiheitsgrade, nicht Struktur. Quantenbezug: Analogie.

## Runde 2 (Welle 4, d1)
- Byte-fair im Kleinmodell: VBQ [arXiv:2607.02893] misst Qualität pro Byte bei 55-131M (131M bei 1,82 Bit schlägt 55M FP16 bei 3,8x weniger Speicher; Selbstangabe, ungeprüft). Gemischte Bitbreite je 64er-Gruppe ist damit ein neuer Gegner. Decode-Speedup erst ab ~1B (Parität bei 131M, Apple Silicon).
- GPTQ-Nachfolger: Qronos-Theorie [arXiv:2508.04853], Compensation-aware error [arXiv:2604.07955]; Rundung bleibt eigener Hebel (siehe Schluss 2).
- Kleinmodell-Sensitivität: Ausreißer sind Trainingsrezept-, nicht Skalenfolge [arXiv:2305.19268]; PTQ-Voraussagemodell [arXiv:2410.12119]; übertrainierte Modelle leiden stärker (Sekundärsnippet).
- CPU: Vec-LUT [arXiv:2512.06443], T-MAC [arXiv:2407.00088]; Ternär von Grund auf: Spectra 1.1 [arXiv:2506.23025]. Nichtuniforme Gitter mit CPU-LUT-Kern: nichts gefunden.
- Weiter Lücke: keine Studie rechnet Skalenbytes streng im <1B-Vergleich Rotation gegen Gruppenskala (Suchergebnis, kein Beweis).
- Sättigung Runde 2: 9 Treffer neu / 0 bestätigen nur Bekanntes für K1-Kernfrage; Kernlücke unverändert, also für die Hauptfrage gesättigt.

| Runde | neue Quellen | Anteil neu |
|---|---|---|
| 4 (d1, Runde 2) | 7 (K1 anteilig) | hoch bei Randthemen, 0 bei Kernlücke |

## Runde 3 (Welle 5, e1)
- Neue Begriffe gesucht (Skalen-Overhead, effektive Bits je Gewicht, Sub-1B). Byte-Buchhaltung: Skala FP16 je Gruppe g kostet 16/g Bit je Gewicht (g=128: 0,125; g=16: 1,0), Skala+Zero-Point 0,19 bei g128; AWQ g128 ca. 0,25 [arXiv:2510.11234, arXiv:2505.08620, Suchtreffer, ungeprüft]. Bei g16 sind die Skalen so teuer wie ein halbes 2-Bit-Gewicht: der faire Vergleich muss sie einrechnen (stützt S-002/S-003).
- Neuer rotationsfreier Gegner MSB [arXiv:2509.03054] (Llama 3.2 3B, 4 Bit blockweise 8,43 gegen GPTQ 12,23; Abstract bestätigt, Zahl ungeprüft).
- Kleinmodelle: Einbettung, Normen, LM-Kopf bleiben oft unquantisiert; bei <1B verzerrt das die "Bit je Gewicht"-Angabe (eigene Überlegung, keine Quelle). Für QIRE: Bytes der Gesamtdatei berichten.
- Keine Studie Sub-1B Rotation gegen Gruppenskala bytegleich gefunden; Kernlücke unverändert.
- Sättigung Runde 3: 1 Suche K1, 1 neue Quelle, 0 zur Kernlücke: gesättigt.

| Runde | neue Quellen | Anteil neu |
|---|---|---|
| 5 (e1, Runde 3) | 1 | niedrig; Kernlücke 0 |
