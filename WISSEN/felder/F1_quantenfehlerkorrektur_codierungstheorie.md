# F1: Quantenfehlerkorrektur/Codierungstheorie

Stand Welle 1, Agent a1 (≈ 9 Suchen). Nur Abstracts/Suchauszüge, kein Volltext; alle Zahlen ungeprüft.

## Landkarte
- **Codes für Gewichte im Speicher:** SECDED/Hamming [arXiv:2508.12347], selektiver Schutz wichtiger Bits [arXiv:2001.03814], Hamming-Umcodierung quantisierter Gewichte (DeepNcode) [arXiv:2405.13891], reellwertige lineare Prüfungen auf Gewichten ohne Zusatzparameter [arXiv:2602.00076], billige Alternative: Gewichte mit Konstanten skalieren [arXiv:2411.18993].
- **Codes für Ausgaben/Labels:** Paritätscode-Klassifikation (CodNN) [arXiv:2004.10700]; ECOC gegen Gewichtsfehler (NeurIPS 2024, Yu et al., keine arXiv-ID gefunden, daher nicht eingetragen).
- **Bitfehler-Robustheit niedrigbittiger Netze ohne Code:** Clipping + Random Bit Error Training [arXiv:2006.13977], [arXiv:2104.08323]. Bei Soft Errors sind Integer-Formate robuster als FP16/BF16 [arXiv:2606.02430].
- **Angriffsseite:** Wenige Vorzeichenbits genügen, Codebuchschutz ist umgehbar [arXiv:2502.07408].
- **Rotation als Schutz:** Householder-Rotation senkt Bitflip-Kollaps von 3,15 % auf 0 % (Qwen2.5-7B, ungeprüft) [arXiv:2603.16382].
- **Umgekehrte Richtung (NN für QEC):** neuronale Decoder (AlphaQubit-Linie, Mamba-/Transformer-Varianten) [arXiv:2510.22724], [arXiv:2509.03815]. Hilft QIRE nicht direkt.
- Gegeneinander: ECC-Redundanz (Fläche ~47,5 %) gegen redundanzfreie Tricks (Clipping, Skalierung, Rotation).

## Stärkste Baselines/Gegner für QIRE
- Weight Clipping + RandBET [arXiv:2006.13977]: einfach, ohne Redundanz.
- SECDED [arXiv:2508.12347] als „volle“ Lösung.
- Gewichtsskalierung [arXiv:2411.18993] als billigster Gegner.

## Wichtige Zahlen (alle ungeprüft)
- RandBET: ca. 20 % (8 Bit) / 30 % (4 Bit) Energie gespart bei < 0,8 / 2 pp Verlust, CIFAR-10.
- DeepNcode: Schutzmarge bis 7,6x (4 Bit) / 12,4x (8 Bit).
- RoR: Speicher +0,31 %, Latenz +9,1 bis 19,2 %.

## Quantenbezug
- Echt quanten: nur QEC selbst (Stabilizer, Surface Code) und die neuronalen Decoder dafür. Die klassische Gewichts-ECC ist klassische Codierungstheorie.
- Analogie: „Gewichte als Codewort“, Hadamard/Walsh als Code-Matrix. Hadamard-Codes sind klassisch (Reed-Muller 1. Ordnung).
- Hebel für QIRE: Die feste Hadamard-Mischung verteilt Information über alle Koordinaten (Streuung wie ein Code). Das ist eine Hypothese über Robustheit, bisher nicht von uns gemessen.

## Bezug zu eigenen Befunden
- Z3: Rotation ≈ gleichwertig gegen faire Quantisierer (Gruppenskalen). Die Robustheitsliteratur sagt Ähnliches: Clipping allein ist schon stark [arXiv:2006.13977], also wäre ein Rotationsvorteil gegen Bitflips gegen Clipping/Gruppenskalen zu prüfen, nicht vorauszusetzen.
- RoR [arXiv:2603.16382] stützt „Rotation hilft gegen Ausreißer-getriebene Fehler“, aber Gegner dort sind ungeschützte Modelle, nicht faire Baselines.
- I-038: unberührt.

## Lücken/offene Fragen
- Keine Arbeit gefunden, die Hadamard-Mischung mit Clipping/RandBET bei Bitflips fair vergleicht.
- ECOC-Arbeit ohne arXiv-ID; kein Survey zu Codierung für NN-Gewichte gefunden.
- Kein Volltext gelesen.

## Runde 2 (Welle 3, c3; gemeinsame Suchen mit F5, nur Abstracts, Zahlen ungeprüft)
- **ECOC bestätigt:** Yu et al., NeurIPS 2024 [url:https://neurips.cc/virtual/2024/poster/96461]: Mehrbit-ECOC gegen Gewichtsfehler, NTK-Schwelle; Codeentwurf balanciert Orthogonalität und Distanz. Lücke aus Runde 1 (ID) damit geschlossen. Walsh-Hadamard-Ausgabecode wäre ein Spezialfall dieser Linie.
- **Fehlerkorrektur in Analog/In-Memory-Hardware:** analoge ECC für Crossbars bei gemischtem Rauschen (klein plus Ausreißer) [arXiv:2603.03723]; Prüfsummen-/AN-Code-Ansätze [arXiv:2404.09818]; Lipschitz-Regularisierung statt Redundanz [arXiv:2211.14917]. Gemeinsame Linie: kleines Rauschen tolerieren, nur große Ausreißer korrigieren.
- **ABFT für Transformer:** [arXiv:2507.16676], [arXiv:2504.02211] (Overhead ca. 13,9 %, ungeprüft); Training: stille Datenkorruption [arXiv:2604.00726]. Das ist Rechenfehlerschutz, nicht Gewichtscodierung; für QIRE nur Randthema.
- **Rotation gegen Bitflips gegen faire Gegner:** erneute Suche ergibt weiter nur RoR [arXiv:2603.16382]; kein Clipping-Gegner im Suchauszug belegt (Volltext prüfen). Lücke bleibt offen.
- **Neue Lücke:** analoge ECC-Literatur trennt „kleines Rauschen" und „Ausreißer" explizit; passt zu Z3 (Rotation glättet Ausreißer, Gruppenskalen auch).

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1, a1 (F1/F5 gemeinsam, ≈ 9 Suchen) | siehe Welle 1 | hoch |
| Welle 3, c3 Runde 2 (F1, 8 gemeinsame Suchen) | 7 (F1; davon 1 ohne arXiv-ID) | ca. 7 neue auf ca. 8 Fundstellen, noch nicht gesättigt |

## Schlüsse und neue Wege (Agent)
1. **Bitflip-Robustheit als neue QIRE-Achse testen.** Beleg: [arXiv:2603.16382] (Rotation), [arXiv:2606.02430] (Integer robuster). Langweiligste Erklärung: Rotation macht nur Ausreißer kleiner, Clipping leistet dasselbe. Quantenbezug: Analogie (Streuung der Information über Basis).
2. **Fester Gegner: Clipping + Gewichtsskalierung** vor jedem Rotationsanspruch. Beleg: [arXiv:2006.13977], [arXiv:2411.18993]. Langweiligste Erklärung: Gleichstand wie in Z3. Quantenbezug: keiner.
3. **Hadamard-Zeilen als Codewörter für Ausgabe-Codierung (ECOC mit Walsh-Code).** Beleg: [arXiv:2004.10700], ECOC-Linie. Langweiligste Erklärung: bekannte ECOC mit anderem Namen. Quantenbezug: Analogie.
