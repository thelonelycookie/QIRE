# K2: Aktivierungs- und KV-Quantisierung

Stand Welle 1 (Agent a5, 2026-11-02). Nur Abstracts/Suchauszüge, kein Volltext gelesen; alle Zahlen `geprueft=false`.

## Landkarte
- **Rotation (fest):** QuaRot [arXiv:2404.00456], TurboQuant [arXiv:2504.19874] (Zufallsrotation + Lloyd-Max), RotateKV [arXiv:2501.16383], KVLinC [arXiv:2510.05373].
- **Rotation (gelernt/affin):** SpinQuant [arXiv:2405.16406], FlatQuant [arXiv:2410.09426] (Kronecker-affin, W4A4 < 1 % Verlust bei LLaMA-3-70B, +7,5 % vs SpinQuant, laut Abstract).
- **Ohne Rotation:** KIVI [arXiv:2402.02750] (asymmetrisch, Keys pro Kanal), KVQuant [arXiv:2401.18079] (pro Kanal, pre-RoPE, nichtuniform, Dense-and-Sparse; < 0,1 PPL bei 3 Bit), QServe/QoQ [arXiv:2405.04532] (SmoothAttention, W4A8KV4), PolarQuant-Key [arXiv:2502.00527], AMXFP4 [arXiv:2411.09909] (asymmetrische Gruppenskala).
- **Kombination:** Rotated Runtime Smooth [arXiv:2409.20361], Skalierung vor Rotation [arXiv:2506.01967], GyRot [arXiv:2607.27694] (grobe Rotation + feine Gruppen).
- **Theorie/Übersicht:** „Great Inversion“ [arXiv:2608.25188]: Transformcodierung belohnt Energiekonzentration, gruppenskalierte Hardware-Quantisierung belohnt Flachheit innerhalb der Gruppe; die Vorschriften zeigen in entgegengesetzte Richtungen, für generische Spektren gibt es keine übertragbare Optimalitätsgarantie.

## Wann schlägt Rotation faire Nicht-Rotations-Verfahren, wann nicht? (zentral, Z3)
Belege aus Abstracts (kein Volltext):
- **Nicht (kleine Gruppe, MX/NV-Formate):** [arXiv:2509.23202] NVFP4-Gruppe 16 neutralisiert Ausreißer-Gegenmittel beweisbar; [arXiv:2511.04214] GPTQ ohne Rotation stark, globale Rotation unverträglich mit MXFP4 (Zweierpotenz-Skala); [arXiv:2411.09909] (Suchauszug) SpinQuant bei g32 nicht besser als ohne; [arXiv:2607.27694] globale Rotation widerspricht lokalen Gruppenskalen. Das deckt sich mit Z3 (Rotation ≈ g16/g32).
- **Ja (grobe Granularität, Pro-Zeile/Pro-Tensor, massive Ausreißer, 2-Bit-KV):** QuaRot/SpinQuant entstanden gegen Pro-Kanal/Pro-Token-Basen; RotateKV und KVLinC zeigen bei 2 Bit Gewinne, aber ihre Gegner sind meist KIVI-artig, nicht g16 + Lloyd.
- **Rettung der Rotation:** Blockrotation mit Blockgröße = Gruppe [arXiv:2511.04214, arXiv:2509.23202], gelernte Transforms (FlatQuant), Kanalskalierung davor [arXiv:2506.01967].
- **Offen:** keine Arbeit gefunden, die g16/g32 + Lloyd-Gitter (nichtuniform) gegen Rotation sauber in Aktivierungen/KV isoliert. Unser Z3b (Lloyd ohne Rotation ≤ Hadamard bei 2 Bit) ist hier vermutlich neu, aber nur auf Raven-Skala.

## Stärkste Baselines/Gegner für QIRE
1. Gruppenskala g16/g32, asymmetrisch (AMXFP4, KIVI-Stil).
2. KVQuant: pro Kanal + nichtuniformes Gitter + Ausreißer-Sparse (rotationsfrei, Lloyd-verwandt).
3. GPTQ ohne Rotation bei MXFP4.
4. FlatQuant als gelernter Gegner (Obergrenze).

## Wichtige Zahlen (alle ungeprüft)
- FlatQuant: W4A4 < 1 % Verlust LLaMA-3-70B, +7,5 % vs SpinQuant, 2,3x Prefill (Abstract).
- KVQuant: < 0,1 PPL bei 3 Bit (Abstract).
- QServe: 1,2–2,4x Durchsatz A100 (Abstract).
- MR-GPTQ: bis 3,6x schichtweise, 2,2x Ende-zu-Ende B200 (Abstract).

## Quantenbezug
- Echt quanten: keiner. Hadamard/Zufallsrotation ist klassische Zufalls-/Incoherence-Verarbeitung.
- Analogie: Rotation als „Basiswechsel, der Energie verteilt“ (Born-artige Gaußisierung, Beta-Verteilung); TurboQuant-Argument (Konzentration in hoher Dimension) ist Maßkonzentration, kein Quanteneffekt.
- Hebel für QIRE: ehrlich als quanten-inspirierte Mischung + Lloyd-Gitter formulieren; Wirkung klassisch (I-038).

## Bezug zu eigenen Befunden
- Z3 (Rotation ≈ faire Gegner, Gitterartefakt ohne Nullstufe): durch [arXiv:2509.23202], [arXiv:2511.04214], [arXiv:2607.27694] gestützt, Mechanismus (Gruppenskala fängt Ausreißer lokal) passt zu Steigung k (g16 0,71 vs rot 0,10).
- Möglicher Widerspruch: Im Großmodell-Bereich mit massiven Ausreißern (LLaMA-3 70B) schlagen gelernte Transforms Gruppen-Baselines laut FlatQuant; Raven-Skala könnte das nicht abbilden.

## Lücken/offene Fragen
- Kein Volltext gelesen; g32-Zahlen aus [arXiv:2411.09909] nicht bestätigt.
- KV-spezifisch: kein Vergleich Rotation vs g16 + Lloyd gefunden.
- INT4 vs FP4: Ergebnisse hängen am Format (Skalenformat Zweierpotenz).

## Schlüsse und neue Wege (Agent)
1. **Schluss:** Rotation gewinnt nur, wenn die Granularität grob ist oder das Format ungünstig (Zweierpotenz-Skala); bei g16/g32 ist sie neutral bis schädlich. Belege: [arXiv:2509.23202], [arXiv:2511.04214], Z3. Langweiligste Erklärung: Gruppenskala fängt dieselben Ausreißer lokal ab. Quantenbezug: keiner (klassisch).
2. **Schluss:** Der vermutlich neue Beitrag ist der saubere Vergleich Lloyd-Gitter ohne Rotation vs Hadamard (Z3b) auf Aktivierungen/KV. Belege: keine Gegenarbeit gefunden (Lücke, kein Beweis). Langweiligste Erklärung: Eigener Test ist klein, andere haben es nicht isoliert, weil es unter Format-Zwängen uninteressant ist. Quantenbezug: Analogie.
3. **Vorschlag:** Blockrotation (Block = Gruppengröße 16/32) als Kombi-Arm gegen g16-Lloyd testen; entspricht „grobe Rotation + feine Gruppen“ [arXiv:2607.27694]. Langweiligste Erklärung: Ergebnis wird ≈ g16. Quantenbezug: Analogie (lokale Basisdrehung).
4. **Schluss:** Great-Inversion-Theorie [arXiv:2608.25188] liefert Begründung, warum kein universeller Sieger existiert; als Erklärungsrahmen für die Arbeit nutzen (Volltext vorher lesen).

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1 (a5) | 10 | – (Feld hatte bereits ~25 Einträge) |
