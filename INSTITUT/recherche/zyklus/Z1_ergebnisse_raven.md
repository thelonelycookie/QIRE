# Z1b auf Raven (Job 31624689, Tuning-Seeds 950–959)

| Skript | Vergleich | Gewinn (Mittel ± Streuung) | plus | grün |
|---|---|---|---|---|
| Bitzuteilung | gegen max(gleich_rot, klein_fp16) | −0,23 ± 0,13 pp | 0/10 | 0/10 |
| PTQ-Länge, LM | Steigung Schaden (roh − rot), 2 Bit | **+0,031 ± 0,010 bpc je Verdopplung** | 10/10 | **10/10** |
| PTQ-Länge, MLP | Steigung Schaden (roh − rot) | +0,18 ± 0,16 pp je Verdopplung | 9/10 | 3/10 |
| PTQ-Länge, MLP | rotiertes 2-Bit-Großmodell gegen klein fp16 gleicher Bytes | +0,35 ± 0,17 pp | 10/10 | 1/10 |
| Destillation rotiert | gegen beste Kontrolle | −0,02 ± 0,06 pp | 3/10 | 0/10 |
| Seed-Codebuch | gegen härtesten Gegner | −0,14 ± 0,10 pp | 1/10 | 0/10 |

- **Grün: Rotation flacht den Quantisierungsschaden über die Trainingslänge ab.** Ohne Rotation wächst der Schaden mit längerem Training schneller (vgl. Kumar et al.). Das stützt die Kernaussage „Speicher statt Rechenzeit“ gegen den Frühregime-Einwand.
- **Tendenz, knapp unter der Schwelle:** Das rotierte 2-Bit-Großmodell schlägt den kleinen fp16-Gegner bei gleichen Bytes auf 10/10 Seeds.
- **Gestrichen:** Bitzuteilung, Destillation und Seed-Codebuch.
- **Seed-Panne:** Die Seeds 960–969 wurden abgelehnt, weil der Code nur 950–959 zuließ. Behoben auf main (aa125a8), erlaubt sind jetzt 950–989.
