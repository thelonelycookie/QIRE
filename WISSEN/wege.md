# Neue Wege und Perspektiven (Querschnitt 1, 02.11.2026)

Hier steht noch **keine Ideenbewertung**. Die Wege sind Material für Phase 2 (Kreativ-Neustart). Jeder Weg nennt den Quantenbezug, den Schluss (S-xxx), aus dem er folgt, und die langweiligste Erklärung. Die Reihenfolge entspricht meiner Einschätzung der Aussicht für QIRE.

| # | Weg | aus | Quantenbezug | Langweiligste Erklärung | Billigster erster Schritt |
|---|---|---|---|---|---|
| W1 | **Betriebskosten statt Güte:** Hadamard gegen Gruppenskala bei gleicher Güte in Bytes, Wallclock und Joule auf dem Ryzen messen. Der Gewinn läge in den Kosten, nicht in der Güte. | S-001, S-004 | Unitarität (Analogie) | Gruppenskala ist auch im Betrieb billiger | RAPL-Verfügbarkeit am PC prüfen; Mikrobenchmark FWHT gegen Skalenlesen |
| W2 | **Byte-faire Messreihe im Kleinmodell als Hauptbeitrag:** volle Gegnerliste (S-002), ≥ 4 Trainingslängen, Skalen eingerechnet. | S-002, S-003 | Rahmen der Arbeit | Ergebnis „alle gleich“ (auch das ist publizierbar) | Bestehende Z3-Skripte um Bytes inkl. Skalen und T-Reihe erweitern |
| W3 | **Strukturierte FFN richtig skaliert:** TT/BTT/Monarch/Fastfood mit muP-artiger Init und LR je T. | S-005 | Tensornetze, 2-Design (direkt) | Dicht bleibt bei kleinem d optimal | I-047-Code mit angepasster Init, Rauchtest auf Tuning-Seeds |
| W4 | **Rotation im Gradientenpfad** beim Niedrigbit-Training (Wgrad), fest gegen zufällig gegen ohne. | S-007 | Unitarität | Nur Ausreißerglättung, Clipping gleich gut | Kleiner 4-Bit-Trainingslauf mit Hadamard vor dem Wgrad |
| W5 | **Bitflip-Robustheit:** Rotation gegen Clipping/RandBET/Hamming bei gleichen Bytes, Fehlerrate p variiert. | S-007 | Fehlerkorrektur, Reed-Muller (Analogie) | Clipping allein reicht | MNIST-MLP, Flips zur Testzeit |
| W6 | **Clifford-/Kerdock-Codebuch** für Paare/Blöcke rotierter Gewichte gegen Lloyd-2D und E8. In der Literatur nicht gefunden. | S-008, K4 | Stabilisatorzustände (direkt) | Raumfüllgewinn in 2D ≈ 0,17 dB, also klein | Synthetischer MSE-Test, 2 min |
| W7 | **Quadrat-Familie unter Quantisierung:** z² gegen ReLU² gegen \|z\| gegen GELU bei 4-Bit-Aktivierungen, mit Kurtosis. | S-006 | Born-Regel (Analogie) | Quadrat erzeugt Ausreißer, ReLU² gewinnt | Byte-LM, zwei Längen |
| W8 | **Seed-Gewichte mit Hadamard-Struktur** (SeedLM + Fastfood, 2-Design-begründet). | S-005, N6 | 2-Design (direkt) | Zufallsbasis + Koeffizienten = Low-Rank-ähnlich | Kompressionsrate gegen Rekonstruktionsfehler an einer trainierten Schicht |
| W9 | **Multiskala MERA/Wavelet** statt Hourglass+LRU: Haar-Pyramide gegen Hadamard bei gleichen Parametern. | F2, F3 | MERA (direkt) | Feste Mischer sind alle gleich (WHTMix) | Mischer-Tausch im Byte-LM |

**Vorschlag für die Reihenfolge in Phase 2:** W1 und W2 bilden das Rückgrat der Arbeit (ehrlich, machbar, neu). W3 bis W5 sind Chancen auf einen echten Vorteil. W6 bis W9 sind billige Vortests mit hohem Quantenbezug.
