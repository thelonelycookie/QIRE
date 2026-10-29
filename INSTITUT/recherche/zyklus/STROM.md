# STROM (Späher, 29.10.2026)
Quelle: 7 WebSearch, nur Snippets/Aggregatorseiten, alle Zahlen UNGEPRÜFT. Nicht aufgenommen, da schon in Z3_ideen: HARP (2605.29843), ButterflyQuant (2509.09679); in ARCHIV: Qift (2606.02823). Keine neuen 2026-Funde zu Seed-Basis und MPO (nur PicoGPT 2603.28534, schon in Z1).

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-05 | QAM-W: Joint 2D Codebook Quantization via Hadamard Rotation and Activation-Aware Scaling | arXiv 2605.26339 | ~5,5 Bit/Gewicht, PPL innerhalb 0,4 % von BF16 (WikiText-2) | neuer Gegner/Lloyd: Hadamard + 2D-Lloyd-Max-Buch; zeigt Lloyd-Gitter nach Rotation als Standard; hohe Bitzahl, schwacher Vergleich zu 2 Bit |
| 2026 | PolarQuant (Vicentino): Walsh-Hadamard + skalares Lloyd-Max-Buch | laut QAM-W-Snippet, ID fehlt | keine | neuer Gegner für Hadamard+Lloyd-Arm; Namensverwechslung mit PolarQuant/TurboQuant hüten |
| 2025-02 (EMNLP-F 2025) | Rotate, Clip, and Partition (RCP): W2A4KV4 mit Rotation + lernbarem nicht-uniformem Quantisierer | arXiv 2502.15779 | LLaMA-2-7B W2A4KV4: PPL-Verlust 2,84, 5,29x kleiner | stützt Aktivierungsachse (Rotation trägt A4); QAT-Gegner zu Z2c-FFN-Befund; W2A2 selbst: kein Treffer gefunden |
| 2026-02 | 1-Bit Wonder: QAT durch K-Means-Quantisierung | arXiv 2602.15563 | Skalierungsgitter Größe x Token x Präzision, Nemotron-CC | neue Idee/Gegner: k-Means-Gitter statt uniform im QAT, passt zur Gitterachse (Z3) und Trainingslängen-Achse |
| 2026 (ICML 2026, PMLR 306) | Scaling Law for Quantization-Aware Training (Chen et al.) | proceedings.mlr.press/v306/chen26gl.html | W4A4, 268 Läufe: Fehler sinkt mit Modellgröße, steigt mit Tokens und gröberer Gruppe | stützt (Schaden wächst mit T, wie Z1b); neuer Gegner/Messlatte: Gesetz mit Gruppengröße; QIRE-Rotation als Steigungs-/Niveau-Term prüfen |
| 2026-02 | Scaling laws for precision in high-dimensional linear regression (Zhang et al.) | arXiv 2602.19241 | multiplikatives vs additives Quantisierungsrauschen skalieren verschieden | neue Idee: Theorierahmen für "Rotation senkt Niveau a, nicht Exponent b"; Rotation macht Rauschen eher additiv? (nur Hypothese) |
| 2024-11 / 2025-02 | Low-Bit Quantization Favors Undertrained LLMs (2411.17691); ParetoQ (2502.02631) | arxiv.org/abs/2411.17691, 2502.02631 | >1500 Checkpoints; ParetoQ: 2-Bit/ternär liegt auf Pareto-Front | stützt Befund "PTQ-Schaden wächst mit Trainingslänge"; falls nicht in ARCHIV zitiert: nachtragen (nicht 2026) |
| 2026 (ICLR) | Accelerating Inference for Multilayer NNs with Quantum Computers (Rattew) | iclr.cc 2026 poster 10009571 | quadratischer/quartischer Speedup nur mit Quantenhardware | irrelevant für klassische Wirkung (I-038), nur Abgrenzung |

## Lücken
- W2A2 mit Rotation: keine 2026-Arbeit gefunden (nur W2A4). Unser A-Bit-Befund (FFN-Mitte 4 Stufen) liegt im selben Regime wie Qift/RCP.
- Seed-Basis, MPO, quanten-inspiriert-effizient: 2026 nichts Neues im Snippet.
