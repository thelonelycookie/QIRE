# Related Work — Grundlage (Stand 07.10.2026)

Ergänzt `LITERATUR_UND_ALLEINSTELLUNG.md` und `../../SKALIERUNG_UND_VORTEIL.md`; wiederholt deren Tabellen nicht,
sondern ordnet sie nach Effizienz und Quantenbezug und prüft sie nach.

**Prüfstatus (ehrlich):** `WebFetch` auf arxiv.org wurde vom Netzwerk-Proxy blockiert (EGRESS_BLOCKED). Geprüft wurde
deshalb nur über **WebSearch-Zusammenfassungen** (Abstract-Niveau, aus Seiten wie proceedings.mlr.press, iclr.cc,
neurips.cc, arxiv-Snippets). **Zahlen stehen nur, wenn sie dort auftauchten.** Alles mit „(abgeleitet)“ ist eigene
Rechnung aus einer Strukturformel, nicht aus dem Paper zitiert. Keine Volltexte gelesen.
Eigene QIRE-Werte: 2nL Winkel, bei d=256 (n=8) also 16·L (L=1: 16; L=5: 80).

## 1. Tabelle

| Arbeit (Jahr, Venue, URL) | Struktur | Parameter bei d=256 | Gemessener Effizienzgewinn | Nichtlinearität/Auslese | Quantenbezug | Abgrenzung zu QIRE |
|---|---|---|---|---|---|---|
| Dao u. a., Butterfly-Faktorisierung (ICML 2019) https://proceedings.mlr.press/v97/dao19a.html | Produkt aus log₂d dünnen Butterfly-Faktoren, lernt DFT/DCT/Hadamard | log₂256=8 Faktoren à 2d=512 Einträge → 4.096 (abgeleitet) | Abstract nennt „subquadratisch“; konkrete Zahl nicht gefunden | keine (linear) | keiner | Gleiche Rechenstruktur wie QIRE, aber ungeteilte, freie Einträge und rein linear. |
| Dao u. a., Kaleidoscope (ICLR 2020), arXiv 2012.14966 (nicht abrufbar; nur im Vorbericht bestätigt) | Produkte von Butterfly-Matrizen als universelle Klasse strukturierter Matrizen | wie oben, K Produkte → K·4.096 (abgeleitet) | Zahl nicht geprüft | keine | keiner | Universalitätsaussage; QIRE ist ein Spezialfall mit geteilten Winkeln. |
| Dao u. a., Monarch (ICML 2022) https://arxiv.org/abs/2204.00595 | Zwei Blockdiagonale + Permutation | √d=16 Blöcke à 16×16, zwei Faktoren: 2·16·256=8.192 (abgeleitet) | Suchergebnis: GPT-2-Pretraining auf OpenWebText 2× schneller ohne Qualitätsverlust (Hardware dort nicht geprüft) | keine in der Schicht | keiner | Hardwarefreundlicher als Butterfly (GPU-Blockmatmul); QIRE ist es auf der gemessenen CPU erst ab d=16.384. |
| Liu u. a., BOFT (ICLR 2024) https://proceedings.iclr.cc/paper_files/paper/2024/hash/a63ce8e6867a1bf4b4ca62e5077814d9-Abstract-Conference.html | Orthogonale Butterfly-Parametrisierung für OFT, Cooley-Tukey-motiviert | 1.024 laut vorhandener Tabelle (b=2, m=8; Formel ½(b−1)·d·m, im Volltext nachprüfen) | Abstract: nur „effektiv“, keine Zahl | keine (linearer Adapter) | Analogie (FFT), kein Quantenbezug | Orthogonal + Butterfly wie QIRE, aber ungeteilte Winkel und lineares ΔW ohne Born-Auslese. |
| Qiu u. a., OFT (NeurIPS 2023) https://proceedings.neurips.cc/paper_files/paper/2023/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract.html | W ← R·W, R orthogonal (blockweise) | nicht berechnet (Blockgröße unbekannt) | „weniger Parameter, kein Inferenz-Overhead“; keine Zahl gefunden | keine | keiner | Orthogonale Gewichtsänderung als Prinzip; QIRE nutzt sie als Schicht mit Auslese. |
| Koike-Akino u. a., Quantum-PEFT (ICLR 2025) https://arxiv.org/pdf/2503.05431 | Pauli-/unitäre Schaltung, Parameter wachsen logarithmisch mit d | laut vorhandener Tabelle (2L+1)·8−2L = 22 (L=1) / 78 (L=5) (abgeleitet, Formel nicht von mir geprüft) | Abstract: „vanishingly smaller“ als LoRA-Rang 1 bei wachsender Dimension; keine Zahl gefunden | keine (additiv/unitär linear) | echt-motiviert (Quantenschaltung als Parametrisierung), klassisch gerechnet | Fast dieselbe Schaltung und Parameterzahl; Unterschied einzig Born-Auslese. |
| Chen u. a., QuanTA (NeurIPS 2024) https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7c17115db36193f6b83b71b0fe1d416-Abstract.html | Tensoren auf Achsen des umgeformten Eingangs wie Gatter; hochrangig | nicht berechenbar (hängt von Achsenzerlegung ab) | „weniger Parameter bei besserer Leistung“, keine Zahl gefunden | keine | Analogie (quanteninformiert) | Hochrangig und linear; QIRE hat feste 2-Dim-Achsen, geteilte Winkel, Auslese. |
| Xu u. a., ButterflyQuant (arXiv 2509.09679, 2025) https://arxiv.org/pdf/2509.09679 | Lernbare orthogonale Butterfly (Givens-Winkel) statt Hadamard für Quantisierung | n·log n/2 = 256·8/2 = 1.024 (aus Abstract-Formel) | Lernen mit 128 Kalibrierproben, „Minuten auf einer GPU“; Laufzeitvorteil nicht belegt gefunden | keine (Rotation vor Quantisierung) | Analogie (Givens-Winkel), kein Quantenbezug | Butterfly mit Winkeln ist also bekannt; Ziel dort Quantisierung, nicht Schicht mit Auslese. |
| Group-and-Shuffle, GSOFT (arXiv 2406.10019, 2024) https://arxiv.org/pdf/2406.10019 | Orthogonal aus nur zwei Matrixtypen (Gruppen + Shuffle) | nicht berechnet | Zusammenfassung: besser als LoRA/BOFT bei weniger Speicher und Trainingszeit; Zahlen nicht gefunden | keine | keiner | Weniger Faktoren als BOFT; keine Quantenstruktur, keine Auslese. |
| „Quantum-enhanced LLMs on Quantum Hardware via Cayley Unitary Adapters“ (arXiv 2605.05914, 2026) https://arxiv.org/pdf/2605.05914 | Cayley-Blöcke aus schiefsymmetrischen K, Adapter auf eingefrorenem Backbone (Llama-3.1-8B u. a.), Destillation | nicht berechnet | Zahlen nicht geprüft (nur Trainingsaufbau im Snippet) | keine gefunden | echt-orientiert (Ziel: Ausführung auf Quantenhardware) | Zielt auf echte Hardware; QIRE rechnet klassisch. Nur Snippet gelesen, Urteil vorläufig. |
| Landman u. a., Pyramidale orthogonale QNN (Quantum 2022); Cherrat u. a., Quantum Vision Transformers (Quantum 2024) https://quantum-journal.org/papers/q-2024-02-22-1265/ | Orthogonale Schichten als Quantenschaltungen (Pyramide, Butterfly), Transformer-Varianten | nicht berechnet | Suchergebnis: „wettbewerbsfähig bei weniger Parametern“, Laufzeitvorteil theoretisch (Hardware-Annahmen) | Schicht orthogonal; Nichtlinearität klassisch | echt (für Quantenrechner entworfen) | Orthogonale Schichten sind gesetzt; QIRE ersetzt Quantenhardware durch klassisches Butterfly. |
| Ma u. a., BitNet b1.58 (2024) https://arxiv.org/pdf/2402.17764 | Ternäre Gewichte {−1,0,1} | 65.536 Gewichte à 1,58 Bit ≈ 12,9 kB (abgeleitet) vs. 16 Bit ≈ 128 kB | Suchergebnis: ab 3B 2,71× schneller, 3,55× weniger GPU-Speicher; 70B: 8,9× Durchsatz | normale LLM-Nichtlinearitäten | keiner | Anderer Effizienzweg (Wertebreite statt Struktur); für T-ternär der Vergleichsmaßstab. |
| Qiu u. a., Compute Better Spent (ICML 2024) https://proceedings.mlr.press/v235/qiu24f.html | Strukturierte Matrizen, Block-Tensor-Train (BTT, enthält Monarch), µP-Skalierung | nicht berechnet (BTT-Form frei) | Abstract: BTT = dichtes ViT-S/32 auf ImageNet-1k mit 3,8× weniger Rechenaufwand | keine | keiner | Struktur gewinnt nur mit eigener Init/Lernrate; Pflichtregel auch für QIRE-Vergleiche. |
| Tomut u. a., CompactifAI (arXiv 2401.14109, 2024) https://arxiv.org/abs/2401.14109 | Matrix-Product-Operatoren in Attention/MLP, Bonddimension steuert Kompression | nicht berechnet | LLaMA-2 7B: −93 % Speicher, −70 % Parameter, Training 50 %, Inferenz 25 % schneller, 2–3 % Genauigkeitsverlust (mit Quantisierung kombiniert; Hardware nicht angegeben) | keine | Analogie (quanteninspirierte Tensornetze), klassisch | Komprimiert fertige Gewichte; QIRE ist eine Schicht mit Winkelparametrisierung. |
| **Neu:** Gonon u. a., Fast Inference with Kronecker-Sparse Matrices (ICML 2025) https://proceedings.mlr.press/v267/gonon25b.html | Kronecker-sparse (Butterfly/Monarch) GPU-Kernel, fusioniert | — | Median 1,4× (FP32, 600 Muster), −15 % Energie; GPU | — | keiner | Zeigt: Butterfly-Speedup hängt am Kernel; QIRE-Messung (CPU, ab 16.384) passt dazu. |
| **Neu:** Raj/Coyle, QuIC (arXiv 2502.06916) https://arxiv.org/html/2502.06916v2 | Hamming-gewichtserhaltende Quantenschaltungen als Adapter | nicht berechnet | 99,2 % von LoRA bei 44× weniger Parametern; 98 % von OFT/BOFT bei 25× weniger (Abstract) | keine gefunden | Quanteninspiriert | Weitere Parametereffizienz-Konkurrenz; nur Parameter, kein Wall-Clock. |

Suchergebnis zu „quantum-inspired butterfly / orthogonal butterfly fine-tuning 2025–2026“: außer den oben genannten
(BOFT, GSOFT, ButterflyQuant, Cayley, QuIC) fand sich **keine** Arbeit, die Butterfly **mit Born-Auslese** kombiniert.
„Nicht gefunden“ heißt nur: nicht in diesen wenigen Suchen. Zur Zusatzliteratur „structured matrices wall-clock“:
MonarchAttention (NeurIPS 2025) und Gonon 2025 (Snippets, nicht vertieft).

## 2. Was QIRE nicht als Alleinstellung beanspruchen darf

1. **Logarithmische Parameterzahl / geteilte Winkel** — Quantum-PEFT (22/78 vs. QIRE 16/80 bei d=256), Kronecker Recurrent Units (laut Vorgängerdokument).
2. **Orthogonale Butterfly mit Winkeln** — BOFT, ButterflyQuant (n log n/2 Winkel), GSOFT, EUNN.
3. **O(d log d) statt d²** — Dao 2019/2020; Rechenstruktur ist das Standardwissen der Butterfly-/Monarch-Linie.
4. **„Hardwareeffizient“ ohne Einschränkung** — Pixelated Butterfly, Monarch und Gonon 2025: Der Gewinn hängt vom Kernel. QIREs 1,6–2,3× (Inferenz) / 1,18× (Training) gelten nur für CPU (Ryzen) ab d=16.384 und liegen im Bereich anderer Kronecker-Sparse-Messungen (Gonon: Median 1,4× auf GPU). Nicht verallgemeinern.
5. **Parametereffizienz als Vorteil** — QuIC (44×/25× weniger), CompactifAI (−70 % Parameter), BitNet (Speicher). Allein kein Argument.
6. **Quanteninspiriert als Neuheit** — QuanTA, CompactifAI, Quantum-PEFT, QuIC nutzen das Etikett längst.
7. **Orthogonale Schichten aus Quantenschaltungen** — Landman/Cherrat (Quantum 2022/2024).
8. **Quanten-Geschwindigkeitsvorteil** — ausgeschlossen (Cerezo 2025, laut Vorgängerdokument).

## 3. Wo QIRE sich abgrenzt (zu belegen durch Q-020 P4 Born-Ablation und T-ternär)

- **Die Born-Auslese als einzige strukturelle Differenz zu Quantum-PEFT.** Alle oben gefundenen verwandten Butterfly-/Orthogonal-Adapter sind linear. Die Behauptung „Born-Auslese (|ψ|², ⟨Z⟩) trägt, linearisierte Auslese nicht“ ist durch **Q-020 P4 (Born-Ablation)** zu belegen: gleiche Butterfly-Struktur, gleiche Parameterzahl, Auslese quadratisch vs. linear, mit TOST und Positivkontrolle. Fällt P4 gleichwertig aus, verschwindet diese Abgrenzung, und QIRE ist „Quantum-PEFT-Schaltung, andersherum aufgeschrieben“ (Ausgang schon in SKALIERUNG_UND_VORTEIL V-Serie vorgesehen).
- **Messmethodik** (vorregistriert, TOST, lineare Referenz je Kontrolle): kein verwandtes Paper aus dieser Tabelle weist sie aus. Gilt nur, wenn Q-020 die schwache Kontrolle repariert.
- **T-ternär:** Wertebreite (BitNet-Weg) und Struktur (Butterfly-Weg) sind getrennte Effizienzachsen. Ein Arm „Butterfly-Winkel ternär/Clifford-gerundet“ gegen BitNet-artige ternäre dichte Schicht bei gleichem Speicher zeigt, ob die Kombination etwas bringt. Als Kombination nicht gefunden (siehe Teil 3 Vorgängerdokument; vorläufig).
- **Fairness gegenüber Compute Better Spent:** Jeder Strukturarm braucht eigene Init und Lernrate, sonst ist der Vergleich angreifbar.
- **Immer getrennt berichten:** Qualität bei trainierbarer Breite (256) und Effizienz bei ihrer eigenen Breite (ab 16.384).

## 4. Offen
- Zahlen zu Kaleidoscope, OFT, BOFT, QuanTA, GSOFT, Cayley nicht aus den Papers gesichert (arxiv.org blockiert).
- Parameterformeln von Quantum-PEFT und BOFT stammen aus der älteren Tabelle, nicht von mir geprüft.
- Cayley-Adapter (2605.05914): nur Trainingsaufbau-Snippet; Abgrenzung vorläufig.
