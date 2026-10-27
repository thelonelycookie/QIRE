# Synthese Ideenwelle 27.10.2026 (Erkundung, keine Befunde)

12 Berichte (w1–w12), Tuning-Seeds 900–964, je ≤ 30 min CPU. Wirkung überall klassisch (I-038).

## Rangliste
| Rang | Idee | Bericht | Ampel | Kernzahl (Erkundung) | Was fehlt |
|---|---|---|---|---|---|
| 1 | Hadamard-Rotation vor Niedrigbit-Quantisierung | w4, w8, w11 | grün | MNIST-MLP 2 Bit + Clip: 94,9 → 97,3 % (fp 97,8 %); Speicher 1,57 → 0,22 MB | Vergleich bei **gleichen Bytes** gegen kleineres dichtes Modell/Niedrigrang; LM länger trainiert |
| 2 | Hourglass/RG-Multiskala (Token-Pooling ×4) | w10 | grün, ungetestet | einzige Idee mit echten FLOP-Einsparungen (Literatur) | Rauchtest |
| 3 | I-045 nachprüfen: Datenregime + FLOP-Zählung + Lernrate | w9, w12 | Pflicht (Ehrlichkeit) | T=16000 = 10,15 Epochen; ideal gezählt hat Sz 9–12× weniger Ops (fairer Gegner h≈45) | Train-bpc, Lernratensuche je T; beide Zählungen in die Arbeit |
| 4 | Phasen-Gewichte + Born-Auslese | w7 | gelb | MNIST 5k Param.: reell 93,4 → Nur-Phase 95,6 % (2× FLOPs); gleiche FLOPs gleichauf | Kontrolle reell + z² (vermutlich reiner Quadrat-Effekt) |
| 5 | Rangbudget nach Verschränkungsentropie | w5 | gelb | 10 %: 95,1 vs. 94,5 % (gleichmäßig) | exakt gleiche Parameter, schmal-dichte Kontrolle |
| 6 | Gelernte Monarch-FFN | w2, w8 | gelb | schlägt Niedrigrang (−0,046 bpc), gleichauf mit schmal dicht | längeres Training |
| 7 | Diagonale komplexe Rekurrenz (LRU/Lindblad) | w10, w11 | gelb, ungetestet | O(d) je Schritt | Rauchtest |
| – | Tensor-Train-FFN | w1 | rot | spart nur Parameter, mehr FLOPs | – |
| – | Born-Attention (q·k)² | w3 | rot/gelb (zu kurz trainiert) | kein bpc-Vorteil, auf CPU nie schneller als Softmax bis N=4096 | – |
| – | Frühes Aussteigen | w6 | rot/gelb | +0,4 ± 0,2 pp MNIST, Fashion nichts | – |

## Leitlinie für die Arbeit
Die bestehbare Aussage ist eine **Speicher-/Bit-Aussage**: Quanten-inspirierte Basiswechsel (Hadamard) und Phasen/Born-Quadrat sparen Bytes. FLOPs spart nur Multiskala (Rang 2). Vergleiche immer auf beiden Achsen (FLOPs, Bytes).

## Nächste Schritte
Ohne Freigabe (Rauchtests auf Tuning-Seeds, Analyse):
- R1: w4 bei gleichen Bytes (d=256 mit 2/3 Bit + Rotation gegen fp16 d=128 und Niedrigrang).
- R2: Hourglass und LRU im Byte-LM.
- R3: w7-Kontrolle „reell + z²“.

Mit Maltes Freigabe (vorregistriert, Raven):
- **L1 Kompressionsstudie bei gleichen Bytes** (Rang 1, 4, 5 zusammen): Arme dicht klein fp16, Niedrigrang, Entropie-Rang, Hadamard + 2/3 Bit, Zufalls-Orthogonale + 2/3 Bit (Kontrolle), Nur-Phase; 10 Seeds, ≥ 8000 Schritte.
- **L2 I-045-Nachprüfung**: Datenmenge × Schritte, Lernrate je T, Sz gegen dicht h≈45 (ideal gleiche Ops).
- **L3 Hourglass/LRU**, falls R2 positiv.

## Nachtrag: Rauchtests R1–R3 (27.10., Tuning-Seeds 970–984)
- **R1 (gleiche Bytes, MNIST):** großes Modell + Rotation + 2 Bit 98,30 ± 0,11 gegen klein fp16 97,90 ± 0,11 (+0,4 pp, 3/3 Seeds). Zufalls-Orthogonale = Hadamard (98,21): es zählt die Rotation, Hadamard ist nur billiger. Reine Bytes-Aussage (2,5–6× mehr MACs). → Rang 1 bestätigt (gelb-grün).
- **R2 (Byte-LM, 190 Schritte):** Hourglass+LRU 3,372 ± 0,008 gegen flach gleicher FLOPs 3,703 ± 0,010, bei weniger FLOPs. Hourglass allein nur −0,037. Vermutlich Frühregime-Effekt. → Rang 2 bleibt, Kandidat L3.
- **R3 (Phasen-Kontrolle):** reell z² bei gleichen FLOPs (95,83) schlägt Nur-Phase (95,44); Zufallsphasen = gelernte Phasen. → Rang 4 gestrichen; Phase nur als Speicher-Aussage.
