# N5: Energie/Hardware (Bit-Op-Kosten, CPU/Edge)

Stand Welle 3 (c1), 2026-11-02. 10 neue Quellen, keine im Volltext gelesen; Zahlen aus Abstracts/Snippets, daher alle `geprueft=false`.

## Landkarte
- **Op-Energie nach Bitbreite:** Horowitz ISSCC 2014 (45 nm) [bib:horowitz_2014_isscc], Nachdrucke [arXiv:1602.04183], [arXiv:2603.26053]. Kernaussage: Ganzzahl-Op ≪ FP-Op ≪ Cache ≪ DRAM; DRAM 64-Bit ca. 1300-2600 pJ gegen 0,4-3,7 pJ für FP-Ops.
- **Speicher vs. Rechnen:** [arXiv:1703.09039] (Sze), [arXiv:2403.14123] (FLOPS 3,0x/2J, DRAM 1,6x, Interconnect 1,4x). Decoding ist bandbreitengebunden.
- **LUT-/Niedrigbit-Kerne auf CPU:** T-MAC [arXiv:2407.00088], bitnet.cpp [arXiv:2410.16144], T-SAR [arXiv:2511.13676], Hardware-LUT: FIGLUT [arXiv:2503.06862], LUT Tensor Core [arXiv:2408.06003]. Edge-Messung: LLMPi [arXiv:2504.02118].
- **Op-Zählmodelle (nur modelliert):** L-Mul [arXiv:2410.00907] (-80..95 %), BlockDialect (4 vs 8 Bit ca. 2,45x), Jack Unit.
- **Messmethodik:** ML.ENERGY [arXiv:2505.06371], Position Joule/Token [arXiv:2605.11733], GreenBench [arXiv:2608.28667] (Paketleistung/Latenz).
- **Rotation im Betrieb:** HadaCore [arXiv:2412.08832]; Dao-Kernel ca. 1,0x memcpy bis 512 Dim, 1,3x bei 16384 (Sekundärangabe, A100). GyRot [arXiv:2607.27694]: Vorteil nur mit Co-Design.

## Stärkste Baselines/Gegner für QIRE
- **bitnet.cpp / T-MAC auf Maltes Ryzen:** wenn QIRE-Gewinne nicht gegen diese LUT-Kerne bestehen, ist "Energiegewinn" nicht belegt. Ihre Selbstangaben (-55..-82 % Energie ggü. FP16-llama.cpp) sind der Maßstab, aber Basis ist FP16, nicht ein gut optimiertes INT4.
- **Einfaches INT4 mit Gruppenskalen** (K1) als faire Baseline: GyRot und Z3 zeigen Rotation ≈ Gruppenskala.
- Hardware-seitig: FIGLUT/LUT Tensor Core (Co-Design, nicht auf CPU übertragbar).

## Wichtige Zahlen (alle ungeprüft)
| Zahl | Quelle | Status |
|---|---|---|
| Int8 add ≈ 0,03 pJ, mult ≈ 0,2 pJ; FP16 mult ≈ 1,1; FP32 mult ≈ 3,7; Int32 mult ≈ 3,1 (45 nm) | Horowitz, **aus Gedächtnis** | ungeprüft, vor Zitat im Original lesen |
| Int16 add 0,18 / mult 0,62; FP64 5 / 20 pJ | [arXiv:1602.04183] Snippet | ungeprüft |
| DRAM 64-Bit 1300-2600 pJ, Cache 10-100 pJ | [arXiv:2603.26053] Snippet | ungeprüft |
| 28-nm-Beschleuniger: 1 TOPS/W ≈ 1 pJ/Op; INT4 27,4 vs INT8 22,9 TOPS/W | bib-Eintrag Welle 1 | nur Größenordnung, Bedingungen verschieden |
| Edge 2 Bit: 5,18 Token/J (4,3x ggü. FP16), BitNet 1,58: 4,66 Token/J | [arXiv:2504.02118] | ungeprüft |
| Qwen2.5-0,5B: 2,61 J/Token | [arXiv:2504.03360] | ungeprüft |
| BitNet 2B4T: 0,028 J (geschätzt), CPU-Decoding 29 ms vs 41-124 ms | [arXiv:2504.12285] via Zweitquelle | ungeprüft |

## Kostenmodell für QIRE (Vorschlag, eigene Herleitung)
Energie je Token:  E ≈ B_DRAM·e_B + Σ_b N_b·e_op(b) + N_LUT·e_LUT + N_had·e_add + E_statisch·t
- B_DRAM: Bytes, die je Token aus DRAM gelesen werden. Beim Decoding ≈ Modellgröße in Bytes (Gewichte + Skalen + KV-Cache). **Dominanter Term**; Näherung e_B ≈ 1300-2600 pJ / 8 B ≈ 160-325 pJ/Byte (45 nm, Größenordnung; heutige DDR4/5 eher niedriger, messen).
- N_b: MACs je Bitbreite, e_op(b) aus Horowitz-Tabelle. Bei Decoding klein gegen B_DRAM·e_B (Faktor ≥ 100 auf 45-nm-Zahlen).
- Hadamard online: n·log2(n) Additionen je Vektor der Länge n, gegen d_out·n MACs der Schicht: Verhältnis log2(n)/d_out (n=d_out=4096: 12/4096 ≈ 0,3 %). Rechnerisch vernachlässigbar, solange fusioniert und im Cache; real speichergebunden (Zusatzpass über Aktivierungen), nicht-fusioniert bis ca. 1,3x memcpy je Vektor. Wirkung nur bei kleinen Schichten/Batch 1 messbar.
- E_statisch·t: Leerlauf-/Paketleistung × Wallclock; auf CPU oft größer als dynamische Op-Energie. Deshalb folgt Energie näherungsweise Wallclock bei Decoding.
**Umrechnung für QIRE:** Gewinn in Energie ≈ ΔBytes·e_B (Byte-Budget) – ΔZusatzbytes (Skalen, Codebuch, Seed-Metadaten) – Hadamard-Zusatz. Bei gleichem Byte-Budget (unser Setting) ist der Energiegewinn von QIRE ≈ 0, solange Genauigkeit gleich ist; Gewinn entsteht nur über weniger Bytes bei gleicher Güte oder über LUT-Kerne.

## Faire Messmethodik
- Drei Ebenen getrennt berichten: FLOPs/Op-Zahl (Modell), Bytes (Speicher-Verkehr), Wallclock+Joule (Messung). L-Mul/BlockDialect zeigen: Op-Modell ohne Hardware sagt nichts über CPU.
- CPU: Paketenergie (RAPL; auf Ryzen 3600 evtl. nur über Software-Schätzung, prüfen) minus Leerlauf, mehrere Wiederholungen, Thermik/Boost fixieren, Prefill und Decode getrennt, Energie je Aufgabe statt nur je Token [arXiv:2505.06371].
- Gleiche Kerne für beide Seiten (gleiche Optimierungsstufe), sonst misst man Kerngüte statt Format.

## Quantenbezug
- Echt quanten: keiner in diesem Feld. Nur QRAM-/Fehlerkorrektur-Randeinträge (Welle 1), die zeigen, dass Quantenvorteile durch Steuerhardware schrumpfen.
- Analogie: Hadamard/Rotation ist quanteninspiriert, auf klassischer Hardware aber nur zwei Additionsstufen; der Hebel ist kein Energie-, sondern ein Genauigkeitshebel bei fester Bytezahl (passt zu I-038: klassische Wirkung).
- Inspirierter Hebel: Energie über Bytes (weniger Bits je Gewicht bei gleicher Güte), nicht über Op-Energie.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3: Rotation ≈ Gruppenskala → stützt [arXiv:2607.27694]; Hadamard-Betriebskosten kämen noch obendrauf, wären also ein reiner Nachteil, wenn die Güte gleich bleibt.
- I-038 (klassische Wirkung): konsistent; kein Energiebonus durch "Quanten".
- Kein Widerspruch gefunden.

## Lücken/offene Fragen
- Horowitz-Originaltabelle und aktuelle (7-5 nm, LPDDR/DDR5) Werte nicht verifiziert.
- Kein End-to-end-Prozentwert für Online-Hadamard (QuaRot/SpinQuant) gefunden.
- Ryzen-3600-Energiemessung (RAPL-Zugriff unter Linux/Windows) ungeklärt.
- T-MAC/bitnet.cpp-Basislinien sind FP16; Vergleich gegen INT4-Gruppenskala fehlt.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 3 c1 | 10 | hoch (Feld bisher nur Snippets), 10 von ca. 31 |

## Schlüsse und neue Wege (Agent)
1. **Energie folgt Bytes, nicht Ops.** Belege: [arXiv:2403.14123], Horowitz-Verhältnis DRAM:Op ≈ 10²-10³, [arXiv:2503.06862]. Langweiligste Erklärung: Decoding ist bandbreitengebunden, also Energie ∝ Modellgröße. Quantenbezug: keiner; QIRE-Nutzen = Bytes sparen.
2. **Hadamard-Betrieb kostet ≈ 0,3 % der Ops, aber Speicherpässe.** Belege: eigene Rechnung, [arXiv:2412.08832]. Langweiligste Erklärung: bei fusionierter Implementierung nicht messbar. Daher Wallclock-Test auf dem Ryzen statt Op-Zählung. Quantenbezug: Analogie.
3. **Fairer Energievergleich nur gegen LUT-Kerne (T-MAC/bitnet.cpp) mit Joule+Wallclock+Bytes.** Belege: [arXiv:2407.00088], [arXiv:2410.16144], [arXiv:2505.06371]. Langweiligste Erklärung: Gewinne ggü. FP16 sind kleine Bitbreite, nicht das Verfahren. Quantenbezug: keiner.
4. **Op-Zählmodelle (L-Mul, FIGLUT-Spalte FL) nur als Sekundärspalte, nie als Hauptergebnis.** Belege: [arXiv:2410.00907]. Langweiligste Erklärung: Modell ohne passende Hardware. Quantenbezug: keiner.
