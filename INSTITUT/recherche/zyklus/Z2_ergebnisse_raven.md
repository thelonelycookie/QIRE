# Z2c und Z1_z2lr auf Raven (Tuning-Seeds, Erkundung, kein Befund)

## Z2_ptq_varianten (Job 31626444, n = 20)
Ampel: Steigung des Schadens (roh − Hadamard) je Verdopplung von T, 2 Bit mit 4 Stufen, Schwelle 0,005 bpc.

| Ampel | Gewinn | plus | grün |
|---|---|---|---|
| Gewichte | +0,013 ± 0,027 bpc | 16/20 | 10/20 |
| KV | +0,003 ± 0,009 bpc | 10/20 | 4/20 |
| **FFN-Mitte GELU** | **+0,409 ± 0,186 bpc** | 20/20 | **20/20** |
| **FFN-Mitte z²** | **+0,210 ± 0,055 bpc** | 20/20 | **20/20** |

- **Rotation senkt das Niveau, nicht den Exponenten:** Im Fit a·T^b gilt meist b > 1, die Rotation verkleinert vor allem a.
- **Verdacht auf Artefakt:** Ohne Rotation ist der FFN-Schaden bei 4 Stufen (≈1,5 bpc) viel größer als bei 3 Stufen (≈0,2–0,3). Vermutlich fehlt dem 4-Stufen-Gitter die Nullstufe. Vor jeder Deutung braucht es eine Kontrolle mit einem asymmetrischen Gitter mit Null bzw. einer Skala je Kanal.

## Z2_rauschformung (n = 20)
- **Rotiertes 2-Bit-Großmodell gegen klein fp16 gleicher Bytes (MLP):** +0,538 ± 0,176 pp, plus 20/20, grün 10/20.
- **GPTQ-Rückführung gegen Runden nach Rotation:** −0,010 pp, kein Nutzen. Der Gewinn kommt also von der Rotation, nicht von der Rauschformung.

## Z2_kv_treppe
- Kein Effekt: grün 2/20, Gewinn +0,006 bpc bei Schwelle 0,01.

## Z1_z2lr, Seed 950 (Lernrate je Arm, d und T gesucht)
Differenz arm − GELU in Test-bpc, positiv heißt schlechter als GELU.

| (d, T) | z² | z²+LN | ReLU² |
|---|---|---|---|
| 256, 4000 | +0,029 | +0,026 | +0,020 |
| 256, 16000 | +0,026 | +0,024 | −0,005 |
| 512, 4000 | −0,012 | −0,016 | −0,025 |
| 512, 16000 | **+0,056** | +0,038 | +0,009 |

- **Kippen nicht durch die Lernrate erklärt:** z² bleibt bei langem Training schlechter als GELU.
- **Randtreffer:** In der Zelle (512, 4000) wählten alle Arme die kleinste Lernrate des Gitters. Diese Zelle ist deshalb unsicher.
- **Seeds 951–954:** warten auf das CPU-Kontingent.
