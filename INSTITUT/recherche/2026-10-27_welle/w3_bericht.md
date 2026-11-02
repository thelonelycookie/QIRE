# w3 — Born-Regel-Attention (Erkundung, Tuning-Seeds 920–922, kein Befund)

## 1. Kurzfassung — Ampel: GELB (rot für "bpc-Vorteil", gelb für "Laufzeit bei langem Kontext")
- Born-Attention a_ij ∝ (q·k)² = lineare Attention mit φ(x)=x⊗x, kausal per Präfixsumme: rechnerisch verifiziert (par vs. Präfixsumme max. Abw. 2e-5, Kausalitätstest 0).
- bpc: alle 4 Arme gleich innerhalb Rauschen (Differenzen ≤ 0,01 bpc, sd 0,003–0,008). Aber: Training mit nur 800 Schritten (~0,8 M Tokens) ist weit unterfordert (bpc ≈ 3,6, das Modell nutzt Kontext kaum) — **keine Aussage über Qualität möglich**.
- FLOPs: Born spart bei hd=16 **nichts** (Attention-Zustand 256·17 pro Kopf; Break-even bei N ≈ 1000 gegen Softmax). Hadamard-Variante (m=8, F=64) ist 4× billiger als Born, Break-even N ≈ 270.
- Laufzeit (1 Thread, Vorwärts, µs/Token, Präfixsummen-Form): Softmax wächst mit N, linear nicht; bei N=4096 elu 21, hada 35, Born 59, Softmax 39 (SDPA ist stark optimiert). Born gewinnt auf dieser CPU nie gegen Softmax bis N=4096.

## 2. Idee, Bezug, Literatur
Born-Regel: Wahrscheinlichkeit = |Amplitude|² ⇒ Gewichte ∝ (q·k)², normiert. Quadrat-Kern ist exakt ein Polynomkern Grad 2 → endliche Merkmale x⊗x (Dim hd²), Zustand S∈R^{hd²×hd}, O(N·hd²·hd). Bekanntes: Lineare Attention (Katharopoulos 2020), Polynomielle Attention/„Based“, Performer/Random Features (Choromanski 2021), Hadamard-Projektion/Fastfood (Le 2013) [alle Gedächtnis]. „Quanten-inspiriert“ nur im Sinn Born-Regel, Wirkung klassisch (I-038).

## 3. Aufbau und Ergebnisse
Byte-LM UD_German-GSD (train; Test = erste 65 536 Byte), d=64, 4 Köpfe (hd=16), 2 Lagen, FFN 4d, gelernte Positionen, gebundene Ausgabe. **Alle Arme 149 248 Parameter** (Hadamard-Projektion fest, Puffer). Adam, lr 6e-3 (aus kleinem Val-Check, 300 Schritte, seed 920: 8e-3 > 3e-3 > 1e-3, Randwert; je Arm nicht getuned), 800 Schritte, 1024 Tokens/Schritt (8×128 bzw. 2×512), Kosinus, 5 % Aufwärmen. Seeds 920–922. Training in paralleler maskierter Form (mathematisch identisch).

| Kontext | Arm | Test-bpc (Mittel ± sd) | Attn-FLOPs/Token (alle Lagen) | FLOPs/Token gesamt | µs/Token (seq) | µs/Token (par) | Trainzeit s |
|---|---|---|---|---|---|---|---|
| 128 | Softmax | 3,572 ± 0,003 | 32 768 | 262 144 | 9,9 | 7,6 | 38 |
| 128 | Born | 3,579 ± 0,006 | 141 312 | 370 688 | 25,0 | 9,3 | 51 |
| 128 | elu+1 | 3,570 ± 0,007 | 8 832 | 238 208 | 10,2 | 9,9 | 36 |
| 128 | Born+Hadamard (m=8) | 3,579 ± 0,004 | 35 328 | 264 704 | 29,3 | 15,9 | 42 |
| 512 | Softmax | 3,597 ± 0,008 | 131 072 | 360 448 | 18,7 | 14,9 | 47 |
| 512 | Born | 3,600 ± 0,006 | 141 312 | 370 688 | 37,0 | 38,6 | 106 |
| 512 | elu+1 | 3,600 ± 0,007 | 8 832 | 238 208 | 13,2 | 51,2 | 78 |
| 512 | Born+Hadamard | 3,603 ± 0,007 | 35 328 | 264 704 | 17,8 | 67,2 | 77 |

Skalierung (nur Laufzeit, seq, µs/Token): N=2048: Softmax 33,6, Born 47,2, elu 10,0, hada 13,8; N=4096: 38,5 / 58,8 / 20,6 / 34,7 (Rohdaten `w3/zeit.json`; Zeiten rauschen, da 8 Agenten die Kerne teilen). FLOPs: analytisch (2/MAC; Softmax mit mittlerer Kausallänge N/2); Speicher des Zustands je Kopf/Lage: Born 256·17·4 B ≈ 17 kB, Hadamard 4,4 kB, elu 1,1 kB, Softmax-KV-Cache wächst mit N (N·2·16·4 B je Kopf: 66 kB bei N=512).

## 4. Deutung (inkl. langweiligster Erklärung)
- Langweiligste Erklärung für „alle gleich“: **Unterfordertes Training**. bpc 3,57–3,60 liegt nahe an dem, was ein Modell ohne nutzbaren Kontext erreicht; Attention-Arme können sich so gar nicht unterscheiden. Kontext 512 ist sogar leicht schlechter (weniger Batch-Elemente, gleiche Schritte). Passt zu I-044-v3/I-045-v3: Unterschiede zeigen sich erst mit mehr Training.
- Born (hd²-Merkmal) ist bei hd=16 ein schlechter Tausch: mehr FLOPs als Softmax bis N≈1000. Es lohnt erst bei großem N und kleinem hd; der Vorteil ist **Speicher/Länge**, nicht bpc.
- Hadamard-Reduktion (m=8) macht Born billig, ist aber eine feste Teilprojektion von q,k (verliert Information); bpc nicht unterscheidbar (Training zu kurz).
- Elu+1 ist am billigsten und gleich gut → Born muss sich **gegen elu+1 und Softmax** durch bpc behaupten, nicht nur gegen Softmax.
- Zeitmessungen (1 Thread, geteilte Kerne, Python-Schleife über Chunks) sind grob; Softmax nutzt fusioniertes SDPA, die Präfixsumme nicht.

## 5. Vorschlag für vorregistrierten Lauf (nur falls Malte will; sonst Erkundung beenden)
- Hypothesen: H1 Born ≤ Softmax + 0,02 bpc bei N=512/1024 (Nichtunterlegenheit); H2 Born < elu+1 (bpc, gleiche Parameter); H3 Born+Hadamard(m=8) ≤ Born + 0,02 bei 4× weniger Attention-FLOPs; H4 Laufzeit/Token Born(seq) < Softmax bei N=4096.
- Arme: Softmax, Born, elu+1, Born+Hadamard (m=8; Nebenarm m=4); Kontrolle: gleiche Parameter, Lr je Arm auf Tuning-Seed tunen, **≥ 10–20 k Schritte** (bpc < 2,5 als Eintrittsbedingung, sonst Test sinnlos); N ∈ {128, 512, 2048}; Negativkontrolle permutierte Bytes.
- 5 Seeds, Effekt 0,02 bpc. Laufzeit geschätzt: Container 4 Kerne ≈ 1–2 h pro (Arm, N=512) über 5 Seeds bei 10 k Schritte (hier: ~0,1 s/Schritt) → insgesamt ≈ 15–25 h, also **Raven/Maltes PC**; N=2048 nur mit Präfixsumme-Training (Speicher). Rote Linie: Wenn H2 bei Eintritt fehlt (elu+1 gleich gut), Born nicht weiterverfolgen.

## 6. Skripte
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w3/born_attn.py` (Modi `test`, `zeit`, `lauf`)
- `.../w3/ergebnisse.jsonl` (24 Läufe), `.../w3/zeit.json`, `.../w3/lauf.log`
- CPU-Verbrauch ≈ 28 min (1 Thread). Keine Einträge in agenda/Laborbuch, kein git.
