# w4 — Hadamard-Rotation vor Niedrigbit-Quantisierung (Erkundung, Seeds 930–932)

## 1. Kurzfassung
- **Ampel: GELB** (Methode echt ressourceneffizient, aber Vorteil nur bei ≤ 3 Bit und im Sprachmodell hier winzig).
- Sprachmodell (d=128, 4 Lagen, nur 1200 Schritte, fp 3,564 ± 0,014 bpc): 8/4/3 Bit **verlustfrei** (≤ +0,003 bpc) mit und ohne Rotation. 2 Bit: roh RTN 3,684 ± 0,032, Rotation RTN 3,623 ± 0,003, roh+Clip 3,584, Rotation+Clip 3,572 ± 0,015.
- MNIST-MLP (fp 97,84 ± 0,12 %): 3 Bit roh 97,21 → rotiert 97,70 (RTN); 2 Bit roh+Clip 94,86 → rotiert+Clip 97,30; 2 Bit RTN 30,2 → 90,4.
- Speicher der Linearschichten: fp16 1,57 MB → 4 Bit 0,41 MB (−74 %) → 2 Bit 0,22 MB (−86 %); Rotation kostet +448 B (Vorzeichen), +29 184 Additionen/Token (≈ 3,7 % der MACs).
- Ausreißer-Mechanismus **nicht** nachgewiesen: Gewichte-Kurtosis ≈ 2,9 (Gauß) vor wie nach Rotation. Das Modell ist zu schwach trainiert, um Ausreißer zu haben.

## 2. Idee, Bezug, Literatur
- Gewichte W (aus, ein) werden mit orthogonalem Q = H·D/√n (Hadamard, zufällige Vorzeichen) gedreht: y = W x = (W Qᵀ)(Q x). Gedrehte Zeilen sind „gleichmäßiger“ verteilt, die Zeilen-Skala (max) wird kleiner, der Rundungsfehler sinkt. Q x kostet n·log₂n Additionen (schnelle Walsh-Hadamard-Transformation).
- Quanten-inspiriert: Hadamard-Gatter als Basiswechsel, der Amplituden gleich verteilt. Wirkung klassisch (I-038); Formulierung „quanten-inspiriert“.
- Literatur [Gedächtnis, nicht per Suche geprüft]: QuIP# (Tseng et al. 2024), QuaRot (Ashkboos et al. 2024), SpinQuant (Liu et al. 2024), QuIP (Chee et al. 2023), LLM.int8()/SmoothQuant für Aktivierungs-Ausreißer. Bekanntes Verfahren, keine Neuheit (Vorgabe 07.10.: okay).

## 3. Aufbau
- Sprache: `abteilungen/struktur_sprache.py` (UD_German-GSD, Byte-Ebene), Arm „D“ dicht, **d=128**, 4 Lagen, FFN 4d, Kontext 128, 1200 Adam-Schritte, Batch 8×128, lr 3e-3 (Kosinus), **nicht** getunt (keine Zeit; Hinweis: Validierungsanteil nur in der MNIST-Variante mitgeführt, nicht zur Auswahl benutzt). Test-bpc auf den **ersten 100 000 Test-Bytes** (fester Satz). 3 Seeds.
- Quantisierung nachträglich, alle `nn.Linear` der Lagen (qkv, proj, FFN: 786 432 Gewichte), Einbettung/Positionen/LayerNorm/Bias bleiben fp (nicht mitgezählt). Pro Zeile symmetrisch; RTN = Rundung zum Nächsten mit max-Skala, „Clip“ = Skala je Zeile per MSE-Gittersuche (13 Werte 0,4–1,0×max). Aktivierungen optional A8 (pro Token, symmetrisch; nur für 8/4 Bit geprüft). Quantisierung simuliert (Fake-Quant), Rechnung in float.
- Rotation: Q je Schicht eigenes D (Vorzeichen); Eingangsdimension 128 oder 512; MNIST 784 → auf 1024 aufgefüllt (Nullen), Q 1024×1024 als dichte Matrix simuliert (real: FWHT).
- MNIST: MLP 784-256-256-10, ReLU, 8 Epochen Adam 2e-3, 55 000 Train / 5 000 Val / 10 000 offizieller Test, 3 Seeds.
- Bytes = Bits·Gewichte/8 + 2 B Skala je Zeile + 2 B Bias je Zeile + n/8 B Vorzeichen (nur rotiert).

### Sprache, Test-bpc (Mittel ± sd, n=3; Bytes der Linearschichten; fp16 = 1 572 864 B; Zusatz-Additionen/Token durch Rotation 29 184, MACs/Token 786 432)
| Bit | roh RTN | Rot. RTN | roh Clip | Rot. Clip | Bytes |
|---|---|---|---|---|---|
| fp16 | 3,5641 ± 0,0135 | | | | 1 572 864 |
| 8 | 3,5641 ± 0,0136 | 3,5641 ± 0,0135 | – | – | 804 864 |
| 4 | 3,5648 ± 0,0139 | 3,5644 ± 0,0136 | 3,5649 | 3,5645 | 411 648 |
| 3 | 3,5671 ± 0,0121 | 3,5666 ± 0,0126 | 3,5676 | 3,5659 | 313 344 |
| 2 | 3,6835 ± 0,0324 | 3,6228 ± 0,0027 | 3,5836 ± 0,0148 | **3,5722 ± 0,0154** | 215 040 (+448) |
A8 (8 und 4 Bit) ändert nichts (≤ 0,0001 bpc).

### MNIST, Test-Genauigkeit (Mittel ± sd, n=3; Parameter 269 322; fp32 ≈ 1,08 MB)
| Bit | roh RTN | Rot. RTN | roh Clip | Rot. Clip | Bytes |
|---|---|---|---|---|---|
| fp | 0,9784 ± 0,0012 | | | | |
| 8 | 0,9782 | 0,9785 | – | – | 332 328 |
| 4 | 0,9775 ± 0,0016 | 0,9781 ± 0,0014 | 0,9777 | 0,9784 ± 0,0008 | 167 208 |
| 3 | 0,9721 ± 0,0011 | 0,9770 ± 0,0002 | 0,9714 ± 0,0038 | 0,9777 ± 0,0016 | 125 928 |
| 2 | 0,3016 ± 0,1126 | 0,9044 ± 0,0043 | 0,9486 ± 0,0077 | **0,9730 ± 0,0033** | 84 648 (+192) |
Rotation: +14 336 Additionen/Beispiel (MACs 269 k → ≈ 5 %). A8 ohne Wirkung.

## 4. Deutung (inkl. langweiligster Erklärung)
- Langweiligste Erklärung für das Sprachmodell: 1200 Schritte → Gewichte fast gaußisch (Kurtosis 2,9), Modell ist nahe an einem schwach trainierten Netz, das fast nichts „weiß“ und deshalb robust gegen Rauschen ist. 8–3 Bit sind dort gratis, die Rotation hat nichts zu glätten. **Das ist kein Beleg für das QuaRot-Prinzip bei echten LLMs** (dort entstehen Ausreißer erst bei Größe und langem Training).
- MNIST zeigt den erwarteten Effekt bei 3 und 2 Bit klar, aber ein Teil davon ist Eingangs-Struktur (784 Eingänge, viele fast konstante Pixel → Rotation mischt tote Pixel in die Zeilen; die Skala/Zeile profitiert). Clip allein holt schon viel; Rotation+Clip ist additiv besser (97,30 vs. 94,86 bei 2 Bit, 3 Seeds, Streuung 0,3–0,8 pp, Abstand deutlich, kein Test gerechnet).
- Echte Ressourcen: Speicher −74 % bei 4 Bit praktisch verlustfrei. Das gilt aber **genauso ohne Rotation** (4 Bit roh RTN ist hier schon verlustfrei). Rotation lohnt sich nur ab ≤ 3 Bit. Integer-FLOPs: MACs werden Integer-MACs; Rotation fügt 4–5 % Additionen hinzu (laufzeitreal auf CPU/GPU ohne Integer-Kernel nicht gemessen, nur gezählt).
- Fehlt: Vergleich bei gleichen Bytes mit **kleinerem dichten Modell** (z. B. d=64 fp16 gegen d=128 mit 4 Bit). Ohne ihn ist „effizienter als dicht/Niedrigrang“ nicht belegt. Kein GPTQ-artiges Fehlerausgleichen verwendet (RTN ist die schwächste Basis, ein Verfahren wie GPTQ verkleinert den Rotationsvorteil womöglich).
- Schwächen: lr nicht getunt, nur 100 kB Test, Embedding/Ausgabe fp, Fake-Quant, 3 Seeds ohne Test.

## 5. Vorschlag für einen vorregistrierten Lauf (nur bei Maltes Zustimmung)
- Frage: Bei **gleichem Speicher** (Bytes) schlägt ein d=256-Sprachmodell mit 3 oder 2 Bit + Rotation(+Clip) ein dichtes fp16-/8-Bit-Modell kleinerer Breite?
- Arme: (a) d=128 fp16, (b) d=192/256 mit 4/3/2 Bit roh+Clip, (c) dasselbe mit Rotation+Clip; zusätzlich (d) Niedrigrang gleicher Bytes. Training **ausreichend lang** (≥ 8 000 Schritte, bpc ≈ 2,3), damit Ausreißer überhaupt entstehen; Quantisierung nachträglich, optional QAT-Variante als Nebenarm.
- Hypothesen: H1 Rotation+Clip ≤ roh+Clip − 0,02 bpc bei 2 Bit (einseitig, gepaart); H2 3-Bit-d=256 (+Rot) ≤ fp16-d=128 bei gleichen Bytes (±0,02 Nichtunterlegenheit). Kontrollen: A8 ohne Rotation, Zufallsrotation ohne Hadamard-Struktur (dichtes orthogonales Q als Gegenprobe), Negativkontrolle permutierte Bytes. MNIST-Achse als beschreibender Nebenarm.
- 10 Seeds, Effektgröße 0,02 bpc; Laufzeit Container ≈ 10 Seeds × ~15 min = 2,5 h (bei 1 Thread/Schritt langsam, besser Raven/Maltes PC), MNIST < 10 min.

## 6. Skripte und Daten
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w4/w4_hadamard_quant.py` (Aufruf: `OMP_NUM_THREADS=1 python3 w4_hadamard_quant.py sprache 1200` bzw. `mnist`)
- Rohzahlen: `.../w4/w4_sprache.json`, `.../w4/w4_mnist.json`; Log `.../w4/sprache.log`
- Hinweis: deutsche Daten und MNIST wurden nach `INSTITUT/daten/` geladen (gitignored). CPU-Gesamtzeit ≈ 20 min.
