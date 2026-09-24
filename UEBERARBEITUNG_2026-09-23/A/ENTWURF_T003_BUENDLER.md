# Entwurf und Vorregistrierung T-003 — der Rechen-Bündler

Angelegt **24. September 2026, vor jeder Zeitmessung.** Nichts hier ist
gemessen. Jede Zahl ist entweder aus dem Tagebuch übernommen (mit Quelle), eine
Formel, eine Verifikationszahl (mit Log-Datei) oder eine **Erwartung** (als
solche markiert). Rohdaten laufen unter der Kennung **`T-003-vorlauf`**; ob der
Lauf als T-003 ins Tagebuch geht, entscheidet Malte.

Code: `quantum_sim/buendler.py` (Strategien, Verifikation, Aufwandsformeln),
`server_setup/gpu_benchmark_buendler.py` (Messung CPU und GPU, Auswertung,
Vortest).

---

## 1. Anlass

T-001 (Tagebuch 10. August): `butterfly_fast` ist 1,6× schneller als die
Referenz, der Übergangspunkt gegen dicht bleibt bei **16.384**. Die
Tagebuchdiagnose: n·L kleine Stufen, jede ein voller Durchlauf über den
Zustand. Der Entwurf der T-Serie hielt fest, dass die eigentliche Fusion einen
kompilierten Kernel braucht — `numba`/`triton` fehlen.

**Maltes Idee:** nicht fusionieren, sondern **bündeln** — die vielen kleinen
Schritte in wenige große Matrixprodukte (GEMM) umschreiben. GEMM ist die eine
Operation, die jede CPU- und GPU-Bibliothek nahe an der Spitzenleistung
ausführt. Das braucht keinen Compiler, nur Mathematik.

## 2. Die Umschreibung (verhaltenserhaltend, nachgewiesen)

- **Eine Lage ohne CNOT** ist das Kronecker-Produkt G₀⊗…⊗G_{n−1}. Qubits in
  Gruppen A (k) und B (n−k) geteilt, Zustand als Matrix X (2ᵏ × 2ⁿ⁻ᵏ):
  **K·x = U_A · X · U_Bᵀ** — zwei GEMMs statt n Paardrehungen (`kron2`). Mit
  drei Gruppen drei GEMMs (`kron3`). m Gruppen = „Radix 2^(n/m)“; m = n ist
  der Butterfly, m = 1 die volle Matrix.
- **Ohne CNOT** kollabieren alle L Lagen auf eine (Produkt der Gatter je
  Qubit): immer 2 GEMMs, unabhängig von der Tiefe (`kron2_ohne`).
- **Mit CNOT-Ring:** je Lage 2 GEMMs + 1 Gather (R1). Die Permutation lässt
  sich **in die Faktoren falten** (`kron2_gefaltet`): Der Ring zerfällt an der
  Trennstelle in CNOTs innerhalb A (in U_A multipliziert), innerhalb B (in
  U_B), einen CNOT A→B (die B-Seite bekommt zwei Matrizen, gewählt durch ein
  Bit der Zeile) und einen CNOT B→A (die A-Seite der *nächsten* Lage bekommt
  zwei Matrizen, gewählt durch ein Bit der Spalte; nach der letzten Lage wandert
  er in die Vorzeichenmatrix der Auslese). **Kein Gather mehr** — dafür
  gruppierte GEMMs. Monarch-artig (Dao et al. 2022): blockdiagonal ×
  Permutation × blockdiagonal, hier mit geteilten Blöcken.
- **`vorkompiliert`:** aus den Winkeln einmal die volle Matrix U bauen, dann
  ein reelles GEMM (x reell ⇒ Ux = Re(U)x + i·Im(U)x). 2·d² Arithmetik,
  8·d² Byte Speicher. Nur Inferenz mit eingefrorenen Winkeln.
- **`reell_kron2` ist ein eigener Arm**, keine Umschreibung: nur Ry, alles
  float32. Andere Funktion (n·L statt 2·n·L Parameter). R4 ist damit
  beantwortet: Rz ist **nicht** streichbar — mit gegen ohne Rz weicht die
  Ausgabe um bis zu **0,703** ab (n=6, L=3, `rz_streichbar()`).

### Verifikation — bestanden, bevor eine Zeit gemessen wurde

`python -m quantum_sim.buendler`, Log:
`results/experiments/T-003-vorlauf__verifikation_20260924.txt`.

| Strategie | max. \|B − fast\| | max. \|B − float64\| | max. \|fast − float64\| | max. \|Layer − float64\| |
|---|---|---|---|---|
| kron2 | 5,4·10⁻⁷ | 8,5·10⁻⁷ | 9,0·10⁻⁷ | 1,4·10⁻⁶ |
| kron3 | 5,4·10⁻⁷ | 8,6·10⁻⁷ | 9,0·10⁻⁷ | 1,4·10⁻⁶ |
| kron2_gefaltet | 1,07·10⁻⁶ | 8,2·10⁻⁷ | 8,5·10⁻⁷ | 1,4·10⁻⁶ |
| vorkompiliert (n ≤ 10) | 4,2·10⁻⁷ | 3,6·10⁻⁷ | 4,0·10⁻⁷ | 4,9·10⁻⁷ |
| reell_kron2 (gegen Rz=0) | 6,0·10⁻⁷ | 9,6·10⁻⁷ | 8,8·10⁻⁷ | 1,4·10⁻⁶ |

455 Konfigurationen: n = 4…14, L = 1…5, mit und ohne CNOT. Gradienten nach
Winkeln **und** Eingabe gegen `ButterflyFast`: größte Abweichung 1,4·10⁻⁶
(Schranke 10⁻⁴; 102 Konfigurationen, n = 4, 6, 8, 11).

**Offen gelegte Kriterienänderung:** Der erste Durchlauf prüfte „≤ 10⁻⁶
gegen `ButterflyFast` UND gegen `ButterflyQuantumLayer`“ und fiel bei n = 13
und 14 in 39 Konfigurationen durch — **ausschließlich** gegen
`ButterflyQuantumLayer` (bis 1,6·10⁻⁶), das selbst bis 1,4·10⁻⁶ von der
float64-Rechnung abweicht, also die ungenaueste der drei Referenzen ist. Die
Schranke 10⁻⁶ stammt aus T-001, wo nur bis n = 10 geprüft wurde. Schiedsrichter
ist seitdem eine unabhängige complex128-Rechnung (`_referenz_float64`), Schranke
unverändert 10⁻⁶. Die Bündler-Strategien liegen dort **nicht weiter** von
float64 entfernt als `ButterflyFast`. Die Marge ist dünn (reell 9,6·10⁻⁷):
Ab n ≥ 15 ist 10⁻⁶ absolut für float32 keine sinnvolle Schranke mehr.

## 3. Aufwand nach Formel — und warum das nichts entscheidet

`aufwand_tabelle()`; reelle Multiply-Adds relativ zu dicht (komplex = 4 reell),
Batch 64, L = 5, mit CNOT; in Klammern Durchläufe über den Zustand:

| d | referenz | kron2 | kron3 | vorkompiliert | reell_kron2 |
|---|---|---|---|---|---|
| 1.024 | 0,39 (57) | 1,25 (17) | 0,63 (22) | 2,0 (1) | 0,31 (17) |
| 4.096 | 0,12 (67) | 0,63 (17) | 0,23 (22) | 2,0 (1) | 0,16 (17) |
| 16.384 | 0,034 (77) | 0,31 (17) | 0,098 (22) | 2,0 (1) | 0,078 (17) |
| 65.536 | 0,010 (87) | 0,16 (17) | 0,039 (22) | 2,0 (1) | 0,039 (17) |

Ohne CNOT (`kron2_ohne`): 0,25 bei 1.024, 0,125 bei 4.096, 4 Durchläufe.

Der Bündler rechnet **mehr** als der Butterfly (kron2 ≈ 2^(n/2)/n-mal mehr),
aber in 17 statt 67 Durchläufen und in GEMMs statt elementweisen Stufen. Ob
sich das lohnt, sagt die Formel nicht — T-001 hatte 340× weniger Arithmetik
und war 5× langsamer. **Deshalb wird gemessen.**

## 4. Vorregistrierung

### Hypothesen

> **H1 (PRIMÄR, CPU, einfädig, Batch 64, Vorwärts, L = 5, mit CNOT — wie
> T-001):** Der Übergangspunkt von **`kron2`** gegen die dichte Schicht — die
> kleinste Breite, bei der der Median der gepaarten Reihenfaktoren dicht/kron2
> über 1 liegt — sinkt von 16.384 (T-001) auf **≤ 4.096**.
>
> - **T1:** Übergang ≤ 4.096 → die Lücke war Ausführung, nicht Struktur.
> - **T0:** Übergang > 8.192 oder nirgends bis 16.384 → Bündelung genügt nicht.
> - **TU:** Übergang = 8.192 → Verbesserung berichten, Ziel verfehlt.
>
> **Auflösungsprüfung:** Bei d = 16.384 muss kron2 mindestens so schnell sein
> wie `referenz` (= `ButterflyFast`), Median der gepaarten Reihenfaktoren ≥ 1.
> Sonst misst der Aufbau nicht Bündelung, und **kein** Ergebnis des Laufs ist
> interpretierbar (Ausgang TU mit diesem Text).
>
> **Maßstabsprüfung (berichtet, nicht urteilsbestimmend):** referenz gegen
> dicht bei 4.096 soll in der Nähe von T-001 liegen (0,331). Weicht er um mehr
> als Faktor 1,5 ab, wird das neben dem Urteil ausgewiesen.

Genau ein Arm (kron2) trägt die Primärhypothese; kron3 und gefaltet sind
sekundär. Wer den besten Arm nachträglich auswählt, testet drei Hypothesen und
meldet eine.

> **H2 (sekundär):** `kron2_ohne` (ohne CNOT, L Lagen gefaltet zu einer)
> schlägt dicht ab **d ≤ 1.024** (gleicher Lauf wie H1).
>
> **H3 (sekundär, Lauf T64):** Vorwärts + Rückwärts, Batch 64, Winkel
> trainierbar: Übergang kron2 gegen dicht **≤ 8.192**.
>
> **Beschreibend, ohne Urteil:** kron3 und kron2_gefaltet gegen kron2;
> vorkompiliert gegen dicht; reell_kron2 gegen kron2; alle Faktoren bei
> Batch 256; Skalierung der Bündler-Arme bei 2¹⁵ und 2¹⁶.

Keine Signifikanzaussage: Wie in T-001 wird ein Übergangspunkt aus Medianen
von sieben gepaarten Reihenfaktoren bestimmt, dazu die Zahl der Reihen, in
denen der Arm gewinnt. Die Familie für eine spätere Holm-Korrektur, falls doch
getestet wird: {H1, H2, H3}.

### Erwartungen, vorher notiert (nicht die Hypothesen)

- **H1:** Nach Formel braucht kron2 bei 4.096 0,63 der Arithmetik von dicht,
  in GEMMs mit innerer Dimension 64. MKL ist dort weniger effizient als bei der
  großen dichten GEMM, dazu 5 Gathers. **Erwartet: Übergang bei 4.096 oder
  8.192, eher 8.192 → TU.** T1 ist möglich, aber nicht der erwartete Ausgang.
- **Auflösung:** kron2 gegen referenz bei 16.384 **erwartet 3–10×**.
- **H2:** 0,25 der Arithmetik bei 1.024, 4 Durchläufe → **erwartet bestätigt.**
- **H3:** Rückwärts verdoppelt beide Seiten etwa → **erwartet ≤ 8.192.**
- **kron3** hat bei 4.096 nur 0,23 der Arithmetik, aber GEMMs mit innerer
  Dimension 16 — erwartet: schneller als kron2 ab 8.192, bei kleinem d nicht.
- **kron2_gefaltet** spart den Gather, bezahlt ihn aber mit zwei Stack-Kopien
  je Lage (PyTorch kann nicht in strided Ausgaben schreiben). **Erwartet: nicht
  schneller als kron2 auf CPU.** Sein Wert liegt im Kernel (Abschnitt 6).
- **vorkompiliert:** 2 × die Arithmetik von dicht → **erwartet ≈ 0,5× dicht.**
  Sein Punkt ist die Parameterzahl (2nL) bei GEMM-Tempo, nicht Geschwindigkeit.
- **reell_kron2:** ¼ der Arithmetik von kron2 → erwartet 2–4× schneller.

### Methodik — T-001 unverändert

Sieben Reihen; innerhalb jeder Reihe alle Arme unmittelbar nacheinander in
fester Reihenfolge; je Messung 2 Aufwärmläufe, Median aus 3; Faktor **je Reihe
gepaart**, nicht Verhältnis zweier Mediane. Einfädig (`torch.set_num_threads(1)`).
Batch 64 und 256. Vorwärts = Inferenz mit **eingefrorenen Winkeln** für alle
Strukturarme (R5 für alle gleich: Gatter/Faktoren einmal gebaut); Training =
Vorwärts + Rückwärts, Winkel trainierbar, Gewichtsgradient der dichten Schicht.
Eingabe reell, Batch × d, zufällig.

Arme (in dieser Reihenfolge): referenz, kron2, kron3, kron2_gefaltet,
kron2_ohne, reell_kron2, vorkompiliert (nur Vorwärts, d ≤ 8.192), dicht
(d ≤ 16.384). referenz bis 16.384; Bündler-Arme bis 65.536.

**Vortest** (`--vortest`, läuft vor jeder Messung automatisch): jeder Ausgang
T1/T1/TU/T0/T0/Auflösung-durchgefallen auf synthetischen Daten erzwungen, je
5 Rauschziehungen, geprüft auf den Verzweigungstext; dazu der Fremdlastfall.
Der erste Vortest fiel durch — **am Datengenerator**, nicht an der Auswertung
(kron2 lag in den synthetischen Daten bei 16.384 langsamer als referenz). Nach
Korrektur des Generators bestanden, `urteil()` unverändert.

**Geräteprüfung** vor jeder Messung: jeder Arm auf dem Zielgerät gegen
`ButterflyFast` auf der CPU, n ≤ 12, Schranke 10⁻⁵.

**Fremdlast:** Das Skript bricht ab, wenn ein anderer Python-Prozess > 20 % CPU
zieht. Mit `--fremdlast-erlaubt` läuft es, markiert aber `verwertbar: false`,
und das Urteil lautet „NICHT VERWERTBAR“.

### Die Läufe (je höchstens 15 min, Schutzgrenze im Skript)

Aus dem Ordner `A`, **allein auf dem Rechner**:

```
python server_setup/gpu_benchmark_buendler.py --geraet cpu --threads 1 --batch 64  --modus vorwaerts --qubits 8-16
python server_setup/gpu_benchmark_buendler.py --geraet cpu --threads 1 --batch 256 --modus vorwaerts --qubits 8-14
python server_setup/gpu_benchmark_buendler.py --geraet cpu --threads 1 --batch 64  --modus training  --qubits 8-15
python server_setup/gpu_benchmark_buendler.py --geraet cpu --threads 1 --batch 256 --modus training  --qubits 8-13 --max-n-dicht 13
```

Der erste ist der Primärlauf (H1, H2). Reißt die Schutzgrenze, wird bis zur
erreichten Breite gespeichert und `abgebrochen_vor_n` vermerkt; dann ist H1
nur auswertbar, wenn d = 16.384 erreicht wurde.

### GPU (MPI-Rechner) — explorativ

```
python server_setup/gpu_benchmark_buendler.py --geraet cuda --batch 64  --modus vorwaerts --qubits 8-16
python server_setup/gpu_benchmark_buendler.py --geraet cuda --batch 256 --modus vorwaerts --qubits 8-16
python server_setup/gpu_benchmark_buendler.py --geraet cuda --batch 64  --modus training  --qubits 8-16
```

`torch.cuda.synchronize()` vor und nach jedem Lauf; TF32 bleibt aus (wird
protokolliert). Für die GPU gibt es **keine** vorregistrierte
Richtungshypothese — die Datenbasis fehlt. Vorab notierte Erwartung: kron2
gegen referenz bei 4.096 **≥ 5×** (≈ 20 statt ≈ 70+ Kernelstarts je Lage-
Satz); gegen dicht ist die GPU-GEMM so stark, dass ein Übergang ≤ 16.384
**nicht** erwartet wird. Wird ein Übergang gefunden, ist er eine neue
Hypothese für einen eigenen, vorregistrierten Lauf.

## 5. Was dieser Entwurf nicht behauptet

- Keine Aussage über Qualität. Der Bündler rechnet **dieselbe Funktion**; ob
  sie als Schicht taugt, entscheiden weiter die Q-Läufe.
- Keine Aussage über den Chatbot bei d_model 256. Dort liegt jede Struktur
  unter ihrem Übergangspunkt; T1 würde den Faktor 64 zwischen „trainierbar“
  und „gewinnt“ auf 16 senken, nicht aufheben.
- `vorkompiliert` ist kein Effizienzgewinn gegen dicht, sondern zeigt, dass
  die 2nL Parameter ohne Tempoverlust gegenüber einer dichten Schicht
  auskommen, wenn man d² Speicher zahlt.

## 6. Einordnung in die T-Serie (Phase 2: Zwischendarstellung)

Der Bündler ist die erste **Umschreibregel, die aus der Darstellung in
Abschnitt 3 des T-Serien-Entwurfs folgt, statt aus einem Handkernel**:
`[Permutation ∘ PaarDrehung(n−1) ∘ … ∘ PaarDrehung(0)]^L` wird zu
`[Permutation ∘ GEMM_B ∘ GEMM_A]^L`, und R2 („Permutation in die Adressierung
ziehen“) ist mit `kron2_gefaltet` erstmals **ausgeführt und verifiziert** —
in PyTorch kostet sie Stack-Kopien, in einem eigenen Kernel (Triton-GEMM mit
Indexberechnung im Epilog) wäre sie kostenlos. Die Wahl der Gruppengröße ist
ein Parameter des Codegenerators (Radix), kein neuer Baustein.
