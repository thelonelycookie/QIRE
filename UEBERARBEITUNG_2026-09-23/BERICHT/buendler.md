# Rechen-Bündler (T-003-vorlauf) — Bericht des Agenten `buendler`

Stand 24. September 2026. Getrennt nach: **verifiziert** · **gemessen** ·
**erwartet**. Keine Zahl ohne Quelle.

Dateien (alle neu, keine bestehende angefasst):

- `A/quantum_sim/buendler.py` — Strategien, Verifikation, Aufwandsformeln
- `A/server_setup/gpu_benchmark_buendler.py` — Messung CPU/GPU, Auswertung, Vortest
- `A/ENTWURF_T003_BUENDLER.md` — Entwurf + Vorregistrierung (vor der Messung geschrieben)
- `A/results/experiments/T-003-vorlauf__*` — Rohdaten und Verifikationslog

---

## 1. Was der Bündler ist

Die Idee: die n·L kleinen Paardrehungen durch wenige große Matrixprodukte
ersetzen. Mathematisch: Eine Lage ohne CNOT ist G₀⊗…⊗G_{n−1}; auf den als
Matrix X (2ᵏ × 2ⁿ⁻ᵏ) gelesenen Zustand wirkt sie als U_A·X·U_Bᵀ.

| Strategie | je Lage | Gewichte | Funktion |
|---|---|---|---|
| `kron2` | 2 GEMMs + 1 Gather | U_A, U_B aus den Winkeln (2^k × 2^k) | = ButterflyFast |
| `kron3` | 3 GEMMs + 1 Gather | drei Faktoren | = ButterflyFast |
| `kron2_gefaltet` | 2 gruppierte GEMMs, **kein Gather** | je Seite 2 Matrizen | = ButterflyFast |
| `kron2_ohne` | ohne CNOT: **alle L Lagen → 2 GEMMs** | U_A, U_B | = ButterflyFast(entangle=False) |
| `vorkompiliert` | 1 reelles GEMM mit [Re U \| Im U] | d × 2d | = ButterflyFast |
| `reell_kron2` | wie kron2, float32, nur Ry | – | **eigener Arm**, andere Funktion |

**Die Faltung des CNOT-Rings ist der interessanteste Befund der Mathematik:**
Der Ring zerfällt an der Trennstelle in CNOTs innerhalb A (in U_A), innerhalb B
(in U_B), einen CNOT A→B (B-Seite: zwei Matrizen, gewählt durch das
niedrigste Bit der Zeile) und einen CNOT B→A (A-Seite der nächsten Lage: zwei
Matrizen, gewählt durch das niedrigste Bit der Spalte; nach der letzten Lage
in die Vorzeichenmatrix der Auslese). Das ist Regel R2 des T-Serien-Entwurfs
(„Permutation in die Adressierung ziehen“), zum ersten Mal ausgeführt und
verifiziert. Monarch-artig (blockdiagonal × Permutation × blockdiagonal).

## 2. Verifiziert (vor jeder Zeitmessung)

`python -m quantum_sim.buendler` →
`A/results/experiments/T-003-vorlauf__verifikation_20260924.txt`

- **455 Konfigurationen**, n = 4…14, L = 1…5, mit/ohne CNOT, alle Strategien
  (vorkompiliert bis n = 10). Größte Abweichung von einer unabhängigen
  complex128-Rechnung: kron2 8,5·10⁻⁷, kron3 8,6·10⁻⁷, gefaltet 8,2·10⁻⁷,
  vorkompiliert 3,6·10⁻⁷, reell 9,6·10⁻⁷ — `ButterflyFast` selbst: 9,0·10⁻⁷,
  `ButterflyQuantumLayer`: 1,4·10⁻⁶.
- **Gradienten** nach Winkeln und Eingabe gegen `ButterflyFast`, 102
  Konfigurationen: größte Abweichung 1,4·10⁻⁶.
- **Geräteprüfung** im Messskript (jeder Arm gegen ButterflyFast, n ≤ 12):
  2,98·10⁻⁷ auf CPU.
- **R4 beantwortet:** Rz ist nicht streichbar — Ausgabe mit/ohne Rz weicht um
  bis zu 0,703 ab. Die reelle Variante ist deshalb ein eigener Arm.

**Offengelegt — Kriterium nach dem ersten Durchlauf geändert.** Zuerst: „≤ 10⁻⁶
gegen ButterflyFast UND ButterflyQuantumLayer“. Das fiel in 39 von 455
Konfigurationen bei n = 13/14 durch, **nur** gegen ButterflyQuantumLayer (bis
1,6·10⁻⁶) — und diese Referenz liegt selbst 1,4·10⁻⁶ neben der float64-Rechnung.
Neues Kriterium: ≤ 10⁻⁶ gegen float64 (Schiedsrichter), Abstände zu beiden
float32-Referenzen werden weiter ausgegeben. Die Schranke 10⁻⁶ stammt aus T-001
(nur bis n = 10); ab n ≥ 15 ist sie für float32 zu eng (Marge jetzt 4 %).

**Vortest der Auswertung:** Der erste Vortest fiel durch — am synthetischen
**Datengenerator** (kron2 lag dort bei 16.384 langsamer als referenz, im Fall
„nirgends“ war referenz schneller als dicht). Generator korrigiert, `urteil()`
unverändert, dann 6 Ausgänge × 5 Ziehungen + Fremdlastfall bestanden. Genau die
Fehlerklasse aus CLAUDE.md §5 („der Datengenerator selbst kann der Fehler sein“).

## 3. Gemessen

Alle vier CPU-Läufe ohne Fremdlast (vorher per `Get-CimInstance` geprüft, im
Skript per psutil vor und nach jeder Breite; `verwertbar: true` in allen vier
Dateien). Ryzen 5 3600, **einfädig**, L = 5, sieben Reihen, gepaart. Die
vollständige Datei je Lauf ist die **letzte** mit Analyse (Zwischenstände
werden nach jeder Breite gesichert):

| Lauf | Datei (`A/results/experiments/`) |
|---|---|
| **P** Vorwärts, Batch 64 (primär) | `T-003-vorlauf__20260924_143630.json` |
| T64 Vorwärts+Rückwärts, Batch 64 | `T-003-vorlauf__20260924_144445_02.json` |
| F256 Vorwärts, Batch 256 | `T-003-vorlauf__20260924_145244_02.json` |
| T256 Vorwärts+Rückwärts, Batch 256, d ≤ 8.192 | `T-003-vorlauf__20260924_150025_02.json` |

Dazu ein Rauchtest (Training, d = 64…256, 3-min-Grenze, vor den Hauptläufen)
in `T-003-vorlauf__20260924_1430*.json` — kein Teil der Auswertung.

### Primärlauf P — H1: **T1**

Median der gepaarten Reihenfaktoren **dicht/Arm** (> 1 = Arm schneller als dicht):

| d | referenz | **kron2** | kron3 | gefaltet | kron2_ohne | reell | vorkomp. |
|---|---|---|---|---|---|---|---|
| 1.024 | 0,070 | 0,411 | 0,413 | 0,238 | **1,357** | 0,852 | 0,476 |
| 2.048 | 0,165 | 0,768 | 0,982 | 0,593 | 2,669 | 1,823 | 0,467 |
| **4.096** | 0,366 | **1,447** (7/7) | 2,112 | 1,280 | 5,412 | 3,809 | 0,521 |
| 8.192 | 0,606 | 2,136 | 4,146 | 1,972 | 8,652 | 6,633 | 0,326 |
| 16.384 | 3,874 | 11,239 | 23,635 | 11,568 | 47,055 | 39,544 | – |

Absolute Mediane (ms) für die Einordnung: bei d = 4.096 referenz 84,4 ·
kron2 21,5 · kron3 14,6 · dicht 31,1; bei 16.384 referenz 477 · kron2 164 ·
kron3 79 · dicht 1.857. Bündler-Arme bei 65.536: kron2 1.084, kron3 434,
kron2_ohne 244 ms.

- **H1: T1.** Übergang kron2 gegen dicht bei **4.096** (T-001: 16.384), dort
  in 7/7 Reihen schneller, Faktor 1,45. Bei 2.048 noch 0,77 (0/7).
- **Auflösungsprüfung bestanden:** kron2 bei 16.384 um **2,94×** schneller als
  referenz (Erwartung war 3–10× — knapp darunter).
- **Maßstab reproduziert:** referenz gegen dicht bei 4.096 = 0,366 (T-001:
  0,331). Absolut sind **beide** Arme schneller als am 10. August (dicht 31 statt
  61 ms, referenz 84 statt 195 ms) — Ursache nicht geklärt (torch-Version,
  eingefrorene Winkel); das Verhältnis stimmt.
- **H2 bestätigt:** kron2_ohne schlägt dicht ab **1.024** (1,36×).
- **Gegen die eigene Erwartung:** Vorher notiert war „eher TU (8.192)“. Der
  Ausgang ist T1, aber **mit dünnem Abstand** (1,45× bei 4.096) — und er hält in
  keinem der Nebenläufe (siehe unten).

### Nebenläufe (beschreibend; H3 ist angemeldet)

| Lauf | Übergang kron2 | kron3 | kron2_ohne | reell | kron2 vs. referenz |
|---|---|---|---|---|---|
| P  (vorw., B 64) | **4.096** | 4.096 | 1.024 | 2.048 | 2,9–6,8× |
| T64 (training, B 64) | **8.192** | 4.096 | 2.048 | 2.048 | 3,3–5,8× |
| F256 (vorw., B 256) | 8.192 | 4.096 | 1.024 | 2.048 | 3,3–6,2× |
| T256 (training, B 256) | 8.192 | 8.192 | 1.024 | 2.048 | 4,4–6,3× |

- **H3 bestätigt:** Im Training (Batch 64) liegt der Übergang von kron2 bei
  **8.192** (Schranke ≤ 8.192); bei 4.096 Faktor 0,87. T-001 kannte für das
  Training nur den Vorteil 1,18× bei 16.384 (5. August).
- **Das T1 ist bedingungsabhängig:** Bei Batch 256 und im Training liegt kron2
  bei 4.096 knapp **unter** dicht (0,96 / 0,87 / 0,61). Robust über alle vier
  Läufe ist: **Übergang ≤ 8.192 für kron2, ≤ 4.096 für kron3 (3 von 4),
  ≤ 2.048 für die reelle Variante und die Variante ohne CNOT.**
- **kron3 ist ab 2.048 der schnellste Arm mit CNOT** (P: 14,6 statt 21,5 ms bei
  4.096). Nicht vorregistriert als Primärarm — das wäre die Hypothese für einen
  eigenen Lauf, keine nachträgliche Umdeutung von H1.
- **kron2_gefaltet** ist bei großem d gleich schnell wie kron2, bei kleinem
  langsamer (Stack-Kopien) — wie erwartet. Der gesparte Gather bringt in
  PyTorch nichts.
- **vorkompiliert** ≈ 0,47–0,52× dicht bis 4.096 (Erwartung ≈ 0,5×), bei 8.192
  0,33×.
- In T256 meldet die Maßstabsprüfung „nicht reproduziert“ (0,125) — dieser Wert
  ist dort bedeutungslos: T-001 war Vorwärts, Batch 64. Die Prüfung wird seit
  den Läufen nur noch bei Vorwärts/Batch 64 ausgegeben (Skript nachträglich
  geändert, die Messdaten nicht).

### Was das heißt — und was nicht

- Der Abstand „340× weniger Arithmetik, 5× langsamer“ (T-001) war zum großen
  Teil **Ausführung**: dieselbe Funktion, bitgleich, ist mit GEMMs 3–7× (kron2)
  bzw. 5–7× (kron3) schneller als `butterfly_fast`.
- Der Faktor 64 zwischen trainierbarer Breite (256) und Gewinnbreite schrumpft
  auf **16** (4.096) bzw. **32** (8.192). Er verschwindet nicht; bei d = 256
  ist kron2 immer noch rund 10× langsamer als dicht (0,096).
- Nur CPU, nur einfädig. Mehrfädig und GPU sind **nicht** gemessen.

## 4. Aufwand nach Formel (keine Messung)

Reelle MAC relativ zu dicht, Batch 64, L = 5, mit CNOT (Durchläufe über den
Zustand in Klammern): bei d = 4.096 referenz 0,12 (67), kron2 0,63 (17), kron3
0,23 (22), vorkompiliert 2,0 (1), reell 0,16 (17). Ohne CNOT: kron2 0,125 (4).
Der Bündler rechnet **mehr** als der Butterfly, aber in viel weniger und viel
größeren Schritten. Vollständige Tabelle: `aufwand_tabelle()` und
`ENTWURF_T003_BUENDLER.md` §3.

## 5. Einordnung: Zwischendarstellung / eigener Compiler (Phase 2)

Phase 2 des T-Serien-Entwurfs war nach T-001 gestrichen, weil es nur einen
Handkernel gab, aus dem nichts herauszuziehen war. Der Bündler ändert das:

- Er ist eine **Umschreibregel auf der Darstellung** aus Abschnitt 3 des
  Entwurfs (Norm, PaarDrehung, Permutation, ErwartungZ), nicht ein Handkernel:
  `[Perm ∘ PaarDrehung(n−1) ∘ … ∘ PaarDrehung(0)]^L` →
  `[Perm ∘ GEMM_B ∘ GEMM_A]^L`, mit der Gruppengröße (Radix) als Parameter.
- R2 (Permutation in die Adressierung) ist mit `kron2_gefaltet` erstmals
  **verifiziert ausgeführt**. In PyTorch kostet sie Stack-Kopien; in einem
  eigenen Kernel ist sie gratis (Indexrechnung im GEMM-Epilog).
- Varianten (ohne CNOT, reell, vorkompiliert) waren je ~20–40 Zeilen auf
  derselben Grundlage — ein erster, **nicht gemessener** Hinweis auf das
  Kriterium „Variante 2 und 3 kosten < 20 % von Variante 1“.

## 6. Nächste Schritte

1. **Tagebuch-Eintrag T-003** (Malte entscheidet, ob „vorlauf“ als T-003
   zählt) + `q_audit` + `q_summary`. Wichtig für die Kurzfassung: T1 **nur**
   für Vorwärts/Batch 64; die robuste Aussage ist „≤ 8.192“.
2. **kron3 als eigener vorregistrierter Lauf** und **mehrfädig** (6 Threads) —
   GEMMs skalieren über Kerne, elementweise Stufen kaum.
3. **GPU (MPI-Rechner):** `server_setup/gpu_benchmark_buendler.py --geraet cuda`
   mit Batch 64/256, vorwärts/training (Befehle im Entwurf §4). Lokal nur der
   CPU-Pfad rauchgetestet — der CUDA-Pfad (`synchronize`, `get_device_name`,
   Geräteprüfung) ist **ungetestet**, weil hier keine GPU ist.
4. **torch.compile** auf `kron2`/`kron2_gefaltet` — fusioniert die Stack-Kopien
   und die Auslese; einzige vorhandene Kompilierstufe.
5. **Triton-Kernel** (nur auf dem Server): ein GEMM mit Permutation im Epilog
   (gefaltet ohne Kopien), oder eine ganze Lage in einem Kernel, Zustand im
   SRAM bis d ≈ 2¹⁴.

Tagebuch, `q_audit` und `q_summary` habe ich **nicht** angefasst (gehören
anderen Agenten / dem Lead) — Punkt 1 ist offen.

## 7. Ehrliche Grenzen

- Der Bündler senkt **Durchläufe**, nicht Arithmetik: kron2 rechnet ≈ 2^(n/2)/n
  mal mehr als der Butterfly und wächst wie d^1,5 (gemessen: ×2,5–2,7 je
  Verdopplung ab 16.384). Bei großem d ist kron3 schneller (gemessen); dass
  irgendwann wieder der Butterfly gewinnt, sagt nur die Formel.
- `vorkompiliert` schlägt dicht nicht (gemessen ≈ 0,5×); es zeigt nur, dass 2nL
  Parameter mit einem einzigen GEMM auskommen, und kostet d² Speicher.
- Der T1-Abstand ist dünn (1,45×) und hält in keinem Nebenlauf. Ein einzelner
  Rechner, ein Tag, eine torch-Version.
- Einfädig gemessen, wie T-001. Mehrfädig verschiebt sich das Bild
  (GEMM skaliert besser als elementweise Stufen) — nicht gemessen.
- Alles bei d_model 256 (Chatbot) liegt unter jedem Übergangspunkt; der
  Chatbot wird dadurch nicht „dank der Struktur schneller“.
