# QIRE AI — Roadmap zu Jugend forscht

Ziel *(Stand 1. August, inzwischen überholt — siehe Kasten unten)*: belastbar
zeigen, dass quanteninspirierte Routing-Mechanismen (QIRE) bei gleicher
Leistung ressourceneffizienter sind als klassische Baselines — mit sauberer
wissenschaftlicher Absicherung (Theorie + Statistik), nicht nur einem Demo-Lauf.

**Tatsächliches Ziel seit dem 3. August:** Diese Ausgangsthese ist widerlegt —
sieben Mechanismen, sieben faire Abgrenzungstests, keiner hat gehalten. Die
Arbeit fragt jetzt nicht mehr *ob* Quantenprinzipien Ressourcen sparen, sondern
**unter welchen Bedingungen ein solcher Anspruch überhaupt prüfbar ist** — und
liefert dafür ein Kriterium, sieben durchgeführte Abgrenzungstests und drei
einzeln gemessene Gründe, warum frühere Messungen (auch eigene) das Gegenteil
zu sehen glaubten.

Anmeldeschluss Jugend forscht: **30. November** (jährlich). Arbeitsmodus: allein,
kein fester Betreuer. Theorie und Empirie sollen gleich stark einfließen.

Stand: 2026-08-01. Pipeline läuft Ende-zu-Ende (Doctor-Check, Baseline- und
QIRE-Modelle, Multi-Seed-Run mit 80 Jobs, 0 Fehler).

---

## Lesehinweis — nachgetragen am 4. August 2026, abends

Diese Datei brach bis zum 4. August nach der K-Serie (Quantenkerne, 3. August)
ab und war damit rund anderthalb Tage hinter dem Stand zurück — ausgerechnet um
den Zeitraum, in dem vier Hauptaussagen gefallen sind. Nachgetragen wurde:

- **die komplette B-Serie B-003 bis B-014** (eigener Abschnitt „B-Serie — die
  Methodenarbeit"), inklusive der beiden einzigen belegten Positivbefunde
  (Rang-64-Projektion aus B-011, Effektivrang-Gesetz aus B-014)
- **die Gesamtbilanz mit sieben statt sechs Mechanismen** — die Quantenkerne
  fehlten in der alten Tabelle
- der aktuelle **Stand der laufenden Läufe** (B-007, Q-003-replication)
- Phasenplan, Zielsetzung und Checkliste auf den tatsächlichen Stand gebracht

**Frühere Einträge bleiben unverändert stehen**, auch wo sie inzwischen
widerlegt sind — sie dokumentieren den Verlauf, und der ist Teil des Ergebnisses.
Wo eine Aussage nicht mehr gilt, steht ein Marker direkt dabei. Betrifft vor
allem die Tensor-Train-Abschnitte und die als „Hauptaussage des Projekts"
ausgerufene 819×-Aussage.

**Im Zweifel gilt `PROJEKTTAGEBUCH.md`.** Dort steht jede Zahl mit ihrer
Herleitung; diese Datei ist die Kurzfassung des Wegs. Zwei parallel gepflegte
Verlaufsdokumente sind genau der Grund, warum diese Divergenz entstanden ist.

**Update 2026-08-01 — Punkt 1 erledigt:** Die vermeintliche "Instabilität" des
softmax_state-Gates auf `wine` (Std.-Abw. ≈0.21) war kein Modellverhalten,
sondern ein **Bug in der Analyse-Gruppierung**: `QIREModel.method`
(`models_qire.py`) und `StandardMoE.method` (`models_efficient.py`) haben
`num_experts`/`expert_hidden_dim` nicht in den Namen codiert. Dadurch wurden
zwei strukturell verschiedene Architekturen (z. B. 4 Experten/12 hidden vs.
2 Experten/8 hidden) unter demselben Label gepoolt und ihre unterschiedliche
Accuracy fälschlich als Seed-Streuung interpretiert. Betroffen war nicht nur
das Seed-Stability-Chart, sondern auch `verifier.py` (Comparisons/Claims),
da beide über `(dataset, model_name, method)` gruppieren. Fix: Methodennamen
um `_e{num_experts}_h{expert_hidden_dim}` erweitert, Testdaten neu erzeugt.
Nach dem Fix liegen alle Modelle bei Std.-Abw. ≈0.014–0.022 — unauffällig.
Sauberer 80-Job-Vergleich zeigt jetzt: QIRE (Born-Gate) auf `wine`
"preliminarily supported" gegen MLP, auf `breast_cancer` noch "unclear or not
confirmed".

## Bekannte Lücke

Vergleich basiert weiterhin nur auf Mittelwert/Std.-Abweichung — es gibt
**keinen echten Signifikanztest** (kein scipy, kein t-Test/Wilcoxon in
`verifier.py` / `scientific_gatekeeper.py`). Muss vor belastbaren Aussagen
behoben werden (siehe Punkt 2 unten).

**Update 2026-08-01 — Punkt 2 erledigt:** Gepaarter Wilcoxon-Signed-Rank-Test
plus paarweiser t-Test über exakt gematchte Seeds in `verifier.py` eingebaut
(`paired_significance_test`, genutzt in `build_comparisons` /
`build_verification_reports`). scipy war bereits transitiv über scikit-learn
installiert, jetzt auch explizit in `requirements.txt` und
`requirements_server.txt` sowie im `--doctor`-Check gelistet.

Ergebnis auf den aktuellen 10-Seed-Daten (wine, breast_cancer):

| Modell | Datensatz | Δ Balanced Acc. | Ops-Ersparnis | p-Wert (Wilcoxon/t) | Bewertung |
|---|---|---|---|---|---|
| QIRE (Born) | wine | -0.016 | +19.3% | 0.176 / 0.147 (n.s.) | preliminarily supported |
| QIRE (Born) | breast_cancer | -0.002 | -37% (mehr Ops) | 0.605 / 0.518 | unclear, keine Effizienz |
| StandardMoE | wine | -0.020 | +39.8% | 0.035 / 0.028 (signifikant) | nicht bestätigt (echter Verlust) |
| StandardMoE | breast_cancer | -0.010 | +38.3% | 0.004 / 0.002 (signifikant) | nicht bestätigt (echter Verlust) |
| PrunedMLP | beide | ~-0.003 | +50% | n.s. | preliminarily supported |

Wichtigster Fund: Vorher (nur Mittelwert-Heuristik) wurde StandardMoE
fälschlich als "partly confirmed" gewertet, obwohl der Genauigkeitsverlust
statistisch echt ist. QIRE (Born-Gate) auf `wine` ist bislang die einzige
QIRE-Konfiguration mit realer Ops-Ersparnis und keinem nachweisbaren
Genauigkeitsverlust — das ist der aktuell stärkste Kandidat für die
Kernaussage des Projekts. Auf `breast_cancer` bringt QIRE noch keine
Ops-Ersparnis (aktuell sogar mehr Ops als Baseline) und braucht eine andere
Architekturkonfiguration.

**Update 2026-08-01 — Punkt 3 erledigt, wichtigster Fund bisher:** Geprüft, ob
"weniger Ops" auch "wirklich schneller" bedeutet. Direkter Benchmark
(MLP vs. QIRE, verschiedene Batch-Größen und Modellgrößen):

| Setup | Batch | QIRE / MLP Laufzeit |
|---|---|---|
| hidden=32, 4 Experten, top_k=1 | 30 | 16.8x langsamer |
| hidden=32, 4 Experten, top_k=1 | 100.000 | 4.2x langsamer |
| hidden=256, 16 Experten, top_k=1 (6.25% aktiv) | 100 | 26.4x langsamer |
| hidden=256, 16 Experten, top_k=1 (6.25% aktiv) | 200.000 | 2.16x langsamer |

Ergebnis: **kein einziger getesteter Fall zeigt einen echten Geschwindigkeits-
vorteil** — auch nicht bei riesigen Batches oder günstiger Sparsity (6.25%
aktive Experten). Der Overhead schrumpft mit Skalierung, verschwindet aber
nicht. Ursache: `SparseExpertLayer.forward()` (`sparse_experts.py`) berechnet
zwar nur die ausgewählten Experten (echte Sparsity), aber über eine
Python-`for`-Schleife mit `.nonzero()`/Gather pro Experte — dieser
Dispatch-Overhead dominiert auf CPU bei diesen Modellgrößen jede
FLOP-Ersparnis. Das ist kein Bug, sondern der bekannte Grund, warum reale
MoE-Systeme fusionierte/vektorisierte Routing-Kernel statt Python-Schleifen
brauchen.

Damit war die Kennzahl `estimated_ops` (theoretischer FLOP-Proxy) im ganzen
Projekt bislang von der tatsächlich gemessenen Inferenzzeit entkoppelt — teils
sogar gegenläufig. Fix in `verifier.py`: neues Flag `ops_time_conflict`
(ops_reduction_percent > 0 aber time_saving_percent < -20%) wird jetzt in
`build_comparisons`/`build_verification_reports` automatisch erkannt und labelt
den Claim explizit als "ops-count proxy improves but measured wall-clock time
is much worse" statt es als "preliminarily supported" durchgehen zu lassen.
Nach dem Fix: **alle** bisherigen Ops-basierten "partly confirmed"-Ergebnisse
(PrunedMLP, QIRE auf wine, StandardMoE) sind jetzt korrekt als "kein echter
Speedup" gekennzeichnet.

**Für die Projektthese heißt das:** "Ressourceneffizient bei gleicher Leistung"
ist aktuell nur als *theoretische* Ops-Reduktion belegt, nicht als gemessene
Beschleunigung. Zwei ehrliche, beide wissenschaftlich wertvolle Wege für
Phase 2:
1. Die theoretische Ops-Reduktion als eigenständige, klar benannte Kennzahl
   führen (üblich in ML-Papers), und *getrennt* diskutieren, warum sie sich
   hier (noch) nicht in Wall-Time niederschlägt — das ist selbst ein valider
   Forschungsbeitrag.
2. Optional (größeres Vorhaben): `SparseExpertLayer` vektorisiert neu
   implementieren (z. B. sortierte Tokens + gebündelte Matmuls statt
   Python-Schleife) und prüfen, ob sich dann bei großen Batches ein echter
   Crossover zeigt. Wäre ein starkes, eigenständiges Kapitel ("Wir haben den
   Engpass gefunden und behoben").

**Update 2026-08-01 — Vektorisierter Fix umgesetzt und sauber gemessen:**
`SparseExpertLayer.forward()` (`sparse_experts.py`) neu geschrieben:
Grouped-GEMM-Dispatch (Tokens per `argsort` nach zugewiesenem Experten sortieren,
dann ein zusammenhängender Batch-Forward pro Experte) statt der alten
Python-Schleife mit `.nonzero()`/Gather pro Experte. Korrektheit verifiziert
(identische Outputs, max abs diff = 0.0, Gradienten getestet), komplette
Test-Suite (Ablation, 80-Job-Seed-Stability) läuft weiterhin fehlerfrei durch.

Wichtig: Erster Benchmark-Durchlauf lief versehentlich Multi-Threaded und
zeigte einen zu optimistischen Crossover. Die echte Pipeline erzwingt pro Job
`torch.set_num_threads(1)` (`benchmark.py: set_seed()`) — das musste im
Benchmark exakt nachgebildet werden, sonst ist der Vergleich nicht fair
(ein einzelner großer Dense-Matmul profitiert sonst unfair von
Multi-Core-BLAS, was Sparse-Routing gegenüber der Baseline benachteiligt).
Reproduzierbares Benchmark-Skript jetzt unter `qire_ai/benchmark_sparse_dispatch.py`
(`python -m qire_ai.benchmark_sparse_dispatch`), Ergebnis-Plot unter
`results/figures/sparse_dispatch_crossover.png`.

Ergebnis mit korrekt Single-Threaded gemessenen Zahlen:

| Modellgröße | Batch | naive/MLP | vektorisiert/MLP |
|---|---|---|---|
| klein (hidden=32, 4 Experten) | 30 | 10.1x langsamer | 7.0x langsamer |
| klein (hidden=32, 4 Experten) | 100.000 | 1.9x langsamer | 1.9x langsamer |
| groß (hidden=256, 16 Experten, 6.25% aktiv) | 1.000 | 2.1x langsamer | 1.1x langsamer |
| groß (hidden=256, 16 Experten, 6.25% aktiv) | 10.000 | **0.56x (44% schneller)** | **0.42x (58% schneller)** |
| groß (hidden=256, 16 Experten, 6.25% aktiv) | 100.000 | 0.62x (38% schneller) | **0.43x (57% schneller)** |

Fazit: Ein echter, gemessener Crossover **existiert** — aber nur bei
ausreichend großem Modell (viele Experten, hoher Sparsity-Grad) und großem
Batch (≥ ca. 10.000). Der vektorisierte Fix bringt in diesem Bereich eine
zusätzliche, deutliche Verbesserung gegenüber der alten Implementierung
(z. B. bei Batch=10.000: 44% → 58% schneller). Auf den tatsächlichen
Projekt-Datensätzen (`wine`: ~36 Testzeilen, `breast_cancer`: ~114) ist
weder Modell noch Batch groß genug für den Crossover — dort bleibt QIRE
weiterhin langsamer in echter Zeit (aber spürbar weniger stark als vorher,
z. B. QIRE auf wine: -1539% → -1279% Zeit-Differenz nach dem Fix).

**Für die Projektthese:** Das ist eine differenzierte, aber sehr gut belegbare
Aussage: "Die quanteninspirierte Sparse-Routing-Idee bringt bei den aktuell
verwendeten kleinen Datensätzen keinen echten Geschwindigkeitsvorteil, aber
das Prinzip funktioniert nachweislich ab einer bestimmten Modell-/Batch-Größe
— und der ursprüngliche Software-Engpass (naive Python-Dispatch) wurde
gefunden und behoben." Das ist ein eigenständiges, präsentables Ergebnis
(inkl. Diagramm) unabhängig vom Ausgang der Accuracy-Vergleiche.

**Update 2026-08-01 — Phase 2 gestartet, weiterer latenter Bug gefunden:**
Vor dem Start der Ablation geprüft, ob der Punkt-1-Bug (Methodenname ohne
Kapazitätsparameter) auch bei den Baseline-Modellen existiert — ja: `DenseMLP`,
`DeepMLP`, `LowRankMLP`, `EarlyExitMLP`, `PrunedMLP`, `QuantizedMLP` hatten alle
einen reinen Klassenkonstante-`method`-String ohne `hidden_dim`/`rank_dim`/
`quant_bits`/`prune_amount`. Da MLP die primäre Baseline für *alle* Vergleiche
ist (`_baseline_for()` in `verifier.py`), wäre das potenziell noch kritischer
als der QIRE-Fund gewesen — hat sich in den bisherigen Läufen nur zufällig
nicht ausgewirkt, weil `hidden_dim`/`prune_amount` zwischen den verwendeten
Presets bislang übereinstimmten. Für zukünftige Läufe mit variierendem
`hidden_dim` (z. B. `classical`-Preset) wäre es sonst scharf geworden. Fix in
`models_baseline.py`/`models_efficient.py`: alle betroffenen Klassen setzen
`self.method` jetzt in `__init__` mit den kapazitätsbestimmenden Parametern;
`DistilledStudentMLP` (erbt von `PrunedMLP` ohne eigenes `__init__`) bekam ein
`_method_prefix`-Klassenattribut, damit die Vererbung den Label-Unterschied
nicht überschreibt. Mit Unit-Test verifiziert.

Breiterer Multi-Seed-Lauf abgeschlossen: 6 Datensätze (wine, breast_cancer, iris,
digits, synthetic_routing_challenge, synthetic_expert_specialization — die
letzten beiden sind extra für Routing-/Experten-Spezialisierung gebaut) ×
20 Seeds × 4 Modelle = 480 Jobs, alle erfolgreich. Ablationsgrid (Gate ×
top_k × Experten × Expert-Hidden × State-Mapper × Temperatur, 3 Seeds,
wine+breast_cancer) läuft im Hintergrund.

**Update 2026-08-01 — wichtige Korrektur mit mehr Seeds:** Mit 20 statt 10
Seeds über 6 Datensätze ausgewertet (gepaarter Signifikanztest, read-only aus
den Rohdaten, ohne die laufende Ablation zu stören):

| Datensatz | QIRE (Born) Δ Bal.Acc | p-Wert | Ops-Ersparnis | Bewertung |
|---|---|---|---|---|
| wine | -0.016 | **0.020 (signifikant)** | 19.3% | nicht bestätigt (vorher bei n=10 noch "preliminarily supported"!) |
| breast_cancer | -0.002 | 0.193 (n.s.) | -37% (mehr Ops) | unclear, keine Effizienz |
| iris | +gut | 0.058 (knapp n.s.) | 48.2% | am ehesten noch interessant |
| digits | schlecht | 0.000002 (signifikant) | -121% (mehr Ops) | klar schlechter |
| synthetic_routing_challenge | schlecht | 0.000002 (signifikant) | 1.7% | klar schlechter |
| synthetic_expert_specialization | schlecht | 0.000002 (signifikant) | -162% (mehr Ops) | klar schlechter |

**Das ist die wichtigste Korrektur der ganzen Untersuchung:** Der bei n=10
Seeds noch vielversprechende QIRE-Befund auf `wine` (p=0.176, nicht
signifikant) kippt bei n=20 Seeds zu einem **statistisch signifikanten
Genauigkeitsverlust** (p=0.020). Das ist exakt der Grund, warum Punkt 2
(echter Signifikanztest) und "mehr Seeds" so wichtig sind: ein Ergebnis, das
mit wenig Daten "vielversprechend" aussieht, kann bei mehr Daten kippen. Auf
den beiden extra für Routing/Experten-Spezialisierung gebauten synthetischen
Datensätzen schneidet QIRE klar schlechter ab (teils sogar mit *mehr* Ops als
die Baseline). Einzig `iris` bleibt mit p=0.058 knapp interessant.
PrunedMLP bleibt über alle 6 Datensätze die robusteste "effizientere"
Baseline (50% Ops-Ersparnis, meist kein signifikanter Genauigkeitsverlust) --
allerdings mit demselben `ops_time_conflict` wie QIRE (gemessene Zeit ist
trotzdem schlechter, s. Punkt 3).

Kleinere Robustheitskorrektur nebenbei: `paired_significance_test` in
`verifier.py` warf bei fast identischen Wertepaaren eine scipy-`RuntimeWarning`
("precision loss... results may be unreliable") und konnte NaN-p-Werte
zurückgeben. Jetzt abgefangen: NaN-Fälle landen als erklärende Notiz statt
stiller NaN im Report, Warnung sauber unterdrückt (verifiziert mit
`-W error::RuntimeWarning`, inkl. Gegentest, dass echte signifikante Fälle
weiterhin korrekt erkannt werden). Ein Entwurf für die Kurzfassung
(`KURZFASSUNG_ENTWURF.md`) wurde ebenfalls vorbereitet -- als Gerüst mit den
echten Zahlen, zur eigenen Ausformulierung durch den Schüler, nicht als
fertiger Text.

**Ehrliche Zwischenbilanz für die Projektthese:** Nach der aktuellen Datenlage
ist die These "QIRE ressourceneffizienter bei gleicher Leistung" auf den
getesteten Datensätzen **nicht haltbar** -- bei mehr Seeds verschwinden die
vorher vielversprechenden Signale. Das ist kein Scheitern des Projekts,
sondern ein normales (und für eine Jugend-forscht-Arbeit gut dokumentierbares)
Forschungsergebnis: "Erste Hinweise bei kleiner Stichprobe hielten der
strengeren Prüfung nicht stand." Die ehrliche Doku dieses Verlaufs (10 Seeds
vielversprechend -> 20 Seeds widerlegt) ist selbst ein Beleg für sauberes
wissenschaftliches Arbeiten. Für Phase 3 lohnt sich eine gezielte Ablation
(läuft gerade im Hintergrund), um zu prüfen, ob eine andere Gate-/Mapper-
Kombination stabiler abschneidet, bevor die Kernthese final formuliert wird.

**Update 2026-08-01 — Ablation abgeschlossen, entscheidender Befund:**
1992 Jobs (0 Fehler) über das volle Gitter (Gate-Typ × State-Mapper × top_k ×
Anzahl Experten × Expert-Hidden × Temperatur, 3 Seeds, wine + breast_cancer).

**Haupteffekte (Mittelwert Balanced Accuracy über alle anderen Parameter):**

| Parameter | Werte, sortiert nach Effekt |
|---|---|
| Gate-Typ | softmax_state 0.9679, born 0.9665, interference_born 0.9662 — praktisch gleich |
| State-Mapper | trig 0.9680, compact_trig 0.9670, linear 0.9656 — praktisch gleich |
| top_k | 2: 0.9722, 1: 0.9616 — großer Effekt (mehr aktive Experten = besser) |
| Anzahl Experten | 2: 0.9708, 4: 0.9673, 8: 0.9625 — klarer Effekt (weniger Experten = besser bei diesen kleinen Datensätzen) |
| Temperatur | praktisch kein Effekt |

**Kontrollierter gepaarter Test (Born-Gate vs. klassisches Softmax-Gate, alle
anderen Parameter exakt gleichgehalten, 648 gematchte Konfigurationen):**
mittlere Differenz Balanced Accuracy = **-0.0014** (Born leicht schlechter, nicht
besser), Wilcoxon p=0.247, t-Test p=0.096 -- **kein signifikanter Unterschied.**
Interference-Born vs. Softmax: Differenz -0.0017, Wilcoxon p=0.164, t-Test
p=0.039 (uneinheitlich, und Effektrichtung ebenfalls leicht negativ statt
positiv).

**Das ist der entscheidende, sauber belegte Befund der gesamten Untersuchung:**
Über 648 kontrollierte Vergleichspaare bringt das quanteninspirierte
Born-Regel-Gate **keinen** messbaren Genauigkeitsvorteil gegenüber klassischem
Softmax-Gating -- bei ansonsten exakt gleicher Architektur. Was tatsächlich
die Genauigkeit treibt, sind ganz klassische Kapazitäts-/Sparsity-Parameter
(`top_k`, Anzahl Experten), die für jede Mixture-of-Experts-Architektur gelten,
nicht spezifisch für den quanteninspirierten Mechanismus sind.

**Zusammengeführtes Gesamtbild aller drei Untersuchungsstränge:**
1. Keine signifikante Effizienz bei gleicher Leistung (Punkt 2/3, 20-Seed-Lauf).
2. Kein Crossover in gemessener Zeit bei den getesteten Datengrößen (Punkt 3).
3. Der spezifisch "quanteninspirierte" Teil (Born-Regel-Gate) trägt selbst
   nichts Messbares zur Genauigkeit bei (Ablation) -- die Unterschiede, die
   es gibt, kommen von klassischen Architektur-Stellschrauben.

Das ist ein vollständiges, in sich konsistentes, ehrlich negatives Ergebnis
auf den bisher getesteten Datensätzen -- kein Grund zur Enttäuschung, sondern
eine seltene Gelegenheit für eine Jugend-forscht-Arbeit, die eigene
Ausgangshypothese mit sauberer Methodik selbst zu widerlegen und genau zu
erklären, warum. Das ist wissenschaftlich oft überzeugender als ein
bestätigtes Ergebnis.

## Sofort sinnvolle nächste Schritte

- [x] 1. Softmax-State-Gate-Instabilität auf `wine` untersuchen — war ein
      Gruppierungs-Bug in `models_qire.py`/`models_efficient.py`, behoben.
- [x] 2. Echten Signifikanztest einbauen (gepaarter Wilcoxon- oder t-Test über
      matched Seeds) statt nur Mittelwert/Std. — erledigt, s. Ergebnistabelle oben.
- [x] 3. Faire Ressourcen-Vergleiche sicherstellen — Ops-Proxy vs. gemessene
      Zeit geprüft: kein echter Speedup auf den Projekt-Datensätzen, Ursache
      gefunden (Python-Schleife im Routing) und behoben (vektorisierter
      Grouped-GEMM-Dispatch); echter Crossover bei großem Modell/Batch
      nachgewiesen und mit Diagramm belegt (`benchmark_sparse_dispatch.py`).
- [x] 4. Gleiches Parameterbudget zwischen den Armen — erledigt in B-004: vier
      Familien bei **exakt** 524.288 Parametern (Monarch, Blockdiagonale,
      Tensor-Train, Low-Rank), nicht nur gleiche `hidden_dim`. Dazu Lernrate je
      Arm auf eigenen Tuning-Seeds, weil ein fester Wert den Vergleich gar nicht
      fair machen konnte.
      *(Dieser Punkt stand bis zum 4. August doppelt in der Liste — einmal
      abgehakt, einmal offen, beide als „3." nummeriert.)*
- [ ] 5. `--discover` laufen lassen für unabhängige Hypothesen aus bestehenden Daten
- [x] 6. Projekttagebuch führen — `PROJEKTTAGEBUCH.md`, seit dem 1. August
      durchgehend; inzwischen das maßgebliche Dokument des Projekts.

## Phasenplan

### Phase 1 — Fundament festigen (bis Ende August 2026)
- Theorie: Born-Regel-Gating (|Amplitude|² als Routing-Wahrscheinlichkeit)
  mathematisch herleiten — als *Inspiration*, nicht als "ist Quantencomputing"
- Empirie: Gate-Instabilität klären, Signifikanztest einbauen, faire Ops-Budgets prüfen
- Output: kurze interne Notiz "Warum könnte QIRE funktionieren, was zeigt Iteration 1"

### Phase 2 — Breiter validieren (September 2026)
- Mehr Datensätze (`EXTENDED_DATASETS`, nicht nur wine/breast_cancer), mehr Seeds (≥20)
- Ablation: State-Mapper vs. Gate-Typ vs. Sparsity einzeln testen
- Literatur-Karten (`sources/literature_cards/`) als Related Work einarbeiten
  (MoE, Pruning, Tensor Networks, QAOA/QUBO, Energy-Aware Training, Pareto)
- Output: erste belastbare Diagramme mit Signifikanzangabe

**Update 2026-08-02 — strategischer Kurswechsel nach Phase 2:** Da die
Ablation zeigt, dass der klassisch-simulierte Born-Regel-Mechanismus selbst
nichts beiträgt, wird die These nicht fallengelassen, sondern um zwei neue,
besser abgesicherte Ansätze erweitert (Entscheidung mit dem Schüler
besprochen):

1. **Tensor-Netzwerke / Tensor-Train-Zerlegung** — quantenphysik-inspiriert
   (Matrix-Product-States aus der Quanten-Vielteilchen-Simulation), mit
   echtem Literatur-Beleg für reale Effizienzgewinne (Novikov et al. 2015:
   bis zu 200.000-fache Parameterreduktion einzelner Schichten bei
   gleichbleibender Genauigkeit) — anders als das Gate ein strukturell
   anderer Mechanismus, nicht nur eine andere Normierungsformel.
2. **QUBO/Ising-basiertes Pruning** — Sparsity-Muster als kombinatorisches
   Optimierungsproblem statt per Magnitude-Heuristik finden.
3. **Echte Quantenhardware** (statt nur klassischer Simulation): eigener,
   von Grund auf selbst geschriebener Quantenschaltkreis-Simulator unter
   `quantum_sim/` als neues Modul begonnen (State-Vektor-Ansatz, Tensor-
   Kontraktion statt volle 2^n×2^n-Matrizen — Standardtechnik echter
   Simulatoren). Validiert gegen 9 bekannte Schaltungen (Bell-Zustand,
   GHZ-Zustand, Superposition, Toffoli, Rz(2π), Stichproben-Häufigkeiten) —
   alle bestehen. Bewusst als eigenes Python-Modul, nicht als neue
   Programmiersprache (Aufwand/Nutzen-Abwägung mit dem Schüler besprochen).
   Nächster Schritt: dieselben Schaltungen über Qiskit auf echter
   IBM-Quantenhardware laufen lassen (kostenloser Zugang: 10 Freiminuten/Monat
   auf 100+-Qubit-Prozessoren) und mit der exakten Simulation vergleichen.

**Update 2026-08-02 — Qiskit-Anbindung:** `qiskit`/`qiskit-ibm-runtime`
installiert (jetzt auch in `requirements.txt`). Bell- und GHZ-Zustand
zusätzlich in Qiskit nachgebaut und per `qiskit_bridge.cross_validate()`
gegen den eigenen Simulator geprüft — beide unabhängigen Implementierungen
stimmen exakt überein (nach Korrektur der Bit-Reihenfolgen-Konvention:
Qiskit zählt Qubit 0 von rechts, `quantum_sim` von links — bewusst nicht
ignoriert, sondern in `to_quantum_sim_order()` explizit behandelt).
`qiskit_bridge.run_on_real_hardware()` ist vorbereitet, aber noch nicht
ausgeführt: braucht einen eigenen (kostenlosen) IBM-Quantum-Account des
Schülers -- Account-Erstellung ist etwas, das der Assistent nicht selbst
übernimmt. Anleitung dazu liegt in `quantum_sim/README.md`.

**Update 2026-08-02 — erster echter Hardware-Lauf, erfolgreich:** Account
eingerichtet, Token bereitgestellt (lokale Datei `ibm_quantum_token.txt`,
robust gegen die von PowerShell-Umleitung erzeugte UTF-16-Kodierung
eingelesen). Verbindung bestätigt: 3 echte 156-Qubit-Prozessoren verfügbar
(ibm_fez, ibm_marrakesh, ibm_kingston). `run_on_real_hardware()` brauchte noch
einen Fix (`transpile()` auf den nativen Gate-Satz der Hardware vor dem
Absenden -- H/CNOT müssen erst in die von der Hardware unterstützten
Basis-Gates umgeschrieben werden). Bell-Zustand erfolgreich auf `ibm_marrakesh`
gelaufen, Job-ID `d9n7998qs0bc73e33lgg`, 1024 Shots:

| Zustand | Ideal (Simulation) | Echte Hardware |
|---|---|---|
| `00` | 50,0 % | 50,1 % |
| `11` | 50,0 % | 48,0 % |
| `01` | 0 % | 1,1 % |
| `10` | 0 % | 0,9 % |

Reale Fehlerrate ≈2% (Anteil außerhalb der theoretisch erlaubten Zustände) --
ein greifbarer, direkter Beleg für echtes Quantenrauschen, gemessen gegen die
exakte Vorhersage des selbstgeschriebenen Simulators. Das ist ein
eigenständiges, präsentables Ergebnis: "Ich habe verstanden, simuliert *und*
auf echter Hardware verifiziert, wie groß der Unterschied zwischen Theorie
und echtem NISQ-Rauschen ist."

GHZ-Zustand (3 Qubits) ebenfalls gelaufen: Job-ID `d9n7accsfqic73aqn670`,
Fehlerrate 2,25 % (23/1024) vs. 1,95 % (20/1024) beim Bell-Zustand. Richtung
passt zur Erwartung (mehr Gates -> mehr Fehler), aber **ehrlich zu klein, um
das als belegten Trend zu behaupten** -- 20 vs. 23 Fehler-Ereignisse aus je
1024 Shots liegt im Bereich von Stichprobenrauschen. Für eine belastbare
Aussage "Fehlerrate wächst mit Schaltungstiefe" bräuchte es mehrere
Wiederholungen pro Schaltungstiefe (mehrere unabhängige Jobs, nicht nur
einen) und idealerweise mehr als zwei Tiefen-Stufen -- exakt dasselbe Prinzip
wie bei den QIRE-Seed-Läufen: ein einzelner vielversprechender Wert ist kein
Beweis, das gilt auch für Hardware-Rauschen.

**Update 2026-08-02 — 2 weitere Wiederholungen je Schaltung (n=3 gesamt):**
Bell-Fehlerraten: 1,95 % / 0,88 % / 1,56 % (Mittel 1,46 %, Std 0,54 %).
GHZ-Fehlerraten: 2,25 % / 1,76 % / 1,07 % (Mittel 1,69 %, Std 0,59 %).
Unabhängiger t-Test: p=0,65; Mann-Whitney (einseitig, GHZ>Bell): p=0,35 --
**kein signifikanter Unterschied.** Die Richtung stimmt (GHZ im Mittel höher),
aber bei n=3 pro Gruppe ist das statistisch nicht von Rauschen zu
unterscheiden. Wichtig: das widerlegt die Hypothese "mehr Gates -> mehr
Fehler" nicht (die ist physikalisch gut etabliert), es zeigt nur, dass 3
Wiederholungen nicht ausreichen, um es *hier* selbst zu belegen. Für eine
belastbare eigene Messung bräuchte es deutlich mehr Wiederholungen und/oder
mehr Tiefenstufen -- das kostet reales (begrenztes) Hardware-Kontingent,
deshalb bewusst mit dem Schüler abgestimmt, bevor mehr verbraucht wird.
Alle 6 Job-IDs sind über quantum.ibm.com nachprüfbar, gutes Rohmaterial für
die Methodik-Sektion der Arbeit.

**Update 2026-08-02 — auf n=10 pro Schaltung aufgestockt, jetzt signifikant:**
Kontingent-Check zuerst: nur 12 von 600 Sekunden (2 %) verbraucht -- deutlich
mehr Spielraum als gedacht, deshalb 7 weitere Wiederholungen je Schaltung
nachgelegt (14 zusätzliche Jobs, alle erfolgreich, Job-IDs `d9n7dv...` bis
`d9n7g2...`).

| | Mittelwert Fehlerrate (n=10) | Std.-Abw. |
|---|---|---|
| Bell (2 Gates) | 1,57 % | 0,44 % |
| GHZ (3 Gates) | 2,06 % | 0,56 % |

Unabhängiger t-Test: p=0,043. Mann-Whitney (einseitig): p=0,014. **Beide
signifikant bei n=10** -- anders als bei n=3 vorher. Damit ist "mehr
Gates/Schaltungstiefe -> mehr Fehler auf echter Hardware" jetzt nicht nur
plausibel, sondern selbst mit angemessener Stichprobengröße statistisch
belegt. Das ist ein rundes, präsentables Ergebnis für das Hardware-Kapitel:
eigener Simulator gebaut -> gegen Qiskit verifiziert -> auf echter Hardware
laufen lassen -> Rauschen erst bei n=3 nicht, dann bei n=10 sauber
nachgewiesen. Gutes Beispiel für den gesamten roten Faden der Arbeit: nicht
bei der ersten vielversprechenden Zahl aufhören.

Gesamt-Kontingent-Verbrauch für alle 20 Jobs: 40 von 600 Sekunden (6,7 %) --
reichlich Reserve für weitere Experimente in diesem Monat übrig.

## Weg 1: Tensor-Train-Kompression -- erster echter positiver Befund

**Update 2026-08-02:** `qire_ai/tensor_train.py` (`TTLinear`) implementiert:
Gewichtsmatrix als Kette kleiner Kern-Tensoren statt einer dichten Matrix,
angelehnt an Matrix-Product-States aus der Quanten-Vielteilchen-Simulation.
Validiert (`tensor_train_validate.py`, 6 Checks, u. a. gegen eine unabhängige
Einstein-Summations-Referenz): Rekonstruktion korrekt, Gradienten fließen,
kann bei ausreichendem Rang beliebige Zielmatrizen lernen, Kompressionsrate
verbessert sich mit der Matrixgröße wie von der Theorie vorhergesagt
(`tensor_train_benchmark.py`, Diagramm unter
`results/figures/tensor_train_crossover.png`).

Als `TensorTrainMLP` in die echte Pipeline eingebaut (`model_factory.py`,
neue Felder `tt_rank`/`tt_cores` in `ExperimentJob`) und über 10 Seeds auf
wine/breast_cancer/digits gegen die MLP-Baseline getestet (60 Jobs, 0 Fehler):

| Datensatz | Parameter TT/MLP | Δ Balanced Accuracy | p-Wert (Wilcoxon/t) | Bewertung |
|---|---|---|---|---|
| wine | 63,4% (-37%) | -0,0025 | 0,734 / 0,758 (n.s.) | Parameter gespart, kein Genauigkeitsverlust |
| breast_cancer | 32,3% (-68%) | -0,0036 | 0,156 / 0,096 (n.s.) | Parameter gespart, kein Genauigkeitsverlust |
| digits | 27,6% (-72%) | -0,0225 | 0,002 / 0,00002 (signifikant) | Zu aggressiv komprimiert, Genauigkeit leidet |

**Erster echter, statistisch abgesicherter positiver Befund im ganzen
Projekt:** Auf 2 von 3 Datensätzen real 37-68% weniger Parameter bei keinem
nachweisbaren Genauigkeitsverlust. Anders als beim Born-Regel-Gate ist das
kein Nullresultat. `digits` (mehr Klassen/Features) zeigt ehrlich, dass zu
aggressive Kompression relativ zur Aufgabenkomplexität schadet -- ein
erklärbarer, kein widersprüchlicher Grenzfall.

> ⚠ Dieser Befund gilt **nicht mehr**. A-002 (weiter unten) zeigt, dass eine
> Blockdiagonale Tensor-Train bei gleichem Budget schlägt; nach B-008/B-009
> ersetzt überhaupt keine strukturierte Familie eine dichte Schicht unter
> fairen Bedingungen. Der Eintrag bleibt stehen, weil er den Verlauf
> dokumentiert.

**Wichtiger Vorbehalt (gleiches Muster wie bei `SparseExpertLayer`):**
`TTLinear.forward()` rekonstruiert die volle Matrix bei jedem Aufruf und
rechnet dann normal weiter -- das spart also **Parameter/Speicherplatz**
(`parameter_count`, bereits in der Pipeline getrackt), aber **nicht**
Rechenzeit: `estimated_ops` und gemessene `inference_time_ms` sind für
TensorTrainMLP sogar höher als für MLP in allen drei Fällen. Für echte
Rechenzeit-Ersparnis bräuchte es eine "Streaming"-Kontraktion, die die Kerne
direkt auf den Input anwendet, ohne je die volle Matrix zu bilden -- als
nächster Schritt festgehalten, nicht fälschlich als bereits erledigt
dargestellt.
*(Dieser Absatz stand bis zum 4. August irrtümlich im Abschnitt zur K-Serie,
wo er thematisch nichts zu suchen hatte. Hierher zurückgesetzt.)*

### Verschränkungsentropie als Vorhersage-Kriterium (2026-08-02)

Hypothese: Die Verschränkungsentropie der Gewichtsmatrix sollte vorhersagen,
wo Tensor-Train funktioniert -- analog zum Flächengesetz in der
Quanten-Vielteilchenphysik (wenig Verschränkung -> kleine Bond-Dimension
reicht). Umgesetzt in `qire_ai/entanglement_analysis.py`: Gewichtsmatrix in
denselben Tensor höherer Ordnung umformen, den `TTLinear` nutzt, dann an
jedem Bond das Schmidt-Spektrum, die Entropie und den bei Rang 4 erfassten
Spektralanteil messen. Zufällig initialisierte Matrizen als Kontrolle.

| Datensatz | S (trainiert) | eff. Rang | erfasst@r=4 | S (zufällig) | erfasst zufällig | TT-Ergebnis |
|---|---|---|---|---|---|---|
| wine | 2,143 | 8,5 | **0,662** | 2,165 | 0,643 | kein Verlust (p=0,73) |
| breast_cancer | 2,473 | 11,9 | **0,533** | 2,588 | 0,457 | kein Verlust (p=0,16) |
| digits | 2,649 | 14,1 | **0,425** | 2,693 | 0,382 | signifikanter Verlust (p=0,002) |

**Was bestätigt wurde:** Die Reihenfolge stimmt exakt. `erfasst@r=4` ordnet die
drei Datensätze genau so, wie der TT-Erfolg es tut. Bei n=3 Datensätzen ist
das allerdings eine Beobachtung, kein Nachweis -- drei Punkte in der richtigen
Reihenfolge können Zufall sein.

**Was die Kontrolle widerlegt hat:** Die Entropie trainierter Matrizen liegt
kaum unter der zufällig initialisierter (z. B. wine 2,143 vs. 2,165). Die
Größe ist also fast vollständig durch die *Dimensionen* der Matrix bestimmt,
nicht durch gelernte Struktur. Die erhoffte physikalische Erzählung ("Training
erzeugt Zustände geringer Verschränkung") trägt so nicht. Ohne diese
Kontrolle wäre der Befund falsch interpretiert worden.

**Was die Theorie klar falsifiziert:** Nach dem MPS-Trunkierungskriterium
bräuchte man für 90 % Spektralgewicht Rang 7,8 (wine) bis 12,8 (digits) --
tatsächlich funktioniert Rang 4 auf zwei von drei Datensätzen problemlos,
obwohl er nur 66 %/53 % erfasst. Das Kriterium überschätzt den Bedarf massiv.

**Die Erklärung dafür ist der eigentliche Erkenntnisgewinn:** Die
Schmidt-Trunkierungstheorie gilt für das *Annähern einer gegebenen Matrix*
(nachträgliche Kompression). Hier wird das Netz aber von Anfang an *in der
eingeschränkten Form trainiert* -- es muss die dichte Matrix gar nicht
reproduzieren, sondern findet innerhalb der eingeschränkten Familie eine
andere, ebenso gute Lösung. Das sind zwei verschiedene Regime:

| Regime | Theorie anwendbar? |
|---|---|
| nachträgliche Kompression eines trainierten Modells | ja, Schmidt-Trunkierung gilt direkt |
| Training direkt in komprimierter Form | nein, das Netz umgeht die Einschränkung |

Daraus folgt ein sauberes, konkretes Folgeexperiment: **beide Regime direkt
gegeneinander testen.** Vorhersage: Bei nachträglicher Kompression sagt die
Entropie den Genauigkeitsverlust gut vorher, bei Training-von-Anfang-an nicht.
Falls das zutrifft, ist das ein eigenständiger, gut belegbarer Befund über
die Grenzen der Übertragbarkeit quantenphysikalischer Kriterien auf
neuronale Netze.

### Die beiden Kompressionsregime im direkten Vergleich (2026-08-02)

Umgesetzt in `qire_ai/compression_regimes.py`. TT-SVD (nachträgliche
Kompression) implementiert und vorab abgesichert: verlustfreie Umformung,
bei vollem Rang exakte Rekonstruktion (Fehler ~1e-15), Trunkierungsfehler
monoton fallend im Rang.

Aufbau: 3 Datensätze × 4 Ränge (2/4/8/12) × 5 Seeds.
- **Regime A (nachträglich):** dicht trainieren, Gewichtsmatrix durch
  Rang-r-Näherung ersetzen, *nicht* nachtrainieren.
- **Regime B (komprimiert trainiert):** TT-Modell mit Rang r von Anfang an.

| Datensatz | Rang | verworfen | dicht | post-hoc | Verlust | scratch | Verlust |
|---|---|---|---|---|---|---|---|
| wine | 2 | 0,595 | 0,987 | 0,876 | 0,111 | 0,979 | 0,008 |
| wine | 4 | 0,338 | 0,987 | 0,963 | 0,024 | 0,980 | 0,006 |
| wine | 8 | 0,067 | 0,987 | 0,973 | 0,014 | 0,994 | -0,007 |
| wine | 12 | 0,000 | 0,987 | 0,987 | 0,000 | 0,991 | -0,004 |
| breast_cancer | 2 | 0,677 | 0,968 | 0,924 | 0,043 | 0,968 | -0,000 |
| breast_cancer | 4 | 0,467 | 0,968 | 0,962 | 0,005 | 0,966 | 0,002 |
| breast_cancer | 8 | 0,190 | 0,968 | 0,966 | 0,002 | 0,967 | 0,000 |
| breast_cancer | 12 | 0,054 | 0,968 | 0,964 | 0,004 | 0,969 | -0,002 |
| digits | 2 | 0,766 | 0,970 | **0,450** | **0,520** | 0,930 | 0,039 |
| digits | 4 | 0,575 | 0,970 | **0,664** | **0,306** | 0,942 | 0,028 |
| digits | 8 | 0,291 | 0,970 | 0,928 | 0,042 | 0,955 | 0,015 |
| digits | 12 | 0,107 | 0,970 | 0,964 | 0,006 | 0,967 | 0,003 |

**Selbstkorrektur zur Teststatistik:** Ich hatte als Test die *Korrelation*
zwischen verworfenem Spektralgewicht und Genauigkeitsverlust gewählt. Die
kommt in beiden Regimen fast gleich heraus (Pearson r=0,688 vs. 0,675) --
womit die Hypothese scheinbar widerlegt wäre. Das war die falsche Statistik:
Korrelation misst nur, ob die *Reihenfolge* mitläuft, und natürlich wird es
in beiden Regimen mit stärkerer Trunkierung schlechter. Die eigentliche
Frage ist, *wie stark*.

**Mit der richtigen Statistik ist die Hypothese klar bestätigt:**

| Kennzahl | Regime A (nachträglich) | Regime B (trainiert) |
|---|---|---|
| mittlerer Genauigkeitsverlust | 0,0896 | 0,0073 |
| Steigung (Verlust je verworfenem Spektralanteil) | **+0,414 ± 0,138** | **+0,035 ± 0,012** |

- Verhältnis der Steigungen: **11,9-fach**, Unterschied signifikant (z=2,73, p=0,006)
- Regime B ist in **12 von 12** Fällen besser (gepaarter Wilcoxon p=0,00049)
- Extremfall `digits` bei Rang 2 (76,6 % des Spektrums verworfen):
  nachträgliche Kompression zerstört das Modell (0,970 → 0,450, minus 52
  Prozentpunkte), komprimiert trainiert bleibt es brauchbar (0,970 → 0,930).

**Interpretation:** Dasselbe Modell, derselbe Rang, dieselbe strukturelle
Einschränkung -- aber ob man sie *nachträglich aufzwingt* oder *von Anfang an
mittrainiert*, macht rund den Faktor 12 im Genauigkeitsverlust aus. Das Netz
umgeht die Einschränkung tatsächlich, indem es innerhalb der eingeschränkten
Familie eine andere Lösung sucht. Damit ist auch geklärt, warum das
MPS-Trunkierungskriterium den Rangbedarf so massiv überschätzt hat: Es
beschreibt Regime A, angewandt wurde es auf Regime B.

Nebenbefund: In 5 von 12 Fällen ist das komprimiert trainierte Modell sogar
*besser* als das dichte (negative Verluste) -- die Tensor-Train-Struktur wirkt
offenbar regularisierend.

### Gilt der Regime-Effekt allgemein? (2026-08-02)

`qire_ai/regime_generalization.py`: derselbe Test für drei verschiedene Arten,
Struktur aufzuzwingen -- jeweils auf dieselbe Schicht desselben Modells, damit
nur die *Art* der Beschränkung variiert. Vorab abgesichert: bei voller Stärke
reproduzieren alle drei Post-hoc-Projektionen das Original exakt.
27 Zellen (3 Datensätze × 3 Familien × 3 Stärken), 5 Seeds.

| Beschränkung | post-hoc | mittrainiert | Verhältnis | besser in | Wilcoxon p |
|---|---|---|---|---|---|
| width (Neuronen entfernen) | 0,1119 | 0,0064 | **17,4×** | 9/9 | 0,0039 |
| tensor_train | 0,1184 | 0,0101 | **11,8×** | 9/9 | 0,0039 |
| low_rank | 0,0981 | 0,0352 | 2,8× | 7/9 | 0,164 (n.s.) |
| **gesamt** | **0,1095** | **0,0172** | **6,4×** | **25/27** | **0,000005** |

**Der Effekt ist also nicht Tensor-Train-spezifisch**, sondern ein allgemeines
Verhalten beim Aufzwingen von Struktur. Damit ist der TT-Befund ein Sonderfall
eines breiteren Prinzips.

**Die Ausnahme verfeinert das Prinzip:** Warum ist die Lücke bei `low_rank`
klein und nicht signifikant? Die dortige Post-hoc-Projektion ist eine
SVD-Trunkierung -- nach dem **Eckart-Young-Theorem die beweisbar optimale**
Rang-r-Näherung. Bei den anderen ist sie es nicht: Neuronen-Pruning nutzt eine
Magnitude-Heuristik, TT-SVD arbeitet sequentiell-gierig. Die Reihenfolge passt
exakt invers:

| Güte der nachträglichen Projektion | Lücke |
|---|---|
| optimal (Eckart-Young) → low_rank | 2,8× |
| quasi-optimal (gierig) → tensor_train | 11,8× |
| Heuristik → width | 17,4× |

Das heißt: Die Lücke entsteht **nicht nur** daraus, dass Training sich an die
Beschränkung anpasst -- ein erheblicher Teil ist schlicht, dass übliche
nachträgliche Projektionsverfahren suboptimal sind.

Vorbehalt: Bei `low_rank` war auch das mittrainierte Modell schlechter
(0,0352 vs. 0,0064/0,0101) -- der harte Flaschenhals scheint schwerer
trainierbar, die Lücke wird also von beiden Seiten zusammengedrückt. Und
n=3 Familien ist zu wenig für einen Beweis. Die Übereinstimmung der
Reihenfolge ist aber auffällig genug, um sie als Hypothese zu formulieren.

Zwei weitere klare Muster:
- Die Lücke wächst mit der Kompressionsstärke (Spearman −0,549, p=0,003)
- Sie hängt stark von der Aufgabenschwierigkeit ab: digits 0,224 --
  wine 0,041 -- breast_cancer 0,012

**Praktische Folgerung für das Projektziel:** Wer ein Modell
ressourceneffizient machen will, sollte die Beschränkung von Anfang an
mittrainieren statt nachträglich zu komprimieren -- das ist bei starker
Kompression der Unterschied zwischen brauchbar und unbrauchbar. Und wenn
nachträglich komprimiert werden muss, lohnt sich ein beweisbar optimales
Projektionsverfahren gegenüber einer Heuristik.

### Regularisierungseffekt: widerlegt (2026-08-02)

In mehreren früheren Zellen war das komprimierte Modell *besser* als das
dichte (negative Verluste). Vermutung: Die Strukturbeschränkung wirkt als
Regularisierer. `qire_ai/regularization_effect.py` prüft das.

Wichtig am Aufbau: Die getesteten Konfigurationen wurden anhand der alten
Daten ausgewählt. Sie auf denselben Seeds nachzutesten wäre zirkulär gewesen
-- genau das Rauschen, das sie auffällig machte, hätte den Effekt reproduziert.
Deshalb frische Seeds 101-125, die kein früheres Experiment berührt hat.

**Frage 1 -- repliziert der Vorteil? Nein.**

| Datensatz | Konfiguration | Nutzen | Siege | p |
|---|---|---|---|---|
| wine | tensor_train_r8 | +0,0032 | 7/25 | 0,298 |
| wine | width_16 | **−0,0087** | 1/25 | **0,008** |
| breast_cancer | tensor_train_r8 | +0,0038 | 14/25 | 0,076 |
| breast_cancer | width_16 | +0,0023 | 8/25 | 0,377 |
| iris | tensor_train_r8 | −0,0036 | 4/25 | 0,857 |
| iris | width_16 | **−0,0062** | 2/25 | **0,039** |
| digits | tensor_train_r8 | **−0,0099** | 3/25 | **0,000** |
| digits | width_16 | **−0,0080** | 5/25 | **0,000** |

Von 8 Zellen sind 4 signifikant -- und **alle vier zeigen das komprimierte
Modell als schlechter**. Keine einzige Zelle zeigt einen signifikanten
Vorteil. `wine`/`width_16` war in den alten 5-Seed-Daten noch besser als
dicht, ist jetzt signifikant schlechter. Der beste verbleibende Kandidat
(`breast_cancer`/`tensor_train_r8`, p=0,076) verfehlt die Signifikanz.

**Damit waren die früheren "Siege" Rauschen.** Genau dafür war der Aufbau mit
frischen Seeds gedacht, und er hat funktioniert.

**Frage 2 -- Fehler in meiner eigenen Auswertung.** Ich hatte die
Seed-Ebenen-Korrelation zwischen Überanpassung und Nutzen als Mechanismus-Test
gerechnet (Spearman +0,170, p=0,016 -- scheinbar signifikant). Das ist ein
Artefakt der Konstruktion: `dense_overfit = dense_train − dense_test` und
`benefit = model_test − dense_test` enthalten **beide** den Term
`−dense_test`. Ein Seed mit zufällig niedrigem `dense_test` punktet dadurch
automatisch auf beiden Größen.

Simulation mit vollständig unabhängigen Modellen (wahrer Zusammenhang = 0):

| Streuung von dense_train | Spearman (Null) |
|---|---|
| 0,000 | +0,689 |
| 0,005 | +0,665 |
| 0,010 | +0,615 |
| 0,020 | +0,480 |

Die Nullverteilung liegt bei +0,5 bis +0,7. Mein gemessener Wert von +0,170
liegt **darunter** -- also eher ein Argument *gegen* die Hypothese.

Der ehrliche Test ist die Datensatz-Ebene (dort mittelt sich der gemeinsame
Term heraus):

| Datensatz | Überanpassung | Nutzen |
|---|---|---|
| wine | 0,0209 | −0,0028 |
| breast_cancer | 0,0293 | +0,0031 |
| iris | 0,0251 | −0,0049 |
| digits | **0,0313** | **−0,0090** |

`digits` hat die höchste Überanpassung und den schlechtesten Nutzen -- das
genaue Gegenteil der Regularisierungs-Vorhersage.

**Fazit:** Der Regularisierungseffekt existiert nicht. Die
Strukturbeschränkung kostet Genauigkeit, sie schenkt keine. Warnhinweis
direkt im Code ergänzt, damit die artefaktbehaftete Korrelation später nicht
doch als Beleg gelesen wird.

### Strukturiertes Pruning: erste echte Zeitersparnis (2026-08-02)

`qire_ai/structured_pruning.py`. Ganze Neuronen entfernen macht die Matrizen
*wirklich* kleiner -- kein Maskieren, keine Rekonstruktion, kein zusätzlicher
Verwaltungsaufwand. Damit der einzige bisher untersuchte Ansatz, bei dem die
Ersparnis strukturell in der Laufzeit ankommen *muss*.

**Teil A -- Nachtrainieren schließt die Regime-Lücke.** Drei Arme, 15 Seeds
(frische Seeds 201-215), hidden 32 -> 16:

| | Verlust ggü. dicht |
|---|---|
| gepruned, ohne Nachtrainieren | 0,0690 |
| gepruned, mit Nachtrainieren (20 Ep.) | **0,0061** |
| schmal von Anfang an trainiert | 0,0048 |

- mit vs. ohne Nachtrainieren: p=0,00001
- mit Nachtrainieren vs. von-Anfang-an: **p=0,34, kein nachweisbarer Unterschied**

**Das schränkt das Regime-Prinzip ein und muss so in die Arbeit.** Die
17,4-fache Lücke bei `width` in `regime_generalization.py` wurde *ohne*
Nachtrainieren gemessen. Korrigierte Fassung:

> Struktur nachträglich aufzuzwingen kostet deutlich mehr Genauigkeit als sie
> mitzutrainieren -- **es sei denn, man trainiert danach kurz nach**. Dann
> verschwindet der Unterschied.

Praktische Anmerkung: Wenn die Zielgröße vorab bekannt ist, bleibt
schmal-von-Anfang-an günstiger (80 Epochen klein statt 80 groß + 20 Nachlauf).
Der Wert von prune+fine-tune liegt darin, aus *einem* Training mehrere
Zielgrößen ableiten zu können.

Auf `digits` bleibt auch mit Nachtrainieren ein signifikanter Rest
(−0,0148, p=0,0001) -- bei der schwersten Aufgabe reicht der kurze Nachlauf nicht.

**Teil B -- die Ersparnis kommt endlich in gemessener Zeit an.**
Einfädig gemessen, Neuronenzahl halbiert (Ideal wäre 2,00×):

| Neuronen | Batch 32 | Batch 1000 | Batch 10000 |
|---|---|---|---|
| 32 | 1,07× | 1,66× | 1,14× |
| 128 | 1,29× | 2,81×* | 1,74× |
| 512 | 1,67× | 1,89× | 1,78× |
| 2048 | 1,73× | **2,13×** | **2,06×** |

*Der Wert über dem theoretischen Ideal deutet auf einen Cache-Effekt hin und
sollte nicht als Recheneinsparung gelesen werden.

**Erster Ansatz im gesamten Projekt, bei dem eine Ressourcenersparnis
tatsächlich in Wall-Clock-Zeit ankommt.** Ab ca. 512 Neuronen 1,8-1,9×, bei
2048 das volle Ideal. Bei der tatsächlichen Projektgröße (32 Neuronen) nur
1,07-1,66× -- dasselbe Overhead-Muster wie überall, aber diesmal mit klar
bezifferter Grenze, ab der es funktioniert.

Damit ist auch die wiederkehrende Frage des Projekts quantitativ beantwortet:
**Ressourcenersparnis wird ab etwa 512 Neuronen zu echter Zeitersparnis**,
darunter frisst der Aufruf-Overhead sie auf.

### Skalierung: die Befunde oberhalb der Schwelle (2026-08-02)

Alle bisherigen Experimente lagen bei 32 Neuronen und 13-80 Merkmalen -- also
*unterhalb* der selbst gemessenen 512-Neuronen-Schwelle, ab der Ersparnis
überhaupt in Zeit ankommt. `qire_ai/scale_up.py` wiederholt die Kernbefunde auf
MNIST (784 Merkmale, echte Bilddaten) bei 512 und 2048 Neuronen.

Vorarbeit: MNIST über `fetch_openml` eingebunden, lokal zwischengespeichert.
Dabei ein Skalierungsproblem gefunden und behoben -- dazu unten mehr.

| Neuronen | Arm | Genauigkeit | Verlust | Parameter | Anteil | Zeit | Speedup |
|---|---|---|---|---|---|---|---|
| 512 | dicht | 0,9639 | — | 407.050 | 1,000 | 10,46 ms | 1,00× |
| 512 | tensor_train r16 | 0,9558 | +0,008 | **26.506** | **0,065** | 12,37 ms | 0,85× |
| 512 | tensor_train r32 | 0,9473 | +0,017 | 84.234 | 0,207 | 13,01 ms | 0,80× |
| 512 | pruned_half+ft | 0,9652 | **−0,001** | 203.530 | 0,500 | 5,52 ms | **1,89×** |
| 2048 | dicht | 0,9681 | — | 1.628.170 | 1,000 | 45,87 ms | 1,00× |
| 2048 | tensor_train r16 | 0,9544 | +0,014 | **56.442** | **0,035** | 56,05 ms | 0,82× |
| 2048 | tensor_train r32 | 0,9430 | +0,025 | 150.250 | 0,092 | 61,58 ms | 0,74× |
| 2048 | pruned_half+ft | 0,9648 | +0,003 | 814.090 | 0,500 | 23,56 ms | **1,95×** |

**Befund 1 bestätigt und deutlich verstärkt.** Tensor-Train skaliert genau wie
die Theorie vorhersagt: bei den kleinen Modellen 37-68 % Parameterersparnis,
bei 2048 Neuronen **96,5 %** (28-fache Kompression) für 1,4 Prozentpunkte
Genauigkeit. Das ist echte, große Ressourcenersparnis durch einen
quanteninspirierten Mechanismus, belegt auf realen Bilddaten.

**Aber Parameter- und Zeitersparnis bleiben entkoppelt.** Tensor-Train ist bei
*keiner* Konfiguration schneller (0,74-0,85×), und der Rückstand wächst mit der
Modellgröße, weil die Matrixrekonstruktion pro Vorwärtslauf mitwächst. Der
Ausweg wäre eine Streaming-Kontraktion, die die Kerne direkt auf die Eingabe
anwendet -- jetzt erstmals lohnenswert, weil die Modelle groß genug sind.

**Befund 3 bestätigt.** Strukturiertes Pruning erreicht 1,89× und 1,95× echte
Beschleunigung bei praktisch unverändertem Ergebnis (bei 512 sogar minimal
besser als dicht). Konsistent mit den 1,8-2,1× aus `structured_pruning.py`.

**Damit hat das Projekt eine saubere zweiteilige Antwort:** Der
quanteninspirierte Ansatz (Tensor-Train) liefert herausragende
Speicher-/Parameterersparnis, die mit der Modellgröße besser wird. Echte
Zeitersparnis liefert bisher nur ein klassisches Verfahren (strukturiertes
Pruning). Die beiden sind komplementär, nicht konkurrierend -- Tensor-Train
für speicherbegrenzte Umgebungen, Pruning für Latenz.

**Ungeklärte Auffälligkeit:** Rang 32 ist auf beiden Achsen schlechter als
Rang 16 -- mehr Parameter *und* weniger Genauigkeit (0,9430 vs. 0,9544 bei
2048). Mehr Kapazität sollte nicht schaden. Vermutlich ein Optimierungsproblem
bei höherem Rang, aber ungeprüft. Gehört als offene Frage in die Arbeit.

### Skalierungsproblem bei MNIST gefunden (2026-08-02)

Beim Einbinden von MNIST fiel auf: Die Pipeline standardisiert jedes Merkmal
durch seine eigene Standardabweichung. Bei Bilddaten sind viele Randpixel
nahezu konstant -- MNISTs ruhigstes Pixel hat std 0,000119, wird also um
Faktor **8368** aufgeblasen, auf über 200 Standardabweichungen.

Gemessene Auswirkung (784×512-Netz, 3 Seeds):

| Skalierung | Genauigkeit | max \|x\| |
|---|---|---|
| nur /255 (keine Standardisierung) | **0,9518** | 1,00 |
| global skaliert | 0,9471 | 2,24 |
| Varianz-Floor 1 % | 0,9391 | 83,67 |
| StandardScaler (bisheriges Verhalten) | 0,9311 | 221 |

Wichtig: **Ein globaler Fix wäre falsch gewesen.** Auf `breast_cancer` *kostet*
ein Varianz-Floor 0,0198 Genauigkeit (p=0,002) -- die Merkmale mit kleiner
Varianz tragen dort echtes Signal. Auf `digits` ändert er nichts (p=0,31).
Deshalb ist die Standardisierung jetzt pro Datensatz abschaltbar
(`PRESCALED_DATASETS`), Standardverhalten unverändert. Mit Regressionstest
abgesichert: wine, breast_cancer, digits, iris und synthetic_large liefern
bit-genau dieselben Tensoren wie vorher, **alle früheren Ergebnisse bleiben
gültig.**

### Streaming-Kontraktion: Tensor-Train wird erstmals schneller (2026-08-02)

`TTLinear.forward_streaming()` implementiert: die Kerne werden direkt gegen die
Eingabe kontrahiert, ohne je die volle Matrix zu bilden. Numerisch identisch zur
Rekonstruktion verifiziert (max. Abweichung 1e-15 über 6 Konfigurationen),
Gradienten fließen.

**Meine Kostenvorhersage war falsch -- und zwar aus dem Kernthema des Projekts
heraus.** Aus der FLOP-Zählung hatte ich abgeleitet: Streaming/dicht ≈
d·r²·q^(1−d), also gewinnen viele Kerne und kleiner Rang. Gemessen bei 784×512:

| Kerne | Rang | Vorhersage | gemessen |
|---|---|---|---|
| 3 | 2 | 0,16× | 2,60× |
| 5 | 2 | 0,10× | 8,83× |
| 6 | 2 | 0,09× | 9,15× |
| 3 | 16 | 10,30× | 9,07× |

Korrelation Vorhersage/Messung: Spearman ρ=+0,42, p=0,10 -- **nicht einmal
signifikant.** Grund: Die Anzahl sequentieller `tensordot`-Aufrufe entspricht der
Kernanzahl. Mehr Kerne senken die FLOPs, erhöhen aber die Zahl kleiner, schlecht
ausgelasteter Operationen. Das ist exakt derselbe Fehler, den dieses Projekt
schon dreimal an `estimated_ops` nachgewiesen hat -- diesmal im eigenen
Kostenmodell.

**Der Crossover existiert, liegt aber bei der Modellgröße, nicht bei der
Faktorisierung.** Quadratische Schichten, 4 Kerne, Rang 4, Batch 256:

| Schicht | dicht | Rekonstruktion | Streaming | Streaming/dicht |
|---|---|---|---|---|
| 512² | 1,47 ms | 3,28 ms | 11,35 ms | 7,71× |
| 1024² | 6,55 ms | 13,33 ms | 23,28 ms | 3,55× |
| 2048² | 27,36 ms | 53,20 ms | 53,57 ms | 1,96× |
| **4096²** | 111,38 ms | 200,73 ms | **72,35 ms** | **0,65× (1,54× schneller)** |

**Zum ersten Mal im gesamten Projekt ist Tensor-Train schneller als dicht** --
bei 4096×4096 um Faktor 1,54, und das bei etwa **0,04 % der Parameter**
(2.560 statt 16,7 Mio.).

Strukturelle Einsicht nebenbei: **Die Rekonstruktion kann prinzipiell nie
gewinnen**, weil sie die dichte Multiplikation *plus* den Matrixaufbau
ausführt. Die Messwerte (durchweg ~2× dicht) bestätigen das. Nur Streaming
vermeidet die große Multiplikation überhaupt und ist damit der einzige Weg zu
echter Beschleunigung.

**Vorbehalt:** Die Genauigkeit bei 4096 Neuronen mit Rang 4 ist ungetestet --
gemessen ist bisher nur die Geschwindigkeit. Ob so wenig Kapazität dort
ausreicht, ist offen und muss vor jeder Behauptung geprüft werden.

**Damit ergibt sich ein gestuftes Gesamtbild mit bezifferten Schwellen:**

| Modellgröße | was Zeit spart |
|---|---|
| unter ~512 Neuronen | nichts |
| ~512 bis ~3000 | strukturiertes Pruning (1,9×) |
| ab ~4096 | zusätzlich Tensor-Train-Streaming (1,54×) bei 0,04 % der Parameter |

### Genauigkeit an der Gewinn-Konfiguration geprüft (2026-08-02)

`qire_ai/tt_large_accuracy.py`. Der 1,54×-Speedup war reine Zeitmessung -- ob
eine Rang-4-Kette bei Breite 4096 überhaupt genug Kapazität hat, war offen.

**Vorab ein formabhängiger Befund:** Der Vorteil hängt nicht nur an der Größe,
sondern an der *Form* der Schicht:

| Schicht | Rang | Streaming/dicht |
|---|---|---|
| 784→4096 (MNIST-Eingangsschicht) | 4 | 1,89× langsamer |
| **4096→4096 (quadratisch)** | **4** | **0,72× (1,39× schneller)** |
| 4096→4096 | 16 | 1,85× langsamer |

Damit wäre es falsch gewesen, den Quadrat-Speedup mit einer MNIST-Genauigkeit
bei 784→4096 zu kombinieren. Stattdessen ein Netz mit quadratischer
Zwischenschicht (784 → 4096 → 4096 → 10), bei dem die komprimierte Schicht
exakt die vermessene Form hat. Genauigkeit und Geschwindigkeit stammen so aus
derselben Konfiguration.

**Genauigkeit (3 Seeds, MNIST, quadratische Zwischenschicht):**

| Arm | Genauigkeit | Std | Parameter Zwischenschicht | Anteil |
|---|---|---|---|---|
| dicht | 0,9202 | 0,0019 | 16.781.312 | 1,00000 |
| **TT Rang 4** | 0,9267 | 0,0063 | **6.656** | **0,00040** |
| TT Rang 16 | 0,9266 | 0,0050 | 38.912 | 0,00232 |

**Rang 4 hat genug Kapazität: 2521-fache Kompression ohne messbaren
Genauigkeitsverlust.**

**Ausdrücklich nicht behauptet:** dass TT *besser* ist. Die Differenz von
+0,0065 liegt bei einer Streuung von 0,0063 über nur 3 Seeds -- etwa eine
Standardabweichung. Exakt dieses Muster ist beim vermeintlichen
Regularisierungseffekt auf frischen Seeds vollständig zusammengebrochen
(siehe oben). Belastbar ist "kein messbarer Verlust", nicht "besser". Für eine
Überlegenheitsaussage bräuchte es ≥15 frische Seeds und einen gepaarten Test.

**Messstreuung offen benannt:** Für dieselbe Konfiguration wurde vorher 0,48×
(2,08× schneller) gemessen, jetzt 0,72× (1,39×). Korrekt ist die Spanne
**1,4× bis 2,1×**, nicht der günstigere Einzelwert.

**Damit gewinnt der quanteninspirierte Ansatz erstmals auf beiden Achsen
gleichzeitig** -- Speicher *und* Zeit -- unter klar benannten Bedingungen:
quadratische Schicht ab etwa 4096, Rang 4, Batch-Inferenz. Rang 16 gewinnt
nur beim Speicher, nicht bei der Zeit.

Weitere Vorbehalte: Trainingsbudget war klein (6000 Beispiele, 10 Epochen), die
absolute Genauigkeit von 0,92 liegt deutlich unter dem, was MNIST hergibt. Der
Vergleich zwischen den Armen ist davon nicht betroffen, die absolute Zahl aber
nicht als MNIST-Ergebnis zitierbar.

### Konfigurationsraum richtig ausgemessen: Faktor 17,8 statt 1,4 (2026-08-02)

`qire_ai/tt_frontier.py`. Zwei Fehler in der bisherigen Vorgehensweise behoben:

**Fehler 1: Die (Kerne, Rang)-Landkarte lag im falschen Bereich.** Sie wurde bei
784×512 erstellt -- einer Form, bei der *keine* Konfiguration gewinnt. Daraus
"4 Kerne, Rang 4 ist optimal" zu schließen, war nicht haltbar. Neu vermessen bei
4096×4096, wo Streaming tatsächlich gewinnt:

| Kerne | Rang | Parameter | Anteil | vs. dicht |
|---|---|---|---|---|
| **2** | **2** | 20.480 | 0,122 % | **5,44× schneller** |
| 2 | 4 | 36.864 | 0,220 % | 3,31× schneller |
| 3 | 2 | 6.144 | 0,037 % | 3,18× schneller |
| 4 | 4 (bisherige Wahl) | 6.656 | 0,040 % | 1,72× schneller |
| 6 | 2 | 4.416 | 0,026 % | 1,19× schneller |

14 von 20 Konfigurationen schlagen dicht. **Die bisherige Wahl war dreifach
suboptimal.** Und wieder gilt das Gegenteil meines FLOP-Modells: *weniger*
Kerne sind schneller, nicht mehr -- zum dritten Mal in diesem Projekt sagt die
Operationszählung das Falsche vorher.

Es gibt eine echte Pareto-Front: maximale Kompression bei 6 Kernen/Rang 2
(0,026 %, 1,19×), maximale Geschwindigkeit bei 2 Kernen/Rang 2
(0,122 %, 5,44×).

**Fehler 2: Alle Messungen liefen bei Batch 256.** Das verzerrt stark, denn
Rekonstruktion baut die Matrix einmal pro Aufruf und amortisiert sie über den
Batch, während Streaming linear mitskaliert. Bei Batch 1 -- Echtzeit-Inferenz
auf schwacher Hardware, also dem eigentlichen Ressourcen-Anwendungsfall --
kehrt sich das Bild dramatisch:

Beste Konfiguration (2 Kerne, Rang 2) über Batch-Größen:

| Batch | dicht | Streaming | Faktor |
|---|---|---|---|
| 1 | 3,57 ms | 0,23 ms | **15,8×** |
| 8 | 14,03 ms | 0,79 ms | **17,8×** |
| 64 | 36,69 ms | 5,60 ms | 6,6× |
| 256 | 120,47 ms | 24,15 ms | 5,0× |
| 1024 | 456,82 ms | 103,88 ms | 4,4× |

**Statt der bisher berichteten 1,4-2,1× sind es bis zu 17,8×** -- bei 0,122 %
der Parameter (819-fache Kompression). Die Rekonstruktionsvariante ist bei
Batch 1 dagegen 25,9× *langsamer* als dicht, weil sie für ein einziges
Beispiel die komplette 16,7-Mio.-Matrix aufbaut.

**Offener Punkt, zwingend vor jeder Behauptung:** Die Genauigkeit wurde bei
4 Kernen/Rang 4 gemessen, nicht bei 2/2. Die neue Konfiguration hat zwar *mehr*
Parameter (20.480 gegenüber 6.656), was für ausreichende Kapazität spricht,
aber gemessen ist das nicht. Ohne diesen Test ist die 17,8× keine
verwertbare Aussage.

### Kernaussage abgesichert: 10 frische Seeds (2026-08-02)

Die Gewinner-Konfigurationen wurden aus Geschwindigkeitsmessungen ausgewählt,
die Genauigkeit vorher nur mit 3 Seeds geprüft. Jetzt mit 10 frischen Seeds
(501-510, disjunkt zu allen früheren) und gepaartem Wilcoxon-Test, quadratische
4096×4096-Zwischenschicht auf MNIST:

| Arm | Genauigkeit | Std | vs. dicht | Parameter | Anteil | Siege | p |
|---|---|---|---|---|---|---|---|
| dicht | 0,9238 | 0,0092 | — | 16.781.312 | 1,00000 | — | — |
| **tt 2 Kerne, Rang 2** | 0,9205 | 0,0164 | −0,0033 | 20.480 | 0,00122 | 5/10 | 0,557 |
| tt 3 Kerne, Rang 2 | 0,9277 | 0,0079 | +0,0039 | 6.144 | 0,00037 | 7/10 | 0,432 |
| tt 4 Kerne, Rang 4 | 0,9291 | 0,0053 | +0,0053 | 6.656 | 0,00040 | 5/10 | 0,232 |

**Keine einzige Differenz ist signifikant** (alle p ≥ 0,23). Das ist genau die
gesuchte Aussage: Kompression ohne messbaren Genauigkeitsverlust.

**Die Vorsicht war berechtigt.** Der bei n=3 beobachtete Vorteil von +0,0065
schrumpft bei n=10 auf +0,0053 mit p=0,232 -- also kein Vorteil, sondern
Rauschen, wie schon beim Regularisierungseffekt. Hätte ich damals "besser als
dicht" behauptet, wäre es hier gekippt.

Streuungsprüfung (F-Test): Die schnellste Konfiguration (2/2) streut mit
std=0,0164 stärker über Seeds als dicht (0,0092), F=3,18, p=0,100 -- knapp
nicht signifikant, aber ein Hinweis auf geringere Stabilität. 3/2 und 4/4 sind
dagegen eher *stabiler* als dicht.

**Damit steht die Hauptaussage des Projekts:**

> ⚠ **ZURÜCKGEZOGEN — diese Aussage gilt nicht mehr.** Sie stand hier vom
> 2. August bis zum Nachtrag am 4. August unmarkiert als „die Hauptaussage des
> Projekts". Widerrufen in zwei Schritten: A-002 (weiter unten in dieser Datei)
> zeigt, dass eine simple Blockdiagonale Tensor-Train bei gleichem Budget
> schlägt (p=0,004); B-003/B-004 (nur im `PROJEKTTAGEBUCH.md`) zeigen, dass die
> 819×-Aussage nur in einer Architektur gilt, deren Blindheit gemessen ist
> (Positivkontrolle p=0,23) — ohne die dichten Enden dreht das Vorzeichen.
> Die Geschwindigkeitsaussage hatte zusätzlich die falsche Vergleichsgruppe:
> Low-Rank erreicht am selben Budget rund 120× gegenüber dicht.
> **Der gültige Stand steht in `PROJEKTTAGEBUCH.md`, Abschnitt „Gesamtbilanz".**

> Eine quanteninspirierte Tensor-Train-Schicht ersetzt eine dichte
> 4096×4096-Schicht mit **0,12 % der Parameter** (819-fache Kompression) bei
> **bis zu 17,8× schnellerer Inferenz** (Batch 8) und **keinem statistisch
> messbaren Genauigkeitsverlust** (p=0,56, n=10).

**Praktische Empfehlung aus den Daten:** 3 Kerne/Rang 2 ist womöglich die
bessere Wahl als 2/2 -- nur 0,037 % der Parameter (2731-fach), 3,18× schneller
bei Batch 256, und über Seeds *stabiler* als dicht. Die 17,8× bei Batch 8 wurde
allerdings nur für 2/2 gemessen; die Kleinbatch-Messung für 3/2 fehlt noch.

### Kleinbatch-Messung vervollständigt (2026-08-02)

Median aus 5 Messreihen × 200 Wiederholungen, 4096×4096, einfädig:

| Batch | dicht | 2 Kerne/R2 | 3 Kerne/R2 |
|---|---|---|---|
| 1 | 3,47 ms | 15,3× | 13,9× |
| 2 | 5,75 ms | 18,7× | 14,7× |
| 4 | 14,38 ms | (30,1×) | (21,9×) |
| 8 | 14,27 ms | 18,8× | 12,8× |
| 16 | 17,30 ms | 12,5× | 8,0× |
| 256 | 112,60 ms | 5,0× | 3,2× |

**Der Spitzenwert bei Batch 4 ist nicht verwertbar.** Die dichte Referenz
braucht bei Batch 4 und 8 exakt gleich lang (14,38 vs. 14,27 ms), obwohl
Batch 2 nur 5,75 ms kostet -- reproduzierbar über 5 Messreihen. Das ist ein
Kernel-Wechsel in BLAS (Batch 4 wird faktisch wie Batch 8 gerechnet), also
eine Ineffizienz der *Referenz*, nicht eine Stärke von Tensor-Train. Als
Kennzahl taugen deshalb nur die Werte bei Batch 1, 2, 8 und darüber.

**Belastbare Aussage: 13-19× bei kleinen Batches, 3-5× bei Batch 256.**

**Empfehlung aus den Daten: 3 Kerne / Rang 2.** Nur 0,037 % der Parameter
(2731-fache Kompression) statt 0,122 % bei 2/2, dabei 13-15× schneller bei
kleinen Batches, und über Seeds *stabiler* als dicht (F=0,74 gegenüber F=3,18
bei 2/2). Die zusätzliche Geschwindigkeit von 2/2 kostet 3,3× mehr Parameter
und mehr Streuung -- für den praktischen Einsatz kein guter Tausch.

## Forschungsprotokoll A-001 — Abgrenzungstest gegen Low-Rank (2026-08-02)

**Forschungsfrage:** Schlägt die Tensor-Netzwerk-Struktur bei gleichem
Parameterbudget eine gewöhnliche Low-Rank-Faktorisierung? Bis hierher wurde TT
nur gegen *dicht* verglichen -- das kann "Tensor-Netzwerke helfen" nicht von
"irgendeine Low-Rank-Kompression hilft" trennen. Ohne diese Baseline war die
Kernaussage nicht abgesichert.

**Quantenprinzip:** bewusst keines. Dies ist die nicht-quanteninspirierte
Kontrollbedingung, die die Behauptung überstehen muss.

**Aufbau:** MNIST, quadratische 4096×4096-Zwischenschicht, 10 Seeds (601-610),
gepaarter Wilcoxon-Test. Wo Budgets nicht exakt angleichbar waren, bekam
Low-Rank das *größere* -- konservativ gegen TT.

| Verfahren | Genauigkeit | Std | Parameter | b=1 | b=256 |
|---|---|---|---|---|---|
| dicht | 0,9243 | 0,0123 | 16.781.312 | 4,411 ms | 130,342 ms |
| TT 3 Kerne/R2 | 0,9279 | 0,0102 | 6.144 | 0,325 ms | 43,531 ms |
| TT 2 Kerne/R2 | 0,9241 | 0,0148 | 20.480 | 0,254 ms | 25,855 ms |
| Low-Rank r=1 | 0,6827 | 0,0344 | 12.288 | 0,038 ms | 0,353 ms |
| Low-Rank r=2 | 0,8848 | 0,0122 | 20.480 | 0,039 ms | 1,046 ms |
| Low-Rank r=5 | 0,9205 | 0,0063 | 45.056 | 0,034 ms | 1,177 ms |

**Entscheidende Vergleiche:**

| Vergleich | Differenz | p | Budget |
|---|---|---|---|
| TT 3K/R2 (6.144) vs. LR r1 (12.288) | **+0,2452** | **0,002** | TT halb so viele Parameter |
| TT 2K/R2 (20.480) vs. LR r2 (20.480) | **+0,0393** | **0,002** | exakt gleich |
| TT 2K/R2 (20.480) vs. LR r5 (45.056) | +0,0036 | 0,557 | TT 2,2× weniger |

**Ergebnis: Falsifikationskriterium nicht eingetreten.** Bei exakt identischem
Budget ist TT signifikant genauer. Um TTs Genauigkeit zu erreichen, braucht
Low-Rank 2,2× mehr Parameter. Bei TTs kleinstem Budget bricht Low-Rank
vollständig ein (0,68 vs. 0,93).

**Interpretation:** Der Vorteil stammt aus dem Strukturmerkmal, um das es bei
MPS geht -- Low-Rank erzwingt einen *einzelnen* globalen Flaschenhals, TT
verteilt die Beschränkung auf eine *Kette von Bonds* und kann so
Korrelationsstrukturen darstellen, die ein einzelner Rang-r-Schnitt nicht
erfasst. Damit ist der Quantum-Inspiration-Check inklusive Frage 6
(Abgrenzung) bestanden: Es ist nicht "Low-Rank mit Physik-Vokabular".

**Notwendige Korrektur der bisherigen Darstellung:** Auf der *Zeitachse* ist TT
unterlegen -- Low-Rank ist bei gleichem Budget 24× schneller (1,046 ms
gegenüber 25,855 ms bei Batch 256). Die frühere Kennzahl "13-15× schneller als
dicht" hatte die falsche Vergleichsgruppe; Low-Rank erreicht am selben Budget
etwa 120× gegenüber dicht. Es gibt keine gemeinsame Gewinnerachse:

| Ziel | Verfahren |
|---|---|
| minimale Parameter / Speicher | **Tensor-Train** (signifikant besser pro Parameter) |
| minimale Latenz | **Low-Rank** (24× schneller bei gleichem Budget) |

**Entscheidung:** fortsetzen. Die Kernaussage wird auf Parametereffizienz
umgestellt, die Geschwindigkeitsaussage fällt weg bzw. wird als Nachteil
gegenüber Low-Rank ausgewiesen.

**Nächste Forschungsfrage:** Ist TTs Vorteil pro Parameter auch dann noch da,
wenn Low-Rank mehrfach hintereinandergeschaltet wird (mehrere kleine
Flaschenhälse statt einem)? Das wäre die härtere Kontrollbedingung, weil sie
die Ketteneigenschaft ohne Tensorstruktur nachbaut.

## Forschungsprotokoll A-002 — Lokalität vs. Bonds (2026-08-02)

**Forschungsfrage:** A-001 zeigte, dass TT eine *einzelne* Low-Rank-Engstelle
schlägt. Darin waren aber zwei Eigenschaften vermischt: TT ist *lokal* (jeder
Kern wirkt auf einen Index) *und* trägt *Bonds* (Korrelation zwischen den
lokalen Faktoren). Welche der beiden trägt den Vorteil?

**Aufbau:** Kontrollen mit jeweils nur einer Eigenschaft, alle bei ~20.480
Parametern, 10 Seeds (701-710). Block-Diagonal (1024 unabhängige 4×4-Blöcke,
verifiziert: Information kreuzt keine Blockgrenze) = Lokalität ohne Bonds.
Deep Bottleneck (4096→2→2→2→4096) = Verkettung ohne Lokalität.

| Arm | Struktur | Genauigkeit | Std | b=256 |
|---|---|---|---|---|
| dicht | voll | 0,9223 | 0,0074 | 119,6 ms |
| TT 2 Kerne/R2 | lokal + Bonds | 0,9228 | 0,0085 | 24,0 ms |
| Low-Rank r2 | global, ein Schnitt | 0,8739 | 0,0142 | 1,0 ms |
| Deep Bottleneck | global, verkettet | 0,8370 | 0,0836 | 1,0 ms |
| **Block-Diagonal** | **lokal, ohne Bonds** | **0,9397** | 0,0101 | 8,5 ms |

| TT gegen | Differenz | p |
|---|---|---|
| Low-Rank | +0,0490 | 0,002 (TT besser) |
| Deep Bottleneck | +0,0859 | 0,002 (TT besser) |
| **Block-Diagonal** | **−0,0168** | **0,004 (TT SCHLECHTER)** |

**Ergebnis: Falsifikationskriterium eingetreten.** Der Vorteil stammt aus der
**Lokalität**, nicht aus den Bonds. Block-Diagonal -- bei dem Information die
Blockgrenzen überhaupt nicht überquert -- ist bei identischem Budget
signifikant besser als TT und schlägt sogar die dichte Schicht (0,9397 vs.
0,9223) mit 0,12 % der Parameter. Zusätzlich ist es 2,8× schneller als TT
(8,5 ms vs. 24,0 ms). **TT ist auf beiden Achsen unterlegen.**

Die Bond-Struktur ist damit nicht neutral, sondern **kostet** bei gleichem
Budget Leistung: Parameter in Bonds sind schlechter angelegt als in lokaler
Struktur.

**Sauber ausgeschlossen:** Die Verkettungs-Erklärung. Deep Bottleneck ist mit
Abstand am schlechtesten (0,8370, Streuung 0,0836) -- mehrere globale
Flaschenhälse hintereinander sind schlechter als einer.

**Konsequenz für die Projektaussage:** Die Eigenschaft, die tatsächlich wirkt
(Lokalität durch Blockstruktur), ist ein klassisches, gut bekanntes Konzept,
verwandt mit gruppierten Faltungen. Der MPS-spezifische Teil -- die Bond-Kette,
also genau das Quanteninspirierte -- trägt hier nicht. Nach A-001 sah es so
aus, als hätte die Tensorstruktur ihren Existenzbeweis erbracht; A-002 zeigt,
dass dort nur die aussagekräftigste Kontrollbedingung fehlte.

**Entscheidung:** Die Kernaussage "Tensor-Train ist das effizienteste
Verfahren" ist nicht haltbar und muss zurückgezogen werden. Tensor-Train
bleibt besser als Low-Rank, ist aber Block-Diagonal unterlegen.

**Offene Fragen aus diesem Ergebnis:**
1. Gilt das auch, wenn die komprimierte Schicht die einzige Mischung leisten
   muss? Hier liegen dichte Schichten davor und dahinter, die global mischen --
   Block-Diagonal profitiert davon möglicherweise unfair.
2. Wie verhält sich Block-Diagonal bei verschiedenen Blockgrößen? Bei
   Blockgröße 1 wäre es eine Diagonalmatrix, bei 4096 eine dichte -- es gibt
   vermutlich ein Optimum dazwischen.
3. Lässt sich ein quanteninspiriertes Prinzip finden, das *Lokalität* statt
   Bonds operationalisiert? Dann wäre der Befund nicht das Ende der
   Quanteninspiration, sondern ihre Neuausrichtung.

## Forschungsprotokoll A-003 — Braucht Block-Diagonal dichte Nachbarn? (2026-08-02)

**Forschungsfrage:** A-002 könnte verzerrt gewesen sein: Block-Diagonal mischt
nie über Blockgrenzen, hatte aber dichte Schichten davor und dahinter, die
global mischen. Reitet es auf deren Mischleistung?

**Aufbau:** Dieselben Verfahren, aber zusätzlich mit **drei gestapelten**
komprimierten Schichten (statt einer), sodass die Mischung intern passieren
muss. 8 Seeds (801-808), gleiches Budget je Tiefe (61.440 bei depth 3).

| Arm | depth 1 | depth 3 | Änderung |
|---|---|---|---|
| TT 2 Kerne/R2 | 0,9241 ±0,0099 | 0,9225 ±0,0111 | −0,0016 |
| **Block-Diagonal** | 0,9381 ±0,0089 | **0,9309** ±0,0048 | −0,0072 |
| Low-Rank r2 | 0,8747 ±0,0138 | **0,1517** ±0,0565 | **−0,7230** |
| dicht | 0,9239 ±0,0065 | 0,9000 ±0,0105 | −0,0238 |

**Interaktionstest:** Block-Diagonal −0,0072 vs. TT −0,0016, Differenz −0,0056,
**p=0,383 — nicht signifikant.** Die Rettungshypothese H1 wird nicht gestützt.

**Direktvergleich bei depth 3:** Block-Diagonal 0,9309 vs. TT 0,9225,
**p=0,008 — Block-Diagonal weiterhin signifikant besser**, auch wenn die
Mischung intern erfolgen muss.

**Positivkontrolle belegt die Empfindlichkeit des Tests:** Low-Rank bricht beim
Stapeln von 0,8747 auf 0,1517 ein — praktisch Rateniveau (0,10 bei 10 Klassen).
Der Aufbau *konnte* also einen Zusammenbruch beim Stapeln detektieren; bei
Block-Diagonal trat er nicht ein. Damit ist dies ein Negativbefund mit
Positivkontrolle, nicht bloß ein Nullresultat aus mangelnder Sensitivität.

**Ergebnis: H0 bestätigt.** A-002 steht. Die Bond-Struktur ist nicht nötig.

### Gesamtbilanz des quanteninspirierten Kerns

Dreimal unabhängig getestet, dreimal widerlegt:

| Test | Mechanismus | Ergebnis |
|---|---|---|
| Gate-Ablation | Born-Regel-Gating | kein Effekt (p=0,25, 648 gematchte Paare) |
| Entropie-Analyse | Verschränkung sagt Kompressionserfolg vorher | Kontrolle widerlegt (trainiert ≈ zufällig) |
| A-002 / A-003 | Bond-Kette trägt globale Korrelation | Block-Diagonal besser (p=0,004 / p=0,008) |

Was tatsächlich wirkt: **Lokalität durch Blockstruktur** -- ein klassisches
Konzept (verwandt mit gruppierten Faltungen), mit 0,12 % der Parameter besser
als die dichte Schicht und 2,8× schneller als Tensor-Train.

**Entscheidung:** Der quanteninspirierte Mechanismus wird als *nicht belegt*
geführt. Das Projekt hat damit ein konsistentes, mehrfach abgesichertes
Negativergebnis zur Ausgangsthese -- und einen klassischen Positivbefund
(Blockstruktur), der nicht als quanteninspiriert ausgegeben werden darf.

**Verbleibende ehrliche Optionen:**
1. Projekt als sauber dokumentierte Widerlegung der Ausgangsthese führen, mit
   dem Blockstruktur-Befund als praktischem Nebenergebnis.
2. Nach einem quanteninspirierten Prinzip suchen, das *Lokalität* statt Bonds
   operationalisiert -- dann wäre der Befund eine Neuausrichtung statt eines
   Endes. Offen, ob es ein solches Prinzip mit echter Substanz gibt.
3. Ansatz B (DMRG-adaptive Bond-Dimension) oder C (Dichtematrix-gesteuerte
   Berechnung) aus der Bestandsaufnahme testen -- beide adressieren andere
   Ressourcenachsen (Trainingszeit bzw. Rechenaufwand pro Beispiel) und sind
   von A-001 bis A-003 nicht widerlegt.

## Forschungsprotokoll B-002 — Orthogonalität vs. Normkontrolle (2026-08-02)

**Frage:** B-001 zeigte, dass orthogonale Blöcke +0,0138 bringen (p=0,010).
Wirkt die Unitaritätseigenschaft, oder täte es jede Normkontrolle auch?

| Arm | Beschränkung | Genauigkeit | Std | vs. unbeschränkt |
|---|---|---|---|---|
| unbeschränkt | — | 0,9338 | 0,0089 | — |
| orthogonal | alle σ = 1 | 0,9475 | 0,0073 | +0,0138 (p=0,010) |
| spektral | größter σ = 1 | 0,9451 | 0,0051 | +0,0114 (p=0,010) |
| Frobenius | Energie fixiert | 0,9321 | 0,0066 | −0,0016 (n.s.) |
| weight decay | Adam 1e-4 | 0,1000 | 0,0000 | degeneriert, s.u. |

**Entscheidend: orthogonal vs. spektral = +0,0024, p=0,322 — nicht signifikant.**

Spektralnormalisierung zieht gleich. Der Effekt hängt am **Spektralradius**,
nicht an vollständiger Unitarität. Frobenius (gleiche Energie, freies Spektrum)
bringt nichts -- das stützt die Deutung sauber.

**Der weight-decay-Arm ist kein Befund:** exakt 0,1000 bei Streuung exakt
0,0000 über 10 Seeds = Rateniveau bei 10 Klassen. Vermutlich sterben bei drei
gestapelten Blockschichten mit ReLU die Aktivierungen ab. Vermutung, ungeprüft,
wird nicht als Beleg geführt.

**Einschränkung:** "nicht signifikant" ist kein Äquivalenznachweis. Bei n=10 und
+0,0024 könnte ein kleiner echter Unterschied unentdeckt bleiben. Die Beweislast
liegt aber bei der Quantenbehauptung, und sie wurde nicht erfüllt.

## Forschungsprotokoll C-001 — Dichtematrix-Routing: nicht testbar (2026-08-02)

**Ansatz:** Von-Neumann-Entropie einer Dichtematrix über K Kandidaten-
Repräsentationen steuert, wie viel pro Beispiel gerechnet wird. Die Entropie
kostet nur O(K³) statt O(d³), weil AᵀA (K×K) das Spektrum von ρ = AAᵀ teilt.

**Implementierung verifiziert:** Entropie exakt korrekt gegen vier analytisch
bekannte Fälle (identische Kandidaten S=0; zwei orthogonale S=log 2; vier
orthogonale S=log 4; dominantes Gewicht S≈0). Der Mechanismus unterscheidet
sich nachweislich von Gate-Konfidenz: bei identischen Kandidaten mit
p=(0,5, 0,5) sagt die Entropie *sicher* (S=0), die Konfidenz *unsicher*.

**Ergebnis: in dieser Umgebung nicht prüfbar.** Adaptive Berechnung setzt
voraus, dass der teure Pfad substantiell besser ist als der billige. Gemessen:

| Datensatz | n | billig | teuer | Lücke | max. Spielraum bei perfektem Routing |
|---|---|---|---|---|---|
| MNIST | 6.000 | 0,9000 | 0,8989 | −0,0011 | +0,0100 |
| Fashion-MNIST | 6.000 | 0,8017 | 0,7678 | −0,0339 | +0,0317 |
| Fashion-MNIST | 40.000 | 0,8692 | 0,8744 | +0,0052 | +0,0197 |
| MNIST | 40.000 | 0,9532 | 0,9648 | +0,0116 | +0,0066 |

Selbst perfektes Routing brächte höchstens +0,0066 bei einer Seed-Streuung von
0,005-0,010. Ein Unterschied zwischen Entropie- und Konfidenzsteuerung wäre
darin nicht nachweisbar.

**Klassifikation: vorläufig zurückgestellt wegen fehlender Testbarkeit** --
ausdrücklich *nicht* falsifiziert. Auf dieser Hardware und mit diesen
Datensätzen kauft zusätzliche Kapazität keine Genauigkeit; ein linearer Kopf
liegt bereits an der Datengrenze. Ein Durchlauf hätte nur Rauschen erzeugt.

Prüfbar wäre der Ansatz bei Aufgaben, wo Kapazität nachweislich zählt (große
Bilddatensätze, Sprache) -- außerhalb der hier verfügbaren Rechenleistung.

## Forschungsprotokoll B-003 — DMRG-adaptive Rangzuweisung (2026-08-02)

> ⚠ **ID-Kollision, bemerkt am 4. August.** Die Kennung `B-003` ist zweimal
> vergeben: hier für die DMRG-Rangzuweisung (2. August) und weiter unten für den
> Blockschicht-Vergleich (3. August, Seeds ab 1101). Das sind zwei völlig
> verschiedene Experimente.
>
> Faktisch entschieden ist es durch die Rohdaten: unter
> `results/experiments/B-003__*.json` liegt der **Blockschicht**-Lauf
> (80 Zeilen, 10 Seeds, mnist), und `python -m qire_ai.audit` prüft die
> Tagebuchzahlen gegen genau diese Datei. Dieser DMRG-Lauf hier hat **keine**
> Rohdaten und muss deshalb umbenannt werden, nicht der andere.
>
> **Vor der schriftlichen Arbeit erledigen** — sonst ist jede Zitierung von
> „B-003" mehrdeutig. In der Gesamtbilanz ist die DMRG-Rangzuweisung Nr. 5.

**Prinzip:** DMRG (White 1992) passt die Bond-Dimension *während* der
Optimierung anhand des verworfenen Spektralgewichts an. Übertragen: alle
Schichten starten klein, wachsen nur dort, wo das Spektrum noch nicht abfällt.
Zielachse **Trainingsaufwand** -- die einzige Ressource, die nie optimiert wurde.

**Zwei Implementierungsfehler im Smoke-Test gefunden und behoben:**
1. Das Sättigungsmaß griff auf `sv[-1]` zu. Da W = UV rangdefizitär ist, war
   das immer ~0 und das Maß wertlos. Korrigiert auf σ_r/σ_1 über die
   *behaltenen* Richtungen; gegen sechs Matrizen mit analytisch bekanntem
   Spektrum verifiziert (Abweichung 0).
2. Start bei Rang 1 setzte die Wachstumsarme in das Kollapsregime, das A-003
   dokumentiert hat -- das hätte "Wachstumsstrategie" mit "zu klein zum
   Trainieren" vermengt. Start auf Rang 3 korrigiert.

| Arm | Genauigkeit | Std | Trainingsaufwand | End-Ränge |
|---|---|---|---|---|
| fixed_uniform | 0,9175 | 0,0059 | 26.112 (100 %) | 8,0/8,0/8,0 |
| dmrg_adaptive | 0,9176 | 0,0098 | 20.811 (79,7 %) | 7,3/8,0/8,7 |
| random_growth | 0,9182 | 0,0111 | 20.811 (79,7 %) | 9,1/7,4/7,5 |

| Vergleich | Differenz | p |
|---|---|---|
| DMRG vs. Zufall | −0,0006 | 1,000 |
| DMRG vs. fest | +0,0001 | 0,768 |

**Ergebnis: H0 bestätigt.** Das spektrale Kriterium ist von Zufall
ununterscheidbar. Auch die Zuweisungen selbst unterscheiden sich kaum
(7,3/8,0/8,7 gegen 9,1/7,4/7,5) -- das Kriterium erzeugt gar keine
distinktive Verteilung, kann also auch kein anderes Ergebnis liefern.

**Nebenbefund (klassisch, nicht quanteninspiriert):** Beide Wachstumsvarianten
erreichen dieselbe Genauigkeit wie das feste Modell bei **79,7 % des
Trainingsaufwands** (p=0,768 gegen fest). Progressives Wachstum spart also ein
Fünftel der Trainingskosten ohne Verlust -- aber unabhängig davon, *wie*
gewachsen wird. Das ist eine bekannte klassische Technik.

### Gesamtbilanz: fünf Mechanismen, fünf Abgrenzungstests

| # | Mechanismus | Kontrolle | Ergebnis |
|---|---|---|---|
| 1 | Born-Regel-Gating | Softmax | kein Unterschied (p=0,25) |
| 2 | Verschränkungsentropie | Zufallsmatrizen | Kontrolle widerlegt |
| 3 | Tensor-Train-Bonds | Blockdiagonal | Kontrolle besser (p=0,004) |
| 4 | Unitarität | Spektralnorm | kein Unterschied (p=0,32) |
| 5 | DMRG-Kriterium | Zufallswachstum | kein Unterschied (p=1,00) |

Plus C-001 (Dichtematrix-Routing): in dieser Umgebung nicht prüfbar.

**Konsistentes Muster:** Wo ein quanteninspirierter Mechanismus gewirkt hat,
liess sich die Wirkung stets durch eine einfachere klassische Struktur
vollständig reproduzieren.

## Q-Serie — Echte Quantenmechanik statt Analogie (2026-08-02)

Nach fünf gescheiterten quanten*inspirierten* Ansätzen der Wechsel zu echter
Quantenmechanik: `quantum_sim/torch_simulator.py` portiert den bereits gegen
Qiskit validierten Simulator nach PyTorch mit komplexen Zahlen, sodass
Gradienten durch Superposition und Verschränkung fliessen. Darauf aufbauend
`quantum_layer.py`: parametrisierte Quantenschaltung als trainierbare Schicht.

Verifiziert (6/6): identisch zur NumPy-Fassung bei Bell/GHZ/Rotationen,
Normerhaltung, Gradientenfluss (Norm 2,37), ⟨Z⟩ gegen analytische Werte,
Amplituden-Kodierung, Batch-Unabhängigkeit. Verschränkung nachgewiesen:
p(00)·p(11)=0,25 gegen p(01)·p(10)=0,00, also nachweislich nicht separabel.

Parameterbilanz: 2·n·L Winkel wirken auf einen 2^n-dimensionalen Zustandsraum.
Bei n=8, L=5 sind das 80 Parameter gegenüber 65.536 einer dichten Entsprechung.

### Q-001 — erstes signifikant positives Ergebnis

| Arm | Genauigkeit | Parameter |
|---|---|---|
| hybrid | 0,9027 | 80 |
| hybrid_noent (ohne CNOT) | 0,8989 | 80 |
| classical_linear | 0,7645 | 2.056 |
| classical_dense | 0,8481 | 67.848 |

hybrid vs. classical_dense: +0,0546, p=0,0078 -- 848× weniger Parameter bei
höherer Genauigkeit. Verschränkung trägt aber nichts bei (+0,0039, p=0,141).

### Q-002 — Normierung als Erklärung ausgeschlossen

Verdacht: Die Amplituden-Kodierung normiert den Eingabevektor, die klassischen
Arme in Q-001 bekamen das nicht. Kontrolle mit L2-Normierung:

| Arm | Genauigkeit | Parameter |
|---|---|---|
| hybrid | 0,9097 | 80 |
| classical_norm | 0,8604 | 2.056 |
| classical_norm_deep | 0,8390 | 67.848 |

Vorteil hält (+0,0707, p=0,0078). Normierung allein erklärt nichts
(−0,0042, p=0,547). **Erster Mechanismus im Projekt, der eine faire Kontrolle
übersteht.**

### Q-003 — und dann fällt auch dieser

Die Schaltung leistet drei Schritte: normieren, normerhaltende Drehung mit
wenigen Parametern, **quadratische** Auslese (⟨Z⟩ summiert über |Amplitude|²).
Der letzte Punkt war bis dahin übersehen -- die klassischen Kontrollen
rechneten Linear/Tanh, eine ganz andere Funktionsklasse. Klassischer Nachbau
aller drei Schritte:

| Arm | Genauigkeit | Std | Parameter |
|---|---|---|---|
| hybrid | 0,8975 | 0,0090 | 80 |
| classical_quadratic (orthogonal) | 0,8202 | 0,0590 | 2.048 |
| **classical_quad_free (freie Matrix)** | **0,9122** | 0,0040 | 2.048 |
| classical_norm_deep | 0,8382 | 0,0079 | 67.848 |

**hybrid vs. classical_quad_free: −0,0147, p=0,0078 -- die klassische
Nachbildung schlägt die Quantenschaltung signifikant.**

Aufschlussreich im Detail: die *orthogonale* Variante (also die
Unitaritätseigenschaft, das eigentlich Quantenhafte) schneidet am schlechtesten
ab und ist hochgradig instabil (Streuung 0,0590 gegen 0,0040). Die
quantenspezifische Beschränkung schadet hier aktiv.

Was wirkt, ist die **Funktionsform**: normieren → lineare Transformation →
quadrieren → gruppenweise summieren. Klassisch vollständig beschreibbar.

### Gesamtbilanz: sechs Mechanismen, sechs Abgrenzungstests

| # | Mechanismus | Kontrolle | Ergebnis |
|---|---|---|---|
| 1 | Born-Regel-Gating | Softmax | kein Unterschied (p=0,25) |
| 2 | Verschränkungsentropie | Zufallsmatrizen | widerlegt |
| 3 | Tensor-Train-Bonds | Blockdiagonal | Kontrolle besser (p=0,004) |
| 4 | Unitarität | Spektralnorm | kein Unterschied (p=0,32) |
| 5 | DMRG-Kriterium | Zufallswachstum | kein Unterschied (p=1,00) |
| 6 | echte Quantenschaltung | klassischer Nachbau | Kontrolle besser (p=0,008) |

Der sechste Test nutzte tatsächliche Quantenmechanik, keine Analogie -- und
auch dort war der Effekt klassisch reproduzierbar bzw. klassisch sogar besser.

**Positivbefund, der aus der Quantenarbeit stammt:** Die quadratische Auslese.
`classical_quad_free` schlägt mit 2.048 Parametern das dichte Netz mit 67.848
um +0,074. Auf diese Architektur führt der klassische Weg nicht -- sie kam aus
dem Studium der Quantenschaltung. Entdeckt durch Quantenphysik, erklärbar ohne
sie.

## Q-004b — In Simulation trainiert, auf echter Hardware ausgefuehrt (2026-08-03)

Der Kreis ist geschlossen: ein KI-Modell, dessen Quantenteil in Simulation
gelernt und anschliessend auf einem echten IBM-Prozessor ausgefuehrt wurde.

**Notwendige Umstellung auf Winkel-Kodierung.** Das Modell aus Q-001 bis Q-003
nutzt Amplituden-Kodierung (256-dim Vektor als 8-Qubit-Zustand). Auf echter
Hardware braeuchte diese Zustandspraeparation hunderte Gatter -- bei rund 2 %
Fehler je drei Gatter waere das Ergebnis reines Rauschen. Winkel-Kodierung
(jedes Merkmal wird ein Drehwinkel) kostet nur n Gatter und macht den Vergleich
ueberhaupt erst moeglich.

**Vorab verifiziert:** Torch-Schaltung und Qiskit-Schaltung liefern identische
⟨Z⟩-Werte (Abweichung 4,6e-06). Ohne diese Pruefung waere jeder
Hardware-Vergleich wertlos gewesen. Bit-Reihenfolge explizit behandelt (Qiskit
zaehlt Qubit 0 von rechts).

Backend `ibm_fez`, 20 Testbeispiele je Konfiguration, 1024 Shots:

| | flach (4 Qubits, 33 Gatter) | tief (6 Qubits, 67 Gatter) |
|---|---|---|
| Job-ID | `d9ns0pc60llc73caf80g` | `d9ns10ssfqic73ardag0` |
| Genauigkeit Simulation | 0,500 | 0,450 |
| **Genauigkeit Hardware** | **0,500** | **0,450** |
| Uebereinstimmung der Vorhersagen | 0,850 | 0,750 |
| mittlerer ⟨Z⟩-Fehler | 0,1076 | 0,1498 |
| Daempfung ⟨Z⟩ | 0,852 | 0,707 |

**Das Modell funktioniert auf echter Quantenhardware** -- die Genauigkeit ist
trotz Rauschen identisch zur Simulation, bei der flachen Schaltung stimmen
85 % der Einzelvorhersagen exakt ueberein.

**Der Tiefeneffekt bestaetigt sich an einer echten Anwendung:** ⟨Z⟩-Fehler
waechst von 0,108 auf 0,150, die Daempfung faellt von 0,852 auf 0,707. Bei
doppelter Gatterzahl bleiben statt 85 % nur noch 71 % der Signalstaerke.
Konsistent mit der frueheren Bell/GHZ-Messung (1,57 % -> 2,06 %, p=0,043),
jetzt an einem laufenden KI-Modell statt an einer Testschaltung.

**Damit existiert die Vereinigung von KI und Quantenraum als lauffaehiges
System:**

    784 Pixel -> klassischer Encoder -> Drehwinkel
              -> ECHTER QUANTENPROZESSOR (ibm_fez)
                 Superposition, Verschraenkung, Messung
              -> ⟨Z⟩ je Qubit -> klassischer Kopf -> Ziffer

Kein Effizienzvorteil (die sechs Abgrenzungstests bleiben gueltig), aber ein
funktionierendes System mit nachpruefbaren Job-IDs.

## K-Serie — Quantenkerne und das Konzentrationsprinzip (2026-08-03)

**Neue Perspektive:** Alle sechs vorherigen Ansaetze veraenderten *das Modell*
und massen Genauigkeit, Parameter oder Zeit. Hier wird nichts am Modell
veraendert -- der Quantenzustandsraum dient als *Merkmalsraum*, und gemessen
wird **Datenbedarf**, die einzige Ressource der Projektliste, die nie
untersucht wurde.

Kern: k(x,y) = |<phi(x)|phi(y)>|^2 mit der ZZ-Merkmalsabbildung
(Havlicek et al. 2019). Umgesetzt in `quantum_sim/quantum_kernel.py`,
vorab geprueft: positiv semidefinit (kleinster Eigenwert 0,77), symmetrisch
(Abweichung exakt 0), Diagonale exakt 1, voller Rang.

Alle Kerne erhalten dieselben PCA-Merkmale (labelfrei) und dieselbe
C-Gitter-Suche. SVM mit vorberechnetem Kern gegen RBF, polynomial, linear.

### Ergebnis: signifikant schlechter bei jeder Konfiguration

| Qubits | Quantenkern (n=200) | bester klassischer | Abstand | p |
|---|---|---|---|---|
| 3 | 0,311 | 0,451 (RBF) | −0,140 | 0,0078 |
| 4 | 0,274 | 0,564 (poly) | −0,290 | 0,0078 |
| 6 | 0,165 | 0,717 (RBF) | −0,552 | (5 Seeds, s.u.) |

Methodischer Hinweis: Der erste Lauf nutzte 5 Seeds -- damit ist im
Wilcoxon-Test p=0,0625 das rechnerische Minimum, Signifikanz war also gar
nicht erreichbar. Mit 8 Seeds nachgeholt.

### Die Ursache, quantitativ vermessen: exponentielle Konzentration

Mittlerer ausserdiagonaler Kernwert auf echten MNIST-Daten:

| Qubits | mittlere Aehnlichkeit |
|---|---|
| 2 | 0,349 |
| 3 | 0,162 |
| 4 | 0,082 |
| 6 | 0,024 |

**Die Aehnlichkeit halbiert sich mit jedem Qubit.** Bei 6 Qubits ist der Median
0,013 -- die Kernmatrix ist praktisch die Einheitsmatrix. Die SVM kann den
Trainingssatz auswendig lernen, hat aber nichts zum Verallgemeinern. Der Kern
ist dabei mathematisch einwandfrei; das Problem ist die Geometrie.

Der wachsende Abstand zu den klassischen Kernen (−0,14 → −0,29 → −0,55) folgt
exakt dieser Konzentration.

### Das Prinzip -- und warum es mehr erklaert als nur diesen Versuch

> In einem 2^n-dimensionalen Hilbertraum sind zwei zufaellige Zustaende fast
> immer orthogonal. Was Quantensysteme ausdrucksstark macht -- die exponentiell
> wachsende Dimension -- zerstoert zugleich die geteilte Struktur, die
> Lernverfahren brauchen.

Das ist ein plausibler gemeinsamer Grund fuer mehrere der sechs vorherigen
Befunde: Ein grosser Zustandsraum ist kein Vorteil, wenn die Daten darin keine
gemeinsame Struktur mehr teilen. Damit hat das Projekt kein Buendel isolierter
Nullresultate, sondern ein quantitativ belegtes Prinzip, das sie verbindet.

**Klassifikation:** verworfen fuer diese Aufgabe. Nicht widerlegt ist, dass
Quantenkerne bei Daten mit passender Symmetrie funktionieren koennten -- dort
ist die Merkmalsabbildung auf die Struktur der Daten abgestimmt statt generisch.

## B-Serie — die Methodenarbeit (2026-08-03 bis 04)

*Nachgetragen am 4. August. Diese Serie fehlte in der Roadmap vollständig,
obwohl in ihr vier Hauptaussagen gefallen sind und die beiden einzigen belegten
Positivbefunde des Projekts entstanden. Ausführliche Fassung mit allen
Zwischenschritten im `PROJEKTTAGEBUCH.md`.*

### Vorweg: kein einziges Skript hatte je Ergebnisse gespeichert

`run()` gab ein Dictionary zurück, `analyse()` druckte auf die Konsole, danach
war jede Zahl weg. **Sämtliche Werte in Roadmap und Tagebuch bis zum 3. August
stammen aus Konsolenausgaben, die es nicht mehr gibt** — nicht nachrechenbar,
einer Jury nicht vorlegbar. Behoben mit `qire_ai/result_store.py`: jede
einzelne Messung als eigene Zeile, dazu Konfiguration und Softwareumgebung, als
JSON und CSV unter `results/experiments/`. `python -m qire_ai.audit` prüft
seitdem Tagebuchzahlen gegen die Rohdaten und listet, was unbelegt ist.

Der Preis war sofort spürbar: B-002 lässt sich nicht mit B-003 zusammenlegen,
weil B-002 seine Rohdaten nicht hat.

### Forschungsprotokoll B-003 — die 819×-Aussage war nie gemessen worden

*Achtung: Diese Kennung ist doppelt vergeben — siehe den Kasten beim
DMRG-Protokoll weiter oben. Die gespeicherten Rohdaten unter
`results/experiments/B-003__*` gehören zu **diesem** Lauf hier.*

Beim Nachsehen im Code war die Aussage *„spektral-normalisierte Blockschicht,
819× weniger Parameter, besser als dicht"* aus zwei Läufen zusammengesetzt, die
es so nie zusammen gab: A-002 (unbeschränkt gegen dicht, Seeds 701–710,
Tiefe 1) und B-002 (spektral gegen unbeschränkt, Seeds 1001–1010, Tiefe 3,
**ohne jeden dichten Vergleichsarm**). Aus „unbeschränkt schlägt dicht" und
„spektral schlägt unbeschränkt" folgt rechnerisch nichts.

Alle vier Arme erstmals in *einem* Lauf, 10 frische Seeds ab 1101:

| Tiefe 1 | Genauigkeit | Parameter | gegen dicht |
|---|---|---|---|
| dicht | 0,9237 | 16.781.312 | — |
| Block, unbeschränkt | 0,9362 | 20.480 | +0,0125, 10/10, p=0,0020 |
| Block, spektral | 0,9394 | 20.480 | +0,0157, 10/10, p=0,0020 |
| Block, orthogonal | 0,9442 | 20.480 | +0,0204, 10/10, p=0,0020 |

Der Zusatz „spektral-normalisiert" trägt bei Tiefe 1 nichts (+0,0032, p=0,43),
bei Tiefe 3 etwas (+0,0075, p=0,027) — die Aussage war schon hier in ihrer
pauschalen Form nicht haltbar.

**Ein Fund, der ausdrücklich nicht als Ergebnis geführt wird:**
`block_orthogonal` gegen `block_spectral` ergibt **+0,0068 bei p=0,037 (Tiefe 3)**,
bei Tiefe 1 nicht signifikant (p=0,065). B-002 hat *exakt denselben Vergleich*
gerechnet — gleiche Architektur, Blockgröße, Tiefe, Epochen, ebenfalls
10 Seeds, nur andere Seeds — und kam auf p=0,322. Dasselbe Experiment,
gegenteilige Schlussfolgerung, allein durch die Seed-Wahl. Nicht
vorregistriert, sechs Tests auf demselben Datensatz, repliziert nicht über
beide Tiefen — deshalb kein Befund, aber der Anlass für B-007 (siehe unten).

### Forschungsprotokoll B-004 — erst fällt die Lernrate, dann die Aussage

Alle bisherigen Experimente trainierten jeden Arm mit lr=0,01. Lernratensuche
je Arm auf **eigenen** Tuning-Seeds (1301–1303), Hauptlauf auf 1201–1210:

| dichte Schicht, dichte Enden | Genauigkeit |
|---|---|
| lr 0,003 | **0,9359** |
| lr 0,01 — bisherige Einstellung | 0,9302 |
| lr 0,03 | 0,8878 |

**Die dichte Schicht war in jedem bisherigen Vergleich mit einer Lernrate
trainiert, die sie rund 0,006 Genauigkeit gekostet hat.**

Monarch (Dao et al., ICML 2022) selbst implementiert und vorab verifiziert
(schneller Pfad gegen explizite Blockdiagonale·Permutation·Blockdiagonale,
größte Abweichung 3,3·10⁻¹⁶). Damit vier Familien bei **exakt** 524.288
Parametern: Monarch 64×64, Blockdiagonale 128, Tensor-Train 2 Kerne/Rang 64,
Low-Rank Rang 64.

Die Architekturfrage aus A-003 gemessen statt diskutiert, über die
Jacobi-Matrix der Aktivierungen:

| Aufbau | größter Jacobi-Eintrag zu einer fremden Gruppe |
|---|---|
| gruppenweise Enden, Blockdiagonale | **exakt 0** |
| gruppenweise Enden, Monarch | 3,3·10⁻² |
| **alter Aufbau mit dichten Enden, Blockdiagonale** | **6,9·10⁻²** |

Im alten Aufbau floss auch mit blockdiagonaler Mitte Information zwischen den
Gruppen — A-003 konnte seine eigene Frage nicht beantworten.

Die 819×-Aussage fair gemessen (`block_diag_4`, 20.480 Parameter, jeder Arm bei
seiner besten Lernrate, Bootstrap-Intervalle):

| Architektur | Differenz | 95 %-KI | Siege | p |
|---|---|---|---|---|
| dichte Enden (alter Aufbau) | +0,0054 | [+0,0013, +0,0104] | 8/10 | 0,037 |
| gruppenweise Enden | **−0,0058** | [−0,0102, −0,0018] | 2/10 | 0,027 |
| gruppenweise Enden, Pixel permutiert | **−0,0109** | [−0,0137, −0,0082] | 0/10 | 0,002 |

**Die Aussage gilt nur in der Architektur mit dichten Enden. Nimmt man die freie
globale Mischung weg, dreht das Vorzeichen.** Dazu: in genau dem Aufbau, aus dem
A-002 und A-003 stammen, trennt sich die dichte Schicht mit 16,8 Mio.
Parametern **nicht** signifikant von der Blockschicht mit 528.384 (+0,0046,
p=0,23) — **die Positivkontrolle fällt durch**, ein Budgetunterschied von
Faktor 32 ist dort nicht auflösbar. Präzisierung, um nicht zu überziehen:
„blind" wäre zu stark, denn der spektrale Blockarm trennt sich sehr wohl von
dicht (+0,0095, Holm p=0,023). Der Aufbau kann nur den einen Vergleich nicht
auflösen, auf dem die zurückgezogene Aussage beruhte.

**Die Pixelpermutation trifft den anderen Arm als erwartet:**

| gruppenweise Enden → gemischte Pixel | Änderung | p |
|---|---|---|
| Blockdiagonale | +0,0013 | 0,92 |
| dicht | +0,0018 | 0,56 |
| **Monarch** | **−0,0133** | **0,0020** |

Die Blockdiagonale ist gegen Pixelmischung gleichgültig, Monarch verliert
signifikant. Damit lösen sich die Urteile auf: bei natürlicher Bildstruktur
schlägt Monarch die Blockdiagonale (+0,0115, p=0,0039), bei gemischten Pixeln
nicht mehr (−0,0031, p=0,23) — beide Male mit bestandener Positivkontrolle.
**Monarchs Vorteil ist nicht globale Mischung an sich, sondern das Ausnutzen
räumlicher Struktur.** Diese Kontrolle fehlt auch in der Literatur.

### Forschungsprotokoll B-005 — was die dichte Schicht wirklich lernt

Statt eines weiteren Pferderennens: hineinschauen. Die gelernte
4096×4096-Matrix in 32×32 Blöcke à 128 zerlegt, Betragsmasse auf der Diagonale
gegen zufällige Permutation als Nullverteilung.

| | bei Initialisierung | trainiert |
|---|---|---|
| Blockmasse auf der Diagonale (Zufall: 0,0312) | 0,0313 | **0,0312** |
| beste per Co-Clustering gefundene Anordnung | 0,0341 | 0,0330 |
| **Energie im Top-64-Rang** | **0,0542** | **0,7144** |

**Die dichte Schicht lernt exakt nichts Blockartiges** — Faktor 1,00 gegenüber
Zufall, auch in keiner anderen Anordnung. Sie presst stattdessen die
Spektralenergie in wenige Richtungen. Dazu passt ein Messwert aus B-004, der
vorher nicht aufgefallen war: Low-Rank mit Rang 64 gegen dicht ist +0,0003 bei
p=0,85 — bei 32× weniger Parametern. Erstmals eine *Erklärung* statt einer
Messung: Blöcke sind die falsche Struktur für das, was die Schicht tut.

*(Zahlenkorrektur, siehe B-008: die 71 % sind mit genäherter SVD gemessen;
exakt sind es 64,6 %.)*

### Forschungsprotokoll B-006 — nach eigener Regel nicht verwertbar

Konvergenzkontrolle mit 15.000 Beispielen und 30 Epochen statt 6.000 und 10.
Bei dichten Enden erreichen alle Arme 0,960 ± 0,001 und **kein** Unterschied
bleibt signifikant. Bei gruppenweisen Enden repliziert die Richtung (Monarch
+0,0152, p=0,031), aber die Positivkontrolle verfehlt die Schwelle (p=0,0625;
bei 6 Seeds ist 0,031 das rechnerische Minimum). Vorher festgelegtes Kriterium:
verwerfen — wird nicht umgedeutet.

Unabhängig davon zeigt B-006: **ein Teil der Unterschiede in B-004 ist ein
Untertrainings-Artefakt**, bei längerem Training verschwinden sie bei dichten
Enden vollständig.

### Äquivalenztest — und er nimmt der neuen Hauptaussage die Schärfe

„Rang 64 ersetzt eine dichte Schicht **ohne messbaren Verlust**" ist eine
Äquivalenzbehauptung; ein Wilcoxon-Test mit p=0,85 belegt sie nicht. Richtiges
Werkzeug ist TOST, Grenze vorab auf ±0,005 festgelegt
(`qire_ai/equivalence.py`):

| | |
|---|---|
| Differenz | +0,0003 |
| Wilcoxon p / nach Holm | 0,85 / 1,00 |
| 90-%-Intervall | [−0,0072, +0,0078] |
| **TOST p bei ±0,005** | **0,139 — Äquivalenz NICHT nachgewiesen** |
| erreichte Schranke | **±0,0078** |

Erlaubt ist also nur „gleichwertig innerhalb ±0,008". Für die strengere Aussage
bräuchte es rund 30 Seeds.

### Was nach dem 3. August vormittags zurückgezogen ist

1. *„Spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als
   dicht."* Gilt nur in einer Architektur, deren Blindheit gemessen ist; in
   beiden faireren Aufbauten dreht das Vorzeichen.
2. *„Bonds tragen nichts, es ist die Lokalität."* Monarch schlägt die
   Blockdiagonale bei gleichem Budget auf natürlichen Bilddaten.

### Forschungsprotokoll B-008 / B-009 — auch der Ersatzbefund fällt (2026-08-03)

Beide Ersatzaussagen standen auf 10 Seeds. Wiederholt mit **30 Seeds** und auf
**zwei Datensätzen**.

**B-008** sollte aus B-005 eine Regel machen: Wenn die dichte Schicht ihre
Energie auf wenige Richtungen konzentriert, muss die Genauigkeit dort sättigen,
wo die Energie sättigt. Vorher festgelegt: **eine einzige** Energieschwelle muss
auf **beiden** Datensätzen passen.

| Rang | Parameter | Genauigkeit | vs. dicht | TOST p | kum. Energie |
|---|---|---|---|---|---|
| dicht | 16.781.312 | 0,9379 | — | — | 1,0000 |
| 8 | 69.632 | 0,9332 | −0,0047 | 0,46 | 0,4598 |
| 64 | 528.384 | 0,9313 | −0,0066 | 0,72 | 0,6455 |
| 256 | 2.101.248 | 0,9269 | −0,0110 | 1,00 | 0,7416 |

**Kein einziger Rang erreicht Gleichwertigkeit innerhalb ±0,005.** Die
Vorhersage ist widerlegt. Die Genauigkeit ist über die Ränge **flach** — Rang 8
mit 69.632 Parametern ist so gut wie Rang 256 mit 2,1 Millionen, und alle
liegen etwa 0,005 unter dicht. Mehr Rang schließt die Lücke nicht.

Dabei ein eigener Messfehler aufgefallen: die aus B-005 berichteten **71 %**
Spektralenergie im Top-64-Rang sind exakt gemessen **64,6 %** (bei Rang 256
erst 74 %). Zwei mögliche Ursachen, nicht sauber trennbar: B-005 nutzte
`svd_lowrank` (genähert) statt exakter SVD, und lr=0,01 statt der gesuchten
0,003. Für die schriftliche Arbeit gilt die exakte Zahl.

**B-009** — Monarch gegen dicht in der lokalen Architektur, 30 Seeds:

| Datensatz | Differenz | p (Holm) | Lesart |
|---|---|---|---|
| MNIST | −0,0045 | 0,0010 | Unterschied belegt |
| Fashion-MNIST | −0,0112 | 0,0001 | Unterschied belegt |

Bei 10 Seeds war es −0,0008 mit p=0,92 und galt als bester Äquivalenzkandidat
des Projekts. **Zum dritten Mal an einem Tag** kippt ein knapper Befund
zwischen 10 und 30 Seeds.

Monarch direkt gegen die Blockdiagonale repliziert nur auf einem Datensatz:
MNIST +0,0094 (27/30, p<0,0001), Fashion-MNIST −0,0028 (11/30, p=0,077).
Damit ist auch die Erklärung „räumliche Bildstruktur" zu einfach —
Fashion-MNIST *hat* räumliche Struktur, und dort gibt es den Vorteil nicht.

**Bilanz von B nach diesem Tag:** vier Hauptaussagen gefallen, kein einziger
positiver Effizienzbefund. Was bleibt, ist tragfähiger als das Ersetzte:
**Keine der getesteten strukturierten Familien ersetzt unter fairen Bedingungen
eine dichte Schicht** (alle 0,005–0,014 darunter, zwei Datensätze, 30 Seeds,
Lernrate je Arm) — und drei einzeln gemessene Gründe, warum frühere Messungen
das Gegenteil zeigten: eine Architektur, die Faktor 32 nicht auflöst; eine
feste Lernrate, die den dichten Arm 0,006 kostete; zu kleine Seed-Zahlen.

### Forschungsprotokoll B-010 / B-011 — das falsche Spektrum (2026-08-04)

**B-010** fragte, ob sich Struktur auszahlt, wenn die Daten knapp werden. Sechs
Trainingsgrößen von 200 bis 14.000, fester großer Testsatz, angeglichene Zahl
der Gradientenschritte, Lernrate je Arm *und* Datenmenge, 15 Seeds, zwei
Datensätze. **Urteil H0** — kein Arm erfüllt die drei vorher festgelegten
Bedingungen.

Eine Sache repliziert trotzdem: Bei n=200 gewinnt die blockdiagonale Schicht
auf **beiden** Datensätzen (MNIST +0,0149, Fashion +0,0139, beide signifikant).
Gescheitert ist nur der Trendtest (ρ=−0,600 / −0,257, sechs Punkte reichen
nicht). Die naheliegende Erklärung ist von denselben Daten widerlegt: „weniger
Parameter, weniger Überanpassung" kann es nicht sein, denn `lowrank_r8` hat mit
69.632 die wenigsten Parameter und ist bei n=200 am schlechtesten (−0,0204).
Offenes Rätsel, konkret genug für einen eigenen Nachtest.

**B-011 — der eigentliche Fund: B-008 hat das falsche Spektrum gemessen.** Für
die Frage, was eine Trunkierung kostet, zählt nicht die Gewichtsmatrix W
allein, sondern W **gewichtet mit der Verteilung der echten Eingaben**. Eine
Richtung mit großem Singulärwert ist gleichgültig, wenn die Daten sie nie
ansprechen. Beides an derselben trainierten Schicht:

| | Rang 8 | Rang 64 |
|---|---|---|
| Spektrum der Gewichte W (das aus B-008) | 0,459 | 0,646 |
| **Spektrum der Ausgaben auf echten Daten** | 0,923 | **0,998** |

Auf Fashion-MNIST dasselbe Bild (0,963 / 0,999). **Rang 64 erfasst 99,8 % der
tatsächlich genutzten Varianz, nicht 65 %.** Der Negativbefund aus B-008 war
kein Befund über die Sache, sondern über die Messgröße.

Daraus der **erste bewiesene Äquivalenzbefund des Projekts** (MNIST, 15 Seeds,
dichte Referenz 0,9403):

| Arm | Genauigkeit | vs. dicht | TOST |
|---|---|---|---|
| von null trainiert | 0,9290 | −0,0114 | schlechter (p=0,0067) |
| **SVD-Projektion der trainierten dichten Schicht** | **0,9407** | **+0,0003** | **gleichwertig, p<0,0001** |
| Destillation der dichten Funktion | 0,9387 | −0,0016 | gleichwertig, p<0,0001 |

Auf Fashion-MNIST repliziert die entscheidende Zeile (+0,0011, TOST p=0,0001).
**Das ist die erste Äquivalenz, die nicht bloß nicht widerlegt, sondern
nachgewiesen ist** — mit einem Test, der dafür gebaut ist, nicht mit einem
p>0,05.

Bei Rang 8 dreht sich das Bild um: Projektion (−0,0321) und Destillation
(−0,0226) fallen ab, das von null trainierte Modell (−0,0059) ist besser als
beide. Dort kann die Familie die dichte Funktion nicht darstellen, und das
Training findet eine eigene, bessere Lösung. Das erklärt auch die flache
Rangkurve aus B-008.

Einschränkungen, die mitmüssen: Die Optimierungslücke gilt **nur für MNIST**
(Fashion −0,0011, p=0,80). Die Projektion setzt eine fertig trainierte dichte
Schicht voraus — **ein Kompressions-, kein Effizienzbefund**. Und sie
widerspricht dem eigenen Faktor-6,4-Befund, der bei 32 Neuronen gemessen wurde;
der hat offenbar eine Grenze, und die ist hier überschritten.

### Forschungsprotokoll B-012 bis B-014 — die Erzählung fällt, ein Gesetz bleibt (2026-08-04)

**B-012** widerlegt die eigene Hypothese „der Rang kollabiert während des
Trainings". Gemessen je Epoche: Effektivrang MNIST 5,1 (Epoche 1) → 6,7
(Epoche 10), Energie@64 0,9990 → 0,9977. **Der Rang steigt** (ρ=+0,988,
p<0,0001; Fashion 3,0 → 4,8), und die niedrigdimensionale Struktur ist schon
nach einer Epoche da. Zweite Korrektur: der schwellenfreie Effektivrang liegt
bei **5 bis 7**, nicht bei 64 — „Rang 64 erfasst 99,8 %" ist eine kumulierte
Größe mit langem Schwanz, die anschauliche Aussage lautet „die Schicht arbeitet
in rund sieben Richtungen". Um eine Größenordnung danebengelegen.

**B-013** prüft eine vorab ins Skript geschriebene Vorhersage: Wenn die Struktur
schon nach einer Epoche steht, darf ein über acht Epochen verteiltes Curriculum
nicht besser sein als ein einzelner Schnitt.

| MNIST | Genauigkeit |
|---|---|
| dicht (16,8 Mio. Parameter) | 0,9367 |
| fester Rang 64 ab Epoche 0 | 0,9303 |
| **ein Schnitt bei Epoche 5** | **0,9399** |
| Curriculum (512→256→128→64) | 0,9350 |

**H0** — das Curriculum schlägt festen Rang nicht, und ist gegen den einzelnen
Schnitt auf Fashion-MNIST signifikant **schlechter** (−0,0106, Holm p=0,0012),
vermutlich weil jede Projektion den Adam-Zustand zurücksetzt. Verwertbar bleibt:
**ein einzelner Schnitt zur richtigen Zeit schlägt festen Rang** (MNIST +0,0097,
Holm p=0,0215). Gegenüber dicht liegt er auf beiden Datensätzen numerisch vorn
(+0,0032 / +0,0049), aber bei 15 Seeds ist weder Überlegenheit noch
Gleichwertigkeit nachgewiesen — die Datenlage trägt nur ±0,007 bzw. ±0,010.

**B-014 — der erste kontrollierte Befund des Projekts.** Synthetische Daten mit
vorgegebener intrinsischer Dimension d, gekreuzt mit der Klassenzahl C. Zwei
echte unabhängige Variablen, plus untrainierte Kontrolle; 6×4-Gitter, 10 Seeds.

Effektivrang **nach** dem Training:

| d \ C | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| 2 | 1,4 | 1,7 | 1,7 | 1,7 |
| 10 | 1,5 | 2,4 | 2,1 | 2,7 |
| 100 | 1,7 | 3,0 | 3,7 | 4,6 |

Am **untrainierten** Netz dagegen 3,7 / 19,1 / 165,5 — unabhängig von C.

- **Untrainiert** folgt der Rang exakt der Datendimension (ρ=+0,986,
  p=3·10⁻¹⁸⁷) und ist von der Klassenzahl **vollständig unabhängig**
  (ρ=+0,002, p=0,98).
- **Trainiert** hängt er von **beiden** ab (ρ=+0,957 bzw. +0,833) und liegt
  drastisch niedriger (1,4–4,7 statt 3,7–165).

Die Klassenabhängigkeit existiert am untrainierten Netz nicht — **sie entsteht
beim Lernen**. Das Muster sieht aus wie ein *Minimum* der beiden Zwänge. Die
einfache Vorhersage „Rang = Klassenzahl − 1" trifft nicht (2,7 statt 9 bei
C=10), es ist also keine Neural-Collapse-Aussage, sondern etwas Eigenes.

**Warum das die wichtigste Zahl des Teilprojekts ist:** Sie erklärt alle
Negativbefunde von B auf einmal. Rang 8, 64 und 256 lieferten in B-008
praktisch dieselbe Genauigkeit, und mehr Budget half nie. Der Grund steht
jetzt da: **Der Rang war nie die bindende Beschränkung.** Deshalb konnte keine
strukturierte Familie durch mehr Kapazität gewinnen — sie hatten alle längst
genug.

## Gesamtbilanz: sieben Mechanismen, sieben Abgrenzungstests (Stand 4. August)

*Ersetzt die Sechser-Tabelle weiter oben, in der die Quantenkerne fehlten.*

| # | Mechanismus | Kontrolle | Ergebnis |
|---|---|---|---|
| 1 | Born-Regel-Gating | Softmax | kein Unterschied (p=0,25) |
| 2 | Verschränkungsentropie | Zufallsmatrizen | widerlegt |
| 3 | Tensor-Train-Bonds | Blockdiagonal | Kontrolle besser (p=0,004) — **überholt, s. B-004** |
| 4 | Unitarität | Spektralnorm | kein Unterschied (p=0,32) — **unsicher, B-007 läuft** |
| 5 | DMRG-Kriterium | Zufallswachstum | kein Unterschied (p=1,00) |
| 6 | echte Quantenschaltung | klassischer Nachbau | Kontrolle besser (p=0,008) |
| 7 | Quantenkerne | RBF/poly/linear | Kontrolle besser (p=0,008) |

Plus C-001 (Dichtematrix-Routing): in dieser Umgebung nicht prüfbar, nicht
falsifiziert.

**Das verbindende Prinzip**, aus Nr. 7 quantitativ hergeleitet: In einem
2ⁿ-dimensionalen Hilbertraum sind zufällige Zustände fast immer orthogonal. Die
exponentielle Dimension macht Quantensysteme ausdrucksstark und zugleich als
Lernmaschine untauglich.

### Was klassisch dabei herauskam

| Verfahren | Ergebnis |
|---|---|
| normieren + Transformation + quadrieren | 33× weniger Parameter, +7,4 Prozentpunkte — **Rohdaten fehlen, Replikation läuft** |
| Rang-64-Projektion einer **fertig trainierten** dichten Schicht | 32× weniger Parameter, **nachgewiesen gleichwertig** (TOST p<0,0001, beide Datensätze, 15 Seeds) |
| Effektivrang wird von der Aufgabe gesetzt, nicht von der Architektur | kontrolliertes 6×4-Gitter mit untrainierter Kontrolle (B-014) |

## Laufend (Stand 4. August, 22:45 Uhr)

- **B-007** (`qire_ai/ablation_unitarity.py`) — vorregistriert, 20 Seeds,
  Lernrate je Arm, zwei Datensätze, TOST. Klärt Zeile 4 der Gesamtbilanz:
  Orthogonalität gegen Spektralnorm. Beide früheren Messungen (p=0,32 und
  p=0,037) hatten 10 Seeds und damit etwa Münzwurf-Wahrscheinlichkeit, den
  Effekt überhaupt zu finden.
- **Q-003-replication** (`quantum_sim/hybrid_replication.py`) — startet
  automatisch, sobald B-007 die Kerne freigibt. Belegt Q-001 bis Q-003 mit
  Rohdaten; 15 Seeds, Lernrate je Arm, zwei Datensätze, Holm, TOST,
  Positivkontrolle. Die Protokolländerungen sind vor dem Lauf im
  `PROJEKTTAGEBUCH.md` festgehalten.

### Phase 3 — Kernaussage schärfen — **im Wesentlichen erledigt, 2.–4. August**
*Der ursprüngliche Zeitplan sah Oktober vor; die Empirie war Anfang August
durch.*
- [x] Klare, verteidigbare These formulieren — steht, aber anders als geplant:
      nicht „Tensor-Netzwerke sparen Ressourcen", sondern ein Prüfmaßstab plus
      sieben Abgrenzungstests. Die Neuausrichtung auf Tensor-Netzwerke ist
      selbst gescheitert (A-002, B-008).
- [x] Fair-Baseline-Vergleich mit exakt gematchter Parameterzahl — B-004,
      vier Familien bei exakt 524.288 Parametern.
- [x] Qiskit-Anbindung an `quantum_sim/`, erster Hardware-Lauf auf IBM Quantum —
      am 2. August erledigt (`ibm_marrakesh`, später `ibm_fez`).
- [ ] Reproduktionslauf — teilweise: B-007 und Q-003-replication laufen gerade.
      Q-001 bis Q-003 haben bis dahin **keine Rohdaten**.

### Phase 3b — was jetzt wirklich noch offen ist (August–Oktober 2026)
- [ ] `KURZFASSUNG_ENTWURF.md` vollständig gegen die Gesamtbilanz neu schreiben,
      sobald B-007 und Q-003-replication durch sind. Der Entwurf ist vom
      2. August und enthält einen zurückgezogenen Befund als Hauptergebnis.
- [ ] Entscheiden, ob A (Physik) und B (Mathematik/Informatik) getrennt
      angemeldet werden — bis zu drei Projekte sind erlaubt. Entscheidungsgrund
      ist der Aufwand, nicht die Zulässigkeit.
- [ ] Für B die Abgrenzung zum Stand der Technik belegen (Blockstruktur vs.
      gruppierte Faltungen, Monarch) — ohne sie ist B als eigenes Projekt
      angreifbar.
- [ ] Literaturkarten in `sources/literature_cards/` auf den echten Lesestand
      bringen; sie sind pauschal als ungelesen markiert, tatsächlich sind rund
      drei Viertel gelesen.
- [ ] `results/reports/RUN_BOOK.md` und `recommend_next.md` sind automatisch
      erzeugte Reste vom 1. August und empfehlen längst Überholtes
      („mindestens 5 Seeds") — löschen oder datieren.

### Phase 4 — Anmeldung vorbereiten (bis 30.11.2026)
- Kurzfassung (max. 2 Seiten): Forschungsfrage, Methode, vorläufige Ergebnisse
- Anmeldeformular ausfüllen

### Phase 5 — Schriftliche Ausarbeitung (Dezember 2026 – Januar 2027)
- Theorieteil, Methodik, Ergebnisse (mit Signifikanz), Diskussion der Grenzen
  (Transparenz über Limitationen ist ein Pluspunkt, keine Schwäche)
- Finale, breiteste Experimentrunde (ggf. über Nacht / Oracle-Cloud-Profil)

### Phase 6 — Präsentation (Februar 2027 bis Regionalwettbewerb)
- Poster/Stand vorbereiten
- Kritische Nachfragen üben: Abgrenzung zu klassischem MoE-Gating, statistischer
  Beweis, Verhalten bei größeren/echten Datensätzen
