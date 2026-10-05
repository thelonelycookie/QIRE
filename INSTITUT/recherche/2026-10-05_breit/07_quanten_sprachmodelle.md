# 07 – Quanten-Sprachmodelle und Quanten-NLP (Stand 2024–2026)

Recherche vom 05.10.2026. WICHTIGE EINSCHRÄNKUNG: arxiv.org, export.arxiv.org, semanticscholar-API, quantinuum.com, multiversecomputing.com, quantumzeitgeist.com und phys.org sind per Egress-Proxy für WebFetch gesperrt, und das WebSearch-Budget (200 Anfragen/Sitzung) war nach ~12 Anfragen für dieses Thema erschöpft. Alle Angaben unten stammen daher aus **Suchergebnis-Auszügen (nur Auszug)**, nicht aus Volltexten. Wo eine Zahl fehlt, steht [?]. Nichts wurde ergänzt.

## 1. Kurzfassung

- Echte Quanten-Sprachmodelle sind 2024–2026 klein: Quixer (Quantinuum) ist das erste QML-Modell auf einem realistischen Sprachdatensatz (Penn Treebank), aber nur **simuliert** und nur „konkurrenzfähig mit gleich großen klassischen Baselines“ (kein Sieg über große Modelle).
- Der größte „Quanten“-Eingriff in ein echtes LLM ist **Cayley Unitary Adapters (Multiverse Computing, 2026)**: Llama 3.1 8B, ~6.000 Zusatzparameter, **1,4 % Perplexity-Verbesserung**, validiert auf IBM-156-Qubit-Hardware. Das ist QIREs engster Konkurrent/Verwandter (unitäre Adapter in eingefrorenen Projektionsmatrizen) und muss in der schriftlichen Arbeit zitiert werden.
- Alles Übrige ist entweder Hybrid mit klassischem Rest (HyQuT: 10 Qubits ersetzen ~10 % der Parameter eines 150M-Modells), Training-only-Quanten (QPA, QTHA) oder **quanteninspiriert** (CompactifAI, QuanTA, TT-Embeddings) und läuft rein klassisch.
- Adapter mit **logarithmisch in der Breite wachsender Parameterzahl** habe ich in den Treffern nicht gefunden. Nahe dran: QPA (Qubit-Zahl ⌈log₂ m⌉, aber Parameter über klassischen MLP-Mapper) und QuanTA (wenige Tensor-Achsen, Parameter nicht log). QIREs log₂(d)-Winkel je Matrix wirkt damit als Nische (Neuheit nur bedingt geprüft, siehe 4).
- Keine Arbeit erzielt nachgewiesenen Quantenvorteil bei Sprache. Die ehrliche Einordnung für QIRE: Ressourceneffizienz-Argument ja, Quantenvorteil-Argument nein.

## 2. Stand der Forschung

### 2.1 Grundlagen und Werkzeuge (DisCoCat, lambeq, QDisCoCirc)
- **lambeq** (Quantinuum): Python-Bibliothek, die Sätze (DisCoCat-Diagramme) in Quantenschaltungen übersetzt. https://arxiv.org/pdf/2110.04236 ; Doku https://docs.quantinuum.com/lambeq/bibliography.html
- **lambeq Gen II**: Text-Ebene statt Satz-Ebene, „Efficient Generation of Parameterised Quantum Circuits from Large Texts“ https://arxiv.org/html/2505.13208 (nur Auszug; Zahlen [?]). Blog: https://www.quantinuum.com/blog/lambeq-gen-ii-a-quantum-enhanced-interpretable-and-scalable-text-based-nlp-software-package
- **Duneau et al. 2024**, „Scalable and interpretable QNLP: an implementation on trapped ions“, https://arxiv.org/abs/2409.08777 : Modell QDisCoCirc, Aufgabe Fragebeantwortung auf **handgebauten Spielzeugdaten**, Lauf auf Quantinuum H1-1 (Ionenfalle). Idee: Komponenten klassisch trainieren, dann zu größeren Texten komponieren („compositional generalisation“), Auswertung großer Instanzen bräuchte asymptotisch einen Quantenrechner; umgeht Trainierbarkeitsprobleme (Barren Plateaus). Qubit-/Gatterzahlen [?] (nur Auszug).
- Begleitpapier „Quantum Algorithms for Compositional Text Processing“ https://arxiv.org/pdf/2408.06061 (nur Titel gesehen).

### 2.2 Architekturen
- **Quixer** (Khatri, Matos, Coopmans, Clark; Quantinuum), https://arxiv.org/pdf/2406.04305 , Code https://github.com/Quantinuum/Quixer : Token-Superposition per Linear Combination of Unitaries plus trainierbare nichtlineare Transformation per QSVT. Qubits: q (Datenregister) + ⌈log₂ n⌉ (Kontrolle) + 3 Hilfsqubits. Penn Treebank (966K Trainings-Token). Verglichen mit LSTM, Transformer, FNet; Ergebnis laut Auszug „competitive with equivalent classical baselines“. Konkrete Perplexity-Werte und Parameterzahl konnte ich nicht lesen [?]. Simulation, keine Hardware-Läufe in den Auszügen. Blog: https://www.quantinuum.com/blog/announcing-quixer---quantinuums-state-of-the-art-quantum-transformer-making-quantum-ai-a-little-more-realistic
- **QSANN** (Li, Zhao, Wang; Baidu), Sci. China Inf. Sci. 67 (2024), https://link.springer.com/article/10.1007/s11432-023-3879-7 : Q/K/V-Abbildungen als parametrisierte Schaltungen, Gauß-projizierte Quanten-Selbstaufmerksamkeit, Attention-Koeffizienten klassisch. Textklassifikation auf öffentlichen (kleinen) Datensätzen; schlägt laut Auszug das beste syntaktische QNLP-Modell und eine einfache klassische Self-Attention. Größen [?].
- **Quantum Vision Transformers** (Cherrat, Kerenidis et al.), Quantum 8, 1265 (2024), https://quantum-journal.org/papers/q-2024-02-22-1265/ : Datenlader + trainierbare Quanten-Orthogonalschichten; Simulationen auf medizinischen Bilddatensätzen „competitive, at times better“, Hardware-Experimente auf supraleitenden Chips bis **6 Qubits**. Orthogonale Schichten sind QIRE strukturell verwandt (Ry+CNOT = orthogonal).
- **Hybrid Quantum Transformer / HyQuT** (Kong), https://arxiv.org/abs/2511.10653 : VQCs im Transformer bei **8M und 150M** Parametern; **10 Qubits mit 80 Gattern ersetzen ~10 % der klassischen Parameter** des 150M-Modells bei vergleichbarer Konvergenz und Generierungsqualität (Dialog). Simulation [?]; Baseline = gleiches klassisches Modell ohne Quantenteil.
- **Quantum LLM Fine-Tuning** (Kim, Mei, Girotto, Yamada, Roetteler), https://arxiv.org/abs/2504.08732 : Satz-Transformer klassisch + VQC mit Langstrecken-Kopplung als Kopf, bis **+3,14 % Genauigkeit** gegenüber klassisch gleich großer Architektur (Sentiment). Klein.
- **QTHA – Quantum-Enhanced LLM Efficient Fine Tuning**, https://arxiv.org/abs/2503.12790 : Gewichtszerlegung in QNN + Tensornetz; **−76 % trainierbare Parameter gegenüber LoRA**, bis 17 % niedrigerer Trainingsverlust/besserer Test; Basismodelle DeepSeek-R1-Distill-Qwen-7B und Qwen2-7B-Instruct; Datensätze CPsyCounD, R1-Distill-SFT, CH-R1-Math mit nur 300–3000 Beispielen. Simulation [?].
- **QPA – Quantum Parameter Adaptation**, ICLR 2025, https://arxiv.org/abs/2410.09846 : QNN erzeugt beim Training die Gewichte (LoRA/DoRA) über klassischen MLP; Qubits N = ⌈log₂ m⌉; GPT-2: 52,06 % der LoRA-Parameter (+0,75 %), Gemma-2: 16,84 % (+0,07 %). Inferenz rein klassisch.
- **Cayley Unitary Adapters (CUA)**, Aizpurua, Singh, Kshetrimayum, Jahromi, Orús (Multiverse), https://arxiv.org/pdf/2605.05914 : Cayley-parametrisierte unitäre Blöcke in eingefrorenen Projektionsschichten; **Llama 3.1 8B: +1,4 % Perplexity mit ~6.000 Zusatzparametern**, validiert auf IBM-156-Qubit-System; SmolLM2-135M: Leistung steigt konsistent mit Größe des Quantenblocks; Quanten-Variante beantwortete Fragen richtig, die zwei klassische Varianten falsch hatten. Presse: https://phys.org/news/2026-06-quantum-circuits-ai-memory-limitations.html . Datensatz, Qubits pro Block und genaue Baselines [?] (nur Auszug).
- **HyperQ – Circuit Hypernetworks** (Wang et al., Montréal/Mila), https://arxiv.org/abs/2609.24657v1 : tokenabhängige Quanten-Residualzweige in einem eingefrorenen Masked-Diffusion-LM (1,1 Mrd. Parameter); IQP-Schaltungen auf Ring+Sehne-Gerüst → **exakt klassisch auswertbar, Kosten linear in Qubitzahl**; 16→64 Qubits, Durchschnittsscore 47,65→54,30; bei 64 Qubits +4,71 gegenüber Backbone, +3,67 gegenüber LoRA. Wichtig: das ist „quantum“ nur im Namen (klassisch simulierbar), und das ehrlich zu sagen ist ein Vorbild für QIRE.
- **Quanten-Selbstaufmerksamkeit/Weiteres** (nur Titel): Quantum-Inspired Self-Attention in a LLM https://arxiv.org/html/2603.03318 ; Quantum Attention for ViT in HEP https://arxiv.org/pdf/2411.13520 ; Survey https://arxiv.org/html/2504.03192v1 ; QPSAN (Bild) https://ar5iv.labs.arxiv.org/html/2605.25365 .
- **QLSTM / Quantum Embeddings**: Suche wegen Budgetende nicht mehr möglich. Lücke, kein Befund [?].

### 2.3 Abgrenzung: quanteninspiriert (läuft klassisch)
- **CompactifAI** (Multiverse), https://arxiv.org/pdf/2401.14109 : Tensornetz-Zerlegung (MPO) der Schichten, komprimiert Korrelationsraum statt Neuronenzahl; **LLaMA-2 7B auf 30 % der Größe, >90 % der Genauigkeit nach kurzem verteiltem Nachtraining**; Firmenangaben: −70…80 % Parameter, −93 % Speicher, −50 % Trainingszeit, −25 % Inferenzzeit bei ~2 % Genauigkeitsverlust (Marketing, Quelle https://multiversecomputing.com/compactifai). Komprimierte Llama 3.1 8B / 3.3 70B, DeepSeek, Mistral (2025), Finanzierung 189 Mio. € (https://tech.eu/2025/06/12/multiverse-computing-lands-eur189m-to-rewire-the-llm-ecosystem/). Zusatzmeldung (nur Titel): „Recovers 83 % of compression loss in LLMs with quantum blocks“ [?].
- **QuanTA** (Chen et al., NeurIPS 2024), https://arxiv.org/pdf/2406.00132 : hochrangiges Fine-Tuning mit quantenschaltungsartigen Tensoroperationen, ohne Inferenz-Overhead; bis **0,041 % der Parameter** (DROP, Commonsense; LLaMA2/3). Strukturell QIREs nächster klassischer Verwandter (Zerlegung der Breite in Tensor-Achsen).
- **QuIC** Quantum-Inspired Compound Adapters https://awesomepapers.io/quantum-computing/papers/2502.06916 (nur Titel).
- **TT/MPS-Embeddings**: Tensorized Embedding Layers https://arxiv.org/pdf/1901.10787 (WikiText-103, 3,8× Kompression, +1,3 Test-Perplexity); TensorGPT https://arxiv.org/pdf/2307.00526 (bis 38,4× Embedding-Kompression; bei 3,31× besser als Original-GPT-2); TensorSLM https://arxiv.org/pdf/2506.13514 ; **Language Modeling Using Tensor Trains** https://arxiv.org/html/2405.04590 (TT-Sprachmodell, Details [?]).

### 2.4 Kerntabelle: Was wurde real erreicht?

| Arbeit | Größe (Modell/Quanten) | Daten | Baseline | Ergebnis | Hardware? |
|---|---|---|---|---|---|
| Duneau 2024 (QDisCoCirc) | Spielzeug, Qubits [?] | handgebaute QA-Daten | – (Machbarkeit) | Lauf auf H1-1 | ja, Ionenfalle |
| Quixer 2024 | q+⌈log₂n⌉+3 Qubits, Param. [?] | Penn Treebank | LSTM, Transformer, FNet | „competitive“ | nein (Simulation) |
| QSANN 2024 | klein [?] | öffentl. Textklassif. | QNLP-Syntax-Modell, klass. Self-Attn | besser | Simulation |
| QViT 2024 | bis 6 Qubits (HW) | med. Bilder | klass. ViT | konkurrenzfähig | ja, ≤6 Qubits |
| QPA 2025 | ⌈log₂m⌉ Qubits | Text-Generierung | LoRA | 52 % / 17 % der LoRA-Param., ±0,1–0,75 % | Simulation, Inferenz klassisch |
| QTHA 2025 | 7B-Modelle | 300–3000 Beispiele | LoRA | −76 % Param., bis +17 % | Simulation [?] |
| HyQuT 2025 | 10 Qubits/80 Gatter; 8M, 150M | Dialog | gleiches klass. Modell | ~10 % Param. ersetzt, vergleichbar | Simulation [?] |
| CUA 2026 | 8B (Llama 3.1); 135M (SmolLM2) | Sprachmodellierung (Datensatz [?]) | Llama ohne Adapter, 2 klass. Varianten | −1,4 % Perplexity, ~6.000 Param. | ja, IBM 156 Qubit |
| HyperQ 2026 | 16–64 Qubit in 1,1B | Downstream-Benchmarks | Backbone, LoRA | +4,71 / +3,67 Punkte | klassisch exakt simulierbar |
| CompactifAI 2024– | 7B–70B | Standard-Benchmarks | Original | 30 % Größe, >90 % Genauigkeit | klassisch |
| QuanTA 2024 | LLaMA2/3 | DROP, Commonsense | LoRA, Voll-FT | 0,041 % Param. | klassisch |

Lesehilfe: kein Eintrag zeigt Quantenvorteil; die größten Gewinne kommen aus Tensornetz-Struktur, die sich klassisch rechnen lässt.

## 3. Was das für QIRE bedeutet

1. **Einordnung.** QIRE-Adapter (Kronecker-Drehung, log₂ d Winkel je Matrix) liegt zwischen QuanTA/CompactifAI (quanteninspiriert, klassisch) und CUA (unitäre Quantenblöcke in eingefrorenen Projektionen). Unterschied zu CUA: QIRE hat pro Matrix nur log₂ d Winkel (z. B. d=768 → ~10), CUA ~6.000 Parameter über ein 8B-Modell. [EIGEN, Vergleich nur aus Auszügen; die genaue Parameterzahl pro Block bei CUA kenne ich nicht.]
2. **Maßstab setzen.** Dein 120M-Ergebnis („55 % des LoRA-Gewinns mit 1/16 der Parameter“) ist mit QPA (17–52 % der LoRA-Parameter bei gleicher Leistung) und QTHA (−76 %) direkt vergleichbar und nicht außergewöhnlich schlecht; LoRA-relative Parameterzahl ist das gängige Maß. Achtung: QPA/QTHA behaupten volle LoRA-Leistung, QIRE nur 55 % – ehrlich so berichten.
3. **Log-Wachstum.** Aussage „Parameterzahl wächst nur logarithmisch mit der Breite“ habe ich bei keiner gefundenen Arbeit gesehen (mit Vorbehalt: nur Auszüge, kein Volltext, Suche abgebrochen). QPA hat log-Qubits, aber Parameter über MLP. Das wäre ein Alleinstellungsmerkmal, muss aber in Etappe 2 mit gezielten Suchen abgesichert werden.
4. **Ehrlichkeit zu „Quanten“.** HyperQ zeigt: wenn die Schaltung klassisch exakt simulierbar ist, ist es Tensornetz/IQP-Struktur, kein Quantenvorteil. Ry+CNOT-Ring und Clifford-Zellen sind klassisch simulierbar (Clifford: Gottesman-Knill). QIRE sollte sich als **ressourceneffiziente strukturierte Orthogonal-/Kronecker-Parametrisierung mit Quantenbezug** positionieren, nicht als Quantenvorteil.
5. **Skalierung (Ziel 2).** „Unendlich skalierbar?“: Qubitzahl wächst log d, Winkelzahl log d je Matrix – aber Expressivität eines Kronecker-Produkts aus 2×2-Drehungen ist stark begrenzt (Rang/Struktur). CUA zeigt, dass die Leistung mit Blockgröße steigt (SmolLM2) – das spricht dafür, dass nur-log-Parameter irgendwo sättigen. Dein monotoner Abfall bis L=10 ist der passende Befund; Sättigungs-Ort ist offen.
6. **Born-Auslese.** In Quixer/QSANN wird gemessen und klassisch weiterverarbeitet; QIREs Born-Auslese |ψ|² ist eine quadratische, nichtnegative Nichtlinearität – verwandt mit Quixers QSVT-Nichtlinearität nur lose [EIGEN].

## 4. Neue Ideen / Forschungsfragen

Neuheitsprüfung: Nur wenige Suchen möglich (Budget). „Gesucht“ bedeutet hier: die oben aufgeführten Treffer geprüft. Alle Neuheitsurteile sind **vorläufig**.

### Idee A [EIGEN] – 0-Magie-Sprachadapter: Clifford-Suche statt Gradient
- **Frage:** Kann ein Sprachadapter, dessen Matrix-Drehungen auf Clifford-Winkel (k·π/2) beschränkt sind und diskret gesucht werden (0 T-Gatter), einen messbaren Teil des Adapter-Gewinns auf LLäMmlein-120M erreichen?
- **Warum neu:** Treffer (CUA, QPA, QTHA, QuanTA, HyperQ) trainieren alle kontinuierlich per Gradient. Diskret/Clifford in LLM-Adaptern habe ich nicht gefunden. Von Maltes MNIST-Befund (Clifford ≈ volles Training) übertragen. Neuheit: gesucht, nichts gefunden (Suche begrenzt).
- **Billigster Test:** An einer Matrix/Schicht des 120M-Modells: Winkel auf {0, π/2, π, 3π/2} rastern, Coordinate-Descent über log₂ d Winkel × L Lagen (Schleife ≈ 10·L·4 Auswertungen/Matrix), Verlust auf kleinem Kalibrierset. Laufzeit grob 10–60 min CPU je Matrix [EIGEN-Schätzung]. Gradientfreie Suche ist bei so wenigen Winkeln billig.
- **Kontrolle:** (i) kontinuierlich trainierter gleicher Adapter, (ii) zufällige Clifford-Winkel, (iii) Zufalls-Kronecker mit gleicher Parameterzahl.
- **Quantenstufe:** Q1 (klassisch simulierbar, Clifford); als Bonus Q2-Hardware-Lauf billig, da kein T.

### Idee B [EIGEN] – Magie-Budget eines Sprachadapters
- **Frage:** Wie viele Nicht-Clifford-Winkel (T-artige „Magie“) braucht ein Sprachadapter für X % des Gewinns? Kurve Verlust vs. Magie-Winkelzahl (0,1,2,4,…), analog „Fashion braucht ~4 Magiewinkel“.
- **Warum neu:** Magie als Ressourcenmaß in ML-Adaptern habe ich nirgends gefunden; vorhanden nur Qubit-/Parameterzahl als Ressourcen. Gesucht, nichts gefunden.
- **Billigster Test:** Aus Idee A starten, dann greedy 1,2,4… Winkel auf kontinuierlich lösen. Laufzeit ~1–2 h CPU für eine Matrix-Familie. Metrik: Stabilizer-Rényi-Entropie oder einfache Zahl der Nicht-Clifford-Winkel (Wahl der Metrik [?]).
- **Kontrolle:** Gleich viele Winkel an zufälligen Positionen; Gradient-Referenz.
- **Quantenstufe:** Q1 bis Q2 (Magie ist der Bezug zu Quanten-Ressourcentheorie; für Fehlerkorrektur relevante Größe).

### Idee C [EIGEN] – Born-Auslese als Attention
- **Frage:** Kann |ψ|² über Token-Positionen (Amplituden = Query-Key-Kompatibilität nach Drehung) als nichtnegative, normierte Attention-Gewichtung dienen, ohne Softmax?
- **Warum neu:** Quixer nutzt LCU/QSVT-Superposition, QSANN klassische Softmax-Koeffizienten, HyperQ Erwartungswerte. „Born-Regel als Attention“ ist in der Nähe von Quixer; direkte Gleichung nicht gefunden. Neuheit unsicher, möglicherweise in Quixer/Quantum-Inspired-Self-Attention (https://arxiv.org/html/2603.03318) schon angelegt [?] – in Etappe 2 prüfen.
- **Billigster Test:** Kleines Zeichenmodell (Shakespeare/PTB-Teil), 1 Kopf, Attention = |Wq x|²-normiert statt Softmax; vergleiche Perplexity mit Softmax-Attention bei gleicher Parameterzahl. Laufzeit ~10–30 min CPU.
- **Kontrolle:** Softmax-Attention; quadrierte ReLU-Attention (ohne Quantenbezug), um Quanten-Spezifisches von „quadratisch + normiert“ zu trennen.
- **Quantenstufe:** Q1 (klassisch gerechnet), Q2 möglich auf 8 Qubits.

### Idee D [EIGEN] – Skalierungsgesetz für log-Parameter-Adapter
- **Frage:** Wie skaliert der Adaptergewinn mit Modellbreite d, wenn nur L·log₂ d Winkel je Matrix genutzt werden? Sättigt er bei festem L?
- **Warum neu:** Skalierungsgesetze für Adapter existieren für LoRA-Rang, nicht für log-Parameter-Kronecker-Adapter (nicht gefunden). CUA zeigt Zuwachs mit Blockgröße, aber nicht in log-Skalierung.
- **Billigster Test:** Gleicher Adapter auf Modellen mit d = 256, 512, 768 (kleine GPT-2/SmolLM2-Varianten, soweit CPU-fähig); Evaluierung auf festem Set. Laufzeit Stunden, aber CPU-machbar mit kleinen Modellen.
- **Kontrolle:** LoRA Rang 1 (≈ 2d Parameter) als Referenz.
- **Quantenstufe:** Q1.

### Idee E [EIGEN] – Ehrlicher Klassifikator „wann ist es nicht-simulierbar?“
- **Frage:** Ab welcher Winkel-/Magiezahl und Kopplung verlässt ein QIRE-Adapter den klassisch effizient simulierbaren Bereich (Clifford + wenige T, IQP-Gerüst wie HyperQ)? Grundlage für Ziel (2) und (3).
- **Neuheit:** Eigene Einordnung anhand HyperQ; gesucht, nur dieser Verwandte gefunden.
- **Test:** Rechnung auf Papier plus Stabilizer-Rank-Abschätzung, Minuten bis Stunden.
- **Quantenstufe:** Q1→Q3 (Grenze).

## 5. Offene Fragen für Etappe 2
1. CUA-Volltext lesen (Datensatz, Qubits/Block, Parameter je Block, was „1,4 %“ genau misst, Hardware-Rolle). Gleiche Vergleichslinie für QIRE aufbauen.
2. Quixer: konkrete Perplexity- und Parameterwerte; bisher nur „competitive“.
3. QLSTM, Quantum Embeddings, DisCoCat-Alternativen (Bob Coecke/Quantinuum Folgearbeiten): nicht mehr gesucht, Lücke.
4. Gezielte Suche nach Adaptern mit log-Parameterwachstum (Stichworte: Kronecker/Butterfly/Givens-Produkt/Monarch/„Orthogonal Butterfly“ – BOFT/OFT verwandt, hier nicht geprüft!). Wichtig: **Butterfly-Orthogonal-Fine-Tuning (BOFT)** hat Parameterzahl O(d log d)-artig und ist wahrscheinlich der nächste klassische Verwandte von Kronecker/Ry-Lagen [?, aus Gedächtnis, nicht belegt].
5. Fundstellen zu Magie/Stabilizer-Rényi-Entropie in ML, Clifford-Training („Clifford circuit learning“, „Clifford+T neural“): Neuheit von Idee A/B absichern.
6. Prüfen, ob QPA/QTHA-Zahlen mit gleichen Datensätzen wie QIRE vergleichbar sind (Perplexity auf welchem Korpus?).
7. Quanteneffizienz im Großen: Energie-/Latenzzahlen von CompactifAI (außer Firmenangaben) unabhängig prüfen.

## 6. Quellenliste (alle nur Suchauszug, keine Volltexte)
- Quixer: https://arxiv.org/pdf/2406.04305 ; https://github.com/Quantinuum/Quixer ; https://thequantuminsider.com/2024/06/14/quantinuum-ai-teams-advance-quickens-pace-toward-merging-quantum-computing-and-ai/
- Duneau 2024: https://arxiv.org/abs/2409.08777 ; https://thequantuminsider.com/2024/09/18/quantinuum-unveils-first-contribution-toward-responsible-ai-uniting-power-of-its-quantum-processors-with-experimental-work-on-integrating-classical-quantum-computing/
- lambeq: https://arxiv.org/pdf/2110.04236 ; https://arxiv.org/html/2505.13208 ; https://arxiv.org/pdf/2408.06061
- QSANN: https://link.springer.com/article/10.1007/s11432-023-3879-7
- QViT: https://quantum-journal.org/papers/q-2024-02-22-1265/ ; https://arxiv.org/html/2504.03192v1
- HyQuT: https://arxiv.org/abs/2511.10653
- Quantum LLM Fine-Tuning: https://arxiv.org/abs/2504.08732
- QTHA: https://arxiv.org/abs/2503.12790
- QPA: https://arxiv.org/abs/2410.09846
- CUA: https://arxiv.org/pdf/2605.05914 ; https://phys.org/news/2026-06-quantum-circuits-ai-memory-limitations.html
- HyperQ: https://arxiv.org/abs/2609.24657v1
- QuanTA: https://arxiv.org/pdf/2406.00132
- QuIC: https://awesomepapers.io/quantum-computing/papers/2502.06916
- CompactifAI: https://arxiv.org/pdf/2401.14109 ; https://multiversecomputing.com/compactifai ; https://tech.eu/2025/06/12/multiverse-computing-lands-eur189m-to-rewire-the-llm-ecosystem/
- TT-Embeddings/LMs: https://arxiv.org/pdf/1901.10787 ; https://arxiv.org/pdf/2307.00526 ; https://arxiv.org/pdf/2506.13514 ; https://arxiv.org/html/2405.04590
- Quantum-Inspired Self-Attention in an LLM: https://arxiv.org/html/2603.03318
