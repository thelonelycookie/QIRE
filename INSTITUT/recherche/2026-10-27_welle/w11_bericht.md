# w11 — Stand der Forschung ressourceneffiziente KI 2024–2026 und Schnittstellen zu QIRE

Erkundung, kein Befund, nichts gerechnet. 5 WebSearch (nur Auszüge, kein Volltext). Alles ohne Marke [Web] ist [Gedächtnis] und vor Zitat zu prüfen. Wirkung klassisch (I-038): "quanten-inspiriert".

## 1. Kurzfassung
- Ampel gesamt: GELB. Belegte Schnittstellen sind Rotationen/Hadamard für Quantisierung und ReLU²-Sparsität; die rein "quantenartigen" Bausteine (Born-Quadrat, Phasen) haben wenig direkten Beleg.
- Stärkste Anknüpfung: Hadamard-Rotation vor Quantisierung (QuaRot/SpinQuant) und ReLU² (Primer, BitNet 2B4T, ReLU² Wins). Beides ist FLOP-ehrlich testbar.
- Lehre I-045-v4 (nur gleiche FLOPs zählen) ist hart: Fastfood spart Parameter, nicht FLOPs. Ideen müssen daher bei Bytes/Bits (Quantisierung) oder bei echter Aktivierungssparsität FLOPs sparen.
- Aussichtsreichster kleinster Test: K1 (Hadamard-Rotation + ternäre/4-Bit-Gewichte im Byte-LM), dann K2 (ReLU²-Sparsität als FLOP-Hebel).
- KAN, MoE, Early Exit: kaum Quantenbezug, KAN fair verglichen meist unterlegen [Web].

## 2. Themen: Wo hilft ein quanten-inspiriertes Element?

### 2.1 BitNet b1.58 / ternäre LLMs
- BitNet b1.58 2B4T: ~2 Mrd. Parameter, 4 Bio. Token, W1.58A8 (ternär, Aktivierung 8 Bit, absmax pro Token), BitLinear, ReLU² in der FFN, RoPE, ohne Bias [Web, arXiv 2504.12285]. Original BitNet b1.58 (Ma et al. 2024, 2402.17764) [Gedächtnis].
- Quanten-Element: Hadamard-Transform vor der Aktivierungsquantisierung. BitNet v2 (H-BitLinear, 4-Bit-Aktivierungen) und QuaRot/SpinQuant [Web: QuaRot 2405.16406; SpinQuant gelernte Rotation, bis 13 Punkte Spreizung zwischen Zufallsrotationen, schließt Lücke um 30,2 %/34,1 % gegenüber QuaRot bei LLaMA-2 7B/LLaMA-3 8B]. Beleg: stark (Ausreißer werden verschmiert, Rotation ist verlustfrei im Fließkomma).
- Passung QIRE: feste Hadamard·Diagonal-Mischung ist schon da (ternaer_butterfly.py). Gewinn liegt in Bits/Bytes, nicht in der Mischung selbst.
- Phasen/Unitarität: ternäre Gewichte mit Rotation = nahe an orthogonaler Parametrisierung; kein direkter Beleg.

### 2.2 State-Space-Modelle (Mamba, RWKV, LRU)
- Mamba/Mamba-2 (Gu/Dao 2023/24), RWKV-7 (arXiv 2503.14456, Diagonal minus Rang-1 Korrektur, Delta-Regel-Verwandtschaft) [Web]; LRU (Orvieto 2023, komplexe diagonale Eigenwerte auf dem Ring |λ|<1, Phase gelernt) [Gedächtnis]; DeltaNet / Gated DeltaNet [Gedächtnis].
- Neuere Hinweise [Web, nur Auszüge]: MuonSSM "Orthogonalizing State Space Models" (2606.30461); Block-diagonale/höherer Ordnung (2602.12021); Vergleichsstudie assoziativer Abruf bei fester Zustandsgröße (2609.16183, "Interferenzwand").
- Quanten-Element: Unitarität/Phase der Zustandsmatrix (stabiles Langzeitgedächtnis, keine Normverluste). Beleg: Orthogonale RNNs (Arjovsky uRNN 2016) helfen bei langen Abhängigkeiten, bei LLM-Skala gewinnen aber selektive, nicht-unitäre Zerfälle (Mamba). Beleg für Unitarität im Sprachmodell: gemischt/schwach.
- Passung: kleines Byte-LM mit LRU-artiger Phasen-Rekurrenz gegen diagonale reelle Rekurrenz bei gleichen FLOPs.

### 2.3 Mixture-of-Experts
- Mixtral, DeepSeekMoE/V3, OLMoE [Gedächtnis]. Vorteil: aktive FLOPs << Parameter; echter FLOP-Hebel.
- Quanten-Element: kaum. Möglich: Router als Born-Wahrscheinlichkeit (|α|², Softmax-Ersatz); feste Hadamard-Mischung als billige Expertenbasis. Kein Beleg; Router-Qualität und Lastverteilung dominieren.

### 2.4 KAN
- Fairer Vergleich (Yu et al. 2024, arXiv 2407.16674): KAN gewinnt nur bei symbolischer Formeldarstellung, sonst MLP meist besser, bei gleichen Parametern mehr Operationen [Web]. Tabellen-Benchmark ähnlich [Web 2406.14529].
- Quanten-Element: Fourier/Sinus-Basen (SineKAN 2407.04149 [Web]); Walsh-Basis ist eine naheliegende Variante, Beleg dafür nicht gefunden. Passung zu Fastfood: Basisfunktionen = feste Transformation.
- Erwartung gemäß I-045-v4: KAN verliert gegen dicht bei gleichen FLOPs. Niedrige Priorität.

### 2.5 Sparse/strukturierte Aktivierungen (ReLU²)
- Primer (So et al. 2021, squared ReLU) [Gedächtnis]; ReLU² Wins (arXiv 2402.03804: beste Wahl bei Sparsität vs. Güte, Vorhersagbarkeit, Hardwarenähe); ReLU Strikes Back (2310.04564); Q-Sparse (2407.10969, Top-K mit ReLU²); Meta 2:4-Aktivierungssparsität (2503.16672: FFN-Aktivierungen 84–98 % dünn, bis 1,3× schnellere FFN) [alle Web].
- Quanten-Element: Born-Quadrat |z|² ist formal ReLU²-verwandt (Quadrat) plus Vorzeichenverlust. Das ist die ehrlichste Brücke zu QIRE: Quadrat-Auslese hat in der Literatur realen Nutzen, aber als Aktivierungssparsität, nicht als Quanteneffekt.
- FLOP-Hebel: Sparsität in der zweiten FFN-Matrix spart echte FLOPs, falls nutzbar (2:4 bzw. Top-K).

### 2.6 Distillation kleiner Modelle
- Gemma 2/3, Phi-3/4, MiniLLM, Minitron (Pruning+Distillation) [Gedächtnis]. Keine direkte Quantenschnittstelle. Mögliche Kombination: Struktur-Schüler (Fastfood/ternär) vom dichten Lehrer lernen lassen. Beleg dafür für strukturierte Matrizen gering, aber plausibel, weil gelernte Struktur Trainingsrauschen mindert.

### 2.7 Test-Time-Compute / Early Exit
- Early Exit (CALM, LayerSkip 2024), spekulatives Decoding, Mixture-of-Depths [Gedächtnis]. Quanten-Element: Born-Wahrscheinlichkeit als Konfidenz; Amplituden-Verstärkung als Metapher. Kein Beleg; Entropie-Schwelle genügt. Echter FLOP-Hebel, aber unabhängig von QIRE.

## 3. Top 8 Kombinationen mit kleinstem Test
Basis für alle: `abteilungen/struktur_sprache.py` (Byte-LM UD_German-GSD, bpc), kleine Modelle (d=128–256), 1 Thread, Tuning-Seeds 900–999, ≥3 Seeds, gleiche FLOPs UND Parameter, ausreichend langes Training (I-045-v3: Vorsprünge verschwinden sonst).

| # | Kombination | Hypothese | Kleinster Test | Erwartung/Beleg | Raven? |
|---|---|---|---|---|---|
| K1 | Hadamard-Rotation vor Quantisierung (ternär/4 Bit) in FFN | Rotation senkt Quantisierungsverlust bei gleichen Bits | Dichtes Byte-LM trainieren, danach post-training ternär/4 Bit mit/ohne feste Hadamard·Diag (QuaRot-Stil); Δbpc | Stark (QuaRot, SpinQuant) | nein, ~5 min/Seed |
| K2 | ReLU² vs. GELU mit Top-K/2:4-Sparsität in der FFN | ReLU² hält Güte bei 50–75 % Aktivierungssparsität; FLOPs wirklich sparbar | Byte-LM, d=256, Sparsitätsgrade 0/50/75 %, FLOPs zählen (dünn gezählte MACs) | Gut (ReLU² Wins, Q-Sparse, Meta 2:4) | nein |
| K3 | BitNet-QAT (ternär) + feste Hadamard-Mischung statt zweiter Matrix | Ternär+Fastfood bringt Bytes/Param-Gewinn, bpc-Verlust klein | I-045-v4 Rahmen: Bytes-gleiche Baselines (dicht 4 Bit, Niedrigrang) | Mittel; spart Bytes, nicht FLOPs | ja für lange Läufe |
| K4 | LRU-artige Phasen-Rekurrenz vs. reelle diagonale Rekurrenz | Komplexe Eigenwerte (Phase) verbessern Langzeitkontext bei gleichen FLOPs | 1-Schicht-Rekurrenz-LM, Kontext 256 vs. 1024 Bytes, bpc | Schwach/gemischt | nein |
| K5 | Orthogonale (Cayley/Hadamard-Block) Zustandsmatrix im SSM | Unitarität stabilisiert, Rang-1-Korrektur à la DeltaNet ergänzt Speicher | Assoziativer-Abruf-Spielaufgabe (synthetisch) + Byte-LM | Schwach (MuonSSM, Auszüge) | nein |
| K6 | MoE mit Hadamard-gemischten Mini-Experten, Born-Router | Aktive FLOPs sinken bei gleicher Güte | 4 Experten, Top-1, gleiche aktive FLOPs wie dichtes FFN | Mittel für MoE, kein Beleg für Born-Router | nein |
| K7 | Strukturierter Schüler (Fastfood/Niedrigrang) per Distillation vom dichten Lehrer | Distillation schließt Lücke aus I-045-v4 bei gleichen FLOPs | Lehrer dicht d=256, Schüler gleiche FLOPs, KL-Verlust | Plausibel, kein direkter Beleg | nein |
| K8 | Early Exit mit Born-Konfidenz (|logit|²-normiert) vs. Entropieschwelle | Gleiche Güte bei weniger mittleren FLOPs | Byte-LM 4 Schichten, Ausstieg nach 2 bei Schwelle; mittlere FLOPs gegen bpc | Gut für Early Exit, Quantenanteil null | nein |

Reihenfolge nach Aussicht: K1 > K2 > K3 > K7 > K8 > K6 > K4 > K5.

## 4. Deutung und Vorbehalte
- Langweiligste Erklärung für K1/K2: Es sind klassische Standardtricks (Rotation glättet Ausreißer; ReLU erzeugt Nullen). Das ist kein Quanteneffekt; QIRE liefert nur die vertraute Mathematik (Hadamard, Quadrat). Formulierung: "quanten-inspiriert".
- Zitierbar ist, dass Hadamard und ReLU² Teil der Praxis sind; das stützt Maltes Ziel (Neuheit nicht nötig).
- FLOP-Ehrlichkeit: Quantisierung spart Bytes und Energie je Rechenoperation, nicht die Zahl der Operationen. Daher zwei Achsen berichten: FLOPs und Bytes.
- Quellenlage: Auszüge nur; neuere arXiv-IDs (26xx) nicht inhaltlich geprüft. BitNet-Hadamard (BitNet v2) und LayerSkip nur aus dem Gedächtnis.

## 5. Vorschlag für vorregistrierten Lauf (nach Maltes Zustimmung)
- Name: "K1+K2 Rotation/Sparsität im Byte-LM". Hypothesen: H1 Hadamard-Rotation senkt Δbpc der 4-Bit- und ternären Quantisierung gegenüber ohne Rotation um ≥ 0,03 bpc; H2 ReLU² mit 50 % Top-K-Sparsität liegt ≤ 0,02 bpc hinter dicht-GELU bei ≤ 60 % der FFN-FLOPs.
- Arme: dicht-GELU, dicht-ReLU², +Top-K 50/75 %, jeweils fp32 / 4 Bit / ternär, mit und ohne feste Hadamard·Diag; Kontrolle Zufallsorthogonal (Haar) statt Hadamard.
- Seeds: 10 Mess-Seeds, Effektgröße 0,03 bpc, TOST für Gleichwertigkeit Hadamard vs. Haar. Laufzeit: Container ca. 2–3 h (d=128, T≈8000), Raven ca. 30 min für d=512.

## 6. Skriptpfade
Keine Skripte, nichts gerechnet. Bausteine: `INSTITUT/abteilungen/struktur_sprache.py`, `INSTITUT/abteilungen/ternaer_butterfly.py`, `INSTITUT/abteilungen/struktur_kopf_fair.py`.

## BibTeX (aus Suchauszügen bzw. Gedächtnis, vor Nutzung prüfen)
```
@article{bitnet2b4t, title={BitNet b1.58 2B4T Technical Report}, journal={arXiv:2504.12285}, year={2025}}
@article{bitnet158, title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits}, author={Ma, Shuming and others}, journal={arXiv:2402.17764}, year={2024}}
@article{quarot, title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs}, author={Ashkboos, Saleh and others}, journal={arXiv:2405.16406}, year={2024}}
@article{spinquant, title={SpinQuant: LLM Quantization with Learned Rotations}, author={Liu, Zechun and others}, journal={arXiv:2405.16406 (ID prüfen)}, year={2024}}
@article{relu2wins, title={ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs}, journal={arXiv:2402.03804}, year={2024}}
@article{relustrikes, title={ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models}, journal={arXiv:2310.04564}, year={2023}}
@article{qsparse, title={Q-Sparse: All Large Language Models can be Fully Sparsely-Activated}, journal={arXiv:2407.10969}, year={2024}}
@article{sparse24, title={Accelerating Transformer Inference and Training with 2:4 Activation Sparsity}, journal={arXiv:2503.16672}, year={2025}}
@article{primer, title={Primer: Searching for Efficient Transformers for Language Modeling}, author={So, David R. and others}, journal={arXiv:2109.08668}, year={2021}}
@article{rwkv7, title={RWKV-7 "Goose"}, journal={arXiv:2503.14456}, year={2025}}
@article{lru, title={Resurrecting Recurrent Neural Networks for Long Sequences}, author={Orvieto, Antonio and others}, journal={arXiv:2303.06349}, year={2023}}
@article{mamba, title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces}, author={Gu, Albert and Dao, Tri}, journal={arXiv:2312.00752}, year={2023}}
@article{uRNN, title={Unitary Evolution Recurrent Neural Networks}, author={Arjovsky, Martin and others}, journal={arXiv:1511.06464}, year={2016}}
@article{kanfair, title={KAN or MLP: A Fairer Comparison}, author={Yu, Runpeng and others}, journal={arXiv:2407.16674}, year={2024}}
@article{sinekan, title={SineKAN}, journal={arXiv:2407.04149}, year={2024}}
@article{muonssm, title={MuonSSM: Orthogonalizing State Space Models for Sequence Modeling}, journal={arXiv:2606.30461}, year={2026}}
@article{fastfood, title={Fastfood: Approximating Kernel Expansions in Loglinear Time}, author={Le, Quoc and Sarl{\'o}s, Tam{\'a}s and Smola, Alex}, journal={ICML}, year={2013}}
```
Hinweis: Die SpinQuant-ID oben ist ein Platzhalter und muss geprüft werden (sie wurde fälschlich mit QuaRot gleichgesetzt).
