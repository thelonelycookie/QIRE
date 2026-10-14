# Recherche: Chatbot-Idee "C1" — geteilter Butterfly-/Kronecker-Adapter auf kleinem deutschem Sprachmodell

Stand: 2026-09-28. Alle Angaben mit Quelle; Unsicheres ist markiert.

---

## 1. Kleine vortrainierte deutsche/mehrsprachige Modelle ≤ 1,5 B Parameter

| Modell | Größe | Deutsch | Instruct-Variante | Lizenz | Link |
|---|---|---|---|---|---|
| **LLäMmlein 120M / 1B / (7B)** | 120M, 1B (7B zu groß) | rein deutsch, from scratch auf RedPajama V2 (DE) trainiert, eigener dt. Tokenizer | ja, `LLaMmlein_1B_chat_all` existiert | offen, Code/Daten/Checkpoints veröffentlicht (genaue Lizenz auf HF prüfen) | [Paper arXiv:2411.11171](https://arxiv.org/html/2411.11171v2), [Collection](https://huggingface.co/collections/LSX-UniWue/llammlein), [GitHub](https://github.com/LSX-UniWue/LLaMmlein) |
| **BübleLM** | 2 B | Gemma-2-2B-Basis, per Trans-Tokenisierung auf dt. SentencePiece-Tokenizer (20k Vokabular) umgestellt; SFT-Variante mit LoRA auf mehreren dt. Instruktions-Datensätzen (alpaca-gpt4, openschnabeltier, evol_instruct, dolphin u. a.) | ja (`bueble-lm-2b-sft`) | Basis Gemma 2 → Gemma-Lizenz beachten (nicht vollständig „offen" wie Apache) | [johannhartmann/bueble-lm-2b-sft](https://huggingface.co/johannhartmann/bueble-lm-2b-sft) |
| **Qwen2.5-0.5B-Instruct** | 0,5 B | mehrsprachig (>29 Sprachen inkl. Deutsch), keine dedizierten DE-Qualitätsmetriken gefunden — **unsicher, wie gut das Deutsch wirklich ist** | ja, offizielle Instruct-Variante | Apache 2.0 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| **SmolLM2 (135M/360M/1.7B)** | bis 1,7 B | laut HF-Doku 6 Sprachen inkl. Deutsch — **unsicher, wie stark DE-Anteil im Trainingskorpus tatsächlich war**, primär EN-fokussiert (FineWeb-Edu, DCLM) | ja | Apache 2.0 (üblich bei HF SmolLM-Reihe, im Repo verifizieren) | [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B), [arXiv:2502.02737](https://arxiv.org/html/2502.02737v1) |
| **EuroLLM-1.7B-Instruct** | 1,7 B | 35 Sprachen inkl. Deutsch, EU-gefördert, Instruction-Tuning auf EuroBlocks | ja | Apache 2.0 | [utter-project/EuroLLM-1.7B-Instruct](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct) |
| **Teuken-7B** | 7 B (zu groß für die Zielgröße, aber als Referenz für dt. Qualität relevant) | >50 % nicht-englische Trainingsdaten, alle 24 EU-Sprachen, dt. Behörden-Projekt (BMWK/Fraunhofer) | ja (`Teuken-7B-instruct-research-v0.4`) | Apache 2.0 | [openGPT-X/Teuken-7B-base-v0.6](https://huggingface.co/openGPT-X/Teuken-7B-base-v0.6), [Paper arXiv:2410.03730](https://arxiv.org/pdf/2410.03730) |

**Einschätzung für C1:** LLäMmlein 1B (rein deutsch, offen, klein) und Qwen2.5-0.5B-Instruct (kleinste getestete Größe, aber DE-Qualität unklar) sind die aussichtsreichsten Kandidaten für CPU-Training. EuroLLM-1.7B ist die größte noch plausible Option und hat eine offizielle Instruct-Variante mit sauberer Apache-2.0-Lizenz.

---

## 2. Fertige Adapter-Implementierungen

### 2.1 HF PEFT: BOFT / OFT / LoRA — direkt nutzbar

Die HF-`peft`-Bibliothek enthält `BOFTConfig` fertig implementiert ([Doku](https://huggingface.co/docs/peft/main/en/package_reference/boft), [Paper arXiv:2311.06243](https://hf.co/papers/2311.06243)). Relevante Parameter:

- `boft_block_size` **oder** `boft_block_num` (nicht beide) — Produkt muss `in_features` der Ziel-Schicht ergeben. Größerer `boft_block_num` → **weniger** Parameter (sparsere Update-Matrix).
- `boft_n_butterfly_factor` — Anzahl Butterfly-Faktoren; bei `=1` entspricht BOFT dem einfachen OFT, bei `=2` verdoppelt sich der effektive Blockgröße bei halbierter Blockzahl.
- Butterfly-Struktur garantiert `O(d log d)`-Parameter statt `O(d²)` bei OFT — genau das gesuchte Prinzip für "geteilte Butterfly-Drehung, wenige hundert Parameter".
- Mit sehr großem `boft_block_num` (z. B. gleich der vollen Dimension) und kleinem `boft_n_butterfly_factor` lässt sich die Parameterzahl in den Bereich von wenigen hundert drücken — **das genaue Minimum ist nicht dokumentiert und muss empirisch bestimmt werden** (unsicher, erfordert eigene Rechnung/Tests).
- LoRA ist als Vergleichsarm ebenfalls direkt in PEFT (`LoraConfig`) verfügbar — für einen fairen Vergleich bei gleicher Parameterzahl `r` entsprechend klein wählen.

BOFT/OFT-Beispiele: [Dreambooth-BOFT](https://github.com/huggingface/peft/blob/main/examples/boft_dreambooth/boft_dreambooth.md), [BOFT-ControlNet](https://github.com/huggingface/peft/blob/main/examples/boft_controlnet/boft_controlnet.md) (Bild-Anwendungen, aber Konfigurationsmuster übertragbar auf Sprachmodelle über `target_modules`).

### 2.2 Quantum-PEFT (ICLR 2025)

- Paper: **Koike-Akino, Tonin, Wu, Wu, Candogan, Cevher**, "Quantum-PEFT: Ultra Parameter-Efficient Fine-Tuning", ICLR 2025. arXiv: [2503.05431](https://arxiv.org/abs/2503.05431), auch als MERL Tech Report [TR2025-051](https://www.merl.com/publications/docs/TR2025-051.pdf) verfügbar. **Existenz verifiziert.**
- **Zusammenfassung (3 Sätze):** Quantum-PEFT nutzt eine Pauli-Parametrisierung unitärer Transformationen, bei der die Zahl der trainierbaren Parameter nur **logarithmisch** mit der Modelldimension wächst (statt linear wie bei LoRA-Rang). Dadurch erreicht es bei wachsender Dimension eine noch geringere Parameterzahl als LoRA mit kleinstmöglichem Rang, bei voller Rang-Ausdrucksstärke der Transformation. Für C1 übernehmbar: das Prinzip einer strukturierten, parameterarmen unitären/orthogonalen Reparametrisierung als Alternative bzw. Ergänzung zu Butterfly-Faktoren — konkreter Code wurde in der Recherche **nicht auffindbar** (kein öffentliches GitHub-Repo von MERL gefunden; **Verfügbarkeit als Code unsicher**, ggf. Autoren direkt anfragen oder aus Paper-Formeln nachbauen).

### 2.3 QuIC — arXiv:2502.06916

- Titel: **"QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning"**, Autoren **Snehal Raj, Brian Coyle**. **Existenz verifiziert** ([abs](https://arxiv.org/abs/2502.06916), [PDF](https://arxiv.org/pdf/2502.06916)).
- **Zusammenfassung (3 Sätze):** QuIC-Adapter sind von Hamming-Gewicht-erhaltenden Quantenschaltkreisen inspiriert und erzwingen Orthogonalität in den Gewichtsparametern, um die vortrainierten Repräsentationen zu erhalten. Sie benötigen laut Paper unter 0,02 % des Speicherbedarfs des Basismodells und erreichen über 40-fach kleinere Parameterzahlen als LoRA, getestet u. a. an LLaMA-Modellen sowie Vision-Transformern. Für C1 übernehmbar: die Idee orthogonaler, kompakter "Compound"-Adapter als zweiter Vergleichs-/Inspirationsansatz neben BOFT — auch hier wurde **kein öffentliches Code-Repository gefunden** (unsicher, evtl. Supplementary Material im Paper direkt prüfen).

### 2.4 arXiv:2506.13605 — "High-expressibility Quantum Neural Networks using only classical resources"

- Autoren: **Marco Maronese, Francesco Ferrari, Matteo Vandelli, Daniele Dragoni** (Leonardo S.p.A.), eingereicht Juni 2025, überarbeitet März 2026, auch veröffentlicht in *Quantum Machine Intelligence* (Springer). **Existenz verifiziert.**
- **Zusammenfassung (3 Sätze):** Die Arbeit zeigt, dass Ausdrucksstärke ("Expressibility"), die üblicherweise parametrisierten Quantenschaltkreisen (PQCs) zugeschrieben wird, auch **klassisch simulierbar** durch Matrix-Produkt-Zustände (MPS) und Clifford-erweiterte MPS (CMPS) reproduziert werden kann. CMPS nähern sich der Haar-Verteilung (ein Maß für "maximale Zufälligkeit/Ausdrucksstärke") schneller an als reine MPS, sowohl bezüglich Verschränkung als auch "Magic" (ein Maß für Nicht-Klassikalität). Für C1 relevant als **Absicherung/Gegen-Argument**: Es untermauert die These des Projekts, dass "quanteninspirierte" Strukturen (wie die orthogonale Butterfly-Drehung) mit rein klassischer Hardware sinnvoll nachgebildet werden können, ohne echte Quantenhardware zu benötigen — nützlich als Zitat für den wissenschaftlichen Kontext der Arbeit.

---

## 3. Deutsche Instruktions-/Dialogdatensätze

| Datensatz | Größe | Lizenz/Herkunft | Link |
|---|---|---|---|
| **alpaca-data-german (perfood)** | ~50.000 Beispiele | Alpaca-Ansatz, generiert mit GPT-3.5 | [GitHub perfood/alpaca-data-german](https://github.com/perfood/alpaca-data-german) |
| **FreedomIntelligence/alpaca-gpt4-deutsch** | ~52.000 (Alpaca-GPT4-Standardgröße) | abgeleitet von Alpaca-GPT4, deutsch | [HF](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-deutsch) |
| **mayflowergmbh/alpaca-gpt4_de** | ähnlich | reformatierte Version für LLaMA-Factory | [HF](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de) |
| **FreedomIntelligence/evol-instruct-deutsch** | variabel (Evol-Instruct-Ansatz) | dt. Übersetzung von Evol-Instruct | [HF](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-deutsch) |
| **Zicklein (avocardio)** | Alpaca-Übersetzung via facebook/wmt19-en-de | Projekt zum Finetuning von LLaMA auf Deutsch | [GitHub](https://github.com/avocardio/Zicklein) |
| **OpenAssistant OASST1 (deutscher Anteil)** | Gesamt 161.443 Nachrichten, 35 Sprachen; deutscher Teil ca. **6.000 Samples** (unsicher — grobe Schätzung aus Quelle, nicht exakt nachgerechnet) | Apache 2.0, echte Mensch-Dialoge mit Qualitätsratings statt synthetisch | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/main) |

**Einschätzung:** Für C1 sind die Alpaca-GPT4-deutsch-Varianten (Apache/offene Lizenz, ~50k Beispiele, sauber strukturiert Instruction/Input/Output) am praktikabelsten. OASST1-DE ist kleiner, aber "echter" (Mensch-generiert, mit Bewertungen) — für einen zweiten, kleineren Validierungslauf interessant.

---

## 4. Rechenbedarf

- **CPU (Ryzen 5 3600, kein GPU):** Für PEFT-Finetuning (LoRA/BOFT mit wenigen hundert bis wenigen tausend trainierbaren Parametern) auf einem 0,5-B-Modell ist CPU-Training technisch möglich, aber **deutlich langsamer** als GPU — konkrete verlässliche Zeitangaben für exakt diese Kombination (0,5B + PEFT + Ryzen 5 3600) wurden in der Recherche **nicht gefunden**; die Schätzung unten basiert auf Analogieschlüssen und ist **unsicher**.
- **T4-GPU-Referenzwerte aus der Recherche** (nicht exakt für 0,5B/50k, aber als Anhaltspunkt):
  - QLoRA, 50k Alpaca-Beispiele, r=256/α=512: ~3 Std. auf **A100** (nicht T4) ([Quelle](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)).
  - 1,5B-Modell-Finetuning: ~30 Min. auf T4 (Datensatzgröße dort nicht spezifiziert).
  - 60M-Parameter-Modell mit LoRA auf T4: 184–229 Sekunden Trainingszeit ([arXiv:2607.25583](https://arxiv.org/pdf/2607.25583)).
  - **Grobe Interpolation (unsicher, eigene Schätzung):** Ein 0,5B-Modell mit PEFT-Adapter (BOFT/LoRA, wenige hundert bis wenige tausend Parameter) auf 50k Beispielen dürfte auf einer T4 **im Bereich von 1–3 Stunden pro Trainingslauf** liegen, je nach Epochenzahl, Sequenzlänge und Batchgröße. Auf CPU ist grob mit dem **10- bis 30-fachen** zu rechnen (mehrere Stunden bis über einen Tag) — reine Schätzung, sollte durch einen kurzen Testlauf auf einer kleinen Datenteilmenge kalibriert werden.
- **Kaggle-GPU-Kontingent 2026:** Aktuell **30 Stunden GPU pro Woche** (P100 oder T4, inkl. T4×2-Option), einzelne Sessions bis zu 9 Stunden, Reset wöchentlich (Sa. Mitternacht UTC). Kaggle testet zusätzlich ein "floating quota", das je nach Auslastung mehr als 30 Std. gewähren kann ([Kaggle-Diskussion](https://www.kaggle.com/discussions/product-feedback/173129), [Kaggle Docs](https://www.kaggle.com/docs/efficient-gpu-usage)). **Damit sind mehrere vollständige Trainingsläufe pro Woche realistisch machbar.**

---

## 5. Minimalplan bis 30.11.2026

1. **Modellwahl:** LLäMmlein-1B (rein deutsch, offen) als Hauptkandidat; Qwen2.5-0.5B-Instruct als schnellerer Zweitkandidat für schnelle Iterationen auf CPU. EuroLLM-1.7B als optionaler "großer" Vergleichspunkt, falls Kaggle-Kontingent reicht.
2. **Datensatz:** `FreedomIntelligence/alpaca-gpt4-deutsch` (~50k Beispiele, sauber, offen lizenziert) als Haupttrainingsdatensatz; kleine Teilmenge (z. B. 2–5k) für schnelle CPU-Testläufe und Hyperparameter-Suche; optional OASST1-DE als kleiner qualitativer Zusatz-/Validierungssatz.
3. **Adapter A (Kern-Idee C1):** HF `peft.BOFTConfig` mit `boft_block_num` groß und `boft_n_butterfly_factor` klein wählen, um auf wenige hundert Parameter zu kommen (Zielwert experimentell über `print_trainable_parameters()` von PEFT verifizieren).
4. **Vergleichsarm B:** HF `peft.LoraConfig` mit `r` so gewählt, dass die trainierbare Parameterzahl exakt (oder sehr nah) an Adapter A angeglichen ist.
5. **Ablauf:**
   - Schritt 1 (CPU, diese Woche): Pipeline mit winziger Datenteilmenge (~200 Beispiele) end-to-end zum Laufen bringen, für beide Adapter-Typen (Smoke-Test, keine langen Läufe, keine GPU-Reservierung nötig).
   - Schritt 2 (Kaggle, T4): Vollständiger Trainingslauf BOFT vs. LoRA bei gleicher Parameterzahl auf vollem Alpaca-DE-Datensatz (je Lauf ca. 1–3 Std. geschätzt, passt gut ins 30-Std.-Wochenkontingent auch bei mehreren Wiederholungen/Seeds).
   - Schritt 3: Zweiter Modell-Kandidat (z. B. EuroLLM-1.7B) mit demselben Setup wiederholen, falls Zeit reicht.
6. **Messgröße:** (a) Perplexity/Loss auf einem zurückgehaltenen deutschen Validierungssplit, (b) kleine kuratierte Menge deutscher Dialog-Prompts, von Menschen (Betreuer/Jury-nah) oder eines separaten LLM-Richters bewertet, (c) Parameterzahl beider Adapter exakt protokolliert (Kernvergleich der Arbeit: "gleiche Parameterzahl, welcher Ansatz erreicht bessere Qualität").
7. **Puffer:** Zwei bis drei Wochen vor dem 30.11. für Fehlerbehebung, Wiederholungsläufe und Schreiben des Berichts einplanen, da CPU-Zeiten unsicher sind und Kaggle-Kontingent gelegentlich schwankt.

---

## Offene / unsichere Punkte (Markierung wie gefordert)

- Exakte Mindest-Parameterzahl, die BOFT in PEFT bei extremer Block-Konfiguration erreicht — nicht dokumentiert, muss selbst gemessen werden.
- Öffentliche Code-Verfügbarkeit von Quantum-PEFT (MERL) und QuIC — in der Recherche kein GitHub-Repo gefunden; ggf. direkt bei Autoren/Paper-Supplement nachsehen.
- DE-Trainingsanteil von SmolLM2 und tatsächliche DE-Qualität von Qwen2.5-0.5B-Instruct — keine belastbaren Benchmark-Zahlen gefunden.
- Konkrete CPU-Trainingszeiten für 0,5B-PEFT auf einem Ryzen 5 3600 — keine Quelle gefunden, Schätzung im Bericht ist eine grobe Interpolation und sollte kalibriert werden.
- Größe des deutschen OASST1-Anteils (~6.000) — aus einer Sekundärquelle übernommen, nicht selbst nachgezählt.
