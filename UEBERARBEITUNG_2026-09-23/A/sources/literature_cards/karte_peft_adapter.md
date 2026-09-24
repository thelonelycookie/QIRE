# Literaturkarte: Parameter-effizientes Finetuning (PEFT) — klassisch und quanteninspiriert

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts per Websuche
geprüft; bei BOFT und Quantum-PEFT zusätzlich die HTML-Fassung auf arXiv
(Parameterformeln). Volltexte noch nicht von Malte gelesen.

**PEFT** = man trainiert ein fertiges großes Modell nicht komplett nach, sondern nur
eine kleine Zusatzkomponente („Adapter“). Das ist heute *der* Weg, KI mit wenig
Rechenaufwand an neue Aufgaben anzupassen — und das dichteste Konkurrenzfeld für QIRE.

---

## Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2022) — LoRA
- **Titel:** LoRA: Low-Rank Adaptation of Large Language Models
- **Venue:** ICLR 2022
- **Link:** https://arxiv.org/abs/2106.09685 · https://openreview.net/forum?id=nZeVKeeFYf9
- **Was sie zeigt:** Vortrainierte Gewichte werden eingefroren; trainiert wird nur
  eine additive Niedrigrang-Änderung ΔW = B·A (2·d·r Parameter je Schicht). Bis zu
  10.000× weniger trainierbare Parameter bei gleicher Qualität auf GPT-3.
- **Für QIRE:** **Standard-Baseline** für jede Adapter-Richtung. Jeder QIRE-Adapter
  muss gegen LoRA mit **gleicher Parameterzahl** antreten.

## Qiu, Liu, Feng, Xue, Feng, Liu, Zhang, Weller, Schölkopf (2023) — OFT
- **Titel:** Controlling Text-to-Image Diffusion by Orthogonal Finetuning
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2306.07280
- **Was sie zeigt:** Statt additiver Änderung wird das vortrainierte Gewicht mit einer
  **orthogonalen** Matrix multipliziert; das erhält die „hypersphärische Energie“
  (paarweise Winkel zwischen Neuronen).
- **Für QIRE:** **Vorarbeit.** Eine unitäre QIRE-Schaltung als multiplikativer
  Adapter wäre eine OFT-Variante.

## Liu, Qiu, Feng, Xiu, Xue u. a. (2024) — BOFT
- **Titel:** Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization
- **Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2311.06243 · https://openreview.net/forum?id=7NzgkEdGyr · https://boft.wyliu.com/
- **Was sie zeigt:** Orthogonale Finetuning-Matrix als Produkt von m
  Butterfly-Komponenten (inspiriert von Cooley-Tukey-FFT), jeder Block eigene
  Cayley-parametrisierte Orthogonalmatrix. Parameterzahl laut HTML-Fassung
  **½·(b−1)·d·m**, mit b=2 und m=log d also O(d log d). **Keine Teilung** zwischen Blöcken.
  Getestet auf Vision-Transformern, LLMs und Diffusionsmodellen.
- **Für QIRE:** **Hauptkonkurrenz für einen QIRE-Adapter** (Butterfly + orthogonal +
  Finetuning). Beispiel d=256: BOFT(m=8, b=2) ≈ 1.024 Parameter, QIRE 16–80.
  In Hugging Face PEFT fertig verfügbar (siehe unten) — also billig als Baseline.

## Edalati, Tahaei, Kobyzev, Partovi Nia, Clark, Rezagholizadeh (2022) — KronA
- **Titel:** KronA: Parameter Efficient Tuning with Kronecker Adapter
- **Venue:** arXiv 2212.10650 (2022); ein Eintrag auf nips.cc/virtual/2023 existiert (vermutlich Workshop, nicht geprüft)
- **Link:** https://arxiv.org/abs/2212.10650
- **Was sie zeigt:** Ersetzt LoRAs ΔW = W_A·W_B durch ein **Kronecker-Produkt**
  ΔW = W_A ⊗ W_B — besseres Verhältnis von Rang zu Parameterzahl.
- **Für QIRE:** **Vorarbeit.** QIRE ohne CNOT = Kronecker-Produkt von 2×2-Faktoren,
  also der Extremfall von KronA mit n Faktoren.

## Chen u. a. (2024) — QuanTA
- **Titel:** QuanTA: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2406.00132 · Code: https://github.com/quanta-fine-tuning/quanta
- **Was sie zeigt:** Von Quantenschaltungen inspirierte Tensoren, die auf einzelnen
  Achsen der umgeformten Eingabe wirken („Gatter auf Qubits“) — ergibt
  **hochrangige** Anpassungen mit wenigen Parametern; gestützt durch
  Universalitäts- und Rangdarstellungssatz; besser als LoRA bei Reasoning-Aufgaben.
- **Für QIRE:** **Konkurrenz.** „Quanteninspiriert = Gatter auf Achsen eines
  umgeformten Vektors“ ist genau QIREs klassische Umsetzung, hier schon auf LLMs.

## Koike-Akino, Tonin, Wu, Wu, Candogan, Cevher (2025) — Quantum-PEFT
- **Titel:** Quantum-PEFT: Ultra parameter-efficient fine-tuning
- **Venue:** ICLR 2025
- **Link:** https://arxiv.org/abs/2503.05431 · https://mlanthology.org/iclr/2025/koikeakino2025iclr-quantumpeft/
- **Was sie zeigt:** Unitäre Parametrisierung per **RY-Drehungen und CZ-Verschränkern
  im Wechsel** über L Lagen auf q=log₂N Qubits. Parameterzahl laut HTML-Fassung
  **(2L+1)·log₂N − 2L** — logarithmisch statt linear wie LoRA. 5- bis 25-fach weniger
  Parameter als LoRA bei konkurrenzfähiger Qualität (DeBERTaV3 GLUE, GPT-2 E2E,
  Mistral-7B, ViT).
- **Für QIRE:** **Stärkste Konkurrenzarbeit überhaupt.** Das ist praktisch QIREs
  Schaltung (Ry + Verschränkerring, geteilte Winkel, O(log d) Parameter) — bei d=256
  und L=5 sind es 78 Parameter, QIRE hat 80. USP-Kandidat (a) ist damit **bekannt**.
  Unterschied zu QIRE: Quantum-PEFT nutzt die Schaltung als **lineare** unitäre
  Gewichtsänderung, QIRE als Schicht mit **Born-Auslese** (quadratische Nichtlinearität).

## Liu, Yang, Goan, Hsieh (2025) — Quantum Parameter Adaptation (QPA)
- **Titel:** A Quantum Circuit-Based Compression Perspective for Parameter-Efficient Learning
- **Venue:** ICLR 2025
- **Link:** https://arxiv.org/abs/2410.09846 · https://openreview.net/forum?id=bB0OKNpznp
- **Was sie zeigt:** Ein Quanten-Netz *erzeugt* die LoRA-Parameter nur während des
  Trainings; die Inferenz läuft klassisch. LoRA-Parameter auf 52 % (GPT-2) bzw. 17 %
  (Gemma-2) reduziert, Qualität gleich oder leicht besser.
- **Für QIRE:** **Konkurrenz** — anderer Mechanismus (Hypernetz), gleiche Zielgröße.

## Raj, Coyle (2025) — QuIC
- **Titel:** QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning
- **Venue:** arXiv 2502.06916 (v2 Okt. 2025)
- **Link:** https://arxiv.org/abs/2502.06916
- **Was sie zeigt:** Adapter aus Hamming-Gewicht-erhaltenden Schaltungen und
  Compound-Matrizen; laut Abstract über 40× kleiner als LoRA, mit mäßigem Qualitätsverlust.
- **Für QIRE:** **Konkurrenz** (quanteninspirierter Adapter, sehr wenige Parameter).

## Kong, Li, Chen u. a. (2025) — QTHA
- **Titel:** Quantum-Enhanced LLM Efficient Fine Tuning
- **Venue:** arXiv 2503.12790
- **Link:** https://arxiv.org/abs/2503.12790
- **Was sie zeigt:** Hybrid aus Quanten-Netz und Tensornetz als Adapter; 76 % weniger
  trainierbare Parameter als LoRA bei vergleichbarer Qualität (laut Abstract).
- **Für QIRE:** **Konkurrenz**; zeigt, wie voll das Feld „Quanten-PEFT“ 2025 ist.

## Hugging Face PEFT — OFT/BOFT-Dokumentation
- **Link:** https://huggingface.co/docs/peft/main/en/conceptual_guides/oft
- **Was es ist:** Die Standardbibliothek enthält LoRA, OFT und BOFT als fertige
  Konfigurationen (`LoraConfig`, `OFTConfig`, `BOFTConfig`).
- **Für QIRE:** **Werkzeug.** Faire Baselines kosten keine Eigenimplementierung.
