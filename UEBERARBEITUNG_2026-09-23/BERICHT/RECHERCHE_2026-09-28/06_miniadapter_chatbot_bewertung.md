# 06 Miniadapter, Chatbot-Bewertung, Lizenz (T-004)

Stand: 2026-09-29. Methode: WebSearch/WebFetch, Seiten teils nur als KI-Zusammenfassung des Abrufs gelesen. Markierung: [belegt] = in der Quelle so gefunden, [unsicher] = nicht direkt geprüft oder Zusammenfassung, [Schluss] = meine Folgerung, kein Beleg.

## 1. Extrem parameterarme Verfahren

### Überblick

| Verfahren | Idee | Parameterzahl (Belege) | Quelle |
|---|---|---|---|
| Quantum-PEFT (Pauli) | Unitäre aus Quantenschaltung (RY + CZ), Parameter wachsen ~logarithmisch mit Dimension | DeBERTaV3 GLUE: 0,013 M Parameter, 86,42 % Schnitt vs. LoRA r=1 mit 0,33 M und 86,12 %; GPT-2 Medium E2E: 0,098 M, BLEU 67,46 vs. LoRA 0,39 M, 66,88; Mistral-7B GLUE 0,758 M, 87,39 % vs. 3,54 M, 86,19 %; ViT CIFAR10 0,007 M, 98,46 % [belegt, Zusammenfassung] | [arXiv 2503.05431](https://arxiv.org/abs/2503.05431), [HTML](https://arxiv.org/html/2503.05431v1) |
| LoRA-XS | r×r-Matrix zwischen eingefrorenen SVD-Faktoren der Gewichte; r² je Modul, herunter bis 1 Parameter | GLUE RoBERTa-large: 24,6 K Parameter, 88,03 vs. VeRA 61 K, 87,83 [belegt]; laut Suchtreffer LoRA 1,57 M, VeRA 50,4 K, LoRA-XS 12,3 K (Setup nicht geprüft) [unsicher] | [arXiv 2405.17604](https://arxiv.org/html/2405.17604v3) |
| VeRA | eingefrorene Zufallsmatrizen, nur Skalierungsvektoren trainiert | s. o. (61 K bzw. 50,4 K) | im LoRA-XS-Paper mitverglichen |
| TinyLoRA (2026) | LoRA-XS + feste Zufallsprojektion + geteilte Gewichte, bis 1 Parameter | Qwen2.5-8B, GSM8K: 13 Parameter, 91 % mit RL, aber nur 83 % mit SFT bei gleichem Budget [belegt] | [arXiv 2602.04118](https://arxiv.org/html/2602.04118v1) |
| IA3 | lernt Skalierungsvektoren auf Aktivationen | ca. 0,01 % der Parameter bei T0-3B (also ~300 K) [belegt] | [PEFT-Doku](https://huggingface.co/docs/peft/v0.21.0/package_reference/ia3) |
| BitFit | nur Bias-Terme | Vorsicht: LLaMA-Architektur hat kaum Biases, also nicht direkt übertragbar [Schluss] | [ACL 2022](https://aclanthology.org/2022.acl-short.1/) |
| OFT / BOFT | orthogonale Blockdrehungen, BOFT mit Butterfly: O(d log d) statt O(d²) | Nicht unter 1.000 Parameter; BOFT-Ziel ist Effizienz gegenüber OFT [belegt] | [BOFT arXiv 2311.06243](https://arxiv.org/abs/2311.06243), [PEFT-Doku](https://huggingface.co/docs/peft/conceptual_guides/oft) |
| OFTv2 | Cayley-Neumann-Parametrisierung, matrixfreie Berechnung; bis 10x schneller, 3x weniger Speicher | Rechen-, nicht Parameterersparnis [belegt] | [arXiv 2506.19847](https://arxiv.org/abs/2506.19847) |
| Prompt-Tuning | trainierbare Soft-Prompt-Vektoren | typ. 20-100 Token × Einbettungsdim, also >10 K; bei kleinen Modellen bekanntlich schwach [Schluss, keine Quelle geholt] | – |

### „Was bringen ~200 Parameter?“ – ehrliche Antwort

- Keine Quelle gefunden, die ~200 Parameter auf einem ~100-M-Sprachmodell mit überwachtem Instruktions-Finetuning und Kreuzentropie misst. Direkte Vergleichszahlen fehlen [belegt durch Nichtfinden, [unsicher]].
- Nächste Belege: Quantum-PEFT ~7 K–13 K Parameter, kompetitiv mit LoRA r=1 (Encoder/ViT, Klassifikation, nicht Generierung). TinyLoRA 13 Parameter auf 8B, aber nur mit RL wirklich stark; mit SFT deutlich schwächer (83 %), und die Autoren sagen, größere Modelle brauchen kleinere Updates [belegt]. Ergebnisse nur Mathematik [belegt].
- [Schluss] Ein 120-M-Modell ist der ungünstige Fall (kleines Modell, SFT-Signal, generative Aufgabe). Erwartung: Verlustgewinn im Bereich weniger Promille bis wenige Prozent der Kreuzentropie, im Rauschen möglich. Das ist eine Hypothese, kein Beleg. Deshalb Seed-Streuung messen (siehe 3.).
- Quantum-PEFT verwendet mindestens L Verschränkungsschichten; Nutzen nimmt ab L=3 ab [belegt]. Das passt zur T-004-Variante mit CNOT-Ring (960 vs. 192).
- Hinweis zu deiner Zahl: LoRA r=1 mit 30.720 Parametern konnte ich nicht gegen die Modelldimension prüfen. Laut Paper hat das 120M-Modell 12 Layer, 12 Köpfe, 32.000 Vokabular [belegt, [arXiv 2411.11171](https://arxiv.org/html/2411.11171v2)]. Hidden-Größe bitte aus der config.json gegenrechnen.

## 2. Glaubwürdige, billige Bewertung eines kleinen deutschen Chatbots

1. Kreuzentropie/Perplexität auf Antwort-Tokens, held-out. Billig und exakt, aber: nur Vergleich derselben Tokenisierung/Daten zulässig; sagt nichts über Nützlichkeit. Immer mit Konfidenzintervall (Bootstrap über Beispiele, mehrere Seeds) berichten, dazu Basismodell und Nullbaseline (Zufallsrotation / nur Norm-Skalen).
2. Kleine menschliche Bewertung: 30-50 feste deutsche Prompts, blinde paarweise Präferenz (Basis vs. QIRE vs. LoRA), Reihenfolge randomisiert, 3+ Bewerter (Mitschüler), Übereinstimmung (Kappa) angeben. Dann ehrlich „nicht unterscheidbar“ berichten, falls so. [Schluss, Standardpraxis]
3. LLM-as-Judge nur als Zusatz. Risiken laut MT-Bench-Paper [belegt]: Positionsbias (Claude-v1 nur 23,8 % Konsistenz nach Tausch), Längenbias (Angriff mit repetitiver Liste bei Claude-v1/GPT-3.5 91,3 % Fehlerrate), Selbstbevorzugung (GPT-4 +10 %, Claude-v1 +25 % Siegrate); GPT-4 erreichte 85 % Übereinstimmung mit Menschen, aber Rechenfehler beim Bewerten. Quelle: [Zheng et al.](https://arxiv.org/html/2306.05685v4). Zudem englisch, für deutsche 120-M-Ausgaben nicht validiert [unsicher]. Gegenmaßnahmen: beide Reihenfolgen, Länge kontrollieren, Stichprobe gegen Menschen prüfen. Und Jury-Frage: Ausgaben eines 120-M-Modells sind oft inkohärent, ein Judge-Score wäre wenig aussagekräftig.
4. Deutsche Benchmarks: SuperGLEBer (29 Aufgaben, Klassifikation, Tagging, Ähnlichkeit, QA; [NAACL 2024](https://aclanthology.org/2024.naacl-long.438/)) ist stark auf Encoder-/Verständnis ausgelegt; das Paper nennt für 120M z. B. NLI 0,629, WebCAGe 0,687 [belegt]. Ob ein 192-Parameter-Adapter dort etwas verändert, ist fraglich; als „verschlechtert nichts“-Kontrolle brauchbar [Schluss]. Ein deutscher MT-Bench-Ersatz wurde nicht verifiziert [unsicher]; nicht behaupten.
5. Für die Jury: Kernaussage sollte die saubere Messung mit Streuung sein, nicht ein Chatbot, der „gut antwortet“. Qualitative Beispiele (5 feste Prompts, Basis vs. Adapter nebeneinander) zeigen, nichts herauspicken (vorab festlegen).

## 3. Verbesserungsideen für T-004 (CPU-tauglich)

Alle [Schluss] aus dem Stand der Literatur, nicht auf diesem Modell geprüft.

- Initialisierung: Winkel exakt 0 (Identität), sodass Start = Basismodell. Bei Quantum-PEFT ist Pauli/RY-Struktur so gebaut; kleine Zufallsstartwinkel (~0,01) nur als Variante. Nullgradient-Fallen prüfen: bei identischer Init muss jeder Winkel einen Gradienten ≠ 0 haben (Sanity-Check am ersten Schritt).
- Lernrate: bei ≤1.000 Parametern deutlich höher als LoRA-Standard testen (Raster z. B. 1e-3, 1e-2, 3e-2, 1e-1), da jeder Parameter viele Gewichte bewegt. Kleines Raster mit 200 Beispielen, vor dem vollen Lauf.
- Zielmodule: Drehung nur auf q_proj/v_proj ist ungünstig, wenn dieselbe Drehung auf q und k das Skalarprodukt unverändert ließe (Invarianz). Nur q zu drehen ändert die Aufmerksamkeit, nur v ändert den Wert-Kanal; als Ablation q allein, v allein, o_proj, down_proj vergleichen. Rotation auf v/o_proj folgt der Idee von OFT.
- Skalierung zusätzlich: IA3-artige Skalen auf v (nur wenige hundert Parameter) als Vergleichsbaseline auf gleichem Budget. Das ist der ehrliche Gegner für „Drehung“.
- Kombination QIRE+LoRA: erst QIRE-Drehung trainieren und einfrieren, dann LoRA r=1 obendrauf; oder gleiche Trainingszeit, gleiche Daten. Frage: bringt die Drehung über LoRA hinaus etwas? Nachbau der LoRA-XS-Idee (SVD-Init, r×r-Kern) als zweite Vergleichsmethode ist mit r=2 nur 4 Parameter je Matrix, also fast gleiches Budget wie 8 Parameter.
- Rauschboden: 3-5 Seeds, gleiche Datenreihenfolge variiert; Verlustgewinn nur „real“, wenn größer als Seed-Streuung. Val-Loss auch gepaart pro Beispiel (Bootstrap) berichten.
- Skalierung mit Parametern: Kurve 0/8/16/32/96/960 Parameter je Matrix plotten (Quantum-PEFT-Stil), das ist die Jury-taugliche Grafik.
- CPU: nur wenige Schichten testweise, Sequenzlänge kürzen, Batch klein; Aktivationen nicht speichern, die Orthogonalität der Kronecker-Form macht die Anwendung billig (Faktorenweise statt volle Matrix).

## 4. Rechtliches: RAIL-M research-only

Kein Rechtsrat. Quelle: [Modellkarte 120M](https://huggingface.co/LSX-UniWue/LLaMmlein_120M), [license.md](https://huggingface.co/LSX-UniWue/LLaMmlein_120M/raw/main/license.md), gelesen nur als Abrufzusammenfassung, Wortlaut unbedingt selbst gegenlesen.

- Erlaubter Zweck laut Lizenz: „For academic or research purposes only“ (Abschnitt 1(m)) [belegt]. Kommerzieller Einsatz ausgeschlossen [belegt].
- Hosting: Abschnitt 6 erlaubt ausdrücklich Hosting für Dritte (SaaS) und Weitergabe, sofern Lizenzkopie beigelegt und Nutzungsbeschränkungen weitergegeben werden und der Zweck Forschung bleibt [belegt, Zusammenfassung]. Online-Demo ist also nicht automatisch verboten, aber Folge: Nutzer müssen an research-only gebunden werden; ein offen zugänglicher Chatbot für Beliebige ist heikel [Schluss].
- Anhang A verbietet u. a. Diskriminierung, Militär, Rechtsentscheidungen, Desinformation, Datenschutzverletzungen, klinische Entscheidungen [belegt, Kategorien]. Für einen Demo-Chatbot: Hinweis, dass Ausgaben unzuverlässig sind, keine personenbezogenen Eingaben.
- Jugend forscht ist ein Wettbewerb mit wissenschaftlichem Ziel; lokale Demo am Stand auf eigenem Laptop ohne Weitergabe der Gewichte ist am unkritischsten [Schluss]. Wenn Sponsoren/Preise: „research“ bleibt plausibel, aber nicht kommerziell verwerten (kein Produkt, keine Werbung).
- Die trainierten Adapter (192 Parameter) sind eine „Derivative“ des Modells; bei Weitergabe (GitHub) Lizenz mitgeben, Gewichte des Basismodells nicht selbst hochladen, nur Verweis [Schluss].
- Widerspruch beachten: Das LLäMmlein-Paper (v2) nennt für die instruction-tuned Varianten CC BY-NC-ND 4.0 [belegt, [arXiv v2](https://arxiv.org/html/2411.11171v2)], die Modellkarten nennen RAIL-M. Maßgeblich ist die Lizenz der genutzten Gewichte; bei Zweifel Autoren (Uni Würzburg, LSX) anschreiben. ND (keine Bearbeitungen) würde Finetuning-Weitergabe einschränken [unsicher].
- Jugend forscht: Teilnahmebedingungen verlangen eigenständige Arbeit und Angabe aller Quellen und Unterstützer; KI wird nicht ausdrücklich genannt [belegt, [Teilnahmebedingungen](https://www.jugend-forscht.de/teilnahme/wichtige-infos/teilnahmebedingungen.html)]. Die Schweizer Variante verlangt Angabe von KI-Tools; auch die Stiftung hat ein „Leitbild zum Umgang mit KI“ [Suchergebnis, nicht gelesen, [unsicher]]. Empfehlung: KI-Unterstützung (auch dieses Recherche-/Coding-Werkzeug) und Fremdmodelle/-daten (LLäMmlein, alpaca-gpt4-deutsch inkl. dessen Lizenz) in der Arbeit ausweisen. Lizenz von alpaca-gpt4-deutsch nicht geprüft [unsicher]; die Daten stammen von GPT-4-Ausgaben, dessen Nutzungsbedingungen könnten Einschränkungen enthalten [unsicher].
- Empfehlung: Demo lokal am Stand, kein öffentliches Online-Hosting, Lizenz und Quellenangaben auf dem Poster.

## Quellenliste
- https://arxiv.org/abs/2503.05431
- https://arxiv.org/html/2405.17604v3
- https://arxiv.org/html/2602.04118v1
- https://huggingface.co/docs/peft/v0.21.0/package_reference/ia3
- https://aclanthology.org/2022.acl-short.1/
- https://arxiv.org/abs/2311.06243
- https://arxiv.org/abs/2506.19847
- https://arxiv.org/html/2306.05685v4
- https://aclanthology.org/2024.naacl-long.438/
- https://arxiv.org/html/2411.11171v2
- https://huggingface.co/LSX-UniWue/LLaMmlein_120M
- https://huggingface.co/LSX-UniWue/LLaMmlein_120M/raw/main/license.md
- https://www.jugend-forscht.de/teilnahme/wichtige-infos/teilnahmebedingungen.html
