# T-004 — Der Chatbot: QIRE-Drehung als Adapter auf einem deutschen Sprachmodell

**Stand:** 28. September 2026, Entwurf, **vor** jeder Zahl. Das ist Weg C1 aus
`../BERICHT/01_VERBESSERUNGEN_UND_ZUKUNFT.md`. Die Recherche dazu steht in
`../BERICHT/RECHERCHE_2026-09-28/05_chatbot_adapter_c1.md`. Fremde Bausteine
(PEFT, BOFT, LoRA, LLäMmlein, alpaca-gpt4-deutsch) werden offen mitbenutzt und
im Quellenverzeichnis genannt.

## 1. Warum dieser Weg

- T-002 trainiert ein 256-dim Sprachmodell von null. Dabei entsteht kein
  brauchbarer Chatbot, und es kostet 59–93 GPU-h.
- Ein **vortrainiertes** deutsches Modell nur anzupassen löst den Faktor-64-
  Konflikt: Das Modell ist schon groß, und trainiert werden nur wenige Winkel.
  Dafür reichen die 30 GPU-h, die Kaggle pro Woche kostenlos gibt.
- **LLäMmlein** (Uni Würzburg, rein deutsch, offen) hat die versteckte Breite
  **2048 = 2¹¹** (beim 120M-Modell 768, das ist keine Zweierpotenz, siehe §6).
  Die QIRE-Schaltung passt also mit **11 Qubits** exakt, ohne Auffüllen.

## 2. Der Adapter

Die Gewichtsmatrix W (d_out × d_in) einer Zielschicht bleibt eingefroren. Gelernt
wird eine orthogonale Drehung R, und es gilt W′ = W · R, wie bei OFT/BOFT.

| Arm | R | Parameter je Matrix (d = 2048) |
|---|---|---|
| `qire_kron` | R₁ ⊗ … ⊗ R₁₁, reelle 2×2-Drehungen (QIRE ohne CNOT, 1 Lage) | **11** |
| `qire_5` | 5 Lagen aus Ry je Qubit + CNOT-Ring (als Permutation), geteilte Winkel | **55** |
| `boft` | PEFT `BOFTConfig`, kleinste lauffähige Konfiguration | wird gemessen (`print_trainable_parameters`) |
| `lora_r1` | PEFT `LoraConfig`, r = 1 | 4.096 |
| `basis` | kein Adapter, nur das Modell (Untergrenze) | 0 |

Alle Arme bei R = Identität initialisiert: Winkel 0, LoRA-B = 0. Das Modell
startet in jedem Arm also exakt als Basismodell.
**Zielschichten:** q_proj und v_proj aller Blöcke (Standardwahl bei LoRA).
`qire_*` ist ein eigenes kleines `nn.Module`, das R in O(d log d) anwendet
(Kronecker = 11 Tensor-Achsen-Drehungen). Es braucht keinen PEFT-Code, aber
dieselbe Einhängestelle.

## 3. Aufgabe, Daten, Messgröße

- **Daten:** `FreedomIntelligence/alpaca-gpt4-deutsch`, Instruktion → Antwort,
  im Chatformat von LLäMmlein-Chat.
  - Training 5.000 Beispiele
  - Validierung 1.000, fest, bei allen Seeds gleich
  - Prüfsatz 50 feste Fragen, nur zum Vorzeigen und für die Jury
- **Messgröße:** Kreuzentropie (nats/Token) **nur auf den Antwort-Tokens** der
  Validierungsmenge, bei gleicher Schrittzahl.
- **Vorregistrierte Familie** (Entwurf, freigegeben wird mit Zahlen aus dem
  Vorlauf):
  - **PRIMÄR:** `qire_kron` gegen `basis`, einseitig. Hilft die 11-Winkel-Drehung
    überhaupt?
  - **P2:** `qire_kron` gegen `lora_r1`: TOST auf Gleichwertigkeit, Grenze aus dem
    Vorlauf. Kommt man mit 372-fach weniger Parametern gleich weit?
  - **P3:** `qire_5` gegen `qire_kron`. Trägt die Verschränkung (CNOT) etwas bei?
    Das ist die Lehre aus Q-020 (b).
  - Holm über 3.
- `boft` wird nur beschreibend ausgewertet, als nächster Verwandter aus der
  Literatur.
- **Auflösungsprüfung:** `lora_r1` muss sich von `basis` trennen. Sonst sieht der
  Aufbau keine Anpassung, und kein Urteil ist deutbar.
- **Seeds:** 5 wären zu wenig für den Wilcoxon-Test (Mindest-p 0,0625). Deshalb
  8, nach Holm ×3 ist das kleinste p = 0,023.

## 4. Protokoll (CLAUDE.md §4)

- **Lernrate** je Arm auf eigenen Tuning-Seeds, volles Budget, Gitter in beide
  Richtungen.
- **Wiederaufnahme** nach jeder Zelle, eigene Datei je Lauf-ID.
- **Vortest** mit synthetischen Records, jeder Ausgang erzwungen, dazu die
  Mutationsprüfung.
- **Theorieprüfung** vor allem anderen:
  - `qire_kron` ist orthogonal (RᵀR = I, < 10⁻⁵).
  - Winkel 0 ergibt R = I.
  - `qire_kron` in d = 2048 ist gleich dem Kronecker-Produkt, explizit gebaut
    (d = 16 als Stichprobe).

## 5. Rechenplan

| Schritt | Wo | Was |
|---|---|---|
| 1 | eigener Rechner, CPU, 1 Thread | Rauchtest mit LLäMmlein-**120M**: Pipeline, Adapter, 20 Schritte (nur Durchlauf, keine Zahl) |
| 2 | Kaggle T4 | Zeitmessung 1B: Schritte/s je Arm → Budget festlegen |
| 3 | Kaggle T4 | Lernratensuche + 5 Arme × 8 Seeds. Grob 1–3 h je Zelle laut Recherche **unsicher**, deshalb erst Schritt 2 |
| 4 | eigener Rechner | Chat-Demo mit dem besten `qire`-Adapter (Inferenz auf CPU geht bei 1B) |

**Rückfallebene**, falls 1B auf Kaggle zu langsam ist: Qwen2.5-0.5B-Instruct.
Breite 896, keine Zweierpotenz, daher auf 1024 = 2¹⁰ auffüllen und das in der
Arbeit offen nennen. Oder LLäMmlein-120M: 768 → 1024.

## 6. Offen, vor der Freigabe (Malte)

1. Installation von `transformers`, `peft`, `datasets` in die venv und Download
   von LLäMmlein-120M (~0,5 GB) für den Rauchtest.
2. Kaggle-Konto vorhanden? Die Läufe laufen dort als Notebook.
3. Genaue Breiten von LLäMmlein-120M/1B aus der `config.json` bestätigen
   (hier aus der Recherche abgeleitet, **noch nicht geprüft**).
