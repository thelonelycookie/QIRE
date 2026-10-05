# Forschungsplan 05.10.2026 — „MNIST mit ~100 Parametern“ und „Skaliert ein QLM?“

*Entwurf zur Freigabe durch Malte. Nichts davon ist vorregistriert oder gestartet.
Grundlage: zwei Recherchen vom 05.10. (Sonnet-Agenten; arXiv-Volltexte waren
gesperrt, Literaturzahlen also nur aus Suchauszügen — vor Zitat prüfen).*

---

## 1. Was die Recherche ergeben hat

### 1a. Wo die Parameter wirklich stecken

| Modell | Schaltung | Kopf | Encoder | **gesamt** | Genauigkeit |
|---|---|---|---|---|---|
| Institut-QIRE-Zelle (I-005-v2, 16×16) | 32 Winkel | 257×10 = 2.570 | 0 (Amplitudenkodierung, fest) | **≈ 2.600** | 0,940 |
| Clifford-Suche (0 Magie) | 0 trainiert | 2.570 | 0 | **2.570** | 0,937–0,940 |
| Q-020 `q1` („16 Winkel schlagen MLP“) | 16 | 90 | **200.960** | **≈ 201.000** | 0,935 |

→ **99 % der Parameter sitzen im Kopf (Institut) bzw. im Encoder (Q-020).**
Die Schaltung selbst ist schon winzig. Der Hebel für „~100 Parameter“ ist
also die **Auslese**, nicht die Schaltung.

→ Ehrlichkeitspunkt: „16 Winkel schlagen ein MLP“ (Q-020) zählt einen
Encoder mit 200.960 Parametern nicht mit. In der schriftlichen Arbeit
immer **alle** Parameter nennen.

### 1b. Was mit ~100 Parametern realistisch ist (10 Klassen, MNIST)

- PCA + logistische Regression (eigener Schnelltest des Agenten, nicht getunt,
  ein Lauf): k = 8 → 90 Kopfparameter → **79 %**; k = 10 → 110 → **81 %** —
  *ohne* die PCA-Matrix zu zählen (die wäre k×784 und gelernt).
- Kleinste belastbar gefundene CNNs: ~1.000 Parameter → ~97,7 % (unbestätigt).
  Unter 1.000 Parametern **nichts Belastbares gefunden**.
- Quanten-ML: Die oft zitierten „24–51 Parameter, 98 %“ (Hur/Kim/Park) sind
  sehr wahrscheinlich **binär** (0 vs. 1) mit PCA-Vorverarbeitung. Für
  10 Klassen mit ehrlicher Zählung unter 200 Parametern: **nichts gefunden**.
- Realistische Spanne bei ~100 Parametern, alles mitgezählt: **≈ 75–90 %**.
  Über 95 % wäre bei 100 Parametern sehr überraschend.

**Lücke = mögliche echte Neuheit (Vorgabe 7):** Ein 10-Klassen-MNIST-
Klassifikator mit **insgesamt ≤ ~100 Parametern, alles ehrlich gezählt**
(Kodierung fest, Schaltung + Auslese), dazu mit gemessenem **Magie-Budget**
(T-Gatter). Das haben wir so in der Literatur nicht gefunden.

### 1c. „Ist ein QLM unendlich skalierbar?“ — ehrliche Antwort: **nein**

- n Qubits haben 2ⁿ Amplituden, **auslesbar sind höchstens n Bit** (Holevo).
- **Daten laden** kostet bei allgemeinen klassischen Daten so viel, wie die
  Daten groß sind — der exponentielle Vorteil verschwindet.
- **Auslesen** kostet ~1/ε² Messungen je Wert.
- **Trainierbarkeit:** Barren Plateaus (McClean 2018); gerade die gut
  trainierbaren Modelle sind meist klassisch simulierbar (Cerezo et al. 2025).
- Reale Quanten-Sprachmodelle: 4–16 Qubits (z. B. Quixer, 6 Qubits).

**Was im Projekt tatsächlich gut skaliert:** Der QIRE-Adapter braucht je Matrix
nur **log₂(Breite)** Winkel (8 bei 256, 11 bei 2.048) — weniger als BOFT
(d·log d) oder LoRA (d). T-005: Verlust fällt **streng monoton** bis L = 10,
ohne erkennbare Sättigung. Die richtige Frage ist also nicht „unendlich?“,
sondern: **Folgt die Kurve einem Potenzgesetz wie bei LLMs, und wo sättigt sie?**
Erste Studien zu Skalierungsgesetzen von Quantenmodellen gibt es erst seit
2025/26 (QViT, RydbergGPT) — das Feld ist offen.

---

## 2. Vorgeschlagene Forschung (zur Freigabe)

### Linie E — „MNIST mit ~100 Parametern“ (Hauptvorschlag)

**E-0 Ehrliche Referenzkurve (billig, ~30 min, Container).**
Genauigkeit gegen *gesamte* Parameterzahl für klassische Winzlinge:
logistische Regression auf fest verkleinerten Bildern (4×4, 6×6), PCA-LR
(einmal ohne, einmal mit PCA-Matrix gezählt), kleinstes CNN ~100 Parameter,
nächster Schwerpunkt (0 Parameter). 10 Seeds, offizieller Testsatz.
→ Die Messlatte, gegen die jedes QIRE-Modell antreten muss. Kein Urteil,
reine Kartierung.

**E-1 „Kopflose“ QIRE-Zelle (Kern, ~1–2 h, Container).**
Amplitudenkodierung (fest, 0 Parameter) → L Lagen Ry + CNOT-Ring →
**feste Born-Auslese**: 4 Qubits messen → 16 Ausgänge, 10 davon sind die
Klassen (oder Summen über feste Nebenklassen) → nur 10 Bias + 1 Temperatur
trainierbar. Parameter: L·8 + 11 → **L = 4: 43, L = 8: 75, L = 12: 107**.
Trainiert (a) per Gradient voll, (b) Clifford-Suche + k Magiewinkel.
- Hypothese H-1: Bei ≈ 100 Parametern schlägt QIRE die beste klassische
  Referenz gleicher Parameterzahl aus E-0 (TOST/Wilcoxon, 20 Seeds).
- Pflicht-Kontrollen: dieselbe Architektur **ohne CNOT** (= klassische
  Kronecker-Drehung, gleiche Parameterzahl — Lehre aus Q-020: dort war
  `kron` gleich gut), Zufallsschaltung mit trainiertem Bias, lineares Modell.
- Positivkontrolle: die kopflose Zelle muss mit großem Kopf das bekannte
  0,94 erreichen (sonst ist der Aufbau kaputt).

**E-2 Magie-Budget bei 100 Parametern (danach, ~1 h).**
Wie viele T-Gatter braucht das 100-Parameter-Modell? (Verbindet die
Hauptlinie mit dem Ziel.) Nur, wenn E-1 trägt.

**Erwartung ehrlich:** 80–90 % sind realistisch, ein Sieg gegen gleich große
klassische Modelle ist **offen** — genau deshalb ist es eine Forschungsfrage.
Fällt H-1 aus, ist das trotzdem ein sauberer Befund: „Bei ehrlicher Zählung
liegt die Quantenschaltung bei 100 Parametern dort und dort“.

### Linie S — Skalierung des QIRE-Sprachadapters (auf Maltes PC, mehrstündig)

**S-1 Sättigung der Lagenkurve.** T-005 fortsetzen mit L = 20 und L = 40
(8 Seeds, gleiche Lernrate je Arm mit Suche). Fit Potenzgesetz vs. Sättigung.
Kontrolle: Lagen ohne CNOT-Ring (Theorie: = L 1). **~6–10 h auf dem PC.**

**S-2 Potenzgesetz in der Datenmenge.** Adapter L = 5 und LoRA r = 1 mit
500 / 1.000 / 2.500 / 5.000 Trainingsbeispielen. Exponent mit KI.
**~6–8 h auf dem PC.**

**S-3 Breite** (120M → 1B): Auf der CPU vermutlich zu langsam — erst eine
Zeitmessung (10 min), dann Entscheidung Malte (evtl. Kaggle/RunPod = Geld).

---

## 3. Ablauf und Modelle

- **Planung, Design-Entscheidungen, Prüfen der Vortests: Opus.**
- **Modulbau, Rauchtests, Läufe, Auswertungstabellen: Sonnet-Agenten**
  (je Agent eine neue Datei, kein Commit; der Direktor prüft und committet).
- Opus springt nur ein, wenn ein Sonnet-Agent an einer Stelle scheitert oder
  eine statistische/theoretische Entscheidung ansteht.
- Jeder Lauf: Literatursuche vorher, Vorregistrierung vor den Zahlen
  committet, Vortest Pflicht, Wiederaufnahme je Seed (Container-Lehre).

**Vorgeschlagene Reihenfolge:** E-0 → E-1 → (E-2) im Container; parallel
S-1 auf Maltes PC, sobald freigegeben.

## 4. Was Malte entscheiden muss

1. Linie E freigeben (E-0 + E-1)? — Container, kostenlos, ~2–3 h.
2. Linie S freigeben? Welche (S-1, S-2)? — PC, je 6–10 h.
3. S-3 (1B-Modell) nur nach Zeitmessung; ggf. Geld (Kaggle/RunPod).
4. Ist „100 Token“ = **100 trainierbare Parameter gesamt** gemeint?
   (Alternativ: 100 T-Gatter = Magie-Budget — dann wäre E-2 die Hauptlinie.)
