# Durchsicht der Kernaussage und Abgleich der Chats (07.10.2026)

*Recherche-Notiz, keine Befunde. Grundlage: `KERNAUSSAGE_2026-10-07.md`, `LABORBUCH.md` (I-038 … I-044-r),
`recherche/2026-10-06_literatur_struktur.md`, Branch `claude/research-requirements-uo3j9m` (`SKALIERUNG_UND_VORTEIL.md`, T-ternaer).*

## 1. Zwei Linien laufen auseinander
- Der Branch `claude/research-requirements-uo3j9m` baut auf `main` mit Stand 25.09. auf. Er kennt I-038 bis I-045 nicht.
- Sein Plan (V1) nennt die Born-Ablation (Quadrat gegen linear/ReLU) „die wichtigste Einzelmessung“. Sie ist hier schon beantwortet:
  - I-038 H-E: Quadrat trägt gegenüber linear (+2,2 / +1,3 pp).
  - I-038 H-G, I-039-v2, I-041 H3, I-042 H2: kein Vorteil von z² gegenüber |z| oder ReLU (Wechsel je Aufgabe).
- **Empfehlung:** V1 nicht neu rechnen. Neu und ohne Doppelung bleiben T-ternaer (Clifford-Zelle gegen BitNet-ternär) und S3 (Winkelzahl gegen Effektivrang).
- `main` ist veraltet: Die Hauptlinie liegt auf `claude/remote-control-pc-start-gj69o1`. Zusammenführen sollte Malte entscheiden.

## 2. Schwachstellen der Kernaussage (Jurorenfragen)
| Aussage | Problem | Folge |
|---|---|---|
| Strukturkopf +1,9 pp gegen dichten Kopf (I-044-r) | Der dichte Kopf (4,2 Mio, ohne Dropout/Weight Decay, 4 Epochen) ist schwach. Kcd (1×1-Conv + **linearer** Kopf, ≈ 10 k) erreicht 73,70 % gegen Ks 74,33 %. | **I-044-v2** vorregistriert: Ks gegen linearen, regularisierten und budgetgleichen Kopf |
| Idee „Fastfood statt FC-Kopf im CNN“ | bekannt: Deep Fried Convnets (Yang u. a., ICCV 2015) | Als Replikation einordnen. Beitrag nur der faire, vorregistrierte Vergleich |
| „Trainierte Diagonalen schlagen niedrigen Rang um +21–30 pp“ (I-042 H3) | Bei gleichem Budget hat der Rang-Arm Rang ≈ 1, damit ist das Ergebnis trivial | Nicht als Befund in die Arbeit, höchstens als Fußnote |
| Fashion: 16,8 Mio Parameter ersetzt ohne Verlust (I-042 H1) | Ein kleines MLP (≈ 100 k) erreicht auf Fashion ebenfalls ≈ 89 % | Gegen einen budgetgleichen dichten Arm formulieren, nicht gegen 16,8 Mio |
| Born-Quadrat als Alleinstellung gegenüber Quantum-PEFT | bereits widerlegt (s. o.) | Alleinstellung über Messung/Kontrollen, nicht über die Nichtlinearität |

## 3. Wo echte Neuheit am plausibelsten ist
1. **Information pro Schuss gegen exakte Genauigkeit (I-040-v2 H4).** Folgefrage (Hypothese, nicht vorregistriert):
   *Lässt sich die Mischung auf die Pareto-Front Genauigkeit × Bit/Schuss trainieren?* Ein Arm mit Strafterm −λ·I(Klasse; 1 Schuss),
   λ-Reihe, gegen Identität und Haar. Vorher Literatur zu „accessible information“, „measurement basis optimization“, „shot-frugal“.
2. **Adapter-Skalierung (T-006):** die Gradientengeometrie (√L) statt der Lie-Algebra als Grenze. Das ist ein eigenes Skalierungsgesetz, falls K3 hält.
3. **Clifford gegen ternär (T-ternaer, anderer Branch):** Diese Kombination wurde nicht gefunden.

## 4. Blockade
`arxiv.org` ist im Container von der Netzwerk-Policy gesperrt. Volltextprüfungen (Liste in `2026-10-06_literatur_struktur.md`, Ende) gehen erst,
wenn `arxiv.org` in den Umgebungseinstellungen unter Allowed domains steht, oder am PC.
