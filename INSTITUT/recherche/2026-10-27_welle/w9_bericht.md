# w9: Fundgrube im eigenen Repo (27.10.2026)

Erkundung, nichts gerechnet, kein git, nichts in agenda/Laborbuch eingetragen. Gelesen: KERNAUSSAGE, IDEEN.md, ideen.json (35 Karten), LABORBUCH (I-038 bis I-046-v2, T-ternaer(-butterfly)), FEHLERBUCH (Lauf-Auszüge), recherche (breit/11, durchsicht_kernaussage, literatur_struktur_quadrat, volltext_2, hardware_demo, w8), IDEEN_KI/PLAN, Projekt A (BEFUNDE_UND_RICHTUNGEN, STAND_UND_OFFENES, T-006-Eintrag im Tagebuch, ENTWURF_T007). Nicht vollständig gelesen: LITERATUR/notizen.md (190 kB, nur per grep), breit/01–10 (nur Köpfe). Alle Zahlen stammen aus den genannten Dateien, eigene Rechnungen sind als [Rechnung] markiert.

## 1. Kurzfassung

- **Ampel gesamt: gelb.** Der Hauptbefund des Repos ist ein Muster: Jeder Vorsprung der Struktur (I-044, I-045-v2) verschwindet mit mehr Training oder bei gleichen FLOPs. Übrig bleiben **Speicher/Parameter** und **knappes Budget**.
- Größte Lücke in der bisherigen Logik: In I-045-v3/v4 wird 16.000 Schritte lang auf nur 1,6 MB Text trainiert, rechnerisch ≈ 10 Epochen [Rechnung: 16.000 × 8 × 128 = 16,4 M Tokens; Batch/Kontext laut Vorregistrierung I-045]. „Schmale dichte FFN schlägt volle Breite“ ist dann sehr wahrscheinlich Überanpassung, nicht „dicht gewinnt“. Das stellt auch die Schlussfolgerung „dicht gewinnt“ unter Vorbehalt (Punkt 1).
- Echte, bisher nicht ausgeschöpfte Nischen: (a) **Dekodieren mit Batch 1 ist speicherbandbreitenbegrenzt**, der Vorteil von 43 k statt 8 Mio FFN-Parametern wurde nur bei d = 256 und mit Matmul-FLOPs „as implemented“ gemessen (Punkte 3, 4); (b) **Adapter**: bei 300 Schritten ist QIRE pro Parameter 2,5× effizienter als LoRA (T-006), parametergleiche Gegner fehlen (Punkt 5); (c) **Datenknappheit** (B-010: Blockdiagonal gewinnt bei n = 200, nie auf Struktur-Schichten übertragen, Punkt 7); (d) **Gesamt-Bit-Pareto** (Punkt 9), (e) **Information pro Schuss** (I-046-v2 wartet, Punkt 10).
- Sofort billig und ehrlich: Punkt 8 (kleiner MLP als Fashion-Referenz fehlt), Punkt 13 (Rotation vor Quantisierung, Minuten), Punkt 4 (nur Rechnung).
- Widersprüche zwischen Läufen: 12 gefunden (Abschnitt 3), vor allem Sz hängt von d ab (W2) und z² hilft bei d = 512, T = 16000 nicht mehr (W3).

## 2. Nebenbefunde, die nie verfolgt wurden (Auszug, Details in Abschnitt 4)

| Fundstelle | Befund (beschreibend) | Wurde verfolgt? |
|---|---|---|
| I-045-v4 Nebenbefund | Bei T = 16000 schlagen schmale dichte FFN (h = 580: 2,097 / 2,086 bpc) die volle Breite (h = 2048: 2,163) | nein („vermutlich Überanpassung oder Lernrate“) |
| I-045-v3 Tabelle | Sz wird mit d = 512 schlechter als bei d = 256 (T = 1000: 3,482 gegen 3,404; T = 4000: 2,593 gegen 2,558), die dichten Arme werden besser | nein; Vorbehalt „Lernrate nur bei T = 4000“ |
| I-044-v2 | „Jede dichte versteckte Schicht ist bei 4 Epochen schlechter als gar keine (Kd −1,2, Kklein −0,8 gegen Klin)“ | durch v3 nur teils (Untertraining) |
| I-042 H1 + durchsicht_kernaussage | Ein kleiner MLP (≈ 100 k) erreicht auf Fashion ≈ 89 % wie die 16,8-Mio-Schicht | Gegentest nicht gerechnet |
| B-010 (Projekt A) | Blockdiagonal gewinnt bei n = 200 (+1,5 pp), nicht durch Parameterzahl erklärbar, „offenes Rätsel“ | nein |
| T-006 | Pro Parameter ist QIRE-Adapter besser (2,2e-5 gegen 8,9e-6 je Parameter); offen, ob Grenze am Budget oder an der Struktur (qire_L10_600 steht aus); Lernrate für L20/L40 am untersten Gitterwert | Auswertung offen |
| I-034-fashion | `zwei_0_8_m48` erreicht 504 T = 6,5 % bei 81,61 % (Referenz 81,88 %), nur nebenbei, nicht vorregistriert | nein |
| I-046 | Front ist flach: λ = 0,25 holt 92 % der Information für −1,1 pp; S = 30/100 und Fashion nur „Folgefragen“ | I-046-v2 wartet auf Freigabe, dort nur Budgetfrage |
| I-035 | M50 mit Rauschen eher fallend, durchgehend negatives KI, als „echte Überraschung“ notiert | nein |
| breit/11 | Clifford-Maske als Rotation für Quantisierung: „bestes Verhältnis Aufwand/Nutzen“, Test in Minuten | im Laborbuch nicht gefunden |

## 3. Widersprüche zwischen Läufen und Dokumenten

- **W1 Born gegen |z|:** I-038 H-G MNIST +0,35 pp (BORN_BESSER), Fashion 0,00; I-039-v2 −0,39 pp GLEICHWERTIG; I-041 H3 MNIST −0,07 (HU), Fashion schrumpft −0,57; I-042 H2 Vorzeichen wechselt je Aufgabe. Konsistente Lesart: Effekt aufgaben- und größenabhängig, im Mittel ≈ 0.
- **W2 Sz und Breite:** I-045-v2/v3 bei d = 256 und T = 1000: Sz 3,404 ist **besser** als D_z2 (3,471). Bei d = 512 und T = 1000: Sz 3,482 ist **schlechter als bei d = 256** und 0,34 bpc hinter D_z2 (3,140). Ein breiteres Modell mit gleichem Training sollte nicht schlechter werden. Zeichen für Lernrate/Initialisierung (Lernrate nur bei T = 4000, d-Skalierung der Diagonal-Init nicht begründet) und nicht für „Struktur skaliert nicht“.
- **W3 z² dicht:** I-045-v2: D_z2 besser als D_gelu um 0,043 bpc („Primer-Effekt“). I-045-v3 d = 512, T = 16000: D_gelu 2,106 **besser** als D_z2 2,153. Bei d = 256, T = 16000 ist es umgekehrt (2,103 gegen 2,116). Der Quadrat-Effekt ist nicht robust über (d, T).
- **W4 „Struktur schlägt dicht“ gegen „dicht schlägt Struktur“:** I-044/I-044-r (+1,8/+1,9 pp) gegen I-044-v2 (+0,83 gegen linear) gegen I-044-v3 (−0,5 bei 24 Epochen). Gleiches Muster I-045-v2 gegen v3/v4. Der Vorsprung hängt vom Gegner (voller dichter Kopf) und der Epochenzahl ab, nicht von der Struktur.
- **W5 Kernaussage Zeile 22 gegen durchsicht_kernaussage §2:** „Struktur ersetzt 16,8 Mio ohne Verlust“ (Fashion) wird durch einen kleinen MLP relativiert; der budgetgleiche Test steht nirgends im Laborbuch.
- **W6 Zeitgewinn:** I-041: Fastfood schneller ab N = 1.024 (Batch 1), 89× weniger Operationen. breit/11: FWHT bei 256 Batch 64 **5× langsamer**, Batch 1 17× langsamer. Projekt A (T-001): Übergang erst bei 16.384, 1,59×/1,99×/2,31×, später Bündler 4.096–8.192. I-045 H3: Sb bei d = 256 6× langsamer als dicht. Drei Implementierungen, drei Übergangspunkte; kein einheitlicher Kernel-Vergleich.
- **W7 Kernaussage §1 gegen Projekt A:** Kernaussage nennt Tiefe/Verschränkung gleichwertig (I-038); Projekt A hat sie am 23.09. von „belegt“ auf „offen“ gesetzt (Encoder-Artefakt, durchgefallene Positivkontrolle), Q-018 zeigt +0,41 für trainierte Winkel ohne Encoder. Nicht auflösbar, weil Encoder-/Kopfhilfe in I-038 nicht ausgeschlossen ist (Kopf 2.570 Parameter, Winkel 32).
- **W8 Hybrid-Kaskade:** MNIST 2,25 % der Hardware-T ohne Verlust (I-034), Fashion 12,4 % und schlechter (I-034-fashion); I-005-v2 „MNIST braucht keine Magie, Fashion schon“. Die Zahl „44-fach“ gilt nur für MNIST (Laborbuch selbst).
- **W9 Bit-Pareto:** T-ternaer (cliff0 mit 64 Bit schlägt ternär 103.904 Bit um +1,8 pp) gegen I-039-v2 (klassisch mit gleichem Bitbudget, k2_relu 32 Bit: 79,8 % ≈ b_ff 80,3 %). Der erste Vergleich hat keinen gleich kleinen klassischen Gegner, der zweite zeigt, dass es ihn gibt. Außerdem zählt der Kopf (2.570 Gleitkommazahlen) viel mehr Speicher als die „64 Bit“.
- **W10 Holevo:** I-040-v2 H3 (η sagt Schussbedarf nicht voraus, nur 5 Arme) gegen die Rolle von η in Erklärungstexten; und der Zielkonflikt (H4) gegen I-046 „kein Naturgesetz, Folge der festen Wahl“ — kein echter Widerspruch, aber in der Kernaussage §3 steht beides getrennt.
- **W11 Rang-Arme:** I-042 H3 (+21 bis +30 pp gegen niedrigen Rang) ist trivial (Rang ≈ 1, durchsicht §2), I-045-v4 zeigt dagegen Lk (Rang 116) gleichauf mit dichtem. Der +30-pp-Satz darf nicht als Stärke der Struktur zitiert werden.
- **W12 Plan T-007:** Vorregistriert auf der Grundlage I-045-v2 (Sz besser als dicht); I-045-v3/v4 haben das für langes Training widerlegt. T-007 braucht eine Neubegründung (Punkt 5).

## 4. Priorisierte Liste (14 Punkte)

Bewertung: Wert für die Arbeit (hoch/mittel) × Kosten des kleinsten Tests. Alle Tests: erst ≥ 3 Tuning-Seeds (900–999) im Container, danach ggf. Vorregistrierung.

### 1. Datenregime als Störgröße der Sprachmodell-Linie (Epochenzahl)
- **Fundstelle:** I-045-v4 Nebenbefund; I-045-v3 (T = 16000); I-045 Vorregistrierung (1,61 Mio Byte Train).
- **Idee:** Alle „dicht gewinnt bei viel Training“-Schlüsse laufen bei ≈ 10 Epochen über denselben 1,6 MB. In diesem Regime schlägt Überanpassung zu (schmale FFN > breite). Die Frage „Struktur gegen dicht“ ist erst nach Trennung von Training und Datenmenge beantwortbar. Dort, wo Daten knapp und Training lang sind, ist **geringe Kapazität** ein echter Vorteil (Regularisierung) und könnte Sz/Dk begünstigen.
- **Warum jetzt:** Es entscheidet, ob die Linie I-045 „abgeschlossen“ ist (Laborbuch 27.10.) oder nur im überangepassten Regime geprüft wurde. Billigste Korrektur eines möglichen Fehlschlusses.
- **Kleinster Test:** d = 256, 4 Lagen. Zwei Datenmengen (25 % und 100 % der Trainingsbytes) × T ∈ {1000, 4000} (= ¼, 1, 2,5 und 10 Epochen); Arme Sz, D_z2 (h = 1024), Df_z2 (h ≈ ¼·1024), Dk (h = 10). Metrik: Validierungs-bpc über die Zeit (Kurve, kein Endpunkt), 3 Seeds. Ersatzdaten: Maltes 2,9-GB-Korpus nur auf dem PC. Erwartung vorab: Abstand dicht − schmal schrumpft mit Epochen, Sz bleibt hinten. Kosten: ≈ 12 Läufe × ≈ 15 min (aus Schrittzeit 0,22 s/Schritt, 4000 Schritte ≈ 15 min) = zu teuer für das 30-min-Budget des Containers pro Agent; Aufteilung auf Raven oder kleiner (d = 128, T ≤ 2000).

### 2. Sz wird mit d schlechter: Lernrate/Initialisierung neu suchen (W2)
- **Fundstelle:** I-045-v3 Tabelle; Vorbehalt „Lernrate je (Arm, d) nur bei T = 4000 gesucht“.
- **Idee:** Prüfen, ob Sz bei d = 512 mit eigener Lernrate und passender Diagonal-Init (z. B. D ~ 1/√(N/d) oder getrennte Rate für Diagonalen gegen Bias/Einbettung) auf das d = 256-Niveau oder besser kommt. Fastfood-Diagonalen mit Rate 1,5e-3 und Gaußinit sind bei 4× Breite ein anderes Problem (Summe über 4 Blöcke, 1/√-Skalierung fehlt?).
- **Warum jetzt:** Wenn Sz bei d = 512 nur durch Tuning schwach war, ändert das Aussage 1 und alle Kosten-Nutzen-Schlüsse in v4 (Sz verlor bei d = 512 um 0,081 bpc gegen Df_z2). Billig: Rate und Init sind zwei Zahlen.
- **Kleinster Test:** d = 512 (oder d = 128 gegen 256 als Skalierungspaar), T = 1000, Sz mit 5 Lernraten × 2 Init-Skalen, 3 Seeds. Erfolg: Sz(d=512) ≤ Sz(d=256) und Lücke zu D_z2 schrumpft um > 50 %. Kosten: Sz ≈ 0,23–0,67 s/Schritt, 1000 Schritte ≈ 4–11 min je Lauf: 10 Kombinationen × 1 Seed erst als Kurve (≈ 1 Stunde CPU, nur kleine Version im Container: d = 128 gegen 256).

### 3. Dekodier-Mikrobenchmark: Speicherbandbreite statt FLOPs (Batch 1, große d)
- **Fundstelle:** I-045 H3 (Sb bei d = 256 6× langsamer), breit/11 („nur BLAS ist so optimiert“), Projekt A T-001/Q-009 (dichte Schicht bricht ab 16.384 ein, Cache-Kante; 32.768 passt nicht mehr in 16 GB), I-041 (89× weniger Operationen).
- **Idee:** Beim Token-für-Token-Dekodieren ist die FFN speicherbandbreitenbegrenzt (alle Gewichte einmal pro Token lesen). Ein FFN mit O(N) Parametern liest 43 k statt Millionen Werte. Das einzige Regime, in dem „weniger Parameter“ direkt Zeit spart, wurde nie sauber gemessen: Ein reiner FFN-Block mit **echter FWHT** (nicht als Blockmatmul) bei d ∈ {1024, 2048, 4096, 8192} und Batch 1, Gewichte fp32 und bf16/int8, gegen dichte FFN und gegen Niedrigrang gleichen Speichers.
- **Warum jetzt:** Löst W6 (drei Übergangspunkte) in einem einzigen Aufbau; entscheidet, ob die Nische „Speicher“ auch Zeit bringt. Kein Training, nur Zeitmessung, daher keine Vorregistrierung nötig (Modulbau/Messung ohne Freigabe laut CLAUDE.md).
- **Kleinster Test:** Skript mit torch (1 Thread), FWHT vektorisiert in-place (log₂N Stufen von Reshape-Additionen), je Konfiguration Median von 200 Aufrufen nach Aufwärmen, L2/L3-Größe des Ryzen 5 3600 (32 MB L3) notieren. Erwartung: dicht fällt ab, wenn d·4d·4 B den L3 überschreitet (d ≈ 1536); FWHT-Variante gewinnt dann. Kosten: wenige Minuten, 1 Thread.

### 4. FLOPs „as implemented“ gegen ideale Operationszahl (Rechnung)
- **Fundstelle:** I-045-v4 (Sz hat 58× mehr FLOPs als Dk; FLOPs „Fastfood = zwei kleine Matmuls je Walsh-Transformation“); I-041 (Operationsverhältnis 89× bei N = 1.024).
- **Idee:** v4 vergleicht gegen dichte FFN mit gleichen **Matmul-FLOPs**. Ideal gezählt (FWHT: N·log₂N Additionen, zwei Transformationen je Block) hat Sz bei d = 512 etwa 90 k Operationen je Lage gegen 1.187.840 [Rechnung: 4 Blöcke 512→ je 2·512·9 ≈ 9,2 k plus 2.048-Block 2·2.048·11 ≈ 45 k; Größenordnung ≈ 13× weniger, nachzurechnen]. Eine Dichte mit gleich vielen idealen Operationen hätte h ≈ 45, nicht 580. Der Gegner „gleiche Operationen“ ist dann schwächer, und Sz hat gegen Dk_z2 (h = 10) bereits +0,09 bpc.
- **Warum jetzt:** Der Satz „kein Effizienzvorteil bei gleichem Rechenaufwand“ (Laborbuch 27.10.) gilt nur für Matmul-Zählung. Mit Zählung der echten Operationen und einem passenden Gegner (h ≈ 45–60, Rang ≈ 12) kann die Lage anders sein; ehrlich beide Zählungen nebeneinander.
- **Kleinster Test:** Erst nur Rechnung (`flops_je_token` um `ideal`-Variante ergänzen, Tabelle). Dann d = 256, T = 4000: Df_z2 mit h passend zu idealen Operationen von Sz (h ≈ 15–25 bei d = 256), 3 Seeds, gegen Sz und Dk. Kosten: Rechnung Minuten; Training Dk-/Df-klein sind billig (0,12–0,37 s/Schritt).

### 5. Adapter-Nische: parametergleiche PEFT-Gegner und T-007 neu begründen
- **Fundstelle:** T-006 (QIRE L40: 7.680 Parameter holen 62 % des LoRA-r1-Gewinns mit 30.720 Parametern; Lernrate L20/L40 am untersten Gitterwert; qire_L10_600 offen); ENTWURF_T007 (ff_born, 43.008 Parameter gegen lora_r1); W12.
- **Idee:** Adapter-Feinabstimmung ist **von Natur ein knappes Budget** mit eingefrorenen guten Merkmalen — also genau das Regime, in dem Struktur (CNN-Kopf, I-045-v2) gewann. Die Gegner fehlen: BitFit, (IA)³, VeRA, LoRA mit Rang 1 nur auf q oder nur auf v (≈ 2–8 k Parameter), Diagonal-/Fastfood-Adapter. Pro Parameter ist QIRE-L40 2,5× besser als LoRA-r1; ob das gegen einen gleich teuren LoRA-Ausschnitt oder (IA)³ hält, ist unbekannt.
- **Warum jetzt:** T-007 ist unter falscher Begründung vorregistriert (W12), die Nische ist aber stärker als vorher angenommen; die T-006-Lernratenkante ist ein offenes Qualitätsproblem.
- **Kleinster Test (Container, ohne LLäMmlein):** Byte-Sprachmodell d = 256 (struktur_sprache) vortrainieren (T = 2000, 1 Seed Basismodell), dann Anpassung an eine andere Teilverteilung (z. B. nur Sätze mit Wort „nicht“ oder Rezensionen gegen Nachrichten aus GSD; Teile aus dem vorhandenen Train-Split), 300 Schritte, Arme Fastfood-Adapter (D3 = 0-Start, ff_born), ff_gelu, LoRA r = 1 auf ein Teilbudget, (IA)³, BitFit, mit gleicher Parameterzahl (≈ 2 k und ≈ 8 k), 3 Seeds. Metrik: bpc der Zielverteilung, Gewinn je Parameter. Kosten: ≈ 8 Arme × 3 Seeds × ≈ 1 min.

### 6. Speicher-Pareto bpc gegen FFN-Bytes mit starken parametergleichen Gegnern
- **Fundstelle:** I-045-v4 („Sz tauscht Rechenzeit gegen Speicher“; Dk_z2 ist ein Strohmann, Laborbuch 26.10.: „Dk mit h = 10 ist ein Strohmann“); I-045-v2 (L_z2 3,563 gegen Sz 3,403, aber nur T = 1000).
- **Idee:** Die verbleibende Aussage „Sz ist speicherarm“ ist nur dann stark, wenn Gegner mit gleichen 43 k Parametern **bessere** Formen sind als h = 10: (a) FFN mit über die 4 Lagen **geteilten** Gewichten (ALBERT), (b) Niedrigrang r gleichen Speichers, (c) Monarch/Kronecker/TT-FFN, (d) ternäre/int4-dichte FFN gleicher Bytes (Bytes statt Parameter zählen!). BitNet-artige Gewichte haben 1,6 Bit/Parameter: 8 M Parameter ≈ 1,6 MB, Sz fp32 43 k ≈ 172 kB — nur noch Faktor 10, nicht 200 [Rechnung].
- **Warum jetzt:** Der Speichervorteil ist das, was laut Auftrag übrig bleibt; er muss gegen **Bytes** und starke Gegner bestehen, sonst fällt die letzte Nische. Passt zu w8 (R2/R3: Pareto bpc gegen Bytes).
- **Kleinster Test:** d = 256, T = 4000, 3 Seeds, Arme Sz, geteilte dichte FFN (h so, dass Parameter = Sz), Niedrigrang gleichen Speichers, ternäre dichte FFN (STE) gleicher Bytes; Metrik bpc gegen FFN-Bytes (Kurve über 3 Größen). Kosten: ≈ 5 Arme × 3 Seeds × ≈ 15 min = 3,75 h CPU, im Container nur in Verkleinerung (d = 128, T = 1500).

### 7. Datenknappheit statt Trainingsknappheit: Struktur-Prior bei n klein
- **Fundstelle:** B-010 (Projekt A, STAND_UND_OFFENES): Blockdiagonal gewinnt bei n = 200 (+1,49 pp MNIST, +1,39 Fashion), „offenes Rätsel, konkret genug für Nachtest“; I-024 (Gewinn ≈ 3,5 pp datenunabhängig); I-039 (110 Parameter).
- **Idee:** Die Struktur-Linie wurde nur gegen **Trainingsknappheit** (wenige Epochen) getestet, wo Aufholen mit mehr Training das Ergebnis aufgelöst hat (I-044-v3). Datenknappheit (n = 100–1.000) ist ein anderes Regime: dort wirkt ein Prior (feste Mischung, wenige Parameter) auch bei beliebig langem Training. Das ist zugleich die ehrliche Lesart des „110-Parameter“-Ergebnisses.
- **Warum jetzt:** Das Regime ist noch nie mit I-038/I-041-Bausteinen (Haar², Fastfood², Pixel-Softmax, kleiner MLP) auf n-Reihe gemessen; die Lehre aus I-044-v3 („Vorsprung = knappes Budget“) trifft dort nicht automatisch zu, weil man bis zur Konvergenz trainieren kann.
- **Kleinster Test:** MNIST/Fashion 16×16, n ∈ {100, 300, 1.000, 3.000}, jeweils bis Konvergenz (feste hohe Schrittzahl, Validierungsstopp), Arme Fastfood² + linear, Haar² + linear, kleiner MLP gleicher Parameterzahl (3 Größen), linear, 3 Seeds. Metrik: Testgenauigkeit gegen n. Kosten: kleine Modelle, wenige Minuten je Punkt; passt in 30 min.

### 8. Fehlende Referenz: kleiner MLP auf Fashion (budgetgleich) für I-042
- **Fundstelle:** durchsicht_kernaussage §2 („Ein kleines MLP (≈ 100 k) erreicht auf Fashion ≈ 89 %“) und Kernaussage §2 Zeile 22 (n = 5, 16,8 Mio).
- **Idee:** Der Satz „feste Struktur ersetzt 16,8-Mio-Schicht ohne Verlust (Fashion)“ ist nur dann informativ, wenn er einen **gleich teuren** dichten Arm schlägt oder gleicht. Fehlt im Laborbuch (I-042-r wiederholt nur den schwachen Vergleich).
- **Warum jetzt:** Juror-Frage Nummer eins; ohne den Test ist der Satz irreführend. Sehr billig.
- **Kleinster Test:** Fashion 32×32, N = 16.384 wie I-042, Arme A (dicht 16,8 M), E (Fastfood, trainierte Diagonalen), plus M_klein (dicht N → 128 → 10, ≈ 2 Mio) und M_budget (Parameter gleich E: ≈ 100 k), 3 Seeds, 5 Epochen, Lernrate je Arm auf Tuning-Seed. Metrik: Genauigkeit, Parameter, FLOPs. Kosten: kleine Schichten schneller als I-042-Läufe (A ≈ 0,5 min/Seed Fashion laut Kernsekunden), insgesamt ≈ 10 min.

### 9. Gesamt-Bit-Pareto: Kopf mitzählen, gegen kleine klassische Netze
- **Fundstelle:** T-ternaer (cliff0 „64 Bit Mitte“, Kopf 2.570 Zahlen), I-039-v2 (klassisch mit gleichem Bitbudget gleich gut), IDEEN.md §7, breit/12 (Hebel ist der Kopf).
- **Idee:** Die Pareto-Aussage „im Bereich sehr kleiner Speicher ist die Clifford-Zelle die beste“ (T-ternaer) gilt für die Mitte; der float-Kopf dominiert den Speicher (82 kb gegen 64 Bit). Echte Gesamtaussage: Gesamtbits (Zelle + auf 2–8 Bit quantisierter Kopf) gegen gleich kleine klassische Netze (256→8/16 ternär + Quad, 1-Bit-Projektion + Popcount, k2_relu aus I-039-v2). I-039 zeigt kopffreie Auslese 71,7 % mit 0 reellen Parametern; hier wäre der Punkt „kleinstes Modell > 80 %“.
- **Warum jetzt:** Verhindert, dass die Arbeit einen Speichervorteil behauptet, der am Kopf hängt, und liefert falls er hält genau die gewünschte Nische (Speicher). T-ternaer-butterfly wartet auf Freigabe und deckt die Mitte ab; diese Erweiterung ist gratis dazu.
- **Kleinster Test:** MNIST 16×16 wie T-ternaer; Kopf mit 1/2/4/8 Bit quantisiert (Nachtraining), Arme cliff0, voll, ternär 256→k (k = 8, 16, 32) + Quad, identitaet; Metrik: Genauigkeit gegen Gesamtbits (log-Achse), 3 Seeds. Kosten: ≈ 1 min je Lauf (60–230 s laut Vorregistrierung): ≈ 20 Läufe ≈ 30 min; Tuning-Seeds.

### 10. Information pro Schuss: S = 30/100 als Hauptgröße und Fashion
- **Fundstelle:** I-046 (Front flach, λ = 0,25 holt 92 %), I-046-v2 (wartet, nur Budget), I-040-v2 H2 (schussbewusst +1,5/+1,2/+0,3 pp), I-008 (≈ 1.000 Schüsse je Bild nötig), hardware_demo (IBM 10 min QPU).
- **Idee:** Die Nische „wenige Schüsse“ ist die einzige, in der die Born-Auslese als Messung (nicht als Funktion) real ist. Offen: Genauigkeit bei S = 30/100 als Zielgröße im Verlust statt exakt, plus Fashion; ob eine feste Fastfood-Mischung mit **wenigen** trainierten Winkeln (I-046-v2 U_ff: 384 Parameter) die Front hält. Diese Fragen sind in I-046 explizit als Folgefragen (2) und (3) genannt, nicht in v2 vorregistriert.
- **Warum jetzt:** Das Laborbuch markiert I-046 als „Kandidat für den eigenen Befund der Arbeit“; v2 prüft nur Budget. S-Training ist günstig.
- **Kleinster Test:** I-046-Rahmen (MNIST 16×16, 6.000/5.000), λ ∈ {0, 0,25}, Verlust mit Schussrauschen S ∈ {30, 100} (Straight-Through/Stichprobe), Arme id, haar, voll25, ff25; Metrik: acc bei S, I_C, 3 Seeds. Kosten: voll ≈ 71 s, ff ≈ 141 s je Aufgabe: 4 Arme × 2 S × 3 Seeds ≈ 1 h; nur voll25/ff25 bei S = 30 ≈ 15 min.

### 11. Hybrid-Kaskade auf Fashion mit `zwei_0_8_m48`
- **Fundstelle:** I-034-fashion (6,5 % T bei 81,61 % gegen 81,88 %, nur nebenbei), I-005-v3 (M50 = 4 Winkel), I-024.
- **Idee:** Die einzige Stelle mit dem Ressourcenmaß „T-Gatter“ (echte Quantenressource) und einem MNIST-Erfolg (88 % → 2,25 %), der auf Fashion „nicht repliziert“. Die Nebenkonfiguration ist nahe am Ziel; vorab nicht als Hauptkonfiguration festgelegt, daher kein Befund. Als Zeichen „Aufgabenabhängigkeit mit Zahl pro Aufgabe“ ehrlich nutzbar.
- **Warum jetzt:** Klein, vorhandene Module (Kaskade), und es beantwortet W8 (MNIST gegen Fashion).
- **Kleinster Test:** Fashion 16×16, Konfigurationen `zwei_0_8_m48` und eine Variante mit m = 64, 3 Seeds, Metrik T-Anteil und Genauigkeitsverlust gegen die Referenz. Kosten: I-034-fashion lief 43 min für 10 Seeds; 3 Seeds, 2 Konfigurationen ≈ 10 min. Wert für „ressourceneffiziente KI“ gering (Laden frisst Ersparnis, C-kodierung); als Nebenpunkt.

### 12. Strukturkopf bei vielen Klassen (CIFAR-100)
- **Fundstelle:** I-044-v2/v3 (Klin 10 k Parameter gleichauf mit Ks 49 k bei CIFAR-10), I-042 (Struktur trägt „hinter guten Merkmalen“).
- **Idee:** Ein linearer Kopf wächst mit **N·C**, ein Fastfood-Kopf mit **N** (die C Ausgänge sind Gruppen/Teilausgänge). Bei C = 10 ist Klin schon billig (10 k), deshalb kein Vorteil. Bei C = 100–1.000 ändert sich das Verhältnis; das war nie Teil einer Prüfung. Auch Auflösung (große N) erhöht den Abstand.
- **Warum jetzt:** Einzige Stelle, an der „85× weniger Kopfparameter“ gegenüber **dem richtigen Gegner (linear)** noch ein Faktor sein könnte. Benötigt keine neuen Bausteine.
- **Kleinster Test:** CIFAR-100, CNN-Rumpf wie I-044 (4–12 Epochen), Köpfe Klin, Kklein, Ks (Fastfood mit C Gruppen), 3 Seeds, Metrik Genauigkeit gegen Kopfparameter und Epochen. Kosten: ≈ 170 s je Seed und Arm bei CIFAR-10 (I-044), also 3 Arme × 3 Seeds ≈ 25 min.

### 13. Clifford-Maske/Hadamard als Rotation vor Gewichtsquantisierung
- **Fundstelle:** breit/11 §1 (EIGEN-Idee mit bestem Aufwand/Nutzen, „Minuten“), w8 (R2/R3: BitNet v2, SpinQuant, ButterflyQuant), IDEEN.md §1 (Hadamard vor Quantisierung als QuaRot-Nähe), I-036-v3 (Formen übertragbar, 52–78 % des Gewinns vom Rang).
- **Idee:** Die gefundene Clifford-Maske als feste Rotation vor 2–4-Bit-Quantisierung eines trainierten Netzes gegen Identität, zufällige Vorzeichen·Hadamard, QuaRot-Stil. Im Laborbuch finde ich diesen Test nicht (grep QuaRot/SpinQuant ohne Treffer in LABORBUCH/agenda).
- **Warum jetzt:** w8 schlägt Rotation-vor-Quantisierung als realistisch grün vor; hier die institutseigene Zutat (Clifford-Form) mit Kontrollen.
- **Kleinster Test:** Kleiner MLP/Transformer-FFN (MNIST bzw. Byte-Modell), nach dem Training Gewichte (und optional Aktivierungen) mit 3/4 Bit pro Zeile quantisieren, vorher je Rotation; Metrik: Genauigkeit/bpc und Quantisierungsfehler, 3 Seeds. Kosten: Minuten (kein Training nötig).

### 14. Quanten-Hardware-Schuss und Verschränkung: W7 sauber auflösen (Nebenpunkt)
- **Fundstelle:** Projekt A STAND_UND_OFFENES (Verschränkung/Tiefe [OFFEN], Q-020 (b)); I-038 H-A/H-B.
- **Idee:** Nur, falls die Arbeit „Verschränkung trägt nichts“ schreibt: Aufbau ohne Encoder und ohne trainierten Kopf-Ersatz (Q-018-Muster) gegen CNOT-Ring und gegen Rotation ohne CNOT. Ohne diesen Test sollte der Satz in der Arbeit als Beobachtung im I-038-Aufbau formuliert werden.
- **Warum jetzt:** Juror-Frage „Verschränkung?“ ist wahrscheinlich; billig, wenn Q-020 (b) fertig ist.
- **Kleinster Test:** erst nachsehen, ob Q-020 (b) ausgewertet ist (`A/results`); sonst I-038-Zelle mit/ohne CNOT, Kopf eingefroren auf feste Zufallsmatrix, 3 Seeds. Kosten: ≈ 5 min.

## 5. Wo der Vorteil echt war und wie man genau diese Nische ehrlich ausbaut

| Nische | Beleg | Was fehlt für Ehrlichkeit | Ausbau |
|---|---|---|---|
| Speicher/Parameter in der FFN | I-045-v4: Sz 43 k gegen Dk_z2 43 k: −0,09 bpc bei T = 16000 | starke parametergleiche Gegner, Bytes statt Parameter (Punkt 6) | Pareto bpc gegen Bytes, Sz + ternäre/gemeinsame FFN |
| Dekodier-Geschwindigkeit bei großem d | I-041 (89× Operationen), T-001 (16.384: 1,6–2,3×), Projekt A „Machbarkeitsgrenze“ ab 32.768 | Messung mit echter FWHT, Batch 1, Cache-Kante (Punkt 3) | Mikrobenchmark; falls grün, Kernel in Raven-Läufen messen |
| Knappes Adapterbudget | T-006 (2,5× je Parameter), I-045-v2 (Sz bei 1000 Schritten) | parametergleiche PEFT-Gegner (Punkt 5) | Adapter-Pareto statt Einzelvergleich mit LoRA r1 |
| Datenknappheit | B-010 (n = 200), I-024 | nicht mit Struktur-Schichten gemessen (Punkt 7) | n-Reihe mit Baseline kleiner MLP |
| CNN-Kopf mit wenig Training | I-044-r +1,9 pp, v2 +0,83 pp gegen linear (4 Epochen) | v3: bei ≥ 12 Epochen weg; Klin hat weniger Parameter | nur als „knappes Budget“ ausweisen; CIFAR-100 (Punkt 12) |
| Schussarme Auslese | I-046 (flach), I-040-v2 H2 | Budgetgleichheit (v2), S = 30/100 (Punkt 10) | v2 freigeben, S-Verlust nachziehen |
| Minimaler Zellspeicher | T-ternaer (64 Bit) | Kopf mitgezählt, gleich kleine klassische Gegner (Punkt 9) | Gesamt-Bit-Pareto |

Durchgehende Regel für alle Punkte: bei gleichen FLOPs UND gleichen Parametern (Bytes) gegen dicht/Niedrigrang, mit ausreichend Training und Datenmenge; Lernrate je Arm und je Trainingslänge neu suchen (Vorbehalt in v3/v4). Wirkung klassisch (I-038): „quanten-inspirierter Vorteil“.

## 6. Reihenfolge-Vorschlag

1. Rechnungen und Messungen ohne Training: Punkt 4 (Rechnung), 3 (Mikrobenchmark), 13 (Rotation), 8.
2. Danach ein Container-Block mit kleinen Läufen: Punkt 7, 9, 5 (verkleinert).
3. Raven/PC: Punkte 1, 2, 6 (Kurvenläufe), danach Vorregistrierung; Punkt 10 über I-046-v2 + Zusatz.
4. Nebenpunkte 11, 12, 14 nur bei Zeit.

## 7. Offenlegung und Grenzen

- Nicht gerechnet, keine Skripte erstellt (Auftrag: lesen und sammeln). Alle „Erwartungen“ sind Vermutungen, keine Befunde.
- 10×-Epochenrechnung (Punkt 1) und Operationszählung (Punkt 4) sind eigene Überschlagsrechnungen aus Vorregistrierungsangaben und müssen vor Gebrauch gegen den Code (`struktur_sprache.py`, `flops_je_token`) geprüft werden.
- Kosten-/Laufzeitangaben stammen aus Laborbuch-Messungen (Container, 1 Thread), nicht aus dieser Sitzung.
- Quellen: nur Repo-Dateien; kein WebSearch, kein [Gedächtnis]-Zitat außer wo ausdrücklich als Zitat aus den Repo-Berichten übernommen.
- Dateien: `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w9_bericht.md` (keine Skripte).
