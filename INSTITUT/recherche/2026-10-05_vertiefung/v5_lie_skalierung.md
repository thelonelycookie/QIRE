# v5 — Wo muss der QIRE-Adapter sättigen? Lie-Algebra, Jacobi-Rang, „unendlich skalierbar“

Stand 06.10.2026, Sonnet (Vertiefung Etappe 2). **Erkundung, kein Befund, nichts vorregistriert.**
Seeds 970–979 reserviert; verwendet: Bootstrap-Zufall 970 (Fit), Winkelpunkt θ ~ U[0,2π) mit Generator-Seed 0 (Rang, deterministische Algebra, kein Trainings-Seed nötig). Kein Training, keine Daten, keine Modellläufe.

## 1. Kurzfassung
- Der Adapter erzeugt R = P^-L ∏(P·K_l) mit K_l = Ry-Kronecker-Lage (n Winkel) und P = fester CNOT-Ring. Die DLA ist (numerisch für n = 3, 4, 5, 6) die volle so(2ⁿ), dim d(n) = 2^(n−1)(2ⁿ−1): 28, 120, 496, 2016. Für n = 8 (so(256), d = 32.640) **nicht direkt geschlossen** (Speicher/Zeit), nur nach Muster erwartet.
- Die Lie-Algebra ist **schnell** voll (n = 4, 5: schon mit 2 Lagen-Erzeugern, n = 6: 4). Entscheidend für die Parametrisierung ist aber der **Jacobi-Rang**: rank = min(n·L, d(n)), bestätigt für n = 4, 5, 6 bis auf die Einheit genau (Sättigung exakt bei L* = ⌈d/n⌉ = 30, 100, 336) und für n = 8 (bis L = 200) und n = 11 (bis L = 100) ohne jede Redundanz.
- T-005: Blockdimension 256 = 2⁸ (Code: `BLOCK = 2**QUBITS`, 768 = 3 Blöcke mit geteilten Winkeln). **L\* = 32.640 / 8 = 4.080 Lagen**, das 408-Fache von L = 10. Die algebraische Sättigung ist für T-005 (und für L = 20, 40) irrelevant.
- **Gemessener Zusatzbefund:** die trainierten Winkel sind winzig (mittleres |θ| 0,015–0,03, Maximum ≤ 0,124 rad, ‖θ‖₂ je Matrix 0,11 → 0,17 von L = 1 bis 10). Der Adapter arbeitet im linearen Bereich um R = I. Die Kurve wird also durch **Trainingsbudget (300 Schritte, lr 1e-3)** und Gradientenstruktur begrenzt, nicht durch die Gruppe.
- Neu-Fit unter der Obergrenze: Die Obergrenze L* bindet nichts. Aber ein reines Potenzgesetz (b ≈ 0,51) würde bei L* einen **negativen** Verlust ergeben und muss daher irgendwo zwischen L = 10 und 4.080 abknicken; wo, sagen die vier Punkte nicht. Vorhersagen für L = 40 reichen von 2,25 bis 2,37 — ein Lauf mit L = 20 und 40 trennt die Modelle sauber.
- Nebenbefund: Die 13,5 h von T-005 stecken zu 6,3 h in **einer** Störzelle (qire_L5, ein Seed 385 min statt ~9 min). Bereinigt ≈ 7,2 h.

## 2. Was gemessen / hergeleitet wurde

### 2.1 Aufbau
Skripte im Scratchpad (Pfad §6). `kern.py` ruft die Adapter-Kette mit freiem n auf und ist gegen `chatbot_adapter._kette` (n = 8, L = 3) verifiziert (Abweichung 0,0; R orthogonal bis 1,6e-15). Erzeuger: Tangenten ∂R/∂θ_{l,q} bei θ = 0 (antisymmetrisch, in so(2ⁿ), `jacfwd`). Abschluss: Lie-Abschluss mit Gram-Schmidt in float64, Toleranz 1e-9. Rang: Singulärwerte der Jacobi-Matrix θ → Rᵀ bei zufälligem θ, Rang = #(σ > 1e-9·σ₁); Lücke zum nächsten Wert stets ≥ 10⁷ (nächster ≈ 1e-16).

### 2.2 DLA je Lagenzahl (dim der von den Erzeugern der Lagen 1..L erzeugten Algebra)
| n | dim so(2ⁿ) | L=1 | L=2 | L=3 | L=4 | Ordnung von P |
|---|---|---|---|---|---|---|
| 3 | 28 | 3 | 8 | 16 | **28** | 7 |
| 4 | 120 | 4 | **120** | | | 15 |
| 5 | 496 | 5 | **496** | | | 21 |
| 6 | 2016 | 6 | 510 | 1023 | **2016** | — |
| 8 | 32.640 | 8 (kommutierend) | nicht geschlossen | | | |

Lesart: L = 1 ist abelsch (n kommutierende Ry), dim = n. Danach springt die Algebra, sobald der CNOT-Ring wirkt. Ohne CNOT bliebe sie abelsch (T-004: 5 geteilte Lagen = 1 Lage).
**Formel (Vermutung aus n = 3…6):** DLA = so(2ⁿ), dim = 2^(n−1)(2ⁿ−1). Das passt zur Literatur-Erwartung für Ry + CNOT [Gedächtnis: Larocca et al., Wiersema et al.]. Für n = 8 ist es **nicht** nachgerechnet; als Teilbeleg dienen nur die 1.600 unabhängigen Richtungen bei L = 200 (untere Schranke dim ≥ 1.600).
Das Lagenzahl-Minimum für die volle Algebra wächst nicht einfach mit n (2, 2, 4 für n = 4, 5, 6); eine Formel dafür habe ich nicht.

### 2.3 Jacobi-Rang gegen L (Parameter = n·L)
| n | L=1 | 2 | 5 | 10 | 20 | 40 | 100 | 200 | Sättigung |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 8 | 20 | 40 | 80 | 120 (30: 120; 28: 112) | 120 | — | **L\* = 30**, Rang 120 |
| 5 | 5 | 10 | 25 | 50 | — | — | 496 (99: 495) | — | **L\* = 100**, Rang 496 |
| 6 | 6 | — | — | — | — | — | 600 | — | L=335: 2010, **L\*=336**: 2016, 340: 2016 |
| 8 | 8 | 16 | 40 | 80 | 160 | 320 | 800 | 1600 | nicht erreicht, L\* = 4.080 |
| 11 | 11 | 22 | 55 | 110 | 220 | 440 | 1100 | — | nicht erreicht, L\* = 190.558 |
(n = 8: L ≤ 40 mit vollem Rang, L = 100/200 sowie n = 11 mit Zufallsskizze von r = 8 bzw. 3 Vektoren; Skizze gegen vollen Rang bei n = 8, L = 40 geprüft: beide 320.)
Zahl der **effektiven Parameter** = min(n·L, d(n)): überall, wo geprüft, **keine** Redundanz, auch nicht bei L > Ordnung von P. **Ausnahme n = 3** (Rang 3, 6, 8, 10, 12, …, 28 = 2L + 2 ab L = 2, Sättigung bei L = 13): dort liegen n·L-Parameter nicht frei, ein Sonderfall des 3er-Rings.
Sättigungsformel: L\*(n) = ⌈2^(n−1)(2ⁿ−1)/n⌉ — n = 4: 30, 5: 100, 6: 336, 8: **4.080**, 11: **190.558**. Beim Näherungswert L\* wird das kleinste Singulärwertverhältnis winzig (2,7e-6 bei n = 6), die Rangbestimmung ist dort numerisch heikel; nach L\* steigt es wieder.

## 3. T-005 und die Theorie
- Code: `BREITE, QUBITS = 768, 8; BLOCK = 2**QUBITS` → 256; R = blockdiag(R₁, R₁, R₁), Winkel über die 3 Blöcke geteilt, 24 Matrizen mit je eigenem θ (L×8). Je Matrix lebt R also in SO(256), nicht SO(768).
- Winkel je Matrix 8L = 8, 16, 40, 80 (L = 1, 2, 5, 10) gegen d = 32.640: 0,02–0,25 % der Gruppe. **L\* = 4.080** (nicht 10, nicht 100). Auch L = 20 (160) und L = 40 (320) liegen bei ≤ 1 % von L\*.
- Das Larocca-Kriterium (Überparametrisierung bei M ≥ dim DLA) ist damit für den Adapter auf **L ≳ 4.000** verschoben; bis dahin unterparametrisiert, jede neue Lage liefert eine neue unabhängige Richtung (Rang). Dass ein Ry-Kronecker-Adapter überhaupt eine „volle“ Algebra hat, bedeutet nur Steuerbarkeit, nicht Reichweite mit wenigen Winkeln.
- Annahme (nicht geprüft): W·R ist in R injektiv für das echte W; ein Rangverlust der Abbildung R ↦ W·R (W 256 × 768 bei v_proj!) würde die effektive Zahl senken. v_proj hat nur 256 Ausgabezeilen, Blockbild 256 × 256 je Block, also vermutlich ok.
- **Der Rang-Sättigungsort ist nicht der Verlust-Sättigungsort.** Er ist nur eine harte Obergrenze für die *Parametrisierung*. Der Verlust kann früher abflachen, weil (a) die Gradientenrichtungen nicht isotrop sind, (b) der Verlust einen Boden hat, (c) das Training nur 300 Schritte hat.
- Gemessene Winkel (T-005-Gewichte `results/t005_adapter/T-005`, 8 Seeds je Arm): mittleres |θ| 0,030 / 0,026 / 0,019 / 0,015, max 0,103 / 0,124 / 0,085 / 0,085 für L = 1, 2, 5, 10; ‖θ‖₂ je Matrix 0,106 / 0,131 / 0,156 / 0,174. Mit Adam, lr 1e-3, 300 Schritten ist jeder Winkel ≤ ~0,3 rad bewegbar; er nutzt weniger als die Hälfte. R ≈ I + Σθ_iT_i (erste Ordnung). In diesem Bereich ist der Gewinn ≈ Projektion des Gradienten auf den 8L-dimensionalen Raum. Bei isotroper Ausrichtung wüchse die L2-Projektion wie √(8L/d); der Fit zeigt b ≈ 0,51 — **konsistent mit √L**, aber keine Bestätigung (4 Punkte, Hypothese aus der Zahlenähnlichkeit). Test siehe §4/5.
- Folge: die „Sättigung“, die man in T-005-Aufbau bei L = 20, 40 sieht, wird eine Sättigung des *Gradienten-Signals/Budgets* sein, nicht der Algebra. Das ist der interessantere Befund für die Arbeit.

## 4. Neu-Fit der T-005-Kurve (Gewinn = 2,5616 − Verlust, 4 Punkte)
Daten: L = 1, 2, 5, 10 → 2,5174 / 2,4964 / 2,4442 / 2,4115 (Basis 2,5616). Fehlerannahme SE = 0,001 je Punkt (sd 0,0004–0,0029 / √8; **Annahme**, keine Rohdaten je Seed verwendet). 90 %-Bänder aus 400 parametrischen Bootstraps (nur Parameterunsicherheit, nicht Modellunsicherheit). Skript `fit.py`.

| Modell | RSS | L=20 | L=40 | L=100 | L\*=4.080 |
|---|---|---|---|---|---|
| Potenz a·L^b (a = 0,048, b = 0,51) | 1,2e-4 | 2,342 [2,339–2,346] | 2,249 [2,242–2,256] | 2,063 | **−0,74 (unmöglich)** |
| Wurzel a·√L (a = 0,049) | 1,2e-4 | 2,345 | 2,255 | 2,076 | −0,54 |
| log: d·ln(1+L/L₀) (d = 0,065, L₀ = 1,05) | 3,7e-5 | 2,368 | 2,325 | 2,267 | 2,03 |
| Potenz mit Boden E = 0, b ≤ 0,479 | 1,4e-4 | 2,350 | 2,267 | | ≥ 0 |
| Potenz mit Boden E = 1,5, b ≤ 0,373 | 5,4e-4 | 2,377 | 2,323 | | |
| Potenz mit Boden E = 2,0, b ≤ 0,297 | 1,2e-3 | 2,397 | 2,360 | | |
| Lomax (Boden E, 98 % Plateau bei L\*): E = 2,0 | 1,3e-3 | 2,308 | 2,224 | 2,134 | 2,01 |
| E = 2,2 | 3,6e-4 | 2,348 | 2,305 | 2,264 | 2,21 |
| E = 2,3 | 2,6e-5 | 2,380 | 2,356 | 2,334 | 2,30 |
| E = 2,35 | 2,5e-5 | 2,385 | 2,369 | 2,358 | 2,35 |

Nebenbedingung = „Verlust ≥ Boden E und Plateau spätestens bei L\*“. Lesart:
1. Mein freier Potenz-Exponent b = 0,51 (ungewichtet, Basis fest) liegt unter dem 0,546 aus Bericht 06; die Wahl der Gewichtung ändert das Ergebnis, das zeigt die Fragilität.
2. Die Obergrenze L\* = 4.080 **schränkt die Kurve bis L = 40 nicht ein**: Die Fits mit einem Plateau bei L\* und Boden E ≥ 2,2 passen genauso gut wie das freie Potenzgesetz (RSS 2,6e-5…1,2e-4); nur Böden E ≤ 2,0 passen schlecht (RSS ≥ 1,3e-3), weil die 98-%-Plateau-Bedingung bei kleinem Boden die Kurve verbiegt, die Daten sie aber nicht verbiegen wollen. Ob die 98-%-Schwelle sinnvoll ist, ist selbst eine Annahme.
3. Der **Boden** (Verlust ≥ E), nicht L\*, ist die wirksame Nebenbedingung: b ≤ 0,48 bei E = 0, ≤ 0,37 bei E = 1,5. Das Potenzgesetz muss abknicken, und das tut es bestimmt vor L\* — aber ein Boden unter ~2,2 (z. B. Vollfeintuning) ist unbekannt.
4. Aussage für den Lauf: Bei L = 40 sagen die Modelle 2,22 … 2,37 voraus, Spanne 0,15 ≫ Seed-SE ≈ 0,001. **Beide L (20, 40) trennen die Hypothesen.** Das Potenz/Wurzelgesetz würde LoRA r1 (2,2875) bei L ≈ 31 erreichen (Potenzfit); L = 40 läge dann darunter (2,25).

## 5. Ist ein QLM unendlich skalierbar? (ehrlicher Absatz)
Für den QIRE-Adapter gilt: Er ist **nicht** unendlich skalierbar, aber seine harte Grenze liegt sehr weit. Die Lie-Algebra der Ry-CNOT-Ring-Zelle ist (für n = 3…6 numerisch) die volle so(2ⁿ); die Parametrisierung ist jenseits aller Redundanz frei, bis die Winkelzahl n·L die Gruppendimension 2^(n−1)(2ⁿ−1) erreicht (n = 8: bei L = 4.080, n = 11: bei L ≈ 190.000). Darüber bringen weitere Lagen im Adapter *beweisbar nichts mehr* (Rang 32.640, dann konstant). Das ist die Grenze „unendlich“, die man sauber sagen kann: **endlich, aber mit n = 8 weit entfernt von dem, was T-005 geprüft hat (L ≤ 10, 0,25 % der Dimension)**. In der Praxis dürfte die Skalierung nicht an der Algebra, sondern früher an (i) einem Verlustboden, (ii) dem Trainingsbudget, (iii) der Messgenauigkeit enden; die gemessenen Winkel (≤ 0,12 rad) legen nahe, dass bei 300 Schritten das Budget zuerst bindet. Ob die Kurve bei L = 20/40 noch wie √L wächst, ist offen, das ist der Lauf. „Unendlich“ für ein QLM allgemein folgt daraus **nicht**: das Argument gilt für diese Zelle, mit n = log₂(Breite) Qubits, und es sagt nichts über Daten-, Rechenzeit- oder Generalisierungsskalierung. Es gilt auch nicht für echte Quantenhardware (Barren Plateaus, Rauschen; hier klassisch simuliert).

## 6. Vorschlag für einen vorregistrierten Lauf (Erkundung → Lauf)
Aufbau wie T-005 (Daten, Aufteilung, 300 Schritte, Batch 4, Adam, Clip 1,0, 8 Seeds, `chatbot_kurve` mit LAGEN erweitert). Neue Mess-Seeds z. B. 60001–60008, Tuning 60401/60402 (nicht 900–999).
- **Arme:** qire_L10 (Brücke, lr 1e-3 übernommen), qire_L20, qire_L40 (lr-Gitter 3e-4/1e-3/3e-3, Verlängerung wie T-005; Optimum könnte mit L nach unten rücken), lora_r1 (Kontrolle).
- **Hypothesen (Holm über 3):** K1 Monotonie L10 → L20 → L40 (Page + IUT wie T-005). K2 qire_L40 − lora_r1 (zweiseitig + TOST ±20 % Gewinn; Potenzgesetz sagt −0,04, Sättigungsmodelle +0,08…+0,10). K3 Form: Gewinn(L40) − √4·Gewinn(L10)·… (Potenz-vs-Sättigung); einfachere Auswertung: Gewinn L20/L10 und L40/L20 jeweils gegen 2^b (b = 0,5 ⇒ 1,41), beschreibend mit Bootstrap über Seeds.
- **Positivkontrolle:** qire_L10 reproduziert T-005 (2,4115 ± 0,003).
- **Optionale Kontrollen (je ca. 1–2,5 h):** (a) qire_L10 mit 600 Schritten: trennt Budget- von Algebra-Grenze; (b) „SO(256)-Decke“: volle Block-Drehung, Cayley-Parametrisierung je Matrix (783.360 Parameter, dichte 256 × 256-Matrizen, ähnliche Zeit wie LoRA), liefert die Obergrenze L → ∞ direkt; (c) erste-Ordnung-Test: Gradient des Verlusts nach Rᵀ in θ = 0 (eine Messung, keine Training), kumulierte Projektion auf die L-Lagen-Tangenten gegen √L.
- **Erwartete Effekte:** L20: 2,34–2,38, L40: 2,25–2,37 (Spanne aus §4), Seed-sd 0,001–0,003. Unterschiede der Modelle ≫ sd, TOST-Gleichwertigkeit mit LoRA wäre nur erreichbar, falls L40 zufällig innerhalb ±0,055 liegt (Potenzmodell ja).
- **Laufzeit (Maltes PC, 6 Threads).** Aus den T-005-Rohdaten, bereinigt: s/Schritt ≈ 1,05 + 0,06·L (L1 1,17, L2 1,09, L10 1,65, lora 0,87); Zelle = 300 Schritte + ca. 130 s Auswertung. Prognose (Extrapolation, unsicher): L20 ≈ 2,25 s/Schritt → **13,6 min/Zelle**; L40 ≈ 3,45 s/Schritt → **19,8 min/Zelle**. Dazu Referenzzellen L10 10,5, lora 6,1 min.
| Posten | Zellen | Zeit |
|---|---|---|
| lr-Suche L20 (3 lr × 2 Seeds) | 6 | 1,4 h |
| lr-Suche L40 | 6 | 2,0 h |
| Messung L20, 8 Seeds | 8 | 1,8 h |
| Messung L40 | 8 | 2,6 h |
| Messung L10 (Brücke) | 8 | 1,4 h |
| Messung lora_r1 (Kontrolle) | 8 | 0,8 h |
| **Summe Kern, ungestört** | | **≈ 10 h** |
Mit Störungen (T-005: eine Zelle 385 min statt 9) und einer Gitterverlängerung: 12–16 h; auf freiem Rechner über Nacht plus halber Tag. Optionale Kontrollen +1 bis 2,5 h je. Nur L40 + L20 ohne Brücke/LoRA (bitgleich zu T-005-Werten nicht möglich, andere Seeds): ≈ 7,8 h.
Zum Vergleich: T-005 = 13,5 h Summe der Rohdaten für 56 Zellen = **7,2 h ohne die eine Störzelle**; die Vorregistrierung nannte 8,2–9,0 h (mit Tuning) — die Abweichung ist also fast vollständig diese Störzelle, nicht ein falsches Zeitmodell.

## 7. Deutung / langweiligste Erklärung
- Die Langweiligste für „Rang = n·L“: Jede Lage fügt n Richtungen hinzu, die Gruppe ist riesig; der Satz ist fast trivial. Der Gehalt liegt in der Zahl L\* und im Fehlen von Redundanz (außer n = 3).
- Die langweiligste Erklärung für √L: lineares Regime mit isotropem Gradienten. Der Befund b ≈ 0,5 ist dann Folge der Geometrie, nicht „Quantenskalierung“.
- Grenzen: 4-Punkt-Fit (1 Freiheitsgrad), SE angenommen, n = 8-DLA nicht direkt geschlossen, Rang nur an einem Zufallspunkt, nur Ry+CNOT-Ring (nicht Clifford-Zelle, nicht LayerNorm-/Daten-Effekte).

## 8. Offene Fragen
1. Ist DLA(n = 8) wirklich so(256)? (Direkter Abschluss braucht ≈ 8,5 GB float64 oder sparse Darstellung.)
2. Warum braucht n = 6 vier Lagen, n = 4/5 nur zwei (n = 3: vier)? Formel fehlt.
3. Wo knickt die echte Kurve? Boden des Verlusts bei diesem Modell (Vollfeintuning-Referenz)?
4. Gilt √L wirklich über den ersten-Ordnung-Test (c)?

## 9. Pfad der Skripte
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v5/`: `kern.py` (Kette, Verifikation), `dla.py` (Lie-Abschluss), `jac.py` (Jacobi-Rang), `fit.py` (Neu-Fit), Ergebnisse `dla_n*.json`, `jac_n*_*.json`, `jac*.log`. Quellen: Tagebuch `UEBERARBEITUNG_2026-09-23/A/PROJEKTTAGEBUCH.md` (T-004/T-005), `quantum_sim/chatbot_adapter.py`, `chatbot_kurve.py`, Rohdaten `results/experiments/T-005__20260930_084358_02.json`, Adapter `results/t005_adapter/T-005`.
