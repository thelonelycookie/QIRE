# T-007 (Institut I-043) — Fastfood-Adapter im Sprachmodell

**Stand:** 7. Oktober 2026, Vorregistrierung, **vor** jeder T-007-Zahl. **Kein Mess- und kein Tuning-Seed
gerechnet; Lauf wartet auf Maltes Freigabe.** Maßgeblich ist der Modulkopf von
`quantum_sim/chatbot_fastfood.py` (Stand Commit); Änderungen erst ins Tagebuch. T-004 bis T-006
(`chatbot_adapter.py`, `chatbot_kurve.py`, `chatbot_skalierung.py`) bleiben unverändert; T-007 importiert sie.
Herleitung: `INSTITUT/LABORBUCH.md` I-044 ERGEBNIS (Struktur trägt dort, wo die Eingabe schon gute Merkmale hat:
CNN-Kopf +1,8 pp mit 85× weniger Parametern), `INSTITUT/abteilungen/struktur_gross.py` (Fastfood-Baustein),
Tagebuch T-006 (QIRE-Adapter sättigt, b = 0,08; L40 holt 62 % des LoRA-Gewinns).

## 1. Frage

T-006: Der QIRE-Adapter (Drehungen) sättigt, LoRA r=1 bleibt klar vorn (0,273 gegen 0,168 Gewinn). Kann eine
**strukturierte Fastfood-Abbildung mit O(d) Parametern** (hier: gleich viele wie LoRA r=1) die Lücke schließen?
Das ist die „Schicht hinter gelernten Merkmalen“ aus I-044, jetzt im Sprachmodell. Wirkung ist klassisch (I-038);
Sprachregelung „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.

## 2. Der Adapter

Je Projektionsmatrix W (q_proj 768→768, v_proj 768→256; 12 Lagen = 24 Matrizen) gilt y = W x + ΔW x mit

ΔW x = c · (H D_aus ⋯) fold (H D_ein ⋯) x

- H = orthonormale Walsh-Hadamard-Matrix 256×256 (= H₁₆ ⊗ H₁₆, zwei kleine Matmuls), **blockweise** auf den
  3 Blöcken zu 256 der Eingangsdimension (wie die QIRE-Blöcke).
- **Stufen je Seite s:** s trainierte Diagonalen D_ein (Länge d_in), je gefolgt von H; ist d_aus < d_in (v_proj),
  werden die d_in/d_aus = 3 Blöcke summiert (Faktor 1/√3); dann s trainierte Diagonalen D_aus (Länge d_aus), je
  gefolgt von H. ff_adapter: s = 1, also **H D₂ H D₁** (zwei Diagonalstufen). ff_adapter_breit: s = 2, vier Stufen.
- **Ausgang skaliert:** c = 2,0 fest, nicht trainiert (analog LoRA alpha/r = 2; die Lernrate des Gitters nimmt den Rest auf).
- **Init so, dass ΔW = 0:** alle Diagonalen Zufallsvorzeichen ±1 (Generator je Seed), nur die **letzte** Diagonale = 0.
  Start = Basismodell exakt (Vortest: Logits-Differenz 0). Gradient fließt am Start nur in die letzte Diagonale, ab dem
  ersten Schritt in alle (Vortest).
- Mathematisch ist H D H ein „dyadischer Faltungsoperator“ im Walsh-Bereich; ΔW ist also eine eingabe-unabhängig
  gemischte, aber voll besetzte Abbildung, kein Rang-1-Term wie bei LoRA. Das ist die Frage der Studie, nicht ihr Ergebnis.

## 3. Arme und Parametergleichheit (fairer Vergleich)

| Arm | Aufbau | Parameter | Schritte |
|---|---|---|---|
| basis | kein Training | 0 | – |
| qire_L10 | Brücke zu T-005/T-006, **Positivkontrolle 2,4111** | 1.920 | 300 |
| lora_r1 | Referenz (r=1, alpha=2) | 30.720 | 300 |
| ff_adapter | Fastfood, 2 Diagonalstufen | **30.720** | 300 |
| ff_adapter_breit | Fastfood, 4 Diagonalstufen (2 Stufen mehr) | 61.440 | 300 |

**Parametergleichheit, exakt:** LoRA r=1 hat je Matrix d_in (A) + d_aus (B): q 768+768 = 1.536, v 768+256 = 1.024,
×12 Lagen = **30.720**. ff_adapter hat je Matrix d_in (D₁) + d_aus (D₂): **dieselben Zahlen, 30.720**. Der Vortest prüft
die Gleichheit und die Zahl am echten Modell (`trainierbar(m)`). ff_adapter_breit = 2× (kein Paritätsarm; er prüft nur,
ob mehr Stufen helfen). Daten, Aufteilung, Verlust (nats/Token, nur Antwort-Tokens), Metrik `val_verlust`, Batch 4, Adam,
Clip 1,0, 300 Schritte: wie T-004 bis T-006.

## 4. Seeds und Lernraten

- Messung **70001–70008**, Tuning **70401/70402** (disjunkt zu T-004 40001–40008/40401–40402, T-005 50001–50008/50401–50402,
  T-006 60001–60008/60401–60402; `pruefe_theorie` prüft es).
- qire_L10 und lora_r1 übernehmen lr = 1e-3 aus T-005 (innen; wie T-006; `pruefe_theorie` vergleicht mit der Rohdatei).
- ff_adapter und ff_adapter_breit werden getunt: **Startgitter 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2** (reicht von Anfang an bis
  1e-4 nach unten; Lehre aus T-006: dort lag das Optimum von L20/L40 bei 3e-4, also nahe dem damaligen Rand), Verlängerung
  ×3 bzw. ÷3 in **beide** Richtungen, höchstens 3 Schritte (`sweep_lr`), Tuning-Seeds, volles Budget (300 Schritte), Abstimmmenge.
- **Randprüfung:** Optimum nie am Rand der gemessenen Raten, sonst „ACHTUNG Rand“ im Urteil. **Alle Tuningzellen und die
  Mittel je Rate stehen in den Rohdaten** (`config["tuning"]`, `config["lr"][arm]["mittel"]`). `rand_aus_tuning` rechnet den
  Randvermerk unabhängig daraus nach; `analyse` meldet Abweichungen (`lr_inkonsistent`), `q_audit` prüft sie.

## 5. Hypothesen (Holm über genau 3, α = 0,05)

- **H1 ff_adapter − lora_r1** (gleiche Parameterzahl), zweiseitig (Wilcoxon) **plus TOST**, Grenze ±20 % × |Gewinn lora_r1|
  (Erwartung ≈ ±0,055). TOST wird vor der Signifikanz geprüft (wie T-004 bis T-006).
  Urteile: `FF_BESSER` / `GLEICHWERTIG` / `LORA_BESSER` / `UNENTSCHIEDEN`.
- **H2 ff_adapter < qire_L10**, Wilcoxon einseitig (ff besser). **Nicht parametergleich** (30.720 gegen 1.920): Brücke zu
  T-005/T-006, kein Effizienzbeweis. Urteile: `FF_BESSER_ALS_QIRE` / `KEIN_NACHWEIS`.
- **H3 ff_adapter_breit < ff_adapter**, Wilcoxon einseitig (mehr Stufen helfen). Urteile: `BREIT_BESSER` / `KEIN_NACHWEIS`.

Rangfolge: `HU_UNVOLLSTAENDIG` → `HU_DIVERGENZ` (nicht endliche Werte) → `HU_POSKONTROLLE` → `HU_AUFLOESUNG` → Urteil.
- **Positivkontrolle:** Mittel qire_L10 über 8 Seeds = **2,4111 ± 0,003** (T-006, andere Seeds). Verfehlt → `HU_POSKONTROLLE`
  für alle drei, keine Aussage.
- **Auflösung:** lora_r1 trennt sich von basis (Wilcoxon zweiseitig p < 0,05); sonst `HU_AUFLOESUNG`, kein Nullbefund.
- **Erreichbarkeit (Regel 2):** 8 Seeds, Wilcoxon einseitig min 0,0039, zweiseitig 0,0078, je ×3 (Holm) ≤ 0,023 < 0,05.
  Seed-Streuung in T-004..T-006 0,001–0,004 nats; TOST-Grenze ±0,055 ist damit auflösbar.
- **Beschreibend (ohne Holm):** Gewinn je Arm, Anteil am LoRA-Gewinn, Gewinn je Parameter, s/Schritt, ff_adapter_breit − lora_r1,
  Randvermerke.
- **Vorher angesagt (Erwartung, keine Hypothese):** Gewinn ff_adapter wahrscheinlich zwischen qire_L10 (0,150) und lora_r1
  (0,273); H1 am ehesten `LORA_BESSER` oder `UNENTSCHIEDEN`, H2 plausibel, H3 unsicher. Ein Gewinn für ff_adapter würde heißen:
  bei gleicher Parameterzahl trägt eine Struktur, die alle Ausgänge mischt, mehr als ein Rang-1-Term. Die langweiligste
  Erklärung für jeden Gewinn bleibt klassisch (I-038).

## 6. Vortest (Pflicht, vor dem Lauf)

Ohne Modell (Container, bestanden, `vortest_statistik`): Hadamard gegen scipy, Fastfood-Vorwärtslauf gegen explizite
Matrix (scipy-Hadamard) für q_proj und v_proj bei s = 1/2, ΔW = 0 am Start, Gradientenfluss, Normerhaltung,
Parameterzahl und **Parametergleichheit ff = lora_r1 = 30.720**, Mini-Llama (Zufallsgewichte, gleiche Form: Breite 768,
v_proj 768→256; kein Download) für Einbau, Start = Basis, Lernfähigkeit, eingefrorene Basis; Gitterverlängerung und
Randnachrechnung aus Rohdaten in beiden Richtungen; **26 Szenarien** (jeder Urteilszweig von H1–H3 inklusive Holm-Fall,
`HU_DIVERGENZ`, `HU_POSKONTROLLE`, `HU_AUFLOESUNG`, Seeds ungleich); **8 Mutationen** (Metrikfeld, Signifikanz vor TOST,
Auflösung nach Richtung, absolute TOST-Grenze, H2 zweiseitig, H3 vertauscht, Holm aus, Positivkontrolle aus), jede wird
entdeckt. Die Zelle (`zelle`) wurde an einem Mini-Llama mit nachgebauter Datenlage durchgespielt (deterministisch,
Verlust fällt).
Mit Modell (Maltes PC, `vortest`): zusätzlich Start-Logits = Basismodell für alle Arme, `trainierbar` je Arm gegen die
Tabelle, **Bitgleichheit** der T-007-Zelle mit der T-005-Zelle (qire_L10, lora_r1) nach 3 Schritten, ff-Zellen laufen
(3 Schritte, Verlust endlich, nahe Basis). `lauf` startet nur, wenn alles besteht.
**Noch nicht geprüfbar im Container:** echte s/Schritt der ff-Arme und die Bitgleichheit (Modell liegt nur auf Maltes PC).

## 7. Laufzeit (Maltes PC, 6 Threads, ungestört)

Aus T-006-Ist: qire_L10 1,64 s/Schritt, 10,4 min/Zelle; lora_r1 0,84 s, 5,9 min. Zelle = 300 × s + ca. 100 s.
**ff-Zeiten sind Annahmen** (Hadamard als zwei kleine Matmuls je Block, ca. +10–20 % gegen LoRA; breit ca. +40 %):
ff_adapter ≈ 1,0 s (6,7 min/Zelle), ff_adapter_breit ≈ 1,2 s (7,7 min). `zeit` misst sie vor dem Start.

| Posten | Zellen | Zeit |
|---|---|---|
| lr-Suche ff_adapter, ff_adapter_breit (5 Raten × 2 Seeds je) | 20 | ≈ 2,2 h |
| Messung qire_L10, lora_r1, ff_adapter, ff_adapter_breit (je 8 Seeds) | 32 | ≈ 4,0 h |
| **Kern, ungestört** | | **≈ 6,3 h** |
| je Gitterverlängerung (2 Zellen je Arm) | +4 | +0,4 h |

Mit Störungen und einer Verlängerung 7–9 h (T-005: eine Zelle brauchte einmal 385 min statt 9). Reihenfolge:
qire_L10 (Positivkontrolle früh sichtbar), lora_r1, ff_adapter (H1, H2), ff_adapter_breit (H3). Bricht der Lauf ab,
stehen H1 und H2 zuerst.

## 8. Wiederaufnahme, Q-Werkzeuge

- Wiederaufnahme **je Zelle** (`results/t007_zwischenstand/T-007_wiederaufnahme.json`, Fingerabdruck der Konfiguration;
  andere Konfiguration → Abbruch statt Überschreiben); Adaptergewichte je Zelle in `results/t007_adapter/`.
  Gleicher Startbefehl noch einmal = Fortsetzung. Zwischenstände committen.
- `q_audit` (Eintrag `T-007`: vor dem Lauf „noch nicht gerechnet“; nach dem Lauf Struktur, Parametergleichheit, Tuning-Rohdaten,
  Randvermerk; **Platzhalter `TAGEBUCH_T007`** für Mittel je Arm und H1–H3, beim Eintrag der Ergebnisse zu füllen) und
  `q_summary` (Frage eingetragen) sind angelegt.

## 9. Startbefehl (Maltes Windows-PC, PowerShell, Ordner A)

```powershell
cd "C:\Users\Malte\Downloads\claude\projects\QIRE AI\A"
git pull --no-rebase
.\py.cmd -m quantum_sim.chatbot_fastfood vortest          # Pflicht, 10-20 min
.\py.cmd -m quantum_sim.chatbot_fastfood zeit 6 4         # optional, Ist-Zeiten
.\py.cmd -m quantum_sim.chatbot_fastfood lauf *> results\T-007_run.log
```

Auswertung nachholen: `.\py.cmd -m quantum_sim.chatbot_fastfood auswertung T-007`. Der PC darf währenddessen nicht schlafen;
6 Threads, keine anderen Rechenläufe (`institut.cmd` pausieren). Unterbrechung: derselbe `lauf`-Befehl setzt fort.
**Vor dem Start auf Maltes Zustimmung warten** (Nutzungslimit).
