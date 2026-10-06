# T-006 — Adapter-Skalierung L = 20/40 (Nachfolger von T-005)

**Stand:** 6. Oktober 2026, Vorregistrierung, **vor** jeder T-006-Zahl. Maßgeblich ist der
Modulkopf von `quantum_sim/chatbot_skalierung.py` (Stand Commit); Änderungen erst ins Tagebuch.
Herleitung: `INSTITUT/recherche/2026-10-05_vertiefung/v5_lie_skalierung.md` §6,
`INSTITUT/PLAN_2026-10-06_gesamt.md` §2 P3. Code: T-004 (`chatbot_adapter.py`) und T-005
(`chatbot_kurve.py`) bleiben unverändert; T-006 importiert sie und fügt nur Arme, Seeds,
Hypothesen und Auswertung hinzu. Die Zelle ist `chatbot_kurve.zelle`.

## 1. Frage und Hintergrund

Wie wächst der Gewinn des QIRE-Adapters bei L = 10 → 20 → 40 Lagen, und kommt qire_L40
(7.680 Winkel) an lora_r1 (30.720) heran? Die Lie-Algebra ist so(256), der Jacobi-Rang ist
min(8·L, 32.640), die algebraische Sättigung läge bei L\* = 4.080. Bei L ≤ 40 begrenzt also
**nicht die Gruppe**. Die trainierten Winkel sind winzig (|θ| ≈ 0,02), der Adapter arbeitet im
linearen Bereich. Begrenzend wäre dann Trainingsbudget bzw. Gradientengeometrie (√L). Die frühere
Hypothese „Plateau beim Jacobi-Rang“ ist verworfen (v5). T-005: Gewinn 0,150 (L10) bei 0,274 (LoRA).

## 2. Arme (q_proj, v_proj der 12 Lagen = 24 Matrizen, CNOT-Ring, Batch 4, Adam, Clip 1,0)

| Arm | Aufbau | Parameter | Schritte |
|---|---|---|---|
| basis | kein Training | 0 | – |
| qire_L10 | Brücke, **Positivkontrolle** | 1.920 | 300 |
| qire_L20 | 20 Lagen | 3.840 | 300 |
| qire_L40 | 40 Lagen | 7.680 | 300 |
| lora_r1 | Kontrolle (T-004/T-005) | 30.720 | 300 |
| qire_L10_600 | qire_L10 mit doppeltem Budget, **optional**, außerhalb Holm | 1.920 | 600 |

Daten, Aufteilung, Verlust (nats/Token, nur Antwort-Tokens), Metrik `val_verlust`: wie T-005.

## 3. Seeds und Lernraten

- Messung **60001–60008**, Tuning **60401/60402** (disjunkt zu T-004 40001–40008/40401–40402 und
  T-005 50001–50008/50401–50402; `pruefe_theorie` prüft es).
- qire_L10 und lora_r1 übernehmen lr = 1e-3 aus T-005 (innen; `pruefe_theorie` vergleicht mit der
  Rohdatei). qire_L10_600 übernimmt die Rate von qire_L10.
- qire_L20 und qire_L40 werden getunt: Gitter 3e-4 / 1e-3 / 3e-3, Verlängerung ×3 bzw. ÷3
  (höchstens 3 Schritte) wie T-005 (`sweep_lr`). Das Optimum kann mit L nach unten rücken.
  Liegt es trotz Verlängerung am Rand, steht im Urteil „ACHTUNG Rand“.

## 4. Hypothesen (Holm über genau 3)

- **K1 Monotonie L10 → L20 → L40.** Exakter Page-Trendtest (k = 3, n = 8, einseitig: Verlust fällt
  mit L), dazu IUT über beide Nachbarstufen (Wilcoxon einseitig, max p < 0,05/3).
  Urteile: `STRENG_MONOTON` (Holm < 0,05 und IUT) / `WAECHST_MIT_PLATEAU` (nur Trend) / `KEIN_TREND`.
- **K2 qire_L40 − lora_r1,** zweiseitig plus TOST, Grenze ±20 % × |Gewinn lora_r1 gegen basis|
  (Potenzgesetz sagt −0,04, Sättigungsmodelle +0,08…+0,10; Gleichwertigkeit nur, falls L40 innerhalb
  ±0,055 liegt). Urteile: `QIRE_BESSER` / `GLEICHWERTIG` / `LORA_BESSER` / `UNENTSCHIEDEN`;
  TOST wird vor der Signifikanz geprüft (wie T-004/T-005).
- **K3 Form des Gewinns.** G(L) = Verlust(basis) − Verlust(L). Exponent b = log(G40/G10)/log 4
  (Potenzgesetz G ∝ L^b; b = 0,5 = Wurzelgesetz). H0: b = 0,5, zweiseitig, gepaarter Bootstrap über
  Seeds (B = 10.000, rng-Seed 20261006, p = 2·min(P(b\* ≤ 0,5), P(b\* ≥ 0,5)) mit (k+1)/(B+1)).
  Urteile: `SAETTIGT_UNTER_WURZEL` (Holm < 0,05, b < 0,5) / `STEIGT_UEBER_WURZEL` (b > 0,5) /
  `OFFEN_ZU_WURZEL` (Wurzelgesetz nicht verworfen, **keine** Gleichwertigkeit).
  Beschreibend dazu: G20/G10 und G40/G20 gegen 2^0,5 = 1,414 mit 95-%-Bootstrap-Intervall.
  Nicht positive Gewinne: `HU_UNVOLLSTAENDIG` (Exponent nicht definiert).

**Optional, beschreibend (ohne Holm):** qire_L10_600 − qire_L10 gegen qire_L20 − qire_L10 trennt
Budget- von Strukturgrenze: `BUDGETGRENZE` (L10_600 im Mittel ≤ L20) / `STRUKTURGRENZE_EHER`
(Zuwachs < Hälfte des L20-Zuwachses) / `GEMISCHT`.

## 5. Positivkontrolle und Rangfolge

- **Positivkontrolle:** Mittel qire_L10 über 8 Seeds = **2,4115 ± 0,003** (T-005, andere Seeds,
  Seed-sd 0,001–0,003). Verfehlt → `HU_POSKONTROLLE` für alle drei Tests, keine Aussage.
- Rangfolge: `HU_UNVOLLSTAENDIG` → `HU_POSKONTROLLE` → `HU_AUFLOESUNG` (lora_r1 trennt sich von
  basis, Wilcoxon zweiseitig p < 0,05, wie T-004/T-005) → Urteil.
- Erreichbarkeit (Regel 2): Page min (1/3!)⁸ = 6e-7; IUT min 0,0039 < 0,0167; Bootstrap min 1e-4;
  Wilcoxon zweiseitig min 0,0078 ×3 (Holm) ≤ 0,023.
- **Vorher angesagt (Erwartung aus v5 §4, keine Hypothese):** L20 2,34–2,38, L40 2,25–2,37,
  Seed-sd 0,001–0,003; Unterschiede der Modelle ≫ sd.
- Langweiligste Erklärung für b ≈ 0,5: lineares Regime mit isotropem Gradienten, also Geometrie,
  keine „Quantenskalierung“. Ergebnis wird so berichtet.

## 6. Wiederaufnahme, Q-Werkzeuge

- Wiederaufnahme **je Zelle** (`results/t006_zwischenstand/T-006_wiederaufnahme.json`, Fingerabdruck
  der Konfiguration; andere Konfiguration → Abbruch statt Überschreiben). Zwischenstand der Rohdaten
  und das beste Adaptergewicht nach jedem Arm (`results/t006_adapter/`). Ein Herunterfahren kostet
  höchstens eine Zelle. Gleicher Startbefehl noch einmal = Fortsetzung.
- `q_audit` (Eintrag `T-006`, vor dem Lauf „noch nicht gerechnet“; nach dem Lauf Tagebuchzahlen
  gegen Rohdaten ergänzen) und `q_summary` (Frage eingetragen) sind angelegt.

## 7. Laufzeit (Maltes PC, 6 Threads, ungestört)

s/Schritt ≈ 1,05 + 0,06·L (Fit an T-005; Extrapolation, unsicher); Zelle = Schritte × s + ca. 130 s.
Zelle: L10 10,4 min, lora 6,5, L20 13,4, L40 19,4, L10_600 18,7 min.

| Posten | Zellen | Zeit |
|---|---|---|
| lr-Suche L20 und L40 (3 Raten × 2 Seeds je) | 12 | 3,2 h |
| Messung L10, lora, L20, L40 (je 8 Seeds) | 32 | 6,6 h |
| **Kern, ungestört** | | **≈ 9,8 h** (v5: ≈ 10 h) |
| optional qire_L10_600 | 8 | +2,5 h |

Mit Störungen und einer Gitterverlängerung 12–16 h (T-005: eine Zelle 385 min statt 9). Über Nacht
plus halber Tag. Reihenfolge: qire_L10 (Positivkontrolle früh sichtbar), lora_r1, qire_L40, qire_L20,
zuletzt qire_L10_600. Bricht der Lauf ab, stehen K2 und K1 zuerst. `zeit` misst vor dem Start die
Ist-Zeiten (s/Schritt je Arm).

## 8. Vortest (Pflicht, vor dem Lauf)

Ohne Modell (Container): `vortest_statistik` — Theorie für L10/20/40, Page gegen scipy, Exponent-Test,
Gitterverlängerung, 27 Szenarien (jeder Urteilszweig von K1–K3, HU_*, Positivkontrolle), 9 Mutationen,
Budgetmarken. Mit Modell (Maltes PC): `vortest` ergänzt Startwert = Basismodell-Logits für alle Arme,
Parameterzahlen, Bitgleichheit der T-006-Zelle mit der T-005-Zelle (qire_L10, lora_r1) nach 3 Schritten.
`lauf` startet nur, wenn alles besteht.

## 9. Startbefehl (Maltes Windows-PC, PowerShell, Ordner A)

```powershell
cd "C:\Users\Malte\Downloads\claude\projects\QIRE AI\A"
git pull --no-rebase
.\py.cmd -m quantum_sim.chatbot_skalierung vortest          # Pflicht, 10-20 min
.\py.cmd -m quantum_sim.chatbot_skalierung zeit 6 4         # optional, Ist-Zeiten
.\py.cmd -m quantum_sim.chatbot_skalierung lauf *> results\T-006_run.log
```

Auswertung nachholen: `.\py.cmd -m quantum_sim.chatbot_skalierung auswertung T-006`.
Der PC darf währenddessen nicht schlafen; 6 Threads, keine anderen Rechenläufe (`institut.cmd`
pausieren). Unterbrechung: derselbe `lauf`-Befehl setzt fort.
