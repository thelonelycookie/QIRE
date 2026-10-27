# w7: komplexwertige Phasen-Schichten mit Interferenz (Erkundung, kein Befund)

## 1. Kurzfassung
- **Ampel: gelb.** Kleiner, konsistenter Vorsprung von |z|²-Schichten gegen reell+ReLU bei knappem Training (MNIST bis +2,1 pp bei 5k Parametern), aber er schrumpft mit Größe (18k: +0,3 bis +0,7 pp) und ist auf Fashion fast null.
- Komplex dicht hat bei gleichen Parametern **automatisch gleiche FLOPs** (Eingang reell: 2 reelle Matmuls bei halber Breite). Dort ist der Vorsprung die ehrlichste Zahl: MNIST +1,2 / +0,4 pp, Fashion +0,9 / +0,3 pp, LM -0,06 bpc.
- Nur-Phase hat bei gleichen Parametern **2x FLOPs**; bei gleichen FLOPs (halbe Parameter) liegt es im MNIST/Fashion-MLP nur noch gleichauf bis leicht über reell; im LM -0,04 bpc bei 33 % weniger FFN-Parametern.
- Langweiligste Erklärung: |z|² = (a·x)² + (b·x)² ist eine Square-Aktivierung (Primer/z²) mit Paarsumme, keine "Interferenz"-Magie; Wirkung klassisch (I-038).
- LM stark untertrainiert (500 Schritte, bpc ca. 3,5); Vorsprünge bei knappem Training verschwinden erfahrungsgemäß (I-044-v3).

## 2. Idee, Bezug, Literatur
Gewicht W_jk = r·e^{iθ_jk}, r fest = 1/sqrt(n); z = W x (x reell), Ausgang Born |z|² = (Σcos θ·x)² + (Σsin θ·x)². Quanten-inspiriert: Amplituden interferieren, Messung = Betragsquadrat. Verwandt [Gedächtnis]: komplexe Netze (Trabelsi et al. 2018), unitäre RNN (Arjovsky 2016), Phasen-/Unitary-Layer, Primer (z²-ReLU, So et al. 2021). Zähl-Konvention: 1 MAC = 2 FLOP; komplexe Mult. = 4 reelle (hier Eingang reell, daher 2 Matmuls = 4nH FLOPs).

## 3. Aufbau
- Netz: Eingang 256 (16x16) -> Schicht 1 (real+ReLU | kdicht | phase, jeweils danach LayerNorm ohne Parameter, gleich für alle) -> real Linear -> 10 Klassen. Phase: 1 Parameter je Eintrag plus 1 globale Skala, Bias je Einheit.
- Arme: **real**, **kdicht** (2 Param/Eintrag, gleiche Parameter), **phase** (gleiche Parameter), **phase_flop** (gleiche FLOPs wie real, also halbe Breite).
- Daten: MNIST/Fashion 10k Training, 5k Test (je Seed andere Teilmenge), Adam, Kosinus, 30 Epochen, Batch 256. Lernrate je Arm/Budget/Datensatz auf Validierung (letztes Viertel von 7,5k, Seed 960) aus {0,01; 0,03; 0,1; 0,3}. Seeds 960-964 (n=5), nur Tuning-Seeds.
- LM: Byte-LM UD_German-GSD (Test: erste 32768 Bytes), d=64, 2 Lagen, Kontext 64, Batch 16, **500 Schritte**, nur FFN-Aufwärtsprojektion (64->H) wechselt, Abwärts real; lr-Wahl auf Dev (250 Schritte, Seed 960), alle Arme lr 0,003.

### MLP, Test-Genauigkeit % (Mittel ± sd, n=5)
| Daten | Budget | Arm | H | Param | FLOPs | Acc | lr |
|---|---|---|---|---|---|---|---|
| mnist | 5k | real | 19 | 5083 | 10108 | 93,43 ± 0,29 | 0,01 |
| mnist | 5k | kdicht | 10 | 5240 | 10440 | 94,63 ± 0,35 | 0,03 |
| mnist | 5k | phase | 19 | 5084 | 19836 | 95,58 ± 0,33 | 0,3 |
| mnist | 5k | phase_flop | 10 | 2681 | 10440 | 94,71 ± 0,38 | 0,3 |
| mnist | 18k | real | 67 | 17899 | 35644 | 95,88 ± 0,24 | 0,03 |
| mnist | 18k | kdicht | 34 | 17792 | 35496 | 96,32 ± 0,27 | 0,01 |
| mnist | 18k | phase | 67 | 17900 | 69948 | 96,54 ± 0,20 | 0,1 |
| mnist | 18k | phase_flop | 34 | 9089 | 35496 | 96,00 ± 0,31 | 0,1 |
| fashion | 5k | real | 19 | 5083 | 10108 | 83,76 ± 0,46 | 0,3 |
| fashion | 5k | kdicht | 10 | 5240 | 10440 | 84,63 ± 0,43 | 0,01 |
| fashion | 5k | phase | 19 | 5084 | 19836 | 84,42 ± 0,45 | 0,03 |
| fashion | 5k | phase_flop | 10 | 2681 | 10440 | 83,10 ± 0,26 | 0,3 |
| fashion | 18k | real | 67 | 17899 | 35644 | 85,68 ± 0,35 | 0,01 |
| fashion | 18k | kdicht | 34 | 17792 | 35496 | 85,98 ± 0,25 | 0,01 |
| fashion | 18k | phase | 67 | 17900 | 69948 | 85,73 ± 0,19 | 0,03 |
| fashion | 18k | phase_flop | 34 | 9089 | 35496 | 85,41 ± 0,25 | 0,1 |

### Byte-LM, Test-bpc (kleiner = besser), FFN je Lage
| Arm | H | FFN-Param | FFN-FLOPs/Token | bpc (n=5) |
|---|---|---|---|---|
| real | 256 | 33088 | 65536 | 3,591 ± 0,014 |
| kdicht | 171 | 33067 | 65664 | 3,535 ± 0,061 |
| phase | 256 | 33089 | 98304 | 3,524 ± 0,024 |
| phase_flop | 171 | 22124 | 65664 | 3,550 ± 0,023 |
Bytes: Parameter x 4 (float32); Phase-Gewichte ließen sich als Winkel quantisieren (nicht getestet).

## 4. Deutung
- Boring: |z|² ist Square-Aktivierung; kdicht = reell dicht mit z² und Paarsumme. Der Vorsprung stammt wahrscheinlich aus der Aktivierung (ReLU-Baseline mit knappem Training, 30 Epochen), nicht aus Phasenstruktur. Eine Kontrolle "real mit z²" fehlt (siehe 5).
- Phase_flop (halbe Parameter, gleiche FLOPs) schlägt reell nur auf MNIST 5k klar; sonst ±. Nur-Phase spart also Parameter/Speicher, keine FLOPs (Muster wie Fastfood, I-045-v4); bei gleichen FLOPs gewinnt kdicht eher.
- Phase braucht hohe Lernraten (0,1-0,3, teils am Gitterrand) -> Lernraten-Vorbehalt. Lernraten auf einem Seed gewählt; reell wählte bei Fashion 5k 0,3 (Gitterrand, vermutlich Rauschen).
- Reelle Baseline ist klein (H=19); Abstand wächst nicht mit Größe, eher umgekehrt.
- Nicht getestet: komplexe Eingänge in tieferen Schichten (4 reelle Mult.), mehrschichtiges Interferenznetz, Training bis Konvergenz. LM nur 500 Schritte.

## 5. Vorschlag für vorregistrierten Lauf (nur nach Maltes Zustimmung)
- Hypothesen: H1 kdicht ≥ real+z² bei gleichen Parametern und FLOPs (MLP und LM, mit voller Konvergenz); H2 phase_flop ≥ real bei gleichen FLOPs; H3 phase bei gleichen Parametern bpc ≤ real - 0,02.
- Arme: real-ReLU, real-z² (Paarsumme ungepaart = Primer), real-z² gepaart (= kdicht), kdicht, phase, phase_flop, Niedrigrang gleiche FLOPs. Kontrolle: Zufallsphasen fest (nur Kopf trainiert).
- Seeds 10, Lernrate je Arm auf Tuning-Seeds in beide Richtungen, LM 3000+ Schritte, d=128. Effektgröße: 0,5 pp (MLP), 0,02 bpc (LM).
- Laufzeit: MLP ca. 20 min Container (4 Kerne), LM ca. 2-3 h Raven/PC.

## 6. Skripte
`INSTITUT/recherche/2026-10-27_welle/w7/`: `komplex.py` (Schichten, Zählung), `lauf_mlp.py`, `lauf_lm.py`, `auswerten.py`; Rohdaten `mlp_ergebnis.json`, `lm_ergebnis.json`, Logs. Ca. 25 min CPU insgesamt, 1 Thread (teils mit anderen Läufen konkurrierend).
