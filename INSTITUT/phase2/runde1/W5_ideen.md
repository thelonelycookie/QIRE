# W5 Bitflip-Robustheit: Ideen (Phase 2, Runde 1, 02.11.2026)

Status: Ideenliste, keine Läufe, kein Code ausgeführt. Alle Literaturzahlen sind aus Abstracts und ungeprüft (Quelle: `WISSEN/quellen.json`, Feld F1/K3/K4). Eigene Z3-Ergebnisse sind Erkundungen, keine Befunde.

## Gemeinsamer Rahmen für alle Vortests
- **Fehlermodell:** Jedes gespeicherte Bit (Gewichtscodes UND Skalen/Bias in fp16) kippt unabhängig mit Wahrscheinlichkeit p, nur zur Testzeit. p aus {1e-4, 1e-3, 3e-3, 1e-2, 3e-2}. Mindestens 20 Fehlermuster je Modell und p.
- **Daten/Modell:** MNIST 16x16 (wie `labor/gemeinsam.py: bilder`), MLP wie in Z3. Training einmal, Flips nur im gespeicherten Zustand. CPU je Idee ≤ 30 min, nur Tuning-Seeds (nie Mess-Seeds).
- **Kernpunkt, der oft übersehen wird:** Eine orthogonale Rotation erhält die L2-Energie des Fehlers. Der erwartete MSE durch einen Flip ist in der rotierten Domäne derselbe. Sie ändert nur die Form: ein großer Einzelfehler (MSB-Flip) wird auf n Koordinaten verteilt (kleiner L∞, gaußartig). Ob das dem Netz nützt, ist die empirische Frage. Gegen die Skala selbst (fp16-Exponentenbit) schützt sie nicht.
- **Pflichtgegner (aus F1/S-002):** Clipping (+ RandBET) [arXiv:2006.13977], [arXiv:2104.08323]; SECDED/Hamming [arXiv:2508.12347]; Gewichtsskalierung [arXiv:2411.18993]; Gruppenskalen g16/g32; Lloyd-Gitter ohne Rotation; 2-Bit-Gitter ist evtl. von sich aus robuster [arXiv:2603.22770].
- **Bytes-Regel:** Alle Arme ≤ Bytes des Rotationsarms (Skalen, Bias, Vorzeichenvektor n/8 B je Schicht, Padding 784->1024 eingerechnet; Vorzeichenvektor ggf. aus Seed erzeugbar = 0 B, dann ausweisen). Ergebnis als Pareto (Bytes x p), nicht als Einzelpunkt.

## Idee 1: Rotation gegen Clipping, Skalenbits inklusive (Kernvortest)
- **Hypothese:** Bei gleichen Bytes und gleicher Bitzahl b (4 und 3) hält ein hadamard-rotiertes Modell mit Zeilenskala bei mittleren p (1e-3 bis 1e-2) mehr Genauigkeit als dasselbe Gitter ohne Rotation, aber NICHT mehr als ein Modell mit Clipping (gelerntes/abgeschnittenes Gitter). Erwartung ehrlich: Gleichstand mit Clipping.
- **Nächste bekannte Arbeit:** [arXiv:2603.16382] (RoR, Householder auf Aktivierungen, Gegner nur ungeschützt, Selbstbericht). Gegenüber: [arXiv:2006.13977].
- **Stärkster Gegner:** Clipping + RandBET [arXiv:2006.13977] (F1); danach g16-Gruppenskala [K3].
- **Quantenbezug:** Analogie (unitäre Basiswahl verteilt Information wie ein Code, Reed-Muller 1. Ordnung). Keine Quantenwirkung (I-038).
- **Langweiligste Erklärung:** Rotation = Ausreißerglättung = Clipping in anderer Verkleidung (S-001, S-007). Dann ist der Zugewinn gleich null, sobald Clipping dabei ist.
- **Billigster Test:** Ein MLP, 5 Arme: roh, roh+Clip, Hadamard, Hadamard+Clip, g16. Flips auf gespeicherten Codes plus fp16-Skalen. Reine Inferenzsimulation, ~10 min CPU. Auswerten: Genauigkeit gegen p, Wilcoxon über Fehlermuster (Tuning-Seeds).
- **Bytes-Fairness:** Roh+Clip: gleiche Codes, 2 B je Zeile für Skala + Clipwert je Schicht (Vernachlässigbar). Hadamard: + n/8 B Vorzeichen, + Padding. g16: 1 Bit/Gewicht Skalen (S-004) = deutlich mehr Bytes; wird entweder mit kleinerem b angesetzt (b-1, damit Bytes gleich) oder als "teurer Gegner" ausgewiesen. Ziel nach Muster des LABORBUCH-Byte-Budgets (Zeile ~1925).

## Idee 2: Skalenbits sind die Schwachstelle: wenige Skalen (Rotation) gegen viele (g16)
- **Hypothese:** Ein Flip im Exponenten einer fp16-Skala zerstört eine ganze Gruppe (Faktor bis 2^16). g16 hat 16x mehr Skalen als eine Zeilenskala und damit eine viel größere Angriffsfläche; Rotation mit Zeilenskala braucht wenige Skalen. Bei gleichen Bytes ist der Rotationsarm bei p ≥ 1e-3 robuster als g16, selbst wenn g16 ohne Flips besser ist. Der Vorteil wäre dann kein Rotationsvorteil, sondern "weniger Skalen = weniger Angriffsfläche".
- **Nächste bekannte Arbeit:** [arXiv:2411.18993] (Skalierung als Schutz), [arXiv:2606.02430] (Integer robuster als FP16/BF16 bei Soft Errors, passt zum Skalenproblem).
- **Stärkster Gegner:** g16/g32 mit geschützten Skalen (Skalen dreifach gespeichert oder als int8 im Logarithmus, kostet wenige Bytes) und SECDED [arXiv:2508.12347] nur auf den Skalen.
- **Quantenbezug:** keiner (reine Methodik). Nur der Rahmen "Fehlerkorrektur" bleibt Analogie.
- **Langweiligste Erklärung:** Jeder Entwurf mit geschützten Skalen (Log-Skala, Majority) hebt den Nachteil von g16 auf; dann bleibt nichts übrig.
- **Billigster Test:** Fällt aus Idee 1 ab (gleiche Simulation, nur Skalenflips getrennt gezählt: "nur Code-Bits kippen" gegen "nur Skalenbits kippen"). Zusatz ≈ 5 min.
- **Bytes-Fairness:** Skalenschutz (z. B. 3x) wird mitgezählt. Vergleich: g16 mit Majority-Skala gegen Hadamard + Zeilenskala mit gleichem Schutz.

## Idee 3: Redundanter Hadamard-Rahmen (analoger Fehlerkorrekturcode) gegen SECDED
- **Hypothese:** Speichert man die rotierten Gewichte mit einer kleinen Übervollständigkeit (z. B. 12,5 % zusätzliche Hadamard-Zeilen, Rahmen mit fester Redundanz), kann ein kleiner Prüfwert-Satz große Einzelfehler lokalisieren und teilweise korrigieren ("nur große Ausreißer korrigieren, kleines Rauschen tolerieren"). Bei gleichen Bytes (+12,5 %) schlägt das den gleichen Zusatz an Bytes, in SECDED oder in 1 Extra-Bit gesteckt, bei mittleren p.
- **Nächste bekannte Arbeit:** [arXiv:2602.00076] (reellwertige lineare Prüfungen auf Gewichten ohne Zusatzparameter); [arXiv:2603.03723] (analoge ECC, kleines Rauschen plus Ausreißer).
- **Stärkster Gegner:** SECDED/Hamming auf den Codes [arXiv:2508.12347], DeepNcode [arXiv:2405.13891], Schutzmarge bis 7,6x (4 Bit, ungeprüft).
- **Quantenbezug:** Analogie (Stabilisator/Reed-Muller, klassisch). Echter Quantenbezug nur bei QEC selbst (F1).
- **Langweiligste Erklärung:** Es ist klassische Rahmentheorie/ECC mit anderem Namen; SECDED korrigiert Einzelfehler exakt, ein analoger Code kann das gegen binäre Flips kaum besser. Der Dekoder ist aufwendiger.
- **Billigster Test:** Simulation an einer fertig trainierten Schicht, Zufallsflips, Rekonstruktionsfehler und danach Genauigkeit; SECDED nur analytisch simuliert (1 Fehler je 64-Bit-Wort korrigiert, 2 erkannt = Nullsetzen). ~20 min, mehr Aufwand als Idee 1 wegen Dekoder.
- **Bytes-Fairness:** SECDED (72,64) = +12,5 %. Rahmen: gleiche +12,5 % an Zusatzzeilen. Dritter Gegner: +12,5 % Bytes als halbes Extra-Bit (gemischt 4/5 Bit je Schicht). Die Überschneidung verhindert einen Punktvergleich, daher Pareto.

## Idee 4: Flip-bewusste Indexzuordnung auf rotierten (gaußartigen) Koordinaten
- **Hypothese:** Nach der Hadamard-Rotation sind die Koordinaten näherungsweise gaußisch und gleich verteilt (PolarQuant [arXiv:2603.29078]: Hadamard allein ≈ 98 % des Gewinns). Damit ist EIN fester Lloyd-Max-Code mit einer flip-optimierten Indexzuordnung (Gray-/Pseudo-Gray-Zuordnung, die erwarteten Flip-Fehler minimiert) universell ohne pro-Gruppe-Skala. Er senkt den erwarteten Flip-MSE bei gleichen Bytes unter den Standardcode (Zweierkomplement/binär).
- **Nächste bekannte Arbeit:** [arXiv:2603.22770] (Lookup-Netze, Bitflip-Erwartungsfehler analytisch). Index-Assignment/Pseudo-Gray (Farvardin 1990) steht NICHT in `quellen.json`, aus dem Gedächtnis, vor Verwendung prüfen.
- **Stärkster Gegner:** Clipping [arXiv:2006.13977] und derselbe flip-bewusste Code ohne Rotation (Lloyd-Gitter ohne Rotation, S-001: 0,041 gegen Hadamard 0,052 bpc).
- **Quantenbezug:** schwache Analogie (Quelle-Kanal-Codierung); der Anker "Rotation verteilt gleichmäßig" gilt nur für die Gaußisierung.
- **Langweiligste Erklärung:** Die Indexzuordnung allein bringt den Gewinn, Rotation ist nebensächlich (Lloyd ohne Rotation schlägt in Z3 Hadamard). Dann ist das Ergebnis "Rotation unnötig", wie S-001.
- **Billigster Test:** Rein numerisch, ohne Training: 1000 Gaußproben, Code mit b = 3 und 4 Bit, erwarteter Flip-MSE für Binär/Gray/optimierte Zuordnung (Annealing über 8 bis 16 Etiketten). 2 min. Danach 1 MLP-Lauf wie Idee 1.
- **Bytes-Fairness:** Gleiche Codebits, kein Zusatz außer dem festen Codebuch (b-Bit, 2^b Einträge fp16, vernachlässigbar). Das Gegenüber "Skala + Standardcode" hat dieselben Bytes; die Skala fällt weg, sofern der Code universell ist (Kleiner Bytevorteil, ausweisen).

## Idee 5: Rotation zusätzlich zu RandBET: ergänzen oder ersetzen?
- **Hypothese:** Nach RandBET-Training ist die Zusatzwirkung der Rotation null (Rotation und Training lösen dasselbe Problem: Ausreißer). Ohne Training hilft Rotation, mit Training nicht. Alternativ verbleibt ein Rest bei hohem p (≥ 1e-2).
- **Nächste bekannte Arbeit:** [arXiv:2104.08323] (Fortsetzung von RandBET), [arXiv:2006.13977]; mit Rotation im Trainingspfad HALO-Stil (zitiert in `quellen.json` als HALO).
- **Stärkster Gegner:** RandBET + Clipping allein.
- **Quantenbezug:** Analogie, siehe Idee 1.
- **Langweiligste Erklärung:** Ergibt Gleichstand mit RandBET; das ist die Vorhersage des Gegnerwissens (S-007).
- **Billigster Test:** MLP mit RandBET-Training (Flips im Vorwärtspass bei p=1e-2), 2 Trainingslängen (z. B. 300 und 1500 Schritte, S-002) mit und ohne feste Hadamard-Schicht. ~25 min. Ausdrücklich zweite Stufe nach Idee 1, weil Training und mehr Gitter hinzukommen.
- **Bytes-Fairness:** Gleiche Bytes wie Idee 1 (nur das Training ändert sich). Beide Trainingslängen berichten (I-045-v4 und I-049: Frühregime täuscht).

## Empfehlung
1. **Idee 1 (inkl. Idee 2 als Zusatzauswertung)**: ein Skript, ein Vortest, beantwortet die Kernfrage von W5 ehrlich: Rotation gegen Clipping bei gleichen Bytes. Wichtige Neuerung ist, dass Skalen-Bits Teil der Flip-Fläche sind, wodurch g16 mit seinen Skalen (1 Bit/Gewicht) real benachteiligt sein kann. Das ist der wahrscheinlichste echte Befund (kein Rotationsvorteil, aber ein ehrlicher Bytevorteil gegenüber g16).
2. **Idee 4 als Zweitwahl**: Der numerische Vortest dauert 2 min, kann ein sauberes, quantitatives Ergebnis liefern (Flip-MSE nach Zuordnung) und prüft, ob die Gaußisierung durch Rotation einen strukturellen Grund hat. Ideen 3 und 5 erst, wenn Idee 1 einen Rest zeigt.
Erwartung vorab (keine Befunde): Gleichstand mit Clipping; echter Unterschied vor allem bei hohem p oder bei g16-Skalenflips.
