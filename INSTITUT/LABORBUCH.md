# Laborbuch des Instituts

Datiert, ehrlich, zurückgenommene Aussagen bleiben stehen.

---

## 24.09.2026 — Gründung, erste Linie, eine Idee auf Papier getötet

**Ausgangslage (aus Projekt A/B gelesen):** 16 Winkel liegen auf MNIST vor dem
2.048-Parameter-Quadratmodell; Verschränkung und Tiefe sind *nicht* belegt;
Struktur wird erst ab 16.384 Dimensionen schneller; die Kontrolle der alten
Kernaussage war zu schwach (Review 23.09.). Das Institut soll deshalb nicht
„noch einen Quantenbaustein“ bauen, sondern eine **Ressource** finden, die
wirklich an der Quantenstruktur hängt.

**Entscheidung des Direktors:** erste Linie = *Magie-Budget* (IDEEN Nr. 1).
Begründung: (a) es ist die Ressourcengröße, mit der die Quanteninformatik
selbst Kosten misst (T-Zahl); (b) es gibt eine harte, nachzählbare
Effizienzaussage (Multiplikationen → Additionen), die nicht an der
Faktor-64-Breitenfrage hängt; (c) ein Test dauert Minuten; (d) in der
Websuche nicht gefunden.

**Verifikation des Bausteins, vor jeder Zahl:**
- Schaltung als Matrix orthogonal: Abweichung 4,8·10⁻⁷.
- CNOT-Ring gegen Gatterdefinition: |100000⟩ → |011111⟩ (CNOT 0→1 … 4→5
  setzt alle Bits, CNOT 5→0 löscht Bit 0) — stimmt.
- Zufällige Clifford-Winkel: Matrixeinträge ∈ {−1, 0, 1}·c — die
  Schicht ist **tatsächlich multiplikationsfrei**.
- Vortest des Urteilsbaums: 3 Szenarien × 5 Ziehungen + fehlschlagende
  Auflösungsprüfung — bestanden.
- Laufzeit: Gatterform 0,1 s/Schritt → Matrixform 0,023 s/Schritt
  (mathematisch identisch; die Multiplikationszahl der *Inferenz* wird
  analytisch für die Gatterform gezählt).

**Vorregistrierung** (Code `abteilungen/clifford.py`, Folgefragen in
`agenda.json`, beides committet *bevor* das Ergebnis vorlag):
C-0 Schaltung > Identität; C-1 Rundung kostet ≤ 2 pp; C-2 gerundet > zufällige
Clifford **und** > feste Hadamard; C-3 M\* beschreibend. 10 Seeds (kleinster
exakter Wilcoxon-p 0,002), Lernrate je Arm auf Tuning-Seeds 900/901,
Gitter in beide Richtungen, 1.500 Schritte in Suche und Messung.

### Idee Z-1 auf Papier widerlegt (kein Rechenlauf nötig)

*Idee:* Zero-Noise-Extrapolation aus der Quanten-Fehlermitigation auf
Quantisierung übertragen — Logits mit b und b−1 Bit (verschachtelt: der
(b−1)-Code ist der gerundete halbe b-Code) auf „unendlich viele Bit“
extrapolieren. Neuheit laut Suche gegeben.

*Rechnung:* In einer linearen Schicht ist Extrapolation der Ausgaben
dasselbe wie Extrapolation der Gewichte. Sei c = round(W/s). Für gerade c ist
W_{b−1} = W_b, es passiert nichts. Für ungerade c ist W_b − W_{b−1} = ∓s, und
die lineare Extrapolation 2W_b − W_{b−1} = s(c ∓ 1) schiebt den Wert **auf
die andere Seite, weg vom Sollwert**. Mittlerer quadratischer Fehler:
s²/12 (gerade) bzw. s²/12 + s² (ungerade) → gesamt ≈ 7× schlechter als
b Bit. Quadratische Richardson-Variante: ≈ 1,7× schlechter.
*Nachgerechnet (200.000 Normalgewichte, b=4):* 7,21× bzw. 1,74×.

*Warum es scheitert:* ZNE setzt voraus, dass der Fehler eine **glatte
Funktion der Rauschstärke** ist (Erwartungswert über Zufall). Deterministisches
Runden ist das Gegenteil: sägezahnförmig. *Was übrig bleibt:* nur für
echtes, zufälliges Rauschen (analoge/photonische Rechenwerke) — dort ist aber
die Streuung eines Einzeldurchlaufs größer als der Bias, den ZNE entfernt.
Zurückgestellt als MPI-Frage.

*Lehre (→ FEHLERBUCH):* Eine Idee, die „aus der Quantenphysik kommt“, trägt
deren **Voraussetzungen** mit. Zuerst die Voraussetzung prüfen, dann bauen.
Hat 0 Rechenstunden gekostet.

## 24.09.2026, nachmittags — C-clifford v1: ein Artefakt, sauber gefangen

Lauf 15 min, 10 Seeds, Rohdaten `ergebnisse/C-clifford__20260924_160847.json`.

| Arm | Genauigkeit (Mittel ± SD) | Mult./Inferenz |
|---|---|---|
| voll (24 Winkel + Kopf) | 0,9664 ± 0,0067 | 3.776 |
| dicht_linear | 0,9584 ± 0,0092 | 640 |
| ohne_schaltung (Identität) | 0,9432 ± 0,0096 | – |
| hadamard_fest | 0,9429 ± 0,0098 | 704 |
| zufall_clifford | 0,9386 ± 0,0179 | 704 |
| voll_z (nur ⟨Z_q⟩) | 0,8566 ± 0,0209 | – |
| clifford_k0 (v1, geerbter Kopf) | 0,8696 ± 0,0419 | 704 |

- **C-0: H1** — die trainierte Schaltung schlägt die Identität (+2,3 pp,
  p = 0,002, 10/10 Seeds). *Erste belastbare Zahl des Instituts.*
- **C-1/C-2/C-3 (v1): nicht verwertbar** — siehe FEHLERBUCH. Die
  explorative Diagnose mit frischem Kopf gab k=0: 0,950; k=4: 0,962;
  k=8: 0,965 (Rate nach Testergebnis gewählt → nur Hinweis, kein Befund).
- **Winkel liegen nicht am Gitter:** mittlerer Abstand 0,390 rad — exakt
  der Wert für gleichverteilte Winkel (π/8 = 0,393). Das Training zieht die
  Winkel *nicht* von selbst zu Clifford-Punkten. → Begründet `C-bewusst`.
- **Ehrliche Effizienzeinordnung:** Auf 8×8-Ziffern ist ein rein linearer
  Klassifikator (640 Mult., 0,958) billiger als jede QIRE-Zelle. Das
  Magie-Budget misst die Kosten der *Mischschicht* (64×64 dicht: 4.096 Mult.;
  Clifford: 0 Mult.; volle Winkel: 3.072). Relevant wird das erst in der
  Mitte eines größeren Netzes — dieselbe Lage wie in Projekt A.
- **⟨Z_q⟩ allein (6 Zahlen) verliert 11 pp** gegenüber der vollen
  Born-Verteilung (64 Zahlen). Bestätigt den Engpass aus dem Review →
  stützt `B-paritaet`.

**Design v2 (vor jeder neuen Zahl festgelegt, committet):** Rundungsarme mit
frischem Kopf und eigener, auf Tuning-Seeds 902/903 gesuchter Rate
(an k=0); C-1 als Nichtunterlegenheit (einseitiges 95-%-KI ganz über −2 pp →
H1, ganz darunter → H0); frische Mess-Seeds 10–19. `C-mnist` war durch
C-2: H0 vorregistriert freigeschaltet und läuft ebenfalls mit v2 — die
Änderung liegt vor jeder MNIST-Zahl.

## 24.09.2026, 16:30 — C-clifford-v2 (frische Seeds 10–19, faires Design)

Rohdaten `ergebnisse/C-clifford-v2__20260924_162826.json`, 11,7 min.

| Arm | Genauigkeit | | Arm | Genauigkeit |
|---|---|---|---|---|
| voll | 0,9683 ± 0,0101 | | clifford_k0 | 0,9499 ± 0,0087 |
| dicht_linear | 0,9623 ± 0,0097 | | clifford_k1 | 0,9571 ± 0,0114 |
| ohne_schaltung | 0,9516 ± 0,0062 | | clifford_k2 | 0,9601 ± 0,0138 |
| hadamard_fest (entartet) | 0,9516 ± 0,0062 | | clifford_k4 | 0,9642 ± 0,0101 |
| zufall_clifford | 0,9495 ± 0,0218 | | clifford_k8 | 0,9660 ± 0,0075 |
| voll_z | 0,8410 ± 0,0211 | | clifford_k12 | 0,9686 ± 0,0054 |

**Vorregistrierte Urteile:** C-0 **H1** (repliziert: +1,7 pp, p = 0,002,
frische Seeds). C-1 **HU** (−1,84 pp, KI [−2,57; −1,11] überdeckt −2 pp).
C-2 **H0** (gerundet = zufällige Clifford = Identität). C-3 M\* = 0 — aber
nichtssagend, siehe FEHLERBUCH (Grenze > Effekt).

**Was das heißt:** Die Rundung auf das Clifford-Gitter vernichtet den
*ganzen* Gewinn der Schaltung — der Nutzen lebt vollständig in den
Nicht-Clifford-Winkeln, also in der „Magie“. Das ist die quantenphysikalisch
interessante Lesart: *Clifford allein bringt nichts, genau wie Clifford-
Schaltungen allein keinen Quantenvorteil bringen (Gottesman–Knill).*

**Nachträglich, beschreibend (nicht vorregistriert):** Gewinnanteil
G(k) = (acc_k − acc_Identität)/(acc_voll − acc_Identität):

| k (freie Winkel von 24) | 0 | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|---|
| G(k) | −0,10 | 0,33 | **0,51** | 0,76 | 0,87 | 1,02 |

**Die Hälfte des Nutzens steckt in 2 von 24 Winkeln, drei Viertel in 4.**
Die Magie ist *konzentriert*. Das ist die Hypothese, die jetzt zählt; G(k),
M50 und M90 sind für MNIST vorregistriert (Code-Änderung vor der ersten
MNIST-Zahl). Wenn sich das auf MNIST und Fashion hält, ist das die Größe,
die einen Namen verdient: **die Magie-Konzentration einer Aufgabe.**

**Verfolgt:** `C-bewusst` ist durch C-1: HU vorregistriert freigeschaltet
und als Modul gebaut (Vortest bestanden — erst nach Korrektur eines Fehlers
im eigenen synthetischen Grenzfall).

## 24.09.2026, 17:10 — V-kompass (MNIST 16×16, 8 Seeds, 34 min)

Relativer Fehler / Genauigkeit des Netzes mit komprimiertem ΔW der ersten
Schicht (dicht: 0,9562):

| Budget | MI-Fiedler | Zufall (Ø 8) | natürlich | verzahnt | Morton | **SVD** |
|---|---|---|---|---|---|---|
| 512 | 0,978 / 0,49 | 0,988 | 0,983 / 0,54 | 0,988 / 0,50 | 0,984 / 0,47 | **0,930** / 0,45 |
| 2.048 | 0,916 / 0,50 | 0,961 | 0,916 / 0,47 | 0,961 / 0,38 | 0,918 / 0,47 | **0,777** / 0,53 |
| 4.096 | 0,831 / 0,62 | 0,927 | 0,831 / 0,55 | 0,926 / 0,38 | 0,832 / 0,54 | **0,644** / 0,73 |

Urteile: V-1 **HU** (MI/Zufall = 0,954, p = 0,008, aber < 10 %). V-2 **HU**
(MI = natürlich, Δ = 0,0004, p = 0,84). Die MI-Ordnung legt in allen Seeds
die 8 Ausgangsbits und die 8 Eingangsbits je in eine Hälfte — sie
*entdeckt* die natürliche Ordnung, mehr nicht.

**Einordnung:** Negativ für die Tensornetz-Kompression gelernter Filter,
positiv für die Methode des Instituts: Das Rote Team (SVD) hat den Befund
gefunden, den keine der „quanteninspirierten“ Varianten zeigt. Linie
V-kompass ruht; Folgeidee „Rang außen, Tensornetz innen“ in FEHLERBUCH.

## 24.09.2026, 18:00 — Neue Abteilung: Ideenschmiede (auf Maltes Wunsch)

Aufbau siehe README. Erste Runde: fünf Karten aus dem Befund
„Magie-Konzentration“ abgeleitet, Neuheit von einem Recherche-Agenten geprüft
(11 Websuchen, nur Trefferlisten — „nicht gefunden“ heißt nicht „existiert
nicht“). Alle fünf durchs Tor an Abt. Clifford:

| Karte | Idee | Neuheit |
|---|---|---|
| I-001 | Magie-Pruning (Winkel nach Verlustanstieg statt Gitterabstand) | teilweise |
| I-002 | Stabilisator-Rényi-Entropie sagt M50 vorher, ohne Rundungsversuche | nicht gefunden |
| I-003 | Magie-Konzentration als Kennzahl einer *Aufgabe* (Kontrolle: Zufallsetiketten) | teilweise |
| I-004 | Clifford-Gerüst + k Winkel von Anfang an | teilweise (CAFQA, VQE) |
| I-005 | nur diskrete Clifford-Wahl, Magie allein aus den Daten | teilweise (Stabilizer Bootstrap) |

Reihenfolge-Empfehlung des Direktors: **I-002 zuerst** (kein Training,
Minuten, rechnet auf gespeicherten Winkeln), dann I-001, dann I-003 mit
Zufallsetiketten. I-005 hat nach C-2 (feste Clifford = Identität) die
schlechteste Papierprognose — bleibt trotzdem stehen, weil ein klares Nein
dort die C-2-Aussage verallgemeinern würde.

## 24.09.2026, 19:15 — C-mnist: die Magie-Konzentration hält (vorregistriert)

MNIST 16×16 (8 Qubits, 256 Amplituden, 4 Lagen = 32 Winkel), 6.000
Trainings-/5.000 Testbilder je Seed, 10 Seeds, Design v3 (Kontrolle
Hadamard korrigiert, G(k)/M50/M90 vor der ersten MNIST-Zahl im Code).
Rohdaten `ergebnisse/C-mnist__20260924_181325.json`, 71 min.

| Arm | Genauigkeit | Mult./Inferenz |
|---|---|---|
| voll (32 Winkel) | **0,9385 ± 0,0035** | 19.200 |
| dicht_linear | 0,9118 ± 0,0038 | 2.560 |
| ohne_schaltung (Identität) | 0,9006 ± 0,0037 | – |
| hadamard_fest (jetzt echt) | 0,8831 ± 0,0039 | 2.816 |
| zufall_clifford | 0,9054 ± 0,0100 | 2.816 |
| voll_z (nur 8 ⟨Z⟩) | 0,7637 ± 0,0109 | – |
| clifford_k0 | 0,9069 ± 0,0160 | 2.816 |
| clifford_k1 | 0,9192 ± 0,0092 | 3.328 |
| clifford_k2 | 0,9208 ± 0,0079 | 3.840 |
| clifford_k4 | 0,9264 ± 0,0072 | 4.864 |
| clifford_k8 | 0,9342 ± 0,0069 | 6.912 |
| clifford_k12 | 0,9353 ± 0,0054 | 8.960 |

**Urteile:** C-0 **H1** (+3,8 pp über der Identität, p = 0,002, 10/10).
C-1 **H0** (Rundung kostet 3,2 pp, KI [−3,95; −2,36] — Magie wird gebraucht).
C-2 **HU** (gerundet > echte Hadamard-Mischung, p = 0,010; ≈ zufällige
Clifford). C-3: **G(1) = 0,49, G(2) = 0,53, G(4) = 0,68, G(8) = 0,89,
G(12) = 0,92 → M50 = 2 (bereits k = 1 fast 0,5), M90 = 12 von 32.**

**Was das heißt:**
1. **Magie-Konzentration repliziert, jetzt vorregistriert:** Auf den Ziffern
   (explorativ) trugen 2 von 24 Winkeln die Hälfte des Nutzens, auf MNIST
   (vorregistriert) trägt **ein einziger von 32** fast die Hälfte. Die
   Kurve ist steil am Anfang und flach danach — der Nutzen der
   Quantenschaltung sitzt in sehr wenigen magischen Gattern.
2. **Clifford allein trägt nichts** — auch auf MNIST ist die gerundete
   Schaltung nicht besser als eine zufällige Clifford-Schaltung.
   Quantenphysikalisch sauber: Clifford = klassisch simulierbar
   (Gottesman–Knill) = kein Gewinn; der Gewinn kommt aus der Magie.
3. **Auf MNIST schlägt die Schaltung den linearen Klassifikator** (+2,7 pp,
   beschreibend, nicht vorregistriert) — auf den Ziffern war es nur +0,6 pp.
   Mit k = 2 Magiewinkeln: 0,921 bei 3.840 Mult. gegen linear 0,912 bei 2.560.
4. **Die Auslese bleibt der Engpass:** nur 8 ⟨Z⟩-Werte verlieren 17 pp.

**Freigeschaltet (vorregistriert):** C-fashion (C-1: H0). C-bewusst ist
schon offen. Beide mit Priorität 1 für Maltes PC (`institut.cmd alle`).

**Arbeitstitel für die Größe, falls Fashion es bestätigt:** *Magie-
Konzentration* M50/M90 einer Aufgabe — „wie viele magische Gatter braucht
eine Lernaufgabe für die Hälfte / 90 % ihres Quantennutzens?“

## 24.09.2026, 18:40 — Skalierung: 3 Rechenstränge + 3 Forscher-Agenten parallel

Auf Maltes Wunsch vergrößert. Parallel: C-fashion (2 Kerne), C-bewusst
(2 Kerne), I-008 (1 Kern); drei Agenten bauen I-002 (SRE-Vorhersage),
I-001 (Magie-Pruning) und C-hardware (Qiskit, Clifford+T). Regel für die
Agenten: nur neue Dateien, keine gemeinsamen Dateien anfassen, kein Commit —
der Direktor integriert. Ideenschmiede Runde 2: I-008 (Q3) und I-010 (Q2)
weitergeleitet, I-006/007/009/011 mit Begründung verworfen (I-009
nachgerechnet: 10× teurer; I-011: Quantenlauf auf dem Hyperwürfel *ist*
die QIRE-Lage).

### I-008 Schuss-Budget (Q3), Ziffern 8 Seeds + MNIST 3 Seeds beschreibend, 8 min

Rohdaten `ergebnisse/I-008__20260924_183851.json`. Genauigkeit bei S Messungen je Bild:

| Auslese | S=10 | 100 | 1.000 | 10.000 | exakt |
|---|---|---|---|---|---|
| born (64) | 0,444 | 0,877 | 0,959 | 0,967 | 0,967 |
| paritaet (22) | 0,328 | 0,704 | 0,884 | 0,906 | 0,907 |
| z (6) | 0,215 | 0,456 | 0,625 | 0,645 | 0,650 |
| MNIST born (256) | 0,327 | 0,769 | 0,924 | 0,935 | 0,938 |

A-1 **HU** (−0,81 pp bei S=1.000, KI [−1,14; −0,48]). A-2 **H0** (born
schlägt paritaet bei S=100 um 17 pp, p = 0,008) — Hypothese falsch, siehe
FEHLERBUCH. **Hardware-Einordnung:** Die Schaltung braucht auf echter
Hardware rund **1.000 Schüsse je Bild** (MNIST: 1.000 → −1,4 pp,
10.000 → −0,3 pp). Das ist die Zahl, die in jede ehrliche Aussage über
„Quantenhardware als KI-Beschleuniger“ gehört — und eine gute Frage fürs
MPI-Gespräch.

## 24.09.2026, 18:45 — C-hardware (Q3): Magie-Konzentration in T-Gattern

Forscher-Agent, Modul `abteilungen/hardware.py`, Rohdaten `ergebnisse/C-hardware__20260924_184246.json`.
Die QIRE-Zelle als Qiskit-Schaltung (Bitordnung: Modell-Qubit q =
Qiskit-Qubit n−1−q), Operatorfehler 5·10⁻¹⁶ (float64); vom Direktor mit
eigenen Zufallswinkeln nachgeprüft: 2,5·10⁻⁷ (float32-Modell). Negativ-
kontrolle ohne Umnummerierung: 0,4–0,7 (der Test *kann* die Bitordnung
erkennen). Clifford-Winkel exakt als h/x/z, magische Ry-Winkel mit
Ross–Selinger-gridsynth bzw. Solovay–Kitaev nach Clifford+T übersetzt.

| k magische Winkel (MNIST, 32 CNOT) | T-Gatter ε=10⁻² | ε=10⁻³ | ε=10⁻⁴ | Gewinnanteil G(k) |
|---|---|---|---|---|
| 0 | **0** | **0** | **0** | 0,17 |
| 1 | 11 | 30 | 42 | 0,49 |
| 2 | 28 | 62 | 84 | 0,53 |
| 4 | 69 | 123 | 168 | 0,68 |
| 8 | 152 | 247 | 337 | 0,89 |
| 12 | 236 | 371 | 503 | 0,92 |

H-1 **bestätigt** (k=0 → 0 T-Gatter, exakter Operator, alle Seeds).
T ≈ 30,9·k bei ε=10⁻³ (R² = 1,000). Born-Verteilung der Clifford+T-Schaltung
weicht bei ε=10⁻³ höchstens 1,3·10⁻⁵ vom Modell ab.

**Bedeutung:** Damit ist das Magie-Budget in die Kostenwährung eines
fehlertoleranten Quantencomputers übersetzt: *die Hälfte des Quantennutzens
für ~30–60 T-Gatter, 90 % für ~370.* Die volle Schaltung (32 magische
Winkel) bräuchte ~990 T-Gatter. Grenzen: nur rauschfrei simuliert; die Kosten
der Amplitudenkodierung (Zustandspräparation, selbst sehr T-teuer) sind nicht
mitgezählt — das ist die nächste ehrliche Frage.

## 24.09.2026, 19:10 — C-bewusst: Magie lässt sich nicht wegtrainieren

Ziffern, frische Seeds 20–29, λ auf Tuning-Seeds gesucht (0,3, innen),
Rohdaten `ergebnisse/C-bewusst__20260924_185324.json`, 27 min.

| | normal | Clifford-bewusst |
|---|---|---|
| volles Modell | 0,9609 ± 0,0133 | 0,9473 ± 0,0200 |
| Abstand zum Gitter (rad) | 0,396 | 0,229 |
| gerundet k=0 | 0,9477 | 0,9438 |
| k=2 | 0,9536 | 0,9514 |
| k=4 | 0,9623 | 0,9590 |

B-1 **H0** (−0,4 pp, p = 1,0). B-2 **HU**. Deutung im FEHLERBUCH: Der
Strafterm zieht *alle* Winkel in den Clifford-Bereich, der nachweislich
nichts bringt (C-2). Das Institut hat damit drei unabhängige Belege für
denselben Satz: **Der Nutzen der Schaltung sitzt in der Magie, und zwar in
wenigen Winkeln.** (C-2 auf zwei Datensätzen, C-3/G(k), jetzt C-bewusst.)

## 24.09.2026, 19:30 — C-kodierung (Q3): das Laden frisst die Einsparung

Forscher-Agent, Modul `abteilungen/kodierung.py`, Rohdaten `ergebnisse/C-kodierung__20260924_192627.json`,
MNIST 16×16, 1 Seed (Seed 10 exakt reproduziert), 20 Bilder, ε=10⁻³ je Rotation.

| Kodierung | T-Gatter | CNOT | Treue | Gen. direkt | Gen. Kopf neu |
|---|---|---|---|---|---|
| exakt (Möttönen, nur Ry) | 7.636 | 250 | 1,000 | 93,2 % | 93,3 % |
| MPS χ=4 | 6.329 (Untergrenze ~3.250) | 107 | 0,950 | 92,6 % | 93,2 % |
| MPS χ=2 | 1.176 | 38 | 0,736 | 70,8 % | 87,6 % |
| MPS χ=1 (Produkt) | 225 | 0 | 0,321 | 31,8 % | 64,4 % |
| Winkel auf π/2 (Clifford) | 0 | 250 | 0,102 | 13,3 % | 50,7 % |

K-1 **bestätigt**: Laden 7.636 T gegen 989 T für die ganze Schaltung (k=32)
und 62 T für den Halbnutzen-Punkt. K-2 **nein**: keine Kodierung unter 10 %
der T-Kosten ohne großen Verlust; χ=2 kostet 15 % und verliert 5,7 pp.

**Einordnung, ehrlich:** Das ist der aus der Quanten-KI bekannte
Datenlade-Flaschenhals (Aaronson, „Read the fine print“, Nature Physics 2015),
jetzt in T-Gattern für QIREs eigene Schaltung beziffert. Auf echter
fehlertoleranter Hardware dominiert die Kodierung die Kosten; die
Magie-Konzentration der *Schaltung* spart dort nur ~13 % der Gesamtkosten.
Im Simulator (Multiplikationen) gilt die Aussage weiter.

**Nicht aufgeben, vertiefen:** Die Kodierung ist selbst der größte
Magieverbraucher — also muss das Magie-Budget über **Kodierung + Schaltung
gemeinsam** gerechnet und optimiert werden (neue Karte I-020). Noch nicht
gemessen: dünn besetzte Präparation (71 % der Pixel sind null) und eine
Zelle, die direkt auf der billigen Kodierung trainiert wird.

## 24.09.2026, 19:45 — I-001 Magie-Pruning und I-002 Stabilisator-Entropie

**I-001** (Forscher-Agent, Ziffern, Seeds 30–39, eigene Rate je Arm):
G(k) für k = 0 / 1 / 2 / 4 —

| Rangfolge | k=1 | k=2 | k=4 |
|---|---|---|---|
| Abstand zum Gitter (bisher) | −0,04 | 0,60 | 0,69 |
| Verlustanstieg (neu) | 0,15 | **0,67** | 0,79 |
| gierig | 0,20 | 0,54 | **0,88** |
| Zufall | 0,15 | 0,38 | 0,80 |

P-0 H1 (Schaltung > Identität, p = 0,002). P-1 **HU** (+0,07 G gegenüber
Abstand, p = 0,375 — Auflösung auf den Ziffern zu klein). P-2 **H1**
(Wirkung schlägt Zufall bei k=2: +0,29 G, p = 0,027). → *Welche* Winkel
magisch bleiben, ist nicht egal; die Wahl nach Wirkung ist messbar besser als
Zufall. Wiederholung auf MNIST vorgemerkt.

**I-002** (Forscher-Agent, exakte Operator-SRE über alle 4²ⁿ Pauli-Strings,
Vortest gegen Brute Force n=1–3, π/4-Winkel → log₂(4/3)): S-1 **H0**
(ρ = 0,26, p = 0,26; innerhalb der Datensätze p = 0,58), S-2 HU.
**Einsicht:** gelernte Schaltungen tragen *so viel* Magie wie zufällige
(Ziffern 4,38 vs. 4,58; MNIST 6,26 vs. 6,28) — nutzen davon aber nur wenige
Winkel. **Vorhandene Magie ≠ genutzte Magie.** Ein Maß am Zustand oder
Operator allein sieht die Aufgabe nicht; G(k) tut es.

## 24.09.2026, 21:45 — C-fashion: Konzentration repliziert, Designfehler gefunden

Fashion-MNIST 16×16, 8 Qubits, 32 Winkel, Seeds 10–19, Design v3 wie C-mnist
(Rohdaten `ergebnisse/C-fashion__20260924_214236.json`, 68 min, 4 Kerne).

| Arm | Fashion | MNIST (C-mnist) |
|---|---|---|
| Identität (ohne Schaltung) | 0,796 | 0,901 |
| zufällige Clifford | 0,816 | 0,905 |
| k = 0 (gerundet) | 0,817 | 0,907 |
| k = 2 | 0,827 | 0,921 |
| k = 4 | 0,833 | 0,926 |
| k = 8 | **0,838** | 0,934 |
| voll | 0,833 | 0,939 |

- C-0 **H1**: Schaltung schlägt Identität, +3,6 pp, p = 0,002 (MNIST +3,8).
- C-1 HU (knapp: KI der Rundung [−2,05; −1,02] pp streift die 2-pp-Grenze).
- C-2: k0 = zufällige Clifford (+0,17 pp, p = 0,77) — wie auf beiden
  anderen Datensätzen. Anders als auf MNIST trägt auf Fashion aber schon
  *irgendeine* Clifford-Schaltung 2,1 pp (p = 0,002) über die Identität:
  G(0) = 0,58 statt 0,17. Das ist Clifford-Durchmischung, keine Magie.
- **Magie-Anteil** (Basis k0, Nachrechnung Direktor): k=1 0,24 · k=2 0,64 ·
  **k=4 1,01** · k=8 1,34. MNIST: 0,39 · 0,44 · 0,62 · 0,86. Die
  Konzentration ist auf Fashion *stärker*: M50 = 2, M90 = 4 von 32 Winkeln
  (vorregistriertes Maß mit Basis Identität: M50 = 0, M90 = 4).

**Warnsignal:** k=8 und k=12 *schlagen* den vollen Arm (+0,53 pp, p = 0,010;
+0,46 pp, p = 0,025). Ursache gefunden: `_gerundet` startet beim fertig
trainierten Modell und trainiert **noch einmal 1.500 Schritte** (Rate 0,32,
frischer Kopf) — der volle Arm bekommt diese zweite Phase nicht. Ob k8 > voll
Regularisierung durch Runden ist oder bloß mehr Training, ist so nicht
trennbar, und alle G(k) gegen „voll“ können aufgebläht sein (FEHLERBUCH).
**Konsequenz, vor jeder weiteren Zahl:** C-referenz (Arm *nachtrainiert* =
alle Winkel frei, identische zweite Phase; MNIST + Fashion, frische Seeds
30–39) vorregistriert; C-skala und I-024 bekommen denselben Arm als
Referenz. Der Name „Magie-Konzentration“ wartet auf C-referenz.

## 24.09.2026, 22:10 — I-003: Aufgabe oder Architektur? Positivkontrolle scheitert am Lernverfahren

Ziffern 8×8 (6 Qubits, 24 Winkel), Seeds 60–69, Forscher-Agent-Modul
`magie_aufgabe.py` (Rohdaten `ergebnisse/I-003__*.json`, 22 min).

| Aufgabe (Messgröße) | Identität | k0 | k1 | k2 | k4 | k8 | k12 | voll |
|---|---|---|---|---|---|---|---|---|
| echt (Test) | 0,945 | 0,941 | 0,953 | 0,957 | 0,962 | 0,958 | 0,963 | 0,959 |
| zufall (Training, Memorieren) | 0,240 | 0,252 | 0,250 | 0,254 | 0,251 | 0,278 | 0,285 | 0,272 |
| struktur (Test, Clifford-Lehrer) | 0,784 | 0,735 | 0,737 | 0,751 | 0,766 | 0,773 | **0,786** | **0,726** |

- T-1 **HU (Vorbedingung)**: Auf der Strukturaufgabe — per Konstruktion mit
  Magie 0 exakt lösbar, Kopf auf den Lehrer-Merkmalen 0,934 — landet die
  voll trainierte Zelle *unter* der Identität (−5,7 pp, p = 0,010).
  Gradientenabstieg aus Zufallswinkeln findet die diskrete Clifford-Lösung
  nicht. T-2 damit nicht auswertbar (vorregistriert).
- Zweiter Beleg für den Nachtrain-Fehler (C-fashion): k12 schlägt voll auf
  der Strukturaufgabe um **+6 pp**, allein durch die zweite Trainingsphase.
- Explorativ (T-3, beschreibend): echt M50 = 1, M90 = 4; zufall (Memorieren)
  M50 = 8. Die G-Kurve von echt liegt über zufall (+0,51 G, p = 0,002).
  Hinweis, dass Generalisieren weniger Magie braucht als Auswendiglernen —
  **nicht belastbar**, bis Referenz (C-referenz) und Positivkontrolle
  (I-003-v2) stehen.

Lehre: *exakt lösbar* ≠ *lernbar*. Eine Positivkontrolle muss mit dem
Verfahren feuern können, nicht nur mit der Architektur. I-005 (diskrete
Clifford-Suche) bekommt dadurch Gewicht; I-003-v2 als Idee eingetragen.

## 24.09.2026, 23:05 — I-020 Gesamtbudget (Q3): Laden halbiert, nicht geviertelt

MNIST 16×16, 3 Seeds, beschreibend (vorregistriert, kein Signifikanztest);
Rohdaten `ergebnisse/I-020__20260924_230045.json`, 56 min (Modul Forscher-Agent).
T-Zahl gesamt = Laden (Qiskit-Synthese, exakt) + Schaltung (≈ 31·k).

| Kodierung | k | T gesamt | Genauigkeit | Anteil Laden |
|---|---|---|---|---|
| dicht exakt (Referenz) | 32 | 8.776 | 0,940 | — |
| **dünn besetzt exakt** | 32 | **4.372** | **0,940** | 77 % |
| dünn besetzt exakt | 4 | 3.506 | 0,933 | 96 % |
| dünn besetzt exakt | 0 | 3.382 | 0,911 | 100 % |
| MPS χ=2 | 32 | 2.178 | 0,881 | 55 % |
| MPS χ=2 | 4 | 1.312 | 0,857 | 91 % |
| dünn, π/4-gerundet | 32 | 1.001 | 0,731 | 1 % |

- G-1 **H0** (Abbruchkriterium der Karte): keine Kombination ≤ 2.194 T (25 %)
  hält 2 pp an der Referenz. Aber: **dünn besetzte Präparation halbiert die
  T-Zahl ohne Verlust**, mit k=4 auf 40 % bei −0,7 pp.
- Kodierungs-Verluste (MPS, π/4) sind nicht durch Schaltungs-Magie
  aufzuholen: Die Zelle auf billiger Kodierung neu zu trainieren rettet 6–20
  pp nicht.
- Kernsatz wird schärfer: *Mit Magie-Konzentration (k=4) ist die Schaltung
  4 % der Rechnung; 96 % sind Datenladen.* Die nächste Front ist eindeutig
  die Kodierung selbst.
- Vorbehalt: Die Magie-Arme nutzen `_gerundet` (Nachtrain-Fehler, C-referenz).

## 24.09.2026, 23:50 — C-referenz: Verdacht widerlegt, Konzentration repliziert auf frischen Seeds

MNIST + Fashion 16×16, frische Seeds 30–39, Raten aus C-mnist/C-fashion,
Arm *nachtrainiert* = alle 32 Winkel frei mit derselben zweiten Phase wie die
Rundungsarme (Rohdaten `ergebnisse/C-referenz__20260924_234210.json`, 80 min).

| | Identität | k0 | k1 | k2 | k4 | k8 | nachtrainiert | voll |
|---|---|---|---|---|---|---|---|---|
| MNIST | 0,900 | 0,910 | 0,916 | 0,918 | 0,926 | 0,931 | 0,938 | 0,938 |
| Fashion | 0,798 | 0,816 | 0,822 | 0,827 | 0,831 | 0,834 | 0,804* | 0,832 |

\* 4 von 10 Seeds kollabieren (0,725–0,812) bei Rate 0,32.

- R-1 (vorregistriert) **HU** gesamt; **MNIST: keine Verzerrung** (d = −0,03 pp,
  KI [−0,24; +0,18] ganz in ±0,3 pp). Fashion: Referenz instabil, auf den 6
  stabilen Seeds d = +0,03 pp.
- R-2 **HU (Vorbedingung)** wegen der kaputten Fashion-Referenz. MNIST allein:
  Magieanteil Gm*(4) = 0,59, KI [0,30; 0,77], Magiegewinn +2,7 pp (p = 0,002).
- R-3 formal „H1“, aber **Artefakt** der instabilen Referenz — verworfen.
  k8 − voll auf Fashion: +0,24 pp, p = 0,08 (repliziert +0,53 pp nicht).
- **Replikation auf frischen Seeds** (G mit Basis Identität, gegen voll):

| k von 32 | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| MNIST C-mnist (Seeds 10–19) | 0,17 | 0,49 | 0,53 | 0,68 | 0,89 |
| MNIST C-referenz (Seeds 30–39) | 0,27 | 0,43 | 0,49 | 0,70 | 0,83 |
| Fashion C-fashion (10–19) | 0,58 | 0,68 | 0,85 | 1,01 | 1,14 |
| Fashion C-referenz (30–39) | 0,52 | 0,73 | 0,85 | 0,99 | 1,07 |

**Ergebnis:** Die Magie-Konzentration ist auf zwei Datensätzen und je zwei
unabhängigen Seed-Sätzen stabil: **4 von 32 Winkeln tragen 70 % (MNIST) bzw.
~100 % (Fashion) des Schaltungsgewinns**. M50 = 2–4 (MNIST liegt bei k=2
genau an der Grenze), M90 ≈ 8–12 bzw. 4. Der Störfaktor war nicht das
Nachtraining, sondern eine für wenige Parameter getunte Lernrate (FEHLERBUCH).

## 24.09.2026, 23:55 — I-015 Magie-Kaskade (Q3): spart Schaltungs-T, das Laden frisst es — außer die Clifford-Stufe läuft klassisch

MNIST 16×16, Seeds 90–94 (5, beschreibend für K-2), Kaskade 0 → 2 → 8,
Schwelle τ auf Validierung (Rohdaten `ergebnisse/I-015__20260924_234941.json`, 7 min).

- K-1 **H1**: Kaskade erreicht die Testgenauigkeit von fest k=2 (+0,05 pp,
  KI [−0,45; +0,55]) mit **17 T = 28 %** der Schaltungs-T von k=2.
- K-2 B+: Konfidenz schlägt Zufallsweiterleitung in 5/5 Seeds (+1,2 pp;
  p = 0,0625 = kleinstmöglich bei 5 Seeds).
- K-3: **mit Ladekosten 114 %** von k=2 — 10–20 % der Bilder werden neu
  geladen (7.636 T je Lauf). Papierschranke der Karte bestätigt (lohnt erst
  unter 0,8 % Weiterleitung).
- **Nachträgliche Rechnung (Direktor, nicht vorregistriert):** Die Stufe k=0
  ist eine Clifford-Schaltung auf einem *klassischen* Vektor — klassisch
  exakt mit ≤ 8.192 Additionen, ohne Multiplikation und ohne Laden. Läuft sie
  klassisch und gehen nur die unsicheren ~10 % der Bilder auf den
  Quantenrechner, kostet die Kaskade **1.118 T = 14,5 % der Hardware-T von
  k=2** bei gleicher Genauigkeit (92,28 % gegen 92,24 %). 0 → 8 direkt: 9,4 %.
  → Als **I-015-hybrid** mit 10 frischen Seeds vorregistriert (H-1: ≤ 25 %
  und Nichtunterlegenheit 0,5 pp).

## 25.09.2026, 00:00 — I-004 Clifford-Gerüst von Anfang an (Ziffern): Auflösung reicht nicht

Ziffern 8×8, Seeds 80–89 (Rohdaten `ergebnisse/I-004__20260924_235757.json`, 8 min).
R-0 H1 (Gewinn +2,1 pp, p = 0,004); R-1 HU (zufälliges Gerüst + 4 Winkel:
G = 0,71, KI 0,47–0,96); R-2 HU (+0,28 G für nachträgliches Runden, Holm-p =
0,15); R-3 HU (Gerüst mit neu initialisierten Magiewinkeln vs. Nachtraining,
−0,06 G, KI −0,23…+0,10).

| G(k) | k=1 | k=2 | k=4 | Trainings |
|---|---|---|---|---|
| nachträglich runden | 0,41 | 0,71 | 0,77 | 2 |
| Gerüst aus vollem Modell, Winkel neu | 0,26 | 0,65 | 0,83 | 2 |
| **zufälliges Gerüst**, Winkel von Anfang an | 0,23 | 0,43 | 0,71 | **1** |

Beschreibend: Wenig-magische Schaltungen brauchen das volle Modell nicht —
zufälliges Clifford-Gerüst + 4 trainierte Winkel holt ~70 % des Gewinns mit
halbem Trainingsaufwand. Belastbar erst auf MNIST (I-004-mnist, Idee).

## 25.09.2026, 00:10 — I-015-hybrid (Q3, vorregistriert): **Hybrid-Magie-Kaskade spart 88 % der Hardware-T** ✔

MNIST 16×16, 10 frische Seeds 100–109, Kaskade 0 → 2 → 8, Stufe 0 (Clifford,
0 T) **klassisch** (≤ 8.192 Additionen, 0 Multiplikationen, kein Laden), jede
Quantenstufe = Laden 7.636 T + Schaltung; τ auf Validierung nach den
Hybrid-Kosten (Rohdaten `ergebnisse/I-015-hybrid__20260925_000945.json`, 12 min).

- **H-1 H1**: mittlere Hardware-T je Bild **902 = 11,7 % von k=2 auf
  Hardware** (7.698 T); Genauigkeit 91,74 % gegen 91,68 % (+0,06 pp,
  einseitiges KI [−0,20; +0,31], Grenze −0,5 pp). Nur **7,9 %** der Bilder
  erreichen den Quantenrechner.
- **H-2 H1**: Konfidenz schlägt Zufallsweiterleitung gleicher Quote
  (+1,06 pp, p = 0,031).
- H-3: zweistufig 0 → 2: 9,5 %; 0 → 8: 8,0 % (beide ≈ k2-Genauigkeit).
- **Robustheit (Direktor):** Auf 4/10 Seeds wählt die Validierung τ* = 0
  (die klassische Clifford-Stufe allein genügt dort; Test −0,2 … −0,7 pp).
  Ohne diese Seeds: 1.503 T = **19,5 %** — immer noch unter 25 %.
- Replikation des Post-hoc-Werts (I-015, Seeds 90–94: 14,5 %) auf frischen Seeds.

**Einordnung:** Das ist der erste Befund des Instituts, der eine echte
Hardware-Ersparnis *mit* Ladekosten zeigt. Er verbindet die beiden
Hauptbefunde: Magie-Konzentration (wenige Winkel reichen) und
Clifford = klassisch billig (die magiefreie Stufe braucht keinen
Quantenrechner). Ehrlich: Es ist ein **Hybrid**-Befund — die Ersparnis
entsteht, weil ein klassisches multiplikationsfreies Netz die meisten Bilder
selbst entscheidet und nur schwierige an die Quantenschaltung abgibt.

## 25.09.2026, 01:50 — I-024 Magie gegen Datenmenge: Gegenrichtung — Konzentration schärft sich mit Daten

MNIST 16×16, n_train ∈ {500, 2.000, 10.000} (geschachtelt), Seeds 100–109,
faire Referenz *nachtrainiert*, Rate je Arm und n eigen (Rohdaten
`ergebnisse/I-024__20260925_014438.json`, 95 min).

| n_train | Identität | k0 | k1 | k2 | k4 | k8 | nachtrainiert | voll | G(2) | G(8) |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 0,835 | 0,831 | 0,842 | 0,847 | 0,855 | 0,867 | 0,873 | 0,872 | 0,34 | 0,86 |
| 2.000 | 0,883 | 0,882 | 0,892 | 0,898 | 0,898 | 0,910 | 0,919 | 0,917 | 0,43 | 0,75 |
| 10.000 | 0,908 | 0,914 | 0,921 | 0,929 | 0,932 | 0,941 | 0,943 | 0,942 | **0,61** | 0,96 |

- D-0 **H1** (+3,5 pp bei 10.000, p = 0,002).
- D-1 **H0 (Gegenrichtung)**: Der Magie-Abstand (nachtrainiert − k0) ist bei
  wenig Daten *größer* (4,18 gegen 2,91 pp, p = 0,019).
- D-2 **H0 (Gegenrichtung)**: 2 Magiewinkel tragen bei 500 Bildern *weniger*
  (G(2) −0,28, p = 0,042). Beide Gegenrichtungs-Tests einseitig ohne Holm:
  deutlicher Hinweis, kein Beweis.
- d_nach ≈ 0 bei allen n → dritter Beleg gegen den Nachtrain-Verdacht.

**Einordnung:** Der Gewinn der Schaltung ist von der Datenmenge fast
unabhängig (~3,5 pp), aber seine *Verteilung* nicht: Mit mehr Daten
konzentriert das Training die genutzte Magie auf weniger Winkel. Die
Magie-Konzentration ist also eine Eigenschaft des **gelernten** Modells, die
mit der Datenmenge wächst — nicht der Architektur allein (passt zum
explorativen I-003-Hinweis). Die Regularisierungsidee war falsch gestellt
(Runden danach ≠ Training im Clifford-Raum) → I-004-mnist mit n = 500.

## 25.09.2026, 02:00 — I-015-hybrid-fashion: Hybrid-Kaskade repliziert auf Fashion ✔

Fashion-MNIST 16×16, 10 frische Seeds 110–119, sonst identisch vorregistriert
(Rohdaten `ergebnisse/I-015-hybrid-fashion__20260925_015647.json`, 12 min).

- **H-1 H1**: 1.837 T = **23,9 % der Hardware-T von k=2**, Genauigkeit
  81,98 % gegen 81,67 % (**+0,31 pp**, KI [+0,04; +0,57] — sogar besser als
  k2 allein, weil unsichere Bilder bis k=8 durchgereicht werden). 14,2 % der
  Bilder gehen auf den Quantenrechner.
- **H-2 H1**: Konfidenz schlägt Zufall (+0,82 pp, p = 0,004).
- H-3: 0 → 8 direkt: 12,1 % bei 81,79 %.
- Robustheit: nur 1/10 Seeds mit τ* = 0; ohne ihn 26,5 % (knapp über der
  25-%-Marke) — die Ersparnis ist auf Fashion kleiner als auf MNIST (11,7 %),
  weil die Clifford-Stufe dort öfter unsicher ist.

**Zwischenfazit Hybrid-Magie-Kaskade (zwei Datensätze, 20 frische Seeds):**
Ein klassisches, multiplikationsfreies Clifford-Netz entscheidet 86–92 % der
Bilder selbst; nur der Rest geht auf die Quantenschaltung mit wenigen
Magiewinkeln. Hardware-Aufwand **12–24 %** von „alles auf den
Quantenrechner, k=2“, bei gleicher oder besserer Genauigkeit.

## 25.09.2026, 03:10 — I-033 (Q3, vorregistriert): **Magie-Konzentration gilt auch für das Laden** ✔

MNIST 16×16, Seeds 140–149, gerundet werden die Gatterwinkel φ der
Gray-Code-Zerlegung der dünn besetzten Präparation (Basis I-020: im Mittel
~110 von 255 φ ≠ 0, 3.260 T); faire Referenz = exakt geladen mit gleicher
zweiter Phase (Rohdaten `ergebnisse/I-033__20260925_030825.json`, 72 min).

| Laden | alle π/2 | Top-4 | Top-16 | Top-32 | **Top-48** | Ebene 6 | exakt |
|---|---|---|---|---|---|---|---|
| Lade-T je Bild | 0 | 123 | 495 | 988 | **1.479** | 1.211 | 3.260 |
| Anteil | 0 % | 4 % | 15 % | 30 % | **45 %** | 37 % | 100 % |
| Genauigkeit | 0,535 | 0,677 | 0,881 | 0,928 | **0,938** | 0,883 | 0,944 |

- L-0 H1 (+3,6 pp über Identität, p = 0,002).
- **L-1 H1**: Top-48 hält die Referenz per TOST innerhalb 1 pp (−0,57 pp,
  Holm-p = 0,002) bei **45 % der Lade-T** — in allen 10 Seeds −0,1 … −0,9 pp.
- **L-2 H1**: *welche* Winkel exakt bleiben, entscheidet: Top-16 schlägt 16
  zufällige um **+31 pp** (p = 0,002).
- L-3: grob-zu-fein (obere Baumebenen) ist schwächer als Top-m. Lade-M50 =
  16 Winkel (15 % der T), Lade-M90 = 32 Winkel (30 %).

**Einordnung:** Dieselbe Struktur wie in der Schaltung: Wenige Winkel tragen
den Nutzen, wenn man sie nach Wirkung (hier: Abstand zum Clifford-Gitter)
wählt. Damit gibt es die Magie-Konzentration jetzt auf **beiden** Seiten der
Q3-Rechnung — Modell (4 von 32 Winkeln) und Daten (32–48 von ~110).
Kombiniert mit der Hybrid-Kaskade (nur 8–14 % der Bilder auf Hardware) ist
eine Größenordnung von ~2 % der heutigen Hardware-T denkbar — **nicht
gemessen**, nur geschätzt → Karte I-034.

## 25.09.2026, 04:50 — C-skala: Hinweis auf sublineares Magie-Wachstum, saubere Kontrolle fällt der Lernrate zum Opfer

MNIST, n = 6/8/10 Qubits (8×8, 16×16, 32×32) + fest10 (8×8 hochskaliert ⊗
|+⟩⁴), L = 4, 2.500 Trainingsbilder, 500 Schritte, Seeds 120–129, faire
Referenz *nachtrainiert* mit eigener Rate (Rohdaten
`ergebnisse/C-skala__20260925_044543.json`, 97 min).

| Konfiguration | Winkel | S-0 (nachtrainiert > Identität) | instabil | M50 | M90 | G(1) |
|---|---|---|---|---|---|---|
| echt6 | 24 | H1 (+3,8 pp) | 0/10 | 2 | 12 | 0,45 |
| echt8 | 32 | H0 (−1,3 pp) | **2/10** | – | – | – |
| echt10 | 40 | H1 (+5,3 pp) | 0/10 | **1** | 10 | 0,63 |
| fest10 | 40 | H0 (−8,8 pp) | **5/10** | – | – | – |

- **S-1 H1**: bei gleichem Winkelanteil (1/8, 1/4) ist der Gewinnanteil für
  n = 10 höher als für n = 6 (+0,14 G, KI +0,05 … +0,23, p = 0,0098, Holm).
  Magiebedarf wächst langsamer als n·L — **aber** mit der Bildauflösung
  konfundiert.
- **S-3 HU (Vorbedingung)**: die saubere Trennung (fest10) fällt aus, weil
  *nachtrainiert* auf 5/10 Seeds kollabiert.
- Ursache: Die Ratensuche (2 Tuning-Seeds) wählt 0,32 direkt an der
  Stabilitätskante (1,28 kollabiert). → Lehre Stabilitätsabstand
  (FEHLERBUCH), Folge C-skala-v2 (Idee).

## 25.09.2026, 05:25 — I-034 (Q3, vorregistriert): **Gesamt-Magie-Budget kombiniert: 2,25 % der Hardware-T, ohne Verlust** ✔✔

MNIST 16×16, 10 frische Seeds 210–219. Kaskade: Stufe 0 = Clifford-Schaltung
**klassisch** (0 T, kein Laden) → k=2 → k=8 auf dem Quantenrechner mit
**Top-48-geladenen** Zuständen (Lade-T bildweise exakt gezählt), τ auf
Validierung nach Hardware-Kosten (Rohdaten `ergebnisse/I-034__20260925_052057.json`, 35 min).
Referenz „heute“: k=2 fest, dicht geladen = 7.698 T je Bild.

| Verfahren | Hardware-T je Bild | Anteil | Genauigkeit |
|---|---|---|---|
| k=2 dicht (heute) | 7.698 | 100 % | 91,63 % |
| k=8 dicht | 7.883 | 102 % | 92,88 % |
| Hybrid-Kaskade, dicht geladen (I-015-hybrid) | ~900 | ~12 % | ≈ k2 |
| k=8, Top-48-Laden, ohne Kaskade | 1.727 | 22 % | 92,19 % |
| **Kombi m=48** | **174** | **2,25 %** | **91,94 %** |
| Kombi m=32 | 113 | 1,47 % | 91,59 % |
| k0 klassisch allein | 0 | 0 % | 90,83 % |

- PK H1 (k2 exakt schlägt Identität, +2,2 pp, p = 0,002).
- **G-1 H1**: m=48: 2,25 % der Hardware-T, **+0,31 pp** (einseitiges KI
  [+0,12; +0,51]); m=32: 1,47 %, −0,04 pp (KI [−0,20; +0,12]). Beide
  nichtunterlegen (Holm-p < 0,001). Teuerster Seed 370 T (4,8 %); ohne den
  einen τ* = 0-Seed im Mittel 2,5 %.
- **G-2 H0 (TOST)**: genähertes Laden kostet *in der Kaskade* nichts
  (−0,03 pp gegen die Kaskade mit exaktem Laden, TOST p = 0,005) — obwohl es
  die feste Stufe k=2 −0,93 pp kostet. **Wechselwirkung +0,90 pp (p = 0,002)**:
  Die Kaskade schickt nur unsichere Bilder weiter und reicht sie bis k=8
  durch; dort gleicht die zusätzliche Magie den Ladefehler aus.
- Nur 7,8 % der Bilder erreichen den Quantenrechner.

**Das ist der Kernbefund des Instituts bisher:** Drei Magie-Einsparungen —
*welche Bilder* (klassische Clifford-Stufe), *wie viel Laden* (Top-m),
*wie viel Schaltung* (wenige Magiewinkel) — multiplizieren sich, statt sich
zu stören: **~44-mal weniger T-Gatter** je Bild als der naive Weg, bei
gleicher Genauigkeit. Einschränkung: MNIST, ein Datensatz, T-Zahl mit
ε = 10⁻³ je Rotation; Fashion-Replikation und echte Hardware stehen aus.

## 25.09.2026, 07:15 — C-skala-v2: Mit stabiler Referenz löst sich der Skalierungs-Hinweis auf

Gleiche Konfigurationen wie C-skala (echt6/echt10/fest10), aber Ratenwahl mit
Stabilitätsabstand (min über 3 Tuning-Seeds 930–932, Gitter Faktor 2, oberer
Nachbar darf nicht kollabieren), frische Mess-Seeds 220–229 (Rohdaten
`ergebnisse/C-skala-v2__20260925_071112.json`, 110 min).

| Konfiguration | Gewinn (nachtrainiert − Id.) | instabil | Kollaps | G(1) | M50 |
|---|---|---|---|---|---|
| echt6 (24 Winkel) | +3,2 pp (p=0,002) | 0/10 | keiner | 0,49 | 2 |
| echt10 (40 Winkel) | +5,1 pp (p=0,002) | 0/10 | keiner | 0,61 | 0 |
| fest10 (40 Winkel) | +4,9 pp (p=0,002) | 1/10 | keiner | 0,31 | 5 |

- S-0 dreimal **H1** — diesmal mit sauberer, stabiler Referenz (d_nach ≈ 0
  in allen drei Konfigurationen).
- **S-1 HU**: Der Hinweis aus C-skala v1 (+0,14 G für n=10 gegenüber n=6)
  **verschwindet** — mit stabiler Rate d = **−0,08 G** (n.s., KI
  [−0,16; 0,00], p_rechts = 0,93). Kein Beleg mehr für sublineares Wachstum.
- **S-3 HU**: Die saubere Trennung Breite/Auflösung (fest10 gegen echt6)
  bleibt unentschieden (+0,13 G, KI [−0,04; 0,29], n.s.) — diesmal aber mit
  **sauberer** Methode (fast keine Instabilität), nicht wegen eines
  Auflösungsproblems.

**Wichtige methodische Lehre (FEHLERBUCH):** Der v1-Befund (S-1 „H1“) war
ein **Artefakt der instabilen Lernrate**, kein echtes, nur ungenau
gemessenes Signal. Nach der Korrektur löst er sich auf, statt sich zu
bestätigen. **Die Magie-Skalierungsfrage bleibt offen** — weder bestätigt
noch widerlegt, mit jetzt solider Methode für einen künftigen, größer
angelegten Versuch (mehr Seeds, mehr Breiten).

## 25.09.2026, 07:55 — I-034-fashion: Kombination repliziert NICHT — wichtiges Gegenbeispiel

Fashion-MNIST, 10 frische Seeds 230–239, sonst identisch zu I-034 (Rohdaten
`ergebnisse/I-034-fashion__20260925_075400.json`, 43 min).

- PK H1 (k2 exakt schlägt Identität, +2,4 pp).
- **G-1 H0** (Abbruchkriterium erfüllt): kombi_m48 kostet **12,4 %** statt
  ≤5 % und ist **nachweislich schlechter** (−0,34 pp, KI [−0,65; −0,03]);
  kombi_m32 kostet 10,3 %, −0,64 pp. Beide über der 10-%-Schranke.
- G-2 **H1**: Genähertes Laden kostet in der Kaskade auf Fashion diesmal
  *doch* nachweisbar Genauigkeit (−0,34 pp, p = 0,008) — anders als auf
  MNIST, wo die Wechselwirkung den Verlust genau ausglich.
- Ursache (Einordnung, nicht Teil der Vorregistrierung): Top-48-Laden ist auf
  Fashion verlustreicher (I-033 war nur MNIST) und mehr Bilder brauchen
  überhaupt die Quantenstufe (31–36 % statt 8 %, weil Fashion mehr Magie
  braucht, siehe C-fashion). Ein einzelner Blick lohnt sich: `zwei_0_8_m48`
  (0→8 ohne k2-Zwischenstufe) erreicht 504 T = **6,5 %** bei 81,61 % (Referenz
  81,88 %) — nahe dran, in der Vorregistrierung aber nicht die Hauptkonfiguration.

**Einordnung: Das ist kein Rückschlag, sondern der Beleg, dass wir ehrlich
vorregistrieren.** Der MNIST-Befund (I-034, 2,25 %) gilt für MNIST, nicht
allgemein — die Kombination ist datensatzabhängig und muss je Aufgabe neu
kalibriert werden (Top-m-Größe, welche Stufen). Genau das relativiert auch
den "44-fach"-Satz im Kernsatz: er gilt nachgewiesen für MNIST, nicht per se.

## 25.09.2026, 08:10 — Werkzeug (kein Befund): GPU-Encoder/Decoder-Stapelung, Korrektheit geprüft

Auf Maltes Wunsch (Vorgabe 7): `labor/stapel.py`. Idee: viele kleine,
unabhängige Rechenaufträge (z. B. N Seeds derselben QireZelle) zu EINEM
großen Tensor-Batch zusammenfassen (Encoder), auf der GPU in einem Rutsch
rechnen, wieder auseinandernehmen (Decoder) — statt N kleiner CPU-Prozesse.

**Offen gesagt: Das ist Werkzeug/Infrastruktur, kein neuer physikalischer
Befund** — Batched-Statevector-Simulation ist etablierte HPC-Technik
(cuQuantum, PennyLane-Lightning-GPU, arXiv:2509.04955 u. a.).

- **Korrektheit**: batched (`torch.func.vmap`) gegen Python-Schleife exakt
  gleich (Unterschied 0,00, bitgleich, Datentyp float32).
- **Laufzeit auf CPU** (keine GPU in diesem Container, `torch.cuda.is_available()
  == False`): nur 1,0–1,4× schneller, kein durchschlagender Gewinn — CPU
  profitiert kaum vom Batching, weil sie ohnehin nicht massiv parallel ist.
- **Der eigentliche Test steht aus**: Der Sinn der Übung ist GPU-Durchsatz.
  Das braucht eine CUDA-fähige Maschine — offene Frage an Malte, ob sein PC
  eine NVIDIA-GPU hat, dann lässt sich das dort wirklich messen.

## 25.09.2026, 10:25 — I-035: Magiebedarf vs. Aufgaben-Schwierigkeit — HU, aber mit einer echten Überraschung

MNIST 16×16, 10 Rausch-Stufen (0–100 % vertauschte Trainingsetiketten),
6 frische Seeds je Stufe (250–255), K_STUFEN=(0,2,8) (Rohdaten
`ergebnisse/I-035__20260925_102309.json`, 56 min). **Flaggschiff-Experiment
für „echte Neuheit" (Vorgabe 7): nichts Vergleichbares in der Literatur
gefunden** (verwandt, aber andere Frage: arXiv 2403.00919, 2409.13008).

**Rohlauf hatte einen Fehler, robuste Nachrechnung nötig:** Seed 253 kollabierte
bei Rauschanteil 0 % (Rate 0,32 an der Stabilitätskante — drittes Mal in dieser
Nacht). Das kippte P-0 zu einem falschen H0 und ließ die populationsbasierte
M50-Rechnung die wichtigste Stufe (0 % Rauschen) als undefiniert fallen.
Nachgerechnet mit derselben `auswertung()`-Funktion, nur die beiden vom Modul
selbst erkannten Ausreißer (Seed 253 bei 0 %, Seed 252 bei 85 %) entfernt:

| | Rohlauf (verzerrt) | Robust (10/10 Stufen) |
|---|---|---|
| P-0 (voll vs. Identität, 0 % Rauschen) | H0: −2,2 pp (Artefakt) | **+3,9 pp**, p=0,0625 (n=5, kann α=0,05 nicht erreichen) |
| P-1 (M50 ~ Rauschanteil) | rho=−0,31, HU | **rho=−0,44**, KI [−0,70;−0,30] durchgehend negativ, Permutations-p=0,196 → **HU** |
| P-2 (Generalisierungslücke ~ M50) | rho=−0,36, HU | rho=−0,14 → **H0**: kein Zusammenhang |

- **P-2 ist ein sauberes, informatives Negativergebnis**: Die
  Generalisierungslücke ist **kein** billiger Ersatz für den teuren
  Rundungs-Sweep (nach SRE in I-002 der zweite gescheiterte Kandidat).
- **P-1 bleibt HU**, aber mit einem durchgehend negativen Konfidenzintervall
  — eher ein Hinweis auf *weniger* statt *mehr* Magiebedarf bei mehr Rauschen.
  Das läuft der explorativen I-003-Beobachtung (Memorieren M50=8, echt M50=1)
  entgegen. **Beide Ergebnisse sind einzeln nicht belastbar** (I-003 nur zwei
  Extrempunkte, I-035 mit breitem, das Nullband nicht ausschließendem KI) —
  aber der Widerspruch selbst ist interessant: die Beziehung zwischen
  Aufgaben-Schwierigkeit und Magiebedarf ist offenbar **nicht** die einfache
  Monotonie, die I-003 nahelegte.

**Wichtigere Lehre (FEHLERBUCH):** Das ist die DRITTE Instabilität dieser Art
in einer Nacht (C-referenz, I-003-Struktur, jetzt I-035) — immer dieselbe
Ursache: Ratensuche ohne Stabilitätsabstand aus Zeitgründen. Ab sofort
gehört eine automatische Ausreißer-Erkennung vor die Urteilsbildung, nicht
nur als deskriptiver Zähler danach.

## 25.09.2026, 17:22 (nachgetragen 28.09.) — I-004-mnist: **ein Training genügt** ✔

MNIST, n = 500 und n = 10.000. Rohdaten `ergebnisse/I-004-mnist__20260925_172251.json`.
Der Lauf wurde um 23:00 ein zweites Mal gestartet und lieferte **identische**
Urteile und Zahlen (`…_230020.json`). Das ist keine zweite Stichprobe, zeigt
aber, dass die Pipeline deterministisch reproduziert. Nachgetragen am 28.09.,
weil der Lauf nie ausgewertet und committet worden war.

| Test | Urteil | Zahl |
|---|---|---|
| R-0 (Nachtraining schlägt Identität, n = 10.000) | H1 | +3,98 pp, p = 0,002 |
| R-P (Gradient erreicht im bekannten Gerüst die Nachtrain-Lösung) | H1 | fest2 − nach2 = −0,12 G, KI −0,19 … −0,06 |
| **R-1 (zufälliges Clifford-Gerüst + 4 Magiewinkel, EIN Training)** | **H1** | **G = 0,69**, Holm-p = 0,0059 |
| R-2 (nachträgliches Runden schlägt das Gerüst um ≥ 0,15 G) | H0 | −0,11 G, KI oben +0,02 |
| R-3 (Gerüst regularisiert bei wenig Daten) | H0, Gegenrichtung | −0,22 G, p = 0,003 |

- **R-1 + R-2 zusammen sind praktisch wichtig.** Man braucht weder ein volles
  Training noch ein nachträgliches Runden. Ein **zufälliges** Clifford-Gerüst,
  in dem von Anfang an nur 4 Winkel magisch und trainierbar sind, holt mit
  **einem** Training 69 % des Nutzens. Das nachträgliche Runden (zwei
  Trainings) ist nachweislich nicht um 0,15 G besser. Für Hardware heißt das:
  Die Schaltung braucht höchstens 4 Nicht-Clifford-Winkel, und das steht
  schon *vor* dem Training fest.
- **R-3 repliziert I-024 in der Gegenrichtung:** Auch im Gerüst trägt die Magie
  bei wenig Daten **weniger** bei (G sinkt von n = 10.000 auf 500). Die Idee
  „Clifford-Gerüst als Regularisierer für Datenknappheit“ ist damit zweimal
  widerlegt.
- Beschreibend: Bei n = 10.000 liegt das volle Training 1,24 pp über gz4
  (p = 0,002). Der Rest des Nutzens sitzt also in mehr als 4 Winkeln. Der
  I-001-Vorblick zeigt eine hohe Übereinstimmung der Rangfolgen „nach
  Verlust“ und „nach Abstand“ (Spearman +0,97/+0,98).
- Offen, und der natürliche nächste Schritt: R-1 auf **Fashion** (I-034-fashion
  hat gezeigt, dass MNIST-Befunde dort nicht automatisch halten).

## 28.09.2026, 12:09 — I-005: **Diskrete Suche findet reine Clifford-Schaltungen auf dem Niveau des vollen Trainings** ✔✔

MNIST 16×16, 10 Mess-Seeds (190–199), Strukturaufgabe mit Seeds 200–209.
Simulated Annealing über 2^(nL) Clifford-Wahlen, 6.000 Schritte. Auswahl nur
nach Ridge-LOO auf Nicht-Testdaten, ein gemeinsamer L-BFGS-Kopf für alle Arme.
Rohdaten `ergebnisse/I-005__20260928_120905.json`, 49 min. Vorregistriert und
committet vor dem Lauf.

| Test | Urteil | Zahl |
|---|---|---|
| Q-0 Positivkontrolle (Lehrer-Lösung finden) | **H1** | G_L = 0,96, Lehrer-äquivalent in 8/10 Seeds (Zufallssuche gleichen Aufwands: 2/10) |
| Q-1 Suche gegen Median von 100 Zufalls-Cliffords | **H1** | +2,56 pp, KI [+2,25; +2,87], 10/10, Holm-p = 0,006 |
| Q-2 Suche gegen k0 (gerundetes Gradientenmodell) | **H1** | +3,38 pp, KI [+2,56; +4,19], 10/10, Holm-p = 0,006 |
| Q-4 Suche gegen Zufallssuche gleicher Bewertungszahl | **H1** | +1,09 pp, KI [+0,63; +1,55], 9/10, Holm-p = 0,006 |

**Beschreibend (Q-3), Test:**

| Arm | Genauigkeit |
|---|---|
| Suche (**0 Magiewinkel**) | **0,9352** |
| voll trainiert (alle Winkel frei) | 0,9344 |
| beste von 100 Zufalls-Cliffords | 0,9230 |
| k0 (gerundet) | 0,9014 |
| Identität | 0,8969 |

- G_voll der Suche ist **1,02**, der von k0 nur 0,12.
- Siegerfluch: Kriterium minus Test liegt bei +0,46 pp. Die Auswahl
  überschätzt sich leicht.
- Aufwand: 5.242 Bewertungen, also 544 s Suche gegen 133 s Gradiententraining
  je Seed.

**Was das heißt:**
1. **Die Positivkontrolle, an der I-003 gescheitert ist, feuert jetzt.** Das
   Lernverfahren war das Problem, nicht die Aufgabe. Eine diskrete Suche findet
   die Clifford-Lösung, der Gradient nicht.
2. **Das stellt die Magie-Linie auf den Kopf:** Auf MNIST braucht die Zelle
   **gar keine Magie**, wenn man sie nicht per Gradient, sondern diskret
   trainiert. Suche und volles Training liegen gleich auf (0,9352 gegen
   0,9344). Das ist nur beschreibend, ohne TOST, und gehört als nächste
   vorregistrierte Frage auf die Agenda. Die „Magie-Konzentration“ aus
   C-mnist/C-referenz misst also vermutlich vor allem, **wie viel Magie der
   Gradient braucht**, nicht wie viel die Aufgabe braucht.
3. **Ehrlich dazu gehört:** Reine Clifford-Schaltungen sind nach
   Gottesman-Knill klassisch effizient simulierbar. Eine gute Clifford-Zelle
   ist damit ein Befund über **Hardware-Kosten** (0 T-Gatter, kein
   Magic-State-Distilling), nicht über einen Quantenvorteil. Für
   Q3-Hardware ist sie ideal, für „Quantenvorteil“ ist sie das Gegenteil.
4. Kaskade: Die Suche ist bei 61,6 % der Bilder sehr sicher (Konfidenz > 0,9),
   k0 nur bei 48,2 %. Eine Clifford-Stufe 0 aus der Suche würde die
   Hybrid-Kaskade (I-015-hybrid) noch billiger machen.

**Folgefragen (Stand 12:09, beantwortet in I-005-v2, siehe unten):** Repliziert das auf Fashion (I-005-fashion)? Suche gegen voll
per TOST, vorregistriert. Braucht Fashion (G(4) ≈ 1,0 aus C-fashion) dann
ebenfalls keine Magie? Und: Suche plus k Magiewinkel, also ein diskretes
Gerüst mit Politur, statt eines zufälligen Gerüsts wie in I-004.

## 28.09.2026, 20:34 — I-005-v2: **MNIST braucht keine Magie, Fashion schon** ✔

MNIST und Fashion-MNIST 16×16, **je 20 Seeds** (300–309 + 330–339 bzw.
310–319 + 340–349). Strukturaufgabe 360–369. Die Lernrate von `voll` wurde je
Datensatz mit Stabilitätsabstand gesucht (Ergebnis 0,08 bzw. 0,16), die
Kopf-L2 je Datensatz auf eigenen Tuning-Seeds (beide 0,003).
Rohdaten `ergebnisse/I-005-v2__20260928_203403.json`, etwa 3,3 h. Vortest und
Rauchtest bestanden, committet vor dem Lauf.

| Test | Urteil | Zahl |
|---|---|---|
| Q-0 Positivkontrolle | H1 | G_L = 0,94, Lehrer-äquivalent 6/10 (Zufallssuche 1/10) |
| **V-1 MNIST: Suche gegen voll** | **GLEICHWERTIG** | d = −0,14 pp, KI [−0,39; +0,11], Holm-p_TOST = 0,032, Sensitivität voll − id +3,91 pp |
| **V-2 Fashion: Suche gegen voll** | **VOLL_BESSER** | d = **−1,31 pp**, KI [−1,40; −1,21], 0/20 Seeds positiv, Holm-p < 10⁻⁴ |
| V-3 Fashion: Suche gegen Median Zufall | H1 | +0,73 pp, 20/20 |

**Beschreibend:**

| | Suche | voll | k0 | Median Zufall | Identität | G_voll(Suche) |
|---|---|---|---|---|---|---|
| MNIST | 0,9391 | 0,9405 | 0,9094 | 0,9135 | 0,9013 | 0,96 |
| Fashion | 0,8238 | 0,8368 | 0,8161 | 0,8165 | 0,7995 | 0,65 |

**Was das heißt:**
1. **Auf MNIST ist „keine Magie nötig“ jetzt vorregistriert belegt**, gegen
   einen fair getunten Gegenarm: 0 T-Gatter bei gleicher Genauigkeit
   (±0,5 pp). Das erste GLEICHWERTIG der Magie-Linie.
2. **Auf Fashion trägt die Magie wirklich etwas bei:** Die beste Clifford-
   Schaltung holt nur 65 % des Schaltungsgewinns. Das passt zu C-fashion
   (dort brauchte es mehr Magiewinkel). Der Magiebedarf ist also eine
   **Eigenschaft der Aufgabe**, nicht nur des Gradienten. Das korrigiert die
   Deutung von I-005 (12:09): Dort war die Vermutung „misst nur den Gradienten“
   aus MNIST allein zu weit gegriffen.
3. **Zwei `voll`-Seeds sind auf Fashion kollabiert** (314, 315, eigener Kopf
   mehr als 5 pp unter dem Median). Der Kollaps macht `voll` schlechter,
   VOLL_BESSER ist also konservativ. Das Urteil hält auch ohne die beiden
   Seeds (0/20 positiv). Der Zweig VOLL_BESSER prüft den Kollaps nicht; das ist
   hier unschädlich, gehört aber in den nächsten Modulbau.
4. Ehrlich dazu: Clifford-Schaltungen sind klassisch effizient simulierbar.
   „Gleichwertig auf MNIST“ ist ein Befund über **Hardware-Kosten**
   (0 T-Gatter), kein Quantenvorteil. Der Quantenanteil liegt genau in den
   1,3 pp, die Fashion zusätzlich braucht.

**Nächste Frage (natürliche Folge):** Wie viele Magiewinkel **auf der
gesuchten Clifford-Schaltung** schließen die Fashion-Lücke? Das hieße
Suche + k polierte Nicht-Clifford-Winkel, k ∈ {1, 2, 4, 8}. Damit würde
„M50/M90 einer Aufgabe“ zum ersten Mal unabhängig vom Gradienten-Startpunkt
gemessen.

## 29.09.2026, 16:20 — I-005-v3: 4 Magiewinkel schließen die halbe Fashion-Lücke, 8 nicht die ganze ✔

Fashion-MNIST 16×16, 20 Seeds (420–439), Tuning 960–962. Rohdaten
`ergebnisse/I-005-v3__20260929_162054.json`. Vorregistriert und committet am
28.09. Der erste Lauf ging am 28.09. um 23:04 beim Herunterfahren des
Rechners verloren; er hatte keine Messzahl gesichert. Am 29.09. um 13:30 neu
gestartet, mit unverändertem Code.

| Test | Urteil | Zahl |
|---|---|---|
| **M-1** k8 (Verlust-Auswahl) gegen voll | **VOLL_BESSER** | −0,43 pp, KI [−0,63; −0,24], 3/20 positiv, Holm-p = 0,002 |
| **M-2** k2 nach Verlust gegen k2 zufällig | **H1** | +0,30 pp, KI [+0,10; +0,49], 15/20, Holm-p = 0,024 |
| **M-3** voll gegen suche (Replikation I-005-v2 V-2) | **H1** | +1,38 pp, 20/20 |

Kein Kollaps bei den Urteilsarmen. Nur `k32_alle` ist auf Seed 420
kollabiert (beschreibender Arm).

**Dosis-Wirkung (beschreibend):** G_S(k) ist der Anteil der Lücke
suche → voll, der geschlossen wird.

| k | 0 | 1 | 2 | 4 | 8 | 32 (alle ab Suche) |
|---|---|---|---|---|---|---|
| nach Verlust | 0 | 0,21 | 0,43 | **0,57** | 0,69 | 0,42 |
| zufällig | 0 | 0,13 | 0,22 | 0,36 | 0,51 | – |

→ **M50_S = 4 Magiewinkel (≈ 123 T-Gatter)**, M90_S > 8.

**Was das heißt:**
1. **Der Magiebedarf von Fashion ist jetzt unabhängig vom Gradienten-Start
   gemessen:** Auf der besten Clifford-Schaltung schließen 4 gezielt
   gewählte Nicht-Clifford-Winkel die halbe Lücke. Die volle Lücke schließen
   auch 8 nicht (Restlücke 0,43 pp).
2. **Die Auswahl nach Verlust trägt** (M-2), besonders bei kleinem k. Die
   Magie-Konzentration gibt es also auch über der Clifford-Suche.
3. **Die Überraschung: `k32_alle` (alle 32 Winkel ab der Suche-Schaltung
   nachtrainiert) erreicht nur 0,42**, also *weniger* als k = 8 und 0,80 pp
   hinter voll aus dem Zufallsstart. Die Suche liegt in einem anderen Becken
   der Verlustlandschaft als das Gradiententraining, und der Gradient kommt
   von dort nicht zur besseren Lösung. Das ist die in der Karte vorab
   genannte Folgefrage („sonst sitzt die Suche in einem anderen Becken“), jetzt
   beantwortet. Einschränkung: nur beschreibend, und ein Seed ist kollabiert.
4. **Einordnung:** Die Rz-Drehungen sind auf heutiger Hardware virtuell
   (Recherche 29.09. Hardware). Die T-Gatter-Zahl zählt erst bei
   fehlertoleranten Rechnern. Auf NISQ ist „0 bzw. 4 Magiewinkel“ vor allem ein
   Befund über klassische Simulierbarkeit.

**Folgen:**
- Gierige Auswahl oder k = 12/16, um die Restlücke zu schließen.
- Die Klassische Kontrolle aus der Recherche 29.09.: quadrierte
  Walsh-Hadamard-Merkmale ohne Suche. Erreichen sie die Suche, ist die
  Clifford-Zelle auf MNIST ein Walsh-Klassifikator, was Q-019W und die
  Institutslinie verbinden würde.
- Die Becken-Frage: Suche + k32 mit kleinerer Rate oder mehr Schritten.

## 05.10.2026, 16:42 — I-036: Die Clifford-Suche ist kein Walsh-Klassifikator — weder global noch als fester 2×2-Block ✔

`ergebnisse/I-036__20261005_164224.json`, vorregistriert und committet am
29.09. (`f1a669a`), gerechnet im Cloud-Container (4 Prozesse, zweiter
Anlauf; der erste ging mit einem Container-Neustart verloren, siehe
FEHLERBUCH). 20 Mess-Seeds je Datensatz, gleicher L-BFGS-Kopf (L2 0,003),
TOST ±0,5 pp, Holm über W-1…W-4.

| Arm (Testgenauigkeit, Mittel 20 Seeds) | MNIST | Fashion |
|---|---|---|
| Suche (0 Magiewinkel) | **0,9371** | **0,8268** |
| fester block2x2 (k=2) | 0,9278 | 0,8234 |
| fester block8x8 (k=6) | 0,9256 | 0,8148 |
| fester block4x4 (k=4) | 0,9107 | 0,8081 |
| Zufalls-Clifford, bestes von 100 | 0,9280 | 0,8233 |
| Zufalls-Clifford, Median | 0,9138 | 0,8180 |
| Identität (Pixelquadrate) | 0,9015 | 0,8001 |
| walsh_fest (global, k=8) | 0,8812 | 0,7968 |

**Urteile (vorregistriert):**
- Q-0 **H1**: Positivkontrolle — die Suche findet die Lehrer-Lösung
  (lehrer-äquivalent in 6/10, Zufallssuche gleichen Aufwands 0/10).
- W-1 **SUCHE_BESSER** (MNIST): Suche − walsh_fest = +5,59 pp,
  KI [+5,30; +5,89], 20/20 Seeds, Holm-p < 10⁻⁴.
- W-2 **SUCHE_BESSER** (Fashion): +3,00 pp, KI [+2,82; +3,18], 20/20.
- W-3 **H0 (Gegenrichtung)**, wie vorhergesagt: walsh_fest − Identität =
  −2,03 pp (MNIST), 0/20.
- W-4 **SUCHE_BESSER** (MNIST): Suche − block2x2 = +0,93 pp,
  KI [+0,67; +1,20], 19/20, Holm-p = 0,0001; TOST verfehlt (p = 0,99).
  block2x2 war nur auf den Rauchtest-Seeds 942–947 gewählt (kein Siegerfluch
  auf den Mess-Seeds).

**Beschreibend:** Alle 20 + 20 Suchergebnisse haben Bruhat-Form, alle
brauchen eine Vorzeichenmaske, keins ist Walsh-äquivalent. Blockdimension
k der Suche: MNIST Mittel 5,7 (4–7), Fashion 6,45 (5–8). Auf Fashion liegt
Suche − block2x2 nur bei +0,34 pp (14/20, außerhalb der Familie).

**Was das heißt:**
1. **„Die Clifford-Zelle ist ein verkappter Walsh-Klassifikator“ ist
   widerlegt** — global klar (−5,6 pp), als fester lokaler 2×2-Block knapp
   aber signifikant (−0,9 pp MNIST). Der 0-T-Befund aus I-005-v2 lässt sich
   also *nicht* auf eine feste klassische Transformation zurückführen; die
   gesuchte Maske/Blockwahl trägt.
2. **Ehrliche Relativierung:** Ein fester 2×2-Block-Walsh *ohne jede Suche*
   holt auf MNIST 72 % des Abstands Identität → Suche (0,9278 von 0,9371) und
   liegt gleichauf mit dem besten von 100 Zufalls-Clifford-Schaltungen. Auf
   Fashion ist der Rest-Vorsprung der Suche klein (+0,34 pp). Der größte Teil
   des Clifford-Nutzens ist *lokale* Hadamard-Mischung; die Suche liefert den
   letzten Prozentpunkt.
3. Die feste Dosis-Kurve ist nicht monoton in k: lokal schlägt global. Das
   passt zu Q-019W (Walsh trägt dort über Gewichtsteilung, nicht über
   Globalität) und ist eine Brücke zur Hauptlinie A.

**Folge (vorregistriert für W-4 SUCHE_BESSER):** Gesuchte Blockform mit
fester Maske vs. volle Suche → trägt die Maske oder die Blockwahl? In der
Agenda als **I-036-v2** (Status *idee*), vor dem Bauen Literatursuche
(Vorgabe 7).

## 05.10.2026, 18:24 — I-036-v2: Maske und Blöcke tragen nur **gemeinsam** — starke Wechselwirkung ✔

`ergebnisse/I-036-v2__20261005_182417.json`, vorregistriert und committet vor
dem Lauf (`2c669c9`), 66 min im Container (4 Prozesse, Wiederaufnahme je
Seed, Zwischenstände committet). Modul `abteilungen/maske_block.py`
(Forscher-Agent, Vortest vom Direktor wiederholt). Je Seed wird die gefundene
Suchschaltung in Bruhat-Form (V_s, m_s) zerlegt und in vier Varianten mit
demselben Kopf gemessen. 20/20 Seeds je Datensatz gültig.

| Testgenauigkeit (Mittel 20) | MNIST | Fashion |
|---|---|---|
| suche (V_s + Maske m_s) | **0,9396** | **0,8247** |
| block2x2 (fest, ohne Maske) | 0,9285 | 0,8226 |
| block2x2_maske (fester Block + m_s) | 0,9266 | 0,8192 |
| ohne_maske (V_s, Maske weg) | 0,9204 | 0,8012 |
| Identität | 0,9014 | 0,8004 |

**Urteile (vorregistriert):**
- Q-0 **H1** (Lehrer-Lösung in 8/10), **P-0 bestanden**: suche − block2x2 =
  +1,11 pp MNIST, 20/20 — I-036 W-4 repliziert auf frischen Seeds.
- P-0F (Fashion) **durchgefallen**, wie vorhergesagt: +0,21 pp (< 0,5 pp).
  → Keine Gleichwertigkeitsaussage aus dem Fashion-Teil (Regel 6); M-3 ist
  eine Richtungsaussage.
- **M-1 SUCHE_BESSER** (MNIST): suche − ohne_maske = +1,93 pp,
  KI [+1,28; +2,57], 17/20, p < 10⁻⁴.
- **M-3 SUCHE_BESSER** (Fashion): +2,36 pp, KI [+2,10; +2,62], 20/20.
- M-2/M-4 beschreibend: Die Maske der Suche ist auf 2×2-Blöcken nur in 4/20
  Seeds wirksam; dort *schadet* sie dem festen Block (−0,96 / −1,73 pp, 0/4).

**Der eigentliche Befund (beschreibend, 2×2-Zerlegung):**
Wechselwirkung **+2,12 pp** (MNIST, 17/20) und **+2,70 pp** (Fashion, 20/20).
Die gesuchten Blöcke *ohne* ihre Maske sind schlechter als der einfache
2×2-Block (MNIST −0,82 pp, Fashion −2,14 pp, 0/20), und die Maske *ohne* ihre
Blöcke hilft auch nicht. Erst das Paar (V_s, m_s) liegt vorn.

**Was das heißt:**
1. Die Suche findet weder „gute Blöcke“ noch „eine gute Maske“ getrennt,
   sondern **aufeinander abgestimmte Paare** aus Walsh-Unterraum und
   quadratischer Vorzeichenmaske. Für eine Clifford-Schaltung ist das
   natürlich: Die Bruhat-Form ist *eine* Gruppenoperation, deren Teile nur
   zusammen eine sinnvolle Basis bilden. Die quadratische Phase (S-/CZ-Anteil)
   ist also nicht schmückendes Beiwerk, sondern trägt die Hälfte des Nutzens.
2. Damit ist die naheliegende klassische Abkürzung („nimm feste Block-Walsh-
   Merkmale“) zweimal widerlegt (I-036 W-4, I-036-v2 M-1/M-3). Der 0-T-Befund
   ist eine Leistung der **Suche im Clifford-Raum**, nicht einer festen
   Transformation.
3. Einschränkung: Alles ist exakt klassisch simulierbar (Gottesman-Knill);
   die Aussage ist eine über die *Struktur* des Clifford-Raums als
   Modellklasse, nicht über Quantenvorteil.

**Folge (vorregistriert für M-1 SUCHE_BESSER):** Welche Maskeneigenschaft
trägt — Rang der quadratischen Form, Transfer einer Maske zwischen Seeds?
In der Agenda als **I-036-v3** (Idee).

## 05.10.2026, 20:05 — I-036-v3: Der **Rang** der Maske trägt gut die Hälfte, die **konkrete Form** den Rest — und gute Formen sind übertragbar ✔

`ergebnisse/I-036-v3__20261005_200505.json`, vorregistriert und committet vor
dem Lauf (`db0a336`), 73 min im Container (4 Prozesse, Wiederaufnahme,
Zwischenstände committet). Modul `abteilungen/maske_eigenschaft.py`
(Forscher-Agent, Vortest vom Direktor wiederholt). Je Seed: gefundene
Bruhat-Form (V_s, B_s); `zufall_gleicher_rang` = Median über 20 Formen
B_r = PᵀB_sP mit P gleichverteilt in GL(k,2) (gleicher Rang der
alternierenden Form, der vollen Invariante des merkmalswirksamen Teils).
20/20 Seeds je Datensatz analysierbar, kein Rang-0-Seed.

| Testgenauigkeit (Mittel 20) | MNIST | Fashion |
|---|---|---|
| suche (V_s, B_s) | **0,9359** | **0,8233** |
| zufall_gleicher_rang (V_s, B_r) | 0,9238 | 0,8184 |
| ohne_maske (V_s, 0) | 0,9105 | 0,8009 |
| Identität | 0,9017 | 0,7993 |

**Urteile (vorregistriert):**
- Q-0 **H1** (Lehrer-Lösung 7/10). **P-0 und P-0F bestanden**: suche −
  ohne_maske = +2,54 pp (MNIST, 20/20), +2,24 pp (Fashion, 20/20) — I-036-v2
  M-1/M-3 auf frischen Seeds repliziert.
- **R-1 SUCHE_BESSER** (MNIST): suche − zufall_gleicher_rang = +1,22 pp,
  KI [+1,01; +1,42], 20/20, Holm-p < 10⁻⁴.
- **R-2 ZUFALL_BESSER** (MNIST): zufall_gleicher_rang − ohne_maske = +1,32 pp,
  KI [+0,67; +1,98], 14/20, Holm-p = 0,008.
- **R-3 SUCHE_BESSER** (Fashion): +0,49 pp, KI [+0,37; +0,62], 19/20.
- **R-4 ZUFALL_BESSER** (Fashion): +1,75 pp, KI [+1,53; +1,96], 20/20.
- Zusammen (R-Z, beschreibend): **RANG_UND_FORM** auf beiden Datensätzen;
  Anteil des Rangs am Maskengewinn **52 %** (MNIST), **78 %** (Fashion).
  Genau die vor dem Lauf angesagte Vorhersage (RANG_UND_FORM, ~60 %).

**Beschreibend:**
- Rang ↔ Maskengewinn über Seeds: Spearman +0,79 (MNIST), +0,49 (Fashion);
  Rang ↔ Zufallsmasken-Gewinn +0,90 (MNIST). Höherer Rang = mehr Nutzen.
  Ränge MNIST {2: 4, 4: 6, 6: 10}, Fashion {2: 1, 4: 5, 6: 14}.
- **Fremde Maske** (Form eines anderen Seeds, wo V_s ⊆ V_t; 16/20 MNIST):
  fremde − ohne +2,19 pp, fremde − zufall +0,84 pp (16/16), suche − fremde nur
  +0,29 pp. → **Gute Formen sind zwischen Seeds übertragbar**: Die Form ist
  eher eine Eigenschaft der *Aufgabe* als des einzelnen Suchlaufs.
- Siegerfluch der Formwahl (Vorsprung im LOO-Kriterium minus im Test):
  +0,55 pp (MNIST), +0,99 pp (Fashion). Die Suche überschätzt ihren Vorsprung
  im eigenen Kriterium; die Urteile oben sind aber auf dem Testsatz gemessen.
  Auf Fashion ist der verbleibende Formvorteil (+0,49 pp) klein — knapp an der
  TOST-Grenze, aber 19/20 positiv.

**Was das heißt:**
1. Der CZ/S-Anteil der Clifford-Schaltung (die quadratische Phase) wirkt auf
   zwei Ebenen: Schon **wie viel** Verschränkungsphase (Rang der Form) zählt —
   eine zufällige Form gleichen Rangs holt gut die Hälfte (MNIST) bis drei
   Viertel (Fashion) des Gewinns. **Welche** Form, entscheidet den Rest.
2. Der Rang einer quadratischen Form über GF(2) ist eine echte
   Quanteneigenschaft der Schaltung (er bestimmt z. B. die Verschränkung des
   zugehörigen Graphzustands). „Mehr Verschränkungsphase → bessere Merkmale“
   ist damit über Seeds gemessen, nicht nur behauptet — innerhalb einer
   klassisch simulierbaren Modellklasse.
3. Übertragbarkeit legt eine billigere Methode nahe: eine gute Form einmal
   suchen und für neue Seeds bzw. Blöcke wiederverwenden.

**Folgen (vorregistriert):** R-1 SUCHE_BESSER → Form-Suche bei festem V_s und
Rang (nur CZ/S-Lage); R-2 ZUFALL_BESSER → Dosis-Kurve über den Rang 0…k.
Neu aus dem Beschreibenden: Übertragung einer festen „Aufgabenform“ auf neue
Seeds als eigene Karte (Methode statt Suche je Seed).

## 06.10.2026 — I-039 „MNIST mit ~100 Parametern“ vorregistriert (Modul `abteilungen/mnist100.py`)
Vorregistrierung steht im Modulkopf (Arme a–f, Zählregel reell/Strukturbit/Bit-pro-pp, Hypothesen H1–H3, Holm, Kriterien
und Urteilsstrings im Code). Seeds 2020–2039, Tuning-Seeds 2090–2092. Vortest (jeder Urteilszweig erzwungen) bestanden,
Rauchtest 1 Seed (2095) gelaufen. **Offenlegung:** ein früher Rauchtest sah die Testzeile von Seed 2020 (150 Schritte);
Kriterien blieben unverändert. Lauf nicht gestartet (wartet auf Freigabe E2).

## 06.10.2026 — I-038 „Struktur-Äquivalenz“: VORREGISTRIERUNG (vor jeder Zahl des Volllaufs)

Modul `abteilungen/struktur_aequivalenz.py`, Plan `PLAN_2026-10-06_gesamt.md` §2 P1, Vorlage `recherche/2026-10-05_vertiefung/v3_kontrollen.md`. Vortest (alle Urteilszweige erzwungen) und Rauchtest (1 Seed, kleine Daten) bestanden; der Volllauf ist **nicht** gestartet.

- **Daten/Seeds:** MNIST und Fashion, 16×16, 6.000 Training / 5.000 Test je Seed; Seeds 2000–2019; Clifford-Suche gepaart nur auf 2000–2009; Tuning-Seed 2099 (Validierung 1.000 aus 6.000, nie Test).
- **Arme (gleicher Kopf, Softmax-Regression, L2 3e-3, 2.570 P.):** Pixel-Softmax, Identität², Haar², Haar linear, Fastfood-2 (H·D₂·H·D₁)², QIRE L=4 Zufallswinkel / gelernt / ohne CNOT gelernt, Clifford-Suche, Cayley gelernt; Erweiterung: Haar-Mischung mit ReLU, tanh(a·z) (a auf Tuning-Seed), |z|, z² ohne Normierung von x, ReLU-MLP 256→9→10; Negativkontrolle Zufallslabel (Haar², permutierte Trainingslabels).
- **Lernraten** (QIRE, QIRE ohne CNOT, Cayley, MLP) je Arm und Datensatz mit `gittersuche_stabil` auf Tuning-Seed 2099, 300 Adam-Schritte wie v3.
- **Hypothesen, Holm x7 je Datensatz:** H-A QIRE ≡ Haar² (TOST ±0,5 pp); H-B QIRE ≡ QIRE-Zufallswinkel (TOST); H-C Fastfood-2 ≡ Haar² (TOST); H-D Haar² − Pixel-Softmax > 0; H-E Haar² − Haar linear > 0; H-F Suche < Haar² (einseitig, 10 Seeds); H-G Haar² ≥ bester klassischer Nichtlinearitäts-Arm (auf Tuning-Seed bestimmt; Superiorität einseitig, TOST zusätzlich mit Bonferroni x2). Cayley − QIRE nur berichtet, mit Parameterzählung.
- **Kontrollen:** NK Zufallslabel-Mittel in [0,09; 0,115], sonst alle Urteile HU/UNKLAR. PK je Datensatz: H-D und H-E je Holm-signifikant mit ≥ 0,5 pp, sonst kein GLEICHWERTIG. Kollaps-Prüfung (Seed < Median − 5 pp) macht betroffene Urteile HU/UNKLAR.
- **Urteile (Strings):** GLEICHWERTIG / X_BESSER / HU (A–C), HAAR_BESSER / H0 / HU (D), QUADRAT_TRAEGT / H0 / HU (E), SUCHE_NICHT_BESSER / H0 / HU (F), BORN_BESSER / GLEICHWERTIG / KLASSISCH_BESSER / UNKLAR (G). Kriterien: `urteil()` im Modul.
- **Erwartung (v3):** H-A/B/C GLEICHWERTIG, H-D und H-E ja (+2,9/+1,6 bzw. +2,5/+1,2 pp), H-F SUCHE_NICHT_BESSER auf Fashion; H-G offen (Quadrat gegen |z| nur +0,5/+0,3 pp in v3, daher KLASSISCH_BESSER/GLEICHWERTIG möglich).
- **Start:** `INSTITUT_PROZESSE=4 python direktor.py lauf I-038` (Wiederaufnahme je Seed über `ergebnisse/_zwischen_I-038.json`). Laufzeit ≈ 40–55 min bei 4 Prozessen.

## 06.10.2026 — I-040 „Holevo-Wirkungsgrad“: VORREGISTRIERUNG (vor jeder Zahl des Volllaufs)

Modul `abteilungen/holevo_wirkungsgrad.py`, Plan `PLAN_2026-10-06_gesamt.md` §2 P4, Vorlage `recherche/2026-10-05_vertiefung/v4_holevo_schuesse.md` (Erkundung, Seeds 960–969). Vortest (alle Urteilszweige auf synthetischen Daten erzwungen) bestanden; Rauchtest NUR auf dem Tuning-Seed 2098 (kleine Daten, Mini-Suche) gelaufen, **kein Mess-Seed 2050–2059 wurde berührt**. Volllauf **nicht** gestartet (wartet auf Freigabe E2).

- **Frage:** Wie viel der Holevo-Größe χ_C nutzt eine Born-Auslese pro Schuss, und was kostet das an Genauigkeit unter Schussrauschen?
- **Daten/Seeds:** MNIST 16×16 (6.000 Training / 5.000 offizielle Test), Mess-Seeds 2050–2059, Tuning-Seed 2098 (Validierung 1.200 der 6.000, nie Test). Fashion optional (`fashion=True`), nur beschreibend.
- **Arme:** Identität, Haar-Orthogonale, Fastfood-2, feste Bent-4×4 (Rang 4, Kreuzkopplung), Clifford-Suche **seed-eigen** (wie I-038); Negativkontrolle (Identität, permutierte Labels).
- **Köpfe** (Softmax-Regression, 2.570 P., L2 je Kopftyp auf Tuning-Seed 2098 aus {1e-3, 3e-3, 1e-2}, Gitter beidseitig verlängert): *exakt* (auf exakten Wahrscheinlichkeiten trainiert, auf Schusshäufigkeiten getestet) und *schussbewusst* (auf Häufigkeiten gleicher Schusszahl trainiert). S ∈ {10, 30, 100, 300, 1000}; 3 Schussrealisierungen je Testbild, beide Köpfe gepaart auf denselben Ziehungen.
- **Maße:** I(Klasse; 1 Schuss) aus Test-Klassenmitteln (Prior = Trainingsanteile), χ_C aus Trainings-Zustandsmatrizen, η = I/χ_C; Schussbedarf S90 = log-lineare Kreuzung von 0,90 auf der seedgemittelten schussbewussten Kurve, nicht erreicht → zensiert (S90 := 2000).
- **Hypothesen, Holm über 3:** H1 I(Identität) > I(Haar) und > I(Suche) (einseitiger exakter Wilcoxon, p = max beider, zusätzlich beide Mittel > 0). H2 schussbewusst − exakt > 0 bei S = 10, 30 und 100 (Mittel über die 5 Arme, je S einseitiger exakter Wilcoxon, p = max). H3 Spearman η ~ S90 über die 5 Arme, einseitig ρ < 0, exakter Permutationstest (5! Anordnungen, kleinster p = 1/120).
- **Kontrollen:** PK-1 Identität exakter Kopf, S→∞ ≥ 0,88; PK-2 exakter Kopf S = 10 mindestens 10 pp unter S = 1000; NK Information < 0,03 Bit und Genauigkeit ≤ 0,15. NK fällt → alle Urteile HU; PK fällt → H2/H3 HU. Instabilität (≥ 2 Seeds unter Median − 5 pp, bewusster Kopf S = 100) → H2/H3 HU INSTABIL.
- **Urteile (Strings):** H1 IDENTITAET_BESTE / H0 NICHT_UEBERLEGEN / HU NICHT_ENTSCHIEDEN; H2 SCHUSSBEWUSST_BESSER / H0 EXAKT_BESSER / H0 KEIN_VORTEIL / HU NICHT_ENTSCHIEDEN / HU INSTABIL / HU PK_VERSAGT; H3 ETA_SAGT_VORAUS / H0 ETA_NICHT_PROGNOSTISCH / HU ZENSIERT / HU NICHT_ENTSCHIEDEN / HU INSTABIL / HU PK_VERSAGT. Kriterien: `urteil()` im Modul.
- **Erwartung (v4):** H1 ja (≈ 0,36 gegen ≈ 0,16 Bit, praktisch deterministisch). H2 bei S = 10/30 ja, bei S = 100 knapp; der Schnitt-Vereinigungs-Test ist streng. **H3 eher nein oder unentschieden**: In v4 hat die Identität das höchste η, erreicht aber 90 % nicht (zensiert), Bent/Suche haben weniger η und erreichen 90 % — das wäre die Gegenrichtung.
- **Grenzen/Offenlegung:** Testbilder stammen aus denselben 10.000 offiziellen Bildern (sd über Seeds unterschätzt die Genauigkeitsunsicherheit); eine Schussziehung je Trainingsbild; Bent-Form fest; η-Neuheit nicht geprüft (Plan E4) — bis dahin nicht als neu formulieren. Beim Rauchtest wurden Zeitwerte gemessen (4 s je Kopf-Aufgabe, 0,06 s je Suchbewertung), keine Mess-Seed-Zahlen.
- **Start:** `INSTITUT_PROZESSE=4 python direktor.py lauf I-040` (Wiederaufnahme je (Seed, Arm) in `ergebnisse/_zwischen_I-040/`). Laufzeit ≈ 20–25 min bei 4 Prozessen (Suche ≈ 6 min je Seed dominiert; Fashion +≈ 20 min).

## 06.10.2026 – I-039 ERGEBNIS (MNIST mit ~100 Parametern, 20 Seeds 2020–2039, Container)
Rohdaten: `ergebnisse/I-039__20261006_051108.json`. Offenlegung (siehe Vorregistrierung): Rauchtest einmal auf Seed 2020 mit 150 Schritten; die Kriterien waren da schon fixiert.
- PK-1 OK (Voll-Kopf Haar² 92,1 %), PK-2 OK (klassisch d 74,8 %), NK OK (Zufallslabel 9,95 %).
- **H1 BESTÄTIGT:** b_ff (Fastfood-Mischung → Born-Quadrat → Gruppensummen + 10×10-Kopf) **80,46 ± 0,81 %** bei **110 reellen Parametern** (Strukturbits: 882 formal / 3.046 streng).
- **H2 ÜBERLEGEN:** b_ff − klassisch 5×5→3 tanh→10 (118 P.) = **+5,61 pp** (Holm-p < 10⁻⁴).
- **H3 MISCHUNG_TRÄGT:** b_ff − quadrierte Pixel ohne Mischung = **+2,74 pp**.
- Beschreibend: (a) nur 10 Bias 74,1/74,3 %; (e) Fisher-Top-9 63,4 %; (f) kopffreie Klassenregister-Auslese, 0 reelle Parameter: 71,7/72,7 %.
- Einordnung: Das ist besser als die v1-Erkundung (76,4 %), weil Fastfood statt Bent verwendet und die Lernrate je Arm stabil gesucht wurde. Ein Vorteil gegenüber der klassischen Referenz bei gleicher Zahl reeller Parameter, aber mit ~900 Strukturbits. Der faire Vergleich nach Bits steht in H2 nicht und ist eine Folgefrage (klassische Arme mit gleichem Bitbudget).

## 06.10.2026 – I-038 ERGEBNIS (Struktur-Äquivalenz, 20 Seeds 2000–2019, Suche gepaart 2000–2009, Container)
Rohdaten: `ergebnisse/I-038__20261006_065007.json`. Kontrollen: Die NK ist bestanden (Zufallslabel 10,0 / 10,2 %), die PK (H-D/H-E) auf beiden Datensätzen ebenfalls.

| Hypothese | MNIST | Fashion |
|---|---|---|
| H-A QIRE gelernt ≡ Haar² | GLEICHWERTIG (+0,18 pp) | GLEICHWERTIG (+0,10) |
| H-B QIRE gelernt ≡ Zufallswinkel | GLEICHWERTIG (+0,22) | GLEICHWERTIG (+0,07) |
| H-C Fastfood-2 ≡ Haar² | GLEICHWERTIG (+0,09) | GLEICHWERTIG (+0,01) |
| H-D Haar² − Pixel-Softmax | **+2,84 pp** | **+1,43 pp** |
| H-E Quadrat − linear | **+2,21 pp** | **+1,27 pp** |
| H-F Suche < Haar² | HU (−0,04) | SUCHE_NICHT_BESSER (−1,11) |
| **H-G Born² − beste klassische Nichtlinearität (|z|)** | **BORN_BESSER +0,35 pp** [0,27; 0,43] | GLEICHWERTIG (0,00) |

Beschreibend: Cayley − QIRE +1,61 / +1,35 pp bei 35.210 gegen 2.602 Parametern.

**Deutung:** v3 ist bestätigt. Die Zelle ist gleichwertig zu einer festen dichten Mischung (Haar oder Fastfood); gelernte Winkel tragen nichts Messbares. Die Born-Nichtlinearität (Quadrat) schlägt auf MNIST die beste klassische Alternative knapp (+0,35 pp), auf Fashion ist sie gleichwertig. Der Effekt ist klein, aber signifikant.

Technik: `direktor.py` brach nach dem Sichern ab, weil die Folgen-Schlüssel in agenda.json „H-A GLEICHWERTIG“ statt „H-A:GLEICHWERTIG“ lauteten. Ich habe den Eintrag von Hand nachgetragen und die Schlüssel korrigiert.

## 06.10.2026 – I-040 ERGEBNIS (Holevo-Wirkungsgrad, 10 Seeds 2050–2059, Container)
Rohdaten: `ergebnisse/I-040__20261006_070724.json`.
- **PK-1 VERSAGT:** Die Identität mit exaktem Kopf erreicht 0,868 < 0,88 (vorregistrierte Schwelle). Damit sind **H2 und H3 HU**, weil ihre Vorbedingung fehlt.
- PK-2 OK, NK OK (I 0,004 Bit, acc 0,099).
- **H1 IDENTITÄT_BESTE:** I(Klasse; 1 Schuss): Identität − Haar **+0,197 Bit**, Identität − Suche **+0,193 Bit** (Holm-p 0,003, 10/10 Seeds). **Dichte Mischung verschlechtert die Information pro Schuss**, obwohl sie die Genauigkeit bei exakter Auslese verbessert.
- Vermutung zu PK-1: Die Schwelle 0,88 wurde aus v4 (andere Kopfregularisierung) übernommen und lag zu knapp. Kein nachträgliches Umdeuten; eine Wiederholung braucht eine neue Vorregistrierung (I-040-v2).

## 06.10.2026 — I-040-v2 „Holevo-Wirkungsgrad, Wiederholung“: VORREGISTRIERUNG (vor jeder Zahl auf Mess-Seeds)

Modul `abteilungen/holevo_wirkungsgrad_v2.py` (Docstring enthält alle Kriterien), Folge von I-040 (PK-1 versagt: Identität exakt 0,868 < 0,88; H2/H3 HU; H1 bestätigt). Vortest (jeder Urteilszweig erzwungen, inkl. PK-1-Grenze, H4-Zweige, Wiederaufnahme) bestanden; Rauchtest nur auf dem Tuning-Seed 2198 (kleine Daten). **Volllauf nicht gestartet** (wartet auf Maltes Zustimmung).

- **Ursache der 0,868 (aus Dateien/Tuning-Seed, nicht aus Mess-Seeds):** I-038 hat Identität² = 0,9005 (MNIST) mit gleichem Datenumfang (6.000/5.000, 16×16), Kopf L2 = 3e-3, max_iter = 500. I-040 wählte L2 des exakten Kopfs auf Validierung **unter Schussrauschen** (0,09, 30× stärker) und max_iter = 200 → schlechtere Genauigkeit bei exakter Auslese. Bestätigung auf dem Tuning-Seed 2198 (Volldaten, nur Kopf): I-038-Kopf 0,904, I-040-Kopf 0,874, Pixel-Softmax 0,915 (Abstand −1,1 pp). Die absolute Schwelle 0,88 kam aus v4 ohne Bezug zu I-038.
- **Änderungen:** (i) Kopf „exakt“ = I-038-Kopf (L2 3e-3 fest, 500 Iterationen). (ii) **PK-1 relativ:** Identität exakt ≥ Pixel-Softmax − 2 pp (Mittel über Seeds; Pixel-Softmax = I-038-Kopf auf Rohpixeln, gleiche Seeds). Begründung der 2 pp: I-038-Abstand MNIST −0,9 pp (Fashion −2,3), Tuning-Seed −1,1 pp; 2 pp lässt gut 1 pp Luft, fängt aber einen kaputten oder überregularisierten Kopf. (iii) **H2 strenger:** Gegner je (Seed, Arm, S) der bessere von „exakt“ und „exakt_rob“ (exakt trainiert, L2 auf Tuning-Seed unter Schussrauschen gewählt wie I-040); schussbewusster Kopf L2 auf Tuning-Seed 2198 (Gitter {1e-3 … 1e-1}, beidseitig verlängerbar).
- **Seeds:** Mess 2150–2159, Tuning 2198 (disjunkt zu I-040 und allen früheren). MNIST 16×16, 6.000/5.000; Fashion optional, nur beschreibend.
- **Hypothesen, Holm über 4, α = 0,05:** H1 I(Identität) > I(Haar) und > I(Suche) (Wilcoxon einseitig exakt, p = max). H2 schussbewusst − bester exakter Kopf > 0 bei S = 10, 30, 100 (Mittel über 5 Arme, p = max). H3 Spearman η ~ S90 über 5 Arme, ρ < 0, exakter Permutationstest. **H4 Zielkonflikt:** Haar² exakt > Identität exakt (acc_inf) UND I(Identität) > I(Haar), beide gepaart einseitig, Schnitt-Vereinigungs-Test (p = max, beide Mittel > 0). Der Informationsteil von H4 ist dieselbe Differenz wie H1-Haar (offengelegt; Holm bleibt gültig, nur konservativ).
- **Kontrollen:** PK-1 relativ (s. o.), PK-2 (id, exakter Kopf, S=10 mind. 10 pp unter S=1000), NK (I < 0,03 Bit, Genauigkeit ≤ 0,15). NK fällt: alle HU. PK-1/PK-2 fällt: H2, H3 HU; H4 HU bei NK- oder PK-1-Versagen. Instabilität (≥ 2 Seeds < Median − 5 pp, bewusst S=100): H2/H3 HU.
- **Urteile (Strings):** H1 IDENTITAET_BESTE / H0 NICHT_UEBERLEGEN / HU NICHT_ENTSCHIEDEN; H2 SCHUSSBEWUSST_BESSER / H0 EXAKT_BESSER / H0 KEIN_VORTEIL / HU NICHT_ENTSCHIEDEN / INSTABIL / PK_VERSAGT; H3 ETA_SAGT_VORAUS / H0 ETA_NICHT_PROGNOSTISCH / HU ZENSIERT / NICHT_ENTSCHIEDEN / INSTABIL / PK_VERSAGT; H4 ZIELKONFLIKT / H0 KEIN_KONFLIKT_GENAUIGKEIT / H0 KEIN_KONFLIKT_INFORMATION / HU NICHT_ENTSCHIEDEN / PK_VERSAGT. Kriterien: `urteil()`.
- **Erwartung:** PK-1 besteht (≈ 0,90 gegen ≈ 0,91). H1 ja, H4 ja (Haar² ≈ +3,7 pp nach I-038, I ≈ −0,2 Bit nach I-040). H2 bei S = 10/30 wahrscheinlich, bei S = 100 knapp, gegen den stärkeren Gegner schwerer als in I-040. H3 offen bis eher nein.
- **Grenzen:** wie I-040; η-Neuheit ungeprüft (E4); die 2-pp-Schwelle ist eine begründete Setzung.
- **Start:** `INSTITUT_PROZESSE=4 python direktor.py lauf I-040-v2` (Wiederaufnahme je (Seed, Arm) in `ergebnisse/_zwischen_I-040-v2/`). Laufzeit ≈ 25 min bei 4 Prozessen.
