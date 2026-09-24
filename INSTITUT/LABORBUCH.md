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
