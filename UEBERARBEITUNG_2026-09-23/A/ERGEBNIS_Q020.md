# Q-020 — Ergebnis (gerechnet 07.10.2026, Cloud, 4 Kerne, 15 Seeds, 480 Messzeilen)

Rohdaten: `results/experiments/Q-020__20261007_221800.{json,csv}`. Vorregistrierung im Modulkopf
von `quantum_sim/kernaussage.py` (P4 am 07.10. vor jeder Messzahl ergänzt). Kontrollen: Boden und
Auflösung in beiden Aufbauten bestanden, Äquivalenzkontrolle kron ≈ q1_ohne bestanden (+0,0001 / +0,0003).

## Urteile (Holm über 4)
| | Urteil | Zahl |
|---|---|---|
| PRIMAER (a) MNIST: q1 gegen MLP | **SCHALTUNG_BESSER** | 0,9347 vs 0,9231, +1,16 pp, 15/15, Holm p = 0,0002 |
| P2 (b): Verschränkung | **VERSCHRAENKUNG_TRAEGT** | q5 0,687 vs q5_ohne 0,345, +34 pp |
| P3 (b): Tiefe | **TIEFE_TRAEGT** | q5 0,687 vs q1 0,355, +33 pp |
| P4 (b): Born-Auslese | **BORN_TRAEGT** | q5 0,687 vs q5_linear 0,631, +5,6 pp, 15/15 |
Wiederholung Fashion (beschreibend): q1 − MLP +0,45 pp (p = 0,013) — gleiche Richtung.

## Deutung (Opus) — was hält und was nicht
1. **Innerhalb der Schaltungsfamilie** sind jetzt alle drei Quantenzutaten belegt: Verschränkung,
   Tiefe und die Born-Regel. Die alten Nullbefunde (Q-003) waren Encoder-Artefakte. P4 ist die
   Abgrenzung zu Quantum-PEFT (lineare Auslese).
2. **Aber in Aufbau (b) verliert die Schaltung gegen jede klassische Kontrolle:** q5 0,687 gegen
   linear ≈ 0,91, MLP 1,00, quad_free 1,00, q5_ungeteilt 1,00. Mit 80 Parametern gegen 2.056+ ist das
   kein fairer Effizienzvergleich, aber es heißt: auf der Haufen-Aufgabe ist die Schaltung
   **kapazitätsbegrenzt**. Das darf nicht verschwiegen werden.
3. **Unitarität kostet:** q5_frei (frei statt orthogonal, 160 Parameter) +14 pp über q5. Ob das an
   der doppelten Parameterzahl oder an der Normerhaltung liegt, ist offen → Folgefrage (frei mit 80).
4. **PRIMAER** (mit Encoder): Die 8 wirksamen Winkel schlagen ein MLP mit 67.848 Parametern in der
   Engstelle. Die Äquivalenzkontrolle zeigt zugleich: q1 ist mathematisch eine klassische
   Kronecker-Drehung mit Paritäts-Auslese. Der Vorteil ist **quanten­abgeleitete Struktur, klassisch
   gerechnet** — genau die QIRE-These, ehrlich formuliert.
5. Ehrliche Kernaussage für die Arbeit: *Die Quantenstruktur (Verschränkung, Tiefe, Born-Regel)
   trägt jeweils messbar; als Engstelle hinter einem Encoder schlägt sie ein großes MLP mit einem
   Bruchteil der Parameter; ohne Encoder ist sie mit 80 Parametern zu klein für die schwere Aufgabe.*

## Folgefragen
- Pareto-Kurve (b): Schaltung mit mehr Lagen/Qubits gegen linear/MLP bei steigender Parameterzahl (V2).
- Unitarität bei gleicher Parameterzahl (q5_frei mit 80).
- Kette statt Ring (S1b/S1c) — könnte die Kapazitätsgrenze in (b) verschieben.
