# Runde 1, Prüfung (02.11.)
- **W5:** Weiter gehen Idee 1 (Rotation gegen Clipping bei gleichen Bytes, Skalenbits liegen in der Flip-Fläche) und Idee 2 (fp16-Skalen-Flips, wenige Skalen gegen g16). Erwartet wird ein Gleichstand mit Clipping; ein echter Befund wäre am ehesten ein Bytevorteil gegenüber g16. Idee 4 ist Reserve.
- **W6:** Weiter gehen Idee 2 (Orbit-Lloyd, Idee 1 ist ein Spezialfall davon) und Idee 3 (Kerdock-Winkelbuch, 1-Minuten-Kontrolle). Gegner sind Lloyd-1D/2D, E8 nachgebaut (nicht die Papierzahlen) und Gruppenskala; alle Bits zählen inklusive Skalen und Codebuch.
- **Vor einem Zitat prüfen:** Farvardin, Slepian-Gruppencodes und Pseudo-Gray stehen nicht in `quellen.json`. Außerdem prüfen: reale 3-Qubit-Clifford ⊂ W(E8).
- **Vortest:** nur Tuning-Seeds, ≤ 30 min CPU, keine Mess-Seeds.
