# Runde 1, Prüfung (02.11.)
- **W5:** Weiter gehen Idee 1 (Rotation gegen Clipping bei gleichen Bytes, Skalenbits liegen in der Flip-Fläche) und Idee 2 (fp16-Skalen-Flips, wenige Skalen gegen g16). Erwartet wird ein Gleichstand mit Clipping; ein echter Befund wäre am ehesten ein Bytevorteil gegenüber g16. Idee 4 ist Reserve.
- **W6:** Weiter gehen Idee 2 (Orbit-Lloyd, Idee 1 ist ein Spezialfall davon) und Idee 3 (Kerdock-Winkelbuch, 1-Minuten-Kontrolle). Gegner sind Lloyd-1D/2D, E8 nachgebaut (nicht die Papierzahlen) und Gruppenskala; alle Bits zählen inklusive Skalen und Codebuch.
- **Vor einem Zitat prüfen:** Farvardin, Slepian-Gruppencodes und Pseudo-Gray stehen nicht in `quellen.json`. Außerdem prüfen: reale 3-Qubit-Clifford ⊂ W(E8).
- **Vortest:** nur Tuning-Seeds, ≤ 30 min CPU, keine Mess-Seeds.

## Bewertung der Vortests (Erkundung, kein Befund)
- **W5:** Bei 4 Bit sind Rotation und Clipping gleichauf, bei 2 Bit ist Clipping besser. g16 ist gegen Bitflips anfällig, aber wegen der vielen fp16-Skalen, nicht wegen fehlender Rotation. Langweiligste Erklärung: „wenige Skalen“ reicht, die Rotation trägt nichts Eigenes bei. W5 wird nicht vorregistriert; es bleibt eine Randnotiz (Skalen sind die Angriffsfläche) für die Arbeit.
- **W6:** Orbit-Lloyd 8D (Clifford-artige Gruppe D8³) liegt bei +1,24 dB gegen Lloyd-1D, der E8-Nachbau bei +1,13 dB. Das heißt: Ein Codebuch aus Quantenstruktur ist gleichwertig mit E8 (+0,12 dB, Std ≤ 0,03). Das Kerdock-Winkelbuch ist tot. Die reale Clifford-Gruppe liegt **nicht** in W(E8).
  - Langweiligste Erklärung: Jede gute 8D-VQ landet bei ≈ +1,1–1,2 dB.
  - Offen: Dekodierkosten (Brute-Force), echte Gewichte, Skalen eingerechnet, Gegner E8P.
- **Vorschlag:** W6 kommt in den nächsten Schritt: Vortest auf echten Gewichten (Byte-LM, PTQ) mit Bytes inkl. Skalen und gemessener Dekodierzeit. Danach eventuell Vorregistrierung (Freigabe Malte).
