# Ideen-Loop (ab 28.10.2026, Auftrag Malte: konstant neue Ideen prüfen)

Ziel: quanten-inspirierte, ressourceneffiziente KI. Ein Vorteil zählt nur bei gleichen FLOPs oder gleichen Bytes, mit genug Training (Frühregime-Regel: Rauchtests immer bei zwei Trainingslängen).

## Ein Zyklus (Z-n), etwa alle 3 h
1. **Ernten:** Raven-Ergebnisse (Erkundungspaket, I-048, I-045-v5) mit `erkundung/auswertung.py` auswerten. Ampeln und Brier in `zyklus/Z<n>_ernte.md`.
2. **Kombinieren:** Ein Agent kombiniert Grüne und Gelbe untereinander und mit bestehenden Systemen (BitNet, QuIP#, QuaRot, Mamba, MoE, LoRA, spekulatives Dekodieren, Distillation).
3. **Literatur:** Ein Agent liest gezielt zu den Grünen und Gelben (neueste Arbeiten, stärkste Baseline).
4. **Grenzen:** Ein Agent prüft die theoretische Grenze (Bits je Gewicht, Rate-Distortion, Skalengesetze für Präzision) und sucht Ideen jenseits davon.
5. **Killer und Turnier:** kurz, nur die Top 5 des Zyklus.
6. **Bauen:** Die Top 3–5 werden Erkundungsskripte `erkundung/Z<n>_*.py` mit den härtesten Gegenspielern und zwei Trainingslängen. Kurzlauf im Container, dann Push auf main und Nachauftrag an Raven.
7. **Vorregistrieren:** Was im Erkundungslauf grün ist, wird ein vorregistrierter Lauf. Malte hat am 28.10. „alles“ freigegeben; jeder neue Lauf wird ihm trotzdem in einer Zeile gemeldet.

Regeln: Tuning-Seeds 950–989 für Erkundung. Ergebnisse auf diesem Branch, Code auf main. Commits nur unter Maltes Namen.

## Änderungen ab Zyklus Z2 (aus Z1_kreativforschung.md)
1. **Archiv nach MAP-Elites:** Tabelle Quantenprinzip × Ressource in `zyklus/ARCHIV.md`. Leere Zellen werden zum Pflichtauftrag des nächsten Zyklus.
2. **Neuheitsprüfung vor dem Turnier:** Jede Idee wird mit Archiv und LABORBUCH abgeglichen. Duplikate fliegen raus, Neues bekommt einen Bonus.
3. **Evolution mit Prüfer:** Alle Rauchtests nutzen dasselbe Bewertungsmodul (gleiche FLOPs, gleiche Bytes, zwei Trainingslängen). Grüne Ideen werden im nächsten Zyklus mutiert.
4. **Pflichtfeld „langweiligste Erklärung“** samt passender Kontrolle. Fehlt das Feld, zählt der Rauchtest nicht.
5. **Kontingente:** mindestens ein Platz der Top 5 für Maltes ++, mindestens eine Fernanalogie aus einem fremden Fach je Zyklus. Mischung nach Uzzi: konventionelle Basis plus eine atypische Verbindung.
