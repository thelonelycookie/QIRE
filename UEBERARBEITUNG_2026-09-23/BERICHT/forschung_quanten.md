# Kurzbericht: Agent `forschung_quanten`

Vollständiger Katalog: `FORSCHUNG/QUANTEN_IDEENKATALOG.md` (15 Ideenkarten,
Rangliste, Zusammenhänge-Karte). Grundlage: Review-Dokumente vom 23./24.
September, `FORSCHUNGSIDEEN.md` und `LITERATUR_UND_ALLEINSTELLUNG.md` (nicht
wiederholt, nur ergänzt), plus eigene Websuche mit belegten Quellen.

## Was neu dazukommt

Neun vorgegebene Forschungsfelder wurden geprüft; in sechs davon stand schon
etwas im Projekt, dort wurde ergänzt statt wiederholt. Neue Bausteine kamen
vor allem aus: Tensornetz-Kompression jenseits von Butterfly (MPO/CompactifAI,
MERA), einer feineren Abstufung zwischen „keine Verschränkung" und „volle
Universalität" (Matchgates, feste t-Design-Mischung), weiteren klassischen
Struktur-Kontrollen (zirkulante Matrizen aus der Post-Quanten-Kryptographie,
Gruppenmatrizen), und Methodik-Bausteinen (Dequantisierungs-Checkliste,
QUBO-Pruning-Literatur für Q-006).

Ehrlich negativ geprüft: Quantenschlüsselaustausch (BB84/QKD) und
Quanten-Zufallszahlen tragen zur Ressourceneffizienz nichts bei (belegt durch
eine Studie ohne messbaren Unterschied bei der Netzinitialisierung).
Stabilizer-Codes als Vorlage für KI-Architektur: keine passende Arbeit
gefunden, nur die umgekehrte Richtung (KI hilft Fehlerkorrektur).

## Top 5 (nach Erfolgsaussicht × Wirkung / Aufwand)

1. **Feste unitäre t-Design-Mischung statt CNOT-Ring** — 1 Tag, testet direkt
   und unabhängig, ob überhaupt irgendeine feste Mischung etwas bringt oder
   speziell CNOT nichts beiträgt.
2. **Encoder als Matrix-Product-Operator (MPO), Prinzip von CompactifAI** —
   3–4 Tage, konkurriert mit der bestehenden „Encoder als Butterfly"-Idee,
   könnte das 848×-nur-in-der-Mitte-Problem für das Gesamtmodell lösen.
3. **Gruppenmatrix-Kontrollarm** — 2 Tage, unabhängiger externer Test der
   eigenen, schon zweifach geprüften Theorie „Parameterzahl zählt,
   Parametrisierung nicht".
4. **Zirkulante Matrix (Ring-LWE-Struktur) als weitere Q-020-Kontrolle** —
   0,5 Tage, macht die faire-Kontrollen-Batterie vollständiger, ohne
   Quantenbezug zu beanspruchen.
5. **Dequantisierungs-Checkliste als Methodik-Baustein** — praktisch
   kostenlos, stärkt den schon erkannten USP „vorregistrierte
   Abgrenzungsmethodik".

Kein Punkt hier verspricht ein neues physikalisches Phänomen; wo eine Idee
nur eine mathematische Analogie zur Quantenmechanik ist (z. B. Idee 10,
Hyperdimensional Computing), steht das im Katalog ausdrücklich so.
