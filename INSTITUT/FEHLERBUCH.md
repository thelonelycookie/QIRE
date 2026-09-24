# Fehlerbuch — was schiefging und was wir daraus lernen

Ein gescheiterter Ausgang wird nie gelöscht. Jeder Eintrag endet mit der
Lehre und der Folgefrage. Der Direktor trägt H0/HU-Urteile automatisch ein;
die Lehre schreibt ein Mensch (oder Claude) dazu.

---

### 24.09.2026 — Z-zne (auf Papier)
- **Was:** Zero-Noise-Extrapolation über verschachtelte Bitbreiten.
- **Warum gescheitert:** Voraussetzung „Fehler glatt in der Rauschstärke“
  gilt für Runden nicht. Extrapolation macht den Fehler 7× größer.
- **Lehre:** Bei jeder Übertragung aus der Physik die *Gültigkeitsbedingung*
  der Methode mit übertragen und zuerst prüfen.
- **Folgefrage:** nur bei echtem Zufallsrauschen (analoge Hardware) — geparkt.

### 24.09.2026 — Rauchtest C-clifford
- **Was:** Gatterweise Simulation war 0,1 s/Schritt → ganzer Lauf ~8 h.
- **Lehre:** Training in der schnellsten *äquivalenten* Form (ganze Matrix),
  Kosten der Inferenz getrennt und analytisch zählen. Gleiche Trennung wie in
  Projekt A (Qualität bei trainierbarer Breite, Effizienz bei eigener Breite).
