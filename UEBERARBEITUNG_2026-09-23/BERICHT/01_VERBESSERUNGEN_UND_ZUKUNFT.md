# Verbesserungen und Zukunft — priorisierte Liste

Stand 23. September 2026. Die Begründungen stehen in `00_REVIEW_BEFUNDE.md`; was die
Agenten umgesetzt haben, steht in den übrigen Dateien dieses Ordners.

Legende: **[U]** in dieser Überarbeitung umgesetzt (Details im jeweiligen Agentenbericht) ·
**[M]** Entscheidung von Malte nötig · **[R]** braucht Rechenzeit

---

## A. Muss vor der Anmeldung (30. November)

1. **Kernaussage fair nachmessen (Q-020)** [U: gebaut und vorregistriert] [M] [R]
   Die Kontrolle mit 67.848 Parametern ist schwächer als ein lineares Modell. Neue Kontrollen:
   ReLU-Netz, lineares Modell, klassische Kronecker-Drehung. Zusätzlich Verschränkung und Tiefe
   ohne Encoder nachmessen.
2. **Verschränkung/Tiefe nicht mehr als „belegt“ führen** [U: Dokumente]
   Die Messung lief mit trainierbarem Encoder, und die Positivkontrolle ist durchgefallen.
3. **Kurzfassung neu schreiben** [U: `A/KURZFASSUNG_ENTWURF_v2.md`] [M]
   Sie enthält veraltete und vermischte Zahlen.
4. **T-002 reparieren, bevor GPU-Zeit fließt** [U] [M: Schrittzahl, Zusatzarm]
   Betrifft Lernratensuche, Wiederaufnahme, Schrittzahl und das Feld `lr`.
5. **B-014-Korrektur in den Code** [U]
   Außerdem das B-Audit um die tragenden Befunde erweitern.
6. **IBM-Token erneuern** [M]
   Die Datei liegt im Klartext im Archiv, und der PC war infiziert. Danach die Datei löschen.
7. **Git einführen** [M]
   Ein Commit-Zeitstempel vor den Rohdaten ist der härteste Beweis für eine Vorregistrierung.

## B. Sollte bis zur schriftlichen Arbeit (Mitte Januar)

8. **T-002 auf dem MPI-Rechner** [R]
   `butterfly_qkv` ist der realistische „Quanten-Chatbot“.
9. **Q-011 zu Ende rechnen** [U: Wiederaufnahme, Gitter in beide Richtungen] [R]
   Das ist der aussichtsreichste Kandidat für einen bestandenen Abgrenzungstest.
10. **Q-019 umbauen oder ergänzen** [M]
    Die Primärhypothese ist fast per Konstruktion wahr. Vorschläge:
    - eine Lernkurve über die Datenmenge,
    - eine Walsh-Hadamard-Aufgabe (siehe C3).
11. **Literatur einarbeiten** [U: `A/LITERATUR_UND_ALLEINSTELLUNG.md`]
    Butterfly/Monarch, EUNN, BOFT, QuanTA, Bowles 2024, Thanasilp 2024, Neural Collapse.
12. **Hardware-Rohdaten sichern** [M]
    Die Counts der IBM-Jobs über die Job-IDs herunterladen und ablegen.
13. **Offiziellen MNIST-Testsatz benutzen**
    Bisher sind es 1.200 zufällige Testbilder je Seed. Mit den 10.000 offiziellen werden die
    Zahlen mit der Literatur vergleichbar.
14. **B-014 absichern**
    - mehrere Rangmaße,
    - Minimum-Modell gegen Alternativen,
    - CNN- und Mehrschicht-Lauf (für die Überschrift „nicht von der Architektur“).
15. **Dokumente entschlacken** [U: Lesereihenfolge, historische Dokumente markiert]

## C. Zukunft — wo ein echtes Alleinstellungsmerkmal entstehen kann

Nach Erfolgsaussicht geordnet. Die Einordnung in die Literatur steht in
`A/LITERATUR_UND_ALLEINSTELLUNG.md`.

### C1. Die Schaltung als sparsamer Adapter für vortrainierte Sprachmodelle
Das ist der stärkste Hebel für „ressourceneffiziente KI“.

**Idee:** Ein fertig trainiertes kleines deutsches Sprachmodell nachtrainieren. Dabei werden
nur die Winkel einer geteilten Butterfly-Drehung gelernt, die vor bzw. hinter die vorhandenen
Gewichte geschaltet wird.

**Vergleich:** gegen LoRA und BOFT bei *gleicher* Parameterzahl (fair).

**Warum das passt:**
- Es löst den Faktor-64-Konflikt: Man muss kein großes Modell mehr selbst trainieren, nur anpassen.
- Aus dem Chatbot wird ein echter Chatbot, wenn auf Dialogen nachtrainiert wird.
- BOFT (ICLR 2024) zeigt, dass orthogonale Butterfly-Adapter funktionieren.
- Die geteilten Winkel von QIRE brauchen noch einmal viel weniger Parameter. Genau das ist die offene Frage.

**Kosten:** GPU (MPI).

### C2. Ein beweisbarer Quantenvorteil — mit Quantendaten
Der einzige Weg zu einem *echten* Quantenvorteil.

**Hintergrund:** Huang et al. (Science 2022) zeigen: Mit Quantengedächtnis braucht man
exponentiell weniger Experimente.

**Versuch:** Ein kleiner Fall ist simulierbar und auf IBM-Hardware lauffähig. Beispiel:
- zwei Kopien eines Zustands per SWAP-Test vergleichen,
- „fast rein“ gegen „maximal gemischt“ unterscheiden,
- mit Quantengedächtnis genügen O(1) Kopien, ohne braucht man Ω(2^(n/2)).

**Warum das passt:** Dasselbe exponentielle Wachstum, das die sieben Negativbefunde erklärt,
erzeugt hier den Vorteil. Das ist eine elegante Klammer für die Arbeit.

**Stand:** Diese Idee stand am 4. August schon einmal im Raum und ist in keinem Dokument angekommen.

### C3. Walsh-Hadamard statt Fourier — die Aufgabe, für die die Struktur gebaut ist
**Beobachtung:** Ohne CNOT ist die geteilte Butterfly ein Kronecker-Produkt von
2×2-Drehungen. Darin ist die Walsh-Hadamard-Transformation *exakt* enthalten.

**Vorschlag:** Eine Signalaufgabe in Walsh-Bändern (Rechteck- bzw. Schaltsignale).
- **Vorhersage:** Die Struktur löst sie mit 8–40 Parametern, eine dichte Schicht braucht 65.536
  und viele Daten.
- **Kontrolle:** dieselbe Aufgabe in Fourier-Bändern und permutiert.

**Nutzen:** Das wäre ein „inhaltlicher Vorteil“ mit sauberer Begründung, statt der
Fourier-Erzählung, die Q-019 bereits widerlegt hat.

### C4. Wenig Daten statt viel Daten
40 Parameter gegen 65.536 ist vor allem ein Argument für **Dateneffizienz**.

**Vorschlag:** Lernkurven für N = 50 bis 6.000. Vorregistrierte Vorhersage: Die Struktur gewinnt
unterhalb einer bestimmten Datenmenge.

**Bezug:** Das ist auch „ressourceneffizient“ (Datenbedarf) und passt zu B-010 (Blockdiagonale
gewinnt bei 200 Beispielen).

### C5. Photonische und neuromorphe Hardware (Brücke zum MPI)
**Beobachtung:** Photonische Chips rechnen unitäre Matrizen als Netze aus 2×2-Drehungen
(Mach-Zehnder-Interferometer). Eine geteilte Butterfly braucht **eine Einstellung pro Ebene
statt eine pro Bauteil**. Das bedeutet weniger Steuerleitungen, weniger Kalibrierung und
weniger Energie.

**Stand:** Das ist nicht geprüft, aber eine konkrete Frage an Physiker, die solche Chips bauen.

**Mögliche Messung:** Robustheit gegen Fertigungsungenauigkeit der Winkel (Rauschen auf θ)
im Vergleich zu einem ungeteilten Netz.

### C6. Fusionierter Kernel (GPU oder CPU)
**Ziel:** Faktor ~3, um den Übergangspunkt von 16.384 auf 4.096 zu schieben (T-001).

**Vorbilder:** FlashFFTConv und die Monarch-Kernel.

**Stand:** Das braucht eine eigene Vorregistrierung und viel Programmierarbeit. Für Jugend
forscht nur, wenn C1–C4 fertig sind.

### C7. Methodik als eigenständiger Beitrag
**Idee:** Die Abgrenzungstests, der Vortest auf künstlichen Daten und `q_audit` als kleines,
wiederverwendbares Prüfpaket für „quanteninspirierte“ Behauptungen.

**Bezug:** Bowles/Ahmed/Schuld (2024) zeigen, dass der Bedarf dafür in der Forschung real ist.

**Aufwand:** gering (Aufräumen und Dokumentieren), Wirkung für die Jury hoch.

---

## D. Was ich an Maltes Stelle als Linie für die Arbeit wählen würde

**Titelidee:** „Wie viel Quantenphysik steckt in quanteninspirierter KI? Sieben
Abgrenzungstests, eine gemeinsame Ursache — und eine Schaltung, die mit 16 Parametern leistet,
wofür klassisch Tausende nötig sind.“

- Der letzte Halbsatz steht nur, wenn Q-020 ihn bestätigt.
- Sonst lautet er: „… und was davon klassisch bleibt.“
- C1 oder C2 wäre das Zukunftskapitel. Mit MPI-Rechenzeit ist C1 bis Januar realistisch.
