# Ideenbörse des Instituts

Stand 24.09.2026. Nur Ideen, die **nicht** schon in
`UEBERARBEITUNG_2026-09-23/FORSCHUNG/QUANTEN_IDEENKATALOG.md` oder
`EIGENES_MODELL_KONZEPT.md` stehen. Jede Idee hat: Quantenwurzel,
Papierprüfung, Neuheitsprüfung (Websuche, nur Treffer­listen, kein Volltext),
billigstes entscheidendes Experiment, Abbruchkriterium.

Bewertung: **Aussicht** (dass es trägt) × **Neuheit** (dass es noch keiner so
gemacht hat), je niedrig/mittel/hoch.

---

## 1. Magie-Budget und Clifford-Rundung — *in Arbeit* (`C-clifford`)

**Quantenwurzel.** Ressourcentheorie der Magie: Clifford-Schaltungen sind
klassisch billig (Gottesman–Knill), die Kosten stecken allein in den
Nicht-Clifford-Gattern (T-Zahl, Stabilisator-Rényi-Entropie).

**Übertragung.** Eine reelle Ry-Drehung mit Winkel k·π/2 ist eine
Vorzeichen-Permutation oder Hadamard-artig (Einträge ±1/√2). Kronecker-Produkt
+ CNOT-Ring aus solchen Drehungen = Matrix mit Einträgen nur aus {0, ±c} →
**keine Multiplikation**, nur Additionen (geprüft: `unitaer()` liefert auf dem
Gitter genau {−1, 0, 1}·c). Die Frage „wie viel Magie braucht eine Aufgabe?“
wird zur messbaren Größe **M\*** (kleinste Zahl freier Nicht-Clifford-Winkel
für ≤ 2 pp Verlust).

**Neuheit.** Gefunden: Hadamard-Drehungen *vor* der Quantisierung (QuaRot,
PolarQuant, arXiv:2603.29078) und Magie-Schätzung für Quantenzustände
(Graph-NN-Schätzer, IOP 2026). **Nicht gefunden:** Magie als Kompressionsmaß
für trainierte KI-Schichten, Clifford-Rundung der Winkel. → Neuheit mittel–hoch.

**Abbruch.** Wenn gerundete Schaltung nicht besser als eine *zufällige*
Clifford-Schaltung ist (C-2: H0), hat das Training nichts gelernt, was die
Rundung überlebt → auf schwererer Aufgabe prüfen, bevor verworfen wird.

## 2. Zero-Noise-Extrapolation über Bitbreiten — *auf Papier verworfen* (`Z-zne`)

Idee aus der Quanten-Fehlermitigation (Temme/Bravyi/Gambetta 2017):
Ergebnisse bei zwei Rauschstärken rechnen und auf Rauschen null
extrapolieren. Übertragung: Gewichte mit b und b−1 Bit (verschachtelt wie
bei Matryoshka Quantization, arXiv:2502.06786) → Richardson auf die Logits.
Neuheit wäre gegeben (nicht gefunden), **aber die Rechnung widerlegt es** —
siehe `LABORBUCH.md` 24.09. Das Ergebnis wird 7× schlechter, nicht besser.
Rest-Idee, die übrig bleibt: nur für *echtes, zufälliges* Rauschen (analoge
Rechenwerke, photonische Chips) sinnvoll → als Frage fürs MPI-Gespräch.

## 3. Verschränkungs-Kompass: Qubit-Ordnung wie in der Quantenchemie (`V-kompass`)

**Quantenwurzel.** In der DMRG-Quantenchemie hängt die Güte eines
Matrixprodukts dramatisch von der **Reihenfolge der Orbitale** ab; man ordnet
sie nach gegenseitiger Information (Legeza & Sólyom 2003; Rissler, Noack,
White 2006), sodass stark verschränkte Orbitale nebeneinander liegen.

**Übertragung.** Eine trainierte Gewichtsmatrix 256×256 = Operator auf
8+8 Qubits. Vor der Tensor-Train-Kompression die Qubits nach
Mutual-Information-Matrix (aus den Operator-Schmidt-Spektren aller
Zweierschnitte) ordnen → bei gleicher Parameterzahl kleinerer Fehler.
Passt zu B-014 (Rang wird von der Aufgabe gesetzt) und zu CompactifAI
(Katalog Idee 2), das die Reihenfolge fest lässt.

**Neuheit.** Tensornetz-Struktursuche existiert (Permutationssuche,
Li u. a. ICML 2022), dort per Stichprobensuche, nicht per
Mutual-Information-Ordnung aus der Quantenchemie. Neuheit mittel, Aussicht
mittel. **Billigster Test:** 20 min CPU — trainierte MLP-Gewichte, drei
Ordnungen (Standard, zufällig, MI), Fehler und Genauigkeit bei gleichem
Parameterbudget. **Abbruch:** MI-Ordnung nicht besser als zufällige.

## 4. Magie-Skalierungsgesetz (`C-skala`)

Wenn M\* mit der Breite *langsamer* wächst als die Winkelzahl n·L, wird eine
breite Schicht fast vollständig multiplikationsfrei. Das wäre die
Effizienzaussage, die bei 16.384 Dimensionen zählt (Projekt A: Struktur
gewinnt erst dort). Nur sinnvoll, wenn Idee 1 trägt. Aussicht unbekannt,
Neuheit hoch.

## 5. Paritäts-Auslese aus einer Messbasis (`B-paritaet`)

Alle Erwartungswerte ⟨Z_S⟩ (Produkte beliebiger Z) kommen aus **einer
einzigen** Messung in der Rechenbasis: sie sind die Walsh-Hadamard-
Transformation der Born-Verteilung. Projekt A liest nur n = 8 von 256
Zahlen aus (Review: „nur 3 tragen Bandinformation“). Paritäten bis Ordnung 2
geben 1 + n + n(n−1)/2 = 37 Merkmale ohne zusätzliche Schaltungsläufe — auf
Hardware kostenlos, im Simulator eine schnelle WHT. Ehrlich: im Simulator ist
die volle Born-Verteilung ohnehin da; die Frage ist nur, ob niedrige
Paritäten die *sparsamste* Auswahl sind. Aussicht mittel, Neuheit niedrig–mittel.

## 6. Übernehmen statt erfinden: Tensorprodukt-Einbettung für den Chatbot

Bei einem kleinen Sprachmodell (Vokabular 4.096 × Breite 256) ist die
Einbettung **eine Million Parameter** — so viel wie mehrere Blöcke.
word2ket (Panahi u. a., ICLR 2020) bzw. TT-Embeddings (Hrinchuk u. a. 2020)
schreiben die Einbettung als Summe von Tensorprodukten („verschränkte
Wortvektoren“), mit 10–100× weniger Parametern. **Bekannt**, aber für T-002
direkt nutzbar und klar quanteninspiriert. Gegen eine gewöhnliche
Low-Rank-Einbettung gleicher Größe prüfen.

---

## Geprüft und bewusst nicht aufgenommen

- **Quanten-Zeno-Frühausstieg** (Messung stoppt die Rechnung): ist
  inhaltlich identisch mit bekannten Early-Exit-Netzen, nur umbenannt.
- **Lieb-Robinson-Lichtkegel** (nur Kausalkegel rechnen): bei Tensornetzen
  (MERA) seit Jahren Standard, kein neuer Hebel für QIRE.
- **Quanten-Annealing-Training**: als „robuste Ensembles“/Entropy-SGD
  (Baldassi u. a. 2016) bekannt.
