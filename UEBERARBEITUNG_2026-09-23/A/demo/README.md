# Stand-Demo: Ziffernerkennung mit Quantenschaltung

Interaktive Demonstration für den Wettbewerbsstand. Besucher zeichnen eine
Ziffer und sehen sofort, was drei verschiedene Modelle daraus machen.

## Starten

```bash
python -m demo.demo_app
```

Läuft vollständig **offline** — kein Internet, keine Wartezeit, kein
IBM-Kontingent nötig.

Falls die Modelldatei fehlt, einmalig vorher:

```bash
python -m demo.train_demo_model
```

## Bedienung

| Eingabe | Wirkung |
|---|---|
| linke Maustaste | zeichnen |
| rechte Maustaste | radieren |
| Leertaste | Feld löschen |

## Was gezeigt wird

Drei Modelle nebeneinander, jeweils mit Vorhersage und
Wahrscheinlichkeitsverteilung über alle zehn Ziffern:

| Modell | Was es ist |
|---|---|
| **Quantenschaltung (Simulation)** | 8 Qubits, 256-dimensionaler Zustandsraum, 80 Parameter. Erreicht klassisches Niveau, ist aber wegen der aufwendigen Zustandspräparation nicht auf echter Hardware ausführbar. |
| **Quantenschaltung (hardwaretauglich)** | 10 Qubits, Winkel-Kodierung. Schwächer, dafür nachweislich auf `ibm_fez` gelaufen. |
| **Klassische Schicht** | Gleiche Breite, gleicher Encoder. Die Kontrollbedingung. |

Darunter die Belege der echten Hardware-Läufe mit Job-IDs.

## Warum diese drei

Die Gegenüberstellung **ist** die Aussage der Arbeit. Der Besucher sieht
unmittelbar:

1. In Simulation zieht die Quantenschaltung mit dem klassischen Modell mit —
   mit deutlich weniger Parametern.
2. Die hardwaretaugliche Variante fällt spürbar ab, weil die flache Kodierung
   weniger Information in die Schaltung bringt.
3. Genau das ist der Grund, warum Quanten-KI heute noch nicht praktisch ist.

Das ist ehrlicher und interessanter, als nur ein gut funktionierendes Modell zu
zeigen — und es lädt zur Nachfrage ein.

## Für Gespräche am Stand

**„Rechnet das gerade auf einem Quantencomputer?"**
Nein — am Stand rechnet die Simulation, damit die Antwort sofort kommt. Aber
dieselben Schaltungen liefen echt auf `ibm_fez`, einem IBM-Prozessor mit 156
Qubits. Die Job-IDs stehen unten und sind auf quantum.ibm.com nachprüfbar.

**„Warum ist das Quantenmodell schlechter?"**
Weil die hardwaretaugliche Kodierung flach sein muss. Jedes zusätzliche Gatter
bringt Rauschen: bei doppelter Schaltungstiefe bleiben statt 85 % nur noch
71 % der Signalstärke übrig — das habe ich selbst gemessen.

**„Was bringt der Quantenteil dann?"**
Bei der Genauigkeit nach meinen Messungen nichts. Ich habe sieben
quanteninspirierte Mechanismen gegen faire klassische Kontrollen getestet,
keiner hat gehalten. Was bleibt, ist die Parametereffizienz: 80 Winkel wirken
auf einen 256-dimensionalen Zustandsraum.

**„Ist die Zeichnung dieselbe wie beim Training?"**
Nein — die Modelle wurden auf MNIST trainiert, also auf zentrierten,
größennormierten Ziffern. Die Demo bildet diese Vorverarbeitung nach
(zuschneiden, auf 20×20 skalieren, zentrieren). Frei gezeichnete Ziffern sind
trotzdem schwieriger als MNIST-Testbilder.

## Dateien

| Datei | Zweck |
|---|---|
| `demo_app.py` | die Oberfläche |
| `train_demo_model.py` | trainiert und speichert die drei Modelle |
| `artifacts/demo_models.pt` | gespeicherte Gewichte |
| `artifacts/hardware_evidence.json` | Belege der echten Hardware-Läufe |
