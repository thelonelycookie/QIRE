# Was nicht in diesem Ordner liegt — und warum

Der Ordner ist mit **~8 MB** absichtlich klein gehalten. Zwei Dinge sind
draußen geblieben. Beide sind ersetzbar, aber nicht kostenlos.

**Archivordner:**

```
C:/Users/Malte/Downloads/qire_ai_console_v1_0_remote_research_platform/qire_ai_console_v1_0/
```

> **Diesen Ordner bitte nicht löschen, ohne diese Datei gelesen zu haben.**
> Die Systemplatte ist zu 99 % voll (29 GB frei von 1,9 TB), das ist vermutlich
> der Grund, warum man auf die Idee kommen könnte.

---

## 1. Der deutsche Korpus — 2,9 GB

Liegt unter `results/korpus/` **im Archivordner**:

| Datei | Größe | Was es ist | Ersetzbar? |
|---|---|---|---|
| `de_0.parquet` | 781 MB | Rohdownload, deutsche Wikipedia 20231101 | ja, Download |
| `de_text.txt` | 1,3 GB | daraus extrahierter Reintext | ja, Zwischenschritt |
| `bpe_train.txt` | 211 MB | Ausschnitt zum Trainieren des Tokenizers | ja, Zwischenschritt |
| **`train.npy`** | **795 MB** | **die Trainingstoken, 397,4 Mio.** | ja, aber teuer |

**Mitgenommen** habe ich die beiden kleinen, die die Reproduzierbarkeit
festlegen — sie liegen hier unter `results/korpus/`:

- `bpe_4096.json` (262 KB) — der Tokenizer. **Ohne ihn ist `train.npy` wertlos**,
  weil die Token-IDs sonst nicht mehr dieselben Wörter bedeuten.
- `val.npy` (4 MB) — die Validierungsmenge, auf der alle Verlustwerte gemessen
  werden.

**Neu erzeugen:**

```bash
"$VENV" -m quantum_sim.korpus_deutsch pruefen      # nur Größen anzeigen
"$VENV" -m quantum_sim.korpus_deutsch laden        # Shard herunterladen
"$VENV" -m quantum_sim.korpus_deutsch aufbereiten  # Text + BPE + Token
```

Das kostet Download plus etwa eine Stunde Aufbereitung, und **es erzeugt einen
anderen Tokenizer**, wenn `bpe_4096.json` nicht vorher an seinen Platz gelegt
wird. Also: erst die Datei von hier nach `results/korpus/` kopieren, dann
aufbereiten.

**Für T-002 auf dem Server ist das ohnehin gleichgültig** — `runpod_start.sh`
baut den Korpus auf dem Pod neu. Lokal wird er nur gebraucht, wenn das
Sprachmodell auf diesem Rechner laufen soll, und dafür ist der Rechner zu klein.

---

## 2. Projekt B — Rohdaten, Abbildungen, Modelle

Ebenfalls im Archivordner:

- `results/experiments/B-*.json` und `.csv` — 49 Dateien, ~2,5 MB
- `results/figures/` — 2,9 MB, die Abbildungen der B-Serie
- `results/datasets/` — 71 MB MNIST / Fashion-MNIST (lädt torchvision bei Bedarf neu)
- `results/modelle/` — 33 MB, `dense.pt` und `narrow.pt` aus dem T-002-Vorlauf

Projekt B ist **abgeschlossen**; seine Zahlen werden hier nur zitiert, nicht
neu gerechnet. Der Code dafür (`qire_ai/`) liegt vollständig in diesem Ordner,
weil `quantum_sim` sieben Module daraus importiert — nur die Rohdaten fehlen.

`qire_ai/audit.py` läuft deshalb hier **nicht** durch. Für die B-Serie im
Archivordner arbeiten oder die B-Dateien nachkopieren:

```bash
cp <archiv>/results/experiments/B-*.json <archiv>/results/experiments/B-*.csv results/experiments/
```

---

## 3. Was mitgekommen ist

- alle Q- und T-Rohdaten (55 Dateien), einschließlich der **verworfenen**
  Q-011-Vorläufe unter `results/experiments/vorlaeufe_q011/`
- alle Lauf-Logs der Q- und T-Serie (23 Dateien)
- der vollständige Code beider Pakete
- alle Dokumente
- die Serverskripte
- die acht Literaturkarten unter `sources/literature_cards/`
