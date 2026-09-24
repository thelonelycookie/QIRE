# T-002 auf einem SLURM-Cluster (z. B. MPI für Mikrostrukturphysik)

Stand 24. September 2026. Alles hier ist für einen Cluster geschrieben, den wir
**noch nicht kennen**. Jede Annahme ist als **ANNAHME** markiert und muss vor dem
ersten Job geprüft werden. Nichts davon ist auf einem echten SLURM-System
getestet — lokal geprüft sind nur die Python-Seite (Selbsttest, Rauchtest auf
CPU) und die Syntax der Shellskripte (`bash -n`).

## 1. Was gerechnet wird

T-002 vergleicht vier Sprachmodell-Arme (`dense`, `butterfly_ffn`,
`butterfly_qkv`, `narrow`) auf deutschem Wikipedia-Text, 8 Seeds je Arm,
**13.000 Gradientenschritte je Lauf** (Konstante `SCHRITTE` in
`quantum_sim/sprachmodell.py`, vorregistriert). Vorher eine Lernratensuche je
Arm mit **demselben Budget** (Regel 1c), Gitter 3e-4 … 2,4e-3, bei Randlage bis
zu dreimal verlängert, nach oben *oder* unten (Regel 1b).

Jede Einheit ist eine **Zelle** = ein Trainingslauf auf **einer GPU**:

| Zelle | Anzahl | hängt ab von |
|---|---|---|
| LR-Startgitter (Arm × 4 Raten × 1 Tuning-Seed) | 16 | nichts |
| LR-Verlängerung, bis zu 3 Runden (≤ 1 Zelle je Arm und Runde) | 0–12 | vorige Runde |
| Messzelle (Arm × 8 Seeds) | 32 | fertige Lernratenkarte |

Jede Zelle schreibt eine eigene JSON-Datei. Nichts wird überschrieben; eine
doppelt gerechnete Zelle landet daneben und wird beim Zusammenführen als
Duplikat erkannt (bei abweichendem Wert gilt die ältere Messung).

## 2. Rechenbedarf — SCHÄTZUNG

Grundlage: **eine** Messung, `butterfly_ffn` auf einer **RTX 4090**, Batch 64:
0,334 s je Schritt (11. August). Für **alle** Arme angesetzt — `dense` und
`narrow` sind vermutlich schneller, `butterfly_qkv` ist ungemessen. Die Zahlen
sind deshalb eher eine **Obergrenze**. Auswertung (~2,7 %) ist über eine
Annahme (Vorwärtslauf ≈ ⅓ Trainingsschritt) eingerechnet. Erzeugt mit
`python -m quantum_sim.sprachmodell rechenbedarf`.

**Minuten je Zelle: ~74** (auf RTX 4090; andere Karten nicht gemessen)

| Phase | Zellen | GPU-Stunden |
|---|---:|---:|
| LR-Suche Startgitter | 16 | 19,8 |
| LR-Verlängerungen (höchstens) | 0–12 | 0–14,9 |
| Messlauf | 32 | 39,6 |
| **Summe** | **48–60** | **59,4–74,3** |

Wanduhrzeit (die LR-Runden laufen nacheinander, der Messlauf erst danach):

| GPUs parallel | ohne Verlängerung | mit 3 Verlängerungen |
|---:|---:|---:|
| 1 | 59,4 h | 74,3 h |
| 4 | 14,9 h | 18,6 h |
| 8 | 7,4 h | 11,1 h |
| 16 | 3,7 h | 7,4 h |
| 32 | 2,5 h | 6,2 h |

Mit dem (nicht vorregistrierten) Zusatzarm `butterfly_ffn_zentriert`:
60–75 Zellen, 74,3–92,9 GPU-Stunden; 16 GPUs: 6,2–9,9 h.

Zum Einordnen der Review-Zahl „~30–35 GPU-h“: Die galt für die alte, zu kleine
Lernratensuche und für schnellere dichte Arme. Mit vollem Suchbudget ist die
Suche so teuer wie ein halber Messlauf.

**Welche GPU?** Laut Messung vom 11. August ist `butterfly_ffn` auf der GPU
*startaufwandbegrenzt* (Kernel-Starts, nicht Rechenarbeit). Eine schnellere
Karte (A100/H100) hilft daher vermutlich wenig, **mehr parallele Karten helfen
linear**. Ungeprüfte Idee: mehrere Zellen auf einer Karte gleichzeitig (MPS);
Speicherbedarf einer Zelle bei Batch 64 ist nicht gemessen, bei Batch 256
waren es 18,3 GB.

## 3. Fragen fürs MPI

1. SLURM? Welche Partition, welcher Account, wie fordert man eine GPU an
   (`--gres=gpu:1` oder `--gpus=1`)? Welches Zeitlimit je Job?
2. Welche GPUs, wie viele gleichzeitig für uns? (Tabelle oben: 8 Karten ≈
   ein halber Tag, 16 Karten ≈ ein Abend.)
3. Haben Rechenknoten Internet? (**ANNAHME: nein** — deshalb wird alles
   mitgebracht, siehe 4.)
4. Gemeinsames Dateisystem zwischen Login- und Rechenknoten, und wo darf ~1 GB
   liegen?
5. Python: gibt es ein Modul/conda mit PyTorch + CUDA, oder eigene venv?
6. Darf ein Job selbst `sbatch` aufrufen? (Nur für `auto` nötig.)
7. Wie kommen Ergebnisse zurück (scp/rsync vom Login-Knoten)?

## 4. Was mitgebracht werden muss

| Was | Woher | Größe |
|---|---|---|
| Ordner `A` (Code: `quantum_sim/`, `qire_ai/`, `server_setup/`) | dieser Ordner | klein (ohne `results/`) |
| `train.npy` | Archiv `…/qire_ai_console_v1_0/results/korpus/` | 795 MB |
| `val.npy` | `A/results/korpus/` oder Archiv | 4 MB |
| `bpe_4096.json` | `A/results/korpus/` oder Archiv | 0,3 MB |

Den Korpus **nicht** auf dem Cluster neu aufbereiten (`korpus_deutsch`
lädt von HuggingFace — ANNAHME: geht dort nicht). Die drei Dateien in einen
Ordner legen und `QIRE_KORPUS` darauf zeigen lassen.

Python-Pakete: `torch` (mit CUDA), `numpy`, `scipy`, `scikit-learn`, `psutil`,
`tokenizers`. (`scikit-learn` und `psutil` werden über `qire_ai.benchmark`
importiert, obwohl T-002 sie nicht braucht — ohne sie bricht der Import ab.)

## 5. Ablauf

Auf dem Login-Knoten, im Ordner `A`:

```bash
export QIRE_KORPUS=/pfad/zu/korpus          # train.npy, val.npy, bpe_4096.json
export QIRE_PYTHON=/pfad/zu/venv/bin/python # oder: QIRE_UMGEBUNG=/pfad/umgebung.sh
export SBATCH_EXTRA="--partition=gpu --account=xyz"   # ANNAHME: Namen erfragen
# optional: export QIRE_T002_ZELLEN=/schneller/pfad/t002_zellen

bash server_setup/cluster/t002_steuerung.sh pruefen       # Korpus, Pakete, Selbsttest (Sekunden, CPU)
bash server_setup/cluster/t002_steuerung.sh rechenbedarf  # Tabelle wie oben
```

Dann **entweder** alles automatisch (braucht `sbatch` auf Rechenknoten):

```bash
bash server_setup/cluster/t002_steuerung.sh auto
```

`auto` schickt die nächste offene Phase als Job-Array ab und hängt einen
kleinen CPU-Job an, der nach dem Array `auto` erneut aufruft. Das geht so
weiter, bis alle Zellen da sind; zum Schluss läuft `zusammenfuehren`.

**oder** von Hand, Phase für Phase:

```bash
bash server_setup/cluster/t002_steuerung.sh lr        # Startgitter abschicken
# ... warten, bis squeue leer ist ...
bash server_setup/cluster/t002_steuerung.sh lr        # Verlängerungen (falls nötig), bis "vollständig"
bash server_setup/cluster/t002_steuerung.sh messung   # 32 Messzellen
# ... warten ...
bash server_setup/cluster/t002_steuerung.sh status
bash server_setup/cluster/t002_steuerung.sh zusammenfuehren
```

Schutzmechanismen:
- Laufen noch `t002`-Jobs, wird nichts neu abgeschickt (sonst doppelte Zellen).
- Ist der neue Plan identisch mit dem vorigen, scheitern die Zellen — dann
  wird **nicht** endlos neu abgeschickt; Logs in `$QIRE_T002_ZELLEN/logs/`.
- Eine Zelle, deren Datei schon existiert, wird übersprungen (`--requeue` ist
  damit gefahrlos). Eine Zelle, die mitten im Lauf abbricht, verliert bis zu
  ~74 min; Zwischenstände innerhalb einer Zelle gibt es nicht.
- Andere Schrittzahl als 13.000 wird verweigert (`--nicht-vorregistriert`
  nur nach Tagebucheintrag).

Der Zusatzarm läuft nur mit `MIT_ZUSATZARM=1` — und nur, wenn Malte das
vorher entschieden und ins Tagebuch geschrieben hat.

## 6. Ergebnisse nach Hause holen

```
$QIRE_T002_ZELLEN/T-002/lr/*.json        LR-Zellen
$QIRE_T002_ZELLEN/T-002/lr_map.json      Lernratenkarte mit Randvermerken
$QIRE_T002_ZELLEN/T-002/messung/*.json   Messzellen
A/results/experiments/T-002__*.json      Sammeldatei mit Auswertung (von zusammenfuehren)
A/results/modelle/*.pt                   Gewichte des ersten Seeds je Arm (Chatbot)
```

Zu Hause in denselben Pfaden ablegen; `python -m quantum_sim.sprachmodell
zusammenfuehren` rechnet die Auswertung dann lokal nach, und `q_summary` liest
die Sammeldatei.

## 7. Einzelbefehle (für Fehlersuche)

```bash
python -m quantum_sim.sprachmodell plan lr                 # offene LR-Zellen (eine je Zeile)
python -m quantum_sim.sprachmodell plan messung            # offene Messzellen
python -m quantum_sim.sprachmodell lernratensuche --arm dense --lr 0.0006 --seed 2199
python -m quantum_sim.sprachmodell zelle --arm dense --seed 2101
python -m quantum_sim.sprachmodell status
```
