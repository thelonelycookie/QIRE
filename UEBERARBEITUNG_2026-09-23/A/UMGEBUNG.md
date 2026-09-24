# Umgebung — wie der Code hier läuft

## Der kurze Weg

```
C:/Users/Malte/Downloads/qire_ai_console_v1_0_remote_research_platform/qire_ai_console_v1_0/.venv/Scripts/python.exe
```

Python 3.11.9. Aus **diesem** Ordner heraus aufgerufen, zum Beispiel:

```bash
cd "C:/Users/Malte/Downloads/claude/projects/QIRE AI/A"
"C:/Users/Malte/Downloads/qire_ai_console_v1_0_remote_research_platform/qire_ai_console_v1_0/.venv/Scripts/python.exe" -m quantum_sim.q_audit
```

In PowerShell geht auch die Abkürzung im Ordner:

```powershell
.\py.cmd -m quantum_sim.q_audit
```

## Warum die Umgebung nicht mitkopiert ist

Sie ist **1,3 GB** groß, und die Systemplatte hat noch **29 GB von 1,9 TB**
frei. Eine zweite Kopie wäre 4 % des verbleibenden Platzes für nichts.

## Warum das trotzdem den Code von hier benutzt

Alle Pfade im Projekt lösen über `Path(__file__).resolve().parent.parent` auf,
**nicht** über das Arbeitsverzeichnis. Weil `python -m` das aktuelle
Verzeichnis vorne auf `sys.path` legt, gewinnen die Pakete aus diesem Ordner,
und `results/` zeigt dann ebenfalls hierher. Nachprüfen:

```bash
"$VENV" -c "import quantum_sim, qire_ai; print(quantum_sim.__file__); print(qire_ai.__file__)"
```

Beide Pfade müssen `...\projects\QIRE AI\A\...` enthalten. Tun sie das nicht,
stimmt das Arbeitsverzeichnis nicht.

**Geprüft am 30. August 2026:** `q_audit` läuft aus diesem Ordner durch,
**106 von 106** Prüfungen bestanden, `q_summary` ebenfalls.

## Falls der Archivordner verschwindet

Dann fehlt die Umgebung, und sie wird hier neu gebaut:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install pyarrow tokenizers requests
```

`requirements.txt` deckt die Q-Serie und Projekt B ab. Die drei Nachzügler
braucht die T-Serie: `pyarrow` und `tokenizers` für den Korpus, `requests` für
den Download. **`scikit-learn` und `psutil` nicht vergessen** — sie stehen zwar
in `requirements.txt`, sind aber schon einmal übersehen worden und haben einen
Serverlauf nach der Korpusaufbereitung abgebrochen, weil `qire_ai.benchmark`
über `qire_ai.metrics` beide zieht.

Stand der Umgebung am 30. August 2026:

| Paket | Version |
|---|---|
| torch | 2.12.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| pyarrow | 25.0.1 |
| tokenizers | 0.23.1 |
| matplotlib | 3.11.0 |
| psutil | 7.2.2 |
| qiskit | 2.5.1 |

`customtkinter` und `openpyxl` gehören zur alten GUI und zum Excel-Export und
werden für die Q/T-Serie nicht gebraucht.

## Rechner und Grenzen

**Ryzen 5 3600, 6 physische Kerne, 16 GB RAM, keine brauchbare GPU.**

- Immer nur **ein** schwerer Lauf. Genauigkeiten werden von Fremdlast nicht
  verfälscht (alle Arme laufen sequentiell), Zeitmessungen schon.
- Ein Lauf über Nacht überlebt das Sitzungsende nicht zuverlässig. Q-011 ist
  daran gescheitert. Für lange Läufe entweder eine Wiederaufnahme einbauen
  (Vorlage: `weiter=True` in `sprachmodell.py`) oder auf den Server gehen.
- **RunPod hat derzeit kein Guthaben.** Skripte in `server_setup/`:
  `runpod_start.sh` (Ersteinrichtung inklusive Korpus), `runpod_t002.sh`
  (der T-002-Lauf), `alles_durchlaufen.sh`.
