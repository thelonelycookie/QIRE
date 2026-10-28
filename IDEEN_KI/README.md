# QIRE-Ideen-KI (lokal) -- Stufe S1

Eine lokale KI erzeugt rund um die Uhr Ideen fuer QIRE in zwei Kanaelen (**QIRE-nah** = pruefbar,
**Neuland** = radikal neu), filtert und bewertet sie selbst und liefert die besten ~10 % als Karten im
`INSTITUT/ideen.json`-Stil. Sie **schlaegt nur vor**: kein Commit, kein Lauf, kein Internet, keine Handy-Nachrichten.
Plan: `PLAN_2026-10-13.md`.

Ablauf einer Runde: Kontext (RAG) -> Idee erzeugen (Operator) -> Duplikatfilter -> Physik-Pruefer ->
Rubrik 1-5 -> beste ~10 % zu Karten ausarbeiten -> `ausgabe/karten_JJJJMMTT.json` + `ausgabe/TOP.md`.

## Einrichtung Windows (RTX 3050 8 GB, Ryzen 5 3600, 16 GB RAM)

1. **Python 3.10+** installieren (python.org, "Add to PATH"). Dann in diesem Ordner:
   `pip install requests numpy pyyaml psutil pytest` (psutil nur fuer den RAM-Waechter, optional).
2. **llama.cpp (Windows, CUDA)**: Release-Seite `github.com/ggml-org/llama.cpp/releases`, das Paket
   `llama-*-bin-win-cuda-*-x64.zip` samt passendem `cudart-*.zip` laden und nach `IDEEN_KI\llama.cpp\` entpacken
   (dort muss `llama-server.exe` liegen; sonst `server.exe_pfad` in `config.yaml` setzen). *Paketnamen bei M0 pruefen.*
3. **Modelle (GGUF, Hugging Face)** nach `IDEEN_KI\modelle\` legen (Malte hat die Downloads erlaubt und fuehrt sie selbst aus).
   Alle Namen und Groessen sind **bei M0 zu pruefen**:

   | Rolle | Kandidat | Datei in config.yaml | ungefaehre Groesse |
   |---|---|---|---|
   | Erfinder 1 | gpt-oss-20b (MXFP4) | `modelle/gpt-oss-20b-mxfp4.gguf` | ~12 GB (bei M0 pruefen) |
   | Erfinder 2 | Qwen3-30B-A3B (Q4_K_M) | `modelle/Qwen3-30B-A3B-Q4_K_M.gguf` | ~18 GB (bei M0 pruefen) |
   | Ausweichloesung | dichtes ~8B-Modell, Q4 | `modelle/dicht-8b-Q4_K_M.gguf` | ~5 GB (bei M0 pruefen) |
   | Embedding (optional) | kleines Embedding-GGUF | `modelle/embedding-klein.gguf` | 0,1-0,6 GB |

   Download z. B. mit `pip install huggingface_hub` und `huggingface-cli download <Repo> <Datei> --local-dir modelle`
   (Repo-Namen auf huggingface.co suchen, bei M0 pruefen). Platz: ~10-35 GB je nach Auswahl.
4. **Windows-Einstellungen**: PC wach halten, Energiesparen aus (Dauerbetrieb).
5. Duplikate: Standard ist der lokale TF-IDF-Fallback (`modelle.embedding.modus: tfidf`). Mit `server` bzw. `haupt` nutzt die KI
   `/v1/embeddings` eines zweiten (`--embedding`, Port 8081, selbst starten) bzw. des Hauptservers.

## Schritt M0 (Modellmessung, keine Annahme)

`python m0_messung.py --minuten 15` misst je Kandidat 15 min: Tokens/s, VRAM-/RAM-/GPU-Temperatur-Spitzen und
beantwortet die 20 Probeaufgaben (`daten_fest/m0_aufgaben.txt`). Ergebnis in `ausgabe\m0\`:
`messung.json`, `blindbewertung.csv` (Malte traegt Noten 1-5 in Spalte `note_1_bis_5` ein, Modelle sind als A/B/C
verdeckt), `schluessel.json` (erst NACH der Bewertung ansehen). Sieger in `config.yaml` unter `modelle.aktiv`
eintragen und `n_cpu_moe` / `ctx` an die Messung anpassen (zu wenig VRAM -> `n_cpu_moe` erhoehen).
Trockenlauf ohne GPU: `python m0_messung.py --attrappe --minuten 0.05`.

## Start, Stopp, Status, Wiederaufnahme

| Aktion | Windows | Linux |
|---|---|---|
| Start (Dauerbetrieb) | `start.cmd` | `./start.sh` |
| Sanfter Stopp (< 5 min) | `stopp.cmd` oder Datei `STOP` anlegen | `./stopp.sh` |
| Notfall-Stopp | `stopp.cmd /hart` (taskkill aller Prozesse) | `./stopp.sh --hart` |
| Trockenlauf ohne Modell | `python -m schmiede.betrieb --attrappe --max-ideen 30` | dito |

- **Stopp-Garantie**: `STOP` wird vor jedem LLM-Aufruf und nach jedem Schritt geprueft; jeder Aufruf hat ein Timeout <= 90 s.
  Danach wird gespeichert (Checkpoint, Banditen, Lueckenkarte) und der llama.cpp-Server beendet.
- **Status**: `status.json` (Herzschlag alle 5 min): Ideen/Stunde, letzte Aktivitaet, Fehler, Waechterzustand.
  Log: `daten\betrieb.log`, Serverlog: `daten\server.log`.
- **Wiederaufnahme**: einfach erneut `start.cmd`. Jede Idee steht sofort in `daten\ideen.jsonl` (append + fsync);
  angefangene Ideen werden in ihrer Stufe fortgesetzt, die Runde laeuft ab dem Checkpoint weiter.
- **Sicherung**: stuendlich `backup\backup_*.zip` (24 rotierend) von `daten\` und `ausgabe\`.
- **Waechter**: GPU-Temperatur und VRAM (`nvidia-smi`), RAM (`psutil`) -> drosseln bzw. pausieren (Schwellen in `config.yaml`).

## Rueckkopplung (S2-Anfang)

- Urteile: Eintrag in `INSTITUT/agenda.json` mit Feld `"quelle_ideen_ki": "<Karten-ID>"`; sobald in `INSTITUT/ergebnisse/<ID>__*.json`
  Urteile H1/H0/HU stehen, bekommt der Operator der Karte Belohnung (H1 1,0 / HU 0,3 / H0 0,2) und die Karte wandert als
  Few-shot-Beispiel in `daten/beispiele.jsonl`.
- Blindnoten: `ausgabe\blindnoten.csv` (Vorlage wird bei der ersten Karte erzeugt, `;`-getrennt). Malte traegt Noten 1-5 ein
  (Komma erlaubt); wird bei jeder Runde eingelesen und nie ueberschrieben.

## Ordner

```
config.yaml            Modelle, Server, Temperaturen, Rubrik-Gewichte, Intervalle, Schwellen
schmiede/              llm, server, kontext, operatoren, filter, pruefer, karten, rueckmeldung, betrieb
daten_fest/            Konzeptlisten (QM / Fremdgebiete), M0-Aufgaben
literatur/             (optional) eigene Notizen .md/.txt, fliessen in den Kontext ein
tests/                 pytest, nur mit Attrappen-Modell:  python -m pytest -q
daten/ backup/ ausgabe/ modelle/ llama.cpp/   nicht im Git (siehe .gitignore; ausgabe/TOP.md ausgenommen)
```

## Annahmen und offene Punkte

- llama.cpp-Optionen (`--n-cpu-moe`, `-ctk/-ctv`, `-fa on`) aendern sich zwischen Versionen: bei M0 gegen `llama-server --help` pruefen
  (`server.flash_attn_args`, `server.zusatz_args`).
- Modell- und Dateinamen sind unbestaetigt. gpt-oss kann "Denk"-Ausgaben liefern; der Parser ignoriert `<think>`-Bloecke.
- Der zweite Embedding-Server wird nicht automatisch gestartet (Standard: TF-IDF-Fallback).
- Die Rubrik-Bewertung durch dasselbe Modell ist ein grober Filter; Maltes Blindnoten sind der Massstab (Abnahme S1: 20 Karten blind).
