# MPI-Cluster: Vorbereitung für lange CPU-Läufe und QLoRA-Training (Plan R5.2)

*Stand 14.10.2026. **Der Zugang ist zugesagt (HANDOVER 06.10.: „Malte bekommt einen Zugang zum MPI in München, Details folgen“), alles Weitere ist unbekannt.** Jede Angabe, die nicht aus dem Repo stammt, ist mit **ANNAHME** gekennzeichnet und vor dem ersten Lauf an die Cluster-Dokumentation anzupassen. Nichts hier wurde auf dem Cluster getestet; getestet sind nur Skript-Syntax, `starte.sh --trocken` und die Datenprüfung (`python -m pytest MPI -q`).*

## MPCDF Garching: Schnellstart erster Test (Stand 22.10.2026)
Quelle: docs.mpcdf.mpg.de (Connecting-FAQ, Viper-GPU-Guide). Was nicht dort steht, ist **ANNAHME**.
- **Systeme:** Raven (CPU + A100), Viper (CPU + MI300A-APUs). Scheduler Slurm.
- **Zeitlimit:** 24 h je Job (Viper-GPU belegt; Raven ANNAHME gleich). PhD-/Postdoc-Status ändert das nicht. Längere Läufe: `starte.sh cpu <ID> --kette N` (Wiederaufnahme je Seed), kein eigener Worker nötig. Prüfen: `sinfo -o "%P %l"`.
- **Login:** nur SSH mit 2FA (Pflicht, OTP-App). Von außen über `gate1`/`gate2.mpcdf.mpg.de` als Jump-Host. Claude kann sich deshalb nicht selbst einloggen; einreichen macht Malte.
  ```
  Host mpcdf-gate
      HostName gate1.mpcdf.mpg.de
      User <mpcdf-name>
      ControlMaster auto
      ControlPath ~/.ssh/cm-%r@%h
      ControlPersist 12h          # 2FA nur einmal je 12 h
  Host raven
      HostName raven.mpcdf.mpg.de
      User <mpcdf-name>
      ProxyJump mpcdf-gate
  ```
- **Rechenknoten ohne Internet (ANNAHME):** Daten (`INSTITUT/daten/*.npz`) mit `uebertragen.sh` hochladen. Große Dateien nach `/ptmp/$USER` (ANNAHME: wird nach Wochen ohne Zugriff gelöscht).
- **GitLab (gitlab.mpcdf.mpg.de):** nicht nötig. Code per `git clone`/`git pull` von GitHub auf dem Login-Knoten (nur lesen), Ergebnisse per rsync zurück, Commit vom PC.

**Erster Test, Schritt für Schritt (ca. 30 min):**
1. `ssh raven` (2FA eingeben).
2. `git clone https://github.com/thelonelycookie/QIRE.git ~/qire && cd ~/qire && mkdir -p logs`
3. `bash MPI/pruefe_cluster.sh | tee logs/cluster_info.txt` → Partition, Limits, Python-Modul ablesen.
4. `module load <python-modul>; python3 -m venv ~/venvs/qire && . ~/venvs/qire/bin/activate && pip install -r MPI/requirements_cpu.txt`
5. `cp MPI/lokal.env.beispiel MPI/lokal.env`, Werte aus Schritt 3 eintragen. Daten: `cd INSTITUT && python -c "from labor import gemeinsam as g; g.bilder('mnist', 0)" && cd ..`
6. `MPI/slurm/starte.sh test` → `squeue -u $USER`, danach `cat logs/qire-test-*.out`.
7. `logs/cluster_info.txt` und das Test-Log ins Repo kopieren (vom PC committen) oder Claude in den Chat geben → Claude stellt `lokal.env`-Werte und Sbatch-Zeiten fest ein.

## 0. Wofür
| Zweck | Quelle im Repo | Vorlage |
|---|---|---|
| Lange CPU-Läufe des Instituts (`direktor.py lauf <ID>`), später Läufe, die auf Maltes Ryzen 5 3600 zu lang dauern | CLAUDE.md „Technik“; Wiederaufnahme je Seed ist Pflicht (`labor/training.lauf_je_seed`) | `slurm/cpu_lauf.sbatch` |
| QLoRA-Training von Bewerter (S3) und Erfinder (S4) der Ideen-KI, Zyklus S5 | `IDEEN_KI/PLAN_2026-10-13.md` §6 (ab ~150–300 bewerteten Ideen) | `slurm/gpu_qlora.sbatch`, `training/qlora_train.py` |
| Größere Modelle (≥ 1 Mrd. Parameter) | HANDOVER 06.10., Punkt 3 der Reihenfolge | wie GPU-Vorlage |

Dateien: `slurm/starte.sh` (Einreichen), `uebertragen.sh` (rsync), `umgebung.yml` / `requirements_cpu.txt` / `requirements_gpu.txt`, `lokal.env.beispiel` (→ `lokal.env`, gitignored), `.gitignore`, `test_mpi.py`.

## 1. Was beim ersten Gespräch mit dem MPI zu klären ist (Checkliste)
1. Welches Institut/Rechenzentrum, welches System? Scheduler wirklich **Slurm**? (ANNAHME: ja.)
2. Zugang: SSH-Schlüssel, 2-Faktor, Jump-Host/VPN? Wer legt den Account an, wie lange gilt er, wer ist Ansprechpartner?
3. Wie viel darf Malte (Jugend forscht, Schüler) nutzen: Kontingent (CPU-Stunden, GPU-Stunden), maximale Laufzeit je Job, maximale Jobs gleichzeitig?
4. Partitionen und Namen: CPU-Partition, GPU-Partition, `--account`, GPU-Typ und VRAM (A100 40/80 GB? ANNAHME), `--gres`-Schreibweise.
5. **Internet von Rechenknoten?** (ANNAHME: nein, nur Login-Knoten.) Wenn nein: Daten und Modelle vorab hochladen (Abschnitt 4).
6. Software: `module`-System? Python-Version, CUDA-Version, conda/mamba erlaubt? Container (Apptainer) statt venv?
7. Speicher: Home-Quota, Projekt-/Scratch-Bereich, Löschfristen (Scratch wird oft automatisch geleert → Ergebnisse zeitnah zurückholen).
8. Darf `scontrol requeue` benutzt werden? (Sonst `starte.sh … --kette N` verwenden.)
9. Regeln zu Nutzungszweck, Datenschutz, Namensnennung in der Arbeit (Danksagung).

## 2. Zugang (keine Zugangsdaten ins Repo)
- Schlüsselpaar **nur auf Maltes PC** erzeugen (`ssh-keygen -t ed25519`), mit Passphrase. Nur der **öffentliche** Schlüssel geht an das MPI. Privater Schlüssel, Passwörter, Tokens, 2FA-Seeds stehen **nie** im Repo, in Skripten, in `lokal.env`-Kopien oder in Chats. `MPI/.gitignore` sperrt `*.key`, `*.pem`, `id_*`, `*token*`, `lokal.env` usw.; zusätzlich vor jedem Commit `git status` ansehen.
- Zugang als Alias in `~/.ssh/config` (außerhalb des Repos). Beispiel mit Platzhaltern (Werte vom MPI):
  ```
  Host mpi
      HostName <login-knoten.beispiel.invalid>
      User <benutzername>
      IdentityFile ~/.ssh/id_ed25519
      # ProxyJump <jumphost>       # falls verlangt
  ```
  Im Repo erscheint nur der Alias (`MPI_SSH_ALIAS=mpi` in `lokal.env`).
- Auf dem Cluster **kein** GitHub-Zugang einrichten und nichts committen/pushen (Regeln: Commits nur unter Maltes Namen, vom PC aus). Ergebnisse kommen per rsync zurück (Abschnitt 4).
- Die Zusage gilt Malte persönlich: Zugangsdaten nicht an Claude/Agenten weitergeben. Agenten bereiten Skripte vor; eingeloggt und eingereicht wird von Malte.

## 3. Einrichtung auf dem Cluster (einmalig, ANNAHME: Linux, Login-Knoten mit Internet)
```bash
ssh mpi
mkdir -p $PROJEKT/qire && cd $PROJEKT/qire      # $PROJEKT = Projekt-/Scratch-Pfad laut MPI
# Code von Maltes PC hochladen (siehe Abschnitt 4), dann:
python3 -m venv ~/venvs/qire && . ~/venvs/qire/bin/activate
pip install -r MPI/requirements_cpu.txt        # oder: mamba env create -f MPI/umgebung.yml
cp MPI/lokal.env.beispiel MPI/lokal.env && nano MPI/lokal.env     # Alias, Pfade, Partition, Account, Module ausfüllen
```
- Torch-Rad passend wählen (CPU-Rad reicht für Institutsläufe; für QLoRA CUDA-Rad passend zur Cluster-CUDA). Nach der Einrichtung `pip freeze > MPI/requirements_cpu.lock` und die Versionen in LABORBUCH/Methodik festhalten (Reproduzierbarkeit).
- **Daten vorab hinlegen:** `labor/gemeinsam.py` lädt MNIST/Fashion beim ersten Zugriff per `urllib` aus dem Internet (`_roh`). Auf Rechenknoten ohne Internet schlägt das fehl. Deshalb `INSTITUT/daten/*.npz` vom PC mit hochladen (oder einmal auf dem Login-Knoten `python -c "from labor import gemeinsam as g; g.bilder('mnist', 0)"` in `INSTITUT/` ausführen) und `QIRE_DATEN` auf den Ordner zeigen lassen (`lokal.env`).
- **Vor dem ersten echten Lauf:** `cd INSTITUT && python direktor.py vortest` auf dem Login-Knoten bzw. in einem kurzen Interaktiv-Job (`srun --pty`), dann `python -m pytest -q` im Repo, damit die Umgebung belegt funktioniert.

## 4. Datenübertragung
```bash
MPI/uebertragen.sh hin --trocken     # zeigen, was hochgeladen würde (rsync -n)
MPI/uebertragen.sh hin               # Code (INSTITUT, A, MPI) + Zwischenstände; ohne .git, Geheimnisse, lokal.env
MPI/uebertragen.sh zurueck           # Ergebnisse (_zwischen_*, *.json), laeufe/*.status.json, Adapter, Logs
MPI/uebertragen.sh modell ./modelle/Qwen3-4B     # große Modelle einzeln, einmalig
```
- rsync mit `--update`/`--partial`, **nie `--delete`**: nichts auf dem PC wird überschrieben, was neuer ist.
- Größen (Stand heute): `INSTITUT/ergebnisse` ≈ 16 MB, `A/results` ≈ 5,5 MB → unkritisch. Modelle (GGUF/HF) sind groß (GB): nur einmal hochladen, nicht in Git.
- `agenda.json`, `LABORBUCH.md`, `FEHLERBUCH.md`, `seeds.json` werden **nicht** zurückgespielt. `direktor.py lauf` ändert die Agenda auf dem Cluster (setzt `erledigt`, schaltet Folgefragen frei); diese Kopie ist Arbeitskopie. Auf dem PC/durch Claude werden Urteil und Agenda aus den zurückgeholten Ergebnis-JSONs nachgezogen (`direktor.py status`, Befund ins LABORBUCH, FEHLERBUCH-Lehre per `labor/nachtragen.py`).
- Seeds: Vorregistrierung und Seed-Register (`INSTITUT/seeds.json`, `werkzeuge/seeds_pruefen.py`) gelten unverändert; **neue vorregistrierte Läufe weiterhin nur mit Maltes Freigabe** (CLAUDE.md). Der Cluster ändert daran nichts.

## 5. CPU-Lauf einreichen und überwachen
```bash
MPI/slurm/starte.sh cpu I-050 --schnell --trocken    # Rauchtest-Befehl ansehen (nur Tuning-Seeds!)
MPI/slurm/starte.sh cpu I-050                        # echter Lauf (nach Freigabe)
MPI/slurm/starte.sh cpu I-050 --kette 3              # 3 verkettete Glieder, falls requeue verboten
squeue -u $USER ; tail -f logs/qire-lauf-<JOBID>.out
cd INSTITUT && python direktor.py warte I-050 --timeout=600     # liest NUR laeufe/I-050.status.json (R1.4), nie das Log
```
- **Ressourcen (ANNAHME):** 8 Kerne, 32 GB, 12 h. Das Institut parallelisiert über Seeds/Aufgaben (`labor.gemeinsam.parallel`), ein Thread je Prozess (`OMP_NUM_THREADS=1`), Prozesszahl = `INSTITUT_PROZESSE` = Kerne des Jobs. Mehr Kerne helfen nur bis zur Zahl der Seeds/Aufgaben. RAM je Prozess aus dem letzten PC-Lauf abschätzen (T-006 starb schon einmal an OOM): lieber `--mem` großzügig als knapp.
- **Wiederaufnahme:** Die Module speichern nach **jedem Seed** atomar in `INSTITUT/ergebnisse/_zwischen_<ID>*`. Ein Neustart (nach Zeitlimit, Knotenausfall, `scancel`) überspringt fertige Seeds. Der Job bekommt 15 min vor Zeitende `SIGUSR1`, gibt python eine Gnadenfrist, beendet es und reiht sich per `scontrol requeue` neu ein (max. 20 Neustarts, danach Abbruch). Bei Fehlern (python-Exitcode ≠ 0 ohne Signal) wird **nicht** neu gestartet, damit keine Endlosschleife entsteht.
- **Vor dem ersten großen Lauf die Wiederaufnahme testen:** kurzen Lauf mit `--schnell` einreichen, nach dem ersten Seed `scancel`, erneut einreichen, im Log muss „Seed N: aus Zwischenstand übernommen“ stehen.
- Der Status `laeuft` bleibt nach hartem Abbruch stehen und wird nach 10 min ohne Herzschlag automatisch `tot` (`labor/laufstatus.py`); der Neustart überschreibt ihn.
- Nur eine Sache auf einmal pro Knoten reservieren; nicht mehr Jobs einreichen, als das Kontingent hergibt (Abschnitt 1, Punkt 3).

## 6. GPU-Job: QLoRA für Ideen-KI (S3 Bewerter, S4 Erfinder) — später
Voraussetzungen laut Plan: ≥ 150–300 bewertete Ideen (Urteile H1/H0/HU, Maltes Blindnoten); Basismodell aus M0 (Modellname/Größe noch **unbestätigt**).
```bash
export BASIS_MODELL=$HF_HOME/<modellordner>     # vorab hochgeladen (kein Internet auf Rechenknoten, ANNAHME)
export TRAIN_DATEN=$PWD/MPI/daten/bewerter.jsonl
python MPI/training/qlora_train.py --daten-pruefen --daten $TRAIN_DATEN      # Format + Zahlen prüfen (auch auf dem PC)
MPI/slurm/starte.sh gpu s3 --trocken ; MPI/slurm/starte.sh gpu s3
```
- **Datenformat (ANNAHME)** JSONL `{"prompt": …, "antwort": …}`; die Erzeugung aus `IDEEN_KI/ausgabe` + `schmiede/rueckmeldung.py` ist noch zu bauen. 10 % der Beispiele bleiben deterministisch (Hash des Prompts) zurückgehalten und sind die einzige Grundlage für „schlägt Basis-Bewerter“ (Abnahme S3) bzw. den Blindtest (S4).
- **Checkpoints:** alle 100 Schritte, die letzten 3 bleiben, Fortsetzung ab dem neuesten; bei `SIGUSR1`/`SIGTERM` wird gesichert und mit Exit-Code 75 beendet, der Job reiht sich neu ein. Fertiger Adapter: `MPI/adapter/<stufe>/adapter_final/` samt `qire_lauf.json` (Basis, Datenzahl, Seed, Versionen).
- **Speicherbedarf (ANNAHME, grobe Faustregel, am Cluster nachmessen):** 4-Bit-Basis plus LoRA für ein 3–4B-Modell braucht je nach Kontextlänge und Batch grob 8–16 GB VRAM; ein ~20–30B-Modell (Erfinder) deutlich mehr (> 24 GB). Bei OOM zuerst `--batch` senken, `--akkumulation` erhöhen, `--max-laenge` kürzen.
- **Zurück auf Maltes PC:** Adapter per `uebertragen.sh zurueck`; für llama.cpp muss der Adapter bzw. das gemergte Modell nach GGUF konvertiert werden (Werkzeuge von llama.cpp, Vorgehen bei Bedarf prüfen); danach in `IDEEN_KI/config.yaml` eintragen.
- **Ehrlich in der Arbeit:** „auf QIRE spezialisiertes, feinjustiertes Basismodell (Name, Version)“, nicht „von Grund auf trainiert“ (Plan §7).
- `qlora_train.py` ist eine **ungetestete Vorlage** (transformers/peft/bitsandbytes-Versionen am Cluster anpassen). Erst mit einem Mini-Datensatz (20 Beispiele, 1 Epoche) trocken fahren.

## 6a. GPU-Institutslauf auf Raven (I-043-1B, Stand 27.10.2026)
Getestet am 27.10. (`gpu_test.sbatch`): A100-SXM4-40GB, torch 2.14.1+cu126, LLaMmlein_1B offline aus `HF_HOME=/ptmp/$USER/hf`, venv `~/venvs/qire-gpu`, 13.400 Tokens/s (LoRA r=16, Batch 8×512), 10,6 GB. Partition `gpu`, Account `msp_gpu`, `--gres=gpu:a100:N` (bis 4 je Knoten), 24 h.
```bash
ssh raven; cd ~/qire && git pull --no-rebase origin main
bash MPI/daten_i043.sh                                   # einmalig, Login-Knoten (Internet): deutsche Wikipedia -> /ptmp/$USER/i043 (Lizenz CC BY-SA 3.0/GFDL)
QIRE_ZEIT=01:00:00 MPI/slurm/starte.sh gpulauf I-043-1B --schnell   # Rauchtest NUR Tuning-Seed 2692 (~10-20 min), misst Tokens/s je Arm
MPI/slurm/starte.sh gpulauf I-043-1B --kette 3           # Volllauf nach Maltes Start-Entscheidung; 4 GPUs, 1 Aufgabe je GPU
squeue -u $USER ; tail -f logs/qire-gpulauf-<JOBID>.out ; cd INSTITUT && python direktor.py warte I-043-1B --timeout=600
```
- `starte.sh gpulauf <ID>`: ganzer Knoten (`QIRE_GPUS`, Standard 4; 18 Kerne und 120 GB je GPU), `SLURM_PARTITION_GPU` / `SLURM_ACCOUNT_GPU` aus `lokal.env` (sonst `gpu` / `msp_gpu` aus der sbatch-Datei); `--kette N` wie bei `cpu`.
- Wiederaufnahme: jede fertige Aufgabe steht in `INSTITUT/ergebnisse/_zwischen_I-043-1B.json`, innerhalb einer Aufgabe gibt es alle 500 Schritte einen Checkpoint (`_ckpt_I-043-1B/`, nur Adapter + Optimierer, wenige MB, nicht in Git). Nach Zeitlimit/Requeue/Neueinreihen einfach dasselbe Kommando.
- Danach `MPI/uebertragen.sh zurueck`; Ergebnisdatei und `_zwischen_*` vom PC committen.

## 7. Fehlerfälle
| Symptom | Ursache (wahrscheinlich) | Maßnahme |
|---|---|---|
| Job startet nicht, keine `logs/…out` | Ordner `logs/` fehlt (Slurm legt ihn nicht an) | `starte.sh` benutzen (legt `logs/` an) oder `mkdir -p logs` |
| `Connection refused`/`urlopen error` beim Datenladen | Rechenknoten ohne Internet | Daten vorab (Abschnitt 3), `QIRE_DATEN` setzen |
| Neustartschleife | Fehler im Modul, aber Signal früher | Log lesen; `MAX_NEUSTARTS` begrenzt auf 20 (CPU) bzw. 10 (GPU) |
| `scontrol requeue` verweigert | Cluster erlaubt es nicht | `starte.sh … --kette N` |
| CUDA OOM | Batch/Länge/Modell zu groß | Abschnitt 6 |
| Lauf „tot“ im Status, Prozess läuft doch | Modul ruft `laufstatus.herzschlag` nicht auf (< 10 min je Seed nötig) | Herzschlag in der Seed-Schleife ergänzen; sonst Status ignorieren |
| Scratch geleert | Löschfrist | Ergebnisse sofort `zurueck` holen |

## 8. Erster Test (empfohlene Reihenfolge, wenn der Account da ist)
1. Zugang + Alias einrichten, `ssh mpi hostname`.
2. Repo-Teile hochladen (`uebertragen.sh hin --trocken`, dann echt), Umgebung bauen (Abschnitt 3).
3. `python direktor.py vortest` und `pytest -q` auf dem Cluster.
4. `starte.sh cpu <kurze ID> --schnell` → Wiederaufnahmetest (Abschnitt 5).
5. Ergebnis `zurueck`, auf dem PC mit gleichem Seed gegenprüfen (bitgleich oder mit erklärter Abweichung durch BLAS/Version).
6. Erst dann echte, freigegebene Läufe; GPU erst mit Mini-Datensatz.

## 9. Annahmen auf einen Blick
Scheduler Slurm · Linux mit Login-Knoten (Internet) und Rechenknoten (kein Internet) · `scontrol requeue` erlaubt · 8 Kerne/32 GB/12 h CPU, 1 GPU/64 GB/24 h GPU sind realistisch · venv oder conda möglich, Python ≥ 3.10 · Partitions- und Account-Namen leer, aus `lokal.env` · GPU ≥ 16 GB VRAM · JSONL-Trainingsformat wie oben · Kontingent ausreichend für Seeds-Läufe von wenigen Stunden.
