# Auftrag an die Raven-Sitzung (zum Einfügen)

Erkundungspaket Welle 2 starten (Freigabe Malte 28.10.):
1. `cd ~/qire && git stash; git pull` (main ≥ 5792029). Prüfen: `ls INSTITUT/erkundung/[KE][0-9][0-9]*.py | wc -l` muss 27 ergeben.
2. Daten in INSTITUT/daten (mnist.npz, fashion_mnist.npz, de_gsd_*.conllu) prüfen, sonst wie bei I-048 vorladen.
3. `mkdir -p logs && sbatch MPI/slurm/erkundung.sbatch` (1 Knoten, 72 Kerne, 24 h). 135 Aufgaben (27 Skripte × Tuning-Seeds 950–954), je ≤ 50 min. Fertige roh/*.json werden übersprungen; bei Abbruch einfach neu einreichen.
4. Das sind Erkundungen auf Tuning-Seeds: nichts in LABORBUCH, agenda.json oder seeds.json. Fehler (roh/*.log mit FEHLER) melden, nicht umbauen.
5. Wenn fertig: `tar czf logs/erkundung_roh.tgz INSTITUT/erkundung/roh` und Bescheid geben.
6. Für die I-048-Auswertung als Vorbehalt vermerken: `quant_rows(…, bits=2)` nutzt nur 3 Stufen (ternär, 1,58 Bit), gezählt werden 2 Bit; bei 3 Bit 7 statt 8 Stufen. Das benachteiligt die Quantisierungsarme (konservativ).
