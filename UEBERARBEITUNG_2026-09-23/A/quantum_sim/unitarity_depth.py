from __future__ import annotations

"""RESEARCH ID: Q-011 -- Unitaritaet zeigt sich erst in der Tiefe.

DIE IDEE, UND WARUM SIE AUS DEN EIGENEN DATEN KOMMT
Befund 4 der Gesamtbilanz lautete: "Orthogonalitaet gegen Spektralnorm, kein
Unterschied (p=0,32)". B-007 hat das am 4./5. August vorregistriert
nachgemessen, mit 20 Seeds und Lernrate je Arm:

    mnist, Tiefe 1:  +0,0025   17/20   Holm p=0,0020
    mnist, Tiefe 3:  +0,0051   17/20   Holm p=0,0001
    fashion_mnist:   repliziert nicht -> Urteil HU, Befund 4 bleibt unsicher

Zwei Punkte sind daran wichtig. Erstens: auf mnist ist der Unterschied sehr
wohl da, entgegen dem alten p=0,32. Zweitens, und das ist der Anlass fuer
diesen Lauf: **er ist bei Tiefe 3 doppelt so gross wie bei Tiefe 1.**

WARUM DAS THEORETISCH ZU ERWARTEN IST
Spektralnormalisierung fixiert nur den GROESSTEN Singulaerwert auf 1. Alle
uebrigen Richtungen duerfen schrumpfen. Wird eine solche Abbildung d-mal
hintereinander angewandt, faellt das Signal in diesen Richtungen wie sigma^d --
exponentiell in der Tiefe. Echte Orthogonalitaet haelt **alle** Singulaerwerte
bei 1; dort passiert das nicht.

Bei Tiefe 1 kann sich dieser Unterschied gar nicht aufbauen. Genau deshalb sind
unitaere Netze in der Literatur ein Thema fuer tiefe und rekurrente
Architekturen, nicht fuer flache. **Befund 4 wurde also bei einer Tiefe
gemessen, bei der die eigene Frage nicht auflösbar ist** -- dieselbe Diagnose
wie bei der blinden Architektur in B-004 und bei den Befunden 3 und 5, wo der
Rang nie die bindende Beschraenkung war.

WAS HIER ANDERS IST
Derselbe Aufbau wie B-007, unveraendert uebernommen (`ConstrainedBlock`,
`MiddleNet`, `_cell` werden importiert, nicht nachgebaut) -- nur die **Tiefe**
wird durchgefahren: 1, 3, 8, 16. Dadurch sind die Zahlen bei Tiefe 1 und 3
direkt mit B-007 vergleichbar.

DIE ARME
  orthogonal   alle Singulaerwerte 1 -- das klassische Gegenstueck unitaerer
               Quantengatter, also der quanteninspirierte Mechanismus
  spectral     nur der groesste Singulaerwert 1 -- die klassische Kontrolle,
               die in B-002 gleichzog
  plain        unbeschraenkt -- Bezugspunkt und Teil der Aufloesungspruefung

  `frobenius` und `weight_decay` fallen weg: der eine hat in B-007 nichts
  beigetragen, der andere kollabiert reproduzierbar auf Rateniveau (dreimal
  gemessen) und wuerde bei Tiefe 16 nur Rechenzeit kosten.

VORREGISTRIERUNG
  Getestet wird balancierte Genauigkeit, Lernrate je Arm UND Tiefe auf eigenen
  Tuning-Seeds, 15 Seeds je Zelle.

  PRIMAER  Der Abstand (orthogonal - spectral) waechst monoton mit der Tiefe.
           Spearman ueber die vier Tiefen auf mnist, einseitig positiv.
  P2       Bei der groessten Tiefe ist der Abstand signifikant (Wilcoxon).
  Holm ueber genau diese beiden.

  H1 : beide halten, und die Wiederholung auf fashion_mnist zeigt dieselbe
       Richtung -> Unitaritaet ist NICHT durch Spektralnormalisierung
       ersetzbar, sobald die Abbildung oft genug angewandt wird. Befund 4 waere
       damit nicht bestaetigt, sondern **als bei falscher Tiefe gemessen
       widerlegt** -- und das waere der erste quanteninspirierte Mechanismus
       des Projekts, der seine Abgrenzung besteht.
  H0 : kein Zuwachs mit der Tiefe (TOST auf die Steigung) -> Befund 4 haelt,
       und zwar jetzt aus dem richtigen Grund statt aus Mangel an Aufloesung.
  H0-: der Abstand SCHRUMPFT signifikant mit der Tiefe. Ergaenzt am 27. August
       vor dem Lauf; dieser Ausgang war nicht kodiert, und ein synthetischer
       Fall mit rho=-1,000 wurde dadurch als "kein Zuwachs, gleichwertig"
       gemeldet -- eine Aussage, die den Daten widerspricht. Dritter Fall
       desselben Fehlers nach Q-014 und Q-015.
  HU : weder noch -> erreichte Schranke berichten.

  GRENZE DES TRENDTESTS, ausdruecklich festgehalten: Spearman rechnet auf den
  sechs Mittelwerten je Tiefe und ignoriert deren Streuung. Eine flache Reihe
  kann durch Rauschen eine Scheinordnung erzeugen -- im Vortest ergab eine
  konstruktionsbedingt flache Reihe rho=-0,886. Deshalb wird sowohl H1 als auch
  H0- nur dann als Befund gefuehrt, wenn die Wiederholung auf fashion_mnist
  dieselbe Richtung zeigt; sonst gilt es als Beobachtung.

  AUFLOESUNGSPRUEFUNG (Regel 6) -- ZWEI STUFEN. Praezisiert am 27. August vor
  dem Lauf, weil der Code strenger prueft, als der Kopf hier beschrieb; die
  strengere Fassung ist die richtige und steht jetzt auch hier.

    (a) KOLLAPSSCHWELLE je Tiefe: mindestens ein Arm muss deutlich ueber
        Rateniveau liegen (>0,3 bei 10 Klassen). Kollabieren bei grosser Tiefe
        alle Arme -- wie es `weight_decay` schon bei Tiefe 3 tat --, misst die
        Zelle nur noch Trainingsversagen und wird aus der Trendrechnung
        ausgeschlossen, ausgewiesen und nicht stillschweigend.

    (b) TRENNSCHAERFE bei der kleinsten Tiefe: `orthogonal` muss `plain`
        signifikant schlagen. B-001 hat dafuer +0,0139 gemessen -- ein grober,
        bekannter Effekt. Reproduziert der Aufbau ihn nicht, kann er den
        feineren Unterschied zur Spektralnorm erst recht nicht aufloesen, und
        kein Trend ist interpretierbar. Die Schwelle (a) allein sagt nur, dass
        nicht alles zusammengebrochen ist -- nicht, dass der Aufbau trennt.

  ABBRUCHREGEL: Kollabieren bei einer Tiefe ALLE Arme, werden groessere Tiefen
  uebersprungen; das Ergebnis wird bis zur letzten auswertbaren Tiefe berichtet.

WIEDERAUFNAHME UND GITTER (ergaenzt am 23. September, vor jedem Neustart)
  * `run(weiter=True)` liest alle passenden Q-011-Dateien, fuehrt sie nach
    (Datensatz, Tiefe, Arm, Seed) zusammen und rechnet nur, was fehlt.
    Teilweise gerechnete Tiefen behalten ihre gespeicherten Lernraten.
  * Die Lernratensuche verlaengert in BEIDE Richtungen (Regel 1b) und
    vermerkt flache Suchen (Kollaps) statt sie zu verlaengern.
  * Gesichert wird alle fuenf Seeds, nicht mehr nur je Tiefe.

Run: python -m quantum_sim.unitarity_depth          Lauf MIT Wiederaufnahme
     python -m quantum_sim.unitarity_depth neu      Lauf ab null
     python -m quantum_sim.unitarity_depth test     Tests, kein Training
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qire_ai.ablation_unitarity import _cell
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

ARMS = ("plain", "orthogonal", "spectral")
# Von (1, 3, 8, 16) auf sechs Tiefen erweitert am 27. August, VOR dem Lauf.
# Grund, nachgerechnet: Der Primaertest ist ein Spearman ueber die Tiefen, und
# bei n Punkten ist 1/n! der kleinstmoegliche exakte einseitige p-Wert.
#   4 Tiefen -> 0,0417, nach Holm ueber zwei Vergleiche 0,0833  -> unerreichbar
#   5 Tiefen -> 0,0083, nach Holm 0,0167                        -> reicht
#   6 Tiefen -> 0,0014, nach Holm 0,0028                        -> sicher
# Mit vier Tiefen haette der Primaertest nur bestehen koennen, wenn P2 einen
# kleineren p-Wert hat -- er waere also nicht eigenstaendig gewesen. Das ist
# derselbe Entwurfsfehler wie in Q-009, nur eine Stufe subtiler.
# Tiefe 1 und 3 bleiben erhalten, damit die Zahlen mit B-007 vergleichbar
# bleiben; 2 und 4 sind die billigsten Ergaenzungen (Summe 28 -> 34).
DEPTHS = (1, 2, 3, 4, 8, 16)
DATASETS = ("mnist", "fashion_mnist")
PRIMARY_DATASET = "mnist"
EVAL_SEEDS = tuple(range(1801, 1816))       # 15 Seeds
TUNING_SEEDS = (1851, 1852, 1853)   # drei statt zwei: die Wahl der
                                    # Rate darf nicht am Rauschen haengen
# Nach unten erweitert gegenueber B-007 (dort 0,003 / 0,01): bei Tiefe 16 ist
# mit deutlich kleineren Raten zu rechnen, und Q-009 hat heute Nacht gezeigt,
# dass ein zu enges Gitter alle Arme an den Rand druecken kann.
LR_GRID = (0.0003, 0.001, 0.003, 0.01)
ERWEITERUNGEN = 3          # wie oft JE RICHTUNG verlaengert werden darf,
                           # wenn ein Arm auf dem Rand landet (Regel 1b)
WIDTH = 4096                                 # wie B-007, damit vergleichbar
MAX_SAMPLES = 6000
EPOCHS = 10
BATCH = 128
CHANCE = 0.3                                 # Schwelle der Aufloesungspruefung

# --- Wiederaufnahme (ergaenzt am 23. September) ---------------------------
# Der Fingerabdruck der Lernratensuche. Er steht ab jetzt in jeder Rohdatei,
# weil die Konfiguration des verworfenen ZWEITEN Anlaufs (halbes Suchbudget,
# zwei Tuning-Seeds) sich von der des dritten in KEINEM gespeicherten Feld
# unterscheidet: max_samples, epochs, lr_grid -- alles gleich. Ohne diesen
# Schluessel koennte eine Wiederaufnahme die entwerteten Zellen still
# uebernehmen, sobald jemand eine Datei aus `vorlaeufe_q011/` zurueckkopiert.
SUCHBUDGET = {"max_samples": MAX_SAMPLES, "epochs": EPOCHS,
              "tuning_seeds": list(TUNING_SEEDS)}
# Die Zwischenstaende des DRITTEN Anlaufs (27. August, 21:05 bis 23:00) tragen
# den Schluessel noch nicht. Dass sie mit vollem Budget gesucht haben, steht im
# Tagebuch (30. August) und in HANDOFF_AKTUELL §0; sie werden deshalb namentlich
# zugelassen -- und nur sie.
ALTDATEIEN_VOLLES_BUDGET = frozenset({
    "Q-011__20260827_210519.json", "Q-011__20260827_212401.json",
    "Q-011__20260827_214653.json", "Q-011__20260827_221758.json",
    "Q-011__20260827_230017.json"})
# Felder, in denen eine wiederaufgenommene Datei mit dem jetzigen Lauf
# uebereinstimmen muss.
PASSEND = {"width": WIDTH, "max_samples": MAX_SAMPLES, "epochs": EPOCHS,
           "batch_size": BATCH, "lr_grid": list(LR_GRID)}


def sweep_lr(dataset: str, depth: int,
             threads: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Lernrate je Arm UND Tiefe, Verlaengerung in BEIDE Richtungen.

    Regel 1 verlangt, dass das Gitter das Optimum ENTHAELT. Der erste Anlauf am
    27. August wurde nach zwei Tiefen abgebrochen, weil bei Tiefe 2 genau die
    ungluecklichste Kombination auftrat:

        orthogonal -> 0,01 (oberer Rand, stieg noch)
        spectral   -> 0,003 (innen, Optimum gefunden)

    Der Mechanismus-Arm war untertuniert, der Kontroll-Arm nicht. Der
    Primaerabstand (orthogonal - spectral) waere damit systematisch zu KLEIN
    gemessen worden -- also gegen die eigene Hypothese, und genau das ist der
    "blinde Aufbau", den das Protokoll ausschliessen soll.

    Landet ein Arm auf einem Rand, wird in DIESE Richtung verlaengert -- nach
    oben mal 3, nach unten durch 3, bis zu ERWEITERUNGEN Mal je Richtung.

    GEAENDERT AM 23. SEPTEMBER: bis dahin nur nach OBEN (und der Kopf dieser
    Funktion sprach faelschlich von "verdoppelt"; gerechnet wurde mal 3). Ein
    Arm am unteren Rand ist genauso untertuniert -- genau das hat Q-015 auf
    fashion_mnist entwertet. Vorlage: `zeitreihen.sweep_lr`, dort mit
    Gittertest. Der Test hier ist `test_gitterverlaengerung()`.

    NEU GEGENUEBER DER VORLAGE: eine FLACHE Suche (alle Mittel gleich) wird
    nicht verlaengert, sondern als "flach" vermerkt. Bei Tiefe 8 und 16 lieferte
    `plain` an allen Raten exakt 0,1000 (Kollaps); `max()` nimmt dann die erste
    Rate, also den unteren Rand, und die Verlaengerung nach unten wuerde drei
    Runden mit je drei teuren Tiefe-16-Zellen fuer nichts verbrennen.

    Rueckgabe: (beste Rate je Arm, Randvermerk je Arm: innen/oben/unten/flach).
    """
    best: Dict[str, float] = {}
    rand: Dict[str, str] = {}
    for arm in ARMS:
        means: Dict[float, float] = {}
        raten = list(LR_GRID)
        vermerk = "innen"
        oben_rest = unten_rest = ERWEITERUNGEN
        while True:
            for lr in raten:
                if lr in means:
                    continue
                # VOLLES Budget, nicht das halbe. Der zweite Anlauf am
                # 27. August ist genau daran gescheitert: gesucht wurde mit
                # 3000 Beispielen und 5 Epochen, gemessen mit 6000 und 10.
                # Bei kleinerem Budget sehen groessere Raten besser aus, die
                # Suche waehlt also systematisch zu hoch -- und welchen Arm
                # es trifft, ist Zufall. Ergebnis: in 5 von 5 Tiefen gewann
                # ausnahmslos der Arm mit der KLEINEREN Rate, um +-0,013.
                # Der gemessene "Unterschied zwischen Orthogonalitaet und
                # Spektralnorm" war der Unterschied zwischen 0,003 und 0,01.
                vals = [_cell(arm, depth, s, dataset, lr, WIDTH, MAX_SAMPLES,
                              EPOCHS, BATCH,
                              threads)["balanced_accuracy"] for s in TUNING_SEEDS]
                means[lr] = float(np.mean(vals))
            werte = list(means.values())
            if max(werte) - min(werte) < 1e-12:
                vermerk = "flach"
                break
            bester = max(means, key=means.get)
            if bester == max(means):
                if oben_rest == 0:
                    vermerk = "oben"
                    break
                oben_rest -= 1
                raten = [max(means) * 3]
                print(f"      {arm}: Optimum am oberen Rand ({bester:g}) -- "
                      f"verlaengere auf {raten[0]:g}")
            elif bester == min(means):
                if unten_rest == 0:
                    vermerk = "unten"
                    break
                unten_rest -= 1
                raten = [min(means) / 3]
                print(f"      {arm}: Optimum am unteren Rand ({bester:g}) -- "
                      f"verlaengere auf {raten[0]:g}")
            else:
                break
        best[arm] = max(means, key=means.get)
        rand[arm] = vermerk
        print(f"    Tiefe {depth:>2} / {arm:<11} -> lr {best[arm]:g}"
              + (f"  ACHTUNG: Rand {vermerk}" if vermerk != "innen" else "")
              + "   " + "  ".join(f"{k:g}:{means[k]:.4f}" for k in sorted(means)))
    return best, rand


def lade_fertige(verzeichnis: Optional[Path] = None) -> Dict[str, object]:
    """Liest ALLE passenden Q-011-Dateien und fuehrt sie zusammen.

    Muster aus `sprachmodell.trainieren(weiter=True)`, mit zwei Korrekturen,
    die der Review vom 23. September dort gefunden hat:

      * Dort wird nur die NEUESTE Datei gelesen. Ist die neueste eine Probe
        oder ein paralleler Teillauf, faengt der Hauptlauf still bei null an.
        Hier werden alle Dateien gelesen und nach
        (Datensatz, Tiefe, Arm, Seed) zusammengefuehrt; bei doppelten Zellen
        gewinnt die neuere Datei, abweichende Werte werden gemeldet.
      * Dort wird die Herkunft nicht geprueft. Hier nur Dateien mit
        demselben Suchbudget (Schluessel `suchbudget`, oder namentlich
        zugelassen, siehe ALTDATEIEN_VOLLES_BUDGET) und denselben Feldern in
        PASSEND. Alles andere wird mit Grund verworfen, nicht still.

    Unterordner (z. B. `vorlaeufe_q011/`) werden nicht gelesen.

    Rueckgabe: records (Liste), lr (Konfigurationsschluessel lr_map_*/lr_rand_*/
    lr_edge_* je (Datensatz, Tiefe)), gelesen, verworfen (Name, Grund),
    widersprueche (Anzahl Zellen mit abweichendem Wert).
    """
    from qire_ai import result_store

    ordner = Path(verzeichnis) if verzeichnis is not None else result_store.RESULTS_DIR
    zellen: Dict[Tuple[str, int, str, int], Dict] = {}
    lr_schluessel: Dict[str, object] = {}
    gelesen: List[str] = []
    verworfen: List[Tuple[str, str]] = []
    widersprueche = 0
    for pfad in sorted(ordner.glob("Q-011__*.json")):
        try:
            payload = json.loads(pfad.read_text(encoding="utf-8"))
        except Exception as exc:                  # noqa: BLE001
            verworfen.append((pfad.name, f"nicht lesbar: {exc!r}"))
            continue
        cfg = payload.get("config") or {}
        budget = cfg.get("suchbudget")
        if budget is None and pfad.name not in ALTDATEIEN_VOLLES_BUDGET:
            verworfen.append((pfad.name, "kein Suchbudget vermerkt und nicht "
                                         "als Datei des dritten Anlaufs bekannt"))
            continue
        if budget is not None and budget != SUCHBUDGET:
            verworfen.append((pfad.name, f"anderes Suchbudget {budget}"))
            continue
        abw = [k for k, v in PASSEND.items() if cfg.get(k) != v]
        if abw:
            verworfen.append((pfad.name, "abweichend in " + ", ".join(abw)))
            continue
        gelesen.append(pfad.name)
        for k, v in cfg.items():
            if k.startswith(("lr_map_", "lr_rand_", "lr_edge_")):
                lr_schluessel[k] = v
        for r in payload.get("records") or []:
            key = (str(r["dataset"]), int(r["depth"]), str(r["arm"]), int(r["seed"]))
            alt = zellen.get(key)
            if alt is not None and abs(float(alt["balanced_accuracy"])
                                       - float(r["balanced_accuracy"])) > 1e-9:
                widersprueche += 1
            zellen[key] = dict(r)
    # Eine Zelle zaehlt nur, wenn ihre Rate zur gespeicherten Wahl ihrer Tiefe
    # passt -- innerhalb einer Tiefe darf es nur EINE Rate je Arm geben.
    records = []
    for (ds, d, arm, _), r in sorted(zellen.items()):
        karte = lr_schluessel.get(f"lr_map_{ds}_d{d}")
        if not isinstance(karte, dict) or abs(float(karte.get(arm, -1)) - float(r["lr"])) > 1e-12:
            verworfen.append((f"Zelle {ds}/Tiefe {d}/{arm}/{r['seed']}",
                              "Rate passt nicht zur gespeicherten Lernratenwahl"))
            continue
        records.append(r)
    return {"records": records, "lr": lr_schluessel, "gelesen": gelesen,
            "verworfen": verworfen, "widersprueche": widersprueche}


def run(datasets: Sequence[str] = DATASETS, depths: Sequence[int] = DEPTHS,
        seeds: Sequence[int] = EVAL_SEEDS, threads: int = 6,
        weiter: bool = True, verzeichnis: Optional[Path] = None) -> Recorder:
    rec = Recorder(
        experiment_id="Q-011",
        description=("Waechst der Vorteil von Orthogonalitaet gegenueber "
                     "Spektralnormalisierung mit der Tiefe? Aufbau unveraendert "
                     "aus B-007 uebernommen, nur die Tiefe wird durchgefahren "
                     "(1, 2, 3, 4, 8, 16). 15 Seeds, Lernrate je Arm und Tiefe."),
        config={"arms": list(ARMS), "depths": list(depths),
                "datasets": list(datasets), "seeds": list(seeds),
                "width": WIDTH, "lr_grid": list(LR_GRID),
                "erweiterungen_je_richtung": ERWEITERUNGEN,
                "suchbudget": dict(SUCHBUDGET),
                "max_samples": MAX_SAMPLES, "epochs": EPOCHS,
                "batch_size": BATCH, "torch_threads": threads, "margin": MARGIN,
                "chance_threshold": CHANCE,
                "test_metric": "balanced_accuracy",
                "primary": "Spearman(Tiefe, orthogonal - spectral) > 0 auf mnist",
                "p2": "Abstand bei der groessten Tiefe signifikant",
                "holm_family": ["PRIMAER", "P2"],
                "primary_dataset": PRIMARY_DATASET,
                "anlass": ("B-007: mnist Tiefe 1 +0,0025, Tiefe 3 +0,0051 -- "
                           "der Effekt waechst, gemessen wurde aber nur bis 3")},
    )
    # --- Wiederaufnahme ----------------------------------------------------
    # Q-011 ist dreimal abgebrochen; der dritte Anlauf starb bei Tiefe 16 und
    # hinterliess 225 fertige Zeilen. Ohne Wiederaufnahme kostet ein Neustart
    # die vollen 10-14 Stunden noch einmal. Regel 5 zu Ende gedacht: wer
    # Zwischenstaende sichert, muss sie auch wieder einlesen koennen.
    fertig = set()
    if weiter:
        alt = lade_fertige(verzeichnis)
        for r in alt["records"]:
            rec.add(**r)
            fertig.add((r["dataset"], int(r["depth"]), r["arm"], int(r["seed"])))
        rec.config.update(alt["lr"])
        rec.config["wiederaufnahme"] = {"gelesen": alt["gelesen"],
                                        "verworfen": [list(v) for v in alt["verworfen"]],
                                        "widersprueche": alt["widersprueche"]}
        print(f"  WIEDERAUFNAHME: {len(fertig)} Zellen aus {len(alt['gelesen'])} "
              f"Datei(en) uebernommen")
        for name, grund in alt["verworfen"]:
            print(f"    verworfen: {name} -- {grund}")
        if alt["widersprueche"]:
            print(f"    ACHTUNG: {alt['widersprueche']} Zelle(n) mit abweichenden "
                  f"Werten in verschiedenen Dateien -- die neuere gilt")

    for dataset in datasets:
        collapsed_at = None
        for depth in depths:
            if collapsed_at is not None:
                print(f"  {dataset}, Tiefe {depth}: uebersprungen "
                      f"(bei Tiefe {collapsed_at} kollabierten alle Arme)")
                rec.config.setdefault("skipped", []).append(
                    {"dataset": dataset, "depth": depth,
                     "reason": f"Kollaps bei Tiefe {collapsed_at}"})
                continue
            offen = [(s, a) for s in seeds for a in ARMS
                     if (dataset, depth, a, int(s)) not in fertig]
            schluessel = f"lr_map_{dataset}_d{depth}"
            if not offen:
                print(f"\n  {dataset}, Tiefe {depth}: vollstaendig vorhanden -- "
                      f"uebersprungen")
            elif schluessel in rec.config:
                # Eine teilweise gerechnete Tiefe behaelt ihre Raten. Eine neue
                # Suche koennte eine andere Rate waehlen, und dann stuenden in
                # einer Zelle zwei Raten nebeneinander.
                lr_map = {a: float(v) for a, v in rec.config[schluessel].items()}
                print(f"\n  {dataset}, Tiefe {depth}: {len(offen)} Zellen offen, "
                      f"Lernraten aus der Aufzeichnung uebernommen {lr_map}")
            else:
                print(f"\n  {dataset}, Tiefe {depth} -- Lernratensuche")
                lr_map, rand = sweep_lr(dataset, depth, threads)
                rec.config[schluessel] = dict(lr_map)
                rec.config[f"lr_rand_{dataset}_d{depth}"] = dict(rand)
                edge = [a for a, v in rand.items() if v != "innen"]
                if edge:
                    print(f"    WARNUNG: am Gitterrand trotz Verlaengerung -> "
                          + ", ".join(f"{a} ({rand[a]})" for a in edge))
                rec.config[f"lr_edge_{dataset}_d{depth}"] = edge

            for i, seed in enumerate(seeds, 1):
                arme = [a for a in ARMS if (dataset, depth, a, int(seed)) not in fertig]
                for arm in arme:
                    rec.add(**_cell(arm, depth, seed, dataset, lr_map[arm],
                                    WIDTH, MAX_SAMPLES, EPOCHS, BATCH, threads))
                    fertig.add((dataset, depth, arm, int(seed)))
                if arme and (i % 5 == 0 or i == len(seeds)):
                    print(f"    seed {seed} fertig ({i}/{len(seeds)})")
                    # Regel 5, enger gefasst: Tiefe 16 kostet Stunden. Der
                    # dritte Anlauf hat mit Sichern je TIEFE zehn fertige
                    # Seeds von Tiefe 16 verloren.
                    rec.save()

            best = max(np.mean([r["balanced_accuracy"] for r in rec.records
                                if r["arm"] == a and int(r["depth"]) == depth
                                and r["dataset"] == dataset]) for a in ARMS)
            print(f"    bester Arm bei Tiefe {depth}: {best:.4f}")
            if best < CHANCE:
                collapsed_at = depth
                print(f"    KOLLAPS -- alle Arme unter {CHANCE}. "
                      f"Groessere Tiefen werden uebersprungen.")
            if offen:
                print(f"    Zwischenstand gesichert:")
                rec.save()
    return rec


def spearman_exakt(x, y) -> tuple:
    """Spearman mit EXAKTEM einseitigem Permutations-p-Wert.

    WARUM NICHT scipy: Bei wenigen Punkten ist scipys asymptotische Formel
    wertlos. Nachgemessen am 27. August mit vier Tiefen und perfekt monotonem
    Zuwachs: scipy meldet p=0,0000, der exakte Test ueber alle 4! = 24
    Anordnungen gibt p=0,0417 -- ein Faktor von ueber 400.

    Genau dieser Fehler hat Q-009 falsch berichtet ("Spearman ueber 3 Punkte:
    scipy meldet p=0,0000, der exakte Permutationstest gibt p=0,3333"), und
    diese Datei hatte ihn geerbt, weil sie am selben Tag geschrieben wurde.

    Die Vorregistrierung verlangt EINSEITIG positiv; scipy liefert zweiseitig.
    Auch das ist hier korrigiert.

    Rueckgabe: (rho, p_positiv, kleinstmoeglicher_p, p_negativ)

    p_negativ wird mitgeliefert, weil ein einseitig-positiver Test einen
    FALLENDEN Trend niemals anzeigen kann -- er meldet dort p=1,0. Ohne den
    zweiten Rand waere ein perfekt monotoner Abfall (rho=-1) unsichtbar, und
    genau das ist beim ersten Korrekturversuch am 27. August passiert.
    """
    from itertools import permutations

    from scipy import stats

    x, y = list(x), list(y)
    rho = float(stats.spearmanr(x, y).statistic)
    alle = [stats.spearmanr(x, list(perm)).statistic
            for perm in permutations(y)]
    n_ge = sum(1 for r in alle if r >= rho - 1e-12)
    p_eins = n_ge / len(alle)
    # Kleinstmoeglicher erreichbarer p-Wert: nur die perfekte Anordnung.
    p_min = sum(1 for r in alle if r >= max(alle) - 1e-12) / len(alle)
    p_neg = sum(1 for r in alle if r <= rho + 1e-12) / len(alle)
    return rho, float(p_eins), float(p_min), float(p_neg)


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    datasets = [str(d) for d in config["datasets"]]
    out: Dict[str, object] = {}

    def vals(arm: str, depth: int, ds: str) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm
                       and int(r["depth"]) == depth and r["dataset"] == ds),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print("RESEARCH ID Q-011 | Unitaritaet gegen Spektralnorm ueber die Tiefe")

    trends: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        depths = sorted({int(r["depth"]) for r in records if r["dataset"] == ds})
        print()
        print(f"--- {ds} " + "-" * max(4, 60 - len(ds)))
        head = (f'  {"Tiefe":>6} {"plain":>9} {"orthogonal":>11} {"spectral":>9} '
                f'{"orth-spec":>10} {"Siege":>7} {"p":>8}')
        print(head); print("  " + "-" * (len(head) - 2))
        ds_depths, ds_deltas = [], []
        for d in depths:
            p_, o, s = vals("plain", d, ds), vals("orthogonal", d, ds), vals("spectral", d, ds)
            if not len(o) or not len(s):
                continue
            best = max(p_.mean() if len(p_) else 0, o.mean(), s.mean())
            delta = float(o.mean() - s.mean())
            pv = float(stats.wilcoxon(o, s).pvalue)
            flag = "  <- KOLLAPS, ausgeschlossen" if best < config.get("chance_threshold", 0.3) else ""
            print(f'  {d:>6} {p_.mean() if len(p_) else float("nan"):>9.4f} '
                  f'{o.mean():>11.4f} {s.mean():>9.4f} {delta:>+10.4f} '
                  f'{int((o > s).sum()):>4}/{len(o):<2} {pv:>8.4f}{flag}')
            out[f"{ds}|d{d}"] = {"delta": delta, "p": pv,
                                 "orthogonal": float(o.mean()),
                                 "spectral": float(s.mean()),
                                 "collapsed": bool(best < config.get("chance_threshold", 0.3))}
            if best >= config.get("chance_threshold", 0.3):
                ds_depths.append(d); ds_deltas.append(delta)
        if len(ds_depths) >= 3:
            rho, pr, p_min, p_neg = spearman_exakt(ds_depths, ds_deltas)
            trends[ds] = {"rho": float(rho), "p": float(pr),
                          "p_min_erreichbar": p_min, "p_negativ": p_neg,
                          "depths": ds_depths, "deltas": ds_deltas}
            print(f'  Trend ueber die Tiefe: rho={rho:+.3f}  p={pr:.4f} '
                  f'(exakt, einseitig; kleinstmoeglich bei {len(ds_depths)} '
                  f'Punkten: {p_min:.4f})')
            if p_min > 0.025:
                print(f'  WARNUNG: bei nur {len(ds_depths)} auswertbaren Tiefen '
                      f'kann der Trendtest nach Holm-Korrektur kaum bestehen.')
        else:
            print("  zu wenige auswertbare Tiefen fuer einen Trend")

    # ---- vorregistrierte Familie ----
    pd_ = PRIMARY_DATASET
    tr = trends.get(pd_, {})
    p_trend = float(tr.get("p", 1.0))
    depths_ok = tr.get("depths", [])
    if depths_ok:
        dmax = max(depths_ok)
        o, s = vals("orthogonal", dmax, pd_), vals("spectral", dmax, pd_)
        p_deep = float(stats.wilcoxon(o, s).pvalue)
        eq = tost_paired(o, s, MARGIN)
    else:
        dmax, p_deep, eq = None, 1.0, {"p_tost": 1.0, "achieved": float("nan")}
    adj = holm([p_trend, p_deep])

    print()
    print("=" * 92)
    print(f"VORREGISTRIERTE FAMILIE auf {pd_}")
    print(f"  PRIMAER  Trend ueber die Tiefe   rho={tr.get('rho', float('nan')):+.3f}  "
          f"p={p_trend:.4f}  Holm={adj[0]:.4f}")
    print(f"  P2       Abstand bei Tiefe {dmax}   p={p_deep:.4f}  Holm={adj[1]:.4f}  "
          f"TOST p={eq['p_tost']:.4f}")
    out["primary"] = {"rho": tr.get("rho"), "p": p_trend, "p_holm": adj[0]}
    out["p2"] = {"depth": dmax, "p": p_deep, "p_holm": adj[1],
                 "p_tost": eq["p_tost"], "achieved": eq["achieved"]}

    repl = trends.get([d for d in datasets if d != pd_][0], {}) if len(datasets) > 1 else {}
    same_dir = bool(repl) and float(repl.get("rho", 0)) > 0

    # ---- Aufloesungspruefung (Regel 6) ----
    # Die Kollapsschwelle allein reicht nicht: sie sagt nur, dass nicht alles
    # zusammengebrochen ist, nicht dass der Aufbau Unterschiede trennen KANN.
    # B-001 hat gemessen, dass Orthogonalitaet gegen UNBESCHRAENKT +0,0139
    # bringt -- ein Effekt, der hier reproduzierbar sein muss. Tut er es nicht,
    # kann der Aufbau auch den feineren Unterschied zur Spektralnorm nicht
    # aufloesen, und kein Trend oben ist interpretierbar.
    res_ok, res_txt = None, ""
    if depths_ok:
        d1 = min(depths_ok)
        o1, pl1 = vals("orthogonal", d1, pd_), vals("plain", d1, pd_)
        if len(o1) and len(pl1) == len(o1):
            rp = float(stats.wilcoxon(o1, pl1).pvalue)
            # bool() ist hier nicht kosmetisch: o1.mean() > pl1.mean()
            # liefert np.bool_, und `np.False_ is False` ist in Python
            # FALSCH. Die Auflösungspruefung wurde dadurch berechnet,
            # gedruckt -- und beim Urteil stillschweigend uebergangen.
            # Gefunden am 27. August im synthetischen Vortest.
            res_ok = bool(o1.mean() > pl1.mean() and rp < 0.05)
            res_txt = (f"Aufloesungspruefung {'BESTANDEN' if res_ok else 'NICHT BESTANDEN'}: "
                       f"orthogonal gegen unbeschraenkt bei Tiefe {d1} "
                       f"{o1.mean() - pl1.mean():+.4f}, p={rp:.4f}."
                       + ("" if res_ok else " Der Aufbau reproduziert nicht "
                          "einmal den groben, aus B-001 bekannten Effekt -- der "
                          "feinere Unterschied zur Spektralnorm ist damit erst "
                          "recht nicht aufloesbar, und kein Trend oben zaehlt."))
            print()
            print(f"  {res_txt}")
            out["resolution_check"] = {"passed": bool(res_ok), "p": rp,
                                       "depth": d1, "text": res_txt}

    print()
    if res_ok is not None and not res_ok:
        v = ("HU -- Aufloesungspruefung durchgefallen. Der Aufbau reproduziert "
             "den bekannten groben Effekt nicht; kein Ergebnis dieses Laufs ist "
             "interpretierbar.")
    elif adj[0] < 0.05 and float(tr.get("rho", 0)) > 0 and adj[1] < 0.05:
        v = ("H1 -- der Vorteil der Orthogonalitaet waechst mit der Tiefe und ist "
             f"bei Tiefe {dmax} belegt. Spektralnormalisierung ersetzt Unitaritaet "
             "also NICHT, sobald die Abbildung oft genug angewandt wird. Befund 4 "
             "war bei einer Tiefe gemessen, bei der die Frage nicht auflösbar ist.")
        v += (" Die Wiederholung zeigt dieselbe Richtung."
              if same_dir else " Die Wiederholung zeigt sie NICHT -- nach dem "
              "vorregistrierten Kriterium reicht das nicht.")
    elif float(tr.get("p_negativ", 1.0)) * 2 < 0.05 and float(tr.get("rho", 0)) < 0:
        # Zweig ergaenzt am 27. August, VOR dem Lauf. Er fehlte, und ein
        # synthetischer Fall mit perfekt monotonem ABFALL (rho=-1,000) wurde
        # dadurch als "H0 -- kein Zuwachs, gleichwertig" gemeldet: eine Aussage,
        # die den Daten widerspricht. Dritter Fall desselben Fehlers nach Q-014
        # und Q-015 -- wer nur die erwarteten Ausgaenge kodiert, verliert den
        # interessanten.
        v = (f"H0- -- der Abstand SCHRUMPFT mit der Tiefe (rho={tr.get('rho'):+.3f}, "
             f"Holm p={adj[0]:.4f}). Das ist nicht 'kein Trend', sondern das "
             f"Gegenteil der Hypothese: Orthogonalitaet verliert ihren Vorsprung, "
             f"je oefter die Abbildung angewandt wird. Fuer Befund 4 ist das ein "
             f"staerkeres Argument als jeder Nullbefund -- und es gehoert als "
             f"eigenstaendiges Ergebnis berichtet, nicht als Ausbleiben eines "
             f"Effekts.")
        v += (" Die Wiederholung auf dem zweiten Datensatz zeigt dieselbe "
              "Richtung." if (repl and float(repl.get("rho", 0)) < 0) else
              " ACHTUNG: die Wiederholung zeigt sie NICHT. Der Trendtest sieht "
              "nur sechs Mittelwerte und ignoriert deren Streuung -- ohne "
              "Replikation ist eine Scheinordnung durch Rauschen nicht "
              "auszuschliessen. Dann nur als Beobachtung fuehren.")
    elif eq.get("p_tost", 1) < 0.05:
        v = (f"H0 -- kein Zuwachs mit der Tiefe; bei Tiefe {dmax} sind die beiden "
             f"gleichwertig innerhalb +-{MARGIN}. Befund 4 haelt, und zwar jetzt "
             f"aus dem richtigen Grund.")
    else:
        v = (f"HU -- unentschieden; die Daten tragen erst ab "
             f"+-{eq.get('achieved', float('nan')):.4f}.")
    print(f"BEWERTUNG: {v}")
    print("=" * 92)
    out["verdict"] = v
    out["trends"] = trends
    return out


# ---------------------------------------------------------------------------
# Tests der beiden Korrekturen vom 23. September -- auf synthetischen Daten.
# Eine Korrektur ist kein Beleg, sondern eine neue Behauptung.
# ---------------------------------------------------------------------------
def test_gitterverlaengerung() -> bool:
    """Verlaengerung in BEIDE Richtungen plus der flache Fall.

    Kuenstliche Genauigkeit, glockenfoermig in log(lr) mit bekanntem Optimum.
    Reichweite: Startgitter 0,0003 .. 0,01, drei Schritte mit Faktor 3 je
    Richtung, also 0,0003/27 = 1,1e-05 bis 0,01*27 = 0,27. Ein Optimum
    ausserhalb wird am Ende der Reichweite als Rand gemeldet (wie in
    `zeitreihen.test_gitterverlaengerung`: ohne Messung jenseits des Gipfels
    ist "Gipfel" nicht von "steigt noch" zu unterscheiden).
    """
    import math as _math
    global _cell
    echt = _cell
    faelle = [
        ("Optimum im Startgitter (0,003)", 0.003, "innen", 0.003),
        ("Optimum eine Verlaengerung OBEN (0,03)", 0.03, "innen", 0.03),
        ("Optimum eine Verlaengerung UNTEN (1e-04)", 1e-4, "innen", 0.0003 / 3),
        ("Optimum ausserhalb oben (300)", 300.0, "oben", 0.27),
        ("Optimum ausserhalb unten (1e-09)", 1e-9, "unten", 0.0003 / 27),
        ("flach: Kollaps, alle Raten 0,1000", None, "flach", 0.0003),
    ]
    print("=" * 92)
    print("TEST GITTERVERLAENGERUNG Q-011 -- Regel 1(b), beide Richtungen")
    print("=" * 92)
    alle_ok = True
    for name, opt, erwarteter_rand, erwartete_rate in faelle:
        aufrufe = [0]

        def gefaelscht(arm, depth, seed, dataset, lr, *rest, _opt=opt):
            aufrufe[0] += 1
            if _opt is None:
                return {"balanced_accuracy": 0.1}
            return {"balanced_accuracy": 1.0 / (1.0 + (_math.log10(lr / _opt)) ** 2)}
        _cell = gefaelscht
        try:
            best, rand = sweep_lr("mnist", 16, 1)
        finally:
            _cell = echt
        gewaehlt = best["orthogonal"]
        ok = (abs(_math.log10(gewaehlt / erwartete_rate)) < 1e-6
              and rand["orthogonal"] == erwarteter_rand)
        if opt is None:   # flach: keine einzige Verlaengerung
            ok = ok and aufrufe[0] == len(ARMS) * len(LR_GRID) * len(TUNING_SEEDS)
        alle_ok = alle_ok and ok
        print(f"  {name:<44} -> lr {gewaehlt:<10.6g} Rand {rand['orthogonal']:<6} "
              f"Aufrufe {aufrufe[0]:>3}  {'OK' if ok else 'FEHLER'}")
    print("TEST BESTANDEN" if alle_ok else "TEST FEHLGESCHLAGEN")
    return alle_ok


def test_wiederaufnahme() -> bool:
    """Wiederaufnahme auf synthetischen Rohdateien in einem Wegwerfordner.

    Aufbau (3 Seeds, Tiefen 1, 2, 3, nur mnist):
      A  gueltig: Tiefe 1 vollstaendig; Tiefe 2 Seeds 1-2 vollstaendig, Seed 3
         nur `plain`; eine Zelle (Tiefe 2, spectral, Seed 2) mit FALSCHER Rate
      B  gueltig, neuer: Tiefe 1 noch einmal, identisch -> keine Doppelzeilen
      C  ohne Suchbudget, nicht zugelassen (wie ein zurueckkopierter Vorlauf)
      D  anderes Budget (epochs 5)
      Unterordner vorlaeufe/ mit einer weiteren Datei -> wird nicht gelesen
    Erwartet: Tiefe 1 ohne jeden Aufruf; Tiefe 2 OHNE neue Lernratensuche und
    genau drei Zellen mit den GESPEICHERTEN Raten; Tiefe 3 mit Suche und neun
    Zellen; am Ende 27 eindeutige Zellen. Gegenprobe weiter=False: alles neu.
    """
    import tempfile

    from qire_ai import result_store
    global _cell
    echt_cell, echt_dir = _cell, result_store.RESULTS_DIR
    lr_d2 = {"plain": 0.001, "orthogonal": 0.003, "spectral": 0.01}

    def zeile(arm, depth, seed, lr):
        return {"arm": arm, "depth": depth, "seed": seed, "dataset": "mnist",
                "lr": lr, "balanced_accuracy": 0.9 - 0.001 * depth
                + 0.0001 * seed + 0.001 * ARMS.index(arm)}

    def datei(ordner, name, records, lr_maps, budget=True, epochs=EPOCHS):
        cfg = {**PASSEND, "epochs": epochs, "datasets": ["mnist"],
               **{f"lr_map_mnist_d{d}": m for d, m in lr_maps.items()}}
        if budget:
            cfg["suchbudget"] = dict(SUCHBUDGET)
        (ordner / name).write_text(json.dumps(
            {"experiment_id": "Q-011", "config": cfg, "records": records,
             "n_records": len(records), "analysis": {}}), encoding="utf-8")

    def lauf(ordner, weiter):
        aufrufe: List[Tuple[str, int, int, float]] = []

        def gefaelscht(arm, depth, seed, dataset, lr, *rest):
            aufrufe.append((arm, depth, seed, lr))
            return zeile(arm, depth, seed, lr) if seed not in TUNING_SEEDS else \
                {"balanced_accuracy": 1.0 / (1.0 + np.log10(lr / 0.003) ** 2)}
        global _cell
        _cell = gefaelscht
        try:
            rec = run(datasets=("mnist",), depths=(1, 2, 3), seeds=(1, 2, 3),
                      threads=1, weiter=weiter, verzeichnis=ordner)
        finally:
            _cell = echt_cell
        return rec, aufrufe

    print("=" * 92)
    print("TEST WIEDERAUFNAHME Q-011 -- synthetische Rohdateien")
    print("=" * 92)
    with tempfile.TemporaryDirectory() as tmp:
        ordner = Path(tmp)
        result_store.RESULTS_DIR = ordner      # Zwischenstaende hierher
        try:
            d1 = [zeile(a, 1, s, 0.003) for s in (1, 2, 3) for a in ARMS]
            d2 = [zeile(a, 2, s, lr_d2[a]) for s in (1, 2) for a in ARMS]
            d2 = [z if not (z["arm"] == "spectral" and z["seed"] == 2)
                  else {**z, "lr": 0.003} for z in d2]           # falsche Rate
            d2.append(zeile("plain", 2, 3, lr_d2["plain"]))
            m1 = {a: 0.003 for a in ARMS}
            datei(ordner, "Q-011__20990101_000001.json", d1 + d2, {1: m1, 2: lr_d2})
            datei(ordner, "Q-011__20990101_000002.json", d1, {1: m1})
            datei(ordner, "Q-011__20990101_000003.json",
                  [zeile(a, 3, s, 0.003) for s in (1, 2, 3) for a in ARMS],
                  {3: m1}, budget=False)
            datei(ordner, "Q-011__20990101_000004.json",
                  [zeile(a, 3, s, 0.003) for s in (1, 2, 3) for a in ARMS],
                  {3: m1}, epochs=5)
            (ordner / "vorlaeufe").mkdir()
            datei(ordner / "vorlaeufe", "Q-011__20990101_000005.json",
                  [zeile(a, 3, s, 0.003) for s in (1, 2, 3) for a in ARMS], {3: m1})

            rec, aufrufe = lauf(ordner, weiter=True)
            mess = [a for a in aufrufe if a[2] not in TUNING_SEEDS]
            such = [a for a in aufrufe if a[2] in TUNING_SEEDS]
            schluessel = [(r["dataset"], int(r["depth"]), r["arm"], int(r["seed"]))
                          for r in rec.records]
            wa = rec.config["wiederaufnahme"]
            pruef = [
                ("Tiefe 1: kein einziger Aufruf",
                 not any(a[1] == 1 for a in aufrufe)),
                ("Tiefe 2: keine neue Lernratensuche",
                 not any(a[1] == 2 for a in such)),
                ("Tiefe 2: genau die drei offenen Zellen, mit gespeicherten Raten",
                 sorted((a[0], a[2], a[3]) for a in mess if a[1] == 2)
                 == sorted([("orthogonal", 3, 0.003), ("spectral", 3, 0.01),
                            ("spectral", 2, 0.01)])),
                ("Tiefe 3: Suche gelaufen und neun Zellen gerechnet",
                 bool([a for a in such if a[1] == 3])
                 and len([a for a in mess if a[1] == 3]) == 9),
                ("27 eindeutige Zellen, keine Doppelzeilen",
                 len(schluessel) == 27 and len(set(schluessel)) == 27),
                ("gelesen: genau die beiden gueltigen Dateien",
                 wa["gelesen"] == ["Q-011__20990101_000001.json",
                                   "Q-011__20990101_000002.json"]),
                ("verworfen: C (kein Budget), D (epochs), 1 Zelle (falsche Rate)",
                 sorted(v[0][:31] for v in wa["verworfen"])
                 == sorted(["Q-011__20990101_000003.json",
                            "Q-011__20990101_000004.json",
                            "Zelle mnist/Tiefe 2/spectral/2"])),
                ("keine Widersprueche zwischen A und B", wa["widersprueche"] == 0),
            ]
            rec0, aufrufe0 = lauf(ordner, weiter=False)
            pruef.append(("Gegenprobe weiter=False: alle 27 Zellen neu",
                          len([a for a in aufrufe0 if a[2] not in TUNING_SEEDS]) == 27))
        finally:
            result_store.RESULTS_DIR = echt_dir
    alle_ok = True
    for name, ok in pruef:
        alle_ok = alle_ok and bool(ok)
        print(f"  {name:<66} {'OK' if ok else 'FEHLER'}")
    print("TEST BESTANDEN" if alle_ok else "TEST FEHLGESCHLAGEN")
    return alle_ok


def test_echte_rohdaten() -> bool:
    """Liest die ECHTEN Rohdaten (nur lesen, kein Training) und zeigt, was eine
    Wiederaufnahme heute uebernehmen wuerde."""
    alt = lade_fertige()
    tiefen = sorted({(r["dataset"], int(r["depth"])) for r in alt["records"]})
    print("=" * 92)
    print("TROCKENLAUF WIEDERAUFNAHME auf den echten Rohdaten (kein Training)")
    print(f"  gelesen: {alt['gelesen']}")
    for v in alt["verworfen"]:
        print(f"  verworfen: {v[0]} -- {v[1]}")
    print(f"  uebernommen: {len(alt['records'])} Zellen, {tiefen}")
    ok = len(alt["records"]) == 225 and not alt["widersprueche"]
    print(f"  225 Zellen, keine Widersprueche: {'OK' if ok else 'FEHLER'}")
    return ok


if __name__ == "__main__":
    import sys

    befehl = sys.argv[1] if len(sys.argv) > 1 else ""
    if befehl == "test":
        ergebnisse = [test_gitterverlaengerung(), test_wiederaufnahme(),
                      test_echte_rohdaten()]
        raise SystemExit(0 if all(ergebnisse) else 1)
    if befehl not in ("", "neu"):
        raise SystemExit(f"unbekannter Befehl {befehl!r} -- test | neu | (leer)")
    # Ohne Befehl: Lauf MIT Wiederaufnahme. "neu": ohne (faengt bei null an).
    recorder = run(weiter=(befehl != "neu"))
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.unitarity_depth '
              "import analyse; analyse('Q-011')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
