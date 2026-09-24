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

Run: python -m quantum_sim.unitarity_depth
"""

from typing import Dict, List, Sequence

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
ERWEITERUNGEN = 3          # wie oft nach oben verlaengert werden darf,
                           # wenn ein Arm auf dem Rand landet (Regel 1)
WIDTH = 4096                                 # wie B-007, damit vergleichbar
MAX_SAMPLES = 6000
EPOCHS = 10
BATCH = 128
CHANCE = 0.3                                 # Schwelle der Aufloesungspruefung


def sweep_lr(dataset: str, depth: int, threads: int) -> Dict[str, float]:
    """Lernrate je Arm UND Tiefe, mit automatischer Verlaengerung nach oben.

    Regel 1 verlangt, dass das Gitter das Optimum ENTHAELT. Der erste Anlauf am
    27. August wurde nach zwei Tiefen abgebrochen, weil bei Tiefe 2 genau die
    ungluecklichste Kombination auftrat:

        orthogonal -> 0,01 (oberer Rand, stieg noch)
        spectral   -> 0,003 (innen, Optimum gefunden)

    Der Mechanismus-Arm war untertuniert, der Kontroll-Arm nicht. Der
    Primaerabstand (orthogonal - spectral) waere damit systematisch zu KLEIN
    gemessen worden -- also gegen die eigene Hypothese, und genau das ist der
    "blinde Aufbau", den das Protokoll ausschliessen soll.

    Landet ein Arm auf dem oberen Rand, wird das Gitter deshalb verdoppelt und
    nachgemessen, bis zu ERWEITERUNGEN Mal. Uebernommen aus der T-002-Suche,
    wo sich dasselbe Verfahren bewaehrt hat (dense: 0,0024 -> 0,0048 -> 0,0096,
    Optimum am Ende innen).
    """
    best: Dict[str, float] = {}
    for arm in ARMS:
        means: Dict[float, float] = {}
        raten = list(LR_GRID)
        for runde in range(ERWEITERUNGEN + 1):
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
            bester = max(means, key=means.get)
            if bester != max(means) or runde == ERWEITERUNGEN:
                break
            naechste = max(means) * 3
            print(f"      {arm}: Optimum am oberen Rand ({bester:g}) -- "
                  f"verlaengere auf {naechste:g}")
            raten = [naechste]
        best[arm] = max(means, key=means.get)
        print(f"    Tiefe {depth:>2} / {arm:<11} -> lr {best[arm]}  "
              + "  ".join(f"{k:g}:{means[k]:.4f}" for k in sorted(means)))
    return best


def run(datasets: Sequence[str] = DATASETS, depths: Sequence[int] = DEPTHS,
        seeds: Sequence[int] = EVAL_SEEDS, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-011",
        description=("Waechst der Vorteil von Orthogonalitaet gegenueber "
                     "Spektralnormalisierung mit der Tiefe? Aufbau unveraendert "
                     "aus B-007 uebernommen, nur die Tiefe wird durchgefahren "
                     "(1, 2, 3, 4, 8, 16). 15 Seeds, Lernrate je Arm und Tiefe."),
        config={"arms": list(ARMS), "depths": list(depths),
                "datasets": list(datasets), "seeds": list(seeds),
                "width": WIDTH, "lr_grid": list(LR_GRID),
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
            print(f"\n  {dataset}, Tiefe {depth} -- Lernratensuche")
            lr_map = sweep_lr(dataset, depth, threads)
            rec.config[f"lr_map_{dataset}_d{depth}"] = dict(lr_map)
            edge = [a for a, v in lr_map.items() if v == min(LR_GRID)]
            if edge:
                print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")
            rec.config[f"lr_edge_{dataset}_d{depth}"] = edge

            for i, seed in enumerate(seeds, 1):
                for arm in ARMS:
                    rec.add(**_cell(arm, depth, seed, dataset, lr_map[arm],
                                    WIDTH, MAX_SAMPLES, EPOCHS, BATCH, threads))
                if i % 5 == 0 or i == len(seeds):
                    print(f"    seed {seed} fertig ({i}/{len(seeds)})")

            best = max(np.mean([r["balanced_accuracy"] for r in rec.records
                                if r["arm"] == a and r["depth"] == depth
                                and r["dataset"] == dataset]) for a in ARMS)
            print(f"    bester Arm bei Tiefe {depth}: {best:.4f}")
            if best < CHANCE:
                collapsed_at = depth
                print(f"    KOLLAPS -- alle Arme unter {CHANCE}. "
                      f"Groessere Tiefen werden uebersprungen.")
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


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.unitarity_depth '
              "import analyse; analyse('Q-011')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
