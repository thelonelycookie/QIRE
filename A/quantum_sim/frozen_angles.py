from __future__ import annotations

"""RESEARCH ID: Q-018 -- Traegt es ueberhaupt etwas bei, die Winkel zu trainieren?

WOHER DIE FRAGE KOMMT
Q-014 (10. August) sollte messen, ob die Gewichtsteilung Qualitaet kostet. Der
Lauf ist an seiner eigenen Aufloesungspruefung gescheitert, hat dabei aber etwas
anderes gezeigt: Derselbe Block mit **eingefrorenen Zufallswinkeln** war auf
MNIST nicht von der trainierten Fassung zu unterscheiden (+0,0018, p=0,1205)
und auf Fashion-MNIST **besser** (-0,0211, 14 von 15 Seeds, p=0,0009).

Zusammen mit den beiden TOST-Gleichwertigkeiten vom 5. August -- Verschraenkung
traegt nichts bei, Tiefe traegt nichts bei -- ergaebe das eine geschlossene und
unbequeme Aussage: Von der Schaltung wirkt die feste Struktur, und der
trainierbare Teil ist der Teil, der nichts tut.

WARUM Q-014 DAS NICHT BEANTWORTEN KONNTE
In jedem Arm sass ein trainierbarer Encoder 784 -> 256. Der kann sich an jede
feste nachgeschaltete Drehung anpassen und uebernimmt genau die Arbeit, die der
Block tun sollte. Solange er frei trainiert, ist die Frage nicht stellbar.

WAS HIER ANDERS IST
1. **Kein Encoder.** Die Haufen-Aufgabe aus `isotime_accuracy` erzeugt Eingaben
   der Dimension 256 = 2^8. Sie gehen direkt in den 8-Qubit-Block. Zwischen
   Eingabe und Kopf steht damit nichts als der Block selbst.
2. **Lernratengitter bis 1,0.** In Q-014 sassen auf Fashion-MNIST zwei Arme auf
   dem oberen Rand 0,3 und stiegen dort noch.
3. **Aufgabe mit Kapazitaetsbedarf.** Die Haufenzahl steuert, wie viele
   Richtungen noetig sind; MNIST braucht laut B-014 nur rund sieben und kann
   den Unterschied deshalb gar nicht zeigen. Die Stufe wird vor dem Lauf
   kalibriert (siehe `vortest()`), nicht geraten.

DIE ARME (8 Qubits, 256 Amplituden, 5 Lagen, CNOT-Ring)
  tied_trained      80 trainierbar   geteilte Winkel, trainiert -- die Schaltung
  tied_frozen        0 trainierbar   dieselbe Struktur, Winkel EINGEFROREN
  untied_trained 10.240 trainierbar   ungeteilt, trainiert
  frozen_4ch         0 trainierbar   wie tied_frozen, aber nur 4 der 8 Kanaele
                                      erreichen den Kopf  -> AUFLOESUNGSPRUEFUNG

VORREGISTRIERUNG
  Balancierte Genauigkeit, Lernrate je Arm auf eigenen Tuning-Seeds, 15 Seeds,
  zwei Schwierigkeitsstufen. Primaerstufe ist die kalibrierte.

  PRIMAER  tied_trained gegen tied_frozen: TOST auf Gleichwertigkeit innerhalb
           +-0,005. Gleichwertig hiesse: das Training der Winkel traegt nichts
           bei, auch wenn kein Encoder mehr aushelfen kann.
  P2       untied_trained gegen tied_frozen. Beantwortet, ob 10.240 trainierbare
           Winkel schaffen, woran 80 scheitern -- also ob es an der Zahl der
           trainierbaren Parameter liegt oder am Training ueberhaupt.
  Holm ueber genau diese beiden.

  H1 : PRIMAER gleichwertig -> das Training der Winkel traegt nichts bei. Der
       Befund aus Q-014 haelt unter verschaerften Bedingungen. Der Block ist
       eine feste Transformation, und das ist fuer die T-Serie die guenstigste
       Nachricht: keine Rueckwaertsrichtung noetig, Konstanten faltbar.
  H1+: tied_frozen ist signifikant BESSER -> Training schadet. Dann ist die
       zufaellige orthogonale Mischung nicht nur ausreichend, sondern der
       Gradientenabstieg verschlechtert sie aktiv.
  H0 : tied_trained ist signifikant besser -> das Training traegt sehr wohl bei,
       und der Q-014-Befund war ein Artefakt des freien Encoders. Dann ist er
       zurueckzunehmen, und die T-Serie braucht den Rueckwaertslauf.
  HU : weder noch -> erreichte Schranke berichten.

  Faellt ein Ergebnis in H1 und H1+ zugleich (signifikant, aber kleiner als
  +-0,005), gilt die Gleichwertigkeitsaussage; die Richtung wird nur
  beschreibend genannt. So festgelegt vor dem Lauf.

  AUFLOESUNGSPRUEFUNG: `tied_frozen` muss sich von `frozen_4ch` **trennen**.
  Beide haben null trainierbare Blockparameter und dieselbe feste Drehung; sie
  unterscheiden sich nur darin, dass bei `frozen_4ch` die Haelfte der
  Ausgabekanaele verworfen wird. Trennen sie sich nicht, ist der
  8-Kanal-Engpass nicht bindend -- dann kann der Aufbau Unterschiede in der
  Blockqualitaet grundsaetzlich nicht sehen, und kein Vergleich oben ist
  interpretierbar. Verlangt wird TRENNUNG, nicht Richtung: faellt sie
  umgekehrt aus, ist das ein Ergebnis und kein Messversagen (Lehre aus Q-015).

Run:  python -m quantum_sim.frozen_angles           (Vortest + Lauf)
      python -m quantum_sim.frozen_angles vortest   (nur Kalibrierung)
"""

import math
import sys
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import ButterflyQuantumLayer

NUM_QUBITS = 8
INPUT_DIM = 2 ** NUM_QUBITS        # 256 -- passt exakt auf 8 Qubits, kein Encoder
INTRINSIC_DIM = 64
LAYERS = 5
N_TRAIN, N_TEST = 6000, 3000
EPOCHS = 12
BATCH = 64

# (geteilt, eingefroren, Zahl der Ausgabekanaele)
SPECS = {
    "tied_trained":   (True,  False, NUM_QUBITS),
    "tied_frozen":    (True,  True,  NUM_QUBITS),
    "untied_trained": (False, False, NUM_QUBITS),
    "frozen_4ch":     (True,  True,  NUM_QUBITS // 2),
}
ARMS = tuple(SPECS)
REGISTERED = ("PRIMAER", "P2")

EVAL_SEEDS = tuple(range(7101, 7116))      # 15 Seeds -> kleinstes Wilcoxon-p 6.1e-5
TUNING_SEEDS = (7151, 7152)
# Obergrenze 1,0: in Q-014 sassen auf Fashion-MNIST zwei Arme auf dem Rand 0,3
# und stiegen dort noch.
LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

# Kandidatengitter fuer den Vortest, ueber ZWEI Achsen.
#
# Erster Anlauf am 10. August benutzte nur die Haufenzahlen (64, 256, 1024) aus
# `isotime_accuracy`. Dort lief ein Modell der Breite 16.384; fuer einen
# 8-Kanal-Block sind diese Stufen unbrauchbar -- gemessen wurden 0,2892 /
# 0,1808 / 0,1268 bei Rateniveau 0,10, das Modell lernte also praktisch nichts.
# Die Aufgabe war zu SCHWER, nicht zu leicht. Der Engpass ist die Zahl der
# Klassen gegen die Zahl der Kanaele, deshalb wird sie mitkalibriert.
# Geaendert vor jedem Messlauf; es existieren keine Q-018-Rohdaten.
KLASSEN_KANDIDATEN = (3, 4, 6)
CLUSTER_KANDIDATEN = (32, 128, 512)


def make_data(seed: int, clusters: int, num_classes: int):
    """Haufen mit zufaellig zugewiesenen Klassen, nach
    `isotime_accuracy.make_data`. Die Haufenzahl steuert, wie viele Richtungen
    die Aufgabe braucht -- genau die Achse, an der MNIST zu leicht war. Die
    Klassenzahl steuert, wie stark der 8-Kanal-Engpass bindet."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(INTRINSIC_DIM, INPUT_DIM, generator=g) / math.sqrt(INTRINSIC_DIM)
    centres = torch.randn(clusters, INTRINSIC_DIM, generator=g) * 2.0
    labels = torch.randint(0, num_classes, (clusters,), generator=g)
    out = []
    for n in (N_TRAIN, N_TEST):
        c = torch.randint(0, clusters, (n,), generator=g)
        z = centres[c] + torch.randn(n, INTRINSIC_DIM, generator=g) * 0.55
        out += [z @ basis, labels[c]]
    return out


class BlockModel(nn.Module):
    """Eingabe (256) -> Butterfly-Block -> Kopf. KEIN Encoder: damit ist der
    Block das einzige, was zwischen Daten und Kopf steht."""

    def __init__(self, arm: str, num_classes: int):
        super().__init__()
        tie, frozen, channels = SPECS[arm]
        self.channels = channels
        self.block = ButterflyQuantumLayer(NUM_QUBITS, LAYERS, entangle=True,
                                           tie_weights=tie)
        if frozen:
            for p in self.block.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(channels, num_classes)

    def block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters())

    def trainable_block_parameters(self) -> int:
        return sum(p.numel() for p in self.block.parameters() if p.requires_grad)

    def forward(self, x):
        return self.head(self.block(x)[:, :self.channels])


def _cell(arm: str, clusters: int, num_classes: int, seed: int, lr: float,
          threads: int, epochs: int = EPOCHS,
          n_train: int = N_TRAIN) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    xtr, ytr, xte, yte = make_data(seed, clusters, num_classes)
    xtr, ytr = xtr[:n_train], ytr[:n_train]
    model = BlockModel(arm, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = xtr.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte).argmax(-1)
        classes = torch.unique(yte)
        bal = float(np.mean([float((pred[yte == c] == c).float().mean())
                             for c in classes]))
    tie, frozen, channels = SPECS[arm]
    return {"arm": arm, "clusters": clusters, "num_classes": num_classes,
            "seed": seed, "lr": lr,
            "tied": tie, "frozen": frozen, "channels": channels,
            "balanced_accuracy": bal,
            "accuracy": float((pred == yte).float().mean()),
            "block_parameters": model.block_parameters(),
            "trainable_block_parameters": model.trainable_block_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def vortest(threads: int = 6) -> Tuple[int, int, int]:
    """Kalibriert die Schwierigkeit VOR dem Lauf, ueber zwei Achsen.

    Q-014 ist daran gescheitert, dass die Auflösungspruefung nicht feuern
    konnte, und das war vorher nicht geprueft worden. Hier wird genau das
    geprueft, und zwar an drei Bedingungen zugleich:

      (1) `tied_frozen` muss sich von `frozen_4ch` abheben -- sonst ist der
          Kanal-Engpass nicht bindend und der Aufbau ist blind.
      (2) `tied_frozen` muss deutlich ueber Rateniveau liegen -- sonst lernt
          das Modell gar nichts und der Vergleich misst Rauschen.
      (3) `tied_frozen` darf nicht gesaettigt sein -- sonst ist die Aufgabe zu
          leicht, der Fall von Q-014 und Q-009.

    Billig gehalten: 2 Seeds, halbe Daten, halbe Epochen, feste mittlere Rate.
    Es geht um Richtung und Groessenordnung, nicht um den Messwert.

    Rueckgabe: (Klassenzahl, Primaerstufe, Zweitstufe).
    """
    print("\n" + "=" * 94)
    print("VORTEST -- welche Aufgabe macht den Kanal-Engpass bindend?")
    print("Bedingungen: Abstand zu frozen_4ch > 0,02 | deutlich ueber Zufall")
    print("             (>= Zufall + 0,15) | nicht gesaettigt (<= 0,95)")
    print("=" * 94)
    head = (f'  {"Klassen":>8} {"Zufall":>7} {"Haufen":>7} {"tied_frozen":>13} '
            f'{"frozen_4ch":>12} {"Abstand":>9} {"brauchbar":>10}')
    print(head)
    print("  " + "-" * (len(head) - 2))
    tauglich = []
    for k in KLASSEN_KANDIDATEN:
        zufall = 1.0 / k
        for c in CLUSTER_KANDIDATEN:
            werte = {}
            for arm in ("tied_frozen", "frozen_4ch"):
                werte[arm] = float(np.mean([
                    _cell(arm, c, k, s, 0.03, threads, epochs=6,
                          n_train=N_TRAIN // 2)["balanced_accuracy"]
                    for s in TUNING_SEEDS]))
            abstand = werte["tied_frozen"] - werte["frozen_4ch"]
            ok = (abstand > 0.02 and werte["tied_frozen"] >= zufall + 0.15
                  and werte["tied_frozen"] <= 0.95)
            if ok:
                tauglich.append((abstand, k, c))
            print(f'  {k:>8} {zufall:>7.3f} {c:>7} {werte["tied_frozen"]:>13.4f} '
                  f'{werte["frozen_4ch"]:>12.4f} {abstand:>+9.4f} '
                  f'{"ja" if ok else "nein":>10}')
    if not tauglich:
        print("\n  KEINE Kombination erfuellt die Bedingungen. Der Lauf wird NICHT")
        print("  gestartet -- genau der Fehler, der Q-014 vier Stunden gekostet hat.")
        raise SystemExit(1)

    # --- Stufe 2: Saettigungspruefung ueber ALLE Arme ---------------------
    #
    # Nachgetragen am 10. August, nachdem der erste Q-018-Lauf lief. Stufe 1
    # prueft nur `tied_frozen` und `frozen_4ch`, weil die die Auflösungspruefung
    # bilden. Damit blieb ungeprueft, ob die TRAINIERTEN Arme an die Decke
    # stossen -- und genau das ist passiert: `untied_trained` stand bei 32
    # Haufen auf 1,0000 ueber fast das ganze Lernratengitter. Der Primaertest
    # blieb brauchbar (der Abstand 0,9978 zu 0,5386 ist echt und wird durch die
    # Decke hoechstens unterschaetzt), aber P2 verglich einen gesaettigten Arm
    # und war damit entwertet.
    #
    # Eine Kalibrierung, die nur die Kontrolle prueft und nicht die Arme, ist
    # halb gemacht.
    print("\n  Saettigungspruefung ueber ALLE Arme (bester Wert aus drei Raten)")
    kandidaten = sorted(tauglich, reverse=True)      # bester Abstand zuerst
    proberaten = (0.03, 0.1, 0.3)
    gewaehlt = None
    for abstand, k, c in kandidaten:
        spitzen = {}
        for arm in ARMS:
            spitzen[arm] = max(
                float(np.mean([
                    _cell(arm, c, k, s, lr, threads, epochs=6,
                          n_train=N_TRAIN // 2)["balanced_accuracy"]
                    for s in TUNING_SEEDS])) for lr in proberaten)
        gesaettigt = [a for a, v in spitzen.items() if v > 0.95]
        print(f'    {k} Klassen / {c:>4} Haufen: '
              + "  ".join(f"{a}={spitzen[a]:.4f}" for a in ARMS))
        if gesaettigt:
            print(f"      -> verworfen, gesaettigt: {', '.join(gesaettigt)}")
            continue
        print(f"      -> brauchbar, kein Arm ueber 0,95")
        gewaehlt = (k, c)
        break
    if gewaehlt is None:
        print("\n  ALLE Kandidaten haben mindestens einen gesaettigten Arm.")
        print("  Der Lauf wird NICHT gestartet -- eine gesaettigte Aufgabe kann")
        print("  Unterschiede zwischen den Armen grundsaetzlich nicht zeigen.")
        raise SystemExit(1)

    klassen, primaer = gewaehlt
    # Zweitstufe: dieselbe Klassenzahl, naechsthoehere Haufenzahl (haerter);
    # gibt es keine, die naechstniedrigere.
    hoeher = [c for c in CLUSTER_KANDIDATEN if c > primaer]
    zweit = min(hoeher) if hoeher else max(c for c in CLUSTER_KANDIDATEN if c != primaer)
    print(f"\n  GEWAEHLT: {klassen} Klassen, Primaerstufe {primaer} Haufen, "
          f"Zweitstufe {zweit} Haufen")
    print("=" * 94)
    return klassen, primaer, zweit


def sweep_lr(clusters: int, num_classes: int, threads: int) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for arm in ARMS:
        means = {lr: float(np.mean([
            _cell(arm, clusters, num_classes, s, lr, threads, epochs=6,
                  n_train=N_TRAIN // 2)["balanced_accuracy"]
            for s in TUNING_SEEDS])) for lr in LR_GRID}
        best[arm] = max(means, key=means.get)
        print(f"    {clusters:>6} Haufen / {arm:<16} -> lr {best[arm]}  "
              + "  ".join(f"{k}:{v:.4f}" for k, v in means.items()))
    return best


def run(primaer: int, zweitstufe: int, num_classes: int,
        seeds: Sequence[int] = EVAL_SEEDS, threads: int = 6) -> Recorder:
    stufen = (primaer, zweitstufe)
    rec = Recorder(
        experiment_id="Q-018",
        description=("Traegt das Training der Schaltungswinkel etwas bei, wenn "
                     "kein Encoder mehr aushelfen kann? Kein Encoder, Aufgabe "
                     "mit kalibriertem Kapazitaetsbedarf, 15 Seeds."),
        config={"arms": list(ARMS), "clusters": list(stufen),
                "primary_clusters": primaer, "num_classes": num_classes,
                "chance_level": 1.0 / num_classes, "seeds": list(seeds),
                "num_qubits": NUM_QUBITS, "layers": LAYERS,
                "input_dim": INPUT_DIM, "encoder": "keiner -- Eingabe ist 2^8",
                "specs": {k: {"tied": v[0], "frozen": v[1], "channels": v[2]}
                          for k, v in SPECS.items()},
                "lr_grid": list(LR_GRID), "epochs": EPOCHS, "batch_size": BATCH,
                "n_train": N_TRAIN, "n_test": N_TEST, "margin": MARGIN,
                "torch_threads": threads, "test_metric": "balanced_accuracy",
                "primary": "tied_trained gleichwertig zu tied_frozen",
                "p2": "untied_trained gegen tied_frozen",
                "holm_family": list(REGISTERED),
                "resolution_check": "tied_frozen muss sich von frozen_4ch trennen",
                "vorlauf": "Q-014 -- dort fiel die Auflösungspruefung durch",
                "erwartung_vorher": ("H1 -- Q-014 zeigte eingefroren gleichwertig "
                                     "(MNIST) bis besser (Fashion). Ohne Encoder "
                                     "koennte sich das umkehren; genau dafuer "
                                     "ist der Lauf da.")},
    )
    for clusters in stufen:
        print(f"\n  Lernratensuche fuer {clusters} Haufen")
        lr_map = sweep_lr(clusters, num_classes, threads)
        rec.config[f"lr_map_{clusters}"] = dict(lr_map)
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_{clusters}"] = edge
        if edge:
            print(f"    WARNUNG: am Gitterrand -> {', '.join(edge)}")
        for i, seed in enumerate(seeds, 1):
            for arm in ARMS:
                rec.add(**_cell(arm, clusters, num_classes, seed,
                                lr_map[arm], threads))
            print(f"    {clusters} Haufen seed {seed} fertig ({i}/{len(seeds)})")
        print(f"    Zwischenstand nach {clusters} Haufen gesichert:")
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    stufen = [int(c) for c in config["clusters"]]
    primaer = int(config["primary_clusters"])
    out: Dict[str, object] = {}

    def vals(arm: str, c: int) -> np.ndarray:
        rows = sorted((r for r in records
                       if r["arm"] == arm and int(r["clusters"]) == c),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    print()
    print(f"RESEARCH ID Q-018 | Eingefrorene gegen trainierte Winkel, "
          f"{NUM_QUBITS} Qubits, kein Encoder")
    for c in stufen:
        print()
        marke = "  <== PRIMAERSTUFE" if c == primaer else ""
        print(f"--- {c} Haufen " + "-" * max(4, 48 - len(str(c))) + marke)
        head = (f'  {"Arm":>16} {"trainierbar":>12} {"Kanaele":>8} '
                f'{"balanced":>10} {"std":>8} {"lr":>6}')
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, c)
            if not len(v):
                continue
            r0 = next(r for r in records
                      if r["arm"] == arm and int(r["clusters"]) == c)
            print(f'  {arm:>16} {r0["trainable_block_parameters"]:>12,} '
                  f'{r0["channels"]:>8} {v.mean():>10.4f} '
                  f'{v.std(ddof=1):>8.4f} {r0["lr"]:>6}')

    fam = [("tied_trained", "tied_frozen", "PRIMAER"),
           ("untied_trained", "tied_frozen", "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, primaer), vals(b, primaer)
        p = float(stats.wilcoxon(va, vb).pvalue)
        raw.append(p)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN), int((va > vb).sum()), len(va)))
    adj = holm(raw)

    print()
    print("=" * 94)
    print(f"VORREGISTRIERTE FAMILIE auf {primaer} Haufen, Grenze +-{MARGIN}")
    for (a, b, tag, delta, p, eq, w, nn_), p_h in zip(cells, adj):
        pa = next(r["trainable_block_parameters"] for r in records if r["arm"] == a)
        pb = next(r["trainable_block_parameters"] for r in records if r["arm"] == b)
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} ({pa:,} trainierbar) vs {b} ({pb:,}): '
              f'{delta:>+8.4f}  {w}/{nn_}  p={p:.4f}  Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"],
                    "trainable_a": pa, "trainable_b": pb}

    # --- Auflösungspruefung: verlangt TRENNUNG, nicht Richtung (Lehre Q-015) ---
    fv, cv = vals("tied_frozen", primaer), vals("frozen_4ch", primaer)
    c_delta = float(fv.mean() - cv.mean())
    c_p = float(stats.wilcoxon(fv, cv).pvalue)
    c_ok = c_p < 0.05
    richtung = ("8 Kanaele besser, wie erwartet" if c_delta > 0
                else "4 KANAELE BESSER -- unerwartet")
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"tied_frozen gegen frozen_4ch {c_delta:+.4f}, p={c_p:.4f} -- {richtung}."
            + ("" if c_ok else " Der Kanal-Engpass ist nicht bindend -- der Aufbau "
               "kann Unterschiede in der Blockqualitaet nicht sehen, und kein "
               "Vergleich oben ist interpretierbar.")
            + (" Die Richtung ist der Erwartung entgegengesetzt; das ist ein "
               "Ergebnis, kein Messversagen." if c_ok and c_delta < 0 else ""))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p,
                               "delta": c_delta, "text": ctxt}

    prim, p2 = out["PRIMAER"], out["P2"]
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta"] < 0:
        v = (f"H1+ -- das Training der Winkel SCHADET: eingefroren ist um "
             f"{abs(prim['delta']):.4f} besser, {prim['trainable_b']:,} gegen "
             f"{prim['trainable_a']:,} trainierbare Blockparameter. Die "
             f"zufaellige orthogonale Mischung ist nicht nur ausreichend, der "
             f"Gradientenabstieg verschlechtert sie aktiv. Der Q-014-Befund "
             f"haelt unter verschaerften Bedingungen und verstaerkt sich.")
        if prim["p_tost"] < 0.05:
            v += (f" ACHTUNG: zugleich TOST-gleichwertig innerhalb +-{MARGIN} -- "
                  f"dann gilt die Gleichwertigkeitsaussage und die Richtung ist "
                  f"nur beschreibend.")
    elif prim["p_tost"] < 0.05:
        v = (f"H1 -- das Training der Winkel traegt nichts bei: gleichwertig "
             f"innerhalb +-{MARGIN}, auch ohne Encoder, der aushelfen koennte. "
             f"Zusammen mit den TOST-Gleichwertigkeiten fuer Verschraenkung und "
             f"Tiefe wirkt von der Schaltung die feste Struktur und nicht der "
             f"gelernte Teil. Fuer die T-Serie ist das die guenstigste Lage: "
             f"kein Rueckwaertslauf noetig, Konstanten faltbar.")
        if p2["p_holm"] < 0.05 and p2["delta"] > 0:
            v += (" Vorbehalt: untied_trained ist signifikant besser -- mit "
                  "10.240 trainierbaren Winkeln traegt Training also doch etwas "
                  "bei, nur mit 80 nicht.")
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        v = (f"H0 -- das Training traegt sehr wohl bei (+{prim['delta']:.4f}). "
             f"Der Q-014-Befund war ein Artefakt des freien Encoders und ist "
             f"zurueckzunehmen. Fuer die T-Serie heisst das: der Rueckwaertslauf "
             f"wird gebraucht, R5 (Konstantenfaltung) faellt weg.")
    else:
        v = f"HU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f}."
    print(f"BEWERTUNG: {v}")
    print("=" * 94)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    nur_vortest = len(sys.argv) > 1 and sys.argv[1] == "vortest"
    klassen, gewaehlt, zweit = vortest()
    if nur_vortest:
        raise SystemExit(0)
    recorder = run(primaer=gewaehlt, zweitstufe=zweit, num_classes=klassen)
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.frozen_angles '
              "import analyse; analyse('Q-018')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
