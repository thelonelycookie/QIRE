"""Abteilung Clifford -- Clifford-Geruest + k Magiewinkel VON ANFANG AN (Karte I-004, Q3).

HEILMEIER:
  Was wollen wir?  Eine Schaltung mit T-Zahl ~k (Clifford-Geruest + k
      trainierbare Nicht-Clifford-Winkel) direkt trainieren, statt erst das
      volle Modell zu trainieren, dann zu runden und k Winkel nachzutrainieren.
  Wie heute?  clifford.py / magie_pruning.py (I-001): voll trainieren ->
      Rangfolge „verlust“ -> runden -> k Winkel frei + frischer Kopf
      nachtrainieren. Braucht immer ein volles Modell (zwei Trainings).
  Was ist neu?  Zerlegung, WOHER der Nutzen der k Winkel kommt:
      (a) nachtraeglich   -- der Weg aus I-001 (Positionen, Geruest UND
                             Startwerte der Magie aus dem vollen Modell).
      (b) geruest_fest    -- Geruest = gerundete Winkel des vollen Modells
                             desselben Seeds, Positionen aus derselben
                             Rangfolge „verlust“, dann ALLES neu: Magiewinkel
                             zufaellig, Kopf frisch. Nur k Winkel + Kopf werden
                             trainiert. (a) gegen (b) misst: braucht man die
                             GELERNTEN Magie-Werte oder nur die Positionen?
      (c) geruest_zufall  -- zufaelliges Clifford-Geruest, k zufaellige
                             Positionen, k Winkel + Kopf trainiert. Keine
                             Information aus einem vollen Modell: der echte
                             „von Anfang an“-Fall, billigste Pipeline (ein
                             Training). Abgrenzung CAFQA (arXiv:2202.12924,
                             2308.11616): dort Clifford-Suche als VQE-Start,
                             hier Klassifikation und Multiplikationszahl.
      (d) Referenzen voll, ohne_schaltung (Identitaet).
  Woran merken wir es?  Gewinnanteil G(k) = (acc_k - acc_ohne) / (acc_voll -
      acc_ohne) wie clifford.py v3 (Mittel ueber Seeds im Zaehler und Nenner).

Gleicher frischer Kopf in allen k-Armen eines Seeds (torch.manual_seed(seed),
dann kopf.reset_parameters(), wie _gerundet_mit), gleiche zufaellige
Startwerte der Magie in (b) und (c) (Generator 60_000 + seed): die Arme sind
gepaart bis auf das, was sie unterscheiden soll.

VORREGISTRIERT (vor dem ersten Lauf dieses Moduls, Regel 7):
  R-0 Vorbedingung: voll schlaegt ohne_schaltung (Gewinn > 0, exakter
      Wilcoxon p < 0,05). Sonst R-1..R-3 „HU (Vorbedingung)“.
  Skala: Differenzen werden durch Gewinn = mittel(voll - ohne) geteilt (eine
      positive Konstante, aendert den Wilcoxon nicht), damit sind Mittel und
      KI direkt in Einheiten von G.
  Familie fuer Holm: {R-1, R-2} (die beiden gerichteten Wilcoxon-Aussagen).
      R-3 ist eine Aequivalenzfrage (TOST) und nicht Teil der Familie.
  R-1 geruest_zufall erreicht bei k = 4 mindestens den halben Gewinn (G >= 0,5).
      Saubere Form ohne geschaetzten Nenner je Seed: e_s = acc_gz4(s) -
      (acc_ohne(s) + acc_voll(s)) / 2, also mittel(e)/Gewinn = G(4) - 0,5.
        Vorbedingung verletzt                      -> HU (Vorbedingung)
        mittel(e) > 0 und Holm-p(Wilcoxon e) < 0,05 -> H1
        einseitiges 95-%-KI-oben(e) < 0:
            gz4 schlaegt ohne (mittel > 0, Wilcoxon p < 0,05, roh)
                                                   -> H0 (teilweise)
            sonst                                  -> H0
        sonst                                      -> HU
      Beschreibend dazu: G(4) von geruest_zufall.
  R-2 nachtraeglich schlaegt geruest_zufall bei k = 2 (gepaart, exakter
      Wilcoxon), d_s = (acc_nach2 - acc_gz2) / Gewinn:
        Vorbedingung verletzt                      -> HU (Vorbedingung)
        mittel(d) > 0 und Holm-p < 0,05             -> H1
        mittel(d) < 0 und Holm-p < 0,05             -> H0 (umgekehrt)
        KI-oben(d) < 0,10                          -> H0 (billige Pipeline reicht)
        sonst                                      -> HU
  R-3 geruest_fest gegen nachtraeglich bei k = 2, TOST mit Grenze +-0,15 G
      (zwei einseitige t-Tests je alpha = 0,05 = 90-%-KI), d_s = (acc_fest2 -
      acc_nach2) / Gewinn:
        Vorbedingung verletzt                      -> HU (Vorbedingung)
        -0,15 < KI-unten und KI-oben < +0,15       -> H1 (nur Positionen noetig)
        KI-oben < -0,15                            -> H0 (Werte noetig)
        KI-unten > +0,15                           -> H0 (Neustart besser)
        sonst                                      -> HU
  R-4 beschreibend: G(k)-Kurven der drei Wege (k = 1, 2, 4); fest - zufall bei
      k = 2 (Wert von Geruest + Positionen); Trainingslaeufe je Pipeline
      (nachtraeglich 2, geruest_fest 2, geruest_zufall 1); Multiplikationen.
  AUFLOESUNG (offen benannt): Ziffern-Test = 539 Bilder, 1 Bild = 0,19 pp; bei
  ~1,7 pp Gewinn ist 0,1 G ~ 1 Testbild je Seed. Entschieden wird ueber das
  gepaarte Mittel von 10 Seeds; HU ist hier ein erwartbarer Ausgang.

PROTOKOLL: Ziffern (n = 6 Qubits), frische Mess-Seeds 80-89, Tuning-Seeds
910/911 (Validierung aus dem Trainingsteil). Eigene Rate fuer JEDEN Arm (Regel
1a), Startgitter 0,02/0,08/0,32, am Rand in beide Richtungen verlaengert (1b,
bis 6 Runden), volles Budget 1500 Schritte (1c). Rohdaten sofort sichern (5).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen.clifford import GITTER, LAGEN, QUBITS_JE_DATENSATZ, QireZelle, _baue, multiplikationen, runde_clifford
from abteilungen.magie_pruning import _gepaart, _gerundet_mit, _grundlauf, _teile, rangfolgen
from labor.gemeinsam import genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, ziffern, zusammenfassung

MESS_SEEDS = tuple(range(80, 90))
TUNING_SEEDS = (910, 911)
STUFEN = [1, 2, 4]
K_R1, K_ENTSCHEID = 4, 2
HALB = 0.5
SCHWELLE_R2 = 0.10
AEQUI = 0.15
ALPHA = 0.05
START_GITTER = (0.02, 0.08, 0.32)
MAX_RUNDEN = 6
WEGE = ["nachtraeglich", "geruest_fest", "geruest_zufall"]
GRUNDARME = ["voll", "ohne_schaltung"]
ARME_K = [f"{w}_k{k}" for w in WEGE for k in STUFEN]
ARME_OHNE_VOLL = GRUNDARME + [f"geruest_zufall_k{k}" for k in STUFEN]   # brauchen kein volles Modell
ARME_MIT_VOLL = [a for a in ARME_K if not a.startswith("geruest_zufall")]
TRAININGS_JE_PIPELINE = {"nachtraeglich": 2, "geruest_fest": 2, "geruest_zufall": 1}

HYPOTHESEN = {
    "R-0": "Vorbedingung: voll schlaegt die Identitaet (Gewinn > 0, Wilcoxon p < 0,05).",
    "R-1": "Zufaelliges Clifford-Geruest + 4 trainierte Magiewinkel erreicht mindestens den "
           "halben Gewinn der vollen Schaltung (G(4) >= 0,5; Wilcoxon auf acc_gz4 - (ohne+voll)/2).",
    "R-2": "Nachtraegliches Runden (I-001-Weg) schlaegt das zufaellige Geruest bei k = 2.",
    "R-3": "Geruest aus dem vollen Modell mit NEU initialisierten Magiewinkeln ist aequivalent "
           "zum Nachtraining (TOST +-0,15 G, k = 2): nur die Positionen zaehlen, nicht die Werte.",
    "R-4": "beschreibend: G(k)-Kurven, fest - zufall, Trainingskosten, Multiplikationen.",
}


# ------------------------------------------------------------------ Bausteine
def magie_start(seed: int, anzahl: int) -> torch.Tensor:
    """Zufaellige Startwerte der Magiewinkel (wie QireZelle: gleichverteilt in [0, 2pi))."""
    g = torch.Generator().manual_seed(60_000 + seed)
    return torch.rand(anzahl, generator=g) * 2 * math.pi


def zufalls_geruest(seed: int, anzahl: int):
    """Zufaelliges Clifford-Geruest (Vielfache von pi/2) und zufaellige Positionsfolge
    (geschachtelt: die k freien Winkel sind die ersten k)."""
    g = torch.Generator().manual_seed(40_000 + seed)
    geruest = torch.randint(0, 4, (anzahl,), generator=g).float() * GITTER
    return geruest, torch.randperm(anzahl, generator=g).tolist()


def baue_geruest(n: int, geruest: torch.Tensor, frei_idx, seed: int) -> QireZelle:
    """Clifford-Geruest fest, k Winkel an frei_idx trainierbar (Start: magie_start),
    Kopf frisch -- mit derselben Kopf-Initialisierung wie _gerundet_mit."""
    anzahl = LAGEN * n
    m = QireZelle(n=n)
    torch.manual_seed(seed)
    m.kopf.reset_parameters()
    maske = torch.zeros(anzahl)
    maske[torch.as_tensor(list(frei_idx), dtype=torch.long)] = 1
    m.fixiere(geruest.reshape(LAGEN, n), maske.reshape(LAGEN, n))
    with torch.no_grad():
        m.theta.copy_(torch.where(m.frei.bool(), magie_start(seed, anzahl).reshape(LAGEN, n), m.fest))
    return m


def modell_fuer(arm: str, seed: int, n: int, winkel=None, rang=None) -> QireZelle:
    """arm '<weg>_k<k>' -> untrainiertes Modell. winkel/rang: trainiertes volles Modell
    desselben Seeds (nur fuer nachtraeglich und geruest_fest)."""
    weg, k = arm.rsplit("_k", 1)
    k = int(k)
    anzahl = LAGEN * n
    if weg == "nachtraeglich":
        voll = QireZelle(n=n)
        with torch.no_grad():
            voll.theta.copy_(torch.tensor(winkel, dtype=torch.float32).reshape(voll.theta.shape))
        return _gerundet_mit(voll, rang["verlust"][:k], seed)
    if weg == "geruest_fest":
        geruest = runde_clifford(torch.tensor(winkel, dtype=torch.float32))
        frei = rang["verlust"][:k]
    elif weg == "geruest_zufall":
        geruest, perm = zufalls_geruest(seed, anzahl)
        frei = perm[:k]
    else:
        raise ValueError(arm)
    return baue_geruest(n, geruest, frei, seed)


# ------------------------------------------------------------------ Arbeiter
def _arbeit(a):
    """(seed, arm, lr, schritte, ds, teil, winkel_voll, rang) -> Zeilenteil (dict)."""
    s, arm, lr, schritte, ds, teil, winkel, rang = a
    if arm in GRUNDARME:
        return _grundlauf((s, arm, schritte, lr, ds, teil))
    xtr, ytr, xev, yev = _teile(ds, s, teil)
    m = modell_fuer(arm, s, QUBITS_JE_DATENSATZ[ds], winkel, rang)
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    frei = m.frei.flatten().bool()
    return {"seed": s, arm: genauigkeit(m(xev), yev),
            f"{arm}_winkel": m.winkel().detach().flatten()[frei].tolist()}


def _gitter_suche(arme, bewerte, start=START_GITTER, runden=MAX_RUNDEN):
    """Regel 1: eigene Rate je Arm; liegt das Optimum am Rand, wird das Gitter in
    diese Richtung verlaengert (Faktor 4, beide Richtungen). bewerte(paare) ->
    {(arm, lr): mittlere Validierungsgenauigkeit}."""
    gitter = {arm: sorted(start) for arm in arme}
    wert = {arm: {} for arm in arme}
    offen, info = set(arme), {}
    for _ in range(runden):
        paare = [(arm, lr) for arm in sorted(offen) for lr in gitter[arm] if lr not in wert[arm]]
        for (arm, lr), v in bewerte(paare).items():
            wert[arm][lr] = float(v)
        for arm in list(offen):
            g = sorted(gitter[arm])
            best = max(g, key=lambda v: wert[arm][v])
            if best == g[0]:
                gitter[arm] = [g[0] / 4] + g
            elif best == g[-1]:
                gitter[arm] = g + [g[-1] * 4]
            else:
                info[arm] = {"lr": best, "gitter": wert[arm], "am_rand": False}
                offen.discard(arm)
        if not offen:
            break
    for arm in offen:
        info[arm] = {"lr": max(wert[arm], key=wert[arm].get), "gitter": wert[arm], "am_rand": True}
    return info


# ------------------------------------------------------------------ Lauf
def lauf(seeds=MESS_SEEDS, schritte=1500, datensatz="ziffern", schnell=False, tuning_seeds=TUNING_SEEDS):
    if schnell:
        seeds, schritte, tuning_seeds = (0, 1), 150, (910,)
    t0 = time.time()
    n = QUBITS_JE_DATENSATZ[datensatz]
    zaehler = {"tuning": 0, "mess": 0}
    ablage = {}      # (lr, seed) -> Zeilenteil des vollen Modells auf dem Tuning-Teil

    def bewerte(paare, basis=None):
        aufg = [(s, arm, lr, schritte, datensatz, "tuning",
                 basis[s]["winkel_voll"] if basis else None, basis[s]["rang"] if basis else None)
                for arm, lr in paare for s in tuning_seeds]
        zaehler["tuning"] += len(aufg)
        tmp = {}
        for (s, arm, lr, *_), z in zip(aufg, parallel(_arbeit, aufg)):
            tmp.setdefault((arm, lr), []).append(z[arm])
            if arm == "voll":
                ablage[(lr, s)] = z
        return {k: float(np.mean(v)) for k, v in tmp.items()}

    # Tuning 1: alles, was kein volles Modell braucht (voll, ohne, geruest_zufall)
    lr_info = _gitter_suche(ARME_OHNE_VOLL, bewerte)
    lr_voll = lr_info["voll"]["lr"]
    # Tuning 2: die Voll-Modelle der Tuning-Seeds mit der gewaehlten Rate liegen schon vor
    basis = {s: ablage[(lr_voll, s)] for s in tuning_seeds}
    lr_info.update(_gitter_suche(ARME_MIT_VOLL, lambda p: bewerte(p, basis)))
    lr = {arm: v["lr"] for arm, v in lr_info.items()}
    print(f"  Raten gewaehlt nach {time.time() - t0:.0f} s: "
          + ", ".join(f"{a} {lr[a]:g}" for a in GRUNDARME + ARME_K), flush=True)

    zeilen = {s: {"seed": s} for s in seeds}
    aufg = [(s, arm, lr[arm], schritte, datensatz, "mess", None, None) for s in seeds for arm in ARME_OHNE_VOLL]
    zaehler["mess"] += len(aufg)
    for z in parallel(_arbeit, aufg):
        zeilen[z["seed"]].update(z)
    aufg = [(s, arm, lr[arm], schritte, datensatz, "mess", zeilen[s]["winkel_voll"], zeilen[s]["rang"])
            for s in seeds for arm in ARME_MIT_VOLL]
    zaehler["mess"] += len(aufg)
    for z in parallel(_arbeit, aufg):
        zeilen[z["seed"]].update(z)
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: voll {z['voll']:.4f}  ohne {z['ohne_schaltung']:.4f}  k2: "
              + "  ".join(f"{w} {z[f'{w}_k2']:.4f}" for w in WEGE), flush=True)
    ergebnis = {"abteilung": "clifford", "modul": "geruest", "karte": "I-004", "version": 1,
                "datensatz": datensatz, "qubits": n, "hypothesen": HYPOTHESEN, "lr": lr_info,
                "schritte": schritte, "seeds": list(seeds), "tuning_seeds": list(tuning_seeds),
                "trainingslaeufe": zaehler, "trainings_je_pipeline": TRAININGS_JE_PIPELINE,
                "zeilen": zeilen, "sekunden": time.time() - t0,
                "multiplikationen": {"voll": multiplikationen(n),
                                     **{f"k{k}": multiplikationen(n, magie=k) for k in STUFEN}}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def holm(p: dict) -> dict:
    """Holm-Bonferroni ueber die angemeldete Familie (Regel 4), monoton."""
    namen = sorted(p, key=p.get)
    m, lauf_max, aus = len(namen), 0.0, {}
    for i, h in enumerate(namen):
        lauf_max = max(lauf_max, min(1.0, (m - i) * p[h]))
        aus[h] = lauf_max
    return aus


def auswertung(zeilen) -> dict:
    a = lambda k: np.array([z[k] for z in zeilen], dtype=float)
    aus = {k: zusammenfassung(a(k)) for k in GRUNDARME + ARME_K}
    d0 = a("voll") - a("ohne_schaltung")
    gewinn = float(d0.mean())
    aus["R0_gewinn"], aus["R0_p"] = gewinn, wilcoxon_exakt(d0)
    aus["vorbedingung"] = bool(gewinn > 0 and aus["R0_p"] < ALPHA)
    G = lambda arm: float((a(arm) - a("ohne_schaltung")).mean()) / gewinn if gewinn > 0 else None
    aus["R4_kurven"] = {w: {str(k): G(f"{w}_k{k}") for k in STUFEN} for w in WEGE}
    skala = gewinn if gewinn > 0 else 1.0      # bei verletzter Vorbedingung nur beschreibend
    mitte = (a("ohne_schaltung") + a("voll")) / 2
    aus["R1"] = _gepaart((a(f"geruest_zufall_k{K_R1}") - mitte) / skala)          # Mittel = G(4) - 0,5
    aus["R1_gegen_ohne"] = _gepaart((a(f"geruest_zufall_k{K_R1}") - a("ohne_schaltung")) / skala)
    kk = K_ENTSCHEID
    aus["R2"] = _gepaart((a(f"nachtraeglich_k{kk}") - a(f"geruest_zufall_k{kk}")) / skala)
    aus["R3"] = _gepaart((a(f"geruest_fest_k{kk}") - a(f"nachtraeglich_k{kk}")) / skala)
    ph = holm({"R-1": aus["R1"]["p"], "R-2": aus["R2"]["p"]})
    aus["R1"]["p_holm"], aus["R2"]["p_holm"] = ph["R-1"], ph["R-2"]
    aus["R4_fest_minus_zufall"] = {str(k): _gepaart((a(f"geruest_fest_k{k}") - a(f"geruest_zufall_k{k}")) / skala)
                                   for k in STUFEN}
    aus["R4_fest_minus_nach"] = {str(k): _gepaart((a(f"geruest_fest_k{k}") - a(f"nachtraeglich_k{k}")) / skala)
                                 for k in STUFEN}
    aus["n"] = len(zeilen)
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Eindeutige Praefixe:
    'H1:', 'H0:', 'H0 (teilweise):', 'H0 (umgekehrt):', 'H0 (Werte noetig):',
    'H0 (Neustart besser):', 'HU:', 'HU (Vorbedingung):'."""
    u = {"R-0": (f"H1: Schaltung schlaegt Identitaet (Gewinn {a['R0_gewinn']:+.4f}, p = {a['R0_p']:.4f})"
                 if a["vorbedingung"] else
                 f"H0: Gewinn ueber Identitaet nicht nachweisbar ({a['R0_gewinn']:+.4f}, p = {a['R0_p']:.4f})")}
    if not a["vorbedingung"]:
        for h in ("R-1", "R-2", "R-3"):
            u[h] = "HU (Vorbedingung): kein nachweisbarer Gewinn der Schaltung -- G undefiniert"
    else:
        r1, r1o = a["R1"], a["R1_gegen_ohne"]
        g4 = r1["mittel"] + HALB
        if r1["mittel"] > 0 and r1["p_holm"] < ALPHA:
            u["R-1"] = (f"H1: zufaelliges Geruest + {K_R1} Magiewinkel erreicht nachweislich mehr als den "
                        f"halben Gewinn (G = {g4:.2f}, Holm-p = {r1['p_holm']:.4f}) -- ein Training genuegt")
        elif r1["ki_oben"] < 0:
            if r1o["mittel"] > 0 and r1o["p"] < ALPHA:
                u["R-1"] = (f"H0 (teilweise): zufaelliges Geruest hilft (G = {g4:.2f}, gegen Identitaet "
                            f"p = {r1o['p']:.4f}), bleibt aber nachweislich unter dem halben Gewinn")
            else:
                u["R-1"] = (f"H0: zufaelliges Geruest bleibt nachweislich unter dem halben Gewinn und ist "
                            f"nicht nachweisbar besser als Identitaet (G = {g4:.2f})")
        else:
            u["R-1"] = (f"HU: KI ueberdeckt G = 0,5 (G = {g4:.2f}, KI {r1['ki_unten'] + HALB:.2f} .. "
                        f"{r1['ki_oben'] + HALB:.2f}, Holm-p = {r1['p_holm']:.4f})")
        r2 = a["R2"]
        if r2["mittel"] > 0 and r2["p_holm"] < ALPHA:
            u["R-2"] = (f"H1: nachtraeglich schlaegt zufaelliges Geruest bei k={K_ENTSCHEID} "
                        f"({r2['mittel']:+.3f} G, Holm-p = {r2['p_holm']:.4f}) -- das volle Training "
                        f"liefert noetige Information")
        elif r2["mittel"] < 0 and r2["p_holm"] < ALPHA:
            u["R-2"] = (f"H0 (umgekehrt): zufaelliges Geruest schlaegt nachtraeglich ({r2['mittel']:+.3f} G, "
                        f"Holm-p = {r2['p_holm']:.4f}) -- nachtraegliches Runden schadet sogar")
        elif r2["ki_oben"] < SCHWELLE_R2:
            u["R-2"] = (f"H0: nachtraeglich nachweislich nicht um >= {SCHWELLE_R2} G besser "
                        f"({r2['mittel']:+.3f} G, KI oben {r2['ki_oben']:+.3f}) -- billige Pipeline reicht")
        else:
            u["R-2"] = (f"HU: unentschieden ({r2['mittel']:+.3f} G, Holm-p = {r2['p_holm']:.4f}, "
                        f"KI oben {r2['ki_oben']:+.3f})")
        r3 = a["R3"]
        ki = f"90-%-KI {r3['ki_unten']:+.3f} .. {r3['ki_oben']:+.3f}"
        if r3["ki_unten"] > -AEQUI and r3["ki_oben"] < AEQUI:
            u["R-3"] = (f"H1: geruest_fest aequivalent zu nachtraeglich (+-{AEQUI} G; {ki}) -- "
                        f"nur die Positionen zaehlen, die gelernten Magie-Werte nicht")
        elif r3["ki_oben"] < -AEQUI:
            u["R-3"] = (f"H0 (Werte noetig): Neustart der Magiewinkel kostet nachweislich > {AEQUI} G "
                        f"({r3['mittel']:+.3f} G; {ki})")
        elif r3["ki_unten"] > AEQUI:
            u["R-3"] = (f"H0 (Neustart besser): neu initialisierte Magiewinkel sind nachweislich um "
                        f"> {AEQUI} G besser ({r3['mittel']:+.3f} G; {ki}) -- Nachtraining klemmt")
        else:
            u["R-3"] = f"HU: Aequivalenz weder gezeigt noch widerlegt ({r3['mittel']:+.3f} G; {ki})"
    k = a["R4_kurven"]
    fz = a["R4_fest_minus_zufall"][str(K_ENTSCHEID)]
    u["R-4"] = ("G(k) " + "; ".join(f"{w}: " + ", ".join("-" if v is None else f"{v:.2f}" for v in k[w].values())
                                    for w in WEGE)
                + f" (k = {', '.join(map(str, STUFEN))}); fest - zufall bei k={K_ENTSCHEID}: "
                  f"{fz['mittel']:+.3f} G (p = {fz['p']:.4f}); Trainings je Pipeline {TRAININGS_JE_PIPELINE}")
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_struktur() -> bool:
    """Tun die Bausteine, was ihr Name sagt? Billig: untrainiertes „volles“ Modell
    als Stellvertreter, 300 Ziffern, hoechstens 5 Trainingsschritte."""
    ok = True

    def fehler(text):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", text)

    x, y, _, _ = ziffern(0)
    x, y = x[:300], y[:300]
    n, s = 6, 3
    anzahl = LAGEN * n
    torch.manual_seed(1)
    voll = QireZelle(n=n)
    winkel = voll.theta.detach().flatten().tolist()
    rang = rangfolgen(voll, x, y, seed=1)
    _, perm = zufalls_geruest(s, anzahl)
    start = magie_start(s, anzahl)
    koepfe = {}
    for k in STUFEN:
        for w in WEGE:
            arm = f"{w}_k{k}"
            m = modell_fuer(arm, s, n, winkel, rang)
            wk = m.winkel().detach().flatten()
            frei = m.frei.flatten().bool()
            koepfe[arm] = (m.kopf.weight.detach().clone(), m.kopf.bias.detach().clone())
            # genau k Winkel frei, an den richtigen Positionen
            soll = rang["verlust"][:k] if w != "geruest_zufall" else perm[:k]
            if int(frei.sum()) != k or sorted(torch.nonzero(frei).flatten().tolist()) != sorted(soll):
                fehler(f"falsche freie Positionen {arm}")
            # Geruestwinkel liegen EXAKT auf dem Gitter (Rundung idempotent, ganzzahlige Vielfache)
            g = wk[~frei]
            if not torch.equal(g, runde_clifford(g)) or \
                    bool(((g / GITTER - torch.round(g / GITTER)).abs() > 1e-6).any()):
                fehler(f"Geruest nicht auf dem Gitter {arm}")
            if w == "geruest_fest" and not torch.equal(g, runde_clifford(torch.tensor(winkel))[~frei]):
                fehler("geruest_fest: Geruest ist nicht das gerundete volle Modell")
            if w in ("geruest_fest", "geruest_zufall") and not torch.equal(wk[frei], start[frei]):
                fehler(f"{arm}: Magiewinkel nicht neu (magie_start) initialisiert")
            if w == "nachtraeglich" and not torch.equal(wk[frei], torch.tensor(winkel)[frei]):
                fehler(f"{arm}: Magiewinkel starten nicht beim trainierten Wert")
            if torch.equal(m.kopf.weight, voll.kopf.weight):
                fehler(f"{arm}: Kopf geerbt statt frisch")
            # Gradient: Geruest-Winkel exakt 0, freie Winkel und Kopf nicht 0
            m.zero_grad()
            torch.nn.functional.cross_entropy(m(x[:64]), y[:64]).backward()
            gr = m.theta.grad.flatten()
            if not torch.all(gr[~frei] == 0) or torch.any(gr[frei] == 0) or m.kopf.weight.grad.abs().sum() == 0:
                fehler(f"{arm}: Gradientenmaske falsch")
            # 5 echte Adam-Schritte: Geruest unveraendert, freie Winkel bewegt
            trainiere(m, x, y, 5, 0.1, seed=s)
            wk2 = m.winkel().detach().flatten()
            if not torch.equal(wk2[~frei], wk[~frei]) or torch.equal(wk2[frei], wk[frei]):
                fehler(f"{arm}: Training bewegt das Geruest oder nicht die Magie")
    # frischer Kopf: in allen k-Armen eines Seeds derselbe (gepaart)
    ref = koepfe[ARME_K[0]]
    if any(not (torch.equal(kw, ref[0]) and torch.equal(kb, ref[1])) for kw, kb in koepfe.values()):
        fehler("Koepfe der k-Arme nicht identisch initialisiert")
    # geruest_zufall haengt nicht vom vollen Modell ab; Positionen geschachtelt
    a1 = modell_fuer("geruest_zufall_k2", s, n, winkel, rang)
    a2 = modell_fuer("geruest_zufall_k2", s, n, [0.1] * anzahl, None)
    if any(not torch.equal(a1.state_dict()[q], a2.state_dict()[q]) for q in a1.state_dict()):
        fehler("geruest_zufall benutzt Information aus dem vollen Modell")
    if torch.equal(zufalls_geruest(s, anzahl)[0], runde_clifford(torch.tensor(winkel))):
        fehler("Zufallsgeruest faellt mit gerundetem vollen Modell zusammen")
    if zufalls_geruest(s, anzahl)[1][:4] == rang["verlust"][:4]:
        fehler("Zufallspositionen fallen mit Verlust-Rangfolge zusammen")
    # Gittersuche: verlaengert in beide Richtungen und findet ein inneres Optimum
    for spitze in (0.02 / 16, 0.32 * 16, 0.08):
        info = _gitter_suche(["a"], lambda p: {(arm, lr): -abs(math.log(lr / spitze)) for arm, lr in p})
        if info["a"]["am_rand"] or not math.isclose(info["a"]["lr"], spitze):
            fehler(f"Gittersuche findet Spitze {spitze} nicht: {info['a']}")
    # Holm gegen Handrechnung
    h = holm({"a": 0.01, "b": 0.04})
    if not (math.isclose(h["a"], 0.02) and math.isclose(h["b"], 0.04)):
        fehler(f"Holm falsch: {h}")
    h = holm({"a": 0.03, "b": 0.02})
    if not (math.isclose(h["b"], 0.04) and math.isclose(h["a"], 0.04)):
        fehler(f"Holm nicht monoton: {h}")
    # Multiplikationszahl: Geruest-Arme kosten wie k Magiewinkel, nicht wie voll
    if not multiplikationen(n, magie=2) < multiplikationen(n):
        fehler("Multiplikationszahl")
    return ok


def _synth(rng, G, gewinn=0.07, sd=0.004, streu=None, n=10):
    """Synthetische Zeilen. G: {arm: wahrer Gewinnanteil} (fehlende k-Arme: 0,5).
    streu: {arm: zusaetzliche Streuung in Genauigkeit} fuer Grenzfaelle."""
    streu = streu or {}
    out = []
    for s in range(n):
        ohne = 0.90 + rng.normal(0, sd)
        z = {"seed": s, "ohne_schaltung": ohne, "voll": ohne + gewinn + rng.normal(0, sd)}
        for arm in ARME_K:
            z[arm] = ohne + gewinn * G.get(arm, 0.5) + rng.normal(0, sd) + rng.normal(0, streu.get(arm, 0.0) + 1e-12)
        out.append(z)
    return out


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: jeden Zweig auf synthetischen Zeilen erzwingen. Null- und
    Grenzfaelle kippen mit Rate ~alpha -- dort ist je Hypothese Toleranz erlaubt."""
    ok = _vortest_struktur()
    rng = np.random.default_rng(4)
    faelle = [
        # alle H1: zufall k4 weit ueber der Haelfte, nach2 >> zufall2, fest2 = nach2
        (dict(G={"geruest_zufall_k4": 0.90, "geruest_zufall_k2": 0.20, "nachtraeglich_k2": 0.80,
                 "geruest_fest_k2": 0.80}),
         {"R-0": "H1:", "R-1": "H1:", "R-2": "H1:", "R-3": "H1:"}, 0),
        # teilweise / umgekehrt / Werte noetig
        (dict(G={"geruest_zufall_k4": 0.25, "geruest_zufall_k2": 0.90, "nachtraeglich_k2": 0.50,
                 "geruest_fest_k2": 0.10}),
         {"R-1": "H0 (teilweise):", "R-2": "H0 (umgekehrt):", "R-3": "H0 (Werte noetig):"}, 0),
        # nutzlos / kein Unterschied (Nullfall: Wilcoxon kippt mit ~alpha) / Neustart besser
        (dict(G={"geruest_zufall_k4": 0.0, "geruest_zufall_k2": 0.50, "nachtraeglich_k2": 0.50,
                 "geruest_fest_k2": 0.95}),
         {"R-1": "H0:", "R-2": "H0:", "R-3": "H0 (Neustart besser):"}, 2),
        # Grenzfaelle genau auf den Schwellen, grosse Streuung -> HU; kippt je Seite mit ~alpha.
        # Kalibriert (400 Ziehungen, 24.09.): Kipprate je Hypothese hier <= 9 %, im Nullfall
        # oben (R-2) 9 %, in der Aufloesungspruefung unten <= 9 %. Mit Toleranz 2 von 5 ist ein
        # Fehlalarm des Vortests < 1 %; ein echter Fehler im Baum kippt fast immer.
        (dict(G={"geruest_zufall_k4": 0.50, "geruest_zufall_k2": 0.40, "nachtraeglich_k2": 0.50,
                 "geruest_fest_k2": 0.35},
              streu={"geruest_zufall_k4": 0.03, "nachtraeglich_k2": 0.03, "geruest_fest_k2": 0.03}),
         {"R-1": "HU:", "R-2": "HU:", "R-3": "HU:"}, 2),
        # Vorbedingung verletzt
        (dict(G={"geruest_zufall_k4": 0.90, "nachtraeglich_k2": 0.80}, gewinn=-0.005),
         {"R-0": "H0:", "R-1": "HU (Vorbedingung):", "R-2": "HU (Vorbedingung):", "R-3": "HU (Vorbedingung):"}, 0),
        # Aufloesungspruefung: alles verrauscht -> kein Zweig ausser HU darf feuern
        (dict(G={"geruest_zufall_k4": 0.60, "nachtraeglich_k2": 0.55, "geruest_fest_k2": 0.50},
              gewinn=0.07, sd=0.004, streu={a: 0.05 for a in ARME_K}),
         {"R-1": "HU:", "R-2": "HU:", "R-3": "HU:"}, 2),
    ]
    for par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth(rng, **par)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest: {h} erwartet '{praefix}' -> {u[h]}")
        for h, f in fehl.items():
            if f > toleranz:
                ok = False
                print(f"  VORTEST FEHLER: {h} bei {par}: {f} von {ziehungen} falsch")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-004" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
