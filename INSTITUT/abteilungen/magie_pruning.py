"""Abteilung Clifford -- Magie-Pruning nach WIRKUNG (Karte I-001, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  Mit k = 1, 2, 4 freien Nicht-Clifford-Winkeln („Magie“)
      moeglichst viel vom Gewinn der Schaltung erhalten.
  Wie heute?  C-clifford waehlt die freien Winkel nach ABSTAND zum Clifford-
      Gitter (_magie_reihenfolge). Das ist blind fuer die Wirkung: ein Winkel
      nahe am Gitter kann trotzdem wichtig sein, ein weiter entfernter egal.
  Was ist neu?  Rangfolge nach WIRKUNG: je Winkel den Anstieg des Trainings-
      verlusts messen, wenn NUR dieser Winkel gerundet wird (Kopf fest,
      einmalige Messung, 24 Vorwaertsdurchlaeufe). Die mit dem groessten
      Anstieg bleiben magisch. Dazu „gierig“: schrittweise den Winkel
      freigeben, der den Verlust der sonst gerundeten Schaltung am meisten
      senkt. Kontrolle „zufall“: zufaellige Rangfolge.
  Woran merken wir es?  Gewinnanteil G(k) = (acc_k - acc_identitaet) /
      (acc_voll - acc_identitaet) wie in clifford.py v3, gepaart je Seed.

Rangfolgen (alle geschachtelt: die k freien Winkel sind die ersten k):
  abstand -- bisher: groesster Abstand zum Gitter k*pi/2 zuerst.
  verlust -- groesster Anstieg des Trainingsverlusts beim Einzelrunden zuerst.
  gierig  -- Vorwaertsauswahl aus der voll gerundeten Schaltung (Kopf fest).
  zufall  -- zufaellige Permutation (Generator 20_000 + seed), Kontrolle.
Die Messung von verlust/gierig benutzt den TRAINIERTEN Kopf (nur zum Ordnen).
Das Nachtraining benutzt stets einen FRISCHEN Kopf (Lehre „geerbter Kopf“,
FEHLERBUCH 24.09.): _gerundet_mit ist _gerundet aus clifford.py mit frei
waehlbarer Rangfolge -- der Vortest prueft, dass beide fuer „abstand“
bitgleich sind.

VORREGISTRIERT (vor dem ersten Lauf dieses Moduls, Regel 7):
  P-0 Vorbedingung: voll schlaegt Identitaet (Gewinn > 0, exakter Wilcoxon
      p < 0,05). Sonst ist G undefiniert bzw. reines Rauschen -> P-1/P-2 „HU
      (Vorbedingung)“.
  P-1 Entscheidung bei k = 2, gepaart ueber die Mess-Seeds:
      d_s = (acc_verlust_k2 - acc_abstand_k2) / Gewinn, Gewinn = mittlere
      Differenz voll - identitaet ueber alle Seeds (so ist G(k) in clifford.py
      v3 definiert; ein Nenner je Seed wuerde bei Seeds mit Gewinn ~ 0
      explodieren; eine positive Konstante aendert den Wilcoxon nicht).
        Gewinn <= 0 oder P-0 nicht signifikant -> HU (Vorbedingung)
        mittel(d) >= 0,10 und Wilcoxon p < 0,05  -> H1
        sonst: einseitiges 95-%-KI-oben(d) < 0,10 -> H0; liegt dabei
               G_abstand(2) schon >= 0,90, heisst der Zweig „H0 (Decke)“
               (kein Raum fuer +0,10 -- Lehre „falsch skalierte Grenze“)
        sonst                                    -> HU (KI ueberdeckt 0,10)
  P-2 verlust schlaegt zufall bei k = 2, d_s wie oben mit zufall statt abstand:
        Vorbedingung verletzt -> HU (Vorbedingung)
        mittel(d) > 0 und Wilcoxon p < 0,05      -> H1
        sonst KI-oben(d) < 0,10                  -> H0
        sonst                                    -> HU
  P-3 beschreibend: Kurven G(k) je Rangfolge (k = 0, 1, 2, 4), Ueberlappung
      der freien Mengen |A ∩ B| / k je Paar (Zufallserwartung k/24), Spearman
      zwischen Abstand und Verlustanstieg; gierig vs. abstand und abstand vs.
      zufall bei k = 2 (nur berichtet, nicht Teil der Familie).
  AUFLOESUNG (offen benannt): Ziffern-Test = 539 Bilder, 1 Bild = 0,19 pp.
  Bei ~1,7 pp Gewinn (C-clifford-v2) ist 0,10 G ~ 0,17 pp ~ ein Testbild je
  Seed. Die Schwelle ist damit an der Aufloesungsgrenze einzelner Seeds;
  entschieden wird ueber das gepaarte Mittel von 10 Seeds.

PROTOKOLL: Mess-Seeds 30-39 (frisch), Tuning-Seeds 906/907 (Validierung aus
dem Trainingsteil). Rate je Arm eigen (Regel 1a): voll und identitaet ueber
clifford._lr_waehlen, jede Nachtrainings-Rate je (Rangfolge, k) eigen,
Gitter in beide Richtungen verlaengert (1b), volles Budget (1c). Voll-Modell
und Identitaet je Seed selbst trainiert (keine alten Rohdaten: verbrauchte
Seeds). Rohdaten sofort sichern (Regel 5).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen.clifford import (GITTER, LAGEN, QUBITS_JE_DATENSATZ, QireZelle, _baue, _gerundet,
                                  _lr_waehlen, _magie_reihenfolge, daten, multiplikationen, runde_clifford)
from labor.gemeinsam import (genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, ziffern,
                             zusammenfassung)

RANGFOLGEN = ["abstand", "verlust", "gierig", "zufall"]
STUFEN = [1, 2, 4]
K_ENTSCHEID = 2
SCHWELLE_G = 0.10
ALPHA = 0.05
DECKE = 0.90
KOPF_GITTER = (0.02, 0.08, 0.32)   # C-clifford-v2 fand 0,08 fuer den frischen Kopf
ARME_NACH = ["gerundet_k0"] + [f"{r}_k{k}" for r in RANGFOLGEN for k in STUFEN]

HYPOTHESEN = {
    "P-0": "Vorbedingung: voll schlaegt die Identitaet (Gewinn > 0, Wilcoxon p < 0,05).",
    "P-1": "Verlust-Rangfolge hebt den Gewinnanteil G(2) um >= 0,10 ueber die "
           "Abstands-Rangfolge (gepaart, exakter Wilcoxon p < 0,05).",
    "P-2": "Verlust-Rangfolge schlaegt die zufaellige Rangfolge bei k = 2.",
    "P-3": "beschreibend: G(k)-Kurven, Ueberlappung der gewaehlten Winkel, "
           "Spearman(Abstand, Verlustanstieg).",
}


# ------------------------------------------------------------------ Rangfolgen
def _ce(m, th, x, y) -> float:
    """Trainingsverlust mit Winkeln th (Kopf fest), m bleibt unveraendert."""
    alt = m.theta.detach().clone()
    with torch.no_grad():
        m.theta.copy_(th.reshape(alt.shape))
        wert = torch.nn.functional.cross_entropy(m(x), y).item()
        m.theta.copy_(alt)
    return wert


def rangfolgen(voll, x, y, seed, k_max=max(STUFEN)) -> dict:
    """Alle vier Rangfolgen (volle Permutationen der L*n Winkel) + Messgroessen.
    voll muss alle Winkel frei haben (frisch trainiert)."""
    th = voll.theta.detach().clone().flatten()
    gerundet = runde_clifford(th)
    anzahl = len(th)
    basis = _ce(voll, th, x, y)
    delta = torch.zeros(anzahl)
    for i in range(anzahl):
        w = th.clone()
        w[i] = gerundet[i]
        delta[i] = _ce(voll, w, x, y) - basis
    r_verlust = torch.argsort(delta, descending=True, stable=True)
    # gierig: aus der voll gerundeten Schaltung jeweils den Winkel freigeben
    # (auf den trainierten Wert), der den Verlust am staerksten senkt.
    frei, letzte = [], None
    for _ in range(min(k_max, anzahl)):
        kand = {}
        for i in range(anzahl):
            if i in frei:
                continue
            w = gerundet.clone()
            idx = frei + [i]
            w[idx] = th[idx]
            kand[i] = _ce(voll, w, x, y)
        bester = min(kand, key=kand.get)
        frei.append(bester)
        letzte = kand
    rest = sorted((i for i in range(anzahl) if i not in frei), key=lambda i: letzte.get(i, 0.0))
    r_gierig = torch.tensor(frei + rest)
    g = torch.Generator().manual_seed(20_000 + seed)
    r_zufall = torch.randperm(anzahl, generator=g)
    abstand = (th - gerundet).abs()
    return {"abstand": _magie_reihenfolge(voll.theta.detach().clone()).tolist(),
            "verlust": r_verlust.tolist(), "gierig": r_gierig.tolist(), "zufall": r_zufall.tolist(),
            "verlust_delta": delta.tolist(), "abstand_je_winkel": abstand.tolist(),
            "verlust_basis": basis}


def _gerundet_mit(voll, frei_idx, seed):
    """Wie clifford._gerundet, aber mit vorgegebenen freien Winkeln: runden,
    freie Winkel starten beim trainierten Wert, Kopf FRISCH initialisiert."""
    n = voll.n
    th = voll.theta.detach().clone()
    m = QireZelle(n=n)
    m.load_state_dict(voll.state_dict())
    torch.manual_seed(seed)
    m.kopf.reset_parameters()
    frei = torch.zeros(LAGEN * n)
    frei[torch.as_tensor(list(frei_idx), dtype=torch.long)] = 1
    m.fixiere(runde_clifford(th), frei.reshape(LAGEN, n))
    with torch.no_grad():
        m.theta.copy_(torch.where(m.frei.bool(), th, runde_clifford(th)))
    return m


def _frei_fuer(arm, rang):
    """arm 'gerundet_k0' oder '<rangfolge>_k<k>' -> Indizes der freien Winkel."""
    name, k = arm.rsplit("_k", 1)
    return [] if name == "gerundet" else rang[name][:int(k)]


# ------------------------------------------------------------------ Arbeiter
def _teile(ds, s, teil):
    """teil 'tuning': Training auf xtr[n_val:], Bewertung auf xtr[:n_val];
    teil 'mess': Training auf xtr, Bewertung auf dem Testteil."""
    xtr, ytr, xte, yte = daten(ds, s)
    if teil == "tuning":
        nv = len(ytr) // 4
        return xtr[nv:], ytr[nv:], xtr[:nv], ytr[:nv]
    return xtr, ytr, xte, yte


def _grundlauf(a):
    """(seed, arm 'voll'|'ohne_schaltung', schritte, lr, ds, teil) -> Zeilenteil.
    Fuer 'voll' zusaetzlich trainierte Winkel und alle Rangfolgen."""
    s, arm, schritte, lr, ds, teil = a
    xtr, ytr, xev, yev = _teile(ds, s, teil)
    m, _ = _baue(arm, s, QUBITS_JE_DATENSATZ[ds])
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    z = {"seed": s, arm: genauigkeit(m(xev), yev)}
    if arm == "voll":
        z["winkel_voll"] = m.theta.detach().flatten().tolist()
        z["rang"] = rangfolgen(m, xtr, ytr, s)
    return z


def _nachtraining(a):
    """(seed, arm, lr, schritte, ds, teil, winkel_voll, rang) -> Genauigkeit."""
    s, arm, lr, schritte, ds, teil, winkel, rang = a
    xtr, ytr, xev, yev = _teile(ds, s, teil)
    voll = QireZelle(n=QUBITS_JE_DATENSATZ[ds])
    with torch.no_grad():
        voll.theta.copy_(torch.tensor(winkel).reshape(voll.theta.shape))
    m = _gerundet_mit(voll, _frei_fuer(arm, rang), s)
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    return genauigkeit(m(xev), yev)


def _lr_nachtraining(tuning_seeds, schritte, ds, lr_voll, gitter=KOPF_GITTER):
    """Regel 1 fuer die Nachtrainings-Arme: eigene Rate je (Rangfolge, k), auf
    den Tuning-Seeds, volles Budget, Gitter in beide Richtungen verlaengert.
    Die Voll-Modelle der Tuning-Seeds werden einmal trainiert und geteilt."""
    basis = {z["seed"]: z for z in parallel(
        _grundlauf, [(s, "voll", schritte, lr_voll, ds, "tuning") for s in tuning_seeds])}
    gitter = {arm: sorted(gitter) for arm in ARME_NACH}
    wert = {arm: {} for arm in ARME_NACH}
    offen, info = set(ARME_NACH), {}
    for _ in range(4):
        aufg = [(s, arm, lr, schritte, ds, "tuning", basis[s]["winkel_voll"], basis[s]["rang"])
                for arm in sorted(offen) for lr in gitter[arm] if lr not in wert[arm] for s in tuning_seeds]
        erg = parallel(_nachtraining, aufg)
        tmp = {}
        for (s, arm, lr, *_), acc in zip(aufg, erg):
            tmp.setdefault((arm, lr), []).append(acc)
        for (arm, lr), v in tmp.items():
            wert[arm][lr] = float(np.mean(v))
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
def lauf(seeds=range(30, 40), schritte=1500, datensatz="ziffern", schnell=False, tuning_seeds=(906, 907)):
    if schnell:
        seeds, schritte, tuning_seeds = range(2), 150, (906,)
    t0 = time.time()
    n = QUBITS_JE_DATENSATZ[datensatz]
    lr_info = _lr_waehlen(["voll", "ohne_schaltung"], tuning_seeds, schritte, datensatz)
    lr = {arm: v["lr"] for arm, v in lr_info.items()}
    lr_info.update(_lr_nachtraining(tuning_seeds, schritte, datensatz, lr["voll"]))
    lr.update({arm: lr_info[arm]["lr"] for arm in ARME_NACH})
    print(f"  Raten gewaehlt nach {time.time() - t0:.0f} s", flush=True)
    grund = parallel(_grundlauf, [(s, arm, schritte, lr[arm], datensatz, "mess")
                                  for s in seeds for arm in ("voll", "ohne_schaltung")])
    zeilen = {s: {"seed": s} for s in seeds}
    for g in grund:
        zeilen[g["seed"]].update(g)
    aufg = [(s, arm, lr[arm], schritte, datensatz, "mess", zeilen[s]["winkel_voll"], zeilen[s]["rang"])
            for s in seeds for arm in ARME_NACH]
    for (s, arm, *_), acc in zip(aufg, parallel(_nachtraining, aufg)):
        zeilen[s][arm] = acc
    zeilen = [zeilen[s] for s in seeds]
    for z in zeilen:
        print(f"  Seed {z['seed']}: voll {z['voll']:.4f}  ohne {z['ohne_schaltung']:.4f}  k2: "
              + "  ".join(f"{r} {z[f'{r}_k2']:.4f}" for r in RANGFOLGEN), flush=True)
    ergebnis = {"abteilung": "clifford", "modul": "magie_pruning", "karte": "I-001", "version": 1,
                "datensatz": datensatz, "qubits": n, "hypothesen": HYPOTHESEN, "lr": lr_info,
                "schritte": schritte, "seeds": list(seeds), "tuning_seeds": list(tuning_seeds),
                "zeilen": zeilen, "sekunden": time.time() - t0,
                "multiplikationen": {"voll": multiplikationen(n),
                                     **{f"k{k}": multiplikationen(n, magie=k) for k in [0] + STUFEN}}}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def _gepaart(d):
    """Mittel, exakter Wilcoxon, einseitige 95-%-KI-Grenzen (t)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    if len(d) < 2:
        return {"mittel": m, "p": 1.0, "ki_unten": -math.inf, "ki_oben": math.inf}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return {"mittel": m, "p": wilcoxon_exakt(d), "ki_unten": m - q * se, "ki_oben": m + q * se}


def auswertung(zeilen) -> dict:
    a = lambda k: np.array([z[k] for z in zeilen], dtype=float)
    aus = {k: zusammenfassung(a(k)) for k in ["voll", "ohne_schaltung"] + ARME_NACH}
    d0 = a("voll") - a("ohne_schaltung")
    gewinn = float(d0.mean())
    aus["P0_gewinn"], aus["P0_p"] = gewinn, wilcoxon_exakt(d0)
    aus["vorbedingung"] = bool(gewinn > 0 and aus["P0_p"] < ALPHA)
    G = lambda arm: float((a(arm) - a("ohne_schaltung")).mean()) / gewinn if gewinn > 0 else None
    aus["P3_kurven"] = {r: {"0": G("gerundet_k0"), **{str(k): G(f"{r}_k{k}") for k in STUFEN}}
                        for r in RANGFOLGEN}
    skala = gewinn if gewinn > 0 else 1.0     # bei verletzter Vorbedingung nur beschreibend
    kk = K_ENTSCHEID
    for name, (x, y) in {"P1": ("verlust", "abstand"), "P2": ("verlust", "zufall"),
                         "P3_gierig_abstand": ("gierig", "abstand"),
                         "P3_abstand_zufall": ("abstand", "zufall")}.items():
        aus[name] = _gepaart((a(f"{x}_k{kk}") - a(f"{y}_k{kk}")) / skala)
    # Ueberlappung der freien Mengen und Spearman (nur mit Rangdaten)
    if all("rang" in z for z in zeilen):
        from scipy.stats import spearmanr
        ueber = {}
        for i, r1 in enumerate(RANGFOLGEN):
            for r2 in RANGFOLGEN[i + 1:]:
                ueber[f"{r1}|{r2}"] = {str(k): float(np.mean(
                    [len(set(z["rang"][r1][:k]) & set(z["rang"][r2][:k])) / k for z in zeilen]))
                    for k in STUFEN}
        anzahl = len(zeilen[0]["rang"]["verlust"])
        ueber["zufallserwartung"] = {str(k): k / anzahl for k in STUFEN}
        aus["P3_ueberlappung"] = ueber
        rho = [spearmanr(z["rang"]["abstand_je_winkel"], z["rang"]["verlust_delta"])[0] for z in zeilen]
        aus["P3_spearman_abstand_verlust"] = float(np.nanmean(rho))
    aus["n"] = len(zeilen)
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Eindeutige Praefixe:
    'H1:', 'H0:', 'H0 (Decke):', 'HU:', 'HU (Vorbedingung):'."""
    u = {"P-0": (f"H1: Schaltung schlaegt Identitaet (Gewinn {a['P0_gewinn']:+.4f}, p = {a['P0_p']:.4f})"
                 if a["vorbedingung"] else
                 f"H0: Gewinn ueber Identitaet nicht nachweisbar ({a['P0_gewinn']:+.4f}, p = {a['P0_p']:.4f})")}
    if not a["vorbedingung"]:
        for h in ("P-1", "P-2"):
            u[h] = "HU (Vorbedingung): kein nachweisbarer Gewinn der Schaltung -- G undefiniert"
    else:
        p1 = a["P1"]
        g_abst = a["P3_kurven"]["abstand"][str(K_ENTSCHEID)]
        if p1["mittel"] >= SCHWELLE_G and p1["p"] < ALPHA:
            u["P-1"] = f"H1: Verlust-Rangfolge hebt G(2) um {p1['mittel']:+.3f} (p = {p1['p']:.4f})"
        elif p1["ki_oben"] < SCHWELLE_G:
            if g_abst >= DECKE:
                u["P-1"] = (f"H0 (Decke): Abstand erhaelt schon G(2) = {g_abst:.2f} -- kein Raum fuer "
                            f"+{SCHWELLE_G}; Differenz {p1['mittel']:+.3f}")
            else:
                u["P-1"] = (f"H0: Verlust-Rangfolge hebt G(2) nachweislich um < {SCHWELLE_G} "
                            f"(Differenz {p1['mittel']:+.3f}, KI oben {p1['ki_oben']:+.3f})")
        else:
            u["P-1"] = (f"HU: KI ueberdeckt {SCHWELLE_G} (Differenz {p1['mittel']:+.3f}, "
                        f"p = {p1['p']:.4f}, KI oben {p1['ki_oben']:+.3f})")
        p2 = a["P2"]
        if p2["mittel"] > 0 and p2["p"] < ALPHA:
            u["P-2"] = f"H1: Verlust-Rangfolge schlaegt Zufall bei k=2 ({p2['mittel']:+.3f} G, p = {p2['p']:.4f})"
        elif p2["ki_oben"] < SCHWELLE_G:
            u["P-2"] = f"H0: nicht besser als Zufall (Differenz {p2['mittel']:+.3f} G, KI oben {p2['ki_oben']:+.3f})"
        else:
            u["P-2"] = f"HU: unentschieden (Differenz {p2['mittel']:+.3f} G, p = {p2['p']:.4f})"
    k = a["P3_kurven"]
    u["P-3"] = "G(k) " + "; ".join(
        f"{r}: " + ", ".join("-" if v is None else f"{v:.2f}" for v in k[r].values()) for r in RANGFOLGEN) \
        + f" (k = 0, {', '.join(map(str, STUFEN))})"
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_struktur() -> bool:
    """Tun die Bausteine, was ihr Name sagt? (Lehre „entartete Kontrolle“.)
    Billig: untrainiertes Modell, 300 Ziffern, keine Trainingsschritte."""
    ok = True
    x, y, _, _ = ziffern(0)
    x, y = x[:300], y[:300]
    torch.manual_seed(1)
    voll = QireZelle()
    th = voll.theta.detach().clone()
    rang = rangfolgen(voll, x, y, seed=1)
    anzahl = th.numel()
    for r in RANGFOLGEN:
        if sorted(rang[r]) != list(range(anzahl)):
            ok = False
            print("  VORTEST FEHLER: keine Permutation", r)
    for k in [0] + STUFEN:
        alt = _gerundet(voll, k, 7)
        neu = _gerundet_mit(voll, rang["abstand"][:k], 7)
        if any(not torch.equal(alt.state_dict()[n], neu.state_dict()[n]) for n in alt.state_dict()):
            ok = False
            print("  VORTEST FEHLER: _gerundet_mit weicht von _gerundet ab, k =", k)
        for r in RANGFOLGEN:
            m = _gerundet_mit(voll, rang[r][:k], 7)
            w = m.winkel().detach().flatten()
            frei = m.frei.flatten().bool()
            if int(frei.sum()) != k or not torch.equal(w[frei], th.flatten()[frei]):
                ok = False
                print("  VORTEST FEHLER: falsche freie Winkel", r, k)
            if not torch.allclose(w[~frei] / GITTER, torch.round(w[~frei] / GITTER)):
                ok = False
                print("  VORTEST FEHLER: gerundete Winkel nicht auf dem Gitter", r, k)
            if torch.equal(m.kopf.weight, voll.kopf.weight):
                ok = False
                print("  VORTEST FEHLER: Kopf geerbt statt frisch", r, k)
    d = torch.tensor(rang["verlust_delta"])
    if not torch.all(d[rang["verlust"]][:-1] >= d[rang["verlust"]][1:]):
        ok = False
        print("  VORTEST FEHLER: Verlust-Rangfolge nicht absteigend")
    ger = runde_clifford(th.flatten())
    kand = []
    for i in range(anzahl):
        w = ger.clone()
        w[i] = th.flatten()[i]
        kand.append(_ce(voll, w, x, y))
    if rang["gierig"][0] != int(np.argmin(kand)):
        ok = False
        print("  VORTEST FEHLER: gierig waehlt nicht den besten ersten Winkel")
    if rang["zufall"][:4] == rang["abstand"][:4]:
        ok = False
        print("  VORTEST FEHLER: Zufallskontrolle faellt mit Abstand zusammen")
    # Identitaet: Ry(0) + CNOT-Ring ist eine reine Permutation (Kopf kann sie
    # exakt ausgleichen) -- sonst waere die Kontrolle keine Identitaet.
    ohne, _ = _baue("ohne_schaltung", 0)
    U = ohne.unitaer().detach()
    if not (torch.allclose(U.abs().sum(1), torch.ones(len(U))) and int((U.abs() > 1e-6).sum()) == len(U)):
        ok = False
        print("  VORTEST FEHLER: Identitaets-Kontrolle ist keine Permutation")
    return ok


def _synth(rng, g_abst, g_verl, g_zuf, gewinn=0.07, sd=0.004, sd_diff=0.0, n=10):
    """Synthetische Zeilen: G-Werte je Rangfolge bei k=2; sd_diff streut die
    verlust-Arme zusaetzlich (fuer Grenzfaelle)."""
    out = []
    for s in range(n):
        ohne = 0.90 + rng.normal(0, sd)
        z = {"seed": s, "ohne_schaltung": ohne, "voll": ohne + gewinn + rng.normal(0, sd),
             "gerundet_k0": ohne + rng.normal(0, sd)}
        for r, gk in (("abstand", g_abst), ("verlust", g_verl), ("zufall", g_zuf), ("gierig", g_verl)):
            extra = rng.normal(0, sd_diff) if r == "verlust" else 0.0
            for k in STUFEN:
                z[f"{r}_k{k}"] = ohne + gewinn * gk * min(1.0, k / 2) + rng.normal(0, sd) + extra
        perm = [rng.permutation(24).tolist() for _ in RANGFOLGEN]
        z["rang"] = {r: p for r, p in zip(RANGFOLGEN, perm)}
        z["rang"]["abstand_je_winkel"] = rng.uniform(0, 0.8, 24).tolist()
        z["rang"]["verlust_delta"] = rng.uniform(0, 0.1, 24).tolist()
        out.append(z)
    return out


def vortest(ziehungen=5) -> bool:
    """Pflicht-Vortest: jeden Zweig auf synthetischen Zeilen erzwingen. Null- und
    Grenzfaelle kippen mit Rate ~alpha -- je Fall und Hypothese einmal erlaubt."""
    ok = _vortest_struktur()
    rng = np.random.default_rng(11)
    # (Parameter fuer _synth, {Hypothese: erwartetes Praefix}, Toleranz)
    faelle = [
        (dict(g_abst=0.45, g_verl=0.75, g_zuf=0.30), {"P-1": "H1:", "P-2": "H1:", "P-0": "H1:"}, 0),
        (dict(g_abst=0.45, g_verl=0.45, g_zuf=0.45), {"P-1": "H0:", "P-2": "H0:"}, 1),
        (dict(g_abst=0.97, g_verl=0.98, g_zuf=0.40), {"P-1": "H0 (Decke):", "P-2": "H1:"}, 1),
        # Grenzfall: wahre Differenz 0,11 G, grosse Streuung. Simuliert (400 Ziehungen):
        # kippt je Hypothese mit ~10 % (4 % H1, 6 % H0) -- zwei Seiten zu je ~alpha.
        # Bei 5 Ziehungen waeren >= 2 Kipper in 8 % der Vortests zu erwarten, daher
        # Toleranz 2 (Fehlalarm des Vortests dann < 1 %); ein echter Fehler kippt fast immer.
        (dict(g_abst=0.45, g_verl=0.56, g_zuf=0.45, sd_diff=0.10), {"P-1": "HU:", "P-2": "HU:"}, 2),
        (dict(g_abst=0.45, g_verl=0.75, g_zuf=0.30, gewinn=-0.005),
         {"P-0": "H0:", "P-1": "HU (Vorbedingung):", "P-2": "HU (Vorbedingung):"}, 0),
    ]
    for par, erwartet, toleranz in faelle:
        fehl = {h: 0 for h in erwartet}
        for _ in range(ziehungen):
            u = urteil(auswertung(_synth(rng, **par)))
            for h, praefix in erwartet.items():
                if not u[h].startswith(praefix):
                    fehl[h] += 1
                    print(f"  Vortest: {par} {h} -> {u[h]}")
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
    print(speichere("I-001" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
