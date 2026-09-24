"""Abteilung Clifford -- Magie-Kaskade (Karte I-015, Quantenstufe Q3).

HEILMEIER:
  Was wollen wir?  T-Gatter (Magie) nur dort ausgeben, wo ein Bild sie
      braucht: erst die Clifford-Schaltung (k = 0, 0 T-Gatter) rechnen, nur
      unsichere Bilder an eine Schaltung mit k = 2, dann k = 8 magischen
      Winkeln weiterreichen. Kostengroesse ist die MITTLERE T-Zahl je Bild
      statt der festen T-Zahl einer Stufe.
  Wie heute?  Feste Stufe: jedes Bild bezahlt dieselbe T-Zahl (C-hardware:
      k=2 -> 62 T, k=8 -> 247 T bei eps = 1e-3). Early-Exit/Kaskaden sind
      aus der klassischen KI bekannt (BranchyNet 2016); dort sind die Stufen
      Schichttiefen.
  Was ist neu?  Die Stufen sind durch MAGIE definiert (T-Zahl), die teuerste
      Ressource eines fehlertoleranten Quantenrechners.
  Woran merken wir es?  Genauigkeit gegen mittlere T je Bild, verglichen mit
      den festen Stufen k = 0 / 2 / 8 -- ohne und MIT Ladekosten.

AUFBAU (je Seed, MNIST 16x16 = 8 Qubits, 4 Lagen = 32 Winkel):
  1. Validierungsteil = erstes Viertel der Trainingsbilder (1.500); trainiert
     wird auf den uebrigen 4.500. Der Testteil (5.000) wird NUR zur
     Bewertung benutzt, nie zur Wahl von tau.
  2. Volles Modell trainieren (lr 0,08), dann je Stufe k in STUFEN_K wie
     clifford.py v3: _gerundet(voll, k, seed) = Winkel auf das Clifford-Gitter
     runden, die k Winkel mit dem groessten Abstand (_magie_reihenfolge) frei
     lassen, FRISCHER Kopf, Nachtraining (lr 0,08).
  3. Konfidenz einer Stufe = max. Softmax-Wahrscheinlichkeit ihres Kopfes.
     Ein Bild verlaesst die Kaskade an Stufe i, wenn konf_i > tau; sonst geht
     es zur naechsten Stufe (die letzte Stufe entscheidet immer). Dasselbe tau
     an allen Uebergaengen. tau = 0 -> alle bleiben in Stufe 0; tau = 1 ->
     alle laufen durch alle Stufen.
  4. Kosten je Bild = Summe der T-Gatter aller DURCHLAUFENEN Stufen: jede
     Stufe ist ein eigener Schaltungslauf. Mit Ladekosten kostet jede Stufe
     zusaetzlich das Laden (Amplitudenkodierung, 7.636 T, C-kodierung) --
     quantenphysikalisch zwingend: die Messung der Stufe i zerstoert den
     Zustand, und No-Cloning verbietet eine Kopie auf Vorrat.
     Beide Kosten gelten JE SCHUSS; die Schusszahl (~1.000 je Lauf, I-008)
     ist fuer alle Stufen gleich und kuerzt sich in jedem Verhaeltnis heraus.

ABWEICHUNG VON REGEL 1 (offen benannt): Es wird KEINE eigene Rate je Arm
gesucht. lr = 0,08 fuer das volle Modell und alle Nachtrainings stammt aus
C-mnist (dort je Arm auf Tuning-Seeds 902/903 gesucht, Gitter in beide
Richtungen, nie am Rand). Grund: Laufzeit; die Kaskade vergleicht Stufen
DERSELBEN Modelle, eine falsche Rate traefe Kaskade und feste Stufen gleich.

KOSTEN (Konstanten aus dem LABORBUCH, gemessen, nicht geschaetzt):
  C-hardware, eps = 1e-3: k=0 -> 0, k=2 -> 62, k=8 -> 247, k=32 -> 989 T
  (T ~ 30,9 k, R^2 = 1,000). C-kodierung: exaktes Laden 7.636 T je Lauf.

VORREGISTRIERT (vor der ersten MNIST-Zahl dieses Moduls, Regel 7):
  K-1 Es gibt ein tau, auf der VALIDIERUNG je Seed gewaehlt, bei dem die
      Kaskade die Testgenauigkeit der festen Stufe k=2 erreicht
      (>= acc_k2 - 0,2 pp im Seed-Mittel) mit <= 50 % von deren mittlerer
      Schaltungs-T (62 -> <= 31 T je Bild, Seed-Mittel).
      Wahlregel je Seed: unter allen tau aus TAUS mit
        acc_val(tau) >= acc_val(k2) - 0,2 pp
      das tau mit der kleinsten mittleren Schaltungs-T auf der Validierung
      (Gleichstand -> kleineres tau). Erfuellt keines die Bedingung, gilt
      tau = 1 (alle Stufen) und der Seed wird als „Ausweich“ gezaehlt.
        beide Bedingungen erfuellt       -> H1
        nur Genauigkeit verfehlt         -> H0 (Genauigkeit)
        nur Kosten verfehlt              -> H0 (Kosten)
        beides verfehlt                  -> H0 (beides)
      Entschieden wird auf dem Seed-Mittel (Punktschaetzung, wie in der
      Karte); das gepaarte t-KI der Genauigkeitsdifferenz wird beschreibend
      mitberichtet. Aufloesung: 5.000 Testbilder, 1 Bild = 0,02 pp, die
      Grenze 0,2 pp = 10 Bilder je Seed.
  K-2 (Kontrolle) Konfidenzbasierte Weiterleitung schlaegt ZUFAELLIGE
      Weiterleitung mit gleicher Quote. Die Zufallskontrolle leitet an jedem
      Uebergang genau so viele Bilder weiter wie die Kaskade, aber zufaellig
      gewaehlt -> identische Kosten. Ihre Genauigkeit wird als exakter
      Erwartungswert gerechnet: bei zufaelligen, geschachtelten Teilmengen
      endet jedes Bild mit Wahrscheinlichkeit n_i/N in Stufe i, also
      E[acc] = sum_i (n_i/N) acc_i (kein Ziehungsrauschen; der Vortest prueft
      das gegen Monte-Carlo).
      Statistik: Die tau-Werte eines Seeds teilen Modelle und Bilder, sie
      sind KEINE unabhaengigen Wiederholungen. Gepaart wird deshalb je Seed:
      d_s = Mittel ueber alle tau mit 0 < Weiterleitungsquote(Stufe 0) < 1
      (tau = 0 und tau = 1 sind per Konstruktion gleich) der Differenz
      acc_konfidenz - acc_zufall. Einheit der Statistik = Seed.
      Exakter zweiseitiger Wilcoxon ueber die Seeds; sein kleinstmoegliches
      p ist 2/2^n -- bei n <= 5 Seeds >= 0,0625, also NIE < 0,05. Dann ist
      K-2 nur beschreibend (vorab so festgelegt):
        n <= 5: „B+“ alle Seeds d_s > 0; „B-“ alle d_s < 0; „B~“ gemischt
        n >= 6: H1 mittel > 0 und p < 0,05; H0 mittel < 0 und p < 0,05
                (Zufall besser); sonst HU
  K-3 beschreibend: dieselbe Rechnung MIT Ladekosten. Mittlere T je Bild bei
      tau* (aus K-1) und bei tau*_L (gleiche Wahlregel, aber Kosten mit
      Laden) gegen die feste Stufe k=2 mit Laden (7.698 T). Auf Papier: die
      Kaskade spart mit Laden nur, wenn weniger als ~0,8 % der Bilder
      weitergeleitet werden (62 / 7.698) -- der Lauf misst, wie weit sie
      davon entfernt ist.
  Zusatz (beschreibend, nicht Teil der Familie): zweistufige Kaskade 0 -> 2
      und das volle Modell (32 Winkel, 989 T) als Referenzpunkt.

PROTOKOLL: Mess-Seeds 90-94 (frisch), Rohdaten sofort sichern (Regel 5):
je Seed, Teil und Stufe Vorhersage + Konfidenz jedes Bildes -- daraus ist
jede Zahl dieses Moduls nachrechenbar.
GRENZEN (vorab benannt): Konfidenz aus exakten Born-Wahrscheinlichkeiten;
auf Hardware wird sie aus ~1.000 Schuessen geschaetzt und ist verrauscht
(I-008) -- hier nicht simuliert.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from abteilungen.clifford import (GITTER, LAGEN, QUBITS_JE_DATENSATZ, QireZelle, _gerundet, daten)
from labor.gemeinsam import genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, zusammenfassung

DATENSATZ = "mnist"
STUFEN_K = [0, 2, 8]
T_JE_K = {0: 0, 1: 30, 2: 62, 4: 123, 8: 247, 12: 371, 32: 989}   # C-hardware, eps = 1e-3
T_LADEN = 7_636                                                   # C-kodierung, exakt (Moettoenen)
LR, SCHRITTE = 0.08, 1500      # aus C-mnist (dort je Arm gesucht, nie am Rand) -- Abweichung Regel 1
SEEDS = range(90, 95)
TOL_ACC = 0.002                # 0,2 pp
KOSTEN_ANTEIL = 0.50           # <= 50 % der Schaltungs-T von k=2
ALPHA = 0.05
TAUS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99,
        0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0]

HYPOTHESEN = {
    "K-1": "Mit tau (auf Validierung gewaehlt) erreicht die Kaskade 0->2->8 die Testgenauigkeit "
           "der festen Stufe k=2 (>= acc_k2 - 0,2 pp, Seed-Mittel) mit <= 50 % ihrer Schaltungs-T.",
    "K-2": "Kontrolle: konfidenzbasierte Weiterleitung schlaegt zufaellige Weiterleitung mit "
           "gleicher Quote (gepaart je Seed; bei n <= 5 Seeds nur beschreibend).",
    "K-3": "beschreibend: dieselbe Kurve mit Ladekosten (7.636 T je Schaltungslauf).",
}


# ------------------------------------------------------------------ Kaskade
def t_stufen(stufen_k=STUFEN_K):
    return np.array([T_JE_K[k] for k in stufen_k], dtype=float)


def kaskade(konf, korrekt, tau, t_st, laden=T_LADEN) -> dict:
    """konf, korrekt: Felder (S, N) je Stufe und Bild. Bild verlaesst Stufe i < S-1,
    wenn konf_i > tau; die letzte Stufe entscheidet immer. Kosten = Summe der
    T-Zahlen aller durchlaufenen Stufen (+ laden je Lauf in der Variante mit Laden)."""
    konf, korrekt = np.asarray(konf, float), np.asarray(korrekt, bool)
    S, N = konf.shape
    aktiv = np.ones(N, bool)
    ende = np.full(N, S - 1)
    for i in range(S - 1):
        raus = aktiv & (konf[i] > tau)
        ende[raus] = i
        aktiv &= ~raus
    laeufe = ende + 1
    T = np.cumsum(t_st)[ende]
    anteil_ende = np.bincount(ende, minlength=S) / N
    acc_stufe = korrekt.mean(1)
    return {"tau": float(tau), "acc": float(korrekt[ende, np.arange(N)].mean()),
            "T": float(T.mean()), "T_laden": float((T + laden * laeufe).mean()),
            "laeufe": float(laeufe.mean()), "anteil_ende": anteil_ende.tolist(),
            "quote_weiter0": float(1 - anteil_ende[0]),
            "acc_zufall": float((anteil_ende * acc_stufe).sum())}


def zufall_gezogen(konf, korrekt, tau, rng) -> float:
    """Eine Ziehung der Zufallsweiterleitung (nur fuer den Vortest): an jedem Uebergang
    werden genau so viele zufaellige Bilder weitergereicht wie von der Kaskade."""
    konf, korrekt = np.asarray(konf, float), np.asarray(korrekt, bool)
    S, N = konf.shape
    aktiv, ende = np.ones(N, bool), np.full(N, S - 1)
    zaktiv = np.arange(N)
    zende = np.full(N, S - 1)
    for i in range(S - 1):
        raus = aktiv & (konf[i] > tau)
        ende[raus] = i
        aktiv &= ~raus
        weiter = int(aktiv.sum())
        zaktiv = rng.permutation(zaktiv)
        zende[zaktiv[weiter:]] = i
        zaktiv = zaktiv[:weiter]
    return float(korrekt[zende, np.arange(N)].mean())


def kurve(konf, korrekt, t_st, taus=TAUS):
    return [kaskade(konf, korrekt, t, t_st) for t in taus]


def waehle_tau(kurve_val, acc_ziel, kosten="T"):
    """Wahlregel K-1 (nur Validierung): kleinste mittlere Kosten unter allen tau mit
    acc >= acc_ziel; Gleichstand -> kleineres tau. Keines -> tau = 1, Ausweich."""
    ok = [p for p in kurve_val if p["acc"] >= acc_ziel - 1e-12]
    if not ok:
        return 1.0, True
    best = min(ok, key=lambda p: (p[kosten], p["tau"]))
    return best["tau"], False


def _bei(kurve_, tau):
    return next(p for p in kurve_ if p["tau"] == tau)


# ------------------------------------------------------------------ Arbeiter
def _teile(ds, s):
    xtr, ytr, xte, yte = daten(ds, s)
    nv = len(ytr) // 4
    return xtr[nv:], ytr[nv:], xtr[:nv], ytr[:nv], xte, yte


def _voll(a):
    """(seed, schritte, lr, ds) -> trainierte Winkel des vollen Modells (+ Genauigkeiten)."""
    s, schritte, lr, ds = a
    xtr, ytr, xva, yva, xte, yte = _teile(ds, s)
    torch.manual_seed(s)
    m = QireZelle(n=QUBITS_JE_DATENSATZ[ds])
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    with torch.no_grad():
        return {"seed": s, "winkel": m.theta.detach().flatten().tolist(),
                "val": genauigkeit(m(xva), yva), "test": genauigkeit(m(xte), yte)}


def _stufe(a):
    """(seed, k, schritte, lr, ds, winkel_voll) -> Vorhersage und Konfidenz je Bild
    fuer Validierung und Test (Modell wie clifford.py v3)."""
    s, k, schritte, lr, ds, winkel = a
    xtr, ytr, xva, yva, xte, yte = _teile(ds, s)
    voll = QireZelle(n=QUBITS_JE_DATENSATZ[ds])
    with torch.no_grad():
        voll.theta.copy_(torch.tensor(winkel).reshape(voll.theta.shape))
    m = _gerundet(voll, k, s)
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    out = {"seed": s, "k": k, "frei": int(m.frei.sum().item())}
    with torch.no_grad():
        for teil, x, y in (("val", xva, yva), ("test", xte, yte)):
            p = torch.softmax(m(x), 1)
            konf, pred = p.max(1)
            out[teil] = {"pred": pred.numpy(), "konf": konf.double().numpy(), "y": y.numpy()}
    return out


# ------------------------------------------------------------------ Lauf
def seed_zeile(s, stufen, voll, stufen_k=STUFEN_K) -> dict:
    """stufen: {k: Ausgabe von _stufe}. Rechnet Kurven, tau-Wahl und Rohdaten eines Seeds."""
    t_st = t_stufen(stufen_k)
    z = {"seed": s, "voll": {"val": voll["val"], "test": voll["test"], "T": T_JE_K[32]}, "fest": {}, "roh": {}}
    teile = {}
    for teil in ("val", "test"):
        konf = np.stack([stufen[k][teil]["konf"] for k in stufen_k])
        korrekt = np.stack([stufen[k][teil]["pred"] == stufen[k][teil]["y"] for k in stufen_k])
        teile[teil] = (konf, korrekt)
        z["roh"][teil] = {"y": stufen[stufen_k[0]][teil]["y"].tolist(),
                          **{f"k{k}": {"pred": stufen[k][teil]["pred"].tolist(),
                                       "konf": np.round(stufen[k][teil]["konf"], 7).tolist()} for k in stufen_k}}
    for i, k in enumerate(stufen_k):
        z["fest"][f"k{k}"] = {"val": float(teile["val"][1][i].mean()), "test": float(teile["test"][1][i].mean()),
                              "T": T_JE_K[k], "T_laden": T_JE_K[k] + T_LADEN}
    z["kurve_val"] = kurve(*teile["val"], t_st)
    z["kurve_test"] = kurve(*teile["test"], t_st)
    ziel = z["fest"]["k2"]["val"] - TOL_ACC
    z["tau_stern"], z["ausweich"] = waehle_tau(z["kurve_val"], ziel, "T")
    z["tau_stern_laden"], z["ausweich_laden"] = waehle_tau(z["kurve_val"], ziel, "T_laden")
    z["bei_tau_stern"] = _bei(z["kurve_test"], z["tau_stern"])
    z["bei_tau_stern_laden"] = _bei(z["kurve_test"], z["tau_stern_laden"])
    # Zusatz: zweistufig 0 -> 2
    i2 = [stufen_k.index(0), stufen_k.index(2)]
    t2 = t_stufen([0, 2])
    kv = kurve(teile["val"][0][i2], teile["val"][1][i2], t2)
    kt = kurve(teile["test"][0][i2], teile["test"][1][i2], t2)
    tau2, aus2 = waehle_tau(kv, ziel, "T")
    z["zweistufig"] = {"kurve_test": kt, "tau_stern": tau2, "ausweich": aus2, "bei_tau_stern": _bei(kt, tau2)}
    return z


def lauf(schnell=False, seeds=SEEDS, schritte=SCHRITTE, datensatz=DATENSATZ):
    if schnell:
        seeds, schritte = (90,), 150
    t0 = time.time()
    seeds = list(seeds)
    grund = {g["seed"]: g for g in parallel(_voll, [(s, schritte, LR, datensatz) for s in seeds])}
    print(f"  volle Modelle nach {time.time() - t0:.0f} s", flush=True)
    aufg = [(s, k, schritte, LR, datensatz, grund[s]["winkel"]) for s in seeds for k in STUFEN_K]
    stufen = {}
    for e in parallel(_stufe, aufg):
        stufen.setdefault(e["seed"], {})[e["k"]] = e
    zeilen = []
    for s in seeds:
        z = seed_zeile(s, stufen[s], grund[s])
        z["winkel_voll"] = grund[s]["winkel"]
        z["frei_je_k"] = {f"k{k}": stufen[s][k]["frei"] for k in STUFEN_K}
        zeilen.append(z)
        b = z["bei_tau_stern"]
        print(f"  Seed {s}: k0 {z['fest']['k0']['test']:.4f}  k2 {z['fest']['k2']['test']:.4f}  "
              f"k8 {z['fest']['k8']['test']:.4f} | Kaskade tau*={z['tau_stern']} acc {b['acc']:.4f} "
              f"T {b['T']:.1f}  zufall {b['acc_zufall']:.4f}", flush=True)
    ergebnis = {"abteilung": "clifford", "modul": "kaskade", "karte": "I-015", "version": 1,
                "datensatz": datensatz, "qubits": QUBITS_JE_DATENSATZ[datensatz], "stufen_k": STUFEN_K,
                "T_je_k": T_JE_K, "T_laden": T_LADEN, "taus": TAUS, "lr": LR,
                "lr_herkunft": "C-mnist (je Arm gesucht, nie am Rand); keine eigene Suche -- Abweichung Regel 1",
                "hypothesen": HYPOTHESEN, "schritte": schritte, "seeds": seeds, "zeilen": zeilen,
                "sekunden": time.time() - t0}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    return ergebnis


# ------------------------------------------------------------------ Auswertung
def _t_ki(d):
    from scipy.stats import t as t_vert
    d = np.asarray(d, float)
    if len(d) < 2:
        return -math.inf, math.inf
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return float(d.mean()) - q * se, float(d.mean()) + q * se


def auswertung(zeilen) -> dict:
    n = len(zeilen)
    a = lambda f: np.array([f(z) for z in zeilen], dtype=float)
    aus = {"n": n}
    aus["fest"] = {k: {"acc": zusammenfassung(a(lambda z: z["fest"][k]["test"])),
                       "T": zeilen[0]["fest"][k]["T"], "T_laden": zeilen[0]["fest"][k]["T_laden"]}
                   for k in zeilen[0]["fest"]}
    aus["voll"] = {"acc": zusammenfassung(a(lambda z: z["voll"]["test"])), "T": T_JE_K[32],
                   "T_laden": T_JE_K[32] + T_LADEN}
    # Kurve: Seed-Mittel je tau (Test)
    taus = [p["tau"] for p in zeilen[0]["kurve_test"]]
    aus["kurve"] = [{"tau": t, **{g: float(np.mean([z["kurve_test"][j][g] for z in zeilen]))
                                  for g in ("acc", "acc_zufall", "T", "T_laden", "laeufe", "quote_weiter0")}}
                    for j, t in enumerate(taus)]
    aus["kurve_zweistufig"] = [{"tau": t, **{g: float(np.mean([z["zweistufig"]["kurve_test"][j][g] for z in zeilen]))
                                             for g in ("acc", "acc_zufall", "T", "T_laden")}}
                               for j, t in enumerate(taus)]
    # K-1
    acc_k2 = a(lambda z: z["fest"]["k2"]["test"])
    d = a(lambda z: z["bei_tau_stern"]["acc"]) - acc_k2
    T = a(lambda z: z["bei_tau_stern"]["T"])
    T_k2 = float(T_JE_K[2])
    ki = _t_ki(d)
    aus["K1"] = {"diff_acc": float(d.mean()), "diff_ki_unten": ki[0], "diff_ki_oben": ki[1],
                 "T_mittel": float(T.mean()), "T_anteil_k2": float(T.mean()) / T_k2,
                 "taus": [z["tau_stern"] for z in zeilen], "ausweich": int(sum(z["ausweich"] for z in zeilen)),
                 "acc_ok": bool(d.mean() >= -TOL_ACC), "kosten_ok": bool(T.mean() <= KOSTEN_ANTEIL * T_k2),
                 "laeufe": float(a(lambda z: z["bei_tau_stern"]["laeufe"]).mean())}
    # K-2
    ds = []
    for z in zeilen:
        innen = [p["acc"] - p["acc_zufall"] for p in z["kurve_test"] if 0 < p["quote_weiter0"] < 1]
        ds.append(float(np.mean(innen)) if innen else 0.0)
    ds = np.array(ds)
    d_stern = a(lambda z: z["bei_tau_stern"]["acc"] - z["bei_tau_stern"]["acc_zufall"])
    aus["K2"] = {"d_je_seed": ds.tolist(), "mittel": float(ds.mean()), "p": wilcoxon_exakt(ds),
                 "p_minimum": 2.0 / 2 ** n, "positiv": int((ds > 0).sum()), "negativ": int((ds < 0).sum()),
                 "d_bei_tau_stern": float(d_stern.mean())}
    # K-3
    TL = a(lambda z: z["bei_tau_stern"]["T_laden"])
    TL2 = a(lambda z: z["bei_tau_stern_laden"]["T_laden"])
    k2L = T_k2 + T_LADEN
    aus["K3"] = {"T_laden_bei_tau_stern": float(TL.mean()), "anteil_k2_bei_tau_stern": float(TL.mean()) / k2L,
                 "T_laden_bei_tau_stern_laden": float(TL2.mean()), "anteil_k2_bei_tau_stern_laden": float(TL2.mean()) / k2L,
                 "acc_bei_tau_stern_laden": float(a(lambda z: z["bei_tau_stern_laden"]["acc"]).mean()),
                 "taus_laden": [z["tau_stern_laden"] for z in zeilen],
                 "T_laden_k0": float(T_LADEN), "T_laden_k2": k2L,
                 "quote_weiter0_bei_tau_stern": float(a(lambda z: z["bei_tau_stern"]["quote_weiter0"]).mean())}
    zw = a(lambda z: z["zweistufig"]["bei_tau_stern"]["acc"])
    aus["zweistufig"] = {"diff_acc": float((zw - acc_k2).mean()),
                         "T_mittel": float(a(lambda z: z["zweistufig"]["bei_tau_stern"]["T"]).mean()),
                         "T_laden": float(a(lambda z: z["zweistufig"]["bei_tau_stern"]["T_laden"]).mean())}
    return aus


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum (siehe Kopfkommentar). Praefixe:
    K-1 'H1:', 'H0 (Genauigkeit):', 'H0 (Kosten):', 'H0 (beides):';
    K-2 'B+:', 'B-:', 'B~:' (n <= 5) bzw. 'H1:', 'H0:', 'HU:' (n >= 6); K-3 'beschreibend:'."""
    u = {}
    k1 = a["K1"]
    text = (f"diff {k1['diff_acc'] * 100:+.2f} pp (KI {k1['diff_ki_unten'] * 100:+.2f}..{k1['diff_ki_oben'] * 100:+.2f}), "
            f"T {k1['T_mittel']:.1f} = {k1['T_anteil_k2'] * 100:.0f} % von k2, tau* {k1['taus']}, "
            f"Ausweich {k1['ausweich']}")
    if k1["acc_ok"] and k1["kosten_ok"]:
        u["K-1"] = "H1: Kaskade erreicht k2 mit <= 50 % der Schaltungs-T; " + text
    elif k1["kosten_ok"]:
        u["K-1"] = "H0 (Genauigkeit): " + text
    elif k1["acc_ok"]:
        u["K-1"] = "H0 (Kosten): " + text
    else:
        u["K-1"] = "H0 (beides): " + text
    k2 = a["K2"]
    t2 = (f"d = {k2['mittel'] * 100:+.2f} pp ({k2['positiv']}/{a['n']} Seeds positiv, Wilcoxon p = {k2['p']:.3f}, "
          f"kleinstes moegliches p = {k2['p_minimum']:.4f})")
    if k2["p_minimum"] >= ALPHA:
        if k2["positiv"] == a["n"]:
            u["K-2"] = "B+: nur beschreibend (zu wenige Seeds fuer p < 0,05) -- alle Seeds Konfidenz > Zufall; " + t2
        elif k2["negativ"] == a["n"]:
            u["K-2"] = "B-: nur beschreibend -- alle Seeds Zufall >= Konfidenz; " + t2
        else:
            u["K-2"] = "B~: nur beschreibend -- gemischt; " + t2
    elif k2["mittel"] > 0 and k2["p"] < ALPHA:
        u["K-2"] = "H1: Konfidenz schlaegt Zufall; " + t2
    elif k2["mittel"] < 0 and k2["p"] < ALPHA:
        u["K-2"] = "H0: Zufall besser als Konfidenz; " + t2
    else:
        u["K-2"] = "HU: kein Unterschied nachweisbar; " + t2
    k3 = a["K3"]
    u["K-3"] = (f"beschreibend: mit Laden bei tau* {k3['T_laden_bei_tau_stern']:.0f} T = "
                f"{k3['anteil_k2_bei_tau_stern'] * 100:.1f} % von k2 ({k3['T_laden_k2']:.0f} T); bei tau*_L "
                f"{k3['T_laden_bei_tau_stern_laden']:.0f} T = {k3['anteil_k2_bei_tau_stern_laden'] * 100:.1f} %; "
                f"Weiterleitungsquote {k3['quote_weiter0_bei_tau_stern'] * 100:.1f} % (Sparen erst unter ~0,8 %)")
    return u


# ------------------------------------------------------------------ Vortest
def _vortest_bausteine(rng) -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    S, N = 3, 400
    konf = rng.uniform(0.1, 1.0, (S, N))
    korrekt = rng.uniform(size=(S, N)) < np.array([[0.7], [0.8], [0.9]])
    t_st = t_stufen()
    # tau = 0 -> Stufe 0 allein
    p = kaskade(konf, korrekt, 0.0, t_st)
    if not (p["anteil_ende"][0] == 1.0 and p["T"] == 0.0 and p["laeufe"] == 1.0
            and p["T_laden"] == T_LADEN and abs(p["acc"] - korrekt[0].mean()) < 1e-12):
        fehler(f"tau=0 ist nicht Stufe 0: {p}")
    # tau = 1 -> alle Stufen, auch bei Konfidenz exakt 1,0 (float-Saettigung)
    k1 = konf.copy()
    k1[:, :50] = 1.0
    p = kaskade(k1, korrekt, 1.0, t_st)
    if not (p["anteil_ende"][-1] == 1.0 and p["T"] == t_st.sum() and p["laeufe"] == S
            and p["T_laden"] == t_st.sum() + S * T_LADEN and abs(p["acc"] - korrekt[-1].mean()) < 1e-12):
        fehler(f"tau=1 laeuft nicht durch alle Stufen: {p}")
    # Kostenzaehlung von Hand: 4 Bilder, Ende in Stufe 0, 1, 2, 2
    kh = np.array([[0.9, 0.5, 0.5, 0.2], [0.1, 0.9, 0.5, 0.3], [0.1, 0.1, 0.1, 0.1]])
    rh = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], bool)
    p = kaskade(kh, rh, 0.8, t_st, laden=1000)
    erw_T = (0 + 62 + 309 + 309) / 4
    erw_L = erw_T + 1000 * (1 + 2 + 3 + 3) / 4
    if not (p["anteil_ende"] == [0.25, 0.25, 0.5] and abs(p["T"] - erw_T) < 1e-9
            and abs(p["T_laden"] - erw_L) < 1e-9 and p["acc"] == 0.75 and p["laeufe"] == 2.25):
        fehler(f"Kostenzaehlung von Hand: {p}")
    # Zufallskontrolle: exakter Erwartungswert = Monte-Carlo
    for tau in (0.4, 0.8):
        p = kaskade(konf, korrekt, tau, t_st)
        mc = np.mean([zufall_gezogen(konf, korrekt, tau, rng) for _ in range(3000)])
        if abs(mc - p["acc_zufall"]) > 0.003:
            fehler(f"Zufallserwartung {p['acc_zufall']:.4f} != Monte-Carlo {mc:.4f} (tau {tau})")
    # Konfidenzsignal wirkt: sind die sicheren Bilder richtig, schlaegt die Kaskade den Zufall
    kk = np.where(korrekt, 0.95, 0.5)
    p = kaskade(kk, korrekt, 0.9, t_st)
    if not p["acc"] > p["acc_zufall"] + 0.05:
        fehler("Konfidenz-Kaskade schlaegt Zufall bei perfektem Signal nicht")
    # tau-Wahl: billigstes tau mit genug Genauigkeit; keines -> Ausweich tau = 1
    kv = [{"tau": 0.0, "acc": 0.90, "T": 0}, {"tau": 0.5, "acc": 0.92, "T": 20},
          {"tau": 0.9, "acc": 0.925, "T": 30}, {"tau": 1.0, "acc": 0.93, "T": 309}]
    if waehle_tau(kv, 0.918) != (0.5, False):
        fehler(f"tau-Wahl: {waehle_tau(kv, 0.918)}")
    if waehle_tau(kv, 0.95) != (1.0, True):
        fehler("tau-Wahl: Ausweich-Zweig")
    # Stufenmodelle wie clifford.py v3: k freie Winkel, Rest auf dem Gitter, frischer Kopf
    torch.manual_seed(3)
    voll = QireZelle(n=QUBITS_JE_DATENSATZ[DATENSATZ])
    for k in STUFEN_K:
        m = _gerundet(voll, k, 5)
        frei = m.frei.flatten().bool()
        w = m.winkel().detach().flatten()
        if int(frei.sum()) != k or not torch.allclose(w[~frei] / GITTER, torch.round(w[~frei] / GITTER)):
            fehler(f"Stufenmodell k={k} falsch")
        if torch.equal(m.kopf.weight, voll.kopf.weight):
            fehler(f"Kopf geerbt, k={k}")
    if m.theta.numel() != LAGEN * QUBITS_JE_DATENSATZ[DATENSATZ]:
        fehler("Winkelzahl")
    return ok


def _synth_zeile(rng, s, acc=(0.90, 0.92, 0.93), signal=1.0, N=3000, nv=1500):
    """Synthetischer Seed. signal = 1 -> Konfidenz trennt richtig/falsch; 0 -> reines Rauschen;
    -1 -> falsche Bilder sind SICHERER (Zufall gewinnt)."""
    stufen = {}
    for k, pk in zip(STUFEN_K, acc):
        e = {"seed": s, "k": k, "frei": k}
        for teil, n in (("val", nv), ("test", N)):
            y = rng.integers(0, 10, n)
            richtig = rng.uniform(size=n) < pk
            pred = np.where(richtig, y, (y + 1) % 10)
            konf = np.clip(rng.uniform(0.3, 0.9, n) + signal * np.where(richtig, 0.1, -0.2), 0.11, 0.9999)
            e[teil] = {"pred": pred, "konf": konf, "y": y}
        stufen[k] = e
    voll = {"val": 0.94, "test": 0.94}
    return seed_zeile(s, stufen, voll)


def vortest() -> bool:
    """Pflicht-Vortest: Bausteine + jeden Zweig von K-1/K-2 erzwingen, K-3 laeuft."""
    rng = np.random.default_rng(15)
    ok = _vortest_bausteine(rng)

    def pruefe(zeilen, erwartet, name):
        nonlocal ok
        u = urteil(auswertung(zeilen))
        for h, praefix in erwartet.items():
            if not u[h].startswith(praefix):
                ok = False
                print(f"  VORTEST FEHLER {name}: {h} -> {u[h]}")
        if not u["K-3"].startswith("beschreibend:"):
            ok = False
            print("  VORTEST FEHLER: K-3")

    # K-1 H1: starkes Signal, Stufe 0 fast so gut wie k2
    pruefe([_synth_zeile(rng, s, acc=(0.915, 0.92, 0.93), signal=3.0) for s in range(5)], {"K-1": "H1:", "K-2": "B+:"}, "H1")
    # K-1 H0 (Kosten): Signal da, aber k0 schwach -> viele Bilder muessen weiter
    pruefe([_synth_zeile(rng, s, acc=(0.60, 0.92, 0.93)) for s in range(5)], {"K-1": "H0 (Kosten):"}, "Kosten")
    # K-2 B-: Konfidenz verkehrt herum
    pruefe([_synth_zeile(rng, s, signal=-1.0) for s in range(5)], {"K-2": "B-:"}, "B-")
    # K-2 n >= 6: H1 bzw. H0
    pruefe([_synth_zeile(rng, s) for s in range(7)], {"K-2": "H1:"}, "n7 H1")
    pruefe([_synth_zeile(rng, s, signal=-1.0) for s in range(7)], {"K-2": "H0:"}, "n7 H0")
    # K-1 Genauigkeit / beides und K-2 B~ / HU: synthetische Auswertungen (Zweige direkt)
    basis = auswertung([_synth_zeile(rng, s) for s in range(5)])
    for acc_ok, kosten_ok, praefix in ((False, True, "H0 (Genauigkeit):"), (False, False, "H0 (beides):"),
                                       (True, True, "H1:"), (True, False, "H0 (Kosten):")):
        a = {**basis, "K1": {**basis["K1"], "acc_ok": acc_ok, "kosten_ok": kosten_ok}}
        if not urteil(a)["K-1"].startswith(praefix):
            ok = False
            print("  VORTEST FEHLER K-1-Zweig", praefix)
    for n, pos, neg, mittel, p, praefix in ((5, 3, 2, 0.001, 0.5, "B~:"), (7, 4, 3, 0.001, 0.6, "HU:"),
                                            (7, 7, 0, 0.01, 0.016, "H1:"), (7, 0, 7, -0.01, 0.016, "H0:")):
        a = {**basis, "n": n, "K2": {**basis["K2"], "positiv": pos, "negativ": neg, "mittel": mittel, "p": p,
                                     "p_minimum": 2.0 / 2 ** n}}
        if not urteil(a)["K-2"].startswith(praefix):
            ok = False
            print("  VORTEST FEHLER K-2-Zweig", praefix)
    # Aufloesung: bei n = 5 kann der exakte Wilcoxon nie < 0,05 werden
    if wilcoxon_exakt([0.01, 0.02, 0.03, 0.04, 0.05]) < ALPHA:
        ok = False
        print("  VORTEST FEHLER: Wilcoxon-Minimum bei n=5")
    return ok


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell)
    print(speichere("I-015" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
