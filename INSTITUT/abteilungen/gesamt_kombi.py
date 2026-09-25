"""Abteilung Clifford/Hardware (Karte I-034, Quantenstufe Q3) -- Gesamt-Magie-Budget KOMBINIERT.

HEILMEIER:
- Was wollen wir?  Die drei Spar-Befunde des Instituts zusammensetzen und die
  Hardware-T je Bild MESSEN statt schaetzen:
    (1) Hybrid-Kaskade (I-015-hybrid): Stufe 0 = Clifford-Schaltung (0 T) laeuft
        KLASSISCH auf dem exakten Bildvektor (kein Laden); nur unsichere Bilder
        gehen auf den Quantenrechner;
    (2) Top-m-Laden (I-033): nur die m Gray-Code-Winkel phi der duennen
        Praeparation mit dem groessten Abstand zum pi/2-Gitter bleiben exakt;
    (3) Magie-Konzentration (C-mnist/C-referenz): Quantenstufen mit k = 2 bzw.
        k = 8 Magiewinkeln.
  Kaskade 0 (klassisch, exakt) -> k2 (Top-m geladen) -> k8 (Top-m geladen).
- Wie heute?  Feste Stufe k = 2 auf Hardware, dicht exakt geladen: 7.636 + 62
  = 7.698 T je Bild (Referenz „heute“). Hybrid-Kaskade mit exaktem Laden:
  I-015-hybrid 902 T = 11,7 %. Top-48-Laden allein: 1.479 T (45 % von duenn
  exakt), -0,57 pp (I-033).
- Was ist neu?  Die Kombination -- und ob sie sich stoert: genaehertes Laden
  macht die Quantenstufen fuer genau die schwierigen Bilder unsicherer, auf die
  sich die Kaskade verlaesst (Wechselwirkung, G-2).
- Woran merken wir es?  Mittlere Hardware-T je Bild (gridsynth, eps = 1e-3,
  bildweise) gegen Testgenauigkeit, gepaart je Seed gegen k2 dicht.

KOSTENMODELL (Q3, vorab festgelegt):
  Stufe 0 (k = 0, Clifford) klassisch: 0 T, kein Laden (Additionen beschreibend,
  kaskade_hybrid.additionen_stufe0).  Jede BESUCHTE Quantenstufe k >= 1 kostet
  Lade-T(m, Bild) + T_JE_K[k]; das Bild wird fuer jede Quantenstufe NEU geladen
  (Messung zerstoert den Zustand, No-Cloning). Ein Bild, das bis k8 laeuft,
  kostet also 2 x Lade-T(m, Bild) + 62 + 247.  Lade-T(m, Bild) = Summe der
  gridsynth-T ueber die exakt behaltenen Nicht-Clifford-Winkel phi dieses Bildes
  (lade_magie.t_formel, wie I-033); gerundete Winkel kosten 0 T.
  Varianten „exakt“: dicht = 7.636 T konstant (C-kodierung, I-015-hybrid);
  duenn = alle magischen phi der duennen Praeparation, bildweise (~3.260 T).
  Zaehlung: gridsynth kostet ~1,5 ms je Winkel -> nicht fuer alle 6.500 Bilder.
    * tau-Wahl auf VALIDIERUNG mit Kostenmodell L(Bild) = #magische behaltene
      phi(Bild) x t_rot, t_rot = mittlere gridsynth-T je magischem Winkel,
      exakt kalibriert auf den ersten N_KALIB Validierungsbildern je Variante;
    * berichtete Test-T bei tau*: EXAKT gezaehlt auf allen an den Quanten-
      rechner weitergeleiteten Testbildern (bei mehr als N_EXAKT_MAX eine
      Zufallsstichprobe davon, Verhaeltnisschaetzer); dieser Wert entscheidet G-1.
      Das Modell wird daneben berichtet (Pruefung, beschreibend).
    * feste Stufen ohne Kaskade: exakt auf den ersten N_KOSTEN_FEST Testbildern.

AUFBAU JE SEED (Kaskaden-Rezept I-015/I-015-hybrid, Laden-Rezept I-033):
  Daten wie kaskade.py: MNIST 16x16, 6.000 Trainingsbilder (erstes Viertel =
  1.500 Validierung, 4.500 Training), 5.000 Testbilder; Test nie zur Wahl.
  Phase 1: volle Zelle (clifford._baue("voll")) auf EXAKTEN Zustaenden, 1.500
  Schritte.  Stufenmodelle je (k, Laden), k in {0, 2, 8}: clifford._gerundet(voll,
  k, seed) (Rundung, k Magiewinkel frei, FRISCHER Kopf), 1.500 Schritte, gleiche
  Rate, trainiert UND ausgewertet auf den Zustaenden ihres Ladens:
    k0_exakt (Stufe 0, klassisch), k2_exakt, k8_exakt (Kaskade exakt geladen,
    I-015-hybrid-Replikation; k2_exakt = Referenz „k2 dicht“), k2_top32, k8_top32,
    k2_top48, k8_top48.  = 1 + 7 Trainings je Seed.  Identitaet (Kopf auf |x|^2
    der exakten Zustaende, nur Kopf) = Positivkontrolle.
  Konfidenz = max. Softmax; Kaskade und tau-Wahl wie kaskade_hybrid.seed_hybrid:
  gleiches tau an allen Uebergaengen, Kandidaten kaskade.TAUS, Wahl = kleinste
  (Modell-)Hybrid-Kosten auf Validierung mit acc_val >= acc_val(k2_exakt) - 0,2 pp
  (Ziel ist die Referenz „heute“, NICHT die k2-Stufe desselben Ladens); keines ->
  tau = 1, „Ausweich“ gezaehlt.  kaskade.py bleibt unveraendert: die bildweise
  Kostenrechnung steht hier (kaskade_bild), der Vortest prueft sie gegen
  kaskade.kaskade mit konstanten Ladekosten.

LERNRATE (Regel 1, begruendet uebernommen): 0,08 fuer voll, Stufenmodelle und
  Identitaet aus ergebnisse/C-mnist (je Arm auf Tuning-Seeds gesucht, Gitter in
  beide Richtungen, Optimum innen; voll = 32 freie Winkel und rundung = 0 freie
  Winkel fanden BEIDE 0,08 -> die Rate traegt ueber 0..32 freie Winkel). Auf
  Top-m-Zustaenden ist 0,08 mit ALLEN 32 Winkeln frei in I-033 auf 10/10 Seeds
  stabil (top_48 -0,1 ... -0,9 pp); hier sind nur 2 bzw. 8 Winkel frei.  Keine
  0,32.  Ersatz ohne C-mnist-Datei: 0,08 (lade_magie._raten).  Instabilitaet wird
  trotzdem je Arm gezaehlt (G-3): Seeds mit acc < acc_k2_dicht(Seed) - 1 pp,
  und die Referenz selbst: Seeds mit k2_dicht < Median(k2_dicht) - 1 pp.

VORREGISTRIERUNG (Regel 7; steht vor jeder Zahl dieser Seeds):
  Mess-Seeds 210-219 (frisch, 10), Entwurfs-Seeds fuer Rauchtests 990/991.
  Referenz acc_ref = Testgenauigkeit k2_exakt (feste Stufe k=2, exakt geladen;
  dicht und duenn geladen ist derselbe Zustand), T_ref = 7.698 T.
  PK Positivkontrolle: d = acc(k2_exakt) - acc(identitaet); H1 wenn mittel > 0
     und exakter Wilcoxon zweiseitig p < 0,05; sonst H0.  Feuert sie nicht, ist
     G-1 „HU (Vorbedingung)“ (die Referenz waere dann nichts, was man halten muss).
  G-1 Haupt (= Abbruchkriterium der Karte): Hauptkonfiguration kombi_m48, zweite
     kombi_m32 (vorab benannt, Holm ueber genau diese beiden, feste Familie).
     Je Konfiguration: d_s = acc_kombi - acc_ref (Seed s), T_hw = Seed-Mittel der
     exakt gezaehlten Test-T bei tau*, Anteil = T_hw / 7.698.
     p_NI = einseitiger gepaarter t-Test H0: mittel(d) <= -0,5 pp (entspricht der
     unteren Grenze des einseitigen 95-%-KI), Holm-adjustiert ueber {m48, m32}.
       erfuellt(m): Anteil <= 5 % UND p_NI,Holm < 0,05;
       versagt(m):  Anteil > 10 % ODER obere Grenze des einseitigen 95-%-KI von d
                    < -0,5 pp (nachweislich schlechter).
     H1: mindestens eine Konfiguration erfuellt (m48 zuerst berichtet).
     H0: JEDE Konfiguration versagt (geschaerft gegenueber „beide > 10 % oder
         beide schlechter“: auch gemischt -- eine zu teuer, eine schlechter -- ist
         H0; beides ist ein Versagen der Kombination). Sonst HU.
  G-2 Wechselwirkung (Test): d_s = acc(kombi_m48) - acc(hybrid_dicht) (Kaskade
     mit exaktem Laden, gleiche Stufe 0, eigenes tau* nach ihren Kosten 7.636 T).
     Frage: kostet genaehertes Laden IN der Kaskade Genauigkeit?
       H1: mittel < 0 und exakter Wilcoxon zweiseitig p < 0,05 (Verlust belegt);
       H0: TOST +-0,5 pp mit exaktem einseitigem Wilcoxon (beide p < 0,05) ->
           kein relevanter Verlust; ebenso H0 (Gegenrichtung) bei mittel > 0 und
           p < 0,05;  sonst HU.
     Beschreibend dazu: dasselbe fuer m32 und gegen hybrid_duenn; Wechselwirkung
     (kombi_m - hybrid_dicht) - (k2_topm - k2_exakt), also Verlust in der Kaskade
     gegen Verlust der festen Stufe (Wilcoxon berichtet, kein Urteil).
  G-3 beschreibend: Pareto T gegen Genauigkeit ueber (Laden in {m32, m48, exakt
     dicht, exakt duenn}) x (Kaskade ja/nein) + zweistufig 0->2/0->8 (m48);
     Quantenanteil der Bilder; Einzelseed-Instabilitaet (Arm < Referenz - 1 pp);
     Referenz-Ausreisser; Kostenmodell gegen exakte Zaehlung.

AUFLOESUNG (Regel 6, vor dem Lauf): 5.000 Testbilder, 1 Bild = 0,02 pp; die
  Grenze 0,5 pp = 25 Bilder je Seed.  Streuung der gepaarten Differenz
  Kaskade - k2 in I-015-hybrid: einseitiges KI [-0,20; +0,31] um +0,06 bei 10
  Seeds -> sd ~0,44 pp, Halbbreite des einseitigen 95-%-KI ~0,26 pp; mit Holm
  (m48 allein, alpha/2) ~0,32 pp.  G-1 kann also H1 werden, wenn der wahre
  Verlust der Kombination kleiner als ~0,2 pp ist; der feste Top-48-Verlust aus
  I-033 (-0,57 pp) wuerde allein schon scheitern -- die Kaskade muss ihn
  auffangen (nur ~10 % der Bilder sind betroffen).  Kosten: Papierschaetzung
  der Karte 2-3 % (Quantenanteil 8-14 % x (1.479 + 247) T) -- die 5-%-Grenze
  ist an dieser ausgerechneten Schranke ausgerichtet (Lehre I-020), mit ~2x
  Abstand; >10 % hiesse, die Kombination bringt gegenueber I-015-hybrid (11,7 %)
  nichts.  Exakter Wilcoxon: kleinstes p bei 10 Seeds 2/1024.

LAUFZEIT (4 Prozesse): je Seed 8 Trainings a ~50-70 s (4.500 Bilder, 1.500
  Schritte; I-033: 10.000 Bilder ~125-155 s), OMP der duennen Winkel fuer 11.000
  Bilder ~95 s, gridsynth-Zaehlung ~1-2 min (Cache je Winkel: Top-32 c Top-48 c
  duenn) -> ~11-13 CPU-min je Seed, 10 Seeds / 4 Prozesse ~ 35-45 min.

GRENZEN (vorab): rauschfrei simuliert, Konfidenz aus exakten Born-
  Wahrscheinlichkeiten (auf Hardware ~1.000 Schuesse, I-008); eps je Rotation
  fest; OMP-Duennloesung heuristisch (T obere Schranke); Top-m nach Gitter-
  abstand, nicht nach Wirkung; Test-T bei sehr vielen weitergeleiteten Bildern
  aus Stichprobe; Schuesse kuerzen sich in jedem T-Verhaeltnis heraus.
"""
from __future__ import annotations

import base64
import math
import os
import shutil
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from abteilungen import kaskade as K
from abteilungen.clifford import _baue, _gerundet
from abteilungen.datenmenge import holm, wilcoxon_einseitig
from abteilungen.gesamtbudget import baum_zustand
from abteilungen.kaskade_hybrid import _ki_einseitig, additionen_stufe0, t_hybrid
from abteilungen.kodierung import normiere
from abteilungen.lade_magie import _raten, arm_winkel, arm_zustand, duenne_winkel, ebenen, magisch, t_formel
from labor.gemeinsam import bilder, genauigkeit, parallel, speichere, trainiere, wilcoxon_exakt, zusammenfassung

SEEDS = tuple(range(210, 220))
ENTWURF_SEEDS = (990, 991)            # nur Rauchtests
N = 8
N_TRAIN_GESAMT, N_TEST, SCHRITTE = 6_000, 5_000, 1500   # wie clifford.daten / kaskade._teile
N_KALIB, N_KOSTEN_FEST, N_EXAKT_MAX = 60, 150, 300
T_K2_DICHT = K.T_LADEN + K.T_JE_K[2]  # 7.698 T, Referenz „heute“
ANTEIL_H1, ANTEIL_H0 = 0.05, 0.10
GRENZE = 0.005                        # Nichtunterlegenheit / TOST 0,5 pp
INSTAB = 0.01                         # Einzelseed-Instabilitaet: < Referenz - 1 pp
ALPHA = 0.05
TOP_M = (48, 32)
HAUPT = ("kombi_m48", "kombi_m32")    # G-1-Familie (Holm), m48 = Hauptkonfiguration

# Stufenmodell -> (k, Laden). Laden: "exakt" oder lade_magie-Arm "top_m".
STUFENMODELLE = {"k0_exakt": (0, "exakt"), "k2_exakt": (2, "exakt"), "k8_exakt": (8, "exakt"),
                 **{f"k{k}_top{m}": (k, f"top_{m}") for m in TOP_M for k in (2, 8)}}
# Kosten-Laden je Variante: "dicht" konstant, sonst bildweise (lade_magie-Arm)
LADE_ARM = {"top_48": "top_48", "top_32": "top_32", "duenn": "referenz"}
VARIANTEN = {"kombi_m48": (["k0_exakt", "k2_top48", "k8_top48"], "top_48"),
             "kombi_m32": (["k0_exakt", "k2_top32", "k8_top32"], "top_32"),
             "hybrid_dicht": (["k0_exakt", "k2_exakt", "k8_exakt"], "dicht"),
             "hybrid_duenn": (["k0_exakt", "k2_exakt", "k8_exakt"], "duenn"),
             "zwei_0_2_m48": (["k0_exakt", "k2_top48"], "top_48"),
             "zwei_0_8_m48": (["k0_exakt", "k8_top48"], "top_48")}
# feste Stufen (ohne Kaskade): Modell, Kosten-Laden ("klassisch" = 0 T)
FESTE = {"k0_klassisch": ("k0_exakt", "klassisch"), "k2_dicht": ("k2_exakt", "dicht"),
         "k2_duenn": ("k2_exakt", "duenn"), "k8_dicht": ("k8_exakt", "dicht"),
         "k2_top48": ("k2_top48", "top_48"), "k8_top48": ("k8_top48", "top_48"),
         "k2_top32": ("k2_top32", "top_32"), "k8_top32": ("k8_top32", "top_32")}

HYPOTHESEN = {
    "PK": "Positivkontrolle: feste Stufe k=2 exakt geladen (Referenz) schlaegt die Identitaet (nur Kopf "
          "auf |x|^2). H1: mittel > 0 und exakter Wilcoxon zweiseitig p < 0,05; sonst H0 (dann G-1 HU).",
    "G-1": "Haupt (Abbruchkriterium I-034): Kaskade 0(klassisch, exakt)->k2(Top-m)->k8(Top-m), tau auf "
           "Validierung nach bildweisen Hybrid-Kosten. Hauptkonfiguration m=48, zweite m=32 (Holm ueber "
           "beide). H1, wenn eine Konfiguration mittlere Hardware-T je Bild <= 5 % von 7.698 T hat UND "
           "die Nichtunterlegenheit gegen k2 dicht haelt (einseitiger gepaarter t-Test gegen -0,5 pp, "
           "Holm-p < 0,05 = untere KI-Grenze >= -0,5 pp). H0, wenn jede Konfiguration > 10 % kostet oder "
           "nachweislich schlechter ist (obere einseitige KI-Grenze < -0,5 pp). Sonst HU.",
    "G-2": "Wechselwirkung: d = acc(Kaskade Top-48) - acc(Kaskade exakt geladen, 7.636 T). H1 (genaehertes "
           "Laden kostet in der Kaskade): mittel < 0 und exakter Wilcoxon zweiseitig p < 0,05. H0: TOST "
           "+-0,5 pp (exakter einseitiger Wilcoxon, beide p < 0,05) oder Gegenrichtung (mittel > 0, "
           "p < 0,05). Sonst HU. Beschreibend: m=32, gegen duenn exakt, Verlust Kaskade vs. feste Stufe.",
    "G-3": "beschreibend: Pareto T gegen Genauigkeit ueber (m, Kaskade ja/nein), Quantenanteil der Bilder, "
           "Einzelseed-Instabilitaet (Arm < Referenz - 1 pp), Referenz-Ausreisser, Kostenmodell vs. exakt.",
}


# ------------------------------------------------------------------ 1. Kaskade mit bildweisen Kosten
def verlauf(konf, tau) -> np.ndarray:
    """Endstufe je Bild, identische Regel wie kaskade.kaskade (konf_i > tau -> raus)."""
    konf = np.asarray(konf, float)
    S, n = konf.shape
    aktiv, ende = np.ones(n, bool), np.full(n, S - 1)
    for i in range(S - 1):
        raus = aktiv & (konf[i] > tau)
        ende[raus] = i
        aktiv &= ~raus
    return ende


def _kosten_je_bild(ende, stufen_k, laden_bild):
    """(Besuche der Quantenstufen, Schaltungs-T, Gesamt-T) je Bild. Stufe k = 0 ist klassisch."""
    quant = np.array([k >= 1 for k in stufen_k], dtype=float)
    schalt = np.array([K.T_JE_K[k] if k >= 1 else 0.0 for k in stufen_k], dtype=float)
    besuche = np.cumsum(quant)[ende]
    t_schalt = np.cumsum(schalt)[ende]
    return besuche, t_schalt, besuche * np.asarray(laden_bild, float) + t_schalt


def kaskade_bild(konf, korrekt, tau, stufen_k, laden_bild) -> dict:
    """Wie kaskade.kaskade, aber Hybrid-Kosten mit BILDWEISEN Ladekosten laden_bild (Skalar oder (N,)):
    T(Bild) = Summe ueber besuchte Quantenstufen von (laden_bild + T_JE_K[k])."""
    konf, korrekt = np.asarray(konf, float), np.asarray(korrekt, bool)
    S, n = konf.shape
    ende = verlauf(konf, tau)
    besuche, _, T = _kosten_je_bild(ende, stufen_k, laden_bild)
    anteil_ende = np.bincount(ende, minlength=S) / n
    return {"tau": float(tau), "acc": float(korrekt[ende, np.arange(n)].mean()), "T": float(T.mean()),
            "besuche": float(besuche.mean()), "quote_quantum": float((besuche > 0).mean()),
            "anteil_ende": anteil_ende.tolist(), "acc_zufall": float((anteil_ende * korrekt.mean(1)).sum())}


def t_test_exakt(ende, stufen_k, laden_fn, cap, rng) -> dict:
    """Mittlere Test-T je Bild mit EXAKT gezaehlten Ladekosten der weitergeleiteten Bilder.
    laden_fn(b) -> exakte Lade-T von Bild b. Mehr als cap weitergeleitete Bilder ->
    Zufallsstichprobe (Verhaeltnisschaetzer fuer die Summe der Ladekosten)."""
    n = len(ende)
    besuche, t_schalt, _ = _kosten_je_bild(ende, stufen_k, 0.0)
    fwd = np.flatnonzero(besuche > 0)
    if len(fwd) == 0:
        return {"T": float(t_schalt.mean()), "gezaehlt": 0, "weiter": 0, "L_mittel_weiter": None}
    s = fwd if len(fwd) <= cap else np.sort(rng.choice(fwd, cap, replace=False))
    L = np.array([laden_fn(int(b)) for b in s], dtype=float)
    summe = float((besuche[s] * L).sum()) * len(fwd) / len(s)
    return {"T": (float(t_schalt.sum()) + summe) / n, "gezaehlt": int(len(s)), "weiter": int(len(fwd)),
            "L_mittel_weiter": float(L.mean())}


# ------------------------------------------------------------------ 2. Rohdaten kompakt
def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes()).decode("ascii")


def roh_lesen(s: str, dtype) -> np.ndarray:
    """Gegenstueck zu _b64: konf mit np.float32, pred/y mit np.uint8, n_mag mit np.uint8."""
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


# ------------------------------------------------------------------ 3. Arbeiter
def _pfad(ordner, seed):
    return os.path.join(ordner, f"seed_{seed}.npz"), os.path.join(ordner, f"voll_{seed}.pt")


def _vorbereitung(a):
    """Je Seed: Daten (kaskade-Teilung), duenne Winkel (Platte), Phase 1 (voll, exakt), Identitaet."""
    seed, n_train, n_test, schritte, lr, ordner = a[:6]
    ds = a[6] if len(a) > 6 else "mnist"            # Direktor 25.09.: Datensatz waehlbar (I-034-fashion)
    t0 = time.time()
    xtr, ytr, xte, yte = bilder(ds, seed, n_train=n_train, n_test=n_test, seite=16)
    nv = len(ytr) // 4                                  # wie kaskade._teile
    p_all, p_te = normiere(xtr.numpy()), normiere(xte.numpy())
    p_tr, p_va = p_all[nv:], p_all[:nv]
    y_tr, y_va = ytr[nv:], ytr[:nv]
    phi = {t: duenne_winkel(p) for t, p in (("tr", p_tr), ("va", p_va), ("te", p_te))}
    fehler = max(float(np.abs(baum_zustand(ebenen(phi[t][:30])) - p[:30]).max())
                 for t, p in (("tr", p_tr), ("va", p_va), ("te", p_te)))
    t_omp = time.time() - t0
    np.savez(_pfad(ordner, seed)[0], phi_tr=phi["tr"], phi_va=phi["va"], phi_te=phi["te"], p_tr=p_tr, p_va=p_va,
             p_te=p_te, y_tr=y_tr.numpy(), y_va=y_va.numpy(), y_te=yte.numpy())
    f = lambda p: torch.tensor(p, dtype=torch.float32)
    voll, _ = _baue("voll", seed, N)
    trainiere(voll, f(p_tr), y_tr, schritte, lr["voll"], seed=seed)
    torch.save(voll.state_dict(), _pfad(ordner, seed)[1])
    with torch.no_grad():
        acc_voll = {"val": genauigkeit(voll(f(p_va)), y_va), "test": genauigkeit(voll(f(p_te)), yte)}
    ident, _ = _baue("ohne_schaltung", seed, N)        # Winkel fest -> nur Kopf (wie lade_magie)
    with torch.no_grad():
        mtr, mte = ident.merkmale(f(p_tr)), ident.merkmale(f(p_te))
    trainiere(ident.kopf, mtr, y_tr, schritte, lr["ohne_schaltung"], seed=seed)
    with torch.no_grad():
        acc_ident = genauigkeit(ident.kopf(mte), yte)
    return {"seed": seed, "voll": acc_voll, "identitaet": acc_ident, "rekonstruktion_fehler": fehler,
            "sekunden_omp": t_omp, "sekunden": time.time() - t0}


def _zustaende(d, laden, teil):
    if laden == "exakt":
        return d[f"p_{teil}"]
    return arm_zustand(d[f"phi_{teil}"], laden)


def _stufe_lauf(a):
    """Stufenmodell (k, Laden): _gerundet(voll, k) mit frischem Kopf, Training auf den Zustaenden
    seines Ladens, Vorhersage + Konfidenz je Bild auf Validierung und Test."""
    seed, name, schritte, lr, ordner = a
    k, laden = STUFENMODELLE[name]
    t0 = time.time()
    d = dict(np.load(_pfad(ordner, seed)[0]))
    f = lambda p: torch.tensor(p, dtype=torch.float32)
    voll, _ = _baue("voll", seed, N)
    voll.load_state_dict(torch.load(_pfad(ordner, seed)[1]))
    m = _gerundet(voll, k, seed)
    trainiere(m, f(_zustaende(d, laden, "tr")), torch.tensor(d["y_tr"]), schritte, lr["rundung"], seed=seed)
    out = {"seed": seed, "name": name, "k": k, "laden": laden, "frei": int(m.frei.sum().item())}
    with torch.no_grad():
        for teil in ("va", "te"):
            z = _zustaende(d, laden, teil)
            p = torch.softmax(m(f(z)), 1)
            konf, pred = p.max(1)
            out[teil] = {"konf": konf.double().numpy(), "pred": pred.numpy().astype(np.int64)}
            if teil == "te":
                out["treue_test"] = float(np.mean(np.sum(z * d["p_te"], axis=1) ** 2))
                out["norm_fehler"] = float(np.abs(np.linalg.norm(z, axis=1) - 1).max())
    out["sekunden"] = time.time() - t0
    return out


def seed_zeile(seed, stufen, vor, phi_va, phi_te, y_va, y_te, n_kalib=N_KALIB, n_fest=N_KOSTEN_FEST,
               cap=N_EXAKT_MAX) -> dict:
    """Kosten (Modell + exakt), Kaskaden, tau-Wahl und Rohdaten eines Seeds.
    stufen: {name: Ausgabe von _stufe_lauf}; vor: Ausgabe von _vorbereitung."""
    t0 = time.time()
    rng = np.random.default_rng([int(seed), 34])
    phi = {"va": phi_va, "te": phi_te}
    y = {"va": np.asarray(y_va), "te": np.asarray(y_te)}
    cache = {}

    def t_exakt(teil, arm, b):
        key = (teil, arm, b)
        if key not in cache:
            cache[key] = t_formel(arm_winkel(phi[teil][b:b + 1], arm)[0])
        return cache[key]

    # Kostenmodell je Lade-Variante: #magische behaltene phi x t_rot (Kalibrierung auf Validierung)
    laden = {}
    for lv, arm in LADE_ARM.items():
        nm = {t: magisch(arm_winkel(phi[t], arm)).sum(1) for t in ("va", "te")}
        kv = min(n_kalib, len(phi_va))
        t_sum = sum(t_exakt("va", arm, b) for b in range(kv))
        t_rot = t_sum / max(1, int(nm["va"][:kv].sum()))
        kf = min(n_fest, len(phi_te))
        t_fest = [t_exakt("te", arm, b) for b in range(kf)]
        laden[lv] = {"arm": arm, "t_rot": float(t_rot), "n_mag_val": nm["va"], "n_mag_test": nm["te"],
                     "modell_val": nm["va"] * t_rot, "modell_test": nm["te"] * t_rot,
                     "T_fest_exakt": float(np.mean(t_fest)), "T_fest_modell": float(np.mean(nm["te"] * t_rot)),
                     "n_mag_mittel_test": float(nm["te"].mean())}

    def laden_modell(lv, teil):
        if lv == "dicht":
            return float(K.T_LADEN)
        return laden[lv]["modell_val" if teil == "va" else "modell_test"]

    korrekt = {n: {t: stufen[n][t]["pred"] == y[t] for t in ("va", "te")} for n in stufen}
    z = {"seed": seed, "identitaet": vor["identitaet"], "voll": vor["voll"], "fest": {}, "kaskaden": {},
         "frei": {n: stufen[n]["frei"] for n in stufen},
         "treue_test": {n: stufen[n]["treue_test"] for n in stufen},
         "norm_fehler": max(stufen[n]["norm_fehler"] for n in stufen)}
    # feste Stufen
    for fname, (modell, lv) in FESTE.items():
        k = STUFENMODELLE[modell][0]
        if lv == "klassisch":
            T, Tm = 0.0, 0.0
        elif lv == "dicht":
            T = Tm = float(K.T_LADEN + K.T_JE_K[k])
        else:
            T = laden[lv]["T_fest_exakt"] + K.T_JE_K[k]
            Tm = laden[lv]["T_fest_modell"] + K.T_JE_K[k]
        z["fest"][fname] = {"val": float(korrekt[modell]["va"].mean()), "test": float(korrekt[modell]["te"].mean()),
                            "T": float(T), "T_modell": float(Tm)}
    ziel = z["fest"]["k2_dicht"]["val"] - K.TOL_ACC
    # Kaskaden
    for vname, (namen, lv) in VARIANTEN.items():
        st = [STUFENMODELLE[n][0] for n in namen]
        mat = {t: (np.stack([stufen[n][t]["konf"] for n in namen]), np.stack([korrekt[n][t] for n in namen]))
               for t in ("va", "te")}
        kv = [kaskade_bild(*mat["va"], tau, st, laden_modell(lv, "va")) for tau in K.TAUS]
        kt = [kaskade_bild(*mat["te"], tau, st, laden_modell(lv, "te")) for tau in K.TAUS]
        tau, aus = K.waehle_tau(kv, ziel, "T")
        p = K._bei(kt, tau)
        ende = verlauf(mat["te"][0], tau)
        if lv == "dicht":
            ex = {"T": p["T"], "gezaehlt": 0, "weiter": int(round(p["quote_quantum"] * len(ende))),
                  "L_mittel_weiter": float(K.T_LADEN)}
        else:
            arm = LADE_ARM[lv]
            ex = t_test_exakt(ende, st, lambda b, arm=arm: t_exakt("te", arm, b), cap, rng)
        z["kaskaden"][vname] = {"stufen": namen, "laden": lv, "tau": tau, "ausweich": bool(aus), "acc": p["acc"],
                                "T_hw": float(ex["T"]), "T_modell": p["T"], "anteil": float(ex["T"]) / T_K2_DICHT,
                                "quote_quantum": p["quote_quantum"], "besuche": p["besuche"],
                                "anteil_ende": p["anteil_ende"], "acc_zufall": p["acc_zufall"],
                                "gezaehlt": ex["gezaehlt"], "weiter": ex["weiter"],
                                "L_mittel_weiter": ex["L_mittel_weiter"],
                                "kurve_test": [{g: q[g] for g in ("tau", "acc", "T", "quote_quantum")} for q in kt]}
    z["laden"] = {lv: {g: v[g] for g in ("arm", "t_rot", "T_fest_exakt", "T_fest_modell", "n_mag_mittel_test")}
                  for lv, v in laden.items()}
    z["roh"] = {"y_va": _b64(y["va"], np.uint8), "y_te": _b64(y["te"], np.uint8),
                "stufen": {n: {t: {"konf": _b64(stufen[n][t]["konf"], np.float32),
                                   "pred": _b64(stufen[n][t]["pred"], np.uint8)} for t in ("va", "te")}
                           for n in stufen},
                "n_mag_test": {lv: _b64(np.minimum(v["n_mag_test"], 255), np.uint8) for lv, v in laden.items()},
                "T_exakt_gezaehlt": {f"{t}|{a}|{b}": v for (t, a, b), v in cache.items()},
                "format": "base64; konf float32, pred/y/n_mag uint8 (roh_lesen)"}
    z["sekunden_kosten"] = time.time() - t0
    return z


def _kosten_lauf(a):
    """Phase C je Seed (parallel): liest die duennen Winkel und ruft seed_zeile."""
    seed, stufen, vor, ordner, n_kalib, n_fest, cap = a
    d = np.load(_pfad(ordner, seed)[0])
    return seed_zeile(seed, stufen, vor, d["phi_va"], d["phi_te"], d["y_va"], d["y_te"], n_kalib, n_fest, cap)


# ------------------------------------------------------------------ 4. Statistik
def _p_ni(d, grenze=GRENZE) -> float:
    """Einseitiger gepaarter t-Test H0: mittel(d) <= -grenze (klein = nichtunterlegen)."""
    from scipy.stats import t as t_vert
    d = np.asarray(d, float)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    if sd == 0.0:
        return 0.0 if m > -grenze else 1.0
    return float(1.0 - t_vert.cdf((m + grenze) / (sd / math.sqrt(len(d))), len(d) - 1))


def _ki(d):
    d = np.asarray(d, float)
    if len(d) < 2 or float(d.std(ddof=1)) == 0.0:
        return float(d.mean()), float(d.mean())
    return _ki_einseitig(d)


def _gepaart(d) -> dict:
    d = np.asarray(d, float)
    return {"mittel": float(d.mean()), "p": wilcoxon_exakt(d), "n": int(len(d))}


def _tost(d, g=GRENZE) -> dict:
    d = np.asarray(d, float)
    pu, po = wilcoxon_einseitig(d + g, "groesser"), wilcoxon_einseitig(d - g, "kleiner")
    return {"p_unten": pu, "p_oben": po, "p_tost": max(pu, po)}


def _pareto(punkte):
    """punkte: {name: (T, acc)} -> Namen der nicht dominierten Punkte (weniger T, mehr acc)."""
    out = []
    for n, (t, a) in punkte.items():
        if not any((t2 <= t and a2 >= a) and (t2 < t or a2 > a) for m, (t2, a2) in punkte.items() if m != n):
            out.append(n)
    return sorted(out, key=lambda n: punkte[n][0])


def auswertung(zeilen) -> dict:
    zeilen = sorted(zeilen, key=lambda z: z["seed"])
    n = len(zeilen)
    arr = lambda f: np.array([f(z) for z in zeilen], dtype=float)
    ref = arr(lambda z: z["fest"]["k2_dicht"]["test"])
    a = {"n": n, "seeds": [z["seed"] for z in zeilen], "T_ref": T_K2_DICHT,
         "acc_ref": zusammenfassung(ref), "identitaet": zusammenfassung(arr(lambda z: z["identitaet"])),
         "fest": {f: {"acc": zusammenfassung(arr(lambda z: z["fest"][f]["test"])),
                      "T": float(arr(lambda z: z["fest"][f]["T"]).mean()),
                      "T_modell": float(arr(lambda z: z["fest"][f]["T_modell"]).mean())}
                  for f in zeilen[0]["fest"]},
         "kaskaden": {}}
    for v in zeilen[0]["kaskaden"]:
        acc = arr(lambda z: z["kaskaden"][v]["acc"])
        d = acc - ref
        lo, hi = _ki(d)
        T = arr(lambda z: z["kaskaden"][v]["T_hw"])
        a["kaskaden"][v] = {"acc": float(acc.mean()), "diff": float(d.mean()), "ki_unten": lo, "ki_oben": hi,
                            "p_ni": _p_ni(d), "p_ni_wilcoxon": wilcoxon_einseitig(d + GRENZE, "groesser"),
                            "T_hw": float(T.mean()), "anteil": float(T.mean()) / T_K2_DICHT,
                            "T_modell": float(arr(lambda z: z["kaskaden"][v]["T_modell"]).mean()),
                            "quote_quantum": float(arr(lambda z: z["kaskaden"][v]["quote_quantum"]).mean()),
                            "besuche": float(arr(lambda z: z["kaskaden"][v]["besuche"]).mean()),
                            "ausweich": int(sum(z["kaskaden"][v]["ausweich"] for z in zeilen)),
                            "tau_null": int(sum(z["kaskaden"][v]["tau"] == 0.0 for z in zeilen)),
                            "taus": [z["kaskaden"][v]["tau"] for z in zeilen],
                            "d_zufall": float((acc - arr(lambda z: z["kaskaden"][v]["acc_zufall"])).mean())}
    # PK
    a["PK"] = _gepaart(ref - arr(lambda z: z["identitaet"]))
    # G-1
    fam = [h for h in HAUPT if h in a["kaskaden"]]
    ph = holm({h: a["kaskaden"][h]["p_ni"] for h in fam})
    g1 = {}
    for h in fam:
        r = a["kaskaden"][h]
        g1[h] = {"anteil": r["anteil"], "diff": r["diff"], "ki_unten": r["ki_unten"], "ki_oben": r["ki_oben"],
                 "p_ni": r["p_ni"], "p_ni_holm": ph[h],
                 "erfuellt": bool(r["anteil"] <= ANTEIL_H1 and ph[h] < ALPHA),
                 "versagt": bool(r["anteil"] > ANTEIL_H0 or r["ki_oben"] < -GRENZE)}
    a["G1"] = {"familie": fam, "konfig": g1}
    # G-2
    g2 = {}
    for m in TOP_M:
        for ex in ("hybrid_dicht", "hybrid_duenn"):
            d = arr(lambda z: z["kaskaden"][f"kombi_m{m}"]["acc"] - z["kaskaden"][ex]["acc"])
            fest = arr(lambda z: z["fest"][f"k2_top{m}"]["test"] - z["fest"]["k2_dicht"]["test"])
            ww = d - fest
            g2[f"m{m}_gegen_{ex}"] = {**_gepaart(d), **_tost(d), "verlust_fest_k2": float(fest.mean()),
                                      "wechselwirkung": float(ww.mean()), "p_wechselwirkung": wilcoxon_exakt(ww)}
    a["G2"] = g2
    # G-3
    punkte = {f: (a["fest"][f]["T"], a["fest"][f]["acc"]["mittel"]) for f in a["fest"]}
    punkte.update({v: (a["kaskaden"][v]["T_hw"], a["kaskaden"][v]["acc"]) for v in a["kaskaden"]})
    instab = {}
    for f in a["fest"]:
        if f in ("k0_klassisch", "k2_dicht", "k2_duenn"):
            continue
        instab[f] = int(sum(z["fest"][f]["test"] < z["fest"]["k2_dicht"]["test"] - INSTAB for z in zeilen))
    for v in a["kaskaden"]:
        instab[v] = int(sum(z["kaskaden"][v]["acc"] < z["fest"]["k2_dicht"]["test"] - INSTAB for z in zeilen))
    med = float(np.median(ref))
    a["G3"] = {"punkte": {k: {"T": t, "anteil": t / T_K2_DICHT, "acc": ac} for k, (t, ac) in punkte.items()},
               "pareto": _pareto(punkte), "instabil": instab,
               "referenz_ausreisser": [z["seed"] for z in zeilen if z["fest"]["k2_dicht"]["test"] < med - INSTAB],
               "voll_unter_ref": int(sum(z["voll"]["test"] < z["fest"]["k2_dicht"]["test"] - INSTAB for z in zeilen)),
               "modell_gegen_exakt": {v: {"T_modell": a["kaskaden"][v]["T_modell"], "T_exakt": a["kaskaden"][v]["T_hw"]}
                                      for v in a["kaskaden"]},
               "additionen_stufe0": additionen_stufe0()}
    if "laden" in zeilen[0]:
        a["laden"] = {lv: {g: float(np.mean([z["laden"][lv][g] for z in zeilen]))
                           for g in ("t_rot", "T_fest_exakt", "T_fest_modell", "n_mag_mittel_test")}
                      for lv in zeilen[0]["laden"]}
    return a


def urteil(a: dict) -> dict:
    """Vorregistrierter Entscheidungsbaum; jeder Wert ist ein String."""
    u = {}
    pk = a["PK"]
    pk_ok = pk["mittel"] > 0 and pk["p"] < ALPHA
    u["PK"] = (f"H1: k2 exakt schlaegt Identitaet um {100 * pk['mittel']:+.2f} pp (p = {pk['p']:.4f})" if pk_ok else
               f"H0: k2 exakt nicht nachweisbar besser als Identitaet ({100 * pk['mittel']:+.2f} pp, "
               f"p = {pk['p']:.4f}) -- Positivkontrolle feuert nicht")
    g1 = a["G1"]["konfig"]

    def txt(h):
        r, k = g1[h], a["kaskaden"][h]
        return (f"{h}: {k['T_hw']:.0f} T = {100 * r['anteil']:.2f} % von {T_K2_DICHT} T, acc {100 * k['acc']:.2f} % "
                f"(ref {100 * a['acc_ref']['mittel']:.2f} %), d {100 * r['diff']:+.2f} pp, KI95 einseitig "
                f"[{100 * r['ki_unten']:+.2f}; {100 * r['ki_oben']:+.2f}], p_NI {r['p_ni']:.4f} (Holm "
                f"{r['p_ni_holm']:.4f}), Quantenanteil {100 * k['quote_quantum']:.1f} %, Ausweich {k['ausweich']}")

    alle = "; ".join(txt(h) for h in g1)
    if not pk_ok:
        u["G-1"] = "HU (Vorbedingung): Positivkontrolle feuert nicht -- " + alle
    elif any(r["erfuellt"] for r in g1.values()):
        b = next(h for h in HAUPT if h in g1 and g1[h]["erfuellt"])
        u["G-1"] = f"H1: {b} <= {100 * ANTEIL_H1:.0f} % der Hardware-T und nichtunterlegen (0,5 pp) -- " + alle
    elif g1 and all(r["versagt"] for r in g1.values()):
        u["G-1"] = (f"H0: jede Konfiguration > {100 * ANTEIL_H0:.0f} % oder nachweislich > 0,5 pp schlechter -- "
                    "Abbruchkriterium der Karte erfuellt -- " + alle)
    else:
        u["G-1"] = "HU: weder erfuellt noch jede Konfiguration versagt -- " + alle
    r = a["G2"]["m48_gegen_hybrid_dicht"]
    t2 = (f"d(kombi_m48 - hybrid_dicht) = {100 * r['mittel']:+.2f} pp (Wilcoxon p = {r['p']:.4f}, p_TOST "
          f"{r['p_tost']:.4f}); Verlust feste Stufe k2 durch Top-48 {100 * r['verlust_fest_k2']:+.2f} pp, "
          f"Wechselwirkung {100 * r['wechselwirkung']:+.2f} pp (p = {r['p_wechselwirkung']:.4f})")
    if r["mittel"] < 0 and r["p"] < ALPHA:
        u["G-2"] = "H1: genaehertes Laden kostet in der Kaskade nachweislich Genauigkeit -- " + t2
    elif r["mittel"] > 0 and r["p"] < ALPHA:
        u["G-2"] = "H0 (Gegenrichtung): Kaskade mit Top-48-Laden nachweislich besser -- " + t2
    elif r["p_tost"] < ALPHA:
        u["G-2"] = "H0: kein relevanter Verlust (TOST +-0,5 pp) -- " + t2
    else:
        u["G-2"] = "HU: Verlust weder belegt noch ausgeschlossen -- " + t2
    g3 = a["G3"]
    pk_ = lambda k: f"{k} {g3['punkte'][k]['T']:.0f} T ({100 * g3['punkte'][k]['anteil']:.1f} %) {100 * g3['punkte'][k]['acc']:.2f} %"
    u["G-3"] = ("beschreibend: Pareto " + ", ".join(pk_(k) for k in g3["pareto"])
                + "; Quantenanteil " + ", ".join(f"{v} {100 * a['kaskaden'][v]['quote_quantum']:.1f} %"
                                                 for v in a["kaskaden"])
                + "; instabil (< Ref - 1 pp) " + ", ".join(f"{k} {c}/{a['n']}" for k, c in g3["instabil"].items() if c)
                + f"; Referenz-Ausreisser {g3['referenz_ausreisser']}; tau*=0 in kombi_m48 "
                  f"{a['kaskaden'].get('kombi_m48', {}).get('tau_null', '-')}/{a['n']}")
    return u


# ------------------------------------------------------------------ 5. Lauf
def lauf(schnell=False, seeds=None, schritte=None, n_train=None, n_test=None, datensatz="mnist", **kw) -> dict:
    t0 = time.time()
    seeds = tuple(seeds) if seeds is not None else (ENTWURF_SEEDS if schnell else SEEDS)
    schritte = schritte or (150 if schnell else SCHRITTE)
    n_train = n_train or (800 if schnell else N_TRAIN_GESAMT)
    n_test = n_test or (400 if schnell else N_TEST)
    n_kalib, n_fest, cap = (10, 20, 30) if schnell else (N_KALIB, N_KOSTEN_FEST, N_EXAKT_MAX)
    lr, lr_quelle = _raten()
    ordner = tempfile.mkdtemp(prefix="gesamt_kombi_")
    try:
        vor = {v["seed"]: v for v in parallel(_vorbereitung, [(s, n_train, n_test, schritte, lr, ordner, datensatz) for s in seeds])}
        print(f"  Vorbereitung fertig ({time.time() - t0:.0f} s)", flush=True)
        st = parallel(_stufe_lauf, [(s, nm, schritte, lr, ordner) for s in seeds for nm in STUFENMODELLE])
        print(f"  Stufenmodelle fertig ({time.time() - t0:.0f} s)", flush=True)
        stufen = {}
        for e in st:
            stufen.setdefault(e["seed"], {})[e["name"]] = e
        zeilen = parallel(_kosten_lauf, [(s, stufen[s], vor[s], ordner, n_kalib, n_fest, cap) for s in seeds])
    finally:
        shutil.rmtree(ordner, ignore_errors=True)
    for z in sorted(zeilen, key=lambda z: z["seed"]):
        z["sekunden_vorbereitung"] = vor[z["seed"]]["sekunden"]
        z["rekonstruktion_fehler"] = vor[z["seed"]]["rekonstruktion_fehler"]
        z["sekunden_stufen"] = {n: stufen[z["seed"]][n]["sekunden"] for n in STUFENMODELLE}
        k = z["kaskaden"]["kombi_m48"]
        print(f"  Seed {z['seed']}: ident {z['identitaet']:.4f}  k2 dicht {z['fest']['k2_dicht']['test']:.4f}  "
              f"k2 top48 {z['fest']['k2_top48']['test']:.4f} | kombi_m48 tau*={k['tau']} acc {k['acc']:.4f} "
              f"T {k['T_hw']:.0f} ({100 * k['anteil']:.2f} %) Quant {100 * k['quote_quantum']:.1f} %", flush=True)
    ergebnis = {"abteilung": "clifford/hardware", "karte": "I-034", "quantenstufe": "Q3 (lokal simuliert)",
                "hypothesen": HYPOTHESEN, "datensatz": datensatz, "seeds": list(seeds), "entwurf": bool(schnell), "schritte": schritte,
                "n_train_gesamt": n_train, "n_val": n_train // 4, "n_test": n_test, "lr": lr, "lr_quelle": lr_quelle,
                "stufenmodelle": {k: list(v) for k, v in STUFENMODELLE.items()},
                "varianten": {k: [v[0], v[1]] for k, v in VARIANTEN.items()}, "taus": K.TAUS,
                "T_je_k": {str(k): v for k, v in K.T_JE_K.items()}, "T_laden_dicht": K.T_LADEN,
                "T_ref": T_K2_DICHT, "n_kalib": n_kalib, "n_kosten_fest": n_fest, "n_exakt_max": cap,
                "zeilen": sorted(zeilen, key=lambda z: z["seed"])}
    ergebnis["auswertung"] = auswertung(zeilen)
    ergebnis["urteil"] = urteil(ergebnis["auswertung"])
    ergebnis["sekunden"] = time.time() - t0
    return ergebnis


# ------------------------------------------------------------------ 6. Vortest
def _synth(rng, n=10, d48=0.0, a48=0.03, d32=0.0, a32=0.02, streu=0.002, d_ex=0.0, ident=0.90, ref=0.92,
           muster=False):
    """Synthetische Seed-Zeilen: d = Genauigkeitsabstand zur Referenz, a = T-Anteil. muster = True:
    deterministische Streuung (Mittel exakt d, sd exakt streu; exakte Kaskaden mit umgekehrter Reihenfolge)
    fuer Grenzfaelle, die nicht vom Zufall abhaengen duerfen."""
    zmu = np.linspace(-1, 1, n)
    zmu = (zmu - zmu.mean()) / zmu.std(ddof=1) if n > 1 else zmu * 0
    zeilen = []
    for s in range(n):
        r = ref + rng.normal(0, 0.003)
        fest = {f: {"val": r, "test": r + (0.0 if f.startswith("k2_d") else rng.normal(-0.002, 0.001)),
                    "T": 1000.0, "T_modell": 1000.0} for f in FESTE}
        fest["k2_dicht"]["T"] = fest["k2_dicht"]["T_modell"] = float(T_K2_DICHT)
        kas = {}
        for v in VARIANTEN:
            d, an = (d48, a48) if "m48" in v else (d32, a32) if "m32" in v else (d_ex, 0.12)
            acc = r + d + (streu * (zmu[n - 1 - s] if v.startswith("hybrid") else zmu[s]) if muster
                           else rng.normal(0, streu))
            kas[v] = {"acc": acc, "T_hw": an * T_K2_DICHT, "T_modell": an * T_K2_DICHT, "quote_quantum": 0.1,
                      "besuche": 0.12, "ausweich": False, "tau": 0.9, "acc_zufall": acc - 0.01}
        zeilen.append({"seed": s, "identitaet": ident + rng.normal(0, 0.002), "voll": {"test": r + 0.01},
                       "fest": fest, "kaskaden": kas})
    return zeilen


def vortest(verbose=True) -> bool:
    import json
    ok = True

    def pruefe(name, bed, wert=""):
        nonlocal ok
        ok &= bool(bed)
        if verbose or not bed:
            print(f"  {name}: {wert}  {'ok' if bed else 'FEHLER'}")

    rng = np.random.default_rng(34)
    # (a) Kostenmodell von Hand: Bild 0 bleibt in Stufe 0 -> 0 T; Bild 1 bis k8 -> 2 L(1) + 62 + 247;
    #     Bild 2 endet in k2 -> L(2) + 62
    L = np.array([1000.0, 1479.0, 1300.0])
    konf = np.array([[0.99, 0.10, 0.50], [0.10, 0.10, 0.95], [0.50, 0.50, 0.50]])
    korr = np.ones((3, 3), bool)
    ende = verlauf(konf, 0.8)
    _, _, T = _kosten_je_bild(ende, [0, 2, 8], L)
    soll = np.array([0.0, 2 * 1479 + 62 + 247, 1300 + 62])
    pruefe("(a) Kosten je Bild [Stufe 0, bis k8, bis k2]", np.allclose(T, soll), T.tolist())
    p = kaskade_bild(konf, korr, 0.8, [0, 2, 8], L)
    pruefe("(a) mittlere Kosten + Quantenanteil", abs(p["T"] - soll.mean()) < 1e-9 and abs(p["quote_quantum"] - 2 / 3) < 1e-12,
           p["T"])
    for m, lm in ((48, 1479.0), (32, 988.0)):
        _, _, T1 = _kosten_je_bild(np.array([0, 2]), [0, 2, 8], lm)
        pruefe(f"(a) m={m}: Stufe 0 -> 0 T, bis k8 -> 2 x {lm:.0f} + 62 + 247",
               T1[0] == 0 and T1[1] == 2 * lm + 62 + 247, T1.tolist())
    # (b) konstante Ladekosten == kaskade.kaskade mit kaskade_hybrid.t_hybrid (ungeaenderte Referenz)
    kz = rng.uniform(0.1, 1.0, (3, 500))
    kz[:, :20] = 1.0
    kk = rng.uniform(size=(3, 500)) < np.array([[0.8], [0.85], [0.9]])
    fehl = 0
    for tau in K.TAUS:
        for st in ([0, 2, 8], [0, 8]):
            q = K.kaskade(kz[:len(st)], kk[:len(st)], tau, t_hybrid(st, 7636.0), laden=0)
            r = kaskade_bild(kz[:len(st)], kk[:len(st)], tau, st, 7636.0)
            fehl += not (abs(q["T"] - r["T"]) < 1e-6 and q["acc"] == r["acc"] and q["anteil_ende"] == r["anteil_ende"]
                         and abs(q["acc_zufall"] - r["acc_zufall"]) < 1e-12
                         and abs(q["quote_weiter0"] - r["quote_quantum"]) < 1e-12)
    pruefe("(b) kaskade_bild == kaskade.kaskade bei konstanten Ladekosten (alle tau, 3- und 2-stufig)", fehl == 0, fehl)
    # (c) exakte Test-T: cap >= Weiterleitungen -> exakte Summe; cap klein -> Stichprobe erwartungstreu-nah
    Lb = rng.uniform(900, 1600, 500)
    e = verlauf(kz, 0.7)
    _, _, Tb = _kosten_je_bild(e, [0, 2, 8], Lb)
    r = t_test_exakt(e, [0, 2, 8], lambda b: Lb[b], 10_000, rng)
    pruefe("(c) exakte Zaehlung ohne Stichprobe == direkte Summe", abs(r["T"] - Tb.mean()) < 1e-9, (r["T"], Tb.mean()))
    ms = np.mean([t_test_exakt(e, [0, 2, 8], lambda b: Lb[b], 40, np.random.default_rng(i))["T"] for i in range(200)])
    pruefe("(c) Stichprobe (cap 40) im Mittel == Summe (< 1 %)", abs(ms - Tb.mean()) / Tb.mean() < 0.01, (ms, Tb.mean()))
    r0 = t_test_exakt(np.zeros(5, int), [0, 2, 8], lambda b: 1 / 0, 10, rng)
    pruefe("(c) niemand weitergeleitet -> 0 T, keine Zaehlung", r0["T"] == 0 and r0["gezaehlt"] == 0, r0["T"])
    # (d) echte Bausteine: MNIST-Bilder, Top-m-Zustaende, T-Formel, Kette seed_zeile
    _, _, xte, yte = bilder("mnist", ENTWURF_SEEDS[0], n_train=10, n_test=16, seite=16)
    psis = normiere(xte.numpy())
    phi = duenne_winkel(psis)
    pruefe("(d) top_255 == exakter Zustand", np.abs(arm_zustand(phi, "top_255") - psis).max() < 1e-10)
    nmag = magisch(phi).sum(1)
    pruefe("(d) #magisch(top_48) == min(48, #magisch)",
           np.array_equal(magisch(arm_winkel(phi, "top_48")).sum(1), np.minimum(48, nmag)))
    pruefe("(d) Normierung Top-32/48 exakt",
           max(np.abs(np.linalg.norm(arm_zustand(phi, a), axis=1) - 1).max() for a in ("top_32", "top_48")) < 1e-12)
    t48 = [t_formel(arm_winkel(phi[b:b + 1], "top_48")[0]) for b in range(4)]
    pruefe("(d) Lade-T(top_48) je Bild ~ 31 T je magischem Winkel",
           all(20 * min(48, nmag[b]) <= t48[b] <= 40 * min(48, nmag[b]) for b in range(4)), t48)
    stufen = {}
    yy = yte.numpy()
    for nme in STUFENMODELLE:
        e = {"frei": STUFENMODELLE[nme][0], "treue_test": 1.0, "norm_fehler": 0.0}
        for t in ("va", "te"):
            # Stufe 0 schwach, Quantenstufen perfekt -> die tau-Wahl MUSS weiterleiten
            richtig = rng.uniform(size=16) < 0.5 if nme == "k0_exakt" else np.ones(16, bool)
            e[t] = {"pred": np.where(richtig, yy, (yy + 1) % 10), "konf": np.where(richtig, 0.97, 0.6) + rng.uniform(0, 0.02, 16)}
        stufen[nme] = e
    z = seed_zeile(1, stufen, {"identitaet": 0.8, "voll": {"val": 0.9, "test": 0.9}}, phi, phi, yy, yy,
                   n_kalib=4, n_fest=4, cap=100)
    kd = z["kaskaden"]["hybrid_dicht"]
    pruefe("(d) Kette: dicht exakt == Modell (konstant 7.636 T)", abs(kd["T_hw"] - kd["T_modell"]) < 1e-9, kd["T_hw"])
    k48 = z["kaskaden"]["kombi_m48"]
    pruefe("(d) Kette: Top-48 leitet weiter, Kosten Modell ~ exakt (< 15 %), alle Weitergeleiteten gezaehlt",
           k48["T_hw"] > 0 and abs(k48["T_hw"] - k48["T_modell"]) <= 0.15 * k48["T_hw"]
           and k48["gezaehlt"] == k48["weiter"] > 0 and k48["acc"] == 1.0, (k48["T_hw"], k48["T_modell"], k48["weiter"]))
    pruefe("(d) Kette: k0 klassisch 0 T, k2 dicht 7.698 T",
           z["fest"]["k0_klassisch"]["T"] == 0 and z["fest"]["k2_dicht"]["T"] == T_K2_DICHT)
    ruck = roh_lesen(z["roh"]["stufen"]["k2_top48"]["te"]["konf"], np.float32)
    pruefe("(d) Rohdaten base64 rueckholbar", np.allclose(ruck, stufen["k2_top48"]["te"]["konf"], atol=1e-6))
    json.dumps(z)
    pruefe("(d) Seed-Zeile JSON-serialisierbar", True)

    # (e) Urteilszweige erzwingen
    def fall(name, erwartet, ziehungen=3, **kw):
        for i in range(ziehungen):
            u = urteil(auswertung(_synth(np.random.default_rng(200 + i), **kw)))
            if not all(isinstance(v, str) for v in u.values()):
                pruefe(f"(e) {name}: Strings", False)
                return u
            for k, marke in erwartet.items():
                if not u[k].startswith(marke):
                    pruefe(f"(e) {name}: {k} erwartet '{marke}', bekam '{u[k][:90]}'", False)
                    return u
        pruefe(f"(e) Fall {name} -> {erwartet}", True)
        return u

    fall("alles gut", {"PK": "H1", "G-1": "H1: kombi_m48", "G-2": "H0: kein relevanter", "G-3": "beschreibend"})
    fall("nur m32 billig genug", {"G-1": "H1: kombi_m32"}, a48=0.07)
    fall("beide zu teuer", {"G-1": "H0"}, a48=0.15, a32=0.12)
    fall("beide schlechter", {"G-1": "H0", "G-2": "H1"}, d48=-0.02, d32=-0.02)
    fall("gemischt: m48 zu teuer, m32 schlechter", {"G-1": "H0"}, a48=0.2, d32=-0.02)
    fall("zwischen 5 und 10 %", {"G-1": "HU:"}, a48=0.07, a32=0.08)
    fall("billig, auf der Grenze, verrauscht", {"G-1": "HU:", "G-2": "HU"}, streu=0.008, d48=-0.005, d32=-0.005,
         d_ex=-0.005, muster=True)
    fall("Positivkontrolle feuert nicht", {"PK": "H0", "G-1": "HU (Vorbedingung)"}, ident=0.95)
    fall("Top-48-Kaskade besser als exakt", {"G-2": "H0 (Gegenrichtung)"}, d48=0.01, d_ex=0.0)
    # Holm wirkt: m48 knapp nichtunterlegen (p_NI zwischen 0,025 und 0,05), m32 schlecht -> kein H1
    rng2 = np.random.default_rng(5)
    for _ in range(2000):
        zz = _synth(rng2, streu=0.004, d48=-0.0028, d32=-0.03)
        aa = auswertung(zz)
        p = aa["G1"]["konfig"]["kombi_m48"]["p_ni"]
        if 0.025 < p < 0.05 and aa["kaskaden"]["kombi_m48"]["anteil"] <= ANTEIL_H1:
            u = urteil(aa)
            pruefe(f"(e) Holm: m48 p_NI {p:.3f} allein < 0,05, Holm >= 0,05 -> nicht H1",
                   not u["G-1"].startswith("H1") and aa["G1"]["konfig"]["kombi_m48"]["p_ni_holm"] >= ALPHA, u["G-1"][:40])
            break
    else:
        pruefe("(e) Holm-Fall nicht erzeugt", False)
    # Nichtunterlegenheit <-> untere KI-Grenze (p_NI < 0,05 genau dann, wenn KI unten > -0,5 pp)
    wid = 0
    for i in range(200):
        d = np.random.default_rng(i).normal(-0.003, 0.004, 10)
        lo, _ = _ki_einseitig(d)
        wid += (_p_ni(d) < ALPHA) != (lo > -GRENZE)
    pruefe("(e) p_NI < 0,05 <=> untere einseitige KI-Grenze > -0,5 pp", wid == 0, wid)
    a = auswertung(_synth(rng))
    json.dumps(a)
    json.dumps(urteil(a))
    pruefe("(f) Auswertung und Urteil JSON-serialisierbar, Urteile Strings",
           all(isinstance(v, str) for v in urteil(a).values()))
    return bool(ok)


def drucke(e):
    a = e["auswertung"]
    print(f"\nI-034 Gesamt-Magie-Budget (MNIST 16x16, Seeds {e['seeds']}), Referenz k2 dicht {T_K2_DICHT} T, "
          f"acc {a['acc_ref']['mittel']:.4f}")
    print(" Arm            |   T je Bild | Anteil  | Genauigkeit")
    for k, p in sorted(a["G3"]["punkte"].items(), key=lambda kv: kv[1]["T"]):
        print(f" {k:14s} | {p['T']:11.0f} | {100 * p['anteil']:6.2f} % | {p['acc']:.4f}")
    for k, v in e["urteil"].items():
        print(f" {k}: {v}")


if __name__ == "__main__":
    import sys
    schnell = "--schnell" in sys.argv
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden or "--vortest" in sys.argv:
        sys.exit(0 if bestanden else 1)
    e = lauf(schnell=schnell)
    drucke(e)
    print(speichere("I-034" + ("-schnell" if schnell else ""), e))
