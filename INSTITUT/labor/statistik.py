"""Zentrale Statistik des Instituts (Plan R2.1) -- kanonische Fassung.

Bisher hat fast jedes Modul in abteilungen/ seinen eigenen exakten Wilcoxon, TOST, Holm und
Bootstrap/t-KI mitgebracht (Kopien mit kleinen Unterschieden: Raenge 1..n vs. Mittelraenge,
Aufzaehlung vs. Zaehl-Rekursion, p/2-Abkuerzung usw.). Diese Datei ist die EINE Version fuer NEUE
Module. Bestehende Module bleiben unveraendert (ihre Ergebnisse sind festgeschrieben);
tests_labor/test_statistik_aequivalenz.py belegt, dass die zentralen Funktionen auf deren Eingaben
dieselben p-Werte liefern, und dokumentiert die wenigen Abweichungen.

Protokollregeln (INSTITUT/UEBERGABE.md §3), auf die sich die Funktionen beziehen:
  * "genug Seeds (exakter Wilcoxon, kleinster p = 2/2^n)"   -> wilcoxon_exakt, kleinster_p
  * "nicht signifikant != gleich (TOST)"                     -> tost_t, tost_wilcoxon
  * "Holm nur ueber die angemeldete Familie"                 -> holm_adjustiert, holm_signifikant
  * "Grenzen relativ zum Effekt setzen"                      -> (Aufrufer: tost_*(d, grenze))
  * Q-011: asymptotische Naeherung war falsch                -> alles exakt bzw. Permutation, nie asymptotisch
  * "Module liefern Urteile als Strings"                     -> urteil_h1_h0_hu, urteil_etikett

Die Differenzen d sind immer "Arm minus Referenz", gepaart ueber Seeds (gleiche Reihenfolge).
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

ALPHA = 0.05


# ----------------------------------------------------------------- Wilcoxon
def kleinster_p(n: int, zweiseitig: bool = True) -> float:
    """Kleinster mit n Seeds erreichbarer exakter Wilcoxon-p (Protokollregel "genug Seeds"):
    zweiseitig 2/2^n, einseitig 1/2^n. Vor dem Lauf pruefen, dass er unter Alpha/Familiengroesse liegt."""
    return (2.0 if zweiseitig else 1.0) / 2.0 ** n


def _ganze_raenge(d: np.ndarray, raenge: str) -> np.ndarray:
    """Ganzzahlige Raenge der |d|. 'mittel': Durchschnittsraenge bei Bindungen (verdoppelt, damit ganzzahlig);
    'ordinal': Raenge 1..n ohne Mittelung (wie labor.gemeinsam.wilcoxon_exakt; Bindungen werden nach
    Reihenfolge aufgeloest). Ohne Bindungen sind beide gleich (bis auf den Faktor 2)."""
    if raenge == "mittel":
        return np.round(2 * stats.rankdata(np.abs(d))).astype(int)
    if raenge == "ordinal":
        return (np.argsort(np.argsort(np.abs(d))) + 1).astype(int)
    raise ValueError("raenge muss 'mittel' oder 'ordinal' sein")


def _rangsummen_verteilung(r: np.ndarray) -> np.ndarray:
    """zaehl[s] = Anzahl der Vorzeichenbelegungen mit Rangsumme s (Zaehl-Rekursion, exakt, kein 2^n)."""
    summe = int(r.sum())
    zaehl = np.zeros(summe + 1)
    zaehl[0] = 1.0
    for ri in r:
        zaehl[ri:] = zaehl[ri:] + zaehl[:summe + 1 - ri].copy()
    return zaehl


def wilcoxon_exakt(diff, alternative: str = "zweiseitig", raenge: str = "mittel") -> float:
    """Exakter Wilcoxon-Vorzeichen-Rang-Test fuer gepaarte Differenzen (Nullen entfallen).

    alternative: 'zweiseitig' | 'groesser' (H1: Median(d) > 0, P(W+ >= w)) | 'kleiner' (H1: Median(d) < 0,
    P(W+ <= w)). raenge: 'mittel' (Standard, korrekt bei Bindungen) oder 'ordinal' (reproduziert die aelteren
    Module, die Raenge 1..n ohne Mittelung nutzen). Exakt per Zaehl-Rekursion, auch fuer n >= 20.
    Protokollregel: exakter Wilcoxon, kleinster p = 2/2^n (siehe kleinster_p); nie asymptotisch (Q-011)."""
    d = np.asarray([v for v in np.asarray(diff, dtype=float) if v != 0.0], dtype=float)
    n = len(d)
    if n == 0:
        return 1.0
    r = _ganze_raenge(d, raenge)
    zaehl = _rangsummen_verteilung(r)
    gesamt = zaehl.sum()
    w = int(r[d > 0].sum())
    if alternative == "groesser":
        return float(zaehl[w:].sum() / gesamt)
    if alternative == "kleiner":
        return float(zaehl[:w + 1].sum() / gesamt)
    if alternative == "zweiseitig":
        s = np.arange(len(zaehl))
        abw = np.abs(2 * s - int(r.sum()))
        return float(zaehl[abw >= abs(2 * w - int(r.sum()))].sum() / gesamt)
    raise ValueError("alternative muss 'zweiseitig', 'groesser' oder 'kleiner' sein")


def wilcoxon_einseitig(diff, richtung: str = "groesser", raenge: str = "mittel") -> float:
    """Kurzform fuer die einseitige Variante (richtung 'groesser' | 'kleiner')."""
    return wilcoxon_exakt(diff, alternative=richtung, raenge=raenge)


# ----------------------------------------------------------------- TOST
def tost_t(diff, grenze: float) -> float:
    """Gepaarter t-TOST: H0 |mu| >= grenze, p = max der beiden einseitigen t-p. p < alpha <=> das
    (1-2*alpha)-KI (90 % bei alpha = 5 %) liegt in (-grenze, +grenze). Protokollregel: "nicht signifikant
    != gleich"; Grenze vorab und relativ zum Effekt setzen. sd = 0: 0 bzw. 1; n < 2: 1."""
    d = np.asarray(diff, dtype=float)
    if len(d) < 2:
        return 1.0
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    if se == 0.0:
        return 0.0 if abs(m) < grenze else 1.0
    p_unten = float(stats.t.sf((m + grenze) / se, len(d) - 1))     # gegen mu <= -grenze
    p_oben = float(stats.t.cdf((m - grenze) / se, len(d) - 1))     # gegen mu >= +grenze
    return max(p_unten, p_oben)


def tost_wilcoxon(diff, grenze: float, raenge: str = "ordinal") -> dict:
    """Wilcoxon-TOST (verteilungsfrei): zwei einseitige exakte Tests der um -/+grenze verschobenen
    Differenzen. Standard 'ordinal' wie in lade_magie/gesamt_kombi. Rueckgabe p_unten, p_oben, p_tost
    (= max), p_schlechter (H1: d < -grenze, Nichtunterlegenheit verletzt)."""
    d = np.asarray(diff, dtype=float)
    pu = wilcoxon_exakt(d + grenze, "groesser", raenge)
    po = wilcoxon_exakt(d - grenze, "kleiner", raenge)
    return {"p_unten": pu, "p_oben": po, "p_tost": max(pu, po),
            "p_schlechter": wilcoxon_exakt(d + grenze, "kleiner", raenge)}


# ----------------------------------------------------------------- Holm
def holm_adjustiert(p: dict) -> dict:
    """Holm-adjustierte p-Werte {name: p} -> {name: p_holm}, monoton, bei 1 gedeckelt. Protokollregel:
    NUR ueber die angemeldete Familie (also genau die vorregistrierten Hypothesen uebergeben)."""
    namen = sorted(p, key=p.get)
    aus, lauf = {}, 0.0
    for i, k in enumerate(namen):
        lauf = max(lauf, min(1.0, (len(namen) - i) * p[k]))
        aus[k] = lauf
    return aus


def holm_signifikant(p: dict, alpha: float = ALPHA) -> dict:
    """Holm-Stufenverfahren als Ja/Nein {name: bool} (gleiches Ergebnis wie holm_adjustiert < alpha, nur
    mit '<=' an der Schwelle wie in skala/ternaer)."""
    sig = {k: False for k in p}
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        if v <= alpha / (len(p) - i):
            sig[k] = True
        else:
            break
    return sig


# ----------------------------------------------------------------- Konfidenzintervalle
def t_ki(diff, niveau: float = 0.90) -> tuple:
    """Zweiseitiges t-KI des Mittels (Standard 90 % = zwei einseitige 95-%-Tests, passt zu TOST bei alpha
    5 %). n < 2 -> (-inf, +inf); sd = 0 -> (Mittel, Mittel)."""
    d = np.asarray(diff, dtype=float)
    if len(d) < 2:
        return -math.inf, math.inf
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(stats.t.ppf(0.5 + niveau / 2, len(d) - 1))
    m = float(d.mean())
    return m - q * se, m + q * se


def bootstrap_ki(daten: Sequence, statistik: Callable, ziehungen: int = 4000, seed: int = 0,
                 niveau: float = 0.95, min_gueltig: Optional[float] = 0.5) -> tuple:
    """Perzentil-Bootstrap ueber gemeinsam gezogene Seed-Indizes (Paarung bleibt erhalten).
    daten: Sequenz gleich langer Arrays; statistik(*gezogene_arrays) -> float oder None (ungueltig).
    Gueltig muessen >= min_gueltig*ziehungen Ziehungen sein, sonst (None, None) -- entartete Nenner nicht
    wegmitteln. Reproduziert referenz._boot_ki bitgleich (rng.integers je Ziehung)."""
    arrays = [np.asarray(a) for a in daten]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    werte = []
    for _ in range(ziehungen):
        i = rng.integers(0, n, n)
        g = statistik(*[a[i] for a in arrays])
        if g is not None and np.isfinite(g):
            werte.append(g)
    if min_gueltig is not None and len(werte) < ziehungen * min_gueltig:
        return None, None
    a = (1 - niveau) / 2
    return float(np.quantile(werte, a)), float(np.quantile(werte, 1 - a))


# ----------------------------------------------------------------- Permutationstests
def permutationstest(x, y, statistik: Callable, n_perm: int = 10_000, seed: int = 0,
                     alternative: str = "zweiseitig") -> dict:
    """Allgemeiner Permutationstest auf Zuordnung x<->y: y wird zufaellig permutiert, p = (1 + #extrem)/(1 + N)
    (nie 0). statistik(x, y) -> float. Monte Carlo, daher seed Pflicht-Argument im Protokoll."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    beob = statistik(x, y)
    rng = np.random.default_rng(seed)
    perm = np.array([statistik(x, y[rng.permutation(len(y))]) for _ in range(n_perm)])
    if alternative == "zweiseitig":
        k = int(np.sum(np.abs(perm) >= abs(beob) - 1e-12))
    elif alternative == "groesser":
        k = int(np.sum(perm >= beob - 1e-12))
    elif alternative == "kleiner":
        k = int(np.sum(perm <= beob + 1e-12))
    else:
        raise ValueError("alternative ungueltig")
    return {"statistik": float(beob), "p": (1 + k) / (1 + n_perm), "methode": f"Monte Carlo ({n_perm} Perm.)"}


def vorzeichen_permutation(diff, n_perm: int = 20_000, seed: int = 0) -> float:
    """Permutationstest (Vorzeichen-Umkehr) fuer das Mittel gepaarter Differenzen, zweiseitig; fuer n <= 15
    exakt (alle 2^n), sonst Monte Carlo (1+k)/(1+N). Verteilungsfreie Gegenprobe zum Wilcoxon."""
    d = np.asarray(diff, float)
    n = len(d)
    beob = abs(d.mean())
    if n <= 15:
        vz = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
        m = np.abs(((2 * vz - 1) * d).mean(1))
        return float(np.mean(m >= beob - 1e-12))
    rng = np.random.default_rng(seed)
    vz = rng.choice([-1.0, 1.0], size=(n_perm, n))
    k = int(np.sum(np.abs((vz * d).mean(1)) >= beob - 1e-12))
    return (1 + k) / (1 + n_perm)


def _rho_raenge(rx, ry) -> float:
    rx, ry = rx - rx.mean(), ry - ry.mean()
    nenner = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / nenner) if nenner > 0 else 0.0


def spearman_permutation(x, y, gruppen=None, n_perm: int = 20_000, seed: int = 0,
                         alternative: str = "zweiseitig", exakt_bis: int = 10) -> dict:
    """Spearman-rho (Durchschnittsraenge) mit PERMUTATIONS-p, nie asymptotisch (Q-011).
    Ohne Gruppen und n <= exakt_bis: exakt ueber alle n! Permutationen; sonst Monte Carlo
    p = (1 + #extrem)/(1 + N); mit Gruppen wird y nur innerhalb der Gruppen permutiert (haelt den
    Versatz zwischen Datensaetzen fest). alternative 'zweiseitig' | 'kleiner' (rho < 0) | 'groesser'.
    Konstanter Vektor -> rho 0, p 1, methode 'konstant'. Zweiseitig & Standardargumente reproduzieren
    sre.spearman_perm bitgleich."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = _rho_raenge(rx, ry)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    nenner = math.sqrt(float(rxc @ rxc) * float(ryc @ ryc))
    if nenner == 0:
        return {"rho": 0.0, "p": 1.0, "methode": "konstant"}
    n = len(x)

    def zaehle(r):
        if alternative == "zweiseitig":
            return int(np.sum(np.abs(r) >= abs(rho) - 1e-12))
        if alternative == "kleiner":
            return int(np.sum(r <= rho + 1e-12))
        if alternative == "groesser":
            return int(np.sum(r >= rho - 1e-12))
        raise ValueError("alternative ungueltig")

    if gruppen is None and n <= exakt_bis:
        treffer, gesamt = 0, math.factorial(n)
        perms = np.array(list(permutations(range(n))), dtype=np.int8)
        for i in range(0, len(perms), 400_000):
            treffer += zaehle((ryc[perms[i:i + 400_000]] @ rxc) / nenner)
        return {"rho": rho, "p": treffer / gesamt, "methode": f"exakt ({gesamt} Perm.)"}
    rng = np.random.default_rng(seed)
    gruppen = np.zeros(n, int) if gruppen is None else np.asarray(gruppen)
    perm = np.tile(np.arange(n), (n_perm, 1))
    for g in np.unique(gruppen):
        spalten = np.where(gruppen == g)[0]
        perm[:, spalten] = spalten[rng.permuted(np.tile(np.arange(len(spalten)), (n_perm, 1)), axis=1)]
    treffer = zaehle((ryc[perm] @ rxc) / nenner)
    return {"rho": rho, "p": (1 + treffer) / (1 + n_perm),
            "methode": f"Monte Carlo ({n_perm} Perm."
                       + (", geschichtet)" if len(np.unique(gruppen)) > 1 else ")")}


# ----------------------------------------------------------------- Zusammenfassung / Urteile
def zusammenfassung(werte) -> dict:
    """Mittel, Stichproben-SD (ddof=1), min, max, n -- wie labor.gemeinsam.zusammenfassung."""
    a = np.asarray(werte, dtype=float)
    return {"mittel": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def gepaart(diff, grenze: Optional[float] = None) -> dict:
    """Standardauswertung gepaarter Differenzen in einem Aufruf: Mittel, SD, exakter Wilcoxon (zweiseitig und
    beide einseitig), 90-%-t-KI, optional t-TOST-p gegen +-grenze."""
    d = np.asarray(diff, dtype=float)
    ki = t_ki(d)
    aus = {"n": int(len(d)), "mittel": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "p_zweiseitig": wilcoxon_exakt(d), "p_groesser": wilcoxon_exakt(d, "groesser"),
           "p_kleiner": wilcoxon_exakt(d, "kleiner"), "ki90_unten": ki[0], "ki90_oben": ki[1],
           "positiv": int((d > 0).sum())}
    if grenze is not None:
        aus["p_tost"] = tost_t(d, grenze)
    return aus


def urteil_h1_h0_hu(p_h1: float, p_gegen: float, p_tost: Optional[float], alpha: float = ALPHA,
                    familie: int = 1, kontrollen_ok: bool = True, stabil: bool = True,
                    zahlen: str = "") -> str:
    """Urteils-String H1 / H0 / HU nach Protokoll (Module liefern Urteile als Strings).
      HU  : Vorbedingung verletzt (Kontrolle nicht bestanden oder Instabilitaet) -- NIE ein stilles H0.
      H1  : p_h1 (bereits die vorab genannte, einseitige Hypothese) < alpha/familie (Bonferroni-Schranke
            als einfache Holm-Obergrenze; bei echter Holm-Familie vorher holm_adjustiert nutzen und
            familie=1 setzen).
      H0  : Gleichwertigkeit belegt (p_tost < alpha) ODER signifikante Gegenrichtung (p_gegen < alpha/familie).
      HU  : sonst ("nicht signifikant != gleich"). Der Zusatz in Klammern traegt die Zahlen."""
    z = f" ({zahlen})" if zahlen else ""
    if not stabil:
        return f"HU (Instabilitaet/Kollaps){z}"
    if not kontrollen_ok:
        return f"HU (Kontrolle nicht bestanden){z}"
    schwelle = alpha / max(1, familie)
    if p_h1 < schwelle:
        return f"H1{z}"
    if p_tost is not None and p_tost < alpha:
        return f"H0 (gleichwertig){z}"
    if p_gegen < schwelle:
        return f"H0 (Gegenrichtung){z}"
    return f"HU{z}"


def urteil_etikett(urteil: str) -> str:
    """Erstes Wort eines Urteils-Strings ('H1', 'H0', 'HU', 'GLEICHWERTIG' ...), zum Auswerten/Zaehlen."""
    return urteil.strip().split(" ")[0].split("(")[0] if urteil else ""
