"""I-015-hybrid -- Hybrid-Magie-Kaskade: Clifford-Stufe KLASSISCH, nur unsichere
Bilder auf den Quantenrechner (Direktor, 24.09., vorregistriert vor jeder Zahl
dieser Seeds).

ANLASS: I-015 (Seeds 90-94) zeigte K-1 H1 (Kaskade 0->2->8 erreicht k=2 mit
28 % der Schaltungs-T), aber mit Ladekosten 114 % -- jede Stufe laedt das Bild
neu (7.636 T). Nachtraeglich gerechnet (LABORBUCH 24.09., 23:55): Die Stufe
k=0 ist eine reine Clifford-Schaltung auf einem KLASSISCHEN Vektor; ihre
Born-Wahrscheinlichkeiten sind Vorzeichen-Permutationen und
Hadamard-Additionen -- klassisch exakt, ohne Multiplikation und ohne Laden.
Rechnet man Stufe 0 klassisch und nur die weitergeleiteten Bilder auf dem
Quantenrechner, kostete die Kaskade 14,5 % der Hardware-T von k=2 bei gleicher
Genauigkeit. Das war post hoc, 5 Seeds -> hier vorregistriert, 10 frische
Seeds 100-109, tau auf Validierung nach den HYBRID-Kosten gewaehlt.

Kostenmodell (Q3): klassische Stufe 0: 0 T. Jede Quantenstufe k >= 1:
T_LADEN + T_JE_K[k] (neu laden, da ein Quantenzustand nach der Messung weg ist).
Referenz: feste Stufe k=2 auf Hardware = 7.636 + 62 = 7.698 T je Bild.
Ehrlich: Die klassische Stufe kostet Additionen (beschreibend H-4 gezaehlt);
der Befund ist ein Hybrid-Befund, keine reine Quantenersparnis.
"""
from __future__ import annotations

import math
import time

import numpy as np

from abteilungen import kaskade as K
from labor.gemeinsam import wilcoxon_exakt

SEEDS = range(100, 110)
ANTEIL_H1 = 0.25          # <= 25 % der Hardware-T von k=2
ANTEIL_H0 = 0.50          # > 50 % -> keine lohnende Ersparnis
GRENZE_ACC = 0.005        # Nichtunterlegenheit 0,5 pp (einseitiges 95-%-KI)
ALPHA = 0.05
STUFEN = [0, 2, 8]

HYPOTHESEN = {
    "H-1": "Hybrid-Kaskade 0(klassisch)->2->8, tau auf Validierung nach Hybrid-Kosten gewaehlt "
           "(kleinste Kosten mit val-acc >= val-acc_k2 - 0,2 pp): H1, wenn die mittlere Hardware-T je "
           "Bild <= 25 % von k=2 auf Hardware (7.698 T) UND die untere Grenze des einseitigen 95-%-KI "
           "(t, Seeds) von acc_hybrid - acc_k2 >= -0,5 pp. H0, wenn die T > 50 % ODER die obere "
           "KI-Grenze < -0,5 pp. Sonst HU.",
    "H-2": "Kontrolle: konfidenzbasierte Weiterleitung schlaegt zufaellige Weiterleitung mit gleicher "
           "Quote bei tau* (gepaart, exakter Wilcoxon zweiseitig, p < 0,05) -> H1; sonst H0/HU.",
    "H-3": "beschreibend: zweistufig 0(klassisch)->2 und 0(klassisch)->8, Weiterleitungsquoten, "
           "Kosten mit Stufe 0 auf Hardware (wie I-015 K-3).",
    "H-4": "beschreibend: klassische Additionen der Stufe 0 je Bild.",
}


def t_hybrid(stufen_k=STUFEN, laden=K.T_LADEN):
    """Kosten je Stufe: 0 klassisch -> 0; sonst Laden + T der Schaltung."""
    return np.array([0.0 if k == 0 else laden + K.T_JE_K[k] for k in stufen_k])


def additionen_stufe0(n=8, lagen=K.LAGEN if hasattr(K, "LAGEN") else 4):
    """Clifford-Ry je Qubit: Vorzeichen-Permutation (0 Add.) oder Hadamard-Paar
    (2^n Additionen je Qubit und Lage, obere Schranke); CNOT = Permutation."""
    return int(lagen * n * 2 ** n)


def _aus_roh(z, teil, stufen_k):
    r = z["roh"][teil]
    y = np.array(r["y"])
    konf = np.stack([np.array(r[f"k{k}"]["konf"]) for k in stufen_k])
    korrekt = np.stack([np.array(r[f"k{k}"]["pred"]) == y for k in stufen_k])
    return konf, korrekt


def seed_hybrid(z) -> dict:
    ziel = z["fest"]["k2"]["val"] - K.TOL_ACC
    h = {"seed": z["seed"], "acc_k2": z["fest"]["k2"]["test"], "acc_k0": z["fest"]["k0"]["test"],
         "acc_k8": z["fest"]["k8"]["test"], "T_k2_hw": K.T_LADEN + K.T_JE_K[2]}
    for name, st in (("dreistufig", [0, 2, 8]), ("zwei_0_2", [0, 2]), ("zwei_0_8", [0, 8])):
        kv = K.kurve(*_aus_roh(z, "val", st), t_hybrid(st), K.TAUS)
        kt = K.kurve(*_aus_roh(z, "test", st), t_hybrid(st), K.TAUS)
        tau, aus = K.waehle_tau(kv, ziel, "T")      # "T" = Hybrid-Kosten (laden steckt in t_hybrid)
        p = K._bei(kt, tau)
        h[name] = {"tau": tau, "ausweich": aus, "acc": p["acc"], "T_hw": p["T"],
                   "anteil": p["T"] / h["T_k2_hw"], "quote_quantum": p["quote_weiter0"],
                   "anteil_ende": p["anteil_ende"], "acc_zufall": p["acc_zufall"]}
    return h


def _ki_einseitig(d):
    from scipy.stats import t as t_vert
    d = np.asarray(d, float)
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return float(d.mean()) - q * se, float(d.mean()) + q * se


def auswertung(hz) -> dict:
    a = {"n": len(hz)}
    for name in ("dreistufig", "zwei_0_2", "zwei_0_8"):
        acc = np.array([h[name]["acc"] for h in hz])
        k2 = np.array([h["acc_k2"] for h in hz])
        d = acc - k2
        lo, hi = _ki_einseitig(d)
        zuf = np.array([h[name]["acc_zufall"] for h in hz])
        a[name] = {"acc": float(acc.mean()), "acc_k2": float(k2.mean()), "diff": float(d.mean()),
                   "ki_unten": lo, "ki_oben": hi,
                   "T_hw": float(np.mean([h[name]["T_hw"] for h in hz])),
                   "anteil": float(np.mean([h[name]["T_hw"] for h in hz])) / hz[0]["T_k2_hw"],
                   "quote_quantum": float(np.mean([h[name]["quote_quantum"] for h in hz])),
                   "d_zufall": float((acc - zuf).mean()), "p_zufall": wilcoxon_exakt(acc - zuf),
                   "ausweich": int(sum(h[name]["ausweich"] for h in hz))}
    a["additionen_stufe0"] = additionen_stufe0()
    return a


def urteil(a) -> dict:
    u = {}
    r = a["dreistufig"]
    txt = (f"Hardware-T {r['T_hw']:.0f} = {r['anteil'] * 100:.1f} % von k2 ({r['acc'] * 100:.2f} % vs "
           f"{r['acc_k2'] * 100:.2f} %, diff {r['diff'] * 100:+.2f} pp, KI [{r['ki_unten'] * 100:+.2f}, "
           f"{r['ki_oben'] * 100:+.2f}]), Quantenanteil {r['quote_quantum'] * 100:.1f} % der Bilder, Ausweich {r['ausweich']}")
    if r["anteil"] <= ANTEIL_H1 and r["ki_unten"] >= -GRENZE_ACC:
        u["H-1"] = "H1: Hybrid-Kaskade spart >= 75 % Hardware-T ohne Genauigkeitsverlust -- " + txt
    elif r["anteil"] > ANTEIL_H0 or r["ki_oben"] < -GRENZE_ACC:
        u["H-1"] = "H0: keine lohnende Hybrid-Ersparnis -- " + txt
    else:
        u["H-1"] = "HU: " + txt
    t = f"d = {r['d_zufall'] * 100:+.2f} pp, p = {r['p_zufall']:.4f}"
    if r["d_zufall"] > 0 and r["p_zufall"] < ALPHA:
        u["H-2"] = "H1: Konfidenz schlaegt Zufall -- " + t
    elif r["d_zufall"] <= 0 and r["p_zufall"] < ALPHA:
        u["H-2"] = "H0: Zufall besser -- " + t
    else:
        u["H-2"] = "HU: " + t
    u["H-3"] = "beschreibend: " + "; ".join(
        f"{n}: T {a[n]['anteil'] * 100:.1f} %, acc {a[n]['acc'] * 100:.2f} % (k2 {a[n]['acc_k2'] * 100:.2f} %), "
        f"Quantenanteil {a[n]['quote_quantum'] * 100:.1f} %" for n in ("zwei_0_2", "zwei_0_8"))
    u["H-4"] = f"beschreibend: Stufe 0 klassisch <= {a['additionen_stufe0']} Additionen je Bild, 0 Multiplikationen"
    return u


def lauf(schnell=False, seeds=SEEDS):
    t0 = time.time()
    e = K.lauf(schnell=schnell, seeds=seeds)
    hz = [seed_hybrid(z) for z in e["zeilen"]]
    a = auswertung(hz) if len(hz) > 1 else {"n": len(hz), "roh": hz}
    return {"abteilung": "Clifford", "hypothesen": HYPOTHESEN, "seeds": list(seeds) if not schnell else [90],
            "kaskade": {k: v for k, v in e.items() if k != "zeilen"}, "hybrid_zeilen": hz,
            "zeilen_kurz": [{k: v for k, v in z.items() if k != "roh"} for z in e["zeilen"]],
            "sekunden": time.time() - t0, "auswertung": a,
            "urteil": urteil(a) if len(hz) > 1 else {"H-1": "HU: Rauchtest (1 Seed)"}}


# ------------------------------------------------------------------ Vortest
def _synth(anteil, diff, streu, d_zuf, p_klein=True, n=10):
    rng = np.random.default_rng(0)
    hz = []
    for s in range(n):
        h = {"seed": s, "acc_k2": 0.92, "acc_k0": 0.90, "acc_k8": 0.93, "T_k2_hw": 7698}
        for name in ("dreistufig", "zwei_0_2", "zwei_0_8"):
            acc = 0.92 + diff + rng.normal(0, streu)
            h[name] = {"tau": 0.6, "ausweich": False, "acc": acc, "T_hw": anteil * 7698, "anteil": anteil,
                       "quote_quantum": 0.1, "anteil_ende": [0.9, 0.05, 0.05],
                       "acc_zufall": acc - d_zuf - (abs(rng.normal(0, 0.0005)) if p_klein else rng.normal(0, 0.01))}
        hz.append(h)
    return hz


def vortest() -> bool:
    ok = True
    faelle = [("spart", _synth(0.15, 0.0, 0.002, 0.01), {"H-1": "H1", "H-2": "H1"}),
              ("zu teuer", _synth(0.6, 0.0, 0.002, 0.01), {"H-1": "H0"}),
              ("schlechter", _synth(0.15, -0.02, 0.002, 0.01), {"H-1": "H0"}),
              ("unklar", _synth(0.35, 0.0, 0.002, 0.0, p_klein=False), {"H-1": "HU", "H-2": "HU"})]
    for name, hz, erw in faelle:
        u = urteil(auswertung(hz))
        for f, m in erw.items():
            if not u[f].startswith(m):
                print(f"  VORTEST FEHLER {name}: {f} erwartet {m}, bekam {u[f][:80]}")
                ok = False
    # Kostenmodell: Bild bleibt in Stufe 0 -> 0 T; geht bis k8 -> 2 Ladungen + 62 + 247
    konf = np.array([[0.99, 0.1], [0.1, 0.1], [0.5, 0.5]])
    korrekt = np.ones((3, 2), bool)
    p = K.kaskade(konf, korrekt, 0.8, t_hybrid(), laden=0)
    soll = (0 + (2 * K.T_LADEN + K.T_JE_K[2] + K.T_JE_K[8])) / 2
    if abs(p["T"] - soll) > 1e-9:
        print("  VORTEST FEHLER Kostenmodell", p["T"], soll)
        ok = False
    return ok
