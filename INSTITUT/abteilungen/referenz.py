"""C-referenz -- faire Referenz fuer die Magie-Konzentration (Direktor, 24.09.).

ANLASS (FEHLERBUCH 24.09., C-fashion): Die Rundungsarme in clifford.py
(_gerundet) starten beim fertig trainierten vollen Modell, bekommen einen
frischen Kopf und dann eine ZWEITE Trainingsphase mit vollem Budget. Der Arm
„voll“ bekommt diese zweite Phase nicht. Auf Fashion schlug k=8 den vollen
Arm um +0,53 pp (p = 0,01, explorativ) -- das kann Regularisierung durch das
Runden sein ODER einfach das zusaetzliche Training. Dann waeren alle G(k)
und M50 gegen „voll“ aufgeblaeht.

FIX: Arm `nachtrainiert` = _gerundet(voll, k = n*L) -- alle Winkel frei,
frischer Kopf, exakt dieselbe zweite Phase (Rate „rundung“, gleiche
Schritte) wie die Rundungsarme. Er ist die faire Referenz: G*(k) und M50*
werden gegen ihn gemessen. Frische Seeds 30-39, Lernraten aus den getunten
C-mnist/C-fashion-Laeufen (dort auf eigenen Tuning-Seeds bestimmt).

Quantenstufe Q2 (exakte Simulation der Schaltung), sichert den Q3-Befund.
"""
from __future__ import annotations

import math
import time

import numpy as np

from abteilungen.clifford import (LAGEN, QUBITS_JE_DATENSATZ, _baue, _gerundet, daten)
from labor.gemeinsam import genauigkeit, neuestes, parallel, trainiere, wilcoxon_exakt, zusammenfassung

STUFEN = [0, 1, 2, 4, 8]
DATENSAETZE = ["mnist", "fashion_mnist"]
QUELLE = {"mnist": "C-mnist", "fashion_mnist": "C-fashion"}
GRENZE_NACH = 0.003      # 0,3 pp: ab hier verzerrt das Nachtraining G(k) sichtbar
                         # (ein Zehntel des Schaltungsgewinns von ~3,7 pp)
ALPHA = 0.05
LR_ERSATZ = {"voll": 0.08, "ohne_schaltung": 0.08, "rundung": 0.32}

HYPOTHESEN = {
    "R-1": "Nachtraining verzerrt die Referenz: d = acc(nachtrainiert) - acc(voll) je Datensatz. "
           "H1: |mittel(d)| >= 0,3 pp und exakter Wilcoxon zweiseitig p < alpha_Holm (Holm ueber die "
           "zwei Datensaetze). H0: 90-%-KI von d ganz in +-0,3 pp (TOST). Sonst HU.",
    "R-2": "Magie-Konzentration haelt gegen die faire Referenz: Magieanteil Gm*(k) = "
           "(acc_k - acc_k0) / (acc_nachtrainiert - acc_k0) (Verhaeltnis der Mittel). Vorbedingung: "
           "nachtrainiert schlaegt k0 (Wilcoxon p < 0,05). H1: untere Grenze des 95-%-Bootstrap-KI "
           "(Seeds) von Gm*(4) >= 0,5 auf BEIDEN Datensaetzen (4 von 32 Winkeln tragen mindestens die "
           "Haelfte des Magie-Nutzens). H0: obere Grenze < 0,5 auf mindestens einem. Sonst HU.",
    "R-3": "Rundung als Regularisierer (Nachpruefung des explorativen Fashion-Befunds k8 > voll): "
           "d = acc(k8) - acc(nachtrainiert) auf fashion_mnist. H1: mittel > 0 und einseitiger "
           "exakter Wilcoxon p < 0,05. H0: TOST +-0,3 pp oder mittel < 0 mit p < 0,05. Sonst HU. "
           "MNIST beschreibend.",
    "R-4": "beschreibend: G*(k) (Basis Identitaet), Gm*(k), M50*/M90* je Datensatz, neben den alten "
           "voll-basierten Werten.",
}


def _raten(ds):
    """Lernraten aus dem getunten Vorlauf (C-mnist / C-fashion)."""
    e = neuestes(QUELLE[ds])
    if not e:
        return dict(LR_ERSATZ)
    return {arm: e["lr"][arm]["lr"] for arm in LR_ERSATZ}


def _seed_lauf(a):
    s, ds, schritte, lr = a
    n = QUBITS_JE_DATENSATZ[ds]
    xtr, ytr, xte, yte = daten(ds, s)
    z = {"seed": s, "datensatz": ds}
    for arm in ("voll", "ohne_schaltung"):
        m, _ = _baue(arm, s, n)
        trainiere(m, xtr, ytr, schritte, lr[arm], seed=s)
        z[arm] = genauigkeit(m(xte), yte)
        if arm == "voll":
            voll = m
    for k in STUFEN + [LAGEN * n]:
        m = _gerundet(voll, k, s)
        trainiere(m, xtr, ytr, schritte, lr["rundung"], seed=s)
        z["nachtrainiert" if k == LAGEN * n else f"clifford_k{k}"] = genauigkeit(m(xte), yte)
    return z


# ---------------------------------------------------------------- Statistik
def _ki90(d):
    from scipy.stats import t as t_vert
    se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
    q = float(t_vert.ppf(0.95, len(d) - 1))
    return float(np.mean(d)) - q * se, float(np.mean(d)) + q * se


def _p_einseitig(d, rechts=True):
    p2 = wilcoxon_exakt(d)
    richtig = np.mean(d) > 0 if rechts else np.mean(d) < 0
    return p2 / 2 if richtig else 1 - p2 / 2


def _anteil(acc_k, basis, ref):
    nenner = float(np.mean(ref - basis))
    return float(np.mean(acc_k - basis)) / nenner if nenner > 0 else None


def _boot_ki(acc_k, basis, ref, ziehungen=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ref)
    werte = []
    for _ in range(ziehungen):
        i = rng.integers(0, n, n)
        g = _anteil(acc_k[i], basis[i], ref[i])
        if g is not None:
            werte.append(g)
    if len(werte) < ziehungen // 2:
        return None, None
    return float(np.quantile(werte, 0.025)), float(np.quantile(werte, 0.975))


def auswertung(zeilen) -> dict:
    aus = {}
    for ds in DATENSAETZE:
        zz = [z for z in zeilen if z["datensatz"] == ds]
        if not zz:
            continue
        a = lambda k: np.array([z[k] for z in zz])
        r = {k: zusammenfassung(a(k)) for k in zz[0] if k not in ("seed", "datensatz")}
        d = a("nachtrainiert") - a("voll")
        r["R1_mittel"], r["R1_p"], r["R1_ki90"] = float(d.mean()), wilcoxon_exakt(d), _ki90(d)
        dm = a("nachtrainiert") - a("clifford_k0")
        r["R2_vor_mittel"], r["R2_vor_p"] = float(dm.mean()), wilcoxon_exakt(dm)
        r["R2_Gm4"] = _anteil(a("clifford_k4"), a("clifford_k0"), a("nachtrainiert"))
        r["R2_Gm4_ki95"] = _boot_ki(a("clifford_k4"), a("clifford_k0"), a("nachtrainiert"))
        d3 = a("clifford_k8") - a("nachtrainiert")
        r["R3_mittel"], r["R3_p_rechts"], r["R3_p_links"] = (float(d3.mean()), _p_einseitig(d3, True),
                                                             _p_einseitig(d3, False))
        r["R3_ki90"] = _ki90(d3)
        for name, basis, ref in (("G_fair", "ohne_schaltung", "nachtrainiert"),
                                 ("Gm_fair", "clifford_k0", "nachtrainiert"),
                                 ("G_alt", "ohne_schaltung", "voll"),
                                 ("Gm_alt", "clifford_k0", "voll")):
            kurve = {k: _anteil(a(f"clifford_k{k}"), a(basis), a(ref)) for k in STUFEN}
            r[f"R4_{name}"] = kurve
            for st in (50, 90):
                r[f"R4_{name}_M{st}"] = next((k for k in STUFEN if kurve[k] is not None
                                              and kurve[k] >= st / 100), f"> {STUFEN[-1]}")
        r["n"] = len(zz)
        aus[ds] = r
    return aus


def urteil(a: dict) -> dict:
    u = {}
    # R-1 mit Holm ueber die zwei Datensaetze
    ps = sorted(((a[ds]["R1_p"], ds) for ds in DATENSAETZE if ds in a))
    sig = {}
    for i, (p, ds) in enumerate(ps):
        sig[ds] = p < ALPHA / (len(ps) - i) and all(sig.get(x, True) for _, x in ps[:i])
    teile, stufe = [], []
    for ds in DATENSAETZE:
        if ds not in a:
            continue
        r = a[ds]
        lo, hi = r["R1_ki90"]
        m = r["R1_mittel"]
        if abs(m) >= GRENZE_NACH and sig[ds]:
            stufe.append("H1")
            teile.append(f"{ds}: d = {m * 100:+.2f} pp (p = {r['R1_p']:.4f}) -- verzerrt")
        elif -GRENZE_NACH < lo and hi < GRENZE_NACH:
            stufe.append("H0")
            teile.append(f"{ds}: d = {m * 100:+.2f} pp, KI [{lo * 100:+.2f}, {hi * 100:+.2f}] in +-0,3")
        else:
            stufe.append("HU")
            teile.append(f"{ds}: d = {m * 100:+.2f} pp, KI [{lo * 100:+.2f}, {hi * 100:+.2f}], p = {r['R1_p']:.4f}")
    if "H1" in stufe:
        kopf = "H1: Nachtraining verzerrt die Referenz"
    elif stufe and all(s == "H0" for s in stufe):
        kopf = "H0: Nachtraining verzerrt nicht (TOST)"
    else:
        kopf = "HU: Verzerrung weder belegt noch ausgeschlossen"
    u["R-1"] = kopf + " -- " + "; ".join(teile)
    # R-2
    teile, ok, schlecht, vor = [], True, False, True
    for ds in DATENSAETZE:
        if ds not in a:
            ok = False
            continue
        r = a[ds]
        lo, hi = r["R2_Gm4_ki95"]
        if not (r["R2_vor_mittel"] > 0 and r["R2_vor_p"] < ALPHA) or lo is None:
            vor = False
        else:
            ok = ok and lo >= 0.5
            schlecht = schlecht or hi < 0.5
        teile.append(f"{ds}: Gm*(4) = {r['R2_Gm4']:.2f} KI [{lo if lo is None else round(lo, 2)}, "
                     f"{hi if hi is None else round(hi, 2)}], Magiegewinn {r['R2_vor_mittel'] * 100:+.2f} pp "
                     f"(p = {r['R2_vor_p']:.4f})" if r["R2_Gm4"] is not None else f"{ds}: kein Magiegewinn")
    if not vor:
        u["R-2"] = "HU (Vorbedingung): nachtrainiert schlaegt k0 nicht nachweisbar -- " + "; ".join(teile)
    elif ok:
        u["R-2"] = "H1: Magie-Konzentration haelt gegen faire Referenz -- " + "; ".join(teile)
    elif schlecht:
        u["R-2"] = "H0: 4 Winkel tragen nachweislich weniger als die Haelfte -- " + "; ".join(teile)
    else:
        u["R-2"] = "HU: KI ueberdeckt 0,5 -- " + "; ".join(teile)
    # R-3 (fashion)
    if "fashion_mnist" in a:
        r = a["fashion_mnist"]
        lo, hi = r["R3_ki90"]
        m = r["R3_mittel"]
        txt = f"k8 - nachtrainiert = {m * 100:+.2f} pp (p_rechts = {r['R3_p_rechts']:.4f}, KI90 [{lo * 100:+.2f}, {hi * 100:+.2f}])"
        if m > 0 and r["R3_p_rechts"] < ALPHA:
            u["R-3"] = "H1: Runden regularisiert (Fashion) -- " + txt
        elif (-GRENZE_NACH < lo and hi < GRENZE_NACH) or (m < 0 and r["R3_p_links"] < ALPHA):
            u["R-3"] = "H0: kein Regularisierungsgewinn durch Runden -- " + txt
        else:
            u["R-3"] = "HU: " + txt
    # R-4 beschreibend
    u["R-4"] = "; ".join(f"{ds}: M50/M90 fair (Basis Identitaet) {a[ds]['R4_G_fair_M50']}/{a[ds]['R4_G_fair_M90']}, "
                         f"Magieanteil fair {a[ds]['R4_Gm_fair_M50']}/{a[ds]['R4_Gm_fair_M90']}, "
                         f"alt {a[ds]['R4_G_alt_M50']}/{a[ds]['R4_G_alt_M90']}"
                         for ds in DATENSAETZE if ds in a)
    return u


def lauf(schnell=False, seeds=range(30, 40), schritte=1500):
    if schnell:
        seeds, schritte = range(30, 32), 60
    t0 = time.time()
    raten = {ds: _raten(ds) for ds in DATENSAETZE}
    aufg = [(s, ds, schritte, raten[ds]) for ds in DATENSAETZE for s in seeds]
    zeilen = parallel(_seed_lauf, aufg)
    a = auswertung(zeilen)
    return {"abteilung": "Clifford", "hypothesen": HYPOTHESEN, "raten": raten, "schritte": schritte,
            "zeilen": zeilen, "sekunden": time.time() - t0, "auswertung": a, "urteil": urteil(a)}


# ---------------------------------------------------------------- Vortest
def _synth(ds, nach_minus_voll, k8_minus_nach, magie4, rausch=0.001, seed=0):
    rng = np.random.default_rng(seed)
    zz = []
    for s in range(10):
        idn, k0, voll = 0.80, 0.815, 0.835
        nach = voll + nach_minus_voll
        z = {"seed": s, "datensatz": ds, "voll": voll, "ohne_schaltung": idn, "nachtrainiert": nach,
             "clifford_k0": k0, "clifford_k1": k0 + 0.1 * (nach - k0), "clifford_k2": k0 + 0.3 * (nach - k0),
             "clifford_k4": k0 + magie4 * (nach - k0), "clifford_k8": nach + k8_minus_nach}
        zz.append({k: (v + rng.normal(0, rausch) if isinstance(v, float) else v) for k, v in z.items()})
    return zz


def vortest() -> bool:
    ok = True

    def pruefe(name, zeilen, erwartet):
        nonlocal ok
        u = urteil(auswertung(zeilen))
        for frage, marke in erwartet.items():
            if not u[frage].startswith(marke):
                print(f"  VORTEST FEHLER {name}: {frage} erwartet {marke!r}, bekam {u[frage][:90]}")
                ok = False

    pruefe("verzerrt", _synth("mnist", 0.006, 0, 0.8) + _synth("fashion_mnist", 0.006, 0, 0.8, seed=1),
           {"R-1": "H1", "R-2": "H1", "R-3": "H0"})
    pruefe("unverzerrt", _synth("mnist", 0.0, 0, 0.8) + _synth("fashion_mnist", 0.0, 0, 0.8, seed=1),
           {"R-1": "H0", "R-2": "H1"})
    pruefe("unklar", _synth("mnist", 0.002, 0, 0.5, rausch=0.004) +
           _synth("fashion_mnist", 0.002, 0, 0.5, rausch=0.004, seed=1), {"R-1": "HU", "R-2": "HU"})
    pruefe("keine Konzentration", _synth("mnist", 0.0, 0, 0.2) + _synth("fashion_mnist", 0.0, 0, 0.8, seed=1),
           {"R-2": "H0"})
    pruefe("Regularisierer", _synth("mnist", 0.0, 0, 0.8) + _synth("fashion_mnist", 0.0, 0.006, 0.8, seed=1),
           {"R-3": "H1"})
    pruefe("Regularisierer unklar", _synth("mnist", 0.0, 0, 0.8) +
           _synth("fashion_mnist", 0.0, 0.003, 0.8, rausch=0.006, seed=1), {"R-3": "HU"})
    z = _synth("mnist", 0.0, 0, 0.8) + _synth("fashion_mnist", 0.0, 0, 0.8, seed=1)
    for r in z:
        r["nachtrainiert"] = r["clifford_k0"] - 0.001
    pruefe("Vorbedingung", z, {"R-2": "HU (Vorbedingung)"})
    # Struktur: k = n*L laesst alle Winkel frei und startet exakt bei voll, Kopf frisch
    import torch
    for ds in DATENSAETZE:
        n = QUBITS_JE_DATENSATZ[ds]
        voll, _ = _baue("voll", 5, n)
        with torch.no_grad():
            voll.theta.add_(torch.randn_like(voll.theta) * 0.3)
        m = _gerundet(voll, LAGEN * n, 5)
        gleich = torch.allclose(m.theta, voll.theta) and bool(m.frei.bool().all())
        kopf_neu = not torch.allclose(m.kopf.weight, voll.kopf.weight)
        if not (gleich and kopf_neu):
            print(f"  VORTEST FEHLER Struktur {ds}: Winkel gleich {gleich}, Kopf frisch {kopf_neu}")
            ok = False
    return ok
