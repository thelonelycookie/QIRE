"""Gleichheitstest R2.1: die zentrale labor.statistik liefert auf den Eingaben bestehender Module dieselben
p-Werte (<= 1e-12) wie deren eigene Kopien -- synthetisch und auf Rohdaten aus ergebnisse/*.json.
Bestehende Module werden nur importiert, nie veraendert. Abweichungen stehen in ABWEICHUNGEN (unten)
und werden getestet, nicht weggefixt."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from labor import statistik as S

ERG = Path(__file__).resolve().parent.parent / "ergebnisse"
TOL = 1e-12


# ------------------------------------------------------------- Eingaben
def _synthetisch():
    """Differenzen mit/ohne Bindungen, Nullen, verschiedene n (bis 20)."""
    aus = []
    for n in (1, 2, 3, 5, 8, 10, 12, 20):
        rng = np.random.default_rng(100 + n)
        aus.append(rng.normal(0.004, 0.01, n))                         # ohne Bindungen
        aus.append(np.round(rng.normal(0.0, 0.01, n) * 200) / 200)      # Bindungen + Nullen
        aus.append(-np.abs(rng.normal(0.01, 0.005, n)))                 # alles negativ
        aus.append(np.abs(rng.normal(0.01, 0.005, n)))                  # alles positiv
    aus.append(np.zeros(6))
    return aus


def _json(muster):
    pfade = sorted(ERG.glob(muster))
    if not pfade:
        pytest.skip(f"keine Rohdaten {muster}")
    return json.loads(pfade[-1].read_text())


def _paare_aus_zeilen(zeilen, felder=None, maxpaare=40):
    """Gepaarte Differenzen zwischen skalaren Feldern je Seed."""
    felder = felder or [k for k, v in zeilen[0].items() if isinstance(v, float)]
    paare = list(itertools.combinations(felder, 2))[:maxpaare]
    return [np.array([z[a] - z[b] for z in zeilen]) for a, b in paare]


def _realdaten():
    aus = []
    z = _json("I-039__*.json")["zeilen"]                                # acc je Arm (dict)
    arme = list(z[0]["acc"])
    for a, b in itertools.combinations(arme, 2):
        aus.append(np.array([r["acc"][a] - r["acc"][b] for r in z]))
    aus += _paare_aus_zeilen(_json("C-clifford__*.json")["zeilen"])
    aus += _paare_aus_zeilen(_json("I-001__*.json")["zeilen"])
    aus += _paare_aus_zeilen(_json("T-ternaer__*.json")["zeilen"])
    # I-044-v3: (arm, seed) -> acc, gepaart ueber Seed innerhalb desselben Teils
    zz = _json("I-044-v3__*.json")["zeilen"]
    for teil in sorted({r["teil"] for r in zz}):
        je = {}
        for r in zz:
            if r["teil"] == teil:
                je.setdefault(r["arm"], {})[r["seed"]] = r["acc"]
        for a, b in itertools.combinations(sorted(je), 2):
            seeds = sorted(set(je[a]) & set(je[b]))
            if len(seeds) >= 3:
                aus.append(np.array([je[a][s] - je[b][s] for s in seeds]))
    return aus


@pytest.fixture(scope="module")
def alle_d():
    return _synthetisch() + _realdaten()


# ------------------------------------------------------------- 1. mnist100 (Mittelraenge, einseitig, Holm)
def test_mnist100_wilcoxon_und_holm(alle_d):
    from abteilungen import mnist100 as m
    for d in alle_d:
        assert abs(m.wilcoxon_einseitig(d) - S.wilcoxon_exakt(d, "groesser", "mittel")) <= TOL
    p = {f"h{i}": S.wilcoxon_exakt(d, "groesser") for i, d in enumerate(alle_d[:12])}
    assert m.holm(p) == S.holm_adjustiert(p)


# ------------------------------------------------------------- 2. holevo_wirkungsgrad (Mittelraenge, Tupel, Spearman)
def test_holevo_wilcoxon_tupel_und_spearman(alle_d):
    from abteilungen import holevo_wirkungsgrad as h, skala as sk
    for d in alle_d:
        if len([v for v in d if v != 0]) > 14:       # dort 2^n-Aufzaehlung: nur kleine n
            continue
        assert abs(h.wilcoxon_einseitig(d) - S.wilcoxon_exakt(d, "groesser")) <= TOL
        pr, pl = sk.wilcoxon_einseitig(d)             # skala: Tupel (p_rechts, p_links), Mittelraenge
        assert abs(pr - S.wilcoxon_exakt(d, "groesser")) <= TOL
        assert abs(pl - S.wilcoxon_exakt(d, "kleiner")) <= TOL
    rng = np.random.default_rng(3)
    for n in (4, 6, 7):
        x = rng.normal(size=n)
        y = -x + rng.normal(scale=0.5, size=n)
        y[0] = y[1]                                    # Bindung
        r0, p = h.spearman_perm(x, y)
        e = S.spearman_permutation(x, y, alternative="kleiner")
        assert abs(r0 - e["rho"]) <= TOL and abs(p - e["p"]) <= TOL
    assert h.spearman_perm([1, 1, 1], [1, 2, 3]) == (None, None)


# ------------------------------------------------------------- 3. datenmenge + lade_magie + gesamt_kombi (ordinale Raenge, TOST-Wilcoxon)
def test_datenmenge_ordinal_holm_und_tost(alle_d):
    from abteilungen import datenmenge as dm
    from abteilungen import lade_magie as lm
    from abteilungen import gesamt_kombi as gk
    for d in alle_d:
        for r in ("groesser", "kleiner"):
            assert abs(dm.wilcoxon_einseitig(d, r) - S.wilcoxon_exakt(d, r, "ordinal")) <= TOL
        t_lm, t_gk = lm._tost(d, 0.005), gk._tost(d, 0.005)
        z = S.tost_wilcoxon(d, 0.005)
        for k in ("p_unten", "p_oben", "p_tost"):
            assert abs(t_lm[k] - z[k]) <= TOL and abs(t_gk[k] - z[k]) <= TOL
        assert abs(t_lm["p_schlechter"] - z["p_schlechter"]) <= TOL
    p = {f"h{i}": S.wilcoxon_exakt(d, "groesser", "ordinal") for i, d in enumerate(alle_d[:9])}
    assert dm.holm(p) == S.holm_adjustiert(p)


# ------------------------------------------------------------- 4. gemeinsam.wilcoxon_exakt + walsh_kontrolle (zweiseitig)
def test_zweiseitig_gemeinsam_und_walsh(alle_d):
    from labor import gemeinsam as g
    from abteilungen import walsh_kontrolle as w, maske_block as mb
    for d in alle_d:
        n = len([v for v in d if v != 0])
        if n <= 14:                                    # gemeinsam: 2^n
            assert abs(g.wilcoxon_exakt(d) - S.wilcoxon_exakt(d, "zweiseitig", "ordinal")) <= TOL
        assert abs(w.wilcoxon_schnell(d) - S.wilcoxon_exakt(d, "zweiseitig", "ordinal")) <= TOL
        assert abs(mb.wilcoxon_einseitig(d) - S.wilcoxon_exakt(d, "groesser", "ordinal")) <= TOL


# ------------------------------------------------------------- 5. clifford_suche_v2: TOST (t) und t-KI (referenz, kaskade, kaskade_hybrid)
def test_tost_t_und_ki(alle_d):
    from abteilungen import clifford_suche_v2 as c2
    from abteilungen import referenz as ref
    from abteilungen import kaskade as ka
    from abteilungen import kaskade_hybrid as kh
    for d in alle_d:
        for delta in (0.005, 0.02):
            assert abs(c2.tost_p(d, delta) - S.tost_t(d, delta)) <= TOL
        if len(d) < 2 or d.std(ddof=1) == 0:
            continue
        lo, hi = S.t_ki(d)
        for fn in (ref._ki90, ka._t_ki, kh._ki_einseitig):
            a, b = fn(d)
            assert abs(a - lo) <= TOL and abs(b - hi) <= TOL
    # ABWEICHUNG A4: kaskade._t_ki gibt fuer n<2 (-inf, inf) -- zentral ebenso
    assert ka._t_ki([0.1]) == S.t_ki([0.1]) == (-math.inf, math.inf)


# ------------------------------------------------------------- 6. referenz: Bootstrap-KI bitgleich
def test_bootstrap_ki_referenz():
    from abteilungen import referenz as ref
    z = _json("C-referenz__*.json")["zeilen"]
    arme = [k for k, v in z[0].items() if isinstance(v, float)]
    voll, ohne, k2 = (np.array([r[a] for r in z]) for a in ("voll", "ohne_schaltung", "clifford_k2"))
    assert "clifford_k2" in arme
    alt = ref._boot_ki(k2, ohne, voll, ziehungen=500, seed=0)
    neu = S.bootstrap_ki([k2, ohne, voll], ref._anteil, ziehungen=500, seed=0, min_gueltig=0.5)
    assert alt == neu                                  # bitgleich
    # entartet: Nenner <= 0 -> (None, None)
    assert S.bootstrap_ki([k2, ohne, ohne], ref._anteil, ziehungen=50) == (None, None)


# ------------------------------------------------------------- 7. sre: Spearman-Permutation (exakt + Monte Carlo + geschichtet), Bootstrap-Diff nicht zentralisiert
def test_spearman_permutation_sre():
    from abteilungen import sre
    rng = np.random.default_rng(7)
    for n in (5, 8, 10):
        x = rng.normal(size=n)
        y = x + rng.normal(size=n)
        a, b = sre.spearman_perm(x, y), S.spearman_permutation(x, y)
        assert a["methode"] == b["methode"] and abs(a["rho"] - b["rho"]) <= TOL and abs(a["p"] - b["p"]) <= TOL
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    g = np.repeat([0, 1], 20)
    for kw in ({}, {"gruppen": g, "seed": 2}):
        a = sre.spearman_perm(x, y, n_perm=3000, **kw)
        b = S.spearman_permutation(x, y, n_perm=3000, **kw)
        assert a == b                                  # bitgleich (gleiche RNG-Folge)
    assert sre.spearman_perm([1, 1, 1, 1], [1, 2, 3, 4]) == S.spearman_permutation([1, 1, 1, 1], [1, 2, 3, 4])


# ------------------------------------------------------------- 8. Holm-Varianten: skala/ternaer/magie_schwierigkeit (Ja/Nein)
def test_holm_signifikant_und_geruest():
    from abteilungen import skala, ternaer, magie_schwierigkeit as ms, geruest, clifford_suche as cs
    rng = np.random.default_rng(5)
    for _ in range(200):
        p = {f"h{i}": float(v) for i, v in enumerate(rng.choice([0.001, 0.004, 0.01, 0.02, 0.05, 0.2, 0.9], 5))}
        ref = S.holm_signifikant(p)
        assert skala._holm(p) == ternaer._holm(p) == ms._holm(p) == ref
        assert geruest.holm(p) == cs.holm(p) == S.holm_adjustiert(p)
        # Konsistenz: Holm-Ja/Nein ('<=' an der Schwelle) == adjustiert <= alpha
        adj = S.holm_adjustiert(p)
        assert {k: v <= 0.05 + 1e-15 for k, v in adj.items()} == ref


# ------------------------------------------------------------- 9. ABWEICHUNGEN (dokumentiert, nicht weggefixt)
def test_abweichung_struktur_aequivalenz_halbierung():
    """A1: struktur_aequivalenz.wilcoxon_einseitig nutzt p2/2 bzw. 1 - p2/2 statt der exakten Schwanzwahrscheinlichkeit.
    In Datenrichtung identisch (Symmetrie der Nullverteilung); in Gegenrichtung ist 1 - p2/2 um P(W = w) KLEINER
    als der exakte Wert P(W >= w); bei w == Mitte gibt das Modul 1.0 statt P(W >= Mitte) > 0,5."""
    from abteilungen import struktur_aequivalenz as sa
    rng = np.random.default_rng(1)
    d = rng.normal(0.01, 0.01, 12)
    assert abs(sa.wilcoxon_einseitig(d, True) - S.wilcoxon_exakt(d, "groesser", "ordinal")) <= TOL
    dn = -np.abs(d)
    alt, neu = sa.wilcoxon_einseitig(dn, True), S.wilcoxon_exakt(dn, "groesser", "ordinal")
    assert alt <= neu + TOL and neu - alt < 1e-3          # praktisch irrelevant, aber nicht gleich
    sym = np.array([1.0, -2.0, -3.0, 4.0])               # w = Mitte
    assert sa.wilcoxon_einseitig(sym) == 1.0
    assert S.wilcoxon_exakt(sym, "groesser", "ordinal") < 1.0


def test_abweichung_ordinal_vs_mittelraenge_bei_bindungen():
    """A2: Module mit Raengen 1..n (datenmenge, lade_magie, gesamt_kombi, walsh, gemeinsam) und Module mit
    Mittelraengen (mnist100, holevo, struktur_aequivalenz ueber Mnist-Pfad) unterscheiden sich NUR bei
    Bindungen in |d|. Zentral: raenge='mittel' (korrekt) ist Standard, 'ordinal' reproduziert die alten Werte."""
    d = np.array([0.01, -0.01, 0.02, 0.01, 0.03, -0.02, 0.01])
    mittel, ordinal = S.wilcoxon_exakt(d), S.wilcoxon_exakt(d, raenge="ordinal")
    assert mittel != ordinal
    from abteilungen import mnist100 as m, datenmenge as dm
    assert m.wilcoxon_einseitig(d) == pytest.approx(S.wilcoxon_exakt(d, "groesser", "mittel"), abs=TOL)
    assert dm.wilcoxon_einseitig(d) == pytest.approx(S.wilcoxon_exakt(d, "groesser", "ordinal"), abs=TOL)


def test_abweichung_gesamt_kombi_ki_entartet():
    """A3: gesamt_kombi._ki liefert bei sd = 0 (Mittel, Mittel) -- zentral t_ki ebenso; bei n < 2 liefert
    gesamt_kombi einen Fehler/Mittel-Intervall, zentral (-inf, inf). Hier nur sd = 0 geprueft."""
    from abteilungen import gesamt_kombi as gk
    d = np.array([0.01, 0.01, 0.01])
    assert gk._ki(d) == S.t_ki(d) == (0.01, 0.01) or gk._ki(d) == pytest.approx(S.t_ki(d))


# ------------------------------------------------------------- Eigenschaften der zentralen Funktionen
def test_eigenschaften():
    assert S.kleinster_p(10) == 2 / 1024
    d = np.arange(1, 11, dtype=float)                    # alle positiv, n = 10
    assert S.wilcoxon_exakt(d) == pytest.approx(S.kleinster_p(10))
    assert S.wilcoxon_exakt(d, "groesser") == pytest.approx(1 / 1024)
    assert S.wilcoxon_exakt([0.0, 0.0]) == 1.0
    # Aufzaehlung (scipy) als unabhaengige Pruefung
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1, 9)
    assert S.wilcoxon_exakt(x) == pytest.approx(wilcoxon(x, method="exact").pvalue, abs=1e-12)
    assert S.wilcoxon_exakt(x, "groesser") == pytest.approx(wilcoxon(x, method="exact", alternative="greater").pvalue, abs=1e-12)
    # TOST: gleich -> klein, 1 pp Differenz bei Grenze 0,5 pp -> gross
    assert S.tost_t(np.random.default_rng(0).normal(0, 0.001, 20), 0.005) < 0.01
    assert S.tost_t(np.random.default_rng(0).normal(0.01, 0.001, 20), 0.005) > 0.5
    # Permutations-Vorzeichentest ~ Wilcoxon-Groessenordnung, exakt <= 15
    assert 0 < S.vorzeichen_permutation(x) <= 1
    r = S.permutationstest(x, x + rng.normal(size=9), lambda a, b: float(np.corrcoef(a, b)[0, 1]), n_perm=500)
    assert 0 < r["p"] <= 1


def test_urteile():
    assert S.urteil_h1_h0_hu(0.001, 0.9, 0.8).startswith("H1")
    assert S.urteil_h1_h0_hu(0.4, 0.9, 0.01).startswith("H0")
    assert S.urteil_h1_h0_hu(0.4, 0.001, 0.9).startswith("H0")
    assert S.urteil_h1_h0_hu(0.4, 0.4, 0.4).startswith("HU")
    assert S.urteil_h1_h0_hu(0.0001, 0.9, None, stabil=False).startswith("HU")        # Instabilitaet schlaegt H1
    assert S.urteil_h1_h0_hu(0.0001, 0.9, None, kontrollen_ok=False).startswith("HU")
    assert S.urteil_etikett("H0 (gleichwertig) x") == "H0"
