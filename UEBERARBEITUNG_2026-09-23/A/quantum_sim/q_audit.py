from __future__ import annotations

"""Prueft die Q-Zahlen des Tagebuchs gegen die gespeicherten Rohdaten.

WARUM ES DAS BRAUCHT
`qire_ai.audit` prueft eine fest verdrahtete Liste von B-Zahlen. Keine einzige
Q-Zahl steht darin -- sein "Alle geprueften Kernzahlen stimmen" sagt also nichts
ueber die Aussagen, die im August dazugekommen sind.

Und der konkrete Anlass: In der Nacht zum 6. August habe ich in einem
Tagebucheintrag "+0,0044 (p=0,49)" neben eine Genauigkeit von 0,9339 gestellt.
Die erste Zahl stammte aus dem abgebrochenen Lauf mit engem Lernratengitter,
die zweite aus dem zweiten Lauf -- **zwei Laeufe in einer Aussage vermischt.**
Genau so ist die 819x-Aussage entstanden, die am 3. August zurueckgezogen
werden musste (A-002 und B-002 zusammengesetzt).

Ein Mensch sieht so etwas nicht, wenn beide Zahlen plausibel aussehen. Ein
Werkzeug schon: Es rechnet jede Behauptung aus **einer** Datei neu nach.

Run: python -m quantum_sim.q_audit
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from qire_ai.result_store import load

TOL = 0.0002          # Toleranz fuer Genauigkeiten (Rundung im Tagebuch)


def _vals(records, arm: str, dataset: str, field: str = "balanced_accuracy") -> np.ndarray:
    rows = sorted((r for r in records if r["arm"] == arm and r["dataset"] == dataset),
                  key=lambda r: r["seed"])
    return np.array([r[field] for r in rows])


def _param(records, arm: str, field: str = "middle_parameters") -> int:
    return int(next(r[field] for r in records if r["arm"] == arm))


def check_q003() -> List[Tuple[str, bool, str]]:
    """Alle Q-Zahlen aus dem Eintrag '5. August, nachts' und der Kurzfassung.

    WICHTIG: Es wird bewusst **eine einzige Datei** geladen (die neueste, also
    der Lauf mit erweitertem Gitter). Jede Behauptung muss aus derselben Quelle
    folgen -- sonst wiederholt sich der Vermischungsfehler.
    """
    payload = load("Q-003-replication")
    rec = payload["records"]
    out: List[Tuple[str, bool, str]] = []

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        ok = abs(claimed - actual) <= tol
        out.append((label, ok, f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    def cmp_int(label: str, claimed: int, actual: int):
        out.append((label, claimed == actual,
                    f"Tagebuch {claimed:,} | Rohdaten {actual:,}"))

    hyb_m = _vals(rec, "hybrid", "mnist")
    deep_m = _vals(rec, "classical_norm_deep", "mnist")
    free_m = _vals(rec, "classical_quad_free", "mnist")
    one_m = _vals(rec, "hybrid_1layer", "mnist")
    noent_m = _vals(rec, "hybrid_no_entangle", "mnist")

    hyb_f = _vals(rec, "hybrid", "fashion_mnist")
    deep_f = _vals(rec, "classical_norm_deep", "fashion_mnist")
    free_f = _vals(rec, "classical_quad_free", "fashion_mnist")

    # --- Einzelwerte der Tabelle ---
    cmp("hybrid, mnist, balanced", 0.9339, float(hyb_m.mean()))
    cmp("hybrid_1layer, mnist", 0.9337, float(one_m.mean()))
    cmp("classical_quad_free, mnist", 0.9213, float(free_m.mean()))
    cmp("classical_norm_deep, mnist", 0.8407, float(deep_m.mean()))

    # --- die Kernaussage ---
    cmp("PRIMAER mnist: hybrid - norm_deep", 0.0932,
        float(hyb_m.mean() - deep_m.mean()))
    cmp("PRIMAER fashion: hybrid - norm_deep", 0.0599,
        float(hyb_f.mean() - deep_f.mean()))

    # --- die beiden Gleichwertigkeiten ---
    cmp("Tiefe: hybrid - hybrid_1layer, mnist", 0.0002,
        float(hyb_m.mean() - one_m.mean()))
    cmp("Verschraenkung: hybrid - no_entangle, mnist", -0.0015,
        float(hyb_m.mean() - noent_m.mean()))

    # --- der nachgerechnete Vergleich (nicht vorregistriert) ---
    cmp("hybrid - quad_free, mnist", 0.0126, float(hyb_m.mean() - free_m.mean()))
    cmp("hybrid - quad_free, fashion", 0.0051, float(hyb_f.mean() - free_f.mean()))

    # --- die Zerlegung: wie viel traegt die Quadratur allein? ---
    for ds, hyb, deep, free, claimed in (
            ("mnist", hyb_m, deep_m, free_m, 86.5),
            ("fashion_mnist", hyb_f, deep_f, free_f, 91.5)):
        total = float(hyb.mean() - deep.mean())
        share = float(free.mean() - deep.mean()) / total * 100
        ok = abs(share - claimed) <= 0.6
        out.append((f"Anteil der Quadratur, {ds}", ok,
                    f"Tagebuch {claimed:.1f} % | Rohdaten {share:.1f} %"))

    # --- Parameterzahlen ---
    cmp_int("Mitte: hybrid", 80, _param(rec, "hybrid"))
    cmp_int("Mitte: hybrid_1layer", 16, _param(rec, "hybrid_1layer"))
    cmp_int("Mitte: classical_quad_free", 2048, _param(rec, "classical_quad_free"))
    cmp_int("Mitte: classical_norm_deep", 67848, _param(rec, "classical_norm_deep"))
    cmp_int("gesamt: hybrid", 201130, _param(rec, "hybrid", "total_parameters"))
    cmp_int("gesamt: classical_norm_deep", 268898,
            _param(rec, "classical_norm_deep", "total_parameters"))

    # --- Seedzahl ---
    out.append(("15 Seeds je Zelle", len(hyb_m) == 15,
                f"Tagebuch 15 | Rohdaten {len(hyb_m)}"))
    return out


def check_q015() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '6. August -- Q-015: die Schaltung schlaegt die
    volle Drehung'. Nachgetragen am 10. August: Q-015 ist der einzige Lauf mit
    Urteil 'gilt', und keine seiner Zahlen stand hier -- genau die Luecke, vor
    der HANDOFF_PROJEKT_A.md Abschnitt 7 warnt."""
    rec = load("Q-015")["records"]
    out: List[Tuple[str, bool, str]] = []

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        out.append((label, abs(claimed - actual) <= tol,
                    f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    hyb = _vals(rec, "hybrid", "mnist")
    glob = _vals(rec, "global_orth", "mnist")
    block = _vals(rec, "block_orth", "mnist")
    free = _vals(rec, "quad_free", "mnist")

    cmp("hybrid, mnist, balanced", 0.9327, float(hyb.mean()))
    cmp("global_orth, mnist", 0.8653, float(glob.mean()))
    cmp("block_orth, mnist", 0.8994, float(block.mean()))
    cmp("quad_free, mnist", 0.9115, float(free.mean()))

    cmp("Streuung hybrid", 0.0079, float(hyb.std(ddof=1)))
    cmp("Streuung global_orth", 0.0346, float(glob.std(ddof=1)))

    cmp("PRIMAER: hybrid - global_orth", 0.0674, float(hyb.mean() - glob.mean()))
    cmp("P2: global_orth - block_orth", -0.0341, float(glob.mean() - block.mean()))

    out.append(("PRIMAER: 15 von 15 Seeds", int((hyb > glob).sum()) == 15,
                f"Tagebuch 15 | Rohdaten {int((hyb > glob).sum())}"))
    out.append(("P2: 0 von 15 Seeds", int((glob > block).sum()) == 0,
                f"Tagebuch 0 | Rohdaten {int((glob > block).sum())}"))

    fg = _param(rec, "hybrid")
    gg = _param(rec, "global_orth")
    out.append(("Freiheitsgrade: hybrid", fg == 80, f"Tagebuch 80 | Rohdaten {fg:,}"))
    out.append(("Freiheitsgrade: global_orth", gg == 32640,
                f"Tagebuch 32,640 | Rohdaten {gg:,}"))
    out.append(("Verhaeltnis 408-fach", round(gg / max(fg, 1)) == 408,
                f"Tagebuch 408 | Rohdaten {gg / max(fg, 1):.0f}"))
    return out


def check_q014() -> List[Tuple[str, bool, str]]:
    """Zahlen aus den Eintraegen '10. August -- Q-014'.

    ACHTUNG BEIM ZITIEREN: Der Lauf ist nach eigener Vorregistrierung NICHT
    VERWERTBAR -- die Auflösungspruefung ist durchgefallen. Dieses Werkzeug
    prueft nur, ob die im Tagebuch stehenden Zahlen aus den Rohdaten folgen,
    nicht ob sie zitierfaehig sind. Sie sind es nicht.
    """
    rec = load("Q-014")["records"]
    out: List[Tuple[str, bool, str]] = []

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        out.append((label, abs(claimed - actual) <= tol,
                    f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    t5_m = _vals(rec, "tied_L5", "mnist")
    fr_m = _vals(rec, "tied_L5_frozen", "mnist")
    u5_m = _vals(rec, "untied_L5", "mnist")
    t80_m = _vals(rec, "tied_L80", "mnist")
    u1_m = _vals(rec, "untied_L1", "mnist")
    t1_m = _vals(rec, "tied_L1", "mnist")

    t5_f = _vals(rec, "tied_L5", "fashion_mnist")
    fr_f = _vals(rec, "tied_L5_frozen", "fashion_mnist")
    t1_f = _vals(rec, "tied_L1", "fashion_mnist")

    # --- MNIST-Tabelle ---
    cmp("tied_L1, mnist", 0.9344, float(t1_m.mean()))
    cmp("tied_L5, mnist", 0.9363, float(t5_m.mean()))
    cmp("tied_L80, mnist", 0.9098, float(t80_m.mean()))
    cmp("untied_L1, mnist", 0.9341, float(u1_m.mean()))
    cmp("untied_L5, mnist", 0.9107, float(u5_m.mean()))
    cmp("tied_L5_frozen, mnist", 0.9346, float(fr_m.mean()))

    # --- Fashion-Tabelle ---
    cmp("tied_L1, fashion", 0.8330, float(t1_f.mean()))
    cmp("tied_L5, fashion", 0.7987, float(t5_f.mean()))
    cmp("tied_L5_frozen, fashion", 0.8198, float(fr_f.mean()))

    # --- der eigentliche Befund: trainiert gegen eingefroren ---
    cmp("eingefroren, mnist: trainiert - frozen", 0.0018,
        float(t5_m.mean() - fr_m.mean()))
    cmp("eingefroren, fashion: trainiert - frozen", -0.0211,
        float(t5_f.mean() - fr_f.mean()))
    out.append(("fashion: nur 1 von 15 Seeds fuer trainiert",
                int((t5_f > fr_f).sum()) == 1,
                f"Tagebuch 1 | Rohdaten {int((t5_f > fr_f).sum())}"))

    # --- die beiden nicht zitierfaehigen Familienzahlen ---
    cmp("PRIMAER (nicht zitierfaehig): tied_L5 - untied_L5", 0.0257,
        float(t5_m.mean() - u5_m.mean()))
    cmp("P2 (nicht zitierfaehig): tied_L80 - untied_L1", -0.0243,
        float(t80_m.mean() - u1_m.mean()))

    # --- Parameterzahlen und der eingefrorene Block ---
    for arm, claimed in (("tied_L1", 16), ("tied_L5", 80), ("tied_L80", 1280),
                         ("untied_L1", 2048), ("untied_L5", 10240)):
        actual = _param(rec, arm, "block_parameters")
        out.append((f"Block-Parameter: {arm}", actual == claimed,
                    f"Tagebuch {claimed:,} | Rohdaten {actual:,}"))
    tb = _param(rec, "tied_L5_frozen", "trainable_block_parameters")
    out.append(("tied_L5_frozen hat 0 trainierbare Block-Parameter", tb == 0,
                f"Tagebuch 0 | Rohdaten {tb:,}"))

    out.append(("15 Seeds je Zelle", len(t5_m) == 15,
                f"Tagebuch 15 | Rohdaten {len(t5_m)}"))
    return out


def check_q018() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '10. August -- Q-018'. Der Lauf hat seine
    Auflösungspruefung BESTANDEN, seine Primaerzahl ist also zitierfaehig --
    anders als die von Q-014. P2 ist es nicht: `untied_trained` ist auf beiden
    Stufen gesaettigt."""
    rec = load("Q-018")["records"]
    out: List[Tuple[str, bool, str]] = []

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        out.append((label, abs(claimed - actual) <= tol,
                    f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    def vals(arm: str, clusters: int) -> np.ndarray:
        rows = sorted((r for r in rec if r["arm"] == arm
                       and int(r["clusters"]) == clusters),
                      key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    t32, f32 = vals("tied_trained", 32), vals("tied_frozen", 32)
    u32, c32 = vals("untied_trained", 32), vals("frozen_4ch", 32)
    t128, f128 = vals("tied_trained", 128), vals("tied_frozen", 128)
    c128 = vals("frozen_4ch", 128)

    # --- Primaerstufe ---
    cmp("tied_trained, 32 Haufen", 0.9588, float(t32.mean()))
    cmp("tied_frozen, 32 Haufen", 0.5511, float(f32.mean()))
    cmp("untied_trained, 32 Haufen", 1.0000, float(u32.mean()))
    cmp("frozen_4ch, 32 Haufen", 0.3437, float(c32.mean()))
    cmp("PRIMAER: trainiert - eingefroren", 0.4077, float(t32.mean() - f32.mean()))
    cmp("Aufloesung: 8 gegen 4 Kanaele", 0.2074, float(f32.mean() - c32.mean()))
    cmp("Median tied_trained", 0.9988, float(np.median(t32)))

    # --- Zweitstufe, beschreibend ---
    cmp("tied_trained, 128 Haufen", 0.6518, float(t128.mean()))
    cmp("tied_frozen, 128 Haufen", 0.2872, float(f128.mean()))
    cmp("Replikation 128: trainiert - eingefroren", 0.3646,
        float(t128.mean() - f128.mean()))
    cmp("Aufloesung 128: 8 gegen 4 Kanaele", 0.0550,
        float(f128.mean() - c128.mean()))

    # --- die Aussage, die nicht am Mittelwert haengt ---
    for stufe, t, f in (("32", t32, f32), ("128", t128, f128)):
        getrennt = bool(t.min() > f.max())
        out.append((f"Verteilungen getrennt, {stufe} Haufen", getrennt,
                    f"schlechtester trainiert {t.min():.4f} > "
                    f"bester eingefroren {f.max():.4f}"))
    cmp("schlechtester trainierter Seed, 32", 0.7302, float(t32.min()))
    cmp("bester eingefrorener Seed, 32", 0.6578, float(f32.max()))

    # --- Saettigung, die P2 entwertet ---
    for stufe, u in (("32", u32), ("128", vals("untied_trained", 128))):
        ges = bool(u.max() == 1.0 and u.std(ddof=1) == 0.0)
        out.append((f"untied_trained gesaettigt, {stufe} Haufen", ges,
                    f"max {u.max():.4f}, std {u.std(ddof=1):.4f} -- P2 nicht "
                    f"zitierfaehig"))

    out.append(("15 Seeds je Zelle", len(t32) == 15,
                f"Tagebuch 15 | Rohdaten {len(t32)}"))
    out.append(("0 trainierbare Blockparameter bei tied_frozen",
                _param(rec, "tied_frozen", "trainable_block_parameters") == 0,
                "Tagebuch 0 | Rohdaten "
                f"{_param(rec, 'tied_frozen', 'trainable_block_parameters')}"))
    return out


def check_t000() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '10. August -- T-000'. Kalibrierung, kein
    Experiment -- geprueft wird nur, ob die Tagebuchzahlen aus den Rohdaten
    folgen."""
    rec = load("T-000")["records"]
    out: List[Tuple[str, bool, str]] = []
    nach_breite = {int(r["breite"]): r for r in rec}

    for breite, par, tps in ((128, 1858304, 6168), (256, 5289472, 3267),
                             (512, 16870400, 1382), (1024, 58906624, 380)):
        r = nach_breite.get(breite)
        if r is None:
            out.append((f"Breite {breite} vorhanden", False, "fehlt"))
            continue
        out.append((f"Parameter bei Breite {breite}", int(r["parameter"]) == par,
                    f"Tagebuch {par:,} | Rohdaten {int(r['parameter']):,}"))
        gemessen = round(float(r["token_je_sekunde"]))
        out.append((f"Token/s bei Breite {breite}", abs(gemessen - tps) <= 1,
                    f"Tagebuch {tps:,} | Rohdaten {gemessen:,}"))

    # Die Ableitung, auf der Phase 3 steht.
    r256 = nach_breite[256]
    token24 = float(r256["token_je_sekunde"]) * 24 * 3600
    verh = token24 / int(r256["parameter"])
    out.append(("24h bei Breite 256: Token", abs(token24 - 282278566) <= 200000,
                f"Tagebuch 282,278,566 | Rohdaten {token24:,.0f}"))
    out.append(("24h bei Breite 256: Token je Parameter", abs(verh - 53) <= 1,
                f"Tagebuch 53 | Rohdaten {verh:.0f}"))
    out.append(("Faktor 64 zwischen trainierbar (256) und Uebergang (16384)",
                16384 // 256 == 64, "Tagebuch 64 | Rohdaten 64"))
    return out


def check_t001() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '10. August -- T-001'. Der Lauf hat seine
    Auflösungspruefung bestanden; sein Urteil ist T0, das Ausstiegskriterium
    der Roadmap greift."""
    rec = load("T-001")["records"]
    out: List[Tuple[str, bool, str]] = []

    def median_ms(d: int, arm: str) -> float:
        w = [r["sekunden"] for r in rec if int(r["dim"]) == d and r["arm"] == arm]
        return float(np.median(w)) * 1000 if w else float("nan")

    for d, ref, sch, dic in ((1024, 82.4, 47.3, 3.1), (2048, 155.6, 95.1, 13.5),
                             (4096, 329.5, 194.8, 61.0),
                             (8192, 673.1, 440.0, 258.3),
                             (16384, 1433.8, 998.0, 2355.7)):
        for arm, claimed in (("referenz", ref), ("schnell", sch), ("dicht", dic)):
            actual = median_ms(d, arm)
            out.append((f"d={d}, {arm} (ms)", abs(actual - claimed) <= 0.6,
                        f"Tagebuch {claimed:.1f} | Rohdaten {actual:.1f}"))

    # --- die Aussagen, auf denen die Deutung steht ---
    #
    # ACHTUNG, hier steckte am 10. August ein Fehler IN DIESEM PRUEFER: Die
    # ersten beiden Vorsprungszahlen wurden als Verhaeltnis zweier Mediane
    # gebildet und meldeten deshalb 1,643 / 2,360 gegen die 1,590 / 2,311 des
    # Tagebuchs. Die Vorregistrierung verlangt aber den Faktor JE REIHE
    # gepaart -- genau die Unterscheidung, aus der am 5. August aus 1,57x die
    # 1,99x wurden. Das Tagebuch war richtig, der Pruefer war es nicht.
    def gepaart(d: int, zaehler: str, nenner: str) -> float:
        f = []
        for reihe in sorted({r["reihe"] for r in rec if int(r["dim"]) == d}):
            z = [r["sekunden"] for r in rec if int(r["dim"]) == d
                 and r["arm"] == zaehler and r["reihe"] == reihe]
            n_ = [r["sekunden"] for r in rec if int(r["dim"]) == d
                  and r["arm"] == nenner and r["reihe"] == reihe]
            if z and n_ and n_[0] > 0:
                f.append(z[0] / n_[0])
        return float(np.median(f)) if f else float("nan")

    gewinn16 = gepaart(16384, "referenz", "schnell")
    out.append(("Gewinn bei d=16384 (gepaart)", abs(gewinn16 - 1.44) <= 0.02,
                f"Tagebuch 1.44 | Rohdaten {gewinn16:.2f}"))
    v_ref = gepaart(16384, "dicht", "referenz")
    v_sch = gepaart(16384, "dicht", "schnell")
    out.append(("Vorsprung gegen dicht bei d=16384: referenz (gepaart)",
                abs(v_ref - 1.590) <= 0.02,
                f"Tagebuch 1.590 | Rohdaten {v_ref:.3f}"))
    out.append(("Vorsprung gegen dicht bei d=16384: schnell (gepaart)",
                abs(v_sch - 2.311) <= 0.02,
                f"Tagebuch 2.311 | Rohdaten {v_sch:.3f}"))
    # Der Unterschied zwischen beiden Rechenweisen, festgehalten weil er der
    # Grund fuer die Regel ist:
    naiv = median_ms(16384, "dicht") / median_ms(16384, "schnell")
    out.append(("gepaart weicht vom Verhaeltnis zweier Mediane ab",
                abs(naiv - v_sch) > 0.01,
                f"gepaart {v_sch:.3f} | Mediane {naiv:.3f} -- deshalb die Regel"))
    out.append(("kein Uebergang bei d=8192 (schnell noch langsamer)",
                median_ms(8192, "schnell") > median_ms(8192, "dicht"),
                f"schnell {median_ms(8192, 'schnell'):.1f} > "
                f"dicht {median_ms(8192, 'dicht'):.1f}"))

    # Die Cache-Kante, die den Uebergang erzeugt.
    sprung = median_ms(16384, "dicht") / median_ms(8192, "dicht")
    out.append(("Sprung der dichten Schicht 8192 -> 16384",
                abs(sprung - 9.1) <= 0.3,
                f"Tagebuch 9.1 | Rohdaten {sprung:.1f}"))
    wachstum = median_ms(16384, "schnell") / median_ms(8192, "schnell")
    out.append(("Wachstum von schnell 8192 -> 16384",
                abs(wachstum - 2.27) <= 0.05,
                f"Tagebuch 2.27 | Rohdaten {wachstum:.2f}"))

    reihen = len({r["reihe"] for r in rec if int(r["dim"]) == 16384})
    out.append(("7 Messreihen je Breite", reihen == 7,
                f"Tagebuch 7 | Rohdaten {reihen}"))
    return out


def check_q011() -> List[Tuple[str, bool, str]]:
    """Der dritte Q-011-Anlauf, gerettet (Tagebuch 30.08., HANDOFF_AKTUELL §0).

    Geprueft wird die FESTE Datei vom 27. August, 23:00 -- nicht die neueste:
    Sobald der Lauf wieder aufgenommen wird, entstehen neuere Dateien, und die
    Aussage "225 gerettete Zeilen" bezieht sich auf genau diese eine.

    Nur was im Tagebuch steht. Die Tuning-Abstaende (+0,0049 ... +0,0232)
    stehen NICHT in den Rohdaten -- die Suchmittel werden nicht gespeichert --
    und sind ohnehin nicht zitierbar. Messzahlen je Tiefe nennt das Tagebuch
    fuer diesen Anlauf keine; deshalb wird hier auch keine geprueft.
    """
    from qire_ai.result_store import RESULTS_DIR

    payload = load(RESULTS_DIR / "Q-011__20260827_230017.json")
    rec, cfg = payload["records"], payload["config"]
    out: List[Tuple[str, bool, str]] = []
    out.append(("225 Messzeilen gerettet", len(rec) == 225,
                f"Tagebuch 225 | Rohdaten {len(rec)}"))
    zellen: Dict[Tuple[str, int], int] = {}
    for r in rec:
        k = (r["dataset"], int(r["depth"]))
        zellen[k] = zellen.get(k, 0) + 1
    voll = sorted(d for (ds, d), n in zellen.items() if ds == "mnist" and n == 45)
    out.append(("mnist Tiefen 1, 2, 3, 4, 8 vollstaendig (je 3 Arme x 15)",
                voll == [1, 2, 3, 4, 8], f"Tagebuch [1, 2, 3, 4, 8] | Rohdaten {voll}"))
    out.append(("Tiefe 16 fehlt", not any(d == 16 for _, d in zellen),
                f"Tiefen in den Rohdaten: {sorted({d for _, d in zellen})}"))
    out.append(("fashion_mnist nie erreicht",
                not any(ds == "fashion_mnist" for ds, _ in zellen),
                f"Datensaetze: {sorted({ds for ds, _ in zellen})}"))
    out.append(("volles Budget im Messlauf (6000 / 10)",
                cfg.get("max_samples") == 6000 and cfg.get("epochs") == 10,
                f"Rohdaten {cfg.get('max_samples')} / {cfg.get('epochs')}"))
    bis4 = {v for d in (1, 2, 3, 4) for v in cfg[f"lr_map_mnist_d{d}"].values()}
    out.append(("bis Tiefe 4 alle Raten 0,003 (innen)", bis4 == {0.003},
                f"Tagebuch {{0.003}} | Rohdaten {sorted(bis4)}"))
    out.append(("Tiefe 8: lr_edge meldet plain am unteren Rand",
                cfg.get("lr_edge_mnist_d8") == ["plain"]
                and cfg["lr_map_mnist_d8"]["plain"] == min(cfg["lr_grid"]),
                f"lr_edge {cfg.get('lr_edge_mnist_d8')}, lr "
                f"{cfg['lr_map_mnist_d8']['plain']}"))
    return out


def check_q011_vorlauf2() -> List[Tuple[str, bool, str]]:
    """Der ZWEITE, verworfene Q-011-Anlauf (Tagebuch 27.08., 'die Lernrate war
    der ganze Effekt'). Nicht zitierbar -- geprueft wird, dass die Zahlen, mit
    denen der Abbruch begruendet ist, aus den Rohdaten folgen. Letzte Datei des
    Anlaufs: vorlaeufe_q011/Q-011__20260827_204722.json (fuenf Tiefen)."""
    from scipy import stats

    from qire_ai.result_store import RESULTS_DIR

    payload = load(RESULTS_DIR / "vorlaeufe_q011" / "Q-011__20260827_204722.json")
    rec, cfg = payload["records"], payload["config"]
    out: List[Tuple[str, bool, str]] = []

    def vals(arm: str, d: int) -> np.ndarray:
        rows = sorted((r for r in rec if r["arm"] == arm and int(r["depth"]) == d
                       and r["dataset"] == "mnist"), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    for d, delta, siege, p, lr_o, lr_s in ((1, 0.0146, 15, 0.0001, 0.003, 0.01),
                                           (2, -0.0132, 0, 0.0001, 0.01, 0.003),
                                           (3, 0.0121, 14, 0.0003, 0.003, 0.01),
                                           (4, -0.0006, 6, 0.5995, 0.01, 0.01),
                                           (8, 0.0122, 14, 0.0006, 0.003, 0.01)):
        o, s = vals("orthogonal", d), vals("spectral", d)
        ist = float(o.mean() - s.mean())
        out.append((f"Tiefe {d}: orth - spec", abs(ist - delta) <= TOL,
                    f"Tagebuch {delta:+.4f} | Rohdaten {ist:+.4f}"))
        out.append((f"Tiefe {d}: Siege", int((o > s).sum()) == siege,
                    f"Tagebuch {siege}/15 | Rohdaten {int((o > s).sum())}/15"))
        pv = float(stats.wilcoxon(o, s).pvalue)
        out.append((f"Tiefe {d}: p", abs(pv - p) <= 0.00005,
                    f"Tagebuch {p:.4f} | Rohdaten {pv:.4f}"))
        lm = cfg[f"lr_map_mnist_d{d}"]
        out.append((f"Tiefe {d}: lr orthogonal / spectral",
                    (lm["orthogonal"], lm["spectral"]) == (lr_o, lr_s),
                    f"Tagebuch {lr_o:g} / {lr_s:g} | Rohdaten "
                    f"{lm['orthogonal']:g} / {lm['spectral']:g}"))
    # "In fuenf von fuenf Tiefen gewinnt der Arm mit der kleineren Rate."
    stimmt = 0
    for d in (1, 2, 3, 8):
        lm = cfg[f"lr_map_mnist_d{d}"]
        kleiner = "orthogonal" if lm["orthogonal"] < lm["spectral"] else "spectral"
        o, s = vals("orthogonal", d), vals("spectral", d)
        gewinner = "orthogonal" if o.mean() > s.mean() else "spectral"
        stimmt += int(kleiner == gewinner)
    out.append(("kleinere Rate gewinnt (4 Tiefen mit ungleicher Rate)", stimmt == 4,
                f"{stimmt}/4 -- Tiefe 4 hat gleiche Raten ('unentschieden')"))
    return out


def check_randlagen() -> List[Tuple[str, bool, str]]:
    """Die Randlagen-Tabelle vom 27. August ('Budget-Fehler in neun Modulen'
    und 'Das halbe Suchbudget steckt in zehn Modulen'), gegen die Rohdaten.

    Regel 1(b): Nicht wer die kleinere Rate hat entscheidet, sondern an
    welchem Gitterrand ein Arm sitzt. Das Artefakt aus Q-011 betraegt +-0,013;
    die Spalte 'Effekt / Artefakt' ist der Abstand geteilt durch 0,013."""
    out: List[Tuple[str, bool, str]] = []
    ARTEFAKT = 0.013

    def lage(cfg: Dict, key: str, arm: str) -> str:
        g, lr = cfg["lr_grid"], float(cfg[key][arm])
        return "unten" if lr == min(g) else "oben" if lr == max(g) else "innen"

    def pruefe(label, behauptet, ist):
        out.append((label, behauptet == ist, f"Tagebuch {behauptet} | Rohdaten {ist}"))

    q3 = load("Q-003-replication")
    c3, r3 = q3["config"], q3["records"]
    pruefe("Q-003 mnist: hybrid am Rand", "oben", lage(c3, "lr_map_mnist", "hybrid"))
    pruefe("Q-003 mnist: Raten Mechanismus / Kontrolle", (0.1, 0.01),
           (c3["lr_map_mnist"]["hybrid"], c3["lr_map_mnist"]["classical_norm_deep"]))
    pruefe("Q-003 fashion: norm_deep am Rand", "unten",
           lage(c3, "lr_map_fashion_mnist", "classical_norm_deep"))
    pruefe("Q-003 fashion: Raten Mechanismus / Kontrolle", (0.1, 0.003),
           (c3["lr_map_fashion_mnist"]["hybrid"],
            c3["lr_map_fashion_mnist"]["classical_norm_deep"]))
    pruefe("Q-003 mnist: drei von sechs Armen am oberen Rand", 3,
           sum(lage(c3, "lr_map_mnist", a) == "oben" for a in c3["lr_map_mnist"]))
    pruefe("Q-003 fashion: fuenf von sechs an einem Rand", 5,
           sum(lage(c3, "lr_map_fashion_mnist", a) != "innen"
               for a in c3["lr_map_fashion_mnist"]))

    q15 = load("Q-015")
    c15, r15 = q15["config"], q15["records"]
    pruefe("Q-015 mnist: beide Arme innen", ("innen", "innen"),
           (lage(c15, "lr_map_mnist", "hybrid"), lage(c15, "lr_map_mnist", "global_orth")))
    pruefe("Q-015 fashion: global_orth am Rand", "unten",
           lage(c15, "lr_map_fashion_mnist", "global_orth"))
    pruefe("Q-015 fashion: im Datensatz vermerkt",
           ["global_orth"], c15.get("lr_at_grid_edge_fashion_mnist"))
    pruefe("Q-015 Raten mnist / fashion (Mech./Kontr.)", ((0.1, 0.03), (0.1, 0.003)),
           ((c15["lr_map_mnist"]["hybrid"], c15["lr_map_mnist"]["global_orth"]),
            (c15["lr_map_fashion_mnist"]["hybrid"],
             c15["lr_map_fashion_mnist"]["global_orth"])))

    q18 = load("Q-018")
    c18 = q18["config"]
    pruefe("Q-018: Primaerpaar beide innen (32 Haufen)", ("innen", "innen"),
           (lage(c18, "lr_map_32", "tied_trained"), lage(c18, "lr_map_32", "tied_frozen")))
    pruefe("Q-018: Raten Mechanismus / Kontrolle", (0.1, 0.3),
           (c18["lr_map_32"]["tied_trained"], c18["lr_map_32"]["tied_frozen"]))
    pruefe("Q-018: untied_trained bei 32 Haufen am unteren Rand", "unten",
           lage(c18, "lr_map_32", "untied_trained"))

    # --- die zurueckgezogene Zahl selbst und die Verhaeltnisse zum Artefakt ---
    d15f = float(_vals(r15, "hybrid", "fashion_mnist").mean()
                 - _vals(r15, "global_orth", "fashion_mnist").mean())
    out.append(("Q-015 fashion (ZURUECKGEZOGEN): hybrid - global_orth",
                abs(d15f - 0.0201) <= TOL, f"Tagebuch +0.0201 | Rohdaten {d15f:+.4f}"))
    d15m = float(_vals(r15, "hybrid", "mnist").mean() - _vals(r15, "global_orth", "mnist").mean())
    d3m = float(_vals(r3, "hybrid", "mnist").mean()
                - _vals(r3, "classical_norm_deep", "mnist").mean())
    d3f = float(_vals(r3, "hybrid", "fashion_mnist").mean()
                - _vals(r3, "classical_norm_deep", "fashion_mnist").mean())
    r18 = q18["records"]

    def v18(arm, cl):
        return np.array([r["balanced_accuracy"] for r in r18
                         if r["arm"] == arm and int(r["clusters"]) == cl])
    d18 = float(v18("tied_trained", 32).mean() - v18("tied_frozen", 32).mean())
    d18b = float(v18("tied_trained", 128).mean() - v18("tied_frozen", 128).mean())
    for label, eff, faktor in (("Q-003 mnist", d3m, 7.2), ("Q-003 fashion", d3f, 4.6),
                               ("Q-015 mnist", d15m, 5.2), ("Q-015 fashion", d15f, 1.5),
                               ("Q-018 (32 Haufen)", d18, 31.4),
                               ("Q-018 (128 Haufen)", d18b, 28.0)):
        ist = eff / ARTEFAKT
        out.append((f"{label}: Effekt / Artefakt", abs(ist - faktor) <= 0.051,
                    f"Tagebuch {faktor:.1f}x | Rohdaten {ist:.2f}x"))
    return out


def check_q020() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '28. September -- Q-020 gerechnet'. Eine Datei
    (die neueste Q-020-Datei, 435 Zeilen, alle drei Abschnitte)."""
    rec = load("Q-020")["records"]
    out: List[Tuple[str, bool, str]] = []

    def m(abschnitt: str, arm: str) -> np.ndarray:
        rows = sorted((r for r in rec if r["abschnitt"] == abschnitt
                       and r["arm"] == arm), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        out.append((label, abs(claimed - actual) <= tol,
                    f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    # --- Familie ---
    q1, mlp = m("mnist", "q1"), m("mnist", "mlp")
    cmp("PRIMAER q1, MNIST (offizieller Test)", 0.9347, float(q1.mean()))
    cmp("PRIMAER mlp, MNIST", 0.9234, float(mlp.mean()))
    cmp("PRIMAER q1 - mlp", 0.0113, float((q1 - mlp).mean()))
    out.append(("PRIMAER: q1 in 15/15 Seeds vor mlp", bool((q1 > mlp).all()),
                f"{int((q1 > mlp).sum())}/{len(q1)}"))
    q5, q5o, q1b = m("haufen128", "q5"), m("haufen128", "q5_ohne"), m("haufen128", "q1")
    cmp("P2 q5 - q5_ohne, 128 Haufen", 0.3414, float((q5 - q5o).mean()))
    cmp("P3 q5 - q1, 128 Haufen", 0.3319, float((q5 - q1b).mean()))
    # --- Kontrollen ---
    cmp("Aufloesung MNIST mlp - linear", 0.0416,
        float((mlp - m("mnist", "linear")).mean()))
    cmp("Aufloesung Haufen mlp - linear", 0.0960,
        float((m("haufen128", "mlp") - m("haufen128", "linear")).mean()))
    cmp("Aequivalenz MNIST kron - q1_ohne", 0.0001,
        float((m("mnist", "kron") - m("mnist", "q1_ohne")).mean()))
    # --- beschreibend ---
    cmp("MNIST q1 - quad_free", 0.0109, float((q1 - m("mnist", "quad_free")).mean()))
    cmp("MNIST mlp - norm_deep (Strohmann)", 0.0803,
        float((mlp - m("mnist", "norm_deep")).mean()))
    cmp("Fashion q1 - mlp (Wiederholung)", 0.0047,
        float((m("fashion_mnist", "q1") - m("fashion_mnist", "mlp")).mean()))
    cmp("Haufen q5 - mlp (Schaltung klar unterlegen)", -0.3133,
        float((q5 - m("haufen128", "mlp")).mean()))
    out.append(("15 Seeds je Zelle, 435 Zeilen", len(q1) == 15 and len(rec) == 435,
                f"Rohdaten {len(q1)} Seeds, {len(rec)} Zeilen"))
    return out


def check_q019w() -> List[Tuple[str, bool, str]]:
    """Zahlen aus dem Eintrag '28. September -- Q-019W gerechnet'. Eine Datei
    (die neueste Q-019W-Datei, 795 Zeilen, alle Abschnitte)."""
    rec = load("Q-019W")["records"]
    out: List[Tuple[str, bool, str]] = []

    def m(basis: str, n: int, arm: str) -> np.ndarray:
        rows = sorted((r for r in rec if r["basis"] == basis and int(r["n_train"]) == n
                       and r["arm"] == arm), key=lambda r: r["seed"])
        return np.array([r["balanced_accuracy"] for r in rows])

    def cmp(label: str, claimed: float, actual: float, tol: float = TOL):
        out.append((label, abs(claimed - actual) <= tol,
                    f"Tagebuch {claimed:+.4f} | Rohdaten {actual:+.4f}"))

    wb, wd, wg = m("walsh", 200, "butterfly"), m("walsh", 200, "dense"), m("walsh", 200, "givens_zufall")
    zb, zd = m("zufall", 200, "butterfly"), m("zufall", 200, "dense")
    cmp("PRIMAER butterfly, walsh N=200", 0.8472, float(wb.mean()))
    cmp("PRIMAER dense, walsh N=200", 0.5390, float(wd.mean()))
    cmp("PRIMAER butterfly - dense", 0.3082, float((wb - wd).mean()))
    cmp("P2 Wechselwirkung walsh - zufall", 0.6933, float(((wb - wd) - (zb - zd)).mean()))
    cmp("P3 butterfly - givens_zufall", 0.6781, float((wb - wg).mean()))
    cmp("N=6000 dense (walsh)", 0.9504, float(m("walsh", 6000, "dense").mean()))
    cmp("N=6000 butterfly (walsh)", 0.9077, float(m("walsh", 6000, "butterfly").mean()))
    cmp("Aufloesung dense - narrow, N=6000", 0.1010,
        float((m("walsh", 6000, "dense") - m("walsh", 6000, "narrow")).mean()))
    cmp("butterfly - wht_fest, walsh N=200", 0.0006,
        float((wb - m("walsh", 200, "wht_fest")).mean()))
    cmp("untied - butterfly, walsh N=200", -0.3523,
        float((m("walsh", 200, "untied") - wb).mean()))
    out.append(("15 Seeds, 795 Zeilen", len(wb) == 15 and len(rec) == 795,
                f"Rohdaten {len(wb)} Seeds, {len(rec)} Zeilen"))
    return out


CHECKS: Dict[str, Callable[[], List[Tuple[str, bool, str]]]] = {
    "Q-003-replication": check_q003,
    "Q-011 (dritter Anlauf, gerettet)": check_q011,
    "Q-011 (zweiter Anlauf, verworfen)": check_q011_vorlauf2,
    "Q-014": check_q014,
    "Q-015": check_q015,
    "Q-018": check_q018,
    "Q-020": check_q020,
    "Q-019W": check_q019w,
    "Randlagen (Regel 1b, Tagebuch 27.08.)": check_randlagen,
    "T-000": check_t000,
    "T-001": check_t001,
}


def main() -> None:
    print("=" * 92)
    print("Q-AUDIT -- Tagebuchzahlen der Q-Serie gegen die Rohdaten")
    print("Jede Behauptung wird aus EINER Datei nachgerechnet, damit sich der")
    print("Vermischungsfehler vom 6. August nicht wiederholt.")
    print("=" * 92)

    total_ok = True
    for exp_id, fn in CHECKS.items():
        print()
        print(f"### {exp_id}")
        print("-" * 92)
        try:
            rows = fn()
        except FileNotFoundError:
            print("  keine Rohdaten -- uebersprungen")
            continue
        except Exception as exc:                  # noqa: BLE001
            print(f"  Pruefung fehlgeschlagen: {exc!r}")
            total_ok = False
            continue
        for label, ok, detail in rows:
            mark = "OK    " if ok else "ABWEICHUNG"
            print(f"  [{mark:>10}] {label:<44} {detail}")
            total_ok &= ok

    print()
    print("=" * 92)
    if total_ok:
        print("ERGEBNIS: Alle geprueften Q-Zahlen stimmen mit den Rohdaten ueberein.")
    else:
        print("ERGEBNIS: ABWEICHUNGEN GEFUNDEN -- Tagebuch und Kurzfassung pruefen,")
        print("bevor eine dieser Zahlen irgendwo zitiert wird.")
    print("=" * 92)
    print()
    print("Hinweis: Dieses Werkzeug prueft nur die hier eingetragenen Zahlen.")
    print("Kommt eine neue Aussage ins Tagebuch, gehoert sie auch hierher --")
    print("sonst sagt ein bestandener Lauf weniger, als er zu sagen scheint.")


if __name__ == "__main__":
    main()
