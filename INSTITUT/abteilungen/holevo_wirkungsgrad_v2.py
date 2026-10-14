"""Abteilung Clifford/Information -- I-040-v2 „Holevo-Wirkungsgrad, Wiederholung mit begruendeter Positivkontrolle“
(Quantenstufe Q2: exakte Simulation; die Frage ist informationstheoretisch/klassisch).

FOLGE VON I-040 (LABORBUCH „I-040 ERGEBNIS“): PK-1 versagte (Identitaet, exakter Kopf, 0,868 < absolute Schwelle 0,88),
  daher H2/H3 HU; H1 (Identitaet beste I pro Schuss, +0,20 Bit gegen Haar/Suche) bestaetigt.

URSACHE DER 0,868 (Diagnose aus Dateien, keine Messzahl auf Mess-Seeds): I-038 erreicht fuer Identitaet² 0,9005 (MNIST, 20 Seeds)
  mit DEMSELBEN Datenumfang (6.000 Training / 5.000 Test, 16x16), aber Kopf L2 = 3e-3, L-BFGS max_iter = 500.
  I-040 waehlte die L2-Staerke des exakten Kopfs auf dem Tuning-Seed 2098 nach der Validierung unter SCHUSSRAUSCHEN
  (S in {30,100}) -> L2 = 0,09 (30x staerker), dazu max_iter = 200. Die starke Regularisierung stuetzt die
  verrauschte Auslese, kostet aber die Genauigkeit bei exakter Auslese (S -> unendlich). Die Daten waren NICHT die Ursache.
  Die absolute Schwelle 0,88 stammte ausserdem aus v4 (anderer Kopf) und war ohne Bezug zu I-038 gesetzt.

NEU IN v2 (alles vor jeder Zahl auf Mess-Seeds festgelegt):
  (i) KOPF-ANGLEICH: der Kopf „exakt“ ist der I-038-Kopf (Softmax-Regression, standardisierte Merkmale, L2 = 3e-3 FEST,
      max_iter = 500, keine Auswahl). Auf ihm beruhen PK-1 und acc_inf.
  (ii) RELATIVE PK-1: Identitaet exakt >= Pixel-Softmax - 2 pp (Pixel-Softmax = I-038-Kopf auf Rohpixeln, gleiche Seeds,
      gleiche Daten, gepaart). Begruendung der 2 pp: I-038 (20 Seeds) Identitaet² 0,9005 gegen Pixel-Softmax 0,9094 =
      -0,9 pp (Fashion -2,3 pp); die Abstandsgrenze 2 pp laesst dem erwarteten Wert (-0,9 pp) gut 1 pp Luft und
      bleibt eine echte Pruefung (Kopf-Maschinerie kaputt oder viel zu stark regularisiert fiele um > 2 pp).
      Tuning-Seed-Referenz (nur Kopf, 2198, Volldaten) steht unten in TUNING_REFERENZ und im LABORBUCH.
  (iii) FAIRER H2: der exakte Gegner ist je (Seed, Arm, S) der BESSERE von zwei exakt trainierten Koepfen:
      „exakt“ (L2 3e-3, s. o.) und „exakt_rob“ (auf exakten Wahrscheinlichkeiten trainiert, L2 auf Tuning-Seed 2198
      nach Validierung unter Schussrauschen gewaehlt, wie I-040). Das schussbewusste Training muss also auch den
      rauschoptimierten exakten Kopf schlagen.
  (iv) H4 ZIELKONFLIKT (neu): Haar² exakt > Identitaet exakt (Genauigkeit, acc_inf) UND I(Identitaet) > I(Haar)
      (Information pro Schuss), beide gepaart einseitig, Schnitt-Vereinigungs-Test (p = max).
  (v) Familie {H1, H2, H3, H4}, Holm, alpha = 0,05. Seeds 2150-2159, Tuning 2198 (disjunkt zu I-040: 2050-2059/2098).

ARME (je Seed dieselben Daten; Born-Merkmale p = (W x/|x|)^2, 256 Ausgaenge, 8 Qubits; Details wie I-040):
  id, haar, ff2, bent4, suche (seed-eigen, LOO-Ridge wie I-038), nk (Negativkontrolle: permutierte Labels),
  pix (Referenz, nur fuer PK-1: Pixel-Softmax, I-038-Kopf auf Rohpixeln).
KOEPFE: exakt (I-038-Kopf), exakt_rob (L2 Tuning), bewusst (auf Schusshaeufigkeiten gleicher Schusszahl S trainiert,
  L2 auf Tuning-Seed 2198; Wahl: Mittel der Validierungsgenauigkeit ueber die vier Nicht-Such-Arme und S in {30,100},
  Gitter {1e-3,3e-3,1e-2,3e-2,1e-1} beidseitig um Faktor 3 verlaengerbar, hoechstens 4 Runden). S in {10,30,100,300,1000},
  3 Schussrealisierungen je Testbild, alle Koepfe auf denselben Ziehungen (gepaart).
INFORMATION: I(Klasse;1 Schuss) = I_C (Klassenmittel der Testbilder, Prior = Trainingsanteile), chi_C (Holevo, Trainings-
  Zustandsmatrizen), eta = I_C/chi_C; S90 = log-lineare Kreuzung von 0,90 auf der seedgemittelten Kurve des bewussten
  Kopfes (nicht erreicht: zensiert, S := 2000).

HYPOTHESEN (Familie {H1..H4}, Holm; 10 Seeds, kleinster einseitiger exakter Wilcoxon-p = 2^-10):
  H1  I(id) > I(haar) UND > I(suche) (gepaart, p = max beider; Replikation von I-040).
  H2  bewusst > bester exakter Kopf bei S in {10,30,100}: Differenz je S gemittelt ueber 5 Arme, gepaart ueber Seeds,
      p = max der drei. Beschreibend: je Arm, S >= 300, und gegen jeden der beiden exakten Koepfe einzeln.
  H3  Spearman eta ~ S90 ueber die 5 Arme (seedgemittelt), exakter Permutationstest, einseitig rho < 0 (nur 5 Punkte).
  H4  Zielkonflikt: (a) acc_inf(haar) - acc_inf(id) > 0 UND (b) I(id) - I(haar) > 0, je einseitiger exakter Wilcoxon,
      p = max(a, b) (Schnitt-Vereinigungs-Test), zusaetzlich beide Mittel > 0. (b) ist dieselbe Differenz wie
      H1-haar; ihr p-Wert geht in beiden Hypothesen ein (Holm bleibt gueltig, nur konservativ).
KONTROLLEN (in pruefe_vorbedingungen): PK-1 relativ (s. o.), PK-2 (exakter Kopf, id, S=10 mind. 10 pp unter S=1000),
  NK (I_C < 0,03 Bit und Genauigkeit <= 0,15). NK faellt: alle HU. PK-1/PK-2 faellt: H2, H3, H4 HU (H4 nur PK-1).
  Instabilitaet (Arm mit >= 2 Seeds, bewusste Genauigkeit S=100 < Median - 5 pp): H2, H3 HU.
URTEILSZWEIGE (Strings, Kriterien in urteil()): H1 IDENTITAET_BESTE | H0 NICHT_UEBERLEGEN | HU NICHT_ENTSCHIEDEN | HU NK_VERSAGT;
  H2 SCHUSSBEWUSST_BESSER | H0 EXAKT_BESSER | H0 KEIN_VORTEIL | HU NICHT_ENTSCHIEDEN | HU INSTABIL | HU PK_VERSAGT;
  H3 ETA_SAGT_VORAUS | H0 ETA_NICHT_PROGNOSTISCH | HU ZENSIERT | HU NICHT_ENTSCHIEDEN | HU INSTABIL | HU PK_VERSAGT;
  H4 ZIELKONFLIKT | H0 KEIN_KONFLIKT_GENAUIGKEIT | H0 KEIN_KONFLIKT_INFORMATION | HU NICHT_ENTSCHIEDEN | HU PK_VERSAGT.
DATEN/SEEDS: MNIST 16x16 (6.000/5.000), Mess-Seeds 2150-2159, Tuning 2198; Rauchtest NUR auf 2198. Fashion optional,
  nur beschreibend (fashion=True), ohne Urteil.
REALISTISCHE ERWARTUNG: PK-1 besteht (Erwartung Identitaet ~0,90 gegen Pixel ~0,91). H1 ja. H4 ja (Haar² ~+3,7 pp
  Genauigkeit nach I-038, I -0,2 Bit nach I-040). H2 bei S=10/30 wahrscheinlich, bei S=100 knapp, jetzt gegen den staerkeren
  Gegner schwerer als in I-040. H3 offen bis eher nein (v4: Identitaet hoechstes eta, aber 90 % nicht erreicht).
GRENZEN: wie I-040 (Testbilder aus denselben 10.000 offiziellen Bildern, eine Schussziehung je Trainingsbild,
  feste Bent-Form); eta-Neuheit nicht geprueft (Plan E4) -- nicht als neu formulieren. Die 2-pp-Schwelle ist eine
  begruendete Setzung, kein Messwert.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abteilungen import clifford_suche as v1c  # noqa: E402
from abteilungen import holevo_wirkungsgrad as v1  # noqa: E402
from abteilungen.holevo_wirkungsgrad import (S_GITTER, S_H2, N_REAL, ARME, ALPHA, S_ZENS, KOLLAPS_PP, PK2_ABSTAND,  # noqa: E402
                                             NK_I_MAX, NK_ACC_MAX, RHO_NEIN, H, info_C, klassenmittel, holevo_chi,
                                             schuesse, kopf_fit, kopf_acc, _gen, wilcoxon_einseitig, holm,
                                             spearman_perm, s_ziel, _stat, _kollaps, merkmale, normiere)
from labor import gemeinsam as gem  # noqa: E402

KARTE = "I-040-v2"
SEEDS = tuple(range(2150, 2160))
TUNING_SEED = 2198
SCHNELL_SEEDS = (2198,)
L2_EXAKT = 3e-3                      # I-038-Kopf, fest
ITER_EXAKT = 500                     # I-038-Kopf
MAX_ITER = 200                       # exakt_rob / bewusst wie I-040
L2_GITTER = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
PK1_REL = 0.02                       # Identitaet exakt >= Pixel-Softmax - 2 pp
TUNING_REFERENZ = {"seed": 2198, "stand": "Kopf-Referenz auf dem Tuning-Seed, siehe LABORBUCH (Vorregistrierung)"}
BELEGT = set(range(900, 1001)) | set(range(2000, 2070)) | {2090, 2091, 2092, 2095, 2098, 2099}
KOPF_ARME = ARME + ("nk", "pix")
HYPOTHESEN = {
    "H1": "I(Klasse;1 Schuss): Identitaet > Haar UND > Suche (gepaart, einseitiger exakter Wilcoxon, p = max).",
    "H2": "Schussbewusster Kopf > bester exakter Kopf bei S in {10,30,100} (Mittel ueber 5 Arme, je S einseitiger "
          "exakter Wilcoxon, p = max).",
    "H3": "Hoeheres eta = I/chi sagt kleineren Schussbedarf S90 voraus (Spearman ueber 5 Arme, exakter "
          "Permutationstest, einseitig rho < 0).",
    "H4": "Zielkonflikt: Haar² exakt > Identitaet exakt UND I(Identitaet) > I(Haar) (gepaart, einseitiger exakter "
          "Wilcoxon, p = max, Schnitt-Vereinigungs-Test).",
}
FOLGEFRAGEN = {
    "H4 ZIELKONFLIKT": "Messbasis, die beides liefert: Suche mit Ziel I(Klasse;Schuss) plus Genauigkeit (Idee B).",
    "H2 SCHUSSBEWUSST_BESSER": "Schussbewusstes Training als Standard fuer Hardware-Demo.",
    "H3 ETA_SAGT_VORAUS": "eta als Gueteklasse neben Genauigkeit berichten; Neuheitspruefung (E4) vor Formulierung.",
}


# ------------------------------------------------------------------ Koepfe
def kopf_kurven2(Ptr, ytr, Pte, yte, l2_rob, l2_bew, slist, seed, arm, max_iter=MAX_ITER, iter_exakt=ITER_EXAKT):
    """acc_inf, acc_exakt[S] (I-038-Kopf), acc_exakt_rob[S], acc_bewusst[S]; alle auf denselben Schussziehungen."""
    he = kopf_fit(Ptr, ytr, L2_EXAKT, iter_exakt)
    hr = kopf_fit(Ptr, ytr, l2_rob, max_iter)
    out = {"acc_inf": kopf_acc(he, Pte, yte), "acc_exakt": {}, "acc_exakt_rob": {}, "acc_bewusst": {}}
    tests = [schuesse(Pte, slist, _gen(seed, arm, "test", r)) for r in range(N_REAL)]
    for S in slist:
        out["acc_exakt"][str(S)] = float(np.mean([kopf_acc(he, t[S], yte) for t in tests]))
        out["acc_exakt_rob"][str(S)] = float(np.mean([kopf_acc(hr, t[S], yte) for t in tests]))
        ftr = schuesse(Ptr, [S], _gen(seed, arm, "train", S))[S]
        hb = kopf_fit(ftr, ytr, l2_bew, max_iter)
        out["acc_bewusst"][str(S)] = float(np.mean([kopf_acc(hb, t[S], yte) for t in tests]))
    return out


def _konfig(klein, sa, politur, nz, max_iter, hp):
    roh = json.dumps([klein, sa, politur, nz, max_iter, hp, list(S_GITTER), N_REAL, list(KOPF_ARME), L2_EXAKT,
                      ITER_EXAKT, "v2"])
    return hashlib.sha1(roh.encode()).hexdigest()[:12]


def _aufgabe(a):
    """(ds, seed, arm, hp, klein, sa, politur, nz, max_iter, ordner, konfig) -> Ergebnis (Wiederaufnahme je Seed/Arm)."""
    ds, seed, arm, hp, klein, sa, politur, nz, max_iter, ordner, konfig = a
    pfad = Path(ordner) / f"{ds}_{seed}_{arm}.json"
    if pfad.exists():
        z = json.loads(pfad.read_text())
        if z.get("konfig") == konfig:
            return z
    torch.set_num_threads(1)
    t0 = time.time()
    xtr, ytr, xte, yte = v1.daten(ds, seed, klein)
    iter_ex = ITER_EXAKT if not klein else max_iter
    z = {"ds": ds, "seed": seed, "arm": arm, "konfig": konfig}
    if arm == "pix":                      # Referenz fuer PK-1: I-038-Kopf auf Rohpixeln
        z["acc_inf"] = kopf_acc(kopf_fit(xtr, ytr, L2_EXAKT, iter_ex), xte, yte)
    else:
        xn_tr, xn_te = normiere(xtr), normiere(xte)
        ytr_n, yte_n = ytr.numpy(), yte.numpy()
        pi = np.bincount(ytr_n, minlength=10) / len(ytr_n)
        U, extra = v1._arm_U(arm, xtr, ytr, seed, sa, politur, nz)
        Ptr, Pte = merkmale(xn_tr, U), merkmale(xn_te, U)
        z.update(extra)
        if arm == "nk":
            rng = np.random.default_rng([seed, 4041])
            ytr_p = torch.tensor(rng.permutation(ytr_n))
            yte_p = rng.permutation(yte_n)
            z["I_C_test"] = info_C(klassenmittel(Pte.numpy().astype(np.float64), yte_p), pi)
            k = kopf_kurven2(Ptr, ytr_p, Pte, yte, hp["exakt_rob"], hp["bewusst"], (100,), seed, arm, max_iter, iter_ex)
            z["acc_bewusst"] = k["acc_bewusst"]
        else:
            z["chi_C"] = holevo_chi(xn_tr, ytr_n, pi)
            Pn, Pt = Ptr.numpy().astype(np.float64), Pte.numpy().astype(np.float64)
            z["I_C_train"] = info_C(klassenmittel(Pn, ytr_n), pi)
            z["I_C_test"] = info_C(klassenmittel(Pt, yte_n), pi)
            z.update(kopf_kurven2(Ptr, ytr, Pte, yte, hp["exakt_rob"], hp["bewusst"], S_GITTER, seed, arm, max_iter, iter_ex))
    z["sekunden"] = time.time() - t0
    Path(ordner).mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(z))        # Rohdaten sofort sichern
    return z


# ------------------------------------------------------------------ Auswertung
def _best_exakt(z_arm, S):
    return max(z_arm["acc_exakt"][str(S)], z_arm["acc_exakt_rob"][str(S)])


def auswertung(zeilen: list) -> dict:
    """zeilen: je Seed {'seed', 'arme': {arm: ...}, 'nk': {...}, 'pix': {'acc_inf'}}."""
    seeds = [z["seed"] for z in zeilen]
    arr = lambda arm, k: np.array([z["arme"][arm][k] for z in zeilen], float)
    sk = lambda arm, typ, S: np.array([z["arme"][arm][typ][str(S)] for z in zeilen], float)
    bestex = lambda arm, S: np.array([_best_exakt(z["arme"][arm], S) for z in zeilen], float)
    a = {"seeds": seeds, "arme": {}}
    for arm in ARME:
        I, chi = arr(arm, "I_C_test"), arr(arm, "chi_C")
        kb = {str(S): float(sk(arm, "acc_bewusst", S).mean()) for S in S_GITTER}
        s90 = s_ziel(kb)
        a["arme"][arm] = {"I": _stat(I), "chi": _stat(chi), "eta": _stat(I / chi), "acc_inf": _stat(arr(arm, "acc_inf")),
                          "acc_bewusst": {str(S): _stat(sk(arm, "acc_bewusst", S)) for S in S_GITTER},
                          "acc_exakt": {str(S): _stat(sk(arm, "acc_exakt", S)) for S in S_GITTER},
                          "acc_exakt_rob": {str(S): _stat(sk(arm, "acc_exakt_rob", S)) for S in S_GITTER},
                          "S90": s90, "S90_zensiert": s90 is None, "kurve_bewusst": kb,
                          "kollaps_seeds": _kollaps(sk(arm, "acc_bewusst", 100), seeds)}
    I = {arm: arr(arm, "I_C_test") for arm in ARME}
    inf = {arm: arr(arm, "acc_inf") for arm in ARME}
    # H1
    d1 = {"haar": I["id"] - I["haar"], "suche": I["id"] - I["suche"]}
    p1 = {k: wilcoxon_einseitig(v) for k, v in d1.items()}
    a["H1"] = {"diff": {k: _stat(v) for k, v in d1.items()}, "p_je": p1, "p": max(p1.values()),
               "anzahl_positiv": {k: int((v > 0).sum()) for k, v in d1.items()}}
    # H2 (Gegner = besserer der zwei exakten Koepfe); Zusatz: je einzelner exakter Kopf
    dS = {S: np.mean([sk(arm, "acc_bewusst", S) - bestex(arm, S) for arm in ARME], axis=0) for S in S_H2}
    p2 = {str(S): wilcoxon_einseitig(v) for S, v in dS.items()}
    dagg = np.mean([dS[S] for S in S_H2], axis=0)
    a["H2"] = {"diff_je_S": {str(S): _stat(v) for S, v in dS.items()}, "p_je_S": p2, "p": max(p2.values()),
               "diff_mittel_S": _stat(dagg), "p_gegenrichtung": wilcoxon_einseitig(-dagg),
               "beschreibend_gegen_exakt": {str(S): _stat(np.mean([sk(arm, "acc_bewusst", S) - sk(arm, "acc_exakt", S)
                                                                   for arm in ARME], axis=0)) for S in S_GITTER},
               "beschreibend_gegen_exakt_rob": {str(S): _stat(np.mean([sk(arm, "acc_bewusst", S) - sk(arm, "acc_exakt_rob", S)
                                                                      for arm in ARME], axis=0)) for S in S_GITTER},
               "beschreibend_grosse_S": {str(S): _stat(np.mean([sk(arm, "acc_bewusst", S) - bestex(arm, S)
                                                               for arm in ARME], axis=0)) for S in (300, 1000)}}
    # H3
    eta = [a["arme"][arm]["eta"]["mittel"] for arm in ARME]
    s90 = [a["arme"][arm]["S90"] if a["arme"][arm]["S90"] is not None else S_ZENS for arm in ARME]
    rho, p3 = spearman_perm(eta, s90)
    a["H3"] = {"eta": dict(zip(ARME, eta)), "S90": dict(zip(ARME, s90)),
               "zensiert": [arm for arm in ARME if a["arme"][arm]["S90_zensiert"]], "rho": rho, "p": p3}
    # H4
    dg, di = inf["haar"] - inf["id"], I["id"] - I["haar"]
    pg, pi_ = wilcoxon_einseitig(dg), wilcoxon_einseitig(di)
    a["H4"] = {"diff_genauigkeit_haar_minus_id": _stat(dg), "diff_information_id_minus_haar": _stat(di),
               "p_genauigkeit": pg, "p_information": pi_, "p": max(pg, pi_),
               "anzahl_positiv": {"genauigkeit": int((dg > 0).sum()), "information": int((di > 0).sum())}}
    ps = {"H1": a["H1"]["p"], "H2": a["H2"]["p"], "H3": 1.0 if p3 is None else p3, "H4": a["H4"]["p"]}
    a["holm"] = holm(ps)
    a["kontrollen"] = pruefe_vorbedingungen(a, zeilen)
    return a


def pruefe_vorbedingungen(a: dict, zeilen: list) -> dict:
    k = {}
    inf = a["arme"]["id"]["acc_inf"]["mittel"]
    pix = float(np.mean([z["pix"]["acc_inf"] for z in zeilen]))
    schwelle = pix - PK1_REL
    k["PK-1"] = ("PK_OK" if inf >= schwelle else
                 f"HU PK_VERSAGT (Identitaet exakt {inf:.3f} < Pixel-Softmax {pix:.3f} - {PK1_REL})") + \
        f" [id {inf:.3f}, pix {pix:.3f}, Schwelle {schwelle:.3f}]"
    e10, e1000 = (a["arme"]["id"]["acc_exakt"][s]["mittel"] for s in ("10", "1000"))
    k["PK-2"] = ("PK_OK" if e1000 - e10 >= PK2_ABSTAND else
                 f"HU PK_VERSAGT (exakter Kopf S=10 nur {e1000 - e10:.3f} unter S=1000)") + f" [{e1000 - e10:.3f}]"
    nkI = float(np.mean([z["nk"]["I_C_test"] for z in zeilen]))
    nkA = float(np.mean([z["nk"]["acc_bewusst"]["100"] for z in zeilen]))
    k["NK"] = ("NK_OK" if nkI < NK_I_MAX and nkA <= NK_ACC_MAX else
               f"HU NK_VERSAGT (I {nkI:.3f} Bit, Genauigkeit {nkA:.3f})") + f" [I {nkI:.4f}, acc {nkA:.3f}]"
    k["instabil"] = {arm: a["arme"][arm]["kollaps_seeds"] for arm in ARME if len(a["arme"][arm]["kollaps_seeds"]) >= 2}
    k["werte"] = {"pk1_id": inf, "pk1_pix": pix, "pk1_schwelle": schwelle, "pk2_abstand": e1000 - e10,
                  "nk_I": nkI, "nk_acc": nkA}
    return k


def urteil(a: dict) -> dict:
    """Urteile als Strings; ALLE Kriterien stehen hier (vorregistriert)."""
    k, hp = a["kontrollen"], a["holm"]
    u = {"PK-1": k["PK-1"], "PK-2": k["PK-2"], "NK": k["NK"]}
    nk_ok = k["NK"].startswith("NK_OK")
    pk1_ok = k["PK-1"].startswith("PK_OK")
    pk_ok = pk1_ok and k["PK-2"].startswith("PK_OK")
    instab = bool(k["instabil"])
    n = len(a["seeds"])
    # H1
    h1 = a["H1"]
    mp = [h1["diff"][x]["mittel"] for x in ("haar", "suche")]
    txt = (f"I(id)-I(haar) {mp[0]:+.3f} Bit, I(id)-I(suche) {mp[1]:+.3f} Bit, Holm-p {hp['H1']:.4f}, "
           f"positiv {h1['anzahl_positiv']['haar']}/{n} bzw. {h1['anzahl_positiv']['suche']}/{n}")
    if not nk_ok:
        u["H1"] = "HU NK_VERSAGT: Informationsmass faellt die Negativkontrolle durch"
    elif min(mp) > 0 and hp["H1"] < ALPHA:
        u["H1"] = "H1 IDENTITAET_BESTE: " + txt
    elif min(mp) <= 0:
        u["H1"] = "H0 NICHT_UEBERLEGEN: " + txt
    else:
        u["H1"] = "HU NICHT_ENTSCHIEDEN: " + txt
    # H2
    h2 = a["H2"]
    m = [h2["diff_je_S"][str(S)]["mittel"] for S in S_H2]
    txt = (f"Differenz bewusst - bester exakter Kopf je S=10/30/100: {m[0]:+.3f}/{m[1]:+.3f}/{m[2]:+.3f}, "
           f"Holm-p {hp['H2']:.4f}, Gegenrichtung p {h2['p_gegenrichtung']:.4f}")
    if not nk_ok or not pk_ok:
        u["H2"] = "HU PK_VERSAGT: Vorbedingung nicht erfuellt"
    elif instab:
        u["H2"] = f"HU INSTABIL: {k['instabil']}"
    elif min(m) > 0 and hp["H2"] < ALPHA:
        u["H2"] = "H1 SCHUSSBEWUSST_BESSER: " + txt
    elif h2["p_gegenrichtung"] < ALPHA and h2["diff_mittel_S"]["mittel"] < 0:
        u["H2"] = "H0 EXAKT_BESSER: " + txt
    elif h2["diff_mittel_S"]["mittel"] <= 0:
        u["H2"] = "H0 KEIN_VORTEIL: " + txt
    else:
        u["H2"] = "HU NICHT_ENTSCHIEDEN: " + txt
    # H3
    h3 = a["H3"]
    txt = (f"rho {h3['rho'] if h3['rho'] is None else round(h3['rho'], 3)}, p {h3['p'] if h3['p'] is None else round(h3['p'], 4)}, "
           f"Holm-p {hp['H3']:.4f}, zensiert {h3['zensiert']}")
    if not nk_ok or not pk_ok:
        u["H3"] = "HU PK_VERSAGT: Vorbedingung nicht erfuellt"
    elif instab:
        u["H3"] = f"HU INSTABIL: {k['instabil']}"
    elif len(h3["zensiert"]) >= 3 or h3["rho"] is None:
        u["H3"] = "HU ZENSIERT: " + txt
    elif h3["rho"] < 0 and hp["H3"] < ALPHA:
        u["H3"] = "H1 ETA_SAGT_VORAUS: " + txt
    elif h3["rho"] >= RHO_NEIN:
        u["H3"] = "H0 ETA_NICHT_PROGNOSTISCH: " + txt
    else:
        u["H3"] = "HU NICHT_ENTSCHIEDEN: " + txt
    # H4 (braucht NK und PK-1: Kopf-Maschinerie gesund)
    h4 = a["H4"]
    mg, mi = h4["diff_genauigkeit_haar_minus_id"]["mittel"], h4["diff_information_id_minus_haar"]["mittel"]
    txt = (f"Genauigkeit Haar-id {mg:+.4f}, Information id-Haar {mi:+.3f} Bit, Holm-p {hp['H4']:.4f}, "
           f"positiv {h4['anzahl_positiv']['genauigkeit']}/{n} bzw. {h4['anzahl_positiv']['information']}/{n}")
    if not nk_ok or not pk1_ok:
        u["H4"] = "HU PK_VERSAGT: Vorbedingung nicht erfuellt"
    elif mg > 0 and mi > 0 and hp["H4"] < ALPHA:
        u["H4"] = "H1 ZIELKONFLIKT: " + txt
    elif mg <= 0:
        u["H4"] = "H0 KEIN_KONFLIKT_GENAUIGKEIT: " + txt
    elif mi <= 0:
        u["H4"] = "H0 KEIN_KONFLIKT_INFORMATION: " + txt
    else:
        u["H4"] = "HU NICHT_ENTSCHIEDEN: " + txt
    return u


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, seeds=SEEDS, tuning_seed=TUNING_SEED, ordner=None, fashion=False, prozesse=None, **kw):
    if schnell:
        seeds, klein, sa, politur, nz, max_iter = SCHNELL_SEEDS, True, 60, 20, 5, 30
        ordner = ordner or gem.ERGEBNISSE / "_zwischen_I-040-v2-schnell"
    else:
        klein, sa, politur, nz, max_iter = False, v1c.SA_SCHRITTE, v1c.POLITUR_MAX, v1c.N_ZUFALL, MAX_ITER
        if tuple(seeds) != SEEDS or tuning_seed != TUNING_SEED:
            raise ValueError("Seeds sind vorregistriert: 2150-2159, Tuning 2198")
    ordner = Path(ordner or gem.ERGEBNISSE / "_zwischen_I-040-v2")
    seeds = list(seeds)
    if not schnell and set(seeds) & {tuning_seed}:
        raise ValueError("Mess- und Tuning-Seed ueberschneiden sich")
    ordner.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dsets = ["mnist"] + (["fashion_mnist"] if fashion else [])
    # Stufe 0: L2 fuer exakt_rob und bewusst auf dem Tuning-Seed (Validierung, nie Test); einmal, dann Datei
    tp = ordner / "tuning.json"
    kt = hashlib.sha1(json.dumps([klein, max_iter, list(L2_GITTER), tuning_seed, "v2"]).encode()).hexdigest()[:12]
    tun = json.loads(tp.read_text()) if tp.exists() else None
    if not tun or tun.get("konfig") != kt:
        gitter, roh = list(L2_GITTER), {arm: {"exakt": {}, "bewusst": {}} for arm in v1.TUNING_ARME}
        for _runde in range(4):
            neu = [g for g in gitter if str(g) not in roh[v1.TUNING_ARME[0]]["exakt"]]
            res = gem.parallel(v1._tune_aufgabe, [("mnist", tuning_seed, arm, klein, max_iter, neu) for arm in v1.TUNING_ARME],
                               prozesse=prozesse)
            for arm, o in res:
                for typ in o:
                    roh[arm][typ].update(o[typ])
            w = v1.waehle_l2(roh)
            ext = {w["am_rand"][t] for t in w["am_rand"]} - {None}
            if not ext:
                break
            if "unten" in ext:
                gitter = [min(gitter) / 3.0] + gitter
            if "oben" in ext:
                gitter = gitter + [max(gitter) * 3.0]
        tun = {"konfig": kt, "roh": roh, "wahl": w, "tuning_seed": tuning_seed}
        tp.write_text(json.dumps(tun, indent=1))
    hp = {"exakt_rob": tun["wahl"]["exakt"], "bewusst": tun["wahl"]["bewusst"]}
    print(f"  L2-Wahl (Validierung, vor Test): exakt fest {L2_EXAKT}, {hp}  am Rand: {tun['wahl']['am_rand']}", flush=True)
    konfig = _konfig(klein, sa, politur, nz, max_iter, hp)
    aufg = [(ds, s, arm, hp, klein, sa, politur, nz, max_iter, str(ordner), konfig)
            for arm in ("suche", "id", "haar", "ff2", "bent4", "nk", "pix") for ds in dsets for s in seeds]
    erg = gem.parallel(_aufgabe, aufg, prozesse=prozesse)
    tab = {(z["ds"], z["seed"], z["arm"]): z for z in erg}
    e = {"abteilung": "clifford", "modul": "holevo_wirkungsgrad_v2", "agenda": "I-040-v2", "hypothesen": HYPOTHESEN,
         "folgefragen": FOLGEFRAGEN, "seeds": seeds, "tuning_seed": tuning_seed, "tuning": tun,
         "l2": {"exakt_fest": L2_EXAKT, **hp},
         "einstellungen": {"klein": klein, "sa": sa, "politur": politur, "n_zufall": nz, "max_iter": max_iter,
                           "iter_exakt": ITER_EXAKT, "S": list(S_GITTER), "n_real": N_REAL, "pk1_rel": PK1_REL},
         "zeilen": {}, "auswertung": {}}
    for ds in dsets:
        zeilen = [{"seed": s, "arme": {arm: tab[(ds, s, arm)] for arm in ARME}, "nk": tab[(ds, s, "nk")],
                   "pix": tab[(ds, s, "pix")]} for s in seeds]
        e["zeilen"][ds] = zeilen
        if schnell and len(seeds) < 2:
            e["auswertung"][ds] = "Rauchtest: 1 Seed, keine Statistik"
            continue
        e["auswertung"][ds] = auswertung(zeilen)
    e["urteil"] = urteil(e["auswertung"]["mnist"]) if not schnell else {"rauchtest": "kein Urteil (1 Seed)"}
    if fashion and not schnell:
        e["fashion_beschreibend"] = e["auswertung"]["fashion_mnist"]
    e["sekunden"] = time.time() - t0
    return e


# ------------------------------------------------------------------ Vortest
def _synth(par: dict, n=10, seed=0) -> list:
    rng = np.random.default_rng(seed)
    kurve = v1._kurve
    z = []
    for i in range(n):
        arme = {}
        for arm in ARME:
            kb = kurve(par["s90"][arm])
            row = {"I_C_test": par["I"][arm] + par["sdI"] * rng.standard_normal(), "chi_C": par["chi"],
                   "acc_inf": par["acc_inf"][arm] + par["sd_acc"] * rng.standard_normal(),
                   "acc_bewusst": {}, "acc_exakt": {}, "acc_exakt_rob": {}}
            for S in S_GITTER:
                b = kb(S) + par["sd_acc"] * rng.standard_normal()
                dl = par["delta"][S] + par["sd_delta"] * rng.standard_normal()
                row["acc_bewusst"][str(S)] = b
                row["acc_exakt"][str(S)] = b - dl
                row["acc_exakt_rob"][str(S)] = b - dl - par.get("rob_minus", 0.01)   # rob meist schwaecher
            if arm in par.get("kollaps", {}) and i in par["kollaps"][arm]:
                row["acc_bewusst"]["100"] -= 0.2
            arme[arm] = row
        z.append({"seed": 2150 + i, "arme": arme,
                  "nk": {"I_C_test": par["nk_I"], "acc_bewusst": {"100": par["nk_acc"]}},
                  "pix": {"acc_inf": par["pix"] + par["sd_acc"] * rng.standard_normal()}})
    return z


def _basis(**ueber) -> dict:
    p = {"I": {"id": 0.36, "bent4": 0.307, "haar": 0.165, "ff2": 0.163, "suche": 0.161}, "sdI": 0.0005, "chi": 0.98,
         "acc_inf": {"id": 0.90, "haar": 0.937, "ff2": 0.937, "bent4": 0.92, "suche": 0.93}, "pix": 0.909, "sd_acc": 0.002,
         "s90": {"id": 100, "bent4": 150, "ff2": 400, "haar": 450, "suche": 500},
         "delta": {10: 0.10, 30: 0.06, 100: 0.03, 300: 0.01, 1000: 0.0}, "sd_delta": 0.004, "nk_I": 0.004, "nk_acc": 0.10}
    p.update(ueber)
    return p


def vortest() -> bool:
    ok = True

    def fehler(t):
        nonlocal ok
        ok = False
        print("  VORTEST FEHLER:", t)

    if set(SEEDS) & BELEGT or TUNING_SEED in BELEGT or TUNING_SEED in SEEDS or len(SEEDS) != 10:
        fehler("Seeds nicht frisch/disjunkt")
    if not (L2_EXAKT == 3e-3 and ITER_EXAKT == 500 and PK1_REL == 0.02):
        fehler("Kopf-/Schwellenkonstanten veraendert")

    def erwarte(name, par, urteile: dict, seed=0):
        a = auswertung(_synth(par, seed=seed))
        u = urteil(a)
        for h, marke in urteile.items():
            if not u[h].startswith(marke):
                fehler(f"{name}: {h} = {u[h][:80]!r}, erwartet {marke!r}")
        return u, a

    erwarte("Basis", _basis(), {"PK-1": "PK_OK", "PK-2": "PK_OK", "NK": "NK_OK", "H1": "H1 IDENTITAET_BESTE",
                                "H2": "H1 SCHUSSBEWUSST_BESSER", "H3": "H1 ETA_SAGT_VORAUS", "H4": "H1 ZIELKONFLIKT"})
    # PK-1 relativ: id 0,868 gegen pix 0,909 -> faellt; id 0,895 -> besteht (Grenze pix - 0,02 = 0,889)
    ai = _basis()["acc_inf"]
    erwarte("PK-1 faellt", _basis(acc_inf={**ai, "id": 0.868}), {"PK-1": "HU PK_VERSAGT", "H2": "HU PK_VERSAGT",
                                                               "H3": "HU PK_VERSAGT", "H4": "HU PK_VERSAGT",
                                                               "H1": "H1 IDENTITAET_BESTE"})
    erwarte("PK-1 relativ besteht (absolut waere 0,88 gefallen bei pix 0,88)", _basis(acc_inf={**ai, "id": 0.865}, pix=0.880),
            {"PK-1": "PK_OK"})
    erwarte("PK-1 Grenze innen", _basis(acc_inf={**ai, "id": 0.892}), {"PK-1": "PK_OK"})
    # PK-2 erzwingen
    flach = _synth(_basis(delta={10: 0.0, 30: 0.0, 100: 0.0, 300: 0.0, 1000: 0.0}, rob_minus=0.0))
    for r in flach:
        for arm in ARME:
            r["arme"][arm]["acc_exakt"]["10"] = r["arme"][arm]["acc_exakt"]["1000"] - 0.01
    uu = urteil(auswertung(flach))
    if not (uu["PK-2"].startswith("HU PK_VERSAGT") and uu["H2"].startswith("HU PK_VERSAGT") and uu["H4"].startswith("H")
            and not uu["H4"].startswith("HU PK")):
        fehler("PK-2 erzwingt H2-HU nicht bzw. H4 faelschlich betroffen")
    erwarte("NK faellt (Information)", _basis(nk_I=0.2), {"NK": "HU NK_VERSAGT", "H1": "HU NK_VERSAGT", "H2": "HU PK_VERSAGT",
                                                          "H3": "HU PK_VERSAGT", "H4": "HU PK_VERSAGT"})
    erwarte("NK faellt (Genauigkeit)", _basis(nk_acc=0.5), {"NK": "HU NK_VERSAGT"})
    # H1
    gleich = {"id": 0.165, "bent4": 0.2, "haar": 0.163, "ff2": 0.163, "suche": 0.161}
    erwarte("H1 H0", _basis(I={**_basis()["I"], "id": 0.15}), {"H1": "H0 NICHT_UEBERLEGEN"})
    erwarte("H1 HU", _basis(I={**gleich, "id": 0.1635}, sdI=0.02), {"H1": "HU NICHT_ENTSCHIEDEN"}, seed=0)
    # H2
    erwarte("H2 H0 EXAKT_BESSER", _basis(delta={10: -0.03, 30: -0.03, 100: -0.03, 300: 0.0, 1000: 0.0}, rob_minus=-0.01),
            {"H2": "H0 EXAKT_BESSER"})
    erwarte("H2 H0 KEIN_VORTEIL", _basis(delta={10: 0.0, 30: 0.0, 100: 0.0, 300: 0.0, 1000: 0.0}, sd_delta=0.0, sd_acc=0.0,
                                         rob_minus=0.0), {"H2": "H0 KEIN_VORTEIL"})
    erwarte("H2 HU gemischt", _basis(delta={10: 0.10, 30: 0.06, 100: -0.01, 300: 0.0, 1000: 0.0}, rob_minus=0.0),
            {"H2": "HU NICHT_ENTSCHIEDEN"})
    erwarte("H2 HU verrauscht", _basis(delta={10: 0.003, 30: 0.003, 100: 0.003, 300: 0.0, 1000: 0.0}, sd_delta=0.03),
            {"H2": "HU NICHT_ENTSCHIEDEN"}, seed=5)
    # H2: der staerkere der zwei exakten Koepfe zaehlt (rob viel besser als exakt -> kein Vorteil)
    u, _ = erwarte("H2 Gegner = Maximum", _basis(rob_minus=-0.2), {"H2": "H0 EXAKT_BESSER"})
    # H3
    erwarte("H3 H0", _basis(s90={"id": 500, "bent4": 450, "ff2": 400, "haar": 150, "suche": 100}), {"H3": "H0 ETA_NICHT_PROGNOSTISCH"})
    erwarte("H3 HU ZENSIERT", _basis(s90={"id": None, "bent4": None, "ff2": None, "haar": 450, "suche": 500}), {"H3": "HU ZENSIERT"})
    teil = {"id": 100, "bent4": 400, "ff2": 150, "haar": 450, "suche": 500}
    a_t = auswertung(_synth(_basis(s90=teil), seed=0))
    if not (-0.95 < a_t["H3"]["rho"] < -0.2) or not urteil(a_t)["H3"].startswith("HU NICHT_ENTSCHIEDEN"):
        fehler("H3 mittlere Korrelation erzwingt HU nicht")
    erwarte("Instabilitaet", _basis(kollaps={"haar": [1, 4]}), {"H2": "HU INSTABIL", "H3": "HU INSTABIL", "H4": "H1 ZIELKONFLIKT"})
    u, _ = erwarte("ein Kollaps-Seed", _basis(kollaps={"haar": [1]}), {})
    if u["H2"].startswith("HU INSTABIL"):
        fehler("ein Kollaps-Seed darf nicht reichen")
    # H4-Zweige
    erwarte("H4 H0 Genauigkeit", _basis(acc_inf={**ai, "haar": 0.88}), {"H4": "H0 KEIN_KONFLIKT_GENAUIGKEIT"})
    erwarte("H4 H0 Information", _basis(I={**_basis()["I"], "id": 0.15}), {"H4": "H0 KEIN_KONFLIKT_INFORMATION"})
    erwarte("H4 HU", _basis(acc_inf={**ai, "haar": 0.9005}, sd_acc=0.01), {"H4": "HU NICHT_ENTSCHIEDEN"}, seed=0)
    # Mechanik End-to-End auf Mini-Daten (Tuning-Seed, Temp-Ordner), inkl. Wiederaufnahme und pix
    try:
        import tempfile
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as td:
            hp = {"exakt_rob": 3e-2, "bewusst": 1e-2}
            kf = _konfig(True, 10, 5, 3, 10, hp)
            for arm in ("id", "pix"):
                r1 = _aufgabe(("mnist", TUNING_SEED, arm, hp, True, 10, 5, 3, 10, td, kf))
                r2 = _aufgabe(("mnist", TUNING_SEED, arm, hp, True, 10, 5, 3, 10, td, kf))
                if r1 != r2 or not (0.0 <= r1["acc_inf"] <= 1.0):
                    fehler(f"Aufgabe/Wiederaufnahme {arm}")
            if set(r1["acc_inf"] if False else _aufgabe(("mnist", TUNING_SEED, "id", hp, True, 10, 5, 3, 10, td, kf))["acc_exakt_rob"]) != set(map(str, S_GITTER)):
                fehler("acc_exakt_rob unvollstaendig")
    except Exception as ex:  # pragma: no cover
        fehler(f"Aufgabe Ausnahme {ex!r}")
    return ok


if __name__ == "__main__":
    schnell = "--schnell" in sys.argv
    if "--referenz" in sys.argv:           # Tuning-Seed-Referenz (nur Kopf, Volldaten, kein Mess-Seed)
        assert TUNING_SEED not in SEEDS
        xtr, ytr, xte, yte = v1.daten("mnist", TUNING_SEED)
        U = torch.eye(256)
        he = kopf_fit(merkmale(normiere(xtr), U), ytr, L2_EXAKT, ITER_EXAKT)
        h2 = kopf_fit(merkmale(normiere(xtr), U), ytr, 0.09, 200)
        hp_ = kopf_fit(xtr, ytr, L2_EXAKT, ITER_EXAKT)
        Pte = merkmale(normiere(xte), U)
        print(json.dumps({"seed": TUNING_SEED, "id_exakt_I038Kopf": kopf_acc(he, Pte, yte), "id_L2_0.09_iter200": kopf_acc(h2, Pte, yte),
                          "pixel_softmax": kopf_acc(hp_, xte, yte)}))
        sys.exit(0)
    bestanden = vortest()
    print("Vortest:", "bestanden" if bestanden else "DURCHGEFALLEN", flush=True)
    if not bestanden:
        sys.exit(1)
    e = lauf(schnell=schnell, fashion="--fashion" in sys.argv)
    print(gem.speichere("I-040-v2" + ("-schnell" if schnell else ""), e))
    for h, t in e["urteil"].items():
        print(f"  {h}: {t}")
