"""Abteilung Struktur -- I-044-v2 „Strukturkopf gegen faire Köpfe“ (CIFAR-10-CNN).
(Folge von I-044/I-044-r. Wirkung klassisch (I-038); formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.)

ANLASS (Durchsicht 07.10., vor jeder Zahl dieses Laufs): In I-044-r lag der Strukturkopf Ks 74,33 % vor dem dichten Kopf Kd 72,44 %
(+1,89 pp). Der dichte Kopf (1024->4096 ReLU, 4,2 Mio Parameter, ohne Dropout/Weight Decay, 4 Epochen) ist aber ein schwacher
Gegner: der Arm Kcd (1x1-Conv + LINEARER Kopf, ~10 k Kopfparameter) erreichte schon 73,70 %. Jurorenfrage: „Schlägt die Struktur
einen ordentlichen Kopf oder nur einen schlecht regularisierten?“ Außerdem ist die Idee (Fastfood statt FC-Kopf im CNN) bekannt:
Deep Fried Convnets (Yang u. a., ICCV 2015). Beitrag hier nur: kontrollierter, fairer Vergleich.

ARME (CNN-Rumpf identisch zu struktur_tief.Netz Teil 2; 1.024 Merkmale nach Pooling):
  Ks       Fastfood 1024->4096 (trainierte Diagonalen) + z² + LayerNorm + linear 4096->10   (wie I-044, ≈ 49 k Kopfparameter)
  Kd       dicht 1024->4096 + ReLU + LayerNorm + linear (wie I-044, ≈ 4,2 Mio)              (Brücke zu I-044-r, nur beschreibend)
  Klin     KEINE breite Schicht: LayerNorm + linear 1024->10 (≈ 10 k)                        (fairer Gegner 1)
  Kreg     wie Kd, aber Dropout 0,5 nach der breiten Schicht + AdamW Weight Decay 5e-4      (fairer Gegner 2: regularisiert)
  Kklein   dicht 1024->48 + ReLU + LayerNorm + linear 48->10 (≈ 49,7 k = Kopfbudget von Ks)  (fairer Gegner 3: gleiches Budget)
TRAINING: wie I-044 (Adam bzw. AdamW, Batch 256, Kosinus, 4 Epochen, Test nur am Ende). Lernrate: Ks und Kd ÜBERNOMMEN aus I-044
  (Tuning-Seed 2399); Klin, Kreg, Kklein NEU auf demselben Tuning-Seed 2399 mit demselben Verfahren (gittersuche_stabil, Gitter
  1e-3/3e-3/1e-2, Faktor 3, beidseitig verlängerbar, Validierung = letzte 10 % Training). Nichtlinearität Ks = z² (aus I-044).
SEEDS: Mess 2330-2344 (15, frisch), Tuning 2399.
HYPOTHESEN (Holm über H1a, H1b, H1c, H2; gepaarter einseitiger t-Test, n = 15; Kriterien unten im Code):
  H1a Ks >= Klin   - 0,5 pp (Nichtunterlegenheit)    H1b Ks >= Kreg - 0,5 pp    H1c Ks >= Kklein - 0,5 pp
  H2  Ks - Klin > 0 (Superiorität: bringt die breite Strukturschicht überhaupt etwas gegenüber KEINER breiten Schicht?)
  Je H1x: NICHT_UNTERLEGEN | H0 (sicher um mehr als 0,5 pp unterlegen) | HU.  H2: STRUKTUR_HILFT | H0 (Ks sicher schlechter) | HU.
  NK: Ks mit permutierten Labels, Mittel in (0,05; 0,15), sonst alles HU. Kollaps (> 5 pp unter Median) in einem Arm -> dessen Hypothesen HU.
  B (beschreibend): alle Mittel, Ks - Kd (Brücke I-044-r), Kopfparameter je Arm, gepoolt nichts.
EHRLICHE ERWARTUNG (vorab): H1a/H1c wahrscheinlich NICHT_UNTERLEGEN, H2 klein positiv oder HU; der Vorsprung gegen faire Köpfe
  liegt eher bei 0-1 pp als bei 1,9 pp. Jeder Ausgang ist berichtbar.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abteilungen import struktur_tief as st
from abteilungen.clifford_suche import holm
from abteilungen.struktur_gross import _arr, _f, _rechne, _stat, daten, kollaps_seeds
from labor.gemeinsam import ERGEBNISSE, gittersuche_stabil, parallel, zusammenfassung

KARTE = "I-044-v2"
SEEDS = tuple(range(2330, 2345))
TUNING_SEED = st.TUNING_SEED
ARME = ("Ks", "Kd", "Klin", "Kreg", "Kklein")
NEU_GETUNT = ("Klin", "Kreg", "Kklein")
TUNING_UEBERNOMMEN = {"Ks": 0.003, "Kd": 0.003}          # aus I-044 (= st.TUNING_R)
EPOCHEN, NK_EPOCHEN = 4, 2
NI = 0.005
ALPHA = 0.05
NK_BAND = st.NK_BAND
KOLLAPS = st.KOLLAPS
DROPOUT, WD = 0.5, 5e-4
N_KLEIN = 48
FAIR = {"H1a": "Klin", "H1b": "Kreg", "H1c": "Kklein"}
_BELEGT = st._BELEGT | set(st.SEEDS) | set(st.SEEDS_R) | {TUNING_SEED}

HYPOTHESEN = {
    "H1a": "Ks >= Klin - 0,5 pp (einseitig gepaart t, Holm über H1a-c, H2). NICHT_UNTERLEGEN | H0 | HU.",
    "H1b": "Ks >= Kreg (dichter Kopf mit Dropout 0,5 + Weight Decay) - 0,5 pp. NICHT_UNTERLEGEN | H0 | HU.",
    "H1c": "Ks >= Kklein (dichter Kopf gleichen Budgets, 1024->48->10) - 0,5 pp. NICHT_UNTERLEGEN | H0 | HU.",
    "H2": "Ks - Klin > 0 (breite Strukturschicht hilft gegenüber keiner). STRUKTUR_HILFT | H0 | HU.",
    "NK": f"Ks mit permutierten Labels, Mittel in {NK_BAND}. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: Mittel aller Arme, Ks - Kd, Kopfparameter, Lernraten.",
}
FOLGEFRAGEN = {
    "H2 HU/H0": "Der I-044-Vorsprung kam vom schwachen dichten Kopf; Kernaussage auf 'gleich gut mit weniger Parametern' zurücknehmen.",
    "H1 alle NICHT_UNTERLEGEN + H2 STRUKTUR_HILFT": "Strukturkopf trägt auch gegen faire Köpfe; Aussage hält.",
}


# ------------------------------------------------------------------ Modell
class KopfNetz(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        basis = st.Netz("Ks" if arm == "Ks" else "Kd", seed, {"Ks": ("z2",)})
        self.conv = basis.conv
        if arm in ("Ks", "Kd", "Kreg"):
            self.breit, self.kopf = basis.breit, basis.kopf
        elif arm == "Kklein":
            self.breit, self.kopf = nn.Linear(1024, N_KLEIN), nn.Linear(N_KLEIN, 10)
        else:                                               # Klin
            self.breit, self.kopf = nn.Identity(), nn.Linear(1024, 10)
        self.drop = nn.Dropout(DROPOUT) if arm == "Kreg" else nn.Identity()

    def forward(self, x):
        z = F.max_pool2d(self.conv(x.reshape(-1, 3, 32, 32)), 2).flatten(1)
        if self.arm == "Kklein":
            z = self.breit(z)
            z = F.layer_norm(F.relu(z), z.shape[-1:])
        elif self.arm == "Klin":
            z = F.layer_norm(z, z.shape[-1:])
        else:
            z = self.breit(z)
        return self.kopf(self.drop(z))


def param_kopf(netz):
    return int(sum(p.numel() for m in (netz.breit, netz.kopf) for p in m.parameters() if p.requires_grad))


def trainiere(arm, lr, seed, epochen, klein=False, val=False, permutiert=False):
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = daten("cifar10", klein)
    if val:
        nv = len(ytr) // 10
        xtr, ytr, xte, yte = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 5 + 100 * ord(arm[0]) + ord(arm[-1])))]
    torch.manual_seed(seed)
    netz = KopfNetz(arm, seed)
    ps = [p for p in netz.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=WD) if arm == "Kreg" else torch.optim.Adam(ps, lr=lr)
    n = len(ytr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochen * -(-n // st.BATCH))
    netz.train()
    for _ in range(epochen):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, st.BATCH):
            j = perm[i:i + st.BATCH]
            l = F.cross_entropy(netz(xtr[j]), ytr[j])
            if not torch.isfinite(l):
                return {"acc": 0.1, "instabil": True}
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
    netz.eval()
    with torch.no_grad():
        richtig = sum(int((netz(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum()) for i in range(0, len(yte), 1000))
    return {"acc": richtig / len(yte), "instabil": False}


def _lr_arbeiter(a):
    arm, s, lr, ep, klein = a
    return trainiere(arm, lr, s, ep, klein=klein, val=True)


def tuning(ep, klein, tseed):
    info, _ = gittersuche_stabil(list(NEU_GETUNT), lambda k, lr: [(k, tseed, lr, ep, klein)],
                                 lambda aufg: parallel(_lr_arbeiter, aufg), gitter=st.LR_GITTER, faktor=st.LR_FAKTOR, kollaps=KOLLAPS)
    return {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _arbeiter(a):
    teil, arm, s, lr, ep, klein = a
    t0 = time.time()
    r = trainiere(arm, lr, s, ep, klein=klein, permutiert=(teil == "nk"))
    return f"einheit|{teil}|{arm}|{s}", {"teil": teil, "arm": arm, "seed": s, "acc": r["acc"], "instabil": r["instabil"],
                                         "sekunden": time.time() - t0}


def _schl(a):
    return f"einheit|{a[0]}|{a[1]}|{a[2]}"


# ------------------------------------------------------------------ Auswertung und Urteile (Kriterien im Code)
def auswertung(zeilen: list, params: dict) -> dict:
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] == "t"})
    acc = {}
    for z in zeilen:
        acc.setdefault((z["teil"], z["arm"]), {})[z["seed"]] = z["acc"]
    zl = [{"seed": s, **{a: acc[("t", a)][s] for a in ARME if ("t", a) in acc and s in acc[("t", a)]}} for s in seeds]
    r = {"arme": {}, "kollaps": {}, "negativ": {}, "diff": {}, "p_holm": {}, "parameter": params, "beschreibend": {}}
    vorhanden = [a for a in ARME if zl and all(a in z for z in zl)]
    for a in vorhanden:
        r["arme"][a] = zusammenfassung(_arr(zl, a))
        r["kollaps"][a] = kollaps_seeds(zl, a)
    perm = [v for (t, a), d in acc.items() if t == "nk" for v in d.values()]
    r["negativ"] = {"mittel": float(np.mean(perm)), "n": len(perm)} if perm else {}
    d = lambda x, y: _arr(zl, x) - _arr(zl, y)
    if "Ks" in vorhanden:
        for h, g in FAIR.items():
            if g in vorhanden:
                r["diff"][h] = _stat(d("Ks", g), versatz=NI)
        if "Klin" in vorhanden:
            r["diff"]["H2"] = _stat(d("Ks", "Klin"))
        for y in ("Kd", "Kreg", "Kklein"):
            if y in vorhanden:
                r["beschreibend"][f"Ks - {y}"] = _stat(d("Ks", y))
        for y in ("Kd", "Kreg", "Kklein"):
            if y in vorhanden and "Klin" in vorhanden:
                r["beschreibend"][f"{y} - Klin"] = _stat(d(y, "Klin"))
    prim = {h: r["diff"][h].get("p_gr", 1.0) for h in ("H1a", "H1b", "H1c", "H2") if h in r["diff"]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def _nk(r):
    n = r["negativ"]
    return "BESTANDEN" if n and NK_BAND[0] <= n["mittel"] <= NK_BAND[1] else "DURCHGEFALLEN"


def _zahlen(r, h):
    x = r["diff"][h]
    return f"d {_f(x['mittel_pp'])} pp, 90-%-KI [{_f(x['ki_pp'][0])}; {_f(x['ki_pp'][1])}], n {x['n']}, p_Holm {r['p_holm'][h]:.3g}"


def _vorab(r, nk, h, arme):
    if h not in r["diff"] or "mittel_pp" not in r["diff"][h]:
        return "HU (Arme fehlen oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    k = sorted({s for a in arme for s in r["kollaps"].get(a, [])})
    return f"HU (Instabilitaet/Kollaps in Seeds {k})" if k else None


def _urteil(r, nk, h, arme, ja):
    v = _vorab(r, nk, h, arme)
    if v:
        return v
    z = _zahlen(r, h)
    if r["p_holm"][h] < ALPHA:
        return f"{ja} ({z})"
    if r["diff"][h]["p_kl"] < ALPHA / 2:
        return f"H0 ({z})"
    return f"HU ({z})"


def urteil(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutierte Labels Ks: {r['negativ'].get('mittel', float('nan')):.3f}; Band {NK_BAND})"}
    for h, g in FAIR.items():
        u[h] = _urteil(r, nk, h, ("Ks", g), "NICHT_UNTERLEGEN")
    u["H2"] = _urteil(r, nk, "H2", ("Ks", "Klin"), "STRUKTUR_HILFT")
    g = lambda a: f"{r['arme'][a]['mittel'] * 100:.2f}".replace(".", ",") if a in r["arme"] else "n/a"
    p = r["parameter"]
    u["B"] = "beschreibend: Genauigkeit % " + ", ".join(f"{a} {g(a)}" for a in ARME) + "; Kopfparameter " + ", ".join(
        f"{a} {p[a]['kopf']}" for a in ARME if a in p)
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, off=None, sd=0.001, n=15, koll=None, nk=0.10):
    base = {"Ks": 0.743, "Kd": 0.724, "Klin": 0.735, "Kreg": 0.738, "Kklein": 0.736}
    base.update(off or {})
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        for a, v in base.items():
            x = v + gem + rng.normal(0, sd) - (0.2 if koll == a and i == 0 else 0)
            zl.append({"teil": "t", "arm": a, "seed": 2330 + i, "acc": x})
        zl.append({"teil": "nk", "arm": "Ks", "seed": 2330 + i, "acc": nk + rng.normal(0, 0.01)})
    return zl


def _synth_param():
    return {a: {"kopf": k} for a, k in zip(ARME, (49_162, 4_235_274, 10_250, 4_235_274, 49_738))}


def vortest() -> bool:
    ok = True
    rng = np.random.default_rng(44)

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-044-v2): {name}", flush=True)
            ok = False
    pruefe("Seeds frisch", not (set(SEEDS) & _BELEGT) and len(SEEDS) == 15)
    pruefe("übernommene Raten = I-044-r", all(st.TUNING_R[a] == lr for a, lr in TUNING_UEBERNOMMEN.items()))
    # Modelle: Formen, Kopfbudgets, Startzustand des Rumpfs je Seed gleich
    torch.manual_seed(0)
    netze = {}
    for a in ARME:
        torch.manual_seed(7)
        netze[a] = KopfNetz(a, 7)
    x = torch.randn(3, 3072)
    pruefe("Ausgabeform", all(n(x).shape == (3, 10) for n in netze.values()))
    pk = {a: param_kopf(n) for a, n in netze.items()}
    pruefe(f"Kklein ≈ Kopfbudget Ks ({pk['Kklein']} vs {pk['Ks']})", 0.9 <= pk["Kklein"] / pk["Ks"] <= 1.1)
    pruefe("Klin ≈ 10 k", 10_000 <= pk["Klin"] <= 10_300)
    pruefe("Kreg = Kd Parameter", pk["Kreg"] == pk["Kd"])
    w0 = netze["Ks"].conv[0].weight
    pruefe("gleicher Rumpf-Start je Seed", all(torch.equal(n.conv[0].weight, w0) for n in netze.values()))
    pruefe("Dropout nur in Kreg", isinstance(netze["Kreg"].drop, nn.Dropout) and all(isinstance(netze[a].drop, nn.Identity) for a in ARME if a != "Kreg"))
    # Urteilszweige erzwingen
    u = lambda **kw: urteil(auswertung(_synth(rng, **kw), _synth_param()))
    gesehen = set()

    def erwarte(name, ur, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k, w.split(" (")[0]))
            if not ur[k].startswith(w):
                print(f"  Vortest FEHLER (I-044-v2) {name}: {k} soll {w}, ist {ur[k]}", flush=True)
                ok = False
    erwarte("Normalfall", u(), NK="BESTANDEN", H1a="NICHT_UNTERLEGEN", H1b="NICHT_UNTERLEGEN", H1c="NICHT_UNTERLEGEN", H2="STRUKTUR_HILFT")
    erwarte("Ks schwach", u(off={"Ks": 0.725}), H1a="H0", H1b="H0", H1c="H0", H2="H0")
    erwarte("Rauschen", u(sd=0.03, off={"Ks": 0.735}), H1a="HU", H1b="HU", H1c="HU", H2="HU")
    erwarte("NK fällt", u(nk=0.3), NK="DURCHGEFALLEN", H1a="HU (Negativkontrolle", H2="HU (Negativkontrolle")
    erwarte("Kollaps Kreg", u(koll="Kreg"), H1b="HU (Instabilitaet", H1a="NICHT_UNTERLEGEN")
    erwarte("Arm fehlt", urteil(auswertung([z for z in _synth(rng) if z["arm"] != "Kklein"], _synth_param())), H1c="HU (Arme fehlen")
    for h, ja in (("H1a", "NICHT_UNTERLEGEN"), ("H1b", "NICHT_UNTERLEGEN"), ("H1c", "NICHT_UNTERLEGEN"), ("H2", "STRUKTUR_HILFT")):
        pruefe(f"Zweige {h}", {ja, "H0", "HU"} <= {w for k, w in gesehen if k == h})
    return ok


# ------------------------------------------------------------------ Lauf
def lauf(schnell=False, prozesse=None, zwischen=None, **_):
    if schnell:          # Rauchtest NUR Tuning-Seed, kleine Daten, 1 Epoche
        seeds, klein, ep, epnk = (TUNING_SEED,), True, 1, 1
    else:
        seeds, klein, ep, epnk = SEEDS, False, EPOCHEN, NK_EPOCHEN
        if set(seeds) & _BELEGT:
            raise ValueError("Seeds nicht frisch")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "seeds": list(seeds), "epochen": ep, "nk": epnk, "klein": klein}
    t0 = time.time()
    from abteilungen.struktur_gross import _lade_zwischen, _schreibe_zwischen
    z = _lade_zwischen(pfad, kennung)
    if "tuning" not in z:
        z["tuning"] = tuning(ep, klein, TUNING_SEED)
        _schreibe_zwischen(pfad, kennung, z)
    lr = dict(TUNING_UEBERNOMMEN, **{a: z["tuning"][a]["lr"] for a in NEU_GETUNT})
    aufg = [("t", a, s, lr[a], ep, klein) for a in ARME for s in seeds] + [("nk", "Ks", s, lr["Ks"], epnk, klein) for s in seeds]
    kosten = {"Kd": 3, "Kreg": 3, "Ks": 3}
    aufg.sort(key=lambda a: -kosten.get(a[1], 1))
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [v for k, v in alle.items() if k.startswith("einheit|")]
    params = {a: {"kopf": param_kopf(KopfNetz(a, 0))} for a in ARME}
    r = auswertung(zeilen, params)
    e = {"abteilung": "Struktur", "modul": "struktur_kopf_fair", "karte": KARTE, "version": 1, "seeds": list(seeds),
         "tuning_seed": TUNING_SEED, "lernraten": lr, "tuning": z["tuning"], "einstellungen": {"epochen": ep, "nk": epnk, "batch": st.BATCH,
         "dropout": DROPOUT, "weight_decay": WD, "n_klein": N_KLEIN}, "zwischendatei": str(pfad), "zeilen": zeilen,
         "sekunden": time.time() - t0, "auswertung": r}
    e["urteil"] = urteil(r)
    return e


# ================================================================== I-044-v3: Epochenreihe (vorregistriert 07.10.2026, vor jeder Zahl)
# Frage (aus recherche/2026-10-07_volltext_2.md, Teil A): Hält der Vorsprung des Strukturkopfs (I-044-v2: Ks - Klin +0,83 pp bei
# 4 Epochen) bei längerem Training, oder holen lineare/dichte Köpfe auf? Literatur sagt eher: schrumpft. Hält er bei 24 Epochen,
# spricht das für Kapazität (Grad-2-Merkmale), nicht für schnelleres Training.
# Arme Ks, Klin, Kklein; Epochen 4, 12, 24; Lernraten FEST aus I-044-v2 (keine neue Suche je Epochenzahl -- Vorbehalt, im LABORBUCH).
# Seeds 2350-2357 (8, frisch). NK: Ks permutiert, 4 Epochen.
# Hypothesen (Holm über H1, H2; gepaarter einseitiger t, n = 8):
#   H1 Ks - Klin > 0 bei 24 Epochen   (STRUKTUR_HAELT | H0 | HU)
#   H2 Ks - Kklein > 0 bei 24 Epochen (STRUKTUR_HAELT | H0 | HU)
#   H3 (nicht in Holm, zweiseitig) Trend: d24 - d4 für Ks - Klin (SCHRUMPFT | WAECHST | HU, zweiseitiger t, alpha 0,05)
#   B: alle Mittel je Epochenzahl.
KARTE_V3 = "I-044-v3"
SEEDS_V3 = tuple(range(2350, 2358))
ARME_V3 = ("Ks", "Klin", "Kklein")
EPOCHEN_V3 = (4, 12, 24)
LR_V3 = {"Ks": 0.003, "Klin": 0.003, "Kklein": 0.03}      # aus ergebnisse/I-044-v2__20261007_112259.json
HYPOTHESEN_V3 = {
    "H1": "Ks - Klin > 0 bei 24 Epochen (einseitig gepaart t, Holm über H1, H2). STRUKTUR_HAELT | H0 | HU.",
    "H2": "Ks - Kklein > 0 bei 24 Epochen. STRUKTUR_HAELT | H0 | HU.",
    "H3": "Trend (Ks - Klin)(24) - (Ks - Klin)(4), zweiseitig, nicht in Holm. SCHRUMPFT | WAECHST | HU.",
    "NK": f"Ks permutiert (4 Epochen), Mittel in {NK_BAND}.",
}


def auswertung_v3(zeilen: list) -> dict:
    acc = {}
    for z in zeilen:
        acc.setdefault((z["teil"], z["arm"]), {})[z["seed"]] = z["acc"]
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] != "nk"})
    r = {"arme": {}, "kollaps": {}, "diff": {}, "p_holm": {}, "negativ": {}}
    zl = {}
    for ep in EPOCHEN_V3:
        t = f"e{ep}"
        zl[ep] = [{"seed": s, **{a: acc[(t, a)][s] for a in ARME_V3 if (t, a) in acc and s in acc[(t, a)]}} for s in seeds]
        for a in ARME_V3:
            if zl[ep] and all(a in z for z in zl[ep]):
                r["arme"][f"{a}@{ep}"] = zusammenfassung(_arr(zl[ep], a))
                r["kollaps"][f"{a}@{ep}"] = kollaps_seeds(zl[ep], a)
    perm = [v for (t, a), d in acc.items() if t == "nk" for v in d.values()]
    r["negativ"] = {"mittel": float(np.mean(perm)), "n": len(perm)} if perm else {}
    ok = lambda ep, *a: zl[ep] and all(all(x in z for x in a) for z in zl[ep])
    d = lambda ep, x, y: _arr(zl[ep], x) - _arr(zl[ep], y)
    if ok(24, "Ks", "Klin"):
        r["diff"]["H1"] = _stat(d(24, "Ks", "Klin"))
    if ok(24, "Ks", "Kklein"):
        r["diff"]["H2"] = _stat(d(24, "Ks", "Kklein"))
    if ok(24, "Ks", "Klin") and ok(4, "Ks", "Klin"):
        r["diff"]["H3"] = _stat(d(24, "Ks", "Klin") - d(4, "Ks", "Klin"))
    for ep in EPOCHEN_V3:
        for x, y in (("Ks", "Klin"), ("Ks", "Kklein"), ("Kklein", "Klin")):
            if ok(ep, x, y):
                r.setdefault("beschreibend", {})[f"{x} - {y} @{ep}"] = _stat(d(ep, x, y))
    prim = {h: r["diff"][h].get("p_gr", 1.0) for h in ("H1", "H2") if h in r["diff"]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def urteil_v3(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutierte Labels Ks: {r['negativ'].get('mittel', float('nan')):.3f}; Band {NK_BAND})"}
    for h, g in (("H1", "Klin"), ("H2", "Kklein")):
        v = _vorab(r, nk, h, (f"Ks@24", f"{g}@24"))
        u[h] = v or _urteil(r, nk, h, (f"Ks@24", f"{g}@24"), "STRUKTUR_HAELT")
    if "H3" not in r["diff"] or "mittel_pp" not in r["diff"]["H3"]:
        u["H3"] = "HU (Arme fehlen oder zu wenige Seeds)"
    elif nk != "BESTANDEN":
        u["H3"] = "HU (Negativkontrolle durchgefallen)"
    else:
        x = r["diff"]["H3"]
        z = f"d {_f(x['mittel_pp'])} pp, 90-%-KI [{_f(x['ki_pp'][0])}; {_f(x['ki_pp'][1])}], n {x['n']}, p_zwei {x['p_zwei']:.3g}"
        u["H3"] = (f"SCHRUMPFT ({z})" if x["mittel_pp"] < 0 else f"WAECHST ({z})") if x["p_zwei"] < ALPHA else f"HU ({z})"
    g = lambda k: f"{r['arme'][k]['mittel'] * 100:.2f}".replace(".", ",") if k in r["arme"] else "n/a"
    u["B"] = "beschreibend: " + "; ".join(f"{ep} Ep.: " + ", ".join(f"{a} {g(f'{a}@{ep}')}" for a in ARME_V3) for ep in EPOCHEN_V3)
    return u


def _synth_v3(rng, sd=0.001, n=8, trend=-0.004, ks24=0.01, nk=0.10):
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        for ep in EPOCHEN_V3:
            f = (ep - 4) / 20
            base = {"Klin": 0.734 + 0.01 * f, "Kklein": 0.726 + 0.02 * f}
            base["Ks"] = base["Klin"] + 0.008 + trend * f + (ks24 - 0.008 - trend) * f
            for a, v in base.items():
                zl.append({"teil": f"e{ep}", "arm": a, "seed": 2350 + i, "acc": v + gem + rng.normal(0, sd)})
        zl.append({"teil": "nk", "arm": "Ks", "seed": 2350 + i, "acc": nk + rng.normal(0, 0.01)})
    return zl


def _vortest_v3() -> bool:
    ok = True
    rng = np.random.default_rng(443)

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-044-v3): {name}", flush=True)
            ok = False
    pruefe("Seeds frisch", not (set(SEEDS_V3) & (_BELEGT | set(SEEDS))))
    u = lambda **kw: urteil_v3(auswertung_v3(_synth_v3(rng, **kw)))
    a = u(ks24=0.012)                                  # Ks vorn bei 24
    pruefe("Haelt", a["H1"].startswith("STRUKTUR_HAELT") and a["H2"].startswith("STRUKTUR_HAELT"))
    b = u(ks24=-0.012)
    pruefe("H0", b["H1"].startswith("H0") and b["H2"].startswith("H0"))
    c = u(sd=0.03, ks24=0.0)
    pruefe("HU", c["H1"].startswith("HU") and c["H3"].startswith("HU"))
    pruefe("schrumpft", u(ks24=0.002)["H3"].startswith("SCHRUMPFT"))
    pruefe("waechst", u(ks24=0.02)["H3"].startswith("WAECHST"))
    pruefe("NK", u(nk=0.3)["H1"].startswith("HU (Negativkontrolle"))
    pruefe("Arm fehlt", urteil_v3(auswertung_v3([z for z in _synth_v3(rng) if not (z["arm"] == "Kklein" and z["teil"] == "e24")]))["H2"].startswith("HU (Arme fehlen"))
    return ok


def lauf_v3(schnell=False, prozesse=None, zwischen=None, **_):
    if schnell:
        seeds, klein, eps, epnk = (TUNING_SEED,), True, (1, 2, 3), 1
    else:
        seeds, klein, eps, epnk = SEEDS_V3, False, EPOCHEN_V3, 4
        if set(seeds) & (_BELEGT | set(SEEDS)):
            raise ValueError("Seeds nicht frisch")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V3}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE_V3, "version": 1, "seeds": list(seeds), "epochen": list(eps), "nk": epnk, "klein": klein}
    t0 = time.time()
    aufg = [(f"e{ep_v}", a, s, LR_V3[a], ep, klein) for ep_v, ep in zip(EPOCHEN_V3, eps) for a in ARME_V3 for s in seeds]
    aufg += [("nk", "Ks", s, LR_V3["Ks"], epnk, klein) for s in seeds]
    aufg.sort(key=lambda a: -a[4])                     # lange Einheiten zuerst
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter, schluessel=_schl, prozesse=prozesse)
    zeilen = [v for k, v in alle.items() if k.startswith("einheit|")]
    r = auswertung_v3(zeilen)
    e = {"abteilung": "Struktur", "modul": "struktur_kopf_fair", "karte": KARTE_V3, "version": 1, "seeds": list(seeds),
         "lernraten": LR_V3, "epochen": list(eps), "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0,
         "auswertung": r}
    e["urteil"] = urteil_v3(r)
    return e


# ================================================================== I-044-v4: Epochenreihe mit Lernratensuche je Epochenzahl und Augmentierung
# (vorregistriert 27.10.2026, vor jeder Zahl; Rechnen auf MPCDF Raven)
# Anlass: I-044-v3 (Ks verliert bei 24 Epochen gegen Klin) hatte zwei Vorbehalte: (1) Lernraten aus I-044-v2 übernommen, nicht je
# Epochenzahl gesucht; (2) Abfall ALLER Arme von 12 auf 24 Epochen (Überanpassung, keine Augmentierung). Beides wird geschlossen.
# Arme Ks, Klin, Kklein (wie v3). Epochen 4, 12, 24, 48. Lernrate je (Arm, Epochenzahl) auf dem Tuning-Seed 2399
# (gittersuche_stabil, Gitter 1e-3/3e-3/1e-2, Faktor 3, beidseitig verlängerbar, Validierung = letzte 10 % Training, Augmentierung
# im Training wie in der Messung, nie in der Validierung). Standard-Augmentierung CIFAR: zufälliger Ausschnitt 32 nach Auffüllen mit
# 4 Pixeln (Null im standardisierten Raum = Mittelwertfarbe) + horizontales Spiegeln (p = 0,5), für alle Arme gleich: derselbe
# Augmentierungsstrom je Seed und Epochenzahl (eigener Generator seed + 9, unabhängig von Arm, Rate und Reihenfolge der Bilder).
# AUSWAHL der Epochenzahl je Arm: die Epochenzahl mit der höchsten Validierungsgenauigkeit (bei der gewählten Rate, Tuning-Seed),
# Gleichstand -> kleinere Epochenzahl. Sie fällt im Tuning, VOR jeder Test-Messung, und steht in der Zwischendatei fest.
# Seeds 2710-2719 (10, frisch), Tuning 2399. NK: Ks mit permutierten Labels, 4 Epochen, mit Augmentierung, Rate Ks@4.
# Hypothesen (Holm über H1, H2; gepaarter einseitiger t, n = 10; Urteile STRUKTUR_HAELT | H0 | HU):
#   H1 Ks - Klin > 0, jeder Arm bei SEINER gewählten Epochenzahl     H2 Ks - Kklein > 0, ebenso
#   H3 (nicht in Holm, zweiseitig, alpha 0,05) Trend (Ks - Klin)@48 - (Ks - Klin)@4 (SCHRUMPFT | WAECHST | HU)
#   B: alle Mittel je (Arm, Epochenzahl), Ks - Klin und Ks - Kklein je Epochenzahl, gewählte Epochenzahlen und Raten.
KARTE_V4 = "I-044-v4"
SEEDS_V4 = tuple(range(2710, 2720))
ARME_V4 = ("Ks", "Klin", "Kklein")
EPOCHEN_V4 = (4, 12, 24, 48)
NK_EPOCHEN_V4 = 4
PAD = 4
AUG_OFFSET = 9
KOSTEN_V4 = {"Ks": 1.0, "Klin": 0.9, "Kklein": 0.9}      # grobe relative Kosten je Epoche (nur Sortierung)
HYPOTHESEN_V4 = {
    "H1": "Ks - Klin > 0, jeder Arm bei seiner auf Validierung gewählten Epochenzahl (einseitig gepaart t, Holm über H1, H2). STRUKTUR_HAELT | H0 | HU.",
    "H2": "Ks - Kklein > 0, ebenso. STRUKTUR_HAELT | H0 | HU.",
    "H3": "Trend (Ks - Klin)(48) - (Ks - Klin)(4), zweiseitig, nicht in Holm. SCHRUMPFT | WAECHST | HU.",
    "NK": f"Ks permutiert (4 Epochen, mit Augmentierung), Mittel in {NK_BAND}. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
}


def _augmentiere(x, ga, pad=PAD):
    """Standard-CIFAR-Augmentierung für einen Stapel x (B, 3072, standardisiert): zufälliger Ausschnitt nach Null-Auffüllung + Spiegeln.
    Zufall NUR aus dem Generator ga (deterministisch je Seed)."""
    b = x.shape[0]
    off = torch.randint(0, 2 * pad + 1, (b, 2), generator=ga)
    flip = torch.rand(b, generator=ga) < 0.5
    return _ausschnitt(x, off[:, 0], off[:, 1], flip, pad)


def _ausschnitt(x, oy, ox, flip, pad=PAD):
    b = x.shape[0]
    xp = F.pad(x.reshape(b, 3, 32, 32), (pad, pad, pad, pad))
    r = torch.arange(32)
    zeilen = oy[:, None] + r
    spalten = torch.where(flip[:, None], ox[:, None] + 31 - r, ox[:, None] + r)
    out = xp[torch.arange(b)[:, None, None, None], torch.arange(3)[None, :, None, None], zeilen[:, None, :, None], spalten[:, None, None, :]]
    return out.reshape(b, 3072)


def trainiere_v4(arm, lr, seed, epochen, klein=False, val=False, permutiert=False):
    """Wie trainiere (I-044-v2), aber MIT Augmentierung im Training. Test/Validierung ohne Augmentierung."""
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = daten("cifar10", klein)
    if val:
        nv = len(ytr) // 10
        xtr, ytr, xte, yte = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
    g = torch.Generator().manual_seed(seed + 5)
    ga = torch.Generator().manual_seed(seed + AUG_OFFSET)
    if permutiert:
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 5 + 100 * ord(arm[0]) + ord(arm[-1])))]
    torch.manual_seed(seed)
    netz = KopfNetz(arm, seed)
    ps = [p for p in netz.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=lr)
    n = len(ytr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochen * -(-n // st.BATCH))
    netz.train()
    for _ in range(epochen):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, st.BATCH):
            j = perm[i:i + st.BATCH]
            l = F.cross_entropy(netz(_augmentiere(xtr[j], ga)), ytr[j])
            if not torch.isfinite(l):
                return {"acc": 0.1, "instabil": True}
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
    netz.eval()
    with torch.no_grad():
        richtig = sum(int((netz(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum()) for i in range(0, len(yte), 1000))
    return {"acc": richtig / len(yte), "instabil": False}


def _arbeiter_v4(a):
    teil, arm, s, lr, ep, klein = a
    t0 = time.time()
    r = trainiere_v4(arm, lr, s, ep, klein=klein, val=(teil == "tune"), permutiert=(teil == "nk"))
    return _schl_v4(a), {"teil": teil, "arm": arm, "seed": s, "lr": lr, "epochen": ep, "acc": r["acc"], "instabil": r["instabil"],
                         "sekunden": time.time() - t0}


def _schl_v4(a):
    return f"einheit|{a[0]}|{a[1]}|{a[4]}|{a[2]}|{a[3]!r}"


def tuning_v4(eps, klein, tseed, pfad, kennung, prozesse=None, fn=_arbeiter_v4):
    """Rate je (Arm, Epochenzahl) auf dem Tuning-Seed; jede einzelne Rechnung steht in der Zwischendatei (Wiederaufnahme je Aufgabe).
    Rückgabe: (info, auswahl). auswahl = {'epochen': {arm: gewählte Epochenzahl (Etikett)}, 'val': {arm: {ep: Validierungsgenauigkeit}}}."""
    from abteilungen.struktur_gross import _rechne
    etikett = dict(zip(EPOCHEN_V4, eps))

    def aufgaben_fuer(k, lr):
        return [(k, tseed, lr)]

    def auswerten(aufg):
        a2 = [("tune", k.split("@")[0], s, lr, etikett[int(k.split("@")[1])], klein) for k, s, lr in aufg]
        a2.sort(key=lambda a: -a[4] * KOSTEN_V4[a[1]])
        zeilen = _rechne(a2, pfad, kennung, fn=fn, schluessel=_schl_v4, prozesse=prozesse)
        return [zeilen[_schl_v4(("tune", k.split("@")[0], s, lr, etikett[int(k.split("@")[1])], klein))] for k, s, lr in aufg]

    schl = [f"{a}@{e}" for a in ARME_V4 for e in EPOCHEN_V4]
    info, _ = gittersuche_stabil(schl, aufgaben_fuer, auswerten, gitter=st.LR_GITTER, faktor=st.LR_FAKTOR, kollaps=KOLLAPS)
    res = {k: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"], "runden": v["runden"],
               "val": v["gitter"][str(v["lr"])]["mittel"], "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}}
           for k, v in info.items()}
    return res, auswahl_epochen(res)


def auswahl_epochen(res):
    """Gewählte Epochenzahl je Arm: höchste Validierungsgenauigkeit bei der gewählten Rate, Gleichstand -> kleinere Epochenzahl."""
    val = {a: {e: res[f"{a}@{e}"]["val"] for e in EPOCHEN_V4} for a in ARME_V4}
    return {"epochen": {a: max(EPOCHEN_V4, key=lambda e: (round(val[a][e], 12), -e)) for a in ARME_V4}, "val": val}


def auswertung_v4(zeilen: list, auswahl: dict) -> dict:
    acc = {}
    for z in zeilen:
        if z["teil"] != "tune":
            acc.setdefault((z["teil"], z["arm"]), {})[z["seed"]] = z["acc"]
    seeds = sorted({z["seed"] for z in zeilen if z["teil"] not in ("nk", "tune")})
    best = auswahl["epochen"]
    r = {"arme": {}, "kollaps": {}, "diff": {}, "p_holm": {}, "negativ": {}, "auswahl": best}
    zl = {}
    for ep in EPOCHEN_V4:
        t = f"e{ep}"
        zl[ep] = [{"seed": s, **{a: acc[(t, a)][s] for a in ARME_V4 if (t, a) in acc and s in acc[(t, a)]}} for s in seeds]
        for a in ARME_V4:
            if zl[ep] and all(a in z for z in zl[ep]):
                r["arme"][f"{a}@{ep}"] = zusammenfassung(_arr(zl[ep], a))
                r["kollaps"][f"{a}@{ep}"] = kollaps_seeds(zl[ep], a)
    perm = [v for (t, a), d in acc.items() if t == "nk" for v in d.values()]
    r["negativ"] = {"mittel": float(np.mean(perm)), "n": len(perm)} if perm else {}
    ok = lambda ep, *a: bool(zl[ep]) and all(all(x in z for x in a) for z in zl[ep])
    v = lambda a, ep: _arr(zl[ep], a)
    for h, g in (("H1", "Klin"), ("H2", "Kklein")):
        if ok(best["Ks"], "Ks") and ok(best[g], g):
            r["diff"][h] = _stat(v("Ks", best["Ks"]) - v(g, best[g]))
    if ok(48, "Ks", "Klin") and ok(4, "Ks", "Klin"):
        r["diff"]["H3"] = _stat((v("Ks", 48) - v("Klin", 48)) - (v("Ks", 4) - v("Klin", 4)))
    for ep in EPOCHEN_V4:
        for x, y in (("Ks", "Klin"), ("Ks", "Kklein"), ("Kklein", "Klin")):
            if ok(ep, x, y):
                r.setdefault("beschreibend", {})[f"{x} - {y} @{ep}"] = _stat(v(x, ep) - v(y, ep))
    prim = {h: r["diff"][h].get("p_gr", 1.0) for h in ("H1", "H2") if h in r["diff"]}
    if prim:
        r["p_holm"] = holm(prim)
    return r


def urteil_v4(r: dict) -> dict:
    nk = _nk(r)
    u = {"NK": f"{nk} (Mittel permutierte Labels Ks: {r['negativ'].get('mittel', float('nan')):.3f}; Band {NK_BAND})"}
    b = r["auswahl"]
    for h, g in (("H1", "Klin"), ("H2", "Kklein")):
        arme = (f"Ks@{b['Ks']}", f"{g}@{b[g]}")
        u[h] = _urteil(r, nk, h, arme, "STRUKTUR_HAELT") if h in r["diff"] else "HU (Arme fehlen oder zu wenige Seeds)"
    if "H3" not in r["diff"] or "mittel_pp" not in r["diff"]["H3"]:
        u["H3"] = "HU (Arme fehlen oder zu wenige Seeds)"
    elif nk != "BESTANDEN":
        u["H3"] = "HU (Negativkontrolle durchgefallen)"
    else:
        x = r["diff"]["H3"]
        z = f"d {_f(x['mittel_pp'])} pp, 90-%-KI [{_f(x['ki_pp'][0])}; {_f(x['ki_pp'][1])}], n {x['n']}, p_zwei {x['p_zwei']:.3g}"
        u["H3"] = (f"SCHRUMPFT ({z})" if x["mittel_pp"] < 0 else f"WAECHST ({z})") if x["p_zwei"] < ALPHA else f"HU ({z})"
    g = lambda k: f"{r['arme'][k]['mittel'] * 100:.2f}".replace(".", ",") if k in r["arme"] else "n/a"
    u["B"] = ("beschreibend: gewählte Epochenzahl " + ", ".join(f"{a} {b[a]}" for a in ARME_V4) + "; "
              + "; ".join(f"{ep} Ep.: " + ", ".join(f"{a} {g(f'{a}@{ep}')}" for a in ARME_V4) for ep in EPOCHEN_V4))
    return u


def lauf_v4(schnell=False, prozesse=None, zwischen=None, **_):
    """Rauchtest (schnell=True): NUR Tuning-Seed, kleine Daten, Epochen 1-4; Ergebnis verwerfen (eigene Zwischendatei -schnell)."""
    from abteilungen.struktur_gross import _rechne
    if schnell:
        seeds, klein, eps, epnk = (TUNING_SEED,), True, (1, 2, 3, 4), 1
    else:
        seeds, klein, eps, epnk = SEEDS_V4, False, EPOCHEN_V4, NK_EPOCHEN_V4
        if set(seeds) & (_BELEGT | set(SEEDS) | set(SEEDS_V3)):
            raise ValueError("Seeds nicht frisch")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_V4}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE_V4, "version": 1, "seeds": list(seeds), "epochen": list(eps), "nk": epnk, "klein": klein,
               "lr_gitter": list(st.LR_GITTER), "tuning_seed": TUNING_SEED, "pad": PAD}
    t0 = time.time()
    from abteilungen.struktur_gross import _lade_zwischen, _schreibe_zwischen
    # Tuning: jede Rechnung in der Zwischendatei (Wiederaufnahme je Aufgabe); Rate und Epochenauswahl danach fest unter "auswahl|..."
    tun, auswahl = tuning_v4(eps, klein, TUNING_SEED, pfad, kennung, prozesse=prozesse)
    z = _lade_zwischen(pfad, kennung)
    if "tuning_fest" not in z:
        z["tuning_fest"] = {"tuning": tun, "auswahl": auswahl}
        _schreibe_zwischen(pfad, kennung, z)
    tun, auswahl = z["tuning_fest"]["tuning"], z["tuning_fest"]["auswahl"]
    lr = {k: v["lr"] for k, v in tun.items()}
    etikett = dict(zip(EPOCHEN_V4, eps))
    aufg = [(f"e{ev}", a, s, lr[f"{a}@{ev}"], etikett[ev], klein) for ev in EPOCHEN_V4 for a in ARME_V4 for s in seeds]
    aufg += [("nk", "Ks", s, lr[f"Ks@{NK_EPOCHEN_V4}"], epnk, klein) for s in seeds]
    aufg.sort(key=lambda a: -a[4] * KOSTEN_V4[a[1]])           # lange Aufgaben zuerst
    alle = _rechne(aufg, pfad, kennung, fn=_arbeiter_v4, schluessel=_schl_v4, prozesse=prozesse)
    zeilen = [v for k, v in alle.items() if k.startswith("einheit|") and v["teil"] != "tune"]
    r = auswertung_v4(zeilen, auswahl)
    e = {"abteilung": "Struktur", "modul": "struktur_kopf_fair", "karte": KARTE_V4, "version": 1, "seeds": list(seeds),
         "tuning_seed": TUNING_SEED, "lernraten": lr, "tuning": tun, "auswahl": auswahl, "epochen": list(eps),
         "einstellungen": {"batch": st.BATCH, "pad": PAD, "spiegeln": 0.5, "nk_epochen": epnk},
         "zwischendatei": str(pfad), "zeilen": zeilen, "sekunden": time.time() - t0, "auswertung": r}
    e["urteil"] = urteil_v4(r)
    return e


def _synth_v4(rng, sd=0.001, n=10, ks=0.0, kk=0.0, nk=0.10, koll=None):
    """Synthetische Zeilen. Klin/Kklein: wachsen bis 24 Epochen, dann leicht fallend; Ks = Klin + ks bei 48, Kklein = Klin - 0,002 + kk."""
    zl = []
    for i in range(n):
        gem = rng.normal(0, 0.003)
        for ep in EPOCHEN_V4:
            f = {4: 0.0, 12: 0.6, 24: 1.0, 48: 0.9}[ep]
            klin = 0.74 + 0.02 * f
            base = {"Klin": klin, "Kklein": klin - 0.002 + kk, "Ks": klin + 0.008 + (ks - 0.008) * (ep - 4) / 44}
            for a, v in base.items():
                x = v + gem + rng.normal(0, sd) - (0.2 if koll == (a, ep) and i == 0 else 0)
                zl.append({"teil": f"e{ep}", "arm": a, "seed": 2710 + i, "acc": x})
        zl.append({"teil": "nk", "arm": "Ks", "seed": 2710 + i, "acc": nk + rng.normal(0, 0.01)})
    return zl


AUSWAHL_SYNTH = {"epochen": {"Ks": 48, "Klin": 24, "Kklein": 24}, "val": {}}


def _vortest_v4() -> bool:
    import tempfile
    ok = True
    rng = np.random.default_rng(444)

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-044-v4): {name}", flush=True)
            ok = False
    pruefe("Seeds frisch (10, nicht in v2/v3/Altbestand)", len(SEEDS_V4) == 10 and not (set(SEEDS_V4) & (_BELEGT | set(SEEDS) | set(SEEDS_V3))) and TUNING_SEED not in SEEDS_V4)
    # ---- Augmentierung
    x = torch.randn(5, 3072)
    img = x.reshape(5, 3, 32, 32)
    nul = torch.full((5,), PAD)
    aus = _ausschnitt(x, nul, nul, torch.zeros(5, dtype=torch.bool)).reshape(5, 3, 32, 32)
    pruefe("Ausschnitt ohne Versatz = Identität", torch.equal(aus, img))
    sp = _ausschnitt(x, nul, nul, torch.ones(5, dtype=torch.bool)).reshape(5, 3, 32, 32)
    pruefe("Spiegeln = horizontale Umkehr", torch.equal(sp, img.flip(-1)))
    hoch = _ausschnitt(x, torch.zeros(5, dtype=torch.long), nul, torch.zeros(5, dtype=torch.bool)).reshape(5, 3, 32, 32)
    pruefe("Versatz nach unten: oben Null, Rest verschoben", bool((hoch[:, :, :PAD] == 0).all()) and torch.equal(hoch[:, :, PAD:], img[:, :, :32 - PAD]))
    rechts = _ausschnitt(x, nul, torch.full((5,), 2 * PAD), torch.zeros(5, dtype=torch.bool)).reshape(5, 3, 32, 32)
    pruefe("Versatz nach links: rechts Null", bool((rechts[:, :, :, -PAD:] == 0).all()) and torch.equal(rechts[:, :, :, :32 - PAD], img[:, :, :, PAD:]))
    g1, g2, g3 = (torch.Generator().manual_seed(s) for s in (2710 + AUG_OFFSET, 2710 + AUG_OFFSET, 2711 + AUG_OFFSET))
    a1 = [_augmentiere(x, g1) for _ in range(3)]
    a2 = [_augmentiere(x, g2) for _ in range(3)]
    a3 = [_augmentiere(x, g3) for _ in range(3)]
    pruefe("Augmentierung deterministisch je Seed", all(torch.equal(p, q) for p, q in zip(a1, a2)) and not all(torch.equal(p, q) for p, q in zip(a1, a3)))
    pruefe("Augmentierung verändert Bilder und erhält Form/Betrag", a1[0].shape == x.shape and not torch.equal(a1[0], x)
           and bool((a1[0].abs().max() <= x.abs().max() + 1e-6)))
    pruefe("Augmentierungsstrom unabhängig von Arm und Rate (nur Generator seed+9)", "lr" not in _augmentiere.__code__.co_varnames and "arm" not in _augmentiere.__code__.co_varnames)
    # ---- Tuning (Rate je Epochenzahl), Epochenauswahl, Wiederaufnahme: synthetischer Arbeiter, 1 Prozess
    aufrufe = []

    def fake(a):
        teil, arm, s, lr, ep, klein = a
        aufrufe.append(a)
        # Optimum der Rate wandert mit der Epochenzahl (3e-3 bei <=12, 1e-3 bei 24, 3e-4 bei 48 -> Gitter muss nach unten wachsen)
        opt = {4: 3e-3, 12: 3e-3, 24: 1e-3, 48: 3.3333e-4}[{1: 4, 2: 12, 3: 24, 4: 48}[ep]]
        gipfel = {"Ks": 0.70, "Klin": 0.72, "Kklein": 0.71}[arm] + 0.02 * {1: 0.0, 2: 0.6, 3: 1.0, 4: 1.4}[ep]
        if arm == "Klin" and ep == 1:
            gipfel += 0.05          # Klin hat seinen Bestwert bei 48 NICHT: Auswahl muss je Arm verschieden ausfallen können
        acc = gipfel - 0.03 * abs(np.log10(lr / opt))
        return _schl_v4(a), {"teil": teil, "arm": arm, "seed": s, "lr": lr, "epochen": ep, "acc": float(acc), "instabil": False, "sekunden": 0.0}
    with tempfile.TemporaryDirectory() as td:
        pfad = Path(td) / "z.json"
        kennung = {"karte": "vortest"}
        tun, aw = tuning_v4((1, 2, 3, 4), True, TUNING_SEED, pfad, kennung, prozesse=1, fn=fake)
        n1 = len(aufrufe)
        pruefe("Rate je (Arm, Epochenzahl) verschieden und Gitter wächst nach unten",
               all(abs(tun[f"{a}@4"]["lr"] - 3e-3) < 1e-9 for a in ARME_V4) and all(tun[f"{a}@48"]["lr"] < 1e-3 for a in ARME_V4)
               and all(tun[f"{a}@24"]["lr"] < tun[f"{a}@12"]["lr"] + 1e-12 for a in ARME_V4))
        pruefe("Epochenauswahl: Ks/Kklein 48, Klin 4 (je Arm eigene Wahl)", aw["epochen"] == {"Ks": 48, "Klin": 4, "Kklein": 48})
        tun2, aw2 = tuning_v4((1, 2, 3, 4), True, TUNING_SEED, pfad, kennung, prozesse=1, fn=fake)
        pruefe("Wiederaufnahme: zweiter Tuning-Lauf rechnet nichts neu und liefert dasselbe", len(aufrufe) == n1 and tun2 == tun and aw2 == aw)
        # Wiederaufnahme mitten im Lauf: Zwischendatei mit der Hälfte der Aufgaben
        from abteilungen.struktur_gross import _rechne
        aufg = [(f"e{ev}", "Ks", s, 3e-3, ev, True) for ev in (1, 2) for s in (1, 2, 3)]
        p2 = Path(td) / "w.json"
        aufrufe.clear()
        _rechne(aufg[:3], p2, kennung, fn=fake, schluessel=_schl_v4, prozesse=1)
        aufrufe.clear()
        _rechne(aufg, p2, kennung, fn=fake, schluessel=_schl_v4, prozesse=1)
        pruefe("Wiederaufnahme je Aufgabe (nur 3 offene gerechnet)", len(aufrufe) == 3)
    # Auswahlregel: Gleichstand -> kleinere Epochenzahl
    gl = {f"{a}@{e}": {"val": 0.7} for a in ARME_V4 for e in EPOCHEN_V4}
    pruefe("Gleichstand -> kleinste Epochenzahl", auswahl_epochen(gl)["epochen"] == {a: 4 for a in ARME_V4})
    # ---- Urteilszweige
    u = lambda aw=AUSWAHL_SYNTH, **kw: urteil_v4(auswertung_v4(_synth_v4(rng, **kw), aw))
    a = u(ks=0.012)                                    # Ks vorn
    pruefe("Haelt", a["H1"].startswith("STRUKTUR_HAELT") and a["H2"].startswith("STRUKTUR_HAELT") and a["NK"].startswith("BESTANDEN"))
    b = u(ks=-0.012)
    pruefe("H0", b["H1"].startswith("H0") and b["H2"].startswith("H0"))
    c = urteil_v4(auswertung_v4(_synth_v4(np.random.default_rng(5), sd=0.05, ks=0.001), AUSWAHL_SYNTH))
    pruefe("HU", c["H1"].startswith("HU") and c["H2"].startswith("HU") and c["H3"].startswith("HU"))
    pruefe("SCHRUMPFT", u(ks=0.002)["H3"].startswith("SCHRUMPFT"))
    pruefe("WAECHST", u(ks=0.02)["H3"].startswith("WAECHST"))
    n = u(nk=0.3)
    pruefe("NK fällt -> alles HU", n["NK"].startswith("DURCHGEFALLEN") and all(n[h].startswith("HU (Negativkontrolle") for h in ("H1", "H2", "H3")))
    pruefe("Arm fehlt", urteil_v4(auswertung_v4([z for z in _synth_v4(rng) if not (z["arm"] == "Kklein" and z["teil"] == "e24")], AUSWAHL_SYNTH))["H2"].startswith("HU (Arme fehlen"))
    k = urteil_v4(auswertung_v4(_synth_v4(rng, ks=0.012, koll=("Ks", 48)), AUSWAHL_SYNTH))
    pruefe("Kollaps im gewählten Arm -> HU", k["H1"].startswith("HU (Instabilitaet") and k["H2"].startswith("HU (Instabilitaet"))
    k2 = urteil_v4(auswertung_v4(_synth_v4(rng, ks=0.012, koll=("Ks", 4)), AUSWAHL_SYNTH))
    pruefe("Kollaps in nicht gewählter Epochenzahl ändert H1 nicht", k2["H1"].startswith("STRUKTUR_HAELT"))
    # H1 nutzt WIRKLICH die gewählten Epochenzahlen (Ks@24 gegen Klin@4: Klin@4 schlechter -> hält; Ks@48 gegen Klin@24 -> hält nicht)
    zl = _synth_v4(rng, ks=-0.012)
    r1 = auswertung_v4(zl, {"epochen": {"Ks": 48, "Klin": 24, "Kklein": 24}, "val": {}})
    r2 = auswertung_v4(zl, {"epochen": {"Ks": 4, "Klin": 4, "Kklein": 4}, "val": {}})
    pruefe("H1 folgt der Auswahl", r1["diff"]["H1"]["mittel_pp"] < -1.0 and r2["diff"]["H1"]["mittel_pp"] > 0.5)
    return ok


_vortest_v2 = vortest


def vortest() -> bool:          # noqa: F811  (deckt I-044-v2, -v3 und -v4 ab)
    a = _vortest_v2()
    b = _vortest_v3()
    c = _vortest_v4()
    return a and b and c
