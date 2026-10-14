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


_vortest_v2 = vortest


def vortest() -> bool:          # noqa: F811  (deckt I-044-v2 und I-044-v3 ab)
    a = _vortest_v2()
    b = _vortest_v3()
    return a and b
