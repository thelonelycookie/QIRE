"""Abteilung Struktur -- I-042 „Strukturschicht in groß“: ersetzt eine feste/strukturierte Mischung + Nichtlinearität
eine dichte erste Schicht in echten Netzen (CIFAR-10, Fashion-MNIST 32x32) über wachsende Breite N?
(Plan: PLAN_2026-10-06_gesamt.md; Folge von I-038/I-041. Die Wirkung ist klassisch (I-038); Quantenstufe Q1/Q2:
 das „Born-Quadrat“ z² wird gegen |z| und ReLU geprüft, mehr nicht. Formuliert wird „quanten-inspirierter Vorteil“.)

HEILMEIER:
  Was wollen wir?  Zeigen oder widerlegen, dass Fastfood-2 (0 trainierte Mischparameter) + z² bei echten Netzen auf
      Rohpixeln mit der dichten Schicht gleichzieht (Nichtunterlegenheit 1 pp) und >= 100x weniger Parameter braucht.
  Wie heute?  I-041: bei Merkmalen + linearem Kopf (kein Netz, 1.024 Merkmale) Fastfood² ≡ Haar², Vorsprung vs. Pixel
      wächst mit n; Born-Quadrat ≡ |z|. Offen: Gilt das im trainierten 2-Schicht-MLP, bei N bis 16.384, auf CIFAR?
  Was ist neu?  Breitenreihe N = 1024/4096/16384 mit 7 Armen + 1 Budgetkontrolle, gemessene Inferenzzeit und FLOPs.

DATEN (festgeschrieben): CIFAR-10 (50.000 Training / 10.000 Test, 3x32x32 = 3.072 Eingänge, je Kanal standardisiert;
  Quelle fast.ai-Spiegel cifar10.tgz, PNG) und Fashion-MNIST 32x32 (60.000 / 10.000; bilineares Hochskalieren aus 28x28 wie
  I-041, als uint8 gerundet; skalar standardisiert). Rohpixel, keine Augmentierung. Volle Mengen; der Seed bestimmt
  Initialisierung, Fastfood-Vorzeichen, Batchfolge. Test nur am Ende (letzte Epoche, kein Early Stopping).
MODELL: x -> Schicht(Arm) -> Nichtlinearität -> LayerNorm (ohne Parameter) -> linearer Kopf N -> 10 (identisch bei allen Armen).
  Strukturierte Arme arbeiten auf D = Eingang auf Zweierpotenz aufgefüllt (CIFAR 4.096, Fashion 1.024); N > D: N/D unabhängige Blöcke;
  N < D (nur CIFAR, N=1024): feste Zufallsauswahl von N der D Ausgänge.
ARME: A dicht (ReLU) | B Fastfood-2 fest + z² | C Fastfood-2 fest + |z| | D Fastfood-2 fest + ReLU |
  E Fastfood, Diagonalen D1, D2 trainiert + z² | F Monarch (2 Blockdiagonale + Permutation, g = √D) + z² |
  G niedriger Rang r = rund(Param_F / (Eingang+N)) + z² | K (Budgetkontrolle für H3) Rang r = aufgerundet(Param_E/(Eingang+N)) + z²
  (K hat mindestens E's Budget: für E also die strengere Bedingung). Fastfood-2 = H D2 H D1 (wie I-041), H als Kronecker H_a ⊗ H_b.
TRAINING: Adam, feste Epochen, Batch 256, Kosinus-Lernratenabfall; Lernrate je (Datensatz, N, Arm) auf Tuning-Seed 2299
  (gittersuche_stabil, Gitter in beide Richtungen, Validierung = letzte 10 % der Trainingsbilder, nie Test).
HYPOTHESEN je Datensatz (Seeds 2200-2204, gepaart; konfirmatorisch gepaarter einseitiger t-Test, weil exakter Wilcoxon
  bei n = 5 nie unter p = 1/32 kommt und Holm das zerstört; Wilcoxon steht beschreibend daneben; Holm über die Familie je Datensatz).
"""
from __future__ import annotations

import json
import math
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abteilungen.clifford_suche import holm
from abteilungen.walsh_kontrolle import wilcoxon_schnell
from labor.gemeinsam import ERGEBNISSE, WURZEL, gittersuche_stabil, parallel, zusammenfassung

KARTE = "I-042"
DATENSAETZE = ("cifar10", "fashion32")
NAME = {"cifar10": "CIFAR", "fashion32": "Fashion"}
BREITEN = (1024, 4096, 16384)
ARME = ("A", "B", "C", "D", "E", "F", "G", "K")
SEEDS = tuple(range(2200, 2205))
TUNING_SEED = 2299
_BELEGT = set(range(2000, 2070)) | set(range(2090, 2100)) | set(range(2100, 2120)) | set(range(2150, 2160)) | set(range(900, 1001))
LR_GITTER = (3e-4, 1e-3, 3e-3, 1e-2)
LR_FAKTOR = 3
BATCH = 256
ALPHA = 0.05
NI_GRENZE = 0.01                 # Nichtunterlegenheit 1 pp
PARAM_FAKTOR = 100               # H1: >= 100x weniger Parameter in der Schicht
KOLLAPS = 0.05
# NK-Band breit: ein auf permutierten Labels trainiertes Netz sagt Cluster -> feste Labelzuordnung voraus, die Testgenauigkeit
# ist dann (Zahl der Fixpunkte)/10 mit großer Streuung je Lauf (gemessen 0,05-0,20 auf Tuning-Seeds, Mittel ≈ 0,10).
# Deshalb: gepoolter Mittelwert über A und B und alle Seeds (10 Läufe, je Arm eigene Permutation) je Datensatz in (0,05; 0,15);
# ein echter Test-Leck/Auswertungsfehler (> 0,15) wird weiter gefangen.
NK_BAND = (0.05, 0.15)
FAMILIE = ("H1B", "H1E", "H2B-C", "H2B-D", "H3")
ZEIT_BATCH = (1, 64)

# ---- Laufzeitprotokoll (VOR dem Lauf festgelegt, aus Zeitmessung 06.10., siehe LABORBUCH) ----
# Stufen: Epochen und welche Arme bei N=16384 mitlaufen. Welche Stufe gilt, steht in `PROFIL` (Kommandozeile/Umgebung
# I042_PROFIL, Standard "gekuerzt"); die Stufe wird in der Zwischendatei-Kennung festgeschrieben.
# Gemessen 06.10. (1 Thread, s je Epoche, alle Arme je Seed, Summe CIFAR+Fashion): voll ≈ 630 s -> 5 Seeds x 8 Epochen + Tuning ≈ 3 h (zu lang).
# „gekuerzt“ (Standard, VOR dem Lauf festgelegt): 5 Epochen; bei N=16384 nur die Arme der konfirmatorischen Hypothesen
# (A, B, C, D, E, K); F und G (nur beschreibend/Pareto) laufen bei N=16384 nicht. Schätzung ≈ 1,6 h bei 4 Prozessen.
PROFILE = {
    "voll": {"epochen": {"cifar10": 8, "fashion32": 8}, "arme_gross": ARME},
    "gekuerzt": {"epochen": {"cifar10": 5, "fashion32": 5}, "arme_gross": ("A", "B", "C", "D", "E", "K")},
}

HYPOTHESEN = {
    "H1": "N=16384: (B) oder (E) ≥ (A) − 1 pp (Nichtunterlegenheit, einseitig gepaart t, Holm) UND ≥ 100x weniger Schichtparameter. "
          "NICHT_UNTERLEGEN (B|E|B+E) | H0 (unterlegen) | HU.",
    "H2": "N=16384: (B) − (C) und (B) − (D), zweiseitig gepaart t, Holm. BORN_BESSER | KLASSISCH_BESSER | HU (je Gegner).",
    "H3": "N=16384: Pareto Genauigkeit gegen Parameter: (E) − (K) bei Budget(K) ≥ Budget(E), einseitig gepaart t, Holm. "
          "E_DOMINIERT | H0 (K besser) | HU.",
    "H4": "beschreibend + gepaart: (B) schneller als (A) bei N ≥ 4096 (Batch 1 und 64, 1 Thread, Median 7 Wdh.). B_SCHNELLER | A_SCHNELLER | GEMISCHT.",
    "NK": f"Negativkontrolle (A, B mit permutierten Trainingslabels, N=1024, je Datensatz über Arme und Seeds gepoolt): Mittel in {NK_BAND}. BESTANDEN | DURCHGEFALLEN (dann alles HU).",
    "B": "beschreibend: alle Arm-Mittel je (Datensatz, N), Parameter, FLOPs, Zeiten, Lernraten.",
}
FOLGEFRAGEN = {
    "H1 NICHT_UNTERLEGEN": "Strukturschicht trägt im Großen: nächste Stufe tieferes Netz oder Sprachmodell-Block.",
    "H1 H0": "Dichte Schicht bleibt bei echten Daten überlegen: Aussage auf lineare Köpfe (I-041) begrenzen.",
    "H2 BORN_BESSER": "z² schlägt klassische Nichtlinearität im Netz: Mechanismus (Zweitmoment) isolieren.",
}


# ------------------------------------------------------------------ Daten
def _cifar_laden():
    """uint8 (N,32,32,3), Labels; Cache daten/cifar10.npz. Quelle: daten/cifar10.tgz (fast.ai-Spiegel, PNG)."""
    cache = WURZEL / "daten" / "cifar10.npz"
    if not cache.exists():
        from PIL import Image
        tgz = WURZEL / "daten" / "cifar10.tgz"
        if not tgz.exists():
            import urllib.request
            urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", tgz)
        klassen = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        xs = {"train": [], "test": []}
        ys = {"train": [], "test": []}
        with tarfile.open(tgz) as t:
            for m in t:
                p = m.name.split("/")
                if m.isfile() and len(p) == 4 and p[1] in xs and p[2] in klassen:
                    xs[p[1]].append(np.asarray(Image.open(t.extractfile(m)).convert("RGB"), dtype=np.uint8))
                    ys[p[1]].append(klassen.index(p[2]))
        o = {f"x_{k}": np.stack(v) for k, v in xs.items()}
        o.update({f"y_{k}": np.array(v, dtype=np.int64) for k, v in ys.items()})
        np.savez_compressed(cache, **o)
    b = np.load(cache)
    # Die PNG-Ordner sind nach Klasse sortiert: EINMAL fest mischen (Seed 0), sonst sind Validierungs-/Rauchtest-Ausschnitte einklassig.
    pt = np.random.default_rng(0).permutation(len(b["y_train"]))
    pe = np.random.default_rng(1).permutation(len(b["y_test"]))
    return b["x_train"][pt], b["y_train"][pt], b["x_test"][pe], b["y_test"][pe]


def _fashion_laden():
    """uint8 (N,32,32,1) aus 28x28 bilinear (I-041), 60k Training / 10k Test."""
    from labor.gemeinsam import _roh
    x, y = _roh("fashion_mnist")
    xs = torch.tensor(x).reshape(-1, 1, 28, 28)
    xs = F.interpolate(xs, size=(32, 32), mode="bilinear", align_corners=False).clamp(0, 1)
    xs = (xs * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return xs[:60_000], y[:60_000], xs[60_000:], y[60_000:]


_CACHE = {}


_FLOAT = {}


def daten(ds, klein=False):
    """(xtr float (n,din) standardisiert, ytr, xte, yte); klein = Rauchtest (kleine Teilmengen). Je Prozess gecacht."""
    if (ds, klein) not in _FLOAT:
        _FLOAT[(ds, klein)] = _daten(ds, klein)
    return _FLOAT[(ds, klein)]


def _daten(ds, klein=False):
    if ds not in _CACHE:
        xtr, ytr, xte, yte = _cifar_laden() if ds == "cifar10" else _fashion_laden()
        _CACHE[ds] = (xtr, torch.tensor(ytr), xte, torch.tensor(yte))
    xtr, ytr, xte, yte = _CACHE[ds]
    if klein:
        xtr, ytr, xte, yte = xtr[:2560], ytr[:2560], xte[:1000], yte[:1000]
    a = torch.tensor(xtr).float().div(255).permute(0, 3, 1, 2)
    b = torch.tensor(xte).float().div(255).permute(0, 3, 1, 2)
    mu = a.mean((0, 2, 3), keepdim=True)
    sd = a.std((0, 2, 3), keepdim=True)
    return ((a - mu) / sd).reshape(len(a), -1), ytr, ((b - mu) / sd).reshape(len(b), -1), yte


# ------------------------------------------------------------------ Bausteine
def walsh_klein(n):
    H = torch.ones(1, 1)
    for _ in range(n):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(2 ** n)


def _faktoren(D):
    n = int(round(math.log2(D)))
    return 2 ** (n // 2), 2 ** (n - n // 2)


def pad2(d):
    return 1 << (d - 1).bit_length()


class Fastfood(nn.Module):
    """(H D2 H D1) x je Block; H = H_a ⊗ H_b (zwei kleine Matmuls, exakt dieselbe Matrix wie die volle Walsh-Matrix)."""

    def __init__(self, din, N, trainiert, gen):
        super().__init__()
        self.din, self.N, self.D = din, N, pad2(din)
        self.m = max(1, N // self.D)
        a, b = _faktoren(self.D)
        self.a, self.b = a, b
        self.register_buffer("Ha", walsh_klein(int(math.log2(a))))
        self.register_buffer("Hb", walsh_klein(int(math.log2(b))))
        v = lambda: (torch.randint(0, 2, (self.m, self.D), generator=gen) * 2 - 1).float()
        d1, d2 = v(), v()
        if trainiert:
            self.d1, self.d2 = nn.Parameter(d1), nn.Parameter(d2)
        else:
            self.register_buffer("d1", d1)
            self.register_buffer("d2", d2)
        self.register_buffer("idx", torch.randperm(self.D, generator=gen)[:N] if N < self.D else torch.arange(0))

    def wht(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.a, self.b)
        return (self.Ha @ x @ self.Hb).reshape(B, -1)

    def forward(self, x):
        B = x.shape[0]
        if x.shape[1] < self.D:
            x = F.pad(x, (0, self.D - x.shape[1]))
        z = x.unsqueeze(1) * self.d1                      # (B,m,D)
        z = self.wht(z.reshape(B * self.m, self.D)) .reshape(B, self.m, self.D) * self.d2
        z = self.wht(z.reshape(B * self.m, self.D)).reshape(B, self.m * self.D)
        return z[:, self.idx] if self.N < self.D else z

    def zaehle(self):
        return {"parameter": int(self.d1.numel() + self.d2.numel()) if isinstance(self.d1, nn.Parameter) else 0}


class Monarch(nn.Module):
    """g Blöcke (D/g -> N/g), Transposition, N/g Blöcke (g -> g). Parameter D·N/g + N·g."""

    def __init__(self, din, N, gen):
        super().__init__()
        self.D = pad2(din)
        self.N = N
        self.g = 1 << (int(math.log2(self.D)) // 2)
        g, D = self.g, self.D
        s = N // g
        self.W1 = nn.Parameter(torch.randn(g, D // g, s, generator=gen) / math.sqrt(D // g))
        self.W2 = nn.Parameter(torch.randn(s, g, g, generator=gen) / math.sqrt(g))

    def forward(self, x):
        B = x.shape[0]
        if x.shape[1] < self.D:
            x = F.pad(x, (0, self.D - x.shape[1]))
        h = torch.einsum("bgi,gio->bgo", x.reshape(B, self.g, self.D // self.g), self.W1)   # (B,g,s)
        h = h.transpose(1, 2)                                                                  # (B,s,g)
        return torch.einsum("bsi,sio->bso", h, self.W2).reshape(B, self.N)


class NiedrigerRang(nn.Module):
    def __init__(self, din, N, r, gen):
        super().__init__()
        self.r = r
        self.U = nn.Parameter(torch.randn(din, r, generator=gen) / math.sqrt(din))
        self.V = nn.Parameter(torch.randn(r, N, generator=gen) / math.sqrt(r))

    def forward(self, x):
        return (x @ self.U) @ self.V


def param_F(din, N):
    D = pad2(din)
    g = 1 << (int(math.log2(D)) // 2)
    return D * N // g + N * g


def param_E(din, N):
    D = pad2(din)
    return 2 * max(1, N // D) * D


def rang_G(din, N):
    return max(1, round(param_F(din, N) / (din + N)))


def rang_K(din, N):
    return max(1, math.ceil(param_E(din, N) / (din + N)))


def param_schicht(arm, din, N):
    """Trainierte Parameter der ersten Schicht (ohne Kopf)."""
    if arm == "A":
        return din * N + N
    if arm in ("B", "C", "D"):
        return 0
    if arm == "E":
        return param_E(din, N)
    if arm == "F":
        return param_F(din, N)
    if arm == "G":
        return rang_G(din, N) * (din + N)
    return rang_K(din, N) * (din + N)


NICHTLIN = {"A": F.relu, "B": torch.square, "C": torch.abs, "D": F.relu, "E": torch.square, "F": torch.square,
            "G": torch.square, "K": torch.square}


class Netz(nn.Module):
    def __init__(self, arm, din, N, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed * 31 + N)
        self.arm, self.act = arm, NICHTLIN[arm]
        if arm == "A":
            self.mix = nn.Linear(din, N)
        elif arm in ("B", "C", "D", "E"):
            self.mix = Fastfood(din, N, arm == "E", gen)
        elif arm == "F":
            self.mix = Monarch(din, N, gen)
        else:
            self.mix = NiedrigerRang(din, N, rang_G(din, N) if arm == "G" else rang_K(din, N), gen)
        self.kopf = nn.Linear(N, 10)
        if arm in ("B", "C", "D"):
            for p in self.mix.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        if self.arm in ("B", "C", "D"):
            with torch.no_grad():
                z = self.act(self.mix(x))
        else:
            z = self.act(self.mix(x))
        return self.kopf(F.layer_norm(z, z.shape[-1:]))


def flops(arm, din, N):
    """Gleitkommaoperationen je Bild, nur Vorwärtsrechnung (Mul+Add = 2). 'fwht' = algorithmisch (2 FWHT je Block),
    'impl' = wie hier gerechnet (Kronecker-Matmuls). Kopf 2·N·10 bei beiden."""
    D = pad2(din)
    kopf = 20 * N
    if arm == "A":
        return {"impl": 2 * din * N + kopf, "fwht": 2 * din * N + kopf}
    if arm in ("B", "C", "D", "E"):
        m = max(1, N // D)
        a, b = _faktoren(D)
        fw = m * (2 * D * int(math.log2(D)) + 2 * D)
        im = m * (2 * 2 * D * (a + b) + 2 * D)
        return {"impl": im + kopf, "fwht": fw + kopf}
    if arm == "F":
        g = 1 << (int(math.log2(D)) // 2)
        f = 2 * (D * N // g + N * g)
        return {"impl": f + kopf, "fwht": f + kopf}
    r = rang_G(din, N) if arm == "G" else rang_K(din, N)
    f = 2 * r * (din + N)
    return {"impl": f + kopf, "fwht": f + kopf}


# ------------------------------------------------------------------ Training
def trainiere(ds, arm, N, lr, seed, epochen, klein=False, val=False, permutiert=False):
    """val=True: Tuning (Training auf ersten 90 %, Bewertung auf letzten 10 %); sonst volles Training, Bewertung auf Test."""
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = daten(ds, klein)
    if val:
        nv = len(ytr) // 10
        xtr, ytr, xte, yte = xtr[:-nv], ytr[:-nv], xtr[-nv:], ytr[-nv:]
    g = torch.Generator().manual_seed(seed + 5)
    if permutiert:       # je Arm eigene Permutation (nicht korrelierte Kontrollen)
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 5 + 100 * ord(arm)))]
    torch.manual_seed(seed)
    netz = Netz(arm, xtr.shape[1], N, seed)
    opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=lr)
    n = len(ytr)
    schritte = epochen * math.ceil(n / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, schritte)
    verlust = float("nan")
    for _ in range(epochen):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            l = F.cross_entropy(netz(xtr[j]), ytr[j])
            if not torch.isfinite(l):
                return {"acc": 0.1, "verlust": float("nan"), "instabil": True}
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
            verlust = float(l.detach())
    netz.eval()
    with torch.no_grad():
        richtig = sum(int((netz(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum()) for i in range(0, len(yte), 1000))
    return {"acc": richtig / len(yte), "verlust": verlust, "instabil": False}


def _tuning_arbeiter(a):
    (ds, arm, N), s, lr, ep, klein = a
    return trainiere(ds, arm, N, lr, s, ep, klein=klein, val=True)


def tuning(ds, N, arme, epochen, klein, tseed):
    """Lernrate je Arm (gittersuche_stabil, ein Tuning-Seed)."""
    schl = [(ds, a, N) for a in arme]
    info, _ = gittersuche_stabil(schl, lambda k, lr: [(k, tseed, lr, epochen, klein)], lambda aufg: parallel(_tuning_arbeiter, aufg),
                                 gitter=LR_GITTER, faktor=LR_FAKTOR, kollaps=KOLLAPS)
    return {k[1]: {"lr": v["lr"], "am_rand": v["am_rand"], "abgestuft": v["abgestuft"],
                   "gitter": {lr: g["je_seed"][0] for lr, g in v["gitter"].items()}} for k, v in info.items()}


def _arbeiter(a):
    """Eine (Datensatz, N, Seed)-Einheit: alle Arme des Profils nacheinander (Wiederaufnahme auf dieser Ebene)."""
    ds, N, s, arme, lrs, epochen, klein = a
    t0 = time.time()
    z = {"seed": s, "datensatz": ds, "N": N}
    for arm in arme:
        z[arm] = trainiere(ds, arm, N, lrs[arm]["lr"], s, epochen, klein=klein)["acc"]
    if N == BREITEN[0]:
        for arm in ("A", "B"):
            z[f"{arm}_perm"] = trainiere(ds, arm, N, lrs[arm]["lr"], s, epochen, klein=klein, permutiert=True)["acc"]
    z["sekunden"] = time.time() - t0
    return f"einheit|{ds}|{N}|{s}", z


# ------------------------------------------------------------------ Wiederaufnahme (je Datensatz, N, Seed)
def _lade_zwischen(pfad: Path, kennung: dict) -> dict:
    if not pfad.exists():
        return {}
    inhalt = json.loads(pfad.read_text())
    if inhalt.get("kennung") != kennung:
        raise ValueError(f"Zwischendatei {pfad} stammt aus anderen Einstellungen -- nicht mischen; umbenennen")
    return inhalt["zeilen"]


def _schreibe_zwischen(pfad: Path, kennung: dict, zeilen: dict):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps({"karte": kennung.get("karte", KARTE), "kennung": kennung, "zeilen": zeilen}, ensure_ascii=False))
    os.replace(tmp, pfad)


def _rechne(aufgaben, pfad: Path, kennung: dict, fn=None, prozesse=None, schluessel=None) -> dict:
    import multiprocessing as mp
    from labor.gemeinsam import _init_arbeiter
    fn = fn or _arbeiter
    schluessel = schluessel or (lambda a: f"einheit|{a[0]}|{a[1]}|{a[2]}")
    zeilen = _lade_zwischen(pfad, kennung)
    offen = [a for a in aufgaben if schluessel(a) not in zeilen]
    if len(offen) < len(aufgaben):
        print(f"  Wiederaufnahme: {len(aufgaben) - len(offen)} von {len(aufgaben)} aus {pfad.name}", flush=True)
    prozesse = prozesse or int(os.environ.get("INSTITUT_PROZESSE", max(1, (os.cpu_count() or 2) - 1)))

    def ablegen(erg):
        k, z = erg
        zeilen[k] = z
        _schreibe_zwischen(pfad, kennung, zeilen)
        print(f"  fertig: {k} ({len(zeilen)} in Zwischendatei)", flush=True)

    if prozesse <= 1:
        _init_arbeiter()
        for a in offen:
            ablegen(fn(a))
    elif offen:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        with mp.get_context("spawn").Pool(prozesse, initializer=_init_arbeiter) as pool:
            for erg in pool.imap_unordered(fn, offen):
                ablegen(erg)
    return zeilen


# ------------------------------------------------------------------ Zeitmessung (Inferenz, 1 Thread, ganzes Netz)
def _stoppuhr(f, mindest=0.05, wdh=7):
    f()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        dt = time.perf_counter() - t0
        if dt >= mindest or n >= 1024:
            break
        n *= 4
    ts = [dt / n]
    for _ in range(wdh - 1):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        ts.append((time.perf_counter() - t0) / n)
    return float(np.median(ts))


def zeit_messung(dins, breiten=BREITEN, arme=ARME, batches=ZEIT_BATCH):
    torch.set_num_threads(1)
    aus = []
    for ds, din in dins.items():
        for N in breiten:
            for arm in arme:
                netz = Netz(arm, din, N, 0).eval()
                z = {"datensatz": ds, "N": N, "arm": arm, "flops": flops(arm, din, N), "parameter_schicht": param_schicht(arm, din, N)}
                for B in batches:
                    x = torch.randn(B, din)
                    with torch.no_grad():
                        z[f"batch{B}_s"] = _stoppuhr(lambda: netz(x))
                aus.append(z)
                del netz
    return aus


def _zeit_arbeiter(a):
    return zeit_messung(*a)


def zeit_im_ein_thread(dins, breiten=BREITEN, arme=ARME, batches=ZEIT_BATCH):
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_zeit_arbeiter, ((dins, tuple(breiten), tuple(arme), tuple(batches)),))


def zeit_urteil(zm) -> dict:
    """H4 beschreibend: B gegen A bei N >= 4096, je Datensatz und Batch (Verhältnis A-Zeit / B-Zeit)."""
    u = {}
    for ds in sorted({z["datensatz"] for z in zm}):
        v = {}
        for B in ZEIT_BATCH:
            for N in sorted({z["N"] for z in zm}):
                a = next((z for z in zm if z["datensatz"] == ds and z["N"] == N and z["arm"] == "A"), None)
                b = next((z for z in zm if z["datensatz"] == ds and z["N"] == N and z["arm"] == "B"), None)
                if a and b:
                    v[f"N{N}_batch{B}"] = a[f"batch{B}_s"] / b[f"batch{B}_s"]
        gross = [w for k, w in v.items() if int(k.split("_")[0][1:]) >= 4096]
        u[ds] = {"verhaeltnis_A_durch_B": {k: round(w, 3) for k, w in v.items()}, "gross": gross}
    return u


def _h4(zu) -> dict:
    out = {}
    for ds, r in zu.items():
        g = r["gross"]
        if not g:
            out[ds] = "HU (keine Zeitwerte N>=4096)"
        elif all(w > 1 for w in g):
            out[ds] = f"B_SCHNELLER (Verhältnis A/B bei N≥4096: min {min(g):.2f}, max {max(g):.2f})"
        elif all(w < 1 for w in g):
            out[ds] = f"A_SCHNELLER (Verhältnis A/B bei N≥4096: min {min(g):.2f}, max {max(g):.2f})"
        else:
            out[ds] = f"GEMISCHT (Verhältnis A/B bei N≥4096: min {min(g):.2f}, max {max(g):.2f})"
    return out


# ------------------------------------------------------------------ Statistik und Auswertung
def _t_p(d, seite):
    """Gepaarter t-Test: seite 'gr' (Mittel>0), 'kl' (<0), 'zwei'."""
    from scipy.stats import t as tv
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return 1.0
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() > 0 and seite == "gr" else (0.0 if d.mean() < 0 and seite == "kl" else 1.0)
    t = d.mean() / (sd / math.sqrt(n))
    if seite == "gr":
        return float(tv.sf(t, n - 1))
    if seite == "kl":
        return float(tv.cdf(t, n - 1))
    return float(2 * tv.sf(abs(t), n - 1))


def _stat(d, versatz=0.0) -> dict:
    """Differenz d (pro Seed); versatz: für Nichtunterlegenheit wird gegen −versatz geprüft (d + versatz > 0)."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return {"n": int(len(d))}
    from scipy.stats import t as tv
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(tv.ppf(0.95, len(d) - 1))
    return {"mittel_pp": m * 100, "sd_pp": float(d.std(ddof=1)) * 100, "ki_pp": [(m - q * se) * 100, (m + q * se) * 100],
            "positiv": int((d > 0).sum()), "n": int(len(d)),
            "p_gr": _t_p(d + versatz, "gr"), "p_kl": _t_p(d + versatz, "kl"), "p_zwei": _t_p(d, "zwei"),
            "wilcoxon_zweiseitig": wilcoxon_schnell(d) if len(set(np.round(d, 12))) > 1 else 1.0}


def _arr(zz, k):
    return np.array([z[k] for z in zz], dtype=float)


def kollaps_seeds(zz, k):
    w = _arr(zz, k)
    med = float(np.median(w))
    return [z["seed"] for z in zz if not (z[k] >= med - KOLLAPS)]


def _ds_auswertung(zz_n: dict, param) -> dict:
    """zz_n[N] = Liste der Seed-Zeilen; param[N][arm] = Schichtparameter."""
    r = {"arme": {}, "kollaps": {}}
    for N, zz in zz_n.items():
        for arm in ARME:
            if arm in zz[0]:
                r["arme"][f"{arm}_{N}"] = zusammenfassung(_arr(zz, arm))
                r["kollaps"][f"{arm}_{N}"] = kollaps_seeds(zz, arm)
    r["negativ"] = {}
    z0 = zz_n.get(BREITEN[0])
    if z0 and "A_perm" in z0[0]:
        r["negativ"] = {a: float(_arr(z0, f"{a}_perm").mean()) for a in ("A", "B")}
    G = zz_n.get(BREITEN[-1])
    r["diff"] = {}
    if G:
        A = lambda k: _arr(G, k)
        r["diff"]["H1B"] = _stat(A("B") - A("A"), versatz=NI_GRENZE)
        r["diff"]["H1E"] = _stat(A("E") - A("A"), versatz=NI_GRENZE)
        if "C" in G[0]:
            r["diff"]["H2B-C"] = _stat(A("B") - A("C"))
            r["diff"]["H2B-D"] = _stat(A("B") - A("D"))
        if "K" in G[0]:
            r["diff"]["H3"] = _stat(A("E") - A("K"))
        r["beschreibend"] = {f"{x} - {y} @{N}": _stat(_arr(z, x) - _arr(z, y)) for N, z in zz_n.items()
                             for x, y in (("B", "A"), ("E", "A"), ("F", "A"), ("E", "G"), ("B", "C"), ("B", "D")) if x in z[0] and y in z[0]}
    prim = {h: (r["diff"][h]["p_gr"] if h in ("H1B", "H1E", "H3") else r["diff"][h]["p_zwei"]) if "p_gr" in r["diff"][h] else 1.0
            for h in FAMILIE if h in r["diff"]}
    r["p_holm"] = holm(prim) if prim else {}
    r["parameter"] = {str(N): p for N, p in param.items()}
    return r


def auswertung(zeilen: dict, params: dict) -> dict:
    return {"datensaetze": {ds: _ds_auswertung(zn, params[ds]) for ds, zn in zeilen.items()}}


# ------------------------------------------------------------------ Urteile (Kriterien im Code)
def _f(x):
    return f"{x:+.2f}".replace(".", ",")


def _nk(a) -> str:
    ok = all(r["negativ"] and NK_BAND[0] <= sum(r["negativ"].values()) / len(r["negativ"]) <= NK_BAND[1]
             for r in a["datensaetze"].values())
    return "BESTANDEN" if ok else "DURCHGEFALLEN"


def _zahlen(r, h):
    d = r["diff"][h]
    return (f"d {_f(d['mittel_pp'])} pp, 90-%-KI [{_f(d['ki_pp'][0])}; {_f(d['ki_pp'][1])}], n {d['n']}, "
            f"p_Holm {r['p_holm'][h]:.3g}")


def _koll(r, spalten):
    return sorted({s for k in spalten for s in r["kollaps"].get(k, [])})


def _urteil_h1(r, nk, ds) -> str:
    G = BREITEN[-1]
    if "H1B" not in r["diff"] or "mittel_pp" not in r["diff"]["H1B"]:
        return "HU (N=16384 fehlt oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    koll = _koll(r, [f"A_{G}", f"B_{G}", f"E_{G}"])
    if koll:
        return f"HU (Instabilitaet/Kollaps in Seeds {koll})"
    pA = int(r["parameter"][str(G)]["A"])
    ok = []
    for h, arm in (("H1B", "B"), ("H1E", "E")):
        pa = int(r["parameter"][str(G)][arm])
        ratio_ok = pa == 0 or pA / pa >= PARAM_FAKTOR
        if r["p_holm"][h] < ALPHA and ratio_ok:
            ok.append(arm)
    zahl = "; ".join(f"{h}: {_zahlen(r, h)}" for h in ("H1B", "H1E"))
    if ok:
        return f"NICHT_UNTERLEGEN ({'+'.join(ok)}; Parameter A {pA}; {zahl})"
    # Unterlegenheit: beide Arme sicher schlechter als A − 1 pp (einseitig, Holm-bereinigt über dieselbe Familie nicht nötig: konservativ ungeholmt x2)
    if all(r["diff"][h]["p_kl"] < ALPHA / 2 for h in ("H1B", "H1E")):
        return f"H0 (beide unterlegen um mehr als 1 pp; {zahl})"
    return f"HU ({zahl})"


def _urteil_h2(r, nk, h, gegner) -> str:
    G = BREITEN[-1]
    if h not in r["diff"] or "mittel_pp" not in r["diff"][h]:
        return "HU (N=16384 fehlt oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    koll = _koll(r, [f"B_{G}", f"{gegner}_{G}"])
    if koll:
        return f"HU (Instabilitaet/Kollaps in Seeds {koll})"
    d = r["diff"][h]
    z = _zahlen(r, h)
    if r["p_holm"][h] < ALPHA:
        return f"{'BORN_BESSER' if d['mittel_pp'] > 0 else 'KLASSISCH_BESSER'} ({z})"
    return f"HU ({z})"


def _urteil_h3(r, nk) -> str:
    G = BREITEN[-1]
    if "H3" not in r["diff"] or "mittel_pp" not in r["diff"]["H3"]:
        return "HU (Arm K bei N=16384 fehlt oder zu wenige Seeds)"
    if nk != "BESTANDEN":
        return "HU (Negativkontrolle durchgefallen)"
    koll = _koll(r, [f"E_{G}", f"K_{G}"])
    if koll:
        return f"HU (Instabilitaet/Kollaps in Seeds {koll})"
    if int(r["parameter"][str(G)]["E"]) > int(r["parameter"][str(G)]["K"]):
        return "HU (Budget von K kleiner als E -- Kontrolle ungültig)"
    d = r["diff"]["H3"]
    z = _zahlen(r, "H3")
    if r["p_holm"]["H3"] < ALPHA:
        return f"E_DOMINIERT ({z})"
    if d["p_kl"] < ALPHA and d["mittel_pp"] < 0:
        return f"H0 (K besser bei gleichem Budget; {z})"
    return f"HU ({z})"


def urteil(a: dict, zeit=None) -> dict:
    nk = _nk(a)
    u = {"NK": f"{nk} (Mittel permutierte Labels A/B: " + ", ".join(
        f"{NAME[ds]} {k} {v:.3f}" for ds, r in a["datensaetze"].items() for k, v in r["negativ"].items()) + f"; Band {NK_BAND})"}
    for ds, r in a["datensaetze"].items():
        n = NAME[ds]
        u[f"H1-{n}"] = _urteil_h1(r, nk, ds)
        u[f"H2B-C-{n}"] = _urteil_h2(r, nk, "H2B-C", "C")
        u[f"H2B-D-{n}"] = _urteil_h2(r, nk, "H2B-D", "D")
        u[f"H3-{n}"] = _urteil_h3(r, nk)
        g = lambda k: _f(r["arme"][k]["mittel"] * 100).lstrip("+") if k in r["arme"] else "n/a"
        u[f"B-{n}"] = "beschreibend: Genauigkeit % bei N=16384 " + ", ".join(f"{x} {g(f'{x}_{BREITEN[-1]}')}" for x in ARME) + \
                      "; Rest in auswertung"
    if zeit is not None:
        for ds, s in _h4(zeit_urteil(zeit)).items():
            u[f"H4-{NAME[ds]}"] = s
    return u


# ------------------------------------------------------------------ Vortest
def _synth(rng, off=None, sd=0.001, n=5, koll=None, nk=0.10, mitK=True, params=None):
    base = {"A": 0.560, "B": 0.555, "C": 0.540, "D": 0.535, "E": 0.560, "F": 0.545, "G": 0.520, "K": 0.530}
    base.update(off or {})
    zeilen = {}
    for N in BREITEN:
        zz = []
        for i in range(n):
            gem = rng.normal(0, 0.003)
            z = {"seed": 2200 + i, "N": N}
            for arm, v in base.items():
                if arm == "K" and not mitK:
                    continue
                z[arm] = v + gem + rng.normal(0, sd)
            if N == BREITEN[0]:
                z["A_perm"] = nk + rng.normal(0, 0.002)
                z["B_perm"] = nk + rng.normal(0, 0.002)
            if koll and i == 0 and N == BREITEN[-1]:
                z[koll] -= 0.2
            zz.append(z)
        zeilen[N] = zz
    return zeilen


def _synth_param(din=3072):
    return {N: {a: param_schicht(a, din, N) for a in ARME} for N in BREITEN}


def _u(rng, p=None, **kw):
    off_p = kw.pop("param_off", None)
    pr = _synth_param()
    if off_p:
        for N in pr:
            pr[N].update(off_p)
    z = _synth(rng, **kw)
    return urteil(auswertung({"cifar10": z}, {"cifar10": pr}))


def _vortest_bausteine() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER: {name}", flush=True)
            ok = False

    # Fastfood-Block = volle Matrix H D2 H D1
    for D in (256, 1024):
        gen = torch.Generator().manual_seed(1)
        ff = Fastfood(D, D, False, gen)
        x = torch.randn(4, D)
        H = walsh_klein(int(math.log2(D)))
        U = H @ torch.diag(ff.d2[0]) @ H @ torch.diag(ff.d1[0])
        pruefe(f"Fastfood = Matrix D={D}", torch.allclose(ff(x), x @ U.T, atol=1e-4))
        pruefe(f"Fastfood orthogonal D={D}", float((U @ U.T - torch.eye(D)).abs().max()) < 1e-4)
    # Auffüllen (3072 -> 4096), Blöcke (N > D), Auswahl (N < D)
    gen = torch.Generator().manual_seed(2)
    ff = Fastfood(3072, 8192, False, gen)
    pruefe("Fastfood 2 Blöcke Form", ff(torch.randn(3, 3072)).shape == (3, 8192) and ff.m == 2)
    ff = Fastfood(3072, 1024, False, gen)
    pruefe("Fastfood Auswahl Form", ff(torch.randn(3, 3072)).shape == (3, 1024))
    # Normerhalt (orthonormal, 1 Block)
    ff = Fastfood(1024, 1024, False, gen)
    x = torch.randn(5, 1024)
    pruefe("Fastfood normerhaltend", torch.allclose(ff(x).norm(dim=1), x.norm(dim=1), rtol=1e-4))
    # Parameterzählung stimmt mit tatsächlichen Parametern
    for arm in ARME:
        for din, N in ((3072, 1024), (1024, 4096)):
            netz = Netz(arm, din, N, 0)
            echt = sum(p.numel() for p in netz.mix.parameters() if p.requires_grad)
            pruefe(f"Parameterzählung {arm} {din}/{N}", echt == param_schicht(arm, din, N))
    pruefe("B/C/D fest (0 Parameter)", all(param_schicht(a, 3072, 16384) == 0 for a in "BCD"))
    pruefe("Budget K >= E", all(param_schicht("K", d, N) >= param_schicht("E", d, N) for d in (3072, 1024) for N in BREITEN))
    pruefe("Budget G ~ F (Faktor <= 2)", all(0.5 <= param_schicht("G", d, N) / param_schicht("F", d, N) <= 2.0
                                             for d in (3072, 1024) for N in BREITEN))
    pruefe("Parameterverhältnis A/E CIFAR 16384 >= 100", param_schicht("A", 3072, 16384) / param_schicht("E", 3072, 16384) >= 100)
    # Monarch läuft, Gradient fließt zu beiden Stufen; Trainierbar-Flags
    m = Netz("F", 1024, 1024, 0)
    m(torch.randn(2, 1024)).sum().backward()
    pruefe("Monarch Gradient", m.mix.W1.grad is not None and float(m.mix.W2.grad.abs().sum()) > 0)
    e = Netz("E", 1024, 1024, 0)
    e(torch.randn(2, 1024)).sum().backward()
    pruefe("E trainiert Diagonalen", e.mix.d1.grad is not None and float(e.mix.d1.grad.abs().sum()) > 0)
    b = Netz("B", 1024, 1024, 0)
    pruefe("B hat nur Kopf trainierbar", {n for n, p in b.named_parameters() if p.requires_grad} == {"kopf.weight", "kopf.bias"})
    # FLOPs-Modell
    pruefe("FLOPs A > B (N=16384)", flops("A", 3072, 16384)["impl"] > flops("B", 3072, 16384)["impl"])
    # Kurztrainings: Positivkontrolle der Trainierbarkeit (Verlust fällt) -- synthetische lineare Aufgabe
    torch.manual_seed(0)
    X = torch.randn(512, 64)
    Y = (X[:, :10].sum(1) > 0).long()
    for arm in ("A", "B", "E", "F", "G"):
        netz = Netz(arm, 64, 256, 0)
        opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=3e-3)
        l0 = float(F.cross_entropy(netz(X), Y))
        for _ in range(60):
            opt.zero_grad()
            F.cross_entropy(netz(X), Y).backward()
            opt.step()
        pruefe(f"Trainierbar {arm}", float(F.cross_entropy(netz(X), Y)) < 0.6 * l0)
    # Seeds
    pruefe("Seeds frisch", len(SEEDS) == 5 and not (set(SEEDS) & _BELEGT) and TUNING_SEED not in SEEDS and TUNING_SEED not in _BELEGT)
    # Statistik: Nichtunterlegenheit mit Versatz
    d = np.array([-0.004, -0.006, -0.005, -0.004, -0.006])
    pruefe("NI-Test: -0,5 pp ist nichtunterlegen", _stat(d, NI_GRENZE)["p_gr"] < 0.01)
    pruefe("NI-Test: -2 pp ist es nicht", _stat(d - 0.015, NI_GRENZE)["p_gr"] > 0.9 and _stat(d - 0.015, NI_GRENZE)["p_kl"] < 0.05)
    h = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    pruefe("Holm", abs(h["a"] - 0.03) < 1e-12 and abs(h["b"] - 0.04) < 1e-12)
    # Zeitmessung liefert Werte, H4-Urteil
    zm = zeit_messung({"x": 64}, breiten=(256, 512), arme=("A", "B"), batches=(1, 64))
    pruefe("Zeitmessung", all(z["batch1_s"] > 0 and z["batch64_s"] > 0 for z in zm))
    fake = [{"datensatz": "x", "N": N, "arm": arm, "batch1_s": t, "batch64_s": t} for N in (4096, 16384)
            for arm, t in (("A", 2.0), ("B", 1.0))]
    pruefe("H4 B_SCHNELLER", _h4(zeit_urteil(fake))["x"].startswith("B_SCHNELLER"))
    pruefe("H4 A_SCHNELLER", _h4(zeit_urteil([dict(z, batch1_s=3.0 - z["batch1_s"], batch64_s=3.0 - z["batch64_s"]) for z in fake]))["x"].startswith("A_SCHNELLER"))
    gem = [dict(z, batch1_s=(z["batch1_s"] if z["N"] == 4096 else 3.0 - z["batch1_s"])) for z in fake]
    pruefe("H4 GEMISCHT", _h4(zeit_urteil(gem))["x"].startswith("GEMISCHT"))
    return ok


def _fn_test(a):
    return f"einheit|{a[0]}|{a[1]}|{a[2]}", {"v": a[1] * 2}


def _vortest_wiederaufnahme() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as t:
        pfad = Path(t) / "z.json"
        kennung = {"k": 1}
        aufg = [("cifar10", 1024, i) for i in range(4)]
        z1 = _rechne(aufg[:2], pfad, kennung, fn=_fn_test, prozesse=1)
        z2 = _rechne(aufg, pfad, kennung, fn=_fn_test, prozesse=1)
        ok &= len(z1) == 2 and len(z2) == 4 and z2["einheit|cifar10|1024|1"] == z1["einheit|cifar10|1024|1"]
        try:
            _rechne(aufg, pfad, {"k": 2}, fn=_fn_test, prozesse=1)
            ok = False
        except ValueError:
            pass
        ok &= not pfad.with_name(pfad.name + ".tmp").exists()
    if not ok:
        print("  Vortest FEHLER: Wiederaufnahme", flush=True)
    return ok


def _vortest_urteile() -> bool:
    """Jeden Urteilszweig erzwingen (synthetische Daten, n=5)."""
    rng = np.random.default_rng(11)
    ok = True
    gesehen = set()

    def erwarte(name, u, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k.split("-")[0], w.split(" (")[0]))
            if not u[k].startswith(w):
                print(f"  Vortest FEHLER {name}: {k} soll {w}, ist {u[k]}", flush=True)
                ok = False

    C = "-CIFAR"
    erwarte("Normalfall", _u(rng), **{"NK": "BESTANDEN", "H1" + C: "NICHT_UNTERLEGEN", "H2B-C" + C: "BORN_BESSER",
                                      "H2B-D" + C: "BORN_BESSER", "H3" + C: "E_DOMINIERT"})
    erwarte("H1 nur E", _u(rng, off={"B": 0.52}), **{"H1" + C: "NICHT_UNTERLEGEN (E"})
    erwarte("H1 nur B", _u(rng, off={"E": 0.52}), **{"H1" + C: "NICHT_UNTERLEGEN (B"})
    erwarte("H1 beide unterlegen", _u(rng, off={"B": 0.50, "E": 0.50}), **{"H1" + C: "H0 (beide unterlegen"})
    erwarte("H1 Parameter reichen nicht", _u(rng, param_off={"B": 10 ** 6, "E": 10 ** 6}), **{"H1" + C: "HU"})
    erwarte("H2 klassisch besser", _u(rng, off={"C": 0.58, "D": 0.58}), **{"H2B-C" + C: "KLASSISCH_BESSER",
                                                                          "H2B-D" + C: "KLASSISCH_BESSER"})
    erwarte("H3 K besser", _u(rng, off={"K": 0.60}), **{"H3" + C: "H0 (K besser"})
    erwarte("Rauschen", _u(rng, sd=0.02, off={"B": 0.56, "C": 0.56, "D": 0.56, "K": 0.56}),
            **{"H1" + C: "HU", "H2B-C" + C: "HU", "H2B-D" + C: "HU", "H3" + C: "HU"})
    erwarte("NK faellt", _u(rng, nk=0.30), **{"NK": "DURCHGEFALLEN", "H1" + C: "HU (Negativkontrolle",
                                              "H2B-C" + C: "HU (Negativkontrolle", "H3" + C: "HU (Negativkontrolle"})
    erwarte("Kollaps B", _u(rng, koll="B"), **{"H1" + C: "HU (Instabilitaet", "H2B-C" + C: "HU (Instabilitaet"})
    erwarte("Budget K zu klein", _u(rng, param_off={"K": 10}), **{"H3" + C: "HU (Budget"})
    erwarte("ohne K (gekuerzt)", _u(rng, mitK=False), **{"H3" + C: "HU (Arm K"})
    # Mutationstest: Richtung muss das Urteil ändern
    a = _u(rng, off={"C": 0.50})["H2B-C" + C].split(" ")[0]
    b = _u(rng, off={"C": 0.60})["H2B-C" + C].split(" ")[0]
    if a == b:
        print("  Vortest FEHLER: H2 unterscheidet die Richtung nicht", flush=True)
        ok = False
    erlaubt = ("NICHT_UNTERLEGEN", "BORN_BESSER", "KLASSISCH_BESSER", "E_DOMINIERT", "HU", "H0", "BESTANDEN",
               "DURCHGEFALLEN", "beschreibend")
    u = _u(rng)
    if not all(isinstance(v, str) and v.startswith(erlaubt) for v in u.values()):
        print("  Vortest FEHLER: Urteil ausserhalb des Vokabulars", flush=True)
        ok = False
    soll = {"H1": {"NICHT_UNTERLEGEN", "H0", "HU"}, "H2B": {"BORN_BESSER", "KLASSISCH_BESSER", "HU"},
            "H3": {"E_DOMINIERT", "H0", "HU"}}
    for h, ws in soll.items():
        erreicht = {w for k, w in gesehen if k == h}
        if not ws <= erreicht:
            print(f"  Vortest FEHLER: Zweige {ws - erreicht} fuer {h} nicht erzwungen", flush=True)
            ok = False
    return ok


def vortest() -> bool:
    ok = _vortest_bausteine()
    ok &= _vortest_wiederaufnahme()
    ok &= _vortest_urteile()
    ok &= _vortest_r()
    return bool(ok)


# ------------------------------------------------------------------ Lauf
def _dins():
    return {"cifar10": 3072, "fashion32": 1024}


def lauf(schnell=False, datensaetze=DATENSAETZE, prozesse=None, zwischen=None, zeit=True, profil=None, breiten=None):
    profil = profil or os.environ.get("I042_PROFIL", "gekuerzt")
    P = PROFILE[profil]
    datensaetze = tuple(datensaetze)
    if schnell:
        seeds, tseed, klein, ep = (TUNING_SEED,), TUNING_SEED, True, {ds: 2 for ds in datensaetze}   # Rauchtest NUR Tuning-Seed
        breiten = tuple(breiten or (BREITEN[0],))
    else:
        seeds, tseed, klein, ep = SEEDS, TUNING_SEED, False, P["epochen"]
        breiten = tuple(breiten or BREITEN)
        if set(seeds) & _BELEGT or tseed in seeds:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE, "version": 1, "profil": profil, "epochen": ep, "lr_gitter": list(LR_GITTER), "tuning_seed": tseed,
               "klein": klein, "datensaetze": list(datensaetze), "breiten": list(breiten), "arme_gross": list(P["arme_gross"])}
    t0 = time.time()
    zw = _lade_zwischen(pfad, kennung)
    arme_von = lambda N: ARME if N < BREITEN[-1] else P["arme_gross"]
    hp = zw.setdefault("tuning", {})
    for ds in datensaetze:
        for N in breiten:
            k = f"{ds}|{N}"
            if k in hp:
                print(f"  Wiederaufnahme: Tuning {k}", flush=True)
                continue
            hp[k] = tuning(ds, N, arme_von(N), ep[ds], klein, tseed)
            _schreibe_zwischen(pfad, kennung, zw)
            print(f"  Tuning fertig: {k} {({a: v['lr'] for a, v in hp[k].items()})}", flush=True)
    aufg = [(ds, N, s, arme_von(N), hp[f"{ds}|{N}"], ep[ds], klein) for N in breiten for ds in datensaetze for s in seeds]
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zeilen = {ds: {N: [alle[f"einheit|{ds}|{N}|{s}"] for s in seeds] for N in breiten} for ds in datensaetze}
    zm = None
    if zeit:
        if "zeit" in alle:
            zm = alle["zeit"]
        else:
            zm = zeit_im_ein_thread({ds: _dins()[ds] for ds in datensaetze}, breiten if schnell else BREITEN)
            alle["zeit"] = zm
            _schreibe_zwischen(pfad, kennung, alle)
    params = {ds: {N: {a: param_schicht(a, _dins()[ds], N) for a in ARME} for N in breiten} for ds in datensaetze}
    e = {"abteilung": "Struktur", "modul": "struktur_gross", "karte": KARTE, "version": 1, "profil": profil,
         "hypothesen": HYPOTHESEN, "folgefragen": FOLGEFRAGEN, "seeds": list(seeds), "tuning_seed": tseed, "tuning": hp,
         "einstellungen": {"epochen": ep, "batch": BATCH, "lr_gitter": list(LR_GITTER), "ni_grenze_pp": NI_GRENZE * 100,
                           "flops": {ds: {N: {a: flops(a, _dins()[ds], N) for a in ARME} for N in breiten} for ds in datensaetze}},
         "zwischendatei": str(pfad), "zeilen": {ds: {str(N): z for N, z in zn.items()} for ds, zn in zeilen.items()},
         "sekunden": time.time() - t0}
    e["auswertung"] = auswertung(zeilen, params)
    if zm is not None:
        e["zeitmessung"] = {"roh": zm, "zusammenfassung": zeit_urteil(zm)}
    e["urteil"] = urteil(e["auswertung"], zm)
    return e


# ================================================================== I-042-r: vorregistrierte Replikation (mehr Seeds)
# Nur die tragenden Vergleiche aus I-042 bei N=16384: Fashion32 H1 (B, E gegen A) und H3 (E gegen K), CIFAR H1.
# Arme A, B, E (+K nur Fashion). Hypothesen und Kriterien WÖRTLICH wie in I-042 (_urteil_h1/_urteil_h3, Band, Kollaps, 1 pp, 100x).
# Holm über die 5 übernommenen Tests gemeinsam (Fashion H1B, H1E, H3; CIFAR H1B, H1E). H2/H4 entfallen. NK (A, B, N=1024) bleibt als Tor.
# Lernraten: aus I-042 ÜBERNOMMEN (keine neue Suche), hier fest eingetragen (Quelle: _zwischen_I-042.json, Tuning-Seed 2299).
# Gepoolte Auswertung (Original 2200-2204 + Replikation) nur beschreibend (keine Tests, kein Urteil).
KARTE_R = "I-042-r"
SEEDS_R = tuple(range(2210, 2225))
PROFILE_R = {"standard": SEEDS_R, "zehn": tuple(range(2210, 2220))}      # „zehn“ nur bei Zeitnot (vorab festgelegt)
EPOCHEN_R = 5
ARME_R = {"cifar10": ("A", "B", "E"), "fashion32": ("A", "B", "E", "K")}
FAMILIE_R = (("fashion32", "H1B"), ("fashion32", "H1E"), ("fashion32", "H3"), ("cifar10", "H1B"), ("cifar10", "H1E"))
TUNING_R = {"cifar10|1024": {"A": 0.001, "B": 0.001}, "cifar10|16384": {"A": 9.999999999999999e-05, "B": 0.0003, "E": 0.0003}, "fashion32|1024": {"A": 0.001, "B": 0.01}, "fashion32|16384": {"A": 9.999999999999999e-05, "B": 0.001, "E": 0.001, "K": 0.001}}


def _tuning_r():
    return {k: {a: {"lr": lr} for a, lr in v.items()} for k, v in TUNING_R.items()}


def _holm_r(a: dict):
    """Gemeinsames Holm über FAMILIE_R; schreibt p_holm je Datensatz zurück (Familie = die übernommenen 5 Tests)."""
    prim = {}
    for ds, h in FAMILIE_R:
        d = a["datensaetze"].get(ds, {}).get("diff", {}).get(h)
        prim[f"{ds}:{h}"] = d["p_gr"] if d and "p_gr" in d else 1.0
    ph = holm(prim)
    for ds, r in a["datensaetze"].items():
        r["p_holm"] = {h: ph[f"{ds}:{h}"] for d2, h in FAMILIE_R if d2 == ds and f"{ds}:{h}" in ph}
    return a


def urteil_r(a: dict) -> dict:
    nk = _nk(a)
    u = {"NK": f"{nk} (Mittel permutierte Labels A/B: " + ", ".join(
        f"{NAME[ds]} {k} {v:.3f}" for ds, r in a["datensaetze"].items() for k, v in r["negativ"].items()) + f"; Band {NK_BAND})"}
    for ds, r in a["datensaetze"].items():
        n = NAME[ds]
        u[f"H1-{n}"] = _urteil_h1(r, nk, ds)
        if ds == "fashion32":
            u[f"H3-{n}"] = _urteil_h3(r, nk)
        g = lambda k: _f(r["arme"][k]["mittel"] * 100).lstrip("+") if k in r["arme"] else "n/a"
        u[f"B-{n}"] = "beschreibend: Genauigkeit % bei N=16384 " + ", ".join(f"{x} {g(f'{x}_{BREITEN[-1]}')}" for x in ARME_R[ds])
    return u


def _gepoolt(zeilen_r: dict, original=None) -> dict:
    """NUR beschreibend: Original (I-042, Seeds 2200-2204) + Replikation, N=16384, Differenzen mit Mittel und 90-%-KI (ohne p-Werte)."""
    if original is None:
        dat = sorted(ERGEBNISSE.glob("I-042__*.json"))
        original = json.loads(dat[-1].read_text())["zeilen"] if dat else None
    if not original:
        return {"hinweis": "Originaldatei I-042 nicht gefunden"}
    out = {}
    G = str(BREITEN[-1])
    for ds, zr in zeilen_r.items():
        zo = original.get(ds, {}).get(G, [])
        zz = list(zo) + list(zr[BREITEN[-1]])
        for x, y in (("B", "A"), ("E", "A")) + ((("E", "K"),) if ds == "fashion32" else ()):
            if all(x in z and y in z for z in zz):
                s = _stat(_arr(zz, x) - _arr(zz, y))
                out[f"{NAME[ds]} {x} - {y}"] = {k: s[k] for k in ("mittel_pp", "ki_pp", "n", "sd_pp") if k in s} | {"n_original": len(zo)}
    return out


def lauf_r(schnell=False, prozesse=None, zwischen=None, profil=None, **_):
    profil = profil or os.environ.get("I042R_PROFIL", "standard")
    ds_alle = tuple(ARME_R)
    if schnell:      # Rauchtest NUR Tuning-Seed, kleine Daten, 1 Epoche
        seeds, klein, ep = (TUNING_SEED,), True, 1
    else:
        seeds, klein, ep = PROFILE_R[profil], False, EPOCHEN_R
        if set(seeds) & _BELEGT or set(seeds) & set(SEEDS) or TUNING_SEED in seeds:
            raise ValueError("Seeds nicht frisch/vorregistriert")
    hp = _tuning_r()
    pfad = Path(zwischen) if zwischen else ERGEBNISSE / f"_zwischen_{KARTE_R}{'-schnell' if schnell else ''}.json"
    kennung = {"karte": KARTE_R, "version": 1, "profil": "rauch" if schnell else profil, "epochen": ep, "seeds": list(seeds),
               "tuning": TUNING_R, "klein": klein}
    t0 = time.time()
    aufg = []
    for N in (BREITEN[0], BREITEN[-1]):
        for ds in ds_alle:
            for s in seeds:       # N=1024: nur die NK (arme leer -> _arbeiter rechnet A_perm, B_perm)
                aufg.append((ds, N, s, () if N == BREITEN[0] else ARME_R[ds], hp[f"{ds}|{N}"], ep, klein))
    aufg.sort(key=lambda a: -(a[1] == BREITEN[-1]) * (3 if a[0] == "cifar10" else 2))    # teure CIFAR-Einheiten zuerst
    alle = _rechne(aufg, pfad, kennung, prozesse=prozesse)
    zeilen = {ds: {} for ds in ds_alle}
    for ds in ds_alle:
        for N in (BREITEN[0], BREITEN[-1]):
            zeilen[ds][N] = [alle[f"einheit|{ds}|{N}|{s}"] for s in seeds]
    # 1024-Zeilen tragen nur A_perm/B_perm, 16384-Zeilen nur die Arme; für die Auswertung zusammenführen (NK steht bei N=1024)
    params = {ds: {N: {a: param_schicht(a, _dins()[ds], N) for a in ARME} for N in (BREITEN[0], BREITEN[-1])} for ds in ds_alle}
    a = _holm_r(auswertung(zeilen, params))
    e = {"abteilung": "Struktur", "modul": "struktur_gross", "karte": KARTE_R, "version": 1, "profil": profil,
         "hinweis": "Replikation von I-042; Kriterien wörtlich wie I-042; Holm über 5 übernommene Tests; gepoolt nur beschreibend",
         "seeds": list(seeds), "tuning_seed": TUNING_SEED, "tuning_uebernommen": TUNING_R,
         "einstellungen": {"epochen": ep, "batch": BATCH, "arme": ARME_R}, "zwischendatei": str(pfad),
         "zeilen": {ds: {str(N): z for N, z in zn.items()} for ds, zn in zeilen.items()}, "sekunden": time.time() - t0,
         "auswertung": a, "gepoolt_beschreibend": _gepoolt(zeilen)}
    e["urteil"] = urteil_r(a)
    return e


def _vortest_r() -> bool:
    ok = True

    def pruefe(name, bed):
        nonlocal ok
        if not bed:
            print(f"  Vortest FEHLER (I-042-r): {name}", flush=True)
            ok = False
    for pn, sd in PROFILE_R.items():
        pruefe(f"Seeds frisch {pn}", not (set(sd) & (_BELEGT | set(SEEDS) | {TUNING_SEED})))
    pruefe("15 Seeds", len(SEEDS_R) == 15 and len(PROFILE_R["zehn"]) == 10)
    pruefe("Tuning vollständig", all(a in TUNING_R[f"{ds}|{N}"] for ds in ARME_R for N in (BREITEN[0], BREITEN[-1])
                                      for a in (ARME_R[ds] if N == BREITEN[-1] else ("A", "B"))))
    orig = ERGEBNISSE / "_zwischen_I-042.json"
    if orig.exists():
        t = json.loads(orig.read_text())["zeilen"]["tuning"]
        pruefe("Tuning stimmt mit I-042 überein", all(t[k][a]["lr"] == lr for k, v in TUNING_R.items() for a, lr in v.items()))
    # Urteilszweige erzwingen: Daten wie _synth (Arme A,B,E,K; N=1024 für NK), je Datensatz
    rng = np.random.default_rng(23)

    def u(n=15, **kw):
        off = kw.pop("off", None); pr_off = kw.pop("param_off", None); sdk = kw.pop("sd", 0.001)
        zeilen, pr = {}, {}
        for ds, din in (("cifar10", 3072), ("fashion32", 1024)):
            kk = dict(kw); kk["mitK"] = ds == "fashion32"
            z = _synth(rng, off=off, sd=sdk, n=n, **kk)
            zeilen[ds] = z
            pr[ds] = {N: {a: param_schicht(a, din, N) for a in ARME} for N in BREITEN}
            if pr_off:
                for N in pr[ds]:
                    pr[ds][N].update(pr_off)
        return urteil_r(_holm_r(auswertung(zeilen, pr)))
    gesehen = set()

    def erwarte(name, ur, **soll):
        nonlocal ok
        for k, w in soll.items():
            gesehen.add((k.split("-")[0], w.split(" (")[0]))
            if not ur[k].startswith(w):
                print(f"  Vortest FEHLER (I-042-r) {name}: {k} soll {w}, ist {ur[k]}", flush=True)
                ok = False
    erwarte("Normalfall", u(), **{"NK": "BESTANDEN", "H1-CIFAR": "NICHT_UNTERLEGEN", "H1-Fashion": "NICHT_UNTERLEGEN", "H3-Fashion": "E_DOMINIERT"})
    erwarte("H1 beide unterlegen", u(off={"B": 0.50, "E": 0.50}), **{"H1-CIFAR": "H0 (beide unterlegen", "H1-Fashion": "H0 (beide unterlegen"})
    erwarte("H3 K besser", u(off={"K": 0.60}), **{"H3-Fashion": "H0 (K besser"})
    erwarte("Rauschen", u(sd=0.02, off={"B": 0.56, "K": 0.56}), **{"H1-CIFAR": "HU", "H1-Fashion": "HU", "H3-Fashion": "HU"})
    erwarte("NK faellt", u(nk=0.30), **{"NK": "DURCHGEFALLEN", "H1-Fashion": "HU (Negativkontrolle", "H3-Fashion": "HU (Negativkontrolle"})
    erwarte("Kollaps B", u(koll="B"), **{"H1-Fashion": "HU (Instabilitaet"})
    erwarte("Parameter reichen nicht", u(param_off={"B": 10 ** 6, "E": 10 ** 6}), **{"H1-Fashion": "HU"})
    erwarte("Budget K zu klein", u(param_off={"K": 10}), **{"H3-Fashion": "HU (Budget"})
    erwarte("10 Seeds (Profil zehn)", u(n=10), **{"H1-Fashion": "NICHT_UNTERLEGEN"})
    for h, ws in {"H1": {"NICHT_UNTERLEGEN", "H0", "HU"}, "H3": {"E_DOMINIERT", "H0", "HU"}}.items():
        pruefe(f"Zweige {h}", ws <= {w for k, w in gesehen if k == h})
    # gemeinsames Holm wirkt: 5 Tests, kleinstes p wird mit 5 multipliziert
    ph = holm({f"t{i}": 0.01 for i in range(5)})
    pruefe("Holm über 5", all(abs(v - 0.05) < 1e-12 for v in ph.values()))
    # gepoolte Auswertung (beschreibend) läuft auf synthetischen Originaldaten
    zr = {"fashion32": {BREITEN[-1]: _synth(rng, n=15)[BREITEN[-1]]}}
    orig_s = {"fashion32": {str(BREITEN[-1]): _synth(rng, n=5)[BREITEN[-1]]}}
    g = _gepoolt(zr, orig_s)
    pruefe("gepoolt n=20, ohne p", g["Fashion B - A"]["n"] == 20 and "p_gr" not in g["Fashion B - A"])
    return ok
