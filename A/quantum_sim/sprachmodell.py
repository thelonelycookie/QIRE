from __future__ import annotations

"""RESEARCH ID: T-002 -- Das Sprachmodell (Phase 3 der Roadmap).

DIE FRAGE
Haelt die quantenabgeleitete Struktur in einem echten Sprachmodell, und was
kostet sie? Drei Arme auf deutschem Wikipedia-Text, gleiche Zahl an
Gradientenschritten.

DIE ARME (d_model 256 = 2^8, 4 Lagen, 4 Koepfe)
  dense           Referenz. FFN d->4d->d, Q/K/V als dichte d x d.
  butterfly_ffn   Der FFN-Block wird ersetzt durch normieren -> Butterfly ->
                  Born-Regel. Der Block braucht eine Nichtlinearitaet, und die
                  quadratische Auslese IST sie -- 87 bis 92 Prozent des
                  Vorteils der Schaltung entfallen laut Q-003 auf sie.
  butterfly_qkv   Q-, K- und V-Projektionen werden ersetzt durch eine REELLE
                  orthogonale Butterfly-Drehung, OHNE Born-Regel.
  narrow          AUFLOESUNGSPRUEFUNG. FFN d->d/4->d, also dieselbe Bauform wie
                  dense, nur mit einem Viertel der Breite. `dense` muss diesen
                  Arm schlagen -- sonst sieht der Aufbau Kapazitaetsunterschiede
                  im FFN gar nicht, und kein FFN-Vergleich ist interpretierbar.

VORREGISTRIERUNG
  Gemessen wird die Kreuzentropie auf der Validierungsmenge in nats je Token,
  bei GLEICHER Zahl an Gradientenschritten. 8 Seeds je Arm.

  PRIMAER  butterfly_ffn gegen dense: TOST auf Gleichwertigkeit innerhalb
           +-0,05 nats (dieselbe Grenze, die Q-013 angemeldet hatte).
  P2       butterfly_qkv gegen dense, gleiche Grenze.
  Holm ueber genau diese beiden.

  WARUM 8 SEEDS UND NICHT 5: Bei n Seeds ist 2/2^n der kleinstmoegliche
  Wilcoxon-p-Wert. n=5 gibt 0,0625 -- Signifikanz waere unerreichbar, derselbe
  Entwurfsfehler wie in Q-009. Q-013 war mit 5 Seeds geplant und haette ihn
  geerbt. Bei n=8 sind es 0,0078, nach Holm ueber zwei Vergleiche 0,0156.

  T1  PRIMAER gleichwertig -> die Struktur haelt im Sprachmodell, bei 6.488-fach
      weniger Parametern im FFN-Block.
  T1+ butterfly_ffn ist signifikant BESSER -> unerwartet, aber kodiert.
  T0  dense ist signifikant besser -> die Struktur kostet, und zwar messbar.
  TU  weder noch -> erreichte Schranke berichten.

  AUFLOESUNGSPRUEFUNG: dense muss `narrow` schlagen (Trennung verlangt, nicht
  Richtung -- faellt sie umgekehrt aus, ist das ein Ergebnis).

  WAS DIESER LAUF NICHT ZEIGT: Effizienz. Bei d_model 256 liegt die Struktur
  weit unterhalb ihres Uebergangspunkts von 16.384 (T-001) und ist gemessen
  5,9-fach LANGSAMER. Die Qualitaetsaussage wird deshalb bei gleicher
  Schrittzahl gefuehrt und die Wanduhrzeit getrennt daneben berichtet. "Der
  Chatbot laeuft dank der Struktur schneller" waere falsch.

WARUM Q/K/V OHNE BORN-REGEL
`BEFUNDE_UND_RICHTUNGEN.md` haelt fest: "Anders als der FFN-Block sind sie
linear, also die natuerlichere Einsatzstelle." Eine Wahrscheinlichkeitsverteilung
als Q-Vektor waere sinnlos -- Aufmerksamkeit braucht vorzeichenbehaftete
Richtungen, keine Energieanteile. Verwendet wird deshalb die reelle
Givens-Butterfly: dieselbe Verschraenkungsstruktur, dieselbe Gewichtsteilung
ueber die Ebene, aber als orthogonale Abbildung R^d -> R^d. Das spart nebenbei
den Faktor 4 fuer komplexe Arithmetik.

DIE FAIRNESSFRAGE, DIE VORHER GEKLAERT WERDEN MUSS
T-000 hat den Durchsatz nur des DICHTEN Modells gemessen. Die Butterfly-Arme
sind langsamer -- T-001 hat gezeigt, dass die Struktur unterhalb von 16.384
Dimensionen verliert, und d_model ist hier 256. Bei gleichem ZEITbudget saehen
sie also weniger Token und waeren durch genau den Umsetzungsnachteil
benachteiligt, den T-001 bereits getrennt berichtet hat.

Deshalb: **gleiche Zahl an Gradientenschritten**, Wanduhrzeit getrennt daneben.
Das ist dieselbe Lehre wie aus B-010 ("Epochen statt Gradientenschritte") und
trennt Qualitaet von Umsetzungsgeschwindigkeit, wie es die Roadmap verlangt.

`durchsatz_alle_arme()` misst den Durchsatz aller drei Arme, damit die
Wanduhrzeit vorher bekannt ist und nicht hinterher ueberrascht.

Run:
  python -m quantum_sim.sprachmodell durchsatz   # Tokens/s je Arm
  python -m quantum_sim.sprachmodell probe       # kurzer Testlauf, 200 Schritte
  python -m quantum_sim.sprachmodell trainieren  # der eigentliche Lauf
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.result_store import Recorder

from .butterfly_layer import apply_1q, cnot_permutation
from .sequence_platform import ButterflyFFN

WURZEL = Path(__file__).resolve().parent.parent
KORPUS = WURZEL / "results" / "korpus"
MODELLE = WURZEL / "results" / "modelle"

D_MODEL = 256                  # 2^8 -- Butterfly braucht eine Zweierpotenz
LAGEN = 4
KOEPFE = 4
SEQ = 128
# Von 16 auf 64 erhoeht am 11. August, VOR dem ersten Messlauf.
# Grund, auf der GPU nachgemessen: Der Nachteil der Butterfly-Struktur ist dort
# Startaufwand, nicht Rechenarbeit. Von Batch 16 auf 64 faellt die Zeit je
# Schritt sogar leicht (0,3468 -> 0,3343 s bei butterfly_ffn), waehrend
# viermal so viele Token verarbeitet werden -- der Durchsatz steigt um 4,1x
# zum Nulltarif. Bei 256 wird der Speicher mit 18,3 von 24 GB eng und der
# Gewinn flacht ab. 64 ist der Punkt, an dem die GPU ausgelastet ist, ohne
# dass die Batchgroesse das Training veraendert.
BATCH = 64
VOCAB = 4096
FFN_LAGEN = 5                  # Butterfly-Tiefe im FFN-Block
QKV_LAGEN = 5
ARME = ("dense", "butterfly_ffn", "butterfly_qkv", "narrow")
REGISTRIERT = ("PRIMAER", "P2")
SEEDS = (2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108)   # 8 -> min p 0,0078
MARGIN_NATS = 0.05             # wie in Q-013 angemeldet
# Auf der Validierungsmenge gemessen: eine Vorhersage aus reinen
# Token-Haeufigkeiten ohne jeden Kontext. Ein Arm, der das nicht deutlich
# unterbietet, hat ueber Kontext nichts gelernt -- unabhaengig davon, wie der
# Vergleich mit den anderen Armen ausfaellt.
UNIGRAMM_NATS = 7.0756         # ppl 1.183; Gleichverteilung waere 8,3178
LR = 3e-4                      # nur noch Rueckfallwert; gesucht wird je Arm
# Regel 1 des Protokolls: Lernrate je Arm auf eigenen Tuning-Seeds, und das
# Gitter muss das Optimum ENTHALTEN. Randlagen werden gemeldet.
LR_GITTER = (3e-4, 6e-4, 1.2e-3, 2.4e-3)
ERWEITERUNGEN = 3              # wie oft das Gitter nach oben verlaengert werden
                               # darf, wenn ein Arm auf dem Rand landet
LR_SUCHE_SCHRITTE = 1000
LR_SUCHE_SEED = 2199           # bewusst NICHT aus SEEDS -- sonst waere die
                               # Lernrate auf denselben Seeds gesucht, auf
                               # denen spaeter gemessen wird
WARMUP_SCHRITTE = 200
EVAL_ALLE = 500
EVAL_STAPEL = 40


# --------------------------------------------------------------------------
# Reelle orthogonale Butterfly fuer die linearen Projektionen
# --------------------------------------------------------------------------

class ButterflyLinear(nn.Module):
    """Orthogonale Abbildung R^d -> R^d mit L*n Winkeln statt d^2 Gewichten.

    Dieselbe Struktur wie die Quantenschaltung -- Givens-Drehung auf jedem
    Amplitudenpaar einer Qubit-Ebene, danach der CNOT-Ring als Permutation --
    aber reell und ohne Born-Regel. Ein Winkel je (Lage, Ebene), also die
    quantenabgeleitete Gewichtsteilung: ein Gatter wirkt auf ein Qubit,
    unabhaengig von der Groesse des Raums.

    Bei d=256: 4 Lagen x 8 Ebenen = 40 Winkel gegen 65.536 Gewichte.
    Orthogonal per Konstruktion, also normerhaltend -- die Eigenschaft, um die
    es in Q-011 geht.
    """

    def __init__(self, d: int, lagen: int = QKV_LAGEN):
        super().__init__()
        self.n = int(math.log2(d))
        assert 2 ** self.n == d, "d muss eine Zweierpotenz sein"
        self.L = lagen
        self.theta = nn.Parameter(torch.rand(lagen, self.n) * 2 * math.pi)
        perms = [cnot_permutation(self.n, q, (q + 1) % self.n)
                 for q in range(self.n)]
        # R1 aus T-001: die n Permutationen zu einer zusammenfassen.
        acc = perms[0]
        for p in perms[1:]:
            acc = acc[p]
        self.register_buffer("perm", acc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        form = x.shape
        v = x.reshape(-1, form[-1])
        n = self.n
        for lage in range(self.L):
            c = torch.cos(self.theta[lage])
            s = torch.sin(self.theta[lage])
            for q in range(n):
                links, rechts = 2 ** q, 2 ** (n - 1 - q)
                w = v.reshape(-1, links, 2, rechts)
                a, b = w[:, :, 0, :], w[:, :, 1, :]
                v = torch.stack((c[q] * a - s[q] * b,
                                 s[q] * a + c[q] * b), dim=2).reshape(-1, 2 ** n)
            v = v[:, self.perm]
        return v.reshape(form)


class ButterflyAttention(nn.Module):
    """Mehrkopf-Aufmerksamkeit, bei der Q, K und V von je einer orthogonalen
    Butterfly erzeugt werden statt von dichten Matrizen. Die Ausgabeprojektion
    bleibt dicht -- sonst waeren zwei Dinge gleichzeitig geaendert."""

    def __init__(self, d: int, koepfe: int):
        super().__init__()
        self.d, self.h = d, koepfe
        self.kopf_dim = d // koepfe
        self.q = ButterflyLinear(d)
        self.k = ButterflyLinear(d)
        self.v = ButterflyLinear(d)
        self.aus = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, maske: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        def teile(t):
            return t.reshape(B, T, self.h, self.kopf_dim).transpose(1, 2)
        q, k, v = teile(self.q(x)), teile(self.k(x)), teile(self.v(x))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=maske[:T, :T])
        return self.aus(y.transpose(1, 2).reshape(B, T, D))


# --------------------------------------------------------------------------
# Das Modell
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, arm: str, d: int, koepfe: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if arm == "butterfly_qkv":
            self.att = ButterflyAttention(d, koepfe)
            self.dicht_att = False
        else:
            self.att = nn.MultiheadAttention(d, koepfe, batch_first=True)
            self.dicht_att = True
        if arm == "butterfly_ffn":
            self.ffn = ButterflyFFN(d, num_layers=FFN_LAGEN, tie_weights=True)
        elif arm == "narrow":
            # Auflösungspruefung: dieselbe Bauform, ein Viertel der Breite.
            self.ffn = nn.Sequential(nn.Linear(d, d // 4), nn.GELU(),
                                     nn.Linear(d // 4, d))
        else:
            self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                     nn.Linear(4 * d, d))

    def forward(self, x, maske):
        h = self.n1(x)
        if self.dicht_att:
            x = x + self.att(h, h, h, attn_mask=maske[:x.shape[1], :x.shape[1]],
                             need_weights=False)[0]
        else:
            x = x + self.att(h, maske)
        return x + self.ffn(self.n2(x))


class SprachModell(nn.Module):
    def __init__(self, arm: str, vocab: int = VOCAB, d: int = D_MODEL,
                 lagen: int = LAGEN, koepfe: int = KOEPFE, seq: int = SEQ):
        super().__init__()
        self.arm = arm
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(arm, d, koepfe) for _ in range(lagen)])
        self.norm = nn.LayerNorm(d)
        self.kopf = nn.Linear(d, vocab, bias=False)
        self.register_buffer("maske", torch.triu(
            torch.full((seq, seq), float("-inf")), diagonal=1))

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, self.maske)
        return self.kopf(self.norm(x))

    def struktur_parameter(self) -> int:
        """Parameter in dem Teil, der zwischen den Armen unterschiedlich ist."""
        gesamt = 0
        for b in self.blocks:
            if self.arm == "butterfly_ffn":
                gesamt += sum(p.numel() for p in b.ffn.parameters())
            elif self.arm == "butterfly_qkv":
                gesamt += sum(p.numel() for n, p in b.att.named_parameters()
                              if not n.startswith("aus"))
            else:
                gesamt += sum(p.numel() for p in b.ffn.parameters())
        return gesamt


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------

def lade_token(teil: str) -> np.ndarray:
    pfad = KORPUS / f"{teil}.npy"
    if not pfad.exists():
        raise FileNotFoundError(
            f"{pfad} fehlt. Erst 'python -m quantum_sim.korpus_deutsch "
            f"aufbereiten' laufen lassen.")
    return np.load(pfad, mmap_mode="r")


def stapel(daten: np.ndarray, batch: int, seq: int,
           g: torch.Generator) -> tuple:
    i = torch.randint(0, len(daten) - seq - 1, (batch,), generator=g)
    x = torch.stack([torch.from_numpy(daten[j:j + seq].astype(np.int64)) for j in i])
    y = torch.stack([torch.from_numpy(daten[j + 1:j + seq + 1].astype(np.int64))
                     for j in i])
    return x, y


# --------------------------------------------------------------------------
# Durchsatz -- die Zahl, die T-000 fuer die Butterfly-Arme schuldig blieb
# --------------------------------------------------------------------------

def durchsatz_alle_arme(schritte: int = 8, threads: int = 6) -> Dict[str, float]:
    """Durchsatz je Arm AUF DEM GERAET, auf dem auch trainiert wird.

    ZWEI FEHLER, BEHOBEN AM 11. AUGUST, nachdem der Pod eine unbrauchbare
    Tabelle geliefert hatte:

    (1) Modell und Daten wurden nie auf die GPU verschoben. Die Funktion mass
        auf einem GPU-Pod den Prozessor. Aufgefallen ist es daran, dass `dense`
        dort 7.146 Token/s erreichte gegen 5.522 auf dem Laptop -- eine 4090
        ist nicht 1,3-fach schneller als sechs CPU-Kerne, sondern ein bis zwei
        Groessenordnungen.

    (2) CUDA arbeitet asynchron. Ohne `torch.cuda.synchronize()` vor dem
        Stoppen der Uhr misst man, wie lange das ABSCHICKEN der Kernel dauert,
        nicht ihre Ausfuehrung -- und bekommt absurd gute Zahlen. Der Fehler
        haette den ersten ueberdeckt.
    """
    dev = geraet()
    print("=" * 86)
    print(f"DURCHSATZ JE ARM auf {dev} -- was 'gleiche Schrittzahl' kostet")
    if dev.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print("=" * 86)
    kopf = (f'  {"Arm":>16} {"Parameter":>12} {"Struktur":>10} '
            f'{"s/Schritt":>10} {"Token/s":>10} {"relativ":>8}')
    print(kopf); print("  " + "-" * (len(kopf) - 2))
    x = torch.randint(0, VOCAB, (BATCH, SEQ)).to(dev)
    y = torch.randint(0, VOCAB, (BATCH, SEQ)).to(dev)
    werte: Dict[str, float] = {}

    def gleichstand():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for arm in ARME:
        set_seed(9000)
        torch.set_num_threads(threads)
        m = SprachModell(arm).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

        def schritt():
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)).backward()
            opt.step()

        for _ in range(5):          # mehr Warmlauf: CUDA baut beim ersten Mal
            schritt()               # Kernel und Speicherpools auf
        gleichstand()
        t0 = time.perf_counter()
        for _ in range(schritte):
            schritt()
        gleichstand()
        dauer = (time.perf_counter() - t0) / schritte
        tps = BATCH * SEQ / dauer
        werte[arm] = tps
        rel = tps / werte[ARME[0]]
        print(f'  {arm:>16} {sum(p.numel() for p in m.parameters()):>12,} '
              f'{m.struktur_parameter():>10,} {dauer:>10.3f} {tps:>10,.0f} '
              f'{rel:>7.2f}x')
    print()
    langsamster = min(werte, key=werte.get)
    faktor = werte[ARME[0]] / werte[langsamster]
    print(f"  Langsamster Arm: {langsamster}, Faktor {faktor:.1f} gegenueber dense.")
    print(f"  Bei gleicher Schrittzahl kostet der Lauf also das {faktor:.1f}-fache")
    print(f"  der Wanduhrzeit des dichten Arms. Das ist der Umsetzungsnachteil")
    print(f"  aus T-001 und gehoert getrennt berichtet, nicht in die Qualitaet.")
    print("=" * 86)
    return werte


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def geraet() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def bewerte(modell, daten, dev, g: torch.Generator) -> float:
    """Kreuzentropie in nats je Token auf der Validierungsmenge."""
    modell.eval()
    summe = 0.0
    for _ in range(EVAL_STAPEL):
        x, y = stapel(daten, BATCH, SEQ, g)
        x, y = x.to(dev), y.to(dev)
        summe += float(F.cross_entropy(modell(x).reshape(-1, VOCAB), y.reshape(-1)))
    modell.train()
    return summe / EVAL_STAPEL


def trainiere_arm(arm: str, seed: int, schritte: int, dev: torch.device,
                  zug: np.ndarray, val: np.ndarray,
                  threads: int = 6, lr: float = LR,
                  sichern: bool = True) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    modell = SprachModell(arm).to(dev)
    opt = torch.optim.AdamW(modell.parameters(), lr=lr)
    # Bei kurzen Laeufen (Probe) waere WARMUP_SCHRITTE/schritte groesser als 1
    # und OneCycleLR bricht ab -- in der ersten Fassung genau so passiert und
    # von der Probe gefunden, bevor GPU-Zeit dafuer bezahlt wurde.
    anteil = min(0.3, WARMUP_SCHRITTE / max(schritte, 1))
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=schritte,
        pct_start=anteil, anneal_strategy="cos")
    g = torch.Generator().manual_seed(seed)
    gval = torch.Generator().manual_seed(seed + 500000)
    verlauf: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for schritt in range(1, schritte + 1):
        x, y = stapel(zug, BATCH, SEQ, g)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(modell(x).reshape(-1, VOCAB), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
        opt.step()
        plan.step()
        if schritt % EVAL_ALLE == 0 or schritt == schritte:
            gval2 = torch.Generator().manual_seed(seed + 500000)  # immer dieselben Stapel
            v = bewerte(modell, val, dev, gval2)
            verlauf.append({"schritt": schritt, "val_nats": v,
                            "sekunden": time.perf_counter() - t0})
            print(f"      {arm:>14} seed {seed} Schritt {schritt:>6}/{schritte}  "
                  f"val {v:.4f} nats  ppl {math.exp(v):>8.1f}  "
                  f"({time.perf_counter() - t0:,.0f} s)")
    dauer = time.perf_counter() - t0
    end_val = verlauf[-1]["val_nats"]
    # Gewichte des ERSTEN Seeds je Arm sichern -- daraus entsteht der Chatbot.
    # Ohne das haette der Lauf am Ende Zahlen, aber kein Modell.
    if sichern and seed == SEEDS[0]:
        MODELLE.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "schritte": schritte,
                    "val_nats": end_val, "d_model": D_MODEL, "lagen": LAGEN,
                    "koepfe": KOEPFE, "seq": SEQ, "vocab": VOCAB,
                    "state_dict": modell.state_dict()},
                   MODELLE / f"{arm}.pt")
        print(f"      Gewichte gesichert: {MODELLE / (arm + '.pt')}")
    return {"arm": arm, "seed": seed, "schritte": schritte,
            "val_nats": end_val, "val_perplexitaet": math.exp(end_val),
            "sekunden": dauer, "token": schritte * BATCH * SEQ,
            "token_je_sekunde": schritte * BATCH * SEQ / dauer,
            "parameter": sum(p.numel() for p in modell.parameters()),
            "struktur_parameter": modell.struktur_parameter(),
            "geraet": str(dev), "verlauf": verlauf,
            "d_model": D_MODEL, "lagen": LAGEN, "koepfe": KOEPFE,
            "seq": SEQ, "batch": BATCH, "vocab": VOCAB, "lr": LR}


def lernratensuche(threads: int = 6) -> Dict[str, float]:
    """Regel 1: Lernrate je Arm, auf einem eigenen Tuning-Seed gesucht.

    Die bisherigen 3e-4 waren fuer Batch 16 gesetzt. Bei Batch 64 ist das
    vermutlich zu niedrig -- geraten wird aber nicht, sondern gemessen. Das
    Ergebnis landet in `results/korpus/lr_map.json` und wird vom Hauptlauf
    gelesen; Randlagen werden gemeldet, wie in allen Q-Laeufen.
    """
    import json

    dev = geraet()
    zug, val = lade_token("train"), lade_token("val")
    print("=" * 86)
    print(f"LERNRATENSUCHE auf {dev} -- {LR_SUCHE_SCHRITTE} Schritte je Zelle, "
          f"Seed {LR_SUCHE_SEED}")
    print("=" * 86)
    karte: Dict[str, float] = {}
    tabelle: Dict[str, Dict[float, float]] = {}
    for arm in ARME:
        werte: Dict[float, float] = {}
        raten = list(LR_GITTER)
        # Regel 1 verlangt, dass das Gitter das Optimum ENTHAELT. Landet ein Arm
        # auf dem oberen Rand und steigt dort noch, wird nach oben verlaengert
        # -- bis zu ERWEITERUNGEN Mal. Das ist die vorgeschriebene Reaktion auf
        # eine Randlagen-Warnung und keine nachtraegliche Anpassung: Das
        # Auswahlkriterium bleibt unveraendert, nur der Suchbereich waechst.
        # Am 4. August bei Q-003-replication genauso gemacht, nur von Hand.
        for runde in range(ERWEITERUNGEN + 1):
            for lr in raten:
                if lr in werte:
                    continue
                z = trainiere_arm(arm, LR_SUCHE_SEED, LR_SUCHE_SCHRITTE, dev,
                                  zug, val, threads, lr=lr, sichern=False)
                werte[lr] = z["val_nats"]
            bester = min(werte, key=werte.get)
            if bester != max(werte) or runde == ERWEITERUNGEN:
                break
            naechste = max(werte) * 2
            print(f"    {arm}: bester Wert am oberen Rand ({bester:g}) -- "
                  f"verlaengere auf {naechste:g}")
            raten = [naechste]
        karte[arm] = min(werte, key=werte.get)
        tabelle[arm] = werte
        print(f"  {arm:>16} -> lr {karte[arm]:<8} "
              + "  ".join(f"{k:g}:{werte[k]:.4f}" for k in sorted(werte)))
    rand = [a for a, v in karte.items()
            if v == min(tabelle[a]) or v == max(tabelle[a])]
    print()
    if rand:
        print(f"  WARNUNG: am Gitterrand -> {', '.join(rand)}")
        print(f"  Das Optimum koennte ausserhalb liegen; der Vorteil dieser Arme")
        print(f"  ist dann nicht ausgereizt. So berichten.")
    else:
        print("  Kein Arm am Gitterrand -- das Gitter enthaelt die Optima.")
    ziel = KORPUS / "lr_map.json"
    ziel.write_text(json.dumps(
        {"lr_map": karte, "tabelle": {a: {str(k): v for k, v in t.items()}
                                      for a, t in tabelle.items()},
         "gitter": list(LR_GITTER), "am_rand": rand,
         "schritte": LR_SUCHE_SCHRITTE, "seed": LR_SUCHE_SEED,
         "batch": BATCH}, indent=2), encoding="utf-8")
    print(f"  gesichert: {ziel}")
    print("=" * 86)
    return karte


def lies_lr_map() -> Dict[str, float]:
    import json
    pfad = KORPUS / "lr_map.json"
    if not pfad.exists():
        print(f"  WARNUNG: {pfad} fehlt -- alle Arme laufen mit {LR}. "
              f"Regel 1 ist damit verletzt.")
        return {arm: LR for arm in ARME}
    d = json.loads(pfad.read_text(encoding="utf-8"))
    return {a: float(v) for a, v in d["lr_map"].items()}


def trainieren(schritte: int = 20000, seeds: Sequence[int] = SEEDS,
               arme: Sequence[str] = ARME, threads: int = 6,
               weiter: bool = True) -> Recorder:
    dev = geraet()
    zug, val = lade_token("train"), lade_token("val")
    lr_map = lies_lr_map()
    rec = Recorder(
        experiment_id="T-002",
        description=("Haelt die quantenabgeleitete Struktur in einem "
                     "Sprachmodell? Deutsches Wikipedia, gleiche Zahl an "
                     "Gradientenschritten, 8 Seeds."),
        config={"arme": list(arme), "seeds": list(seeds), "schritte": schritte,
                "d_model": D_MODEL, "lagen": LAGEN, "koepfe": KOEPFE,
                "seq": SEQ, "batch": BATCH, "vocab": VOCAB, "lr_map": lr_map,
                "ffn_lagen": FFN_LAGEN, "qkv_lagen": QKV_LAGEN,
                "margin_nats": MARGIN_NATS, "geraet": str(dev),
                "korpus": "deutsche Wikipedia 20231101, 397 Mio. Token",
                "zeichen_je_token": 3.27,
                "messgroesse": "val_nats (Kreuzentropie je Token)",
                "primary": "butterfly_ffn gleichwertig zu dense (+-0,05 nats)",
                "p2": "butterfly_qkv gegen dense",
                "holm_family": list(REGISTRIERT),
                "resolution_check": "dense muss narrow schlagen",
                "budget": "gleiche Gradientenschritte, NICHT gleiche Zeit -- "
                          "sonst misst die Qualitaet den Umsetzungsnachteil aus "
                          "T-001 mit",
                "torch_threads": threads},
    )
    # --- Wiederaufnahme -----------------------------------------------
    # Am 12. August ist ein fertiger 22-Stunden-Lauf verloren gegangen, weil
    # der gemietete Rechner geloescht wurde. Seither gilt: Ein abgebrochener
    # Lauf darf nicht bei null anfangen. Bereits gerechnete (Arm, Seed)-Paare
    # werden aus der letzten Rohdatendatei uebernommen und uebersprungen.
    #
    # Das ist Regel 5 des Protokolls ("Rohdaten nach jedem Abschnitt sichern"),
    # eine Ebene hoeher gedacht: Nicht nur speichern, sondern auch wieder
    # aufsetzen koennen.
    fertig = set()
    if weiter:
        try:
            from qire_ai.result_store import load
            vorher = load("T-002")["records"]
            for r in vorher:
                if int(r.get("schritte", -1)) == schritte:
                    fertig.add((r["arm"], int(r["seed"])))
                    rec.add(**r)
        except FileNotFoundError:
            pass
        except Exception as exc:                      # noqa: BLE001
            print(f"  Wiederaufnahme nicht moeglich ({exc!r}) -- starte neu")
            fertig, rec.records = set(), []

    print("=" * 90)
    print(f"T-002 | {len(arme)} Arme x {len(seeds)} Seeds x {schritte:,} Schritte "
          f"auf {dev}")
    if fertig:
        print(f"WIEDERAUFNAHME: {len(fertig)} von {len(arme)*len(seeds)} Zellen "
              f"schon gerechnet, werden uebersprungen")
    print("=" * 90)
    for seed in seeds:
        offen = [a for a in arme if (a, int(seed)) not in fertig]
        if not offen:
            print(f"    seed {seed} vollstaendig vorhanden -- uebersprungen")
            continue
        for arm in offen:
            rec.add(**trainiere_arm(arm, seed, schritte, dev, zug, val,
                                    threads, lr=lr_map.get(arm, LR)))
        rec.save()          # nach JEDEM Seed sichern, nicht am Ende
        print(f"    Zwischenstand nach seed {seed} gesichert "
              f"({len(rec.records)} Zellen gesamt)")
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats
    from qire_ai.equivalence import holm, tost_paired

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    out: Dict[str, object] = {}

    def vals(arm: str) -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm),
                      key=lambda r: r["seed"])
        return np.array([r["val_nats"] for r in rows])

    print()
    print(f"RESEARCH ID T-002 | Sprachmodell, d_model {D_MODEL}, deutscher Text")
    kopf = (f'  {"Arm":>16} {"Parameter":>11} {"Struktur":>9} {"val nats":>9} '
            f'{"std":>7} {"Perplex.":>9} {"Token/s":>9}')
    print(kopf); print("  " + "-" * (len(kopf) - 2))
    for arm in config["arme"]:
        v = vals(arm)
        if not len(v):
            continue
        r0 = next(r for r in records if r["arm"] == arm)
        tps = float(np.median([r["token_je_sekunde"] for r in records
                               if r["arm"] == arm]))
        print(f'  {arm:>16} {r0["parameter"]:>11,} {r0["struktur_parameter"]:>9,} '
              f'{v.mean():>9.4f} {v.std(ddof=1):>7.4f} '
              f'{math.exp(v.mean()):>9.1f} {tps:>9,.0f}')

    fam = [("butterfly_ffn", "dense", "PRIMAER"), ("butterfly_qkv", "dense", "P2")]
    roh, zellen = [], []
    for a, b, tag in fam:
        va, vb = vals(a), vals(b)
        p = float(stats.wilcoxon(va, vb).pvalue)
        roh.append(p)
        # Niedriger ist besser bei Kreuzentropie -> Vorzeichen umdrehen, damit
        # "positiv = besser" gilt wie in den Q-Laeufen.
        zellen.append((a, b, tag, float(vb.mean() - va.mean()), p,
                       tost_paired(va, vb, MARGIN_NATS), int((va < vb).sum()), len(va)))
    adj = holm(roh)

    print()
    print("=" * 90)
    print(f"VORREGISTRIERTE FAMILIE, Grenze +-{MARGIN_NATS} nats "
          f"(positiv = Butterfly besser)")
    for (a, b, tag, delta, p, eq, w, nn_), p_h in zip(zellen, adj):
        zustand = ("Unterschied belegt" if p_h < 0.05 else
                   f"gleichwertig (+-{MARGIN_NATS})" if eq["p_tost"] < 0.05 else
                   f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} vs {b}: {delta:>+8.4f} nats  {w}/{nn_}  '
              f'p={p:.4f}  Holm={p_h:.4f}  {zustand}')
        out[tag] = {"delta_nats": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"]}

    # --- Bodenpruefung: hat ueberhaupt irgendein Arm Kontext gelernt? ---
    print()
    print(f"BODENPRUEFUNG gegen die Unigramm-Baseline ({UNIGRAMM_NATS:.4f} nats, "
          f"ppl {math.exp(UNIGRAMM_NATS):,.0f})")
    boden_ok = True
    for arm in config["arme"]:
        v = vals(arm)
        if not len(v):
            continue
        besser = float(UNIGRAMM_NATS - v.mean())
        ok = besser > 0.10
        boden_ok &= ok
        print(f'  {arm:>16}  {v.mean():.4f} nats  '
              f'{"unterbietet um " + format(besser, "+.4f") if besser > 0 else "SCHLECHTER um " + format(-besser, ".4f")}'
              f'  {"ok" if ok else "ZU NAH AM BODEN"}')
    if not boden_ok:
        print("  ACHTUNG: mindestens ein Arm liegt nicht deutlich unter der "
              "Unigramm-Baseline. Er hat ueber Kontext nichts gelernt, und ein "
              "Vergleich zwischen solchen Armen misst Rauschen.")
    out["bodenpruefung"] = {"passed": bool(boden_ok),
                            "unigramm_nats": UNIGRAMM_NATS}

    dv, nv = vals("dense"), vals("narrow")
    c_p = float(stats.wilcoxon(dv, nv).pvalue)
    c_ok = c_p < 0.05
    richtung = "dense besser, wie erwartet" if dv.mean() < nv.mean() else "NARROW BESSER -- unerwartet"
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"dense gegen narrow {nv.mean() - dv.mean():+.4f} nats, p={c_p:.4f} "
            f"-- {richtung}."
            + ("" if c_ok else " Der Aufbau sieht Kapazitaetsunterschiede im FFN "
               "nicht -- kein FFN-Vergleich oben ist interpretierbar.")
            + (" Die Richtung ist unerwartet; das ist ein Ergebnis, kein "
               "Messversagen." if c_ok and dv.mean() > nv.mean() else ""))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    prim = out["PRIMAER"]
    print()
    if not c_ok:
        v = "TU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_holm"] < 0.05 and prim["delta_nats"] > 0:
        v = (f"T1+ -- der Butterfly-FFN ist um {prim['delta_nats']:+.4f} nats "
             f"BESSER als der dichte, bei 6.488-fach weniger Parametern im "
             f"Block. Unerwartet und deshalb ausdruecklich als solches "
             f"berichtet.")
    elif prim["p_tost"] < 0.05:
        v = (f"T1 -- die Struktur haelt im Sprachmodell: gleichwertig innerhalb "
             f"+-{MARGIN_NATS} nats bei 6.488-fach weniger Parametern im "
             f"FFN-Block. Das ist eine Qualitaetsaussage, keine "
             f"Effizienzaussage -- bei d_model {D_MODEL} ist die Struktur "
             f"gemessen 5,9-fach langsamer.")
    elif prim["p_holm"] < 0.05 and prim["delta_nats"] < 0:
        v = (f"T0 -- die Struktur kostet {abs(prim['delta_nats']):.4f} nats. "
             f"Das ist der Preis fuer 6.488-fach weniger Parameter und gehoert "
             f"neben jede Parameterangabe.")
    else:
        v = f"TU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f} nats."
    print(f"BEWERTUNG: {v}")
    print("=" * 90)
    out["verdict"] = v
    return out


# --------------------------------------------------------------------------
# Textausgabe -- der Teil, der aus dem Lauf einen Chatbot macht
# --------------------------------------------------------------------------

@torch.no_grad()
def erzeuge(arm: str = "dense", anstoss: str = "Die Quantenmechanik ist",
            laenge: int = 120, temperatur: float = 0.8,
            top_k: int = 40) -> str:
    """Laedt die gesicherten Gewichte und schreibt weiter.

    Ehrlich zur Erwartung: 5,3 Millionen Parameter auf deutschem Wikipedia
    ergeben grammatisch brauchbare Saetze mit erfundenem Inhalt. Das ist ein
    Sprachmodell, kein Wissensspeicher -- und es ist die Demonstration der
    QUALITAET, nicht der Effizienz.
    """
    from tokenizers import Tokenizer

    pfad = MODELLE / f"{arm}.pt"
    if not pfad.exists():
        raise FileNotFoundError(f"{pfad} fehlt -- erst trainieren.")
    stand = torch.load(pfad, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_file(str(KORPUS / "bpe_4096.json"))
    modell = SprachModell(arm)
    modell.load_state_dict(stand["state_dict"])
    modell.eval()

    ids = tok.encode(anstoss).ids[-SEQ:]
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(laenge):
        fenster = idx[:, -SEQ:]
        logits = modell(fenster)[:, -1, :] / max(temperatur, 1e-6)
        if top_k:
            grenze = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
            logits = logits.masked_fill(logits < grenze, float("-inf"))
        naechstes = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, naechstes], dim=1)
    return tok.decode(idx[0].tolist())


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else "durchsatz"
    if befehl == "durchsatz":
        durchsatz_alle_arme()
    elif befehl == "erzeuge":
        arm = sys.argv[2] if len(sys.argv) > 2 else "dense"
        anstoss = sys.argv[3] if len(sys.argv) > 3 else "Die Quantenmechanik ist"
        print(erzeuge(arm=arm, anstoss=anstoss))
    elif befehl == "lernratensuche":
        lernratensuche()
    elif befehl == "probe":
        # Kurzer Testlauf: laeuft alles durch, bevor Stunden investiert werden.
        r = trainieren(schritte=60, seeds=SEEDS[:2])
        try:
            analyse(r)
        except Exception as exc:                  # noqa: BLE001
            print(f"  Auswertung fehlgeschlagen: {exc!r}")
        r.save()
    elif befehl == "trainieren":
        schritte = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        r = trainieren(schritte=schritte)
        try:
            a = analyse(r)
        except Exception as exc:                  # noqa: BLE001
            print(f"  Auswertung fehlgeschlagen: {exc!r}")
            a = {"error": repr(exc)}
        r.save(analysis=a)
    else:
        print(f"unbekannt: {befehl}. Erlaubt: durchsatz | lernratensuche | "
              f"probe | trainieren | erzeuge")
        raise SystemExit(1)
