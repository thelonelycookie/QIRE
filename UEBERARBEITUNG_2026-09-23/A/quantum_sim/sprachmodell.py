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

ZUSATZARM (NICHT vorregistriert, standardmaessig AUS, nur beschreibend)
  butterfly_ffn_zentriert   wie butterfly_ffn, aber die Ausgabe wird um ihren
                  Mittelwert ueber die Kanaele verschoben und je Kanal mit einem
                  lernbaren Gewinn multipliziert (256 Parameter je Block).
                  Grund: `ButterflyFFN` gibt |Amplitude|^2 * 256 * scale aus --
                  nichtnegativ, feste Summe, EIN skalarer Gewinn. Der FFN-Block
                  kann damit nichts vom Residualstrom abziehen. Der Arm steht
                  AUSSERHALB der Holm-Familie; sein Ergebnis bestaetigt nichts,
                  es beschreibt nur. Zuschalten mit `--mit-zusatzarm`. Ob er
                  mitlaeuft, entscheidet Malte VOR dem Lauf (Tagebuch).

VORREGISTRIERUNG
  Gemessen wird die Kreuzentropie auf der Validierungsmenge in nats je Token,
  bei GLEICHER Zahl an Gradientenschritten (`SCHRITTE`, einmal festgelegt).
  8 Seeds je Arm.

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

  ERGAENZT AM 23. SEPTEMBER, VOR JEDEM MESSLAUF (Vortest-Befund, siehe
  `vortest()`), Entscheidung Malte:
  (a) Signifikanz wird in BEIDE Richtungen vor der Gleichwertigkeit geprueft.
      Vorher kam T1+ vor T1, T0 aber NACH T1: Ein butterfly_ffn, der
      signifikant, aber innerhalb der Grenze schlechter ist, hiess "T1 -- die
      Struktur haelt", und die messbaren Kosten verschwanden aus dem Urteil.
      Ein besserer hiess dagegen T1+. Die Asymmetrie lief FUER die Hypothese.
      Jetzt: T0 mit dem Zusatz "zugleich innerhalb der angemeldeten Grenze".
  (b) Liegt `dense` selbst nicht deutlich unter der Unigramm-Baseline, lautet
      das Urteil TU (Bodenpruefung) -- vorher wurde die Bodenpruefung nur
      gedruckt und floss in kein Urteil ein.
  (c) Arme werden ueber die SEED-NUMMER gepaart, nicht ueber die Position in
      einer Liste. Fehlte eine Zelle (abgebrochener Clusterjob), stuerzte die
      alte Auswertung ab oder paarte -- bei Luecken in zwei Armen -- still
      falsche Seeds. Unvollstaendige Daten bekommen den Vermerk VORLAEUFIG.

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

UEBERARBEITUNG 23. SEPTEMBER 2026 (Review, BERICHT/t002.md)
  1. Lernratensuche mit VOLLEM Budget (Regel 1c) und Gitterverlaengerung in
     BEIDE Richtungen (Regel 1b). Vorher: 1.000 Schritte gegen 13.000 bzw.
     20.000 im Messlauf, und nur nach oben.
  2. Die Schrittzahl steht EINMAL hier (`SCHRITTE`); CLI und Shellskripte
     leiten sich davon ab. Eine andere Zahl verlangt `--nicht-vorregistriert`.
  3. Jede Zelle (Arm, Seed) bzw. (Arm, Lernrate, Seed) liegt in einer EIGENEN
     Datei. Die Wiederaufnahme liest ALLE Dateien und fuehrt nach (Arm, Seed)
     zusammen, statt nur die neueste -- vorher liess die Probe (60 Schritte)
     als neueste Datei den Hauptlauf still bei null anfangen. Die Probe hat
     jetzt eine eigene ID (T-002-probe).
  4. `"lr"` im Datensatz ist die tatsaechlich benutzte Rate (vorher die
     Konstante 3e-4).
  5. Clusterbetrieb: einzelne Zellen per CLI, `plan` erzeugt die offenen
     Zellen als Zeilen fuer ein SLURM-Job-Array, `zusammenfuehren` sammelt ein
     und wertet aus (server_setup/cluster/).

Run:
  python -m quantum_sim.sprachmodell durchsatz        # Tokens/s je Arm
  python -m quantum_sim.sprachmodell schritte         # vorregistrierte Schrittzahl
  python -m quantum_sim.sprachmodell rechenbedarf     # GPU-Stunden (Schaetzung)
  python -m quantum_sim.sprachmodell selbsttest       # Vortest + Gitter + Wiederaufnahme
  python -m quantum_sim.sprachmodell rauchtest        # Mini-Durchlauf auf CPU
  python -m quantum_sim.sprachmodell probe            # 60 Schritte, eigene ID
  python -m quantum_sim.sprachmodell lernratensuche   # alle LR-Zellen der Reihe nach
  python -m quantum_sim.sprachmodell trainieren       # alle Messzellen der Reihe nach
  python -m quantum_sim.sprachmodell plan lr|messung  # offene Zellen (Cluster)
  python -m quantum_sim.sprachmodell zelle --arm dense --seed 2101
  python -m quantum_sim.sprachmodell zusammenfuehren  # einsammeln + analyse()
"""

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import qire_ai.result_store as _rs
from qire_ai.benchmark import set_seed
from qire_ai.result_store import Recorder, _versions

from .butterfly_layer import apply_1q, cnot_permutation
from .sequence_platform import ButterflyFFN

WURZEL = Path(__file__).resolve().parent.parent
# Pfade per Umgebungsvariable ueberschreibbar. Auf einem Rechencluster liegt
# der Korpus meist auf einem anderen Dateisystem als der Code, und die
# Rechenknoten haben oft kein Internet (server_setup/cluster/README.md).
KORPUS = Path(os.environ.get("QIRE_KORPUS", WURZEL / "results" / "korpus"))
MODELLE = Path(os.environ.get("QIRE_MODELLE", WURZEL / "results" / "modelle"))
ZELLEN_WURZEL = Path(os.environ.get("QIRE_T002_ZELLEN",
                                    WURZEL / "results" / "t002_zellen"))

EXPERIMENT = "T-002"
PROBE_ID = "T-002-probe"       # eigene ID: die Probe darf den Hauptlauf nie
                               # wieder als "neueste Datei" verdraengen

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

# DIE SCHRITTZAHL -- EINMAL, HIER. Vorher stand 13.000 in runpod_t002.sh und
# alles_durchlaufen.sh, 20.000 in runpod_start.sh und im CLI-Standard; welcher
# Wert gemessen worden waere, hing davon ab, welches Skript man startet.
#
# Begruendung fuer 13.000: Token je Parameter.
#   13.000 Schritte x 64 x 128 = 106,5 Mio. Token je Lauf.
#   `dense` hat 5.289.472 Parameter (davon 1.081.344 in Token- und
#   Positionseinbettung) -> 20,1 Token je Parameter insgesamt, 25,3 ohne
#   Einbettungen. Hoffmann et al. 2022 ("Chinchilla") nennen rund 20 Token je
#   Parameter als rechenoptimales Verhaeltnis. 20.000 Schritte waeren 31,0
#   Token je Parameter -- 54 Prozent mehr GPU-Zeit fuer einen Bereich, in dem
#   die Kurve vermutlich schon flacher ist (Erwartung, nicht gemessen). Die
#   kleineren Arme sehen bei gleicher Schrittzahl mehr Token je Parameter
#   (butterfly_ffn 33,4) -- gewollt, das Budget ist die Schrittzahl. 106,5 Mio. Token sind rund ein Viertel des
#   Korpus (397 Mio.); die Stapel werden zufaellig gezogen, eine Epoche wird
#   also nicht wiederholt.
#   (Die Parameterzahlen oben sind mit `SprachModell("dense")` nachgezaehlt,
#   nicht geschaetzt.)
# Eine Abweichung verlangt `--nicht-vorregistriert` und gehoert VOR dem Lauf
# ins Tagebuch.
SCHRITTE = 13000

ARME = ("dense", "butterfly_ffn", "butterfly_qkv", "narrow")
ZUSATZ_ARME = ("butterfly_ffn_zentriert",)   # nicht vorregistriert, s. Kopf
ALLE_ARME = ARME + ZUSATZ_ARME
REGISTRIERT = ("PRIMAER", "P2")
SEEDS = (2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108)   # 8 -> min p 0,0078
MARGIN_NATS = 0.05             # wie in Q-013 angemeldet
# Auf der Validierungsmenge gemessen: eine Vorhersage aus reinen
# Token-Haeufigkeiten ohne jeden Kontext. Ein Arm, der das nicht deutlich
# unterbietet, hat ueber Kontext nichts gelernt -- unabhaengig davon, wie der
# Vergleich mit den anderen Armen ausfaellt.
UNIGRAMM_NATS = 7.0756         # ppl 1.183; Gleichverteilung waere 8,3178
BODEN_ABSTAND = 0.10           # so weit muss ein Arm darunter liegen
LR = 3e-4                      # nur noch Rueckfallwert fuer Probe/Rauchtest
# Regel 1 des Protokolls: Lernrate je Arm auf eigenen Tuning-Seeds, und das
# Gitter muss das Optimum ENTHALTEN. Randlagen werden gemeldet.
LR_GITTER = (3e-4, 6e-4, 1.2e-3, 2.4e-3)
LR_FAKTOR = 2.0                # Verlaengerung: mal 2 nach oben, durch 2 nach
                               # unten -- derselbe Abstand wie im Gitter
ERWEITERUNGEN = 3              # hoechstens so viele Verlaengerungen je Arm,
                               # jeweils in die Richtung des Randes, auf dem
                               # das Optimum liegt. Reichweite damit
                               # 3e-4/8 = 3,75e-05 bis 2,4e-3*8 = 0,0192.
LR_SUCHE_SCHRITTE = SCHRITTE   # Regel 1(c): DASSELBE Budget wie der Messlauf.
                               # Vorher 1.000 -- ein anderer Versuch, der das
                               # Optimum systematisch nach oben verschiebt.
LR_SUCHE_SEEDS = (2199,)       # bewusst NICHT aus SEEDS -- sonst waere die
                               # Lernrate auf denselben Seeds gesucht, auf
                               # denen spaeter gemessen wird. Ein Tupel, damit
                               # ein zweiter Tuning-Seed nur eine Zeile kostet.
WARMUP_SCHRITTE = 200
EVAL_ALLE = 500
EVAL_STAPEL = 40
# Gemessen am 11. August auf dem Pod (RTX 4090, Batch 64), butterfly_ffn --
# Grundlage der Rechenbedarf-SCHAETZUNG, keine Messung der anderen Arme.
S_JE_SCHRITT_4090 = 0.334


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

    Bei d=256: 5 Lagen x 8 Ebenen = 40 Winkel gegen 65.536 Gewichte.
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


class ButterflyFFNZentriert(nn.Module):
    """ZUSATZARM, nicht vorregistriert: Butterfly-FFN mit zentrierter Ausgabe.

    Der Einwand aus dem Review vom 23. September: `ButterflyFFN` liefert
    |Amplitude|^2 * 256 * scale. Das ist
      * nichtnegativ -- der Block kann dem Residualstrom nichts abziehen,
      * von fester Summe (256 * scale je Token) -- er kann nur umverteilen,
      * mit EINEM skalaren Gewinn -- kein Kanal laesst sich einzeln lauter
        oder leiser stellen.
    Ein dichter FFN-Block kann alle drei. Faellt butterfly_ffn deutlich hinter
    dense zurueck (die Probe zeigte -0,32 nats nach 60 Schritten), ist offen,
    ob das die Struktur ist oder nur diese Ausgabeform.

    Diese Variante aendert NUR die Ausgabeform: Mittelwert ueber die Kanaele
    abziehen (die Summe ist dann null, Vorzeichen sind moeglich) und je Kanal
    ein lernbarer Gewinn (256 Parameter). Der Kern bleibt unveraendert
    `ButterflyFFN` aus sequence_platform.py. Dessen `scale` wird eingefroren,
    weil er sonst mit dem Gewinn je Kanal nur eine doppelte Beschreibung
    derselben Groesse waere (ein Freiheitsgrad, der nichts kann).

    Parameter je Block: 80 Winkel + 256 Gewinne = 336 statt 81.
    """

    def __init__(self, d: int, lagen: int = FFN_LAGEN):
        super().__init__()
        self.kern = ButterflyFFN(d, num_layers=lagen, tie_weights=True)
        self.kern.scale.requires_grad_(False)
        self.gewinn = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.kern(x)
        return (p - p.mean(dim=-1, keepdim=True)) * self.gewinn


# --------------------------------------------------------------------------
# Das Modell
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, arm: str, d: int, koepfe: int):
        super().__init__()
        if arm not in ALLE_ARME:
            raise ValueError(f"unbekannter Arm {arm!r}")
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if arm == "butterfly_qkv":
            self.att = ButterflyAttention(d, koepfe)
            self.dicht_att = False
        else:
            self.att = nn.MultiheadAttention(d, koepfe, batch_first=True)
            self.dicht_att = True
        if arm == "butterfly_ffn":
            self.ffn = ButterflyFFN(d, num_layers=FFN_LAGEN, tie_weights=True)
        elif arm == "butterfly_ffn_zentriert":
            self.ffn = ButterflyFFNZentriert(d, lagen=FFN_LAGEN)
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
        """TRAINIERBARE Parameter in dem Teil, der zwischen den Armen
        unterschiedlich ist. (Fuer die vier registrierten Arme ist jeder
        Parameter trainierbar, die Zahlen sind also unveraendert; beim Zusatzarm
        zaehlt das eingefrorene `scale` nicht mit.)"""
        gesamt = 0
        for b in self.blocks:
            if self.arm == "butterfly_qkv":
                gesamt += sum(p.numel() for n, p in b.att.named_parameters()
                              if not n.startswith("aus") and p.requires_grad)
            else:
                gesamt += sum(p.numel() for p in b.ffn.parameters()
                              if p.requires_grad)
        return gesamt


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------

def lade_token(teil: str) -> np.ndarray:
    pfad = KORPUS / f"{teil}.npy"
    if not pfad.exists():
        raise FileNotFoundError(
            f"{pfad} fehlt. Erst 'python -m quantum_sim.korpus_deutsch "
            f"aufbereiten' laufen lassen oder QIRE_KORPUS auf den Ordner mit "
            f"train.npy, val.npy und bpe_4096.json setzen.")
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

def durchsatz_alle_arme(schritte: int = 8, threads: int = 6,
                        arme: Sequence[str] = ARME) -> Dict[str, float]:
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
    kopf = (f'  {"Arm":>24} {"Parameter":>12} {"Struktur":>10} '
            f'{"s/Schritt":>10} {"Token/s":>10} {"relativ":>8}')
    print(kopf); print("  " + "-" * (len(kopf) - 2))
    x = torch.randint(0, VOCAB, (BATCH, SEQ)).to(dev)
    y = torch.randint(0, VOCAB, (BATCH, SEQ)).to(dev)
    werte: Dict[str, float] = {}

    def gleichstand():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for arm in arme:
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
        rel = tps / werte[arme[0]]
        print(f'  {arm:>24} {sum(p.numel() for p in m.parameters()):>12,} '
              f'{m.struktur_parameter():>10,} {dauer:>10.3f} {tps:>10,.0f} '
              f'{rel:>7.2f}x')
    print()
    langsamster = min(werte, key=werte.get)
    faktor = werte[arme[0]] / werte[langsamster]
    print(f"  Langsamster Arm: {langsamster}, Faktor {faktor:.1f} gegenueber {arme[0]}.")
    print(f"  Bei gleicher Schrittzahl kostet der Lauf also das {faktor:.1f}-fache")
    print(f"  der Wanduhrzeit des dichten Arms. Das ist der Umsetzungsnachteil")
    print(f"  aus T-001 und gehoert getrennt berichtet, nicht in die Qualitaet.")
    print("=" * 86)
    return werte


# --------------------------------------------------------------------------
# Training einer Zelle
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
                  threads: int = 6, lr: float = LR, lr_quelle: str = "rueckfall",
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
            print(f"      {arm:>14} seed {seed} lr {lr:g} Schritt {schritt:>6}/{schritte}  "
                  f"val {v:.4f} nats  ppl {math.exp(v):>8.1f}  "
                  f"({time.perf_counter() - t0:,.0f} s)", flush=True)
    dauer = time.perf_counter() - t0
    end_val = verlauf[-1]["val_nats"]
    # Gewichte des ERSTEN Seeds je Arm sichern -- daraus entsteht der Chatbot.
    # Ohne das haette der Lauf am Ende Zahlen, aber kein Modell.
    if sichern and seed == SEEDS[0]:
        MODELLE.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "schritte": schritte,
                    "val_nats": end_val, "d_model": D_MODEL, "lagen": LAGEN,
                    "koepfe": KOEPFE, "seq": SEQ, "vocab": VOCAB, "lr": lr,
                    "state_dict": modell.state_dict()},
                   MODELLE / f"{arm}.pt")
        print(f"      Gewichte gesichert: {MODELLE / (arm + '.pt')}")
    # "lr" ist die TATSAECHLICH benutzte Rate. Bis zum 23. September stand hier
    # die Konstante LR (3e-4) -- jede Zeile haette dieselbe, falsche Rate
    # getragen, egal was die Lernratensuche ergeben hatte.
    return {"arm": arm, "seed": seed, "schritte": schritte,
            "val_nats": end_val, "val_perplexitaet": math.exp(end_val),
            "sekunden": dauer, "token": schritte * BATCH * SEQ,
            "token_je_sekunde": schritte * BATCH * SEQ / dauer,
            "parameter": sum(p.numel() for p in modell.parameters()),
            "struktur_parameter": modell.struktur_parameter(),
            "geraet": str(dev), "verlauf": verlauf,
            "d_model": D_MODEL, "lagen": LAGEN, "koepfe": KOEPFE,
            "seq": SEQ, "batch": BATCH, "vocab": VOCAB,
            "lr": float(lr), "lr_quelle": lr_quelle}


# --------------------------------------------------------------------------
# Zellen auf Platte -- eine Datei je Zelle
# --------------------------------------------------------------------------
# WARUM EINE DATEI JE ZELLE
# Die alte Wiederaufnahme las nur die NEUESTE T-002-Datei. `runpod_start.sh`
# faehrt vor jedem Lauf die Probe (60 Schritte) -- danach war die Probe die
# neueste Datei, alle ihre Zeilen hatten die falsche Schrittzahl, und der
# Hauptlauf fing still bei null an. Auf einem Cluster mit parallelen Jobs
# bricht "die neueste Datei" als Prinzip ganz: jeder Job haette nur seine
# eigene Zelle gesehen.
#
# Jetzt: jede Zelle schreibt ihre eigene Datei mit festem Namen, und die
# Wiederaufnahme liest ALLE Dateien und fuehrt sie nach (Arm, Seed) zusammen.
# Eine vorhandene Datei wird NIE ueberschrieben -- rechnet ein Job eine Zelle
# doppelt, landet das Ergebnis daneben und wird beim Zusammenfuehren als
# Duplikat erkannt.

def zellen_ordner(experiment_id: str = EXPERIMENT) -> Path:
    return ZELLEN_WURZEL / experiment_id


def lr_map_pfad() -> Path:
    return zellen_ordner(EXPERIMENT) / "lr_map.json"


def _lr_text(lr: float) -> str:
    return f"{float(lr):.6g}"


def _lr_norm(lr: float) -> float:
    """Eine Rate, die lokal und auf jedem Clusterknoten gleich aussieht."""
    return float(_lr_text(lr))


def _jetzt() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def _lies_json(pfad: Path) -> Optional[dict]:
    try:
        return json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNUNG: {pfad.name} nicht lesbar ({exc!r}) -- uebergangen",
              file=sys.stderr)
        return None


def _sichere_zelle(pfad: Path, daten: dict) -> Path:
    """Atomar schreiben (erst Temporaerdatei, dann umbenennen), nie
    ueberschreiben. Existiert die Datei schon, bekommt das neue Ergebnis einen
    eigenen Namen daneben."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    ziel = pfad
    if ziel.exists():
        marke = f"{os.environ.get('SLURM_JOB_ID', 'lokal')}_{os.getpid()}"
        ziel = pfad.with_name(f"{pfad.stem}__doppelt_{marke}.json")
        print(f"  HINWEIS: {pfad.name} existiert schon -- neues Ergebnis als "
              f"{ziel.name} daneben (wird beim Zusammenfuehren als Duplikat "
              f"geprueft)")
    tmp = ziel.with_name(f"{ziel.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(daten, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ziel)
    return ziel


def messzelle_pfad(experiment_id: str, arm: str, seed: int, schritte: int) -> Path:
    return zellen_ordner(experiment_id) / "messung" / f"{arm}__s{seed}__n{schritte}.json"


def lrzelle_pfad(arm: str, lr: float, seed: int, schritte: int) -> Path:
    return (zellen_ordner(EXPERIMENT) / "lr"
            / f"{arm}__lr{_lr_text(lr)}__s{seed}__n{schritte}.json")


def _zellen_kopf(experiment_id: str, art: str) -> dict:
    return {"experiment_id": experiment_id, "art": art, "created": _jetzt(),
            "rechner": os.environ.get("SLURMD_NODENAME") or os.environ.get(
                "COMPUTERNAME") or os.environ.get("HOSTNAME", "unbekannt"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "environment": _versions()}


Trainer = Callable[..., Dict[str, object]]


def _echter_trainer(daten_cache: dict, threads: int) -> Trainer:
    """Trainer, der den Korpus erst beim ersten Bedarf laedt -- eine
    Wiederaufnahme, bei der alles schon gerechnet ist, braucht keinen Korpus."""
    def trainer(arm, seed, schritte, lr, lr_quelle, sichern):
        if "zug" not in daten_cache:
            daten_cache["zug"], daten_cache["val"] = lade_token("train"), lade_token("val")
        return trainiere_arm(arm, seed, schritte, geraet(), daten_cache["zug"],
                             daten_cache["val"], threads, lr=lr,
                             lr_quelle=lr_quelle, sichern=sichern)
    return trainer


def rechne_zelle(arm: str, seed: int, schritte: int, lr: float, lr_quelle: str,
                 experiment_id: str = EXPERIMENT, threads: int = 6,
                 trainer: Optional[Trainer] = None) -> Dict[str, object]:
    """Eine Messzelle (Arm, Seed). Liegt sie schon vor, wird sie gelesen, nicht
    neu gerechnet -- ein SLURM-Job, der nach einem Abbruch neu startet, kostet
    dann nichts."""
    lr = _lr_norm(lr)
    pfad = messzelle_pfad(experiment_id, arm, seed, schritte)
    if pfad.exists():
        alt = _lies_json(pfad)
        if alt is not None and "record" in alt:
            alt_lr = float(alt["record"].get("lr", float("nan")))
            if not math.isclose(alt_lr, lr, rel_tol=1e-6):
                raise SystemExit(
                    f"ABBRUCH: {pfad} wurde mit lr {alt_lr:g} gerechnet, jetzt "
                    f"verlangt ist {lr:g}. Die Lernratenkarte hat sich nach der "
                    f"Messung geaendert. Nichts wird ueberschrieben -- einen "
                    f"anderen Zellenordner (QIRE_T002_ZELLEN) waehlen und ins "
                    f"Tagebuch schreiben, warum.")
            print(f"    Zelle {arm} seed {seed} liegt schon vor -- uebersprungen")
            return alt["record"]
    trainer = trainer or _echter_trainer({}, threads)
    z = trainer(arm, seed, schritte, lr, lr_quelle,
                experiment_id == EXPERIMENT)
    kopf = _zellen_kopf(experiment_id, "messzelle")
    kopf["record"] = z
    ziel = _sichere_zelle(pfad, kopf)
    print(f"    Zelle gesichert: {ziel}")
    return z


def rechne_lr_zelle(arm: str, lr: float, seed: int, schritte: int,
                    threads: int = 6,
                    trainer: Optional[Trainer] = None) -> Dict[str, object]:
    lr = _lr_norm(lr)
    pfad = lrzelle_pfad(arm, lr, seed, schritte)
    if pfad.exists():
        alt = _lies_json(pfad)
        if alt is not None and "record" in alt:
            print(f"    LR-Zelle {arm} lr {lr:g} seed {seed} liegt schon vor -- uebersprungen")
            return alt["record"]
    trainer = trainer or _echter_trainer({}, threads)
    z = trainer(arm, seed, schritte, lr, "lernratensuche", False)
    kopf = _zellen_kopf(EXPERIMENT, "lr_zelle")
    kopf["record"] = z
    ziel = _sichere_zelle(pfad, kopf)
    print(f"    LR-Zelle gesichert: {ziel}")
    return z


# --------------------------------------------------------------------------
# Lernratensuche -- volles Budget, Verlaengerung in BEIDE Richtungen
# --------------------------------------------------------------------------

def lr_entscheidung(werte: Dict[float, float], runde: int
                    ) -> Tuple[Optional[float], str]:
    """Ein Schritt der Regel 1(b), rein und ohne Seiteneffekte.

    `werte`: Lernrate -> val_nats (NIEDRIGER ist besser).
    Rueckgabe: (naechste Rate oder None, Vermerk "innen" | "oben" | "unten").

    Liegt das Optimum auf einem Rand, wird in DIESE Richtung verlaengert --
    nach oben mal LR_FAKTOR, nach unten durch LR_FAKTOR, bis zu ERWEITERUNGEN
    Mal. Die alte Fassung verlaengerte nur nach oben. Ein Arm am unteren Rand
    ist genauso untertuniert wie einer am oberen -- nur faellt es weniger auf,
    weil die Verzerrung dort oft FUER die eigene Hypothese laeuft (Q-015 auf
    fashion_mnist, 27. August). Als "innen" gilt ein Optimum erst, wenn auf
    BEIDEN Seiten ein schlechterer Punkt gemessen wurde.
    """
    bester = min(werte, key=werte.get)
    if bester == max(werte):
        return (max(werte) * LR_FAKTOR if runde < ERWEITERUNGEN else None), "oben"
    if bester == min(werte):
        return (min(werte) / LR_FAKTOR if runde < ERWEITERUNGEN else None), "unten"
    return None, "innen"


def _lies_lr_zellen(arm: str, schritte: int) -> Dict[str, Dict[int, float]]:
    """{lr_text: {seed: val_nats}} aus allen LR-Zellen dieses Arms. Bei
    Duplikaten gilt die AELTESTE Datei -- nie die bessere, sonst waere das
    Aussuchen nach dem Ergebnis."""
    ordner = zellen_ordner(EXPERIMENT) / "lr"
    if not ordner.exists():
        return {}
    eintraege = []
    for pfad in ordner.glob(f"{arm}__lr*__n{schritte}*.json"):
        d = _lies_json(pfad)
        if not d or "record" not in d:
            continue
        r = d["record"]
        if (r.get("arm") != arm or int(r.get("schritte", -1)) != schritte
                or int(r.get("batch", -1)) != BATCH):
            continue
        eintraege.append((d.get("created", ""), pfad.name, r))
    out: Dict[str, Dict[int, float]] = {}
    for _, _, r in sorted(eintraege, key=lambda e: (e[0], e[1])):
        out.setdefault(_lr_text(r["lr"]), {}).setdefault(int(r["seed"]),
                                                         float(r["val_nats"]))
    return out


def lr_stand(arm: str, schritte: int) -> Dict[str, object]:
    """Wo steht die Suche fuer diesen Arm? Rechnet den Suchpfad aus den
    vorhandenen Zellen nach -- lokal und auf dem Cluster derselbe Code."""
    vorhanden = _lies_lr_zellen(arm, schritte)
    raten = [_lr_norm(x) for x in LR_GITTER]
    werte: Dict[float, float] = {}
    for runde in range(ERWEITERUNGEN + 1):
        fehlend = [(lr, s) for lr in raten for s in LR_SUCHE_SEEDS
                   if s not in vorhanden.get(_lr_text(lr), {})]
        werte = {lr: float(np.mean([vorhanden[_lr_text(lr)][s]
                                    for s in LR_SUCHE_SEEDS]))
                 for lr in raten if (lr, LR_SUCHE_SEEDS[0]) not in fehlend
                 and all((lr, s) not in fehlend for s in LR_SUCHE_SEEDS)}
        if fehlend:
            return {"fertig": False, "fehlend": fehlend, "werte": werte,
                    "vermerk": "offen", "runde": runde, "bester": None}
        naechste, vermerk = lr_entscheidung(werte, runde)
        if naechste is None:
            return {"fertig": True, "fehlend": [], "werte": werte,
                    "vermerk": vermerk, "runde": runde,
                    "bester": min(werte, key=werte.get)}
        raten = raten + [_lr_norm(naechste)]
    raise AssertionError("unerreichbar: die letzte Runde verlaengert nie")


def offene_lr_zellen(arme: Sequence[str], schritte: int
                     ) -> List[Tuple[str, float, int]]:
    offen = []
    for arm in arme:
        st = lr_stand(arm, schritte)
        offen += [(arm, lr, s) for lr, s in st["fehlend"]]
    return offen


def schreibe_lr_map(arme: Sequence[str], schritte: int) -> Dict[str, float]:
    """Schreibt lr_map.json, sobald die Suche fuer alle Arme fertig ist, und
    meldet Randlagen -- auf der Konsole UND in der Datei."""
    staende = {a: lr_stand(a, schritte) for a in arme}
    offen = [a for a, st in staende.items() if not st["fertig"]]
    if offen:
        raise RuntimeError(f"Lernratensuche nicht fertig fuer: {', '.join(offen)}")
    karte = {a: float(st["bester"]) for a, st in staende.items()}
    vermerk = {a: str(st["vermerk"]) for a, st in staende.items()}
    rand = [a for a in arme if vermerk[a] != "innen"]
    print(f"  LERNRATENKARTE ({schritte:,} Schritte, Seeds {list(LR_SUCHE_SEEDS)})")
    for a in arme:
        w = staende[a]["werte"]
        print(f"  {a:>24} -> lr {karte[a]:<9g}"
              + (f" ACHTUNG: Rand {vermerk[a]}" if vermerk[a] != "innen" else " innen    ")
              + "  " + "  ".join(f"{k:g}:{w[k]:.4f}" for k in sorted(w)))
    if rand:
        print(f"  WARNUNG: am Gitterrand nach {ERWEITERUNGEN} Verlaengerungen -> "
              f"{', '.join(rand)}. Das Optimum koennte ausserhalb liegen; der")
        print(f"  Arm ist dann nicht ausgereizt. So berichten -- und pruefen, ob "
              f"die Verzerrung FUER oder GEGEN die Hypothese laeuft.")
    else:
        print("  Kein Arm am Gitterrand -- das Gitter enthaelt die Optima.")
    ziel = lr_map_pfad()
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_name(ziel.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(
        {"lr_map": karte, "vermerk": vermerk, "am_rand": rand,
         "tabelle": {a: {_lr_text(k): v for k, v in sorted(staende[a]["werte"].items())}
                     for a in arme},
         "gitter": list(LR_GITTER), "faktor": LR_FAKTOR,
         "erweiterungen": ERWEITERUNGEN, "schritte": schritte,
         "seeds": list(LR_SUCHE_SEEDS), "batch": BATCH,
         "erstellt": _jetzt()}, indent=2), encoding="utf-8")
    os.replace(tmp, ziel)
    print(f"  gesichert: {ziel}")
    return karte


def lernratensuche(schritte: int = LR_SUCHE_SCHRITTE, arme: Sequence[str] = ARME,
                   threads: int = 6, trainer: Optional[Trainer] = None,
                   nicht_vorregistriert: bool = False) -> Dict[str, float]:
    """Regel 1 in allen drei Teilen: eigene Tuning-Seeds (a), Gitter muss das
    Optimum enthalten, Verlaengerung in beide Richtungen (b), dasselbe Budget
    wie der Messlauf (c).

    Die Suche ist wiederaufnehmbar: jede Zelle liegt in einer eigenen Datei,
    der Suchpfad wird bei jedem Aufruf aus den vorhandenen Zellen neu
    nachgerechnet (`lr_stand`). Auf dem Cluster rechnet `plan lr` dieselben
    offenen Zellen aus und verteilt sie auf ein Job-Array.
    """
    _pruefe_schritte(schritte, nicht_vorregistriert)
    trainer = trainer or _echter_trainer({}, threads)
    print("=" * 86)
    print(f"LERNRATENSUCHE -- {schritte:,} Schritte je Zelle (= Messlauf), "
          f"Seeds {list(LR_SUCHE_SEEDS)}, Arme {', '.join(arme)}")
    print("=" * 86)
    vorige = None
    while True:
        offen = offene_lr_zellen(arme, schritte)
        if not offen:
            break
        if offen == vorige:
            raise RuntimeError(f"Dieselben Zellen sind nach dem Rechnen noch "
                               f"offen -- Schreibfehler? {offen[:3]}")
        vorige = offen
        for arm, lr, seed in offen:
            rechne_lr_zelle(arm, lr, seed, schritte, threads, trainer=trainer)
    return schreibe_lr_map(arme, schritte)


def lies_lr_map(schritte: int, arme: Sequence[str], rueckfall: bool = False
                ) -> Tuple[Dict[str, float], str]:
    """Die Lernratenkarte -- STRENG. Fehlt sie, wurde sie mit einer anderen
    Schrittzahl gesucht oder fehlt ein Arm, bricht der Messlauf ab. Vorher lief
    er dann mit 3e-4 weiter und druckte nur eine Warnung, die im Log eines
    unbeaufsichtigten Pods niemand liest. Nur Probe und Rauchtest duerfen mit
    `rueckfall=True` auf LR ausweichen."""
    pfad = lr_map_pfad()
    fehler = None
    d: dict = {}
    if not pfad.exists():
        fehler = f"{pfad} fehlt"
    else:
        d = _lies_json(pfad) or {}
        if int(d.get("schritte", -1)) != schritte:
            fehler = (f"lr_map.json wurde mit {d.get('schritte')} Schritten gesucht, "
                      f"gemessen wird mit {schritte} -- Regel 1(c)")
        elif int(d.get("batch", -1)) != BATCH:
            fehler = f"lr_map.json gilt fuer Batch {d.get('batch')}, nicht {BATCH}"
        else:
            fehlt = [a for a in arme if a not in d.get("lr_map", {})]
            if fehlt:
                fehler = f"lr_map.json enthaelt keine Rate fuer {', '.join(fehlt)}"
    if fehler:
        if rueckfall:
            print(f"  WARNUNG: {fehler} -- alle Arme laufen mit {LR:g} "
                  f"(nur fuer Probe/Rauchtest zulaessig, Regel 1 ist verletzt).")
            return {a: LR for a in arme}, "rueckfall"
        raise SystemExit(f"ABBRUCH: {fehler}. Erst 'lernratensuche' (bzw. auf "
                         f"dem Cluster 'plan lr') vollstaendig laufen lassen.")
    return {a: float(d["lr_map"][a]) for a in arme}, "lr_map"


def _pruefe_schritte(schritte: int, nicht_vorregistriert: bool) -> None:
    if schritte != SCHRITTE and not nicht_vorregistriert:
        raise SystemExit(
            f"ABBRUCH: {schritte:,} Schritte sind nicht die vorregistrierte "
            f"Schrittzahl {SCHRITTE:,} (Konstante SCHRITTE). Eine Abweichung "
            f"gehoert VOR dem Lauf ins Tagebuch und verlangt "
            f"--nicht-vorregistriert.")


# --------------------------------------------------------------------------
# Messlauf, Wiederaufnahme, Zusammenfuehren
# --------------------------------------------------------------------------

def basis_config(arme: Sequence[str], seeds: Sequence[int], schritte: int,
                 lr_map: Dict[str, float], lr_quelle: str, geraet_text: str,
                 threads: Optional[int]) -> Dict[str, object]:
    return {"arme": list(arme), "seeds": list(seeds), "schritte": schritte,
            "vorregistrierte_schritte": SCHRITTE,
            "d_model": D_MODEL, "lagen": LAGEN, "koepfe": KOEPFE,
            "seq": SEQ, "batch": BATCH, "vocab": VOCAB, "lr_map": lr_map,
            "lr_quelle": lr_quelle,
            "lr_suche": {"gitter": list(LR_GITTER), "faktor": LR_FAKTOR,
                         "erweiterungen": ERWEITERUNGEN,
                         "seeds": list(LR_SUCHE_SEEDS),
                         "schritte": LR_SUCHE_SCHRITTE},
            "ffn_lagen": FFN_LAGEN, "qkv_lagen": QKV_LAGEN,
            "margin_nats": MARGIN_NATS, "geraet": geraet_text,
            "korpus": "deutsche Wikipedia 20231101, 397 Mio. Token",
            "zeichen_je_token": 3.27,
            "messgroesse": "val_nats (Kreuzentropie je Token)",
            "primary": "butterfly_ffn gleichwertig zu dense (+-0,05 nats)",
            "p2": "butterfly_qkv gegen dense",
            "holm_family": list(REGISTRIERT),
            "zusatz_arme_beschreibend": [a for a in arme if a in ZUSATZ_ARME],
            "resolution_check": "dense muss narrow schlagen",
            "budget": "gleiche Gradientenschritte, NICHT gleiche Zeit -- "
                      "sonst misst die Qualitaet den Umsetzungsnachteil aus "
                      "T-001 mit",
            "torch_threads": threads}


def sammle_messungen(experiment_id: str, schritte: int,
                     lr_map: Optional[Dict[str, float]] = None
                     ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Liest ALLE Rohdaten eines Experiments und fuehrt sie nach (Arm, Seed)
    zusammen.

    Quellen: jede `results/experiments/<ID>__*.json` (die Sammeldateien, die
    `trainieren` nach jedem Seed schreibt) und jede Zellendatei unter
    `results/t002_zellen/<ID>/messung/`. Die Probe hat eine eigene ID und wird
    deshalb gar nicht erst gelesen.

    Regeln:
      * andere Schrittzahl oder Batchgroesse -> ausgeschlossen (gezaehlt),
      * andere Lernrate als in der Karte -> ausgeschlossen (gezaehlt),
      * dieselbe Zelle mehrfach mit GLEICHEM Wert -> ein Duplikat, eines zaehlt,
      * dieselbe Zelle mehrfach mit ANDEREM Wert -> Widerspruch, gemeldet; es
        gilt die AELTESTE Messung. Nie die bessere -- das waere Aussuchen nach
        dem Ergebnis.
    """
    eintraege = []
    quellen: List[str] = []
    ergebnisse = _rs.RESULTS_DIR
    if ergebnisse.exists():
        for pfad in sorted(ergebnisse.glob(f"{experiment_id}__*.json")):
            d = _lies_json(pfad)
            if not d or d.get("experiment_id") != experiment_id:
                continue
            quellen.append(str(pfad))
            for r in d.get("records", []):
                eintraege.append((d.get("created", ""), pfad.name, r))
    ordner = zellen_ordner(experiment_id) / "messung"
    if ordner.exists():
        for pfad in sorted(ordner.glob("*.json")):
            d = _lies_json(pfad)
            if not d or "record" not in d:
                continue
            quellen.append(str(pfad))
            eintraege.append((d.get("created", ""), pfad.name, d["record"]))
    eintraege.sort(key=lambda e: (e[0], e[1]))

    gewaehlt: Dict[Tuple[str, int], Tuple[str, Dict[str, object]]] = {}
    ausgeschlossen: Dict[str, int] = {}
    identisch = 0
    widersprueche: List[Dict[str, object]] = []

    def raus(grund: str) -> None:
        ausgeschlossen[grund] = ausgeschlossen.get(grund, 0) + 1

    for _, quelle, r in eintraege:
        if int(r.get("schritte", -1)) != schritte:
            raus("andere Schrittzahl"); continue
        if int(r.get("batch", -1)) != BATCH:
            raus("andere Batchgroesse"); continue
        arm = str(r["arm"])
        if lr_map is not None and arm in lr_map and not math.isclose(
                float(r.get("lr", float("nan"))), lr_map[arm], rel_tol=1e-6):
            raus("andere Lernrate als in lr_map.json"); continue
        schluessel = (arm, int(r["seed"]))
        if schluessel not in gewaehlt:
            gewaehlt[schluessel] = (quelle, r)
            continue
        erste_quelle, erste = gewaehlt[schluessel]
        if abs(float(erste["val_nats"]) - float(r["val_nats"])) < 1e-9:
            identisch += 1
        else:
            widersprueche.append({"arm": arm, "seed": schluessel[1],
                                  "behalten": float(erste["val_nats"]),
                                  "behalten_aus": erste_quelle,
                                  "verworfen": float(r["val_nats"]),
                                  "verworfen_aus": quelle})
    records = [gewaehlt[k][1] for k in sorted(gewaehlt)]
    bericht = {"quellen": quellen, "zellen": len(records),
               "duplikate_identisch": identisch, "widersprueche": widersprueche,
               "ausgeschlossen": ausgeschlossen}
    return records, bericht


def _drucke_sammelbericht(bericht: Dict[str, object]) -> None:
    print(f"  Gelesen: {len(bericht['quellen'])} Dateien -> {bericht['zellen']} "
          f"Zellen, {bericht['duplikate_identisch']} identische Duplikate")
    for grund, n in dict(bericht["ausgeschlossen"]).items():
        print(f"  ausgeschlossen ({grund}): {n}")
    for w in bericht["widersprueche"]:
        print(f"  WIDERSPRUCH {w['arm']} seed {w['seed']}: {w['behalten']:.6f} "
              f"({w['behalten_aus']}) gegen {w['verworfen']:.6f} "
              f"({w['verworfen_aus']}) -- die aeltere Messung gilt")


def trainieren(schritte: int = SCHRITTE, seeds: Sequence[int] = SEEDS,
               arme: Sequence[str] = ARME, threads: int = 6,
               experiment_id: str = EXPERIMENT, lr_rueckfall: bool = False,
               nicht_vorregistriert: bool = False,
               trainer: Optional[Trainer] = None) -> Recorder:
    _pruefe_schritte(schritte, nicht_vorregistriert)
    dev = geraet()
    lr_map, lr_quelle = lies_lr_map(schritte, arme, rueckfall=lr_rueckfall)
    rec = Recorder(
        experiment_id=experiment_id,
        description=("Haelt die quantenabgeleitete Struktur in einem "
                     "Sprachmodell? Deutsches Wikipedia, gleiche Zahl an "
                     "Gradientenschritten, 8 Seeds."),
        config=basis_config(arme, seeds, schritte, lr_map, lr_quelle,
                            str(dev), threads),
    )
    # --- Wiederaufnahme -----------------------------------------------
    # Am 12. August ist ein fertiger 22-Stunden-Lauf verloren gegangen, weil
    # der gemietete Rechner geloescht wurde. Seither gilt: Ein abgebrochener
    # Lauf darf nicht bei null anfangen. Bereits gerechnete (Arm, Seed)-Paare
    # werden aus ALLEN Rohdatendateien dieses Experiments uebernommen (nicht
    # nur aus der neuesten -- siehe `sammle_messungen`).
    #
    # Das ist Regel 5 des Protokolls ("Rohdaten nach jedem Abschnitt sichern"),
    # eine Ebene hoeher gedacht: Nicht nur speichern, sondern auch wieder
    # aufsetzen koennen.
    vorher, bericht = sammle_messungen(experiment_id, schritte, lr_map)
    gewollt = {(a, int(s)) for a in arme for s in seeds}
    fertig = set()
    for r in vorher:
        k = (str(r["arm"]), int(r["seed"]))
        if k in gewollt:
            fertig.add(k)
            rec.add(**r)
    rec.config["wiederaufnahme"] = bericht

    print("=" * 90)
    print(f"{experiment_id} | {len(arme)} Arme x {len(seeds)} Seeds x "
          f"{schritte:,} Schritte auf {dev}")
    if fertig:
        print(f"WIEDERAUFNAHME: {len(fertig)} von {len(gewollt)} Zellen "
              f"schon gerechnet, werden uebersprungen")
        _drucke_sammelbericht(bericht)
    print("=" * 90)
    trainer = trainer or _echter_trainer({}, threads)
    for seed in seeds:
        offen = [a for a in arme if (a, int(seed)) not in fertig]
        if not offen:
            print(f"    seed {seed} vollstaendig vorhanden -- uebersprungen")
            continue
        for arm in offen:
            rec.add(**rechne_zelle(arm, int(seed), schritte, lr_map[arm],
                                   lr_quelle, experiment_id, threads,
                                   trainer=trainer))
        rec.save()          # nach JEDEM Seed sichern, nicht am Ende
        print(f"    Zwischenstand nach seed {seed} gesichert "
              f"({len(rec.records)} Zellen gesamt)")
    return rec


def mess_plan(arme: Sequence[str], schritte: int,
              experiment_id: str = EXPERIMENT) -> List[Tuple[str, int]]:
    """Offene Messzellen, in Seed-Reihenfolge -- so sind nach einem Teilabbruch
    moeglichst viele Seeds VOLLSTAENDIG (alle Arme), und die gepaarte Analyse
    verliert am wenigsten."""
    lr_map, _ = lies_lr_map(schritte, arme)
    vorher, _ = sammle_messungen(experiment_id, schritte, lr_map)
    da = {(str(r["arm"]), int(r["seed"])) for r in vorher}
    return [(a, int(s)) for s in SEEDS for a in arme if (a, int(s)) not in da]


def zusammenfuehren(experiment_id: str = EXPERIMENT, schritte: int = SCHRITTE,
                    speichern: bool = True) -> Tuple[Recorder, Dict[str, object]]:
    """Sammelt alle Zellen ein (Cluster, Pod, Laptop -- egal woher), meldet
    Duplikate, Widersprueche und Luecken, ruft `analyse()` auf und speichert
    EINE vollstaendige Sammeldatei `results/experiments/<ID>__*.json` mit der
    Auswertung. Diese Datei liest dann auch `q_summary`."""
    lr_map: Optional[Dict[str, float]] = None
    pfad = lr_map_pfad()
    if pfad.exists():
        d = _lies_json(pfad) or {}
        if int(d.get("schritte", -1)) == schritte:
            lr_map = {a: float(v) for a, v in d.get("lr_map", {}).items()}
    if lr_map is None:
        print("  HINWEIS: keine passende lr_map.json -- Lernraten werden beim "
              "Zusammenfuehren NICHT geprueft.")
    records, bericht = sammle_messungen(experiment_id, schritte, lr_map)
    print("=" * 90)
    print(f"ZUSAMMENFUEHREN {experiment_id}, {schritte:,} Schritte")
    _drucke_sammelbericht(bericht)
    arme = list(ARME) + [a for a in ZUSATZ_ARME
                         if any(r["arm"] == a for r in records)]
    da = {(str(r["arm"]), int(r["seed"])) for r in records}
    fehlend = [(a, s) for s in SEEDS for a in arme if (a, s) not in da]
    if fehlend:
        print(f"  FEHLEND ({len(fehlend)}): "
              + ", ".join(f"{a}/{s}" for a, s in fehlend[:12])
              + (" ..." if len(fehlend) > 12 else ""))
    print("=" * 90)
    rec = Recorder(
        experiment_id=experiment_id,
        description=("Haelt die quantenabgeleitete Struktur in einem "
                     "Sprachmodell? Zusammengefuehrt aus Einzelzellen."),
        config=basis_config(arme, SEEDS, schritte, lr_map or {},
                            "lr_map" if lr_map else "ungeprueft",
                            "siehe Messzeilen", None))
    rec.config["zusammenfuehrung"] = dict(bericht, fehlend=[list(x) for x in fehlend])
    for r in records:
        rec.add(**r)
    if not records:
        print("  Keine Zellen gefunden -- nichts auszuwerten.")
        return rec, {}
    out = analyse(rec)
    if speichern:
        rec.save(analysis=out)
    return rec, out


# --------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------

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
    soll_n = len(config.get("seeds", SEEDS))

    def nach_seed(arm: str) -> Dict[int, float]:
        return {int(r["seed"]): float(r["val_nats"])
                for r in records if r["arm"] == arm}

    def vals(arm: str) -> np.ndarray:
        d = nach_seed(arm)
        return np.array([d[s] for s in sorted(d)])

    def gepaart(a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
        # Paarung ueber die SEED-NUMMER. Die alte Fassung sortierte jeden Arm
        # fuer sich und paarte nach Position: Fehlt in EINEM Arm ein Seed,
        # stuerzte sie ab (nachgeprueft, Vortest Szenario I); fehlt in zwei
        # Armen je ein ANDERER Seed, sind die Laengen gleich, und ab der Luecke
        # waeren falsche Seeds gepaart worden -- ohne Fehlermeldung.
        da, db = nach_seed(a), nach_seed(b)
        gem = sorted(set(da) & set(db))
        return np.array([da[s] for s in gem]), np.array([db[s] for s in gem])

    def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.all(x - y == 0):
            return 1.0
        return float(stats.wilcoxon(x, y).pvalue)

    print()
    print(f"RESEARCH ID T-002 | Sprachmodell, d_model {D_MODEL}, deutscher Text")
    kopf = (f'  {"Arm":>24} {"n":>3} {"Parameter":>11} {"Struktur":>9} {"val nats":>9} '
            f'{"std":>7} {"Perplex.":>9} {"Token/s":>9}')
    print(kopf); print("  " + "-" * (len(kopf) - 2))
    for arm in config["arme"]:
        v = vals(arm)
        if not len(v):
            continue
        r0 = next(r for r in records if r["arm"] == arm)
        tps = float(np.median([r["token_je_sekunde"] for r in records
                               if r["arm"] == arm]))
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f'  {arm:>24} {len(v):>3} {r0["parameter"]:>11,} {r0["struktur_parameter"]:>9,} '
              f'{v.mean():>9.4f} {sd:>7.4f} '
              f'{math.exp(v.mean()):>9.1f} {tps:>9,.0f}')

    fam = [("butterfly_ffn", "dense", "PRIMAER"), ("butterfly_qkv", "dense", "P2")]
    roh, zellen = [], []
    for a, b, tag in fam:
        va, vb = gepaart(a, b)
        p = wilcoxon_p(va, vb)
        roh.append(p)
        eq = (tost_paired(va, vb, MARGIN_NATS) if len(va) >= 2
              else {"p_tost": 1.0, "achieved": float("inf")})
        # Niedriger ist besser bei Kreuzentropie -> Vorzeichen umdrehen, damit
        # "positiv = besser" gilt wie in den Q-Laeufen.
        delta = float(vb.mean() - va.mean()) if len(va) else float("nan")
        zellen.append((a, b, tag, delta, p, eq, int((va < vb).sum()), len(va)))
    adj = holm(roh)

    print()
    print("=" * 90)
    print(f"VORREGISTRIERTE FAMILIE, Grenze +-{MARGIN_NATS} nats "
          f"(positiv = Butterfly besser)")
    for (a, b, tag, delta, p, eq, w, nn_), p_h in zip(zellen, adj):
        zustand = ("zu wenige gepaarte Seeds" if nn_ < 2 else
                   "Unterschied belegt" if p_h < 0.05 else
                   f"gleichwertig (+-{MARGIN_NATS})" if eq["p_tost"] < 0.05 else
                   f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} vs {b}: {delta:>+8.4f} nats  {w}/{nn_}  '
              f'p={p:.4f}  Holm={p_h:.4f}  {zustand}')
        out[tag] = {"delta_nats": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"], "n": nn_}

    # --- Bodenpruefung: hat ueberhaupt irgendein Arm Kontext gelernt? ---
    print()
    print(f"BODENPRUEFUNG gegen die Unigramm-Baseline ({UNIGRAMM_NATS:.4f} nats, "
          f"ppl {math.exp(UNIGRAMM_NATS):,.0f})")
    boden_ok = True
    boden_je_arm: Dict[str, bool] = {}
    for arm in config["arme"]:
        v = vals(arm)
        if not len(v):
            continue
        besser = float(UNIGRAMM_NATS - v.mean())
        ok = besser > BODEN_ABSTAND
        boden_je_arm[arm] = ok
        boden_ok &= ok
        print(f'  {arm:>24}  {v.mean():.4f} nats  '
              f'{"unterbietet um " + format(besser, "+.4f") if besser > 0 else "SCHLECHTER um " + format(-besser, ".4f")}'
              f'  {"ok" if ok else "ZU NAH AM BODEN"}')
    if not boden_ok:
        print("  ACHTUNG: mindestens ein Arm liegt nicht deutlich unter der "
              "Unigramm-Baseline. Er hat ueber Kontext nichts gelernt, und ein "
              "Vergleich zwischen solchen Armen misst Rauschen.")
    dense_boden = boden_je_arm.get("dense", False)
    out["bodenpruefung"] = {"passed": bool(boden_ok), "dense_passed": bool(dense_boden),
                            "je_arm": boden_je_arm, "unigramm_nats": UNIGRAMM_NATS}

    dv, nv = gepaart("dense", "narrow")
    c_p = wilcoxon_p(dv, nv)
    c_ok = c_p < 0.05
    d_mean = float(dv.mean()) if len(dv) else float("nan")
    n_mean = float(nv.mean()) if len(nv) else float("nan")
    richtung = ("dense besser, wie erwartet" if d_mean < n_mean
                else "NARROW BESSER -- unerwartet")
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"dense gegen narrow {n_mean - d_mean:+.4f} nats, p={c_p:.4f} "
            f"-- {richtung}."
            + ("" if c_ok else " Der Aufbau sieht Kapazitaetsunterschiede im FFN "
               "nicht -- kein FFN-Vergleich oben ist interpretierbar.")
            + (" Die Richtung ist unerwartet; das ist ein Ergebnis, kein "
               "Messversagen." if c_ok and d_mean > n_mean else ""))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    # --- Beschreibend, AUSSERHALB der Familie -----------------------------
    beschreibend: Dict[str, object] = {}
    for zarm in ZUSATZ_ARME:
        if not nach_seed(zarm):
            continue
        if not beschreibend:
            print()
            print("BESCHREIBEND -- ausserhalb der vorregistrierten Familie, "
                  "ohne Holm, bestaetigt nichts")
        for gegen in ("dense", "butterfly_ffn"):
            za, zb = gepaart(zarm, gegen)
            if len(za) < 2:
                continue
            dz = float(zb.mean() - za.mean())
            pz = wilcoxon_p(za, zb)
            print(f"  {zarm} vs {gegen}: {dz:>+8.4f} nats  "
                  f"{int((za < zb).sum())}/{len(za)}  p(roh)={pz:.4f}")
            beschreibend[f"{zarm}_vs_{gegen}"] = {"delta_nats": dz, "p_roh": pz,
                                                  "n": len(za)}
    if beschreibend:
        out["beschreibend"] = beschreibend

    prim = out["PRIMAER"]
    n_gepaart = min([int(prim["n"]), int(out["P2"]["n"]), len(dv)])
    vollstaendig = n_gepaart >= soll_n
    out["vollstaendig"] = bool(vollstaendig)
    print()
    if not dense_boden:
        v = ("TU -- Bodenpruefung durchgefallen: dense liegt nicht deutlich "
             "unter der Unigramm-Baseline, hat also ueber Kontext nichts "
             "gelernt. Lauf nicht verwertbar.")
    elif not c_ok:
        v = "TU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["n"] < 2:
        v = f"TU -- zu wenige gepaarte Seeds ({prim['n']}) fuer eine Auswertung."
    elif prim["p_holm"] < 0.05 and prim["delta_nats"] > 0:
        v = (f"T1+ -- der Butterfly-FFN ist um {prim['delta_nats']:+.4f} nats "
             f"BESSER als der dichte, bei 6.488-fach weniger Parametern im "
             f"Block. Unerwartet und deshalb ausdruecklich als solches "
             f"berichtet."
             + (f" Der Unterschied liegt zugleich innerhalb der angemeldeten "
                f"Grenze +-{MARGIN_NATS} (TOST p={prim['p_tost']:.4f})."
                if prim["p_tost"] < 0.05 else ""))
    elif prim["p_holm"] < 0.05 and prim["delta_nats"] < 0:
        # Vor T1 geprueft (Ergaenzung vom 23. September, siehe Kopf): ein
        # signifikanter Nachteil darf nicht hinter "die Struktur haelt"
        # verschwinden, nur weil er kleiner als die Grenze ist.
        v = (f"T0 -- die Struktur kostet {abs(prim['delta_nats']):.4f} nats."
             + (f" Der Unterschied liegt zugleich innerhalb der angemeldeten "
                f"Grenze +-{MARGIN_NATS} nats (TOST p={prim['p_tost']:.4f}): "
                f"gleichwertig im Sinne der Anmeldung, aber messbar "
                f"schlechter -- beides berichten."
                if prim["p_tost"] < 0.05 else "")
             + f" Das ist der Preis fuer 6.488-fach weniger Parameter und "
               f"gehoert neben jede Parameterangabe.")
    elif prim["p_tost"] < 0.05:
        v = (f"T1 -- die Struktur haelt im Sprachmodell: gleichwertig innerhalb "
             f"+-{MARGIN_NATS} nats bei 6.488-fach weniger Parametern im "
             f"FFN-Block. Das ist eine Qualitaetsaussage, keine "
             f"Effizienzaussage -- bei d_model {D_MODEL} ist die Struktur "
             f"gemessen 5,9-fach langsamer.")
    else:
        v = f"TU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f} nats."
    if not vollstaendig:
        v += (f" [VORLAEUFIG: nur {n_gepaart} von {soll_n} Seeds gepaart -- "
              f"Zellen fehlen]")
    print(f"BEWERTUNG: {v}")
    print("=" * 90)
    out["verdict"] = v
    return out


# --------------------------------------------------------------------------
# Vortest -- Pflicht vor jedem mehrstuendigen Lauf (CLAUDE.md, Abschnitt 5)
# --------------------------------------------------------------------------

def _kunst(seed: int, delta_ffn: float = 0.0, delta_qkv: float = 0.0,
           streuung: float = 0.01, narrow_ab: float = 0.10, niveau: float = 4.5,
           zusatz: Optional[float] = None, fehlend: Sequence[Tuple[str, int]] = ()
           ) -> Recorder:
    """Synthetische Recorder-Daten mit bekannter Wahrheit.

    val_nats = niveau + Seed-Anteil + Versatz des Arms + EIGENES Rauschen je
    (Arm, Seed). Positives delta heisst: Butterfly BESSER (weniger nats).
    Jeder Arm bekommt eigenes Rauschen -- bei Q-019 bekam der Basisarm keines,
    und der gemeinsame Seed-Anteil kuerzte sich in der gepaarten Differenz weg.
    Der Seed-Anteil ist gross (0,05), damit eine Auswertung, die NICHT paart,
    anders urteilen wuerde als eine, die paart.

    Der letzte Verlaufspunkt ist gleich val_nats; ein frueherer Verlaufspunkt
    ist bewusst NICHT konstant versetzt -- eine Auswertung, die das falsche
    Feld nimmt, wuerde anders urteilen.
    """
    import zlib

    def _rng(*teile) -> np.random.Generator:
        marke = "|".join(str(x) for x in teile)
        return np.random.default_rng([seed, zlib.crc32(marke.encode())])

    arme = list(ARME) + ([ZUSATZ_ARME[0]] if zusatz is not None else [])
    rec = Recorder(experiment_id="T-002-vortest", description="synthetisch",
                   config={"arme": arme, "seeds": list(SEEDS)})
    versatz = {"dense": 0.0, "butterfly_ffn": -delta_ffn,
               "butterfly_qkv": -delta_qkv, "narrow": narrow_ab}
    if zusatz is not None:
        versatz[ZUSATZ_ARME[0]] = -zusatz
    for s in SEEDS:
        grund = niveau + _rng("grund", s).normal(0, 0.05)
        for arm in arme:
            if (arm, s) in fehlend:
                continue
            w = float(grund + versatz[arm] + _rng(arm, s).normal(0, streuung))
            frueh = float(w + 0.3 + _rng("frueh", arm, s).normal(0, 0.2))
            rec.add(arm=arm, seed=s, schritte=SCHRITTE, val_nats=w,
                    val_perplexitaet=math.exp(w), token_je_sekunde=1000.0,
                    parameter=100, struktur_parameter=10, lr=LR, batch=BATCH,
                    verlauf=[{"schritt": SCHRITTE // 2, "val_nats": frueh},
                             {"schritt": SCHRITTE, "val_nats": w}])
    return rec


def vortest() -> bool:
    """Jeder vorregistrierte Ausgang wird einzeln erzwungen, ueber mehrere
    Rauschziehungen, und geprueft wird der VERZWEIGUNGSTEXT, nicht nur die
    Marke -- T0 mit und ohne "zugleich innerhalb der Grenze" teilen sich eine
    Marke, TU hat vier Zweige.

    Befund (23./24. September), nachgeprueft mit der ALTEN analyse() aus
    projects/QIRE AI/A auf denselben synthetischen Daten:
      * Szenario D ("signifikant schlechter, aber innerhalb der Grenze"):
        alt 5 von 5 Ziehungen "T1 -- die Struktur haelt", neu T0 mit dem
        Zusatz "zugleich innerhalb der angemeldeten Grenze". Behoben durch die
        Reihenfolge in `analyse()`, siehe Kopf der Datei, Ergaenzung (a).
      * Szenario I (eine Zelle fehlt): alt 5 von 5 Absturz ("Array shapes are
        incompatible"), neu Urteil mit Vermerk VORLAEUFIG.
    """
    import contextlib
    import io
    from collections import Counter

    # (Name, Aufbau, Marke, muss enthalten, darf nicht enthalten, Ziehungen,
    #  geforderter Anteil)
    #
    # Anteil 1,0 bei klar getrennten Faellen. Bei Faellen, deren Wahrheit ein
    # Nulleffekt ist (A, E, F, H), spricht ein Test auf Niveau 0,05 in einem
    # Teil der Ziehungen an -- das ist sein Niveau, kein Fehler der
    # Auswertung. Dort wird ueber mehr Ziehungen der ANTEIL geprueft (wie
    # Szenario G in zeitreihen.py) und die Fehlurteilsrate ausgewiesen.
    faelle = [
        ("A  kein Unterschied, kleine Streuung -> gleichwertig",
         dict(delta_ffn=0.0, streuung=0.005), "T1", "die Struktur haelt", None, 20, 0.8),
        ("B  Butterfly deutlich besser",
         dict(delta_ffn=0.10, streuung=0.01), "T1+", "BESSER als der dichte", None, 5, 1.0),
        ("C  Butterfly deutlich schlechter",
         dict(delta_ffn=-0.30, streuung=0.02), "T0", "die Struktur kostet",
         "zugleich innerhalb", 5, 1.0),
        ("D  signifikant schlechter, aber innerhalb der Grenze",
         dict(delta_ffn=-0.02, streuung=0.003), "T0", "zugleich innerhalb der angemeldeten Grenze",
         None, 5, 1.0),
        # narrow_ab=0,5: Beim ersten Lauf stand hier der Standard 0,10 -- bei
        # Streuung 0,08 trennte narrow dann nicht sicher, 11 von 20 Ziehungen
        # landeten im Zweig "Aufloesungspruefung durchgefallen" statt
        # "unentschieden". Die Marke (TU) war trotzdem 20/20 richtig -- genau
        # der Fall, fuer den der Zweigtext geprueft wird. Der Fehler lag im
        # Szenario, nicht in analyse().
        ("E  Nulleffekt, grosse Streuung -> unentschieden",
         dict(delta_ffn=0.0, streuung=0.08, narrow_ab=0.5), "TU", "unentschieden",
         None, 20, 0.8),
        ("F  Aufloesungspruefung faellt (narrow = dense)",
         dict(delta_ffn=0.0, streuung=0.01, narrow_ab=0.0), "TU",
         "Aufloesungspruefung durchgefallen", None, 20, 0.8),
        ("G  Bodenpruefung faellt (dense auf Unigramm-Niveau)",
         dict(delta_ffn=0.0, streuung=0.01, niveau=7.05), "TU",
         "Bodenpruefung durchgefallen", None, 5, 1.0),
        ("H  Aufloesung in GEGENRICHTUNG (narrow besser), sonst gleichwertig",
         dict(delta_ffn=0.0, streuung=0.005, narrow_ab=-0.10), "T1",
         "NARROW BESSER", None, 20, 0.8),
        ("I  eine Zelle fehlt (butterfly_ffn, Seed 2105) -> vorlaeufig",
         dict(delta_ffn=0.10, streuung=0.01, fehlend=(("butterfly_ffn", 2105),)),
         "T1+", "VORLAEUFIG", None, 5, 1.0),
    ]
    print("=" * 92)
    print("VORTEST T-002 -- jeder Ausgang wird einzeln erzwungen, je ueber "
          "mehrere Rauschziehungen")
    print("=" * 92)
    alle_ok = True
    for i, (name, kw, marke, muss, verboten, ziehungen, anteil) in enumerate(faelle):
        marken, treffer = [], 0
        for zug in range(ziehungen):
            rec = _kunst(seed=100 * i + zug, **kw)
            with contextlib.redirect_stdout(io.StringIO()):
                out = analyse(rec)
            urteil = str(out["verdict"])
            text = urteil + " | " + str(out["resolution_check"]["text"])
            m = urteil.split(" ")[0]
            marken.append(m)
            if m == marke and muss in text and (verboten is None or verboten not in text):
                treffer += 1
        quote = treffer / ziehungen
        ok = quote >= anteil - 1e-9
        alle_ok = alle_ok and ok
        print(f"  {'OK    ' if ok else 'FEHLER'} {name:<68} {treffer:>2}/{ziehungen} "
              f"(gefordert {anteil:.0%})  Urteile {dict(Counter(marken))}")

    # Der Zusatzarm darf das Urteil NICHT veraendern -- er steht ausserhalb der
    # Familie. Geprueft auf denselben Daten mit und ohne ihn.
    gleich = 0
    for zug in range(5):
        ohne = _kunst(seed=900 + zug, delta_ffn=-0.02, streuung=0.003)
        mit = _kunst(seed=900 + zug, delta_ffn=-0.02, streuung=0.003, zusatz=0.05)
        with contextlib.redirect_stdout(io.StringIO()):
            a, b = analyse(ohne), analyse(mit)
        if a["verdict"] == b["verdict"] and "beschreibend" in b and "beschreibend" not in a:
            gleich += 1
    ok = gleich == 5
    alle_ok = alle_ok and ok
    print(f"  {'OK    ' if ok else 'FEHLER'} {'Z  Zusatzarm aendert das Urteil nicht, erscheint nur beschreibend':<68} "
          f"{gleich:>2}/5 (gefordert 100%)")
    print("=" * 92)
    print("VORTEST BESTANDEN" if alle_ok else "VORTEST FEHLGESCHLAGEN")
    print("=" * 92)
    return alle_ok


def test_gitterverlaengerung() -> bool:
    """Prueft Regel 1(b) in BEIDE Richtungen mit einer kuenstlichen
    Verlustfunktion (Minimum in log(lr) genau bei `opt`), ohne Training.

    Reichweite: Startgitter 3e-4 .. 2,4e-3, ERWEITERUNGEN = 3 mit Faktor 2,
    also 3,75e-05 bis 0,0192. Ein Optimum ausserhalb wird als Rand gemeldet,
    und ein Optimum GENAU auf dem letzten erreichbaren Punkt auch -- ohne eine
    Messung dahinter laesst sich "Gipfel" nicht von "faellt noch" trennen
    (dieselbe Lehre wie in zeitreihen.test_gitterverlaengerung).

    Geprueft wird ueber die echten Dateifunktionen (Zellen werden in einen
    temporaeren Ordner geschrieben), also derselbe Pfad wie auf dem Cluster.
    """
    import tempfile

    global ZELLEN_WURZEL
    faelle = [
        ("Optimum im Startgitter (1,2e-3)", 1.2e-3, "innen", 1.2e-3),
        ("Optimum eine Verlaengerung OBEN (4,8e-3)", 4.8e-3, "innen", 4.8e-3),
        ("Optimum eine Verlaengerung UNTEN (1,5e-4)", 1.5e-4, "innen", 1.5e-4),
        ("Optimum zwei Verlaengerungen UNTEN (7,5e-5)", 7.5e-5, "innen", 7.5e-5),
        ("Optimum ausserhalb der Reichweite oben (10)", 10.0, "oben", 0.0192),
        ("Optimum ausserhalb der Reichweite unten (1e-9)", 1e-9, "unten", 3.75e-5),
    ]
    print("=" * 92)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen, volles Budget")
    print("=" * 92)
    alle_ok = True
    echt = ZELLEN_WURZEL
    for name, opt, erwarteter_rand, erwartete_rate in faelle:
        with tempfile.TemporaryDirectory(prefix="t002_gitter_") as tmp:
            ZELLEN_WURZEL = Path(tmp)
            try:
                def gefaelscht(arm, seed, schritte, lr, lr_quelle, sichern):
                    return {"arm": arm, "seed": seed, "schritte": schritte,
                            "batch": BATCH, "lr": float(lr),
                            "val_nats": 3.0 + math.log10(lr / opt) ** 2}
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    karte = lernratensuche(schritte=SCHRITTE, arme=("dense",),
                                           trainer=gefaelscht)
                d = json.loads(lr_map_pfad().read_text(encoding="utf-8"))
                n_zellen = len(list((zellen_ordner() / "lr").glob("*.json")))
            finally:
                ZELLEN_WURZEL = echt
        gewaehlt = karte["dense"]
        rand = d["vermerk"]["dense"]
        ok = (abs(math.log10(gewaehlt / erwartete_rate)) < 1e-6
              and rand == erwarteter_rand and d["schritte"] == SCHRITTE)
        alle_ok = alle_ok and ok
        print(f"  {name:<50} -> lr {gewaehlt:<9.4g} Rand {rand:<6} "
              f"{n_zellen} Zellen  "
              + ("OK" if ok else f"FEHLER (erwartet lr {erwartete_rate:g}, Rand {erwarteter_rand})"))
    print("=" * 92)
    print("TEST BESTANDEN" if alle_ok else "TEST FEHLGESCHLAGEN")
    print("=" * 92)
    return alle_ok


def test_wiederaufnahme() -> bool:
    """Prueft das Zusammenfuehren gegen genau die Fehler, die es beheben soll.

    Aufbau in einem temporaeren Ordner:
      * Sammeldatei T-002 (alt) mit (dense, 2101), (butterfly_ffn, 2101)
      * Zellendatei (dense, 2101) mit GLEICHEM Wert        -> Duplikat
      * Zellendatei (butterfly_ffn, 2101) mit ANDEREM Wert -> Widerspruch,
        die aeltere Messung muss gelten
      * Zellendateien (dense, 2102), (butterfly_ffn, 2102)
      * Zellendatei (narrow, 2102) mit falscher Lernrate  -> ausgeschlossen
      * NEUESTE Sammeldatei T-002 mit einer Zeile bei 60 Schritten (der alte
        Probe-Fehler) -> darf die aelteren nicht verdraengen
      * Probe-Datei T-002-probe                           -> gar nicht gelesen
    Erwartet: genau 4 Zellen, 1 Duplikat, 1 Widerspruch, 1 falsche Rate,
    1 falsche Schrittzahl.
    """
    import tempfile

    global ZELLEN_WURZEL
    echt_zellen, echt_ergebnisse = ZELLEN_WURZEL, _rs.RESULTS_DIR
    print("=" * 92)
    print("TEST WIEDERAUFNAHME -- alle Dateien lesen, nach (Arm, Seed) zusammenfuehren")
    print("=" * 92)

    def zeile(arm, seed, wert, schritte=SCHRITTE, lr=1e-3):
        return {"arm": arm, "seed": seed, "schritte": schritte, "batch": BATCH,
                "lr": lr, "val_nats": wert}

    with tempfile.TemporaryDirectory(prefix="t002_wieder_") as tmp:
        ZELLEN_WURZEL = Path(tmp) / "zellen"
        _rs.RESULTS_DIR = Path(tmp) / "experiments"
        try:
            _rs.RESULTS_DIR.mkdir(parents=True)

            def sammeldatei(eid, stempel, created, zeilen):
                (_rs.RESULTS_DIR / f"{eid}__{stempel}.json").write_text(json.dumps(
                    {"experiment_id": eid, "created": created, "records": zeilen}),
                    encoding="utf-8")

            def zellendatei(pfad, created, r):
                pfad.parent.mkdir(parents=True, exist_ok=True)
                pfad.write_text(json.dumps({"experiment_id": EXPERIMENT,
                                            "created": created, "record": r}),
                                encoding="utf-8")

            sammeldatei(EXPERIMENT, "20261001_100000", "2026-10-01T10:00:00",
                        [zeile("dense", 2101, 4.10), zeile("butterfly_ffn", 2101, 4.20)])
            zellendatei(messzelle_pfad(EXPERIMENT, "dense", 2101, SCHRITTE),
                        "2026-10-01T11:00:00", zeile("dense", 2101, 4.10))
            zellendatei(messzelle_pfad(EXPERIMENT, "butterfly_ffn", 2101, SCHRITTE),
                        "2026-10-01T11:00:01", zeile("butterfly_ffn", 2101, 4.05))
            zellendatei(messzelle_pfad(EXPERIMENT, "dense", 2102, SCHRITTE),
                        "2026-10-01T12:00:00", zeile("dense", 2102, 4.11))
            zellendatei(messzelle_pfad(EXPERIMENT, "butterfly_ffn", 2102, SCHRITTE),
                        "2026-10-01T12:00:01", zeile("butterfly_ffn", 2102, 4.21))
            zellendatei(messzelle_pfad(EXPERIMENT, "narrow", 2102, SCHRITTE),
                        "2026-10-01T12:00:02", zeile("narrow", 2102, 4.30, lr=3e-4))
            sammeldatei(EXPERIMENT, "20261002_090000", "2026-10-02T09:00:00",
                        [zeile("dense", 2101, 5.5, schritte=60)])
            sammeldatei(PROBE_ID, "20261002_090100", "2026-10-02T09:01:00",
                        [zeile("narrow", 2101, 5.9)])

            lr_map = {"dense": 1e-3, "butterfly_ffn": 1e-3, "narrow": 1e-3}
            records, bericht = sammle_messungen(EXPERIMENT, SCHRITTE, lr_map)
        finally:
            ZELLEN_WURZEL, _rs.RESULTS_DIR = echt_zellen, echt_ergebnisse

    schluessel = sorted((r["arm"], r["seed"]) for r in records)
    bf2101 = next(r["val_nats"] for r in records
                  if r["arm"] == "butterfly_ffn" and r["seed"] == 2101)
    pruefungen = [
        ("genau 4 Zellen, die richtigen",
         schluessel == [("butterfly_ffn", 2101), ("butterfly_ffn", 2102),
                        ("dense", 2101), ("dense", 2102)]),
        ("1 identisches Duplikat erkannt", bericht["duplikate_identisch"] == 1),
        ("1 Widerspruch erkannt", len(bericht["widersprueche"]) == 1),
        ("bei Widerspruch gilt die AELTERE Messung (4,20, nicht die bessere 4,05)",
         abs(bf2101 - 4.20) < 1e-12),
        ("falsche Lernrate ausgeschlossen",
         bericht["ausgeschlossen"].get("andere Lernrate als in lr_map.json") == 1),
        ("neueste Datei mit 60 Schritten verdraengt nichts",
         bericht["ausgeschlossen"].get("andere Schrittzahl") == 1),
        ("Probe-ID wird nicht gelesen",
         not any("probe" in q for q in bericht["quellen"])),
    ]
    alle_ok = True
    for text, ok in pruefungen:
        alle_ok = alle_ok and ok
        print(f"  {'OK    ' if ok else 'FEHLER'} {text}")
    print("=" * 92)
    print("TEST BESTANDEN" if alle_ok else "TEST FEHLGESCHLAGEN")
    print("=" * 92)
    return alle_ok


def rauchtest(schritte: int = 3, threads: int = 2, seeds: int = 3) -> bool:
    """Mini-Durchlauf der ECHTEN Kette auf der CPU: Lernratensuche (mit
    Verlaengerung), Messzellen, Wiederaufnahme, Zusammenfuehren, analyse(),
    Textprobe. Batch 8, 2 Auswertungsstapel, `schritte` Schritte.

    DIE ZAHLEN SIND BEDEUTUNGSLOS. Geprueft wird nur, dass alles durchlaeuft
    und die Dateien stimmen. Alle Ausgaben landen in einem temporaeren Ordner,
    nichts in results/. Ohne train.npy (z. B. in der Ueberarbeitungskopie)
    wird val.npy AUCH als Trainingsmenge benutzt -- ausdruecklich nur fuer
    diesen Rauchtest; mit QIRE_KORPUS auf dem Archivordner wird train.npy per
    memmap gelesen, nicht kopiert.
    """
    import tempfile

    global ZELLEN_WURZEL, MODELLE, BATCH, EVAL_STAPEL, EVAL_ALLE, lade_token
    alt = (ZELLEN_WURZEL, MODELLE, BATCH, EVAL_STAPEL, EVAL_ALLE, lade_token,
           _rs.RESULTS_DIR)
    ordner = Path(tempfile.mkdtemp(prefix="t002_rauchtest_"))
    t0 = time.perf_counter()
    print("=" * 92)
    print(f"RAUCHTEST -- {schritte} Schritte, Batch 8, CPU, Ausgaben in {ordner}")
    echtes_laden = lade_token
    if not (KORPUS / "train.npy").exists():
        print(f"  RAUCHTEST: {KORPUS / 'train.npy'} fehlt -- val.npy wird AUCH "
              f"als Trainingsmenge benutzt. Nur fuer diesen Test zulaessig.")
        lade_token = lambda teil: echtes_laden("val")   # noqa: E731
    else:
        print(f"  Korpus: {KORPUS} (train.npy per memmap)")
    print("=" * 92)
    ok = True
    try:
        ZELLEN_WURZEL = ordner / "zellen"
        MODELLE = ordner / "modelle"
        _rs.RESULTS_DIR = ordner / "experiments"
        BATCH, EVAL_STAPEL, EVAL_ALLE = 8, 2, 2
        arme = ALLE_ARME
        karte = lernratensuche(schritte=schritte, arme=arme, threads=threads,
                               nicht_vorregistriert=True)
        n_lr = len(list((zellen_ordner() / "lr").glob("*.json")))
        print(f"  -> {n_lr} LR-Zellen, Karte {karte}")
        rec = trainieren(schritte=schritte, seeds=SEEDS[:seeds], arme=arme,
                         threads=threads, nicht_vorregistriert=True)
        ok &= all(math.isclose(r["lr"], karte[r["arm"]]) for r in rec.records)
        # Wiederaufnahme: ein zweiter Aufruf darf nichts neu rechnen.
        vorher = len(list((zellen_ordner() / "messung").glob("*.json")))
        rec2 = trainieren(schritte=schritte, seeds=SEEDS[:seeds], arme=arme,
                          threads=threads, nicht_vorregistriert=True)
        nachher = len(list((zellen_ordner() / "messung").glob("*.json")))
        ok &= (vorher == nachher == len(arme) * seeds
               and len(rec2.records) == len(arme) * seeds)
        # Einzelzelle wie auf dem Cluster: liegt vor -> uebersprungen.
        rechne_zelle("dense", SEEDS[0], schritte, karte["dense"], "lr_map")
        ok &= len(list((zellen_ordner() / "messung").glob("*.json"))) == nachher
        rec3, out = zusammenfuehren(schritte=schritte)
        ok &= len(rec3.records) == len(arme) * seeds and "verdict" in out
        text = erzeuge("dense", "Die Quantenmechanik ist", laenge=8)
        print(f"  Textprobe (bedeutungslos): {text!r}")
    finally:
        (ZELLEN_WURZEL, MODELLE, BATCH, EVAL_STAPEL, EVAL_ALLE, lade_token,
         _rs.RESULTS_DIR) = alt
    print("=" * 92)
    print(f"RAUCHTEST {'BESTANDEN' if ok else 'FEHLGESCHLAGEN'} "
          f"({time.perf_counter() - t0:.0f} s, Ausgaben in {ordner})")
    print("=" * 92)
    return bool(ok)


# --------------------------------------------------------------------------
# Rechenbedarf -- SCHAETZUNG fuer die Anfrage nach Rechenzeit
# --------------------------------------------------------------------------

def rechenbedarf(arme: Sequence[str] = ARME, schritte: int = SCHRITTE,
                 s_je_schritt: float = S_JE_SCHRITT_4090,
                 gpus: Sequence[int] = (1, 4, 8, 16, 32)) -> Dict[str, float]:
    """GPU-Stunden und Wanduhrzeit, aus den Konstanten dieser Datei gerechnet.

    SCHAETZUNG, keine Messung:
      * s_je_schritt stammt aus EINER Messung von butterfly_ffn auf einer
        RTX 4090 (11. August). Fuer ALLE Arme angesetzt -- dense und narrow
        sind vermutlich schneller, butterfly_qkv ist ungemessen. Die Zahlen
        sind damit eher eine Obergrenze.
      * Auswertung: je EVAL_ALLE Schritte EVAL_STAPEL Vorwaertslaeufe, ANNAHME
        ein Vorwaertslauf ~ 1/3 eines Trainingsschritts.
      * Andere GPUs (A100, H100): nicht gemessen. Laut Kommentar oben ist der
        Butterfly-Arm startaufwandbegrenzt -- eine schnellere Karte hilft
        vermutlich wenig, mehr parallele Karten helfen linear.
    """
    n_eval = math.ceil(schritte / EVAL_ALLE)
    min_zelle = (schritte + n_eval * EVAL_STAPEL / 3) * s_je_schritt / 60
    ns = len(LR_SUCHE_SEEDS)
    lr_start = len(arme) * len(LR_GITTER) * ns
    lr_verl = len(arme) * ns                  # hoechstens eine Zelle je Arm und Runde
    mess = len(arme) * len(SEEDS)
    zellen_min = lr_start + mess
    zellen_max = lr_start + ERWEITERUNGEN * lr_verl + mess
    print("=" * 92)
    print(f"RECHENBEDARF T-002 -- SCHAETZUNG ({len(arme)} Arme, {schritte:,} Schritte, "
          f"{s_je_schritt} s/Schritt aus butterfly_ffn/RTX 4090)")
    print("=" * 92)
    print(f"  Minuten je Zelle (inkl. ~{n_eval * EVAL_STAPEL / 3 / schritte:.1%} "
          f"Auswertung): {min_zelle:.0f}")
    print(f'  {"Phase":<44} {"Zellen":>7} {"GPU-h":>8}')
    zeilen = [
        (f"LR-Suche Startgitter ({len(LR_GITTER)} Raten x {ns} Seed)", lr_start),
        (f"LR-Verlaengerungen (0 bis {ERWEITERUNGEN} Runden)", ERWEITERUNGEN * lr_verl),
        (f"Messlauf ({len(SEEDS)} Seeds)", mess),
    ]
    for text, n in zeilen:
        print(f"  {text:<44} {n:>7} {n * min_zelle / 60:>8.1f}"
              + ("  (hoechstens)" if "Verlaengerungen" in text else ""))
    print(f"  {'SUMME ohne / mit allen Verlaengerungen':<44} "
          f"{zellen_min:>3}-{zellen_max:<3} {zellen_min * min_zelle / 60:>5.1f}-"
          f"{zellen_max * min_zelle / 60:.1f}")
    print()
    print(f'  {"GPUs":>5} {"Wanduhr ohne Verl.":>20} {"mit 3 Verl.":>14}   (Wellen x Minuten je Zelle)')
    for g in gpus:
        wellen_min = math.ceil(lr_start / g) + math.ceil(mess / g)
        wellen_max = wellen_min + ERWEITERUNGEN * math.ceil(lr_verl / g)
        print(f"  {g:>5} {wellen_min * min_zelle / 60:>17.1f} h {wellen_max * min_zelle / 60:>11.1f} h")
    print("  Die LR-Runden laufen NACHEINANDER (jede haengt von der vorigen ab);")
    print("  der Messlauf beginnt erst, wenn die Lernratenkarte steht.")
    print("=" * 92)
    return {"minuten_je_zelle": min_zelle, "zellen_min": zellen_min,
            "zellen_max": zellen_max, "gpu_h_min": zellen_min * min_zelle / 60,
            "gpu_h_max": zellen_max * min_zelle / 60}


def status(arme: Sequence[str], schritte: int) -> None:
    print("=" * 90)
    print(f"STATUS T-002 ({schritte:,} Schritte), Zellen in {ZELLEN_WURZEL}")
    for arm in arme:
        st = lr_stand(arm, schritte)
        print(f"  LR  {arm:>24}: "
              + (f"fertig, lr {st['bester']:g} ({st['vermerk']})" if st["fertig"]
                 else f"offen, {len(st['fehlend'])} Zelle(n) in Runde {st['runde']}"))
    try:
        offen = mess_plan(arme, schritte)
        print(f"  Messung: {len(arme) * len(SEEDS) - len(offen)} von "
              f"{len(arme) * len(SEEDS)} Zellen fertig")
    except SystemExit as exc:
        print(f"  Messung: noch nicht moeglich ({exc})")
    print("=" * 90)


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
    # weights_only=True: die Datei enthaelt nur Tensoren und Grunddaten; so
    # fuehrt das Laden einer fremden .pt-Datei (z. B. vom Cluster) keinen
    # beliebigen Python-Code aus.
    stand = torch.load(pfad, map_location="cpu", weights_only=True)
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


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m quantum_sim.sprachmodell",
                                description="T-002, das Sprachmodell")
    sub = p.add_subparsers(dest="befehl")

    def zusatz(sp):
        sp.add_argument("--mit-zusatzarm", action="store_true",
                        help="butterfly_ffn_zentriert mitrechnen (nicht "
                             "vorregistriert, nur beschreibend)")

    def schritte_opt(sp):
        sp.add_argument("--schritte", type=int, default=SCHRITTE)

    def vorreg(sp):
        sp.add_argument("--nicht-vorregistriert", action="store_true",
                        help="andere Schrittzahl als SCHRITTE zulassen")

    sp = sub.add_parser("durchsatz"); zusatz(sp)
    sub.add_parser("schritte", help="vorregistrierte Schrittzahl ausgeben")
    sp = sub.add_parser("lernratensuche",
                        help="ohne --arm: ganze Suche; mit --arm/--lr: EINE Zelle")
    sp.add_argument("--arm", choices=ALLE_ARME)
    sp.add_argument("--lr", type=float)
    sp.add_argument("--seed", type=int, default=LR_SUCHE_SEEDS[0])
    sp.add_argument("--threads", type=int, default=6)
    schritte_opt(sp); zusatz(sp); vorreg(sp)
    sp = sub.add_parser("zelle", help="EINE Messzelle (Arm, Seed)")
    sp.add_argument("--arm", choices=ALLE_ARME, required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--threads", type=int, default=6)
    schritte_opt(sp); vorreg(sp)
    sp = sub.add_parser("trainieren")
    sp.add_argument("schritte_pos", nargs="?", type=int,
                    help="(veraltet) Schrittzahl; Standard ist SCHRITTE")
    sp.add_argument("--threads", type=int, default=6)
    schritte_opt(sp); zusatz(sp); vorreg(sp)
    sp = sub.add_parser("probe"); zusatz(sp)
    sp.add_argument("--threads", type=int, default=6)
    sp = sub.add_parser("plan", help="offene Zellen als Zeilen fuer ein Job-Array")
    sp.add_argument("phase", choices=("lr", "messung"))
    schritte_opt(sp); zusatz(sp)
    sp = sub.add_parser("status"); schritte_opt(sp); zusatz(sp)
    sp = sub.add_parser("zusammenfuehren"); schritte_opt(sp)
    sp = sub.add_parser("rechenbedarf"); zusatz(sp); schritte_opt(sp)
    sp.add_argument("--s-je-schritt", type=float, default=S_JE_SCHRITT_4090)
    sub.add_parser("vortest")
    sub.add_parser("test_gitter")
    sub.add_parser("test_wiederaufnahme")
    sub.add_parser("selbsttest", help="vortest + test_gitter + test_wiederaufnahme")
    sp = sub.add_parser("rauchtest")
    sp.add_argument("--schritte", type=int, default=3)
    sp.add_argument("--threads", type=int, default=2)
    sp = sub.add_parser("erzeuge")
    sp.add_argument("arm", nargs="?", default="dense")
    sp.add_argument("anstoss", nargs="?", default="Die Quantenmechanik ist")

    a = p.parse_args(argv)
    befehl = a.befehl or "durchsatz"
    arme = ALLE_ARME if getattr(a, "mit_zusatzarm", False) else ARME

    if befehl == "durchsatz":
        durchsatz_alle_arme(arme=arme)
    elif befehl == "schritte":
        print(SCHRITTE)
    elif befehl == "lernratensuche":
        if a.arm is not None or a.lr is not None:
            if a.arm is None or a.lr is None:
                p.error("--arm und --lr nur zusammen")
            _pruefe_schritte(a.schritte, a.nicht_vorregistriert)
            rechne_lr_zelle(a.arm, a.lr, a.seed, a.schritte, a.threads)
        else:
            lernratensuche(schritte=a.schritte, arme=arme, threads=a.threads,
                           nicht_vorregistriert=a.nicht_vorregistriert)
    elif befehl == "zelle":
        _pruefe_schritte(a.schritte, a.nicht_vorregistriert)
        lr_map, quelle = lies_lr_map(a.schritte, [a.arm])
        rechne_zelle(a.arm, a.seed, a.schritte, lr_map[a.arm], quelle,
                     threads=a.threads)
    elif befehl == "trainieren":
        schritte = a.schritte_pos if a.schritte_pos is not None else a.schritte
        r = trainieren(schritte=schritte, arme=arme, threads=a.threads,
                       nicht_vorregistriert=a.nicht_vorregistriert)
        try:
            aus = analyse(r)
        except Exception as exc:                  # noqa: BLE001
            print(f"  Auswertung fehlgeschlagen: {exc!r}")
            aus = {"error": repr(exc)}
        r.save(analysis=aus)
    elif befehl == "probe":
        # Kurzer Testlauf unter EIGENER ID: laeuft alles durch, bevor Stunden
        # investiert werden -- und verdraengt den Hauptlauf nicht mehr.
        r = trainieren(schritte=60, seeds=SEEDS[:2], arme=arme, threads=a.threads,
                       experiment_id=PROBE_ID, lr_rueckfall=True,
                       nicht_vorregistriert=True)
        try:
            aus = analyse(r)
        except Exception as exc:                  # noqa: BLE001
            print(f"  Auswertung fehlgeschlagen: {exc!r}")
            aus = {"error": repr(exc)}
        r.save(analysis=aus)
    elif befehl == "plan":
        # Nur die Zellen auf stdout -- eine je Zeile, direkt als Argumente
        # fuer `python -m quantum_sim.sprachmodell`. Alles andere auf stderr.
        if a.phase == "lr":
            offen = offene_lr_zellen(arme, a.schritte)
            for arm, lr, seed in offen:
                print(f"lernratensuche --arm {arm} --lr {_lr_text(lr)} "
                      f"--seed {seed} --schritte {a.schritte}")
            if not offen:
                sys.stdout, echt = sys.stderr, sys.stdout
                try:
                    schreibe_lr_map(arme, a.schritte)
                finally:
                    sys.stdout = echt
        else:
            try:
                offen_m = mess_plan(arme, a.schritte)
            except SystemExit as exc:
                print(str(exc), file=sys.stderr)
                return 2
            for arm, seed in offen_m:
                print(f"zelle --arm {arm} --seed {seed} --schritte {a.schritte}")
    elif befehl == "status":
        status(arme, a.schritte)
    elif befehl == "zusammenfuehren":
        zusammenfuehren(schritte=a.schritte)
    elif befehl == "rechenbedarf":
        rechenbedarf(arme=arme, schritte=a.schritte, s_je_schritt=a.s_je_schritt)
    elif befehl == "vortest":
        return 0 if vortest() else 1
    elif befehl == "test_gitter":
        return 0 if test_gitterverlaengerung() else 1
    elif befehl == "test_wiederaufnahme":
        return 0 if test_wiederaufnahme() else 1
    elif befehl == "selbsttest":
        ergebnisse = [vortest(), test_gitterverlaengerung(), test_wiederaufnahme()]
        print("SELBSTTEST " + ("BESTANDEN" if all(ergebnisse) else "FEHLGESCHLAGEN"))
        return 0 if all(ergebnisse) else 1
    elif befehl == "rauchtest":
        return 0 if rauchtest(schritte=a.schritte, threads=a.threads) else 1
    elif befehl == "erzeuge":
        print(erzeuge(arm=a.arm, anstoss=a.anstoss))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
