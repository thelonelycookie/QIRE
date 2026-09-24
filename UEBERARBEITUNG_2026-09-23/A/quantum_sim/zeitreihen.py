from __future__ import annotations

"""RESEARCH ID: Q-019 -- Zeitreihen, die zweite KI-Art.

DIE IDEE
Alle bisherigen Anwendungen des Projekts fragen: *Ist die Struktur genauso gut
wie eine dichte Schicht, nur billiger?* Die Antwort war bestenfalls
"gleichwertig". Zeitreihen sind die einzige Anwendung, bei der ein
**inhaltlicher** Vorteil zu erwarten ist:

    Die Butterfly-Struktur IST die schnelle Fouriertransformation. Fuer
    periodische Signale ist das die natuerliche Basis, und dort sollte die
    Architektur nicht nur gleich gut, sondern besser sein als eine dichte
    Schicht.

Das ist eine staerkere Behauptung als alles bisher Gepruefte -- und deshalb
auch leichter zu widerlegen. Genau darum wird sie gemessen.

WARNUNG, VOR DEM LAUF EINGETRAGEN: Der Satz oben ist in dieser Form FALSCH,
und das ist beim Bauen dieses Moduls aufgefallen. Diese Schicht hat GETEILTE
Winkel -- einer je Lage und Bit, 40 insgesamt. Die schnelle
Fouriertransformation braucht je Paar einen EIGENEN Drehfaktor, das waere die
ungeteilte Variante mit (d/2)*n*L = 5.120. Gemeinsam ist beiden das
VERBINDUNGSMUSTER und damit die Kostenordnung O(d log d) -- nicht die
Transformation. Der Vorlauf (zwei Seeds, ausgetunte Lernrate) bestaetigt es:
`butterfly` erreicht 0,686, `fft_fest` 0,804, `dense` 0,913.

Der Satz bleibt stehen, weil er der Anlass des Laufs war. Die Hypothese wird
trotzdem gemessen -- sie ist jetzt nur besser verstanden, und der wahrschein-
liche Ausgang ist im Zweig "HU -- geteiltes Ergebnis" vorweggenommen.

DIE AUFGABE
Synthetische Signale der Laenge d = 256 = 2^8: Summe weniger Sinusanteile mit
zufaelliger Frequenz, Phase und Amplitude, dazu Rauschen. Die Klasse ist das
Frequenzband, in dem der **staerkste** Anteil liegt (8 Baender, 8 Klassen).

Im Frequenzraum ist das trivial, im Zeitraum schwer. Ein Modell mit der
richtigen Basis muss fast nichts lernen; eines ohne sie muss die
Fouriertransformation aus den Daten rekonstruieren.

DIE ARME
  butterfly   orthogonale Butterfly (ein Winkel je Lage und Bit, also n*L = 40)
              + quadratische Auslese
  dense       freie dichte 256x256-Drehung + dieselbe Auslese      65.536
  fft_fest    ECHTE Fouriertransformation, untrainiert, + dieselbe Auslese   0
  narrow      dichte Engstelle 256 -> 64 -> 256 + dieselbe Auslese  32.768

`fft_fest` ist der wichtigste der drei Kontrollarme: Zieht die
Butterfly-Struktur ihren Vorteil daraus, dass sie *fourieraehnlich* ist, dann
muss die echte Fouriertransformation mindestens so gut sein. Ist sie es nicht,
liegt der Vorteil woanders -- und die Fourier-Erzaehlung waere falsch.

DIE KONTROLLE, DIE DEN BEFUND ERST TRAEGT
Dieselben Signale, aber mit einer **festen zufaelligen Vertauschung der
Zeitachse**. Alle Werte bleiben erhalten, die Klasse bleibt dieselbe, nur die
Periodizitaet ist zerstoert. Ist der Vorteil der Butterfly-Struktur strukturell,
MUSS er dort verschwinden. Bleibt er, kommt er nicht aus der
Fourier-Verwandtschaft, und die Deutung ist hinfaellig.

Das ist exakt die Kontrolle, mit der B-004 Monarchs Vorteil eingeordnet hat:
+0,0115 auf Bildern, verschwunden bei permutierten Pixeln (p=0,23).

VORREGISTRIERUNG
  Balancierte Genauigkeit, Lernrate je Arm UND Bedingung auf eigenen
  Tuning-Seeds, 15 Seeds, beide Datenbedingungen.

  PRIMAER  Die WECHSELWIRKUNG. Der Abstand (butterfly - dense) ist auf
           periodischen Signalen groesser als auf permutierten. Wilcoxon,
           einseitig, ueber die seedweise gepaarten Differenzen der
           Differenzen.
  P2       Auf periodischen Signalen ist (butterfly - dense) positiv.
           Wilcoxon, einseitig.
  Holm ueber genau diese beiden -- und ueber nichts sonst (Regel 4).

  Der Primaertest ist bewusst die Wechselwirkung und nicht der einfache
  Vergleich. "Butterfly schlaegt dicht" allein waere auch mit einem
  Rechenartefakt erklaerbar; "Butterfly schlaegt dicht NUR DORT, wo die
  Struktur zu den Daten passt", ist es nicht.

  H1   beide halten -> die Struktur bringt einen inhaltlichen Vorteil, und
       zwar nachweislich aus der Passung zur Datenstruktur. Das waere der
       erste Befund des Projekts, der ueber "gleichwertig, aber billiger"
       hinausgeht.
  H1+  zusaetzlich ist `butterfly` nicht schlechter als `fft_fest` (TOST) ->
       die gelernte Naeherung erreicht die echte Transformation.
  H0   keine Wechselwirkung (TOST auf die Wechselwirkung) -> der Vorteil ist
       nicht strukturell, die Fourier-Erzaehlung faellt.
       ERREICHBARKEIT, vor dem Lauf ausgerechnet: TOST gegen +-0,005 besteht
       bei 15 Seeds nur, wenn die seedweise Streuung der Wechselwirkung unter
       sd = 0,005*sqrt(15)/1,7613 = 0,0110 liegt. Ist sie groesser, KANN H0
       nicht feuern, und der Lauf kann nur H1, H0- oder HU liefern. Das ist
       kein Mangel, aber es muss vorher dastehen: sonst wird ein unerreichbarer
       Ausgang hinterher als "nicht eingetreten" gelesen. `macht()` misst diese
       Streuung und sagt es an.
  H0-  die Wechselwirkung ist signifikant NEGATIV, der Abstand ist auf
       permutierten Signalen GROESSER. Dieser Zweig ist vorhanden, weil sein
       Fehlen in Q-011, Q-014 und Q-015 dreimal in Folge zu einer Aussage
       gefuehrt hat, die den Daten widersprach. Er ist kein Nullbefund,
       sondern das Gegenteil der Hypothese und gehoert als eigenstaendiges
       Ergebnis berichtet.
  HU   weder noch -> erreichte Schranke berichten.

  AUFLOESUNGSPRUEFUNG (Regel 6): `dense` muss `narrow` auf periodischen
  Signalen schlagen. Das ist ein grober Kapazitaetsunterschied (Faktor 2 in
  den Parametern, eine echte Engstelle auf 64 Dimensionen). Sieht der Aufbau
  ihn nicht, kann er den feineren Strukturunterschied erst recht nicht sehen,
  und kein Ergebnis dieses Laufs ist interpretierbar.

  BODENPRUEFUNG: Mindestens ein Arm muss deutlich ueber Rateniveau liegen
  (0,125 bei 8 Klassen; Schwelle BODEN). Sonst misst die Zelle
  Trainingsversagen und nicht Architektur.

  SAETTIGUNGSPRUEFUNG (vor dem Lauf, `saettigung()`): KEIN Arm darf die
  Aufgabe fast perfekt loesen. Bei `fft_fest` ist das eine reale Gefahr --
  die Auslese liest Bits des Frequenzindex, und bei frequenzsortierter Basis
  liegen die drei hoechstwertigen Bits genau auf der Bandgrenze. Loest ein Arm
  die Aufgabe mit ueber SAETTIGUNG, trennt nichts mehr, und die Schwierigkeit
  wird VOR dem Lauf erhoeht (mehr Anteile, mehr Rauschen, kleinerer
  Amplitudenabstand). Jede solche Aenderung gehoert ins Tagebuch, bevor Zahlen
  entstehen (Regel 7).

WAS AN DIESEM SKRIPT AUS DEN FEHLERN VOM 27. AUGUST FOLGT
  * Die Lernratensuche laeuft mit dem VOLLEN Budget des Messlaufs (Regel 1c).
    Der zweite Q-011-Anlauf ist an halbem Suchbudget gescheitert: die Suche
    waehlt dann systematisch zu hoch, und in 5 von 5 Tiefen gewann der Arm mit
    der kleineren Rate, um +-0,013.
  * Das Gitter wird in BEIDE Richtungen verlaengert (Regel 1b, ergaenzt am
    27. August spaet abends). Die Verlaengerung in `unitarity_depth.py` geht
    nur nach oben; genau ein Arm am UNTEREN Rand ist es, der Q-015 auf
    fashion_mnist entwertet.
  * Kein `stats.spearmanr`. Hier wird nichts ueber wenige Stuetzstellen
    korreliert, also entfaellt die Fehlerklasse aus Q-011 von selbst.
  * Alle Wilcoxon-Tests sind ausdruecklich EINSEITIG, wie die Vorregistrierung
    es verlangt. scipy liefert zweiseitig, und dieser Unterschied hat Q-011
    schon einmal falsch berichtet.
  * Die Auswertung benutzt `bool(...)` statt `is False` auf NumPy-Wahrheits-
    werten. `np.False_ is False` ist FALSCH, und dadurch wurde die
    Aufloesungspruefung in Q-011 berechnet, gedruckt und beim Urteil
    stillschweigend uebergangen.
  * Rohdaten werden nach JEDER Bedingung gesichert, nicht am Ende (Regel 5).

Run:
    python -m quantum_sim.zeitreihen vortest     # Pflicht vor dem Lauf
    python -m quantum_sim.zeitreihen saettigung  # Schwierigkeit pruefen
    python -m quantum_sim.zeitreihen macht       # Regel 2(b)
    python -m quantum_sim.zeitreihen             # der Lauf
"""

import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qire_ai.equivalence import MARGIN, holm, tost_paired
from qire_ai.result_store import Recorder

# ---------------------------------------------------------------------------
# Aufbau -- alle Zahlen VOR dem Lauf festgelegt (Regel 7)
# ---------------------------------------------------------------------------
QUBITS = 8                       # d = 2**8 = 256
DIM = 2 ** QUBITS
LAGEN = 5                        # Butterfly-Lagen -> n*L = 40 Winkel
KLASSEN = 8                      # acht Frequenzbaender
ENGSTELLE = 64                   # Engstelle des Arms `narrow`

ARME = ("butterfly", "untied", "dense", "fft_fest", "narrow")
# `untied` ist BESCHREIBEND und steht ausserhalb der angemeldeten
# Familie -- siehe Klassenkommentar in `UngeteilteButterfly`.
FAMILIE = ("butterfly", "dense")   # nur diese beiden tragen Holm
BEDINGUNGEN = ("periodisch", "permutiert")
PRIMAERE_BEDINGUNG = "periodisch"

EVAL_SEEDS = tuple(range(1901, 1916))        # 15 Seeds
TUNING_SEEDS = (1951, 1952, 1953)            # drei, wie in Q-011
LR_GITTER = (0.001, 0.003, 0.01, 0.03)
ERWEITERUNGEN = 3                            # je Richtung

N_TRAIN = 6000
N_TEST = 2000
EPOCHS = 10
BATCH = 128

# Schwierigkeit der Aufgabe. Wird von `saettigung()` geprueft und darf VOR dem
# Lauf geaendert werden -- jede Aenderung gehoert vorher ins Tagebuch.
ANTEILE = 3                      # Sinusanteile je Signal
RAUSCHEN = 0.45                  # Standardabweichung des additiven Rauschens
AMPLITUDENABSTAND = 0.15         # Mindestvorsprung des staerksten Anteils

BODEN = 0.125 + 0.10             # Rateniveau plus Sicherheitsabstand
SAETTIGUNG = 0.95                # darueber gilt ein Arm als gesaettigt


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
def signale(seed: int, n: int, bedingung: str,
            anteile: int = ANTEILE, rauschen: float = RAUSCHEN,
            abstand: float = AMPLITUDENABSTAND) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetische Signale und ihr Frequenzband.

    Jedes Signal ist die Summe von `anteile` Sinusanteilen mit zufaelliger
    Frequenz, Phase und Amplitude, dazu weisses Rauschen. Die Klasse ist das
    Band des staerksten Anteils.

    Damit die Klasse ueberhaupt wohldefiniert ist, muss der staerkste Anteil
    einen Mindestvorsprung haben; Signale ohne diesen Vorsprung werden neu
    gezogen. Ohne das gaebe es Beispiele, deren richtige Antwort vom Rauschen
    abhaengt -- eine Obergrenze der Genauigkeit, die nichts mit Architektur zu
    tun hat und den Vergleich verwaessert.

    `bedingung="permutiert"` wendet eine FESTE (seedunabhaengige) Vertauschung
    der Zeitachse an. Fest, weil sonst jeder Seed eine andere Aufgabe bekaeme
    und die Streuung ueber die Seeds zwei Quellen haette.
    """
    if bedingung not in BEDINGUNGEN:
        raise ValueError(f"unbekannte Bedingung: {bedingung}")
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(DIM, dtype=torch.float32) / DIM

    # Frequenzen 1..127, in acht Baender zu je 16 Bins (das letzte hat 15).
    # Nyquist (k = 128) ist AUSGENOMMEN: in der reellen Fourierbasis unten
    # liegt dieser Anteil auf Index 1 und wuerde von der Auslese Band 0
    # zugeschlagen statt Band 7. Das betrifft nur 1 von 128 Frequenzen, ist
    # aber ein Etikettenfehler -- die Aufgabe soll schwer sein, weil sie
    # schwer ist, und nicht, weil sie falsch gebaut ist.
    bins_je_band = (DIM // 2) // KLASSEN
    x = torch.empty(n, DIM)
    y = torch.empty(n, dtype=torch.long)

    gezogen = 0
    while gezogen < n:
        rest = n - gezogen
        block = max(rest * 2, 256)          # mit Reserve ziehen, dann filtern
        k = torch.randint(1, DIM // 2, (block, anteile), generator=g)   # 1..127
        phi = torch.rand(block, anteile, generator=g) * 2 * math.pi
        amp = 0.3 + torch.rand(block, anteile, generator=g) * 0.7

        sortiert, _ = amp.sort(dim=1, descending=True)
        if anteile > 1:
            gueltig = (sortiert[:, 0] - sortiert[:, 1]) >= abstand
        else:
            gueltig = torch.ones(block, dtype=torch.bool)
        k, phi, amp = k[gueltig], phi[gueltig], amp[gueltig]
        if not len(k):
            continue

        wellen = torch.sin(2 * math.pi * k[..., None].float() * t + phi[..., None])
        sig = (amp[..., None] * wellen).sum(dim=1)
        sig = sig + torch.randn(sig.shape, generator=g) * rauschen

        stark = amp.argmax(dim=1)
        k_stark = k[torch.arange(len(k)), stark]
        band = ((k_stark - 1) // bins_je_band).clamp(max=KLASSEN - 1)

        nimm = min(rest, len(sig))
        x[gezogen:gezogen + nimm] = sig[:nimm]
        y[gezogen:gezogen + nimm] = band[:nimm]
        gezogen += nimm

    if bedingung == "permutiert":
        # FESTE Vertauschung -- eigener Generator, unabhaengig vom Daten-Seed
        gp = torch.Generator().manual_seed(20260827)
        x = x[:, torch.randperm(DIM, generator=gp)]
    return x, y


# ---------------------------------------------------------------------------
# Die Auslese -- fuer alle Arme identisch
# ---------------------------------------------------------------------------
def auslese(h: torch.Tensor, n: int = QUBITS) -> torch.Tensor:
    """Quadratische Auslese ueber ueberlappende Zweiteilungen: <Z_q>.

    Warum ueberlappende Zweiteilungen und nicht n disjunkte Gruppen: Bei einer
    normerhaltenden Abbildung summieren sich n gruppenweise Quadratsummen auf
    exakt 1 -- eine der n Ausgabedimensionen traegt dann nur eine Konstante,
    und der Effektivrang faellt von 8 auf 6,99. In Q-015 ist das nachgemessen
    und war der Grund, warum `classical_quadratic` schlechter abschnitt als
    `classical_quad_free` bei identischer Parameterzahl.

    <Z_q> = P(Bit q = 0) - P(Bit q = 1) hat dieses Problem nicht: Jedes Bit
    halbiert den Raum anders, es gibt keine feste Summe.

    Qubit 0 ist das hoechstwertige Bit -- Konvention aus `simulator.py`.

    EINE ASYMMETRIE, DIE BENANNT GEHOERT: `butterfly` und `fft_fest` sind
    normerhaltend, die Quadratsummen addieren sich also zu 1 und <Z_q> liegt in
    [-1, 1]. `dense` und `narrow` sind es nicht -- dort ist der Massstab frei
    und wird mitgelernt. Der lineare Kopf kann jeden festen Massstab aufheben,
    aber die GRADIENTEN sind unterschiedlich gross, und das wirkt wie eine
    andere Lernrate. Genau deshalb wird die Lernrate je Arm gesucht (Regel 1);
    ohne sie waere dieser Vergleich verzerrt, und zwar unbekannt in welche
    Richtung.
    """
    p = h ** 2                                        # Born-Regel
    aus = []
    for q in range(n):
        pr = p.reshape(-1, 2 ** q, 2, 2 ** (n - 1 - q))
        aus.append(pr[:, :, 0, :].sum(dim=(1, 2)) - pr[:, :, 1, :].sum(dim=(1, 2)))
    return torch.stack(aus, dim=1)


# ---------------------------------------------------------------------------
# Die vier Mittelschichten
# ---------------------------------------------------------------------------
class OrthoButterfly(nn.Module):
    """Reelle orthogonale Butterfly mit geteilten Winkeln -- 40 Parameter.

    Ein Winkel je (Lage, Bit). Die 2x2-Drehung wirkt auf ALLE d/2 Amplituden-
    paare, deren Indizes sich in diesem Bit unterscheiden -- das ist die
    Gewichtsteilung, auf die der klassische Weg nicht fuehrt (ein gewoehnliches
    Butterfly-Netz haette (d/2)*n*L Winkel, hier 5.120 statt 40).

    Reell und orthogonal statt komplex und unitaer: Die Aufgabe ist reell, und
    eine reelle Drehung haelt die Norm genauso. Das haelt den Vergleich mit
    `dense` und `fft_fest` sauber -- alle drei sind reelle Abbildungen von
    R^256 nach R^256.
    """

    def __init__(self, n: int = QUBITS, lagen: int = LAGEN):
        super().__init__()
        self.n, self.L = n, lagen
        self.theta = nn.Parameter(torch.rand(lagen, n) * 2 * math.pi)

    def wirksame_parameter(self) -> int:
        return self.theta.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        h = x
        for lage in range(self.L):
            for q in range(n):
                c = torch.cos(self.theta[lage, q])
                s = torch.sin(self.theta[lage, q])
                v = h.reshape(-1, 2 ** q, 2, 2 ** (n - 1 - q))
                a, b = v[:, :, 0, :], v[:, :, 1, :]
                h = torch.stack([c * a - s * b, s * a + c * b], dim=2).reshape(-1, 2 ** n)
        return h


class UngeteilteButterfly(nn.Module):
    """Butterfly mit EIGENEM Winkel je Paar -- 5.120 Parameter.

    Dasselbe Verbindungsmuster wie `OrthoButterfly`, aber ohne Gewichtsteilung:
    je Lage, Bit und Amplitudenpaar ein eigener Drehwinkel, also L*n*(d/2) =
    5*8*128 = 5.120 statt 40.

    WOZU DIESER ARM DA IST, und warum er NICHT in der angemeldeten Familie
    steht: Er ist am 27. August nachts ergaenzt, nachdem der Vorlauf gezeigt
    hat, dass `butterfly` (0,686) die echte Fouriertransformation (0,804) nicht
    erreicht. Der Verdacht: nicht die Struktur ist schuld, sondern die
    GEWICHTSTEILUNG -- und die ist gerade der quantenabgeleitete Anteil der
    Architektur (ein Qubit-Gatter wirkt auf ein Qubit, gleichgueltig wie gross
    der Zustandsraum ist).

    `untied` misst, WAS DIE GEWICHTSTEILUNG AUF DIESER AUFGABE KOSTET. Der
    Abstand (untied - butterfly) ist genau dieser Preis.

    WAS ER NICHT ZEIGT, und das war beim Einbauen anders gedacht: Ob die
    ungeteilte Familie die Fouriertransformation ueberhaupt DARSTELLEN kann,
    ist offen. `fourier_erreichbarkeit()` sollte das klaeren und ist an der
    eigenen Positivkontrolle gescheitert -- bei einem Ziel, das per
    Konstruktion in der Familie liegt, bleibt derselbe Restfehler stehen wie
    bei der Fouriermatrix (0,944 gegen 0,945). Die Anpassung konvergiert nicht;
    die Zahlen sind ihr Boden, keine Grenze der Familie.

    Deshalb gilt hier nur die schwaechere Deutung: Erreicht `untied` die
    Leistung von `fft_fest` und `butterfly` nicht, kostet die Gewichtsteilung
    messbar etwas. Ob der Rueckstand gegen `fft_fest` an der Darstellbarkeit
    oder am Training liegt, bleibt unentschieden.

    Er wird BESCHREIBEND berichtet und geht in KEINEN Holm-Vergleich ein. Die
    Vorregistrierung von Q-019 bleibt unveraendert; Regel 4 erlaubt die
    Korrektur nur ueber die angemeldete Familie, und ein nach dem Vorlauf
    ergaenzter Arm gehoert nicht hinein. Wer ihn nachtraeglich in die Familie
    zoege, haette die Kriterien nach dem Blick auf die Daten geaendert.
    """

    def __init__(self, n: int = QUBITS, lagen: int = LAGEN):
        super().__init__()
        self.n, self.L = n, lagen
        self.theta = nn.Parameter(torch.rand(lagen, n, 2 ** (n - 1)) * 2 * math.pi)

    def wirksame_parameter(self) -> int:
        return self.theta.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        h = x
        for lage in range(self.L):
            for q in range(n):
                th = self.theta[lage, q].reshape(2 ** q, 2 ** (n - 1 - q))
                c, s = torch.cos(th), torch.sin(th)
                v = h.reshape(-1, 2 ** q, 2, 2 ** (n - 1 - q))
                a, b = v[:, :, 0, :], v[:, :, 1, :]
                h = torch.stack([c * a - s * b, s * a + c * b], dim=2).reshape(-1, 2 ** n)
        return h


class DichteDrehung(nn.Module):
    """Freie dichte 256x256-Matrix -- 65.536 Parameter, keine Struktur."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def wirksame_parameter(self) -> int:
        return self.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class FestesFourier(nn.Module):
    """Die ECHTE Fouriertransformation, untrainiert -- null Parameter.

    Reelle orthonormale Fourierbasis, nach Frequenz sortiert: Index 2j traegt
    cos(2 pi j t), Index 2j+1 traegt sin(2 pi j t). Damit liegen die
    hoechstwertigen Bits des Index auf der Frequenz und das niederwertigste auf
    der Phase -- die natuerliche Anordnung fuer diese Aufgabe.

    Genau diese Anordnung macht die Saettigungspruefung noetig: Die drei
    hoechstwertigen Bits fallen mit den acht Baendern zusammen, und dieser Arm
    koennte die Aufgabe fast perfekt loesen. Ob er es tut, entscheidet
    `saettigung()` VOR dem Lauf, nicht die Hoffnung.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        t = torch.arange(dim, dtype=torch.float64) / dim
        basis = torch.empty(dim, dim, dtype=torch.float64)
        for j in range(dim // 2):
            basis[2 * j] = torch.cos(2 * math.pi * j * t)
            basis[2 * j + 1] = torch.sin(2 * math.pi * j * t)
        # j=0: sin ist identisch null -> durch die Nyquist-Kosinusreihe ersetzen,
        # sonst waere die Basis singulaer und die Abbildung nicht normerhaltend.
        basis[1] = torch.cos(math.pi * torch.arange(dim, dtype=torch.float64))
        basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("basis", basis.float())

    def wirksame_parameter(self) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.basis.T


class SchmaleSchicht(nn.Module):
    """Dichte Engstelle 256 -> 64 -> 256 -- 32.768 Parameter.

    Der Arm der Aufloesungspruefung: `dense` muss ihn schlagen. Die Engstelle
    ist ein grober, unstrittiger Kapazitaetsunterschied. Sieht der Aufbau den
    nicht, sieht er den feineren Strukturunterschied erst recht nicht.
    """

    def __init__(self, dim: int = DIM, eng: int = ENGSTELLE):
        super().__init__()
        self.ab = nn.Parameter(torch.randn(eng, dim) / math.sqrt(dim))
        self.auf = nn.Parameter(torch.randn(dim, eng) / math.sqrt(eng))

    def wirksame_parameter(self) -> int:
        return self.ab.numel() + self.auf.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.ab.T) @ self.auf.T


class ZeitreihenModell(nn.Module):
    """normieren -> Mittelschicht -> quadratische Auslese -> lineare Klassen.

    Kein Encoder. Das Signal IST die 256-dimensionale Eingabe, und ein
    trainierbarer Encoder davor waere genau der Fehler, an dem Q-014
    gescheitert ist: Er kann die Arbeit der Mittelschicht uebernehmen und macht
    den gemessenen Unterschied unsichtbar.
    """

    def __init__(self, arm: str):
        super().__init__()
        if arm == "butterfly":
            self.mitte: nn.Module = OrthoButterfly()
        elif arm == "untied":
            self.mitte = UngeteilteButterfly()
        elif arm == "dense":
            self.mitte = DichteDrehung()
        elif arm == "fft_fest":
            self.mitte = FestesFourier()
        elif arm == "narrow":
            self.mitte = SchmaleSchicht()
        else:
            raise ValueError(f"unbekannter Arm: {arm}")
        self.kopf = nn.Linear(QUBITS, KLASSEN)
        self.arm = arm

    def mittel_parameter(self) -> int:
        return self.mitte.wirksame_parameter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self.kopf(auslese(self.mitte(x)))


# ---------------------------------------------------------------------------
# Eine Zelle
# ---------------------------------------------------------------------------
def _zelle(arm: str, bedingung: str, seed: int, lr: float, threads: int,
           n_train: int = N_TRAIN, epochs: int = EPOCHS,
           anteile: int = ANTEILE, rauschen: float = RAUSCHEN,
           abstand: float = AMPLITUDENABSTAND) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)

    x_tr, y_tr = signale(seed, n_train, bedingung, anteile, rauschen, abstand)
    x_te, y_te = signale(seed + 500_000, N_TEST, bedingung, anteile, rauschen, abstand)

    modell = ZeitreihenModell(arm)
    opt = torch.optim.Adam(modell.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        ordnung = torch.randperm(len(x_tr), generator=g)
        for s in range(0, len(x_tr), BATCH):
            idx = ordnung[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(modell(x_tr[idx]), y_tr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 5.0)
            opt.step()

    modell.eval()
    with torch.no_grad():
        pred = modell(x_te).argmax(-1)
    klassen = torch.unique(y_te)
    bal = float(np.mean([float((pred[y_te == c] == c).float().mean())
                         for c in klassen]))
    return {"arm": arm, "bedingung": bedingung, "seed": seed, "lr": lr,
            "balanced_accuracy": bal,
            "accuracy": float((pred == y_te).float().mean()),
            "middle_parameters": modell.mittel_parameter(),
            "total_parameters": sum(p.numel() for p in modell.parameters())}


# ---------------------------------------------------------------------------
# Lernratensuche -- volles Budget, Verlaengerung in BEIDE Richtungen
# ---------------------------------------------------------------------------
def sweep_lr(bedingung: str, threads: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Regel 1 in allen drei Teilen, plus die Ergaenzung vom 27. August.

    (a) Eigene Tuning-Seeds je Arm.
    (b) Das Gitter muss das Optimum ENTHALTEN. Landet ein Arm auf einem Rand,
        wird in DIESE Richtung verlaengert -- nach oben mal 3, nach unten
        durch 3, bis zu ERWEITERUNGEN Mal. `unitarity_depth.py` verlaengert
        nur nach oben; genau ein Arm am unteren Rand ist es aber, der Q-015
        auf fashion_mnist entwertet (`global_orth` bei 0,003, dem
        Gitterminimum). Die Luecke ist hier geschlossen.
    (c) Die Suche hat DASSELBE Budget wie der Messlauf -- volle N_TRAIN und
        EPOCHS. Ein reduziertes Budget ist kein Sparmodell, sondern ein
        anderer Versuch: es verschiebt das Optimum systematisch nach oben.

    Rueckgabe: (beste Rate je Arm, Randvermerk je Arm).
    """
    best: Dict[str, float] = {}
    rand: Dict[str, str] = {}
    for arm in ARME:
        mittel: Dict[float, float] = {}
        raten = list(LR_GITTER)
        vermerk = "innen"
        for runde in range(ERWEITERUNGEN + 1):
            for lr in raten:
                if lr in mittel:
                    continue
                mittel[lr] = float(np.mean([
                    _zelle(arm, bedingung, s, lr, threads)["balanced_accuracy"]
                    for s in TUNING_SEEDS]))
            bester = max(mittel, key=mittel.get)
            if runde == ERWEITERUNGEN:
                if bester in (max(mittel), min(mittel)):
                    vermerk = "oben" if bester == max(mittel) else "unten"
                break
            if bester == max(mittel):
                naechste = max(mittel) * 3
                print(f"      {arm}: Optimum am oberen Rand ({bester:g}) -- "
                      f"verlaengere auf {naechste:g}")
                raten = [naechste]
            elif bester == min(mittel):
                naechste = min(mittel) / 3
                print(f"      {arm}: Optimum am unteren Rand ({bester:g}) -- "
                      f"verlaengere auf {naechste:g}")
                raten = [naechste]
            else:
                break
        best[arm] = max(mittel, key=mittel.get)
        rand[arm] = vermerk
        print(f"    {bedingung:<11} / {arm:<10} -> lr {best[arm]:g}"
              + (f"  ACHTUNG: Rand {vermerk}" if vermerk != "innen" else "")
              + "   " + "  ".join(f"{k:g}:{mittel[k]:.4f}" for k in sorted(mittel)))
    return best, rand


# ---------------------------------------------------------------------------
# Der Lauf
# ---------------------------------------------------------------------------
def run(bedingungen: Sequence[str] = BEDINGUNGEN,
        seeds: Sequence[int] = EVAL_SEEDS, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-019",
        description=("Traegt die Butterfly-Struktur auf periodischen Signalen "
                     "einen INHALTLICHEN Vorteil? Primaertest ist die "
                     "Wechselwirkung zwischen Architektur und Periodizitaet: "
                     "der Abstand (butterfly - dense) muss auf periodischen "
                     "Signalen groesser sein als auf zeitlich permutierten."),
        config={"arme": list(ARME), "bedingungen": list(bedingungen),
                "seeds": list(seeds), "dim": DIM, "qubits": QUBITS,
                "lagen": LAGEN, "klassen": KLASSEN, "engstelle": ENGSTELLE,
                "n_train": N_TRAIN, "n_test": N_TEST, "epochs": EPOCHS,
                "batch_size": BATCH, "lr_gitter": list(LR_GITTER),
                "erweiterungen": ERWEITERUNGEN, "torch_threads": threads,
                "anteile": ANTEILE, "rauschen": RAUSCHEN,
                "amplitudenabstand": AMPLITUDENABSTAND,
                "margin": MARGIN, "boden": BODEN, "saettigung": SAETTIGUNG,
                "test_metric": "balanced_accuracy",
                "primaer": ("Wechselwirkung: (butterfly-dense) periodisch > "
                            "(butterfly-dense) permutiert, Wilcoxon einseitig"),
                "p2": ("(butterfly-dense) auf periodisch positiv, "
                       "Wilcoxon einseitig"),
                "holm_familie": ["PRIMAER", "P2"],
                "primaere_bedingung": PRIMAERE_BEDINGUNG,
                "aufloesungspruefung": "dense schlaegt narrow auf periodisch",
                "anlass": ("Die Butterfly-Struktur IST die schnelle "
                           "Fouriertransformation -- die einzige Anwendung des "
                           "Projekts mit erwartetem inhaltlichem statt nur "
                           "Effizienzvorteil.")},
    )
    for bedingung in bedingungen:
        print(f"\n  {bedingung} -- Lernratensuche (volles Budget)")
        lr_map, rand = sweep_lr(bedingung, threads)
        rec.config[f"lr_map_{bedingung}"] = dict(lr_map)
        rec.config[f"lr_rand_{bedingung}"] = dict(rand)

        for i, seed in enumerate(seeds, 1):
            for arm in ARME:
                rec.add(**_zelle(arm, bedingung, seed, lr_map[arm], threads))
            if i % 5 == 0 or i == len(seeds):
                print(f"    seed {seed} fertig ({i}/{len(seeds)})")

        bester = max(np.mean([r["balanced_accuracy"] for r in rec.records
                              if r["arm"] == a and r["bedingung"] == bedingung])
                     for a in ARME)
        print(f"    bester Arm bei {bedingung}: {bester:.4f}")
        if bester < BODEN:
            print(f"    BODEN UNTERSCHRITTEN -- kein Arm ueber {BODEN:.3f}. "
                  f"Diese Bedingung misst Trainingsversagen.")
        # Regel 5: nach JEDER Bedingung sichern, nicht am Ende.
        print("    Zwischenstand gesichert:")
        rec.save()
    return rec


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def _werte(records, arm: str, bedingung: str) -> np.ndarray:
    zeilen = sorted((r for r in records if r["arm"] == arm
                     and r["bedingung"] == bedingung), key=lambda r: r["seed"])
    return np.array([r["balanced_accuracy"] for r in zeilen])


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]

    boden = float(config.get("boden", BODEN))
    aus: Dict[str, object] = {}

    print()
    print("RESEARCH ID Q-019 | Butterfly gegen dicht auf periodischen Signalen")

    # ---- Uebersicht je Bedingung ----
    mittel: Dict[str, Dict[str, float]] = {}
    boden_ok: Dict[str, bool] = {}
    for bed in BEDINGUNGEN:
        print()
        print(f"--- {bed} " + "-" * max(4, 60 - len(bed)))
        kopf = f'  {"Arm":<12}{"bal. Genauigkeit":>18}{"Parameter":>12}'
        print(kopf); print("  " + "-" * (len(kopf) - 2))
        mittel[bed] = {}
        for arm in ARME:
            v = _werte(records, arm, bed)
            if not len(v):
                continue
            mittel[bed][arm] = float(v.mean())
            par = next((r["middle_parameters"] for r in records
                        if r["arm"] == arm and r["bedingung"] == bed), 0)
            print(f"  {arm:<12}{v.mean():>18.4f}{par:>12,}")
        if mittel[bed]:
            bester = max(mittel[bed].values())
            boden_ok[bed] = bool(bester >= boden)
            print(f"  bester Arm: {bester:.4f}   Bodenpruefung: "
                  f"{'bestanden' if boden_ok[bed] else 'NICHT BESTANDEN'} "
                  f"(Schwelle {boden:.3f})")
        aus[f"mittel_{bed}"] = dict(mittel[bed])
    aus["bodenpruefung"] = dict(boden_ok)

    # ---- die beiden vorregistrierten Tests ----
    b_per = _werte(records, "butterfly", "periodisch")
    d_per = _werte(records, "dense", "periodisch")
    b_perm = _werte(records, "butterfly", "permutiert")
    d_perm = _werte(records, "dense", "permutiert")

    vollstaendig = bool(len(b_per) and len(b_per) == len(d_per) == len(b_perm) == len(d_perm))
    if not vollstaendig:
        print("\n  Unvollstaendige Daten -- die vorregistrierte Familie kann "
              "nicht gerechnet werden.")
        aus["verdict"] = ("HU -- unvollstaendiger Lauf, keine Aussage. "
                          f"periodisch n={len(b_per)}, permutiert n={len(b_perm)}")
        print(f"BEWERTUNG: {aus['verdict']}")
        return aus

    delta_per = b_per - d_per
    delta_perm = b_perm - d_perm
    wechsel = delta_per - delta_perm

    # EINSEITIG, wie vorregistriert. scipy liefert sonst zweiseitig -- genau
    # dieser Unterschied hat Q-011 einmal falsch berichtet.
    p_wechsel = float(stats.wilcoxon(wechsel, alternative="greater").pvalue)
    p_wechsel_neg = float(stats.wilcoxon(wechsel, alternative="less").pvalue)
    p_p2 = float(stats.wilcoxon(delta_per, alternative="greater").pvalue)
    p_p2_neg = float(stats.wilcoxon(delta_per, alternative="less").pvalue)
    adj = holm([p_wechsel, p_p2])
    gleich_wechsel = tost_paired(delta_per, delta_perm, MARGIN)

    print()
    print("=" * 92)
    print("VORREGISTRIERTE FAMILIE")
    print(f"  Abstand (butterfly - dense) periodisch : {delta_per.mean():+.4f}   "
          f"Siege {int((b_per > d_per).sum())}/{len(b_per)}")
    print(f"  Abstand (butterfly - dense) permutiert : {delta_perm.mean():+.4f}   "
          f"Siege {int((b_perm > d_perm).sum())}/{len(b_perm)}")
    print(f"  PRIMAER  Wechselwirkung {wechsel.mean():+.4f}  p={p_wechsel:.4f}  "
          f"Holm={adj[0]:.4f}   (Gegenrichtung p={p_wechsel_neg:.4f})")
    print(f"  P2       periodisch     {delta_per.mean():+.4f}  p={p_p2:.4f}  "
          f"Holm={adj[1]:.4f}   (Gegenrichtung p={p_p2_neg:.4f})")
    print(f"  TOST auf die Wechselwirkung: p={gleich_wechsel['p_tost']:.4f}, "
          f"erreicht ±{gleich_wechsel['achieved']:.4f}")

    aus["primaer"] = {"wechselwirkung": float(wechsel.mean()), "p": p_wechsel,
                      "p_holm": adj[0], "p_gegenrichtung": p_wechsel_neg,
                      "p_tost": gleich_wechsel["p_tost"],
                      "erreicht": gleich_wechsel["achieved"]}
    aus["p2"] = {"delta": float(delta_per.mean()), "p": p_p2, "p_holm": adj[1],
                 "p_gegenrichtung": p_p2_neg}
    aus["delta_permutiert"] = float(delta_perm.mean())

    # ---- H1+ : erreicht die gelernte Naeherung die echte Transformation? ----
    f_per = _werte(records, "fft_fest", "periodisch")
    if len(f_per) == len(b_per) and len(f_per):
        gg = tost_paired(b_per, f_per, MARGIN)
        p_ff = float(stats.wilcoxon(b_per, f_per, alternative="less").pvalue)
        print(f"  H1+      butterfly gegen fft_fest {b_per.mean() - f_per.mean():+.4f}  "
              f"TOST p={gg['p_tost']:.4f}  schlechter? p={p_ff:.4f}")
        aus["h1plus"] = {"delta": float(b_per.mean() - f_per.mean()),
                         "p_tost": gg["p_tost"], "p_schlechter": p_ff,
                         "erreicht": gg["achieved"]}
        nicht_schlechter = bool(gg["p_tost"] < 0.05 or b_per.mean() >= f_per.mean())
    else:
        aus["h1plus"] = None
        nicht_schlechter = False

    # ---- BESCHREIBEND, ausserhalb der Familie: der ungeteilte Butterfly ----
    # Kein Holm, kein Urteil haengt daran. Regel 4 erlaubt die Korrektur nur
    # ueber die angemeldete Familie, und dieser Arm ist nach dem Vorlauf
    # ergaenzt worden. Er beantwortet eine eigene Frage: Liegt der Rueckstand
    # von `butterfly` an der Struktur oder an der Gewichtsteilung?
    u_per = _werte(records, "untied", "periodisch")
    u_perm = _werte(records, "untied", "permutiert")
    if len(u_per) == len(b_per) and len(u_per):
        f_p = _werte(records, "fft_fest", "periodisch")
        print()
        print("BESCHREIBEND (ausserhalb der Familie, ohne Holm-Korrektur)")
        print(f"  untied - butterfly  periodisch {u_per.mean() - b_per.mean():+.4f}"
              f"   was die Gewichtsteilung kostet")
        if len(f_p) == len(u_per):
            print(f"  untied - fft_fest   periodisch {u_per.mean() - f_p.mean():+.4f}"
                  f"   erreicht die gelernte Struktur die echte Transformation?")
        print(f"  untied - dense      periodisch {u_per.mean() - d_per.mean():+.4f}"
              f"   {u_per.mean() - d_perm.mean() * 0 + (u_perm.mean() - d_perm.mean()):+.4f} permutiert")
        aus["untied"] = {
            "periodisch": float(u_per.mean()), "permutiert": float(u_perm.mean()),
            "minus_butterfly": float(u_per.mean() - b_per.mean()),
            "minus_dense": float(u_per.mean() - d_per.mean()),
            "minus_fft": (float(u_per.mean() - f_p.mean())
                          if len(f_p) == len(u_per) else None),
            "wechselwirkung": float((u_per - d_per).mean() - (u_perm - d_perm).mean()),
            "hinweis": ("beschreibend, ausserhalb der angemeldeten Familie, "
                        "nach dem Vorlauf ergaenzt -- kein Holm, kein Urteil")}

    # ---- Aufloesungspruefung (Regel 6) ----
    # bool() ist hier nicht kosmetisch: `np.False_ is False` ist FALSCH, und
    # genau dadurch wurde die Pruefung in Q-011 berechnet, gedruckt und beim
    # Urteil uebergangen.
    n_per = _werte(records, "narrow", "periodisch")
    res_ok = None
    if len(n_per) == len(d_per) and len(n_per):
        p_res = float(stats.wilcoxon(d_per, n_per, alternative="greater").pvalue)
        res_ok = bool(d_per.mean() > n_per.mean() and p_res < 0.05)
        text = (f"Aufloesungspruefung {'BESTANDEN' if res_ok else 'NICHT BESTANDEN'}: "
                f"dense gegen narrow auf periodisch "
                f"{d_per.mean() - n_per.mean():+.4f}, p={p_res:.4f}."
                + ("" if res_ok else " Der Aufbau sieht nicht einmal den groben "
                   "Kapazitaetsunterschied einer Engstelle auf 64 Dimensionen -- "
                   "der feinere Strukturunterschied ist damit erst recht nicht "
                   "aufloesbar, und kein Ergebnis dieses Laufs zaehlt."))
        print()
        print(f"  {text}")
        aus["aufloesungspruefung"] = {"bestanden": res_ok, "p": p_res,
                                      "delta": float(d_per.mean() - n_per.mean()),
                                      "text": text}

    # ---- Randvermerke der Lernratensuche (Regel 1b) ----
    raender = {k: v for k, v in config.items() if k.startswith("lr_rand_")}
    schlimm = {k: {a: r for a, r in v.items() if r != "innen"}
               for k, v in raender.items()}
    schlimm = {k: v for k, v in schlimm.items() if v}
    if schlimm:
        print(f"\n  ACHTUNG, Lernrate am Gitterrand trotz Verlaengerung: {schlimm}")
        aus["lr_rand_warnung"] = schlimm

    # ---- Urteil ----
    print()
    boden_gesamt = all(boden_ok.get(b, False) for b in BEDINGUNGEN)

    # REIHENFOLGE DER ZWEIGE -- am 27. August VOR dem Lauf festgelegt und
    # begruendet, weil sie das Urteil bestimmt und nicht beliebig ist.
    #
    # Die Gleichwertigkeitspruefung (TOST) kommt VOR H1 und VOR H0-. Grund:
    # Ein einseitiger Test bei alpha=0,05 spricht bei einem WAHREN Nulleffekt
    # in 5 % der Faelle an, in jede Richtung. Steht H1 davor, wird ein
    # Nulleffekt in rund 10 % der Faelle als "inhaltlicher Vorteil" oder als
    # "Gegenrichtung" gemeldet -- obwohl TOST im selben Atemzug beweist, dass
    # der Effekt KLEINER ist als die eigene Erheblichkeitsgrenze von +-0,005.
    # Das ist Regel 3 in der anderen Richtung gelesen: "nicht signifikant" ist
    # nicht "gleich" -- aber "signifikant" ist eben auch nicht "erheblich".
    #
    # Gefunden am 27. August im Vortest: nachdem der Arm `untied` die
    # Zufallsziehungen verschoben hatte, meldete Szenario F (kein Unterschied
    # zwischen den Bedingungen, kleines Rauschen) einmal H0- und einmal H1.
    # Beides waere falsch gewesen, und beides haette an der Auswertung
    # gelegen, nicht an den Daten.
    #
    # Die Vorregistrierung nennt die vier Ausgaenge, aber keine Rangfolge.
    # Sie wird hier festgelegt, bevor Messzahlen existieren, und steht so
    # auch im Tagebuch.
    if res_ok is not None and not res_ok:
        v = ("HU -- Aufloesungspruefung durchgefallen. Der Aufbau reproduziert "
             "den groben Kapazitaetsunterschied nicht; kein Ergebnis dieses "
             "Laufs ist interpretierbar. Das ist KEIN Nullbefund -- der Lauf "
             "muss mit groesserem Kontrast wiederholt werden.")
    elif not boden_gesamt:
        offen = [b for b in BEDINGUNGEN if not boden_ok.get(b, False)]
        v = (f"HU -- Bodenpruefung durchgefallen bei: {', '.join(offen)}. "
             f"Kein Arm kommt dort deutlich ueber Rateniveau; die Zelle misst "
             f"Trainingsversagen, nicht Architektur.")
    elif gleich_wechsel["p_tost"] < 0.05:
        v = (f"H0 -- keine Wechselwirkung; die beiden Abstaende sind gleichwertig "
             f"innerhalb +-{MARGIN} (TOST p={gleich_wechsel['p_tost']:.4f}, "
             f"gemessen {wechsel.mean():+.4f}). Der Vorteil der Butterfly-Struktur "
             f"haengt nicht an der Periodizitaet, die Fourier-Erzaehlung faellt. "
             f"Positiver Gleichwertigkeitsnachweis, nicht blosses p>0,05.")
        if adj[0] < 0.05:
            v += (f" ANMERKUNG: Der Primaertest ist zugleich signifikant "
                  f"(Holm p={adj[0]:.4f}). Ein Effekt kann statistisch "
                  f"nachweisbar UND kleiner als die Erheblichkeitsgrenze sein; "
                  f"massgeblich ist hier die Grenze.")
    elif adj[0] < 0.05 and wechsel.mean() > 0 and adj[1] < 0.05:
        v = (f"H1 -- die Struktur bringt einen inhaltlichen Vorteil. Der Abstand "
             f"(butterfly - dense) ist auf periodischen Signalen {delta_per.mean():+.4f} "
             f"und auf permutierten {delta_perm.mean():+.4f}; die Wechselwirkung "
             f"betraegt {wechsel.mean():+.4f} (Holm p={adj[0]:.4f}). Der Vorteil "
             f"verschwindet also, sobald die Periodizitaet zerstoert ist -- er "
             f"kommt aus der Passung zur Datenstruktur und nicht aus der "
             f"Parameterzahl. Das ist der erste Befund des Projekts, der ueber "
             f"'gleichwertig, aber billiger' hinausgeht.")
        v += (" H1+ gilt zusaetzlich: die gelernte Naeherung erreicht die echte "
              "Fouriertransformation." if nicht_schlechter else
              " H1+ gilt NICHT: fft_fest bleibt besser -- die gelernte Naeherung "
              "erreicht die echte Transformation nicht.")
    elif adj[0] < 0.05 and wechsel.mean() > 0 and adj[1] >= 0.05:
        # Ergaenzt nach dem Vortest: Ohne diesen Zweig faellt dieser Fall in
        # das HU unten und wird als "weder Wechselwirkung noch Gleichwertigkeit
        # belegt" gemeldet -- obwohl die Wechselwirkung gerade belegt IST. Das
        # ist dieselbe Fehlerklasse wie der fehlende H0--Zweig in Q-011: ein
        # Urteil, das den eigenen Daten widerspricht. Nach dem Vorlauf ist das
        # zugleich der WAHRSCHEINLICHSTE Ausgang dieses Laufs.
        v = (f"HU -- geteiltes Ergebnis, und das ist inhaltlich interessant. Die "
             f"Wechselwirkung HAELT ({wechsel.mean():+.4f}, Holm p={adj[0]:.4f}), "
             f"aber P2 faellt: auf periodischen Signalen ist "
             f"(butterfly - dense) nur {delta_per.mean():+.4f} (Holm p={adj[1]:.4f}). "
             f"Der Unterschied zwischen den Bedingungen kommt also nicht daher, "
             f"dass die Struktur auf periodischen Signalen GEWINNT, sondern "
             f"daher, dass sie auf permutierten VERLIERT ({delta_perm.mean():+.4f}). "
             f"Nach der Vorregistrierung reicht das fuer H1 nicht -- Holm gilt "
             f"ueber die angemeldete Familie, und dazu gehoeren beide Tests. "
             f"Als Beobachtung gehoert es trotzdem berichtet.")
    elif p_wechsel_neg < 0.05 and wechsel.mean() < 0:
        # Kommt NACH dem Gleichwertigkeitstest: ein signifikant negativer, aber
        # nachweislich vernachlaessigbarer Effekt ist H0, nicht H0-.
        # Der Zweig selbst ist da, weil sein Fehlen in Q-011, Q-014 und Q-015
        # dreimal zu einer Aussage gefuehrt hat, die den Daten widersprach.
        v = (f"H0- -- die Wechselwirkung geht in die GEGENRICHTUNG "
             f"({wechsel.mean():+.4f}, p={p_wechsel_neg:.4f}). Der Abstand "
             f"(butterfly - dense) ist auf PERMUTIERTEN Signalen groesser als "
             f"auf periodischen. Das ist kein Nullbefund, sondern das Gegenteil "
             f"der Hypothese: Was die Butterfly-Struktur beitraegt, haengt gerade "
             f"NICHT an der Periodizitaet. Die Fourier-Erzaehlung ist damit nicht "
             f"nur unbelegt, sondern widerlegt -- und das gehoert als "
             f"eigenstaendiges Ergebnis berichtet.")
    else:
        v = (f"HU -- unentschieden. Weder Wechselwirkung noch Gleichwertigkeit "
             f"belegt; die Daten tragen erst ab +-{gleich_wechsel['achieved']:.4f}, "
             f"gemessen wurde {wechsel.mean():+.4f}.")

    print(f"BEWERTUNG: {v}")
    print("=" * 92)
    aus["verdict"] = v
    return aus


def fourier_erreichbarkeit(schritte: int = 2500, lr: float = 0.05) -> Dict[str, float]:
    """Wie nah kommen die beiden Butterfly-Familien an die Fouriermatrix?

    WARUM DAS GEMESSEN UND NICHT BEHAUPTET WIRD: Die Begruendung fuer den Arm
    `untied` lautet, die GETEILTEN Winkel koennten die Fouriertransformation
    strukturell nicht darstellen, die ungeteilten schon. Der erste Halbsatz
    folgt aus der Konstruktion; der zweite ist eine Behauptung -- das
    Verbindungsmuster hier hat keine Bit-Umkehr-Permutation und eine feste
    Stufenreihenfolge, und ob die reelle DFT darin liegt, ist damit offen.

    Eine Behauptung ist kein Beleg. Hier wird sie gemessen: die Winkel werden
    direkt auf die Fouriermatrix angepasst und der relative Restfehler
    berichtet. Kostet Sekunden und entscheidet, ob der Arm die Frage
    ueberhaupt beantworten kann.
    """
    eye = torch.eye(DIM)
    ergebnis: Dict[str, float] = {}

    def anpassen(bauer, ziel_matrix) -> float:
        torch.manual_seed(4711)
        netz = bauer()
        opt = torch.optim.Adam(netz.parameters(), lr=lr)
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, schritte)
        for _ in range(schritte):
            opt.zero_grad(set_to_none=True)
            # Zeilen der dargestellten Matrix: Butterfly angewandt auf die
            # Einheitsvektoren, also B = netz(I) zeilenweise.
            (netz(eye) - ziel_matrix).pow(2).sum().backward()
            opt.step(); plan.step()
        with torch.no_grad():
            return float((netz(eye) - ziel_matrix).norm() / ziel_matrix.norm())

    print("=" * 92)
    print("ERREICHBARKEIT DER FOURIERMATRIX -- gemessen, nicht behauptet")
    print("=" * 92)

    # POSITIVKONTROLLE ZUERST (Regel 6). Ohne sie waere ein grosser Restfehler
    # nicht von einem schlecht konvergierenden Anpassungsvorgang zu
    # unterscheiden, und "die Familie KANN es nicht" waere unbelegt. Ziel ist
    # hier eine Matrix, die per Konstruktion IN der Familie liegt: eine
    # zufaellig gewuerfelte ungeteilte Butterfly. Der Rest muss klein werden.
    torch.manual_seed(20260827)
    with torch.no_grad():
        in_familie = UngeteilteButterfly()(eye)
    rest_pk = anpassen(UngeteilteButterfly, in_familie)
    ergebnis["positivkontrolle"] = rest_pk
    marke = "OK" if rest_pk < 0.10 else "ACHTUNG -- Anpassung konvergiert nicht"
    print(f"  Positivkontrolle (Ziel liegt in der Familie)   Rest {rest_pk:.4f}  {marke}")
    print()

    ziel = FestesFourier().basis.T                     # (256, 256), orthonormal
    for name, bauer in (("butterfly (geteilt, 40)", OrthoButterfly),
                        ("untied (ungeteilt, 5.120)", UngeteilteButterfly)):
        rest = anpassen(bauer, ziel)
        ergebnis[name] = rest
        print(f"  {name:<28} relativer Restfehler {rest:.4f}  "
              f"({bauer().wirksame_parameter():>5,} Parameter)")
    print()
    print("  Bezugspunkt: zwei zufaellige orthogonale 256x256-Matrizen haben")
    print("  einen relativen Abstand von sqrt(2) = 1,414. Ein Wert nahe daran")
    print("  heisst 'keinerlei Ausrichtung'.")
    print()
    # Die Lesart haengt an der Positivkontrolle, und zwar vollstaendig. Ohne
    # sie wuerde hier "die Familie kann es nicht" stehen, wo in Wahrheit "die
    # Anpassung findet es nicht" richtig ist. Am 27. August genau so passiert:
    # Kontrolle 0,9441, Fourier 0,9448 -- nicht unterscheidbar.
    if rest_pk >= 0.10:
        print("  KEINE AUSSAGE MOEGLICH. Die Positivkontrolle erreicht selbst")
        print(f"  bei einem Ziel, das per Konstruktion IN der Familie liegt, nur")
        print(f"  {rest_pk:.4f}. Die Anpassung konvergiert also nicht, und die")
        print("  Fourier-Zahlen oben sind ihr Boden, keine Darstellbarkeits-")
        print("  grenze. Ein Rueckstand gegen `fft_fest` im Lauf laesst sich")
        print("  damit NICHT als Darstellbarkeitsfrage deuten.")
        ergebnis["aussagekraeftig"] = 0.0
    else:
        print("  Lesart: 0 heisst 'stellt die Fouriermatrix exakt dar'. Ein")
        print("  grosser Rest heisst, die Familie KANN sie nicht -- dann ist ein")
        print("  Rueckstand gegen `fft_fest` im Lauf keine Trainings-, sondern")
        print("  eine Darstellbarkeitsfrage.")
        ergebnis["aussagekraeftig"] = 1.0
    print("=" * 92)
    return ergebnis


# ---------------------------------------------------------------------------
# Pflichtvortest: jeden Ausgang EINZELN erzwingen
# ---------------------------------------------------------------------------
def _kunst(delta_per: float, delta_perm: float, streuung: float,
           narrow_ab: float = 0.08, niveau: float = 0.60,
           n: int = 15, seed: int = 0) -> Recorder:
    """Synthetischer Recorder mit vorgegebenen Abstaenden.

    `niveau` ist die Genauigkeit von `dense`, `narrow_ab` der Abstand fuer die
    Aufloesungspruefung, `streuung` die Streuung JE ARM.

    ACHTUNG, und das ist der Grund, warum diese Funktion ueberarbeitet wurde:
    Im ersten Entwurf bekam `dense` gar kein eigenes Rauschen, und die
    Seed-Schwankung war allen Armen gemeinsam. In der gepaarten Differenz
    (butterfly - dense) kuerzt sich ein gemeinsamer Anteil aber weg -- das
    Szenario "schwach und verrauscht" war deshalb in Wahrheit ein starkes,
    sauberes Szenario, und der Vortest hat einen Fehler gemeldet, wo keiner
    war. Ein falsch gebauter Pruefstand ist genauso schaedlich wie ein falsch
    gebauter Test.

    Jetzt: gemeinsamer Seed-Anteil PLUS unabhaengiges Rauschen je Arm. Damit
    hat die Differenz zweier Arme die Streuung sqrt(2)*streuung und die
    Wechselwirkung 2*streuung -- nachrechenbar, und die Szenarien lassen sich
    gezielt auf "erreichbar" oder "unerreichbar" stellen.
    """
    import zlib

    def _rng(*teile) -> np.random.Generator:
        """Eigener Strom je (Arm, Bedingung, Seed).

        WARUM NICHT EIN GEMEINSAMER STROM: Beim Ergaenzen des Arms `untied`
        verschob sich dadurch jede nachfolgende Ziehung, und zwei Szenarien
        kippten -- ohne dass sich an der Auswertung etwas geaendert haette.
        Ein Vortest, dessen Ergebnis vom Hinzufuegen eines unbeteiligten Arms
        abhaengt, prueft nicht die Auswertung, sondern die Reihenfolge des
        Zufallsgenerators. `zlib.crc32` statt `hash`, weil Pythons String-Hash
        zwischen Prozessen zufaellig ist.
        """
        marke = "|".join(str(x) for x in teile)
        return np.random.default_rng([seed, zlib.crc32(marke.encode())])

    rec = Recorder(experiment_id="Q-019-vortest", description="synthetisch",
                   config={"boden": BODEN, "margin": MARGIN,
                           "lr_rand_periodisch": {a: "innen" for a in ARME},
                           "lr_rand_permutiert": {a: "innen" for a in ARME}})
    for bed, d in (("periodisch", delta_per), ("permutiert", delta_perm)):
        for seed_ in range(1901, 1901 + n):
            grund = niveau + _rng("grund", seed_).normal(0, 0.01)
            versatz = {"dense": 0.0, "butterfly": d, "untied": d * 0.5,
                       "fft_fest": 0.02, "narrow": -narrow_ab}
            for arm, ab in versatz.items():
                w = grund + ab + _rng(arm, bed, seed_).normal(0, streuung)
                rec.add(arm=arm, bedingung=bed, seed=seed_, lr=0.003,
                        balanced_accuracy=float(w), accuracy=float(w),
                        middle_parameters=40, total_parameters=100)
    return rec


def vortest() -> bool:
    """Der Pflichtvortest. Er hat vor Q-011 fuenf Fehler gefunden, davon einen,
    der den Lauf unbemerkt entwertet haette.

    Geprueft wird nicht, ob die Auswertung laeuft, sondern ob sie **jeden**
    Ausgang trifft -- auch die unerwarteten. Wer nur die erwarteten Ausgaenge
    kodiert, verliert den interessanten.
    """
    # Die Streuungen sind nachgerechnet, nicht geraten: bei Streuung s je Arm
    # hat die Differenz (butterfly - dense) die Streuung sqrt(2)*s und die
    # Wechselwirkung 2*s. Bei 15 Seeds ist der Standardfehler der
    # Wechselwirkung also 2*s/sqrt(15).
    #
    # Die beiden Gleichwertigkeits-Szenarien (B und F) brauchen s = 0,0015.
    # Mit s = 0,003 bestand der Vortest nur in zwei von fuenf Ziehungen -- das
    # Urteil haette dort am Zufall gehangen. Der Grund ist ausrechenbar und
    # gehoert vor den Lauf: TOST besteht nur, wenn |Mittel| + t_krit*SE unter
    # der Grenze +-0,005 bleibt. Bei 15 Seeds heisst das
    #     sd(Wechselwirkung) < 0,005 * sqrt(15) / 1,7613 = 0,0110
    # und selbst dann nur, wenn das Mittel fast exakt null trifft. Siehe die
    # Anmerkung zu H0 im Kopf dieser Datei.
    # Je Szenario: (Name, Aufbau, erwartete Marke, erwarteter Zweig,
    #               Ziehungen, geforderter Anteil)
    #
    # Die meisten Szenarien muessen in JEDER Ziehung dasselbe Urteil liefern --
    # Anteil 1,0. Szenario G ist anders, und das ist kein Mangel des Codes:
    # Ein Fall, der "unentschieden" ergeben soll, liegt konstruktionsbedingt
    # ZWISCHEN den Schwellen, und ein einseitiger Test bei alpha=0,05 spricht
    # bei einem wahren Nulleffekt in 5 % der Ziehungen an -- unabhaengig davon,
    # wie gross das Rauschen ist (der Fall ist skaleninvariant). 5/5 zu fordern
    # hiesse, vom Test zu verlangen, dass er sein eigenes Niveau
    # unterschreitet. G laeuft deshalb ueber 40 Ziehungen und prueft den
    # ANTEIL; die Rate der Fehlurteile wird ausgewiesen statt weggewuenscht.
    faelle = [
        ("A  starke Wechselwirkung, beide Tests halten",
         dict(delta_per=0.06, delta_perm=0.00, streuung=0.008),
         "H1", "inhaltlichen Vorteil", 5, 1.0),
        ("B  gar kein Unterschied, beide Bedingungen gleich",
         dict(delta_per=0.000, delta_perm=0.000, streuung=0.0015),
         "H0", "gleichwertig", 5, 1.0),
        ("C  Wechselwirkung in der GEGENRICHTUNG",
         dict(delta_per=0.00, delta_perm=0.06, streuung=0.008),
         "H0-", "GEGENRICHTUNG", 5, 1.0),
        ("D  Aufloesungspruefung faellt durch (narrow schlaegt dense)",
         dict(delta_per=0.06, delta_perm=0.00, streuung=0.008, narrow_ab=-0.08),
         "HU", "Aufloesungspruefung durchgefallen", 5, 1.0),
        ("E  Boden unterschritten (alles auf Rateniveau)",
         dict(delta_per=0.06, delta_perm=0.00, streuung=0.008, niveau=0.14),
         "HU", "Bodenpruefung durchgefallen", 5, 1.0),
        ("F  Butterfly gewinnt ueberall gleich viel -> keine Wechselwirkung",
         dict(delta_per=0.06, delta_perm=0.06, streuung=0.0015),
         "H0", "gleichwertig", 5, 1.0),
        ("G  reiner Nulleffekt bei grossem Rauschen -> unentschieden",
         dict(delta_per=0.000, delta_perm=0.000, streuung=0.025),
         "HU", "unentschieden", 40, 0.80),
        ("H  Wechselwirkung haelt, aber P2 faellt (Struktur verliert nur "
         "auf permutiert)",
         dict(delta_per=-0.010, delta_perm=-0.060, streuung=0.008),
         "HU", "geteiltes Ergebnis", 5, 1.0),
    ]
    print("=" * 92)
    print("VORTEST Q-019 -- jeder Ausgang wird einzeln erzwungen")
    print("Jedes Szenario laeuft ueber mehrere Rauschziehungen. Ein Urteil, das "
          "vom Zufall\nder Ziehung abhaengt, ist kein Urteil -- genau das hat "
          "der erste Entwurf dieses\nVortests verdeckt.")
    print("=" * 92)
    alle_ok = True
    for i, (name, kw, erwartet, zweig, ziehungen, anteil) in enumerate(faelle):
        print(f"\n########## Szenario {name}   erwartet: {erwartet} ({zweig!r})"
              f"   {ziehungen} Ziehungen, gefordert {anteil:.0%}")
        marken, zweige = [], []
        for zug in range(ziehungen):
            rec = _kunst(seed=100 * i + zug, **kw)
            urteil = str(analyse(rec)["verdict"])
            marken.append(urteil.split(" ")[0])
            zweige.append(zweig in urteil)
        # Die Marke allein reicht NICHT: vier verschiedene Ausgaenge heissen
        # "HU" (Aufloesung, Boden, geteilt, unentschieden). Wuerden zwei davon
        # verschmelzen, saehe der Vortest nur die Marke und meldete
        # "bestanden" -- dieselbe Blindheit, an der Q-011 die stillschweigend
        # uebergangene Aufloesungspruefung hatte. Deshalb wird zusaetzlich der
        # ZWEIG selbst geprueft.
        from collections import Counter
        treffer = sum(1 for m, z in zip(marken, zweige) if m == erwartet and z)
        quote = treffer / len(marken)
        ok = quote >= anteil - 1e-9
        alle_ok = alle_ok and ok
        print(f">>> Szenario {name[0]}: {'OK' if ok else 'FEHLER'} "
              f"-- {treffer}/{len(marken)} = {quote:.0%} "
              f"(gefordert {anteil:.0%}), Urteile {dict(Counter(marken))}")
    print("\n" + "=" * 92)
    print("VORTEST BESTANDEN" if alle_ok else "VORTEST FEHLGESCHLAGEN")
    print("=" * 92)
    return alle_ok


def test_gitterverlaengerung() -> bool:
    """Prueft die Verlaengerung des Lernratengitters in BEIDE Richtungen.

    WARUM ES DIESEN TEST GIBT: Die Verlaengerung nach unten ist eine Korrektur,
    die am 27. August nachts entstanden ist, nachdem ein Arm am unteren Rand
    Q-015 auf fashion_mnist entwertet hatte. Eine Korrektur ist aber kein
    Beleg, sondern eine neue Behauptung -- und gehoert genauso gegen
    synthetische Daten geprueft wie das Original. Beim ersten Q-011-Vortest war
    genau das der Fund: der erste Korrekturversuch war selbst falsch.

    Geprueft wird mit einer kuenstlichen Genauigkeitsfunktion, deren Optimum
    ausserhalb des Startgitters (0,001 .. 0,03) liegt -- einmal darueber,
    einmal darunter, einmal mittendrin.
    """
    import math as _math
    global _zelle
    echt = _zelle
    # Die Reichweite ist ausrechenbar und gehoert hierher: Startgitter
    # 0,001 .. 0,03, ERWEITERUNGEN = 3 Schritte mit Faktor 3 je Richtung, also
    # erreichbar von 0,001/27 = 3,7e-05 bis 0,03*27 = 0,81.
    #
    # WICHTIG, und beim ersten Anlauf dieses Tests falsch erwartet: Ein Optimum
    # GENAU am Rand der Reichweite wird als "Rand" gemeldet, nicht als "innen"
    # -- auch wenn die gewaehlte Rate zufaellig die richtige ist. Das ist die
    # ehrliche Antwort: ohne eine Messung jenseits des Optimums laesst sich
    # "Gipfel" nicht von "steigt noch" unterscheiden. Als "innen" gilt ein
    # Optimum erst, wenn dahinter noch ein SCHLECHTERER Punkt gemessen wurde.
    faelle = [
        ("Optimum im Startgitter (0,003)", 0.003, "innen", 0.003),
        ("Optimum eine Verlaengerung OBEN (0,09)", 0.09, "innen", 0.09),
        ("Optimum eine Verlaengerung UNTEN (3,3e-04)", 1e-3 / 3, "innen", 1e-3 / 3),
        ("Optimum ausserhalb der Reichweite oben (300)", 300.0, "oben", 0.81),
        ("Optimum ausserhalb der Reichweite unten (1e-09)", 1e-9, "unten", 1e-3 / 27),
    ]
    print("=" * 92)
    print("TEST GITTERVERLAENGERUNG -- Regel 1(b), beide Richtungen")
    print("=" * 92)
    alle_ok = True
    for name, opt, erwarteter_rand, erwartete_rate in faelle:
        def gefaelscht(arm, bedingung, seed, lr, threads, **kw):
            # glockenfoermig in log(lr), Maximum genau bei `opt`
            return {"balanced_accuracy": 1.0 / (1.0 + (_math.log10(lr / opt)) ** 2)}
        _zelle = gefaelscht
        try:
            best, rand = sweep_lr("periodisch", 1)
        finally:
            _zelle = echt
        gewaehlt = best["butterfly"]
        rate_ok = abs(_math.log10(gewaehlt / erwartete_rate)) < 1e-6
        rand_ok = rand["butterfly"] == erwarteter_rand
        ok = rate_ok and rand_ok
        alle_ok = alle_ok and ok
        print(f"  {name:<48} -> lr {gewaehlt:<10.6g} Rand {rand['butterfly']:<6} "
              f"{'OK' if ok else 'FEHLER (erwartet lr ' + format(erwartete_rate, '.6g') + ', Rand ' + erwarteter_rand + ')'}")
    print("=" * 92)
    print("TEST BESTANDEN" if alle_ok else "TEST FEHLGESCHLAGEN")
    print("=" * 92)
    return alle_ok


# ---------------------------------------------------------------------------
# Regel 2(b): ist der erwartete Effekt bei dieser Streuung ueberhaupt findbar?
# ---------------------------------------------------------------------------
def macht(threads: int = 2, seeds: Sequence[int] = (1971, 1972, 1973, 1974, 1975),
          lr: float = 0.003) -> Dict[str, object]:
    """Machtanalyse aus einem echten Vorlauf.

    Regel 2 hat seit heute zwei Teile. Teil (a) ist rechnerisch:
    2/2^15 = 6,1e-05, nach Holm ueber zwei Vergleiche 1,2e-04 -- Signifikanz
    ist erreichbar. Teil (b) verlangt mehr: Ist der ERWARTETE Effekt bei der
    VORHANDENEN Streuung nachweisbar? Dafuer braucht es eine Schaetzung der
    Streuung, und die kommt nur aus Daten.

    Der Entwurf in `ENTWURF_ZEITREIHEN.md` prueft nur Teil (a). B-007 ist genau
    an Teil (b) gescheitert, unbemerkt, drei Wochen lang: auf Fashion-MNIST war
    die Streuung viermal so gross wie auf MNIST, und die dort gemessenen
    Effekte lagen unter der Nachweisgrenze.

    Gerechnet wird die Nachweisgrenze bei 80 % Macht, alpha=0,05, gepaart.
    """
    print("=" * 92)
    print("MACHTANALYSE Q-019 -- Regel 2(b), Streuung aus einem echten Vorlauf")
    print(f"  {len(seeds)} Seeds, lr={lr} fest (nur zur Streuungsschaetzung)")
    print("=" * 92)
    daten: Dict[Tuple[str, str], List[float]] = {}
    for bed in BEDINGUNGEN:
        for arm in ("butterfly", "dense"):
            werte = []
            for s in seeds:
                werte.append(_zelle(arm, bed, s, lr, threads)["balanced_accuracy"])
            daten[(arm, bed)] = werte
            print(f"  {bed:<11} {arm:<10} {np.mean(werte):.4f} "
                  f"(sd {np.std(werte, ddof=1):.4f})")

    d_per = np.array(daten[("butterfly", "periodisch")]) - np.array(daten[("dense", "periodisch")])
    d_perm = np.array(daten[("butterfly", "permutiert")]) - np.array(daten[("dense", "permutiert")])
    wechsel = d_per - d_perm
    sd = float(np.std(wechsel, ddof=1))
    # Nachweisgrenze bei 80 % Macht, gepaart, einseitig alpha=0,05
    grenze = 2.49 * sd / math.sqrt(len(EVAL_SEEDS))
    print()
    print(f"  Abstand periodisch : {d_per.mean():+.4f}")
    print(f"  Abstand permutiert : {d_perm.mean():+.4f}")
    print(f"  Wechselwirkung     : {wechsel.mean():+.4f}  (sd {sd:.4f})")
    print(f"  Mit {len(EVAL_SEEDS)} Seeds nachweisbar ab: {grenze:+.4f}")
    reicht = bool(abs(wechsel.mean()) >= grenze)
    print(f"  -> {'reicht' if reicht else 'REICHT NICHT'}: der im Vorlauf "
          f"gesehene Effekt liegt {'ueber' if reicht else 'UNTER'} der Grenze.")
    if not reicht and sd > 0:
        noetig = int(math.ceil((2.49 * sd / max(abs(wechsel.mean()), 1e-9)) ** 2))
        print(f"  Fuer diesen Effekt waeren rund {noetig} Seeds noetig.")
    print("=" * 92)
    return {"sd_wechselwirkung": sd, "wechselwirkung": float(wechsel.mean()),
            "nachweisgrenze": grenze, "reicht": reicht,
            "delta_periodisch": float(d_per.mean()),
            "delta_permutiert": float(d_perm.mean())}


def saettigung(threads: int = 2, seeds: Sequence[int] = (1961, 1962),
               lr: float = 0.003) -> Dict[str, float]:
    """Darf kein Arm die Aufgabe fast perfekt loesen?

    Bei `fft_fest` ist das eine reale Gefahr: die Auslese liest Bits des
    Frequenzindex, und in der frequenzsortierten Basis fallen die drei
    hoechstwertigen Bits mit den acht Baendern zusammen. Loest ein Arm die
    Aufgabe mit ueber SAETTIGUNG, trennt der Aufbau nichts mehr -- dann muss
    die Schwierigkeit VOR dem Lauf erhoeht werden, und die Aenderung gehoert
    ins Tagebuch, bevor Zahlen entstehen.

    Zugleich die Gegenprobe nach unten: liegt ALLES auf Rateniveau, ist die
    Aufgabe zu schwer und der Lauf waere ein Nullbefund mit Ansage.
    """
    print("=" * 92)
    print("SAETTIGUNGSPRUEFUNG Q-019 -- vor dem Lauf")
    print(f"  Aufgabe: {ANTEILE} Anteile, Rauschen {RAUSCHEN}, "
          f"Amplitudenabstand {AMPLITUDENABSTAND}")
    print(f"  Rateniveau {1/KLASSEN:.3f} | Boden {BODEN:.3f} | "
          f"Saettigung {SAETTIGUNG:.2f}")
    print("=" * 92)
    ergebnis: Dict[str, float] = {}
    for bed in BEDINGUNGEN:
        for arm in ARME:
            werte = [_zelle(arm, bed, s, lr, threads)["balanced_accuracy"]
                     for s in seeds]
            m = float(np.mean(werte))
            ergebnis[f"{arm}|{bed}"] = m
            marke = ""
            if m > SAETTIGUNG:
                marke = "  <== GESAETTIGT"
            elif m < BODEN:
                marke = "  <== unter dem Boden"
            print(f"  {bed:<11} {arm:<10} {m:.4f}{marke}")
    gesaettigt = [k for k, v in ergebnis.items() if v > SAETTIGUNG]
    unten = [k for k, v in ergebnis.items() if v < BODEN]
    print()
    if gesaettigt:
        print(f"  GESAETTIGT: {', '.join(gesaettigt)}")
        print("  -> Schwierigkeit erhoehen (mehr ANTEILE, mehr RAUSCHEN, "
              "kleinerer AMPLITUDENABSTAND) und die Aenderung ins Tagebuch, "
              "BEVOR der Lauf startet.")
    if unten and len(unten) == len(ergebnis):
        print("  ALLES unter dem Boden -> Aufgabe zu schwer, der Lauf waere "
              "ein Nullbefund mit Ansage.")
    if not gesaettigt and len(unten) < len(ergebnis):
        print("  In Ordnung: kein Arm gesaettigt, mindestens einer ueber dem Boden.")
    print("=" * 92)
    return ergebnis


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else "lauf"
    if befehl == "vortest":
        ok = test_gitterverlaengerung()
        raise SystemExit(0 if (vortest() and ok) else 1)
    elif befehl == "fourier":
        fourier_erreichbarkeit()
    elif befehl == "gitter":
        raise SystemExit(0 if test_gitterverlaengerung() else 1)
    elif befehl == "saettigung":
        saettigung()
    elif befehl == "macht":
        macht()
    else:
        if not vortest():
            raise SystemExit("Vortest fehlgeschlagen -- kein Lauf.")
        recorder = run()
        try:
            auswertung = analyse(recorder)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
            print('  Nachholen mit: python -c "from quantum_sim.zeitreihen '
                  "import analyse; analyse('Q-019')\"")
            auswertung = {"error": repr(exc)}
        recorder.save(analysis=auswertung)
