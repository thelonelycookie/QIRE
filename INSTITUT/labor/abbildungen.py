"""Einheitlicher Abbildungsstil des Instituts (Plan R2.2): deutsch beschriftet (Dezimalkomma), ein Stil,
Export als PDF (Vektor, fuer die Arbeit) + kleines PNG (Vorschau, Dashboard), direkt aus Ergebnis-JSONs.

Funktionen:
  stil()                          -- setzt matplotlib-rcParams (idempotent)
  befund_balken(...)              -- Balken mit Konfidenzintervall, optional nach Urteil H1/H0/HU eingefaerbt
  kurven_ueber_seeds(...)         -- duenne Kurve je Seed + Mittel mit t-KI
  speichere(fig, name)            -- PDF + PNG (klein) nach INSTITUT/abbildungen/
  beispiele()                     -- erzeugt 3 Abbildungen aus vorhandenen Ergebnis-JSONs

Aufruf der Beispiele:  python3 -m labor.abbildungen   (aus INSTITUT/)  oder  python3 labor/abbildungen.py
Reine Darstellung: Zahlen kommen unveraendert aus den JSONs bzw. aus labor.statistik (KI), nichts wird
neu gemessen. Farben sind farbenblind-tauglich (Okabe-Ito) und tragen nie allein die Aussage (Beschriftung).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

if __package__ in (None, ""):                              # direkter Aufruf: INSTITUT in den Pfad
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labor import statistik as S

WURZEL = Path(__file__).resolve().parent.parent
ERGEBNISSE = WURZEL / "ergebnisse"
AUSGABE = WURZEL / "abbildungen"

# Okabe-Ito
FARBEN = {"blau": "#0072B2", "orange": "#E69F00", "gruen": "#009E73", "rot": "#D55E00",
          "himmel": "#56B4E9", "purpur": "#CC79A7", "grau": "#7F7F7F", "schwarz": "#222222"}
URTEIL_FARBE = {"H1": FARBEN["gruen"], "H0": FARBEN["rot"], "HU": FARBEN["grau"]}
URTEIL_NAME = {"H1": "H1 bestätigt", "H0": "H0 (widerlegt/gleichwertig)", "HU": "HU (unentschieden)"}


# ----------------------------------------------------------------- Stil und Zahlenformat
def komma(wert: float, stellen: int = 1) -> str:
    """Deutsche Zahl mit Dezimalkomma ('93,3')."""
    return f"{wert:.{stellen}f}".replace(".", ",").replace("-", "−")


def _achsenformat(stellen: int = 0):
    return FuncFormatter(lambda v, _: komma(v, stellen))


def stil():
    """Ein Stil fuer alle Abbildungen: serifenlose Schrift, klare Achsen, Raster dezent, Beschriftung gross
    genug fuer eine A4-Spalte; Dezimalkomma ueber _achsenformat/komma() (nicht ueber das Locale)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.unicode_minus": True,
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.prop_cycle": plt.cycler(color=list(FARBEN.values())[:7]),
    })


stil()


def speichere(fig, name: str, ordner: Optional[Path] = None, png_dpi: int = 110) -> dict:
    """Speichert PDF (Vektor) + PNG (klein, `png_dpi`) unter <ordner>/<name>.*; Rueckgabe {'pdf', 'png'}."""
    ordner = Path(ordner) if ordner else AUSGABE
    ordner.mkdir(parents=True, exist_ok=True)
    pdf, png = ordner / f"{name}.pdf", ordner / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=png_dpi)
    plt.close(fig)
    return {"pdf": pdf, "png": png}


# ----------------------------------------------------------------- Befund-Balken
def befund_balken(namen: Sequence[str], mittel: Sequence[float], ki: Sequence[tuple], urteile: Optional[Sequence[str]] = None,
                  titel: str = "", x_beschriftung: str = "Wert", referenz: Optional[float] = None,
                  referenz_name: str = "Referenz", stellen: int = 1, horizontal: bool = True):
    """Waagerechte Balken (Mittel) mit KI-Fehlerbalken; optional nach Urteil ('H1'/'H0'/'HU' oder ein
    Urteils-String, es zaehlt das erste Wort -- labor.statistik.urteil_etikett) eingefaerbt und mit Legende.
    ki: Liste (unten, oben) absolut (nicht als Abstand). referenz: senkrechte gestrichelte Linie (z. B. Null-
    Effekt oder Vergleichsarm). Beschriftung mit Dezimalkomma; der Mittelwert steht in der Achsenbeschriftung (kein Ueberlappen mit KI)."""
    stil()
    n = len(namen)
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * n + 1.4) if horizontal else (0.8 * n + 2.4, 3.6))
    farben = [FARBEN["blau"]] * n
    if urteile is not None:
        et = [S.urteil_etikett(u) for u in urteile]
        farben = [URTEIL_FARBE.get(e, FARBEN["himmel"]) for e in et]
    pos = np.arange(n)[::-1] if horizontal else np.arange(n)
    mittel = np.asarray(mittel, float)
    fehl = np.array([[m - k[0], k[1] - m] for m, k in zip(mittel, ki)]).T
    fehl = np.where(np.isfinite(fehl), fehl, 0.0)
    if horizontal:
        ax.barh(pos, mittel, color=farben, xerr=fehl, error_kw={"ecolor": FARBEN["schwarz"], "capsize": 3, "lw": 1},
                height=0.62)
        ax.set_yticks(pos, [f"{nm}  ({komma(m, stellen)})" for nm, m in zip(namen, mittel)])
        ax.set_xlabel(x_beschriftung)
        ax.xaxis.set_major_formatter(_achsenformat(stellen if stellen < 2 else 1))
        if referenz is not None:
            ax.axvline(referenz, color=FARBEN["schwarz"], ls="--", lw=1, label=referenz_name)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(pos, mittel, color=farben, yerr=fehl, error_kw={"ecolor": FARBEN["schwarz"], "capsize": 3, "lw": 1})
        ax.set_xticks(pos, namen, rotation=30, ha="right")
        ax.set_ylabel(x_beschriftung)
        ax.yaxis.set_major_formatter(_achsenformat(stellen if stellen < 2 else 1))
        if referenz is not None:
            ax.axhline(referenz, color=FARBEN["schwarz"], ls="--", lw=1, label=referenz_name)
        ax.grid(axis="x", visible=False)
    handles, marken = ax.get_legend_handles_labels()
    if urteile is not None:
        from matplotlib.patches import Patch
        for e in ("H1", "H0", "HU"):
            if e in et:
                handles.append(Patch(color=URTEIL_FARBE[e]))
                marken.append(URTEIL_NAME[e])
    if handles:
        ax.legend(handles, marken, loc="upper center", bbox_to_anchor=(0.5, -0.18 if horizontal else -0.4),
                  ncol=min(3, len(handles)), frameon=False)
    if titel:
        ax.set_title(titel, loc="left")
    return fig


# ----------------------------------------------------------------- Kurven ueber Seeds
def kurven_ueber_seeds(x: Sequence[float], werte: np.ndarray, titel: str = "", x_beschriftung: str = "x",
                       y_beschriftung: str = "Wert", logx: bool = False, mehrere: Optional[dict] = None, stellen: int = 0):
    """Kurve(n) ueber Seeds: duenne Linie je Seed (zeigt die Streuung, keine versteckte Mittelung) + dicke Linie
    Mittel mit 90-%-t-KI-Band (labor.statistik.t_ki). `werte` hat Form (Seeds, Stuetzstellen); `mehrere`
    {name: werte} zeichnet mehrere Arme (dann je Arm eine Farbe, Seed-Linien blasser, Legende)."""
    stil()
    serien = mehrere if mehrere is not None else {"": np.asarray(werte, float)}
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.asarray(x, float)
    for i, (name, w) in enumerate(serien.items()):
        w = np.asarray(w, float)
        farbe = list(FARBEN.values())[i % 7]
        for z in w:
            ax.plot(x, z, color=farbe, lw=0.6, alpha=0.25)
        m = w.mean(0)
        ki = np.array([S.t_ki(w[:, j]) for j in range(w.shape[1])])
        lo, hi = ki[:, 0], ki[:, 1]
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=farbe, alpha=0.18, lw=0)
        ax.plot(x, m, color=farbe, lw=2, marker="o", ms=3.5, label=name or f"Mittel, n = {w.shape[0]} Seeds")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x_beschriftung)
    ax.set_ylabel(y_beschriftung)
    ax.yaxis.set_major_formatter(_achsenformat(stellen))
    if titel:
        ax.set_title(titel, loc="left")
    ax.legend(frameon=False, loc="lower right")
    return fig


# ----------------------------------------------------------------- Beispiele aus Ergebnis-JSONs
def _neueste(muster: str) -> dict:
    pfade = sorted(ERGEBNISSE.glob(muster))
    if not pfade:
        raise FileNotFoundError(muster)
    return json.loads(pfade[-1].read_text())


def beispiele(ordner: Optional[Path] = None) -> list:
    """Drei Abbildungen aus vorhandenen Ergebnis-JSONs (nichts wird neu gerechnet):
    1. I-039: Testgenauigkeit je Arm (%), Mittel + 90-%-t-KI ueber 20 Seeds.
    2. C-clifford: Differenz zur vollen Ebene je Arm (Prozentpunkte) mit KI, Faerbung nach KI-Lage zu 0.
    3. I-008: Genauigkeit der Born-Messung gegen Schusszahl, eine Kurve je Seed (Ziffern)."""
    erg = []
    # 1
    z = _neueste("I-039__*.json")["zeilen"]
    arme = [a for a in z[0]["acc"] if a not in ("pk_haar", "nk", "f_index")]
    acc = {a: np.array([r["acc"][a] for r in z]) * 100 for a in arme}
    fig = befund_balken(arme, [acc[a].mean() for a in arme], [S.t_ki(acc[a]) for a in arme],
                        titel=f"I-039: Testgenauigkeit je Arm (Mittel, 90-%-KI, n = {len(z)} Seeds)",
                        x_beschriftung="Genauigkeit in %", stellen=1)
    erg.append(speichere(fig, "i039_genauigkeit", ordner))
    # 2
    z = _neueste("C-clifford__*.json")["zeilen"]
    ref, vergl = "voll", ["dicht_linear", "ohne_schaltung", "hadamard_fest", "zufall_clifford"]
    d = {a: (np.array([r[a] - r[ref] for r in z]) * 100) for a in vergl}
    ki = {a: S.t_ki(d[a]) for a in vergl}
    urteile = ["H1" if ki[a][0] > 0 else "H0" if ki[a][1] < 0 else "HU" for a in vergl]
    fig = befund_balken(vergl, [d[a].mean() for a in vergl], [ki[a] for a in vergl], urteile=urteile,
                        titel="C-clifford: Differenz zur vollen Ebene (Arm − voll)", x_beschriftung="Prozentpunkte",
                        referenz=0.0, referenz_name="kein Unterschied", stellen=1)
    erg.append(speichere(fig, "c_clifford_differenz", ordner))
    # 3
    z = [r for r in _neueste("I-008__*.json")["zeilen"] if r["datensatz"] == "ziffern"]
    schuesse = [10, 100, 1000, 10000]
    w = np.array([[r[f"born_S{s}"] for s in schuesse] for r in z]) * 100
    fig = kurven_ueber_seeds(schuesse, w, titel=f"I-008: Born-Messung auf Ziffern (n = {len(z)} Seeds)",
                             x_beschriftung="Schüsse je Bild", y_beschriftung="Genauigkeit in %", logx=True)
    erg.append(speichere(fig, "i008_schuesse", ordner))
    return erg


if __name__ == "__main__":
    for e in beispiele():
        print(e["pdf"], e["png"])
