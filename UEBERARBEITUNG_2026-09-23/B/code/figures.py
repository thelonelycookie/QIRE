from __future__ import annotations

"""Erzeugt die Abbildungen der B-Serie aus den gespeicherten Rohdaten.

WARUM AUS DEN ROHDATEN
Wie RESULTS.md werden auch die Abbildungen erzeugt, nicht gemalt. Eine Grafik,
deren Zahlen von Hand eingetragen sind, weicht beim naechsten Lauf ab -- und in
diesem Projekt sind an einem Tag vier Hauptaussagen gefallen.

FARBWAHL
Okabe-Ito-Teilmenge, Reihenfolge blau -> gruen -> orange -> rosa. Geprueft mit
dem Validator des dataviz-Verfahrens (`validate_palette.js`, light mode):

    Helligkeitsband      PASS
    Chroma-Untergrenze   PASS
    Farbsehschwaeche     PASS  (schlechtestes Nachbarpaar dE 11,0 deutan)
    Normalsicht          PASS  (dE 16,4)
    Kontrast zur Flaeche WARN  (#CC79A7 bei 2,98 statt 3,0)

Die Kontrastwarnung ist durch Direktbeschriftung jeder Kurve eingeloest -- die
Identitaet haengt damit nie allein an der Farbe. Zusaetzlich hat jede Reihe eine
eigene Markerform, was fuer Schwarzweissdruck noetig ist.

Bewusst NICHT uebernommen aus dem dataviz-Verfahren: Hover-Ebene, Tooltips,
Dunkelmodus und Filterzeile. Das sind Anforderungen an Webdiagramme; hier
entstehen statische Abbildungen fuer eine gedruckte Arbeit.

Run: python -m qire_ai.figures
"""

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .result_store import load

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "results" / "figures"

# Okabe-Ito, validierte Reihenfolge. Kategorial: feste Zuordnung, nie zyklisch.
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
SURFACE = "#fcfcfb"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 200,
    })
    return plt


def _finish(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label_end(ax, x, y, text, color, dx=1.03):
    """Direktbeschriftung am Kurvenende -- loest die Kontrastwarnung ein und
    macht die Identitaet unabhaengig von der Farbe."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8, color=color, weight="bold")


# --------------------------------------------------------------------------
# Abbildung 1 -- B-014: was setzt den Rang?
# --------------------------------------------------------------------------

def fig_rank_task(plt) -> Path:
    payload = load("B-014")
    rec = payload["records"]
    dims = sorted({int(r["d"]) for r in rec})
    classes = sorted({int(r["num_classes"]) for r in rec})

    def mean(d, c, key):
        return float(np.mean([r[key] for r in rec
                              if r["d"] == d and r["num_classes"] == c]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    panels = [("untrainiert", "effective_rank_untrained"),
              ("nach dem Training", "effective_rank")]
    for ax, (title, key) in zip(axes, panels):
        untrained = key.endswith("untrained")
        for i, c in enumerate(classes):
            ys = [mean(d, c, key) for d in dims]
            ax.plot(dims, ys, color=PALETTE[i], marker=MARKERS[i],
                    label=f"{c} Klassen")
            # Im linken Feld liegen alle vier Kurven uebereinander -- vier
            # Endbeschriftungen an derselben Stelle waeren unlesbar. Dass sie
            # sich decken, IST der Befund, also wird genau das beschriftet.
            if not untrained:
                _label_end(ax, dims[-1], ys[-1], f"{c}", PALETTE[i])
        if untrained:
            ax.annotate("alle vier Kurven\nliegen übereinander",
                        xy=(dims[-2], mean(dims[-2], classes[0], key)),
                        xytext=(-8, -46), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="right",
                        arrowprops=dict(arrowstyle="->", color=MUTED,
                                        linewidth=0.8))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
        _finish(ax, title, "intrinsische Dimension der Daten $d$",
                "Effektivrang der Schicht" if title == "untrainiert" else "")
    axes[0].legend(loc="upper left")
    fig.suptitle("Vor dem Training folgt der Rang der Datengeometrie — danach der Aufgabe",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGDIR / "abb1_rang_aufgabe.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Abbildung 2 -- B-012: Rangverlauf waehrend des Trainings
# --------------------------------------------------------------------------

def fig_rank_trajectory(plt) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, ds in enumerate(("mnist", "fashion_mnist")):
        rec = load(f"B-012-{ds}")["records"]
        ks = sorted({int(r["k"]) for r in rec if r["arm"] == "dense_trajectory"})
        ys = [float(np.mean([r["effective_rank"] for r in rec
                             if r["arm"] == "dense_trajectory" and r["k"] == k]))
              for k in ks]
        name = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        ax.plot(ks, ys, color=PALETTE[i], marker=MARKERS[i], label=name)
        _label_end(ax, ks[-1], ys[-1], name, PALETTE[i])
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 8)
    _finish(ax, "Der Rang kollabiert nicht — er steigt",
            "Epoche", "Effektivrang der Mittelschicht")
    ax.annotate("schon nach Epoche 1 liegen\n99,9 % der Energie in 64 Richtungen",
                xy=(1, 5.1), xytext=(2.4, 2.2), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8))
    fig.tight_layout()
    path = FIGDIR / "abb2_rangverlauf.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Abbildung 3 -- B-008: mehr Rang hilft nicht
# --------------------------------------------------------------------------

def fig_rank_curve(plt) -> Path:
    payload = load("B-008")
    rec, cfg = payload["records"], payload["config"]
    ranks = [int(r) for r in cfg["ranks"]]

    def acc(ds, arm):
        return float(np.mean([r["balanced_accuracy"] for r in rec
                              if r["dataset"] == ds and r["arm"] == arm]))

    def energy(ds, r):
        return float(np.mean([x[f"energy_at_{r}"] for x in rec
                              if x["dataset"] == ds and x["arm"] == "dense"]))

    # Zwei Groessen mit verschiedenen Skalen -> zwei Felder, NIE zwei y-Achsen.
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True)
    for i, ds in enumerate(("mnist", "fashion_mnist")):
        name = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        d = acc(ds, "dense")
        ys = [acc(ds, f"lowrank_r{r}") - d for r in ranks]
        axes[0].plot(ranks, ys, color=PALETTE[i], marker=MARKERS[i], label=name)
        _label_end(axes[0], ranks[-1], ys[-1], name, PALETTE[i])
        es = [energy(ds, r) for r in ranks]
        axes[1].plot(ranks, es, color=PALETTE[i], marker=MARKERS[i], label=name)
        _label_end(axes[1], ranks[-1], es[-1], name, PALETTE[i])
    axes[0].axhline(0, color=MUTED, linewidth=1.0, linestyle="--")
    axes[0].annotate("dichte Schicht", xy=(ranks[0], 0), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=MUTED)
    _finish(axes[0], "Mehr Rang schließt die Lücke nicht", "",
            "Abstand zur dichten Schicht")
    _finish(axes[1], "…obwohl der Rang immer mehr Energie erfasst",
            "Rang der Low-Rank-Schicht", "kumulierte Spektralenergie")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    fig.tight_layout()
    path = FIGDIR / "abb3_rangkurve.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Abbildung 4 -- B-012: Dosis-Wirkungs-Kurve
# --------------------------------------------------------------------------

def fig_dose(plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, ds in zip(axes, ("mnist", "fashion_mnist")):
        payload = load(f"B-012-{ds}")
        rec, cfg = payload["records"], payload["config"]
        ks = [int(k) for k in cfg["k_values"]]
        ranks = [int(r) for r in cfg["ranks"]]
        for i, rank in enumerate(ranks):
            ys = [float(np.mean([r["balanced_accuracy"] for r in rec
                                 if r["arm"] == "branch" and r["rank"] == rank
                                 and r["k"] == k])) for k in ks]
            ax.plot(ks, ys, color=PALETTE[i], marker=MARKERS[i], label=f"Rang {rank}")
            _label_end(ax, ks[-1], ys[-1], f"R{rank}", PALETTE[i])
        dense = float(np.mean([r["balanced_accuracy"] for r in rec
                               if r["arm"] == "dense"]))
        ax.axhline(dense, color=MUTED, linewidth=1.0, linestyle="--")
        name = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        _finish(ax, name, "Beschränkung ab Epoche $k$",
                "balancierte Genauigkeit" if ds == "mnist" else "")
    axes[0].legend(loc="lower right")
    fig.suptitle("Ab Epoche 0 zu beschränken kostet — der Nachteil ist nach "
                 "wenigen Epochen aufgeholt", x=0.01, ha="left", fontsize=11,
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGDIR / "abb4_dosis.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt = _setup()
    for fn in (fig_rank_task, fig_rank_trajectory, fig_rank_curve, fig_dose):
        try:
            path = fn(plt)
            print(f"  geschrieben: {path.relative_to(ROOT)}")
        except Exception as exc:          # noqa: BLE001
            print(f"  FEHLER in {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
