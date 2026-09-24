from __future__ import annotations

"""Gemeinsames Messinstrument fuer die Rang-Untersuchungen.

WARUM EINE EIGENE DATEI
Die offenen Fragen -- kollabiert der Rang waehrend des Trainings, setzen die
Daten ihn fest, hat jede Klasse einen eigenen Unterraum, sagt die
Spektralentropie Kompression vorher, wie sieht die Bahn im Funktionenraum aus --
brauchen alle dieselbe Groesse: das Spektrum der AUSGABEN einer Schicht auf
echten Daten, gemessen an beliebigen Punkten im Training.

B-008 hat das Spektrum der Gewichtsmatrix gemessen und daraus einen falschen
Negativbefund gezogen (Rang 64 erfasst dort nur 65 % der Energie). Auf den
Ausgaben sind es 99,8 %. Diese Datei stellt sicher, dass ab jetzt ueberall
dieselbe, richtige Groesse gemessen wird.

WAS DRIN IST
  output_spectrum      kumulierte Energie und Effektivrang der Schichtausgaben
  spectral_entropy     Entropie des normierten Spektrums -- das klassische
                       Gegenstueck zur Verschraenkungsentropie, aber auf der
                       Datenverteilung statt auf der Gewichtsmatrix. Genau die
                       Groesse, die in Befund 2 der Gesamtbilanz nie gemessen
                       wurde.
  per_class_spectra    Effektivrang je Klasse und Hauptwinkel zwischen den
                       klassenweisen Unterraeumen. Beantwortet, ob die Schicht
                       in Wahrheit eingabeabhaengige Unterraeume benutzt --
                       also implizit das, was ein Mixture-of-Experts tun soll.
  function_probe       Ausgaben auf einem festen Pruefsatz, flach. Damit laesst
                       sich die Bahn durch den Funktionenraum verfolgen und
                       zeichnen.
  train_with_probe     Trainingsschleife wie in benchmark.train_model, aber mit
                       Haken nach jeder Epoche. Bewusst identisch parametriert
                       (Adam, gleiche Batch-Reihenfolge, gleiches Grad-Clipping),
                       damit die Ergebnisse mit allen frueheren Laeufen
                       vergleichbar bleiben.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import _batch_iter


# --------------------------------------------------------------------------
# Spektren
# --------------------------------------------------------------------------

def _singular_values(y: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Singulaerwerte der (zentrierten) Ausgabematrix, exakt gerechnet."""
    if center:
        y = y - y.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(y.double())


def cumulative_energy(s: torch.Tensor, ranks: Sequence[int]) -> Dict[int, float]:
    e = s ** 2
    total = float(e.sum())
    if total <= 0:
        return {r: float("nan") for r in ranks}
    cum = torch.cumsum(e, dim=0)
    return {r: float(cum[min(r, len(e)) - 1] / total) for r in ranks}


def effective_rank(s: torch.Tensor) -> float:
    """Teilnahmezahl (participation ratio) des Spektrums.

    (sum s^2)^2 / sum s^4 -- eine schwellenfreie Rangzahl. Bei r gleich grossen
    Werten ergibt sie genau r, bei einem dominanten Wert genau 1. Besser als
    "Rang bei 99 % Energie", weil keine willkuerliche Schwelle noetig ist.
    """
    e = (s ** 2).double()
    denom = float((e ** 2).sum())
    return float(e.sum() ** 2 / denom) if denom > 0 else float("nan")


def spectral_entropy(s: torch.Tensor) -> float:
    """Shannon-Entropie des normierten Energiespektrums, in Bit.

    Das ist die klassische Entsprechung der Verschraenkungsentropie -- nur auf
    der Verteilung der tatsaechlichen Aktivierungen statt auf der
    Gewichtsmatrix. exp2(dieser Wert) ist wieder eine Rangzahl.
    """
    e = (s ** 2).double()
    total = float(e.sum())
    if total <= 0:
        return float("nan")
    p = (e / total).clamp_min(1e-300)
    return float(-(p * torch.log2(p)).sum())


def output_spectrum(outputs: torch.Tensor, ranks: Sequence[int]) -> Dict[str, object]:
    s = _singular_values(outputs)
    out: Dict[str, object] = {
        "effective_rank": effective_rank(s),
        "spectral_entropy_bits": spectral_entropy(s),
        "entropy_rank": float(2 ** spectral_entropy(s)),
    }
    for r, v in cumulative_energy(s, ranks).items():
        out[f"energy_{r}"] = v
    return out


def middle_outputs(model: nn.Module, x: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
    """Ausgaben der Mittelschicht auf echten Daten."""
    model.eval()
    with torch.no_grad():
        h = torch.relu(model.fc_in(x[:max_rows]))
        return model.middles[0](h)


# --------------------------------------------------------------------------
# Klassenweise Unterraeume
# --------------------------------------------------------------------------

def _top_basis(y: torch.Tensor, k: int) -> torch.Tensor:
    """Orthonormale Basis der k staerksten Ausgaberichtungen."""
    y = y - y.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(y.double(), full_matrices=False)
    return Vh[:k].T                       # (dim, k)


def per_class_spectra(outputs: torch.Tensor, labels: torch.Tensor,
                      k: int = 16, min_rows: int = 32) -> Dict[str, float]:
    """Effektivrang je Klasse und Ueberlappung der klassenweisen Unterraeume.

    `mean_principal_cosine` nahe 1 heisst: alle Klassen benutzen denselben
    Unterraum. Nahe 0 heisst: jede Klasse hat ihren eigenen -- dann waere die
    Schicht faktisch eine eingabeabhaengige Auswahl von Unterraeumen, also
    genau das, was ein Mixture-of-Experts leisten soll.
    """
    classes = [int(c) for c in torch.unique(labels)]
    bases: List[torch.Tensor] = []
    ranks: List[float] = []
    for c in classes:
        rows = outputs[labels == c]
        if rows.shape[0] < min_rows:
            continue
        ranks.append(effective_rank(_singular_values(rows)))
        bases.append(_top_basis(rows, min(k, rows.shape[0] - 1)))
    if len(bases) < 2:
        return {"per_class_effective_rank": float("nan"),
                "mean_principal_cosine": float("nan"), "n_classes": len(bases)}
    cosines: List[float] = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            m = bases[i].T @ bases[j]
            cosines.append(float(torch.linalg.svdvals(m).mean()))
    return {"per_class_effective_rank": float(np.mean(ranks)),
            "mean_principal_cosine": float(np.mean(cosines)),
            "n_classes": len(bases)}


# --------------------------------------------------------------------------
# Bahn im Funktionenraum
# --------------------------------------------------------------------------

def function_probe(model: nn.Module, x_probe: torch.Tensor) -> np.ndarray:
    """Ausgaben des GESAMTEN Netzes auf einem festen Pruefsatz, flach.

    Zwei Modelle mit denselben Gewichten sind identisch; zwei Modelle mit
    verschiedenen Gewichten koennen trotzdem dieselbe Funktion berechnen. Fuer
    die Frage "nimmt das dichte Netz einen Umweg" ist deshalb die Funktion die
    richtige Groesse, nicht der Parametervektor.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_probe)
        return logits.flatten().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Training mit Haken
# --------------------------------------------------------------------------

def train_with_probe(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int = 256,
    seed: int = 0,
    grad_clip: float = 5.0,
    weight_decay: float = 0.0,
    on_epoch: Optional[Callable[[int, nn.Module], None]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[torch.optim.Optimizer, float]:
    """Wie `benchmark.train_model`, aber mit Haken nach jeder Epoche.

    Bewusst identisch parametriert (Adam, `_batch_iter` mit seed+epoch,
    Grad-Clipping 5.0), damit Zahlen aus dieser Schleife mit allen frueheren
    Experimenten vergleichbar bleiben.

    Der Optimierer kann uebergeben und zurueckgegeben werden -- noetig fuer
    Laeufe, die das Training unterbrechen, die Schicht ersetzen und dann
    weitertrainieren, ohne den Adam-Zustand zu verlieren.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        for xb, yb in _batch_iter(x_train, y_train, batch_size, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                break
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        if losses:
            best = min(best, float(np.mean(losses)))
        if on_epoch is not None:
            on_epoch(epoch, model)
    return optimizer, (best if math.isfinite(best) else float("nan"))
