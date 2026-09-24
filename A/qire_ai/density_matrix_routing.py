from __future__ import annotations

"""RESEARCH ID: C-001 -- Von-Neumann-Entropie steuert den Rechenaufwand pro Beispiel.

QUANTENPRINZIP
Ein gemischter Zustand wird durch eine Dichtematrix rho = sum_i p_i |psi_i><psi_i|
beschrieben. Ihre von-Neumann-Entropie S(rho) = -tr(rho log rho) misst, wie
"unrein" der Zustand ist: S = 0 heisst reiner Zustand (das System ist in einem
eindeutigen Zustand), hohes S heisst echte Mischung konkurrierender Zustaende.
Die Messung reduziert den Zustand -- Information wird kontrolliert verworfen.

WARUM DAS HIER KEINE UMBENENNUNG IST
Die Schicht erzeugt K Kandidaten-Repraesentationen v_1..v_K mit Gewichten
p_1..p_K. Daraus wird rho = sum_i p_i v_i v_i^T gebildet (v_i normiert). Der
entscheidende Punkt: S(rho) haengt davon ab, wie stark die Kandidaten
*ueberlappen*, nicht nur von den Gewichten.

  Zwei Kandidaten mit p = (0.5, 0.5) und identischen Vektoren
      -> rho ist rein, S = 0  (das Netz ist sich einig, nur der Gate schwankt)
  Zwei Kandidaten mit p = (0.5, 0.5) und orthogonalen Vektoren
      -> rho maximal gemischt, S = log 2  (echte Uneinigkeit)

Max-Softmax-Konfidenz kann diese beiden Faelle nicht unterscheiden -- sie sieht
nur p. Genau das ist die falsifizierbare Behauptung.

EFFIZIENZMECHANISMUS
Beispiele mit niedrigem S sind eindeutig und werden mit dem billigen Pfad
abgeschlossen; nur bei hohem S laeuft der teure Pfad. Das adressiert den
Rechenaufwand *pro Eingabe* -- die einzige Ressourcenachse, die in diesem
Projekt bisher nie optimiert wurde, und genau die Luecke, die der Befund
"energy_aware_loss ist wirkungslos, weil top_k ein fester Hyperparameter ohne
Gradientenpfad ist" offengelegt hat.

RECHENAUFWAND DER ENTROPIE SELBST
rho ist d x d, aber hoechstens Rang K. Mit A = [sqrt(p_i) v_i] (d x K) gilt
rho = A A^T, und A^T A ist nur K x K mit denselben Eigenwerten ungleich null.
Die Entropie kostet daher O(K^3), nicht O(d^3) -- bei K = 4 vernachlaessigbar.

BASELINES (alle bei identischer mittlerer Rechenlast)
  always_full   kein adaptives Verhalten, immer teurer Pfad (Obergrenze)
  entropy       von-Neumann-Entropie entscheidet
  confidence    Max-Softmax des Gates entscheidet (klassische Kontrolle)
  random        zufaellig bei gleicher Rate (Sanity-Check: hilft Adaptivitaet
                ueberhaupt, oder ist der teure Pfad einfach selten noetig?)

Die Schwellen werden je Arm auf dem Trainingssatz so kalibriert, dass alle
dieselbe Ausstiegsrate erreichen. Ohne das waeren die Arme nicht vergleichbar.

HYPOTHESEN
  H1 (Haupt) : entropy > confidence bei gleicher mittlerer Rechenlast
  H0 (Null)  : kein Unterschied -- die Dichtematrix-Formulierung liefert nichts,
               was Gate-Konfidenz nicht auch liefert
  H2 (Alt)   : beide ~ random -- die Auswahl ist egal, nur die Rate zaehlt

FALSIFIKATION
Schlaegt entropy die Konfidenz-Kontrolle bei gleicher Rechenlast nicht, ist der
Dichtematrix-Rahmen fuer diesen Zweck ohne Mehrwert.

Run: python -m qire_ai.density_matrix_routing
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import set_seed
from .data_loader import load_dataset


def von_neumann_entropy(candidates: torch.Tensor, probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """S(rho) for rho = sum_i p_i v_i v_i^T, batched.

    candidates: (batch, K, d), probs: (batch, K) -> (batch,)
    Uses that A^T A (K x K) shares the nonzero spectrum of rho = A A^T.
    """
    v = F.normalize(candidates, dim=-1)
    a = v * probs.clamp_min(0).sqrt().unsqueeze(-1)      # (batch, K, d)
    gram = torch.bmm(a, a.transpose(1, 2))               # (batch, K, K)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(eps)
    eigenvalues = eigenvalues / eigenvalues.sum(-1, keepdim=True)
    return -(eigenvalues * eigenvalues.log()).sum(-1)


class AdaptiveNet(nn.Module):
    """Cheap trunk -> K candidates + gate -> either cheap head (early exit) or
    expensive refinement. Which samples take the expensive path is decided by
    the chosen signal at evaluation time; during training both paths run, so all
    arms share identical weights and differ only in the routing signal."""

    model_name = "AdaptiveNet"

    def __init__(self, input_dim: int, trunk: int = 512, num_candidates: int = 4,
                 expensive: int = 2048, num_classes: int = 10):
        super().__init__()
        self.K = num_candidates
        self.trunk = nn.Linear(input_dim, trunk)
        self.candidates = nn.Linear(trunk, num_candidates * trunk)
        self.gate = nn.Linear(trunk, num_candidates)
        self.cheap_head = nn.Linear(trunk, num_classes)
        self.expensive = nn.Sequential(nn.Linear(trunk, expensive), nn.ReLU(),
                                       nn.Linear(expensive, num_classes))
        self.trunk_dim = trunk

    def _shared(self, x):
        h = torch.relu(self.trunk(x))
        cands = self.candidates(h).reshape(x.shape[0], self.K, self.trunk_dim)
        probs = torch.softmax(self.gate(h), dim=-1)
        mixed = (cands * probs.unsqueeze(-1)).sum(1)
        return h, cands, probs, mixed

    def forward(self, x):
        """Training path: both heads active, loss combines them."""
        h, cands, probs, mixed = self._shared(x)
        cheap_logits = self.cheap_head(torch.relu(mixed))
        expensive_logits = self.expensive(h)
        return cheap_logits, expensive_logits, cands, probs

    def signals(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (entropy, one_minus_confidence). Both are 'uncertainty':
        higher means the expensive path is more warranted."""
        _, cands, probs, _ = self._shared(x)
        entropy = von_neumann_entropy(cands, probs)
        confidence = probs.max(dim=-1).values
        return entropy, 1.0 - confidence

    @torch.no_grad()
    def evaluate_adaptive(self, x, y, uncertainty: torch.Tensor, threshold: float) -> Tuple[float, float]:
        """Samples above threshold take the expensive path. Returns
        (balanced-ish accuracy, fraction routed to the expensive path)."""
        cheap_logits, expensive_logits, _, _ = self.forward(x)
        use_expensive = uncertainty > threshold
        logits = torch.where(use_expensive.unsqueeze(-1), expensive_logits, cheap_logits)
        correct = (logits.argmax(-1) == y).float().mean().item()
        return correct, use_expensive.float().mean().item()


def train(model, x, y, epochs: int, lr: float, batch_size: int, seed: int):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            cheap_logits, expensive_logits, _, _ = model(x[idx])
            # both heads trained; the cheap one must be usable on its own
            loss = 0.5 * F.cross_entropy(cheap_logits, y[idx]) + \
                   0.5 * F.cross_entropy(expensive_logits, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def calibrate(uncertainty: torch.Tensor, target_rate: float) -> float:
    """Threshold such that `target_rate` of samples exceed it."""
    return float(torch.quantile(uncertainty, 1.0 - target_rate))


def run(
    seeds: Sequence[int] = tuple(range(1101, 1111)),
    target_rates: Sequence[float] = (0.25, 0.50),
    max_samples: int = 8000,
    epochs: int = 15,
    lr: float = 0.01,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, object]:
    results: Dict[tuple, List[float]] = {}
    rates: Dict[tuple, List[float]] = {}

    for seed in seeds:
        set_seed(seed)
        torch.set_num_threads(threads)
        bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
        model = AdaptiveNet(bundle.input_dim, num_classes=bundle.num_classes)
        train(model, bundle.x_train, bundle.y_train, epochs, lr, batch_size, seed)

        model.eval()
        with torch.no_grad():
            ent_train, conf_train = model.signals(bundle.x_train)
            ent_test, conf_test = model.signals(bundle.x_test)
            rand_train = torch.rand(bundle.x_train.shape[0], generator=torch.Generator().manual_seed(seed))
            rand_test = torch.rand(bundle.x_test.shape[0], generator=torch.Generator().manual_seed(seed + 1))

            full_acc, _ = model.evaluate_adaptive(bundle.x_test, bundle.y_test,
                                                  torch.full_like(ent_test, 1e9), 0.0)
            results.setdefault(("always_full", 1.0), []).append(full_acc)
            rates.setdefault(("always_full", 1.0), []).append(1.0)

            for rate in target_rates:
                for name, train_sig, test_sig in (("entropy", ent_train, ent_test),
                                                  ("confidence", conf_train, conf_test),
                                                  ("random", rand_train, rand_test)):
                    tau = calibrate(train_sig, rate)
                    acc, actual = model.evaluate_adaptive(bundle.x_test, bundle.y_test, test_sig, tau)
                    results.setdefault((name, rate), []).append(acc)
                    rates.setdefault((name, rate), []).append(actual)
        print(f"  seed {seed} done")
    return {"acc": results, "rates": rates, "target_rates": list(target_rates)}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[tuple, List[float]] = result["acc"]      # type: ignore[assignment]
    rates: Dict[tuple, List[float]] = result["rates"]  # type: ignore[assignment]
    n = len(acc[("always_full", 1.0)])

    print()
    print(f"RESEARCH ID C-001 | MNIST, {n} Seeds")
    full = np.array(acc[("always_full", 1.0)])
    print(f'  always_full (100% teurer Pfad): {full.mean():.4f} +/- {full.std(ddof=1):.4f}')
    print()
    header = f'{"Rate":>6} {"Signal":>12} {"accuracy":>9} {"std":>7} {"tats. Rate":>11}'
    print(header); print("-" * len(header))
    for rate in result["target_rates"]:          # type: ignore[index]
        for name in ("entropy", "confidence", "random"):
            vals = np.array(acc[(name, rate)])
            actual = np.mean(rates[(name, rate)])
            print(f'{rate:>6.2f} {name:>12} {vals.mean():>9.4f} {vals.std(ddof=1):>7.4f} {actual:>11.3f}')

    print()
    print("ENTSCHEIDEND -- Entropie gegen Konfidenz bei gleicher Rechenlast:")
    verdict_ok = True
    for rate in result["target_rates"]:          # type: ignore[index]
        e = np.array(acc[("entropy", rate)])
        c = np.array(acc[("confidence", rate)])
        r = np.array(acc[("random", rate)])
        p_ec = stats.wilcoxon(e, c).pvalue
        p_er = stats.wilcoxon(e, r).pvalue
        if not (e.mean() > c.mean() and p_ec < 0.05):
            verdict_ok = False
        print(f'  Rate {rate:.2f}: Entropie-Konfidenz {e.mean()-c.mean():+.4f} (p={p_ec:.4f})'
              + (" *signifikant*" if p_ec < 0.05 else " (n.s.)")
              + f' | Entropie-Zufall {e.mean()-r.mean():+.4f} (p={p_er:.4f})')

    print()
    if verdict_ok:
        print("  -> Entropie schlaegt Konfidenz bei gleicher Rechenlast.")
        print("     Die Dichtematrix erfasst Uneinigkeit, die der Gate allein nicht sieht.")
    else:
        print("  -> Entropie schlaegt Konfidenz nicht.")
        print("     Der Dichtematrix-Rahmen liefert hier keinen Mehrwert gegenueber")
        print("     einfacher Gate-Konfidenz -- H0 bestaetigt.")


if __name__ == "__main__":
    analyse(run())
