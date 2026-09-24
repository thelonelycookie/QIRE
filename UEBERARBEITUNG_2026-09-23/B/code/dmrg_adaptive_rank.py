from __future__ import annotations

"""RESEARCH ID: B-015 -- DMRG-artige adaptive Rangzuweisung waehrend des Trainings.

UMBENANNT am 4. August. Dieser Lauf hiess urspruenglich B-003. Dieselbe Kennung
wurde am 3. August ein zweites Mal fuer den Blockschicht-Vergleich vergeben
(Seeds ab 1101). Entschieden wurde zugunsten des Blockschicht-Laufs, weil unter
`results/experiments/B-003__*` dessen Rohdaten liegen und `qire_ai.audit` gegen
genau diese Datei prueft. Dieser Lauf hier hat keine gespeicherten Rohdaten und
konnte deshalb folgenlos umbenannt werden. Wer aeltere Notizen liest: „B-003"
vor dem 3. August meint diesen Lauf, danach den Blockschicht-Vergleich.

QUANTENPRINZIP
DMRG (Density Matrix Renormalization Group, White 1992) ist ein echter
Quanten-Vielteilchen-Algorithmus. Sein Kern: die Bond-Dimension wird nicht
vorab festgelegt, sondern *waehrend* der Optimierung angepasst -- anhand des
verworfenen Spektralgewichts an jedem Bond. Wo viel Gewicht verloren ginge,
waechst die Bond-Dimension; wo kaum etwas verloren geht, bleibt sie klein.
Kapazitaet wandert also dorthin, wo die Struktur sie verlangt.

WARUM DAS EINE ANDERE FRAGE IST ALS A-001 BIS B-002
Alle bisherigen Tests verglichen *feste* Strukturen miteinander. Hier geht es
um den Zuweisungs*prozess*: Kann ein spektrales Kriterium Kapazitaet besser
verteilen als eine gleichmaessige Vorabwahl? Die Zielachse ist ausserdem eine
neue -- **Trainingsaufwand**, die einzige Ressource, die in diesem Projekt nie
optimiert wurde. Ein Modell, das klein startet und nur dort waechst, wo noetig,
verbraucht ueber den gesamten Trainingslauf weniger Rechenzeit als eines, das
von Anfang an auf Endgroesse trainiert.

KLASSISCHE UMSETZUNG
Jede komprimierte Schicht ist W = U V mit Rang r. Alle Schichten starten bei
r = 1. Periodisch wird fuer jede Schicht das Spektrum von W = U V bestimmt und
geprueft, wieviel Gewicht die *letzte noch vorhandene* Richtung traegt:

    saturation_l = sigma_r^2 / sum_i sigma_i^2

Ist dieser Anteil gross, traegt die aeusserste Richtung noch echte Information
-- der Rang ist vermutlich zu knapp und wird um 1 erhoeht. Das ist die auf das
Beobachtbare uebertragene Discarded-Weight-Logik: bei einer bereits
faktorisierten Matrix laesst sich das Verworfene nicht direkt messen, wohl aber,
ob die letzte behaltene Richtung noch gesaettigt ist.

BASELINES
  fixed_uniform : alle Schichten von Anfang an auf dem End-Rang (Standardwahl)
  dmrg_adaptive : Start bei Rang 1, Wachstum nach obigem Kriterium
  random_growth : Start bei Rang 1, Wachstum nach identischem Zeitplan, aber
                  die wachsende Schicht wird ZUFAELLIG gewaehlt

`random_growth` ist die Abgrenzungskontrolle. Sie hat denselben
Trainingsaufwand und dasselbe Endbudget wie `dmrg_adaptive` und unterscheidet
sich nur darin, dass das spektrale Kriterium durch Zufall ersetzt ist. Bringt
sie dasselbe, traegt nicht das Kriterium, sondern lediglich das Wachsen an sich.

HYPOTHESEN
  H1 (Haupt) : dmrg_adaptive erreicht die Genauigkeit von fixed_uniform bei
               geringerem Trainingsaufwand UND schlaegt random_growth
  H0 (Null)  : dmrg_adaptive ~ random_growth -- das Kriterium ist irrelevant
  H2 (Alt)   : beide Wachstumsvarianten schlechter als fixed_uniform -- spaet
               hinzugefuegte Kapazitaet wird nicht mehr sinnvoll genutzt

FALSIFIKATION
Schlaegt dmrg_adaptive die Zufallskontrolle bei gleichem Budget und Aufwand
nicht, liefert das DMRG-Kriterium keinen Beitrag.

Run: python -m qire_ai.dmrg_adaptive_rank
"""

import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import set_seed
from .data_loader import load_dataset


class GrowableLowRank(nn.Module):
    """W = U V mit wachsendem Rang. Neue Richtungen werden klein initialisiert,
    damit das bisher Gelernte beim Wachsen nicht gestoert wird."""

    def __init__(self, dim: int, rank: int, max_rank: int):
        super().__init__()
        self.dim, self.max_rank = dim, max_rank
        self.u = nn.Parameter(torch.empty(dim, rank))
        self.v = nn.Parameter(torch.empty(rank, dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
        self.bias = nn.Parameter(torch.zeros(dim))

    @property
    def rank(self) -> int:
        return self.u.shape[1]

    def forward(self, x):
        return (x @ self.v.t()) @ self.u.t() + self.bias

    @torch.no_grad()
    def saturation(self) -> float:
        """Wie stark traegt die aeusserste noch vorhandene Richtung?

        W = U V hat Rang <= r, also sind von den d Singulaerwerten nur die
        ersten r ungleich null. Betrachtet wird deshalb ausdruecklich sv[r-1]
        und nicht sv[-1] -- letzteres waere immer ~0 und das Mass wertlos
        (dieser Fehler ist im Smoke-Test aufgefallen).

        Zurueckgegeben wird sigma_r / sigma_1: ein skalenfreies Mass dafuer, ob
        das Spektrum innerhalb der behaltenen Richtungen bereits abgefallen ist.
        Nahe 1 heisst, es faellt noch gar nicht ab -- der Rang ist vermutlich zu
        knapp. Nahe 0 heisst, die aeusserste Richtung traegt kaum noch etwas.
        """
        w = self.u @ self.v
        sv = torch.linalg.svdvals(w)[: self.rank]
        return float(sv[-1] / sv[0].clamp_min(1e-12))

    @torch.no_grad()
    def grow(self, amount: int = 1) -> bool:
        if self.rank + amount > self.max_rank:
            return False
        scale = 0.01
        new_u = torch.cat([self.u, torch.randn(self.dim, amount) * scale], dim=1)
        new_v = torch.cat([self.v, torch.randn(amount, self.dim) * scale], dim=0)
        self.u = nn.Parameter(new_u)
        self.v = nn.Parameter(new_v)
        return True


class GrowNet(nn.Module):
    def __init__(self, input_dim: int, width: int, num_classes: int,
                 depth: int, start_rank: int, max_rank: int):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(
            [GrowableLowRank(width, start_rank, max_rank) for _ in range(depth)]
        )
        self.fc_out = nn.Linear(width, num_classes)

    def forward(self, x):
        h = torch.relu(self.fc_in(x))
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.fc_out(h)

    def ranks(self) -> List[int]:
        return [l.rank for l in self.layers]

    def compressed_parameters(self) -> int:
        return sum(p.numel() for l in self.layers for p in l.parameters())


def train_with_growth(model: GrowNet, x, y, epochs: int, lr: float, batch_size: int,
                      seed: int, mode: str, total_budget: int, grow_every: int) -> Dict[str, float]:
    """mode: 'fixed' | 'dmrg' | 'random'. Gibt Trainingszeit und einen
    Aufwands-Proxy zurueck (kumulierte Parameterzahl ueber alle Schritte)."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    cumulative_params = 0.0
    step = 0
    start = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            cumulative_params += model.compressed_parameters()
            step += 1

            if mode != "fixed" and step % grow_every == 0 and sum(model.ranks()) < total_budget:
                if mode == "dmrg":
                    sat = [l.saturation() for l in model.layers]
                    target = int(np.argmax(sat))
                else:
                    target = int(rng.integers(len(model.layers)))
                if model.layers[target].grow(1):
                    # Optimizer neu aufsetzen, da sich die Parametertensoren geaendert haben
                    opt = torch.optim.Adam(model.parameters(), lr=lr)

    return {
        "train_seconds": time.perf_counter() - start,
        "cumulative_params": cumulative_params / max(1, step),
    }


ARMS = ["fixed_uniform", "dmrg_adaptive", "random_growth"]


def run(
    width: int = 512,
    depth: int = 3,
    final_rank_each: int = 8,
    start_rank: int = 3,
    seeds: Sequence[int] = tuple(range(1201, 1211)),
    max_samples: int = 6000,
    epochs: int = 15,
    lr: float = 0.003,
    batch_size: int = 256,
    threads: int = 6,
) -> Dict[str, object]:
    total_budget = final_rank_each * depth
    acc: Dict[str, List[float]] = {}
    cost: Dict[str, List[float]] = {}
    secs: Dict[str, List[float]] = {}
    final_ranks: Dict[str, List[List[int]]] = {}

    # Wachstumstakt so, dass das Budget gegen Ende des Trainings erreicht wird
    steps_total = epochs * int(np.ceil(max_samples * 0.7 / batch_size))
    # Start bei start_rank je Schicht, nicht bei 1: A-003 hat gezeigt, dass
    # gestapelte Rang-1-Schichten das Signal vollstaendig zerstoeren -- dort zu
    # starten wuerde "Wachstumsstrategie" mit "zu klein zum Trainieren" vermengen.
    grow_steps = total_budget - depth * start_rank
    grow_every = max(1, int(steps_total * 0.7 / max(1, grow_steps)))

    for seed in seeds:
        for arm in ARMS:
            set_seed(seed)
            torch.set_num_threads(threads)
            bundle = load_dataset("mnist", seed=seed, max_samples=max_samples)
            initial = final_rank_each if arm == "fixed_uniform" else start_rank
            model = GrowNet(bundle.input_dim, width, bundle.num_classes,
                            depth, initial, max_rank=total_budget)
            mode = {"fixed_uniform": "fixed", "dmrg_adaptive": "dmrg",
                    "random_growth": "random"}[arm]
            info = train_with_growth(model, bundle.x_train, bundle.y_train, epochs, lr,
                                     batch_size, seed, mode, total_budget, grow_every)
            model.eval()
            with torch.no_grad():
                pred = model(bundle.x_test).argmax(-1)
                a = (pred == bundle.y_test).float().mean().item()
            acc.setdefault(arm, []).append(a)
            cost.setdefault(arm, []).append(info["cumulative_params"])
            secs.setdefault(arm, []).append(info["train_seconds"])
            final_ranks.setdefault(arm, []).append(model.ranks())
        print(f"  seed {seed} done")
    return {"acc": acc, "cost": cost, "secs": secs, "ranks": final_ranks,
            "budget": total_budget}


def analyse(result: Dict[str, object]) -> None:
    from scipy import stats

    acc: Dict[str, List[float]] = result["acc"]     # type: ignore[assignment]
    cost: Dict[str, List[float]] = result["cost"]   # type: ignore[assignment]
    secs: Dict[str, List[float]] = result["secs"]   # type: ignore[assignment]
    ranks: Dict[str, List[List[int]]] = result["ranks"]  # type: ignore[assignment]
    n = len(acc["fixed_uniform"])

    print()
    print(f"RESEARCH ID B-015 | MNIST, {n} Seeds, Rangbudget {result['budget']}")
    header = (f'{"arm":>16} {"accuracy":>9} {"std":>7} {"mittl. Param":>13} '
              f'{"Trainingszeit":>14} {"End-Raenge":>16}')
    print(header); print("-" * len(header))
    for arm in ARMS:
        a = np.array(acc[arm]); c = np.array(cost[arm]); s = np.array(secs[arm])
        mean_ranks = np.mean(np.array(ranks[arm]), axis=0)
        rank_str = "/".join(f"{r:.1f}" for r in mean_ranks)
        print(f'{arm:>16} {a.mean():>9.4f} {a.std(ddof=1):>7.4f} {c.mean():>13,.0f} '
              f'{s.mean():>13.1f}s {rank_str:>16}')

    fixed = np.array(acc["fixed_uniform"])
    dmrg = np.array(acc["dmrg_adaptive"])
    rand = np.array(acc["random_growth"])

    print()
    print("Trainingsaufwand (mittlere Parameterzahl ueber alle Schritte):")
    for arm in ARMS:
        c = np.array(cost[arm])
        base = np.array(cost["fixed_uniform"]).mean()
        print(f'  {arm:>16}: {c.mean():>10,.0f}  ({100*c.mean()/base:>5.1f}% von fixed)')

    print()
    print("ENTSCHEIDEND -- traegt das DMRG-Kriterium oder nur das Wachsen?")
    p_dr = stats.wilcoxon(dmrg, rand).pvalue
    p_df = stats.wilcoxon(dmrg, fixed).pvalue
    print(f'  dmrg vs random: {dmrg.mean()-rand.mean():+.4f}, p={p_dr:.4f}'
          + (" *signifikant*" if p_dr < 0.05 else " (nicht signifikant)"))
    print(f'  dmrg vs fixed : {dmrg.mean()-fixed.mean():+.4f}, p={p_df:.4f}'
          + (" *signifikant*" if p_df < 0.05 else " (nicht signifikant)"))

    print()
    if dmrg.mean() > rand.mean() and p_dr < 0.05:
        print("  -> Das spektrale Kriterium schlaegt die Zufallskontrolle.")
        print("     Die DMRG-artige Zuweisung traegt einen eigenen Beitrag.")
    else:
        print("  -> Das Kriterium schlaegt den Zufall nicht.")
        print("     Was wirkt, ist hoechstens das Wachsen an sich, nicht die")
        print("     spektrale Auswahl -- H0 bestaetigt.")


if __name__ == "__main__":
    analyse(run())
