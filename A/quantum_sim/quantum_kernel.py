from __future__ import annotations

"""RESEARCH ID: K-001 -- Quantenkerne: Datenbedarf statt Modellgroesse.

WARUM DIESER ANSATZ ANDERS IST ALS ALLE SECHS VORHERIGEN
A-001 bis Q-003 haben durchgehend *das Modell* veraendert und Genauigkeit,
Parameterzahl oder Rechenzeit gemessen. Jedes Mal gab es eine klassische
Alternative, die dasselbe leistete.

Hier wird nichts am Modell veraendert. Der Quantenzustandsraum dient als
*Merkmalsraum*: jeder Datenpunkt wird in einen 2^n-dimensionalen Hilbertraum
eingebettet, und der Kern misst die Ueberlappung zweier Zustaende.

    k(x, y) = |<phi(x)|phi(y)>|^2

Das ist keine Parametrisierung, die man klassisch anders hinschreiben koennte,
sondern eine Aehnlichkeitsstruktur, die aus der Geometrie des Hilbertraums
folgt. Und die Zielgroesse ist eine andere: Kernmethoden sind dateneffizient,
gemessen wird also **Datenbedarf** -- die einzige Ressource der Projektliste,
die nie untersucht wurde.

DER MERKMALSABBILDUNG
Verwendet wird die in der Literatur uebliche ZZ-Merkmalsabbildung
(Havlicek et al. 2019): je Qubit H und Rz(x_i), dann fuer jedes Paar (i,j) ein
verschraenkender Term Rzz((pi - x_i)(pi - x_j)). Der ganze Block wird
wiederholt. Die Paar-Terme sind entscheidend -- ohne sie zerfaellt der Kern in
ein Produkt aus Einzelqubit-Beitraegen und waere trivial klassisch.

KEIN RECHENVORTEIL -- vorab und ausdruecklich
Bei simulierbarer Qubitzahl ist der Kern klassisch berechenbar, ein
Geschwindigkeitsvorteil ist ausgeschlossen und wird nicht behauptet.
Untersucht wird der **induktive Bias**: ob diese Geometrie besser zu den Daten
passt als die klassischen Standardkerne.

BASELINES
  quantum      k(x,y) = |<phi(x)|phi(y)>|^2
  rbf          Gauss-Kern, der Standard fuer solche Aufgaben
  poly         polynomialer Kern (Grad 3)
  linear       linearer Kern (Untergrenze)

Alle bekommen exakt dieselben n Merkmale (PCA auf n Dimensionen, ohne Labels),
und alle SVMs werden ueber dieselbe C-Gitter-Suche optimiert. Ohne das waere
der Vergleich unfair.

HYPOTHESEN
  H1 : Der Quantenkern erreicht bei GLEICHER Datenmenge hoehere Genauigkeit,
       insbesondere bei wenigen Beispielen
  H0 : kein Unterschied zu den klassischen Kernen
  H2 : der Quantenkern ist schlechter -- die Geometrie passt nicht zu Bilddaten

FALSIFIKATION
Schlaegt der Quantenkern die klassischen Kerne bei keiner Datenmenge, ist der
Ansatz fuer diese Aufgabe ohne Mehrwert.

Run: python -m quantum_sim.quantum_kernel
"""

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from qire_ai.benchmark import set_seed
from qire_ai.data_loader import load_dataset
from .torch_simulator import H, TorchStatevector


def _rzz(theta: float | torch.Tensor) -> torch.Tensor:
    """exp(-i theta/2 Z (x) Z), diagonal in der Rechenbasis."""
    t = torch.as_tensor(theta, dtype=torch.float32)
    phase = torch.stack([-t / 2, t / 2, t / 2, -t / 2])
    diag = torch.complex(torch.cos(phase), torch.sin(phase))
    return torch.diag(diag)


def _rz_batched(state: torch.Tensor, angles: torch.Tensor, qubit: int) -> torch.Tensor:
    """Rz mit eigenem Winkel je Batch-Element."""
    half = angles / 2
    e_minus = torch.complex(torch.cos(half), -torch.sin(half))
    e_plus = torch.complex(torch.cos(half), torch.sin(half))
    axis = qubit + 1
    moved = state.movedim(axis, -1)
    shape = [-1] + [1] * (moved.dim() - 2)
    a = moved[..., 0] * e_minus.reshape(shape)
    b = moved[..., 1] * e_plus.reshape(shape)
    return torch.stack([a, b], dim=-1).movedim(-1, axis)


def _rzz_batched(state: torch.Tensor, angles: torch.Tensor, q1: int, q2: int) -> torch.Tensor:
    """Verschraenkender ZZ-Term mit eigenem Winkel je Batch-Element.

    Diagonal in der Rechenbasis: die Phase haengt vom Produkt der beiden
    Bits ab, deshalb genuegt eine Multiplikation mit einem Phasenfaktor.
    """
    half = angles / 2
    a1, a2 = q1 + 1, q2 + 1
    moved = state.movedim(a1, -2).movedim(a2, -1)      # (..., 2, 2)
    shape = [-1] + [1] * (moved.dim() - 3)
    e_minus = torch.complex(torch.cos(half), -torch.sin(half)).reshape(shape)
    e_plus = torch.complex(torch.cos(half), torch.sin(half)).reshape(shape)
    out = torch.empty_like(moved)
    out[..., 0, 0] = moved[..., 0, 0] * e_minus        # ZZ = +1
    out[..., 0, 1] = moved[..., 0, 1] * e_plus         # ZZ = -1
    out[..., 1, 0] = moved[..., 1, 0] * e_plus
    out[..., 1, 1] = moved[..., 1, 1] * e_minus
    return out.movedim(-1, a2).movedim(-2, a1)


def zz_feature_map(x: np.ndarray, num_qubits: int, reps: int = 2) -> torch.Tensor:
    """Bettet jeden Datenpunkt in einen 2^n-dimensionalen Zustand ein.

    x: (N, n) -> Zustaende (N, 2^n) komplex
    """
    features = torch.tensor(x, dtype=torch.float32)
    batch = features.shape[0]
    sv = TorchStatevector(num_qubits, batch=batch)
    for _ in range(reps):
        for q in range(num_qubits):
            sv.apply(H, [q])
            sv.state = _rz_batched(sv.state, 2.0 * features[:, q], q)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                angle = 2.0 * (math.pi - features[:, i]) * (math.pi - features[:, j])
                sv.state = _rzz_batched(sv.state, angle, i, j)
    return sv.state.reshape(batch, -1)


def quantum_kernel_matrix(states_a: torch.Tensor, states_b: torch.Tensor) -> np.ndarray:
    """k(x,y) = |<phi(x)|phi(y)>|^2 fuer alle Paare."""
    overlap = states_a @ states_b.conj().T
    return (overlap.real ** 2 + overlap.imag ** 2).numpy()


def prepare_data(dataset: str, num_features: int, seed: int, pool: int = 4000):
    """PCA auf num_features Dimensionen, dann auf [0, pi] skaliert.

    Die PCA nutzt keine Labels, ist also fuer alle Kerne dieselbe faire
    Vorverarbeitung. Die Skalierung auf [0, pi] ist noetig, weil die Merkmale
    als Drehwinkel dienen.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    bundle = load_dataset(dataset, seed=seed, max_samples=pool)
    x_train = bundle.x_train.numpy()
    x_test = bundle.x_test.numpy()

    pca = PCA(n_components=num_features, random_state=seed)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    scaler = MinMaxScaler(feature_range=(0.0, math.pi))
    x_train = scaler.fit_transform(x_train)
    x_test = np.clip(scaler.transform(x_test), 0.0, math.pi)

    return (x_train, bundle.y_train.numpy(), x_test, bundle.y_test.numpy(),
            float(pca.explained_variance_ratio_.sum()))


def evaluate_kernels(train_sizes: Sequence[int] = (25, 50, 100, 200, 400),
                     num_qubits: int = 6,
                     seeds: Sequence[int] = (1, 2, 3, 4, 5),
                     dataset: str = "mnist",
                     test_size: int = 500,
                     c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0)) -> Dict:
    from sklearn.svm import SVC

    results: Dict[Tuple[str, int], List[float]] = {}
    variance = None
    timings: Dict[str, float] = {"quantum": 0.0, "classical": 0.0}

    for seed in seeds:
        set_seed(seed)
        x_tr_all, y_tr_all, x_te_all, y_te_all, variance = prepare_data(
            dataset, num_qubits, seed)
        rng = np.random.default_rng(seed)
        test_idx = rng.choice(len(x_te_all), size=min(test_size, len(x_te_all)), replace=False)
        x_te, y_te = x_te_all[test_idx], y_te_all[test_idx]

        t0 = time.perf_counter()
        states_te = zz_feature_map(x_te, num_qubits)
        timings["quantum"] += time.perf_counter() - t0

        for n_train in train_sizes:
            idx = rng.choice(len(x_tr_all), size=n_train, replace=False)
            x_tr, y_tr = x_tr_all[idx], y_tr_all[idx]

            # --- Quantenkern ---
            t0 = time.perf_counter()
            states_tr = zz_feature_map(x_tr, num_qubits)
            k_train = quantum_kernel_matrix(states_tr, states_tr)
            k_test = quantum_kernel_matrix(states_te, states_tr)
            timings["quantum"] += time.perf_counter() - t0

            best = 0.0
            for c in c_grid:
                svm = SVC(kernel="precomputed", C=c).fit(k_train, y_tr)
                best = max(best, float(svm.score(k_test, y_te)))
            results.setdefault(("quantum", n_train), []).append(best)

            # --- klassische Kerne, identische Merkmale und C-Gitter ---
            t0 = time.perf_counter()
            for name, kwargs in (("rbf", {"kernel": "rbf"}),
                                 ("poly", {"kernel": "poly", "degree": 3}),
                                 ("linear", {"kernel": "linear"})):
                best_c = 0.0
                for c in c_grid:
                    svm = SVC(C=c, **kwargs).fit(x_tr, y_tr)
                    best_c = max(best_c, float(svm.score(x_te, y_te)))
                results.setdefault((name, n_train), []).append(best_c)
            timings["classical"] += time.perf_counter() - t0
        print(f"  seed {seed} fertig")

    return {"results": results, "train_sizes": list(train_sizes),
            "num_qubits": num_qubits, "explained_variance": variance,
            "timings": timings, "seeds": list(seeds)}


def analyse(out: Dict) -> None:
    from scipy import stats

    results = out["results"]
    sizes = out["train_sizes"]
    kernels = ["quantum", "rbf", "poly", "linear"]
    n_seeds = len(out["seeds"])

    print()
    print(f"RESEARCH ID K-001 | {out['num_qubits']} Qubits ({2**out['num_qubits']}-dim "
          f"Hilbertraum), {n_seeds} Seeds")
    print(f"PCA auf {out['num_qubits']} Merkmale, erklaerte Varianz "
          f"{out['explained_variance']*100:.1f}%")
    print()
    header = f'{"Trainingsdaten":>15}' + "".join(f'{k:>12}' for k in kernels)
    print(header); print("-" * len(header))
    for n in sizes:
        row = f'{n:>15}'
        for k in kernels:
            row += f'{np.mean(results[(k, n)]):>12.4f}'
        print(row)

    print()
    print("Quantenkern gegen den besten klassischen Kern, je Datenmenge:")
    any_win = False
    for n in sizes:
        q = np.array(results[("quantum", n)])
        best_name, best_vals = max(
            ((k, np.array(results[(k, n)])) for k in kernels if k != "quantum"),
            key=lambda kv: kv[1].mean())
        p = stats.wilcoxon(q, best_vals).pvalue if not np.allclose(q, best_vals) else 1.0
        wins = q.mean() > best_vals.mean()
        if wins and p < 0.05:
            any_win = True
        flag = " *signifikant*" if p < 0.05 else " (n.s.)"
        print(f'  n={n:>4}: quantum {q.mean():.4f} vs {best_name} {best_vals.mean():.4f} '
              f'-> {q.mean()-best_vals.mean():+.4f}, p={p:.4f}{flag}')

    print()
    print(f"Rechenzeit: Quantenkern {out['timings']['quantum']:.1f}s, "
          f"klassische Kerne {out['timings']['classical']:.1f}s")
    print()
    if any_win:
        print("  -> Der Quantenkern schlaegt die klassischen Kerne bei mindestens")
        print("     einer Datenmenge signifikant. Der induktive Bias traegt.")
    else:
        print("  -> Der Quantenkern schlaegt die klassischen Kerne bei keiner")
        print("     Datenmenge. Fuer diese Aufgabe kein Mehrwert -- H0 bzw. H2.")


if __name__ == "__main__":
    analyse(evaluate_kernels())
