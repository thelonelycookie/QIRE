from __future__ import annotations

"""RESEARCH ID: Q-013 -- Die Architektur an einem Sprachmodell.

WARUM DAS DIE GROESSTE LUECKE IST
Jedes Ergebnis dieses Projekts steht auf Bildklassifikation. Das erklaerte Ziel
ist aber ein ressourceneffizienteres Sprachmodell. Solange nichts auf
Sequenzdaten gemessen ist, laesst sich auf die naheliegendste Frage einer Jury
-- "gilt das auch fuer Sprache?" -- nur antworten: weiss ich nicht.

Und es gibt einen konkreten Grund, es gerade jetzt zu messen. Der
Uebergangspunkt der Butterfly-Architektur liegt bei **16.384 Dimensionen**; die
Feed-Forward-Bloecke echter Sprachmodelle liegen bei **4.096 bis 16.384**. Das
ist genau der Bereich, in dem O(d log d) gegen O(d^2) gewinnt -- kein Zufall,
sondern der Bereich, fuer den die Struktur gebaut ist. Ob der Vorteil dort auch
bei der QUALITAET besteht, ist offen.

DIE DATEN -- bewusst synthetisch
Erzeugt wird Text aus einer **Markow-Kette k-ter Ordnung** ueber ein kleines
Alphabet. Drei Gruende:

  1. **Kein Download, keine Lizenzfrage, vollstaendig reproduzierbar.** Der
     Korpus haengt nur am Seed.
  2. **Die Schwierigkeit ist einstellbar.** Die Ordnung k sagt, wie viel
     Vorgeschichte noetig ist; das ist die Kapazitaetsachse, genau wie die
     intrinsische Dimension in B-014.
  3. **Die untere Schranke ist bekannt.** Die Entropie der Quelle laesst sich
     ausrechnen, also weiss man, wie gut ein perfektes Modell waere. Das ist bei
     echtem Text nicht so, und es macht jede Aussage ueberpruefbar.

Der Preis: Das ist keine natuerliche Sprache. Ergebnisse hier sind ein
notwendiger erster Schritt, kein Ersatz fuer eine Messung an echtem Text. So
gehoert es auch berichtet.

DIE ARME -- ersetzt wird der Feed-Forward-Block, sonst nichts
Der Transformer bleibt in allen Armen identisch (Einbettung, Aufmerksamkeit,
Normierung, Kopf). Nur der FFN-Block wechselt:

  dense_ffn       Linear(d, 4d) -> GELU -> Linear(4d, d).  8*d^2 Parameter.
                  Der Standardaufbau, gegen den anzutreten ist.
  butterfly_ffn   geteiltes Butterfly ueber log2(d) Ebenen, Born-Regel als
                  Nichtlinearitaet.  2*log2(d)*L Parameter.
  butterfly_untied  dasselbe, aber ein Winkel je Paar statt je Ebene.
                  ABGRENZUNGSKONTROLLE: trennt, ob die Gewichtsteilung -- der
                  quantenabgeleitete Anteil -- etwas beitraegt oder ob nur die
                  Butterfly-Struktur zaehlt.
  narrow_ffn      Linear(d, d/4) -> GELU -> Linear(d/4, d).
                  AUFLOESUNGSPRUEFUNG: muss schlechter sein als `dense_ffn`.
                  Trennt sich das nicht, ist der FFN-Block fuer diese Aufgabe
                  ueberdimensioniert und kein Vergleich daran aussagekraeftig.

VORREGISTRIERUNG
  Gemessen wird **Kreuzentropie auf dem Testteil** (nats je Zeichen), niedriger
  ist besser. Zusaetzlich berichtet: Abstand zur bekannten Quellentropie,
  Parameterzahl des FFN-Blocks, Trainingszeit.

  PRIMAER  butterfly_ffn gegen dense_ffn, TOST auf Gleichwertigkeit innerhalb
           **+-0,05 nats**. Weite Grenze mit Absicht: hier stehen ~80 gegen
           ~500.000 Parameter, und die Frage ist nicht "genau gleich", sondern
           "nah genug, dass der Faktor die Sache wert ist".
  P2       butterfly_ffn gegen butterfly_untied -- traegt die Gewichtsteilung?
  Holm ueber genau diese beiden.

  H1 : PRIMAER gleichwertig -> der FFN-Block eines Sprachmodells laesst sich
       durch die quantenabgeleitete Struktur ersetzen. Das waere der erste
       Befund des Projekts auf Sequenzdaten und der direkte Anschluss an das
       Projektziel.
  H0 : dense_ffn ist signifikant besser -> der Vorteil aus der Bildklassifikation
       uebertraegt sich nicht; der Abstand gehoert neben jede Zeitangabe.
  HU : weder noch -> erreichte Schranke berichten.

  ERWARTUNG, vorher notiert: Ich rechne mit H0. 80 Parameter gegen 500.000 ist
  ein sehr grosser Unterschied, und anders als bei der Bildaufgabe gibt es hier
  keinen Hinweis, dass die Aufgabe wenig Kapazitaet braucht. Sollte H1 doch
  eintreten, ist das umso staerker -- und sollte H0 eintreten, ist der
  interessante Wert der ABSTAND je gespartem Parameter, nicht das Vorzeichen.

Run: python -m quantum_sim.sequence_platform
"""

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qire_ai.benchmark import set_seed
from qire_ai.equivalence import holm, tost_paired
from qire_ai.result_store import Recorder

from .butterfly_layer import apply_1q, apply_1q_untied, cnot_permutation

ARMS = ("dense_ffn", "butterfly_ffn", "butterfly_untied", "narrow_ffn")
REGISTERED = ("PRIMAER", "P2")
ALPHABET = 32
ORDERS = (2, 4)                 # Ordnung der Markow-Quelle = Schwierigkeit
D_MODEL = 256                   # 2^8, damit das Butterfly passt
N_HEADS = 4
N_BLOCKS = 2
CONTEXT = 64
FFN_LAYERS = 5
N_TRAIN_TOKENS, N_TEST_TOKENS = 120_000, 30_000
EPOCHS = 3
BATCH = 32
EVAL_SEEDS = (5501, 5502, 5503, 5504, 5505)
TUNING_SEED = 5551
LR_GRID = (0.0003, 0.001, 0.003)
MARGIN_NATS = 0.05


# --------------------------------------------------------------------------
# Quelle mit bekannter Entropie
# --------------------------------------------------------------------------

def markov_corpus(seed: int, order: int, n_tokens: int,
                  alphabet: int = ALPHABET) -> Tuple[torch.Tensor, float]:
    """Text aus einer Markow-Kette k-ter Ordnung, plus die wahre Entropie.

    Die Uebergangsverteilungen werden aus einer Dirichlet-aehnlichen Ziehung
    gebildet (normierte Gamma-Werte), damit sie weder gleichverteilt noch
    entartet sind. Die Entropie der Quelle ist der Mittelwert der bedingten
    Entropien und damit die untere Schranke, die ein perfektes Modell erreicht.
    """
    rng = np.random.default_rng(seed)
    n_states = alphabet ** order
    # konzentrierte Verteilungen: wenige wahrscheinliche Folgezeichen
    probs = rng.gamma(shape=0.35, size=(n_states, alphabet))
    probs /= probs.sum(axis=1, keepdims=True)
    entropy = float(np.mean(-(probs * np.log(probs + 1e-12)).sum(axis=1)))

    tokens = np.empty(n_tokens, dtype=np.int64)
    tokens[:order] = rng.integers(0, alphabet, size=order)
    state = 0
    for i in range(order):
        state = state * alphabet + int(tokens[i])
    cum = probs.cumsum(axis=1)
    draws = rng.random(n_tokens)
    for i in range(order, n_tokens):
        nxt = int(np.searchsorted(cum[state], draws[i]))
        tokens[i] = nxt
        state = (state * alphabet + nxt) % n_states
    return torch.from_numpy(tokens), entropy


# --------------------------------------------------------------------------
# Die FFN-Varianten
# --------------------------------------------------------------------------

class ButterflyFFN(nn.Module):
    """Geteiltes Butterfly + Born-Regel als Nichtlinearitaet, d -> d."""

    def __init__(self, d_model: int, num_layers: int = FFN_LAYERS,
                 tie_weights: bool = True):
        super().__init__()
        self.n = int(math.log2(d_model))
        assert 2 ** self.n == d_model, "d_model muss eine Zweierpotenz sein"
        self.L, self.tie = num_layers, tie_weights
        if tie_weights:
            self.theta = nn.Parameter(torch.rand(num_layers, self.n, 2) * 2 * math.pi)
        else:
            self.theta = nn.Parameter(
                torch.rand(num_layers, self.n, d_model // 2, 2) * 2 * math.pi)
        for q in range(self.n):
            self.register_buffer(f"perm_{q}",
                                 cnot_permutation(self.n, q, (q + 1) % self.n))
        self.scale = nn.Parameter(torch.ones(1))

    def _gate(self, layer: int, q: int) -> torch.Tensor:
        """2x2-Gatter fuer diese Ebene -- geteilt eines, ungeteilt eines je Paar.

        FEHLER, BEHOBEN AM 10. AUGUST, VOR DEM ERSTEN LAUF VON Q-013:
        Hier stand `self.theta[layer, q].mean(0)` fuer den ungeteilten Fall. Das
        mittelte die d/2 Paar-Winkel zu EINEM und wandte ihn ueber `apply_1q`
        auf alle Paare an -- der ungeteilte Arm rechnete damit exakt die
        geteilte Funktion. Nachgemessen bei d=64: 1.921 Parameter, aber nur 61
        wirksame Freiheitsgrade, Abweichung zum geteilten Arm exakt 0,0.

        Q-013s vorregistriertes P2 ("butterfly_ffn gegen butterfly_untied --
        traegt die Gewichtsteilung?") haette damit eine Sache gegen sich selbst
        gestellt und zuverlaessig "gleichwertig" gemeldet. Der Lauf war noch
        nicht gestartet, es ist also kein Ergebnis betroffen.

        Derselbe Fehler war in `butterfly_layer.py` schon einmal gefunden und
        dort behoben worden -- hier nicht. Siehe BEFUNDE_UND_RICHTUNGEN.md,
        "Ungeteilter Butterfly-Pfad falsch implementiert".
        """
        t = self.theta[layer, q]
        a, b = (t[0], t[1]) if self.tie else (t[..., 0], t[..., 1])
        ca, sa = torch.cos(a / 2).to(torch.complex64), torch.sin(a / 2).to(torch.complex64)
        eb, ebc = torch.exp(-0.5j * b.to(torch.complex64)), torch.exp(0.5j * b.to(torch.complex64))
        gate = torch.stack([torch.stack([ca * eb, -sa * eb], dim=-1),
                            torch.stack([sa * ebc, ca * ebc], dim=-1)], dim=-2)
        if self.tie:
            return gate
        # (d/2, 2, 2) -> (2**q, 2**(n-1-q), 2, 2) fuer diese Ebene
        return gate.reshape(2 ** q, 2 ** (self.n - 1 - q), 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        state = flat.to(torch.complex64)
        norm = state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        state = state / norm
        schritt = apply_1q if self.tie else apply_1q_untied
        for layer in range(self.L):
            for q in range(self.n):
                state = schritt(state, self._gate(layer, q), q, self.n)
            for q in range(self.n):
                state = state[:, getattr(self, f"perm_{q}")]
        prob = (state.real ** 2 + state.imag ** 2) * float(2 ** self.n)
        return (prob * self.scale).reshape(shape).to(x.dtype)


def make_ffn(arm: str, d: int) -> nn.Module:
    if arm == "dense_ffn":
        return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    if arm == "narrow_ffn":
        return nn.Sequential(nn.Linear(d, d // 4), nn.GELU(), nn.Linear(d // 4, d))
    if arm == "butterfly_ffn":
        return ButterflyFFN(d, tie_weights=True)
    if arm == "butterfly_untied":
        return ButterflyFFN(d, tie_weights=False)
    raise ValueError(arm)


class Block(nn.Module):
    def __init__(self, arm: str, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = make_ffn(arm, d)

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.n2(x))


class TinyTransformer(nn.Module):
    def __init__(self, arm: str, vocab: int = ALPHABET, d: int = D_MODEL,
                 heads: int = N_HEADS, blocks: int = N_BLOCKS,
                 context: int = CONTEXT):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, context, d))
        self.blocks = nn.ModuleList([Block(arm, d, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(
            torch.full((context, context), float("-inf")), diagonal=1))

    def ffn_parameters(self) -> int:
        return sum(p.numel() for b in self.blocks for p in b.ffn.parameters())

    def forward(self, idx):
        x = self.tok(idx) + self.pos[:, :idx.shape[1]]
        m = self.mask[:idx.shape[1], :idx.shape[1]]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.norm(x))


def _batches(tokens: torch.Tensor, batch: int, context: int, gen: torch.Generator):
    n = tokens.shape[0] - context - 1
    order = torch.randperm(n, generator=gen)
    for s in range(0, n - batch, batch):
        idx = order[s:s + batch]
        x = torch.stack([tokens[i:i + context] for i in idx])
        y = torch.stack([tokens[i + 1:i + context + 1] for i in idx])
        yield x, y


def _cell(arm: str, order: int, seed: int, lr: float, threads: int,
          epochs: int = EPOCHS) -> Dict[str, object]:
    set_seed(seed)
    torch.set_num_threads(threads)
    train, ent = markov_corpus(seed, order, N_TRAIN_TOKENS)
    test, _ = markov_corpus(seed + 10_000, order, N_TEST_TOKENS)

    model = TinyTransformer(arm)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y in _batches(train, BATCH, CONTEXT, g):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            F.cross_entropy(logits.reshape(-1, ALPHABET), y.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    train_s = time.perf_counter() - t0

    model.eval()
    losses, gt = [], torch.Generator().manual_seed(0)
    with torch.no_grad():
        for x, y in _batches(test, BATCH, CONTEXT, gt):
            logits = model(x)
            losses.append(float(F.cross_entropy(
                logits.reshape(-1, ALPHABET), y.reshape(-1))))
    ce = float(np.mean(losses))
    return {"arm": arm, "order": order, "seed": seed, "lr": lr,
            "cross_entropy": ce, "source_entropy": ent, "excess": ce - ent,
            "perplexity": float(np.exp(ce)),
            "ffn_parameters": model.ffn_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": train_s}


def run(orders: Sequence[int] = ORDERS, seeds: Sequence[int] = EVAL_SEEDS,
        threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="Q-013",
        description=("Butterfly-FFN gegen dichten FFN in einem kleinen "
                     "Transformer, auf Markow-Text mit bekannter Entropie. "
                     "Erste Messung des Projekts auf Sequenzdaten."),
        config={"arms": list(ARMS), "orders": list(orders), "seeds": list(seeds),
                "alphabet": ALPHABET, "d_model": D_MODEL, "n_heads": N_HEADS,
                "n_blocks": N_BLOCKS, "context": CONTEXT,
                "ffn_layers": FFN_LAYERS, "epochs": EPOCHS, "batch_size": BATCH,
                "n_train_tokens": N_TRAIN_TOKENS, "lr_grid": list(LR_GRID),
                "margin_nats": MARGIN_NATS, "torch_threads": threads,
                "test_metric": "cross_entropy (nats/Zeichen, kleiner ist besser)",
                "primary": "butterfly_ffn gleichwertig zu dense_ffn (+-0,05 nats)",
                "p2": "butterfly_ffn gegen butterfly_untied -- traegt die Teilung?",
                "holm_family": list(REGISTERED),
                "resolution_check": "narrow_ffn muss schlechter sein als dense_ffn",
                "erwartung_vorher": "H0 -- 80 gegen 500.000 Parameter ist viel",
                "grenze": ("synthetischer Markow-Text, keine natuerliche "
                           "Sprache; notwendiger erster Schritt, kein Ersatz")},
    )
    for order in orders:
        print(f"\n=== Markow-Ordnung {order} ===")
        lr_map = {}
        for arm in ARMS:
            best, best_ce = LR_GRID[0], float("inf")
            for lr in LR_GRID:
                ce = _cell(arm, order, TUNING_SEED, lr, threads,
                           epochs=1)["cross_entropy"]
                if ce < best_ce:
                    best, best_ce = lr, ce
            lr_map[arm] = best
            print(f"  {arm:>18}: lr {best}")
        rec.config[f"lr_map_order{order}"] = dict(lr_map)
        # Randlagen melden -- zweimal in dieser Nacht hat ein zu enges Gitter
        # das Optimum abgeschnitten (Q-003-replication und Q-009).
        edge = [a for a, v in lr_map.items() if v in (min(LR_GRID), max(LR_GRID))]
        rec.config[f"lr_at_grid_edge_order{order}"] = edge
        if edge:
            print(f"  WARNUNG: am Gitterrand -> {', '.join(edge)}")
        for seed in seeds:
            for arm in ARMS:
                rec.add(**_cell(arm, order, seed, lr_map[arm], threads))
            print(f"    seed {seed} fertig")
        for arm in ARMS:
            v = [r["cross_entropy"] for r in rec.records
                 if r["arm"] == arm and r["order"] == order]
            e = [r["excess"] for r in rec.records
                 if r["arm"] == arm and r["order"] == order]
            print(f"    {arm:>18}  CE {np.mean(v):.4f}  ueber Quelle "
                  f"{np.mean(e):+.4f}")
        print("    Zwischenstand gesichert:")
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    from scipy import stats

    if isinstance(rec_or_id, Recorder):
        records, config = rec_or_id.records, rec_or_id.config
    else:
        from qire_ai.result_store import load
        payload = load(rec_or_id)
        records, config = payload["records"], payload["config"]
    orders = sorted({int(r["order"]) for r in records})
    out: Dict[str, object] = {}

    def vals(arm: str, order: int, field: str = "cross_entropy") -> np.ndarray:
        rows = sorted((r for r in records if r["arm"] == arm
                       and int(r["order"]) == order), key=lambda r: r["seed"])
        return np.array([r[field] for r in rows])

    print()
    print("RESEARCH ID Q-013 | Sprachmodell, Kreuzentropie in nats "
          "(kleiner ist besser)")
    for order in orders:
        ent = float(np.mean([r["source_entropy"] for r in records
                             if int(r["order"]) == order]))
        print()
        print(f"--- Markow-Ordnung {order}, Quellentropie {ent:.4f} " + "-" * 20)
        head = (f'  {"Arm":>18} {"CE":>8} {"ueber Quelle":>13} {"Perplexitaet":>13} '
                f'{"FFN-Par.":>11} {"Zeit":>8}')
        print(head); print("  " + "-" * (len(head) - 2))
        for arm in ARMS:
            v = vals(arm, order)
            if not len(v):
                continue
            ex = vals(arm, order, "excess")
            pp = vals(arm, order, "perplexity")
            par = next(r["ffn_parameters"] for r in records
                       if r["arm"] == arm and int(r["order"]) == order)
            t = vals(arm, order, "train_seconds")
            print(f'  {arm:>18} {v.mean():>8.4f} {ex.mean():>+13.4f} '
                  f'{pp.mean():>13.3f} {par:>11,} {t.mean():>7.0f}s')

    o0 = orders[0]
    fam = [("butterfly_ffn", "dense_ffn", "PRIMAER"),
           ("butterfly_ffn", "butterfly_untied", "P2")]
    raw, cells = [], []
    for a, b, tag in fam:
        va, vb = vals(a, o0), vals(b, o0)
        p = float(stats.wilcoxon(va, vb).pvalue) if len(va) > 2 else float("nan")
        raw.append(p if p == p else 1.0)
        cells.append((a, b, tag, float(va.mean() - vb.mean()), p,
                      tost_paired(va, vb, MARGIN_NATS)))
    adj = holm(raw)

    print()
    print("=" * 92)
    print(f"VORREGISTRIERTE FAMILIE, Ordnung {o0}, Grenze +-{MARGIN_NATS} nats")
    for (a, b, tag, delta, p, eq), p_h in zip(cells, adj):
        # niedrigere Kreuzentropie ist besser -> negatives delta = a ist besser
        state = ("Unterschied belegt" if p_h < 0.05 else
                 f"gleichwertig (+-{MARGIN_NATS})" if eq["p_tost"] < 0.05 else
                 f"unentschieden (+-{eq['achieved']:.4f})")
        print(f'  [{tag:>7}] {a} vs {b}: {delta:>+8.4f} nats  p={p:.4f}  '
              f'Holm={p_h:.4f}  {state}')
        out[tag] = {"delta": delta, "p": p, "p_holm": p_h,
                    "p_tost": eq["p_tost"], "achieved": eq["achieved"]}

    nv, dv = vals("narrow_ffn", o0), vals("dense_ffn", o0)
    c_p = float(stats.wilcoxon(nv, dv).pvalue) if len(nv) > 2 else 1.0
    c_ok = nv.mean() > dv.mean() and c_p < 0.05
    print()
    ctxt = (f"Aufloesungspruefung {'BESTANDEN' if c_ok else 'NICHT BESTANDEN'}: "
            f"schmaler FFN gegen dichten {nv.mean() - dv.mean():+.4f} nats, "
            f"p={c_p:.4f}." + ("" if c_ok else " Der FFN-Block ist fuer diese "
            "Aufgabe ueberdimensioniert -- kein Vergleich daran ist aussagekraeftig."))
    print(f"  {ctxt}")
    out["resolution_check"] = {"passed": bool(c_ok), "p": c_p, "text": ctxt}

    prim = out["PRIMAER"]
    bf = next(r["ffn_parameters"] for r in records if r["arm"] == "butterfly_ffn")
    df = next(r["ffn_parameters"] for r in records if r["arm"] == "dense_ffn")
    print()
    if not c_ok:
        v = "HU -- Aufloesungspruefung durchgefallen, Lauf nicht verwertbar."
    elif prim["p_tost"] < 0.05:
        v = (f"H1 -- der FFN-Block laesst sich durch die quantenabgeleitete "
             f"Struktur ersetzen: {bf:,} statt {df:,} Parameter "
             f"(Faktor {df / bf:.0f}), gleichwertig innerhalb "
             f"+-{MARGIN_NATS} nats. Erster Befund des Projekts auf Sequenzdaten.")
    elif prim["p_holm"] < 0.05 and prim["delta"] > 0:
        v = (f"H0 -- der dichte FFN ist besser ({prim['delta']:+.4f} nats). Der "
             f"Vorteil aus der Bildklassifikation uebertraegt sich nicht. Der "
             f"verwertbare Wert ist der Abstand je gespartem Parameter: "
             f"{prim['delta']:.4f} nats fuer den Faktor {df / bf:.0f}.")
    else:
        v = f"HU -- unentschieden; die Daten tragen erst ab +-{prim['achieved']:.4f}."
    print(f"BEWERTUNG: {v}")
    print("=" * 92)
    out["verdict"] = v
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        print('  Nachholen mit: python -c "from quantum_sim.sequence_platform '
              "import analyse; analyse('Q-013')\"")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
