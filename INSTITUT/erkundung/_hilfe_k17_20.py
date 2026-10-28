"""Gemeinsame Hilfen der Erkundungsskripte K17-K20 (keine Befunde; nur Rauchtests auf Tuning-Seeds 950-959)."""
import argparse, json, math, os, sys, time
HIER = os.path.dirname(os.path.abspath(__file__))
INST = os.path.dirname(HIER)
if INST not in sys.path:
    sys.path.insert(0, INST)
TUNING_SEEDS = tuple(range(950, 960))


def pruefe_seed(seed):
    if seed not in TUNING_SEEDS:
        raise SystemExit(f"Seed {seed} ist kein Tuning-Seed (950-959); Erkundungen laufen nie auf Mess-Seeds.")


def sicher(x):
    if isinstance(x, dict):
        return {str(k): sicher(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sicher(v) for v in x]
    if hasattr(x, "item") and getattr(x, "ndim", 1) == 0:
        return sicher(x.item())
    if hasattr(x, "tolist"):
        return sicher(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def speichere(kid, seed, inhalt, schnell=False):
    os.makedirs(os.path.join(HIER, "roh"), exist_ok=True)
    pfad = os.path.join(HIER, "roh", f"{kid}_s{seed}{'_schnell' if schnell else ''}.json")
    tmp = pfad + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sicher(inhalt), f, ensure_ascii=False, indent=1)
    os.replace(tmp, pfad)
    return pfad


def cli(kid, main):
    a = argparse.ArgumentParser(description=f"Erkundung {kid} (Rauchtest, kein Befund)")
    a.add_argument("--seed", type=int, default=950)
    a.add_argument("--schnell", action="store_true", help="Rauchtest <= 2 min (Seed 959)")
    o = a.parse_args()
    t = time.time()
    r = main(o.seed, schnell=o.schnell)
    print(f"{kid} s{o.seed}{' schnell' if o.schnell else ''}: fertig in {(time.time() - t) / 60:.1f} min", flush=True)
    return r
