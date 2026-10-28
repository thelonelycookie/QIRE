"""Gemeinsame Bausteine fuer K05-K08 (Erkundung Welle 2, KEINE Befunde). Nur Tuning-Seeds 950-959."""
import argparse
import json
import math
import os
import sys
import time

HIER = os.path.dirname(os.path.abspath(__file__))
INSTITUT = os.path.dirname(HIER)
if INSTITUT not in sys.path:
    sys.path.insert(0, INSTITUT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import abteilungen.kompression_bytes as kb  # noqa: E402
import abteilungen.struktur_sprache as ss  # noqa: E402


def _sicher(x):
    if isinstance(x, dict):
        return {str(k): _sicher(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sicher(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def speichere(kid, seed, inhalt, schnell=False):
    os.makedirs(os.path.join(HIER, "roh"), exist_ok=True)
    name = f"{kid}_schnell_s{seed}.json" if schnell else f"{kid}_s{seed}.json"
    pfad = os.path.join(HIER, "roh", name)
    tmp = pfad + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_sicher(inhalt), f, indent=1, ensure_ascii=False)
    os.replace(tmp, pfad)
    return pfad


def cli(kid, main):
    a = argparse.ArgumentParser(description=f"Erkundung {kid} (keine Befunde)")
    a.add_argument("--seed", type=int, default=950)
    a.add_argument("--schnell", action="store_true", help="Rauchtest <= 2 min (Datei roh/<KID>_schnell_s<seed>.json)")
    o = a.parse_args()
    if not (950 <= o.seed <= 959):
        print("WARNUNG: nur Tuning-Seeds 950-959 erlaubt", file=sys.stderr)
        sys.exit(2)
    t = time.time()
    r = main(o.seed, o.schnell)
    r["sekunden_gesamt"] = time.time() - t
    p = speichere(kid, o.seed, r, o.schnell)
    print(f"{kid} seed {o.seed}{' (schnell)' if o.schnell else ''}: {r['sekunden_gesamt'] / 60:.1f} min -> {p}")
    if "zusammenfassung" in r:
        print(r["zusammenfassung"])


def lr_faktor(s, n):
    warm = max(1, n // 20)
    return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm)))


def lm_trainiere(seed, T, lr, opt_fn, klein=False, d=128, lagen=4):
    """Byte-LM (kb.LM) mit EIGENEM Optimierer opt_fn(netz) -> Objekt mit .step(lr_s). Gleiche Initialisierung und Batches fuer alle Arme eines Seeds."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    netz = kb.LM(d, lagen)
    opt = opt_fn(netz)
    tr, _, _ = ss.daten(klein)
    g = torch.Generator().manual_seed(seed + 5)
    offs = torch.arange(ss.KONTEXT)
    t0 = time.time()
    letzte = []
    netz.train()
    for s in range(T):
        i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
        x = tr[i[:, None] + offs].long()
        y = tr[i[:, None] + offs + 1].long()
        loss = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            return netz, {"instabil": True, "schritt": s, "sekunden": time.time() - t0}
        netz.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step(lr * lr_faktor(s, T))
        letzte.append(loss.item())
        letzte = letzte[-50:]
    return netz, {"instabil": False, "sekunden": time.time() - t0, "train_bpc_letzte50": float(np.mean(letzte)) / math.log(2)}


def bpc(netz, klein, teil):
    _, va, te = ss.daten(klein)
    return ss.bpc_auswerten(netz, va[:ss.VAL_BYTES] if teil == "val" else te)


def lm_flops_schritt(netz, d=128, lagen=4):
    """Trainings-FLOPs je Schritt (ideal): 6 · MAC je Token · Tokens je Batch (Vorwaerts 2, Rueckwaerts 4)."""
    mac = kb.lm_macs_gesamt(d, lagen, kb.macs_linear(netz))
    return 6 * mac * ss.BATCH * ss.KONTEXT, mac


def n_param(netz):
    return int(sum(p.numel() for p in netz.parameters()))
