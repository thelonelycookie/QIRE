"""Z1 -- z² gegen GELU/ReLU², Lernrate je (Arm, d, T) neu gesucht (Erkundung Welle 3, KEIN Befund).

FRAGE: Kippt der z²-Vorteil bei T = 16000 (I-045-v3: d=512 z² +0,047 bpc schlechter als GELU) auch dann, wenn die Lernrate je T gesucht wird?
MODELL: Byte-LM wie _bytelm.py / struktur_sprache.py (UD_German-GSD, 4 Lagen, FFN 4d dicht, Batch 8x128, Adam, 5 % Aufwärmen, Kosinus, Clip 1,0).
ARME (nur FFN-Aktivierung; Parameter und Matmul-FLOPs exakt gleich):
  gelu   GELU                      relu2  relu(z)² (Primer)
  z2     z²                        z2ln   LayerNorm ohne Parameter vor dem Quadrat: LN(z)²
BEDINGUNGEN: d in {256, 512}, T in {4000, 16000}.  Lernrate je (Arm, d, T) auf VOLLER Schrittzahl T, Gitter 3e-4 / 6e-4 / 1,2e-3 / 2,4e-3, Wahl nach Val-bpc.
ABLAUF: Seed 950 (Tuning-Seed) fährt alle 4 Raten je Zelle, wählt die beste (Val) und schreibt roh/Z1_lr_tabelle.json (mit Randflag).
  Jeder weitere Seed 951-959 fährt nur die gewählte Rate je Zelle (Tabelle muss existieren; sonst Abbruch). Die Tabelle gilt je Zelle, nicht je Seed.
  Ausgewiesen: Val-, Test-, Train-bpc (erste 64 kB Trainingsbytes), Lücke, Parameter, FLOPs, Sekunden, Instabilität.
  Randtreffer (beste Rate = 3e-4 oder 2,4e-3) stehen in der Tabelle als rand=true; das Gitter wird NICHT automatisch verlängert (dann bewusst nachziehen).
PARALLEL: Umgebungsvariable Z1_PROZESSE=<n> verteilt die Einzelläufe (Zelle x Rate) auf n Prozesse (je 1 Thread). Wiederaufnahme je Einzellauf (.partial).
ZEIT (Messwerte aus Welle 2: d=256 ~0,32 s/Schritt, d=512 ~1,0 s/Schritt, 1 Kern): ein Seed nach Tabelle = 4 Arme x 20000 Schritte x (0,32+1,0) s ~ 29 Kernstunden;
  Seed 950 mit Tuning = 4x so viel ~ 117 Kernstunden. Wand bei Z1_PROZESSE >= 16: Seed 950 ~ 4,4 h (längster Einzellauf d512/T16000 ~ 4,4 h), weitere Seeds ~ 4,4 h.
  Auf 1 Kern ist ein Seed NICHT in 3 h möglich (29 Kernstunden).
Aufruf: python3 erkundung/Z1_z2lr.py --seed 950 [--schnell]   ->   erkundung/roh/Z1_z2lr_s<seed>.json
--schnell: d in {32,64}, T in {20,60}, kleine Daten, tuned im selben Aufruf (tabelle _schnell)."""
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bytelm as B  # noqa: E402
from _bytelm import LM, FFNDicht, HIER, bpc, cli, lade_teil, speichere, speichere_teil, trainiere  # noqa: E402

KID = "Z1_z2lr"
ARME = ("gelu", "relu2", "z2", "z2ln")
GITTER = (3e-4, 6e-4, 1.2e-3, 2.4e-3)
TUNE_SEED = 950
TRAIN_BYTES = 65536


def bauer(d, arm, seed):
    return LM(d, lambda dd, i: FFNDicht(dd, 4 * dd, arm))


def extra(netz, va, te, mess):
    tr = B.daten(B_KLEIN[0])[0]
    return {"bpc_train": bpc(netz, tr[:(8192 if B_KLEIN[0] else TRAIN_BYTES)])}


B_KLEIN = [False]


def einzel(arm, d, T, lr, seed, schnell):
    B_KLEIN[0] = schnell
    r = trainiere(functools.partial(bauer, d, arm), lr, seed, T, schnell, ("val", "test"), extra)
    if not r["instabil"]:
        r["luecke_test_minus_train"] = r["bpc_test"] - r["bpc_train"]
        r.update(bauer(d, arm, seed).zahlen())
    return r


def tabelle_pfad(schnell):
    return HIER / "roh" / ("Z1_lr_tabelle_schnell.json" if schnell else "Z1_lr_tabelle.json")


def zellen_liste(schnell):
    ds, Ts = ((32, 64), (20, 60)) if schnell else ((256, 512), (4000, 16000))
    return [(a, d, T) for d in ds for T in Ts for a in ARME]


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    zellen = zellen_liste(schnell)
    tp = tabelle_pfad(schnell)
    tuned = seed == TUNE_SEED or schnell
    tabelle = {}
    if not tuned:
        if not tp.exists():
            raise SystemExit(f"Lernraten-Tabelle {tp} fehlt: zuerst Seed {TUNE_SEED} laufen lassen.")
        tabelle = json.loads(tp.read_text())
    teil = lade_teil(KID, seed, schnell)
    # Aufträge
    jobs = [(a, d, T, lr) for (a, d, T) in zellen for lr in (GITTER if tuned else (tabelle[f"{a}/{d}/{T}"]["lr"],))]
    key = lambda j: f"{j[0]}/d{j[1]}/T{j[2]}/lr{j[3]:g}"
    offen = [j for j in jobs if key(j) not in teil]
    n = int(os.environ.get("Z1_PROZESSE", "1"))
    print(f"[{KID} s{seed}] {len(jobs)} Einzelläufe, offen {len(offen)}, Prozesse {n}", flush=True)
    offen.sort(key=lambda j: -(j[1] * j[2]))              # längste zuerst
    if n <= 1:
        for j in offen:
            teil[key(j)] = einzel(*j, seed, schnell)
            speichere_teil(KID, seed, schnell, teil)
            print(f"  {key(j)} val={teil[key(j)]['bpc_val']:.4f} test={teil[key(j)]['bpc_test']:.4f}", flush=True)
    else:
        with ProcessPoolExecutor(n) as ex:
            fut = {ex.submit(einzel, *j, seed, schnell): j for j in offen}
            from concurrent.futures import as_completed
            for f in as_completed(fut):
                j = fut[f]
                teil[key(j)] = f.result()
                speichere_teil(KID, seed, schnell, teil)
                print(f"  {key(j)} val={teil[key(j)]['bpc_val']:.4f} test={teil[key(j)]['bpc_test']:.4f}", flush=True)
    # Auswertung je Zelle
    zellen_out, neu_tab = {}, {}
    for (a, d, T) in zellen:
        zn = f"{a}/{d}/{T}"
        if tuned:
            kand = {lr: teil[key((a, d, T, lr))] for lr in GITTER}
            best = min(kand, key=lambda l: kand[l]["bpc_val"])
            neu_tab[zn] = {"lr": best, "rand": best in (GITTER[0], GITTER[-1]),
                           "val_je_lr": {f"{l:g}": kand[l]["bpc_val"] for l in GITTER}}
            r = dict(kand[best])
            r["lr_gitter_val"] = neu_tab[zn]["val_je_lr"]
        else:
            best = tabelle[zn]["lr"]
            r = dict(teil[key((a, d, T, best))])
            neu_tab[zn] = tabelle[zn]
        r.update({"lr": best, "lr_rand": neu_tab[zn]["rand"], "arm": a, "d": d, "T": T})
        zellen_out[zn] = r
    if tuned:
        tp.write_text(json.dumps(neu_tab, indent=1))
    out = {"kid": KID, "seed": seed, "schnell": schnell, "beschreibung": __doc__, "zellen": zellen_out,
           "einzellaeufe_roh": teil, "sekunden_gesamt_dieser_aufruf": time.time() - t0}
    # Differenzen z2 - gelu je (d, T) (negativ = z2 besser)
    diff = {}
    for (a, d, T) in zellen:
        if a == "gelu":
            for b in ("z2", "z2ln", "relu2"):
                x, y = zellen_out[f"{b}/{d}/{T}"], zellen_out[f"gelu/{d}/{T}"]
                diff[f"{b}-gelu/d{d}/T{T}"] = x["bpc_test"] - y["bpc_test"]
    out["diff_test_bpc_arm_minus_gelu"] = diff
    p = speichere(KID, seed, schnell, out)
    print("geschrieben", p)
    return out


if __name__ == "__main__":
    cli(KID, main)
