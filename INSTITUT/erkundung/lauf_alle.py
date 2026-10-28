"""Führt alle Erkundungsskripte K*.py / E*.py über die Tuning-Seeds 950-954 parallel aus (Wiederaufnahme: fertige JSON werden übersprungen).
Aufruf (Raven, im INSTITUT-Ordner):  OMP_NUM_THREADS=1 python3 erkundung/lauf_alle.py [--seeds 950 951 ...] [--nur K01 K03]"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
HIER = os.path.dirname(os.path.abspath(__file__))
def eine(kid, seed):
    ziel = os.path.join(HIER, "roh", f"{kid}_s{seed}.json")
    if os.path.exists(ziel): return kid, seed, "vorhanden", 0
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HIER, f"{kid}.py"), "--seed", str(seed)],
                       cwd=os.path.dirname(HIER), capture_output=True, text=True)
    with open(os.path.join(HIER, "roh", f"{kid}_s{seed}.log"), "w") as f: f.write(r.stdout + r.stderr)
    return kid, seed, "ok" if r.returncode == 0 else f"FEHLER {r.returncode}", time.time() - t
if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--seeds", type=int, nargs="*", default=[950, 951, 952, 953, 954])
    a.add_argument("--nur", nargs="*"); a.add_argument("--prozesse", type=int, default=int(os.environ.get("INSTITUT_PROZESSE", "4")))
    o = a.parse_args(); os.makedirs(os.path.join(HIER, "roh"), exist_ok=True)
    kids = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HIER, "[KE][0-9][0-9]*.py")))
    if o.nur: kids = [k for k in kids if k in o.nur]
    auf = [(k, s) for k in kids for s in o.seeds]
    print(f"{len(auf)} Aufgaben, {o.prozesse} Prozesse", flush=True)
    with ProcessPoolExecutor(o.prozesse) as ex:
        for f in as_completed([ex.submit(eine, k, s) for k, s in auf]):
            k, s, st, dt = f.result(); print(f"{k} s{s}: {st} ({dt/60:.1f} min)", flush=True)
