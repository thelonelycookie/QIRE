"""W5-Vortest (Phase 2, Runde 1): Bitflips auf gespeicherten Bits, Rotation gegen Clipping gegen g16.
ERKUNDUNG, kein Befund. Nur Tuning-Seeds (900, 901). Aufruf: OMP_NUM_THREADS=2 python3 w5_vortest.py
"""
import sys, json, time, math
from pathlib import Path
import numpy as np, torch
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parents[1] / "labor"))
import gemeinsam as G

MODELL_SEEDS = [900, 901]          # Tuning-Seeds (Konvention LABORBUCH 900-903)
BITS = [2, 4]
PS = [1e-5, 1e-4, 1e-3, 1e-2]
N_FLIP = 20
MODI = ["alle", "nur_codes", "nur_skalen"]   # Idee 2: Flipfläche getrennt
H = 256
torch.set_num_threads(2)

def had(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)

def quant_zeile(w, L, skala):
    """Mittpunkt-Gitter, Codes 0..L-1, Wert=(c-(L-1)/2)*step. w:(Gruppen,k). skala:(Gruppen,1) fp16 = step."""
    step = skala.float()
    c = torch.round(w / step + (L - 1) / 2).clamp(0, L - 1)
    return c.to(torch.uint8)

def deq(c, skala, L):
    return (c.float() - (L - 1) / 2) * skala.float()

def skala_absmax(w, L):
    return (w.abs().amax(1, keepdim=True) / ((L - 1) / 2)).clamp_min(1e-8)

def skala_opt(w, L):
    """MSE-optimales Clip je Gruppe: Suche über Faktoren des absmax."""
    am = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
    best, bs = None, None
    for f in torch.linspace(0.2, 1.0, 33):
        s = (am * f / ((L - 1) / 2)).half()
        e = ((deq(quant_zeile(w, L, s), s, L) - w) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs = e, s.clone()
        else:
            m = e < best
            best = torch.where(m, e, best); bs = torch.where(m, s, bs)
    return bs

# Arme: (name, rotation, scheme)   scheme: tensor | zeile_abs | zeile_opt | g16
ARME = {
    "d_roh":            (False, "tensor"),
    "b_clip_zeile":     (False, "zeile_opt"),
    "a_had_zeile_opt":  (True,  "zeile_opt"),
    "a0_had_zeile_abs": (True,  "zeile_abs"),
    "b0_zeile_abs":     (False, "zeile_abs"),
    "c_g16":            (False, "g16"),
    "c_g16_bytegleich": (False, "g16"),   # b-1 Bit Codes (+1 Bit/Gewicht Skala = ca. b Bit)
}

def bauen(Ws, bs, arm, b):
    rot, sch = ARME[arm]
    bb = b - 1 if arm == "c_g16_bytegleich" else b
    L = 2 ** bb
    lagen = []; nbyte = 0
    for li, (W, bias) in enumerate(zip(Ws, bs)):
        n_in = W.shape[1]
        sg = None
        if rot:
            g = torch.Generator().manual_seed(1234 + li)
            sg = (torch.randint(0, 2, (n_in,), generator=g) * 2 - 1).float()   # Vorzeichen aus Seed = 0 B
            Hm = had(n_in) if n_in == 256 else None
            Wq = (W * sg) @ Hm  # W D H
        else:
            Wq = W
        if sch == "tensor":
            s = skala_absmax(Wq.reshape(1, -1), L).half(); cw = quant_zeile(Wq.reshape(1, -1), L, s); shp = Wq.shape
            code, sk = cw.reshape(-1), s
        elif sch == "g16":
            wg = Wq.reshape(-1, 16); s = skala_absmax(wg, L).half()
            code, sk = quant_zeile(wg, L, s).reshape(-1), s
        else:
            s = (skala_abs(Wq, L) if sch == "zeile_abs" else skala_opt(Wq, L)).half()
            code, sk = quant_zeile(Wq, L, s).reshape(-1), s
        lagen.append(dict(code=code, bits=bb, L=L, sk=sk.half().clone(), bias=bias.half().clone(), shape=tuple(Wq.shape),
                          sch=sch, rot=rot, sg=sg))
        nbyte += math.ceil(code.numel() * bb / 8) + sk.numel() * 2 + bias.numel() * 2
    return lagen, nbyte

def skala_abs(w, L):
    return skala_absmax(w, L)

def rekonstruiere(lag, flips=None):
    Ws, bs = [], []
    for l in lag:
        code, sk, bias = l["code"], l["sk"], l["bias"]
        if flips is not None:
            code, sk, bias = flips(l)
        L = l["L"]; shp = l["shape"]
        sk = torch.nan_to_num(sk.float(), nan=0.0, posinf=65504., neginf=-65504.)
        bias = torch.nan_to_num(bias.float(), nan=0.0, posinf=65504., neginf=-65504.)
        if l["sch"] == "tensor":
            w = deq(code.reshape(1, -1), sk.reshape(1, 1), L).reshape(shp)
        elif l["sch"] == "g16":
            w = deq(code.reshape(-1, 16), sk, L).reshape(shp)
        else:
            w = deq(code.reshape(shp), sk, L)
        if l["rot"]:
            w = (w @ had(shp[1])) * l["sg"]   # (W D H) H D^-1 ; H symmetrisch, orthonormal -> (WDH)H=WD, dann *D
        Ws.append(w); bs.append(bias)
    return Ws, bs

def vor(Ws, bs, x):
    h = torch.relu(x @ Ws[0].T + bs[0]); h = torch.relu(h @ Ws[1].T + bs[1])
    return h @ Ws[2].T + bs[2]

def flip_u(t, bits, p, gen, maske):
    """Flip auf Bits eines uint8-Tensors (nur niedrige `bits` Bits) bzw. int16-Sicht."""
    if p <= 0: return t
    out = t.clone()
    for k in range(bits):
        m = torch.rand(t.shape, generator=gen) < p
        out = out ^ (m.to(out.dtype) << k)
    return out

def flip_fp16(t, p, gen):
    v = t.contiguous().view(torch.int16).clone()
    for k in range(16):
        m = torch.rand(v.shape, generator=gen) < p
        v = v ^ (m.to(torch.int16) << k)
    return v.view(torch.float16)

def main():
    t0 = time.time()
    erg = {"meta": dict(modell_seeds=MODELL_SEEDS, bits=BITS, ps=PS, n_flip=N_FLIP, modi=MODI, hidden=H,
                        hinweis="ERKUNDUNG, kein Befund; nur Tuning-Seeds"), "zeilen": []}
    for ms in MODELL_SEEDS:
        x, y, xt, yt = G.bilder("mnist", ms)
        torch.manual_seed(ms)
        net = torch.nn.Sequential(torch.nn.Linear(256, H), torch.nn.ReLU(), torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, 10))
        G.trainiere(net, x, y, schritte=1500, lr=0.003, seed=ms)
        with torch.no_grad():
            acc_fp = G.genauigkeit(net(xt), yt)
            Ws = [net[i].weight.detach().clone() for i in (0, 2, 4)]
            bs = [net[i].bias.detach().clone() for i in (0, 2, 4)]
            print(f"seed {ms} fp32 acc {acc_fp:.4f}", flush=True)
            # # alle Lagen haben 256 Eingänge
            for b in BITS:
                for arm in ARME:
                    lag, nb = bauen(Ws, bs, arm, b)
                    a0 = G.genauigkeit(vor(*rekonstruiere(lag), xt), yt)
                    for modus in MODI:
                        for p in PS:
                            accs = []
                            for r in range(N_FLIP):
                                gen = torch.Generator().manual_seed(10_000 * ms + 100 * r + int(-math.log10(p)) + 7 * b)
                                def fl(l, gen=gen, p=p, modus=modus):
                                    c, s, bi = l["code"], l["sk"], l["bias"]
                                    if modus in ("alle", "nur_codes"):
                                        c = flip_u(c, l["bits"], p, gen, None)
                                    if modus in ("alle", "nur_skalen"):
                                        s = flip_fp16(s, p, gen); bi = flip_fp16(bi, p, gen)
                                    return c, s, bi
                                accs.append(G.genauigkeit(vor(*rekonstruiere(lag, fl), xt), yt))
                            erg["zeilen"].append(dict(modell_seed=ms, bits=b, arm=arm, modus=modus, p=p, bytes=nb,
                                                      acc_ohne_flip=a0, acc_fp32=acc_fp, accs=accs))
                    print(f"  seed {ms} b={b} {arm}: Bytes {nb}, acc ohne Flip {a0:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    (HIER / "W5_vortest.json").write_text(json.dumps(erg))
    schreibe_md(erg, time.time() - t0)

def schreibe_md(erg, dauer):
    Z = erg["zeilen"]
    L = ["# W5-Vortest: Bitflips auf gespeicherten Bits (Phase 2, Runde 1)", "",
         "**Status: Erkundung, kein Befund.** Nur Tuning-Seeds 900/901 (je 1 Modell, 20 Flipmuster je Zelle, also n=40 je Zelle). "
         f"MNIST 16x16 (`gemeinsam.bilder`), MLP 256-256-256-10, 1500 Adam-Schritte, Test 5000 Bilder. Laufzeit {dauer/60:.1f} min.", "",
         "Fehlermodell: jedes gespeicherte Bit kippt unabhängig mit p: Gewichtscodes (b Bit), fp16-Skalen UND fp16-Bias. "
         "fp16-Werte, die durch Flips NaN/Inf werden, werden auf 0 bzw. ±65504 gesetzt (keine NaN-Ausbreitung; das ist günstig für Arme mit vielen Skalen). "
         "Modus `alle` = Hauptergebnis; `nur_codes` / `nur_skalen` trennen die Flipfläche (Idee 2).", "",
         "Arme: d_roh = eine Skala je Schicht; b_clip_zeile = Zeilenskala, MSE-optimales Clip (33 Faktoren); "
         "a_had_zeile_opt = Hadamard (Zufallsvorzeichen aus Seed, 0 B) + Zeilenskala mit optimalem Clip; "
         "a0 / b0 = wie a / b, aber ohne Clip (absmax); c_g16 = Gruppenskala fp16 je 16 Gewichte (1 Bit/Gewicht Skala, also deutlich mehr Bytes); "
         "c_g16_bytegleich = g16 mit b-1 Bit Codes (≈ gleiche Bytes). Mittpunkt-Gitter, binäre Codes. "
         "Hinweis: a und b enthalten beide Clipping; die Rotation allein ist a vs. b, nicht a vs. d. Das Gitter für 2 Bit hat 4 Stufen, g16_bytegleich bei 2 Bit nur 2 Stufen (±).", ""]
    for modus in MODI:
        for b in BITS:
            L += [f"## Modus `{modus}`, {b} Bit", ""]
            arme = list(ARME)
            L += ["| Arm | Bytes | acc ohne Flip | " + " | ".join(f"p={p:g}" for p in PS) + " |", "|---|---|---|" + "---|" * len(PS)]
            for arm in arme:
                zz = [z for z in Z if z["arm"] == arm and z["bits"] == b and z["modus"] == modus]
                by = zz[0]["bytes"]; a0 = np.mean([z["acc_ohne_flip"] for z in zz if z["p"] == PS[0]])
                cells = []
                for p in PS:
                    v = np.array([a for z in zz if z["p"] == p for a in z["accs"]])
                    cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
                L.append(f"| {arm} | {by} | {a0:.3f} | " + " | ".join(cells) + " |")
            L.append("")
    fp = np.mean([z["acc_fp32"] for z in Z[::len(Z) // len(MODELL_SEEDS)]]) if Z else 0
    L += [f"fp32-Genauigkeit (Mittel über 2 Seeds): {fp:.3f}.", "",
          "Bytes pro Modell (Seed 900), inkl. Skalen und fp16-Bias; Hadamard-Vorzeichen aus Seed erzeugt = 0 B (als gespeicherter Vektor wären es n/8 = 32 B je Schicht).", "",
          "Grenzen: 2 Modelle, 1 Datensatz, kleines MLP, kein RandBET-Training, keine Signifikanztests; Std über alle 40 Flipmuster (Modell+Muster gemischt). "
          "Nur Vortest; Aussagen über Rotation vs. Clipping bleiben Hypothese."]
    (HIER / "W5_vortest.md").write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
