"""Z5_bitfehler -- Z4-12: Verteilt Rotation Speicher-Bitfehler? Gespeicherte 2-/3-Bit-Gewichte mit zufälligen Bitflips, roh gegen rotiert (Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
ANLASS (Z4_ideen Z4-12, Rang 1): Hypothese: Ein Flip im höchsten Code-Bit zerstört roh einen Kanal (Absmax-Skala, Ausreißer); rotiert verteilt sich der Schaden gleichmäßig und klein (graceful degradation). Langweilig: Gruppenskala (kleinerer Schaden je Flip) oder
  nur die Skala zählt, nicht die Basis. QuaRot/SpinQuant testen keine Bitflips; hier nur Messung, kein Neutraining.
MODELLE: MLP MNIST 784-512-512-10 (Epochen Z5_EP, Standard 6; SCHNELL 2) und Byte-LM d=128/f=512, 4 Lagen (Schritte Z5_BT, Standard 800; SCHNELL 60); jeweils EIN fp-Netz (Seed), danach nur Quantisieren, Flippen, Auswerten. Alle nn.Linear werden gespeichert (PTQ).
SPEICHERFORMAT (Bits): Code je Gewicht (b Bit, no-zero-Mittelpunktsgitter c in 0..2^b-1, Wert (c - 2^(b-1) + 0,5)*s), Skala fp16 je Zeile bzw. je 32 Eingänge (MSE-Suche, dann fp16-gerundet).
ARME:
  roh_g32     Gruppenskala g=32, keine Rotation (fairer Pflichtgegner)       roh_zeile   Zeilenskala, keine Rotation (Referenz)
  rot_zeile   Vorzeichen*Hadamard (auf 2er-Potenz aufgefüllt), Zeilenskala     rot_g32     Vorzeichen*Hadamard + Gruppenskala g=32
  zufall      Zufalls-Orthogonale (Haar, dicht, Breite = Eingang), Zeilenskala (Kontrolle: nur Hadamard-Struktur oder jede Rotation?)
  Bytes sind NICHT gleich (Skalen/Auffüllung unterscheiden sich, Feld "bytes_je_arm"); die Frage ist die Robustheit je Fehlerrate, nicht Qualität bei gleichen Bytes.
FEHLER: Bitflip-Rate p in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, jedes gespeicherte Bit unabhängig. Zwei Varianten: "code" (nur Code-Bits der Gewichte) und "alles" (zusätzlich die 16 Bit jeder fp16-Skala; nicht endliche Skala -> 0).
  Wiederholungen: MLP 20, LM 3 (SCHNELL 3 bzw. 2). Gleiches Fehlermuster-Seed je (Wiederholung, Schicht) für alle Arme, soweit die Form gleich ist (sonst unabhängig).
AUSWERTUNG: tabelle[modell][bits][variante][arm][p] = {q_mittel (Genauigkeit in % bzw. Val-bpc), q_std}; q0 = ohne Fehler; verlust = q0 - q (pp) bzw. bpc_mit_Fehler - bpc_ohne (bpc).
AMPEL (Einzelseed, vorläufig): "ampel_<mlp|lm>_b<bits>_<code|alles>": gewinn = Minimum über p in {1e-3, 3e-3, 1e-2} von [min Verlust(roh_g32, roh_zeile) - min Verlust(rot_zeile, rot_g32)] (pp bzw. bpc); Schwelle 0,5 pp bzw. 0,01 bpc; grün = rot-Arm verliert bei gleicher Rate klar weniger.
  Zusatz "verhaeltnis_1e-2" = Verlust bester roh / Verlust bester rot bei p = 1e-2 (Hypothese Z4-12: >= 2) und "zufall_gegen_hadamard" (Verlust bester Hadamard - Verlust zufall, negativ = Zufall besser).
VORAB-P (vor dem Lauf): Z4-12-Hypothese (rot verliert bei p = 1e-2 mindestens 2x weniger als roh) 55 % laut Idee; ich schätze nach Z3 (Rotation = additives Rauschen, g32 gleich gut): Ampel grün MLP 30 %, LM 25 %; Gruppenskala (roh_g32) leistet dasselbe wie Rotation 40 %;
  Variante "alles": rot_zeile schlägt roh_g32 wegen weniger Skalenbits 45 %; Zufallsorthogonale ≈ Hadamard (Verlust innerhalb 20 %) 75 %.
Lies zuerst: bytes_je_arm, ampel_*, dann verlust und tabelle."""
from __future__ import annotations
import os
os.environ.setdefault("Z2_VAL", "8192")
from Z3_bytegleich import pow2, lm_formen, lm_rest, linears
from Z4_bitkurve import quant_L
from _z2 import *

BITS = (2, 3)
RATEN = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
RATEN_AMPEL = (1e-3, 3e-3, 1e-2)
ARME = ("roh_g32", "roh_zeile", "rot_zeile", "rot_g32", "zufall")
HAD = ("rot_zeile", "rot_g32")
ROH = ("roh_g32", "roh_zeile")
VARIANTEN = ("code", "alles")
EPOCHEN = int(os.environ.get("Z5_EP", "6"))
SCHRITTE = int(os.environ.get("Z5_BT", "800"))


# ------------------------------------------------------------------ Kodierung
def kodiere(X, bits, g):
    """X (o x n) -> Codes (o x ng x g, int16), Skalen fp16-gerundet (o x ng x 1). g=None: Zeilenskala."""
    L = 2 ** bits
    o, n = X.shape
    g = g or n
    ng = math.ceil(n / g)
    Xg = F.pad(X, (0, ng * g - n)).reshape(o, ng, g)
    _, s = quant_L(Xg, L, 12)
    s = s.half().float().clamp_min(1e-12)
    c = ((Xg / s - 0.5).round() + L // 2).clamp(0, L - 1).to(torch.int16)
    return c, s


def dekodiere(c, s, bits, n):
    o = c.shape[0]
    return ((c.float() - 2 ** bits / 2 + 0.5) * s).reshape(o, -1)[:, :n]


def flip_codes(c, bits, p, gen):
    if p <= 0:
        return c
    m = torch.zeros_like(c)
    for k in range(bits):
        m |= ((torch.rand(c.shape, generator=gen) < p).to(torch.int16) << k)
    return c ^ m


def flip_skala(s, p, gen):
    if p <= 0:
        return s
    h = s.half().contiguous().view(torch.int16)
    m = torch.zeros_like(h)
    for k in range(16):
        m |= ((torch.rand(h.shape, generator=gen) < p).to(torch.int16) << k)
    r = (h ^ m).view(torch.float16).float()
    return torch.where(torch.isfinite(r), r, torch.zeros_like(r))


class Gespeichert:
    """Alle Linearschichten eines fp-Netzes im Speicherformat eines Arms."""

    def __init__(self, netz, arm, bits, seed):
        self.arm, self.bits = arm, bits
        self.rot = "rot" in arm
        self.zuf = arm == "zufall"
        g = 32 if arm.endswith("g32") else None
        self.netz = copy.deepcopy(netz).eval()
        self.lins, self.enc, self.bytes = linears(self.netz), [], 0.0
        for k, m in enumerate(self.lins, 1):
            W = m.weight.data
            o, i = W.shape
            if self.rot:
                sg = vorzeichen(pow2(i), seed * 101 + k)
                X, Q = fwht(F.pad(W, (0, len(sg) - i)) * sg), None
            elif self.zuf:
                sg, Q = None, kb.orth_matrix(i, seed * 101 + k)
                X = W @ Q.T
            else:
                sg, Q, X = None, None, W
            c, s = kodiere(X, bits, g)
            self.enc.append({"c": c, "s": s, "sg": sg, "Q": Q, "i": i, "n": X.shape[1]})
            self.bytes += c.numel() * bits / 8 + s.numel() * 2 + 2 * o + (len(sg) / 8 if sg is not None else 0)

    def setze(self, p, mit_skala, rep, seed):
        for k, (m, e) in enumerate(zip(self.lins, self.enc), 1):
            gen = torch.Generator().manual_seed(seed * 10007 + rep * 131 + k)
            c = flip_codes(e["c"], self.bits, p, gen)
            s = flip_skala(e["s"], p, gen) if mit_skala else e["s"]
            Xq = dekodiere(c, s, self.bits, e["n"])
            if e["sg"] is not None:
                Xq = (fwht(Xq) * e["sg"])[:, :e["i"]]
            elif e["Q"] is not None:
                Xq = Xq @ e["Q"]
            m.weight.data = Xq.contiguous()
        return self.netz


def lm_bauen(d, f):
    return LMX([FFNDichtX(d, f) for _ in range(4)], d=d)


def verlust_tab(tab, modell):
    """tab[arm][p] = {q_mittel}; Verlust relativ zum fehlerfreien Wert des Arms (pp bzw. bpc, größer = schlechter)."""
    return {a: {p: (tab[a][0.0]["q_mittel"] - v["q_mittel"]) if modell == "mlp" else (v["q_mittel"] - tab[a][0.0]["q_mittel"]) for p, v in tab[a].items()} for a in tab}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    ep = 2 if schnell else EPOCHEN
    T = 60 if schnell else SCHRITTE
    reps = {"mlp": 3 if schnell else 20, "lm": 2 if schnell else 3}
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"epochen": ep, "schritte": T, "raten": RATEN, "reps": reps, "val_bytes": VAL_N, "tabelle": {"mlp": {}, "lm": {}}, "verlust": {"mlp": {}, "lm": {}}, "bytes_je_arm": {"mlp": {}, "lm": {}}, "fp": {}}
    t0 = time.time()
    torch.manual_seed(seed)
    mlp = kb.baue_bild([512, 512])
    mlp_train(mlp, xtr, ytr, seed, ep, 1e-3)
    out["fp"]["mlp"] = acc(mlp, xva, yva)
    torch.manual_seed(seed)
    lm = lm_bauen(128, 512)
    r = lm_trainiere(lm, seed, T, schnell=schnell)
    lm.eval()
    out["fp"]["lm"] = val_bpc(lm, schnell)
    out["fp"]["lm_instabil"] = bool(r["instabil"])
    print(f"Z5_bitfehler s{seed}: fp32 MLP {out['fp']['mlp']:.4f}, LM {out['fp']['lm']:.4f} bpc ({time.time() - t0:.0f}s)", flush=True)

    def lm_mess(n):
        v = val_bpc(n, schnell)
        return v if v == v and v < 8.0 else 8.0
    mess = {"mlp": lambda n: 100 * acc(n, xva, yva), "lm": lm_mess}
    for modell, netz in (("mlp", mlp), ("lm", lm)):
        for bits in BITS:
            out["tabelle"][modell][bits], out["verlust"][modell][bits], out["bytes_je_arm"][modell][bits] = {}, {}, {}
            gs = {a: Gespeichert(netz, a, bits, seed) for a in ARME}
            out["bytes_je_arm"][modell][bits] = {a: g.bytes for a, g in gs.items()}
            for var in VARIANTEN:
                tab = {a: {} for a in ARME}
                for a, g in gs.items():
                    for p in (0.0,) + RATEN:
                        n_ = 1 if p == 0 else reps[modell]
                        qs = [mess[modell](g.setze(p, var == "alles", rep, seed)) for rep in range(n_)]
                        tab[a][p] = {"q_mittel": float(np.mean(qs)), "q_std": float(np.std(qs)), "q_werte": [float(q) for q in qs]}
                out["tabelle"][modell][bits][var] = tab
                vl = verlust_tab(tab, modell)
                out["verlust"][modell][bits][var] = vl
                einheit = "pp" if modell == "mlp" else "bpc"
                schw = 0.5 if modell == "mlp" else 0.01
                je_p = {p: min(vl[a][p] for a in ROH) - min(vl[a][p] for a in HAD) for p in RATEN_AMPEL}
                a_ = ampel(min(je_p.values()), schw, einheit, f"{modell} {bits} Bit, Variante {var}: min Verlust roh - min Verlust Hadamard, Minimum über p in 1e-3..1e-2")
                a_["je_rate"] = je_p
                zr = min(vl[a][1e-2] for a in HAD)
                a_["verhaeltnis_1e-2"] = float(min(vl[a][1e-2] for a in ROH) / zr) if zr > 1e-6 else float("nan")
                a_["zufall_gegen_hadamard"] = {p: float(min(vl[a][p] for a in HAD) - vl["zufall"][p]) for p in RATEN}
                out[f"ampel_{modell}_b{bits}_{var}"] = a_
                print(f"Z5_bitfehler s{seed} {modell} b{bits} {var}: q0 " + " ".join(f"{a} {tab[a][0.0]['q_mittel']:.3f}" for a in ARME) + f" ({time.time() - t0:.0f}s)", flush=True)
                for p in RATEN:
                    print(f"   p={p:g} Verlust " + " ".join(f"{a} {vl[a][p]:+.3f}" for a in ARME), flush=True)
                print(f"   AMPEL {a_['farbe']} {a_['gewinn']:+.4f} {einheit}, Verhältnis(1e-2) {a_['verhaeltnis_1e-2']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z5_bitfehler", main)
