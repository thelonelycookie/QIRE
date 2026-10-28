"""E03 = Ergaenzung I-048  E3 (QAT-Paket): Ist der Vorteil der Rotation nur ein Nachher-Quantisierungs-Effekt?
MLP 784-512-512-10 (MNIST; --schnell: Teilmenge), 10 Epochen (statt 30: Zeitbudget, alle Arme gleich lang) Adam lr 1e-3 (identisch fuer alle Arme, gleiche Initialisierung/Batchreihenfolge), 2 Bit (kb: 3 Stufen, Skala je Zeile per MSE-Gitter), Eingang aufgefuellt 784->1024 beim Drehen.
Arme (jeweils roh = ohne Rotation, had = Vorzeichen-Hadamard in kb-Form; Bytes: had = I-048-Ziel, roh = Ziel minus Vorzeichenvektoren):
  fp32          gleiches Training ohne Quantisierung (Bezug)
  ptq           fp32 nachher quantisiert (kb.quantisiere; I-048-Arm)
  qat           Fake-Quant (STE) von Schritt 0 an; Skala je Zeile per MSE-Gitter, alle 100 Schritte neu gesucht, dazwischen gehalten
  zeno          fp32-Training, alle 64 Schritte harte Projektion der Hauptgewichte aufs Gitter (in der gedrehten Basis), am Ende ptq
  lin           weicher Uebergang W_eff = W + lam*(Wq - W) (STE), lam linear 0 -> 1 ueber die ersten 80 % der Schritte, dann 1
  luecke        wie lin, aber lam steigt nur um 0,02 je 50 Schritte, solange die Luecke L(lam=1) - L(lam=0) auf dem Batch < 0,05 nat ist (Lueckenregel); ab 70 % Zwangs-Anstieg, 1 spaetestens bei 90 %
  klein_fp16    kleines dichtes fp16-Modell gleicher Bytes (flach/zwei, 10 Epochen, lr 1e-3, Wahl nach Val): haerteste Klassik-Kontrolle
Gemessen wird immer das HART quantisierte Endmodell (lam = 1, Skala frisch gesucht). Bytes/MACs ausgewiesen.
Lesart: Bleibt had - roh bei qat/lin/luecke positiv, ist die Rotation mehr als Nachher-Quantisierung; schrumpft es auf ~0, war es ein PTQ-Effekt.
Kosten Raven (geschaetzt): ~35 min je Seed (8 Trainingslaeufe mit Fake-Quant). Aufruf: python3 erkundung/E03.py --seed 950 [--schnell]"""
import math
import numpy as np

import e_basis as eb
from e_basis import F, kb, nn, torch


def suche_f(Wr, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = Wr.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bf = None, None
    for f in np.linspace(0.4, 1.0, 13):
        s = amax * float(f) / qmax
        e = ((Wr - (Wr / s).round().clamp(-qmax, qmax) * s) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bf = e, torch.full_like(e, float(f))
        else:
            m = e < best
            best, bf = torch.where(m, e, best), torch.where(m, torch.full_like(e, float(f)), bf)
    return bf


def quant_f(Wr, bits, f):
    qmax = 2 ** (bits - 1) - 1
    s = Wr.abs().amax(1, keepdim=True).clamp_min(1e-12) * f / qmax
    return (Wr / s).round().clamp(-qmax, qmax) * s


class QLin(nn.Module):
    def __init__(self, lin, bits, rot, seed, refresh=100):
        super().__init__()
        self.weight, self.bias = nn.Parameter(lin.weight.data.clone()), nn.Parameter(lin.bias.data.clone())
        self.bits, self.lam, self.rot, self.refresh, self.zaehl, self.f = bits, 1.0, rot, refresh, 0, None
        o, i = self.weight.shape
        self.i = i
        if rot:
            self.ip = eb.pad_n(i, "voll")
            self.register_buffer("Q", eb.rot_block(self.ip, "voll", seed))

    def hart(self, neu_f=False):
        with torch.no_grad():
            W = self.weight.detach()
            Wr = F.pad(W, (0, self.ip - self.i)) @ self.Q.T if self.rot else W
            if self.f is None or neu_f or (self.training and self.zaehl % self.refresh == 0):
                self.f = suche_f(Wr, self.bits)
            self.zaehl += self.training
            Wq = quant_f(Wr, self.bits, self.f)
            return (Wq @ self.Q)[:, :self.i] if self.rot else Wq

    def forward(self, x):
        if self.lam > 0:
            W = self.weight + self.lam * (self.hart(neu_f=not self.training) - self.weight).detach()
        else:
            W = self.weight
        return F.linear(x, W, self.bias)

    def bytes(self):
        o, i = self.weight.shape
        return kb.bytes_q_linear(o, i, self.bits, self.rot, pad=self.rot)


def baue_q(netz_fp, bits, rot, seed):
    return kb.Netz([QLin(l, bits, rot, seed * 101 + k + 1) for k, l in enumerate(netz_fp.ls)])


def setze_lam(netz, v):
    for m in netz.ls:
        m.lam = v


def trainiere_q(netz, ds, seed, lr, ep, klein, modus, projiziere=None):
    """modus: fp | qat | lin | luecke | zeno. Gleiche Batchfolge/Plan wie kb.trainiere_bild."""
    xtr, ytr, _, _, _, _ = kb.bild_daten(ds, klein)
    N, bs = len(xtr), 128
    spe = math.ceil(N / bs)
    n = ep * spe
    cache = {"e": -1, "p": None, "lam": 0.0 if modus in ("lin", "luecke") else (1.0 if modus == "qat" else 0.0)}
    setze_lam(netz, cache["lam"])
    spur = []

    def schritt(s):
        e, k = divmod(s, spe)
        if cache["e"] != e:
            cache["p"] = torch.randperm(N, generator=torch.Generator().manual_seed(seed * 7919 + e))
            cache["e"] = e
        b = cache["p"][k * bs:(k + 1) * bs]
        if modus == "lin":
            cache["lam"] = min(1.0, s / (0.8 * n))
        elif modus == "luecke":
            if s % 50 == 0 and s > 0:
                with torch.no_grad():
                    setze_lam(netz, 1.0)
                    l1 = F.cross_entropy(netz(xtr[b]), ytr[b])
                    setze_lam(netz, 0.0)
                    l0 = F.cross_entropy(netz(xtr[b]), ytr[b])
                if float(l1 - l0) < 0.05:
                    cache["lam"] = min(1.0, cache["lam"] + 0.02)
                spur.append((s, float(l1 - l0), cache["lam"]))
            if s >= 0.7 * n:
                cache["lam"] = max(cache["lam"], min(1.0, (s - 0.7 * n) / (0.2 * n)))
        if modus in ("lin", "luecke"):
            setze_lam(netz, cache["lam"])
        if modus == "zeno" and s > 0 and s % 64 == 0:
            with torch.no_grad():
                for m in netz.ls:
                    m.weight.data = m.hart(neu_f=True)
        return F.cross_entropy(netz(xtr[b]), ytr[b])

    info = kb._schrittlauf(netz, n, lr, schritt, None, 0, {}, torch.Generator().manual_seed(seed + 5), clip=None)
    return info, spur


def auswerten_hart(netz, ds, klein):
    setze_lam(netz, 1.0)
    for m in netz.ls:
        m.f = None
    netz.eval()
    r = {"test": eb.acc(netz, ds, klein), "val": eb.acc(netz, ds, klein, "val"), "bytes": float(sum(m.bytes() for m in netz.ls))}
    return r


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein, lr, bits = P["klein"], 1e-3, 2
    ep = 3 if schnell else 10       # 10 statt 30 Epochen: Fake-Quant+Drehung kostet ~7x je Schritt, sonst > 45 min je Seed; ALLE Arme (auch fp32/klein) gleich lang
    ds = "mnist"
    out = {"profil": P, "bits": bits, "arme": {}}
    torch.manual_seed(seed)
    start = kb.baue_bild([512, 512])
    # ---- fp32-Bezug (gleiches Training) und ptq
    fp = eb.copy.deepcopy(start)
    kb.trainiere_bild(fp, ds, seed, lr, ep, klein=klein)
    out["fp32"] = {"test": eb.acc(fp, ds, klein), "val": eb.acc(fp, ds, klein, "val")}
    for rot, nm in ((False, "roh"), (True, "had")):
        nq = kb.quantisiere(fp, bits, "had" if rot else "roh", seed, pad_to={784: 1024})
        out["arme"][f"ptq_{nm}"] = {"test": eb.acc(nq, ds, klein), "val": eb.acc(nq, ds, klein, "val"), "bytes": float(sum(m.bytes() for m in nq.modules() if isinstance(m, kb.QLinear)))}
    # ---- QAT-Familie
    for modus in ("qat", "zeno", "lin", "luecke"):
        for rot, nm in ((False, "roh"), (True, "had")):
            t = eb.time.time()
            netz = baue_q(start, bits, rot, seed)
            info, spur = trainiere_q(netz, ds, seed, lr, ep, klein, modus)
            r = auswerten_hart(netz, ds, klein)
            r.update({"instabil": info["instabil"], "sekunden": round(eb.time.time() - t, 1)})
            if spur:
                r["luecke_spur"] = spur[::max(1, len(spur) // 40)]
            out["arme"][f"{modus}_{nm}"] = r
            eb.lg(f"[E03] {modus}_{nm}: test {r['test']:.4f} ({r['sekunden']} s)")
    # ---- Kontrolle: kleines dichtes fp16-Modell gleicher Bytes
    out["klein_fp16"] = eb.klein_bester(ds, kb.ziel_bytes(ds, bits), seed, ep, klein, formen=("flach", "zwei"), lrs=(lr,))
    out["differenzen_pp"] = {m: 100 * (out["arme"][f"{m}_had"]["test"] - out["arme"][f"{m}_roh"]["test"]) for m in ("ptq", "qat", "zeno", "lin", "luecke")}
    out["had_minus_klein_pp"] = {m: 100 * (out["arme"][f"{m}_had"]["test"] - out["klein_fp16"]["test"]) for m in ("ptq", "qat", "zeno", "lin", "luecke")}
    eb.lg("[E03] had-roh (pp):", {k: round(v, 2) for k, v in out["differenzen_pp"].items()})
    return out


if __name__ == "__main__":
    eb.cli("E03", main)
