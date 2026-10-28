"""K02 -- Aufgaben-Superposition per Vorzeichen-Schlüssel (Erkundung Welle 2, 28.10.2026; kein Befund, nicht in LABORBUCH).
FRAGE: Drei Aufgaben (MNIST, Fashion-MNIST, MNIST mit fester Pixelpermutation) in EINEM Gewichtssatz; ein fester Vorzeichen-Schlüssel je Aufgabe
  (W_t = diag(s_aus_t) W diag(s_ein_t), aus dem Seed, 0 Bytes) wählt die Aufgabe. Schlägt das drei getrennte Modelle gleicher Bytes?
ARME (alle MLP 784-h-h-10, ReLU; aufgabenspezifische Biases in ALLEN Armen und bezahlt):
  geteilt_k   ein Modell h=512, Schlüssel je Aufgabe                   (Hauptarm)
  geteilt_o   dasselbe ohne Schlüssel, nur aufgabenspezifische Biases  (Kontrolle: „Etikett“)
  getrennt_B  3 getrennte Modelle, Breite so, dass Gesamtbytes = geteilt (Hartkontrolle gleicher Bytes)
  getrennt_3x 3 getrennte Modelle h=512 (3x Bytes; obere Schranke, beschreibend)
TRAINING: 15 Epochen (--schnell: 2 auf 2000 Bildern), Adam lr 1e-3 (5 % Aufwärmen, Kosinus), Batch 128 je Aufgabe und Schritt (geteilt: die 3 Batches pro Schritt, ein Optimiererschritt auf der Summe der Verluste).
  Pixelpermutation aus Seed. Gemessen: Val-Genauigkeit je Aufgabe (5000 Bilder), nie Test.
BYTES: fp16 (2 B) je Gewicht und Bias; Schlüssel 0 B (Seed). FLOPs (MACs je Bild) gleich für geteilt und je ein getrenntes Modell h=512; getrennt_B hat weniger.
Lies zuerst: geteilt_k - geteilt_o (Schlüsselwirkung, mittlere Genauigkeit) und geteilt_k - getrennt_B (gleiche Bytes)."""
from __future__ import annotations
from _basis import *

AUFGABEN = ("mnist", "fashion", "mnist_perm")


class KMLP(nn.Module):
    """MLP mit geteilten Gewichten, aufgabenspezifischen Biases und optionalem Schlüssel (Vorzeichen je Aufgabe und Schicht)."""

    def __init__(self, h, nt, seed, schluessel):
        super().__init__()
        self.dims = [784, h, h, 10]
        self.ws = nn.ParameterList([nn.Parameter(torch.randn(o, i) * math.sqrt(2.0 / i)) for i, o in zip(self.dims[:-1], self.dims[1:])])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(nt, o)) for o in self.dims[1:]])
        self.sk = schluessel
        if schluessel:
            g = torch.Generator().manual_seed(seed * 313 + 7)
            for l, (i, o) in enumerate(zip(self.dims[:-1], self.dims[1:])):
                self.register_buffer(f"si{l}", (torch.randint(0, 2, (nt, i), generator=g) * 2 - 1).float())
                self.register_buffer(f"so{l}", (torch.randint(0, 2, (nt, o), generator=g) * 2 - 1).float())

    def forward(self, x, t):
        for l, W in enumerate(self.ws):
            if self.sk:
                x = x * getattr(self, f"si{l}")[t]
            x = F.linear(x, W) 
            if self.sk:
                x = x * getattr(self, f"so{l}")[t]
            x = x + self.bs[l][t]
            if l < len(self.ws) - 1:
                x = F.relu(x)
        return x


def param_k(h, nt):
    dims = [784, h, h, 10]
    return sum(i * o for i, o in zip(dims[:-1], dims[1:])), nt * sum(dims[1:])


def breite_gleiche_bytes(h_ziel, nt):
    ziel = sum(param_k(h_ziel, nt))
    return min(range(8, 2048), key=lambda h: abs(nt * sum(param_k(h, 1)) - ziel))


def aufgaben_daten(seed, klein):
    xm, ym, vm, wm = kb.bild_daten("mnist", klein)[:4]
    xf, yf, vf, wf = kb.bild_daten("fashion", klein)[:4]
    perm = torch.randperm(784, generator=torch.Generator().manual_seed(seed + 4242))
    return [(xm, ym, vm, wm), (xf, yf, vf, wf), (xm[:, perm], ym, vm[:, perm], wm)]


def trainiere_gruppe(modelle, daten, seed, ep, lr=1e-3, bs=128):
    """modelle: Liste (ein KMLP für alle Aufgaben, Liste der Länge 1) oder eins je Aufgabe. Je Schritt ein Batch je Aufgabe."""
    N = min(len(d[0]) for d in daten)
    spe = math.ceil(N / bs)
    n = ep * spe
    params = [p for m in modelle for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    warm = max(1, n // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm))))
    for m in modelle:
        m.train()
    for e in range(ep):
        ps = [torch.randperm(len(d[0]), generator=torch.Generator().manual_seed(seed * 7919 + e * 10 + t)) for t, d in enumerate(daten)]
        for k in range(spe):
            l = 0.0
            for t, d in enumerate(daten):
                b = ps[t][k * bs:(k + 1) * bs]
                m = modelle[0] if len(modelle) == 1 else modelle[t]
                l = l + F.cross_entropy(m(d[0][b], t if len(modelle) == 1 else 0), d[1][b])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()


def bewerte(modelle, daten):
    r = []
    for t, d in enumerate(daten):
        m = modelle[0] if len(modelle) == 1 else modelle[t]
        m.eval()
        with torch.no_grad():
            r.append(float((m(d[2], t if len(modelle) == 1 else 0).argmax(1) == d[3]).float().mean()))
    return r


def main(seed: int, schnell: bool = False) -> dict:
    ep = 2 if schnell else 15
    h = 128 if schnell else 512
    daten = aufgaben_daten(seed, schnell)
    nt = 3
    hB = breite_gleiche_bytes(h, nt)
    arme = {"geteilt_k": (h, True, 1), "geteilt_o": (h, False, 1), "getrennt_B": (hB, False, 3), "getrennt_3x": (h, False, 3)}
    out = {"ep": ep, "h_geteilt": h, "h_getrennt_B": hB, "lr": 1e-3, "arme": {}}
    for name, (hh, sk, anz) in arme.items():
        torch.manual_seed(seed)
        modelle = [KMLP(hh, 1 if anz == 3 else nt, seed + 17 * i, sk) for i in range(anz)]
        t0 = time.time()
        trainiere_gruppe(modelle, daten, seed, ep)
        a = bewerte(modelle, daten)
        w, b = param_k(hh, 1 if anz == 3 else nt)
        w, b = w * anz, b * anz
        out["arme"][name] = {"acc": dict(zip(AUFGABEN, a)), "acc_mittel": sum(a) / 3, "bytes_fp16": 2 * (w + b), "gewichte": w, "biases": b,
                             "macs_je_bild_und_aufgabe": param_k(hh, 1)[0], "sekunden": time.time() - t0}
        print(f"K02 s{seed} {name}: {[round(x, 4) for x in a]} mittel {sum(a)/3:.4f} Bytes {2*(w+b)}", flush=True)
    A = out["arme"]
    out["differenzen_acc_mittel"] = {"schluessel (geteilt_k-geteilt_o)": A["geteilt_k"]["acc_mittel"] - A["geteilt_o"]["acc_mittel"],
                                     "gleiche_Bytes (geteilt_k-getrennt_B)": A["geteilt_k"]["acc_mittel"] - A["getrennt_B"]["acc_mittel"],
                                     "vs_3x (geteilt_k-getrennt_3x)": A["geteilt_k"]["acc_mittel"] - A["getrennt_3x"]["acc_mittel"]}
    return out


if __name__ == "__main__":
    rahmen("K02", main)
