"""K07 (Erkundung, kein Befund): Optimiererzustand (Adams m und v) rotiert und mit 2-8 Bit gespeichert. Byte-LM d=128, 4 Lagen (kb.LM), gleiche Initialisierung und Batches in allen Armen.
Mass: Test-bpc gegen Bytes des Optimiererzustands. Parameter, Zustand-Bytes und Trainings-FLOPs werden ausgewiesen (FLOPs je Schritt bei allen Adam-Armen gleich; Rotation/Rundung der Zustaende kommt als
Zusatz-Operationen hinzu: ideal gezaehlt 2·(Hadamard 64: 6 Add. je Wert)+Rundung ~4 je Wert und Schritt).
Zustand (Tensoren >= 4096 Werte, Bloecke zu 64; kleine Tensoren (LayerNorm, Bias) bleiben fp32 und zaehlen als Bytes):
  m: Block gedreht (optional, Zufalls-Vorzeichen-Hadamard 64), absmax-Skala fp16 je Block, b Bit, stochastisches Runden (gesaeter Generator), zurueckgedreht
  v: u = sqrt(v); Blockmittel mu (fp16) abgezogen, Rest wie m behandelt (gedreht/ungedreht, b Bit, Skala fp16), u_hat = max(mu + Rest, 0,1·mu), v_hat = u_hat²
  (Roh- und Rotationsarm benutzen genau denselben Code, nur die Drehung unterscheidet sich.)  Bytes je Wert: m: b+0,25 Bit, v: b+0,5 Bit.
  Aktualisierung wie bei 8-Bit-Adam: neuer Zustand voll praezise berechnet, Schritt damit, danach neu gerundet und gespeichert.
Arme:  adam32 | b8 (8-Bit-Block, Kontrolle) | b4roh | b4rot | b2roh | b2rot | adafactor (faktorisiertes v, kein m; lr 1e-3 und 3e-3: die haertere Kontrolle in Bytes)
Alle Adam-Arme mit lr 2e-3 (Aufwaermen 5 %, Kosinus, Clip 1,0 wie kb.trainiere_lm), Adafactor 2 Raten. T = 2500 (schnell: 50, klein).
Aufruf: python3 erkundung/K07.py --seed 950 [--schnell]"""
import math
import time

import _hilfe_k0508 as H
import torch

kb = H.kb
LR_ADAM = 2e-3
LR_ADAF = (1e-3, 3e-3)
T_VOLL, T_SCHNELL = 2500, 50
BLOCK = 64
MIN_N = 4096


class ZustandsCodec:
    """Blockweise b-Bit-Speicherung, optional gedreht; liefert den gespeicherten (dequantisierten) Wert."""

    def __init__(self, bits, rot, seed):
        self.bits, self.rot = bits, rot
        self.qmax = 2 ** (bits - 1) - 1
        self.Q = kb.rot_matrix(BLOCK, seed) if rot else None
        self.g = torch.Generator().manual_seed(seed + 7)

    def _rund(self, y):
        """y: (nb, 64) -> dequantisiert (nb, 64); stochastisch gerundet."""
        if self.Q is not None:
            y = y @ self.Q.T
        s = y.abs().amax(1, keepdim=True).clamp_min(1e-30).half().float().clamp_min(1e-30)
        z = y / s * self.qmax
        z = torch.floor(z + torch.rand(z.shape, generator=self.g)).clamp(-self.qmax, self.qmax)
        yq = z / self.qmax * s
        if self.Q is not None:
            yq = yq @ self.Q
        return yq

    def m(self, x):
        return self._rund(x.reshape(-1, BLOCK)).reshape(x.shape)

    def v(self, x):
        u = x.clamp_min(0).sqrt().reshape(-1, BLOCK)
        mu = u.mean(1, keepdim=True).half().float()
        uh = (mu + self._rund(u - mu)).clamp_min(0.1 * mu)
        return (uh ** 2).reshape(x.shape)


class AdamZ:
    """Adam (beta 0,9/0,999, eps 1e-8, Biaskorrektur) mit optional quantisiertem Zustand."""

    def __init__(self, netz, bits=None, rot=False, seed=0):
        self.p = [p for p in netz.parameters()]
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]
        self.t = 0
        self.bits = bits
        self.q = [bits is not None and p.numel() >= MIN_N and p.numel() % BLOCK == 0 for p in self.p]
        self.cod = [ZustandsCodec(bits, rot, seed * 1009 + k) if q else None for k, (p, q) in enumerate(zip(self.p, self.q))]

    def step(self, lr):
        self.t += 1
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        with torch.no_grad():
            for k, p in enumerate(self.p):
                if p.grad is None:
                    continue
                g = p.grad
                m = b1 * self.m[k] + (1 - b1) * g
                v = b2 * self.v[k] + (1 - b2) * g * g
                p.sub_(lr * (m / c1) / ((v / c2).sqrt() + 1e-8))
                if self.q[k]:
                    m, v = self.cod[k].m(m), self.cod[k].v(v)
                self.m[k], self.v[k] = m, v

    def zustand_bytes(self):
        by = 0.0
        for p, q in zip(self.p, self.q):
            n = p.numel()
            by += (n * (self.bits + 0.25) / 8 + n * (self.bits + 0.5) / 8) if q else 8.0 * n
        return by


class Adafactor:
    """Adafactor ohne Moment (beta1=None), faktorisiertes v fuer 2D-Tensoren, beta2_t = 1 - t^-0,8, Update-Clipping d=1, feste Lernrate."""

    def __init__(self, netz):
        self.p = [p for p in netz.parameters()]
        self.t = 0
        self.R = [torch.zeros(p.shape[0]) if p.ndim == 2 else None for p in self.p]
        self.C = [torch.zeros(p.shape[1]) if p.ndim == 2 else None for p in self.p]
        self.V = [torch.zeros_like(p) if p.ndim != 2 else None for p in self.p]

    def step(self, lr):
        self.t += 1
        b2 = 1 - self.t ** -0.8
        with torch.no_grad():
            for k, p in enumerate(self.p):
                if p.grad is None:
                    continue
                g2 = p.grad ** 2 + 1e-30
                if p.ndim == 2:
                    self.R[k] = b2 * self.R[k] + (1 - b2) * g2.mean(1)
                    self.C[k] = b2 * self.C[k] + (1 - b2) * g2.mean(0)
                    vh = self.R[k][:, None] * self.C[k][None, :] / self.R[k].mean().clamp_min(1e-30)
                else:
                    self.V[k] = b2 * self.V[k] + (1 - b2) * g2
                    vh = self.V[k]
                u = p.grad / vh.sqrt().clamp_min(1e-30)
                u = u / max(1.0, float(u.pow(2).mean().sqrt()))
                p.sub_(lr * u)

    def zustand_bytes(self):
        return float(sum(4.0 * (p.shape[0] + p.shape[1]) if p.ndim == 2 else 4.0 * p.numel() for p in self.p))


def arme(seed):
    return {
        "adam32": (LR_ADAM, lambda n: AdamZ(n)),
        "b8": (LR_ADAM, lambda n: AdamZ(n, 8, False, seed)),
        "b4roh": (LR_ADAM, lambda n: AdamZ(n, 4, False, seed)),
        "b4rot": (LR_ADAM, lambda n: AdamZ(n, 4, True, seed)),
        "b2roh": (LR_ADAM, lambda n: AdamZ(n, 2, False, seed)),
        "b2rot": (LR_ADAM, lambda n: AdamZ(n, 2, True, seed)),
        "adafactor_lr1e-3": (LR_ADAF[0], lambda n: Adafactor(n)),
        "adafactor_lr3e-3": (LR_ADAF[1], lambda n: Adafactor(n)),
    }


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    klein = schnell
    T = T_SCHNELL if schnell else T_VOLL
    zeilen = {}
    for name, (lr, ctor) in arme(seed).items():
        holder = {}

        def fn(n, ctor=ctor, holder=holder):
            holder["o"] = ctor(n)
            return holder["o"]

        t1 = time.time()
        netz, inf = H.lm_trainiere(seed, T, lr, fn, klein=klein)
        z = {"lr": lr, "info": inf, "zustand_bytes": holder["o"].zustand_bytes(), "sekunden": time.time() - t1}
        if not inf["instabil"]:
            z["val_bpc"] = H.bpc(netz, klein, "val")
            z["test_bpc"] = H.bpc(netz, klein, "test")
        zeilen[name] = z
        print(f"  {name}: {z.get('test_bpc')} bpc, Zustand {z['zustand_bytes'] / 1e6:.3f} MB, {z['sekunden']:.0f} s", flush=True)
    netz = kb.LM(128, 4)
    fl, mac = H.lm_flops_schritt(netz)
    n = H.n_param(netz)
    ref = zeilen["adam32"].get("test_bpc")
    r = {"kid": "K07", "seed": seed, "schnell": schnell, "T": T, "klein": klein, "parameter": n, "mac_je_token": mac, "trainings_flops_je_schritt": fl,
         "trainings_flops_gesamt": fl * T, "zusatz_ops_je_schritt_ideal": int(n * (2 * 6 + 4) * 2),
         "hinweis": "FLOPs der Vorwaerts/Rueckwaerts-Rechnung in allen Armen gleich; Adam-Arme: Zustand-Bytes fp32 = 8 B/Param; Adafactor ohne m", "arme": zeilen,
         "abstand_zu_adam32_bpc": {k: (v["test_bpc"] - ref) if (ref is not None and "test_bpc" in v) else None for k, v in zeilen.items()}}
    r["zusammenfassung"] = "\n".join(f"{k}: test {v.get('test_bpc')} bpc, Zustand {v['zustand_bytes'] / 1e6:.3f} MB" for k, v in zeilen.items())
    r["sekunden"] = time.time() - t0
    return r


if __name__ == "__main__":
    H.cli("K07", main)
