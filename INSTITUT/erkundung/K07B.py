"""K07B -- Zusatzarme zu K07: 4-Bit-Adam nach Li, Chen, Zhu (NeurIPS 2023), blockweise, OHNE Rotation (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K07.py bleibt unverändert; AdamZ, arme(), Konstanten werden importiert. Gleiches Netz (kb.LM d=128, 4 Lagen), gleiche Initialisierung und Batches, lr 2e-3, T=2500 (schnell 50), Test-/Val-bpc, Bytes-Zählung wie K07 (kleine Tensoren < 4096 Werte bleiben fp32 und zählen 8 B/Wert).
LI-ARME (deterministische Rundung zum Nächsten, wie bei Li; K07 benutzt stochastisches Runden):
  li4         m: Block 128, Normierung durch Block-Absmax, 4 Bit vorzeichenbehaftet mit Dynamic-Exponent-Codebuch (3 Betragsbits: Null, 10^-e·linspace(0,1;n]-Stufen, nach Dettmers-Muster);
              v: Rang-1-Normierung (Zeilen-/Spaltenmaximum, Skala = min(r_i, c_j)) und lineare 4-Bit-Stufen OHNE Null {1/16..1} (löst das Nullpunktproblem); 1-D-Tensoren: Block 128.
  li4_sr      wie li4, aber STOCHASTISCHES Runden zwischen den Nachbarstufen (wie in K07); li4 deterministisch wie bei Li (Gefahr: v friert ein, weil 0,001·g² unter der halben Stufe liegt).
  li4_blockv_sr  wie li4_sr, aber v ebenfalls Block 128 + lineare Stufen ohne Null (trennt den Rang-1-Effekt)
Bytes: m: 4 Bit + bf16-Skala je 128 Werte (bf16 statt fp16, weil v ~ g² in fp16 unterläuft) (4,125 Bit); v li4: 4 Bit + fp16 je Zeile und Spalte; v blockweise 4,125 Bit.
ANKER aus K07 (gleiche Klassen/Seeds): adam32, b8, b4roh, b4rot. Details der Annäherung: das DE-Codebuch ist nicht bit-genau das von Li (dort 16 feste Werte); Lernrate nicht nachgetunt.
Lies zuerst: li4_sr - adam32, li4_sr - b4roh (Bytes beachten: b4roh 4,25/4,5 Bit je Wert), li4_sr - li4_blockv_sr; li4 (deterministisch) zeigt, ob das Einfrieren in diesem Aufbau real ist."""
import time

import _hilfe_k0508 as H
import torch
import K07 as K

kb = H.kb
BL = 128


def _codebuch_de():
    mags = [0.0]
    for e, nfrac in ((0, 2), (1, 1), (2, 0)):
        n = 2 ** nfrac
        mags += [10.0 ** -e * (0.1 + 0.9 * (k + 1) / n) for k in range(n)]
    m = sorted(set(mags))
    return torch.tensor(sorted(set([-x for x in m] + m)))        # 15 Werte, in [-1, 1]


DE = _codebuch_de()
LIN0 = torch.arange(1, 17).float() / 16                         # lineare Stufen ohne Null


GEN = torch.Generator().manual_seed(4711)
SR = [False]          # Umschalter: stochastisches Runden zwischen den zwei Nachbarstufen (wie K07)


def naechstes(x, cb):
    if SR[0]:
        hi = torch.bucketize(x.contiguous(), cb).clamp(1, len(cb) - 1)
        lo_v, hi_v = cb[hi - 1], cb[hi]
        p = ((x - lo_v) / (hi_v - lo_v)).clamp(0, 1)
        return torch.where(torch.rand(x.shape, generator=GEN) < p, hi_v, lo_v)
    mid = (cb[1:] + cb[:-1]) / 2
    return cb[torch.bucketize(x.contiguous(), mid)]


def blockweise(x, cb, absmax_signed):
    """Flach in Blöcke à 128 (aufgefüllt), Normierung durch Block-Absmax (fp16), Codebuch cb."""
    n = x.numel()
    pad = (-n) % BL
    y = torch.nn.functional.pad(x.reshape(-1), (0, pad)).reshape(-1, BL)
    s = y.abs().amax(1, keepdim=True).clamp_min(1e-30).bfloat16().float().clamp_min(1e-30)
    q = naechstes(y / s, cb) * s
    return q.reshape(-1)[:n].reshape(x.shape)


def rang1(v):
    """v >= 0, 2-D: Skala min(Zeilenmax, Spaltenmax) (fp16), lineare Stufen ohne Null."""
    r = v.amax(1, keepdim=True).clamp_min(1e-30).bfloat16().float().clamp_min(1e-30)
    c = v.amax(0, keepdim=True).clamp_min(1e-30).bfloat16().float().clamp_min(1e-30)
    s = torch.minimum(r, c)
    return naechstes((v / s).clamp(0, 1), LIN0) * s


class AdamLi(K.AdamZ):
    def __init__(self, netz, v_art="rang1", sr=False):
        super().__init__(netz)
        self.v_art, self.sr = v_art, sr
        self.q = [p.numel() >= K.MIN_N for p in self.p]

    def _qv(self, v):
        if self.v_art == "rang1" and v.ndim == 2:
            return rang1(v)
        return self._v_block(v)

    @staticmethod
    def _v_block(v):
        n = v.numel()
        pad = (-n) % BL
        y = torch.nn.functional.pad(v.reshape(-1), (0, pad)).reshape(-1, BL)
        s = y.amax(1, keepdim=True).clamp_min(1e-30).bfloat16().float().clamp_min(1e-30)
        return (naechstes((y / s).clamp(0, 1), LIN0) * s).reshape(-1)[:n].reshape(v.shape)

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
                    SR[0] = self.sr
                    m, v = blockweise(m, DE, True), self._qv(v)
                self.m[k], self.v[k] = m, v

    def zustand_bytes(self):
        by = 0.0
        for p, q in zip(self.p, self.q):
            n = p.numel()
            if not q:
                by += 8.0 * n
                continue
            by += n * 4 / 8 + (n / BL) * 2                                    # m
            by += n * 4 / 8 + (2.0 * (p.shape[0] + p.shape[1]) if (self.v_art == "rang1" and p.ndim == 2) else (n / BL) * 2)
        return by


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    klein = schnell
    T = K.T_SCHNELL if schnell else K.T_VOLL
    alle = K.arme(seed)
    arme = {k: alle[k] for k in ("adam32", "b8", "b4roh", "b4rot")}
    arme["li4"] = (K.LR_ADAM, lambda n: AdamLi(n, "rang1", False))
    arme["li4_sr"] = (K.LR_ADAM, lambda n: AdamLi(n, "rang1", True))
    arme["li4_blockv_sr"] = (K.LR_ADAM, lambda n: AdamLi(n, "block", True))
    zeilen = {}
    for name, (lr, ctor) in arme.items():
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
    r = {"kid": "K07B", "seed": seed, "schnell": schnell, "T": T, "klein": klein, "parameter": n, "mac_je_token": mac, "trainings_flops_je_schritt": fl,
         "trainings_flops_gesamt": fl * T, "zusatz_ops_je_schritt_ideal": int(n * 4 * 2),
         "hinweis": "FLOPs der Vorwaerts/Rueckwaerts-Rechnung in allen Armen gleich; li4/li4_blockv ohne Rotation: Zusatz nur Block-Absmax + Codebuch-Suche (~4 Op. je Wert und Moment); Adam-Arme: Zustand-Bytes fp32 = 8 B/Param",
         "arme": zeilen, "abstand_zu_adam32_bpc": {k: (v["test_bpc"] - ref) if (ref is not None and "test_bpc" in v) else None for k, v in zeilen.items()}}
    r["zusammenfassung"] = "\n".join(f"{k}: test {v.get('test_bpc')} bpc, Zustand {v['zustand_bytes'] / 1e6:.3f} MB" for k, v in zeilen.items())
    r["sekunden"] = time.time() - t0
    return r


if __name__ == "__main__":
    H.cli("K07B", main)
