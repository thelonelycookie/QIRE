"""Z2_walsh_seed -- Walsh-Basis statt Zufalls-Basis: Gewichte als W ≈ Σ c_k · (Walsh-Basismatrix k), nur Koeffizienten gespeichert (Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: SeedLM speichert pro Block einen Seed (Zufalls-Basis aus PRNG) und wenige Koeffizienten. Die Walsh-Hadamard-Zeilen sind eine feste, ORTHONORMALE Basis, die ohne Speicher erzeugt wird
  (Zeile i, Spalte j = (-1)^popcount(i&j): Index -> Vorzeichenmuster) und per schneller Transformation angewandt wird. Schlägt "Walsh + Auswahl der größten Koeffizienten" bei gleichen Bytes
  (a) die Zufalls-Basis aus Seed (Matching Pursuit über einen Pool), (b) Niedrigrang-SVD, (c) 2-Bit-Skalar nach Rotation, (d) ein kleines fp16-Modell? Plus Adapter-Variante (VeRA-artig).
VORAB-P (vor dem Lauf): Teil A (Gewichte, PTQ): bester Walsh-Arm schlägt alle Gegner (grün) 5 % (MLP-Gewichte sind nach dem Training nicht spärlich in Walsh-Basis, 2 Bit/Gewicht = nur ~8 % Koeffizienten
  bei Indexkosten; der 2-Bit-Skalar nach Rotation behält alle Koeffizienten); Walsh schlägt die Zufalls-Basis aus Seed: 65 % (orthonormal, energie-optimale Auswahl ist exakt, Zufalls-Pool nur greedy);
  Walsh schlägt SVD: 50 %. Teil B (Adapter): vera_walsh schlägt vera_zufall um > 0,5 pp: 15 % (Basis-Wahl ist bei VeRA erfahrungsgemäß fast egal); schlägt alle Adapter-Gegner: 10 %.
MODELL: MNIST-MLP 784-512-512-10 fp32, Adam 1e-3, Kosinus, ZWEI Trainingslängen EP = 3 und 12 (--schnell: 1 und 4); Val 5 000 Bilder; Seed nur Tuning-Seeds.
TEIL A (alle PTQ, ohne Nachtraining; Bytes <= skalar4_rot = 2 Bit/Gewicht inkl. Zeilenskala, Bias, Vorzeichendiagonale = kb.bytes_q_linear; Koeffizienten 4 Bit, Skala per MSE-Suche):
  walsh2d_roh      W (auf 2^a x 2^b aufgefüllt) -> 2D-Walsh-Transformation, die K größten |c| je Schicht (Energie), Index roh = ceil(log2(Zeilen*Spalten)) Bit + 4 Bit Wert
  walsh2d_gap      wie walsh2d_roh, Indizes sortiert und Golomb/Rice-codiert: ~ log2(N/K)+2 Bit je Index (Schätzung, nicht tatsächlich codiert)
  walsh1d_gap      Zeilenweise Walsh (= Hadamard ohne Vorzeichen), je Zeile gleich viele k größte Koeffizienten, Rice-Index log2(n/k)+2, Zeilenskala fp16 (wie Skalar-Arm)
  seedlm_basis     Zufalls-Basis aus Seed: Pool 4 096 (1 024 schnell) Gauß-Einheitsvektoren (PRNG, 0 Byte), je Zeile k Atome per Matching Pursuit (mit Gram-Update), danach Koeffizienten per
                   kleinste Quadrate, 4 Bit; Index 12 Bit roh
  svd_rang         Niedrigrang r so, dass fp16-Faktoren + Bias <= Budget je Schicht
  skalar4_rot      4 echte Stufen je Zeile nach Rotation (Gegner mit allen Koeffizienten)
  klein_fp16       gleiche Bytes, gleiche Epochen
  FLOPs: flops_inferenz_schnell = Hadamard des Eingangs + dünn besetztes Produkt + (2D) Hadamard der Ausgänge, nur Walsh/Seed-Arme; flops_inferenz_dicht = 2*MACs der rekonstruierten Matrix
  (+ Hadamard), flops_rekonstruktion_einmalig = Matrix aus Koeffizienten aufbauen. Seed-Basis braucht zusätzlich den Pool-Eingang (Pool x Eingang) -> im Feld gezählt.
TEIL B (Adapter, VeRA-artig, Aufgabe: auf MNIST trainiertes Grundmodell auf Fashion-MNIST anpassen; Adapter-Epochen 3, schnell 1; Val 5 000): je Schicht dW = diag(g) B diag(dv) A mit EINGEFRORENEN A (r x n), B (o x r),
  trainiert werden nur dv (r) und g (o); r = 256; dv Start 0,1, g Start 0; Lernrate 3e-2 für alle Adapter-Arme (Vortest auf Tuning-Seed 959: 1e-2 und 3e-3 waren für LoRA und BitFit zu klein).
  vera_walsh       A = r zufällig (Seed) gewählte Walsh-Zeilen (Index -> Muster, 0 Byte), B = Walsh-Spalten ebenso
  vera_zufall      A, B Gauß aus Seed (Normierung wie Walsh)
  lora_r1          LoRA Rang 1 (MEHR trainierbare Parameter als VeRA, härterer Gegner)
  bitfit           nur Biases
  voll_ft          alle Gewichte (Obergrenze), ohne_anpassung (Untergrenze)
  Zählung: trainierbare Parameter und fp16-Bytes der zu speichernden Anpassung (Basis 0 Byte); Walsh/Zufall-Basen kosten 0 Byte.
AMPEL (Einzelseed, vorläufig): ampel_teil_a: bester Walsh-Arm − max(seedlm_basis, svd_rang, skalar4_rot, klein_fp16); ampel_walsh_gegen_seed: bester Walsh-Arm − seedlm_basis;
  ampel_adapter: vera_walsh − max(vera_zufall, lora_r1, bitfit); jeweils Pp, Minimum über beide Längen, Schwelle 0,5 pp.
Lies zuerst: teil_a[<länge>] walsh_* gegen seedlm_basis und skalar4_rot (und Gewichtsfehler), dann svd_rang; dann adapter."""
from __future__ import annotations
from _z1 import *
import K03 as K

GEW = 784 * 512 + 512 * 512 + 512 * 10
FORMEN = ((512, 784), (512, 512), (10, 512))
CB = 4                                # Bit je Koeffizient


# ------------------------------------------------------------------ Walsh-Zeilen ohne Speicher
def walsh_zeilen(idx, N):
    """Zeilen idx (Tensor) der N x N Walsh-Hadamard-Matrix (±1/sqrt(N)): Eintrag (i,j) = (-1)^popcount(i&j); keine Matrix gespeichert."""
    j = torch.arange(N)
    v = idx[:, None] & j[None, :]
    par = torch.zeros_like(v)
    while bool((v > 0).any()):
        par = par ^ (v & 1)
        v = v >> 1
    return (1 - 2 * par).float() / math.sqrt(N)


def selbsttest_walsh():
    """fwht (schnelle Transformation) und die Formel (Index -> Vorzeichen) müssen dieselbe Matrix liefern."""
    N = 32
    H = walsh_zeilen(torch.arange(N), N)
    return bool((fwht(torch.eye(N)) - H).abs().max() < 1e-6)


def pot2(n):
    return 1 << (n - 1).bit_length()


def wht2d(W):
    return fwht(fwht(W).T).T


# ------------------------------------------------------------------ Koeffizienten-Quantisierung / Bits
def quant_koef(v):
    """(m, k) -> 4-Bit-Gleichgitter je Zeile, Skala per MSE-Suche."""
    return quant_gitter(v, CB, nf=12)


def idx_bits_roh(N):
    return math.ceil(math.log2(N))


def idx_bits_rice(N, k):
    return math.log2(max(N / k, 1.0)) + 2.0


def waehle_k(bits_frei, N, rice):
    """Größtes K mit K*(CB + Indexbits) <= bits_frei."""
    if not rice:
        return int(bits_frei // (CB + idx_bits_roh(N)))
    k = 1
    for _ in range(40):
        kn = int(bits_frei // (CB + idx_bits_rice(N, max(k, 1))))
        if kn == k:
            break
        k = max(kn, 1)
    return k


def budget_bytes(o, i):
    return kb.bytes_q_linear(o, i, 2, True, pad=True)


class FixLin(nn.Module):
    """Linearschicht mit fest gesetztem Gewicht (rekonstruiert), Bytes werden von außen gesetzt."""

    def __init__(self, W, b, bytes_, fehl, extra=None):
        super().__init__()
        self.W, self.b, self.nbytes, self.fehl, self.shape, self.extra = W, b, bytes_, fehl, tuple(W.shape), extra or {}

    def forward(self, x):
        return F.linear(x, self.W, self.b)

    def bytes(self):
        return self.nbytes


def rel(W, Wq):
    return float(((W - Wq) ** 2).sum() / (W ** 2).sum())


# ------------------------------------------------------------------ Arme
def walsh2d(lin, rice):
    W, b = lin.weight.data, lin.bias.data
    o, i = W.shape
    Op, Np = pot2(o), pot2(i)
    Wp = torch.zeros(Op, Np)
    Wp[:o, :i] = W
    C = wht2d(Wp)
    N = Op * Np
    bud = budget_bytes(o, i)
    frei = 8 * (bud - 2 * o - 2)                      # Bias fp16 und eine Skala fp16 je Schicht
    k = min(waehle_k(frei, N, rice), N)
    top = C.abs().flatten().topk(k).indices
    v = C.flatten()[top]
    vq = quant_koef(v[None])[0]
    Ch = torch.zeros(N)
    Ch[top] = vq
    Wq = wht2d(Ch.reshape(Op, Np))[:o, :i]
    bits = k * (CB + (idx_bits_rice(N, k) if rice else idx_bits_roh(N)))
    by = math.ceil(bits / 8) + 2 * o + 2
    schnell = int(Np * math.log2(Np) + k + Op * math.log2(Op))
    return FixLin(Wq, b.clone(), by, rel(W, Wq), {"k": k, "anteil_koef": k / N, "flops_schnell": schnell, "flops_dicht": 2 * o * i, "flops_rek": int(2 * Op * Np * (math.log2(Op) + math.log2(Np)))})


def walsh1d(lin):
    W, b = lin.weight.data, lin.bias.data
    o, i = W.shape
    Np = pot2(i)
    Wp = F.pad(W, (0, Np - i))
    C = fwht(Wp)
    bud = budget_bytes(o, i)
    frei = 8 * (bud - 2 * o - 2 * o)                  # Bias + Zeilenskala
    k = min(max(waehle_k(frei / o, Np, True), 1), Np)
    top = C.abs().topk(k, dim=1).indices
    v = C.gather(1, top)
    vq = quant_koef(v)
    Ch = torch.zeros_like(C).scatter_(1, top, vq)
    Wq = fwht(Ch)[:, :i]
    bits = o * k * (CB + idx_bits_rice(Np, k))
    by = math.ceil(bits / 8) + 4 * o
    schnell = int(Np * math.log2(Np) + o * k)
    return FixLin(Wq, b.clone(), by, rel(W, Wq), {"k": k, "anteil_koef": k / Np, "flops_schnell": schnell, "flops_dicht": 2 * o * i, "flops_rek": int(o * Np * math.log2(Np))})


_POOL = {}


def pool_gram(n, P, seed):
    if (n, P, seed) not in _POOL:
        g = torch.Generator().manual_seed(seed * 29 + n)
        Pm = torch.randn(P, n, generator=g)
        Pm = Pm / Pm.norm(dim=1, keepdim=True)
        _POOL[(n, P, seed)] = (Pm, Pm @ Pm.T)
    return _POOL[(n, P, seed)]


def seedlm_basis(lin, P, seed):
    W, b = lin.weight.data, lin.bias.data
    o, i = W.shape
    Pm, Gm = pool_gram(i, P, seed)
    pb = int(math.log2(P))
    bud = budget_bytes(o, i)
    frei = 8 * (bud - 2 * o - 2 * o - 4)              # Bias, Zeilenskala, Seed
    k = max(1, int(frei / o // (CB + pb)))
    S = W @ Pm.T                                      # (o, P) Korrelationen
    sel = torch.empty(o, k, dtype=torch.long)
    ar = torch.arange(o)
    for t in range(k):
        j = S.abs().argmax(1)
        c = S[ar, j]
        S = S - c[:, None] * Gm[j]
        sel[:, t] = j
    A = Pm[sel]                                       # (o, k, n)
    # kleinste Quadrate: min |W_r - c A_r|, c = Gram^-1 (A W)
    Gk = A @ A.transpose(1, 2) + 1e-4 * torch.eye(k)
    rhs = (A @ W[:, :, None])
    c = torch.linalg.solve(Gk, rhs)[:, :, 0]
    cq = quant_koef(c)
    Wq = (cq[:, :, None] * A).sum(1)
    by = math.ceil(o * k * (CB + pb) / 8) + 4 * o + 4
    return FixLin(Wq, b.clone(), by, rel(W, Wq), {"k": k, "flops_schnell": int(P * i + o * k), "flops_dicht": 2 * o * i, "flops_rek": int(2 * o * k * i)})


def svd_rang(lin):
    W, b = lin.weight.data, lin.bias.data
    o, i = W.shape
    bud = budget_bytes(o, i)
    r = max(1, int((bud - 2 * o) // (2 * (o + i))))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Uh = (U[:, :r] * S[:r].sqrt()).half().float()
    Vr = (S[:r].sqrt()[:, None] * Vh[:r]).half().float()
    Wq = Uh @ Vr
    return FixLin(Wq, b.clone(), 2 * r * (o + i) + 2 * o, rel(W, Wq), {"rang": r, "flops_schnell": int(r * (o + i)), "flops_dicht": 2 * o * i, "flops_rek": int(2 * o * r * i)})


def ersetze_schichten(netz, fn):
    ls = [fn(l) for l in netz.ls]
    return kb.Netz(ls), ls


# ------------------------------------------------------------------ Adapter (VeRA-artig)
class VLin(nn.Module):
    """Eingefrorenes Gewicht + dW = diag(g) B diag(dv) A (A, B eingefroren; dv, g trainiert)."""

    def __init__(self, lin, A, B, mode):
        super().__init__()
        self.register_buffer("W", lin.weight.data.clone())
        self.b = nn.Parameter(lin.bias.data.clone(), requires_grad=(mode == "bitfit"))
        self.mode = mode
        if mode in ("vera",):
            self.register_buffer("A", A)
            self.register_buffer("B", B)
            self.dv = nn.Parameter(torch.full((A.shape[0],), 0.1))
            self.g = nn.Parameter(torch.zeros(B.shape[0]))
        elif mode == "lora":
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(torch.zeros(B.shape))

    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        if self.mode == "vera":
            y = y + self.g * F.linear(self.dv * F.linear(x, self.A), self.B)
        elif self.mode == "lora":
            y = y + F.linear(F.linear(x, self.A), self.B)
        return y

    def param_n(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def basen(art, o, n, r, seed):
    """A (r x n), B (o x r) eingefroren."""
    g = torch.Generator().manual_seed(seed * 53 + o * 3 + n)
    if art == "walsh":
        Nn, No = pot2(n), pot2(max(o, r))
        ra = torch.randperm(Nn, generator=g)[:r]
        A = walsh_zeilen(ra, Nn)[:, :n] * math.sqrt(Nn / n)          # Zeilennorm ~1
        ro = torch.randperm(No, generator=g)[:o]
        co = torch.randperm(No, generator=g)[:r]
        B = walsh_zeilen(ro, No)[:, co] * math.sqrt(No / r)
    else:
        A = torch.randn(r, n, generator=g) / math.sqrt(n)
        B = torch.randn(o, r, generator=g) / math.sqrt(r)
    return A, B


def adapter_netz(basis, art, seed, r=256):
    ls = []
    for k, l in enumerate(basis.ls):
        o, n = l.weight.shape
        if art in ("walsh", "zufall"):
            A, B = basen(art, o, n, r, seed + k)
            ls.append(VLin(l, A, B, "vera"))
        elif art == "lora":
            g = torch.Generator().manual_seed(seed * 7 + k)
            ls.append(VLin(l, torch.randn(1, n, generator=g) / math.sqrt(n), torch.zeros(o, 1), "lora"))
        else:
            ls.append(VLin(l, None, None, "bitfit"))
    return kb.Netz(ls)


def teil_b(basis, seed, schnell, ftrain, fval):
    xtr, ytr, xva, yva = ftrain
    ep = 1 if schnell else 3
    res = {}
    res["ohne_anpassung"] = {"acc": acc(basis, xva, yva), "param": 0, "bytes": 0}
    for art, name, lr in (("walsh", "vera_walsh", 3e-2), ("zufall", "vera_zufall", 3e-2), ("lora", "lora_r1", 3e-2), ("bitfit", "bitfit", 3e-2)):
        t = time.time()
        nz = adapter_netz(basis, art, seed)
        mlp_train(nz, xtr, ytr, seed, ep, lr)
        n = sum(l.param_n() for l in nz.ls)
        res[name] = {"acc": acc(nz, xva, yva), "param": n, "bytes_fp16": 2 * n, "sekunden": time.time() - t}
        print(f"   adapter {name:14s} acc {res[name]['acc']:.4f}  Param {n}  ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    voll = copy.deepcopy(basis)
    mlp_train(voll, xtr, ytr, seed, ep, 1e-3)
    res["voll_ft"] = {"acc": acc(voll, xva, yva), "param": zaehle_param(voll), "sekunden": time.time() - t}
    print(f"   adapter voll_ft        acc {res['voll_ft']['acc']:.4f}  ohne {res['ohne_anpassung']['acc']:.4f}", flush=True)
    return res


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 4) if schnell else (3, 12)
    P = 1024 if schnell else 4096
    xtr, ytr, xva, yva = mlp_daten(schnell)
    ftrain = kb.bild_daten("fashion_mnist", schnell)[:4]
    out = {"epochen": eps, "pool": P, "selbsttest_walsh_formel_gleich_fwht": selbsttest_walsh(), "teil_a": {}, "adapter": {}}
    mac = mlp_macs([512, 512])
    for ep in eps:
        tl = time.time()
        torch.manual_seed(seed)
        netz = kb.baue_bild([512, 512])
        mlp_train(netz, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(netz, xva, yva)
        with torch.no_grad():
            z0 = netz(xva)
        A = out["teil_a"][ep] = {"fp32": {"acc": a0, "bytes_fp16": kb.bytes_fp16_linear(netz)}}
        print(f"Z2_walsh_seed s{seed} ep{ep} fp32 {a0:.4f}", flush=True)

        def eintrag(name, q, ls, t):
            q.eval()
            with torch.no_grad():
                zq = q(xva)
            b = sum(l.bytes() for l in ls)
            e = {"acc": acc(q, xva, yva), "gewichtsfehler_rel": sum(l.fehl * l.shape[0] * l.shape[1] for l in ls) / GEW, "logit_fehler_rel": float((zq - z0).norm() / z0.norm()),
                 "bytes": b, "bits_pro_gewicht_effektiv": 8 * b / GEW, "sekunden": time.time() - t}
            if hasattr(ls[0], "extra") and "flops_schnell" in ls[0].extra:
                e["flops_inferenz_schnell"] = int(sum(l.extra["flops_schnell"] for l in ls))
                e["flops_inferenz_dicht"] = 2 * mac
                e["flops_rekonstruktion_einmalig"] = int(sum(l.extra["flops_rek"] for l in ls))
                e["je_schicht"] = [l.extra for l in ls]
            A[name] = e
            print(f"   {name:14s} acc {e['acc']:.4f}  Gew.fehler {e['gewichtsfehler_rel']:.4f}  Bytes {b:.0f}  ({time.time() - t:.0f}s)", flush=True)

        for name, fn in (("walsh2d_roh", lambda l: walsh2d(l, False)), ("walsh2d_gap", lambda l: walsh2d(l, True)), ("walsh1d_gap", walsh1d),
                         ("seedlm_basis", lambda l: seedlm_basis(l, P, seed)), ("svd_rang", svd_rang)):
            t = time.time()
            qn, ls = ersetze_schichten(netz, fn)
            eintrag(name, qn, ls, t)
        t = time.time()
        qn, ls = K.ersetze(netz, "skalar4", True, seed, None)
        for l in ls:
            l.extra = {}
        eintrag("skalar4_rot", qn, ls, t)
        A["skalar4_rot"].update({"flops_inferenz_dicht": 2 * mac + int(1024 * 10 + 512 * 9 * 2), "flops_rekonstruktion_einmalig": GEW})
        h = mlp_gleiche_bytes_breite(A["skalar4_rot"]["bytes"])
        torch.manual_seed(seed + 1)
        kl = kb.baue_bild([h, h])
        mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
        kl = fp16_runden(kl)
        A["klein_fp16"] = {"acc": acc(kl, xva, yva), "bytes": kb.bytes_fp16_linear(kl), "breite": h, "flops_inferenz_dicht": 2 * mlp_macs([h, h]), "flops_rekonstruktion_einmalig": 0}
        print(f"   klein_fp16 h={h} acc {A['klein_fp16']['acc']:.4f}  Bytes {A['klein_fp16']['bytes']}", flush=True)
        out["adapter"][ep] = teil_b(netz, seed, schnell, ftrain, None)
        print(f"  Länge {ep} Ep. gesamt {time.time() - tl:.0f}s", flush=True)
    walsh = ("walsh2d_roh", "walsh2d_gap", "walsh1d_gap")
    g_a, g_s, g_ad = [], [], []
    for ep in eps:
        A, B = out["teil_a"][ep], out["adapter"][ep]
        bw = max(A[n]["acc"] for n in walsh)
        g_a.append(100 * (bw - max(A[k]["acc"] for k in ("seedlm_basis", "svd_rang", "skalar4_rot", "klein_fp16"))))
        g_s.append(100 * (bw - A["seedlm_basis"]["acc"]))
        g_ad.append(100 * (B["vera_walsh"]["acc"] - max(B[k]["acc"] for k in ("vera_zufall", "lora_r1", "bitfit"))))
    out["gewinn_pp_je_laenge"] = {"teil_a_walsh_gegen_haertesten": g_a, "walsh_gegen_seedlm": g_s, "adapter_walsh_gegen_haertesten": g_ad}
    out["ampel_teil_a"] = ampel(min(g_a), 0.5, "pp", "bester Walsh-Arm gegen max(seedlm_basis, svd_rang, skalar4_rot, klein_fp16)")
    out["ampel_walsh_gegen_seed"] = ampel(min(g_s), 0.5, "pp", "bester Walsh-Arm gegen seedlm_basis")
    out["ampel_adapter"] = ampel(min(g_ad), 0.5, "pp", "vera_walsh gegen max(vera_zufall, lora_r1, bitfit)")
    print("Ampel A", out["ampel_teil_a"]["farbe"], "Walsh/Seed", out["ampel_walsh_gegen_seed"]["farbe"], "Adapter", out["ampel_adapter"]["farbe"], out["gewinn_pp_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_walsh_seed", main)
