"""Gemeinsame Bausteine der Z1-Erkundungsskripte (Welle 3, 28.10.2026). Erkundung, kein Befund.
Ergänzt _basis.py um: echte 2/3/4-Stufen-Gleichgitter-Quantisierung (Mittelpunktsgitter, Zeilenskala per MSE-Suche), schnelle Rotation (Vorzeichen*Hadamard = kb.rot_matrix, ohne dichte Matrix),
QAT-/PTQ-Schicht, KD-fähiges MLP-Training, Gleiche-Bytes-Breite des fp16-Gegenspielers, Einzelseed-Ampel.
Umgebungsvariable Z1_THREADS (Standard 1) setzt die Rechenfäden je Prozess."""
from __future__ import annotations
import copy
from _basis import *

torch.set_num_threads(int(os.environ.get("Z1_THREADS", "1")))
SCHNELL = os.environ.get("SCHNELL", "") not in ("", "0")


# ------------------------------------------------------------------ Quantisierer
def quant_skala(W, bits, s):
    hi = 2 ** bits / 2 - 0.5
    return ((W / s - 0.5).round() + 0.5).clamp(-hi, hi) * s


def quant_gitter(W, bits, nf=12, skala=False):
    """Zeilenweise Gleichgitter mit L = 2^bits Stufen (Mittelpunkte ±0,5 s, ±1,5 s, ...), Skala s je Zeile per MSE-Suche (nf Kandidaten 0,3..1,0 x Maximum)."""
    if bits == 0:
        return torch.zeros_like(W)
    L = 2 ** bits
    bs_s, best_s = None, None
    hi = L / 2 - 0.5
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bs = None, None
    for f in np.linspace(0.3, 1.0, nf):
        s = amax * float(f) / (L / 2)
        Wq = ((W / s - 0.5).round() + 0.5).clamp(-hi, hi) * s
        e = ((W - Wq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs, bs_s = e, Wq, s
        else:
            m = e < best
            best, bs, bs_s = torch.where(m, e, best), torch.where(m, Wq, bs), torch.where(m, s, bs_s)
    return (bs, bs_s) if skala else bs


def vorzeichen(n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()


def rot_vor(x, s):
    """x @ Q.T mit Q = kb.rot_matrix(n, seed) = H diag(s): schnelle Hadamard nach Vorzeichen."""
    return fwht(x * s)


class QATLin(nn.Module):
    """Linearschicht, deren Gewicht in (optional) rotierter Basis liegt und bei jedem Vorwärtslauf auf `bits` Bit gerundet wird (STE).
    bits=None: fp (nur Rotation, falls rot). einfrieren(): PTQ = einmal runden und festhalten."""

    def __init__(self, lin, bits, rot, seed, pad784=True):
        super().__init__()
        W = lin.weight.data.clone()
        self.i0, self.o = W.shape[1], W.shape[0]
        self.pad = (1024 - self.i0) if (rot and self.i0 == 784 and pad784) else 0
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.bits, self.rot = bits, rot
        self.register_buffer("s", vorzeichen(W.shape[1], seed) if rot else torch.ones(1))
        self.V = nn.Parameter(rot_vor(W, self.s) if rot else W)
        self.b = nn.Parameter(lin.bias.data.clone())
        self.fest, self._skala, self._zaehl = None, None, 0

    def gewicht(self):
        if self.fest is not None:
            return self.fest
        if self.bits is None:
            return self.V
        with torch.no_grad():
            if self._skala is None or self._zaehl % 16 == 0:     # Skala nur alle 16 Vorwärtsläufe neu suchen (Tempo)
                _, self._skala = quant_gitter(self.V.detach(), self.bits, nf=8, skala=True)
            self._zaehl += 1
            q = quant_skala(self.V.detach(), self.bits, self._skala)
        return self.V + (q - self.V).detach()

    def einfrieren(self):
        with torch.no_grad():
            self.fest = self.V.detach() if self.bits is None else quant_gitter(self.V.detach(), self.bits)
        return self

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (0, self.pad))
        if self.rot:
            x = rot_vor(x, self.s)
        return F.linear(x, self.gewicht(), self.b)

    def bytes(self):
        return kb.bytes_q_linear(self.o, self.i0, self.bits, self.rot, pad=self.rot)

    def rel_fehler(self):
        with torch.no_grad():
            q = quant_gitter(self.V.detach(), self.bits) if self.fest is None else self.fest
            return float(((q - self.V) ** 2).sum() / (self.V ** 2).sum())


def ersetze_linear(netz, bits, rot, seed, pad784=True):
    """Tiefe Kopie, alle nn.Linear -> QATLin (bits, rot). Seeds seed*101+k wie K03."""
    netz = copy.deepcopy(netz)
    k = [0]

    def rek(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, QATLin(c, bits, rot, seed * 101 + k[0], pad784))
            else:
                rek(c)
    rek(netz)
    return netz


def qlins(netz):
    return [m for m in netz.modules() if isinstance(m, QATLin)]


def einfrieren(netz):
    for m in qlins(netz):
        m.einfrieren()
    netz.eval()
    return netz


def bytes_q(netz):
    return float(sum(m.bytes() for m in qlins(netz)))


def fp16_runden(netz):
    n = copy.deepcopy(netz)
    for p in n.parameters():
        p.data = p.data.half().float()
    return n


# ------------------------------------------------------------------ MLP
def mlp_daten(schnell):
    return kb.bild_daten("mnist", schnell)[:4]


def mlp_gleiche_bytes_breite(ziel_bytes, tiefe=2):
    """Größte Breite h (zwei verdeckte Schichten h-h) mit fp16-Bytes <= ziel."""
    best = 8
    for h in range(8, 1024):
        b = kb.bytes_fp16_linear(kb.baue_bild([h] * tiefe))
        if b <= ziel_bytes:
            best = h
    return best


def mlp_train(netz, X, Y, seed, ep, lr, bs=128, lehrer=None, alpha=0.9, temp=2.0):
    """Adam, 5 % Aufwärmen, Kosinus (wie mlp_trainiere). lehrer = Logits (N,10) der Trainingsdaten -> Verlust (1-alpha)*CE + alpha*T²*KL(Lehrer||Schüler)."""
    N = len(X)
    spe = math.ceil(N / bs)
    n = ep * spe
    opt = torch.optim.Adam([p for p in netz.parameters() if p.requires_grad], lr=lr)
    warm = max(1, n // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm))))
    netz.train()
    l = torch.tensor(0.0)
    for e in range(ep):
        p = torch.randperm(N, generator=torch.Generator().manual_seed(seed * 7919 + e))
        for k in range(spe):
            b = p[k * bs:(k + 1) * bs]
            z = netz(X[b])
            if lehrer is None:
                l = F.cross_entropy(z, Y[b])
            else:
                kd = F.kl_div(F.log_softmax(z / temp, 1), F.softmax(lehrer[b] / temp, 1), reduction="batchmean") * temp * temp
                l = (1 - alpha) * F.cross_entropy(z, Y[b]) + alpha * kd
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
    netz.eval()
    return float(l.detach())


def logits(netz, X, bs=2000):
    netz.eval()
    with torch.no_grad():
        return torch.cat([netz(X[a:a + bs]) for a in range(0, len(X), bs)])


def mlp_macs(dims):
    return int(sum(a * b for a, b in zip([784] + list(dims), list(dims) + [10])))


# ------------------------------------------------------------------ Ampel (Einzelseed, vorläufig)
def ampel(gewinn, schwelle, einheit, text=""):
    """Gewinn > 0: Idee besser als die härteste Kontrolle. Regel wie auswertung.py für n=1: grün = Gewinn > Schwelle, gelb = > 0, rot sonst (mehrere Seeds: auswertung.py)."""
    farbe = "gruen" if gewinn > schwelle else ("gelb" if gewinn > 0 else "rot")
    return {"gewinn": float(gewinn), "schwelle": schwelle, "einheit": einheit, "farbe": farbe, "vorlaeufig_ein_seed": True, "vergleich": text}


def steigung(xs, ys):
    """Kleinste Quadrate von y über log2(x)."""
    x = np.log2(np.array(xs, float))
    y = np.array(ys, float)
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])
