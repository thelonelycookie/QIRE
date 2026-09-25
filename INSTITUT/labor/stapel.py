"""Encoder/Decoder fuer GPU-Batching vieler kleiner, unabhaengiger Auftraege
(Direktor, 25.09., auf Maltes Wunsch -- WERKZEUG fuer den Arbeitsablauf,
KEIN physikalischer Befund).

IDEE (Malte, 25.09.): Ein grosser Teil unserer Rechnungen sind N unabhaengige
KLEINE Auftraege -- typischerweise ein Seed = eine kleine QireZelle (wenige
KB Gewichte) auf einem kleinen Bild-Batch. Einzeln lohnt sich fuer so etwas
keine GPU (der Aufruf-Overhead frisst den Vorteil); wir rechnen sie deshalb
heute mit einem CPU-Prozesspool parallel (labor.gemeinsam.parallel).
Eine GPU lohnt sich erst, wenn man die N kleinen Auftraege zu EINEM grossen
Tensor-Batch zusammenfasst: **Encoder** stapelt N Modelle + N Datensaetze zu
einer gemeinsamen "Instanz"-Dimension, **Decoder** liest die Ergebnisse
wieder pro Auftrag auseinander.

EHRLICH: GPU-Batching vieler kleiner Quantenschaltungs-Simulationen ist eine
etablierte HPC-Technik (Batched-Statevector-Simulatoren, z. B. NVIDIA
cuQuantum, PennyLane-Lightning-GPU, "Scalable Parallel Simulation of Quantum
Circuits on CPU and GPU Systems" arXiv:2509.04955) -- das ist hier also
WERKZEUG, kein neuer physikalischer Befund. Neu ist nur, es fuer unseren
Institutsablauf (viele Seeds derselben kleinen QireZelle) nutzbar zu machen.

STAND 25.09.: Diese Cloud-Umgebung hat KEINE GPU (torch.cuda.is_available()
== False, kein nvidia-smi). Der Code ist geraeteunabhaengig (device
waehlbar) und HIER NUR AUF CPU KORREKTHEITS- UND LAUFZEITGEPRUEFT. Der
eigentliche Durchsatzgewinn (das ist der ganze Sinn der Uebung) muss auf
einer Maschine mit CUDA-GPU gemessen werden -- offene Frage an Malte, ob
sein PC eine NVIDIA-GPU hat.

Mechanismus: torch.func.vmap ueber eine Instanz-Dimension der Modell-
Parameter (theta, Kopf-Gewichte) UND der Daten -- eine einzige vektorisierte
Operation statt einer Python-Schleife ueber N Instanzen. Am Beispiel der
QireZelle-Rechnung (Kronecker-Schaltung -> Born -> linearer Kopf), aber ohne
clifford.py anzufassen: die Rechnung ist hier eigenstaendig nachgebaut
(gleiche Formel, per Vortest gegen clifford.QireZelle geprueft), damit dieses
Werkzeug unabhaengig bleibt und niemand sonst laufende Faecher-Datei anfasst.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.func import vmap


# ---------------------------------------------------------------- Encoder/Decoder
def encode(liste_von_tensoren: list[torch.Tensor]) -> torch.Tensor:
    """N gleich geformte Tensoren -> EIN Tensor mit neuer Instanz-Dimension 0.
    Das ist der ganze 'Encoder': aus N kleinen Auftraegen wird ein grosser Batch."""
    return torch.stack(liste_von_tensoren, dim=0)


def decode(batch: torch.Tensor) -> list[torch.Tensor]:
    """Kehrt encode() um: EIN Batch-Tensor -> Liste von N Einzel-Ergebnissen."""
    return list(batch.unbind(0))


# ---------------------------------------------------------------- QireZelle-Rechnung, batchbar nachgebaut
def _cnot_ring_perm(n: int) -> torch.Tensor:
    """Identisch zu abteilungen.clifford._cnot_ring_perm (hier eigenstaendig,
    damit dieses Werkzeug keine Abteilungsdatei importiert/aendert)."""
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    for q in range(n):
        t = (q + 1) % n
        bits[:, t] ^= bits[:, q]
    neu = (bits << (n - 1 - np.arange(n))).sum(1)
    perm = np.empty_like(neu)
    perm[neu] = idx
    return torch.tensor(perm)


def _unitaer_einzeln(theta: torch.Tensor, perm: torch.Tensor, n: int, lagen: int) -> torch.Tensor:
    """Eine einzelne QireZelle-Schaltung als 2^n x 2^n-Matrix (wie clifford.QireZelle.unitaer)."""
    U = torch.eye(2 ** n, dtype=theta.dtype)
    for l in range(lagen):
        K = torch.ones(1, 1, dtype=theta.dtype)
        for q in range(n):
            c, s = torch.cos(theta[l, q] / 2), torch.sin(theta[l, q] / 2)
            K = torch.kron(K, torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
        U = K @ U
        U = U[perm]
    return U


def merkmale_einzeln(theta: torch.Tensor, x: torch.Tensor, perm: torch.Tensor, n: int, lagen: int) -> torch.Tensor:
    """Born-Merkmale EINER Instanz auf ihrem Bild-Batch x (b, 2^n)."""
    psi = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    psi = psi @ _unitaer_einzeln(theta, perm, n, lagen).T
    return psi.reshape(x.shape[0], -1) ** 2


def merkmale_geschleift(thetas: list[torch.Tensor], xs: list[torch.Tensor], perm, n, lagen) -> list[torch.Tensor]:
    """Referenz: Python-Schleife ueber N Instanzen (= heutiges Vorgehen ohne Encoder)."""
    return [merkmale_einzeln(th, x, perm, n, lagen) for th, x in zip(thetas, xs)]


def merkmale_gestapelt(theta_batch: torch.Tensor, x_batch: torch.Tensor, perm, n, lagen) -> torch.Tensor:
    """Encoder-Pfad: EIN vmap-Aufruf ueber alle N Instanzen statt N Python-Aufrufe.
    theta_batch: (N, lagen, n); x_batch: (N, b, 2^n) -> (N, b, 2^n)."""
    fn = vmap(lambda th, x: merkmale_einzeln(th, x, perm, n, lagen))
    return fn(theta_batch, x_batch)


# ---------------------------------------------------------------- Selbstpruefung + Messung
def _pruefe_korrektheit(n=6, lagen=4, N=12, b=20, seed=0) -> float:
    """Batch-Pfad == Schleifen-Pfad, elementweise. Gibt den groessten Unterschied zurueck."""
    g = torch.Generator().manual_seed(seed)
    perm = _cnot_ring_perm(n)
    thetas = [torch.rand(lagen, n, generator=g) * 2 * 3.14159265 for _ in range(N)]
    xs = [torch.rand(b, 2 ** n, generator=g) for _ in range(N)]
    referenz = merkmale_geschleift(thetas, xs, perm, n, lagen)
    gestapelt = decode(merkmale_gestapelt(encode(thetas), encode(xs), perm, n, lagen))
    return max(float((r - s).abs().max()) for r, s in zip(referenz, gestapelt))


def messung(n=8, lagen=4, N_werte=(5, 20, 50, 100), b=64, wiederholungen=3) -> dict:
    """Wanduhrzeit Schleife vs. Batch (Encoder/Decoder) auf DIESEM Geraet (hier: CPU).
    Kein physikalischer Befund -- ehrliche Werkzeug-Messung fuer den Institutsablauf."""
    perm = _cnot_ring_perm(n)
    aus = {"geraet": "cuda" if torch.cuda.is_available() else "cpu", "n": n, "lagen": lagen, "b": b, "stufen": {}}
    for N in N_werte:
        thetas = [torch.rand(lagen, n) * 2 * 3.14159265 for _ in range(N)]
        xs = [torch.rand(b, 2 ** n) for _ in range(N)]
        t0 = time.perf_counter()
        for _ in range(wiederholungen):
            merkmale_geschleift(thetas, xs, perm, n, lagen)
        t_schleife = (time.perf_counter() - t0) / wiederholungen
        theta_b, x_b = encode(thetas), encode(xs)
        vmap(lambda th, x: merkmale_einzeln(th, x, perm, n, lagen))(theta_b, x_b)  # Aufwaermen (vmap-Tracing)
        t0 = time.perf_counter()
        for _ in range(wiederholungen):
            merkmale_gestapelt(theta_b, x_b, perm, n, lagen)
        t_batch = (time.perf_counter() - t0) / wiederholungen
        aus["stufen"][N] = {"sekunden_schleife": t_schleife, "sekunden_batch": t_batch,
                            "beschleunigung": t_schleife / t_batch if t_batch > 0 else None}
    return aus


if __name__ == "__main__":
    fehler = _pruefe_korrektheit()
    print(f"Korrektheit (groesster Unterschied Batch vs. Schleife): {fehler:.2e} {'ok' if fehler < 1e-5 else 'FEHLER'}")
    erg = messung()
    print(f"Geraet: {erg['geraet']}")
    for N, s in erg["stufen"].items():
        print(f"  N={N:>4}: Schleife {s['sekunden_schleife']*1000:7.2f} ms | "
              f"Batch {s['sekunden_batch']*1000:7.2f} ms | Faktor {s['beschleunigung']:.2f}x")
