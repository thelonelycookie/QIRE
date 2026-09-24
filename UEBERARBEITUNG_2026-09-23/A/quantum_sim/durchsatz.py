from __future__ import annotations

"""RESEARCH ID: T-000 -- Wie gross darf das Sprachmodell auf diesem Rechner sein?

WARUM DIESE MESSUNG VOR DER PLANUNG KOMMT
Phase 3 der Roadmap ist ein Sprachmodell, und die Zielgroesse ist bisher
geraten. Raten ist in diesem Projekt schon zweimal teuer geworden: Q-009 lief
mit drei Seeds, weil die Seedzahl nach Rechenzeit statt nach erreichbarer
Signifikanz gewaehlt wurde, und Q-018s erster Aufgabenzuschnitt stammte aus
einem Lauf mit Breite 16.384 und war fuer 8 Kanaele sinnlos.

Deshalb dasselbe Vorgehen wie bei der isochronen Kalibrierung fuer Q-009:
messen, was der Rechner leistet, und die Modellgroesse daraus ABLEITEN.

DIES IST KEIN EXPERIMENT, SONDERN EINE KALIBRIERUNG.
Es gibt keine Hypothese, keine Kontrollgruppe und keinen p-Wert. Die Zahlen
sind Durchsatzmessungen und werden auch nur so berichtet. Sie gehoeren nicht in
`q_summary` (das sortiert nach Urteilen) -- die Rohdaten werden trotzdem
gespeichert, damit die Ableitung nachvollziehbar bleibt.

WAS GEMESSEN WIRD
Ein schlichter Transformer-Block-Stapel in mehreren Breiten, jeweils
Vorwaerts + Rueckwaerts + Optimierungsschritt, also echter Trainingsdurchsatz.
Gemessen wird in Token je Sekunde, dazu der Speicherbedarf.

WAS DARAUS ABGELEITET WIRD
Als Faustregel fuer "genug Daten gesehen" werden 20 Token je Parameter benutzt
(Hoffmann et al. 2022, Chinchilla). Damit ergibt sich fuer jedes Zeitbudget die
groesste Breite, die in diesem Budget noch ausreichend Token sieht.

WICHTIG ZUR EINORDNUNG: Diese Messung sagt, was **trainierbar** ist. Sie sagt
nichts darueber, ob die Butterfly-Struktur dort gewinnt -- laut Tagebuch vom
5. August liegt der Uebergangspunkt bei 16.384 Dimensionen, also weit oberhalb
von allem, was hier herauskommen wird. Genau diese Luecke ist der Grund fuer
die T-Serie.

Run: python -m quantum_sim.durchsatz
"""

import time
from typing import Dict, List, Sequence

import torch
from torch import nn

from qire_ai.benchmark import set_seed
from qire_ai.result_store import Recorder

# Breiten als Zweierpotenzen, damit dieselben Werte spaeter fuer die
# Butterfly-Variante benutzbar sind (die braucht d = 2^n).
BREITEN = (128, 256, 512, 1024)
LAGEN = 4
KOEPFE = 4
SEQ = 128
BATCH = 16
VOCAB = 4096
WARMUP = 3
MESS_SCHRITTE = 10
BUDGETS_STUNDEN = (12, 24, 48, 96)
TOKEN_JE_PARAMETER = 20        # Chinchilla-Faustregel


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, maske):
        h = self.n1(x)
        x = x + self.att(h, h, h, attn_mask=maske, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class MiniLM(nn.Module):
    def __init__(self, d: int, lagen: int, heads: int, vocab: int, seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(lagen)])
        self.norm = nn.LayerNorm(d)
        self.kopf = nn.Linear(d, vocab, bias=False)
        self.register_buffer("maske", torch.triu(torch.full((seq, seq), float("-inf")),
                                                 diagonal=1))

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for b in self.blocks:
            x = b(x, self.maske[:t, :t])
        return self.kopf(self.norm(x))


def _messe(d: int, threads: int) -> Dict[str, object]:
    set_seed(4200)
    torch.set_num_threads(threads)
    modell = MiniLM(d, LAGEN, KOEPFE, VOCAB, SEQ)
    opt = torch.optim.AdamW(modell.parameters(), lr=3e-4)
    daten = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
    x, y = daten[:, :-1], daten[:, 1:]
    verlust = nn.CrossEntropyLoss()

    def schritt():
        opt.zero_grad(set_to_none=True)
        aus = modell(x)
        verlust(aus.reshape(-1, VOCAB), y.reshape(-1)).backward()
        opt.step()

    for _ in range(WARMUP):
        schritt()
    t0 = time.perf_counter()
    for _ in range(MESS_SCHRITTE):
        schritt()
    dauer = time.perf_counter() - t0

    par = sum(p.numel() for p in modell.parameters())
    token = BATCH * SEQ * MESS_SCHRITTE
    tps = token / dauer
    # Adam haelt zwei Zusatzkopien; dazu Gradienten. Faktor 4 auf die Gewichte,
    # dieselbe Rechnung wie im Handoff fuer die dichte 14.400er Schicht.
    speicher_mb = par * 4 * 4 / 1024 ** 2
    return {"breite": d, "lagen": LAGEN, "koepfe": KOEPFE, "seq": SEQ,
            "batch": BATCH, "vocab": VOCAB, "parameter": par,
            "sekunden_je_schritt": dauer / MESS_SCHRITTE,
            "token_je_sekunde": tps, "zustands_mb": speicher_mb,
            "threads": threads, "mess_schritte": MESS_SCHRITTE}


def run(breiten: Sequence[int] = BREITEN, threads: int = 6) -> Recorder:
    rec = Recorder(
        experiment_id="T-000",
        description=("Durchsatzkalibrierung: wie gross darf das Sprachmodell auf "
                     "diesem Rechner sein? Kalibrierung, kein Experiment."),
        config={"breiten": list(breiten), "lagen": LAGEN, "koepfe": KOEPFE,
                "seq": SEQ, "batch": BATCH, "vocab": VOCAB,
                "warmup": WARMUP, "mess_schritte": MESS_SCHRITTE,
                "token_je_parameter": TOKEN_JE_PARAMETER,
                "budgets_stunden": list(BUDGETS_STUNDEN), "threads": threads,
                "art": "kalibrierung -- keine Hypothese, kein p-Wert"},
    )
    print("\n" + "=" * 94)
    print("T-000 -- TRAININGSDURCHSATZ")
    print(f"{LAGEN} Lagen, {KOEPFE} Koepfe, Sequenz {SEQ}, Batch {BATCH}, "
          f"Vokabular {VOCAB}, {threads} Threads")
    print("=" * 94)
    head = (f'  {"Breite":>7} {"Parameter":>12} {"s/Schritt":>10} '
            f'{"Token/s":>10} {"Adam-MB":>9}')
    print(head); print("  " + "-" * (len(head) - 2))
    for d in breiten:
        z = _messe(d, threads)
        rec.add(**z)
        print(f'  {d:>7} {z["parameter"]:>12,} {z["sekunden_je_schritt"]:>10.3f} '
              f'{z["token_je_sekunde"]:>10,.0f} {z["zustands_mb"]:>9,.0f}')
        rec.save()
    return rec


def analyse(rec_or_id) -> Dict[str, object]:
    if isinstance(rec_or_id, Recorder):
        records = rec_or_id.records
    else:
        from qire_ai.result_store import load
        records = load(rec_or_id)["records"]
    rows = sorted(records, key=lambda r: r["breite"])
    out: Dict[str, object] = {}

    print()
    print("=" * 94)
    print(f"ABLEITUNG -- groesste trainierbare Breite je Zeitbudget")
    print(f"Kriterium: mindestens {TOKEN_JE_PARAMETER} Token je Parameter "
          f"(Chinchilla-Faustregel)")
    print("=" * 94)
    head = (f'  {"Budget":>8} {"Breite":>7} {"Parameter":>12} {"Token im Budget":>17} '
            f'{"Token/Par.":>11}')
    print(head); print("  " + "-" * (len(head) - 2))
    empfehlung: Dict[int, object] = {}
    for stunden in BUDGETS_STUNDEN:
        beste = None
        for r in rows:
            token = r["token_je_sekunde"] * stunden * 3600
            verhaeltnis = token / r["parameter"]
            if verhaeltnis >= TOKEN_JE_PARAMETER:
                beste = (r, token, verhaeltnis)
        if beste is None:
            print(f'  {stunden:>7}h {"--":>7} {"--":>12} {"--":>17} '
                  f'{"keine Breite reicht":>11}')
            empfehlung[stunden] = None
            continue
        r, token, verhaeltnis = beste
        print(f'  {stunden:>7}h {r["breite"]:>7} {r["parameter"]:>12,} '
              f'{token:>17,.0f} {verhaeltnis:>11,.0f}')
        empfehlung[stunden] = {"breite": r["breite"], "parameter": r["parameter"],
                               "token": token, "token_je_parameter": verhaeltnis}
    out["empfehlung"] = empfehlung

    print()
    print("EINORDNUNG")
    groesste = max(r["breite"] for r in rows)
    print(f"  Der Uebergangspunkt der Butterfly-Struktur liegt laut Tagebuch")
    print(f"  (5. August, 7 Messreihen) bei 16.384 Dimensionen. Die groesste hier")
    print(f"  vermessene Breite ist {groesste}. Zwischen dem, was trainierbar ist,")
    print(f"  und dem, wo die Struktur gewinnt, liegen damit Faktor "
          f"{16384 / groesste:.0f} und mehr.")
    print(f"  Das ist keine Panne, sondern die Lage, aus der die T-Serie folgt:")
    print(f"  Qualitaet wird bei trainierbarer Breite gemessen, Effizienz dort,")
    print(f"  wo sie auftritt. Beides gehoert getrennt berichtet.")
    print("=" * 94)
    out["groesste_vermessene_breite"] = groesste
    return out


if __name__ == "__main__":
    recorder = run()
    try:
        analysis = analyse(recorder)
    except Exception as exc:                      # noqa: BLE001
        print(f"\n  Auswertung fehlgeschlagen: {exc!r}")
        analysis = {"error": repr(exc)}
    recorder.save(analysis=analysis)
