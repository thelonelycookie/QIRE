from __future__ import annotations

"""Datenaufbereitung fuer das deutsche Sprachmodell (Phase 3).

QUELLE
Deutsche Wikipedia, Stand 20231101, als Parquet ueber die HuggingFace-API.
20 Shards à 745 MB, zusammen 14,6 GB -- gebraucht wird davon **einer**.

WARUM WIKIPEDIA UND NICHT GUTENBERG
Beide waren erreichbar. Gutenberg liefert deutsches 19. Jahrhundert, Wikipedia
modernes Deutsch. Fuer etwas, das am Ende wie ein Chatbot antworten soll, ist
die zweite Wahl die richtige.

WAS DAZUGEHOERT GESAGT ZU WERDEN
Deutsch ist fuer ein Modell dieser Groesse schwerer als Englisch: groesserer
Wortschatz, Komposita, reiche Morphologie. Der Gegenhebel ist der
Tokenisierer -- BPE zerlegt Komposita in wiederverwendbare Stuecke, waehrend
eine Wortebene an jedem "Donaudampfschifffahrtsgesellschaft" scheitert. Ohne
BPE waere die Aufgabe fuer 5,3 Millionen Parameter aussichtslos; mit BPE ist
sie schwer, aber sinnvoll.

WORTSCHATZGROESSE -- eine bewusste Festlegung
Voreingestellt sind 4.096 Stuecke, damit die Parameterzahl mit der
T-000-Kalibrierung uebereinstimmt (5.289.472 bei Breite 256, worauf das
24-Stunden-Budget beruht). Ein groesserer Wortschatz waere fuer Deutsch
besser, verschiebt aber die Parameterzahl und damit die Kalibrierung: Bei
Breite 256 kostet jede Verdopplung des Wortschatzes rund 1 Million Parameter
allein an Einbettung. Wer ihn aendert, muss T-000 neu rechnen.

Run:
  python -m quantum_sim.korpus_deutsch pruefen     # nur Groessen anzeigen
  python -m quantum_sim.korpus_deutsch laden       # Shard herunterladen
  python -m quantum_sim.korpus_deutsch aufbereiten # Text + BPE + Token
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

WURZEL = Path(__file__).resolve().parent.parent
KORPUS = WURZEL / "results" / "korpus"
API = ("https://huggingface.co/api/datasets/wikimedia/wikipedia"
       "/parquet/20231101.de/train")

SHARDS = 1                 # ein Shard reicht: rund 175 Mio. BPE-Token
VOCAB = 4096               # siehe Kopf -- an T-000 gebunden
BPE_TRAINIER_MB = 200      # BPE auf einem Ausschnitt trainieren, wie ueblich
MIN_ZEICHEN = 200          # sehr kurze Artikel (Begriffsklaerungen) verwerfen
VAL_ANTEIL = 0.005         # 0,5 % als Validierung, der Rest Training


def _mb(pfad: Path) -> float:
    return pfad.stat().st_size / 1024 ** 2 if pfad.exists() else 0.0


def shard_urls() -> List[str]:
    return requests.get(API, timeout=30).json()


def pruefen() -> None:
    """Zeigt an, was heruntergeladen wuerde -- ohne etwas zu laden."""
    urls = shard_urls()
    kopf = requests.head(urls[0], timeout=30, allow_redirects=True)
    gr = int(kopf.headers.get("Content-Length", 0)) / 1024 ** 2
    print("=" * 78)
    print("KORPUS -- was heruntergeladen wuerde")
    print("=" * 78)
    print(f"  Quelle          deutsche Wikipedia 20231101 (HuggingFace)")
    print(f"  Shards gesamt   {len(urls)} à {gr:,.0f} MB = {gr * len(urls) / 1024:,.1f} GB")
    print(f"  davon geladen   {SHARDS} ({gr * SHARDS:,.0f} MB)")
    print(f"  Ziel            {KORPUS}")
    print(f"  Platzbedarf     etwa {gr * SHARDS * 2:,.0f} MB "
          f"(Parquet + Text + Token)")
    print("=" * 78)
    for i in range(SHARDS):
        z = KORPUS / f"de_{i}.parquet"
        print(f"  {z.name}: " + (f"schon da ({_mb(z):,.0f} MB)" if z.exists()
                                 else "fehlt"))


def laden() -> List[Path]:
    """Laedt die Shards. Bereits vorhandene werden uebersprungen."""
    KORPUS.mkdir(parents=True, exist_ok=True)
    urls = shard_urls()
    ziele = []
    for i in range(SHARDS):
        ziel = KORPUS / f"de_{i}.parquet"
        ziele.append(ziel)
        if ziel.exists():
            print(f"  {ziel.name} schon vorhanden ({_mb(ziel):,.0f} MB)")
            continue
        print(f"  lade {ziel.name} ...")
        t0 = time.perf_counter()
        with requests.get(urls[i], stream=True, timeout=120) as r:
            r.raise_for_status()
            gesamt = int(r.headers.get("Content-Length", 0))
            geladen = 0
            with open(ziel, "wb") as f:
                for brocken in r.iter_content(chunk_size=1 << 20):
                    f.write(brocken)
                    geladen += len(brocken)
                    if gesamt and geladen % (50 << 20) < (1 << 20):
                        print(f"    {geladen / 1024**2:,.0f} / "
                              f"{gesamt / 1024**2:,.0f} MB")
        print(f"    fertig, {_mb(ziel):,.0f} MB in "
              f"{time.perf_counter() - t0:,.0f} s")
    return ziele


def text_ziehen() -> Path:
    """Parquet -> eine Textdatei. Sehr kurze Artikel fliegen raus."""
    import pyarrow.parquet as pq

    ziel = KORPUS / "de_text.txt"
    if ziel.exists():
        print(f"  {ziel.name} schon vorhanden ({_mb(ziel):,.0f} MB)")
        return ziel
    behalten = verworfen = 0
    with open(ziel, "w", encoding="utf-8") as aus:
        for i in range(SHARDS):
            quelle = KORPUS / f"de_{i}.parquet"
            datei = pq.ParquetFile(quelle)
            print(f"  {quelle.name}: {datei.metadata.num_rows:,} Artikel")
            for stapel in datei.iter_batches(batch_size=2000, columns=["text"]):
                for t in stapel.column("text").to_pylist():
                    if t and len(t) >= MIN_ZEICHEN:
                        aus.write(t.replace("\n\n", "\n").strip())
                        aus.write("\n\n")
                        behalten += 1
                    else:
                        verworfen += 1
    print(f"  behalten {behalten:,} Artikel, verworfen {verworfen:,} "
          f"(unter {MIN_ZEICHEN} Zeichen)")
    print(f"  {ziel.name}: {_mb(ziel):,.0f} MB")
    return ziel


def tokenisierer_bauen(text: Path) -> Path:
    """BPE auf einem Ausschnitt trainieren -- das ist gaengige Praxis und
    spart viel Zeit, ohne die Zerlegung merklich zu aendern."""
    from tokenizers import ByteLevelBPETokenizer

    ziel = KORPUS / f"bpe_{VOCAB}"
    if (ziel.with_suffix(".json")).exists():
        print(f"  Tokenisierer schon vorhanden")
        return ziel.with_suffix(".json")
    ausschnitt = KORPUS / "bpe_train.txt"
    if not ausschnitt.exists():
        print(f"  schreibe {BPE_TRAINIER_MB} MB Ausschnitt fuer das BPE-Training")
        with open(text, encoding="utf-8") as ein, \
             open(ausschnitt, "w", encoding="utf-8") as aus:
            rest = BPE_TRAINIER_MB * 1024 ** 2
            for zeile in ein:
                aus.write(zeile)
                rest -= len(zeile.encode("utf-8"))
                if rest <= 0:
                    break
    print(f"  trainiere BPE mit {VOCAB} Stuecken ...")
    t0 = time.perf_counter()
    tok = ByteLevelBPETokenizer()
    tok.train([str(ausschnitt)], vocab_size=VOCAB, min_frequency=2,
              special_tokens=["<|endoftext|>"])
    tok.save(str(ziel.with_suffix(".json")))
    print(f"    fertig in {time.perf_counter() - t0:,.0f} s")
    return ziel.with_suffix(".json")


def tokenisieren(text: Path, tok_datei: Path) -> None:
    """Text -> uint16-Feld. uint16 reicht, solange VOCAB < 65.536."""
    from tokenizers import Tokenizer

    zug, val = KORPUS / "train.npy", KORPUS / "val.npy"
    if zug.exists() and val.exists():
        print(f"  Token schon vorhanden: {_mb(zug):,.0f} + {_mb(val):,.0f} MB")
        return
    assert VOCAB < 65536, "uint16 reicht nicht mehr -- VOCAB zu gross"
    tok = Tokenizer.from_file(str(tok_datei))
    stuecke, gesamt = [], 0
    t0 = time.perf_counter()
    with open(text, encoding="utf-8") as ein:
        puffer = []
        for zeile in ein:
            puffer.append(zeile)
            if len(puffer) >= 20000:
                for kod in tok.encode_batch(puffer):
                    stuecke.append(np.array(kod.ids, dtype=np.uint16))
                gesamt += sum(len(s) for s in stuecke[-len(puffer):])
                print(f"    {gesamt / 1e6:,.1f} Mio. Token "
                      f"({time.perf_counter() - t0:,.0f} s)")
                puffer = []
        if puffer:
            for kod in tok.encode_batch(puffer):
                stuecke.append(np.array(kod.ids, dtype=np.uint16))
    alle = np.concatenate(stuecke)
    schnitt = int(len(alle) * (1 - VAL_ANTEIL))
    np.save(zug, alle[:schnitt])
    np.save(val, alle[schnitt:])
    print(f"  Training  {len(alle[:schnitt]) / 1e6:,.1f} Mio. Token")
    print(f"  Validierung {len(alle[schnitt:]) / 1e6:,.1f} Mio. Token")

    # Zeichen je Token -- die Zahl, an der man sieht, ob BPE fuer Deutsch taugt
    zeichen = text.stat().st_size
    print(f"  Zeichen je Token: {zeichen / len(alle):.2f} "
          f"(englisches BPE liegt typisch bei 4; deutlich weniger heisst, "
          f"dass Deutsch staerker zerfaellt)")


def aufbereiten() -> None:
    print("=" * 78)
    print("KORPUS AUFBEREITEN")
    print("=" * 78)
    laden()
    text = text_ziehen()
    tok = tokenisierer_bauen(text)
    tokenisieren(text, tok)
    print("=" * 78)


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else "pruefen"
    if befehl == "pruefen":
        pruefen()
    elif befehl == "laden":
        laden()
    elif befehl == "aufbereiten":
        aufbereiten()
    else:
        print(f"unbekannt: {befehl}. Erlaubt: pruefen | laden | aufbereiten")
        raise SystemExit(1)
