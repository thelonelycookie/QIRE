# -*- coding: utf-8 -*-
"""Daten fuer I-043-1B (Strukturadapter auf LLaeMmlein-1B): deutscher Text -> Token-Straeme (uint16) mit festem Split.

NUR AUF DEM LOGIN-KNOTEN ausfuehren (hat Internet; Rechenknoten nicht). Aufruf ueber MPI/daten_i043.sh.

QUELLE (Standard "wikipedia"): HF-Datensatz wikimedia/wikipedia, Ausgabe 20231101.de (deutsche Wikipedia, Stand 01.11.2023).
  LIZENZ: CC BY-SA 3.0 und GFDL (Datensatzkarte). Frei nutzbar, auch fuer Training und Veroeffentlichung der Ergebnisse; bei Weitergabe von Text
  Quellenangabe und Weitergabe unter gleichen Bedingungen. In der Arbeit: "Deutschsprachige Wikipedia (wikimedia/wikipedia, 20231101.de),
  CC BY-SA 3.0 / GFDL". Es werden keine Texte veroeffentlicht, nur Messwerte.
  WARUM Wikipedia: frei lizenziert, sehr gross (Zehn- bis Hundertmillionen Tokens, kein Wiederholen noetig), sauber getrennt in Artikel (Split nach
  Artikel), allgemeiner Text statt Einzelsatz-Treebank; kein Konto/Token noetig. GRENZE: LLaeMmlein wurde auf deutschem Web-Text (RedPajama-V2 deutsch)
  vortrainiert, Wikipedia kann teilweise darin vorkommen. Deshalb ist der Verlust des unveraenderten Modells als Bezug Pflicht (Arm "basis");
  gemessen wird der GEWINN durch Feinabstimmung bei gleicher Parameterzahl, nicht der absolute Verlust.
FALLBACK ("gsd"): UD_German-GSD (Universal Dependencies, CC BY-SA 4.0, wie I-045): nur ~0,5 Mio Tokens, daher nur mit Profil "gsd" (kleines Budget).

SPLIT: nach Artikel, deterministisch ueber SHA-1 der Artikel-ID: Bucket = SHA-1 mod 100; Bucket < 6 -> Test, 6..8 -> Validierung, sonst Training;
  zusaetzlich nur 40 % der Artikel (zweiter Hash), damit die Auswahl ueber die ganze Datei streut (Artikel liegen nach Seiten-ID geordnet).
  Test wird nur am Ende gelesen. Je Artikel hoechstens 4096 Tokens (Anfang), Artikel unter 200 Zeichen entfallen, Artikel durch </s> getrennt.
AUSGABE (Ordner --ziel, Standard /ptmp/$USER/i043): train.npy, val.npy, test.npy (uint16, Token-IDs des LLaeMmlein-Tokenizers), meta.json
  (Quelle, Lizenz, Tokenzahlen, Zeichenzahlen von val/test, SHA-256 der Dateien, Tokenizer).

    python MPI/daten_i043.py [--ziel DIR] [--quelle wikipedia|gsd] [--train-tokens 31000000] [--val-tokens 1050000] [--test-tokens 2100000] [--shard 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

MODELL = "LSX-UniWue/LLaMmlein_1B"
MAX_ARTIKEL_TOKENS = 4096
MIN_ZEICHEN = 200
LIZENZ_WIKI = "CC BY-SA 3.0 und GFDL (wikimedia/wikipedia, Datensatzkarte)"
LIZENZ_GSD = "CC BY-SA 4.0 (Universal Dependencies UD_German-GSD)"
GSD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-{}.conllu"


ANTEIL = 40          # nur 40 % der Artikel werden verwendet (Hash 2): streut ueber die ganze Parquet-Datei statt nur ueber ihren Anfang (Artikel liegen nach Seiten-ID)


def bucket(artikel_id: str) -> int:
    return int(hashlib.sha1(str(artikel_id).encode("utf-8")).hexdigest()[:8], 16) % 100


def verwendet(artikel_id: str) -> bool:
    return int(hashlib.sha1(str(artikel_id).encode("utf-8")).hexdigest()[8:16], 16) % 100 < ANTEIL


def teil_von(artikel_id: str) -> str:
    b = bucket(artikel_id)
    return "test" if b < 6 else ("val" if b < 9 else "train")


def sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lade_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELL)


def wikipedia(ziel: Path, tok, ziele: dict, shard: int, max_shards: int):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    dateien = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith("20231101.de/") and f.endswith(".parquet"))
    if not dateien:
        raise SystemExit("keine Parquet-Dateien fuer 20231101.de gefunden")
    eos = tok.eos_token_id
    strom = {t: [] for t in ziele}
    zahl = {t: 0 for t in ziele}
    zeichen = {t: 0 for t in ziele}
    artikel = {t: 0 for t in ziele}
    benutzt = []
    for k in range(max_shards):
        name = dateien[(shard + k) % len(dateien)]
        print(f"lade {name} ...", flush=True)
        pfad = hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset")
        benutzt.append(name)
        pf = pq.ParquetFile(pfad)
        for batch in pf.iter_batches(batch_size=1000, columns=["id", "text"]):
            ids, texte = batch.column("id").to_pylist(), batch.column("text").to_pylist()
            wahl = [(i, t) for i, t in zip(ids, texte) if len(t) >= MIN_ZEICHEN and verwendet(i) and zahl[teil_von(i)] < ziele[teil_von(i)]]
            if not wahl:
                if all(zahl[t] >= ziele[t] for t in ziele):
                    break
                continue
            enc = tok([t for _, t in wahl], add_special_tokens=False)["input_ids"]
            for (i, t), ids_tok in zip(wahl, enc):
                teil = teil_von(i)
                if zahl[teil] >= ziele[teil]:
                    continue
                ids_tok = ids_tok[:MAX_ARTIKEL_TOKENS] + [eos]
                strom[teil].append(np.asarray(ids_tok, dtype=np.uint32))
                zahl[teil] += len(ids_tok)
                zeichen[teil] += len(t) + 1
                artikel[teil] += 1
            print(f"  Tokens {zahl}", flush=True)
            if all(zahl[t] >= ziele[t] for t in ziele):
                break
        if all(zahl[t] >= ziele[t] for t in ziele):
            break
    return strom, zahl, zeichen, artikel, {"datei": benutzt, "lizenz": LIZENZ_WIKI, "datensatz": "wikimedia/wikipedia", "ausgabe": "20231101.de"}


def gsd(ziel: Path, tok, ziele: dict, *_):
    eos = tok.eos_token_id
    strom, zahl, zeichen, artikel = {}, {}, {}, {}
    for teil, name in (("train", "train"), ("val", "dev"), ("test", "test")):
        pfad = ziel / f"de_gsd_{name}.conllu"
        if not pfad.exists():
            print(f"lade {GSD_URL.format(name)}", flush=True)
            urllib.request.urlretrieve(GSD_URL.format(name), pfad)
        saetze = [z[len("# text = "):].strip() for z in pfad.read_text(encoding="utf-8").splitlines() if z.startswith("# text = ")]
        ids = []
        for s in tok(saetze, add_special_tokens=False)["input_ids"]:
            ids.extend(s + [eos])
        strom[teil] = [np.asarray(ids, dtype=np.uint32)]
        zahl[teil], zeichen[teil], artikel[teil] = len(ids), sum(len(s) + 1 for s in saetze), len(saetze)
    return strom, zahl, zeichen, artikel, {"datei": [f"de_gsd_{n}.conllu" for n in ("train", "dev", "test")], "lizenz": LIZENZ_GSD,
                                           "datensatz": "UD_German-GSD", "ausgabe": "master"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ziel", default=os.environ.get("I043_DATEN") or f"/ptmp/{os.environ.get('USER', 'user')}/i043")
    ap.add_argument("--quelle", choices=("wikipedia", "gsd"), default="wikipedia")
    ap.add_argument("--train-tokens", type=int, default=31_000_000)
    ap.add_argument("--val-tokens", type=int, default=1_050_000)
    ap.add_argument("--test-tokens", type=int, default=2_100_000)
    ap.add_argument("--shard", type=int, default=0, help="erste Parquet-Datei (0..19); weitere werden bei Bedarf angehaengt")
    ap.add_argument("--max-shards", type=int, default=3)
    a = ap.parse_args()
    ziel = Path(a.ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = lade_tokenizer()
    ziele = {"train": a.train_tokens, "val": a.val_tokens, "test": a.test_tokens}
    strom, zahl, zeichen, artikel, quelle = (wikipedia if a.quelle == "wikipedia" else gsd)(ziel, tok, ziele, a.shard, a.max_shards)
    if a.quelle == "wikipedia" and any(zahl[t] < ziele[t] for t in ziele):
        print(f"WARNUNG: Ziel nicht erreicht {zahl} < {ziele}; mit --max-shards erhoehen", file=sys.stderr)
    meta = {"quelle": ("wikipedia" if a.quelle == "wikipedia" else "gsd"), "quelle_details": quelle, "tokenizer": MODELL, "tokens": {}, "zeichen": {}, "artikel": artikel,
            "sha256": {}, "split": f"SHA-1(Artikel-ID) mod 100: <6 Test, 6-8 Validierung, sonst Training; nur {ANTEIL} % der Artikel (zweiter Hash)" if a.quelle == "wikipedia" else "Split der Treebank (train/dev/test)",
            "max_artikel_tokens": MAX_ARTIKEL_TOKENS, "erstellt": time.strftime("%Y-%m-%d %H:%M:%S")}
    for teil in ("train", "val", "test"):
        arr = np.concatenate(strom[teil]) if strom[teil] else np.zeros(0, dtype=np.uint32)
        if len(arr) and arr.max() >= 65536:
            raise SystemExit("Token-ID >= 65536: uint16 reicht nicht")
        pfad = ziel / f"{teil}.npy"
        np.save(pfad, arr.astype(np.uint16))
        meta["tokens"][teil] = int(len(arr))
        meta["zeichen"][teil] = int(zeichen[teil])
        meta["sha256"][teil] = sha256(pfad)
    (ziel / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s: {ziel}  Tokens {meta['tokens']}  Zeichen {meta['zeichen']}  Artikel {artikel}")
    print("Lizenz:", quelle["lizenz"])


if __name__ == "__main__":
    main()
