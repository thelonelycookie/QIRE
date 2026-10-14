# -*- coding: utf-8 -*-
"""QLoRA-Training fuer die Ideen-KI (Plan S3 Bewerter / S4 Erfinder) -- VORLAGE, UNGETESTET auf GPU.

Warum Vorlage: Trainingsdaten (Urteile H1/H0/HU, Maltes Blindnoten, Top-Karten) gibt es erst ab ~150-300 bewerteten
Ideen (Plan §6), und die Cluster-Umgebung ist unbekannt. Getestet ist hier nur die Datenpruefung (`--daten-pruefen`),
die ohne torch/transformers laeuft. Die Trainingsteile folgen den dokumentierten Schnittstellen von transformers/peft/
bitsandbytes; Versionsabweichungen am Cluster sind zu erwarten.

Datenformat (ANNAHME, an IDEEN_KI/schmiede/rueckmeldung.py angleichen):
    JSONL, je Zeile {"prompt": "<Eingabe>", "antwort": "<Soll-Ausgabe>", optional "id": "...", "system": "..."}
    S3 Bewerter:  prompt = Idee (+ Kontext), antwort = Urteil/Note/Begruendung (kurz).
    S4 Erfinder:  prompt = Operator-Anweisung + Kontext, antwort = Top-Idee (1-2 Saetze).

Wiederaufnahme: neuester checkpoint-* in --ausgabe wird fortgesetzt. SIGTERM/SIGUSR1 -> Checkpoint speichern,
Training beenden, Exit-Code 75 (EX_TEMPFAIL = "bitte fortsetzen"; gpu_qlora.sbatch reiht dann neu ein).
Zurueckgehaltene Daten: --val-anteil (Standard 0.1), deterministisch ueber Hash des Prompts -> dieselben Ideen
bleiben bei jedem Neustart und fuer jede Modellversion draussen (Vergleich "schlaegt Basis" nur darauf).

    python MPI/training/qlora_train.py --daten-pruefen --daten ideen.jsonl            # nur pruefen, lokal ausfuehrbar
    python MPI/training/qlora_train.py --basis <HF-Name|Pfad> --daten ideen.jsonl --ausgabe MPI/adapter/s3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

EX_TEMPFAIL = 75


# ------------------------------------------------------------------ Daten (ohne ML-Bibliotheken testbar)
def lade_jsonl(pfad: Path) -> list[dict]:
    zeilen, fehler = [], []
    for i, z in enumerate(Path(pfad).read_text(encoding="utf-8").splitlines(), 1):
        if not z.strip():
            continue
        try:
            d = json.loads(z)
        except ValueError as e:
            fehler.append(f"Zeile {i}: kein JSON ({e})")
            continue
        if not isinstance(d.get("prompt"), str) or not isinstance(d.get("antwort"), str) or not d["prompt"].strip() or not d["antwort"].strip():
            fehler.append(f"Zeile {i}: 'prompt' und 'antwort' muessen nichtleere Texte sein")
            continue
        zeilen.append(d)
    if fehler:
        raise ValueError("Datenfehler:\n  " + "\n  ".join(fehler[:20]) + (f"\n  ... {len(fehler)} insgesamt" if len(fehler) > 20 else ""))
    return zeilen


def ist_validierung(d: dict, anteil: float) -> bool:
    """Deterministische Zuteilung ueber den Hash des Prompts (stabil ueber Neustarts und Modellversionen)."""
    h = int(hashlib.sha256(d["prompt"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < anteil


def teile(daten: list[dict], anteil: float) -> tuple[list[dict], list[dict]]:
    val = [d for d in daten if ist_validierung(d, anteil)]
    train = [d for d in daten if not ist_validierung(d, anteil)]
    return train, val


def pruefe_daten(pfad: Path, val_anteil: float) -> int:
    daten = lade_jsonl(pfad)
    train, val = teile(daten, val_anteil)
    doppelt = len(daten) - len({d["prompt"] for d in daten})
    print(f"{len(daten)} Beispiele, Training {len(train)}, zurueckgehalten {len(val)} (Anteil {val_anteil}), doppelte Prompts: {doppelt}")
    if len(daten) < 150:
        print("WARNUNG: weniger als 150 Beispiele -- laut Plan §6 zu wenig fuer S3/S4.", file=sys.stderr)
    if not val:
        print("WARNUNG: keine zurueckgehaltenen Beispiele -- kein fairer Vergleich moeglich.", file=sys.stderr)
    return 0


def letzter_checkpoint(ausgabe: Path):
    cps = sorted((p for p in Path(ausgabe).glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                 key=lambda p: int(p.name.split("-")[-1]))
    return str(cps[-1]) if cps else None


# ------------------------------------------------------------------ Training (braucht torch, transformers, peft, bitsandbytes)
def trainiere(a) -> int:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    set_seed(a.seed)
    ausgabe = Path(a.ausgabe)
    ausgabe.mkdir(parents=True, exist_ok=True)
    daten = lade_jsonl(Path(a.daten))
    train, val = teile(daten, a.val_anteil)
    if not train:
        print("Keine Trainingsdaten.", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(a.basis, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def kodiere(d):
        prompt = (d.get("system", "") + "\n\n" if d.get("system") else "") + d["prompt"].strip() + "\n\nAntwort:\n"
        p_ids = tok(prompt, add_special_tokens=True, truncation=True, max_length=a.max_laenge // 2)["input_ids"]
        a_ids = tok(d["antwort"].strip() + (tok.eos_token or ""), add_special_tokens=False,
                    truncation=True, max_length=a.max_laenge - len(p_ids))["input_ids"]
        return {"input_ids": p_ids + a_ids, "attention_mask": [1] * (len(p_ids) + len(a_ids)),
                "labels": [-100] * len(p_ids) + a_ids}      # Verlust nur auf der Antwort

    class Liste(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = [kodiere(r) for r in rows]
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    modell = AutoModelForCausalLM.from_pretrained(a.basis, quantization_config=bnb, device_map={"": 0})
    modell = prepare_model_for_kbit_training(modell, use_gradient_checkpointing=True)
    modell = get_peft_model(modell, LoraConfig(r=a.rang, lora_alpha=2 * a.rang, lora_dropout=0.05, bias="none",
                                               target_modules="all-linear", task_type="CAUSAL_LM"))
    modell.print_trainable_parameters()

    stopp = {"angefordert": False}

    def _signal(signum, frame):
        stopp["angefordert"] = True
        print(f"Signal {signum}: Checkpoint speichern und beenden.", flush=True)
    signal.signal(signal.SIGTERM, _signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _signal)

    class SignalStopp(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stopp["angefordert"]:
                control.should_save = True
                control.should_training_stop = True
            return control

    args = TrainingArguments(
        output_dir=str(ausgabe), num_train_epochs=a.epochen, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.akkumulation, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_steps=a.speichere_alle, save_total_limit=3, bf16=bf16, fp16=not bf16,
        optim="paged_adamw_8bit", gradient_checkpointing=True, report_to="none", remove_unused_columns=False,
        seed=a.seed, eval_strategy="steps" if val else "no", eval_steps=a.speichere_alle if val else None,
        per_device_eval_batch_size=a.batch)
    trainer = Trainer(model=modell, args=args, train_dataset=Liste(train), eval_dataset=Liste(val) if val else None,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[SignalStopp()])
    cp = letzter_checkpoint(ausgabe)
    print(f"Fortsetzen ab {cp}" if cp else "Neustart ohne Checkpoint", flush=True)
    trainer.train(resume_from_checkpoint=cp)

    if stopp["angefordert"]:
        print("Unterbrochen, Checkpoint gesichert -> Exit 75 (fortsetzen).", flush=True)
        return EX_TEMPFAIL
    modell.save_pretrained(str(ausgabe / "adapter_final"))
    tok.save_pretrained(str(ausgabe / "adapter_final"))
    import transformers, peft
    (ausgabe / "adapter_final" / "qire_lauf.json").write_text(json.dumps({
        "stufe": a.stufe, "basis": a.basis, "daten": str(a.daten), "n_train": len(train), "n_validierung": len(val),
        "seed": a.seed, "epochen": a.epochen, "lr": a.lr, "rang": a.rang,
        "versionen": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "hinweis": "Auf QIRE spezialisiertes, feinjustiertes Basismodell (Name, Version) -- nicht von Grund auf trainiert."},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--daten", required=True)
    ap.add_argument("--daten-pruefen", action="store_true", help="nur Format/Zahlen pruefen (ohne torch)")
    ap.add_argument("--basis", help="HF-Name oder lokaler Pfad des Basismodells")
    ap.add_argument("--ausgabe", default="MPI/adapter/s3")
    ap.add_argument("--stufe", default="s3", choices=["s3", "s4"])
    ap.add_argument("--val-anteil", type=float, default=0.1)
    ap.add_argument("--epochen", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rang", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--akkumulation", type=int, default=4)
    ap.add_argument("--max-laenge", dest="max_laenge", type=int, default=1024)
    ap.add_argument("--speichere-alle", dest="speichere_alle", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    if a.daten_pruefen:
        return pruefe_daten(Path(a.daten), a.val_anteil)
    if not a.basis:
        ap.error("--basis fehlt")
    return trainiere(a)


if __name__ == "__main__":
    sys.exit(main())
