#!/usr/bin/env python
"""
finetune_whisper.py  -  Fine-tune Whisper (LoRA) to make it BETTER for *you*
============================================================================

IMPORTANT — READ THIS FIRST
---------------------------
You cannot meaningfully run this on the laptop we set up (no NVIDIA GPU).
This script is meant to run on a **GPU**: a free Google Colab / Kaggle notebook,
or any rented cloud GPU (vast.ai, runpod, Lambda). See training/README.md.

WHAT "BETTER" REALISTICALLY MEANS
---------------------------------
Whisper large-v3 is already world-class on general audio. You will NOT beat it
on everything. What fine-tuning DOES give you is a model that is clearly better
on a *specific* target, for example:
  - your own voice / accent
  - a specific language Whisper is weak at
  - a domain vocabulary (medical, legal, gaming, names, slang)
This is the honest, achievable win.

HOW IT WORKS
------------
We use LoRA (Low-Rank Adaptation): instead of retraining all 1.5B parameters
(needs huge GPUs), we train tiny "adapter" layers (~1% of the weights). This
fits on a free Colab T4 and trains in hours, not weeks.

TWO WAYS TO PROVIDE TRAINING DATA
---------------------------------
1) A public dataset, e.g. Common Voice for a language:
     python finetune_whisper.py --dataset common_voice --lang hi

2) YOUR OWN audio (recommended for "make it better for me"):
     Put files in a folder with a metadata.csv:
         data/
           clip001.wav
           clip002.wav
           metadata.csv     # two columns:  file,text
     metadata.csv example:
         file,text
         clip001.wav,"hello this is a test recording"
         clip002.wav,"the quick brown fox"
     Then:
     python finetune_whisper.py --dataset local --data-dir ./data

AFTER TRAINING
--------------
The LoRA adapter is saved to ./whisper-finetuned. To use it in transcribe.py,
merge it into the base model and convert to CTranslate2 — see training/README.md.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

try:
    import evaluate
except ImportError:
    evaluate = None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_local_dataset(data_dir: str) -> DatasetDict:
    """Build a dataset from a folder + metadata.csv (columns: file,text)."""
    import csv

    data_dir = Path(data_dir)
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"metadata.csv not found in {data_dir}")

    rows = []
    with open(meta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"audio": str(data_dir / r["file"]), "sentence": r["text"]})

    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
    split = ds.train_test_split(test_size=0.1, seed=42)
    return DatasetDict(train=split["train"], test=split["test"])


def load_common_voice(lang: str) -> DatasetDict:
    """Load Mozilla Common Voice for a language (needs HF login + dataset terms)."""
    name = "mozilla-foundation/common_voice_17_0"
    train = load_dataset(name, lang, split="train+validation", trust_remote_code=True)
    test = load_dataset(name, lang, split="test", trust_remote_code=True)
    keep = {"audio", "sentence"}
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    test = test.remove_columns([c for c in test.column_names if c not in keep])
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    test = test.cast_column("audio", Audio(sampling_rate=16000))
    return DatasetDict(train=train, test=test)


def make_prepare_fn(processor: WhisperProcessor):
    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer adds a BOS token; the model adds it again, so drop it here.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune Whisper on a GPU.")
    ap.add_argument("--base-model", default="openai/whisper-large-v3",
                    help="try openai/whisper-small first to learn the pipeline fast")
    ap.add_argument("--dataset", choices=["local", "common_voice"], default="local")
    ap.add_argument("--data-dir", default="./data", help="for --dataset local")
    ap.add_argument("--lang", default="en", help="language code (whisper + dataset)")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--output-dir", default="./whisper-finetuned")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--load-8bit", action="store_true",
                    help="load base model in 8-bit (needs bitsandbytes; saves VRAM)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("\n*** WARNING: no CUDA GPU detected. Training will be far too slow.")
        print("*** Run this on Google Colab / Kaggle / a rented GPU. See README.\n")

    print(f"[data] loading dataset ({args.dataset}) ...")
    if args.dataset == "local":
        ds = load_local_dataset(args.data_dir)
    else:
        ds = load_common_voice(args.lang)
    print(ds)

    print(f"[model] loading processor + base model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.lang, task=args.task
    )

    ds = ds.map(
        make_prepare_fn(processor),
        remove_columns=ds["train"].column_names,
        num_proc=1,
    )

    model_kwargs = {}
    if args.load_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # ----- LoRA: train tiny adapters only -----
    from peft import LoraConfig, get_peft_model

    if args.load_8bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    metric = evaluate.load("wer") if evaluate else None

    def compute_metrics(pred):
        if metric is None:
            return {}
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        report_to=["tensorboard"],
        remove_unused_columns=False,   # required for PEFT + Whisper
        label_names=["labels"],        # required for PEFT
        predict_with_generate=False,   # generation eval is slow; WER computed post-hoc
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("[train] starting ...")
    trainer.train()

    print(f"[save] writing LoRA adapter + processor to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("[done] To use it in transcribe.py, merge + convert to CTranslate2 "
          "(see training/README.md).")


if __name__ == "__main__":
    main()
