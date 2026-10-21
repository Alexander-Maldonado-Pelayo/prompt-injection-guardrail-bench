"""
train/finetune.py — fine-tune a small encoder on text -> {benign, injection}.

Designed for a Colab T4 (16 GB) but runs on any GPU; on CPU it's slow but works.

    python -m train.finetune                                   # DeBERTa-v3-small, defaults
    python -m train.finetune --base answerdotai/ModernBERT-base --epochs 4
    python -m train.finetune --push xRainbowdartx/deberta-injection-hardneg

Reads data/train.jsonl and data/validation.jsonl (run `python -m data.load` first).
Picks the best epoch by validation F1, saves to train/out/best, optionally pushes to the Hub.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "train" / "out"
LABELS = ["benign", "injection"]


def load_split(name: str) -> Dataset:
    rows = [json.loads(l) for l in (REPO / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return Dataset.from_list([{"text": r["text"], "label": int(r["label"])} for r in rows])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    # FPR = benign rows predicted injection / all benign rows
    benign = labels == 0
    fpr = float(((preds == 1) & benign).sum() / max(benign.sum(), 1))
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1, "fpr": fpr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--push", help="HF repo id to push the best model to, e.g. user/name")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
    )

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = load_split("train").map(tokenize, batched=True)
    val_ds = load_split("validation").map(tokenize, batched=True)
    print(f"train={len(train_ds)} val={len(val_ds)} base={args.base}")

    targs = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        fp16=True,                 # T4 supports fp16; set False on CPU
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    final = trainer.evaluate()
    print("best validation:", {k: round(v, 4) for k, v in final.items() if k.startswith("eval_")})

    best = OUT / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    (best / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    print("saved ->", best)

    if args.push:
        model.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print("pushed ->", f"https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
