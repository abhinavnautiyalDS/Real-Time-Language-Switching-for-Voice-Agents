from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv.gz")
    parser.add_argument("--validation", default="data/processed/validation.csv.gz")
    parser.add_argument("--model-name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--output-dir", default="models/bert-tiny-language-classifier")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, compression="gzip")
    val_df = pd.read_csv(args.validation, compression="gzip")
    for name, frame in [("train", train_df), ("validation", val_df)]:
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")

    labels = sorted(set(train_df["label"].astype(str)) | set(val_df["label"].astype(str)))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_dataset(frame):
        data = frame[["text", "label"]].copy()
        data["labels"] = data["label"].astype(str).map(label2id)
        ds = Dataset.from_pandas(data[["text", "labels"]], preserve_index=False)
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    def metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": p,
            "macro_recall": r,
            "macro_f1": f,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        fp16=bool(__import__("torch").cuda.is_available()),
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(json.dumps(trainer.evaluate(), indent=2))


if __name__ == "__main__":
    main()
