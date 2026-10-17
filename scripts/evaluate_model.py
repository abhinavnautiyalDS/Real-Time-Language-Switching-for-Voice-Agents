from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/test.csv.gz")
    parser.add_argument("--model-dir", default="models/bert-tiny-language-classifier")
    parser.add_argument("--output-dir", default="reports/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test, compression="gzip")
    if not {"text", "label"}.issubset(test_df.columns):
        raise ValueError("Test CSV must contain text and label columns.")

    clf = pipeline("text-classification", model=args.model_dir, tokenizer=args.model_dir, device=-1)
    texts = test_df["text"].astype(str).tolist()
    preds = []
    for start in range(0, len(texts), args.batch_size):
        batch = clf(texts[start:start + args.batch_size], truncation=True, batch_size=args.batch_size)
        preds.extend(item["label"] for item in batch)

    labels = sorted(set(test_df["label"].astype(str)) | set(preds))
    report = classification_report(test_df["label"].astype(str), preds, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_df["label"].astype(str), preds, labels=labels).tolist()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "matrix": matrix}, f, ensure_ascii=False, indent=2)
    result_df = test_df.copy()
    result_df["predicted_label"] = preds
    result_df.to_csv(out / "test_predictions.csv.gz", index=False, compression="gzip")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
