from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text: str, model_dir: str, max_length: int = 96) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    latency_ms = (time.perf_counter() - start) * 1000
    idx = int(torch.argmax(probs))
    return {
        "label": model.config.id2label[idx],
        "confidence": float(probs[idx]),
        "probabilities": {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))},
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/bert-tiny-language-classifier")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(predict(args.text, args.model_dir))
