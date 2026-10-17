from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", value).strip().casefold()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_and_validate(
    csv_path: str | Path,
    category_to_label: dict[str, str],
    text_column_candidates: list[str] | None = None,
    category_column_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    text_column_candidates = text_column_candidates or ["text", "sentence", "utterance"]
    category_column_candidates = category_column_candidates or ["category", "language_category"]

    df = pd.read_csv(csv_path, low_memory=False)
    text_col = next((c for c in text_column_candidates if c in df.columns), None)
    category_col = next((c for c in category_column_candidates if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Missing text column. Tried {text_column_candidates}; found {df.columns.tolist()}")
    if category_col is None:
        raise ValueError(f"Missing category column. Tried {category_column_candidates}; found {df.columns.tolist()}")

    if "label" in df.columns:
        df["original_label"] = df["label"].astype("string")

    df["text"] = df[text_col].astype("string").fillna("").str.strip()
    df["category"] = df[category_col].astype("string").fillna("").str.strip().str.lower()
    df["category"] = df["category"].str.replace(r"\s+", "_", regex=True)

    blank = df["text"].eq("")
    blank_count = int(blank.sum())
    df = df.loc[~blank].copy()

    unknown = sorted(set(df["category"]) - set(category_to_label))
    if unknown:
        raise ValueError(f"Unknown categories; review config before proceeding: {unknown}")

    df["label"] = df["category"].map(category_to_label)
    df["_normalized_text"] = df["text"].map(normalize_text)
    df["_text_sha256"] = df["_normalized_text"].map(sha256_text)

    conflict_counts = df.groupby("_text_sha256")["label"].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    if conflicts:
        conflict_rows = df[df["_text_sha256"].isin(conflicts)]
        raise ValueError(
            f"{len(conflicts)} normalized texts have conflicting labels. "
            f"Inspect these rows before training. Sample: {conflict_rows[['text','category','label']].head(10).to_dict('records')}"
        )

    before_dedup = len(df)
    df = df.drop_duplicates("_text_sha256", keep="first").copy()
    report = {
        "input_rows": int(len(pd.read_csv(csv_path, usecols=[text_col], low_memory=False))),
        "blank_rows_removed": blank_count,
        "rows_before_dedup": before_dedup,
        "duplicates_removed": before_dedup - len(df),
        "rows_after_dedup": int(len(df)),
        "text_column": text_col,
        "category_column": category_col,
        "category_counts": {str(k): int(v) for k, v in df["category"].value_counts().items()},
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
    }
    return df, report
