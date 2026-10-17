from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import load_and_validate, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--category-map", default="configs/category_to_label.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    category_map = load_json(args.category_map)
    df, report = load_and_validate(args.input, category_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def first_nonempty(row, columns):
        for col in columns:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                value = str(row[col]).strip()
                if value.lower() not in {"nan", "none", "null"}:
                    return value
        return ""

    def group_id(row):
        source = first_nonempty(row, ["source_dataset", "source", "source_file"]) or "unknown_source"
        explicit = first_nonempty(row, ["source_group_id", "group_id"])
        if explicit:
            return f"source::{source}::{explicit}"
        synthetic = str(row.get("is_synthetic", "")).strip().lower() in {"true", "1", "yes"}
        template = first_nonempty(row, ["template_id"])
        generator = first_nonempty(row, ["generator"])
        if synthetic and template:
            return f"template::{source}::{generator}::{template}"
        return f"text::{row['_text_sha256']}"

    df["_group_id"] = df.apply(group_id, axis=1)

    # Reserve whole source groups marked as official test.
    test_names = {"test", "testing", "official_test", "public_test", "test_public"}
    if "source_split" in df.columns:
        split_norm = df["source_split"].astype("string").fillna("").str.strip().str.lower()
        official_groups = set(df.loc[split_norm.isin(test_names), "_group_id"])
    else:
        official_groups = set()

    official_test = df[df["_group_id"].isin(official_groups)].copy()
    pool = df[~df["_group_id"].isin(official_groups)].copy()

    def holdout(frame, folds, fold_index=0):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
        dummy = np.zeros((len(frame), 1), dtype=np.uint8)
        train_i, holdout_i = next(splitter.split(dummy, frame["label"], frame["_group_id"]))
        return frame.iloc[holdout_i], frame.iloc[train_i]

    test_new, remainder = holdout(pool, 10)
    validation, train = holdout(remainder, 9)
    test = pd.concat([official_test, test_new], axis=0)

    sets = {name: set(frame["_group_id"]) for name, frame in {
        "train": train, "validation": validation, "test": test
    }.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])

    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        frame.drop(columns=["_normalized_text", "_group_id"], errors="ignore").to_csv(
            output_dir / f"{name}.csv.gz", index=False, compression="gzip"
        )

    report.update({
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_group_overlap": False,
        "seed": args.seed,
        "split_method": "StratifiedGroupKFold; official source test groups reserved for test",
    })
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
