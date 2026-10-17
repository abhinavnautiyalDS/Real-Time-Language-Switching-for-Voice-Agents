from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import load_and_validate, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Final candidate CSV")
    parser.add_argument("--output-dir", default="reports/dataset")
    parser.add_argument("--category-map", default="configs/category_to_label.json")
    args = parser.parse_args()

    category_map = load_json(args.category_map)
    _, report = load_and_validate(args.input, category_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
