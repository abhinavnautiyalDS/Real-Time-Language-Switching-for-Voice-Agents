# Version 4.4 — Dataset

This folder is the home for the Version 4.4 dataset used by the real-time language-switching classifier.

## Expected files
- `train.csv`
- `validation.csv`
- `test.csv`
- `dataset_manifest.json`
- `label_mapping.json`

Keep raw source data and private credentials out of this folder. Record provenance, source split, license status, and synthetic/review status for each example where available. Do not put official test examples into training data.

## Expected CSV schema
At minimum: `text,label`. Prefer retaining audit columns such as `language_category,source_dataset,source_url,source_row_id,group_id,source_split,is_synthetic,review_status,license_status`.

Supported labels: `as-IN,bn-IN,en-IN,gu-IN,hi-IN,kn-IN,ml-IN,mr-IN,none,pa-IN,ta-IN,te-IN`.

Dataset counts and quality status should be written to the manifest after the final split is uploaded.
