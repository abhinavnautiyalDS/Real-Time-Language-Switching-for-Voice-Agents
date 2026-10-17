# Dataset preparation policy

## Task and labels
This is utterance-level language classification, not script classification. Romanized Hindi remains `hi-IN`; Romanized Punjabi remains `pa-IN`. Latin script does not imply English. `none` means unsupported language or combinations according to the project policy, not low confidence or non-Latin script.

## Required columns
At minimum, each row must have text and category. Preserve provenance fields whenever available, such as source dataset, source URL, source row ID, source split, source group ID, synthetic flag, generator/template ID, license status, and review status.

## Validation
- Reject blank text.
- Stop on unknown categories.
- Derive final `label` from the canonical category map.
- Normalize text for duplicate detection but preserve original text.
- Stop for normalized-identical text with conflicting labels.
- Report category counts before and after deduplication.
- Do not fabricate rows to reach category targets.

## Splits and leakage
Keep related records together using source group IDs or synthetic template IDs. Do not split grammar-correction pairs, paraphrases, translations, or synthetic variants across train/validation/test. Preserve official source test partitions where available. Inspect all split distributions.

## Evaluation
Do not use the test split for training or model selection. Keep real call transcripts in a separate untouched evaluation set. Report macro-F1, per-label precision/recall/F1, confusion matrix, class support, CPU latency p50/p90, and model size.

## Licensing and privacy
Review licenses before redistributing text. Keep raw data local/private when redistribution is not permitted. Never upload credentials, personal data, or private customer call transcripts.
