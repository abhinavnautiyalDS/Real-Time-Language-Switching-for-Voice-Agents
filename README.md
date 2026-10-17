# Multilingual Voice Language Classifier

Language classification for voice-agent pipelines. Predict the language of an utterance, not merely its script, so downstream LLM/TTS components can switch languages correctly.

## Target labels

- `en-IN` — English
- `hi-IN` — Hindi, including supported Hindi-English Hinglish
- `pa-IN` — Punjabi, including supported Punjabi-Hindi/English combinations
- `bn-IN` — Bengali, including Bengali-English
- `gu-IN` — Gujarati, including Gujarati-English
- `kn-IN` — Kannada, including Kannada-English
- `ml-IN` — Malayalam, including Malayalam-English
- `mr-IN` — Marathi, including Marathi-English and Marathi-Hindi
- `te-IN` — Telugu, including Telugu-English
- `as-IN` — Assamese, including Assamese-English
- `ta-IN` — Tamil, including Tamil-English
- `none` — unsupported languages/combinations under project policy

Romanized Hindi remains `hi-IN`; Romanized Punjabi remains `pa-IN`. Latin script alone does not imply English.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── label_map.json
│   ├── category_to_label.json
│   └── dataset_config.json
├── notebooks/
│   ├── 01_prepare_dataset.ipynb
│   └── 02_finetune_bert_tiny.ipynb
├── scripts/
│   ├── validate_dataset.py
│   ├── prepare_splits.py
│   └── evaluate_model.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── inference.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
│   ├── dataset/
│   └── evaluation/
└── docs/
    └── DATASET_POLICY.md
```

## Workflow

1. Upload the finalized candidate CSV in Colab.
2. Validate categories, labels, empty text, duplicates, label conflicts, and provenance.
3. Preserve source metadata and create group-safe train/validation/test splits.
4. Inspect category and label reports; resolve unexplained shortages and conflicts before training.
5. Use train for optimization, validation for model selection, and test only for final evaluation.
6. Keep real-call evaluation data separate from training data.
7. Record configuration, dataset hash, random seed, metrics, and model revision.

## Data licensing and privacy

Do not commit third-party text unless its source license permits redistribution. Record source URLs, licenses, attribution requirements, source row/group IDs, and transformations. This repository is public, so consider making it private until licensing is verified. For large datasets and model weights, use Git LFS or an appropriate artifact host. Never push credentials, private call transcripts, or customer data.

## Category count note

Latest requested targets include Punjabi-English 5,000 rows, Marathi-English 10,000, and Marathi-Hindi 10,000. The preparation notebook must report actual counts and the difference from the earlier 500,000-row headline; it must not invent rows to meet a target.

## Metrics

Report accuracy, macro-F1, per-label precision/recall/F1, confusion matrix, class support, CPU latency p50/p90, model size, and performance on untouched real call transcripts.
