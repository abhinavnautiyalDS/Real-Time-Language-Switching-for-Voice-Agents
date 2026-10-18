# Fine-tuning artifacts

Large binary artifacts are stored with Git LFS (configured in `.gitattributes`).

```text
artifacts/
├── datasets/
│   └── dataset_splits_audit1.zip
└── models/
    ├── bert_tiny_e2_multilingual.zip
    └── bert_tiny_e4_multilingual.zip
```

## Artifact contents

- `datasets/dataset_splits_audit1.zip`: group-aware train/validation/test CSVs and dataset audit reports.
- `models/bert_tiny_e2_multilingual.zip`: fine-tuned BERT-Tiny E2 model, tokenizer, label mapping, test metrics and evaluation reports.
- `models/bert_tiny_e4_multilingual.zip`: fine-tuned BERT-Tiny E4 model, tokenizer, label mapping, test metrics and evaluation reports.

Upload the files from the Colab runtime using Git LFS. Keep real-call transcripts and private/customer data out of this public repository. Confirm source licenses permit redistribution before publishing dataset text.
