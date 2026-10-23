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

## Colab notebooks

- [01 — Prepare dataset](notebooks/01_prepare_dataset.ipynb): upload the final candidate CSV, validate categories and labels, audit counts/duplicates, create group-aware splits, and optionally push raw/interim/processed data with Git LFS.
- [02 — Fine-tune BERT-Tiny baseline](notebooks/02_finetune_bert_tiny.ipynb): load the splits, inspect tokenizer coverage, fine-tune, evaluate the test split, measure CPU latency, and optionally push the model.

Open either notebook on GitHub and choose **Open in Colab**, or download it and upload it to Colab.

## Workflow

1. Run notebook 01 and upload the finalized candidate CSV.
2. Validate categories, labels, empty text, duplicates, label conflicts, and provenance.
3. Preserve source metadata and create group-safe train/validation/test splits.
4. Inspect category and label reports; resolve unexplained shortages and conflicts before training.
5. Use train for optimization, validation for model selection, and test only for final evaluation.
6. Keep real-call evaluation data separate from training data.
7. Record configuration, dataset hash, random seed, metrics, and model revision.

The notebook's latest category targets total 440,000 rows after the requested reductions; this differs from the earlier 500,000 headline. The audit reports actual counts instead of creating replacement rows.

## Data licensing and privacy

Do not commit third-party text unless its source license permits redistribution. Record source URLs, licenses, attribution requirements, source row/group IDs, and transformations. This repository is public, so consider making it private until licensing is verified. For large datasets and model weights, use Git LFS or an appropriate artifact host. Never push credentials, private call transcripts, or customer data.

## Category count note

Latest requested targets include Punjabi-English 5,000 rows, Marathi-English 10,000, and Marathi-Hindi 10,000. The preparation notebook must report actual counts and the difference from the earlier 500,000-row headline; it must not invent rows to meet a target.

## Metrics

Report accuracy, macro-F1, per-label precision/recall/F1, confusion matrix, class support, CPU latency p50/p90, model size, and performance on untouched real call transcripts.

---

## Branch-specific problem and solution

**Problem being addressed:** Dataset preparation: prevent translation-pair leakage, duplicates, noisy labels and unreproducible splits.

**Solution and evidence:** The sections below connect the broader project experiments to this branch. Treat claims as verified only where a dataset archive, model evaluation, committed configuration or changelog supports them; otherwise confirm the exact intervention against this branch's commits before presenting it as historical fact.


## Engineering history: problem → change → evidence

### Why the project started
A voice agent receives speech-to-text (STT) transcripts and must decide which language to use. Script alone is not language: Hindi can be Romanized, English and Hinglish can share Latin script, and short turns such as “hello”, “okay” or “haan ji” may not contain enough evidence to switch.

### Iteration logic
For each model/data iteration, the engineering question is: what failed in the previous attempt, what changed, and what evidence shows whether it helped?

- **Script classifier:** fast script signal, but cannot distinguish English from Romanized Hindi or resolve code-mixing. It remains a separate task, not a universal language detector.
- **Classical English/Hinglish baseline:** TF-IDF character n-grams + LinearSVC with OTHER/UNCERTAIN rejection. Its saved benchmark reports 9,389 test rows, 97.2628% accuracy, 97.0005% macro-F1, p50 ≈ 1.337 ms and p95 ≈ 1.509 ms. This binary benchmark is not directly comparable to the later 12-label benchmark.
- **Qwen + LoRA experiment:** explored contextual/generative classification. LoRA trains small adapter matrices while most base weights stay frozen; inference still needs the base Qwen model. Exact checkpoint, parameter count and LoRA settings must come from the committed config/logs, not be guessed from the family name.
- **BERT-Tiny E2/E4:** moved toward compact fixed-label classification for a CPU-constrained pipeline. The current E2 v4.4 config uses 2 transformer layers, hidden size 128, 2 attention heads, intermediate size 512, vocabulary 30,522 and 12 output labels.
- **Version 4 data audit:** the v4 notes report removing 9,992 of 10,000 synthetic Punjabi rows because the samples were mostly Hindi-looking Devanagari text with Punjabi labels. Repetitive templates were trimmed, STT-like noise was added and group-aware splits were rebuilt.
- **Versions 4.1–4.3:** treat archive sizes as distinct snapshots. Where a branch has no build report/changelog, describe the exact purpose as a hypothesis and verify it against commits before claiming it as historical fact.
- **Version 4.4:** targets false switches on short acknowledgements. It adds 15,783 acknowledgement rows to none and relabels 294 existing rows to none, including native/Romanized forms and case/punctuation variants. “Okay” alone is neutral; “Okay, I want to cancel my order” contains language-bearing evidence.

### Current multilingual snapshot
The uploaded v4.4 dataset has 903,414 rows, 12 labels and 141 categories: 723,240 train / 90,066 validation / 90,108 test. The uploaded BERT-Tiny E2 v4.4 report records 99.868% accuracy and 99.868% macro-F1 on the prepared test split; none has the lowest reported class F1 at about 99.446%. These are offline prepared-test results, not a guarantee for live STT calls.

### Data sources and model references
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA) — human-annotated Hindi/English/Hinglish source.
- [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC) — Romanized Hinglish and English translations.
- [Tatoeba English–Hindi selection](https://www.manythings.org/anki/hin-eng.zip) — paired monolingual examples.
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID) — Indic language-identification reference.
- [Google BERT-Tiny base checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- [Sarvam Text Language Detection API](https://docs.sarvam.ai/api-reference/text/identify-language) — hosted competitor/reference, not a downloadable open-source model.
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html) · [CLD3](https://github.com/google/cld3) · [Lingua](https://github.com/pemistahl/lingua).

### Open-source release gates
Before release, publish: (1) accuracy/precision/recall/macro-F1, confusion matrix, calibration and failure analysis; (2) leakage/near-duplicate checks, annotation audit and independent unseen data; (3) CPU p50/p90/p95, throughput, RAM, cold/warm start and concurrency; (4) noisy, misspelled, short/long, mixed-script, unsupported-language and malformed-Unicode tests; (5) same-sample competitor benchmarks; (6) API validation, timeouts, concurrency and rollback; (7) dataset/model/dependency license, attribution, privacy and hostile-input checks; and (8) model card, reproducible evaluation, pinned install, known limitations and versioned artifacts. Report Sarvam's network-inclusive latency separately from local CPU inference.

