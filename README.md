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

## Streamlit demo

The repository includes `app.py`, a small interactive UI for the saved fine-tuned classifier. Enter a sentence to view the predicted language tag, top-class softmax score, and scores for all 12 classes.

### Run locally

From the repository root:

```bash
git lfs install
git lfs pull origin main
pip install -r requirements.txt
streamlit run app.py
```

The model is loaded from `models/multilingual-bert-tiny-final/` and inference runs on CPU.

### Deploy with Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **Create app** and choose this GitHub repository.
3. Select branch `main`, main file path `app.py`, then deploy.
4. If deployment cannot find the weights, check that the repository's Git LFS objects are downloadable and that the model files exist in `models/multilingual-bert-tiny-final/`.

The displayed confidence is a softmax score, not a guarantee of correctness or a calibrated probability. The script-count panel shows visible text cues; it is not a faithful explanation of the model's internal decision. Validate against real voice-call transcripts before production use.

