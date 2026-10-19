# Multilingual Voice Language Classifier

A small language classifier for voice-agent transcripts, with a focus on Indian languages and code-switching.

I built this because identifying a transcript's language is not as simple as checking its script. A user might say *"Eligibility क्या है?"*, write Hindi in Latin characters, or switch between English and another Indian language in the same sentence. A script-based rule can get these cases wrong, so this project treats the task as **utterance-level language classification**.

The goal is to predict a language tag that a voice pipeline can use downstream, while keeping inference light enough to test on a CPU.

## What it predicts

The classifier has 12 output classes:

| Tag | What it covers |
|---|---|
| `en-IN` | English |
| `hi-IN` | Hindi, including supported Hindi-English Hinglish |
| `pa-IN` | Punjabi, including supported Punjabi-Hindi/English combinations |
| `bn-IN` | Bengali, including Bengali-English |
| `gu-IN` | Gujarati, including Gujarati-English |
| `kn-IN` | Kannada, including Kannada-English |
| `ml-IN` | Malayalam, including Malayalam-English |
| `mr-IN` | Marathi, including supported Marathi-English combinations |
| `te-IN` | Telugu, including Telugu-English |
| `as-IN` | Assamese, including Assamese-English |
| `ta-IN` | Tamil, including Tamil-English |
| `none` | Languages or combinations outside the project's supported categories |

The label follows the language policy, not just the characters used. For example, Romanized Hindi should be classified as `hi-IN`, not automatically as English. `none` means unsupported under this project's label policy; it does not mean the model is uncertain.

## What I've built

- **Dataset preparation:** validation for missing text, unknown categories, duplicate utterances and conflicting labels.
- **Leakage-aware splits:** group-aware train, validation and test splits, with official source test groups kept in the test set when available.
- **Custom tokenizer:** a case-preserving WordPiece tokenizer trained on the training text to improve coverage of the scripts in the dataset.
- **Two model sizes:** fine-tuned BERT-Tiny E2 (2 layers) and E4 (4 layers), compared with the existing E2 model.
- **Evaluation:** accuracy, macro-F1, per-class metrics and a confusion matrix, with a separate focus on CPU latency and real-call testing.
- **Streamlit test app:** choose a model, enter a sentence, and inspect its predicted tag, top-class score and scores for all 12 classes.

## Try the Streamlit app locally

### 1. Clone the repository

```bash
git clone --branch finetune/e4-native-script-confusion https://github.com/abhinavnautiyalDS/multilingual-voice-language-classifier.git
cd multilingual-voice-language-classifier
```

### 2. Create an environment and install dependencies

Python 3.10–3.12 is a reasonable starting point for this stack.

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the app dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

### 3. Make sure the model files are available

The app expects the existing model under `models/multilingual-bert-tiny-final/`. The New E2 and New E4 options expect these archives:

- `artifacts/models/bert_tiny_e2_multilingual.zip`
- `artifacts/models/bert_tiny_e4_multilingual.zip`

The archives must be real ZIP files, not Git LFS pointer files. If the repository uses Git LFS, install Git LFS and pull the tracked files:

```bash
git lfs install
git lfs pull
```

If a model archive is not present or has not been uploaded, that model option will not load.

### 4. Launch the app

```bash
streamlit run app.py
```

Try the default example, `Eligibility क्या है?`, and then compare it with English-only, native-script and code-switched sentences. The selected model's output and the all-class scores are shown in the app.

## Model comparison

The repository contains three options in the app:

| App option | Architecture | Notes |
|---|---|---|
| Old E2 | 2-layer BERT-Tiny | Existing model used by the app |
| New E2 | 2-layer BERT-Tiny | Fine-tuned with the new tokenizer |
| New E4 | 4-layer BERT-Tiny | Larger model trained with the same tokenizer approach |

The saved model notes report the following results on the prepared test split:

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| New E2 | 99.9745% | 99.9756% |
| New E4 | 99.9682% | 99.9689% |

These are the reported prepared-split results, not a claim of equivalent accuracy on live calls. E2 is slightly ahead on these numbers, but the model choice should be based on the same untouched real-call evaluation set and CPU latency measurements. Transcription errors, short utterances and natural code-switching can make real traffic quite different from a prepared dataset.

The custom vocabulary also required adapting the original English BERT-Tiny embeddings. The saved artifact notes report that 5,083 of 30,522 embedding rows were transferred from the original checkpoint; the remaining rows were initialized before fine-tuning. This is one reason to evaluate beyond a single aggregate score.

## Training and evaluation

The notebooks are the main record of the experiments:

- [Dataset preparation notebook](notebooks/01_prepare_dataset.ipynb) — validation, category mapping, auditing and data splits.
- [E2/E4 fine-tuning notebook](notebooks/Fine_tuning_BERT_TINY_E2_and_E4.ipynb) — tokenizer training, fine-tuning and model comparisons.
- [Baseline fine-tuning notebook](notebooks/02_finetune_bert_tiny.ipynb) — earlier Hugging Face Trainer workflow.

The reusable Python pipeline is also available:

- `src/data.py` validates and deduplicates the input data.
- `scripts/prepare_splits.py` creates group-aware splits.
- `src/train.py` fine-tunes a sequence classifier.
- `scripts/evaluate_model.py` writes classification reports, confusion matrices and test predictions.
- `src/inference.py` runs a saved model on one input.

For the final evaluation, I want to look beyond accuracy: per-language precision/recall/F1, the confusion matrix, model size, and CPU p50/p90 latency all matter. Real call transcripts should remain separate from training and model selection.

## Dataset notes

The preparation config defines category targets totalling 440,000 rows after the latest reductions. That is a target, not a promise that every category has that many usable examples. The notebook reports the actual counts; it should not create rows just to hit a target.

The split pipeline uses source/group identifiers and synthetic template identifiers where available, so related examples can stay together. This reduces leakage risk, but it does not by itself prove that the dataset is free of near-duplicates or synthetic-pattern leakage. Those checks are still important when interpreting very high test scores.

## Repository layout

```text
.
├── app.py
├── configs/
│   ├── category_to_label.json
│   ├── dataset_config.json
│   └── label_map.json
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   └── DATASET_POLICY.md
├── models/
├── artifacts/
│   └── models/
├── notebooks/
├── scripts/
└── src/
    ├── data.py
    ├── inference.py
    └── train.py
```

Some datasets and model weights are stored as large artifacts rather than ordinary text files. Check that those files are available locally before running the relevant notebook or app.

## A few things to keep in mind

- Softmax confidence is a model score, not a guarantee that the prediction is correct or a calibrated probability.
- `none` represents the project's unsupported categories; it is not a general low-confidence fallback.
- Script counts shown in the app are observable clues, not an explanation of the model's internal decision.
- Check dataset licenses before redistributing source text. Keep private customer transcripts, credentials and other sensitive data out of this public repository.
- Before choosing a production model, evaluate E2 and E4 on the same held-out real-call set and measure latency on the target CPU.

## Tech stack

Python · PyTorch · Hugging Face Transformers · Tokenizers · scikit-learn · pandas · Streamlit
