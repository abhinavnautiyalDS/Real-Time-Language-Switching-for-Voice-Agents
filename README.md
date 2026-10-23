# Real-Time Language Switching for Voice Agents
<img width="2048" height="768" alt="Monochrome Multilingual Voice Classifier" src="https://github.com/user-attachments/assets/3e67e704-973c-4a1b-92e6-37bbcd237188" />


A small language classifier for voice-agent transcripts, with a focus on Indian languages and code-switching.

I built this because identifying a transcript's language is not as simple as checking its script. A user might say *"Eligibility क्या है?"*, write Hindi in Latin characters, or switch between English and another Indian language in the same sentence. A script-based rule can get these cases wrong, so this project treats the task as **utterance-level language classification**.

The goal is to predict a language tag immediately after STT so the LLM gets a signal for the current turn and can respond in the user's language without waiting for a later turn. Inference needs to stay lightweight so it does not add a noticeable pause.



## Why this exists in the voice pipeline

This project started with a language-switching problem in a real-time voice agent.

A typical voice pipeline looks like this:

```text
User speaks
   ↓
STT (speech → text)
   ↓
LLM (understands the turn and generates a reply)
   ↓
TTS (text → speech)
   ↓
Agent speaks
```

The delay is not only the time taken by each component. The LLM also has to work out which language the user is speaking and decide how to respond. When a user switches language mid-conversation, the agent can sometimes keep using the previous language and only switch after another turn or two. That makes the conversation feel slow and unnatural.

The approach here is to add a **small, fast classifier immediately after STT**. It reads the transcript, predicts a language tag, and passes that tag to the LLM alongside the transcript:

```text
User speaks
   ↓
STT → transcript
   ↓
Lightweight language classifier → language tag
   ↓
LLM receives: transcript + language tag
   ↓
TTS speaks the response
```

The tag gives the LLM an explicit, up-to-date signal about the language of the current user turn, instead of making it infer the language from conversation history alone. The aim is to help the agent switch on the **same turn**, including when the user changes language during a conversation.

This classifier is a routing/context signal, not a replacement for STT, the LLM or TTS. It does add a small amount of work after transcription, so inference speed matters. The goal is to keep that overhead low enough that faster language switching does not come at the cost of a noticeable pause.

The classifier can only guide the response if the voice-agent integration actually forwards its predicted tag to the LLM and the LLM prompt/runtime follows it. The model alone does not guarantee instant switching; that needs to be checked in end-to-end voice tests.

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
- **Evaluation:** accuracy, macro-F1, per-class metrics and a confusion matrix, with a separate focus on CPU latency and end-to-end language-switching tests.
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

---

## Branch-specific problem and solution

**Problem being addressed:** v4.4 release narrative: explicit neutral acknowledgements, none labels and the current prepared-test report.

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

