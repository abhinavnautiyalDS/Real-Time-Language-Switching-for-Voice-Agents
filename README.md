# Version 4.1 — Multilingual Voice-Agent Language Classifier

This branch represents a dataset snapshot in the evolution toward reliable real-time language switching.

## Problem → change → evidence

### Problem
The task needs broader multilingual coverage while preserving reliable labels and a fair held-out evaluation. Larger datasets alone do not guarantee better generalization.

### Snapshot evidence
The supplied Version 4.1 archive contains 700,907 rows across train/validation/test and 69 categories. The archive alone does not provide a complete build report explaining every transformation.

### Improvement hypothesis
This iteration is treated as a coverage/scale pass following Version 4. To make the history reproducible, verify exact additions, relabels and cleaning operations against branch commits or a build report before stating them as confirmed facts.

## What should be checked
- Per-split and per-label row counts.
- Exact/near-duplicate and cross-split leakage.
- Source provenance, licenses and attribution.
- Group-aware split logic and random seeds.
- Confusion matrix and per-class precision/recall/F1 compared with Version 4.
- Short, Romanized, mixed-script and STT-noisy examples.

## Target labels in the later multilingual project
as-IN, bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN and none.

## Data and model references
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA)
- [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC)
- [Tatoeba English–Hindi](https://www.manythings.org/anki/hin-eng.zip)
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID)
- [Google BERT-Tiny base checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- [Main real-time repository](https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents)

## Open-source readiness
Do not publish a score without its exact split and script. Include data leakage checks, independent real STT evaluation, class-wise metrics, calibration, CPU p50/p90/p95, throughput/RAM, robustness tests, licenses, reproducible scripts and a model card.
