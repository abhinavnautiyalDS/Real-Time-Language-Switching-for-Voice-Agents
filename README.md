# Version 4 — Multilingual Voice-Agent Language Classifier

This branch captures the Version 4 dataset/model stage of the real-time language-switching project.

## Problem → change → why it mattered

### Problem
A multilingual classifier can learn incorrect labels or repeated templates instead of language evidence. The Version 4 dataset notes identify synthetic Punjabi examples that were mostly Hindi-looking Devanagari text but labeled Punjabi.

### Change
- Removed 9,992 of 10,000 flagged synthetic Punjabi rows.
- Trimmed repetitive/templated examples.
- Added STT-style noisy variants.
- Rebuilt group-aware train/validation/test splits to reduce related-example leakage.

### Intended outcome
Improve label integrity and reduce shortcut learning. This data cleanup is not proof by itself that a later model score improved; use controlled comparisons and class-wise error analysis.

## Project task
The real-time project ultimately targets 12 labels: as-IN, bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN and none.

## Data sources
- [COMI-LINGUA](https://github.com/lingo-iitgn/COMI-LINGUA)
- [PHINC](https://huggingface.co/datasets/LingoIITGN/PHINC)
- [Tatoeba English–Hindi](https://www.manythings.org/anki/hin-eng.zip)
- [AI4Bharat IndicLID](https://github.com/AI4Bharat/IndicLID)

## Related model and project
- [Google BERT-Tiny base checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- [Real-Time Language Switching for Voice Agents](https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents)

## Evidence to retain
Commit the exact dataset archive, source/license attribution, build report, split manifest, model config, confusion matrix and evaluation metrics. For open-source release, also report leakage checks, robustness, CPU p50/p90/p95, memory, dependency licenses and known limitations.
