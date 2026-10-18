# Fine-tuned BERT-Tiny models

## E2

- Checkpoint family: `google/bert_uncased_L-2_H-128_A-2`
- Transformer layers: 2
- Tokenizer vocabulary: 30,522
- Test accuracy reported: 99.9745%
- Test macro-F1 reported: 99.9756%

## E4

- Checkpoint family: `google/bert_uncased_L-4_H-128_A-2`
- Transformer layers: 4
- Tokenizer vocabulary: 30,522
- Test accuracy reported: 99.9682%
- Test macro-F1 reported: 99.9689%
- Model directory size reported: 20.02 MB

These are results on the prepared test split, not a guarantee of real-call performance. Before production selection, benchmark both models on the same untouched real-call set and measure Windows CPU p50/p90 latency. The custom tokenizer transferred 5,083 of 30,522 embedding rows (16.65%) from the original English checkpoint; other rows were randomly initialized before fine-tuning.
