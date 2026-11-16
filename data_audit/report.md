# Data audit (counts only, no chat text)

- **WARNING**: 12.8% of the long validation replies already appear somewhere in the training text (window mode repeats messages; a little is normal, a lot means validation is partly memorized).
- **WARNING**: 15.7% of the long validation messages also appear in training.

## Training files

| file | examples | trained replies | share of replies | real |
|---|---|---|---|---|
| merged_chats_clean_mood_cleanData.jsonl | 973 | 973 | 21.1% | yes |
| synthetic_chats_100k_cleanData.jsonl | 3638 | 3638 | 78.9% | no |

## De-duplication

- examples: 4611
- exact duplicate examples: 0
- unique examples: 4611
- trained replies: 4611
- distinct replies: 3582
- distinct reply buckets by times used: {'1': 2935, '2-3': 639, '4-10': 8, 'over 10': 0}
- share of replies used more than 3 times: 0.0082
- replies over clean data limit: 0

## Train/val leak check

- exact examples in both: 0
- val long replies: 94
- val long replies seen in train: 12
- val long reply overlap: 0.1277
- val short replies: 188
- val short replies seen in train: 122
- val long messages: 1090
- val long messages seen in train: 171
- val long message overlap: 0.1569
- files with session check: 2
- sessions in both: 0
