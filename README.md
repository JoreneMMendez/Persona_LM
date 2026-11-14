# persona_lm: a language model from scratch (pure Python + NumPy, optional PyTorch for GPUs)

The model, tokenizer and training code are written from scratch with NumPy (Python 3.10+, `pip install numpy`). PyTorch is **optional**: install it only to train on a GPU (see `docs/GPU_GUIDE.md`). Without it everything still works on the CPU.

**Project handoff / status for another AI or a new session: read `HANDOFF.md` (do not delete it).**
Headless: command line only. Everything is written here: tokenizer, transformer with hand-written backprop, AdamW,
training loop with exact resume, chat tuning, sampling, and a GGUF exporter so the finished model can run on the phone.

## Workflow

```bash
# 0. How fast is this computer? (tokens/s and rough days for 1e8 / 1e9 / 3e9 training tokens)
python -m persona_lm benchmark --size s10m --vocab 8000

# 1. Learn a vocabulary from your text (.txt .md, or .json/.jsonl). Use the same kind of text you will train on.
python -m persona_lm tokenizer --data corpus/ --vocab 8000 --out tok.json

# 2. Pretraining data: plain text -> token files
python -m persona_lm prepare --kind pretrain --tokenizer tok.json --data corpus/ --out data_pre

# 3. Pretrain. Ctrl+C is safe. Run the SAME command again to continue exactly where it stopped.
python -m persona_lm train --run runs/pre --data-dir data_pre --size s10m --steps 50000

# 4. Chat data (JSON) -> token files, then chat-tune starting from the pretrained weights
python -m persona_lm prepare --kind chat --tokenizer tok.json --data my_chats.json --out data_chat
python -m persona_lm train --run runs/chat --data-dir data_chat --size s10m \
       --init-from runs/pre/ckpt.npz --steps 3000 --lr 0.0001 --min-lr 0.00001 --warmup 50

# 5. Talk to it
python -m persona_lm chat --ckpt runs/chat/ckpt.npz --tokenizer tok.json --system "You are Aria."
python -m persona_lm generate --ckpt runs/pre/ckpt.npz --tokenizer tok.json --prompt "Once upon a time"

# 6. Export for the phone (GGUF). f16 is the safe default; q8_0 is half the size.
python -m persona_lm export-gguf --ckpt runs/chat/ckpt.npz --tokenizer tok.json --out aria.gguf --type f16 --verify
```

Sizes: `tiny` (0.1M, for testing), `s10m` and `s30m` (context 512), `s125m` (parameter counts exclude the vocabulary; the benchmark prints the total).
Training must use the same `--size` for pretraining and chat tuning.

## Your Messenger data (Stage 2; see docs/MESSENGER_GUIDE.md)
```bash
python -m persona_lm messenger-scan EXPORT_FOLDER --me "Your Name"                          # list the chats (read-only)
python -m persona_lm messenger-import EXPORT_FOLDER --out data_msgr --me "Your Name" --chats 1,2,3
python -m persona_lm prepare --kind chat --tokenizer tok.json --data data_msgr/train.jsonl --val-data data_msgr/val.jsonl --out data_chat
```
Local only. Anonymized by default (names, phones, e-mails, links, addresses) with a self-audit; loss only on your replies; validation split by session. Tested on synthetic data only.

## Guided pipeline (Stage 3; see docs/STAGE3_GUIDE.md)
```bash
python -m persona_lm pipeline init --dir work --languages en,tl --vocab 16000 --pretrain-text public_text/ --chat-train data_msgr/train.jsonl --chat-val data_msgr/val.jsonl
python -m persona_lm pipeline plan --dir work            # every step with its exact command, warnings
python -m persona_lm pipeline confirm --dir work --yes   # the tokenizer cannot be made before this, and is never overwritten afterwards
python -m persona_lm pipeline run --dir work             # runs the next undone step (repeat)
python -m persona_lm data-report --data-dir work/data_chat --size s10m --tokenizer work/tok.json
python -m persona_lm status work
```
The chat screen has slash commands (`/help /reset /history /save /system /temp /max /quit`; `/memory /mood /identity /approve /reject` are placeholders for later stages).

## Long-term memory (Stage 4; see docs/MEMORY_GUIDE.md)
```bash
python -m persona_lm memory add "[[person:Maria]] is my sister #love *0.9 !pin"
python -m persona_lm memory search what does Maria like
python -m persona_lm memory maintain            # a plan; add --apply to do it
python -m persona_lm memory export-vault vault  # Obsidian folder with [[wiki-links]]
python -m persona_lm memory export-graph memory_graph.html
python -m persona_lm chat --ckpt runs/chat/ckpt.npz --tokenizer tok.json --memory memory.json
```
Memories live in `memory.json` outside the model. Only typed commands change them. Recall is by words and names, not meaning. Tested on synthetic data only.

## Emotion, mood and Person card (Stage 5; see docs/EMOTION_GUIDE.md)
```bash
python -m persona_lm emotion tag "pagod na ako sooo"
python -m persona_lm emotion label --data data_msgr/train.jsonl --out emotion_samples.jsonl
python -m persona_lm emotion review --samples emotion_samples.jsonl      # correct guesses by typing; saved at once
python -m persona_lm emotion train --samples emotion_samples.jsonl --out emotion_model.json
python -m persona_lm person init --name "Your name"
python -m persona_lm person build --data data_msgr/train.jsonl --memory memory.json      # a plan; add --apply
python -m persona_lm chat --ckpt runs/chat/ckpt.npz --tokenizer tok.json --mood mood.json --emotion-model emotion_model.json --person person_card.md
```
Only what you type moves the mood. Tested on synthetic data only; the Tagalog/Bisaya word lists are unchecked.

## GPU training (Stage 1; see docs/GPU_GUIDE.md)
```bash
python -m persona_lm check-gpu                       # does the GPU code match the NumPy reference on this machine?
python -m persona_lm benchmark --size s10m --vocab 8000 --try-batches 1,2,4,8,16    # tokens/s, memory, largest micro-batch that fits
python -m persona_lm train ... --backend auto --device auto --precision auto --batch 4 --accum 8
```
`--backend auto` uses PyTorch if installed (GPU if present, else CPU) and falls back to NumPy. `--precision auto` picks bf16 on NVIDIA Ampere or newer, fp16 on older GPUs, fp32 on CPU.
`--batch` is the micro-batch that must fit in memory; `--batch x --accum` is the effective batch. Both backends read and write the same `ckpt.npz`.

## Data formats
- Pretraining: `.txt`, `.md` (any folder of them), `.jsonl`/`.json` with strings or `{"text": "..."}`.
- Chat: `{"messages": [{"role": "system|user|assistant", "content": "..."}]}`, or `{"prompt": ..., "reply": ...}`,
  or `{"instruction": ..., "output": ...}` (one per line in `.jsonl`, or a list in `.json`).
  The loss is counted only on the assistant's words.

## What a run folder contains
`ckpt.npz` (latest, written atomically; the previous one is kept as `ckpt.prev.npz`), `metrics.jsonl` (loss, validation loss, speed).
A crash or power cut loses at most `--ckpt-every` steps (default 100) or `--ckpt-minutes` (default 10).

## Tests
`python -m unittest discover -s tests -v`  (262 tests, a few seconds; 2 of them need PyTorch and are skipped without it). They check, among other things:
the gradients against finite differences (and that the check catches deliberately injected bugs), that loss falls,
that stop-and-resume gives bit-identical weights, tokenizer round trips, and that the GGUF file reads back.

## Honest limits
- **The NumPy backend is CPU only and slow.** The PyTorch GPU backend (Stage 1) is written but was NOT run by its author (no GPU/PyTorch available): run `check-gpu` first. Run `benchmark` on your laptop and trust that number, not my guess.
  Measured here (2 CPU cores): about 980 tokens/s for the 13.7M-parameter `s10m` model.
  A 125M model is roughly 10x slower per token and needs billions of tokens: months on a CPU.
- **Quality is set by data, not by time.** A small model needs a lot of clean text, and your own JSON alone is far too little for a general chatbot.
  Expect a simple conversational style, frequent mistakes and invented facts.
- **GGUF is not tested inside llama.cpp** (it is not available here). The writer follows the format and the "llama" tensor layout, the file reads back
  correctly with my own reader, and RoPE uses the adjacent-pair layout llama.cpp expects. The first thing to do on your side:
  load `aria.gguf` with `llama-cli`, and compare `llama-tokenize` output against `Tokenizer.encode` on a sentence.
  If either differs, tell me the exact output and I will fix the writer. Q4_K_M: run llama.cpp's `llama-quantize` on the f16 file.
- Text generation recomputes the window each token (no KV cache). Fine for testing; slow for long replies.
- Not in this package yet: identity core, self-modification, personas, export package. See `HANDOFF.md` for the plan and status.
