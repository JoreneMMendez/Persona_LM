"""persona_lm: a small language model built from scratch (pure Python + NumPy, no ML libraries).

Pieces: pretok (GPT-2 style word splitter), bpe (byte-level BPE tokenizer), model (Llama-style transformer with hand-written
backward pass), optim (AdamW), data (corpus -> token files), train (training loop with exact resume), generate, gguf (exporter).
"""
__version__ = "0.1.0"
