"""Download Tagalog (fil_Latn) and Cebuano (ceb_Latn) text from FineWeb-2.

Setup:   pip install datasets
Run:     python grab_data.py
Output:  public_text/tl/tl.jsonl   (about 300 MB of Tagalog)
         public_text/ceb/ceb.jsonl (all the Cebuano)

Run it in a folder OUTSIDE your Persona_LM project so big files do not end up in git.
It streams, so nothing else is downloaded. Ctrl+C is safe: what was written stays.
"""
import json
import os

from datasets import load_dataset

OUT_DIR = "public_text"
TAGALOG_MAX_CHARS = 300_000_000      # about 90M tokens
CEBUANO_MAX_CHARS = 10 ** 12         # effectively all of it


def grab(name, out_path, max_chars):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-2", name=name, split="train", streaming=True)
    total = docs = 0
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                text = row["text"]
                if not text or not text.strip():
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total += len(text)
                docs += 1
                if docs % 10000 == 0:
                    print("  %s: %d docs, %d MB" % (name, docs, total // 1_000_000), flush=True)
                if total >= max_chars:
                    break
    except KeyboardInterrupt:
        print("Stopped early; keeping what was written.")
    print("%s: %d docs, %d MB of text -> %s" % (name, docs, round(total / 1e6), out_path))


if __name__ == "__main__":
    grab("fil_Latn", os.path.join(OUT_DIR, "tl", "tl.jsonl"), TAGALOG_MAX_CHARS)
    grab("ceb_Latn", os.path.join(OUT_DIR, "ceb", "ceb.jsonl"), CEBUANO_MAX_CHARS)
