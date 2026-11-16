"""Download English pretraining text from FineWeb-Edu (sample-10BT, license ODC-By).

Run:     python grab_english.py            -> about 420 MB (full English corpus)
         python grab_english.py 60         -> about 60 MB (small sample, enough for the tokenizer)
         python grab_english.py 60 --tiny  -> also add up to 40 MB of TinyStories first
Output:  public_clean/en/en.jsonl   (overwritten each run)
Ctrl+C is safe: what was written stays. If the network fails, what was written stays too.

FineWeb-Edu is already cleaned and deduplicated. Documents under 300 characters are dropped.
TinyStories is OFF by default (its download server was refusing connections on your PC).
"""
import json
import os
import sys

from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

OUT_PATH = os.path.join("public_clean", "en", "en.jsonl")
TINY_MAX_CHARS = 40_000_000
MIN_CHARS = 300


def stream_into(f, repo, config, max_chars, label, already=0):
    kwargs = {"split": "train", "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(repo, **kwargs)
    docs = chars = 0
    try:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            f.flush()
            docs += 1
            chars += len(text)
            if docs % 5000 == 0:
                print("  %s: %d docs, %d MB" % (label, docs, (already + chars) // 1_000_000), flush=True)
            if already + chars >= max_chars:
                break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("  %s stopped by a network error (%s); keeping %d docs." % (label, type(e).__name__, docs))
    return docs, chars


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    total_mb = int(args[0]) if args else 420
    use_tiny = "--tiny" in sys.argv
    max_total = total_mb * 1_000_000
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    total = 0
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            if use_tiny:
                d, c = stream_into(f, "roneneldan/TinyStories", None, min(TINY_MAX_CHARS, max_total), "TinyStories")
                total += c
                print("TinyStories: %d stories, %d MB" % (d, round(c / 1e6)))
            d, c = stream_into(f, "HuggingFaceFW/fineweb-edu", "sample-10BT", max_total, "FineWeb-Edu", already=total)
            total += c
            print("FineWeb-Edu: %d documents, %d MB" % (d, round(c / 1e6)))
    except KeyboardInterrupt:
        print("Stopped early; keeping what was written.")
    print("English total: about %d MB of text -> %s" % (round(os.path.getsize(OUT_PATH) / 1e6), OUT_PATH))
