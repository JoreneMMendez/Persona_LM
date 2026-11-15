"""Download Taglish / Tagalog forum conversations (PinoyExchange, "gabtan99/pex-conversations") as pretraining text.

Setup:   pip install datasets
Run:     python grab_taglish.py            -> public_text/taglish/pex.jsonl   ({"text": one whole thread per line})
         python grab_taglish.py 100        -> stop after about 100 MB of text
Run it in a folder OUTSIDE your Persona_LM project so big files do not end up in git. Ctrl+C is safe.

NOTE: the dataset's license is listed as "unknown" and the posts come from real forum users. Use it for your own private
training only; do not re-upload the data, and do not publish a model trained on it without checking the site's terms.

Next:    python clean_corpus.py   (or add public_text/taglish/pex.jsonl to your `prepare --kind pretrain --data ...` list)
"""
import json
import os
import sys

OUT_PATH = os.path.join("public_text", "taglish", "pex.jsonl")
MIN_CHARS = 200


def flatten(node, out, depth=0):
    """A thread is {"text": ..., "replies": [same shape, ...]}. Returns the messages in reading order, one per line."""
    if isinstance(node, dict):
        text = (node.get("text") or "").strip()
        if text:
            out.append(text)
        for r in node.get("replies") or []:
            flatten(r, out, depth + 1) if depth < 200 else None
    elif isinstance(node, list):
        for n in node:
            flatten(n, out, depth)
    return out


def main(max_mb):
    from datasets import load_dataset
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ds = load_dataset("gabtan99/pex-conversations", split="train", streaming=True)
    total = threads = 0
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for row in ds:
                for thread in row["conversations"]:
                    text = "\n".join(flatten(thread, []))
                    if len(text) < MIN_CHARS:
                        continue
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    total += len(text)
                    threads += 1
                    if threads % 2000 == 0:
                        print("  %d threads, %d MB" % (threads, total // 1_000_000), flush=True)
                    if max_mb and total >= max_mb * 1_000_000:
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Stopped; keeping what was written.")
    print("%d threads, %d MB -> %s" % (threads, total // 1_000_000, OUT_PATH))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
