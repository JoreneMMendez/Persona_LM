"""Make a BALANCED text sample for learning the tokenizer (same amount of each language).

Why: the tokenizer step reads text file by file in alphabetical order and only checks its size
limit after finishing a whole file. Pointed at your public_clean folder it would read ALL of the
Tagalog file (295 MB) and nothing else, so Cebuano and English would get no vocabulary.

Run (from the project folder):  python make_tok_sample.py
         python make_tok_sample.py 20      (20 MB per language instead of 25)
Reads :  public_clean/en/en.jsonl, public_clean/tl/tl.jsonl, public_clean/ceb/ceb.jsonl
Writes:  tok_sample/en.jsonl, tok_sample/tl.jsonl, tok_sample/ceb.jsonl
It picks random documents (the same ones every run). It changes nothing in public_clean
and does not touch the tokenizer. Takes about a minute.
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

SOURCES = [("en", "public_clean/en/en.jsonl"), ("tl", "public_clean/tl/tl.jsonl"),
           ("ceb", "public_clean/ceb/ceb.jsonl")]
OUT_DIR = "tok_sample"


def sample(name, path, budget):
    if not os.path.exists(path):
        print("%s: MISSING %s (skipped)" % (name, path))
        return
    lengths = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lengths.append(len(json.loads(line)["text"]))
    order = list(range(len(lengths)))
    random.Random(1).shuffle(order)
    chosen, total = set(), 0
    for i in order:
        if total >= budget:
            break
        chosen.add(i)
        total += lengths[i]
    kept = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in chosen:
                kept.append(json.loads(line)["text"])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name + ".jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print("%s: %d of %d documents, %.1f MB (of %.0f MB) -> %s" % (
        name, len(kept), len(lengths), total / 1e6, sum(lengths) / 1e6, out))
    print("   example:", kept[0][:150].replace("\n", " "))


if __name__ == "__main__":
    mb = float(sys.argv[1]) if len(sys.argv) > 1 else 25
    for name, path in SOURCES:
        sample(name, path, int(mb * 1_000_000))
