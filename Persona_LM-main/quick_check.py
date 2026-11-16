"""Quick quality check of the downloaded text.

Run from Persona_LM-main AFTER grab_data.py has finished:
    python quick_check.py
Prints, per file: documents, size, length stats, share that looks like Bible text,
share of very short documents, share of repeated openings, and 5 random samples.
"""
import json
import random
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

FILES = ["public_text/tl/tl.jsonl", "public_text/ceb/ceb.jsonl"]
BIBLE = re.compile(r"\b\d{1,3}:\d{1,3}\b|\b(Jesus|Jesukristo|Ginoo|Diyos|Dios|Panginoon|Kristo|Hesus|Bibliya|Bible)\b", re.I)


def check(path, n_samples=5, seed=1):
    random.seed(seed)
    docs = lengths = bible = short = chars = 0
    openings = Counter()
    reservoir = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            docs += 1
            n = len(text)
            chars += n
            lengths += 1
            if n < 300:
                short += 1
            if len(BIBLE.findall(text)) >= 3:
                bible += 1
            openings[text[:80]] += 1
            if len(reservoir) < n_samples:
                reservoir.append(text)
            else:
                j = random.randint(0, docs - 1)
                if j < n_samples:
                    reservoir[j] = text
    repeated = sum(c for c in openings.values() if c > 1)
    print("=" * 70)
    print(path)
    print("documents: %d   size: %.0f MB   average length: %d chars" % (docs, chars / 1e6, chars // max(docs, 1)))
    print("looks like Bible/religious (3+ markers): %.1f%%" % (100 * bible / max(docs, 1)))
    print("very short (<300 chars): %.1f%%" % (100 * short / max(docs, 1)))
    print("documents sharing an identical first 80 chars: %.1f%%" % (100 * repeated / max(docs, 1)))
    print("most common openings:")
    for text, c in openings.most_common(3):
        print("   %5d x  %r" % (c, text[:60]))
    print("random samples:")
    for t in reservoir:
        print("   -", t[:300].replace("\n", " "))


for p in FILES:
    try:
        check(p)
    except FileNotFoundError:
        print("missing:", p)
