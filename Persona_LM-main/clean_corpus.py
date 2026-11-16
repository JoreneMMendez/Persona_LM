"""Clean the downloaded text and pick the best Cebuano.

Run from Persona_LM-main after grab_data.py:   python clean_corpus.py
Reads : public_text/tl/tl.jsonl, public_text/ceb/ceb.jsonl
Writes: public_clean/tl/tl.jsonl, public_clean/ceb/ceb.jsonl

Both languages: drop documents under MIN_CHARS and documents whose first 80 characters
appear more than MAX_SAME_OPENING times (website boilerplate).
Cebuano also: drop Lsjbot / Wikipedia-scrape pages and obvious factory-product listings,
then rank the rest by how much everyday spoken-Bisaya vocabulary they contain (machine
translation uses it less) and keep the best until CEB_TARGET_CHARS.
These rules are crude: read the printed samples before trusting the result.
"""
import json
import os
import random
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

MIN_CHARS = 300
MAX_SAME_OPENING = 3
CEB_TARGET_CHARS = 150_000_000

CEB_BAD = re.compile(
    r"Lsjbot|Paghimo ni bot|Jump to navigation|Jump to search|Numero sa aytem|Item No\b|\bMOQ\b|\bFOB\b|\bOEM\b|\bODM\b",
    re.I)
# everyday spoken Cebuano / Bisaya words
CEB_SPOKEN = {
    "nako", "nimo", "imo", "kaayo", "gyud", "jud", "gyod", "pud", "pod", "unsa", "dili", "naa", "wala",
    "lang", "bitaw", "unya", "diay", "gani", "ayaw", "ko", "ta", "mi", "nato", "ra", "kay", "mao",
    "murag", "kanus-a", "asa", "ngano", "nimong", "akong", "imong", "namo", "nila", "ka", "pa",
}
WORD = re.compile(r"[A-Za-z']+")


def read_docs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def opening_counts(path):
    c = Counter()
    for t in read_docs(path):
        c[t[:80]] += 1
    return c


def spoken_score(text):
    words = WORD.findall(text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in CEB_SPOKEN)
    return 100.0 * hits / len(words)


def write_lines(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chars = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            chars += len(t)
    return chars


def clean_tl(src, dst):
    openings = opening_counts(src)
    drop = Counter()
    keep = []
    for t in read_docs(src):
        if len(t) < MIN_CHARS:
            drop["too short"] += 1
        elif openings[t[:80]] > MAX_SAME_OPENING:
            drop["repeated opening"] += 1
        else:
            keep.append(t)
    chars = write_lines(dst, keep)
    print("\nTagalog: kept %d docs, %.0f MB. Dropped: %s" % (len(keep), chars / 1e6, dict(drop)))


def clean_ceb(src, dst):
    openings = opening_counts(src)
    drop = Counter()
    scored = []                     # (score, line number, length)
    for i, t in enumerate(read_docs(src)):
        if len(t) < MIN_CHARS:
            drop["too short"] += 1
        elif openings[t[:80]] > MAX_SAME_OPENING:
            drop["repeated opening"] += 1
        elif CEB_BAD.search(t):
            drop["bot / scrape / product listing"] += 1
        else:
            scored.append((spoken_score(t), i, len(t)))
    scored.sort(reverse=True)
    chosen, total = set(), 0
    for score, i, n in scored:
        if total + n > CEB_TARGET_CHARS:
            break
        chosen.add(i)
        total += n
    cutoff = min((s for s, i, n in scored if i in chosen), default=0.0)
    kept = [t for i, t in enumerate(read_docs(src)) if i in chosen]
    chars = write_lines(dst, kept)
    print("\nCebuano: %d docs passed the filters, kept the best %d (%.0f MB). Dropped: %s" % (
        len(scored), len(kept), chars / 1e6, dict(drop)))
    print("lowest spoken-word score kept: %.1f per 100 words" % cutoff)
    random.seed(1)
    print("5 random kept Cebuano samples:")
    for t in random.sample(kept, min(5, len(kept))):
        print("   -", t[:300].replace("\n", " "))


if __name__ == "__main__":
    clean_tl("public_text/tl/tl.jsonl", "public_clean/tl/tl.jsonl")
    clean_ceb("public_text/ceb/ceb.jsonl", "public_clean/ceb/ceb.jsonl")
