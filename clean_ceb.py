"""Clean Cebuano by WEBSITE (most web Cebuano is machine-translated).

Run from Persona_LM-main after ceb_domains.py:   python clean_ceb.py
Reads : public_text/ceb/ceb_url.jsonl   ({"text", "url"})
Writes: public_clean/ceb/ceb.jsonl      (overwrites the earlier, worse version)

Keeps a document only if its website is on the keep list or ends in .ph (Philippine sites),
never a ceb.* subdomain. Also drops short documents, repeated boilerplate openings and
bot / scrape pages, and normalizes fancy Unicode letters (NFKC).
"""
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join("public_text", "ceb", "ceb_url.jsonl")
DST = os.path.join("public_clean", "ceb", "ceb.jsonl")
MIN_CHARS = 300
MAX_SAME_OPENING = 3
KEEP_EXTRA = {"rpnradio.com", "cdeocitycouncil.com", "bomboradyo.com", "pressreader.com",
              "newsaffaircenter.blogspot.com", "jw.org"}
BAD = re.compile(r"Lsjbot|Paghimo ni bot|Jump to navigation|Jump to search|Numero sa aytem|\bMOQ\b|\bFOB\b", re.I)


def domain(url):
    d = urlparse(url or "").netloc.lower()
    return d[4:] if d.startswith("www.") else d


def site_ok(d):
    if d.startswith("ceb."):
        return False
    return d in KEEP_EXTRA or d.endswith(".ph")


def read():
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield r["text"], domain(r.get("url", ""))


def main():
    openings = Counter()
    for text, d in read():
        openings[text[:80]] += 1
    drop = Counter()
    kept = []
    kept_sites = Counter()
    kept_chars = Counter()
    for text, d in read():
        if not site_ok(d):
            drop["website not on keep list"] += 1
            continue
        text = unicodedata.normalize("NFKC", text)
        if len(text) < MIN_CHARS:
            drop["too short"] += 1
        elif openings[text[:80]] > MAX_SAME_OPENING:
            drop["repeated opening"] += 1
        elif BAD.search(text):
            drop["bot / scrape / listing"] += 1
        else:
            kept.append((text, d))
            kept_sites[d] += 1
            kept_chars[d] += len(text)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for text, d in kept:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    total = sum(kept_chars.values())
    print("Kept %d documents, %.0f MB, from %d websites -> %s" % (len(kept), total / 1e6, len(kept_sites), DST))
    print("Dropped:", dict(drop))
    print("\nBiggest kept websites:")
    for d, c in kept_sites.most_common(15):
        print("  %-36s %6d docs  %5.1f MB" % (d, c, kept_chars[d] / 1e6))
    random.seed(1)
    print("\n8 random kept samples (website: text):")
    for text, d in random.sample(kept, min(8, len(kept))):
        print("  [%s] %s" % (d, text[:220].replace("\n", " ")))


if __name__ == "__main__":
    main()
