"""Download Cebuano again, this time KEEPING the url, and list the biggest websites.

Run from Persona_LM-main:   python ceb_domains.py
Writes : public_text/ceb/ceb_url.jsonl   ({"text":..., "url":...} per line; ignored by git)
Prints : the top 40 websites by document count, with size and two short samples each.
Paste the printed list back to Claude to decide which sites to keep.
Ctrl+C stops early and still prints the list for what was downloaded.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join("public_text", "ceb", "ceb_url.jsonl")
TOP = 40


def domain(url):
    d = urlparse(url or "").netloc.lower()
    return d[4:] if d.startswith("www.") else d


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="ceb_Latn", split="train", streaming=True)
    docs = Counter()
    chars = Counter()
    samples = defaultdict(list)
    n = 0
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            for row in ds:
                text, url = row["text"], row.get("url", "")
                if not text or not text.strip():
                    continue
                f.write(json.dumps({"text": text, "url": url}, ensure_ascii=False) + "\n")
                d = domain(url)
                docs[d] += 1
                chars[d] += len(text)
                if len(samples[d]) < 2 and docs[d] in (1, 40):
                    samples[d].append(text[:150].replace("\n", " "))
                n += 1
                if n % 20000 == 0:
                    print("  %d docs" % n, flush=True)
    except KeyboardInterrupt:
        print("Stopped early; using what was downloaded.")

    print("\n%d documents from %d websites -> %s" % (n, len(docs), OUT))
    top = docs.most_common(TOP)
    share = 100 * sum(c for _, c in top) / max(n, 1)
    print("The top %d websites hold %.0f%% of the documents.\n" % (TOP, share))
    for d, c in top:
        print("%-38s %6d docs  %5.1f MB" % (d, c, chars[d] / 1e6))
        for s in samples[d]:
            print("      -", s)


if __name__ == "__main__":
    main()
