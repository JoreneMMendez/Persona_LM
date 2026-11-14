"""Check how well work/tok.json handles each language. Changes nothing.

Run (project folder):  python check_tok.py            (about 30 seconds)
For each language it takes 300 random documents (a different random pick from the tokenizer sample;
the first 1000 characters of each) and your chat messages from data_msgr/val.jsonl (messages the
tokenizer did not train on), and prints:
  chars/token   : higher = better (more text per slot in the 512-token window)
  tokens/word   : lower  = better (English is usually about 1.3; 2.0+ means words get chopped up)
It also checks that decode(encode(text)) gives the text back exactly, and shows one short
example of how a sentence is split.
"""
import json
import os
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
from persona_lm.bpe import Tokenizer

TOK = os.path.join("work", "tok.json")
SOURCES = [("English", "public_clean/en/en.jsonl"), ("Tagalog", "public_clean/tl/tl.jsonl"),
           ("Cebuano", "public_clean/ceb/ceb.jsonl")]
WORD = re.compile(r"\w+", re.UNICODE)


def stats(tok, texts):
    chars = toks = words = bad = 0
    for t in texts:
        ids = tok.encode(t)
        if tok.decode(ids) != t:
            bad += 1
        chars += len(t)
        toks += len(ids)
        words += len(WORD.findall(t))
    return chars / max(toks, 1), toks / max(words, 1), bad


def pieces(tok, text):
    return " | ".join(tok.decode([i]) for i in tok.encode(text))


def pick_docs(path, n=300, seed=7):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["text"])
    random.Random(seed).shuffle(rows)
    return [t[:1000] for t in rows[:n]]


def chat_messages(path, limit=500):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for m in rec.get("messages", []):
                c = m.get("content")
                if isinstance(c, str) and 3 <= len(c) <= 500:
                    out.append(c)
    random.Random(7).shuffle(out)
    return out[:limit]


if __name__ == "__main__":
    tok = Tokenizer.load(TOK)
    print("tokenizer: %d tokens, fingerprint %s...\n" % (tok.vocab_size, tok.fingerprint()[:12]))
    print("%-22s %12s %12s %10s" % ("text", "chars/token", "tokens/word", "bad round-trips"))
    examples = {}
    for name, path in SOURCES:
        if os.path.exists(path):
            docs = pick_docs(path)
            c, w, bad = stats(tok, docs)
            print("%-22s %12.2f %12.2f %10d" % (name, c, w, bad))
            examples[name] = docs[0].split(".")[0][:90]
    val = os.path.join("data_msgr", "val.jsonl")
    if os.path.exists(val):
        msgs = chat_messages(val)
        c, w, bad = stats(tok, msgs)
        print("%-22s %12.2f %12.2f %10d" % ("Your chat messages", c, w, bad))
        examples["Your chats"] = msgs[0][:90]
    print("\nHow one sentence from each is split:")
    for name, s in examples.items():
        print("  %s: %s" % (name, pieces(tok, s)))
