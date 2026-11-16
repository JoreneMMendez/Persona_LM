"""Build the pretraining mix: public_clean/ -> public_mix/  (balanced English / Tagalog / Cebuano).

Run (project folder):  python make_mix.py                 (about 330 MB English, 195 MB Tagalog, all Cebuano)
        or:            python make_mix.py 330 195 65      (MB of English, Tagalog, Cebuano; edit to taste)
Reads :  public_clean/en/en.jsonl, public_clean/tl/tl.jsonl, public_clean/ceb/ceb.jsonl
Writes:  public_mix/en.jsonl, public_mix/tl.jsonl, public_mix/ceb.jsonl
Takes a minute or two and uses up to about 1 GB of memory. Nothing in public_clean changes.

How it picks: random documents (same ones every run), written in shuffled order inside each file,
so the "every 50th document is validation" rule of `prepare` samples every kind of text.
Cebuano is used ONCE (no copies): copies would land in validation too and flatter the loss.
The "tokens" printed are rough guesses (characters / 4): nothing here is measured.
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

SOURCES = [("en", "public_clean/en/en.jsonl"), ("tl", "public_clean/tl/tl.jsonl"),
           ("ceb", "public_clean/ceb/ceb.jsonl")]
DEFAULT_MB = {"en": 330, "tl": 195, "ceb": 65}
OUT_DIR = "public_mix"
TRAIN_TOKENS = 10000 * 4 * 8 * 512        # the pipeline's default steps x batch x accum x window


def build(name, path, budget_chars):
    lengths = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lengths.append(len(json.loads(line)["text"]))
    order = list(range(len(lengths)))
    random.Random(1).shuffle(order)
    chosen, total = [], 0
    for i in order:
        if total >= budget_chars:
            break
        chosen.append(i)
        total += lengths[i]
    wanted = set(chosen)
    texts = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                texts[i] = json.loads(line)["text"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name + ".jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i in chosen:                                  # `chosen` is already in shuffled order
            f.write(json.dumps({"text": texts[i]}, ensure_ascii=False) + "\n")
    print("%s: %d of %d documents, %.0f MB of %.0f MB -> %s" % (
        name, len(chosen), len(lengths), total / 1e6, sum(lengths) / 1e6, out))
    print("   example:", texts[chosen[0]][:120].replace("\n", " "))
    return total


if __name__ == "__main__":
    mbs = dict(DEFAULT_MB)
    if len(sys.argv) == 4:
        mbs = {"en": float(sys.argv[1]), "tl": float(sys.argv[2]), "ceb": float(sys.argv[3])}
    elif len(sys.argv) != 1:
        raise SystemExit("give three numbers (MB of English, Tagalog, Cebuano) or none")
    done = {}
    for name, path in SOURCES:
        if not os.path.exists(path):
            raise SystemExit("missing %s: run the downloads first" % path)
        done[name] = build(name, path, int(mbs[name] * 1_000_000))
    total = sum(done.values())
    print("\nShare of the mix (by characters):  " + "   ".join(
        "%s %.0f%%" % (n, 100 * c / total) for n, c in done.items()))
    est = total / 4
    print("Rough size: about %.0fM tokens (a guess). The default pipeline trains about %.0fM tokens, about %.1f passes." % (
        est / 1e6, TRAIN_TOKENS / 1e6, TRAIN_TOKENS / est))
