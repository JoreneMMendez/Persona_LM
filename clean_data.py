"""clean_data.py - clean chat .json / .jsonl files, ONE OUTPUT FILE PER INPUT FILE (nothing is merged).

Every cleaned file gets "cleanData" added to its name and is put in the training folder:

    python clean_data.py --input merged_chats_clean_mood.jsonl synthetic_chats_100k.jsonl

    data_clean\\train\\merged_chats_clean_mood_cleanData.jsonl
    data_clean\\train\\synthetic_chats_100k_cleanData.jsonl
    data_clean\\val\\merged_chats_clean_mood_cleanData.jsonl      (the last 5 % of each chat's sessions)
    data_clean\\val\\synthetic_chats_100k_cleanData.jsonl
    data_clean\\report.json                                        (counts only, no chat text)

The AI uses ALL of them: `prepare` reads whole folders, so every *_cleanData file in train\\ is trained on and every
one in val\\ is used for validation. Add a new file later, run this script again, run prepare again.

--input accepts files (.jsonl, or .json holding a list) and folders (every .json/.jsonl inside). Files whose name
already contains "cleanData" are skipped, so you can point it at a folder that holds earlier output.

What it cleans (the same rules as clean_chat.py, which this script imports):
  1. unreadable lines, records without a usable "messages" list, unknown roles, empty messages
  2. system notices that leaked in ("joined the call", "missed a call", ...)
  3. targets spoken by OTHER people in group chats ("Person_65: ...")
  4. exact duplicate examples
  5. repeated identical replies, counted over ALL files together (short <= 25 chars: --max-dup, longer: --max-dup-long),
     so no stock line can be memorized. Files are processed in the order you list them: list your own chats FIRST so
     your own replies are the ones that are kept.
Chat text is never printed unless you pass --show-top.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import clean_chat as CC

TAG = "cleanData"
ROLES = {"system", "user", "assistant"}


def find_files(inputs):
    """Expands folders, keeps the order given, skips files that are already cleanData output."""
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, names in sorted(os.walk(p)):
                for n in sorted(names):
                    if os.path.splitext(n)[1].lower() in (".json", ".jsonl"):
                        out.append(os.path.join(root, n))
        elif os.path.exists(p):
            out.append(p)
        else:
            sys.exit("not found: " + p)
    keep = []
    for p in out:
        if TAG.lower() in os.path.basename(p).lower():
            print("skipped (already %s): %s" % (TAG, p))
        elif p not in keep:
            keep.append(p)
    return keep


def read_records(path, counts):
    """Yields dict records. A bad line is counted and skipped, never fatal."""
    with open(path, encoding="utf-8-sig") as f:
        if path.lower().endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    counts["unreadable_lines"] += 1
                    continue
                yield rec
        else:
            try:
                data = json.load(f)
            except ValueError:
                counts["unreadable_lines"] += 1
                return
            for rec in (data if isinstance(data, list) else [data]):
                yield rec


def normalise(rec, counts):
    """Returns a tidy example or None. Keeps only role / content / train."""
    if not isinstance(rec, dict) or not isinstance(rec.get("messages"), list):
        counts["bad_record"] += 1
        return None
    msgs = []
    for m in rec["messages"]:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            counts["bad_message"] += 1
            continue
        text = m["content"].strip()
        if not text:
            counts["empty_message"] += 1
            continue
        msgs.append({"role": m["role"], "content": text, "train": bool(m.get("train", False))})
    if not msgs:
        counts["bad_record"] += 1
        return None
    ex = {"chat": str(rec.get("chat", "unknown")), "session": rec.get("session", 0), "messages": msgs}
    if not isinstance(ex["session"], int):
        ex["session"] = 0
    return ex


def fingerprint(ex):
    raw = json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).digest()


def clean_file(path, keep_others, max_dup, max_dup_long, seen_replies, val_fraction, max_examples, seed):
    """Cleans ONE file. seen_replies is shared by all files so repeated replies are limited overall."""
    counts = collections.Counter()
    by_chat = collections.OrderedDict()
    for rec in read_records(path, counts):
        ex = normalise(rec, counts)
        if ex is not None:
            by_chat.setdefault(ex["chat"], []).append(ex)
            counts["examples_in"] += 1
    if max_examples and counts["examples_in"] > max_examples:
        # keep a random but repeatable subset of WHOLE chats' examples, in the original order
        flat = [(c, i) for c, exs in by_chat.items() for i in range(len(exs))]
        rng = random.Random(seed)
        keep = set(rng.sample(range(len(flat)), max_examples))
        picked = collections.defaultdict(set)
        for n, (c, i) in enumerate(flat):
            if n in keep:
                picked[c].add(i)
        by_chat = collections.OrderedDict((c, [e for i, e in enumerate(exs) if i in picked[c]]) for c, exs in by_chat.items() if picked[c])
        counts["cut_by_max_examples"] = counts["examples_in"] - max_examples
    train, val = [], []
    seen_examples = set()
    for cid in sorted(by_chat, key=CC.num_key):
        tr, va = CC.split_sessions(by_chat[cid], val_fraction)
        for split, items, bucket in (("train", tr, train), ("val", va, val)):
            for ex in items:
                fp = fingerprint(ex)
                if fp in seen_examples:
                    counts["dropped_duplicate_example"] += 1
                    continue
                seen_examples.add(fp)
                seen = seen_replies if split == "train" else collections.Counter()   # val is not limited by the train counts
                st = collections.Counter()
                out = CC.clean_example(ex, keep_others, max_dup, max_dup_long, seen, st)
                for k, v in st.items():
                    if k != "examples_in":
                        counts[k] += v
                if out:
                    bucket.append(out)
    counts["train_out"] = len(train)
    counts["val_out"] = len(val)
    return train, val, counts


def out_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return "%s_%s.jsonl" % (stem, TAG)


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def remove_old_output(out_dir):
    """Only files this script makes (*_cleanData.jsonl) are removed, so nothing else in the folder is touched."""
    for sub in ("train", "val"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith("_%s.jsonl" % TAG):
                    os.remove(os.path.join(d, fn))


def main(argv=None):
    ap = argparse.ArgumentParser(description="clean chat json/jsonl files; one *_cleanData file per input, nothing merged")
    ap.add_argument("--input", nargs="+", required=True, help="files and/or folders; list your own chats first")
    ap.add_argument("--out", default="data_clean", help="training folder (train\\ and val\\ are made inside it)")
    ap.add_argument("--val-fraction", type=float, default=0.05, help="share of each chat's last sessions held out for validation")
    ap.add_argument("--keep-others", action="store_true", help="keep targets spoken by other people in group chats")
    ap.add_argument("--max-dup", type=int, default=3, help="keep each identical short reply (<=25 chars) at most this many times in total (0 = no limit)")
    ap.add_argument("--max-dup-long", type=int, default=5, help="same for longer replies (0 = no limit)")
    ap.add_argument("--max-examples", type=int, default=0, help="per file: keep at most this many examples (random, repeatable). 0 = all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-top", action="store_true", help="also print the most common replies (shows chat text, cut to 30 characters)")
    a = ap.parse_args(argv)

    files = find_files(a.input)
    if not files:
        sys.exit("no .json / .jsonl files to clean")
    names = [out_name(p) for p in files]
    if len(set(names)) != len(names):
        sys.exit("two inputs would give the same output name; rename one of them")

    remove_old_output(a.out)
    seen_replies = collections.Counter()
    report = []
    for path, name in zip(files, names):
        train, val, c = clean_file(path, a.keep_others, a.max_dup, a.max_dup_long, seen_replies, a.val_fraction, a.max_examples, a.seed)
        if train:
            write_jsonl(os.path.join(a.out, "train", name), train)
        if val:
            write_jsonl(os.path.join(a.out, "val", name), val)
        report.append({"input": path, "output": name, **{k: v for k, v in sorted(c.items())}})
        print("\n%s  ->  %s" % (path, name))
        print("   examples read %d, kept: train %d, val %d" % (c["examples_in"], c["train_out"], c["val_out"]))
        for k in sorted(c):
            if k not in ("examples_in", "examples_out", "train_out", "val_out") and c[k]:
                print("   %-30s %d" % (k, c[k]))
        if not train:
            print("   WARNING: nothing left for training, no train file written")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"files": report}, f, indent=2, ensure_ascii=False)

    sizes = [r["train_out"] for r in report if r["train_out"]]
    if len(sizes) > 1 and max(sizes) > 20 * min(sizes):
        big = max(report, key=lambda r: r["train_out"])
        small = min((r for r in report if r["train_out"]), key=lambda r: r["train_out"])
        print("\nNOTE: %s has %d training examples and %s only %d. The big file will drown out the small one."
              % (big["output"], big["train_out"], small["output"], small["train_out"]))
        print("      If the small file is YOUR real chat, run again with --max-examples %d to cap each file." % max(small["train_out"] * 5, 1000))

    if a.show_top:
        mine = collections.Counter()
        print("\nMost repeated replies kept (train):")
        for key, n in seen_replies.most_common(6):
            print("   %r x%d" % (key[:30], min(n, a.max_dup if len(key) <= 25 else a.max_dup_long) if (a.max_dup or a.max_dup_long) else n))

    print("\nDone. %d file(s) in %s and %s (nothing merged)." % (len(files), os.path.join(a.out, "train"), os.path.join(a.out, "val")))
    print("The AI uses ALL of them. Next:\n  python -m persona_lm prepare --kind chat --tokenizer work\\tok.json --data %s --val-data %s --out data_chat"
          % (os.path.join(a.out, "train"), os.path.join(a.out, "val")))
    return report


if __name__ == "__main__":
    main()
