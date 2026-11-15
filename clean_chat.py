"""clean_chat.py - clean chat training data and write ONE FILE PER CHAT (nothing is merged).

Two ways to run it (from the project folder):

  A) Your own ready-made .jsonl files (no separate val file - the script holds out the last sessions of each chat):
       python clean_chat.py --input merged_chats_clean_mood.jsonl synthetic_chats_100k.jsonl

  B) The output of messenger-import (data_msgr\\train.jsonl + val.jsonl):
       python clean_chat.py --dir data_msgr

Output (default folder data_msgr_chats):
    train\\chat_001.jsonl, chat_002.jsonl, ...   one numbered file per chat
    val\\chat_001.jsonl,   chat_002.jsonl, ...   that chat's validation examples (if it has any)
    index.json                                   file number -> original chat id, example counts

Train on ALL of them at once (prepare reads whole folders):
    python -m persona_lm prepare --kind chat --tokenizer tok.json --data data_msgr_chats\\train --val-data data_msgr_chats\\val --out data_chat

What it cleans:
  1. System notices that leaked in ("joined the video call", "missed a call", ...) - dropped as targets and removed
     from the context. The <|mood|> line (role "system") is kept.
  2. Targets spoken by OTHER people in group chats ("Person_65: ...") - the model copied that prefix. Their turns stay
     as context. (--keep-others keeps them)
  3. Repeated identical replies, counted over ALL chats together: a short reply (25 characters or less) is kept at most
     --max-dup times (default 3), a longer one at most --max-dup-long times (default 5). Stops the model memorizing
     stock lines. 0 = no limit.
Chats left with no training examples get no file.
"""
import argparse
import collections
import json
import os
import re
import sys

SYSTEM_PATTERNS = [
    r"joined the (video|audio) call", r"left the (video|audio) call", r"(started|ended) (a|the) (video|audio) (call|chat)",
    r"you joined the call", r"missed (a|your) (video|audio)? ?call", r"(video|audio) (call|chat) ended", r"\bcall ended\b",
    r"\bcalled you\b", r"joined the (group|chat)", r"left the (group|chat)", r"joined (via|using) .*link",
    r"(changed|set|named|removed|added) (the )?(group|chat|nickname|theme|emoji|photo|name)",
    r"(reacted|liked|loved|laughed at|emphasized) .* (message|a message)", r"\bwaved at\b", r"sent an attachment",
    r"is now (your )?(friend|connected)", r"you are now connected", r"(created|started) (a )?(poll|group|plan)",
    r"unsent a message", r"removed a message",
]
SYS_RE = re.compile("|".join(SYSTEM_PATTERNS), re.I)
SPEAKER_RE = re.compile(r"^Person_\S+?:\s")


def is_system(text):
    t = (text or "").strip()
    if not t:
        return True
    return bool(SYS_RE.search(t)) and len(t) < 200


def clean_example(ex, keep_others, max_short, max_long, seen, stats):
    """Return the cleaned example, or None if it must be dropped."""
    stats["examples_in"] += 1
    kept = []
    for m in ex["messages"]:
        if m.get("role") != "system" and is_system(m.get("content")):
            stats["system_msgs_removed"] += 1
            continue
        kept.append(m)
    targets = [m for m in kept if m.get("train")]
    if not targets:
        stats["dropped_no_target"] += 1
        return None
    target = targets[-1]
    if not keep_others and target["role"] == "user" and SPEAKER_RE.match(target["content"]):
        stats["dropped_other_group_speaker"] += 1
        return None
    if target["role"] == "assistant":
        short = len(target["content"]) <= 25
        limit = max_short if short else max_long
        if limit > 0:
            key = target["content"].strip().lower()
            seen[key] += 1
            if seen[key] > limit:
                stats["dropped_repeated_short" if short else "dropped_repeated_long"] += 1
                return None
    ex["messages"] = kept
    stats["examples_out"] += 1
    return ex


def num_key(cid):
    nums = re.findall(r"\d+", str(cid))
    return (int(nums[-1]) if nums else 10 ** 9, str(cid))


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def split_sessions(examples, val_fraction):
    """Hold out the LAST sessions of one chat. Whole sessions go to one side, so overlapping windows never straddle."""
    sessions = sorted({e.get("session", 0) for e in examples})
    n_val = int(round(len(sessions) * val_fraction))
    if n_val == 0 and len(sessions) >= 10:
        n_val = 1
    held = set(sessions[len(sessions) - n_val:]) if n_val else set()
    return [e for e in examples if e.get("session", 0) not in held], [e for e in examples if e.get("session", 0) in held]


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_dir(d):
    """Reads d/train.jsonl and d/val.jsonl (the output of messenger-import). Returns (raw_train, raw_val): lists of (order, chat id, [examples])."""
    raw_train, raw_val = [], []
    for split, bucket in (("train", raw_train), ("val", raw_val)):
        path = os.path.join(d, split + ".jsonl")
        if not os.path.exists(path):
            if split == "train":
                sys.exit("not found: " + path)
            continue
        by_chat = collections.OrderedDict()
        for ex in read_jsonl(path):
            by_chat.setdefault(str(ex.get("chat", "unknown")), []).append(ex)
        for cid in sorted(by_chat, key=num_key):
            bucket.append(((0, num_key(cid)), cid, by_chat[cid]))
    return raw_train, raw_val


def run_clean(raw_train, raw_val, keep_others, max_dup, max_dup_long):
    """The cleaning itself, in memory, nothing written. Returns (cleaned, stats). Note: it edits the examples it keeps."""
    stats = {"train": collections.Counter(), "val": collections.Counter()}
    cleaned = {"train": collections.OrderedDict(), "val": {}}
    for split, bucket in (("train", raw_train), ("val", raw_val)):
        seen = collections.Counter()
        for order, cid, exs in sorted(bucket, key=lambda b: b[0]):
            out = [e for e in (clean_example(x, keep_others, max_dup, max_dup_long, seen, stats[split]) for x in exs) if e]
            if out:
                cleaned[split][(order, cid)] = out
    return cleaned, stats


def write_output(out_dir, cleaned):
    """Writes out_dir/train/chat_NNN.jsonl, val/chat_NNN.jsonl and index.json (earlier chat files in it are removed first). Returns (index, lost_val)."""
    for sub in ("train", "val"):                      # start clean: files from an earlier run must not stay
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".jsonl"):
                    os.remove(os.path.join(d, fn))
    index = []
    for n, key in enumerate(cleaned["train"], 1):
        name = "chat_%03d.jsonl" % n
        write_jsonl(os.path.join(out_dir, "train", name), cleaned["train"][key])
        v = cleaned["val"].get(key, [])
        if v:
            write_jsonl(os.path.join(out_dir, "val", name), v)
        index.append({"number": n, "file": name, "original_chat": key[1], "train_examples": len(cleaned["train"][key]), "val_examples": len(v)})
    lost_val = [k[1] for k in cleaned["val"] if k not in cleaned["train"]]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"files": index, "val_chats_without_train_file": lost_val}, f, indent=2, ensure_ascii=False)
    return index, lost_val


def main():
    ap = argparse.ArgumentParser(description="clean chat data and write one numbered file per chat")
    ap.add_argument("--input", nargs="+", help="ready-made .jsonl files (chat/session/messages); val is split off automatically")
    ap.add_argument("--dir", default="data_msgr", help="folder with train.jsonl + val.jsonl from messenger-import (used when --input is not given)")
    ap.add_argument("--out", default="data_msgr_chats", help="output folder (train\\, val\\, index.json)")
    ap.add_argument("--val-fraction", type=float, default=0.05, help="--input only: share of each chat's last sessions held out for validation")
    ap.add_argument("--keep-others", action="store_true", help="keep targets spoken by other people in group chats")
    ap.add_argument("--max-dup", type=int, default=3, help="keep each identical short reply (<=25 chars) at most this many times in total")
    ap.add_argument("--max-dup-long", type=int, default=5, help="same for longer replies")
    a = ap.parse_args()

    # raw[split] = list of (order, chat id, [examples])
    raw_train, raw_val = [], []
    if a.input:
        for fi, path in enumerate(a.input):
            if not os.path.exists(path):
                sys.exit("not found: " + path)
            by_chat = collections.OrderedDict()
            for ex in read_jsonl(path):
                by_chat.setdefault(str(ex.get("chat", "unknown")), []).append(ex)
            for cid in sorted(by_chat, key=num_key):
                tr, va = split_sessions(by_chat[cid], a.val_fraction)
                raw_train.append(((fi, num_key(cid)), cid, tr))
                if va:
                    raw_val.append(((fi, num_key(cid)), cid, va))
    else:
        raw_train, raw_val = load_dir(a.dir)

    cleaned, stats = run_clean(raw_train, raw_val, a.keep_others, a.max_dup, a.max_dup_long)

    index, lost_val = write_output(a.out, cleaned)

    for split in ("train", "val"):
        s = stats[split]
        print("%s: %d -> %d examples" % (split, s["examples_in"], s["examples_out"]))
        for k in ("system_msgs_removed", "dropped_no_target", "dropped_other_group_speaker", "dropped_repeated_short", "dropped_repeated_long"):
            print("   %-30s %d" % (k, s[k]))
    mine = [m["content"] for exs in cleaned["train"].values() for e in exs for m in e["messages"] if m.get("train") and m["role"] == "assistant"]
    print("   reply targets kept (train): %d, distinct %d" % (len(mine), len(set(c.strip().lower() for c in mine))))
    top = collections.Counter(c.strip().lower() for c in mine).most_common(6)
    print("   most common replies now:", ", ".join("%r x%d" % (t[0][:30], t[1]) for t in top))
    if lost_val:
        print("   note: %d chat(s) have val examples but no train file left (their val was not written)" % len(lost_val))
    print("\nWrote %d chat files to %s (index: %s)" % (len(index), os.path.join(a.out, "train"), os.path.join(a.out, "index.json")))
    print("Next:\n  python -m persona_lm prepare --kind chat --tokenizer tok.json --data %s --val-data %s --out data_chat" % (
        os.path.join(a.out, "train"), os.path.join(a.out, "val")))


if __name__ == "__main__":
    main()
