"""clean_chat.py - clean data_msgr\\train.jsonl and val.jsonl for fine-tuning.

Usage (from the project folder):
    python clean_chat.py            # writes data_msgr_clean\\train.jsonl and val.jsonl, prints a report
    python clean_chat.py --apply    # same, then backs up the originals (*.orig) and replaces them

What it removes:
  1. System notices that leaked through the anonymizer ("[HANDLE] Person_12 joined the video call", "Person_03 missed a call", ...)
     - dropped as a training target, and removed from the context.
  2. Targets spoken by OTHER people in group chats ("Person_65: ...") - the model copied that prefix. Their turns stay as context.
     (use --keep-others to keep them)
  3. Over-repeated identical short replies ("Gg", "Pubg", ...): each identical target is kept at most --max-dup times (default 3).
Your own replies are never touched, except by rule 1 and 3.
"""
import json, os, re, sys, argparse, collections

SYSTEM_PATTERNS = [
    r"joined the (video|audio) call", r"left the (video|audio) call", r"(started|ended) (a|the) (video|audio) (call|chat)",
    r"missed (a|your) (video|audio)? ?call", r"(video|audio) (call|chat) ended", r"\bcall ended\b", r"\bcalled you\b",
    r"joined the (group|chat)", r"left the (group|chat)", r"joined (via|using) .*link",
    r"(changed|set|named|removed|added) (the )?(group|chat|nickname|theme|emoji|photo|name)",
    r"(reacted|liked|loved|laughed at|emphasized) .* (message|a message)", r"\bwaved at\b", r"sent an attachment",
    r"is now (your )?(friend|connected)", r"you are now connected", r"(created|started) (a )?(poll|group|plan)",
    r"unsent a message", r"removed a message",
]
SYS_RE = re.compile("|".join(SYSTEM_PATTERNS), re.I)
SPEAKER_RE = re.compile(r"^Person_\S+?:\s")


def is_system(text):
    t = text.strip()
    if not t:
        return True
    # a short line that is mostly a placeholder + a notice phrase
    return bool(SYS_RE.search(t)) and len(t) < 200


def clean_file(src, dst, keep_others, max_dup, stats, seen):
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            stats["examples_in"] += 1
            kept = []
            for m in msgs:
                if m["role"] != "system" and is_system(m["content"]):
                    stats["system_msgs_removed"] += 1
                    continue
                kept.append(m)
            # which messages carry the loss?
            if not any(m.get("train") for m in kept):
                stats["dropped_no_target"] += 1
                continue
            target = [m for m in kept if m.get("train")][-1]
            if not keep_others and target["role"] == "user" and SPEAKER_RE.match(target["content"]):
                stats["dropped_other_group_speaker"] += 1
                continue
            if target["role"] == "assistant" and len(target["content"]) <= 25:
                key = target["content"].strip().lower()
                seen[key] += 1
                if seen[key] > max_dup:
                    stats["dropped_repeated_short"] += 1
                    continue
            # context must end before the target and start with a non-assistant turn is not required; keep as is
            ex["messages"] = kept
            out.append(ex)
            stats["examples_out"] += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_msgr")
    ap.add_argument("--out", default="data_msgr_clean")
    ap.add_argument("--keep-others", action="store_true", help="keep targets spoken by other people in group chats")
    ap.add_argument("--max-dup", type=int, default=3)
    ap.add_argument("--apply", action="store_true", help="back up the originals as *.orig and replace them with the clean files")
    a = ap.parse_args()
    for name in ("train.jsonl", "val.jsonl"):
        src, dst = os.path.join(a.dir, name), os.path.join(a.out, name)
        if not os.path.exists(src):
            sys.exit("not found: " + src)
        stats, seen = collections.Counter(), collections.Counter()
        out = clean_file(src, dst, a.keep_others, a.max_dup, stats, seen)
        print("%s: %d -> %d examples" % (name, stats["examples_in"], stats["examples_out"]))
        for k in ("system_msgs_removed", "dropped_no_target", "dropped_other_group_speaker", "dropped_repeated_short"):
            print("   %-30s %d" % (k, stats[k]))
        mine = [m["content"] for e in out for m in e["messages"] if m.get("train") and m["role"] == "assistant"]
        print("   your reply targets kept:       %d" % len(mine))
        if name == "train.jsonl":
            top = collections.Counter(c.strip().lower() for c in mine).most_common(8)
            print("   most common replies now:", ", ".join("%r x%d" % t for t in top))
    if a.apply:
        for name in ("train.jsonl", "val.jsonl"):
            src = os.path.join(a.dir, name)
            if not os.path.exists(src + ".orig"):
                os.replace(src, src + ".orig")
            os.replace(os.path.join(a.out, name), src)
        print("\nApplied. Originals saved as data_msgr\\train.jsonl.orig and val.jsonl.orig")
    else:
        print("\nNothing replaced yet. Check the numbers, then run again with --apply.")


if __name__ == "__main__":
    main()
