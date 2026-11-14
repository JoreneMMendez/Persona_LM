"""prep_mood_data.py - turn the cleaned + mood-tagged file into data_msgr\\train.jsonl and val.jsonl that `prepare --kind chat` reads correctly.

Usage (project folder):  python prep_mood_data.py PATH\\merged_chats_clean_mood.jsonl            (preview, writes nothing)
                         python prep_mood_data.py PATH\\merged_chats_clean_mood.jsonl --apply    (backs up train/val as *.bak, writes the new ones)

What it fixes:
 1. The mood line arrives as {"role":"system","content":"<|mood|> joy +0.4 +0.2 <|end|>"}. `prepare` would encode that as the role marker <|system|>
    plus ~12 ordinary text tokens. The chat screen builds the REAL special token <|mood|> + "joy +0.4 +0.2" + <|end|>. So it becomes
    {"role":"mood","content":"joy +0.4 +0.2"} (needs the one-line change in persona_lm\\bpe.py that goes with this script).
 2. Drops the few examples whose trained target is another person's turn (role user): the model should learn YOUR replies.
 3. Splits into train and val by whole conversation sessions (about 1 in 10 sessions to val), so overlapping context never sits on both sides.
"""
import json, os, re, sys, zlib, collections
MOOD_RE = re.compile(r"^<\|mood\|>\s+(\S+ [+-]\d\.\d [+-]\d\.\d)\s+<\|end\|>$")

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    src, apply_ = args[0], "--apply" in sys.argv
    train, val, st = [], [], collections.Counter()
    for line in open(src, encoding="utf-8"):
        if not line.strip():
            continue
        ex = json.loads(line); st["in"] += 1
        msgs = []
        for m in ex["messages"]:
            mm = MOOD_RE.match(m["content"]) if m["role"] == "system" else None
            if mm:
                msgs.append({"role": "mood", "content": mm.group(1), "train": False}); st["mood_converted"] += 1
            else:
                msgs.append(m)
        tgt = [m for m in msgs if m.get("train")]
        if not tgt or any(m["role"] != "assistant" for m in tgt):
            st["dropped_target_not_mine"] += 1
            continue
        ex["messages"] = msgs
        key = "%s/%s" % (ex.get("chat"), ex.get("session"))
        (val if zlib.crc32(key.encode()) % 10 == 0 else train).append(ex)
    print("examples in %d | mood lines converted %d | dropped (target not yours) %d" % (st["in"], st["mood_converted"], st["dropped_target_not_mine"]))
    print("train %d | val %d" % (len(train), len(val)))
    if not val or not train:
        sys.exit("a split is empty: stop and tell the AI")
    if not apply_:
        print("Nothing written. Run again with --apply."); return
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        p = os.path.join("data_msgr", name)
        if os.path.exists(p) and not os.path.exists(p + ".bak"):
            os.replace(p, p + ".bak")
        with open(p, "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Written data_msgr\\train.jsonl and val.jsonl (old ones kept as *.bak).")

if __name__ == "__main__":
    main()
