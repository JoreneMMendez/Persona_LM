"""Stage 6 data tools: `python -m persona_lm data-audit` (READ-ONLY).

  python -m persona_lm data-audit --train data_clean\\train --val data_clean\\val --out data_audit --real merged_chats

Reads the cleaned chat files (*.jsonl made by clean_data.py) and writes a report into a NEW folder (`--out` must not exist yet or
must be empty, and may not sit inside the data folders). The input files are opened for reading only.
It reports:
  1. de-duplication: exact duplicate examples, how often each reply is repeated (stock replies), replies above clean_data's limits
  2. real vs synthetic: how much of what the model LEARNS FROM (the trained replies) comes from your own chats (`--real` = parts of file names)
  3. train/val leak check: examples, sessions or long messages that sit on both sides, which would make validation loss look too good
No chat text is ever printed or written: only counts, file names, line numbers and short hashes.
"""
import collections
import hashlib
import json
import os

SHORT = 25                      # same limit as clean_data.py (--max-dup counts short replies, --max-dup-long the rest)
MAX_DUP, MAX_DUP_LONG = 3, 5    # clean_data.py defaults


def _norm(text):
    return " ".join(text.lower().split())


def _hash(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _example_fp(msgs):
    return _hash(json.dumps([[m["role"], m["content"]] for m in msgs], ensure_ascii=False))


def _trained(m):
    t = m.get("train")
    return (m.get("role") == "assistant") if t is None else bool(t)


def read_examples(path):
    """Yields (line_number, example) for every readable record; (line, None) for a bad line. Read-only."""
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                msgs = [m for m in rec["messages"] if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()]
            except (ValueError, KeyError, TypeError):
                yield n, None
                continue
            if msgs:
                yield n, {"chat": str(rec.get("chat", "unknown")), "session": rec.get("session", 0), "messages": msgs}
            else:
                yield n, None


def list_files(folder):
    return sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(".jsonl"))


class Split:
    """Everything the audit needs to know about one folder (train or val), kept as hashes only."""
    def __init__(self, folder):
        self.folder = folder
        self.files = []                                   # per-file stats
        self.example_first = {}                           # example fingerprint -> (file name, line)
        self.dups = []                                    # (file, line, first file, first line)
        self.reply_count = collections.Counter()          # normalised trained reply hash -> times
        self.reply_short = {}                             # hash -> True when short
        self.msg_hashes = set()                           # every message longer than SHORT
        self.reply_hashes = set()                         # every trained reply (any length)
        self.sessions = collections.defaultdict(set)      # file name -> {(chat, session)}
        for path in list_files(folder):
            name = os.path.basename(path)
            st = dict(file=name, examples=0, bad_lines=0, trained_replies=0, trained_chars=0)
            for n, ex in read_examples(path):
                if ex is None:
                    st["bad_lines"] += 1
                    continue
                st["examples"] += 1
                fp = _example_fp(ex["messages"])
                if fp in self.example_first:
                    self.dups.append((name, n) + self.example_first[fp])
                else:
                    self.example_first[fp] = (name, n)
                self.sessions[name].add((ex["chat"], ex["session"]))
                for m in ex["messages"]:
                    h = _hash(_norm(m["content"]))
                    if len(m["content"].strip()) > SHORT:
                        self.msg_hashes.add(h)
                    if _trained(m):
                        st["trained_replies"] += 1
                        st["trained_chars"] += len(m["content"])
                        self.reply_count[h] += 1
                        self.reply_short[h] = len(m["content"].strip()) <= SHORT
                        self.reply_hashes.add(h)
            self.files.append(st)

    def total(self, key):
        return sum(f[key] for f in self.files)


def dedup_report(tr):
    counts = tr.reply_count
    buckets = collections.OrderedDict((k, 0) for k in ("1", "2-3", "4-10", "over 10"))
    for c in counts.values():
        buckets["1" if c == 1 else "2-3" if c <= 3 else "4-10" if c <= 10 else "over 10"] += 1
    total = sum(counts.values())
    repeated_instances = sum(c for c in counts.values() if c > 3)
    over = sum(1 for h, c in counts.items() if c > (MAX_DUP if tr.reply_short[h] else MAX_DUP_LONG))
    ex = tr.total("examples")
    return {"examples": ex, "exact_duplicate_examples": len(tr.dups), "unique_examples": ex - len(tr.dups),
            "trained_replies": total, "distinct_replies": len(counts), "distinct_reply_buckets_by_times_used": dict(buckets),
            "share_of_replies_used_more_than_3_times": round(repeated_instances / total, 4) if total else 0.0,
            "replies_over_clean_data_limit": over}


def real_vs_synthetic(tr, real_parts):
    rows = []
    tot_r = max(tr.total("trained_replies"), 1)
    for f in tr.files:
        is_real = any(p.lower() in f["file"].lower() for p in real_parts) if real_parts else None
        rows.append({**f, "share_of_trained_replies": round(f["trained_replies"] / tot_r, 4), "real": is_real})
    out = {"files": rows, "real_marked": bool(real_parts)}
    if real_parts:
        real = [r for r in rows if r["real"]]
        syn = [r for r in rows if not r["real"]]
        rr, sr = sum(r["trained_replies"] for r in real), sum(r["trained_replies"] for r in syn)
        out.update(real_replies=rr, synthetic_replies=sr, real_share=round(rr / max(rr + sr, 1), 4),
                   real_files=[r["file"] for r in real], unmatched_parts=[p for p in real_parts if not any(p.lower() in r["file"].lower() for r in rows)])
    return out


def leak_check(tr, va):
    res = {}
    exact = sum(1 for fp in va.example_first if fp in tr.example_first)
    long_vals = [h for h in va.reply_hashes if va.reply_short[h] is False]
    long_seen = sum(1 for h in long_vals if h in tr.msg_hashes)
    short_vals = [h for h in va.reply_hashes if va.reply_short[h]]
    short_seen = sum(1 for h in short_vals if h in tr.reply_hashes)
    ctx = len(va.msg_hashes)
    ctx_seen = sum(1 for h in va.msg_hashes if h in tr.msg_hashes)
    shared, checked = 0, 0
    for name, keys in va.sessions.items():
        tk = tr.sessions.get(name)
        if tk is None or len(keys) < 2 or len(tk) < 2:
            continue                                                    # no real session numbers (e.g. synthetic data): cannot check
        checked += 1
        shared += len(keys & tk)
    res.update(exact_examples_in_both=exact,
               val_long_replies=len(long_vals), val_long_replies_seen_in_train=long_seen,
               val_long_reply_overlap=round(long_seen / len(long_vals), 4) if long_vals else 0.0,
               val_short_replies=len(short_vals), val_short_replies_seen_in_train=short_seen,
               val_long_messages=ctx, val_long_messages_seen_in_train=ctx_seen,
               val_long_message_overlap=round(ctx_seen / ctx, 4) if ctx else 0.0,
               files_with_session_check=checked, sessions_in_both=shared)
    return res


def verdicts(dd, rs, lk):
    """(level, message) list in plain words."""
    v = []
    if lk["exact_examples_in_both"] or lk["sessions_in_both"]:
        v.append(("PROBLEM", "%d example(s) and %d session(s) are in BOTH train and val, so validation loss looks better than the model really is. "
                  "Run clean_data.py again (it splits by session) and do not trust the current val numbers." % (lk["exact_examples_in_both"], lk["sessions_in_both"])))
    if lk["val_long_reply_overlap"] > 0.02:
        v.append(("WARNING", "%.1f%% of the long validation replies already appear somewhere in the training text (window mode repeats messages; "
                  "a little is normal, a lot means validation is partly memorized)." % (100 * lk["val_long_reply_overlap"])))
    if lk["val_long_message_overlap"] > 0.05:
        v.append(("WARNING", "%.1f%% of the long validation messages also appear in training." % (100 * lk["val_long_message_overlap"])))
    if dd["exact_duplicate_examples"]:
        v.append(("WARNING", "%d exact duplicate example(s) in training; clean_data.py normally removes them (were the files cleaned together?)." % dd["exact_duplicate_examples"]))
    if dd["replies_over_clean_data_limit"]:
        v.append(("WARNING", "%d reply(ies) are repeated more often than clean_data.py allows (3 short / 5 long). The model will memorize them as stock replies. "
                  "Clean ALL files again in one run." % dd["replies_over_clean_data_limit"]))
    if not rs["real_marked"]:
        v.append(("INFO", "Tell me which files are your real chats with --real <part of the file name> to get the real-vs-synthetic ratio."))
    else:
        if rs["unmatched_parts"]:
            v.append(("WARNING", "--real part(s) matching no file: %s" % ", ".join(rs["unmatched_parts"])))
        if rs["real_share"] < 0.2:
            v.append(("WARNING", "Your real chats are only %.0f%% of what the model learns from. Replies may sound generic: lower --max-examples in clean_data.py "
                      "so the synthetic files cannot drown out your own." % (100 * rs["real_share"])))
    if not v:
        v.append(("OK", "No leak, no duplicates above the limits."))
    return v


def audit(train_dir, val_dir, out_dir, real_parts=(), say=print):
    for d in (train_dir, val_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError("folder not found: %s" % d)
    out_abs = os.path.abspath(out_dir)
    for d in (train_dir, val_dir):
        da = os.path.abspath(d)
        if out_abs == da or out_abs.startswith(da + os.sep):
            raise ValueError("--out must be a NEW folder outside the data folders (the data is never written to)")
    if os.path.isdir(out_abs) and os.listdir(out_abs):
        raise ValueError("--out %s already holds files; give a new folder name (nothing is overwritten)" % out_dir)
    if not list_files(train_dir):
        raise ValueError("no .jsonl files in %s" % train_dir)
    tr, va = Split(train_dir), Split(val_dir)
    dd, rs, lk = dedup_report(tr), real_vs_synthetic(tr, list(real_parts)), leak_check(tr, va)
    vs = verdicts(dd, rs, lk)
    os.makedirs(out_abs, exist_ok=True)
    report = {"train_dir": train_dir, "val_dir": val_dir, "dedup": dd, "real_vs_synthetic": rs, "leak_check": lk,
              "val_files": va.files, "verdicts": [{"level": a, "message": b} for a, b in vs]}
    with open(os.path.join(out_abs, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_abs, "exact_duplicates.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("file\tline\tsame_as_file\tsame_as_line\n")
        for row in tr.dups[:100000]:
            f.write("%s\t%d\t%s\t%d\n" % row)
    lines = ["# Data audit (counts only, no chat text)", ""]
    for lvl, msg in vs:
        lines.append("- **%s**: %s" % (lvl, msg))
    lines += ["", "## Training files", "", "| file | examples | trained replies | share of replies | real |", "|---|---|---|---|---|"]
    for r in rs["files"]:
        lines.append("| %s | %d | %d | %.1f%% | %s |" % (r["file"], r["examples"], r["trained_replies"], 100 * r["share_of_trained_replies"],
                                                         "?" if r["real"] is None else ("yes" if r["real"] else "no")))
    lines += ["", "## De-duplication", ""] + ["- %s: %s" % (k.replace("_", " "), v) for k, v in dd.items()]
    lines += ["", "## Train/val leak check", ""] + ["- %s: %s" % (k.replace("_", " "), v) for k, v in lk.items() ]
    with open(os.path.join(out_abs, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    say("Data audit (read-only). Report written to %s (report.md, report.json, exact_duplicates.tsv)" % out_dir)
    say("  train: %d examples in %d file(s), %d trained replies (%d distinct); val: %d examples" % (
        dd["examples"], len(tr.files), dd["trained_replies"], dd["distinct_replies"], va.total("examples")))
    if rs["real_marked"]:
        say("  real chats: %.1f%% of trained replies (%d real, %d other)" % (100 * rs["real_share"], rs["real_replies"], rs["synthetic_replies"]))
    for lvl, msg in vs:
        say("  [%s] %s" % (lvl, msg))
    return 1 if any(l == "PROBLEM" for l, _ in vs) else 0


def add_arguments(s):
    s.add_argument("--train", default=os.path.join("data_clean", "train"), help="folder with the cleaned training .jsonl files")
    s.add_argument("--val", default=os.path.join("data_clean", "val"), help="folder with the cleaned validation .jsonl files")
    s.add_argument("--out", required=True, help="NEW folder for the report (must not exist yet or be empty)")
    s.add_argument("--real", nargs="*", default=[], help="parts of file names that are YOUR real chats, e.g. --real merged_chats")


def cmd(a):
    try:
        code = audit(a.train, a.val, a.out, a.real)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit("error: %s" % e)
    if code:
        raise SystemExit(code)
