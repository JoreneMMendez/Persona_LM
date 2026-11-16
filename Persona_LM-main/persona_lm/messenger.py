"""Facebook Messenger JSON export -> chat training data (all local, no network).

The pipeline, in the order the code does it:
  1. find_threads / load_thread   read every conversation folder (message_1.json, message_2.json, ...), repair the broken text encoding,
                                  drop everything that is not a typed message (photos, stickers, calls, system notices, unsent ...)
  2. you choose which sender is "me" and which chats to include
  3. Anonymizer                   (anonymize.py) masks names, phones, e-mails, links, addresses
  4. split_sessions               a long silence (default 6 h) starts a new "session": context never reaches across days
  5. merge_turns                  consecutive messages from the same sender become ONE turn (joined with newlines)
  6. window_examples / session_examples
                                  turn the conversation into chat examples. Other people = role "user", you = role "assistant".
                                  Loss is placed ONLY on your messages (a per-message "train" flag, see data.encode_chat)
  7. time split                   the most recent sessions become validation data (whole sessions, so no overlap with training)
  8. write train.jsonl / val.jsonl / stats.json, plus a private/ folder that must never be uploaded

The result is read by:  python -m persona_lm prepare --kind chat --data train.jsonl --val-data val.jsonl ...
"""
import collections
import datetime
import json
import math
import os
import random
import re
from dataclasses import dataclass, field

from .anonymize import Anonymizer, capitalised_word_report

# message_1.json, and also the name Android gives a file it moved to its trash: .trashed-1797028852-message_1.json (the content is the same JSON)
THREAD_FILE_RE = re.compile(r"^(?:\.trashed-(\d+)-)?message_(\d+)\.json$")


def thread_files(names):
    """[(number, file name)] of the message files in a folder, oldest number first. If a number exists both as message_N.json and as a
    .trashed-...-message_N.json copy, only the normal file is used (never both: that would count every message twice); between several
    trashed copies the most recently trashed one wins."""
    best = {}
    for n in names:
        m = THREAD_FILE_RE.match(n)
        if not m:
            continue
        rank = (1, 0) if m.group(1) is None else (0, int(m.group(1)))
        num = int(m.group(2))
        if num not in best or rank > best[num][0]:
            best[num] = (rank, n)
    return sorted((num, name) for num, (_, name) in best.items())
KNOWN_KINDS = ("inbox", "archived_threads", "filtered_threads", "message_requests", "e2ee_cutover")
URL_ONLY_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.I)

# Notices that Messenger stores as ordinary "Generic" messages. These are English; an export made with the account language set to
# something else has system notices in that language that this list will NOT recognise (they would then be treated as normal text).
SYSTEM_TEXT_RES = [re.compile(p, re.I | re.S) for p in [
    r".* named the group .*", r".* changed the group (photo|name)\.?", r".* (added|removed) .* (to|from) the (group|chat)\.?",
    r".* left the (group|chat)\.?", r".* joined the (group|chat)\.?", r".* joined (via|using) .*link.*",
    r".* (set|changed|cleared) (the |his |her |their |your )?(own )?nickname.*", r".* set the nickname for .*",
    r".* changed the (chat )?(colou?rs?|theme|emoji).*", r".* sent an attachment\.?", r".* sent (a|an) (photo|video|sticker|voice message|gif|file)\.?",
    r".* (missed|started|ended) (a|the) (video |audio |group )?(call|chat).*", r"you (missed|started|ended) .*call.*",
    r"the (video|audio) (chat|call) ended.*", r".* reacted .* to (your|a|this) message.*", r"^reacted .* to .*",
    r"^(liked|loved) a message\.?$", r".* created (the |a )?(group|poll|plan|event).*", r".* shared (a |an )?(location|contact).*",
    r"you are now connected on messenger\.?", r"you can now (message|call) .*", r".* (waved|poked) .*", r".* is now your .*",
    r"this message was unsent\.?", r".* unsent a message\.?", r"you unsent a message\.?", r".* pinned a message.*", r".* turned (on|off) .*",
]]


def fix_mojibake(s):
    """Messenger exports UTF-8 text that was wrongly decoded as latin-1, so "é" arrives as "Ã©" and an emoji as "ð\x9f\x98\x80".
    Undo it: turn the characters back into the bytes they came from, then decode those bytes as UTF-8.
    Text that is already correct (it contains characters latin-1 cannot hold, or it is not valid UTF-8 as bytes) is returned unchanged."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


@dataclass
class Msg:
    sender: str
    ts: int          # milliseconds since 1970
    text: str


@dataclass
class Thread:
    folder: str
    kind: str
    title: str
    participants: list
    messages: list                                   # chronological, typed text only
    dropped: collections.Counter = field(default_factory=collections.Counter)
    raw_count: int = 0
    index: int = 0                                   # 1-based position in the scan table; chat ids in the output are chat_<index>

    @property
    def is_group(self):
        return len(self.participants) > 2

    def sender_counts(self):
        return collections.Counter(m.sender for m in self.messages)


# ------------------------------------------------------------------ 1. reading the export
def find_threads(root):
    """Every folder under `root` that contains message_N.json files. Works whether you point at the whole unzipped export,
    at its messages/ folder or at a single inbox. Returns [(folder, kind)]."""
    found = []
    for dirpath, _, names in os.walk(root):
        if any(THREAD_FILE_RE.match(n) for n in names):
            parent = os.path.basename(os.path.dirname(dirpath))
            found.append((dirpath, parent if parent in KNOWN_KINDS else "other"))
    return sorted(found)


def clean_message(m):
    """One raw message dict -> (Msg or None, reason it was dropped)."""
    if not isinstance(m, dict) or not isinstance(m.get("sender_name"), str) or not isinstance(m.get("timestamp_ms"), (int, float)):
        return None, "malformed"
    if m.get("is_unsent"):
        return None, "unsent"
    if "call_duration" in m or m.get("type") == "Call":
        return None, "call"
    if m.get("type", "Generic") != "Generic":
        return None, "system_or_other_type"                      # Subscribe, Unsubscribe, Share, Payment, Plan ...
    content = m.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "no_text"                                   # photo, video, sticker, voice note, gif or file without a caption
    text = fix_mojibake(content).strip()
    if any(r.match(text) for r in SYSTEM_TEXT_RES):
        return None, "system_text"
    if URL_ONLY_RE.match(text):
        return None, "link_only"
    return Msg(fix_mojibake(m["sender_name"]).strip(), int(m["timestamp_ms"]), text), None


def load_thread(folder, kind="inbox"):
    files = thread_files(os.listdir(folder))
    participants, title, msgs, dropped, raw = [], "", [], collections.Counter(), 0
    for _, name in files:
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            data = json.load(f)
        if not participants:
            participants = [fix_mojibake(p.get("name", "")).strip() for p in data.get("participants", []) if isinstance(p, dict)]
            title = fix_mojibake(data.get("title", "") or "")
        for m in data.get("messages", []):
            raw += 1
            msg, reason = clean_message(m)
            if reason:
                dropped[reason] += 1
            else:
                msgs.append(msg)
    msgs.sort(key=lambda x: x.ts)                                # the files are newest-first; sort() is stable for equal timestamps
    return Thread(folder, kind, title, participants, msgs, dropped, raw)


def load_table(export_dir, log=print):
    """Load every conversation. Sorted by number of typed messages (largest first) and numbered from 1: that number is what you pass to --chats."""
    found = find_threads(export_dir)
    if not found:
        raise SystemExit("no conversations found under %s (looking for folders that contain message_1.json). Did you export in JSON format "
                         "(not HTML)?" % export_dir)
    threads = []
    for i, (folder, kind) in enumerate(found, 1):
        if i % 50 == 0:
            log("  read %d/%d conversation folders ..." % (i, len(found)))
        threads.append(load_thread(folder, kind))
    threads.sort(key=lambda t: (-len(t.messages), t.kind, t.folder))
    for i, t in enumerate(threads, 1):
        t.index = i
    return threads


def rank_senders(threads):
    c = collections.Counter()
    for t in threads:
        c.update(t.sender_counts())
    return c


def _date(ts_ms):
    return datetime.datetime.fromtimestamp(ts_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")


def format_table(threads, me_set=None, top=60):
    lines = ["  #   kind              people  typed msgs%s  period                  title" % ("  mine" if me_set else "")]
    for t in threads[:top]:
        mine = ("  %5d" % sum(1 for m in t.messages if m.sender.casefold() in me_set)) if me_set else ""
        period = "%s..%s" % (_date(t.messages[0].ts), _date(t.messages[-1].ts)) if t.messages else "(empty)"
        lines.append("%3d   %-16s %6d  %10d%s  %-22s  %s" % (t.index, t.kind, len(t.participants), len(t.messages), mine, period, t.title[:40]))
    if len(threads) > top:
        lines.append("  ... %d more (use --top N to show more)" % (len(threads) - top))
    return "\n".join(lines)


# ------------------------------------------------------------------ 2. choosing "me" and the chats
def parse_selection(text, n):
    """"all", "3", "1,3,5-9" -> sorted list of 1-based numbers in 1..n."""
    text = text.strip().lower()
    if text in ("all", "*"):
        return list(range(1, n + 1))
    out = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    bad = sorted(i for i in out if not 1 <= i <= n)
    if bad or not out:
        raise ValueError("choose numbers between 1 and %d (got %s)" % (n, text))
    return sorted(out)


def choose_me(senders, input_fn=None, out=print):
    """Interactive: show the busiest senders, ask which one is you. Returns a list of names (you may be known under more than one)."""
    input_fn = input_fn or input            # looked up NOW (not when the file loads), so tests can replace the keyboard
    ranked = senders.most_common(12)
    out("Who are you? The senders with the most messages:")
    for i, (name, n) in enumerate(ranked, 1):
        out("  %2d  %s  (%d messages)" % (i, name, n))
    while True:
        raw = input_fn("Type the number (or the exact name). Several names for the same person: separate with commas: ").strip()
        names = []
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            if part.isdigit() and 1 <= int(part) <= len(ranked):
                names.append(ranked[int(part) - 1][0])
            elif part in senders:
                names.append(part)
            else:
                names = []
                out("  '%s' is not one of the senders, try again." % part)
                break
        if names:
            return names


# ------------------------------------------------------------------ 4-6. sessions, turns, examples
@dataclass
class Turn:
    who: str         # "me" or "other"
    speaker: str     # display label (a pseudonym when anonymizing)
    text: str
    ts: int


def split_sessions(msgs, gap_hours):
    gap = gap_hours * 3600 * 1000
    sessions, cur = [], []
    for m in msgs:
        if cur and m.ts - cur[-1].ts > gap:
            sessions.append(cur)
            cur = []
        cur.append(m)
    if cur:
        sessions.append(cur)
    return sessions


def merge_turns(msgs, is_me, label):
    """Consecutive messages from the same sender -> one turn (texts joined by a newline)."""
    turns, last_sender = [], None
    for m in msgs:
        if turns and m.sender == last_sender:
            turns[-1].text += "\n" + m.text
        else:
            turns.append(Turn("me" if is_me(m.sender) else "other", label(m.sender), m.text, m.ts))
        last_sender = m.sender
    return turns


def render(turn, group):
    """In a group chat the reader must know who is speaking, so other people's turns get a "Person_02: " prefix."""
    return "%s: %s" % (turn.speaker, turn.text) if (group and turn.who == "other") else turn.text


def _msg(turn, group, train):
    return {"role": "assistant" if turn.who == "me" else "user", "content": render(turn, group), "train": train}


def window_examples(turns, group, context, max_chars, system=None, learn_others=False):
    """One example per turn of yours: up to `context` previous turns as context (no loss), then your turn (loss).
    With learn_others=True there is one example for EVERY turn (other people's too): the last turn carries the loss, whoever wrote it.
    Returns (examples, skipped_too_long). The oldest context turns are dropped first when the example would exceed max_chars."""
    out, skipped = [], 0
    for i, t in enumerate(turns):
        if t.who != "me" and not learn_others:
            continue
        target = _msg(t, group, True)
        if len(target["content"]) > max_chars:
            skipped += 1                                          # truncating a reply would teach the wrong thing
            continue
        ctx = [_msg(c, group, False) for c in turns[max(0, i - context):i]]
        budget = max_chars - len(target["content"])
        while ctx and sum(len(c["content"]) + 8 for c in ctx) > budget:
            ctx.pop(0)
        msgs = ([{"role": "system", "content": system, "train": False}] if system else []) + ctx + [target]
        out.append(msgs)
    return out, skipped


def session_examples(turns, group, max_chars, system=None, learn_others=False):
    """Whole stretches of conversation (cut at turn boundaries to fit max_chars), loss on ALL of your turns in them
    (on every turn, other people's included, when learn_others=True).
    Cheaper than window mode (every turn appears once) but the amount of context you get varies along the example."""
    chunks, cur, size = [], [], 0
    for t in turns:
        n = len(render(t, group)) + 8
        if cur and size + n > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += n
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        if any(t.who == "me" for t in c):
            out.append(([{"role": "system", "content": system, "train": False}] if system else []) + [_msg(t, group, learn_others or t.who == "me") for t in c])
    return out


# ------------------------------------------------------------------ reporting
def size_verdict(my_tokens, n_params):
    """Plain-language answer to "is my data enough?" (rough: tokens are estimated from characters)."""
    ideal = 20 * n_params           # rule of thumb for training a small model well from scratch: ~20 training tokens per parameter
    ratio = my_tokens / ideal
    if my_tokens < 100_000:
        level = "VERY SMALL"
        text = ("Your own messages are far too little to teach a model a language or a personality on their own. A model trained only on them would "
                "mostly memorise phrases. Plan: pretrain on public text first, then fine-tune on your messages, and let the memory graph and the "
                "Person card (later stages) carry most of 'knowing you'.")
    elif my_tokens < 1_000_000:
        level = "SMALL"
        text = ("Enough to copy your STYLE (length, slang, emoji habits) after pretraining, not enough to learn language from scratch. "
                "Plan: pretrain on public text, then fine-tune on your messages for a few epochs and watch the validation loss: it will start rising "
                "when the model begins memorising.")
    else:
        level = "MODERATE"
        text = ("Useful for fine-tuning and for style, still far below what a from-scratch model needs. Plan: pretrain on public text first, "
                "then fine-tune on your messages.")
    return level, ("%s: about %.2fM tokens of YOUR text. A %.1fM-parameter model would want roughly %.0fM tokens to train well from scratch, so your "
                   "messages alone are about %.1f%% of that. %s" % (level, my_tokens / 1e6, n_params / 1e6, ideal / 1e6, 100 * ratio, text))


# ------------------------------------------------------------------ 3-8. the whole import
def import_export(export_dir, out_dir, me_names, chats="all", mode="window", context=8, gap_hours=6.0, max_chars=1400, val_fraction=0.05,
                  system=None, anonymize=True, extra_names=(), aliases=None, paranoid=False, min_my_messages=20, include_groups=True,
                  max_participants=12, seed=1337, log=print, threads=None, learn_others=False):
    """Runs the whole pipeline and writes out_dir/{train.jsonl, val.jsonl, stats.json, private/}. Returns the stats dict."""
    if mode not in ("window", "session"):
        raise ValueError("mode must be window or session")
    threads = threads if threads is not None else load_table(export_dir, log)
    me_set = {n.casefold() for n in me_names}
    is_me = lambda sender: sender.casefold() in me_set
    wanted = set(range(1, len(threads) + 1)) if chats in (None, "all") else set(chats)

    # -- which chats count
    chosen, skipped = [], collections.Counter()
    for t in threads:
        mine = sum(1 for m in t.messages if is_me(m.sender))
        if t.index not in wanted:
            skipped["not selected"] += 1
        elif mine == 0 and not learn_others:
            skipped["you never wrote in it"] += 1
        elif mine < min_my_messages:
            skipped["fewer than --min-my-messages of yours"] += 1
        elif t.is_group and not include_groups:
            skipped["group chat (excluded)"] += 1
        elif max_participants and len(t.participants) > max_participants:
            skipped["more than --max-participants people"] += 1
        else:
            chosen.append(t)
    if not chosen:
        raise SystemExit("no chats left after filtering (%s). Check --me spelling (it must match the sender name exactly), --chats and --min-my-messages."
                         % dict(skipped))
    log("importing %d chats (skipped: %s)" % (len(chosen), dict(skipped) or "none"))

    # -- the anonymizer knows every name it will meet
    everyone = set()
    for t in chosen:
        everyone.update(t.participants)
        everyone.update(m.sender for m in t.messages)
    everyone = {n for n in everyone if n and n.casefold() not in me_set}
    anon = Anonymizer(me_names=me_names, people=everyone, extra_names=extra_names, aliases=aliases, enabled=anonymize, paranoid=paranoid)

    # -- conversation -> examples
    rng = random.Random(seed)
    per_session = []                                             # (start_ts, chat_id, session_no, [examples])
    st = collections.Counter()
    first_ts, last_ts = None, None
    my_chars = other_chars = 0
    for t in chosen:
        cid = "chat_%03d" % t.index
        prepared = []
        for m in t.messages:
            text = anon.scrub(m.text)
            if text is None or not text.strip():
                st["messages dropped by anonymizer"] += 1
                continue
            prepared.append(Msg(m.sender, m.ts, text))
        for k, sess in enumerate(split_sessions(prepared, gap_hours), 1):
            turns = merge_turns(sess, is_me, anon.label)
            if not any(x.who == "me" for x in turns) and not learn_others:
                continue                                         # nothing of yours to learn from (with learn_others, other people's turns count)
            st["sessions"] += 1
            st["my_messages"] += sum(1 for x in sess if is_me(x.sender))
            st["my_turns"] += sum(1 for x in turns if x.who == "me")
            my_chars += sum(len(x.text) for x in turns if x.who == "me")
            other_chars += sum(len(x.text) for x in turns if x.who == "other")
            first_ts = sess[0].ts if first_ts is None else min(first_ts, sess[0].ts)
            last_ts = sess[-1].ts if last_ts is None else max(last_ts, sess[-1].ts)
            if mode == "window":
                exs, long_skipped = window_examples(turns, t.is_group, context, max_chars, system, learn_others)
                st["turns skipped (longer than --max-chars)"] += long_skipped
            else:
                exs = session_examples(turns, t.is_group, max_chars, system, learn_others)
            per_session.append((sess[0].ts, cid, k, [{"chat": cid, "session": k, "messages": e} for e in exs]))

    # -- split by time: whole sessions, the most recent ones are validation
    per_session.sort(key=lambda x: x[0])
    n_val = max(1, math.ceil(len(per_session) * val_fraction)) if len(per_session) >= 20 and val_fraction > 0 else 0
    val_part = per_session[len(per_session) - n_val:] if n_val else []
    train_part = per_session[:len(per_session) - n_val]
    train_ex = [e for _, _, _, exs in train_part for e in exs]
    val_ex = [e for _, _, _, exs in val_part for e in exs]
    rng.shuffle(train_ex)

    os.makedirs(os.path.join(out_dir, "private"), exist_ok=True)
    for name, exs in (("train.jsonl", train_ex), ("val.jsonl", val_ex)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # -- the second look: did anything slip through?
    leaks, suspicious = collections.Counter(), collections.Counter()
    samples = []
    if anonymize:
        for e in train_ex + val_ex:
            for m in e["messages"]:
                lk, sp = anon.audit(m["content"])
                for kind, snip in lk:
                    leaks[kind] += 1
                    samples.append("LEAK %s in %s: %r" % (kind, e["chat"], snip))
                for kind, snip in sp:
                    suspicious[kind] += 1
                    if len(samples) < 200:
                        samples.append("look at (%s) in %s: %r" % (kind, e["chat"], snip))

    # -- private files (real names!): never upload these
    with open(os.path.join(out_dir, "private", "name_map.json"), "w", encoding="utf-8") as f:
        json.dump({"me": list(me_names), "people": anon.real_to_standin, "ambiguous_first_names": [list(a) for a in anon.ambiguous],
                   "chats": {"chat_%03d" % t.index: {"title": t.title, "folder": t.folder, "kind": t.kind} for t in chosen}}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "private", "audit.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(samples) + ("\n" if samples else ""))
    with open(os.path.join(out_dir, "private", "DO_NOT_UPLOAD.txt"), "w", encoding="utf-8") as f:
        f.write("This folder contains real names and the map between real and masked names. Never upload it, never share it.\n"
                "Only train.jsonl / val.jsonl (and the token files made from them) are meant to leave your computer, and only if stats.json says safe_to_upload.\n")

    from .model import preset
    n_params = preset("s10m", 8000).n_params()
    my_tokens = int(my_chars / 3.5)
    all_tokens = int((my_chars + other_chars) / 3.5)
    level, verdict = size_verdict(all_tokens if learn_others else my_tokens, n_params)
    if learn_others:                                              # the AI learns from everyone's text, so count all of it
        verdict = verdict.replace("tokens of YOUR text", "tokens of text (yours and other people's)").replace("Your own messages", "These messages") \
                         .replace("your messages alone", "these messages alone").replace("A model trained only on them", "A model trained only on them")
    all_text = [m["content"] for e in train_ex + val_ex for m in e["messages"]]
    stats = {
        "chats_imported": len(chosen), "chats_skipped": dict(skipped),
        "dropped_message_kinds": dict(sum((t.dropped for t in chosen), collections.Counter())),
        "sessions": st["sessions"], "my_messages": st["my_messages"], "my_turns": st["my_turns"],
        "my_characters": my_chars, "other_characters": other_chars, "estimated_tokens_of_my_text": my_tokens, "estimated_tokens_of_all_text": all_tokens,
        "examples_train": len(train_ex), "examples_val": len(val_ex), "val_sessions": n_val,
        "period": [_date(first_ts), _date(last_ts)] if first_ts else None,
        "mode": mode, "learn_from_others": bool(learn_others), "context_turns": context, "session_gap_hours": gap_hours, "max_chars": max_chars,
        "other_counters": {k: v for k, v in st.items() if k not in ("sessions", "my_messages", "my_turns")},
        "anonymized": anonymize, "paranoid": paranoid, "anonymizer_counts": dict(anon.counts),
        "audit": {"leaks": dict(leaks), "suspicious": dict(suspicious)},
        "safe_to_upload": bool(anonymize and not leaks),
        "data_size_level": level, "data_size_verdict": verdict,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    if anonymize:
        guess = capitalised_word_report(all_text, known=[], top=25)
        with open(os.path.join(out_dir, "private", "capitalised_words.txt"), "w", encoding="utf-8") as f:
            f.write("Frequent capitalised words in your messages. Look for NAMES of people who are not in the chats (they are not masked unless you list them in an "
                    "extra-names file) and re-run the import.\n" + "\n".join("%5d  %s" % (n, w) for w, n in guess) + "\n")
    return stats
