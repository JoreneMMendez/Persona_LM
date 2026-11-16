"""`python -m persona_lm chat-probe`: does a chat model answer the INPUT, or does it give the same stock reply to everything?

  python -m persona_lm chat-probe --ckpt work3\\run_chat2\\ckpt.npz
  python -m persona_lm chat-probe --ckpt work3\\run_chat2\\ckpt.npz work3\\run_chat3\\ckpt.npz --seeds 2     # compare runs
  python -m persona_lm chat-probe --ckpt work3\\run_chat2\\ckpt.npz --show                                     # also print the replies

It sends about 20 plain test messages (English, Tagalog, Bisaya, Taglish; none of them from your chats) to the model and counts:
how many different replies came back, how often the most common reply came back, and how often the reply just repeats the message.
By default it prints NUMBERS ONLY, because a model can repeat text it memorized from your chats; `--show` prints the replies on your screen.
It reads the checkpoint and changes nothing.
"""
import collections
import os
import re
from types import SimpleNamespace

from .bpe import END, ROLE_ID
from .doctor import OK, PROBLEM, WARN

PROMPTS = [
    "Hello!", "Hi, how are you?", "What are you doing now?", "Where are you going?", "How was your day?",
    "What is your name?", "Did you eat already?", "Why are you laughing?", "Can you help me?", "See you tomorrow, okay?",
    "Kumusta ka?", "Nakakain ka na ba?", "Anong ginagawa mo ngayon?", "Saan ka pupunta?", "Pwede ba tayong magkita bukas?",
    "Unsa imong ginabuhat?", "Asa ka karon?", "Nakakaon na ka?", "Ngano wala ka nag-reply?", "Kumusta ang imong adlaw?",
]


def normalize(text):
    """Lower case, letters and digits only, single spaces: so 'Ok!' and 'ok' count as the same reply."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def is_echo(prompt, reply):
    """The reply is the message again (or a piece of it, or the message plus a little)."""
    p, r = normalize(prompt), normalize(reply)
    if not r:
        return False
    return r == p or (len(r) >= 4 and r in p) or (len(p) >= 4 and p in r)


def analyse(pairs):
    """pairs = [(message, reply)]. Returns the counts and a verdict (level, plain words)."""
    n = len(pairs)
    norm = [normalize(r) for _, r in pairs]
    empty = sum(1 for x in norm if not x)
    counts = collections.Counter(x for x in norm if x)
    answered = n - empty
    echo = sum(1 for p, r in pairs if is_echo(p, r))
    top_text, top_count = counts.most_common(1)[0] if counts else ("", 0)
    res = {"n": n, "empty": empty, "distinct": len(counts), "distinct_ratio": len(counts) / answered if answered else 0.0,
           "top_count": top_count, "top_share": top_count / answered if answered else 0.0, "top_text": top_text, "echo": echo,
           "echo_rate": echo / n if n else 0.0, "empty_rate": empty / n if n else 0.0}
    res["level"], res["verdict"] = verdict(res)
    return res


def verdict(r):
    if r["n"] < 5:
        return WARN, "too few messages (%d) to say anything; use at least 10" % r["n"]
    if r["empty"] == r["n"]:
        return PROBLEM, "every reply was empty: the model ends the reply at once (wrong checkpoint, a base run, or a different tokenizer?)"
    if r["top_count"] >= 3 and r["top_share"] >= 0.30:
        return PROBLEM, "STOCK REPLY: one reply came back for %.0f%% of the messages. The model ignores the message and says its favourite line (overfitting)" % (100 * r["top_share"])
    if r["distinct_ratio"] < 0.5:
        return PROBLEM, "only %.0f%% of the replies are different from each other: the model mostly ignores the message" % (100 * r["distinct_ratio"])
    why = []
    if r["top_count"] >= 3 and r["top_share"] >= 0.15:
        why.append("one reply came back %d times" % r["top_count"])
    if r["distinct_ratio"] < 0.75:
        why.append("only %.0f%% of the replies are different" % (100 * r["distinct_ratio"]))
    if r["echo_rate"] >= 0.30:
        why.append("%.0f%% of the replies just repeat the message" % (100 * r["echo_rate"]))
    if r["empty_rate"] >= 0.20:
        why.append("%.0f%% of the replies are empty" % (100 * r["empty_rate"]))
    if why:
        return WARN, "; ".join(why) + ". A little of this is normal for a small model, but it is the start of the stock-reply problem"
    return OK, "the replies are varied and they are not just the message repeated (this does not say the answers make sense; read some with --show)"


def run_probe(generate, tok, prompts, seeds=1, system=None):
    """generate(ids, seed) -> new token ids. Returns [(message, reply_text)]."""
    pairs = []
    for text in prompts:
        msgs = ([("system", system)] if system else []) + [("user", text)]
        ids = []
        for role, content in msgs:
            ids += [ROLE_ID[role]] + tok.encode(content, allow_special=False) + [END]
        ids += [ROLE_ID["assistant"]]
        for seed in range(seeds):
            pairs.append((text, tok.decode(generate(ids, seed)).strip()))
    return pairs


def format_report(name, res, pairs=None, show=False):
    lines = ["%s" % name,
             "  messages sent: %d   different replies: %d (%.0f%%)   empty: %d" % (res["n"], res["distinct"], 100 * res["distinct_ratio"], res["empty"]),
             "  most common reply: %d time(s) = %.0f%%   replies that repeat the message: %d (%.0f%%)" % (res["top_count"], 100 * res["top_share"], res["echo"], 100 * res["echo_rate"]),
             "  [%-7s] %s" % (res["level"], res["verdict"])]
    if show and pairs:
        lines.append("  the replies (private: this is on your screen only):")
        for p, r in pairs:
            lines.append("    %s  ->  %s" % (p, r if r else "(empty)"))
    elif res["top_count"] >= 2:
        lines.append("  (add --show to read the replies on your screen; they are not printed by default because a model can repeat text from your chats)")
    return lines


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        out = [ln.strip() for ln in f if ln.strip()]
    if not out:
        raise SystemExit("error: %s has no lines. Put one test message per line." % path)
    return out


def add_arguments(s):
    s.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints (a run folder also works); several are compared")
    s.add_argument("--tokenizer", default=None, help="the tokenizer the model was trained with (default: work\\tok.json, else tok.json)")
    s.add_argument("--prompts-file", default=None, help="your own test messages, one per line (default: 20 built-in plain messages)")
    s.add_argument("--seeds", type=int, default=1, help="replies per message with different random seeds (default 1)")
    s.add_argument("--system", default=None, help="a system line, as in chat")
    s.add_argument("--max-new", type=int, default=40)
    s.add_argument("--temperature", type=float, default=0.5)
    s.add_argument("--top-k", type=int, default=20)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--rep-penalty", type=float, default=1.0)
    s.add_argument("--stop-repeat", action="store_true")
    s.add_argument("--show", action="store_true", help="also print every reply (on your screen only)")


def cmd(a, say=print):
    import sys
    from . import cli
    from .fastgen import sample_ids_cached
    from .tokcheck import find_tokenizer
    tokpath = find_tokenizer(a.tokenizer)
    if not tokpath:
        sys.exit("error: no tokenizer found (looked for work\\tok.json and tok.json). Give it with --tokenizer.")
    prompts = load_prompts(a.prompts_file) if a.prompts_file else PROMPTS
    rows, worst = [], OK
    for path in a.ckpt:
        if os.path.isdir(path):
            path = os.path.join(path, "ckpt.npz")
        params, cfg, tok = cli._load_model(SimpleNamespace(ckpt=path, tokenizer=tokpath), chat=True)
        extra = cli._extra_sampling(a)
        gen = lambda ids, seed: sample_ids_cached(params, cfg, ids, max_new=a.max_new, temperature=a.temperature, top_k=a.top_k, seed=seed, **extra)
        pairs = run_probe(gen, tok, prompts, a.seeds, a.system)
        res = analyse(pairs)
        for line in format_report(path, res, pairs, a.show):
            say(line)
        rows.append((path, res))
        if res["level"] == PROBLEM:
            worst = PROBLEM
    if len(rows) > 1:
        say("")
        say("%-40s %10s %10s %8s  %s" % ("checkpoint", "different", "top reply", "echo", "verdict"))
        for path, r in rows:
            say("%-40s %9.0f%% %9.0f%% %7.0f%%  %s" % (path[-40:], 100 * r["distinct_ratio"], 100 * r["top_share"], 100 * r["echo_rate"], r["level"]))
    if worst == PROBLEM:
        raise SystemExit(1)
