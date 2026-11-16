"""The guided pipeline (Stage 3): tokenizer -> pretraining data -> pretrain -> chat data -> fine-tune -> chat.

It does not add any new maths. It writes down the owner's decisions in `project.json`, turns them into the exact commands
of the existing CLI, runs ONE step at a time (training takes hours; the owner decides when), and refuses the one step that
cannot be undone, creating the tokenizer, until the decisions are confirmed.

Why the tokenizer is special: every checkpoint, every memory and every later upgrade is tied to it. A different tokenizer makes a
trained model read garbage. So `pipeline run --step tokenizer` needs `pipeline confirm`, and once tok.json exists it is never overwritten.
"""
import json
import os
import subprocess
import time

from .bpe import Tokenizer
from .model import PRESETS
from .report import data_summary, read_ckpt_meta

PROJECT_FILE = "project.json"
STEP_ORDER = ["tokenizer", "pretrain-data", "pretrain", "chat-data", "finetune"]

DEFAULTS = {
    "version": 1,
    "languages": [],                 # language codes the owner wants, e.g. ["en", "tl"]
    "vocab": 16000,
    "size": "s10m",
    "ctx": 4096,
    "pretrain_text": [],             # files/folders of public text (all wanted languages) for the tokenizer and for pretraining
    "chat_train": None,              # train.jsonl from messenger-import
    "chat_val": None,                # val.jsonl from messenger-import
    "pretrain_steps": 10000,
    "finetune_epochs": 3.0,          # about; windows are random
    "finetune_lr": 1e-4,             # low on purpose: the model already knows language, we only want to bend its style
    "finetune_min_lr": 1e-5,
    "finetune_warmup": 20,
    "batch": None,                   # micro-batch (must fit in memory); None = the CLI default
    "accum": None,
    "train_args": [],                # extra flags for `train`, e.g. ["--precision", "bf16"]
    "system": None,                  # system line for the chat command
    "decisions_confirmed": False,
    "confirmed_at": None,
    "tokenizer_sha256": None,        # filled in when the tokenizer is made: later steps check it
}

# The script each language is written in: used only to warn when the text given for the tokenizer has (almost) none of it.
LANG_SCRIPT = {"ja": "Japanese kana", "zh": "Chinese characters", "ko": "Hangul", "ar": "Arabic", "fa": "Arabic", "ur": "Arabic",
               "ru": "Cyrillic", "uk": "Cyrillic", "bg": "Cyrillic", "hi": "Devanagari", "th": "Thai", "he": "Hebrew", "el": "Greek"}
LATIN_LANGS = {"en", "tl", "fil", "es", "fr", "de", "pt", "it", "nl", "id", "ms", "vi", "tr", "pl", "ceb", "sv", "da", "no", "ro", "cs"}


class ProjectError(Exception):
    """A mistake the owner can fix: shown as a plain message, not a traceback."""


def script_of(ch):
    o = ord(ch)
    if o < 0x250 or 0x1E00 <= o <= 0x1EFF:
        return "Latin"
    for lo, hi, name in ((0x370, 0x3FF, "Greek"), (0x400, 0x52F, "Cyrillic"), (0x590, 0x5FF, "Hebrew"), (0x600, 0x6FF, "Arabic"),
                         (0x900, 0x97F, "Devanagari"), (0xE00, 0xE7F, "Thai"), (0x3040, 0x30FF, "Japanese kana"),
                         (0x4E00, 0x9FFF, "Chinese characters"), (0xAC00, 0xD7AF, "Hangul"), (0x1100, 0x11FF, "Hangul")):
        if lo <= o <= hi:
            return name
    return "other"


def script_census(texts, limit_chars=3_000_000):
    """{script: share of letters} over (up to limit_chars of) the given texts."""
    counts, total = {}, 0
    for t in texts:
        for ch in t:
            if ch.isalpha():
                s = script_of(ch)
                counts[s] = counts.get(s, 0) + 1
                total += 1
        if total >= limit_chars:
            break
    return {k: v / total for k, v in counts.items()} if total else {}


def language_warnings(languages, census):
    """Plain-words warnings: a wanted script that is missing from the tokenizer text, and the Latin-script limitation."""
    out = []
    if not census:
        return ["The tokenizer text is empty."]
    latin_langs = [l for l in languages if l in LATIN_LANGS]
    for lang in languages:
        script = "Latin" if lang in LATIN_LANGS else LANG_SCRIPT.get(lang)
        if script is None:
            out.append("Language '%s': I do not know its script, so I cannot check that your text contains it." % lang)
        elif census.get(script, 0.0) < 0.01:
            out.append("Language '%s' is written in %s, but only %.1f%% of the tokenizer text is. Add text in that language BEFORE freezing the tokenizer, "
                       "or it will be split into many tiny pieces and waste the window." % (lang, script, 100 * census.get(script, 0.0)))
    if len(latin_langs) > 1:
        out.append("%s all use the Latin alphabet, so I cannot tell from the letters whether each of them is really in your text: make sure the text includes some of each." %
                   ", ".join(latin_langs))
    return out


# ---------------------------------------------------------------- project file

def project_path(workdir):
    return os.path.join(workdir, PROJECT_FILE)


def load_project(workdir):
    path = project_path(workdir)
    if not os.path.exists(path):
        raise ProjectError("No %s in %s. Create it first with `pipeline init`." % (PROJECT_FILE, workdir))
    with open(path, encoding="utf-8") as f:
        return {**DEFAULTS, **json.load(f)}


def save_project(workdir, proj):
    os.makedirs(workdir, exist_ok=True)
    tmp = project_path(workdir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(proj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, project_path(workdir))


def _p(workdir, *parts):
    return os.path.join(workdir, *parts)


def locked_fields(workdir):
    """What can no longer be changed because something already depends on it."""
    locked = set()
    if os.path.exists(_p(workdir, "tok.json")):
        locked |= {"vocab", "languages"}
    if any(os.path.exists(_p(workdir, r, "ckpt.npz")) for r in ("run_pre", "run_chat")):
        locked |= {"size", "ctx"}
    return locked


def init_project(workdir, **given):
    """Create project.json, or change some fields of an existing one. Only the given (not None) fields change."""
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ProjectError("unknown setting(s): %s" % ", ".join(sorted(unknown)))
    given = {k: v for k, v in given.items() if v is not None}
    exists = os.path.exists(project_path(workdir))
    proj = load_project(workdir) if exists else dict(DEFAULTS)
    if "size" in given and given["size"] not in PRESETS:
        raise ProjectError("size must be one of: %s" % ", ".join(sorted(PRESETS)))
    changed = {k for k, v in given.items() if proj.get(k) != v}
    blocked = changed & locked_fields(workdir)
    if blocked:
        raise ProjectError("Cannot change %s: a tokenizer or a trained model already depends on it. Changing it would make them useless. "
                           "(Start a new work folder if you really want that.)" % ", ".join(sorted(blocked)))
    proj.update(given)
    if changed & {"languages", "vocab", "ctx", "size"}:
        proj["decisions_confirmed"], proj["confirmed_at"] = False, None          # a changed decision must be confirmed again
    save_project(workdir, proj)
    return proj, sorted(changed)


def confirm_decisions(workdir):
    proj = load_project(workdir)
    if not proj["languages"]:
        raise ProjectError("No languages are recorded yet. Set them with `pipeline init --languages en,tl` (the languages you want the AI to learn).")
    if not proj["chat_train"]:
        raise ProjectError("No chat data recorded yet (--chat-train path\\to\\train.jsonl from messenger-import).")
    proj["decisions_confirmed"] = True
    proj["confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_project(workdir, proj)
    return proj


# ---------------------------------------------------------------- the steps

def _step_hp(proj):
    from .train import DEFAULT_HP
    return (proj["batch"] or DEFAULT_HP["batch"]), (proj["accum"] or DEFAULT_HP["accum"])


def _train_flags(proj):
    flags = []
    if proj["batch"]:
        flags += ["--batch", str(proj["batch"])]
    if proj["accum"]:
        flags += ["--accum", str(proj["accum"])]
    return flags + list(proj["train_args"])


def finetune_steps(proj, workdir):
    """Number of fine-tuning steps for about `finetune_epochs` passes over the chat data (None until that data exists)."""
    meta = _p(workdir, "data_chat", "meta.json")
    if not os.path.exists(meta):
        return None
    b, a = _step_hp(proj)
    tokens = data_summary(_p(workdir, "data_chat"))["train_tokens"]
    return max(50, int(round(proj["finetune_epochs"] * tokens / (b * a * proj["ctx"]))))


def _ckpt_done(run_dir, steps):
    meta = read_ckpt_meta(os.path.join(run_dir, "ckpt.npz"))
    return bool(meta and steps and meta.get("step", 0) >= steps)


def build_steps(proj, workdir):
    """The list of steps with their exact CLI arguments and whether each is done. A step with argv=None cannot run yet (see `note`)."""
    if not proj["chat_train"]:
        raise ProjectError("The project has no chat data yet: `pipeline init --chat-train PATH\\train.jsonl --chat-val PATH\\val.jsonl`.")
    tok, d_pre, d_chat = _p(workdir, "tok.json"), _p(workdir, "data_pre"), _p(workdir, "data_chat")
    r_pre, r_chat = _p(workdir, "run_pre"), _p(workdir, "run_chat")
    has_pre = bool(proj["pretrain_text"])
    size_flags = ["--size", proj["size"], "--ctx", str(proj["ctx"])]
    steps = []
    steps.append(dict(name="tokenizer", title="Learn the vocabulary (this FREEZES it: every model depends on it)",
                      argv=["tokenizer", "--data", *proj["pretrain_text"], proj["chat_train"], "--vocab", str(proj["vocab"]), "--out", tok],
                      done=os.path.exists(tok), skipped=False, note=""))
    if has_pre:
        steps.append(dict(name="pretrain-data", title="Turn the public text into token files",
                          argv=["prepare", "--kind", "pretrain", "--tokenizer", tok, "--data", *proj["pretrain_text"], "--out", d_pre],
                          done=os.path.exists(_p(d_pre, "meta.json")), skipped=False, note=""))
        steps.append(dict(name="pretrain", title="Pretrain (hours; Ctrl+C is safe, run the same command to continue)",
                          argv=["train", "--run", r_pre, "--data-dir", d_pre, *size_flags, "--steps", str(proj["pretrain_steps"]), *_train_flags(proj)],
                          done=_ckpt_done(r_pre, proj["pretrain_steps"]), skipped=False, note=""))
    else:
        for n, t in (("pretrain-data", "Turn the public text into token files"), ("pretrain", "Pretrain")):
            steps.append(dict(name=n, title=t, argv=None, done=False, skipped=True,
                              note="skipped: no public pretraining text given (--pretrain-text). Fine-tuning from scratch on your chats alone teaches the model very little."))
    chat_cmd = ["prepare", "--kind", "chat", "--tokenizer", tok, "--data", proj["chat_train"], "--out", d_chat]
    if proj["chat_val"]:
        chat_cmd += ["--val-data", proj["chat_val"]]
    steps.append(dict(name="chat-data", title="Turn your chat examples into token files", argv=chat_cmd, done=os.path.exists(_p(d_chat, "meta.json")), skipped=False, note=""))
    n_ft = finetune_steps(proj, workdir)
    ft = dict(name="finetune", title="Fine-tune on your chats (low learning rate, a few passes)", skipped=False)
    if n_ft is None:
        ft.update(argv=None, done=False, note="the number of steps is worked out from your data once the chat-data step is done")
    else:
        argv = ["train", "--run", r_chat, "--data-dir", d_chat, *size_flags, "--steps", str(n_ft), "--lr", str(proj["finetune_lr"]),
                "--min-lr", str(proj["finetune_min_lr"]), "--warmup", str(proj["finetune_warmup"]), *_train_flags(proj)]
        if has_pre:
            argv += ["--init-from", _p(r_pre, "ckpt.npz")]
        ft.update(argv=argv, done=_ckpt_done(r_chat, n_ft), note="about %.1f passes over your data = %d steps" % (proj["finetune_epochs"], n_ft))
    steps.append(ft)
    return steps


def next_step(steps):
    for s in steps:
        if not s["done"] and not s["skipped"]:
            return s
    return None


def chat_command(proj, workdir):
    argv = ["chat", "--ckpt", _p(workdir, "run_chat", "ckpt.npz"), "--tokenizer", _p(workdir, "tok.json")]
    if proj["system"]:
        argv += ["--system", proj["system"]]
    return argv


def show(argv):
    return "python -m persona_lm " + subprocess.list2cmdline(argv)


def plan_lines(workdir):
    proj = load_project(workdir)
    steps = build_steps(proj, workdir)
    lines = ["Project: languages %s, vocabulary %d, window %d tokens, model %s. Decisions confirmed: %s" % (
        ",".join(proj["languages"]) or "(none yet)", proj["vocab"], proj["ctx"], proj["size"],
        ("yes (%s)" % proj["confirmed_at"]) if proj["decisions_confirmed"] else "NO (run `pipeline confirm` after you have said OK to them)")]
    nxt = next_step(steps)
    for i, s in enumerate(steps, 1):
        mark = "done" if s["done"] else "skipped" if s["skipped"] else "NEXT" if s is nxt else "todo"
        lines.append("")
        lines.append("%d. [%s] %s" % (i, mark, s["title"]))
        if s["note"]:
            lines.append("   " + s["note"])
        if s["argv"]:
            lines.append("   " + show(s["argv"]))
    lines.append("")
    lines.append("6. [any time after fine-tuning] Talk to it")
    lines.append("   " + show(chat_command(proj, workdir)))
    if not os.path.exists(_p(workdir, "tok.json")):
        texts = _tokenizer_texts(proj)
        for w in language_warnings(proj["languages"], script_census(texts)):
            lines.append("WARNING: " + w)
    return lines


def _tokenizer_texts(proj):
    from .cli import _texts_for_tokenizer
    return _texts_for_tokenizer([*proj["pretrain_text"], proj["chat_train"]], 3)


def run_step(workdir, name, main, log=print):
    """Run one step through the normal CLI entry point `main(argv)`. name = a step name or 'next'."""
    proj = load_project(workdir)
    steps = build_steps(proj, workdir)
    step = next_step(steps) if name == "next" else next((s for s in steps if s["name"] == name), None)
    if step is None:
        if name == "next":
            log("Every step is done. Talk to it with:\n  " + show(chat_command(proj, workdir)))
            return None
        raise ProjectError("unknown step '%s' (choose from: %s, next)" % (name, ", ".join(STEP_ORDER)))
    if step["skipped"]:
        raise ProjectError(step["note"])
    if step["argv"] is None:
        raise ProjectError("This step is not ready yet: " + step["note"])
    tok_path = _p(workdir, "tok.json")
    if step["name"] == "tokenizer":
        if os.path.exists(tok_path):
            raise ProjectError("tok.json already exists and is frozen. Not overwritten: every model made with it would break.")
        if not proj["decisions_confirmed"]:
            raise ProjectError("The tokenizer cannot be made before you confirm the decisions (languages, vocabulary, window). "
                               "Read `pipeline plan`, then say OK with: pipeline confirm --dir ... --yes")
    else:
        if not os.path.exists(tok_path):
            raise ProjectError("There is no tokenizer yet: run the tokenizer step first.")
        if proj["tokenizer_sha256"] and Tokenizer.load(tok_path).fingerprint() != proj["tokenizer_sha256"]:
            raise ProjectError("tok.json is not the tokenizer this project froze (its fingerprint changed). Restore the original file.")
        if not proj["tokenizer_sha256"]:                          # a tokenizer made by hand: remember it from now on
            proj["tokenizer_sha256"] = Tokenizer.load(tok_path).fingerprint()
            save_project(workdir, proj)
    log("Running step '%s':\n  %s\n" % (step["name"], show(step["argv"])))
    main(step["argv"])
    if step["name"] == "tokenizer" and os.path.exists(tok_path):
        proj["tokenizer_sha256"] = Tokenizer.load(tok_path).fingerprint()
        save_project(workdir, proj)
        log("Tokenizer frozen (fingerprint %s...). Never replace tok.json." % proj["tokenizer_sha256"][:12])
    after = next_step(build_steps(load_project(workdir), workdir))
    if after is not None and after["name"] == step["name"]:
        log("\nStep '%s' is not finished (stopped early?). Run the same command again to continue exactly where it stopped." % step["name"])
    else:
        log("\nStep '%s' finished." % step["name"] + (("  Next: pipeline run --dir \"%s\"" % workdir) if after else "  All steps done."))
    return step["name"]
