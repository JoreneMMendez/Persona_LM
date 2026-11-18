"""The launcher menu: `python -m persona_lm` with no arguments (or `python -m persona_lm menu`).

Three rounded boxes like the training dashboard: the big one is the list (arrow keys or number keys), the small ones show what the
highlighted item does and what the menu found in your project folder (runs, token folders, tokenizer, PyTorch).
Picking an item asks for the few things it needs, with the answers it found already filled in (press Enter to accept), prints the
exact command it is about to run, and runs it. Nothing here does anything that the normal commands cannot do.

No extra packages. A console that cannot do this (output to a file, very small window) gets a plain numbered list instead.
"""
import contextlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap

from .pretty import MAX_W, MIN_WIDTH, Skin, enable_vt, fit
from .model import PRESETS
from .train import DEFAULT_HP
from .menu_text import (CONFIRM, DETAIL_FIELDS, FIELD_TEXT, FORM, GROUPS, ITEM_TEXT, NEXT, PRESET_TEXT, PROBLEMS, SCREEN,
                        SETTING_TEXT, SETTINGS_UI, TECH_NAMES)

SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "site-packages", "private", "tests", "docs"}
SIZES = sorted(PRESETS, key=lambda k: PRESETS[k]["d_model"])


# ------------------------------------------------------------------ what is in the project folder
def run_meta(ckpt_path):
    """The small meta block of a checkpoint (settings, model size, step); the weights are not loaded. None if unreadable."""
    try:
        import numpy as np
        with np.load(ckpt_path) as z:
            return json.loads(bytes(z["meta"].tolist()).decode("utf-8"))
    except Exception:
        return None


def run_info(ckpt_path):
    """(size name or '?', step, total steps) read from a checkpoint's small meta block; the weights are not loaded."""
    try:
        import numpy as np
        with np.load(ckpt_path) as z:
            meta = json.loads(bytes(z["meta"].tolist()).decode("utf-8"))
        cfg = meta["cfg"]
        size = next((k for k, v in PRESETS.items() if all(cfg.get(f) == v[f] for f in ("n_layer", "d_model", "n_head", "d_ff"))), "?")
        return size, int(meta.get("step", 0)), int((meta.get("hp") or {}).get("steps", 0))
    except Exception:
        return "?", 0, 0


def _is_tokenizer(path):
    """True only for a real tokenizer file (has "merges" and "specials"); other tok*.json files (e.g. tok.seal.json) are ignored."""
    try:
        if os.path.getsize(path) > 20_000_000:
            return False
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        return isinstance(j, dict) and "merges" in j and "specials" in j
    except (OSError, ValueError):
        return False


def scan(root, depth=3):
    """Finds runs (folders with ckpt.npz), token folders (meta.json with vocab_size) and tokenizers (tok*.json) up to `depth` folders down."""
    found = {"runs": [], "data": [], "toks": [], "torch": importlib.util.find_spec("torch") is not None}
    root = os.path.abspath(root)
    for d, dirs, files in os.walk(root):
        rel = os.path.relpath(d, root)
        if rel != "." and rel.count(os.sep) + 1 >= depth:
            dirs[:] = []
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        shown = "." if rel == "." else rel
        try:
            if "ckpt.npz" in files:
                p = os.path.join(d, "ckpt.npz")
                found["runs"].append((os.path.getmtime(p), shown, p))
            if "meta.json" in files:
                with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                    if "vocab_size" in json.load(f):
                        found["data"].append((os.path.getmtime(os.path.join(d, "meta.json")), shown))
            for fn in files:
                if fn.lower().startswith("tok") and fn.lower().endswith(".json") and _is_tokenizer(os.path.join(d, fn)):
                    # tok.json (the frozen tokenizer the models were trained with) always comes first; other valid ones by date
                    found["toks"].append((fn.lower() == "tok.json", os.path.getmtime(os.path.join(d, fn)), os.path.normpath(os.path.join(shown, fn))))
        except (OSError, ValueError):
            continue
    found["runs"].sort(reverse=True)
    found["data"].sort(reverse=True)
    found["toks"].sort(reverse=True)
    found["runs"] = [dict(path=rel, ckpt=p, size=None, step=None, steps=None) for _, rel, p in found["runs"]]
    for r in found["runs"][:1]:                          # only the newest is opened (the others when asked), so the menu opens fast
        r["size"], r["step"], r["steps"] = run_info(r["ckpt"])
    found["data"] = [p for _, p in found["data"]]
    found["toks"] = [p for _, _, p in found["toks"]]
    return found


def _find_run(found, answer):
    a = os.path.normpath(answer or "")
    for r in found["runs"]:
        if os.path.normpath(r["path"]) == a:
            if r["size"] is None:
                r["size"], r["step"], r["steps"] = run_info(r["ckpt"])
            return r
    return None


# ------------------------------------------------------------------ the items and what each one asks
def _field(key, ask, flag=None, default=None, required=False, choices=None, hint=None, check=None, switch=False, help=None):
    """check(value) -> an error text or None (checked before the answer is accepted); switch=True: the flag takes no value (answer yes -> flag added)."""
    return dict(key=key, ask=ask, flag=flag, default=default or (lambda a, f: ""), required=required, choices=choices, hint=hint, check=check, switch=switch, help=help)


def _d_run(a, f):
    return f["runs"][0]["path"] if f["runs"] else ""


def _d_ckpt(a, f):
    return os.path.join(f["runs"][0]["path"], "ckpt.npz") if f["runs"] else ""


def _check_recipe_id(val):
    """Error text if the typed recipe id cannot be a recipe id, else None."""
    from .hfdata import ID_RE
    return None if ID_RE.match(val) else PROBLEMS["recipe_id"]


def _d_chat_ckpt(a, f):
    """For chatting: the newest run whose folder name has 'chat' (chat-tuned); a base run (run_pre) only when there is nothing else."""
    for r in f["runs"]:
        if "chat" in os.path.basename(os.path.normpath(r["path"])).lower():
            return os.path.join(r["path"], "ckpt.npz")
    return _d_ckpt(a, f)


def _is_base_name(path):
    name = os.path.basename(os.path.normpath(path)).lower()
    return "pre" in name and "chat" not in name


def _d_ft_init(a, f):
    """Fine-tuning starts from a pretrained BASE model: the newest run whose folder name has 'pre' (not 'chat')."""
    for r in f["runs"]:
        if _is_base_name(r["path"]):
            return os.path.join(r["path"], "ckpt.npz")
    return _d_ckpt(a, f)


def _d_ft_run(a, f):
    """A NEW folder next to the base run: run_chat<n>, one more than the highest run_chat<n> already there."""
    init = a.get("init") or ""
    folder = os.path.dirname(os.path.dirname(init)) if init else ""
    scan_dir = folder or "."
    top = 0
    try:
        for name in os.listdir(scan_dir):
            m = re.fullmatch(r"run_chat(\d+)", name.lower())
            if m:
                top = max(top, int(m.group(1)))
    except OSError:
        pass
    return os.path.join(folder, "run_chat%d" % (top + 1)) if folder else "run_chat%d" % (top + 1)


def _d_ft_data(a, f):
    """The newest token folder whose name says chat; otherwise the newest token folder."""
    for d in f["data"]:
        if "chat" in os.path.basename(os.path.normpath(d)).lower():
            return d
    return _d_data(a, f)


def _d_ft_size(a, f):
    init = a.get("init") or ""
    r = _find_run(f, os.path.dirname(init)) if init else None
    return r["size"] if r and r["size"] not in (None, "?") else _d_size({}, f)


def _check_base_ckpt(val):
    if os.path.isfile(val) or os.path.isfile(val.replace(".npz", ".prev.npz")):
        return None
    return PROBLEMS["base_missing"]


def _check_new_run(val):
    if os.path.isfile(os.path.join(val, "ckpt.npz")):
        return PROBLEMS["run_exists"].format(name=_d_ft_run({"init": os.path.join(val, "ckpt.npz")}, {}))
    return None


def _d_data(a, f):
    return f["data"][0] if f["data"] else ""


def _d_tok(a, f):
    return f["toks"][0] if f["toks"] else ""


def _d_size(a, f):
    r = _find_run(f, a.get("run")) if a.get("run") else (f["runs"][0] if f["runs"] else None)
    return r["size"] if r and r["size"] not in (None, "?") else "s10m"


def _langs_default(a, f):
    return "en,tl,ceb,taglish"


def _check_langs(val):
    """Error text if the typed language list is not a comma list of en / tl / ceb / taglish, else None."""
    from .corpus import CorpusError, check_langs
    try:
        check_langs([x.strip().lower() for x in val.split(",") if x.strip()])
    except CorpusError:
        return PROBLEMS["langs"]
    return None


HELP = {"lang": "which public text to get, comma separated: en (English), tl (Tagalog), ceb (Cebuano), taglish. Enter = all four",
        "mode": "all = download, then clean and check; clean = only clean what is already in public_text\\",
        "mix": "yes = also rebuild public_mix\\ (the pretraining mix) from the cleaned text",
        "run": "where this run's checkpoints are saved (an existing folder = resume it)",
        "data": "the folder of token files made by `prepare`",
        "size": "model size; must match the run you are continuing",
        "init": "only for a NEW run: copy the weights of another model (e.g. runs\\pre\\ckpt.npz) - this is how chat fine-tuning starts. Leave empty to continue or start fresh",
        "ckpt": "the model file to load (ckpt.npz)",
        "tok": "the tokenizer the model was trained with",
        "system": "optional: a role to start the chat with, e.g. You are Aria.",
        "prompt": "the text for the model to continue",
        "dir": "the folder to look at",
        "hf_action": "new = a wizard that makes a recipe; list; show; peek = look at the first rows; run = stream + clean + write the output; assemble = build the output from what is saved; tokens = make token files; colab-kit = a zip for Google Drive",
        "hf_recipe": "the recipe id (a file in dataset_recipes\\). Leave empty for new, or when you have only one recipe"}

def field_text(item, f):
    # plain-words record of one question; the technical text is only a fallback for a field without a record
    rec = FIELD_TEXT.get(item["label"], {}).get(f["key"])
    if rec:
        return rec
    h = f.get("help") or HELP.get(f["key"], "")
    return dict(label=f["ask"], help=h, more=h, example="", default="")


def enter_note(rec, f, value):
    # what pressing Enter does, in words
    if value:
        return FORM["enter"].format(note=(rec.get("default") or "the answer shown: {value}").replace("{value}", value))
    return FORM["enter_none"] if f["required"] else FORM["enter_skip"]


# ------------------------------------------------------------------ the training settings screen
SETTINGS = [dict(key=k, kind=kind, lo=lo, hi=hi) for k, kind, lo, hi in (
    ("steps", int, 1, 10 ** 7), ("lr", float, 1e-6, 1e-1), ("min_lr", float, 0.0, 1e-1), ("warmup", int, 0, 100000),
    ("weight_decay", float, 0.0, 1.0), ("clip", float, 0.0, 100.0), ("batch", int, 1, 512), ("accum", int, 1, 1024),
    ("eval_every", int, 1, 100000))]
for _d in SETTINGS:
    _d.update(SETTING_TEXT[_d["key"]])                   # name, short, what, up, down, affects, tip (plain words, in menu_text.py)

# the numbers behind the ready-made choices of menu_text.PRESET_TEXT (same order)
PRESET_VALUES = (dict(batch=1, accum=4, eval_every=100),
                 dict(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000),
                 dict(weight_decay=0.2, lr=3e-4, min_lr=3e-5, eval_every=100))
QUICK = [(name, vals, why) for (name, why), vals in zip(PRESET_TEXT, PRESET_VALUES)]


def _fmt(v):
    return format(v, ",") if isinstance(v, int) else "%g" % v


def _parse(text, kind):
    return kind(float(text.replace(",", "").replace("_", ""))) if kind is int else float(text.replace(",", ""))


def _train_tokens(answers):
    # tokens in the training file of the chosen prepared text folder; only the file SIZE is looked at (2 bytes per token)
    try:
        return os.path.getsize(os.path.join(answers["data"], "train.bin")) // 2 if answers.get("data") else None
    except OSError:
        return None


def edit_settings(base, ctx, step, ask=input, say=print, start=None, train_tokens=None, width=None):
    """Shows the training settings in plain words and lets the person change any of them before the run starts.
    `base` = the values the run uses now (saved in the run, or the defaults for a new one); `ctx` = tokens per sequence; `step` = where
    a resumed run is (0 for a new one); `train_tokens` = size of the prepared text, if known (for the passes warning).
    Returns only the settings that were CHANGED, as {name: value}."""
    U = SETTINGS_UI
    w = min((width or shutil.get_terminal_size((80, 24)).columns) - 1, 79)
    out = lambda text, ind="  ", sub="  ": [say(l) for l in textwrap.wrap(text, w, initial_indent=ind, subsequent_indent=sub)] or say("")
    begin = dict(base, **(start or {}))                  # a new fine-tune begins on the gentle chat-tuning numbers
    cur = dict(begin)
    n = len(SETTINGS)
    say("\n  " + U["title"])
    out(U["intro"])
    while True:
        say("")
        for i, d in enumerate(SETTINGS, 1):
            k = d["key"]
            mark = "*" if cur[k] != base[k] else " "
            was = U["was"].format(old=_fmt(base[k])) if mark == "*" else ""
            out("%s%d  %-16s %9s   %s%s" % (mark, i, d["name"], _fmt(cur[k]), d["short"], was), "  ", " " * 33)
        per = cur["batch"] * cur["accum"] * ctx
        say("")
        out(U["per_step"].format(per=format(per, ","), steps=format(cur["steps"], ","), total=format(per * cur["steps"], ",")))
        if train_tokens:
            passes = cur["steps"] * per / train_tokens
            out(U["passes"].format(passes="%.1f" % passes) + (" " + U["passes_warn"] if passes > 4 else ""))
        else:
            out(U["no_data"])
        if step and any(cur[k] != base[k] for k in ("lr", "min_lr", "steps")):
            out(U["heads_resume"].format(step=format(step, ",")))
        if cur["batch"] * cur["accum"] != base["batch"] * base["accum"]:
            out(U["heads_reads"])
        k = ask("\n  " + U["prompt"].format(n=n)).strip().lower()
        if not k:
            break
        if k == "r":
            cur = dict(begin)
        elif k == "p":
            say("")
            names = {d["key"]: d["name"] for d in SETTINGS}
            for j, (name, vals, why) in enumerate(QUICK, 1):
                say("  %d  %s" % (j, name))
                out(why, "     ", "     ")
                out("Sets: " + ", ".join("%s = %s" % (names[a], _fmt(b)) for a, b in vals.items()), "     ", "       ")
            c = ask("\n  " + U["apply"]).strip()
            if c.isdigit() and 1 <= int(c) <= len(QUICK):
                cur.update(QUICK[int(c) - 1][1])
        elif k.isdigit() and 1 <= int(k) <= n:
            d = SETTINGS[int(k) - 1]
            say("")
            out(U["detail"].format(name=d["name"], key=d["key"], short=d["short"]))
            out(U["now"].format(now=_fmt(cur[d["key"]]), old=_fmt(base[d["key"]])))
            for label, field in U["labels"]:
                out("%s: %s" % (label, d[field]), "    ", "      ")
            raw = ask("\n  " + U["value_prompt"]).strip()
            if raw:
                try:
                    v = _parse(raw, d["kind"])
                except ValueError:
                    out(U["bad_number"])
                    continue
                key = d["key"]
                if not d["lo"] <= v <= d["hi"]:
                    out(U["out_of_range"].format(name=d["name"], lo=_fmt(d["lo"]), hi=_fmt(d["hi"])))
                elif key == "steps" and step and v <= step:
                    out(U["steps_low"].format(step=format(step, ",")))
                elif key == "lr" and v < cur["min_lr"]:
                    out(U["lr_low"].format(v=_fmt(cur["min_lr"])))
                elif key == "min_lr" and v > cur["lr"]:
                    out(U["min_high"].format(v=_fmt(cur["lr"])))
                else:
                    cur[key] = v
        else:
            out(U["junk"].format(n=n))
    return {k: cur[k] for k in cur if cur[k] != base[k]}


def settings_base(found, answers):
    """(settings in use now, tokens per sequence, current step) for the run folder / size that were typed."""
    r = _find_run(found, answers.get("run"))
    meta = run_meta(r["ckpt"]) if r else None
    hp = dict(DEFAULT_HP)
    if meta:
        hp.update(meta.get("hp") or {})
        return {d["key"]: hp[d["key"]] for d in SETTINGS}, int((meta.get("cfg") or {}).get("ctx") or 0) or PRESETS[answers["size"]]["ctx"], int(meta.get("step", 0))
    return {d["key"]: hp[d["key"]] for d in SETTINGS}, PRESETS[answers["size"]]["ctx"], 0



ITEMS = [
    dict(label="Train or resume a run", cmd="train", settings=True, desc="Start training, or continue a run exactly where it stopped. You can review and change the numbers first.",
         fields=[_field("run", "Run folder", "--run", _d_run, True), _field("data", "Token folder", "--data-dir", _d_data, True),
                 _field("size", "Model size", "--size", _d_size, True, SIZES), _field("init", "Start from weights", "--init-from")]),
    dict(label="Chat with a model", cmd="chat", desc="Talk to a trained model in this window. Starts with the newest chat-tuned run (run_chat*); a base run (run_pre) cannot chat.",
         fields=[_field("ckpt", "Checkpoint", "--ckpt", _d_chat_ckpt, True), _field("tok", "Tokenizer", "--tokenizer", _d_tok, True),
                 _field("system", "System prompt", "--system")]),
    dict(label="Fine-tune a model (chat tuning)", cmd="train", settings=True, preset=dict(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000),
         desc="Teach a pretrained base model to chat: starts a NEW run from the base model's weights on your chat token folder, with gentle settings you can review (lr 1e-4, 3000 steps).",
         fields=[_field("init", "Base model to start from", "--init-from", _d_ft_init, True, check=_check_base_ckpt,
                        help="the ckpt.npz of the pretrained BASE run (folder name like run_pre). Its weights are copied; the base run is not changed"),
                 _field("run", "New run folder", "--run", _d_ft_run, True, check=_check_new_run,
                        help="where the fine-tuned model is saved. It must be a NEW folder (run_chat1, run_chat2, ...)"),
                 _field("data", "Chat token folder", "--data-dir", _d_ft_data, True,
                        help="the token files made with `prepare --kind chat` (or Language data -> token files). Pick the chat folder, not the pretraining one"),
                 _field("size", "Model size", "--size", _d_ft_size, True, SIZES,
                        help="must be the same size as the base model")]),
    dict(label="Generate text", cmd="generate", desc="Continue a prompt with a model, once, and print it.",
         fields=[_field("ckpt", "Checkpoint", "--ckpt", _d_ckpt, True), _field("tok", "Tokenizer", "--tokenizer", _d_tok, True),
                 _field("prompt", "Prompt", "--prompt", None, True)]),
    dict(label="Project status", cmd="status", desc="What exists in this folder (tokenizers, token files, runs) and what to do next.",
         fields=[_field("dir", "Folder", None, lambda a, f: ".")]),
    dict(label="Open the Web UI", cmd="ui", desc="The local, offline web page for the whole system. Opens your browser.", fields=[]),
    dict(label="Speed benchmark", cmd="benchmark", desc="Measure tokens per second on this computer and estimate training time.",
         fields=[_field("size", "Model size", "--size", _d_size, True, SIZES)]),
    dict(label="Check the GPU", cmd="check-gpu", desc="Check the PyTorch / GPU backend against the reference and print a report.", fields=[]),
    dict(label="Dataset recipes (create / run)", cmd="hfdata",
         desc="Stream a public Hugging Face dataset with a saved recipe (filters + cleaning), then make token files. 'new' asks questions and saves a recipe.",
         fields=[_field("hf_action", "What to do", None, lambda a, f: "new", True, ("new", "list", "show", "peek", "run", "assemble", "tokens", "colab-kit")),
                 _field("hf_recipe", "Recipe id", None, None, False, check=_check_recipe_id)]),
    dict(label="Language data (download)", cmd="corpus", desc="Download public English, Tagalog, Cebuano and Taglish text, clean it and check it.",
         fields=[_field("lang", "Languages", "--lang", _langs_default, True, check=_check_langs),
                 _field("mode", "What to do", "--mode", lambda a, f: "all", True, ("all", "clean")),
                 _field("mix", "Also rebuild the training mix", "--mix", lambda a, f: "no", False, ("no", "yes"), switch=True)]),
    dict(label="Language data -> token files", cmd="build-tokens", desc="One click: turn the cleaned language data into token files in a NEW folder. Uses work\\tok.json (only read). No internet.", fields=[]),
    dict(label="Quit", cmd=None, desc="Leave the menu.", fields=[]),
]
QUIT = len(ITEMS) - 1


def plain_text(item):
    # plain-words record (title, what, when, need, result, time, safe) of a menu item
    return ITEM_TEXT[item["label"]]


# What the main screen shows: the groups of menu_text.GROUPS, then Quit. ITEMS keeps its own order; the screen maps number keys through DISPLAY.
_AT = {it["label"]: i for i, it in enumerate(ITEMS)}
LAYOUT = [(head, [_AT[x] for x in labels]) for head, labels in GROUPS] + [(None, [QUIT])]
DISPLAY = [i for _, idxs in LAYOUT for i in idxs]       # ITEMS indexes in the order shown
START = DISPLAY[0]
MENU_H, TOP_H = 27, 19                                    # lines of the whole main screen, and of its list/details row


def number_key(i):
    # the number shown next to ITEMS[i] on the screen (and typed in the numbered list)
    return str(DISPLAY.index(i) + 1)


def project_checks(found):
    # (label, value, ok) lines about what exists, in words; only scan() results are used
    S, runs = SCREEN, found["runs"]
    if runs:
        r = runs[0]
        pct = "%d%% taught" % (100 * r["step"] // r["steps"]) if r["steps"] and r["step"] is not None else ""
        info = ", ".join(x for x in (r["size"], pct) if x)
        model = os.path.basename(os.path.normpath(r["path"])) + (" (%s)" % info if info else "") + ("  +%d more" % (len(runs) - 1) if len(runs) > 1 else "")
    data = found["data"]
    return [(S["model"], model if runs else S["model_none"], bool(runs)),
            (S["data"], data[0] + ("  +%d more" % (len(data) - 1) if len(data) > 1 else "") if data else S["data_none"], bool(data)),
            (S["torch"], S["torch_yes"] if found["torch"] else S["torch_no"], bool(found["torch"]))]


def next_suggestion(found):
    # ONE plain sentence about what to do next, from scan() only
    t = lambda label: ITEM_TEXT[label]["title"]
    v = dict(teach=t("Train or resume a run"), chat=t("Chat with a model"), finetune=t("Fine-tune a model (chat tuning)"),
             download=t("Language data (download)"), prepare=t("Language data -> token files"))
    runs = found["runs"]
    if not runs:
        return NEXT["data_only" if found["data"] else "empty"].format(**v)
    r = runs[0]
    v["name"] = os.path.basename(os.path.normpath(r["path"]))
    if r["steps"] and r["step"] is not None and r["step"] < r["steps"]:
        v["pct"] = 100 * r["step"] // r["steps"]
        return NEXT["unfinished"].format(**v)
    if _is_base_name(r["path"]):
        return NEXT["base_only"].format(**v)
    return NEXT["chat_ready" if "chat" in v["name"].lower() else "other"].format(**v)


# ------------------------------------------------------------------ keys
def _plain(ch):
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x1b":
        return "esc"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch.lower()


def read_key():
    """One key press as 'up' 'down' 'home' 'end' 'enter' 'esc' or the lower-case character."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "G": "home", "O": "end"}.get(msvcrt.getwch(), "")
        return _plain(ch)
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = os.read(fd, 1).decode("latin-1")
        if ch == "\x1b":
            if select.select([fd], [], [], 0.05)[0]:
                return {"[A": "up", "[B": "down", "OA": "up", "OB": "down", "[H": "home", "[F": "end"}.get(os.read(fd, 8).decode("latin-1")[:2], "")
            return "esc"
        return _plain(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ------------------------------------------------------------------ the screen
class Menu(Skin):
    def __init__(self, stream=None, color=True, width=None, unicode_ok=None):
        Skin.__init__(self, stream or sys.stdout, color, unicode_ok)
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.n = 0

    def _write(self, text):
        self.out.write(text)
        self.out.flush()

    def _erase(self):
        if self.n:
            self._write("\x1b[%dF\x1b[J" % self.n)
            self.n = 0

    def header(self):
        self._write("\n" + self.c(" PersonaLM ", "1;36") + self.c(self.dot + " menu", "90") + "\n" + self.c(self.rule * min(self.width - 1, MAX_W - 1), "90") + "\n")

    def render(self, sel, found):
        W = min(self.width - 1, MAX_W)
        lw = max(34, W * 38 // 100)
        rw = W - lw - 1
        li, ri, g = lw - 4, rw - 4, self.c
        rows, sel_row = [], 0
        for head, idxs in LAYOUT:
            if head:
                rows.append(g(head, "1"))
            for i in idxs:
                label, n = ITEMS[i]["label"], DISPLAY.index(i) + 1
                key = "q" if i == QUIT else (str(n) if n <= 9 else " ")      # a single key press can only pick 1-9
                title, tech = ITEM_TEXT[label]["title"], TECH_NAMES.get(label)
                tail = " (%s)" % tech if tech and len(tech) + 3 <= li - 5 - len(title) else ""
                if i == sel:
                    sel_row = len(rows)
                    rows.append(g(fit("%s %s  %s%s" % (self.ptr, key, title, tail), li), "30;46"))
                else:
                    rows.append("  " + g(key, "90") + "  " + title + g(tail, "90"))
        inner = TOP_H - 2
        start = min(max(sel_row - inner // 2, 0), len(rows) - inner) if len(rows) > inner else 0
        big = rows[start:start + inner]

        label = ITEMS[sel]["label"]
        t, tech = ITEM_TEXT[label], TECH_NAMES.get(label)
        det = [g(t["title"], "1") + (g("  (%s)" % tech, "90") if tech else "")]
        for lab, key in DETAIL_FIELDS:
            words = textwrap.wrap("%s %s" % (lab, t[key]), ri, subsequent_indent="  ")
            det += [g(lab, "90") + words[0][len(lab):]] + words[1:]

        chk = [g(lab, "90") + " " + g(val, "32" if ok else "33") for lab, val, ok in project_checks(found)]
        nxt = textwrap.wrap("%s %s" % (SCREEN["next"], next_suggestion(found)), W - 4, subsequent_indent="  ")
        nxt[0] = g(SCREEN["next"], "1;36") + nxt[0][len(SCREEN["next"]):]

        left = self.box(SCREEN["menu_box"], big, lw, TOP_H)
        right = self.box(SCREEN["details_box"], det, rw, TOP_H)
        top = [a + " " + b for a, b in zip(left, right)]
        return top + self.box(SCREEN["project_box"], chk + nxt, W, MENU_H - TOP_H - 1) + [g(fit(SCREEN["hint"], W), "90")]

    def select(self, read, found, sel=None):
        """Draws the menu and returns the chosen item's index in ITEMS (QUIT for q / Esc). Keys move through the order on screen (DISPLAY)."""
        pos = DISPLAY.index(START if sel is None else sel)
        self._write("\x1b[?25l")
        try:
            while True:
                lines = self.render(DISPLAY[pos], found)
                self._write("\n".join(lines) + "\n")
                self.n = len(lines)
                k = read()
                if k == "enter" or (k.isdigit() and 1 <= int(k) <= QUIT):
                    self.n = 0                                           # leave the menu on screen: the questions appear under it
                    return DISPLAY[pos if k == "enter" else int(k) - 1]
                self._erase()
                if k in ("up", "k"):
                    pos = (pos - 1) % len(DISPLAY)
                elif k in ("down", "j"):
                    pos = (pos + 1) % len(DISPLAY)
                elif k == "home":
                    pos = 0
                elif k == "end":
                    pos = len(DISPLAY) - 1
                elif k in ("q", "esc"):
                    return QUIT
        finally:
            self._write("\x1b[?25h")


# ------------------------------------------------------------------ boxed questions (forms) and confirmations
def _edit_plain(ch):
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x1b":
        return "esc"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "\t":
        return "tab"
    if ch in ("\x08", "\x7f"):
        return "backspace"
    if ch == "\x15":
        return "clear"
    return ch if ch >= " " else ""


_ESC_KEYS = {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "OA": "up", "OB": "down", "OC": "right", "OD": "left",
             "[H": "home", "[F": "end", "OH": "home", "OF": "end", "[1": "home", "[4": "end", "[3": "delete"}


_KB = {"on": False}


@contextlib.contextmanager
def keyboard_mode():
    """Keeps the terminal in key-by-key mode for a whole screen, so fast typing or pasting is not lost between two keys (POSIX; Windows needs nothing)."""
    old = fd = None
    if os.name != "nt":
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            _KB["on"] = True
        except Exception:                                   # not a terminal: nothing to switch
            fd = None
    try:
        yield
    finally:
        _KB["on"] = False
        if fd is not None:
            try:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass


def _posix_key(fd):
    import select
    b = os.read(fd, 1)
    if b >= b"\xc0":                                          # the first byte of a multi-byte character (e.g. an accented letter): read the rest
        b += os.read(fd, 1 if b < b"\xe0" else (2 if b < b"\xf0" else 3))
    ch = b.decode("utf-8", "replace")
    if ch == "\x1b":
        seq = ""
        while len(seq) < 6 and select.select([fd], [], [], 0.03)[0]:
            seq += os.read(fd, 1).decode("latin-1")
            if len(seq) >= 2 and (seq[-1].isalpha() or seq[-1] == "~"):
                break
        return _ESC_KEYS.get(seq[:2], "") if seq else "esc"
    return _edit_plain(ch)


def read_edit_key():
    """One key press for the forms: 'up' 'down' 'left' 'right' 'home' 'end' 'delete' 'enter' 'esc' 'tab' 'backspace' 'clear' or the exact character typed."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "S": "delete"}.get(msvcrt.getwch(), "")
        return _edit_plain(ch)
    import termios
    import tty
    fd = sys.stdin.fileno()
    if _KB["on"]:
        return _posix_key(fd)
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return _posix_key(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Live(Skin):
    """A block of lines that is redrawn in place (cursor up, erase, write), like the menu and the training dashboard."""

    def __init__(self, stream=None, color=True, width=None, unicode_ok=None):
        Skin.__init__(self, stream or sys.stdout, color, unicode_ok)
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.W = min(self.width - 1, MAX_W)
        self.n = 0

    def _write(self, text):
        self.out.write(text)
        self.out.flush()

    def erase(self):
        if self.n:
            self._write("\x1b[%dF\x1b[J" % self.n)
            self.n = 0

    def draw(self, lines):
        self.erase()
        self._write("\n".join(lines) + "\n")
        self.n = len(lines)


def build_argv(item, values):
    """The normal command line for an item from the answers {key: text}; same rules as collect()."""
    argv = [item["cmd"]]
    for f in item["fields"]:
        val = values.get(f["key"], "")
        if f.get("switch"):
            if val == "yes":
                argv.append(f["flag"])
        elif val:
            argv += [f["flag"], val] if f["flag"] else [val]
    return argv


class Form(Live):
    """The questions of one menu item as a box: arrows move, typing replaces the shown answer, left/right choose, Enter goes on, the last row starts."""

    def __init__(self, item, found, stream=None, color=True, width=None, unicode_ok=None):
        Live.__init__(self, stream, color, width, unicode_ok)
        self.item, self.found, self.fields = item, found, item["fields"]
        n = len(self.fields)
        self.val, self.touched, self.fresh = [""] * n, [False] * n, [True] * n
        self.focus, self.msg, self.long = 0, "", False
        self.refresh()

    # ---- answers
    def refresh(self):
        """Untouched answers follow the ones before them (e.g. the model size follows the run folder)."""
        answers = {}
        for j, f in enumerate(self.fields):
            if not self.touched[j]:
                self.val[j] = f["default"](answers, self.found) or ""
            answers[f["key"]] = self.val[j]

    def answers(self):
        return {f["key"]: self.val[j] for j, f in enumerate(self.fields)}

    def problem(self, j):
        f, v = self.fields[j], self.val[j]
        if f["required"] and not v:
            return PROBLEMS["required"].format(label=field_text(self.item, f)["label"])
        if f["choices"] and v and v not in f["choices"]:
            return PROBLEMS["choice"].format(choices=", ".join(f["choices"]))
        if f.get("check") and v:
            return f["check"](v)
        return None

    # ---- keys; returns "done", "cancel" or None
    def key(self, k):
        n = len(self.fields)
        self.msg, self.long = "", False
        if k == "?" and self.focus < n and (self.fields[self.focus]["choices"] or not self.touched[self.focus]):
            self.long = True                                               # "?" is a letter once you type in a text answer
            return None
        if k == "esc":
            return "cancel"
        if k in ("down", "tab"):
            self.focus = min(n, self.focus + 1)
        elif k == "up":
            self.focus = max(0, self.focus - 1)
        elif self.focus == n:                                              # the Run row
            if k == "enter":
                for j in range(n):
                    p = self.problem(j)
                    if p:
                        self.focus, self.msg = j, p
                        return None
                return "done"
        else:
            j, f = self.focus, self.fields[self.focus]
            if k == "enter":
                p = self.problem(j)
                if p:
                    self.msg = p
                else:
                    self.focus += 1
            elif f["choices"]:
                ch = list(f["choices"])
                cur = ch.index(self.val[j]) if self.val[j] in ch else -1
                if k in ("right", " "):
                    self.val[j], self.touched[j] = ch[(cur + 1) % len(ch)], True
                elif k == "left":
                    self.val[j], self.touched[j] = ch[(cur - 1) % len(ch)], True
                elif len(k) == 1 and k.isalpha():
                    hit = [c for c in ch if c.lower().startswith(k.lower())]
                    if hit:
                        self.val[j], self.touched[j] = hit[0], True
            else:
                if k in ("right", "end"):
                    self.fresh[j] = False                                  # keep the shown text and add to it
                elif k in ("backspace", "delete"):
                    self.val[j] = "" if self.fresh[j] else self.val[j][:-1]
                    self.touched[j], self.fresh[j] = True, False
                elif k == "clear":
                    self.val[j], self.touched[j], self.fresh[j] = "", True, False
                elif len(k) == 1:
                    self.val[j] = k if self.fresh[j] else self.val[j] + k
                    self.touched[j], self.fresh[j] = True, False
        self.refresh()
        return None

    # ---- drawing
    def lines(self):
        W, n = self.W, len(self.fields)
        recs = [field_text(self.item, f) for f in self.fields]
        label_w = min(34, max(len(r["label"]) for r in recs) + 2)
        room = W - 4 - 2 - label_w - 1
        rows = []
        for j, f in enumerate(self.fields):
            cur = j == self.focus
            v = self.val[j]
            label = recs[j]["label"] + (" *" if f["required"] else "")
            head = self.c((self.ptr if cur else " ") + " " + label.ljust(label_w), "1;36" if cur else "0")
            if f["choices"]:
                body = " ".join(self.c(" %s " % c, "30;46") if c == v and cur else (self.c(" %s " % c, "1;32") if c == v else self.c(" %s " % c, "90")) for c in f["choices"])
            else:
                shown = v if len(v) <= room - 1 else self.ell + v[-(room - 2):]
                if cur:
                    body = self.c(shown, "1") + (self.c(" ", "7") if self.color else "_")
                elif v:
                    body = shown
                else:
                    body = self.c("(optional)" if not f["required"] else "(needed)", "90")
            rows.append(head + body)
        go = self.focus == n
        rows.append(self.c((self.ptr if go else " ") + " ", "1;32" if go else "90") + (self.c(" Start ", "30;42") if go else self.c(" Start ", "90")))
        box1 = self.box("Questions", rows, W, len(rows) + 2)

        if self.focus < n:
            f, rec = self.fields[self.focus], recs[self.focus]
            if self.long:
                body = textwrap.wrap(rec["more"], W - 4)[:4]
                body += [""] * (4 - len(body))
            else:
                body = textwrap.wrap(rec["help"], W - 4)[:2]
                body += [""] * (2 - len(body))
                body += [self.c("%s %s" % (FORM["example"], rec["example"]), "90") if rec["example"] else "",
                         self.c(enter_note(rec, f, self.val[self.focus]), "90")]
            info = FORM["info_choice"] if f["choices"] else (FORM["info_text"] if self.fresh[self.focus] and self.val[self.focus] else FORM["info_typing"])
        else:
            body, info = [FORM["start_help"]] + [""] * 3, ""
        last = self.c(self.msg, "1;31") if self.msg else self.c(info, "90")
        cmd = "$ persona_lm " + subprocess.list2cmdline(build_argv(self.item, self.answers()))
        box2 = self.box("About this question", body + [last, self.c(cmd, "36")], W, 8)
        text = plain_text(self.item)["title"]
        tech = TECH_NAMES.get(self.item["label"])
        title = self.c(" %s " % text, "1;36") + (self.c("(%s)" % tech, "90") if tech else "")
        return ["", title, self.c(fit(" " + FORM["footer"], W), "90")] + box1 + box2

    def run(self, read):
        self._write("\x1b[?25l")
        try:
            with keyboard_mode():
                while True:
                    self.draw(self.lines())
                    r = self.key(read())
                    if r:
                        self.erase()
                        return r == "done"
        except KeyboardInterrupt:
            self.erase()
            return False
        finally:
            self._write("\x1b[?25h")


class Confirm(Live):
    """A boxed yes/no. The answer is NO unless you press Y: Enter, N and Esc all cancel. `strict` only changes the wording (a licence is accepted)."""

    def __init__(self, title, lines, strict=False, notice=False, stream=None, color=True, width=None, unicode_ok=None):
        Live.__init__(self, stream, color, width, unicode_ok)
        self.title, self.strict, self.notice = title, strict, notice
        self.text = []
        for j, l in enumerate(lines):
            ind = "    " if l.startswith("  -") else ("  " if j else "")
            self.text += textwrap.wrap(l, self.W - 4, subsequent_indent=ind) or [""]

    def lines(self):
        C = CONFIRM
        if self.notice:
            foot = self.c(C["foot_notice"], "90")
        else:
            foot = self.c(" Y ", "30;42") + " " + (C["foot_accept"] if self.strict else C["foot_yes"]) + "     " + self.c(" N ", "30;41") + " / Enter / Esc " + (C["foot_cancel"] if self.strict else C["foot_no"])
        return [""] + self.box(self.title, self.text + ["", foot], self.W, len(self.text) + 4)

    def run(self, read):
        self._write("\x1b[?25l")
        try:
            self.draw(self.lines())
            with keyboard_mode():
                while True:
                    k = read()
                    if self.notice:
                        return False
                    if k in ("y", "Y"):
                        return True
                    if k in ("n", "N", "esc", "enter"):
                        return False
        except KeyboardInterrupt:
            return False
        finally:
            self.erase()
            self._write("\x1b[?25h")


def form_collect(item, found, out, color, width, read_edit, ask=input, say=print):
    """collect() with a boxed form. Returns the argument list, or None when the person backed out."""
    form = Form(item, found, out, color, width)
    if not form.run(read_edit):
        return None
    argv = build_argv(item, form.answers())
    if item.get("settings"):
        try:
            base, ctx, step = settings_base(found, form.answers())
            for k, v in edit_settings(base, ctx, step, ask, say, start=None if step else item.get("preset"), train_tokens=_train_tokens(form.answers())).items():
                argv += ["--" + k.replace("_", "-"), str(v)]
        except (KeyboardInterrupt, EOFError):
            say("")
            return None
    return argv


def confirm_box(argv, found, out, color, width, read_edit):
    """The boxed version of confirm_train / confirm_corpus. Commands that need no confirmation return True."""
    if argv[0] == "train":
        return Confirm(CONFIRM["title_train"], _train_lines(argv, found), False, False, out, color, width).run(read_edit)
    if argv[0] == "corpus":
        lines, blocker, strict = _corpus_lines(argv)
        if blocker:
            Confirm(CONFIRM["title_blocked"], lines, False, True, out, color, width).run(read_edit)
            return False
        return Confirm(CONFIRM["title_corpus"], lines, strict, False, out, color, width).run(read_edit)
    return True


# ------------------------------------------------------------------ asking and running
def collect(item, found, ask=input, say=print):
    """Asks what the item needs, one numbered question at a time. Returns the argument list for the normal command line,
    or None if the person backed out."""
    argv = [item["cmd"]]
    answers = {}
    n = len(item["fields"])
    if n:
        say("\n  %s" % plain_text(item)["title"])
        say("  " + FORM["collect_intro"])
    try:
        for i, f in enumerate(item["fields"], 1):
            default = f["default"](answers, found)
            hint = f["hint"](answers, found) if f["hint"] else ""
            note = "[%s]" % default if default else ("[%s]" % hint if hint else ("" if f["required"] else "(optional, Enter to skip)"))
            if f["choices"]:
                note += " choices: %s" % "/".join(f["choices"])
            rec = field_text(item, f)
            say("\n  %d/%d  %s%s: %s" % (i, n, rec["label"], " *" if f["required"] else "", rec["help"]))
            if rec["example"]:
                say("       %s %s" % (FORM["example"], rec["example"]))
            say("       " + enter_note(rec, f, default))
            while True:
                val = ask("       %s > " % note if note else "       > ").strip().strip('"').strip()
                if val == "?":
                    say("       " + rec["more"])
                    continue
                val = val or default
                if f["choices"] and val and val not in f["choices"]:
                    say("       " + PROBLEMS["choice"].format(choices=", ".join(f["choices"])))
                    continue
                if f.get("check") and val:
                    problem = f["check"](val)
                    if problem:
                        say("       %s" % problem)
                        continue
                break
            if not val and f["required"]:
                say("  " + FORM["needed_stop"].format(label=rec["label"]))
                return None
            answers[f["key"]] = val
            if f.get("switch"):
                if val == "yes":
                    argv.append(f["flag"])
            elif val:
                argv += [f["flag"], val] if f["flag"] else [val]
        if item.get("settings"):
            base, ctx, step = settings_base(found, answers)
            for k, v in edit_settings(base, ctx, step, ask, say, start=None if step else item.get("preset"), train_tokens=_train_tokens(answers)).items():
                argv += ["--" + k.replace("_", "-"), str(v)]
    except (KeyboardInterrupt, EOFError):
        say("")
        return None
    return argv


def _train_lines(argv, found):
    """The 'You are about to ...' lines for a training run, in plain words. The exact command comes last."""
    C = CONFIRM
    arg = lambda name: argv[argv.index(name) + 1] if name in argv else ""
    run, data, init = arg("--run"), arg("--data-dir"), arg("--init-from")
    r = _find_run(found, run)
    if r and r["step"]:
        total = int(arg("--steps") or 0) or r["steps"]
        first = C["about_continue"].format(run=run, step=format(r["step"], ","), of=" of %s" % format(total, ",") if total else "")
        create, new = C["create_continue"].format(run=run), False
    else:
        first, create, new = C["about_new"].format(run=run), C["create_new"].format(run=run), True
    happen = C["happen"].format(data=data or "your prepared text files") + (C["happen_copy"].format(init=init) if init and new else "")
    return [first, happen, create, C["safe_base"] if init else C["safe"], C["time"], C["careful"], C["tech"].format(cmd=show_cmd(argv))]


def _ask_yes(ask, prompt):
    # a long action starts only on an explicit yes (Enter = no)
    try:
        return ask("  " + prompt).strip().lower() in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


def confirm_train(argv, found, ask=input, say=print):
    """Training is the one command that is slow and heavy, so say plainly what it will do and ask first."""
    lines = _train_lines(argv, found)
    say("\n  " + lines[0])
    for l in lines[1:]:
        say("  " + l)
    return _ask_yes(ask, CONFIRM["ask_start"])


def _corpus_lines(argv):
    """(lines, blocker, strict): what a language download will do; blocker = text if it cannot start; strict = needs a deliberate yes (Taglish licence)."""
    from .corpus import LANGS, check_langs
    C = CONFIRM
    langs = check_langs([x.strip().lower() for x in argv[argv.index("--lang") + 1].split(",")]) if "--lang" in argv else list(LANGS)
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "all"
    names = ", ".join(LANGS[l]["label"] for l in langs)
    if mode == "all":
        if importlib.util.find_spec("datasets") is None:
            return [C["blocked"], C["blocked_next"]], True, False
        lines = [C["about_dl"], C["happen_dl"].format(langs=names)]
    else:
        lines = [C["about_clean"], C["happen_clean"].format(langs=names)]
    for l in langs:
        d = LANGS[l]
        lines.append("  - %-8s %s  (%s, licence: %s)" % (d["label"], d["source"], "all of it" if not d["default_mb"] else "about %d MB" % d["default_mb"], d["license"]))
    folders = ("public_text and public_clean" if mode == "all" else "public_clean") + (" and public_mix" if "--mix" in argv else "")
    lines += [C["create_corpus"].format(folders=folders), C["safe_corpus"]]
    mbs = [LANGS[l]["default_mb"] for l in langs]
    if mode == "all":
        lines.append(C["size_dl"].format(mb=sum(mbs)) if all(mbs) else C["size_unknown"])
    lines.append(C["time_corpus"])
    strict = "taglish" in langs
    if strict:
        lines += ["", C["taglish"]]
    return lines + [C["tech"].format(cmd=show_cmd(argv))], False, strict


def confirm_corpus(argv, ask=input, say=print):
    """Downloading is slow and uses the internet, so say plainly what will be fetched and ask first. Taglish needs a typed yes (licence)."""
    lines, blocker, strict = _corpus_lines(argv)
    say("")
    for l in lines:
        say((("  " + l) if l else ""))
    if blocker:
        return False
    try:
        if strict and ask("  " + CONFIRM["ask_taglish"]).strip().lower() != "yes":      # the licence needs the whole word
            return False
    except (KeyboardInterrupt, EOFError):
        return False
    return _ask_yes(ask, CONFIRM["ask_start"])


def show_cmd(argv):
    return "python -m persona_lm " + (subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv))


def _fallback(found, ask, say, width=None):
    """A plain numbered list for consoles that cannot do the boxes. It says the same words as the boxed screen."""
    w = min((width or shutil.get_terminal_size((80, 24)).columns) - 1, 79)
    fill = lambda text, ind: textwrap.fill(text, w, subsequent_indent=ind)
    say("\n" + SCREEN["fallback_title"])
    for lab, val, _ok in project_checks(found):
        say(fill("%s %s" % (lab, val), "    "))
    say(fill("%s %s" % (SCREEN["next"], next_suggestion(found)), "    "))
    for head, idxs in LAYOUT:
        if head:
            say("\n" + head)
        for i in idxs:
            label = ITEMS[i]["label"]
            tech = TECH_NAMES.get(label)
            say(fill("  %s) %s%s: %s" % ("q" if i == QUIT else DISPLAY.index(i) + 1, ITEM_TEXT[label]["title"], " (%s)" % tech if tech else "", ITEM_TEXT[label]["short"]), "      "))
    while True:
        try:
            k = ask(SCREEN["fallback_prompt"]).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return QUIT
        if k in ("q", ""):
            return QUIT
        if k.isdigit() and 1 <= int(k) <= QUIT:
            return DISPLAY[int(k) - 1]


def run_menu(root=".", stream=None, read=read_key, ask=input, call=None, width=None, color=None, interactive=None, forms=None, read_edit=None):
    """Shows the menu until Quit. `call(argv)` runs a command (default: the real command line).
    forms: boxed questions and confirmations instead of typed lines; by default only with the real keyboard on a real terminal."""
    out = stream or sys.stdout
    if call is None:
        from .cli import main as call
    if interactive is None:
        isatty = getattr(out, "isatty", None)
        size = shutil.get_terminal_size((100, 24))
        interactive = bool(isatty and isatty() and sys.stdin.isatty() and size.columns >= MIN_WIDTH and size.lines >= MENU_H + 2
                           and os.environ.get("TERM") != "dumb" and enable_vt())
    menu = Menu(out, color=(not os.environ.get("NO_COLOR")) if color is None else color, width=width)
    use_forms = bool(interactive and read is read_key and ask is input) if forms is None else bool(forms)
    read_edit = read_edit or read_edit_key
    found, sel = scan(root), START
    if interactive:
        menu.header()
    while True:
        try:
            sel = menu.select(read, found, sel) if interactive else _fallback(found, ask, print, width)
        except KeyboardInterrupt:
            sel = QUIT
        if sel == QUIT:
            print("bye")
            return 0
        item = ITEMS[sel]
        boxed = use_forms and item["fields"] and shutil.get_terminal_size((100, 24)).lines >= len(item["fields"]) + 15
        if boxed:
            argv = form_collect(item, found, out, menu.color, width, read_edit, ask)
            if argv is not None and not confirm_box(argv, found, out, menu.color, width, read_edit):
                argv = None
                print("  not started.")
        else:
            argv = collect(item, found, ask)
            if argv is not None and argv[0] == "train" and not confirm_train(argv, found, ask):
                print("  not started.")
                argv = None
            if argv is not None and argv[0] == "corpus" and not confirm_corpus(argv, ask):
                print("  not started.")
                argv = None
        if argv is not None:
            print("\n  $ %s\n" % show_cmd(argv))
            try:
                call(argv)
            except SystemExit as e:
                if e.code not in (None, 0):
                    print("\nstopped: %s" % e.code)
            except KeyboardInterrupt:
                print("\nstopped")
            except Exception as e:                                 # a failed command must not close the menu
                print("\nerror: %s: %s" % (type(e).__name__, e))
            try:
                ask("\n  Press Enter to go back to the menu...")
            except (EOFError, KeyboardInterrupt):
                print("bye")
                return 0
        found = scan(root)
        print()
