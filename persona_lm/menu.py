"""The launcher menu: `python -m persona_lm` with no arguments (or `python -m persona_lm menu`).

Three rounded boxes like the training dashboard: the big one is the list (arrow keys or number keys), the small ones show what the
highlighted item does and what the menu found in your project folder (runs, token folders, tokenizer, PyTorch).
Picking an item asks for the few things it needs, with the answers it found already filled in (press Enter to accept), prints the
exact command it is about to run, and runs it. Nothing here does anything that the normal commands cannot do.

No extra packages. A console that cannot do this (output to a file, very small window) gets a plain numbered list instead.
"""
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap

from .pretty import MAX_W, MIN_WIDTH, PANEL_H, Skin, enable_vt, fit
from .model import PRESETS
from .train import DEFAULT_HP

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
                if fn.lower().startswith("tok") and fn.lower().endswith(".json"):
                    found["toks"].append((os.path.getmtime(os.path.join(d, fn)), os.path.normpath(os.path.join(shown, fn))))
        except (OSError, ValueError):
            continue
    found["runs"].sort(reverse=True)
    found["data"].sort(reverse=True)
    found["toks"].sort(reverse=True)
    found["runs"] = [dict(path=rel, ckpt=p, size=None, step=None, steps=None) for _, rel, p in found["runs"]]
    for r in found["runs"][:1]:                          # only the newest is opened (the others when asked), so the menu opens fast
        r["size"], r["step"], r["steps"] = run_info(r["ckpt"])
    found["data"] = [p for _, p in found["data"]]
    found["toks"] = [p for _, p in found["toks"]]
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
def _field(key, ask, flag=None, default=None, required=False, choices=None, hint=None):
    return dict(key=key, ask=ask, flag=flag, default=default or (lambda a, f: ""), required=required, choices=choices, hint=hint)


def _d_run(a, f):
    return f["runs"][0]["path"] if f["runs"] else ""


def _d_ckpt(a, f):
    return os.path.join(f["runs"][0]["path"], "ckpt.npz") if f["runs"] else ""


def _d_data(a, f):
    return f["data"][0] if f["data"] else ""


def _d_tok(a, f):
    return f["toks"][0] if f["toks"] else ""


def _d_size(a, f):
    r = _find_run(f, a.get("run")) if a.get("run") else (f["runs"][0] if f["runs"] else None)
    return r["size"] if r and r["size"] not in (None, "?") else "s10m"


HELP = {"run": "where this run's checkpoints are saved (an existing folder = resume it)",
        "data": "the folder of token files made by `prepare`",
        "size": "model size; must match the run you are continuing",
        "init": "only for a NEW run: copy the weights of another model (e.g. runs\\pre\\ckpt.npz) - this is how chat fine-tuning starts. Leave empty to continue or start fresh",
        "ckpt": "the model file to load (ckpt.npz)",
        "tok": "the tokenizer the model was trained with",
        "system": "optional: a role to start the chat with, e.g. You are Aria.",
        "prompt": "the text for the model to continue",
        "dir": "the folder to look at"}

# ------------------------------------------------------------------ the training settings screen
SETTINGS = [
    dict(key="steps", short="how long the run is", kind=int, lo=1, hi=10 ** 7,
         what="How many training steps the whole run lasts. Each step reads a batch of text and updates the model once.",
         up="Longer training. The model sees the data more times: it can learn more, but on small data it starts memorizing (overfitting).",
         down="A shorter, faster run. Less risk of memorizing, but a less-trained model.",
         affects="Run time, and the learning-rate curve (it shrinks gradually until the last step).",
         tip="Must be more than the current step when resuming. Compare 'tokens seen' below with the size of your token files."),
    dict(key="lr", short="size of each update", kind=float, lo=1e-6, hi=1e-1,
         what="Learning rate: how big a change each step makes to the model. It starts here and slowly shrinks to min_lr.",
         up="Learns faster, but can become unstable (loss spikes, or nan).",
         down="Gentler, steadier updates. Slower progress, and usually less overfitting.",
         affects="Speed and stability of learning. Changing it in the middle of a run makes the rate jump to the new curve at once.",
         tip="3e-4 for a gentler pretraining run; 1e-4 for chat fine-tuning."),
    dict(key="min_lr", short="size of updates at the end", kind=float, lo=0.0, hi=1e-1,
         what="The smallest learning rate, reached at the last step.",
         up="The model is still changed strongly late in the run.",
         down="The run settles gently at the end, which polishes fine details.",
         affects="How the end of the run behaves.",
         tip="About one tenth of lr. Must not be higher than lr."),
    dict(key="warmup", short="slow start", kind=int, lo=0, hi=100000,
         what="Steps at the start where the learning rate climbs slowly from 0 up to lr, so a fresh model is not shaken by big early updates.",
         up="A safer start, slightly slower.",
         down="A faster start, with a risk of early loss spikes.",
         affects="Only the first steps of a NEW run. No effect when you resume a run that is already past this step.",
         tip="50 to 100 is typical."),
    dict(key="weight_decay", short="brake on memorizing", kind=float, lo=0.0, hi=1.0,
         what="A small pull of every weight toward zero at each step. It works as a brake on memorizing the training text.",
         up="A stronger brake: less overfitting. Too high and the model cannot learn (loss stays high).",
         down="A weaker brake: fits the training text more closely and overfits sooner.",
         affects="How much the model memorizes instead of generalizing.",
         tip="0.1 is the default. 0.2 is a good try when data is small."),
    dict(key="clip", short="cap on one update", kind=float, lo=0.0, hi=100.0,
         what="Gradient clipping: limits how large one update may be, so a single odd batch cannot wreck the model.",
         up="Allows bigger updates.",
         down="Safer against loss spikes; too low slows learning.",
         affects="Stability. 0 turns it off.",
         tip="Leave it at 1.0 unless you see sudden loss spikes."),
    dict(key="batch", short="texts per GPU pass", kind=int, lo=1, hi=512,
         what="How many text sequences the GPU reads at once (one micro-batch).",
         up="Faster per token, but needs more GPU memory. A 6 GB laptop GPU runs out quickly.",
         down="Uses less memory, but slower.",
         affects="GPU memory and speed. Together with accum it sets the effective batch.",
         tip="If you get an out-of-memory error, halve batch and double accum."),
    dict(key="accum", short="passes per update", kind=int, lo=1, hi=1024,
         what="How many micro-batches are added up before the model is updated once. It is a way to get a big batch on a small GPU.",
         up="Smoother, more reliable updates (a bigger effective batch). Each step takes longer.",
         down="Quicker steps, but noisier updates.",
         affects="Effective batch = batch x accum, and the time per step.",
         tip="Keep batch x accum the same when you only want to fit the GPU memory."),
    dict(key="eval_every", short="how often to check val", kind=int, lo=1, hi=100000,
         what="How often (in steps) the model is tested on validation text that it never trains on. This is where the val number comes from.",
         up="Fewer checks, slightly faster, but you notice overfitting later.",
         down="You spot overfitting sooner, with a tiny slowdown.",
         affects="How closely you can watch for overfitting (and which step can be 'best').",
         tip="100 to 200."),
]

QUICK = [
    ("Fight overfitting", dict(weight_decay=0.2, lr=3e-4, min_lr=3e-5, eval_every=100),
     "Stronger brake on memorizing, gentler updates, validation checked every 100 steps. It cannot replace more data."),
    ("Chat fine-tune", dict(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000),
     "A short, gentle run for tuning a pretrained model on chat data. Needs a NEW run folder with 'Start from weights' filled in."),
]


def _fmt(v):
    return format(v, ",") if isinstance(v, int) else "%g" % v


def _parse(text, kind):
    return kind(float(text.replace(",", "").replace("_", ""))) if kind is int else float(text.replace(",", ""))


def edit_settings(base, ctx, step, ask=input, say=print):
    """Shows the training settings with plain-words help and lets the person change any of them before the run starts.
    `base` = the values the run uses now (saved in the run, or the defaults for a new one); `ctx` = tokens per sequence; `step` = where
    a resumed run is (0 for a new one). Returns only the settings that were CHANGED, as {name: value}."""
    cur = dict(base)
    n = len(SETTINGS)
    say("\n  Training settings")
    say("  These are the numbers training will use. Press Enter to keep them all, or type a number to see what a setting does and change it.")
    while True:
        say("")
        for i, d in enumerate(SETTINGS, 1):
            k = d["key"]
            mark = "*" if cur[k] != base[k] else " "
            was = "  (was %s)" % _fmt(base[k]) if mark == "*" else ""
            say("  %s%d  %-13s %9s   %s%s" % (mark, i, k, _fmt(cur[k]), d["short"], was))
        per = cur["batch"] * cur["accum"] * ctx
        say("\n  Each step trains on %s tokens, so %s steps = %s tokens in total. Far more tokens than your data holds means memorizing." % (
            format(per, ","), format(cur["steps"], ","), format(per * cur["steps"], ",")))
        if step and any(cur[k] != base[k] for k in ("lr", "min_lr", "steps")):
            say("  Heads-up: this run is at step %s. The learning rate follows a curve from lr down to min_lr over all the steps, so changing lr, min_lr or steps makes it jump to the new curve right now." % format(step, ","))
        if cur["batch"] * cur["accum"] != base["batch"] * base["accum"]:
            say("  Heads-up: batch x accum changed, so each update now uses a different amount of text. Fine, just not the same as before.")
        k = ask("\n  Change which? number 1-%d, p = quick presets, r = reset, Enter = continue > " % n).strip().lower()
        if not k:
            break
        if k == "r":
            cur = dict(base)
        elif k == "p":
            say("")
            for j, (name, vals, why) in enumerate(QUICK, 1):
                say("  %d  %s: %s" % (j, name, ", ".join("%s=%s" % (a, _fmt(b)) for a, b in vals.items())))
                say("     %s" % why)
            c = ask("\n  Apply which? number, or Enter for none > ").strip()
            if c.isdigit() and 1 <= int(c) <= len(QUICK):
                cur.update(QUICK[int(c) - 1][1])
        elif k.isdigit() and 1 <= int(k) <= n:
            d = SETTINGS[int(k) - 1]
            say("\n  %s   (now %s, was %s)" % (d["key"], _fmt(cur[d["key"]]), _fmt(base[d["key"]])))
            for label, field in (("What it does", "what"), ("Raise it", "up"), ("Lower it", "down"), ("Affects", "affects"), ("Tip", "tip")):
                for j, line in enumerate(textwrap.wrap(d[field], 74)):
                    say("    %-13s %s" % (label + ":" if j == 0 else "", line))
            raw = ask("\n  New value (Enter = keep) > ").strip()
            if raw:
                try:
                    v = _parse(raw, d["kind"])
                except ValueError:
                    say("  that is not a number, nothing changed")
                    continue
                key = d["key"]
                if not d["lo"] <= v <= d["hi"]:
                    say("  %s must be between %s and %s, nothing changed" % (key, _fmt(d["lo"]), _fmt(d["hi"])))
                elif key == "steps" and step and v <= step:
                    say("  steps must be more than %s, where this run already is, nothing changed" % format(step, ","))
                elif key == "lr" and v < cur["min_lr"]:
                    say("  lr cannot be lower than min_lr (%s): lower min_lr first, nothing changed" % _fmt(cur["min_lr"]))
                elif key == "min_lr" and v > cur["lr"]:
                    say("  min_lr cannot be higher than lr (%s), nothing changed" % _fmt(cur["lr"]))
                else:
                    cur[key] = v
        else:
            say("  type a number from 1 to %d, p, r, or just press Enter" % n)
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
    dict(label="Chat with a model", cmd="chat", desc="Talk to a trained model in this window.",
         fields=[_field("ckpt", "Checkpoint", "--ckpt", _d_ckpt, True), _field("tok", "Tokenizer", "--tokenizer", _d_tok, True),
                 _field("system", "System prompt", "--system")]),
    dict(label="Generate text", cmd="generate", desc="Continue a prompt with a model, once, and print it.",
         fields=[_field("ckpt", "Checkpoint", "--ckpt", _d_ckpt, True), _field("tok", "Tokenizer", "--tokenizer", _d_tok, True),
                 _field("prompt", "Prompt", "--prompt", None, True)]),
    dict(label="Project status", cmd="status", desc="What exists in this folder (tokenizers, token files, runs) and what to do next.",
         fields=[_field("dir", "Folder", None, lambda a, f: ".")]),
    dict(label="Open the Web UI", cmd="ui", desc="The local, offline web page for the whole system. Opens your browser.", fields=[]),
    dict(label="Speed benchmark", cmd="benchmark", desc="Measure tokens per second on this computer and estimate training time.",
         fields=[_field("size", "Model size", "--size", _d_size, True, SIZES)]),
    dict(label="Check the GPU", cmd="check-gpu", desc="Check the PyTorch / GPU backend against the reference and print a report.", fields=[]),
    dict(label="Quit", cmd=None, desc="Leave the menu.", fields=[]),
]
QUIT = len(ITEMS) - 1


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
        lw = W * 57 // 100
        rw = W - lw - 1
        li, ri = lw - 4, rw - 4
        rows = []
        for i, it in enumerate(ITEMS):
            key = "q" if i == QUIT else str(i + 1)
            if i == sel:
                rows.append(self.c(fit("%s %s  %s" % (self.ptr, key, it["label"]), li), "30;46"))
            else:
                rows.append("  " + self.c(key, "90") + "  " + it["label"])
        big = rows + ["", self.c("up/down move  Enter select  q quit" if not self.unicode else "↑↓ move   Enter select   q quit", "90")]

        it = ITEMS[sel]
        words = textwrap.wrap(it["desc"], ri)[:3]
        cmd = "$ persona_lm %s" % it["cmd"] if it["cmd"] else ""
        det = words + [""] * (3 - len(words)) + [self.c(cmd, "36")]

        runs, g = found["runs"], self.c
        if runs:
            r = runs[0]
            extra = "  +%d" % (len(runs) - 1) if len(runs) > 1 else ""
            prog = "  %d%%" % (100 * r["step"] // r["steps"]) if r["steps"] else ""
            run_l = g("run     ", "90") + os.path.basename(r["path"].rstrip("\\/")) + g("  %s%s%s" % (r["size"], prog, extra), "90")
        else:
            run_l = g("run     ", "90") + g("none yet", "33")
        data_l = g("data    ", "90") + (found["data"][0] + (g("  +%d" % (len(found["data"]) - 1), "90") if len(found["data"]) > 1 else "") if found["data"] else g("none yet", "33"))
        tok_l = g("tokens  ", "90") + (found["toks"][0] if found["toks"] else g("none yet", "33"))
        tor_l = g("torch   ", "90") + (g("installed", "32") if found["torch"] else g("not found (slow CPU mode)", "33"))

        left = self.box("Menu", big, lw, PANEL_H)
        right = self.box("Details", det, rw, 6) + self.box("Project", [run_l, data_l, tok_l, tor_l], rw, PANEL_H - 6)
        return [a + " " + b for a, b in zip(left, right)]

    def select(self, read, found, sel=0):
        """Draws the menu and returns the chosen item's index (QUIT for q / Esc)."""
        self._write("\x1b[?25l")
        try:
            while True:
                lines = self.render(sel, found)
                self._write("\n".join(lines) + "\n")
                self.n = len(lines)
                k = read()
                if k == "enter" or (k.isdigit() and 1 <= int(k) <= QUIT):
                    self.n = 0                                           # leave the menu on screen: the questions appear under it
                    return sel if k == "enter" else int(k) - 1
                self._erase()
                if k in ("up", "k"):
                    sel = (sel - 1) % len(ITEMS)
                elif k in ("down", "j"):
                    sel = (sel + 1) % len(ITEMS)
                elif k == "home":
                    sel = 0
                elif k == "end":
                    sel = QUIT
                elif k in ("q", "esc"):
                    return QUIT
        finally:
            self._write("\x1b[?25h")


# ------------------------------------------------------------------ asking and running
def collect(item, found, ask=input, say=print):
    """Asks what the item needs, one numbered question at a time. Returns the argument list for the normal command line,
    or None if the person backed out."""
    argv = [item["cmd"]]
    answers = {}
    n = len(item["fields"])
    if n:
        say("\n  %s" % item["label"])
        say("  Answer each question, or press Enter to use the value in [brackets]. Ctrl+C goes back to the menu.")
    try:
        for i, f in enumerate(item["fields"], 1):
            default = f["default"](answers, found)
            hint = f["hint"](answers, found) if f["hint"] else ""
            note = "[%s]" % default if default else ("[%s]" % hint if hint else ("" if f["required"] else "(optional, Enter to skip)"))
            if f["choices"]:
                note += " choices: %s" % "/".join(f["choices"])
            say("\n  %d/%d  %s: %s" % (i, n, f["ask"], HELP.get(f["key"], "")))
            while True:
                val = ask("       %s > " % note if note else "       > ").strip().strip('"').strip()
                val = val or default
                if f["choices"] and val and val not in f["choices"]:
                    say("       choose one of: %s" % ", ".join(f["choices"]))
                    continue
                break
            if not val and f["required"]:
                say("  %s is needed, so nothing was started." % f["ask"])
                return None
            answers[f["key"]] = val
            if val:
                argv += [f["flag"], val] if f["flag"] else [val]
        if item.get("settings"):
            base, ctx, step = settings_base(found, answers)
            for k, v in edit_settings(base, ctx, step, ask, say).items():
                argv += ["--" + k.replace("_", "-"), str(v)]
    except (KeyboardInterrupt, EOFError):
        say("")
        return None
    return argv


def confirm_train(argv, found, ask=input, say=print):
    """Training is the one command that is slow and heavy, so say plainly what it will do and ask first."""
    run = argv[argv.index("--run") + 1] if "--run" in argv else ""
    r = _find_run(found, run)
    if r and r["step"]:
        total = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else r["steps"]
        what = "CONTINUE %s from step %s%s" % (run, format(r["step"], ","), " of %s" % format(total, ",") if total else "")
    else:
        what = "start a NEW run in %s" % run
    say("\n  This will %s." % what)
    say("  If this run is already training in another window, do not start it a second time.")
    try:
        return ask("  Start now? [Y/n] > ").strip().lower() not in ("n", "no")
    except (KeyboardInterrupt, EOFError):
        return False


def show_cmd(argv):
    return "python -m persona_lm " + (subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv))


def _fallback(found, ask, say):
    """A plain numbered list for consoles that cannot do the boxes."""
    say("\nPersonaLM menu")
    for i, it in enumerate(ITEMS):
        say("  %s) %s" % ("q" if i == QUIT else i + 1, it["label"]))
    while True:
        try:
            k = ask("Choose 1-%d or q: " % QUIT).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return QUIT
        if k in ("q", ""):
            return QUIT
        if k.isdigit() and 1 <= int(k) <= QUIT:
            return int(k) - 1


def run_menu(root=".", stream=None, read=read_key, ask=input, call=None, width=None, color=None, interactive=None):
    """Shows the menu until Quit. `call(argv)` runs a command (default: the real command line)."""
    out = stream or sys.stdout
    if call is None:
        from .cli import main as call
    if interactive is None:
        isatty = getattr(out, "isatty", None)
        size = shutil.get_terminal_size((100, 24))
        interactive = bool(isatty and isatty() and sys.stdin.isatty() and size.columns >= MIN_WIDTH and size.lines >= PANEL_H + 4
                           and os.environ.get("TERM") != "dumb" and enable_vt())
    menu = Menu(out, color=(not os.environ.get("NO_COLOR")) if color is None else color, width=width)
    found, sel = scan(root), 0
    if interactive:
        menu.header()
    while True:
        try:
            sel = menu.select(read, found, sel) if interactive else _fallback(found, ask, print)
        except KeyboardInterrupt:
            sel = QUIT
        if sel == QUIT:
            print("bye")
            return 0
        argv = collect(ITEMS[sel], found, ask)
        if argv is not None and argv[0] == "train" and not confirm_train(argv, found, ask):
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
