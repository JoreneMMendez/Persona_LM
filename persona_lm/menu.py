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

SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "site-packages", "private", "tests", "docs"}
SIZES = sorted(PRESETS, key=lambda k: PRESETS[k]["d_model"])


# ------------------------------------------------------------------ what is in the project folder
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


def _h_steps(a, f):
    r = _find_run(f, a.get("run"))
    return "keep %s" % format(r["steps"], ",") if r and r["steps"] else ""


ITEMS = [
    dict(label="Train or resume a run", cmd="train", desc="Start training, or continue a run exactly where it stopped. Shows the live dashboard.",
         fields=[_field("run", "Run folder", "--run", _d_run, True), _field("data", "Token folder", "--data-dir", _d_data, True),
                 _field("size", "Model size", "--size", _d_size, True, SIZES), _field("steps", "Total steps", "--steps", None, False, None, _h_steps)]),
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
                self._erase()
                if k in ("up", "k"):
                    sel = (sel - 1) % len(ITEMS)
                elif k in ("down", "j"):
                    sel = (sel + 1) % len(ITEMS)
                elif k == "home":
                    sel = 0
                elif k == "end":
                    sel = QUIT
                elif k == "enter":
                    return sel
                elif k in ("q", "esc"):
                    return QUIT
                elif k.isdigit() and 1 <= int(k) <= QUIT:
                    return int(k) - 1
        finally:
            self._write("\x1b[?25h")


# ------------------------------------------------------------------ asking and running
def collect(item, found, ask=input, say=print):
    """Asks what the item needs. Returns the argument list for the normal command line, or None if the person backed out."""
    argv = [item["cmd"]]
    answers = {}
    if item["fields"]:
        say("\n  %s: Enter accepts [what is shown]. Ctrl+C goes back." % item["label"])
    try:
        for f in item["fields"]:
            default = f["default"](answers, found)
            note = "[%s]" % default if default else ("[%s]" % f["hint"](answers, found) if f["hint"] and f["hint"](answers, found) else ("" if f["required"] else "(optional)"))
            if f["choices"]:
                note += " {%s}" % "/".join(f["choices"])
            while True:
                val = ask("  %s %s: " % (f["ask"], note)).strip().strip('"').strip()
                val = val or default
                if f["choices"] and val and val not in f["choices"]:
                    say("  choose one of: %s" % ", ".join(f["choices"]))
                    continue
                break
            if not val and f["required"]:
                say("  %s is needed, so nothing was started." % f["ask"])
                return None
            answers[f["key"]] = val
            if val:
                argv += [f["flag"], val] if f["flag"] else [val]
    except (KeyboardInterrupt, EOFError):
        say("")
        return None
    return argv


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
