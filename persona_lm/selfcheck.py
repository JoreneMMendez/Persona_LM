"""`python -m persona_lm doctor --windows`: one command that tries the parts that were only ever tested in a sandbox, on YOUR computer.

  python -m persona_lm doctor --windows                 # the whole self-check (about a minute)
  python -m persona_lm doctor --windows --no-tests      # skip running tests\\test_hybrid.py
  python -m persona_lm doctor --windows --root C:\\Users\\user\\persona_lm_project

What it checks, each one in plain words (OK / warning / PROBLEM, and what to do):
  1. Python, NumPy, PyTorch, the GPU (name, memory, bf16), free disk space, a path that is too long or inside OneDrive.
  2. The console: can it show the box characters of the training panel and the menu, and are colour codes switched on?
  3. A tiny NumPy training run, then a chat sample from it.
  4. A tiny PyTorch training run (on the GPU when there is one), with the speed.
  5. CPU+GPU hybrid training (--cpu-share), and `python -m unittest tests.test_hybrid` when the tests folder is there.
  6. The training panel and a menu form, drawn into a stream that behaves like your console (this is the check that catches
     UnicodeEncodeError crashes on a Windows console).
Everything runs on made-up text in a temporary folder. It never reads your chats, never touches work\\, data_* or any checkpoint,
and never changes your project (it only reads the project folder's free space and whether it is writable).
"""
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from .doctor import OK, PROBLEM, WARN, Report

SYNTH_CHAT = [{"prompt": "what is the cat doing", "reply": "the cat sat on the mat"},
              {"prompt": "where is the dog", "reply": "the dog sat on the log"},
              {"prompt": "kumusta ka", "reply": "ok ra ko, ikaw?"}]
SYNTH_TEXT = "the cat sat on the mat. the dog sat on the log. a cat and a dog are friends. kumusta ka, ok ra ko. "
HP = dict(batch=4, accum=1, steps=24, lr=3e-3, min_lr=3e-4, warmup=4, ckpt_every=12, log_every=4, eval_every=8, eval_batches=2)


class StrictStream(io.StringIO):
    """Behaves like a console with a fixed encoding: writing a character it cannot encode raises UnicodeEncodeError, like a Windows cp1252 console."""

    def __init__(self, encoding):
        io.StringIO.__init__(self)
        self._enc = encoding

    @property
    def encoding(self):
        return self._enc

    def write(self, s):
        s.encode(self.encoding)
        return io.StringIO.write(self, s)

    def isatty(self):
        return True


def console_encoding():
    return getattr(sys.stdout, "encoding", None) or "ascii"


def _guard(rep, where, fn, *args):
    """Runs one check; a crash inside it becomes a PROBLEM line instead of ending the whole self-check."""
    try:
        return fn(*args)
    except (Exception, SystemExit) as e:
        if os.environ.get("PERSONA_DEBUG"):
            import traceback
            traceback.print_exc()
        rep.add(PROBLEM, where, "crashed: %s: %s" % (type(e).__name__, str(e)[:200]),
                "send this line (and `python -m persona_lm doctor --windows` output) to whoever helps with the project; PERSONA_DEBUG=1 shows the full traceback")
        return None


# ---------------------------------------------------------------- 1 + 2: the computer and the console

def check_computer(rep, root):
    import numpy as np
    py = sys.version_info
    rep.add(OK if py >= (3, 10) else PROBLEM, "python", "%s on %s" % (platform.python_version(), platform.platform()),
            None if py >= (3, 10) else "install Python 3.10 or newer")
    if os.name != "nt":
        rep.add(WARN, "system", "this is not Windows (%s). The checks still run, but they are meant for your Windows computer" % platform.system())
    rep.add(OK, "numpy", np.__version__)
    info = torch_info()
    if not info["installed"]:
        rep.add(WARN, "pytorch", "not installed: training runs on the CPU only (slow). Fine for chat and the tiny tests", "for the GPU see docs\\GPU_GUIDE.md")
    elif not info["cuda"]:
        rep.add(WARN, "pytorch", "%s is installed but it cannot use a GPU (%s)" % (info["version"], info["why"]), "see docs\\GPU_GUIDE.md: install the CUDA build of PyTorch")
    else:
        rep.add(OK, "pytorch", "%s, GPU: %s, %.1f GB%s" % (info["version"], info["name"], info["gb"], ", bf16 works" if info["bf16"] else ", bf16 NOT supported (training will use float32 and be slower)"))
    try:
        free = shutil.disk_usage(root).free / 1e9
        rep.add(OK if free >= 10 else (WARN if free >= 2 else PROBLEM), "disk", "%.1f GB free where the project is" % free,
                None if free >= 10 else "checkpoints and backups are several hundred MB each: free some space")
    except OSError as e:
        rep.add(WARN, "disk", "could not read free space (%s)" % type(e).__name__)
    full = os.path.abspath(root)
    if len(full) > 150:
        rep.add(WARN, full, "the project path is long (%d characters); Windows stops at 260 including the file name" % len(full), "move the project to a short folder such as C:\\persona")
    if "onedrive" in full.lower():
        rep.add(WARN, full, "the project is inside OneDrive: sync can lock checkpoint files while training saves them", "keep the project outside OneDrive, or pause syncing while training")
    if not os.access(root, os.W_OK):
        rep.add(PROBLEM, full, "this folder is not writable: training cannot save", "run from a folder you own")


def torch_info():
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    import torch
    info = {"installed": True, "version": torch.__version__, "cuda": False, "why": ""}
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(cuda=True, name=p.name, gb=p.total_memory / 1e9, bf16=bool(torch.cuda.is_bf16_supported()))
        else:
            info["why"] = "this is the CPU-only build" if not getattr(torch.version, "cuda", None) else "no NVIDIA GPU or driver found"
    except Exception as e:
        info["why"] = "%s while asking the GPU" % type(e).__name__
    return info


def check_console(rep):
    from . import pretty
    enc = console_encoding()
    sample = "█░▁↓↑·─╭╮╰╯│…▸"
    try:
        sample.encode(enc)
        rep.add(OK, "console", "encoding %s shows the box characters: the training panel and menu use them" % enc)
    except (UnicodeEncodeError, LookupError):
        rep.add(OK, "console", "encoding %s cannot show box characters: the panel and menu use the plain ASCII look automatically" % enc,
                "optional nicer look: set PYTHONUTF8=1 or run `chcp 65001` first")
    if os.name == "nt":
        if pretty.enable_vt():
            rep.add(OK, "console", "colour/cursor codes are switched on")
        else:
            rep.add(WARN, "console", "colour/cursor codes cannot be switched on here (old console, or output is redirected)", "train with --plain; use Windows Terminal for the live panel")
        if importlib.util.find_spec("msvcrt") is None:
            rep.add(PROBLEM, "console", "msvcrt is missing: arrow keys in the menu cannot work", "use number keys, or run in a normal Python for Windows")
    size = shutil.get_terminal_size((100, 24))
    if sys.stdout.isatty() and (size.columns < pretty.MIN_WIDTH or size.lines < pretty.MIN_HEIGHT):
        rep.add(WARN, "console", "the window is %dx%d; the live panel needs at least %dx%d, otherwise training prints plain lines" % (size.columns, size.lines, pretty.MIN_WIDTH, pretty.MIN_HEIGHT), "make the window bigger")


# ---------------------------------------------------------------- 3 + 4 + 5: tiny runs

def make_tiny_data(tmp):
    """A tokenizer, token folder and config built from made-up text. Returns (tok, cfg, data_dir)."""
    from . import data as D
    from .bpe import Tokenizer
    from .model import Config
    tok = Tokenizer.train([SYNTH_TEXT * 40], 300)
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    with open(os.path.join(src, "c.jsonl"), "w", encoding="utf-8") as f:
        for r in SYNTH_CHAT * 30:
            f.write(json.dumps(r) + "\n")
    out = os.path.join(tmp, "data")
    D.prepare(tok, D.chat_docs([src], tok), out, with_mask=True, log=lambda *_: None)
    return tok, Config(vocab_size=tok.vocab_size, n_layer=2, d_model=32, n_head=4, d_ff=64, ctx=16), out


def _losses(run_dir):
    from .ckpt import load_ckpt
    got = load_ckpt(os.path.join(run_dir, "ckpt.npz"))
    if got is None:
        return None, None
    return got, [r["loss"] for r in got[2].get("history", []) if "loss" in r]


def check_numpy_run(rep, tmp, tok, cfg, data, lines):
    from .fastgen import sample_ids_cached
    from .train import train
    run = os.path.join(tmp, "run_np")
    t0 = time.time()
    train(run, data, cfg, dict(HP), log=lines.append, backup=False)
    got, losses = _losses(run)
    if got is None:
        rep.add(PROBLEM, "numpy training", "finished but the checkpoint cannot be read back", "check the disk and the antivirus: it may be blocking file renames")
        return
    if not losses or not all(math.isfinite(x) for x in losses):
        rep.add(PROBLEM, "numpy training", "the loss is not a normal number (%s)" % losses[-1:], "the maths or NumPy is broken on this computer; reinstall NumPy")
    elif losses[-1] < losses[0]:
        rep.add(OK, "numpy training", "24 steps in %.1f s, loss %.2f -> %.2f, checkpoint saved and read back" % (time.time() - t0, losses[0], losses[-1]))
    else:
        rep.add(WARN, "numpy training", "ran, but the loss did not go down (%.2f -> %.2f)" % (losses[0], losses[-1]))
    out = sample_ids_cached(got[0], cfg, tok.encode("kumusta", allow_special=False), max_new=6, temperature=0, stop_ids=())
    rep.add(OK if len(out) == 6 else WARN, "chat sampling", "produced %d token(s) from the tiny model" % len(out))


def _torch_hp(**kw):
    return {**HP, **kw}


def check_torch_run(rep, tmp, cfg, data, info, lines):
    from .pretty import STEP_RE
    from .torch_backend import train_torch
    run = os.path.join(tmp, "run_torch")
    t0 = time.time()
    train_torch(run, data, cfg, _torch_hp(), log=lines.append, device="auto", precision="auto", backup=False)
    got, losses = _losses(run)
    where = "on the GPU" if info.get("cuda") else "on the CPU"
    if got is None or not losses:
        rep.add(PROBLEM, "pytorch training", "finished but no checkpoint/loss could be read back", "try `python -m persona_lm train --backend numpy ...` and tell whoever helps you")
        return
    steps = [m for m in (STEP_RE.match(x) for x in lines) if m]
    tps = steps[-1].group(6) if steps else "?"
    if not all(math.isfinite(x) for x in losses):
        rep.add(PROBLEM, "pytorch training", "the loss is NaN or infinite %s" % where, "update the GPU driver; try --precision fp32")
    else:
        rep.add(OK, "pytorch training", "24 steps %s in %.1f s, loss %.2f -> %.2f, speed about %s tokens/s (tiny model: only a smoke test)" % (where, time.time() - t0, losses[0], losses[-1], tps))


def check_hybrid(rep, tmp, cfg, data, info, root, run_tests):
    from .torch_backend import train_torch
    if info.get("cuda"):
        lines = []
        try:
            train_torch(os.path.join(tmp, "run_hybrid"), data, cfg, _torch_hp(batch=2, accum=2, steps=8, eval_every=4, ckpt_every=4), log=lines.append, device="auto",
                        precision="auto", cpu_share="1", backup=False)
            rep.add(OK, "hybrid (--cpu-share 1)", "8 steps with one micro-batch on the CPU worked")
        except (Exception, SystemExit) as e:
            rep.add(WARN, "hybrid (--cpu-share 1)", "failed: %s: %s" % (type(e).__name__, str(e)[:160]), "leave --cpu-share off (it is off by default); GPU-only training is not affected")
    else:
        rep.add(OK, "hybrid (--cpu-share)", "skipped: it needs a GPU")
    path = os.path.join(root, "tests", "test_hybrid.py")
    if not run_tests:
        return
    if not os.path.isfile(path):
        rep.add(OK, "tests", "skipped: no tests\\test_hybrid.py under %s" % root)
        return
    r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_hybrid"], cwd=root, capture_output=True, text=True, timeout=900)
    tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["(no output)"]
    if r.returncode == 0 and "skipped" in tail[0] and info.get("installed"):
        rep.add(WARN, "tests.test_hybrid", "passed but some checks were skipped: %s" % tail[0])
    elif r.returncode == 0:
        rep.add(OK, "tests.test_hybrid", tail[0] + (" (PyTorch is not installed, so its checks were skipped)" if not info.get("installed") else ""))
    else:
        rep.add(PROBLEM, "tests.test_hybrid", "FAILED: %s" % tail[0], "run `python -m unittest tests.test_hybrid -v` and read the first error")


# ---------------------------------------------------------------- 6: the panel and a menu form

def check_panel(rep, lines):
    from .pretty import TrainDisplay
    stream = StrictStream(console_encoding())
    d = TrainDisplay("self-check", stream, color=False)
    for ln in lines:
        d.log(ln)
    rep.add(OK, "training panel", "drew %d training lines into a console-like stream (%s look) without an error" % (len(lines), "box-character" if d.unicode else "plain ASCII"))


def check_menu_form(rep, tmp):
    from . import menu as M
    item = [it for it in M.ITEMS if it["cmd"] == "corpus"][0]
    stream = StrictStream(console_encoding())
    found = {"runs": [], "data": [], "toks": [], "torch": False}
    form = M.Form(item, found, stream, color=False, width=100)
    out = None
    for _ in range(len(item["fields"]) + 1):
        out = form.key("enter")
    argv = M.build_argv(item, form.answers())
    if out == "done" and argv and argv[0] == "corpus":
        rep.add(OK, "menu form", "drew and filled in a menu form (%s look); it built: %s" % ("box-character" if form.unicode else "plain ASCII", " ".join(argv)))
    else:
        rep.add(WARN, "menu form", "the form did not finish as expected (last answer %r)" % (out,), "use the plain menu: python -m persona_lm with the number keys")
    M.scan(tmp)         # read-only look at an empty temporary folder: must not crash


# ---------------------------------------------------------------- the whole thing

def run(root=".", run_tests=True, say=print):
    rep = Report(say)
    say("Persona_LM Windows self-check (changes nothing in your project): %s" % os.path.abspath(root))
    _guard(rep, "computer", check_computer, rep, root)
    _guard(rep, "console", check_console, rep)
    info = torch_info()
    with tempfile.TemporaryDirectory(prefix="persona_selfcheck_") as tmp:
        made = _guard(rep, "tiny data", make_tiny_data, tmp)
        lines = []
        if made:
            tok, cfg, data = made
            _guard(rep, "numpy training", check_numpy_run, rep, tmp, tok, cfg, data, lines)
            if info.get("installed"):
                tl = []
                _guard(rep, "pytorch training", check_torch_run, rep, tmp, cfg, data, info, tl)
                _guard(rep, "hybrid", check_hybrid, rep, tmp, cfg, data, info, root, run_tests)
            else:
                rep.add(OK, "pytorch training", "skipped: PyTorch is not installed")
                if run_tests:
                    _guard(rep, "hybrid", check_hybrid, rep, tmp, cfg, data, info, root, run_tests)
            if lines:
                _guard(rep, "training panel", check_panel, rep, lines)
        _guard(rep, "menu form", check_menu_form, rep, tmp)
    p, w = rep.count(PROBLEM), rep.count(WARN)
    say("")
    say("Result: %d problem(s), %d warning(s). %s" % (p, w, "Everything checked works on this computer." if not p and not w else
                                                       ("Fix the PROBLEM lines first. " if p else "No problems. ") + "Warnings are things that will be slower or plainer, not broken."))
    return 1 if p else 0
