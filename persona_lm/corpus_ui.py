"""A terminal progress screen for `python -m persona_lm corpus` (and menu item 8, Language data).

    ╭─ Steps ───────────────────╮ ╭─ Now ───────────────────────────╮
    │ ✓ Download English  4:12  │ │ Download Tagalog     96 / 300 MB│
    │ ▸ Download Tagalog        │ │ ████████░░░░░░░░░░░░░░░░░░  32% │
    │ · Download Cebuano        │ │ speed 3.1 MB/s   ETA 1:05       │
    │ · ...                     │ │ step 2/10        elapsed 5:30   │
    │                           │ ╰─────────────────────────────────╯
    │                           │ ╭─ Log ───────────────────────────╮
    │                           │ │ last lines the download printed │
    ╰───────────────────────────╯ ╰─────────────────────────────────╯

How it works: the download and cleaning code is unchanged and still prints its normal lines. While the screen is shown those lines (and the
download library's own progress bars) are caught and shown in the Log box instead of scrolling over the panel. Progress is measured from the
size of the `.part` file that every step writes first, so it needs no cooperation from the downloaders.

Only on a real terminal. Output to a file or pipe, the web UI's background job (it passes --stopfile / --progress), `--plain` and
PERSONA_PLAIN=1 all get the plain lines exactly as before. Ctrl+C asks the run to stop after the current step; a second Ctrl+C quits at once
(partial `.part` files are kept and never replace a good file).
"""
import collections
import io
import os
import re
import shutil
import sys
import tempfile
import threading
import time

from . import corpus as C
from .pretty import MAX_W, MIN_HEIGHT, MIN_WIDTH, PANEL_H, Skin, enable_vt, hms, spread

SPEED_WINDOW = 20.0          # seconds of samples the speed is averaged over


def wanted(stream=None, plain=False):
    """True when the live screen should be used: a real terminal that is big enough, and not switched off."""
    out = stream or sys.stdout
    if plain or os.environ.get("PERSONA_PLAIN"):
        return False
    isatty = getattr(out, "isatty", None)
    if not (isatty and isatty()):
        return False
    size = shutil.get_terminal_size((100, 24))
    if size.columns < MIN_WIDTH or size.lines < MIN_HEIGHT + 1 or os.environ.get("TERM") == "dumb":
        return False
    return bool(enable_vt())


class Capture(io.TextIOBase):
    """Stands in for sys.stdout / sys.stderr while the screen is up. Every finished line goes to sink(text, replaces_previous);
    text that follows a carriage return (a progress bar redrawing itself) replaces the previous line instead of adding a new one."""
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, sink):
        self.sink = sink
        self.buf = ""
        self.lead = "\n"                  # what ended the piece before: "\r" means the next text redraws the same line
        self.lock = threading.Lock()

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode("utf-8", "replace")
        with self.lock:
            self.buf += s
            while True:
                m = re.search(r"[\r\n]", self.buf)
                if not m:
                    break
                piece, end = self.buf[:m.start()], self.buf[m.start()]
                self.buf = self.buf[m.end():]
                if piece.strip():
                    self.sink(piece, self.lead == "\r")
                self.lead = end
        return len(s)

    def flush(self):
        pass

    def finish(self):
        with self.lock:
            if self.buf.strip():
                self.sink(self.buf, self.lead == "\r")
            self.buf = ""


def _took(seconds):
    s = int(max(0, seconds))
    return "%d:%02d" % (s // 60, s % 60) if s < 3600 else hms(s)


def short_label(stage):
    kind, lang = stage["kind"], stage["lang"]
    if kind == "download":
        return "Download " + C.LANGS[lang]["label"]
    if kind == "clean":
        return ("Check " if lang == "en" else "Clean ") + C.LANGS[lang]["label"]
    return "Build training mix" if kind == "mix" else "Check the files"


class CorpusDisplay(Skin):
    def __init__(self, langs, mode="all", mb=None, mix=False, stream=None, color=True, width=None, clock=time.time, unicode_ok=None):
        Skin.__init__(self, stream or sys.stdout, color, unicode_ok)
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.clock = clock
        self.langs = C.check_langs(langs)
        self.mode, self.mix, self.mb = mode, mix, dict(mb or {})
        self.stages = C.plan_stages(self.langs, mode, mix)
        self.state = ["todo"] * len(self.stages)
        self.began = [None] * len(self.stages)
        self.took = [None] * len(self.stages)
        self.cur = None
        self.samples = collections.deque()
        self.log = collections.deque(maxlen=60)
        self.t0 = clock()
        self.n = 0                          # lines of the live panel on screen
        self.finished = False
        self.stopping = False
        self.failed = None
        self.mark = {"done": "✓", "now": self.ptr, "todo": self.dot, "fail": "x"} if self.unicode else {"done": "v", "now": ">", "todo": ".", "fail": "x"}

    # ---------------------------------------------------------------- what the run tells us
    def on_stage(self, index, total, title, lang):
        now = self.clock()
        self._close_current(now, "done")
        if title == "Finished":
            self.finished = True
            self.cur = None
            return
        self.cur = index - 1
        self.state[self.cur] = "now"
        self.began[self.cur] = now
        self.samples.clear()

    def _close_current(self, now, state):
        if self.cur is not None and self.state[self.cur] == "now":
            self.state[self.cur] = state
            self.took[self.cur] = now - self.began[self.cur]

    def fail(self):
        self._close_current(self.clock(), "fail")

    def feed(self, text, replaces=False):
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text).rstrip()
        if not text:
            return
        if replaces and self.log:
            self.log[-1] = text
        else:
            self.log.append(text)

    # ---------------------------------------------------------------- measuring
    def watch(self):
        """(path to measure, wanted bytes or None) for the running step."""
        st = self.stages[self.cur]
        kind, lang = st["kind"], st["lang"]
        if kind == "download":
            spec = C.LANGS[lang]
            cap = self.mb.get(lang)
            cap = spec["default_mb"] if cap is None else cap
            path = (spec["clean"] if lang == "en" else spec["raw"]) + ".part"
            return path, (cap * 1_000_000 if cap else None)
        if kind == "clean":
            return (C.LANGS[lang]["clean"] + ".part" if lang != "en" else None), None
        if kind == "mix":
            return os.path.join("public_mix", ".part"), sum(C.LANGS[l]["mix_mb"] for l in self.langs) * 1_000_000
        return None, None

    @staticmethod
    def size_of(path):
        if not path:
            return 0
        try:
            if os.path.isdir(path):
                return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
            return os.path.getsize(path)
        except OSError:
            return 0

    def speed(self):
        """bytes per second over the last SPEED_WINDOW seconds, or None while there is too little to say."""
        if len(self.samples) < 2:
            return None
        (t1, b1), (t2, b2) = self.samples[0], self.samples[-1]
        return (b2 - b1) / (t2 - t1) if t2 - t1 >= 1.0 and b2 >= b1 else None

    # ---------------------------------------------------------------- drawing
    def bar(self, frac, w, now):
        if frac is None:                                           # unknown length: a block that slides back and forth
            span = max(1, w - 6)
            k = int(now * 6) % (2 * span)
            pos = k if k < span else 2 * span - k
            return self.c(self.empty * pos, "90") + self.c(self.full * 6, "36") + self.c(self.empty * (w - 6 - pos), "90")
        filled = int(round(min(1.0, max(0.0, frac)) * w))
        return self.c(self.full * filled, "32") + self.c(self.empty * (w - filled), "90")

    def render(self):
        now = self.clock()
        W = min(self.width - 1, MAX_W)
        lw = W * 45 // 100
        rw = W - lw - 1
        li, ri = lw - 4, rw - 4

        steps = []
        for j, st in enumerate(self.stages):
            state = self.state[j]
            col = {"done": "32", "now": "1;36", "todo": "90", "fail": "1;31"}[state]
            left = self.c(self.mark[state] + " ", col) + self.c(short_label(st), col if state != "todo" else "90")
            steps.append(spread(left, self.c(_took(self.took[j]), "90") if self.took[j] is not None else "", li))

        total = len(self.stages)
        size, want, label, size_txt, frac = 0, None, "", "", None
        if self.finished:
            label, frac = "All done", 1.0
        elif self.cur is not None:
            label = short_label(self.stages[self.cur])
            path, want = self.watch()
            size = self.samples[-1][1] if self.samples else self.size_of(path)
            kind = self.stages[self.cur]["kind"]
            if want:
                frac = size / want
                size_txt = "%d / %d MB" % (size / 1e6, want / 1e6)
            elif kind in ("download", "clean") and path:
                size_txt = "%d MB" % (size / 1e6)
            else:
                size_txt = ""
        else:
            label = "starting..."
        pct = " %3d%%" % int((frac or 0) * 100) if frac is not None else "     "
        barw = max(8, ri - 5)
        l1 = spread(self.c(label, "1"), self.c(size_txt, "36"), ri)
        l2 = self.bar(frac, barw, now) + self.c(pct, "1")
        sp = self.speed()
        if self.stopping and not self.finished:
            l3 = self.c("stopping after this step...", "1;33")
        elif self.finished:
            l3 = self.c("finished", "1;32")
        elif sp is not None:
            eta = (want - size) / sp if want and sp > 0 and size < want else None
            l3 = self.c("speed ", "90") + self.c("%.1f MB/s" % (sp / 1e6), "36") + ("   " + self.c("ETA ", "90") + self.c(hms(eta), "1") if eta is not None else "")
        elif self.cur is not None and self.stages[self.cur]["kind"] == "download" and size == 0:
            l3 = self.c("connecting / waiting for the first data...", "90")
        else:
            l3 = ""
        done = sum(1 for x in self.state if x == "done")
        l4 = spread(self.c("step %d/%d" % (min(total, done + (0 if self.finished or self.cur is None else 1)), total), "90"), self.c("elapsed ", "90") + hms(now - self.t0), ri)

        tail = list(self.log)[-4:]
        left = self.box("Steps", steps, lw, PANEL_H)
        right = self.box("Now", [l1, l2, l3, l4], rw, 6) + self.box("Log", [self.c(x, "90") for x in tail], rw, PANEL_H - 6)
        hint = [] if self.finished else [self.c("Ctrl+C: stop after this step   Ctrl+C twice: quit now", "90")]
        return [a + " " + b for a, b in zip(left, right)] + hint

    def _write(self, text):
        self.out.write(text)
        self.out.flush()

    def erase(self):
        if self.n:
            self._write("\x1b[%dF\x1b[J" % self.n)
            self.n = 0

    def header(self):
        title = " PersonaLM language data " + self.c(self.dot + " %s  %s  mode %s  mix %s" % (", ".join(C.LANGS[l]["label"] for l in self.langs), self.dot, self.mode, "yes" if self.mix else "no"), "90")
        self._write("\n" + self.c(title, "1;36") + "\n" + self.c(self.rule * min(self.width - 1, MAX_W - 1), "90") + "\n")

    def tick(self):
        if self.cur is not None and not self.finished:
            path, _ = self.watch()
            now = self.clock()
            self.samples.append((now, self.size_of(path)))
            while len(self.samples) > 2 and now - self.samples[0][0] > SPEED_WINDOW:
                self.samples.popleft()
        self.erase()
        lines = self.render()
        self._write("\n".join(lines) + "\n")
        self.n = len(lines)

    def summary(self, outcome):
        """Lines printed under the last panel: how it ended and what files exist now."""
        rule = self.c(self.rule * min(self.width - 1, MAX_W - 1), "90")
        head = {"done": self.c("finished", "1;32") + "  all %d steps in %s" % (len(self.stages), hms(self.clock() - self.t0)),
                "stopped": self.c("stopped", "1;33") + "  nothing half-written replaced a good file; run it again to continue",
                "error": self.c("failed", "1;31") + "  %s" % (self.failed or "")}[outcome]
        lines = [rule, " " + head]
        for l in self.langs:
            path = C.LANGS[l]["clean"]
            if os.path.exists(path):
                lines.append("   %-8s %7.1f MB  %s" % (C.LANGS[l]["label"], os.path.getsize(path) / 1e6, path.replace("/", os.sep)))
        if outcome == "done":
            lines.append(" Next: turn it into token files with `prepare` into a NEW folder (keep work%stok.json). See the Language data page or HANDOFF." % os.sep)
        return "\n".join(lines)


def run_with_display(langs, mode="all", mb=None, mix=False, stream=None, color=None, poll=0.5):
    """Runs the language-data pipeline in a worker thread and keeps the progress screen alive in this one. Raises what the run raised
    (CorpusError, or KeyboardInterrupt when stopped) after printing the summary."""
    out = stream or sys.stdout
    disp = CorpusDisplay(langs, mode, mb, mix, out, color=(not os.environ.get("NO_COLOR")) if color is None else color)
    fd, stopfile = tempfile.mkstemp(prefix="persona_corpus_stop_")
    os.close(fd)
    os.remove(stopfile)                                       # its existence is the "stop" signal
    result = {}
    cap_out, cap_err = Capture(disp.feed), Capture(disp.feed)

    over = threading.Event()                                  # set when the worker is done (safer than Thread.join, which Ctrl+C can upset)

    def work():
        try:
            C.run(langs, mode=mode, mb=mb, mix=mix, stopfile=stopfile)
        except BaseException as e:                            # handed to the main thread below
            result["err"] = e
        finally:
            over.set()

    real = sys.stdout, sys.stderr
    C.STAGE_HOOK = disp.on_stage
    worker = threading.Thread(target=work, daemon=True)
    disp.header()
    out.write("\x1b[?25l")
    sys.stdout, sys.stderr = cap_out, cap_err
    forced = False
    try:
        worker.start()
        while not over.is_set():
            try:
                disp.tick()
                over.wait(poll)
            except KeyboardInterrupt:
                if disp.stopping:                             # second Ctrl+C: leave now
                    forced = True
                    break
                disp.stopping = True
                open(stopfile, "w").close()
    finally:
        sys.stdout, sys.stderr = real
        C.STAGE_HOOK = None
        cap_out.finish()
        cap_err.finish()
        out.write("\x1b[?25h")
        out.flush()
        if os.path.exists(stopfile):
            os.remove(stopfile)
    err = result.get("err")
    if err is None and not forced:
        disp.tick()
        out.write(disp.summary("done") + "\n")
        out.flush()
        return
    disp.fail()
    disp.tick()
    if isinstance(err, C.CorpusError):
        disp.failed = str(err)
        out.write(disp.summary("error") + "\n")
        raise err
    if forced or isinstance(err, KeyboardInterrupt):
        out.write(disp.summary("stopped") + "\n")
        raise KeyboardInterrupt()
    disp.failed = "%s: %s" % (type(err).__name__, err)
    out.write(disp.summary("error") + "\n")
    raise err
