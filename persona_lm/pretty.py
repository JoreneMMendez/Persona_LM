"""A nicer terminal screen for training: a live dashboard of three rounded boxes (one big, two small).

    ╭─ Training ────────────────────╮ ╭─ Validation ─────────╮
    │ step, progress bar, loss, lr  │ │ val, best, trend,    │
    │ and a loss-history chart      │ │ overfitting check    │
    │                               │ ╰──────────────────────╯
    │                               │ ╭─ Speed & Time ───────╮
    │                               │ │ tok/s, elapsed, ETA, │
    │                               │ │ finish time, GPU     │
    ╰───────────────────────────────╯ ╰──────────────────────╯

How it works: training still writes exactly the same log lines as before. This module only CHANGES HOW THEY ARE SHOWN, and only when
the output is a real terminal. When the output goes to a file or a pipe (the web UI's training runs, tests, `> log.txt`) the plain
lines are printed unchanged, so nothing that reads those logs can break. No extra packages: ANSI colours only.

Switch it off with `--plain` or the environment variable PERSONA_PLAIN=1 (NO_COLOR=1 keeps the dashboard but removes the colours).
"""
import math
import os
import re
import shutil
import sys
import time

STEP_RE = re.compile(r"^step (\d+)/(\d+) loss (\S+?)(?: val (\S+))? lr (\S+)\s+(\d+) tok/s\s+ETA (\S+) h\s*$")
SAVED_RE = re.compile(r"^saved checkpoint at step (\d+)")
BACKEND_RE = re.compile(r"^backend: PyTorch (\S+) on (\S+) \((.*)\), precision (\S+)")
RESUME_RE = re.compile(r"^resumed at step (\d+)")
NEWRUN_RE = re.compile(r"^new run: ([\d.]+)M parameters")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MIN_WIDTH = 80
MIN_HEIGHT = 16                 # the dashboard is PANEL_H lines tall; shorter windows get the plain lines
PANEL_H = 12                    # lines on screen: 6 + 6 on the right, one 12-line box on the left
MAX_W = 104
WARN_ABOVE_BEST = 0.03          # same 3 % rule as the web UI's Training page


def enable_vt():
    """Turns on ANSI escape codes in the Windows console (Windows 10 and later). Returns True when escape codes will work."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004)) or bool(mode.value & 0x0004)
    except Exception:
        return False


def _can_encode(stream, text):
    try:
        text.encode(getattr(stream, "encoding", None) or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def hms(seconds):
    s = int(max(0, seconds))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def vlen(text):
    """Length of a string as it looks on screen (colour codes take no room)."""
    return len(ANSI_RE.sub("", text))


def fit(text, w, ell="…"):
    """Pads with spaces, or cuts with an ellipsis, so that the text takes exactly w columns."""
    n = vlen(text)
    if n <= w:
        return text + " " * (w - n)
    plain = ANSI_RE.sub("", text)
    return (plain[:max(0, w - len(ell))] + ell)[:w] if w > 0 else ""


def spread(left, right, w):
    """left text at the left edge, right text at the right edge of a w-column line."""
    gap = w - vlen(left) - vlen(right)
    return left + " " * max(1, gap) + right


class Skin:
    """Colours, glyphs and the rounded box shared by the training dashboard and the menu (falls back to ASCII on consoles without box characters)."""

    def __init__(self, stream, color=True, unicode_ok=None):
        self.out = stream
        self.color = color
        u = self.unicode = _can_encode(self.out, "█░▁↓↑·─╭╮╰╯│…▸") if unicode_ok is None else unicode_ok
        self.full, self.empty, self.arrow_dn, self.arrow_up, self.dot, self.rule, self.ell, self.ptr = ("█", "░", "↓", "↑", "·", "─", "…", "▸") if u else ("#", "-", "v", "^", "-", "-", "~", ">")
        self.tl, self.tr, self.bl, self.br, self.hz, self.vt = ("╭", "╮", "╰", "╯", "─", "│") if u else ("+", "+", "+", "+", "-", "|")

    def c(self, text, code):
        return "\x1b[%sm%s\x1b[0m" % (code, text) if self.color else text

    def box(self, title, lines, w, h):
        """A rounded box w columns wide and h lines tall with the title in its top border; `lines` fill the inside (cut or padded to fit)."""
        b = lambda t: self.c(t, "90")
        head = " %s " % title
        top = b(self.tl + self.hz) + self.c(head, "1;36") + b(self.hz * max(0, w - 2 - 1 - len(head)) + self.tr)
        out = [top]
        inner = w - 4
        for i in range(h - 2):
            out.append(b(self.vt) + " " + fit(lines[i] if i < len(lines) else "", inner, self.ell) + " " + b(self.vt))
        out.append(b(self.bl + self.hz * (w - 2) + self.br))
        return out


class TrainDisplay(Skin):
    def __init__(self, title, stream=None, color=True, width=None, clock=time.time, unicode_ok=None):
        Skin.__init__(self, stream or sys.stdout, color, unicode_ok)
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.clock = clock
        self.t0 = clock()
        self.spark = "▁▂▃▄▅▆▇█" if self.unicode else "_.-=+*#@"
        self.blocks = " ▁▂▃▄▅▆▇█" if self.unicode else " .:-=+*#@"      # 0..8 eighths of a chart cell
        self.n = 0                      # lines of the live panel on screen
        self.step = self.total = 0
        self.steps, self.losses, self.vals = [], [], []
        self.best = None                # (val loss, step)
        self.last = None                # the last parsed record
        self.prev_val = None
        self.finished = False
        self.info = {}                  # what the start-up messages told us: device, precision, parameters, resumed step
        self._write("\n" + self.c(" %s " % title, "1;36") + "\n" + self.c(self.rule * min(self.width - 1, MAX_W - 1), "90") + "\n")

    # ---------------------------------------------------------------- helpers
    def _write(self, text):
        self.out.write(text)
        self.out.flush()

    def _erase(self):
        if self.n:
            self._write("\x1b[%dF\x1b[J" % self.n)
            self.n = 0

    def _draw(self):
        lines = self.render()
        self._write("\n".join(lines) + "\n")
        self.n = len(lines)

    def sparkline(self, values, n=30):
        v = [x for x in values[-n:] if math.isfinite(x)]
        if len(v) < 2:
            return ""
        lo, hi = min(v), max(v)
        span = (hi - lo) or 1.0
        top = len(self.spark) - 1
        return "".join(self.spark[int(round((x - lo) / span * top))] for x in v)

    def chart(self, values, width, rows):
        """A filled bar chart of the loss history, `rows` lines tall and `width` columns wide: (lines top first, low, high).
        Longer histories are averaged down to fit, so the whole run is always visible."""
        v = [x for x in values if math.isfinite(x)]
        if not v or width < 2:
            return [" " * max(0, width)] * rows, 0.0, 0.0
        if len(v) > width:
            n = len(v)
            v = [sum(v[i * n // width:(i + 1) * n // width]) / ((i + 1) * n // width - i * n // width) for i in range(width)]
        lo, hi = min(v), max(v)
        span = (hi - lo) or 1.0
        levels = [int(round((x - lo) / span * (rows * 8 - 1))) + 1 for x in v]
        grid = []
        for r in range(rows - 1, -1, -1):                                  # top line first
            cells = "".join(self.blocks[max(0, min(8, lv - r * 8))] for lv in levels)
            grid.append(self.c(cells, "36") + " " * (width - len(levels)))
        return grid, lo, hi

    # ---------------------------------------------------------------- the dashboard
    def render(self):
        r = self.last
        W = min(self.width - 1, MAX_W)
        lw = W * 57 // 100                  # the big box on the left
        rw = W - lw - 1                     # the two small boxes on the right
        li = lw - 4                         # text room inside the big box
        ri = rw - 4
        pct = self.step / self.total if self.total else 0.0

        # ---- big box: progress, loss, loss history
        digits = len(format(self.total, ","))
        l_step = "step %s / %s" % (format(self.step, ",").rjust(digits), format(self.total, ","))
        barw = max(8, li - 5)
        filled = int(round(pct * barw))
        bar = self.c(self.full * filled, "32") + self.c(self.empty * (barw - filled), "90") + self.c("%3d%%" % int(pct * 100), "1")
        loss_line = spread(self.c("loss ", "90") + self.c(r["loss"], "1;37"), self.c("lr ", "90") + r["lr"], li)
        lab = 5                             # width of the y-axis labels
        grid, lo, hi = self.chart(self.losses, li - lab - 1, 5)
        chart = []
        for i, row in enumerate(grid):
            tag = ("%.2f" % hi) if i == 0 else (("%.2f" % lo) if i == len(grid) - 1 else "")
            chart.append(self.c(tag.rjust(lab), "90") + " " + row)
        axis = " " * (lab + 1) + spread(self.c(str(self.steps[0]) if self.steps else "", "90"), self.c("step %d" % self.step, "90"), li - lab - 1)
        big = [l_step, bar, "", loss_line] + chart + [axis]

        # ---- small box 1: validation
        if self.vals:
            v = self.vals[-1]
            arrow = ""
            if self.prev_val is not None and v < self.prev_val - 1e-9:
                arrow = " " + self.c(self.arrow_dn, "32")
            elif self.prev_val is not None and v > self.prev_val + 1e-9:
                arrow = " " + self.c(self.arrow_up, "31")
            vl = [self.c("val   ", "90") + self.c("%.4f" % v, "1;33") + arrow,
                  self.c("best  ", "90") + self.c("%.4f" % self.best[0], "32") + self.c("  step %d" % self.best[1], "90"),
                  self.c(self.sparkline(self.vals, ri), "33")]
            if v > self.best[0] * (1 + WARN_ABOVE_BEST):
                vl.append(self.c("! overfitting? +%.0f%% vs best" % ((v / self.best[0] - 1) * 100), "1;31"))
            else:
                vl.append(self.c("on track", "32"))
        else:
            vl = [self.c("waiting for the first", "90"), self.c("validation check...", "90")]

        # ---- small box 2: speed and time
        eta = float(r["eta"]) * 3600
        sl = [self.c("speed   ", "90") + self.c("%s tok/s" % format(int(r["tps"]), ","), "36"),
              self.c("elapsed ", "90") + hms(self.clock() - self.t0),
              self.c("ETA     ", "90") + self.c(hms(eta), "1") + self.c("  ~%s" % time.strftime("%H:%M", time.localtime(self.clock() + eta)), "90")]
        if self.info.get("gpu"):
            sl.append(self.c(self.info["gpu"], "90"))

        left = self.box("Training", big, lw, PANEL_H)
        right = self.box("Validation", vl, rw, 6) + self.box("Speed & Time", sl, rw, PANEL_H - 6)
        return [a + " " + b for a, b in zip(left, right)]

    # ---------------------------------------------------------------- the log function handed to training
    def _learn(self, msg):
        """Keeps what start-up messages say (device, precision, size, resumed step) for the dashboard; the message is still printed."""
        m = BACKEND_RE.match(msg)
        if m:
            name = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", m.group(3)).replace(" Laptop GPU", " Laptop")
            self.info["gpu"] = "%s %s %s" % (name, self.dot, m.group(4))
        m = RESUME_RE.match(msg)
        if m:
            self.info["resumed"] = int(m.group(1))
        m = NEWRUN_RE.match(msg)
        if m:
            self.info["params"] = m.group(1)

    def log(self, msg):
        msg = str(msg)
        m = STEP_RE.match(msg)
        if m:
            self._erase()
            step, total, loss, val, lr, tps, eta = m.groups()
            self.step, self.total = int(step), int(total)
            try:
                self.losses.append(float(loss))
                self.steps.append(self.step)
            except ValueError:
                pass
            self.prev_val = self.vals[-1] if self.vals else None
            if val is not None:
                v = float(val)
                self.vals.append(v)
                if self.best is None or v < self.best[0]:
                    self.best = (v, self.step)
            self.last = {"loss": loss, "val": val, "lr": lr, "tps": tps, "eta": eta}
            self._draw()
            return
        self._erase()
        self._learn(msg)
        s = SAVED_RE.match(msg)
        if s and self.last is not None:
            self.finished = True
            self._write(self.summary(int(s.group(1)), msg) + "\n")
            return
        self._write(msg + "\n")
        if self.last is not None:
            self._draw()

    def summary(self, step, msg):
        rule = self.c(self.rule * min(self.width - 1, MAX_W - 1), "90")
        lines = [rule, " " + self.c("finished" if "stopped early" not in msg else "stopped early", "1;32") + "  " + msg, ]
        if self.vals and self.best:
            lines.append(" best validation loss %s at step %d   last %.4f   time %s" % (self.c("%.4f" % self.best[0], "1;32"), self.best[1], self.vals[-1], hms(self.clock() - self.t0)))
            if self.best[1] < step:
                lines.append(" note: the best step (%d) is earlier than the last one. The trainer only keeps the latest two saves." % self.best[1])
        return "\n".join(lines)


def make_log(title, plain=False, stream=None):
    """Returns the function to pass as `log=` to training: the live dashboard on a real terminal, plain `print` everywhere else."""
    out = stream or sys.stdout
    if plain or os.environ.get("PERSONA_PLAIN"):
        return print
    isatty = getattr(out, "isatty", None)
    if not (isatty and isatty()):
        return print
    size = shutil.get_terminal_size((100, 24))
    if size.columns < MIN_WIDTH or size.lines < MIN_HEIGHT or os.environ.get("TERM") == "dumb":
        return print
    if not enable_vt():
        return print
    return TrainDisplay(title, out, color=not os.environ.get("NO_COLOR")).log
