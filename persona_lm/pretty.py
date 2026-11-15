"""A nicer terminal screen for training: a live panel with a progress bar, loss, validation (best so far), speed, ETA and a loss sparkline.

How it works: training still writes exactly the same log lines as before. This module only CHANGES HOW THEY ARE SHOWN, and only when
the output is a real terminal. When the output goes to a file or a pipe (the web UI's training runs, tests, `> log.txt`) the plain
lines are printed unchanged, so nothing that reads those logs can break. No extra packages: ANSI colours only.

Switch it off with `--plain` or the environment variable PERSONA_PLAIN=1 (NO_COLOR=1 keeps the panel but removes the colours).
"""
import os
import re
import shutil
import sys
import time

STEP_RE = re.compile(r"^step (\d+)/(\d+) loss (\S+?)(?: val (\S+))? lr (\S+)\s+(\d+) tok/s\s+ETA (\S+) h\s*$")
SAVED_RE = re.compile(r"^saved checkpoint at step (\d+)")
MIN_WIDTH = 80
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


class TrainDisplay:
    def __init__(self, title, stream=None, color=True, width=None, clock=time.time, unicode_ok=None):
        self.out = stream or sys.stdout
        self.color = color
        self.width = width or shutil.get_terminal_size((100, 24)).columns
        self.clock = clock
        self.t0 = clock()
        u = _can_encode(self.out, "█░▁↓↑·─") if unicode_ok is None else unicode_ok
        self.full, self.empty, self.arrow_dn, self.arrow_up, self.dot, self.rule = ("█", "░", "↓", "↑", "·", "─") if u else ("#", "-", "v", "^", "-", "-")
        self.spark = "▁▂▃▄▅▆▇█" if u else "_.-=+*#@"
        self.n = 0                      # lines of the live panel on screen
        self.step = self.total = 0
        self.losses, self.vals = [], []
        self.best = None                # (val loss, step)
        self.last = None                # the last parsed record
        self.prev_val = None
        self.finished = False
        self._write("\n" + self.c(" %s " % title, "1;36") + "\n" + self.c(self.rule * min(self.width - 1, 72), "90") + "\n")

    # ---------------------------------------------------------------- helpers
    def c(self, text, code):
        return "\x1b[%sm%s\x1b[0m" % (code, text) if self.color else text

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
        v = values[-n:]
        if len(v) < 2:
            return ""
        lo, hi = min(v), max(v)
        span = (hi - lo) or 1.0
        top = len(self.spark) - 1
        return "".join(self.spark[int(round((x - lo) / span * top))] for x in v)

    # ---------------------------------------------------------------- the panel
    def render(self):
        r = self.last
        pct = self.step / self.total if self.total else 0.0
        barw = max(10, min(40, self.width - 34))
        filled = int(round(pct * barw))
        bar = self.c(self.full * filled, "32") + self.c(self.empty * (barw - filled), "90")
        l1 = " step %s/%d  %s  %3d%%" % (str(self.step).rjust(len(str(self.total))), self.total, bar, int(pct * 100))
        loss = self.c("loss %s" % r["loss"], "1")
        l2 = " %s" % loss
        if r["val"] is not None:
            v = float(r["val"])
            arrow = ""
            if self.prev_val is not None and v < self.prev_val - 1e-9:
                arrow = " " + self.c(self.arrow_dn, "32")
            elif self.prev_val is not None and v > self.prev_val + 1e-9:
                arrow = " " + self.c(self.arrow_up, "31")
            l2 += "   val %s%s" % (self.c(r["val"], "1;33"), arrow)
        elif self.vals:
            l2 += "   val %.4f" % self.vals[-1]
        if self.best:
            l2 += "   best %s (step %d)" % (self.c("%.4f" % self.best[0], "32"), self.best[1])
        l2 += "   lr %s" % r["lr"]
        l3 = " %s   elapsed %s   ETA %s" % (self.c("%s tok/s" % format(int(r["tps"]), ","), "36"), hms(self.clock() - self.t0), hms(float(r["eta"]) * 3600))
        lines = [l1, l2, l3]
        sp = self.sparkline(self.losses)
        if sp:
            lines.append(" loss " + sp)
        if self.best and self.vals and self.vals[-1] > self.best[0] * (1 + WARN_ABOVE_BEST):
            lines.append(" " + self.c("! val is %.0f%% above its best: overfitting? Ctrl+C keeps the latest save" % ((self.vals[-1] / self.best[0] - 1) * 100), "1;31"))
        return lines

    # ---------------------------------------------------------------- the log function handed to training
    def log(self, msg):
        msg = str(msg)
        m = STEP_RE.match(msg)
        if m:
            self._erase()
            step, total, loss, val, lr, tps, eta = m.groups()
            self.step, self.total = int(step), int(total)
            try:
                self.losses.append(float(loss))
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
        s = SAVED_RE.match(msg)
        if s and self.last is not None:
            self.finished = True
            self._write(self.summary(int(s.group(1)), msg) + "\n")
            return
        self._write(msg + "\n")
        if self.last is not None:
            self._draw()

    def summary(self, step, msg):
        rule = self.c(self.rule * min(self.width - 1, 72), "90")
        lines = [rule, " " + self.c("finished" if "stopped early" not in msg else "stopped early", "1;32") + "  " + msg, ]
        if self.vals and self.best:
            lines.append(" best validation loss %s at step %d   last %.4f   time %s" % (self.c("%.4f" % self.best[0], "1;32"), self.best[1], self.vals[-1], hms(self.clock() - self.t0)))
            if self.best[1] < step:
                lines.append(" note: the best step (%d) is earlier than the last one. The trainer only keeps the latest two saves." % self.best[1])
        return "\n".join(lines)


def make_log(title, plain=False, stream=None):
    """Returns the function to pass as `log=` to training: the live panel on a real terminal, plain `print` everywhere else."""
    out = stream or sys.stdout
    if plain or os.environ.get("PERSONA_PLAIN"):
        return print
    isatty = getattr(out, "isatty", None)
    if not (isatty and isatty()):
        return print
    if shutil.get_terminal_size((100, 24)).columns < MIN_WIDTH or os.environ.get("TERM") == "dumb":
        return print
    if not enable_vt():
        return print
    return TrainDisplay(title, out, color=not os.environ.get("NO_COLOR")).log
