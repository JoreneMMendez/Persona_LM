"""The terminal progress screen for `corpus` (persona_lm/corpus_ui.py). No network: the pipeline is replaced by a small fake that writes .part files."""
import io
import os
import re
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import corpus as C, corpus_ui as U  # noqa: E402
from persona_lm.pretty import ANSI_RE, vlen  # noqa: E402


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def display(langs=("en", "tl", "ceb", "taglish"), mix=True, width=100, mode="all", clock=None, **kw):
    return U.CorpusDisplay(list(langs), mode, kw.get("mb"), mix, FakeTTY(), color=False, width=width, clock=clock or Clock(), unicode_ok=True)


class Render(unittest.TestCase):
    def test_panel_lines_have_one_width_at_every_size(self):
        for w in (80, 100, 140):
            d = display(width=w)
            d.on_stage(1, 10, "x", "en")
            lines = d.render()
            panel = lines[:-1]
            self.assertIn("Ctrl+C", lines[-1])
            self.assertEqual(len(panel), 12)
            self.assertEqual(len({vlen(x) for x in panel}), 1, w)
            self.assertLessEqual(vlen(panel[0]), min(w - 1, 104))

    def test_all_ten_steps_are_listed_with_marks(self):
        d = display()
        d.on_stage(1, 10, "t", "en")
        d.on_stage(2, 10, "t", "en")
        d.on_stage(3, 10, "t", "tl")
        text = "\n".join(d.render())
        for name in ("Download English", "Download Tagalog", "Download Cebuano", "Download Taglish", "Clean Tagalog", "Check English", "Build training mix", "Check the files"):
            self.assertIn(name, text)
        self.assertIn("✓ Check English", text)
        self.assertIn("▸ Download Tagalog", text)
        self.assertIn("· Download Cebuano", text)

    def test_ascii_marks_when_the_console_has_no_unicode(self):
        d = U.CorpusDisplay(["en"], "all", None, False, FakeTTY(), color=False, width=100, clock=Clock(), unicode_ok=False)
        d.on_stage(1, 3, "t", "en")
        text = "\n".join(d.render())
        self.assertIn("> Download English", text)
        self.assertNotIn("▸", text)

    def test_download_shows_size_percent_speed_and_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                clock = Clock()
                d = display(("tl",), mix=False, clock=clock)
                d.on_stage(1, 4, "t", "tl")
                path = C.LANGS["tl"]["raw"] + ".part"
                os.makedirs(os.path.dirname(path))
                for i in range(1, 6):
                    with open(path, "wb") as f:
                        f.truncate(i * 30_000_000)
                    clock.t += 10
                    d.tick()
                text = ANSI_RE.sub("", "\n".join(d.render()))
                self.assertIn("150 / 300 MB", text)
                self.assertIn("50%", text)
                self.assertIn("3.0 MB/s", text)
                self.assertIn("ETA 0:00:50", text)
            finally:
                os.chdir(old)

    def test_custom_mb_is_the_target_and_zero_means_unknown_length(self):
        d = display(("tl",), mix=False, mb={"tl": 120})
        d.on_stage(1, 3, "t", "tl")
        self.assertEqual(d.watch()[1], 120_000_000)
        d2 = display(("tl",), mix=False, mb={"tl": 0})
        d2.on_stage(1, 3, "t", "tl")
        self.assertIsNone(d2.watch()[1])
        d3 = display(("ceb",), mix=False)
        d3.on_stage(1, 3, "t", "ceb")
        self.assertIsNone(d3.watch()[1])                           # Cebuano always downloads all of it

    def test_waiting_message_before_the_first_data(self):
        d = display(("en",), mix=False)
        d.on_stage(1, 3, "t", "en")
        self.assertIn("waiting for the first data", "\n".join(d.render()))

    def test_unknown_length_bar_moves(self):
        d = display(("ceb",), mix=False)
        self.assertNotEqual(d.bar(None, 30, 0.0), d.bar(None, 30, 1.0))
        self.assertEqual(vlen(d.bar(None, 30, 0.37)), 30)
        self.assertEqual(vlen(d.bar(0.5, 30, 0.0)), 30)

    def test_finished_and_failed_states(self):
        d = display(("en",), mix=False)
        d.on_stage(1, 3, "t", "en")
        d.on_stage(2, 3, "t", "en")
        d.on_stage(3, 3, "t", None)
        d.on_stage(3, 3, "Finished", None)
        text = "\n".join(d.render())
        self.assertIn("All done", text)
        self.assertNotIn("▸", text)
        self.assertNotIn("Ctrl+C", text)
        e = display(("en",), mix=False)
        e.on_stage(1, 3, "t", "en")
        e.fail()
        self.assertIn("x Download English", "\n".join(e.render()))

    def test_stopping_message(self):
        d = display(("en",), mix=False)
        d.on_stage(1, 3, "t", "en")
        d.stopping = True
        self.assertIn("stopping after this step", "\n".join(d.render()))

    def test_log_box_shows_only_the_last_lines(self):
        d = display(("en",), mix=False)
        d.on_stage(1, 3, "t", "en")
        for i in range(10):
            d.feed("line %d" % i)
        text = "\n".join(d.render())
        self.assertIn("line 9", text)
        self.assertIn("line 6", text)
        self.assertNotIn("line 5", text)

    def test_progress_bar_lines_replace_each_other(self):
        d = display(("en",), mix=False)
        cap = U.Capture(d.feed)
        cap.write("Resolving 10%\rResolving 50%\rResolving 100%\r\nnext line\n")
        self.assertEqual(list(d.log), ["Resolving 100%", "next line"])

    def test_capture_keeps_partial_line_until_finish(self):
        got = []
        cap = U.Capture(lambda t, r: got.append(t))
        cap.write("abc")
        self.assertEqual(got, [])
        cap.finish()
        self.assertEqual(got, ["abc"])
        self.assertFalse(cap.isatty())


class Gate(unittest.TestCase):
    def test_wanted_only_on_a_real_terminal(self):
        self.assertFalse(U.wanted(io.StringIO()))
        self.assertFalse(U.wanted(FakeTTY(), plain=True))
        os.environ["PERSONA_PLAIN"] = "1"
        try:
            self.assertFalse(U.wanted(FakeTTY()))
        finally:
            del os.environ["PERSONA_PLAIN"]

    def test_cmd_goes_plain_for_the_web_ui_and_plain_flag(self):
        import argparse
        p = argparse.ArgumentParser()
        C.add_arguments(p)
        seen = []
        orig_run, orig_ui, orig_wanted = C.run, U.run_with_display, U.wanted
        C.run = lambda *a, **k: seen.append("plain")
        U.run_with_display = lambda *a, **k: seen.append("screen")
        U.wanted = lambda *a, **k: True
        try:
            C.cmd(p.parse_args(["--lang", "en"]))
            C.cmd(p.parse_args(["--lang", "en", "--plain"]))
            C.cmd(p.parse_args(["--lang", "en", "--stopfile", "x", "--progress", "y"]))
        finally:
            C.run, U.run_with_display, U.wanted = orig_run, orig_ui, orig_wanted
        self.assertEqual(seen, ["screen", "plain", "plain"])


class Whole(unittest.TestCase):
    def fake_pipeline(self, behaviour):
        def fake_run(langs, mode="all", mb=None, mix=False, stopfile=None, progress=None, root=None):
            stages = C.plan_stages(langs, mode, mix)
            stop = C._Stop(stopfile)
            for i, st in enumerate(stages, 1):
                stop.check()
                C._write_progress(progress, i, len(stages), st["title"], st["lang"])
                print("[STEP %d/%d] %s" % (i, len(stages), st["title"]))
                print("noise from a library", file=sys.stderr)
                behaviour(i, st)
            C._write_progress(progress, len(stages), len(stages), "Finished", None)
        return fake_run

    def run_screen(self, behaviour, langs=("en", "ceb")):
        out = FakeTTY()
        orig = C.run
        C.run = self.fake_pipeline(behaviour)
        real = sys.stdout, sys.stderr
        try:
            with tempfile.TemporaryDirectory() as tmp:
                old = os.getcwd()
                os.chdir(tmp)
                try:
                    U.run_with_display(list(langs), "all", None, False, stream=out, color=False, poll=0.02)
                finally:
                    os.chdir(old)
        finally:
            C.run = orig
            self.assertEqual((sys.stdout, sys.stderr), real)          # always put back
            self.assertIsNone(C.STAGE_HOOK)
        return out.getvalue()

    def test_a_good_run_ends_with_a_summary_and_hides_library_noise_from_the_terminal(self):
        text = self.run_screen(lambda i, st: time.sleep(0.05))
        self.assertIn("finished", text)
        self.assertIn("all 5 steps", text)
        self.assertIn("\x1b[?25l", text)
        self.assertTrue(text.rstrip().endswith("HANDOFF.") or "Next:" in text)
        self.assertIn("\x1b[?25h", text)

    def test_a_pipeline_error_is_shown_and_raised(self):
        def boom(i, st):
            if i == 2:
                raise C.CorpusError("the network is down")
        out = FakeTTY()
        orig = C.run
        C.run = self.fake_pipeline(boom)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                old = os.getcwd()
                os.chdir(tmp)
                try:
                    with self.assertRaises(C.CorpusError):
                        U.run_with_display(["en"], "all", None, False, stream=out, color=False, poll=0.02)
                finally:
                    os.chdir(old)
        finally:
            C.run = orig
        self.assertIn("failed", out.getvalue())
        self.assertIn("the network is down", out.getvalue())
        self.assertIsNone(C.STAGE_HOOK)

    def test_hook_does_not_change_the_progress_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.json")
            seen = []
            C.STAGE_HOOK = lambda *a: seen.append(a)
            try:
                C._write_progress(path, 1, 3, "Download English", "en")
            finally:
                C.STAGE_HOOK = None
            self.assertEqual(seen, [(1, 3, "Download English", "en")])
            self.assertTrue(os.path.exists(path))

    def test_a_broken_hook_never_stops_the_run(self):
        def bad(*a):
            raise RuntimeError("display bug")
        C.STAGE_HOOK = bad
        try:
            C._write_progress(None, 1, 3, "t", "en")
        finally:
            C.STAGE_HOOK = None

    def _interrupt_after(self, *delays):
        import _thread

        def go():
            for d in delays:
                time.sleep(d)
                _thread.interrupt_main()
        threading.Thread(target=go, daemon=True).start()

    def _run_interrupted(self, behaviour, *delays):
        import signal
        out = FakeTTY()
        orig = C.run
        old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)         # a test runner may have SIGINT ignored
        self.addCleanup(signal.signal, signal.SIGINT, old_handler)
        C.run = self.fake_pipeline(behaviour)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                old = os.getcwd()
                os.chdir(tmp)
                try:
                    self._interrupt_after(*delays)
                    with self.assertRaises(KeyboardInterrupt):
                        U.run_with_display(["en"], "all", None, False, stream=out, color=False, poll=0.02)
                finally:
                    os.chdir(old)
        finally:
            C.run = orig
        self.assertIsNone(C.STAGE_HOOK)
        return out.getvalue()

    def test_one_ctrl_c_stops_after_the_step(self):
        def slow(i, st):
            time.sleep(0.4)
        text = self._run_interrupted(slow, 0.15)
        self.assertIn("stopping after this step", text)
        self.assertIn("stopped", text)

    def test_second_ctrl_c_quits_at_once(self):
        def stuck(i, st):
            time.sleep(1.5)
        t0 = time.time()
        text = self._run_interrupted(stuck, 0.15, 0.15)
        self.assertLess(time.time() - t0, 1.2)
        self.assertIn("stopped", text)


if __name__ == "__main__":
    unittest.main()
