import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import pretty as P  # noqa: E402


class FakeTTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def line(step, loss, val=None, total=105):
    return "step %d/%d loss %.4f%s lr 1.00e-04  16203 tok/s  ETA 0.1 h" % (step, total, loss, (" val %.4f" % val) if val is not None else "")


class Display(unittest.TestCase):
    def make(self, **kw):
        self.out = FakeTTY()
        self.t = [0.0]
        return P.TrainDisplay("title", self.out, color=kw.get("color", False), width=100, clock=lambda: self.t[0], unicode_ok=kw.get("unicode_ok", True))

    def test_panel_shows_progress_loss_speed_and_eta(self):
        d = self.make()
        self.t[0] = 125
        d.log(line(45, 1.2345))
        text = "\n".join(d.render())
        for want in ("45/105", "42%", "1.2345", "16,203 tok/s", "elapsed 0:02:05", "ETA 0:06:00"):
            self.assertIn(want, text)

    def test_best_val_and_arrows(self):
        d = self.make()
        d.log(line(15, 2.0, 1.5))
        d.log(line(30, 1.5, 1.0))
        d.log(line(45, 1.2, 1.1))
        text = "\n".join(d.render())
        self.assertIn("best 1.0000 (step 30)", text)
        self.assertEqual(d.best, (1.0, 30))
        self.assertIn("↑", text)           # val went up since the last check

    def test_overfitting_warning_only_when_val_is_3_percent_above_best(self):
        d = self.make()
        d.log(line(15, 1.0, 1.00))
        d.log(line(30, 0.9, 1.02))
        self.assertNotIn("overfitting", "\n".join(d.render()))
        d.log(line(45, 0.8, 1.10))
        self.assertIn("overfitting", "\n".join(d.render()))

    def test_other_messages_are_printed_and_the_panel_is_redrawn_under_them(self):
        d = self.make()
        d.log(line(15, 2.0))
        d.log("fp16 overflow, update skipped")
        out = self.out.getvalue()
        self.assertIn("fp16 overflow, update skipped\n", out)
        self.assertEqual(out.count("15/105"), 2)                    # drawn, erased, message printed, drawn again under it
        self.assertEqual(out.count("\x1b[3F\x1b[J"), 1)             # the 3-line panel was erased exactly once

    def test_summary_on_save(self):
        d = self.make()
        d.log(line(15, 2.0, 1.5))
        d.log(line(30, 1.5, 1.0))
        d.log("saved checkpoint at step 30")
        self.assertIn("best validation loss 1.0000 at step 30", self.out.getvalue())
        self.assertTrue(d.finished)

    def test_ascii_fallback_when_the_console_cannot_show_blocks(self):
        d = self.make(unicode_ok=False)
        d.log(line(15, 2.0))
        d.log(line(30, 1.5))
        text = self.out.getvalue()
        self.assertNotIn("█", text)
        self.assertTrue(all(ord(ch) < 128 for ch in "\n".join(d.render())))

    def test_colours_can_be_off(self):
        d = self.make(color=False)
        d.log(line(15, 2.0, 1.5))
        self.assertNotIn("\x1b[3", "\n".join(d.render()))

    def test_nan_loss_does_not_crash(self):
        d = self.make()
        d.log("step 10/105 loss nan lr 1.00e-04  100 tok/s  ETA 0.1 h")
        self.assertIn("nan", "\n".join(d.render()))

    def test_lines_fit_a_80_column_terminal(self):
        d = P.TrainDisplay("t", FakeTTY(), color=False, width=80, clock=lambda: 0.0)
        for i, v in enumerate((1.0, 0.9, 1.5), 1):
            d.log(line(i * 35, 1.0, v, total=105))
        for ln in d.render():
            self.assertLessEqual(len(ln), 79, ln)


class MakeLog(unittest.TestCase):
    def test_plain_print_when_not_a_terminal(self):
        self.assertIs(P.make_log("t", stream=io.StringIO()), print)

    def test_plain_flag_and_environment_switch_it_off(self):
        self.assertIs(P.make_log("t", plain=True, stream=FakeTTY()), print)
        os.environ["PERSONA_PLAIN"] = "1"
        try:
            self.assertIs(P.make_log("t", stream=FakeTTY()), print)
        finally:
            del os.environ["PERSONA_PLAIN"]

    def test_train_flag_exists(self):
        from persona_lm.cli import build_parser
        self.assertTrue(build_parser().parse_args(["train", "--run", "r", "--data-dir", "d", "--plain"]).plain)


if __name__ == "__main__":
    unittest.main()
