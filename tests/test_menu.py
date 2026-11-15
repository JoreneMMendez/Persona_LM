import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import menu as M  # noqa: E402
from persona_lm import pretty as P  # noqa: E402
from persona_lm.model import PRESETS  # noqa: E402


class FakeTTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def make_project(root, size="s30m", step=18300, steps=36000):
    run = os.path.join(root, "work3", "run_pre")
    os.makedirs(run)
    meta = {"cfg": dict(PRESETS[size]), "hp": {"steps": steps}, "step": step}
    np.savez(os.path.join(run, "ckpt.npz"), meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
    data = os.path.join(root, "work3", "data_pre")
    os.makedirs(data)
    with open(os.path.join(data, "meta.json"), "w") as f:
        json.dump({"vocab_size": 8000}, f)
    with open(os.path.join(root, "work3", "tok.json"), "w") as f:
        f.write("{}")


def keys(*seq):
    it = iter(seq)
    return lambda: next(it)


class Scan(unittest.TestCase):
    def test_finds_runs_data_and_tokenizer_and_reads_the_size(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            f = M.scan(d)
            self.assertEqual(f["runs"][0]["path"], os.path.join("work3", "run_pre"))
            self.assertEqual((f["runs"][0]["size"], f["runs"][0]["step"], f["runs"][0]["steps"]), ("s30m", 18300, 36000))
            self.assertEqual(f["data"], [os.path.join("work3", "data_pre")])
            self.assertEqual(f["toks"], [os.path.join("work3", "tok.json")])

    def test_empty_folder_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            f = M.scan(d)
            self.assertEqual((f["runs"], f["data"], f["toks"]), ([], [], []))


class Screen(unittest.TestCase):
    def found(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        make_project(self.d.name)
        return M.scan(self.d.name)

    def test_three_boxes_that_line_up_and_fit_80_columns(self):
        for kw in ({"color": False}, {"color": True}, {"color": False, "unicode_ok": False}):
            m = M.Menu(FakeTTY(), width=80, **{k: v for k, v in kw.items()})
            lines = m.render(0, self.found())
            self.assertEqual(len(lines), P.PANEL_H)
            self.assertEqual(len({P.vlen(x) for x in lines}), 1)
            self.assertLessEqual(P.vlen(lines[0]), 79)
            text = "\n".join(lines)
            for want in ("Menu", "Details", "Project", "Train or resume a run", "s30m", "work3"):
                self.assertIn(want, text)

    def test_arrow_keys_wrap_and_enter_selects(self):
        m = M.Menu(FakeTTY(), color=False, width=100)
        f = self.found()
        self.assertEqual(m.select(keys("down", "down", "enter"), f), 2)
        self.assertEqual(m.select(keys("up", "enter"), f), M.QUIT)       # wraps to the last item
        self.assertEqual(m.select(keys("3"), f), 2)                      # a number picks at once
        self.assertEqual(m.select(keys("esc"), f), M.QUIT)
        self.assertEqual(m.select(keys("j", "j", "k", "enter"), f), 1)

    def test_screen_is_erased_between_key_presses(self):
        out = FakeTTY()
        M.Menu(out, color=False, width=100).select(keys("down", "enter"), self.found())
        self.assertEqual(out.getvalue().count("\x1b[%dF\x1b[J" % P.PANEL_H), 2)


class Collect(unittest.TestCase):
    def test_train_fills_in_what_was_found_and_enter_accepts(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            f = M.scan(d)
            asked = []
            argv = M.collect(M.ITEMS[0], f, ask=lambda q: asked.append(q) or "", say=lambda *_: None)
            self.assertEqual(argv, ["train", "--run", os.path.join("work3", "run_pre"), "--data-dir", os.path.join("work3", "data_pre"), "--size", "s30m"])
            self.assertIn("keep 36,000", asked[3])                       # blank steps keeps the saved total

    def test_bad_choice_is_asked_again_and_required_blank_backs_out(self):
        f = {"runs": [], "data": [], "toks": [], "torch": False}
        answers = iter(["r", "d", "huge", "s10m", "500"])
        argv = M.collect(M.ITEMS[0], f, ask=lambda q: next(answers), say=lambda *_: None)
        self.assertEqual(argv, ["train", "--run", "r", "--data-dir", "d", "--size", "s10m", "--steps", "500"])
        self.assertIsNone(M.collect(M.ITEMS[1], f, ask=lambda q: "", say=lambda *_: None))

    def test_ctrl_c_goes_back(self):
        def boom(q):
            raise KeyboardInterrupt
        self.assertIsNone(M.collect(M.ITEMS[0], {"runs": [], "data": [], "toks": [], "torch": False}, ask=boom, say=lambda *_: None))


class Loop(unittest.TestCase):
    def test_runs_the_command_and_comes_back(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            calls = []
            asked = iter(["", "", "", "", ""])                           # train: 4 answers, then "press Enter"
            M.run_menu(d, stream=FakeTTY(), read=keys("enter", "q"), ask=lambda q: next(asked), call=calls.append, width=100, color=False, interactive=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "train")

    def test_a_failing_command_does_not_close_the_menu(self):
        def bad(argv):
            raise RuntimeError("boom")
        with tempfile.TemporaryDirectory() as d:
            asked = iter([""])
            self.assertEqual(M.run_menu(d, stream=FakeTTY(), read=keys("5", "q"), ask=lambda q: next(asked), call=bad, width=100, color=False, interactive=True), 0)

    def test_numbered_fallback_when_not_interactive(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            answers = iter(["7", "", "q"])                               # 7 = GPU check (no questions), Enter, then quit
            M.run_menu(d, stream=io.StringIO(), ask=lambda q: next(answers), call=calls.append, interactive=False)
            self.assertEqual(calls, [["check-gpu"]])


class Cli(unittest.TestCase):
    def test_menu_subcommand_exists_and_no_arguments_means_menu(self):
        from persona_lm import cli
        self.assertEqual(cli.build_parser().parse_args(["menu"]).cmd, "menu")
        seen = []
        orig = cli.cmd_menu
        cli.cmd_menu = lambda a: seen.append(a)
        try:
            cli.main([])
        finally:
            cli.cmd_menu = orig
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
