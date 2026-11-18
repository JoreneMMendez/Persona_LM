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
        f.write('{"merges": [], "specials": []}')


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
            self.assertEqual(len(lines), M.MENU_H)
            self.assertEqual(len({P.vlen(x) for x in lines}), 1)
            self.assertLessEqual(P.vlen(lines[0]), 79)
            text = "\n".join(lines)
            for want in ("Menu", "Details", "Your project", "START HERE", "Teach your AI", "s30m", "work3", "Next suggestion"):
                self.assertIn(want, text)

    def test_arrow_keys_wrap_and_enter_selects(self):
        m = M.Menu(FakeTTY(), color=False, width=100)
        f = self.found()
        D = M.DISPLAY                                                    # keys follow the order shown on screen
        self.assertEqual(m.select(keys("down", "down", "enter"), f), D[2])
        self.assertEqual(m.select(keys("up", "enter"), f), M.QUIT)       # wraps to the last item
        self.assertEqual(m.select(keys("3"), f), D[2])                   # a number picks the item shown with that number
        self.assertEqual(m.select(keys("esc"), f), M.QUIT)
        self.assertEqual(m.select(keys("j", "j", "k", "enter"), f), D[1])
        self.assertEqual(m.select(keys("enter"), f), M.START)

    def test_screen_is_erased_between_key_presses(self):
        out = FakeTTY()
        M.Menu(out, color=False, width=100).select(keys("down", "enter"), self.found())
        self.assertEqual(out.getvalue().count("\x1b[%dF\x1b[J" % M.MENU_H), 1)      # the choice itself leaves the menu on screen


class Collect(unittest.TestCase):
    def test_train_fills_in_what_was_found_and_enter_accepts_everything(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            f = M.scan(d)
            argv = M.collect(M.ITEMS[0], f, ask=lambda q: "", say=lambda *_: None)
            self.assertEqual(argv, ["train", "--run", os.path.join("work3", "run_pre"), "--data-dir", os.path.join("work3", "data_pre"), "--size", "s30m"])

    def test_bad_choice_is_asked_again_and_required_blank_backs_out(self):
        f = {"runs": [], "data": [], "toks": [], "torch": False}
        answers = iter(["r", "d", "huge", "s10m", "", ""])        # run, data, size (bad, then good), init-from, then Enter on the settings screen
        argv = M.collect(M.ITEMS[0], f, ask=lambda q: next(answers), say=lambda *_: None)
        self.assertEqual(argv, ["train", "--run", "r", "--data-dir", "d", "--size", "s10m"])
        self.assertIsNone(M.collect(M.ITEMS[1], f, ask=lambda q: "", say=lambda *_: None))

    def test_only_changed_settings_become_flags(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            answers = iter(["", "", "", "", "5", "0.2", "9", "100", ""])    # 4 questions, then weight_decay=0.2, eval_every=100, Enter
            argv = M.collect(M.ITEMS[0], M.scan(d), ask=lambda q: next(answers), say=lambda *_: None)
            self.assertEqual(argv[-4:], ["--weight-decay", "0.2", "--eval-every", "100"])
            self.assertNotIn("--lr", argv)

    def test_the_flags_it_builds_are_accepted_by_the_real_train_command(self):
        from persona_lm.cli import build_parser
        changed, _ = edit(["p", "2", ""])
        argv = ["train", "--run", "r", "--data-dir", "d"] + [w for k, v in changed.items() for w in ("--" + k.replace("_", "-"), str(v))]
        a = build_parser().parse_args(argv)
        self.assertEqual((a.lr, a.min_lr, a.warmup, a.steps), (1e-4, 1e-5, 50, 3000))

    def test_init_from_is_passed_when_given(self):
        f = {"runs": [], "data": [], "toks": [], "torch": False}
        answers = iter(["r", "d", "s10m", "runs/pre/ckpt.npz", ""])
        argv = M.collect(M.ITEMS[0], f, ask=lambda q: next(answers), say=lambda *_: None)
        self.assertEqual(argv[-2:], ["--init-from", "runs/pre/ckpt.npz"])

    def test_ctrl_c_goes_back(self):
        def boom(q):
            raise KeyboardInterrupt
        self.assertIsNone(M.collect(M.ITEMS[0], {"runs": [], "data": [], "toks": [], "torch": False}, ask=boom, say=lambda *_: None))


BASE = {"steps": 36000, "lr": 6e-4, "min_lr": 6e-5, "warmup": 100, "weight_decay": 0.1, "clip": 1.0, "batch": 2, "accum": 2, "eval_every": 200}


def edit(answers, step=0, base=BASE):
    it, said = iter(answers), []
    changed = M.edit_settings(dict(base), 4096, step, ask=lambda q: next(it), say=said.append)
    return changed, "\n".join(said)


class Settings(unittest.TestCase):
    def test_enter_keeps_everything(self):
        self.assertEqual(edit([""])[0], {})

    def test_every_setting_is_explained(self):
        for d in M.SETTINGS:
            for field in ("short", "what", "up", "down", "affects", "tip"):
                self.assertTrue(d[field], (d["key"], field))
        self.assertEqual({d["key"] for d in M.SETTINGS} - set(M.DEFAULT_HP), set())      # every name is a real training setting

    def test_changing_a_value_shows_the_explanation_and_returns_only_that(self):
        changed, text = edit(["5", "0.2", ""])
        self.assertEqual(changed, {"weight_decay": 0.2})
        for want in ("brake on memorizing", "What it does", "Raise it", "Lower it", "Affects", "(was 0.1)"):
            self.assertIn(want, text)

    def test_numbers_may_have_commas_and_exponents(self):
        self.assertEqual(edit(["1", "50,000", "2", "3e-4", ""])[0], {"steps": 50000, "lr": 3e-4})

    def test_bad_values_are_refused_and_nothing_changes(self):
        for bad in ("abc", "-1", "5"):                                   # not a number, below range, above range for weight_decay
            changed, text = edit(["5", bad if bad != "5" else "5", ""])
            self.assertEqual(changed, {}, bad)
        self.assertEqual(edit(["1", "5000", ""], step=19403)[0], {})       # steps must be beyond the step the run is at
        self.assertEqual(edit(["3", "0.01", ""])[0], {})                   # min_lr may not be above lr
        self.assertEqual(edit(["2", "1e-5", ""])[0], {})                   # lr may not be below min_lr

    def test_preset_and_reset(self):
        changed, text = edit(["p", "1", ""])
        self.assertEqual(changed, {"weight_decay": 0.2, "lr": 3e-4, "min_lr": 3e-5, "eval_every": 100})
        self.assertIn("Heads-up", edit(["p", "1", ""], step=19403)[1])      # on a resumed run the lr-curve jump is explained
        self.assertEqual(edit(["p", "1", "r", ""])[0], {})

    def test_tokens_seen_is_shown(self):
        self.assertIn("16,384 tokens", edit([""])[1])
        self.assertIn("589,824,000", edit([""])[1])

    def test_junk_input_is_answered_not_crashed(self):
        changed, text = edit(["zzz", "99", ""])
        self.assertEqual(changed, {})
        self.assertIn("type a number", text)


class Confirm(unittest.TestCase):
    def test_train_says_what_it_will_do_and_n_cancels(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            f, said = M.scan(d), []
            argv = ["train", "--run", os.path.join("work3", "run_pre")]
            self.assertFalse(M.confirm_train(argv, f, ask=lambda q: "n", say=said.append))
            self.assertIn("CONTINUE", said[0])
            self.assertIn("18,300 of 36,000", said[0])
            self.assertTrue(M.confirm_train(argv, f, ask=lambda q: "", say=lambda *_: None))
            said2 = []
            M.confirm_train(["train", "--run", "x"], f, ask=lambda q: "y", say=said2.append)
            self.assertIn("NEW run", said2[0])

    def test_declining_does_not_run_anything(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            calls, asked = [], iter(["", "", "", "", "", "n", ""])
            M.run_menu(d, stream=FakeTTY(), read=keys(M.number_key(0), "q"), ask=lambda q: next(asked), call=calls.append, width=100, color=False, interactive=True)
            self.assertEqual(calls, [])


class Loop(unittest.TestCase):
    def test_runs_the_command_and_comes_back(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            calls = []
            asked = iter([""] * 7)                                         # train: 4 questions, settings Enter, "Start now?", "press Enter"
            M.run_menu(d, stream=FakeTTY(), read=keys(M.number_key(0), "q"), ask=lambda q: next(asked), call=calls.append, width=100, color=False, interactive=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "train")

    def test_a_failing_command_does_not_close_the_menu(self):
        def bad(argv):
            raise RuntimeError("boom")
        with tempfile.TemporaryDirectory() as d:
            asked = iter([""])
            self.assertEqual(M.run_menu(d, stream=FakeTTY(), read=keys(M.number_key([i["cmd"] for i in M.ITEMS].index("ui")), "q"), ask=lambda q: next(asked), call=bad, width=100, color=False, interactive=True), 0)

    def test_numbered_fallback_when_not_interactive(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            answers = iter([M.number_key([i["cmd"] for i in M.ITEMS].index("check-gpu")), "", "q"])   # the GPU check (no questions), Enter, then quit
            M.run_menu(d, stream=io.StringIO(), ask=lambda q: next(answers), call=calls.append, interactive=False)
            self.assertEqual(calls, [["check-gpu"]])


class LanguageData(unittest.TestCase):
    def item(self):
        return [it for it in M.ITEMS if it["cmd"] == "corpus"][0]

    def test_item_is_in_the_menu_before_quit(self):
        self.assertLess(M.ITEMS.index(self.item()), M.QUIT)
        self.assertIsNone(M.ITEMS[M.QUIT]["cmd"])

    def test_defaults_give_all_four_languages_and_no_mix(self):
        argv = M.collect(self.item(), {"runs": [], "data": [], "toks": [], "torch": False}, ask=lambda q: "", say=lambda *_: None)
        self.assertEqual(argv, ["corpus", "--lang", "en,tl,ceb,taglish", "--mode", "all"])

    def test_choices_and_mix_switch(self):
        answers = iter(["tl,ceb", "clean", "yes"])
        argv = M.collect(self.item(), {}, ask=lambda q: next(answers), say=lambda *_: None)
        self.assertEqual(argv, ["corpus", "--lang", "tl,ceb", "--mode", "clean", "--mix"])

    def test_bad_language_is_asked_again(self):
        said, answers = [], iter(["french", "en", "", ""])
        argv = M.collect(self.item(), {}, ask=lambda q: next(answers), say=said.append)
        self.assertEqual(argv[:3], ["corpus", "--lang", "en"])
        self.assertTrue(any("separated by commas" in x for x in said))

    def test_taglish_needs_a_typed_yes(self):
        said = []
        self.assertFalse(M.confirm_corpus(["corpus", "--lang", "taglish", "--mode", "clean"], ask=lambda q: "y", say=said.append))
        self.assertTrue(M.confirm_corpus(["corpus", "--lang", "taglish", "--mode", "clean"], ask=lambda q, it=iter(["yes", ""]): next(it), say=said.append))
        self.assertTrue(any("Private training only" in x for x in said))

    def test_no_taglish_just_asks_start(self):
        self.assertFalse(M.confirm_corpus(["corpus", "--lang", "ceb", "--mode", "clean"], ask=lambda q: "n", say=lambda *_: None))
        self.assertTrue(M.confirm_corpus(["corpus", "--lang", "ceb", "--mode", "clean"], ask=lambda q: "", say=lambda *_: None))

    def test_download_refused_when_datasets_missing(self):
        said = []
        orig = M.importlib.util.find_spec
        M.importlib.util.find_spec = lambda name, *a: None if name == "datasets" else orig(name, *a)
        try:
            self.assertFalse(M.confirm_corpus(["corpus", "--lang", "en", "--mode", "all"], ask=lambda q: "", say=said.append))
        finally:
            M.importlib.util.find_spec = orig
        self.assertTrue(any("pip install datasets" in x for x in said))

    def test_menu_runs_it_through_the_loop(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            idx = M.number_key(M.ITEMS.index(self.item()))
            answers = iter([idx, "ceb", "clean", "", "", "", "q"])      # pick, 3 questions, Start now?, press Enter, quit
            M.run_menu(d, stream=io.StringIO(), ask=lambda q: next(answers), call=calls.append, interactive=False)
            self.assertEqual(calls, [["corpus", "--lang", "ceb", "--mode", "clean"]])

    def test_corpus_command_accepts_what_the_menu_builds(self):
        from persona_lm import cli
        a = cli.build_parser().parse_args(["corpus", "--lang", "tl,ceb", "--mode", "clean", "--mix"])
        self.assertTrue(a.mix)


class Forms(unittest.TestCase):
    """The boxed questions and confirmations (used on a real terminal)."""

    def corpus(self):
        return [it for it in M.ITEMS if it["cmd"] == "corpus"][0]

    def form(self, item=None, found=None, width=100):
        return M.Form(item or self.corpus(), found or {"runs": [], "data": [], "toks": [], "torch": False}, FakeTTY(), color=False, width=width, unicode_ok=True)

    def drive(self, form, *ks):
        out = None
        for k in ks:
            out = form.key(k)
        return out

    def test_defaults_then_start_builds_the_normal_command(self):
        f = self.form()
        self.assertEqual(self.drive(f, "enter", "enter", "enter", "enter"), "done")
        self.assertEqual(M.build_argv(f.item, f.answers()), ["corpus", "--lang", "en,tl,ceb,taglish", "--mode", "all"])

    def test_typing_replaces_the_shown_answer_and_right_arrow_edits_it(self):
        f = self.form()
        self.drive(f, *"taglish")
        self.assertEqual(f.val[0], "taglish")
        g = self.form()
        self.drive(g, "right", ",", "x")
        self.assertEqual(g.val[0], "en,tl,ceb,taglish,x")
        self.drive(g, "backspace")
        self.assertEqual(g.val[0], "en,tl,ceb,taglish,")

    def test_backspace_on_an_untouched_answer_clears_it(self):
        f = self.form()
        self.drive(f, "backspace")
        self.assertEqual(f.val[0], "")

    def test_choices_cycle_and_a_letter_picks(self):
        f = self.form()
        self.drive(f, "enter", "right")
        self.assertEqual(f.val[1], "clean")
        self.drive(f, "right")
        self.assertEqual(f.val[1], "all")
        self.drive(f, "left")
        self.assertEqual(f.val[1], "clean")
        self.drive(f, "enter", "y")
        self.assertEqual(f.val[2], "yes")
        self.assertEqual(M.build_argv(f.item, f.answers())[-1], "--mix")

    def test_a_bad_answer_stays_put_and_says_why(self):
        f = self.form()
        self.drive(f, *"french")
        self.drive(f, "enter")
        self.assertEqual(f.focus, 0)
        self.assertIn("separated by commas", f.msg)
        self.assertIn("separated by commas", "\n".join(f.lines()))
        self.drive(f, "clear", *"ceb")
        self.drive(f, "enter")
        self.assertEqual(f.focus, 1)
        self.assertEqual(f.msg, "")

    def test_start_row_checks_everything_and_jumps_to_the_first_problem(self):
        f = self.form()
        self.drive(f, "clear", "down", "down", "down")
        self.assertEqual(f.focus, 3)
        self.assertIsNone(f.key("enter"))
        self.assertEqual(f.focus, 0)
        self.assertIn("is needed", f.msg)

    def test_esc_goes_back(self):
        self.assertEqual(self.drive(self.form(), "esc"), "cancel")

    def test_untouched_answers_follow_earlier_ones(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            f = self.form(M.ITEMS[0], M.scan(d))
            self.assertEqual(f.val[2], "s30m")                         # size of the found run
            self.drive(f, *"zzz")                                       # a run folder that does not exist
            self.assertEqual(f.val[2], "s10m")
            self.drive(f, "enter", "enter", "right")                    # touch the size: it now stays
            self.assertEqual(f.touched[2], True)

    def test_every_line_has_the_same_width_and_fits_the_window(self):
        for w in (80, 100, 140):
            f = self.form(width=w)
            for line in f.lines()[2:]:
                self.assertEqual(len(line), len(f.lines()[2]), w)
            self.assertLessEqual(len(f.lines()[2]), w - 1)

    def test_run_draws_and_cleans_up_after_itself(self):
        f = self.form()
        ks = iter(["enter", "enter", "enter", "enter"])
        self.assertTrue(f.run(lambda: next(ks)))
        out = f.out.getvalue()
        self.assertIn("Questions", out)
        self.assertTrue(out.rstrip().endswith("\x1b[?25h"))
        self.assertEqual(f.n, 0)

    def test_ctrl_c_in_the_form_goes_back(self):
        def boom():
            raise KeyboardInterrupt
        self.assertFalse(self.form().run(boom))

    def test_confirm_strict_needs_a_deliberate_y(self):
        c = lambda strict: M.Confirm("t", ["line"], strict, False, FakeTTY(), False, 100)
        self.assertFalse(c(True).run(keys("enter", "n")))
        self.assertTrue(c(True).run(keys("enter", "y")))
        self.assertTrue(c(False).run(keys("enter")))
        self.assertFalse(c(False).run(keys("esc")))

    def test_confirm_notice_returns_on_any_key(self):
        self.assertFalse(M.Confirm("t", ["x"], False, True, FakeTTY(), False, 100).run(keys("a")))

    def test_confirm_box_for_taglish_download_and_missing_datasets(self):
        argv = ["corpus", "--lang", "taglish", "--mode", "clean"]
        self.assertTrue(M.confirm_box(argv, {}, FakeTTY(), False, 100, keys("y")))
        self.assertFalse(M.confirm_box(argv, {}, FakeTTY(), False, 100, keys("enter", "n")))
        orig = M.importlib.util.find_spec
        M.importlib.util.find_spec = lambda name, *a: None if name == "datasets" else orig(name, *a)
        try:
            out = FakeTTY()
            self.assertFalse(M.confirm_box(["corpus", "--lang", "en", "--mode", "all"], {}, out, False, 100, keys("x")))
            self.assertIn("pip install datasets", out.getvalue())
        finally:
            M.importlib.util.find_spec = orig

    def test_train_confirm_box_says_new_or_continue(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(d)
            out = FakeTTY()
            self.assertTrue(M.confirm_box(["train", "--run", os.path.join("work3", "run_pre")], M.scan(d), out, False, 100, keys("enter")))
            self.assertIn("CONTINUE", out.getvalue())
            out2 = FakeTTY()
            M.confirm_box(["train", "--run", "brand_new"], M.scan(d), out2, False, 100, keys("n"))
            self.assertIn("NEW run", out2.getvalue())

    def test_whole_menu_with_boxed_questions(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            idx = M.number_key(M.ITEMS.index(self.corpus()))
            edit = keys("enter", "right", "enter", "enter", "enter", "y")           # languages ok, mode -> clean, mix no, Start, accept licence
            M.run_menu(d, stream=FakeTTY(), read=keys(idx, "q"), ask=lambda q: "", call=calls.append, width=100, color=False,
                       interactive=True, forms=True, read_edit=edit)
            self.assertEqual(calls, [["corpus", "--lang", "en,tl,ceb,taglish", "--mode", "clean"]])

    def test_whole_menu_backing_out_runs_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            idx = M.number_key(M.ITEMS.index(self.corpus()))
            M.run_menu(d, stream=FakeTTY(), read=keys(idx, "q"), ask=lambda q: "", call=calls.append, width=100, color=False,
                       interactive=True, forms=True, read_edit=keys("esc"))
            self.assertEqual(calls, [])

    def test_edit_keys_are_mapped(self):
        self.assertEqual([M._edit_plain(c) for c in ("\r", "\x1b", "\t", "\x7f", "\x08", "\x15", "A", "\\", "\x01")],
                         ["enter", "esc", "tab", "backspace", "backspace", "clear", "A", "\\", ""])
        with self.assertRaises(KeyboardInterrupt):
            M._edit_plain("\x03")


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
