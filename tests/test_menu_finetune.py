import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import menu as M  # noqa: E402
from persona_lm.cli import build_parser  # noqa: E402
from persona_lm.model import PRESETS  # noqa: E402


def _ckpt(folder, size="s30m", step=100, steps=200):
    os.makedirs(folder, exist_ok=True)
    meta = {"cfg": dict(PRESETS[size]), "hp": {"steps": steps}, "step": step}
    np.savez(os.path.join(folder, "ckpt.npz"), meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def _tokens(folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"vocab_size": 8000}, f)


def project(root, chat_runs=()):
    w = os.path.join(root, "work3")
    _ckpt(os.path.join(w, "run_pre"))
    for n in chat_runs:
        _ckpt(os.path.join(w, "run_chat%d" % n))
    _tokens(os.path.join(w, "data_pre"))
    _tokens(os.path.join(w, "data_chat4096"))
    with open(os.path.join(w, "tok.json"), "w") as f:
        f.write('{"merges": [], "specials": []}')


class InDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.old)


class FineTuneItem(InDir):
    def item(self):
        return [i for i in M.ITEMS if i["label"].startswith("Fine-tune")][0]

    def test_item_exists_and_the_first_two_items_did_not_move(self):
        self.assertLess(M.ITEMS.index(self.item()), M.QUIT)
        self.assertEqual(M.ITEMS[0]["label"], "Train or resume a run")
        self.assertEqual(M.ITEMS[1]["label"], "Chat with a model")
        self.assertIsNone(M.ITEMS[M.QUIT]["cmd"])

    def test_defaults_pick_the_base_run_the_chat_data_and_a_new_folder(self):
        project(".")
        f = M.scan(".")
        a = {}
        a["init"] = M._d_ft_init(a, f)
        self.assertEqual(a["init"], os.path.join("work3", "run_pre", "ckpt.npz"))
        self.assertEqual(M._d_ft_run(a, f), os.path.join("work3", "run_chat1"))
        self.assertEqual(M._d_ft_data(a, f), os.path.join("work3", "data_chat4096"))
        self.assertEqual(M._d_ft_size(a, f), "s30m")

    def test_next_run_number_is_one_more_than_the_highest_existing(self):
        project(".", chat_runs=(2, 3))
        f = M.scan(".")
        a = {"init": os.path.join("work3", "run_pre", "ckpt.npz")}
        self.assertEqual(M._d_ft_run(a, f), os.path.join("work3", "run_chat4"))
        # a chat run is never offered as the base model
        self.assertTrue(M._d_ft_init({}, f).endswith(os.path.join("run_pre", "ckpt.npz")))

    def test_enter_on_everything_builds_the_fine_tune_command_with_gentle_settings(self):
        project(".")
        argv = M.collect(self.item(), M.scan("."), ask=lambda q: "", say=lambda *_: None)
        self.assertEqual(argv[:1], ["train"])
        self.assertEqual(argv[argv.index("--init-from") + 1], os.path.join("work3", "run_pre", "ckpt.npz"))
        self.assertEqual(argv[argv.index("--run") + 1], os.path.join("work3", "run_chat1"))
        self.assertEqual(argv[argv.index("--data-dir") + 1], os.path.join("work3", "data_chat4096"))
        self.assertEqual(argv[argv.index("--size") + 1], "s30m")
        a = build_parser().parse_args(argv)                      # the real train command accepts it
        self.assertEqual((a.lr, a.min_lr, a.warmup, a.steps), (1e-4, 1e-5, 50, 3000))
        self.assertEqual(a.init_from, os.path.join("work3", "run_pre", "ckpt.npz"))

    def test_a_changed_setting_still_wins_over_the_preset(self):
        project(".")
        answers = iter(["", "", "", "", "2", "5e-5", ""])       # four questions, setting 2 (lr) = 5e-5, Enter
        argv = M.collect(self.item(), M.scan("."), ask=lambda q: next(answers), say=lambda *_: None)
        self.assertEqual(build_parser().parse_args(argv).lr, 5e-5)

    def test_the_boxed_form_gives_the_same_command(self):
        project(".")
        form = M.Form(self.item(), M.scan("."), __import__("io").StringIO(), False, 100)
        ans = form.answers()
        self.assertEqual(ans["init"], os.path.join("work3", "run_pre", "ckpt.npz"))
        self.assertEqual(ans["run"], os.path.join("work3", "run_chat1"))
        self.assertEqual(ans["size"], "s30m")

    def test_an_existing_run_folder_and_a_missing_base_are_refused(self):
        project(".", chat_runs=(1,))
        self.assertIn("CONTINUE", M._check_new_run(os.path.join("work3", "run_chat1")))
        self.assertIsNone(M._check_new_run(os.path.join("work3", "run_chat9")))
        self.assertIn("no such file", M._check_base_ckpt(os.path.join("work3", "nope", "ckpt.npz")))
        self.assertIsNone(M._check_base_ckpt(os.path.join("work3", "run_pre", "ckpt.npz")))

    def test_the_preset_applies_to_the_new_run_only_not_to_normal_train(self):
        project(".", chat_runs=(1,))
        f = M.scan(".")
        answers = iter([os.path.join("work3", "run_pre", "ckpt.npz"), os.path.join("work3", "run_chat9"), "", "", ""])
        argv = M.collect(self.item(), f, ask=lambda q: next(answers), say=lambda *_: None)
        self.assertIn("--lr", argv)                               # a NEW folder: the preset applies
        # the normal Train item never gets a preset
        argv2 = M.collect(M.ITEMS[0], f, ask=lambda q: "", say=lambda *_: None)
        self.assertNotIn("--lr", argv2)

    def test_the_confirmation_says_a_new_run(self):
        project(".")
        argv = M.collect(self.item(), M.scan("."), ask=lambda q: "", say=lambda *_: None)
        self.assertIn("NEW run", M._train_lines(argv, M.scan("."))[0])

    def test_every_field_has_its_own_help(self):
        for f in self.item()["fields"]:
            self.assertTrue(f["help"], f["key"])


if __name__ == "__main__":
    unittest.main()


class MenuScrolls(unittest.TestCase):
    FOUND = {"runs": [], "data": [], "toks": [], "torch": False}

    def render(self, sel):
        import io
        from persona_lm import pretty as P

        class T(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True
        lines = M.Menu(T(), color=False, width=100).render(sel, self.FOUND)
        self.assertEqual(len(lines), M.MENU_H)
        self.assertEqual(len({P.vlen(x) for x in lines}), 1)
        return "\n".join(lines)

    def test_every_item_can_be_reached_and_seen_when_selected(self):
        for i, it in enumerate(M.ITEMS):
            text = self.render(i)
            self.assertIn(M.plain_text(it)["title"], text, it["label"])       # the plain title (T2 renamed the rows)

    def test_quit_is_visible_on_the_last_item(self):
        self.assertIn("Quit", self.render(M.QUIT))
