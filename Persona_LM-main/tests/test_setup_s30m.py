"""Tests for setup_s30m.py (repo root): builds a new s30m work folder from the s10m one. Tiny tokenizer, no training, no network."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import setup_s30m as S
from persona_lm import pipeline as PL
from persona_lm.bpe import Tokenizer


def slurp(path, mode="r"):
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        return f.read()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.p = lambda *a: os.path.join(self.root, *a)
        PL.init_project(self.p("work"), languages=["en", "tl"], vocab=300, size="s10m", ctx=512, pretrain_text=["public_mix"],
                        chat_train=self.p("data_msgr", "train.jsonl"), system="You are Aria.")
        Tokenizer.train(["the cat sat on the mat " * 50], 300).save(self.p("work", "tok.json"))
        os.makedirs(self.p("work", "data_pre"))
        with open(self.p("work", "data_pre", "meta.json"), "w") as f:
            json.dump({"x": 1}, f)
        os.makedirs(self.p("data_msgr_clean"))
        for n in ("train.jsonl", "val.jsonl"):
            with open(self.p("data_msgr_clean", n), "w") as f:
                f.write("")

    def run_setup(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = S.main(["--from-dir", self.p("work"), "--dir", self.p("work2"), "--chat-dir", self.p("data_msgr_clean"), *extra])
        return code, buf.getvalue()


class SetupTests(Base):
    def test_makes_an_s30m_project_that_reuses_the_old_settings_and_points_at_the_merged_set(self):
        code, out = self.run_setup()
        self.assertEqual(code, 0)
        proj = PL.load_project(self.p("work2"))
        self.assertEqual((proj["size"], proj["ctx"], proj["vocab"], proj["languages"]), ("s30m", 512, 300, ["en", "tl"]))
        self.assertEqual(proj["chat_train"], self.p("data_msgr_clean", "train.jsonl"))
        self.assertEqual(proj["chat_val"], self.p("data_msgr_clean", "val.jsonl"))
        self.assertEqual(proj["pretrain_steps"], 18000)
        self.assertEqual((proj["batch"], proj["accum"], proj["system"]), (4, 8, "You are Aria."))

    def test_the_frozen_tokenizer_and_public_tokens_are_copied_and_the_fingerprint_matches(self):
        self.run_setup()
        self.assertEqual(slurp(self.p("work", "tok.json"), "rb"), slurp(self.p("work2", "tok.json"), "rb"))
        self.assertTrue(os.path.exists(self.p("work2", "data_pre", "meta.json")))
        proj = PL.load_project(self.p("work2"))
        self.assertEqual(proj["tokenizer_sha256"], Tokenizer.load(self.p("work2", "tok.json")).fingerprint())

    def test_it_does_not_copy_old_chat_tokens_or_s10m_checkpoints(self):
        os.makedirs(self.p("work", "data_chat")); os.makedirs(self.p("work", "run_pre"))
        for d, f in (("data_chat", "meta.json"), ("run_pre", "ckpt.npz")):
            with open(self.p("work", d, f), "w") as fh:
                fh.write("x")
        self.run_setup()
        self.assertFalse(os.path.exists(self.p("work2", "data_chat")))
        self.assertFalse(os.path.exists(self.p("work2", "run_pre")))

    def test_the_pipeline_plan_works_and_pretraining_is_not_skipped(self):
        self.run_setup()
        steps = PL.build_steps(PL.load_project(self.p("work2")), self.p("work2"))
        by = {s["name"]: s for s in steps}
        self.assertTrue(by["tokenizer"]["done"])
        self.assertTrue(by["pretrain-data"]["done"])
        self.assertFalse(by["pretrain"]["done"])
        self.assertFalse(by["pretrain"]["skipped"])
        self.assertIn("s30m", by["pretrain"]["argv"])
        self.assertIn("--steps", by["pretrain"]["argv"])
        self.assertEqual(by["pretrain"]["argv"][by["pretrain"]["argv"].index("--steps") + 1], "18000")
        self.assertEqual(PL.next_step(steps)["name"], "pretrain")

    def test_the_old_s10m_folder_is_untouched(self):
        before = slurp(self.p("work", "project.json"))
        self.run_setup()
        self.assertEqual(slurp(self.p("work", "project.json")), before)
        self.assertEqual(PL.load_project(self.p("work"))["size"], "s10m")

    def test_dry_run_changes_nothing(self):
        code, out = self.run_setup("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("nothing was changed", out.lower())
        self.assertFalse(os.path.exists(self.p("work2")))

    def test_running_it_twice_is_harmless(self):
        self.run_setup()
        code, out = self.run_setup()
        self.assertEqual(code, 0)
        self.assertIn("already there", out)

    def test_missing_merged_set_or_tokenizer_is_a_clear_message(self):
        os.remove(self.p("data_msgr_clean", "val.jsonl"))
        with contextlib.redirect_stdout(io.StringIO()) as buf, self.assertRaises(SystemExit):
            S.main(["--from-dir", self.p("work"), "--dir", self.p("work2"), "--chat-dir", self.p("data_msgr_clean")])
        self.assertIn("clean_chat.py", buf.getvalue())
        self.assertFalse(os.path.exists(self.p("work2")))

    def test_same_folder_is_refused(self):
        with self.assertRaises(SystemExit):
            S.main(["--from-dir", self.p("work"), "--dir", self.p("work")])


if __name__ == "__main__":
    unittest.main()
