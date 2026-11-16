"""Stage 5: safety tools (backup before training, best checkpoint, early stop, doctor)."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import cli, doctor, safety  # noqa: E402
from persona_lm.bpe import Tokenizer  # noqa: E402
from persona_lm.ckpt import load_ckpt  # noqa: E402
from persona_lm.train import DEFAULT_HP, train  # noqa: E402
from tests.test_persona_lm import CORPUS, make_data, tiny_cfg  # noqa: E402

HP = dict(batch=4, accum=1, steps=60, lr=3e-3, min_lr=3e-4, warmup=5, ckpt_every=20, log_every=10, eval_every=10, eval_batches=2)


def H(*vals):
    """A history with one validation check per value."""
    return [{"step": (i + 1) * 10, "loss": 1.0, "val_loss": v} for i, v in enumerate(vals)]


class History(unittest.TestCase):
    def test_best_and_counts(self):
        h = H(3.0, 2.0, 2.5, 2.6)
        self.assertEqual(safety.best_so_far(h), (20, 2.0))
        self.assertEqual(safety.evals_without_improvement(h), 2)
        self.assertIsNone(safety.best_so_far([{"step": 1, "loss": 1.0}]))

    def test_is_new_best_only_for_strict_improvement(self):
        self.assertTrue(safety.is_new_best(H(3.0)))
        self.assertTrue(safety.is_new_best(H(3.0, 2.9)))
        self.assertFalse(safety.is_new_best(H(3.0, 3.0)))
        self.assertFalse(safety.is_new_best(H(3.0, 3.5)))
        self.assertFalse(safety.is_new_best(H(3.0) + [{"step": 11, "loss": 1.0}]))      # last record has no validation loss

    def test_should_early_stop(self):
        self.assertFalse(safety.should_early_stop(H(3.0, 2.0, 2.5, 2.6), 0))            # off
        self.assertFalse(safety.should_early_stop(H(3.0, 2.0, 2.5), 3))
        self.assertTrue(safety.should_early_stop(H(3.0, 2.0, 2.5, 2.6, 2.7), 3))
        self.assertFalse(safety.should_early_stop(H(3.0, 2.0, 2.5, 2.6, 1.9), 3))       # a new best resets the count


class Backup(unittest.TestCase):
    def test_backup_copies_and_never_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            tokp = os.path.join(tmp, "tok.json")
            with open(tokp, "w") as f:
                f.write("{}")
            self.assertIsNone(safety.backup_before_training(run, None, log=lambda *_: None))       # nothing to back up
            with open(os.path.join(run, "ckpt.npz"), "wb") as f:
                f.write(b"abc")
            out = safety.backup_before_training(run, tokp, log=lambda *_: None, now=1000)
            self.assertTrue(os.path.isfile(os.path.join(out, "ckpt.npz")))
            self.assertTrue(os.path.isfile(os.path.join(out, "tok.json")))
            self.assertIsNone(safety.backup_before_training(run, tokp, log=lambda *_: None, now=2000))   # identical: no second copy
            with open(os.path.join(run, "ckpt.npz"), "wb") as f:
                f.write(b"abcd")
            out2 = safety.backup_before_training(run, tokp, log=lambda *_: None, now=1000)               # same second: new folder, old one kept
            self.assertNotEqual(out, out2)
            self.assertEqual(open(os.path.join(out, "ckpt.npz"), "rb").read(), b"abc")
            self.assertEqual(open(os.path.join(tmp, "tok.json")).read(), "{}")                          # original untouched

    def test_backup_problem_never_stops_training(self):
        logs = []
        run = tempfile.mkdtemp()
        open(os.path.join(run, "ckpt.npz"), "wb").write(b"x")
        orig = safety.shutil.copy2
        safety.shutil.copy2 = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            self.assertIsNone(safety.backup_before_training(run, None, log=logs.append))
        finally:
            safety.shutil.copy2 = orig
        self.assertTrue(any("backup failed" in l for l in logs))


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([CORPUS], 300)

    def test_defaults_are_off_or_harmless(self):
        self.assertEqual(DEFAULT_HP["early_stop"], 0)
        self.assertEqual(DEFAULT_HP["save_best"], 1)

    def test_best_checkpoint_written_and_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(tmp, self.tok)
            run = os.path.join(tmp, "run")
            logs = []
            train(run, data, tiny_cfg(self.tok.vocab_size), dict(HP), log=logs.append)
            best = os.path.join(run, "ckpt.best.npz")
            self.assertTrue(os.path.isfile(best))
            self.assertFalse(os.path.exists(os.path.join(run, "ckpt.best.prev.npz")))
            params, opt, meta = load_ckpt(best)
            self.assertEqual(opt, {})                                       # weights only
            self.assertTrue(all(np.isfinite(v).all() for v in params.values()))
            vals = [r["val_loss"] for r in meta["history"] if "val_loss" in r]
            self.assertEqual(meta["history"][-1]["val_loss"], min(vals))    # saved exactly at the lowest validation loss
            self.assertTrue(any("new best" in l for l in logs))

    def test_save_best_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(tmp, self.tok)
            run = os.path.join(tmp, "run")
            train(run, data, tiny_cfg(self.tok.vocab_size), dict(HP, save_best=0, steps=20), log=lambda *_: None)
            self.assertFalse(os.path.exists(os.path.join(run, "ckpt.best.npz")))

    def test_early_stop_stops_and_keeps_normal_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(tmp, self.tok)
            run = os.path.join(tmp, "run")
            logs = []
            # lr so high that validation loss will not keep improving for long
            train(run, data, tiny_cfg(self.tok.vocab_size), dict(HP, steps=2000, lr=0.5, min_lr=0.5, warmup=1, early_stop=2, clip=0.0), log=logs.append)
            _, _, meta = load_ckpt(os.path.join(run, "ckpt.npz"))
            self.assertLess(meta["step"], 2000)
            self.assertTrue(any("early stop" in l for l in logs))
            self.assertTrue(safety.should_early_stop(meta["history"], 2))

    def test_resume_makes_backup_and_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(tmp, self.tok)
            cfg = tiny_cfg(self.tok.vocab_size)
            a = train(os.path.join(tmp, "a"), data, cfg, dict(HP), log=lambda *_: None)
            run = os.path.join(tmp, "b")
            train(run, data, cfg, dict(HP), log=lambda *_: None, max_steps_this_call=20)
            self.assertFalse(os.path.isdir(os.path.join(run, "backups")))             # a new run has nothing to back up
            b = train(run, data, None, None, log=lambda *_: None)
            self.assertEqual(len(os.listdir(os.path.join(run, "backups"))), 1)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])                              # backup and best-saving do not change the training

    def test_no_backup_flag_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(tmp, self.tok)
            cfg = tiny_cfg(self.tok.vocab_size)
            run = os.path.join(tmp, "r")
            train(run, data, cfg, dict(HP, steps=20), log=lambda *_: None, max_steps_this_call=10)
            train(run, data, None, None, log=lambda *_: None, backup=False)
            self.assertFalse(os.path.isdir(os.path.join(run, "backups")))

    def test_cli_has_new_flags(self):
        a = cli.build_parser().parse_args(["train", "--run", "r", "--data-dir", "d", "--early-stop", "3", "--no-backup", "--tokenizer", "t.json"])
        self.assertEqual((a.early_stop, a.no_backup, a.tokenizer, a.save_best), (3, True, "t.json", None))


class Doctor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([CORPUS], 300)

    def project(self, tmp):
        os.makedirs(os.path.join(tmp, "work"))
        self.tok.save(os.path.join(tmp, "work", "tok.json"))
        data = make_data(os.path.join(tmp, "work"), self.tok)
        os.makedirs(os.path.join(tmp, "work", "run_chat"), exist_ok=True)
        train(os.path.join(tmp, "work", "run_chat"), data, tiny_cfg(self.tok.vocab_size), dict(HP, steps=30), log=lambda *_: None)
        return data

    def run_doctor(self, tmp, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = doctor.run(tmp, say=print, **kw)
        return code, buf.getvalue()

    def test_healthy_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.project(tmp)
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "work", "tok.json"), deep=True)
            self.assertEqual(code, 0, out)
            self.assertIn("0 problem(s)", out)
            self.assertIn("all weights are finite", out)

    def test_wrong_tokenizer_and_changed_data_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.project(tmp)
            other = Tokenizer.train([CORPUS + " zzzz qqqq xxxx " * 50], 320)
            other.save(os.path.join(tmp, "other.json"))
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "other.json"))
            self.assertEqual(code, 1)
            self.assertIn("DIFFERENT tokenizer", out)
            with open(os.path.join(data, "train.bin"), "ab") as f:
                f.write(b"\0\0")
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "work", "tok.json"))
            self.assertIn("meta.json says", out)

    def test_missing_and_bad_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "nope.json"))
            self.assertEqual(code, 1)
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                f.write("{}")
            code, out = self.run_doctor(tmp, tokenizer=bad)
            self.assertEqual(code, 1)
            self.assertIn("not a readable tokenizer", out)

    def test_overfit_warning_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.project(tmp)
            run = os.path.join(tmp, "work", "run_chat")
            from persona_lm.ckpt import save_ckpt
            params, opt, meta = load_ckpt(os.path.join(run, "ckpt.npz"))
            meta["history"] = H(1.0, 0.9, 1.3)
            save_ckpt(os.path.join(run, "ckpt.npz"), params, opt, meta)
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "work", "tok.json"))
            self.assertIn("overfitting", out)
            self.assertIn("use ckpt.best.npz", out)
            os.remove(os.path.join(run, "ckpt.best.npz"))                      # (a copy in a temp folder) without a best file the advice changes
            before = sorted((n, os.path.getsize(os.path.join(run, n))) for n in os.listdir(run))
            code, out = self.run_doctor(tmp, tokenizer=os.path.join(tmp, "work", "tok.json"))
            self.assertIn("--early-stop 3", out)
            self.assertEqual(before, sorted((n, os.path.getsize(os.path.join(run, n))) for n in os.listdir(run)))       # nothing changed


if __name__ == "__main__":
    unittest.main()
