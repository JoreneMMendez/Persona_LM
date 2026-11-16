import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import cli  # noqa: E402
from persona_lm import menu as M  # noqa: E402
from persona_lm.bpe import Tokenizer  # noqa: E402
from persona_lm.ckpt import save_ckpt  # noqa: E402
from persona_lm.model import PRESETS, init_params, Config  # noqa: E402


def make_tok(path):
    Tokenizer.train(["hello world hello world"] * 20, 300).save(path)
    return Tokenizer.load(path)


class Stage3(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = self.d.name

    def run_dir(self, name, tok_sha=None):
        p = os.path.join(self.root, name)
        os.makedirs(p)
        cfg = dict(PRESETS["tiny"])
        meta = {"cfg": cfg, "step": 1}
        if tok_sha:
            meta["tokenizer_sha256"] = tok_sha
        save_ckpt(os.path.join(p, "ckpt.npz"), init_params(Config.from_dict(cfg), 0), {}, meta)
        return os.path.join(p, "ckpt.npz")

    def test_is_base_run(self):
        self.assertTrue(cli.is_base_run(os.path.join("work3", "run_pre", "ckpt.npz")))
        self.assertFalse(cli.is_base_run(os.path.join("work3", "run_chat2", "ckpt.npz")))

    def test_menu_chat_default_is_newest_chat_run_not_base(self):
        base = self.run_dir("run_pre")
        chat = self.run_dir("run_chat2")
        os.utime(base, (9e9 / 5, 9e9 / 5))                      # the base run is the newest file
        f = M.scan(self.root)
        self.assertTrue(f["runs"][0]["path"].endswith("run_pre"))
        self.assertTrue(M._d_chat_ckpt({}, f).endswith(os.path.join("run_chat2", "ckpt.npz")))
        self.assertTrue(M._d_ckpt({}, f).endswith(os.path.join("run_pre", "ckpt.npz")))   # train/generate unchanged

    def test_menu_chat_falls_back_to_base_when_no_chat_run(self):
        self.run_dir("run_pre")
        f = M.scan(self.root)
        self.assertTrue(M._d_chat_ckpt({}, f).endswith(os.path.join("run_pre", "ckpt.npz")))

    def test_missing_files_give_friendly_exit(self):
        a = SimpleNamespace(ckpt=os.path.join(self.root, "nope", "ckpt.npz"), tokenizer=os.path.join(self.root, "tok.json"))
        with self.assertRaises(SystemExit) as c:
            cli._load_model(a)
        self.assertIn("checkpoint not found", str(c.exception))
        ck = self.run_dir("run_chat")
        with self.assertRaises(SystemExit) as c:
            cli._load_model(SimpleNamespace(ckpt=ck, tokenizer=os.path.join(self.root, "missing.json")))
        self.assertIn("tokenizer file not found", str(c.exception))

    def test_wrong_tokenizer_file_is_friendly(self):
        ck = self.run_dir("run_chat")
        bad = os.path.join(self.root, "tok.seal.json")
        with open(bad, "w") as f:
            json.dump({"hash": "x"}, f)
        with self.assertRaises(SystemExit) as c:
            cli._load_model(SimpleNamespace(ckpt=ck, tokenizer=bad))
        self.assertIn("not a Persona_LM tokenizer", str(c.exception))

    def test_fingerprint_mismatch_and_base_run_warn(self):
        import contextlib
        import io
        tok = make_tok(os.path.join(self.root, "tok.json"))
        ck = self.run_dir("run_pre", tok_sha="0" * 64)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cli._load_model(SimpleNamespace(ckpt=ck, tokenizer=os.path.join(self.root, "tok.json")), chat=True)
        self.assertIn("different tokenizer", err.getvalue())
        self.assertIn("BASE run", err.getvalue())
        ok = self.run_dir("run_chat", tok_sha=tok.fingerprint())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cli._load_model(SimpleNamespace(ckpt=ok, tokenizer=os.path.join(self.root, "tok.json")), chat=True)
        self.assertEqual(err.getvalue(), "")

    def test_main_turns_keyerror_into_friendly_message(self):
        os.environ.pop("PERSONA_DEBUG", None)

        def boom(a):
            raise KeyError("merges")
        orig = cli.build_parser
        cli.build_parser = lambda: SimpleNamespace(parse_args=lambda args: SimpleNamespace(fn=boom))
        try:
            with self.assertRaises(SystemExit) as c:
                cli.main(["x"])
            self.assertIn("error:", str(c.exception))
            os.environ["PERSONA_DEBUG"] = "1"
            with self.assertRaises(KeyError):
                cli.main(["x"])
        finally:
            os.environ.pop("PERSONA_DEBUG", None)
            cli.build_parser = orig


if __name__ == "__main__":
    unittest.main()
