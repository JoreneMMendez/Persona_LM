"""Tests for persona_lm. Run from the project folder:  python -m unittest discover -s tests -v"""
import json
import os
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import data as D
from persona_lm import gguf
from persona_lm.bpe import ASSISTANT, END, EOS, USER, Tokenizer
from persona_lm.generate import chat_prompt_ids, sample_ids
from persona_lm.model import Config, forward, init_params, loss_and_grads, zero_grads
from persona_lm.pretok import split_words
from persona_lm.train import train

CORPUS = ("the cat sat on the mat. the dog sat on the log. a cat and a dog are friends. "
          "she said it's fine, they'd go, we've seen it. numbers 12345 and symbols !? ... \n\n  indented line\n"
          "Ünïcödé テキスト 😀 emoji. ") * 40


def tiny_cfg(vocab):
    return Config(vocab_size=vocab, n_layer=2, d_model=32, n_head=4, d_ff=64, ctx=16)


class PreTok(unittest.TestCase):
    def test_known_splits(self):
        self.assertEqual(split_words("Hello, world!"), ["Hello", ",", " world", "!"])
        self.assertEqual(split_words("don't"), ["don", "'t"])
        self.assertEqual(split_words("a  b"), ["a", " ", " b"])
        self.assertEqual(split_words("x\n\ny"), ["x", "\n", "\n", "y"])
        self.assertEqual(split_words("abc123 def"), ["abc", "123", " def"])
        self.assertEqual(split_words("end  "), ["end", "  "])

    def test_always_reassembles(self):
        rnd = random.Random(1)
        alphabet = "ab Z9 \n\t'.,!?éテ😀_-" + "  "
        for _ in range(300):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
            self.assertEqual("".join(split_words(s)), s)


class Tok(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([CORPUS], 400)

    def test_round_trip(self):
        for s in ["", "hello", CORPUS[:300], "😀 テキスト ünï", "  \n\n tabs\tand  spaces ", "<|user|>hi<|end|>"]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_learns_merges_and_compresses(self):
        self.assertGreater(len(self.tok.merges), 20)
        self.assertLess(len(self.tok.encode("the cat sat on the mat.")), len("the cat sat on the mat."))

    def test_specials(self):
        self.assertEqual(self.tok.encode("<|end|>"), [END])
        ids = self.tok.encode("<|end|>", allow_special=False)
        self.assertNotIn(END, ids)
        self.assertEqual(self.tok.decode(ids), "<|end|>")

    def test_save_load_same(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.json")
            self.tok.save(p)
            t2 = Tokenizer.load(p)
            self.assertEqual(t2.encode(CORPUS[:500]), self.tok.encode(CORPUS[:500]))

    def test_invalid_utf8_split_is_safe(self):
        ids = self.tok.encode("😀")
        self.assertEqual(self.tok.decode(ids[:1]).count("�") >= 0, True)


class Gradients(unittest.TestCase):
    def test_finite_difference_matches_backprop(self):
        cfg = Config(vocab_size=23, n_layer=2, d_model=16, n_head=2, d_ff=24, ctx=7)
        params = init_params(cfg, 3, dtype=np.float64)
        rng = np.random.default_rng(5)
        for k, v in params.items():                       # make every parameter matter (norms are 1.0 by default)
            if v.ndim == 1:
                v += rng.normal(0, 0.1, v.shape)
            else:
                v *= 8                                     # larger weights -> clearer gradients
        x = rng.integers(0, 23, (2, 7))
        y = rng.integers(0, 23, (2, 7))
        mask = (rng.random((2, 7)) > 0.3).astype(np.float64)
        _, grads = loss_and_grads(params, cfg, x, y, mask)
        worst = 0.0
        eps = 1e-6
        for name, arr in params.items():
            flat = arr.reshape(-1)
            idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)
            for i in idx:
                old = flat[i]
                flat[i] = old + eps
                lp, _ = loss_and_grads(params, cfg, x, y, mask)
                flat[i] = old - eps
                lm, _ = loss_and_grads(params, cfg, x, y, mask)
                flat[i] = old
                num = (lp - lm) / (2 * eps)
                ana = grads[name].reshape(-1)[i]
                err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
                worst = max(worst, err if abs(num) + abs(ana) > 1e-7 else 0.0)
                self.assertLess(err if abs(num) + abs(ana) > 1e-7 else 0.0, 1e-4, "%s[%d] numeric %.3e analytic %.3e" % (name, i, num, ana))

    def test_gradient_accumulation_scale(self):
        cfg = tiny_cfg(30)
        p = init_params(cfg, 1)
        x = np.random.default_rng(0).integers(0, 30, (2, 8))
        y = np.roll(x, -1, 1)
        _, g1 = loss_and_grads(p, cfg, x, y)
        g2 = zero_grads(p)
        loss_and_grads(p, cfg, x, y, grads=g2, scale=0.5)
        loss_and_grads(p, cfg, x, y, grads=g2, scale=0.5)
        for k in g1:
            np.testing.assert_allclose(g1[k], g2[k], rtol=1e-4, atol=1e-6)


class ModelBehaviour(unittest.TestCase):
    def test_causal(self):
        cfg = tiny_cfg(30)
        p = init_params(cfg, 2)
        a = np.array([[1, 2, 3, 4, 5, 6]])
        b = a.copy()
        b[0, 5] = 9
        la, _ = forward(p, cfg, a)
        lb, _ = forward(p, cfg, b)
        np.testing.assert_allclose(la[:, :5], lb[:, :5], atol=1e-6)
        self.assertGreater(np.abs(la[:, 5] - lb[:, 5]).max(), 0)

    def test_last_only_matches_full(self):
        cfg = tiny_cfg(30)
        p = init_params(cfg, 2)
        a = np.array([[1, 2, 3, 4]])
        full, _ = forward(p, cfg, a)
        last, _ = forward(p, cfg, a, last_only=True)
        np.testing.assert_allclose(full[:, -1], last[:, 0], atol=1e-6)

    def test_initial_loss_is_about_log_vocab(self):
        cfg = tiny_cfg(100)
        p = init_params(cfg, 0)
        x = np.random.default_rng(0).integers(0, 100, (4, 16))
        loss, _ = loss_and_grads(p, cfg, x, np.roll(x, -1, 1))
        self.assertAlmostEqual(loss, np.log(100), delta=0.3)


def make_data(tmp, tok, kind="pretrain"):
    src = os.path.join(tmp, "src_" + kind)
    os.makedirs(src, exist_ok=True)
    if kind == "pretrain":
        with open(os.path.join(src, "a.txt"), "w", encoding="utf-8") as f:
            f.write(CORPUS)
        docs = D.pretrain_docs([src], tok)
    else:
        recs = [{"messages": [{"role": "system", "content": "You are kind."}, {"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello friend"}]},
                {"prompt": "what is the cat doing", "reply": "the cat sat on the mat"},
                {"instruction": "say it", "input": "the dog", "output": "the dog sat on the log"}] * 30
        with open(os.path.join(src, "c.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        docs = D.chat_docs([src], tok)
    out = os.path.join(tmp, "data_" + kind)
    D.prepare(tok, docs, out, with_mask=(kind == "chat"), log=lambda *_: None)
    return out


class DataAndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([CORPUS], 300)

    def test_prepare_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = make_data(tmp, self.tok)
            meta = json.load(open(os.path.join(out, "meta.json")))
            self.assertGreater(meta["train_tokens"], 100)
            self.assertGreater(meta["val_tokens"], 0)
            toks = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
            self.assertTrue((toks < self.tok.vocab_size).all())

    def test_chat_mask_only_on_assistant(self):
        ids, mask = D.encode_chat(self.tok, [("user", "hi there"), ("assistant", "hello friend")])
        text_masked = self.tok.decode([int(i) for i, m in zip(ids, mask) if m])
        self.assertEqual(text_masked, "hello friend<|end|>")
        self.assertEqual(int(ids[0]), USER)
        self.assertEqual(int(ids[-1]), EOS)

    def test_loss_goes_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = make_data(tmp, self.tok)
            cfg = tiny_cfg(self.tok.vocab_size)
            logs = []
            train(os.path.join(tmp, "run"), out, cfg, dict(batch=4, accum=1, steps=80, lr=3e-3, min_lr=3e-4, warmup=5, ckpt_every=40, log_every=10, eval_every=40, eval_batches=2), log=logs.append)
            lines = [json.loads(l) for l in open(os.path.join(tmp, "run", "metrics.jsonl"))]
            self.assertLess(lines[-1]["loss"], lines[0]["loss"] - 1.0)

    def test_resume_gives_identical_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = make_data(tmp, self.tok)
            cfg = tiny_cfg(self.tok.vocab_size)
            hp = dict(batch=2, accum=2, steps=12, lr=2e-3, min_lr=2e-4, warmup=2, ckpt_every=1000, log_every=100, eval_every=1000, eval_batches=1, ckpt_minutes=1000)
            straight = train(os.path.join(tmp, "a"), out, cfg, hp, log=lambda *_: None)
            run_b = os.path.join(tmp, "b")
            train(run_b, out, cfg, hp, log=lambda *_: None, max_steps_this_call=5)
            resumed = train(run_b, out, None, None, log=lambda *_: None)
            for k in straight:
                np.testing.assert_array_equal(straight[k], resumed[k], err_msg=k)

    def test_resume_survives_damaged_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = make_data(tmp, self.tok)
            cfg = tiny_cfg(self.tok.vocab_size)
            hp = dict(batch=2, accum=1, steps=6, ckpt_every=2, log_every=100, eval_every=1000, eval_batches=1)
            run = os.path.join(tmp, "r")
            train(run, out, cfg, hp, log=lambda *_: None, max_steps_this_call=4)
            with open(os.path.join(run, "ckpt.npz"), "wb") as f:
                f.write(b"garbage")
            logs = []
            train(run, out, None, None, log=logs.append)
            self.assertTrue(any("resumed at step" in l for l in logs))

    def test_chat_tuning_from_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre = make_data(tmp, self.tok, "pretrain")
            chat = make_data(tmp, self.tok, "chat")
            cfg = tiny_cfg(self.tok.vocab_size)
            hp = dict(batch=2, accum=1, steps=4, ckpt_every=100, log_every=100, eval_every=1000, eval_batches=1)
            train(os.path.join(tmp, "pre"), pre, cfg, hp, log=lambda *_: None)
            train(os.path.join(tmp, "sft"), chat, cfg, hp, init_from=os.path.join(tmp, "pre", "ckpt.npz"), log=lambda *_: None)
            self.assertTrue(os.path.exists(os.path.join(tmp, "sft", "ckpt.npz")))

    def test_generation_is_deterministic_and_stops(self):
        cfg = tiny_cfg(self.tok.vocab_size)
        p = init_params(cfg, 0)
        ids = chat_prompt_ids(self.tok, [("user", "hi")])
        self.assertEqual(ids[-1], ASSISTANT)
        a = sample_ids(p, cfg, ids, max_new=8, seed=3)
        b = sample_ids(p, cfg, ids, max_new=8, seed=3)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 8)
        g = sample_ids(p, cfg, ids, max_new=5, temperature=0)
        self.assertEqual(g, sample_ids(p, cfg, ids, max_new=5, temperature=0))


class Gguf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([CORPUS], 300)
        cls.cfg = Config(vocab_size=cls.tok.vocab_size, n_layer=2, d_model=64, n_head=4, d_ff=96, ctx=32)
        cls.params = init_params(cls.cfg, 7)
        cls.x = np.array([[1, 5, 9, 20, 7, 3, 2, 30]])

    def _round(self, kind):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.gguf")
            gguf.export_model(p, self.params, self.cfg, self.tok, kind=kind)
            meta, tensors = gguf.read_gguf(p)
            size = os.path.getsize(p)
        return meta, tensors, size

    def test_f32_is_exact(self):
        meta, t, _ = self._round("f32")
        p2 = gguf.params_from_gguf(t, self.cfg)
        for k in self.params:
            np.testing.assert_array_equal(self.params[k], p2[k], err_msg=k)

    def test_f16_and_q8_are_close(self):
        l0, _ = forward(self.params, self.cfg, self.x)
        for kind, tol in (("f16", 0.01), ("q8_0", 0.05)):
            _, t, _ = self._round(kind)
            l1, _ = forward(gguf.params_from_gguf(t, self.cfg), self.cfg, self.x)
            self.assertLess(np.abs(l0 - l1).max(), tol, kind)

    def test_sizes_shrink(self):
        s32 = self._round("f32")[2]
        s16 = self._round("f16")[2]
        s8 = self._round("q8_0")[2]
        self.assertLess(s16, s32 * 0.6)
        self.assertLess(s8, s16)

    def test_metadata_and_layout(self):
        meta, t, _ = self._round("f16")
        self.assertEqual(meta["_version"], 3)
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(meta["llama.embedding_length"], 64)
        self.assertEqual(meta["llama.attention.head_count"], 4)
        self.assertEqual(meta["llama.rope.dimension_count"], 16)
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), self.tok.vocab_size)
        self.assertEqual(len(meta["tokenizer.ggml.token_type"]), self.tok.vocab_size)
        self.assertEqual(len(meta["tokenizer.ggml.merges"]), len(self.tok.merges))
        self.assertEqual(meta["tokenizer.ggml.token_type"][EOS], 3)
        self.assertEqual(t["token_embd.weight"].shape, (self.tok.vocab_size, 64))     # numpy order; GGUF stores [64, vocab]
        self.assertEqual(t["blk.0.ffn_down.weight"].shape, (64, 96))
        self.assertIn("output.weight", t)

    def test_dims_are_stored_fastest_first(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.gguf")
            gguf.write_gguf(p, [("general.architecture", "str", "x")], [("w", np.zeros((3, 5), np.float32), 0)])
            raw = open(p, "rb").read()
        i = raw.index(b"\x01\x00\x00\x00\x00\x00\x00\x00w") + 9      # after the 1-char name string
        n_dims = struct.unpack_from("<I", raw, i)[0]
        dims = struct.unpack_from("<QQ", raw, i + 4)
        self.assertEqual((n_dims, dims), (2, (5, 3)))

    def test_q8_needs_multiple_of_32(self):
        with self.assertRaises(ValueError):
            gguf.tensor_bytes(np.zeros((2, 40), np.float32), gguf.T_Q8_0)

    def test_data_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.gguf")
            gguf.export_model(p, self.params, self.cfg, self.tok, kind="f16")
            self.assertEqual(os.path.getsize(p) % 32, 0)


if __name__ == "__main__":
    unittest.main()
