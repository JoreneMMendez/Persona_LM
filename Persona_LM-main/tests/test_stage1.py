"""Stage 1 tests (GPU backend groundwork). Everything here runs without PyTorch.
The PyTorch checks live in test_torch_backend.py and are skipped when PyTorch is missing."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import backend
from persona_lm import bpe
from persona_lm.bpe import Tokenizer
from persona_lm.ckpt import FORMAT_VERSION, load_ckpt, save_ckpt
from persona_lm.model import Config, init_params, loss_and_grads, preset, zero_grads
from persona_lm.train import step_denominators, train

CORPUS = ("the cat sat on the mat. the dog sat on the log. a cat and a dog are friends. " * 40)


def tiny_cfg(vocab):
    return Config(vocab_size=vocab, n_layer=2, d_model=32, n_head=4, d_ff=64, ctx=16)


def rel_diff(a, b):
    num = np.sqrt(sum(((a[k] - b[k]) ** 2).sum() for k in a))
    return float(num / np.sqrt(sum((b[k] ** 2).sum() for k in b)))


def write_token_data(root, vocab, tok_hash=None, masked=False, n=4000):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(0, vocab, n).astype(np.uint16).tofile(os.path.join(root, "train.bin"))
    rng.integers(0, vocab, 600).astype(np.uint16).tofile(os.path.join(root, "val.bin"))
    if masked:
        (rng.random(n) < 0.3).astype(np.uint8).tofile(os.path.join(root, "train_mask.bin"))
        (rng.random(600) < 0.3).astype(np.uint8).tofile(os.path.join(root, "val_mask.bin"))
    meta = {"vocab_size": vocab, "kind": "chat" if masked else "pretrain"}
    if tok_hash:
        meta["tokenizer_sha256"] = tok_hash
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return root


class AccumulationFix(unittest.TestCase):
    """The bug found in the Milestone 0 audit: with a loss mask, splitting a batch into micro-batches changed the gradients."""

    def setUp(self):
        self.cfg = tiny_cfg(64)
        self.p = init_params(self.cfg, 0, dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(1))
        self.x = rng.integers(0, 64, (8, 16))
        self.y = rng.integers(0, 64, (8, 16))
        self.m = np.zeros((8, 16), np.uint8)
        for i, k in enumerate([1, 1, 2, 2, 3, 12, 14, 16]):       # very uneven: like chat data
            self.m[i, -k:] = 1

    def _grads(self, split, loss_norm):
        slices = [slice(i, i + split) for i in range(0, 8, split)]
        batches = [(self.x[s], self.y[s], self.m[s]) for s in slices]
        denoms = step_denominators(batches, loss_norm)
        g = zero_grads(self.p)
        loss = 0.0
        for (x, y, m), dn in zip(batches, denoms):
            l, _ = loss_and_grads(self.p, self.cfg, x, y, m, grads=g, denom=dn)
            loss += l
        return loss, g

    def test_masked_accumulation_equals_one_big_batch(self):
        l_full, g_full = self._grads(8, "batch")
        for split in (4, 2, 1):
            l, g = self._grads(split, "batch")
            self.assertAlmostEqual(l, l_full, places=10, msg="split %d" % split)
            self.assertLess(rel_diff(g, g_full), 1e-10, "split %d" % split)

    def test_legacy_mode_still_has_the_old_behaviour(self):
        # kept on purpose so that checkpoints from before the fix resume exactly; it IS different on masked data
        _, g_full = self._grads(8, "micro")
        _, g_split = self._grads(2, "micro")
        self.assertGreater(rel_diff(g_split, g_full), 0.01)

    def test_unmasked_data_is_unaffected(self):
        batches = [(self.x[:4], self.y[:4], None), (self.x[4:], self.y[4:], None)]
        self.assertEqual(step_denominators(batches, "batch"), step_denominators(batches, "micro"))

    def test_changing_microbatch_size_keeps_the_same_training(self):
        """The VRAM use case: 4x2 and 2x4 (same effective batch of 8) draw the same data and train the same."""
        with tempfile.TemporaryDirectory() as tmp:
            data = write_token_data(os.path.join(tmp, "d"), 64, masked=True)
            cfg = tiny_cfg(64)
            common = dict(steps=3, lr=1e-3, min_lr=1e-4, warmup=1, eval_every=1000, ckpt_every=1000, log_every=1000)
            a = train(os.path.join(tmp, "a"), data, cfg, dict(batch=4, accum=2, **common), log=lambda *_: None)
            b = train(os.path.join(tmp, "b"), data, cfg, dict(batch=2, accum=4, **common), log=lambda *_: None)
            self.assertLess(rel_diff(a, b), 1e-4)


class ResumeCompatibility(unittest.TestCase):
    def test_old_checkpoint_resumes_in_legacy_loss_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write_token_data(os.path.join(tmp, "d"), 64, masked=True)
            run = os.path.join(tmp, "run")
            hp = dict(batch=2, accum=2, steps=4, lr=1e-3, min_lr=1e-4, warmup=1, eval_every=1000, ckpt_every=1000, log_every=1000)
            train(run, data, tiny_cfg(64), hp, log=lambda *_: None, max_steps_this_call=2)
            params, opt, meta = load_ckpt(os.path.join(run, "ckpt.npz"))
            self.assertEqual(meta["format_version"], FORMAT_VERSION)
            self.assertEqual(meta["hp"]["loss_norm"], "batch")
            del meta["hp"]["loss_norm"]                       # what a pre-fix checkpoint looks like
            del meta["format_version"]
            save_ckpt(os.path.join(run, "ckpt.npz"), params, opt, meta)
            train(run, data, None, None, log=lambda *_: None)
            _, _, meta2 = load_ckpt(os.path.join(run, "ckpt.npz"))
            self.assertEqual(meta2["hp"]["loss_norm"], "micro")
            self.assertEqual(meta2["step"], 4)


class TokenizerSafety(unittest.TestCase):
    def test_fingerprint_is_stable_and_sensitive(self):
        t1, t2 = Tokenizer.train([CORPUS], 320), Tokenizer.train([CORPUS], 320)
        t3 = Tokenizer.train(["zebras quietly examine jumbled xylophone quizzes. " * 40], 320)      # different text -> different merges
        self.assertEqual(t1.fingerprint(), t2.fingerprint())
        self.assertNotEqual(t1.fingerprint(), t3.fingerprint())

    def test_training_on_data_from_another_tokenizer_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            d1 = write_token_data(os.path.join(tmp, "d1"), 64, tok_hash="aaa")
            d2 = write_token_data(os.path.join(tmp, "d2"), 64, tok_hash="bbb")
            hp = dict(batch=2, accum=1, steps=2, warmup=1, eval_every=1000, ckpt_every=1000, log_every=1000)
            train(run, d1, tiny_cfg(64), hp, log=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
                train(run, d2, None, None, log=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):          # --init-from path too
                train(os.path.join(tmp, "new"), d2, tiny_cfg(64), hp, init_from=os.path.join(run, "ckpt.npz"), log=lambda *_: None)

    def test_prepare_records_the_fingerprint(self):
        from persona_lm import data as D
        tok = Tokenizer.train([CORPUS], 300)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "c.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write(CORPUS)
            D.prepare(tok, D.pretrain_docs([src], tok), os.path.join(tmp, "o"), with_mask=False, log=lambda *_: None)
            with open(os.path.join(tmp, "o", "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["tokenizer_sha256"], tok.fingerprint())


class ReservedSpecials(unittest.TestCase):
    def test_block_layout(self):
        self.assertEqual(len(bpe.SPECIALS), bpe.N_SPECIAL_SLOTS)
        self.assertEqual(len(set(bpe.SPECIALS)), len(bpe.SPECIALS))
        # the original five keep their ids forever
        self.assertEqual(bpe.SPECIALS[:5], ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"])
        self.assertEqual((bpe.EOS, bpe.SYSTEM, bpe.USER, bpe.ASSISTANT, bpe.END), (0, 1, 2, 3, 4))
        self.assertEqual(bpe.SPECIALS[bpe.MEMORY], "<|memory|>")

    def test_byte_tokens_start_after_the_block_and_specials_never_move(self):
        tok = Tokenizer.train([CORPUS], 320)
        self.assertEqual(tok.n_special, 32)
        self.assertEqual(tok.tok_to_id[bpe.BYTE2UNI[0]], 32)
        self.assertEqual(tok.encode("<|memory|>"), [bpe.MEMORY])

    def test_old_five_special_tokenizer_files_still_load(self):
        tok = Tokenizer([], specials=bpe._CORE_SPECIALS)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "old.json")
            tok.save(p)
            self.assertEqual(Tokenizer.load(p).n_special, 5)

    def test_s10m_default_context_is_4096(self):
        self.assertEqual(preset("s10m", 8000).ctx, 4096)


class BackendChoice(unittest.TestCase):
    def test_auto_falls_back_to_numpy_without_torch(self):
        msgs = []
        with mock.patch.object(backend, "torch_installed", return_value=False):
            self.assertEqual(backend.choose_backend("auto", log=msgs.append), "numpy")
            self.assertTrue(msgs and "NumPy" in msgs[0])
            with self.assertRaises(SystemExit):
                backend.choose_backend("torch")

    def test_auto_prefers_torch_when_installed(self):
        with mock.patch.object(backend, "torch_installed", return_value=True):
            self.assertEqual(backend.choose_backend("auto"), "torch")
            self.assertEqual(backend.choose_backend("numpy"), "numpy")

    def test_torch_backend_module_imports_without_torch(self):
        import persona_lm.torch_backend as tb
        if tb.torch is None:
            with self.assertRaises(RuntimeError):
                tb.pick_device()


if __name__ == "__main__":
    unittest.main()
